#!/usr/bin/env python3
"""LLaVA-1.5 (Vicuna-7B) MileBench inference under VFlowOpt-style eviction.

Counterpart of run_onevision_milebench_vflowopt.py for LLaVA-1.5. Targets the
same datasets used in the OneVision sweep (T-2/T-3/T-4 multi-choice + S-3
ROUGE-L):
  ObjectExistence, ObjectInteraction, MovingAttribute, ObjectShuffle,
  EgocentricNavigation, MovingDirection,
  CounterfactualInference, StateChange, CharacterOrder, SceneTransition,
  CLEVR-Change, IEdit, Spot-the-Diff.

Key differences vs OneVision:
  * model class: llava_llama_training_free (LLaMA backbone, vicuna_v1 conv).
  * 576 fixed image tokens per image (CLIP-ViT-L/14-336, no anyres, no spatial pool).
  * No video modality — multi-image as a list of (1, 3, 336, 336) tensors.
  * 4096 max context — look-m's standard truncation budget.
  * `forward_illava` here uses LLM self-attention at layer (k-1) for token
    importance (no ViT attention map needed). The wrapper.generate() takes
    `keep_ratio_total` and computes per-stage `illava_llm_r` itself, so the
    script just passes the total-token-basis keep ratio in `illava_config`.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
LLAVA_ROOT = REPO_ROOT / "src" / "LLaVA-OneVision"
LOOKM_ROOT = Path("/workspace/look-m")
for p in (str(REPO_ROOT), str(LLAVA_ROOT), str(LOOKM_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
# Trigger AutoConfig registration of LlavaLlamaTrainingFreeConfig.
import llava.model.language_model.llava_llama_training_free  # noqa: F401

from utils import MileBenchDataset

LOOKM_IMAGE_PLACEHOLDER = "<ImageHere>"
DEFAULT_MODEL_PATH = "/workspace/zap/ckpts/llava-v1.5-7b"
DEFAULT_DATA_DIR = "/workspace/zap/data/MileBench"

DEFAULT_DATASETS = [
    # T-2 / T-3 / T-4 multi-choice
    "ObjectExistence", "ObjectInteraction", "MovingAttribute", "ObjectShuffle",
    "EgocentricNavigation", "MovingDirection",
    "CounterfactualInference", "StateChange", "CharacterOrder", "SceneTransition",
    # S-3 open-ended (ROUGE-L)
    "CLEVR-Change", "IEdit", "Spot-the-Diff",
]

# LLaMA-7B has 32 layers; layers 10/21 are the iLLaVA paper's progressive
# prune points for LLaMA (matching scripts/run_llava15_vflowopt_keep05_eval.sh).
ILLAVA_LLM_K = [10, 21]
N_TOKENS_PER_IMAGE = 576  # CLIP-ViT-L/14-336 → 24×24 patches.


def build_passthrough_illava_config(image_start_index: int = 0) -> dict[str, Any]:
    """LLaVA-1.5 forward_illava is gated only by enable_illava_llm. The wrapper
    .generate() in llava_llama_training_free.py reads keep_ratio_total from
    this dict and overwrites illava_llm_r per-sample."""
    return {
        "enable_illava_llm": False,
        "illava_llm_k": ILLAVA_LLM_K,
        "illava_llm_r": [1.0] * len(ILLAVA_LLM_K),
        "illava_llm_image_token_start_index": image_start_index,
        "keep_ratio_total": 1.0,
    }


def build_pruned_illava_config(keep_ratio: float, image_start_index: int = 0) -> dict[str, Any]:
    cfg = build_passthrough_illava_config(image_start_index)
    cfg["enable_illava_llm"] = True
    cfg["keep_ratio_total"] = float(keep_ratio)
    return cfg


def build_prompt(question: str, conv_template: str) -> tuple[str, str]:
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question.strip())
    conv.append_message(conv.roles[1], None)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return conv.get_prompt(), stop_str


def find_image_start_index(input_ids: torch.Tensor) -> int:
    ids = input_ids[0]
    positions = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
    return int(positions[0].item()) if positions.numel() > 0 else 0


def open_images(paths: list[str]) -> list[Image.Image]:
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(im.convert("RGB").copy())
    return out


def prepare_image_tensors(images: list[Image.Image], image_processor: Any, model_cfg,
                          device: torch.device, dtype: torch.dtype):
    """Process N images individually so prepare_inputs_labels_for_multimodal
    takes the multi-image list path. LLaVA-1.5 has no video modality and no
    anyres — every image becomes (1, 3, 336, 336) → 576 tokens after projection.
    """
    tensors = []
    for im in images:
        out = process_images([im], image_processor, model_cfg)
        if isinstance(out, list):
            t = out[0]
        else:
            t = out[0] if out.ndim == 4 else out
        if t.ndim == 3:
            t = t.unsqueeze(0)
        tensors.append(t.to(device=device, dtype=dtype))
    return tensors


@torch.no_grad()
def generate_answer(*, model, tokenizer, image_processor, sample, device, dtype,
                    conv_template: str, max_new_tokens: int, mode: str,
                    keep_ratio: float) -> str:
    images = open_images(sample["image_paths"])
    image_tensors = prepare_image_tensors(images, image_processor, model.config, device, dtype)
    image_sizes = [img.size for img in images]
    modalities = ["image"] * len(image_tensors)

    prompt_text, stop_str = build_prompt(sample["question"], conv_template)
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    image_start_index = find_image_start_index(input_ids)

    if mode == "fullcache":
        cfg = build_passthrough_illava_config(image_start_index)  # enable_illava_llm=False
    else:
        cfg = build_pruned_illava_config(keep_ratio, image_start_index)

    outputs = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        modalities=modalities,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0,
        use_cache=True,
        illava_config=cfg,
    )
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    if stop_str and text.endswith(stop_str):
        text = text[: -len(stop_str)].strip()
    return text


def split_data(data):
    by_n = {}
    for d in data:
        n_img = len(d["task_instance"]["images_path"])
        by_n.setdefault(n_img, []).append(d)
    return by_n


def evaluate_dataset(*, dataset_name: str, args, model, tokenizer, image_processor,
                     device, dtype) -> None:
    out_dir = Path(args.output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "pred.json"
    if pred_path.exists() and not args.overwrite:
        print(f"[skip] {dataset_name}: {pred_path} exists (use --overwrite to redo)")
        return

    annotation_path = Path(args.data_dir) / dataset_name / f"{dataset_name}.json"
    core = json.loads(annotation_path.read_text())
    img_dir = str(Path(args.data_dir) / dataset_name / "images")
    by_n = split_data(core["data"])

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    t_start = time.time()

    for n_img in sorted(by_n.keys()):
        sub = by_n[n_img]
        ds = MileBenchDataset(
            annotation=sub,
            task_instructions=core["meta_data"]["task_instruction"],
            img_dir=img_dir,
            max_context_len=args.max_context_len,
            n_tokens_per_image=N_TOKENS_PER_IMAGE,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            combine_image=None,
        )
        if args.eval_samples is not None:
            limit = max(0, args.eval_samples - len(results) - len(failures))
            if limit == 0:
                break
            indices = list(range(min(limit, len(ds))))
        else:
            indices = list(range(len(ds)))

        for i in tqdm(indices, desc=f"{dataset_name} n_img={n_img} {args.mode}"
                                   + (f"(keep={args.keep_ratio})" if args.mode == "pruned" else "")):
            item = ds[i]
            sid = item["sample_id"]
            question = item["context"].replace(LOOKM_IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)
            sample = {
                "id": sid,
                "question": question,
                "image_paths": item["raw_img_list"],
                "gt_response": item["response"],
            }
            try:
                pred = generate_answer(
                    model=model, tokenizer=tokenizer, image_processor=image_processor,
                    sample=sample, device=device, dtype=dtype,
                    conv_template=args.conv_template,
                    max_new_tokens=args.max_new_tokens,
                    mode=args.mode, keep_ratio=args.keep_ratio,
                )
            except Exception as exc:  # noqa: BLE001
                import traceback
                tb = traceback.format_exc()
                failures.append({"sample_id": sid, "error": repr(exc), "tb": tb})
                print(f"[FAIL] {dataset_name} sample {sid}: {exc}\n{tb}", flush=True)
                torch.cuda.empty_cache()
                continue
            finally:
                torch.cuda.empty_cache()

            results.append({
                "sample_id": sid,
                "question": item["context"],
                "gt_response": item["response"],
                "pred_response": pred,
            })

        if args.eval_samples is not None and len(results) + len(failures) >= args.eval_samples:
            break

    elapsed = time.time() - t_start
    pred_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    summary = {
        "dataset": dataset_name,
        "mode": args.mode,
        "keep_ratio": args.keep_ratio if args.mode == "pruned" else None,
        "n_samples": len(results),
        "n_failures": len(failures),
        "elapsed_seconds": elapsed,
        "sec_per_sample": elapsed / max(1, len(results) + len(failures)),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(summary, indent=2))
    if failures:
        (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    print(f"[DONE] {dataset_name} mode={args.mode} n={len(results)} fail={len(failures)} -> {pred_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["fullcache", "pruned"], default="fullcache")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-context-len", type=int, default=4096,
                        help="LLaVA-1.5 max context. look-m's MileBench truncation drops "
                             "earliest frames once 576*n_img + question exceeds this.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--model-name", default="llava-v1.5-7b-training_free",
                        help="Suffix triggers LlavaLlamaTrainingFreeForCausalLM in load_pretrained_model.")
    parser.add_argument("--conv-template", default="vicuna_v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dtype = getattr(torch, args.torch_dtype)
    device = torch.device(args.device)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, args.model_name,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    if args.device_map != "auto":
        model = model.to(device)
    else:
        device = next(model.parameters()).device
    if dtype != torch.float16:
        model = model.to(dtype=dtype)

    print(f"[mode] {args.mode} keep_ratio={args.keep_ratio if args.mode == 'pruned' else 'N/A'}")
    print(f"[n_tokens_per_image] {N_TOKENS_PER_IMAGE}  max_context_len={args.max_context_len}")
    print(f"[illava_llm_k] {ILLAVA_LLM_K} (LLaMA layers)")

    for dataset_name in args.datasets:
        evaluate_dataset(
            dataset_name=dataset_name, args=args,
            model=model, tokenizer=tokenizer, image_processor=image_processor,
            device=device, dtype=dtype,
        )


if __name__ == "__main__":
    main()
