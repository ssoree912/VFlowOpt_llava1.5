#!/usr/bin/env python3
"""LLaVA-OneVision MMVet / detail_1k inference under VFlowOpt-style eviction.

This is the pruned counterpart of `zap/scripts/eval_onevision_mmvet_detail.py`.
It loads the same OneVision-Qwen2-7B-OV checkpoint via the LLaVA-OneVision
training-free model class so that the project's iLLaVA / VFlowOpt token
pruning hooks (ViT entropy-augmented attention reduction + LLM-side
progressive pruning) are active during prefill and decode.

Modes:
  * --mode fullcache             : iLLaVA disabled. Writes fullcache_rouge_ref.json.
  * --mode pruned --keep-ratio R : ViT path is enabled but configured to be a
                                   passthrough (illava_vit_r=1.0, illava_vit_m
                                   large enough that no merging triggers) so that
                                   the LLM forward_illava can read the VIT-derived
                                   importance map. LLM-side progressive pruning
                                   then runs at illava_llm_k=[9,18], with per-stage
                                   keep ratios computed *per sample* so that the
                                   end-of-prefill total-token retention matches
                                   --keep-ratio (LLaVA-1.5 VFlowOpt convention:
                                   only image tokens are evicted, but keep_ratio
                                   is on the total prompt-token basis).

Protocol (matches eval_onevision_mmvet_detail.py):
  * PPL: teacher-force the dataset answer (GT).
  * ROUGE: score generations against the full-cache predictions if
    --rouge-ref-path is given, otherwise against the current-run predictions.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LLAVA_ONEVISION_ROOT = REPO_ROOT / "src" / "LLaVA-OneVision"
if LLAVA_ONEVISION_ROOT.exists() and str(LLAVA_ONEVISION_ROOT) not in sys.path:
    sys.path.insert(0, str(LLAVA_ONEVISION_ROOT))

from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


def patch_siglip_loader_to_local_init() -> None:
    """Avoid hard-coded external SigLIP path when initializing the vision tower."""
    from llava.model.multimodal_encoder import siglip_encoder

    def _load_model(self, device_map=None):  # noqa: ANN001, ARG001
        if self.is_loaded:
            return
        self.vision_tower = siglip_encoder.SigLipVisionModel(self.config)
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = siglip_encoder.nn.Identity()
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    siglip_encoder.SigLipVisionTower.load_model = _load_model


DEFAULT_MODEL_PATH = "/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov"
DATASET_DEFAULTS = {
    "mmvet": {
        "data_path": "/workspace/data/mm-vet/mm-vet.json",
        "image_path": "/workspace/data/mm-vet",
        "eval_samples": 218,
        "max_new_tokens": 128,
    },
    "detail_1k": {
        "data_path": "/workspace/data/detail_1k.json",
        "image_path": "/workspace/data",
        "eval_samples": 1000,
        "max_new_tokens": 512,
    },
}


# LLM-side progressive pruning layers — matches the README example for
# OneVision (illava_llm_k=9-18) and the LLaVA-1.5 wrapper's default span.
ILLAVA_LLM_K = [9, 18]


def build_passthrough_illava_config() -> dict[str, Any]:
    """ViT-side passthrough config that still populates VIT_ATTN_MAP.

    The Qwen2 forward_illava reads VIT_ATTN_MAP from os.environ to rank image
    tokens during LLM-side pruning. We need enable_illava_vit=True for that env
    var to be set, but vit_r=1.0 + vit_m large enough (so num_merged_tokens
    floors to 0 for typical anyres counts) keeps every image token alive
    through the ViT-side pre-projection step.
    """
    return {
        "enable_illava_llm": False,
        "enable_illava_vit": True,
        "illava_llm_k": ILLAVA_LLM_K,
        "illava_llm_r": [1.0] * len(ILLAVA_LLM_K),
        "illava_vit_k": [25],
        "illava_vit_t": 0.0,
        "illava_vit_alpha_v": 0.0,
        "illava_vit_m": 1000.0,  # a*a > image_token_length → no merging
        "illava_vit_r": 1.0,
        "illava_llm_image_token_start_index": 14,
        "illava_track_vit_source": False,
        "illava_track_llm_source": False,
    }


def build_fullcache_illava_config() -> dict[str, Any]:
    cfg = build_passthrough_illava_config()
    cfg["enable_illava_vit"] = False
    return cfg


def compute_pruned_illava_config(
    keep_ratio: float, image_token_length: int, total_len: int, image_start_index: int
) -> dict[str, Any]:
    """LLaVA-1.5 VFlowOpt convention.

    keep_ratio is on the TOTAL prompt-token basis. Only image tokens are
    evicted, so:
        n_keep_img = max(1, n_img - (1 - keep_ratio) * total_len)
    The cumulative LLM-side keep ratio over image tokens equals
        n_keep_img / n_img
    Split that geometrically across ``len(ILLAVA_LLM_K)`` stages so that the
    product of per-stage ratios reproduces the target.
    """
    base = build_passthrough_illava_config()
    if keep_ratio >= 1.0 or image_token_length <= 0:
        base["enable_illava_llm"] = False
        base["illava_llm_image_token_start_index"] = image_start_index
        return base
    n_img = int(image_token_length)
    n_keep_img = int(round(n_img - (1.0 - keep_ratio) * total_len))
    n_keep_img = max(1, min(n_img, n_keep_img))
    final_image_keep = n_keep_img / max(1, n_img)
    num_stages = len(ILLAVA_LLM_K)
    per_stage_ratio = final_image_keep ** (1.0 / num_stages)
    base["enable_illava_llm"] = True
    base["illava_llm_r"] = [per_stage_ratio] * num_stages
    base["illava_llm_image_token_start_index"] = image_start_index
    return base


def load_samples(data_path: str, image_path: str, eval_samples: int | None) -> list[dict[str, Any]]:
    with open(data_path) as f:
        raw = json.load(f)
    items = [{"id": k, **v} for k, v in raw.items()] if isinstance(raw, dict) else list(raw)
    samples: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if "conversations" in item:
            convs = item["conversations"]
            if len(convs) < 2:
                raise ValueError(f"Expected >=2 conversation turns in sample {idx}")
            question = convs[0]["value"]
            answer = convs[1]["value"]
            image_rel = item["image"]
        else:
            question = item["question"]
            answer = item["answer"]
            image_rel = item.get("image") or item.get("imagename")
            if image_rel and "mm-vet" in data_path and not image_rel.startswith("images/"):
                image_rel = f"images/{image_rel}"
        question = question.replace("<image>", "").replace("\n\n", "\n").strip()
        sample_id = item.get("id", item.get("sample_id", str(idx)))
        samples.append({
            "id": sample_id,
            "image": image_rel,
            "image_file": str(Path(image_path) / image_rel),
            "question": question,
            "answer": answer,
        })
    if eval_samples is not None:
        samples = samples[:eval_samples]
    return samples


def build_prompt(question: str, conv_template: str) -> tuple[str, str]:
    conv = copy.deepcopy(conv_templates[conv_template])
    content = f"{DEFAULT_IMAGE_TOKEN}\n{question.strip()}"
    conv.append_message(conv.roles[0], content)
    conv.append_message(conv.roles[1], None)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return conv.get_prompt(), stop_str


def encode_prompt(tokenizer: Any, question: str, conv_template: str, device: torch.device) -> tuple[torch.Tensor, str]:
    prompt_text, stop_str = build_prompt(question, conv_template)
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    return input_ids, stop_str


def find_image_start_index(input_ids: torch.Tensor) -> int:
    ids = input_ids[0]
    positions = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
    return int(positions[0].item()) if positions.numel() > 0 else 0


def prepare_image_tensors(image: Image.Image, image_processor: Any, model: torch.nn.Module,
                          device: torch.device, dtype: torch.dtype) -> tuple[Any, list[tuple[int, int]]]:
    image_tensors = process_images([image], image_processor, model.config)
    if isinstance(image_tensors, torch.Tensor):
        image_tensors = image_tensors.to(device=device, dtype=dtype)
    else:
        image_tensors = [t.to(device=device, dtype=dtype) for t in image_tensors]
    return image_tensors, [image.size]


@torch.no_grad()
def _prepare_prompt_embeds(model, prompt_ids, image_tensors, image_sizes, base_cfg):
    """Run prepare_inputs_labels_for_multimodal once; return (embeds, image_token_length)."""
    prep = model.prepare_inputs_labels_for_multimodal(
        prompt_ids, None, None, None, None,
        image_tensors, modalities=["image"], image_sizes=image_sizes,
        illava_config=base_cfg,
    )
    if len(prep) == 8:
        _, _, _, _, prompt_embeds, _, _, image_token_length = prep
    else:
        _, _, _, _, prompt_embeds, _, _ = prep
        image_token_length = 0
    if prompt_embeds is None:
        raise RuntimeError("prepare_inputs_labels_for_multimodal returned no embeddings")
    return prompt_embeds, int(image_token_length)


@torch.no_grad()
def generate_answer(*, model, tokenizer, image_processor, sample, device, dtype,
                    conv_template, max_new_tokens, mode, keep_ratio) -> str:
    input_ids, stop_str = encode_prompt(tokenizer, sample["question"], conv_template, device)
    image_start_index = find_image_start_index(input_ids)
    with Image.open(sample["image_file"]) as im:
        image = im.convert("RGB")
    image_tensors, image_sizes = prepare_image_tensors(image, image_processor, model, device, dtype)

    if mode == "fullcache":
        cfg = build_fullcache_illava_config()
        cfg["illava_llm_image_token_start_index"] = image_start_index
    else:
        # Step 1: a passthrough-VIT prepare to discover image_token_length and
        # populate VIT_ATTN_MAP. The wrapper.generate() will run its own
        # prepare_inputs_labels_for_multimodal again, which is fine — it
        # produces the same passthrough result and resets VIT_ATTN_MAP for
        # the actual generate-time prefill.
        passthrough = build_passthrough_illava_config()
        passthrough["illava_llm_image_token_start_index"] = image_start_index
        prompt_embeds, image_token_length = _prepare_prompt_embeds(
            model, input_ids, image_tensors, image_sizes, passthrough
        )
        total_len = int(prompt_embeds.shape[1])
        cfg = compute_pruned_illava_config(
            keep_ratio=keep_ratio,
            image_token_length=image_token_length,
            total_len=total_len,
            image_start_index=image_start_index,
        )

    outputs = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0,
        use_cache=True,
        illava_config=cfg,
    )
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    if text.endswith(stop_str):
        text = text[: -len(stop_str)].strip()
    return text


@torch.no_grad()
def compute_answer_nll(*, model, tokenizer, image_processor, sample, device, dtype,
                       conv_template, max_answer_tokens, mode, keep_ratio) -> tuple[float, int]:
    """Teacher-force the GT answer.

    LLM forward_illava prunes image tokens at illava_llm_k layers. Image tokens
    are at the start (after a short text prefix); answer tokens are at the end
    and are never pruned. So logits at positions [-(n_ans+1):-1] still predict
    the answer tokens regardless of how many image tokens were dropped.
    """
    prompt_ids, _ = encode_prompt(tokenizer, sample["question"], conv_template, device)
    image_start_index = find_image_start_index(prompt_ids)
    with Image.open(sample["image_file"]) as im:
        image = im.convert("RGB")
    image_tensors, image_sizes = prepare_image_tensors(image, image_processor, model, device, dtype)

    answer_ids = tokenizer.encode(sample["answer"], add_special_tokens=False, return_tensors="pt").to(device)
    if max_answer_tokens is not None:
        answer_ids = answer_ids[:, :max_answer_tokens]
    n_answer = int(answer_ids.shape[1])
    if n_answer == 0:
        return 0.0, 0

    # First prepare with a passthrough config so we know image_token_length and
    # so VIT_ATTN_MAP is set in os.environ. Then build the per-sample illava
    # config (setting enable_illava_llm appropriately) and run the LM forward.
    base_cfg = build_passthrough_illava_config() if mode == "pruned" else build_fullcache_illava_config()
    base_cfg["illava_llm_image_token_start_index"] = image_start_index

    prompt_embeds, image_token_length = _prepare_prompt_embeds(
        model, prompt_ids, image_tensors, image_sizes, base_cfg
    )
    total_len = int(prompt_embeds.shape[1])

    if mode == "pruned":
        cfg = compute_pruned_illava_config(
            keep_ratio=keep_ratio,
            image_token_length=image_token_length,
            total_len=total_len,
            image_start_index=image_start_index,
        )
    else:
        cfg = base_cfg

    answer_embeds = model.get_model().embed_tokens(answer_ids)
    full_embeds = torch.cat([prompt_embeds, answer_embeds], dim=1)

    if cfg.get("enable_illava_llm", False) and image_token_length:
        outputs = model.forward_illava(
            inputs_embeds=full_embeds,
            illava_config=cfg,
            image_token_length=image_token_length,
            use_cache=True,
            return_dict=True,
        )
    else:
        # Bypass the broken wrapper.forward by calling Qwen2Model directly.
        inner = model.model(inputs_embeds=full_embeds, use_cache=True, return_dict=True)
        logits = model.lm_head(inner.last_hidden_state)
        outputs = type("O", (), {"logits": logits})

    logits = outputs.logits
    shift_logits = logits[:, -(n_answer + 1):-1, :].float()
    shift_labels = answer_ids
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="mean",
    )
    return float(loss.item()) * n_answer, n_answer


def load_rouge_refs(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "per_sample" in raw:
        records = raw["per_sample"]
    elif isinstance(raw, dict):
        records = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "answer": v}
                   for k, v in raw.items()]
    else:
        records = list(raw)
    refs: dict[str, str] = {}
    for rec in records:
        sid = str(rec.get("id", rec.get("sample_id")))
        ref = (
            rec.get("fullcache_answer")
            or rec.get("rouge_reference")
            or rec.get("pred")
            or rec.get("prediction")
            or rec.get("answer")
        )
        if sid and ref is not None:
            refs[sid] = str(ref)
    return refs


def _rouge_tokens(text: str) -> list[str]:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)
    return tokens or ["<empty>"]


def _lcs_len(a: list[str], b: list[str]) -> int:
    short, long = (a, b) if len(a) < len(b) else (b, a)
    prev = [0] * (len(short) + 1)
    for tok_long in long:
        cur = [0]
        for j, tok_short in enumerate(short, start=1):
            cur.append(prev[j - 1] + 1 if tok_long == tok_short else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _rouge_l_f1(pred: str, reference: str) -> float:
    p, r = _rouge_tokens(pred), _rouge_tokens(reference)
    lcs = _lcs_len(p, r)
    if lcs == 0:
        return 0.0
    precision = lcs / len(p)
    recall = lcs / len(r)
    return 2.0 * precision * recall / (precision + recall)


def rouge_l_score(pred: str, reference: str) -> float:
    pred_text = pred.strip() or "<empty>"
    ref_text = reference.strip() or "<empty>"
    alternatives = [a.strip() for a in re.split(r"<OR>|<AND>", ref_text) if a.strip()] or ["<empty>"]
    return max(_rouge_l_f1(pred_text, alt) for alt in alternatives)


def evaluate_dataset(*, dataset, args, model, tokenizer, image_processor,
                     device, dtype) -> None:
    defaults = DATASET_DEFAULTS[dataset]
    data_path = args.data_path or defaults["data_path"]
    image_path = args.image_path or defaults["image_path"]
    eval_samples = args.eval_samples if args.eval_samples is not None else defaults["eval_samples"]
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else defaults["max_new_tokens"]

    output_dir = Path(args.output_dir) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    predictions_path = output_dir / "predictions.jsonl"
    fullcache_ref_path = output_dir / "fullcache_rouge_ref.json"

    if result_path.exists() and not args.overwrite:
        print(f"[skip] {dataset}: {result_path} exists")
        return
    if predictions_path.exists():
        predictions_path.unlink()

    samples = load_samples(data_path, image_path, eval_samples)
    if not samples:
        raise ValueError(f"No samples loaded from {data_path}")

    rouge_refs = load_rouge_refs(args.rouge_ref_path)
    use_self_ref = not bool(rouge_refs)

    per_sample: list[dict[str, Any]] = []
    ref_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    scores: list[float] = []
    total_nll = 0.0
    total_tokens = 0
    t_start = time.time()

    desc = f"{dataset} onevision {args.mode}"
    if args.mode == "pruned":
        desc += f"(keep={args.keep_ratio})"
    iterator = tqdm(samples, desc=desc)
    for sample in iterator:
        try:
            pred = generate_answer(
                model=model, tokenizer=tokenizer, image_processor=image_processor,
                sample=sample, device=device, dtype=dtype,
                conv_template=args.conv_template, max_new_tokens=max_new_tokens,
                mode=args.mode, keep_ratio=args.keep_ratio,
            )
            sum_nll, n_tokens = compute_answer_nll(
                model=model, tokenizer=tokenizer, image_processor=image_processor,
                sample=sample, device=device, dtype=dtype,
                conv_template=args.conv_template,
                max_answer_tokens=args.max_answer_tokens,
                mode=args.mode, keep_ratio=args.keep_ratio,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            failures.append({"id": sample["id"], "error": repr(exc), "tb": traceback.format_exc()})
            print(f"[FAIL] sample {sample['id']}: {exc}\n{traceback.format_exc()}")
            torch.cuda.empty_cache()
            continue
        finally:
            # Anyres OneVision can spike to 20+ GB on a single sample; without
            # reclaiming PyTorch's reserved-but-unused blocks, the next sample's
            # peak allocation can collide with that fragmentation and OOM.
            torch.cuda.empty_cache()

        sid = str(sample["id"])
        rouge_ref = pred if use_self_ref else rouge_refs.get(sid, "")
        rouge_l = rouge_l_score(pred, rouge_ref)
        total_nll += sum_nll
        total_tokens += n_tokens
        scores.append(rouge_l)

        record = {
            "id": sid,
            "image": sample["image"],
            "question": sample["question"],
            "pred": pred,
            "rouge_reference": rouge_ref,
            "gt_answer": sample["answer"],
            "rouge_l_f": rouge_l,
            "n_answer_tokens": n_tokens,
            "sum_nll": sum_nll,
        }
        per_sample.append(record)
        ref_records.append({
            "id": sid,
            "image": sample["image"],
            "question": sample["question"],
            "answer": pred,
            "fullcache_answer": pred,
            "gt_answer": sample["answer"],
        })

        with predictions_path.open("a" if predictions_path.exists() else "w") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if total_tokens:
            iterator.set_postfix(
                rouge=f"{sum(scores) / len(scores):.4f}",
                ppl=f"{math.exp(total_nll / total_tokens):.3f}",
            )

    if not per_sample:
        raise RuntimeError(f"{dataset}: all samples failed")
    if total_tokens == 0:
        raise RuntimeError(f"{dataset}: no answer tokens scored")

    elapsed = time.time() - t_start
    result = {
        "dataset": dataset,
        "mode": args.mode,
        "keep_ratio": args.keep_ratio if args.mode == "pruned" else None,
        "model_path": args.model_path,
        "rouge_reference": "self_fullcache" if use_self_ref else args.rouge_ref_path,
        "rouge_l_f_mean": sum(scores) / len(scores),
        "ppl": math.exp(total_nll / total_tokens),
        "n_samples": len(per_sample),
        "n_failures": len(failures),
        "n_total_answer_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "sec_per_sample": elapsed / max(len(per_sample), 1),
        "args": vars(args),
        "per_sample": per_sample,
        "failures": failures,
    }
    with result_path.open("w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    if args.mode == "fullcache":
        with fullcache_ref_path.open("w") as f:
            json.dump(ref_records, f, ensure_ascii=False, indent=2)
        print(f"[REF] {dataset} full-cache ROUGE reference -> {fullcache_ref_path}")

    print(
        f"[DONE] {dataset} mode={args.mode} rouge_l={result['rouge_l_f_mean']:.4f} "
        f"ppl={result['ppl']:.4f} n={len(per_sample)} fail={len(failures)} -> {result_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_DEFAULTS),
                        default=["mmvet", "detail_1k"])
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-answer-tokens", type=int, default=None)
    parser.add_argument("--rouge-ref-path", default=None,
                        help="JSON of full-cache predictions to use as ROUGE reference. "
                             "Required (in spirit) for --mode pruned; if omitted, ROUGE "
                             "is self-referential.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["fullcache", "pruned"], default="fullcache")
    parser.add_argument("--keep-ratio", type=float, default=0.5,
                        help="Total-token-basis keep ratio (only used when --mode pruned). "
                             "Only image tokens are evicted; the projector budgets the cut.")
    parser.add_argument("--sweep", action="store_true",
                        help="Single-process sweep: run fullcache first, then pruned at "
                             "each --keep-ratio-list value, reusing the loaded model.")
    parser.add_argument("--keep-ratio-list", type=float, nargs="+", default=[0.2, 0.5, 0.8],
                        help="Keep ratios to run when --sweep is set.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--model-name", default="llava_qwen_training_free")
    parser.add_argument("--conv-template", default="qwen_1_5")
    parser.add_argument("--mm-spatial-pool-stride", type=int, default=2)
    parser.add_argument("--mm-spatial-pool-mode", default="bilinear")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (args.data_path or args.image_path) and len(args.datasets) != 1:
        raise ValueError("--data-path/--image-path overrides require selecting exactly one dataset")

    dtype = getattr(torch, args.torch_dtype)
    device = torch.device(args.device)
    patch_siglip_loader_to_local_init()

    overwrite_config = {
        "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
        "mm_spatial_pool_mode": args.mm_spatial_pool_mode,
    }
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, args.model_name,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        overwrite_config=overwrite_config,
        multimodal=True,
    )
    model.eval()
    if args.device_map != "auto":
        model = model.to(device)
    else:
        device = next(model.parameters()).device
    if dtype != torch.float16:
        model = model.to(dtype=dtype)

    print(f"[mode] {args.mode} keep_ratio={args.keep_ratio if args.mode == 'pruned' else 'N/A'}")
    print(f"[illava_llm_k] {ILLAVA_LLM_K} (per-sample illava_llm_r computed from keep_ratio)")

    for dataset in args.datasets:
        evaluate_dataset(
            dataset=dataset, args=args,
            model=model, tokenizer=tokenizer, image_processor=image_processor,
            device=device, dtype=dtype,
        )


if __name__ == "__main__":
    main()
