#!/usr/bin/env python3
"""LLaVA-OneVision MileBench inference under VFlowOpt-style eviction.

Targets the 10 MileBench datasets where image placeholders are contiguous at
the start of the prompt (T-2/T-3/T-4 in the MileBench taxonomy):
  ObjectExistence, ObjectInteraction, MovingAttribute, ObjectShuffle,
  EgocentricNavigation, MovingDirection,
  CounterfactualInference, StateChange, CharacterOrder, SceneTransition.

For these the {image#1}{image#2}...{image#N} prefix becomes a single contiguous
block of image tokens after `prepare_inputs_labels_for_multimodal`, so the
Qwen2 `forward_illava` LLM-side pruning (which assumes contiguous image tokens
in [start : start + image_token_length]) applies as-is.

Modes:
  * --mode fullcache             : iLLaVA disabled.
  * --mode pruned --keep-ratio R : VIT passthrough + LLM-side pruning at
                                   illava_llm_k=[9,18], per-stage ratios derived
                                   so that *image-only* eviction gives
                                   total-token retention ≈ R (LLaVA-1.5
                                   VFlowOpt convention).

Image processing:
  * `image_aspect_ratio` is overridden to "pad" so each frame is a single
    729-token base patch (no anyres patch explosion). This keeps the prompt
    affordable for high-frame samples (EgocentricNavigation can hit 100+ frames).
  * look-m-style image-aware truncation drops the *earliest* frames when
    n_images × 729 + question + instruction would exceed `--max-context-len`.

Predictions are written in look-m's pred.json schema so look-m's
`evaluate.py` (multi-choice / ROUGE / needle scorer) consumes them directly.
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
LLAVA_ONEVISION_ROOT = REPO_ROOT / "src" / "LLaVA-OneVision"
LOOKM_ROOT = Path("/workspace/look-m")

for p in (str(REPO_ROOT), str(LLAVA_ONEVISION_ROOT), str(LOOKM_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from utils import MileBenchDataset  # look-m's dataset wrapper

# look-m's MileBenchDataset emits "<ImageHere>" (LLaVA-1.5 convention).
# OneVision tokenize_image_token expects DEFAULT_IMAGE_TOKEN ("<image>"),
# so we substitute when reading samples out.
LOOKM_IMAGE_PLACEHOLDER = "<ImageHere>"


DEFAULT_MODEL_PATH = "/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov"
DEFAULT_DATA_DIR = "/workspace/zap/data/MileBench"

DEFAULT_DATASETS = [
    "ObjectExistence", "ObjectInteraction", "MovingAttribute", "ObjectShuffle",
    "EgocentricNavigation", "MovingDirection",
    "CounterfactualInference", "StateChange", "CharacterOrder", "SceneTransition",
]

# LLM-side progressive pruning layers (matches run_onevision_mmvet_detail_vflowopt.py).
ILLAVA_LLM_K = [9, 18]
# OneVision video frame after SigLIP (27×27=729) → spatial pool stride 2 bilinear
# (14×14=196) + 1 image_newline = 197 tokens/frame. Round up for the look-m
# truncation budget; the actual measured count is recomputed per-sample anyway.
N_TOKENS_PER_IMAGE = 200


def patch_siglip_loader_to_local_init() -> None:
    """Bypass hard-coded external SigLIP path when initializing the vision tower."""
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


def patch_total_image_token_length(model) -> None:
    """Fix multi-image `image_token_length` returned by prepare_inputs_labels_for_multimodal.

    The upstream LlavaQwenTrainingFree implementation reassigns image_token_length
    inside its per-image loop, so the returned value reflects only the LAST
    image's token count. For multi-image MileBench samples the LLM-side
    eviction range [start : start + image_token_length] then covers only one
    image — the rest of the contiguous image-token block is never pruned.

    Wrap the method so the returned image_token_length is recomputed as
        inputs_embeds.shape[1] - count_of_non_IMAGE_TOKEN_INDEX_in_input_ids
    which is exact for any contiguous-or-interleaved layout (text tokens
    survive prepare 1:1; everything else is image features).
    """
    if getattr(model, "_milebench_total_itl_patched", False):
        return
    orig = model.prepare_inputs_labels_for_multimodal

    def wrapped(input_ids, position_ids, attention_mask, past_key_values, labels,
                images, modalities=["image"], image_sizes=None, **kwargs):
        result = orig(input_ids, position_ids, attention_mask, past_key_values,
                      labels, images, modalities=modalities,
                      image_sizes=image_sizes, **kwargs)
        if isinstance(result, tuple) and len(result) == 8 and result[4] is not None and input_ids is not None:
            inp, pos, attn, pkv, embeds, lab, src, raw_itl = result
            text_count = int((input_ids != IMAGE_TOKEN_INDEX).sum().item())
            correct_itl = int(embeds.shape[1]) - text_count
            if os.environ.get("MILEBENCH_DEBUG_ITL"):
                print(f"[itl-patch] raw={raw_itl} corrected={correct_itl} "
                      f"embeds={embeds.shape} text_count={text_count}", flush=True)
            return (inp, pos, attn, pkv, embeds, lab, src, max(0, correct_itl))
        return result

    model.prepare_inputs_labels_for_multimodal = wrapped
    model._milebench_total_itl_patched = True


def build_passthrough_illava_config(image_start_index: int = 0) -> dict[str, Any]:
    """ViT passthrough config that still populates VIT_ATTN_MAP env var."""
    return {
        "enable_illava_llm": False,
        "enable_illava_vit": True,
        "illava_llm_k": ILLAVA_LLM_K,
        "illava_llm_r": [1.0] * len(ILLAVA_LLM_K),
        "illava_vit_k": [25],
        "illava_vit_t": 0.0,
        "illava_vit_alpha_v": 0.0,
        "illava_vit_m": 1000.0,
        "illava_vit_r": 1.0,
        "illava_llm_image_token_start_index": image_start_index,
        "illava_track_vit_source": False,
        "illava_track_llm_source": False,
    }


def build_fullcache_illava_config(image_start_index: int = 0) -> dict[str, Any]:
    cfg = build_passthrough_illava_config(image_start_index)
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
    Split that geometrically across len(ILLAVA_LLM_K) stages so the product
    of per-stage ratios reproduces the target.
    """
    base = build_passthrough_illava_config(image_start_index)
    if keep_ratio >= 1.0 or image_token_length <= 0:
        base["enable_illava_llm"] = False
        return base
    n_img = int(image_token_length)
    n_keep_img = int(round(n_img - (1.0 - keep_ratio) * total_len))
    n_keep_img = max(1, min(n_img, n_keep_img))
    final_image_keep = n_keep_img / max(1, n_img)
    num_stages = len(ILLAVA_LLM_K)
    per_stage_ratio = final_image_keep ** (1.0 / num_stages)
    base["enable_illava_llm"] = True
    base["illava_llm_r"] = [per_stage_ratio] * num_stages
    return base


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


def count_text_tokens(input_ids: torch.Tensor) -> int:
    """Number of non-image tokens in input_ids — what survives prepare()."""
    return int((input_ids[0] != IMAGE_TOKEN_INDEX).sum().item())


def total_image_tokens(prompt_embeds: torch.Tensor, input_ids: torch.Tensor) -> int:
    """Total contiguous image-token count = prefill_len − text_token_count.

    `prepare_inputs_labels_for_multimodal` only returns the *last* image's
    token count when there are multiple images, so we recompute the total
    for the multi-image-but-contiguous case.
    """
    return int(prompt_embeds.shape[1]) - count_text_tokens(input_ids)


def prepare_video_tensor(images: list[Image.Image], image_processor: Any,
                         device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Stack the MileBench frames into a single (T, 3, 384, 384) video tensor.

    OneVision's video modality is the natural fit for multi-frame samples: the
    `prepare_inputs_labels_for_multimodal` "video" branch produces a single
    contiguous block of image tokens after spatial pooling, and (importantly
    for VFlowOpt) it populates `os.environ["VIT_ATTN_MAP"]` from real ViT
    attention so the LLM-side eviction has a meaningful importance map.
    """
    bg = tuple(int(x * 255) for x in image_processor.image_mean)
    processed = []
    for im in images:
        # expand2square first so we don't squash aspect ratio.
        w, h = im.size
        if w != h:
            side = max(w, h)
            sq = Image.new(im.mode, (side, side), bg)
            sq.paste(im, ((side - w) // 2, (side - h) // 2))
            im = sq
        out = image_processor.preprocess(im, return_tensors="pt")["pixel_values"][0]
        processed.append(out)
    video = torch.stack(processed, dim=0).to(device=device, dtype=dtype)
    return video


def open_images(paths: list[str]) -> list[Image.Image]:
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(im.convert("RGB").copy())
    return out


@torch.no_grad()
def _prepare_prompt_embeds(model, prompt_ids, image_tensors, image_sizes,
                           modalities, illava_cfg):
    prep = model.prepare_inputs_labels_for_multimodal(
        prompt_ids, None, None, None, None,
        image_tensors, modalities=modalities, image_sizes=image_sizes,
        illava_config=illava_cfg,
    )
    if len(prep) == 8:
        _, _, _, _, prompt_embeds, _, _, _ = prep
    else:
        _, _, _, _, prompt_embeds, _, _ = prep
    if prompt_embeds is None:
        raise RuntimeError("prepare_inputs_labels_for_multimodal returned no embeddings")
    return prompt_embeds


@torch.no_grad()
def generate_answer(*, model, tokenizer, image_processor, sample, device, dtype,
                    conv_template: str, max_new_tokens: int, mode: str,
                    keep_ratio: float) -> str:
    images = open_images(sample["image_paths"])
    video_tensor = prepare_video_tensor(images, image_processor, device, dtype)
    # Single 4D tensor wrapped in a list — prepare_inputs_labels_for_multimodal
    # expects either a list of 4D tensors or a 5D tensor. video modality reads
    # the first list element as a (T, 3, H, W) frame stack.
    image_tensors = [video_tensor]
    image_sizes = [(images[0].size[0], images[0].size[1])]
    modalities = ["video"]

    # Collapse all per-frame `<image>` placeholders to a single `<image>` so the
    # video branch produces one contiguous spatial-pool image-token block.
    question = sample["question"]
    n_placeholders = question.count(DEFAULT_IMAGE_TOKEN)
    if n_placeholders > 1:
        question = question.replace(DEFAULT_IMAGE_TOKEN, "", n_placeholders - 1)
    prompt_text, stop_str = build_prompt(question, conv_template)
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    image_start_index = find_image_start_index(input_ids)

    if mode == "fullcache":
        cfg = build_fullcache_illava_config(image_start_index)
    else:
        # Run a passthrough prepare to (a) measure the spatial-pool image-token
        # block and total prefill length, (b) populate VIT_ATTN_MAP for the
        # generate-time pass.
        passthrough = build_passthrough_illava_config(image_start_index)
        prompt_embeds = _prepare_prompt_embeds(
            model, input_ids, image_tensors, image_sizes, modalities, passthrough
        )
        n_img = total_image_tokens(prompt_embeds, input_ids)
        total_len = int(prompt_embeds.shape[1])
        cfg = compute_pruned_illava_config(
            keep_ratio=keep_ratio,
            image_token_length=n_img,
            total_len=total_len,
            image_start_index=image_start_index,
        )
        del prompt_embeds

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
    """Group samples by image count (look-m's batching strategy)."""
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
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help="MileBench dataset names. Default = the 10 T-2/T-3/T-4 datasets.")
    parser.add_argument("--output-dir", required=True,
                        help="Per-run output dir; predictions saved as <dir>/<dataset>/pred.json.")
    parser.add_argument("--mode", choices=["fullcache", "pruned"], default="fullcache")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--eval-samples", type=int, default=None,
                        help="Cap on samples per dataset (for smoke tests).")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Generation budget. Multi-choice answers are short; 64 is plenty.")
    parser.add_argument("--max-context-len", type=int, default=12288,
                        help="Image-aware truncation budget (look-m semantics). With "
                             "video modality and stride-2 spatial pool, frames are "
                             "~200 tok each, so 12288 fits ~60 frames + question.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--model-name", default="llava_qwen_training_free")
    parser.add_argument("--conv-template", default="qwen_1_5")
    parser.add_argument("--mm-spatial-pool-stride", type=int, default=2)
    parser.add_argument("--mm-spatial-pool-mode", default="bilinear")
    parser.add_argument("--image-aspect-ratio", default="pad",
                        help="Override OneVision aspect ratio. 'pad' = single 729-token "
                             "patch per image (no anyres). 'anyres_max_9' = upstream default.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dtype = getattr(torch, args.torch_dtype)
    device = torch.device(args.device)
    patch_siglip_loader_to_local_init()

    overwrite_config = {
        "image_aspect_ratio": args.image_aspect_ratio,
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
    patch_total_image_token_length(model)
    if args.device_map != "auto":
        model = model.to(device)
    else:
        device = next(model.parameters()).device
    if dtype != torch.float16:
        model = model.to(dtype=dtype)

    print(f"[mode] {args.mode} keep_ratio={args.keep_ratio if args.mode == 'pruned' else 'N/A'}")
    print(f"[image_aspect_ratio] {args.image_aspect_ratio}  n_tokens_per_image={N_TOKENS_PER_IMAGE}")
    print(f"[max_context_len] {args.max_context_len}")
    print(f"[illava_llm_k] {ILLAVA_LLM_K}")

    for dataset_name in args.datasets:
        evaluate_dataset(
            dataset_name=dataset_name, args=args,
            model=model, tokenizer=tokenizer, image_processor=image_processor,
            device=device, dtype=dtype,
        )


if __name__ == "__main__":
    main()
