#!/usr/bin/env python3
"""Run LLaVA-1.5-7B inference on MileBench.

The data path, prompt construction, output layout, and optional scoring match
the official MileBench generate/evaluate flow. The model loader targets the
local Hugging Face LLaVA-1.5 checkpoint used in this workspace.

HF LLaVA-1.5 accepts one image tensor per prompt in this environment, so the
default uses MileBench's official `combine_image=1` path. Pass
`--combine-image 0` only when running with a backend that supports true
multi-image LLaVA prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, set_seed


MILEBENCH_DATASETS = [
    "ALFRED",
    "ActionLocalization",
    "ActionPrediction",
    "ActionSequence",
    "CLEVR-Change",
    "CharacterOrder",
    "CounterfactualInference",
    "DocVQA",
    "EgocentricNavigation",
    "GPR1200",
    "IEdit",
    "ImageNeedleInAHaystack",
    "MMCoQA",
    "MovingAttribute",
    "MovingDirection",
    "MultiModalQA",
    "OCR-VQA",
    "ObjectExistence",
    "ObjectInteraction",
    "ObjectShuffle",
    "SceneTransition",
    "SlideVQA",
    "Spot-the-Diff",
    "StateChange",
    "TQA",
    "TextNeedleInAHaystack",
    "WebQA",
    "WikiVQA",
    "nuscenes",
]

MILEBENCH_IMAGE_TOKEN = "<ImageHere>"
LLAVA_IMAGE_TOKEN = "<image>"


def parse_torch_dtype(value: str) -> torch.dtype:
    if not hasattr(torch, value):
        raise ValueError(f"Unsupported torch dtype: {value}")
    dtype = getattr(torch, value)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {value}")
    return dtype


def configure_llava_processor(processor: Any, config: Any) -> Any:
    """Fill processor metadata needed for LLaVA image-token expansion."""
    vision_config = getattr(config, "vision_config", None)
    if getattr(processor, "patch_size", None) is None and vision_config is not None:
        patch_size = getattr(vision_config, "patch_size", None)
        if patch_size is not None:
            processor.patch_size = int(patch_size)

    if getattr(processor, "vision_feature_select_strategy", None) is None:
        strategy = getattr(config, "vision_feature_select_strategy", None)
        if strategy is not None:
            processor.vision_feature_select_strategy = str(strategy)

    if getattr(processor, "num_additional_image_tokens", None) is None:
        additional = getattr(config, "num_additional_image_tokens", None)
        processor.num_additional_image_tokens = 0 if additional is None else int(additional)

    return processor


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def move_batch_to_device(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def append_choice_list(context: str, choice_list: list[Any] | None, dataset_name: str) -> str:
    if not choice_list:
        return context

    choice_str = "\nChoice list: \n"
    choice_str += "\n".join(
        (f"{chr(65 + idx)}. " if dataset_name != "GPR1200" else "") + str(item)
        for idx, item in enumerate(choice_list)
    )
    choice_str += "\nYour answer is: "
    return context + choice_str


def replace_milebench_placeholders(context: str, image_count: int, combine_image: int | None) -> str:
    for i in range(image_count):
        image_key = f"{{image#{i + 1}}}"
        table_key = f"{{table#{i + 1}}}"
        if combine_image:
            context = context.replace(image_key, f"<Image {i + 1}> ")
            context = context.replace(table_key, f"<Image {i + 1}> ")
        else:
            context = context.replace(image_key, MILEBENCH_IMAGE_TOKEN)
            context = context.replace(table_key, MILEBENCH_IMAGE_TOKEN)
    return context


def make_image_paths(
    sample: dict[str, Any],
    img_dir: Path,
    combine_image: int | None,
) -> list[Path]:
    if combine_image:
        key = f"combined_{combine_image}_images"
        combined_dir = img_dir.parent / key
        return [combined_dir / path for path in sample["task_instance"][key]]
    return [img_dir / path for path in sample["task_instance"]["images_path"]]


def build_milebench_sample(
    sample: dict[str, Any],
    task_instructions: dict[str, str],
    img_dir: Path,
    tokenizer: Any,
    dataset_name: str,
    max_context_len: int,
    n_tokens_per_image: int,
    combine_image: int | None,
) -> dict[str, Any]:
    """Build a sample using the official MileBench left-truncation policy."""
    task_instruction = task_instructions[sample["task_instruction_id"]]
    task_instance = sample["task_instance"]
    image_count = len(task_instance["images_path"])

    context = task_instance["context"]
    context = append_choice_list(context, task_instance.get("choice_list"), dataset_name)
    context = replace_milebench_placeholders(context, image_count, combine_image)
    raw_img_list = make_image_paths(sample, img_dir, combine_image)

    instruction_ids = tokenizer(task_instruction, add_special_tokens=False).input_ids
    length_for_context: int | None = None
    if max_context_len > 0:
        length_for_context = max(max_context_len - len(instruction_ids), 0)

    fragments = context.split(MILEBENCH_IMAGE_TOKEN)[::-1]
    past_total_len = 0
    context_id_chunks: list[list[int]] = []
    retained_images: list[Path] = []
    image_start = False
    truncated = False

    for fragment in fragments:
        cur_ids = tokenizer(fragment, add_special_tokens=False).input_ids
        cur_len = len(cur_ids)
        if length_for_context is not None and cur_len + past_total_len > length_for_context:
            if not context_id_chunks and length_for_context > 0:
                context_id_chunks.insert(0, cur_ids[-length_for_context:])
            truncated = True
            break

        image_start = False
        context_id_chunks.insert(0, cur_ids)
        past_total_len += cur_len

        if not combine_image:
            if length_for_context is not None and n_tokens_per_image + past_total_len > length_for_context:
                truncated = True
                break
            if raw_img_list:
                image_start = True
                retained_images.insert(0, raw_img_list.pop(-1))
                past_total_len += n_tokens_per_image

    ret_context = ""
    if context_id_chunks:
        for chunk in context_id_chunks[:-1]:
            ret_context += tokenizer.decode(chunk)
            ret_context += MILEBENCH_IMAGE_TOKEN
        ret_context += tokenizer.decode(context_id_chunks[-1])

    if combine_image:
        if len(raw_img_list) != 1:
            raise ValueError(
                f"combine_image={combine_image} expects one combined image, got {len(raw_img_list)}"
            )
        retained_images.insert(0, raw_img_list.pop(-1))
        ret_context = f"{MILEBENCH_IMAGE_TOKEN}\n{task_instruction}\n{ret_context}"
    else:
        if image_start:
            ret_context = MILEBENCH_IMAGE_TOKEN + ret_context
        ret_context = f"{task_instruction}\n{ret_context}"
        if raw_img_list:
            truncated = True

    return {
        "sample_id": sample["sample_id"],
        "question": ret_context,
        "image_paths": retained_images,
        "gt_response": str(sample["response"]),
        "was_truncated": truncated,
    }


def format_for_llava(question: str) -> str:
    text = question
    while f"{MILEBENCH_IMAGE_TOKEN}{MILEBENCH_IMAGE_TOKEN}" in text:
        text = text.replace(
            f"{MILEBENCH_IMAGE_TOKEN}{MILEBENCH_IMAGE_TOKEN}",
            f"{MILEBENCH_IMAGE_TOKEN}\n{MILEBENCH_IMAGE_TOKEN}",
        )
    return text.replace(MILEBENCH_IMAGE_TOKEN, LLAVA_IMAGE_TOKEN)


def build_llava_prompt(question: str) -> str:
    return f"USER: {format_for_llava(question)}\nASSISTANT:"


def generate_one(
    model: LlavaForConditionalGeneration,
    processor: Any,
    sample: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    gen_kwargs: dict[str, Any],
) -> str:
    prompt = build_llava_prompt(sample["question"])
    expected_images = prompt.count(LLAVA_IMAGE_TOKEN)
    image_paths: list[Path] = sample["image_paths"]
    if expected_images != len(image_paths):
        raise ValueError(
            f"Prompt/image mismatch for sample_id={sample['sample_id']}: "
            f"{expected_images} placeholders vs {len(image_paths)} images"
        )
    if len(image_paths) > 1:
        raise ValueError(
            "HF LLaVA-1.5 supports one image per prompt here. "
            "Use MileBench --combine-image 1 for this script."
        )

    images = [load_image(path) for path in image_paths]
    inputs = processor(images=images, text=prompt, return_tensors="pt")
    inputs = move_batch_to_device(dict(inputs), device=device, dtype=dtype)
    prompt_len = int(inputs["input_ids"].shape[1])

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    with torch.inference_mode():
        generation_kwargs = dict(gen_kwargs)
        if not generation_kwargs.get("do_sample", False) and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        output_ids = model.generate(
            **inputs,
            pad_token_id=pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )

    new_tokens = output_ids[0, prompt_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def output_record(
    sample: dict[str, Any],
    prediction: str,
    model_id: str,
    gen_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "image": [str(path) for path in sample["image_paths"]],
        "question": sample["question"],
        "gt_response": sample["gt_response"],
        "gen_model_id": model_id,
        "pred_response": prediction,
        "gen_kwargs": gen_kwargs,
    }


def load_annotation(data_dir: Path, dataset_name: str, combine_image: int | None) -> dict[str, Any]:
    dataset_dir = data_dir / dataset_name
    if combine_image and combine_image != 1:
        annotation_path = dataset_dir / f"{dataset_name}_combined_{combine_image}.json"
    else:
        annotation_path = dataset_dir / f"{dataset_name}.json"
    with annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluate(args: argparse.Namespace, dataset_name: str) -> None:
    result_dir = Path(args.output_dir) / args.model_id
    command = [
        sys.executable,
        args.eval_script,
        "--data-dir",
        str(args.data_dir),
        "--dataset",
        dataset_name,
        "--result-dir",
        str(result_dir),
    ]
    subprocess.run(command, check=args.strict_eval)


def run_score(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        args.score_script,
        "--result-dir",
        str(args.output_dir),
        "--models",
        args.model_id,
    ]
    subprocess.run(command, check=args.strict_eval)


def run_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    model: LlavaForConditionalGeneration,
    processor: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    annotation = load_annotation(Path(args.data_dir), dataset_name, args.combine_image)
    data = annotation["data"]
    if args.limit is not None:
        data = data[: args.limit]

    dataset_dir = Path(args.data_dir) / dataset_name
    img_dir = dataset_dir / "images"
    pred_dir = Path(args.output_dir) / args.model_id / dataset_name
    pred_path = pred_dir / "pred.json"
    meta_path = pred_dir / "run_config.json"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if pred_path.exists() and not args.overwrite:
        print(f"[skip] {dataset_name}: {pred_path} exists")
        if args.evaluate and args.limit is None:
            run_evaluate(args, dataset_name)
        elif args.evaluate:
            print(f"[skip-eval] {dataset_name}: --limit is set, official MileBench eval needs full predictions")
        return

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p

    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    n_truncated = 0

    task_instructions = annotation["meta_data"]["task_instruction"]
    pbar = tqdm(data, desc=dataset_name)
    for raw_sample in pbar:
        try:
            sample = build_milebench_sample(
                raw_sample,
                task_instructions=task_instructions,
                img_dir=img_dir,
                tokenizer=processor.tokenizer,
                dataset_name=dataset_name,
                max_context_len=args.max_context_len,
                n_tokens_per_image=args.n_tokens_per_image,
                combine_image=args.combine_image,
            )
            n_truncated += int(sample["was_truncated"])
            answer = generate_one(model, processor, sample, device, dtype, gen_kwargs)
            predictions.append(output_record(sample, answer, args.model_id, gen_kwargs))
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_id": raw_sample.get("sample_id"), "error": repr(exc)})
            if not args.continue_on_error:
                raise
            fallback = {
                "sample_id": raw_sample["sample_id"],
                "question": "",
                "image_paths": [],
                "gt_response": str(raw_sample.get("response", "")),
                "was_truncated": False,
            }
            predictions.append(output_record(fallback, "", args.model_id, gen_kwargs))
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    run_config = {
        "dataset": dataset_name,
        "model_path": args.model_path,
        "model_id": args.model_id,
        "n_samples": len(predictions),
        "n_failures": len(failures),
        "n_truncated": n_truncated,
        "combine_image": args.combine_image,
        "max_context_len": args.max_context_len,
        "n_tokens_per_image": args.n_tokens_per_image,
        "gen_kwargs": gen_kwargs,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    if failures:
        with (pred_dir / "failures.json").open("w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)

    print(f"[done] {dataset_name}: wrote {pred_path}")
    if args.evaluate and args.limit is None:
        run_evaluate(args, dataset_name)
    elif args.evaluate:
        print(f"[skip-eval] {dataset_name}: --limit is set, official MileBench eval needs full predictions")


def resolve_datasets(values: list[str]) -> list[str]:
    if len(values) == 1 and values[0] == "all":
        return MILEBENCH_DATASETS
    datasets: list[str] = []
    for value in values:
        if value == "all":
            datasets.extend(MILEBENCH_DATASETS)
        elif "," in value:
            datasets.extend(part for part in value.split(",") if part)
        else:
            datasets.append(value)
    return list(dict.fromkeys(datasets))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLaVA-1.5-7B MileBench inference")
    parser.add_argument("--data-dir", default="/workspace/zap/data/MileBench")
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument("--model-path", default="/workspace/zap/ckpts/llava-1.5-7b-hf")
    parser.add_argument("--model-id", default="llava15_origin_hf")
    parser.add_argument("--output-dir", default="outputs/milebench")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--max-context-len", type=int, default=4096)
    parser.add_argument("--n-tokens-per-image", type=int, default=576)
    parser.add_argument(
        "--combine-image",
        type=int,
        default=1,
        help="MileBench combined-image mode. Use 1 for HF LLaVA-1.5; use 0 to disable.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--strict-eval", action="store_true")
    parser.add_argument("--eval-script", default="/workspace/look-m/evaluate.py")
    parser.add_argument("--score-script", default="/workspace/look-m/score.py")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    if args.combine_image is not None and args.combine_image <= 0:
        args.combine_image = None

    datasets = resolve_datasets(args.dataset)
    dtype = parse_torch_dtype(args.dtype)
    device = torch.device(args.device)

    print(f"[load] model={args.model_path} device={device} dtype={dtype}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor = configure_llava_processor(processor, model.config)
    processor.tokenizer.padding_side = "left"

    for dataset_name in datasets:
        run_dataset(args, dataset_name, model, processor, device, dtype)

    if args.score:
        run_score(args)


if __name__ == "__main__":
    main()
