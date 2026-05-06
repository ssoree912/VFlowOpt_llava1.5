# EXP-20260502-001: LLaVA-1.5 VFlowOpt Total-Ratio Eval

## Goal

Run the VFlowOpt/lmms-eval protocol on original-format LLaVA-1.5-7B with keep ratios 0.30 and 0.10, where the removal budget is computed from the full prompt sequence length and only image tokens are evicted.

## Model

- Checkpoint: `/workspace/look-m/models/llava-v1.5-7b`
- lmms-eval model: `llava_llama_training_free`
- LLaVA loader model name: `llava_llama_training_free`
- Conversation template: `vicuna_v1`

## Tasks

- `coco2014_cap_val`
- `coco2017_cap_val`
- `flickr30k_test`
- `nocaps_val`
- `textcaps_val`
- `mmvet`
- `textvqa_val`
- `scienceqa_img_local`
- `gqa_local`
- `docvqa_val_local`
- `chartqa_local`
- `mme_local`

## Protocol

- Keep ratios: `0.30`, `0.10`
- Total-token basis: `illava_total_keep_ratio=True`
- Default eviction layers: `illava_llm_k=9-18`
- Per-layer keep schedule is computed as `target_keep_ratio ** (1 / number_of_eviction_layers)` so the two-stage sequence-level target is approximately the requested keep ratio before the image-only cap.
- `MMVet` requires `OPENAI_API_KEY` for GPT-based aggregation; the run script skips it unless the key is present or `RUN_MMVET=1`.

## Commands

```bash
GPU_INDEX=1 bash experiments/EXP-20260502-001-llava15-vflowopt-total-ratio/run.sh
```

For a quick single-task smoke test:

```bash
GPU_INDEX=1 KEEP_RATIOS=0.30 TASKS=chartqa_local LIMIT=1 bash experiments/EXP-20260502-001-llava15-vflowopt-total-ratio/run.sh
```
