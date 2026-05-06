# EXP-20260502-002: LLaVA-1.5 VFlowOpt Keep 0.50/0.20 Eval

## Goal

Run VFlowOpt on the datasets whose prompts do not hit the text-token floor at keep ratio 0.20, excluding ScienceQA.

## Model

- Checkpoint: `/workspace/look-m/models/llava-v1.5-7b`
- lmms-eval model: `llava_llama_training_free`
- Conversation template: `vicuna_v1`

## Tasks

- `textvqa_val`
- `gqa_local`
- `docvqa_val_local`
- `chartqa_local`
- `mme_local`
- `mmvet`
- `coco2017_cap_val`
- `nocaps_val`
- `textcaps_val`

## Protocol

- Keep ratios: `0.50`, `0.20`
- Total-token basis: `illava_total_keep_ratio=True`
- Actual eviction target: image tokens only
- Eviction layers: `illava_llm_k=9-18`
- Caption tasks use ROUGE_L-only aggregation to avoid BLEU/METEOR/CIDEr overhead.
- `MMVet` requires `OPENAI_API_KEY` for GPT-based evaluation; the shared run script skips it when the key is not set.

## Reproduction

```bash
cd /workspace/VFlowOpt
GPU_INDEX=1 \
KEEP_RATIOS="0.50 0.20" \
TASKS="textvqa_val gqa_local docvqa_val_local chartqa_local mme_local mmvet coco2017_cap_val nocaps_val textcaps_val" \
OUT_DIR=/workspace/VFlowOpt/experiments/EXP-20260502-002-llava15-vflowopt-keep50-20/outputs \
bash experiments/EXP-20260502-001-llava15-vflowopt-total-ratio/run.sh
```
