# EXP-20260502-001 Result

## Status

Partial: ScienceQA completed. Other planned datasets are not completed in this result file.

## ScienceQA

Run date: 2026-05-02

| Model | Dataset | keep ratio | Metric | Value | Stderr |
|---|---|---:|---|---:|---:|
| LLaVA-1.5-7B + VFlowOpt | scienceqa_img_local | 0.30 | exact_match | 0.690630 | 0.010295 |
| LLaVA-1.5-7B + VFlowOpt | scienceqa_img_local | 0.10 | exact_match | 0.692613 | 0.010276 |

## Protocol Notes

- Checkpoint: `/workspace/look-m/models/llava-v1.5-7b`
- lmms-eval model: `llava_llama_training_free`
- Total-token keep ratio enabled: `illava_total_keep_ratio=True`
- Eviction layers: `illava_llm_k=9-18`
- The ScienceQA postprocess bug for outputs like `B.` was fixed before this run.

## Artifacts

- Log: `experiments/EXP-20260502-001-llava15-vflowopt-total-ratio/logs/scienceqa_20260502_081317.log`
- keep 0.30 result: `experiments/EXP-20260502-001-llava15-vflowopt-total-ratio/outputs_scienceqa/keep030/scienceqa_img_local/models__llava-v1.5-7b/20260502_161321_results.json`
- keep 0.10 result: `experiments/EXP-20260502-001-llava15-vflowopt-total-ratio/outputs_scienceqa/keep010/scienceqa_img_local/models__llava-v1.5-7b/20260502_161647_results.json`

## Reproduction

```bash
cd /workspace/VFlowOpt
GPU_INDEX=1 KEEP_RATIOS="0.30 0.10" TASKS=scienceqa_img_local OUT_DIR=/workspace/VFlowOpt/experiments/EXP-20260502-001-llava15-vflowopt-total-ratio/outputs_scienceqa bash experiments/EXP-20260502-001-llava15-vflowopt-total-ratio/run.sh
```
