## Experiment Plan

**ID**: EXP-20260501-001-milebench-llava15-origin
**Author**: Codex
**Date**: 2026-05-01
**Status**: [x] Planned  [ ] Running  [ ] Done  [ ] Abandoned

### 1. Motivation
Run LLaVA-1.5-7B on `/workspace/zap/data/MileBench` with the official MileBench prompt construction, output structure, and scoring flow.

### 2. Hypothesis
The local HF checkpoint `/workspace/zap/ckpts/llava-1.5-7b-hf` can be evaluated on MileBench by matching the official MileBench sample building and using HF LLaVA generation.

### 3. Independent Variables
- Model checkpoint: `/workspace/zap/ckpts/llava-1.5-7b-hf`
- Dataset subset: all MileBench datasets, or a named subset for smoke tests
- MileBench image mode: `combine_image=1` for HF LLaVA-1.5 compatibility
- Context budget: default 4096 tokens with 576 tokens per image

### 4. Dependent Variables
- MileBench per-dataset `eval.json`
- Aggregated `result.csv`
- Generation failures and truncation counts

### 5. Fixed Conditions
- Data root: `/workspace/zap/data/MileBench`
- Decoding: greedy, `max_new_tokens=512`, `temperature=0.0`
- Input mode: official MileBench `combined_1_images`
- Output root: this experiment's `outputs` directory

### 6. Baseline
- LOOK-M origin/MileBench outputs under `/workspace/look-m/outputs/llava15_origin_full` when available.

### 7. Expected Result
The script should produce MileBench-compatible `pred.json` files and optional official `eval.json` files for each dataset.

### 8. Success Criteria
- `scripts/milebench_llava15_infer.py --limit 1 --dataset ActionLocalization` writes a valid `pred.json`.
- Full run writes predictions and evaluation files without changing MileBench data.

### 9. Non-Claims
This run does not compare pruning methods. It is an origin/full-cache LLaVA-1.5 inference path. It uses MileBench's combined-image route rather than the LOOK-M true multi-image LLaVA backend.

### 10. Runtime / Resources
- GPU: one 24GB+ CUDA GPU recommended
- Runtime: full MileBench may take many hours
- Disk: predictions and eval files under `outputs`
