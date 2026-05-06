## Experiment Plan

**ID**: EXP-20260429-003
**Author**: Codex
**Date**: 2026-04-29
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. Motivation
Evaluate LLaVA-OneVision with the training-free 0.50 keep-ratio configuration across the same 10 requested LVLM benchmarks.

### 2. Hypothesis
The 0.50 keep-ratio run should improve over 0.25 and 0.10 on most tasks because it preserves more visual/textual context, with the largest gains expected on OCR, chart, document, and detailed VQA tasks.

### 3. What We Change
- Method: `llava_onevision_training_free`
- Keep ratio: `0.50`

### 4. What We Measure
- Main metrics: each task's native `lmms-eval` metric
- Secondary metrics: task completion status, failures, output artifacts

### 5. What Stays The Same
- Model checkpoint: `/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov`
- GPU: GPU0 via `CUDA_VISIBLE_DEVICES=0`
- Batch size: 1
- Harness: local `lmms-eval`
- Local data root: `/workspace/zap/data/eval_hf`
- Task list: `mme_local`, `mmbench_en_dev_local`, `scienceqa_img_local`, `vizwiz_vqa_val_local`, `gqa_local`, `pope_local`, `vqav2_val_local`, `chartqa_local`, `docvqa_val_local`, `mmstar_local`

### 6. Baseline
- Same harness 0.25 run: `EXP-20260429-001`
- Same harness 0.10 run: `EXP-20260429-002`
- ChartQA full-token baseline measured separately under `logs/chartqa_local_full`

### 7. Expected Result
0.50 keep-ratio should score higher than lower keep ratios where the compression path is the limiting factor. VQAv2 may still fail or dominate memory/runtime because the 0.25 run was killed while building VQAv2 contexts.

### 8. Success Criteria
The run produces JSON results for all tasks that can finish locally; failed tasks are recorded and do not block the remaining tasks.

### 9. What This Experiment Cannot Prove
- It does not prove parity with VLMEvalKit because prompts, parsing, and metrics may differ.
- It does not compare against a same-run full-token baseline for all 10 datasets.
- MMBench dev scoring may use heuristic/random fallback for ambiguous responses if no OpenAI API key is available.
- If VQAv2 is killed during context construction, that failure is a harness/data-scale issue for this local setup, not a model quality result.

### 10. Runtime / Resources
- GPU: 1x RTX 4090, GPU0
- Expected runtime: long; VQAv2 validation has over 200k samples.
- Disk: outputs and logs under `/workspace/VFlowOpt/experiments/EXP-20260429-003`.
