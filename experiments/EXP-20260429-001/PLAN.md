## Experiment Plan

**ID**: EXP-20260429-001
**Author**: Codex
**Date**: 2026-04-29
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. Motivation
Evaluate LLaVA-OneVision with the training-free 0.25 keep-ratio configuration across the 10 requested LVLM benchmarks.

### 2. Hypothesis
The 0.25 keep-ratio run will preserve a useful fraction of full-token accuracy, with larger drops expected on OCR/document-heavy or fine-grained VQA tasks.

### 3. What We Change
- Method: `llava_onevision_training_free`
- Keep ratio: 0.25 token keep configuration

### 4. What We Measure
- Main metrics: each task's native lmms-eval metric
- Secondary metrics: task completion status, failures, output artifacts

### 5. What Stays The Same
- Model checkpoint: `/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov`
- GPU: GPU0 via `CUDA_VISIBLE_DEVICES=0`
- Batch size: 1
- Harness: local `lmms-eval`
- Local data root: `/workspace/zap/data/eval_hf`

### 6. Baseline
- ChartQA full-token baseline previously measured separately.
- This run itself is keep-ratio 0.25 only.

### 7. Expected Result
0.25 keep-ratio should run end-to-end on all 10 tasks, but VQAv2 is expected to dominate runtime.

### 8. Success Criteria
All 10 requested tasks complete and produce lmms-eval JSON results under the experiment output directory.

### 9. What This Experiment Cannot Prove
- It does not prove parity with VLMEvalKit because prompts, parsing, and metrics may differ.
- It does not compare against a same-run full-token baseline for all 10 datasets.
- MMBench dev scoring may use heuristic/random fallback for ambiguous responses if no OpenAI API key is available.

### 10. Runtime / Resources
- GPU: 1x RTX 4090, GPU0
- Expected runtime: long; VQAv2 validation has over 200k samples.
- Disk: outputs and sample logs under `/workspace/VFlowOpt/experiments/EXP-20260429-001/outputs`.
