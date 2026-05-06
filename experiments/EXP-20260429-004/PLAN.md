## Experiment Plan

**ID**: EXP-20260429-004
**Author**: Codex
**Date**: 2026-04-29
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. Motivation
Resume missing and failed LLaVA-OneVision training-free evaluations after fixing the forced `max_new_tokens=1024` OOM issue.

### 2. Hypothesis
Tasks that failed from excessive generation length at keep ratio 0.50 should now complete because each task's configured generation length is respected. DocVQA and MMStar should also run for keep ratios 0.25 and 0.10 because they were previously blocked by VQAv2.

### 3. What We Change
- Use the patched `llava_onevision_training_free` generation wrapper.
- Run selected missing tasks only:
  - keep 0.50: `mme_local`, `vizwiz_vqa_val_local`, `chartqa_local`, `docvqa_val_local`, `mmstar_local`
  - keep 0.25: `docvqa_val_local`, `mmstar_local`
  - keep 0.10: `docvqa_val_local`, `mmstar_local`
  - full cache: `docvqa_val_local`, `mmstar_local`

### 4. What We Measure
- Main metrics: each task's native `lmms-eval` metric.
- Secondary metrics: task completion status and whether a result JSON is produced.

### 5. What Stays The Same
- Model checkpoint: `/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov`
- GPU: GPU0 via `CUDA_VISIBLE_DEVICES=0`
- Batch size: 1
- Harness: local `lmms-eval`
- Local data root: `/workspace/zap/data/eval_hf`

### 6. Baseline
- Original keep 0.50 taskwise run: `EXP-20260429-003`
- Original keep 0.25 taskwise run: `EXP-20260429-001`
- Original keep 0.10 taskwise run: `EXP-20260429-002`

### 7. Expected Result
The short-answer/OCR tasks should be less likely to OOM after restoring task-specific `max_new_tokens`. VQAv2 is not included here because it was killed during CPU-side context construction and should be handled separately by sharding or limiting.

### 8. Success Criteria
Each selected task either produces a new result JSON or records a failed task row and continues to the next selected task.

### 9. What This Experiment Cannot Prove
- It does not solve VQAv2 CPU memory pressure.
- It only adds same-run full-token baselines for DocVQA and MMStar, not all datasets.
- It does not prove parity with VLMEvalKit because prompts, parsing, and metrics may differ.

### 10. Runtime / Resources
- GPU: 1x RTX 4090, GPU0
- Expected runtime: several hours depending on DocVQA and MMStar.
- Disk: outputs and logs under `/workspace/VFlowOpt/experiments/EXP-20260429-004`.
