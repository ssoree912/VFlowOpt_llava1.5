## Experiment Plan

**ID**: EXP-20260429-002
**Author**: Codex
**Date**: 2026-04-29
**Status**: [ ] Planned  [x] Running  [ ] Done  [ ] Abandoned

---

### 1. Motivation
Run the same 10 requested LVLM benchmarks with LLaVA-OneVision at 0.10 keep ratio on GPU1 while the 0.25 run continues on GPU0.

### 2. Hypothesis
The 0.10 keep-ratio configuration will be faster/lighter than full retention but should lose more accuracy than 0.25, especially on document, chart, and VQA tasks.

### 3. What We Change
- Keep ratio: 0.10
- GPU: GPU1

### 4. What We Measure
- Each task's native lmms-eval metric.
- Completion status and failures per task.

### 5. What Stays The Same
- Model checkpoint: `/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov`
- Batch size: 1
- Harness: local `lmms-eval`
- Task set: same 10 local tasks as EXP-20260429-001

### 6. Baseline
- Parallel 0.25 keep-ratio run in EXP-20260429-001.

### 7. Expected Result
All 10 tasks should run, but VQAv2 is expected to dominate runtime.

### 8. Success Criteria
All taskwise outputs are written under `/workspace/VFlowOpt/experiments/EXP-20260429-002/outputs/keep10_10bench`.

### 9. What This Experiment Cannot Prove
- It does not establish full-token parity without same-run full-token baselines.
- MMBench dev scoring may use heuristic/random fallback for ambiguous responses if no OpenAI API key is available.

### 10. Runtime / Resources
- GPU: 1x RTX 4090, GPU1
- Expected runtime: long; taskwise execution avoids one-process request-memory blowups.
