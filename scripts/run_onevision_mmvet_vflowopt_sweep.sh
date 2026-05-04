#!/usr/bin/env bash
# OneVision MMVet-only VFlowOpt eval sweep on GPU 1.
# 1) full-cache pass — produces fullcache_rouge_ref.json (= ROUGE reference).
# 2) pruned passes at keep_ratio = 0.2, 0.5, 0.8 — ROUGE scored against (1).
# detail_1k is intentionally skipped here (anyres OOM-prone on 24GB).
set -euo pipefail

ROOT=/workspace/VFlowOpt
ENV=${ENV:-/workspace/VFlowOpt/.conda/VFlowOpt}
PY="$ENV/bin/python"
SCRIPT="$ROOT/scripts/run_onevision_mmvet_detail_vflowopt.py"

GPU=${GPU:-1}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-$ROOT/logs/onevision_vflowopt_mmvet_$TS}
mkdir -p "$OUT_ROOT"
echo "OUT_ROOT=$OUT_ROOT"

KEEP_RATIOS=(0.2 0.5 0.8)
DATASET=mmvet

run() {
  local mode=$1 keep=$2
  local subdir
  if [[ "$mode" == "fullcache" ]]; then
    subdir="$OUT_ROOT/fullcache"
  else
    subdir="$OUT_ROOT/keep${keep}"
  fi
  mkdir -p "$subdir"
  local logf="$subdir/${DATASET}.log"
  local extra=()
  if [[ "$mode" == "pruned" ]]; then
    extra+=(--mode pruned --keep-ratio "$keep" \
            --rouge-ref-path "$OUT_ROOT/fullcache/${DATASET}/fullcache_rouge_ref.json")
  else
    extra+=(--mode fullcache)
  fi
  echo "===== $mode keep=$keep dataset=$DATASET -> $logf ====="
  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "$SCRIPT" \
    --datasets "$DATASET" \
    --output-dir "$subdir" \
    --device-map cuda:0 --device cuda:0 \
    --overwrite \
    "${extra[@]}" \
    >"$logf" 2>&1
  echo "[ok] $mode keep=$keep $DATASET"
}

run fullcache 0.0
for keep in "${KEEP_RATIOS[@]}"; do
  run pruned "$keep"
done

echo "ALL DONE -> $OUT_ROOT"
