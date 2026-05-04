#!/usr/bin/env bash
# OneVision MMVet + detail_1k VFlowOpt eval sweep on GPU 1.
# 1) full-cache pass (PPL on GT, ROUGE on self) — produces fullcache_rouge_ref.json.
# 2) pruned passes at keep_ratio = 0.2, 0.5, 0.8 — ROUGE scored against the
#    full-cache reference from step 1.
set -euo pipefail

ROOT=/workspace/VFlowOpt
ENV=${ENV:-/workspace/VFlowOpt/.conda/VFlowOpt}
PY="$ENV/bin/python"
SCRIPT="$ROOT/scripts/run_onevision_mmvet_detail_vflowopt.py"

GPU=${GPU:-1}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-$ROOT/logs/onevision_vflowopt_sweep_$TS}
mkdir -p "$OUT_ROOT"
echo "OUT_ROOT=$OUT_ROOT"

# DATASETS / KEEP_RATIOS / MAX_NEW_TOKENS overridable via env (space-separated lists).
read -r -a DATASETS <<<"${DATASETS:-mmvet detail_1k}"
read -r -a KEEP_RATIOS <<<"${KEEP_RATIOS:-0.2 0.5 0.8}"
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-}

run() {
  local mode=$1 keep=$2 dataset=$3
  local subdir
  if [[ "$mode" == "fullcache" ]]; then
    subdir="$OUT_ROOT/fullcache"
  else
    subdir="$OUT_ROOT/keep${keep}"
  fi
  mkdir -p "$subdir"
  local logf="$subdir/${dataset}.log"
  local extra=()
  if [[ "$mode" == "pruned" ]]; then
    extra+=(--mode pruned --keep-ratio "$keep" \
            --rouge-ref-path "$OUT_ROOT/fullcache/${dataset}/fullcache_rouge_ref.json")
  else
    extra+=(--mode fullcache)
  fi
  if [[ -n "$MAX_NEW_TOKENS" ]]; then
    extra+=(--max-new-tokens "$MAX_NEW_TOKENS")
  fi
  echo "===== $mode keep=$keep dataset=$dataset -> $logf ====="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$SCRIPT" \
    --datasets "$dataset" \
    --output-dir "$subdir" \
    --device-map cuda:0 --device cuda:0 \
    --overwrite \
    "${extra[@]}" \
    >"$logf" 2>&1
  echo "[ok] $mode keep=$keep $dataset"
}

# Step 1: fullcache for each dataset (each invocation re-loads model — fine).
for ds in "${DATASETS[@]}"; do
  run fullcache 0.0 "$ds"
done

# Step 2: pruned at each keep ratio for each dataset.
for keep in "${KEEP_RATIOS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    run pruned "$keep" "$ds"
  done
done

echo "ALL DONE -> $OUT_ROOT"
