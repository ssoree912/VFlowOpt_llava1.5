#!/usr/bin/env bash
set -euo pipefail

cd /workspace/VFlowOpt

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

PYTHON="${PYTHON:-/opt/conda/envs/vflowopt_chartqa_eval/bin/python}"
DATA_DIR="${DATA_DIR:-/workspace/zap/data/MileBench}"
MODEL_PATH="${MODEL_PATH:-/workspace/zap/ckpts/llava-1.5-7b-hf}"
MODEL_ID="${MODEL_ID:-llava15_origin_hf}"
EXP_DIR="/workspace/VFlowOpt/experiments/EXP-20260501-001-milebench-llava15-origin"
OUT_DIR="${OUT_DIR:-${EXP_DIR}/outputs}"
LOG_DIR="${EXP_DIR}/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${RUN_ID}.log"

mkdir -p "$OUT_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== EXP-20260501-001 START $(date -Is) ====="
echo "[config] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[config] data=${DATA_DIR}"
echo "[config] model=${MODEL_PATH}"
echo "[config] model_id=${MODEL_ID}"
echo "[config] output=${OUT_DIR}"

"$PYTHON" scripts/milebench_llava15_infer.py \
  --data-dir "$DATA_DIR" \
  --dataset all \
  --model-path "$MODEL_PATH" \
  --model-id "$MODEL_ID" \
  --output-dir "$OUT_DIR" \
  --device cuda:0 \
  --combine-image 1 \
  --evaluate \
  --score \
  --continue-on-error

echo "===== EXP-20260501-001 DONE $(date -Is) ====="
