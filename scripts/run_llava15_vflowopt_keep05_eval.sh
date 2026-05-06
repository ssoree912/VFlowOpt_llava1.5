#!/usr/bin/env bash
# VFlowOpt-style LLM-side progressive image-token pruning on LLaVA-1.5 7B.
# keep_ratio is on the TOTAL token basis (text + image), only image tokens
# are evicted, end-of-prefill image-token count matches keep_ratio*prompt_len.
# Datasets are loaded locally from /workspace/zap/data/eval/ via the
# scripts/lmms_eval_local_run.py wrapper (same approach as the fullcache run).
set -euo pipefail

ENV_DIR="/workspace/VFlowOpt/.conda/VFlowOpt"
MODEL_DIR="/workspace/zap/ckpts/llava-v1.5-7b"
WRAPPER="/workspace/VFlowOpt/scripts/lmms_eval_local_run.py"
KEEP_RATIO="${KEEP_RATIO:-0.5}"
ILLAVA_LLM_K="${ILLAVA_LLM_K:-10-21}"
GPU="${GPU:-1}"
OUT_ROOT="/workspace/VFlowOpt/logs/llava15_vflowopt_keep${KEEP_RATIO}_$(date +%Y%m%d_%H%M%S)"
SUFFIX="llava15_7b_vflowopt_keep${KEEP_RATIO}"

export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/workspace/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${GPU}"

mkdir -p "$OUT_ROOT"
echo "Logs: $OUT_ROOT"
echo "keep_ratio (total-token basis): $KEEP_RATIO"
echo "illava_llm_k (LLaMA layer indices for progressive prune): $ILLAVA_LLM_K"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

TASKS=(
  docvqa_val
  coco2017_cap_val
  nocaps_val
  textcaps_val
)

for task in "${TASKS[@]}"; do
  task_log="$OUT_ROOT/${task}.log"
  echo "========== $task =========="
  echo "  log: $task_log"
  if python "$WRAPPER" \
      --model llava_15_training_free \
      --model_args "pretrained=${MODEL_DIR},conv_template=vicuna_v1,model_name=llava-v1.5-7b-training_free,device_map=cuda:0,keep_ratio=${KEEP_RATIO},illava_llm_k=${ILLAVA_LLM_K}" \
      --tasks "$task" \
      --batch_size 1 \
      --log_samples \
      --log_samples_suffix "$SUFFIX" \
      --output_path "$OUT_ROOT" \
        2>&1 | tee "$task_log"; then
    echo "[ok] $task"
  else
    echo "[fail] $task — see $task_log" | tee -a "$OUT_ROOT/FAILURES.log"
  fi
done

echo "Done. Logs and lmms-eval results under $OUT_ROOT"
