#!/usr/bin/env bash
# Full-cache (no token pruning) LLaVA-OneVision-Qwen2-7B-OV eval over docvqa_val,
# chartqa, textvqa_val using locally-saved datasets at /workspace/zap/data/eval.
# Same wrapper as the LLaVA-1.5 fullcache run (lmms_eval_local_run.py) so all
# load_dataset calls are routed to load_from_disk.
set -euo pipefail

ENV_DIR="/workspace/VFlowOpt/.conda/VFlowOpt"
MODEL_DIR="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov"
WRAPPER="/workspace/VFlowOpt/scripts/lmms_eval_local_run.py"
OUT_ROOT="/workspace/VFlowOpt/logs/onevision_fullcache_$(date +%Y%m%d_%H%M%S)"
SUFFIX="onevision_qwen2_7b_fullcache"

export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/workspace/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT_ROOT"
echo "Logs: $OUT_ROOT"

# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

TASKS=(
  docvqa_val
  chartqa
  textvqa_val
)

for task in "${TASKS[@]}"; do
  task_log="$OUT_ROOT/${task}.log"
  echo "========== $task =========="
  echo "  log: $task_log"
  python "$WRAPPER" \
      --model llava_onevision \
      --model_args "pretrained=${MODEL_DIR},conv_template=qwen_1_5,model_name=llava_qwen,device_map=cuda:0" \
      --tasks "$task" \
      --batch_size 1 \
      --log_samples \
      --log_samples_suffix "$SUFFIX" \
      --output_path "$OUT_ROOT" \
        2>&1 | tee "$task_log" || true
  # lmms_eval logs "Error during evaluation" on internal failure but still
  # exits 0; detect that explicitly so the sweep doesn't silently succeed.
  if grep -q "Error during evaluation" "$task_log"; then
    echo "[fail] $task — see $task_log" | tee -a "$OUT_ROOT/FAILURES.log"
  else
    echo "[ok] $task"
  fi
done

echo "Done. Logs and lmms-eval results under $OUT_ROOT"
