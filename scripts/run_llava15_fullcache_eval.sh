#!/usr/bin/env bash
# Full-cache (no token pruning) LLaVA-1.5 7B eval over the 9 datasets stored
# locally under /workspace/zap/data/eval/. Uses scripts/lmms_eval_local_run.py
# which monkey-patches datasets.load_dataset to load from those local dirs
# (via load_from_disk) rather than the HF datasets cache or the network.
set -euo pipefail

ENV_DIR="/workspace/VFlowOpt/.conda/VFlowOpt"
MODEL_DIR="/workspace/zap/ckpts/llava-v1.5-7b"
WRAPPER="/workspace/VFlowOpt/scripts/lmms_eval_local_run.py"
OUT_ROOT="/workspace/VFlowOpt/logs/llava15_fullcache_$(date +%Y%m%d_%H%M%S)"
SUFFIX="llava15_7b_fullcache"

# CXXABI_1.3.15 from conda libstdc++ (system one is too old).
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
# Datasets are local; CLIP vision tower (openai/clip-vit-large-patch14-336)
# still resolves through HF the first time.
export HF_HOME=/workspace/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
# RTX 4090 P2P quirk
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=1

mkdir -p "$OUT_ROOT"
echo "Logs: $OUT_ROOT"

# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

TASKS=(
  textvqa_val
  gqa
  docvqa_val
  chartqa
  mme
  scienceqa
  coco2017_cap_val
  nocaps_val
  textcaps_val
)

for task in "${TASKS[@]}"; do
  task_log="$OUT_ROOT/${task}.log"
  echo "========== $task =========="
  echo "  log: $task_log"
  if python "$WRAPPER" \
      --model llava \
      --model_args "pretrained=${MODEL_DIR},conv_template=vicuna_v1,model_name=llava-v1.5-7b,device_map=cuda:0" \
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
