#!/usr/bin/env bash
# GPU 0: student_onevision_A_ep20, keep_ratio=0.50, all 10 benchmarks (skip if done)
set -uo pipefail

cd /workspace/VFlowOpt

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GQA_IMAGE_PARQUET=/workspace/zap/data/eval_hf/GQA/testdev_balanced_images/testdev-00000-of-00001.parquet

KEEP_RATIO="0.50"
STUDENT_PATH="/workspace/zap/ckpts/student_onevision_A_ep20"
PRETRAINED="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov-hf"
MODEL_ARGS="pretrained=${PRETRAINED},student_path=${STUDENT_PATH},keep_ratio=${KEEP_RATIO},device=cuda:0"
OUT_DIR="/workspace/VFlowOpt/experiments/EXP-20260429-005/outputs/student_ep20/keep050"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
FAIL_LOG="${OUT_DIR}/failed_tasks_${RUN_ID}.csv"

mkdir -p "$OUT_DIR"
printf "task,exit_code,ended_at\n" > "$FAIL_LOG"

TASKS=(
  mme_local
  mmbench_en_dev_local
  scienceqa_img_local
  vizwiz_vqa_val_local
  gqa_local
  pope_local
  textvqa_val
  chartqa_local
  docvqa_val_local
  mmstar_local
)

run_task() {
  local task="$1"
  local task_out="${OUT_DIR}/${task}"

  # skip if result already exists
  if find "$task_out" -name "*_results.json" -quit 2>/dev/null | grep -q .; then
    echo "===== SKIP (exists) keep_ratio=${KEEP_RATIO} task=${task} ====="
    return 0
  fi

  mkdir -p "$task_out"
  local marker="${task_out}/.start_${RUN_ID}"
  : > "$marker"

  echo "===== START keep_ratio=${KEEP_RATIO} task=${task} run_id=${RUN_ID} $(date -Is) ====="
  if /opt/conda/envs/vflowopt_chartqa_eval/bin/lmms-eval \
    --model llava_onevision_student \
    --model_args "$MODEL_ARGS" \
    --tasks "$task" \
    --batch_size 1 \
    --output_path "$task_out"; then
    status=0
  else
    status=$?
  fi

  local new_result
  new_result="$(find "$task_out" -path '*/ckpts__*/*_results.json' -newer "$marker" -print -quit 2>/dev/null)"
  if [ "$status" -eq 0 ] && [ -n "$new_result" ]; then
    echo "===== DONE keep_ratio=${KEEP_RATIO} task=${task} result=${new_result} $(date -Is) ====="
  else
    [ "$status" -eq 0 ] && status="0_no_result_json"
    echo "===== FAILED keep_ratio=${KEEP_RATIO} task=${task} exit_code=${status} $(date -Is) ====="
    printf "%s,%s,%s\n" "$task" "$status" "$(date -Is)" >> "$FAIL_LOG"
  fi
}

for task in "${TASKS[@]}"; do
  run_task "$task"
done

if [ "$(wc -l < "$FAIL_LOG")" -gt 1 ]; then
  echo "Some tasks failed. See $FAIL_LOG"
  exit 1
fi
echo "===== ALL DONE gpu0 keep_ratio=${KEEP_RATIO} $(date -Is) ====="
