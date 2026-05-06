#!/usr/bin/env bash
# GPU 2: student_onevision_A_ep20, keep_ratio=0.25 then 0.10, all 10 benchmarks (skip if done)
set -uo pipefail

cd /workspace/VFlowOpt

export CUDA_VISIBLE_DEVICES=2
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GQA_IMAGE_PARQUET=/workspace/zap/data/eval_hf/GQA/testdev_balanced_images/testdev-00000-of-00001.parquet

STUDENT_PATH="/workspace/zap/ckpts/student_onevision_A_ep20"
PRETRAINED="/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov-hf"
EXP_ROOT="/workspace/VFlowOpt/experiments/EXP-20260429-005/outputs/student_ep20"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

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
  local keep_ratio="$1"
  local task="$2"
  local ratio_label="${keep_ratio/./}"
  local out_dir="${EXP_ROOT}/keep${ratio_label}"
  local task_out="${out_dir}/${task}"
  local fail_log="${out_dir}/failed_tasks_${RUN_ID}.csv"
  local model_args="pretrained=${PRETRAINED},student_path=${STUDENT_PATH},keep_ratio=${keep_ratio},device=cuda:0"

  # init fail log header once
  if [ ! -f "$fail_log" ]; then
    mkdir -p "$out_dir"
    printf "task,exit_code,ended_at\n" > "$fail_log"
  fi

  # skip if result already exists
  if find "$task_out" -name "*_results.json" -quit 2>/dev/null | grep -q .; then
    echo "===== SKIP (exists) keep_ratio=${keep_ratio} task=${task} ====="
    return 0
  fi

  mkdir -p "$task_out"
  local marker="${task_out}/.start_${RUN_ID}"
  : > "$marker"

  echo "===== START keep_ratio=${keep_ratio} task=${task} run_id=${RUN_ID} $(date -Is) ====="
  if /opt/conda/envs/vflowopt_chartqa_eval/bin/lmms-eval \
    --model llava_onevision_student \
    --model_args "$model_args" \
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
    echo "===== DONE keep_ratio=${keep_ratio} task=${task} result=${new_result} $(date -Is) ====="
  else
    [ "$status" -eq 0 ] && status="0_no_result_json"
    echo "===== FAILED keep_ratio=${keep_ratio} task=${task} exit_code=${status} $(date -Is) ====="
    printf "%s,%s,%s\n" "$task" "$status" "$(date -Is)" >> "$fail_log"
  fi
}

echo "===== Starting keep_ratio=0.25 $(date -Is) ====="
for task in "${TASKS[@]}"; do
  run_task "0.25" "$task"
done

echo "===== Starting keep_ratio=0.10 $(date -Is) ====="
for task in "${TASKS[@]}"; do
  run_task "0.10" "$task"
done

echo "===== ALL DONE gpu2 $(date -Is) ====="
