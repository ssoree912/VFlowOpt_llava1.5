#!/usr/bin/env bash
set -uo pipefail

cd /workspace/VFlowOpt

export CUDA_VISIBLE_DEVICES=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export GQA_IMAGE_PARQUET=/workspace/zap/data/eval_hf/GQA/testdev_balanced_images/testdev-00000-of-00001.parquet

TASKS="mme_local,mmbench_en_dev_local,scienceqa_img_local,vizwiz_vqa_val_local,gqa_local,pope_local,vqav2_val_local,chartqa_local,docvqa_val_local,mmstar_local"
MODEL_ARGS="pretrained=/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen_training_free,device_map=auto,enable_illava_vit=True,illava_vit_k=25,enable_illava_llm=True,illava_llm_k=9-18,illava_keep_ratio=0.10"
OUT_DIR="/workspace/VFlowOpt/experiments/EXP-20260429-002/outputs/keep10_10bench"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
FAIL_LOG="$OUT_DIR/failed_tasks_${RUN_ID}.csv"

mkdir -p "$OUT_DIR"
printf "task,exit_code,ended_at\n" > "$FAIL_LOG"

IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"

for task in "${TASK_ARRAY[@]}"; do
  task_out="$OUT_DIR/$task"
  mkdir -p "$task_out"
  marker="$task_out/.start_${RUN_ID}_${task}"
  : > "$marker"
  echo "===== START task=$task run_id=$RUN_ID $(date -Is) ====="
  if /opt/conda/envs/vflowopt_chartqa_eval/bin/lmms-eval \
    --model llava_onevision_training_free \
    --model_args "$MODEL_ARGS" \
    --tasks "$task" \
    --batch_size 1 \
    --output_path "$task_out"; then
    status=0
  else
    status=$?
  fi
  new_result="$(find "$task_out" -path '*/ckpts__*/*_results.json' -newer "$marker" -print -quit)"
  if [ "$status" -eq 0 ] && [ -n "$new_result" ]; then
    echo "===== DONE task=$task run_id=$RUN_ID result=$new_result $(date -Is) ====="
  else
    if [ "$status" -eq 0 ]; then
      status="0_no_result_json"
    fi
    echo "===== FAILED task=$task run_id=$RUN_ID exit_code=$status $(date -Is) ====="
    printf "%s,%s,%s\n" "$task" "$status" "$(date -Is)" >> "$FAIL_LOG"
  fi
done

if [ "$(wc -l < "$FAIL_LOG")" -gt 1 ]; then
  exit 1
fi
