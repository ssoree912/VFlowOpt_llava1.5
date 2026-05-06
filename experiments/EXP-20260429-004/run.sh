#!/usr/bin/env bash
set -uo pipefail

cd /workspace/VFlowOpt

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GQA_IMAGE_PARQUET=/workspace/zap/data/eval_hf/GQA/testdev_balanced_images/testdev-00000-of-00001.parquet

MODEL_BASE="pretrained=/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen_training_free,device_map=auto,enable_illava_vit=True,illava_vit_k=25,enable_illava_llm=True,illava_llm_k=9-18"
FULL_MODEL_ARGS="pretrained=/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,device_map=auto"
OUT_DIR="/workspace/VFlowOpt/experiments/EXP-20260429-004/outputs/resume_missing_gpu0"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
FAIL_LOG="$OUT_DIR/failed_tasks_${RUN_ID}.csv"

mkdir -p "$OUT_DIR"
printf "keep_ratio,task,exit_code,ended_at\n" > "$FAIL_LOG"

run_task() {
  local keep_ratio="$1"
  local task="$2"
  local ratio_label="${keep_ratio/./}"
  local task_out="$OUT_DIR/keep${ratio_label}/$task"
  local marker="$task_out/.start_${RUN_ID}_${task}"
  local model_args="${MODEL_BASE},illava_keep_ratio=${keep_ratio}"

  mkdir -p "$task_out"
  : > "$marker"

  echo "===== START keep_ratio=$keep_ratio task=$task run_id=$RUN_ID $(date -Is) ====="
  if /opt/conda/envs/vflowopt_chartqa_eval/bin/lmms-eval \
    --model llava_onevision_training_free \
    --model_args "$model_args" \
    --tasks "$task" \
    --batch_size 1 \
    --output_path "$task_out"; then
    status=0
  else
    status=$?
  fi

  new_result="$(find "$task_out" -path '*/ckpts__*/*_results.json' -newer "$marker" -print -quit)"
  if [ "$status" -eq 0 ] && [ -n "$new_result" ]; then
    echo "===== DONE keep_ratio=$keep_ratio task=$task run_id=$RUN_ID result=$new_result $(date -Is) ====="
  else
    if [ "$status" -eq 0 ]; then
      status="0_no_result_json"
    fi
    echo "===== FAILED keep_ratio=$keep_ratio task=$task run_id=$RUN_ID exit_code=$status $(date -Is) ====="
    printf "%s,%s,%s,%s\n" "$keep_ratio" "$task" "$status" "$(date -Is)" >> "$FAIL_LOG"
  fi
}

run_full_task() {
  local task="$1"
  local task_out="$OUT_DIR/full_cache/$task"
  local marker="$task_out/.start_${RUN_ID}_${task}"

  mkdir -p "$task_out"
  : > "$marker"

  echo "===== START keep_ratio=full task=$task run_id=$RUN_ID $(date -Is) ====="
  if /opt/conda/envs/vflowopt_chartqa_eval/bin/lmms-eval \
    --model llava_onevision \
    --model_args "$FULL_MODEL_ARGS" \
    --tasks "$task" \
    --batch_size 1 \
    --output_path "$task_out"; then
    status=0
  else
    status=$?
  fi

  new_result="$(find "$task_out" -path '*/ckpts__*/*_results.json' -newer "$marker" -print -quit)"
  if [ "$status" -eq 0 ] && [ -n "$new_result" ]; then
    echo "===== DONE keep_ratio=full task=$task run_id=$RUN_ID result=$new_result $(date -Is) ====="
  else
    if [ "$status" -eq 0 ]; then
      status="0_no_result_json"
    fi
    echo "===== FAILED keep_ratio=full task=$task run_id=$RUN_ID exit_code=$status $(date -Is) ====="
    printf "%s,%s,%s,%s\n" "full" "$task" "$status" "$(date -Is)" >> "$FAIL_LOG"
  fi
}

# keep 0.50 tasks that failed or produced no valid JSON in EXP-20260429-003.
for task in mme_local vizwiz_vqa_val_local chartqa_local docvqa_val_local mmstar_local; do
  run_task "0.50" "$task"
done

# keep 0.25 tasks blocked by VQAv2 in EXP-20260429-001.
for task in docvqa_val_local mmstar_local; do
  run_task "0.25" "$task"
done

# keep 0.10 tasks blocked by VQAv2 in EXP-20260429-002.
for task in docvqa_val_local mmstar_local; do
  run_task "0.10" "$task"
done

# full-cache baselines for the DocVQA/MMStar tasks being resumed above.
for task in docvqa_val_local mmstar_local; do
  run_full_task "$task"
done

if [ "$(wc -l < "$FAIL_LOG")" -gt 1 ]; then
  exit 1
fi
