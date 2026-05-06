#!/usr/bin/env bash
set -uo pipefail

cd /workspace/VFlowOpt

GPU_INDEX="${GPU_INDEX:-0}"
MODEL_PATH="${MODEL_PATH:-/workspace/zap/ckpts/llava-onevision-qwen2-7b-ov}"
KEEP_RATIO="0.10"
TASKS="${TASKS:-textvqa_val gqa_local docvqa_val_local chartqa_local coco2017_cap_val nocaps_val textcaps_val}"
LIMIT="${LIMIT:-}"
ILLAVA_LLM_K="${ILLAVA_LLM_K:-9-18}"
ILLAVA_VIT_K="${ILLAVA_VIT_K:-25}"
OUT_DIR="${OUT_DIR:-/workspace/VFlowOpt/experiments/EXP-20260506-001-onevision-vflowopt-keep10/outputs/keep10}"

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GQA_IMAGE_PARQUET="${GQA_IMAGE_PARQUET:-/workspace/zap/data/eval_hf/GQA/testdev_balanced_images/testdev-00000-of-00001.parquet}"
export PYTHONPATH="/workspace/VFlowOpt/src/lmms_eval-0.2.4:/workspace/VFlowOpt/src/LLaVA-OneVision:/workspace/VFlowOpt/src/transformers-4.46.0/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/opt/conda/envs/vflowopt_chartqa_eval/lib:${LD_LIBRARY_PATH:-}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
FAIL_LOG="${OUT_DIR}/failed_tasks_${RUN_ID}.csv"
mkdir -p "$OUT_DIR"
printf "task,exit_code,ended_at\n" > "$FAIL_LOG"

# keep_ratio=0.10: keep 10% of image tokens (text tokens are never pruned)
# VFlowOpt optimized config for this ratio is loaded automatically by the model
MODEL_ARGS="pretrained=${MODEL_PATH},conv_template=qwen_1_5,model_name=llava_qwen_training_free,device_map=auto,enable_illava_vit=True,illava_vit_k=${ILLAVA_VIT_K},enable_illava_llm=True,illava_llm_k=${ILLAVA_LLM_K},illava_keep_ratio=${KEEP_RATIO}"

for task in $TASKS; do
  task_out="${OUT_DIR}/${task}"
  if find "$task_out" -name "*_results.json" -print -quit 2>/dev/null | grep -q .; then
    echo "===== SKIP ${task} (exists) ====="
    continue
  fi
  mkdir -p "$task_out"
  marker="${task_out}/.start_${RUN_ID}_${task}"
  : > "$marker"
  echo "===== START keep_ratio=${KEEP_RATIO} task=${task} gpu=${GPU_INDEX} run_id=${RUN_ID} $(date -Is) ====="

  cmd=(/opt/conda/envs/vflowopt_chartqa_eval/bin/lmms-eval
    --model llava_onevision_training_free
    --model_args "$MODEL_ARGS"
    --tasks "$task"
    --batch_size 1
    --output_path "$task_out")
  if [ -n "$LIMIT" ]; then
    cmd+=(--limit "$LIMIT")
  fi

  if "${cmd[@]}"; then
    status=0
  else
    status=$?
  fi

  new_result="$(find "$task_out" -path '*/ckpts__*/*_results.json' -newer "$marker" -print -quit 2>/dev/null)"
  if [ "$status" -eq 0 ] && [ -n "$new_result" ]; then
    echo "===== DONE keep_ratio=${KEEP_RATIO} task=${task} result=${new_result} $(date -Is) ====="
  else
    [ "$status" -eq 0 ] && status="0_no_result_json"
    echo "===== FAILED keep_ratio=${KEEP_RATIO} task=${task} exit_code=${status} $(date -Is) ====="
    printf "%s,%s,%s\n" "$task" "$status" "$(date -Is)" >> "$FAIL_LOG"
  fi
done

echo "===== ALL DONE run_id=${RUN_ID} $(date -Is) ====="
echo "Failure log: ${FAIL_LOG}"
