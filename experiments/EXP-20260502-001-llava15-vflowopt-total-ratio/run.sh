#!/usr/bin/env bash
set -uo pipefail

cd /workspace/VFlowOpt

GPU_INDEX="${GPU_INDEX:-1}"
MODEL_PATH="${MODEL_PATH:-/workspace/look-m/models/llava-v1.5-7b}"
KEEP_RATIOS="${KEEP_RATIOS:-0.30 0.10}"
TASKS="${TASKS:-coco2014_cap_val coco2017_cap_val flickr30k_test nocaps_val textcaps_val mmvet textvqa_val scienceqa_img_local gqa_local docvqa_val_local chartqa_local mme_local}"
LIMIT="${LIMIT:-}"
ILLAVA_LLM_K="${ILLAVA_LLM_K:-9-18}"
ILLAVA_VIT_K="${ILLAVA_VIT_K:-23}"
OUT_DIR="${OUT_DIR:-/workspace/VFlowOpt/experiments/EXP-20260502-001-llava15-vflowopt-total-ratio/outputs}"

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GQA_IMAGE_PARQUET="${GQA_IMAGE_PARQUET:-/workspace/zap/data/eval/GQA/testdev_balanced_images/testdev-00000-of-00001.parquet}"
export PYTHONPATH="/workspace/VFlowOpt/src/lmms_eval-0.2.4:/workspace/VFlowOpt/src/LLaVA-OneVision:/workspace/VFlowOpt/src/transformers-4.46.0/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/opt/conda/envs/vflowopt_chartqa_eval/lib:${LD_LIBRARY_PATH:-}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
FAIL_LOG="${OUT_DIR}/failed_tasks_${RUN_ID}.csv"
mkdir -p "$OUT_DIR"
printf "keep_ratio,task,exit_code,ended_at\n" > "$FAIL_LOG"

run_task() {
  local keep_ratio="$1"
  local task="$2"
  local ratio_label="${keep_ratio/./}"
  local task_out="${OUT_DIR}/keep${ratio_label}/${task}"
  local marker="${task_out}/.start_${RUN_ID}_${task}"
  local model_args

  if [ "$task" = "mmvet" ] && [ -z "${OPENAI_API_KEY:-}" ] && [ "${RUN_MMVET:-0}" != "1" ]; then
    echo "===== SKIP keep_ratio=${keep_ratio} task=mmvet: OPENAI_API_KEY is not set ====="
    return 0
  fi

  model_args="pretrained=${MODEL_PATH},conv_template=vicuna_v1,model_name=llava_llama_training_free,device_map=cuda:0,enable_illava_vit=True,illava_vit_k=${ILLAVA_VIT_K},enable_illava_llm=True,illava_llm_k=${ILLAVA_LLM_K},illava_total_keep_ratio=True,illava_keep_ratio=${keep_ratio}"

  mkdir -p "$task_out"
  : > "$marker"

  echo "===== START keep_ratio=${keep_ratio} task=${task} gpu=${GPU_INDEX} run_id=${RUN_ID} $(date -Is) ====="
  cmd=(/opt/conda/envs/vflowopt_chartqa_eval/bin/lmms-eval
    --model llava_llama_training_free
    --model_args "$model_args"
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

  new_result="$(find "$task_out" -path '*/*_results.json' -newer "$marker" -print -quit 2>/dev/null)"
  if [ "$status" -eq 0 ] && [ -n "$new_result" ]; then
    echo "===== DONE keep_ratio=${keep_ratio} task=${task} result=${new_result} $(date -Is) ====="
  else
    if [ "$status" -eq 0 ]; then
      status="0_no_result_json"
    fi
    echo "===== FAILED keep_ratio=${keep_ratio} task=${task} exit_code=${status} $(date -Is) ====="
    printf "%s,%s,%s,%s\n" "$keep_ratio" "$task" "$status" "$(date -Is)" >> "$FAIL_LOG"
  fi
}

for keep_ratio in $KEEP_RATIOS; do
  for task in $TASKS; do
    run_task "$keep_ratio" "$task"
  done
done

echo "===== ALL DONE run_id=${RUN_ID} $(date -Is) ====="
echo "Failure log: ${FAIL_LOG}"
