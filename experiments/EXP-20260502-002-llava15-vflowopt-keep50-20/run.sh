#!/usr/bin/env bash
set -uo pipefail

cd /workspace/VFlowOpt

GPU_INDEX="${GPU_INDEX:-1}"
KEEP_RATIOS="${KEEP_RATIOS:-0.50 0.20}"
TASKS="${TASKS:-textvqa_val gqa_local docvqa_val_local chartqa_local mme_local mmvet coco2017_cap_val nocaps_val textcaps_val}"
OUT_DIR="${OUT_DIR:-/workspace/VFlowOpt/experiments/EXP-20260502-002-llava15-vflowopt-keep50-20/outputs}"

GPU_INDEX="$GPU_INDEX" \
KEEP_RATIOS="$KEEP_RATIOS" \
TASKS="$TASKS" \
OUT_DIR="$OUT_DIR" \
bash experiments/EXP-20260502-001-llava15-vflowopt-total-ratio/run.sh
