#!/usr/bin/env bash
# Sweep VFlowOpt eviction on OneVision-Qwen2-7B-OV across the 10 MileBench
# datasets with contiguous-prefix image placeholders (T-2/T-3/T-4):
#   ObjectExistence, ObjectInteraction, MovingAttribute, ObjectShuffle,
#   EgocentricNavigation, MovingDirection,
#   CounterfactualInference, StateChange, CharacterOrder, SceneTransition.
#
# For each dataset runs: fullcache + keep_ratio={0.2, 0.5, 0.8} (4 settings).
# After each run, scores with look-m's evaluate.py and aggregates results
# into a CSV.
#
# Configuration via env:
#   GPU                — CUDA_VISIBLE_DEVICES (default 0)
#   OUT_ROOT           — output dir (default logs/onevision_milebench_<ts>)
#   DATASETS           — space-separated list (default all 10)
#   KEEP_RATIOS        — space-separated (default "0.2 0.5 0.8")
#   SKIP_FULLCACHE=1   — skip the fullcache pass
#   EVAL_SAMPLES=N     — cap samples per dataset (smoke tests)
#   OVERWRITE=1        — re-run even if pred.json exists
set -euo pipefail

ROOT=/workspace/VFlowOpt
ENV=${ENV:-/workspace/VFlowOpt/.conda/VFlowOpt}
PY="$ENV/bin/python"
LOOKM_ROOT=/workspace/look-m
LOOKM_PY="${LOOKM_ROOT}/.conda/look-m/bin/python"
SCRIPT="$ROOT/scripts/run_onevision_milebench_vflowopt.py"
DATA_DIR=${DATA_DIR:-/workspace/zap/data/MileBench}

GPU=${GPU:-0}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-$ROOT/logs/onevision_milebench_$TS}
mkdir -p "$OUT_ROOT"
echo "OUT_ROOT=$OUT_ROOT"

DEFAULT_DATASETS=(
  ObjectExistence ObjectInteraction MovingAttribute ObjectShuffle
  EgocentricNavigation MovingDirection
  CounterfactualInference StateChange CharacterOrder SceneTransition
)
read -r -a DATASETS <<<"${DATASETS:-${DEFAULT_DATASETS[@]}}"
read -r -a KEEP_RATIOS <<<"${KEEP_RATIOS:-0.2 0.5 0.8}"

extra_eval=()
if [[ -n "${EVAL_SAMPLES:-}" ]]; then
  extra_eval+=(--eval-samples "$EVAL_SAMPLES")
fi
extra_run=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  extra_run+=(--overwrite)
fi
if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
  extra_run+=(--max-new-tokens "$MAX_NEW_TOKENS")
fi

run_one() {
  local mode=$1 keep=$2
  local subdir
  if [[ "$mode" == "fullcache" ]]; then
    subdir="$OUT_ROOT/fullcache"
  else
    subdir="$OUT_ROOT/keep${keep}"
  fi
  mkdir -p "$subdir"
  local extra=()
  if [[ "$mode" == "pruned" ]]; then
    extra+=(--mode pruned --keep-ratio "$keep")
  else
    extra+=(--mode fullcache)
  fi

  local logf="$subdir/run.log"
  echo "===== $mode keep=$keep -> $subdir ====="
  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "$SCRIPT" \
    --model-path /workspace/zap/ckpts/llava-onevision-qwen2-7b-ov \
    --data-dir "$DATA_DIR" \
    --datasets "${DATASETS[@]}" \
    --output-dir "$subdir" \
    --device-map cuda:0 --device cuda:0 \
    "${extra[@]}" "${extra_run[@]}" "${extra_eval[@]}" \
    2>&1 | tee "$logf"
}

score_one() {
  local subdir=$1
  for ds in "${DATASETS[@]}"; do
    local pred="$subdir/$ds/pred.json"
    if [[ ! -f "$pred" ]]; then
      echo "[skip-score] $subdir/$ds: no pred.json"
      continue
    fi
    echo "----- score $subdir/$ds -----"
    (cd "$LOOKM_ROOT" && "$LOOKM_PY" evaluate.py \
        --data-dir "$DATA_DIR" \
        --dataset "$ds" \
        --result-dir "$subdir") \
      2>&1 | tee -a "$subdir/score.log"
  done
}

# 1. Generations
if [[ "${SKIP_FULLCACHE:-0}" != "1" ]]; then
  run_one fullcache 0.0
fi
for keep in "${KEEP_RATIOS[@]}"; do
  run_one pruned "$keep"
done

# 2. Scoring
if [[ "${SKIP_FULLCACHE:-0}" != "1" ]]; then
  score_one "$OUT_ROOT/fullcache"
fi
for keep in "${KEEP_RATIOS[@]}"; do
  score_one "$OUT_ROOT/keep${keep}"
done

# 3. Summary CSV
SUMMARY="$OUT_ROOT/summary.csv"
"$PY" - "$OUT_ROOT" "$SUMMARY" "${DATASETS[@]}" <<'PY'
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
datasets = sys.argv[3:]

settings = []
fc = out_root / "fullcache"
if fc.exists():
    settings.append(("fullcache", fc))
for sub in sorted(out_root.glob("keep*")):
    settings.append((sub.name, sub))

rows = [["setting", "dataset", "metric", "value", "n_samples", "n_failures", "sec_per_sample"]]
for name, subdir in settings:
    for ds in datasets:
        eval_p = subdir / ds / "eval.json"
        meta_p = subdir / ds / "run_meta.json"
        if not eval_p.exists():
            rows.append([name, ds, "?", "", "", "", ""])
            continue
        ev = json.loads(eval_p.read_text())
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        # multi-choice → 'Accuracy', open-ended → 'Rouge-L f', needle → 'Accuracy'
        if "Accuracy" in ev:
            metric, val = "Accuracy", ev["Accuracy"]
        elif "Rouge-L f" in ev:
            metric, val = "Rouge-L f", ev["Rouge-L f"]
        else:
            metric, val = next(iter(ev.items()))
        rows.append([
            name, ds, metric, f"{val:.4f}" if isinstance(val, float) else str(val),
            str(meta.get("n_samples", "")),
            str(meta.get("n_failures", "")),
            f"{meta.get('sec_per_sample', 0):.2f}" if meta.get("sec_per_sample") else "",
        ])

with summary_path.open("w") as f:
    for row in rows:
        f.write(",".join(row) + "\n")
print(f"[summary] wrote {summary_path}")
print("\n".join(",".join(r) for r in rows))
PY

echo "ALL DONE -> $OUT_ROOT"
echo "Summary: $OUT_ROOT/summary.csv"
