"""Download eval-only splits for VFlowOpt benchmarks into ../zap/data/eval.

Each dataset is materialised with `datasets.load_dataset(..., split=...)` and
written via `save_to_disk` so it can later be reloaded with `load_from_disk`
(or pointed at via `dataset_path:` in a task YAML).
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/workspace/.cache/huggingface/datasets")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from datasets import load_dataset  # noqa: E402

OUT_ROOT = Path("/workspace/zap/data/eval")

# (out_subdir, hf_path, hf_config or None, split)
JOBS = [
    ("textvqa_val",       "lmms-lab/textvqa",          None,                              "validation"),
    ("gqa/instructions",  "lmms-lab/GQA",              "testdev_balanced_instructions",   "testdev"),
    ("gqa/images",        "lmms-lab/GQA",              "testdev_balanced_images",         "testdev"),
    ("docvqa_val",        "lmms-lab/DocVQA",           "DocVQA",                          "validation"),
    ("chartqa",           "lmms-lab/ChartQA",          None,                              "test"),
    ("mme",               "lmms-lab/MME",              None,                              "test"),
    ("scienceqa",         "lmms-lab/ScienceQA",        "ScienceQA-FULL",                  "test"),
    ("coco2017_cap_val",  "lmms-lab/COCO-Caption2017", None,                              "val"),
    ("nocaps_val",        "lmms-lab/NoCaps",           None,                              "validation"),
    ("textcaps_val",      "lmms-lab/TextCaps",         None,                              "val"),
]


def run_job(subdir: str, repo: str, config: str | None, split: str, only: set[str] | None) -> bool:
    out_path = OUT_ROOT / subdir
    if only and subdir not in only and subdir.split("/")[0] not in only:
        return True
    if (out_path / "dataset_info.json").exists():
        print(f"[skip] {subdir} already exists at {out_path}")
        return True
    print(f"[load] {repo} (config={config}, split={split}) -> {out_path}")
    try:
        ds = load_dataset(repo, name=config, split=split, token=True)
        out_path.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(out_path))
        print(f"[done] {subdir}: {len(ds)} rows")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {subdir}: {exc}")
        traceback.print_exc()
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these out_subdir names (top level ok, e.g. 'gqa')")
    args = parser.parse_args()
    only = set(args.only) if args.only else None

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    failures = []
    for subdir, repo, config, split in JOBS:
        ok = run_job(subdir, repo, config, split, only)
        if not ok:
            failures.append(subdir)
    if failures:
        print(f"\n[summary] {len(failures)} failures: {failures}")
        return 1
    print("\n[summary] all datasets downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
