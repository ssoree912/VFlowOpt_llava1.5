#!/usr/bin/env python
"""Download missing local evaluation datasets from Hugging Face.

The target layout matches the existing /workspace/zap/data/eval_hf snapshots.
Only evaluation/inference splits are selected; train shards are intentionally
excluded.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    repo_id: str
    local_name: str
    allow_patterns: tuple[str, ...]


COMMON = ("README.md", ".gitattributes")

DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        label="MME",
        repo_id="lmms-lab/MME",
        local_name="MME",
        allow_patterns=(*COMMON, "data/test-*.parquet"),
    ),
    DatasetSpec(
        label="MMBench",
        repo_id="lmms-lab/MMBench",
        local_name="MMBench",
        allow_patterns=(
            *COMMON,
            "en/dev-*.parquet",
            "en/test-*.parquet",
            "cn/dev-*.parquet",
            "cn/test-*.parquet",
            "cc/test-*.parquet",
        ),
    ),
    DatasetSpec(
        label="MMBench-ru",
        repo_id="deepvk/MMBench-ru",
        local_name="MMBench-ru",
        allow_patterns=(*COMMON, "mmbench_ru_dev.parquet"),
    ),
    DatasetSpec(
        label="POPE",
        repo_id="lmms-lab/POPE",
        local_name="POPE",
        allow_patterns=(
            *COMMON,
            "data/test-*.parquet",
            "Full/adversarial-*.parquet",
            "Full/popular-*.parquet",
            "Full/random-*.parquet",
        ),
    ),
    DatasetSpec(
        label="MMStar",
        repo_id="Lin-Chen/MMStar",
        local_name="MMStar",
        allow_patterns=(*COMMON, "MMStar.tsv", "mmstar.parquet"),
    ),
    DatasetSpec(
        label="VQAv2",
        repo_id="lmms-lab/VQAv2",
        local_name="VQAv2",
        allow_patterns=(*COMMON, "data/validation-*.parquet", "data/test-*.parquet"),
    ),
    DatasetSpec(
        label="VizWiz-VQA",
        repo_id="lmms-lab/VizWiz-VQA",
        local_name="VizWiz-VQA",
        allow_patterns=(*COMMON, "data/val-*.parquet", "data/test-*.parquet"),
    ),
)


def selected_size_gib(api: HfApi, spec: DatasetSpec) -> float | None:
    try:
        info = api.dataset_info(spec.repo_id, files_metadata=True)
    except Exception:
        return None

    total = 0
    import fnmatch

    for sibling in info.siblings:
        path = sibling.rfilename
        if any(fnmatch.fnmatch(path, pattern) for pattern in spec.allow_patterns):
            total += getattr(sibling, "size", None) or 0
    return total / 1024**3


def has_expected_files(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/zap/data/eval_hf")
    parser.add_argument("--dataset", action="append", choices=[d.label for d in DATASETS])
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    requested = set(args.dataset or [d.label for d in DATASETS])
    specs = [d for d in DATASETS if d.label in requested]
    api = HfApi()

    print(f"root={root}", flush=True)
    print("selected_datasets=" + ",".join(d.label for d in specs), flush=True)

    for index, spec in enumerate(specs, start=1):
        local_dir = root / spec.local_name
        if not args.force and has_expected_files(local_dir):
            print(f"[{index}/{len(specs)}] skip {spec.label}: already has parquet files at {local_dir}", flush=True)
            continue

        size = selected_size_gib(api, spec)
        if size is None:
            print(f"[{index}/{len(specs)}] start {spec.label}: {spec.repo_id} -> {local_dir}", flush=True)
        else:
            print(
                f"[{index}/{len(specs)}] start {spec.label}: {spec.repo_id} -> {local_dir} "
                f"(selected ~= {size:.2f} GiB)",
                flush=True,
            )
        print("allow_patterns=" + ",".join(spec.allow_patterns), flush=True)

        snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=list(spec.allow_patterns),
            max_workers=args.max_workers,
            token=os.environ.get("HF_TOKEN") or None,
        )
        files = [p for p in local_dir.rglob("*") if p.is_file()]
        total_bytes = sum(p.stat().st_size for p in files)
        print(
            f"[{index}/{len(specs)}] done {spec.label}: files={len(files)} "
            f"size={total_bytes / 1024**3:.2f} GiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
