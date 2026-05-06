"""Run lmms-eval against locally-saved datasets in /workspace/zap/data/eval.

We monkey-patch `datasets.load_dataset` so that when a known lmms-lab path is
requested (the original YAMLs use it as `dataset_path:`), we instead load the
matching `save_to_disk` directory under /workspace/zap/data/eval and return a
`DatasetDict` with the split name the YAML expects. This avoids any HF Hub
network access at eval time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import datasets as _ds
from datasets import DatasetDict, load_from_disk

LOCAL_ROOT = Path("/workspace/zap/data/eval")

# (path, name) -> (local_dir, split_name_in_yaml)
ROUTES: dict[tuple[str, str | None], tuple[Path, str]] = {
    ("lmms-lab/textvqa",          None):                              (LOCAL_ROOT / "textvqa_val",       "validation"),
    ("lmms-lab/GQA",              "testdev_balanced_instructions"):   (LOCAL_ROOT / "gqa/instructions",  "testdev"),
    ("lmms-lab/GQA",              "testdev_balanced_images"):         (LOCAL_ROOT / "gqa/images",        "testdev"),
    ("lmms-lab/DocVQA",           "DocVQA"):                          (LOCAL_ROOT / "docvqa_val",        "validation"),
    ("lmms-lab/ChartQA",          None):                              (LOCAL_ROOT / "chartqa",           "test"),
    ("lmms-lab/MME",              None):                              (LOCAL_ROOT / "mme",               "test"),
    ("lmms-lab/ScienceQA",        "ScienceQA-FULL"):                  (LOCAL_ROOT / "scienceqa",         "test"),
    ("lmms-lab/COCO-Caption2017", None):                              (LOCAL_ROOT / "coco2017_cap_val",  "val"),
    ("lmms-lab/NoCaps",           None):                              (LOCAL_ROOT / "nocaps_val",        "validation"),
    ("lmms-lab/TextCaps",         None):                              (LOCAL_ROOT / "textcaps_val",      "val"),
}

_orig_load_dataset = _ds.load_dataset


def _patched_load_dataset(path=None, name=None, *args, **kwargs):
    key = (path, name)
    if key in ROUTES:
        local_dir, split_name = ROUTES[key]
        ds = load_from_disk(str(local_dir))
        requested_split = kwargs.get("split", None)
        if requested_split is None:
            # YAML-driven case (lmms-eval/api/task.py:1022)
            print(f"[local-route] {path} (name={name}) -> {local_dir} as DatasetDict[{split_name}]", file=sys.stderr)
            return DatasetDict({split_name: ds})
        # Direct split request (e.g. gqa utils call) -> return Dataset
        print(f"[local-route] {path} (name={name}, split={requested_split}) -> {local_dir}", file=sys.stderr)
        return ds
    return _orig_load_dataset(path, name, *args, **kwargs)


_ds.load_dataset = _patched_load_dataset
# Also expose under load module so any `from datasets import load_dataset` after import works
import datasets.load as _ds_load  # noqa: E402

_ds_load.load_dataset = _patched_load_dataset


def _patch_siglip_loader_to_local_init() -> None:
    """OneVision's vendored SigLipVisionTower hard-codes a /mnt/petrelfs/...
    path that doesn't exist locally. The local checkpoint already contains the
    vision tower weights, so we initialize the module from config and rely on
    `from_pretrained` to fill it from the local shards.
    """
    try:
        from llava.model.multimodal_encoder import siglip_encoder  # type: ignore[import-not-found]
    except Exception:
        return  # LLaVA-1.5 path doesn't import OneVision modules

    def _load_model(self, device_map=None):  # noqa: ANN001, ARG001
        if self.is_loaded:
            return
        self.vision_tower = siglip_encoder.SigLipVisionModel(self.config)
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = siglip_encoder.nn.Identity()
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    siglip_encoder.SigLipVisionTower.load_model = _load_model


def main() -> int:
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/workspace/.cache/huggingface/datasets")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    LLAVA_ONEVISION_ROOT = Path("/workspace/VFlowOpt/src/LLaVA-OneVision")
    if LLAVA_ONEVISION_ROOT.exists() and str(LLAVA_ONEVISION_ROOT) not in sys.path:
        sys.path.insert(0, str(LLAVA_ONEVISION_ROOT))
    _patch_siglip_loader_to_local_init()
    from lmms_eval.__main__ import cli_evaluate  # noqa: WPS433

    cli_evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
