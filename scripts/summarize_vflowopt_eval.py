"""Print a short summary of metrics from an lmms-eval results.json directory."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(out_root: str) -> int:
    root = Path(out_root)
    if not root.exists():
        print(f"missing: {root}", file=sys.stderr)
        return 1
    for results in sorted(root.rglob("results.json")):
        try:
            data = json.loads(results.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {results}: {e}")
            continue
        for task, metrics in (data.get("results") or {}).items():
            kept = {k: v for k, v in metrics.items() if not k.endswith("_stderr") and k != "alias"}
            print(f"{task}: {kept}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python summarize_vflowopt_eval.py <log_dir>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
