#!/usr/bin/env python3
"""Revalidate a retained CudaNav multi-GPU suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cudanav_multi_gpu import evaluate_multi_gpu_suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_directory", type=Path)
    args = parser.parse_args()
    root = args.suite_directory.resolve()
    manifest_path = root / "multi_gpu_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing suite manifest: {manifest_path}")
    suite = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = evaluate_multi_gpu_suite(suite, root)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
