#!/usr/bin/env python3
"""Validate a CudaNav real-rosbag evidence directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cudanav_rosbag_evidence import evaluate_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--profile", choices=("smoke", "release"))
    parser.add_argument(
        "--no-verify-source",
        action="store_true",
        help="Skip re-reading the external input bag (for archived evidence only).",
    )
    args = parser.parse_args()
    root = args.run_directory.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    profile = args.profile or manifest.get("profile")
    result = evaluate_manifest(
        manifest,
        root,
        profile,
        verify_source=not args.no_verify_source,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
