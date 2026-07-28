#!/usr/bin/env python3
"""Validate a CudaNav run directory without rerunning ROS or CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cudanav_evidence import evaluate_manifest, evaluate_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument(
        "--profile", choices=("smoke", "release"), default="smoke"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_directory = args.run_directory.resolve()
    manifest_path = run_directory / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    summary_relative = artifacts.get("summary", "mission_summary.json")
    summary_path = run_directory / summary_relative
    if not summary_path.is_file():
        raise SystemExit(f"missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = {
        "summary_gate": evaluate_summary(summary, args.profile),
        "manifest_gate": evaluate_manifest(
            manifest, run_directory, args.profile
        ),
    }
    result["artifact_binding"] = {
        "trajectory_matches_summary": (
            artifacts.get("trajectory") == summary.get("trajectory_csv")
        ),
        "traversal_count_matches": (
            manifest.get("traversal_count")
            == summary.get("traversals_requested")
        ),
    }
    result["passed"] = (
        result["summary_gate"]["passed"]
        and result["manifest_gate"]["passed"]
        and all(result["artifact_binding"].values())
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
