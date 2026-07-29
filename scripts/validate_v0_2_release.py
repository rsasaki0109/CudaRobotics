#!/usr/bin/env python3
"""Evaluate the complete v0.2.0 release-candidate evidence set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v0_2_release_evidence import evaluate_release


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--cpu-preflight", type=Path, required=True)
    parser.add_argument("--gpu-preflight", type=Path, required=True)
    parser.add_argument("--build-ci", type=Path, required=True)
    parser.add_argument("--python-ci", type=Path, required=True)
    parser.add_argument("--ros-ci", type=Path, required=True)
    parser.add_argument("--python-artifacts", type=Path, required=True)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument(
        "--rosbag-report",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "mppi_real_rosbag_erl_prueba2_2026-07-28.md"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = evaluate_release(
            expected_commit=args.commit,
            cpu_preflight_dir=args.cpu_preflight.resolve(),
            gpu_preflight_dir=args.gpu_preflight.resolve(),
            build_ci_path=args.build_ci.resolve(),
            python_ci_path=args.python_ci.resolve(),
            ros_ci_path=args.ros_ci.resolve(),
            python_artifacts_path=args.python_artifacts.resolve(),
            dist_dir=args.dist_dir.resolve(),
            rosbag_report_path=args.rosbag_report.resolve(),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        result = {
            "schema_version": 1,
            "evidence_mode": "v0_2_release_gate",
            "status": "not_ready",
            "git_commit": args.commit,
            "passed": False,
            "error": str(error),
        }
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
