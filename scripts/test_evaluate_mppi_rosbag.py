#!/usr/bin/env python3
"""CPU-only checks for the real-rosbag MPPI report and quality gates."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("evaluate_mppi_rosbag.py")
SPEC = importlib.util.spec_from_file_location("evaluate_mppi_rosbag", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> int:
    motion = {
        "duration_s": 12.0,
        "path_length_m": 4.0,
        "mean_speed_mps": 0.33,
        "command_samples": 240,
        "odometry_samples": 120,
    }
    clearance = {
        "command_pair_ratio": 0.98,
        "minimum_front_range_m": 0.31,
        "mean_front_clearance_m": 1.4,
        "front_below_0_5m_ratio": 0.02,
    }
    diagnostics = {
        "solve_mean_ms": 8.2,
        "solve_p95_ms": 12.4,
        "valid_rollout_ratio_mean": 0.82,
        "all_colliding_cycles": 0,
        "retreat_cycles": 1,
    }
    report = MODULE.build_report(
        motion,
        clearance,
        diagnostics,
        minimum_clearance_m=0.10,
        maximum_solve_p95_ms=50.0,
        minimum_valid_ratio=0.50,
    )
    assert report["quality_pass"]
    assert report["evidence_mode"] == "shadow_controller_with_recorded_motion"
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        MODULE.write_report(report, output)
        assert (output / "evaluation.json").exists()
        assert "Overall quality gate: **PASS**" in (
            output / "evaluation.md"
        ).read_text(encoding="utf-8")
    diagnostics["solve_p95_ms"] = 75.0
    failed = MODULE.build_report(
        motion,
        clearance,
        diagnostics,
        minimum_clearance_m=0.10,
        maximum_solve_p95_ms=50.0,
        minimum_valid_ratio=0.50,
    )
    assert not failed["quality_pass"]
    assert not failed["checks"]["solve_p95_budget"]
    print("real-rosbag MPPI evaluation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
