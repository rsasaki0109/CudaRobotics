#!/usr/bin/env python3
"""Run and freeze the native all-GPU CudaNav S-course closed loop."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = (
    ROOT / "bin" / "Release" / "cudanav_gpu_closed_loop_s_course.exe"
    if os.name == "nt"
    else ROOT / "bin" / "cudanav_gpu_closed_loop_s_course"
)
STAGES = [
    "gpu_kiss_icp", "gpu_voxel_mapping", "gpu_esdf",
    "cuda_mppi", "command_driven_plant",
]
CLAIMS = {
    "native_gpu_core_closed_loop": True,
    "ros2_runtime": False,
    "real_data": False,
}
SCENARIO = {
    "outer_bounds": [-1.0, -2.5, 10.0, 2.5],
    "obstacles": [[3.6, -2.5, 4.3, 0.55], [6.1, -0.55, 6.8, 2.5]],
    "waypoints": [
        [0.0, 0.0], [2.7, 0.0], [3.15, 1.15], [4.75, 1.15],
        [5.55, -1.15], [7.25, -1.15], [9.0, 0.0],
    ],
    "robot_radius_m": 0.24,
    "ray_count": 240,
    "z_levels_m": [-0.45, 0.0, 0.45],
    "control_period_s": 0.1,
}
THRESHOLDS = {
    "maximum_goal_distance_m": 0.30,
    "maximum_collision_count": 0,
    "maximum_odometry_drift_percent": 5.0,
    "maximum_command_deadline_miss_rate": 0.05,
    "minimum_command_effect_distance_m": 5.0,
    "minimum_inliers": 30,
    "minimum_observed_voxels": 500,
    "minimum_occupied_cells": 10,
}
CONTRACT_SOURCES = [
    "CMakeLists.txt",
    "include/cuda_mppi_controller/mppi_gpu.hpp",
    "include/cudarobotics/esdf_2d_gpu.hpp",
    "include/cudarobotics/kiss_icp_gpu.hpp",
    "include/cudarobotics/voxel_mapping_gpu.hpp",
    "src/esdf_2d_gpu.cu",
    "src/gpu_kiss_icp.cu",
    "src/mppi_gpu.cu",
    "src/voxel_mapping_gpu.cu",
    "tools/cudanav_gpu_closed_loop_s_course.cu",
    "scripts/run_cudanav_gpu_closed_loop.py",
    "ros2_ws/src/cuda_nav_bringup/cuda_nav_bringup/simulation_geometry.py",
    "ros2_ws/src/cuda_nav_bringup/cuda_nav_bringup/loopback_simulator.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def source_digest() -> str:
    digest = hashlib.sha256()
    for relative in CONTRACT_SOURCES:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def evaluate_result(result: dict[str, Any]) -> dict[str, bool]:
    return {
        "scenario": result.get("scenario") == "cudanav_s_course",
        "stages": result.get("stages") == STAGES,
        "claims": result.get("claims") == CLAIMS,
        "goal_reached": result.get("goal_reached") is True,
        "goal_distance": result.get("ground_truth_goal_distance_m", 1e9)
        <= THRESHOLDS["maximum_goal_distance_m"],
        "collision_free": result.get("collision_count", 1e9)
        <= THRESHOLDS["maximum_collision_count"],
        "odometry_drift": result.get("odometry_drift_percent", 1e9)
        < THRESHOLDS["maximum_odometry_drift_percent"],
        "deadline": result.get("command_deadline_miss_rate", 1.0)
        < THRESHOLDS["maximum_command_deadline_miss_rate"],
        "causal_command_effect": (
            result.get("causal_command_effect") is True
            and result.get("command_effect_distance_m", 0.0)
            >= THRESHOLDS["minimum_command_effect_distance_m"]
        ),
        "finite_commands": result.get("invalid_commands", 1) == 0,
        "odometry_inliers": result.get("minimum_inliers", 0)
        >= THRESHOLDS["minimum_inliers"],
        "voxel_mapping": result.get("final_observed_voxels", 0)
        >= THRESHOLDS["minimum_observed_voxels"],
        "occupied_cells": result.get("maximum_occupied_cells", 0)
        >= THRESHOLDS["minimum_occupied_cells"],
        "gpu_identity": bool(result.get("gpu", {}).get("name"))
        and result.get("gpu", {}).get("driver_version", 0) > 0,
        "native_quality_gate": result.get("quality_pass") is True,
    }


def trajectory_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def portable_evidence(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "result_id": "cudanav_gpu_closed_loop_s_course_2026-07-29",
        "source_commit": manifest["source_commit"],
        "source_digest": manifest["source_digest"],
        "scenario": SCENARIO,
        "stages": STAGES,
        "claims": CLAIMS,
        "thresholds": THRESHOLDS,
        "metrics": manifest["result"],
        "checks": manifest["checks"],
        "artifacts": manifest["artifacts"],
        "scope_note": (
            "Native deterministic S-course simulation. CUDA MPPI commands are "
            "applied to the plant and affect later LiDAR scans. This is not a "
            "ROS 2 runtime result and does not use recorded real-world data."
        ),
    }


def render_markdown(evidence: dict[str, Any]) -> str:
    metrics = evidence["metrics"]
    passed = all(evidence["checks"].values())
    return f"""# CudaNav native all-GPU closed-loop S-course

Date: 2026-07-29

Source commit: `{evidence["source_commit"]}`

Result: **{"PASS" if passed else "FAIL"}**

The deterministic S-course plant generates LiDAR from ground truth, but the
controller only receives the GPU KISS-ICP estimate. Each CUDA MPPI command is
applied to the plant before the next scan. GPU voxel mapping and GPU ESDF build
the controller costmap in the same process.

## Result

- Goal reached: {str(metrics["goal_reached"]).lower()}
- Final ground-truth goal distance: {metrics["ground_truth_goal_distance_m"]:.3f} m
- Collision count: {metrics["collision_count"]}
- Ground-truth distance: {metrics["ground_truth_distance_m"]:.3f} m
- Command-effect distance: {metrics["command_effect_distance_m"]:.3f} m
- KISS-ICP ATE RMSE: {metrics["odometry_ate_rmse_m"]:.3f} m
- KISS-ICP final drift: {metrics["odometry_drift_percent"]:.3f}%
- Minimum ICP inliers: {metrics["minimum_inliers"]}
- Final observed voxels: {metrics["final_observed_voxels"]}
- Peak occupied 2D cells: {metrics["maximum_occupied_cells"]}
- MPPI solve p95: {metrics["mppi_solve_ms_p95"]:.3f} ms
- Full frame p95: {metrics["frame_ms_p95"]:.3f} ms
- Command deadline miss rate: {metrics["command_deadline_miss_rate"]:.3%}
- All-colliding evaluations: {metrics["all_colliding_evaluations"]}
- Minimum nonzero valid-rollout ratio: {metrics["minimum_nonzero_valid_rollout_ratio"]:.3f}

## Scope

{evidence["scope_note"]}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--maximum-steps", type=int, default=1800)
    parser.add_argument("--publish-json", type=Path)
    parser.add_argument("--publish-markdown", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runner = args.runner.resolve()
    if not runner.is_file():
        raise SystemExit(f"runner does not exist: {runner}")
    result_path = output / "result.json"
    trajectory_path = output / "trajectory.csv"
    command = [
        str(runner), "--json", str(result_path), "--csv", str(trajectory_path),
        "--maximum-steps", str(args.maximum_steps), "--check",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    log_path = output / "runner.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise SystemExit(
            f"closed-loop runner failed ({completed.returncode}); see {log_path}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checks = evaluate_result(result)
    checks["trajectory_rows"] = trajectory_rows(trajectory_path) == result["frames"]
    checks["source_commit"] = len(git_commit()) == 40
    artifacts = {
        name: {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in {
            "runner": runner, "result": result_path,
            "trajectory": trajectory_path, "runner_log": log_path,
        }.items()
    }
    manifest = {
        "schema_version": 1,
        "source_commit": git_commit(),
        "source_digest": source_digest(),
        "scenario": SCENARIO,
        "thresholds": THRESHOLDS,
        "result": result,
        "checks": checks,
        "artifacts": artifacts,
        "command": command,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence = portable_evidence(manifest)
    if args.publish_json:
        args.publish_json.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.publish_markdown:
        args.publish_markdown.write_text(render_markdown(evidence), encoding="utf-8")
    if not all(checks.values()):
        failed = ", ".join(key for key, value in checks.items() if not value)
        raise SystemExit(f"evidence checks failed: {failed}")
    print(
        f"PASS: {result['frames']} frames, "
        f"{result['ground_truth_goal_distance_m']:.3f} m goal distance"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
