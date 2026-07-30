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
        [0.0, 0.0], [2.7, 0.0], [2.9, 1.4], [5.05, 1.4],
        [5.35, -1.4], [7.5, -1.4], [9.0, 0.0],
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
    "maximum_all_colliding_evaluations": 3,
    "minimum_nonzero_valid_rollout_ratio": 0.001,
    "maximum_ground_truth_distance_m": 13.0,
    "maximum_frames": 400,
}
RELEASE_THRESHOLDS = {
    **THRESHOLDS,
    "maximum_odometry_drift_percent": 1.0,
    "maximum_command_deadline_miss_rate": 0.01,
    "minimum_simulated_duration_s": 600.0,
    "traversals": 30,
    "maximum_all_colliding_evaluations": 90,
    "maximum_ground_truth_distance_m": 390.0,
    "maximum_frames": 12000,
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


def active_gpu_identity() -> dict[str, str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=15.0,
    )
    if completed.returncode:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
    devices = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            devices.append(
                {
                    "physical_index": fields[0],
                    "name": fields[1],
                    "uuid": fields[2],
                    "driver_version": fields[3],
                }
            )
    if not devices:
        raise RuntimeError("nvidia-smi returned no physical GPUs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    selector = visible.split(",", 1)[0].strip() if visible else "0"
    selected = next(
        (
            device
            for device in devices
            if selector in {device["physical_index"], device["uuid"]}
        ),
        None,
    )
    if selected is None:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES selector {selector!r} has no nvidia-smi match"
        )
    return selected


def source_digest() -> str:
    digest = hashlib.sha256()
    for relative in CONTRACT_SOURCES:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def evaluate_result(
    result: dict[str, Any],
    profile: str = "smoke",
    expected_traversals: int | None = None,
) -> dict[str, bool]:
    thresholds = dict(
        RELEASE_THRESHOLDS if profile == "release" else THRESHOLDS
    )
    if expected_traversals is None:
        expected_traversals = int(thresholds.get("traversals", 1))
    if profile == "smoke" and expected_traversals != 1:
        thresholds["maximum_all_colliding_evaluations"] = (
            THRESHOLDS["maximum_all_colliding_evaluations"]
            * expected_traversals
        )
        thresholds["maximum_ground_truth_distance_m"] = (
            THRESHOLDS["maximum_ground_truth_distance_m"]
            * expected_traversals
        )
        thresholds["maximum_frames"] = (
            THRESHOLDS["maximum_frames"] * expected_traversals
        )
    return {
        "scenario": result.get("scenario") == "cudanav_s_course",
        "stages": result.get("stages") == STAGES,
        "claims": result.get("claims") == CLAIMS,
        "goal_reached": result.get("goal_reached") is True,
        "goal_distance": result.get("ground_truth_goal_distance_m", 1e9)
        <= thresholds["maximum_goal_distance_m"],
        "collision_free": result.get("collision_count", 1e9)
        <= thresholds["maximum_collision_count"],
        "odometry_drift": result.get("odometry_drift_percent", 1e9)
        < thresholds["maximum_odometry_drift_percent"],
        "deadline": result.get("command_deadline_miss_rate", 1.0)
        < thresholds["maximum_command_deadline_miss_rate"],
        "causal_command_effect": (
            result.get("causal_command_effect") is True
            and result.get("command_effect_distance_m", 0.0)
            >= thresholds["minimum_command_effect_distance_m"]
        ),
        "finite_commands": result.get("invalid_commands", 1) == 0,
        "odometry_inliers": result.get("minimum_inliers", 0)
        >= thresholds["minimum_inliers"],
        "voxel_mapping": result.get("final_observed_voxels", 0)
        >= thresholds["minimum_observed_voxels"],
        "occupied_cells": result.get("maximum_occupied_cells", 0)
        >= thresholds["minimum_occupied_cells"],
        "bounded_safety_interventions": result.get(
            "all_colliding_evaluations", 1e9
        )
        <= thresholds["maximum_all_colliding_evaluations"],
        "valid_rollouts": result.get(
            "minimum_nonzero_valid_rollout_ratio", 0.0
        )
        >= thresholds["minimum_nonzero_valid_rollout_ratio"],
        "bounded_path_length": result.get("ground_truth_distance_m", 1e9)
        <= thresholds["maximum_ground_truth_distance_m"],
        "bounded_completion": result.get("frames", 1e9)
        <= thresholds["maximum_frames"],
        "traversals": (
            result.get("traversals_requested") == expected_traversals
            and result.get("traversals_completed") == expected_traversals
            and len(result.get("traversal_frames", [])) == expected_traversals
        ),
        "release_duration": (
            profile != "release"
            or result.get("simulated_duration_s", 0.0)
            >= thresholds["minimum_simulated_duration_s"]
        ),
        "gpu_identity": bool(result.get("gpu", {}).get("name"))
        and result.get("gpu", {}).get("driver_version", 0) > 0,
        "native_quality_gate": result.get("quality_pass") is True,
    }


def trajectory_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def portable_evidence(manifest: dict[str, Any]) -> dict[str, Any]:
    profile = manifest["profile"]
    return {
        "schema_version": 1,
        "result_id": (
            "cudanav_gpu_closed_loop_release_2026-07-29"
            if profile == "release"
            else "cudanav_gpu_closed_loop_s_course_2026-07-29"
        ),
        "source_commit": manifest["source_commit"],
        "source_digest": manifest["source_digest"],
        "scenario": SCENARIO,
        "stages": STAGES,
        "claims": CLAIMS,
        "gpu_identity": manifest["gpu_identity"],
        "profile": manifest["profile"],
        "thresholds": manifest["thresholds"],
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
    profile = evidence["profile"]
    title = (
        "CudaNav native all-GPU 10-minute closed-loop release"
        if profile == "release"
        else "CudaNav native all-GPU closed-loop S-course"
    )
    return f"""# {title}

Date: 2026-07-29

Source commit: `{evidence["source_commit"]}`

Profile: `{profile}`

Result: **{"PASS" if passed else "FAIL"}**

The deterministic S-course plant generates LiDAR from ground truth, but the
controller only receives the GPU KISS-ICP estimate. Each CUDA MPPI command is
applied to the plant before the next scan. GPU voxel mapping and GPU ESDF build
the controller costmap in the same process.

## Result

- Goal reached: {str(metrics["goal_reached"]).lower()}
- Traversals: {metrics["traversals_completed"]}/{metrics["traversals_requested"]}
- Simulated duration: {metrics["simulated_duration_s"]:.1f} s
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
    parser.add_argument("--profile", choices=("smoke", "release"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--maximum-steps", type=int)
    parser.add_argument("--traversals", type=int)
    parser.add_argument("--publish-json", type=Path)
    parser.add_argument("--publish-markdown", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    git_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    runner = args.runner.resolve()
    if not runner.is_file():
        raise SystemExit(f"runner does not exist: {runner}")
    thresholds = (
        RELEASE_THRESHOLDS if args.profile == "release" else THRESHOLDS
    )
    traversals = args.traversals or int(thresholds.get("traversals", 1))
    maximum_steps = args.maximum_steps or 400 * traversals
    minimum_duration = (
        float(thresholds.get("minimum_simulated_duration_s", 0.0))
        if args.traversals is None
        else 0.0
    )
    result_path = output / "result.json"
    trajectory_path = output / "trajectory.csv"
    command = [
        str(runner), "--json", str(result_path), "--csv", str(trajectory_path),
        "--maximum-steps", str(maximum_steps),
        "--traversals", str(traversals),
        "--minimum-duration-s", str(minimum_duration),
        "--check",
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
    gpu_identity = active_gpu_identity()
    checks = evaluate_result(result, args.profile, traversals)
    checks["gpu_binding"] = result.get("gpu", {}).get("name") == gpu_identity["name"]
    checks["clean_checkout"] = args.profile != "release" or not git_dirty
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
        "profile": args.profile,
        "source_commit": git_commit(),
        "git_dirty": git_dirty,
        "source_digest": source_digest(),
        "gpu_identity": gpu_identity,
        "scenario": SCENARIO,
        "thresholds": thresholds,
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
