#!/usr/bin/env python3
"""Run the complete GPU CudaNav core as a real-rosbag shadow pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from cudanav_real_dataset import read_json
from cudanav_rosbag_evidence import sha256_file
from run_cudanav_kiss_icp_real import git_identity, sha256_text_lf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "docs" / "cudanav_real_dataset_smoke.json"
EXPORTER = ROOT / "scripts" / "export_cudanav_kiss_icp_sequence.py"
RUNNER_NAME = (
    Path("bin/Release/cudanav_real_gpu_stack_sequence.exe")
    if os.name == "nt"
    else Path("bin/cudanav_real_gpu_stack_sequence")
)
STAGES = [
    "gpu_kiss_icp",
    "gpu_voxel_mapping",
    "gpu_esdf",
    "cuda_mppi",
]
CLAIMS = {
    "real_pointcloud_gpu_odometry": True,
    "real_voxel_mapping": True,
    "real_esdf": True,
    "real_cuda_mppi_shadow": True,
    "ros2_runtime": False,
    "closed_loop": False,
}
PROFILES = {
    "smoke": {
        "start_offset_s": 1.0,
        "maximum_duration_s": 30.0,
        "maximum_frames": 300,
        "control_stride": 10,
        "maximum_ate_rmse_m": 5.0,
        "maximum_final_drift_percent": 10.0,
        "minimum_inliers": 30,
        "minimum_observed_voxels": 500,
        "minimum_occupied_cells": 10,
        "minimum_control_evaluations": 20,
        "maximum_all_colliding_evaluations": 3,
        "minimum_valid_rollout_ratio": 0.01,
        "maximum_safety_stop_speed": 0.05,
    },
    "release": {
        "start_offset_s": 1.0,
        "maximum_duration_s": 120.0,
        "maximum_frames": 1200,
        "control_stride": 10,
        "maximum_ate_rmse_m": 3.0,
        "maximum_final_drift_percent": 5.0,
        "minimum_inliers": 100,
        "minimum_observed_voxels": 1000,
        "minimum_occupied_cells": 50,
        "minimum_control_evaluations": 100,
        "maximum_all_colliding_evaluations": 6,
        "minimum_valid_rollout_ratio": 0.01,
        "maximum_safety_stop_speed": 0.05,
    },
}
CONTRACT_SOURCES = (
    "docs/cudanav_real_dataset_smoke.json",
    "include/cudarobotics/kiss_icp_gpu.hpp",
    "include/cudarobotics/voxel_mapping_gpu.hpp",
    "include/cudarobotics/esdf_2d_gpu.hpp",
    "include/cuda_mppi_controller/mppi_gpu.hpp",
    "src/gpu_kiss_icp.cu",
    "src/voxel_mapping_gpu.cu",
    "src/esdf_2d_gpu.cu",
    "src/mppi_gpu.cu",
    "tools/cudanav_real_gpu_stack_sequence.cu",
    "scripts/export_cudanav_kiss_icp_sequence.py",
    "scripts/run_cudanav_real_gpu_stack.py",
    "ros2_ws/src/cuda_voxel_mapping/src/cuda_voxel_mapper_node.cpp",
    "ros2_ws/src/cuda_voxel_costmap_layer/src/cuda_voxel_costmap_layer.cpp",
    "ros2_ws/src/cuda_nav_bringup/config/controller.yaml",
)
SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")


def artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def sequence_contract(export_report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: export_report[key]
        for key in (
            "pointcloud_topic",
            "pose_topic",
            "pose_type",
            "frame_id",
            "frames",
            "duration_s",
            "start_offset_s",
            "maximum_duration_s",
            "maximum_pose_age_ms",
            "pose_age_p95_ms",
            "minimum_points",
            "mean_points",
            "maximum_points",
            "reference_path_length_m",
        )
    }


def metrics_contract(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "frames",
            "duration_s",
            "wall_time_ms",
            "mean_frame_ms",
            "frame_ms_p95",
            "reference_path_length_m",
            "estimated_path_length_m",
            "ate_rmse_m",
            "final_xy_error_m",
            "final_drift_percent",
            "yaw_error_p95_rad",
            "inliers_min",
            "nn_ms_p95",
            "mapping",
            "esdf",
            "mppi",
            "thresholds",
            "quality_pass",
        )
    }


def make_manifest(
    output: Path,
    *,
    profile: str,
    git_commit: str,
    spec_path: Path,
    database: Path,
    runner: Path,
    export_report: dict[str, Any],
    result: dict[str, Any],
    commands: dict[str, list[str]],
) -> dict[str, Any]:
    spec = read_json(spec_path)
    expected = PROFILES[profile]
    trajectory = output / "trajectory.csv"
    checks = {
        "profile": profile in PROFILES,
        "git_commit": bool(COMMIT.fullmatch(git_commit)),
        "database_filename": database.name
        == spec["acquisition"]["expected_database"],
        "database_bytes": database.stat().st_size
        == spec["acquisition"]["expected_database_bytes"],
        "database_sha256": export_report["database"]["sha256"]
        == spec["acquisition"]["expected_database_sha256"],
        "pointcloud_topic": export_report["pointcloud_topic"]
        == spec["recorded_inputs"]["pointcloud"]["topic"],
        "pose_topic": export_report["pose_topic"]
        == spec["recorded_inputs"]["odometry"]["topic"],
        "start_offset": export_report["start_offset_s"]
        == expected["start_offset_s"],
        "frames": result.get("frames") == export_report.get("frames"),
        "stages": result.get("stages") == STAGES,
        "gpu_identity": (
            isinstance(result.get("gpu"), dict)
            and bool(result["gpu"].get("name"))
            and str(result["gpu"].get("uuid", "")).startswith("GPU-")
            and result["gpu"].get("driver_version", 0) > 0
        ),
        "mapping": (
            result.get("mapping", {}).get("final_observed_voxels", 0)
            >= expected["minimum_observed_voxels"]
            and result.get("mapping", {}).get("maximum_occupied_cells", 0)
            >= expected["minimum_occupied_cells"]
        ),
        "esdf": (
            result.get("esdf", {}).get("unknown_policy") == "free"
            and result.get("esdf", {}).get(
                "footprint_clearing_radius_m"
            )
            == 0.30
            and result.get("esdf", {}).get("gpu_ms_p95", -1.0) >= 0.0
        ),
        "mppi": (
            result.get("mppi", {}).get("evaluations", 0)
            >= expected["minimum_control_evaluations"]
            and result.get("mppi", {}).get(
                "all_colliding_evaluations", 10**9
            )
            <= expected["maximum_all_colliding_evaluations"]
            and result.get("mppi", {}).get(
                "maximum_all_colliding_abs_v", 10**9
            )
            <= expected["maximum_safety_stop_speed"]
            and result.get("mppi", {}).get("invalid_commands", 1) == 0
        ),
        "trajectory_rows": count_csv_rows(trajectory)
        == result.get("frames"),
        "quality_pass": result.get("quality_pass") is True,
        "scope": True,
    }
    artifacts = {
        name: artifact(output / filename, output)
        for name, filename in (
            ("sequence", "sequence.bin"),
            ("export_report", "export.json"),
            ("result", "result.json"),
            ("trajectory", "trajectory.csv"),
            ("runner_log", "runner.log"),
        )
    }
    artifacts["runner"] = {
        "path": str(runner.resolve()),
        "bytes": runner.stat().st_size,
        "sha256": sha256_file(runner),
    }
    return {
        "schema_version": 1,
        "profile": profile,
        "evidence_mode": "real_sensor_all_gpu_core_shadow",
        "git_commit": git_commit,
        "git_dirty": False,
        "dataset_id": spec["dataset_id"],
        "dataset_spec_sha256": sha256_file(spec_path),
        "database": {
            "filename": database.name,
            "bytes": database.stat().st_size,
            "sha256": export_report["database"]["sha256"],
        },
        "sequence_contract": sequence_contract(export_report),
        "gpu": result["gpu"],
        "stages": result["stages"],
        "metrics": metrics_contract(result),
        "commands": commands,
        "artifacts": artifacts,
        "claims": dict(CLAIMS),
        "checks": checks,
        "passed": all(checks.values()),
    }


def evaluate_manifest(
    manifest_path: Path,
    *,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    path = manifest_path.resolve()
    root = path.parent
    payload = read_json(path)
    checks = {
        "schema": payload.get("schema_version") == 1,
        "profile": payload.get("profile") in PROFILES,
        "evidence_mode": payload.get("evidence_mode")
        == "real_sensor_all_gpu_core_shadow",
        "commit": (
            bool(COMMIT.fullmatch(str(payload.get("git_commit", ""))))
            and (
                expected_commit is None
                or payload["git_commit"] == expected_commit
            )
        ),
        "clean": payload.get("git_dirty") is False,
        "stages": payload.get("stages") == STAGES,
        "claims": payload.get("claims") == CLAIMS,
        "checks": (
            isinstance(payload.get("checks"), dict)
            and bool(payload["checks"])
            and all(payload["checks"].values())
            and payload.get("passed") is True
        ),
        "metrics": payload.get("metrics", {}).get("quality_pass") is True,
        "sequence": (
            payload.get("sequence_contract", {}).get("frames")
            == payload.get("metrics", {}).get("frames")
            and payload.get("sequence_contract", {}).get("start_offset_s")
            == PROFILES.get(payload.get("profile"), {}).get(
                "start_offset_s"
            )
        ),
        "artifacts": False,
    }
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        local_names = {
            "sequence",
            "export_report",
            "result",
            "trajectory",
            "runner_log",
        }
        local = {
            name: descriptor
            for name, descriptor in artifacts.items()
            if name in local_names and isinstance(descriptor, dict)
        }
        runner = artifacts.get("runner")
        runner_valid = (
            isinstance(runner, dict)
            and runner.get("bytes", 0) > 0
            and bool(SHA256.fullmatch(str(runner.get("sha256", ""))))
            and bool(runner.get("path"))
        )
        if runner_valid and Path(runner["path"]).is_file():
            runner_path = Path(runner["path"])
            runner_valid = (
                runner_path.stat().st_size == runner["bytes"]
                and sha256_file(runner_path) == runner["sha256"]
            )
        checks["artifacts"] = (
            set(local) == local_names
            and all(
                (root / descriptor.get("path", "")).is_file()
                and (root / descriptor["path"]).stat().st_size
                == descriptor.get("bytes")
                and sha256_file(root / descriptor["path"])
                == descriptor.get("sha256")
                for descriptor in local.values()
            )
            and runner_valid
        )
    return {"valid": all(checks.values()), "checks": checks}


def make_portable_evidence(
    manifest_path: Path,
    *,
    result_id: str,
    publisher_commit: str,
) -> dict[str, Any]:
    validation = evaluate_manifest(manifest_path)
    if not validation["valid"]:
        raise ValueError(json.dumps(validation, sort_keys=True))
    manifest = read_json(manifest_path)
    return {
        "schema_version": 1,
        "result_id": result_id,
        "evidence_mode": manifest["evidence_mode"],
        "profile": manifest["profile"],
        "source_commit": manifest["git_commit"],
        "publisher_commit": publisher_commit,
        "dataset_id": manifest["dataset_id"],
        "dataset_spec_sha256": manifest["dataset_spec_sha256"],
        "database": manifest["database"],
        "sequence_contract": manifest["sequence_contract"],
        "gpu": manifest["gpu"],
        "stages": manifest["stages"],
        "metrics": manifest["metrics"],
        "retained_artifacts": {
            name: {
                "bytes": descriptor["bytes"],
                "sha256": descriptor["sha256"],
            }
            for name, descriptor in manifest["artifacts"].items()
        },
        "source_validation": validation,
        "claims": manifest["claims"],
        "contract_sources": [
            {
                "path": relative,
                "normalization": "text_lf",
                "sha256": sha256_text_lf(ROOT / relative),
            }
            for relative in CONTRACT_SOURCES
        ],
    }


def evaluate_portable_evidence(
    payload: dict[str, Any],
    *,
    expected_source_commit: str | None = None,
    verify_sources: bool = True,
) -> dict[str, Any]:
    sources = payload.get("contract_sources")
    artifacts = payload.get("retained_artifacts")
    metrics = payload.get("metrics", {})
    checks = {
        "schema": payload.get("schema_version") == 1,
        "result_id": bool(payload.get("result_id")),
        "evidence_mode": payload.get("evidence_mode")
        == "real_sensor_all_gpu_core_shadow",
        "profile": payload.get("profile") in PROFILES,
        "source_commit": (
            bool(COMMIT.fullmatch(str(payload.get("source_commit", ""))))
            and (
                expected_source_commit is None
                or payload["source_commit"] == expected_source_commit
            )
        ),
        "publisher_commit": bool(
            COMMIT.fullmatch(str(payload.get("publisher_commit", "")))
        ),
        "dataset": (
            bool(payload.get("dataset_id"))
            and bool(
                SHA256.fullmatch(
                    str(payload.get("dataset_spec_sha256", ""))
                )
            )
            and bool(
                SHA256.fullmatch(
                    str(payload.get("database", {}).get("sha256", ""))
                )
            )
        ),
        "gpu": (
            bool(payload.get("gpu", {}).get("name"))
            and str(payload.get("gpu", {}).get("uuid", "")).startswith("GPU-")
        ),
        "stages": payload.get("stages") == STAGES,
        "metrics": (
            metrics.get("quality_pass") is True
            and metrics.get("frames", 0) >= 2
            and metrics.get("mapping", {}).get(
                "final_observed_voxels", 0
            )
            > 0
            and metrics.get("esdf", {}).get("gpu_ms_p95", -1.0) >= 0.0
            and metrics.get("mppi", {}).get("evaluations", 0) > 0
            and metrics.get("mppi", {}).get("invalid_commands", 1) == 0
        ),
        "source_validation": (
            payload.get("source_validation", {}).get("valid") is True
            and all(
                payload.get("source_validation", {})
                .get("checks", {})
                .values()
            )
        ),
        "claims": payload.get("claims") == CLAIMS,
        "retained_artifacts": (
            isinstance(artifacts, dict)
            and {
                "sequence",
                "export_report",
                "result",
                "trajectory",
                "runner_log",
                "runner",
            }
            <= set(artifacts)
            and all(
                descriptor.get("bytes", 0) > 0
                and bool(
                    SHA256.fullmatch(str(descriptor.get("sha256", "")))
                )
                for descriptor in artifacts.values()
                if isinstance(descriptor, dict)
            )
        ),
        "contract_sources": (
            isinstance(sources, list)
            and len(sources) == len(CONTRACT_SOURCES)
            and {entry.get("path") for entry in sources}
            == set(CONTRACT_SOURCES)
            and all(
                entry.get("normalization") == "text_lf"
                and bool(
                    SHA256.fullmatch(str(entry.get("sha256", "")))
                )
                for entry in sources
            )
        ),
    }
    if checks["contract_sources"] and verify_sources:
        checks["contract_sources"] = all(
            (ROOT / entry["path"]).is_file()
            and sha256_text_lf(ROOT / entry["path"]) == entry["sha256"]
            for entry in sources
        )
    return {"valid": all(checks.values()), "checks": checks}


def render_portable_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    mapping = metrics["mapping"]
    esdf = metrics["esdf"]
    mppi = metrics["mppi"]
    sequence = payload["sequence_contract"]
    return (
        f"# {payload['result_id']}\n\n"
        "Real PointCloud2 shadow execution through GPU KISS-ICP, rolling "
        "voxel mapping, GPU ESDF inflation, and CUDA MPPI. Commands are "
        "evaluated but not applied, so this is not closed-loop evidence.\n\n"
        f"- Source commit: `{payload['source_commit']}`\n"
        f"- Dataset: `{payload['dataset_id']}`\n"
        f"- GPU: `{payload['gpu']['name']}` (`{payload['gpu']['uuid']}`)\n"
        f"- Profile / startup offset: `{payload['profile']}` / "
        f"{sequence['start_offset_s']:.3f} s\n"
        f"- Frames / duration: {metrics['frames']} / "
        f"{metrics['duration_s']:.3f} s\n"
        f"- ATE RMSE / final drift: {metrics['ate_rmse_m']:.3f} m / "
        f"{metrics['final_drift_percent']:.3f}%\n"
        f"- Final observed voxels / peak occupied cells: "
        f"{mapping['final_observed_voxels']} / "
        f"{mapping['maximum_occupied_cells']}\n"
        f"- ESDF p95: {esdf['gpu_ms_p95']:.3f} ms\n"
        f"- MPPI evaluations / solve p95: {mppi['evaluations']} / "
        f"{mppi['solve_ms_p95']:.3f} ms\n"
        f"- Nonzero valid-rollout ratio minimum: "
        f"{mppi['minimum_nonzero_valid_rollout_ratio']:.4f}\n"
        f"- Safety-stop evaluations: "
        f"{mppi['all_colliding_evaluations']} "
        f"(maximum |v| {mppi['maximum_all_colliding_abs_v']:.3f} m/s)\n"
        f"- End-to-end mean / p95 frame time: "
        f"{metrics['mean_frame_ms']:.3f} / "
        f"{metrics['frame_ms_p95']:.3f} ms\n"
        f"- Quality gate: {'PASS' if metrics['quality_pass'] else 'FAIL'}\n\n"
        "## Scope\n\n"
        "- Real PointCloud2 all-GPU core shadow: yes\n"
        "- ROS 2 runtime: no\n"
        "- Commands applied to vehicle or simulator: no\n"
        "- Closed-loop evidence: no\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--runner", type=Path, default=ROOT / RUNNER_NAME)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--validate-portable", type=Path)
    parser.add_argument("--publish", type=Path)
    parser.add_argument("--result-id")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.validate is not None:
        result = evaluate_manifest(
            args.validate, expected_commit=args.commit
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    if args.validate_portable is not None:
        result = evaluate_portable_evidence(
            read_json(args.validate_portable),
            expected_source_commit=args.commit,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    if args.publish is not None:
        if (
            args.result_id is None
            or args.output_json is None
            or args.output_markdown is None
        ):
            parser.error(
                "--publish requires --result-id, --output-json, and "
                "--output-markdown"
            )
        commit, dirty = git_identity()
        if dirty:
            raise SystemExit("refusing to publish from a dirty worktree")
        for path in (args.output_json, args.output_markdown):
            if path.exists():
                raise SystemExit(f"refusing existing output: {path}")
        payload = make_portable_evidence(
            args.publish,
            result_id=args.result_id,
            publisher_commit=commit,
        )
        validation = evaluate_portable_evidence(payload)
        if not validation["valid"]:
            raise SystemExit(json.dumps(validation, indent=2))
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.output_markdown.write_text(
            render_portable_markdown(payload),
            encoding="utf-8",
        )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0
    if args.database is None or args.output_dir is None:
        parser.error("running requires --database and --output-dir")

    commit, dirty = git_identity()
    if dirty:
        raise SystemExit("refusing real-data evidence from a dirty worktree")
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing existing output directory: {output}")
    runner = args.runner.resolve()
    if not runner.is_file():
        raise SystemExit(f"GPU runner not found: {runner}")
    output.mkdir(parents=True)
    spec = read_json(args.spec)
    profile = PROFILES[args.profile]
    sequence = output / "sequence.bin"
    export_json = output / "export.json"
    result_json = output / "result.json"
    trajectory = output / "trajectory.csv"
    runner_log = output / "runner.log"
    export_command = [
        sys.executable,
        str(EXPORTER),
        "--database",
        str(args.database.resolve()),
        "--pointcloud-topic",
        spec["recorded_inputs"]["pointcloud"]["topic"],
        "--pose-topic",
        spec["recorded_inputs"]["odometry"]["topic"],
        "--pose-type",
        spec["recorded_inputs"]["odometry"]["type"],
        "--output",
        str(sequence),
        "--report",
        str(export_json),
        "--start-offset-s",
        str(profile["start_offset_s"]),
        "--maximum-duration-s",
        str(profile["maximum_duration_s"]),
        "--maximum-frames",
        str(profile["maximum_frames"]),
    ]
    subprocess.run(export_command, cwd=ROOT, check=True)
    runner_command = [
        str(runner),
        "--sequence",
        str(sequence),
        "--json",
        str(result_json),
        "--csv",
        str(trajectory),
        "--control-stride",
        str(profile["control_stride"]),
        "--minimum-inliers",
        str(profile["minimum_inliers"]),
        "--minimum-observed-voxels",
        str(profile["minimum_observed_voxels"]),
        "--minimum-occupied-cells",
        str(profile["minimum_occupied_cells"]),
        "--minimum-control-evaluations",
        str(profile["minimum_control_evaluations"]),
        "--maximum-all-colliding-evaluations",
        str(profile["maximum_all_colliding_evaluations"]),
        "--minimum-valid-rollout-ratio",
        str(profile["minimum_valid_rollout_ratio"]),
        "--maximum-safety-stop-speed",
        str(profile["maximum_safety_stop_speed"]),
        "--maximum-ate-rmse-m",
        str(profile["maximum_ate_rmse_m"]),
        "--maximum-final-drift-percent",
        str(profile["maximum_final_drift_percent"]),
        "--check",
    ]
    completed = subprocess.run(
        runner_command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    runner_log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or not result_json.is_file():
        raise SystemExit(
            f"GPU stack runner failed ({completed.returncode}); "
            f"see {runner_log}"
        )
    manifest = make_manifest(
        output,
        profile=args.profile,
        git_commit=commit,
        spec_path=args.spec,
        database=args.database.resolve(),
        runner=runner,
        export_report=read_json(export_json),
        result=read_json(result_json),
        commands={
            "export": export_command,
            "real_gpu_stack": runner_command,
        },
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = evaluate_manifest(
        manifest_path, expected_commit=commit
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
