#!/usr/bin/env python3
"""Run GPU KISS-ICP on a content-addressed real PointCloud2 sequence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from cudanav_real_dataset import read_json
from cudanav_rosbag_evidence import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "docs" / "cudanav_real_dataset_smoke.json"
EXPORTER = ROOT / "scripts" / "export_cudanav_kiss_icp_sequence.py"
RUNNER_NAME = (
    Path("bin/Release/cudanav_kiss_icp_sequence.exe")
    if os.name == "nt"
    else Path("bin/cudanav_kiss_icp_sequence")
)
PROFILES = {
    "smoke": {
        "require_point_time": False,
        "start_offset_s": 1.0,
        "maximum_duration_s": 30.0,
        "maximum_frames": 300,
        "maximum_ate_rmse_m": 5.0,
        "maximum_final_drift_percent": 10.0,
        "minimum_inliers": 30,
    },
    "release": {
        "require_point_time": True,
        "start_offset_s": 1.0,
        "maximum_duration_s": 120.0,
        "maximum_frames": 1200,
        "maximum_ate_rmse_m": 3.0,
        "maximum_final_drift_percent": 5.0,
        "minimum_inliers": 100,
    },
}
SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
CONTRACT_SOURCES = (
    "docs/cudanav_real_dataset_smoke.json",
    "include/cudarobotics/kiss_icp_gpu.hpp",
    "src/gpu_kiss_icp.cu",
    "tools/cudanav_kiss_icp_sequence.cu",
    "scripts/export_cudanav_kiss_icp_sequence.py",
    "scripts/run_cudanav_kiss_icp_real.py",
)
POINT_TIME_UNITS = {"seconds", "milliseconds", "microseconds", "nanoseconds"}


def resolve_pointcloud_auxiliary_fields(
    spec: dict[str, Any], profile: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    pointcloud_contract = spec["recorded_inputs"]["pointcloud"]
    point_time_contract = pointcloud_contract.get("point_time")
    ring_contract = pointcloud_contract.get("ring")
    if profile["require_point_time"] and not isinstance(
        point_time_contract, dict
    ):
        raise ValueError(
            "release profile requires a PointCloud2 point_time contract; "
            "the selected dataset is XYZ-only"
        )
    if point_time_contract is not None and not isinstance(
        point_time_contract, dict
    ):
        raise ValueError("PointCloud2 point_time contract must be an object")
    if isinstance(point_time_contract, dict):
        if (
            not isinstance(point_time_contract.get("field"), str)
            or not point_time_contract["field"]
            or point_time_contract.get("unit") not in POINT_TIME_UNITS
        ):
            raise ValueError(
                "PointCloud2 point_time contract requires a nonempty field "
                "and a supported unit"
            )
    if ring_contract is not None:
        if not isinstance(ring_contract, dict):
            raise ValueError("PointCloud2 ring contract must be an object")
        if (
            not isinstance(ring_contract.get("field"), str)
            or not ring_contract["field"]
        ):
            raise ValueError(
                "PointCloud2 ring contract requires a nonempty field"
            )
    return point_time_contract, ring_contract


def git_identity() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    )
    return commit, dirty


def artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def sha256_text_lf(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


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
    sequence = output / "sequence.bin"
    export_json = output / "export.json"
    result_json = output / "result.json"
    trajectory = output / "trajectory.csv"
    runner_log = output / "runner.log"
    expected_sha = spec["acquisition"]["expected_database_sha256"]
    checks = {
        "profile": profile in PROFILES,
        "git_commit": len(git_commit) == 40,
        "database_filename": database.name
        == spec["acquisition"]["expected_database"],
        "database_bytes": database.stat().st_size
        == spec["acquisition"]["expected_database_bytes"],
        "database_sha256": export_report["database"]["sha256"]
        == expected_sha,
        "pointcloud_topic": export_report["pointcloud_topic"]
        == spec["recorded_inputs"]["pointcloud"]["topic"],
        "pose_topic": export_report["pose_topic"]
        == spec["recorded_inputs"]["odometry"]["topic"],
        "frame_count": result.get("frames") == export_report.get("frames"),
        "point_time_contract": (
            not expected["require_point_time"]
            or (
                export_report.get("sequence_version") == 2
                and export_report.get("point_time", {}).get("present") is True
                and export_report.get("point_time", {}).get(
                    "frames_with_valid_span"
                )
                == result.get("frames")
                and result.get("sequence_version") == 2
                and result.get("deskewed_frames") == result.get("frames")
            )
        ),
        "start_offset": export_report.get("start_offset_s")
        == PROFILES[profile]["start_offset_s"],
        "duration_limit": (
            0.0 < export_report.get("duration_s", 0.0)
            <= PROFILES[profile]["maximum_duration_s"]
        ),
        "point_counts": (
            export_report.get("minimum_points", 0) >= 30
            and export_report.get("minimum_points", 0)
            <= export_report.get("mean_points", 0.0)
            <= export_report.get("maximum_points", 0)
        ),
        "pose_age": (
            0.0 <= export_report.get("pose_age_p95_ms", -1.0)
            <= export_report.get("maximum_pose_age_ms", -1.0)
        ),
        "timestamps": (
            result.get("first_stamp_ns") == export_report.get("first_stamp_ns")
            and result.get("last_stamp_ns") == export_report.get("last_stamp_ns")
        ),
        "gpu_identity": (
            isinstance(result.get("gpu"), dict)
            and bool(result["gpu"].get("name"))
            and str(result["gpu"].get("uuid", "")).startswith("GPU-")
            and result["gpu"].get("driver_version", 0) > 0
        ),
        "gpu_backend": result.get("nn_backend") == "voxel",
        "trajectory_rows": count_csv_rows(trajectory) == result.get("frames"),
        "quality_pass": result.get("quality_pass") is True,
        "scope": True,
    }
    artifacts = {
        "sequence": artifact(sequence, output),
        "export_report": artifact(export_json, output),
        "result": artifact(result_json, output),
        "trajectory": artifact(trajectory, output),
        "runner_log": artifact(runner_log, output),
        "runner": {
            "path": str(runner.resolve()),
            "bytes": runner.stat().st_size,
            "sha256": sha256_file(runner),
        },
    }
    return {
        "schema_version": 1,
        "profile": profile,
        "evidence_mode": "real_sensor_gpu_odometry_with_reference",
        "git_commit": git_commit,
        "git_dirty": False,
        "dataset_id": spec["dataset_id"],
        "dataset_spec_sha256": sha256_file(spec_path),
        "database": {
            "filename": database.name,
            "bytes": database.stat().st_size,
            "sha256": export_report["database"]["sha256"],
        },
        "sequence_contract": {
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
                "sequence_version",
                "point_fields",
                "point_time",
                "ring",
            )
        },
        "gpu": result["gpu"],
        "metrics": {
            key: result[key]
            for key in (
                "frames",
                "duration_s",
                "wall_time_ms",
                "mean_frame_ms",
                "reference_path_length_m",
                "estimated_path_length_m",
                "ate_rmse_m",
                "final_xy_error_m",
                "final_drift_percent",
                "yaw_error_p95_rad",
                "inliers_min",
                "inliers_median",
                "alignment_rmse_p95",
                "nn_ms_p95",
                "sequence_version",
                "deskewed_frames",
                "point_time_span_s_p95",
                "deskew_gpu_ms",
                "thresholds",
                "quality_pass",
            )
        },
        "commands": commands,
        "artifacts": artifacts,
        "claims": {
            "real_pointcloud_gpu_odometry": True,
            "gpu_controller_run": False,
            "closed_loop": False,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def evaluate_manifest(
    manifest_path: Path,
    *,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    payload = read_json(manifest_path)
    checks = {
        "schema": payload.get("schema_version") == 1,
        "profile": payload.get("profile") in PROFILES,
        "evidence_mode": payload.get("evidence_mode")
        == "real_sensor_gpu_odometry_with_reference",
        "commit": (
            isinstance(payload.get("git_commit"), str)
            and len(payload["git_commit"]) == 40
            and (
                expected_commit is None
                or payload["git_commit"] == expected_commit
            )
        ),
        "clean": payload.get("git_dirty") is False,
        "claims": payload.get("claims")
        == {
            "real_pointcloud_gpu_odometry": True,
            "gpu_controller_run": False,
            "closed_loop": False,
        },
        "checks": (
            isinstance(payload.get("checks"), dict)
            and bool(payload["checks"])
            and all(payload["checks"].values())
            and payload.get("passed") is True
        ),
        "artifacts": False,
        "metrics": payload.get("metrics", {}).get("quality_pass") is True,
        "sequence_contract": (
            isinstance(payload.get("sequence_contract"), dict)
            and payload["sequence_contract"].get("frames")
            == payload.get("metrics", {}).get("frames")
            and payload["sequence_contract"].get("start_offset_s")
            == PROFILES.get(payload.get("profile"), {}).get("start_offset_s")
            and 0.0
            < payload["sequence_contract"].get("duration_s", 0.0)
            <= payload["sequence_contract"].get("maximum_duration_s", 0.0)
            and payload["sequence_contract"].get("minimum_points", 0) >= 30
            and 0.0
            <= payload["sequence_contract"].get("pose_age_p95_ms", -1.0)
            <= payload["sequence_contract"].get(
                "maximum_pose_age_ms", -1.0
            )
        ),
    }
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        local = {
            name: descriptor
            for name, descriptor in artifacts.items()
            if name != "runner" and isinstance(descriptor, dict)
        }
        runner = artifacts.get("runner")
        runner_valid = (
            isinstance(runner, dict)
            and runner.get("bytes", 0) > 0
            and bool(SHA256.fullmatch(str(runner.get("sha256", ""))))
            and isinstance(runner.get("path"), str)
            and bool(runner["path"])
        )
        if runner_valid:
            runner_path = Path(runner["path"])
            if runner_path.is_file():
                runner_valid = (
                    runner_path.stat().st_size == runner["bytes"]
                    and sha256_file(runner_path) == runner["sha256"]
                )
        checks["artifacts"] = (
            set(local)
            == {
                "sequence",
                "export_report",
                "result",
                "trajectory",
                "runner_log",
            }
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
        raise ValueError(
            "real KISS-ICP manifest is invalid: "
            + json.dumps(validation["checks"], sort_keys=True)
        )
    manifest = read_json(manifest_path)
    retained = {
        name: {
            "bytes": descriptor["bytes"],
            "sha256": descriptor["sha256"],
        }
        for name, descriptor in manifest["artifacts"].items()
        if isinstance(descriptor, dict)
        and "bytes" in descriptor
        and "sha256" in descriptor
    }
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
        "metrics": manifest["metrics"],
        "retained_artifacts": retained,
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
    artifacts = payload.get("retained_artifacts")
    sources = payload.get("contract_sources")
    checks = {
        "schema": payload.get("schema_version") == 1,
        "result_id": isinstance(payload.get("result_id"), str)
        and bool(payload["result_id"]),
        "evidence_mode": payload.get("evidence_mode")
        == "real_sensor_gpu_odometry_with_reference",
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
            isinstance(payload.get("gpu"), dict)
            and bool(payload["gpu"].get("name"))
            and str(payload["gpu"].get("uuid", "")).startswith("GPU-")
            and payload["gpu"].get("driver_version", 0) > 0
        ),
        "metrics": (
            isinstance(payload.get("metrics"), dict)
            and payload["metrics"].get("quality_pass") is True
            and payload["metrics"].get("frames", 0) >= 2
            and payload["metrics"].get("reference_path_length_m", 0.0) > 0.0
        ),
        "sequence_contract": (
            isinstance(payload.get("sequence_contract"), dict)
            and payload["sequence_contract"].get("frames")
            == payload.get("metrics", {}).get("frames")
            and payload["sequence_contract"].get("start_offset_s")
            == PROFILES.get(payload.get("profile"), {}).get("start_offset_s")
            and 0.0
            < payload["sequence_contract"].get("duration_s", 0.0)
            <= payload["sequence_contract"].get("maximum_duration_s", 0.0)
            and payload["sequence_contract"].get("minimum_points", 0) >= 30
            and 0.0
            <= payload["sequence_contract"].get("pose_age_p95_ms", -1.0)
            <= payload["sequence_contract"].get(
                "maximum_pose_age_ms", -1.0
            )
        ),
        "source_validation": (
            payload.get("source_validation", {}).get("valid") is True
            and all(
                payload.get("source_validation", {})
                .get("checks", {})
                .values()
            )
        ),
        "claims": payload.get("claims")
        == {
            "real_pointcloud_gpu_odometry": True,
            "gpu_controller_run": False,
            "closed_loop": False,
        },
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
            and all(isinstance(entry, dict) for entry in sources)
            and {entry["path"] for entry in sources} == set(CONTRACT_SOURCES)
            and all(
                entry.get("normalization") == "text_lf"
                for entry in sources
            )
            and all(
                bool(SHA256.fullmatch(str(entry.get("sha256", ""))))
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
    gpu = payload["gpu"]
    sequence = payload["sequence_contract"]
    return (
        f"# {payload['result_id']}\n\n"
        "GPU KISS-ICP odometry on a content-addressed real PointCloud2 "
        "sequence. This is not a controller or closed-loop result.\n\n"
        f"- Source commit: `{payload['source_commit']}`\n"
        f"- Dataset: `{payload['dataset_id']}`\n"
        f"- GPU: `{gpu['name']}` (`{gpu['uuid']}`)\n"
        f"- Frames / duration: {metrics['frames']} / "
        f"{metrics['duration_s']:.3f} s\n"
        f"- Declared profile / startup offset: `{payload['profile']}` / "
        f"{sequence['start_offset_s']:.3f} s\n"
        f"- Points per frame (min / mean / max): "
        f"{sequence['minimum_points']} / {sequence['mean_points']:.2f} / "
        f"{sequence['maximum_points']}\n"
        f"- Reference pose age p95: {sequence['pose_age_p95_ms']:.6f} ms\n"
        f"- Reference path: {metrics['reference_path_length_m']:.3f} m\n"
        f"- ATE RMSE: {metrics['ate_rmse_m']:.3f} m\n"
        f"- Final drift: {metrics['final_drift_percent']:.3f}%\n"
        f"- Yaw error p95: {metrics['yaw_error_p95_rad']:.6f} rad\n"
        f"- Mean frame time: {metrics['mean_frame_ms']:.3f} ms\n"
        f"- GPU NN p95: {metrics['nn_ms_p95']:.3f} ms\n"
        f"- Minimum inliers: {metrics['inliers_min']}\n"
        f"- Quality gate: {'PASS' if metrics['quality_pass'] else 'FAIL'}\n\n"
        "## Scope\n\n"
        "- Real PointCloud2 GPU odometry: yes\n"
        "- GPU controller run: no\n"
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
        validation = evaluate_manifest(
            args.validate,
            expected_commit=args.commit,
        )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if validation["valid"] else 1
    if args.validate_portable is not None:
        validation = evaluate_portable_evidence(
            read_json(args.validate_portable),
            expected_source_commit=args.commit,
        )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if validation["valid"] else 1
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
            raise SystemExit(json.dumps(validation, indent=2, sort_keys=True))
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
    spec = read_json(args.spec)
    profile = PROFILES[args.profile]
    try:
        point_time_contract, ring_contract = (
            resolve_pointcloud_auxiliary_fields(spec, profile)
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output.mkdir(parents=True)
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
    if isinstance(point_time_contract, dict):
        export_command.extend(
            [
                "--point-time-field",
                point_time_contract["field"],
                "--point-time-unit",
                point_time_contract["unit"],
            ]
        )
        if profile["require_point_time"]:
            export_command.append("--require-point-time")
    if isinstance(ring_contract, dict):
        export_command.extend(["--ring-field", ring_contract["field"]])
        if ring_contract.get("required", False):
            export_command.append("--require-ring")
    subprocess.run(export_command, cwd=ROOT, check=True)
    runner_command = [
        str(runner),
        "--sequence",
        str(sequence),
        "--json",
        str(result_json),
        "--csv",
        str(trajectory),
        "--minimum-inliers",
        str(profile["minimum_inliers"]),
        "--maximum-ate-rmse-m",
        str(profile["maximum_ate_rmse_m"]),
        "--maximum-final-drift-percent",
        str(profile["maximum_final_drift_percent"]),
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
            f"GPU runner failed ({completed.returncode}); see {runner_log}"
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
            "gpu_kiss_icp": runner_command,
        },
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = evaluate_manifest(manifest_path, expected_commit=commit)
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
