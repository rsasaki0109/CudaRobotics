#!/usr/bin/env python3
"""Freeze a validated CudaNav release suite into portable paper artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cudanav_autonomy_suite import evaluate_suite, sha256_file
from cudanav_multi_gpu import evaluate_multi_gpu_suite
from cudanav_ros_ci_evidence import evaluate as evaluate_ros_ci


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def mode_directory(
    suite_root: Path, suite: dict[str, Any], mode: str
) -> Path:
    relative = suite["modes"][mode]["directory"]
    directory = (suite_root / relative).resolve()
    if not directory.is_relative_to(suite_root) or not directory.is_dir():
        raise ValueError(f"invalid {mode} directory")
    return directory


def checked_artifact(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path is empty")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"artifact is missing or escapes run: {relative}")
    return path


def load_release(
    suite_root: Path, ros_ci_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    suite_root = suite_root.resolve()
    suite_path = suite_root / "manifest.json"
    suite = read_json(suite_path)
    gate = evaluate_suite(suite, suite_root)
    if suite.get("profile") != "release" or not gate["passed"]:
        raise ValueError("CudaNav autonomy suite does not pass the release gate")
    if suite.get("git_dirty") is not False:
        raise ValueError("CudaNav autonomy suite is not bound to a clean checkout")
    ros_ci = read_json(ros_ci_path.resolve())
    ros_gate = evaluate_ros_ci(ros_ci)
    if not ros_gate["passed"]:
        raise ValueError("ROS 2 Jazzy CI evidence does not pass")
    if ros_ci["git_commit"] != suite["git_commit"]:
        raise ValueError("ROS CI and autonomy suite commits differ")
    return suite, ros_ci, gate


def build_artifacts(
    suite_root: Path, ros_ci_path: Path
) -> tuple[dict[str, Any], dict[str, Any], str]:
    suite_root = suite_root.resolve()
    suite, ros_ci, suite_gate = load_release(suite_root, ros_ci_path)
    closed_root = mode_directory(suite_root, suite, "closed_loop")
    rosbag_root = mode_directory(suite_root, suite, "real_rosbag_shadow")
    multi_root = mode_directory(suite_root, suite, "multi_gpu")

    closed_manifest_path = closed_root / "manifest.json"
    closed_manifest = read_json(closed_manifest_path)
    closed_summary_path = checked_artifact(
        closed_root, closed_manifest["artifacts"]["summary"]
    )
    closed_summary = read_json(closed_summary_path)

    rosbag_manifest_path = rosbag_root / "manifest.json"
    rosbag_manifest = read_json(rosbag_manifest_path)
    evaluation_path = checked_artifact(
        rosbag_root, rosbag_manifest["artifacts"]["evaluation"]
    )
    evaluation = read_json(evaluation_path)

    multi_manifest_path = multi_root / "multi_gpu_manifest.json"
    multi_manifest = read_json(multi_manifest_path)
    multi_gate = evaluate_multi_gpu_suite(multi_manifest, multi_root)
    if not multi_gate["passed"]:
        raise ValueError("multi-GPU evidence failed during publication")

    config_hashes = suite_gate["coverage"]["config_sha256"]
    if len(config_hashes) != 1:
        raise ValueError("release suite does not have one controller config")
    gpu_models = multi_gate["coverage"]["gpu_models"]
    gpu_uuids = multi_gate["coverage"]["gpu_uuids"]
    diagnostics = evaluation.get("diagnostics", {})
    motion = evaluation.get("motion", {})

    summary = {
        "schema_version": 1,
        "evidence_mode": "cudanav_systems_release",
        "status": "passed",
        "git_commit": suite["git_commit"],
        "controller_config_sha256": config_hashes[0],
        "closed_loop": {
            "evidence_mode": "closed_loop_simulation",
            "elapsed_sec": closed_summary["elapsed_sec"],
            "collision_count": closed_summary["collision_count"],
            "odometry_drift_percent": closed_summary[
                "odometry_drift_percent"
            ],
            "command_deadline_miss_rate": closed_summary[
                "command_deadline_miss_rate"
            ],
            "traversals_requested": closed_summary[
                "traversals_requested"
            ],
            "traversals_completed": closed_summary[
                "traversals_completed"
            ],
        },
        "real_rosbag_shadow": {
            "evidence_mode": "shadow_controller_with_recorded_motion",
            "quality_pass": evaluation["quality_pass"],
            "input_tree_sha256": rosbag_manifest["input_bag"][
                "tree_sha256"
            ],
            "input_file_count": rosbag_manifest["input_bag"]["file_count"],
            "input_total_bytes": rosbag_manifest["input_bag"]["total_bytes"],
            "duration_sec": motion["duration_s"],
            "diagnostics_samples": diagnostics["samples"],
        },
        "multi_gpu": {
            "run_count": len(multi_manifest["runs"]),
            "repetitions": multi_manifest["repetitions"],
            "physical_device_count": len(gpu_uuids),
            "physical_model_count": len(gpu_models),
            "gpu_models": gpu_models,
        },
        "ros_jazzy_ci": {
            "status": ros_ci["status"],
            "run_url": ros_ci["github"]["run_url"],
            "run_attempt": ros_ci["github"]["run_attempt"],
            "runner_image": ros_ci["platform"]["image"],
            "runner_arch": ros_ci["platform"]["arch"],
            "ros_distro": ros_ci["ros"]["distro"],
            "cuda_toolkit": ros_ci["cuda"]["toolkit"],
            "packages": ros_ci["packages"],
            "checks": ros_ci["checks"],
        },
    }
    provenance = {
        "schema_version": 1,
        "evidence_mode": "cudanav_systems_release_provenance",
        "git_commit": suite["git_commit"],
        "controller_config_sha256": config_hashes[0],
        "source_artifacts": {
            "autonomy_suite_manifest_sha256": sha256_file(
                suite_root / "manifest.json"
            ),
            "closed_loop_manifest_sha256": sha256_file(
                closed_manifest_path
            ),
            "closed_loop_summary_sha256": sha256_file(closed_summary_path),
            "real_rosbag_manifest_sha256": sha256_file(
                rosbag_manifest_path
            ),
            "real_rosbag_evaluation_sha256": sha256_file(evaluation_path),
            "multi_gpu_manifest_sha256": sha256_file(multi_manifest_path),
            "ros_jazzy_ci_sha256": sha256_file(ros_ci_path.resolve()),
        },
        "hardware": {
            "closed_loop_gpu": closed_manifest["gpu"],
            "multi_gpu_models": gpu_models,
            "multi_gpu_uuids": gpu_uuids,
        },
        "input_bag": {
            "tree_sha256": rosbag_manifest["input_bag"]["tree_sha256"],
            "file_count": rosbag_manifest["input_bag"]["file_count"],
            "total_bytes": rosbag_manifest["input_bag"]["total_bytes"],
        },
        "ros_jazzy_ci": ros_ci["github"],
    }
    report = render_report(summary)
    return summary, provenance, report


def render_report(summary: dict[str, Any]) -> str:
    closed = summary["closed_loop"]
    bag = summary["real_rosbag_shadow"]
    multi = summary["multi_gpu"]
    ci = summary["ros_jazzy_ci"]
    models = ", ".join(f"`{model}`" for model in multi["gpu_models"])
    lines = [
        "# CudaNav Systems Release Evidence",
        "",
        f"- Status: **{summary['status']}**",
        f"- Git commit: `{summary['git_commit']}`",
        (
            "- Controller config SHA-256: "
            f"`{summary['controller_config_sha256']}`"
        ),
        "",
        "## Evidence modes",
        "",
        "| Mode | Result | Scope |",
        "|---|---:|---|",
        (
            f"| Closed-loop simulation | {closed['elapsed_sec']:.1f} s, "
            f"{closed['collision_count']} collisions | Commands affect "
            "subsequent simulated state |"
        ),
        (
            f"| Real rosbag shadow | {bag['duration_sec']:.1f} s, "
            f"{bag['diagnostics_samples']} diagnostics | Recorded motion; "
            "not a closed-loop claim |"
        ),
        (
            f"| Physical multi-GPU | {multi['physical_device_count']} devices, "
            f"{multi['physical_model_count']} models | Closed-loop smoke "
            "reproduction |"
        ),
        (
            f"| ROS 2 Jazzy CI | `{ci['status']}` | Ubuntu 24.04, "
            f"CUDA {ci['cuda_toolkit']}, commit-bound workflow artifact |"
        ),
        "",
        "## Closed-loop release gates",
        "",
        f"- Odometry drift: {closed['odometry_drift_percent']:.3f}%",
        (
            "- Controller deadline-miss rate: "
            f"{closed['command_deadline_miss_rate'] * 100.0:.3f}%"
        ),
        (
            f"- Traversals: {closed['traversals_completed']}/"
            f"{closed['traversals_requested']}"
        ),
        "",
        "## Reproduction coverage",
        "",
        f"- Physical GPU models: {models}",
        f"- Real-bag tree SHA-256: `{bag['input_tree_sha256']}`",
        f"- ROS workflow: {ci['run_url']}",
        "",
        "The real-bag result is explicitly shadow-controller evidence. It is "
        "not combined with the closed-loop simulation success claim.",
        "",
    ]
    return "\n".join(lines)


def encoded(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--ros-ci", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.prefix or any(
        token in args.prefix for token in ("/", "\\", "..")
    ):
        raise SystemExit("--prefix must be a safe filename prefix")
    try:
        summary, provenance, report = build_artifacts(
            args.suite_dir, args.ros_ci
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot publish CudaNav evidence: {error}") from error
    output = args.output_dir.resolve()
    targets = {
        output / f"{args.prefix}_summary.json": encoded(summary),
        output / f"{args.prefix}_provenance.json": encoded(provenance),
        output / f"{args.prefix}_report.md": report,
    }
    if args.check:
        stale = [
            str(path)
            for path, content in targets.items()
            if not path.is_file()
            or path.read_text(encoding="utf-8") != content
        ]
        if stale:
            print("stale CudaNav systems artifacts: " + ", ".join(stale))
            return 1
        print("CudaNav systems artifacts are current")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    for path, content in targets.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
