#!/usr/bin/env python3
"""Publish a validated CudaNav real-rosbag run without machine-local paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cudanav_rosbag_evidence import (
    REQUIRED_CUDANAV_OUTPUT_TOPICS,
    evaluate_manifest,
    rosbag_topic_counts,
    sha256_file,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def identity(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "tree_sha256": payload["tree_sha256"],
        "file_count": payload["file_count"],
        "total_bytes": payload["total_bytes"],
        "files": [
            {
                "path": item["path"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in payload["files"]
        ],
    }


def make_evidence(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    gate = evaluate_manifest(manifest, root, "release")
    if not gate["passed"]:
        raise ValueError("real-rosbag release gate did not pass")
    artifacts = manifest["artifacts"]
    evaluation_path = root / artifacts["evaluation"]
    diagnostics_path = root / artifacts["diagnostics"]
    materialization_path = root / artifacts["dataset_materialization"]
    evaluation = read_json(evaluation_path)
    materialization = read_json(materialization_path)
    recording = root / artifacts["recording"]
    topic_counts = rosbag_topic_counts(recording / "metadata.yaml")
    database = materialization["source_bag"]["files"]
    database = next(item for item in database if item["path"].endswith(".db3"))
    diagnostics = evaluation["diagnostics"]
    clearance = evaluation["clearance"]
    all_colliding_ratio = (
        diagnostics["all_colliding_cycles"] / diagnostics["samples"]
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "profile": "release",
        "source_commit": manifest["git_commit"],
        "git_dirty": manifest["git_dirty"],
        "evidence_mode": manifest["evidence_mode"],
        "claims": {
            "ros2_runtime": True,
            "real_sensor_data": True,
            "derived_recorded_path": True,
            "closed_loop": False,
            "commands_modify_recorded_motion": False,
        },
        "dataset": {
            "id": materialization["dataset_id"],
            "input": identity(manifest["input_bag"]),
            "database": database,
            "derived_path": identity(manifest["derived_path_bag"]),
            "materialization_sha256": sha256_file(materialization_path),
        },
        "gpu": manifest["gpu"],
        "metrics": {
            "diagnostics_samples": diagnostics["samples"],
            "diagnostics_duration_s": (
                float(
                    diagnostics_path.read_text(encoding="utf-8")
                    .splitlines()[-1]
                    .split(",", 1)[0]
                )
                - float(
                    diagnostics_path.read_text(encoding="utf-8")
                    .splitlines()[1]
                    .split(",", 1)[0]
                )
            ),
            "solve_mean_ms": diagnostics["solve_mean_ms"],
            "solve_p95_ms": diagnostics["solve_p95_ms"],
            "solve_max_ms": diagnostics["solve_max_ms"],
            "valid_rollout_ratio_mean": diagnostics[
                "valid_rollout_ratio_mean"
            ],
            "all_colliding_cycles": diagnostics["all_colliding_cycles"],
            "all_colliding_ratio": all_colliding_ratio,
            "retreat_cycles": diagnostics["retreat_cycles"],
            "pointcloud_window_samples": clearance[
                "evaluation_window_samples"
            ],
            "paired_command_samples": clearance["paired_command_samples"],
            "command_pair_ratio": clearance["command_pair_ratio"],
            "minimum_front_range_m": clearance["minimum_front_range_m"],
            "mean_front_clearance_m": clearance["mean_front_clearance_m"],
            "quality_pass": evaluation["quality_pass"],
        },
        "thresholds": evaluation["thresholds"],
        "output_recording": {
            **identity(manifest["recording_identity"]),
            "required_topic_messages": {
                topic: topic_counts[topic]
                for topic in REQUIRED_CUDANAV_OUTPUT_TOPICS
            },
        },
        "artifacts": {
            "manifest_sha256": sha256_file(manifest_path),
            "evaluation_sha256": sha256_file(evaluation_path),
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "controller_config_sha256": manifest[
                "controller_config_sha256"
            ],
        },
        "gate": gate,
        "limitations": evaluation["limitations"],
    }


def evaluate_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    claims = payload.get("claims", {})
    metrics = payload.get("metrics", {})
    recording = payload.get("output_recording", {})
    checks = {
        "schema": payload.get("schema_version") == 1,
        "status": payload.get("status") == "passed",
        "release_profile": payload.get("profile") == "release",
        "clean_commit": (
            isinstance(payload.get("source_commit"), str)
            and len(payload["source_commit"]) == 40
            and payload.get("git_dirty") is False
        ),
        "claim_boundary": (
            claims.get("ros2_runtime") is True
            and claims.get("real_sensor_data") is True
            and claims.get("derived_recorded_path") is True
            and claims.get("closed_loop") is False
            and claims.get("commands_modify_recorded_motion") is False
        ),
        "quality": metrics.get("quality_pass") is True,
        "coverage": (
            metrics.get("diagnostics_duration_s", 0.0) >= 60.0
            and metrics.get("diagnostics_samples", 0) >= 100
            and metrics.get("command_pair_ratio", 0.0) >= 0.9
        ),
        "required_outputs": all(
            recording.get("required_topic_messages", {}).get(topic, 0) > 0
            for topic in REQUIRED_CUDANAV_OUTPUT_TOPICS
        ),
        "release_gate": payload.get("gate", {}).get("passed") is True,
    }
    return {"valid": all(checks.values()), "checks": checks}


def render(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    gpu = payload["gpu"][0]
    counts = payload["output_recording"]["required_topic_messages"]
    lines = [
        "# CudaNav ROS 2 real-rosbag shadow release",
        "",
        f"- Source commit: `{payload['source_commit']}` (clean)",
        f"- Dataset: `{payload['dataset']['id']}`",
        f"- GPU: {gpu['name']} (`{gpu['uuid']}`)",
        "- Claim boundary: real sensor data with a derived recorded Path; "
        "commands do not alter recorded motion.",
        "",
        "## Results",
        "",
        f"- Diagnostics: {metrics['diagnostics_samples']} samples over "
        f"{metrics['diagnostics_duration_s']:.3f} s",
        f"- Solve latency: mean {metrics['solve_mean_ms']:.3f} ms, "
        f"p95 {metrics['solve_p95_ms']:.3f} ms",
        f"- Valid rollout ratio: {metrics['valid_rollout_ratio_mean']:.4f}",
        f"- Pointcloud pairing: {metrics['paired_command_samples']}/"
        f"{metrics['pointcloud_window_samples']} "
        f"({metrics['command_pair_ratio'] * 100.0:.2f}%)",
        f"- Front clearance: minimum {metrics['minimum_front_range_m']:.3f} m, "
        f"mean {metrics['mean_front_clearance_m']:.3f} m",
        f"- All-colliding recovery: {metrics['all_colliding_cycles']} cycles "
        f"({metrics['all_colliding_ratio'] * 100.0:.3f}%), "
        f"{metrics['retreat_cycles']} retreat cycles",
        "",
        "## Recorded CudaNav outputs",
        "",
        *[f"- `{topic}`: {counts[topic]} messages" for topic in counts],
        "",
        f"Overall release gate: **{'PASS' if payload['gate']['passed'] else 'FAIL'}**",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path, nargs="?")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--validate-portable", type=Path)
    args = parser.parse_args()
    if args.validate_portable:
        result = evaluate_evidence(read_json(args.validate_portable))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    if args.run_directory is None or args.output_json is None:
        parser.error("run_directory and --output-json are required")
    payload = make_evidence(args.run_directory)
    validation = evaluate_evidence(payload)
    if not validation["valid"]:
        raise SystemExit("portable evidence validation failed")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render(payload), encoding="utf-8")
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
