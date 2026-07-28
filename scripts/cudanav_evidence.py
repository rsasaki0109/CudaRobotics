#!/usr/bin/env python3
"""Schema and gate logic for deterministic CudaNav closed-loop evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class GatePolicy:
    min_elapsed_sec: float
    max_collisions: int
    max_goal_distance_m: float
    max_drift_percent: float
    max_deadline_miss_rate: float
    min_command_intervals: int
    require_bag: bool
    require_video: bool


POLICIES = {
    "smoke": GatePolicy(
        min_elapsed_sec=5.0,
        max_collisions=0,
        max_goal_distance_m=0.30,
        max_drift_percent=5.0,
        max_deadline_miss_rate=0.05,
        min_command_intervals=20,
        require_bag=False,
        require_video=False,
    ),
    "release": GatePolicy(
        min_elapsed_sec=600.0,
        max_collisions=0,
        max_goal_distance_m=0.25,
        max_drift_percent=1.0,
        max_deadline_miss_rate=0.01,
        min_command_intervals=1000,
        require_bag=True,
        require_video=True,
    ),
}


REQUIRED_SUMMARY_FIELDS = {
    "schema_version",
    "success",
    "elapsed_sec",
    "collision",
    "collision_count",
    "ground_truth_distance_m",
    "ground_truth_goal_distance_m",
    "odometry_position_error_m",
    "odometry_drift_percent",
    "command_intervals",
    "command_deadline_misses",
    "command_deadline_miss_rate",
}


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_summary(summary: dict[str, Any]) -> list[str]:
    errors = []
    missing = sorted(REQUIRED_SUMMARY_FIELDS - summary.keys())
    if missing:
        errors.append("missing summary fields: " + ", ".join(missing))
        return errors
    if summary["schema_version"] != 1:
        errors.append("unsupported summary schema_version")
    if not isinstance(summary["success"], bool):
        errors.append("success must be boolean")
    if not isinstance(summary["collision"], bool):
        errors.append("collision must be boolean")
    for field in (
        "elapsed_sec",
        "ground_truth_distance_m",
        "ground_truth_goal_distance_m",
        "odometry_position_error_m",
        "odometry_drift_percent",
        "command_deadline_miss_rate",
    ):
        if not _is_finite_number(summary[field]):
            errors.append(f"{field} must be finite")
    for field in (
        "collision_count",
        "command_intervals",
        "command_deadline_misses",
    ):
        value = summary[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{field} must be a non-negative integer")
    if (
        _is_finite_number(summary["command_deadline_miss_rate"])
        and not 0.0 <= summary["command_deadline_miss_rate"] <= 1.0
    ):
        errors.append("command_deadline_miss_rate must be in [0, 1]")
    for field in (
        "elapsed_sec",
        "ground_truth_distance_m",
        "ground_truth_goal_distance_m",
        "odometry_position_error_m",
        "odometry_drift_percent",
    ):
        if _is_finite_number(summary[field]) and summary[field] < 0.0:
            errors.append(f"{field} must be non-negative")
    if (
        isinstance(summary["command_deadline_misses"], int)
        and isinstance(summary["command_intervals"], int)
        and summary["command_deadline_misses"] > summary["command_intervals"]
    ):
        errors.append("command_deadline_misses cannot exceed command_intervals")
    return errors


def evaluate_summary(
    summary: dict[str, Any], profile: str
) -> dict[str, Any]:
    if profile not in POLICIES:
        raise ValueError(f"unknown CudaNav gate profile: {profile}")
    schema_errors = validate_summary(summary)
    if schema_errors:
        return {
            "profile": profile,
            "passed": False,
            "schema_errors": schema_errors,
            "checks": {},
        }
    policy = POLICIES[profile]
    checks = {
        "action_succeeded": bool(summary["success"]),
        "collision_flag_clear": not summary["collision"],
        "collision_count": summary["collision_count"] <= policy.max_collisions,
        "minimum_duration": summary["elapsed_sec"] >= policy.min_elapsed_sec,
        "goal_distance": (
            summary["ground_truth_goal_distance_m"]
            <= policy.max_goal_distance_m
        ),
        "odometry_drift": (
            summary["odometry_drift_percent"]
            < policy.max_drift_percent
        ),
        "command_samples": (
            summary["command_intervals"] >= policy.min_command_intervals
        ),
        "deadline_miss_rate": (
            summary["command_deadline_miss_rate"]
            < policy.max_deadline_miss_rate
        ),
    }
    return {
        "profile": profile,
        "passed": all(checks.values()),
        "schema_errors": [],
        "checks": checks,
        "thresholds": {
            "min_elapsed_sec": policy.min_elapsed_sec,
            "max_collisions": policy.max_collisions,
            "max_goal_distance_m": policy.max_goal_distance_m,
            "max_drift_percent": policy.max_drift_percent,
            "max_deadline_miss_rate": policy.max_deadline_miss_rate,
            "min_command_intervals": policy.min_command_intervals,
        },
    }


def evaluate_manifest(
    manifest: dict[str, Any],
    run_directory: Path,
    profile: str,
) -> dict[str, Any]:
    policy = POLICIES[profile]
    checks: dict[str, bool] = {
        "manifest_schema": manifest.get("schema_version") == 1,
        "git_commit_recorded": bool(
            re.fullmatch(r"[0-9a-fA-F]{40,64}", str(manifest.get("git_commit", "")))
        ),
        "clean_worktree": manifest.get("git_dirty") is False,
        "config_sha256_recorded": bool(
            re.fullmatch(
                r"[0-9a-fA-F]{64}", str(manifest.get("config_sha256", ""))
            )
        ),
        "gpu_identity_recorded": (
            isinstance(manifest.get("gpu"), list) and bool(manifest["gpu"])
        ),
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        checks["artifact_table"] = False
    else:
        checks["artifact_table"] = True

    root = run_directory.resolve()

    def artifact_exists(relative: Any, require_file: bool) -> bool:
        if not isinstance(relative, str) or not relative:
            return False
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root):
            return False
        return candidate.is_file() if require_file else candidate.exists()

    for artifact in ("summary", "launch_log", "controller_config"):
        relative = artifacts.get(artifact)
        checks[f"artifact_{artifact}"] = artifact_exists(relative, True)
    config_relative = artifacts.get("controller_config")
    if checks["artifact_controller_config"]:
        config_path = (root / str(config_relative)).resolve()
        config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        checks["config_sha256_matches"] = (
            config_digest.lower()
            == str(manifest.get("config_sha256", "")).lower()
        )
    else:
        checks["config_sha256_matches"] = False
    if policy.require_bag:
        relative = artifacts.get("rosbag")
        checks["artifact_rosbag"] = artifact_exists(relative, False)
    if policy.require_video:
        relative = artifacts.get("video")
        checks["artifact_video"] = artifact_exists(relative, True)
    return {
        "profile": profile,
        "passed": all(checks.values()),
        "checks": checks,
    }
