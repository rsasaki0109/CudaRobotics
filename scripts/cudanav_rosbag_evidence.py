#!/usr/bin/env python3
"""Evidence schema and gates for reproducible CudaNav rosbag replay."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


PROFILES = {
    "smoke": {
        "minimum_duration_sec": 5.0,
        "minimum_diagnostics_samples": 10,
        "require_recording": False,
    },
    "release": {
        "minimum_duration_sec": 60.0,
        "minimum_diagnostics_samples": 100,
        "require_recording": True,
    },
}

REQUIRED_CUDANAV_OUTPUT_TOPICS = (
    "/cuda_nav/cmd_vel",
    "/cuda_nav/odom",
    "/cuda_nav/occupancy",
    "/cuda_nav/esdf",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_input(path: Path) -> dict[str, Any]:
    """Return a deterministic content identity for a bag file or directory."""
    source = path.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_symlink():
        raise ValueError("input bag root must not be a symbolic link")
    descendants = [] if source.is_file() else list(source.rglob("*"))
    if any(candidate.is_symlink() for candidate in descendants):
        raise ValueError("input bag must not contain symbolic links")
    files = [source] if source.is_file() else sorted(
        candidate for candidate in descendants if candidate.is_file()
    )
    if not files:
        raise ValueError("input bag has no regular files")
    root = source.parent if source.is_file() else source
    entries = []
    tree_digest = hashlib.sha256()
    for candidate in files:
        relative = candidate.relative_to(root).as_posix()
        size = candidate.stat().st_size
        digest = sha256_file(candidate)
        encoded = f"{relative}\0{size}\0{digest}\n".encode("utf-8")
        tree_digest.update(encoded)
        entries.append({"path": relative, "bytes": size, "sha256": digest})
    return {
        "source": str(source),
        "tree_sha256": tree_digest.hexdigest(),
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }


def rosbag_topic_counts(metadata_path: Path) -> dict[str, int]:
    """Read topic message counts from rosbag2 metadata without a YAML dependency."""
    counts: dict[str, int] = {}
    current_name: str | None = None
    for raw_line in metadata_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("name:"):
            current_name = stripped.split(":", 1)[1].strip().strip("\"'")
        elif current_name is not None and stripped.startswith("message_count:"):
            value = stripped.split(":", 1)[1].strip()
            try:
                counts[current_name] = int(value)
            except ValueError:
                counts[current_name] = -1
            current_name = None
    return counts


def _artifact(root: Path, relative: Any, *, directory: bool = False) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        return None
    if directory:
        return candidate if candidate.is_dir() else None
    return candidate if candidate.is_file() and candidate.stat().st_size > 0 else None


def _finite_at_least(value: Any, threshold: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= threshold
    )


def evaluate_manifest(
    manifest: dict[str, Any],
    run_directory: Path,
    profile: str,
    *,
    verify_source: bool = True,
) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"unknown rosbag evidence profile: {profile}")
    policy = PROFILES[profile]
    root = run_directory.resolve()
    checks: dict[str, bool] = {
        "manifest_schema": manifest.get("schema_version") == 1,
        "profile_matches": manifest.get("profile") == profile,
        "evidence_mode": manifest.get("evidence_mode")
        in {
            "shadow_controller_with_recorded_motion",
            "real_sensor_shadow_with_derived_path",
        },
        "git_commit_recorded": bool(
            re.fullmatch(r"[0-9a-fA-F]{40,64}", str(manifest.get("git_commit", "")))
        ),
        "clean_worktree": manifest.get("git_dirty") is False,
        "gpu_identity_recorded": (
            isinstance(manifest.get("gpu"), list)
            and bool(manifest["gpu"])
            and all(
                isinstance(item, dict)
                and all(
                    isinstance(item.get(field), str) and item[field]
                    for field in (
                        "physical_index",
                        "name",
                        "uuid",
                        "driver_version",
                        "memory_total_mib",
                    )
                )
                for item in manifest["gpu"]
            )
        ),
        "no_launch_errors": manifest.get("launch_errors") == {},
    }
    returncodes = manifest.get("returncodes")
    expected_stop_codes = {None, 0, -2, 130, -15, 143}
    checks["process_returncodes"] = (
        isinstance(returncodes, dict)
        and returncodes.get("evaluate") == 0
        and all(
            returncodes.get(name) in expected_stop_codes
            for name in ("controller", "record", "play")
        )
    )
    source = manifest.get("input_bag")
    derived_mode = (
        manifest.get("evidence_mode")
        == "real_sensor_shadow_with_derived_path"
    )
    checks["input_identity_schema"] = (
        isinstance(source, dict)
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(source.get("tree_sha256", ""))))
        and isinstance(source.get("file_count"), int)
        and source["file_count"] > 0
        and isinstance(source.get("total_bytes"), int)
        and source["total_bytes"] > 0
        and isinstance(source.get("files"), list)
        and len(source["files"]) == source["file_count"]
    )
    checks["input_content_unchanged"] = False
    if checks["input_identity_schema"]:
        if not verify_source:
            checks["input_content_unchanged"] = True
        else:
            try:
                current = describe_input(Path(source["source"]))
                checks["input_content_unchanged"] = (
                    current["tree_sha256"] == source["tree_sha256"]
                    and current["file_count"] == source["file_count"]
                    and current["total_bytes"] == source["total_bytes"]
                )
            except (FileNotFoundError, OSError, TypeError, ValueError):
                checks["input_content_unchanged"] = False
    derived = manifest.get("derived_path_bag")
    checks["derived_path_contract"] = not derived_mode
    if derived_mode:
        checks["derived_path_contract"] = (
            isinstance(derived, dict)
            and bool(
                re.fullmatch(
                    r"[0-9a-f]{64}", str(derived.get("tree_sha256", ""))
                )
            )
            and isinstance(derived.get("file_count"), int)
            and derived["file_count"] >= 2
            and isinstance(derived.get("total_bytes"), int)
            and derived["total_bytes"] > 0
        )
        if checks["derived_path_contract"] and verify_source:
            try:
                current_derived = describe_input(Path(derived["source"]))
                checks["derived_path_contract"] = (
                    current_derived["tree_sha256"] == derived["tree_sha256"]
                    and current_derived["file_count"] == derived["file_count"]
                    and current_derived["total_bytes"] == derived["total_bytes"]
                )
            except (OSError, TypeError, ValueError):
                checks["derived_path_contract"] = False

    artifacts = manifest.get("artifacts")
    checks["artifact_table"] = isinstance(artifacts, dict)
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    evaluation_path = _artifact(root, artifacts.get("evaluation"))
    diagnostics_path = _artifact(root, artifacts.get("diagnostics"))
    config_path = _artifact(root, artifacts.get("controller_config"))
    launch_log = _artifact(root, artifacts.get("controller_log"))
    play_log = _artifact(root, artifacts.get("play_log"))
    dataset_path = _artifact(root, artifacts.get("dataset_materialization"))
    checks.update(
        {
            "artifact_evaluation": evaluation_path is not None,
            "artifact_diagnostics": diagnostics_path is not None,
            "artifact_controller_config": config_path is not None,
            "artifact_controller_log": launch_log is not None,
            "artifact_play_log": play_log is not None,
            "dataset_materialization": (
                not derived_mode
                or (
                    dataset_path is not None
                    and sha256_file(dataset_path)
                    == manifest.get("dataset_materialization_sha256")
                )
            ),
        }
    )
    checks["dataset_materialization_semantics"] = not derived_mode
    if derived_mode and dataset_path is not None:
        try:
            from cudanav_real_dataset import DEFAULT_SPEC
            from validate_cudanav_real_dataset import evaluate as evaluate_dataset

            dataset_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
            dataset_gate = evaluate_dataset(DEFAULT_SPEC, dataset_path)
            checks["dataset_materialization_semantics"] = (
                dataset_gate["ready"]
                and dataset_payload["source_bag"]["tree_sha256"]
                == source["tree_sha256"]
                and dataset_payload["derived_path_bag"]["tree_sha256"]
                == derived["tree_sha256"]
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            checks["dataset_materialization_semantics"] = False
    checks["config_sha256_matches"] = (
        config_path is not None
        and sha256_file(config_path)
        == str(manifest.get("controller_config_sha256", "")).lower()
    )
    checks["diagnostics_sha256_matches"] = (
        diagnostics_path is not None
        and sha256_file(diagnostics_path)
        == str(manifest.get("diagnostics_sha256", "")).lower()
    )
    checks["evaluation_sha256_matches"] = (
        evaluation_path is not None
        and sha256_file(evaluation_path)
        == str(manifest.get("evaluation_sha256", "")).lower()
    )

    evaluation: dict[str, Any] = {}
    if evaluation_path is not None:
        try:
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            evaluation = {}
    diagnostics = evaluation.get("diagnostics")
    evaluation_database = manifest.get("evaluation_database")
    database_path: Path | None = None
    checks["evaluation_database_identity"] = False
    if isinstance(evaluation_database, dict):
        try:
            database_path = Path(evaluation_database["source"]).resolve()
            expected_entry = next(
                (
                    entry
                    for entry in source.get("files", [])
                    if isinstance(entry, dict)
                    and entry.get("path") == evaluation_database.get("relative_path")
                    and entry.get("sha256") == evaluation_database.get("sha256")
                ),
                None,
            )
            if verify_source:
                source_path = Path(source["source"]).resolve()
                source_root = (
                    source_path if source_path.is_dir() else source_path.parent
                )
                checks["evaluation_database_identity"] = (
                    database_path.is_file()
                    and database_path.is_relative_to(source_root)
                    and expected_entry is not None
                    and expected_entry.get("bytes") == database_path.stat().st_size
                    and sha256_file(database_path)
                    == evaluation_database.get("sha256")
                )
            else:
                checks["evaluation_database_identity"] = expected_entry is not None
        except (KeyError, OSError, TypeError):
            checks["evaluation_database_identity"] = False
    reported_motion_database = (
        evaluation.get("motion", {}).get("database")
        if isinstance(evaluation.get("motion"), dict)
        else None
    )
    reported_clearance_database = (
        evaluation.get("clearance", {}).get("database")
        if isinstance(evaluation.get("clearance"), dict)
        else None
    )
    reported_diagnostics_source = (
        diagnostics.get("source") if isinstance(diagnostics, dict) else None
    )
    try:
        checks["evaluation_inputs_bound"] = (
            database_path is not None
            and Path(reported_motion_database).resolve() == database_path
            and Path(reported_clearance_database).resolve() == database_path
            and diagnostics_path is not None
            and Path(reported_diagnostics_source).resolve() == diagnostics_path
        )
    except (TypeError, OSError):
        checks["evaluation_inputs_bound"] = False
    checks.update(
        {
            "evaluation_schema": evaluation.get("schema_version") == 1,
            "evaluation_quality_pass": evaluation.get("quality_pass") is True,
            "evaluation_mode": (
                evaluation.get("evidence_mode")
                == (
                    "real_sensor_shadow_with_derived_path"
                    if derived_mode
                    else "shadow_controller_with_recorded_motion"
                )
            ),
            "minimum_duration": _finite_at_least(
                evaluation.get("motion", {}).get("duration_s")
                if isinstance(evaluation.get("motion"), dict)
                else None,
                policy["minimum_duration_sec"],
            ),
            "diagnostics_coverage": (
                isinstance(diagnostics, dict)
                and isinstance(diagnostics.get("samples"), int)
                and diagnostics["samples"] >= policy["minimum_diagnostics_samples"]
            ),
        }
    )
    checks["pointcloud_evaluation_contract"] = not derived_mode
    if derived_mode:
        try:
            from cudanav_real_dataset import DEFAULT_SPEC, read_json

            quality = read_json(DEFAULT_SPEC)["quality_evaluation"]
            checks["pointcloud_evaluation_contract"] = (
                isinstance(evaluation.get("clearance"), dict)
                and evaluation["clearance"].get("pointcloud_topic")
                == quality["pointcloud_topic"]
                and evaluation["clearance"].get("filter") == quality["filter"]
                and evaluation["clearance"].get("diagnostics_source")
                == str(diagnostics_path)
            )
        except (KeyError, OSError, TypeError, ValueError):
            checks["pointcloud_evaluation_contract"] = False

    recording = _artifact(root, artifacts.get("recording"), directory=True)
    if policy["require_recording"]:
        checks["artifact_recording"] = (
            recording is not None
            and (recording / "metadata.yaml").is_file()
            and (recording / "metadata.yaml").stat().st_size > 0
        )
        record_topics = manifest.get("record_topics")
        required_topics = manifest.get("required_output_topics")
        checks["record_topics_recorded"] = (
            isinstance(record_topics, list)
            and bool(record_topics)
            and len(record_topics) == len(set(record_topics))
            and all(isinstance(topic, str) and topic for topic in record_topics)
        )
        checks["required_output_topics_declared"] = (
            isinstance(required_topics, list)
            and set(required_topics) == set(REQUIRED_CUDANAV_OUTPUT_TOPICS)
        )
        checks["required_output_topics_recorded"] = (
            checks["record_topics_recorded"]
            and checks["required_output_topics_declared"]
            and set(required_topics) <= set(record_topics)
        )
        recording_identity = manifest.get("recording_identity")
        checks["recording_identity_schema"] = (
            isinstance(recording_identity, dict)
            and bool(
                re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(recording_identity.get("tree_sha256", "")),
                )
            )
            and isinstance(recording_identity.get("file_count"), int)
            and recording_identity["file_count"] >= 2
            and isinstance(recording_identity.get("total_bytes"), int)
            and recording_identity["total_bytes"] > 0
        )
        checks["recording_content_unchanged"] = False
        checks["required_output_topic_messages"] = False
        if checks["artifact_recording"]:
            try:
                current_recording = describe_input(recording)
                checks["recording_content_unchanged"] = (
                    checks["recording_identity_schema"]
                    and current_recording["tree_sha256"]
                    == recording_identity["tree_sha256"]
                    and current_recording["file_count"]
                    == recording_identity["file_count"]
                    and current_recording["total_bytes"]
                    == recording_identity["total_bytes"]
                )
                topic_counts = rosbag_topic_counts(
                    recording / "metadata.yaml"
                )
                checks["required_output_topic_messages"] = (
                    checks["required_output_topics_declared"]
                    and all(
                        topic_counts.get(topic, 0) > 0
                        for topic in required_topics
                    )
                )
            except (OSError, TypeError, ValueError):
                pass
    elif artifacts.get("recording"):
        checks["artifact_recording"] = recording is not None

    commands = manifest.get("commands")
    checks["commands_recorded"] = (
        isinstance(commands, dict)
        and all(
            isinstance(commands.get(name), list)
            and commands[name]
            and all(isinstance(token, str) and token for token in commands[name])
            for name in ("controller", "play", "evaluate")
        )
    )
    if checks["commands_recorded"]:
        controller_text = "\0".join(commands["controller"])
        checks["controller_inputs_bound"] = (
            config_path is not None
            and diagnostics_path is not None
            and str(config_path) in controller_text
            and str(diagnostics_path) in controller_text
        )
        checks["play_input_bound"] = (
            isinstance(source, dict)
            and str(Path(source.get("source", "")).resolve()) in commands["play"]
            and (
                not derived_mode
                or (
                    isinstance(derived, dict)
                    and str(Path(derived.get("source", "")).resolve())
                    in commands["play"]
                    and commands["play"].count("-i") == 2
                )
            )
        )
        checks["evaluate_inputs_bound"] = (
            database_path is not None
            and diagnostics_path is not None
            and str(database_path) in commands["evaluate"]
            and str(diagnostics_path) in commands["evaluate"]
        )
        checks["pointcloud_evaluate_command_bound"] = not derived_mode
        if derived_mode:
            try:
                from cudanav_real_dataset import DEFAULT_SPEC, read_json

                spec = read_json(DEFAULT_SPEC)
                expected_options = {
                    "--pointcloud-topic": spec["recorded_inputs"][
                        "pointcloud"
                    ]["topic"],
                    "--odometry-topic": spec["recorded_inputs"]["odometry"][
                        "topic"
                    ],
                    "--pointcloud-half-angle-rad": str(
                        spec["quality_evaluation"]["filter"]["half_angle_rad"]
                    ),
                    "--pointcloud-minimum-z-m": str(
                        spec["quality_evaluation"]["filter"]["minimum_z_m"]
                    ),
                    "--pointcloud-maximum-z-m": str(
                        spec["quality_evaluation"]["filter"]["maximum_z_m"]
                    ),
                    "--pointcloud-minimum-range-m": str(
                        spec["quality_evaluation"]["filter"]["minimum_range_m"]
                    ),
                    "--pointcloud-maximum-range-m": str(
                        spec["quality_evaluation"]["filter"]["maximum_range_m"]
                    ),
                    "--pointcloud-maximum-command-age-ms": str(
                        spec["quality_evaluation"]["filter"][
                            "maximum_command_age_ms"
                        ]
                    ),
                }
                command = commands["evaluate"]
                checks["pointcloud_evaluate_command_bound"] = all(
                    option in command
                    and command.index(option) + 1 < len(command)
                    and command[command.index(option) + 1] == value
                    for option, value in expected_options.items()
                )
            except (KeyError, OSError, TypeError, ValueError):
                checks["pointcloud_evaluate_command_bound"] = False
    else:
        checks["controller_inputs_bound"] = False
        checks["play_input_bound"] = False
        checks["evaluate_inputs_bound"] = False
        checks["pointcloud_evaluate_command_bound"] = False
    if policy["require_recording"]:
        record_command = (
            commands.get("record") if isinstance(commands, dict) else None
        )
        checks["record_command_bound"] = (
            isinstance(record_command, list)
            and recording is not None
            and str(recording) in record_command
            and isinstance(manifest.get("record_topics"), list)
            and set(manifest["record_topics"]) <= set(record_command)
        )
    return {
        "profile": profile,
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": policy,
    }
