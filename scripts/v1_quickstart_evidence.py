#!/usr/bin/env python3
"""Validate fresh-clone, no-cache Docker CudaNav quickstart evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shlex
from typing import Any


REQUIRED_ARTIFACTS = {
    "clone.log",
    "docker_build.log",
    "docker_run.log",
    "result/cudanav_closed_loop.json",
    "result/cudanav_closed_loop.log",
    "support_matrix.json",
}


def evaluate_matrix_snapshot(
    matrix: dict[str, Any],
    component_versions: Any,
) -> dict[str, Any]:
    surfaces = matrix.get("surfaces", {})
    main = matrix.get("main_demo", {})
    if not isinstance(surfaces, dict):
        surfaces = {}
    if not isinstance(main, dict):
        main = {}
    python_source = surfaces.get("python_source", {})
    python_wheels = surfaces.get("python_wheels", {})
    ros2 = surfaces.get("ros2", {})
    if not isinstance(python_source, dict):
        python_source = {}
    if not isinstance(python_wheels, dict):
        python_wheels = {}
    if not isinstance(ros2, dict):
        ros2 = {}
    python_version = (
        component_versions.get("python_version")
        if isinstance(component_versions, dict)
        else None
    )
    ros_versions = (
        component_versions.get("ros_package_versions")
        if isinstance(component_versions, dict)
        else None
    )
    checks = {
        "schema": matrix.get("schema_version") == 1,
        "status": matrix.get("status") in {"development", "release"},
        "target_version": matrix.get("target_version") == "1.0.0",
        "target_tag": matrix.get("target_tag") == "v1.0.0",
        "surfaces": isinstance(surfaces, dict),
        "main_surface": main.get("surface") == "docker_source",
        "time_budget": main.get("time_budget_seconds") == 900,
        "build_command": isinstance(main.get("build_command"), str)
        and bool(main["build_command"]),
        "run_command": isinstance(main.get("run_command"), str)
        and bool(main["run_command"]),
        "result": main.get("result") == "out/cudanav_closed_loop.json"
        and main.get("gate") == "smoke_pass",
        "component_schema": isinstance(python_version, str)
        and bool(python_version)
        and isinstance(ros_versions, dict)
        and bool(ros_versions)
        and all(
            isinstance(package, str)
            and bool(package)
            and isinstance(version, str)
            and bool(version)
            for package, version in ros_versions.items()
        ),
        "python_versions": python_source.get("version") == python_version
        and python_wheels.get("version") == python_version,
        "ros_versions": ros2.get("package_versions") == ros_versions,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "actual": component_versions,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_artifacts(
    root: Path, relative_paths: set[str]
) -> list[dict[str, Any]]:
    entries = []
    for relative in sorted(relative_paths):
        path = (root / relative).resolve()
        if path.is_file():
            entries.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return entries


def evaluate_manifest(
    manifest: dict[str, Any],
    directory: Path,
    *,
    expected_profile: str | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    root = directory.resolve()
    profile = manifest.get("profile")
    target = manifest.get("target_version")
    commit = manifest.get("git_commit")
    artifacts = manifest.get("artifacts")
    matrix_path = root / "support_matrix.json"
    component_versions = manifest.get("component_versions")
    try:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        matrix_gate = (
            evaluate_matrix_snapshot(matrix, component_versions)
            if isinstance(matrix, dict)
            else {"valid": False, "actual": {}}
        )
    except (OSError, ValueError, json.JSONDecodeError):
        matrix = {}
        matrix_gate = {"valid": False, "actual": {}}

    declared_paths: list[str] = []
    artifact_content = True
    if isinstance(artifacts, list):
        for entry in artifacts:
            if not isinstance(entry, dict):
                artifact_content = False
                continue
            relative = entry.get("path")
            declared_paths.append(str(relative))
            path = (root / str(relative)).resolve()
            if (
                not isinstance(relative, str)
                or not relative
                or not path.is_relative_to(root)
                or not path.is_file()
                or not isinstance(entry.get("bytes"), int)
                or entry["bytes"] <= 0
                or path.stat().st_size != entry["bytes"]
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(entry.get("sha256", ""))
                )
                or sha256_file(path) != entry["sha256"]
            ):
                artifact_content = False

    summary: dict[str, Any] = {}
    summary_path = root / "result" / "cudanav_closed_loop.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass

    commands = manifest.get("commands", {})
    if not isinstance(commands, dict):
        commands = {}
    build_command = commands.get("build")
    run_command = commands.get("run")
    clone_command = commands.get("clone")
    matrix_main = matrix.get("main_demo", {}) if isinstance(matrix, dict) else {}
    if not isinstance(matrix_main, dict):
        matrix_main = {}
    returncodes = manifest.get("returncodes", {})
    phase_seconds = manifest.get("phase_seconds", {})
    gpu = manifest.get("gpu")
    docker = manifest.get("docker", {})
    if not isinstance(returncodes, dict):
        returncodes = {}
    if not isinstance(phase_seconds, dict):
        phase_seconds = {}
    if not isinstance(docker, dict):
        docker = {}
    matrix_actual = matrix_gate.get("actual", {})
    if not isinstance(matrix_actual, dict):
        matrix_actual = {}
    matrix_ros_versions = matrix_actual.get("ros_package_versions", {})
    if not isinstance(matrix_ros_versions, dict):
        matrix_ros_versions = {}
    elapsed = manifest.get("duration_seconds")
    checks = {
        "schema": manifest.get("schema_version") == 1,
        "evidence_mode": manifest.get("evidence_mode")
        == "v1_quickstart",
        "status": manifest.get("status") == "passed",
        "profile": profile in {"development", "release"}
        and (expected_profile is None or profile == expected_profile),
        "target_version": target == "1.0.0",
        "git_commit": isinstance(commit, str)
        and bool(re.fullmatch(r"[0-9a-f]{40}", commit))
        and (expected_commit is None or commit == expected_commit),
        "clean_clone": manifest.get("git_dirty") is False,
        "source_ref": isinstance(manifest.get("source_ref"), str)
        and bool(manifest["source_ref"]),
        "repository": manifest.get("repository")
        == "https://github.com/rsasaki0109/CudaRobotics.git",
        "fresh_image": manifest.get("preexisting_image") is False,
        "fresh_container": manifest.get("preexisting_container") is False,
        "time_budget": manifest.get("time_budget_seconds")
        == matrix_main.get("time_budget_seconds")
        == 900,
        "duration": isinstance(elapsed, (int, float))
        and 0 < elapsed <= matrix_main.get("time_budget_seconds", -1),
        "phase_times": isinstance(phase_seconds, dict)
        and {"clone", "build", "run"} <= set(phase_seconds)
        and all(
            isinstance(value, (int, float)) and value >= 0
            for value in phase_seconds.values()
        )
        and sum(phase_seconds.values()) <= elapsed + 1.0,
        "returncodes": isinstance(returncodes, dict)
        and all(returncodes.get(name) == 0 for name in ("clone", "build", "run")),
        "clone_command": isinstance(clone_command, list)
        and "--depth" in clone_command
        and "1" in clone_command
        and "--branch" in clone_command
        and manifest.get("source_ref") in clone_command
        and manifest.get("repository") in clone_command,
        "build_command": isinstance(build_command, list)
        and build_command[:4] == ["docker", "build", "--pull", "--no-cache"]
        and shlex.join(build_command).endswith(
            "-f docker/Dockerfile -t cudarobotics ."
        )
        and manifest.get("build_command_contract")
        == matrix_main.get("build_command"),
        "run_command": isinstance(run_command, list)
        and run_command[:5] == ["docker", "run", "--rm", "--gpus", "all"]
        and run_command[-2:] == ["cudarobotics", "cudanav"]
        and manifest.get("run_command_contract")
        == matrix_main.get("run_command"),
        "docker": isinstance(docker.get("engine_version"), str)
        and bool(docker["engine_version"])
        and bool(
            re.fullmatch(r"sha256:[0-9a-f]{64}", str(docker.get("image_id", "")))
        ),
        "gpu": isinstance(gpu, list)
        and bool(gpu)
        and all(
            isinstance(device, dict)
            and bool(device.get("name"))
            and bool(device.get("uuid"))
            and bool(device.get("driver_version"))
            for device in gpu
        ),
        "artifact_schema": isinstance(artifacts, list)
        and len(declared_paths) == len(artifacts)
        and len(declared_paths) == len(set(declared_paths)),
        "artifact_coverage": set(declared_paths) == REQUIRED_ARTIFACTS,
        "artifact_content": artifact_content,
        "matrix_hash": matrix_path.is_file()
        and manifest.get("support_matrix_sha256") == sha256_file(matrix_path),
        "matrix_valid": bool(matrix_gate.get("valid")),
        "component_versions": component_versions == matrix_actual,
        "result_contract": manifest.get("result")
        == matrix_main.get("result"),
        "result": summary.get("schema_version") == 1
        and summary.get("smoke_pass") is True
        and summary.get("success") is True,
    }
    release_checks = {
        "matrix_release_status": matrix.get("status") == "release",
        "source_tag": manifest.get("source_ref") == matrix.get("target_tag"),
        "python_at_target": matrix_actual.get("python_version") == target,
        "ros_at_target": set(matrix_ros_versions.values()) == {target},
    }
    if profile == "release":
        checks.update(release_checks)
    return {
        "profile": profile,
        "git_commit": commit,
        "passed": all(checks.values()),
        "checks": checks,
    }
