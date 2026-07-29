#!/usr/bin/env python3
"""Aggregate every local and remote v0.2.0 release-candidate gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from cudanav_ros_ci_evidence import evaluate as evaluate_ros_ci
from python_source_provenance import expected_payload
from release_ci_evidence import evaluate as evaluate_release_ci
from release_preflight_evidence import evaluate_manifest as evaluate_preflight
from verify_python_release_artifacts import (
    sha256,
    validate_sdist,
    validate_wheel,
)


VERSION = "0.2.0"
PACKAGE = "cudarobotics"
ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def evaluate_python_artifacts(
    manifest: dict[str, Any],
    dist_dir: Path,
    expected_commit: str,
) -> dict[str, Any]:
    root = dist_dir.resolve()
    entries = manifest.get("artifacts")
    declared_names: list[str] = []
    schema = isinstance(entries, list)
    content_matches = True
    archive_structure = True
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                schema = False
                content_matches = False
                continue
            name = entry.get("name")
            declared_names.append(str(name))
            path = (root / str(name)).resolve()
            if (
                not isinstance(name, str)
                or not name
                or not path.is_relative_to(root)
                or not path.is_file()
                or not isinstance(entry.get("bytes"), int)
                or entry["bytes"] <= 0
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(entry.get("sha256", ""))
                )
            ):
                content_matches = False
                continue
            if (
                path.stat().st_size != entry["bytes"]
                or sha256(path) != entry["sha256"]
            ):
                content_matches = False
            try:
                if entry.get("kind") == "sdist":
                    validate_sdist(path, VERSION)
                elif entry.get("kind") == "wheel":
                    validate_wheel(path, VERSION)
                else:
                    archive_structure = False
            except (AssertionError, OSError, ValueError):
                archive_structure = False
    actual_names = {
        path.name
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name == f"{PACKAGE}-{VERSION}.tar.gz"
            or (
                path.name.startswith(f"{PACKAGE}-{VERSION}-")
                and path.suffix == ".whl"
            )
        )
    } if root.is_dir() else set()
    unique_names = (
        schema
        and len(declared_names) == len(set(declared_names))
        and "" not in declared_names
    )
    name_set = set(declared_names)
    cp310 = any(
        re.fullmatch(
            rf"{PACKAGE}-{re.escape(VERSION)}-cp310-cp310-"
            r".*manylinux.*x86_64\.whl",
            name,
        )
        for name in name_set
    )
    cp312 = any(
        re.fullmatch(
            rf"{PACKAGE}-{re.escape(VERSION)}-cp312-cp312-"
            r".*manylinux.*x86_64\.whl",
            name,
        )
        for name in name_set
    )
    checks = {
        "schema": manifest.get("schema_version") == 1 and schema,
        "package": manifest.get("package") == PACKAGE,
        "version": manifest.get("package_version") == VERSION,
        "git_commit": manifest.get("git_commit") == expected_commit,
        "clean_checkout": manifest.get("git_dirty") is False,
        "source_provenance": (
            manifest.get("source_provenance") == expected_payload()
        ),
        "unique_names": unique_names,
        "complete_directory": unique_names and name_set == actual_names,
        "sdist": f"{PACKAGE}-{VERSION}.tar.gz" in name_set,
        "manylinux_cp310": cp310,
        "manylinux_cp312": cp312,
        "content_unchanged": content_matches,
        "archive_structure": archive_structure,
    }
    return {"passed": all(checks.values()), "checks": checks}


def evaluate_rosbag_negative_report(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    checks = {
        "exists": bool(text),
        "recorded_motion_label": (
            "recorded-motion evidence, not a closed-loop" in text
        ),
        "explicit_failure": "Overall result: **FAIL**" in text,
        "pairing_failure": (
            "FAIL: at least 90% scan/command pairing coverage" in text
        ),
        "clearance_failure": (
            "FAIL: minimum front clearance at least 0.10 m" in text
        ),
        "public_source": "https://doi.org/10.5281/zenodo.10518775" in text,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sha256": file_sha256(path) if path.is_file() else None,
    }


def evaluate_release(
    *,
    expected_commit: str,
    cpu_preflight_dir: Path,
    gpu_preflight_dir: Path,
    build_ci_path: Path,
    python_ci_path: Path,
    ros_ci_path: Path,
    python_artifacts_path: Path,
    dist_dir: Path,
    rosbag_report_path: Path,
) -> dict[str, Any]:
    cpu_manifest_path = cpu_preflight_dir / "manifest.json"
    gpu_manifest_path = gpu_preflight_dir / "manifest.json"
    cpu_manifest = read_json(cpu_manifest_path)
    gpu_manifest = read_json(gpu_manifest_path)
    build_ci = read_json(build_ci_path)
    python_ci = read_json(python_ci_path)
    ros_ci = read_json(ros_ci_path)
    python_artifacts = read_json(python_artifacts_path)

    cpu_gate = evaluate_preflight(
        cpu_manifest,
        cpu_preflight_dir,
        expected_profile="cpu",
        expected_commit=expected_commit,
    )
    gpu_gate = evaluate_preflight(
        gpu_manifest,
        gpu_preflight_dir,
        expected_profile="gpu",
        expected_commit=expected_commit,
    )
    build_gate = evaluate_release_ci(
        build_ci,
        expected_gate="github_build",
        expected_commit=expected_commit,
    )
    python_ci_gate = evaluate_release_ci(
        python_ci,
        expected_gate="python_manylinux_wheels",
        expected_commit=expected_commit,
    )
    ros_gate = evaluate_ros_ci(ros_ci, expected_commit=expected_commit)
    artifacts_gate = evaluate_python_artifacts(
        python_artifacts, dist_dir, expected_commit
    )
    rosbag_gate = evaluate_rosbag_negative_report(rosbag_report_path)
    ci_artifact_manifest = python_ci.get("artifact_manifest", {})
    artifact_manifest_binding = (
        isinstance(ci_artifact_manifest, dict)
        and ci_artifact_manifest.get("name") == python_artifacts_path.name
        and ci_artifact_manifest.get("bytes")
        == python_artifacts_path.stat().st_size
        and ci_artifact_manifest.get("sha256")
        == file_sha256(python_artifacts_path)
    )

    remote_payloads = [build_ci, python_ci, ros_ci]
    remote_refs = {
        str(payload.get("github", {}).get("ref", ""))
        for payload in remote_payloads
    }
    remote_run_ids = {
        payload.get("github", {}).get("run_id")
        for payload in remote_payloads
    }
    checks = {
        "expected_commit": bool(
            re.fullmatch(r"[0-9a-f]{40}", expected_commit)
        ),
        "cpu_preflight": cpu_gate["passed"],
        "gpu_preflight": gpu_gate["passed"],
        "github_build": build_gate["passed"],
        "python_manylinux_wheels": python_ci_gate["passed"],
        "ros2_cuda_mppi": ros_gate["passed"],
        "python_artifacts": artifacts_gate["passed"],
        "python_ci_artifact_binding": artifact_manifest_binding,
        "real_rosbag_explicit_negative": rosbag_gate["passed"],
        "same_remote_ref": len(remote_refs) == 1 and "" not in remote_refs,
        "distinct_remote_runs": (
            len(remote_run_ids) == 3 and None not in remote_run_ids
        ),
    }
    source_paths = {
        "cpu_preflight": cpu_manifest_path,
        "gpu_preflight": gpu_manifest_path,
        "github_build": build_ci_path,
        "python_manylinux_wheels": python_ci_path,
        "ros2_cuda_mppi": ros_ci_path,
        "python_artifacts": python_artifacts_path,
        "real_rosbag_negative": rosbag_report_path,
    }
    return {
        "schema_version": 1,
        "evidence_mode": "v0_2_release_gate",
        "status": "ready" if all(checks.values()) else "not_ready",
        "git_commit": expected_commit,
        "passed": all(checks.values()),
        "checks": checks,
        "gates": {
            "cpu_preflight": cpu_gate,
            "gpu_preflight": gpu_gate,
            "github_build": build_gate,
            "python_manylinux_wheels": python_ci_gate,
            "ros2_cuda_mppi": ros_gate,
            "python_artifacts": artifacts_gate,
            "real_rosbag_negative": rosbag_gate,
        },
        "remote": {
            "ref": next(iter(remote_refs)) if len(remote_refs) == 1 else None,
            "run_ids": sorted(
                run_id for run_id in remote_run_ids if isinstance(run_id, int)
            ),
        },
        "sources": {
            name: {
                "path": portable_path(path),
                "sha256": file_sha256(path) if path.is_file() else None,
            }
            for name, path in source_paths.items()
        },
    }
