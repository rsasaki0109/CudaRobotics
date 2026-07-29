#!/usr/bin/env python3
"""Validate a portable bundle of every v0.2.0 release-candidate artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from v0_2_release_evidence import evaluate_release


MODE = "v0_2_release_evidence_bundle"
VERSION = "0.2.0"
PATHS = {
    "cpu_preflight": "evidence/cpu_preflight/manifest.json",
    "gpu_preflight": "evidence/gpu_preflight/manifest.json",
    "github_build": "evidence/ci/github_build.json",
    "python_manylinux_wheels": "evidence/ci/python_package.json",
    "ros2_cuda_mppi": "evidence/ci/ros_jazzy.json",
    "python_artifacts": "dist/python_artifacts.json",
    "real_rosbag_negative": "evidence/real_rosbag_negative.md",
}
REQUIRED_CATEGORIES = {
    "cpu_preflight",
    "gpu_preflight",
    "remote_ci",
    "distribution",
    "python_artifact_manifest",
    "negative_result",
    "release_gate",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _safe_file(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    path = (root / relative).resolve()
    return path if path.is_relative_to(root) and path.is_file() else None


def evaluate_bundle(
    bundle: dict[str, Any],
    bundle_root: Path,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    root = bundle_root.resolve()
    entries = bundle.get("files")
    file_checks: dict[str, bool] = {}
    categories: set[str] = set()
    declared_paths: list[str] = []
    if isinstance(entries, list):
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                file_checks[f"entry-{index}"] = False
                continue
            relative = entry.get("path")
            key = str(relative)
            path = _safe_file(root, relative)
            valid = (
                path is not None
                and isinstance(entry.get("bytes"), int)
                and entry["bytes"] == path.stat().st_size
                and bool(
                    re.fullmatch(
                        r"[0-9a-f]{64}", str(entry.get("sha256", ""))
                    )
                )
                and sha256_file(path) == entry["sha256"]
                and isinstance(entry.get("category"), str)
                and bool(entry["category"])
            )
            file_checks[key] = valid
            if isinstance(relative, str):
                declared_paths.append(relative)
            if isinstance(entry.get("category"), str):
                categories.add(entry["category"])
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() != "bundle.json"
    }
    path_set = set(declared_paths)

    release_reference = bundle.get("release_gate")
    release_gate_path = _safe_file(
        root,
        release_reference.get("path")
        if isinstance(release_reference, dict)
        else None,
    )
    stored_gate: dict[str, Any] = {}
    if release_gate_path is not None:
        try:
            stored_gate = read_object(release_gate_path)
        except (json.JSONDecodeError, OSError, UnicodeError, ValueError):
            stored_gate = {}

    recomputed: dict[str, Any] = {"passed": False, "checks": {}}
    try:
        recomputed = evaluate_release(
            expected_commit=str(bundle.get("git_commit", "")),
            cpu_preflight_dir=root / "evidence/cpu_preflight",
            gpu_preflight_dir=root / "evidence/gpu_preflight",
            build_ci_path=root / PATHS["github_build"],
            python_ci_path=root / PATHS["python_manylinux_wheels"],
            ros_ci_path=root / PATHS["ros2_cuda_mppi"],
            python_artifacts_path=root / PATHS["python_artifacts"],
            dist_dir=root / "dist",
            rosbag_report_path=root / PATHS["real_rosbag_negative"],
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        pass

    source_bindings = False
    sources = stored_gate.get("sources")
    if isinstance(sources, dict) and set(sources) == set(PATHS):
        source_bindings = all(
            isinstance(sources[name], dict)
            and sources[name].get("path") == relative
            and (path := _safe_file(root, relative)) is not None
            and sources[name].get("sha256") == sha256_file(path)
            for name, relative in PATHS.items()
        )

    commit = bundle.get("git_commit")
    checks = {
        "schema": bundle.get("schema_version") == 1,
        "mode": bundle.get("evidence_mode") == MODE,
        "version": bundle.get("version") == VERSION,
        "status": bundle.get("status") == "ready",
        "git_commit": bool(re.fullmatch(r"[0-9a-f]{40}", str(commit)))
        and (expected_commit is None or commit == expected_commit),
        "file_table": isinstance(entries, list)
        and bool(entries)
        and len(declared_paths) == len(entries)
        and len(declared_paths) == len(path_set)
        and all(file_checks.values()),
        "complete_inventory": path_set == actual_paths,
        "categories": REQUIRED_CATEGORIES <= categories,
        "release_gate_reference": (
            release_gate_path is not None
            and isinstance(release_reference, dict)
            and release_reference.get("sha256") == sha256_file(release_gate_path)
        ),
        "release_gate_identity": (
            stored_gate.get("schema_version") == 1
            and stored_gate.get("evidence_mode") == "v0_2_release_gate"
            and stored_gate.get("status") == "ready"
            and stored_gate.get("git_commit") == commit
            and stored_gate.get("passed") is True
        ),
        "release_gate_recomputed": (
            recomputed.get("passed") is True
            and recomputed.get("status") == "ready"
            and stored_gate.get("checks") == recomputed.get("checks")
            and stored_gate.get("gates") == recomputed.get("gates")
            and stored_gate.get("remote") == recomputed.get("remote")
        ),
        "source_bindings": source_bindings,
    }
    return {
        "valid": all(checks.values()),
        "ready": all(checks.values()),
        "checks": checks,
        "file_checks": file_checks,
        "release_gate": recomputed,
    }


def load_bundle(
    bundle_path: Path, expected_commit: str | None = None
) -> dict[str, Any]:
    try:
        path = bundle_path.resolve()
        bundle = read_object(path)
    except (json.JSONDecodeError, OSError, UnicodeError, ValueError):
        return {
            "valid": False,
            "ready": False,
            "checks": {"bundle_readable": False},
            "file_checks": {},
            "release_gate": {"passed": False},
        }
    return evaluate_bundle(bundle, path.parent, expected_commit)
