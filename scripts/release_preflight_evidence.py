#!/usr/bin/env python3
"""Validate content-bound local evidence for a v0.2 release candidate."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any


EXPECTED_EXTERNAL_GATES = {
    "github_build",
    "python_manylinux_wheels",
    "ros2_cuda_mppi",
    "closed_loop_rosbag_or_explicit_negative_result",
}

CPU_REQUIRED_CHECKS = {
    "version_consistency",
    "python_core_sync",
    "artifact_verifier_tests",
    "python_labelled_ctest",
    "python_package_tests",
    "whitespace",
    "python_release_artifacts",
    "clean_checkout",
}

GPU_REQUIRED_CHECKS = CPU_REQUIRED_CHECKS | {
    "registration_gpu_consistency",
    "registration_gpu_smoke",
}

GENERATED_OUTPUTS = {
    "python_release_artifacts": {"python_artifacts.json"},
    "registration_gpu_smoke": {
        "registration_smoke.csv",
        "registration_smoke.md",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_file(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve()).as_posix()
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def collect_evidence_files(
    output_dir: Path, checks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for check in checks:
        report_log = check.get("report_log")
        if isinstance(report_log, str) and report_log:
            paths.add(output_dir / report_log)
        for relative in GENERATED_OUTPUTS.get(str(check.get("name")), set()):
            paths.add(output_dir / relative)
    return [
        describe_file(path, output_dir)
        for path in sorted(paths, key=lambda item: item.as_posix())
        if path.is_file()
    ]


def _safe_file(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    return path


def evaluate_manifest(
    manifest: dict[str, Any],
    directory: Path,
    *,
    expected_profile: str | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    root = directory.resolve()
    profile = manifest.get("profile")
    checks_table = manifest.get("checks")
    evidence_files = manifest.get("evidence_files")
    check_names: list[str] = []
    if isinstance(checks_table, list):
        check_names = [
            str(check.get("name", ""))
            for check in checks_table
            if isinstance(check, dict)
        ]
    required_checks = (
        GPU_REQUIRED_CHECKS if profile == "gpu" else CPU_REQUIRED_CHECKS
    )
    checks: dict[str, bool] = {
        "schema": manifest.get("schema_version") == 1,
        "status": manifest.get("status") == "passed",
        "profile": profile in {"cpu", "gpu"}
        and (expected_profile is None or profile == expected_profile),
        "git_commit": bool(
            re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("git_commit", "")))
        )
        and (
            expected_commit is None
            or manifest.get("git_commit") == expected_commit
        ),
        "clean_checkout": manifest.get("git_dirty") is False,
        "check_table": (
            isinstance(checks_table, list)
            and len(check_names) == len(checks_table)
            and len(check_names) == len(set(check_names))
            and "" not in check_names
        ),
        "required_checks": required_checks <= set(check_names),
        "all_checks_passed": (
            isinstance(checks_table, list)
            and bool(checks_table)
            and all(
                isinstance(check, dict)
                and check.get("status") == "passed"
                and check.get("returncode") == 0
                for check in checks_table
            )
        ),
        "external_gates_explicit": (
            isinstance(manifest.get("external_gates"), list)
            and set(manifest["external_gates"]) == EXPECTED_EXTERNAL_GATES
            and len(manifest["external_gates"])
            == len(EXPECTED_EXTERNAL_GATES)
        ),
        "evidence_file_schema": isinstance(evidence_files, list),
        "evidence_complete": False,
        "evidence_content_unchanged": False,
    }

    required_paths: set[str] = set()
    if isinstance(checks_table, list):
        for check in checks_table:
            if not isinstance(check, dict):
                continue
            report_log = check.get("report_log")
            if isinstance(report_log, str) and report_log:
                required_paths.add(Path(report_log).as_posix())
            required_paths.update(
                GENERATED_OUTPUTS.get(str(check.get("name")), set())
            )

    declared_paths: list[str] = []
    content_matches = True
    if isinstance(evidence_files, list):
        for entry in evidence_files:
            if not isinstance(entry, dict):
                content_matches = False
                continue
            relative = entry.get("path")
            declared_paths.append(str(relative))
            path = _safe_file(root, relative)
            digest = str(entry.get("sha256", ""))
            size = entry.get("bytes")
            if (
                path is None
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not isinstance(size, int)
                or size < 0
                or path.stat().st_size != size
                or sha256_file(path) != digest
            ):
                content_matches = False
        checks["evidence_file_schema"] = (
            len(declared_paths) == len(evidence_files)
            and len(declared_paths) == len(set(declared_paths))
            and "" not in declared_paths
        )
        checks["evidence_complete"] = (
            checks["evidence_file_schema"]
            and set(declared_paths) == required_paths
        )
        checks["evidence_content_unchanged"] = (
            checks["evidence_file_schema"] and content_matches
        )

    return {
        "profile": profile,
        "git_commit": manifest.get("git_commit"),
        "passed": all(checks.values()),
        "checks": checks,
    }
