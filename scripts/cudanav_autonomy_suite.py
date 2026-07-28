#!/usr/bin/env python3
"""Independent aggregate gate for the complete CudaNav evidence suite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cudanav_evidence import (
    evaluate_manifest as evaluate_closed_loop_manifest,
    evaluate_summary,
)
from cudanav_multi_gpu import evaluate_multi_gpu_suite
from cudanav_rosbag_evidence import evaluate_manifest as evaluate_rosbag_manifest


MODE_MANIFESTS = {
    "closed_loop": "manifest.json",
    "real_rosbag_shadow": "manifest.json",
    "multi_gpu": "multi_gpu_manifest.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_directory(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    try:
        candidate = (root / relative).resolve()
        candidate.relative_to(root)
        return candidate if candidate.is_dir() else None
    except (OSError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload


def evaluate_suite(
    suite: dict[str, Any], suite_directory: Path
) -> dict[str, Any]:
    profile = suite.get("profile")
    required_modes = suite.get("required_modes")
    entries = suite.get("modes")
    expected_modes = (
        set(MODE_MANIFESTS)
        if profile == "release"
        else {"closed_loop"}
    )
    checks: dict[str, bool] = {
        "schema": suite.get("schema_version") == 1,
        "evidence_mode": suite.get("evidence_mode") == "cudanav_autonomy_suite",
        "profile": profile in {"smoke", "release"},
        "declared_passed": suite.get("passed") is True,
        "required_modes": (
            isinstance(required_modes, list)
            and len(required_modes) == len(set(required_modes))
            and set(required_modes) <= set(MODE_MANIFESTS)
            and expected_modes <= set(required_modes)
        ),
        "mode_table": isinstance(entries, dict)
        and isinstance(required_modes, list)
        and set(entries) == set(required_modes),
    }
    root = suite_directory.resolve()
    mode_results: dict[str, dict[str, Any]] = {}
    commits: set[str] = set()
    config_hashes: set[str] = set()

    if not isinstance(required_modes, list) or not isinstance(entries, dict):
        return {
            "passed": False,
            "checks": checks,
            "modes": mode_results,
            "coverage": {},
        }

    for mode in required_modes:
        entry = entries.get(mode)
        result: dict[str, Any] = {"passed": False}
        if mode not in MODE_MANIFESTS or not isinstance(entry, dict):
            result["error"] = "invalid mode entry"
            mode_results[mode] = result
            continue
        directory = _safe_directory(root, entry.get("directory"))
        if directory is None:
            result["error"] = "mode directory is missing or escapes suite"
            mode_results[mode] = result
            continue
        manifest_path = directory / MODE_MANIFESTS[mode]
        if not manifest_path.is_file():
            result["error"] = "mode manifest is missing"
            mode_results[mode] = result
            continue
        try:
            manifest = _read_json(manifest_path)
            manifest_binding = (
                entry.get("manifest_sha256") == sha256_file(manifest_path)
            )
            if mode == "closed_loop":
                artifacts = manifest.get("artifacts", {})
                summary_path = (directory / artifacts["summary"]).resolve()
                summary_path.relative_to(directory)
                summary = _read_json(summary_path)
                summary_gate = evaluate_summary(summary, profile)
                manifest_gate = evaluate_closed_loop_manifest(
                    manifest, directory, profile
                )
                semantic_mode = (
                    manifest.get("evidence_mode") in (None, "closed_loop_simulation")
                )
                mode_passed = (
                    summary_gate["passed"]
                    and manifest_gate["passed"]
                    and artifacts.get("trajectory") == summary.get("trajectory_csv")
                    and manifest.get("traversal_count")
                    == summary.get("traversals_requested")
                )
                commits.add(str(manifest.get("git_commit", "")))
                config_hashes.add(str(manifest.get("config_sha256", "")))
                result.update(
                    {
                        "summary_gate": summary_gate,
                        "manifest_gate": manifest_gate,
                        "semantic_mode": semantic_mode,
                    }
                )
            elif mode == "real_rosbag_shadow":
                manifest_gate = evaluate_rosbag_manifest(
                    manifest, directory, profile
                )
                semantic_mode = (
                    manifest.get("evidence_mode")
                    == "shadow_controller_with_recorded_motion"
                )
                mode_passed = manifest_gate["passed"]
                commits.add(str(manifest.get("git_commit", "")))
                config_hashes.add(
                    str(manifest.get("controller_config_sha256", ""))
                )
                result.update(
                    {
                        "manifest_gate": manifest_gate,
                        "semantic_mode": semantic_mode,
                    }
                )
            else:
                manifest_gate = evaluate_multi_gpu_suite(
                    manifest, directory
                )
                semantic_mode = manifest.get("profile") == "smoke"
                mode_passed = manifest_gate["passed"]
                coverage = manifest_gate.get("coverage", {})
                commits.update(coverage.get("git_commits", []))
                config_hashes.update(coverage.get("config_sha256", []))
                result.update(
                    {
                        "manifest_gate": manifest_gate,
                        "semantic_mode": semantic_mode,
                    }
                )
            result["manifest_binding"] = manifest_binding
            result["passed"] = (
                mode_passed and semantic_mode and manifest_binding
            )
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            result["error"] = str(error)
        mode_results[mode] = result

    checks["all_modes_passed"] = (
        set(mode_results) == set(required_modes)
        and all(result["passed"] for result in mode_results.values())
    )
    checks["same_git_commit"] = len(commits) == 1 and "" not in commits
    checks["same_controller_config"] = (
        len(config_hashes) == 1 and "" not in config_hashes
    )
    checks["suite_git_binding"] = (
        len(commits) == 1 and suite.get("git_commit") in commits
    )
    checks["distinct_evidence_modes"] = (
        len(required_modes) == len(set(required_modes))
        and "closed_loop" in required_modes
        and (
            profile != "release"
            or "real_rosbag_shadow" in required_modes
            and "multi_gpu" in required_modes
        )
    )
    return {
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "modes": mode_results,
        "coverage": {
            "git_commits": sorted(commits),
            "config_sha256": sorted(config_hashes),
        },
    }


__all__ = ["evaluate_suite", "sha256_file"]
