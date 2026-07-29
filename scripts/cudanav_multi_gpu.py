#!/usr/bin/env python3
"""Aggregate validation for CudaNav closed-loop runs across GPU hardware."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cudanav_evidence import evaluate_manifest, evaluate_summary
from run_cudanav_gpu_closed_loop import evaluate_result as evaluate_native_result


def evaluate_multi_gpu_suite(
    suite: dict[str, Any], suite_directory: Path
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "schema": suite.get("schema_version") == 1,
    }
    evidence_kind = suite.get("evidence_kind", "ros2_smoke")
    checks["evidence_kind"] = evidence_kind in {
        "ros2_smoke",
        "native_release",
    }
    checks["profile"] = suite.get("profile") == (
        "release" if evidence_kind == "native_release" else "smoke"
    )
    runs = suite.get("runs")
    if not isinstance(runs, list) or not runs:
        checks["runs_present"] = False
        return {"passed": False, "checks": checks, "runs": []}
    checks["runs_present"] = True
    root = suite_directory.resolve()
    run_results = []
    commits: set[str] = set()
    config_hashes: set[str] = set()
    gpu_uuids: set[str] = set()
    gpu_models: set[str] = set()
    all_runs_passed = True
    for entry in runs:
        relative = entry.get("directory") if isinstance(entry, dict) else None
        run_result: dict[str, Any] = {"directory": relative, "passed": False}
        if not isinstance(relative, str) or not relative:
            run_result["error"] = "invalid run directory"
            all_runs_passed = False
            run_results.append(run_result)
            continue
        run_directory = (root / relative).resolve()
        if not run_directory.is_relative_to(root):
            run_result["error"] = "run directory escapes suite"
            all_runs_passed = False
            run_results.append(run_result)
            continue
        manifest_path = run_directory / "manifest.json"
        if not manifest_path.is_file():
            run_result["error"] = "missing run manifest"
            all_runs_passed = False
            run_results.append(run_result)
            continue
        try:
            manifest_digest = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            manifest_binding = (
                isinstance(entry.get("manifest_sha256"), str)
                and entry["manifest_sha256"] == manifest_digest
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_gpu = entry.get("device")
            if evidence_kind == "native_release":
                artifacts = manifest.get("artifacts", {})
                result_artifact = artifacts.get("result", {})
                trajectory_artifact = artifacts.get("trajectory", {})
                result_path = run_directory / str(result_artifact.get("path", ""))
                trajectory_path = run_directory / str(
                    trajectory_artifact.get("path", "")
                )
                result_payload = json.loads(result_path.read_text(encoding="utf-8"))
                native_checks = evaluate_native_result(result_payload, "release")
                artifact_binding = all(
                    path.is_file()
                    and path.resolve().is_relative_to(run_directory)
                    and artifact.get("bytes") == path.stat().st_size
                    and artifact.get("sha256")
                    == hashlib.sha256(path.read_bytes()).hexdigest()
                    for artifact, path in (
                        (result_artifact, result_path),
                        (trajectory_artifact, trajectory_path),
                    )
                )
                manifest_checks = manifest.get("checks")
                manifest_gate = {
                    "passed": (
                        manifest.get("profile") == "release"
                        and manifest.get("git_dirty") is False
                        and isinstance(manifest_checks, dict)
                        and bool(manifest_checks)
                        and all(manifest_checks.values())
                        and all(native_checks.values())
                    ),
                    "checks": {
                        "profile": manifest.get("profile") == "release",
                        "clean_checkout": manifest.get("git_dirty") is False,
                        "native_result": all(native_checks.values()),
                        "manifest_checks": (
                            isinstance(manifest_checks, dict)
                            and bool(manifest_checks)
                            and all(manifest_checks.values())
                        ),
                    },
                }
                summary_gate = {
                    "passed": all(native_checks.values()),
                    "checks": native_checks,
                }
                binding = artifact_binding
                gpu = manifest.get("gpu_identity")
                one_gpu = isinstance(gpu, dict)
                device_binding = (
                    one_gpu
                    and isinstance(expected_gpu, dict)
                    and str(expected_gpu.get("index", ""))
                    == str(gpu.get("physical_index", ""))
                    and expected_gpu.get("name") == gpu.get("name")
                    and expected_gpu.get("uuid") == gpu.get("uuid")
                    and result_payload.get("gpu", {}).get("name")
                    == gpu.get("name")
                )
                commit = str(manifest.get("source_commit", ""))
                config_hash = str(manifest.get("source_digest", ""))
                gpu_uuid = str(gpu.get("uuid", "")) if one_gpu else ""
                gpu_model = str(gpu.get("name", "")) if one_gpu else ""
            else:
                artifacts = manifest.get("artifacts", {})
                summary_path = run_directory / artifacts["summary"]
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary_gate = evaluate_summary(summary, "smoke")
                manifest_gate = evaluate_manifest(
                    manifest, run_directory, "smoke"
                )
                binding = (
                    artifacts.get("trajectory") == summary.get("trajectory_csv")
                    and manifest.get("traversal_count")
                    == summary.get("traversals_requested")
                )
                gpus = manifest.get("gpu")
                one_gpu = isinstance(gpus, list) and len(gpus) == 1
                device_binding = (
                    one_gpu
                    and isinstance(expected_gpu, dict)
                    and str(expected_gpu.get("index", ""))
                    == str(gpus[0].get("physical_index", ""))
                    and expected_gpu.get("name") == gpus[0].get("name")
                    and expected_gpu.get("uuid") == gpus[0].get("uuid")
                )
                commit = str(manifest.get("git_commit", ""))
                config_hash = str(manifest.get("config_sha256", ""))
                gpu_uuid = (
                    str(gpus[0].get("uuid", "")) if one_gpu else ""
                )
                gpu_model = (
                    str(gpus[0].get("name", "")) if one_gpu else ""
                )
            run_result.update(
                {
                    "summary_gate": summary_gate,
                    "manifest_gate": manifest_gate,
                    "artifact_binding": binding,
                    "one_visible_gpu": one_gpu,
                    "device_binding": device_binding,
                    "manifest_binding": manifest_binding,
                }
            )
            passed = (
                summary_gate["passed"]
                and manifest_gate["passed"]
                and binding
                and one_gpu
                and device_binding
                and manifest_binding
                and entry.get("returncode") == 0
            )
            run_result["passed"] = passed
            all_runs_passed = all_runs_passed and passed
            commits.add(commit)
            config_hashes.add(config_hash)
            if one_gpu:
                gpu_uuids.add(gpu_uuid)
                gpu_models.add(gpu_model)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            run_result["error"] = str(error)
            all_runs_passed = False
        run_results.append(run_result)
    expected_runs = (
        len(suite.get("devices", [])) * int(suite.get("repetitions", 0))
        if isinstance(suite.get("devices"), list)
        and isinstance(suite.get("repetitions"), int)
        else 0
    )
    minimum_devices = int(suite.get("minimum_gpu_devices", 2))
    minimum_models = int(suite.get("minimum_gpu_models", 2))
    checks.update(
        {
            "all_runs_passed": all_runs_passed,
            "run_count": len(runs) == expected_runs and expected_runs > 0,
            "same_git_commit": len(commits) == 1 and "" not in commits,
            "same_config": len(config_hashes) == 1 and "" not in config_hashes,
            "gpu_device_coverage": len(gpu_uuids) >= minimum_devices,
            "gpu_model_coverage": len(gpu_models) >= minimum_models,
        }
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "coverage": {
            "gpu_uuids": sorted(gpu_uuids),
            "gpu_models": sorted(gpu_models),
            "git_commits": sorted(commits),
            "config_sha256": sorted(config_hashes),
        },
        "runs": run_results,
    }
