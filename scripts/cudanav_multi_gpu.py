#!/usr/bin/env python3
"""Aggregate validation for CudaNav closed-loop runs across GPU hardware."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cudanav_evidence import evaluate_manifest, evaluate_summary


def evaluate_multi_gpu_suite(
    suite: dict[str, Any], suite_directory: Path
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "schema": suite.get("schema_version") == 1,
        "profile": suite.get("profile") == "smoke",
    }
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
            artifacts = manifest.get("artifacts", {})
            summary_path = run_directory / artifacts["summary"]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_gate = evaluate_summary(summary, "smoke")
            manifest_gate = evaluate_manifest(manifest, run_directory, "smoke")
            binding = (
                artifacts.get("trajectory") == summary.get("trajectory_csv")
                and manifest.get("traversal_count")
                == summary.get("traversals_requested")
            )
            gpus = manifest.get("gpu")
            one_gpu = isinstance(gpus, list) and len(gpus) == 1
            expected_gpu = entry.get("device")
            device_binding = (
                one_gpu
                and isinstance(expected_gpu, dict)
                and str(expected_gpu.get("index", ""))
                == str(gpus[0].get("physical_index", ""))
                and expected_gpu.get("name") == gpus[0].get("name")
                and expected_gpu.get("uuid") == gpus[0].get("uuid")
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
            commits.add(str(manifest.get("git_commit", "")))
            config_hashes.add(str(manifest.get("config_sha256", "")))
            if one_gpu:
                gpu_uuids.add(str(gpus[0].get("uuid", "")))
                gpu_models.add(str(gpus[0].get("name", "")))
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
