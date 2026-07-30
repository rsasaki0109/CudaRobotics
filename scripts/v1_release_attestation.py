#!/usr/bin/env python3
"""Validate content-bound attestations consumed by the v1 support matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

MODES = {
    "quickstart_15_minute_evidence": "v1_quickstart_release",
    "cudanav_release_evidence": "v1_cudanav_systems_release",
    "docker_gpu_evidence": "v1_docker_gpu_release",
    "documentation_deployment": "v1_documentation_deployment",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def common_checks(
    payload: dict[str, Any],
    *,
    mode: str,
    target_version: str,
    target_tag: str,
) -> dict[str, bool]:
    checks = payload.get("checks")
    return {
        "schema": payload.get("schema_version") == 1,
        "mode": payload.get("evidence_mode") == mode,
        "status": payload.get("status") == "passed",
        "version": payload.get("version") == target_version,
        "target_tag": payload.get("target_tag") == target_tag,
        "git_commit": bool(
            re.fullmatch(r"[0-9a-f]{40}", str(payload.get("git_commit", "")))
        ),
        "source_clean": payload.get("git_dirty") is False,
        "payload_binding": bool(
            re.fullmatch(r"[0-9a-f]{64}", str(payload.get("payload_sha256", "")))
        ),
        "validator_checks": isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values()),
        "details": isinstance(payload.get("details"), dict),
    }


def mode_checks(
    payload: dict[str, Any],
    *,
    key: str,
    target_tag: str,
) -> dict[str, bool]:
    details = payload.get("details", {})
    if not isinstance(details, dict):
        details = {}
    if key == "quickstart_15_minute_evidence":
        duration = details.get("duration_seconds")
        return {
            "profile": details.get("profile") == "release",
            "surface": details.get("surface") == "docker_source",
            "duration": isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and 0 < duration <= 900,
            "result": details.get("result") == "out/cudanav_closed_loop.json",
            "fresh_clone": details.get("fresh_clone") is True,
            "no_cache_build": details.get("no_cache_build") is True,
        }
    if key == "cudanav_release_evidence":
        duration = details.get("closed_loop_duration_seconds")
        models = details.get("physical_gpu_models")
        return {
            "ros2_closed_loop": details.get("ros2_closed_loop") is True,
            "closed_loop_duration": isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and duration >= 600,
            "real_rosbag_shadow": details.get("real_rosbag_shadow") is True,
            "physical_gpu": isinstance(models, list)
            and all(isinstance(model, str) and model for model in models)
            and len(set(models)) >= 1,
            "ros_jazzy": details.get("ros_distribution") == "jazzy",
        }
    if key == "docker_gpu_evidence":
        image = str(details.get("image", ""))
        return {
            "image": image
            == f"ghcr.io/rsasaki0109/cuda-mppi-controller-demo:{target_tag}",
            "image_digest": bool(
                re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(details.get("image_digest", "")),
                )
            ),
            "gpu_uuid": bool(
                re.fullmatch(
                    r"GPU-[0-9a-fA-F-]+",
                    str(details.get("gpu_uuid", "")),
                )
            ),
            "smoke_pass": details.get("smoke_pass") is True,
        }
    if key == "documentation_deployment":
        return {
            "site": details.get("site")
            == "https://rsasaki0109.github.io/CudaRobotics/docs/",
            "deployed_tag": details.get("deployed_tag") == target_tag,
            "install_page": details.get("install_page_pass") is True,
            "nav2_page": details.get("nav2_page_pass") is True,
            "release_links": details.get("release_links_pass") is True,
        }
    return {"known_key": False}


def validate_payload(
    payload: dict[str, Any],
    *,
    key: str,
    target_version: str,
    target_tag: str,
) -> dict[str, Any]:
    mode = MODES.get(key)
    if mode is None:
        return {
            "passed": False,
            "checks": {"known_key": False},
            "git_commit": None,
        }
    checks = common_checks(
        payload,
        mode=mode,
        target_version=target_version,
        target_tag=target_tag,
    )
    checks.update(mode_checks(payload, key=key, target_tag=target_tag))
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "git_commit": payload.get("git_commit"),
    }


def load_reference(
    reference: Any,
    *,
    repo_root: Path,
    key: str,
    target_version: str,
    target_tag: str,
) -> dict[str, Any]:
    reference_checks = {
        "reference_schema": isinstance(reference, dict)
        and set(reference) == {"path", "sha256"},
        "reference_hash": isinstance(reference, dict)
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(reference.get("sha256", "")))),
    }
    path: Path | None = None
    payload: dict[str, Any] = {}
    if isinstance(reference, dict) and isinstance(reference.get("path"), str):
        candidate = (repo_root.resolve() / reference["path"]).resolve()
        if candidate.is_relative_to(repo_root.resolve()):
            path = candidate
    reference_checks["safe_path"] = path is not None
    reference_checks["file_exists"] = path is not None and path.is_file()
    reference_checks["content_bound"] = (
        path is not None
        and path.is_file()
        and sha256_file(path) == reference.get("sha256")
    )
    if reference_checks["content_bound"]:
        try:
            candidate_payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(candidate_payload, dict):
                payload = candidate_payload
        except (OSError, json.JSONDecodeError):
            pass
    payload_gate = validate_payload(
        payload,
        key=key,
        target_version=target_version,
        target_tag=target_tag,
    )
    checks = {**reference_checks, **payload_gate["checks"]}
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "git_commit": payload_gate["git_commit"],
        "path": reference.get("path") if isinstance(reference, dict) else None,
    }
