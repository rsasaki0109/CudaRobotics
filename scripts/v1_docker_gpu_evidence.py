#!/usr/bin/env python3
"""Validate a retained GPU smoke of the published v1 Docker image."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from v1_quickstart_evidence import sha256_file


IMAGE_REPOSITORY = (
    "ghcr.io/rsasaki0109/cuda-mppi-controller-demo"
)
REQUIRED_ARTIFACTS = {
    "docker_pull.log",
    "docker_run.log",
    "result/cudanav_closed_loop.json",
    "result/cudanav_closed_loop.log",
}


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
    expected_commit: str | None = None,
) -> dict[str, Any]:
    root = directory.resolve()
    version = manifest.get("version")
    target_tag = manifest.get("target_tag")
    commit = manifest.get("git_commit")
    image = manifest.get("image", {})
    labels = image.get("labels", {}) if isinstance(image, dict) else {}
    repo_digests = (
        image.get("repo_digests") if isinstance(image, dict) else None
    )
    artifacts = manifest.get("artifacts")
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
    try:
        summary = json.loads(
            (
                root / "result" / "cudanav_closed_loop.json"
            ).read_text(encoding="utf-8")
        )
        if not isinstance(summary, dict):
            summary = {}
    except (OSError, json.JSONDecodeError):
        pass
    gpu = manifest.get("gpu")
    commands = manifest.get("commands", {})
    returncodes = manifest.get("returncodes", {})
    expected_image = f"{IMAGE_REPOSITORY}:{target_tag}"
    expected_repo_digest = (
        f"{IMAGE_REPOSITORY}@{image.get('digest')}"
        if isinstance(image, dict)
        else ""
    )
    checks = {
        "schema": manifest.get("schema_version") == 1,
        "evidence_mode": manifest.get("evidence_mode")
        == "v1_published_docker_gpu_smoke",
        "status": manifest.get("status") == "passed",
        "version": version == "1.0.0",
        "target_tag": target_tag == "v1.0.0",
        "git_commit": isinstance(commit, str)
        and bool(re.fullmatch(r"[0-9a-f]{40}", commit))
        and (expected_commit is None or commit == expected_commit),
        "source_clean": manifest.get("git_dirty") is False,
        "image": isinstance(image, dict)
        and image.get("reference") == expected_image,
        "image_digest": isinstance(image, dict)
        and bool(
            re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(image.get("digest", "")),
            )
        )
        and isinstance(repo_digests, list)
        and expected_repo_digest in repo_digests,
        "image_id": isinstance(image, dict)
        and bool(
            re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(image.get("image_id", "")),
            )
        ),
        "image_revision": isinstance(labels, dict)
        and labels.get("org.opencontainers.image.revision") == commit,
        "image_source": isinstance(labels, dict)
        and labels.get("org.opencontainers.image.source")
        == "https://github.com/rsasaki0109/CudaRobotics",
        "image_version": isinstance(labels, dict)
        and labels.get("org.opencontainers.image.version") == target_tag,
        "gpu": isinstance(gpu, list)
        and bool(gpu)
        and all(
            isinstance(device, dict)
            and bool(device.get("name"))
            and bool(
                re.fullmatch(
                    r"GPU-[0-9a-fA-F-]+",
                    str(device.get("uuid", "")),
                )
            )
            and bool(device.get("driver_version"))
            for device in gpu
        ),
        "commands": isinstance(commands, dict)
        and commands.get("pull") == ["docker", "pull", expected_image]
        and isinstance(commands.get("run"), list)
        and commands["run"][:5]
        == ["docker", "run", "--rm", "--gpus", "all"]
        and commands["run"][-2:] == [expected_image, "cudanav"],
        "returncodes": isinstance(returncodes, dict)
        and returncodes.get("pull") == 0
        and returncodes.get("run") == 0,
        "artifact_schema": isinstance(artifacts, list)
        and len(declared_paths) == len(artifacts)
        and len(declared_paths) == len(set(declared_paths)),
        "artifact_coverage": set(declared_paths) == REQUIRED_ARTIFACTS,
        "artifact_content": artifact_content,
        "result": summary.get("schema_version") == 1
        and summary.get("smoke_pass") is True
        and summary.get("success") is True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "git_commit": commit,
        "image_digest": image.get("digest")
        if isinstance(image, dict)
        else None,
    }
