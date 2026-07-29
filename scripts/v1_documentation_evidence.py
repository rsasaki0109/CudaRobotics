#!/usr/bin/env python3
"""Validate retained HTTP evidence for the deployed v1 documentation site."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from v1_quickstart_evidence import sha256_file


SITE = "https://rsasaki0109.github.io/CudaRobotics/docs/"
REQUIRED_ARTIFACTS = {
    "site/index.html",
    "site/install.html",
    "site/nav2.html",
    "site/release.json",
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


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def evaluate_site_content(
    site_root: Path,
    *,
    version: Any,
    target_tag: Any,
    git_commit: Any,
) -> dict[str, bool]:
    release: dict[str, Any] = {}
    try:
        candidate = json.loads(
            (site_root / "release.json").read_text(encoding="utf-8")
        )
        if isinstance(candidate, dict):
            release = candidate
    except (OSError, json.JSONDecodeError):
        pass
    index = read_text(site_root / "index.html")
    install = read_text(site_root / "install.html")
    nav2 = read_text(site_root / "nav2.html")
    return {
        "release_schema": release.get("schema_version") == 1
        and release.get("version") == version
        and release.get("target_tag") == target_tag
        and release.get("source_commit") == git_commit
        and release.get("site") == SITE,
        "index_page": isinstance(version, str)
        and version in index
        and 'href="install.html"' in index
        and 'href="nav2.html"' in index,
        "install_page": isinstance(version, str)
        and isinstance(target_tag, str)
        and version in install
        and f"releases/tag/{target_tag}" in install
        and f"cuda-mppi-controller-demo:{target_tag}" in install,
        "nav2_page": isinstance(version, str)
        and version in nav2
        and "cuda_nav_bringup cudanav_closed_loop.launch.py" in nav2,
        "release_links": isinstance(target_tag, str)
        and f"/blob/{target_tag}/" in install
        and f"/blob/{target_tag}/" in nav2,
    }


def evaluate_manifest(
    manifest: dict[str, Any],
    directory: Path,
    *,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    root = directory.resolve()
    commit = manifest.get("git_commit")
    version = manifest.get("version")
    tag = manifest.get("target_tag")
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
    site_checks = evaluate_site_content(
        root / "site",
        version=version,
        target_tag=tag,
        git_commit=commit,
    )
    statuses = manifest.get("http_status", {})
    urls = manifest.get("urls", {})
    if not isinstance(statuses, dict):
        statuses = {}
    if not isinstance(urls, dict):
        urls = {}
    expected_urls = {
        "index": SITE,
        "install": SITE + "install.html",
        "nav2": SITE + "nav2.html",
        "release": SITE + "release.json",
    }
    checks = {
        "schema": manifest.get("schema_version") == 1,
        "evidence_mode": manifest.get("evidence_mode")
        == "v1_documentation_http_deployment",
        "status": manifest.get("status") == "passed",
        "version": version == "1.0.0",
        "target_tag": tag == "v1.0.0",
        "git_commit": isinstance(commit, str)
        and bool(re.fullmatch(r"[0-9a-f]{40}", commit))
        and (expected_commit is None or commit == expected_commit),
        "source_clean": manifest.get("git_dirty") is False,
        "site": manifest.get("site") == SITE,
        "urls": urls == expected_urls,
        "http_status": statuses == {
            key: 200 for key in expected_urls
        },
        **site_checks,
        "artifact_schema": isinstance(artifacts, list)
        and len(declared_paths) == len(artifacts)
        and len(declared_paths) == len(set(declared_paths)),
        "artifact_coverage": set(declared_paths) == REQUIRED_ARTIFACTS,
        "artifact_content": artifact_content,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "git_commit": commit,
    }
