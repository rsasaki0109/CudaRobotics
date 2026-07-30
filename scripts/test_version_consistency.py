#!/usr/bin/env python3
"""Ensure v1 source packages and the published-docs declaration stay synced."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def capture(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError(f"version not found in {path}")
    return match.group(1)


def main() -> int:
    matrix = json.loads(
        (REPO / "docs" / "v1_support_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    source = matrix["source_version"]
    declared_ros = matrix["surfaces"]["ros2"]["package_versions"]
    versions = {
        "python_project": capture(
            REPO / "python" / "pyproject.toml",
            r'(?m)^version\s*=\s*"([^"]+)"',
        ),
        "python_extension": capture(
            REPO / "python" / "src" / "cudarobotics" / "bindings.cpp",
            r'm\.attr\("__version__"\)\s*=\s*"([^"]+)"',
        ),
    }
    for package in (*declared_ros, "cuda_robotics"):
        versions[package] = (
            ET.parse(REPO / "ros2_ws" / "src" / package / "package.xml")
            .getroot()
            .findtext("version")
        )
    versions["cuda_nav_bringup_setup"] = capture(
        REPO / "ros2_ws" / "src" / "cuda_nav_bringup" / "setup.py",
        r'(?m)^\s*version="([^"]+)"',
    )

    assert matrix["source_tag"] == f"v{source}"
    assert set(declared_ros.values()) == {source}, declared_ros
    assert set(versions.values()) == {source}, versions

    published = matrix["release_readiness"]["published_version"]
    published_label = f"v{published}"
    docs_pages = (
        REPO / "docs" / "site" / "index.html",
        REPO / "docs" / "site" / "install.html",
        REPO / "docs" / "site" / "nav2.html",
        REPO / "docs" / "site" / "results.html",
    )
    for path in docs_pages:
        assert published_label in path.read_text(encoding="utf-8"), (
            f"{published_label} not found in {path}"
        )
    install = docs_pages[1].read_text(encoding="utf-8")
    assert f"releases/tag/{published_label}" in install
    print(
        "version consistency checks passed: "
        f"source={source}, published={published}, "
        f"future_target={matrix['target_version']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
