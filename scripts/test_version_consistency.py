#!/usr/bin/env python3
"""Ensure package, release-document, and docs-site versions stay synchronized."""

from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def capture(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError(f"version not found in {path}")
    return match.group(1)


def main() -> int:
    versions = {
        "python_project": capture(
            REPO / "python" / "pyproject.toml",
            r'(?m)^version\s*=\s*"([^"]+)"',
        ),
        "python_extension": capture(
            REPO / "python" / "src" / "cudarobotics" / "bindings.cpp",
            r'm\.attr\("__version__"\)\s*=\s*"([^"]+)"',
        ),
        "cuda_mppi_controller": capture(
            REPO / "ros2_ws" / "src" / "cuda_mppi_controller" / "package.xml",
            r"<version>([^<]+)</version>",
        ),
        "cuda_robotics": capture(
            REPO / "ros2_ws" / "src" / "cuda_robotics" / "package.xml",
            r"<version>([^<]+)</version>",
        ),
        "release_notes": capture(
            REPO / "docs" / "releases" / "v0.2.0_notes.md",
            r"(?m)^# v([0-9]+\.[0-9]+\.[0-9]+)\b",
        ),
        "release_checklist": capture(
            REPO / "docs" / "releases" / "v0.2.0_smoke_checklist.md",
            r"(?m)^- Tag: `v([0-9]+\.[0-9]+\.[0-9]+)`$",
        ),
    }
    assert len(set(versions.values())) == 1, versions
    version = next(iter(versions.values()))
    release_label = f"v{version}"
    docs_pages = (
        REPO / "docs" / "site" / "index.html",
        REPO / "docs" / "site" / "install.html",
        REPO / "docs" / "site" / "nav2.html",
        REPO / "docs" / "site" / "results.html",
    )
    for path in docs_pages:
        assert release_label in path.read_text(encoding="utf-8"), (
            f"{release_label} not found in {path}"
        )
    print(f"version consistency checks passed: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
