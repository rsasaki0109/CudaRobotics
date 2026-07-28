#!/usr/bin/env python3
"""Ensure the Python extension and ROS package versions stay synchronized."""

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
    }
    assert len(set(versions.values())) == 1, versions
    print(f"version consistency checks passed: {next(iter(versions.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
