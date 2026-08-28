#!/usr/bin/env python3
"""Install the compatible published CudaRobotics wheel from GitHub Releases."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPPORT_MATRIX = ROOT / "docs" / "v1_support_matrix.json"
RELEASE_BASE = "https://github.com/rsasaki0109/CudaRobotics/releases/download"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the selected requirement only")
    parser.add_argument("--force-reinstall", action="store_true")
    return parser.parse_args(argv)


def load_contract() -> dict:
    return json.loads(SUPPORT_MATRIX.read_text(encoding="utf-8"))


def wheel_requirement(
    contract: dict,
    *,
    system: str,
    machine: str,
    implementation: str,
    major: int,
    minor: int,
) -> tuple[str, str]:
    version = contract["release_readiness"]["published_version"]
    wheel = contract["surfaces"]["python_wheels"]
    python_tag = f"cp{major}{minor}"
    if system != "Linux" or machine.lower() not in {"x86_64", "amd64"}:
        raise ValueError(
            "published wheels support Linux x86_64 only; use Colab or the source-build path"
        )
    if implementation != "cpython" or python_tag not in wheel["python"]:
        supported = ", ".join(wheel["python"])
        raise ValueError(
            f"published wheels support CPython {supported}; use a supported interpreter or build from source"
        )
    filename = (
        f"cudarobotics-{version}-{python_tag}-{python_tag}-"
        "manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    )
    url = f"{RELEASE_BASE}/v{version}/{filename}"
    return version, f"cudarobotics[examples] @ {url}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        version, requirement = wheel_requirement(
            load_contract(),
            system=platform.system(),
            machine=platform.machine(),
            implementation=sys.implementation.name,
            major=sys.version_info.major,
            minor=sys.version_info.minor,
        )
    except (KeyError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Cannot select a published wheel: {exc}", file=sys.stderr)
        return 2

    print(f"Selected CudaRobotics {version} for this interpreter")
    print(requirement)
    if args.dry_run:
        return 0

    command = [sys.executable, "-m", "pip", "install"]
    if args.force_reinstall:
        command.append("--force-reinstall")
    command.append(requirement)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    installed = importlib.metadata.version("cudarobotics")
    if installed != version:
        print(
            f"Installed metadata reports {installed}, expected {version}",
            file=sys.stderr,
        )
        return 1
    print(f"PASS: installed cudarobotics {installed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
