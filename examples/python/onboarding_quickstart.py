#!/usr/bin/env python3
"""Run the two Python quickstarts and write one activation result."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "build" / "onboarding" / "python"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for the GIF, logs, and python_quickstart_result.json",
    )
    parser.add_argument(
        "--recipe",
        choices=("initial", "planning_variant"),
        default="initial",
        help="label this run as the initial activation or a Level 1 variant",
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def classify_failure(output: str) -> str:
    text = output.lower()
    if "no module named" in text or "modulenotfounderror" in text:
        return "import"
    if any(
        marker in text
        for marker in (
            "cuda driver",
            "cuda error",
            "cuda runtime",
            "no cuda-capable device",
            "driver version is insufficient",
        )
    ):
        return "cuda_runtime"
    if any(
        marker in text
        for marker in (
            "goal not reached",
            "all sampled trajectories collided",
            "quickstart checks did not pass",
        )
    ):
        return "algorithm_check"
    return "unknown"


def run_step(name: str, command: list[str], log_path: Path) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    duration = time.perf_counter() - started
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    result = {
        "name": name,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(duration, 3),
        "log": log_path.name,
    }
    if completed.returncode != 0:
        result["failure_category"] = classify_failure(completed.stdout)
    return result


def package_version() -> str | None:
    try:
        return importlib.metadata.version("cudarobotics")
    except importlib.metadata.PackageNotFoundError:
        return None


def source_version() -> str:
    pyproject = (ROOT / "python" / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("python/pyproject.toml does not declare a project version")
    return match.group(1)


def write_result(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def require_mppi_gif(step: dict, gif_path: Path) -> None:
    if step["passed"] and (
        not gif_path.is_file() or gif_path.stat().st_size == 0
    ):
        step["passed"] = False
        step["failure_category"] = "artifact"
        step["message"] = (
            "MPPI ran, but the GIF was not created. Install the example "
            "dependencies with: python -m pip install -e 'python/[examples]'"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "python_quickstart_result.json"
    started = time.perf_counter()

    result = {
        "schema_version": 1,
        "surface": "python_quickstart",
        "recipe": args.recipe,
        "package_version": package_version(),
        "source_version": source_version(),
        "passed": False,
        "failure_category": None,
        "steps": [],
        "artifacts": {
            "mppi_gif": "mppi_quickstart.gif",
            "result": "python_quickstart_result.json",
        },
        "next_steps": [
            "python examples/python/mppi_dlpack_costmap.py",
            "docs/onboarding_recipes.md",
            "ros2_ws/src/cuda_mppi_controller/",
        ],
    }

    if result["package_version"] is None:
        result["failure_category"] = "import"
        result["message"] = (
            "cudarobotics is not installed. Run: "
            "python -m pip install -e 'python/[examples]'"
        )
        result["duration_seconds"] = round(time.perf_counter() - started, 3)
        write_result(result_path, result)
        print(result["message"], file=sys.stderr)
        print(f"wrote failure result: {result_path}", file=sys.stderr)
        return 1

    if (
        result["package_version"] != result["source_version"]
        and not args.allow_version_mismatch
    ):
        result["failure_category"] = "version_mismatch"
        result["message"] = (
            f"installed cudarobotics {result['package_version']} does not match "
            f"this checkout ({result['source_version']}). Reinstall with: "
            "python -m pip install -e 'python/[examples]'"
        )
        result["duration_seconds"] = round(time.perf_counter() - started, 3)
        write_result(result_path, result)
        print(result["message"], file=sys.stderr)
        print(f"wrote failure result: {result_path}", file=sys.stderr)
        return 1

    if shutil.which("nvidia-smi") is None:
        result["failure_category"] = "preflight"
        result["message"] = (
            "nvidia-smi was not found. Use the Colab quickstart or install an "
            "NVIDIA driver before running the local GPU examples."
        )
        result["duration_seconds"] = round(time.perf_counter() - started, 3)
        write_result(result_path, result)
        print(result["message"], file=sys.stderr)
        print(f"wrote failure result: {result_path}", file=sys.stderr)
        return 1

    commands = (
        (
            "mppi",
            [
                sys.executable,
                str(ROOT / "examples" / "python" / "mppi_quickstart.py"),
                str(output_dir / "mppi_quickstart.gif"),
            ],
        ),
        (
            "registration",
            [
                sys.executable,
                str(ROOT / "examples" / "python" / "registration_quickstart.py"),
            ],
        ),
    )

    for name, command in commands:
        print(f"\n== {name} quickstart ==")
        step = run_step(name, command, output_dir / f"{name}.log")
        if name == "mppi" and step["passed"]:
            require_mppi_gif(step, output_dir / "mppi_quickstart.gif")
        result["steps"].append(step)
        if not step["passed"]:
            result["failure_category"] = step["failure_category"]
            result["failed_step"] = name
            break

    result["passed"] = len(result["steps"]) == len(commands) and all(
        step["passed"] for step in result["steps"]
    )
    result["duration_seconds"] = round(time.perf_counter() - started, 3)
    write_result(result_path, result)

    if not result["passed"]:
        print(f"\nPython quickstart failed; see {result_path}", file=sys.stderr)
        return 1

    print("\nPASS: Python quickstart complete")
    print(f"Result: {result_path}")
    print("Next: open docs/onboarding_recipes.md and choose Level 1, 2, or 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
