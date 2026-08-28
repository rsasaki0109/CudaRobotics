#!/usr/bin/env python3
"""Validate a local Python onboarding activation result and its artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    return parser.parse_args(argv)


def validate(result_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read result: {exc}"]

    if result.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if result.get("surface") != "python_quickstart":
        errors.append("surface must be python_quickstart")
    if result.get("passed") is not True:
        errors.append("passed must be true")
    if not isinstance(result.get("duration_seconds"), (int, float)) or result.get(
        "duration_seconds", -1
    ) < 0:
        errors.append("duration_seconds must be non-negative")

    steps = result.get("steps")
    if not isinstance(steps, list) or [step.get("name") for step in steps] != [
        "mppi",
        "registration",
    ]:
        errors.append("steps must contain mppi then registration")
    elif not all(step.get("passed") is True and step.get("returncode") == 0 for step in steps):
        errors.append("every step must pass with returncode 0")
    else:
        for step in steps:
            log_name = step.get("log")
            if not isinstance(log_name, str) or Path(log_name).name != log_name:
                errors.append(f"{step['name']} log must be a local filename")
                continue
            log_path = result_path.parent / log_name
            if not log_path.is_file() or log_path.stat().st_size == 0:
                errors.append(f"{step['name']} log is missing or empty")

    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("artifacts must be an object")
    else:
        if artifacts.get("result") != result_path.name:
            errors.append("artifacts.result must name the result file")
        gif_name = artifacts.get("mppi_gif")
        if not isinstance(gif_name, str) or Path(gif_name).name != gif_name:
            errors.append("artifacts.mppi_gif must be a local filename")
        else:
            gif_path = result_path.parent / gif_name
            if not gif_path.is_file() or gif_path.stat().st_size == 0:
                errors.append("MPPI GIF is missing or empty")
    return errors


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    errors = validate(args.result.resolve())
    if errors:
        print("Python onboarding result: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Python onboarding result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
