#!/usr/bin/env python3
"""Validate a content-bound v0.2 local preflight directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from release_preflight_evidence import evaluate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--profile", choices=("cpu", "gpu"))
    parser.add_argument("--commit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    directory = args.directory.resolve()
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest root must be an object")
        result = evaluate_manifest(
            manifest,
            directory,
            expected_profile=args.profile,
            expected_commit=args.commit,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result = {"passed": False, "error": str(error)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
