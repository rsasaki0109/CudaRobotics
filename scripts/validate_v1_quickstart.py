#!/usr/bin/env python3
"""Validate a retained v1 Docker quickstart evidence directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v1_quickstart_evidence import evaluate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--profile", choices=("development", "release"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    root = args.directory.resolve()
    try:
        manifest = json.loads(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        if not isinstance(manifest, dict):
            raise ValueError("manifest root must be an object")
        result = evaluate_manifest(
            manifest,
            root,
            expected_profile=args.profile,
            expected_commit=args.commit,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        result = {"passed": False, "error": str(error)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
