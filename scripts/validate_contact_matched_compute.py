#!/usr/bin/env python3
"""Independently validate deadline-matched contact evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contact_matched_compute import evaluate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_directory", type=Path)
    parser.add_argument("--profile", choices=("smoke", "release"))
    args = parser.parse_args()
    root = args.evidence_directory.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    profile = args.profile or manifest.get("profile")
    result = evaluate_manifest(manifest, root, profile)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
