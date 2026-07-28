#!/usr/bin/env python3
"""Validate a MuJoCo contact-transfer evidence directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contact_external_fidelity import evaluate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--profile", choices=("smoke", "release"))
    args = parser.parse_args()
    root = args.run_directory.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = evaluate_manifest(
        manifest, root, args.profile or manifest.get("profile")
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
