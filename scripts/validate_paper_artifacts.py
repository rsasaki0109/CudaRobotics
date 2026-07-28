#!/usr/bin/env python3
"""Validate one or all paper claim-to-evidence manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_artifact_contract import validate_manifest


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="*", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    manifests = args.manifests or sorted((ROOT / "paper" / "artifacts").glob("*.json"))
    if not manifests:
        raise SystemExit("no paper artifact manifests found")
    results = {}
    for manifest_path in manifests:
        path = manifest_path if manifest_path.is_absolute() else ROOT / manifest_path
        payload = json.loads(path.read_text(encoding="utf-8"))
        results[str(path.relative_to(ROOT))] = validate_manifest(payload, ROOT)
    print(json.dumps(results, indent=2, sort_keys=True))
    valid = all(result["valid"] for result in results.values())
    ready = all(result["ready"] for result in results.values())
    return 0 if valid and (ready or not args.require_ready) else 1


if __name__ == "__main__":
    raise SystemExit(main())
