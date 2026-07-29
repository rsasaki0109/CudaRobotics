#!/usr/bin/env python3
"""Validate the v1.0 cross-surface support matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v1_release_bundle import load_bundle
from v1_support_matrix import MATRIX_PATH, evaluate, load


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--evidence-bundle", type=Path)
    parser.add_argument("--release-commit")
    args = parser.parse_args()
    if bool(args.evidence_bundle) != bool(args.release_commit):
        parser.error(
            "--evidence-bundle and --release-commit must be provided together"
        )
    try:
        matrix = load(args.matrix.resolve())
        bundle_gate = None
        if args.evidence_bundle:
            bundle_gate = load_bundle(
                args.evidence_bundle,
                target_version=str(matrix.get("target_version", "")),
                target_tag=str(matrix.get("target_tag", "")),
                expected_commit=args.release_commit,
            )
        result = evaluate(
            matrix,
            attestation_root=(
                args.evidence_bundle.resolve().parent
                if args.evidence_bundle
                else MATRIX_PATH.parents[1]
            ),
            readiness_evidence=(
                bundle_gate["references"] if bundle_gate else None
            ),
            expected_release_commit=args.release_commit,
        )
        if bundle_gate is not None:
            result["release_bundle"] = bundle_gate
            result["ready"] = result["ready"] and bundle_gate["passed"]
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        result = {"valid": False, "ready": False, "error": str(error)}
    print(json.dumps(result, indent=2, sort_keys=True))
    passed = result["ready"] if args.require_ready else result["valid"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
