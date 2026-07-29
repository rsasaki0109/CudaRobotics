#!/usr/bin/env python3
"""Validate a commit-bound v0.2 GitHub CI evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from release_ci_evidence import GATE_CONTRACTS, evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--gate", choices=sorted(GATE_CONTRACTS))
    parser.add_argument("--commit")
    args = parser.parse_args()
    try:
        payload = json.loads(args.evidence.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("evidence root must be an object")
        result = evaluate(
            payload,
            expected_gate=args.gate,
            expected_commit=args.commit,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result = {"passed": False, "error": str(error)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
