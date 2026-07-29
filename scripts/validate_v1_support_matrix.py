#!/usr/bin/env python3
"""Validate the v1.0 cross-surface support matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v1_support_matrix import MATRIX_PATH, evaluate, load


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    try:
        result = evaluate(load(args.matrix.resolve()))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        result = {"valid": False, "ready": False, "error": str(error)}
    print(json.dumps(result, indent=2, sort_keys=True))
    passed = result["ready"] if args.require_ready else result["valid"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
