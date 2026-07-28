#!/usr/bin/env python3
"""Independently validate a complete CudaNav autonomy evidence suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cudanav_autonomy_suite import evaluate_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_directory", type=Path)
    args = parser.parse_args()
    root = args.suite_directory.resolve()
    suite = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = evaluate_suite(suite, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
