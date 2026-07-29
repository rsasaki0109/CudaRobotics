#!/usr/bin/env python3
"""Independently validate a CudaNav ROS 2 Jazzy CI evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cudanav_ros_ci_evidence import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--commit")
    args = parser.parse_args()
    try:
        payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"invalid ROS CI evidence: {error}")
        return 1
    result = evaluate(
        payload, expected_commit=args.commit
    ) if isinstance(payload, dict) else {
        "passed": False,
        "checks": {"json_object": False},
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
