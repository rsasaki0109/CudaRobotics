#!/usr/bin/env python3
"""Checks for offline motion comparison helpers."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("compare_rosbag_motion.py")
SPEC = importlib.util.spec_from_file_location("compare_rosbag_motion", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> int:
    rows = [
        {"recorded_ns": 0, "linear_x": 0.0, "linear_y": 0.0},
        {"recorded_ns": 1_000_000_000, "linear_x": 0.0, "linear_y": 0.0},
        {"recorded_ns": 2_000_000_000, "linear_x": 0.2, "linear_y": 0.0},
        {"recorded_ns": 3_000_000_000, "linear_x": 0.0, "linear_y": 0.0},
        {"recorded_ns": 3_500_000_000, "linear_x": 0.2, "linear_y": 0.0},
    ]
    assert MODULE.low_speed_episodes(rows) == [(0.0, 2.0)]
    print("offline motion comparison checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
