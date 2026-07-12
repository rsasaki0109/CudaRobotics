#!/usr/bin/env python3
"""Checks for the dependency-free LaserScan decoder."""

import importlib.util
import math
import struct
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_rosbag_clearance.py")
SPEC = importlib.util.spec_from_file_location("analyze_rosbag_clearance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Writer:
    def __init__(self): self.data = bytearray(b"\x00\x01\x00\x00")
    def add(self, code, size, value):
        self.data.extend(b"\x00" * (-((len(self.data) - 4) % size) % size))
        self.data.extend(struct.pack("<" + code, value))
    def string(self, value):
        raw = value.encode() + b"\x00"; self.add("I", 4, len(raw)); self.data.extend(raw)


def main() -> int:
    writer = Writer()
    writer.add("i", 4, 7); writer.add("I", 4, 9); writer.string("laser")
    for value in (-1.0, 1.0, 1.0, 0.01, 0.1, 0.1, 10.0): writer.add("f", 4, value)
    writer.add("I", 4, 3)
    for value in (2.0, 0.4, math.inf): writer.add("f", 4, value)
    writer.add("I", 4, 0)
    scan = MODULE.parse_laser_scan(bytes(writer.data))
    assert scan["stamp_ns"] == 7_000_000_009
    assert scan["frame_id"] == "laser"
    assert len(scan["ranges"]) == 3
    assert abs(MODULE.finite_min(scan["ranges"], 0.1, 10.0) - 0.4) < 1e-6
    rows = [
        {"recorded_ns": 0, "front_min_range_m": 0.4, "command_speed_mps": 0.2, "command_age_ms": 10.0},
        {"recorded_ns": 100_000_000, "front_min_range_m": 0.3, "command_speed_mps": 0.1, "command_age_ms": 12.0},
        {"recorded_ns": 400_000_000, "front_min_range_m": 0.2, "command_speed_mps": 0.0, "command_age_ms": 14.0},
        {"recorded_ns": 500_000_000, "front_min_range_m": 0.8, "command_speed_mps": 0.2, "command_age_ms": 10.0},
    ]
    episodes = MODULE.proximity_episodes(rows)
    assert len(episodes) == 2
    assert episodes[0]["duration_s"] == 0.1
    assert episodes[0]["min_front_range_m"] == 0.3
    assert episodes[1]["command_speed_at_closest_mps"] == 0.0
    print("offline LaserScan decoder checks passed")
    return 0


if __name__ == "__main__": raise SystemExit(main())
