#!/usr/bin/env python3
"""Checks for the dependency-free CDR motion decoder."""

import importlib.util
import math
import struct
from pathlib import Path


SCRIPT = Path(__file__).with_name("export_rosbag_motion.py")
SPEC = importlib.util.spec_from_file_location("export_rosbag_motion", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Writer:
    def __init__(self):
        self.data = bytearray(b"\x00\x01\x00\x00")

    def add(self, code, size, value):
        self.data.extend(b"\x00" * (-((len(self.data) - 4) % size) % size))
        self.data.extend(struct.pack("<" + code, value))

    def string(self, value):
        raw = value.encode() + b"\x00"
        self.add("I", 4, len(raw))
        self.data.extend(raw)

    def doubles(self, values):
        for value in values:
            self.add("d", 8, value)


def main() -> int:
    twist = b"\x00\x01\x00\x00" + struct.pack("<6d", 0.3, 0, 0, 0, 0, -0.2)
    parsed_twist = MODULE.parse_twist(twist)
    assert parsed_twist["linear_x"] == 0.3
    assert parsed_twist["angular_z"] == -0.2

    writer = Writer()
    writer.add("i", 4, 12)
    writer.add("I", 4, 345)
    writer.string("odom")
    writer.string("base_link")
    writer.doubles([1.0, 2.0, 0.0])
    writer.doubles([0.0, 0.0, math.sin(0.25), math.cos(0.25)])
    writer.doubles([0.0] * 36)
    writer.doubles([0.4, 0.0, 0.0, 0.0, 0.0, 0.1])
    writer.doubles([0.0] * 36)
    odom = MODULE.parse_odometry(bytes(writer.data))
    assert odom["stamp_ns"] == 12_000_000_345
    assert odom["frame_id"] == "odom"
    assert odom["child_frame_id"] == "base_link"
    assert abs(odom["yaw"] - 0.5) < 1e-12
    assert abs(odom["qz"] - math.sin(0.25)) < 1e-12
    assert abs(odom["qw"] - math.cos(0.25)) < 1e-12
    assert odom["linear_x"] == 0.4

    writer = Writer()
    writer.add("i", 4, 15)
    writer.add("I", 4, 678)
    writer.string("map")
    writer.doubles([4.0, 5.0, 6.0])
    writer.doubles([0.0, 0.0, math.sin(-0.1), math.cos(-0.1)])
    pose = MODULE.parse_pose_stamped(bytes(writer.data))
    assert pose["stamp_ns"] == 15_000_000_678
    assert pose["frame_id"] == "map"
    assert pose["x"] == 4.0
    assert pose["z"] == 6.0
    assert abs(pose["yaw"] + 0.2) < 1e-12
    assert (
        MODULE.pose_parser("geometry_msgs/msg/PoseStamped")
        is MODULE.parse_pose_stamped
    )
    print("offline CDR motion decoder checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
