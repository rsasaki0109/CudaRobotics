#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest

from analyze_pointcloud2_clearance import analyze, front_clearance, parse_pointcloud2
from test_export_rosbag_motion import Writer


def pointcloud(
    points: list[tuple[float, float, float]], sec: int = 12
) -> bytes:
    writer = Writer()
    writer.add("i", 4, sec)
    writer.add("I", 4, 345)
    writer.string("pandar")
    writer.add("I", 4, 1)
    writer.add("I", 4, len(points))
    writer.add("I", 4, 3)
    for name, offset in (("x", 0), ("y", 4), ("z", 8)):
        writer.string(name)
        writer.add("I", 4, offset)
        writer.add("B", 1, 7)
        writer.add("I", 4, 1)
    writer.add("B", 1, 0)
    writer.add("I", 4, 12)
    writer.add("I", 4, 12 * len(points))
    payload = b"".join(struct.pack("<3f", *item) for item in points)
    writer.add("I", 4, len(payload))
    writer.data.extend(payload)
    writer.add("B", 1, 1)
    return bytes(writer.data)


class PointCloud2ClearanceTest(unittest.TestCase):
    def test_schema_named_fields_and_front_filter(self) -> None:
        cloud = parse_pointcloud2(
            pointcloud(
                [
                    (2.0, 0.0, 0.0),
                    (0.8, 0.1, 0.2),
                    (0.2, 2.0, 0.0),
                    (0.3, 0.0, -1.0),
                    (float("nan"), 0.0, 0.0),
                ]
            )
        )
        clearance, finite_count, selected_count = front_clearance(
            cloud,
            half_angle_rad=math.pi / 6,
            minimum_z_m=-0.5,
            maximum_z_m=2.5,
            minimum_range_m=0.1,
            maximum_range_m=50.0,
        )
        self.assertEqual(cloud["stamp_ns"], 12_000_000_345)
        self.assertEqual(cloud["frame_id"], "pandar")
        self.assertEqual(finite_count, 4)
        self.assertEqual(selected_count, 2)
        self.assertAlmostEqual(clearance, math.hypot(0.8, 0.1))

    def test_missing_xyz_and_invalid_stride_are_rejected(self) -> None:
        payload = pointcloud([(1.0, 0.0, 0.0)])
        with self.assertRaises(ValueError):
            parse_pointcloud2(payload.replace(b"z\x00", b"q\x00", 1))
        corrupted = bytearray(payload)
        # A truncated payload must never be accepted as declared geometry.
        with self.assertRaises(ValueError):
            parse_pointcloud2(bytes(corrupted[:-8]))

    def test_db3_clouds_pair_with_shadow_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "run.db3"
            connection = sqlite3.connect(database)
            connection.executescript(
                "CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT);"
                "CREATE TABLE messages("
                "id INTEGER PRIMARY KEY, topic_id INTEGER, "
                "timestamp INTEGER, data BLOB);"
                "INSERT INTO topics VALUES(1, '/pandar_points');"
            )
            connection.executemany(
                "INSERT INTO messages(topic_id, timestamp, data) VALUES(1, ?, ?)",
                [
                    (12_000_000_345, pointcloud([(1.0, 0.0, 0.0)], 12)),
                    (13_000_000_345, pointcloud([(0.8, 0.0, 0.0)], 13)),
                ],
            )
            connection.commit()
            connection.close()
            diagnostics = root / "diagnostics.csv"
            diagnostics.write_text(
                "stamp_sec,cmd_v,cmd_vy,cmd_w\n"
                "12.000000345,0.2,0.0,0.1\n"
                "13.000000345,0.1,0.0,0.0\n"
            )
            report = analyze(
                database,
                root / "clearance.csv",
                diagnostics,
                pointcloud_topic="/pandar_points",
            )
            self.assertEqual(report["cloud_samples"], 2)
            self.assertEqual(report["command_pair_ratio"], 1.0)
            self.assertAlmostEqual(report["minimum_front_range_m"], 0.8)


if __name__ == "__main__":
    unittest.main()
