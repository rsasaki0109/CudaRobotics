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
    points: list[tuple[float, float, float]],
    sec: int = 12,
    *,
    point_times: list[float] | None = None,
    rings: list[int] | None = None,
) -> bytes:
    if point_times is not None and len(point_times) != len(points):
        raise ValueError("point_times length mismatch")
    if rings is not None and len(rings) != len(points):
        raise ValueError("rings length mismatch")
    writer = Writer()
    writer.add("i", 4, sec)
    writer.add("I", 4, 345)
    writer.string("pandar")
    writer.add("I", 4, 1)
    writer.add("I", 4, len(points))
    fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7)]
    point_step = 12
    if point_times is not None:
        fields.append(("time", point_step, 7))
        point_step += 4
    if rings is not None:
        fields.append(("ring", point_step, 4))
        point_step += 2
    point_step = (point_step + 3) // 4 * 4
    writer.add("I", 4, len(fields))
    for name, offset, datatype in fields:
        writer.string(name)
        writer.add("I", 4, offset)
        writer.add("B", 1, datatype)
        writer.add("I", 4, 1)
    writer.add("B", 1, 0)
    writer.add("I", 4, point_step)
    writer.add("I", 4, point_step * len(points))
    payload = bytearray(point_step * len(points))
    for index, item in enumerate(points):
        base = index * point_step
        struct.pack_into("<3f", payload, base, *item)
        offset = 12
        if point_times is not None:
            struct.pack_into("<f", payload, base + offset, point_times[index])
            offset += 4
        if rings is not None:
            struct.pack_into("<H", payload, base + offset, rings[index])
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

    def test_bag_record_timestamp_pairs_cross_epoch_headers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "run.db3"
            connection = sqlite3.connect(database)
            connection.executescript(
                "CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT);"
                "CREATE TABLE messages("
                "id INTEGER PRIMARY KEY, topic_id INTEGER, "
                "timestamp INTEGER, data BLOB);"
                "INSERT INTO topics VALUES(1, '/points');"
            )
            connection.execute(
                "INSERT INTO messages(topic_id, timestamp, data) "
                "VALUES(1, ?, ?)",
                (112_000_000_345, pointcloud([(1.0, 0.0, 0.0)], 12)),
            )
            connection.commit()
            connection.close()
            diagnostics = root / "diagnostics.csv"
            diagnostics.write_text(
                "stamp_sec,cmd_v,cmd_vy,cmd_w\n"
                "112.000000345,0.2,0.0,0.1\n"
            )
            report = analyze(
                database,
                root / "clearance.csv",
                diagnostics,
                pointcloud_topic="/points",
                timestamp_basis="bag_record_timestamp",
            )
            self.assertEqual(report["timestamp_basis"], "bag_record_timestamp")
            self.assertEqual(report["command_pair_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
