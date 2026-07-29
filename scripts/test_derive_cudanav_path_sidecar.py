#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path
import sqlite3
import tempfile
import unittest

from derive_cudanav_path_sidecar import (
    derive_path,
    serialize_path,
    write_sqlite_rosbag,
)
from export_rosbag_motion import CdrReader


def row(stamp: int, x: float, y: float, yaw: float) -> dict:
    return {"stamp_ns": stamp, "x": x, "y": y, "z": 2.0, "yaw": yaw}


class DerivedCudaNavPathTest(unittest.TestCase):
    def test_path_is_sorted_deduplicated_and_start_normalized(self) -> None:
        poses = derive_path(
            [
                row(30, 10.0, 3.0, math.pi / 2),
                row(10, 10.0, 2.0, math.pi / 2),
                row(10, 99.0, 99.0, 0.0),
                row(20, 10.0, 2.01, math.pi / 2),
            ],
            0.05,
        )
        self.assertEqual([pose["stamp_ns"] for pose in poses], [10, 30])
        self.assertAlmostEqual(poses[0]["x"], 0.0)
        self.assertAlmostEqual(poses[0]["y"], 0.0)
        self.assertAlmostEqual(poses[-1]["x"], 1.0)
        self.assertAlmostEqual(poses[-1]["y"], 0.0, places=12)
        self.assertAlmostEqual(poses[-1]["yaw"], 0.0)

    def test_translation_decimation_keeps_final_pose(self) -> None:
        poses = derive_path(
            [row(1, 0.0, 0.0, 0.0), row(2, 0.02, 0.0, 0.0), row(3, 0.04, 0.0, 0.0)],
            0.05,
        )
        self.assertEqual([pose["stamp_ns"] for pose in poses], [1, 3])

    def test_smoke_duration_cap_is_explicit_and_keeps_boundary(self) -> None:
        poses = derive_path(
            [
                row(1_000_000_000, 0.0, 0.0, 0.0),
                row(2_000_000_000, 1.0, 0.0, 0.0),
                row(4_000_000_000, 3.0, 0.0, 0.0),
            ],
            0.05,
            1.0,
        )
        self.assertEqual(
            [pose["stamp_ns"] for pose in poses],
            [1_000_000_000, 2_000_000_000],
        )

    def test_empty_singleton_and_invalid_threshold_fail(self) -> None:
        with self.assertRaises(ValueError):
            derive_path([], 0.05)
        with self.assertRaises(ValueError):
            derive_path([row(1, 0.0, 0.0, 0.0)], 0.05)
        with self.assertRaises(ValueError):
            derive_path([row(1, 0.0, 0.0, 0.0), row(2, 1.0, 0.0, 0.0)], -1.0)
        with self.assertRaises(ValueError):
            derive_path(
                [row(1, 0.0, 0.0, 0.0), row(2, 1.0, 0.0, 0.0)],
                0.05,
                0.0,
            )

    def test_dependency_free_sqlite_sidecar_contains_path(self) -> None:
        poses = derive_path(
            [row(1_000_000_001, 0.0, 0.0, 0.0), row(2_000_000_002, 1.0, 0.0, 0.1)],
            0.05,
        )
        payload = serialize_path("odom", poses)
        reader = CdrReader(payload)
        self.assertEqual(reader.int32(), 1)
        self.assertEqual(reader.uint32(), 1)
        self.assertEqual(reader.string(), "odom")
        self.assertEqual(reader.uint32(), 2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sidecar"
            write_sqlite_rosbag(
                output,
                "/cuda_nav/derived_plan",
                "odom",
                poses,
                5_000_000_000,
            )
            connection = sqlite3.connect(output / "path_sidecar_0.db3")
            try:
                topic = connection.execute(
                    "SELECT name, type FROM topics"
                ).fetchone()
                message = connection.execute(
                    "SELECT timestamp, length(data) FROM messages"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(
                topic,
                ("/cuda_nav/derived_plan", "nav_msgs/msg/Path"),
            )
            self.assertEqual(message, (5_000_000_000, len(payload)))
            self.assertIn(
                "storage_identifier: sqlite3",
                (output / "metadata.yaml").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
