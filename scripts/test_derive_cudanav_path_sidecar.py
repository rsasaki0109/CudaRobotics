#!/usr/bin/env python3

from __future__ import annotations

import math
import unittest

from derive_cudanav_path_sidecar import derive_path


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

    def test_empty_singleton_and_invalid_threshold_fail(self) -> None:
        with self.assertRaises(ValueError):
            derive_path([], 0.05)
        with self.assertRaises(ValueError):
            derive_path([row(1, 0.0, 0.0, 0.0)], 0.05)
        with self.assertRaises(ValueError):
            derive_path([row(1, 0.0, 0.0, 0.0), row(2, 1.0, 0.0, 0.0)], -1.0)


if __name__ == "__main__":
    unittest.main()
