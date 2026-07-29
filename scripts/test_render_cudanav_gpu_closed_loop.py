#!/usr/bin/env python3
"""Tests for the content-bound native CudaNav visualizer."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from render_cudanav_gpu_closed_loop import (
    read_evidence,
    read_trajectory,
    sample_indices,
)


COLUMNS = [
    "step",
    "traversal",
    "time_s",
    "truth_x",
    "truth_y",
    "truth_yaw",
    "estimated_x",
    "estimated_y",
    "estimated_yaw",
    "error_m",
    "inliers",
    "observed_voxels",
    "occupied_cells",
    "valid_rollout_ratio",
    "all_colliding",
    "retreating",
    "command_v",
    "command_w",
    "solve_ms",
    "frame_ms",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root: Path) -> tuple[Path, Path]:
    trajectory = root / "trajectory.csv"
    with trajectory.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for step in range(12):
            writer.writerow(
                {
                    "step": step,
                    "traversal": step // 4,
                    "time_s": step * 0.1,
                    "truth_x": step * 0.1,
                    "truth_y": 0,
                    "truth_yaw": 0,
                    "estimated_x": step * 0.1 + 0.001,
                    "estimated_y": 0,
                    "estimated_yaw": 0,
                    "error_m": 0.001,
                    "inliers": 100,
                    "observed_voxels": 500,
                    "occupied_cells": 20,
                    "valid_rollout_ratio": 1,
                    "all_colliding": 0,
                    "retreating": 0,
                    "command_v": 0.2,
                    "command_w": 0,
                    "solve_ms": 0.4,
                    "frame_ms": 4,
                }
            )
    evidence = root / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "profile": "release",
                "claims": {
                    "native_gpu_core_closed_loop": True,
                    "real_data": False,
                    "ros2_runtime": False,
                },
                "checks": {"quality": True},
                "metrics": {"frames": 12, "traversals_completed": 3},
                "artifacts": {
                    "trajectory": {"sha256": sha256(trajectory)}
                },
            }
        ),
        encoding="utf-8",
    )
    return evidence, trajectory


class CudaNavVisualTest(unittest.TestCase):
    def test_trajectory_and_traversal_boundaries_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence_path, trajectory = fixture(Path(directory))
            evidence = read_evidence(evidence_path)
            rows = read_trajectory(trajectory, evidence)
            indices = sample_indices(rows, 8)
            self.assertEqual(len(indices), 8)
            self.assertTrue({0, 3, 4, 7, 8, 11}.issubset(indices))

    def test_tampered_trajectory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence_path, trajectory = fixture(Path(directory))
            evidence = read_evidence(evidence_path)
            trajectory.write_text(
                trajectory.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AssertionError, "SHA-256"):
                read_trajectory(trajectory, evidence)

    def test_failed_source_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence_path, _ = fixture(Path(directory))
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            payload["checks"]["quality"] = False
            evidence_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "every check"):
                read_evidence(evidence_path)


if __name__ == "__main__":
    unittest.main()
