#!/usr/bin/env python3

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from cudanav_evidence import (
    evaluate_manifest,
    evaluate_summary,
    validate_summary,
)
from render_cudanav_trajectory import load_trajectory, render


def valid_summary() -> dict:
    return {
        "schema_version": 1,
        "success": True,
        "elapsed_sec": 40.0,
        "collision": False,
        "collision_count": 0,
        "ground_truth_distance_m": 11.8,
        "ground_truth_goal_distance_m": 0.1,
        "odometry_position_error_m": 0.08,
        "odometry_drift_percent": 0.68,
        "command_intervals": 700,
        "command_deadline_misses": 2,
        "command_deadline_miss_rate": 2 / 700,
        "traversals_requested": 1,
        "traversals_completed": 1,
        "trajectory_csv": "trajectory.csv",
        "diagnostic_error_count": 0,
        "diagnostic_warn_count": 0,
        "diagnostic_status_samples": 30,
        "diagnostic_components": ["esdf", "mapping", "odometry"],
        "failure_counters": {
            "esdf:maps_dropped": 0,
            "mapping:maps_dropped": 0,
            "odometry:schema_failures": 0,
        },
    }


class CudaNavEvidenceTest(unittest.TestCase):
    def test_smoke_passes_and_release_requires_duration(self):
        summary = valid_summary()
        self.assertTrue(evaluate_summary(summary, "smoke")["passed"])
        release = evaluate_summary(summary, "release")
        self.assertFalse(release["passed"])
        self.assertFalse(release["checks"]["minimum_duration"])

    def test_nan_and_missing_values_are_rejected(self):
        summary = valid_summary()
        summary["odometry_drift_percent"] = float("nan")
        self.assertIn(
            "odometry_drift_percent must be finite",
            validate_summary(summary),
        )
        del summary["collision_count"]
        self.assertTrue(validate_summary(summary)[0].startswith("missing"))

    def test_release_manifest_requires_bag_and_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("mission_summary.json", "launch.log", "controller.yaml"):
                (root / name).write_text("{}\n", encoding="utf-8")
            (root / "trajectory.csv").write_text(
                "elapsed_sec,truth_x,truth_y,odom_x,odom_y\n"
                "0.0,0.0,0.0,,\n"
                "1.0,1.0,0.0,0.9,0.0\n",
                encoding="utf-8",
            )
            config_hash = hashlib.sha256(
                (root / "controller.yaml").read_bytes()
            ).hexdigest()
            manifest = {
                "schema_version": 1,
                "git_commit": "a" * 40,
                "git_dirty": False,
                "config_sha256": config_hash,
                "command": [
                    "ros2",
                    "launch",
                    "cuda_nav_bringup",
                    "cudanav_closed_loop.launch.py",
                    f"controller_config:={root / 'controller.yaml'}",
                ],
                "gpu": [
                    {
                        "physical_index": "0",
                        "name": "test",
                        "uuid": "GPU-test",
                        "driver_version": "999.0",
                        "memory_total_mib": "8192",
                    }
                ],
                "artifacts": {
                    "summary": "mission_summary.json",
                    "trajectory": "trajectory.csv",
                    "launch_log": "launch.log",
                    "controller_config": "controller.yaml",
                    "rosbag": None,
                    "video": None,
                },
            }
            smoke = evaluate_manifest(manifest, root, "smoke")
            release = evaluate_manifest(manifest, root, "release")
            self.assertTrue(smoke["passed"])
            self.assertFalse(release["passed"])
            self.assertFalse(release["checks"]["artifact_rosbag"])
            self.assertFalse(release["checks"]["artifact_video"])

            manifest["bag_command"] = ["ros2", "bag", "record"]
            requested_bag = evaluate_manifest(manifest, root, "smoke")
            self.assertFalse(requested_bag["checks"]["artifact_rosbag"])
            manifest["bag_command"] = None
            manifest["config_sha256"] = "0" * 64
            mismatch = evaluate_manifest(manifest, root, "smoke")
            self.assertFalse(mismatch["checks"]["config_sha256_matches"])
            manifest["config_sha256"] = config_hash
            manifest["command"][-1] = (
                f"controller_config:={root / 'other.yaml'}"
            )
            command_mismatch = evaluate_manifest(manifest, root, "smoke")
            self.assertFalse(
                command_mismatch["checks"][
                    "controller_config_command_binding"
                ]
            )
            other = root / "other"
            other.mkdir()
            (other / "controller.yaml").write_text(
                "controller: changed\n", encoding="utf-8"
            )
            manifest["command"][-1] = (
                f"controller_config:={other / 'controller.yaml'}"
            )
            content_mismatch = evaluate_manifest(manifest, root, "smoke")
            self.assertFalse(
                content_mismatch["checks"][
                    "controller_config_command_binding"
                ]
            )
            manifest["command"][-1] = (
                "controller_config:=/unavailable/original/controller.yaml"
            )
            portable = evaluate_manifest(manifest, root, "smoke")
            self.assertTrue(
                portable["checks"]["controller_config_command_binding"]
            )
            manifest["command"][-1] = (
                f"controller_config:={root / 'controller.yaml'}"
            )
            manifest["artifacts"]["summary"] = "../outside.json"
            traversal = evaluate_manifest(manifest, root, "smoke")
            self.assertFalse(traversal["checks"]["artifact_summary"])

    def test_summary_is_strict_json_serializable(self):
        encoded = json.dumps(valid_summary(), allow_nan=False)
        self.assertEqual(json.loads(encoded)["schema_version"], 1)

    def test_negative_and_impossible_counts_are_rejected(self):
        summary = valid_summary()
        summary["ground_truth_distance_m"] = -1.0
        summary["command_deadline_misses"] = (
            summary["command_intervals"] + 1
        )
        errors = validate_summary(summary)
        self.assertIn("ground_truth_distance_m must be non-negative", errors)
        self.assertIn(
            "command_deadline_misses cannot exceed command_intervals",
            errors,
        )

    def test_trajectory_renderer_writes_animated_gif(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "trajectory.csv"
            csv_path.write_text(
                "elapsed_sec,truth_x,truth_y,odom_x,odom_y\n"
                "0.0,0.0,0.0,0.0,0.0\n"
                "1.0,1.0,0.2,0.95,0.18\n"
                "2.0,2.0,0.0,1.9,0.02\n",
                encoding="utf-8",
            )
            output = root / "trajectory.gif"
            render(load_trajectory(csv_path), output, max_frames=3)
            self.assertTrue(output.is_file())
            self.assertEqual(output.read_bytes()[:6], b"GIF89a")


if __name__ == "__main__":
    unittest.main()
