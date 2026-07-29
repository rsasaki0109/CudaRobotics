#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from cudanav_multi_gpu import evaluate_multi_gpu_suite
from run_cudanav_closed_loop import parse_gpu_identity
from run_cudanav_multi_gpu import import_suite


def summary() -> dict:
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
            "odometry:scans_dropped": 0,
        },
    }


def write_run(
    root: Path,
    relative: str,
    gpu_name: str,
    gpu_uuid: str,
    physical_index: str,
    config_text: str = "controller: {}\n",
) -> None:
    run = root / relative
    run.mkdir(parents=True)
    (run / "mission_summary.json").write_text(
        json.dumps(summary()) + "\n", encoding="utf-8"
    )
    (run / "trajectory.csv").write_text(
        "elapsed_sec,truth_x,truth_y,odom_x,odom_y\n"
        "0.0,0.0,0.0,,\n"
        "1.0,1.0,0.0,0.9,0.0\n",
        encoding="utf-8",
    )
    (run / "launch.log").write_text("launch complete\n", encoding="utf-8")
    config = run / "controller.yaml"
    config.write_text(config_text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "artifact_sha256": {
            name: hashlib.sha256((run / relative).read_bytes()).hexdigest()
            for name, relative in {
                "summary": "mission_summary.json",
                "trajectory": "trajectory.csv",
                "launch_log": "launch.log",
                "controller_config": "controller.yaml",
            }.items()
        },
        "command": [
            "ros2",
            "launch",
            "cuda_nav_bringup",
            "cudanav_closed_loop.launch.py",
            f"controller_config:={config}",
        ],
        "gpu": [
            {
                "physical_index": physical_index,
                "name": gpu_name,
                "uuid": gpu_uuid,
                "driver_version": "999.0",
                "memory_total_mib": "8192",
            }
        ],
        "traversal_count": 1,
        "bag_command": None,
        "render_command": None,
        "artifacts": {
            "summary": "mission_summary.json",
            "trajectory": "trajectory.csv",
            "launch_log": "launch.log",
            "controller_config": "controller.yaml",
            "rosbag": None,
            "video": None,
        },
    }
    (run / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )


def manifest_sha256(root: Path, relative: str) -> str:
    return hashlib.sha256(
        (root / relative / "manifest.json").read_bytes()
    ).hexdigest()


def write_native_run(
    root: Path,
    relative: str,
    gpu_name: str,
    gpu_uuid: str,
    physical_index: str = "0",
) -> None:
    run = root / relative
    run.mkdir(parents=True)
    result = {
        "scenario": "cudanav_s_course",
        "stages": [
            "gpu_kiss_icp",
            "gpu_voxel_mapping",
            "gpu_esdf",
            "cuda_mppi",
            "command_driven_plant",
        ],
        "claims": {
            "native_gpu_core_closed_loop": True,
            "ros2_runtime": False,
            "real_data": False,
        },
        "goal_reached": True,
        "ground_truth_goal_distance_m": 0.2,
        "collision_count": 0,
        "odometry_drift_percent": 0.1,
        "command_deadline_miss_rate": 0.0,
        "causal_command_effect": True,
        "command_effect_distance_m": 330.0,
        "invalid_commands": 0,
        "minimum_inliers": 100,
        "final_observed_voxels": 1000,
        "maximum_occupied_cells": 100,
        "all_colliding_evaluations": 0,
        "minimum_nonzero_valid_rollout_ratio": 0.1,
        "ground_truth_distance_m": 330.0,
        "frames": 7000,
        "simulated_duration_s": 700.0,
        "traversals_requested": 30,
        "traversals_completed": 30,
        "traversal_frames": [233] * 30,
        "gpu": {"name": gpu_name, "driver_version": 12000},
        "quality_pass": True,
    }
    result_path = run / "result.json"
    trajectory_path = run / "trajectory.csv"
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    trajectory_path.write_text("step,x\n0,0\n", encoding="utf-8")

    def artifact(path: Path) -> dict:
        return {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    checks = {
        "release": True,
        "gpu_binding": True,
        "clean_checkout": True,
    }
    manifest = {
        "schema_version": 1,
        "profile": "release",
        "source_commit": "a" * 40,
        "source_digest": "b" * 64,
        "git_dirty": False,
        "gpu_identity": {
            "physical_index": physical_index,
            "name": gpu_name,
            "uuid": gpu_uuid,
            "driver_version": "999.0",
        },
        "checks": checks,
        "artifacts": {
            "result": artifact(result_path),
            "trajectory": artifact(trajectory_path),
        },
    }
    (run / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )


class MultiGpuEvidenceTest(unittest.TestCase):
    def test_visible_device_filter_records_physical_gpu(self):
        output = (
            "0, GPU A, GPU-a, 999.0, 8192\n"
            "1, GPU B, GPU-b, 999.0, 16384\n"
        )
        selected = parse_gpu_identity(output, "1")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["physical_index"], "1")
        self.assertEqual(selected[0]["uuid"], "GPU-b")
        self.assertEqual(parse_gpu_identity(output, "-1"), [])
        self.assertEqual(len(parse_gpu_identity(output, "")), 2)

    def test_two_model_suite_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu_a = {"index": "0", "name": "GPU A", "uuid": "GPU-a"}
            gpu_b = {"index": "1", "name": "GPU B", "uuid": "GPU-b"}
            write_run(root, "gpu_0/run_00", "GPU A", "GPU-a", "0")
            write_run(root, "gpu_1/run_00", "GPU B", "GPU-b", "1")
            suite = {
                "schema_version": 1,
                "profile": "smoke",
                "devices": [gpu_a, gpu_b],
                "repetitions": 1,
                "minimum_gpu_devices": 2,
                "minimum_gpu_models": 2,
                "runs": [
                    {
                        "directory": "gpu_0/run_00",
                        "returncode": 0,
                        "device": gpu_a,
                        "manifest_sha256": manifest_sha256(
                            root, "gpu_0/run_00"
                        ),
                    },
                    {
                        "directory": "gpu_1/run_00",
                        "returncode": 0,
                        "device": gpu_b,
                        "manifest_sha256": manifest_sha256(
                            root, "gpu_1/run_00"
                        ),
                    },
                ],
            }
            gate = evaluate_multi_gpu_suite(suite, root)
            self.assertTrue(gate["passed"])
            child_manifest = root / "gpu_0/run_00/manifest.json"
            payload = json.loads(child_manifest.read_text(encoding="utf-8"))
            payload["post_collection_edit"] = True
            child_manifest.write_text(
                json.dumps(payload) + "\n", encoding="utf-8"
            )
            tampered = evaluate_multi_gpu_suite(suite, root)
            self.assertFalse(tampered["passed"])
            self.assertFalse(
                tampered["runs"][0]["manifest_binding"]
            )

    def test_same_model_and_config_drift_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu_a = {"index": "0", "name": "GPU A", "uuid": "GPU-a"}
            gpu_b = {"index": "1", "name": "GPU A", "uuid": "GPU-b"}
            write_run(root, "gpu_0/run_00", "GPU A", "GPU-a", "0")
            write_run(
                root,
                "gpu_1/run_00",
                "GPU A",
                "GPU-b",
                "1",
                config_text="controller: {changed: true}\n",
            )
            suite = {
                "schema_version": 1,
                "profile": "smoke",
                "devices": [gpu_a, gpu_b],
                "repetitions": 1,
                "minimum_gpu_devices": 2,
                "minimum_gpu_models": 2,
                "runs": [
                    {
                        "directory": "gpu_0/run_00",
                        "returncode": 0,
                        "device": gpu_a,
                        "manifest_sha256": manifest_sha256(
                            root, "gpu_0/run_00"
                        ),
                    },
                    {
                        "directory": "gpu_1/run_00",
                        "returncode": 0,
                        "device": gpu_b,
                        "manifest_sha256": manifest_sha256(
                            root, "gpu_1/run_00"
                        ),
                    },
                ],
            }
            gate = evaluate_multi_gpu_suite(suite, root)
            self.assertFalse(gate["passed"])
            self.assertFalse(gate["checks"]["same_config"])
            self.assertFalse(gate["checks"]["gpu_model_coverage"])

    def test_suite_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = {
                "schema_version": 1,
                "profile": "smoke",
                "devices": [
                    {"index": "0", "name": "GPU A", "uuid": "GPU-a"}
                ],
                "repetitions": 1,
                "minimum_gpu_devices": 1,
                "minimum_gpu_models": 1,
                "runs": [
                    {
                        "directory": "../escape",
                        "returncode": 0,
                        "device": {
                            "index": "0",
                            "name": "GPU A",
                            "uuid": "GPU-a",
                        },
                    }
                ],
            }
            gate = evaluate_multi_gpu_suite(suite, root)
            self.assertFalse(gate["passed"])
            self.assertIn("escapes suite", gate["runs"][0]["error"])

    def test_cross_machine_import_accepts_same_commit_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "source_a"
            source_b = root / "source_b"
            output = root / "suite"
            output.mkdir()
            write_run(root, "source_a", "GPU A", "GPU-a", "0")
            write_run(root, "source_b", "GPU B", "GPU-b", "0")
            suite = import_suite(
                [source_a, source_b],
                output,
                minimum_gpu_devices=2,
                minimum_gpu_models=2,
            )
            self.assertEqual(suite["collection_mode"], "cross_machine_import")
            self.assertEqual(len(suite["devices"]), 2)
            gate = evaluate_multi_gpu_suite(suite, output)
            self.assertTrue(gate["passed"], gate)

    def test_cross_machine_import_rejects_invalid_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "suite"
            output.mkdir()
            write_run(root, "source", "GPU A", "GPU-a", "0")
            (source / "controller.yaml").write_text(
                "controller: tampered\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "failed smoke validation"):
                import_suite(
                    [source],
                    output,
                    minimum_gpu_devices=1,
                    minimum_gpu_models=1,
                )

    def test_native_release_import_requires_two_physical_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "native_a"
            source_b = root / "native_b"
            output = root / "suite"
            output.mkdir()
            write_native_run(root, "native_a", "GPU A", "GPU-native-a")
            write_native_run(root, "native_b", "GPU B", "GPU-native-b")
            suite = import_suite(
                [source_a, source_b],
                output,
                minimum_gpu_devices=2,
                minimum_gpu_models=2,
                evidence_kind="native-release",
            )
            self.assertEqual(suite["profile"], "release")
            self.assertEqual(suite["evidence_kind"], "native_release")
            gate = evaluate_multi_gpu_suite(suite, output)
            self.assertTrue(gate["passed"], gate)

    def test_native_release_import_rejects_dirty_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "native"
            output = root / "suite"
            output.mkdir()
            write_native_run(root, "native", "GPU A", "GPU-native-a")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["git_dirty"] = True
            manifest_path.write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "failed native release validation"
            ):
                import_suite(
                    [source],
                    output,
                    minimum_gpu_devices=1,
                    minimum_gpu_models=1,
                    evidence_kind="native-release",
                )


if __name__ == "__main__":
    unittest.main()
