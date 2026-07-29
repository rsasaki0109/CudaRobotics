#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from cudanav_real_dataset import read_json
from cudanav_rosbag_evidence import sha256_file
from run_cudanav_real_gpu_stack import (
    CLAIMS,
    STAGES,
    evaluate_manifest,
    evaluate_portable_evidence,
    make_manifest,
    make_portable_evidence,
    render_portable_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "cudanav_real_dataset_smoke.json"


class CudaNavRealGpuStackTest(unittest.TestCase):
    def make_fixture(self, output: Path) -> Path:
        spec = deepcopy(read_json(SPEC))
        database = output / spec["acquisition"]["expected_database"]
        database.write_bytes(b"database fixture")
        spec["acquisition"]["expected_database_bytes"] = database.stat().st_size
        spec["acquisition"]["expected_database_sha256"] = sha256_file(database)
        spec_path = output / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        for name, content in (
            ("sequence.bin", b"sequence"),
            ("export.json", b"{}"),
            ("result.json", b"{}"),
            ("runner.log", b"PASS\n"),
        ):
            (output / name).write_bytes(content)
        (output / "trajectory.csv").write_text(
            "frame,x\n0,0\n1,1\n",
            encoding="utf-8",
        )
        runner = output / "runner"
        runner.write_bytes(b"runner")
        export_report = {
            "database": {"sha256": sha256_file(database)},
            "pointcloud_topic": spec["recorded_inputs"]["pointcloud"]["topic"],
            "pose_topic": spec["recorded_inputs"]["odometry"]["topic"],
            "pose_type": spec["recorded_inputs"]["odometry"]["type"],
            "frame_id": "base_link",
            "frames": 2,
            "duration_s": 1.0,
            "start_offset_s": 1.0,
            "maximum_duration_s": 30.0,
            "maximum_pose_age_ms": 50.0,
            "pose_age_p95_ms": 0.1,
            "minimum_points": 40,
            "mean_points": 40.0,
            "maximum_points": 40,
            "reference_path_length_m": 1.0,
        }
        result = {
            "frames": 2,
            "duration_s": 1.0,
            "wall_time_ms": 2.0,
            "mean_frame_ms": 1.0,
            "frame_ms_p95": 1.0,
            "reference_path_length_m": 1.0,
            "estimated_path_length_m": 1.0,
            "ate_rmse_m": 0.1,
            "final_xy_error_m": 0.1,
            "final_drift_percent": 1.0,
            "yaw_error_p95_rad": 0.01,
            "inliers_min": 100,
            "nn_ms_p95": 0.1,
            "mapping": {
                "final_observed_voxels": 500,
                "total_integrated_rays": 1000,
                "map_shifts": 0,
                "maximum_occupied_cells": 10,
                "maximum_unknown_cells": 20,
                "raycast_ms_p95": 0.1,
                "projection_ms_p95": 0.1,
            },
            "esdf": {
                "unknown_policy": "free",
                "footprint_clearing_radius_m": 0.30,
                "max_distance_m": 2.0,
                "gpu_ms_p95": 0.1,
            },
            "mppi": {
                "control_stride": 10,
                "evaluations": 20,
                "minimum_nonzero_valid_rollout_ratio": 0.1,
                "maximum_robot_cost": 200,
                "minimum_robot_clearance_m": 0.3,
                "all_colliding_evaluations": 2,
                "retreating_evaluations": 2,
                "maximum_all_colliding_abs_v": 0.0,
                "invalid_commands": 0,
                "solve_ms_p95": 0.2,
            },
            "thresholds": {},
            "quality_pass": True,
            "gpu": {
                "name": "GPU fixture",
                "uuid": "GPU-00000000-0000-0000-0000-000000000000",
                "driver_version": 12000,
            },
            "stages": list(STAGES),
        }
        manifest = make_manifest(
            output,
            profile="smoke",
            git_commit="a" * 40,
            spec_path=spec_path,
            database=database,
            runner=runner,
            export_report=export_report,
            result=result,
            commands={"export": ["export"], "real_gpu_stack": ["runner"]},
        )
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_manifest_and_portable_evidence_bind_all_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self.make_fixture(Path(directory))
            self.assertTrue(
                evaluate_manifest(
                    manifest_path, expected_commit="a" * 40
                )["valid"]
            )
            portable = make_portable_evidence(
                manifest_path,
                result_id="fixture",
                publisher_commit="b" * 40,
            )
            self.assertTrue(
                evaluate_portable_evidence(
                    portable,
                    expected_source_commit="a" * 40,
                )["valid"]
            )
            markdown = render_portable_markdown(portable)
            self.assertIn("Safety-stop evaluations: 2", markdown)
            self.assertIn("ROS 2 runtime: no", markdown)
            self.assertIn("Closed-loop evidence: no", markdown)

    def test_rejects_closed_loop_relabel_and_missing_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self.make_fixture(Path(directory))
            payload = read_json(manifest_path)
            self.assertEqual(payload["claims"], CLAIMS)
            payload["claims"]["closed_loop"] = True
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(evaluate_manifest(manifest_path)["valid"])
            payload["claims"] = dict(CLAIMS)
            payload["stages"].remove("gpu_esdf")
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(evaluate_manifest(manifest_path)["valid"])


if __name__ == "__main__":
    unittest.main()
