#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest

from cudanav_real_dataset import read_json
from cudanav_rosbag_evidence import sha256_file
from export_cudanav_kiss_icp_sequence import MAGIC, export_sequence
from run_cudanav_kiss_icp_real import (
    evaluate_manifest,
    evaluate_portable_evidence,
    make_manifest,
    make_portable_evidence,
    render_portable_markdown,
)
from test_analyze_pointcloud2_clearance import pointcloud
from test_export_rosbag_motion import Writer


ROOT = Path(__file__).resolve().parents[1]
SMOKE_SPEC = ROOT / "docs" / "cudanav_real_dataset_smoke.json"


def pose(sec: int, x: float, y: float, yaw: float = 0.0) -> bytes:
    writer = Writer()
    writer.add("i", 4, sec)
    writer.add("I", 4, 345)
    writer.string("map")
    writer.doubles([x, y, 0.0])
    writer.doubles(
        [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)]
    )
    return bytes(writer.data)


def sequence_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT);"
        "CREATE TABLE messages("
        "id INTEGER PRIMARY KEY, topic_id INTEGER, "
        "timestamp INTEGER, data BLOB);"
        "INSERT INTO topics VALUES("
        "1, '/points', 'sensor_msgs/msg/PointCloud2');"
        "INSERT INTO topics VALUES("
        "2, '/pose', 'geometry_msgs/msg/PoseStamped');"
    )
    points = [
        (1.0 + 0.01 * index, 0.2 * (index % 5), 0.1)
        for index in range(40)
    ]
    for index, sec in enumerate((12, 13, 14)):
        connection.execute(
            "INSERT INTO messages(topic_id, timestamp, data) VALUES(1, ?, ?)",
            (sec * 1_000_000_000 + 345, pointcloud(points, sec)),
        )
        connection.execute(
            "INSERT INTO messages(topic_id, timestamp, data) VALUES(2, ?, ?)",
            (sec * 1_000_000_000 + 345, pose(sec, float(index), 0.0)),
        )
    connection.commit()
    connection.close()


class CudaNavKissIcpRealTest(unittest.TestCase):
    def test_export_sequence_binds_clouds_and_normalized_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "fixture.db3"
            sequence_database(database)
            sequence = root / "sequence.bin"
            report = export_sequence(
                database,
                sequence,
                pointcloud_topic="/points",
                pose_topic="/pose",
                pose_type="geometry_msgs/msg/PoseStamped",
                start_offset_s=0.0,
                maximum_duration_s=5.0,
                maximum_frames=3,
                maximum_pose_age_ms=1.0,
                minimum_range_m=0.1,
                maximum_range_m=10.0,
            )
            self.assertEqual(report["frames"], 3)
            self.assertEqual(report["reference_path_length_m"], 2.0)
            self.assertEqual(report["database"]["sha256"], sha256_file(database))
            with sequence.open("rb") as stream:
                self.assertEqual(stream.read(8), MAGIC)
                self.assertEqual(struct.unpack("<II", stream.read(8)), (1, 3))
                first = struct.unpack("<QffffI", stream.read(28))
                stream.seek(first[-1] * 12, 1)
                second = struct.unpack("<QffffI", stream.read(28))
            self.assertEqual(first[1:5], (0.0, 0.0, 0.0, 0.0))
            self.assertEqual(second[1], 1.0)

    def test_manifest_is_content_bound_and_rejects_relabel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            spec = deepcopy(read_json(SMOKE_SPEC))
            database = output / spec["acquisition"]["expected_database"]
            database.write_bytes(b"database fixture")
            spec["acquisition"]["expected_database_bytes"] = database.stat().st_size
            spec["acquisition"]["expected_database_sha256"] = sha256_file(database)
            spec_path = output / "spec.json"
            spec_path.write_text(json.dumps(spec))
            for name, content in (
                ("sequence.bin", b"sequence"),
                ("export.json", b"{}"),
                ("result.json", b"{}"),
                ("runner.log", b"ok\n"),
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
                "pointcloud_topic": spec["recorded_inputs"]["pointcloud"][
                    "topic"
                ],
                "pose_topic": spec["recorded_inputs"]["odometry"]["topic"],
                "frames": 2,
                "first_stamp_ns": 1,
                "last_stamp_ns": 2,
            }
            result = {
                "frames": 2,
                "first_stamp_ns": 1,
                "last_stamp_ns": 2,
                "gpu": {
                    "name": "GPU fixture",
                    "uuid": "GPU-00000000-0000-0000-0000-000000000000",
                    "driver_version": 12000,
                },
                "nn_backend": "voxel",
                "duration_s": 1.0,
                "wall_time_ms": 2.0,
                "mean_frame_ms": 1.0,
                "reference_path_length_m": 1.0,
                "estimated_path_length_m": 1.0,
                "ate_rmse_m": 0.0,
                "final_xy_error_m": 0.0,
                "final_drift_percent": 0.0,
                "yaw_error_p95_rad": 0.0,
                "inliers_min": 100,
                "inliers_median": 100,
                "alignment_rmse_p95": 0.1,
                "nn_ms_p95": 0.1,
                "thresholds": {},
                "quality_pass": True,
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
                commands={"export": ["export"], "gpu_kiss_icp": ["runner"]},
            )
            manifest_path = output / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            self.assertTrue(
                evaluate_manifest(
                    manifest_path,
                    expected_commit="a" * 40,
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
            rendered = render_portable_markdown(portable)
            self.assertIn("Real PointCloud2 GPU odometry: yes", rendered)
            self.assertIn("GPU controller run: no", rendered)
            self.assertIn("Closed-loop evidence: no", rendered)
            relabeled = deepcopy(portable)
            relabeled["claims"]["closed_loop"] = True
            self.assertFalse(
                evaluate_portable_evidence(relabeled)["valid"]
            )
            manifest["claims"]["closed_loop"] = True
            manifest_path.write_text(json.dumps(manifest))
            self.assertFalse(evaluate_manifest(manifest_path)["valid"])


if __name__ == "__main__":
    unittest.main()
