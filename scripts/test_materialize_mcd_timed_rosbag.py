#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest

from materialize_mcd_timed_rosbag import (
    compose_sensor_pose,
    decimal_stamp_ns,
    initialize_database,
    read_ground_truth,
    read_os_sensor_extrinsic,
    select_ground_truth,
    sha256_file,
    verify_sources,
)


class MaterializeMcdTimedRosbagTest(unittest.TestCase):
    def test_decimal_timestamps_do_not_round_through_float(self) -> None:
        self.assertEqual(
            decimal_stamp_ns("1644824097.448929787"),
            1_644_824_097_448_929_787,
        )

    def test_ground_truth_requires_contiguous_one_based_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pose.csv"
            source.write_text(
                "num,t,x,y,z,qx,qy,qz,qw\n"
                "11,1.000000001,0,0,0,0,0,0,1\n"
                "12,1.100000001,1,0,0,0,0,0,1\n",
                encoding="utf-8",
            )
            rows = read_ground_truth(source)
            self.assertEqual(rows[0]["stamp_ns"], 1_000_000_001)
            self.assertEqual(rows[-1]["num"], 12)
            source.write_text(
                source.read_text(encoding="utf-8").replace(
                    "12,1.100000001", "13,1.100000001"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "contiguous"):
                read_ground_truth(source)

    def test_calibration_parser_selects_body_os_sensor_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "calibration.yaml"
            source.write_text(
                "body:\n"
                "    os_imu:\n"
                "        T:\n"
                "        - [9, 0, 0, 0]\n"
                "    os_sensor:\n"
                "        T:\n"
                "        - [1, 0, 0, 0.1]\n"
                "        - [0, 1, 0, 0.2]\n"
                "        - [0, 0, 1, 0.3]\n"
                "        - [0, 0, 0, 1]\n"
                "    vn100_imu:\n"
                "        T:\n"
                "        - [8, 0, 0, 0]\n",
                encoding="utf-8",
            )
            matrix = read_os_sensor_extrinsic(source)
        self.assertEqual(matrix[0], [1.0, 0.0, 0.0, 0.1])
        self.assertEqual(matrix[2][3], 0.3)

    def test_pose_composition_applies_rotated_sensor_offset(self) -> None:
        half = math.sqrt(0.5)
        body = {
            "x": 10.0,
            "y": 20.0,
            "z": 30.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": half,
            "qw": half,
        }
        extrinsic = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        sensor = compose_sensor_pose(body, extrinsic)
        self.assertAlmostEqual(sensor["x"], 10.0)
        self.assertAlmostEqual(sensor["y"], 21.0)
        self.assertAlmostEqual(sensor["z"], 30.0)
        self.assertAlmostEqual(sensor["qz"], half)
        self.assertAlmostEqual(sensor["qw"], half)

    def test_selection_freezes_one_based_gt_to_zero_based_lidar(self) -> None:
        rows = [
            {
                "num": number,
                "stamp_ns": 1_000_000_000 + (number - 1) * 100_000_000,
            }
            for number in range(2, 22)
        ]
        selected = select_ground_truth(
            rows,
            first_cloud_stamp_ns=1_000_000_000,
            start_offset_s=1.0,
            duration_s=0.3,
            cloud_count=30,
        )
        self.assertEqual(selected[0]["num"], 11)
        self.assertEqual(selected[0]["num"] - 1, 10)
        self.assertEqual(selected[-1]["num"], 14)

    def test_source_verification_rejects_content_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"contract")
            import hashlib

            contract = {
                "source_artifacts": {
                    "source": {
                        "filename": source.name,
                        "bytes": source.stat().st_size,
                        "sha256": hashlib.sha256(b"contract").hexdigest(),
                    }
                }
            }
            verified = verify_sources(contract, root)
            self.assertEqual(verified["source"]["bytes"], 8)
            source.write_bytes(b"replaced")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_sources(contract, root)

    def test_initialized_database_accepts_explicit_bulk_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "output.db3"
            connection = initialize_database(database, "/points")
            try:
                connection.execute("BEGIN")
                connection.execute(
                    "INSERT INTO messages VALUES(1, 1, 10, ?)",
                    (sqlite3.Binary(b"cloud"),),
                )
                connection.commit()
            finally:
                connection.close()

    def test_candidate_contract_is_machine_readable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        candidate = root / "docs" / "cudanav_timed_dataset_mcd_ntu_day_02.json"
        document = json.loads(candidate.read_text(encoding="utf-8"))
        self.assertEqual(
            document["pointcloud"]["point_time"]["unit"], "nanoseconds"
        )
        self.assertEqual(document["pointcloud"]["ring"]["datatype"], 2)

    def test_materialized_contract_binds_repo_sources(self) -> None:
        root = Path(__file__).resolve().parents[1]
        contract = json.loads(
            (
                root
                / "docs"
                / "cudanav_timed_dataset_mcd_ntu_day_02_materialized.json"
            ).read_text(encoding="utf-8")
        )
        bindings = (
            (
                contract["source_contract"]["path"],
                contract["source_contract"]["sha256"],
            ),
            (
                contract["materializer"]["path"],
                contract["materializer"]["sha256_at_materialization"],
            ),
            (
                contract["materializer"]["requirements"],
                contract["materializer"]["requirements_sha256"],
            ),
            (
                contract["timing_admission"]["inspector"],
                contract["timing_admission"]["inspector_sha256"],
            ),
        )
        for relative, expected in bindings:
            self.assertEqual(sha256_file(root / relative), expected)
        self.assertTrue(
            contract["readiness"]["timing_admission_passed_all_frames"]
        )
        self.assertFalse(contract["readiness"]["ready"])


if __name__ == "__main__":
    unittest.main()
