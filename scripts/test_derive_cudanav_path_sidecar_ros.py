#!/usr/bin/env python3
"""ROS Jazzy integration test for the optional derived Path MCAP writer."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import tempfile

from test_export_rosbag_motion import Writer
from cudanav_real_dataset import DEFAULT_SPEC, read_json
from prepare_cudanav_istanbul_dataset import inspect
from validate_cudanav_real_dataset import evaluate


ROOT = Path(__file__).resolve().parents[1]


def odometry(sec: int, x: float, y: float, yaw: float) -> bytes:
    writer = Writer()
    writer.add("i", 4, sec)
    writer.add("I", 4, 0)
    writer.string("map")
    writer.string("base_link")
    writer.doubles([x, y, 1.0])
    writer.doubles([0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)])
    writer.doubles([0.0] * 36)
    writer.doubles([0.2, 0.0, 0.0, 0.0, 0.0, 0.1])
    writer.doubles([0.0] * 36)
    return bytes(writer.data)


def write_source_bag(root: Path) -> Path:
    root.mkdir()
    metadata = (
        "rosbag2_bagfile_information:\n"
        "  storage_identifier: sqlite3\n"
        "  topics_with_message_count:\n"
        "    - topic_metadata:\n"
        "        name: /pandar_points\n"
        "        type: sensor_msgs/msg/PointCloud2\n"
        "      message_count: 1\n"
        "    - topic_metadata:\n"
        "        name: /applanix/lvx_client/odom\n"
        "        type: nav_msgs/msg/Odometry\n"
        "      message_count: 3\n"
        "    - topic_metadata:\n"
        "        name: /tf_static\n"
        "        type: tf2_msgs/msg/TFMessage\n"
        "      message_count: 1\n"
    )
    (root / "metadata.yaml").write_text(metadata)
    spec = read_json(DEFAULT_SPEC)
    database = root / spec["acquisition"]["expected_database"]
    connection = sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE topics("
        "id INTEGER PRIMARY KEY, name TEXT, type TEXT, "
        "serialization_format TEXT, offered_qos_profiles TEXT);"
        "CREATE TABLE messages("
        "id INTEGER PRIMARY KEY, topic_id INTEGER, "
        "timestamp INTEGER, data BLOB);"
        "INSERT INTO topics VALUES("
        "1, '/pandar_points', "
        "'sensor_msgs/msg/PointCloud2', 'cdr', '');"
        "INSERT INTO topics VALUES("
        "2, '/applanix/lvx_client/odom', "
        "'nav_msgs/msg/Odometry', 'cdr', '');"
        "INSERT INTO topics VALUES("
        "3, '/tf_static', "
        "'tf2_msgs/msg/TFMessage', 'cdr', '');"
    )
    connection.executemany(
        "INSERT INTO messages(topic_id, timestamp, data) VALUES(2, ?, ?)",
        [
            (10_000_000_000, odometry(10, 5.0, 2.0, 0.2)),
            (11_000_000_000, odometry(11, 5.5, 2.1, 0.2)),
            (12_000_000_000, odometry(12, 6.0, 2.2, 0.3)),
        ],
    )
    connection.executemany(
        "INSERT INTO messages(topic_id, timestamp, data) VALUES(?, ?, ?)",
        [
            (1, 10_000_000_000, b"pointcloud fixture"),
            (3, 10_000_000_000, b"static tf fixture"),
        ],
    )
    connection.commit()
    connection.close()
    return database


def acquisition_probe(spec: dict) -> dict:
    acquisition = spec["acquisition"]
    return {
        "schema_version": 1,
        "database": {
            "file_id": acquisition["file_id"],
            "filename": acquisition["expected_database"],
            "bytes": acquisition["expected_database_bytes"],
        },
        "metadata": {
            "file_id": acquisition["metadata_file_id"],
            "filename": acquisition["expected_metadata"],
            "bytes": acquisition["expected_metadata_bytes"],
        },
        "checks": {
            "database_filename": True,
            "database_bytes": True,
            "metadata_filename": True,
            "metadata_bytes": True,
        },
        "passed": True,
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        database = write_source_bag(source)
        spec = read_json(DEFAULT_SPEC)
        acquisition_report = source / "inspection.json"
        acquisition_report.write_text(
            json.dumps(
                inspect(source, remote_probe=acquisition_probe(spec)),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        sidecar = root / "sidecar"
        report = root / "generator.json"
        materialization = root / "materialization.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "derive_cudanav_path_sidecar.py"),
                "--source-bag",
                str(source),
                "--database",
                str(database),
                "--output-bag",
                str(sidecar),
                "--report",
                str(report),
                "--acquisition-report",
                str(acquisition_report),
                "--materialization",
                str(materialization),
                "--storage",
                "mcap",
            ],
            cwd=ROOT,
            check=True,
        )
        info = subprocess.run(
            ["ros2", "bag", "info", str(sidecar)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        ).stdout
        if "/cuda_nav/derived_plan" not in info or "nav_msgs/msg/Path" not in info:
            raise AssertionError(info)
        gate = evaluate(
            ROOT / "docs" / "cudanav_real_dataset.json",
            materialization,
        )
        if not gate["ready"]:
            raise AssertionError(json.dumps(gate, indent=2))
        generator = json.loads(report.read_text())
        if (
            generator["input_samples"] != 3
            or generator["output_poses"] < 2
            or generator["storage_id"] != "mcap"
        ):
            raise AssertionError(generator)
    print("ROS derived Path sidecar round trip passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
