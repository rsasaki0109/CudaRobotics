#!/usr/bin/env python3
"""Materialize a content-addressed MCD Ouster window as a ROS 2 SQLite bag."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import time
from typing import Any


DEFAULT_CANDIDATE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "cudanav_timed_dataset_mcd_ntu_day_02.json"
)
POSE_TOPIC = "/mcd/ground_truth/os_sensor_pose"
POSE_TYPE = "geometry_msgs/msg/PoseStamped"
POINTCLOUD_TYPE = "sensor_msgs/msg/PointCloud2"
MAXIMUM_PAIRING_ERROR_NS = 20_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_sources(
    candidate: dict[str, Any], source_dir: Path
) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for name, contract in candidate["source_artifacts"].items():
        path = source_dir / contract["filename"]
        if not path.is_file():
            raise ValueError(f"missing source artifact: {path}")
        actual = source_identity(path)
        if actual["bytes"] != contract["bytes"]:
            raise ValueError(f"source byte count mismatch: {path.name}")
        if actual["sha256"] != contract["sha256"]:
            raise ValueError(f"source SHA-256 mismatch: {path.name}")
        verified[name] = actual
    return verified


def decimal_stamp_ns(value: str) -> int:
    stamp = Decimal(value) * Decimal(1_000_000_000)
    return int(stamp.to_integral_value())


def read_ground_truth(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"num", "t", "x", "y", "z", "qx", "qy", "qz", "qw"}
    if not rows or set(rows[0]) != required:
        raise ValueError("unexpected pose_inW.csv schema")
    parsed = []
    for row in rows:
        pose = {
            "num": int(row["num"]),
            "stamp_ns": decimal_stamp_ns(row["t"]),
            **{key: float(row[key]) for key in required - {"num", "t"}},
        }
        if not all(
            math.isfinite(pose[key])
            for key in ("x", "y", "z", "qx", "qy", "qz", "qw")
        ):
            raise ValueError("non-finite ground-truth pose")
        parsed.append(pose)
    if any(
        right["num"] != left["num"] + 1
        or right["stamp_ns"] <= left["stamp_ns"]
        for left, right in zip(parsed, parsed[1:])
    ):
        raise ValueError("ground-truth rows must be contiguous and time ordered")
    return parsed


def read_os_sensor_extrinsic(path: Path) -> list[list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip() == "os_sensor:"
            and len(line) - len(line.lstrip()) == 4
        ),
        None,
    )
    if start is None:
        raise ValueError("body.os_sensor calibration is missing")
    rows: list[list[float]] = []
    for line in lines[start + 1 :]:
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent <= 4 and stripped:
            break
        if stripped.startswith("- [") and stripped.endswith("]"):
            row = [float(item.strip()) for item in stripped[3:-1].split(",")]
            rows.append(row)
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("body.os_sensor.T must be a 4x4 matrix")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("body.os_sensor.T contains non-finite values")
    if rows[3] != [0.0, 0.0, 0.0, 1.0]:
        raise ValueError("body.os_sensor.T has an invalid homogeneous row")
    return rows


def quaternion_to_matrix(
    qx: float, qy: float, qz: float, qw: float
) -> list[list[float]]:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid ground-truth quaternion")
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    return [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]


def matrix_to_quaternion(matrix: list[list[float]]) -> tuple[float, float, float, float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2][1] - matrix[1][2]) / scale
        qy = (matrix[0][2] - matrix[2][0]) / scale
        qz = (matrix[1][0] - matrix[0][1]) / scale
    else:
        axis = max(range(3), key=lambda index: matrix[index][index])
        if axis == 0:
            scale = math.sqrt(
                1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]
            ) * 2.0
            qw = (matrix[2][1] - matrix[1][2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0][1] + matrix[1][0]) / scale
            qz = (matrix[0][2] + matrix[2][0]) / scale
        elif axis == 1:
            scale = math.sqrt(
                1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]
            ) * 2.0
            qw = (matrix[0][2] - matrix[2][0]) / scale
            qx = (matrix[0][1] + matrix[1][0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1][2] + matrix[2][1]) / scale
        else:
            scale = math.sqrt(
                1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]
            ) * 2.0
            qw = (matrix[1][0] - matrix[0][1]) / scale
            qx = (matrix[0][2] + matrix[2][0]) / scale
            qy = (matrix[1][2] + matrix[2][1]) / scale
            qz = 0.25 * scale
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    result = (qx / norm, qy / norm, qz / norm, qw / norm)
    if result[3] < 0.0:
        result = tuple(-value for value in result)
    return result


def compose_sensor_pose(
    body_pose: dict[str, Any], body_t_sensor: list[list[float]]
) -> dict[str, float]:
    world_r_body = quaternion_to_matrix(
        body_pose["qx"],
        body_pose["qy"],
        body_pose["qz"],
        body_pose["qw"],
    )
    body_r_sensor = [row[:3] for row in body_t_sensor[:3]]
    world_r_sensor = [
        [
            sum(world_r_body[row][inner] * body_r_sensor[inner][column] for inner in range(3))
            for column in range(3)
        ]
        for row in range(3)
    ]
    body_p_sensor = [body_t_sensor[row][3] for row in range(3)]
    world_p_sensor = [
        body_pose[axis]
        + sum(world_r_body[row][inner] * body_p_sensor[inner] for inner in range(3))
        for row, axis in enumerate(("x", "y", "z"))
    ]
    qx, qy, qz, qw = matrix_to_quaternion(world_r_sensor)
    return {
        "x": world_p_sensor[0],
        "y": world_p_sensor[1],
        "z": world_p_sensor[2],
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qw": qw,
    }


def select_ground_truth(
    rows: list[dict[str, Any]],
    first_cloud_stamp_ns: int,
    start_offset_s: float,
    duration_s: float,
    cloud_count: int,
) -> list[dict[str, Any]]:
    target = first_cloud_stamp_ns + round(start_offset_s * 1e9)
    eligible = [row for row in rows if 0 <= row["num"] - 1 < cloud_count]
    if not eligible:
        raise ValueError("ground truth does not address any LiDAR frames")
    first = min(eligible, key=lambda row: abs(row["stamp_ns"] - target))
    if abs(first["stamp_ns"] - target) > MAXIMUM_PAIRING_ERROR_NS:
        raise ValueError("no ground-truth pose near the declared selection start")
    stop = first["stamp_ns"] + round(duration_s * 1e9)
    selected = [row for row in eligible if first["stamp_ns"] <= row["stamp_ns"] <= stop]
    if len(selected) < 2:
        raise ValueError("materialization window contains fewer than two frames")
    if any(
        right["num"] != left["num"] + 1
        for left, right in zip(selected, selected[1:])
    ):
        raise ValueError("materialization window is not contiguous")
    return selected


def cloud_stamp_ns(message: Any) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def ros2_pointcloud(ros1_message: Any, ros2_store: Any) -> Any:
    types = ros2_store.types
    time_type = types["builtin_interfaces/msg/Time"]
    header_type = types["std_msgs/msg/Header"]
    field_type = types["sensor_msgs/msg/PointField"]
    cloud_type = types[POINTCLOUD_TYPE]
    header = header_type(
        time_type(
            int(ros1_message.header.stamp.sec),
            int(ros1_message.header.stamp.nanosec),
        ),
        str(ros1_message.header.frame_id),
    )
    fields = [
        field_type(
            str(field.name),
            int(field.offset),
            int(field.datatype),
            int(field.count),
        )
        for field in ros1_message.fields
    ]
    return cloud_type(
        header,
        int(ros1_message.height),
        int(ros1_message.width),
        fields,
        bool(ros1_message.is_bigendian),
        int(ros1_message.point_step),
        int(ros1_message.row_step),
        ros1_message.data.copy(),
        bool(ros1_message.is_dense),
    )


def ros2_pose_stamped(
    stamp_ns: int, sensor_pose: dict[str, float], ros2_store: Any
) -> Any:
    types = ros2_store.types
    stamp = types["builtin_interfaces/msg/Time"](
        stamp_ns // 1_000_000_000,
        stamp_ns % 1_000_000_000,
    )
    header = types["std_msgs/msg/Header"](stamp, "world")
    point = types["geometry_msgs/msg/Point"](
        sensor_pose["x"], sensor_pose["y"], sensor_pose["z"]
    )
    quaternion = types["geometry_msgs/msg/Quaternion"](
        sensor_pose["qx"],
        sensor_pose["qy"],
        sensor_pose["qz"],
        sensor_pose["qw"],
    )
    pose = types["geometry_msgs/msg/Pose"](point, quaternion)
    return types[POSE_TYPE](header, pose)


def initialize_database(path: Path, cloud_topic: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        "PRAGMA journal_mode=OFF;"
        "PRAGMA synchronous=OFF;"
        "CREATE TABLE schema("
        "schema_version INTEGER PRIMARY KEY, ros_distro TEXT NOT NULL);"
        "CREATE TABLE metadata("
        "id INTEGER PRIMARY KEY, metadata_version INTEGER NOT NULL, "
        "metadata TEXT NOT NULL);"
        "CREATE TABLE topics("
        "id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, "
        "serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL);"
        "CREATE TABLE messages("
        "id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, "
        "timestamp INTEGER NOT NULL, data BLOB NOT NULL);"
        "CREATE INDEX timestamp_idx ON messages (timestamp ASC);"
    )
    connection.execute("INSERT INTO schema VALUES(3, 'jazzy')")
    connection.execute(
        "INSERT INTO topics VALUES(1, ?, ?, 'cdr', '')",
        (cloud_topic, POINTCLOUD_TYPE),
    )
    connection.execute(
        "INSERT INTO topics VALUES(2, ?, ?, 'cdr', '')",
        (POSE_TOPIC, POSE_TYPE),
    )
    connection.commit()
    return connection


def write_metadata(
    output: Path,
    database: Path,
    cloud_topic: str,
    start_ns: int,
    stop_ns: int,
    frames: int,
) -> None:
    duration_ns = stop_ns - start_ns
    metadata = (
        "rosbag2_bagfile_information:\n"
        "  version: 5\n"
        "  storage_identifier: sqlite3\n"
        "  duration:\n"
        f"    nanoseconds: {duration_ns}\n"
        "  starting_time:\n"
        f"    nanoseconds_since_epoch: {start_ns}\n"
        f"  message_count: {frames * 2}\n"
        "  topics_with_message_count:\n"
        "    - topic_metadata:\n"
        f"        name: {cloud_topic}\n"
        f"        type: {POINTCLOUD_TYPE}\n"
        "        serialization_format: cdr\n"
        '        offered_qos_profiles: ""\n'
        f"      message_count: {frames}\n"
        "    - topic_metadata:\n"
        f"        name: {POSE_TOPIC}\n"
        f"        type: {POSE_TYPE}\n"
        "        serialization_format: cdr\n"
        '        offered_qos_profiles: ""\n'
        f"      message_count: {frames}\n"
        '  compression_format: ""\n'
        '  compression_mode: ""\n'
        "  relative_file_paths:\n"
        f"    - {database.name}\n"
        "  files:\n"
        f"    - path: {database.name}\n"
        "      starting_time:\n"
        f"        nanoseconds_since_epoch: {start_ns}\n"
        "      duration:\n"
        f"        nanoseconds: {duration_ns}\n"
        f"      message_count: {frames * 2}\n"
    )
    (output / "metadata.yaml").write_text(metadata, encoding="utf-8")


def materialize(
    candidate_path: Path,
    source_dir: Path,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exception:
        raise RuntimeError(
            "install scripts/requirements-mcd-materialization.txt first"
        ) from exception

    if output.exists():
        raise ValueError(f"refusing existing output path: {output}")
    if report_path.exists():
        raise ValueError(f"refusing existing report path: {report_path}")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate.get("candidate_id") != "mcd_ntu_day_02_os1_128":
        raise ValueError("unexpected MCD candidate contract")
    sources = verify_sources(candidate, source_dir)
    cloud_contract = candidate["pointcloud"]
    materialization = candidate["materialization"]
    cloud_topic = cloud_contract["topic"]
    bag_path = source_dir / candidate["source_artifacts"]["ouster_rosbag"]["filename"]
    gt_path = source_dir / candidate["source_artifacts"]["discrete_ground_truth"]["filename"]
    calibration_path = source_dir / candidate["source_artifacts"]["calibration"]["filename"]

    rows = read_ground_truth(gt_path)
    extrinsic = read_os_sensor_extrinsic(calibration_path)
    ros1_store = get_typestore(Stores.ROS1_NOETIC)
    ros2_store = get_typestore(Stores.ROS2_JAZZY)
    started = time.monotonic()

    reader = Reader(bag_path)
    reader.open()
    try:
        cloud_connection = next(
            (
                connection
                for connection in reader.connections
                if connection.topic == cloud_topic
                and connection.msgtype == POINTCLOUD_TYPE
            ),
            None,
        )
        if cloud_connection is None:
            raise ValueError("contracted PointCloud2 topic is absent")
        indexes = reader.indexes[cloud_connection.id]
        first_entry = indexes[0]
        _, _, first_raw = next(
            reader.messages(
                connections=[cloud_connection],
                start=first_entry.time,
                stop=first_entry.time + 1,
            )
        )
        first_message = ros1_store.deserialize_ros1(first_raw, POINTCLOUD_TYPE)
        first_stamp = cloud_stamp_ns(first_message)
        selected = select_ground_truth(
            rows,
            first_stamp,
            float(materialization["selection_start_offset_s"]),
            float(materialization["selection_duration_s"]),
            len(indexes),
        )
        first_index = selected[0]["num"] - 1
        last_index = selected[-1]["num"] - 1
        start_recorded = indexes[first_index].time
        stop_recorded = indexes[last_index].time + 1

        output.mkdir(parents=True)
        database = output / "mcd_ntu_day_02_timed_0.db3"
        sql = initialize_database(database, cloud_topic)
        pairing_errors_ns: list[int] = []
        schema: list[dict[str, Any]] | None = None
        frame_id: str | None = None
        last_cloud_stamp: int | None = None
        processed = 0
        try:
            sql.execute("BEGIN")
            for offset, (_, _, raw) in enumerate(
                reader.messages(
                    connections=[cloud_connection],
                    start=start_recorded,
                    stop=stop_recorded,
                )
            ):
                gt = selected[offset]
                expected_index = gt["num"] - 1
                if expected_index != first_index + offset:
                    raise ValueError("ground-truth/LiDAR index mapping changed")
                ros1_cloud = ros1_store.deserialize_ros1(raw, POINTCLOUD_TYPE)
                stamp_ns = cloud_stamp_ns(ros1_cloud)
                if last_cloud_stamp is not None and stamp_ns <= last_cloud_stamp:
                    raise ValueError("LiDAR header timestamps are not increasing")
                last_cloud_stamp = stamp_ns
                pairing_error = gt["stamp_ns"] - stamp_ns
                if abs(pairing_error) > MAXIMUM_PAIRING_ERROR_NS:
                    raise ValueError(
                        f"GT/LiDAR pairing error exceeds 20 ms at index {expected_index}"
                    )
                pairing_errors_ns.append(pairing_error)

                current_schema = [
                    {
                        "name": str(field.name),
                        "offset": int(field.offset),
                        "datatype": int(field.datatype),
                        "count": int(field.count),
                    }
                    for field in ros1_cloud.fields
                ]
                if schema is None:
                    schema = current_schema
                    frame_id = str(ros1_cloud.header.frame_id)
                elif current_schema != schema or str(ros1_cloud.header.frame_id) != frame_id:
                    raise ValueError("PointCloud2 schema or frame changed")

                cloud = ros2_pointcloud(ros1_cloud, ros2_store)
                cloud_cdr = bytes(ros2_store.serialize_cdr(cloud, POINTCLOUD_TYPE))
                sensor_pose = compose_sensor_pose(gt, extrinsic)
                pose = ros2_pose_stamped(stamp_ns, sensor_pose, ros2_store)
                pose_cdr = bytes(ros2_store.serialize_cdr(pose, POSE_TYPE))
                message_id = 2 * offset + 1
                sql.execute(
                    "INSERT INTO messages VALUES(?, 1, ?, ?)",
                    (message_id, stamp_ns, cloud_cdr),
                )
                sql.execute(
                    "INSERT INTO messages VALUES(?, 2, ?, ?)",
                    (message_id + 1, stamp_ns, pose_cdr),
                )
                processed += 1
                if processed % 100 == 0:
                    elapsed = time.monotonic() - started
                    print(
                        f"materialized {processed}/{len(selected)} frames "
                        f"({elapsed:.1f} s)",
                        flush=True,
                    )
            if processed != len(selected):
                raise ValueError(
                    f"expected {len(selected)} LiDAR frames, read {processed}"
                )
            sql.commit()
        except Exception:
            sql.rollback()
            raise
        finally:
            sql.close()
    except Exception:
        reader.close()
        if output.exists():
            database = output / "mcd_ntu_day_02_timed_0.db3"
            if database.exists():
                database.unlink()
            metadata = output / "metadata.yaml"
            if metadata.exists():
                metadata.unlink()
            output.rmdir()
        raise
    reader.close()

    start_ns = selected[0]["stamp_ns"] - pairing_errors_ns[0]
    stop_ns = selected[-1]["stamp_ns"] - pairing_errors_ns[-1]
    write_metadata(output, database, cloud_topic, start_ns, stop_ns, processed)
    output_identity = source_identity(database)
    metadata_identity = source_identity(output / "metadata.yaml")
    elapsed = time.monotonic() - started
    report = {
        "schema_version": 1,
        "valid": True,
        "candidate": {
            "path": str(candidate_path.resolve()),
            "sha256": sha256_file(candidate_path),
            "candidate_id": candidate["candidate_id"],
        },
        "sources": sources,
        "selection": {
            "mapping": "pose_inW num N -> zero-based LiDAR index N-1",
            "first_ground_truth_num": selected[0]["num"],
            "last_ground_truth_num": selected[-1]["num"],
            "first_lidar_index": selected[0]["num"] - 1,
            "last_lidar_index": selected[-1]["num"] - 1,
            "frames": processed,
            "first_stamp_ns": start_ns,
            "last_stamp_ns": stop_ns,
            "duration_s": (stop_ns - start_ns) / 1e9,
        },
        "pairing_error_ms": {
            "minimum": min(pairing_errors_ns) / 1e6,
            "median": statistics.median(pairing_errors_ns) / 1e6,
            "maximum": max(pairing_errors_ns) / 1e6,
            "maximum_absolute": max(map(abs, pairing_errors_ns)) / 1e6,
        },
        "pointcloud": {
            "topic": cloud_topic,
            "type": POINTCLOUD_TYPE,
            "frame_id": frame_id,
            "schema": schema,
            "preserves_point_order_and_bytes": True,
        },
        "reference": {
            "topic": POSE_TOPIC,
            "type": POSE_TYPE,
            "frame_id": "world",
            "pose_semantics": "world_T_os_sensor",
            "composition": "world_T_body * body_T_os_sensor",
        },
        "outputs": {
            "database": output_identity,
            "metadata": metadata_identity,
        },
        "elapsed_s": elapsed,
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    argument_parser.add_argument("--source-dir", type=Path, required=True)
    argument_parser.add_argument("--output", type=Path, required=True)
    argument_parser.add_argument("--report", type=Path)
    return argument_parser


def main() -> int:
    arguments = parser().parse_args()
    report_path = arguments.report or arguments.output.with_name(
        arguments.output.name + "_materialization.json"
    )
    report = materialize(
        arguments.candidate.resolve(),
        arguments.source_dir.resolve(),
        arguments.output.resolve(),
        report_path.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
