#!/usr/bin/env python3
"""Derive a deterministic nav_msgs/Path sidecar from recorded ROS 2 odometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
import struct
from typing import Any

from cudanav_real_dataset import DEFAULT_SPEC, make_materialization, read_json
from export_rosbag_motion import messages, pose_parser


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def derive_path(
    rows: list[dict[str, Any]],
    minimum_translation_m: float,
    maximum_duration_s: float | None = None,
) -> list[dict[str, Any]]:
    if minimum_translation_m < 0.0 or not math.isfinite(minimum_translation_m):
        raise ValueError("minimum translation must be finite and non-negative")
    if not rows:
        raise ValueError("odometry topic has no messages")
    if maximum_duration_s is not None and (
        maximum_duration_s <= 0.0 or not math.isfinite(maximum_duration_s)
    ):
        raise ValueError("maximum duration must be finite and positive")
    ordered = sorted(enumerate(rows), key=lambda item: (item[1]["stamp_ns"], item[0]))
    unique: list[dict[str, Any]] = []
    seen_stamps: set[int] = set()
    for _, row in ordered:
        stamp = int(row["stamp_ns"])
        if stamp not in seen_stamps:
            unique.append(row)
            seen_stamps.add(stamp)
    if maximum_duration_s is not None:
        cutoff = unique[0]["stamp_ns"] + round(maximum_duration_s * 1e9)
        unique = [row for row in unique if row["stamp_ns"] <= cutoff]
    first = unique[0]
    c = math.cos(float(first["yaw"]))
    s = math.sin(float(first["yaw"]))

    normalized: list[dict[str, Any]] = []
    for row in unique:
        dx = float(row["x"]) - float(first["x"])
        dy = float(row["y"]) - float(first["y"])
        yaw = wrap_angle(float(row["yaw"]) - float(first["yaw"]))
        normalized.append(
            {
                "stamp_ns": int(row["stamp_ns"]),
                "x": c * dx + s * dy,
                "y": -s * dx + c * dy,
                "z": float(row["z"]) - float(first["z"]),
                "yaw": yaw,
                "qz": math.sin(0.5 * yaw),
                "qw": math.cos(0.5 * yaw),
            }
        )

    selected = [normalized[0]]
    for pose in normalized[1:-1]:
        previous = selected[-1]
        if math.hypot(pose["x"] - previous["x"], pose["y"] - previous["y"]) >= (
            minimum_translation_m
        ):
            selected.append(pose)
    if len(normalized) > 1 and normalized[-1]["stamp_ns"] != selected[-1]["stamp_ns"]:
        selected.append(normalized[-1])
    if len(selected) < 2:
        raise ValueError("derived path requires at least two distinct timestamps")
    return selected


def read_poses(
    database: Path, topic: str, message_type: str
) -> tuple[list[dict[str, Any]], int]:
    parser = pose_parser(message_type)
    connection = sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = []
        first_recorded_ns: int | None = None
        for recorded_ns, payload in messages(connection, topic):
            if first_recorded_ns is None:
                first_recorded_ns = int(recorded_ns)
            rows.append(parser(payload))
    finally:
        connection.close()
    if first_recorded_ns is None:
        raise ValueError(f"recorded pose topic has no messages: {topic}")
    return rows, first_recorded_ns


def write_rosbag(
    output: Path,
    topic: str,
    frame_id: str,
    poses: list[dict[str, Any]],
    recorded_ns: int,
) -> None:
    try:
        import rosbag2_py
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Path as PathMessage
        from rclpy.serialization import serialize_message
    except ImportError as exception:
        raise RuntimeError(
            "ROS 2 Jazzy Python environment is required to write the sidecar"
        ) from exception
    if output.exists():
        raise ValueError(f"refusing existing sidecar path: {output}")
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    metadata_fields = {
        "name": topic,
        "type": "nav_msgs/msg/Path",
        "serialization_format": "cdr",
    }
    try:
        topic_metadata = rosbag2_py.TopicMetadata(id=0, **metadata_fields)
    except TypeError:
        topic_metadata = rosbag2_py.TopicMetadata(**metadata_fields)
    writer.create_topic(topic_metadata)
    message = PathMessage()
    message.header.frame_id = frame_id
    message.header.stamp.sec = poses[0]["stamp_ns"] // 1_000_000_000
    message.header.stamp.nanosec = poses[0]["stamp_ns"] % 1_000_000_000
    for pose in poses:
        item = PoseStamped()
        item.header.frame_id = frame_id
        item.header.stamp.sec = pose["stamp_ns"] // 1_000_000_000
        item.header.stamp.nanosec = pose["stamp_ns"] % 1_000_000_000
        item.pose.position.x = pose["x"]
        item.pose.position.y = pose["y"]
        item.pose.position.z = pose["z"]
        item.pose.orientation.z = pose["qz"]
        item.pose.orientation.w = pose["qw"]
        message.poses.append(item)
    writer.write(topic, serialize_message(message), recorded_ns)
    del writer


class CdrWriter:
    def __init__(self) -> None:
        self.data = bytearray(b"\x00\x01\x00\x00")

    def add(self, code: str, size: int, value: object) -> None:
        relative = len(self.data) - 4
        self.data.extend(b"\x00" * ((-relative) % size))
        self.data.extend(struct.pack("<" + code, value))

    def int32(self, value: int) -> None:
        self.add("i", 4, value)

    def uint32(self, value: int) -> None:
        self.add("I", 4, value)

    def float64(self, value: float) -> None:
        self.add("d", 8, value)

    def string(self, value: str) -> None:
        raw = value.encode("utf-8") + b"\x00"
        self.uint32(len(raw))
        self.data.extend(raw)


def serialize_path(frame_id: str, poses: list[dict[str, Any]]) -> bytes:
    writer = CdrWriter()
    first_stamp = int(poses[0]["stamp_ns"])
    writer.int32(first_stamp // 1_000_000_000)
    writer.uint32(first_stamp % 1_000_000_000)
    writer.string(frame_id)
    writer.uint32(len(poses))
    for pose in poses:
        stamp = int(pose["stamp_ns"])
        writer.int32(stamp // 1_000_000_000)
        writer.uint32(stamp % 1_000_000_000)
        writer.string(frame_id)
        writer.float64(float(pose["x"]))
        writer.float64(float(pose["y"]))
        writer.float64(float(pose["z"]))
        writer.float64(0.0)
        writer.float64(0.0)
        writer.float64(float(pose["qz"]))
        writer.float64(float(pose["qw"]))
    return bytes(writer.data)


def write_sqlite_rosbag(
    output: Path,
    topic: str,
    frame_id: str,
    poses: list[dict[str, Any]],
    recorded_ns: int,
) -> None:
    if output.exists():
        raise ValueError(f"refusing existing sidecar path: {output}")
    output.mkdir(parents=True)
    database = output / "path_sidecar_0.db3"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            "CREATE TABLE schema("
            "schema_version INTEGER PRIMARY KEY, ros_distro TEXT NOT NULL);"
            "CREATE TABLE metadata("
            "id INTEGER PRIMARY KEY, metadata_version INTEGER NOT NULL, "
            "metadata TEXT NOT NULL);"
            "CREATE TABLE topics("
            "id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, "
            "serialization_format TEXT NOT NULL, "
            "offered_qos_profiles TEXT NOT NULL);"
            "CREATE TABLE messages("
            "id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, "
            "timestamp INTEGER NOT NULL, data BLOB NOT NULL);"
        )
        connection.execute("INSERT INTO schema VALUES(3, 'jazzy')")
        connection.execute(
            "INSERT INTO topics VALUES(1, ?, 'nav_msgs/msg/Path', 'cdr', '')",
            (topic,),
        )
        connection.execute(
            "INSERT INTO messages(topic_id, timestamp, data) VALUES(1, ?, ?)",
            (recorded_ns, serialize_path(frame_id, poses)),
        )
        connection.commit()
    finally:
        connection.close()
    metadata = (
        "rosbag2_bagfile_information:\n"
        "  version: 5\n"
        "  storage_identifier: sqlite3\n"
        "  duration:\n"
        "    nanoseconds: 0\n"
        "  starting_time:\n"
        f"    nanoseconds_since_epoch: {recorded_ns}\n"
        "  message_count: 1\n"
        "  topics_with_message_count:\n"
        "    - topic_metadata:\n"
        f"        name: {topic}\n"
        "        type: nav_msgs/msg/Path\n"
        "        serialization_format: cdr\n"
        '        offered_qos_profiles: ""\n'
        "      message_count: 1\n"
        '  compression_format: ""\n'
        '  compression_mode: ""\n'
        "  relative_file_paths:\n"
        f"    - {database.name}\n"
        "  files:\n"
        f"    - path: {database.name}\n"
        "      starting_time:\n"
        f"        nanoseconds_since_epoch: {recorded_ns}\n"
        "      duration:\n"
        "        nanoseconds: 0\n"
        "      message_count: 1\n"
    )
    (output / "metadata.yaml").write_text(metadata, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bag", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-bag", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--acquisition-report", type=Path, required=True)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--storage", choices=("mcap", "sqlite3"), default="mcap")
    args = parser.parse_args()
    spec = read_json(args.spec)
    contract = spec["path_derivation"]
    database = args.database.resolve()
    source_bag = args.source_bag.resolve()
    if not database.is_file() or not database.is_relative_to(source_bag):
        raise SystemExit("--database must be a file contained in --source-bag")
    rows, recorded_ns = read_poses(
        database,
        contract["source_topic"],
        contract.get("source_type", "nav_msgs/msg/Odometry"),
    )
    poses = derive_path(
        rows,
        contract["parameters"]["minimum_translation_m"],
        contract["parameters"].get("maximum_duration_s"),
    )
    writer = write_rosbag if args.storage == "mcap" else write_sqlite_rosbag
    writer(
        args.output_bag.resolve(),
        contract["output_topic"],
        "odom",
        poses,
        recorded_ns,
    )
    report = {
        "schema_version": 1,
        "algorithm": contract["algorithm"],
        "source_topic": contract["source_topic"],
        "source_type": contract["source_type"],
        "output_topic": contract["output_topic"],
        "parameters": contract["parameters"],
        "input_samples": len(rows),
        "output_poses": len(poses),
        "first_stamp_ns": poses[0]["stamp_ns"],
        "last_stamp_ns": poses[-1]["stamp_ns"],
        "frame_id": "odom",
        "recorded_path": False,
        "closed_loop": False,
        "storage_id": args.storage,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    materialization = make_materialization(
        args.spec,
        source_bag,
        args.output_bag.resolve(),
        args.report,
        args.acquisition_report,
    )
    args.materialization.parent.mkdir(parents=True, exist_ok=True)
    args.materialization.write_text(
        json.dumps(materialization, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
