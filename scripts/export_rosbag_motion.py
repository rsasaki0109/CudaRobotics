#!/usr/bin/env python3
"""Export Twist commands and Odometry trajectories from rosbag2 DB3 without ROS."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import struct
from pathlib import Path


class CdrReader:
    def __init__(self, data: bytes):
        if len(data) < 4 or data[:2] not in (b"\x00\x00", b"\x00\x01"):
            raise ValueError("unsupported CDR encapsulation")
        self.data = data
        self.little = data[1] == 1
        self.offset = 4

    def align(self, size: int) -> None:
        relative = self.offset - 4
        self.offset += (-relative) % size

    def unpack(self, code: str, size: int):
        self.align(size)
        value = struct.unpack_from(("<" if self.little else ">") + code, self.data, self.offset)[0]
        self.offset += size
        return value

    def int32(self) -> int:
        return self.unpack("i", 4)

    def uint32(self) -> int:
        return self.unpack("I", 4)

    def float64(self) -> float:
        return self.unpack("d", 8)

    def string(self) -> str:
        length = self.uint32()
        raw = self.data[self.offset:self.offset + length]
        self.offset += length
        return raw.rstrip(b"\x00").decode("utf-8", errors="replace")

    def doubles(self, count: int) -> list[float]:
        return [self.float64() for _ in range(count)]


def parse_twist(data: bytes) -> dict[str, float]:
    reader = CdrReader(data)
    values = reader.doubles(6)
    keys = ("linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z")
    return dict(zip(keys, values))


def parse_odometry(data: bytes) -> dict[str, object]:
    reader = CdrReader(data)
    stamp_sec, stamp_nanosec = reader.int32(), reader.uint32()
    frame_id, child_frame_id = reader.string(), reader.string()
    position = reader.doubles(3)
    quaternion = reader.doubles(4)
    reader.doubles(36)  # pose covariance
    twist = reader.doubles(6)
    reader.doubles(36)  # twist covariance
    qx, qy, qz, qw = quaternion
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return {
        "stamp_ns": stamp_sec * 1_000_000_000 + stamp_nanosec,
        "frame_id": frame_id, "child_frame_id": child_frame_id,
        "x": position[0], "y": position[1], "z": position[2], "yaw": yaw,
        "linear_x": twist[0], "linear_y": twist[1], "linear_z": twist[2],
        "angular_x": twist[3], "angular_y": twist[4], "angular_z": twist[5],
    }


def messages(connection: sqlite3.Connection, topic: str):
    row = connection.execute("SELECT id FROM topics WHERE name = ?", (topic,)).fetchone()
    if row is None:
        raise ValueError(f"topic not found: {topic}")
    yield from connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", (row[0],)
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def export_motion(
    db: Path,
    output_dir: Path,
    command_topic: str = "/mobile_base_controller/cmd_vel",
    odometry_topic: str = "/mobile_base_controller/odom",
) -> dict[str, object]:
    connection = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        commands = []
        for recorded_ns, payload in messages(connection, command_topic):
            commands.append({"recorded_ns": recorded_ns, **parse_twist(payload)})
        odometry = []
        for recorded_ns, payload in messages(connection, odometry_topic):
            odometry.append({"recorded_ns": recorded_ns, **parse_odometry(payload)})
    finally:
        connection.close()
    if not commands or not odometry:
        raise ValueError("command and odometry topics must both contain messages")
    write_rows(output_dir / "cmd_vel.csv", commands)
    write_rows(output_dir / "odometry.csv", odometry)

    distance = sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"])
                   for a, b in zip(odometry, odometry[1:]))
    displacement = math.hypot(odometry[-1]["x"] - odometry[0]["x"],
                              odometry[-1]["y"] - odometry[0]["y"])
    speeds = [math.hypot(row["linear_x"], row["linear_y"]) for row in odometry]
    command_speeds = [math.hypot(row["linear_x"], row["linear_y"]) for row in commands]
    summary = {
        "database": str(db), "command_topic": command_topic,
        "odometry_topic": odometry_topic,
        "command_samples": len(commands), "odometry_samples": len(odometry),
        "duration_s": (odometry[-1]["recorded_ns"] - odometry[0]["recorded_ns"]) / 1e9,
        "path_length_m": distance, "net_displacement_m": displacement,
        "mean_speed_mps": sum(speeds) / len(speeds), "max_speed_mps": max(speeds),
        "mean_command_speed_mps": sum(command_speeds) / len(command_speeds),
        "max_command_speed_mps": max(command_speeds),
        "max_abs_command_yaw_rate_rps": max(abs(row["angular_z"]) for row in commands),
        "stationary_command_ratio": sum(speed < 1e-3 and abs(row["angular_z"]) < 1e-3
                                        for speed, row in zip(command_speeds, commands)) / len(commands),
        "start_xy": [odometry[0]["x"], odometry[0]["y"]],
        "end_xy": [odometry[-1]["x"], odometry[-1]["y"]],
    }
    (output_dir / "motion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("build/rosbag_motion"))
    parser.add_argument(
        "--command-topic", default="/mobile_base_controller/cmd_vel"
    )
    parser.add_argument(
        "--odometry-topic", default="/mobile_base_controller/odom"
    )
    args = parser.parse_args()
    summary = export_motion(
        args.db.resolve(),
        args.output_dir,
        command_topic=args.command_topic,
        odometry_topic=args.odometry_topic,
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
