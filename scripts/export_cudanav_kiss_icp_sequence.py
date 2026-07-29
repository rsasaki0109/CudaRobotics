#!/usr/bin/env python3
"""Export real rosbag PointCloud2 + reference poses for GPU KISS-ICP."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
import sqlite3
import statistics
import struct
from typing import Any

from analyze_pointcloud2_clearance import parse_pointcloud2, xyz_points
from cudanav_rosbag_evidence import sha256_file
from export_rosbag_motion import messages, pose_parser, topic_type


MAGIC = b"CRKICP1\x00"
VERSION = 1


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def reference_poses(
    connection: sqlite3.Connection,
    topic: str,
    message_type: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    parser = pose_parser(message_type)
    rows = [parser(payload) for _, payload in messages(connection, topic)]
    if len(rows) < 2:
        raise ValueError("reference pose topic has fewer than two messages")
    rows.sort(key=lambda row: int(row["stamp_ns"]))
    unique = []
    for row in rows:
        if not unique or row["stamp_ns"] != unique[-1]["stamp_ns"]:
            unique.append(row)
    return [int(row["stamp_ns"]) for row in unique], unique


def normalized_reference(
    first: dict[str, Any],
    current: dict[str, Any],
) -> tuple[float, float, float, float]:
    dx = float(current["x"]) - float(first["x"])
    dy = float(current["y"]) - float(first["y"])
    yaw0 = float(first["yaw"])
    cosine = math.cos(yaw0)
    sine = math.sin(yaw0)
    return (
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        float(current["z"]) - float(first["z"]),
        wrap_angle(float(current["yaw"]) - yaw0),
    )


def nearest_pose(
    stamps: list[int],
    rows: list[dict[str, Any]],
    stamp_ns: int,
) -> tuple[dict[str, Any], int]:
    index = bisect.bisect_left(stamps, stamp_ns)
    candidates = [
        candidate
        for candidate in (index - 1, index)
        if 0 <= candidate < len(stamps)
    ]
    selected = min(candidates, key=lambda candidate: abs(stamps[candidate] - stamp_ns))
    return rows[selected], abs(stamps[selected] - stamp_ns)


def export_sequence(
    database: Path,
    output: Path,
    *,
    pointcloud_topic: str,
    pose_topic: str,
    pose_type: str,
    start_offset_s: float,
    maximum_duration_s: float,
    maximum_frames: int,
    maximum_pose_age_ms: float,
    minimum_range_m: float,
    maximum_range_m: float,
) -> dict[str, Any]:
    if start_offset_s < 0.0 or maximum_duration_s <= 0.0 or maximum_frames < 2:
        raise ValueError("duration and frame count limits must be positive")
    if (
        maximum_pose_age_ms <= 0.0
        or minimum_range_m < 0.0
        or maximum_range_m <= minimum_range_m
    ):
        raise ValueError("pose-age and range limits are invalid")
    connection = sqlite3.connect(
        f"file:{database.resolve().as_posix()}?mode=ro", uri=True
    )
    try:
        if topic_type(connection, pointcloud_topic) != "sensor_msgs/msg/PointCloud2":
            raise ValueError("pointcloud topic type mismatch")
        if topic_type(connection, pose_topic) != pose_type:
            raise ValueError("reference pose topic type mismatch")
        pose_stamps, poses = reference_poses(connection, pose_topic, pose_type)
        frame_count = 0
        window_start_stamp: int | None = None
        first_stamp: int | None = None
        last_stamp: int | None = None
        first_reference: dict[str, Any] | None = None
        frame_id: str | None = None
        pose_ages_ms: list[float] = []
        point_counts: list[int] = []
        reference_xy: list[tuple[float, float]] = []
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as stream:
            stream.write(MAGIC)
            stream.write(struct.pack("<II", VERSION, 0))
            for _, payload in messages(connection, pointcloud_topic):
                cloud = parse_pointcloud2(payload)
                stamp_ns = int(cloud["stamp_ns"])
                if window_start_stamp is None:
                    window_start_stamp = stamp_ns
                elapsed_ns = stamp_ns - window_start_stamp
                if elapsed_ns < round(start_offset_s * 1e9):
                    continue
                if elapsed_ns > round(
                    (start_offset_s + maximum_duration_s) * 1e9
                ):
                    break
                reference, pose_age_ns = nearest_pose(
                    pose_stamps, poses, stamp_ns
                )
                if pose_age_ns > round(maximum_pose_age_ms * 1e6):
                    continue
                if first_reference is None:
                    first_reference = reference
                if frame_id is None:
                    frame_id = str(cloud["frame_id"])
                elif cloud["frame_id"] != frame_id:
                    raise ValueError("PointCloud2 frame_id changed inside sequence")
                xyz = []
                for x, y, z in xyz_points(cloud):
                    distance = math.sqrt(x * x + y * y + z * z)
                    if minimum_range_m <= distance <= maximum_range_m:
                        xyz.extend((float(x), float(y), float(z)))
                if len(xyz) < 30:
                    continue
                if first_stamp is None:
                    first_stamp = stamp_ns
                normalized = normalized_reference(first_reference, reference)
                stream.write(
                    struct.pack(
                        "<QffffI",
                        stamp_ns,
                        *normalized,
                        len(xyz) // 3,
                    )
                )
                stream.write(struct.pack(f"<{len(xyz)}f", *xyz))
                pose_ages_ms.append(pose_age_ns / 1e6)
                point_counts.append(len(xyz) // 3)
                reference_xy.append((normalized[0], normalized[1]))
                frame_count += 1
                last_stamp = stamp_ns
                if frame_count >= maximum_frames:
                    break
            stream.seek(len(MAGIC) + 4)
            stream.write(struct.pack("<I", frame_count))
    finally:
        connection.close()
    if frame_count < 2 or first_stamp is None or last_stamp is None:
        raise ValueError("fewer than two usable PointCloud2 frames")
    reference_distance = sum(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(reference_xy, reference_xy[1:])
    )
    ordered_ages = sorted(pose_ages_ms)
    p95_index = min(
        len(ordered_ages) - 1,
        math.ceil(0.95 * len(ordered_ages)) - 1,
    )
    return {
        "schema_version": 1,
        "format": "cudarobotics.kiss_icp_sequence.v1",
        "database": {
            "filename": database.name,
            "bytes": database.stat().st_size,
            "sha256": sha256_file(database),
        },
        "pointcloud_topic": pointcloud_topic,
        "pose_topic": pose_topic,
        "pose_type": pose_type,
        "frame_id": frame_id,
        "frames": frame_count,
        "first_stamp_ns": first_stamp,
        "last_stamp_ns": last_stamp,
        "duration_s": (last_stamp - first_stamp) / 1e9,
        "maximum_duration_s": maximum_duration_s,
        "start_offset_s": start_offset_s,
        "maximum_pose_age_ms": maximum_pose_age_ms,
        "pose_age_p95_ms": ordered_ages[p95_index],
        "minimum_points": min(point_counts),
        "maximum_points": max(point_counts),
        "mean_points": statistics.fmean(point_counts),
        "reference_path_length_m": reference_distance,
        "sequence": {
            "filename": output.name,
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--pointcloud-topic", required=True)
    parser.add_argument("--pose-topic", required=True)
    parser.add_argument("--pose-type", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--maximum-duration-s", type=float, default=30.0)
    parser.add_argument("--start-offset-s", type=float, default=1.0)
    parser.add_argument("--maximum-frames", type=int, default=300)
    parser.add_argument("--maximum-pose-age-ms", type=float, default=50.0)
    parser.add_argument("--minimum-range-m", type=float, default=0.5)
    parser.add_argument("--maximum-range-m", type=float, default=80.0)
    args = parser.parse_args()
    report = export_sequence(
        args.database,
        args.output,
        pointcloud_topic=args.pointcloud_topic,
        pose_topic=args.pose_topic,
        pose_type=args.pose_type,
        start_offset_s=args.start_offset_s,
        maximum_duration_s=args.maximum_duration_s,
        maximum_frames=args.maximum_frames,
        maximum_pose_age_ms=args.maximum_pose_age_ms,
        minimum_range_m=args.minimum_range_m,
        maximum_range_m=args.maximum_range_m,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
