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

from analyze_pointcloud2_clearance import (
    parse_pointcloud2,
    point_field_values,
    xyz_points,
)
from cudanav_rosbag_evidence import sha256_file
from export_rosbag_motion import messages, pose_parser, topic_type


MAGIC = b"CRKICP1\x00"
VERSION = 1
TIMED_VERSION = 2
TIME_UNIT_SECONDS = {
    "seconds": 1.0,
    "milliseconds": 1e-3,
    "microseconds": 1e-6,
    "nanoseconds": 1e-9,
}


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


def numpy_timed_payload(
    cloud: dict[str, Any],
    point_time_field: str,
    point_time_scale: float,
    minimum_range_m: float,
    maximum_range_m: float,
) -> tuple[bytes, int, float] | None:
    try:
        import numpy as np
    except ImportError:
        return None

    formats = {
        1: "i1",
        2: "u1",
        3: "i2",
        4: "u2",
        5: "i4",
        6: "u4",
        7: "f4",
        8: "f8",
    }
    prefix = ">" if cloud["is_bigendian"] else "<"

    def values(name: str):
        field = cloud["fields"].get(name)
        if field is None or field["count"] != 1:
            raise ValueError(f"PointCloud2 requires scalar field: {name}")
        code = formats.get(field["datatype"])
        if code is None:
            raise ValueError(
                f"PointCloud2 field has unsupported datatype: {name}"
            )
        dtype = np.dtype(prefix + code)
        if field["offset"] + dtype.itemsize > cloud["point_step"]:
            raise ValueError(f"PointCloud2 field exceeds point_step: {name}")
        return np.ndarray(
            shape=(cloud["height"], cloud["width"]),
            dtype=dtype,
            buffer=cloud["data"],
            offset=field["offset"],
            strides=(cloud["row_step"], cloud["point_step"]),
        ).reshape(-1)

    x = values("x").astype(np.float64)
    y = values("y").astype(np.float64)
    z = values("z").astype(np.float64)
    point_times = values(point_time_field).astype(np.float64)
    point_times *= point_time_scale
    finite_times = np.isfinite(point_times)
    if int(finite_times.sum()) < 2:
        raise ValueError("selected frame has no valid point time span")
    first_point_time = float(point_times[finite_times].min())
    point_time_span_s = (
        float(point_times[finite_times].max()) - first_point_time
    )
    finite_xyz = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    distance = np.sqrt(x * x + y * y + z * z)
    selected = (
        finite_times
        & finite_xyz
        & (distance >= minimum_range_m)
        & (distance <= maximum_range_m)
    )
    count = int(selected.sum())
    packed = np.empty((count, 4), dtype="<f4")
    packed[:, 0] = x[selected]
    packed[:, 1] = y[selected]
    packed[:, 2] = z[selected]
    packed[:, 3] = point_times[selected] - first_point_time
    return packed.tobytes(order="C"), count, point_time_span_s


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
    point_time_field: str | None = None,
    point_time_unit: str = "seconds",
    require_point_time: bool = False,
    ring_field: str | None = None,
    require_ring: bool = False,
    numpy_acceleration: bool = True,
) -> dict[str, Any]:
    if start_offset_s < 0.0 or maximum_duration_s <= 0.0 or maximum_frames < 2:
        raise ValueError("duration and frame count limits must be positive")
    if (
        maximum_pose_age_ms <= 0.0
        or minimum_range_m < 0.0
        or maximum_range_m <= minimum_range_m
    ):
        raise ValueError("pose-age and range limits are invalid")
    if point_time_unit not in TIME_UNIT_SECONDS:
        raise ValueError("unsupported point time unit")
    if require_point_time and not point_time_field:
        raise ValueError("point_time_field is required by the timing contract")
    if require_ring and not ring_field:
        raise ValueError("ring_field is required by the ring contract")
    sequence_version = TIMED_VERSION if point_time_field else VERSION
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
        point_time_spans_s: list[float] = []
        point_field_schema: dict[str, dict[str, int]] | None = None
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as stream:
            stream.write(MAGIC)
            stream.write(struct.pack("<II", sequence_version, 0))
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
                if point_field_schema is None:
                    point_field_schema = cloud["fields"]
                elif cloud["fields"] != point_field_schema:
                    raise ValueError("PointCloud2 field schema changed inside sequence")
                if ring_field:
                    # Decode once to validate scalar datatype, stride, and presence.
                    next(point_field_values(cloud, (ring_field,)), None)
                xyz = []
                point_times_s: list[float] = []
                point_time_span_s: float | None = None
                timed_payload: bytes | None = None
                point_count = 0
                if point_time_field:
                    scale = TIME_UNIT_SECONDS[point_time_unit]
                    accelerated = (
                        numpy_timed_payload(
                            cloud,
                            point_time_field,
                            scale,
                            minimum_range_m,
                            maximum_range_m,
                        )
                        if numpy_acceleration
                        else None
                    )
                    if accelerated is not None:
                        timed_payload, point_count, point_time_span_s = accelerated
                    else:
                        records = point_field_values(
                            cloud, ("x", "y", "z", point_time_field)
                        )
                        selected = []
                        frame_point_times_s = []
                        for x, y, z, raw_time in records:
                            point_time_s = float(raw_time) * scale
                            if math.isfinite(point_time_s):
                                frame_point_times_s.append(point_time_s)
                            values = (float(x), float(y), float(z))
                            if not all(math.isfinite(value) for value in values):
                                continue
                            distance = math.sqrt(x * x + y * y + z * z)
                            if (
                                math.isfinite(point_time_s)
                                and minimum_range_m
                                <= distance
                                <= maximum_range_m
                            ):
                                selected.append(
                                    (
                                        float(x),
                                        float(y),
                                        float(z),
                                        point_time_s,
                                    )
                                )
                        if len(frame_point_times_s) >= 2:
                            first_point_time = min(frame_point_times_s)
                            point_time_span_s = (
                                max(frame_point_times_s) - first_point_time
                            )
                            for x, y, z, point_time in selected:
                                xyz.extend((x, y, z))
                                point_times_s.append(
                                    point_time - first_point_time
                                )
                            point_count = len(selected)
                    if (
                        point_time_span_s is not None
                        and not 1e-6 <= point_time_span_s <= 1.0
                    ):
                        raise ValueError(
                            "point time span must be in [1 us, 1 s]; "
                            "check point_time_unit"
                        )
                else:
                    for x, y, z in xyz_points(cloud):
                        distance = math.sqrt(x * x + y * y + z * z)
                        if minimum_range_m <= distance <= maximum_range_m:
                            xyz.extend((float(x), float(y), float(z)))
                    point_count = len(xyz) // 3
                if point_count < 10:
                    continue
                if point_time_field:
                    if point_time_span_s is None:
                        raise ValueError(
                            "selected frame has no valid point time span"
                        )
                    point_time_spans_s.append(point_time_span_s)
                if first_stamp is None:
                    first_stamp = stamp_ns
                normalized = normalized_reference(first_reference, reference)
                stream.write(
                    struct.pack(
                        "<QffffI",
                        stamp_ns,
                        *normalized,
                        point_count,
                    )
                )
                if sequence_version == TIMED_VERSION:
                    stream.write(struct.pack("<ff", 0.0, point_time_span_s))
                    if timed_payload is None:
                        timed_points = []
                        for point_index, point_time in enumerate(point_times_s):
                            timed_points.extend(
                                (
                                    xyz[point_index * 3],
                                    xyz[point_index * 3 + 1],
                                    xyz[point_index * 3 + 2],
                                    point_time,
                                )
                            )
                        timed_payload = struct.pack(
                            f"<{len(timed_points)}f", *timed_points
                        )
                    stream.write(timed_payload)
                else:
                    stream.write(struct.pack(f"<{len(xyz)}f", *xyz))
                pose_ages_ms.append(pose_age_ns / 1e6)
                point_counts.append(point_count)
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
    ordered_time_spans = sorted(point_time_spans_s)
    point_time_p95_s = (
        ordered_time_spans[
            min(
                len(ordered_time_spans) - 1,
                math.ceil(0.95 * len(ordered_time_spans)) - 1,
            )
        ]
        if ordered_time_spans
        else None
    )
    return {
        "schema_version": 1,
        "format": f"cudarobotics.kiss_icp_sequence.v{sequence_version}",
        "sequence_version": sequence_version,
        "database": {
            "filename": database.name,
            "bytes": database.stat().st_size,
            "sha256": sha256_file(database),
        },
        "pointcloud_topic": pointcloud_topic,
        "pose_topic": pose_topic,
        "pose_type": pose_type,
        "frame_id": frame_id,
        "point_fields": point_field_schema,
        "point_time": {
            "present": point_time_field is not None,
            "field": point_time_field,
            "unit": point_time_unit if point_time_field else None,
            "frames_with_valid_span": len(point_time_spans_s),
            "minimum_span_s": min(point_time_spans_s)
            if point_time_spans_s
            else None,
            "p95_span_s": point_time_p95_s,
        },
        "ring": {
            "present": ring_field is not None,
            "field": ring_field,
        },
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
    parser.add_argument("--point-time-field")
    parser.add_argument(
        "--point-time-unit",
        choices=tuple(TIME_UNIT_SECONDS),
        default="seconds",
    )
    parser.add_argument("--require-point-time", action="store_true")
    parser.add_argument("--ring-field")
    parser.add_argument("--require-ring", action="store_true")
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
        point_time_field=args.point_time_field,
        point_time_unit=args.point_time_unit,
        require_point_time=args.require_point_time,
        ring_field=args.ring_field,
        require_ring=args.require_ring,
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
