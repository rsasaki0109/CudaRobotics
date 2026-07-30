#!/usr/bin/env python3
"""Measure front clearance from PointCloud2 in a rosbag2 DB3 without ROS."""

from __future__ import annotations

import bisect
import csv
import math
from pathlib import Path
import sqlite3
import struct
from typing import Any

from export_rosbag_motion import CdrReader, messages
import render_cuda_mppi_diagnostics


FLOAT32 = 7
FLOAT64 = 8
POINT_FIELD_FORMATS = {
    1: ("b", 1),
    2: ("B", 1),
    3: ("h", 2),
    4: ("H", 2),
    5: ("i", 4),
    6: ("I", 4),
    FLOAT32: ("f", 4),
    FLOAT64: ("d", 8),
}


def uint8(reader: CdrReader) -> int:
    if reader.offset >= len(reader.data):
        raise ValueError("truncated CDR uint8")
    value = reader.data[reader.offset]
    reader.offset += 1
    return value


def parse_pointcloud2(data: bytes) -> dict[str, Any]:
    reader = CdrReader(data)
    stamp_sec, stamp_nanosec = reader.int32(), reader.uint32()
    frame_id = reader.string()
    height, width = reader.uint32(), reader.uint32()
    field_count = reader.uint32()
    fields: dict[str, dict[str, int]] = {}
    for _ in range(field_count):
        name = reader.string()
        offset = reader.uint32()
        datatype = uint8(reader)
        count = reader.uint32()
        fields[name] = {"offset": offset, "datatype": datatype, "count": count}
    is_bigendian = bool(uint8(reader))
    point_step, row_step = reader.uint32(), reader.uint32()
    byte_count = reader.uint32()
    payload = reader.data[reader.offset : reader.offset + byte_count]
    reader.offset += byte_count
    is_dense = bool(uint8(reader))
    if point_step <= 0 or row_step < width * point_step:
        raise ValueError("invalid PointCloud2 stride")
    if len(payload) < height * row_step:
        raise ValueError("PointCloud2 data is shorter than declared geometry")
    for axis in ("x", "y", "z"):
        field = fields.get(axis)
        if (
            field is None
            or field["count"] < 1
            or field["datatype"] not in (FLOAT32, FLOAT64)
        ):
            raise ValueError(f"PointCloud2 requires float x/y/z field: {axis}")
        size = 4 if field["datatype"] == FLOAT32 else 8
        if field["offset"] + size > point_step:
            raise ValueError(f"PointCloud2 field exceeds point_step: {axis}")
    return {
        "stamp_ns": stamp_sec * 1_000_000_000 + stamp_nanosec,
        "frame_id": frame_id,
        "height": height,
        "width": width,
        "fields": fields,
        "is_bigendian": is_bigendian,
        "point_step": point_step,
        "row_step": row_step,
        "data": payload,
        "is_dense": is_dense,
    }


def point_field_values(
    cloud: dict[str, Any],
    field_names: tuple[str, ...],
):
    endian = ">" if cloud["is_bigendian"] else "<"
    layouts = []
    for name in field_names:
        field = cloud["fields"].get(name)
        if field is None or field["count"] != 1:
            raise ValueError(f"PointCloud2 requires scalar field: {name}")
        layout = POINT_FIELD_FORMATS.get(field["datatype"])
        if layout is None:
            raise ValueError(f"PointCloud2 field has unsupported datatype: {name}")
        fmt, size = layout
        if field["offset"] + size > cloud["point_step"]:
            raise ValueError(f"PointCloud2 field exceeds point_step: {name}")
        layouts.append((field["offset"], fmt))
    for row in range(cloud["height"]):
        for column in range(cloud["width"]):
            base = row * cloud["row_step"] + column * cloud["point_step"]
            yield tuple(
                struct.unpack_from(endian + fmt, cloud["data"], base + offset)[0]
                for offset, fmt in layouts
            )


def xyz_points(cloud: dict[str, Any]):
    for values in point_field_values(cloud, ("x", "y", "z")):
        if all(math.isfinite(value) for value in values):
            yield values


def front_clearance(
    cloud: dict[str, Any],
    *,
    half_angle_rad: float,
    minimum_z_m: float,
    maximum_z_m: float,
    minimum_range_m: float,
    maximum_range_m: float,
) -> tuple[float | None, int, int]:
    minimum: float | None = None
    finite_points = 0
    selected_points = 0
    for x, y, z in xyz_points(cloud):
        finite_points += 1
        distance = math.hypot(x, y)
        if (
            minimum_z_m <= z <= maximum_z_m
            and minimum_range_m <= distance <= maximum_range_m
            and x > 0.0
            and abs(math.atan2(y, x)) <= half_angle_rad
        ):
            selected_points += 1
            minimum = distance if minimum is None else min(minimum, distance)
    return minimum, finite_points, selected_points


def diagnostics_commands(path: Path) -> list[dict[str, float]]:
    result = []
    for row in render_cuda_mppi_diagnostics.load_rows(path):
        stamp = render_cuda_mppi_diagnostics.as_float(row, "stamp_sec")
        command = {
            "stamp_ns": int(round(stamp * 1e9)),
            "cmd_v": render_cuda_mppi_diagnostics.as_float(row, "cmd_v"),
            "cmd_vy": render_cuda_mppi_diagnostics.as_float(row, "cmd_vy"),
            "cmd_w": render_cuda_mppi_diagnostics.as_float(row, "cmd_w"),
        }
        if all(math.isfinite(value) for value in command.values()):
            result.append(command)
    if not result:
        raise ValueError("diagnostics contain no finite timestamped commands")
    return sorted(result, key=lambda row: row["stamp_ns"])


def analyze(
    db: Path,
    output_csv: Path,
    diagnostics_csv: Path,
    *,
    pointcloud_topic: str,
    half_angle_rad: float = math.pi / 6.0,
    minimum_z_m: float = -0.5,
    maximum_z_m: float = 2.5,
    minimum_range_m: float = 0.05,
    maximum_range_m: float = 50.0,
    maximum_command_age_ms: float = 200.0,
    timestamp_basis: str = "pointcloud_header_stamp",
) -> dict[str, Any]:
    if timestamp_basis not in {
        "pointcloud_header_stamp",
        "bag_record_timestamp",
    }:
        raise ValueError(f"unsupported timestamp basis: {timestamp_basis}")
    commands = diagnostics_commands(diagnostics_csv)
    command_times = [row["stamp_ns"] for row in commands]
    connection = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = []
        for recorded_ns, payload in messages(connection, pointcloud_topic):
            cloud = parse_pointcloud2(payload)
            pairing_stamp_ns = (
                recorded_ns
                if timestamp_basis == "bag_record_timestamp"
                else cloud["stamp_ns"]
            )
            clearance, finite_count, selected_count = front_clearance(
                cloud,
                half_angle_rad=half_angle_rad,
                minimum_z_m=minimum_z_m,
                maximum_z_m=maximum_z_m,
                minimum_range_m=minimum_range_m,
                maximum_range_m=maximum_range_m,
            )
            index = min(
                bisect.bisect_left(command_times, pairing_stamp_ns),
                len(commands) - 1,
            )
            if index and abs(command_times[index - 1] - pairing_stamp_ns) < abs(
                command_times[index] - pairing_stamp_ns
            ):
                index -= 1
            command = commands[index]
            rows.append(
                {
                    "recorded_ns": recorded_ns,
                    "stamp_ns": cloud["stamp_ns"],
                    "pairing_stamp_ns": pairing_stamp_ns,
                    "frame_id": cloud["frame_id"],
                    "finite_points": finite_count,
                    "front_points": selected_count,
                    "front_min_range_m": clearance,
                    "command_age_ms": abs(
                        command["stamp_ns"] - pairing_stamp_ns
                    )
                    / 1e6,
                    "command_speed_mps": math.hypot(
                        command["cmd_v"], command["cmd_vy"]
                    ),
                    "command_yaw_rate_rps": command["cmd_w"],
                }
            )
    finally:
        connection.close()
    if not rows:
        raise ValueError("PointCloud2 topic contains no messages")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row["front_min_range_m"] is not None]
    if not valid:
        raise ValueError("no PointCloud2 samples contain a valid front point")
    evaluation_window = [
        row
        for row in valid
        if command_times[0] <= row["pairing_stamp_ns"] <= command_times[-1]
    ]
    if not evaluation_window:
        raise ValueError("no PointCloud2 samples overlap the diagnostics window")
    paired = [
        row
        for row in evaluation_window
        if row["command_age_ms"] <= maximum_command_age_ms
    ]
    clearances = [
        float(row["front_min_range_m"]) for row in evaluation_window
    ]
    return {
        "database": str(db.resolve()),
        "pointcloud_topic": pointcloud_topic,
        "diagnostics_source": str(diagnostics_csv.resolve()),
        "timestamp_basis": timestamp_basis,
        "cloud_samples": len(rows),
        "valid_front_samples": len(valid),
        "evaluation_window_samples": len(evaluation_window),
        "paired_command_samples": len(paired),
        "command_pair_ratio": len(paired) / len(evaluation_window),
        "mean_front_clearance_m": sum(clearances) / len(clearances),
        "minimum_front_range_m": min(clearances),
        "front_below_0_5m_ratio": sum(value < 0.5 for value in clearances)
        / len(clearances),
        "front_below_1_0m_ratio": sum(value < 1.0 for value in clearances)
        / len(clearances),
        "mean_paired_command_age_ms": (
            sum(float(row["command_age_ms"]) for row in paired) / len(paired)
            if paired
            else None
        ),
        "filter": {
            "half_angle_rad": half_angle_rad,
            "minimum_z_m": minimum_z_m,
            "maximum_z_m": maximum_z_m,
            "minimum_range_m": minimum_range_m,
            "maximum_range_m": maximum_range_m,
            "maximum_command_age_ms": maximum_command_age_ms,
        },
    }
