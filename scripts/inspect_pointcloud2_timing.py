#!/usr/bin/env python3
"""Audit PointCloud2 per-point timing before release-grade GPU deskew."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
import statistics
from typing import Any

from analyze_pointcloud2_clearance import parse_pointcloud2, point_field_values
from cudanav_rosbag_evidence import sha256_file
from export_rosbag_motion import messages, topic_type


TIME_UNIT_SECONDS = {
    "seconds": 1.0,
    "milliseconds": 1e-3,
    "microseconds": 1e-6,
    "nanoseconds": 1e-9,
}
INTEGER_POINT_FIELD_TYPES = {1, 2, 3, 4, 5, 6}


def point_field_statistics(
    cloud: dict[str, Any],
    field_name: str,
    *,
    collect_unique: bool = False,
    numpy_acceleration: bool = True,
) -> dict[str, Any]:
    field = cloud["fields"].get(field_name)
    if not isinstance(field, dict) or field.get("count") != 1:
        raise ValueError(f"PointCloud2 requires scalar field: {field_name}")
    if numpy_acceleration:
        try:
            import numpy as np

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
            code = formats.get(field["datatype"])
            if code is None:
                raise ValueError(
                    f"PointCloud2 field has unsupported datatype: {field_name}"
                )
            prefix = ">" if cloud["is_bigendian"] else "<"
            values = np.ndarray(
                shape=(cloud["height"], cloud["width"]),
                dtype=np.dtype(prefix + code),
                buffer=cloud["data"],
                offset=field["offset"],
                strides=(cloud["row_step"], cloud["point_step"]),
            ).reshape(-1)
            finite = bool(np.isfinite(values).all())
            result = {
                "count": int(values.size),
                "finite": finite,
                "minimum": float(values.min()) if values.size else None,
                "maximum": float(values.max()) if values.size else None,
                "nondecreasing": bool(
                    values.size < 2 or np.all(values[:-1] <= values[1:])
                ),
                "nonincreasing": bool(
                    values.size < 2 or np.all(values[:-1] >= values[1:])
                ),
            }
            if collect_unique:
                result["unique"] = {int(value) for value in np.unique(values)}
            return result
        except ImportError:
            pass

    values = [
        item[0] for item in point_field_values(cloud, (field_name,))
    ]
    finite = all(math.isfinite(float(value)) for value in values)
    result = {
        "count": len(values),
        "finite": finite,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "nondecreasing": all(
            left <= right for left, right in zip(values, values[1:])
        ),
        "nonincreasing": all(
            left >= right for left, right in zip(values, values[1:])
        ),
    }
    if collect_unique:
        result["unique"] = {int(value) for value in values}
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(fraction * len(ordered)) - 1),
    )
    return ordered[index]


def timing_summary(spans: list[float]) -> dict[str, float]:
    return {
        "minimum_s": min(spans),
        "p05_s": percentile(spans, 0.05),
        "median_s": statistics.median(spans),
        "p95_s": percentile(spans, 0.95),
        "maximum_s": max(spans),
    }


def inspect_timing(
    database: Path,
    *,
    pointcloud_topic: str,
    point_time_field: str,
    point_time_datatype: int,
    point_time_unit: str,
    ring_field: str | None,
    ring_datatype: int | None,
    require_ring: bool,
    start_offset_s: float,
    maximum_duration_s: float,
    maximum_frames: int,
    minimum_frames: int,
    minimum_scan_span_s: float,
    maximum_scan_span_s: float,
    require_unambiguous_unit: bool,
) -> dict[str, Any]:
    if point_time_unit not in TIME_UNIT_SECONDS:
        raise ValueError("unsupported declared point-time unit")
    if not point_time_field:
        raise ValueError("point_time_field must be nonempty")
    if (
        not isinstance(point_time_datatype, int)
        or isinstance(point_time_datatype, bool)
        or point_time_datatype not in range(1, 9)
    ):
        raise ValueError("invalid declared point-time datatype")
    if require_ring and not ring_field:
        raise ValueError("ring_field is required by the admission contract")
    if ring_field and (
        not isinstance(ring_datatype, int)
        or isinstance(ring_datatype, bool)
        or ring_datatype not in INTEGER_POINT_FIELD_TYPES
    ):
        raise ValueError("ring_field requires a declared integer datatype")
    if (
        start_offset_s < 0.0
        or maximum_duration_s <= 0.0
        or maximum_frames < 2
        or minimum_frames < 2
        or minimum_frames > maximum_frames
    ):
        raise ValueError("invalid frame selection")
    if not (
        math.isfinite(minimum_scan_span_s)
        and math.isfinite(maximum_scan_span_s)
        and 0.0 < minimum_scan_span_s < maximum_scan_span_s <= 1.0
    ):
        raise ValueError("invalid physical scan-span bounds")

    source = database.resolve()
    connection = sqlite3.connect(
        f"file:{source.as_posix()}?mode=ro", uri=True
    )
    try:
        recorded_type = topic_type(connection, pointcloud_topic)
        if recorded_type != "sensor_msgs/msg/PointCloud2":
            raise ValueError("selected topic is not PointCloud2")

        schema: dict[str, dict[str, int]] | None = None
        frame_id: str | None = None
        first_source_stamp_ns: int | None = None
        selection_start_ns: int | None = None
        last_stamp_ns: int | None = None
        raw_spans: list[float] = []
        point_counts: list[int] = []
        ring_minimum: int | None = None
        ring_maximum: int | None = None
        ring_values_seen: set[int] = set()
        nondecreasing_frames = 0
        nonincreasing_frames = 0
        checks = {
            "topic_type": True,
            "stable_schema": True,
            "stable_frame_id": True,
            "strict_frame_timestamps": True,
            "scalar_time_field": True,
            "point_time_datatype": True,
            "finite_point_times": True,
            "nonzero_raw_time_span": True,
            "scalar_integer_ring": True,
            "ring_datatype": True,
            "minimum_frames": False,
            "declared_unit_plausible": False,
            "unit_unambiguous": False,
        }

        for _, payload in messages(connection, pointcloud_topic):
            cloud = parse_pointcloud2(payload)
            stamp_ns = int(cloud["stamp_ns"])
            if first_source_stamp_ns is None:
                first_source_stamp_ns = stamp_ns
                selection_start_ns = stamp_ns + round(start_offset_s * 1e9)
            assert selection_start_ns is not None
            if stamp_ns < selection_start_ns:
                continue
            if stamp_ns > selection_start_ns + round(maximum_duration_s * 1e9):
                break
            if len(raw_spans) >= maximum_frames:
                break
            if last_stamp_ns is not None and stamp_ns <= last_stamp_ns:
                checks["strict_frame_timestamps"] = False
            last_stamp_ns = stamp_ns

            if schema is None:
                schema = cloud["fields"]
                frame_id = str(cloud["frame_id"])
            else:
                checks["stable_schema"] &= cloud["fields"] == schema
                checks["stable_frame_id"] &= cloud["frame_id"] == frame_id

            time_field = cloud["fields"].get(point_time_field)
            checks["scalar_time_field"] &= (
                isinstance(time_field, dict)
                and time_field.get("count") == 1
            )
            checks["point_time_datatype"] &= (
                isinstance(time_field, dict)
                and time_field.get("datatype") == point_time_datatype
            )
            time_statistics = point_field_statistics(
                cloud, point_time_field
            )
            point_counts.append(time_statistics["count"])
            finite = time_statistics["finite"]
            checks["finite_point_times"] &= finite
            if time_statistics["count"] < 2 or not finite:
                checks["nonzero_raw_time_span"] = False
                raw_spans.append(0.0)
            else:
                raw_span = (
                    time_statistics["maximum"]
                    - time_statistics["minimum"]
                )
                checks["nonzero_raw_time_span"] &= raw_span > 0.0
                raw_spans.append(raw_span)
                nondecreasing_frames += int(
                    time_statistics["nondecreasing"]
                )
                nonincreasing_frames += int(
                    time_statistics["nonincreasing"]
                )

            if ring_field:
                ring_schema = cloud["fields"].get(ring_field)
                checks["scalar_integer_ring"] &= (
                    isinstance(ring_schema, dict)
                    and ring_schema.get("count") == 1
                    and ring_schema.get("datatype")
                    in INTEGER_POINT_FIELD_TYPES
                )
                checks["ring_datatype"] &= (
                    isinstance(ring_schema, dict)
                    and ring_schema.get("datatype") == ring_datatype
                )
                ring_statistics = point_field_statistics(
                    cloud, ring_field, collect_unique=True
                )
                if ring_statistics["count"]:
                    current_min = int(ring_statistics["minimum"])
                    current_max = int(ring_statistics["maximum"])
                    ring_minimum = (
                        current_min
                        if ring_minimum is None
                        else min(ring_minimum, current_min)
                    )
                    ring_maximum = (
                        current_max
                        if ring_maximum is None
                        else max(ring_maximum, current_max)
                    )
                    ring_values_seen.update(ring_statistics["unique"])
            elif require_ring:
                checks["scalar_integer_ring"] = False
    finally:
        connection.close()

    if not raw_spans:
        raise ValueError("selection contains no PointCloud2 frames")

    checks["minimum_frames"] = len(raw_spans) >= minimum_frames
    candidates: dict[str, dict[str, Any]] = {}
    plausible_units = []
    for unit, scale in TIME_UNIT_SECONDS.items():
        scaled = [span * scale for span in raw_spans]
        plausible = all(
            minimum_scan_span_s <= span <= maximum_scan_span_s
            for span in scaled
        )
        if plausible:
            plausible_units.append(unit)
        candidates[unit] = {
            "scale_to_seconds": scale,
            "plausible": plausible,
            "span": timing_summary(scaled),
        }
    checks["declared_unit_plausible"] = candidates[point_time_unit][
        "plausible"
    ]
    checks["unit_unambiguous"] = (
        not require_unambiguous_unit
        or plausible_units == [point_time_unit]
    )
    if not require_ring and ring_field is None:
        checks["scalar_integer_ring"] = True
        checks["ring_datatype"] = True

    valid = all(checks.values())
    return {
        "schema_version": 1,
        "algorithm": "cudarobotics.pointcloud2_timing_admission.v1",
        "database": {
            "filename": source.name,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "pointcloud_topic": pointcloud_topic,
        "pointcloud_type": recorded_type,
        "frame_id": frame_id,
        "point_fields": schema,
        "selection": {
            "start_offset_s": start_offset_s,
            "maximum_duration_s": maximum_duration_s,
            "maximum_frames": maximum_frames,
            "minimum_frames": minimum_frames,
            "frames": len(raw_spans),
            "first_source_stamp_ns": first_source_stamp_ns,
            "last_selected_stamp_ns": last_stamp_ns,
            "minimum_points": min(point_counts),
            "maximum_points": max(point_counts),
        },
        "point_time": {
            "field": point_time_field,
            "declared_datatype": point_time_datatype,
            "datatype": schema[point_time_field]["datatype"]
            if schema and point_time_field in schema
            else None,
            "declared_unit": point_time_unit,
            "physical_span_bounds_s": {
                "minimum": minimum_scan_span_s,
                "maximum": maximum_scan_span_s,
            },
            "raw_span": {
                "minimum": min(raw_spans),
                "maximum": max(raw_spans),
            },
            "candidate_units": candidates,
            "plausible_units": plausible_units,
            "require_unambiguous_unit": require_unambiguous_unit,
            "nondecreasing_frames": nondecreasing_frames,
            "nonincreasing_frames": nonincreasing_frames,
        },
        "ring": {
            "required": require_ring,
            "present": ring_field is not None,
            "field": ring_field,
            "declared_datatype": ring_datatype,
            "minimum": ring_minimum,
            "maximum": ring_maximum,
            "distinct_values": len(ring_values_seen),
        },
        "checks": checks,
        "valid": valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--pointcloud-topic", required=True)
    parser.add_argument("--point-time-field", required=True)
    parser.add_argument("--point-time-datatype", type=int, required=True)
    parser.add_argument(
        "--point-time-unit",
        choices=tuple(TIME_UNIT_SECONDS),
        required=True,
    )
    parser.add_argument("--ring-field")
    parser.add_argument("--ring-datatype", type=int)
    parser.add_argument("--require-ring", action="store_true")
    parser.add_argument("--start-offset-s", type=float, default=0.0)
    parser.add_argument("--maximum-duration-s", type=float, default=120.0)
    parser.add_argument("--maximum-frames", type=int, default=1200)
    parser.add_argument("--minimum-frames", type=int, default=2)
    parser.add_argument("--minimum-scan-span-s", type=float, required=True)
    parser.add_argument("--maximum-scan-span-s", type=float, required=True)
    parser.add_argument("--require-unambiguous-unit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_timing(
        args.database,
        pointcloud_topic=args.pointcloud_topic,
        point_time_field=args.point_time_field,
        point_time_datatype=args.point_time_datatype,
        point_time_unit=args.point_time_unit,
        ring_field=args.ring_field,
        ring_datatype=args.ring_datatype,
        require_ring=args.require_ring,
        start_offset_s=args.start_offset_s,
        maximum_duration_s=args.maximum_duration_s,
        maximum_frames=args.maximum_frames,
        minimum_frames=args.minimum_frames,
        minimum_scan_span_s=args.minimum_scan_span_s,
        maximum_scan_span_s=args.maximum_scan_span_s,
        require_unambiguous_unit=args.require_unambiguous_unit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
