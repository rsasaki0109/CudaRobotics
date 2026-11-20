"""Recorded point-cloud sequences in the ``CRKICP1`` format.

``CRKICP1`` is written by ``scripts/export_cudanav_kiss_icp_sequence.py`` from
ROS 2 bags (no ROS install required) and read by the native
``tools/cudanav_kiss_icp_sequence.cu`` runner:

* header: ``b"CRKICP1\\0"``, little-endian ``uint32`` version (1 or 2) and
  ``uint32`` frame count;
* frame: ``<QffffI`` = stamp (ns), reference x, y, z, yaw relative to the first
  reference pose, point count; version 2 then stores ``<ff`` scan start/end
  and ``(x, y, z, t)`` float32 per point, version 1 stores ``(x, y, z)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import time
from typing import Any, BinaryIO, Iterable, Iterator

import numpy as np

from .evidence import SCHEMA_VERSION, SequenceGate, latency_summary, trajectory_metrics

MAGIC = b"CRKICP1\x00"
XYZ_VERSION = 1
TIMED_VERSION = 2
MIN_POINTS = 30
MAX_POINTS = 200000
_HEADER = struct.Struct("<II")
_FRAME = struct.Struct("<QffffI")
_SCAN_SPAN = struct.Struct("<ff")


class SequenceFormatError(ValueError):
    """The file is not a well-formed CRKICP1 sequence."""


@dataclass(frozen=True, eq=False)
class SequenceFrame:
    index: int
    stamp_ns: int
    reference: tuple[float, float, float, float]
    points: np.ndarray
    point_times: np.ndarray | None = None
    scan_start: float | None = None
    scan_end: float | None = None


@dataclass(frozen=True)
class SequenceHeader:
    version: int
    frame_count: int


def _read_exact(stream: BinaryIO, size: int, what: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise SequenceFormatError(f"truncated {what}")
    return data


def read_header(stream: BinaryIO) -> SequenceHeader:
    if _read_exact(stream, len(MAGIC), "sequence header") != MAGIC:
        raise SequenceFormatError("sequence magic mismatch")
    version, frame_count = _HEADER.unpack(_read_exact(stream, _HEADER.size, "sequence header"))
    if version not in (XYZ_VERSION, TIMED_VERSION) or frame_count < 2:
        raise SequenceFormatError(f"unsupported sequence header (version={version}, frames={frame_count})")
    return SequenceHeader(version=version, frame_count=frame_count)


def read_frame(stream: BinaryIO, version: int, index: int) -> SequenceFrame:
    stamp_ns, x, y, z, yaw, count = _FRAME.unpack(_read_exact(stream, _FRAME.size, "frame header"))
    if not MIN_POINTS <= count <= MAX_POINTS:
        raise SequenceFormatError(f"frame {index} has invalid point count {count}")
    if version == TIMED_VERSION:
        scan_start, scan_end = _SCAN_SPAN.unpack(_read_exact(stream, _SCAN_SPAN.size, "scan span"))
        payload = np.frombuffer(_read_exact(stream, count * 16, "point payload"), dtype="<f4")
        payload = payload.reshape(count, 4)
        points = np.ascontiguousarray(payload[:, :3], dtype=np.float32)
        times = np.ascontiguousarray(payload[:, 3], dtype=np.float32)
        return SequenceFrame(index, stamp_ns, (x, y, z, yaw), points, times, scan_start, scan_end)
    payload = np.frombuffer(_read_exact(stream, count * 12, "point payload"), dtype="<f4")
    points = np.ascontiguousarray(payload.reshape(count, 3), dtype=np.float32)
    return SequenceFrame(index, stamp_ns, (x, y, z, yaw), points)


def iter_sequence(path: str | Path, *, maximum_frames: int = 0) -> Iterator[SequenceFrame]:
    """Yield frames; reject truncated files and trailing bytes like the native reader."""
    with Path(path).open("rb") as stream:
        header = read_header(stream)
        limit = header.frame_count if maximum_frames <= 0 else min(header.frame_count, maximum_frames)
        for index in range(limit):
            yield read_frame(stream, header.version, index)
        if limit == header.frame_count and stream.read(1):
            raise SequenceFormatError("sequence has trailing bytes")


def sequence_header(path: str | Path) -> SequenceHeader:
    with Path(path).open("rb") as stream:
        return read_header(stream)


def write_sequence(path: str | Path, frames: Iterable[SequenceFrame], *, version: int = XYZ_VERSION) -> int:
    """Write frames as CRKICP1 (used for tests and custom exporters)."""
    if version not in (XYZ_VERSION, TIMED_VERSION):
        raise ValueError("version must be 1 or 2")
    count = 0
    with Path(path).open("wb") as stream:
        stream.write(MAGIC)
        stream.write(_HEADER.pack(version, 0))
        for frame in frames:
            points = np.ascontiguousarray(frame.points, dtype="<f4").reshape(-1, 3)
            if not MIN_POINTS <= points.shape[0] <= MAX_POINTS:
                raise ValueError(f"frame {count} has invalid point count {points.shape[0]}")
            stream.write(_FRAME.pack(int(frame.stamp_ns), *map(float, frame.reference), points.shape[0]))
            if version == TIMED_VERSION:
                if frame.point_times is None:
                    raise ValueError("version 2 frames require point_times")
                times = np.ascontiguousarray(frame.point_times, dtype="<f4").reshape(-1)
                if times.shape[0] != points.shape[0]:
                    raise ValueError("point_times must have one entry per point")
                start = 0.0 if frame.scan_start is None else float(frame.scan_start)
                end = float(times.max()) if frame.scan_end is None else float(frame.scan_end)
                stream.write(_SCAN_SPAN.pack(start, end))
                stream.write(np.column_stack([points, times]).astype("<f4").tobytes())
            else:
                stream.write(points.tobytes())
            count += 1
        stream.seek(len(MAGIC) + 4)
        stream.write(struct.pack("<I", count))
    if count < 2:
        raise ValueError("a sequence needs at least two frames")
    return count


SEQUENCE_COLUMNS = (
    "frame",
    "stamp_ns",
    "reference_x",
    "reference_y",
    "reference_z",
    "reference_yaw",
    "estimated_x",
    "estimated_y",
    "estimated_z",
    "estimated_yaw",
    "xy_error_m",
    "inliers",
    "rmse",
    "map_points",
    "observed_voxels",
    "frame_ms",
)


@dataclass(frozen=True, eq=False)
class SequenceResult:
    summary: dict[str, Any]
    rows: list[tuple[Any, ...]]
    last_step: Any
    stage_latency_ms: dict[str, dict[str, float]]


def run_sequence(
    path: str | Path,
    *,
    pipeline: Any = None,
    maximum_frames: int = 0,
    gate: SequenceGate = SequenceGate(),
) -> SequenceResult:
    """Run odometry, mapping, and ESDF over a recorded sequence (no planner)."""
    from .pipeline import CudaNavPipeline

    header = sequence_header(path)  # reject malformed input before allocating GPU state
    nav = CudaNavPipeline(enable_planner=False) if pipeline is None else pipeline
    rows: list[tuple[Any, ...]] = []
    estimated: list[tuple[float, float]] = []
    reference: list[tuple[float, float]] = []
    inliers: list[int] = []
    frame_ms: list[float] = []
    stage_ms: dict[str, list[float]] = {}
    deskewed_frames = 0
    last_step = None
    started = time.perf_counter()
    for frame in iter_sequence(path, maximum_frames=maximum_frames):
        frame_start = time.perf_counter()
        step = nav.step(
            frame.points,
            point_times=frame.point_times,
            scan_start=frame.scan_start,
            scan_end=frame.scan_end,
        )
        elapsed = (time.perf_counter() - frame_start) * 1000.0
        last_step = step
        odom = step.odometry
        deskewed_frames += int(odom.deskewed)
        if frame.index > 0:
            inliers.append(odom.inliers)
        estimated.append((odom.x, odom.y))
        reference.append((frame.reference[0], frame.reference[1]))
        frame_ms.append(elapsed)
        for name, value in step.timings_ms.items():
            stage_ms.setdefault(name, []).append(value)
        error = float(np.hypot(odom.x - frame.reference[0], odom.y - frame.reference[1]))
        rows.append(
            (
                frame.index,
                frame.stamp_ns,
                *frame.reference[:3],
                frame.reference[3],
                float(odom.translation[0]),
                float(odom.translation[1]),
                float(odom.translation[2]),
                odom.yaw,
                error,
                odom.inliers,
                odom.rmse,
                odom.map_points,
                int(step.mapping["observed_voxels"]),
                elapsed,
            )
        )
    if len(rows) < 2:
        raise SequenceFormatError("a sequence run needs at least two frames")
    metrics = trajectory_metrics(estimated, reference)
    minimum_inliers = min(inliers) if inliers else 0
    checks = {
        "ate_rmse": metrics["ate_rmse_m"] <= gate.maximum_ate_rmse_m,
        "final_drift": metrics["final_drift_percent"] <= gate.maximum_final_drift_percent,
        "odometry_inliers": minimum_inliers >= gate.minimum_inliers,
    }
    quality_pass = all(checks.values())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scenario": "cudanav_recorded_sequence",
        "stages": ["gpu_kiss_icp", "gpu_voxel_mapping", "gpu_esdf"],
        "runtime": "python",
        "sequence_version": header.version,
        "frames": len(rows),
        "deskewed_frames": deskewed_frames,
        "success": True,
        "smoke_pass": quality_pass,
        "quality_pass": quality_pass,
        "checks": checks,
        "gate": {
            "maximum_ate_rmse_m": gate.maximum_ate_rmse_m,
            "maximum_final_drift_percent": gate.maximum_final_drift_percent,
            "minimum_inliers": gate.minimum_inliers,
        },
        "odometry_ate_rmse_m": metrics["ate_rmse_m"],
        "odometry_position_error_m": metrics["final_error_m"],
        "odometry_drift_percent": metrics["final_drift_percent"],
        "reference_distance_m": metrics["reference_distance_m"],
        "estimated_distance_m": metrics["estimated_distance_m"],
        "minimum_inliers": minimum_inliers,
        # Recorded data: no controller is in the loop.
        "collision_count": None,
        "command_deadline_miss_rate": None,
        "wall_time_ms": (time.perf_counter() - started) * 1000.0,
        "claims": {
            "gpu_core_closed_loop": False,
            "ros2_runtime": False,
            "real_data": None,
        },
    }
    stage_latency = {name: latency_summary(values) for name, values in stage_ms.items()}
    stage_latency["frame"] = latency_summary(frame_ms)
    return SequenceResult(summary=summary, rows=rows, last_step=last_step, stage_latency_ms=stage_latency)
