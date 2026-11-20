"""Thin, typed wrappers over the CudaNav CUDA cores.

Each class owns exactly one tested C++ core from ``include/cudarobotics``:

* :class:`KissIcpOdometry` -> ``cudarobotics::KissIcpOdometry``
* :class:`VoxelMap` -> ``cudarobotics::VoxelMapperGpu``
* :class:`Esdf2D` -> ``cudarobotics::Esdf2DGpu``

The helpers in this module (:func:`clear_footprint`, :func:`costmap_from_esdf`,
:func:`transform_points`) are NumPy ports of the glue used by the native
``tools/cudanav_gpu_closed_loop_s_course.cu`` closed loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .._cudarobotics import (
    Esdf2DConfig,
    KissIcpConfig,
    UnknownSpacePolicy,
    VoxelMappingConfig,
    _Esdf2D,
    _KissIcpOdometry,
    _VoxelMapper,
    _esdf_2d_cpu_reference,
)

OCCUPANCY_UNKNOWN = -1
OCCUPANCY_FREE = 0
OCCUPANCY_OCCUPIED = 100

_UNKNOWN_POLICIES = {
    "free": UnknownSpacePolicy.Free,
    "occupied": UnknownSpacePolicy.Occupied,
}


def _apply_overrides(config: Any, overrides: Mapping[str, Any], name: str) -> Any:
    for key, value in overrides.items():
        if key.startswith("_") or not hasattr(config, key):
            raise TypeError(f"unknown {name} field: {key}")
        setattr(config, key, value)
    return config


def _coerce_unknown_policy(value: Any) -> UnknownSpacePolicy:
    if isinstance(value, UnknownSpacePolicy):
        return value
    key = str(value).strip().lower()
    if key not in _UNKNOWN_POLICIES:
        raise ValueError(
            f"unknown_policy must be one of {sorted(_UNKNOWN_POLICIES)}, got {value!r}"
        )
    return _UNKNOWN_POLICIES[key]


def as_points(value: Any, name: str = "points") -> np.ndarray:
    """Return a finite, C-contiguous ``(N, 3)`` float32 array."""
    array = np.ascontiguousarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_occupancy(value: Any) -> np.ndarray:
    data = value.data if isinstance(value, OccupancyGrid) else value
    array = np.ascontiguousarray(data, dtype=np.int8)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"occupancy must be a non-empty 2-D array, got {array.shape}")
    return array


def yaw_from_rotation(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return math.atan2(rotation[1, 0], rotation[0, 0])


def transform_points(points: Any, rotation: Any, translation: Any) -> np.ndarray:
    """Apply ``R p + t`` to ``(N, 3)`` points (float32, like the native loop)."""
    xyz = as_points(points)
    r = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float32).reshape(3)
    return np.ascontiguousarray(xyz @ r.T + t, dtype=np.float32)


@dataclass(frozen=True, eq=False)
class OdometryFrame:
    """Result of registering one scan with GPU KISS-ICP."""

    rotation: np.ndarray
    translation: np.ndarray
    iterations: int
    inliers: int
    rmse: float
    threshold: float
    input_points: int
    sampled_points: int
    map_points: int
    map_initialized: bool
    deskewed: bool
    point_time_span_s: float
    nn_ms: float
    normal_equation_ms: float
    deskewed_points: np.ndarray

    @property
    def x(self) -> float:
        return float(self.translation[0])

    @property
    def y(self) -> float:
        return float(self.translation[1])

    @property
    def yaw(self) -> float:
        return yaw_from_rotation(self.rotation)

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix


def _odometry_frame(raw: Mapping[str, Any]) -> OdometryFrame:
    return OdometryFrame(
        rotation=np.asarray(raw["rotation"], dtype=np.float64).reshape(3, 3),
        translation=np.asarray(raw["translation"], dtype=np.float64),
        iterations=int(raw["iterations"]),
        inliers=int(raw["inliers"]),
        rmse=float(raw["rmse"]),
        threshold=float(raw["threshold"]),
        input_points=int(raw["input_points"]),
        sampled_points=int(raw["sampled_points"]),
        map_points=int(raw["map_points"]),
        map_initialized=bool(raw["map_initialized"]),
        deskewed=bool(raw["deskewed"]),
        point_time_span_s=float(raw["point_time_span_s"]),
        nn_ms=float(raw["nn_ms"]),
        normal_equation_ms=float(raw["normal_equation_ms"]),
        deskewed_points=np.asarray(raw["deskewed_xyz"]),
    )


class KissIcpOdometry:
    """GPU KISS-ICP LiDAR odometry (point-to-plane ICP on a voxel-hash map)."""

    def __init__(self, config: KissIcpConfig | None = None, **overrides: Any) -> None:
        self.config = _apply_overrides(
            KissIcpConfig() if config is None else config, overrides, "KissIcpConfig"
        )
        self._odometry = _KissIcpOdometry(self.config)

    def reset(self, rotation: Any = None, translation: Any = None) -> None:
        """Restart odometry at an optional initial pose."""
        r = None if rotation is None else [float(v) for v in np.ravel(rotation)]
        t = None if translation is None else [float(v) for v in np.ravel(translation)]
        self._odometry.reset(r, t)

    def register(
        self,
        points: Any,
        point_times: Any = None,
        *,
        scan_start: float | None = None,
        scan_end: float | None = None,
    ) -> OdometryFrame:
        """Register one sensor-frame scan; optional per-point times enable deskew."""
        xyz = as_points(points)
        times = None
        if point_times is not None:
            times = np.ascontiguousarray(point_times, dtype=np.float32)
            if times.shape != (xyz.shape[0],):
                raise ValueError("point_times must have shape (N,)")
        return _odometry_frame(
            self._odometry.register_scan(xyz, times, scan_start, scan_end)
        )

    @property
    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        raw = self._odometry.pose()
        return (
            np.asarray(raw["rotation"], dtype=np.float64).reshape(3, 3),
            np.asarray(raw["translation"], dtype=np.float64),
        )

    @property
    def frame_count(self) -> int:
        return int(self._odometry.frame_count())

    @property
    def map_point_count(self) -> int:
        return int(self._odometry.map_point_count())

    def map_points(self) -> np.ndarray:
        """Copy of the local voxel-hash map points, ``(M, 3)`` float32."""
        return np.asarray(self._odometry.map_snapshot())

    def timing(self) -> dict[str, float]:
        """Accumulated per-stage timing reported by the CUDA core."""
        return dict(self._odometry.timing())


@dataclass(frozen=True, eq=False)
class OccupancyGrid:
    """2-D occupancy (row-major ``(height, width)`` int8 in ``[-1, 100]``)."""

    data: np.ndarray
    resolution: float
    origin: tuple[float, float]
    gpu_ms: float = 0.0

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def occupied_cells(self) -> int:
        return int(np.count_nonzero(self.data == OCCUPANCY_OCCUPIED))

    @property
    def unknown_cells(self) -> int:
        return int(np.count_nonzero(self.data < 0))

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """World ``(x, y)`` of every cell center as two ``(H, W)`` float32 arrays."""
        res = np.float32(self.resolution)
        columns = np.float32(self.origin[0]) + (
            np.arange(self.width, dtype=np.float32) + np.float32(0.5)
        ) * res
        rows = np.float32(self.origin[1]) + (
            np.arange(self.height, dtype=np.float32) + np.float32(0.5)
        ) * res
        return np.meshgrid(columns, rows)


class VoxelMap:
    """Rolling GPU log-odds voxel map with a 2-D occupancy projection."""

    def __init__(
        self,
        config: VoxelMappingConfig | None = None,
        *,
        center: tuple[float, float] = (0.0, 0.0),
        **overrides: Any,
    ) -> None:
        self.config = _apply_overrides(
            VoxelMappingConfig() if config is None else config,
            overrides,
            "VoxelMappingConfig",
        )
        self._mapper = _VoxelMapper(self.config)
        self.reset(*center)

    def reset(self, center_x: float = 0.0, center_y: float = 0.0) -> None:
        self._mapper.reset(float(center_x), float(center_y))

    def integrate(self, points_world: Any, sensor_origin: Any) -> dict[str, Any]:
        """Raycast world-frame points from ``sensor_origin`` into the map."""
        origin = [float(v) for v in np.ravel(sensor_origin)]
        if len(origin) != 3:
            raise ValueError("sensor_origin must have three values")
        return dict(self._mapper.integrate_scan(as_points(points_world, "points_world"), origin))

    @property
    def grid_info(self) -> dict[str, Any]:
        return dict(self._mapper.grid_info())

    def occupancy(self) -> OccupancyGrid:
        data, grid, gpu_ms = self._mapper.occupancy_projection()
        return OccupancyGrid(
            data=np.asarray(data),
            resolution=float(grid["resolution"]),
            origin=(float(grid["origin_x"]), float(grid["origin_y"])),
            gpu_ms=float(gpu_ms),
        )


@dataclass(frozen=True, eq=False)
class DistanceField:
    """Unsigned 2-D Euclidean distance (metres) to the nearest obstacle cell."""

    distances: np.ndarray
    resolution: float
    max_distance: float
    origin: tuple[float, float]
    occupied_cells: int
    unknown_cells: int
    gpu_ms: float = 0.0

    @property
    def width(self) -> int:
        return int(self.distances.shape[1])

    @property
    def height(self) -> int:
        return int(self.distances.shape[0])


def _distance_field(raw: tuple[Any, Mapping[str, Any]], origin: tuple[float, float]) -> DistanceField:
    distances, info = raw
    return DistanceField(
        distances=np.asarray(distances),
        resolution=float(info["resolution"]),
        max_distance=float(info["max_distance"]),
        origin=(float(origin[0]), float(origin[1])),
        occupied_cells=int(info["occupied_cells"]),
        unknown_cells=int(info["unknown_cells"]),
        gpu_ms=float(info["gpu_ms"]),
    )


def _grid_geometry(
    occupancy: Any, resolution: float | None, origin: tuple[float, float] | None
) -> tuple[float, tuple[float, float]]:
    if isinstance(occupancy, OccupancyGrid):
        res = occupancy.resolution if resolution is None else resolution
        org = occupancy.origin if origin is None else origin
    else:
        if resolution is None:
            raise ValueError("resolution is required for a raw occupancy array")
        res = resolution
        org = (0.0, 0.0) if origin is None else origin
    return float(res), (float(org[0]), float(org[1]))


class Esdf2D:
    """Exact GPU 2-D Euclidean distance transform of an occupancy grid."""

    def __init__(
        self,
        config: Esdf2DConfig | None = None,
        *,
        unknown_policy: Any = None,
        **overrides: Any,
    ) -> None:
        self.config = _apply_overrides(
            Esdf2DConfig() if config is None else config, overrides, "Esdf2DConfig"
        )
        if unknown_policy is not None:
            self.config.unknown_policy = _coerce_unknown_policy(unknown_policy)
        self._esdf = _Esdf2D(self.config)

    def compute(
        self,
        occupancy: OccupancyGrid | Any,
        *,
        max_distance: float = 2.0,
        resolution: float | None = None,
        origin: tuple[float, float] | None = None,
    ) -> DistanceField:
        res, org = _grid_geometry(occupancy, resolution, origin)
        raw = self._esdf.compute(_as_occupancy(occupancy), res, float(max_distance))
        return _distance_field(raw, org)


def esdf_2d_cpu_reference(
    occupancy: OccupancyGrid | Any,
    *,
    max_distance: float = 2.0,
    resolution: float | None = None,
    origin: tuple[float, float] | None = None,
    occupancy_threshold: int = 50,
    unknown_policy: Any = "occupied",
) -> DistanceField:
    """Exhaustive CPU oracle for :class:`Esdf2D` (no GPU required)."""
    res, org = _grid_geometry(occupancy, resolution, origin)
    raw = _esdf_2d_cpu_reference(
        _as_occupancy(occupancy),
        res,
        float(max_distance),
        int(occupancy_threshold),
        _coerce_unknown_policy(unknown_policy),
    )
    return _distance_field(raw, org)


def clear_footprint(grid: OccupancyGrid, x: float, y: float, radius: float) -> OccupancyGrid:
    """Mark cells whose centers lie within ``radius`` of ``(x, y)`` as free.

    The robot's own body is otherwise integrated as an obstacle at start-up.
    """
    if radius <= 0.0:
        return grid
    centers_x, centers_y = grid.cell_centers()
    inside = np.hypot(centers_x - np.float32(x), centers_y - np.float32(y)) <= np.float32(radius)
    data = grid.data.copy()
    data[inside] = OCCUPANCY_FREE
    return OccupancyGrid(data=data, resolution=grid.resolution, origin=grid.origin, gpu_ms=grid.gpu_ms)


def costmap_from_esdf(
    occupancy: OccupancyGrid | Any,
    distance_field: DistanceField | Any,
    *,
    occupied_threshold: int = 50,
    inscribed_radius: float = 0.25,
    inflation_radius: float = 1.0,
    cost_scaling_factor: float = 3.0,
) -> np.ndarray:
    """Build the uint8 MPPI costmap from occupancy plus its distance field.

    ``255`` unknown, ``254`` lethal, ``253`` inscribed, then the Nav2-style
    exponential decay ``252 * exp(-k (d - r_inscribed))`` clamped to
    ``[1, 252]`` out to ``inflation_radius``; free space is ``0``.
    """
    occ = _as_occupancy(occupancy)
    distances = np.asarray(
        distance_field.distances if isinstance(distance_field, DistanceField) else distance_field,
        dtype=np.float32,
    )
    if distances.shape != occ.shape:
        raise ValueError(
            f"distance field shape {distances.shape} does not match occupancy {occ.shape}"
        )
    inscribed = np.float32(inscribed_radius)
    decay = np.float32(252.0) * np.exp(
        np.float32(-cost_scaling_factor) * (distances - inscribed), dtype=np.float32
    )
    inflated = np.clip(decay, np.float32(1.0), np.float32(252.0)).astype(np.uint8)
    costmap = np.zeros(occ.shape, dtype=np.uint8)
    near = distances < np.float32(inflation_radius)
    costmap[near] = inflated[near]
    costmap[distances <= inscribed] = 253
    costmap[occ >= occupied_threshold] = 254
    costmap[occ < 0] = 255
    return costmap
