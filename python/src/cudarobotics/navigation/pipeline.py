"""One object that runs odometry -> mapping -> ESDF -> costmap -> CUDA MPPI."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np

from .. import MppiPlanner
from .._cudarobotics import (
    _CudaNavGridChain,
    Esdf2DConfig,
    KissIcpConfig,
    MppiParams,
    UnknownSpacePolicy,
    VoxelMappingConfig,
)
from .tracking import OccupancyAge, ScanObjectTracker, TrackerResult
from .core import (
    DistanceField,
    Esdf2D,
    KissIcpOdometry,
    OccupancyGrid,
    OdometryFrame,
    VoxelMap,
    clear_footprint,
    costmap_from_esdf,
    transform_points,
)


def default_kiss_icp_config() -> KissIcpConfig:
    """KISS-ICP capacities used by the native S-course closed loop."""
    config = KissIcpConfig()
    config.max_scan_points = 4096
    config.max_map_points = 100000
    config.hash_capacity = 1 << 18
    return config


def default_voxel_mapping_config() -> VoxelMappingConfig:
    """25.6 m x 25.6 m rolling grid at 0.10 m (native closed-loop values)."""
    config = VoxelMappingConfig()
    config.width = 256
    config.height = 256
    config.depth = 32
    config.resolution = 0.10
    config.origin_z = -2.0
    config.max_range = 12.0
    config.projection_min_z = -0.5
    config.projection_max_z = 2.0
    config.rolling_margin_cells = 48
    return config


def default_esdf_config(voxel: VoxelMappingConfig) -> Esdf2DConfig:
    config = Esdf2DConfig()
    config.max_width = voxel.width
    config.max_height = voxel.height
    config.unknown_policy = UnknownSpacePolicy.Free
    return config


def default_mppi_params(*, reverse: bool = False) -> MppiParams:
    """DiffDrive MPPI tuning of the native closed loop (forward or reverse)."""
    params = MppiParams()
    params.batch_size = 2048
    params.time_steps = 56
    params.model_dt = 0.05
    params.v_max = 0.55
    params.v_min = -0.35 if reverse else 0.0
    params.w_max = 1.9
    params.costmap_weight = 8.0
    params.path_weight = 25.0
    params.path_follow_weight = 10.0
    params.path_angle_weight = 0.75
    params.distance_field_weight = 3.0
    params.distance_field_cutoff = 0.75
    # Only active when step() receives dynamic_obstacles.
    params.dynamic_obstacle_weight = 25.0
    params.dynamic_obstacle_margin = 0.4
    params.dynamic_obstacle_robot_radius = 0.25
    params.dynamic_obstacle_growth = 0.1
    if reverse:
        params.backward_weight = 0.0
    return params


@dataclass(frozen=True, eq=False)
class LocalPath:
    xy: np.ndarray
    goal: tuple[float, float, float]
    goal_is_final: bool


def local_path_window(
    path: Any,
    x: float,
    y: float,
    *,
    lookahead_m: float = 3.0,
    max_points: int = 256,
) -> LocalPath:
    """Slice ``path`` from the nearest point forward by ``lookahead_m`` metres."""
    xy = np.ascontiguousarray(path, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2:
        raise ValueError("path must have shape (N, 2) with N >= 2")
    deltas = xy - np.asarray([x, y], dtype=np.float32)
    nearest = int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))
    end = nearest
    arc = 0.0
    for index in range(nearest + 1, xy.shape[0]):
        arc += math.hypot(
            float(xy[index, 0] - xy[index - 1, 0]), float(xy[index, 1] - xy[index - 1, 1])
        )
        end = index
        if arc >= lookahead_m or end - nearest + 1 >= max_points:
            break
    if end == nearest and end + 1 < xy.shape[0]:
        end += 1
    before = end - 1 if end > nearest else nearest
    goal_x, goal_y = float(xy[end, 0]), float(xy[end, 1])
    goal_yaw = math.atan2(goal_y - float(xy[before, 1]), goal_x - float(xy[before, 0]))
    return LocalPath(
        xy=np.ascontiguousarray(xy[nearest : end + 1]),
        goal=(goal_x, goal_y, goal_yaw),
        goal_is_final=end + 1 == xy.shape[0],
    )


STAGE_CHAINS = ("device", "host")
# Extra radius cleared around each tracked moving obstacle in the occupancy
# grid; the planner sees those obstacles through their predicted trajectory.
DYNAMIC_CLEARING_MARGIN = 0.15


def _clearing_discs(
    dynamic: np.ndarray | None, history: np.ndarray | None
) -> list[tuple[float, float, float]]:
    """Occupancy discs to clear: tracked objects (with margin) and their recent path."""
    discs = [
        (float(x), float(y), float(radius) + DYNAMIC_CLEARING_MARGIN)
        for x, y, _vx, _vy, radius in ([] if dynamic is None else dynamic)
    ]
    discs += [(float(x), float(y), float(radius)) for x, y, radius in ([] if history is None else history)]
    return discs


def _annotate(step: "PipelineStep", tracked: TrackerResult | None) -> "PipelineStep":
    if tracked is not None:
        step.tracks = tracked.tracks
        step.track_ids = tracked.track_ids
        step.dynamic_point_count = int(tracked.dynamic_mask.sum())
    return step


def _dynamic_rows(dynamic_obstacles: Any) -> np.ndarray | None:
    if dynamic_obstacles is None:
        return None
    rows = np.ascontiguousarray(dynamic_obstacles, dtype=np.float32)
    if rows.size == 0:
        return None
    if rows.ndim != 2 or rows.shape[1] != 5:
        raise ValueError("dynamic_obstacles must have shape (N, 5): x, y, vx, vy, radius")
    if not np.all(np.isfinite(rows)):
        raise ValueError("dynamic_obstacles must be finite")
    return rows


class PipelineStep:
    """Every intermediate product of one CudaNav frame.

    With ``stage_chain="device"`` the occupancy, distance field and costmap stay
    on the GPU and are copied to NumPy on first access. They can be read until
    the next :meth:`CudaNavPipeline.step` call, which overwrites them; reading
    later raises ``RuntimeError``. ``occupied_cells`` is always available.
    """

    def __init__(
        self,
        *,
        odometry: OdometryFrame,
        mapping: dict[str, Any],
        occupied_cells: int,
        command: tuple[float, float] | None,
        planner: dict[str, Any] | None,
        timings_ms: dict[str, float],
        stage_chain: str,
        occupancy: OccupancyGrid | None = None,
        distance_field: DistanceField | None = None,
        costmap: np.ndarray | None = None,
        loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.odometry = odometry
        self.mapping = mapping
        self.occupied_cells = int(occupied_cells)
        self.command = command
        self.planner = planner
        self.timings_ms = timings_ms
        self.stage_chain = stage_chain
        # Scan tracker outputs (None when no tracker ran this frame).
        self.tracks: np.ndarray | None = None
        self.track_ids: np.ndarray | None = None
        self.dynamic_point_count = 0
        self._grids: dict[str, Any] = {
            "occupancy": occupancy,
            "distance_field": distance_field,
            "costmap": costmap,
        }
        self._loader = loader

    def _grid(self, name: str) -> Any:
        if self._grids[name] is None:
            if self._loader is None:
                raise RuntimeError(f"{name} is not available for this step")
            self._grids[name] = self._loader(name)
        return self._grids[name]

    @property
    def occupancy(self) -> OccupancyGrid:
        return self._grid("occupancy")

    @property
    def distance_field(self) -> DistanceField:
        return self._grid("distance_field")

    @property
    def costmap(self) -> np.ndarray:
        return self._grid("costmap")

    def materialize(self) -> "PipelineStep":
        """Copy every grid to the host now, so the step outlives the next frame."""
        for name in self._grids:
            self._grid(name)
        return self

    def __repr__(self) -> str:
        return (
            f"PipelineStep(stage_chain={self.stage_chain!r}, command={self.command}, "
            f"occupied_cells={self.occupied_cells}, timings_ms={self.timings_ms})"
        )


def _device_grid_loader(
    owner: "CudaNavPipeline",
    generation: int,
    resolution: float,
    origin: tuple[float, float],
    projection_gpu_ms: float,
    esdf_info: dict[str, Any],
) -> Callable[[str], Any]:
    def load(name: str) -> Any:
        # The occupancy and distance buffers belong to the owner's voxel map
        # and ESDF; referencing `owner` here keeps the pipeline alive with the step.
        chain = owner._chain
        if chain.generation != generation:
            raise RuntimeError(
                "a later CudaNavPipeline.step() overwrote this step's GPU grids; "
                "read them (or call materialize()) before the next step"
            )
        if name == "occupancy":
            return OccupancyGrid(
                data=np.asarray(chain.occupancy()),
                resolution=resolution,
                origin=origin,
                gpu_ms=projection_gpu_ms,
            )
        if name == "distance_field":
            return DistanceField(
                distances=np.asarray(chain.distances()),
                resolution=float(esdf_info["resolution"]),
                max_distance=float(esdf_info["max_distance"]),
                origin=origin,
                occupied_cells=int(esdf_info["occupied_cells"]),
                unknown_cells=int(esdf_info["unknown_cells"]),
                gpu_ms=float(esdf_info["gpu_ms"]),
            )
        return np.asarray(chain.costmap())

    return load


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


class CudaNavPipeline:
    """GPU KISS-ICP -> voxel map -> ESDF -> costmap -> CUDA MPPI, one frame at a time.

    ``step`` registers a sensor-frame scan, maps it with the estimated pose,
    and, when a path is supplied, returns the MPPI ``(v, w)`` command. Without a
    path it is an odometry + mapping + ESDF pipeline for recorded data.

    ``stage_chain="device"`` (the default) keeps occupancy, ESDF and costmap on
    the GPU from the voxel projection to MPPI. ``"host"`` copies each product to
    NumPy between stages and builds the costmap with :func:`costmap_from_esdf`.
    """

    def __init__(
        self,
        *,
        kiss_icp: KissIcpConfig | None = None,
        voxel_map: VoxelMappingConfig | None = None,
        esdf: Esdf2DConfig | None = None,
        mppi: MppiParams | None = None,
        reverse_mppi: MppiParams | None = None,
        esdf_max_distance: float = 2.0,
        footprint_clearing_radius: float = 0.30,
        enable_planner: bool = True,
        stage_chain: str = "device",
        tracker: ScanObjectTracker | None = None,
        scan_period: float = 0.1,
    ) -> None:
        if stage_chain not in STAGE_CHAINS:
            raise ValueError(f"stage_chain must be one of {STAGE_CHAINS}")
        self.stage_chain = stage_chain
        voxel_config = default_voxel_mapping_config() if voxel_map is None else voxel_map
        self.odometry = KissIcpOdometry(default_kiss_icp_config() if kiss_icp is None else kiss_icp)
        self.voxel_map = VoxelMap(voxel_config)
        self.esdf = Esdf2D(default_esdf_config(voxel_config) if esdf is None else esdf)
        self.esdf_max_distance = float(esdf_max_distance)
        self.footprint_clearing_radius = float(footprint_clearing_radius)
        # Optional scan-based moving-object tracker; see step().
        self.tracker = tracker
        self.scan_period = float(scan_period)
        self._frames = 0
        self._occupancy_age = OccupancyAge()
        self._chain = (
            _CudaNavGridChain(int(voxel_config.width), int(voxel_config.height))
            if stage_chain == "device"
            else None
        )
        self._mppi_params = {
            False: default_mppi_params() if mppi is None else mppi,
            True: default_mppi_params(reverse=True) if reverse_mppi is None else reverse_mppi,
        }
        if tracker is not None:
            # Tracked discs carry position error; let the planner tolerate a
            # small existing overlap instead of freezing (see TrackerConfig).
            for params in self._mppi_params.values():
                if params.dynamic_obstacle_overlap_allowance <= 0.0:
                    params.dynamic_obstacle_overlap_allowance = (
                        tracker.config.planner_overlap_allowance
                    )
        self._planners: dict[bool, MppiPlanner] = {}
        if enable_planner:
            self._planner(False)

    @property
    def mppi_params(self) -> MppiParams:
        return self._mppi_params[False]

    def _planner(self, reverse: bool) -> MppiPlanner:
        if reverse not in self._planners:
            self._planners[reverse] = MppiPlanner(self._mppi_params[reverse])
        return self._planners[reverse]

    def reset_planner(self, *, reverse: bool = False) -> None:
        """Clear the warm-started control sequence (e.g. after a new mission)."""
        self._planner(reverse).reset()

    def step(
        self,
        points: Any,
        *,
        point_times: Any = None,
        scan_start: float | None = None,
        scan_end: float | None = None,
        path: Any = None,
        reverse: bool = False,
        dynamic_obstacles: Any = None,
        stamp: float | None = None,
    ) -> PipelineStep:
        """Process one scan.

        ``dynamic_obstacles`` is optional ``(N, 5)`` rows of ``x, y, vx, vy,
        radius`` in the map frame (tracked moving objects). Their cells are
        cleared from the occupancy grid and MPPI avoids their constant-velocity
        predictions instead; leave it ``None`` to treat everything as static.

        With a ``tracker`` and no ``dynamic_obstacles``, the pipeline finds the
        moving objects itself: points that land in previously free cells are
        tracked, kept out of the voxel map, and their confirmed tracks (plus
        their recent path) are handled like ``dynamic_obstacles``. ``stamp``
        is the scan time in seconds (default: frame index * ``scan_period``).
        """
        dynamic = _dynamic_rows(dynamic_obstacles)
        stamp = self._frames * self.scan_period if stamp is None else float(stamp)
        self._frames += 1
        timings: dict[str, float] = {}
        started = time.perf_counter()

        stage = time.perf_counter()
        frame = self.odometry.register(
            points, point_times, scan_start=scan_start, scan_end=scan_end
        )
        timings["odometry"] = _ms(stage)

        stage = time.perf_counter()
        scan = frame.deskewed_points if frame.deskewed and frame.deskewed_points.size else points
        world = transform_points(scan, frame.rotation, frame.translation)
        tracked: TrackerResult | None = None
        if self.tracker is not None and dynamic is None:
            candidates, structure = self._occupancy_age.classify(
                world[:, :2], self.tracker.config.static_frames
            )
            tracked = self.tracker.update(
                world[:, :2],
                candidates,
                stamp,
                structure,
                sensor_xy=(float(frame.translation[0]), float(frame.translation[1])),
            )
            static = world[~tracked.dynamic_mask]
            # Keep the moving objects' returns out of the map (if every point is
            # dynamic, map them anyway rather than skip the frame).
            if static.shape[0]:
                world = static
            # Moving tracks with their velocity, plus still objects that are
            # not in the map yet (as stationary obstacles).
            objects = np.concatenate([tracked.tracks, tracked.static_objects])
            dynamic = objects if objects.shape[0] else None
        discs = _clearing_discs(dynamic, None if tracked is None else tracked.history)
        mapping = self.voxel_map.integrate(world, frame.translation)
        if self._chain is not None:
            result = self._step_device(
                frame, mapping, path, reverse, dynamic, discs, timings, stage, started
            )
            return _annotate(result, tracked)

        occupancy = clear_footprint(
            self.voxel_map.occupancy(), frame.x, frame.y, self.footprint_clearing_radius
        )
        if self.tracker is not None:
            self._occupancy_age.update(occupancy)
        for x, y, radius in discs:
            occupancy = clear_footprint(occupancy, x, y, radius)
        timings["mapping"] = _ms(stage)

        stage = time.perf_counter()
        distance_field = self.esdf.compute(occupancy, max_distance=self.esdf_max_distance)
        timings["esdf"] = _ms(stage)

        stage = time.perf_counter()
        costmap = costmap_from_esdf(occupancy, distance_field)
        timings["costmap"] = _ms(stage)

        command = None
        planner = None
        if path is not None:
            stage = time.perf_counter()
            local = local_path_window(path, frame.x, frame.y)
            self._planner(reverse).set_dynamic_obstacles(dynamic)
            v, _vy, w, planner = self._planner(reverse).compute(
                (frame.x, frame.y, frame.yaw),
                costmap,
                local.xy,
                local.goal,
                origin=occupancy.origin,
                resolution=occupancy.resolution,
                goal_is_final=local.goal_is_final,
            )
            command = (float(v), float(w))
            planner = dict(planner)
            timings["mppi"] = _ms(stage)
        timings["total"] = _ms(started)
        return _annotate(PipelineStep(
            odometry=frame,
            mapping=mapping,
            occupied_cells=occupancy.occupied_cells,
            command=command,
            planner=planner,
            timings_ms=timings,
            stage_chain="host",
            occupancy=occupancy,
            distance_field=distance_field,
            costmap=costmap,
        ), tracked)

    def _step_device(
        self,
        frame: OdometryFrame,
        mapping: dict[str, Any],
        path: Any,
        reverse: bool,
        dynamic: np.ndarray | None,
        discs: list[tuple[float, float, float]],
        timings: dict[str, float],
        stage: float,
        started: float,
    ) -> PipelineStep:
        chain = self._chain
        projection = chain.project(
            self.voxel_map._mapper, frame.x, frame.y, self.footprint_clearing_radius
        )
        occupied_cells = int(projection["occupied_cells"])
        if self.tracker is not None:
            grid = projection["grid"]
            self._occupancy_age.update(
                OccupancyGrid(
                    data=np.asarray(chain.occupancy()),
                    resolution=float(grid["resolution"]),
                    origin=(float(grid["origin_x"]), float(grid["origin_y"])),
                )
            )
        for x, y, radius in discs:
            occupied_cells = int(chain.clear_disc(x, y, radius))
        grid = projection["grid"]
        resolution = float(grid["resolution"])
        origin = (float(grid["origin_x"]), float(grid["origin_y"]))
        timings["mapping"] = _ms(stage)

        stage = time.perf_counter()
        esdf_info = dict(chain.distance(self.esdf._esdf, resolution, self.esdf_max_distance))
        timings["esdf"] = _ms(stage)

        stage = time.perf_counter()
        chain.build()
        timings["costmap"] = _ms(stage)

        command = None
        planner = None
        if path is not None:
            stage = time.perf_counter()
            local = local_path_window(path, frame.x, frame.y)
            self._planner(reverse).set_dynamic_obstacles(dynamic)
            v, _vy, w, planner = chain.plan(
                self._planner(reverse)._planner,
                (frame.x, frame.y, frame.yaw),
                local.xy,
                local.goal,
                origin,
                resolution,
                local.goal_is_final,
            )
            command = (float(v), float(w))
            planner = dict(planner)
            timings["mppi"] = _ms(stage)
        timings["total"] = _ms(started)
        return PipelineStep(
            odometry=frame,
            mapping=mapping,
            occupied_cells=occupied_cells,
            command=command,
            planner=planner,
            timings_ms=timings,
            stage_chain="device",
            loader=_device_grid_loader(
                self,
                int(projection["generation"]),
                resolution,
                origin,
                float(projection["gpu_ms"]),
                esdf_info,
            ),
        )
