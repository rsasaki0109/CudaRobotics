"""S-course closed loop with pedestrians crossing both corridors.

Two moving discs sweep across the corridors the robot must pass
(:func:`~cudarobotics.navigation.simulation.crossing_obstacles`). They follow
their schedules and do not steer around the robot, but they pause instead of
stepping into it, so a contact can only come from the robot's own motion.
Three ways of handling them are compared:

``map``
    The obstacles are only LiDAR returns: they enter the voxel map like
    walls and the planner has no prediction. This is the baseline.
``predicted``
    A tracker reports each obstacle's position and velocity (an oracle from
    the simulator, optionally delayed, noisy, and dropping detections). Its
    points are kept out of the map, its cells are cleared from the occupancy
    grid, and CUDA MPPI avoids the constant-velocity prediction.
``tracked_static``
    Like ``predicted`` but the planner gets zero velocities, isolating the
    value of the motion prediction from keeping obstacles out of the map.
``tracked_scan``
    No oracle: the pipeline's :class:`~cudarobotics.navigation.tracking.ScanObjectTracker`
    finds the pedestrians in the scan itself (points in previously free
    cells), keeps them out of the map and gives MPPI its tracks. Tracking
    accuracy against the true pedestrians is reported in ``tracking``.

Everything is synthetic; except in ``tracked_scan`` the tracker is an oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np

from .evidence import SCHEMA_VERSION, latency_summary, percentile
from .pipeline import CudaNavPipeline, PipelineStep
from .tracking import ScanObjectTracker
from .simulation import (
    DT,
    GOAL_TOLERANCE,
    PATH_SPACING,
    WAYPOINTS,
    ROBOT_RADIUS,
    Pose2D,
    collides,
    course_segments,
    crossing_obstacles,
    crowd_obstacles,
    dynamic_clearance,
    follower_obstacle,
    head_on_obstacles,
    interpolate_path,
    point_segment_distance,
    simulate_scan_with_obstacles,
    step_unicycle,
)

SCENARIO = "cudanav_s_course_crossing"
DYNAMIC_MODES = ("map", "tracked_static", "predicted", "tracked_scan")
LAYOUTS = ("crossing", "head_on", "overtake", "crowd")
PATIENCE_S = 3.0   # pedestrians outside ``crossing`` turn around after waiting this long
CROWD_SIZE = 5
TRACK_MATCH_M = 0.5        # a visible pedestrian counts as tracked within this distance
FALSE_TRACK_M = 0.7        # a confirmed track this far from every pedestrian is false
VISIBLE_RAYS = 3           # rays that must hit a pedestrian for it to count as visible
DEADLINE_MS = 150.0
MAXIMUM_FRAMES = 600  # 60 s of simulated time for one traversal
MAXIMUM_DRIFT_PERCENT = 5.0
NON_WALL_SAMPLE_PERIOD = 10  # frames between ghost-cell samples
NON_WALL_DISTANCE_M = 0.15  # occupied cells farther than this from any wall (ghosts + live obstacles)
TRAJECTORY_COLUMNS = (
    "step",
    "time_s",
    "truth_x",
    "truth_y",
    "truth_yaw",
    "estimated_x",
    "estimated_y",
    "estimated_yaw",
    "obstacle_0_x",
    "obstacle_0_y",
    "obstacle_1_x",
    "obstacle_1_y",
    "dynamic_clearance_m",
    "tracked_obstacles",
    "occupied_cells",
    "valid_rollout_ratio",
    "all_colliding",
    "retreating",
    "command_v",
    "command_w",
    "solve_ms",
    "frame_ms",
)


@dataclass(frozen=True)
class ObservationModel:
    """Degradations applied to the oracle tracks in ``predicted`` mode."""

    latency_frames: int = 0
    position_noise_m: float = 0.0
    velocity_noise_mps: float = 0.0
    dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.latency_frames < 0
            or self.position_noise_m < 0.0
            or self.velocity_noise_mps < 0.0
            or not 0.0 <= self.dropout_probability < 1.0
        ):
            raise ValueError("invalid observation model")


@dataclass(frozen=True, eq=False)
class CrossingResult:
    summary: dict[str, Any]
    rows: list[tuple[Any, ...]]
    last_step: PipelineStep | None
    stage_latency_ms: dict[str, dict[str, float]]
    obstacle_tracks: np.ndarray  # (frames, N, 5) true obstacle states


YIELD_CLEARANCE_M = 0.05  # pedestrians pause rather than come closer than this


class PedestrianClock:
    """Per-obstacle schedule time that stops while a pedestrian yields.

    With a finite ``patience_s``, a pedestrian who has waited that long for
    the robot turns around and walks its schedule backwards instead.
    """

    def __init__(self, obstacles, patience_s: float = math.inf) -> None:
        self.obstacles = tuple(obstacles)
        self.elapsed = [0.0] * len(self.obstacles)
        self.paused = [False] * len(self.obstacles)
        self.direction = [1.0] * len(self.obstacles)
        self.waited = [0.0] * len(self.obstacles)
        self.patience_s = float(patience_s)
        self.turnarounds = 0

    def states(self) -> np.ndarray:
        rows = []
        for obstacle, elapsed, paused, direction in zip(
            self.obstacles, self.elapsed, self.paused, self.direction
        ):
            x, y, vx, vy = obstacle.state(elapsed)
            if paused:
                vx = vy = 0.0
            rows.append((x, y, direction * vx, direction * vy, obstacle.radius))
        return np.asarray(rows, dtype=np.float32).reshape(-1, 5)

    def advance(self, robot: Pose2D, dt: float) -> None:
        for index, obstacle in enumerate(self.obstacles):
            step = self.direction[index] * dt
            x, y, _vx, _vy = obstacle.state(self.elapsed[index] + step)
            blocked = (
                math.hypot(x - robot.x, y - robot.y) - obstacle.radius - ROBOT_RADIUS
                < YIELD_CLEARANCE_M
            )
            current = obstacle.state(self.elapsed[index])
            # Only yield when the step brings the pedestrian closer to the robot.
            closer = math.hypot(x - robot.x, y - robot.y) < math.hypot(
                current[0] - robot.x, current[1] - robot.y
            )
            self.paused[index] = blocked and closer
            if not self.paused[index]:
                self.waited[index] = 0.0
                self.elapsed[index] += step
                continue
            self.waited[index] += dt
            if self.waited[index] >= self.patience_s:
                self.direction[index] = -self.direction[index]
                self.waited[index] = 0.0
                self.turnarounds += 1


def _score_tracks(
    tally: dict[str, Any],
    tracks: np.ndarray | None,
    states: np.ndarray,
    truth: Pose2D,
    points: np.ndarray,
    on_obstacle: np.ndarray,
) -> None:
    """Compare one frame's confirmed tracks with the true pedestrians."""
    tracks = np.zeros((0, 5), np.float32) if tracks is None else np.asarray(tracks)
    # Which pedestrian each obstacle return belongs to (nearest centre).
    cos, sin = math.cos(truth.yaw), math.sin(truth.yaw)
    hits = points[on_obstacle]
    world_x = truth.x + cos * hits[:, 0] - sin * hits[:, 1]
    world_y = truth.y + sin * hits[:, 0] + cos * hits[:, 1]
    rays = np.zeros(states.shape[0], dtype=np.int64)
    if hits.shape[0]:
        owner = np.argmin(
            np.hypot(world_x[:, None] - states[None, :, 0], world_y[:, None] - states[None, :, 1]),
            axis=1,
        )
        rays = np.bincount(owner, minlength=states.shape[0]) // 3  # three height layers
    for index, (x, y, vx, vy, _radius) in enumerate(states):
        if rays[index] < VISIBLE_RAYS:
            continue
        tally["visible_pedestrian_frames"] += 1
        if tracks.shape[0] == 0:
            continue
        distance = np.hypot(tracks[:, 0] - x, tracks[:, 1] - y)
        nearest = int(np.argmin(distance))
        if distance[nearest] <= TRACK_MATCH_M:
            tally["tracked_pedestrian_frames"] += 1
            tally["position_squared_errors"].append(float(distance[nearest] ** 2))
            tally["velocity_squared_errors"].append(
                float((tracks[nearest, 2] - vx) ** 2 + (tracks[nearest, 3] - vy) ** 2)
            )
    for x, y, *_rest in tracks:
        if np.min(np.hypot(states[:, 0] - x, states[:, 1] - y)) > FALSE_TRACK_M:
            tally["false_track_frames"] += 1


def _tracking_summary(tally: dict[str, Any]) -> dict[str, Any]:
    visible = tally["visible_pedestrian_frames"]
    matched = tally["tracked_pedestrian_frames"]
    positions = tally["position_squared_errors"]
    velocities = tally["velocity_squared_errors"]
    return {
        "visible_pedestrian_frames": visible,
        "tracked_pedestrian_frames": matched,
        "recall": matched / visible if visible else None,
        "position_rmse_m": math.sqrt(sum(positions) / len(positions)) if positions else None,
        "velocity_rmse_mps": math.sqrt(sum(velocities) / len(velocities)) if velocities else None,
        "false_track_frames": tally["false_track_frames"],
        "match_distance_m": TRACK_MATCH_M,
        "false_track_distance_m": FALSE_TRACK_M,
    }


def crossing_phases(seed: int) -> tuple[float, float]:
    """Deterministic schedule offsets (seconds) for the two pedestrians."""
    rng = np.random.default_rng(seed)
    first = crossing_obstacles()[0]
    second = crossing_obstacles()[1]
    periods = [
        2.0 * math.hypot(o.end[0] - o.start[0], o.end[1] - o.start[1]) / o.speed
        for o in (first, second)
    ]
    return float(rng.uniform(0.0, periods[0])), float(rng.uniform(0.0, periods[1]))


def _segment_phases(obstacles, seed: int) -> tuple[float, ...]:
    rng = np.random.default_rng(seed)
    return tuple(
        float(rng.uniform(0.0, 2.0 * math.hypot(o.end[0] - o.start[0], o.end[1] - o.start[1]) / o.speed))
        for o in obstacles
    )


def layout_obstacles(layout: str, seed: int) -> tuple[tuple[Any, ...], float]:
    """The pedestrians of ``layout`` for ``seed``, and their patience in seconds.

    ``crossing``: two pedestrians cross the corridors and wait for the robot
    indefinitely. The other layouts use pedestrians that turn around after
    ``PATIENCE_S`` of waiting:
    - ``head_on``: two pedestrians walk the corridors against the robot;
    - ``overtake``: the two crossing pedestrians plus a faster one walking the
      robot's route from just behind it;
    - ``crowd``: ``CROWD_SIZE`` pedestrians pace random segments.
    """
    if layout == "crossing":
        return crossing_obstacles(crossing_phases(seed)), math.inf
    if layout == "head_on":
        return head_on_obstacles(_segment_phases(head_on_obstacles(), seed)), PATIENCE_S
    if layout == "overtake":
        rng = np.random.default_rng(seed + 31)
        follower = follower_obstacle(phase=float(rng.uniform(0.0, 1.0)))
        return crossing_obstacles(crossing_phases(seed)) + (follower,), PATIENCE_S
    if layout == "crowd":
        return crowd_obstacles(seed, CROWD_SIZE), PATIENCE_S
    raise ValueError(f"layout must be one of {LAYOUTS}")


def observe(
    history: list[np.ndarray],
    model: ObservationModel,
    rng: np.random.Generator,
) -> np.ndarray:
    """Tracks the planner receives: delayed, noisy, possibly missing true states."""
    states = history[max(0, len(history) - 1 - model.latency_frames)].copy()
    if model.position_noise_m > 0.0:
        states[:, 0:2] += rng.normal(0.0, model.position_noise_m, size=(states.shape[0], 2))
    if model.velocity_noise_mps > 0.0:
        states[:, 2:4] += rng.normal(0.0, model.velocity_noise_mps, size=(states.shape[0], 2))
    if model.dropout_probability > 0.0:
        keep = rng.random(states.shape[0]) >= model.dropout_probability
        states = states[keep]
    return states.astype(np.float32)


def non_wall_cells(step: PipelineStep, segments: np.ndarray) -> int:
    """Occupied cells farther than ``NON_WALL_DISTANCE_M`` from every wall."""
    grid = step.occupancy
    rows, columns = np.nonzero(grid.data == 100)
    if rows.size == 0:
        return 0
    x = grid.origin[0] + (columns + 0.5) * grid.resolution
    y = grid.origin[1] + (rows + 0.5) * grid.resolution
    distance = np.min(
        np.stack([point_segment_distance(cx, cy, segments) for cx, cy in zip(x, y)]), axis=1
    )
    return int(np.count_nonzero(distance > NON_WALL_DISTANCE_M))


def run_crossing(
    *,
    dynamic_mode: str = "predicted",
    seed: int = 0,
    layout: str = "crossing",
    observation: ObservationModel = ObservationModel(),
    maximum_steps: int = MAXIMUM_FRAMES,
    pipeline: CudaNavPipeline | None = None,
    progress: Callable[[int, Pose2D], None] | None = None,
) -> CrossingResult:
    """One S-course traversal while pedestrians move through the course.

    ``layout`` picks the pedestrians (see :func:`layout_obstacles`).
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if dynamic_mode not in DYNAMIC_MODES:
        raise ValueError(f"dynamic_mode must be one of {DYNAMIC_MODES}")
    if maximum_steps < 10:
        raise ValueError("maximum_steps must be >= 10")
    scan_mode = dynamic_mode == "tracked_scan"
    if pipeline is None:
        nav = CudaNavPipeline(tracker=ScanObjectTracker() if scan_mode else None)
    else:
        nav = pipeline
    if scan_mode and nav.tracker is None:
        raise ValueError("tracked_scan needs a CudaNavPipeline created with a tracker")
    segments = course_segments()
    obstacles, patience_s = layout_obstacles(layout, seed)
    clock = PedestrianClock(obstacles, patience_s)
    path = interpolate_path(PATH_SPACING, reverse=False)
    params = nav.mppi_params
    observation_rng = np.random.default_rng(seed + 1_000_003)
    oracle_mode = dynamic_mode in ("tracked_static", "predicted")
    tracking = {
        "visible_pedestrian_frames": 0,
        "tracked_pedestrian_frames": 0,
        "false_track_frames": 0,
        "position_squared_errors": [],
        "velocity_squared_errors": [],
    }

    truth = Pose2D()
    previous_estimate = (0.0, 0.0)
    command_v = command_w = 0.0
    truth_distance = 0.0
    squared_error = 0.0
    frames = wall_collisions = dynamic_collisions = contact_frames = yield_frames = 0
    invalid_commands = all_colliding = retreating = 0
    retreat_blocked = 0
    minimum_clearance = math.inf
    maximum_ghosts = 0
    ghost_samples: list[int] = []
    frame_ms: list[float] = []
    solve_ms: list[float] = []
    stage_ms: dict[str, list[float]] = {}
    goal_reached = False
    rows: list[tuple[Any, ...]] = []
    tracks: list[np.ndarray] = []
    last_step: PipelineStep | None = None

    wall_start = time.perf_counter()
    for step in range(maximum_steps):
        frame_start = time.perf_counter()
        now = step * DT
        if step > 0:
            clock.advance(truth, DT)
            yield_frames += sum(clock.paused)
        states = clock.states()
        if step > 0:
            candidate = step_unicycle(truth, command_v, command_w, DT)
            # The robot is at fault only when its own motion deepens a contact.
            stay_clearance = dynamic_clearance(truth.x, truth.y, states)
            move_clearance = dynamic_clearance(candidate.x, candidate.y, states)
            if collides(candidate.x, candidate.y, segments):
                wall_collisions += 1
                command_v = command_w = 0.0
            elif move_clearance <= 0.0 and move_clearance < stay_clearance:
                dynamic_collisions += 1
                command_v = command_w = 0.0
            else:
                truth_distance += math.hypot(candidate.x - truth.x, candidate.y - truth.y)
                truth = candidate
        clearance = dynamic_clearance(truth.x, truth.y, states)
        minimum_clearance = min(minimum_clearance, clearance)
        contact_frames += int(clearance <= 0.0)
        tracks.append(states)

        points, on_obstacle = simulate_scan_with_obstacles(truth, segments, states)
        tracked = None
        if oracle_mode:
            points = points[~on_obstacle]
            tracked = observe(tracks, observation, observation_rng)
            if dynamic_mode == "tracked_static":
                tracked[:, 2:4] = 0.0  # tracked positions without velocity prediction
        result = nav.step(points, path=path, dynamic_obstacles=tracked, stamp=now)
        if scan_mode:
            _score_tracks(tracking, result.tracks, states, truth, points, on_obstacle)
            tracked = result.tracks
        last_step = result
        odom = result.odometry
        previous_estimate = (odom.x, odom.y)
        error = math.hypot(odom.x - truth.x, odom.y - truth.y)
        squared_error += error * error
        for name, value in result.timings_ms.items():
            stage_ms.setdefault(name, []).append(value)

        planner = result.planner or {}
        command_v, command_w = result.command if result.command is not None else (0.0, 0.0)
        solve = result.timings_ms.get("mppi", 0.0)
        solve_ms.append(solve)
        all_colliding += int(bool(planner.get("all_colliding", False)))
        retreating += int(bool(planner.get("retreating", False)))
        retreat_blocked += int(bool(planner.get("retreat_blocked", False)))
        valid_ratio = float(planner.get("valid_rollout_ratio", 0.0))
        if (
            not math.isfinite(command_v)
            or not math.isfinite(command_w)
            or abs(command_v) > params.v_max + 1.0e-5
            or abs(command_w) > params.w_max + 1.0e-5
        ):
            invalid_commands += 1
            command_v = command_w = 0.0

        elapsed_frame = (time.perf_counter() - frame_start) * 1000.0
        frame_ms.append(elapsed_frame)
        # Non-wall sampling copies the grid to the host, so it runs after timing.
        if step % NON_WALL_SAMPLE_PERIOD == 0:
            ghosts = non_wall_cells(result, segments)
            ghost_samples.append(ghosts)
            maximum_ghosts = max(maximum_ghosts, ghosts)
        rows.append(
            (
                step,
                now,
                truth.x,
                truth.y,
                truth.yaw,
                odom.x,
                odom.y,
                odom.yaw,
                float(states[0, 0]),
                float(states[0, 1]),
                float(states[1, 0]),
                float(states[1, 1]),
                clearance,
                0 if tracked is None else int(tracked.shape[0]),
                result.occupied_cells,
                valid_ratio,
                int(bool(planner.get("all_colliding", False))),
                int(bool(planner.get("retreating", False))),
                command_v,
                command_w,
                solve,
                elapsed_frame,
            )
        )
        frames += 1
        if progress is not None:
            progress(step, truth)
        goal = WAYPOINTS[-1]
        if math.hypot(truth.x - goal[0], truth.y - goal[1]) <= GOAL_TOLERANCE:
            goal_reached = True
            break

    goal = WAYPOINTS[-1]
    goal_distance = math.hypot(truth.x - goal[0], truth.y - goal[1])
    final_error = math.hypot(previous_estimate[0] - truth.x, previous_estimate[1] - truth.y)
    drift_percent = 100.0 * final_error / truth_distance if truth_distance > 1.0e-6 else 0.0
    deadline_miss_rate = (
        sum(1 for value in frame_ms if value > DEADLINE_MS) / len(frame_ms) if frame_ms else 1.0
    )
    checks = {
        "goal_reached": goal_reached,
        "wall_collision_free": wall_collisions == 0,
        "dynamic_collision_free": dynamic_collisions == 0,
        "frame_bound": frames <= MAXIMUM_FRAMES,
        "odometry_drift": drift_percent < MAXIMUM_DRIFT_PERCENT,
        "deadline_miss_rate": deadline_miss_rate < 0.05,
        "valid_commands": invalid_commands == 0,
    }
    quality_pass = all(checks.values())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scenario": SCENARIO,
        "dynamic_mode": dynamic_mode,
        "layout": layout,
        "pedestrians": len(obstacles),
        "seed": seed,
        "obstacle_phases_s": [obstacle.phase for obstacle in obstacles],
        "observation": {
            "source": (
                "simulator_oracle" if oracle_mode else "scan_tracker" if scan_mode else "none"
            ),
            "latency_frames": observation.latency_frames,
            "position_noise_m": observation.position_noise_m,
            "velocity_noise_mps": observation.velocity_noise_mps,
            "dropout_probability": observation.dropout_probability,
        },
        "runtime": "python",
        "gate": "crossing",
        "success": goal_reached,
        "smoke_pass": quality_pass,
        "quality_pass": quality_pass,
        "checks": checks,
        "frames": frames,
        "simulated_duration_s": frames * DT,
        "time_to_goal_s": frames * DT if goal_reached else None,
        "wall_time_ms": (time.perf_counter() - wall_start) * 1000.0,
        "goal_reached": goal_reached,
        "collision_count": wall_collisions + dynamic_collisions,
        "wall_collisions": wall_collisions,
        "dynamic_collisions": dynamic_collisions,
        "contact_frames": contact_frames,
        # Pedestrian-frames spent paused because the robot was in the way.
        "pedestrian_yield_frames": yield_frames,
        "pedestrian_turnarounds": clock.turnarounds,
        "minimum_dynamic_clearance_m": minimum_clearance,
        "ground_truth_distance_m": truth_distance,
        "ground_truth_goal_distance_m": goal_distance,
        "odometry_ate_rmse_m": math.sqrt(squared_error / frames) if frames else 0.0,
        "odometry_drift_percent": drift_percent,
        "command_deadline_miss_rate": deadline_miss_rate,
        "deadline_ms": DEADLINE_MS,
        "frame_ms_p95": percentile(frame_ms, 0.95),
        "mppi_solve_ms_p95": percentile(solve_ms, 0.95),
        "invalid_commands": invalid_commands,
        "all_colliding_evaluations": all_colliding,
        "retreating_evaluations": retreating,
        "retreat_blocked_evaluations": retreat_blocked,
        "maximum_non_wall_occupied_cells": maximum_ghosts,
        "final_non_wall_occupied_cells": ghost_samples[-1] if ghost_samples else 0,
        "tracking": _tracking_summary(tracking) if scan_mode else None,
        "claims": {
            "gpu_core_closed_loop": True,
            "ros2_runtime": False,
            "real_data": False,
            "perception_tracking": scan_mode,
        },
    }
    stage_latency = {name: latency_summary(values) for name, values in stage_ms.items()}
    stage_latency["frame"] = latency_summary(frame_ms)
    return CrossingResult(
        summary=summary,
        rows=rows,
        last_step=last_step,
        stage_latency_ms=stage_latency,
        obstacle_tracks=np.stack(tracks) if tracks else np.zeros((0, 2, 5), np.float32),
    )
