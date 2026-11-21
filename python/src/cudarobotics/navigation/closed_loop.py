"""Python port of ``tools/cudanav_gpu_closed_loop_s_course.cu``.

The plant integrates the MPPI command, so every command affects the next
synthetic scan (closed loop). All randomness comes from the CUDA MPPI core's
fixed cuRAND seed; the simulator itself is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np

from .evidence import (
    MULTI_TRAVERSAL_GATE,
    SCHEMA_VERSION,
    SMOKE_GATE,
    latency_summary,
    percentile,
)
from .pipeline import CudaNavPipeline, PipelineStep
from .simulation import (
    DT,
    GOAL_TOLERANCE,
    PATH_SPACING,
    WAYPOINTS,
    Pose2D,
    collides,
    course_segments,
    interpolate_path,
    simulate_scan,
    step_unicycle,
    wrap_angle,
)

SCENARIO = "cudanav_s_course"
DEADLINE_MS = 150.0
TRAJECTORY_COLUMNS = (
    "step",
    "traversal",
    "time_s",
    "truth_x",
    "truth_y",
    "truth_yaw",
    "estimated_x",
    "estimated_y",
    "estimated_yaw",
    "error_m",
    "inliers",
    "observed_voxels",
    "occupied_cells",
    "valid_rollout_ratio",
    "all_colliding",
    "retreating",
    "command_v",
    "command_w",
    "solve_ms",
    "frame_ms",
)


@dataclass(frozen=True, eq=False)
class ClosedLoopResult:
    summary: dict[str, Any]
    rows: list[tuple[Any, ...]]
    last_step: PipelineStep | None
    stage_latency_ms: dict[str, dict[str, float]]


def run_s_course(
    *,
    traversals: int = 1,
    maximum_steps: int = 1800,
    minimum_duration_s: float = 0.0,
    pipeline: CudaNavPipeline | None = None,
    progress: Callable[[int, Pose2D], None] | None = None,
) -> ClosedLoopResult:
    """Drive the S-course mission with the integrated GPU CudaNav stack."""
    if traversals < 1 or maximum_steps < 10 or minimum_duration_s < 0.0:
        raise ValueError("invalid closed-loop options")
    nav = CudaNavPipeline() if pipeline is None else pipeline
    segments = course_segments()
    path = interpolate_path(PATH_SPACING, reverse=False)
    params = nav.mppi_params

    truth = Pose2D()
    previous_estimate = (0.0, 0.0)
    command_v = command_w = 0.0
    truth_distance = estimate_distance = command_effect_distance = 0.0
    squared_error = 0.0
    frames = collisions = invalid_commands = all_colliding = retreating = 0
    minimum_inliers = math.inf
    final_observed_voxels = 0
    maximum_occupied_cells = 0
    minimum_nonzero_valid_ratio = 1.0
    frame_ms: list[float] = []
    solve_ms: list[float] = []
    stage_ms: dict[str, list[float]] = {}
    goal_reached = False
    reverse_mode = False
    traversals_completed = 0
    traversal_start_step = 0
    traversal_frames: list[int] = []
    rows: list[tuple[Any, ...]] = []
    last_step: PipelineStep | None = None

    wall_start = time.perf_counter()
    for step in range(maximum_steps):
        frame_start = time.perf_counter()
        if step > 0:
            candidate = step_unicycle(truth, command_v, command_w, DT)
            if collides(candidate.x, candidate.y, segments):
                collisions += 1
                command_v = command_w = 0.0
            else:
                truth_step = math.hypot(candidate.x - truth.x, candidate.y - truth.y)
                truth_distance += truth_step
                if abs(command_v) > 1.0e-3:
                    command_effect_distance += truth_step
                truth = candidate

        points = simulate_scan(truth, segments)
        reverse = traversals_completed % 2 == 1
        result = nav.step(points, path=path, reverse=reverse_mode)
        last_step = result
        odom = result.odometry
        if step > 0:
            minimum_inliers = min(minimum_inliers, odom.inliers)
            estimate_distance += math.hypot(odom.x - previous_estimate[0], odom.y - previous_estimate[1])
        previous_estimate = (odom.x, odom.y)
        error = math.hypot(odom.x - truth.x, odom.y - truth.y)
        squared_error += error * error
        final_observed_voxels = int(result.mapping["observed_voxels"])
        occupied = result.occupied_cells
        maximum_occupied_cells = max(maximum_occupied_cells, occupied)
        for name, value in result.timings_ms.items():
            stage_ms.setdefault(name, []).append(value)

        planner = result.planner or {}
        command_v, command_w = result.command if result.command is not None else (0.0, 0.0)
        solve = result.timings_ms.get("mppi", 0.0)
        solve_ms.append(solve)
        all_colliding += int(bool(planner.get("all_colliding", False)))
        retreating += int(bool(planner.get("retreating", False)))
        valid_ratio = float(planner.get("valid_rollout_ratio", 0.0))
        if valid_ratio > 0.0:
            minimum_nonzero_valid_ratio = min(minimum_nonzero_valid_ratio, valid_ratio)
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
        rows.append(
            (
                step,
                traversals_completed,
                step * DT,
                truth.x,
                truth.y,
                truth.yaw,
                odom.x,
                odom.y,
                odom.yaw,
                error,
                odom.inliers,
                final_observed_voxels,
                occupied,
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

        goal = WAYPOINTS[0] if reverse else WAYPOINTS[-1]
        if math.hypot(truth.x - goal[0], truth.y - goal[1]) <= GOAL_TOLERANCE:
            traversals_completed += 1
            traversal_frames.append(step + 1 - traversal_start_step)
            traversal_start_step = step + 1
            if traversals_completed >= traversals:
                goal_reached = True
                break
            path = interpolate_path(PATH_SPACING, reverse=traversals_completed % 2 == 1)
            initial_tangent = math.atan2(float(path[1, 1] - path[0, 1]), float(path[1, 0] - path[0, 0]))
            reverse_mode = math.cos(wrap_angle(initial_tangent - odom.yaw)) < 0.0
            nav.reset_planner(reverse=reverse_mode)
            command_v = command_w = 0.0

    final_goal = WAYPOINTS[-1] if traversals % 2 == 1 else WAYPOINTS[0]
    goal_distance = math.hypot(truth.x - final_goal[0], truth.y - final_goal[1])
    final_error = math.hypot(previous_estimate[0] - truth.x, previous_estimate[1] - truth.y)
    drift_percent = 100.0 * final_error / truth_distance if truth_distance > 1.0e-6 else 0.0
    ate = math.sqrt(squared_error / frames) if frames else 0.0
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    deadline_miss_rate = (
        sum(1 for value in frame_ms if value > DEADLINE_MS) / len(frame_ms) if frame_ms else 1.0
    )
    causal = command_effect_distance > 5.0 and truth_distance > 5.0
    gate = MULTI_TRAVERSAL_GATE if traversals > 1 else SMOKE_GATE
    minimum_inliers_value = 0 if math.isinf(minimum_inliers) else int(minimum_inliers)
    success = goal_reached and traversals_completed == traversals
    checks = {
        "goal_reached": goal_reached,
        "collision_free": collisions <= gate.maximum_collisions,
        "goal_distance": goal_distance <= gate.maximum_goal_distance_m,
        "odometry_drift": drift_percent < gate.maximum_drift_percent,
        "deadline_miss_rate": deadline_miss_rate < gate.maximum_deadline_miss_rate,
        "causal_command_effect": causal,
        "valid_commands": invalid_commands == 0,
        "mapped_voxels": final_observed_voxels >= 500,
        "occupied_cells": maximum_occupied_cells >= 10,
        "odometry_inliers": minimum_inliers_value >= 30,
        "all_colliding_bound": all_colliding <= 3 * traversals,
        "valid_rollout_ratio": minimum_nonzero_valid_ratio >= 0.001,
        "travel_bound": truth_distance <= 13.0 * traversals,
        "frame_bound": frames <= 400 * traversals,
        "minimum_duration": frames * DT >= minimum_duration_s,
        "traversals_completed": traversals_completed == traversals,
    }
    quality_pass = all(checks.values())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scenario": SCENARIO,
        "stages": [
            "gpu_kiss_icp",
            "gpu_voxel_mapping",
            "gpu_esdf",
            "cuda_mppi",
            "command_driven_plant",
        ],
        "runtime": "python",
        "gate": gate.name,
        "success": success,
        "smoke_pass": quality_pass,
        "quality_pass": quality_pass,
        "checks": checks,
        "frames": frames,
        "simulated_duration_s": frames * DT,
        "traversals_requested": traversals,
        "traversals_completed": traversals_completed,
        "traversal_frames": traversal_frames,
        "wall_time_ms": wall_ms,
        "goal_reached": goal_reached,
        "collision_count": collisions,
        "ground_truth_distance_m": truth_distance,
        "ground_truth_goal_distance_m": goal_distance,
        "odometry_ate_rmse_m": ate,
        "odometry_position_error_m": final_error,
        "odometry_drift_percent": drift_percent,
        "estimated_distance_m": estimate_distance,
        "minimum_inliers": minimum_inliers_value,
        "command_effect_distance_m": command_effect_distance,
        "causal_command_effect": causal,
        "command_deadline_miss_rate": deadline_miss_rate,
        "deadline_ms": DEADLINE_MS,
        "frame_ms_p95": percentile(frame_ms, 0.95),
        "mppi_solve_ms_p95": percentile(solve_ms, 0.95),
        "invalid_commands": invalid_commands,
        "all_colliding_evaluations": all_colliding,
        "retreating_evaluations": retreating,
        "minimum_nonzero_valid_rollout_ratio": minimum_nonzero_valid_ratio,
        "final_observed_voxels": final_observed_voxels,
        "maximum_occupied_cells": maximum_occupied_cells,
        "claims": {
            "gpu_core_closed_loop": True,
            "ros2_runtime": False,
            "real_data": False,
        },
    }
    stage_latency = {name: latency_summary(values) for name, values in stage_ms.items()}
    stage_latency["frame"] = latency_summary(frame_ms)
    return ClosedLoopResult(
        summary=summary,
        rows=rows,
        last_step=last_step,
        stage_latency_ms=stage_latency,
    )


def rows_as_array(rows: list[tuple[Any, ...]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(TRAJECTORY_COLUMNS))
