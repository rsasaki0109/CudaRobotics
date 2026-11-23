#!/usr/bin/env python3
"""Attribute every missed goal and robot-caused collision of the layout sweep.

The closed loop is deterministic, so each run of
``scripts/render_cudanav_pedestrian_layouts.py``'s sweep can be replayed
exactly. This replays them with per-frame instrumentation and labels:

- every *stalled* frame (route progress below ``STALL_PROGRESS_M`` over the
  next ``STALL_WINDOW_S``) by its likely cause. With a pedestrian within
  ``NEAR_PEDESTRIAN_M`` of the robot's edge: ``wait`` (the planner still has
  collision-free rollouts), ``pushed_back`` (every rollout collides and the
  pedestrian advances at least ``ONCOMING_M`` against the robot's heading
  over the window) or ``backout`` (every rollout collides otherwise).
  Without one: ``ghost`` (non-wall occupied cells within ``GHOST_RADIUS_M``),
  ``false_track`` (a planner obstacle no pedestrian explains within
  ``GHOST_RADIUS_M``), or ``other``;
- every robot-caused collision by the robot's motion (forward or back-out),
  where the pedestrian was relative to the heading, and how far the
  planner's obstacle input was from the true pedestrian.

Usage (needs the CUDA extension)::

    python scripts/analyze_cudanav_layout_failures.py --output-json out.json --output-md out.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from cudarobotics.navigation import dynamic_scenario as ds
from cudarobotics.navigation.pipeline import CudaNavPipeline
from cudarobotics.navigation.simulation import (
    DT,
    PATH_SPACING,
    ROBOT_RADIUS,
    Pose2D,
    collides,
    course_segments,
    dynamic_clearance,
    interpolate_path,
    point_segment_distance,
    step_unicycle,
)
from cudarobotics.navigation.tracking import ScanObjectTracker

CONDITIONS = {
    "predicted_degraded": ("predicted", ds.ObservationModel(latency_frames=2, position_noise_m=0.05, velocity_noise_mps=0.1)),
    "predicted": ("predicted", ds.ObservationModel()),
    "tracked_scan": ("tracked_scan", ds.ObservationModel()),
}
LAYOUTS = ("crossing", "head_on", "overtake", "crowd")
SEEDS = tuple(range(6))
STALL_WINDOW_S = 2.0
STALL_PROGRESS_M = 0.10
NEAR_PEDESTRIAN_M = 0.6
GHOST_RADIUS_M = 1.0
WALL_SPARE_M = 0.15  # same as NON_WALL_DISTANCE_M in the scenario
ONCOMING_M = 0.10  # pedestrian displacement against the robot's heading over the window
STALL_CAUSES = ("pushed_back", "backout", "wait", "ghost", "false_track", "other")


class RecordingPipeline:
    """Forwards to a CudaNavPipeline and keeps what the analysis needs."""

    def __init__(self, nav: CudaNavPipeline, segments: np.ndarray) -> None:
        self.nav = nav
        self.segments = segments
        self.frames: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.nav, name)

    def step(self, points, *, path, dynamic_obstacles=None, stamp=None):
        result = self.nav.step(points, path=path, dynamic_obstacles=dynamic_obstacles, stamp=stamp)
        planner_input = dynamic_obstacles
        if planner_input is None and result.tracks is not None:
            planner_input = result.tracks
        odom = result.odometry
        self.frames.append(
            {
                "planner": dict(result.planner or {}),
                "obstacles": None if planner_input is None else np.asarray(planner_input, np.float32).copy(),
                "ghosts": self._ghosts_near(result, odom.x, odom.y),
            }
        )
        return result

    def _ghosts_near(self, result, x: float, y: float) -> int:
        grid = result.occupancy
        rows, columns = np.nonzero(grid.data == 100)
        if rows.size == 0:
            return 0
        cx = grid.origin[0] + (columns + 0.5) * grid.resolution
        cy = grid.origin[1] + (rows + 0.5) * grid.resolution
        near = np.hypot(cx - x, cy - y) <= GHOST_RADIUS_M
        if not np.any(near):
            return 0
        cx, cy = cx[near], cy[near]
        wall = np.min(np.stack([point_segment_distance(a, b, self.segments) for a, b in zip(cx, cy)]), axis=1)
        return int(np.count_nonzero(wall > WALL_SPARE_M))


def route_progress(path: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Arc length of the nearest path point, made monotone (best progress so far)."""
    distance = np.hypot(x[:, None] - path[None, :, 0], y[:, None] - path[None, :, 1])
    return np.maximum.accumulate(np.argmin(distance, axis=1) * PATH_SPACING)


def replay(condition: str, layout: str, seed: int) -> dict[str, Any]:
    mode, observation = CONDITIONS[condition]
    segments = course_segments()
    nav = CudaNavPipeline(tracker=ScanObjectTracker() if mode == "tracked_scan" else None)
    recorder = RecordingPipeline(nav, segments)
    result = ds.run_crossing(dynamic_mode=mode, seed=seed, layout=layout, observation=observation, pipeline=recorder)
    rows = np.asarray([row[:22] for row in result.rows], dtype=np.float64)
    states = result.obstacle_tracks  # (frames, N, 5)
    frames = recorder.frames
    count = rows.shape[0]
    x, y, yaw = rows[:, 2], rows[:, 3], rows[:, 4]
    command_v, command_w = rows[:, 18], rows[:, 19]
    path = interpolate_path(PATH_SPACING, reverse=False)
    progress = route_progress(path, x, y)

    # Per-frame stall labels.
    window = int(round(STALL_WINDOW_S / DT))
    stall_time = {cause: 0.0 for cause in STALL_CAUSES}
    stall_labels: list[str | None] = []
    for k in range(count):
        ahead = min(count - 1, k + window)
        if result.summary["goal_reached"] and ahead == count - 1 and k + window >= count:
            stall_labels.append(None)  # the run ended at the goal inside the window
            continue
        if progress[ahead] - progress[k] >= STALL_PROGRESS_M:
            stall_labels.append(None)
            continue
        peds = states[k]
        edge = np.hypot(peds[:, 0] - x[k], peds[:, 1] - y[k]) - peds[:, 4] - ROBOT_RADIUS
        if np.any(edge <= NEAR_PEDESTRIAN_M):
            index = int(np.argmin(edge))
            if not frames[k]["planner"].get("all_colliding", False):
                label = "wait"
            else:
                moved = states[ahead][index, :2] - peds[index, :2]
                against = -(moved[0] * math.cos(yaw[k]) + moved[1] * math.sin(yaw[k]))
                label = "pushed_back" if against >= ONCOMING_M else "backout"
        elif frames[k]["ghosts"] > 0:
            label = "ghost"
        else:
            obstacles = frames[k]["obstacles"]
            label = "other"
            if obstacles is not None and obstacles.shape[0]:
                near = np.hypot(obstacles[:, 0] - x[k], obstacles[:, 1] - y[k]) <= GHOST_RADIUS_M
                for ox, oy in obstacles[near, :2]:
                    if np.min(np.hypot(peds[:, 0] - ox, peds[:, 1] - oy)) > ds.FALSE_TRACK_M:
                        label = "false_track"
                        break
        stall_labels.append(label)
        stall_time[label] += DT

    # Robot-caused collisions, replayed with the scenario's rule.
    collisions = []
    for k in range(1, count):
        pose = Pose2D(x[k - 1], y[k - 1], yaw[k - 1])
        candidate = step_unicycle(pose, command_v[k - 1], command_w[k - 1], DT)
        peds = states[k]
        kind = None
        if collides(candidate.x, candidate.y, segments):
            kind = "wall"
        else:
            stay = dynamic_clearance(pose.x, pose.y, peds)
            move = dynamic_clearance(candidate.x, candidate.y, peds)
            if move <= 0.0 and move < stay:
                kind = "pedestrian"
        if kind is None:
            continue
        event: dict[str, Any] = {
            "frame": k,
            "time_s": round(k * DT, 2),
            "kind": kind,
            "command_v": round(float(command_v[k - 1]), 3),
            "motion": "back-out" if command_v[k - 1] < -1e-3 else ("forward" if command_v[k - 1] > 1e-3 else "turn"),
            "retreating": bool(frames[k - 1]["planner"].get("retreating", False)),
            "all_colliding": bool(frames[k - 1]["planner"].get("all_colliding", False)),
        }
        if kind == "pedestrian":
            gap = np.hypot(peds[:, 0] - candidate.x, peds[:, 1] - candidate.y) - peds[:, 4] - ROBOT_RADIUS
            index = int(np.argmin(gap))
            px, py, pvx, pvy, _r = peds[index]
            bearing = math.degrees(math.atan2(py - pose.y, px - pose.x) - pose.yaw)
            bearing = (bearing + 180.0) % 360.0 - 180.0
            obstacles = frames[k - 1]["obstacles"]
            input_error = None
            if obstacles is not None and obstacles.shape[0]:
                truth_prev = states[k - 1][index]
                input_error = float(np.min(np.hypot(obstacles[:, 0] - truth_prev[0], obstacles[:, 1] - truth_prev[1])))
            event.update(
                {
                    "pedestrian": index,
                    "bearing_deg": round(bearing, 1),
                    "side": "front" if abs(bearing) <= 60 else ("rear" if abs(bearing) >= 120 else "side"),
                    "pedestrian_speed_mps": round(float(math.hypot(pvx, pvy)), 3),
                    "planner_input_error_m": None if input_error is None else round(input_error, 3),
                }
            )
        collisions.append(event)

    # Where the run ended up stuck: the cause of the longest final stall.
    final_cause = None
    if not result.summary["goal_reached"]:
        tail = [label for label in stall_labels[-window * 5 :] if label is not None]
        if tail:
            final_cause = max(STALL_CAUSES, key=tail.count)
    return {
        "condition": condition,
        "layout": layout,
        "seed": seed,
        "goal": bool(result.summary["goal_reached"]),
        "time_to_goal_s": result.summary["time_to_goal_s"],
        "route_progress_m": round(float(progress[-1]), 2),
        "route_length_m": round(float((path.shape[0] - 1) * PATH_SPACING), 2),
        "summary_collisions": int(result.summary["wall_collisions"] + result.summary["dynamic_collisions"]),
        "stall_s": {cause: round(value, 1) for cause, value in stall_time.items()},
        "reverse_distance_m": round(float(np.sum(np.clip(-command_v, 0.0, None)) * DT), 2),
        "final_stall_cause": final_cause,
        "pedestrian_turnarounds": int(result.summary["pedestrian_turnarounds"]),
        "collisions": collisions,
    }


def render(runs: list[dict[str, Any]]) -> str:
    lines = [
        "# CudaNav pedestrian layouts: failure attribution",
        "",
        "Generated by `scripts/analyze_cudanav_layout_failures.py`, which replays the deterministic",
        f"72-run layout sweep ({len(runs)} runs) with per-frame instrumentation. Synthetic data.",
        "",
        f"A frame is stalled when route progress over the next {STALL_WINDOW_S:.0f} s is below {STALL_PROGRESS_M:.2f} m.",
        f"With a pedestrian within {NEAR_PEDESTRIAN_M} m of the robot's edge its cause is `wait` if the planner still has",
        f"collision-free rollouts, `pushed_back` if every rollout collides and the pedestrian advances {ONCOMING_M:.2f} m or more",
        "against the robot's heading over the window, and `backout` if every rollout collides otherwise. Without one it is",
        f"`ghost` if non-wall occupied cells lie within {GHOST_RADIUS_M} m, `false_track` if a planner obstacle",
        f"within {GHOST_RADIUS_M} m is more than {ds.FALSE_TRACK_M} m from every pedestrian, else `other`.",
        "",
        "## Stalled time by cause (seconds, summed over 6 seeds)",
        "",
        "| Layout | Condition | Goal | " + " | ".join(STALL_CAUSES) + " | Reversed |",
        "|---|---|---|" + "---|" * (len(STALL_CAUSES) + 1),
    ]
    for layout in LAYOUTS:
        for condition in CONDITIONS:
            group = [r for r in runs if r["layout"] == layout and r["condition"] == condition]
            if not group:
                continue
            totals = {c: sum(r["stall_s"][c] for r in group) for c in STALL_CAUSES}
            goals = sum(r["goal"] for r in group)
            lines.append(
                f"| {layout} | {condition} | {goals}/{len(group)} | "
                + " | ".join(f"{totals[c]:.1f}" for c in STALL_CAUSES)
                + f" | {sum(r['reverse_distance_m'] for r in group):.1f} m |"
            )
    lines += ["", "## Missed goals", "", "| Layout | Condition | Seed | Progress | Final stall cause | Stalled s (" + " / ".join(STALL_CAUSES) + ") | Reversed | Turnarounds |", "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        if r["goal"]:
            continue
        s = r["stall_s"]
        lines.append(
            f"| {r['layout']} | {r['condition']} | {r['seed']} | {r['route_progress_m']:.1f} / {r['route_length_m']:.1f} m | "
            f"{r['final_stall_cause']} | " + " / ".join(f"{s[c]:.1f}" for c in STALL_CAUSES) + f" | {r['reverse_distance_m']:.1f} m | {r['pedestrian_turnarounds']} |"
        )
    lines += ["", "## Robot-caused collisions", "", "| Layout | Condition | Seed | t (s) | Kind | Motion | Retreating | All colliding | Pedestrian side | Pedestrian speed | Planner input error |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        for e in r["collisions"]:
            error = e.get("planner_input_error_m")
            lines.append(
                f"| {r['layout']} | {r['condition']} | {r['seed']} | {e['time_s']:.1f} | {e['kind']} | {e['motion']} ({e['command_v']:+.2f} m/s) | "
                f"{'yes' if e['retreating'] else 'no'} | {'yes' if e['all_colliding'] else 'no'} | {e.get('side', '-')} ({e.get('bearing_deg', '-')}°) | "
                f"{e.get('pedestrian_speed_mps', '-')} | {'-' if error is None else f'{error:.2f} m'} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layouts", nargs="+", default=list(LAYOUTS), choices=LAYOUTS)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)
    runs = []
    for layout in args.layouts:
        for condition in args.conditions:
            for seed in args.seeds:
                run = replay(condition, layout, seed)
                if run["summary_collisions"] != len(run["collisions"]):
                    print(f"collision replay mismatch: {layout} {condition} {seed}", file=sys.stderr)
                    return 1
                print(f"{layout:9s} {condition:19s} seed {seed}: goal={run['goal']} collisions={len(run['collisions'])} reversed={run['reverse_distance_m']} stall={run['stall_s']}", flush=True)
                runs.append(run)
    args.output_json.write_text(json.dumps({"schema_version": 1, "runs": runs}, indent=2) + "\n", encoding="utf-8")
    if args.output_md is not None:
        args.output_md.write_text(render(runs), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
