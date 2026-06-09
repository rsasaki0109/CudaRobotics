#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

WORKSPACE = 50.0
DT = 0.05
PANEL = 420

SCENES = {
    "dynamic_crossing": {
        "start": (4.0, 6.0),
        "goal": (46.0, 44.0),
        "static_obs": [
            (16.0, 16.0, 2.8),
            (16.0, 34.0, 2.8),
            (34.0, 14.0, 2.6),
            (34.0, 36.0, 2.6),
        ],
        "dynamic_obs": [(11.0, 24.0, 0.0, 2.4, 2.4)],
    }
}

PLANNER_STYLES = {
    "mppi": {"label": "vanilla MPPI", "color": (40, 40, 220), "trail": (180, 180, 255)},
    "step_mppi_smooth": {"label": "step_mppi_smooth", "color": (24, 143, 90), "trail": (170, 230, 200)},
    "tsallis_mppi_smooth": {"label": "tsallis_mppi_smooth", "color": (143, 90, 24), "trail": (230, 210, 170)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a side-by-side MPPI zoo comparison GIF from benchmark_diff_mppi trajectories."
    )
    parser.add_argument("--bin", default="bin/benchmark_diff_mppi", help="Path to benchmark_diff_mppi binary.")
    parser.add_argument("--scenario", default="dynamic_crossing", choices=sorted(SCENES))
    parser.add_argument("--planners", default="mppi,step_mppi_smooth", help="Comma-separated planner names.")
    parser.add_argument("--k-values", default="128", help="Comma-separated K sample counts.")
    parser.add_argument("--seed-count", default="1", help="Number of seeds per cell.")
    parser.add_argument("--trajectory-csv", default="build/mppi_zoo_trajectory.csv", help="Trajectory CSV path.")
    parser.add_argument("--gif-out", default="gif/gpu_mppi_zoo_dynamic_crossing.gif", help="Output GIF path.")
    parser.add_argument("--fps", type=int, default=12, help="Output GIF frame rate.")
    parser.add_argument("--skip-run", action="store_true", help="Render from an existing trajectory CSV.")
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def world_to_pixel(x: float, y: float, size: int) -> tuple[int, int]:
  px = int(x / WORKSPACE * size)
  py = int((WORKSPACE - y) / WORKSPACE * size)
  return px, py


def run_benchmark(args: argparse.Namespace, trajectory_csv: Path) -> None:
    command = [
        str(repo_path(args.bin)),
        "--quick",
        "--scenarios",
        args.scenario,
        "--planners",
        args.planners,
        "--k-values",
        args.k_values,
        "--seed-count",
        str(args.seed_count),
        "--csv",
        str(trajectory_csv.with_suffix(".metrics.csv")),
        "--trajectory-csv",
        str(trajectory_csv),
    ]
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def load_trajectories(path: Path) -> dict[str, list[dict[str, float]]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["planner"]].append(
                {
                    "step": int(float(row["episode_step"])),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "goal_distance": float(row["goal_distance"]),
                }
            )
    for planner in grouped:
        grouped[planner].sort(key=lambda item: item["step"])
    return grouped


def dynamic_obstacle_positions(scene: dict, step: int) -> list[tuple[float, float, float]]:
    tau = step * DT
    positions = []
    for x0, y0, vx, vy, radius in scene["dynamic_obs"]:
        positions.append((x0 + vx * tau, y0 + vy * tau, radius))
    return positions


def draw_panel(
    image: np.ndarray,
    scene: dict,
    trajectory: list[dict[str, float]],
    step: int,
    title: str,
    style: dict,
) -> None:
    image[:] = (248, 248, 248)
    for x, y, radius in scene["static_obs"]:
        center = world_to_pixel(x, y, PANEL)
        cv2.circle(image, center, max(2, int(radius / WORKSPACE * PANEL)), (70, 70, 70), -1)

    for x, y, radius in dynamic_obstacle_positions(scene, step):
        center = world_to_pixel(x, y, PANEL)
        cv2.circle(image, center, max(2, int(radius / WORKSPACE * PANEL)), (40, 120, 200), -1)

    goal = world_to_pixel(scene["goal"][0], scene["goal"][1], PANEL)
    start = world_to_pixel(scene["start"][0], scene["start"][1], PANEL)
    cv2.circle(image, goal, 8, (40, 170, 70), -1)
    cv2.circle(image, start, 6, (40, 40, 40), -1)

    points = []
    for sample in trajectory:
        if sample["step"] > step:
            break
        points.append(world_to_pixel(sample["x"], sample["y"], PANEL))
    if len(points) >= 2:
        cv2.polylines(image, [np.array(points, dtype=np.int32)], False, style["trail"], 2, cv2.LINE_AA)
    if points:
        cv2.circle(image, points[-1], 7, style["color"], -1)

    current = next((sample for sample in trajectory if sample["step"] == step), trajectory[-1])
    status = "success" if current["goal_distance"] < 2.0 else f"dist={current['goal_distance']:.2f}m"
    cv2.putText(image, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(image, f"step={step} {status}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA)


def render_gif(
    trajectories: dict[str, list[dict[str, float]]],
    planners: list[str],
    scene: dict,
    gif_out: Path,
    fps: int,
) -> None:
    max_step = 0
    for planner in planners:
        if planner in trajectories:
            max_step = max(max_step, trajectories[planner][-1]["step"])

    gif_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mppi-zoo-gif-") as tmp_dir:
        avi_path = Path(tmp_dir) / "frames.avi"
        writer = cv2.VideoWriter(
            str(avi_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (PANEL * len(planners), PANEL),
        )
        if not writer.isOpened():
            raise SystemExit(f"Failed to open video writer for {avi_path}")

        for step in range(max_step + 1):
            panels = []
            for planner in planners:
                panel = np.zeros((PANEL, PANEL, 3), dtype=np.uint8)
                style = PLANNER_STYLES.get(
                    planner,
                    {"label": planner, "color": (120, 120, 120), "trail": (200, 200, 200)},
                )
                draw_panel(panel, scene, trajectories[planner], step, style["label"], style)
                panels.append(panel)
            frame = np.concatenate(panels, axis=1)
            writer.write(frame)
        writer.release()

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(avi_path),
            "-vf",
            (
                f"fps={fps},scale={PANEL * len(planners)}:-1:flags=lanczos,"
                "split[a][b];[a]palettegen=stats_mode=diff[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
            ),
            str(gif_out),
        ]
        subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.scenario not in SCENES:
        raise SystemExit(f"Unsupported scenario for GIF rendering: {args.scenario}")

    trajectory_csv = repo_path(args.trajectory_csv)
    gif_out = repo_path(args.gif_out)
    planners = [planner.strip() for planner in args.planners.split(",") if planner.strip()]

    if not args.skip_run:
        trajectory_csv.parent.mkdir(parents=True, exist_ok=True)
        run_benchmark(args, trajectory_csv)
    if not trajectory_csv.exists():
        raise SystemExit(f"Trajectory CSV not found: {trajectory_csv}")

    trajectories = load_trajectories(trajectory_csv)
    missing = [planner for planner in planners if planner not in trajectories]
    if missing:
        raise SystemExit(f"Trajectory CSV is missing planners: {', '.join(missing)}")

    render_gif(trajectories, planners, SCENES[args.scenario], gif_out, args.fps)
    print(f"GIF saved to {gif_out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
