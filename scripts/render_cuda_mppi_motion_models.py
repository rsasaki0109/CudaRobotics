#!/usr/bin/env python3
"""Render Ackermann/Omni motion-model rollout GIFs from controller_benchmark.

Inputs : <bench_dir>/wall_gap/summary.csv and traj_gpu_*_K8192.csv
Outputs: gif/cuda_mppi_motion_models_wall_gap.gif
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[1]
WALL_X0, WALL_X1 = 4.9, 5.1
GAP_Y0, GAP_Y1 = 4.0, 6.0
START = (1.0, 5.0)
GOAL = (9.0, 5.0)
CONTROL_DT = 0.05
MODELS = (
    ("gpu_mppi_K8192", "DiffDrive"),
    ("gpu_ackermann_K8192", "Ackermann"),
    ("gpu_omni_K8192", "Omni"),
)


def load_summary(bench_dir: Path):
    with open(bench_dir / "wall_gap" / "summary.csv", newline="") as f:
        return {r["label"]: r for r in csv.DictReader(f)}


def load_traj(bench_dir: Path, label: str):
    pts = []
    with open(bench_dir / "wall_gap" / f"traj_{label}.csv", newline="") as f:
        for row in csv.DictReader(f):
            pts.append((float(row["x"]), float(row["y"]), float(row["yaw"])))
    return pts


def draw_scene(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(2, 8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.add_patch(plt.Rectangle((WALL_X0, 2), WALL_X1 - WALL_X0, GAP_Y0 - 2,
                               color="#333333"))
    ax.add_patch(plt.Rectangle((WALL_X0, GAP_Y1), WALL_X1 - WALL_X0, 8 - GAP_Y1,
                               color="#333333"))
    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]], "--", color="#aaaaaa", lw=1)
    ax.plot(*GOAL, marker="*", color="#e6b800", markersize=12)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    bench_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/mppi_motion_bench")
    rows = load_summary(bench_dir)
    trajs = [load_traj(bench_dir, label) for label, _ in MODELS]
    n = max(len(t) for t in trajs)
    stride = 4

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    artists = []
    for ax, (label, model), traj in zip(axes, MODELS, trajs):
        row = rows[label]
        title = (f"{model}  K={int(row['batch_size']):,}\n"
                 f"{float(row['sim_s']):.1f}s to goal, "
                 f"{float(row['mean_ms']):.2f} ms/cycle")
        draw_scene(ax, title)
        trail, = ax.plot([], [], "-", color="#76B900", lw=2)
        body, = ax.plot([], [], "o", color="#76B900", markersize=7)
        heading, = ax.plot([], [], "-", color="black", lw=1.5)
        clock = ax.text(0.3, 7.4, "", fontsize=8)
        artists.append((traj, trail, body, heading, clock))

    fig.suptitle(
        "cuda_mppi_controller motion models — closed-loop wall-gap @ 20 Hz",
        fontsize=10)
    fig.tight_layout()

    def update(frame):
        i = min(frame * stride, n - 1)
        out = []
        for traj, trail, body, heading, clock in artists:
            j = min(i, len(traj) - 1)
            xs = [p[0] for p in traj[:j + 1]]
            ys = [p[1] for p in traj[:j + 1]]
            trail.set_data(xs, ys)
            x, y, yaw = traj[j]
            body.set_data([x], [y])
            heading.set_data([x, x + 0.35 * math.cos(yaw)],
                             [y, y + 0.35 * math.sin(yaw)])
            clock.set_text(f"t = {j * CONTROL_DT:4.1f} s")
            out += [trail, body, heading, clock]
        return out

    frames = (n + stride - 1) // stride
    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    out = REPO / "gif" / "cuda_mppi_motion_models_wall_gap.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"wrote {out} ({frames} frames)")


if __name__ == "__main__":
    main()
