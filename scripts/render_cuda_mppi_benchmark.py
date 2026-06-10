#!/usr/bin/env python3
"""Render the nav2 CPU-vs-GPU MPPI controller benchmark artifacts.

Inputs : <bench_dir>/summary.csv and traj_<label>.csv from
         `ros2 run cuda_mppi_controller controller_benchmark <bench_dir>`
Outputs: docs/results/cuda_mppi_vs_nav2_<date>.svg  (solve-time chart)
         gif/cuda_mppi_vs_nav2_cpu.gif              (side-by-side rollout)
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[1]

# scenario constants, must match controller_benchmark.cpp
WALL_X0, WALL_X1 = 4.9, 5.1
GAP_Y0, GAP_Y1 = 4.0, 6.0
START = (1.0, 5.0)
GOAL = (9.0, 5.0)
CONTROL_DT = 0.05


def load_summary(bench_dir: Path):
    with open(bench_dir / "summary.csv", newline="") as f:
        return list(csv.DictReader(f))


def load_traj(bench_dir: Path, label: str):
    pts = []
    with open(bench_dir / f"traj_{label}.csv", newline="") as f:
        for row in csv.DictReader(f):
            pts.append((float(row["x"]), float(row["y"]), float(row["yaw"])))
    return pts


def render_chart(rows, date_tag: str) -> Path:
    cpu = [r for r in rows if r["label"].startswith("cpu")]
    gpu = [r for r in rows if r["label"].startswith("gpu")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.6

    def bars(entries, offset, color, name):
        xs = np.arange(len(entries)) + offset
        means = [float(r["mean_ms"]) for r in entries]
        p95 = [float(r["p95_ms"]) for r in entries]
        ax.bar(xs, means, width, color=color, label=f"{name} mean")
        ax.plot(xs, p95, "k_", markersize=18, label=f"{name} p95" if offset == 0 else None)
        for x, m, r in zip(xs, means, entries):
            ax.text(x, m + 0.3, f"K={int(r['batch_size']):,}\n{m:.1f} ms",
                    ha="center", fontsize=8)
        return xs

    xs_cpu = bars(cpu, 0, "#d95f5f", "CPU")
    xs_gpu = bars(gpu, len(cpu) + 0.6, "#76B900", "GPU")
    ax.axhline(50.0, color="gray", ls="--", lw=1)
    ax.text(0.0, 50.8, "20 Hz control budget (50 ms)", fontsize=8, color="gray")
    ax.set_xticks(list(xs_cpu) + list(xs_gpu))
    ax.set_xticklabels(
        [f"CPU\nK={int(r['batch_size']):,}" for r in cpu]
        + [f"GPU\nK={int(r['batch_size']):,}" for r in gpu], fontsize=8)
    ax.set_ylabel("solve time per control cycle [ms]")
    ax.set_title(
        "nav2_mppi_controller (CPU, benchmark CPU) vs cuda_mppi_controller "
        "(GPU, benchmark GPU)\nclosed-loop wall-gap scenario, T=56, dt=0.05, "
        "same costmap & plan, p95 shown as ticks")
    ax.set_ylim(0, 56)
    out = REPO / "docs" / "results" / f"cuda_mppi_vs_nav2_{date_tag}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=120)
    plt.close(fig)
    return out


def draw_scene(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(2, 8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.add_patch(plt.Rectangle((WALL_X0, 2), WALL_X1 - WALL_X0, GAP_Y0 - 2,
                               color="#333333"))
    ax.add_patch(plt.Rectangle((WALL_X0, GAP_Y1), WALL_X1 - WALL_X0, 8 - GAP_Y1,
                               color="#333333"))
    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]], "--", color="#aaaaaa", lw=1)
    ax.plot(*GOAL, marker="*", color="#e6b800", markersize=15)
    ax.set_xticks([])
    ax.set_yticks([])


def render_gif(bench_dir: Path, cpu_label: str, gpu_label: str,
               cpu_row, gpu_row) -> Path:
    cpu_traj = load_traj(bench_dir, cpu_label)
    gpu_traj = load_traj(bench_dir, gpu_label)
    n = max(len(cpu_traj), len(gpu_traj))
    stride = 4  # 20 Hz sim -> 5 fps worth of frames, played back at 10 fps

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    cpu_title = (f"nav2 MPPI (CPU)  K={int(cpu_row['batch_size']):,}  "
                 f"{float(cpu_row['mean_ms']):.1f} ms/cycle")
    gpu_title = (f"CUDA MPPI (GPU)  K={int(gpu_row['batch_size']):,}  "
                 f"{float(gpu_row['mean_ms']):.1f} ms/cycle")
    artists = []
    for ax, title, traj, color in (
            (axes[0], cpu_title, cpu_traj, "#d95f5f"),
            (axes[1], gpu_title, gpu_traj, "#76B900")):
        draw_scene(ax, title)
        trail, = ax.plot([], [], "-", color=color, lw=2)
        body, = ax.plot([], [], "o", color=color, markersize=8)
        heading, = ax.plot([], [], "-", color="black", lw=1.5)
        clock = ax.text(0.3, 7.4, "", fontsize=9)
        artists.append((traj, trail, body, heading, clock))
    fig.suptitle("Same scenario, same costmap, same global plan — 20 Hz closed loop",
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
    out = REPO / "gif" / "cuda_mppi_vs_nav2_cpu.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    return out


def main():
    bench_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/mppi_bench")
    date_tag = sys.argv[2] if len(sys.argv) > 2 else "latest"
    rows = load_summary(bench_dir)
    by_label = {r["label"]: r for r in rows}
    chart = render_chart(rows, date_tag)
    print(f"wrote {chart}")
    gif = render_gif(bench_dir, "cpu_mppi_K2000", "gpu_mppi_K16384",
                     by_label["cpu_mppi_K2000"], by_label["gpu_mppi_K16384"])
    print(f"wrote {gif}")


if __name__ == "__main__":
    main()
