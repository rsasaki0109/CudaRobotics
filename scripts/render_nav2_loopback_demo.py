#!/usr/bin/env python3
"""Render the nav2 loopback demo recording into a GIF.

Usage: render_nav2_loopback_demo.py [/tmp/nav2_demo]
Reads robot_path.csv plus the tb3_sandbox map, writes
gif/cuda_mppi_nav2_loopback.gif.
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
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
MAP_PGM = "/opt/ros/jazzy/share/nav2_bringup/maps/tb3_sandbox.pgm"
MAP_RES = 0.05
MAP_ORIGIN = (-10.0, -10.0)
START = (-2.0, -0.5)
WAYPOINTS = [(1.8, 1.4), (-0.3, -1.8)]


def main():
    demo_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/nav2_demo")
    trail = []
    with open(demo_dir / "robot_path.csv", newline="") as f:
        for row in csv.DictReader(f):
            trail.append((float(row["x"]), float(row["y"]), float(row["yaw"])))

    img = np.array(Image.open(MAP_PGM))
    h, w = img.shape
    extent = [MAP_ORIGIN[0], MAP_ORIGIN[0] + w * MAP_RES,
              MAP_ORIGIN[1], MAP_ORIGIN[1] + h * MAP_RES]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray", origin="lower", extent=extent,
              vmin=0, vmax=255)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "cuda_mppi_controller driving Nav2\n"
        "loopback sim, GPU MPPI K=8,192 @ 20 Hz", fontsize=11)
    ax.plot(*START, "s", color="#3366cc", markersize=8)
    for i, wp in enumerate(WAYPOINTS):
        ax.plot(*wp, marker="*", color="#e6b800", markersize=16)
        ax.annotate(f"goal {i+1}", wp, textcoords="offset points",
                    xytext=(8, 6), fontsize=9)

    trail_line, = ax.plot([], [], "-", color="#76B900", lw=2.5)
    body, = ax.plot([], [], "o", color="#76B900", markersize=9)
    heading, = ax.plot([], [], "-", color="black", lw=1.5)

    stride = 8  # 10 Hz samples -> ~1.25 fps of sim time per frame at 10 fps
    frames = (len(trail) + stride - 1) // stride

    def update(frame):
        j = min(frame * stride, len(trail) - 1)
        xs = [p[0] for p in trail[:j + 1]]
        ys = [p[1] for p in trail[:j + 1]]
        trail_line.set_data(xs, ys)
        x, y, yaw = trail[j]
        body.set_data([x], [y])
        heading.set_data([x, x + 0.3 * math.cos(yaw)],
                         [y, y + 0.3 * math.sin(yaw)])
        return trail_line, body, heading

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = REPO / "gif" / "cuda_mppi_nav2_loopback.gif"
    fig.tight_layout()
    anim.save(str(out), writer=PillowWriter(fps=10))
    print(f"wrote {out} ({frames} frames, {len(trail)} poses)")


if __name__ == "__main__":
    main()
