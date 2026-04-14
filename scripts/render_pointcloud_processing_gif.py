#!/usr/bin/env python3
"""Render a rotating CudaPointCloud processing GIF for README and GitHub Pages."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


BG = "#f2ede3"
PAPER = "#fffaf1"
CARD = "#fff8ee"
INK = "#1f1a17"
MUTED = "#6f6259"
ACCENT = "#a1401c"
ACCENT_SOFT = "#ead1c5"
COOL = "#517f5d"
COOL_SOFT = "#d9e5db"
HIGHLIGHT = "#e2b55f"
RAW = "#6f6259"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ply",
        default="build/datasets/room_10000.ply",
        help="ASCII PLY file to visualize",
    )
    parser.add_argument(
        "--out",
        default="gif/pointcloud_processing.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1800,
        help="Display sample size after loading the point cloud",
    )
    parser.add_argument(
        "--normal-count",
        type=int,
        default=80,
        help="Number of normal arrows to draw",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=48,
        help="Number of animation frames",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="GIF frame rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for display subsampling",
    )
    return parser.parse_args()


def load_ascii_ply(path: Path) -> np.ndarray:
    header_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            header_lines += 1
            if line.strip() == "end_header":
                break
    pts = np.loadtxt(path, skiprows=header_lines, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts[None, :]
    return pts[:, :3]


def statistical_filter(points: np.ndarray, k: int = 12, std_mul: float = 1.0) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)
    nearest = np.partition(dist2, kth=k - 1, axis=1)[:, :k]
    mean_dist = np.sqrt(nearest).mean(axis=1)
    thresh = mean_dist.mean() + std_mul * mean_dist.std()
    return mean_dist <= thresh


def fit_plane_ransac(
    points: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 220,
    threshold: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    best_mask = np.zeros(len(points), dtype=bool)
    best_plane = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    for _ in range(iterations):
        ids = rng.choice(len(points), size=3, replace=False)
        p0, p1, p2 = points[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        d = -np.dot(normal, p0)
        dist = np.abs(points @ normal + d)
        mask = dist <= threshold
        if mask.sum() > best_mask.sum():
            best_mask = mask
            best_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)
    return best_plane, best_mask


def estimate_normals(
    points: np.ndarray,
    rng: np.random.Generator,
    count: int,
    k: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    ids = rng.choice(len(points), size=min(count, len(points)), replace=False)
    base = points[ids]
    center = points.mean(axis=0)
    normals = []
    for p in base:
        diff = points - p
        dist2 = np.sum(diff * diff, axis=1)
        nn = np.argpartition(dist2, kth=min(k, len(points) - 1))[:k]
        neighbors = points[nn]
        cov = np.cov((neighbors - neighbors.mean(axis=0)).T)
        vals, vecs = np.linalg.eigh(cov)
        normal = vecs[:, np.argmin(vals)]
        if np.dot(normal, p - center) < 0:
            normal = -normal
        normals.append(normal)
    return base, np.asarray(normals, dtype=np.float32)


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor(PAPER)
    ax.set_title(title, color=INK, fontsize=12, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor(ACCENT_SOFT)
    ax.tick_params(colors=MUTED)
    ax.set_aspect("equal", adjustable="box")


def project_points(points: np.ndarray, azim_deg: float, elev_deg: float = 24.0) -> tuple[np.ndarray, np.ndarray]:
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)

    rz = np.array(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az), np.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(el), -np.sin(el)],
            [0.0, np.sin(el), np.cos(el)],
        ],
        dtype=np.float32,
    )
    rotated = points @ rz.T @ rx.T
    projected = rotated[:, :2]
    depth = rotated[:, 2]
    return projected, depth


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ply_path = Path(args.ply)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    points = load_ascii_ply(ply_path)
    if len(points) > args.sample_size:
        ids = rng.choice(len(points), size=args.sample_size, replace=False)
        points = points[ids]

    raw_points = points
    filtered_mask = statistical_filter(raw_points, k=12, std_mul=1.0)
    filtered_points = raw_points[filtered_mask]
    _, plane_mask = fit_plane_ransac(filtered_points, rng)
    normal_points, normals = estimate_normals(filtered_points, rng, args.normal_count)

    all_points = np.vstack([raw_points, filtered_points, normal_points])

    fig = plt.figure(figsize=(11.2, 8.6), facecolor=BG)
    axes = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
    titles = [
        "Raw room cloud",
        "Statistical filter",
        "Plane extraction",
        "Normal estimation",
    ]
    for ax, title in zip(axes, titles):
        style_axis(ax, title)

    fig.suptitle("CudaPointCloud Processing Snapshot", fontsize=20, color=INK, y=0.975, fontweight="bold")
    fig.text(
        0.5,
        0.935,
        "Same synthetic room scene, shown as raw input, filtered cloud, floor-plane inliers, and PCA normals.",
        ha="center",
        color=MUTED,
        fontsize=11,
    )
    fig.text(
        0.5,
        0.03,
        "Benchmark highlights from this repository: normal estimation 3,171x at 10K points | RANSAC plane 547x at 100K points",
        ha="center",
        color=ACCENT,
        fontsize=10,
    )

    outlier_points = raw_points[~filtered_mask]
    plane_inliers = filtered_points[plane_mask]
    plane_outliers = filtered_points[~plane_mask]
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    radius = float(np.max(maxs - mins) * 0.58)

    def update(frame_idx: int):
        azim = 18 + frame_idx * (360.0 / args.frames)
        all_xy, _ = project_points(all_points, azim)
        center_xy = all_xy.mean(axis=0)
        for ax, title in zip(axes, titles):
            ax.cla()
            style_axis(ax, title)
            ax.set_xlim(center_xy[0] - radius, center_xy[0] + radius)
            ax.set_ylim(center_xy[1] - radius, center_xy[1] + radius)

        raw_xy, raw_depth = project_points(raw_points, azim)
        raw_order = np.argsort(raw_depth)
        axes[0].scatter(raw_xy[raw_order, 0], raw_xy[raw_order, 1], s=4, c=RAW, alpha=0.28)
        if len(outlier_points):
            outlier_xy, outlier_depth = project_points(outlier_points, azim)
            outlier_order = np.argsort(outlier_depth)
            axes[0].scatter(outlier_xy[outlier_order, 0], outlier_xy[outlier_order, 1], s=8, c=ACCENT, alpha=0.65)
        axes[0].text(0.05, 0.92, f"{len(raw_points):,} pts | 5% outliers", transform=axes[0].transAxes, color=MUTED, fontsize=10)

        filtered_xy, filtered_depth = project_points(filtered_points, azim)
        filtered_order = np.argsort(filtered_depth)
        axes[1].scatter(raw_xy[raw_order, 0], raw_xy[raw_order, 1], s=3, c=RAW, alpha=0.10)
        axes[1].scatter(filtered_xy[filtered_order, 0], filtered_xy[filtered_order, 1], s=6, c=COOL, alpha=0.40)
        axes[1].text(0.05, 0.92, f"{len(filtered_points):,} pts kept", transform=axes[1].transAxes, color=MUTED, fontsize=10)

        plane_out_xy, plane_out_depth = project_points(plane_outliers, azim)
        plane_out_order = np.argsort(plane_out_depth)
        plane_in_xy, plane_in_depth = project_points(plane_inliers, azim)
        plane_in_order = np.argsort(plane_in_depth)
        axes[2].scatter(plane_out_xy[plane_out_order, 0], plane_out_xy[plane_out_order, 1], s=4, c=RAW, alpha=0.16)
        axes[2].scatter(plane_in_xy[plane_in_order, 0], plane_in_xy[plane_in_order, 1], s=7, c=HIGHLIGHT, alpha=0.78)
        axes[2].text(
            0.05,
            0.92,
            f"best plane: {plane_inliers.shape[0]:,} inliers",
            transform=axes[2].transAxes,
            color=MUTED,
            fontsize=10,
        )

        axes[3].scatter(filtered_xy[filtered_order, 0], filtered_xy[filtered_order, 1], s=4, c=COOL_SOFT, alpha=0.16)
        normal_xy, normal_depth = project_points(normal_points, azim)
        normal_tip_xy, _ = project_points(normal_points + normals * 0.28, azim)
        normal_order = np.argsort(normal_depth)
        for idx in normal_order:
            start = normal_xy[idx]
            delta = normal_tip_xy[idx] - start
            axes[3].arrow(
                start[0],
                start[1],
                delta[0],
                delta[1],
                color=ACCENT,
                linewidth=0.8,
                alpha=0.92,
                head_width=0.05,
                head_length=0.08,
                length_includes_head=True,
            )
        axes[3].text(
            0.05,
            0.92,
            f"{len(normal_points):,} local PCA normals",
            transform=axes[3].transAxes,
            color=MUTED,
            fontsize=10,
        )

        return axes

    anim = FuncAnimation(fig, update, frames=args.frames, interval=1000 / args.fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved GIF to {out_path}")


if __name__ == "__main__":
    main()
