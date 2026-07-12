#!/usr/bin/env python3
"""Compare exported offline rosbag trajectories and controls without plotting dependencies."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items() if key not in {"frame_id", "child_frame_id"}}
                for row in csv.DictReader(stream)]


def low_speed_episodes(rows: list[dict[str, float]], threshold: float = 0.03,
                       minimum_s: float = 1.0) -> list[tuple[float, float]]:
    episodes = []
    start = None
    for row in rows:
        speed = math.hypot(row["linear_x"], row["linear_y"])
        timestamp = row["recorded_ns"] / 1e9
        if speed < threshold and start is None:
            start = timestamp
        elif speed >= threshold and start is not None:
            if timestamp - start >= minimum_s:
                episodes.append((start, timestamp))
            start = None
    if start is not None:
        end = rows[-1]["recorded_ns"] / 1e9
        if end - start >= minimum_s:
            episodes.append((start, end))
    return episodes


def metrics(directory: Path) -> dict[str, object]:
    summary = json.loads((directory / "motion_summary.json").read_text())
    odometry = read_csv(directory / "odometry.csv")
    commands = read_csv(directory / "cmd_vel.csv")
    episodes = low_speed_episodes(odometry)
    sharp = sum(abs(row["angular_z"]) >= 0.8 for row in commands) / len(commands)
    reverse = sum(row["linear_x"] < -0.01 for row in commands) / len(commands)
    return {
        "bag": directory.name.removesuffix("_motion"), "directory": str(directory),
        **summary, "low_speed_episodes": len(episodes),
        "low_speed_time_s": sum(end - start for start, end in episodes),
        "sharp_turn_command_ratio": sharp, "reverse_command_ratio": reverse,
        "odometry": odometry,
    }


def trajectory_svg(reports: list[dict[str, object]], path: Path) -> None:
    width, panel_w, panel_h, margin = 1200, 220, 260, 30
    height = panel_h + 70
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="30" y="25" font-family="sans-serif" font-size="18" font-weight="bold">ERL navigation trajectories (independent scales)</text>']
    for index, report in enumerate(reports):
        points = [(row["x"], row["y"]) for row in report["odometry"]]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        scale = min((panel_w - 2 * margin) / max(xmax - xmin, 1e-9),
                    (panel_h - 2 * margin) / max(ymax - ymin, 1e-9))
        ox, oy = 10 + index * (panel_w + 15), 45
        projected = [(ox + margin + (x - xmin) * scale,
                      oy + panel_h - margin - (y - ymin) * scale) for x, y in points]
        stride = max(1, len(projected) // 1500)
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in projected[::stride])
        parts += [f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" fill="#f8fafc" stroke="#cbd5e1"/>',
                  f'<polyline points="{coords}" fill="none" stroke="{colors[index % len(colors)]}" stroke-width="1.5"/>',
                  f'<circle cx="{projected[0][0]:.1f}" cy="{projected[0][1]:.1f}" r="4" fill="#16a34a"/>',
                  f'<circle cx="{projected[-1][0]:.1f}" cy="{projected[-1][1]:.1f}" r="4" fill="#dc2626"/>',
                  f'<text x="{ox + 5}" y="{oy + 18}" font-family="sans-serif" font-size="14" font-weight="bold">{html.escape(report["bag"])}</text>',
                  f'<text x="{ox + 5}" y="{oy + panel_h + 20}" font-family="sans-serif" font-size="12">path {report["path_length_m"]:.1f} m · {report["duration_s"]:.1f} s</text>']
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(reports: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["bag", "duration_s", "path_length_m", "net_displacement_m", "mean_speed_mps",
              "max_speed_mps", "low_speed_episodes", "low_speed_time_s",
              "sharp_turn_command_ratio", "reverse_command_ratio"]
    with (output_dir / "motion_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: report[key] for key in fields} for report in reports)
    trajectory_svg(reports, output_dir / "trajectories.svg")
    lines = ["# Offline Navigation Motion Comparison", "",
             "Green dots mark starts; red dots mark ends. Each trajectory panel uses its own scale.", "",
             "![trajectories](trajectories.svg)", "",
             "| Bag | Duration (s) | Path (m) | Mean speed (m/s) | Low-speed episodes | Low-speed time (s) | Sharp-turn commands | Reverse commands |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for report in reports:
        lines.append(f"| {report['bag']} | {report['duration_s']:.1f} | {report['path_length_m']:.1f} | "
                     f"{report['mean_speed_mps']:.3f} | {report['low_speed_episodes']} | "
                     f"{report['low_speed_time_s']:.1f} | {report['sharp_turn_command_ratio']:.1%} | "
                     f"{report['reverse_command_ratio']:.1%} |")
    lines += ["", "Low-speed episodes use observed planar speed below 0.03 m/s for at least 1 second.",
              "Sharp-turn commands use absolute commanded yaw rate at or above 0.8 rad/s."]
    (output_dir / "motion_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("build/rosbag_motion_comparison"))
    args = parser.parse_args()
    reports = [metrics(path) for path in args.motion_dirs]
    write_report(reports, args.output_dir)
    print(f"compared {len(reports)} runs")
    print(f"wrote {args.output_dir / 'motion_comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
