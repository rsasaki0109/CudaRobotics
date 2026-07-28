#!/usr/bin/env python3
"""Render retained CudaNav truth/odometry trajectory evidence as a GIF."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "cuda_nav_bringup"))

from cuda_nav_bringup.simulation_geometry import default_segments  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=120)
    return parser.parse_args()


def load_trajectory(path: Path) -> list[dict[str, float | None]]:
    rows = []
    previous_elapsed = -math.inf
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | None] = {}
            for field in (
                "elapsed_sec",
                "truth_x",
                "truth_y",
                "odom_x",
                "odom_y",
            ):
                value = raw.get(field, "")
                row[field] = float(value) if value not in ("", None) else None
            if not all(
                math.isfinite(value)
                for value in row.values()
                if value is not None
            ):
                raise ValueError("trajectory contains non-finite values")
            if row["elapsed_sec"] is None:
                raise ValueError("trajectory elapsed_sec is required")
            if row["truth_x"] is None or row["truth_y"] is None:
                raise ValueError("trajectory truth coordinates are required")
            if float(row["elapsed_sec"]) < previous_elapsed:
                raise ValueError("trajectory timestamps must be monotonic")
            previous_elapsed = float(row["elapsed_sec"])
            rows.append(row)
    if len(rows) < 2:
        raise ValueError("trajectory must contain at least two rows")
    return rows


def render(
    rows: list[dict[str, float | None]],
    output: Path,
    max_frames: int,
) -> None:
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    width, height = 640, 340
    min_x, max_x = -1.4, 10.4
    min_y, max_y = -2.9, 2.9

    def pixel(x: float, y: float) -> tuple[int, int]:
        px = int((x - min_x) / (max_x - min_x) * (width - 1))
        py = int((max_y - y) / (max_y - min_y) * (height - 1))
        return px, py

    stride = max(1, math.ceil((len(rows) - 1) / (max_frames - 1)))
    indices = list(range(0, len(rows), stride))
    if indices[-1] != len(rows) - 1:
        indices.append(len(rows) - 1)
    frames = []
    for index in indices:
        image = Image.new("RGB", (width, height), (248, 249, 250))
        draw = ImageDraw.Draw(image)
        for segment in default_segments():
            draw.line(
                [
                    pixel(segment.x0, segment.y0),
                    pixel(segment.x1, segment.y1),
                ],
                fill=(38, 42, 48),
                width=5,
            )
        truth = [
            pixel(float(row["truth_x"]), float(row["truth_y"]))
            for row in rows[: index + 1]
            if row["truth_x"] is not None and row["truth_y"] is not None
        ]
        odom = [
            pixel(float(row["odom_x"]), float(row["odom_y"]))
            for row in rows[: index + 1]
            if row["odom_x"] is not None and row["odom_y"] is not None
        ]
        if len(truth) > 1:
            draw.line(truth, fill=(20, 105, 210), width=3)
        if len(odom) > 1:
            draw.line(odom, fill=(235, 125, 25), width=2)
        if truth:
            x, y = truth[-1]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(20, 105, 210))
        if odom:
            x, y = odom[-1]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(235, 125, 25))
        elapsed = float(rows[index]["elapsed_sec"])
        draw.rectangle((8, 8, 222, 38), fill=(255, 255, 255))
        draw.text(
            (15, 15),
            f"truth (blue) / odom (orange)  t={elapsed:.1f}s",
            fill=(20, 20, 20),
        )
        frames.append(image)
    elapsed_span = max(
        0.1,
        float(rows[indices[-1]]["elapsed_sec"])
        - float(rows[indices[0]]["elapsed_sec"]),
    )
    frame_duration_ms = max(
        40, min(250, int(1000.0 * elapsed_span / len(frames)))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frames[0].save(
        temporary,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )
    temporary.replace(output)


def main() -> int:
    args = parse_args()
    render(load_trajectory(args.csv), args.output, args.max_frames)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
