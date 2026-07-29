#!/usr/bin/env python3
"""Render content-bound visual evidence for native CudaNav closed-loop runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "step",
    "traversal",
    "time_s",
    "truth_x",
    "truth_y",
    "truth_yaw",
    "estimated_x",
    "estimated_y",
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
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_lf(path: Path) -> str:
    contents = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(contents).hexdigest()


def read_evidence(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload.get("profile") == "release", (
        "visual release evidence requires a native release-profile result"
    )
    assert payload.get("claims") == {
        "native_gpu_core_closed_loop": True,
        "real_data": False,
        "ros2_runtime": False,
    }, "unexpected native evidence claim boundary"
    checks = payload.get("checks")
    assert isinstance(checks, dict) and checks
    assert all(checks.values()), "source release evidence did not pass every check"
    return payload


def read_trajectory(
    path: Path, evidence: dict[str, Any]
) -> list[dict[str, float | int]]:
    expected_hash = evidence["artifacts"]["trajectory"]["sha256"]
    assert sha256_file(path) == expected_hash, (
        "trajectory SHA-256 does not match the published native evidence"
    )
    integer_fields = {
        "step",
        "traversal",
        "inliers",
        "observed_voxels",
        "occupied_cells",
        "all_colliding",
        "retreating",
    }
    rows: list[dict[str, float | int]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        assert not missing, f"trajectory is missing columns: {missing}"
        for raw in reader:
            row: dict[str, float | int] = {}
            for name in REQUIRED_COLUMNS:
                value = int(raw[name]) if name in integer_fields else float(raw[name])
                assert math.isfinite(float(value)), (
                    f"trajectory contains non-finite {name}"
                )
                row[name] = value
            rows.append(row)
    metrics = evidence["metrics"]
    assert len(rows) == metrics["frames"], "trajectory row count mismatch"
    assert rows and rows[0]["step"] == 0
    assert rows[-1]["step"] == len(rows) - 1
    assert all(
        int(row["step"]) == index for index, row in enumerate(rows)
    ), "trajectory steps must be contiguous"
    traversal_ids = sorted({int(row["traversal"]) for row in rows})
    assert traversal_ids == list(range(metrics["traversals_completed"])), (
        "trajectory traversal inventory mismatch"
    )
    return rows


def sample_indices(
    rows: list[dict[str, float | int]], maximum_frames: int
) -> list[int]:
    if maximum_frames < 2:
        raise ValueError("maximum_frames must be at least 2")
    if len(rows) <= maximum_frames:
        return list(range(len(rows)))
    boundaries: set[int] = {0, len(rows) - 1}
    previous = int(rows[0]["traversal"])
    for index, row in enumerate(rows[1:], start=1):
        current = int(row["traversal"])
        if current != previous:
            boundaries.update((index - 1, index))
            previous = current
    assert len(boundaries) <= maximum_frames, (
        "maximum_frames is too small to retain every traversal boundary"
    )
    if maximum_frames > 1:
        for slot in range(maximum_frames):
            boundaries.add(
                round(slot * (len(rows) - 1) / (maximum_frames - 1))
            )
            if len(boundaries) >= maximum_frames:
                break
    if len(boundaries) < maximum_frames:
        for index in range(len(rows)):
            boundaries.add(index)
            if len(boundaries) == maximum_frames:
                break
    return sorted(boundaries)


def traversal_start_indices(
    rows: list[dict[str, float | int]]
) -> dict[int, int]:
    starts: dict[int, int] = {}
    for index, row in enumerate(rows):
        starts.setdefault(int(row["traversal"]), index)
    return starts


def render_gif(
    rows: list[dict[str, float | int]],
    evidence: dict[str, Any],
    output: Path,
    *,
    maximum_frames: int,
    width: int,
    height: int,
) -> list[int]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required: pip install pillow"
        ) from error

    indices = sample_indices(rows, maximum_frames)
    starts = traversal_start_indices(rows)
    bounds = evidence["scenario"]["outer_bounds"]
    x_min, y_min, x_max, y_max = map(float, bounds)
    plot = (38, 66, int(width * 0.66), height - 42)
    plot_width = plot[2] - plot[0]
    plot_height = plot[3] - plot[1]

    def point(x: float, y: float) -> tuple[int, int]:
        px = plot[0] + round((x - x_min) / (x_max - x_min) * plot_width)
        py = plot[3] - round((y - y_min) / (y_max - y_min) * plot_height)
        return px, py

    frames = []
    for source_index in indices:
        row = rows[source_index]
        image = Image.new("RGB", (width, height), "#07111d")
        draw = ImageDraw.Draw(image)
        draw.text(
            (38, 22),
            "CudaNav native all-GPU closed loop",
            fill="#f1f5f9",
        )
        draw.text(
            (width - 265, 22),
            "KISS-ICP -> voxel -> ESDF -> CUDA MPPI",
            fill="#7dd3fc",
        )
        draw.rounded_rectangle(
            plot,
            radius=8,
            fill="#0f1d2b",
            outline="#334155",
            width=2,
        )
        for obstacle in evidence["scenario"]["obstacles"]:
            ox0, oy0, ox1, oy1 = map(float, obstacle)
            left, bottom = point(ox0, oy0)
            right, top = point(ox1, oy1)
            draw.rectangle(
                (left, top, right, bottom),
                fill="#7f1d1d",
                outline="#f87171",
                width=2,
            )
        waypoints = [
            point(float(x), float(y))
            for x, y in evidence["scenario"]["waypoints"]
        ]
        draw.line(waypoints, fill="#64748b", width=2)
        for waypoint in waypoints:
            draw.ellipse(
                (
                    waypoint[0] - 3,
                    waypoint[1] - 3,
                    waypoint[0] + 3,
                    waypoint[1] + 3,
                ),
                fill="#94a3b8",
            )

        traversal = int(row["traversal"])
        start = starts[traversal]
        trail_rows = rows[start : source_index + 1 : 3]
        truth = [
            point(float(item["truth_x"]), float(item["truth_y"]))
            for item in trail_rows
        ]
        estimate = [
            point(
                float(item["estimated_x"]),
                float(item["estimated_y"]),
            )
            for item in trail_rows
        ]
        if len(truth) > 1:
            draw.line(truth, fill="#22d3ee", width=4)
            draw.line(estimate, fill="#fb923c", width=2)
        truth_now = point(float(row["truth_x"]), float(row["truth_y"]))
        estimate_now = point(
            float(row["estimated_x"]), float(row["estimated_y"])
        )
        radius = 7
        draw.ellipse(
            (
                truth_now[0] - radius,
                truth_now[1] - radius,
                truth_now[0] + radius,
                truth_now[1] + radius,
            ),
            fill="#22d3ee",
            outline="#cffafe",
        )
        yaw = float(row["truth_yaw"])
        arrow = point(
            float(row["truth_x"]) + 0.32 * math.cos(yaw),
            float(row["truth_y"]) + 0.32 * math.sin(yaw),
        )
        draw.line((truth_now, arrow), fill="#ffffff", width=3)
        draw.ellipse(
            (
                estimate_now[0] - 4,
                estimate_now[1] - 4,
                estimate_now[0] + 4,
                estimate_now[1] + 4,
            ),
            outline="#fdba74",
            width=2,
        )

        panel_x = plot[2] + 28
        lines = [
            f"Traversal  {traversal + 1:02d} / {evidence['metrics']['traversals_completed']}",
            f"Time       {float(row['time_s']):7.1f} s",
            f"Distance   {float(evidence['metrics']['ground_truth_distance_m']):7.2f} m total",
            f"Odom error {float(row['error_m']):7.4f} m",
            f"ICP inliers {int(row['inliers']):6d}",
            f"Voxels      {int(row['observed_voxels']):6d}",
            f"Occupied    {int(row['occupied_cells']):6d}",
            f"Valid MPPI  {float(row['valid_rollout_ratio']):7.1%}",
            f"MPPI solve  {float(row['solve_ms']):7.3f} ms",
            f"Frame       {float(row['frame_ms']):7.3f} ms",
            f"Command v,w {float(row['command_v']):5.2f}, {float(row['command_w']):5.2f}",
            "",
            f"collision count     {evidence['metrics']['collision_count']}",
            "deadline miss rate  "
            f"{float(evidence['metrics']['command_deadline_miss_rate']):.2%}",
            "final drift         "
            f"{float(evidence['metrics']['odometry_drift_percent']):.4f}%",
        ]
        for line_index, text in enumerate(lines):
            draw.text(
                (panel_x, 76 + line_index * 23),
                text,
                fill="#e2e8f0" if line_index < 11 else "#86efac",
            )
        draw.line(
            ((panel_x, height - 69), (width - 38, height - 69)),
            fill="#334155",
            width=1,
        )
        draw.text(
            (panel_x, height - 52),
            "cyan: ground truth   orange: GPU KISS-ICP",
            fill="#94a3b8",
        )
        frames.append(image)

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=70,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return indices


def build_manifest(
    *,
    evidence_path: Path,
    trajectory_path: Path,
    output_path: Path,
    indices: list[int],
    source_rows: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    renderer = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "evidence_mode": "cudanav_native_release_visual",
        "claim_boundary": {
            "native_gpu_core_closed_loop": True,
            "real_data": False,
            "ros2_runtime": False,
        },
        "source_evidence": {
            "path": evidence_path.name,
            "normalization": "text_lf",
            "sha256": sha256_text_lf(evidence_path),
        },
        "trajectory": {
            "path": trajectory_path.name,
            "rows": source_rows,
            "sha256": sha256_file(trajectory_path),
        },
        "renderer": {
            "path": renderer.relative_to(ROOT).as_posix(),
            "normalization": "text_lf",
            "sha256": sha256_text_lf(renderer),
        },
        "visual": {
            "path": output_path.name,
            "format": "gif",
            "width": width,
            "height": height,
            "frames": len(indices),
            "sampled_source_steps_sha256": hashlib.sha256(
                json.dumps(indices, separators=(",", ":")).encode()
            ).hexdigest(),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
    }


def validate_manifest(
    manifest: dict[str, Any],
    *,
    evidence_path: Path,
    trajectory_path: Path,
    output_path: Path,
) -> dict[str, bool]:
    try:
        from PIL import Image

        with Image.open(output_path) as image:
            actual_dimensions = image.size
            actual_frames = int(getattr(image, "n_frames", 1))
    except (ImportError, OSError):
        actual_dimensions = (0, 0)
        actual_frames = 0
    visual = manifest.get("visual", {})
    renderer = manifest.get("renderer", {})
    claims = manifest.get("claim_boundary")
    checks = {
        "schema": manifest.get("schema_version") == 1,
        "mode": manifest.get("evidence_mode")
        == "cudanav_native_release_visual",
        "claim_boundary": claims
        == {
            "native_gpu_core_closed_loop": True,
            "real_data": False,
            "ros2_runtime": False,
        },
        "source_evidence": manifest.get("source_evidence", {}).get("sha256")
        == sha256_text_lf(evidence_path),
        "trajectory": manifest.get("trajectory", {}).get("sha256")
        == sha256_file(trajectory_path),
        "renderer": renderer.get("sha256")
        == sha256_text_lf(Path(__file__).resolve()),
        "visual_exists": output_path.is_file(),
        "visual_size": output_path.is_file()
        and visual.get("bytes") == output_path.stat().st_size,
        "visual_sha256": output_path.is_file()
        and visual.get("sha256") == sha256_file(output_path),
        "visual_format": output_path.is_file()
        and output_path.read_bytes()[:6] in {b"GIF87a", b"GIF89a"},
        "visual_dimensions": visual.get("width", 0) >= 640
        and visual.get("height", 0) >= 360,
        "visual_frames": visual.get("frames", 0) >= 60,
        "encoded_dimensions": actual_dimensions
        == (visual.get("width"), visual.get("height")),
        "encoded_frames": actual_frames == visual.get("frames"),
    }
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--maximum-frames", type=int, default=180)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence_path = args.evidence.resolve()
    trajectory_path = args.trajectory.resolve()
    output_path = args.output.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else output_path.with_suffix(".json")
    )
    evidence = read_evidence(evidence_path)
    rows = read_trajectory(trajectory_path, evidence)
    if args.check_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        indices = render_gif(
            rows,
            evidence,
            output_path,
            maximum_frames=args.maximum_frames,
            width=args.width,
            height=args.height,
        )
        manifest = build_manifest(
            evidence_path=evidence_path,
            trajectory_path=trajectory_path,
            output_path=output_path,
            indices=indices,
            source_rows=len(rows),
            width=args.width,
            height=args.height,
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    checks = validate_manifest(
        manifest,
        evidence_path=evidence_path,
        trajectory_path=trajectory_path,
        output_path=output_path,
    )
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise SystemExit(f"visual evidence checks failed: {failed}")
    print(
        f"PASS: {manifest['visual']['frames']} frames, "
        f"sha256={manifest['visual']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
