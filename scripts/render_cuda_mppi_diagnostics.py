#!/usr/bin/env python3
"""Render cuda_mppi_controller diagnostics CSV into an SVG and Markdown summary."""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Diagnostics CSV from diagnostics_csv_path.")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help="Output path without suffix. Defaults to docs/results/<csv-stem>.",
    )
    parser.add_argument("--title", default="CUDA MPPI diagnostics")
    return parser.parse_args()


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            # The controller appends and writes a header each time the CSV is
            # opened. Ignore repeated headers in long-running logs.
            if row.get("stamp_sec") == "stamp_sec":
                continue
            if not row.get("stamp_sec"):
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no diagnostics rows in {path}")
    return rows


def finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def percentile(values: list[float], q: float) -> float:
    clean = sorted(finite(values))
    if not clean:
        return float("nan")
    idx = min(len(clean) - 1, max(0, int(round((len(clean) - 1) * q))))
    return clean[idx]


def mean(values: list[float]) -> float:
    clean = finite(values)
    return sum(clean) / len(clean) if clean else float("nan")


def fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def output_stem(args: argparse.Namespace) -> Path:
    if args.output_stem is not None:
        return args.output_stem
    return REPO / "docs" / "results" / args.csv.stem


def render_svg(rows: list[dict[str, str]], stem: Path, title: str) -> Path:
    t0 = as_float(rows[0], "stamp_sec")
    time_s = [as_float(r, "stamp_sec") - t0 for r in rows]
    solve_ms = [as_float(r, "solve_ms") for r in rows]
    valid_ratio = [as_float(r, "valid_rollout_ratio") for r in rows]
    best_cost = [as_float(r, "best_cost") for r in rows]
    mean_cost = [as_float(r, "mean_cost") for r in rows]
    cmd_v = [as_float(r, "cmd_v") for r in rows]
    cmd_vy = [as_float(r, "cmd_vy") for r in rows]
    cmd_w = [as_float(r, "cmd_w") for r in rows]
    retreat_t = [
        t for t, r in zip(time_s, rows)
        if r.get("retreating") == "1" or r.get("all_colliding") == "1"
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.0), sharex=True)
    fig.suptitle(title)

    axes[0].plot(time_s, solve_ms, color="#315c8a", lw=1.5)
    axes[0].axhline(50.0, color="#999999", ls="--", lw=1.0)
    axes[0].set_ylabel("solve [ms]")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_s, valid_ratio, color="#1f6f64", lw=1.5)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("valid ratio")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time_s, best_cost, color="#8e3f32", lw=1.2, label="best")
    axes[2].plot(time_s, mean_cost, color="#b46618", lw=1.2, label="mean")
    axes[2].set_ylabel("cost")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(time_s, cmd_v, color="#1f6f64", lw=1.2, label="v")
    if any(abs(v) > 1.0e-5 for v in finite(cmd_vy)):
        axes[3].plot(time_s, cmd_vy, color="#315c8a", lw=1.2, label="vy")
    axes[3].plot(time_s, cmd_w, color="#8e3f32", lw=1.2, label="w")
    axes[3].set_ylabel("command")
    axes[3].set_xlabel("time [s]")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.25)

    for ax in axes:
        for t in retreat_t:
            ax.axvline(t, color="#222222", alpha=0.15, lw=0.7)

    stem.parent.mkdir(parents=True, exist_ok=True)
    svg = stem.with_suffix(".svg")
    fig.tight_layout()
    fig.savefig(svg)
    plt.close(fig)
    return svg


def write_markdown(rows: list[dict[str, str]], source_csv: Path, svg: Path, stem: Path) -> Path:
    solve_ms = [as_float(r, "solve_ms") for r in rows]
    valid_ratio = [as_float(r, "valid_rollout_ratio") for r in rows]
    sampled = [as_float(r, "sampled_rollouts") for r in rows]
    valid = [as_float(r, "valid_rollouts") for r in rows]
    cmd_v = [as_float(r, "cmd_v") for r in rows]
    cmd_w = [as_float(r, "cmd_w") for r in rows]
    all_colliding = sum(1 for r in rows if r.get("all_colliding") == "1")
    retreating = sum(1 for r in rows if r.get("retreating") == "1")
    t0 = as_float(rows[0], "stamp_sec")
    t1 = as_float(rows[-1], "stamp_sec")
    duration = t1 - t0 if math.isfinite(t0) and math.isfinite(t1) else float("nan")

    md = stem.with_suffix(".md")
    lines = [
        "# CUDA MPPI Diagnostics Summary",
        "",
        f"Source CSV: `{source_csv}`",
        "",
        f"![diagnostics plot]({svg.name})",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| rows | {len(rows)} |",
        f"| duration | {fmt(duration, 2)} s |",
        f"| solve mean | {fmt(mean(solve_ms), 3)} ms |",
        f"| solve p95 | {fmt(percentile(solve_ms, 0.95), 3)} ms |",
        f"| solve max | {fmt(max(finite(solve_ms)) if finite(solve_ms) else float('nan'), 3)} ms |",
        f"| valid rollout ratio mean | {fmt(mean(valid_ratio), 3)} |",
        f"| valid rollout ratio min | {fmt(min(finite(valid_ratio)) if finite(valid_ratio) else float('nan'), 3)} |",
        f"| sampled rollouts median | {fmt(percentile(sampled, 0.50), 0)} |",
        f"| valid rollouts median | {fmt(percentile(valid, 0.50), 0)} |",
        f"| all-colliding cycles | {all_colliding} |",
        f"| retreat cycles | {retreating} |",
        f"| command v range | {fmt(min(finite(cmd_v)) if finite(cmd_v) else float('nan'), 3)} to {fmt(max(finite(cmd_v)) if finite(cmd_v) else float('nan'), 3)} m/s |",
        f"| command w range | {fmt(min(finite(cmd_w)) if finite(cmd_w) else float('nan'), 3)} to {fmt(max(finite(cmd_w)) if finite(cmd_w) else float('nan'), 3)} rad/s |",
        "",
        "## Readout",
        "",
        "- Sustained low valid-rollout ratio points to poor local path windowing,",
        "  overly tight costmap geometry, or insufficient sample count for the",
        "  current scene.",
        "- Repeated retreat cycles mean the controller is recovering from",
        "  all-colliding samples using the last valid sequence; inspect the",
        "  costmap and footprint settings before tuning costs.",
        "- Solve spikes above the control period are visible against the 50 ms",
        "  reference line in the first plot.",
        "",
    ]
    md.write_text("\n".join(lines))
    return md


def main() -> int:
    args = parse_args()
    rows = load_rows(args.csv)
    stem = output_stem(args)
    svg = render_svg(rows, stem, args.title)
    md = write_markdown(rows, args.csv, svg, stem)
    print(f"wrote {svg}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
