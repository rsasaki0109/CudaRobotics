#!/usr/bin/env python3
"""Summarize CUDA MPPI controller benchmarks on extended scenarios.

The input directory is produced by commands such as:

  ros2 run cuda_mppi_controller controller_benchmark /tmp/mppi_ext double_gap cpu_gpu
  ros2 run cuda_mppi_controller controller_benchmark /tmp/mppi_ext moving_crossing quick

Each scenario writes <bench_dir>/<scenario>/summary.csv. This script combines
those summaries into checked-in CSV/Markdown artifacts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = ("double_gap", "moving_crossing")
SUMMARY_FIELDS = (
    "scenario",
    "label",
    "plugin",
    "batch_size",
    "motion_model",
    "success",
    "collided",
    "steps",
    "sim_s",
    "mean_ms",
    "p95_ms",
    "max_ms",
    "exceptions",
    "distance_m",
    "mean_speed_mps",
    "max_speed_mps",
    "mean_abs_w_radps",
    "max_abs_w_radps",
    "mean_abs_curvature",
    "distance_field_weight",
    "distance_field_cutoff",
    "path_angle_weight",
    "curvature_speed_weight",
    "curvature_speed_min",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bench_dir", nargs="?", default="/tmp/mppi_extended_scenarios")
    parser.add_argument("date_tag", nargs="?", default="latest")
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument(
        "--labels",
        nargs="+",
        default=[],
        help="Optional label filter. By default every row in each summary is included.",
    )
    return parser.parse_args()


def load_rows(bench_dir: Path, scenarios: list[str], labels: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for scenario in scenarios:
        summary = bench_dir / scenario / "summary.csv"
        if not summary.exists():
            raise FileNotFoundError(f"missing benchmark summary: {summary}")
        with summary.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if labels and row.get("label", "") not in labels:
                    continue
                rows.append(row)
    if not rows:
        raise RuntimeError("no benchmark rows matched the requested scenarios/labels")
    return rows


def result(row: dict[str, str]) -> str:
    if row.get("success") == "1":
        return "success"
    if row.get("collided") == "1":
        return "collision"
    return "timeout"


def f(row: dict[str, str], key: str, digits: int = 2, suffix: str = "") -> str:
    value = row.get(key, "")
    if value == "":
        return "-"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except ValueError:
        return value


def write_csv(rows: list[dict[str, str]], date_tag: str) -> Path:
    out = REPO / "docs" / "results" / f"cuda_mppi_extended_scenarios_{date_tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    return out


def scenario_note(scenario: str) -> str:
    if scenario == "double_gap":
        return (
            "`double_gap` exercises path-following through two separated wall gaps "
            "with a deliberately bent global path."
        )
    if scenario == "moving_crossing":
        return (
            "`moving_crossing` repaints a crossing obstacle into the costmap during "
            "closed-loop control, so it is a dynamic-map smoke test rather than a "
            "static obstacle benchmark."
        )
    return f"`{scenario}` is an extended synthetic controller benchmark scenario."


def write_markdown(rows: list[dict[str, str]], date_tag: str, scenarios: list[str]) -> Path:
    out = REPO / "docs" / "results" / f"cuda_mppi_extended_scenarios_{date_tag}.md"
    csv_name = out.with_suffix(".csv").name

    lines = [
        f"# CUDA MPPI Extended Controller Scenarios ({date_tag})",
        "",
        "Closed-loop `cuda_mppi_controller` benchmark summary for scenarios beyond",
        "the original wall-gap / narrow-corridor / U-turn smoke set.",
        "",
        "Hardware: local CUDA-capable benchmark machine, ROS 2 workspace, Release build.",
        "Scenario setup: 10 m x 10 m synthetic costmap, 20 Hz closed loop,",
        "T = 56, dt = 0.05. The exact controller rows are preserved in the CSV.",
        "",
        f"CSV: [`{csv_name}`]({csv_name})",
        "",
        "## Scenario Intent",
        "",
    ]
    for scenario in scenarios:
        lines.append(f"- {scenario_note(scenario)}")

    lines += [
        "",
        "## Results",
        "",
        "| scenario | label | result | K | sim time | mean solve | p95 | distance | mean speed | mean \\|w\\| | mean \\|curv\\| | exceptions |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| {scenario} | {label} | {result} | {k} | {sim_s}s | {mean} ms | "
            "{p95} ms | {dist} m | {speed} m/s | {yaw} rad/s | {curv} | {exc} |".format(
                scenario=row.get("scenario", ""),
                label=row.get("label", ""),
                result=result(row),
                k=row.get("batch_size", ""),
                sim_s=f(row, "sim_s", 1),
                mean=f(row, "mean_ms"),
                p95=f(row, "p95_ms"),
                dist=f(row, "distance_m"),
                speed=f(row, "mean_speed_mps"),
                yaw=f(row, "mean_abs_w_radps"),
                curv=f(row, "mean_abs_curvature"),
                exc=row.get("exceptions", ""),
            )
        )

    lines += [
        "",
        "## Readout",
        "",
        "- Treat this as extended controller coverage, not a universal navigation",
        "  benchmark. The scenarios are synthetic and intentionally small enough",
        "  to run during local plugin development.",
        "- `double_gap` is useful for spotting path-window, path-angle, and",
        "  smoothing regressions that a straight wall-gap benchmark can miss.",
        "- `moving_crossing` is useful for costmap-refresh and diagnostics checks;",
        "  it does not replace a real perception or tracking pipeline.",
        "- Inspect per-cycle diagnostics with",
        "  `scripts/render_cuda_mppi_diagnostics.py` when a row times out,",
        "  retreats, or shows low valid-rollout ratios.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "cd ros2_ws",
        "colcon build --packages-select cuda_mppi_controller \\",
        "  --cmake-args -DCMAKE_BUILD_TYPE=Release",
        "source install/setup.bash",
        "ros2 run cuda_mppi_controller controller_benchmark /tmp/mppi_extended_scenarios double_gap cpu_gpu",
        "ros2 run cuda_mppi_controller controller_benchmark /tmp/mppi_extended_scenarios moving_crossing quick",
        "cd ..",
        f"python3 scripts/render_cuda_mppi_extended_scenarios.py /tmp/mppi_extended_scenarios {date_tag}",
        "```",
        "",
    ]

    out.write_text("\n".join(lines))
    return out


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.bench_dir), args.scenarios, set(args.labels))
    csv_path = write_csv(rows, args.date_tag)
    md_path = write_markdown(rows, args.date_tag, args.scenarios)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
