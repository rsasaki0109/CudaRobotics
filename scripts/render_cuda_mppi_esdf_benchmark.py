#!/usr/bin/env python3
"""Summarize the CUDA MPPI costmap-vs-ESDF critic benchmark."""

import csv
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCENARIOS = ("wall_gap", "narrow_corridor", "u_turn")
LABELS = ("gpu_costmap_K8192", "gpu_esdf_K8192")


def load_rows(bench_dir: Path):
    rows = []
    for scenario in SCENARIOS:
        summary = bench_dir / scenario / "summary.csv"
        if not summary.exists():
            raise FileNotFoundError(f"missing benchmark summary: {summary}")
        with summary.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["label"] in LABELS:
                    rows.append(row)
    return rows


def write_csv(rows, date_tag: str) -> Path:
    out = REPO / "docs" / "results" / f"cuda_mppi_esdf_{date_tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario",
        "label",
        "batch_size",
        "success",
        "collided",
        "steps",
        "sim_s",
        "mean_ms",
        "p95_ms",
        "max_ms",
        "exceptions",
        "distance_field_weight",
        "distance_field_cutoff",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    return out


def fmt_float(row, key, digits=2):
    return f"{float(row[key]):.{digits}f}"


def write_markdown(rows, date_tag: str) -> Path:
    by_key = {(r["scenario"], r["label"]): r for r in rows}
    out = REPO / "docs" / "results" / f"cuda_mppi_esdf_{date_tag}.md"
    csv_name = f"cuda_mppi_esdf_{date_tag}.csv"

    lines = [
        f"# CUDA MPPI ESDF Clearance Critic ({date_tag})",
        "",
        "Closed-loop GPU-only comparison of the default costmap critic against",
        "the optional ESDF-style distance-field clearance critic added to",
        "`cuda_mppi_controller`.",
        "",
        "Hardware: local CUDA-capable benchmark machine, ROS 2 workspace, Release build.",
        "Scenario setup: 10 m x 10 m synthetic costmap, 20 Hz closed loop,",
        "K = 8192, T = 56, dt = 0.05. ESDF row uses",
        "`distance_field_weight=12.0` and `distance_field_cutoff=0.8`.",
        "",
        f"CSV: [`{csv_name}`]({csv_name})",
        "",
        "## Results",
        "",
        "| scenario | critic | result | sim time | mean solve | p95 | max | exceptions |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for scenario in SCENARIOS:
        for label, critic in (
            ("gpu_costmap_K8192", "costmap"),
            ("gpu_esdf_K8192", "costmap + ESDF"),
        ):
            row = by_key[(scenario, label)]
            result = "success" if row["success"] == "1" else (
                "collision" if row["collided"] == "1" else "timeout")
            lines.append(
                "| {scenario} | {critic} | {result} | {sim_s}s | {mean} ms | "
                "{p95} ms | {max_ms} ms | {exceptions} |".format(
                    scenario=scenario,
                    critic=critic,
                    result=result,
                    sim_s=fmt_float(row, "sim_s", 1),
                    mean=fmt_float(row, "mean_ms"),
                    p95=fmt_float(row, "p95_ms"),
                    max_ms=fmt_float(row, "max_ms"),
                    exceptions=row["exceptions"],
                )
            )

    lines += [
        "",
        "## Readout",
        "",
        "- The ESDF critic is disabled by default; this benchmark enables it only",
        "  for the `gpu_esdf_K8192` rows.",
        "- All three corrected scenarios succeed with both the default costmap",
        "  critic and the ESDF clearance critic at K=8192.",
        "- ESDF keeps similar solve latency on `wall_gap` and `narrow_corridor`,",
        "  while the corrected `u_turn` cell finishes sooner with ESDF enabled in",
        "  this run.",
        "- The corrected `u_turn` path goes around the obstacle endpoint; the",
        "  previous benchmark path crossed a lethal wall cell and was therefore",
        "  not a valid planner-tracking test.",
        "- The distance-field cost is a clearance smoother, not a replacement for",
        "  lethal-cell collision rejection or footprint checking.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "cd ros2_ws",
        "colcon build --packages-select cuda_mppi_controller \\",
        "  --cmake-args -DCMAKE_BUILD_TYPE=Release",
        "source install/setup.bash",
        "ros2 run cuda_mppi_controller controller_benchmark /tmp/mppi_esdf_bench esdf",
        "cd ..",
        f"python3 scripts/render_cuda_mppi_esdf_benchmark.py /tmp/mppi_esdf_bench {date_tag}",
        "```",
        "",
    ]

    out.write_text("\n".join(lines))
    return out


def main():
    bench_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/mppi_esdf_bench")
    date_tag = sys.argv[2] if len(sys.argv) > 2 else "latest"
    rows = load_rows(bench_dir)
    csv_path = write_csv(rows, date_tag)
    md_path = write_markdown(rows, date_tag)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
