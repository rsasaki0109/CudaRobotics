#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCENARIOS = "dynamic_crossing,narrow_passage"
DEFAULT_PLANNERS = (
    "mppi,"
    "lp_mppi_smooth,"
    "step_mppi_smooth,"
    "tsallis_mppi_smooth,"
    "dra_mppi_soft,"
    "c2u_mppi_smooth,"
    "ducct_mppi_smooth,"
    "dbas_log_mppi_agile,"
    "pa_mppi_smooth"
)
DEFAULT_K_VALUES = "64,128"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MPPI reproduction zoo smoke benchmark and render a compact Markdown report."
    )
    parser.add_argument("--bin", default="bin/benchmark_diff_mppi", help="Path to benchmark_diff_mppi binary.")
    parser.add_argument("--out-dir", default="build/mppi_zoo", help="Directory for CSV and Markdown output.")
    parser.add_argument("--csv", help="CSV path. Defaults to <out-dir>/mppi_zoo_smoke.csv.")
    parser.add_argument("--markdown-out", help="Markdown report path. Defaults to <out-dir>/mppi_zoo_smoke.md.")
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS, help="Comma-separated scenario names.")
    parser.add_argument("--planners", default=DEFAULT_PLANNERS, help="Comma-separated planner names.")
    parser.add_argument("--k-values", default=DEFAULT_K_VALUES, help="Comma-separated K sample counts.")
    parser.add_argument("--seed-count", default="3", help="Number of seeds per scenario/planner/K cell.")
    parser.add_argument("--dry-run", action="store_true", help="Write the command report without running the binary.")
    parser.add_argument("--skip-run", action="store_true", help="Summarize an existing CSV without running the binary.")
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def benchmark_command(args: argparse.Namespace, csv_path: Path) -> list[str]:
    return [
        str(repo_path(args.bin)),
        "--quick",
        "--scenarios",
        args.scenarios,
        "--planners",
        args.planners,
        "--k-values",
        args.k_values,
        "--seed-count",
        str(args.seed_count),
        "--csv",
        str(csv_path),
    ]


def parse_float(row: dict[str, str], field: str) -> float:
    return float(row.get(field, "0") or 0.0)


def parse_int(row: dict[str, str], field: str) -> int:
    return int(float(row.get(field, "0") or 0.0))


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "scenario": row["scenario"],
                    "planner": row["planner"],
                    "seed": parse_int(row, "seed"),
                    "k_samples": parse_int(row, "k_samples"),
                    "success": parse_float(row, "success"),
                    "steps": parse_float(row, "steps"),
                    "final_distance": parse_float(row, "final_distance"),
                    "cumulative_cost": parse_float(row, "cumulative_cost"),
                    "collisions": parse_float(row, "collisions"),
                    "avg_control_ms": parse_float(row, "avg_control_ms"),
                }
            )
    return rows


def group_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scenario"]), str(row["planner"]), int(row["k_samples"]))].append(row)

    summary: list[dict[str, object]] = []
    for (scenario, planner, k_samples), entries in sorted(groups.items()):
        summary.append(
            {
                "scenario": scenario,
                "planner": planner,
                "k_samples": k_samples,
                "episodes": len(entries),
                "success": mean(float(e["success"]) for e in entries),
                "steps": mean(float(e["steps"]) for e in entries),
                "final_distance": mean(float(e["final_distance"]) for e in entries),
                "cumulative_cost": mean(float(e["cumulative_cost"]) for e in entries),
                "collisions": mean(float(e["collisions"]) for e in entries),
                "avg_control_ms": mean(float(e["avg_control_ms"]) for e in entries),
            }
        )
    return summary


def rank_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        -float(row["success"]),
        float(row["final_distance"]),
        float(row["cumulative_cost"]),
        float(row["avg_control_ms"]),
    )


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def render_aggregate(summary: list[dict[str, object]]) -> list[str]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary:
        groups[str(row["planner"])].append(row)

    rows: list[list[str]] = []
    for planner in sorted(groups):
        entries = groups[planner]
        cells = len(entries)
        solved = sum(1 for e in entries if float(e["success"]) >= 0.999)
        rows.append(
            [
                planner,
                str(cells),
                str(solved),
                fmt(mean(float(e["success"]) for e in entries)),
                fmt(mean(float(e["final_distance"]) for e in entries)),
                fmt(mean(float(e["steps"]) for e in entries), 1),
                fmt(mean(float(e["avg_control_ms"]) for e in entries), 3),
                fmt(mean(float(e["collisions"]) for e in entries)),
            ]
        )
    return table(
        ["planner", "cells", "solved", "success", "final_d", "steps", "avg_ms", "collisions"],
        rows,
    )


def render_best(summary: list[dict[str, object]]) -> list[str]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in summary:
        groups[(str(row["scenario"]), int(row["k_samples"]))].append(row)

    rows: list[list[str]] = []
    for (scenario, k_samples), entries in sorted(groups.items()):
        ranked = sorted(entries, key=rank_key)
        best = ranked[0]
        mppi = next((e for e in entries if e["planner"] == "mppi"), None)
        if mppi is None:
            delta_final = ""
            time_ratio = ""
        else:
            delta_final = fmt(float(best["final_distance"]) - float(mppi["final_distance"]))
            time_ratio = fmt(float(best["avg_control_ms"]) / max(1.0e-6, float(mppi["avg_control_ms"])))
        rows.append(
            [
                scenario,
                str(k_samples),
                str(best["planner"]),
                fmt(float(best["success"])),
                fmt(float(best["final_distance"])),
                fmt(float(best["steps"]), 1),
                fmt(float(best["avg_control_ms"]), 3),
                delta_final,
                time_ratio,
            ]
        )
    return table(
        ["scenario", "K", "best", "success", "final_d", "steps", "avg_ms", "delta_final_vs_mppi", "time_ratio"],
        rows,
    )


def render_details(summary: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    scenarios = sorted({str(row["scenario"]) for row in summary})
    for scenario in scenarios:
        lines.append(f"### `{scenario}`")
        lines.append("")
        rows: list[list[str]] = []
        entries = [row for row in summary if row["scenario"] == scenario]
        for row in sorted(entries, key=lambda r: (int(r["k_samples"]), str(r["planner"]))):
            rows.append(
                [
                    str(row["planner"]),
                    str(row["k_samples"]),
                    str(row["episodes"]),
                    fmt(float(row["success"])),
                    fmt(float(row["final_distance"])),
                    fmt(float(row["steps"]), 1),
                    fmt(float(row["avg_control_ms"]), 3),
                    fmt(float(row["collisions"])),
                ]
            )
        lines.extend(table(["planner", "K", "episodes", "success", "final_d", "steps", "avg_ms", "collisions"], rows))
        lines.append("")
    return lines


def render_report(
    args: argparse.Namespace,
    command: list[str],
    csv_path: Path,
    markdown_path: Path,
    rows: list[dict[str, object]] | None,
) -> str:
    lines: list[str] = [
        "# MPPI Zoo Smoke Report",
        "",
        "_Generated by `python3 scripts/run_mppi_zoo_smoke.py`._",
        "",
        "## Command",
        "",
        "```bash",
        shlex.join(command),
        "```",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(
        table(
            ["field", "value"],
            [
                ["csv", rel(csv_path)],
                ["markdown", rel(markdown_path)],
                ["scenarios", args.scenarios],
                ["planners", args.planners],
                ["k_values", args.k_values],
                ["seed_count", str(args.seed_count)],
                ["dry_run", str(bool(args.dry_run))],
                ["skip_run", str(bool(args.skip_run))],
            ],
        )
    )
    lines.append("")

    if rows is None:
        lines.extend(["## Results", "", "Dry run only. No CSV was read.", ""])
        return "\n".join(lines) + "\n"

    summary = group_summary(rows)
    lines.extend(["## Planner Aggregate", ""])
    lines.extend(render_aggregate(summary))
    lines.extend(["", "## Best Per Scenario And K", ""])
    lines.extend(render_best(summary))
    lines.extend(["", "## Per Scenario Details", ""])
    lines.extend(render_details(summary))
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = repo_path(args.csv) if args.csv else out_dir / "mppi_zoo_smoke.csv"
    markdown_path = repo_path(args.markdown_out) if args.markdown_out else out_dir / "mppi_zoo_smoke.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    command = benchmark_command(args, csv_path)
    if args.dry_run:
        markdown_path.write_text(render_report(args, command, csv_path, markdown_path, None))
        print(shlex.join(command))
        print(f"Dry-run report saved to {rel(markdown_path)}")
        return

    if not args.skip_run:
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"CSV has no rows: {csv_path}")

    markdown_path.write_text(render_report(args, command, csv_path, markdown_path, rows))
    print(f"Markdown report saved to {rel(markdown_path)}")


if __name__ == "__main__":
    main()
