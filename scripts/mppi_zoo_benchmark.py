#!/usr/bin/env python3

from __future__ import annotations

import csv
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


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


def benchmark_command(
    *,
    bin_path: str | Path,
    csv_path: Path,
    scenarios: str,
    planners: str,
    k_values: str,
    seed_count: int | str,
) -> list[str]:
    return [
        str(repo_path(bin_path)),
        "--quick",
        "--scenarios",
        scenarios,
        "--planners",
        planners,
        "--k-values",
        k_values,
        "--seed-count",
        str(seed_count),
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
    *,
    title: str,
    generator: str,
    command: list[str],
    csv_path: Path,
    markdown_path: Path,
    scenarios: str,
    planners: str,
    k_values: str,
    seed_count: int | str,
    dry_run: bool,
    skip_run: bool,
    rows: list[dict[str, object]] | None,
) -> str:
    lines: list[str] = [
        f"# {title}",
        "",
        f"_Generated by `{generator}`._",
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
                ["scenarios", scenarios],
                ["planners", planners],
                ["k_values", k_values],
                ["seed_count", str(seed_count)],
                ["dry_run", str(bool(dry_run))],
                ["skip_run", str(bool(skip_run))],
            ],
        )
    )
    lines.append("")

    if rows is None:
        lines.extend(["## Results", "", "Dry run only. No CSV was read.", ""])
        return "\n".join(lines).rstrip() + "\n"

    summary = group_summary(rows)
    lines.extend(["## Planner Aggregate", ""])
    lines.extend(render_aggregate(summary))
    lines.extend(["", "## Best Per Scenario And K", ""])
    lines.extend(render_best(summary))
    lines.extend(["", "## Per Scenario Details", ""])
    lines.extend(render_details(summary))
    return "\n".join(lines).rstrip() + "\n"


def run_benchmark(
    *,
    bin_path: str | Path,
    out_dir: str | Path,
    csv_path: str | Path | None,
    markdown_path: str | Path | None,
    scenarios: str,
    planners: str,
    k_values: str,
    seed_count: int | str,
    title: str,
    generator: str,
    csv_basename: str,
    markdown_basename: str,
    dry_run: bool = False,
    skip_run: bool = False,
) -> tuple[Path, Path, list[str]]:
    out_dir_path = repo_path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    csv_out = repo_path(csv_path) if csv_path else out_dir_path / csv_basename
    markdown_out = repo_path(markdown_path) if markdown_path else out_dir_path / markdown_basename
    markdown_out.parent.mkdir(parents=True, exist_ok=True)

    command = benchmark_command(
        bin_path=bin_path,
        csv_path=csv_out,
        scenarios=scenarios,
        planners=planners,
        k_values=k_values,
        seed_count=seed_count,
    )

    if dry_run:
        markdown_out.write_text(
            render_report(
                title=title,
                generator=generator,
                command=command,
                csv_path=csv_out,
                markdown_path=markdown_out,
                scenarios=scenarios,
                planners=planners,
                k_values=k_values,
                seed_count=seed_count,
                dry_run=True,
                skip_run=skip_run,
                rows=None,
            )
        )
        return csv_out, markdown_out, command

    if not skip_run:
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    if not csv_out.exists():
        raise SystemExit(f"CSV not found: {csv_out}")

    rows = load_rows(csv_out)
    if not rows:
        raise SystemExit(f"CSV has no rows: {csv_out}")

    markdown_out.write_text(
        render_report(
            title=title,
            generator=generator,
            command=command,
            csv_path=csv_out,
            markdown_path=markdown_out,
            scenarios=scenarios,
            planners=planners,
            k_values=k_values,
            seed_count=seed_count,
            dry_run=False,
            skip_run=skip_run,
            rows=rows,
        )
    )
    return csv_out, markdown_out, command
