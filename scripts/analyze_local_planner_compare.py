#!/usr/bin/env python3
"""Render a Markdown comparison report for the cross-planner sweep.

The sweep summary CSV produced by ``sweep_grad_horizon_difficulty.py``
contains one row per (scenario, planner, speed_scale, radius_scale)
cell. This script slices it three ways:

1. Per cell: a wide table comparing every planner side by side
   (success / final_d / ms).
2. Per scenario: a "best planner" ranking — for each cell, which
   planner has the highest success rate (ties broken by lowest
   final_distance), and the gap to the second best.
3. Cross-scenario: a one-line takeaway per planner — number of cells
   solved, mean final_distance over solved cells, mean ms.

Goal of the report: surface where DWA (reactive, short horizon),
STOMP (cost-weighted noise + smoothness) and Diff-MPPI (gradient-
informed nominal updates) each win on the same difficulty grid.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def planner_family(planner: str) -> str:
    if planner.startswith("dwa_"):
        return "DWA"
    if planner.startswith("stomp_"):
        return "STOMP"
    if planner.startswith("diff_mppi_"):
        return "Diff-MPPI"
    if planner == "mppi":
        return "MPPI"
    if planner.startswith("hybrid_astar"):
        return "Hybrid-A*"
    return "other"


def planner_sort_key(planner: str):
    # Sort by family, then by name for stable column ordering.
    fam = planner_family(planner)
    fam_order = {"Hybrid-A*": 0, "DWA": 1, "STOMP": 2, "MPPI": 3,
                 "Diff-MPPI": 4, "other": 5}
    return (fam_order[fam], planner)


def load_summary(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "scenario": r["scenario"],
                "planner": r["planner"],
                "speed": float(r["dyn_speed_scale"]),
                "radius": float(r["dyn_radius_scale"]),
                "success": float(r["success_rate"]),
                "final_d": float(r["final_distance"]),
                "min_d": float(r["min_goal_distance"]),
                "cost": float(r["cumulative_cost"]),
                "collisions": float(r.get("collisions", 0.0)),
                "ms": float(r["avg_control_ms"]),
            })
    return rows


def fmt_succ(value: float) -> str:
    if value >= 0.999:
        return "**1.00**"
    if value <= 0.0:
        return "0.00"
    return f"{value:.2f}"


def render_per_cell(rows: list[dict]) -> list[str]:
    cells = defaultdict(list)
    for r in rows:
        cells[(r["scenario"], r["speed"], r["radius"])].append(r)
    out = ["## Per-cell comparison", ""]
    for key in sorted(cells.keys()):
        scenario, sp, rad = key
        entries = sorted(cells[key], key=lambda e: planner_sort_key(e["planner"]))
        out.append(f"### {scenario} | speed={sp:+.2f} radius={rad:.2f}")
        out.append("")
        out.append("| planner | family | succ | final_d | min_d | ms |")
        out.append("|---|---|---|---|---|---|")
        for e in entries:
            out.append(
                f"| {e['planner']} | {planner_family(e['planner'])} | "
                f"{fmt_succ(e['success'])} | {e['final_d']:.2f} | "
                f"{e['min_d']:.2f} | {e['ms']:.2f} |")
        out.append("")
    return out


def render_best_per_cell(rows: list[dict]) -> list[str]:
    cells = defaultdict(list)
    for r in rows:
        cells[(r["scenario"], r["speed"], r["radius"])].append(r)
    out = ["## Best planner per cell", "",
           "Best = highest success_rate (ties broken by lowest final_d).",
           "",
           "| scenario | speed | radius | best planner | succ | final_d |"
           " runner-up | runner succ | final_d gap |",
           "|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(cells.keys()):
        scenario, sp, rad = key
        ranked = sorted(
            cells[key],
            key=lambda e: (-e["success"], e["final_d"]))
        best = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else best
        gap = runner["final_d"] - best["final_d"]
        out.append(
            f"| {scenario} | {sp:+.2f} | {rad:.2f} | {best['planner']} | "
            f"{fmt_succ(best['success'])} | {best['final_d']:.2f} | "
            f"{runner['planner']} | {fmt_succ(runner['success'])} | "
            f"{gap:+.2f} |")
    out.append("")
    return out


def render_planner_summary(rows: list[dict]) -> list[str]:
    by_planner = defaultdict(list)
    for r in rows:
        by_planner[r["planner"]].append(r)
    out = ["## Per-planner summary across all cells", "",
           "| planner | family | cells | cells solved | mean succ |"
           " mean final_d | mean coll | mean ms |",
           "|---|---|---|---|---|---|---|---|"]
    for planner in sorted(by_planner.keys(), key=planner_sort_key):
        entries = by_planner[planner]
        n = len(entries)
        solved = sum(1 for e in entries if e["success"] >= 0.999)
        succ_mean = mean(e["success"] for e in entries)
        # Mean final_distance is informative even on cells the planner
        # failed -- it captures "how close did you get when you missed?".
        fd_mean = mean(e["final_d"] for e in entries)
        coll_mean = mean(e["collisions"] for e in entries)
        ms_mean = mean(e["ms"] for e in entries)
        out.append(
            f"| {planner} | {planner_family(planner)} | {n} | {solved} | "
            f"{succ_mean:.2f} | {fd_mean:.2f} | {coll_mean:.2f} | {ms_mean:.2f} |")
    out.append("")
    return out


def render_hard_cell_focus(rows: list[dict]) -> list[str]:
    out = ["## Hard-cell focus (speed >= 1.5)", "",
           "Filter cells with dyn_speed_scale >= 1.5 to capture the regime"
           " where the obstacle moves fast enough to force genuine"
           " replanning. Lower bound on success differentiates planners;"
           " mean collisions per cell exposes the paradigm gap for"
           " planners that ignore dynamic obstacles.",
           "",
           "| planner | family | hard cells | hard cells solved |"
           " mean succ | mean final_d | mean coll |",
           "|---|---|---|---|---|---|---|"]
    hard = [r for r in rows if r["speed"] >= 1.5]
    by_planner = defaultdict(list)
    for r in hard:
        by_planner[r["planner"]].append(r)
    for planner in sorted(by_planner.keys(), key=planner_sort_key):
        entries = by_planner[planner]
        n = len(entries)
        solved = sum(1 for e in entries if e["success"] >= 0.999)
        succ_mean = mean(e["success"] for e in entries)
        fd_mean = mean(e["final_d"] for e in entries)
        coll_mean = mean(e["collisions"] for e in entries)
        out.append(
            f"| {planner} | {planner_family(planner)} | {n} | {solved} | "
            f"{succ_mean:.2f} | {fd_mean:.2f} | {coll_mean:.2f} |")
    out.append("")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-csv",
                   default="build/sweep_local_planner_compare_summary.csv")
    p.add_argument("--md-out",
                   default="build/local_planner_compare_report.md")
    args = p.parse_args()

    rows = load_summary(Path(args.summary_csv))
    if not rows:
        raise SystemExit(f"No rows in {args.summary_csv}")

    md = [
        "# Local planner cross-comparison",
        "",
        "DWA, STOMP and Diff-MPPI variants share the same scenario,"
        " bicycle dynamics, cost components and obstacle representation"
        " in benchmark_diff_mppi. Each cell is a"
        " (scenario, dyn_speed_scale, dyn_radius_scale) tuple; success"
        " is averaged across seeds.",
        "",
    ]
    md += render_planner_summary(rows)
    md += render_hard_cell_focus(rows)
    md += render_best_per_cell(rows)
    md += render_per_cell(rows)
    text = "\n".join(md)
    Path(args.md_out).write_text(text + "\n")
    print(text)
    print(f"\nWritten to {args.md_out}")


if __name__ == "__main__":
    main()
