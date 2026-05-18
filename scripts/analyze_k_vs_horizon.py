#!/usr/bin/env python3
"""Render Markdown heatmaps from a K x horizon sweep summary CSV.

Inputs are one or more summary CSVs produced by sweep_k_vs_horizon.py.
Output: a Markdown report with success-rate grid and final-distance
grid for each (scenario, speed, radius) cell, plus a small
"substitution" analysis that asks whether higher K can recover the
quality of a longer horizon.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.horizon_selection.horizon_naming import (
    FULL_HORIZON_SENTINEL,
    parse_grad_update_horizon,
)


PLANNER_LABEL = {
    "diff_mppi_3_early1": "early1",
    "diff_mppi_3_early2": "early2",
    "diff_mppi_3_early4": "early4",
    "diff_mppi_3_early8": "early8",
    "diff_mppi_3_early16": "early16",
    "diff_mppi_3": "full",
}


def horizon_of(planner: str, full_steps: int) -> int:
    raw = parse_grad_update_horizon(planner)
    if raw is None:
        return -1
    return full_steps if raw == FULL_HORIZON_SENTINEL else raw


def load_cells(paths: list[Path]) -> dict[tuple[str, float, float],
                                          list[dict]]:
    cells: dict[tuple[str, float, float], list[dict]] = {}
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                key = (r["scenario"], float(r["dyn_speed_scale"]),
                       float(r["dyn_radius_scale"]))
                cells.setdefault(key, []).append(r)
    return cells


def fmt_succ(value: float) -> str:
    if value >= 0.999:
        return "+"
    if value <= 0.0:
        return "."
    return f"{value:.2f}"


def render_cell(key, rows, full_steps: int) -> list[str]:
    scenario, sp, rad = key
    planners = sorted({r["planner"] for r in rows},
                      key=lambda p: horizon_of(p, full_steps))
    ks = sorted({int(r["k_samples"]) for r in rows})
    succ = {(r["planner"], int(r["k_samples"])): float(r["success_rate"])
            for r in rows}
    fd = {(r["planner"], int(r["k_samples"])):
          float(r["final_distance"]) for r in rows}
    ms = {(r["planner"], int(r["k_samples"])):
          float(r["avg_control_ms"]) for r in rows}

    out = []
    out.append(f"## {scenario} | speed={sp:+.2f} radius={rad:.2f}")
    out.append("")
    out.append("### Success-rate grid")
    out.append("")
    out.append("| planner | h | " +
               " | ".join(f"K={k}" for k in ks) + " |")
    out.append("|" + "|".join(["---"] * (2 + len(ks))) + "|")
    for p in planners:
        h = horizon_of(p, full_steps)
        out.append("| " + PLANNER_LABEL.get(p, p) + f" | {h} | "
                   + " | ".join(fmt_succ(succ.get((p, k), 0))
                                for k in ks) + " |")
    out.append("")
    out.append("### Final-distance grid (lower = better)")
    out.append("")
    out.append("| planner | h | " +
               " | ".join(f"K={k}" for k in ks) + " |")
    out.append("|" + "|".join(["---"] * (2 + len(ks))) + "|")
    for p in planners:
        h = horizon_of(p, full_steps)
        out.append("| " + PLANNER_LABEL.get(p, p) + f" | {h} | "
                   + " | ".join(f"{fd.get((p, k), float('nan')):.2f}"
                                for k in ks) + " |")
    out.append("")
    out.append("### avg_control_ms grid")
    out.append("")
    out.append("| planner | h | " +
               " | ".join(f"K={k}" for k in ks) + " |")
    out.append("|" + "|".join(["---"] * (2 + len(ks))) + "|")
    for p in planners:
        h = horizon_of(p, full_steps)
        out.append("| " + PLANNER_LABEL.get(p, p) + f" | {h} | "
                   + " | ".join(f"{ms.get((p, k), float('nan')):.2f}"
                                for k in ks) + " |")
    out.append("")

    # Substitution analysis: ask whether K compensates for shorter
    # horizon (i.e. does final_distance decrease as K grows, for a
    # given planner?), and what the best (K, planner) overall is.
    out.append("### Substitution analysis")
    out.append("")
    best_fd = min(fd.values())
    best_pk = min(fd.items(), key=lambda kv: kv[1])[0]
    out.append(
        f"Overall best final_d in this cell: **{best_fd:.2f}** at "
        f"{PLANNER_LABEL.get(best_pk[0], best_pk[0])} (h="
        f"{horizon_of(best_pk[0], full_steps)}, K={best_pk[1]}).")
    out.append("")
    out.append("Per-planner: does increasing K reduce final_distance, "
               "and can the planner match the overall best?")
    out.append("")
    out.append("| planner | h | min final_d (K) | max final_d (K) "
               "| range | matches best within 0.5? |")
    out.append("|---|---|---|---|---|---|")
    for p in planners:
        per_k = [(k, fd[(p, k)]) for k in ks if (p, k) in fd]
        if not per_k:
            continue
        kmin, vmin = min(per_k, key=lambda kv: kv[1])
        kmax, vmax = max(per_k, key=lambda kv: kv[1])
        rng = vmax - vmin
        matches = "yes" if vmin <= best_fd + 0.5 else "no"
        out.append(
            f"| {PLANNER_LABEL.get(p, p)} | "
            f"{horizon_of(p, full_steps)} | "
            f"{vmin:.2f} (K={kmin}) | {vmax:.2f} (K={kmax}) | "
            f"{rng:.2f} | {matches} |")
    out.append("")
    out.append("Interpretation: a small `range` and `matches best = no` "
               "means K cannot compensate for that horizon; the cell is "
               "horizon-limited rather than sample-limited.")
    out.append("")

    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-csv", nargs="+", required=True,
                   help="One or more summary CSVs from sweep_k_vs_horizon")
    p.add_argument("--full-horizon-steps", type=int, default=30)
    p.add_argument("--md-out", default="build/k_vs_horizon_report.md")
    args = p.parse_args()

    cells = load_cells([Path(p) for p in args.summary_csv])
    lines = ["# K_samples x gradient-update-horizon sweep", ""]
    for key in sorted(cells.keys()):
        lines.extend(render_cell(key, cells[key], args.full_horizon_steps))

    text = "\n".join(lines)
    print(text)
    Path(args.md_out).write_text(text + "\n")
    print(f"\nWritten to {args.md_out}")


if __name__ == "__main__":
    main()
