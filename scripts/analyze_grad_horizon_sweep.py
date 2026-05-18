#!/usr/bin/env python3
"""Produce a compact Markdown table from the gradient-horizon × difficulty
sweep summary CSV. Highlights:
  - per (speed, radius) cell which planners succeed
  - where early2 already matches full horizon vs where it does not
  - regime where every planner fails (so final_distance is the signal)
"""

import argparse
import csv
from collections import defaultdict


PLANNER_ORDER = [
    "mppi",
    "diff_mppi_3_early1",
    "diff_mppi_3_early2",
    "diff_mppi_3_early4",
    "diff_mppi_3_early8",
    "diff_mppi_3_early16",
    "diff_mppi_3",
]
PLANNER_LABEL = {
    "mppi": "mppi",
    "diff_mppi_3_early1": "early1",
    "diff_mppi_3_early2": "early2",
    "diff_mppi_3_early4": "early4",
    "diff_mppi_3_early8": "early8",
    "diff_mppi_3_early16": "early16",
    "diff_mppi_3": "full",
}


def load_summary(path):
    by_cell = defaultdict(dict)
    scenarios = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            scenarios.add(row["scenario"])
            cell = (float(row["dyn_speed_scale"]),
                    float(row["dyn_radius_scale"]))
            by_cell[cell][row["planner"]] = {
                "success_rate": float(row["success_rate"]),
                "final_distance": float(row["final_distance"]),
                "min_goal_distance": float(row["min_goal_distance"]),
                "cumulative_cost": float(row["cumulative_cost"]),
                "collisions": float(row["collisions"]),
                "avg_control_ms": float(row["avg_control_ms"]),
            }
    return by_cell, sorted(scenarios)


def fmt_succ(rate):
    return "+" if rate >= 0.999 else ("." if rate == 0.0 else f"{rate:.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-csv",
                   default="build/sweep_grad_horizon_difficulty_summary.csv")
    p.add_argument("--md-out",
                   default="build/sweep_grad_horizon_difficulty.md")
    args = p.parse_args()

    by_cell, scenarios = load_summary(args.summary_csv)
    cells = sorted(by_cell.keys())

    lines = []
    lines.append("# Gradient-horizon x dynamic-obstacle difficulty sweep")
    lines.append("")
    scenario_text = ", ".join(scenarios) if scenarios else "(unknown)"
    lines.append(f"Scenario: {scenario_text}. K=4096, 4 seeds per cell. "
                 "Difficulty axes: dyn-obstacle speed scale (vx/vy multiplier) "
                 "and radius scale.")
    lines.append("")

    lines.append("## Success rate (1.00 = all 4 seeds reached goal)")
    lines.append("")
    header = ["speed", "radius"] + [PLANNER_LABEL[p] for p in PLANNER_ORDER]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for (sp, rad) in cells:
        cell = by_cell[(sp, rad)]
        row = [f"{sp:+.1f}", f"{rad:.1f}"]
        for pl in PLANNER_ORDER:
            r = cell.get(pl, {}).get("success_rate", 0.0)
            row.append(fmt_succ(r))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Final distance to goal (mean over 4 seeds; lower = better)")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for (sp, rad) in cells:
        cell = by_cell[(sp, rad)]
        row = [f"{sp:+.1f}", f"{rad:.1f}"]
        for pl in PLANNER_ORDER:
            d = cell.get(pl, {}).get("final_distance", float("nan"))
            row.append(f"{d:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Regime classification
    lines.append("## Regime classification")
    lines.append("")
    lines.append("Per cell we ask three questions:")
    lines.append(" - Does **early2** already match full-horizon success (within +0.05)?")
    lines.append(" - If everyone fails, is **early8** the closest to the goal?")
    lines.append(" - What is the gap between early1 and full in final_distance?")
    lines.append("")
    lines.append("| speed | radius | regime | early2 vs full (final_d gap) "
                 "| best by final_d | early1 - full gap |")
    lines.append("|---|---|---|---|---|---|")
    for (sp, rad) in cells:
        cell = by_cell[(sp, rad)]
        full = cell.get("diff_mppi_3", {})
        e1 = cell.get("diff_mppi_3_early1", {})
        e2 = cell.get("diff_mppi_3_early2", {})
        if not full or not e1 or not e2:
            continue
        if full["success_rate"] >= 0.999 and e2["success_rate"] >= 0.999:
            regime = "easy (e2 OK)"
        elif full["success_rate"] >= 0.999 and e2["success_rate"] < 0.999:
            regime = "needs e4+"
        elif full["success_rate"] < 0.999:
            regime = "all fail (compare final_d)"
        else:
            regime = "?"
        gap_e2 = e2["final_distance"] - full["final_distance"]
        # best by final_distance among diff_* planners
        cand = [(pl, cell[pl]["final_distance"])
                for pl in PLANNER_ORDER if pl != "mppi" and pl in cell]
        best = min(cand, key=lambda x: x[1])
        gap_e1 = e1["final_distance"] - full["final_distance"]
        lines.append(
            f"| {sp:+.1f} | {rad:.1f} | {regime} | {gap_e2:+.2f} "
            f"| {PLANNER_LABEL[best[0]]} ({best[1]:.2f}) | {gap_e1:+.2f} |")
    lines.append("")

    lines.append("## Take-aways (data-driven)")
    lines.append("")
    succ_by_planner = {pl: 0 for pl in PLANNER_ORDER}
    cells_full_success = 0
    cells_e2_matches_full = 0
    hard_regime_cells = []
    for (sp, rad) in cells:
        cell = by_cell[(sp, rad)]
        for pl in PLANNER_ORDER:
            if cell.get(pl, {}).get("success_rate", 0) >= 0.999:
                succ_by_planner[pl] += 1
        full = cell.get("diff_mppi_3", {})
        e2 = cell.get("diff_mppi_3_early2", {})
        if full.get("success_rate", 0) >= 0.999:
            cells_full_success += 1
            if e2.get("success_rate", 0) >= 0.999:
                cells_e2_matches_full += 1
        if all(cell.get(pl, {}).get("success_rate", 0) < 0.999
               for pl in PLANNER_ORDER if pl != "mppi"):
            hard_regime_cells.append((sp, rad))

    total_cells = len(cells)
    lines.append(f"- Out of {total_cells} cells, success counts by planner: " +
                 ", ".join(f"{PLANNER_LABEL[pl]}={succ_by_planner[pl]}"
                           for pl in PLANNER_ORDER) + ".")
    if cells_full_success > 0:
        lines.append(
            f"- Full-horizon succeeds in {cells_full_success}/"
            f"{total_cells} cells; early2 matches full in "
            f"{cells_e2_matches_full}/{cells_full_success} of those.")
    if hard_regime_cells:
        # In the all-fail regime, who is closest to the goal on average?
        from collections import Counter
        best_counter = Counter()
        for (sp, rad) in hard_regime_cells:
            cell = by_cell[(sp, rad)]
            cand = [(pl, cell[pl]["final_distance"])
                    for pl in PLANNER_ORDER
                    if pl != "mppi" and pl in cell]
            best = min(cand, key=lambda x: x[1])[0]
            best_counter[best] += 1
        ranking = ", ".join(
            f"{PLANNER_LABEL[pl]}={n}"
            for pl, n in best_counter.most_common())
        lines.append(
            f"- Hard regime (all diff_mppi variants fail success): "
            f"{len(hard_regime_cells)} cells. Best by final_distance: "
            f"{ranking}.")
    lines.append("")

    output = "\n".join(lines)
    print(output)
    with open(args.md_out, "w") as f:
        f.write(output)
    print(f"\nmarkdown -> {args.md_out}")


if __name__ == "__main__":
    main()
