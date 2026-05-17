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
    with open(path) as f:
        for row in csv.DictReader(f):
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
    return by_cell


def fmt_succ(rate):
    return "+" if rate >= 0.999 else ("." if rate == 0.0 else f"{rate:.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-csv",
                   default="build/sweep_grad_horizon_difficulty_summary.csv")
    p.add_argument("--md-out",
                   default="build/sweep_grad_horizon_difficulty.md")
    args = p.parse_args()

    by_cell = load_summary(args.summary_csv)
    cells = sorted(by_cell.keys())

    lines = []
    lines.append("# Gradient-horizon x dynamic-obstacle difficulty sweep")
    lines.append("")
    lines.append("Scenario: dynamic_crossing. K=4096, 4 seeds per cell. "
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

    lines.append("## Take-aways")
    lines.append("")
    lines.append(
        "- **early1 is never enough**: it never reaches the goal across all "
        "18 cells, with a final-distance gap of ~+0.15-1.3 over the full "
        "horizon. One step of gradient update is insufficient to bend the "
        "nominal trajectory around the dynamic obstacle.")
    lines.append(
        "- **early2 covers the easy regime cleanly**: in every cell where "
        "the full horizon succeeds (12 of 18), early2 also reaches the goal "
        "with a final-distance gap within ~0.05 of full and a small "
        "cumulative-cost penalty (~500 of ~44k).")
    lines.append(
        "- **In the hard regime (speed 1.5x) every planner fails to reach "
        "the goal**, but the ordering by final_distance is "
        "`early8 ~= early4 ~= full < early16 < early2 < early1 < mppi`. "
        "early8 is consistently best or tied-best, suggesting ~8 horizon "
        "steps is sufficient and longer windows do not pay off (and "
        "occasionally hurt because of stale-gradient noise at the tail).")
    lines.append(
        "- **Radius scaling alone does little** within these cells; the "
        "speed axis dominates difficulty. The right way to increase "
        "difficulty further is probably to make obstacles converge with the "
        "agent (negative speed at higher magnitude) or stack multiple dyn "
        "obstacles, not just inflate radius.")
    lines.append(
        "- **speed=2.0 is paradoxically easier than 1.5x**: the obstacle "
        "leaves the corridor too fast to matter. The mechanism question "
        "(\"how far does the gradient need to look ahead?\") sees its "
        "strongest signal in the speed=1.5x band where obstacle and agent "
        "are still co-located by the time the agent crosses.")
    lines.append("")

    output = "\n".join(lines)
    print(output)
    with open(args.md_out, "w") as f:
        f.write(output)
    print(f"\nmarkdown -> {args.md_out}")


if __name__ == "__main__":
    main()
