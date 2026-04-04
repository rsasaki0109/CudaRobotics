#!/usr/bin/env python3
"""Plot compute-quality Pareto frontier for dynamic_slalom.

Shows that non-hybrid methods have a hard ceiling (success=0.00 at any K),
while diff_mppi breaks through this ceiling.
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from collections import defaultdict


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def aggregate(rows):
    """Group by (scenario, planner, k_samples) and average."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["scenario"], r["planner"], int(r["k_samples"]))
        groups[key].append(r)
    result = []
    for (scenario, planner, k), rr in groups.items():
        n = len(rr)
        result.append({
            "scenario": scenario,
            "planner": planner,
            "k_samples": k,
            "success": sum(float(r["success"]) for r in rr) / n,
            "final_distance": sum(float(r["final_distance"]) for r in rr) / n,
            "avg_control_ms": sum(float(r["avg_control_ms"]) for r in rr) / n,
        })
    return result


def main():
    out_dir = "build/plots_paper"
    os.makedirs(out_dir, exist_ok=True)

    # Load the full benchmark CSV (all planners, all K values)
    csv_path = "build/benchmark_diff_mppi_exact_time_full.csv"
    if not os.path.exists(csv_path):
        # Fall back to the standard benchmark
        csv_path = "build/benchmark_diff_mppi_faithful.csv"

    # Also load the wall-clock sweep if available
    extra_csvs = [
        "build/benchmark_diff_mppi_exact_time_final.csv",
        "build/benchmark_diff_mppi_faithful.csv",
        "build/benchmark_diff_mppi_faithful_large_k.csv",
    ]

    all_rows = []
    for path in [csv_path] + extra_csvs:
        if os.path.exists(path):
            all_rows.extend(load_csv(path))

    agg = aggregate(all_rows)

    # Filter to dynamic_slalom
    slalom = [r for r in agg if r["scenario"] == "dynamic_slalom"]

    if not slalom:
        print("No dynamic_slalom data found")
        return

    # Categorize planners
    hybrid = ["diff_mppi_1", "diff_mppi_3"]
    non_hybrid_strong = ["feedback_mppi_ref", "feedback_mppi_fused", "feedback_mppi_hf",
                         "feedback_mppi_cov", "feedback_mppi_faithful"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Panel A: Success vs Compute (Pareto) ----
    color_map = {
        "mppi": "#888888",
        "feedback_mppi_ref": "#2171b5",
        "feedback_mppi_fused": "#08519c",
        "feedback_mppi_hf": "#6baed6",
        "feedback_mppi_cov": "#9ecae1",
        "feedback_mppi_faithful": "#bbbbbb",
        "feedback_mppi_release": "#c6dbef",
        "feedback_mppi_sens": "#deebf7",
        "diff_mppi_1": "#fdae6b",
        "diff_mppi_3": "#e6550d",
        "grad_only_3": "#fee6ce",
    }
    marker_map = {
        "mppi": "s", "diff_mppi_1": "o", "diff_mppi_3": "D",
    }

    for r in slalom:
        p = r["planner"]
        color = color_map.get(p, "#aaaaaa")
        marker = marker_map.get(p, "^")
        is_hybrid = p in hybrid
        size = 100 if is_hybrid else 60
        alpha = 1.0 if r["success"] > 0.5 else 0.5
        edge = "#e6550d" if r["success"] > 0.5 else "gray"
        ax1.scatter(r["avg_control_ms"], r["success"], s=size, alpha=alpha,
                   marker=marker, color=color, edgecolors=edge, linewidths=1.0, zorder=3)

    # Draw the non-hybrid ceiling
    ax1.axhline(y=0.0, color="gray", linestyle="-", alpha=0.3, linewidth=8, zorder=1)
    ax1.text(1.5, 0.03, "Non-hybrid ceiling: success=0.00 at any compute budget",
             fontsize=8, color="gray", style="italic")

    # Draw the hybrid breakthrough
    ax1.axhline(y=1.0, color="#e6550d", linestyle="--", alpha=0.3, linewidth=1)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888888", markersize=8, label="MPPI"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2171b5", markersize=8, label="Feedback baselines"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#e6550d", markersize=10, label="Diff-MPPI-3"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#fdae6b", markersize=8, label="Diff-MPPI-1"),
    ]
    ax1.legend(handles=legend_elements, fontsize=8, loc="center right")

    ax1.set_xlabel("Avg Control Time (ms)", fontsize=10)
    ax1.set_ylabel("Success Rate", fontsize=10)
    ax1.set_title("(a) dynamic_slalom: Pareto frontier", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.08, 1.15)
    ax1.set_xlim(0, 4.5)

    # ---- Panel B: Final Distance vs Compute ----
    for r in slalom:
        p = r["planner"]
        color = color_map.get(p, "#aaaaaa")
        marker = marker_map.get(p, "^")
        is_hybrid = p in hybrid
        size = 100 if is_hybrid else 60
        alpha = 1.0 if r["success"] > 0.5 else 0.4
        edge = "#e6550d" if r["success"] > 0.5 else "gray"
        ax2.scatter(r["avg_control_ms"], r["final_distance"], s=size, alpha=alpha,
                   marker=marker, color=color, edgecolors=edge, linewidths=1.0, zorder=3)

    # Draw the non-hybrid floor
    ax2.axhline(y=10.3, color="#08519c", linestyle=":", alpha=0.4, linewidth=1)
    ax2.text(0.1, 10.6, "Best non-hybrid (fused): ~10.3", fontsize=7, color="#08519c")

    ax2.axhline(y=2.0, color="#e6550d", linestyle="--", alpha=0.3, linewidth=1)
    ax2.text(0.1, 2.3, "Diff-MPPI: ~1.9 (goal reached)", fontsize=7, color="#e6550d")

    ax2.set_xlabel("Avg Control Time (ms)", fontsize=10)
    ax2.set_ylabel("Final Distance", fontsize=10)
    ax2.set_title("(b) dynamic_slalom: distance vs compute", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 16)
    ax2.set_xlim(0, 4.5)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"pareto_frontier.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close()


if __name__ == "__main__":
    main()
