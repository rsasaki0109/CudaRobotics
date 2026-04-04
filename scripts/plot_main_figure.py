#!/usr/bin/env python3
"""Generate the main paper figure: two-panel comparison.

Panel A: dynamic_slalom matched-time results (bar chart)
Panel B: 7-DOF dynamic_avoid compute-quality tradeoff (scatter)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    out_dir = "build/plots_paper"
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ====== Panel A: dynamic_slalom matched-time @ ~1.0 ms ======
    planners_a = [
        "MPPI",
        "FB-faithful",
        "FB-hf",
        "FB-ref",
        "FB-fused",
        "Diff-MPPI-3",
    ]
    # From exact-time full summary (dynamic_slalom @ ~1.0ms)
    success_a = [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]
    dist_a =    [14.16, 14.09, 13.80, 11.84, 10.30, 1.91]
    ms_a =      [0.98, 0.98, 0.98, 1.03, 1.82, 0.98]
    k_a =       [7405, 3294, 292, 1121, 128, 128]

    colors_a = ["#888888", "#bbbbbb", "#6baed6", "#2171b5", "#08519c", "#e6550d"]

    bars = ax1.bar(range(len(planners_a)), dist_a, color=colors_a, edgecolor="black", linewidth=0.5)
    # Highlight success
    for i, s in enumerate(success_a):
        if s >= 1.0:
            bars[i].set_edgecolor("#e6550d")
            bars[i].set_linewidth(2.5)
            ax1.text(i, dist_a[i] + 0.3, "SUCCESS", ha="center", va="bottom",
                     fontsize=8, fontweight="bold", color="#e6550d")

    ax1.set_xticks(range(len(planners_a)))
    ax1.set_xticklabels(planners_a, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("Final Distance", fontsize=10)
    ax1.set_title("(a) dynamic_slalom @ matched ~1.0 ms", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 17)
    ax1.axhline(y=2.0, color="#e6550d", linestyle="--", alpha=0.3, linewidth=0.8)
    ax1.text(5.4, 2.3, "goal tol", fontsize=7, color="#e6550d", alpha=0.5)

    # ====== Panel B: 7-DOF dynamic_avoid compute-quality ======
    # Fixed-budget data: (planner, K, success, dist, ms)
    data_b = [
        ("MPPI",      256, 0.75, 0.273, 0.33),
        ("MPPI",      512, 0.25, 0.635, 0.39),
        ("MPPI",     1024, 0.75, 0.283, 0.49),
        ("FB-ref",    256, 1.00, 0.090, 2.06),
        ("FB-ref",    512, 0.75, 0.283, 4.01),
        ("FB-ref",   1024, 1.00, 0.095, 7.82),
        ("Diff-1",    256, 0.75, 0.268, 0.49),
        ("Diff-1",    512, 0.25, 0.639, 0.54),
        ("Diff-1",   1024, 0.75, 0.268, 0.65),
        ("Diff-3",    256, 0.50, 0.458, 0.79),
        ("Diff-3",    512, 1.00, 0.090, 0.84),
        ("Diff-3",   1024, 0.75, 0.275, 0.96),
    ]

    marker_map = {"MPPI": "s", "FB-ref": "^", "Diff-1": "o", "Diff-3": "D"}
    color_map = {"MPPI": "#888888", "FB-ref": "#2171b5", "Diff-1": "#fdae6b", "Diff-3": "#e6550d"}

    for planner in ["MPPI", "FB-ref", "Diff-1", "Diff-3"]:
        pts = [(ms, succ, dist) for (p, k, succ, dist, ms) in data_b if p == planner]
        ms_vals = [p[0] for p in pts]
        succ_vals = [p[1] for p in pts]
        sizes = [120 if s >= 1.0 else 50 for s in succ_vals]
        alphas = [1.0 if s >= 1.0 else 0.4 for s in succ_vals]
        for i in range(len(pts)):
            ax2.scatter(ms_vals[i], succ_vals[i], s=sizes[i], alpha=alphas[i],
                       marker=marker_map[planner], color=color_map[planner],
                       edgecolors="black", linewidths=0.5,
                       label=planner if i == 0 else None, zorder=3)

    ax2.set_xlabel("Avg Control Time (ms)", fontsize=10)
    ax2.set_ylabel("Success Rate", fontsize=10)
    ax2.set_title("(b) 7-DOF dynamic_avoid, fixed budget", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 9)
    ax2.set_ylim(-0.05, 1.15)
    ax2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
    ax2.legend(fontsize=8, loc="center right")

    # Annotate the key point: Diff-3 K=512
    ax2.annotate("K=512\nsuccess=1.0\n0.84 ms",
                xy=(0.84, 1.00), xytext=(2.5, 0.85),
                fontsize=7, color="#e6550d",
                arrowprops=dict(arrowstyle="->", color="#e6550d", lw=1.2),
                fontweight="bold")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"main_figure.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close()


if __name__ == "__main__":
    main()
