#!/usr/bin/env python3
"""Generate all paper figures for Diff-MPPI submission.

Usage:
    python3 scripts/generate_paper_figures.py [--csv PATH] [--pareto-csv PATH]
                                            [--trace-dir PATH] [--out-dir PATH]

Defaults:
    --csv       build/benchmark_diff_mppi_exact_time_full.csv
    --pareto-csv  (unset; falls back to --csv data)
    --trace-dir build/
    --out-dir   paper/figures/
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# === Publication settings ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

SINGLE_COL = 3.5  # inches
DOUBLE_COL = 7.0  # inches

COLORS = {
    "mppi": "#1f77b4",
    "feedback_mppi": "#aec7e8",
    "feedback_mppi_ref": "#ff7f0e",
    "feedback_mppi_cov": "#2ca02c",
    "feedback_mppi_fused": "#d62728",
    "feedback_mppi_hf": "#8c564b",
    "feedback_mppi_faithful": "#e377c2",
    "feedback_mppi_paper": "#ff9896",
    "diff_mppi_1": "#9467bd",
    "diff_mppi_3": "#17becf",
    "grad_only_3": "#7f7f7f",
    "step_mppi": "#bcbd22",
}

MARKERS = {
    "mppi": "o",
    "feedback_mppi_ref": "s",
    "feedback_mppi_cov": "^",
    "feedback_mppi_fused": "D",
    "feedback_mppi_hf": "p",
    "feedback_mppi_faithful": "h",
    "feedback_mppi_paper": "<",
    "diff_mppi_1": "P",
    "diff_mppi_3": "*",
    "grad_only_3": "x",
    "step_mppi": "v",
}

LABELS = {
    "mppi": "MPPI",
    "feedback_mppi": "Feedback-MPPI",
    "feedback_mppi_ref": "Feedback-MPPI (ref)",
    "feedback_mppi_cov": "Feedback-MPPI (cov)",
    "feedback_mppi_fused": "Feedback-MPPI (fused)",
    "feedback_mppi_hf": "Feedback-MPPI (HF)",
    "feedback_mppi_faithful": "Feedback-MPPI (faithful)",
    "feedback_mppi_paper": "Feedback-MPPI (paper)",
    "diff_mppi_1": "Diff-MPPI-1",
    "diff_mppi_3": "Diff-MPPI-3",
    "grad_only_3": "Grad-only-3",
    "step_mppi": "Step-MPPI",
}

HYBRID_PLANNERS = {"diff_mppi_1", "diff_mppi_3"}

TRACE_FLOAT_FIELDS = {
    "alpha", "goal_distance", "min_obstacle_margin", "control_ms",
    "sampled_accel", "sampled_steer", "final_accel", "final_steer",
    "delta_accel", "delta_steer", "delta_norm",
    "grad_accel", "grad_steer", "grad_norm",
}

TRACE_INT_FIELDS = {"seed", "k_samples", "grad_steps", "episode_step", "horizon_step"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path):
    """Load a CSV file and return a list of dicts (all values as strings)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def aggregate(rows):
    """Group by (scenario, planner, k_samples) and average numeric metrics."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["scenario"], r["planner"], int(r["k_samples"]))
        groups[key].append(r)
    result = []
    for (scenario, planner, k), rr in groups.items():
        n = len(rr)
        entry = {
            "scenario": scenario,
            "planner": planner,
            "k_samples": k,
            "n_seeds": n,
        }
        # Average all numeric fields that exist
        for field in ["success", "final_distance", "avg_control_ms",
                      "cumulative_cost", "collisions", "min_goal_distance",
                      "reached_goal", "collision_free"]:
            vals = []
            for r in rr:
                if field in r:
                    try:
                        vals.append(float(r[field]))
                    except (ValueError, TypeError):
                        pass
            if vals:
                entry[field] = sum(vals) / len(vals)
                # Also compute std for error bars
                mean_val = entry[field]
                if len(vals) > 1:
                    entry[field + "_std"] = (
                        sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)
                    ) ** 0.5
                else:
                    entry[field + "_std"] = 0.0
        result.append(entry)
    return result


def load_trace_rows(path):
    """Load trace CSV with typed fields."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            for key in TRACE_INT_FIELDS:
                if key in parsed:
                    parsed[key] = int(float(parsed[key]))
            for key in TRACE_FLOAT_FIELDS:
                if key in parsed:
                    parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def _mean(values):
    """Safe mean."""
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Figure: Pareto frontier (main result)
# ---------------------------------------------------------------------------

def fig_pareto(data, out_dir):
    """Double-column Pareto plot: compute vs. quality for two scenarios."""
    scenarios = ["dynamic_crossing", "dynamic_slalom"]
    scenario_titles = {
        "dynamic_crossing": "Dynamic Crossing",
        "dynamic_slalom": "Dynamic Slalom",
    }

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6))

    for ax, scenario in zip(axes, scenarios):
        sc_data = [r for r in data if r["scenario"] == scenario]
        if not sc_data:
            ax.set_title(scenario_titles.get(scenario, scenario))
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="gray")
            continue

        # Determine non-hybrid ceiling (best final_distance among non-hybrid)
        non_hybrid = [r for r in sc_data if r["planner"] not in HYBRID_PLANNERS
                      and "final_distance" in r]
        if non_hybrid:
            ceiling = min(r["final_distance"] for r in non_hybrid)
        else:
            ceiling = None

        # Plot each planner variant
        plotted_planners = set()
        for r in sc_data:
            p = r["planner"]
            if "final_distance" not in r or "avg_control_ms" not in r:
                continue
            color = COLORS.get(p, "#aaaaaa")
            marker = MARKERS.get(p, "o")
            is_hybrid = p in HYBRID_PLANNERS
            size = 80 if is_hybrid else 40
            zorder = 5 if is_hybrid else 3
            alpha = 0.9 if is_hybrid else 0.6
            ax.scatter(
                r["avg_control_ms"], r["final_distance"],
                s=size, alpha=alpha, marker=marker, color=color,
                edgecolors="k" if is_hybrid else "gray",
                linewidths=0.5, zorder=zorder,
            )
            plotted_planners.add(p)

        # Draw non-hybrid ceiling line and shaded region
        if ceiling is not None:
            ax.axhline(y=ceiling, color="gray", linestyle="--",
                       alpha=0.5, linewidth=0.8, zorder=2)
            xlim = ax.get_xlim()
            ax.fill_between(
                [xlim[0], xlim[1] * 1.5], ceiling, ax.get_ylim()[1] * 1.2,
                alpha=0.06, color="gray", zorder=1,
            )
            ax.text(
                0.97, 0.97,
                "Non-hybrid\nceiling",
                transform=ax.transAxes, fontsize=6, color="gray",
                ha="right", va="top", style="italic",
            )
            ax.set_xlim(xlim)

        ax.set_xlabel("Avg. control time (ms)")
        ax.set_ylabel("Final distance to goal")
        ax.set_title(scenario_titles.get(scenario, scenario), fontweight="bold")
        ax.grid(True, alpha=0.2, linewidth=0.5)

    # Build unified legend
    legend_elements = []
    all_planners_in_data = {r["planner"] for r in data}
    for p in ["mppi", "feedback_mppi_ref", "feedback_mppi_paper",
              "feedback_mppi_cov", "feedback_mppi_fused",
              "feedback_mppi_hf", "feedback_mppi_faithful",
              "diff_mppi_1", "diff_mppi_3", "grad_only_3", "step_mppi"]:
        if p in all_planners_in_data:
            legend_elements.append(
                Line2D([0], [0], marker=MARKERS.get(p, "o"), color="w",
                       markerfacecolor=COLORS.get(p, "#aaa"),
                       markersize=6 if p in HYBRID_PLANNERS else 5,
                       label=LABELS.get(p, p))
            )
    if legend_elements:
        fig.legend(
            handles=legend_elements, loc="upper center",
            ncol=min(len(legend_elements), 5), frameon=False,
            bbox_to_anchor=(0.5, 1.12), fontsize=6,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(out_dir, "fig_pareto.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure: Mechanism analysis (correction profiles)
# ---------------------------------------------------------------------------

def fig_mechanism(trace_dir, out_dir):
    """Single-column mechanism figure: correction magnitude profiles."""
    # Search for trace CSV files
    trace_path = None
    candidates = [
        os.path.join(trace_dir, "benchmark_diff_mppi_mechanism_trace.csv"),
        os.path.join(trace_dir, "benchmark_diff_mppi_freshness_trace.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            trace_path = c
            break

    # Also try globbing for any trace*.csv
    if trace_path is None:
        for fname in sorted(os.listdir(trace_dir)):
            if "trace" in fname and fname.endswith(".csv"):
                trace_path = os.path.join(trace_dir, fname)
                break

    if trace_path is None:
        print("  [SKIP] fig_mechanism: no trace CSV files found in " + trace_dir)
        return None

    print("  Using trace file: " + trace_path)
    trace_rows = load_trace_rows(trace_path)

    # Prefer dynamic_slalom; fall back to whatever scenario is available
    scenarios_available = sorted(set(r["scenario"] for r in trace_rows))
    if "dynamic_slalom" in scenarios_available:
        scenario = "dynamic_slalom"
    elif scenarios_available:
        scenario = scenarios_available[0]
    else:
        print("  [SKIP] fig_mechanism: no data in trace file")
        return None

    trace_rows = [r for r in trace_rows if r["scenario"] == scenario]
    print("  Scenario: " + scenario + " (" + str(len(trace_rows)) + " rows)")

    # Aggregate by episode step
    ep_groups = defaultdict(list)
    for row in trace_rows:
        ep_groups[(row["planner"], row["episode_step"])].append(row)

    episode_data = defaultdict(list)
    for (planner, ep_step), rows in sorted(ep_groups.items()):
        first = [r for r in rows if r["horizon_step"] == 0]
        episode_data[planner].append({
            "episode_step": ep_step,
            "first_delta_norm": _mean([r["delta_norm"] for r in first]) if first else 0.0,
            "mean_delta_norm": _mean([r["delta_norm"] for r in rows]),
        })

    # Aggregate by horizon step
    hz_groups = defaultdict(list)
    for row in trace_rows:
        hz_groups[(row["planner"], row["horizon_step"])].append(row)

    horizon_data = defaultdict(list)
    for (planner, hz_step), rows in sorted(hz_groups.items()):
        horizon_data[planner].append({
            "horizon_step": hz_step,
            "mean_delta_norm": _mean([r["delta_norm"] for r in rows]),
            "mean_grad_norm": _mean([r["grad_norm"] for r in rows]),
        })

    planners_in_data = sorted(episode_data.keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE_COL, 3.8),
                                    constrained_layout=True)

    # Top: correction magnitude vs episode step
    for planner in planners_in_data:
        rows = episode_data[planner]
        color = COLORS.get(planner, "#444")
        marker = MARKERS.get(planner, "o")
        label = LABELS.get(planner, planner)
        xs = [r["episode_step"] for r in rows]
        ys = [r["first_delta_norm"] for r in rows]
        ax1.plot(xs, ys, label=label, color=color, marker=marker,
                 linewidth=1.0, markersize=2.5)

    ax1.set_xlabel("Episode step")
    ax1.set_ylabel("Correction magnitude")
    ax1.set_title("(a) Correction vs. episode step", fontweight="bold")
    ax1.grid(True, alpha=0.2, linewidth=0.5)
    ax1.legend(frameon=False, fontsize=5, loc="best")

    # Bottom: correction magnitude vs horizon position
    for planner in planners_in_data:
        rows = horizon_data[planner]
        color = COLORS.get(planner, "#444")
        marker = MARKERS.get(planner, "o")
        label = LABELS.get(planner, planner)
        xs = [r["horizon_step"] for r in rows]
        ys = [r["mean_delta_norm"] for r in rows]
        ax2.plot(xs, ys, label=label, color=color, marker=marker,
                 linewidth=1.0, markersize=2.5)

    ax2.set_xlabel("Horizon position")
    ax2.set_ylabel("Mean correction magnitude")
    ax2.set_title("(b) Correction vs. horizon position", fontweight="bold")
    ax2.grid(True, alpha=0.2, linewidth=0.5)
    ax2.legend(frameon=False, fontsize=5, loc="best")

    path = os.path.join(out_dir, "fig_mechanism.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure: 7-DOF manipulator results
# ---------------------------------------------------------------------------

def fig_7dof(data, out_dir):
    """Single-column grouped bar chart for 7-DOF scenarios."""
    # Filter for 7dof scenarios
    dof_data = [r for r in data if "7dof" in r["scenario"]]

    # If not in main CSV, try the dedicated file
    if not dof_data:
        alt_path = os.path.join("build", "benchmark_diff_mppi_manipulator_7dof.csv")
        if os.path.exists(alt_path):
            print("  Loading 7dof data from " + alt_path)
            raw = load_csv(alt_path)
            dof_data = aggregate(raw)
            dof_data = [r for r in dof_data if "7dof" in r.get("scenario", "")]
        else:
            print("  [SKIP] fig_7dof: no 7dof data found")
            return None

    if not dof_data:
        print("  [SKIP] fig_7dof: no 7dof scenarios after filtering")
        return None

    scenarios = sorted(set(r["scenario"] for r in dof_data))
    planners = sorted(set(r["planner"] for r in dof_data))

    # For each scenario, pick the best K for each planner (highest success)
    best_by_planner = {}
    for r in dof_data:
        key = (r["scenario"], r["planner"])
        if key not in best_by_planner or r.get("success", 0) > best_by_planner[key].get("success", 0):
            best_by_planner[key] = r

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))

    n_scenarios = len(scenarios)
    n_planners = len(planners)
    bar_width = 0.7 / max(n_planners, 1)
    x_base = list(range(n_scenarios))

    for i, planner in enumerate(planners):
        successes = []
        errors = []
        times = []
        for scenario in scenarios:
            key = (scenario, planner)
            r = best_by_planner.get(key)
            if r:
                successes.append(r.get("success", 0))
                errors.append(r.get("success_std", 0))
                times.append(r.get("avg_control_ms", 0))
            else:
                successes.append(0)
                errors.append(0)
                times.append(0)

        x_pos = [x + (i - n_planners / 2 + 0.5) * bar_width for x in x_base]
        color = COLORS.get(planner, "#aaa")
        label = LABELS.get(planner, planner)
        bars = ax.bar(x_pos, successes, bar_width * 0.9, yerr=errors,
                      color=color, label=label, edgecolor="white",
                      linewidth=0.3, capsize=2, error_kw={"linewidth": 0.5})

        # Annotate with control time
        for x, s, t in zip(x_pos, successes, times):
            if t > 0:
                ax.text(x, s + 0.04, f"{t:.1f}ms", ha="center", va="bottom",
                        fontsize=4.5, rotation=45)

    ax.set_xticks(x_base)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=6)
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.25)
    ax.set_title("7-DOF Manipulator", fontweight="bold")
    ax.legend(frameon=False, fontsize=5, loc="upper right", ncol=2)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig_7dof.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure: Ablation study
# ---------------------------------------------------------------------------

def fig_ablation(data, out_dir):
    """Single-column ablation bar chart for dynamic_slalom."""
    ablation_planners = ["mppi", "grad_only_3", "diff_mppi_1", "diff_mppi_3"]
    slalom = [r for r in data if r["scenario"] == "dynamic_slalom"
              and r["planner"] in ablation_planners]

    if not slalom:
        print("  [SKIP] fig_ablation: no dynamic_slalom data for ablation planners")
        return None

    # Pick a median K value
    k_values = sorted(set(r["k_samples"] for r in slalom))
    median_k = k_values[len(k_values) // 2]

    # Filter to that K, but if a planner doesn't have it, use closest
    plot_data = {}
    for p in ablation_planners:
        planner_rows = [r for r in slalom if r["planner"] == p]
        if not planner_rows:
            continue
        # Try exact K match first
        exact = [r for r in planner_rows if r["k_samples"] == median_k]
        if exact:
            plot_data[p] = exact[0]
        else:
            # Use closest K
            closest = min(planner_rows,
                          key=lambda r: abs(r["k_samples"] - median_k))
            plot_data[p] = closest

    if not plot_data:
        print("  [SKIP] fig_ablation: no matching data at K=" + str(median_k))
        return None

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))

    x_pos = list(range(len(plot_data)))
    planners_ordered = [p for p in ablation_planners if p in plot_data]
    distances = [plot_data[p].get("final_distance", 0) for p in planners_ordered]
    colors_list = [COLORS.get(p, "#aaa") for p in planners_ordered]
    labels_list = [LABELS.get(p, p) for p in planners_ordered]

    bars = ax.bar(x_pos, distances, color=colors_list, edgecolor="white",
                  linewidth=0.3, width=0.6)

    # Add success rate annotation
    for i, p in enumerate(planners_ordered):
        success = plot_data[p].get("success", 0)
        ax.text(i, distances[i] + 0.3, f"SR={success:.0%}",
                ha="center", va="bottom", fontsize=6, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_list, fontsize=6, rotation=15, ha="right")
    ax.set_ylabel("Final distance to goal")
    ax.set_title(f"Ablation (dynamic_slalom, K={median_k})", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig_ablation.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure: Scenario schematics
# ---------------------------------------------------------------------------

def fig_scenarios(out_dir):
    """Double-column schematic diagrams of benchmark scenarios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4))

    workspace = (0, 0, 50, 30)  # x_min, y_min, x_max, y_max

    for ax in (ax1, ax2):
        # Workspace boundary
        rect = plt.Rectangle((workspace[0], workspace[1]),
                              workspace[2] - workspace[0],
                              workspace[3] - workspace[1],
                              fill=False, edgecolor="black", linewidth=0.8)
        ax.add_patch(rect)
        ax.set_xlim(-2, 52)
        ax.set_ylim(-2, 32)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)", fontsize=7)
        ax.set_ylabel("y (m)", fontsize=7)

        # Start and goal markers
        ax.plot(5, 5, "gs", markersize=8, label="Start", zorder=10)
        ax.plot(45, 25, "r*", markersize=10, label="Goal", zorder=10)

    # --- dynamic_crossing ---
    ax1.set_title("(a) Dynamic Crossing", fontweight="bold")
    # One obstacle crossing horizontally
    obs_x, obs_y = 25, 18
    obs_circle = plt.Circle((obs_x, obs_y), 2.5, color="#e74c3c",
                             alpha=0.6, zorder=5)
    ax1.add_patch(obs_circle)
    ax1.annotate("", xy=(obs_x + 6, obs_y), xytext=(obs_x + 1, obs_y),
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5),
                 zorder=6)
    ax1.text(obs_x + 3.5, obs_y + 1.5, "v", fontsize=7, color="#c0392b",
             ha="center", style="italic")

    # Sketch nominal path
    ax1.plot([5, 25, 45], [5, 16, 25], "b--", alpha=0.3, linewidth=0.8,
             label="Nominal path")
    ax1.legend(fontsize=5, loc="lower right", frameon=False)

    # --- dynamic_slalom ---
    ax2.set_title("(b) Dynamic Slalom", fontweight="bold")
    # Multiple dynamic obstacles
    obstacles = [
        (15, 12, 1.5, 0),    # rightward
        (22, 20, -1.2, 0),   # leftward
        (30, 10, 0, 1.0),    # upward
        (35, 22, 1.0, -0.5), # diagonal
        (40, 15, -0.8, 0.8), # diagonal
    ]
    obs_colors = ["#e74c3c", "#e67e22", "#f39c12", "#e74c3c", "#e67e22"]
    for (ox, oy, vx, vy), oc in zip(obstacles, obs_colors):
        circle = plt.Circle((ox, oy), 2.0, color=oc, alpha=0.5, zorder=5)
        ax2.add_patch(circle)
        if abs(vx) > 0.01 or abs(vy) > 0.01:
            scale = 4.0
            ax2.annotate(
                "", xy=(ox + vx * scale, oy + vy * scale),
                xytext=(ox, oy),
                arrowprops=dict(arrowstyle="->", color=oc, lw=1.2),
                zorder=6,
            )

    # Sketch nominal path
    ax2.plot([5, 15, 25, 35, 45], [5, 15, 12, 20, 25],
             "b--", alpha=0.3, linewidth=0.8, label="Nominal path")
    ax2.legend(fontsize=5, loc="lower right", frameon=False)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig_scenarios.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate all paper figures for Diff-MPPI submission."
    )
    parser.add_argument(
        "--csv", default="build/benchmark_diff_mppi_exact_time_full.csv",
        help="Main benchmark CSV path (default: build/benchmark_diff_mppi_exact_time_full.csv)",
    )
    parser.add_argument(
        "--pareto-csv", default="",
        help="Optional CSV path used only for fig_pareto (default: reuse --csv data)",
    )
    parser.add_argument(
        "--trace-dir", default="build/",
        help="Directory containing trace CSV files (default: build/)",
    )
    parser.add_argument(
        "--out-dir", default="paper/figures/",
        help="Output directory for figures (default: paper/figures/)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    generated = []
    skipped = []

    # Load main CSV data
    data = []
    if os.path.exists(args.csv):
        print("Loading main CSV: " + args.csv)
        raw = load_csv(args.csv)
        data = aggregate(raw)
        print("  Loaded " + str(len(raw)) + " rows -> "
              + str(len(data)) + " aggregated entries")
        scenarios = sorted(set(r["scenario"] for r in data))
        print("  Scenarios: " + ", ".join(scenarios))
    else:
        print("WARNING: Main CSV not found: " + args.csv)
        print("  Will skip figures that require benchmark data.")

    # Also load supplementary CSVs if available (same pattern as plot_pareto_frontier.py)
    extra_csvs = [
        "build/benchmark_diff_mppi_exact_time_final.csv",
        "build/benchmark_diff_mppi_faithful.csv",
        "build/benchmark_diff_mppi_faithful_large_k.csv",
    ]
    extra_rows = []
    for path in extra_csvs:
        if os.path.exists(path) and path != args.csv:
            extra = load_csv(path)
            extra_rows.extend(extra)
    if extra_rows:
        print("  Loaded " + str(len(extra_rows))
              + " extra rows from supplementary CSVs")
        all_raw = load_csv(args.csv) if os.path.exists(args.csv) else []
        all_raw.extend(extra_rows)
        data = aggregate(all_raw)

    pareto_data = data
    if args.pareto_csv:
        if os.path.exists(args.pareto_csv):
            print("Loading pareto CSV: " + args.pareto_csv)
            pareto_data = aggregate(load_csv(args.pareto_csv))
            print("  Pareto entries: " + str(len(pareto_data)))
        else:
            print("WARNING: Pareto CSV not found: " + args.pareto_csv)
            print("  Falling back to main benchmark data for fig_pareto.")

    # --- Generate each figure ---
    figure_funcs = [
        ("fig_pareto", lambda: fig_pareto(pareto_data, args.out_dir)),
        ("fig_mechanism", lambda: fig_mechanism(args.trace_dir, args.out_dir)),
        ("fig_7dof", lambda: fig_7dof(data, args.out_dir)),
        ("fig_ablation", lambda: fig_ablation(data, args.out_dir)),
        ("fig_scenarios", lambda: fig_scenarios(args.out_dir)),
    ]

    for name, func in figure_funcs:
        print("\nGenerating " + name + "...")
        try:
            result = func()
            if result:
                generated.append(result)
                print("  -> " + result)
            else:
                skipped.append(name)
        except Exception as e:
            skipped.append(name)
            print("  [ERROR] " + name + ": " + str(e))

    # --- Summary ---
    print("\n" + "=" * 50)
    print("Figure generation complete.")
    print("  Generated: " + str(len(generated)) + " figures")
    for p in generated:
        print("    " + p)
    if skipped:
        print("  Skipped:   " + str(len(skipped)) + " figures")
        for s in skipped:
            print("    " + s)
    print("=" * 50)


if __name__ == "__main__":
    main()
