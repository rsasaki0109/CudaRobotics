#!/usr/bin/env python3

import argparse
import math
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from summarize_diff_mppi import (
    load_rows,
    parse_float_list,
    select_at_time_targets,
    select_under_time_caps,
    summarize_groups,
)


PLANNER_STYLES = {
    "mppi": {"label": "MPPI", "color": "#1a1a1a", "marker": "o"},
    "feedback_mppi": {"label": "Feedback MPPI", "color": "#7b8c5a", "marker": "P"},
    "feedback_mppi_hf": {"label": "Feedback MPPI (HF)", "color": "#3f7f6b", "marker": "*"},
    "feedback_mppi_sens": {"label": "Feedback MPPI (sens)", "color": "#5a7db8", "marker": "X"},
    "feedback_mppi_cov": {"label": "Feedback MPPI (cov)", "color": "#8b5fbf", "marker": "v"},
    "feedback_mppi_fused": {"label": "Feedback MPPI (fused)", "color": "#b05f3c", "marker": ">"},
    "grad_only_3": {"label": "Grad-Only (3 step)", "color": "#6c757d", "marker": "D"},
    "diff_mppi_1": {"label": "Diff-MPPI (1 grad)", "color": "#c65d33", "marker": "s"},
    "diff_mppi_3": {"label": "Diff-MPPI (3 grad)", "color": "#2a6f97", "marker": "^"},
}

PLOT_ORDER = ["mppi", "feedback_mppi", "feedback_mppi_hf", "feedback_mppi_sens", "feedback_mppi_cov", "feedback_mppi_fused", "grad_only_3", "diff_mppi_1", "diff_mppi_3"]
SCENARIO_TITLES = {
    "arm_dynamic_sweep": "Planar Arm Dynamic Sweep",
    "arm_static_shelf": "Planar Arm Shelf Reach",
    "cartpole_large_angle": "CartPole Large-Angle Recovery",
    "cartpole_recover": "CartPole Recovery",
    "cluttered": "Cluttered Field",
    "corner_turn": "Corner Turn",
    "dynamic_crossing": "Dynamic Crossing",
    "dynamic_slalom": "Dynamic Slalom",
    "dynbike_crossing": "Dynamic Bicycle Crossing",
    "dynbike_slalom": "Dynamic Bicycle Slalom",
    "narrow_passage": "Narrow Passage",
    "slalom": "Slalom",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Diff-MPPI benchmark figures from CSV summaries.")
    parser.add_argument("--csv", default="build/benchmark_diff_mppi.csv", help="Input CSV path")
    parser.add_argument("--out-dir", default="build/plots", help="Output directory for figures")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats")
    parser.add_argument("--time-caps", default="1.1,1.5,2.0", help="Comma-separated wall-clock caps in ms")
    parser.add_argument("--time-targets", default="1.0,1.5", help="Comma-separated equal-time targets in ms")
    return parser.parse_args()


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
    })


def scenario_grid(summary_rows):
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row["scenario"]].append(row)
    return grouped


def metric_key(name):
    return f"{name}_mean", f"{name}_std"


def make_axes(num_scenarios):
    cols = 2
    rows = max(1, math.ceil(num_scenarios / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 3.8 * rows), squeeze=False, constrained_layout=True)
    return fig, axes


def plot_tradeoff_grid(summary_rows, metric, ylabel, title, out_dir, stem, formats):
    grouped = scenario_grid(summary_rows)
    scenarios = sorted(grouped)
    mean_key, std_key = metric_key(metric)
    fig, axes = make_axes(len(scenarios))
    axes_flat = axes.flatten()

    legend_handles = None
    legend_labels = None
    for ax, scenario in zip(axes_flat, scenarios):
        rows = grouped[scenario]
        for planner in PLOT_ORDER:
            planner_rows = sorted((r for r in rows if r["planner"] == planner), key=lambda r: r["k_samples"])
            if not planner_rows:
                continue
            style = PLANNER_STYLES.get(planner, {"label": planner, "color": "#444444", "marker": "o"})
            xs = [r["avg_control_ms_mean"] for r in planner_rows]
            ys = [r[mean_key] for r in planner_rows]
            xerr = [r["avg_control_ms_std"] for r in planner_rows]
            yerr = [r[std_key] for r in planner_rows]
            ax.errorbar(
                xs,
                ys,
                xerr=xerr,
                yerr=yerr,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=6,
                capsize=3,
            )
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            for row in planner_rows:
                ax.annotate(
                    f"K={row['k_samples']}",
                    (row["avg_control_ms_mean"], row[mean_key]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    color=style["color"],
                )
        ax.set_title(SCENARIO_TITLES.get(scenario, scenario.replace("_", " ").title()))
        ax.set_xscale("log")
        ax.set_xlabel("Average control time [ms]")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)

    for ax in axes_flat[len(scenarios):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    save_figure(fig, out_dir, stem, formats)
    plt.close(fig)


def plot_budget_grid(summary_rows, metric, ylabel, title, out_dir, stem, formats):
    grouped = scenario_grid(summary_rows)
    scenarios = sorted(grouped)
    mean_key, std_key = metric_key(metric)
    fig, axes = make_axes(len(scenarios))
    axes_flat = axes.flatten()

    legend_handles = None
    legend_labels = None
    xticks = sorted({r["k_samples"] for r in summary_rows})
    for ax, scenario in zip(axes_flat, scenarios):
        rows = grouped[scenario]
        for planner in PLOT_ORDER:
            planner_rows = sorted((r for r in rows if r["planner"] == planner), key=lambda r: r["k_samples"])
            if not planner_rows:
                continue
            style = PLANNER_STYLES.get(planner, {"label": planner, "color": "#444444", "marker": "o"})
            xs = [r["k_samples"] for r in planner_rows]
            ys = [r[mean_key] for r in planner_rows]
            yerr = [r[std_key] for r in planner_rows]
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=6,
                capsize=3,
            )
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_title(SCENARIO_TITLES.get(scenario, scenario.replace("_", " ").title()))
        ax.set_xlabel("Samples per control step (K)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(scenarios):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    save_figure(fig, out_dir, stem, formats)
    plt.close(fig)


def plot_time_cap_grid(time_cap_rows, metric, ylabel, title, out_dir, stem, formats):
    grouped = defaultdict(list)
    for row in time_cap_rows:
        grouped[row["scenario"]].append(row)
    scenarios = sorted(grouped)
    mean_key, _ = metric_key(metric)
    fig, axes = make_axes(len(scenarios))
    axes_flat = axes.flatten()

    legend_handles = None
    legend_labels = None
    xticks = sorted({row["time_cap_ms"] for row in time_cap_rows})
    for ax, scenario in zip(axes_flat, scenarios):
        rows = grouped[scenario]
        for planner in PLOT_ORDER:
            planner_rows = sorted((r for r in rows if r["planner"] == planner), key=lambda r: r["time_cap_ms"])
            if not planner_rows:
                continue
            style = PLANNER_STYLES.get(planner, {"label": planner, "color": "#444444", "marker": "o"})
            xs = [r["time_cap_ms"] for r in planner_rows]
            ys = [r["selected"][mean_key] for r in planner_rows]
            ax.plot(
                xs,
                ys,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=6,
            )
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            for row in planner_rows:
                ax.annotate(
                    f"K={row['selected']['k_samples']}",
                    (row["time_cap_ms"], row["selected"][mean_key]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=style["color"],
                )
        ax.set_title(SCENARIO_TITLES.get(scenario, scenario.replace("_", " ").title()))
        ax.set_xlabel("Wall-clock cap [ms]")
        ax.set_ylabel(ylabel)
        ax.set_xticks(xticks)
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(scenarios):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    save_figure(fig, out_dir, stem, formats)
    plt.close(fig)


def plot_time_target_grid(time_target_rows, metric, ylabel, title, out_dir, stem, formats):
    grouped = defaultdict(list)
    for row in time_target_rows:
        grouped[row["scenario"]].append(row)
    scenarios = sorted(grouped)
    mean_key, _ = metric_key(metric)
    fig, axes = make_axes(len(scenarios))
    axes_flat = axes.flatten()

    legend_handles = None
    legend_labels = None
    xticks = sorted({row["time_target_ms"] for row in time_target_rows})
    for ax, scenario in zip(axes_flat, scenarios):
        rows = grouped[scenario]
        for planner in PLOT_ORDER:
            planner_rows = sorted((r for r in rows if r["planner"] == planner), key=lambda r: r["time_target_ms"])
            if not planner_rows:
                continue
            style = PLANNER_STYLES.get(planner, {"label": planner, "color": "#444444", "marker": "o"})
            xs = [r["time_target_ms"] for r in planner_rows]
            ys = [r["selected"][mean_key] for r in planner_rows]
            ax.plot(
                xs,
                ys,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=6,
            )
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            for row in planner_rows:
                ax.annotate(
                    f"K={row['selected']['k_samples']}",
                    (row["time_target_ms"], row["selected"][mean_key]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=style["color"],
                )
        ax.set_title(SCENARIO_TITLES.get(scenario, scenario.replace("_", " ").title()))
        ax.set_xlabel("Equal-time target [ms]")
        ax.set_ylabel(ylabel)
        ax.set_xticks(xticks)
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(scenarios):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    save_figure(fig, out_dir, stem, formats)
    plt.close(fig)


def save_figure(fig, out_dir, stem, formats):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path)
        print(f"Saved {path}")


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    if not formats:
        raise SystemExit("No output formats requested")

    configure_style()
    rows = load_rows(csv_path)
    summary_rows = summarize_groups(rows)
    time_caps = parse_float_list(args.time_caps)
    time_cap_rows = select_under_time_caps(summary_rows, time_caps)
    time_targets = parse_float_list(args.time_targets)
    time_target_rows = select_at_time_targets(summary_rows, time_targets)
    out_dir = Path(args.out_dir)

    plot_tradeoff_grid(
        summary_rows,
        metric="final_distance",
        ylabel="Final distance to goal [px]",
        title="Diff-MPPI Quality vs Wall-Clock",
        out_dir=out_dir,
        stem="diff_mppi_final_distance_vs_time",
        formats=formats,
    )
    plot_tradeoff_grid(
        summary_rows,
        metric="cumulative_cost",
        ylabel="Cumulative trajectory cost",
        title="Diff-MPPI Cost vs Wall-Clock",
        out_dir=out_dir,
        stem="diff_mppi_cost_vs_time",
        formats=formats,
    )
    plot_budget_grid(
        summary_rows,
        metric="final_distance",
        ylabel="Final distance to goal [px]",
        title="Diff-MPPI Quality at Fixed Sample Budget",
        out_dir=out_dir,
        stem="diff_mppi_final_distance_vs_budget",
        formats=formats,
    )
    plot_time_cap_grid(
        time_cap_rows,
        metric="final_distance",
        ylabel="Best final distance under cap [px]",
        title="Diff-MPPI at Fixed Wall-Clock Budget",
        out_dir=out_dir,
        stem="diff_mppi_final_distance_vs_time_cap",
        formats=formats,
    )
    plot_time_target_grid(
        time_target_rows,
        metric="final_distance",
        ylabel="Best final distance near target [px]",
        title="Diff-MPPI at Equal-Time Targets",
        out_dir=out_dir,
        stem="diff_mppi_final_distance_vs_equal_time",
        formats=formats,
    )


if __name__ == "__main__":
    main()
