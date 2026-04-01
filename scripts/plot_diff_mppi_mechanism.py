#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize_diff_mppi import summarize_groups


PLANNER_STYLES = {
    "mppi": {"label": "MPPI", "color": "#1a1a1a", "marker": "o"},
    "feedback_mppi": {"label": "Feedback MPPI", "color": "#7b8c5a", "marker": "P"},
    "feedback_mppi_sens": {"label": "Feedback MPPI (sens)", "color": "#5a7db8", "marker": "X"},
    "diff_mppi_1": {"label": "Diff-MPPI (1 grad)", "color": "#c65d33", "marker": "s"},
    "diff_mppi_3": {"label": "Diff-MPPI (3 grad)", "color": "#2a6f97", "marker": "^"},
}

PLOT_ORDER = ["mppi", "feedback_mppi", "feedback_mppi_sens", "diff_mppi_1", "diff_mppi_3"]


TRACE_FLOAT_FIELDS = {
    "alpha",
    "goal_distance",
    "min_obstacle_margin",
    "control_ms",
    "sampled_accel",
    "sampled_steer",
    "final_accel",
    "final_steer",
    "delta_accel",
    "delta_steer",
    "delta_norm",
    "grad_accel",
    "grad_steer",
    "grad_norm",
}

TRACE_INT_FIELDS = {"seed", "k_samples", "grad_steps", "episode_step", "horizon_step"}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Diff-MPPI mechanism figures from trace and benchmark CSV files.")
    parser.add_argument("--trace-csv", required=True, help="Trace CSV emitted by benchmark_diff_mppi --trace-csv")
    parser.add_argument("--benchmark-csv", help="Benchmark CSV for success-vs-K plotting")
    parser.add_argument("--scenario", required=True, help="Scenario name to visualize")
    parser.add_argument("--out-dir", default="build/plots_mechanism", help="Output directory")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats")
    parser.add_argument("--summary-out", help="Optional markdown summary path")
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


def load_trace_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            for key in TRACE_INT_FIELDS:
                parsed[key] = int(float(parsed[key]))
            for key in TRACE_FLOAT_FIELDS:
                parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def load_benchmark_summary_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if key in {"scenario", "planner"}:
                continue
            row[key] = float(value) if "." in value or "e" in value.lower() else int(value)
    return summarize_groups(rows)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_figure(fig, out_dir, stem, formats):
    ensure_dir(out_dir)
    for fmt in formats:
        path = Path(out_dir) / f"{stem}.{fmt}"
        fig.savefig(path)
        print(f"Saved {path}")


def mean(values):
    return sum(values) / len(values) if values else 0.0


def aggregate_by_episode(trace_rows):
    grouped = defaultdict(list)
    for row in trace_rows:
        grouped[(row["planner"], row["episode_step"])].append(row)

    result = defaultdict(list)
    for (planner, episode_step), rows in sorted(grouped.items()):
        first = [row for row in rows if row["horizon_step"] == 0]
        result[planner].append({
            "episode_step": episode_step,
            "first_delta_norm": mean([row["delta_norm"] for row in first]),
            "mean_delta_norm": mean([row["delta_norm"] for row in rows]),
            "mean_grad_norm": mean([row["grad_norm"] for row in rows]),
            "goal_distance": mean([row["goal_distance"] for row in first]),
            "min_obstacle_margin": mean([row["min_obstacle_margin"] for row in first]),
        })
    return result


def aggregate_by_horizon(trace_rows):
    grouped = defaultdict(list)
    for row in trace_rows:
        grouped[(row["planner"], row["horizon_step"])].append(row)

    result = defaultdict(list)
    for (planner, horizon_step), rows in sorted(grouped.items()):
        result[planner].append({
            "horizon_step": horizon_step,
            "mean_delta_norm": mean([row["delta_norm"] for row in rows]),
            "mean_grad_norm": mean([row["grad_norm"] for row in rows]),
        })
    return result


def plot_episode_deltas(episode_rows, scenario, out_dir, formats):
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), constrained_layout=True)

    for planner in PLOT_ORDER:
        rows = episode_rows.get(planner, [])
        if not rows:
            continue
        style = PLANNER_STYLES[planner]
        xs = [row["episode_step"] for row in rows]
        axes[0].plot(xs, [row["first_delta_norm"] for row in rows], label=style["label"],
                     color=style["color"], marker=style["marker"], linewidth=1.8, markersize=4)
        axes[1].plot(xs, [row["mean_delta_norm"] for row in rows], label=style["label"],
                     color=style["color"], marker=style["marker"], linewidth=1.8, markersize=4)

    axes[0].set_title(f"{scenario}: first-action refinement magnitude")
    axes[0].set_xlabel("Episode step")
    axes[0].set_ylabel("||u_final - u_sample||")
    axes[1].set_title(f"{scenario}: mean horizon refinement magnitude")
    axes[1].set_xlabel("Episode step")
    axes[1].set_ylabel("Mean ||u_final - u_sample||")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    save_figure(fig, out_dir, f"{scenario}_correction_vs_episode", formats)
    plt.close(fig)


def plot_horizon_profile(horizon_rows, scenario, out_dir, formats):
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), constrained_layout=True)

    for planner in PLOT_ORDER:
        rows = horizon_rows.get(planner, [])
        if not rows:
            continue
        style = PLANNER_STYLES[planner]
        xs = [row["horizon_step"] for row in rows]
        axes[0].plot(xs, [row["mean_delta_norm"] for row in rows], label=style["label"],
                     color=style["color"], marker=style["marker"], linewidth=1.8, markersize=4)
        axes[1].plot(xs, [row["mean_grad_norm"] for row in rows], label=style["label"],
                     color=style["color"], marker=style["marker"], linewidth=1.8, markersize=4)

    axes[0].set_title(f"{scenario}: correction magnitude by horizon index")
    axes[0].set_xlabel("Horizon step")
    axes[0].set_ylabel("Mean ||u_final - u_sample||")
    axes[1].set_title(f"{scenario}: gradient magnitude by horizon index")
    axes[1].set_xlabel("Horizon step")
    axes[1].set_ylabel("Mean ||grad||")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    save_figure(fig, out_dir, f"{scenario}_correction_vs_horizon", formats)
    plt.close(fig)


def plot_success_vs_k(summary_rows, scenario, out_dir, formats):
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    for planner in PLOT_ORDER:
        rows = sorted(
            [row for row in summary_rows if row["scenario"] == scenario and row["planner"] == planner],
            key=lambda row: row["k_samples"],
        )
        if not rows:
            continue
        style = PLANNER_STYLES[planner]
        ax.plot([row["k_samples"] for row in rows], [row["success_mean"] for row in rows],
                label=style["label"], color=style["color"], marker=style["marker"],
                linewidth=1.8, markersize=5)
    ax.set_title(f"{scenario}: success vs rollout budget / grad steps")
    ax.set_xlabel("Samples per control step (K)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, out_dir, f"{scenario}_success_vs_k", formats)
    plt.close(fig)


def build_summary(trace_rows, episode_rows, horizon_rows, scenario, trace_csv, benchmark_csv):
    lines = []
    lines.append("# Diff-MPPI Mechanism Summary")
    lines.append("")
    lines.append(f"Scenario: `{scenario}`")
    lines.append("")
    lines.append(f"Trace CSV: `{trace_csv}`")
    if benchmark_csv:
        lines.append(f"Benchmark CSV: `{benchmark_csv}`")
    lines.append("")
    lines.append("## Key Signals")
    lines.append("")
    for planner in PLOT_ORDER:
        rows = horizon_rows.get(planner, [])
        if not rows:
            continue
        early = mean([row["mean_delta_norm"] for row in rows[:5]])
        late = mean([row["mean_delta_norm"] for row in rows[-5:]])
        grad_mean = mean([row["mean_grad_norm"] for row in rows])
        episode = episode_rows.get(planner, [])
        first_action_peak = max((row["first_delta_norm"] for row in episode), default=0.0)
        lines.append(
            f"- `{planner}`: early-horizon correction `{early:.3f}`, late-horizon correction `{late:.3f}`, "
            f"mean gradient norm `{grad_mean:.3f}`, peak first-action correction `{first_action_peak:.3f}`."
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Large early-horizon corrections indicate that refinement mostly changes the near-term controls that are actually executed.")
    lines.append("- A front-loaded correction profile supports the claim that the gradient stage sharpens the sampled trajectory locally rather than replacing the whole plan.")
    return "\n".join(lines)


def main():
    args = parse_args()
    configure_style()
    formats = [token.strip() for token in args.formats.split(",") if token.strip()]

    trace_rows = [row for row in load_trace_rows(args.trace_csv) if row["scenario"] == args.scenario]
    if not trace_rows:
        raise SystemExit(f"No trace rows found for scenario {args.scenario}")

    episode_rows = aggregate_by_episode(trace_rows)
    horizon_rows = aggregate_by_horizon(trace_rows)
    plot_episode_deltas(episode_rows, args.scenario, args.out_dir, formats)
    plot_horizon_profile(horizon_rows, args.scenario, args.out_dir, formats)

    benchmark_csv = args.benchmark_csv
    if benchmark_csv:
        summary_rows = load_benchmark_summary_rows(benchmark_csv)
        plot_success_vs_k(summary_rows, args.scenario, args.out_dir, formats)

    summary_path = Path(args.summary_out) if args.summary_out else Path(args.out_dir) / f"{args.scenario}_mechanism_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary(trace_rows, episode_rows, horizon_rows, args.scenario, args.trace_csv, benchmark_csv))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
