#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


NUMERIC_FIELDS = [
    "seed",
    "k_samples",
    "t_horizon",
    "grad_steps",
    "alpha",
    "reached_goal",
    "collision_free",
    "success",
    "steps",
    "final_distance",
    "min_goal_distance",
    "cumulative_cost",
    "collisions",
    "avg_control_ms",
    "total_control_ms",
    "episode_ms",
    "sample_budget",
]

SUMMARY_FIELDS = [
    "success",
    "steps",
    "final_distance",
    "min_goal_distance",
    "cumulative_cost",
    "collisions",
    "avg_control_ms",
    "total_control_ms",
    "episode_ms",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize benchmark_diff_mppi CSV into Markdown and LaTeX tables.")
    parser.add_argument("--csv", default="build/benchmark_diff_mppi.csv", help="Input CSV path")
    parser.add_argument("--markdown-out", help="Output Markdown summary path")
    parser.add_argument("--latex-out", help="Output LaTeX summary path")
    parser.add_argument("--time-caps", default="1.1,1.5,2.0", help="Comma-separated wall-clock caps in ms")
    parser.add_argument("--time-targets", default="1.0,1.5", help="Comma-separated equal-time targets in ms")
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stddev(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def fmt_pm(mu, sigma, digits=2):
    return f"{mu:.{digits}f} +/- {sigma:.{digits}f}"


def fmt_tex_pm(mu, sigma, digits=2):
    return f"${mu:.{digits}f} \\pm {sigma:.{digits}f}$"


def fmt_md(mu, sigma, digits=2):
    return f"{mu:.{digits}f} ± {sigma:.{digits}f}"


def fmt_num(value, digits=2):
    return f"{value:.{digits}f}"


def parse_float_list(text):
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return sorted(set(values))


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            for field in NUMERIC_FIELDS:
                if field in ("seed", "k_samples", "t_horizon", "grad_steps", "reached_goal",
                             "collision_free", "success", "steps", "collisions", "sample_budget"):
                    parsed[field] = int(float(row[field]))
                else:
                    parsed[field] = float(row[field])
            rows.append(parsed)
    return rows


def summarize_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["scenario"],
            row["planner"],
            row["k_samples"],
            row["t_horizon"],
            row["grad_steps"],
            row["alpha"],
        )
        groups[key].append(row)

    summary_rows = []
    for key, group in sorted(groups.items()):
        item = {
            "scenario": key[0],
            "planner": key[1],
            "k_samples": key[2],
            "t_horizon": key[3],
            "grad_steps": key[4],
            "alpha": key[5],
            "episodes": len(group),
        }
        for field in SUMMARY_FIELDS:
            values = [r[field] for r in group]
            item[field + "_mean"] = mean(values)
            item[field + "_std"] = stddev(values)
        summary_rows.append(item)
    return summary_rows


def aggregate_by_planner_k(summary_rows):
    groups = defaultdict(list)
    for row in summary_rows:
        key = (row["planner"], row["k_samples"])
        groups[key].append(row)

    agg = []
    for key, group in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        item = {"planner": key[0], "k_samples": key[1], "num_scenarios": len(group)}
        for field in SUMMARY_FIELDS:
            values = [r[field + "_mean"] for r in group]
            item[field + "_mean"] = mean(values)
            item[field + "_std"] = stddev(values)
        agg.append(item)
    return agg


def best_diff_per_budget(summary_rows):
    result = []
    by_scenario_k = defaultdict(list)
    for row in summary_rows:
        by_scenario_k[(row["scenario"], row["k_samples"])].append(row)

    for (scenario, k_samples), group in sorted(by_scenario_k.items()):
        mppi = next((r for r in group if r["planner"] == "mppi"), None)
        diff_rows = [r for r in group if r["planner"] != "mppi"]
        if not mppi or not diff_rows:
            continue
        best_diff = min(diff_rows, key=lambda r: (r["final_distance_mean"], r["cumulative_cost_mean"]))
        result.append({
            "scenario": scenario,
            "k_samples": k_samples,
            "mppi": mppi,
            "best_diff": best_diff,
            "delta_success": best_diff["success_mean"] - mppi["success_mean"],
            "delta_steps": best_diff["steps_mean"] - mppi["steps_mean"],
            "delta_final_distance": best_diff["final_distance_mean"] - mppi["final_distance_mean"],
            "delta_cost": best_diff["cumulative_cost_mean"] - mppi["cumulative_cost_mean"],
            "time_ratio": best_diff["avg_control_ms_mean"] / max(1e-6, mppi["avg_control_ms_mean"]),
        })
    return result


def rank_summary_row(row):
    return (
        -row["success_mean"],
        row["final_distance_mean"],
        row["cumulative_cost_mean"],
        row["steps_mean"],
        row["avg_control_ms_mean"],
    )


def select_under_time_caps(summary_rows, time_caps):
    selected = []
    by_scenario_planner = defaultdict(list)
    for row in summary_rows:
        by_scenario_planner[(row["scenario"], row["planner"])].append(row)

    for (scenario, planner), group in sorted(by_scenario_planner.items()):
        group = sorted(group, key=lambda r: (r["avg_control_ms_mean"], r["k_samples"]))
        for cap in time_caps:
            feasible = [r for r in group if r["avg_control_ms_mean"] <= cap]
            if not feasible:
                continue
            best = min(feasible, key=rank_summary_row)
            selected.append({
                "scenario": scenario,
                "planner": planner,
                "time_cap_ms": cap,
                "selected": best,
            })
    return selected


def aggregate_by_planner_time_cap(time_cap_rows):
    groups = defaultdict(list)
    for row in time_cap_rows:
        groups[(row["planner"], row["time_cap_ms"])].append(row["selected"])

    agg = []
    for key, group in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        item = {"planner": key[0], "time_cap_ms": key[1], "num_scenarios": len(group)}
        item["k_samples_mean"] = mean([r["k_samples"] for r in group])
        for field in SUMMARY_FIELDS:
            values = [r[field + "_mean"] for r in group]
            item[field + "_mean"] = mean(values)
            item[field + "_std"] = stddev(values)
        agg.append(item)
    return agg


def best_diff_per_time_cap(time_cap_rows):
    result = []
    by_scenario_cap = defaultdict(list)
    for row in time_cap_rows:
        by_scenario_cap[(row["scenario"], row["time_cap_ms"])].append(row)

    for (scenario, time_cap_ms), group in sorted(by_scenario_cap.items()):
        mppi_row = next((r for r in group if r["planner"] == "mppi"), None)
        diff_rows = [r for r in group if r["planner"] != "mppi"]
        if not mppi_row or not diff_rows:
            continue
        mppi = mppi_row["selected"]
        best_diff_row = min(diff_rows, key=lambda r: rank_summary_row(r["selected"]))
        best_diff = best_diff_row["selected"]
        result.append({
            "scenario": scenario,
            "time_cap_ms": time_cap_ms,
            "mppi": mppi,
            "best_diff": best_diff,
            "delta_success": best_diff["success_mean"] - mppi["success_mean"],
            "delta_steps": best_diff["steps_mean"] - mppi["steps_mean"],
            "delta_final_distance": best_diff["final_distance_mean"] - mppi["final_distance_mean"],
            "delta_cost": best_diff["cumulative_cost_mean"] - mppi["cumulative_cost_mean"],
        })
    return result


def select_at_time_targets(summary_rows, time_targets):
    selected = []
    by_scenario_planner = defaultdict(list)
    for row in summary_rows:
        by_scenario_planner[(row["scenario"], row["planner"])].append(row)

    for (scenario, planner), group in sorted(by_scenario_planner.items()):
        for target in time_targets:
            best = min(
                group,
                key=lambda r: (
                    abs(r["avg_control_ms_mean"] - target),
                    -r["success_mean"],
                    r["final_distance_mean"],
                    r["cumulative_cost_mean"],
                    r["steps_mean"],
                ),
            )
            selected.append({
                "scenario": scenario,
                "planner": planner,
                "time_target_ms": target,
                "time_gap_ms": abs(best["avg_control_ms_mean"] - target),
                "selected": best,
            })
    return selected


def aggregate_by_planner_time_target(target_rows):
    groups = defaultdict(list)
    for row in target_rows:
        groups[(row["planner"], row["time_target_ms"])].append(row)

    agg = []
    for key, group in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        selected = [r["selected"] for r in group]
        item = {"planner": key[0], "time_target_ms": key[1], "num_scenarios": len(group)}
        item["k_samples_mean"] = mean([r["k_samples"] for r in selected])
        item["time_gap_ms_mean"] = mean([r["time_gap_ms"] for r in group])
        for field in SUMMARY_FIELDS:
            values = [r[field + "_mean"] for r in selected]
            item[field + "_mean"] = mean(values)
            item[field + "_std"] = stddev(values)
        agg.append(item)
    return agg


def best_diff_per_time_target(target_rows):
    result = []
    by_scenario_target = defaultdict(list)
    for row in target_rows:
        by_scenario_target[(row["scenario"], row["time_target_ms"])].append(row)

    for (scenario, time_target_ms), group in sorted(by_scenario_target.items()):
        mppi_row = next((r for r in group if r["planner"] == "mppi"), None)
        diff_rows = [r for r in group if r["planner"] != "mppi"]
        if not mppi_row or not diff_rows:
            continue
        mppi = mppi_row["selected"]
        best_diff_row = min(
            diff_rows,
            key=lambda r: (
                -r["selected"]["success_mean"],
                r["selected"]["final_distance_mean"],
                r["selected"]["cumulative_cost_mean"],
                r["time_gap_ms"],
            ),
        )
        best_diff = best_diff_row["selected"]
        result.append({
            "scenario": scenario,
            "time_target_ms": time_target_ms,
            "mppi": mppi,
            "mppi_gap_ms": mppi_row["time_gap_ms"],
            "best_diff": best_diff,
            "best_diff_gap_ms": best_diff_row["time_gap_ms"],
            "delta_success": best_diff["success_mean"] - mppi["success_mean"],
            "delta_steps": best_diff["steps_mean"] - mppi["steps_mean"],
            "delta_final_distance": best_diff["final_distance_mean"] - mppi["final_distance_mean"],
            "delta_cost": best_diff["cumulative_cost_mean"] - mppi["cumulative_cost_mean"],
        })
    return result


def markdown_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_markdown(summary_rows, aggregate_rows, budget_rows, time_cap_aggregate_rows, time_cap_rows,
                   time_target_aggregate_rows, time_target_rows, csv_path):
    lines = []
    lines.append("# Diff-MPPI Benchmark Summary")
    lines.append("")
    lines.append(f"Source CSV: `{csv_path}`")
    lines.append("")
    lines.append("## Aggregate Across Scenarios")
    lines.append("")
    agg_rows = []
    for row in aggregate_rows:
        agg_rows.append([
            row["planner"],
            str(row["k_samples"]),
            fmt_md(row["success_mean"], row["success_std"], 2),
            fmt_md(row["steps_mean"], row["steps_std"], 1),
            fmt_md(row["final_distance_mean"], row["final_distance_std"], 2),
            fmt_md(row["cumulative_cost_mean"], row["cumulative_cost_std"], 1),
            fmt_md(row["avg_control_ms_mean"], row["avg_control_ms_std"], 2),
        ])
    lines.append(markdown_table(
        ["Planner", "K", "Success", "Steps", "Final Dist", "Cum. Cost", "Avg Control ms"],
        agg_rows,
    ))
    lines.append("")
    lines.append("## Best Diff Variant at Fixed Rollout Budget")
    lines.append("")
    budget_md_rows = []
    for row in budget_rows:
        budget_md_rows.append([
            row["scenario"],
            str(row["k_samples"]),
            row["best_diff"]["planner"],
            fmt_num(row["mppi"]["final_distance_mean"]),
            fmt_num(row["best_diff"]["final_distance_mean"]),
            f"{row['delta_final_distance']:.2f}",
            f"{row['delta_steps']:.1f}",
            f"{row['time_ratio']:.2f}x",
        ])
    lines.append(markdown_table(
        ["Scenario", "K", "Best Diff", "MPPI Dist", "Diff Dist", "Delta Dist", "Delta Steps", "Time Ratio"],
        budget_md_rows,
    ))
    lines.append("")
    lines.append("## Aggregate Under Fixed Wall-Clock Budget")
    lines.append("")
    time_cap_md_rows = []
    for row in time_cap_aggregate_rows:
        time_cap_md_rows.append([
            fmt_num(row["time_cap_ms"], 2),
            row["planner"],
            str(row["num_scenarios"]),
            fmt_md(row["success_mean"], row["success_std"], 2),
            fmt_md(row["steps_mean"], row["steps_std"], 1),
            fmt_md(row["final_distance_mean"], row["final_distance_std"], 2),
            fmt_md(row["cumulative_cost_mean"], row["cumulative_cost_std"], 1),
            fmt_md(row["avg_control_ms_mean"], row["avg_control_ms_std"], 2),
            fmt_num(row["k_samples_mean"], 0),
        ])
    lines.append(markdown_table(
        ["Cap ms", "Planner", "Scenarios", "Success", "Steps", "Final Dist", "Cum. Cost", "Avg Control ms", "Mean K"],
        time_cap_md_rows,
    ))
    lines.append("")
    lines.append("## Best Diff Variant at Fixed Wall-Clock Budget")
    lines.append("")
    time_budget_md_rows = []
    for row in time_cap_rows:
        time_budget_md_rows.append([
            row["scenario"],
            fmt_num(row["time_cap_ms"], 2),
            f"{row['mppi']['k_samples']} @ {fmt_num(row['mppi']['avg_control_ms_mean'])}",
            f"{row['best_diff']['planner']} ({row['best_diff']['k_samples']} @ {fmt_num(row['best_diff']['avg_control_ms_mean'])})",
            fmt_num(row["mppi"]["final_distance_mean"]),
            fmt_num(row["best_diff"]["final_distance_mean"]),
            f"{row['delta_final_distance']:.2f}",
            f"{row['delta_steps']:.1f}",
        ])
    lines.append(markdown_table(
        ["Scenario", "Cap ms", "MPPI K@ms", "Best Diff K@ms", "MPPI Dist", "Diff Dist", "Delta Dist", "Delta Steps"],
        time_budget_md_rows,
    ))
    lines.append("")
    lines.append("## Aggregate Under Equal-Time Targets")
    lines.append("")
    time_target_md_rows = []
    for row in time_target_aggregate_rows:
        time_target_md_rows.append([
            fmt_num(row["time_target_ms"], 2),
            row["planner"],
            str(row["num_scenarios"]),
            fmt_md(row["success_mean"], row["success_std"], 2),
            fmt_md(row["steps_mean"], row["steps_std"], 1),
            fmt_md(row["final_distance_mean"], row["final_distance_std"], 2),
            fmt_md(row["cumulative_cost_mean"], row["cumulative_cost_std"], 1),
            fmt_md(row["avg_control_ms_mean"], row["avg_control_ms_std"], 2),
            fmt_num(row["k_samples_mean"], 0),
            fmt_num(row["time_gap_ms_mean"], 2),
        ])
    lines.append(markdown_table(
        ["Target ms", "Planner", "Scenarios", "Success", "Steps", "Final Dist", "Cum. Cost", "Avg Control ms", "Mean K", "Mean |Gap| ms"],
        time_target_md_rows,
    ))
    lines.append("")
    lines.append("## Best Diff Variant at Equal-Time Targets")
    lines.append("")
    equal_time_md_rows = []
    for row in time_target_rows:
        equal_time_md_rows.append([
            row["scenario"],
            fmt_num(row["time_target_ms"], 2),
            f"{row['mppi']['k_samples']} @ {fmt_num(row['mppi']['avg_control_ms_mean'])} (gap {fmt_num(row['mppi_gap_ms'])})",
            f"{row['best_diff']['planner']} ({row['best_diff']['k_samples']} @ {fmt_num(row['best_diff']['avg_control_ms_mean'])}, gap {fmt_num(row['best_diff_gap_ms'])})",
            fmt_num(row["mppi"]["final_distance_mean"]),
            fmt_num(row["best_diff"]["final_distance_mean"]),
            f"{row['delta_final_distance']:.2f}",
            f"{row['delta_steps']:.1f}",
        ])
    lines.append(markdown_table(
        ["Scenario", "Target ms", "MPPI K@ms", "Best Diff K@ms", "MPPI Dist", "Diff Dist", "Delta Dist", "Delta Steps"],
        equal_time_md_rows,
    ))
    lines.append("")

    by_scenario = defaultdict(list)
    for row in summary_rows:
        by_scenario[row["scenario"]].append(row)

    for scenario in sorted(by_scenario):
        lines.append(f"## Scenario: `{scenario}`")
        lines.append("")
        scenario_rows = []
        for row in sorted(by_scenario[scenario], key=lambda r: (r["k_samples"], r["planner"])):
            scenario_rows.append([
                row["planner"],
                str(row["k_samples"]),
                str(row["grad_steps"]),
                fmt_num(row["alpha"], 3),
                fmt_md(row["success_mean"], row["success_std"], 2),
                fmt_md(row["steps_mean"], row["steps_std"], 1),
                fmt_md(row["final_distance_mean"], row["final_distance_std"], 2),
                fmt_md(row["cumulative_cost_mean"], row["cumulative_cost_std"], 1),
                fmt_md(row["avg_control_ms_mean"], row["avg_control_ms_std"], 2),
            ])
        lines.append(markdown_table(
            ["Planner", "K", "Grad Steps", "Alpha", "Success", "Steps", "Final Dist", "Cum. Cost", "Avg Control ms"],
            scenario_rows,
        ))
        lines.append("")
    return "\n".join(lines)


def build_latex(summary_rows, aggregate_rows, budget_rows, time_cap_aggregate_rows, time_cap_rows,
                time_target_aggregate_rows, time_target_rows, csv_path):
    lines = []
    lines.append("% Auto-generated by scripts/summarize_diff_mppi.py")
    lines.append(f"% Source CSV: {csv_path}")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Aggregate Diff-MPPI benchmark results across scenarios.}")
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\hline")
    lines.append("Planner & K & Success & Steps & FinalDist & Cost & AvgMs \\\\")
    lines.append("\\hline")
    for row in aggregate_rows:
        lines.append(
            f"{row['planner']} & {row['k_samples']} & "
            f"{fmt_tex_pm(row['success_mean'], row['success_std'], 2)} & "
            f"{fmt_tex_pm(row['steps_mean'], row['steps_std'], 1)} & "
            f"{fmt_tex_pm(row['final_distance_mean'], row['final_distance_std'], 2)} & "
            f"{fmt_tex_pm(row['cumulative_cost_mean'], row['cumulative_cost_std'], 1)} & "
            f"{fmt_tex_pm(row['avg_control_ms_mean'], row['avg_control_ms_std'], 2)} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Best gradient-refined variant at fixed rollout budget relative to vanilla MPPI.}")
    lines.append("\\begin{tabular}{llrrrrrr}")
    lines.append("\\hline")
    lines.append("Scenario & K & BestDiff & MPPIDist & DiffDist & DeltaDist & DeltaSteps & TimeRatio \\\\")
    lines.append("\\hline")
    for row in budget_rows:
        lines.append(
            f"{row['scenario']} & {row['k_samples']} & {row['best_diff']['planner']} & "
            f"{fmt_num(row['mppi']['final_distance_mean'])} & "
            f"{fmt_num(row['best_diff']['final_distance_mean'])} & "
            f"{row['delta_final_distance']:.2f} & {row['delta_steps']:.1f} & {row['time_ratio']:.2f} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Best configurations selected under a fixed wall-clock budget.}")
    lines.append("\\begin{tabular}{llrrrrrr}")
    lines.append("\\hline")
    lines.append("CapMs & Planner & Scenarios & Success & Steps & FinalDist & Cost & AvgMs \\\\")
    lines.append("\\hline")
    for row in time_cap_aggregate_rows:
        lines.append(
            f"{fmt_num(row['time_cap_ms'])} & {row['planner']} & {row['num_scenarios']} & "
            f"{fmt_tex_pm(row['success_mean'], row['success_std'], 2)} & "
            f"{fmt_tex_pm(row['steps_mean'], row['steps_std'], 1)} & "
            f"{fmt_tex_pm(row['final_distance_mean'], row['final_distance_std'], 2)} & "
            f"{fmt_tex_pm(row['cumulative_cost_mean'], row['cumulative_cost_std'], 1)} & "
            f"{fmt_tex_pm(row['avg_control_ms_mean'], row['avg_control_ms_std'], 2)} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Best gradient-refined variant relative to vanilla MPPI under the same wall-clock budget.}")
    lines.append("\\begin{tabular}{llrrrrrr}")
    lines.append("\\hline")
    lines.append("Scenario & CapMs & MPPIDist & DiffDist & DeltaDist & DeltaSteps & MPPIK & DiffK \\\\")
    lines.append("\\hline")
    for row in time_cap_rows:
        lines.append(
            f"{row['scenario']} & {fmt_num(row['time_cap_ms'])} & "
            f"{fmt_num(row['mppi']['final_distance_mean'])} & "
            f"{fmt_num(row['best_diff']['final_distance_mean'])} & "
            f"{row['delta_final_distance']:.2f} & {row['delta_steps']:.1f} & "
            f"{row['mppi']['k_samples']} & {row['best_diff']['k_samples']} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Planner configurations selected by closest equal-time target.}")
    lines.append("\\begin{tabular}{llrrrrrr}")
    lines.append("\\hline")
    lines.append("TargetMs & Planner & Scenarios & Success & Steps & FinalDist & Cost & AvgMs \\\\")
    lines.append("\\hline")
    for row in time_target_aggregate_rows:
        lines.append(
            f"{fmt_num(row['time_target_ms'])} & {row['planner']} & {row['num_scenarios']} & "
            f"{fmt_tex_pm(row['success_mean'], row['success_std'], 2)} & "
            f"{fmt_tex_pm(row['steps_mean'], row['steps_std'], 1)} & "
            f"{fmt_tex_pm(row['final_distance_mean'], row['final_distance_std'], 2)} & "
            f"{fmt_tex_pm(row['cumulative_cost_mean'], row['cumulative_cost_std'], 1)} & "
            f"{fmt_tex_pm(row['avg_control_ms_mean'], row['avg_control_ms_std'], 2)} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Best gradient-refined variant relative to vanilla MPPI at matched equal-time targets.}")
    lines.append("\\begin{tabular}{llrrrrrr}")
    lines.append("\\hline")
    lines.append("Scenario & TargetMs & MPPIDist & DiffDist & DeltaDist & DeltaSteps & MPPIK & DiffK \\\\")
    lines.append("\\hline")
    for row in time_target_rows:
        lines.append(
            f"{row['scenario']} & {fmt_num(row['time_target_ms'])} & "
            f"{fmt_num(row['mppi']['final_distance_mean'])} & "
            f"{fmt_num(row['best_diff']['final_distance_mean'])} & "
            f"{row['delta_final_distance']:.2f} & {row['delta_steps']:.1f} & "
            f"{row['mppi']['k_samples']} & {row['best_diff']['k_samples']} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def default_output_paths(csv_path):
    csv_path = Path(csv_path)
    stem = csv_path.stem
    parent = csv_path.parent
    return parent / f"{stem}_summary.md", parent / f"{stem}_summary.tex"


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    summary_rows = summarize_groups(rows)
    aggregate_rows = aggregate_by_planner_k(summary_rows)
    budget_rows = best_diff_per_budget(summary_rows)
    time_caps = parse_float_list(args.time_caps)
    time_cap_selected_rows = select_under_time_caps(summary_rows, time_caps)
    time_cap_aggregate_rows = aggregate_by_planner_time_cap(time_cap_selected_rows)
    time_cap_rows = best_diff_per_time_cap(time_cap_selected_rows)
    time_targets = parse_float_list(args.time_targets)
    time_target_selected_rows = select_at_time_targets(summary_rows, time_targets)
    time_target_aggregate_rows = aggregate_by_planner_time_target(time_target_selected_rows)
    time_target_rows = best_diff_per_time_target(time_target_selected_rows)

    markdown = build_markdown(summary_rows, aggregate_rows, budget_rows, time_cap_aggregate_rows, time_cap_rows,
                              time_target_aggregate_rows, time_target_rows, csv_path)
    latex = build_latex(summary_rows, aggregate_rows, budget_rows, time_cap_aggregate_rows, time_cap_rows,
                        time_target_aggregate_rows, time_target_rows, csv_path)

    md_out, tex_out = default_output_paths(csv_path)
    if args.markdown_out:
        md_out = Path(args.markdown_out)
    if args.latex_out:
        tex_out = Path(args.latex_out)

    md_out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(markdown)
    tex_out.write_text(latex)

    print(f"Markdown summary saved to {md_out}")
    print(f"LaTeX summary saved to {tex_out}")


if __name__ == "__main__":
    main()
