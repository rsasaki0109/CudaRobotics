#!/usr/bin/env python3

import argparse
import csv
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from summarize_diff_mppi import load_rows, parse_float_list, planner_family, summarize_groups


SUMMARY_METRICS = [
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


PRESETS = {
    "dynamic_nav": {
        "bin": "./bin/benchmark_diff_mppi",
        "scenarios": "dynamic_crossing,dynamic_slalom",
        "planners": "mppi,feedback_mppi,feedback_mppi_ref,feedback_mppi_sens,feedback_mppi_cov,diff_mppi_1,diff_mppi_3",
        "time_targets": "1.0,1.5,2.0",
        "search_seed_count": 4,
        "final_seed_count": 4,
        "k_min": 128,
        "k_max": 16384,
        "tolerance_ms": 0.03,
        "max_evals": 8,
        "csv_out": "build/benchmark_diff_mppi_exact_time.csv",
        "summary_title": "Diff-MPPI Exact-Time Tuning Summary",
    },
    "dynamic_nav_architecture": {
        "bin": "./bin/benchmark_diff_mppi",
        "scenarios": "dynamic_crossing,dynamic_slalom",
        "planners": "mppi,feedback_mppi_hf,feedback_mppi_ref,diff_mppi_3",
        "time_targets": "1.0,1.5,2.0",
        "search_seed_count": 4,
        "final_seed_count": 4,
        "k_min": 128,
        "k_max": 16384,
        "tolerance_ms": 0.03,
        "max_evals": 8,
        "csv_out": "build/benchmark_diff_mppi_exact_time_hf.csv",
        "summary_title": "Diff-MPPI Exact-Time Summary (architecture gap)",
    },
    "dynamic_nav_release": {
        "bin": "./bin/benchmark_diff_mppi",
        "scenarios": "dynamic_crossing,dynamic_slalom",
        "planners": "mppi,feedback_mppi_ref,feedback_mppi_release,diff_mppi_3",
        "time_targets": "1.0,1.5",
        "search_seed_count": 4,
        "final_seed_count": 4,
        "k_min": 128,
        "k_max": 16384,
        "tolerance_ms": 0.03,
        "max_evals": 8,
        "csv_out": "build/benchmark_diff_mppi_exact_time_release.csv",
        "summary_title": "Diff-MPPI Exact-Time Summary (release-style weighting baseline)",
    },
    "dynamic_nav_heavy_feedback": {
        "bin": "./bin/benchmark_diff_mppi",
        "scenarios": "dynamic_crossing,dynamic_slalom",
        "planners": "mppi,feedback_mppi_fused,diff_mppi_3",
        "time_targets": "2.0",
        "search_seed_count": 4,
        "final_seed_count": 4,
        "k_min": 64,
        "k_max": 512,
        "tolerance_ms": 0.05,
        "max_evals": 8,
        "csv_out": "build/benchmark_diff_mppi_exact_time_fused.csv",
        "summary_title": "Diff-MPPI Exact-Time Summary (heavy feedback baseline)",
    },
    "dynamic_nav_covariance": {
        "bin": "./bin/benchmark_diff_mppi",
        "scenarios": "dynamic_crossing,dynamic_slalom",
        "planners": "mppi,feedback_mppi_cov,diff_mppi_3",
        "time_targets": "1.5,2.0",
        "search_seed_count": 4,
        "final_seed_count": 4,
        "k_min": 64,
        "k_max": 512,
        "tolerance_ms": 0.05,
        "max_evals": 8,
        "csv_out": "build/benchmark_diff_mppi_exact_time_cov.csv",
        "summary_title": "Diff-MPPI Exact-Time Summary (covariance baseline)",
    },
    "dynamic_bicycle": {
        "bin": "./bin/benchmark_diff_mppi_dynamic_bicycle",
        "scenarios": "dynbike_crossing,dynbike_slalom",
        "planners": "mppi,feedback_mppi_sens,diff_mppi_1,diff_mppi_3",
        "time_targets": "1.8,2.0",
        "search_seed_count": 1,
        "final_seed_count": 4,
        "k_min": 4,
        "k_max": 16384,
        "tolerance_ms": 0.06,
        "max_evals": 9,
        "csv_out": "build/benchmark_diff_mppi_dynamic_bicycle_exact_time.csv",
        "summary_title": "Diff-MPPI Dynamic-Bicycle Exact-Time Tuning Summary",
    },
    "manipulator_pilot": {
        "bin": "./bin/benchmark_diff_mppi_manipulator",
        "scenarios": "arm_static_shelf,arm_dynamic_sweep",
        "planners": "mppi,feedback_mppi_cov,feedback_mppi_ref,diff_mppi_1,diff_mppi_3",
        "time_targets": "2.0,3.0",
        "search_seed_count": 1,
        "final_seed_count": 4,
        "k_min": 64,
        "k_max": 4096,
        "tolerance_ms": 0.06,
        "max_evals": 8,
        "csv_out": "build/benchmark_diff_mppi_manipulator_exact_time.csv",
        "summary_title": "Diff-MPPI Manipulator Exact-Time Tuning Summary",
    },
    "7dof_manipulator": {
        "bin": "./bin/benchmark_diff_mppi_manipulator_7dof",
        "scenarios": "7dof_shelf_reach,7dof_dynamic_avoid",
        "planners": "mppi,feedback_mppi_ref,diff_mppi_1,diff_mppi_3",
        "time_targets": "3.0,5.0",
        "search_seed_count": 2,
        "final_seed_count": 4,
        "k_min": 32,
        "k_max": 4096,
        "tolerance_ms": 0.08,
        "max_evals": 8,
        "csv_out": "build/benchmark_diff_mppi_7dof_exact_time.csv",
        "summary_title": "Diff-MPPI 7-DOF Manipulator Exact-Time Tuning Summary",
    },
    "uncertain_dynamic_nav": {
        "bin": "./bin/benchmark_diff_mppi",
        "scenarios": "uncertain_crossing,uncertain_slalom",
        "planners": "mppi,feedback_mppi,feedback_mppi_cov,diff_mppi_1,diff_mppi_3",
        "time_targets": "1.0,1.5,2.0",
        "search_seed_count": 4,
        "final_seed_count": 4,
        "k_min": 128,
        "k_max": 16384,
        "tolerance_ms": 0.03,
        "max_evals": 8,
        "csv_out": "build/benchmark_diff_mppi_uncertain_exact_time.csv",
        "summary_title": "Diff-MPPI Uncertain Dynamic Exact-Time Tuning Summary",
    },
    "mujoco_pendulum": {
        "bin": "./bin/benchmark_diff_mppi_mujoco",
        "scenarios": "inverted_pendulum_v4,inverted_pendulum_wide_reset",
        "planners": "mppi,feedback_mppi_ref,diff_mppi_3",
        "time_targets": "1.0,1.5",
        "search_seed_count": 2,
        "final_seed_count": 5,
        "k_min": 128,
        "k_max": 16384,
        "tolerance_ms": 0.06,
        "max_evals": 8,
        "csv_out": "build/benchmark_diff_mppi_mujoco_exact_time.csv",
        "summary_title": "Diff-MPPI MuJoCo Exact-Time Tuning Summary",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune exact wall-clock-matched K values for benchmark_diff_mppi.",
        epilog=(
            "Multi-param mode (--multi-param) searches over additional controller "
            "parameters beyond K.  For each planner family, it tries a grid of "
            "hyperparameters and picks the combination with the best exact-time "
            "rank (timing gap first, then success/final distance).  Expected "
            "runtime: roughly "
            "N_param_combos x single-param runtime.  For the default grids this is "
            "~3x for feedback baselines, ~9x for diff_mppi variants, and ~3x for "
            "step_mppi.  A full dynamic_nav preset with --multi-param may take "
            "30-60 minutes on a modern GPU."
        ),
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="dynamic_nav",
                        help="Named benchmark preset for defaults")
    parser.add_argument("--bin", help="Path to benchmark binary")
    parser.add_argument("--scenarios", help="Comma-separated scenario names")
    parser.add_argument("--planners", help="Comma-separated planner names")
    parser.add_argument("--time-targets", help="Comma-separated target controller times in ms")
    parser.add_argument("--search-seed-count", type=int, help="Episodes per search evaluation")
    parser.add_argument("--seed-count", type=int, help="Episodes per final tuned evaluation")
    parser.add_argument("--k-min", type=int, help="Minimum K for search")
    parser.add_argument("--k-max", type=int, help="Maximum K for search")
    parser.add_argument("--tolerance-ms", type=float, help="Target timing tolerance in ms")
    parser.add_argument("--max-evals", type=int, help="Maximum search evaluations per scenario/planner/target")
    parser.add_argument("--csv-out", help="Detailed tuned CSV output")
    parser.add_argument("--search-csv-out", help="Optional CSV of all search evaluations")
    parser.add_argument("--summary-out", help="Markdown summary output path")
    parser.add_argument("--summary-title", help="Markdown summary title")
    parser.add_argument("--cache-dir", help="Optional persistent directory for per-K benchmark CSV cache")
    parser.add_argument("--multi-param", action="store_true", default=False,
                        help="Enable multi-parameter search beyond K. "
                             "Searches over feedback_gain_scale, grad_steps/alpha, "
                             "and mlp_lr grids per planner family. "
                             "Runtime scales by the number of parameter combinations "
                             "(~3-9x slower depending on planner family).")
    return parser.parse_args()


def parse_string_list(text):
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(token)
    deduped = []
    seen = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def rank_summary(summary):
    return (
        -summary["success_mean"],
        summary["final_distance_mean"],
        summary["cumulative_cost_mean"],
        summary["steps_mean"],
        summary["avg_control_ms_mean"],
    )


def exact_time_rank(summary, target_ms):
    return (
        abs(summary["avg_control_ms_mean"] - target_ms),
        -summary["success_mean"],
        summary["final_distance_mean"],
        summary["cumulative_cost_mean"],
        summary["steps_mean"],
        summary["k_samples"],
    )


class BenchmarkCache:
    def __init__(self, bin_path, seed_count, workdir, override_args=None, cache_dir=None):
        self.bin_path = Path(bin_path)
        self.seed_count = seed_count
        self.workdir = Path(workdir)
        self.override_args = tuple(override_args) if override_args else ()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache = {}

    def evaluate(self, scenario, planner, k_samples, temp_dir):
        key = (scenario, planner, int(k_samples), self.seed_count, self.override_args)
        if key in self.cache:
            return self.cache[key]

        override_tag = "_".join(self.override_args).replace("--", "").replace(" ", "_") if self.override_args else ""
        csv_name = f"{scenario}__{planner}__k{int(k_samples)}__s{self.seed_count}"
        if override_tag:
            csv_name += f"__{override_tag}"
        base_dir = self.cache_dir if self.cache_dir else Path(temp_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        csv_path = base_dir / f"{csv_name}.csv"
        if not csv_path.exists():
            cmd = [
                str(self.bin_path),
                "--scenarios", scenario,
                "--planners", planner,
                "--seed-count", str(self.seed_count),
                "--k-values", str(int(k_samples)),
                "--csv", str(csv_path),
            ]
            cmd.extend(self.override_args)
            subprocess.run(
                cmd,
                cwd=self.workdir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        rows = load_rows(csv_path)
        summary_rows = summarize_groups(rows)
        if len(summary_rows) != 1:
            raise RuntimeError(f"Expected one summary row for {scenario}/{planner}/K={k_samples}, got {len(summary_rows)}")
        record = {
            "scenario": scenario,
            "planner": planner,
            "k_samples": int(k_samples),
            "summary": summary_rows[0],
            "rows": rows,
        }
        self.cache[key] = record
        return record


def tune_exact_target(cache, scenario, planner, target_ms, k_min, k_max, tolerance_ms, max_evals, temp_dir):
    evaluated = {}

    def evaluate(k_value):
        k_value = max(k_min, min(k_max, int(round(k_value))))
        if k_value not in evaluated:
            evaluated[k_value] = cache.evaluate(scenario, planner, k_value, temp_dir)
        return evaluated[k_value]

    low = k_min
    high = k_max
    low_record = evaluate(low)
    high_record = evaluate(high)
    low_time = low_record["summary"]["avg_control_ms_mean"]
    high_time = high_record["summary"]["avg_control_ms_mean"]

    if low_time >= target_ms or low == high:
        best = min((record["summary"] for record in evaluated.values()), key=lambda row: exact_time_rank(row, target_ms))
        return best, list(evaluated.values())
    if high_time <= target_ms:
        best = min((record["summary"] for record in evaluated.values()), key=lambda row: exact_time_rank(row, target_ms))
        return best, list(evaluated.values())

    while len(evaluated) < max_evals:
        if abs(low_time - target_ms) <= tolerance_ms or abs(high_time - target_ms) <= tolerance_ms:
            break
        if high - low <= 1:
            break

        slope = high_time - low_time
        if slope <= 1.0e-6:
            guess = (low + high) // 2
        else:
            guess = int(round(low + (target_ms - low_time) * (high - low) / slope))
            guess = max(low + 1, min(high - 1, guess))

        if guess in evaluated:
            guess = (low + high) // 2
        if guess <= low or guess >= high or guess in evaluated:
            break

        record = evaluate(guess)
        guess_time = record["summary"]["avg_control_ms_mean"]
        if guess_time <= target_ms:
            low = guess
            low_time = guess_time
        else:
            high = guess
            high_time = guess_time

    best_summary = min((record["summary"] for record in evaluated.values()), key=lambda row: exact_time_rank(row, target_ms))
    return best_summary, list(evaluated.values())


FEEDBACK_GAIN_SCALE_GRID = [0.5, 1.0, 2.0]
DIFF_GRAD_STEPS_GRID = [1, 3, 5]
DIFF_ALPHA_GRID = [0.003, 0.006, 0.012]
STEP_MLP_LR_GRID = [0.0005, 0.001, 0.002]


def param_combos_for_planner(planner):
    """Return list of (override_args, param_dict) for multi-param search."""
    family = planner_family(planner)
    if family == "feedback":
        combos = []
        for fgs in FEEDBACK_GAIN_SCALE_GRID:
            override_args = ["--override-feedback-gain-scale", str(fgs)]
            params = {"feedback_gain_scale": fgs, "grad_steps": "", "alpha": "", "mlp_lr": ""}
            combos.append((override_args, params))
        return combos
    if family == "diff":
        combos = []
        for gs in DIFF_GRAD_STEPS_GRID:
            for alpha in DIFF_ALPHA_GRID:
                override_args = [
                    "--override-grad-steps", str(gs),
                    "--override-alpha", str(alpha),
                ]
                params = {"feedback_gain_scale": "", "grad_steps": gs, "alpha": alpha, "mlp_lr": ""}
                combos.append((override_args, params))
        return combos
    if planner == "step_mppi":
        combos = []
        for lr in STEP_MLP_LR_GRID:
            override_args = ["--override-mlp-lr", str(lr)]
            params = {"feedback_gain_scale": "", "grad_steps": "", "alpha": "", "mlp_lr": lr}
            combos.append((override_args, params))
        return combos
    # mppi or other: no extra params to tune
    return [([], {"feedback_gain_scale": "", "grad_steps": "", "alpha": "", "mlp_lr": ""})]


def default_param_dict():
    return {"feedback_gain_scale": "", "grad_steps": "", "alpha": "", "mlp_lr": ""}


def aggregate_selected(selected_rows):
    groups = defaultdict(list)
    for row in selected_rows:
        groups[(row["planner"], row["time_target_ms"])].append(row)

    aggregate_rows = []
    for (planner, target_ms), group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        item = {
            "planner": planner,
            "time_target_ms": target_ms,
            "num_scenarios": len(group),
            "k_samples_mean": sum(row["k_samples"] for row in group) / len(group),
            "time_gap_ms_mean": sum(abs(row["time_gap_ms"]) for row in group) / len(group),
        }
        for metric in SUMMARY_METRICS:
            key = f"{metric}_mean"
            values = [row[key] for row in group]
            mean_value = sum(values) / len(values)
            variance = 0.0
            if len(values) > 1:
                variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
            item[key] = mean_value
            item[f"{metric}_std"] = variance ** 0.5
        aggregate_rows.append(item)
    return aggregate_rows


def best_family_rows(selected_rows, family):
    result = []
    grouped = defaultdict(list)
    for row in selected_rows:
        grouped[(row["scenario"], row["time_target_ms"])].append(row)

    for (scenario, target_ms), group in sorted(grouped.items()):
        mppi = next((row for row in group if row["planner"] == "mppi"), None)
        family_rows = [row for row in group if planner_family(row["planner"]) == family]
        if not mppi or not family_rows:
            continue
        best = min(family_rows, key=lambda row: rank_summary(row))
        result.append({
            "scenario": scenario,
            "time_target_ms": target_ms,
            "mppi": mppi,
            "best": best,
            "delta_success": best["success_mean"] - mppi["success_mean"],
            "delta_steps": best["steps_mean"] - mppi["steps_mean"],
            "delta_final_distance": best["final_distance_mean"] - mppi["final_distance_mean"],
            "delta_cost": best["cumulative_cost_mean"] - mppi["cumulative_cost_mean"],
        })
    return result


def write_selected_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "planner",
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
        "time_target_ms",
        "time_gap_ms",
        "feedback_gain_scale",
        "tuned_grad_steps",
        "tuned_alpha",
        "mlp_lr",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_search_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "planner",
        "time_target_ms",
        "k_samples",
        "success_mean",
        "steps_mean",
        "final_distance_mean",
        "cumulative_cost_mean",
        "avg_control_ms_mean",
        "time_gap_ms",
        "feedback_gain_scale",
        "tuned_grad_steps",
        "tuned_alpha",
        "mlp_lr",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_pm(mean_value, std_value, digits=2):
    return f"{mean_value:.{digits}f} ± {std_value:.{digits}f}"


def build_markdown(selected_rows, aggregate_rows, diff_rows, feedback_rows, search_rows, csv_out, summary_title):
    lines = []
    lines.append(f"# {summary_title}")
    lines.append("")
    lines.append(f"Selected tuned CSV: `{csv_out}`")
    lines.append("")
    lines.append("## Aggregate Across Exact-Time Targets")
    lines.append("")
    lines.append("| Target ms | Planner | Scenarios | Success | Steps | Final Dist | Cum. Cost | Avg Control ms | Mean K | Mean |Gap| ms |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in aggregate_rows:
        lines.append(
            f"| {row['time_target_ms']:.2f} | {row['planner']} | {row['num_scenarios']} | "
            f"{fmt_pm(row['success_mean'], row['success_std'], 2)} | "
            f"{fmt_pm(row['steps_mean'], row['steps_std'], 1)} | "
            f"{fmt_pm(row['final_distance_mean'], row['final_distance_std'], 2)} | "
            f"{fmt_pm(row['cumulative_cost_mean'], row['cumulative_cost_std'], 1)} | "
            f"{fmt_pm(row['avg_control_ms_mean'], row['avg_control_ms_std'], 2)} | "
            f"{row['k_samples_mean']:.0f} | {row['time_gap_ms_mean']:.3f} |"
        )
    lines.append("")
    lines.append("## Best Diff Variant at Exact-Time Targets")
    lines.append("")
    lines.append("| Scenario | Target ms | MPPI K@ms | Best Diff K@ms | MPPI Dist | Diff Dist | Delta Dist | Delta Steps |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in diff_rows:
        lines.append(
            f"| {row['scenario']} | {row['time_target_ms']:.2f} | "
            f"{row['mppi']['k_samples']} @ {row['mppi']['avg_control_ms_mean']:.3f} | "
            f"{row['best']['planner']} ({row['best']['k_samples']} @ {row['best']['avg_control_ms_mean']:.3f}) | "
            f"{row['mppi']['final_distance_mean']:.2f} | {row['best']['final_distance_mean']:.2f} | "
            f"{row['delta_final_distance']:.2f} | {row['delta_steps']:.1f} |"
        )
    lines.append("")
    lines.append("## Best Feedback Variant at Exact-Time Targets")
    lines.append("")
    lines.append("| Scenario | Target ms | MPPI K@ms | Feedback K@ms | MPPI Dist | Feedback Dist | Delta Dist | Delta Steps |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in feedback_rows:
        lines.append(
            f"| {row['scenario']} | {row['time_target_ms']:.2f} | "
            f"{row['mppi']['k_samples']} @ {row['mppi']['avg_control_ms_mean']:.3f} | "
            f"{row['best']['planner']} ({row['best']['k_samples']} @ {row['best']['avg_control_ms_mean']:.3f}) | "
            f"{row['mppi']['final_distance_mean']:.2f} | {row['best']['final_distance_mean']:.2f} | "
            f"{row['delta_final_distance']:.2f} | {row['delta_steps']:.1f} |"
        )
    lines.append("")
    lines.append("## Search Trace")
    lines.append("")
    lines.append("| Scenario | Planner | Target ms | K | Avg Control ms | |Gap| ms | Success | Final Dist |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in search_rows:
        lines.append(
            f"| {row['scenario']} | {row['planner']} | {row['time_target_ms']:.2f} | {row['k_samples']} | "
            f"{row['avg_control_ms_mean']:.3f} | {abs(row['time_gap_ms']):.3f} | "
            f"{row['success_mean']:.2f} | {row['final_distance_mean']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _tune_single_param(search_cache, final_cache, scenario, planner, target_ms,
                       k_min, k_max, tolerance_ms, max_evals, temp_dir, param_dict):
    """Run K binary search and collect results for one parameter combination.

    Returns (best_summary, final_record, search_trace, param_dict) where
    best_summary is the search-phase winner and final_record is re-evaluated
    at the selected K with final_seed_count.
    """
    best_summary, evaluated_records = tune_exact_target(
        search_cache, scenario, planner, target_ms,
        k_min, k_max, tolerance_ms, max_evals, temp_dir,
    )
    final_record = final_cache.evaluate(scenario, planner, best_summary["k_samples"], temp_dir)

    trace = []
    for record in sorted(evaluated_records, key=lambda item: item["k_samples"]):
        summary = record["summary"]
        row = {
            "scenario": scenario,
            "planner": planner,
            "time_target_ms": target_ms,
            "k_samples": summary["k_samples"],
            "success_mean": summary["success_mean"],
            "steps_mean": summary["steps_mean"],
            "final_distance_mean": summary["final_distance_mean"],
            "cumulative_cost_mean": summary["cumulative_cost_mean"],
            "avg_control_ms_mean": summary["avg_control_ms_mean"],
            "time_gap_ms": summary["avg_control_ms_mean"] - target_ms,
        }
        row.update(_tuned_param_columns(param_dict))
        trace.append(row)

    return best_summary, final_record, trace, param_dict


def _tuned_param_columns(param_dict):
    """Map multi-param dict keys to CSV column names."""
    return {
        "feedback_gain_scale": param_dict.get("feedback_gain_scale", ""),
        "tuned_grad_steps": param_dict.get("grad_steps", ""),
        "tuned_alpha": param_dict.get("alpha", ""),
        "mlp_lr": param_dict.get("mlp_lr", ""),
    }


def main():
    args = parse_args()
    preset = PRESETS[args.preset]
    workdir = Path.cwd()
    bin_path = Path(args.bin if args.bin else preset["bin"])
    if not bin_path.is_absolute():
        bin_path = (workdir / bin_path).resolve()
    if not bin_path.exists():
        raise SystemExit(f"Benchmark binary not found: {bin_path}")

    scenarios = parse_string_list(args.scenarios if args.scenarios else preset["scenarios"])
    planners = parse_string_list(args.planners if args.planners else preset["planners"])
    time_targets = parse_float_list(args.time_targets if args.time_targets else preset["time_targets"])
    search_seed_count = max(1, args.search_seed_count if args.search_seed_count is not None else preset["search_seed_count"])
    final_seed_count = max(1, args.seed_count if args.seed_count is not None else preset["final_seed_count"])
    k_min = max(1, args.k_min if args.k_min is not None else preset["k_min"])
    k_max = max(k_min, args.k_max if args.k_max is not None else preset["k_max"])
    tolerance_ms = args.tolerance_ms if args.tolerance_ms is not None else preset["tolerance_ms"]
    max_evals = args.max_evals if args.max_evals is not None else preset["max_evals"]
    summary_title = args.summary_title if args.summary_title else preset["summary_title"]
    multi_param = args.multi_param
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir and not cache_dir.is_absolute():
        cache_dir = (workdir / cache_dir).resolve()

    selected_summary_rows = []
    selected_episode_rows = []
    search_trace_rows = []

    # Build a cache per override_args tuple so evaluations with the same
    # overrides are shared across scenarios/targets (preserving the original
    # cross-scenario caching when multi-param is off).
    search_cache_map = {}  # override_args tuple -> BenchmarkCache
    final_cache_map = {}

    def get_search_cache(override_args):
        key = tuple(override_args) if override_args else ()
        if key not in search_cache_map:
            search_cache_map[key] = BenchmarkCache(
                bin_path, search_seed_count, workdir,
                override_args=override_args, cache_dir=cache_dir,
            )
        return search_cache_map[key]

    def get_final_cache(override_args):
        key = tuple(override_args) if override_args else ()
        if key not in final_cache_map:
            final_cache_map[key] = BenchmarkCache(
                bin_path, final_seed_count, workdir,
                override_args=override_args, cache_dir=cache_dir,
            )
        return final_cache_map[key]

    with tempfile.TemporaryDirectory(prefix="diff_mppi_exact_time_", dir=str(workdir / "build")) as temp_dir:
        for target_ms in time_targets:
            for scenario in scenarios:
                for planner in planners:
                    if multi_param:
                        combos = param_combos_for_planner(planner)
                    else:
                        combos = [([], default_param_dict())]

                    best_candidate = None  # (rank_tuple, summary, final_record, param_dict)

                    for override_args, param_dict in combos:
                        search_cache = get_search_cache(override_args)
                        final_cache = get_final_cache(override_args)

                        best_summary, final_record, trace, pd = _tune_single_param(
                            search_cache, final_cache, scenario, planner, target_ms,
                            k_min, k_max, tolerance_ms, max_evals, temp_dir, param_dict,
                        )

                        # Always record search trace for all combos
                        search_trace_rows.extend(trace)

                        final_summary = final_record["summary"]
                        final_dist = final_summary["final_distance_mean"]
                        final_success = final_summary["success_mean"]
                        candidate_rank = exact_time_rank(final_summary, target_ms)

                        if best_candidate is None or candidate_rank < best_candidate[0]:
                            best_candidate = (candidate_rank, best_summary, final_record, pd)

                        if multi_param and len(combos) > 1:
                            combo_label = ", ".join(f"{k}={v}" for k, v in param_dict.items() if v != "")
                            print(f"  [{scenario}/{planner}@{target_ms}ms] combo ({combo_label}): "
                                  f"K={final_record['summary']['k_samples']} "
                                  f"dist={final_dist:.2f} success={final_success:.2f}")

                    _, best_summary, final_record, winning_params = best_candidate

                    selected_row = dict(final_record["summary"])
                    selected_row["scenario"] = scenario
                    selected_row["planner"] = planner
                    selected_row["time_target_ms"] = target_ms
                    selected_row["time_gap_ms"] = final_record["summary"]["avg_control_ms_mean"] - target_ms
                    selected_row.update(_tuned_param_columns(winning_params))
                    selected_summary_rows.append(selected_row)

                    for row in final_record["rows"]:
                        episode_row = dict(row)
                        episode_row["time_target_ms"] = target_ms
                        episode_row["time_gap_ms"] = final_record["summary"]["avg_control_ms_mean"] - target_ms
                        episode_row.update(_tuned_param_columns(winning_params))
                        selected_episode_rows.append(episode_row)

                    if multi_param and len(param_combos_for_planner(planner)) > 1:
                        combo_label = ", ".join(f"{k}={v}" for k, v in winning_params.items() if v != "")
                        print(f"  -> BEST [{scenario}/{planner}@{target_ms}ms]: ({combo_label}) "
                              f"K={final_record['summary']['k_samples']}")

    csv_out = Path(args.csv_out if args.csv_out else preset["csv_out"])
    summary_out = Path(args.summary_out) if args.summary_out else csv_out.with_name(f"{csv_out.stem}_summary.md")
    search_csv_out = Path(args.search_csv_out) if args.search_csv_out else csv_out.with_name(f"{csv_out.stem}_search.csv")

    write_selected_csv(selected_episode_rows, csv_out)
    write_search_csv(search_trace_rows, search_csv_out)

    aggregate_rows = aggregate_selected(selected_summary_rows)
    diff_rows = best_family_rows(selected_summary_rows, "diff")
    feedback_rows = best_family_rows(selected_summary_rows, "feedback")
    markdown = build_markdown(
        selected_summary_rows,
        aggregate_rows,
        diff_rows,
        feedback_rows,
        search_trace_rows,
        csv_out,
        summary_title,
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(markdown)

    print(f"Exact-time selected CSV saved to {csv_out}")
    print(f"Exact-time search trace saved to {search_csv_out}")
    print(f"Exact-time summary saved to {summary_out}")


if __name__ == "__main__":
    main()
