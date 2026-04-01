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
        "planners": "mppi,feedback_mppi,diff_mppi_1,diff_mppi_3",
        "time_targets": "1.0,1.5",
        "search_seed_count": 4,
        "final_seed_count": 4,
        "k_min": 128,
        "k_max": 16384,
        "tolerance_ms": 0.03,
        "max_evals": 8,
        "csv_out": "build/benchmark_diff_mppi_exact_time.csv",
        "summary_title": "Diff-MPPI Exact-Time Tuning Summary",
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
    "uncertain_dynamic_nav": {
        "bin": "./bin/benchmark_diff_mppi",
        "scenarios": "uncertain_crossing,uncertain_slalom",
        "planners": "mppi,feedback_mppi,diff_mppi_1,diff_mppi_3",
        "time_targets": "1.0,1.5",
        "search_seed_count": 4,
        "final_seed_count": 4,
        "k_min": 128,
        "k_max": 16384,
        "tolerance_ms": 0.03,
        "max_evals": 8,
        "csv_out": "build/benchmark_diff_mppi_uncertain_exact_time.csv",
        "summary_title": "Diff-MPPI Uncertain Dynamic Exact-Time Tuning Summary",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Tune exact wall-clock-matched K values for benchmark_diff_mppi.")
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
    def __init__(self, bin_path, seed_count, workdir):
        self.bin_path = Path(bin_path)
        self.seed_count = seed_count
        self.workdir = Path(workdir)
        self.cache = {}

    def evaluate(self, scenario, planner, k_samples, temp_dir):
        key = (scenario, planner, int(k_samples), self.seed_count)
        if key in self.cache:
            return self.cache[key]

        csv_path = Path(temp_dir) / f"{scenario}__{planner}__k{int(k_samples)}__s{self.seed_count}.csv"
        cmd = [
            str(self.bin_path),
            "--scenarios", scenario,
            "--planners", planner,
            "--seed-count", str(self.seed_count),
            "--k-values", str(int(k_samples)),
            "--csv", str(csv_path),
        ]
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
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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

    search_cache = BenchmarkCache(bin_path, search_seed_count, workdir)
    final_cache = BenchmarkCache(bin_path, final_seed_count, workdir)
    selected_summary_rows = []
    selected_episode_rows = []
    search_trace_rows = []

    with tempfile.TemporaryDirectory(prefix="diff_mppi_exact_time_", dir=str(workdir / "build")) as temp_dir:
        for target_ms in time_targets:
            for scenario in scenarios:
                for planner in planners:
                    best_summary, evaluated_records = tune_exact_target(
                        search_cache,
                        scenario,
                        planner,
                        target_ms,
                        k_min,
                        k_max,
                        tolerance_ms,
                        max_evals,
                        temp_dir,
                    )
                    final_record = final_cache.evaluate(scenario, planner, best_summary["k_samples"], temp_dir)
                    selected_row = dict(final_record["summary"])
                    selected_row["scenario"] = scenario
                    selected_row["planner"] = planner
                    selected_row["time_target_ms"] = target_ms
                    selected_row["time_gap_ms"] = final_record["summary"]["avg_control_ms_mean"] - target_ms
                    selected_summary_rows.append(selected_row)

                    for record in sorted(evaluated_records, key=lambda item: item["k_samples"]):
                        summary = record["summary"]
                        search_trace_rows.append({
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
                        })

                    for row in final_record["rows"]:
                        episode_row = dict(row)
                        episode_row["time_target_ms"] = target_ms
                        episode_row["time_gap_ms"] = final_record["summary"]["avg_control_ms_mean"] - target_ms
                        selected_episode_rows.append(episode_row)

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
