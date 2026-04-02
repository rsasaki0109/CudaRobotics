#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import inspect
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.planner_selector_interface import AggregateBenchmarkRow, Recommendation, SelectionRequest
from core.time_budget_selector_interface import TimeBudgetRecommendation, TimeBudgetRequest
from experiments.planner_selection import build_variants as build_planner_variants
from experiments.time_budget_selection import build_variants as build_time_budget_variants


DEFAULT_DATASETS = [
    "experiments/data/benchmark_diff_mppi.csv",
    "experiments/data/benchmark_diff_mppi_uncertain.csv",
    "experiments/data/benchmark_diff_mppi_dynamic_bicycle.csv",
]


@dataclass(frozen=True)
class StaticCodeMetrics:
    loc: int
    comment_lines: int
    function_count: int
    class_count: int
    dataclass_count: int
    dataclass_fields: int
    branch_count: int
    max_depth: int
    avg_function_len: float
    longest_function_len: int


@dataclass(frozen=True)
class VariantEvaluation:
    name: str
    paradigm: str
    metrics: dict[str, object]
    cases: list[dict[str, object]]


@dataclass(frozen=True)
class ProblemReport:
    slug: str
    title: str
    description_lines: list[str]
    request_summary: str
    metric_notes: list[str]
    request_count: int
    variant_results: list[VariantEvaluation]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment-first design comparisons across concrete design problems.")
    parser.add_argument(
        "--csv",
        nargs="*",
        help="Benchmark CSV files to aggregate. Defaults to the checked-in Diff-MPPI fixture CSVs.",
    )
    parser.add_argument(
        "--docs-dir",
        default="docs",
        help="Output directory for generated markdown docs.",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=200,
        help="How many repeated recommendation passes to use for runtime benchmarking.",
    )
    return parser.parse_args()


def discover_csvs(explicit_paths: list[str] | None) -> list[Path]:
    if explicit_paths:
        paths = [ROOT / path for path in explicit_paths]
    else:
        paths = [ROOT / path for path in DEFAULT_DATASETS]
    return [path for path in paths if path.exists()]


def load_rows(csv_paths: list[Path]) -> list[AggregateBenchmarkRow]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, float]]] = defaultdict(list)
    for path in csv_paths:
        dataset = path.stem
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (
                    dataset,
                    row["scenario"],
                    row["planner"],
                    int(row["k_samples"]),
                )
                grouped[key].append(
                    {
                        "success": float(row["success"]),
                        "steps": float(row["steps"]),
                        "final_distance": float(row["final_distance"]),
                        "cumulative_cost": float(row["cumulative_cost"]),
                        "avg_control_ms": float(row["avg_control_ms"]),
                    }
                )

    aggregated: list[AggregateBenchmarkRow] = []
    for (dataset, scenario, planner, k_samples), values in sorted(grouped.items()):
        count = float(len(values))
        aggregated.append(
            AggregateBenchmarkRow(
                dataset=dataset,
                scenario=scenario,
                planner=planner,
                k_samples=k_samples,
                success=sum(item["success"] for item in values) / count,
                steps=sum(item["steps"] for item in values) / count,
                final_distance=sum(item["final_distance"] for item in values) / count,
                cumulative_cost=sum(item["cumulative_cost"] for item in values) / count,
                avg_control_ms=sum(item["avg_control_ms"] for item in values) / count,
            )
        )
    return aggregated


def build_selection_requests(rows: Sequence[AggregateBenchmarkRow]) -> list[SelectionRequest]:
    keys = {(row.dataset, row.scenario) for row in rows}
    return [SelectionRequest(dataset=dataset, scenario=scenario) for dataset, scenario in sorted(keys)]


def select_budget_levels(runtimes: Sequence[float]) -> list[float]:
    unique = sorted(set(runtimes))
    if not unique:
        return []

    levels: list[float] = []
    for fraction in (0.0, 0.50, 0.80):
        index = round(fraction * (len(unique) - 1))
        candidate = unique[index]
        if not any(abs(candidate - value) < 1.0e-9 for value in levels):
            levels.append(candidate)

    for candidate in unique:
        if len(levels) >= min(3, len(unique)):
            break
        if not any(abs(candidate - value) < 1.0e-9 for value in levels):
            levels.append(candidate)

    return sorted(levels)


def build_time_budget_requests(rows: Sequence[AggregateBenchmarkRow]) -> list[TimeBudgetRequest]:
    grouped: dict[tuple[str, str], list[AggregateBenchmarkRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.dataset, row.scenario)].append(row)

    requests: list[TimeBudgetRequest] = []
    for dataset, scenario in sorted(grouped):
        budgets = select_budget_levels([row.avg_control_ms for row in grouped[(dataset, scenario)]])
        for budget in budgets:
            requests.append(
                TimeBudgetRequest(
                    dataset=dataset,
                    scenario=scenario,
                    time_budget_ms=budget,
                )
            )
    return requests


def normalize(values: Sequence[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if high - low < 1.0e-9:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def utility_map(candidates: Sequence[AggregateBenchmarkRow]) -> dict[tuple[str, int], float]:
    final_distance_norm = normalize([row.final_distance for row in candidates])
    cumulative_cost_norm = normalize([row.cumulative_cost for row in candidates])
    avg_control_ms_norm = normalize([row.avg_control_ms for row in candidates])
    steps_norm = normalize([row.steps for row in candidates])

    utilities: dict[tuple[str, int], float] = {}
    for index, row in enumerate(candidates):
        utilities[(row.planner, row.k_samples)] = (
            5.0 * row.success
            - 2.0 * final_distance_norm[index]
            - 1.0 * cumulative_cost_norm[index]
            - 0.75 * avg_control_ms_norm[index]
            - 0.50 * steps_norm[index]
        )
    return utilities


def find_row(rows: Sequence[AggregateBenchmarkRow], planner: str, k_samples: int) -> AggregateBenchmarkRow:
    for row in rows:
        if row.planner == planner and row.k_samples == k_samples:
            return row
    raise ValueError(f"Selected row {planner} @ K={k_samples} is not present in the candidate set")


def compute_code_metrics(path: Path) -> StaticCodeMetrics:
    source = path.read_text()
    tree = ast.parse(source)
    lines = source.splitlines()

    function_lengths: list[int] = []
    branch_count = 0
    class_count = 0
    dataclass_count = 0
    dataclass_fields = 0
    max_depth = 0

    def visit(node: ast.AST, depth: int) -> None:
        nonlocal branch_count, class_count, dataclass_count, dataclass_fields, max_depth
        max_depth = max(max_depth, depth)
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.Match, ast.IfExp)):
            branch_count += 1
        if isinstance(node, ast.FunctionDef):
            if hasattr(node, "end_lineno") and node.end_lineno is not None:
                function_lengths.append(node.end_lineno - node.lineno + 1)
        if isinstance(node, ast.ClassDef):
            class_count += 1
            is_dataclass = False
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                    is_dataclass = True
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "dataclass":
                    is_dataclass = True
            if is_dataclass:
                dataclass_count += 1
                dataclass_fields += sum(isinstance(child, ast.AnnAssign) for child in node.body)
        for child in ast.iter_child_nodes(node):
            visit(child, depth + 1)

    visit(tree, 0)

    loc = 0
    comment_lines = 0
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comment_lines += 1
            continue
        loc += 1

    avg_function_len = sum(function_lengths) / len(function_lengths) if function_lengths else 0.0
    longest_function_len = max(function_lengths) if function_lengths else 0

    return StaticCodeMetrics(
        loc=loc,
        comment_lines=comment_lines,
        function_count=len(function_lengths),
        class_count=class_count,
        dataclass_count=dataclass_count,
        dataclass_fields=dataclass_fields,
        branch_count=branch_count,
        max_depth=max_depth,
        avg_function_len=avg_function_len,
        longest_function_len=longest_function_len,
    )


def readability_score(metrics: StaticCodeMetrics) -> float:
    score = (
        100.0
        - 0.18 * metrics.loc
        - 1.1 * metrics.branch_count
        - 2.0 * metrics.max_depth
        - 0.30 * metrics.avg_function_len
        + 0.6 * metrics.comment_lines
    )
    return max(0.0, min(100.0, score))


def extensibility_score(metrics: StaticCodeMetrics) -> float:
    score = (
        20.0
        + 10.0 * metrics.dataclass_count
        + 1.8 * metrics.dataclass_fields
        + 3.0 * metrics.function_count
        + 4.0 * metrics.class_count
        - 0.40 * metrics.longest_function_len
        - 1.2 * max(0, metrics.max_depth - 3)
    )
    return max(0.0, min(100.0, score))


def benchmark_variant(variant, rows: Sequence[AggregateBenchmarkRow], requests: Sequence[object], iterations: int) -> float:
    begin = time.perf_counter()
    for _ in range(iterations):
        for request in requests:
            variant.recommend(rows, request)
    elapsed = time.perf_counter() - begin
    return elapsed * 1000.0 / max(1, iterations * len(requests))


def variant_metric_block(variant, runtime_ms: float) -> dict[str, object]:
    source_path = Path(inspect.getsourcefile(variant.__class__) or "")
    static_metrics = compute_code_metrics(source_path)
    return {
        "runtime_ms_per_request": runtime_ms,
        "readability_score": readability_score(static_metrics),
        "extensibility_score": extensibility_score(static_metrics),
        "loc": float(static_metrics.loc),
        "branch_count": float(static_metrics.branch_count),
        "max_depth": float(static_metrics.max_depth),
        "source_path": str(source_path.relative_to(ROOT)),
    }


def evaluate_planner_variant(
    variant,
    rows: Sequence[AggregateBenchmarkRow],
    requests: Sequence[SelectionRequest],
    iterations: int,
) -> VariantEvaluation:
    cases: list[dict[str, object]] = []
    total_regret = 0.0
    match_count = 0

    for request in requests:
        candidates = [row for row in rows if row.dataset == request.dataset and row.scenario == request.scenario]
        utilities = utility_map(candidates)
        best_key, best_utility = max(utilities.items(), key=lambda item: item[1])

        recommendation: Recommendation = variant.recommend(rows, request)
        selected_key = (recommendation.planner, recommendation.k_samples)
        selected_utility = utilities[selected_key]
        regret = best_utility - selected_utility
        total_regret += regret
        if selected_key == best_key:
            match_count += 1

        cases.append(
            {
                "dataset": request.dataset,
                "scenario": request.scenario,
                "planner": recommendation.planner,
                "k_samples": recommendation.k_samples,
                "regret": regret,
                "oracle_planner": best_key[0],
                "oracle_k_samples": best_key[1],
                "rationale": recommendation.rationale,
            }
        )

    runtime_ms = benchmark_variant(variant, rows, requests, iterations)
    metrics = variant_metric_block(variant, runtime_ms)
    metrics.update(
        {
            "avg_regret": total_regret / max(1, len(requests)),
            "oracle_match_rate": match_count / max(1, len(requests)),
        }
    )
    return VariantEvaluation(name=variant.name, paradigm=variant.paradigm, metrics=metrics, cases=cases)


def evaluate_time_budget_variant(
    variant,
    rows: Sequence[AggregateBenchmarkRow],
    requests: Sequence[TimeBudgetRequest],
    iterations: int,
) -> VariantEvaluation:
    cases: list[dict[str, object]] = []
    total_regret = 0.0
    match_count = 0
    budget_hit_count = 0

    for request in requests:
        candidates = [row for row in rows if row.dataset == request.dataset and row.scenario == request.scenario]
        feasible = [row for row in candidates if row.avg_control_ms <= request.time_budget_ms + 1.0e-9]
        if not feasible:
            raise RuntimeError(f"No feasible candidates for {request.dataset}/{request.scenario} at {request.time_budget_ms:.4f} ms")

        utilities = utility_map(feasible)
        best_key, best_utility = max(utilities.items(), key=lambda item: item[1])
        oracle_row = find_row(feasible, best_key[0], best_key[1])

        recommendation: TimeBudgetRecommendation = variant.recommend(rows, request)
        selected_key = (recommendation.planner, recommendation.k_samples)
        selected_row = find_row(candidates, recommendation.planner, recommendation.k_samples)
        budget_hit = selected_row.avg_control_ms <= request.time_budget_ms + 1.0e-9
        budget_hit_count += 1 if budget_hit else 0

        if budget_hit and selected_key in utilities:
            selected_utility = utilities[selected_key]
        else:
            violation = max(0.0, selected_row.avg_control_ms - request.time_budget_ms)
            selected_utility = min(utilities.values()) - 10.0 - 50.0 * violation

        regret = best_utility - selected_utility
        total_regret += regret
        if budget_hit and selected_key == best_key:
            match_count += 1

        cases.append(
            {
                "dataset": request.dataset,
                "scenario": request.scenario,
                "time_budget_ms": request.time_budget_ms,
                "planner": recommendation.planner,
                "k_samples": recommendation.k_samples,
                "selected_time_ms": selected_row.avg_control_ms,
                "budget_hit": budget_hit,
                "regret": regret,
                "oracle_planner": oracle_row.planner,
                "oracle_k_samples": oracle_row.k_samples,
                "oracle_time_ms": oracle_row.avg_control_ms,
                "rationale": recommendation.rationale,
            }
        )

    runtime_ms = benchmark_variant(variant, rows, requests, iterations)
    metrics = variant_metric_block(variant, runtime_ms)
    metrics.update(
        {
            "avg_regret": total_regret / max(1, len(requests)),
            "oracle_match_rate": match_count / max(1, len(requests)),
            "budget_hit_rate": budget_hit_count / max(1, len(requests)),
        }
    )
    return VariantEvaluation(name=variant.name, paradigm=variant.paradigm, metrics=metrics, cases=cases)


def build_problem_reports(rows: Sequence[AggregateBenchmarkRow], iterations: int) -> list[ProblemReport]:
    planner_requests = build_selection_requests(rows)
    planner_results = [
        evaluate_planner_variant(variant, rows, planner_requests, iterations)
        for variant in build_planner_variants()
    ]

    time_budget_requests = build_time_budget_requests(rows)
    time_budget_results = [
        evaluate_time_budget_variant(variant, rows, time_budget_requests, iterations)
        for variant in build_time_budget_variants()
    ]

    return [
        ProblemReport(
            slug="planner_selection",
            title="Planner Selection",
            description_lines=[
                "choose one planner configuration per dataset/scenario pair",
                "keep the input schema fixed while varying only the selector implementation style",
                "score each selector on benchmark regret, runtime, readability, and extensibility proxies",
            ],
            request_summary="dataset/scenario pairs",
            metric_notes=[
                "`Avg Regret`: utility gap from an external oracle scorer; lower is better",
                "`Oracle Match`: fraction of requests where the selector picked the oracle row exactly",
                "`Readability` and `Extensibility`: static-analysis proxies, not human review replacements",
            ],
            request_count=len(planner_requests),
            variant_results=planner_results,
        ),
        ProblemReport(
            slug="time_budget_selection",
            title="Time-Budget Selection",
            description_lines=[
                "choose one planner configuration per dataset/scenario/time-budget request",
                "force all variants to consume the same aggregated benchmark rows and the same wall-clock envelopes",
                "score each selector on constrained regret, budget-hit rate, runtime, readability, and extensibility proxies",
            ],
            request_summary="dataset/scenario/time-budget triples",
            metric_notes=[
                "`Avg Regret`: utility gap from the best feasible row under the requested time budget; lower is better",
                "`Oracle Match`: fraction of requests where the selector matched the best feasible row exactly",
                "`Budget Hit`: fraction of requests where the selected row stayed inside the requested `avg_control_ms` envelope",
            ],
            request_count=len(time_budget_requests),
            variant_results=time_budget_results,
        ),
    ]


def generate_experiments_markdown(csv_paths: Sequence[Path], reports: Sequence[ProblemReport]) -> str:
    lines: list[str] = []
    lines.append("# Experiments")
    lines.append("")
    lines.append("_Generated by `python3 scripts/run_design_experiments.py`._")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for path in csv_paths:
        lines.append(f"- `{path.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## Active Problems")
    lines.append("")
    for report in reports:
        lines.append(f"- `{report.slug}`: {report.title}")
    lines.append("")

    for report in reports:
        lines.append(f"## {report.title}")
        lines.append("")
        lines.append(f"Problem id: `{report.slug}`")
        lines.append("")
        lines.append("Concrete problem under comparison:")
        for description in report.description_lines:
            lines.append(f"- {description}")
        lines.append("")
        lines.append(f"Requests evaluated: `{report.request_count}` {report.request_summary}")
        lines.append("")
        lines.append("### Aggregate Scores")
        lines.append("")

        if report.slug == "planner_selection":
            lines.append("| Variant | Paradigm | Avg Regret | Oracle Match | Runtime ms/request | Readability | Extensibility | Source |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
            for item in report.variant_results:
                metrics = item.metrics
                lines.append(
                    f"| {item.name} | {item.paradigm} | "
                    f"{metrics['avg_regret']:.3f} | {metrics['oracle_match_rate']:.2f} | "
                    f"{metrics['runtime_ms_per_request']:.4f} | {metrics['readability_score']:.1f} | "
                    f"{metrics['extensibility_score']:.1f} | `{metrics['source_path']}` |"
                )
        elif report.slug == "time_budget_selection":
            lines.append("| Variant | Paradigm | Avg Regret | Oracle Match | Budget Hit | Runtime ms/request | Readability | Extensibility | Source |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
            for item in report.variant_results:
                metrics = item.metrics
                lines.append(
                    f"| {item.name} | {item.paradigm} | "
                    f"{metrics['avg_regret']:.3f} | {metrics['oracle_match_rate']:.2f} | "
                    f"{metrics['budget_hit_rate']:.2f} | {metrics['runtime_ms_per_request']:.4f} | "
                    f"{metrics['readability_score']:.1f} | {metrics['extensibility_score']:.1f} | "
                    f"`{metrics['source_path']}` |"
                )
        else:
            raise ValueError(f"Unknown problem report: {report.slug}")

        lines.append("")
        lines.append("Metric notes:")
        for note in report.metric_notes:
            lines.append(f"- {note}")
        lines.append("")
        lines.append("### Per-Case Recommendations")
        lines.append("")

        for item in report.variant_results:
            lines.append(f"#### {item.name}")
            lines.append("")
            if report.slug == "planner_selection":
                lines.append("| Dataset | Scenario | Pick | Oracle | Regret | Rationale |")
                lines.append("|---|---|---|---|---:|---|")
                for case in item.cases:
                    lines.append(
                        f"| {case['dataset']} | {case['scenario']} | "
                        f"{case['planner']} @ K={case['k_samples']} | "
                        f"{case['oracle_planner']} @ K={case['oracle_k_samples']} | "
                        f"{case['regret']:.3f} | {case['rationale']} |"
                    )
            elif report.slug == "time_budget_selection":
                lines.append("| Dataset | Scenario | Budget ms | Pick | Pick ms | Oracle | Oracle ms | Budget Hit | Regret | Rationale |")
                lines.append("|---|---|---:|---|---:|---|---:|---|---:|---|")
                for case in item.cases:
                    lines.append(
                        f"| {case['dataset']} | {case['scenario']} | "
                        f"{case['time_budget_ms']:.4f} | "
                        f"{case['planner']} @ K={case['k_samples']} | "
                        f"{case['selected_time_ms']:.4f} | "
                        f"{case['oracle_planner']} @ K={case['oracle_k_samples']} | "
                        f"{case['oracle_time_ms']:.4f} | "
                        f"{'yes' if case['budget_hit'] else 'no'} | "
                        f"{case['regret']:.3f} | {case['rationale']} |"
                    )
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    csv_paths = discover_csvs(args.csv)
    if not csv_paths:
        print("No benchmark CSVs found. Pass --csv or refresh the fixture CSVs in experiments/data/.", file=sys.stderr)
        return 1

    rows = load_rows(csv_paths)
    reports = build_problem_reports(rows, args.benchmark_iterations)

    docs_dir = ROOT / args.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    experiments_md = docs_dir / "experiments.md"
    experiments_md.write_text(generate_experiments_markdown(csv_paths, reports))
    print(f"Generated {experiments_md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
