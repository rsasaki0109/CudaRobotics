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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.planner_selector_interface import AggregateBenchmarkRow, Recommendation, SelectionRequest
from experiments.planner_selection import build_variants


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment-first design comparisons for planner-selection variants.")
    parser.add_argument(
        "--csv",
        nargs="*",
        help="Benchmark CSV files to aggregate. Defaults to the current Diff-MPPI benchmark outputs if present.",
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


def build_requests(rows: list[AggregateBenchmarkRow]) -> list[SelectionRequest]:
    keys = {(row.dataset, row.scenario) for row in rows}
    return [SelectionRequest(dataset=dataset, scenario=scenario) for dataset, scenario in sorted(keys)]


def normalize(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if high - low < 1.0e-9:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def utility_map(candidates: list[AggregateBenchmarkRow]) -> dict[tuple[str, int], float]:
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


def benchmark_variant(variant, rows: list[AggregateBenchmarkRow], requests: list[SelectionRequest], iterations: int) -> float:
    begin = time.perf_counter()
    for _ in range(iterations):
        for request in requests:
            variant.recommend(rows, request)
    elapsed = time.perf_counter() - begin
    return elapsed * 1000.0 / max(1, iterations * len(requests))


def evaluate_variant(variant, rows: list[AggregateBenchmarkRow], requests: list[SelectionRequest], iterations: int) -> tuple[dict[str, float], list[Recommendation], list[dict[str, object]]]:
    recommendations: list[Recommendation] = []
    per_case: list[dict[str, object]] = []
    total_regret = 0.0
    match_count = 0

    for request in requests:
        candidates = [row for row in rows if row.dataset == request.dataset and row.scenario == request.scenario]
        utilities = utility_map(candidates)
        best_key, best_utility = max(utilities.items(), key=lambda item: item[1])
        recommendation = variant.recommend(rows, request)
        selected_key = (recommendation.planner, recommendation.k_samples)
        selected_utility = utilities[selected_key]
        regret = best_utility - selected_utility
        total_regret += regret
        if selected_key == best_key:
            match_count += 1
        recommendations.append(recommendation)
        per_case.append(
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
    source_path = Path(inspect.getsourcefile(variant.__class__) or "")
    static_metrics = compute_code_metrics(source_path)
    result = {
        "avg_regret": total_regret / max(1, len(requests)),
        "oracle_match_rate": match_count / max(1, len(requests)),
        "runtime_ms_per_request": runtime_ms,
        "readability_score": readability_score(static_metrics),
        "extensibility_score": extensibility_score(static_metrics),
        "loc": float(static_metrics.loc),
        "branch_count": float(static_metrics.branch_count),
        "max_depth": float(static_metrics.max_depth),
        "source_path": str(source_path.relative_to(ROOT)),
    }
    return result, recommendations, per_case


def generate_experiments_markdown(
    csv_paths: list[Path],
    requests: list[SelectionRequest],
    variant_results: list[dict[str, object]],
) -> str:
    lines: list[str] = []
    lines.append("# Experiments")
    lines.append("")
    lines.append("_Generated by `python3 scripts/run_design_experiments.py`._")
    lines.append("")
    lines.append("## Problem")
    lines.append("")
    lines.append("Concrete problem under comparison:")
    lines.append("- choose one planner configuration per benchmark scenario from the current Diff-MPPI CSV outputs")
    lines.append("- keep the input schema fixed while varying only the implementation style of the selector")
    lines.append("- score each selector on benchmark regret, runtime, readability, and extensibility proxies")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for path in csv_paths:
        lines.append(f"- `{path.relative_to(ROOT)}`")
    lines.append("")
    lines.append(f"Requests evaluated: `{len(requests)}` dataset/scenario pairs")
    lines.append("")
    lines.append("## Aggregate Scores")
    lines.append("")
    lines.append("| Variant | Paradigm | Avg Regret | Oracle Match | Runtime ms/request | Readability | Extensibility | Source |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for item in variant_results:
        metrics = item["metrics"]
        lines.append(
            f"| {item['name']} | {item['paradigm']} | "
            f"{metrics['avg_regret']:.3f} | {metrics['oracle_match_rate']:.2f} | "
            f"{metrics['runtime_ms_per_request']:.4f} | {metrics['readability_score']:.1f} | "
            f"{metrics['extensibility_score']:.1f} | `{metrics['source_path']}` |"
        )
    lines.append("")
    lines.append("Metric notes:")
    lines.append("- `Avg Regret`: utility gap from an external oracle scorer; lower is better")
    lines.append("- `Oracle Match`: fraction of dataset/scenario pairs where the selector picked the oracle row exactly")
    lines.append("- `Readability` and `Extensibility`: static-analysis proxies, not human review replacements")
    lines.append("")
    lines.append("## Per-Case Recommendations")
    lines.append("")
    for item in variant_results:
        lines.append(f"### {item['name']}")
        lines.append("")
        lines.append("| Dataset | Scenario | Pick | Oracle | Regret | Rationale |")
        lines.append("|---|---|---|---|---:|---|")
        for case in item["cases"]:
            lines.append(
                f"| {case['dataset']} | {case['scenario']} | "
                f"{case['planner']} @ K={case['k_samples']} | "
                f"{case['oracle_planner']} @ K={case['oracle_k_samples']} | "
                f"{case['regret']:.3f} | {case['rationale']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    csv_paths = discover_csvs(args.csv)
    if not csv_paths:
        print("No benchmark CSVs found. Run the Diff-MPPI benchmarks first or pass --csv.", file=sys.stderr)
        return 1

    rows = load_rows(csv_paths)
    requests = build_requests(rows)
    variants = build_variants()

    variant_results: list[dict[str, object]] = []
    for variant in variants:
        metrics, recommendations, cases = evaluate_variant(variant, rows, requests, args.benchmark_iterations)
        variant_results.append(
            {
                "name": variant.name,
                "paradigm": variant.paradigm,
                "metrics": metrics,
                "recommendations": recommendations,
                "cases": cases,
            }
        )

    docs_dir = ROOT / args.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    experiments_md = docs_dir / "experiments.md"
    experiments_md.write_text(generate_experiments_markdown(csv_paths, requests, variant_results))
    print(f"Generated {experiments_md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
