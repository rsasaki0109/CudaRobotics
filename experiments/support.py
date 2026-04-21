from __future__ import annotations

import ast
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from core.planner_selector_interface import AggregateBenchmarkRow


ROOT = Path(__file__).resolve().parents[1]


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
class MarkdownTable:
    headers: list[str]
    rows: list[list[str]]


@dataclass(frozen=True)
class TitledTable:
    title: str
    table: MarkdownTable


@dataclass(frozen=True)
class ProblemReport:
    slug: str
    title: str
    description_lines: list[str]
    request_summary: str
    metric_notes: list[str]
    request_count: int
    aggregate_table: MarkdownTable
    case_tables: list[TitledTable]


def markdown_table_to_dict(table: MarkdownTable) -> dict[str, object]:
    return {
        "headers": list(table.headers),
        "rows": [list(row) for row in table.rows],
    }


def titled_table_to_dict(titled_table: TitledTable) -> dict[str, object]:
    return {
        "title": titled_table.title,
        "table": markdown_table_to_dict(titled_table.table),
    }


def problem_report_to_dict(report: ProblemReport) -> dict[str, object]:
    return {
        "slug": report.slug,
        "title": report.title,
        "description_lines": list(report.description_lines),
        "request_summary": report.request_summary,
        "metric_notes": list(report.metric_notes),
        "request_count": report.request_count,
        "aggregate_table": markdown_table_to_dict(report.aggregate_table),
        "case_tables": [titled_table_to_dict(table) for table in report.case_tables],
    }


def normalize(values: Sequence[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if high - low < 1.0e-9:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def normalized_row_values(
    rows: Sequence[AggregateBenchmarkRow],
    value: Callable[[AggregateBenchmarkRow], float],
) -> list[float]:
    return normalize([value(row) for row in rows])


def best_scored_row(
    rows: Sequence[AggregateBenchmarkRow],
    scores: Sequence[float],
) -> tuple[AggregateBenchmarkRow, float]:
    if not rows:
        raise ValueError("Cannot select from an empty row set")
    if len(rows) != len(scores):
        raise ValueError("Rows and scores must have the same length")

    best_index = max(range(len(rows)), key=lambda index: scores[index])
    return rows[best_index], scores[best_index]


def rows_for_dataset_scenario(
    rows: Sequence[AggregateBenchmarkRow],
    dataset: str,
    scenario: str,
) -> list[AggregateBenchmarkRow]:
    return [row for row in rows if row.dataset == dataset and row.scenario == scenario]


def feasible_rows(
    rows: Sequence[AggregateBenchmarkRow],
    time_budget_ms: float,
) -> list[AggregateBenchmarkRow]:
    return [row for row in rows if row.avg_control_ms <= time_budget_ms + 1.0e-9]


def fastest_row(rows: Sequence[AggregateBenchmarkRow]) -> AggregateBenchmarkRow:
    return min(rows, key=lambda row: (row.avg_control_ms, row.final_distance, row.k_samples, row.planner))


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


def render_markdown_table(table: MarkdownTable) -> list[str]:
    lines: list[str] = []
    aligns = ["---" for _ in table.headers]
    lines.append("| " + " | ".join(table.headers) + " |")
    lines.append("|" + "|".join(aligns) + "|")
    for row in table.rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines
