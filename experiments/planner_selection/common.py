from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core.planner_selector_interface import (
    AggregateBenchmarkRow,
    Recommendation,
    SelectionRequest,
)
from experiments.support import normalized_row_values, rows_for_dataset_scenario


@dataclass(frozen=True)
class NormalizedPlannerMetrics:
    final_distance: list[float]
    cumulative_cost: list[float]
    avg_control_ms: list[float]
    steps: list[float]


def candidates_for_request(
    rows: Sequence[AggregateBenchmarkRow],
    request: SelectionRequest,
) -> list[AggregateBenchmarkRow]:
    candidates = rows_for_dataset_scenario(rows, request.dataset, request.scenario)
    if not candidates:
        raise ValueError(f"No candidates for {request.dataset}/{request.scenario}")
    return candidates


def normalized_metrics(
    rows: Sequence[AggregateBenchmarkRow],
) -> NormalizedPlannerMetrics:
    return NormalizedPlannerMetrics(
        final_distance=normalized_row_values(rows, lambda row: row.final_distance),
        cumulative_cost=normalized_row_values(rows, lambda row: row.cumulative_cost),
        avg_control_ms=normalized_row_values(rows, lambda row: row.avg_control_ms),
        steps=normalized_row_values(rows, lambda row: row.steps),
    )


def baseline_score(
    row: AggregateBenchmarkRow,
    avg_control_ms_weight: float = 1.0,
) -> float:
    return (
        100.0 * row.success
        - row.final_distance
        - 1.0e-4 * row.cumulative_cost
        - avg_control_ms_weight * row.avg_control_ms
        - 1.0e-3 * row.steps
    )


def recommendation_from_row(
    variant_name: str,
    request: SelectionRequest,
    row: AggregateBenchmarkRow,
    score: float,
    rationale: str,
) -> Recommendation:
    return Recommendation(
        variant=variant_name,
        dataset=request.dataset,
        scenario=request.scenario,
        planner=row.planner,
        k_samples=row.k_samples,
        score=score,
        rationale=rationale,
    )
