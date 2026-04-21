from dataclasses import dataclass
from typing import Sequence

from core.planner_selector_interface import AggregateBenchmarkRow
from core.time_budget_selector_interface import (
    TimeBudgetRecommendation,
    TimeBudgetRequest,
    TimeBudgetSelector,
)
from experiments.support import (
    best_scored_row,
    fastest_row,
    feasible_rows,
    normalized_row_values,
    rows_for_dataset_scenario,
)


@dataclass(frozen=True)
class FunctionalBudgetWeights:
    success: float = 5.0
    final_distance: float = 2.0
    cumulative_cost: float = 1.0
    steps: float = 0.50
    headroom: float = 0.75

class FunctionalBudgetSelector(TimeBudgetSelector):
    name = "functional_budgeted"
    paradigm = "functional"

    def __init__(self, weights: FunctionalBudgetWeights | None = None):
        self.weights = weights or FunctionalBudgetWeights()

    def recommend(
        self,
        rows: Sequence[AggregateBenchmarkRow],
        request: TimeBudgetRequest,
    ) -> TimeBudgetRecommendation:
        candidates = rows_for_dataset_scenario(rows, request.dataset, request.scenario)
        if not candidates:
            raise ValueError(f"No candidates for {request.dataset}/{request.scenario}")

        feasible = feasible_rows(candidates, request.time_budget_ms)
        if not feasible:
            fallback = fastest_row(candidates)
            return TimeBudgetRecommendation(
                variant=self.name,
                dataset=request.dataset,
                scenario=request.scenario,
                time_budget_ms=request.time_budget_ms,
                planner=fallback.planner,
                k_samples=fallback.k_samples,
                score=-1000.0,
                rationale="fallback to the fastest candidate because no row fits the requested budget",
            )

        distance_norm = normalized_row_values(feasible, lambda row: row.final_distance)
        cost_norm = normalized_row_values(feasible, lambda row: row.cumulative_cost)
        steps_norm = normalized_row_values(feasible, lambda row: row.steps)
        headroom_norm = normalized_row_values(feasible, lambda row: request.time_budget_ms - row.avg_control_ms)

        scores = [
            (
                self.weights.success * row.success
                - self.weights.final_distance * distance_norm[index]
                - self.weights.cumulative_cost * cost_norm[index]
                - self.weights.steps * steps_norm[index]
                + self.weights.headroom * headroom_norm[index]
            )
            for index, row in enumerate(feasible)
        ]
        best_row, best_score = best_scored_row(feasible, scores)

        return TimeBudgetRecommendation(
            variant=self.name,
            dataset=request.dataset,
            scenario=request.scenario,
            time_budget_ms=request.time_budget_ms,
            planner=best_row.planner,
            k_samples=best_row.k_samples,
            score=best_score,
            rationale="weighted utility inside the time budget with a mild preference for spare headroom",
        )
