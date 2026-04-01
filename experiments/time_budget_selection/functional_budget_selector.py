from dataclasses import dataclass
from typing import Sequence

from core.planner_selector_interface import AggregateBenchmarkRow
from core.time_budget_selector_interface import (
    TimeBudgetRecommendation,
    TimeBudgetRequest,
    TimeBudgetSelector,
)
from experiments.support import fastest_row, feasible_rows, normalize, rows_for_dataset_scenario


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

        distance_norm = normalize([row.final_distance for row in feasible])
        cost_norm = normalize([row.cumulative_cost for row in feasible])
        steps_norm = normalize([row.steps for row in feasible])
        headroom_norm = normalize([request.time_budget_ms - row.avg_control_ms for row in feasible])

        best_row = None
        best_score = float("-inf")
        for index, row in enumerate(feasible):
            score = (
                self.weights.success * row.success
                - self.weights.final_distance * distance_norm[index]
                - self.weights.cumulative_cost * cost_norm[index]
                - self.weights.steps * steps_norm[index]
                + self.weights.headroom * headroom_norm[index]
            )
            if best_row is None or score > best_score:
                best_row = row
                best_score = score

        assert best_row is not None
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
