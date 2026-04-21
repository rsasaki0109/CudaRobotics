from dataclasses import dataclass
from typing import Sequence

from core.planner_selector_interface import AggregateBenchmarkRow, PlannerSelector, Recommendation, SelectionRequest
from experiments.support import best_scored_row, normalized_row_values, rows_for_dataset_scenario


@dataclass(frozen=True)
class FunctionalWeights:
    success: float = 5.0
    final_distance: float = 2.0
    cumulative_cost: float = 1.0
    avg_control_ms: float = 0.75
    steps: float = 0.50

class FunctionalSelector(PlannerSelector):
    name = "functional_weighted"
    paradigm = "functional"

    def __init__(self, weights: FunctionalWeights | None = None):
        self.weights = weights or FunctionalWeights()

    def recommend(
        self,
        rows: Sequence[AggregateBenchmarkRow],
        request: SelectionRequest,
    ) -> Recommendation:
        candidates = rows_for_dataset_scenario(rows, request.dataset, request.scenario)
        if not candidates:
            raise ValueError(f"No candidates for {request.dataset}/{request.scenario}")

        final_norm = normalized_row_values(candidates, lambda row: row.final_distance)
        cost_norm = normalized_row_values(candidates, lambda row: row.cumulative_cost)
        time_norm = normalized_row_values(candidates, lambda row: row.avg_control_ms)
        step_norm = normalized_row_values(candidates, lambda row: row.steps)

        scores = [
            (
                self.weights.success * row.success
                - self.weights.final_distance * final_norm[index]
                - self.weights.cumulative_cost * cost_norm[index]
                - self.weights.avg_control_ms * time_norm[index]
                - self.weights.steps * step_norm[index]
            )
            for index, row in enumerate(candidates)
        ]
        best_row, best_score = best_scored_row(candidates, scores)

        rationale = (
            "weighted utility over success/final_distance/cumulative_cost/"
            "avg_control_ms/steps"
        )
        return Recommendation(
            variant=self.name,
            dataset=request.dataset,
            scenario=request.scenario,
            planner=best_row.planner,
            k_samples=best_row.k_samples,
            score=best_score,
            rationale=rationale,
        )
