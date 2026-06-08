from dataclasses import dataclass
from typing import Sequence

from core.planner_selector_interface import AggregateBenchmarkRow, PlannerSelector, Recommendation, SelectionRequest
from experiments.planner_selection.common import (
    candidates_for_request,
    normalized_metrics,
    recommendation_from_row,
)
from experiments.support import best_scored_row


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
        candidates = candidates_for_request(rows, request)
        metrics = normalized_metrics(candidates)

        scores = [
            (
                self.weights.success * row.success
                - self.weights.final_distance * metrics.final_distance[index]
                - self.weights.cumulative_cost * metrics.cumulative_cost[index]
                - self.weights.avg_control_ms * metrics.avg_control_ms[index]
                - self.weights.steps * metrics.steps[index]
            )
            for index, row in enumerate(candidates)
        ]
        best_row, best_score = best_scored_row(candidates, scores)

        rationale = (
            "weighted utility over success/final_distance/cumulative_cost/"
            "avg_control_ms/steps"
        )
        return recommendation_from_row(self.name, request, best_row, best_score, rationale)
