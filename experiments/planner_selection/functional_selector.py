from dataclasses import dataclass
from typing import Sequence

from core.planner_selector_interface import AggregateBenchmarkRow, PlannerSelector, Recommendation, SelectionRequest


@dataclass(frozen=True)
class FunctionalWeights:
    success: float = 5.0
    final_distance: float = 2.0
    cumulative_cost: float = 1.0
    avg_control_ms: float = 0.75
    steps: float = 0.50


def _candidates(rows: Sequence[AggregateBenchmarkRow], request: SelectionRequest) -> list[AggregateBenchmarkRow]:
    return [row for row in rows if row.dataset == request.dataset and row.scenario == request.scenario]


def _normalize(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if high - low < 1.0e-9:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


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
        candidates = _candidates(rows, request)
        if not candidates:
            raise ValueError(f"No candidates for {request.dataset}/{request.scenario}")

        final_norm = _normalize([row.final_distance for row in candidates])
        cost_norm = _normalize([row.cumulative_cost for row in candidates])
        time_norm = _normalize([row.avg_control_ms for row in candidates])
        step_norm = _normalize([row.steps for row in candidates])

        best_row = None
        best_score = float("-inf")
        for index, row in enumerate(candidates):
            score = (
                self.weights.success * row.success
                - self.weights.final_distance * final_norm[index]
                - self.weights.cumulative_cost * cost_norm[index]
                - self.weights.avg_control_ms * time_norm[index]
                - self.weights.steps * step_norm[index]
            )
            if best_row is None or score > best_score:
                best_row = row
                best_score = score

        assert best_row is not None
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
