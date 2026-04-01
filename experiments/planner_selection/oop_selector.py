from dataclasses import dataclass
from functools import cmp_to_key
from typing import Sequence

from core.planner_selector_interface import AggregateBenchmarkRow, PlannerSelector, Recommendation, SelectionRequest
from experiments.support import rows_for_dataset_scenario


@dataclass(frozen=True)
class Objective:
    field: str
    maximize: bool

    def compare(self, left: AggregateBenchmarkRow, right: AggregateBenchmarkRow) -> int:
        left_value = getattr(left, self.field)
        right_value = getattr(right, self.field)
        if abs(left_value - right_value) < 1.0e-9:
            return 0
        if self.maximize:
            return -1 if left_value > right_value else 1
        return -1 if left_value < right_value else 1


class LexicographicPolicy:
    def __init__(self, objectives: Sequence[Objective]):
        self.objectives = list(objectives)

    def rank(self, rows: Sequence[AggregateBenchmarkRow]) -> list[AggregateBenchmarkRow]:
        def compare(left: AggregateBenchmarkRow, right: AggregateBenchmarkRow) -> int:
            for objective in self.objectives:
                result = objective.compare(left, right)
                if result != 0:
                    return result
            if left.k_samples != right.k_samples:
                return -1 if left.k_samples < right.k_samples else 1
            return -1 if left.planner < right.planner else (1 if left.planner > right.planner else 0)

        return sorted(rows, key=cmp_to_key(compare))


class OOPSelector(PlannerSelector):
    name = "oop_lexicographic"
    paradigm = "oop"

    def __init__(self):
        self.policy = LexicographicPolicy(
            [
                Objective("success", maximize=True),
                Objective("final_distance", maximize=False),
                Objective("cumulative_cost", maximize=False),
                Objective("avg_control_ms", maximize=False),
                Objective("steps", maximize=False),
            ]
        )

    def recommend(
        self,
        rows: Sequence[AggregateBenchmarkRow],
        request: SelectionRequest,
    ) -> Recommendation:
        candidates = rows_for_dataset_scenario(rows, request.dataset, request.scenario)
        if not candidates:
            raise ValueError(f"No candidates for {request.dataset}/{request.scenario}")

        ranked = self.policy.rank(candidates)
        best = ranked[0]
        rationale = "lexicographic ordering: success > final_distance > cumulative_cost > avg_control_ms > steps"
        score = (
            100.0 * best.success
            - best.final_distance
            - 1.0e-4 * best.cumulative_cost
            - best.avg_control_ms
            - 1.0e-3 * best.steps
        )
        return Recommendation(
            variant=self.name,
            dataset=request.dataset,
            scenario=request.scenario,
            planner=best.planner,
            k_samples=best.k_samples,
            score=score,
            rationale=rationale,
        )
