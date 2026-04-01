from dataclasses import dataclass
from functools import cmp_to_key
from typing import Sequence

from core.planner_selector_interface import AggregateBenchmarkRow
from core.time_budget_selector_interface import (
    TimeBudgetRecommendation,
    TimeBudgetRequest,
    TimeBudgetSelector,
)


@dataclass(frozen=True)
class BudgetRowView:
    row: AggregateBenchmarkRow
    budget_slack_ms: float


@dataclass(frozen=True)
class Objective:
    field: str
    maximize: bool

    def compare(self, left: BudgetRowView, right: BudgetRowView) -> int:
        if self.field == "budget_slack_ms":
            left_value = left.budget_slack_ms
            right_value = right.budget_slack_ms
        else:
            left_value = getattr(left.row, self.field)
            right_value = getattr(right.row, self.field)

        if abs(left_value - right_value) < 1.0e-9:
            return 0
        if self.maximize:
            return -1 if left_value > right_value else 1
        return -1 if left_value < right_value else 1


class LexicographicBudgetPolicy:
    def __init__(self, objectives: Sequence[Objective]):
        self.objectives = list(objectives)

    def rank(self, views: Sequence[BudgetRowView]) -> list[BudgetRowView]:
        def compare(left: BudgetRowView, right: BudgetRowView) -> int:
            for objective in self.objectives:
                result = objective.compare(left, right)
                if result != 0:
                    return result
            if left.row.k_samples != right.row.k_samples:
                return -1 if left.row.k_samples < right.row.k_samples else 1
            return -1 if left.row.planner < right.row.planner else (1 if left.row.planner > right.row.planner else 0)

        return sorted(views, key=cmp_to_key(compare))


class OOPBudgetSelector(TimeBudgetSelector):
    name = "oop_budget_lexicographic"
    paradigm = "oop"

    def __init__(self):
        self.policy = LexicographicBudgetPolicy(
            [
                Objective("success", maximize=True),
                Objective("final_distance", maximize=False),
                Objective("cumulative_cost", maximize=False),
                Objective("budget_slack_ms", maximize=False),
                Objective("steps", maximize=False),
            ]
        )

    def recommend(
        self,
        rows: Sequence[AggregateBenchmarkRow],
        request: TimeBudgetRequest,
    ) -> TimeBudgetRecommendation:
        candidates = [
            row for row in rows
            if row.dataset == request.dataset and row.scenario == request.scenario
        ]
        if not candidates:
            raise ValueError(f"No candidates for {request.dataset}/{request.scenario}")

        feasible = [row for row in candidates if row.avg_control_ms <= request.time_budget_ms + 1.0e-9]
        if not feasible:
            feasible = [min(candidates, key=lambda row: (row.avg_control_ms, row.final_distance, row.k_samples, row.planner))]

        views = [
            BudgetRowView(row=row, budget_slack_ms=max(0.0, request.time_budget_ms - row.avg_control_ms))
            for row in feasible
        ]
        ranked = self.policy.rank(views)
        best = ranked[0]
        score = (
            100.0 * best.row.success
            - best.row.final_distance
            - 1.0e-4 * best.row.cumulative_cost
            - 0.25 * best.budget_slack_ms
            - 1.0e-3 * best.row.steps
        )
        return TimeBudgetRecommendation(
            variant=self.name,
            dataset=request.dataset,
            scenario=request.scenario,
            time_budget_ms=request.time_budget_ms,
            planner=best.row.planner,
            k_samples=best.row.k_samples,
            score=score,
            rationale="lexicographic ordering: success > final_distance > cumulative_cost > budget_slack > steps",
        )
