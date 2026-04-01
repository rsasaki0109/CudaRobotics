from dataclasses import dataclass
from typing import Sequence

from core.planner_selector_interface import AggregateBenchmarkRow
from core.time_budget_selector_interface import (
    TimeBudgetRecommendation,
    TimeBudgetRequest,
    TimeBudgetSelector,
)


@dataclass(frozen=True)
class BudgetPipelineConfig:
    runtime_slack_ms: float = 0.08
    success_slack: float = 0.01
    final_distance_slack: float = 0.20


class PipelineBudgetSelector(TimeBudgetSelector):
    name = "pipeline_budget_staged"
    paradigm = "pipeline"

    def __init__(self, config: BudgetPipelineConfig | None = None):
        self.config = config or BudgetPipelineConfig()

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

        fastest = min(row.avg_control_ms for row in feasible)
        runtime_limit = fastest + self.config.runtime_slack_ms
        stage_runtime = [row for row in feasible if row.avg_control_ms <= runtime_limit]

        best_success = max(row.success for row in stage_runtime)
        success_floor = best_success - self.config.success_slack
        stage_success = [row for row in stage_runtime if row.success >= success_floor]

        best_distance = min(row.final_distance for row in stage_success)
        distance_limit = best_distance + self.config.final_distance_slack
        stage_distance = [row for row in stage_success if row.final_distance <= distance_limit]

        best = min(
            stage_distance,
            key=lambda row: (row.cumulative_cost, row.steps, row.k_samples, row.planner),
        )
        score = (
            100.0 * best.success
            - best.final_distance
            - 1.0e-4 * best.cumulative_cost
            - 0.10 * best.avg_control_ms
            - 1.0e-3 * best.steps
        )
        return TimeBudgetRecommendation(
            variant=self.name,
            dataset=request.dataset,
            scenario=request.scenario,
            time_budget_ms=request.time_budget_ms,
            planner=best.planner,
            k_samples=best.k_samples,
            score=score,
            rationale="staged filters: near-fastest feasible -> near-best success -> near-best final_distance -> lowest cost",
        )
