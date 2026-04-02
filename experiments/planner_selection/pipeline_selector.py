from dataclasses import dataclass
from typing import Sequence

from core.planner_selector_interface import AggregateBenchmarkRow, PlannerSelector, Recommendation, SelectionRequest


@dataclass(frozen=True)
class PipelineConfig:
    final_distance_slack: float = 0.10
    runtime_slack: float = 0.35
    steps_slack: float = 4.0


class PipelineSelector(PlannerSelector):
    name = "pipeline_staged"
    paradigm = "pipeline"

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def recommend(
        self,
        rows: Sequence[AggregateBenchmarkRow],
        request: SelectionRequest,
    ) -> Recommendation:
        candidates = [
            row for row in rows
            if row.dataset == request.dataset and row.scenario == request.scenario
        ]
        if not candidates:
            raise ValueError(f"No candidates for {request.dataset}/{request.scenario}")

        best_success = max(row.success for row in candidates)
        stage_success = [row for row in candidates if row.success >= best_success]

        best_distance = min(row.final_distance for row in stage_success)
        distance_limit = best_distance + self.config.final_distance_slack
        stage_distance = [row for row in stage_success if row.final_distance <= distance_limit]

        best_runtime = min(row.avg_control_ms for row in stage_distance)
        runtime_limit = best_runtime + self.config.runtime_slack
        stage_runtime = [row for row in stage_distance if row.avg_control_ms <= runtime_limit]

        best_steps = min(row.steps for row in stage_runtime)
        steps_limit = best_steps + self.config.steps_slack
        stage_steps = [row for row in stage_runtime if row.steps <= steps_limit]

        best = min(
            stage_steps,
            key=lambda row: (row.cumulative_cost, row.avg_control_ms, row.k_samples, row.planner),
        )
        rationale = (
            "staged filters: max success -> near-best final_distance -> "
            "near-fastest runtime -> near-best steps -> lowest cumulative_cost"
        )
        score = (
            100.0 * best.success
            - best.final_distance
            - 1.0e-4 * best.cumulative_cost
            - 0.5 * best.avg_control_ms
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
