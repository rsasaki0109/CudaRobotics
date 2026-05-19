"""Pipeline-style horizon selector with explicit stages.

The stages mirror how a deployment-time policy would actually think:

  1. ``filter_threshold``: drop any horizon that fails the success bar.
  2. ``filter_runtime``: among survivors, drop rows that are more than
     ``runtime_slack`` ms slower than the fastest survivor. This is the
     "spend horizon, not latency" guard from the README.
  3. ``pick_smallest_horizon``: tie-break the survivors by horizon
     ascending, then by final_distance.

If stage 1 wipes out all candidates, the pipeline falls back to ranking
the original set by final_distance. The class stores the stages as an
explicit ordered list so paradigm-vs-functional differences are visible
in the source: data flows through named stages and each stage is its
own method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from core.horizon_selector_interface import (
    HorizonRecommendation,
    HorizonSelectionRequest,
    HorizonSelector,
)
from core.planner_selector_interface import AggregateBenchmarkRow
from experiments.horizon_selection.horizon_naming import (
    FULL_HORIZON_SENTINEL,
    parse_grad_update_horizon,
)


def _effective_horizon(planner: str, full_horizon_steps: int) -> int:
    raw = parse_grad_update_horizon(planner)
    if raw is None:
        return -1
    if raw == FULL_HORIZON_SENTINEL:
        return full_horizon_steps
    return raw


@dataclass(frozen=True)
class PipelineConfig:
    runtime_slack_ms: float = 0.30


StageFn = Callable[
    [list[AggregateBenchmarkRow], HorizonSelectionRequest],
    list[AggregateBenchmarkRow],
]


class PipelineHorizonSelector(HorizonSelector):
    name = "pipeline_staged"
    paradigm = "pipeline"

    def __init__(
        self,
        full_horizon_steps: int = 30,
        config: PipelineConfig | None = None,
    ) -> None:
        self.full_horizon_steps = full_horizon_steps
        self.config = config or PipelineConfig()
        self.stages: list[tuple[str, StageFn]] = [
            ("filter_threshold", self._stage_threshold),
            ("filter_runtime", self._stage_runtime),
            ("pick_smallest_horizon", self._stage_pick),
        ]

    def _stage_threshold(
        self,
        rows: list[AggregateBenchmarkRow],
        request: HorizonSelectionRequest,
    ) -> list[AggregateBenchmarkRow]:
        return [r for r in rows if r.success >= request.success_threshold]

    def _stage_runtime(
        self,
        rows: list[AggregateBenchmarkRow],
        _request: HorizonSelectionRequest,
    ) -> list[AggregateBenchmarkRow]:
        if not rows:
            return rows
        fastest = min(r.avg_control_ms for r in rows)
        return [
            r for r in rows
            if r.avg_control_ms <= fastest + self.config.runtime_slack_ms
        ]

    def _stage_pick(
        self,
        rows: list[AggregateBenchmarkRow],
        _request: HorizonSelectionRequest,
    ) -> list[AggregateBenchmarkRow]:
        if not rows:
            return rows
        return [min(
            rows,
            key=lambda r: (
                _effective_horizon(r.planner, self.full_horizon_steps),
                r.final_distance,
                r.planner,
            ),
        )]

    def recommend(
        self,
        rows: Sequence[AggregateBenchmarkRow],
        request: HorizonSelectionRequest,
    ) -> HorizonRecommendation:
        candidates = [
            row
            for row in rows
            if row.dataset == request.dataset
            and row.scenario == request.scenario
            and parse_grad_update_horizon(row.planner) is not None
        ]
        if not candidates:
            raise ValueError(
                f"No gradient-horizon candidates for "
                f"{request.dataset}/{request.scenario}"
            )

        state = list(candidates)
        stage_trace: list[str] = []
        for name, fn in self.stages:
            state = fn(state, request)
            stage_trace.append(f"{name}->{len(state)}")
            if not state:
                break

        if state:
            best = state[0]
            score = best.success
            rationale = "pipeline: " + ", ".join(stage_trace)
        else:
            best = min(
                candidates,
                key=lambda r: (r.final_distance, r.planner),
            )
            score = -best.final_distance
            rationale = (
                "pipeline drained all stages; fell back to smallest "
                "final_distance"
            )

        return HorizonRecommendation(
            variant=self.name,
            dataset=request.dataset,
            scenario=request.scenario,
            planner=best.planner,
            grad_update_horizon=_effective_horizon(
                best.planner, self.full_horizon_steps),
            success_rate=best.success,
            final_distance=best.final_distance,
            score=score,
            rationale=rationale,
        )
