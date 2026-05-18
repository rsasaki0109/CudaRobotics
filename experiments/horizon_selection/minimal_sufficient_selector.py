"""Functional minimal-sufficient horizon selector.

Strategy:
  1. Keep only candidates whose planner name encodes a gradient-update
     horizon (so ``mppi`` and feedback baselines are filtered out).
  2. Among those whose ``success_rate >= success_threshold``, return the
     candidate with the smallest horizon (ties broken by final_distance).
  3. If no candidate meets the threshold, fall back to ranking by the
     requested ``fallback_metric`` (default ``final_distance``) and pick
     the best one.

The point of step 2 is to encode the "early8 sweet spot" finding as a
deployable policy: do not pay for a longer horizon than you need.
"""

from __future__ import annotations

from typing import Sequence

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


class MinimalSufficientHorizonSelector(HorizonSelector):
    name = "minimal_sufficient"
    paradigm = "functional"

    def __init__(self, full_horizon_steps: int = 30) -> None:
        # ``full_horizon_steps`` is the C++-side DEFAULT_T_HORIZON used to
        # rank ``diff_mppi_3`` against the ``early*`` variants.
        self.full_horizon_steps = full_horizon_steps

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

        def horizon_of(row: AggregateBenchmarkRow) -> int:
            return _effective_horizon(row.planner, self.full_horizon_steps)

        sufficient = [
            row for row in candidates
            if row.success >= request.success_threshold
        ]

        if sufficient and request.prefer_minimal:
            best = min(
                sufficient,
                key=lambda r: (horizon_of(r), r.final_distance, r.planner),
            )
            rationale = (
                f"smallest horizon with success >= "
                f"{request.success_threshold:.3f} "
                f"({len(sufficient)} of {len(candidates)} candidates met)"
            )
            score = best.success
        else:
            if request.fallback_metric == "final_distance":
                best = min(
                    candidates,
                    key=lambda r: (r.final_distance, horizon_of(r), r.planner),
                )
                score = -best.final_distance
                rationale = (
                    "no candidate met success threshold; ranked by "
                    "final_distance"
                )
            elif request.fallback_metric == "cumulative_cost":
                best = min(
                    candidates,
                    key=lambda r: (r.cumulative_cost, horizon_of(r), r.planner),
                )
                score = -best.cumulative_cost
                rationale = (
                    "no candidate met success threshold; ranked by "
                    "cumulative_cost"
                )
            else:
                raise ValueError(
                    f"Unsupported fallback_metric: {request.fallback_metric}"
                )

        recommended_horizon = horizon_of(best)
        return HorizonRecommendation(
            variant=self.name,
            dataset=request.dataset,
            scenario=request.scenario,
            planner=best.planner,
            grad_update_horizon=recommended_horizon,
            success_rate=best.success,
            final_distance=best.final_distance,
            score=score,
            rationale=rationale,
        )
