"""OOP-style horizon selector that prefers the LARGEST horizon meeting
the success threshold.

Strategy is the opposite of MinimalSufficient: when in doubt, pay for a
longer horizon. The motivation is the off-grid generalisation case in
`scripts/online_horizon_generalization_test.py` -- when the probe sits
between known cells, the cell with the smallest sufficient horizon may
have been an easy outlier, and a probe slightly tougher than that cell
breaks immediately. Defaulting to the max-horizon row meeting the
threshold hedges against that.

The class composes two small policy objects (ThresholdGate, HorizonRanker)
so the OOP paradigm is about object composition rather than top-level
functions -- the same factoring the planner_selection OOPSelector uses.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ThresholdGate:
    success_threshold: float

    def passes(self, row: AggregateBenchmarkRow) -> bool:
        return row.success >= self.success_threshold


class HorizonRanker:
    def __init__(self, full_horizon_steps: int) -> None:
        self.full_horizon_steps = full_horizon_steps

    def descending(
        self,
        rows: Sequence[AggregateBenchmarkRow],
    ) -> list[AggregateBenchmarkRow]:
        return sorted(
            rows,
            key=lambda r: (
                -_effective_horizon(r.planner, self.full_horizon_steps),
                r.final_distance,
                r.planner,
            ),
        )


class OOPHorizonSelector(HorizonSelector):
    name = "robust_max"
    paradigm = "oop"

    def __init__(self, full_horizon_steps: int = 30) -> None:
        self.full_horizon_steps = full_horizon_steps
        self.ranker = HorizonRanker(full_horizon_steps)

    def recommend(
        self,
        rows: Sequence[AggregateBenchmarkRow],
        request: HorizonSelectionRequest,
    ) -> HorizonRecommendation:
        gate = ThresholdGate(request.success_threshold)
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

        sufficient = [row for row in candidates if gate.passes(row)]
        if sufficient:
            ranked = self.ranker.descending(sufficient)
            best = ranked[0]
            score = best.success
            rationale = (
                f"largest horizon with success >= "
                f"{request.success_threshold:.3f} "
                f"({len(sufficient)} of {len(candidates)} met)"
            )
        else:
            ranked = sorted(
                candidates,
                key=lambda r: (r.final_distance, r.planner),
            )
            best = ranked[0]
            score = -best.final_distance
            rationale = (
                "no candidate met success threshold; fell back to "
                "smallest final_distance"
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
