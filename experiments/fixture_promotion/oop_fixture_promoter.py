from dataclasses import dataclass
from typing import Sequence

from core.fixture_promotion_interface import (
    FixturePromotionRecommendation,
    FixturePromotionRequest,
    FixturePromotionSelector,
)
from core.planner_selector_interface import AggregateBenchmarkRow
from experiments.fixture_promotion.common import (
    PortfolioMetrics,
    build_profiles,
    candidate_portfolios,
    score_portfolio,
)


@dataclass(frozen=True)
class Objective:
    name: str
    reverse: bool = True

    def value(self, metrics: PortfolioMetrics) -> float:
        if self.name == "required_hit":
            return 1.0 if metrics.required_hit else 0.0
        if self.name == "required_ratio":
            return metrics.required_ratio
        if self.name == "scenario_ratio":
            return metrics.scenario_ratio
        if self.name == "planner_ratio":
            return metrics.planner_ratio
        if self.name == "tag_ratio":
            return metrics.tag_ratio
        if self.name == "depth_ratio":
            return metrics.depth_ratio
        if self.name == "compactness":
            return -float(len(metrics.datasets))
        raise ValueError(f"Unknown objective {self.name}")


class OOPFixturePromoter(FixturePromotionSelector):
    name = "oop_fixture_lexicographic"
    paradigm = "oop"

    def __init__(self):
        self.objectives = [
            Objective("required_hit"),
            Objective("required_ratio"),
            Objective("scenario_ratio"),
            Objective("planner_ratio"),
            Objective("tag_ratio"),
            Objective("depth_ratio"),
            Objective("compactness"),
        ]

    def recommend(
        self,
        rows: Sequence[AggregateBenchmarkRow],
        request: FixturePromotionRequest,
    ) -> FixturePromotionRecommendation:
        profiles = build_profiles(rows)
        candidates = [
            score_portfolio(profiles, request, portfolio)
            for portfolio in candidate_portfolios(sorted(profiles), request.max_fixtures)
        ]

        def objective_key(metrics: PortfolioMetrics) -> tuple[float, ...]:
            return tuple(objective.value(metrics) for objective in self.objectives)

        best = max(candidates, key=lambda metrics: (objective_key(metrics), metrics.datasets))
        return FixturePromotionRecommendation(
            variant=self.name,
            request_id=request.request_id,
            selected_datasets=best.datasets,
            score=best.score,
            rationale="lexicographic policy over requirement hit, scenario coverage, planner diversity, tag coverage, and compactness",
        )
