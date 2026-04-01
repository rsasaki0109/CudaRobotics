from dataclasses import dataclass
from typing import Sequence

from core.fixture_promotion_interface import (
    FixturePromotionRecommendation,
    FixturePromotionRequest,
    FixturePromotionSelector,
)
from core.planner_selector_interface import AggregateBenchmarkRow
from experiments.fixture_promotion.common import best_portfolio, build_profiles


@dataclass(frozen=True)
class FunctionalPortfolioWeights:
    score_bias: float = 1.0


class FunctionalFixturePromoter(FixturePromotionSelector):
    name = "functional_fixture_weighted"
    paradigm = "functional"

    def __init__(self, weights: FunctionalPortfolioWeights | None = None):
        self.weights = weights or FunctionalPortfolioWeights()

    def recommend(
        self,
        rows: Sequence[AggregateBenchmarkRow],
        request: FixturePromotionRequest,
    ) -> FixturePromotionRecommendation:
        profiles = build_profiles(rows)
        best = best_portfolio(profiles, request)
        rationale = (
            "enumerated all feasible fixture subsets and scored them with a weighted "
            "coverage/diversity/depth utility"
        )
        return FixturePromotionRecommendation(
            variant=self.name,
            request_id=request.request_id,
            selected_datasets=best.datasets,
            score=self.weights.score_bias * best.score,
            rationale=rationale,
        )
