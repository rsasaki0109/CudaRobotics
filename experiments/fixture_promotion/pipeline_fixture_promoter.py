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


class PipelineFixturePromoter(FixturePromotionSelector):
    name = "pipeline_fixture_staged"
    paradigm = "pipeline"

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
        if not candidates:
            raise ValueError("No fixture portfolios available")

        stage = self._keep_top(candidates, key=lambda metrics: (metrics.required_ratio, 1.0 if metrics.required_hit else 0.0))
        stage = self._keep_top(stage, key=lambda metrics: metrics.scenario_ratio + metrics.tag_ratio)
        stage = self._keep_top(stage, key=lambda metrics: metrics.planner_ratio + 0.5 * metrics.depth_ratio)
        stage = self._keep_top(stage, key=lambda metrics: -len(metrics.datasets))
        best = sorted(stage, key=lambda metrics: metrics.datasets)[0]
        return FixturePromotionRecommendation(
            variant=self.name,
            request_id=request.request_id,
            selected_datasets=best.datasets,
            score=best.score,
            rationale="staged filter over requirement coverage, scenario-plus-tag density, planner diversity, and subset compactness",
        )

    @staticmethod
    def _keep_top(
        metrics_list: Sequence[PortfolioMetrics],
        key,
    ) -> list[PortfolioMetrics]:
        best_value = None
        kept: list[PortfolioMetrics] = []
        for metrics in metrics_list:
            value = key(metrics)
            if best_value is None or value > best_value:
                best_value = value
                kept = [metrics]
            elif value == best_value:
                kept.append(metrics)
        return kept
