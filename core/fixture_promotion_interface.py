from dataclasses import dataclass
from typing import Protocol, Sequence

from core.planner_selector_interface import AggregateBenchmarkRow


@dataclass(frozen=True)
class FixturePromotionRequest:
    request_id: str
    max_fixtures: int
    required_tags: tuple[str, ...]
    scenario_weight: float
    planner_weight: float
    tag_weight: float
    depth_weight: float
    compactness_weight: float


@dataclass(frozen=True)
class FixturePromotionRecommendation:
    variant: str
    request_id: str
    selected_datasets: tuple[str, ...]
    score: float
    rationale: str


class FixturePromotionSelector(Protocol):
    name: str
    paradigm: str

    def recommend(
        self,
        rows: Sequence[AggregateBenchmarkRow],
        request: FixturePromotionRequest,
    ) -> FixturePromotionRecommendation:
        ...
