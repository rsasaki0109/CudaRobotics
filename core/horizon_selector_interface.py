from dataclasses import dataclass
from typing import Protocol, Sequence

from core.planner_selector_interface import AggregateBenchmarkRow


@dataclass(frozen=True)
class HorizonSelectionRequest:
    dataset: str
    scenario: str
    success_threshold: float = 0.999
    prefer_minimal: bool = True
    fallback_metric: str = "final_distance"


@dataclass(frozen=True)
class HorizonRecommendation:
    variant: str
    dataset: str
    scenario: str
    planner: str
    grad_update_horizon: int
    success_rate: float
    final_distance: float
    score: float
    rationale: str


class HorizonSelector(Protocol):
    name: str
    paradigm: str

    def recommend(
        self,
        rows: Sequence[AggregateBenchmarkRow],
        request: HorizonSelectionRequest,
    ) -> HorizonRecommendation:
        ...
