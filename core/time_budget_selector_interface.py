from dataclasses import dataclass
from typing import Protocol, Sequence

from core.planner_selector_interface import AggregateBenchmarkRow


@dataclass(frozen=True)
class TimeBudgetRequest:
    dataset: str
    scenario: str
    time_budget_ms: float


@dataclass(frozen=True)
class TimeBudgetRecommendation:
    variant: str
    dataset: str
    scenario: str
    time_budget_ms: float
    planner: str
    k_samples: int
    score: float
    rationale: str


class TimeBudgetSelector(Protocol):
    name: str
    paradigm: str

    def recommend(
        self,
        rows: Sequence[AggregateBenchmarkRow],
        request: TimeBudgetRequest,
    ) -> TimeBudgetRecommendation:
        ...
