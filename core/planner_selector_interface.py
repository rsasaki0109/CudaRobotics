from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class AggregateBenchmarkRow:
    dataset: str
    scenario: str
    planner: str
    k_samples: int
    success: float
    steps: float
    final_distance: float
    cumulative_cost: float
    avg_control_ms: float


@dataclass(frozen=True)
class SelectionRequest:
    dataset: str
    scenario: str


@dataclass(frozen=True)
class Recommendation:
    variant: str
    dataset: str
    scenario: str
    planner: str
    k_samples: int
    score: float
    rationale: str


class PlannerSelector(Protocol):
    name: str
    paradigm: str

    def recommend(
        self,
        rows: Sequence[AggregateBenchmarkRow],
        request: SelectionRequest,
    ) -> Recommendation:
        ...
