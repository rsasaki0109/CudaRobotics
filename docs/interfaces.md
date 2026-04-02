# Interfaces

This repository now treats design exploration as a first-class workflow.
The stable part is intentionally small.

## Stable Core

Current stable interface:
- [`core/planner_selector_interface.py`](../core/planner_selector_interface.py)

Minimal objects:
- `AggregateBenchmarkRow`
- `SelectionRequest`
- `Recommendation`
- `PlannerSelector`

The contract is intentionally narrow:
- every implementation reads the same aggregated benchmark rows
- every implementation answers the same dataset/scenario request
- every implementation returns the same recommendation payload

Current interface shape:

```python
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
```

Only this contract is considered stable today.
Scoring rules, ranking rules, and selection heuristics are explicitly not stable.

## Experimental Variants

Current experimental variants:
- [`experiments/planner_selection/functional_selector.py`](../experiments/planner_selection/functional_selector.py)
- [`experiments/planner_selection/oop_selector.py`](../experiments/planner_selection/oop_selector.py)
- [`experiments/planner_selection/pipeline_selector.py`](../experiments/planner_selection/pipeline_selector.py)

The variants are intentionally heterogeneous:
- `functional_weighted`: weighted utility from normalized metrics
- `oop_lexicographic`: objective objects plus lexicographic ranking
- `pipeline_staged`: staged filtering pipeline

They share the interface, not the design style.

## Evaluation Harness

Comparison entrypoint:
- [`scripts/run_design_experiments.py`](../scripts/run_design_experiments.py)

Common evaluation rules:
- same CSV inputs
- same aggregation path from episode rows to comparable summary rows
- same request set
- same oracle-based regret metric
- same readability and extensibility proxy metrics

## Core vs Experiments

Current separation:
- `core/`: only contracts that multiple variants already share
- `experiments/`: discardable implementations and variant-specific logic
- `docs/`: externally visible experiment state

Promotion rule:
- if a shared piece survives across multiple variants and multiple experiment cycles, it may move into `core/`
- until then, keep it in `experiments/` or the evaluation harness
