# Interfaces

This repository treats design exploration as a first-class workflow.
The stable part is intentionally small and only expands after repeated reuse.

## Stable Core

Current stable interfaces:
- [`core/planner_selector_interface.py`](../core/planner_selector_interface.py)
- [`core/time_budget_selector_interface.py`](../core/time_budget_selector_interface.py)

Shared benchmark row:
- `AggregateBenchmarkRow`

Planner-selection contract:
- `SelectionRequest`
- `Recommendation`
- `PlannerSelector`

Time-budget-selection contract:
- `TimeBudgetRequest`
- `TimeBudgetRecommendation`
- `TimeBudgetSelector`

The contracts are intentionally narrow:
- every implementation reads the same aggregated benchmark rows
- every implementation answers the same request type for its problem
- every implementation returns the same recommendation payload for its problem

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
```

Only these contracts are stable today.
Scoring rules, ranking rules, budget heuristics, and request-generation logic are explicitly not stable.

## Experimental Variants

Current experimental problems:
- [`experiments/planner_selection`](../experiments/planner_selection)
- [`experiments/time_budget_selection`](../experiments/time_budget_selection)

Each experiment module is now self-describing:
- `PROBLEM_KIND`: slug-like metadata describing the problem family
- `INTERFACE_FILE`: points back to the minimum matching contract in `core/`
- `TITLE`, `DESCRIPTION_LINES`, `REQUEST_SUMMARY`, `METRIC_NOTES`: drive generated docs
- `build_requests(rows)`: owns request generation for that problem
- `build_report(rows, iterations)`: owns problem-specific evaluation and report assembly
- `build_variants()`: keeps the live concrete implementations discoverable

Planner-selection variants:
- [`experiments/planner_selection/functional_selector.py`](../experiments/planner_selection/functional_selector.py)
- [`experiments/planner_selection/oop_selector.py`](../experiments/planner_selection/oop_selector.py)
- [`experiments/planner_selection/pipeline_selector.py`](../experiments/planner_selection/pipeline_selector.py)

Time-budget-selection variants:
- [`experiments/time_budget_selection/functional_budget_selector.py`](../experiments/time_budget_selection/functional_budget_selector.py)
- [`experiments/time_budget_selection/oop_budget_selector.py`](../experiments/time_budget_selection/oop_budget_selector.py)
- [`experiments/time_budget_selection/pipeline_budget_selector.py`](../experiments/time_budget_selection/pipeline_budget_selector.py)

The variants are intentionally heterogeneous:
- planner selection keeps weighted utility, lexicographic ranking, and staged filtering alive
- time-budget selection keeps weighted feasible utility, lexicographic slack-aware ranking, and staged budget filtering alive

They share interfaces, not design style.

## Evaluation Harness

Comparison entrypoint:
- [`scripts/run_design_experiments.py`](../scripts/run_design_experiments.py)
- [`scripts/design_doctor.py`](../scripts/design_doctor.py)
- [`scripts/refresh_design_fixtures.py`](../scripts/refresh_design_fixtures.py)
- [`scripts/refresh_design_docs.py`](../scripts/refresh_design_docs.py)
- [`scripts/snapshot_design_experiments.py`](../scripts/snapshot_design_experiments.py)
- [`scripts/compare_design_snapshots.py`](../scripts/compare_design_snapshots.py)
- [`scripts/check_design_regressions.py`](../scripts/check_design_regressions.py)
- [`scripts/check_scaffold_design_problem.py`](../scripts/check_scaffold_design_problem.py)

Common evaluation rules:
- same CSV inputs
- same aggregation path from episode rows to comparable summary rows
- same request set per problem
- same oracle-based regret metric family
- same readability and extensibility proxy metrics

Discovery rule:
- the runner discovers `experiments/*/__init__.py` packages automatically
- the runner discovers fixture CSVs automatically from `experiments/data/*.csv`
- fixture membership is declared in [`experiments/data/manifest.json`](../experiments/data/manifest.json)
- `scripts/design_doctor.py` is the promoted local entrypoint for refresh-and-validate cycles
- fixture drift can be checked locally against build outputs via `scripts/refresh_design_fixtures.py --check-sync`
- the validator checks that every discovered module is represented in generated docs
- the validator also checks that the checked-in `docs/experiments.md` matches the generated output, while normalizing the volatile runtime column
- design-history snapshots are validated under `experiments/history/*.json`
- regression policy lives in [`experiments/history/policy.json`](../experiments/history/policy.json)
- `scripts/check_design_regressions.py` compares the latest two snapshots against that policy
- `scripts/compare_design_snapshots.py` renders a targeted delta between two snapshots without touching checked-in docs
- the scaffold checker verifies that new problem stubs still match the current module contract

Current generated state:
- [`docs/experiments.md`](../docs/experiments.md)
- [`docs/experiments_history.md`](../docs/experiments_history.md)
- [`docs/decisions.md`](../docs/decisions.md)
- [`docs/interfaces.md`](../docs/interfaces.md)
- [`experiments/history`](../experiments/history)

## Core vs Experiments

Current separation:
- `core/`: only contracts that multiple variants already share
- `experiments/`: discardable implementations and variant-specific logic
- `docs/`: externally visible experiment state

Promotion rule:
- if a shared piece survives across multiple variants and multiple experiment cycles, it may move into `core/`
- until then, keep it in `experiments/` or the evaluation harness
