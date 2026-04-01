# Decisions

This file records development-process decisions, not only code choices.

## D-001: Core Must Stay Smaller Than Experiments

Status: accepted

Decision:
- keep only the minimal shared contracts in `core/`
- keep all concrete selector implementations in `experiments/`

Why:
- the shared contracts are already real
- the selection logic is still being explored
- freezing the selectors now would turn experiments into architecture too early

## D-002: Three Implementations Are the Starting Point, Not the End State

Status: accepted

Decision:
- maintain at least three live implementations for each concrete problem
- compare them under one harness before extracting any new abstraction

Current variants:
- planner selection:
  - `functional_weighted`
  - `oop_lexicographic`
  - `pipeline_staged`
- time-budget selection:
  - `functional_budgeted`
  - `oop_budget_lexicographic`
  - `pipeline_budget_staged`

Why:
- it forces design diversity
- it prevents a single early implementation from becoming "the architecture" by inertia

## D-003: Externalize Process State Into Docs

Status: accepted

Decision:
- `docs/experiments.md` is generated from the current benchmark outputs
- `docs/interfaces.md` defines the minimum stable contract
- `docs/decisions.md` records what moved into `core/` and what intentionally did not

Why:
- experiment state should not live only in code comments or chat history
- design decisions should be inspectable without reverse-engineering the code

## D-004: No Variant Is Promoted To Core Yet

Status: accepted

Decision:
- do not move any selector implementations into `core/`
- keep all selector families as experimental variants

Current evidence from `docs/experiments.md`:
- `functional_weighted` is the best benchmark fit on the current CSVs
- `oop_lexicographic` has the strongest extensibility proxy
- `pipeline_staged` is the fastest and reasonably readable
- `functional_budgeted` is currently the best constrained-fit baseline on the shared budgeted requests
- `oop_budget_lexicographic` has the strongest extensibility proxy for the second concrete problem
- `pipeline_budget_staged` is the runtime-oriented budgeted baseline, even though its regret is currently worse

Why:
- the variants optimize different axes
- the repo should keep the tradeoff surface visible instead of hiding it behind one "correct" implementation

Operational note:
- if a single answer is needed for automation right now, use `functional_weighted`
- if a single answer is needed for budget-constrained automation right now, use `functional_budgeted`
- these are temporary operating choices, not core architecture decisions

## D-005: Promotion Requires Repeated Survival

Status: accepted

Decision:
- a shared helper or abstraction can move into `core/` only if it survives multiple experiment cycles
- survival means:
  - reused by at least two distinct variants
  - still useful after at least one new dataset or benchmark pass
  - removable from the variants without increasing interface size

Why:
- this keeps abstraction discovery empirical
- it avoids promoting convenience code into architecture

## D-006: New Work Should Start As A Search Space

Status: accepted

Default workflow for new package-level work:
1. define one concrete problem
2. create at least three implementations with different design biases
3. force them through one interface
4. benchmark them
5. generate or update the three docs
6. only then consider extracting common structure

## D-007: Design Experiments Use Version-Controlled Fixtures

Status: accepted

Decision:
- keep lightweight fixture CSVs in `experiments/data/`
- run `scripts/run_design_experiments.py` and `scripts/validate_design_workflow.py` against those fixtures by default

Why:
- the design-process checks should not depend on rerunning the full benchmark suite
- CI needs deterministic, fast inputs
- heavy research benchmarks and lightweight design comparisons serve different purposes

Implication:
- fixture data can lag the newest benchmark outputs briefly
- when the benchmark meaningfully changes, refresh the fixtures intentionally instead of coupling every code edit to every heavy benchmark rerun

## D-008: The Workflow Must Survive More Than One Concrete Problem

Status: accepted

Decision:
- keep at least two active concrete problems in the experiment-first workflow
- require `scripts/run_design_experiments.py` to surface each active problem in generated docs
- require `scripts/validate_design_workflow.py` to fail if a module exists under `experiments/` but is not represented in generated docs

Why:
- one problem can still be a one-off demo
- two live problems force the workflow itself to be reusable
- if a new problem is not visible in generated docs, the externalized process state is already stale
