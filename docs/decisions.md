# Decisions

This file records development-process decisions, not only code choices.

## D-001: Core Must Stay Smaller Than Experiments

Status: accepted

Decision:
- keep only the minimal planner-selection contract in `core/`
- keep all concrete selector implementations in `experiments/`

Why:
- the shared contract is already real
- the selection logic is still being explored
- freezing the selection logic now would turn an experiment into architecture too early

## D-002: Three Implementations Are the Starting Point, Not the End State

Status: accepted

Decision:
- maintain at least three live implementations for the current concrete problem
- compare them under one harness before extracting any new abstraction

Current variants:
- `functional_weighted`
- `oop_lexicographic`
- `pipeline_staged`

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
- do not move any of the three selector implementations into `core/`
- keep all three as experimental variants

Current evidence from `docs/experiments.md`:
- `functional_weighted` is the best benchmark fit on the current CSVs
- `oop_lexicographic` has the strongest extensibility proxy
- `pipeline_staged` is the fastest and reasonably readable

Why:
- the variants optimize different axes
- the repo should keep the tradeoff surface visible instead of hiding it behind one "correct" implementation

Operational note:
- if a single answer is needed for automation right now, use `functional_weighted`
- that is a temporary operating choice, not a core architecture decision

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
