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
- fixture promotion:
  - `functional_fixture_weighted`
  - `oop_fixture_lexicographic`
  - `pipeline_fixture_staged`
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
- `functional_fixture_weighted` is currently the best fixture-promotion baseline on the shared portfolio requests
- `oop_fixture_lexicographic` has the strongest extensibility proxy for fixture promotion
- `pipeline_fixture_staged` keeps a staged portfolio baseline alive even though its regret is higher
- `functional_budgeted` is currently the best constrained-fit baseline on the shared budgeted requests
- `oop_budget_lexicographic` has the strongest extensibility proxy for time-budget selection
- `pipeline_budget_staged` is the runtime-oriented budgeted baseline, even though its regret is currently worse

Why:
- the variants optimize different axes
- the repo should keep the tradeoff surface visible instead of hiding it behind one "correct" implementation

Operational note:
- if a single answer is needed for automation right now, use `functional_weighted`
- if a single answer is needed for fixture promotion right now, use `functional_fixture_weighted`
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
- fixture discovery should be automatic inside `experiments/data/` so adding a new fixture does not require editing the central runner

Operational note:
- fixture membership is now externalized in `experiments/data/manifest.json`
- `scripts/refresh_design_fixtures.py` is the promoted path for copying selected benchmark CSVs into `experiments/data/`
- `scripts/refresh_design_fixtures.py --check-sync` can be used locally to detect fixture drift against available build outputs

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

## D-009: Experiment Modules Must Describe Themselves

Status: accepted

Decision:
- each module under `experiments/` must expose its own `PROBLEM_KIND`, `INTERFACE_FILE`, and presentation metadata
- each module must own its own `build_requests(rows)` function
- each module must own its own `build_report(rows, iterations)` function
- `scripts/run_design_experiments.py` should discover experiment modules instead of importing a hard-coded list

Why:
- adding a new concrete problem should mostly change that problem's package, not the central runner
- request generation is part of the problem definition, so it belongs with the problem
- report assembly is also part of the problem definition, so it should not live in a central `if/else` ladder
- this keeps the workflow tidy without promoting problem-specific logic into `core/`

Operational note:
- `PROBLEM_KIND` is now validated as a slug-like metadata field, not against a central allow-list

## D-010: Generated Docs Must Match The Checked-In State

Status: accepted

Decision:
- `scripts/validate_design_workflow.py` must compare generated `experiments.md` against the checked-in `docs/experiments.md`
- refreshing the generated doc should have a dedicated command entrypoint

Why:
- externalized state is only useful if the committed doc matches the current code
- CI should fail on stale process docs, not just on missing docs

Operational note:
- the stale-doc comparison normalizes the `Runtime ms/request` column because that metric is environment-sensitive

## D-011: The Scaffolder Is Part Of The Contract

Status: accepted

Decision:
- `scripts/scaffold_design_problem.py` must support writing into an arbitrary root
- CI must run a scaffold self-check against a temporary root

Why:
- the scaffolder is how new search spaces enter the repo
- if the scaffold drifts from the validator contract, the workflow decays silently

## D-012: Experiment History Must Be Snapshotted, Not Reconstructed From Memory

Status: accepted

Decision:
- store design-history snapshots under `experiments/history/`
- generate `docs/experiments_history.md` from those snapshots
- validate snapshot schema and checked-in history docs in `scripts/validate_design_workflow.py`

Why:
- the repo should show how design decisions changed over time, not just the current winner
- snapshots make process evolution inspectable without relying on chat logs or local shells

## D-013: The Workflow Needs One Local Repair Command

Status: accepted

Decision:
- promote `scripts/design_doctor.py` as the local one-command entrypoint
- let it compose fixture refresh or fixture-sync checks, doc refresh, history refresh or snapshotting, workflow validation, and scaffold validation

Why:
- the workflow is only useful if it is cheap enough to run before commits
- the lower-level scripts should stay individually callable, but local maintenance should not require remembering the full sequence

## D-014: History Needs A Delta View, Not Just Snapshots

Status: accepted

Decision:
- `docs/experiments_history.md` must render the latest inter-snapshot delta
- `scripts/compare_design_snapshots.py` should expose the same comparison without rewriting the history doc

Why:
- a history log is only half-useful if readers still need to diff JSON by hand
- the repo should show what changed between experiment states, not just that states exist

## D-015: History Needs Explicit Regression Guardrails

Status: accepted

Decision:
- store regression policy in `experiments/history/policy.json`
- require `scripts/check_design_regressions.py` to compare the latest two snapshots against that policy
- keep runtime out of the default regression gate because it is too environment-sensitive

Why:
- experiment-first should allow exploration, but not silent quality regressions
- the repo needs an explicit notion of what must not get worse as designs evolve

## D-016: Convergence Must Be Read From History, Not Assumed

Status: accepted

Decision:
- generate `docs/convergence.md` from design-history snapshots
- treat repeated quality leadership as a soft signal, not an automatic promotion trigger
- keep runtime out of convergence calls because it is too volatile

Why:
- the repo should show where convergence is beginning without pretending that one snapshot proves it
- this keeps implementation promotion evidence-based while preserving exploration

## D-017: Convergence Must Lead To A Visible Next Move

Status: accepted

Decision:
- generate `docs/next_actions.md` from snapshot history and convergence signals
- express process moves as `diversify`, `hold`, `promotion_watch`, or `promote_shared_helpers`
- keep the output advisory; it should guide the next search step, not collapse the search space automatically

Why:
- observing convergence is not enough if the repo still leaves the next step implicit
- action labels make the process inspectable without turning it into a rigid promotion pipeline

## D-018: Action Heuristics Must Be Policy, Not Hidden Code

Status: accepted

Decision:
- store thresholds and canned action advice in `experiments/history/actions_policy.json`
- keep `scripts/render_design_actions.py` as an interpreter of that policy, not the owner of those numbers

Why:
- the next-step policy will evolve as the repo learns more about its own search process
- changing process thresholds should not require rewriting renderer logic

## D-019: New Problems Start Ungated Until They Have Two Snapshots

Status: accepted

Decision:
- allow `scripts/check_design_regressions.py` to mark newly added policy problems as `PENDING` when they are not present in both compared snapshots yet
- start enforcing regression limits only after the problem has at least two recorded snapshots

Why:
- adding a new concrete problem should not require fabricating historical snapshots
- this keeps the gate strict for mature problems without blocking new search spaces from entering the workflow

## D-020: Shared Helper Extraction Comes Before Variant Promotion

Status: accepted

Decision:
- when a `promotion_watch` problem shows stable quality leadership, first extract repeated helper logic into `experiments/support.py`
- keep policy logic and ranking rules inside the variants unless at least two live variants genuinely share them

Why:
- this is the smallest safe move from convergence evidence to code change
- it reduces duplication without collapsing the search space into one implementation

## D-021: Shared Helpers Need Their Own Watchlist

Status: accepted

Decision:
- generate `docs/helper_promotion.md` from current `experiments.support` usage plus snapshot counts
- classify helpers as `keep_in_experiments`, `promotion_watch`, or `core_candidate` via `experiments/history/helper_policy.json`

Why:
- helper extraction is now part of the improvement track, so it needs explicit visibility
- this keeps promotion pressure focused on genuinely reused helpers instead of on whole implementations
