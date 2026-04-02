from .functional_budget_selector import FunctionalBudgetSelector
from .oop_budget_selector import OOPBudgetSelector
from .pipeline_budget_selector import PipelineBudgetSelector
from core.planner_selector_interface import AggregateBenchmarkRow
from core.time_budget_selector_interface import TimeBudgetRequest


PROBLEM_KIND = "time_budget_selection"
INTERFACE_FILE = "time_budget_selector_interface.py"
TITLE = "Time-Budget Selection"
DESCRIPTION_LINES = [
    "choose one planner configuration per dataset/scenario/time-budget request",
    "force all variants to consume the same aggregated benchmark rows and the same wall-clock envelopes",
    "score each selector on constrained regret, budget-hit rate, runtime, readability, and extensibility proxies",
]
REQUEST_SUMMARY = "dataset/scenario/time-budget triples"
METRIC_NOTES = [
    "`Avg Regret`: utility gap from the best feasible row under the requested time budget; lower is better",
    "`Oracle Match`: fraction of requests where the selector matched the best feasible row exactly",
    "`Budget Hit`: fraction of requests where the selected row stayed inside the requested `avg_control_ms` envelope",
]


def _select_budget_levels(runtimes: list[float]) -> list[float]:
    unique = sorted(set(runtimes))
    if not unique:
        return []

    levels: list[float] = []
    for fraction in (0.0, 0.50, 0.80):
        index = round(fraction * (len(unique) - 1))
        candidate = unique[index]
        if not any(abs(candidate - value) < 1.0e-9 for value in levels):
            levels.append(candidate)

    for candidate in unique:
        if len(levels) >= min(3, len(unique)):
            break
        if not any(abs(candidate - value) < 1.0e-9 for value in levels):
            levels.append(candidate)

    return sorted(levels)


def build_requests(rows: list[AggregateBenchmarkRow]) -> list[TimeBudgetRequest]:
    grouped: dict[tuple[str, str], list[AggregateBenchmarkRow]] = {}
    for row in rows:
        grouped.setdefault((row.dataset, row.scenario), []).append(row)

    requests: list[TimeBudgetRequest] = []
    for dataset, scenario in sorted(grouped):
        budgets = _select_budget_levels([row.avg_control_ms for row in grouped[(dataset, scenario)]])
        for budget in budgets:
            requests.append(
                TimeBudgetRequest(
                    dataset=dataset,
                    scenario=scenario,
                    time_budget_ms=budget,
                )
            )
    return requests


def build_variants():
    return [
        FunctionalBudgetSelector(),
        OOPBudgetSelector(),
        PipelineBudgetSelector(),
    ]
