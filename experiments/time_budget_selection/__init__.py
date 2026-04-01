from .functional_budget_selector import FunctionalBudgetSelector
from .oop_budget_selector import OOPBudgetSelector
from .pipeline_budget_selector import PipelineBudgetSelector
from core.planner_selector_interface import AggregateBenchmarkRow
from core.time_budget_selector_interface import TimeBudgetRecommendation, TimeBudgetRequest
from experiments.support import (
    MarkdownTable,
    ProblemReport,
    TitledTable,
    VariantEvaluation,
    benchmark_variant,
    find_row,
    utility_map,
    variant_metric_block,
)


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


def evaluate_variant(variant, rows: list[AggregateBenchmarkRow], requests: list[TimeBudgetRequest], iterations: int) -> VariantEvaluation:
    cases: list[dict[str, object]] = []
    total_regret = 0.0
    match_count = 0
    budget_hit_count = 0

    for request in requests:
        candidates = [row for row in rows if row.dataset == request.dataset and row.scenario == request.scenario]
        feasible = [row for row in candidates if row.avg_control_ms <= request.time_budget_ms + 1.0e-9]
        if not feasible:
            raise RuntimeError(f"No feasible candidates for {request.dataset}/{request.scenario} at {request.time_budget_ms:.4f} ms")

        utilities = utility_map(feasible)
        best_key, best_utility = max(utilities.items(), key=lambda item: item[1])
        oracle_row = find_row(feasible, best_key[0], best_key[1])

        recommendation: TimeBudgetRecommendation = variant.recommend(rows, request)
        selected_key = (recommendation.planner, recommendation.k_samples)
        selected_row = find_row(candidates, recommendation.planner, recommendation.k_samples)
        budget_hit = selected_row.avg_control_ms <= request.time_budget_ms + 1.0e-9
        budget_hit_count += 1 if budget_hit else 0

        if budget_hit and selected_key in utilities:
            selected_utility = utilities[selected_key]
        else:
            violation = max(0.0, selected_row.avg_control_ms - request.time_budget_ms)
            selected_utility = min(utilities.values()) - 10.0 - 50.0 * violation

        regret = best_utility - selected_utility
        total_regret += regret
        if budget_hit and selected_key == best_key:
            match_count += 1

        cases.append(
            {
                "dataset": request.dataset,
                "scenario": request.scenario,
                "time_budget_ms": request.time_budget_ms,
                "planner": recommendation.planner,
                "k_samples": recommendation.k_samples,
                "selected_time_ms": selected_row.avg_control_ms,
                "budget_hit": budget_hit,
                "regret": regret,
                "oracle_planner": oracle_row.planner,
                "oracle_k_samples": oracle_row.k_samples,
                "oracle_time_ms": oracle_row.avg_control_ms,
                "rationale": recommendation.rationale,
            }
        )

    runtime_ms = benchmark_variant(variant, rows, requests, iterations)
    metrics = variant_metric_block(variant, runtime_ms)
    metrics.update(
        {
            "avg_regret": total_regret / max(1, len(requests)),
            "oracle_match_rate": match_count / max(1, len(requests)),
            "budget_hit_rate": budget_hit_count / max(1, len(requests)),
        }
    )
    return VariantEvaluation(name=variant.name, paradigm=variant.paradigm, metrics=metrics, cases=cases)


def build_report(rows: list[AggregateBenchmarkRow], iterations: int) -> ProblemReport:
    requests = build_requests(rows)
    results = [evaluate_variant(variant, rows, requests, iterations) for variant in build_variants()]

    aggregate_table = MarkdownTable(
        headers=[
            "Variant",
            "Paradigm",
            "Avg Regret",
            "Oracle Match",
            "Budget Hit",
            "Runtime ms/request",
            "Readability",
            "Extensibility",
            "Source",
        ],
        rows=[
            [
                result.name,
                result.paradigm,
                f"{result.metrics['avg_regret']:.3f}",
                f"{result.metrics['oracle_match_rate']:.2f}",
                f"{result.metrics['budget_hit_rate']:.2f}",
                f"{result.metrics['runtime_ms_per_request']:.4f}",
                f"{result.metrics['readability_score']:.1f}",
                f"{result.metrics['extensibility_score']:.1f}",
                f"`{result.metrics['source_path']}`",
            ]
            for result in results
        ],
    )

    case_tables = [
        TitledTable(
            title=result.name,
            table=MarkdownTable(
                headers=[
                    "Dataset",
                    "Scenario",
                    "Budget ms",
                    "Pick",
                    "Pick ms",
                    "Oracle",
                    "Oracle ms",
                    "Budget Hit",
                    "Regret",
                    "Rationale",
                ],
                rows=[
                    [
                        str(case["dataset"]),
                        str(case["scenario"]),
                        f"{case['time_budget_ms']:.4f}",
                        f"{case['planner']} @ K={case['k_samples']}",
                        f"{case['selected_time_ms']:.4f}",
                        f"{case['oracle_planner']} @ K={case['oracle_k_samples']}",
                        f"{case['oracle_time_ms']:.4f}",
                        "yes" if case["budget_hit"] else "no",
                        f"{case['regret']:.3f}",
                        str(case["rationale"]),
                    ]
                    for case in result.cases
                ],
            ),
        )
        for result in results
    ]

    return ProblemReport(
        slug="time_budget_selection",
        title=TITLE,
        description_lines=DESCRIPTION_LINES,
        request_summary=REQUEST_SUMMARY,
        metric_notes=METRIC_NOTES,
        request_count=len(requests),
        aggregate_table=aggregate_table,
        case_tables=case_tables,
    )
