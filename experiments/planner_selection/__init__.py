from .functional_selector import FunctionalSelector
from .oop_selector import OOPSelector
from .pipeline_selector import PipelineSelector
from core.planner_selector_interface import AggregateBenchmarkRow, Recommendation, SelectionRequest
from experiments.support import (
    MarkdownTable,
    ProblemReport,
    TitledTable,
    VariantEvaluation,
    benchmark_variant,
    utility_map,
    variant_metric_block,
)


PROBLEM_KIND = "planner_selection"
INTERFACE_FILE = "planner_selector_interface.py"
TITLE = "Planner Selection"
DESCRIPTION_LINES = [
    "choose one planner configuration per dataset/scenario pair",
    "keep the input schema fixed while varying only the selector implementation style",
    "score each selector on benchmark regret, runtime, readability, and extensibility proxies",
]
REQUEST_SUMMARY = "dataset/scenario pairs"
METRIC_NOTES = [
    "`Avg Regret`: utility gap from an external oracle scorer; lower is better",
    "`Oracle Match`: fraction of requests where the selector picked the oracle row exactly",
    "`Readability` and `Extensibility`: static-analysis proxies, not human review replacements",
]


def build_requests(rows: list[AggregateBenchmarkRow]) -> list[SelectionRequest]:
    keys = {(row.dataset, row.scenario) for row in rows}
    return [SelectionRequest(dataset=dataset, scenario=scenario) for dataset, scenario in sorted(keys)]


def build_variants():
    return [
        FunctionalSelector(),
        OOPSelector(),
        PipelineSelector(),
    ]


def evaluate_variant(variant, rows: list[AggregateBenchmarkRow], requests: list[SelectionRequest], iterations: int) -> VariantEvaluation:
    cases: list[dict[str, object]] = []
    total_regret = 0.0
    match_count = 0

    for request in requests:
        candidates = [row for row in rows if row.dataset == request.dataset and row.scenario == request.scenario]
        utilities = utility_map(candidates)
        best_key, best_utility = max(utilities.items(), key=lambda item: item[1])

        recommendation: Recommendation = variant.recommend(rows, request)
        selected_key = (recommendation.planner, recommendation.k_samples)
        selected_utility = utilities[selected_key]
        regret = best_utility - selected_utility
        total_regret += regret
        if selected_key == best_key:
            match_count += 1

        cases.append(
            {
                "dataset": request.dataset,
                "scenario": request.scenario,
                "planner": recommendation.planner,
                "k_samples": recommendation.k_samples,
                "regret": regret,
                "oracle_planner": best_key[0],
                "oracle_k_samples": best_key[1],
                "rationale": recommendation.rationale,
            }
        )

    runtime_ms = benchmark_variant(variant, rows, requests, iterations)
    metrics = variant_metric_block(variant, runtime_ms)
    metrics.update(
        {
            "avg_regret": total_regret / max(1, len(requests)),
            "oracle_match_rate": match_count / max(1, len(requests)),
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
                headers=["Dataset", "Scenario", "Pick", "Oracle", "Regret", "Rationale"],
                rows=[
                    [
                        str(case["dataset"]),
                        str(case["scenario"]),
                        f"{case['planner']} @ K={case['k_samples']}",
                        f"{case['oracle_planner']} @ K={case['oracle_k_samples']}",
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
        slug="planner_selection",
        title=TITLE,
        description_lines=DESCRIPTION_LINES,
        request_summary=REQUEST_SUMMARY,
        metric_notes=METRIC_NOTES,
        request_count=len(requests),
        aggregate_table=aggregate_table,
        case_tables=case_tables,
    )
