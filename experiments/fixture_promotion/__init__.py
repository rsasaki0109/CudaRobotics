from .functional_fixture_promoter import FunctionalFixturePromoter
from .oop_fixture_promoter import OOPFixturePromoter
from .pipeline_fixture_promoter import PipelineFixturePromoter
from core.fixture_promotion_interface import FixturePromotionRecommendation, FixturePromotionRequest
from core.planner_selector_interface import AggregateBenchmarkRow
from experiments.fixture_promotion.common import best_portfolio, build_profiles, score_portfolio
from experiments.support import (
    MarkdownTable,
    ProblemReport,
    TitledTable,
    VariantEvaluation,
    benchmark_variant,
    variant_metric_block,
)


PROBLEM_KIND = "fixture_promotion"
INTERFACE_FILE = "fixture_promotion_interface.py"
TITLE = "Fixture Promotion"
DESCRIPTION_LINES = [
    "choose which benchmark fixture datasets should survive into the lightweight experiment corpus",
    "keep the raw aggregated benchmark rows fixed while varying only the portfolio-selection style",
    "score each promoter on fixture-subset regret, requirement-hit rate, runtime, readability, and extensibility proxies",
]
REQUEST_SUMMARY = "fixture-promotion requests"
METRIC_NOTES = [
    "`Avg Regret`: utility gap from an external fixture-subset oracle; lower is better",
    "`Oracle Match`: fraction of requests where the promoter selected the oracle subset exactly",
    "`Requirement Hit`: fraction of requests where all required coverage tags were present in the selected subset",
]


def build_requests(rows: list[AggregateBenchmarkRow]) -> list[FixturePromotionRequest]:
    profiles = build_profiles(rows)
    available_tags = {tag for profile in profiles.values() for tag in profile.tags}
    requests = [
        FixturePromotionRequest(
            request_id="nominal_core",
            max_fixtures=1,
            required_tags=("nominal_nav", "high_budget_regime"),
            scenario_weight=2.5,
            planner_weight=1.0,
            tag_weight=1.5,
            depth_weight=1.0,
            compactness_weight=0.75,
        ),
        FixturePromotionRequest(
            request_id="dynamic_budgeted",
            max_fixtures=1,
            required_tags=("dynamic_obstacles", "low_budget_regime"),
            scenario_weight=1.5,
            planner_weight=1.0,
            tag_weight=2.5,
            depth_weight=0.75,
            compactness_weight=0.75,
        ),
        FixturePromotionRequest(
            request_id="uncertainty_focus",
            max_fixtures=1,
            required_tags=("uncertainty", "feedback_baseline"),
            scenario_weight=1.0,
            planner_weight=1.25,
            tag_weight=2.5,
            depth_weight=1.0,
            compactness_weight=0.75,
        ),
        FixturePromotionRequest(
            request_id="balanced_mobile",
            max_fixtures=2,
            required_tags=("dynamic_obstacles", "uncertainty", "low_budget_regime"),
            scenario_weight=1.5,
            planner_weight=1.5,
            tag_weight=2.5,
            depth_weight=0.75,
            compactness_weight=0.50,
        ),
        FixturePromotionRequest(
            request_id="core_plus_robustness",
            max_fixtures=2,
            required_tags=("nominal_nav", "uncertainty"),
            scenario_weight=2.0,
            planner_weight=1.0,
            tag_weight=2.0,
            depth_weight=0.75,
            compactness_weight=0.50,
        ),
        FixturePromotionRequest(
            request_id="planner_diversity",
            max_fixtures=2,
            required_tags=(),
            scenario_weight=0.75,
            planner_weight=2.75,
            tag_weight=0.50,
            depth_weight=0.50,
            compactness_weight=1.00,
        ),
    ]
    return [
        request
        for request in requests
        if set(request.required_tags).issubset(available_tags) or not request.required_tags
    ]


def build_variants():
    return [
        FunctionalFixturePromoter(),
        OOPFixturePromoter(),
        PipelineFixturePromoter(),
    ]


def evaluate_variant(
    variant,
    rows: list[AggregateBenchmarkRow],
    requests: list[FixturePromotionRequest],
    iterations: int,
) -> VariantEvaluation:
    profiles = build_profiles(rows)
    cases: list[dict[str, object]] = []
    total_regret = 0.0
    match_count = 0
    requirement_hit_count = 0

    for request in requests:
        oracle = best_portfolio(profiles, request)
        recommendation: FixturePromotionRecommendation = variant.recommend(rows, request)
        selected = score_portfolio(profiles, request, recommendation.selected_datasets)
        regret = oracle.score - selected.score
        total_regret += regret
        if selected.datasets == oracle.datasets:
            match_count += 1
        if selected.required_hit:
            requirement_hit_count += 1

        cases.append(
            {
                "request_id": request.request_id,
                "budget": request.max_fixtures,
                "required_tags": ",".join(request.required_tags) if request.required_tags else "-",
                "pick": ",".join(selected.datasets),
                "oracle": ",".join(oracle.datasets),
                "requirement_hit": selected.required_hit,
                "regret": regret,
                "rationale": recommendation.rationale,
            }
        )

    runtime_ms = benchmark_variant(variant, rows, requests, iterations)
    metrics = variant_metric_block(variant, runtime_ms)
    metrics.update(
        {
            "avg_regret": total_regret / max(1, len(requests)),
            "oracle_match_rate": match_count / max(1, len(requests)),
            "requirement_hit_rate": requirement_hit_count / max(1, len(requests)),
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
            "Requirement Hit",
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
                f"{result.metrics['requirement_hit_rate']:.2f}",
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
                headers=["Request", "Budget", "Required Tags", "Pick", "Oracle", "Requirement Hit", "Regret", "Rationale"],
                rows=[
                    [
                        str(case["request_id"]),
                        str(case["budget"]),
                        str(case["required_tags"]),
                        str(case["pick"]),
                        str(case["oracle"]),
                        "yes" if case["requirement_hit"] else "no",
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
        slug=PROBLEM_KIND,
        title=TITLE,
        description_lines=DESCRIPTION_LINES,
        request_summary=REQUEST_SUMMARY,
        metric_notes=METRIC_NOTES,
        request_count=len(requests),
        aggregate_table=aggregate_table,
        case_tables=case_tables,
    )
