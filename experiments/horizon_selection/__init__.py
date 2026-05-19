"""Horizon Selection design exercise.

Given an aggregate benchmark over Diff-MPPI variants that differ only in
the gradient-update horizon (``diff_mppi_3_early{1,2,4,8,16}`` plus the
full-horizon ``diff_mppi_3``), recommend which horizon to ship per
(dataset, scenario). The interface and three variants form the same
3-paradigm contract as ``experiments.planner_selection`` so the design
workflow check can enforce parity.
"""

from __future__ import annotations

from statistics import mean
from typing import Sequence

from core.horizon_selector_interface import (
    HorizonRecommendation,
    HorizonSelectionRequest,
    HorizonSelector,
)
from core.planner_selector_interface import AggregateBenchmarkRow
from experiments.horizon_selection.horizon_naming import (
    FULL_HORIZON_SENTINEL,
    parse_grad_update_horizon,
)
from experiments.horizon_selection.minimal_sufficient_selector import (
    MinimalSufficientHorizonSelector,
)
from experiments.horizon_selection.pipeline_selector import (
    PipelineHorizonSelector,
)
from experiments.horizon_selection.robust_max_selector import (
    OOPHorizonSelector,
)
from experiments.support import (
    MarkdownTable,
    ProblemReport,
    TitledTable,
    VariantEvaluation,
    benchmark_variant,
    variant_metric_block,
)


PROBLEM_KIND = "horizon_selection"
INTERFACE_FILE = "horizon_selector_interface.py"
TITLE = "Horizon Selection"
DESCRIPTION_LINES = [
    "pick one Diff-MPPI gradient-update horizon per dataset/scenario",
    "input schema stays fixed -- variants differ only in selection style",
    "score each variant on horizon regret, success vs. picked, runtime,"
    " readability and extensibility proxies",
]
REQUEST_SUMMARY = "dataset/scenario pairs over a gradient-horizon sweep"
METRIC_NOTES = [
    "`Horizon Regret`: picked horizon minus the smallest horizon that"
    " meets the success threshold (0 when the pick is minimal-sufficient,"
    " positive when over-budget)",
    "`Threshold Match`: fraction of requests where the picked row meets"
    " the success threshold",
    "`Mean Picked Horizon`: average horizon across requests; flags"
    " variants biased toward over-spending vs. under-spending",
]

DEFAULT_FULL_HORIZON_STEPS = 30


def _effective_horizon(planner: str) -> int:
    raw = parse_grad_update_horizon(planner)
    if raw is None:
        return -1
    if raw == FULL_HORIZON_SENTINEL:
        return DEFAULT_FULL_HORIZON_STEPS
    return raw


def build_requests(
    rows: Sequence[AggregateBenchmarkRow],
) -> list[HorizonSelectionRequest]:
    pairs = {
        (row.dataset, row.scenario)
        for row in rows
        if parse_grad_update_horizon(row.planner) is not None
    }
    return [
        HorizonSelectionRequest(dataset=dataset, scenario=scenario)
        for dataset, scenario in sorted(pairs)
    ]


def build_variants() -> list[HorizonSelector]:
    return [
        MinimalSufficientHorizonSelector(
            full_horizon_steps=DEFAULT_FULL_HORIZON_STEPS),
        OOPHorizonSelector(
            full_horizon_steps=DEFAULT_FULL_HORIZON_STEPS),
        PipelineHorizonSelector(
            full_horizon_steps=DEFAULT_FULL_HORIZON_STEPS),
    ]


def _oracle_horizon(
    rows: Sequence[AggregateBenchmarkRow],
    request: HorizonSelectionRequest,
) -> int | None:
    """Smallest horizon meeting the threshold for this (dataset, scenario).

    Returns None when no candidate meets the threshold."""
    sufficient = [
        row
        for row in rows
        if row.dataset == request.dataset
        and row.scenario == request.scenario
        and parse_grad_update_horizon(row.planner) is not None
        and row.success >= request.success_threshold
    ]
    if not sufficient:
        return None
    return min(_effective_horizon(row.planner) for row in sufficient)


def evaluate_variant(
    variant: HorizonSelector,
    rows: Sequence[AggregateBenchmarkRow],
    requests: list[HorizonSelectionRequest],
    iterations: int,
) -> VariantEvaluation:
    cases: list[dict[str, object]] = []
    regrets: list[float] = []
    threshold_hits = 0
    picked_horizons: list[int] = []

    for request in requests:
        recommendation: HorizonRecommendation = variant.recommend(rows, request)
        picked = recommendation.grad_update_horizon
        picked_horizons.append(picked)
        oracle = _oracle_horizon(rows, request)
        if oracle is None:
            regret = float("nan")
            oracle_label = "n/a"
        else:
            regret = float(picked - oracle)
            regrets.append(regret)
            oracle_label = str(oracle)
        if recommendation.success_rate >= request.success_threshold:
            threshold_hits += 1
        cases.append({
            "dataset": request.dataset,
            "scenario": request.scenario,
            "planner": recommendation.planner,
            "picked_horizon": picked,
            "oracle_horizon": oracle_label,
            "regret": "n/a" if oracle is None else f"{regret:+.0f}",
            "success": f"{recommendation.success_rate:.2f}",
            "rationale": recommendation.rationale,
        })

    runtime_ms = benchmark_variant(variant, rows, requests, iterations)
    metrics = variant_metric_block(variant, runtime_ms)
    metrics.update({
        "avg_regret": mean(regrets) if regrets else 0.0,
        "threshold_match_rate": threshold_hits / max(1, len(requests)),
        "mean_picked_horizon": mean(picked_horizons) if picked_horizons else 0.0,
    })
    return VariantEvaluation(
        name=variant.name,
        paradigm=variant.paradigm,
        metrics=metrics,
        cases=cases,
    )


def build_report(
    rows: Sequence[AggregateBenchmarkRow],
    iterations: int,
) -> ProblemReport:
    requests = build_requests(rows)
    results = [
        evaluate_variant(variant, rows, requests, iterations)
        for variant in build_variants()
    ]

    aggregate_table = MarkdownTable(
        headers=[
            "Variant",
            "Paradigm",
            "Avg Regret",
            "Threshold Match",
            "Mean Picked Horizon",
            "Runtime ms/request",
            "Readability",
            "Extensibility",
            "Source",
        ],
        rows=[
            [
                result.name,
                result.paradigm,
                f"{result.metrics['avg_regret']:.2f}",
                f"{result.metrics['threshold_match_rate']:.2f}",
                f"{result.metrics['mean_picked_horizon']:.1f}",
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
                    "Dataset", "Scenario", "Pick", "Picked H",
                    "Oracle H", "Regret", "Success", "Rationale",
                ],
                rows=[
                    [
                        str(case["dataset"]),
                        str(case["scenario"]),
                        str(case["planner"]),
                        str(case["picked_horizon"]),
                        str(case["oracle_horizon"]),
                        str(case["regret"]),
                        str(case["success"]),
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
