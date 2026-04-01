from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

from core.fixture_promotion_interface import FixturePromotionRequest
from core.planner_selector_interface import AggregateBenchmarkRow


@dataclass(frozen=True)
class DatasetProfile:
    dataset: str
    scenarios: tuple[str, ...]
    planners: tuple[str, ...]
    tags: tuple[str, ...]
    row_count: int
    avg_control_ms: float


@dataclass(frozen=True)
class PortfolioMetrics:
    datasets: tuple[str, ...]
    required_ratio: float
    required_hit: bool
    scenario_ratio: float
    planner_ratio: float
    tag_ratio: float
    depth_ratio: float
    compactness_penalty: float
    score: float


def infer_tags(scenarios: Sequence[str], planners: Sequence[str], k_values: Sequence[int]) -> tuple[str, ...]:
    tags: set[str] = set()
    if any(scenario.startswith("dynbike_") for scenario in scenarios):
        tags.update({"dynamic_obstacles", "nonholonomic", "low_budget_regime"})
    if any(scenario.startswith("uncertain_") for scenario in scenarios):
        tags.update({"dynamic_obstacles", "uncertainty"})
    if not any(scenario.startswith(("dynbike_", "uncertain_")) for scenario in scenarios):
        tags.add("nominal_nav")
    if max(k_values, default=0) >= 1024:
        tags.add("high_budget_regime")
    if min(k_values, default=10**9) <= 256:
        tags.add("low_budget_regime")
    if any(planner.startswith("feedback_") for planner in planners):
        tags.add("feedback_baseline")
    if "feedback_mppi_sens" in planners:
        tags.add("rollout_sensitivity")
    if any(planner.startswith("diff_mppi") for planner in planners):
        tags.add("hybrid_refinement")
    return tuple(sorted(tags))


def build_profiles(rows: Sequence[AggregateBenchmarkRow]) -> dict[str, DatasetProfile]:
    grouped: dict[str, list[AggregateBenchmarkRow]] = {}
    for row in rows:
        grouped.setdefault(row.dataset, []).append(row)

    profiles: dict[str, DatasetProfile] = {}
    for dataset, grouped_rows in grouped.items():
        scenarios = sorted({row.scenario for row in grouped_rows})
        planners = sorted({row.planner for row in grouped_rows})
        k_values = sorted({row.k_samples for row in grouped_rows})
        profiles[dataset] = DatasetProfile(
            dataset=dataset,
            scenarios=tuple(scenarios),
            planners=tuple(planners),
            tags=infer_tags(scenarios, planners, k_values),
            row_count=len(grouped_rows),
            avg_control_ms=sum(row.avg_control_ms for row in grouped_rows) / max(1, len(grouped_rows)),
        )
    return profiles


def candidate_portfolios(datasets: Sequence[str], max_fixtures: int) -> list[tuple[str, ...]]:
    portfolios: list[tuple[str, ...]] = []
    upper = min(max_fixtures, len(datasets))
    for size in range(1, upper + 1):
        portfolios.extend(tuple(combo) for combo in combinations(sorted(datasets), size))
    return portfolios


def coverage_union(profiles: dict[str, DatasetProfile], datasets: Iterable[str], field: str) -> set[str]:
    covered: set[str] = set()
    for dataset in datasets:
        covered.update(getattr(profiles[dataset], field))
    return covered


def score_portfolio(
    profiles: dict[str, DatasetProfile],
    request: FixturePromotionRequest,
    datasets: Sequence[str],
) -> PortfolioMetrics:
    dataset_tuple = tuple(sorted(datasets))
    all_scenarios = coverage_union(profiles, profiles.keys(), "scenarios")
    all_planners = coverage_union(profiles, profiles.keys(), "planners")
    all_tags = coverage_union(profiles, profiles.keys(), "tags")
    selected_scenarios = coverage_union(profiles, dataset_tuple, "scenarios")
    selected_planners = coverage_union(profiles, dataset_tuple, "planners")
    selected_tags = coverage_union(profiles, dataset_tuple, "tags")
    selected_rows = sum(profiles[dataset].row_count for dataset in dataset_tuple)
    total_rows = sum(profile.row_count for profile in profiles.values())

    required = set(request.required_tags)
    required_ratio = len(required & selected_tags) / max(1, len(required)) if required else 1.0
    required_hit = required.issubset(selected_tags)
    scenario_ratio = len(selected_scenarios) / max(1, len(all_scenarios))
    planner_ratio = len(selected_planners) / max(1, len(all_planners))
    tag_ratio = len(selected_tags) / max(1, len(all_tags))
    depth_ratio = selected_rows / max(1, total_rows)
    compactness_penalty = (len(dataset_tuple) - 1) / max(1, request.max_fixtures)

    score = (
        4.0 * required_ratio
        + request.scenario_weight * scenario_ratio
        + request.planner_weight * planner_ratio
        + request.tag_weight * tag_ratio
        + request.depth_weight * depth_ratio
        - request.compactness_weight * compactness_penalty
    )
    if not required_hit:
        score -= 3.0 * (1.0 - required_ratio)

    return PortfolioMetrics(
        datasets=dataset_tuple,
        required_ratio=required_ratio,
        required_hit=required_hit,
        scenario_ratio=scenario_ratio,
        planner_ratio=planner_ratio,
        tag_ratio=tag_ratio,
        depth_ratio=depth_ratio,
        compactness_penalty=compactness_penalty,
        score=score,
    )


def best_portfolio(
    profiles: dict[str, DatasetProfile],
    request: FixturePromotionRequest,
) -> PortfolioMetrics:
    datasets = sorted(profiles)
    candidates = [
        score_portfolio(profiles, request, portfolio)
        for portfolio in candidate_portfolios(datasets, request.max_fixtures)
    ]
    return max(
        candidates,
        key=lambda item: (
            item.score,
            item.required_hit,
            item.scenario_ratio,
            item.planner_ratio,
            -len(item.datasets),
            item.datasets,
        ),
    )
