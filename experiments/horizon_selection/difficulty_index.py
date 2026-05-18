"""Map an off-grid (scenario, speed, radius) probe to a recommendation
from the sweep-trained HorizonSelector.

Strategy: load a sweep summary CSV, group rows by their difficulty cell
(scenario, dyn_speed_scale, dyn_radius_scale), and at probe time pick
the nearest cell within the same scenario by Euclidean distance on
(speed_scale, radius_scale). The selector is then run on the rows of
that cell, so the returned recommendation reflects the regime the
sweep already characterised.

This is the simplest "deployment" interface: no model training, just
nearest-neighbour over an offline benchmark grid. It also surfaces how
big the extrapolation step is by returning the matched cell distance.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Sequence

from core.horizon_selector_interface import (
    HorizonRecommendation,
    HorizonSelectionRequest,
    HorizonSelector,
)
from core.planner_selector_interface import AggregateBenchmarkRow


@dataclass(frozen=True)
class IndexedRecommendation:
    matched_speed: float
    matched_radius: float
    distance: float
    recommendation: HorizonRecommendation


def _dataset_label(speed: float, radius: float) -> str:
    return f"speed={speed:+.2f}_radius={radius:.2f}"


def load_indexed_rows(summary_csv: Path) -> list[AggregateBenchmarkRow]:
    rows: list[AggregateBenchmarkRow] = []
    with open(summary_csv) as f:
        for r in csv.DictReader(f):
            rows.append(AggregateBenchmarkRow(
                dataset=_dataset_label(
                    float(r["dyn_speed_scale"]),
                    float(r["dyn_radius_scale"])),
                scenario=r["scenario"],
                planner=r["planner"],
                k_samples=4096,
                success=float(r["success_rate"]),
                steps=0.0,
                final_distance=float(r["final_distance"]),
                cumulative_cost=float(r["cumulative_cost"]),
                avg_control_ms=float(r["avg_control_ms"]),
            ))
    return rows


def known_cells(rows: Sequence[AggregateBenchmarkRow], scenario: str,
                ) -> list[tuple[float, float]]:
    seen: set[tuple[float, float]] = set()
    for row in rows:
        if row.scenario != scenario:
            continue
        # Recover (speed, radius) from the dataset label.
        sp = float(row.dataset.split("speed=")[1].split("_radius=")[0])
        rad = float(row.dataset.split("_radius=")[1])
        seen.add((sp, rad))
    return sorted(seen)


def nearest_cell(rows: Sequence[AggregateBenchmarkRow], scenario: str,
                 speed: float, radius: float,
                 speed_weight: float = 1.0,
                 radius_weight: float = 1.0,
                 ) -> tuple[float, float, float]:
    """Return (matched_speed, matched_radius, distance) for the closest
    known cell of ``scenario`` to the probe."""
    cells = known_cells(rows, scenario)
    if not cells:
        raise ValueError(f"No sweep cells found for scenario {scenario!r}")
    best = min(cells,
               key=lambda c: hypot(
                   speed_weight * (c[0] - speed),
                   radius_weight * (c[1] - radius)))
    dist = hypot(speed_weight * (best[0] - speed),
                 radius_weight * (best[1] - radius))
    return best[0], best[1], dist


def recommend_for_probe(
    selector: HorizonSelector,
    rows: Sequence[AggregateBenchmarkRow],
    scenario: str,
    speed: float,
    radius: float,
    success_threshold: float = 0.999,
    speed_weight: float = 1.0,
    radius_weight: float = 1.0,
) -> IndexedRecommendation:
    matched_sp, matched_rad, dist = nearest_cell(
        rows, scenario, speed, radius, speed_weight, radius_weight)
    rec = selector.recommend(
        rows,
        HorizonSelectionRequest(
            dataset=_dataset_label(matched_sp, matched_rad),
            scenario=scenario,
            success_threshold=success_threshold,
        ),
    )
    return IndexedRecommendation(
        matched_speed=matched_sp,
        matched_radius=matched_rad,
        distance=dist,
        recommendation=rec,
    )


def _k_nearest(rows: Sequence[AggregateBenchmarkRow], scenario: str,
               speed: float, radius: float, k: int,
               speed_weight: float, radius_weight: float,
               ) -> list[tuple[float, float, float]]:
    cells = known_cells(rows, scenario)
    if not cells:
        raise ValueError(f"No sweep cells found for scenario {scenario!r}")
    scored = [
        (sp, rad,
         hypot(speed_weight * (sp - speed),
               radius_weight * (rad - radius)))
        for sp, rad in cells
    ]
    scored.sort(key=lambda c: c[2])
    return scored[:k]


def recommend_for_probe_robust(
    selector: HorizonSelector,
    rows: Sequence[AggregateBenchmarkRow],
    scenario: str,
    speed: float,
    radius: float,
    k: int = 3,
    success_threshold: float = 0.999,
    speed_weight: float = 1.0,
    radius_weight: float = 1.0,
) -> IndexedRecommendation:
    """Conservative variant: poll the k nearest cells, take the
    recommendation whose horizon is the maximum across them.

    The motivation is that minimal-sufficient picks the smallest
    horizon meeting the threshold, so a corner case where exactly one
    cell happens to have a tiny horizon succeed can dominate a probe
    sitting between that cell and a tougher one. Taking the max over
    k neighbours hedges against that.

    Tie-breaking among recommendations with equal horizons picks the
    candidate whose matched cell is closest to the probe (so the
    returned ``matched_speed`` / ``matched_radius`` and ``distance``
    remain meaningful).
    """
    neighbours = _k_nearest(
        rows, scenario, speed, radius, k, speed_weight, radius_weight)
    recs: list[IndexedRecommendation] = []
    for matched_sp, matched_rad, dist in neighbours:
        rec = selector.recommend(
            rows,
            HorizonSelectionRequest(
                dataset=_dataset_label(matched_sp, matched_rad),
                scenario=scenario,
                success_threshold=success_threshold,
            ),
        )
        recs.append(IndexedRecommendation(
            matched_speed=matched_sp,
            matched_radius=matched_rad,
            distance=dist,
            recommendation=rec,
        ))

    best = max(
        recs,
        key=lambda r: (
            r.recommendation.grad_update_horizon,
            -r.distance,
        ),
    )
    return best
