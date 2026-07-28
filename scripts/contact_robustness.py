#!/usr/bin/env python3
"""Pure schema, statistics, and reporting for contact-rich Diff-MPPI."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any


SCENARIO_ORDER = (
    "box_turn",
    "box_align",
    "box_pivot",
    "box_swivel",
    "box_align_strict",
    "box_align_detour",
    "box_align_contact_loss",
    "box_align_contact_arc",
)

REQUIRED_FIELDS = {
    "scenario",
    "planner",
    "seed",
    "k_samples",
    "t_horizon",
    "reached_goal",
    "collision_free",
    "success",
    "steps",
    "final_distance",
    "min_goal_distance",
    "cumulative_cost",
    "collisions",
    "avg_control_ms",
    "sample_budget",
}

NUMERIC_FIELDS = REQUIRED_FIELDS - {"scenario", "planner"}


@dataclass(frozen=True)
class Condition:
    name: str
    plant_gain_scale: float = 1.0
    plant_size_scale: float = 1.0
    plant_hx_scale: float = 1.0
    plant_hy_scale: float = 1.0
    true_plant: str = "smooth"
    mu: float = 0.6
    plant_damping_scale: float = 1.0

    def __post_init__(self) -> None:
        scales = (
            self.plant_gain_scale,
            self.plant_size_scale,
            self.plant_hx_scale,
            self.plant_hy_scale,
            self.plant_damping_scale,
        )
        if (
            not self.name
            or self.true_plant not in {"smooth", "hard"}
            or not all(math.isfinite(value) and value > 0.0 for value in scales)
            or not math.isfinite(self.mu)
            or self.mu < 0.0
        ):
            raise ValueError(f"invalid contact condition: {self}")

    def arguments(self) -> list[str]:
        return [
            "--plant-gain-scale",
            f"{self.plant_gain_scale:g}",
            "--plant-size-scale",
            f"{self.plant_size_scale:g}",
            "--plant-hx-scale",
            f"{self.plant_hx_scale:g}",
            "--plant-hy-scale",
            f"{self.plant_hy_scale:g}",
            "--true-plant",
            self.true_plant,
            "--mu",
            f"{self.mu:g}",
            "--plant-damping-scale",
            f"{self.plant_damping_scale:g}",
        ]


RELEASE_CONDITIONS = (
    Condition("nominal"),
    Condition("gain_0p8", plant_gain_scale=0.8),
    Condition("gain_1p2", plant_gain_scale=1.2),
    Condition("size_0p9", plant_size_scale=0.9),
    Condition("size_1p1", plant_size_scale=1.1),
    Condition("wide_box", plant_hx_scale=1.2, plant_hy_scale=0.8),
    Condition("tall_box", plant_hx_scale=0.8, plant_hy_scale=1.2),
    Condition("hard_mu_0p2", true_plant="hard", mu=0.2),
    Condition("hard_mu_0p6", true_plant="hard", mu=0.6),
    Condition("hard_mu_1p0", true_plant="hard", mu=1.0),
    Condition(
        "hard_damping_0p75",
        true_plant="hard",
        mu=0.6,
        plant_damping_scale=0.75,
    ),
    Condition(
        "hard_damping_1p25",
        true_plant="hard",
        mu=0.6,
        plant_damping_scale=1.25,
    ),
)

RELEASE_SCENARIOS = (
    "box_swivel",
    "box_align_strict",
    "box_align_detour",
    "box_align_contact_loss",
    "box_align_contact_arc",
)
RELEASE_PLANNERS = (
    "mppi",
    "diff_mppi_1",
    "diff_mppi_3",
    "soppi",
    "soppi_fast",
    "mppi_hardmodel",
)
RELEASE_K_VALUES = (128, 256, 512)


def profile_spec(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "conditions": (Condition("nominal"),),
            "scenarios": ("box_align_detour", "box_align_contact_loss"),
            "planners": ("mppi", "diff_mppi_3", "soppi_fast"),
            "k_values": (256,),
            "seed_count": 2,
            "horizon": 16,
            "comparison_planners": ("diff_mppi_3", "soppi_fast"),
            "bootstrap_resamples": 500,
        }
    if profile == "release":
        return {
            "conditions": RELEASE_CONDITIONS,
            "scenarios": RELEASE_SCENARIOS,
            "planners": RELEASE_PLANNERS,
            "k_values": RELEASE_K_VALUES,
            "seed_count": 30,
            "horizon": 16,
            "comparison_planners": ("diff_mppi_3", "soppi_fast"),
            "bootstrap_resamples": 5000,
        }
    raise ValueError(profile)


def expected_seed(scenario: str, seed_index: int, k_samples: int) -> int:
    return 6000 + SCENARIO_ORDER.index(scenario) * 100 + seed_index * 7 + k_samples


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_rows(
    rows: list[dict[str, str]],
    *,
    scenarios: tuple[str, ...],
    planners: tuple[str, ...],
    k_values: tuple[int, ...],
    seed_count: int,
) -> list[str]:
    errors: list[str] = []
    expected_count = len(scenarios) * len(planners) * len(k_values) * seed_count
    if len(rows) != expected_count:
        errors.append(f"row count {len(rows)} != expected {expected_count}")
    if not rows:
        return errors or ["CSV has no rows"]
    missing = REQUIRED_FIELDS - rows[0].keys()
    if missing:
        return ["missing CSV fields: " + ", ".join(sorted(missing))]
    expected_cells = {
        (scenario, planner, k, expected_seed(scenario, index, k))
        for scenario in scenarios
        for planner in planners
        for k in k_values
        for index in range(seed_count)
    }
    actual_cells: set[tuple[str, str, int, int]] = set()
    for row_index, row in enumerate(rows, start=2):
        try:
            values = {field: float(row[field]) for field in NUMERIC_FIELDS}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("non-finite numeric value")
            scenario = row["scenario"]
            planner = row["planner"]
            k = int(row["k_samples"])
            seed = int(row["seed"])
            key = (scenario, planner, k, seed)
            if key in actual_cells:
                errors.append(f"row {row_index}: duplicate cell {key}")
            actual_cells.add(key)
            for field in ("reached_goal", "collision_free", "success"):
                if values[field] not in (0.0, 1.0):
                    errors.append(f"row {row_index}: {field} must be binary")
            expected_success = int(
                bool(int(values["reached_goal"]))
                and bool(int(values["collision_free"]))
            )
            if int(values["success"]) != expected_success:
                errors.append(f"row {row_index}: inconsistent success flags")
            if values["steps"] <= 0 or values["avg_control_ms"] < 0:
                errors.append(f"row {row_index}: invalid steps or latency")
            expected_budget = (
                int(values["steps"])
                * int(values["k_samples"])
                * int(values["t_horizon"])
            )
            if int(values["sample_budget"]) != expected_budget:
                errors.append(f"row {row_index}: invalid sample_budget")
            if (
                scenario not in scenarios
                or planner not in planners
                or k not in k_values
            ):
                errors.append(f"row {row_index}: unexpected matrix cell")
        except (KeyError, TypeError, ValueError) as exception:
            errors.append(f"row {row_index}: {exception}")
    missing_cells = expected_cells - actual_cells
    extra_cells = actual_cells - expected_cells
    if missing_cells:
        errors.append(f"missing {len(missing_cells)} expected seed cells")
    if extra_cells:
        errors.append(f"found {len(extra_cells)} unexpected seed cells")
    return errors


def add_condition(
    rows: list[dict[str, str]], condition: Condition
) -> list[dict[str, str]]:
    fields = {f"condition_{key}": str(value) for key, value in asdict(condition).items()}
    return [{**row, **fields} for row in rows]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(successes: int, samples: int) -> tuple[float, float]:
    if samples <= 0 or not 0 <= successes <= samples:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    proportion = successes / samples
    denominator = 1.0 + z * z / samples
    centre = (proportion + z * z / (2.0 * samples)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / samples
            + z * z / (4.0 * samples * samples)
        )
        / denominator
    )
    return centre - half, centre + half


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid percentile input")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stable_seed(parts: tuple[Any, ...]) -> int:
    digest = hashlib.sha256(
        json.dumps(parts, separators=(",", ":"), sort_keys=True).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def paired_bootstrap(
    pairs: list[tuple[float, float]],
    *,
    seed_parts: tuple[Any, ...],
    resamples: int = 10000,
) -> tuple[float, float, float]:
    if not pairs or resamples < 100:
        raise ValueError("paired bootstrap needs pairs and at least 100 resamples")
    differences = [candidate - baseline for baseline, candidate in pairs]
    estimate = sum(differences) / len(differences)
    generator = random.Random(_stable_seed(seed_parts))
    sampled = []
    for _ in range(resamples):
        sampled.append(
            sum(differences[generator.randrange(len(differences))] for _ in pairs)
            / len(differences)
        )
    return estimate, percentile(sampled, 0.025), percentile(sampled, 0.975)


def mcnemar_exact(baseline_only: int, candidate_only: int) -> float:
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(baseline_only, candidate_only) + 1)
    ) / (2.0**discordant)
    return min(1.0, 2.0 * tail)


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, (index, value) in enumerate(indexed):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[index] = running
    return adjusted


def summarize(
    rows: list[dict[str, str]],
    *,
    baseline: str = "mppi",
    comparison_planners: tuple[str, ...] | None = None,
    bootstrap_resamples: int = 10000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["condition_name"],
                row["scenario"],
                row["planner"],
                int(row["k_samples"]),
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        successes = sum(int(row["success"]) for row in group)
        lower, upper = wilson_interval(successes, len(group))
        summaries.append(
            {
                "condition": key[0],
                "scenario": key[1],
                "planner": key[2],
                "k_samples": key[3],
                "episodes": len(group),
                "successes": successes,
                "success_rate": successes / len(group),
                "success_wilson_low": lower,
                "success_wilson_high": upper,
                "final_distance_mean": sum(
                    float(row["final_distance"]) for row in group
                )
                / len(group),
                "cost_mean": sum(float(row["cumulative_cost"]) for row in group)
                / len(group),
                "control_ms_mean": sum(
                    float(row["avg_control_ms"]) for row in group
                )
                / len(group),
                "control_ms_p95": percentile(
                    [float(row["avg_control_ms"]) for row in group], 0.95
                ),
                "collision_mean": sum(float(row["collisions"]) for row in group)
                / len(group),
            }
        )

    comparisons: list[dict[str, Any]] = []
    cells = sorted(
        {
            (row["condition_name"], row["scenario"], int(row["k_samples"]))
            for row in rows
        }
    )
    for condition, scenario, k_samples in cells:
        baseline_rows = {
            int(row["seed"]): row
            for row in groups[(condition, scenario, baseline, k_samples)]
        }
        planners = sorted(
            {
                planner
                for cond, scen, planner, k in groups
                if cond == condition
                and scen == scenario
                and k == k_samples
                and planner != baseline
            }
        )
        if comparison_planners is not None:
            planners = [
                planner for planner in planners if planner in comparison_planners
            ]
        for planner in planners:
            candidate_rows = {
                int(row["seed"]): row
                for row in groups[(condition, scenario, planner, k_samples)]
            }
            seeds = sorted(set(baseline_rows) & set(candidate_rows))
            if len(seeds) != len(baseline_rows) or len(seeds) != len(candidate_rows):
                raise ValueError("paired comparison has mismatched seeds")
            success_pairs = [
                (
                    float(baseline_rows[seed]["success"]),
                    float(candidate_rows[seed]["success"]),
                )
                for seed in seeds
            ]
            distance_pairs = [
                (
                    float(baseline_rows[seed]["final_distance"]),
                    float(candidate_rows[seed]["final_distance"]),
                )
                for seed in seeds
            ]
            success_delta = paired_bootstrap(
                success_pairs,
                seed_parts=(condition, scenario, planner, k_samples, "success"),
                resamples=bootstrap_resamples,
            )
            distance_delta = paired_bootstrap(
                distance_pairs,
                seed_parts=(condition, scenario, planner, k_samples, "distance"),
                resamples=bootstrap_resamples,
            )
            baseline_only = sum(
                baseline > candidate for baseline, candidate in success_pairs
            )
            candidate_only = sum(
                candidate > baseline for baseline, candidate in success_pairs
            )
            comparisons.append(
                {
                    "condition": condition,
                    "scenario": scenario,
                    "planner": planner,
                    "baseline": baseline,
                    "k_samples": k_samples,
                    "paired_episodes": len(seeds),
                    "success_delta": success_delta[0],
                    "success_delta_ci_low": success_delta[1],
                    "success_delta_ci_high": success_delta[2],
                    "final_distance_delta": distance_delta[0],
                    "final_distance_delta_ci_low": distance_delta[1],
                    "final_distance_delta_ci_high": distance_delta[2],
                    "candidate_only_successes": candidate_only,
                    "baseline_only_successes": baseline_only,
                    "mcnemar_p": mcnemar_exact(
                        baseline_only, candidate_only
                    ),
                }
            )
    adjusted = holm_adjust([float(row["mcnemar_p"]) for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["mcnemar_holm_p"] = value
    return summaries, comparisons


def write_report(
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    path: Path,
) -> None:
    significant = [
        row
        for row in comparisons
        if row["mcnemar_holm_p"] < 0.05 and row["success_delta"] > 0.0
    ]
    negative = [
        row
        for row in comparisons
        if row["mcnemar_holm_p"] < 0.05 and row["success_delta"] < 0.0
    ]
    lines = [
        "# Contact-Rich Diff-MPPI Robustness Suite",
        "",
        "This report retains every planned matrix cell, including failures. "
        "Outcome significance is not an integrity gate.",
        "",
        f"- Summary cells: {len(summaries)}",
        f"- Paired comparisons vs MPPI: {len(comparisons)}",
        f"- Holm-significant positive success cells: {len(significant)}",
        f"- Holm-significant negative success cells: {len(negative)}",
        "",
        "## Paired Success Comparisons",
        "",
        "| Condition | Scenario | K | Planner | N | Success Δ [95% bootstrap CI] | McNemar p | Holm p |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['condition']} | {row['scenario']} | {row['k_samples']} | "
            f"{row['planner']} | {row['paired_episodes']} | "
            f"{row['success_delta']:+.3f} "
            f"[{row['success_delta_ci_low']:+.3f}, "
            f"{row['success_delta_ci_high']:+.3f}] | "
            f"{row['mcnemar_p']:.4g} | {row['mcnemar_holm_p']:.4g} |"
        )
    lines += [
        "",
        "Wilson intervals are reported in `summary.csv`. Paired bootstrap uses a "
        "stable cell-derived seed. McNemar p-values are Holm-adjusted across the "
        "full declared comparison family.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_manifest(
    manifest: dict[str, Any], run_directory: Path, profile: str
) -> dict[str, Any]:
    spec = profile_spec(profile)
    root = run_directory.resolve()
    experiment = manifest.get("experiment")
    checks: dict[str, bool] = {
        "manifest_schema": manifest.get("schema_version") == 1,
        "evidence_mode": manifest.get("evidence_mode") == "contact_robustness_gpu",
        "profile": manifest.get("profile") == profile,
        "manifest_passed": manifest.get("passed") is True,
        "experiment_schema": isinstance(experiment, dict),
        "git_commit": (
            isinstance(experiment, dict)
            and bool(
                re.fullmatch(
                    r"[0-9a-fA-F]{40,64}",
                    str(experiment.get("git_commit", "")),
                )
            )
        ),
        "gpu_identity": (
            isinstance(manifest.get("gpu"), list)
            and bool(manifest["gpu"])
            and all(
                isinstance(gpu, dict)
                and all(
                    isinstance(gpu.get(field), str) and gpu[field]
                    for field in (
                        "physical_index",
                        "name",
                        "uuid",
                        "driver_version",
                        "memory_total_mib",
                    )
                )
                for gpu in manifest["gpu"]
            )
        ),
    }
    if not isinstance(experiment, dict):
        return {"profile": profile, "passed": False, "checks": checks}
    identity = dict(experiment)
    recorded_identity = identity.pop("identity_sha256", "")
    canonical = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    checks["experiment_identity"] = recorded_identity == canonical
    checks["matrix_contract"] = (
        experiment.get("profile") == profile
        and experiment.get("conditions")
        == [asdict(condition) for condition in spec["conditions"]]
        and experiment.get("scenarios") == list(spec["scenarios"])
        and experiment.get("planners") == list(spec["planners"])
        and experiment.get("k_values") == list(spec["k_values"])
        and experiment.get("seed_count") == spec["seed_count"]
        and experiment.get("horizon") == spec["horizon"]
        and experiment.get("comparison_planners")
        == list(spec["comparison_planners"])
        and isinstance(experiment.get("bootstrap_resamples"), int)
        and experiment["bootstrap_resamples"]
        >= (5000 if profile == "release" else 100)
    )
    matrix = manifest.get("matrix")
    expected_episodes = (
        len(spec["conditions"])
        * len(spec["scenarios"])
        * len(spec["planners"])
        * len(spec["k_values"])
        * spec["seed_count"]
    )
    checks["matrix_counts"] = matrix == {
        "conditions": len(spec["conditions"]),
        "scenarios": len(spec["scenarios"]),
        "planners": len(spec["planners"]),
        "k_values": len(spec["k_values"]),
        "seeds": spec["seed_count"],
        "episodes": expected_episodes,
    }
    checks["clean_release"] = (
        manifest.get("git_dirty") is False
        if profile == "release"
        else isinstance(manifest.get("git_dirty"), bool)
    )

    artifact_table = manifest.get("artifacts")
    checks["artifact_table"] = isinstance(artifact_table, dict)
    paths: dict[str, Path] = {}
    if isinstance(artifact_table, dict):
        for name in (
            "episodes",
            "summary",
            "comparisons",
            "report",
            "state",
            "plan",
            "binary",
        ):
            artifact = artifact_table.get(name)
            path: Path | None = None
            valid = False
            if isinstance(artifact, dict):
                try:
                    path = (root / artifact["path"]).resolve()
                    valid = (
                        path.is_relative_to(root)
                        and path.is_file()
                        and path.stat().st_size > 0
                        and bool(
                            re.fullmatch(
                                r"[0-9a-f]{64}",
                                str(artifact.get("sha256", "")),
                            )
                        )
                        and sha256_file(path) == artifact["sha256"]
                    )
                except (KeyError, OSError, TypeError):
                    valid = False
            checks[f"artifact_{name}"] = valid
            if valid and path is not None:
                paths[name] = path

    checks["plan_matches_experiment"] = False
    if "plan" in paths:
        try:
            checks["plan_matches_experiment"] = (
                json.loads(paths["plan"].read_text(encoding="utf-8"))
                == experiment
            )
        except (OSError, json.JSONDecodeError):
            pass
    checks["binary_identity"] = (
        "binary" in paths
        and sha256_file(paths["binary"]) == experiment.get("binary_sha256")
        and str(paths["binary"]) == experiment.get("binary")
    )

    episode_rows: list[dict[str, str]] = []
    checks["episode_matrix"] = False
    if "episodes" in paths:
        try:
            episode_rows = load_csv(paths["episodes"])
            matrix_errors = []
            for condition in spec["conditions"]:
                condition_rows = [
                    row
                    for row in episode_rows
                    if row.get("condition_name") == condition.name
                ]
                matrix_errors.extend(
                    validate_rows(
                        condition_rows,
                        scenarios=spec["scenarios"],
                        planners=spec["planners"],
                        k_values=spec["k_values"],
                        seed_count=spec["seed_count"],
                    )
                )
                expected_metadata = {
                    f"condition_{key}": str(value)
                    for key, value in asdict(condition).items()
                }
                if any(
                    any(row.get(key) != value for key, value in expected_metadata.items())
                    for row in condition_rows
                ):
                    matrix_errors.append(
                        f"{condition.name}: condition metadata mismatch"
                    )
            checks["episode_matrix"] = (
                not matrix_errors and len(episode_rows) == expected_episodes
            )
        except (OSError, csv.Error, ValueError):
            checks["episode_matrix"] = False

    expected_summary_rows = (
        len(spec["conditions"])
        * len(spec["scenarios"])
        * len(spec["planners"])
        * len(spec["k_values"])
    )
    expected_comparisons = (
        len(spec["conditions"])
        * len(spec["scenarios"])
        * len(spec["k_values"])
        * len(spec["comparison_planners"])
    )
    try:
        summary_rows = load_csv(paths["summary"])
        checks["summary_coverage"] = (
            len(summary_rows) == expected_summary_rows
            and all(int(row["episodes"]) == spec["seed_count"] for row in summary_rows)
        )
    except (KeyError, OSError, csv.Error, ValueError):
        checks["summary_coverage"] = False
    try:
        comparison_rows = load_csv(paths["comparisons"])
        checks["comparison_coverage"] = (
            len(comparison_rows) == expected_comparisons
            and all(
                int(row["paired_episodes"]) == spec["seed_count"]
                for row in comparison_rows
            )
        )
    except (KeyError, OSError, csv.Error, ValueError):
        checks["comparison_coverage"] = False

    checks["raw_run_provenance"] = False
    if "state" in paths:
        try:
            state = json.loads(paths["state"].read_text(encoding="utf-8"))
            runs = state["runs"]
            provenance_ok = set(runs) == {
                condition.name for condition in spec["conditions"]
            }
            for condition in spec["conditions"]:
                attempts = runs[condition.name]
                successful = [
                    attempt
                    for attempt in attempts
                    if attempt.get("passed") is True
                ]
                attempt_ok = False
                for attempt in successful:
                    raw = Path(attempt["csv"]).resolve()
                    log = Path(attempt["log"]).resolve()
                    attempt_ok = attempt_ok or (
                        raw.is_relative_to(root)
                        and log.is_relative_to(root)
                        and raw.is_file()
                        and log.is_file()
                        and sha256_file(raw) == attempt.get("csv_sha256")
                        and sha256_file(log) == attempt.get("log_sha256")
                        and attempt.get("returncode") == 0
                        and attempt.get("validation_errors") == []
                    )
                provenance_ok = provenance_ok and attempt_ok
            checks["raw_run_provenance"] = provenance_ok
        except (KeyError, OSError, json.JSONDecodeError, TypeError):
            checks["raw_run_provenance"] = False

    outcome = manifest.get("outcome")
    checks["outcome_not_integrity_gate"] = (
        isinstance(outcome, dict)
        and outcome.get("hypothesis_is_integrity_gate") is False
        and outcome.get("comparison_family_size") == expected_comparisons
    )
    return {
        "profile": profile,
        "passed": all(checks.values()),
        "checks": checks,
    }
