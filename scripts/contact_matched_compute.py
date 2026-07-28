#!/usr/bin/env python3
"""Contracts and statistics for deadline-matched contact control."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import tempfile
from typing import Any

from contact_robustness import (
    SCENARIO_ORDER,
    holm_adjust,
    load_csv,
    mcnemar_exact,
    paired_bootstrap,
    sha256_file,
    wilson_interval,
    write_csv,
)


REQUIRED_FIELDS = {
    "scenario",
    "planner",
    "seed",
    "k_samples",
    "success",
    "final_distance",
    "avg_control_ms",
    "p95_control_ms",
    "max_control_ms",
    "control_deadline_ms",
    "avg_control_slot_ms",
    "deadline_misses",
    "deadline_feasible",
    "real_time_success",
}


def profile_spec(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "scenarios": ("box_align_contact_loss",),
            "planners": ("mppi", "diff_mppi_3"),
            "k_candidates": (64, 128),
            "calibration_seed_count": 2,
            "calibration_seed_offset": 0,
            "evaluation_seed_count": 3,
            "evaluation_seed_offset": 100,
            "deadline_ms": 10.0,
            "horizon": 16,
            "bootstrap_resamples": 500,
        }
    if profile == "release":
        return {
            "scenarios": (
                "box_swivel",
                "box_align_strict",
                "box_align_detour",
                "box_align_contact_loss",
                "box_align_contact_arc",
            ),
            "planners": ("mppi", "diff_mppi_3", "soppi_fast"),
            "k_candidates": (64, 128, 256, 512, 1024),
            "calibration_seed_count": 5,
            "calibration_seed_offset": 0,
            "evaluation_seed_count": 30,
            "evaluation_seed_offset": 100,
            "deadline_ms": 10.0,
            "horizon": 16,
            "bootstrap_resamples": 5000,
        }
    raise ValueError(profile)


def infer_seed_index(row: dict[str, str]) -> int:
    scenario_index = SCENARIO_ORDER.index(row["scenario"])
    numerator = (
        int(row["seed"])
        - 6000
        - scenario_index * 100
        - int(row["k_samples"])
    )
    if numerator < 0 or numerator % 7:
        raise ValueError("seed does not match the registered seed formula")
    return numerator // 7


def _validate_metric_contract(row: dict[str, str], deadline_ms: float) -> list[str]:
    errors: list[str] = []
    values = {
        field: float(row[field])
        for field in (
            "avg_control_ms",
            "p95_control_ms",
            "max_control_ms",
            "control_deadline_ms",
            "avg_control_slot_ms",
            "final_distance",
        )
    }
    if not all(math.isfinite(value) and value >= 0.0 for value in values.values()):
        errors.append("non-finite or negative metric")
    if not math.isclose(
        values["control_deadline_ms"], deadline_ms, rel_tol=0.0, abs_tol=1e-5
    ):
        errors.append("deadline differs from registered budget")
    if (
        values["avg_control_ms"] > values["max_control_ms"] + 1e-5
        or values["p95_control_ms"] > values["max_control_ms"] + 1e-5
    ):
        errors.append("control latency order is invalid")
    if values["avg_control_slot_ms"] + 0.01 < deadline_ms:
        errors.append("control slot was not enforced")
    misses = int(row["deadline_misses"])
    feasible = int(row["deadline_feasible"])
    success = int(row["success"])
    real_time_success = int(row["real_time_success"])
    if misses < 0 or feasible not in (0, 1) or success not in (0, 1):
        errors.append("invalid deadline or success indicator")
    if feasible != int(misses == 0):
        errors.append("deadline_feasible disagrees with misses")
    if real_time_success != int(success == 1 and feasible == 1):
        errors.append("real_time_success disagrees with task/deadline result")
    return errors


def validate_rows(
    rows: list[dict[str, str]],
    *,
    spec: dict[str, Any],
    phase: str,
    selected_k: dict[str, int] | None = None,
) -> list[str]:
    if phase not in {"calibration", "evaluation"}:
        raise ValueError(phase)
    errors: list[str] = []
    if not rows:
        return ["CSV is empty"]
    missing = REQUIRED_FIELDS - set(rows[0])
    if missing:
        return [f"missing fields: {sorted(missing)}"]
    seed_count = spec[f"{phase}_seed_count"]
    seed_offset = spec[f"{phase}_seed_offset"]
    k_by_planner = (
        {planner: tuple(spec["k_candidates"]) for planner in spec["planners"]}
        if phase == "calibration"
        else {
            planner: (selected_k or {}).get(planner, ())
            for planner in spec["planners"]
        }
    )
    if phase == "evaluation" and (
        selected_k is None or set(selected_k) != set(spec["planners"])
    ):
        return ["evaluation requires one selected K for every planner"]

    expected = {
        (scenario, planner, int(k), seed_index)
        for scenario in spec["scenarios"]
        for planner in spec["planners"]
        for k in (
            k_by_planner[planner]
            if isinstance(k_by_planner[planner], tuple)
            else (k_by_planner[planner],)
        )
        for seed_index in range(seed_offset, seed_offset + seed_count)
    }
    observed: list[tuple[str, str, int, int]] = []
    for index, row in enumerate(rows):
        try:
            key = (
                row["scenario"],
                row["planner"],
                int(row["k_samples"]),
                infer_seed_index(row),
            )
            observed.append(key)
            errors.extend(
                f"row {index}: {error}"
                for error in _validate_metric_contract(row, spec["deadline_ms"])
            )
        except (KeyError, ValueError) as exception:
            errors.append(f"row {index}: {exception}")
    if len(observed) != len(set(observed)):
        errors.append("duplicate scenario/planner/K/seed-index cells")
    missing_cells = expected - set(observed)
    unexpected_cells = set(observed) - expected
    if missing_cells:
        errors.append(f"missing {len(missing_cells)} registered cells")
    if unexpected_cells:
        errors.append(f"found {len(unexpected_cells)} unexpected cells")
    return errors


def select_largest_feasible_k(
    rows: list[dict[str, str]], spec: dict[str, Any]
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    errors = validate_rows(rows, spec=spec, phase="calibration")
    if errors:
        raise ValueError("; ".join(errors))
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["planner"], int(row["k_samples"]))].append(row)
    selected: dict[str, int] = {}
    calibration: list[dict[str, Any]] = []
    for planner in spec["planners"]:
        feasible: list[int] = []
        for k_samples in spec["k_candidates"]:
            cell = grouped[(planner, k_samples)]
            misses = sum(int(row["deadline_misses"]) for row in cell)
            maximum = max(float(row["max_control_ms"]) for row in cell)
            is_feasible = misses == 0 and maximum <= spec["deadline_ms"] + 1e-5
            if is_feasible:
                feasible.append(k_samples)
            calibration.append(
                {
                    "planner": planner,
                    "k_samples": k_samples,
                    "episodes": len(cell),
                    "deadline_ms": spec["deadline_ms"],
                    "deadline_misses": misses,
                    "max_control_ms": maximum,
                    "median_p95_control_ms": median(
                        float(row["p95_control_ms"]) for row in cell
                    ),
                    "feasible": int(is_feasible),
                    "selected": 0,
                }
            )
        if not feasible:
            raise ValueError(f"no registered K meets the deadline for {planner}")
        selected[planner] = max(feasible)
    for row in calibration:
        row["selected"] = int(selected[row["planner"]] == row["k_samples"])
    return selected, calibration


def summarize_evaluation(
    rows: list[dict[str, str]],
    spec: dict[str, Any],
    selected_k: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors = validate_rows(
        rows, spec=spec, phase="evaluation", selected_k=selected_k
    )
    if errors:
        raise ValueError("; ".join(errors))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["planner"])].append(row)
    summaries: list[dict[str, Any]] = []
    for scenario in spec["scenarios"]:
        for planner in spec["planners"]:
            cell = grouped[(scenario, planner)]
            successes = sum(int(row["real_time_success"]) for row in cell)
            lower, upper = wilson_interval(successes, len(cell))
            summaries.append(
                {
                    "scenario": scenario,
                    "planner": planner,
                    "k_samples": selected_k[planner],
                    "episodes": len(cell),
                    "real_time_successes": successes,
                    "real_time_success_rate": successes / len(cell),
                    "success_ci_low": lower,
                    "success_ci_high": upper,
                    "deadline_misses": sum(
                        int(row["deadline_misses"]) for row in cell
                    ),
                    "mean_control_ms": sum(
                        float(row["avg_control_ms"]) for row in cell
                    )
                    / len(cell),
                    "max_control_ms": max(
                        float(row["max_control_ms"]) for row in cell
                    ),
                    "mean_final_distance": sum(
                        float(row["final_distance"]) for row in cell
                    )
                    / len(cell),
                }
            )

    baseline = spec["planners"][0]
    comparisons: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for scenario in spec["scenarios"]:
        baseline_rows = {
            infer_seed_index(row): row for row in grouped[(scenario, baseline)]
        }
        for planner in spec["planners"][1:]:
            planner_rows = {
                infer_seed_index(row): row for row in grouped[(scenario, planner)]
            }
            pairs = [
                (
                    float(baseline_rows[index]["real_time_success"]),
                    float(planner_rows[index]["real_time_success"]),
                )
                for index in sorted(baseline_rows)
            ]
            baseline_only = sum(a == 1.0 and b == 0.0 for a, b in pairs)
            planner_only = sum(a == 0.0 and b == 1.0 for a, b in pairs)
            delta, low, high = paired_bootstrap(
                pairs,
                seed_parts=("matched_compute", scenario, planner),
                resamples=spec["bootstrap_resamples"],
            )
            p_value = mcnemar_exact(baseline_only, planner_only)
            raw_p_values.append(p_value)
            comparisons.append(
                {
                    "scenario": scenario,
                    "baseline": baseline,
                    "planner": planner,
                    "baseline_k": selected_k[baseline],
                    "planner_k": selected_k[planner],
                    "paired_episodes": len(pairs),
                    "real_time_success_delta": delta,
                    "success_delta_ci_low": low,
                    "success_delta_ci_high": high,
                    "mcnemar_p": p_value,
                }
            )
    for row, adjusted in zip(comparisons, holm_adjust(raw_p_values)):
        row["mcnemar_holm_p"] = adjusted
    return summaries, comparisons


def write_report(
    selected_k: dict[str, int],
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        "# Deadline-Matched Contact Control",
        "",
        "Every planner receives the same enforced wall-clock control slot. "
        "`real_time_success` requires both task success and zero deadline misses.",
        "",
        "## Calibrated budgets",
        "",
        "| Planner | Selected K |",
        "|---|---:|",
    ]
    lines.extend(f"| {planner} | {k_samples} |" for planner, k_samples in selected_k.items())
    lines += [
        "",
        "## Evaluation",
        "",
        "| Scenario | Planner | K | RT success | Deadline misses | Max ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['scenario']} | {row['planner']} | {row['k_samples']} | "
            f"{row['real_time_success_rate']:.3f} | {row['deadline_misses']} | "
            f"{row['max_control_ms']:.3f} |"
        )
    lines += [
        "",
        "## Paired comparisons versus MPPI",
        "",
        "| Scenario | Planner | N | RT success delta [95% CI] | Holm p |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['scenario']} | {row['planner']} | {row['paired_episodes']} | "
            f"{row['real_time_success_delta']:+.3f} "
            f"[{row['success_delta_ci_low']:+.3f}, "
            f"{row['success_delta_ci_high']:+.3f}] | "
            f"{row['mcnemar_holm_p']:.4g} |"
        )
    lines += [
        "",
        "Calibration seeds and evaluation seeds are disjoint. The largest registered "
        "K with zero calibration deadline misses is selected independently for each "
        "planner before held-out evaluation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _safe_artifact(root: Path, item: dict[str, Any]) -> Path | None:
    try:
        path = (root / item["path"]).resolve()
        path.relative_to(root)
        if (
            not path.is_file()
            or sha256_file(path) != item["sha256"]
        ):
            return None
        return path
    except (KeyError, OSError, ValueError):
        return None


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_manifest(
    manifest: dict[str, Any], run_directory: Path, profile: str
) -> dict[str, Any]:
    spec = profile_spec(profile)
    root = run_directory.resolve()
    experiment = manifest.get("experiment")
    if (
        isinstance(experiment, dict)
        and isinstance(experiment.get("deadline_ms"), (int, float))
        and math.isfinite(experiment["deadline_ms"])
        and experiment["deadline_ms"] > 0.0
    ):
        spec["deadline_ms"] = float(experiment["deadline_ms"])
    checks: dict[str, bool] = {
        "manifest_schema": manifest.get("schema_version") == 1,
        "evidence_mode": manifest.get("evidence_mode")
        == "contact_matched_compute_gpu",
        "profile": manifest.get("profile") == profile,
        "manifest_passed": manifest.get("passed") is True,
        "experiment_schema": isinstance(experiment, dict),
        "clean_release": (
            manifest.get("git_dirty") is False
            if profile == "release"
            else isinstance(manifest.get("git_dirty"), bool)
        ),
        "gpu_identified": isinstance(manifest.get("gpu"), list)
        and bool(manifest.get("gpu")),
    }
    if isinstance(experiment, dict):
        identity = dict(experiment)
        declared_identity = identity.pop("identity_sha256", None)
        checks["experiment_identity"] = (
            declared_identity == _canonical_sha256(identity)
        )
        checks["experiment_profile"] = all(
            (
                experiment.get("profile") == profile,
                experiment.get("scenarios") == list(spec["scenarios"]),
                experiment.get("planners") == list(spec["planners"]),
                experiment.get("k_candidates") == list(spec["k_candidates"]),
                experiment.get("calibration_seed_count")
                == spec["calibration_seed_count"],
                experiment.get("calibration_seed_offset")
                == spec["calibration_seed_offset"],
                experiment.get("evaluation_seed_count")
                == spec["evaluation_seed_count"],
                experiment.get("evaluation_seed_offset")
                == spec["evaluation_seed_offset"],
                experiment.get("deadline_ms") == spec["deadline_ms"],
                experiment.get("horizon") == spec["horizon"],
                experiment.get("bootstrap_resamples")
                == spec["bootstrap_resamples"],
            )
        )
    else:
        checks["experiment_identity"] = False
        checks["experiment_profile"] = False

    required_artifacts = {
        "calibration_episodes",
        "calibration",
        "evaluation_episodes",
        "summary",
        "comparisons",
        "report",
        "state",
        "plan",
        "binary",
    }
    table = manifest.get("artifacts")
    paths: dict[str, Path] = {}
    checks["artifact_table"] = (
        isinstance(table, dict) and set(table) == required_artifacts
    )
    if isinstance(table, dict):
        for name in required_artifacts:
            path = _safe_artifact(root, table.get(name, {}))
            checks[f"artifact_{name}"] = path is not None
            if path is not None:
                paths[name] = path
    else:
        for name in required_artifacts:
            checks[f"artifact_{name}"] = False

    if isinstance(experiment, dict) and "binary" in paths:
        checks["binary_binding"] = (
            sha256_file(paths["binary"]) == experiment.get("binary_sha256")
        )
    else:
        checks["binary_binding"] = False
    try:
        plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
        checks["plan_binding"] = plan == experiment
    except (KeyError, OSError, json.JSONDecodeError):
        checks["plan_binding"] = False

    selected_k = manifest.get("selected_k")
    try:
        calibration_rows = load_csv(paths["calibration_episodes"])
        calibration_errors = validate_rows(
            calibration_rows, spec=spec, phase="calibration"
        )
        recomputed_k, calibration_table = select_largest_feasible_k(
            calibration_rows, spec
        )
        checks["calibration_matrix"] = not calibration_errors
        checks["selected_k"] = recomputed_k == selected_k
    except (KeyError, OSError, ValueError):
        calibration_rows = []
        calibration_table = []
        checks["calibration_matrix"] = False
        checks["selected_k"] = False
    try:
        evaluation_rows = load_csv(paths["evaluation_episodes"])
        evaluation_errors = validate_rows(
            evaluation_rows,
            spec=spec,
            phase="evaluation",
            selected_k=selected_k,
        )
        summaries, comparisons = summarize_evaluation(
            evaluation_rows, spec, selected_k
        )
        checks["evaluation_matrix"] = not evaluation_errors
    except (KeyError, OSError, TypeError, ValueError):
        evaluation_rows = []
        summaries = []
        comparisons = []
        checks["evaluation_matrix"] = False

    if calibration_table and summaries:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            generated = {
                "calibration": temporary / "calibration.csv",
                "summary": temporary / "summary.csv",
                "comparisons": temporary / "comparisons.csv",
                "report": temporary / "report.md",
            }
            write_csv(calibration_table, generated["calibration"])
            write_csv(summaries, generated["summary"])
            write_csv(comparisons, generated["comparisons"])
            write_report(
                selected_k, summaries, comparisons, generated["report"]
            )
            checks["derived_artifacts_reproducible"] = all(
                name in paths
                and sha256_file(generated[name]) == sha256_file(paths[name])
                for name in generated
            )
    else:
        checks["derived_artifacts_reproducible"] = False

    matrix = manifest.get("matrix")
    checks["matrix_counts"] = isinstance(matrix, dict) and matrix == {
        "calibration_episodes": (
            len(spec["scenarios"])
            * len(spec["planners"])
            * len(spec["k_candidates"])
            * spec["calibration_seed_count"]
        ),
        "evaluation_episodes": (
            len(spec["scenarios"])
            * len(spec["planners"])
            * spec["evaluation_seed_count"]
        ),
        "scenarios": len(spec["scenarios"]),
        "planners": len(spec["planners"]),
        "evaluation_seeds": spec["evaluation_seed_count"],
    }
    gate = manifest.get("integrity_gate")
    checks["integrity_gate"] = isinstance(gate, dict) and all(
        gate.get(name) is True
        for name in (
            "calibration_complete",
            "held_out_evaluation_complete",
            "calibration_evaluation_seeds_disjoint",
            "all_selected_budgets_zero_miss_in_calibration",
            "clean_worktree",
            "gpu_identified",
        )
    )
    try:
        state = json.loads(paths["state"].read_text(encoding="utf-8"))
        expected_keys = {
            f"{phase}:{planner}"
            for phase in ("calibration", "evaluation")
            for planner in spec["planners"]
        }
        successful = {
            key
            for key, history in state["runs"].items()
            if any(
                attempt.get("passed") is True
                and Path(attempt.get("csv", "")).is_file()
                and sha256_file(Path(attempt["csv"]))
                == attempt.get("csv_sha256")
                and Path(attempt.get("log", "")).is_file()
                and sha256_file(Path(attempt["log"]))
                == attempt.get("log_sha256")
                for attempt in history
            )
        }
        checks["raw_attempt_binding"] = set(state["runs"]) == expected_keys and (
            successful == expected_keys
        )
    except (KeyError, OSError, json.JSONDecodeError):
        checks["raw_attempt_binding"] = False
    return {
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
    }


__all__ = [
    "infer_seed_index",
    "evaluate_manifest",
    "profile_spec",
    "select_largest_feasible_k",
    "summarize_evaluation",
    "validate_rows",
    "write_csv",
    "write_report",
]
