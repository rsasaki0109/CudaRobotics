#!/usr/bin/env python3
"""Contracts and statistics for MuJoCo contact-transfer evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from contact_robustness import (
    load_csv,
    sha256_file,
    summarize,
    validate_rows as validate_base_rows,
    write_csv,
)


@dataclass(frozen=True)
class ExternalCondition:
    name: str
    friction: float = 0.6
    box_mass_scale: float = 1.0
    observation_position_std: float = 0.0
    observation_angle_std: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.friction,
            self.box_mass_scale,
            self.observation_position_std,
            self.observation_angle_std,
        )
        if (
            not self.name
            or not all(math.isfinite(value) for value in values)
            or self.friction < 0.0
            or self.box_mass_scale <= 0.0
            or self.observation_position_std < 0.0
            or self.observation_angle_std < 0.0
        ):
            raise ValueError(f"invalid external-fidelity condition: {self}")

    def arguments(self) -> list[str]:
        return [
            "--friction",
            f"{self.friction:g}",
            "--box-mass-scale",
            f"{self.box_mass_scale:g}",
            "--observation-position-std",
            f"{self.observation_position_std:g}",
            "--observation-angle-std",
            f"{self.observation_angle_std:g}",
        ]


RELEASE_CONDITIONS = (
    ExternalCondition("nominal"),
    ExternalCondition("friction_0p3", friction=0.3),
    ExternalCondition("friction_0p9", friction=0.9),
    ExternalCondition("mass_0p75", box_mass_scale=0.75),
    ExternalCondition("mass_1p25", box_mass_scale=1.25),
    ExternalCondition(
        "sensor_noise_nominal",
        observation_position_std=0.01,
        observation_angle_std=0.02,
    ),
    ExternalCondition(
        "sensor_noise_high",
        observation_position_std=0.02,
        observation_angle_std=0.04,
    ),
)


def profile_spec(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "conditions": (ExternalCondition("nominal"),),
            "scenarios": ("box_align_contact_loss",),
            "planners": ("mppi", "diff_mppi_3"),
            "k_values": (64,),
            "seed_count": 2,
            "seed_offset": 0,
            "horizon": 16,
            "frame_skip": 10,
            "comparison_planners": ("diff_mppi_3",),
            "bootstrap_resamples": 500,
        }
    if profile == "release":
        return {
            "conditions": RELEASE_CONDITIONS,
            "scenarios": (
                "box_swivel",
                "box_align_strict",
                "box_align_detour",
                "box_align_contact_loss",
                "box_align_contact_arc",
            ),
            "planners": ("mppi", "diff_mppi_3", "soppi_fast"),
            "k_values": (256,),
            "seed_count": 30,
            "seed_offset": 0,
            "horizon": 16,
            "frame_skip": 10,
            "comparison_planners": ("diff_mppi_3", "soppi_fast"),
            "bootstrap_resamples": 5000,
        }
    raise ValueError(profile)


def add_condition(
    rows: list[dict[str, str]], condition: ExternalCondition
) -> list[dict[str, str]]:
    metadata = {
        f"condition_{key}": str(value)
        for key, value in asdict(condition).items()
    }
    output = []
    for row in rows:
        output.append({"condition_name": condition.name, **metadata, **row})
    return output


def validate_condition_rows(
    rows: list[dict[str, str]],
    spec: dict[str, Any],
) -> list[str]:
    errors = validate_base_rows(
        rows,
        scenarios=spec["scenarios"],
        planners=spec["planners"],
        k_values=spec["k_values"],
        seed_count=spec["seed_count"],
    )
    for index, row in enumerate(rows):
        for field in (
            "avg_control_ms",
            "episode_ms",
            "final_distance",
            "min_goal_distance",
        ):
            try:
                value = float(row[field])
                if not math.isfinite(value) or value < 0.0:
                    errors.append(f"row {index}: invalid {field}")
            except (KeyError, ValueError):
                errors.append(f"row {index}: invalid {field}")
    return errors


def summarize_external(
    rows: list[dict[str, str]], spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return summarize(
        rows,
        baseline="mppi",
        comparison_planners=spec["comparison_planners"],
        bootstrap_resamples=spec["bootstrap_resamples"],
    )


def write_report(
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    path: Path,
) -> None:
    positive = [
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
        "# MuJoCo Contact-Transfer Evidence",
        "",
        "The CUDA planners retain their nominal smooth contact model while MuJoCo "
        "executes every selected command and returns the next true state. This is "
        "closed-loop sim-to-sim transfer, not open-loop replay or real-robot evidence.",
        "",
        f"- Summary cells: {len(summaries)}",
        f"- Paired comparisons versus MPPI: {len(comparisons)}",
        f"- Holm-significant positive cells: {len(positive)}",
        f"- Holm-significant negative cells: {len(negative)}",
        "",
        "| Condition | Scenario | Planner | K | N | Success | Wilson 95% CI | Mean ms |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['condition']} | {row['scenario']} | {row['planner']} | "
            f"{row['k_samples']} | {row['episodes']} | "
            f"{row['success_rate']:.3f} | "
            f"[{row['success_wilson_low']:.3f}, "
            f"{row['success_wilson_high']:.3f}] | "
            f"{row['control_ms_mean']:.3f} |"
        )
    lines += [
        "",
        "Paired bootstrap intervals, exact McNemar p-values, and Holm-adjusted "
        "p-values are retained in `comparisons.csv`. Every failed and negative "
        "cell remains in the episode and summary tables.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_manifest(
    manifest: dict[str, Any], run_directory: Path, profile: str
) -> dict[str, Any]:
    """Validate completeness, identity, provenance, and artifact hashes."""
    spec = profile_spec(profile)
    root = run_directory.resolve()
    experiment = manifest.get("experiment")
    checks: dict[str, bool] = {
        "manifest_schema": manifest.get("schema_version") == 1,
        "evidence_mode": manifest.get("evidence_mode")
        == "contact_external_fidelity_mujoco_gpu",
        "profile": manifest.get("profile") == profile,
        "manifest_passed": manifest.get("passed") is True,
        "experiment_schema": isinstance(experiment, dict),
        "git_commit": isinstance(experiment, dict)
        and bool(
            re.fullmatch(
                r"[0-9a-fA-F]{40,64}",
                str(experiment.get("git_commit", "")),
            )
        ),
        "gpu_identity": isinstance(manifest.get("gpu"), list)
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
        ),
        "engine_identity": isinstance(manifest.get("engine"), dict)
        and manifest["engine"].get("engine") == "MuJoCo"
        and bool(manifest["engine"].get("version"))
        and isinstance(manifest["engine"].get("version_number"), int),
        "engine_header_match": isinstance(manifest.get("engine"), dict)
        and manifest["engine"].get("header_version_number")
        == manifest["engine"].get("version_number"),
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
        and experiment.get("seed_offset") == spec["seed_offset"]
        and experiment.get("horizon") == spec["horizon"]
        and experiment.get("frame_skip") == spec["frame_skip"]
        and experiment.get("comparison_planners")
        == list(spec["comparison_planners"])
        and isinstance(experiment.get("bootstrap_resamples"), int)
        and experiment["bootstrap_resamples"]
        >= (5000 if profile == "release" else 100)
        and bool(experiment.get("model_sha256"))
        and bool(experiment.get("runtime_library_sha256"))
        and experiment.get("engine") == manifest.get("engine")
    )
    expected_episodes = (
        len(spec["conditions"])
        * len(spec["scenarios"])
        * len(spec["planners"])
        * len(spec["k_values"])
        * spec["seed_count"]
    )
    checks["matrix_counts"] = manifest.get("matrix") == {
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
    gate = manifest.get("integrity_gate")
    checks["integrity_gate"] = (
        isinstance(gate, dict)
        and gate.get("complete_matrix") is True
        and gate.get("all_raw_runs_valid") is True
        and gate.get("clean_worktree")
        == (manifest.get("git_dirty") is False)
        and gate.get("gpu_identified") is True
        and gate.get("mujoco_identified") is True
    )

    artifact_table = manifest.get("artifacts")
    paths: dict[str, Path] = {}
    checks["artifact_table"] = isinstance(artifact_table, dict)
    for name in (
        "episodes",
        "summary",
        "comparisons",
        "report",
        "state",
        "plan",
        "binary",
        "model",
        "runtime_library",
    ):
        artifact = artifact_table.get(name) if isinstance(artifact_table, dict) else None
        valid = False
        if isinstance(artifact, dict):
            try:
                path = (root / artifact["path"]).resolve()
                valid = (
                    path.is_relative_to(root)
                    and path.is_file()
                    and path.stat().st_size > 0
                    and sha256_file(path) == artifact.get("sha256")
                )
                if valid:
                    paths[name] = path
            except (KeyError, OSError, TypeError):
                pass
        checks[f"artifact_{name}"] = valid
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
    checks["model_identity"] = (
        "model" in paths
        and sha256_file(paths["model"]) == experiment.get("model_sha256")
        and str(paths["model"]) == experiment.get("model")
    )
    checks["runtime_identity"] = (
        "runtime_library" in paths
        and sha256_file(paths["runtime_library"])
        == experiment.get("runtime_library_sha256")
        and str(paths["runtime_library"])
        == experiment.get("runtime_library")
    )

    checks["episode_matrix"] = False
    if "episodes" in paths:
        try:
            rows = load_csv(paths["episodes"])
            errors: list[str] = []
            for condition in spec["conditions"]:
                condition_rows = [
                    row
                    for row in rows
                    if row.get("condition_name") == condition.name
                ]
                errors.extend(validate_condition_rows(condition_rows, spec))
                metadata = {
                    f"condition_{key}": str(value)
                    for key, value in asdict(condition).items()
                }
                if any(
                    any(row.get(key) != value for key, value in metadata.items())
                    for row in condition_rows
                ):
                    errors.append(f"{condition.name}: condition metadata mismatch")
            checks["episode_matrix"] = not errors and len(rows) == expected_episodes
        except (OSError, csv.Error, ValueError):
            pass
    expected_summaries = (
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
        summaries = load_csv(paths["summary"])
        checks["summary_coverage"] = len(summaries) == expected_summaries and all(
            int(row["episodes"]) == spec["seed_count"] for row in summaries
        )
    except (KeyError, OSError, csv.Error, ValueError):
        checks["summary_coverage"] = False
    try:
        comparisons = load_csv(paths["comparisons"])
        checks["comparison_coverage"] = len(comparisons) == expected_comparisons and all(
            int(row["paired_episodes"]) == spec["seed_count"]
            for row in comparisons
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
                    attempt for attempt in attempts if attempt.get("passed") is True
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
            pass
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


__all__ = [
    "ExternalCondition",
    "add_condition",
    "evaluate_manifest",
    "profile_spec",
    "summarize_external",
    "validate_condition_rows",
    "write_csv",
    "write_report",
]
