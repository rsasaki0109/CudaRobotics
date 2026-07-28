#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from contact_robustness import (
    add_condition,
    evaluate_manifest,
    expected_seed,
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap,
    profile_spec,
    sha256_file,
    summarize,
    validate_rows,
    wilson_interval,
    write_csv,
    write_report,
)
from run_contact_robustness import canonical_sha256, experiment_identity


ROOT = Path(__file__).resolve().parents[1]


def synthetic_rows() -> list[dict[str, str]]:
    spec = profile_spec("smoke")
    rows = []
    for scenario in spec["scenarios"]:
        for planner in spec["planners"]:
            for k_samples in spec["k_values"]:
                for seed_index in range(spec["seed_count"]):
                    success = 0
                    if scenario == "box_align_contact_loss" and planner != "mppi":
                        success = 1
                    if (
                        scenario == "box_align_detour"
                        and planner == "diff_mppi_3"
                        and seed_index == 0
                    ):
                        success = 1
                    steps = 40 if success else 240
                    rows.append(
                        {
                            "scenario": scenario,
                            "planner": planner,
                            "seed": str(
                                expected_seed(scenario, seed_index, k_samples)
                            ),
                            "k_samples": str(k_samples),
                            "t_horizon": "16",
                            "grad_steps": "0",
                            "alpha": "0",
                            "reached_goal": str(success),
                            "collision_free": "1",
                            "success": str(success),
                            "steps": str(steps),
                            "final_distance": str(0.2 if success else 0.5),
                            "min_goal_distance": "0.1",
                            "cumulative_cost": str(2.0 if success else 5.0),
                            "collisions": "0",
                            "mean_control_delta": "0.2",
                            "control_roughness": "0.1",
                            "avg_control_ms": "2.5",
                            "total_control_ms": str(steps * 2.5),
                            "episode_ms": str(steps * 2.6),
                            "sample_budget": str(steps * k_samples * 16),
                        }
                    )
    return rows


class ContactRobustnessTest(unittest.TestCase):
    def test_benchmark_source_exposes_preregistered_robustness_axes(self):
        source = (
            ROOT / "src" / "benchmark_diff_mppi_pushing_box.cu"
        ).read_text(encoding="utf-8")
        for option in (
            "--plant-gain-scale",
            "--plant-size-scale",
            "--plant-hx-scale",
            "--plant-hy-scale",
            "--true-plant",
            "--mu",
            "--plant-damping-scale",
        ):
            self.assertIn(option, source)
        ordered_scenarios = ", ".join(
            f"make_{scenario}()" for scenario in (
                "box_turn",
                "box_align",
                "box_pivot",
                "box_swivel",
                "box_align_strict",
                "box_align_detour",
                "box_align_contact_loss",
                "box_align_contact_arc",
            )
        )
        self.assertIn(ordered_scenarios, source)

    def test_matrix_validation_retains_every_seed_and_failure(self):
        spec = profile_spec("smoke")
        rows = synthetic_rows()
        self.assertEqual(
            validate_rows(
                rows,
                scenarios=spec["scenarios"],
                planners=spec["planners"],
                k_values=spec["k_values"],
                seed_count=spec["seed_count"],
            ),
            [],
        )
        errors = validate_rows(
            rows[:-1],
            scenarios=spec["scenarios"],
            planners=spec["planners"],
            k_values=spec["k_values"],
            seed_count=spec["seed_count"],
        )
        self.assertTrue(any("row count" in error for error in errors))
        self.assertTrue(any("missing 1 expected seed cells" in error for error in errors))

    def test_statistics_are_paired_deterministic_and_multiplicity_aware(self):
        lower, upper = wilson_interval(4, 4)
        self.assertGreater(lower, 0.5)
        self.assertAlmostEqual(upper, 1.0)
        pairs = [(0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]
        first = paired_bootstrap(pairs, seed_parts=("fixture",), resamples=500)
        second = paired_bootstrap(pairs, seed_parts=("fixture",), resamples=500)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 0.5)
        self.assertEqual(mcnemar_exact(0, 6), 0.03125)
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_summary_covers_all_cells_and_primary_comparisons(self):
        spec = profile_spec("smoke")
        rows = add_condition(synthetic_rows(), spec["conditions"][0])
        summary, comparisons = summarize(
            rows,
            comparison_planners=spec["comparison_planners"],
            bootstrap_resamples=200,
        )
        self.assertEqual(len(summary), 6)
        self.assertEqual(len(comparisons), 4)
        contact_diff = next(
            row
            for row in comparisons
            if row["scenario"] == "box_align_contact_loss"
            and row["planner"] == "diff_mppi_3"
        )
        self.assertEqual(contact_diff["success_delta"], 1.0)
        self.assertEqual(contact_diff["paired_episodes"], 2)

    def make_evidence(self, root: Path) -> tuple[Path, dict]:
        spec = profile_spec("smoke")
        run = root / "run"
        for directory in ("raw", "logs", "artifacts"):
            (run / directory).mkdir(parents=True, exist_ok=True)
        source_binary = root / "source_binary"
        source_binary.write_bytes(b"binary")
        staged_binary = run / "artifacts" / "source_binary"
        staged_binary.write_bytes(source_binary.read_bytes())
        experiment = experiment_identity(
            profile="smoke",
            spec=spec,
            binary_source=source_binary,
            binary=staged_binary,
            binary_sha256=sha256_file(source_binary),
            commit="a" * 40,
        )
        experiment["identity_sha256"] = canonical_sha256(experiment)
        plan = run / "plan.json"
        plan.write_text(json.dumps(experiment), encoding="utf-8")
        condition = spec["conditions"][0]
        raw = run / "raw" / "nominal.attempt-001.csv"
        log = run / "logs" / "nominal.attempt-001.log"
        write_csv(synthetic_rows(), raw)
        log.write_text("benchmark completed\n", encoding="utf-8")
        state = {
            "schema_version": 1,
            "started_at": "2026-07-29T00:00:00+00:00",
            "runs": {
                "nominal": [
                    {
                        "attempt": 1,
                        "command": [str(staged_binary)],
                        "returncode": 0,
                        "elapsed_sec": 1.0,
                        "csv": str(raw),
                        "csv_sha256": sha256_file(raw),
                        "log": str(log),
                        "log_sha256": sha256_file(log),
                        "validation_errors": [],
                        "passed": True,
                    }
                ]
            },
        }
        state_path = run / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        episodes = run / "episodes.csv"
        episode_rows = add_condition(synthetic_rows(), condition)
        write_csv(episode_rows, episodes)
        summaries, comparisons = summarize(
            episode_rows,
            comparison_planners=spec["comparison_planners"],
            bootstrap_resamples=spec["bootstrap_resamples"],
        )
        summary_path = run / "summary.csv"
        comparisons_path = run / "comparisons.csv"
        report = run / "report.md"
        write_csv(summaries, summary_path)
        write_csv(comparisons, comparisons_path)
        write_report(summaries, comparisons, report)
        artifacts = {
            name: {
                "path": path.relative_to(run).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in (
                ("episodes", episodes),
                ("summary", summary_path),
                ("comparisons", comparisons_path),
                ("report", report),
                ("state", state_path),
                ("plan", plan),
                ("binary", staged_binary),
            )
        }
        manifest = {
            "schema_version": 1,
            "evidence_mode": "contact_robustness_gpu",
            "profile": "smoke",
            "experiment": experiment,
            "git_dirty": False,
            "gpu": [
                {
                    "physical_index": "0",
                    "name": "Test GPU",
                    "uuid": "GPU-test",
                    "driver_version": "999",
                    "memory_total_mib": "8192",
                }
            ],
            "matrix": {
                "conditions": 1,
                "scenarios": 2,
                "planners": 3,
                "k_values": 1,
                "seeds": 2,
                "episodes": 12,
            },
            "outcome": {
                "hypothesis_is_integrity_gate": False,
                "holm_significant_positive_success_cells": 0,
                "holm_significant_negative_success_cells": 0,
                "comparison_family_size": 4,
            },
            "artifacts": artifacts,
            "passed": True,
        }
        return run, manifest

    def test_manifest_binds_binary_raw_runs_and_derived_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            run, manifest = self.make_evidence(Path(directory))
            result = evaluate_manifest(manifest, run, "smoke")
            self.assertTrue(result["passed"], result)
            (run / "episodes.csv").write_text("tampered\n", encoding="utf-8")
            result = evaluate_manifest(manifest, run, "smoke")
            self.assertFalse(result["checks"]["artifact_episodes"])
            self.assertFalse(result["passed"])

    def test_manifest_rejects_path_traversal_and_matrix_shrink(self):
        with tempfile.TemporaryDirectory() as directory:
            run, manifest = self.make_evidence(Path(directory))
            manifest["artifacts"]["report"]["path"] = "../report.md"
            manifest["matrix"]["seeds"] = 1
            result = evaluate_manifest(manifest, run, "smoke")
            self.assertFalse(result["checks"]["artifact_report"])
            self.assertFalse(result["checks"]["matrix_counts"])


if __name__ == "__main__":
    unittest.main()
