#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contact_matched_compute import (  # noqa: E402
    evaluate_manifest,
    infer_seed_index,
    profile_spec,
    select_largest_feasible_k,
    summarize_evaluation,
    validate_rows,
    write_csv,
    write_report,
)
from contact_robustness import SCENARIO_ORDER, sha256_file  # noqa: E402
from run_contact_matched_compute import experiment_identity  # noqa: E402
from run_contact_robustness import canonical_sha256  # noqa: E402


def rows_for(
    *,
    phase: str,
    selected_k: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    spec = profile_spec("smoke")
    seed_count = spec[f"{phase}_seed_count"]
    seed_offset = spec[f"{phase}_seed_offset"]
    rows: list[dict[str, str]] = []
    for scenario in spec["scenarios"]:
        scenario_index = SCENARIO_ORDER.index(scenario)
        for planner in spec["planners"]:
            k_values = (
                spec["k_candidates"]
                if phase == "calibration"
                else (selected_k[planner],)
            )
            for k_samples in k_values:
                for seed_index in range(seed_offset, seed_offset + seed_count):
                    max_ms = 8.0
                    misses = 0
                    if phase == "calibration" and planner == "diff_mppi_3" and k_samples == 128:
                        max_ms = 11.0
                        misses = 1
                    success = int(planner == "diff_mppi_3")
                    rows.append(
                        {
                            "scenario": scenario,
                            "planner": planner,
                            "seed": str(
                                6000
                                + scenario_index * 100
                                + seed_index * 7
                                + k_samples
                            ),
                            "k_samples": str(k_samples),
                            "success": str(success),
                            "final_distance": "0.2",
                            "avg_control_ms": "5.0",
                            "p95_control_ms": "7.0",
                            "max_control_ms": str(max_ms),
                            "control_deadline_ms": "10.0",
                            "avg_control_slot_ms": "10.05",
                            "deadline_misses": str(misses),
                            "deadline_feasible": str(int(misses == 0)),
                            "real_time_success": str(
                                int(success == 1 and misses == 0)
                            ),
                        }
                    )
    return rows


class ContactMatchedComputeTest(unittest.TestCase):
    def test_registered_seed_index_is_invertible(self):
        row = rows_for(phase="calibration")[0]
        self.assertEqual(infer_seed_index(row), 0)
        row["seed"] = str(int(row["seed"]) + 1)
        with self.assertRaisesRegex(ValueError, "seed formula"):
            infer_seed_index(row)

    def test_calibration_selects_largest_zero_miss_budget(self):
        spec = profile_spec("smoke")
        selected, table = select_largest_feasible_k(
            rows_for(phase="calibration"), spec
        )
        self.assertEqual(selected, {"mppi": 128, "diff_mppi_3": 64})
        self.assertEqual(sum(row["selected"] for row in table), 2)

    def test_evaluation_requires_disjoint_registered_matrix(self):
        spec = profile_spec("smoke")
        selected = {"mppi": 128, "diff_mppi_3": 64}
        rows = rows_for(phase="evaluation", selected_k=selected)
        self.assertEqual(
            validate_rows(
                rows,
                spec=spec,
                phase="evaluation",
                selected_k=selected,
            ),
            [],
        )
        errors = validate_rows(
            rows[:-1],
            spec=spec,
            phase="evaluation",
            selected_k=selected,
        )
        self.assertTrue(any("missing 1 registered cells" in error for error in errors))

    def test_summary_uses_real_time_success_and_holm_family(self):
        spec = profile_spec("smoke")
        selected = {"mppi": 128, "diff_mppi_3": 64}
        summary, comparisons = summarize_evaluation(
            rows_for(phase="evaluation", selected_k=selected),
            spec,
            selected,
        )
        self.assertEqual(len(summary), 2)
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0]["real_time_success_delta"], 1.0)
        self.assertIn("mcnemar_holm_p", comparisons[0])

    def test_manifest_recomputes_derived_evidence_and_binds_raw_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("raw", "logs", "artifacts"):
                (root / name).mkdir()
            spec = profile_spec("smoke")
            calibration_rows = rows_for(phase="calibration")
            selected, calibration = select_largest_feasible_k(
                calibration_rows, spec
            )
            evaluation_rows = rows_for(
                phase="evaluation", selected_k=selected
            )
            summaries, comparisons = summarize_evaluation(
                evaluation_rows, spec, selected
            )
            files = {
                "calibration_episodes": root / "calibration_episodes.csv",
                "calibration": root / "calibration.csv",
                "evaluation_episodes": root / "evaluation_episodes.csv",
                "summary": root / "summary.csv",
                "comparisons": root / "comparisons.csv",
                "report": root / "report.md",
                "state": root / "state.json",
                "plan": root / "plan.json",
                "binary": root / "artifacts" / "benchmark.exe",
            }
            write_csv(calibration_rows, files["calibration_episodes"])
            write_csv(calibration, files["calibration"])
            write_csv(evaluation_rows, files["evaluation_episodes"])
            write_csv(summaries, files["summary"])
            write_csv(comparisons, files["comparisons"])
            write_report(selected, summaries, comparisons, files["report"])
            files["binary"].write_bytes(b"benchmark")
            experiment = experiment_identity(
                "smoke",
                spec,
                files["binary"],
                files["binary"],
                sha256_file(files["binary"]),
                "a" * 40,
            )
            experiment["identity_sha256"] = canonical_sha256(experiment)
            files["plan"].write_text(
                json.dumps(experiment), encoding="utf-8"
            )
            state = {"schema_version": 1, "runs": {}}
            for phase, phase_rows in (
                ("calibration", calibration_rows),
                ("evaluation", evaluation_rows),
            ):
                for planner in spec["planners"]:
                    selected_rows = [
                        row for row in phase_rows if row["planner"] == planner
                    ]
                    raw = root / "raw" / f"{phase}_{planner}.csv"
                    log = root / "logs" / f"{phase}_{planner}.log"
                    write_csv(selected_rows, raw)
                    log.write_text("completed\n", encoding="utf-8")
                    state["runs"][f"{phase}:{planner}"] = [
                        {
                            "passed": True,
                            "csv": str(raw),
                            "csv_sha256": sha256_file(raw),
                            "log": str(log),
                            "log_sha256": sha256_file(log),
                        }
                    ]
            files["state"].write_text(json.dumps(state), encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "evidence_mode": "contact_matched_compute_gpu",
                "profile": "smoke",
                "experiment": experiment,
                "git_dirty": False,
                "gpu": [{"name": "Test GPU"}],
                "selected_k": selected,
                "matrix": {
                    "calibration_episodes": 8,
                    "evaluation_episodes": 6,
                    "scenarios": 1,
                    "planners": 2,
                    "evaluation_seeds": 3,
                },
                "integrity_gate": {
                    "calibration_complete": True,
                    "held_out_evaluation_complete": True,
                    "calibration_evaluation_seeds_disjoint": True,
                    "all_selected_budgets_zero_miss_in_calibration": True,
                    "clean_worktree": True,
                    "gpu_identified": True,
                },
                "artifacts": {
                    name: {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for name, path in files.items()
                },
                "passed": True,
            }
            result = evaluate_manifest(manifest, root, "smoke")
            self.assertTrue(result["passed"], result)
            files["summary"].write_text("tampered\n", encoding="utf-8")
            result = evaluate_manifest(manifest, root, "smoke")
            self.assertFalse(result["checks"]["artifact_summary"])
            self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
