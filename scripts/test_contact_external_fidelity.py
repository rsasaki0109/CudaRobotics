#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from contact_external_fidelity import (
    ExternalCondition,
    add_condition,
    evaluate_manifest,
    profile_spec,
    summarize_external,
    validate_condition_rows,
    write_csv,
    write_report,
)
from contact_robustness import expected_seed, sha256_file
from publish_contact_external_fidelity import publish
from run_contact_external_fidelity import experiment_identity
from run_contact_robustness import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]


def synthetic_rows() -> list[dict[str, str]]:
    spec = profile_spec("smoke")
    rows = []
    for scenario in spec["scenarios"]:
        for planner in spec["planners"]:
            for k_samples in spec["k_values"]:
                for seed_index in range(spec["seed_count"]):
                    success = int(planner == "diff_mppi_3")
                    rows.append(
                        {
                            "scenario": scenario,
                            "planner": planner,
                            "seed": str(
                                expected_seed(
                                    scenario, seed_index, k_samples
                                )
                            ),
                            "k_samples": str(k_samples),
                            "t_horizon": "16",
                            "grad_steps": "3" if success else "0",
                            "alpha": "0.01" if success else "0",
                            "reached_goal": str(success),
                            "collision_free": "1",
                            "success": str(success),
                            "steps": "100",
                            "final_distance": "0.2",
                            "min_goal_distance": "0.1",
                            "cumulative_cost": "3.0",
                            "collisions": "0",
                            "mean_control_delta": "0.1",
                            "control_roughness": "0.1",
                            "avg_control_ms": "2.0",
                            "total_control_ms": "200",
                            "episode_ms": "220",
                            "sample_budget": "102400",
                        }
                    )
    return rows


class ContactExternalFidelityTest(unittest.TestCase):
    def test_source_uses_mujoco_as_closed_loop_true_plant(self):
        source = (
            ROOT / "src" / "benchmark_diff_mppi_pushing_box_mujoco.cu"
        ).read_text(encoding="utf-8")
        for term in (
            "#include <mujoco/mujoco.h>",
            "mj_step(model_, data_)",
            "external_plant_reset",
            "external_plant_step",
            "external_plant_observe",
            "--box-mass-scale",
            "--observation-position-std",
            "--observation-angle-std",
            "mj_versionString()",
            "mjVERSION_HEADER",
        ):
            self.assertIn(term, source)
        model = (
            ROOT / "mujoco_models" / "contact_box_push.xml"
        ).read_text(encoding="utf-8")
        for name in (
            'name="box_x"',
            'name="box_y"',
            'name="box_yaw"',
            'name="pusher_x_velocity"',
            'name="obstacle_geom"',
        ):
            self.assertIn(name, model)
        root = ET.fromstring(model)
        pusher = root.find(".//body[@name='pusher']")
        self.assertIsNotNone(pusher)
        self.assertEqual(pusher.attrib["pos"].split()[:2], ["0", "0"])

    def test_release_matrix_declares_dynamics_and_sensing_axes(self):
        spec = profile_spec("release")
        self.assertEqual(spec["seed_count"], 30)
        self.assertEqual(len(spec["conditions"]), 7)
        self.assertTrue(
            any(condition.box_mass_scale != 1.0 for condition in spec["conditions"])
        )
        self.assertTrue(
            any(
                condition.observation_position_std > 0.0
                for condition in spec["conditions"]
            )
        )

    def test_condition_validation_and_statistics_retain_failures(self):
        spec = profile_spec("smoke")
        rows = synthetic_rows()
        self.assertEqual(validate_condition_rows(rows, spec), [])
        conditioned = add_condition(rows, spec["conditions"][0])
        summaries, comparisons = summarize_external(conditioned, spec)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0]["success_delta"], 1.0)
        errors = validate_condition_rows(rows[:-1], spec)
        self.assertTrue(any("row count" in error for error in errors))

    def test_invalid_condition_is_rejected(self):
        with self.assertRaises(ValueError):
            ExternalCondition("bad", box_mass_scale=0.0)

    def make_evidence(self, root: Path):
        spec = profile_spec("smoke")
        run = root / "evidence"
        for directory in ("raw", "logs", "artifacts"):
            (run / directory).mkdir(parents=True, exist_ok=True)
        source_binary = run / "source.exe"
        binary = run / "artifacts" / "benchmark.exe"
        source_model = run / "source.xml"
        model = run / "artifacts" / "model.xml"
        source_runtime = run / "mujoco-source.dll"
        runtime = run / "artifacts" / "mujoco.dll"
        source_binary.write_bytes(b"fixture")
        source_model.write_text("<mujoco/>", encoding="utf-8")
        source_runtime.write_bytes(b"runtime")
        binary.write_bytes(source_binary.read_bytes())
        model.write_bytes(source_model.read_bytes())
        runtime.write_bytes(source_runtime.read_bytes())
        experiment = experiment_identity(
            profile="smoke",
            spec=spec,
            source_binary=source_binary,
            binary=binary,
            binary_sha256=sha256_file(binary),
            source_model=source_model,
            model=model,
            model_sha256=sha256_file(model),
            runtime_source=source_runtime,
            runtime=runtime,
            runtime_sha256=sha256_file(runtime),
            commit="b" * 40,
            engine={
                "engine": "MuJoCo",
                "version": "3.3.5",
                "version_number": 335,
                "header_version_number": 335,
            },
        )
        experiment["identity_sha256"] = canonical_sha256(experiment)
        plan = run / "plan.json"
        plan.write_text(json.dumps(experiment), encoding="utf-8")
        condition = spec["conditions"][0]
        raw = run / "raw" / "nominal.attempt-001.csv"
        log = run / "logs" / "nominal.attempt-001.log"
        write_csv(synthetic_rows(), raw)
        log.write_text("complete\n", encoding="utf-8")
        state = {
            "runs": {
                condition.name: [
                    {
                        "returncode": 0,
                        "csv": str(raw),
                        "csv_sha256": sha256_file(raw),
                        "log": str(log),
                        "log_sha256": sha256_file(log),
                        "validation_errors": [],
                        "passed": True,
                    }
                ]
            }
        }
        state_path = run / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        episodes = run / "episodes.csv"
        conditioned = add_condition(synthetic_rows(), condition)
        write_csv(conditioned, episodes)
        summaries, comparisons = summarize_external(conditioned, spec)
        summary = run / "summary.csv"
        comparison = run / "comparisons.csv"
        report = run / "report.md"
        write_csv(summaries, summary)
        write_csv(comparisons, comparison)
        write_report(summaries, comparisons, report)
        artifact_paths = {
            "episodes": episodes,
            "summary": summary,
            "comparisons": comparison,
            "report": report,
            "state": state_path,
            "plan": plan,
            "binary": binary,
            "model": model,
            "runtime_library": runtime,
        }
        manifest = {
            "schema_version": 1,
            "evidence_mode": "contact_external_fidelity_mujoco_gpu",
            "profile": "smoke",
            "experiment": experiment,
            "finished_at": "2026-07-29T00:00:00+00:00",
            "git_dirty": False,
            "gpu": [
                {
                    "physical_index": "0",
                    "name": "fixture",
                    "uuid": "GPU-fixture",
                    "driver_version": "999",
                    "memory_total_mib": "8192",
                }
            ],
            "engine": {
                "engine": "MuJoCo",
                "version": "3.3.5",
                "version_number": 335,
                "header_version_number": 335,
            },
            "matrix": {
                "conditions": 1,
                "scenarios": 1,
                "planners": 2,
                "k_values": 1,
                "seeds": 2,
                "episodes": 4,
            },
            "outcome": {
                "hypothesis_is_integrity_gate": False,
                "comparison_family_size": 1,
            },
            "integrity_gate": {
                "complete_matrix": True,
                "all_raw_runs_valid": True,
                "clean_worktree": True,
                "gpu_identified": True,
                "mujoco_identified": True,
            },
            "artifacts": {
                name: {
                    "path": path.relative_to(run).as_posix(),
                    "sha256": sha256_file(path),
                }
                for name, path in artifact_paths.items()
            },
            "passed": True,
        }
        return run, manifest

    def test_manifest_and_publisher_bind_model_engine_and_raw_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, manifest = self.make_evidence(root)
            result = evaluate_manifest(manifest, run, "smoke")
            self.assertTrue(result["passed"], result)
            (run / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            published = publish(
                run, root / "published", profile="smoke", result_id="fixture"
            )
            self.assertEqual(
                set(published["published_artifacts"]),
                {"summary", "comparisons", "report"},
            )
            model = run / manifest["artifacts"]["model"]["path"]
            model.write_text("<tampered/>", encoding="utf-8")
            result = evaluate_manifest(manifest, run, "smoke")
            self.assertFalse(result["checks"]["artifact_model"])
            with self.assertRaisesRegex(ValueError, "failed validation"):
                publish(
                    run,
                    root / "rejected",
                    profile="smoke",
                    result_id="fixture",
                )


if __name__ == "__main__":
    unittest.main()
