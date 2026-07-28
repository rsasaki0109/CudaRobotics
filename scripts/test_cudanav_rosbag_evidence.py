#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cudanav_rosbag_evidence import (
    describe_input,
    evaluate_manifest,
    sha256_file,
)
from run_cudanav_rosbag_replay import controller_argv


def evaluation(
    database: Path, diagnostics: Path, duration: float = 90.0, samples: int = 120
) -> dict:
    return {
        "schema_version": 1,
        "evidence_mode": "shadow_controller_with_recorded_motion",
        "quality_pass": True,
        "motion": {"duration_s": duration, "database": str(database)},
        "clearance": {"database": str(database)},
        "diagnostics": {"samples": samples, "source": str(diagnostics)},
    }


class CudaNavRosbagEvidenceTest(unittest.TestCase):
    def make_run(self, root: Path) -> tuple[Path, dict]:
        bag = root / "source_bag"
        bag.mkdir()
        (bag / "metadata.yaml").write_text("storage_identifier: sqlite3\n")
        (bag / "run_0.db3").write_bytes(b"representative rosbag bytes")
        run = root / "run"
        (run / "evaluation").mkdir(parents=True)
        diagnostics = run / "diagnostics.csv"
        database = bag / "run_0.db3"
        (run / "evaluation" / "evaluation.json").write_text(
            json.dumps(evaluation(database, diagnostics)) + "\n", encoding="utf-8"
        )
        diagnostics.write_text(
            "solve_ms,valid_rollout_ratio\n4.0,0.9\n", encoding="utf-8"
        )
        (run / "controller.yaml").write_text("controller: cuda\n", encoding="utf-8")
        (run / "controller.log").write_text("controller started\n", encoding="utf-8")
        (run / "play.log").write_text("bag played\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "profile": "smoke",
            "evidence_mode": "shadow_controller_with_recorded_motion",
            "git_commit": "a" * 40,
            "git_dirty": False,
            "launch_errors": {},
            "returncodes": {
                "controller": -2,
                "record": None,
                "play": 0,
                "evaluate": 0,
            },
            "gpu": [
                {
                    "physical_index": "0",
                    "name": "Test GPU",
                    "uuid": "GPU-test",
                    "driver_version": "999.0",
                    "memory_total_mib": "8192",
                }
            ],
            "input_bag": describe_input(bag),
            "evaluation_database": {
                "source": str(database),
                "relative_path": "run_0.db3",
                "sha256": sha256_file(database),
            },
            "controller_config_sha256": sha256_file(run / "controller.yaml"),
            "diagnostics_sha256": sha256_file(diagnostics),
            "evaluation_sha256": sha256_file(
                run / "evaluation" / "evaluation.json"
            ),
            "commands": {
                "controller": [
                    "ros2",
                    "launch",
                    "stack",
                    "bringup.launch.py",
                    str(run / "controller.yaml"),
                    str(diagnostics),
                ],
                "play": ["ros2", "bag", "play", str(bag)],
                "evaluate": [
                    "python3",
                    "scripts/evaluate_mppi_rosbag.py",
                    str(database),
                    str(diagnostics),
                ],
            },
            "artifacts": {
                "evaluation": "evaluation/evaluation.json",
                "diagnostics": "diagnostics.csv",
                "controller_config": "controller.yaml",
                "controller_log": "controller.log",
                "play_log": "play.log",
                "recording": None,
            },
        }
        return run, manifest

    def test_smoke_manifest_passes_with_bound_source_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            run, manifest = self.make_run(Path(directory))
            result = evaluate_manifest(manifest, run, "smoke")
            self.assertTrue(result["passed"], result)

    def test_input_and_diagnostics_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, manifest = self.make_run(root)
            (root / "source_bag" / "run_0.db3").write_bytes(b"tampered")
            (run / "diagnostics.csv").write_text("changed\n", encoding="utf-8")
            result = evaluate_manifest(manifest, run, "smoke")
            self.assertFalse(result["checks"]["input_content_unchanged"])
            self.assertFalse(result["checks"]["diagnostics_sha256_matches"])

    def test_release_requires_longer_run_and_mcap_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, manifest = self.make_run(root)
            manifest["profile"] = "release"
            database = root / "source_bag" / "run_0.db3"
            diagnostics = run / "diagnostics.csv"
            short = evaluation(database, diagnostics, duration=30.0, samples=20)
            (run / "evaluation" / "evaluation.json").write_text(
                json.dumps(short), encoding="utf-8"
            )
            manifest["evaluation_sha256"] = sha256_file(
                run / "evaluation" / "evaluation.json"
            )
            failed = evaluate_manifest(manifest, run, "release")
            self.assertFalse(failed["checks"]["minimum_duration"])
            self.assertFalse(failed["checks"]["diagnostics_coverage"])
            self.assertFalse(failed["checks"]["artifact_recording"])

            (run / "evaluation" / "evaluation.json").write_text(
                json.dumps(evaluation(database, diagnostics)), encoding="utf-8"
            )
            manifest["evaluation_sha256"] = sha256_file(
                run / "evaluation" / "evaluation.json"
            )
            recording = run / "recording"
            recording.mkdir()
            (recording / "metadata.yaml").write_text(
                "storage_identifier: mcap\n", encoding="utf-8"
            )
            manifest["artifacts"]["recording"] = "recording"
            passed = evaluate_manifest(manifest, run, "release")
            self.assertTrue(passed["passed"], passed)

    def test_path_traversal_and_dirty_tree_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, manifest = self.make_run(root)
            manifest["git_dirty"] = True
            manifest["artifacts"]["evaluation"] = "../evaluation.json"
            result = evaluate_manifest(manifest, run, "smoke")
            self.assertFalse(result["checks"]["clean_worktree"])
            self.assertFalse(result["checks"]["artifact_evaluation"])

    def test_controller_command_is_argv_not_shell(self):
        command = controller_argv(
            "ros2 launch pkg stack.launch.py output:='{out_dir}' ; touch pwned",
            {
                "out_dir": "/tmp/output with spaces",
                "diagnostics_csv": "/tmp/diagnostics.csv",
                "controller_config": "/tmp/controller.yaml",
            },
        )
        self.assertEqual(command[4], "output:=/tmp/output with spaces")
        self.assertIn(";", command)
        self.assertEqual(command[-2:], ["touch", "pwned"])


if __name__ == "__main__":
    unittest.main()
