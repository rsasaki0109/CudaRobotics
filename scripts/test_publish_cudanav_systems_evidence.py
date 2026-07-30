#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from publish_cudanav_systems_evidence import (
    build_artifacts,
    build_v1_attestation,
)
import test_cudanav_autonomy_suite as autonomy_fixture
from test_cudanav_ros_ci_evidence import valid_payload
from v1_release_attestation import validate_payload


class PublishCudaNavSystemsEvidenceTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path]:
        helper = autonomy_fixture.CudaNavAutonomySuiteTest()
        suite, suite_root = helper.make_suite(root)
        suite["git_dirty"] = False
        (suite_root / "manifest.json").write_text(
            json.dumps(suite, indent=2) + "\n", encoding="utf-8"
        )
        ros_ci = root / "ros_jazzy_ci_evidence.json"
        ros_ci.write_text(
            json.dumps(valid_payload(), indent=2) + "\n",
            encoding="utf-8",
        )
        return suite_root, ros_ci

    def test_release_suite_renders_portable_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            suite_root, ros_ci = self.fixture(Path(directory))
            summary, provenance, report = build_artifacts(suite_root, ros_ci)
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["closed_loop"]["elapsed_sec"], 650.0)
            self.assertEqual(
                set(summary["closed_loop"]["required_topic_messages"].values()),
                {10},
            )
            self.assertEqual(
                summary["real_rosbag_shadow"]["evidence_mode"],
                "real_sensor_shadow_with_derived_path",
            )
            self.assertEqual(
                set(
                    summary["real_rosbag_shadow"][
                        "required_output_topic_messages"
                    ].values()
                ),
                {10},
            )
            self.assertEqual(summary["multi_gpu"]["physical_model_count"], 2)
            self.assertEqual(summary["ros_jazzy_ci"]["ros_distro"], "jazzy")
            self.assertIn("not a closed-loop claim", report)
            self.assertNotIn(str(suite_root.resolve()), report)
            self.assertEqual(provenance["git_commit"], "a" * 40)
            self.assertIs(provenance["git_dirty"], False)

    def test_release_suite_builds_valid_v1_attestation(self):
        with tempfile.TemporaryDirectory() as directory:
            suite_root, ros_ci = self.fixture(Path(directory))
            summary, provenance, _ = build_artifacts(suite_root, ros_ci)
            attestation = build_v1_attestation(
                summary,
                provenance,
                target_version="1.0.0",
                target_tag="v1.0.0",
            )
            gate = validate_payload(
                attestation,
                key="cudanav_release_evidence",
                target_version="1.0.0",
                target_tag="v1.0.0",
            )
            self.assertTrue(gate["passed"], gate)
            self.assertEqual(
                attestation["details"]["physical_gpu_models"],
                ["GPU A"],
            )

    def test_release_suite_publishes_without_optional_multi_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            suite_root, ros_ci = self.fixture(Path(directory))
            manifest_path = suite_root / "manifest.json"
            suite = json.loads(manifest_path.read_text(encoding="utf-8"))
            suite["modes"].pop("multi_gpu")
            suite["required_modes"].remove("multi_gpu")
            manifest_path.write_text(
                json.dumps(suite, indent=2) + "\n", encoding="utf-8"
            )
            summary, provenance, report = build_artifacts(suite_root, ros_ci)
            self.assertIs(summary["multi_gpu"]["provided"], False)
            self.assertIs(summary["multi_gpu"]["required_for_release"], False)
            self.assertEqual(summary["physical_gpu"]["gpu_models"], ["GPU A"])
            self.assertNotIn(
                "multi_gpu_manifest_sha256",
                provenance["source_artifacts"],
            )
            self.assertIn("not provided (optional)", report)
            attestation = build_v1_attestation(
                summary,
                provenance,
                target_version="1.0.0",
                target_tag="v1.0.0",
            )
            gate = validate_payload(
                attestation,
                key="cudanav_release_evidence",
                target_version="1.0.0",
                target_tag="v1.0.0",
            )
            self.assertTrue(gate["passed"], gate)

    def test_ros_ci_commit_must_match_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            suite_root, ros_ci = self.fixture(Path(directory))
            payload = json.loads(ros_ci.read_text(encoding="utf-8"))
            payload["git_commit"] = "b" * 40
            ros_ci.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "commits differ"):
                build_artifacts(suite_root, ros_ci)

    def test_smoke_suite_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as directory:
            suite_root, ros_ci = self.fixture(Path(directory))
            manifest_path = suite_root / "manifest.json"
            suite = json.loads(manifest_path.read_text(encoding="utf-8"))
            suite["profile"] = "smoke"
            manifest_path.write_text(json.dumps(suite) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "release gate"):
                build_artifacts(suite_root, ros_ci)

    def test_cli_write_and_check_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite_root, ros_ci = self.fixture(root)
            output = root / "published"
            command = [
                sys.executable,
                str(Path(__file__).with_name("publish_cudanav_systems_evidence.py")),
                "--suite-dir",
                str(suite_root),
                "--ros-ci",
                str(ros_ci),
                "--output-dir",
                str(output),
                "--prefix",
                "cudanav_systems_fixture",
                "--v1-attestation-name",
                "v1_cudanav_systems_release.json",
            ]
            subprocess.run(command, check=True)
            subprocess.run([*command, "--check"], check=True)
            self.assertTrue((output / "cudanav_systems_fixture_summary.json").is_file())
            self.assertTrue((output / "v1_cudanav_systems_release.json").is_file())


if __name__ == "__main__":
    unittest.main()
