#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cudanav_ros_ci_evidence import REQUIRED_CHECKS, REQUIRED_PACKAGES, evaluate


def valid_payload() -> dict:
    return {
        "schema_version": 1,
        "evidence_mode": "ros_jazzy_ci",
        "status": "passed",
        "generated_at": "2026-07-29T00:00:00+00:00",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "github": {
            "repository": "rsasaki0109/CudaRobotics",
            "run_id": 123,
            "run_attempt": 1,
            "run_url": (
                "https://github.com/rsasaki0109/CudaRobotics/actions/runs/123"
            ),
        },
        "platform": {
            "os": "Linux",
            "arch": "X64",
            "image": "ubuntu-24.04",
        },
        "ros": {"distro": "jazzy"},
        "cuda": {
            "toolkit": "12.6",
            "compiler": "Cuda compilation tools, release 12.6, V12.6.85",
        },
        "packages": sorted(REQUIRED_PACKAGES),
        "checks": {name: "passed" for name in REQUIRED_CHECKS},
    }


class CudaNavRosCiEvidenceTest(unittest.TestCase):
    def test_complete_attestation_passes(self):
        self.assertTrue(evaluate(valid_payload())["passed"])

    def test_commit_ros_and_each_gate_are_required(self):
        for mutation in ("commit", "ros", "gate"):
            with self.subTest(mutation=mutation):
                payload = valid_payload()
                if mutation == "commit":
                    payload["git_commit"] = "main"
                elif mutation == "ros":
                    payload["ros"]["distro"] = "humble"
                else:
                    payload["checks"].pop("colcon_tests")
                self.assertFalse(evaluate(payload)["passed"])

    def test_writer_and_independent_validator_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ros_jazzy_ci_evidence.json"
            command = [
                sys.executable,
                str(Path(__file__).with_name("cudanav_ros_ci_evidence.py")),
                "--output",
                str(output),
                "--git-commit",
                "a" * 40,
                "--repository",
                "rsasaki0109/CudaRobotics",
                "--run-id",
                "123",
                "--run-attempt",
                "1",
                "--runner-os",
                "Linux",
                "--runner-arch",
                "X64",
                "--runner-image",
                "ubuntu-24.04",
                "--ros-distro",
                "jazzy",
                "--cuda-toolkit",
                "12.6",
                "--cuda-compiler",
                "Cuda compilation tools, release 12.6, V12.6.85",
            ]
            for package in sorted(REQUIRED_PACKAGES):
                command.extend(["--package", package])
            for check in sorted(REQUIRED_CHECKS):
                command.extend(["--check", check])
            subprocess.run(command, check=True)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(evaluate(payload)["passed"])
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("validate_cudanav_ros_ci.py")),
                    str(output),
                ],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
