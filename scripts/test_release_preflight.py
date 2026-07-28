#!/usr/bin/env python3
"""Contract tests for the release-preflight command matrix and report."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import run_release_preflight as preflight


def arguments(profile: str, *, with_dist: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        profile=profile,
        build_dir=Path("build"),
        dist_dir=Path("dist") if with_dist else None,
        output_dir=Path("output"),
        require_clean=False,
        dry_run=True,
    )


class ReleasePreflightTests(unittest.TestCase):
    def test_cpu_matrix_keeps_external_gates_out_of_local_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            specs = preflight.check_specs(
                arguments("cpu"), Path(directory)
            )
        names = {spec["name"] for spec in specs}
        self.assertIn("version_consistency", names)
        self.assertIn("python_labelled_ctest", names)
        self.assertNotIn("registration_gpu_consistency", names)
        self.assertNotIn("github_build", names)

    def test_gpu_matrix_adds_registration_and_artifact_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            specs = preflight.check_specs(
                arguments("gpu", with_dist=True), Path(directory)
            )
        names = {spec["name"] for spec in specs}
        self.assertIn("registration_gpu_consistency", names)
        self.assertIn("registration_gpu_smoke", names)
        self.assertIn("python_release_artifacts", names)

    def test_report_labels_external_gates_as_unverified(self) -> None:
        manifest = {
            "status": "passed",
            "profile": "cpu",
            "git_commit": "abc123",
            "git_dirty": False,
            "platform": "test-platform",
            "python": "3.12",
            "checks": [
                {
                    "name": "example",
                    "status": "passed",
                    "elapsed_seconds": 0.1,
                    "log": "build/release/logs/example.log",
                }
            ],
            "external_gates": ["github_build"],
        }
        report = preflight.render_report(manifest)
        self.assertIn("verify on the final release-candidate commit", report)
        self.assertNotIn("github_build`: passed", report)

    def test_report_exposes_a_dirty_checkout_failure(self) -> None:
        manifest = {
            "status": "failed",
            "profile": "gpu",
            "git_commit": "abc123",
            "git_dirty": True,
            "platform": "test-platform",
            "python": "3.12",
            "checks": [
                {
                    "name": "clean_checkout",
                    "status": "failed",
                    "elapsed_seconds": 0.0,
                }
            ],
            "external_gates": [],
        }
        report = preflight.render_report(manifest)
        self.assertIn("| `clean_checkout` | failed |", report)


if __name__ == "__main__":
    unittest.main()
