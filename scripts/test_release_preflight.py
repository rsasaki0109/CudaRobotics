#!/usr/bin/env python3
"""Contract tests for the release-preflight command matrix and report."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from release_preflight_evidence import (
    CPU_REQUIRED_CHECKS,
    GPU_REQUIRED_CHECKS,
    collect_evidence_files,
    evaluate_manifest,
)
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


def evidence_fixture(root: Path, profile: str = "cpu") -> dict:
    required = (
        GPU_REQUIRED_CHECKS if profile == "gpu" else CPU_REQUIRED_CHECKS
    )
    checks = []
    for name in sorted(required):
        check = {"name": name, "status": "passed", "returncode": 0}
        if name != "clean_checkout":
            relative = f"logs/{name}.log"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{name} passed\n", encoding="utf-8")
            check["report_log"] = relative
        checks.append(check)
    (root / "python_artifacts.json").write_text(
        json.dumps({"status": "passed"}) + "\n", encoding="utf-8"
    )
    if profile == "gpu":
        (root / "registration_smoke.csv").write_text(
            "status\npassed\n", encoding="utf-8"
        )
        (root / "registration_smoke.md").write_text(
            "# Passed\n", encoding="utf-8"
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "profile": profile,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "checks": checks,
        "external_gates": [
            "github_build",
            "python_manylinux_wheels",
            "ros2_cuda_mppi",
            "closed_loop_rosbag_or_explicit_negative_result",
        ],
        "evidence_files": collect_evidence_files(root, checks),
    }


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

    def test_content_bound_cpu_evidence_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = evaluate_manifest(
                evidence_fixture(root),
                root,
                expected_profile="cpu",
                expected_commit="a" * 40,
            )
            self.assertTrue(result["passed"], result)

    def test_modified_log_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = evidence_fixture(root)
            (root / "logs/version_consistency.log").write_text(
                "replacement\n", encoding="utf-8"
            )
            result = evaluate_manifest(manifest, root)
            self.assertFalse(
                result["checks"]["evidence_content_unchanged"]
            )
            self.assertFalse(result["passed"])

    def test_generated_artifact_cannot_be_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = evidence_fixture(root)
            manifest["evidence_files"] = [
                entry
                for entry in manifest["evidence_files"]
                if entry["path"] != "python_artifacts.json"
            ]
            result = evaluate_manifest(manifest, root)
            self.assertFalse(result["checks"]["evidence_complete"])
            self.assertFalse(result["passed"])

    def test_gpu_profile_requires_registration_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = evidence_fixture(root)
            manifest["profile"] = "gpu"
            result = evaluate_manifest(manifest, root)
            self.assertFalse(result["checks"]["required_checks"])
            self.assertFalse(result["passed"])

    def test_evidence_path_cannot_escape_preflight_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "preflight"
            root.mkdir()
            manifest = evidence_fixture(root)
            outside = root.parent / "outside.log"
            outside.write_text("outside\n", encoding="utf-8")
            manifest["evidence_files"][0] = {
                "path": "../outside.log",
                "bytes": outside.stat().st_size,
                "sha256": "0" * 64,
            }
            result = evaluate_manifest(manifest, root)
            self.assertFalse(
                result["checks"]["evidence_content_unchanged"]
            )
            self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
