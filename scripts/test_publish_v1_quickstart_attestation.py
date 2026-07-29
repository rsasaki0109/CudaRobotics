#!/usr/bin/env python3
"""Tests for the v1 quickstart release-attestation publisher."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from publish_v1_quickstart_attestation import build_attestation
from test_v1_quickstart_evidence import fixture
from v1_quickstart_evidence import (
    REQUIRED_ARTIFACTS,
    describe_artifacts,
    sha256_file,
)
from v1_release_attestation import validate_payload


COMMIT = "a" * 40


def release_fixture(root: Path) -> None:
    manifest = fixture(root)
    matrix_path = root / "support_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["status"] = "release"
    matrix["surfaces"]["python_source"]["version"] = "1.0.0"
    matrix["surfaces"]["python_wheels"]["version"] = "1.0.0"
    for package in matrix["surfaces"]["ros2"]["package_versions"]:
        matrix["surfaces"]["ros2"]["package_versions"][package] = "1.0.0"
    matrix_path.write_text(
        json.dumps(matrix) + "\n", encoding="utf-8"
    )
    manifest["profile"] = "release"
    manifest["source_ref"] = "v1.0.0"
    manifest["commands"]["clone"][
        manifest["commands"]["clone"].index("release-candidate")
    ] = "v1.0.0"
    manifest["component_versions"] = {
        "python_version": "1.0.0",
        "ros_package_versions": matrix["surfaces"]["ros2"][
            "package_versions"
        ],
    }
    manifest["support_matrix_sha256"] = sha256_file(matrix_path)
    manifest["artifacts"] = describe_artifacts(
        root, set(REQUIRED_ARTIFACTS)
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )


class PublishV1QuickstartAttestationTest(unittest.TestCase):
    def test_release_manifest_builds_valid_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_fixture(root)
            attestation = build_attestation(root)
            gate = validate_payload(
                attestation,
                key="quickstart_15_minute_evidence",
                target_version="1.0.0",
                target_tag="v1.0.0",
            )
            self.assertTrue(gate["passed"], gate)
            self.assertEqual(attestation["git_commit"], COMMIT)
            self.assertEqual(
                attestation["payload_sha256"],
                sha256_file(root / "manifest.json"),
            )
            self.assertEqual(
                attestation["details"]["source_manifest_sha256"],
                attestation["payload_sha256"],
            )

    def test_failed_source_manifest_cannot_be_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_fixture(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest["duration_seconds"] = 901.0
            manifest_path.write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duration"):
                build_attestation(root)

    def test_post_run_artifact_edit_cannot_be_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_fixture(root)
            (root / "docker_run.log").write_text(
                "edited after capture\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "artifact_content"):
                build_attestation(root)


if __name__ == "__main__":
    unittest.main()
