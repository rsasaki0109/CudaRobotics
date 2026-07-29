#!/usr/bin/env python3
"""Tests for portable post-tag v1 release evidence bundles."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from assemble_v1_release_bundle import assemble
from test_v1_release_attestation import COMMIT, write_attestation
from v1_release_attestation import MODES
from v1_release_bundle import load_bundle
from v1_support_matrix import evaluate, load


VERSION = "1.0.0"
TAG = "v1.0.0"


def sources(root: Path) -> dict[str, Path]:
    source_root = root / "sources"
    source_root.mkdir()
    result = {}
    for key in MODES:
        _, path = write_attestation(source_root, key)
        result[key] = path
    return result


class V1ReleaseBundleTest(unittest.TestCase):
    def test_four_attestations_form_portable_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "bundle"
            assemble(
                sources(root),
                output,
                version=VERSION,
                target_tag=TAG,
                git_commit=COMMIT,
            )
            gate = load_bundle(
                output / "bundle.json",
                target_version=VERSION,
                target_tag=TAG,
                expected_commit=COMMIT,
            )
            self.assertTrue(gate["passed"], gate)
            result = evaluate(
                load(),
                attestation_root=output,
                readiness_evidence=gate["references"],
                expected_release_commit=COMMIT,
            )
            self.assertTrue(
                result["readiness"]["same_release_commit"], result
            )
            self.assertTrue(
                result["readiness"]["release_commit_binding"], result
            )
            self.assertFalse(result["ready"])

    def test_bundle_rejects_mixed_subject_commits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = sources(root)
            path = inputs["docker_gpu_evidence"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["git_commit"] = "b" * 40
            path.write_text(
                json.dumps(payload) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "same_subject_commit"):
                assemble(
                    inputs,
                    root / "bundle",
                    version=VERSION,
                    target_tag=TAG,
                    git_commit=COMMIT,
                )

    def test_post_bundle_edit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "bundle"
            assemble(
                sources(root),
                output,
                version=VERSION,
                target_tag=TAG,
                git_commit=COMMIT,
            )
            (output / "quickstart.json").write_text(
                "{}\n", encoding="utf-8"
            )
            gate = load_bundle(
                output / "bundle.json",
                target_version=VERSION,
                target_tag=TAG,
                expected_commit=COMMIT,
            )
            self.assertFalse(gate["checks"]["all_attestations"])
            self.assertFalse(gate["passed"])

    def test_undeclared_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "bundle"
            assemble(
                sources(root),
                output,
                version=VERSION,
                target_tag=TAG,
                git_commit=COMMIT,
            )
            (output / "undeclared.txt").write_text(
                "not attested\n",
                encoding="utf-8",
            )
            gate = load_bundle(
                output / "bundle.json",
                target_version=VERSION,
                target_tag=TAG,
                expected_commit=COMMIT,
            )
            self.assertFalse(gate["checks"]["complete_inventory"])
            self.assertFalse(gate["passed"])

    def test_noncanonical_attestation_filename_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "bundle"
            assemble(
                sources(root),
                output,
                version=VERSION,
                target_tag=TAG,
                git_commit=COMMIT,
            )
            bundle_path = output / "bundle.json"
            payload = json.loads(bundle_path.read_text(encoding="utf-8"))
            reference = payload["attestations"]["documentation_deployment"]
            source = output / reference["path"]
            renamed = output / "renamed.json"
            source.rename(renamed)
            reference["path"] = renamed.name
            bundle_path.write_text(
                json.dumps(payload) + "\n",
                encoding="utf-8",
            )
            gate = load_bundle(
                bundle_path,
                target_version=VERSION,
                target_tag=TAG,
                expected_commit=COMMIT,
            )
            self.assertFalse(gate["checks"]["canonical_filenames"])
            self.assertFalse(gate["checks"]["complete_inventory"])
            self.assertFalse(gate["passed"])


if __name__ == "__main__":
    unittest.main()
