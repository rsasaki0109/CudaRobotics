#!/usr/bin/env python3
"""Tests for portable v0.2.0 release-candidate evidence bundles."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from assemble_v0_2_release_bundle import assemble
from test_v0_2_release_evidence import COMMIT, complete_fixture
from v0_2_release_bundle import load_bundle


class V02ReleaseBundleTest(unittest.TestCase):
    def make_bundle(self, root: Path) -> Path:
        inputs = root / "inputs"
        inputs.mkdir()
        fixture = complete_fixture(inputs)
        output = root / "bundle"
        assemble(output_dir=output, **fixture)
        return output / "bundle.json"

    def test_complete_release_bundle_is_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.make_bundle(Path(directory))
            gate = load_bundle(bundle, COMMIT)
            self.assertTrue(gate["valid"], gate)
            self.assertTrue(gate["ready"], gate)
            self.assertTrue(gate["release_gate"]["passed"])

    def test_bundle_remains_valid_after_directory_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self.make_bundle(root)
            moved = root / "moved" / "release"
            moved.parent.mkdir()
            shutil.move(str(bundle.parent), moved)
            gate = load_bundle(moved / "bundle.json", COMMIT)
            self.assertTrue(gate["valid"], gate)

    def test_distribution_edit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.make_bundle(Path(directory))
            wheel = next(bundle.parent.glob("dist/*cp310*.whl"))
            wheel.write_bytes(wheel.read_bytes() + b"tampered")
            gate = load_bundle(bundle, COMMIT)
            self.assertFalse(gate["valid"])
            self.assertFalse(
                gate["file_checks"][wheel.relative_to(bundle.parent).as_posix()]
            )

    def test_undeclared_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.make_bundle(Path(directory))
            (bundle.parent / "undeclared.txt").write_text(
                "not in bundle inventory\n", encoding="utf-8"
            )
            gate = load_bundle(bundle, COMMIT)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["complete_inventory"])

    def test_nested_bundle_named_file_is_not_exempt_from_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.make_bundle(Path(directory))
            nested = bundle.parent / "extra" / "bundle.json"
            nested.parent.mkdir()
            nested.write_text("{}\n", encoding="utf-8")
            gate = load_bundle(bundle, COMMIT)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["complete_inventory"])

    def test_malformed_file_entry_is_invalid_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.make_bundle(Path(directory))
            payload = json.loads(bundle.read_text(encoding="utf-8"))
            payload["files"].append("not an object")
            bundle.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            gate = load_bundle(bundle, COMMIT)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["file_table"])

    def test_release_gate_rewrite_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.make_bundle(Path(directory))
            release_gate = bundle.parent / "release_gate.json"
            payload = json.loads(release_gate.read_text(encoding="utf-8"))
            payload["remote"]["ref"] = "refs/heads/relabelled"
            release_gate.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            gate = load_bundle(bundle, COMMIT)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["release_gate_reference"])

    def test_missing_preflight_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.make_bundle(Path(directory))
            log = next(bundle.parent.glob("evidence/cpu_preflight/logs/*.log"))
            log.unlink()
            gate = load_bundle(bundle, COMMIT)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["complete_inventory"])


if __name__ == "__main__":
    unittest.main()
