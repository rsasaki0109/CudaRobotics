#!/usr/bin/env python3
"""Contract tests for the anonymous contact-paper submission bundle."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from assemble_contact_submission_bundle import assemble
from contact_submission_bundle import evaluate_bundle, load_bundle


COMMIT = "a" * 40


class ContactSubmissionBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "bundle"
        cls.manifest = assemble(
            cls.output,
            COMMIT,
            "ICRA",
            "https://anonymous.4open.science/r/contact-mppi-0000",
            False,
        )
        cls.manifest_path = cls.output / "submission_manifest.json"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_bundle_is_portable_anonymous_and_ready(self) -> None:
        gate = load_bundle(self.manifest_path, COMMIT)
        self.assertTrue(gate["valid"], gate)
        self.assertTrue(gate["ready"], gate)
        self.assertTrue(gate["ledger"]["ready"])
        self.assertTrue(self.manifest["redactions"])
        self.assertTrue(
            all(item["replacements"] > 0 for item in self.manifest["redactions"])
        )

    def test_figures_are_derived_from_frozen_published_csvs(self) -> None:
        figure_manifest = json.loads(
            (
                self.output
                / "paper/figures/submission/figure_manifest.json"
            ).read_text(encoding="utf-8")
        )
        robustness = figure_manifest["figures"]["robustness"]
        self.assertEqual(robustness["selected_comparisons"], 180)
        self.assertEqual(
            sum(row["positive"] for row in robustness["conditions"]), 24
        )
        self.assertEqual(
            sum(row["negative"] for row in robustness["conditions"]), 6
        )
        matched = figure_manifest["figures"]["matched_compute"]["rows"]
        contact_loss = {
            row["planner"]: row
            for row in matched
            if row["scenario"] == "box_align_contact_loss"
        }
        self.assertAlmostEqual(contact_loss["mppi"]["success_rate"], 14 / 30)
        self.assertEqual(contact_loss["diff_mppi_3"]["success_rate"], 1.0)
        self.assertEqual(contact_loss["diff_mppi_3"]["deadline_misses"], 0)
        external = [
            row
            for row in figure_manifest["figures"]["external_fidelity"]["rows"]
            if row["planner"] == "diff_mppi_3"
        ]
        self.assertEqual(sum(row["episodes"] for row in external), 1050)
        self.assertEqual(sum(row["successes"] for row in external), 480)
        self.assertAlmostEqual(480 / 1050, 0.45714285714285713)

    def test_tampered_manuscript_is_rejected(self) -> None:
        manuscript = self.output / "paper/diff_mppi_submission_draft.md"
        original = manuscript.read_bytes()
        try:
            manuscript.write_bytes(original + b"\ntampered\n")
            gate = load_bundle(self.manifest_path, COMMIT)
            self.assertFalse(gate["valid"])
            self.assertFalse(
                gate["file_checks"]["paper/diff_mppi_submission_draft.md"]
            )
        finally:
            manuscript.write_bytes(original)

    def test_editorial_placeholders_are_valid_but_not_ready(self) -> None:
        diagnostic = dict(self.manifest)
        diagnostic["venue"] = "unselected"
        diagnostic["artifact_url"] = ""
        diagnostic["git_dirty"] = True
        gate = evaluate_bundle(diagnostic, self.output, COMMIT)
        self.assertTrue(gate["valid"])
        self.assertFalse(gate["ready"])
        self.assertFalse(gate["readiness_checks"]["venue_selected"])
        self.assertFalse(gate["readiness_checks"]["artifact_url"])
        self.assertFalse(gate["readiness_checks"]["clean_commit"])

    def test_undeclared_or_identifying_file_is_rejected(self) -> None:
        leak = self.output / "author-email.txt"
        try:
            leak.write_text("author@example.net\n", encoding="utf-8")
            gate = load_bundle(self.manifest_path, COMMIT)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["complete_inventory"])
        finally:
            leak.unlink()

    def test_identifying_artifact_url_is_not_ready(self) -> None:
        diagnostic = dict(self.manifest)
        diagnostic["artifact_url"] = "https://example.net/rsasaki0109/artifact"
        gate = evaluate_bundle(diagnostic, self.output, COMMIT)
        self.assertTrue(gate["valid"])
        self.assertFalse(gate["ready"])
        self.assertFalse(gate["readiness_checks"]["artifact_url"])


if __name__ == "__main__":
    unittest.main()
