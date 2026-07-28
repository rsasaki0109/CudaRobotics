#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from paper_artifact_contract import sha256_file, validate_manifest


class PaperArtifactContractTest(unittest.TestCase):
    def make_csv_manifest(self, root: Path) -> dict:
        csv_path = root / "result.csv"
        csv_path.write_text(
            "scenario,planner,seed,success\n"
            "contact,mppi,1,0\n"
            "contact,mppi,2,0\n"
            "contact,diff,1,1\n"
            "contact,diff,2,1\n",
            encoding="utf-8",
        )
        return {
            "schema_version": 1,
            "paper_id": "contact-paper",
            "title": "Contact paper",
            "claims": [
                {
                    "id": "contact_success",
                    "statement": "Diff succeeds more often in this fixture.",
                    "submission_required": True,
                    "status": "supported",
                    "evidence": ["contact_csv"],
                    "limitations": ["Two-seed unit fixture only."],
                }
            ],
            "evidence": [
                {
                    "id": "contact_csv",
                    "status": "complete",
                    "kind": "csv_assertions",
                    "generator_command": ["benchmark", "--csv", "result.csv"],
                    "artifacts": [
                        {
                            "path": "result.csv",
                            "sha256": sha256_file(csv_path),
                        }
                    ],
                    "assertions": [
                        {
                            "filters": {"scenario": "contact", "planner": "diff"},
                            "field": "success",
                            "aggregate": "mean",
                            "min_rows": 2,
                            "op": "approx",
                            "value": 1.0,
                            "tolerance": 1e-9,
                        },
                        {
                            "filters": {"scenario": "contact", "planner": "mppi"},
                            "field": "success",
                            "aggregate": "mean",
                            "min_rows": 2,
                            "op": "eq",
                            "value": 0.0,
                        },
                    ],
                }
            ],
        }

    def test_supported_claim_with_bound_csv_is_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = validate_manifest(self.make_csv_manifest(root), root)
            self.assertTrue(result["valid"], result)
            self.assertTrue(result["ready"], result)

    def test_tampered_csv_invalidates_evidence_and_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_csv_manifest(root)
            (root / "result.csv").write_text("tampered\n", encoding="utf-8")
            result = validate_manifest(manifest, root)
            self.assertFalse(result["valid"])
            self.assertFalse(result["ready"])
            self.assertFalse(result["evidence"]["contact_csv"]["valid"])

    def test_partial_claim_with_pending_evidence_is_valid_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_csv_manifest(root)
            manifest["claims"][0]["status"] = "partial"
            manifest["evidence"][0] = {
                "id": "contact_csv",
                "status": "pending",
                "kind": "csv_assertions",
                "generator_command": ["benchmark", "--csv", "result.csv"],
            }
            result = validate_manifest(manifest, root)
            self.assertTrue(result["valid"], result)
            self.assertFalse(result["ready"])

    def test_supported_claim_cannot_use_pending_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_csv_manifest(root)
            manifest["evidence"][0]["status"] = "pending"
            result = validate_manifest(manifest, root)
            self.assertFalse(result["valid"])
            self.assertIn(
                "contact_success: supported/refuted claim lacks complete valid evidence",
                result["errors"],
            )

    def test_json_assertion_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "run.json"
            artifact.write_text(
                json.dumps({"gate": {"passed": True}, "metrics": {"drift": 0.4}}),
                encoding="utf-8",
            )
            manifest = self.make_csv_manifest(root)
            manifest["evidence"][0] = {
                "id": "contact_csv",
                "status": "complete",
                "kind": "json_assertions",
                "generator_command": ["runner", "--output", "run.json"],
                "artifacts": [
                    {
                        "path": "run.json",
                        "normalization": "text_lf",
                        "sha256": sha256_file(artifact, "text_lf"),
                    }
                ],
                "assertions": [
                    {"json_path": "gate.passed", "op": "eq", "value": True},
                    {
                        "json_path": "metrics.drift",
                        "op": "lt",
                        "value": 1.0,
                    },
                ],
            }
            self.assertTrue(validate_manifest(manifest, root)["ready"])
            manifest["evidence"][0]["artifacts"][0]["path"] = "../run.json"
            self.assertFalse(validate_manifest(manifest, root)["valid"])

    def test_text_lf_hash_is_cross_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lines.txt"
            path.write_bytes(b"one\r\ntwo\r\n")
            crlf = sha256_file(path, "text_lf")
            path.write_bytes(b"one\ntwo\n")
            self.assertEqual(crlf, sha256_file(path, "text_lf"))


if __name__ == "__main__":
    unittest.main()
