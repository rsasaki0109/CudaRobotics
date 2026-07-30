#!/usr/bin/env python3
"""Tests for fail-closed systems-paper artifact bundles."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from assemble_systems_paper_bundle import ROOT, assemble
from paper_artifact_contract import sha256_file, validate_manifest
from systems_paper_bundle import evaluate_bundle, load_bundle

COMMIT = "b" * 40
TITLE = "CudaNav Test: A Reproducible End-to-End GPU Autonomy Stack"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ready_source(root: Path) -> None:
    result = root / "docs/result.json"
    protocol = root / "docs/protocol.md"
    manuscript = root / "paper/cudarobotics_systems_paper.md"
    result.parent.mkdir(parents=True)
    result.write_text('{"status": "passed"}\n', encoding="utf-8")
    protocol.write_text("# Reproduction protocol\n", encoding="utf-8")
    manuscript.parent.mkdir(parents=True)
    manuscript.write_text(
        "\n".join(
            [
                f"# {TITLE}",
                "",
                "[Protocol](../docs/protocol.md)",
                "",
                "| Claim | Status |",
                "|---|---|",
                "| `end_to_end` | Supported |",
                "",
                "The `end_to_end` result is content-bound.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    latex_root = root / "paper/latex"
    latex_root.mkdir(parents=True)
    (latex_root / "cudanav_systems.tex").write_text(
        "\n".join(
            [
                r"\documentclass[conference]{IEEEtran}",
                rf"\title{{{TITLE}}}",
                r"\author{\IEEEauthorblockN{Anonymous Authors}}",
                r"\begin{document}",
                r"\maketitle",
                "A second distinct physical GPU model is an optional " "extension.",
                "It separates recorded-data shadow evidence from "
                "real-robot closed-loop navigation.",
                r"\texttt{end\_to\_end}",
                "Results: 1,059.4 352.748 0.003493 0.815 0.812 " "1,325.5 4.801.",
                r"\bibliography{references}",
                r"\end{document}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (latex_root / "references.bib").write_text(
        "@article{vizzo2023kissicp,title={KISS-ICP}}\n"
        "@article{macenski2020marathon2,title={The Marathon 2}}\n"
        "@article{williams2017mppi,title={MPPI}}\n",
        encoding="utf-8",
    )
    ledger = {
        "schema_version": 1,
        "paper_id": "cudarobotics-systems",
        "title": TITLE,
        "claims": [
            {
                "id": "end_to_end",
                "statement": "The end-to-end gate passes.",
                "submission_required": True,
                "status": "supported",
                "evidence": ["end_to_end_result"],
                "limitations": [],
            }
        ],
        "evidence": [
            {
                "id": "end_to_end_result",
                "status": "complete",
                "kind": "file_set",
                "generator_command": ["python3", "run.py"],
                "artifacts": [
                    {
                        "path": "docs/result.json",
                        "normalization": "text_lf",
                        "sha256": sha256_file(result, "text_lf"),
                    }
                ],
            }
        ],
    }
    write_json(root / "paper/artifacts/cudarobotics_systems.json", ledger)


class SystemsPaperBundleTest(unittest.TestCase):
    def make_bundle(self, root: Path) -> Path:
        source = root / "source"
        ready_source(source)
        output = root / "bundle"
        assemble(source, output, COMMIT, False)
        return output / "submission_manifest.json"

    def test_ready_ledger_forms_portable_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_bundle(Path(directory))
            gate = load_bundle(manifest, COMMIT)
            self.assertTrue(gate["valid"], gate)
            self.assertTrue(gate["ready"], gate)
            self.assertTrue(gate["ledger"]["ready"])

    def test_current_repository_ledger_is_ready(self) -> None:
        ledger = json.loads(
            (ROOT / "paper/artifacts/cudarobotics_systems.json").read_text(
                encoding="utf-8"
            )
        )
        gate = validate_manifest(ledger, ROOT)
        self.assertTrue(gate["valid"], gate)
        self.assertTrue(gate["ready"], gate)

    def test_bundle_remains_valid_after_directory_move(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_bundle(root)
            moved = root / "relocated/artifact"
            moved.parent.mkdir()
            shutil.move(str(manifest.parent), moved)
            gate = load_bundle(moved / "submission_manifest.json", COMMIT)
            self.assertTrue(gate["valid"], gate)
            self.assertTrue(gate["ready"], gate)

    def test_incomplete_fixture_ledger_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            ready_source(source)
            ledger_path = source / "paper/artifacts/cudarobotics_systems.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["claims"][0]["status"] = "partial"
            ledger["claims"][0]["limitations"] = ["External run pending."]
            ledger["evidence"][0] = {
                "id": "end_to_end_result",
                "status": "pending",
                "kind": "file_set",
                "generator_command": ["python3", "run.py"],
            }
            write_json(ledger_path, ledger)
            with self.assertRaisesRegex(ValueError, "ledger is not ready"):
                assemble(source, root / "bundle", COMMIT, False)

    def test_tampered_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_bundle(Path(directory))
            result = manifest.parent / "docs/result.json"
            result.write_text('{"status": "tampered"}\n', encoding="utf-8")
            gate = load_bundle(manifest, COMMIT)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["file_checks"]["docs/result.json"])
            self.assertFalse(gate["checks"]["ledger_valid"])

    def test_nested_manifest_named_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_bundle(Path(directory))
            nested = manifest.parent / "extra/submission_manifest.json"
            nested.parent.mkdir()
            nested.write_text("{}\n", encoding="utf-8")
            gate = load_bundle(manifest, COMMIT)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["complete_inventory"])

    def test_stale_source_manuscript_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            ready_source(source)
            manuscript = source / "paper/cudarobotics_systems_paper.md"
            manuscript.write_text(
                manuscript.read_text(encoding="utf-8")
                + "\nThis is not a submission-ready manuscript.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "final_status"):
                assemble(source, root / "bundle", COMMIT, False)

    def test_tampered_stored_validation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self.make_bundle(Path(directory))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["validation"]["ready"] = False
            gate = evaluate_bundle(manifest, manifest_path.parent, COMMIT)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["validation_record"])

    def test_identity_leak_in_submission_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            ready_source(source)
            latex = source / "paper/latex/cudanav_systems.tex"
            latex.write_text(
                latex.read_text(encoding="utf-8") + "\nContact: author@example.com\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "latex_anonymous"):
                assemble(source, root / "bundle", COMMIT, False)

    def test_cli_rejects_commit_that_is_not_source_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/assemble_systems_paper_bundle.py"),
                    "--output-dir",
                    str(Path(directory) / "bundle"),
                    "--commit",
                    "0" * 40,
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not match source HEAD", result.stderr)


if __name__ == "__main__":
    unittest.main()
