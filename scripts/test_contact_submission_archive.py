#!/usr/bin/env python3
"""Tests for canonical anonymous contact-paper submission archives."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from assemble_contact_submission_bundle import assemble
from contact_submission_archive import (
    ARCHIVE_ROOT,
    create_archive,
    load_archive,
    write_checksum,
)
from contact_submission_bundle import FORBIDDEN_IDENTITY_TOKENS
from test_contact_submission_bundle import COMMIT


class ContactSubmissionArchiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.bundle = cls.root / "bundle"
        assemble(
            cls.bundle,
            COMMIT,
            "ICRA",
            "https://anonymous.4open.science/r/contact-mppi-0000",
            False,
        )
        cls.manifest = cls.bundle / "submission_manifest.json"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def create(self, name: str) -> tuple[Path, Path]:
        archive = self.root / name
        checksum = self.root / f"{name}.sha256"
        create_archive(self.manifest, archive, COMMIT)
        write_checksum(archive, checksum)
        return archive, checksum

    def test_archive_and_checksum_are_ready_and_anonymous(self) -> None:
        archive, checksum = self.create("ready.zip")
        gate = load_archive(archive, COMMIT, checksum_path=checksum)
        self.assertTrue(gate["valid"], gate)
        self.assertTrue(gate["bundle"]["ready"], gate)
        with zipfile.ZipFile(archive) as package:
            names = "\n".join(package.namelist()).lower().encode("utf-8")
        self.assertFalse(
            any(token in names for token in FORBIDDEN_IDENTITY_TOKENS)
        )

    def test_archive_is_byte_reproducible(self) -> None:
        first, _ = self.create("first.zip")
        second, _ = self.create("second.zip")
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_checksum_mismatch_is_rejected(self) -> None:
        archive, checksum = self.create("tampered.zip")
        payload = bytearray(archive.read_bytes())
        payload[len(payload) // 2] ^= 0x01
        archive.write_bytes(payload)
        gate = load_archive(archive, COMMIT, checksum_path=checksum)
        self.assertFalse(gate["valid"])
        self.assertFalse(gate["checks"]["archive_sha256"])

    def test_path_traversal_is_rejected_without_extraction(self) -> None:
        archive = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(f"{ARCHIVE_ROOT}/../escaped.txt", b"unsafe")
        gate = load_archive(archive, COMMIT)
        self.assertFalse(gate["valid"])
        self.assertFalse(gate["checks"]["safe_canonical_members"])
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_not_ready_manifest_is_refused(self) -> None:
        diagnostic = self.root / "diagnostic"
        shutil.copytree(self.bundle, diagnostic)
        manifest = diagnostic / "submission_manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["venue"] = "unselected"
        manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "venue_selected"):
            create_archive(
                manifest,
                self.root / "diagnostic.zip",
                COMMIT,
            )

    def test_wrong_submission_commit_is_rejected(self) -> None:
        archive, _ = self.create("wrong-commit.zip")
        gate = load_archive(archive, "f" * 40)
        self.assertFalse(gate["valid"])
        self.assertFalse(gate["checks"]["bundle_valid"])

    def test_archive_inside_bundle_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            create_archive(
                self.manifest,
                self.bundle / "submission.zip",
                COMMIT,
            )


if __name__ == "__main__":
    unittest.main()
