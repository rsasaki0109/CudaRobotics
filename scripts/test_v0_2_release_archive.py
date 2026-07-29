#!/usr/bin/env python3
"""Tests for canonical v0.2.0 release evidence archives."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import zipfile

from assemble_v0_2_release_bundle import assemble
from test_v0_2_release_evidence import COMMIT, complete_fixture
from v0_2_release_archive import (
    ARCHIVE_ROOT,
    create_archive,
    load_archive,
    write_checksum,
)


class V02ReleaseArchiveTest(unittest.TestCase):
    def make_bundle(self, root: Path) -> Path:
        inputs = root / "inputs"
        inputs.mkdir()
        output = root / "bundle"
        assemble(
            output_dir=output,
            **complete_fixture(inputs),
        )
        return output / "bundle.json"

    def make_archive(self, root: Path) -> tuple[Path, Path]:
        bundle = self.make_bundle(root)
        archive = root / "cudarobotics-0.2.0-evidence.zip"
        checksum = root / f"{archive.name}.sha256"
        create_archive(bundle, archive, COMMIT)
        write_checksum(archive, checksum)
        return archive, checksum

    def test_archive_and_checksum_are_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, checksum = self.make_archive(Path(directory))
            gate = load_archive(
                archive,
                COMMIT,
                checksum_path=checksum,
            )
            self.assertTrue(gate["valid"], gate)
            self.assertTrue(gate["bundle"]["valid"], gate)

    def test_archive_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self.make_bundle(root)
            first = root / "first.zip"
            second = root / "second.zip"
            create_archive(bundle, first, COMMIT)
            create_archive(bundle, second, COMMIT)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_archive_payload_edit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, checksum = self.make_archive(root)
            payload = bytearray(archive.read_bytes())
            payload[len(payload) // 2] ^= 0x01
            archive.write_bytes(payload)
            gate = load_archive(
                archive,
                COMMIT,
                checksum_path=checksum,
            )
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["archive_sha256"])

    def test_path_traversal_member_is_rejected_without_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(f"{ARCHIVE_ROOT}/../escaped.txt", b"unsafe")
            gate = load_archive(archive, COMMIT)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["safe_canonical_members"])
            self.assertFalse((root / "escaped.txt").exists())

    def test_duplicate_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "duplicate.zip"
            name = f"{ARCHIVE_ROOT}/bundle.json"
            with self.assertWarns(UserWarning):
                with zipfile.ZipFile(archive, "w") as output:
                    output.writestr(name, b"{}")
                    output.writestr(name, b"{}")
            gate = load_archive(archive, COMMIT)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["member_table"])

    def test_noncanonical_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, checksum = self.make_archive(root)
            checksum.write_text(
                f"{'0' * 64} *{archive.name}\n",
                encoding="ascii",
            )
            gate = load_archive(
                archive,
                COMMIT,
                checksum_path=checksum,
            )
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["checksum_readable"])

    def test_size_limit_is_checked_before_payload_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, _ = self.make_archive(Path(directory))
            gate = load_archive(
                archive,
                COMMIT,
                max_total_bytes=1,
            )
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["size_limit"])
            self.assertFalse(gate["checks"]["payload_crc"])

    def test_archive_inside_bundle_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self.make_bundle(root)
            with self.assertRaisesRegex(ValueError, "outside"):
                create_archive(
                    bundle,
                    bundle.parent / "evidence.zip",
                    COMMIT,
                )


if __name__ == "__main__":
    unittest.main()
