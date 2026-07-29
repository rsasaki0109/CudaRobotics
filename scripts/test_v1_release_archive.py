#!/usr/bin/env python3
"""Tests for canonical post-tag v1 release evidence archives."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import zipfile

from assemble_v1_release_bundle import assemble
from test_v1_release_attestation import COMMIT
from test_v1_release_bundle import TAG, VERSION, sources
from v1_release_archive import (
    ARCHIVE_ROOT,
    create_archive,
    load_archive,
    write_checksum,
)


class V1ReleaseArchiveTest(unittest.TestCase):
    def make_bundle(self, root: Path) -> Path:
        output = root / "bundle"
        assemble(
            sources(root),
            output,
            version=VERSION,
            target_tag=TAG,
            git_commit=COMMIT,
        )
        return output / "bundle.json"

    def make_archive(self, root: Path) -> tuple[Path, Path]:
        bundle = self.make_bundle(root)
        archive = root / "cudarobotics-v1.0.0-evidence.zip"
        checksum = root / f"{archive.name}.sha256"
        create_archive(
            bundle,
            archive,
            target_version=VERSION,
            target_tag=TAG,
            expected_commit=COMMIT,
        )
        write_checksum(archive, checksum)
        return archive, checksum

    def load(self, archive: Path, checksum: Path | None = None):
        return load_archive(
            archive,
            target_version=VERSION,
            target_tag=TAG,
            expected_commit=COMMIT,
            checksum_path=checksum,
        )

    def test_archive_and_checksum_are_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, checksum = self.make_archive(Path(directory))
            gate = self.load(archive, checksum)
            self.assertTrue(gate["valid"], gate)
            self.assertTrue(gate["bundle"]["passed"], gate)

    def test_archive_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self.make_bundle(root)
            first = root / "first.zip"
            second = root / "second.zip"
            for archive in (first, second):
                create_archive(
                    bundle,
                    archive,
                    target_version=VERSION,
                    target_tag=TAG,
                    expected_commit=COMMIT,
                )
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, checksum = self.make_archive(root)
            payload = bytearray(archive.read_bytes())
            payload[len(payload) // 2] ^= 0x01
            archive.write_bytes(payload)
            gate = self.load(archive, checksum)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["archive_sha256"])

    def test_path_traversal_is_rejected_without_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(f"{ARCHIVE_ROOT}/../escaped.txt", b"unsafe")
            gate = self.load(archive)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["safe_canonical_members"])
            self.assertFalse((root / "escaped.txt").exists())

    def test_extra_canonical_member_is_rejected_by_bundle_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self.make_bundle(root)
            (bundle.parent / "extra.txt").write_text(
                "undeclared\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "complete_inventory"):
                create_archive(
                    bundle,
                    root / "evidence.zip",
                    target_version=VERSION,
                    target_tag=TAG,
                    expected_commit=COMMIT,
                )

    def test_wrong_release_commit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, _ = self.make_archive(Path(directory))
            gate = load_archive(
                archive,
                target_version=VERSION,
                target_tag=TAG,
                expected_commit="f" * 40,
            )
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["bundle_valid"])

    def test_archive_inside_bundle_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self.make_bundle(root)
            with self.assertRaisesRegex(ValueError, "outside"):
                create_archive(
                    bundle,
                    bundle.parent / "evidence.zip",
                    target_version=VERSION,
                    target_tag=TAG,
                    expected_commit=COMMIT,
                )


if __name__ == "__main__":
    unittest.main()
