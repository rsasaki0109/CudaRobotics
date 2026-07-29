#!/usr/bin/env python3
"""Tests for canonical CudaRobotics systems-paper archives."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import zipfile

from assemble_systems_paper_bundle import assemble
from systems_paper_archive import (
    ARCHIVE_ROOT,
    create_archive,
    load_archive,
    write_checksum,
)
from test_systems_paper_bundle import COMMIT, ready_source


class SystemsPaperArchiveTest(unittest.TestCase):
    def make_bundle(self, root: Path) -> Path:
        source = root / "source"
        ready_source(source)
        output = root / "bundle"
        assemble(source, output, COMMIT, False)
        return output / "submission_manifest.json"

    def make_archive(self, root: Path) -> tuple[Path, Path]:
        manifest = self.make_bundle(root)
        archive = root / "cudarobotics-systems-paper-artifact.zip"
        checksum = root / f"{archive.name}.sha256"
        create_archive(manifest, archive, COMMIT)
        write_checksum(archive, checksum)
        return archive, checksum

    def test_archive_and_checksum_are_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive, checksum = self.make_archive(Path(directory))
            gate = load_archive(archive, COMMIT, checksum_path=checksum)
            self.assertTrue(gate["valid"], gate)
            self.assertTrue(gate["bundle"]["ready"], gate)

    def test_archive_is_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_bundle(root)
            first = root / "first.zip"
            second = root / "second.zip"
            create_archive(manifest, first, COMMIT)
            create_archive(manifest, second, COMMIT)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_checksum_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive, checksum = self.make_archive(Path(directory))
            payload = bytearray(archive.read_bytes())
            payload[len(payload) // 2] ^= 0x01
            archive.write_bytes(payload)
            gate = load_archive(archive, COMMIT, checksum_path=checksum)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["archive_sha256"])

    def test_path_traversal_is_rejected_without_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(f"{ARCHIVE_ROOT}/../escaped.txt", b"unsafe")
            gate = load_archive(archive, COMMIT)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["safe_canonical_members"])
            self.assertFalse((root / "escaped.txt").exists())

    def test_wrong_source_commit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive, _ = self.make_archive(Path(directory))
            gate = load_archive(archive, "f" * 40)
            self.assertFalse(gate["valid"])
            self.assertFalse(gate["checks"]["bundle_valid"])

    def test_archive_inside_bundle_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_bundle(root)
            with self.assertRaisesRegex(ValueError, "outside"):
                create_archive(
                    manifest,
                    manifest.parent / "artifact.zip",
                    COMMIT,
                )


if __name__ == "__main__":
    unittest.main()
