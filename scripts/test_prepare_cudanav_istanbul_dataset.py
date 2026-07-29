#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from cudanav_real_dataset import DEFAULT_SPEC, read_json
from prepare_cudanav_istanbul_dataset import (
    download_command,
    inspect,
    probe_acquisition,
    write_metadata,
)


class PrepareCudaNavIstanbulDatasetTest(unittest.TestCase):
    def test_download_commands_are_resumable_and_gdown_v6_compatible(
        self,
    ) -> None:
        output = Path("dataset.db3")
        curl = download_command("drive-id", output)
        self.assertEqual(curl[0], "curl")
        self.assertIn("--continue-at", curl)
        self.assertEqual(curl[curl.index("--continue-at") + 1], "-")
        command = download_command("drive-id", output, "gdown")
        self.assertEqual(command[3], "drive-id")
        self.assertNotIn("--id", command)
        self.assertEqual(command[-2:], ["-O", str(output)])

    def test_remote_probe_freezes_current_official_folder_files(self) -> None:
        acquisition = read_json(DEFAULT_SPEC)["acquisition"]

        def metadata(file_id: str) -> dict:
            if file_id == acquisition["file_id"]:
                return {
                    "file_id": file_id,
                    "filename": acquisition["expected_database"],
                    "bytes": acquisition["expected_database_bytes"],
                    "url": "https://example.invalid/database",
                }
            return {
                "file_id": file_id,
                "filename": acquisition["expected_metadata"],
                "bytes": acquisition["expected_metadata_bytes"],
                "url": "https://example.invalid/metadata",
            }

        with patch(
            "prepare_cudanav_istanbul_dataset.remote_file_metadata",
            side_effect=metadata,
        ):
            result = probe_acquisition(acquisition)
        self.assertTrue(result["passed"], result)
        self.assertTrue(all(result["checks"].values()))

    def test_remote_probe_rejects_replaced_database(self) -> None:
        acquisition = read_json(DEFAULT_SPEC)["acquisition"]

        def metadata(file_id: str) -> dict:
            if file_id == acquisition["file_id"]:
                return {
                    "file_id": file_id,
                    "filename": "replacement.db3",
                    "bytes": acquisition["expected_database_bytes"],
                    "url": "https://example.invalid/database",
                }
            return {
                "file_id": file_id,
                "filename": acquisition["expected_metadata"],
                "bytes": acquisition["expected_metadata_bytes"],
                "url": "https://example.invalid/metadata",
            }

        with patch(
            "prepare_cudanav_istanbul_dataset.remote_file_metadata",
            side_effect=metadata,
        ):
            with self.assertRaises(ValueError):
                probe_acquisition(acquisition)

    def test_exact_database_topics_and_content_are_frozen(self) -> None:
        spec = read_json(DEFAULT_SPEC)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / spec["acquisition"]["expected_database"]
            connection = sqlite3.connect(database)
            connection.executescript(
                "CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT);"
                "CREATE TABLE messages("
                "id INTEGER PRIMARY KEY, topic_id INTEGER, "
                "timestamp INTEGER, data BLOB);"
            )
            for index, contract in enumerate(
                spec["recorded_inputs"].values(), start=1
            ):
                connection.execute(
                    "INSERT INTO topics VALUES(?, ?, ?)",
                    (index, contract["topic"], contract["type"]),
                )
                connection.execute(
                    "INSERT INTO messages(topic_id, timestamp, data) "
                    "VALUES(?, ?, ?)",
                    (index, index, b"fixture"),
                )
            connection.commit()
            connection.close()
            report = inspect(root)
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["database"]["source"], str(database.resolve()))
            self.assertEqual(len(report["database"]["sha256"]), 64)
            metadata = root / "metadata.yaml"
            generated = write_metadata(database, metadata)
            self.assertEqual(generated["message_count"], 3)
            self.assertEqual(generated["topic_count"], 3)
            self.assertIn(
                "storage_identifier: sqlite3",
                metadata.read_text(encoding="utf-8"),
            )

    def test_missing_required_messages_fail(self) -> None:
        spec = read_json(DEFAULT_SPEC)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / spec["acquisition"]["expected_database"]
            connection = sqlite3.connect(database)
            connection.executescript(
                "CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT);"
                "CREATE TABLE messages("
                "id INTEGER PRIMARY KEY, topic_id INTEGER, "
                "timestamp INTEGER, data BLOB);"
            )
            for index, contract in enumerate(
                spec["recorded_inputs"].values(), start=1
            ):
                connection.execute(
                    "INSERT INTO topics VALUES(?, ?, ?)",
                    (index, contract["topic"], contract["type"]),
                )
            connection.commit()
            connection.close()
            self.assertFalse(inspect(root)["passed"])


if __name__ == "__main__":
    unittest.main()
