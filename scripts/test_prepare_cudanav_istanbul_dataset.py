#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from cudanav_real_dataset import DEFAULT_SPEC, read_json
from prepare_cudanav_istanbul_dataset import inspect


class PrepareCudaNavIstanbulDatasetTest(unittest.TestCase):
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
