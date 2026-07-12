#!/usr/bin/env python3
"""CPU-only checks for the dependency-free rosbag2 SQLite analyzer."""

import importlib.util
import sqlite3
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_rosbag_db3.py")
SPEC = importlib.util.spec_from_file_location("analyze_rosbag_db3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "run" / "test.db3"
        db.parent.mkdir()
        connection = sqlite3.connect(db)
        connection.executescript("""
            CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT,
              serialization_format TEXT, offered_qos_profiles TEXT);
            CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER,
              timestamp INTEGER, data BLOB);
            INSERT INTO topics VALUES(1, '/scan', 'sensor_msgs/msg/LaserScan', 'cdr', '');
            INSERT INTO topics VALUES(2, '/unused', 'example/msg/Empty', 'cdr', '');
            INSERT INTO messages VALUES(1, 1, 1000000000, X'0102');
            INSERT INTO messages VALUES(2, 1, 1500000000, X'030405');
            INSERT INTO messages VALUES(3, 1, 2000000000, X'06');
        """)
        connection.commit()
        connection.close()
        report = MODULE.analyze_database(db)
        assert MODULE.find_databases([db.parent]) == [db.resolve()]
    assert report["bag"] == "run"
    assert report["duration_s"] == 1.0
    assert report["messages"] == 3
    assert report["topics"][0]["messages"] == 3
    assert report["topics"][0]["rate_hz"] == 2.0
    assert report["topics"][0]["payload_bytes"] == 6
    assert report["topics"][1]["messages"] == 0
    print("offline rosbag DB3 analyzer checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
