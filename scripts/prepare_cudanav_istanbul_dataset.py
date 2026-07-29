#!/usr/bin/env python3
"""Download or inspect the exact Autoware Istanbul raw ROS 2 bag."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any

from cudanav_real_dataset import DEFAULT_SPEC, read_json
from cudanav_rosbag_evidence import sha256_file


def database_topics(database: Path) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(
        f"file:{database.resolve().as_posix()}?mode=ro", uri=True
    )
    try:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(topics)")
        }
        required_columns = {"id", "name", "type"}
        if not required_columns <= columns:
            raise ValueError("rosbag2 topics table has an unsupported schema")
        rows = connection.execute(
            "SELECT topics.name, topics.type, COUNT(messages.id) "
            "FROM topics LEFT JOIN messages ON messages.topic_id = topics.id "
            "GROUP BY topics.id, topics.name, topics.type "
            "ORDER BY topics.name"
        )
        return {
            name: {"type": message_type, "count": int(count)}
            for name, message_type, count in rows
        }
    finally:
        connection.close()


def find_database(root: Path, expected_name: str) -> Path:
    matches = sorted(path for path in root.rglob(expected_name) if path.is_file())
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {expected_name} under {root}, "
            f"found {len(matches)}"
        )
    return matches[0].resolve()


def inspect(root: Path, spec_path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    spec = read_json(spec_path)
    acquisition = spec["acquisition"]
    database = find_database(root.resolve(), acquisition["expected_database"])
    topics = database_topics(database)
    required_topics = spec["recorded_inputs"]
    checks = {
        name: (
            topics.get(contract["topic"], {}).get("type") == contract["type"]
            and topics.get(contract["topic"], {}).get("count", 0) > 0
        )
        for name, contract in required_topics.items()
    }
    report = {
        "schema_version": 1,
        "dataset_id": spec["dataset_id"],
        "inspected_at": datetime.now(timezone.utc).isoformat(),
        "acquisition": {
            "method": acquisition["method"],
            "file_id": acquisition["file_id"],
            "expected_database": acquisition["expected_database"],
        },
        "database": {
            "source": str(database),
            "bytes": database.stat().st_size,
            "sha256": sha256_file(database),
        },
        "topics": topics,
        "required_topic_checks": checks,
        "passed": all(checks.values()),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/datasets/cudanav_istanbul"),
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = read_json(args.spec)
    acquisition = spec["acquisition"]
    if args.download:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gdown",
                "--id",
                acquisition["file_id"],
                "-O",
                str(output) + "/",
            ],
            check=True,
        )
    database = find_database(output, acquisition["expected_database"])
    metadata = database.parent / "metadata.yaml"
    if args.reindex and not metadata.is_file():
        subprocess.run(
            ["ros2", "bag", "reindex", str(database.parent)], check=True
        )
    report = inspect(output, args.spec)
    report_path = (args.report or output / "inspection.json").resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
