#!/usr/bin/env python3
"""Summarize rosbag2 SQLite files without requiring a ROS installation."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


def find_databases(inputs: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for item in inputs:
        if item.is_file() and item.suffix == ".db3":
            found.add(item.resolve())
        elif item.is_dir():
            found.update(path.resolve() for path in item.rglob("*.db3"))
    return sorted(found)


def analyze_database(path: Path) -> dict[str, object]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        topics = connection.execute(
            "SELECT id, name, type, serialization_format FROM topics ORDER BY id"
        ).fetchall()
        aggregates = {
            row[0]: row[1:]
            for row in connection.execute(
                "SELECT topic_id, COUNT(*), MIN(timestamp), MAX(timestamp), "
                "COALESCE(SUM(LENGTH(data)), 0) FROM messages GROUP BY topic_id"
            )
        }
        bounds = connection.execute(
            "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM messages"
        ).fetchone()
    finally:
        connection.close()

    start_ns, end_ns, message_count = bounds
    duration_s = ((end_ns - start_ns) / 1e9) if start_ns is not None and end_ns is not None else 0.0
    topic_rows = []
    for topic_id, name, msg_type, serialization in topics:
        count, first_ns, last_ns, payload_bytes = aggregates.get(topic_id, (0, None, None, 0))
        span_s = ((last_ns - first_ns) / 1e9) if count > 1 else 0.0
        rate_hz = ((count - 1) / span_s) if span_s > 0 else 0.0
        coverage = (span_s / duration_s) if duration_s > 0 else 0.0
        topic_rows.append({
            "name": name, "type": msg_type, "serialization": serialization,
            "messages": count, "first_timestamp_ns": first_ns,
            "last_timestamp_ns": last_ns, "span_s": round(span_s, 6),
            "rate_hz": round(rate_hz, 3), "coverage_ratio": round(coverage, 4),
            "payload_bytes": payload_bytes,
        })
    return {
        "bag": path.parent.name, "database": str(path), "database_bytes": path.stat().st_size,
        "start_timestamp_ns": start_ns, "end_timestamp_ns": end_ns,
        "duration_s": round(duration_s, 6), "messages": message_count,
        "topics": topic_rows,
    }


def write_csv(reports: list[dict[str, object]], path: Path) -> None:
    fields = ["bag", "topic", "type", "messages", "rate_hz", "span_s",
              "coverage_ratio", "payload_bytes", "database"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for topic in report["topics"]:
                writer.writerow({
                    "bag": report["bag"], "topic": topic["name"], "type": topic["type"],
                    "messages": topic["messages"], "rate_hz": topic["rate_hz"],
                    "span_s": topic["span_s"], "coverage_ratio": topic["coverage_ratio"],
                    "payload_bytes": topic["payload_bytes"], "database": report["database"],
                })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="DB3 files or directories to scan.")
    parser.add_argument("--json", type=Path, default=Path("build/rosbag_offline_summary.json"))
    parser.add_argument("--csv", type=Path, default=Path("build/rosbag_offline_topics.csv"))
    args = parser.parse_args()
    databases = find_databases(args.inputs)
    if not databases:
        parser.error("no .db3 files found")
    reports = [analyze_database(path) for path in databases]
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"bags": reports}, indent=2) + "\n", encoding="utf-8")
    write_csv(reports, args.csv)
    for report in reports:
        active = sum(1 for topic in report["topics"] if topic["messages"])
        print(f"{report['bag']}: {report['duration_s']:.1f}s, {report['messages']} messages, {active} active topics")
    print(f"wrote {args.json}")
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
