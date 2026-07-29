#!/usr/bin/env python3
"""Download or inspect the exact Autoware Istanbul raw ROS 2 bag."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.message import Message
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any
from urllib.request import Request, urlopen

from cudanav_real_dataset import DEFAULT_SPEC, read_json
from cudanav_rosbag_evidence import sha256_file


def remote_file_metadata(file_id: str) -> dict[str, Any]:
    url = (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )
    request = Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        disposition = response.headers.get("Content-Disposition", "")
        message = Message()
        message["Content-Disposition"] = disposition
        filename = message.get_filename()
        length = response.headers.get("Content-Length")
    if not filename or length is None:
        raise ValueError(f"Drive probe omitted filename or length: {file_id}")
    return {
        "file_id": file_id,
        "filename": filename,
        "bytes": int(length),
        "url": url,
    }


def probe_acquisition(acquisition: dict[str, Any]) -> dict[str, Any]:
    database = remote_file_metadata(acquisition["file_id"])
    checks = {
        "database_filename": (
            database["filename"] == acquisition["expected_database"]
        ),
        "database_bytes": (
            database["bytes"] == acquisition["expected_database_bytes"]
        ),
    }
    result = {
        "schema_version": 1,
        "database": database,
        "checks": checks,
    }
    if "metadata_file_id" in acquisition:
        metadata = remote_file_metadata(acquisition["metadata_file_id"])
        checks.update(
            {
                "metadata_filename": (
                    metadata["filename"] == acquisition["expected_metadata"]
                ),
                "metadata_bytes": (
                    metadata["bytes"]
                    == acquisition["expected_metadata_bytes"]
                ),
            }
        )
        result["metadata"] = metadata
    result["passed"] = all(checks.values())
    if not result["passed"]:
        raise ValueError(f"remote acquisition contract changed: {result}")
    return result


def reusable_remote_probe(
    report_path: Path,
    spec_path: Path = DEFAULT_SPEC,
) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    try:
        previous = read_json(report_path)
        spec_path = spec_path.resolve()
        acquisition = read_json(spec_path)["acquisition"]
        probe = previous["remote_probe"]
        expected_checks = {
            "database_filename": True,
            "database_bytes": True,
        }
        database = probe["database"]
        valid = (
            previous.get("dataset_spec", {}).get("sha256")
            == sha256_file(spec_path)
            and probe.get("schema_version") == 1
            and probe.get("passed") is True
            and database.get("file_id") == acquisition["file_id"]
            and database.get("filename") == acquisition["expected_database"]
            and database.get("bytes")
            == acquisition["expected_database_bytes"]
        )
        if "metadata_file_id" in acquisition:
            expected_checks.update(
                {
                    "metadata_filename": True,
                    "metadata_bytes": True,
                }
            )
            metadata = probe["metadata"]
            valid = (
                valid
                and metadata.get("file_id")
                == acquisition["metadata_file_id"]
                and metadata.get("filename")
                == acquisition["expected_metadata"]
                and metadata.get("bytes")
                == acquisition["expected_metadata_bytes"]
            )
        if not valid or probe.get("checks") != expected_checks:
            return None
        reused = dict(probe)
        reused["reused_from_inspection"] = {
            "sha256": sha256_file(report_path),
            "inspected_at": previous.get("inspected_at"),
        }
        return reused
    except (KeyError, OSError, TypeError, ValueError):
        return None


def download_command(
    file_id: str, output: Path, backend: str = "curl"
) -> list[str]:
    if backend == "curl":
        return [
            "curl",
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--retry",
            "20",
            "--retry-all-errors",
            "--output",
            str(output),
            (
                "https://drive.usercontent.google.com/download"
                f"?id={file_id}&export=download&confirm=t"
            ),
        ]
    if backend != "gdown":
        raise ValueError(f"unsupported download backend: {backend}")
    return [
        sys.executable,
        "-m",
        "gdown",
        file_id,
        "-O",
        str(output),
    ]


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


def write_metadata(database: Path, output: Path) -> dict[str, Any]:
    connection = sqlite3.connect(
        f"file:{database.resolve().as_posix()}?mode=ro", uri=True
    )
    try:
        bounds = connection.execute(
            "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM messages"
        ).fetchone()
        topics = database_topics(database)
    finally:
        connection.close()
    start, end, total = (int(value) for value in bounds)
    duration = max(0, end - start)
    lines = [
        "rosbag2_bagfile_information:",
        "  version: 5",
        "  storage_identifier: sqlite3",
        "  duration:",
        f"    nanoseconds: {duration}",
        "  starting_time:",
        f"    nanoseconds_since_epoch: {start}",
        f"  message_count: {total}",
        "  topics_with_message_count:",
    ]
    for name, entry in topics.items():
        lines.extend(
            [
                "    - topic_metadata:",
                f"        name: {name}",
                f"        type: {entry['type']}",
                "        serialization_format: cdr",
                '        offered_qos_profiles: ""',
                f"      message_count: {entry['count']}",
            ]
        )
    lines.extend(
        [
            '  compression_format: ""',
            '  compression_mode: ""',
            "  relative_file_paths:",
            f"    - {database.name}",
            "  files:",
            f"    - path: {database.name}",
            "      starting_time:",
            f"        nanoseconds_since_epoch: {start}",
            "      duration:",
            f"        nanoseconds: {duration}",
            f"      message_count: {total}",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "algorithm": "cudarobotics.sqlite_rosbag_metadata.v1",
        "database": str(database.resolve()),
        "database_sha256": sha256_file(database),
        "metadata": str(output.resolve()),
        "metadata_sha256": sha256_file(output),
        "starting_time_ns": start,
        "duration_ns": duration,
        "message_count": total,
        "topic_count": len(topics),
    }


def find_database(root: Path, expected_name: str) -> Path:
    matches = sorted(path for path in root.rglob(expected_name) if path.is_file())
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {expected_name} under {root}, "
            f"found {len(matches)}"
        )
    return matches[0].resolve()


def inspect(
    root: Path,
    spec_path: Path = DEFAULT_SPEC,
    remote_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        "dataset_spec": {
            "path": str(spec_path),
            "sha256": sha256_file(spec_path),
        },
        "acquisition": {
            "method": acquisition["method"],
            "file_id": acquisition["file_id"],
            "expected_database": acquisition["expected_database"],
            "expected_database_bytes": acquisition["expected_database_bytes"],
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
    expected_database_sha = acquisition.get("expected_database_sha256")
    if expected_database_sha is not None:
        database_contract_checks = {
            "database_bytes": (
                report["database"]["bytes"]
                == acquisition["expected_database_bytes"]
            ),
            "database_sha256": (
                report["database"]["sha256"] == expected_database_sha
            ),
        }
        report["database_contract_checks"] = database_contract_checks
        report["passed"] = report["passed"] and all(
            database_contract_checks.values()
        )
    if remote_probe is not None:
        report["remote_probe"] = remote_probe
    for key in (
        "metadata_file_id",
        "expected_metadata",
        "expected_database_sha256",
    ):
        if key in acquisition:
            report["acquisition"][key] = acquisition[key]
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
    parser.add_argument(
        "--download-backend",
        choices=("curl", "gdown"),
        default="curl",
    )
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--generate-metadata", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    spec = read_json(args.spec)
    acquisition = spec["acquisition"]
    output = args.output_dir.resolve()
    report_path = (args.report or output / "inspection.json").resolve()
    remote_probe = None
    if args.download or args.probe or args.probe_only:
        remote_probe = probe_acquisition(acquisition)
    else:
        remote_probe = reusable_remote_probe(report_path, args.spec)
    if args.probe_only:
        print(json.dumps(remote_probe, indent=2, sort_keys=True))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if args.download:
        downloads = [
            (acquisition["file_id"], acquisition["expected_database"])
        ]
        if "metadata_file_id" in acquisition:
            downloads.append(
                (
                    acquisition["metadata_file_id"],
                    acquisition["expected_metadata"],
                )
            )
        for file_id, filename in downloads:
            subprocess.run(
                download_command(
                    file_id,
                    output / filename,
                    args.download_backend,
                ),
                check=True,
            )
    database = find_database(output, acquisition["expected_database"])
    metadata = database.parent / "metadata.yaml"
    metadata_report = None
    if args.generate_metadata:
        metadata_report = write_metadata(database, metadata)
    if args.reindex and not metadata.is_file():
        subprocess.run(
            ["ros2", "bag", "reindex", str(database.parent)], check=True
        )
    report = inspect(output, args.spec, remote_probe)
    if metadata_report is not None:
        report["generated_metadata"] = metadata_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
