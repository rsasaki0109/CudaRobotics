#!/usr/bin/env python3
"""Download and inspect the public ERL-inspired ROS 2 navigation bags."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = "https://zenodo.org/records/10518775/files/ERL_Test_Rosbags.zip?download=1"
ARCHIVE_NAME = "ERL_Test_Rosbags.zip"
ARCHIVE_MD5 = "1fe7a936441723005eb958d649d1836f"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request) as response:
        append = offset > 0 and response.status == 206
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        total = response.headers.get("Content-Length")
        expected = offset + int(total) if total else None
        with partial.open(mode) as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                current = output.tell()
                if expected:
                    print(f"\rdownloaded {current / 1e9:.2f}/{expected / 1e9:.2f} GB", end="", flush=True)
    print()
    partial.replace(destination)


def remote_archive():
    try:
        from remotezip import RemoteZip
    except ImportError as exc:
        raise SystemExit(
            "selective remote access requires: python -m pip install remotezip"
        ) from exc
    return RemoteZip(DATASET_URL)


def list_remote_members() -> list[dict[str, object]]:
    with remote_archive() as bundle:
        return [
            {
                "name": item.filename,
                "size": item.file_size,
                "compressed_size": item.compress_size,
            }
            for item in bundle.infolist()
            if not item.is_dir()
        ]


def extract_remote_member(member: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / Path(member).name
    partial = output.with_suffix(output.suffix + ".part")
    with remote_archive() as bundle:
        names = {item.filename for item in bundle.infolist()}
        if member not in names:
            raise SystemExit(f"remote ZIP member not found: {member}")
        with bundle.open(member) as source, partial.open("wb") as target:
            copied = 0
            while chunk := source.read(8 * 1024 * 1024):
                target.write(chunk)
                copied += len(chunk)
                print(f"\rextracted {copied / 1e9:.2f} GB", end="", flush=True)
    print()
    partial.replace(output)
    return output


def topics_from_metadata(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    names = re.findall(r"^\s*name:\s*['\"]?([^'\"\s]+)", text, re.MULTILINE)
    types = re.findall(r"^\s*type:\s*['\"]?([^'\"\s]+)", text, re.MULTILINE)
    return dict(zip(names, types))


def topic_matching(topics: dict[str, str], *, names: tuple[str, ...] = (), type_suffix: str = "") -> list[str]:
    return sorted(
        name for name, msg_type in topics.items()
        if name in names or (type_suffix and msg_type.endswith(type_suffix))
    )


def inspect_bag(metadata: Path) -> dict[str, object]:
    topics = topics_from_metadata(metadata)
    scan = topic_matching(topics, names=("/scan",), type_suffix="/LaserScan")
    odom = topic_matching(topics, names=("/odom",), type_suffix="/Odometry")
    transforms = topic_matching(topics, names=("/tf", "/tf_static"), type_suffix="/TFMessage")
    commands = topic_matching(topics, names=("/cmd_vel", "/cmd_vel_smoothed"), type_suffix="/Twist")
    maps = topic_matching(topics, type_suffix="/OccupancyGrid")
    plans = topic_matching(topics, type_suffix="/Path")
    if scan and odom and transforms:
        readiness = "shadow_ready"
    elif transforms and (scan or odom):
        readiness = "adapter_required"
    else:
        readiness = "insufficient_for_nav2_replay"
    return {
        "bag": str(metadata.parent), "readiness": readiness,
        "scan_topics": scan, "odom_topics": odom, "tf_topics": transforms,
        "command_topics": commands, "map_topics": maps, "plan_topics": plans,
        "topic_count": len(topics), "topics": topics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("build/datasets/erl_navigation"))
    parser.add_argument("--download", action="store_true", help="Download the 6 GB archive from Zenodo.")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--archive", type=Path, help="Use an archive already downloaded elsewhere.")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--list-remote",
        action="store_true",
        help="List ZIP members using HTTP ranges without downloading the 6 GB archive.",
    )
    parser.add_argument(
        "--remote-member",
        help="Extract one ZIP member with HTTP ranges instead of downloading the full archive.",
    )
    args = parser.parse_args()

    if args.list_remote:
        print(json.dumps({"members": list_remote_members()}, indent=2))
        return 0
    if args.remote_member:
        selected = args.data_dir / "selected" / Path(args.remote_member).parent.name
        output = extract_remote_member(args.remote_member, selected)
        print(f"wrote {output}")
        return 0

    archive = args.archive or args.data_dir / ARCHIVE_NAME
    if args.download:
        download(DATASET_URL, archive)
    if (args.download or args.extract) and not archive.exists():
        parser.error(f"archive not found: {archive}")
    if archive.exists() and (args.download or args.extract):
        actual_md5 = md5(archive)
        if actual_md5 != ARCHIVE_MD5:
            raise SystemExit(f"checksum mismatch: expected {ARCHIVE_MD5}, got {actual_md5}")
    if args.extract:
        extracted = args.data_dir / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)

    metadata_files = sorted(args.data_dir.rglob("metadata.yaml"))
    report = {
        "dataset": "Navigation Benchmark Rosbags Inspired by ERL Competition Test",
        "doi": "10.5281/zenodo.10518775", "license": "CC BY 4.0",
        "bags": [inspect_bag(path) for path in metadata_files],
    }
    report_path = args.report or args.data_dir / "compatibility_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for bag in report["bags"]:
        readiness = str(bag["readiness"])
        counts[readiness] = counts.get(readiness, 0) + 1
    print(f"inspected {len(metadata_files)} bags: {counts}")
    print(f"wrote {report_path}")
    return 0 if metadata_files else 2


if __name__ == "__main__":
    raise SystemExit(main())
