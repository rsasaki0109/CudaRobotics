#!/usr/bin/env python3
"""Materialize the selected real-sensor CudaNav dataset evidence contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from cudanav_rosbag_evidence import describe_input, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "docs" / "cudanav_real_dataset.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def metadata_path(bag: Path) -> Path:
    root = bag.resolve()
    candidates = [root / "metadata.yaml"] if root.is_dir() else []
    candidates = [path for path in candidates if path.is_file()]
    if not candidates and root.is_dir():
        candidates = sorted(root.rglob("metadata.yaml"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one rosbag2 metadata.yaml under {root}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def rosbag_topics(metadata: Path) -> dict[str, dict[str, Any]]:
    """Read topic types and aggregate counts without requiring PyYAML."""
    topics: dict[str, dict[str, Any]] = {}
    current_name: str | None = None
    current_type: str | None = None
    for raw_line in metadata.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("name:"):
            current_name = stripped.split(":", 1)[1].strip().strip("\"'")
            current_type = None
        elif current_name is not None and stripped.startswith("type:"):
            current_type = stripped.split(":", 1)[1].strip().strip("\"'")
            topics.setdefault(current_name, {"type": current_type, "count": 0})
        elif current_name is not None and stripped.startswith("message_count:"):
            value = stripped.split(":", 1)[1].strip()
            entry = topics.setdefault(
                current_name, {"type": current_type or "", "count": 0}
            )
            try:
                entry["count"] = int(entry["count"]) + int(value)
            except ValueError:
                entry["count"] = -1
            current_name = None
            current_type = None
    return topics


def make_materialization(
    spec_path: Path,
    source_bag: Path,
    derived_path_bag: Path,
    generator_report: Path | None = None,
    acquisition_report: Path | None = None,
) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    spec = read_json(spec_path)
    source_bag = source_bag.resolve()
    derived_path_bag = derived_path_bag.resolve()
    source_metadata = metadata_path(source_bag)
    derived_metadata = metadata_path(derived_path_bag)
    source_identity = describe_input(source_bag)
    derived_identity = describe_input(derived_path_bag)
    path_contract = spec["path_derivation"]
    payload = {
        "schema_version": 2,
        "evidence_mode": "real_sensor_shadow_with_derived_path",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": spec["dataset_id"],
        "dataset_spec": {
            "path": str(spec_path),
            "sha256": sha256_file(spec_path),
        },
        "source_bag": source_identity,
        "source_metadata": {
            "relative_path": source_metadata.relative_to(source_bag).as_posix(),
            "sha256": sha256_file(source_metadata),
            "topics": rosbag_topics(source_metadata),
        },
        "derived_path_bag": derived_identity,
        "derived_path_metadata": {
            "relative_path": derived_metadata.relative_to(
                derived_path_bag
            ).as_posix(),
            "sha256": sha256_file(derived_metadata),
            "topics": rosbag_topics(derived_metadata),
        },
        "provenance": {
            "source_tree_sha256": source_identity["tree_sha256"],
            "source_topic": path_contract["source_topic"],
            "output_topic": path_contract["output_topic"],
            "output_type": path_contract["output_type"],
            "algorithm": path_contract["algorithm"],
            "parameters": path_contract["parameters"],
            "derived_tree_sha256": derived_identity["tree_sha256"],
            "recorded_path": False,
            "closed_loop": False,
        },
    }
    if generator_report is not None:
        report_path = generator_report.resolve()
        report = read_json(report_path)
        payload["generator_report"] = {
            **report,
            "source": str(report_path),
            "sha256": sha256_file(report_path),
        }
    if acquisition_report is not None:
        report_path = acquisition_report.resolve()
        report = read_json(report_path)
        payload["acquisition_inspection"] = {
            **report,
            "source": str(report_path),
            "sha256": sha256_file(report_path),
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--source-bag", type=Path, required=True)
    parser.add_argument("--derived-path-bag", type=Path, required=True)
    parser.add_argument("--generator-report", type=Path, required=True)
    parser.add_argument("--acquisition-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = make_materialization(
        args.spec,
        args.source_bag,
        args.derived_path_bag,
        args.generator_report,
        args.acquisition_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
