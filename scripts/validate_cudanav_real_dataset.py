#!/usr/bin/env python3
"""Validate the selected real-sensor dataset and optional materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from cudanav_real_dataset import DEFAULT_SPEC, read_json
from cudanav_rosbag_evidence import describe_input, sha256_file


SHA256 = re.compile(r"[0-9a-f]{64}")
EVIDENCE_MODE = "real_sensor_shadow_with_derived_path"


def identity_schema(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(SHA256.fullmatch(str(value.get("tree_sha256", ""))))
        and isinstance(value.get("file_count"), int)
        and value["file_count"] > 0
        and isinstance(value.get("total_bytes"), int)
        and value["total_bytes"] > 0
        and isinstance(value.get("files"), list)
        and len(value["files"]) == value["file_count"]
    )


def evaluate_spec(spec: dict[str, Any]) -> dict[str, bool]:
    acquisition = spec.get("acquisition", {})
    recorded = spec.get("recorded_inputs", {})
    path = spec.get("path_derivation", {})
    claims = path.get("claims", {}) if isinstance(path, dict) else {}
    quality = spec.get("quality_evaluation", {})
    required = {
        "pointcloud": ("/pandar_points", "sensor_msgs/msg/PointCloud2"),
        "odometry": ("/applanix/lvx_client/odom", "nav_msgs/msg/Odometry"),
        "static_transforms": ("/tf_static", "tf2_msgs/msg/TFMessage"),
    }
    return {
        "schema": spec.get("schema_version") == 1,
        "dataset_selected": (
            spec.get("dataset_id") == "autoware_istanbul_mapping_kit"
            and spec.get("status") in {"selected_unmaterialized", "materialized"}
        ),
        "canonical_primary_source": str(
            spec.get("canonical_documentation", "")
        ).startswith("https://autowarefoundation.github.io/"),
        "acquisition_uri": (
            isinstance(acquisition, dict)
            and str(acquisition.get("uri", "")).startswith("https://")
            and acquisition.get("redistribution_authorized") is False
        ),
        "recorded_topics": (
            isinstance(recorded, dict)
            and all(
                recorded.get(name, {}).get("topic") == expected[0]
                and recorded.get(name, {}).get("type") == expected[1]
                for name, expected in required.items()
            )
        ),
        "derived_path_contract": (
            isinstance(path, dict)
            and path.get("kind")
            == "deterministic_sidecar_from_recorded_odometry"
            and path.get("algorithm")
            == "cudarobotics.recorded_odometry_path.v1"
            and path.get("source_topic") == required["odometry"][0]
            and path.get("output_topic") == "/cuda_nav/derived_plan"
            and path.get("output_type") == "nav_msgs/msg/Path"
            and isinstance(path.get("parameters"), dict)
            and bool(path["parameters"])
        ),
        "claims_are_shadow_only": (
            isinstance(claims, dict)
            and claims.get("recorded_path") is False
            and claims.get("closed_loop") is False
            and claims.get("evidence_mode") == EVIDENCE_MODE
        ),
        "pointcloud_quality_contract": (
            isinstance(quality, dict)
            and quality.get("kind")
            == "pointcloud2_front_clearance_with_shadow_diagnostics"
            and quality.get("pointcloud_topic") == "/pandar_points"
            and quality.get("command_source") == "cuda_mppi_diagnostics_csv"
            and quality.get("timestamp_basis") == "pointcloud_header_stamp"
            and quality.get("filter")
            == {
                "half_angle_rad": 0.5235987755982988,
                "minimum_z_m": -0.5,
                "maximum_z_m": 2.5,
                "minimum_range_m": 0.05,
                "maximum_range_m": 50.0,
                "maximum_command_age_ms": 200.0,
            }
            and quality.get("minimum_command_pair_ratio") == 0.9
        ),
    }


def _topic_matches(
    table: Any, topic: str, message_type: str, *, positive: bool = True
) -> bool:
    if not isinstance(table, dict) or not isinstance(table.get(topic), dict):
        return False
    entry = table[topic]
    return (
        entry.get("type") == message_type
        and isinstance(entry.get("count"), int)
        and (entry["count"] > 0 if positive else entry["count"] >= 0)
    )


def _metadata_bound(identity: Any, metadata: Any) -> bool:
    if not identity_schema(identity) or not isinstance(metadata, dict):
        return False
    relative = metadata.get("relative_path")
    digest = metadata.get("sha256")
    return (
        isinstance(relative, str)
        and bool(relative)
        and bool(SHA256.fullmatch(str(digest)))
        and any(
            entry.get("path") == relative and entry.get("sha256") == digest
            for entry in identity["files"]
            if isinstance(entry, dict)
        )
    )


def evaluate_materialization(
    spec: dict[str, Any],
    spec_path: Path,
    evidence: dict[str, Any],
    *,
    verify_source: bool = True,
) -> dict[str, bool]:
    source = evidence.get("source_bag")
    derived = evidence.get("derived_path_bag")
    source_meta = evidence.get("source_metadata", {})
    derived_meta = evidence.get("derived_path_metadata", {})
    provenance = evidence.get("provenance", {})
    generator = evidence.get("generator_report", {})
    recorded = spec["recorded_inputs"]
    path = spec["path_derivation"]
    input_samples = generator.get("input_samples")
    output_poses = generator.get("output_poses")
    checks = {
        "materialization_schema": evidence.get("schema_version") == 1,
        "materialization_mode": evidence.get("evidence_mode") == EVIDENCE_MODE,
        "dataset_id_matches": evidence.get("dataset_id") == spec["dataset_id"],
        "spec_content_bound": (
            evidence.get("dataset_spec", {}).get("sha256")
            == sha256_file(spec_path)
        ),
        "source_identity": identity_schema(source),
        "derived_identity": identity_schema(derived),
        "source_metadata_bound": _metadata_bound(source, source_meta),
        "derived_metadata_bound": _metadata_bound(derived, derived_meta),
        "recorded_pointcloud": _topic_matches(
            source_meta.get("topics"),
            recorded["pointcloud"]["topic"],
            recorded["pointcloud"]["type"],
        ),
        "recorded_odometry": _topic_matches(
            source_meta.get("topics"),
            recorded["odometry"]["topic"],
            recorded["odometry"]["type"],
        ),
        "recorded_static_tf": _topic_matches(
            source_meta.get("topics"),
            recorded["static_transforms"]["topic"],
            recorded["static_transforms"]["type"],
        ),
        "derived_path_present": _topic_matches(
            derived_meta.get("topics"),
            path["output_topic"],
            path["output_type"],
        ),
        "provenance_bound": (
            isinstance(provenance, dict)
            and isinstance(source, dict)
            and isinstance(derived, dict)
            and provenance.get("source_tree_sha256")
            == source.get("tree_sha256")
            and provenance.get("derived_tree_sha256")
            == derived.get("tree_sha256")
            and provenance.get("source_topic") == path["source_topic"]
            and provenance.get("output_topic") == path["output_topic"]
            and provenance.get("output_type") == path["output_type"]
            and provenance.get("algorithm") == path["algorithm"]
            and provenance.get("parameters") == path["parameters"]
            and provenance.get("recorded_path") is False
            and provenance.get("closed_loop") is False
        ),
        "generator_report_bound": (
            isinstance(generator, dict)
            and bool(SHA256.fullmatch(str(generator.get("sha256", ""))))
            and generator.get("schema_version") == 1
            and generator.get("algorithm") == path["algorithm"]
            and generator.get("source_topic") == path["source_topic"]
            and generator.get("output_topic") == path["output_topic"]
            and generator.get("parameters") == path["parameters"]
            and isinstance(input_samples, int)
            and isinstance(output_poses, int)
            and input_samples >= output_poses >= 2
            and generator.get("frame_id") == "odom"
            and generator.get("recorded_path") is False
            and generator.get("closed_loop") is False
        ),
        "generator_report_content": False,
    }
    try:
        report_path = Path(generator["source"]).resolve()
        report_payload = read_json(report_path)
        checks["generator_report_content"] = (
            report_path.is_file()
            and sha256_file(report_path) == generator["sha256"]
            and all(generator.get(key) == value for key, value in report_payload.items())
            and set(generator) == set(report_payload) | {"source", "sha256"}
        )
    except (KeyError, OSError, TypeError):
        pass
    if verify_source:
        for label, identity in (("source", source), ("derived", derived)):
            try:
                current = describe_input(Path(identity["source"]))
                checks[f"{label}_content_unchanged"] = (
                    current["tree_sha256"] == identity["tree_sha256"]
                    and current["file_count"] == identity["file_count"]
                    and current["total_bytes"] == identity["total_bytes"]
                )
            except (KeyError, OSError, TypeError, ValueError):
                checks[f"{label}_content_unchanged"] = False
    return checks


def evaluate(
    spec_path: Path,
    materialization_path: Path | None = None,
    *,
    verify_source: bool = True,
) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    spec = read_json(spec_path)
    checks = evaluate_spec(spec)
    materialization = None
    if materialization_path is not None:
        materialization = read_json(materialization_path.resolve())
        checks.update(
            evaluate_materialization(
                spec, spec_path, materialization, verify_source=verify_source
            )
        )
    valid = all(checks.values())
    ready = materialization is not None and valid
    return {
        "valid": valid,
        "ready": ready,
        "status": "materialized" if ready else "selected_unmaterialized",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--materialization", type=Path)
    parser.add_argument("--no-verify-source", action="store_true")
    args = parser.parse_args()
    result = evaluate(
        args.spec,
        args.materialization,
        verify_source=not args.no_verify_source,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
