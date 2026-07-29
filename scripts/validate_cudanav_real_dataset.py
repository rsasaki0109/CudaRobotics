#!/usr/bin/env python3
"""Validate the selected real-sensor dataset and optional materialization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any

from cudanav_real_dataset import DEFAULT_SPEC, read_json
from cudanav_rosbag_evidence import describe_input, sha256_file
from export_rosbag_motion import CdrReader


SHA256 = re.compile(r"[0-9a-f]{64}")
EVIDENCE_MODE = "real_sensor_shadow_with_derived_path"
EXPECTED_DATASETS = {
    "autoware_istanbul_mapping_kit": {
        "acquisition": {
            "method": "gdown_file",
            "file_id": "1uta5Xr_ftV4jERxPNVqooDvWerK0dn89",
            "uri": (
                "https://drive.google.com/uc"
                "?id=1uta5Xr_ftV4jERxPNVqooDvWerK0dn89"
            ),
            "folder_id": "1BMPcUhjq_BCLi521X88WpujoOiEi3_CJ",
            "expected_database": "test_20240930_134039_0.db3",
            "expected_database_bytes": 60179423232,
            "metadata_file_id": "10tw3sBZzVAiu9gWbB4mMclGuzY2-86In",
            "expected_metadata": "metadata.yaml",
            "expected_metadata_bytes": 4854,
            "redistribution_authorized": False,
        },
        "recorded": {
            "pointcloud": ("/pandar_points", "sensor_msgs/msg/PointCloud2"),
            "odometry": (
                "/applanix/lvx_client/odom",
                "nav_msgs/msg/Odometry",
            ),
            "static_transforms": ("/tf_static", "tf2_msgs/msg/TFMessage"),
        },
        "path_kind": "deterministic_sidecar_from_recorded_odometry",
        "path_algorithm": "cudarobotics.recorded_odometry_path.v1",
    },
    "autoware_istanbul_localization_smoke": {
        "acquisition": {
            "method": "google_drive_file",
            "file_id": "1yEB5j74gPLLbkkf87cuCxUgHXTkgSZbn",
            "uri": (
                "https://drive.google.com/uc"
                "?id=1yEB5j74gPLLbkkf87cuCxUgHXTkgSZbn"
            ),
            "expected_database": "rosbag2_2024_09_12-14_59_58_0.db3",
            "expected_database_bytes": 1009799168,
            "expected_database_sha256": (
                "eb80d649a41fd557ff3af5df4424051191fb696d0ebecbeb36b385702d2b4c8d"
            ),
            "redistribution_authorized": False,
        },
        "recorded": {
            "pointcloud": (
                "/localization/util/downsample/pointcloud",
                "sensor_msgs/msg/PointCloud2",
            ),
            "odometry": (
                "/sensing/gnss/pose",
                "geometry_msgs/msg/PoseStamped",
            ),
            "static_transforms": ("/tf_static", "tf2_msgs/msg/TFMessage"),
        },
        "path_kind": "deterministic_sidecar_from_recorded_pose",
        "path_algorithm": "cudarobotics.recorded_pose_path.v1",
    },
}


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
    expected = EXPECTED_DATASETS.get(spec.get("dataset_id"), {})
    required = expected.get("recorded", {})
    expected_acquisition = expected.get("acquisition", {})
    expected_odometry = required.get("odometry", (None, None))
    expected_pointcloud = required.get("pointcloud", (None, None))
    return {
        "schema": spec.get("schema_version") == 1,
        "dataset_selected": (
            bool(expected)
            and spec.get("status") in {"selected_unmaterialized", "materialized"}
        ),
        "canonical_primary_source": str(
            spec.get("canonical_documentation", "")
        ).startswith("https://autowarefoundation.github.io/"),
        "acquisition_uri": (
            isinstance(acquisition, dict)
            and bool(expected_acquisition)
            and all(
                acquisition.get(key) == value
                for key, value in expected_acquisition.items()
            )
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
            and path.get("kind") == expected.get("path_kind")
            and path.get("algorithm") == expected.get("path_algorithm")
            and path.get("source_topic") == expected_odometry[0]
            and path.get("source_type") == expected_odometry[1]
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
            and quality.get("pointcloud_topic") == expected_pointcloud[0]
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


def _remote_probe_bound(
    expected: dict[str, Any],
    reported: dict[str, Any],
    probe: dict[str, Any],
) -> bool:
    expected_checks = {
        "database_filename": True,
        "database_bytes": True,
    }
    database = probe.get("database", {})
    valid = (
        probe.get("schema_version") == 1
        and probe.get("passed") is True
        and reported.get("method") == expected.get("method")
        and reported.get("file_id") == expected.get("file_id")
        and reported.get("expected_database")
        == expected.get("expected_database")
        and reported.get("expected_database_bytes")
        == expected.get("expected_database_bytes")
        and database.get("file_id") == expected.get("file_id")
        and database.get("filename") == expected.get("expected_database")
        and database.get("bytes") == expected.get("expected_database_bytes")
    )
    if "metadata_file_id" in expected:
        expected_checks.update(
            {
                "metadata_filename": True,
                "metadata_bytes": True,
            }
        )
        metadata = probe.get("metadata", {})
        valid = (
            valid
            and reported.get("metadata_file_id")
            == expected.get("metadata_file_id")
            and reported.get("expected_metadata")
            == expected.get("expected_metadata")
            and metadata.get("file_id") == expected.get("metadata_file_id")
            and metadata.get("filename") == expected.get("expected_metadata")
            and metadata.get("bytes") == expected.get("expected_metadata_bytes")
        )
    return (
        valid
        and isinstance(probe.get("checks"), dict)
        and probe["checks"] == expected_checks
    )


def _decode_path(payload: bytes) -> dict[str, Any]:
    reader = CdrReader(payload)

    def stamp() -> int:
        seconds = reader.int32()
        nanoseconds = reader.uint32()
        if nanoseconds >= 1_000_000_000:
            raise ValueError("invalid ROS timestamp")
        return seconds * 1_000_000_000 + nanoseconds

    header_stamp = stamp()
    header_frame = reader.string()
    count = reader.uint32()
    if count < 2 or count > len(payload) // 64:
        raise ValueError("implausible Path pose count")
    poses = []
    for _ in range(count):
        pose_stamp = stamp()
        frame_id = reader.string()
        values = reader.doubles(7)
        poses.append(
            {
                "stamp_ns": pose_stamp,
                "frame_id": frame_id,
                "position": values[:3],
                "quaternion": values[3:],
            }
        )
    if reader.offset != len(payload):
        raise ValueError("trailing Path CDR bytes")
    return {
        "header_stamp_ns": header_stamp,
        "header_frame_id": header_frame,
        "poses": poses,
    }


def _sqlite_path_semantics(
    identity: Any,
    contract: dict[str, Any],
    generator: dict[str, Any],
) -> bool:
    if generator.get("storage_id") != "sqlite3":
        return generator.get("storage_id") == "mcap"
    if not identity_schema(identity):
        return False
    try:
        root = Path(identity["source"]).resolve()
        databases = [
            (root / entry["path"]).resolve()
            for entry in identity["files"]
            if isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and entry["path"].endswith(".db3")
        ]
        if len(databases) != 1 or not databases[0].is_relative_to(root):
            return False
        connection = sqlite3.connect(
            f"file:{databases[0].as_posix()}?mode=ro", uri=True
        )
        try:
            topic = connection.execute(
                "SELECT id, type, serialization_format FROM topics "
                "WHERE name = ?",
                (contract["output_topic"],),
            ).fetchone()
            if topic is None or topic[1:] != (
                contract["output_type"],
                "cdr",
            ):
                return False
            messages = connection.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id = ? "
                "ORDER BY timestamp",
                (topic[0],),
            ).fetchall()
        finally:
            connection.close()
        if len(messages) != 1:
            return False
        decoded = _decode_path(bytes(messages[0][1]))
        poses = decoded["poses"]
        stamps = [pose["stamp_ns"] for pose in poses]
        frame_id = generator.get("frame_id")
        finite = all(
            math.isfinite(value)
            for pose in poses
            for values in (pose["position"], pose["quaternion"])
            for value in values
        )
        unit_quaternions = all(
            abs(
                math.sqrt(sum(value * value for value in pose["quaternion"]))
                - 1.0
            )
            <= 1e-5
            for pose in poses
        )
        first = poses[0]
        normalized_origin = all(
            abs(value) <= 1e-6
            for value in (
                *first["position"],
                *first["quaternion"][:3],
                first["quaternion"][3] - 1.0,
            )
        )
        maximum_duration_s = contract["parameters"].get(
            "maximum_duration_s"
        )
        duration_bounded = (
            maximum_duration_s is None
            or stamps[-1] - stamps[0]
            <= round(float(maximum_duration_s) * 1e9)
        )
        return (
            len(poses) == generator.get("output_poses")
            and stamps[0] == generator.get("first_stamp_ns")
            and stamps[-1] == generator.get("last_stamp_ns")
            and decoded["header_stamp_ns"] == stamps[0]
            and decoded["header_frame_id"] == frame_id
            and all(pose["frame_id"] == frame_id for pose in poses)
            and all(left < right for left, right in zip(stamps, stamps[1:]))
            and finite
            and unit_quaternions
            and normalized_origin
            and duration_bounded
        )
    except (
        KeyError,
        OSError,
        sqlite3.Error,
        TypeError,
        ValueError,
    ):
        return False


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
    inspection = evidence.get("acquisition_inspection", {})
    recorded = spec["recorded_inputs"]
    path = spec["path_derivation"]
    input_samples = generator.get("input_samples")
    output_poses = generator.get("output_poses")
    checks = {
        "materialization_schema": evidence.get("schema_version") == 2,
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
            and provenance.get("source_type") == path["source_type"]
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
            and generator.get("source_type") == path["source_type"]
            and generator.get("output_topic") == path["output_topic"]
            and generator.get("parameters") == path["parameters"]
            and isinstance(input_samples, int)
            and isinstance(output_poses, int)
            and input_samples >= output_poses >= 2
            and generator.get("frame_id") == "odom"
            and generator.get("storage_id") in {"mcap", "sqlite3"}
            and generator.get("recorded_path") is False
            and generator.get("closed_loop") is False
        ),
        "derived_sqlite_path_semantics": _sqlite_path_semantics(
            derived, path, generator
        ),
        "generator_report_content": False,
        "acquisition_inspection_bound": False,
        "acquisition_inspection_content": False,
        "source_database_contract": False,
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
    try:
        inspection_path = Path(inspection["source"]).resolve()
        inspection_payload = read_json(inspection_path)
        database = inspection["database"]
        database_path = Path(database["source"]).resolve()
        source_root = Path(source["source"]).resolve()
        database_relative = database_path.relative_to(source_root).as_posix()
        required_checks = inspection["required_topic_checks"]
        acquisition = inspection["acquisition"]
        inspection_topics = inspection["topics"]
        remote_probe = inspection["remote_probe"]
        checks["acquisition_inspection_bound"] = (
            inspection.get("schema_version") == 1
            and inspection.get("dataset_id") == spec["dataset_id"]
            and inspection.get("dataset_spec", {}).get("sha256")
            == sha256_file(spec_path)
            and _remote_probe_bound(
                spec["acquisition"], acquisition, remote_probe
            )
            and database_path.name == spec["acquisition"]["expected_database"]
            and isinstance(database.get("bytes"), int)
            and database["bytes"] > 0
            and bool(SHA256.fullmatch(str(database.get("sha256", ""))))
            and any(
                entry.get("path") == database_relative
                and entry.get("sha256") == database["sha256"]
                and entry.get("bytes") == database["bytes"]
                for entry in source["files"]
                if isinstance(entry, dict)
            )
            and isinstance(required_checks, dict)
            and set(required_checks) == set(spec["recorded_inputs"])
            and all(value is True for value in required_checks.values())
            and all(
                _topic_matches(
                    inspection_topics,
                    contract["topic"],
                    contract["type"],
                )
                for contract in spec["recorded_inputs"].values()
            )
            and inspection.get("passed") is True
        )
        expected_database_sha = spec["acquisition"].get(
            "expected_database_sha256"
        )
        checks["source_database_contract"] = (
            expected_database_sha is None
            or not verify_source
            or (
                inspection.get("database_contract_checks")
                == {
                    "database_bytes": True,
                    "database_sha256": True,
                }
                and database["bytes"]
                == spec["acquisition"]["expected_database_bytes"]
                and database["sha256"] == expected_database_sha
            )
        )
        checks["acquisition_inspection_content"] = (
            inspection_path.is_file()
            and sha256_file(inspection_path) == inspection["sha256"]
            and all(
                inspection.get(key) == value
                for key, value in inspection_payload.items()
            )
            and set(inspection) == set(inspection_payload) | {"source", "sha256"}
        )
    except (KeyError, OSError, TypeError, ValueError):
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
