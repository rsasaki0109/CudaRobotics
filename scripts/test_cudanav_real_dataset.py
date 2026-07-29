#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from cudanav_real_dataset import DEFAULT_SPEC, make_materialization, read_json
from cudanav_rosbag_evidence import sha256_file
from run_cudanav_rosbag_replay import play_argv
from validate_cudanav_real_dataset import evaluate, evaluate_materialization


def write_bag(
    root: Path,
    topics: list[tuple[str, str, int]],
    database_name: str = "data.db3",
) -> Path:
    root.mkdir(parents=True)
    lines = ["rosbag2_bagfile_information:", "  topics_with_message_count:"]
    for name, message_type, count in topics:
        lines.extend(
            [
                "    - topic_metadata:",
                f"        name: {name}",
                f"        type: {message_type}",
                f"      message_count: {count}",
            ]
        )
    (root / "metadata.yaml").write_text("\n".join(lines) + "\n")
    (root / database_name).write_bytes(b"fixture rosbag content")
    return root


def write_report(root: Path, spec: dict, input_samples: int = 5) -> Path:
    contract = spec["path_derivation"]
    path = root / "generator_report.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": contract["algorithm"],
                "source_topic": contract["source_topic"],
                "output_topic": contract["output_topic"],
                "parameters": contract["parameters"],
                "input_samples": input_samples,
                "output_poses": 2,
                "first_stamp_ns": 1,
                "last_stamp_ns": 2,
                "frame_id": "odom",
                "recorded_path": False,
                "closed_loop": False,
            }
        )
        + "\n"
    )
    return path


def write_inspection(source: Path, spec: dict) -> Path:
    database = source / spec["acquisition"]["expected_database"]
    topics = {
        contract["topic"]: {"type": contract["type"], "count": 5}
        for contract in spec["recorded_inputs"].values()
    }
    path = source / "inspection.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": spec["dataset_id"],
                "inspected_at": "2026-07-29T00:00:00+00:00",
                "dataset_spec": {
                    "path": str(DEFAULT_SPEC.resolve()),
                    "sha256": sha256_file(DEFAULT_SPEC),
                },
                "acquisition": {
                    "method": spec["acquisition"]["method"],
                    "file_id": spec["acquisition"]["file_id"],
                    "expected_database": spec["acquisition"][
                        "expected_database"
                    ],
                },
                "database": {
                    "source": str(database.resolve()),
                    "bytes": database.stat().st_size,
                    "sha256": sha256_file(database),
                },
                "topics": topics,
                "required_topic_checks": {
                    name: True for name in spec["recorded_inputs"]
                },
                "passed": True,
            }
        )
        + "\n"
    )
    return path


class CudaNavRealDatasetTest(unittest.TestCase):
    def test_derived_replay_uses_rosbag2_multi_input_ordering(self) -> None:
        command = play_argv(
            Path("/data/source"),
            Path("/data/path"),
            True,
            ["--rate", "0.5"],
        )
        self.assertEqual(
            command,
            [
                "ros2",
                "bag",
                "play",
                "-i",
                str(Path("/data/source")),
                "-i",
                str(Path("/data/path")),
                "--clock",
                "--rate",
                "0.5",
            ],
        )

    def test_selected_spec_is_valid_but_not_materialized(self) -> None:
        result = evaluate(DEFAULT_SPEC)
        self.assertTrue(result["valid"], result)
        self.assertFalse(result["ready"])
        self.assertEqual(result["status"], "selected_unmaterialized")

    def test_materialization_binds_real_inputs_and_derived_path(self) -> None:
        spec = read_json(DEFAULT_SPEC)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_bag(
                root / "source",
                [
                    (entry["topic"], entry["type"], 5)
                    for entry in spec["recorded_inputs"].values()
                ],
                spec["acquisition"]["expected_database"],
            )
            path = spec["path_derivation"]
            derived = write_bag(
                root / "derived",
                [(path["output_topic"], path["output_type"], 2)],
            )
            evidence_path = root / "materialization.json"
            evidence_path.write_text(
                json.dumps(
                    make_materialization(
                        DEFAULT_SPEC,
                        source,
                        derived,
                        write_report(root, spec),
                        write_inspection(source, spec),
                    ),
                    indent=2,
                )
                + "\n"
            )
            result = evaluate(DEFAULT_SPEC, evidence_path)
            self.assertTrue(result["valid"], result)
            self.assertTrue(result["ready"])

    def test_zero_count_and_closed_loop_relabel_are_rejected(self) -> None:
        spec = read_json(DEFAULT_SPEC)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_bag(
                root / "source",
                [
                    (entry["topic"], entry["type"], 5)
                    for entry in spec["recorded_inputs"].values()
                ],
                spec["acquisition"]["expected_database"],
            )
            path = spec["path_derivation"]
            derived = write_bag(
                root / "derived",
                [(path["output_topic"], path["output_type"], 0)],
            )
            evidence = make_materialization(
                DEFAULT_SPEC,
                source,
                derived,
                write_report(root, spec),
                write_inspection(source, spec),
            )
            evidence["provenance"]["closed_loop"] = True
            checks = evaluate_materialization(
                spec, DEFAULT_SPEC.resolve(), evidence
            )
            self.assertFalse(checks["derived_path_present"])
            self.assertFalse(checks["provenance_bound"])

    def test_metadata_digest_cannot_be_detached_from_bag_identity(self) -> None:
        spec = read_json(DEFAULT_SPEC)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_bag(
                root / "source",
                [
                    (entry["topic"], entry["type"], 5)
                    for entry in spec["recorded_inputs"].values()
                ],
                spec["acquisition"]["expected_database"],
            )
            path = spec["path_derivation"]
            derived = write_bag(
                root / "derived",
                [(path["output_topic"], path["output_type"], 2)],
            )
            evidence = make_materialization(
                DEFAULT_SPEC,
                source,
                derived,
                write_report(root, spec),
                write_inspection(source, spec),
            )
            evidence["source_metadata"]["sha256"] = "0" * 64
            checks = evaluate_materialization(
                spec, DEFAULT_SPEC.resolve(), evidence
            )
            self.assertFalse(checks["source_metadata_bound"])

    def test_generator_fields_cannot_be_detached_from_hashed_report(self) -> None:
        spec = read_json(DEFAULT_SPEC)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_bag(
                root / "source",
                [
                    (entry["topic"], entry["type"], 5)
                    for entry in spec["recorded_inputs"].values()
                ],
                spec["acquisition"]["expected_database"],
            )
            path = spec["path_derivation"]
            derived = write_bag(
                root / "derived",
                [(path["output_topic"], path["output_type"], 2)],
            )
            evidence = make_materialization(
                DEFAULT_SPEC,
                source,
                derived,
                write_report(root, spec),
                write_inspection(source, spec),
            )
            evidence["generator_report"]["output_poses"] = 3
            checks = evaluate_materialization(
                spec, DEFAULT_SPEC.resolve(), evidence
            )
            self.assertFalse(checks["generator_report_content"])

    def test_acquisition_database_digest_cannot_be_relabelled(self) -> None:
        spec = read_json(DEFAULT_SPEC)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_bag(
                root / "source",
                [
                    (entry["topic"], entry["type"], 5)
                    for entry in spec["recorded_inputs"].values()
                ],
                spec["acquisition"]["expected_database"],
            )
            path = spec["path_derivation"]
            derived = write_bag(
                root / "derived",
                [(path["output_topic"], path["output_type"], 2)],
            )
            evidence = make_materialization(
                DEFAULT_SPEC,
                source,
                derived,
                write_report(root, spec),
                write_inspection(source, spec),
            )
            evidence["acquisition_inspection"]["database"]["sha256"] = "0" * 64
            checks = evaluate_materialization(
                spec, DEFAULT_SPEC.resolve(), evidence
            )
            self.assertFalse(checks["acquisition_inspection_bound"])
            self.assertFalse(checks["acquisition_inspection_content"])

    def test_spec_cannot_call_derived_path_recorded(self) -> None:
        spec = deepcopy(read_json(DEFAULT_SPEC))
        spec["path_derivation"]["claims"]["recorded_path"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            path.write_text(json.dumps(spec))
            result = evaluate(path)
            self.assertFalse(result["valid"])
            self.assertFalse(result["checks"]["claims_are_shadow_only"])


if __name__ == "__main__":
    unittest.main()
