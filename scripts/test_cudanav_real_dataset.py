#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from cudanav_real_dataset import DEFAULT_SPEC, make_materialization, read_json
from validate_cudanav_real_dataset import evaluate, evaluate_materialization


def write_bag(root: Path, topics: list[tuple[str, str, int]]) -> Path:
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
    (root / "data.db3").write_bytes(b"fixture rosbag content")
    return root


class CudaNavRealDatasetTest(unittest.TestCase):
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
            )
            path = spec["path_derivation"]
            derived = write_bag(
                root / "derived",
                [(path["output_topic"], path["output_type"], 2)],
            )
            evidence_path = root / "materialization.json"
            evidence_path.write_text(
                json.dumps(
                    make_materialization(DEFAULT_SPEC, source, derived),
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
            )
            path = spec["path_derivation"]
            derived = write_bag(
                root / "derived",
                [(path["output_topic"], path["output_type"], 0)],
            )
            evidence = make_materialization(DEFAULT_SPEC, source, derived)
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
            )
            path = spec["path_derivation"]
            derived = write_bag(
                root / "derived",
                [(path["output_topic"], path["output_type"], 2)],
            )
            evidence = make_materialization(DEFAULT_SPEC, source, derived)
            evidence["source_metadata"]["sha256"] = "0" * 64
            checks = evaluate_materialization(
                spec, DEFAULT_SPEC.resolve(), evidence
            )
            self.assertFalse(checks["source_metadata_bound"])

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
