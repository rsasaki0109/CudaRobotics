#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from cudanav_real_dataset import DEFAULT_SPEC, make_materialization, read_json
from publish_cudanav_dataset_evidence import (
    evaluate_portable_evidence,
    make_portable_evidence,
    render_markdown,
)
from test_cudanav_real_dataset import (
    write_bag,
    write_inspection,
    write_report,
)


class PublishCudaNavDatasetEvidenceTest(unittest.TestCase):
    def test_portable_evidence_strips_paths_and_preserves_scope(self) -> None:
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
            derived = write_bag(
                root / "derived",
                [
                    (
                        spec["path_derivation"]["output_topic"],
                        spec["path_derivation"]["output_type"],
                        1,
                    )
                ],
            )
            report = write_report(root, spec)
            materialization = root / "materialization.json"
            materialization.write_text(
                json.dumps(
                    make_materialization(
                        DEFAULT_SPEC,
                        source,
                        derived,
                        report,
                        write_inspection(source, spec),
                    )
                )
                + "\n"
            )
            payload = make_portable_evidence(
                DEFAULT_SPEC,
                materialization,
                result_id="fixture",
                git_commit="a" * 40,
            )
            result = evaluate_portable_evidence(
                payload,
                expected_commit="a" * 40,
            )
            self.assertTrue(result["valid"], result)
            self.assertNotIn(str(root.resolve()), json.dumps(payload))
            self.assertFalse(payload["claims"]["gpu_controller_run"])
            self.assertIn("GPU controller run: no", render_markdown(payload))

            relabelled = deepcopy(payload)
            relabelled["claims"]["closed_loop"] = True
            self.assertFalse(
                evaluate_portable_evidence(relabelled)["valid"]
            )


if __name__ == "__main__":
    unittest.main()
