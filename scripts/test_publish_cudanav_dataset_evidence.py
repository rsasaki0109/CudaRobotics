#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from cudanav_real_dataset import make_materialization, read_json
from derive_cudanav_path_sidecar import derive_path, write_sqlite_rosbag
from publish_cudanav_dataset_evidence import (
    evaluate_portable_evidence,
    make_portable_evidence,
    render_markdown,
)
from test_cudanav_real_dataset import (
    SMOKE_SPEC,
    write_bag,
    write_inspection,
    write_report,
)


class PublishCudaNavDatasetEvidenceTest(unittest.TestCase):
    def test_portable_evidence_strips_paths_and_preserves_scope(self) -> None:
        spec = read_json(SMOKE_SPEC)
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
            poses = derive_path(
                [
                    {
                        "stamp_ns": 1_000_000_000,
                        "x": 2.0,
                        "y": 3.0,
                        "z": 1.0,
                        "yaw": math.pi / 2,
                    },
                    {
                        "stamp_ns": 2_000_000_000,
                        "x": 2.0,
                        "y": 4.0,
                        "z": 1.0,
                        "yaw": math.pi / 2,
                    },
                ],
                0.05,
                120.0,
            )
            derived = root / "derived"
            write_sqlite_rosbag(
                derived,
                spec["path_derivation"]["output_topic"],
                "odom",
                poses,
                5_000_000_000,
            )
            report = write_report(
                root,
                spec,
                storage_id="sqlite3",
                poses=poses,
            )
            materialization = root / "materialization.json"
            materialization.write_text(
                json.dumps(
                    make_materialization(
                        SMOKE_SPEC,
                        source,
                        derived,
                        report,
                        write_inspection(source, spec, SMOKE_SPEC),
                    )
                )
                + "\n"
            )
            payload = make_portable_evidence(
                SMOKE_SPEC,
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
