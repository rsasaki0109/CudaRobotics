#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch
import unittest

from cudanav_real_dataset import DEFAULT_SPEC
from run_cudanav_real_dataset_pipeline import (
    DEFAULT_CONTROLLER,
    DEFAULT_CONTROLLER_COMMAND,
    command_plan,
    parse_args,
    validate_args,
)


class CudaNavRealDatasetPipelineTest(unittest.TestCase):
    def test_dependency_free_sqlite_sidecar_is_the_default(self) -> None:
        with patch(
            "sys.argv",
            ["run_cudanav_real_dataset_pipeline.py"],
        ):
            args = parse_args()
        self.assertEqual(args.sidecar_storage, "sqlite3")
        self.assertTrue(args.probe)

    def arguments(self, root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            spec=DEFAULT_SPEC,
            dataset_dir=root / "dataset",
            work_dir=root / "work",
            download=True,
            download_backend="curl",
            probe=False,
            reindex=True,
            generate_metadata=False,
            sidecar_storage="mcap",
            run_autonomy=True,
            profile="smoke",
            autonomy_output_dir=root / "autonomy",
            controller_config=DEFAULT_CONTROLLER,
            controller_command=DEFAULT_CONTROLLER_COMMAND,
            rosbag_duration_sec=30.0,
            multi_gpu_run=[],
            multi_gpu_devices=None,
            multi_gpu_repetitions=1,
            closed_loop_timeout_sec=None,
            resume=False,
            dry_run=True,
        )

    def test_plan_connects_inspection_to_materialization_and_replay(
        self,
    ) -> None:
        root = Path("build/pipeline_fixture").resolve()
        args = self.arguments(root)
        plan = command_plan(args)
        prepare = plan["stages"]["prepare"]
        derive = plan["stages"]["derive_path"]
        validate = plan["stages"]["validate_materialization"]
        autonomy = plan["stages"]["run_autonomy"]
        inspection = plan["paths"]["inspection"]
        materialization = plan["paths"]["materialization"]
        self.assertIn("--download", prepare)
        self.assertIn("--reindex", prepare)
        self.assertEqual(
            derive[derive.index("--acquisition-report") + 1], inspection
        )
        self.assertEqual(
            derive[derive.index("--materialization") + 1], materialization
        )
        self.assertEqual(
            validate[validate.index("--materialization") + 1],
            materialization,
        )
        self.assertEqual(
            autonomy[autonomy.index("--dataset-materialization") + 1],
            materialization,
        )
        self.assertEqual(
            autonomy[autonomy.index("--evaluation-db") + 1],
            plan["paths"]["database"],
        )

    def test_release_plan_requires_multi_gpu_evidence(self) -> None:
        args = self.arguments(Path("build/pipeline_fixture").resolve())
        args.profile = "release"
        with self.assertRaises(SystemExit):
            validate_args(args)
        args.multi_gpu_devices = "0,1"
        validate_args(args)
        plan = command_plan(args)
        self.assertIn("--multi-gpu-devices", plan["stages"]["run_autonomy"])


if __name__ == "__main__":
    unittest.main()
