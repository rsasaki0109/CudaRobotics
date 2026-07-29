#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from cudanav_autonomy_suite import evaluate_suite, sha256_file
from cudanav_evidence import REQUIRED_CLOSED_LOOP_BAG_TOPICS
from cudanav_real_dataset import DEFAULT_SPEC, make_materialization, read_json
from cudanav_rosbag_evidence import (
    REQUIRED_CUDANAV_OUTPUT_TOPICS,
    describe_input,
)
from run_autonomy_suite import (
    validate_closed_loop,
    validate_multi_gpu,
    validate_rosbag,
)


COMMIT = "a" * 40
CONFIG = "controller: cuda\n"
CONFIG_SHA = hashlib.sha256(CONFIG.encode()).hexdigest()


def summary() -> dict:
    return {
        "schema_version": 1,
        "success": True,
        "elapsed_sec": 650.0,
        "collision": False,
        "collision_count": 0,
        "ground_truth_distance_m": 100.0,
        "ground_truth_goal_distance_m": 0.1,
        "odometry_position_error_m": 0.5,
        "odometry_drift_percent": 0.5,
        "command_intervals": 12000,
        "command_deadline_misses": 10,
        "command_deadline_miss_rate": 10 / 12000,
        "traversals_requested": 10,
        "traversals_completed": 10,
        "trajectory_csv": "trajectory.csv",
        "diagnostic_error_count": 0,
        "diagnostic_warn_count": 0,
        "diagnostic_status_samples": 500,
        "diagnostic_components": ["esdf", "mapping", "odometry"],
        "failure_counters": {
            "esdf:maps_dropped": 0,
            "mapping:maps_dropped": 0,
            "odometry:scans_dropped": 0,
        },
    }


def write_closed(
    run: Path,
    *,
    gpu_name: str = "GPU A",
    gpu_uuid: str = "GPU-a",
    physical_index: str = "0",
) -> dict:
    run.mkdir(parents=True)
    (run / "mission_summary.json").write_text(
        json.dumps(summary()) + "\n", encoding="utf-8"
    )
    (run / "trajectory.csv").write_text(
        "elapsed_sec,truth_x,truth_y,odom_x,odom_y\n"
        "0.0,0.0,0.0,,\n"
        "650.0,100.0,0.0,99.5,0.0\n",
        encoding="utf-8",
    )
    (run / "launch.log").write_text("launch complete\n", encoding="utf-8")
    (run / "controller.yaml").write_bytes(CONFIG.encode("utf-8"))
    bag = run / "rosbag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n"
        "  storage_identifier: mcap\n"
        "  topics_with_message_count:\n"
        + "".join(
            "    - topic_metadata:\n"
            f"        name: {topic}\n"
            "        type: test_msgs/msg/Test\n"
            "      message_count: 10\n"
            for topic in REQUIRED_CLOSED_LOOP_BAG_TOPICS
        )
    )
    (bag / "closed_loop_0.mcap").write_bytes(
        b"representative closed-loop bytes"
    )
    (run / "replay.gif").write_bytes(b"GIF89a" + b"\0" * 16)
    manifest = {
        "schema_version": 1,
        "evidence_mode": "closed_loop_simulation",
        "git_commit": COMMIT,
        "git_dirty": False,
        "config_sha256": CONFIG_SHA,
        "artifact_sha256": {
            name: sha256_file(run / relative)
            for name, relative in {
                "summary": "mission_summary.json",
                "trajectory": "trajectory.csv",
                "launch_log": "launch.log",
                "controller_config": "controller.yaml",
                "video": "replay.gif",
            }.items()
        },
        "command": [
            "ros2",
            "launch",
            "cuda_nav_bringup",
            "cudanav_closed_loop.launch.py",
            f"controller_config:={run / 'controller.yaml'}",
        ],
        "gpu": [
            {
                "physical_index": physical_index,
                "name": gpu_name,
                "uuid": gpu_uuid,
                "driver_version": "999",
                "memory_total_mib": "8192",
            }
        ],
        "traversal_count": 10,
        "bag_topics": list(REQUIRED_CLOSED_LOOP_BAG_TOPICS),
        "rosbag_identity": describe_input(bag),
        "bag_command": [
            "ros2",
            "bag",
            "record",
            "--output",
            str(bag.resolve()),
            *REQUIRED_CLOSED_LOOP_BAG_TOPICS,
        ],
        "render_command": ["python", "render.py"],
        "artifacts": {
            "summary": "mission_summary.json",
            "trajectory": "trajectory.csv",
            "launch_log": "launch.log",
            "controller_config": "controller.yaml",
            "rosbag": "rosbag",
            "video": "replay.gif",
        },
    }
    (run / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return manifest


def write_rosbag(run: Path) -> dict:
    bag = run.parent / "input_bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("storage_identifier: sqlite3\n")
    database = bag / "run.db3"
    database.write_bytes(b"recorded robot data")
    (run / "evaluation").mkdir(parents=True)
    diagnostics = run / "diagnostics.csv"
    diagnostics.write_text(
        "solve_ms,valid_rollout_ratio\n4.0,0.95\n", encoding="utf-8"
    )
    config = run / "controller.yaml"
    config.write_bytes(CONFIG.encode("utf-8"))
    (run / "controller.log").write_text("controller started\n", encoding="utf-8")
    (run / "play.log").write_text("bag played\n", encoding="utf-8")
    recording = run / "recording"
    recording.mkdir()
    (recording / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n"
        "  storage_identifier: mcap\n"
        "  topics_with_message_count:\n"
        + "".join(
            "    - topic_metadata:\n"
            f"        name: {topic}\n"
            "        type: test_msgs/msg/Test\n"
            "      message_count: 10\n"
            for topic in REQUIRED_CUDANAV_OUTPUT_TOPICS
        )
    )
    (recording / "recording_0.mcap").write_bytes(
        b"representative shadow output bytes"
    )
    evaluation = {
        "schema_version": 1,
        "evidence_mode": "shadow_controller_with_recorded_motion",
        "quality_pass": True,
        "motion": {"duration_s": 90.0, "database": str(database.resolve())},
        "clearance": {"database": str(database.resolve())},
        "diagnostics": {
            "samples": 120,
            "source": str(diagnostics.resolve()),
        },
    }
    evaluation_path = run / "evaluation" / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation) + "\n", encoding="utf-8")
    source = describe_input(bag)
    database_entry = next(
        entry for entry in source["files"] if entry["path"] == "run.db3"
    )
    manifest = {
        "schema_version": 1,
        "profile": "release",
        "evidence_mode": "shadow_controller_with_recorded_motion",
        "git_commit": COMMIT,
        "git_dirty": False,
        "launch_errors": {},
        "returncodes": {
            "controller": -2,
            "record": -2,
            "play": 0,
            "evaluate": 0,
        },
        "gpu": [
            {
                "physical_index": "0",
                "name": "GPU A",
                "uuid": "GPU-a",
                "driver_version": "999",
                "memory_total_mib": "8192",
            }
        ],
        "input_bag": source,
        "evaluation_database": {
            "source": str(database.resolve()),
            "relative_path": "run.db3",
            "sha256": database_entry["sha256"],
        },
        "controller_config_sha256": sha256_file(config),
        "record_topics": list(REQUIRED_CUDANAV_OUTPUT_TOPICS),
        "required_output_topics": list(REQUIRED_CUDANAV_OUTPUT_TOPICS),
        "recording_identity": describe_input(recording),
        "diagnostics_sha256": sha256_file(diagnostics),
        "evaluation_sha256": sha256_file(evaluation_path),
        "commands": {
            "controller": [
                "controller",
                str(config.resolve()),
                str(diagnostics.resolve()),
            ],
            "play": ["ros2", "bag", "play", str(bag.resolve())],
            "evaluate": [
                "python",
                "evaluate.py",
                str(database.resolve()),
                str(diagnostics.resolve()),
            ],
            "record": [
                "ros2",
                "bag",
                "record",
                "--output",
                str(recording.resolve()),
                *REQUIRED_CUDANAV_OUTPUT_TOPICS,
            ],
        },
        "artifacts": {
            "evaluation": "evaluation/evaluation.json",
            "diagnostics": "diagnostics.csv",
            "controller_config": "controller.yaml",
            "controller_log": "controller.log",
            "play_log": "play.log",
            "recording": "recording",
        },
    }
    (run / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return manifest


def write_multi_gpu(root: Path) -> dict:
    devices = [
        {"index": "0", "name": "GPU A", "uuid": "GPU-a"},
        {"index": "0", "name": "GPU B", "uuid": "GPU-b"},
    ]
    runs = []
    for index, device in enumerate(devices):
        relative = f"gpu_{index:02d}/run_00"
        write_closed(
            root / relative,
            gpu_name=device["name"],
            gpu_uuid=device["uuid"],
            physical_index=device["index"],
        )
        runs.append(
            {
                "directory": relative,
                "returncode": 0,
                "device": device,
                "manifest_sha256": sha256_file(
                    root / relative / "manifest.json"
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "profile": "smoke",
        "devices": devices,
        "repetitions": 1,
        "minimum_gpu_devices": 2,
        "minimum_gpu_models": 2,
        "runs": runs,
    }
    (root / "multi_gpu_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return manifest


class CudaNavAutonomySuiteTest(unittest.TestCase):
    def make_suite(self, root: Path) -> tuple[dict, Path]:
        suite_root = root / "suite"
        closed = suite_root / "closed_loop"
        rosbag = suite_root / "real_rosbag"
        multi = suite_root / "multi_gpu"
        write_closed(closed)
        write_rosbag(rosbag)
        multi.mkdir()
        write_multi_gpu(multi)
        modes = {
            "closed_loop": {
                "directory": "closed_loop",
                "manifest_sha256": sha256_file(closed / "manifest.json"),
            },
            "real_rosbag_shadow": {
                "directory": "real_rosbag",
                "manifest_sha256": sha256_file(rosbag / "manifest.json"),
            },
            "multi_gpu": {
                "directory": "multi_gpu",
                "manifest_sha256": sha256_file(
                    multi / "multi_gpu_manifest.json"
                ),
            },
        }
        suite = {
            "schema_version": 1,
            "evidence_mode": "cudanav_autonomy_suite",
            "profile": "release",
            "git_commit": COMMIT,
            "required_modes": list(modes),
            "modes": modes,
            "passed": True,
        }
        return suite, suite_root

    def test_release_suite_revalidates_all_distinct_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, root = self.make_suite(Path(directory))
            result = evaluate_suite(suite, root)
            self.assertTrue(result["passed"], result)
            self.assertEqual(
                result["coverage"]["git_commits"], [COMMIT]
            )
            self.assertEqual(
                result["coverage"]["config_sha256"], [CONFIG_SHA]
            )

    def test_manifest_hash_and_shadow_label_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, root = self.make_suite(Path(directory))
            suite["modes"]["real_rosbag_shadow"]["manifest_sha256"] = "0" * 64
            result = evaluate_suite(suite, root)
            self.assertFalse(result["modes"]["real_rosbag_shadow"]["passed"])
            self.assertFalse(result["passed"])

    def test_derived_path_shadow_revalidates_dataset_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            manifest = write_rosbag(run)
            spec = read_json(DEFAULT_SPEC)
            source = root / "input_bag"
            source_metadata = [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
            ]
            for item in spec["recorded_inputs"].values():
                source_metadata.extend(
                    [
                        "    - topic_metadata:",
                        f"        name: {item['topic']}",
                        f"        type: {item['type']}",
                        "      message_count: 5",
                    ]
                )
            (source / "metadata.yaml").write_text(
                "\n".join(source_metadata) + "\n"
            )
            database = source / spec["acquisition"]["expected_database"]
            database.write_bytes(b"source database fixture")
            inspection = source / "inspection.json"
            inspection.write_text(
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
                        "topics": {
                            item["topic"]: {
                                "type": item["type"],
                                "count": 5,
                            }
                            for item in spec["recorded_inputs"].values()
                        },
                        "required_topic_checks": {
                            name: True for name in spec["recorded_inputs"]
                        },
                        "passed": True,
                    }
                )
                + "\n"
            )
            manifest["input_bag"] = describe_input(source)
            derived = root / "derived"
            derived.mkdir()
            contract = spec["path_derivation"]
            (derived / "metadata.yaml").write_text(
                "rosbag2_bagfile_information:\n"
                "  topics_with_message_count:\n"
                "    - topic_metadata:\n"
                f"        name: {contract['output_topic']}\n"
                f"        type: {contract['output_type']}\n"
                "      message_count: 1\n"
            )
            (derived / "path.mcap").write_bytes(b"derived path fixture")
            report = root / "generator.json"
            report.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "algorithm": contract["algorithm"],
                        "source_topic": contract["source_topic"],
                        "output_topic": contract["output_topic"],
                        "parameters": contract["parameters"],
                        "input_samples": 5,
                        "output_poses": 3,
                        "first_stamp_ns": 1,
                        "last_stamp_ns": 5,
                        "frame_id": "odom",
                        "recorded_path": False,
                        "closed_loop": False,
                    }
                )
                + "\n"
            )
            materialization = run / "dataset_materialization.json"
            materialization.write_text(
                json.dumps(
                    make_materialization(
                        DEFAULT_SPEC, source, derived, report, inspection
                    )
                )
                + "\n"
            )
            manifest["evidence_mode"] = (
                "real_sensor_shadow_with_derived_path"
            )
            evaluation_path = run / manifest["artifacts"]["evaluation"]
            evaluation = json.loads(evaluation_path.read_text())
            evaluation["evidence_mode"] = (
                "real_sensor_shadow_with_derived_path"
            )
            evaluation["clearance"].update(
                {
                    "pointcloud_topic": spec["quality_evaluation"][
                        "pointcloud_topic"
                    ],
                    "diagnostics_source": str(
                        (run / "diagnostics.csv").resolve()
                    ),
                    "filter": spec["quality_evaluation"]["filter"],
                }
            )
            evaluation_path.write_text(json.dumps(evaluation) + "\n")
            manifest["evaluation_sha256"] = sha256_file(evaluation_path)
            manifest["derived_path_bag"] = describe_input(derived)
            manifest["dataset_materialization_sha256"] = sha256_file(
                materialization
            )
            manifest["artifacts"]["dataset_materialization"] = (
                materialization.name
            )
            manifest["commands"]["play"] = [
                "ros2",
                "bag",
                "play",
                "-i",
                str(source.resolve()),
                "-i",
                str(derived.resolve()),
            ]
            quality_filter = spec["quality_evaluation"]["filter"]
            manifest["commands"]["evaluate"].extend(
                [
                    "--pointcloud-topic",
                    spec["recorded_inputs"]["pointcloud"]["topic"],
                    "--odometry-topic",
                    spec["recorded_inputs"]["odometry"]["topic"],
                    "--pointcloud-half-angle-rad",
                    str(quality_filter["half_angle_rad"]),
                    "--pointcloud-minimum-z-m",
                    str(quality_filter["minimum_z_m"]),
                    "--pointcloud-maximum-z-m",
                    str(quality_filter["maximum_z_m"]),
                    "--pointcloud-minimum-range-m",
                    str(quality_filter["minimum_range_m"]),
                    "--pointcloud-maximum-range-m",
                    str(quality_filter["maximum_range_m"]),
                    "--pointcloud-maximum-command-age-ms",
                    str(quality_filter["maximum_command_age_ms"]),
                ]
            )
            (run / "manifest.json").write_text(json.dumps(manifest) + "\n")
            result = validate_rosbag(run, "release")
            self.assertTrue(result["passed"], result)

    def test_closed_loop_bag_requires_messages_on_every_evidence_topic(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, root = self.make_suite(Path(directory))
            run = root / "closed_loop"
            metadata = run / "rosbag" / "metadata.yaml"
            metadata.write_text(
                metadata.read_text(encoding="utf-8").replace(
                    "      message_count: 10\n",
                    "      message_count: 0\n",
                    1,
                ),
                encoding="utf-8",
            )
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["rosbag_identity"] = describe_input(run / "rosbag")
            manifest_path.write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )
            suite["modes"]["closed_loop"]["manifest_sha256"] = sha256_file(
                manifest_path
            )

            result = evaluate_suite(suite, root)
            gate = result["modes"]["closed_loop"]["manifest_gate"]
            self.assertTrue(gate["checks"]["rosbag_content_unchanged"])
            self.assertFalse(gate["checks"]["required_bag_topic_messages"])
            self.assertFalse(result["passed"])

    def test_release_cannot_shrink_required_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, root = self.make_suite(Path(directory))
            suite["required_modes"] = ["closed_loop"]
            suite["modes"] = {"closed_loop": suite["modes"]["closed_loop"]}
            result = evaluate_suite(suite, root)
            self.assertFalse(result["checks"]["required_modes"])
            self.assertFalse(result["passed"])

    def test_runner_validation_records_are_json_serializable(self):
        with tempfile.TemporaryDirectory() as directory:
            suite, root = self.make_suite(Path(directory))
            records = {
                "closed": validate_closed_loop(root / "closed_loop", "release"),
                "rosbag": validate_rosbag(root / "real_rosbag", "release"),
                "multi": validate_multi_gpu(root / "multi_gpu"),
            }
            self.assertTrue(all(record["passed"] for record in records.values()))
            json.dumps(records, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
