#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "cuda_nav_bringup"))

from run_cudanav_gpu_closed_loop import CLAIMS, SCENARIO, STAGES, evaluate_result
from cuda_nav_bringup.simulation_geometry import default_segments, mission_waypoints


def passing_result() -> dict:
    return {
        "scenario": "cudanav_s_course",
        "stages": list(STAGES),
        "claims": dict(CLAIMS),
        "goal_reached": True,
        "ground_truth_goal_distance_m": 0.2,
        "collision_count": 0,
        "odometry_drift_percent": 1.0,
        "command_deadline_miss_rate": 0.0,
        "causal_command_effect": True,
        "command_effect_distance_m": 8.0,
        "invalid_commands": 0,
        "minimum_inliers": 100,
        "final_observed_voxels": 1000,
        "maximum_occupied_cells": 100,
        "all_colliding_evaluations": 0,
        "minimum_nonzero_valid_rollout_ratio": 0.1,
        "ground_truth_distance_m": 10.0,
        "frames": 220,
        "gpu": {"name": "GPU fixture", "driver_version": 12000},
        "quality_pass": True,
    }


class CudaNavGpuClosedLoopTest(unittest.TestCase):
    def test_native_scenario_matches_ros_loopback_contract(self) -> None:
        self.assertEqual(
            [list(value) for value in mission_waypoints()],
            SCENARIO["waypoints"],
        )
        self.assertEqual(len(default_segments()), 12)
        self.assertEqual(SCENARIO["robot_radius_m"], 0.24)
        self.assertEqual(SCENARIO["ray_count"], 240)
        self.assertEqual(SCENARIO["z_levels_m"], [-0.45, 0.0, 0.45])

    def test_accepts_complete_causal_closed_loop_result(self) -> None:
        checks = evaluate_result(passing_result())
        self.assertTrue(all(checks.values()), checks)

    def test_rejects_shadow_relabel_and_safety_regressions(self) -> None:
        for key, value in (
            ("causal_command_effect", False),
            ("collision_count", 1),
            ("ground_truth_goal_distance_m", 0.31),
            ("odometry_drift_percent", 5.0),
        ):
            result = deepcopy(passing_result())
            result[key] = value
            self.assertFalse(all(evaluate_result(result).values()), key)
        result = deepcopy(passing_result())
        result["claims"]["ros2_runtime"] = True
        self.assertFalse(evaluate_result(result)["claims"])


if __name__ == "__main__":
    unittest.main()
