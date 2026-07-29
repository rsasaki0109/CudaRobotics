#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import unittest

from v1_support_matrix import evaluate, load


class V1SupportMatrixTest(unittest.TestCase):
    def test_current_development_contract_is_valid_but_not_ready(self) -> None:
        result = evaluate(load())
        self.assertTrue(result["valid"], result)
        self.assertFalse(result["ready"])
        self.assertFalse(result["readiness"]["release_status"])
        self.assertFalse(result["readiness"]["python_at_target"])

    def test_declared_component_version_must_match_source(self) -> None:
        matrix = deepcopy(load())
        matrix["surfaces"]["ros2"]["package_versions"][
            "cuda_kiss_icp"
        ] = "1.0.0"
        result = evaluate(matrix)
        self.assertFalse(result["checks"]["ros_versions"])
        self.assertFalse(result["valid"])

    def test_fifteen_minute_budget_cannot_be_weakened(self) -> None:
        matrix = deepcopy(load())
        matrix["main_demo"]["time_budget_seconds"] = 1800
        result = evaluate(matrix)
        self.assertFalse(result["checks"]["time_budget"])
        self.assertFalse(result["valid"])

    def test_main_demo_command_must_exist_on_every_user_surface(self) -> None:
        matrix = deepcopy(load())
        matrix["main_demo"]["run_command"] = "docker run imaginary"
        result = evaluate(matrix)
        self.assertFalse(result["checks"]["main_command"])
        self.assertFalse(result["valid"])

    def test_current_colab_master_link_is_not_release_ready(self) -> None:
        result = evaluate(load())
        self.assertFalse(result["readiness"]["colab_target_ref"])


if __name__ == "__main__":
    unittest.main()
