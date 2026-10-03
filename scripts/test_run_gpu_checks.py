#!/usr/bin/env python3

from __future__ import annotations

import unittest

from run_gpu_checks import parse_junit, required_targets


JUNIT = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="(empty)" tests="3" failures="1">
  <testcase name="test_autodiff" classname="test_autodiff" time="1.25" status="run">
    <system-out>ALL TESTS PASSED
</system-out>
  </testcase>
  <testcase name="gpu_planner_showdown_gate" classname="gpu_planner_showdown_gate" time="44.28" status="fail">
    <failure message="Failed"/>
    <system-out>Showdown target check: FAIL
</system-out>
  </testcase>
  <testcase name="optional_gate" classname="optional_gate" time="0" status="notrun">
    <skipped message="Disabled"/>
  </testcase>
</testsuite>
"""


class RunGpuChecksTest(unittest.TestCase):
    def test_parse_junit(self) -> None:
        results = parse_junit(JUNIT)
        self.assertEqual([r["name"] for r in results],
                         ["test_autodiff", "gpu_planner_showdown_gate", "optional_gate"])
        self.assertEqual([r["status"] for r in results], ["passed", "failed", "not run"])
        self.assertAlmostEqual(results[1]["seconds"], 44.28)
        self.assertIn("Showdown target check: FAIL", results[1]["output"])

    def test_required_targets(self) -> None:
        tests = [
            {"command": ["C:/repo/bin/Release/test_autodiff.exe"]},
            {"command": ["/repo/bin/gpu_esdf_mppi_3d", "--movers", "6"]},
            {"command": ["/repo/bin/gpu_esdf_mppi_3d"]},
            {"command": ["/usr/bin/python3", "scripts/check.py"]},
            {},
        ]
        self.assertEqual(required_targets(tests), ["test_autodiff", "gpu_esdf_mppi_3d"])


if __name__ == "__main__":
    unittest.main()
