from __future__ import annotations

from copy import deepcopy
import unittest

from publish_cudanav_rosbag_evidence import evaluate_evidence


class PortableRosbagEvidenceTest(unittest.TestCase):
    def test_release_contract_preserves_shadow_claim_boundary(self) -> None:
        payload = {
            "schema_version": 1,
            "status": "passed",
            "profile": "release",
            "source_commit": "a" * 40,
            "git_dirty": False,
            "claims": {
                "ros2_runtime": True,
                "real_sensor_data": True,
                "derived_recorded_path": True,
                "closed_loop": False,
                "commands_modify_recorded_motion": False,
            },
            "metrics": {
                "quality_pass": True,
                "diagnostics_duration_s": 90.0,
                "diagnostics_samples": 800,
                "command_pair_ratio": 1.0,
            },
            "output_recording": {
                "required_topic_messages": {
                    "/cuda_nav/cmd_vel": 800,
                    "/cuda_nav/odom": 800,
                    "/cuda_nav/occupancy": 800,
                    "/cuda_nav/esdf": 800,
                }
            },
            "gate": {"passed": True},
        }
        self.assertTrue(evaluate_evidence(payload)["valid"])
        relabelled = deepcopy(payload)
        relabelled["claims"]["closed_loop"] = True
        self.assertFalse(evaluate_evidence(relabelled)["valid"])


if __name__ == "__main__":
    unittest.main()
