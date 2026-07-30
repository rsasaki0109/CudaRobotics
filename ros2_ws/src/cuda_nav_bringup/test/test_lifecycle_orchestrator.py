from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

from cuda_nav_bringup.lifecycle_orchestrator import contains_transform


def test_contains_transform_matches_exact_frame_pair():
    transform = TransformStamped()
    transform.header.frame_id = "odom"
    transform.child_frame_id = "base_link"
    message = TFMessage(transforms=[transform])

    assert contains_transform(message, "odom", "base_link")
    assert not contains_transform(message, "map", "base_link")
    assert not contains_transform(message, "odom", "lidar")
