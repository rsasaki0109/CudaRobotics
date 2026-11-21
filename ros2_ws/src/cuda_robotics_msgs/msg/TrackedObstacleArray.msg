# Moving objects from a tracker, all observed at header.stamp in
# header.frame_id. Consumers extrapolate them with their velocity to the
# time they plan for.

std_msgs/Header header
TrackedObstacle[] obstacles
