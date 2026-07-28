#pragma once

#include <geometry_msgs/msg/transform.hpp>

#include <vector>

namespace cuda_robotics_common {

// Applies target_T_source to tightly packed XYZ points. The quaternion is
// normalized before use; malformed XYZ storage or transforms are rejected.
std::vector<float> transform_xyz(
  const std::vector<float> & xyz,
  const geometry_msgs::msg::Transform & target_from_source);

}  // namespace cuda_robotics_common
