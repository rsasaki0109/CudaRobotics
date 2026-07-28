#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <cstddef>
#include <vector>

namespace cuda_robotics_common {

struct DecodedPointCloud {
  std::vector<float> xyz;
  std::size_t skipped_non_finite = 0;
};

// Decodes named x/y/z FLOAT32 or FLOAT64 fields while respecting point_step,
// row_step, organized-cloud padding, and message endianness.
DecodedPointCloud decode_xyz(const sensor_msgs::msg::PointCloud2 & message);

}  // namespace cuda_robotics_common
