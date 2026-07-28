#include "cuda_robotics_common/pointcloud_transform.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace cuda_robotics_common {

std::vector<float> transform_xyz(
  const std::vector<float> & xyz,
  const geometry_msgs::msg::Transform & target_from_source)
{
  if (xyz.size() % 3 != 0) {
    throw std::invalid_argument("XYZ storage size must be divisible by three");
  }
  const auto & q = target_from_source.rotation;
  const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (!(norm > 1e-9) || !std::isfinite(norm)) {
    throw std::invalid_argument("transform has an invalid quaternion");
  }
  const double x = q.x / norm;
  const double y = q.y / norm;
  const double z = q.z / norm;
  const double w = q.w / norm;
  const double r00 = 1.0 - 2.0 * (y * y + z * z);
  const double r01 = 2.0 * (x * y - z * w);
  const double r02 = 2.0 * (x * z + y * w);
  const double r10 = 2.0 * (x * y + z * w);
  const double r11 = 1.0 - 2.0 * (x * x + z * z);
  const double r12 = 2.0 * (y * z - x * w);
  const double r20 = 2.0 * (x * z - y * w);
  const double r21 = 2.0 * (y * z + x * w);
  const double r22 = 1.0 - 2.0 * (x * x + y * y);
  const auto & t = target_from_source.translation;
  if (!std::isfinite(t.x) || !std::isfinite(t.y) || !std::isfinite(t.z)) {
    throw std::invalid_argument("transform has a non-finite translation");
  }

  std::vector<float> transformed(xyz.size());
  for (std::size_t index = 0; index < xyz.size(); index += 3) {
    const double px = xyz[index];
    const double py = xyz[index + 1];
    const double pz = xyz[index + 2];
    transformed[index] =
      static_cast<float>(r00 * px + r01 * py + r02 * pz + t.x);
    transformed[index + 1] =
      static_cast<float>(r10 * px + r11 * py + r12 * pz + t.y);
    transformed[index + 2] =
      static_cast<float>(r20 * px + r21 * py + r22 * pz + t.z);
  }
  return transformed;
}

}  // namespace cuda_robotics_common
