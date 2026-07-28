#include "cuda_robotics_common/pointcloud_transform.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

TEST(PointCloudTransform, AppliesCompleteSe3)
{
  geometry_msgs::msg::Transform transform;
  const double half_angle = std::acos(-1.0) / 4.0;
  transform.rotation.z = std::sin(half_angle);
  transform.rotation.w = std::cos(half_angle);
  transform.translation.x = 3.0;
  transform.translation.y = -2.0;
  transform.translation.z = 0.5;
  const std::vector<float> xyz = {1.0f, 0.0f, 2.0f, 0.0f, 2.0f, -1.0f};

  const auto output = cuda_robotics_common::transform_xyz(xyz, transform);
  ASSERT_EQ(output.size(), xyz.size());
  EXPECT_NEAR(output[0], 3.0f, 1e-5f);
  EXPECT_NEAR(output[1], -1.0f, 1e-5f);
  EXPECT_NEAR(output[2], 2.5f, 1e-5f);
  EXPECT_NEAR(output[3], 1.0f, 1e-5f);
  EXPECT_NEAR(output[4], -2.0f, 1e-5f);
  EXPECT_NEAR(output[5], -0.5f, 1e-5f);
}

TEST(PointCloudTransform, NormalizesQuaternion)
{
  geometry_msgs::msg::Transform transform;
  transform.rotation.w = 2.0;
  const auto output =
    cuda_robotics_common::transform_xyz({1.0f, 2.0f, 3.0f}, transform);
  EXPECT_FLOAT_EQ(output[0], 1.0f);
  EXPECT_FLOAT_EQ(output[1], 2.0f);
  EXPECT_FLOAT_EQ(output[2], 3.0f);
}

TEST(PointCloudTransform, RejectsMalformedInput)
{
  geometry_msgs::msg::Transform transform;
  transform.rotation.w = 1.0;
  EXPECT_THROW(
    cuda_robotics_common::transform_xyz({1.0f, 2.0f}, transform),
    std::invalid_argument);

  transform.rotation.w = 0.0;
  EXPECT_THROW(
    cuda_robotics_common::transform_xyz({1.0f, 2.0f, 3.0f}, transform),
    std::invalid_argument);

  transform.rotation.w = 1.0;
  transform.translation.x = std::numeric_limits<double>::infinity();
  EXPECT_THROW(
    cuda_robotics_common::transform_xyz({1.0f, 2.0f, 3.0f}, transform),
    std::invalid_argument);
}
