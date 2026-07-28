#include "cuda_voxel_costmap_layer/cuda_voxel_costmap_layer.hpp"

#include <gtest/gtest.h>
#include <nav2_costmap_2d/cost_values.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace {

nav_msgs::msg::OccupancyGrid make_map()
{
  nav_msgs::msg::OccupancyGrid map;
  map.header.frame_id = "odom";
  map.info.width = 3;
  map.info.height = 2;
  map.info.resolution = 0.5f;
  map.info.origin.position.x = 2.0;
  map.info.origin.position.y = -1.0;
  const double half_yaw = 0.25 * std::acos(-1.0);
  map.info.origin.orientation.z = std::sin(half_yaw);
  map.info.origin.orientation.w = std::cos(half_yaw);
  map.data = {0, 49, 50, -1, 100, 0};
  return map;
}

}  // namespace

TEST(OccupancyBridge, ValidatesShapeFrameAndValues)
{
  auto map = make_map();
  EXPECT_TRUE(
    cuda_voxel_costmap_layer::validate_occupancy_grid(map, "odom").empty());
  EXPECT_FALSE(
    cuda_voxel_costmap_layer::validate_occupancy_grid(map, "map").empty());
  map.data.pop_back();
  EXPECT_FALSE(
    cuda_voxel_costmap_layer::validate_occupancy_grid(map, "odom").empty());
}

TEST(OccupancyBridge, ConvertsStandardOccupancySemantics)
{
  cuda_voxel_costmap_layer::OccupancyBridgeConfig config;
  config.lethal_threshold = 50;
  EXPECT_EQ(
    cuda_voxel_costmap_layer::occupancy_to_nav2_cost(-1, config),
    nav2_costmap_2d::NO_INFORMATION);
  EXPECT_EQ(
    cuda_voxel_costmap_layer::occupancy_to_nav2_cost(0, config),
    nav2_costmap_2d::FREE_SPACE);
  EXPECT_LT(
    cuda_voxel_costmap_layer::occupancy_to_nav2_cost(49, config),
    nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE);
  EXPECT_EQ(
    cuda_voxel_costmap_layer::occupancy_to_nav2_cost(50, config),
    nav2_costmap_2d::LETHAL_OBSTACLE);
  EXPECT_THROW(
    cuda_voxel_costmap_layer::occupancy_to_nav2_cost(-2, config),
    std::invalid_argument);
}

TEST(OccupancyBridge, SamplesRotatedMapAtCellCenters)
{
  const auto map = make_map();
  cuda_voxel_costmap_layer::OccupancyBridgeConfig config;
  unsigned char cost = 0;
  // A +90 degree map yaw maps source cell (2, 0) center to (1.75, 0.25).
  ASSERT_TRUE(cuda_voxel_costmap_layer::sample_occupancy_cost(
      map, 1.75, 0.25, config, cost));
  EXPECT_EQ(cost, nav2_costmap_2d::LETHAL_OBSTACLE);
  EXPECT_FALSE(cuda_voxel_costmap_layer::sample_occupancy_cost(
      map, 20.0, 20.0, config, cost));
}
