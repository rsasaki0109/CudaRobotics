#pragma once

#include <geometry_msgs/msg/pose.hpp>
#include <nav2_costmap_2d/costmap_layer.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

namespace cuda_voxel_costmap_layer {

struct OccupancyBridgeConfig
{
  int lethal_threshold = 50;
  bool unknown_is_free = false;
};

std::string validate_occupancy_grid(
  const nav_msgs::msg::OccupancyGrid & map,
  const std::string & expected_frame);

unsigned char occupancy_to_nav2_cost(
  std::int8_t value, const OccupancyBridgeConfig & config);

bool sample_occupancy_cost(
  const nav_msgs::msg::OccupancyGrid & map,
  double world_x,
  double world_y,
  const OccupancyBridgeConfig & config,
  unsigned char & cost);

bool inside_footprint_clearing_radius(
  double world_x, double world_y,
  double robot_x, double robot_y,
  double radius);

class CudaVoxelCostmapLayer : public nav2_costmap_2d::CostmapLayer
{
public:
  CudaVoxelCostmapLayer() = default;
  ~CudaVoxelCostmapLayer() override = default;

  void onInitialize() override;
  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;
  void reset() override;
  bool isClearable() override {return true;}

private:
  void incoming_map(nav_msgs::msg::OccupancyGrid::ConstSharedPtr message);
  void include_map_bounds(
    const nav_msgs::msg::OccupancyGrid & map,
    double * min_x, double * min_y, double * max_x, double * max_y) const;

  std::string occupancy_topic_;
  std::string global_frame_;
  OccupancyBridgeConfig bridge_config_;
  bool use_maximum_ = false;
  double max_map_age_sec_ = 0.0;
  double footprint_clearing_radius_ = 0.0;
  double robot_x_ = 0.0;
  double robot_y_ = 0.0;
  bool has_robot_pose_ = false;

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_subscription_;
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr map_;
  mutable std::mutex map_mutex_;
  bool input_valid_ = false;
  bool has_new_map_ = false;
  double pending_min_x_ = std::numeric_limits<double>::infinity();
  double pending_min_y_ = std::numeric_limits<double>::infinity();
  double pending_max_x_ = -std::numeric_limits<double>::infinity();
  double pending_max_y_ = -std::numeric_limits<double>::infinity();
};

}  // namespace cuda_voxel_costmap_layer
