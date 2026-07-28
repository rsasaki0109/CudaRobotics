#include "cuda_voxel_costmap_layer/cuda_voxel_costmap_layer.hpp"

#include <nav2_costmap_2d/cost_values.hpp>
#include <pluginlib/class_list_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cuda_voxel_costmap_layer {
namespace {

constexpr double kQuaternionTolerance = 1.0e-3;

bool finite_planar_origin(
  const geometry_msgs::msg::Pose & pose, double * yaw = nullptr)
{
  const auto & p = pose.position;
  const auto & q = pose.orientation;
  if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
    !std::isfinite(q.x) || !std::isfinite(q.y) ||
    !std::isfinite(q.z) || !std::isfinite(q.w))
  {
    return false;
  }
  const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (std::fabs(p.z) > kQuaternionTolerance ||
    norm <= 1.0e-9 || std::fabs(norm - 1.0) > kQuaternionTolerance ||
    std::fabs(q.x) > kQuaternionTolerance || std::fabs(q.y) > kQuaternionTolerance)
  {
    return false;
  }
  if (yaw) {
    *yaw = std::atan2(
      2.0 * (q.w * q.z + q.x * q.y),
      1.0 - 2.0 * (q.y * q.y + q.z * q.z));
  }
  return true;
}

void rotated_map_corner(
  const nav_msgs::msg::OccupancyGrid & map,
  double local_x,
  double local_y,
  double & world_x,
  double & world_y)
{
  double yaw = 0.0;
  finite_planar_origin(map.info.origin, &yaw);
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  world_x = map.info.origin.position.x + c * local_x - s * local_y;
  world_y = map.info.origin.position.y + s * local_x + c * local_y;
}

}  // namespace

std::string validate_occupancy_grid(
  const nav_msgs::msg::OccupancyGrid & map,
  const std::string & expected_frame)
{
  if (expected_frame.empty() || map.header.frame_id != expected_frame) {
    return "occupancy frame does not match the Nav2 global frame";
  }
  if (map.info.width == 0 || map.info.height == 0) {
    return "occupancy dimensions must be non-zero";
  }
  const std::size_t width = map.info.width;
  const std::size_t height = map.info.height;
  if (width > std::numeric_limits<std::size_t>::max() / height ||
    map.data.size() != width * height)
  {
    return "occupancy data size must equal width * height";
  }
  if (!(map.info.resolution > 0.0f) || !std::isfinite(map.info.resolution)) {
    return "occupancy resolution must be finite and positive";
  }
  if (!finite_planar_origin(map.info.origin)) {
    return "occupancy origin must be a finite planar unit-quaternion pose";
  }
  for (const std::int8_t value : map.data) {
    if (value < -1 || value > 100) {
      return "occupancy values must lie in [-1, 100]";
    }
  }
  return {};
}

unsigned char occupancy_to_nav2_cost(
  std::int8_t value, const OccupancyBridgeConfig & config)
{
  if (config.lethal_threshold <= 0 || config.lethal_threshold > 100) {
    throw std::invalid_argument("lethal_threshold must be in [1, 100]");
  }
  if (value < -1 || value > 100) {
    throw std::invalid_argument("occupancy value must lie in [-1, 100]");
  }
  if (value < 0) {
    return config.unknown_is_free ?
           nav2_costmap_2d::FREE_SPACE : nav2_costmap_2d::NO_INFORMATION;
  }
  if (value == 0) return nav2_costmap_2d::FREE_SPACE;
  if (value >= config.lethal_threshold) return nav2_costmap_2d::LETHAL_OBSTACLE;
  const double fraction =
    static_cast<double>(value) / static_cast<double>(config.lethal_threshold);
  const int scaled = 1 + static_cast<int>(
    std::lround(fraction * (nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE - 2)));
  return static_cast<unsigned char>(std::clamp(
    scaled, 1, static_cast<int>(nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) - 1));
}

bool sample_occupancy_cost(
  const nav_msgs::msg::OccupancyGrid & map,
  double world_x,
  double world_y,
  const OccupancyBridgeConfig & config,
  unsigned char & cost)
{
  double yaw = 0.0;
  if (!finite_planar_origin(map.info.origin, &yaw) ||
    !(map.info.resolution > 0.0f) || !std::isfinite(map.info.resolution))
  {
    return false;
  }
  const double dx = world_x - map.info.origin.position.x;
  const double dy = world_y - map.info.origin.position.y;
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  const double local_x = c * dx + s * dy;
  const double local_y = -s * dx + c * dy;
  const double map_x = std::floor(local_x / map.info.resolution);
  const double map_y = std::floor(local_y / map.info.resolution);
  if (!std::isfinite(map_x) || !std::isfinite(map_y) ||
    map_x < 0.0 || map_y < 0.0 ||
    map_x >= static_cast<double>(map.info.width) ||
    map_y >= static_cast<double>(map.info.height))
  {
    return false;
  }
  const std::size_t index =
    static_cast<std::size_t>(map_y) * map.info.width +
    static_cast<std::size_t>(map_x);
  if (index >= map.data.size()) return false;
  cost = occupancy_to_nav2_cost(map.data[index], config);
  return true;
}

void CudaVoxelCostmapLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) throw std::runtime_error("failed to lock Nav2 lifecycle node");
  global_frame_ = layered_costmap_->getGlobalFrameID();

  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("occupancy_topic", rclcpp::ParameterValue("occupancy"));
  declareParameter("lethal_threshold", rclcpp::ParameterValue(50));
  declareParameter("unknown_is_free", rclcpp::ParameterValue(false));
  declareParameter("use_maximum", rclcpp::ParameterValue(false));
  declareParameter("max_map_age_sec", rclcpp::ParameterValue(0.0));

  int threshold = 50;
  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".occupancy_topic", occupancy_topic_);
  node->get_parameter(name_ + ".lethal_threshold", threshold);
  node->get_parameter(name_ + ".unknown_is_free", bridge_config_.unknown_is_free);
  node->get_parameter(name_ + ".use_maximum", use_maximum_);
  node->get_parameter(name_ + ".max_map_age_sec", max_map_age_sec_);
  if (occupancy_topic_.empty() || occupancy_topic_.front() == '/') {
    throw std::invalid_argument("occupancy_topic must be a non-empty relative name");
  }
  if (threshold <= 0 || threshold > 100) {
    throw std::invalid_argument("lethal_threshold must be in [1, 100]");
  }
  if (!std::isfinite(max_map_age_sec_) || max_map_age_sec_ < 0.0) {
    throw std::invalid_argument("max_map_age_sec must be finite and non-negative");
  }
  bridge_config_.lethal_threshold = threshold;
  matchSize();
  current_ = false;
  map_subscription_ = node->create_subscription<nav_msgs::msg::OccupancyGrid>(
    occupancy_topic_, rclcpp::QoS(1).reliable().transient_local(),
    std::bind(&CudaVoxelCostmapLayer::incoming_map, this, std::placeholders::_1));
  RCLCPP_INFO(
    logger_, "CudaNav voxel layer subscribed to %s in frame %s",
    occupancy_topic_.c_str(), global_frame_.c_str());
}

void CudaVoxelCostmapLayer::incoming_map(
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr message)
{
  const std::string error = validate_occupancy_grid(*message, global_frame_);
  if (!error.empty()) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    input_valid_ = false;
    RCLCPP_ERROR(logger_, "Rejected CudaNav occupancy: %s", error.c_str());
    return;
  }
  std::lock_guard<std::mutex> lock(map_mutex_);
  if (map_) {
    include_map_bounds(
      *map_, &pending_min_x_, &pending_min_y_,
      &pending_max_x_, &pending_max_y_);
  }
  include_map_bounds(
    *message, &pending_min_x_, &pending_min_y_,
    &pending_max_x_, &pending_max_y_);
  map_ = std::move(message);
  input_valid_ = true;
  has_new_map_ = true;
}

void CudaVoxelCostmapLayer::include_map_bounds(
  const nav_msgs::msg::OccupancyGrid & map,
  double * min_x, double * min_y, double * max_x, double * max_y) const
{
  const double width = map.info.width * map.info.resolution;
  const double height = map.info.height * map.info.resolution;
  for (const auto & corner : {
      std::pair<double, double>{0.0, 0.0},
      {width, 0.0}, {0.0, height}, {width, height}})
  {
    double x = 0.0;
    double y = 0.0;
    rotated_map_corner(map, corner.first, corner.second, x, y);
    *min_x = std::min(*min_x, x);
    *min_y = std::min(*min_y, y);
    *max_x = std::max(*max_x, x);
    *max_y = std::max(*max_y, y);
  }
}

void CudaVoxelCostmapLayer::updateBounds(
  double, double, double,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) return;
  std::lock_guard<std::mutex> lock(map_mutex_);
  if (!map_ || !input_valid_) {
    current_ = false;
    return;
  }
  if (max_map_age_sec_ > 0.0) {
    const rclcpp::Time stamp(map_->header.stamp, clock_->get_clock_type());
    const double age = (clock_->now() - stamp).seconds();
    if (stamp.nanoseconds() == 0 || age > max_map_age_sec_ || age < -0.05) {
      current_ = false;
      return;
    }
  }
  current_ = true;
  if (has_new_map_) {
    *min_x = std::min(*min_x, pending_min_x_);
    *min_y = std::min(*min_y, pending_min_y_);
    *max_x = std::max(*max_x, pending_max_x_);
    *max_y = std::max(*max_y, pending_max_y_);
  }
  include_map_bounds(*map_, min_x, min_y, max_x, max_y);
  has_new_map_ = false;
  pending_min_x_ = pending_min_y_ = std::numeric_limits<double>::infinity();
  pending_max_x_ = pending_max_y_ = -std::numeric_limits<double>::infinity();
}

void CudaVoxelCostmapLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) return;
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr map;
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    if (!current_) return;
    map = map_;
  }
  if (!map) return;
  const int start_x = std::max(0, min_i);
  const int start_y = std::max(0, min_j);
  const int end_x = std::min(max_i, static_cast<int>(master_grid.getSizeInCellsX()));
  const int end_y = std::min(max_j, static_cast<int>(master_grid.getSizeInCellsY()));
  for (int y = start_y; y < end_y; ++y) {
    for (int x = start_x; x < end_x; ++x) {
      double world_x = 0.0;
      double world_y = 0.0;
      master_grid.mapToWorld(
        static_cast<unsigned int>(x), static_cast<unsigned int>(y),
        world_x, world_y);
      unsigned char layer_cost = nav2_costmap_2d::NO_INFORMATION;
      if (!sample_occupancy_cost(
          *map, world_x, world_y, bridge_config_, layer_cost))
      {
        continue;
      }
      if (!use_maximum_) {
        master_grid.setCost(x, y, layer_cost);
      } else if (layer_cost != nav2_costmap_2d::NO_INFORMATION) {
        const unsigned char master_cost = master_grid.getCost(x, y);
        master_grid.setCost(
          x, y,
          master_cost == nav2_costmap_2d::NO_INFORMATION ?
          layer_cost : std::max(master_cost, layer_cost));
      }
    }
  }
}

void CudaVoxelCostmapLayer::reset()
{
  std::lock_guard<std::mutex> lock(map_mutex_);
  map_.reset();
  input_valid_ = false;
  has_new_map_ = false;
  pending_min_x_ = pending_min_y_ = std::numeric_limits<double>::infinity();
  pending_max_x_ = pending_max_y_ = -std::numeric_limits<double>::infinity();
  current_ = false;
}

}  // namespace cuda_voxel_costmap_layer

PLUGINLIB_EXPORT_CLASS(
  cuda_voxel_costmap_layer::CudaVoxelCostmapLayer,
  nav2_costmap_2d::Layer)
