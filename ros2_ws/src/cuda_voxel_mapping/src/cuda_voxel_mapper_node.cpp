#include "cuda_voxel_mapping/cuda_voxel_mapper_node.hpp"

#include "cuda_robotics_common/pointcloud_decoder.hpp"
#include "cuda_robotics_common/pointcloud_transform.hpp"

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <lifecycle_msgs/msg/transition.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2/exceptions.h>

#include <climits>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cuda_voxel_mapping {
namespace {

using CallbackReturn =
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

diagnostic_msgs::msg::KeyValue key_value(const std::string & key, const std::string & value)
{
  diagnostic_msgs::msg::KeyValue output;
  output.key = key;
  output.value = value;
  return output;
}

}  // namespace

CudaVoxelMapperNode::CudaVoxelMapperNode(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("cuda_voxel_mapper", options)
{
  declare_parameter("input_topic", "points");
  declare_parameter("occupancy_topic", "occupancy");
  declare_parameter("local_map_topic", "local_map");
  declare_parameter("diagnostics_topic", "diagnostics");
  declare_parameter("odom_frame", "odom");
  declare_parameter("expected_sensor_frame", "");
  declare_parameter("transform_timeout_sec", 0.1);
  declare_parameter("max_scan_age_sec", 0.5);
  declare_parameter("max_future_stamp_sec", 0.05);
  declare_parameter("qos_depth", 5);
  declare_parameter("local_map_publish_stride", 10);
  declare_parameter("width", 256);
  declare_parameter("height", 256);
  declare_parameter("depth", 32);
  declare_parameter("resolution", 0.10);
  declare_parameter("origin_z", -1.0);
  declare_parameter("min_range", 0.10);
  declare_parameter("max_range", 20.0);
  declare_parameter("log_odds_occupied", 0.85);
  declare_parameter("log_odds_free", -0.40);
  declare_parameter("log_odds_min", -4.0);
  declare_parameter("log_odds_max", 4.0);
  declare_parameter("occupied_threshold", 0.0);
  declare_parameter("projection_min_z", -1000.0);
  declare_parameter("projection_max_z", 1000.0);
  declare_parameter("rolling_margin_cells", 48);
  declare_parameter("max_scan_points", 200000);
}

CallbackReturn CudaVoxelMapperNode::on_configure(const rclcpp_lifecycle::State &)
{
  std::string error;
  if (!read_and_validate_parameters(error)) {
    RCLCPP_ERROR(get_logger(), "configuration rejected: %s", error.c_str());
    return CallbackReturn::FAILURE;
  }
  try {
    occupancy_publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
      occupancy_topic_, rclcpp::QoS(1).reliable().transient_local());
    local_map_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      local_map_topic_, rclcpp::QoS(1).reliable().durability_volatile());
    diagnostic_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      diagnostics_topic_, rclcpp::QoS(10).reliable().durability_volatile());
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(
      *tf_buffer_, get_node_base_interface(), get_node_logging_interface(),
      get_node_parameters_interface(), get_node_topics_interface(), true);
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_logger(), "configure failed: %s", exception.what());
    release_runtime();
    return CallbackReturn::FAILURE;
  }
  runtime_fault_ = false;
  return CallbackReturn::SUCCESS;
}

CallbackReturn CudaVoxelMapperNode::on_activate(const rclcpp_lifecycle::State &)
{
  if (runtime_fault_) {
    RCLCPP_ERROR(
      get_logger(), "cleanup and reconfigure are required after a runtime fault");
    return CallbackReturn::FAILURE;
  }
  try {
    mapper_ = std::make_unique<cudarobotics::VoxelMapperGpu>(mapping_config_);
    occupancy_publisher_->on_activate();
    local_map_publisher_->on_activate();
    diagnostic_publisher_->on_activate();
    points_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS().keep_last(qos_depth_),
      std::bind(&CudaVoxelMapperNode::pointcloud_callback, this, std::placeholders::_1));
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_logger(), "activation failed: %s", exception.what());
    release_runtime();
    return CallbackReturn::ERROR;
  }
  has_last_stamp_ = false;
  has_map_load_time_ = false;
  scans_received_ = scans_integrated_ = scans_dropped_ = 0;
  transform_failures_ = invalid_clouds_ = non_finite_points_ = capacity_failures_ = 0;
  publish_diagnostic(
    diagnostic_msgs::msg::DiagnosticStatus::OK, "active", get_clock()->now());
  return CallbackReturn::SUCCESS;
}

CallbackReturn CudaVoxelMapperNode::on_deactivate(const rclcpp_lifecycle::State &)
{
  points_subscription_.reset();
  if (diagnostic_publisher_ && diagnostic_publisher_->is_activated()) {
    publish_diagnostic(
      runtime_fault_ ? diagnostic_msgs::msg::DiagnosticStatus::ERROR :
      diagnostic_msgs::msg::DiagnosticStatus::WARN,
      runtime_fault_ ? "deactivated after runtime fault" : "inactive", get_clock()->now());
  }
  if (occupancy_publisher_) occupancy_publisher_->on_deactivate();
  if (local_map_publisher_) local_map_publisher_->on_deactivate();
  if (diagnostic_publisher_) diagnostic_publisher_->on_deactivate();
  mapper_.reset();
  return CallbackReturn::SUCCESS;
}

CallbackReturn CudaVoxelMapperNode::on_cleanup(const rclcpp_lifecycle::State &)
{
  release_runtime();
  runtime_fault_ = false;
  return CallbackReturn::SUCCESS;
}

CallbackReturn CudaVoxelMapperNode::on_shutdown(const rclcpp_lifecycle::State &)
{
  release_runtime();
  return CallbackReturn::SUCCESS;
}

CallbackReturn CudaVoxelMapperNode::on_error(const rclcpp_lifecycle::State &)
{
  RCLCPP_ERROR(get_logger(), "entering lifecycle error processing");
  release_runtime();
  return CallbackReturn::SUCCESS;
}

bool CudaVoxelMapperNode::read_and_validate_parameters(std::string & error)
{
  input_topic_ = get_parameter("input_topic").as_string();
  occupancy_topic_ = get_parameter("occupancy_topic").as_string();
  local_map_topic_ = get_parameter("local_map_topic").as_string();
  diagnostics_topic_ = get_parameter("diagnostics_topic").as_string();
  odom_frame_ = get_parameter("odom_frame").as_string();
  expected_sensor_frame_ = get_parameter("expected_sensor_frame").as_string();
  transform_timeout_sec_ = get_parameter("transform_timeout_sec").as_double();
  max_scan_age_sec_ = get_parameter("max_scan_age_sec").as_double();
  max_future_stamp_sec_ = get_parameter("max_future_stamp_sec").as_double();
  const auto qos_depth = get_parameter("qos_depth").as_int();
  const auto publish_stride = get_parameter("local_map_publish_stride").as_int();
  const auto width = get_parameter("width").as_int();
  const auto height = get_parameter("height").as_int();
  const auto depth = get_parameter("depth").as_int();
  const auto rolling_margin = get_parameter("rolling_margin_cells").as_int();
  const auto max_scan_points = get_parameter("max_scan_points").as_int();
  if (qos_depth <= 0 || qos_depth > INT_MAX ||
    publish_stride <= 0 || publish_stride > INT_MAX ||
    width <= 0 || width > INT_MAX || height <= 0 || height > INT_MAX ||
    depth <= 0 || depth > INT_MAX ||
    rolling_margin <= 0 || rolling_margin > INT_MAX ||
    max_scan_points <= 0)
  {
    error = "integer mapping parameters must be positive and in range";
    return false;
  }
  qos_depth_ = static_cast<int>(qos_depth);
  local_map_publish_stride_ = static_cast<int>(publish_stride);
  mapping_config_.width = static_cast<int>(width);
  mapping_config_.height = static_cast<int>(height);
  mapping_config_.depth = static_cast<int>(depth);
  mapping_config_.rolling_margin_cells = static_cast<int>(rolling_margin);
  mapping_config_.max_scan_points = static_cast<std::size_t>(max_scan_points);
  mapping_config_.resolution =
    static_cast<float>(get_parameter("resolution").as_double());
  mapping_config_.origin_z = static_cast<float>(get_parameter("origin_z").as_double());
  mapping_config_.min_range =
    static_cast<float>(get_parameter("min_range").as_double());
  mapping_config_.max_range =
    static_cast<float>(get_parameter("max_range").as_double());
  mapping_config_.log_odds_occupied =
    static_cast<float>(get_parameter("log_odds_occupied").as_double());
  mapping_config_.log_odds_free =
    static_cast<float>(get_parameter("log_odds_free").as_double());
  mapping_config_.log_odds_min =
    static_cast<float>(get_parameter("log_odds_min").as_double());
  mapping_config_.log_odds_max =
    static_cast<float>(get_parameter("log_odds_max").as_double());
  mapping_config_.occupied_threshold =
    static_cast<float>(get_parameter("occupied_threshold").as_double());
  mapping_config_.projection_min_z =
    static_cast<float>(get_parameter("projection_min_z").as_double());
  mapping_config_.projection_max_z =
    static_cast<float>(get_parameter("projection_max_z").as_double());

  if (input_topic_.empty() || occupancy_topic_.empty() ||
    local_map_topic_.empty() || diagnostics_topic_.empty())
  {
    error = "topic names must be non-empty";
    return false;
  }
  if (input_topic_.front() == '/' || occupancy_topic_.front() == '/' ||
    local_map_topic_.front() == '/' || diagnostics_topic_.front() == '/')
  {
    error = "input and output topics must be relative";
    return false;
  }
  if (odom_frame_.empty() || odom_frame_.front() == '/' ||
    (!expected_sensor_frame_.empty() && expected_sensor_frame_.front() == '/'))
  {
    error = "frame names must be non-empty and must not begin with '/'";
    return false;
  }
  if (!std::isfinite(transform_timeout_sec_) || !std::isfinite(max_scan_age_sec_) ||
    !std::isfinite(max_future_stamp_sec_) ||
    transform_timeout_sec_ < 0.0 || max_scan_age_sec_ < 0.0 ||
    max_future_stamp_sec_ < 0.0)
  {
    error = "timestamp and transform limits must be finite and non-negative";
    return false;
  }
  error = cudarobotics::validate_voxel_mapping_config(mapping_config_);
  return error.empty();
}

void CudaVoxelMapperNode::pointcloud_callback(
  sensor_msgs::msg::PointCloud2::ConstSharedPtr message)
{
  ++scans_received_;
  const rclcpp::Time stamp(message->header.stamp, get_clock()->get_clock_type());
  if (message->header.frame_id.empty() ||
    (!expected_sensor_frame_.empty() &&
    message->header.frame_id != expected_sensor_frame_))
  {
    ++invalid_clouds_;
    ++scans_dropped_;
    publish_diagnostic(
      diagnostic_msgs::msg::DiagnosticStatus::WARN,
      "dropped cloud with invalid sensor frame", stamp);
    return;
  }
  if (stamp.nanoseconds() == 0 || (has_last_stamp_ && stamp <= last_stamp_)) {
    ++scans_dropped_;
    publish_diagnostic(
      diagnostic_msgs::msg::DiagnosticStatus::WARN,
      "dropped zero or non-monotonic sensor timestamp", stamp);
    return;
  }
  const double age = (get_clock()->now() - stamp).seconds();
  if ((max_scan_age_sec_ > 0.0 && age > max_scan_age_sec_) ||
    age < -max_future_stamp_sec_)
  {
    ++scans_dropped_;
    publish_diagnostic(
      diagnostic_msgs::msg::DiagnosticStatus::WARN,
      "dropped stale or future-dated cloud", stamp);
    return;
  }
  const std::size_t declared_points =
    static_cast<std::size_t>(message->width) * message->height;
  if (declared_points > mapping_config_.max_scan_points) {
    ++capacity_failures_;
    ++scans_dropped_;
    handle_fatal_error("declared cloud exceeds max_scan_points", stamp);
    return;
  }

  try {
    auto decoded = cuda_robotics_common::decode_xyz(*message);
    non_finite_points_ += decoded.skipped_non_finite;
    if (decoded.xyz.empty()) {
      ++invalid_clouds_;
      ++scans_dropped_;
      publish_diagnostic(
        diagnostic_msgs::msg::DiagnosticStatus::WARN,
        "dropped cloud without finite points", stamp);
      return;
    }
    const auto transform = tf_buffer_->lookupTransform(
      odom_frame_, message->header.frame_id, stamp,
      rclcpp::Duration::from_seconds(transform_timeout_sec_));
    std::vector<float> points_world;
    try {
      points_world =
        cuda_robotics_common::transform_xyz(decoded.xyz, transform.transform);
    } catch (const std::invalid_argument & exception) {
      throw tf2::TransformException(exception.what());
    }
    const float sensor_origin[3] = {
      static_cast<float>(transform.transform.translation.x),
      static_cast<float>(transform.transform.translation.y),
      static_cast<float>(transform.transform.translation.z)};
    const auto stats = mapper_->integrate_scan(points_world, sensor_origin);
    const auto projection = mapper_->occupancy_projection();
    if (!has_map_load_time_) {
      map_load_time_ = stamp;
      has_map_load_time_ = true;
    }
    publish_occupancy(projection, stamp);
    ++scans_integrated_;
    if (scans_integrated_ == 1 ||
      scans_integrated_ % static_cast<std::size_t>(local_map_publish_stride_) == 0)
    {
      publish_local_map(stamp);
    }
    last_stamp_ = stamp;
    has_last_stamp_ = true;
    publish_diagnostic(
      diagnostic_msgs::msg::DiagnosticStatus::OK, "mapping", stamp, &stats);
  } catch (const tf2::TransformException & exception) {
    ++transform_failures_;
    ++scans_dropped_;
    publish_diagnostic(
      diagnostic_msgs::msg::DiagnosticStatus::WARN,
      std::string("transform lookup failed: ") + exception.what(), stamp);
  } catch (const std::invalid_argument & exception) {
    ++invalid_clouds_;
    ++scans_dropped_;
    publish_diagnostic(
      diagnostic_msgs::msg::DiagnosticStatus::WARN,
      std::string("invalid cloud: ") + exception.what(), stamp);
  } catch (const std::length_error & exception) {
    ++capacity_failures_;
    ++scans_dropped_;
    handle_fatal_error(exception.what(), stamp);
  } catch (const std::exception & exception) {
    ++scans_dropped_;
    handle_fatal_error(exception.what(), stamp);
  }
}

void CudaVoxelMapperNode::publish_occupancy(
  const cudarobotics::OccupancyProjection & projection,
  const rclcpp::Time & stamp)
{
  nav_msgs::msg::OccupancyGrid message;
  message.header.stamp = stamp;
  message.header.frame_id = odom_frame_;
  message.info.map_load_time = map_load_time_;
  message.info.resolution = projection.grid.resolution;
  message.info.width = static_cast<std::uint32_t>(projection.grid.width);
  message.info.height = static_cast<std::uint32_t>(projection.grid.height);
  message.info.origin.position.x = projection.grid.origin_x;
  message.info.origin.position.y = projection.grid.origin_y;
  message.info.origin.position.z = projection.grid.origin_z;
  message.info.origin.orientation.w = 1.0;
  message.data = projection.data;
  occupancy_publisher_->publish(message);
}

void CudaVoxelMapperNode::publish_local_map(const rclcpp::Time & stamp)
{
  const auto snapshot = mapper_->snapshot();
  std::size_t occupied_count = 0;
  for (std::size_t index = 0; index < snapshot.log_odds.size(); ++index) {
    if (snapshot.observed[index] &&
      snapshot.log_odds[index] >= mapping_config_.occupied_threshold)
    {
      ++occupied_count;
    }
  }
  sensor_msgs::msg::PointCloud2 message;
  message.header.stamp = stamp;
  message.header.frame_id = odom_frame_;
  sensor_msgs::PointCloud2Modifier modifier(message);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(occupied_count);
  sensor_msgs::PointCloud2Iterator<float> x_iterator(message, "x");
  sensor_msgs::PointCloud2Iterator<float> y_iterator(message, "y");
  sensor_msgs::PointCloud2Iterator<float> z_iterator(message, "z");
  for (std::size_t index = 0; index < snapshot.log_odds.size(); ++index) {
    if (!snapshot.observed[index] ||
      snapshot.log_odds[index] < mapping_config_.occupied_threshold)
    {
      continue;
    }
    const int x = static_cast<int>(index % snapshot.grid.width);
    const int yz = static_cast<int>(index / snapshot.grid.width);
    const int y = yz % snapshot.grid.height;
    const int z = yz / snapshot.grid.height;
    *x_iterator = snapshot.grid.origin_x + (x + 0.5f) * snapshot.grid.resolution;
    *y_iterator = snapshot.grid.origin_y + (y + 0.5f) * snapshot.grid.resolution;
    *z_iterator = snapshot.grid.origin_z + (z + 0.5f) * snapshot.grid.resolution;
    ++x_iterator;
    ++y_iterator;
    ++z_iterator;
  }
  message.is_dense = true;
  local_map_publisher_->publish(message);
}

void CudaVoxelMapperNode::publish_diagnostic(
  std::uint8_t level, const std::string & message, const rclcpp::Time & stamp,
  const cudarobotics::VoxelMappingStats * stats)
{
  if (!diagnostic_publisher_ || !diagnostic_publisher_->is_activated()) return;
  diagnostic_msgs::msg::DiagnosticArray array;
  array.header.stamp = stamp;
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.level = level;
  status.name = get_fully_qualified_name() + std::string(": voxel mapping");
  status.hardware_id = "cuda";
  status.message = message;
  status.values = {
    key_value("scans_received", std::to_string(scans_received_)),
    key_value("scans_integrated", std::to_string(scans_integrated_)),
    key_value("scans_dropped", std::to_string(scans_dropped_)),
    key_value("transform_failures", std::to_string(transform_failures_)),
    key_value("invalid_clouds", std::to_string(invalid_clouds_)),
    key_value("non_finite_points", std::to_string(non_finite_points_)),
    key_value("capacity_failures", std::to_string(capacity_failures_)),
    key_value("unknown_value", "-1"),
    key_value("free_value", "0"),
    key_value("occupied_value", "100"),
    key_value("runtime_fault", runtime_fault_ ? "true" : "false")};
  if (stats) {
    status.values.push_back(
      key_value("integrated_rays", std::to_string(stats->integrated_rays)));
    status.values.push_back(
      key_value("observed_voxels", std::to_string(stats->observed_voxels)));
    status.values.push_back(
      key_value("grid_shifted", stats->grid_shifted ? "true" : "false"));
    status.values.push_back(
      key_value("raycast_ms", std::to_string(stats->raycast_ms)));
  }
  array.status.push_back(std::move(status));
  diagnostic_publisher_->publish(array);
}

void CudaVoxelMapperNode::handle_fatal_error(
  const std::string & message, const rclcpp::Time & stamp)
{
  runtime_fault_ = true;
  RCLCPP_ERROR(get_logger(), "fatal voxel mapping error: %s", message.c_str());
  publish_diagnostic(
    diagnostic_msgs::msg::DiagnosticStatus::ERROR,
    std::string("fatal voxel mapping error: ") + message, stamp);
  points_subscription_.reset();
  try {
    trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_logger(), "failed to deactivate after fatal error: %s", exception.what());
  }
}

void CudaVoxelMapperNode::release_runtime()
{
  points_subscription_.reset();
  mapper_.reset();
  tf_listener_.reset();
  tf_buffer_.reset();
  occupancy_publisher_.reset();
  local_map_publisher_.reset();
  diagnostic_publisher_.reset();
  has_last_stamp_ = false;
  has_map_load_time_ = false;
}

}  // namespace cuda_voxel_mapping

RCLCPP_COMPONENTS_REGISTER_NODE(cuda_voxel_mapping::CudaVoxelMapperNode)
