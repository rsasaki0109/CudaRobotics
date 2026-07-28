#include "cuda_esdf/cuda_esdf_node.hpp"

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <lifecycle_msgs/msg/transition.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <climits>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cuda_esdf {
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

bool finite_pose(const geometry_msgs::msg::Pose & pose)
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
  return norm > 1e-9 && std::fabs(norm - 1.0) < 1e-3;
}

}  // namespace

CudaEsdfNode::CudaEsdfNode(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("cuda_esdf_node", options)
{
  declare_parameter("occupancy_topic", "occupancy");
  declare_parameter("esdf_topic", "esdf");
  declare_parameter("diagnostics_topic", "diagnostics");
  declare_parameter("expected_frame", "odom");
  declare_parameter("max_distance", 10.0);
  declare_parameter("max_width", 1024);
  declare_parameter("max_height", 1024);
  declare_parameter("occupancy_threshold", 50);
  declare_parameter("unknown_policy", "occupied");
  declare_parameter("max_input_age_sec", 0.0);
  declare_parameter("max_future_stamp_sec", 0.05);
}

CallbackReturn CudaEsdfNode::on_configure(const rclcpp_lifecycle::State &)
{
  std::string error;
  if (!read_and_validate_parameters(error)) {
    RCLCPP_ERROR(get_logger(), "configuration rejected: %s", error.c_str());
    return CallbackReturn::FAILURE;
  }
  try {
    esdf_publisher_ = create_publisher<cuda_robotics_msgs::msg::DistanceField2D>(
      esdf_topic_, rclcpp::QoS(1).reliable().transient_local());
    diagnostic_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      diagnostics_topic_, rclcpp::QoS(10).reliable().durability_volatile());
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_logger(), "configure failed: %s", exception.what());
    release_runtime();
    return CallbackReturn::FAILURE;
  }
  runtime_fault_ = false;
  return CallbackReturn::SUCCESS;
}

CallbackReturn CudaEsdfNode::on_activate(const rclcpp_lifecycle::State &)
{
  if (runtime_fault_) {
    RCLCPP_ERROR(
      get_logger(), "cleanup and reconfigure are required after a runtime fault");
    return CallbackReturn::FAILURE;
  }
  try {
    esdf_ = std::make_unique<cudarobotics::Esdf2DGpu>(esdf_config_);
    esdf_publisher_->on_activate();
    diagnostic_publisher_->on_activate();
    occupancy_subscription_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      occupancy_topic_, rclcpp::QoS(1).reliable().transient_local(),
      std::bind(&CudaEsdfNode::occupancy_callback, this, std::placeholders::_1));
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_logger(), "activation failed: %s", exception.what());
    release_runtime();
    return CallbackReturn::ERROR;
  }
  has_last_stamp_ = false;
  maps_received_ = maps_processed_ = maps_dropped_ = 0;
  schema_failures_ = capacity_failures_ = 0;
  publish_diagnostic(
    diagnostic_msgs::msg::DiagnosticStatus::OK, "active", get_clock()->now());
  return CallbackReturn::SUCCESS;
}

CallbackReturn CudaEsdfNode::on_deactivate(const rclcpp_lifecycle::State &)
{
  occupancy_subscription_.reset();
  if (diagnostic_publisher_ && diagnostic_publisher_->is_activated()) {
    publish_diagnostic(
      runtime_fault_ ? diagnostic_msgs::msg::DiagnosticStatus::ERROR :
      diagnostic_msgs::msg::DiagnosticStatus::WARN,
      runtime_fault_ ? "deactivated after runtime fault" : "inactive", get_clock()->now());
  }
  if (esdf_publisher_) esdf_publisher_->on_deactivate();
  if (diagnostic_publisher_) diagnostic_publisher_->on_deactivate();
  esdf_.reset();
  return CallbackReturn::SUCCESS;
}

CallbackReturn CudaEsdfNode::on_cleanup(const rclcpp_lifecycle::State &)
{
  release_runtime();
  runtime_fault_ = false;
  return CallbackReturn::SUCCESS;
}

CallbackReturn CudaEsdfNode::on_shutdown(const rclcpp_lifecycle::State &)
{
  release_runtime();
  return CallbackReturn::SUCCESS;
}

CallbackReturn CudaEsdfNode::on_error(const rclcpp_lifecycle::State &)
{
  RCLCPP_ERROR(get_logger(), "entering lifecycle error processing");
  release_runtime();
  return CallbackReturn::SUCCESS;
}

bool CudaEsdfNode::read_and_validate_parameters(std::string & error)
{
  occupancy_topic_ = get_parameter("occupancy_topic").as_string();
  esdf_topic_ = get_parameter("esdf_topic").as_string();
  diagnostics_topic_ = get_parameter("diagnostics_topic").as_string();
  expected_frame_ = get_parameter("expected_frame").as_string();
  max_distance_ = static_cast<float>(get_parameter("max_distance").as_double());
  max_input_age_sec_ = get_parameter("max_input_age_sec").as_double();
  max_future_stamp_sec_ = get_parameter("max_future_stamp_sec").as_double();
  const auto max_width = get_parameter("max_width").as_int();
  const auto max_height = get_parameter("max_height").as_int();
  const auto threshold = get_parameter("occupancy_threshold").as_int();
  if (max_width <= 0 || max_width > INT_MAX ||
    max_height <= 0 || max_height > INT_MAX ||
    threshold < 0 || threshold > 100)
  {
    error = "ESDF dimensions or occupancy threshold are out of range";
    return false;
  }
  esdf_config_.max_width = static_cast<int>(max_width);
  esdf_config_.max_height = static_cast<int>(max_height);
  esdf_config_.occupancy_threshold = static_cast<int>(threshold);
  const std::string policy = get_parameter("unknown_policy").as_string();
  if (policy == "occupied") {
    esdf_config_.unknown_policy = cudarobotics::UnknownSpacePolicy::Occupied;
  } else if (policy == "free") {
    esdf_config_.unknown_policy = cudarobotics::UnknownSpacePolicy::Free;
  } else {
    error = "unknown_policy must be 'occupied' or 'free'";
    return false;
  }
  if (occupancy_topic_.empty() || esdf_topic_.empty() || diagnostics_topic_.empty()) {
    error = "topic names must be non-empty";
    return false;
  }
  if (occupancy_topic_.front() == '/' || esdf_topic_.front() == '/' ||
    diagnostics_topic_.front() == '/')
  {
    error = "input and output topics must be relative";
    return false;
  }
  if (expected_frame_.empty() || expected_frame_.front() == '/') {
    error = "expected_frame must be non-empty and must not begin with '/'";
    return false;
  }
  if (!(max_distance_ > 0.0f) || !std::isfinite(max_distance_) ||
    !std::isfinite(max_input_age_sec_) || max_input_age_sec_ < 0.0 ||
    !std::isfinite(max_future_stamp_sec_) || max_future_stamp_sec_ < 0.0)
  {
    error = "distance and timestamp limits must be finite and non-negative";
    return false;
  }
  error = cudarobotics::validate_esdf_2d_config(esdf_config_);
  return error.empty();
}

bool CudaEsdfNode::validate_message(
  const nav_msgs::msg::OccupancyGrid & message, std::string & error) const
{
  if (message.header.frame_id.empty() || message.header.frame_id != expected_frame_) {
    error = "occupancy frame does not match expected_frame";
    return false;
  }
  if (message.info.width == 0 || message.info.height == 0) {
    error = "occupancy dimensions must be non-zero";
    return false;
  }
  const std::size_t width = message.info.width;
  const std::size_t height = message.info.height;
  if (width > std::numeric_limits<std::size_t>::max() / height ||
    message.data.size() != width * height)
  {
    error = "occupancy data size must equal width * height";
    return false;
  }
  if (!(message.info.resolution > 0.0f) || !std::isfinite(message.info.resolution)) {
    error = "occupancy resolution must be finite and positive";
    return false;
  }
  if (!finite_pose(message.info.origin)) {
    error = "occupancy origin pose must be finite with a unit quaternion";
    return false;
  }
  for (std::int8_t value : message.data) {
    if (value < -1 || value > 100) {
      error = "occupancy values must lie in [-1, 100]";
      return false;
    }
  }
  return true;
}

void CudaEsdfNode::occupancy_callback(
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr message)
{
  ++maps_received_;
  const rclcpp::Time stamp(message->header.stamp, get_clock()->get_clock_type());
  if (stamp.nanoseconds() == 0 || (has_last_stamp_ && stamp <= last_stamp_)) {
    ++maps_dropped_;
    publish_diagnostic(
      diagnostic_msgs::msg::DiagnosticStatus::WARN,
      "dropped zero or non-monotonic occupancy timestamp", stamp);
    return;
  }
  const double age = (get_clock()->now() - stamp).seconds();
  if ((max_input_age_sec_ > 0.0 && age > max_input_age_sec_) ||
    age < -max_future_stamp_sec_)
  {
    ++maps_dropped_;
    publish_diagnostic(
      diagnostic_msgs::msg::DiagnosticStatus::WARN,
      "dropped stale or future-dated occupancy", stamp);
    return;
  }
  std::string error;
  if (!validate_message(*message, error)) {
    ++schema_failures_;
    ++maps_dropped_;
    publish_diagnostic(
      diagnostic_msgs::msg::DiagnosticStatus::WARN, error, stamp);
    return;
  }
  if (message->info.width > static_cast<std::uint32_t>(esdf_config_.max_width) ||
    message->info.height > static_cast<std::uint32_t>(esdf_config_.max_height))
  {
    ++capacity_failures_;
    ++maps_dropped_;
    handle_fatal_error("occupancy dimensions exceed ESDF capacity", stamp);
    return;
  }

  try {
    const auto result = esdf_->compute(
      message->data,
      static_cast<int>(message->info.width),
      static_cast<int>(message->info.height),
      message->info.resolution,
      max_distance_);
    if (result.distances.size() != message->data.size()) {
      throw std::runtime_error("ESDF result shape invariant failed");
    }
    for (float distance : result.distances) {
      if (!std::isfinite(distance) || distance < 0.0f ||
        distance > max_distance_)
      {
        throw std::runtime_error("ESDF result range invariant failed");
      }
    }
    cuda_robotics_msgs::msg::DistanceField2D output;
    output.header = message->header;
    output.origin = message->info.origin;
    output.resolution = message->info.resolution;
    output.width = message->info.width;
    output.height = message->info.height;
    output.max_distance = max_distance_;
    output.distances = result.distances;
    esdf_publisher_->publish(output);
    ++maps_processed_;
    last_stamp_ = stamp;
    has_last_stamp_ = true;
    publish_diagnostic(
      diagnostic_msgs::msg::DiagnosticStatus::OK, "esdf ready", stamp, &result);
  } catch (const std::length_error & exception) {
    ++capacity_failures_;
    ++maps_dropped_;
    handle_fatal_error(exception.what(), stamp);
  } catch (const std::exception & exception) {
    ++maps_dropped_;
    handle_fatal_error(exception.what(), stamp);
  }
}

void CudaEsdfNode::publish_diagnostic(
  std::uint8_t level, const std::string & message, const rclcpp::Time & stamp,
  const cudarobotics::Esdf2DResult * result)
{
  if (!diagnostic_publisher_ || !diagnostic_publisher_->is_activated()) return;
  diagnostic_msgs::msg::DiagnosticArray array;
  array.header.stamp = stamp;
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.level = level;
  status.name = get_fully_qualified_name() + std::string(": esdf");
  status.hardware_id = "cuda";
  status.message = message;
  status.values = {
    key_value("maps_received", std::to_string(maps_received_)),
    key_value("maps_processed", std::to_string(maps_processed_)),
    key_value("maps_dropped", std::to_string(maps_dropped_)),
    key_value("schema_failures", std::to_string(schema_failures_)),
    key_value("capacity_failures", std::to_string(capacity_failures_)),
    key_value(
      "unknown_policy",
      cudarobotics::unknown_space_policy_name(esdf_config_.unknown_policy)),
    key_value("max_distance", std::to_string(max_distance_)),
    key_value("runtime_fault", runtime_fault_ ? "true" : "false")};
  if (result) {
    status.values.push_back(
      key_value("occupied_cells", std::to_string(result->occupied_cells)));
    status.values.push_back(
      key_value("unknown_cells", std::to_string(result->unknown_cells)));
    status.values.push_back(key_value("gpu_ms", std::to_string(result->gpu_ms)));
  }
  array.status.push_back(std::move(status));
  diagnostic_publisher_->publish(array);
}

void CudaEsdfNode::handle_fatal_error(
  const std::string & message, const rclcpp::Time & stamp)
{
  runtime_fault_ = true;
  RCLCPP_ERROR(get_logger(), "fatal ESDF error: %s", message.c_str());
  publish_diagnostic(
    diagnostic_msgs::msg::DiagnosticStatus::ERROR,
    std::string("fatal ESDF error: ") + message, stamp);
  occupancy_subscription_.reset();
  try {
    trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_logger(), "failed to deactivate after fatal error: %s", exception.what());
  }
}

void CudaEsdfNode::release_runtime()
{
  occupancy_subscription_.reset();
  esdf_.reset();
  esdf_publisher_.reset();
  diagnostic_publisher_.reset();
  has_last_stamp_ = false;
}

}  // namespace cuda_esdf

RCLCPP_COMPONENTS_REGISTER_NODE(cuda_esdf::CudaEsdfNode)
