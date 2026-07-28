#include "cuda_kiss_icp/cuda_kiss_icp_node.hpp"

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <lifecycle_msgs/msg/transition.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2/exceptions.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cuda_kiss_icp {
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

geometry_msgs::msg::Quaternion quaternion_from_rotation(const cudarobotics::KissIcpMat3 & rotation)
{
  const float * r = rotation.m;
  geometry_msgs::msg::Quaternion q;
  const double trace = static_cast<double>(r[0] + r[4] + r[8]);
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    q.w = 0.25 * s;
    q.x = (r[7] - r[5]) / s;
    q.y = (r[2] - r[6]) / s;
    q.z = (r[3] - r[1]) / s;
  } else if (r[0] > r[4] && r[0] > r[8]) {
    const double s = std::sqrt(1.0 + r[0] - r[4] - r[8]) * 2.0;
    q.w = (r[7] - r[5]) / s;
    q.x = 0.25 * s;
    q.y = (r[1] + r[3]) / s;
    q.z = (r[2] + r[6]) / s;
  } else if (r[4] > r[8]) {
    const double s = std::sqrt(1.0 + r[4] - r[0] - r[8]) * 2.0;
    q.w = (r[2] - r[6]) / s;
    q.x = (r[1] + r[3]) / s;
    q.y = 0.25 * s;
    q.z = (r[5] + r[7]) / s;
  } else {
    const double s = std::sqrt(1.0 + r[8] - r[0] - r[4]) * 2.0;
    q.w = (r[3] - r[1]) / s;
    q.x = (r[2] + r[6]) / s;
    q.y = (r[5] + r[7]) / s;
    q.z = 0.25 * s;
  }
  const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  q.x /= norm;
  q.y /= norm;
  q.z /= norm;
  q.w /= norm;
  return q;
}

std::array<double, 3> body_linear_velocity(
  const cudarobotics::KissIcpPose & previous,
  const cudarobotics::KissIcpPose & current, double inverse_dt)
{
  const double dx = current.t[0] - previous.t[0];
  const double dy = current.t[1] - previous.t[1];
  const double dz = current.t[2] - previous.t[2];
  const float * r = previous.R.m;
  return {
    (r[0] * dx + r[3] * dy + r[6] * dz) * inverse_dt,
    (r[1] * dx + r[4] * dy + r[7] * dz) * inverse_dt,
    (r[2] * dx + r[5] * dy + r[8] * dz) * inverse_dt};
}

std::array<double, 3> body_angular_velocity(
  const cudarobotics::KissIcpPose & previous,
  const cudarobotics::KissIcpPose & current, double inverse_dt)
{
  double relative[9] = {};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      for (int k = 0; k < 3; ++k) {
        relative[row * 3 + column] +=
          previous.R.m[k * 3 + row] * current.R.m[k * 3 + column];
      }
    }
  }
  const double cosine = std::clamp(
    (relative[0] + relative[4] + relative[8] - 1.0) * 0.5, -1.0, 1.0);
  const double angle = std::acos(cosine);
  if (angle < 1e-9) {
    return {
      0.5 * (relative[7] - relative[5]) * inverse_dt,
      0.5 * (relative[2] - relative[6]) * inverse_dt,
      0.5 * (relative[3] - relative[1]) * inverse_dt};
  }
  const double scale = angle / (2.0 * std::sin(angle)) * inverse_dt;
  return {
    (relative[7] - relative[5]) * scale,
    (relative[2] - relative[6]) * scale,
    (relative[3] - relative[1]) * scale};
}

}  // namespace

CudaKissIcpNode::CudaKissIcpNode(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("cuda_kiss_icp_odometry", options)
{
  declare_parameter("input_topic", "points");
  declare_parameter("odom_topic", "odom");
  declare_parameter("diagnostics_topic", "diagnostics");
  declare_parameter("odom_frame", "odom");
  declare_parameter("base_frame", "base_link");
  declare_parameter("expected_sensor_frame", "");
  declare_parameter("publish_tf", true);
  declare_parameter("transform_timeout_sec", 0.1);
  declare_parameter("max_scan_age_sec", 0.5);
  declare_parameter("max_future_stamp_sec", 0.05);
  declare_parameter("qos_depth", 5);
  declare_parameter("map_voxel_size", 0.5);
  declare_parameter("scan_voxel_size", 0.5);
  declare_parameter("map_radius", 40.0);
  declare_parameter("threshold_min", 1.0);
  declare_parameter("threshold_max", 3.0);
  declare_parameter("max_icp_iterations", 12);
  declare_parameter("normal_neighbors", 12);
  declare_parameter("max_scan_points", 200000);
  declare_parameter("max_map_points", 200000);
  declare_parameter("hash_capacity", 524288);
  declare_parameter("nn_backend", "voxel");
}

CallbackReturn CudaKissIcpNode::on_configure(const rclcpp_lifecycle::State &)
{
  std::string error;
  if (!read_and_validate_parameters(error)) {
    RCLCPP_ERROR(get_logger(), "configuration rejected: %s", error.c_str());
    return CallbackReturn::FAILURE;
  }
  try {
    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::QoS(10).reliable().durability_volatile());
    diagnostic_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      diagnostics_topic_, rclcpp::QoS(10).reliable().durability_volatile());
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(
      *tf_buffer_, get_node_base_interface(), get_node_logging_interface(),
      get_node_parameters_interface(), get_node_topics_interface(), true);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_logger(), "configure failed: %s", exception.what());
    release_runtime();
    return CallbackReturn::FAILURE;
  }
  runtime_fault_ = false;
  return CallbackReturn::SUCCESS;
}

CallbackReturn CudaKissIcpNode::on_activate(const rclcpp_lifecycle::State &)
{
  if (runtime_fault_) {
    RCLCPP_ERROR(
      get_logger(), "cleanup and reconfigure are required after a runtime fault");
    return CallbackReturn::FAILURE;
  }
  try {
    odometry_ = std::make_unique<cudarobotics::KissIcpOdometry>(core_config_);
    odometry_->reset();
    odom_publisher_->on_activate();
    diagnostic_publisher_->on_activate();
    points_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS().keep_last(qos_depth_),
      std::bind(&CudaKissIcpNode::pointcloud_callback, this, std::placeholders::_1));
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_logger(), "activation failed: %s", exception.what());
    release_runtime();
    return CallbackReturn::ERROR;
  }
  runtime_fault_ = false;
  has_last_stamp_ = false;
  has_previous_pose_ = false;
  scans_received_ = scans_processed_ = scans_dropped_ = 0;
  transform_failures_ = invalid_clouds_ = non_finite_points_ = 0;
  publish_diagnostic(
    diagnostic_msgs::msg::DiagnosticStatus::OK, "active", get_clock()->now());
  return CallbackReturn::SUCCESS;
}

CallbackReturn CudaKissIcpNode::on_deactivate(const rclcpp_lifecycle::State &)
{
  points_subscription_.reset();
  if (diagnostic_publisher_ && diagnostic_publisher_->is_activated()) {
    publish_diagnostic(
      runtime_fault_ ? diagnostic_msgs::msg::DiagnosticStatus::ERROR :
      diagnostic_msgs::msg::DiagnosticStatus::WARN,
      runtime_fault_ ? "deactivated after runtime fault" : "inactive", get_clock()->now());
  }
  if (odom_publisher_) {
    odom_publisher_->on_deactivate();
  }
  if (diagnostic_publisher_) {
    diagnostic_publisher_->on_deactivate();
  }
  odometry_.reset();
  return CallbackReturn::SUCCESS;
}

CallbackReturn CudaKissIcpNode::on_cleanup(const rclcpp_lifecycle::State &)
{
  release_runtime();
  runtime_fault_ = false;
  return CallbackReturn::SUCCESS;
}

CallbackReturn CudaKissIcpNode::on_shutdown(const rclcpp_lifecycle::State &)
{
  release_runtime();
  return CallbackReturn::SUCCESS;
}

CallbackReturn CudaKissIcpNode::on_error(const rclcpp_lifecycle::State &)
{
  RCLCPP_ERROR(get_logger(), "entering lifecycle error processing");
  release_runtime();
  return CallbackReturn::SUCCESS;
}

bool CudaKissIcpNode::read_and_validate_parameters(std::string & error)
{
  input_topic_ = get_parameter("input_topic").as_string();
  odom_topic_ = get_parameter("odom_topic").as_string();
  diagnostics_topic_ = get_parameter("diagnostics_topic").as_string();
  odom_frame_ = get_parameter("odom_frame").as_string();
  base_frame_ = get_parameter("base_frame").as_string();
  expected_sensor_frame_ = get_parameter("expected_sensor_frame").as_string();
  publish_tf_ = get_parameter("publish_tf").as_bool();
  transform_timeout_sec_ = get_parameter("transform_timeout_sec").as_double();
  max_scan_age_sec_ = get_parameter("max_scan_age_sec").as_double();
  max_future_stamp_sec_ = get_parameter("max_future_stamp_sec").as_double();
  const auto qos_depth = get_parameter("qos_depth").as_int();
  core_config_.map_voxel_size =
    static_cast<float>(get_parameter("map_voxel_size").as_double());
  core_config_.scan_voxel_size =
    static_cast<float>(get_parameter("scan_voxel_size").as_double());
  core_config_.map_radius = static_cast<float>(get_parameter("map_radius").as_double());
  core_config_.threshold_min =
    static_cast<float>(get_parameter("threshold_min").as_double());
  core_config_.threshold_max =
    static_cast<float>(get_parameter("threshold_max").as_double());
  const auto max_icp_iterations = get_parameter("max_icp_iterations").as_int();
  const auto normal_neighbors = get_parameter("normal_neighbors").as_int();
  const auto max_scan_points = get_parameter("max_scan_points").as_int();
  const auto max_map_points = get_parameter("max_map_points").as_int();
  const auto hash_capacity = get_parameter("hash_capacity").as_int();
  if (qos_depth <= 0 || qos_depth > INT_MAX ||
    max_icp_iterations <= 0 || max_icp_iterations > INT_MAX ||
    normal_neighbors <= 0 || normal_neighbors > INT_MAX ||
    max_scan_points <= 0 || max_map_points <= 0 || hash_capacity <= 0)
  {
    error = "queue, iteration, neighbor, point, and hash sizes must be positive and in range";
    return false;
  }
  qos_depth_ = static_cast<int>(qos_depth);
  core_config_.max_icp_iterations = static_cast<int>(max_icp_iterations);
  core_config_.normal_neighbors = static_cast<int>(normal_neighbors);
  core_config_.max_scan_points = static_cast<std::size_t>(max_scan_points);
  core_config_.max_map_points = static_cast<std::size_t>(max_map_points);
  core_config_.hash_capacity = static_cast<std::size_t>(hash_capacity);
  const std::string backend = get_parameter("nn_backend").as_string();
  if (backend == "voxel") {
    core_config_.nn_backend = cudarobotics::KissIcpNnBackend::Voxel;
  } else if (backend == "brute") {
    core_config_.nn_backend = cudarobotics::KissIcpNnBackend::BruteForce;
  } else {
    error = "nn_backend must be 'voxel' or 'brute'";
    return false;
  }
  if (input_topic_.empty() || odom_topic_.empty() || diagnostics_topic_.empty()) {
    error = "topic names must be non-empty";
    return false;
  }
  if (input_topic_.front() == '/' || odom_topic_.front() == '/' ||
    diagnostics_topic_.front() == '/')
  {
    error = "input, odometry, and diagnostics topics must be relative";
    return false;
  }
  if (odom_frame_.empty() || base_frame_.empty() || odom_frame_ == base_frame_) {
    error = "odom_frame and base_frame must be distinct non-empty frame names";
    return false;
  }
  if (odom_frame_.front() == '/' || base_frame_.front() == '/' ||
    (!expected_sensor_frame_.empty() && expected_sensor_frame_.front() == '/'))
  {
    error = "frame names must not begin with '/'";
    return false;
  }
  if (!std::isfinite(transform_timeout_sec_) || !std::isfinite(max_scan_age_sec_) ||
    !std::isfinite(max_future_stamp_sec_) ||
    transform_timeout_sec_ < 0.0 || max_scan_age_sec_ < 0.0 ||
    max_future_stamp_sec_ < 0.0 || qos_depth_ <= 0)
  {
    error = "timeouts/age bounds must be non-negative and qos_depth positive";
    return false;
  }
  error = cudarobotics::validate_kiss_icp_config(core_config_);
  return error.empty();
}

std::vector<float> CudaKissIcpNode::transform_to_base(
  const std::vector<float> & xyz, const std::string & source_frame,
  const rclcpp::Time & stamp)
{
  if (source_frame == base_frame_) {
    return xyz;
  }
  const auto transform = tf_buffer_->lookupTransform(
    base_frame_, source_frame, stamp, rclcpp::Duration::from_seconds(transform_timeout_sec_));
  try {
    return cuda_robotics_common::transform_xyz(xyz, transform.transform);
  } catch (const std::invalid_argument & exception) {
    throw tf2::TransformException(exception.what());
  }
}

void CudaKissIcpNode::pointcloud_callback(
  sensor_msgs::msg::PointCloud2::ConstSharedPtr message)
{
  ++scans_received_;
  const rclcpp::Time stamp(message->header.stamp, get_clock()->get_clock_type());
  if (message->header.frame_id.empty()) {
    ++invalid_clouds_;
    ++scans_dropped_;
    publish_diagnostic(
      diagnostic_msgs::msg::DiagnosticStatus::WARN, "dropped cloud with empty frame_id", stamp);
    return;
  }
  if (!expected_sensor_frame_.empty() &&
    message->header.frame_id != expected_sensor_frame_)
  {
    ++invalid_clouds_;
    ++scans_dropped_;
    publish_diagnostic(
      diagnostic_msgs::msg::DiagnosticStatus::WARN,
      "dropped cloud from unexpected sensor frame", stamp);
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

  try {
    const std::size_t declared_points =
      static_cast<std::size_t>(message->width) * message->height;
    if (declared_points > core_config_.max_scan_points) {
      handle_fatal_error("declared cloud exceeds max_scan_points", stamp);
      return;
    }
    cuda_robotics_common::DecodedPointCloud decoded =
      cuda_robotics_common::decode_xyz(*message);
    non_finite_points_ += decoded.skipped_non_finite;
    if (decoded.xyz.size() / 3 < 10) {
      ++invalid_clouds_;
      ++scans_dropped_;
      publish_diagnostic(
        diagnostic_msgs::msg::DiagnosticStatus::WARN,
        "dropped cloud with fewer than 10 finite points", stamp);
      return;
    }
    std::vector<float> base_points =
      transform_to_base(decoded.xyz, message->header.frame_id, stamp);
    const auto result = odometry_->register_scan(base_points);
    publish_result(result, stamp);
    last_stamp_ = stamp;
    has_last_stamp_ = true;
    ++scans_processed_;
    publish_diagnostic(diagnostic_msgs::msg::DiagnosticStatus::OK, "tracking", stamp);
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
  } catch (const std::exception & exception) {
    ++scans_dropped_;
    handle_fatal_error(exception.what(), stamp);
  }
}

void CudaKissIcpNode::publish_result(
  const cudarobotics::KissIcpFrameResult & result, const rclcpp::Time & stamp)
{
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = odom_frame_;
  odom.child_frame_id = base_frame_;
  odom.pose.pose.position.x = result.pose.t[0];
  odom.pose.pose.position.y = result.pose.t[1];
  odom.pose.pose.position.z = result.pose.t[2];
  odom.pose.pose.orientation = quaternion_from_rotation(result.pose.R);
  odom.pose.covariance.fill(0.0);
  const double position_variance =
    std::max(1e-4, static_cast<double>(result.alignment.rmse * result.alignment.rmse));
  const double rotation_variance = std::max(1e-4, 4.0 * position_variance);
  odom.pose.covariance[0] = odom.pose.covariance[7] = odom.pose.covariance[14] =
    position_variance;
  odom.pose.covariance[21] = odom.pose.covariance[28] = odom.pose.covariance[35] =
    rotation_variance;
  odom.twist.covariance.fill(0.0);
  if (has_previous_pose_ && has_last_stamp_) {
    const double dt = (stamp - last_stamp_).seconds();
    if (dt > 0.0) {
      const auto linear = body_linear_velocity(previous_pose_, result.pose, 1.0 / dt);
      const auto angular = body_angular_velocity(previous_pose_, result.pose, 1.0 / dt);
      odom.twist.twist.linear.x = linear[0];
      odom.twist.twist.linear.y = linear[1];
      odom.twist.twist.linear.z = linear[2];
      odom.twist.twist.angular.x = angular[0];
      odom.twist.twist.angular.y = angular[1];
      odom.twist.twist.angular.z = angular[2];
    }
  }
  odom.twist.covariance = odom.pose.covariance;
  odom_publisher_->publish(odom);

  if (publish_tf_) {
    geometry_msgs::msg::TransformStamped transform;
    transform.header = odom.header;
    transform.child_frame_id = base_frame_;
    transform.transform.translation.x = result.pose.t[0];
    transform.transform.translation.y = result.pose.t[1];
    transform.transform.translation.z = result.pose.t[2];
    transform.transform.rotation = odom.pose.pose.orientation;
    tf_broadcaster_->sendTransform(transform);
  }
  previous_pose_ = result.pose;
  has_previous_pose_ = true;
}

void CudaKissIcpNode::publish_diagnostic(
  std::uint8_t level, const std::string & message, const rclcpp::Time & stamp)
{
  if (!diagnostic_publisher_ || !diagnostic_publisher_->is_activated()) {
    return;
  }
  diagnostic_msgs::msg::DiagnosticArray array;
  array.header.stamp = stamp;
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.level = level;
  status.name = get_fully_qualified_name() + std::string(": odometry");
  status.hardware_id = cudarobotics::kiss_icp_backend_name(core_config_.nn_backend);
  status.message = message;
  status.values = {
    key_value("scans_received", std::to_string(scans_received_)),
    key_value("scans_processed", std::to_string(scans_processed_)),
    key_value("scans_dropped", std::to_string(scans_dropped_)),
    key_value("transform_failures", std::to_string(transform_failures_)),
    key_value("invalid_clouds", std::to_string(invalid_clouds_)),
    key_value("non_finite_points", std::to_string(non_finite_points_)),
    key_value("deskewed", "false"),
    key_value("runtime_fault", runtime_fault_ ? "true" : "false")};
  if (odometry_) {
    status.values.push_back(key_value("frames", std::to_string(odometry_->frame_count())));
    status.values.push_back(
      key_value("map_points", std::to_string(odometry_->map_point_count())));
  }
  array.status.push_back(std::move(status));
  diagnostic_publisher_->publish(array);
}

void CudaKissIcpNode::handle_fatal_error(
  const std::string & message, const rclcpp::Time & stamp)
{
  runtime_fault_ = true;
  RCLCPP_ERROR(get_logger(), "fatal odometry error: %s", message.c_str());
  publish_diagnostic(
    diagnostic_msgs::msg::DiagnosticStatus::ERROR,
    std::string("fatal odometry error: ") + message, stamp);
  points_subscription_.reset();
  // ROS 2 has no public active->error transition for an arbitrary subscription
  // callback. Force the managed node to inactive; cleanup/reconfigure is then
  // required before processing can resume.
  try {
    trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_logger(), "failed to deactivate after fatal error: %s", exception.what());
  }
}

void CudaKissIcpNode::release_runtime()
{
  points_subscription_.reset();
  odometry_.reset();
  tf_broadcaster_.reset();
  tf_listener_.reset();
  tf_buffer_.reset();
  odom_publisher_.reset();
  diagnostic_publisher_.reset();
  has_last_stamp_ = false;
  has_previous_pose_ = false;
}

}  // namespace cuda_kiss_icp

RCLCPP_COMPONENTS_REGISTER_NODE(cuda_kiss_icp::CudaKissIcpNode)
