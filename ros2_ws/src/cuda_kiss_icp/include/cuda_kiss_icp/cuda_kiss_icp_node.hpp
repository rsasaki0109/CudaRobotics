#pragma once

#include "cuda_kiss_icp/pointcloud_decoder.hpp"
#include "cuda_kiss_icp/pointcloud_transform.hpp"
#include "cudarobotics/kiss_icp_gpu.hpp"

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cuda_kiss_icp {

class CudaKissIcpNode : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit CudaKissIcpNode(const rclcpp::NodeOptions & options);

protected:
  CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_error(const rclcpp_lifecycle::State & state) override;

private:
  void pointcloud_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr message);
  std::vector<float> transform_to_base(
    const std::vector<float> & xyz, const std::string & source_frame,
    const rclcpp::Time & stamp);
  void publish_result(
    const cudarobotics::KissIcpFrameResult & result, const rclcpp::Time & stamp);
  void publish_diagnostic(
    std::uint8_t level, const std::string & message, const rclcpp::Time & stamp);
  void handle_fatal_error(const std::string & message, const rclcpp::Time & stamp);
  bool read_and_validate_parameters(std::string & error);
  void release_runtime();

  std::string input_topic_;
  std::string odom_topic_;
  std::string diagnostics_topic_;
  std::string odom_frame_;
  std::string base_frame_;
  std::string expected_sensor_frame_;
  bool publish_tf_ = true;
  double transform_timeout_sec_ = 0.1;
  double max_scan_age_sec_ = 0.5;
  double max_future_stamp_sec_ = 0.05;
  int qos_depth_ = 5;
  cudarobotics::KissIcpConfig core_config_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_subscription_;
  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    diagnostic_publisher_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::unique_ptr<cudarobotics::KissIcpOdometry> odometry_;

  bool runtime_fault_ = false;
  bool has_last_stamp_ = false;
  bool has_previous_pose_ = false;
  rclcpp::Time last_stamp_{0, 0, RCL_ROS_TIME};
  cudarobotics::KissIcpPose previous_pose_;
  std::size_t scans_received_ = 0;
  std::size_t scans_processed_ = 0;
  std::size_t scans_dropped_ = 0;
  std::size_t transform_failures_ = 0;
  std::size_t invalid_clouds_ = 0;
  std::size_t non_finite_points_ = 0;
};

}  // namespace cuda_kiss_icp
