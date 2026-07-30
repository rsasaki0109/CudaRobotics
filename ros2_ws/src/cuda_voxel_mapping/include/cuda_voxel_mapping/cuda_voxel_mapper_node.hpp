#pragma once

#include "cudarobotics/voxel_mapping_gpu.hpp"

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace cuda_voxel_mapping {

nav_msgs::msg::OccupancyGrid make_occupancy_message(
  const cudarobotics::OccupancyProjection & projection,
  const rclcpp::Time & stamp,
  const rclcpp::Time & map_load_time,
  const std::string & frame_id);

class CudaVoxelMapperNode : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit CudaVoxelMapperNode(const rclcpp::NodeOptions & options);

protected:
  CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_error(const rclcpp_lifecycle::State & state) override;

private:
  void pointcloud_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr message);
  void publish_occupancy(
    const cudarobotics::OccupancyProjection & projection,
    const rclcpp::Time & stamp);
  void publish_local_map(const rclcpp::Time & stamp);
  void publish_diagnostic(
    std::uint8_t level, const std::string & message, const rclcpp::Time & stamp,
    const cudarobotics::VoxelMappingStats * stats = nullptr);
  void handle_fatal_error(const std::string & message, const rclcpp::Time & stamp);
  bool read_and_validate_parameters(std::string & error);
  void release_runtime();

  std::string input_topic_;
  std::string occupancy_topic_;
  std::string local_map_topic_;
  std::string diagnostics_topic_;
  std::string odom_frame_;
  std::string expected_sensor_frame_;
  double transform_timeout_sec_ = 0.1;
  double max_scan_age_sec_ = 0.5;
  double max_future_stamp_sec_ = 0.05;
  int qos_depth_ = 5;
  int local_map_publish_stride_ = 10;
  cudarobotics::VoxelMappingConfig mapping_config_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_subscription_;
  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::OccupancyGrid>::SharedPtr
    occupancy_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr
    local_map_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    diagnostic_publisher_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<cudarobotics::VoxelMapperGpu> mapper_;

  bool runtime_fault_ = false;
  bool has_last_stamp_ = false;
  bool has_map_load_time_ = false;
  rclcpp::Time last_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time map_load_time_{0, 0, RCL_ROS_TIME};
  std::size_t scans_received_ = 0;
  std::size_t scans_integrated_ = 0;
  std::size_t scans_dropped_ = 0;
  std::size_t transform_failures_ = 0;
  std::size_t invalid_clouds_ = 0;
  std::size_t non_finite_points_ = 0;
  std::size_t capacity_failures_ = 0;
};

}  // namespace cuda_voxel_mapping
