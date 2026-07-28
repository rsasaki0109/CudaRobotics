#pragma once

#include "cudarobotics/esdf_2d_gpu.hpp"

#include <cuda_robotics_msgs/msg/distance_field2_d.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace cuda_esdf {

class CudaEsdfNode : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit CudaEsdfNode(const rclcpp::NodeOptions & options);

protected:
  CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_error(const rclcpp_lifecycle::State & state) override;

private:
  void occupancy_callback(nav_msgs::msg::OccupancyGrid::ConstSharedPtr message);
  bool validate_message(
    const nav_msgs::msg::OccupancyGrid & message, std::string & error) const;
  void publish_diagnostic(
    std::uint8_t level, const std::string & message, const rclcpp::Time & stamp,
    const cudarobotics::Esdf2DResult * result = nullptr);
  void handle_fatal_error(const std::string & message, const rclcpp::Time & stamp);
  bool read_and_validate_parameters(std::string & error);
  void release_runtime();

  std::string occupancy_topic_;
  std::string esdf_topic_;
  std::string diagnostics_topic_;
  std::string expected_frame_;
  float max_distance_ = 10.0f;
  double max_input_age_sec_ = 0.0;
  double max_future_stamp_sec_ = 0.05;
  cudarobotics::Esdf2DConfig esdf_config_;

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_subscription_;
  rclcpp_lifecycle::LifecyclePublisher<
    cuda_robotics_msgs::msg::DistanceField2D>::SharedPtr esdf_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    diagnostic_publisher_;
  std::unique_ptr<cudarobotics::Esdf2DGpu> esdf_;

  bool runtime_fault_ = false;
  bool has_last_stamp_ = false;
  rclcpp::Time last_stamp_{0, 0, RCL_ROS_TIME};
  std::size_t maps_received_ = 0;
  std::size_t maps_processed_ = 0;
  std::size_t maps_dropped_ = 0;
  std::size_t schema_failures_ = 0;
  std::size_t capacity_failures_ = 0;
};

}  // namespace cuda_esdf
