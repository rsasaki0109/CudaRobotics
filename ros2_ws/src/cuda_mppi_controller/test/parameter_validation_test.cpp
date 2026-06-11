// Fast validation smoke: invalid controller parameters should be rejected
// during configure(), before any CUDA optimizer allocation is attempted.
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cuda_mppi_controller/cuda_mppi_controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "tf2_ros/buffer.h"

namespace
{

bool configureRejects(const std::string & label, const rclcpp::Parameter & parameter)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides({parameter});
  options.arguments({"--ros-args", "-r", "__node:=validation_" + label});

  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("validation_" + label, options);
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  cuda_mppi_controller::CudaMppiController controller;

  try {
    controller.configure(node, "FollowPath", tf, nullptr);
  } catch (const std::exception & ex) {
    std::printf("PASS: %s rejected: %s\n", label.c_str(), ex.what());
    return true;
  }

  std::printf("FAIL: %s was accepted\n", label.c_str());
  controller.cleanup();
  return false;
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  const std::vector<std::pair<std::string, rclcpp::Parameter>> cases = {
    {"batch_size_zero", rclcpp::Parameter("FollowPath.batch_size", 0)},
    {"time_steps_zero", rclcpp::Parameter("FollowPath.time_steps", 0)},
    {"iteration_count_zero", rclcpp::Parameter("FollowPath.iteration_count", 0)},
    {"model_dt_zero", rclcpp::Parameter("FollowPath.model_dt", 0.0)},
    {"bad_motion_model", rclcpp::Parameter("FollowPath.motion_model", "SkidSteer")},
    {"temperature_zero", rclcpp::Parameter("FollowPath.temperature", 0.0)},
    {"negative_v_max", rclcpp::Parameter("FollowPath.v_max", -0.1)},
    {"v_min_above_v_max", rclcpp::Parameter("FollowPath.v_min", 1.0)},
    {"negative_w_max", rclcpp::Parameter("FollowPath.w_max", -1.0)},
    {"negative_distance_field_weight",
      rclcpp::Parameter("FollowPath.distance_field_weight", -1.0)},
    {"negative_path_angle_weight", rclcpp::Parameter("FollowPath.path_angle_weight", -1.0)},
    {"negative_curvature_speed_weight",
      rclcpp::Parameter("FollowPath.curvature_speed_weight", -1.0)},
    {"negative_curvature_speed_min",
      rclcpp::Parameter("FollowPath.curvature_speed_min", -0.1)},
    {"negative_distance_field_cutoff",
      rclcpp::Parameter("FollowPath.distance_field_cutoff", -0.1)},
    {"negative_lookahead", rclcpp::Parameter("FollowPath.lookahead_dist", -1.0)},
    {"negative_transform_tolerance", rclcpp::Parameter("FollowPath.transform_tolerance", -0.1)},
    {"negative_diagnostics_log_period",
      rclcpp::Parameter("FollowPath.diagnostics_log_period", -0.1)},
  };

  bool ok = true;
  for (const auto & test_case : cases) {
    ok = configureRejects(test_case.first, test_case.second) && ok;
  }

  rclcpp::shutdown();
  return ok ? 0 : 1;
}
