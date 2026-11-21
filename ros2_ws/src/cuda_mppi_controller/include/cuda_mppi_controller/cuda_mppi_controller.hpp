#ifndef CUDA_MPPI_CONTROLLER__CUDA_MPPI_CONTROLLER_HPP_
#define CUDA_MPPI_CONTROLLER__CUDA_MPPI_CONTROLLER_HPP_

#include <memory>
#include <mutex>
#include <fstream>
#include <string>
#include <vector>

#include "nav2_core/controller.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav_msgs/msg/path.hpp"
#include "cuda_robotics_msgs/msg/tracked_obstacle_array.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "tf2_ros/buffer.h"

#include "cuda_mppi_controller/mppi_gpu.hpp"

namespace cuda_mppi_controller
{

/**
 * @class CudaMppiController
 * @brief nav2 controller plugin running MPPI rollouts on the GPU
 *        (1 CUDA thread = 1 sampled trajectory).
 */
class CudaMppiController : public nav2_core::Controller
{
public:
  CudaMppiController() = default;
  ~CudaMppiController() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  void setPlan(const nav_msgs::msg::Path & path) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

  void reset();

private:
  struct DiagnosticsCsv
  {
    std::ofstream file;
    bool enabled = false;
  };

  // Extract the local window of the global plan around the robot, transformed
  // into the costmap global frame. Returns flattened [x0,y0,x1,y1,...] points;
  // sets goal pose (window end) and whether it is the true final goal.
  std::vector<float> extractLocalPath(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    float & goal_x, float & goal_y, float & goal_yaw, bool & goal_is_final);

  // Hand the latest tracked obstacles (transformed into the costmap frame and
  // extrapolated to now) to the optimizer, or clear them when there is no
  // fresh message.
  void updateDynamicObstacles(const geometry_msgs::msg::PoseStamped & robot_pose);

  DiagnosticsCsv openDiagnosticsCsv(const std::string & path) const;
  void emitDiagnostics(
    const MppiResult & result, double solve_ms, int path_points,
    int costmap_size_x, int costmap_size_y);

  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  std::string name_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  rclcpp::Logger logger_{rclcpp::get_logger("CudaMppiController")};

  nav_msgs::msg::Path global_plan_;
  std::unique_ptr<MppiGpu> optimizer_;
  MppiParams params_;

  double lookahead_dist_ = 3.0;
  double transform_tolerance_ = 0.1;
  double diagnostics_log_period_ = 0.0;
  std::string diagnostics_csv_path_;
  DiagnosticsCsv diagnostics_csv_;
  rclcpp::Time last_diagnostics_log_time_{0, 0, RCL_ROS_TIME};
  bool has_diagnostics_log_time_ = false;
  std::string dynamic_obstacles_topic_;
  double dynamic_obstacles_timeout_ = 0.5;
  rclcpp::Subscription<cuda_robotics_msgs::msg::TrackedObstacleArray>::SharedPtr
    dynamic_obstacles_sub_;
  std::mutex dynamic_obstacles_mutex_;
  cuda_robotics_msgs::msg::TrackedObstacleArray::ConstSharedPtr latest_dynamic_obstacles_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_;
  bool updateParamsFromNode(const rclcpp_lifecycle::LifecycleNode::SharedPtr & node);
};

}  // namespace cuda_mppi_controller

#endif  // CUDA_MPPI_CONTROLLER__CUDA_MPPI_CONTROLLER_HPP_
