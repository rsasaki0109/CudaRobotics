#include "cuda_mppi_controller/cuda_mppi_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "nav2_core/controller_exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace cuda_mppi_controller
{

void CudaMppiController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  name_ = name;
  tf_ = tf;
  costmap_ros_ = costmap_ros;

  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("CudaMppiController: parent node expired in configure()");
  }
  logger_ = node->get_logger();

  using nav2_util::declare_parameter_if_not_declared;
  auto declare_get = [&](const std::string & param, auto default_value, auto & out) {
      declare_parameter_if_not_declared(
        node, name_ + "." + param, rclcpp::ParameterValue(default_value));
      node->get_parameter(name_ + "." + param, out);
    };

  int batch_size = params_.batch_size;
  int time_steps = params_.time_steps;
  int iteration_count = params_.iteration_count;
  double model_dt = params_.model_dt;
  std::string motion_model = "DiffDrive";
  double v_max = params_.v_max, v_min = params_.v_min, w_max = params_.w_max;
  double vy_max = params_.vy_max, min_turning_r = params_.min_turning_r;
  double v_std = params_.v_std, vy_std = params_.vy_std, w_std = params_.w_std;
  bool consider_footprint = params_.consider_footprint;
  double lambda = params_.lambda;
  double goal_weight = params_.goal_weight;
  double goal_yaw_weight = params_.goal_yaw_weight;
  double path_weight = params_.path_weight;
  double path_follow_weight = params_.path_follow_weight;
  double follow_lookahead = params_.follow_lookahead;
  double costmap_weight = params_.costmap_weight;
  double smoothness_weight = params_.smoothness_weight;
  double backward_weight = params_.backward_weight;
  double speed_weight = params_.speed_weight;
  double angular_weight = params_.angular_weight;
  double yaw_activation = params_.yaw_goal_activation_dist;
  bool enable_retreat = params_.enable_retreat;
  double retreat_scale = params_.retreat_scale;

  declare_get("batch_size", batch_size, batch_size);
  declare_get("time_steps", time_steps, time_steps);
  declare_get("iteration_count", iteration_count, iteration_count);
  declare_get("model_dt", model_dt, model_dt);
  declare_get("motion_model", motion_model, motion_model);
  declare_get("v_max", v_max, v_max);
  declare_get("v_min", v_min, v_min);
  declare_get("vy_max", vy_max, vy_max);
  declare_get("w_max", w_max, w_max);
  declare_get("min_turning_r", min_turning_r, min_turning_r);
  declare_get("v_std", v_std, v_std);
  declare_get("vy_std", vy_std, vy_std);
  declare_get("w_std", w_std, w_std);
  declare_get("consider_footprint", consider_footprint, consider_footprint);
  declare_get("temperature", lambda, lambda);
  declare_get("goal_weight", goal_weight, goal_weight);
  declare_get("goal_yaw_weight", goal_yaw_weight, goal_yaw_weight);
  declare_get("path_weight", path_weight, path_weight);
  declare_get("path_follow_weight", path_follow_weight, path_follow_weight);
  declare_get("follow_lookahead", follow_lookahead, follow_lookahead);
  declare_get("costmap_weight", costmap_weight, costmap_weight);
  declare_get("smoothness_weight", smoothness_weight, smoothness_weight);
  declare_get("backward_weight", backward_weight, backward_weight);
  declare_get("speed_weight", speed_weight, speed_weight);
  declare_get("angular_weight", angular_weight, angular_weight);
  declare_get("yaw_goal_activation_dist", yaw_activation, yaw_activation);
  declare_get("enable_retreat", enable_retreat, enable_retreat);
  declare_get("retreat_scale", retreat_scale, retreat_scale);
  declare_get("lookahead_dist", lookahead_dist_, lookahead_dist_);
  declare_get("transform_tolerance", transform_tolerance_, transform_tolerance_);

  params_.batch_size = batch_size;
  params_.time_steps = time_steps;
  params_.iteration_count = iteration_count;
  params_.model_dt = static_cast<float>(model_dt);
  if (motion_model == "DiffDrive") {
    params_.motion_model = MotionModel::DiffDrive;
  } else if (motion_model == "Ackermann") {
    params_.motion_model = MotionModel::Ackermann;
  } else if (motion_model == "Omni") {
    params_.motion_model = MotionModel::Omni;
  } else {
    throw std::runtime_error(
            "CudaMppiController: unknown motion_model '" + motion_model +
            "' (DiffDrive / Ackermann / Omni)");
  }
  params_.v_max = static_cast<float>(v_max);
  params_.v_min = static_cast<float>(v_min);
  params_.vy_max = static_cast<float>(vy_max);
  params_.w_max = static_cast<float>(w_max);
  params_.min_turning_r = static_cast<float>(min_turning_r);
  params_.v_std = static_cast<float>(v_std);
  params_.vy_std = static_cast<float>(vy_std);
  params_.w_std = static_cast<float>(w_std);
  params_.consider_footprint = consider_footprint;
  params_.lambda = static_cast<float>(lambda);
  params_.goal_weight = static_cast<float>(goal_weight);
  params_.goal_yaw_weight = static_cast<float>(goal_yaw_weight);
  params_.path_weight = static_cast<float>(path_weight);
  params_.path_follow_weight = static_cast<float>(path_follow_weight);
  params_.follow_lookahead = static_cast<float>(follow_lookahead);
  params_.costmap_weight = static_cast<float>(costmap_weight);
  params_.smoothness_weight = static_cast<float>(smoothness_weight);
  params_.backward_weight = static_cast<float>(backward_weight);
  params_.speed_weight = static_cast<float>(speed_weight);
  params_.angular_weight = static_cast<float>(angular_weight);
  params_.yaw_goal_activation_dist = static_cast<float>(yaw_activation);
  params_.enable_retreat = enable_retreat;
  params_.retreat_scale = static_cast<float>(retreat_scale);

  optimizer_ = std::make_unique<MppiGpu>(params_);

  RCLCPP_INFO(
    logger_,
    "Configured CudaMppiController '%s': K=%d, T=%d, dt=%.3f (GPU rollouts)",
    name_.c_str(), params_.batch_size, params_.time_steps, params_.model_dt);
}

void CudaMppiController::cleanup()
{
  optimizer_.reset();
}

void CudaMppiController::activate()
{
  if (optimizer_) {
    optimizer_->reset();
  }
}

void CudaMppiController::deactivate()
{
}

void CudaMppiController::setPlan(const nav_msgs::msg::Path & path)
{
  global_plan_ = path;
}

void CudaMppiController::reset()
{
  if (optimizer_) {
    optimizer_->reset();
  }
}

std::vector<float> CudaMppiController::extractLocalPath(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  float & goal_x, float & goal_y, float & goal_yaw, bool & goal_is_final)
{
  if (global_plan_.poses.empty()) {
    throw nav2_core::InvalidPath("CudaMppiController: received an empty plan");
  }

  const std::string target_frame = costmap_ros_->getGlobalFrameID();
  geometry_msgs::msg::TransformStamped plan_to_costmap;
  try {
    plan_to_costmap = tf_->lookupTransform(
      target_frame, global_plan_.header.frame_id, tf2::TimePointZero,
      tf2::durationFromSec(transform_tolerance_));
  } catch (const tf2::TransformException & ex) {
    throw nav2_core::ControllerTFError(
            std::string("CudaMppiController: failed to transform plan: ") + ex.what());
  }

  // nearest plan point to the robot, in the costmap frame
  size_t nearest = 0;
  double nearest_d2 = std::numeric_limits<double>::max();
  std::vector<geometry_msgs::msg::PoseStamped> transformed(global_plan_.poses.size());
  for (size_t i = 0; i < global_plan_.poses.size(); ++i) {
    tf2::doTransform(global_plan_.poses[i], transformed[i], plan_to_costmap);
    const double dx = transformed[i].pose.position.x - robot_pose.pose.position.x;
    const double dy = transformed[i].pose.position.y - robot_pose.pose.position.y;
    const double d2 = dx * dx + dy * dy;
    if (d2 < nearest_d2) {
      nearest_d2 = d2;
      nearest = i;
    }
  }

  // forward window limited by arc length, downsampled to the GPU path budget
  constexpr size_t kMaxPathPoints = 256;
  size_t end = nearest;
  double arc = 0.0;
  for (size_t i = nearest + 1; i < transformed.size(); ++i) {
    const double dx = transformed[i].pose.position.x - transformed[i - 1].pose.position.x;
    const double dy = transformed[i].pose.position.y - transformed[i - 1].pose.position.y;
    arc += std::hypot(dx, dy);
    end = i;
    if (arc > lookahead_dist_) {
      break;
    }
  }

  const size_t count = end - nearest + 1;
  const size_t stride = std::max<size_t>(1, (count + kMaxPathPoints - 1) / kMaxPathPoints);
  std::vector<float> path_xy;
  path_xy.reserve(2 * kMaxPathPoints);
  for (size_t i = nearest; i <= end; i += stride) {
    path_xy.push_back(static_cast<float>(transformed[i].pose.position.x));
    path_xy.push_back(static_cast<float>(transformed[i].pose.position.y));
  }

  const auto & goal_pose = transformed[end];
  goal_x = static_cast<float>(goal_pose.pose.position.x);
  goal_y = static_cast<float>(goal_pose.pose.position.y);
  goal_yaw = static_cast<float>(tf2::getYaw(goal_pose.pose.orientation));
  goal_is_final = (end + 1 == transformed.size());
  return path_xy;
}

geometry_msgs::msg::TwistStamped CudaMppiController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & /*velocity*/,
  nav2_core::GoalChecker * /*goal_checker*/)
{
  if (!optimizer_) {
    throw nav2_core::ControllerException("CudaMppiController is not configured");
  }

  float goal_x = 0.0f, goal_y = 0.0f, goal_yaw = 0.0f;
  bool goal_is_final = false;
  const std::vector<float> path_xy =
    extractLocalPath(pose, goal_x, goal_y, goal_yaw, goal_is_final);

  std::vector<float> footprint_xy;
  if (params_.consider_footprint) {
    for (const auto & pt : costmap_ros_->getRobotFootprint()) {
      footprint_xy.push_back(static_cast<float>(pt.x));
      footprint_xy.push_back(static_cast<float>(pt.y));
    }
  }

  nav2_costmap_2d::Costmap2D * costmap = costmap_ros_->getCostmap();
  MppiResult result;
  {
    std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*costmap->getMutex());
    result = optimizer_->compute(
      static_cast<float>(pose.pose.position.x),
      static_cast<float>(pose.pose.position.y),
      static_cast<float>(tf2::getYaw(pose.pose.orientation)),
      costmap->getCharMap(),
      static_cast<int>(costmap->getSizeInCellsX()),
      static_cast<int>(costmap->getSizeInCellsY()),
      static_cast<float>(costmap->getOriginX()),
      static_cast<float>(costmap->getOriginY()),
      static_cast<float>(costmap->getResolution()),
      path_xy.data(), static_cast<int>(path_xy.size() / 2),
      goal_x, goal_y, goal_yaw, goal_is_final,
      footprint_xy.data(), static_cast<int>(footprint_xy.size() / 2));
  }

  if (result.all_colliding && !result.retreating) {
    throw nav2_core::NoValidControl(
            "CudaMppiController: all sampled trajectories are in collision");
  }

  geometry_msgs::msg::TwistStamped cmd;
  cmd.header.stamp = pose.header.stamp;
  cmd.header.frame_id = costmap_ros_->getBaseFrameID();
  cmd.twist.linear.x = result.v;
  if (params_.motion_model == MotionModel::Omni) {
    cmd.twist.linear.y = result.vy;
  }
  cmd.twist.angular.z = result.w;
  return cmd;
}

void CudaMppiController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  if (!optimizer_) {
    return;
  }
  if (speed_limit <= 0.0) {
    optimizer_->setSpeedLimit(params_.v_max);
  } else if (percentage) {
    optimizer_->setSpeedLimit(params_.v_max * static_cast<float>(speed_limit / 100.0));
  } else {
    optimizer_->setSpeedLimit(static_cast<float>(speed_limit));
  }
}

}  // namespace cuda_mppi_controller

PLUGINLIB_EXPORT_CLASS(cuda_mppi_controller::CudaMppiController, nav2_core::Controller)
