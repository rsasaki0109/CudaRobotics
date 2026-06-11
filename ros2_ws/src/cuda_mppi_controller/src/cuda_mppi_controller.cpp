#include "cuda_mppi_controller/cuda_mppi_controller.hpp"
#include "cuda_mppi_controller/nav2_compat.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace cuda_mppi_controller
{

namespace
{

MotionModel parseMotionModel(const std::string & motion_model)
{
  if (motion_model == "DiffDrive") {
    return MotionModel::DiffDrive;
  }
  if (motion_model == "Ackermann") {
    return MotionModel::Ackermann;
  }
  if (motion_model == "Omni") {
    return MotionModel::Omni;
  }
  throw std::runtime_error("CudaMppiController: unknown motion_model '" + motion_model +
                           "' (DiffDrive / Ackermann / Omni)");
}

std::string motionModelName(MotionModel motion_model)
{
  switch (motion_model) {
    case MotionModel::DiffDrive:
      return "DiffDrive";
    case MotionModel::Ackermann:
      return "Ackermann";
    case MotionModel::Omni:
      return "Omni";
  }
  return "DiffDrive";
}

void requireParam(bool condition, const std::string & name, const std::string & rule)
{
  if (!condition) {
    throw std::runtime_error("CudaMppiController parameter validation failed: '" + name + "' " +
                             rule);
  }
}

void requireFinite(const std::string & name, double value)
{
  requireParam(std::isfinite(value), name, "must be finite");
}

void requirePositive(const std::string & name, double value)
{
  requireFinite(name, value);
  requireParam(value > 0.0, name, "must be greater than 0");
}

void requireNonNegative(const std::string & name, double value)
{
  requireFinite(name, value);
  requireParam(value >= 0.0, name, "must be non-negative");
}

void validateControllerParams(const MppiParams & params, double lookahead_dist,
                              double transform_tolerance)
{
  requireParam(params.batch_size > 0, "batch_size", "must be greater than 0");
  requireParam(params.time_steps > 0, "time_steps", "must be greater than 0");
  requireParam(params.iteration_count > 0, "iteration_count", "must be greater than 0");

  requirePositive("model_dt", params.model_dt);
  requirePositive("temperature", params.lambda);

  requireFinite("v_min", params.v_min);
  requirePositive("v_max", params.v_max);
  requireParam(params.v_min <= params.v_max, "v_min", "must be <= v_max");
  requireNonNegative("vy_max", params.vy_max);
  requirePositive("w_max", params.w_max);
  requirePositive("min_turning_r", params.min_turning_r);

  requireNonNegative("v_std", params.v_std);
  requireNonNegative("vy_std", params.vy_std);
  requireNonNegative("w_std", params.w_std);

  requireNonNegative("goal_weight", params.goal_weight);
  requireNonNegative("goal_yaw_weight", params.goal_yaw_weight);
  requireNonNegative("path_weight", params.path_weight);
  requireNonNegative("path_follow_weight", params.path_follow_weight);
  requireNonNegative("follow_lookahead", params.follow_lookahead);
  requireNonNegative("costmap_weight", params.costmap_weight);
  requireNonNegative("smoothness_weight", params.smoothness_weight);
  requireNonNegative("backward_weight", params.backward_weight);
  requireNonNegative("speed_weight", params.speed_weight);
  requireNonNegative("angular_weight", params.angular_weight);
  requirePositive("collision_cost", params.collision_cost);
  requireNonNegative("yaw_goal_activation_dist", params.yaw_goal_activation_dist);
  requireNonNegative("retreat_scale", params.retreat_scale);

  requirePositive("lookahead_dist", lookahead_dist);
  requireNonNegative("transform_tolerance", transform_tolerance);
}

bool applyControllerParameter(const std::string & key, const rclcpp::Parameter & parameter,
                              MppiParams & params, double & lookahead_dist,
                              double & transform_tolerance)
{
  if (key == "batch_size") {
    params.batch_size = static_cast<int>(parameter.as_int());
  } else if (key == "time_steps") {
    params.time_steps = static_cast<int>(parameter.as_int());
  } else if (key == "iteration_count") {
    params.iteration_count = static_cast<int>(parameter.as_int());
  } else if (key == "model_dt") {
    params.model_dt = static_cast<float>(parameter.as_double());
  } else if (key == "motion_model") {
    params.motion_model = parseMotionModel(parameter.as_string());
  } else if (key == "v_max") {
    params.v_max = static_cast<float>(parameter.as_double());
  } else if (key == "v_min") {
    params.v_min = static_cast<float>(parameter.as_double());
  } else if (key == "vy_max") {
    params.vy_max = static_cast<float>(parameter.as_double());
  } else if (key == "w_max") {
    params.w_max = static_cast<float>(parameter.as_double());
  } else if (key == "min_turning_r") {
    params.min_turning_r = static_cast<float>(parameter.as_double());
  } else if (key == "v_std") {
    params.v_std = static_cast<float>(parameter.as_double());
  } else if (key == "vy_std") {
    params.vy_std = static_cast<float>(parameter.as_double());
  } else if (key == "w_std") {
    params.w_std = static_cast<float>(parameter.as_double());
  } else if (key == "consider_footprint") {
    params.consider_footprint = parameter.as_bool();
  } else if (key == "temperature") {
    params.lambda = static_cast<float>(parameter.as_double());
  } else if (key == "goal_weight") {
    params.goal_weight = static_cast<float>(parameter.as_double());
  } else if (key == "goal_yaw_weight") {
    params.goal_yaw_weight = static_cast<float>(parameter.as_double());
  } else if (key == "path_weight") {
    params.path_weight = static_cast<float>(parameter.as_double());
  } else if (key == "path_follow_weight") {
    params.path_follow_weight = static_cast<float>(parameter.as_double());
  } else if (key == "follow_lookahead") {
    params.follow_lookahead = static_cast<float>(parameter.as_double());
  } else if (key == "costmap_weight") {
    params.costmap_weight = static_cast<float>(parameter.as_double());
  } else if (key == "smoothness_weight") {
    params.smoothness_weight = static_cast<float>(parameter.as_double());
  } else if (key == "backward_weight") {
    params.backward_weight = static_cast<float>(parameter.as_double());
  } else if (key == "speed_weight") {
    params.speed_weight = static_cast<float>(parameter.as_double());
  } else if (key == "angular_weight") {
    params.angular_weight = static_cast<float>(parameter.as_double());
  } else if (key == "yaw_goal_activation_dist") {
    params.yaw_goal_activation_dist = static_cast<float>(parameter.as_double());
  } else if (key == "enable_retreat") {
    params.enable_retreat = parameter.as_bool();
  } else if (key == "retreat_scale") {
    params.retreat_scale = static_cast<float>(parameter.as_double());
  } else if (key == "lookahead_dist") {
    lookahead_dist = parameter.as_double();
  } else if (key == "transform_tolerance") {
    transform_tolerance = parameter.as_double();
  } else {
    return false;
  }
  return true;
}

}  // namespace

bool CudaMppiController::updateParamsFromNode(
  const rclcpp_lifecycle::LifecycleNode::SharedPtr & node)
{
  if (!node) {
    return false;
  }

  MppiParams next = params_;
  double next_lookahead_dist = lookahead_dist_;
  double next_transform_tolerance = transform_tolerance_;

  int batch_size = next.batch_size;
  int time_steps = next.time_steps;
  int iteration_count = next.iteration_count;
  double model_dt = next.model_dt;
  std::string motion_model = motionModelName(next.motion_model);
  double v_max = next.v_max, v_min = next.v_min, w_max = next.w_max;
  double vy_max = next.vy_max, min_turning_r = next.min_turning_r;
  double v_std = next.v_std, vy_std = next.vy_std, w_std = next.w_std;
  bool consider_footprint = next.consider_footprint;
  double lambda = next.lambda;
  double goal_weight = next.goal_weight;
  double goal_yaw_weight = next.goal_yaw_weight;
  double path_weight = next.path_weight;
  double path_follow_weight = next.path_follow_weight;
  double follow_lookahead = next.follow_lookahead;
  double costmap_weight = next.costmap_weight;
  double smoothness_weight = next.smoothness_weight;
  double backward_weight = next.backward_weight;
  double speed_weight = next.speed_weight;
  double angular_weight = next.angular_weight;
  double yaw_activation = next.yaw_goal_activation_dist;
  bool enable_retreat = next.enable_retreat;
  double retreat_scale = next.retreat_scale;

  node->get_parameter(name_ + ".batch_size", batch_size);
  node->get_parameter(name_ + ".time_steps", time_steps);
  node->get_parameter(name_ + ".iteration_count", iteration_count);
  node->get_parameter(name_ + ".model_dt", model_dt);
  node->get_parameter(name_ + ".motion_model", motion_model);
  node->get_parameter(name_ + ".v_max", v_max);
  node->get_parameter(name_ + ".v_min", v_min);
  node->get_parameter(name_ + ".vy_max", vy_max);
  node->get_parameter(name_ + ".w_max", w_max);
  node->get_parameter(name_ + ".min_turning_r", min_turning_r);
  node->get_parameter(name_ + ".v_std", v_std);
  node->get_parameter(name_ + ".vy_std", vy_std);
  node->get_parameter(name_ + ".w_std", w_std);
  node->get_parameter(name_ + ".consider_footprint", consider_footprint);
  node->get_parameter(name_ + ".temperature", lambda);
  node->get_parameter(name_ + ".goal_weight", goal_weight);
  node->get_parameter(name_ + ".goal_yaw_weight", goal_yaw_weight);
  node->get_parameter(name_ + ".path_weight", path_weight);
  node->get_parameter(name_ + ".path_follow_weight", path_follow_weight);
  node->get_parameter(name_ + ".follow_lookahead", follow_lookahead);
  node->get_parameter(name_ + ".costmap_weight", costmap_weight);
  node->get_parameter(name_ + ".smoothness_weight", smoothness_weight);
  node->get_parameter(name_ + ".backward_weight", backward_weight);
  node->get_parameter(name_ + ".speed_weight", speed_weight);
  node->get_parameter(name_ + ".angular_weight", angular_weight);
  node->get_parameter(name_ + ".yaw_goal_activation_dist", yaw_activation);
  node->get_parameter(name_ + ".enable_retreat", enable_retreat);
  node->get_parameter(name_ + ".retreat_scale", retreat_scale);
  node->get_parameter(name_ + ".lookahead_dist", next_lookahead_dist);
  node->get_parameter(name_ + ".transform_tolerance", next_transform_tolerance);

  next.batch_size = batch_size;
  next.time_steps = time_steps;
  next.iteration_count = iteration_count;
  next.model_dt = static_cast<float>(model_dt);
  next.motion_model = parseMotionModel(motion_model);
  next.v_max = static_cast<float>(v_max);
  next.v_min = static_cast<float>(v_min);
  next.vy_max = static_cast<float>(vy_max);
  next.w_max = static_cast<float>(w_max);
  next.min_turning_r = static_cast<float>(min_turning_r);
  next.v_std = static_cast<float>(v_std);
  next.vy_std = static_cast<float>(vy_std);
  next.w_std = static_cast<float>(w_std);
  next.consider_footprint = consider_footprint;
  next.lambda = static_cast<float>(lambda);
  next.goal_weight = static_cast<float>(goal_weight);
  next.goal_yaw_weight = static_cast<float>(goal_yaw_weight);
  next.path_weight = static_cast<float>(path_weight);
  next.path_follow_weight = static_cast<float>(path_follow_weight);
  next.follow_lookahead = static_cast<float>(follow_lookahead);
  next.costmap_weight = static_cast<float>(costmap_weight);
  next.smoothness_weight = static_cast<float>(smoothness_weight);
  next.backward_weight = static_cast<float>(backward_weight);
  next.speed_weight = static_cast<float>(speed_weight);
  next.angular_weight = static_cast<float>(angular_weight);
  next.yaw_goal_activation_dist = static_cast<float>(yaw_activation);
  next.enable_retreat = enable_retreat;
  next.retreat_scale = static_cast<float>(retreat_scale);

  validateControllerParams(next, next_lookahead_dist, next_transform_tolerance);

  params_ = next;
  lookahead_dist_ = next_lookahead_dist;
  transform_tolerance_ = next_transform_tolerance;
  return true;
}

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
  auto declare_param = [&](const std::string & param, auto default_value) {
    declare_parameter_if_not_declared(node, name_ + "." + param,
                                      rclcpp::ParameterValue(default_value));
  };

  declare_param("batch_size", params_.batch_size);
  declare_param("time_steps", params_.time_steps);
  declare_param("iteration_count", params_.iteration_count);
  declare_param("model_dt", static_cast<double>(params_.model_dt));
  declare_param("motion_model", motionModelName(params_.motion_model));
  declare_param("v_max", static_cast<double>(params_.v_max));
  declare_param("v_min", static_cast<double>(params_.v_min));
  declare_param("vy_max", static_cast<double>(params_.vy_max));
  declare_param("w_max", static_cast<double>(params_.w_max));
  declare_param("min_turning_r", static_cast<double>(params_.min_turning_r));
  declare_param("v_std", static_cast<double>(params_.v_std));
  declare_param("vy_std", static_cast<double>(params_.vy_std));
  declare_param("w_std", static_cast<double>(params_.w_std));
  declare_param("consider_footprint", params_.consider_footprint);
  declare_param("temperature", static_cast<double>(params_.lambda));
  declare_param("goal_weight", static_cast<double>(params_.goal_weight));
  declare_param("goal_yaw_weight", static_cast<double>(params_.goal_yaw_weight));
  declare_param("path_weight", static_cast<double>(params_.path_weight));
  declare_param("path_follow_weight", static_cast<double>(params_.path_follow_weight));
  declare_param("follow_lookahead", static_cast<double>(params_.follow_lookahead));
  declare_param("costmap_weight", static_cast<double>(params_.costmap_weight));
  declare_param("smoothness_weight", static_cast<double>(params_.smoothness_weight));
  declare_param("backward_weight", static_cast<double>(params_.backward_weight));
  declare_param("speed_weight", static_cast<double>(params_.speed_weight));
  declare_param("angular_weight", static_cast<double>(params_.angular_weight));
  declare_param("yaw_goal_activation_dist", static_cast<double>(params_.yaw_goal_activation_dist));
  declare_param("enable_retreat", params_.enable_retreat);
  declare_param("retreat_scale", static_cast<double>(params_.retreat_scale));
  declare_param("lookahead_dist", lookahead_dist_);
  declare_param("transform_tolerance", transform_tolerance_);

  updateParamsFromNode(node);

  optimizer_ = std::make_unique<MppiGpu>(params_);

  param_callback_ =
    node->add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter> & parameters) {
      rcl_interfaces::msg::SetParametersResult result;
      result.successful = true;
      const std::string prefix = name_ + ".";
      MppiParams next_params = params_;
      double next_lookahead_dist = lookahead_dist_;
      double next_transform_tolerance = transform_tolerance_;
      bool changed = false;
      for (const auto & parameter : parameters) {
        const std::string & full_name = parameter.get_name();
        if (full_name.rfind(prefix, 0) != 0) {
          continue;
        }
        const std::string key = full_name.substr(prefix.size());
        changed = applyControllerParameter(key, parameter, next_params, next_lookahead_dist,
                                           next_transform_tolerance) ||
                  changed;
      }
      if (!changed) {
        return result;
      }
      try {
        validateControllerParams(next_params, next_lookahead_dist, next_transform_tolerance);
        std::unique_ptr<MppiGpu> next_optimizer;
        if (optimizer_) {
          next_optimizer = std::make_unique<MppiGpu>(next_params);
        }
        params_ = next_params;
        lookahead_dist_ = next_lookahead_dist;
        transform_tolerance_ = next_transform_tolerance;
        optimizer_ = std::move(next_optimizer);
      } catch (const std::exception & ex) {
        result.successful = false;
        result.reason = ex.what();
      }
      return result;
    });

  RCLCPP_INFO(
    logger_,
    "Configured CudaMppiController '%s': K=%d, T=%d, dt=%.3f (GPU rollouts)",
    name_.c_str(), params_.batch_size, params_.time_steps, params_.model_dt);
}

void CudaMppiController::cleanup()
{
  param_callback_.reset();
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
    throw ControllerInvalidPath("CudaMppiController: received an empty plan");
  }

  const std::string target_frame = costmap_ros_->getGlobalFrameID();
  geometry_msgs::msg::TransformStamped plan_to_costmap;
  try {
    plan_to_costmap = tf_->lookupTransform(
      target_frame, global_plan_.header.frame_id, tf2::TimePointZero,
      tf2::durationFromSec(transform_tolerance_));
  } catch (const tf2::TransformException & ex) {
    throw ControllerTFError(
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
    throw ControllerException("CudaMppiController is not configured");
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
    throw NoValidControl(
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
