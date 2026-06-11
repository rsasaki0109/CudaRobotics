#include "cuda_mppi_controller/cuda_mppi_controller.hpp"
#include "cuda_mppi_controller/nav2_compat.hpp"

#include <algorithm>
#include <chrono>
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
                              double transform_tolerance, double diagnostics_log_period)
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
  requireNonNegative("path_angle_weight", params.path_angle_weight);
  requireNonNegative("curvature_speed_weight", params.curvature_speed_weight);
  requireNonNegative("curvature_speed_min", params.curvature_speed_min);
  requireNonNegative("follow_lookahead", params.follow_lookahead);
  requireNonNegative("costmap_weight", params.costmap_weight);
  requireNonNegative("distance_field_weight", params.distance_field_weight);
  requireNonNegative("distance_field_cutoff", params.distance_field_cutoff);
  requireNonNegative("smoothness_weight", params.smoothness_weight);
  requireNonNegative("backward_weight", params.backward_weight);
  requireNonNegative("speed_weight", params.speed_weight);
  requireNonNegative("angular_weight", params.angular_weight);
  requirePositive("collision_cost", params.collision_cost);
  requireNonNegative("yaw_goal_activation_dist", params.yaw_goal_activation_dist);
  requireNonNegative("retreat_scale", params.retreat_scale);

  requirePositive("lookahead_dist", lookahead_dist);
  requireNonNegative("transform_tolerance", transform_tolerance);
  requireNonNegative("diagnostics_log_period", diagnostics_log_period);
}

bool applyControllerParameter(const std::string & key, const rclcpp::Parameter & parameter,
                              MppiParams & params, double & lookahead_dist,
                              double & transform_tolerance,
                              double & diagnostics_log_period,
                              std::string & diagnostics_csv_path,
                              bool & optimizer_params_changed)
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
  } else if (key == "path_angle_weight") {
    params.path_angle_weight = static_cast<float>(parameter.as_double());
  } else if (key == "curvature_speed_weight") {
    params.curvature_speed_weight = static_cast<float>(parameter.as_double());
  } else if (key == "curvature_speed_min") {
    params.curvature_speed_min = static_cast<float>(parameter.as_double());
  } else if (key == "follow_lookahead") {
    params.follow_lookahead = static_cast<float>(parameter.as_double());
  } else if (key == "costmap_weight") {
    params.costmap_weight = static_cast<float>(parameter.as_double());
  } else if (key == "distance_field_weight") {
    params.distance_field_weight = static_cast<float>(parameter.as_double());
  } else if (key == "distance_field_cutoff") {
    params.distance_field_cutoff = static_cast<float>(parameter.as_double());
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
  } else if (key == "diagnostics_log_period") {
    diagnostics_log_period = parameter.as_double();
    return true;
  } else if (key == "diagnostics_csv_path") {
    diagnostics_csv_path = parameter.as_string();
    return true;
  } else {
    return false;
  }
  optimizer_params_changed = true;
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
  double next_diagnostics_log_period = diagnostics_log_period_;
  std::string next_diagnostics_csv_path = diagnostics_csv_path_;

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
  double path_angle_weight = next.path_angle_weight;
  double curvature_speed_weight = next.curvature_speed_weight;
  double curvature_speed_min = next.curvature_speed_min;
  double follow_lookahead = next.follow_lookahead;
  double costmap_weight = next.costmap_weight;
  double distance_field_weight = next.distance_field_weight;
  double distance_field_cutoff = next.distance_field_cutoff;
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
  node->get_parameter(name_ + ".path_angle_weight", path_angle_weight);
  node->get_parameter(name_ + ".curvature_speed_weight", curvature_speed_weight);
  node->get_parameter(name_ + ".curvature_speed_min", curvature_speed_min);
  node->get_parameter(name_ + ".follow_lookahead", follow_lookahead);
  node->get_parameter(name_ + ".costmap_weight", costmap_weight);
  node->get_parameter(name_ + ".distance_field_weight", distance_field_weight);
  node->get_parameter(name_ + ".distance_field_cutoff", distance_field_cutoff);
  node->get_parameter(name_ + ".smoothness_weight", smoothness_weight);
  node->get_parameter(name_ + ".backward_weight", backward_weight);
  node->get_parameter(name_ + ".speed_weight", speed_weight);
  node->get_parameter(name_ + ".angular_weight", angular_weight);
  node->get_parameter(name_ + ".yaw_goal_activation_dist", yaw_activation);
  node->get_parameter(name_ + ".enable_retreat", enable_retreat);
  node->get_parameter(name_ + ".retreat_scale", retreat_scale);
  node->get_parameter(name_ + ".lookahead_dist", next_lookahead_dist);
  node->get_parameter(name_ + ".transform_tolerance", next_transform_tolerance);
  node->get_parameter(name_ + ".diagnostics_log_period", next_diagnostics_log_period);
  node->get_parameter(name_ + ".diagnostics_csv_path", next_diagnostics_csv_path);

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
  next.path_angle_weight = static_cast<float>(path_angle_weight);
  next.curvature_speed_weight = static_cast<float>(curvature_speed_weight);
  next.curvature_speed_min = static_cast<float>(curvature_speed_min);
  next.follow_lookahead = static_cast<float>(follow_lookahead);
  next.costmap_weight = static_cast<float>(costmap_weight);
  next.distance_field_weight = static_cast<float>(distance_field_weight);
  next.distance_field_cutoff = static_cast<float>(distance_field_cutoff);
  next.smoothness_weight = static_cast<float>(smoothness_weight);
  next.backward_weight = static_cast<float>(backward_weight);
  next.speed_weight = static_cast<float>(speed_weight);
  next.angular_weight = static_cast<float>(angular_weight);
  next.yaw_goal_activation_dist = static_cast<float>(yaw_activation);
  next.enable_retreat = enable_retreat;
  next.retreat_scale = static_cast<float>(retreat_scale);

  validateControllerParams(
    next, next_lookahead_dist, next_transform_tolerance, next_diagnostics_log_period);

  params_ = next;
  lookahead_dist_ = next_lookahead_dist;
  transform_tolerance_ = next_transform_tolerance;
  diagnostics_log_period_ = next_diagnostics_log_period;
  diagnostics_csv_path_ = next_diagnostics_csv_path;
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
  declare_param("path_angle_weight", static_cast<double>(params_.path_angle_weight));
  declare_param("curvature_speed_weight", static_cast<double>(params_.curvature_speed_weight));
  declare_param("curvature_speed_min", static_cast<double>(params_.curvature_speed_min));
  declare_param("follow_lookahead", static_cast<double>(params_.follow_lookahead));
  declare_param("costmap_weight", static_cast<double>(params_.costmap_weight));
  declare_param("distance_field_weight", static_cast<double>(params_.distance_field_weight));
  declare_param("distance_field_cutoff", static_cast<double>(params_.distance_field_cutoff));
  declare_param("smoothness_weight", static_cast<double>(params_.smoothness_weight));
  declare_param("backward_weight", static_cast<double>(params_.backward_weight));
  declare_param("speed_weight", static_cast<double>(params_.speed_weight));
  declare_param("angular_weight", static_cast<double>(params_.angular_weight));
  declare_param("yaw_goal_activation_dist", static_cast<double>(params_.yaw_goal_activation_dist));
  declare_param("enable_retreat", params_.enable_retreat);
  declare_param("retreat_scale", static_cast<double>(params_.retreat_scale));
  declare_param("lookahead_dist", lookahead_dist_);
  declare_param("transform_tolerance", transform_tolerance_);
  declare_param("diagnostics_log_period", diagnostics_log_period_);
  declare_param("diagnostics_csv_path", diagnostics_csv_path_);

  updateParamsFromNode(node);
  diagnostics_csv_ = openDiagnosticsCsv(diagnostics_csv_path_);

  optimizer_ = std::make_unique<MppiGpu>(params_);

  param_callback_ =
    node->add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter> & parameters) {
      rcl_interfaces::msg::SetParametersResult result;
      result.successful = true;
      const std::string prefix = name_ + ".";
      MppiParams next_params = params_;
      double next_lookahead_dist = lookahead_dist_;
      double next_transform_tolerance = transform_tolerance_;
      double next_diagnostics_log_period = diagnostics_log_period_;
      std::string next_diagnostics_csv_path = diagnostics_csv_path_;
      bool changed = false;
      bool optimizer_params_changed = false;
      for (const auto & parameter : parameters) {
        const std::string & full_name = parameter.get_name();
        if (full_name.rfind(prefix, 0) != 0) {
          continue;
        }
        const std::string key = full_name.substr(prefix.size());
        changed = applyControllerParameter(
          key, parameter, next_params, next_lookahead_dist, next_transform_tolerance,
          next_diagnostics_log_period, next_diagnostics_csv_path, optimizer_params_changed) ||
                  changed;
      }
      if (!changed) {
        return result;
      }
      try {
        validateControllerParams(
          next_params, next_lookahead_dist, next_transform_tolerance,
          next_diagnostics_log_period);
        std::unique_ptr<MppiGpu> next_optimizer;
        if (optimizer_ && optimizer_params_changed) {
          next_optimizer = std::make_unique<MppiGpu>(next_params);
        }
        DiagnosticsCsv next_diagnostics_csv;
        const bool diagnostics_csv_changed =
          next_diagnostics_csv_path != diagnostics_csv_path_;
        if (diagnostics_csv_changed) {
          next_diagnostics_csv = openDiagnosticsCsv(next_diagnostics_csv_path);
        }
        params_ = next_params;
        lookahead_dist_ = next_lookahead_dist;
        transform_tolerance_ = next_transform_tolerance;
        diagnostics_log_period_ = next_diagnostics_log_period;
        diagnostics_csv_path_ = next_diagnostics_csv_path;
        if (optimizer_params_changed) {
          optimizer_ = std::move(next_optimizer);
        }
        if (diagnostics_csv_changed) {
          diagnostics_csv_ = std::move(next_diagnostics_csv);
        }
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
  diagnostics_csv_ = DiagnosticsCsv{};
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

CudaMppiController::DiagnosticsCsv CudaMppiController::openDiagnosticsCsv(
  const std::string & path) const
{
  DiagnosticsCsv output;
  if (path.empty()) {
    return output;
  }

  output.file.open(path, std::ios::out | std::ios::app);
  if (!output.file.is_open()) {
    throw std::runtime_error(
            "CudaMppiController: failed to open diagnostics_csv_path '" + path + "'");
  }
  output.enabled = true;
  output.file
    << "stamp_sec,solve_ms,best_cost,mean_cost,sampled_rollouts,valid_rollouts,"
    << "valid_rollout_ratio,all_colliding,retreating,path_points,costmap_size_x,"
    << "costmap_size_y,cmd_v,cmd_vy,cmd_w\n";
  return output;
}

void CudaMppiController::emitDiagnostics(
  const MppiResult & result, double solve_ms, int path_points,
  int costmap_size_x, int costmap_size_y)
{
  const auto node = node_.lock();
  const rclcpp::Time now = node ? node->now() : rclcpp::Clock(RCL_ROS_TIME).now();

  if (diagnostics_log_period_ > 0.0 && node) {
    const bool due =
      !has_diagnostics_log_time_ ||
      (now - last_diagnostics_log_time_).seconds() >= diagnostics_log_period_;
    if (due) {
      last_diagnostics_log_time_ = now;
      has_diagnostics_log_time_ = true;
      RCLCPP_INFO(
        logger_,
        "CUDA MPPI diagnostics: solve=%.2f ms valid=%d/%d (%.1f%%) best=%.3f mean=%.3f "
        "retreat=%s cmd=(%.3f, %.3f, %.3f)",
        solve_ms, result.valid_rollouts, result.sampled_rollouts,
        100.0 * result.valid_rollout_ratio, result.best_cost, result.mean_cost,
        result.retreating ? "true" : "false", result.v, result.vy, result.w);
    }
  }

  if (diagnostics_csv_.enabled && diagnostics_csv_.file.is_open()) {
    diagnostics_csv_.file
      << now.seconds() << ','
      << solve_ms << ','
      << result.best_cost << ','
      << result.mean_cost << ','
      << result.sampled_rollouts << ','
      << result.valid_rollouts << ','
      << result.valid_rollout_ratio << ','
      << (result.all_colliding ? 1 : 0) << ','
      << (result.retreating ? 1 : 0) << ','
      << path_points << ','
      << costmap_size_x << ','
      << costmap_size_y << ','
      << result.v << ','
      << result.vy << ','
      << result.w << '\n';
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
  int costmap_size_x = 0;
  int costmap_size_y = 0;
  double solve_ms = 0.0;
  {
    std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*costmap->getMutex());
    costmap_size_x = static_cast<int>(costmap->getSizeInCellsX());
    costmap_size_y = static_cast<int>(costmap->getSizeInCellsY());
    const auto solve_start = std::chrono::steady_clock::now();
    result = optimizer_->compute(
      static_cast<float>(pose.pose.position.x),
      static_cast<float>(pose.pose.position.y),
      static_cast<float>(tf2::getYaw(pose.pose.orientation)),
      costmap->getCharMap(),
      costmap_size_x,
      costmap_size_y,
      static_cast<float>(costmap->getOriginX()),
      static_cast<float>(costmap->getOriginY()),
      static_cast<float>(costmap->getResolution()),
      path_xy.data(), static_cast<int>(path_xy.size() / 2),
      goal_x, goal_y, goal_yaw, goal_is_final,
      footprint_xy.data(), static_cast<int>(footprint_xy.size() / 2));
    const auto solve_end = std::chrono::steady_clock::now();
    solve_ms = std::chrono::duration<double, std::milli>(solve_end - solve_start).count();
  }
  emitDiagnostics(
    result, solve_ms, static_cast<int>(path_xy.size() / 2), costmap_size_x, costmap_size_y);

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
