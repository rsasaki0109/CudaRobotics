// Head-to-head closed-loop benchmark: nav2_mppi_controller (CPU) vs
// cuda_mppi_controller (GPU), both loaded through pluginlib exactly as
// nav2's controller_server loads them, driving the same unicycle plant
// through synthetic costmaps.
//
// Usage: controller_benchmark <out_dir> [scenario] [preset]
//   scenario: wall_gap | narrow_corridor | u_turn | double_gap | moving_crossing
//             | all | esdf | path_angle | curvature_speed
//             (default: wall_gap)
//   preset  : full | quick | cpu_gpu (standard scenarios only; default: full)
//   writes <out_dir>/<scenario>/summary.csv and <out_dir>/<scenario>/traj_<label>.csv
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav2_core/controller.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav_msgs/msg/path.hpp"
#include "pluginlib/class_loader.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "tf2/utils.h"
#include "tf2_ros/buffer.h"

namespace
{

constexpr double kControlDt = 0.05;   // 20 Hz
constexpr int kMaxSteps = 1200;
constexpr double kInscribedRadius = 0.2;
constexpr double kCostScaling = 3.0;
constexpr double kGoalTol = 0.25;

struct Scenario
{
  std::string name;
  double start_x;
  double start_y;
  double goal_x;
  double goal_y;
  std::function<bool(double, double)> lethal;
  std::function<bool(double, double, double)> dynamic_lethal;
  std::function<void(nav2_costmap_2d::Costmap2D &, double)> paint_costmap;
  std::function<nav_msgs::msg::Path(const rclcpp::Time &)> make_plan;
  bool dynamic_costmap = false;
  int dynamic_repaint_period = 1;
  bool inflate_costmap = true;
};

bool inRect(double x, double y, double x0, double x1, double y0, double y1)
{
  return x >= x0 && x < x1 && y >= y0 && y < y1;
}

bool scenarioLethal(const Scenario & scenario, double x, double y, double sim_time)
{
  return (scenario.lethal && scenario.lethal(x, y)) ||
         (scenario.dynamic_lethal && scenario.dynamic_lethal(x, y, sim_time));
}

void clearCostmap(nav2_costmap_2d::Costmap2D & costmap)
{
  costmap.resetMapToValue(
    0, 0, costmap.getSizeInCellsX(), costmap.getSizeInCellsY(), nav2_costmap_2d::FREE_SPACE);
}

void paintRect(
  nav2_costmap_2d::Costmap2D & costmap,
  double x0, double x1, double y0, double y1, unsigned char cost)
{
  const double origin_x = costmap.getOriginX();
  const double origin_y = costmap.getOriginY();
  const double resolution = costmap.getResolution();
  const int nx = static_cast<int>(costmap.getSizeInCellsX());
  const int ny = static_cast<int>(costmap.getSizeInCellsY());
  const int mx0 = std::max(0, static_cast<int>(std::floor((x0 - origin_x) / resolution)));
  const int mx1 = std::min(nx - 1, static_cast<int>(std::ceil((x1 - origin_x) / resolution)));
  const int my0 = std::max(0, static_cast<int>(std::floor((y0 - origin_y) / resolution)));
  const int my1 = std::min(ny - 1, static_cast<int>(std::ceil((y1 - origin_y) / resolution)));
  if (mx0 > mx1 || my0 > my1) {
    return;
  }
  for (int my = my0; my <= my1; ++my) {
    for (int mx = mx0; mx <= mx1; ++mx) {
      costmap.setCost(static_cast<unsigned int>(mx), static_cast<unsigned int>(my), cost);
    }
  }
}

void appendPathSegment(
  nav_msgs::msg::Path & path,
  double x0, double y0, double x1, double y1,
  double step = 0.05)
{
  const double len = std::hypot(x1 - x0, y1 - y0);
  const int steps = std::max(1, static_cast<int>(len / step));
  for (int i = 0; i <= steps; ++i) {
    if (!path.poses.empty() && i == 0) {
      continue;
    }
    const double t = static_cast<double>(i) / steps;
    geometry_msgs::msg::PoseStamped p;
    p.header = path.header;
    p.pose.position.x = x0 + t * (x1 - x0);
    p.pose.position.y = y0 + t * (y1 - y0);
    p.pose.orientation.w = 1.0;
    path.poses.push_back(p);
  }
}

Scenario makeWallGap()
{
  Scenario s;
  s.name = "wall_gap";
  s.start_x = 1.0;
  s.start_y = 5.0;
  s.goal_x = 9.0;
  s.goal_y = 5.0;
  s.lethal = [](double x, double y) {
      return inRect(x, y, 4.9, 5.1, 0.0, 4.0) || inRect(x, y, 4.9, 5.1, 6.0, 10.0);
    };
  s.make_plan = [](const rclcpp::Time & stamp) {
      nav_msgs::msg::Path path;
      path.header.frame_id = "odom";
      path.header.stamp = stamp;
      for (double x = 1.0; x <= 9.0 + 1.0e-6; x += 0.05) {
        geometry_msgs::msg::PoseStamped p;
        p.header = path.header;
        p.pose.position.x = x;
        p.pose.position.y = 5.0;
        p.pose.orientation.w = 1.0;
        path.poses.push_back(p);
      }
      return path;
    };
  return s;
}

Scenario makeNarrowCorridor()
{
  Scenario s;
  s.name = "narrow_corridor";
  s.start_x = 1.0;
  s.start_y = 5.0;
  s.goal_x = 9.0;
  s.goal_y = 5.0;
  s.lethal = [](double x, double y) {
      if (x < 2.5 || x > 7.5) {
        return false;
      }
      return inRect(x, y, 2.5, 7.5, 0.0, 4.7) || inRect(x, y, 2.5, 7.5, 5.3, 10.0);
    };
  s.make_plan = [](const rclcpp::Time & stamp) {
      nav_msgs::msg::Path path;
      path.header.frame_id = "odom";
      path.header.stamp = stamp;
      for (double x = 1.0; x <= 9.0 + 1.0e-6; x += 0.05) {
        geometry_msgs::msg::PoseStamped p;
        p.header = path.header;
        p.pose.position.x = x;
        p.pose.position.y = 5.0;
        p.pose.orientation.w = 1.0;
        path.poses.push_back(p);
      }
      return path;
    };
  return s;
}

Scenario makeUTurn()
{
  Scenario s;
  s.name = "u_turn";
  s.start_x = 1.5;
  s.start_y = 1.5;
  s.goal_x = 1.5;
  s.goal_y = 8.5;
  s.lethal = [](double x, double y) {
      return inRect(x, y, 1.0, 8.0, 4.5, 5.0);
    };
  s.make_plan = [](const rclcpp::Time & stamp) {
      nav_msgs::msg::Path path;
      path.header.frame_id = "odom";
      path.header.stamp = stamp;
      const std::array<std::array<double, 2>, 4> pts = {{
        {1.5, 1.5}, {8.5, 1.5}, {8.5, 8.5}, {1.5, 8.5}}};
      for (size_t seg = 0; seg + 1 < pts.size(); ++seg) {
        appendPathSegment(path, pts[seg][0], pts[seg][1], pts[seg + 1][0], pts[seg + 1][1]);
      }
      return path;
    };
  return s;
}

Scenario makeDoubleGap()
{
  Scenario s;
  s.name = "double_gap";
  s.start_x = 1.0;
  s.start_y = 5.0;
  s.goal_x = 9.0;
  s.goal_y = 5.0;
  s.lethal = [](double x, double y) {
      const bool first_wall =
        inRect(x, y, 3.8, 4.0, 0.0, 2.2) || inRect(x, y, 3.8, 4.0, 3.8, 10.0);
      const bool second_wall =
        inRect(x, y, 6.0, 6.2, 0.0, 6.2) || inRect(x, y, 6.0, 6.2, 7.8, 10.0);
      return first_wall || second_wall;
    };
  s.make_plan = [](const rclcpp::Time & stamp) {
      nav_msgs::msg::Path path;
      path.header.frame_id = "odom";
      path.header.stamp = stamp;
      const std::array<std::array<double, 2>, 6> pts = {{
        {1.0, 5.0}, {3.4, 3.0}, {4.4, 3.0}, {5.6, 7.0}, {6.6, 7.0}, {9.0, 5.0}}};
      for (size_t seg = 0; seg + 1 < pts.size(); ++seg) {
        appendPathSegment(path, pts[seg][0], pts[seg][1], pts[seg + 1][0], pts[seg + 1][1]);
      }
      return path;
    };
  return s;
}

Scenario makeMovingCrossing()
{
  Scenario s;
  s.name = "moving_crossing";
  s.start_x = 1.0;
  s.start_y = 5.0;
  s.goal_x = 9.0;
  s.goal_y = 5.0;
  s.lethal = [](double, double) {
      return false;
    };
  s.dynamic_lethal = [](double x, double y, double sim_time) {
      const double center_y = 1.0 + 0.5 * sim_time;
      return inRect(x, y, 4.65, 5.35, center_y - 0.45, center_y + 0.45);
    };
  s.dynamic_costmap = true;
  s.dynamic_repaint_period = 4;  // 5 Hz obstacle-map updates inside a 20 Hz control loop.
  s.inflate_costmap = false;  // keep per-step repaint cheap for this moving-obstacle smoke.
  s.paint_costmap = [](nav2_costmap_2d::Costmap2D & costmap, double sim_time) {
      const double center_y = 1.0 + 0.5 * sim_time;
      paintRect(
        costmap, 4.65, 5.35, center_y - 0.45, center_y + 0.45,
        nav2_costmap_2d::LETHAL_OBSTACLE);
    };
  s.make_plan = [](const rclcpp::Time & stamp) {
      nav_msgs::msg::Path path;
      path.header.frame_id = "odom";
      path.header.stamp = stamp;
      appendPathSegment(path, 1.0, 5.0, 9.0, 5.0);
      return path;
    };
  return s;
}

std::vector<Scenario> allScenarios()
{
  return {
    makeWallGap(), makeNarrowCorridor(), makeUTurn(), makeDoubleGap(), makeMovingCrossing()};
}

void paintCostmap(
  nav2_costmap_2d::Costmap2D & costmap, const Scenario & scenario, double sim_time)
{
  const unsigned int nx = costmap.getSizeInCellsX();
  const unsigned int ny = costmap.getSizeInCellsY();
  clearCostmap(costmap);
  if (scenario.paint_costmap) {
    scenario.paint_costmap(costmap, sim_time);
    return;
  }

  std::vector<std::pair<double, double>> lethal_centers;
  for (unsigned int my = 0; my < ny; ++my) {
    for (unsigned int mx = 0; mx < nx; ++mx) {
      double wx, wy;
      costmap.mapToWorld(mx, my, wx, wy);
      if (scenarioLethal(scenario, wx, wy, sim_time)) {
        costmap.setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
        lethal_centers.emplace_back(wx, wy);
      }
    }
  }

  if (!scenario.inflate_costmap || lethal_centers.empty()) {
    return;
  }

  for (unsigned int my = 0; my < ny; ++my) {
    for (unsigned int mx = 0; mx < nx; ++mx) {
      if (costmap.getCost(mx, my) == nav2_costmap_2d::LETHAL_OBSTACLE) {
        continue;
      }
      double wx, wy;
      costmap.mapToWorld(mx, my, wx, wy);
      double d = 1.0e9;
      for (const auto & c : lethal_centers) {
        d = std::min(d, std::hypot(wx - c.first, wy - c.second));
        if (d < 1.0e-3) {
          break;
        }
      }
      if (d <= kInscribedRadius) {
        costmap.setCost(mx, my, nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE);
      } else if (d < 1.2) {
        const double c =
          (nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE - 1) *
          std::exp(-kCostScaling * (d - kInscribedRadius));
        if (c >= 1.0) {
          costmap.setCost(mx, my, static_cast<unsigned char>(c));
        }
      }
    }
  }
}

struct RunResult
{
  bool success = false;
  bool collided = false;
  int steps = 0;
  int exceptions = 0;
  double mean_ms = 0.0;
  double max_ms = 0.0;
  double p95_ms = 0.0;
  double distance_m = 0.0;
  double mean_speed_mps = 0.0;
  double max_speed_mps = 0.0;
  double mean_abs_w_radps = 0.0;
  double max_abs_w_radps = 0.0;
  double mean_abs_curvature = 0.0;
  std::vector<std::array<double, 3>> traj;
};

RunResult runClosedLoop(
  nav2_core::Controller & controller,
  const rclcpp_lifecycle::LifecycleNode::SharedPtr & node,
  nav2_costmap_2d::Costmap2D & costmap,
  const Scenario & scenario)
{
  RunResult res;
  controller.setPlan(scenario.make_plan(node->now()));

  double x = scenario.start_x, y = scenario.start_y, yaw = 0.0;
  geometry_msgs::msg::Twist cmd;
  std::vector<double> solve_ms;
  solve_ms.reserve(kMaxSteps);
  double sum_speed = 0.0;
  double sum_abs_w = 0.0;
  double sum_abs_curvature = 0.0;
  int command_samples = 0;

  for (res.steps = 0; res.steps < kMaxSteps; ++res.steps) {
    const double sim_time = res.steps * kControlDt;
    if (scenario.dynamic_costmap && res.steps % scenario.dynamic_repaint_period == 0) {
      paintCostmap(costmap, scenario, sim_time);
    }

    res.traj.push_back({x, y, yaw});
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = "odom";
    pose.header.stamp = node->now();
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.orientation.z = std::sin(yaw / 2.0);
    pose.pose.orientation.w = std::cos(yaw / 2.0);

    const auto t0 = std::chrono::steady_clock::now();
    try {
      cmd = controller.computeVelocityCommands(pose, cmd, nullptr).twist;
    } catch (const std::exception &) {
      ++res.exceptions;
      cmd = geometry_msgs::msg::Twist();
    }
    const auto t1 = std::chrono::steady_clock::now();
    solve_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

    const double speed = std::hypot(cmd.linear.x, cmd.linear.y);
    const double abs_w = std::abs(cmd.angular.z);
    res.distance_m += speed * kControlDt;
    sum_speed += speed;
    sum_abs_w += abs_w;
    if (speed > 1.0e-3) {
      sum_abs_curvature += abs_w / speed;
    }
    res.max_speed_mps = std::max(res.max_speed_mps, speed);
    res.max_abs_w_radps = std::max(res.max_abs_w_radps, abs_w);
    ++command_samples;

    x += kControlDt * (
      cmd.linear.x * std::cos(yaw) - cmd.linear.y * std::sin(yaw));
    y += kControlDt * (
      cmd.linear.x * std::sin(yaw) + cmd.linear.y * std::cos(yaw));
    yaw = std::atan2(
      std::sin(yaw + kControlDt * cmd.angular.z),
      std::cos(yaw + kControlDt * cmd.angular.z));

    if (scenarioLethal(scenario, x, y, (res.steps + 1) * kControlDt)) {
      res.collided = true;
      break;
    }
    if (std::hypot(x - scenario.goal_x, y - scenario.goal_y) < kGoalTol) {
      res.success = true;
      break;
    }
  }

  if (!solve_ms.empty()) {
    std::vector<double> sorted = solve_ms;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0.0;
    for (double v : solve_ms) {
      sum += v;
    }
    res.mean_ms = sum / solve_ms.size();
    res.max_ms = sorted.back();
    res.p95_ms = sorted[static_cast<size_t>(0.95 * (sorted.size() - 1))];
  }
  if (command_samples > 0) {
    const double inv = 1.0 / static_cast<double>(command_samples);
    res.mean_speed_mps = sum_speed * inv;
    res.mean_abs_w_radps = sum_abs_w * inv;
    res.mean_abs_curvature = sum_abs_curvature * inv;
  }
  return res;
}

struct Config
{
  std::string label;
  std::string plugin;
  int batch_size;
  std::string motion_model = "DiffDrive";
  double distance_field_weight = 0.0;
  double distance_field_cutoff = 0.75;
  double path_angle_weight = 0.25;
  double curvature_speed_weight = 0.0;
  double curvature_speed_min = 0.18;
};

std::vector<Config> benchmarkConfigs(bool include_motion_models)
{
  std::vector<Config> configs = {
    {"cpu_mppi_K1000", "nav2_mppi_controller::MPPIController", 1000},
    {"cpu_mppi_K2000", "nav2_mppi_controller::MPPIController", 2000},
    {"cpu_mppi_K5000", "nav2_mppi_controller::MPPIController", 5000},
    {"cpu_mppi_K10000", "nav2_mppi_controller::MPPIController", 10000},
    {"gpu_mppi_K2048", "cuda_mppi_controller::CudaMppiController", 2048},
    {"gpu_mppi_K8192", "cuda_mppi_controller::CudaMppiController", 8192},
    {"gpu_mppi_K16384", "cuda_mppi_controller::CudaMppiController", 16384},
    {"gpu_mppi_K65536", "cuda_mppi_controller::CudaMppiController", 65536},
  };
  if (include_motion_models) {
    configs.push_back(
      {"gpu_ackermann_K8192", "cuda_mppi_controller::CudaMppiController", 8192, "Ackermann"});
    configs.push_back(
      {"gpu_omni_K8192", "cuda_mppi_controller::CudaMppiController", 8192, "Omni"});
  }
  return configs;
}

std::vector<Config> quickBenchmarkConfigs()
{
  return {
    {"gpu_mppi_K2048", "cuda_mppi_controller::CudaMppiController", 2048},
    {"gpu_mppi_K8192", "cuda_mppi_controller::CudaMppiController", 8192},
  };
}

std::vector<Config> cpuGpuBenchmarkConfigs()
{
  return {
    {"cpu_mppi_K2000", "nav2_mppi_controller::MPPIController", 2000},
    {"gpu_mppi_K8192", "cuda_mppi_controller::CudaMppiController", 8192},
  };
}

std::vector<Config> esdfBenchmarkConfigs()
{
  return {
    {"gpu_costmap_K8192", "cuda_mppi_controller::CudaMppiController", 8192},
    {"gpu_esdf_K8192", "cuda_mppi_controller::CudaMppiController", 8192,
      "DiffDrive", 12.0, 0.8},
  };
}

std::vector<Config> pathAngleBenchmarkConfigs()
{
  return {
    {"gpu_costmap_K8192_no_path_angle", "cuda_mppi_controller::CudaMppiController", 8192,
      "DiffDrive", 0.0, 0.75, 0.0},
    {"gpu_costmap_K8192_path_angle", "cuda_mppi_controller::CudaMppiController", 8192,
      "DiffDrive", 0.0, 0.75, 0.25},
  };
}

std::vector<Config> curvatureSpeedBenchmarkConfigs()
{
  return {
    {"gpu_costmap_K8192_no_curvature_speed", "cuda_mppi_controller::CudaMppiController", 8192,
      "DiffDrive", 0.0, 0.75, 0.25, 0.0, 0.18},
    {"gpu_costmap_K8192_curvature_speed", "cuda_mppi_controller::CudaMppiController", 8192,
      "DiffDrive", 0.0, 0.75, 0.25, 8.0, 0.18},
  };
}

void runScenario(
  const Scenario & scenario,
  const std::string & out_dir,
  const std::vector<Config> & configs,
  pluginlib::ClassLoader<nav2_core::Controller> & loader,
  const std::shared_ptr<tf2_ros::Buffer> & tf)
{
  rclcpp::NodeOptions costmap_options;
  costmap_options.parameter_overrides(
  {
    rclcpp::Parameter("global_frame", "odom"),
    rclcpp::Parameter("robot_base_frame", "odom"),
    rclcpp::Parameter("rolling_window", false),
    rclcpp::Parameter("width", 10),
    rclcpp::Parameter("height", 10),
    rclcpp::Parameter("resolution", 0.05),
    rclcpp::Parameter("robot_radius", kInscribedRadius),
    rclcpp::Parameter("plugins", std::vector<std::string>{}),
    rclcpp::Parameter("filters", std::vector<std::string>{}),
  });
  auto local_costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>(costmap_options);
  local_costmap->configure();
  paintCostmap(*local_costmap->getCostmap(), scenario, 0.0);

  const std::string scenario_dir = out_dir + "/" + scenario.name;
  std::filesystem::create_directories(scenario_dir);

  std::ofstream summary(scenario_dir + "/summary.csv");
  summary << "scenario,label,plugin,batch_size,motion_model,success,collided,steps,sim_s,"
    "mean_ms,p95_ms,max_ms,exceptions,distance_m,mean_speed_mps,max_speed_mps,"
    "mean_abs_w_radps,max_abs_w_radps,mean_abs_curvature,"
    "distance_field_weight,distance_field_cutoff,"
    "path_angle_weight,curvature_speed_weight,curvature_speed_min\n";

  for (const auto & cfg : configs) {
    rclcpp::NodeOptions options;
    std::vector<rclcpp::Parameter> params = {
      rclcpp::Parameter("controller_frequency", 20.0),
      rclcpp::Parameter("FollowPath.batch_size", cfg.batch_size),
      rclcpp::Parameter("FollowPath.time_steps", 56),
      rclcpp::Parameter("FollowPath.model_dt", kControlDt),
      rclcpp::Parameter("FollowPath.iteration_count", 1),
      rclcpp::Parameter("FollowPath.vx_max", 0.5),
      rclcpp::Parameter("FollowPath.vx_min", -0.35),
      rclcpp::Parameter("FollowPath.wz_max", 1.9),
      rclcpp::Parameter("FollowPath.v_max", 0.5),
      rclcpp::Parameter("FollowPath.v_min", -0.35),
      rclcpp::Parameter("FollowPath.w_max", 1.9),
      rclcpp::Parameter("FollowPath.motion_model", cfg.motion_model),
      rclcpp::Parameter("FollowPath.path_angle_weight", cfg.path_angle_weight),
      rclcpp::Parameter("FollowPath.curvature_speed_weight", cfg.curvature_speed_weight),
      rclcpp::Parameter("FollowPath.curvature_speed_min", cfg.curvature_speed_min),
      rclcpp::Parameter("FollowPath.distance_field_weight", cfg.distance_field_weight),
      rclcpp::Parameter("FollowPath.distance_field_cutoff", cfg.distance_field_cutoff),
      rclcpp::Parameter("FollowPath.visualize", false),
      rclcpp::Parameter(
        "FollowPath.critics", std::vector<std::string>{
        "ConstraintCritic", "CostCritic", "GoalCritic", "GoalAngleCritic",
        "PathAlignCritic", "PathFollowCritic", "PathAngleCritic",
        "PreferForwardCritic"}),
    };
    options.parameter_overrides(params);
    options.arguments(
      {"--ros-args", "-r", std::string("__node:=bench_") + cfg.label});
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "bench_" + cfg.label, options);

    auto controller = loader.createSharedInstance(cfg.plugin);
    controller->configure(node, "FollowPath", tf, local_costmap);
    controller->activate();

    std::printf("=== %s / %s (%s) ===\n", scenario.name.c_str(), cfg.label.c_str(), cfg.plugin.c_str());
    const RunResult r = runClosedLoop(*controller, node, *local_costmap->getCostmap(), scenario);
    std::printf(
      "  %s steps=%d sim=%.1fs solve mean=%.2fms p95=%.2fms max=%.2fms "
      "dist=%.2fm mean_v=%.2fm/s max_w=%.2frad/s exc=%d\n",
      r.success ? "SUCCESS" : (r.collided ? "COLLIDED" : "TIMEOUT"),
      r.steps, r.steps * kControlDt, r.mean_ms, r.p95_ms, r.max_ms,
      r.distance_m, r.mean_speed_mps, r.max_abs_w_radps, r.exceptions);

    summary << scenario.name << ',' << cfg.label << ',' << cfg.plugin << ','
            << cfg.batch_size << ',' << cfg.motion_model << ','
            << (r.success ? 1 : 0) << ',' << (r.collided ? 1 : 0) << ','
            << r.steps << ',' << r.steps * kControlDt << ','
            << r.mean_ms << ',' << r.p95_ms << ',' << r.max_ms << ','
            << r.exceptions << ','
            << r.distance_m << ',' << r.mean_speed_mps << ',' << r.max_speed_mps << ','
            << r.mean_abs_w_radps << ',' << r.max_abs_w_radps << ','
            << r.mean_abs_curvature << ','
            << cfg.distance_field_weight << ',' << cfg.distance_field_cutoff << ','
            << cfg.path_angle_weight << ','
            << cfg.curvature_speed_weight << ',' << cfg.curvature_speed_min << '\n';

    std::ofstream traj(scenario_dir + "/traj_" + cfg.label + ".csv");
    traj << "x,y,yaw\n";
    for (const auto & p : r.traj) {
      traj << p[0] << ',' << p[1] << ',' << p[2] << '\n';
    }

    controller->deactivate();
    controller->cleanup();
  }
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  const std::string out_dir = argc > 1 ? argv[1] : ".";
  const std::string scenario_arg = argc > 2 ? argv[2] : "wall_gap";
  const std::string preset_arg = argc > 3 ? argv[3] : "full";
  std::filesystem::create_directories(out_dir);

  rclcpp::NodeOptions costmap_options;
  costmap_options.parameter_overrides(
  {
    rclcpp::Parameter("global_frame", "odom"),
    rclcpp::Parameter("robot_base_frame", "odom"),
    rclcpp::Parameter("rolling_window", false),
    rclcpp::Parameter("width", 10),
    rclcpp::Parameter("height", 10),
    rclcpp::Parameter("resolution", 0.05),
    rclcpp::Parameter("robot_radius", kInscribedRadius),
    rclcpp::Parameter("plugins", std::vector<std::string>{}),
    rclcpp::Parameter("filters", std::vector<std::string>{}),
  });

  std::vector<Scenario> scenarios;
  const bool esdf_benchmark = scenario_arg == "esdf";
  const bool path_angle_benchmark = scenario_arg == "path_angle";
  const bool curvature_speed_benchmark = scenario_arg == "curvature_speed";
  if (scenario_arg == "all" || esdf_benchmark || path_angle_benchmark ||
    curvature_speed_benchmark)
  {
    scenarios = allScenarios();
  } else {
    for (const auto & s : allScenarios()) {
      if (s.name == scenario_arg) {
        scenarios.push_back(s);
      }
    }
    if (scenarios.empty()) {
      std::fprintf(stderr, "Unknown scenario '%s'\n", scenario_arg.c_str());
      return 1;
    }
  }

  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(costmap_options);
  costmap_ros->configure();
  auto tf = std::make_shared<tf2_ros::Buffer>(costmap_ros->get_clock());
  tf->setUsingDedicatedThread(true);

  pluginlib::ClassLoader<nav2_core::Controller> loader(
    "nav2_core", "nav2_core::Controller");

  const bool motion_checks = scenario_arg == "all" || scenario_arg == "wall_gap";
  std::vector<Config> configs;
  if (path_angle_benchmark) {
    if (preset_arg != "full") {
      std::fprintf(stderr, "Preset '%s' is not supported for path_angle\n", preset_arg.c_str());
      return 1;
    }
    configs = pathAngleBenchmarkConfigs();
  } else if (curvature_speed_benchmark) {
    if (preset_arg != "full") {
      std::fprintf(
        stderr, "Preset '%s' is not supported for curvature_speed\n", preset_arg.c_str());
      return 1;
    }
    configs = curvatureSpeedBenchmarkConfigs();
  } else if (esdf_benchmark) {
    if (preset_arg != "full") {
      std::fprintf(stderr, "Preset '%s' is not supported for esdf\n", preset_arg.c_str());
      return 1;
    }
    configs = esdfBenchmarkConfigs();
  } else if (preset_arg == "full") {
    configs = benchmarkConfigs(motion_checks);
  } else if (preset_arg == "quick") {
    configs = quickBenchmarkConfigs();
  } else if (preset_arg == "cpu_gpu") {
    configs = cpuGpuBenchmarkConfigs();
  } else {
    std::fprintf(stderr, "Unknown preset '%s' (full | quick | cpu_gpu)\n", preset_arg.c_str());
    return 1;
  }

  for (const auto & scenario : scenarios) {
    runScenario(scenario, out_dir, configs, loader, tf);
  }

  rclcpp::shutdown();
  return 0;
}
