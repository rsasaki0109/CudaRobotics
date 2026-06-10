// Head-to-head closed-loop benchmark: nav2_mppi_controller (CPU) vs
// cuda_mppi_controller (GPU), both loaded through pluginlib exactly as
// nav2's controller_server loads them, driving the same unicycle plant
// through synthetic costmaps.
//
// Usage: controller_benchmark <out_dir> [scenario]
//   scenario: wall_gap | narrow_corridor | u_turn | all (default: wall_gap)
//   writes <out_dir>/summary.csv and <out_dir>/traj_<label>.csv
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
  std::function<nav_msgs::msg::Path(const rclcpp::Time &)> make_plan;
};

bool inRect(double x, double y, double x0, double x1, double y0, double y1)
{
  return x >= x0 && x < x1 && y >= y0 && y < y1;
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
        {1.5, 1.5}, {7.5, 1.5}, {7.5, 8.5}, {1.5, 8.5}}};
      for (size_t seg = 0; seg + 1 < pts.size(); ++seg) {
        const double x0 = pts[seg][0], y0 = pts[seg][1];
        const double x1 = pts[seg + 1][0], y1 = pts[seg + 1][1];
        const double len = std::hypot(x1 - x0, y1 - y0);
        const int steps = std::max(1, static_cast<int>(len / 0.05));
        for (int i = 0; i <= steps; ++i) {
          const double t = static_cast<double>(i) / steps;
          geometry_msgs::msg::PoseStamped p;
          p.header = path.header;
          p.pose.position.x = x0 + t * (x1 - x0);
          p.pose.position.y = y0 + t * (y1 - y0);
          p.pose.orientation.w = 1.0;
          path.poses.push_back(p);
        }
      }
      return path;
    };
  return s;
}

std::vector<Scenario> allScenarios()
{
  return {makeWallGap(), makeNarrowCorridor(), makeUTurn()};
}

void paintCostmap(nav2_costmap_2d::Costmap2D & costmap, const Scenario & scenario)
{
  const unsigned int nx = costmap.getSizeInCellsX();
  const unsigned int ny = costmap.getSizeInCellsY();
  std::vector<std::pair<double, double>> lethal_centers;
  for (unsigned int my = 0; my < ny; ++my) {
    for (unsigned int mx = 0; mx < nx; ++mx) {
      double wx, wy;
      costmap.mapToWorld(mx, my, wx, wy);
      if (scenario.lethal(wx, wy)) {
        costmap.setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
        lethal_centers.emplace_back(wx, wy);
      }
    }
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
  std::vector<std::array<double, 3>> traj;
};

RunResult runClosedLoop(
  nav2_core::Controller & controller,
  const rclcpp_lifecycle::LifecycleNode::SharedPtr & node,
  const Scenario & scenario)
{
  RunResult res;
  controller.setPlan(scenario.make_plan(node->now()));

  double x = scenario.start_x, y = scenario.start_y, yaw = 0.0;
  geometry_msgs::msg::Twist cmd;
  std::vector<double> solve_ms;
  solve_ms.reserve(kMaxSteps);

  for (res.steps = 0; res.steps < kMaxSteps; ++res.steps) {
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

    x += kControlDt * (
      cmd.linear.x * std::cos(yaw) - cmd.linear.y * std::sin(yaw));
    y += kControlDt * (
      cmd.linear.x * std::sin(yaw) + cmd.linear.y * std::cos(yaw));
    yaw = std::atan2(
      std::sin(yaw + kControlDt * cmd.angular.z),
      std::cos(yaw + kControlDt * cmd.angular.z));

    if (scenario.lethal(x, y)) {
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
  return res;
}

struct Config
{
  std::string label;
  std::string plugin;
  int batch_size;
  std::string motion_model = "DiffDrive";
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
  paintCostmap(*local_costmap->getCostmap(), scenario);

  const std::string scenario_dir = out_dir + "/" + scenario.name;
  std::filesystem::create_directories(scenario_dir);

  std::ofstream summary(scenario_dir + "/summary.csv");
  summary << "scenario,label,plugin,batch_size,motion_model,success,collided,steps,sim_s,"
    "mean_ms,p95_ms,max_ms,exceptions\n";

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
    const RunResult r = runClosedLoop(*controller, node, scenario);
    std::printf(
      "  %s steps=%d sim=%.1fs solve mean=%.2fms p95=%.2fms max=%.2fms exc=%d\n",
      r.success ? "SUCCESS" : (r.collided ? "COLLIDED" : "TIMEOUT"),
      r.steps, r.steps * kControlDt, r.mean_ms, r.p95_ms, r.max_ms, r.exceptions);

    summary << scenario.name << ',' << cfg.label << ',' << cfg.plugin << ','
            << cfg.batch_size << ',' << cfg.motion_model << ','
            << (r.success ? 1 : 0) << ',' << (r.collided ? 1 : 0) << ','
            << r.steps << ',' << r.steps * kControlDt << ','
            << r.mean_ms << ',' << r.p95_ms << ',' << r.max_ms << ','
            << r.exceptions << '\n';

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
  if (scenario_arg == "all") {
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

  pluginlib::ClassLoader<nav2_core::Controller> loader(
    "nav2_core", "nav2_core::Controller");

  const bool motion_checks = scenario_arg == "all" || scenario_arg == "wall_gap";
  const auto configs = benchmarkConfigs(motion_checks);

  for (const auto & scenario : scenarios) {
    runScenario(scenario, out_dir, configs, loader, tf);
  }

  rclcpp::shutdown();
  return 0;
}
