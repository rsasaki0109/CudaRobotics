// Head-to-head closed-loop benchmark: nav2_mppi_controller (CPU) vs
// cuda_mppi_controller (GPU), both loaded through pluginlib exactly as
// nav2's controller_server loads them, driving the same unicycle plant
// through the same synthetic costmap (wall with a gap, inflated).
//
// Usage: controller_benchmark <out_dir>
//   writes <out_dir>/summary.csv and <out_dir>/traj_<label>.csv
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
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

// scenario: 10 m x 10 m, vertical lethal wall at x = 5 with a gap y in [4, 6]
constexpr double kWallX0 = 4.9, kWallX1 = 5.1;
constexpr double kGapY0 = 4.0, kGapY1 = 6.0;
constexpr double kStartX = 1.0, kStartY = 5.0;
constexpr double kGoalX = 9.0, kGoalY = 5.0;
constexpr double kGoalTol = 0.25;
constexpr double kControlDt = 0.05;   // 20 Hz
constexpr int kMaxSteps = 1200;
constexpr double kInscribedRadius = 0.2;
constexpr double kCostScaling = 3.0;

bool insideWall(double x, double y)
{
  return x >= kWallX0 && x < kWallX1 && !(y >= kGapY0 && y < kGapY1);
}

// Paint the wall as lethal cells plus an exponential inflation ring, the
// same shape nav2's inflation layer would produce for a 0.2 m radius robot.
void paintCostmap(nav2_costmap_2d::Costmap2D & costmap)
{
  const unsigned int nx = costmap.getSizeInCellsX();
  const unsigned int ny = costmap.getSizeInCellsY();
  std::vector<std::pair<double, double>> lethal_centers;
  for (unsigned int my = 0; my < ny; ++my) {
    for (unsigned int mx = 0; mx < nx; ++mx) {
      double wx, wy;
      costmap.mapToWorld(mx, my, wx, wy);
      if (insideWall(wx, wy)) {
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
      // distance to the wall rectangle minus the gap: brute force over cells
      // is overkill, use the analytic box distance and handle the gap edges
      double d = 1.0e9;
      for (const auto & c : lethal_centers) {
        const double dd = std::hypot(wx - c.first, wy - c.second);
        d = std::min(d, dd);
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

nav_msgs::msg::Path makePlan(const rclcpp::Time & stamp)
{
  nav_msgs::msg::Path path;
  path.header.frame_id = "odom";
  path.header.stamp = stamp;
  for (double x = kStartX; x <= kGoalX + 1.0e-6; x += 0.05) {
    geometry_msgs::msg::PoseStamped p;
    p.header = path.header;
    p.pose.position.x = x;
    p.pose.position.y = kGoalY;
    p.pose.orientation.w = 1.0;
    path.poses.push_back(p);
  }
  return path;
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
  std::vector<std::array<double, 3>> traj;  // x, y, yaw
};

RunResult runClosedLoop(
  nav2_core::Controller & controller,
  const rclcpp_lifecycle::LifecycleNode::SharedPtr & node)
{
  RunResult res;
  controller.setPlan(makePlan(node->now()));

  double x = kStartX, y = kStartY, yaw = 0.0;
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
    } catch (const std::exception & e) {
      ++res.exceptions;
      cmd = geometry_msgs::msg::Twist();
    }
    const auto t1 = std::chrono::steady_clock::now();
    solve_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

    x += kControlDt * cmd.linear.x * std::cos(yaw);
    y += kControlDt * cmd.linear.x * std::sin(yaw);
    yaw = std::atan2(
      std::sin(yaw + kControlDt * cmd.angular.z),
      std::cos(yaw + kControlDt * cmd.angular.z));

    if (insideWall(x, y)) {
      res.collided = true;
      break;
    }
    if (std::hypot(x - kGoalX, y - kGoalY) < kGoalTol) {
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
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  const std::string out_dir = argc > 1 ? argv[1] : ".";
  std::filesystem::create_directories(out_dir);

  // shared synthetic costmap, configured but never activated (no sensors)
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
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(costmap_options);
  costmap_ros->configure();
  paintCostmap(*costmap_ros->getCostmap());

  auto tf = std::make_shared<tf2_ros::Buffer>(costmap_ros->get_clock());

  const std::vector<Config> configs = {
    {"cpu_mppi_K1000", "nav2_mppi_controller::MPPIController", 1000},
    {"cpu_mppi_K2000", "nav2_mppi_controller::MPPIController", 2000},
    {"cpu_mppi_K5000", "nav2_mppi_controller::MPPIController", 5000},
    {"cpu_mppi_K10000", "nav2_mppi_controller::MPPIController", 10000},
    {"gpu_mppi_K2048", "cuda_mppi_controller::CudaMppiController", 2048},
    {"gpu_mppi_K8192", "cuda_mppi_controller::CudaMppiController", 8192},
    {"gpu_mppi_K16384", "cuda_mppi_controller::CudaMppiController", 16384},
    {"gpu_mppi_K65536", "cuda_mppi_controller::CudaMppiController", 65536},
  };

  pluginlib::ClassLoader<nav2_core::Controller> loader(
    "nav2_core", "nav2_core::Controller");

  std::ofstream summary(out_dir + "/summary.csv");
  summary << "label,plugin,batch_size,success,collided,steps,sim_s,"
    "mean_ms,p95_ms,max_ms,exceptions\n";

  for (const auto & cfg : configs) {
    // fresh node per config so parameter overrides apply at declare time
    rclcpp::NodeOptions options;
    std::vector<rclcpp::Parameter> params = {
      // controller_server-level param the CPU MPPI validates against model_dt
      rclcpp::Parameter("controller_frequency", 20.0),
      rclcpp::Parameter("FollowPath.batch_size", cfg.batch_size),
      rclcpp::Parameter("FollowPath.time_steps", 56),
      rclcpp::Parameter("FollowPath.model_dt", kControlDt),
      rclcpp::Parameter("FollowPath.iteration_count", 1),
      // shared diff-drive limits (nav2 defaults)
      rclcpp::Parameter("FollowPath.vx_max", 0.5),
      rclcpp::Parameter("FollowPath.vx_min", -0.35),
      rclcpp::Parameter("FollowPath.wz_max", 1.9),
      rclcpp::Parameter("FollowPath.v_max", 0.5),
      rclcpp::Parameter("FollowPath.v_min", -0.35),
      rclcpp::Parameter("FollowPath.w_max", 1.9),
      rclcpp::Parameter("FollowPath.motion_model", std::string("DiffDrive")),
      rclcpp::Parameter("FollowPath.visualize", false),
      // stock nav2_bringup critic set for the CPU MPPI baseline
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
    controller->configure(node, "FollowPath", tf, costmap_ros);
    controller->activate();

    std::printf("=== %s (%s) ===\n", cfg.label.c_str(), cfg.plugin.c_str());
    const RunResult r = runClosedLoop(*controller, node);
    std::printf(
      "  %s steps=%d sim=%.1fs solve mean=%.2fms p95=%.2fms max=%.2fms exc=%d\n",
      r.success ? "SUCCESS" : (r.collided ? "COLLIDED" : "TIMEOUT"),
      r.steps, r.steps * kControlDt, r.mean_ms, r.p95_ms, r.max_ms, r.exceptions);

    summary << cfg.label << ',' << cfg.plugin << ',' << cfg.batch_size << ','
            << (r.success ? 1 : 0) << ',' << (r.collided ? 1 : 0) << ','
            << r.steps << ',' << r.steps * kControlDt << ','
            << r.mean_ms << ',' << r.p95_ms << ',' << r.max_ms << ','
            << r.exceptions << '\n';

    std::ofstream traj(out_dir + "/traj_" + cfg.label + ".csv");
    traj << "x,y,yaw\n";
    for (const auto & p : r.traj) {
      traj << p[0] << ',' << p[1] << ',' << p[2] << '\n';
    }

    controller->deactivate();
    controller->cleanup();
  }

  rclcpp::shutdown();
  return 0;
}
