#ifndef CUDA_MPPI_CONTROLLER__DYNAMIC_OBSTACLE_INPUT_HPP_
#define CUDA_MPPI_CONTROLLER__DYNAMIC_OBSTACLE_INPUT_HPP_

// Conversion of tracked objects (from a TrackedObstacleArray) into the MPPI
// core's DynamicObstacle rows. Kept free of ROS types so it can be tested
// without a ROS runtime or a GPU.

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "cuda_mppi_controller/mppi_gpu.hpp"

namespace cuda_mppi_controller
{

// Rigid 2-D transform from the tracker frame into the costmap frame:
// p_costmap = R(yaw) * p_tracker + (x, y).
struct PlanarTransform
{
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
};

// One tracked disc in the tracker frame.
struct TrackedDisc
{
  double x = 0.0;
  double y = 0.0;
  double vx = 0.0;
  double vy = 0.0;
  double radius = 0.0;
};

// Transform the discs into the costmap frame, move each along its velocity by
// `extrapolate_s` seconds (the message age), drop non-finite or negative-radius
// entries, and keep the `max_count` discs closest to the robot.
inline std::vector<DynamicObstacle> toDynamicObstacles(
  const std::vector<TrackedDisc> & discs, const PlanarTransform & transform,
  double extrapolate_s, double robot_x, double robot_y,
  int max_count = kMaxDynamicObstacles)
{
  const double c = std::cos(transform.yaw);
  const double s = std::sin(transform.yaw);
  const double dt = std::isfinite(extrapolate_s) ? std::max(0.0, extrapolate_s) : 0.0;
  std::vector<std::pair<double, DynamicObstacle>> ranked;
  ranked.reserve(discs.size());
  for (const TrackedDisc & d : discs) {
    if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.vx) ||
      !std::isfinite(d.vy) || !std::isfinite(d.radius) || d.radius < 0.0)
    {
      continue;
    }
    const double vx = c * d.vx - s * d.vy;
    const double vy = s * d.vx + c * d.vy;
    const double x = c * d.x - s * d.y + transform.x + vx * dt;
    const double y = s * d.x + c * d.y + transform.y + vy * dt;
    DynamicObstacle o{};
    o.x = static_cast<float>(x);
    o.y = static_cast<float>(y);
    o.vx = static_cast<float>(vx);
    o.vy = static_cast<float>(vy);
    o.radius = static_cast<float>(d.radius);
    ranked.emplace_back(std::hypot(x - robot_x, y - robot_y) - d.radius, o);
  }
  const std::size_t keep = static_cast<std::size_t>(std::max(0, max_count));
  if (ranked.size() > keep) {
    std::partial_sort(
      ranked.begin(), ranked.begin() + keep, ranked.end(),
      [](const auto & a, const auto & b) {return a.first < b.first;});
    ranked.resize(keep);
  }
  std::vector<DynamicObstacle> out;
  out.reserve(ranked.size());
  for (const auto & entry : ranked) {
    out.push_back(entry.second);
  }
  return out;
}

}  // namespace cuda_mppi_controller

#endif  // CUDA_MPPI_CONTROLLER__DYNAMIC_OBSTACLE_INPUT_HPP_
