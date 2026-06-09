// Standalone check of the GPU MPPI optimizer without nav2: a synthetic
// costmap with a wall gap, a straight reference path, and a closed-loop
// simulation. Prints per-cycle solve time and exits non-zero if the goal
// is not reached or the wall is hit.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_mppi_controller/mppi_gpu.hpp"

namespace
{

constexpr int kSizeX = 200;       // 10 m x 10 m @ 0.05 m
constexpr int kSizeY = 200;
constexpr float kRes = 0.05f;
constexpr float kOrigin = 0.0f;

void paintWallWithGap(std::vector<unsigned char> & map)
{
  // vertical lethal wall at x = 5 m with a gap y in [4.0, 6.0]
  const int wx0 = static_cast<int>(4.9f / kRes);
  const int wx1 = static_cast<int>(5.1f / kRes);
  const int gy0 = static_cast<int>(4.0f / kRes);
  const int gy1 = static_cast<int>(6.0f / kRes);
  for (int my = 0; my < kSizeY; ++my) {
    if (my >= gy0 && my < gy1) {
      continue;
    }
    for (int mx = wx0; mx < wx1; ++mx) {
      map[my * kSizeX + mx] = 254;  // LETHAL
    }
  }
}

bool isLethal(const std::vector<unsigned char> & map, float x, float y)
{
  const int mx = static_cast<int>((x - kOrigin) / kRes);
  const int my = static_cast<int>((y - kOrigin) / kRes);
  if (mx < 0 || mx >= kSizeX || my < 0 || my >= kSizeY) {
    return false;
  }
  return map[my * kSizeX + mx] >= 253;
}

}  // namespace

int main(int argc, char ** argv)
{
  cuda_mppi_controller::MppiParams params;
  if (argc > 1) {
    params.batch_size = std::atoi(argv[1]);  // K sweep for benchmarking
  }
  cuda_mppi_controller::MppiGpu mppi(params);

  std::vector<unsigned char> map(kSizeX * kSizeY, 0);
  paintWallWithGap(map);

  // reference path: (1,5) -> (9,5), straight through the gap
  std::vector<float> path_xy;
  const float goal_x = 9.0f, goal_y = 5.0f, goal_yaw = 0.0f;
  for (float x = 1.0f; x <= goal_x; x += 0.1f) {
    path_xy.push_back(x);
    path_xy.push_back(goal_y);
  }
  const int n_path = static_cast<int>(path_xy.size() / 2);
  const float lookahead = 3.0f;  // [m] same windowing as the ROS controller

  float x = 1.0f, y = 5.0f, yaw = 0.0f;
  double total_ms = 0.0, max_ms = 0.0;
  int steps = 0;
  const int max_steps = 1200;

  for (; steps < max_steps; ++steps) {
    // local path window around the robot, like the ROS controller does
    int nearest = 0;
    float nearest_d2 = 1.0e18f;
    for (int i = 0; i < n_path; ++i) {
      const float dx = x - path_xy[i * 2 + 0];
      const float dy = y - path_xy[i * 2 + 1];
      const float d2 = dx * dx + dy * dy;
      if (d2 < nearest_d2) {
        nearest_d2 = d2;
        nearest = i;
      }
    }
    int win_end = nearest;
    float arc = 0.0f;
    for (int i = nearest + 1; i < n_path; ++i) {
      arc += std::hypot(
        path_xy[i * 2 + 0] - path_xy[(i - 1) * 2 + 0],
        path_xy[i * 2 + 1] - path_xy[(i - 1) * 2 + 1]);
      win_end = i;
      if (arc > lookahead) {
        break;
      }
    }
    const bool goal_is_final = (win_end + 1 == n_path);
    const float local_gx = path_xy[win_end * 2 + 0];
    const float local_gy = path_xy[win_end * 2 + 1];

    const auto t0 = std::chrono::steady_clock::now();
    const auto res = mppi.compute(
      x, y, yaw,
      map.data(), kSizeX, kSizeY, kOrigin, kOrigin, kRes,
      path_xy.data() + nearest * 2, win_end - nearest + 1,
      local_gx, local_gy, goal_yaw, goal_is_final);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_ms += ms;
    max_ms = std::max(max_ms, ms);

    if (res.all_colliding) {
      std::printf("FAIL: all sampled trajectories colliding at step %d\n", steps);
      return 1;
    }

    // apply first control to the plant (same unicycle model)
    x += params.model_dt * res.v * std::cos(yaw);
    y += params.model_dt * res.v * std::sin(yaw);
    yaw = std::atan2(
      std::sin(yaw + params.model_dt * res.w),
      std::cos(yaw + params.model_dt * res.w));

    if (isLethal(map, x, y)) {
      std::printf("FAIL: robot hit the wall at step %d (x=%.2f y=%.2f)\n", steps, x, y);
      return 1;
    }
    const float dx = x - goal_x, dy = y - goal_y;
    if (dx * dx + dy * dy < 0.25f * 0.25f) {
      break;
    }
  }

  if (steps >= max_steps) {
    std::printf(
      "FAIL: goal not reached in %d steps (x=%.2f y=%.2f)\n", max_steps, x, y);
    return 1;
  }

  std::printf(
    "PASS: goal reached in %d steps (%.1f sim-seconds)\n",
    steps, steps * params.model_dt);
  std::printf(
    "solve time: mean %.2f ms, max %.2f ms (K=%d, T=%d, incl. costmap upload)\n",
    total_ms / (steps + 1), max_ms, params.batch_size, params.time_steps);
  return 0;
}
