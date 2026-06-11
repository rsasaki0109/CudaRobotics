// Standalone check of the GPU MPPI optimizer without nav2: a synthetic
// costmap with a wall gap, a straight reference path, and a closed-loop
// simulation. Prints per-cycle solve time and exits non-zero if the goal
// is not reached or the wall is hit.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
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
  // exponential inflation ring (needed by the footprint-mode cost gate)
  const float inscribed = 0.2f, scaling = 3.0f;
  auto aabb_dist = [](float x, float y, float x0, float y0, float x1, float y1) {
      const float ddx = std::max({x0 - x, 0.0f, x - x1});
      const float ddy = std::max({y0 - y, 0.0f, y - y1});
      return std::hypot(ddx, ddy);
    };
  for (int my = 0; my < kSizeY; ++my) {
    for (int mx = 0; mx < kSizeX; ++mx) {
      if (map[my * kSizeX + mx] == 254) {
        continue;
      }
      const float x = (mx + 0.5f) * kRes;
      const float y = (my + 0.5f) * kRes;
      const float dist = std::min(
        aabb_dist(x, y, 4.9f, 0.0f, 5.1f, 4.0f),     // wall below the gap
        aabb_dist(x, y, 4.9f, 6.0f, 5.1f, 10.0f));   // wall above the gap
      if (dist <= inscribed) {
        map[my * kSizeX + mx] = 253;  // INSCRIBED
      } else if (dist < 1.0f) {
        const float c = 252.0f * std::exp(-scaling * (dist - inscribed));
        if (c >= 1.0f) {
          map[my * kSizeX + mx] = static_cast<unsigned char>(c);
        }
      }
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
  // usage: mppi_gpu_standalone [K] [diff|ackermann|omni|footprint|esdf]
  cuda_mppi_controller::MppiParams params;
  if (argc > 1) {
    params.batch_size = std::atoi(argv[1]);  // K sweep for benchmarking
  }
  const std::string mode = argc > 2 ? argv[2] : "diff";
  std::vector<float> footprint;
  if (mode == "ackermann") {
    params.motion_model = cuda_mppi_controller::MotionModel::Ackermann;
    params.min_turning_r = 0.5f;
  } else if (mode == "omni") {
    params.motion_model = cuda_mppi_controller::MotionModel::Omni;
  } else if (mode == "footprint") {
    params.consider_footprint = true;
    footprint = {0.15f, 0.15f, -0.15f, 0.15f, -0.15f, -0.15f, 0.15f, -0.15f};
  } else if (mode == "esdf") {
    params.distance_field_weight = 12.0f;
    params.distance_field_cutoff = 0.8f;
  } else if (mode != "diff") {
    std::printf("unknown mode '%s'\n", mode.c_str());
    return 2;
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
  double min_valid_ratio = 1.0;
  double dist = 0.0;
  int wall_cross_step = -1, near_goal_step = -1;
  int retreat_count = 0;
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
      local_gx, local_gy, goal_yaw, goal_is_final,
      footprint.data(), static_cast<int>(footprint.size() / 2));
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_ms += ms;
    max_ms = std::max(max_ms, ms);
    min_valid_ratio = std::min(min_valid_ratio, static_cast<double>(res.valid_rollout_ratio));
    if (res.retreating) {
      ++retreat_count;
    }

    if (res.all_colliding) {
      std::printf("FAIL: all sampled trajectories colliding at step %d\n", steps);
      return 1;
    }

    if (std::getenv("MPPI_TRACE") && steps % 20 == 0) {
      std::fprintf(
        stderr,
        "t=%5.2f x=%.2f y=%.2f yaw=%6.2f v=%5.2f w=%6.2f valid=%d/%d best=%.3f\n",
        steps * params.model_dt, x, y, yaw, res.v, res.w,
        res.valid_rollouts, res.sampled_rollouts, res.best_cost);
    }
    // apply first control to the plant (same model as the rollouts)
    const float px = x, py = y;
    x += params.model_dt * (res.v * std::cos(yaw) - res.vy * std::sin(yaw));
    y += params.model_dt * (res.v * std::sin(yaw) + res.vy * std::cos(yaw));
    dist += std::hypot(x - px, y - py);
    if (wall_cross_step < 0 && x > 5.1f) {
      wall_cross_step = steps;
    }
    if (near_goal_step < 0 && std::hypot(x - goal_x, y - goal_y) < 1.0f) {
      near_goal_step = steps;
    }
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
    "PASS [%s]: goal reached in %d steps (%.1f sim-seconds)\n",
    mode.c_str(), steps, steps * params.model_dt);
  std::printf(
    "profile: mean speed %.3f m/s | wall crossed at %.1fs | last 1 m took %.1fs\n",
    dist / (steps * params.model_dt), wall_cross_step * params.model_dt,
    (steps - near_goal_step) * params.model_dt);
  std::printf(
    "solve time: mean %.2f ms, max %.2f ms (K=%d, T=%d, incl. costmap upload)\n",
    total_ms / (steps + 1), max_ms, params.batch_size, params.time_steps);
  std::printf(
    "diagnostics: min valid rollout ratio %.1f%% | retreat cycles %d\n",
    100.0 * min_valid_ratio, retreat_count);
  return 0;
}
