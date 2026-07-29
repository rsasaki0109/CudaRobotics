#include "cuda_mppi_controller/mppi_gpu.hpp"
#include "cudarobotics/esdf_2d_gpu.hpp"
#include "cudarobotics/kiss_icp_gpu.hpp"
#include "cudarobotics/voxel_mapping_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kDt = 0.10f;
constexpr float kRobotRadius = 0.24f;

struct Segment { float x0, y0, x1, y1; };
struct Pose { float x = 0.0f, y = 0.0f, yaw = 0.0f; };

std::vector<Segment> rectangle(float min_x, float min_y, float max_x, float max_y) {
  return {
    {min_x, min_y, max_x, min_y}, {max_x, min_y, max_x, max_y},
    {max_x, max_y, min_x, max_y}, {min_x, max_y, min_x, min_y}};
}

std::vector<Segment> course_segments() {
  std::vector<Segment> result = {
    {-1.0f, -2.5f, 10.0f, -2.5f}, {10.0f, -2.5f, 10.0f, 2.5f},
    {10.0f, 2.5f, -1.0f, 2.5f}, {-1.0f, 2.5f, -1.0f, -2.5f}};
  const auto first = rectangle(3.6f, -2.5f, 4.3f, 0.55f);
  const auto second = rectangle(6.1f, -0.55f, 6.8f, 2.5f);
  result.insert(result.end(), first.begin(), first.end());
  result.insert(result.end(), second.begin(), second.end());
  return result;
}

const std::vector<std::pair<float, float>>& mission_waypoints() {
  static const std::vector<std::pair<float, float>> value = {
    {0.0f, 0.0f}, {2.7f, 0.0f}, {3.15f, 1.15f}, {4.75f, 1.15f},
    {5.55f, -1.15f}, {7.25f, -1.15f}, {9.0f, 0.0f}};
  return value;
}

float wrap(float angle) { return std::atan2(std::sin(angle), std::cos(angle)); }

float raycast(
  float origin_x, float origin_y, float angle,
  const std::vector<Segment>& segments, float max_range)
{
  const float dx = std::cos(angle);
  const float dy = std::sin(angle);
  float best = max_range;
  for (const auto& segment : segments) {
    const float ex = segment.x1 - segment.x0;
    const float ey = segment.y1 - segment.y0;
    const float denominator = dx * ey - dy * ex;
    if (std::fabs(denominator) < 1.0e-12f) continue;
    const float ax = segment.x0 - origin_x;
    const float ay = segment.y0 - origin_y;
    const float ray_distance = (ax * ey - ay * ex) / denominator;
    const float segment_fraction = (ax * dy - ay * dx) / denominator;
    if (ray_distance >= 0.0f && segment_fraction >= 0.0f &&
        segment_fraction <= 1.0f && ray_distance < best) {
      best = ray_distance;
    }
  }
  return best;
}

float point_segment_distance(float x, float y, const Segment& segment) {
  const float ex = segment.x1 - segment.x0;
  const float ey = segment.y1 - segment.y0;
  const float length_squared = ex * ex + ey * ey;
  if (length_squared <= 1.0e-18f) return std::hypot(x - segment.x0, y - segment.y0);
  const float fraction = std::max(0.0f, std::min(
    1.0f,
    ((x - segment.x0) * ex + (y - segment.y0) * ey) / length_squared));
  return std::hypot(
    x - (segment.x0 + fraction * ex),
    y - (segment.y0 + fraction * ey));
}

bool collides(float x, float y, const std::vector<Segment>& segments) {
  return std::any_of(segments.begin(), segments.end(), [=](const Segment& segment) {
    return point_segment_distance(x, y, segment) <= kRobotRadius;
  });
}

std::vector<float> scan(const Pose& truth, const std::vector<Segment>& segments) {
  std::vector<float> xyz;
  xyz.reserve(240u * 3u * 3u);
  constexpr float z_levels[] = {-0.45f, 0.0f, 0.45f};
  for (int index = 0; index < 240; ++index) {
    const float local_angle = -kPi + 2.0f * kPi * index / 240.0f;
    const float distance = raycast(
      truth.x, truth.y, truth.yaw + local_angle, segments, 12.0f);
    if (distance >= 12.0f) continue;
    for (const float z : z_levels) {
      xyz.push_back(distance * std::cos(local_angle));
      xyz.push_back(distance * std::sin(local_angle));
      xyz.push_back(z);
    }
  }
  return xyz;
}

std::vector<float> interpolate_path(float spacing) {
  std::vector<float> path;
  const auto& waypoints = mission_waypoints();
  for (std::size_t index = 0; index + 1 < waypoints.size(); ++index) {
    const float dx = waypoints[index + 1].first - waypoints[index].first;
    const float dy = waypoints[index + 1].second - waypoints[index].second;
    const float length = std::hypot(dx, dy);
    const int samples = std::max(1, static_cast<int>(std::ceil(length / spacing)));
    for (int sample = 0; sample < samples; ++sample) {
      if (index > 0 && sample == 0) continue;
      const float fraction = static_cast<float>(sample) / samples;
      path.push_back(waypoints[index].first + fraction * dx);
      path.push_back(waypoints[index].second + fraction * dy);
    }
  }
  path.push_back(waypoints.back().first);
  path.push_back(waypoints.back().second);
  return path;
}

Pose estimated_pose(const cudarobotics::KissIcpPose& pose) {
  return {pose.t[0], pose.t[1], std::atan2(pose.R.m[3], pose.R.m[0])};
}

std::vector<float> transform_scan(
  const std::vector<float>& points, const cudarobotics::KissIcpPose& pose)
{
  std::vector<float> world(points.size());
  for (std::size_t index = 0; index < points.size(); index += 3) {
    const float x = points[index];
    const float y = points[index + 1];
    const float z = points[index + 2];
    world[index] = pose.R.m[0] * x + pose.R.m[1] * y + pose.R.m[2] * z + pose.t[0];
    world[index + 1] =
      pose.R.m[3] * x + pose.R.m[4] * y + pose.R.m[5] * z + pose.t[1];
    world[index + 2] =
      pose.R.m[6] * x + pose.R.m[7] * y + pose.R.m[8] * z + pose.t[2];
  }
  return world;
}

void clear_footprint(
  cudarobotics::OccupancyProjection& occupancy, float x, float y, float radius)
{
  for (int row = 0; row < occupancy.grid.height; ++row) {
    for (int column = 0; column < occupancy.grid.width; ++column) {
      const float cell_x =
        occupancy.grid.origin_x + (column + 0.5f) * occupancy.grid.resolution;
      const float cell_y =
        occupancy.grid.origin_y + (row + 0.5f) * occupancy.grid.resolution;
      if (std::hypot(cell_x - x, cell_y - y) <= radius) {
        occupancy.data[static_cast<std::size_t>(row) * occupancy.grid.width + column] = 0;
      }
    }
  }
}

std::vector<unsigned char> make_costmap(
  const cudarobotics::OccupancyProjection& occupancy,
  const cudarobotics::Esdf2DResult& distances)
{
  std::vector<unsigned char> result(occupancy.data.size(), 0);
  for (std::size_t index = 0; index < result.size(); ++index) {
    if (occupancy.data[index] < 0) {
      result[index] = 255;
    } else if (occupancy.data[index] >= 50) {
      result[index] = 254;
    } else if (distances.distances[index] <= 0.25f) {
      result[index] = 253;
    } else if (distances.distances[index] < 1.0f) {
      const float value =
        252.0f * std::exp(-3.0f * (distances.distances[index] - 0.25f));
      result[index] = static_cast<unsigned char>(
        std::max(1.0f, std::min(252.0f, value)));
    }
  }
  return result;
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t index = std::min(
    values.size() - 1,
    static_cast<std::size_t>(std::ceil(fraction * values.size()) - 1.0));
  return values[index];
}

struct Options {
  std::string json;
  std::string csv;
  int maximum_steps = 1800;
  bool check = false;
};

Options options(int argc, char** argv) {
  Options result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() {
      if (++index >= argc) throw std::invalid_argument("missing option value");
      return std::string(argv[index]);
    };
    if (argument == "--json") result.json = value();
    else if (argument == "--csv") result.csv = value();
    else if (argument == "--maximum-steps") result.maximum_steps = std::stoi(value());
    else if (argument == "--check") result.check = true;
    else throw std::invalid_argument("unknown option: " + argument);
  }
  if (result.json.empty() || result.csv.empty() || result.maximum_steps < 10) {
    throw std::invalid_argument("--json, --csv, and a valid --maximum-steps are required");
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options run = options(argc, argv);
    const auto segments = course_segments();
    const auto path = interpolate_path(0.12f);

    int device = 0;
    int driver_version = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
        cudaDriverGetVersion(&driver_version) != cudaSuccess) {
      throw std::runtime_error("failed to query CUDA device");
    }

    cudarobotics::KissIcpConfig kiss_config;
    kiss_config.max_scan_points = 4096;
    kiss_config.max_map_points = 100000;
    kiss_config.hash_capacity = 1u << 18;
    cudarobotics::KissIcpOdometry odometry(kiss_config);

    cudarobotics::VoxelMappingConfig voxel_config;
    voxel_config.width = 256;
    voxel_config.height = 256;
    voxel_config.depth = 32;
    voxel_config.resolution = 0.10f;
    voxel_config.origin_z = -2.0f;
    voxel_config.max_range = 12.0f;
    voxel_config.projection_min_z = -0.5f;
    voxel_config.projection_max_z = 2.0f;
    voxel_config.rolling_margin_cells = 48;
    cudarobotics::VoxelMapperGpu mapper(voxel_config);
    mapper.reset(0.0f, 0.0f);

    cudarobotics::Esdf2DConfig esdf_config;
    esdf_config.max_width = voxel_config.width;
    esdf_config.max_height = voxel_config.height;
    esdf_config.unknown_policy = cudarobotics::UnknownSpacePolicy::Free;
    cudarobotics::Esdf2DGpu esdf(esdf_config);

    cuda_mppi_controller::MppiParams mppi_params;
    mppi_params.batch_size = 2048;
    mppi_params.time_steps = 56;
    mppi_params.model_dt = 0.05f;
    mppi_params.v_max = 0.55f;
    mppi_params.v_min = 0.0f;
    mppi_params.w_max = 1.9f;
    mppi_params.costmap_weight = 5.0f;
    mppi_params.path_weight = 14.0f;
    mppi_params.path_follow_weight = 7.0f;
    mppi_params.path_angle_weight = 0.5f;
    mppi_params.distance_field_weight = 0.0f;
    cuda_mppi_controller::MppiGpu mppi(mppi_params);

    std::ofstream csv(run.csv);
    if (!csv) throw std::runtime_error("cannot open CSV");
    csv << "step,time_s,truth_x,truth_y,truth_yaw,estimated_x,estimated_y,"
           "estimated_yaw,error_m,inliers,observed_voxels,occupied_cells,"
           "valid_rollout_ratio,all_colliding,retreating,command_v,command_w,"
           "solve_ms,frame_ms\n";

    Pose truth;
    Pose previous_truth = truth;
    Pose previous_estimate;
    float command_v = 0.0f;
    float command_w = 0.0f;
    double truth_distance = 0.0;
    double estimate_distance = 0.0;
    double command_effect_distance = 0.0;
    double squared_error = 0.0;
    int frames = 0;
    int collisions = 0;
    int invalid_commands = 0;
    int all_colliding = 0;
    int retreating = 0;
    int minimum_inliers = std::numeric_limits<int>::max();
    std::size_t final_observed_voxels = 0;
    std::size_t maximum_occupied_cells = 0;
    float minimum_nonzero_valid_ratio = 1.0f;
    std::vector<double> frame_ms;
    std::vector<double> solve_ms;
    bool goal_reached = false;

    const auto wall_start = std::chrono::steady_clock::now();
    for (int step = 0; step < run.maximum_steps; ++step) {
      const auto frame_start = std::chrono::steady_clock::now();
      if (step > 0) {
        const float middle_yaw = truth.yaw + 0.5f * command_w * kDt;
        const Pose candidate{
          truth.x + command_v * std::cos(middle_yaw) * kDt,
          truth.y + command_v * std::sin(middle_yaw) * kDt,
          wrap(truth.yaw + command_w * kDt)};
        if (collides(candidate.x, candidate.y, segments)) {
          ++collisions;
          command_v = command_w = 0.0f;
        } else {
          truth = candidate;
        }
        const double truth_step = std::hypot(
          truth.x - previous_truth.x, truth.y - previous_truth.y);
        truth_distance += truth_step;
        if (std::fabs(command_v) > 1.0e-3f) command_effect_distance += truth_step;
        previous_truth = truth;
      }

      const auto points = scan(truth, segments);
      const auto odometry_result = odometry.register_scan(points);
      const Pose estimate = estimated_pose(odometry_result.pose);
      if (step > 0) {
        minimum_inliers = std::min(
          minimum_inliers, static_cast<int>(odometry_result.alignment.inliers));
        estimate_distance += std::hypot(
          estimate.x - previous_estimate.x, estimate.y - previous_estimate.y);
      }
      previous_estimate = estimate;
      const double error = std::hypot(estimate.x - truth.x, estimate.y - truth.y);
      squared_error += error * error;

      const auto world = transform_scan(points, odometry_result.pose);
      const float sensor_origin[] = {
        odometry_result.pose.t[0], odometry_result.pose.t[1], odometry_result.pose.t[2]};
      const auto mapping = mapper.integrate_scan(world, sensor_origin);
      final_observed_voxels = mapping.observed_voxels;
      auto occupancy = mapper.occupancy_projection();
      clear_footprint(occupancy, estimate.x, estimate.y, 0.30f);
      const std::size_t occupied = static_cast<std::size_t>(std::count(
        occupancy.data.begin(), occupancy.data.end(), static_cast<std::int8_t>(100)));
      maximum_occupied_cells = std::max(maximum_occupied_cells, occupied);
      const auto distance_field = esdf.compute(
        occupancy.data, occupancy.grid.width, occupancy.grid.height,
        occupancy.grid.resolution, 2.0f);
      const auto costmap = make_costmap(occupancy, distance_field);

      const auto solve_start = std::chrono::steady_clock::now();
      const auto goal = mission_waypoints().back();
      const float goal_yaw = std::atan2(
        goal.second - mission_waypoints()[mission_waypoints().size() - 2].second,
        goal.first - mission_waypoints()[mission_waypoints().size() - 2].first);
      const auto result = mppi.compute(
        estimate.x, estimate.y, estimate.yaw,
        costmap.data(), occupancy.grid.width, occupancy.grid.height,
        occupancy.grid.origin_x, occupancy.grid.origin_y, occupancy.grid.resolution,
        path.data(), static_cast<int>(path.size() / 2),
        goal.first, goal.second, goal_yaw, true);
      const double solve = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - solve_start).count();
      solve_ms.push_back(solve);
      command_v = result.v;
      command_w = result.w;
      all_colliding += result.all_colliding ? 1 : 0;
      retreating += result.retreating ? 1 : 0;
      if (result.valid_rollout_ratio > 0.0f) {
        minimum_nonzero_valid_ratio =
          std::min(minimum_nonzero_valid_ratio, result.valid_rollout_ratio);
      }
      if (!std::isfinite(command_v) || !std::isfinite(command_w) ||
          std::fabs(command_v) > mppi_params.v_max + 1.0e-5f ||
          std::fabs(command_w) > mppi_params.w_max + 1.0e-5f) {
        ++invalid_commands;
        command_v = command_w = 0.0f;
      }

      const double elapsed_frame = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - frame_start).count();
      frame_ms.push_back(elapsed_frame);
      csv << step << ',' << step * kDt << ',' << truth.x << ',' << truth.y << ','
          << truth.yaw << ',' << estimate.x << ',' << estimate.y << ','
          << estimate.yaw << ',' << error << ','
          << odometry_result.alignment.inliers << ',' << mapping.observed_voxels << ','
          << occupied << ',' << result.valid_rollout_ratio << ','
          << (result.all_colliding ? 1 : 0) << ','
          << (result.retreating ? 1 : 0) << ',' << command_v << ',' << command_w
          << ',' << solve << ',' << elapsed_frame << '\n';
      ++frames;

      if (std::hypot(truth.x - goal.first, truth.y - goal.second) <= 0.30f) {
        goal_reached = true;
        break;
      }
    }
    csv.close();

    const auto goal = mission_waypoints().back();
    const double goal_distance = std::hypot(truth.x - goal.first, truth.y - goal.second);
    const double final_error = std::hypot(
      previous_estimate.x - truth.x, previous_estimate.y - truth.y);
    const double drift_percent =
      truth_distance > 1.0e-6 ? 100.0 * final_error / truth_distance : 0.0;
    const double ate = frames > 0 ? std::sqrt(squared_error / frames) : 0.0;
    const double wall_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - wall_start).count();
    const double deadline_miss_rate = frame_ms.empty() ? 1.0 :
      static_cast<double>(std::count_if(
        frame_ms.begin(), frame_ms.end(), [](double value) { return value > 150.0; })) /
      frame_ms.size();
    const bool causal = command_effect_distance > 5.0 && truth_distance > 5.0;
    const bool quality_pass =
      goal_reached && collisions == 0 && goal_distance <= 0.30 &&
      drift_percent < 5.0 && deadline_miss_rate < 0.05 && causal &&
      invalid_commands == 0 && final_observed_voxels >= 500 &&
      maximum_occupied_cells >= 10 && minimum_inliers >= 30;

    std::ofstream json(run.json);
    if (!json) throw std::runtime_error("cannot open JSON");
    json << std::fixed << std::setprecision(9)
         << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"scenario\": \"cudanav_s_course\",\n"
         << "  \"stages\": [\"gpu_kiss_icp\", \"gpu_voxel_mapping\", "
            "\"gpu_esdf\", \"cuda_mppi\", \"command_driven_plant\"],\n"
         << "  \"frames\": " << frames << ",\n"
         << "  \"simulated_duration_s\": " << frames * kDt << ",\n"
         << "  \"wall_time_ms\": " << wall_ms << ",\n"
         << "  \"goal_reached\": " << (goal_reached ? "true" : "false") << ",\n"
         << "  \"collision_count\": " << collisions << ",\n"
         << "  \"ground_truth_distance_m\": " << truth_distance << ",\n"
         << "  \"ground_truth_goal_distance_m\": " << goal_distance << ",\n"
         << "  \"odometry_ate_rmse_m\": " << ate << ",\n"
         << "  \"odometry_position_error_m\": " << final_error << ",\n"
         << "  \"odometry_drift_percent\": " << drift_percent << ",\n"
         << "  \"minimum_inliers\": " << minimum_inliers << ",\n"
         << "  \"command_effect_distance_m\": " << command_effect_distance << ",\n"
         << "  \"causal_command_effect\": " << (causal ? "true" : "false") << ",\n"
         << "  \"command_deadline_miss_rate\": " << deadline_miss_rate << ",\n"
         << "  \"frame_ms_p95\": " << percentile(frame_ms, 0.95) << ",\n"
         << "  \"mppi_solve_ms_p95\": " << percentile(solve_ms, 0.95) << ",\n"
         << "  \"invalid_commands\": " << invalid_commands << ",\n"
         << "  \"all_colliding_evaluations\": " << all_colliding << ",\n"
         << "  \"retreating_evaluations\": " << retreating << ",\n"
         << "  \"minimum_nonzero_valid_rollout_ratio\": "
         << minimum_nonzero_valid_ratio << ",\n"
         << "  \"final_observed_voxels\": " << final_observed_voxels << ",\n"
         << "  \"maximum_occupied_cells\": " << maximum_occupied_cells << ",\n"
         << "  \"gpu\": {\"name\": \"" << properties.name
         << "\", \"driver_version\": " << driver_version << "},\n"
         << "  \"claims\": {\"native_gpu_core_closed_loop\": true, "
            "\"ros2_runtime\": false, \"real_data\": false},\n"
         << "  \"quality_pass\": " << (quality_pass ? "true" : "false") << "\n"
         << "}\n";
    json.close();
    std::printf(
      "%s: goal_distance=%.3f drift=%.3f%% collisions=%d frames=%d\n",
      quality_pass ? "PASS" : "FAIL", goal_distance, drift_percent, collisions, frames);
    return run.check && !quality_pass ? 2 : 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "cudanav_gpu_closed_loop_s_course: %s\n", error.what());
    return 1;
  }
}
