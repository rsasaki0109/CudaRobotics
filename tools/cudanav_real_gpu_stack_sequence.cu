#include "cuda_mppi_controller/mppi_gpu.hpp"
#include "cudarobotics/esdf_2d_gpu.hpp"
#include "cudarobotics/kiss_icp_gpu.hpp"
#include "cudarobotics/voxel_mapping_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string sequence;
    std::string json;
    std::string csv;
    int control_stride = 10;
    int minimum_inliers = 30;
    int minimum_observed_voxels = 500;
    int minimum_occupied_cells = 10;
    int minimum_control_evaluations = 20;
    int maximum_all_colliding_evaluations = 3;
    double minimum_valid_rollout_ratio = 0.01;
    double maximum_safety_stop_speed = 0.05;
    double maximum_ate_rmse_m = 5.0;
    double maximum_final_drift_percent = 10.0;
    float kiss_map_voxel_size = 0.35f;
    float kiss_scan_voxel_size = 0.22f;
    float kiss_map_radius = 40.0f;
    int kiss_normal_neighbors = 12;
    bool check = false;
};

struct Frame {
    std::uint64_t stamp_ns = 0;
    float reference[4]{};
    std::vector<float> xyz;
    std::vector<float> point_times;
    float scan_start_time_s = 0.0f;
    float scan_end_time_s = 0.0f;
};

template <typename T>
T read_value(std::ifstream& stream) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!stream) throw std::runtime_error("truncated sequence");
    return value;
}

std::vector<Frame> read_sequence(
    const std::string& path,
    std::uint32_t& sequence_version) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open sequence");
    char magic[8]{};
    input.read(magic, sizeof(magic));
    const char expected[8] = {'C', 'R', 'K', 'I', 'C', 'P', '1', '\0'};
    if (!input || std::memcmp(magic, expected, sizeof(magic)) != 0) {
        throw std::runtime_error("sequence magic mismatch");
    }
    const auto version = read_value<std::uint32_t>(input);
    sequence_version = version;
    const auto frame_count = read_value<std::uint32_t>(input);
    if ((version != 1 && version != 2) ||
        frame_count < 2 || frame_count > 100000) {
        throw std::runtime_error("unsupported sequence header");
    }
    std::vector<Frame> frames;
    frames.reserve(frame_count);
    for (std::uint32_t frame_index = 0; frame_index < frame_count; ++frame_index) {
        Frame frame;
        frame.stamp_ns = read_value<std::uint64_t>(input);
        input.read(
            reinterpret_cast<char*>(frame.reference),
            sizeof(frame.reference));
        const auto point_count = read_value<std::uint32_t>(input);
        if (point_count < 30 || point_count > 200000) {
            throw std::runtime_error("invalid sequence point count");
        }
        frame.xyz.resize(static_cast<std::size_t>(point_count) * 3u);
        if (version == 2) {
            frame.scan_start_time_s = read_value<float>(input);
            frame.scan_end_time_s = read_value<float>(input);
            frame.point_times.resize(point_count);
            for (std::uint32_t index = 0; index < point_count; ++index) {
                frame.xyz[index * 3] = read_value<float>(input);
                frame.xyz[index * 3 + 1] = read_value<float>(input);
                frame.xyz[index * 3 + 2] = read_value<float>(input);
                frame.point_times[index] = read_value<float>(input);
            }
        } else {
            input.read(
                reinterpret_cast<char*>(frame.xyz.data()),
                static_cast<std::streamsize>(frame.xyz.size() * sizeof(float)));
        }
        if (!input) throw std::runtime_error("truncated sequence point payload");
        frames.push_back(std::move(frame));
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("sequence has trailing bytes");
    }
    return frames;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t index = std::min(
        values.size() - 1,
        static_cast<std::size_t>(std::ceil(fraction * values.size()) - 1.0));
    return values[index];
}

double wrap_angle(double value) {
    return std::atan2(std::sin(value), std::cos(value));
}

std::string json_string(const std::string& value) {
    std::ostringstream stream;
    stream << '"';
    for (const unsigned char character : value) {
        switch (character) {
            case '"': stream << "\\\""; break;
            case '\\': stream << "\\\\"; break;
            case '\b': stream << "\\b"; break;
            case '\f': stream << "\\f"; break;
            case '\n': stream << "\\n"; break;
            case '\r': stream << "\\r"; break;
            case '\t': stream << "\\t"; break;
            default:
                if (character < 0x20) {
                    stream << "\\u" << std::hex << std::setfill('0')
                           << std::setw(4)
                           << static_cast<unsigned int>(character)
                           << std::dec;
                } else {
                    stream << static_cast<char>(character);
                }
        }
    }
    stream << '"';
    return stream.str();
}

std::string gpu_uuid(const cudaDeviceProp& properties) {
    std::ostringstream stream;
    stream << "GPU-";
    for (int index = 0; index < 16; ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) stream << '-';
        stream << std::hex << std::setfill('0') << std::setw(2)
               << static_cast<unsigned int>(
                      static_cast<unsigned char>(properties.uuid.bytes[index]));
    }
    return stream.str();
}

std::vector<float> transform_scan_navigation_height_frame(
    const std::vector<float>& scan,
    const cudarobotics::KissIcpPose& pose)
{
    std::vector<float> world(scan.size());
    for (std::size_t index = 0; index < scan.size(); index += 3) {
        const float x = scan[index + 0];
        const float y = scan[index + 1];
        const float z = scan[index + 2];
        world[index + 0] =
            pose.R.m[0] * x + pose.R.m[1] * y + pose.R.m[2] * z + pose.t[0];
        world[index + 1] =
            pose.R.m[3] * x + pose.R.m[4] * y + pose.R.m[5] * z + pose.t[1];
        // The rolling map feeds a planar navigation costmap. Keep odometry X/Y,
        // but express height relative to the current LiDAR origin so road
        // elevation cannot leave the map's finite Z extent.
        world[index + 2] =
            pose.R.m[6] * x + pose.R.m[7] * y + pose.R.m[8] * z;
    }
    return world;
}

std::vector<unsigned char> esdf_costmap(
    const cudarobotics::OccupancyProjection& occupancy,
    const cudarobotics::Esdf2DResult& esdf)
{
    if (occupancy.data.size() != esdf.distances.size()) {
        throw std::runtime_error("occupancy and ESDF shapes differ");
    }
    constexpr float inscribed_radius = 0.25f;
    constexpr float inflation_radius = 1.0f;
    constexpr float scaling = 3.0f;
    std::vector<unsigned char> costmap(occupancy.data.size(), 0);
    for (std::size_t index = 0; index < occupancy.data.size(); ++index) {
        if (occupancy.data[index] < 0) {
            costmap[index] = 255;
        } else if (occupancy.data[index] >= 50) {
            costmap[index] = 254;
        } else if (esdf.distances[index] <= inscribed_radius) {
            costmap[index] = 253;
        } else if (esdf.distances[index] < inflation_radius) {
            const float value = 252.0f * std::exp(
                -scaling * (esdf.distances[index] - inscribed_radius));
            costmap[index] = static_cast<unsigned char>(
                std::max(1.0f, std::min(252.0f, value)));
        }
    }
    return costmap;
}

void clear_robot_footprint(
    cudarobotics::OccupancyProjection& occupancy,
    float robot_x,
    float robot_y,
    float radius)
{
    for (int y = 0; y < occupancy.grid.height; ++y) {
        for (int x = 0; x < occupancy.grid.width; ++x) {
            const float world_x =
                occupancy.grid.origin_x + (x + 0.5f) * occupancy.grid.resolution;
            const float world_y =
                occupancy.grid.origin_y + (y + 0.5f) * occupancy.grid.resolution;
            if (std::hypot(world_x - robot_x, world_y - robot_y) <= radius) {
                occupancy.data[
                    static_cast<std::size_t>(y) * occupancy.grid.width + x] = 0;
            }
        }
    }
}

std::vector<float> local_path(
    const std::vector<Frame>& frames,
    std::size_t begin,
    std::size_t& end)
{
    constexpr float lookahead_m = 3.0f;
    end = begin;
    float arc = 0.0f;
    for (std::size_t index = begin + 1; index < frames.size(); ++index) {
        arc += std::hypot(
            frames[index].reference[0] - frames[index - 1].reference[0],
            frames[index].reference[1] - frames[index - 1].reference[1]);
        end = index;
        if (arc >= lookahead_m || end - begin + 1 >= 256) break;
    }
    if (end == begin && begin + 1 < frames.size()) ++end;
    std::vector<float> path;
    path.reserve((end - begin + 1) * 2);
    for (std::size_t index = begin; index <= end; ++index) {
        path.push_back(frames[index].reference[0]);
        path.push_back(frames[index].reference[1]);
    }
    return path;
}

void usage(const char* executable) {
    std::fprintf(
        stderr,
        "Usage: %s --sequence FILE --json FILE --csv FILE [--check] "
        "[--control-stride N] [--minimum-inliers N] "
        "[--minimum-observed-voxels N] [--minimum-occupied-cells N] "
        "[--minimum-control-evaluations N] "
        "[--maximum-all-colliding-evaluations N] "
        "[--minimum-valid-rollout-ratio X] [--maximum-ate-rmse-m X] "
        "[--maximum-final-drift-percent X] "
        "[--maximum-safety-stop-speed X] [--kiss-map-voxel-size X] "
        "[--kiss-scan-voxel-size X] [--kiss-map-radius X] "
        "[--kiss-normal-neighbors N]\n",
        executable);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&]() -> std::string {
            if (++index >= argc) throw std::invalid_argument("missing option value");
            return argv[index];
        };
        if (argument == "--sequence") options.sequence = next();
        else if (argument == "--json") options.json = next();
        else if (argument == "--csv") options.csv = next();
        else if (argument == "--control-stride") {
            options.control_stride = std::stoi(next());
        } else if (argument == "--minimum-inliers") {
            options.minimum_inliers = std::stoi(next());
        } else if (argument == "--minimum-observed-voxels") {
            options.minimum_observed_voxels = std::stoi(next());
        } else if (argument == "--minimum-occupied-cells") {
            options.minimum_occupied_cells = std::stoi(next());
        } else if (argument == "--minimum-control-evaluations") {
            options.minimum_control_evaluations = std::stoi(next());
        } else if (argument == "--maximum-all-colliding-evaluations") {
            options.maximum_all_colliding_evaluations = std::stoi(next());
        } else if (argument == "--minimum-valid-rollout-ratio") {
            options.minimum_valid_rollout_ratio = std::stod(next());
        } else if (argument == "--maximum-safety-stop-speed") {
            options.maximum_safety_stop_speed = std::stod(next());
        } else if (argument == "--maximum-ate-rmse-m") {
            options.maximum_ate_rmse_m = std::stod(next());
        } else if (argument == "--maximum-final-drift-percent") {
            options.maximum_final_drift_percent = std::stod(next());
        } else if (argument == "--kiss-map-voxel-size") {
            options.kiss_map_voxel_size = std::stof(next());
        } else if (argument == "--kiss-scan-voxel-size") {
            options.kiss_scan_voxel_size = std::stof(next());
        } else if (argument == "--kiss-map-radius") {
            options.kiss_map_radius = std::stof(next());
        } else if (argument == "--kiss-normal-neighbors") {
            options.kiss_normal_neighbors = std::stoi(next());
        } else if (argument == "--check") {
            options.check = true;
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.sequence.empty() || options.json.empty() || options.csv.empty()) {
        throw std::invalid_argument("--sequence, --json, and --csv are required");
    }
    if (options.control_stride < 1 || options.minimum_inliers < 1 ||
        options.minimum_observed_voxels < 1 || options.minimum_occupied_cells < 1 ||
        options.minimum_control_evaluations < 1 ||
        options.maximum_all_colliding_evaluations < 0 ||
        options.minimum_valid_rollout_ratio < 0.0 ||
        options.minimum_valid_rollout_ratio > 1.0 ||
        options.maximum_safety_stop_speed < 0.0 ||
        options.maximum_ate_rmse_m <= 0.0 ||
        options.maximum_final_drift_percent <= 0.0 ||
        options.kiss_map_voxel_size <= 0.0f ||
        options.kiss_scan_voxel_size <= 0.0f ||
        options.kiss_map_radius <= 0.0f ||
        options.kiss_normal_neighbors < 1 ||
        options.kiss_normal_neighbors > 20) {
        throw std::invalid_argument("invalid numeric option");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::uint32_t sequence_version = 0;
        const std::vector<Frame> frames =
            read_sequence(options.sequence, sequence_version);

        int device = 0;
        int driver_version = 0;
        cudaDeviceProp device_properties{};
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaGetDeviceProperties(&device_properties, device) != cudaSuccess ||
            cudaDriverGetVersion(&driver_version) != cudaSuccess) {
            throw std::runtime_error("failed to query CUDA device identity");
        }

        cudarobotics::KissIcpConfig kiss_config;
        kiss_config.map_voxel_size = options.kiss_map_voxel_size;
        kiss_config.scan_voxel_size = options.kiss_scan_voxel_size;
        kiss_config.map_radius = options.kiss_map_radius;
        kiss_config.normal_neighbors = options.kiss_normal_neighbors;
        kiss_config.max_scan_points = 200000;
        kiss_config.max_map_points = 200000;
        kiss_config.hash_capacity = 1u << 19;
        cudarobotics::KissIcpOdometry odometry(kiss_config);

        cudarobotics::VoxelMappingConfig voxel_config;
        voxel_config.width = 256;
        voxel_config.height = 256;
        voxel_config.depth = 32;
        voxel_config.resolution = 0.10f;
        voxel_config.origin_z = -2.0f;
        voxel_config.max_range = 20.0f;
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
        mppi_params.v_max = 1.5f;
        mppi_params.v_min = 0.0f;
        mppi_params.costmap_weight = 5.0f;
        mppi_params.distance_field_weight = 0.0f;
        cuda_mppi_controller::MppiGpu mppi(mppi_params);

        std::ofstream csv(options.csv);
        if (!csv) throw std::runtime_error("cannot open trajectory CSV");
        csv << "frame,stamp_ns,reference_x,reference_y,estimated_x,estimated_y,"
               "xy_error_m,inliers,observed_voxels,integrated_rays,occupied_cells,"
               "unknown_cells,projection_gpu_ms,esdf_gpu_ms,mppi_ms,"
               "robot_cost,robot_clearance_m,valid_rollout_ratio,all_colliding,"
               "retreating,command_v,command_w\n";

        std::vector<double> xy_errors;
        std::vector<double> yaw_errors;
        std::vector<double> inliers;
        std::vector<double> nn_ms;
        std::vector<double> raycast_ms;
        std::vector<double> projection_ms;
        std::vector<double> esdf_ms;
        std::vector<double> mppi_ms;
        std::vector<double> frame_ms;
        std::vector<double> valid_ratios;
        std::size_t final_observed_voxels = 0;
        std::size_t maximum_occupied_cells = 0;
        std::size_t maximum_unknown_cells = 0;
        std::size_t total_integrated_rays = 0;
        int control_evaluations = 0;
        int all_colliding_evaluations = 0;
        int retreating_evaluations = 0;
        int nonfinite_commands = 0;
        int map_shifts = 0;
        int maximum_robot_cost = 0;
        double minimum_robot_clearance = std::numeric_limits<double>::infinity();
        double maximum_all_colliding_abs_v = 0.0;
        double reference_distance = 0.0;
        double estimated_distance = 0.0;
        double final_error = 0.0;
        double maximum_abs_estimated_sensor_height = 0.0;
        std::size_t frames_with_integrated_rays = 0;
        std::size_t deskewed_frames = 0;
        std::vector<double> point_time_spans;
        float previous_estimated_x = 0.0f;
        float previous_estimated_y = 0.0f;

        const auto wall_start = std::chrono::steady_clock::now();
        for (std::size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
            const auto frame_start = std::chrono::steady_clock::now();
            const Frame& frame = frames[frame_index];
            const auto odometry_result = frame.point_times.empty()
                ? odometry.register_scan(frame.xyz)
                : odometry.register_scan(
                      frame.xyz.data(),
                      frame.xyz.size() / 3,
                      frame.point_times.data(),
                      frame.scan_start_time_s,
                      frame.scan_end_time_s);
            if (odometry_result.deskewed) {
                ++deskewed_frames;
                point_time_spans.push_back(
                    odometry_result.point_time_span_s);
            }
            const auto& pose = odometry_result.pose;
            const double dx = pose.t[0] - frame.reference[0];
            const double dy = pose.t[1] - frame.reference[1];
            const double xy_error = std::hypot(dx, dy);
            const double yaw = std::atan2(pose.R.m[3], pose.R.m[0]);
            xy_errors.push_back(xy_error);
            yaw_errors.push_back(std::fabs(wrap_angle(yaw - frame.reference[3])));
            final_error = xy_error;
            if (frame_index > 0) {
                reference_distance += std::hypot(
                    frame.reference[0] - frames[frame_index - 1].reference[0],
                    frame.reference[1] - frames[frame_index - 1].reference[1]);
                estimated_distance += std::hypot(
                    pose.t[0] - previous_estimated_x,
                    pose.t[1] - previous_estimated_y);
                inliers.push_back(odometry_result.alignment.inliers);
                nn_ms.push_back(odometry_result.alignment.nn_ms);
            }
            previous_estimated_x = pose.t[0];
            previous_estimated_y = pose.t[1];
            maximum_abs_estimated_sensor_height = std::max(
                maximum_abs_estimated_sensor_height,
                std::fabs(static_cast<double>(pose.t[2])));

            const std::vector<float>& mapping_scan =
                odometry_result.deskewed_xyz.empty()
                    ? frame.xyz
                    : odometry_result.deskewed_xyz;
            const std::vector<float> world =
                transform_scan_navigation_height_frame(mapping_scan, pose);
            const float sensor_origin[3] = {pose.t[0], pose.t[1], 0.0f};
            const auto mapping = mapper.integrate_scan(world, sensor_origin);
            if (mapping.integrated_rays > 0) ++frames_with_integrated_rays;
            final_observed_voxels = mapping.observed_voxels;
            total_integrated_rays += mapping.integrated_rays;
            raycast_ms.push_back(mapping.raycast_ms + mapping.shift_ms);
            if (mapping.grid_shifted) ++map_shifts;

            int occupied_cells = -1;
            int unknown_cells = -1;
            double projection_gpu_ms = -1.0;
            double esdf_gpu_ms = -1.0;
            double control_ms = -1.0;
            int robot_cost = -1;
            double robot_clearance = -1.0;
            double valid_ratio = -1.0;
            int all_colliding = -1;
            int retreating = -1;
            double command_v = std::numeric_limits<double>::quiet_NaN();
            double command_w = std::numeric_limits<double>::quiet_NaN();
            const bool control_frame =
                frame_index % static_cast<std::size_t>(options.control_stride) == 0 ||
                frame_index + 1 == frames.size();
            if (control_frame) {
                auto projection = mapper.occupancy_projection();
                clear_robot_footprint(projection, pose.t[0], pose.t[1], 0.30f);
                occupied_cells = static_cast<int>(std::count(
                    projection.data.begin(), projection.data.end(),
                    static_cast<std::int8_t>(100)));
                unknown_cells = static_cast<int>(std::count(
                    projection.data.begin(), projection.data.end(),
                    static_cast<std::int8_t>(-1)));
                maximum_occupied_cells = std::max(
                    maximum_occupied_cells,
                    static_cast<std::size_t>(occupied_cells));
                maximum_unknown_cells = std::max(
                    maximum_unknown_cells,
                    static_cast<std::size_t>(unknown_cells));
                projection_gpu_ms = projection.gpu_ms;
                projection_ms.push_back(projection_gpu_ms);
                const auto esdf_result = esdf.compute(
                    projection.data,
                    projection.grid.width,
                    projection.grid.height,
                    projection.grid.resolution,
                    2.0f);
                esdf_gpu_ms = esdf_result.gpu_ms;
                esdf_ms.push_back(esdf_gpu_ms);
                const auto costmap = esdf_costmap(projection, esdf_result);
                const int robot_x = static_cast<int>(std::floor(
                    (pose.t[0] - projection.grid.origin_x) /
                    projection.grid.resolution));
                const int robot_y = static_cast<int>(std::floor(
                    (pose.t[1] - projection.grid.origin_y) /
                    projection.grid.resolution));
                if (robot_x >= 0 && robot_x < projection.grid.width &&
                    robot_y >= 0 && robot_y < projection.grid.height) {
                    const std::size_t robot_index =
                        static_cast<std::size_t>(robot_y) * projection.grid.width +
                        robot_x;
                    robot_cost = costmap[robot_index];
                    robot_clearance = esdf_result.distances[robot_index];
                    maximum_robot_cost = std::max(maximum_robot_cost, robot_cost);
                    minimum_robot_clearance = std::min(
                        minimum_robot_clearance, robot_clearance);
                }
                std::size_t path_end = frame_index;
                const auto path = local_path(frames, frame_index, path_end);
                const auto mppi_started = std::chrono::steady_clock::now();
                const auto command = mppi.compute(
                    pose.t[0], pose.t[1], static_cast<float>(yaw),
                    costmap.data(),
                    projection.grid.width,
                    projection.grid.height,
                    projection.grid.origin_x,
                    projection.grid.origin_y,
                    projection.grid.resolution,
                    path.data(),
                    static_cast<int>(path.size() / 2),
                    frames[path_end].reference[0],
                    frames[path_end].reference[1],
                    frames[path_end].reference[3],
                    path_end + 1 == frames.size());
                control_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - mppi_started).count();
                valid_ratio = command.valid_rollout_ratio;
                all_colliding = command.all_colliding ? 1 : 0;
                retreating = command.retreating ? 1 : 0;
                command_v = command.v;
                command_w = command.w;
                ++control_evaluations;
                all_colliding_evaluations += all_colliding;
                retreating_evaluations += retreating;
                if (all_colliding) {
                    maximum_all_colliding_abs_v = std::max(
                        maximum_all_colliding_abs_v,
                        std::fabs(command_v));
                }
                if (!std::isfinite(command_v) || !std::isfinite(command_w) ||
                    std::fabs(command_v) > mppi_params.v_max + 1e-5 ||
                    std::fabs(command_w) > mppi_params.w_max + 1e-5) {
                    ++nonfinite_commands;
                }
                mppi_ms.push_back(control_ms);
                valid_ratios.push_back(valid_ratio);
            }

            frame_ms.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - frame_start).count());
            csv << frame_index << ',' << frame.stamp_ns << ','
                << frame.reference[0] << ',' << frame.reference[1] << ','
                << pose.t[0] << ',' << pose.t[1] << ',' << xy_error << ','
                << odometry_result.alignment.inliers << ','
                << mapping.observed_voxels << ',' << mapping.integrated_rays << ','
                << occupied_cells << ',' << unknown_cells << ','
                << projection_gpu_ms << ',' << esdf_gpu_ms << ',' << control_ms << ','
                << robot_cost << ',' << robot_clearance << ','
                << valid_ratio << ',' << all_colliding << ','
                << retreating << ','
                << command_v << ',' << command_w << '\n';
        }

        const double wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall_start).count();
        const double squared_error = std::inner_product(
            xy_errors.begin(), xy_errors.end(), xy_errors.begin(), 0.0);
        const double ate_rmse = std::sqrt(squared_error / xy_errors.size());
        const double final_drift_percent =
            reference_distance > 1e-6
                ? 100.0 * final_error / reference_distance
                : 0.0;
        const int minimum_observed_inliers = inliers.empty()
            ? 0
            : static_cast<int>(*std::min_element(inliers.begin(), inliers.end()));
        std::vector<double> nonzero_valid_ratios;
        std::copy_if(
            valid_ratios.begin(), valid_ratios.end(),
            std::back_inserter(nonzero_valid_ratios),
            [](double value) { return value > 0.0; });
        const double minimum_nonzero_valid_ratio = nonzero_valid_ratios.empty()
            ? 0.0
            : *std::min_element(
                nonzero_valid_ratios.begin(), nonzero_valid_ratios.end());
        const bool quality_pass =
            ate_rmse <= options.maximum_ate_rmse_m &&
            final_drift_percent <= options.maximum_final_drift_percent &&
            minimum_observed_inliers >= options.minimum_inliers &&
            final_observed_voxels >=
                static_cast<std::size_t>(options.minimum_observed_voxels) &&
            maximum_occupied_cells >=
                static_cast<std::size_t>(options.minimum_occupied_cells) &&
            frames_with_integrated_rays == frames.size() &&
            control_evaluations >= options.minimum_control_evaluations &&
            minimum_nonzero_valid_ratio >= options.minimum_valid_rollout_ratio &&
            all_colliding_evaluations <=
                options.maximum_all_colliding_evaluations &&
            maximum_all_colliding_abs_v <= options.maximum_safety_stop_speed &&
            nonfinite_commands == 0;

        std::ofstream json(options.json);
        if (!json) throw std::runtime_error("cannot open JSON report");
        json << std::setprecision(10)
             << "{\n"
             << "  \"schema_version\": 1,\n"
             << "  \"algorithm\": \"cudarobotics.real_gpu_stack_sequence.v2\",\n"
             << "  \"sequence\": " << json_string(options.sequence) << ",\n"
             << "  \"trajectory_csv\": " << json_string(options.csv) << ",\n"
             << "  \"gpu\": {\n"
             << "    \"device\": " << device << ",\n"
             << "    \"name\": " << json_string(device_properties.name) << ",\n"
             << "    \"uuid\": " << json_string(gpu_uuid(device_properties)) << ",\n"
             << "    \"driver_version\": " << driver_version << ",\n"
             << "    \"compute_capability\": \""
             << device_properties.major << '.' << device_properties.minor
             << "\"\n"
             << "  },\n"
             << "  \"stages\": [\"gpu_kiss_icp\", \"gpu_voxel_mapping\", "
                "\"gpu_esdf\", \"cuda_mppi\"],\n"
             << "  \"sequence_version\": " << sequence_version << ",\n"
             << "  \"deskew\": {\n"
             << "    \"frames\": " << deskewed_frames << ",\n"
             << "    \"point_time_span_s_p95\": "
             << percentile(point_time_spans, 0.95) << ",\n"
             << "    \"gpu_ms\": " << odometry.timing().deskew_ms << "\n"
             << "  },\n"
             << "  \"odometry_config\": {\n"
             << "    \"map_voxel_size_m\": " << kiss_config.map_voxel_size << ",\n"
             << "    \"scan_voxel_size_m\": " << kiss_config.scan_voxel_size << ",\n"
             << "    \"map_radius_m\": " << kiss_config.map_radius << ",\n"
             << "    \"normal_neighbors\": " << kiss_config.normal_neighbors << "\n"
             << "  },\n"
             << "  \"frames\": " << frames.size() << ",\n"
             << "  \"duration_s\": "
             << static_cast<double>(
                    frames.back().stamp_ns - frames.front().stamp_ns) / 1e9
             << ",\n"
             << "  \"wall_time_ms\": " << wall_ms << ",\n"
             << "  \"mean_frame_ms\": " << wall_ms / frames.size() << ",\n"
             << "  \"frame_ms_p95\": " << percentile(frame_ms, 0.95) << ",\n"
             << "  \"reference_path_length_m\": " << reference_distance << ",\n"
             << "  \"estimated_path_length_m\": " << estimated_distance << ",\n"
             << "  \"ate_rmse_m\": " << ate_rmse << ",\n"
             << "  \"final_xy_error_m\": " << final_error << ",\n"
             << "  \"final_drift_percent\": " << final_drift_percent << ",\n"
             << "  \"yaw_error_p95_rad\": " << percentile(yaw_errors, 0.95) << ",\n"
             << "  \"inliers_min\": " << minimum_observed_inliers << ",\n"
             << "  \"nn_ms_p95\": " << percentile(nn_ms, 0.95) << ",\n"
             << "  \"mapping\": {\n"
             << "    \"height_frame\": \"estimated_sensor_relative\",\n"
             << "    \"maximum_abs_estimated_sensor_height_m\": "
             << maximum_abs_estimated_sensor_height << ",\n"
             << "    \"frames_with_integrated_rays\": "
             << frames_with_integrated_rays << ",\n"
             << "    \"final_observed_voxels\": " << final_observed_voxels << ",\n"
             << "    \"total_integrated_rays\": " << total_integrated_rays << ",\n"
             << "    \"map_shifts\": " << map_shifts << ",\n"
             << "    \"maximum_occupied_cells\": " << maximum_occupied_cells << ",\n"
             << "    \"maximum_unknown_cells\": " << maximum_unknown_cells << ",\n"
             << "    \"raycast_ms_p95\": " << percentile(raycast_ms, 0.95) << ",\n"
             << "    \"projection_ms_p95\": " << percentile(projection_ms, 0.95)
             << "\n"
             << "  },\n"
             << "  \"esdf\": {\n"
             << "    \"unknown_policy\": \"free\",\n"
             << "    \"footprint_clearing_radius_m\": 0.30,\n"
             << "    \"max_distance_m\": 2.0,\n"
             << "    \"gpu_ms_p95\": " << percentile(esdf_ms, 0.95) << "\n"
             << "  },\n"
             << "  \"mppi\": {\n"
             << "    \"control_stride\": " << options.control_stride << ",\n"
             << "    \"evaluations\": " << control_evaluations << ",\n"
             << "    \"minimum_nonzero_valid_rollout_ratio\": "
             << minimum_nonzero_valid_ratio << ",\n"
             << "    \"maximum_robot_cost\": " << maximum_robot_cost << ",\n"
             << "    \"minimum_robot_clearance_m\": "
             << minimum_robot_clearance << ",\n"
             << "    \"all_colliding_evaluations\": "
             << all_colliding_evaluations << ",\n"
             << "    \"retreating_evaluations\": "
             << retreating_evaluations << ",\n"
             << "    \"maximum_all_colliding_abs_v\": "
             << maximum_all_colliding_abs_v << ",\n"
             << "    \"invalid_commands\": " << nonfinite_commands << ",\n"
             << "    \"solve_ms_p95\": " << percentile(mppi_ms, 0.95) << "\n"
             << "  },\n"
             << "  \"thresholds\": {\n"
             << "    \"maximum_ate_rmse_m\": " << options.maximum_ate_rmse_m << ",\n"
             << "    \"maximum_final_drift_percent\": "
             << options.maximum_final_drift_percent << ",\n"
             << "    \"minimum_inliers\": " << options.minimum_inliers << ",\n"
             << "    \"minimum_observed_voxels\": "
             << options.minimum_observed_voxels << ",\n"
             << "    \"minimum_occupied_cells\": "
             << options.minimum_occupied_cells << ",\n"
             << "    \"minimum_control_evaluations\": "
             << options.minimum_control_evaluations << ",\n"
             << "    \"maximum_all_colliding_evaluations\": "
             << options.maximum_all_colliding_evaluations << ",\n"
             << "    \"minimum_valid_rollout_ratio\": "
             << options.minimum_valid_rollout_ratio << ",\n"
             << "    \"maximum_safety_stop_speed\": "
             << options.maximum_safety_stop_speed << "\n"
             << "  },\n"
             << "  \"quality_pass\": " << (quality_pass ? "true" : "false") << "\n"
             << "}\n";

        std::printf(
            "real GPU stack frames=%zu ATE=%.3f m drift=%.2f%% "
            "voxels=%zu occupied=%zu controls=%d valid_min=%.3f quality=%s\n",
            frames.size(), ate_rmse, final_drift_percent,
            final_observed_voxels, maximum_occupied_cells,
            control_evaluations, minimum_nonzero_valid_ratio,
            quality_pass ? "PASS" : "FAIL");
        return options.check && !quality_pass ? 2 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "cudanav_real_gpu_stack_sequence: %s\n", error.what());
        usage(argv[0]);
        return 1;
    }
}
