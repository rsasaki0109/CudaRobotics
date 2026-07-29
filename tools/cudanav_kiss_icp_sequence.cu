#include "cudarobotics/kiss_icp_gpu.hpp"

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
    int maximum_frames = 0;
    int minimum_inliers = 30;
    double maximum_ate_rmse_m = 5.0;
    double maximum_final_drift_percent = 10.0;
    bool check = false;
};

struct Frame {
    std::uint64_t stamp_ns = 0;
    float reference[4]{};
    std::vector<float> xyz;
};

template <typename T>
T read_value(std::ifstream& stream) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!stream) throw std::runtime_error("truncated sequence header");
    return value;
}

Frame read_frame(std::ifstream& stream) {
    Frame frame;
    frame.stamp_ns = read_value<std::uint64_t>(stream);
    stream.read(reinterpret_cast<char*>(frame.reference), sizeof(frame.reference));
    const auto point_count = read_value<std::uint32_t>(stream);
    if (point_count < 30 || point_count > 200000) {
        throw std::runtime_error("invalid sequence point count");
    }
    frame.xyz.resize(static_cast<std::size_t>(point_count) * 3u);
    stream.read(
        reinterpret_cast<char*>(frame.xyz.data()),
        static_cast<std::streamsize>(frame.xyz.size() * sizeof(float)));
    if (!stream) throw std::runtime_error("truncated sequence point payload");
    return frame;
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

void usage(const char* executable) {
    std::fprintf(
        stderr,
        "Usage: %s --sequence FILE --json FILE --csv FILE [--check] "
        "[--maximum-frames N] [--minimum-inliers N] "
        "[--maximum-ate-rmse-m X] [--maximum-final-drift-percent X]\n",
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
        else if (argument == "--maximum-frames") {
            options.maximum_frames = std::stoi(next());
        } else if (argument == "--minimum-inliers") {
            options.minimum_inliers = std::stoi(next());
        } else if (argument == "--maximum-ate-rmse-m") {
            options.maximum_ate_rmse_m = std::stod(next());
        } else if (argument == "--maximum-final-drift-percent") {
            options.maximum_final_drift_percent = std::stod(next());
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
    if (options.maximum_frames < 0 || options.minimum_inliers < 1 ||
        options.maximum_ate_rmse_m <= 0.0 ||
        options.maximum_final_drift_percent <= 0.0) {
        throw std::invalid_argument("invalid numeric option");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    using cudarobotics::KissIcpConfig;
    using cudarobotics::KissIcpFrameResult;
    using cudarobotics::KissIcpOdometry;

    Options options;
    try {
        options = parse_options(argc, argv);
        std::ifstream input(options.sequence, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open sequence");
        char magic[8]{};
        input.read(magic, sizeof(magic));
        const char expected[8] = {'C', 'R', 'K', 'I', 'C', 'P', '1', '\0'};
        if (!input || std::memcmp(magic, expected, sizeof(magic)) != 0) {
            throw std::runtime_error("sequence magic mismatch");
        }
        const auto version = read_value<std::uint32_t>(input);
        const auto declared_frames = read_value<std::uint32_t>(input);
        if (version != 1 || declared_frames < 2) {
            throw std::runtime_error("unsupported sequence header");
        }
        const std::uint32_t frame_limit =
            options.maximum_frames > 0
                ? std::min(
                      declared_frames,
                      static_cast<std::uint32_t>(options.maximum_frames))
                : declared_frames;

        KissIcpConfig config;
        config.max_scan_points = 200000;
        config.max_map_points = 200000;
        config.hash_capacity = 1u << 19;
        int device = 0;
        int driver_version = 0;
        cudaDeviceProp device_properties{};
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaGetDeviceProperties(&device_properties, device) != cudaSuccess ||
            cudaDriverGetVersion(&driver_version) != cudaSuccess) {
            throw std::runtime_error("failed to query CUDA device identity");
        }
        KissIcpOdometry odometry(config);

        std::ofstream csv(options.csv);
        if (!csv) throw std::runtime_error("cannot open trajectory CSV");
        csv << "frame,stamp_ns,reference_x,reference_y,reference_z,reference_yaw,"
               "estimated_x,estimated_y,estimated_z,estimated_yaw,xy_error_m,"
               "inliers,rmse,nn_ms,map_points\n";

        std::vector<double> xy_errors;
        std::vector<double> yaw_errors;
        std::vector<double> inliers;
        std::vector<double> rmse;
        std::vector<double> nn_ms;
        double reference_distance = 0.0;
        double estimated_distance = 0.0;
        double previous_reference_x = 0.0;
        double previous_reference_y = 0.0;
        double previous_estimated_x = 0.0;
        double previous_estimated_y = 0.0;
        std::uint64_t first_stamp_ns = 0;
        std::uint64_t last_stamp_ns = 0;
        double final_error = 0.0;

        const auto started = std::chrono::steady_clock::now();
        for (std::uint32_t frame_index = 0; frame_index < frame_limit; ++frame_index) {
            Frame frame = read_frame(input);
            KissIcpFrameResult result = odometry.register_scan(frame.xyz);
            const double estimated_yaw =
                std::atan2(result.pose.R.m[3], result.pose.R.m[0]);
            const double dx = static_cast<double>(result.pose.t[0]) -
                              static_cast<double>(frame.reference[0]);
            const double dy = static_cast<double>(result.pose.t[1]) -
                              static_cast<double>(frame.reference[1]);
            const double xy_error = std::hypot(dx, dy);
            if (frame_index == 0) {
                first_stamp_ns = frame.stamp_ns;
            } else {
                reference_distance += std::hypot(
                    static_cast<double>(frame.reference[0]) - previous_reference_x,
                    static_cast<double>(frame.reference[1]) - previous_reference_y);
                estimated_distance += std::hypot(
                    static_cast<double>(result.pose.t[0]) - previous_estimated_x,
                    static_cast<double>(result.pose.t[1]) - previous_estimated_y);
                inliers.push_back(result.alignment.inliers);
                rmse.push_back(result.alignment.rmse);
                nn_ms.push_back(result.alignment.nn_ms);
            }
            last_stamp_ns = frame.stamp_ns;
            previous_reference_x = frame.reference[0];
            previous_reference_y = frame.reference[1];
            previous_estimated_x = result.pose.t[0];
            previous_estimated_y = result.pose.t[1];
            final_error = xy_error;
            xy_errors.push_back(xy_error);
            yaw_errors.push_back(std::fabs(
                wrap_angle(estimated_yaw - static_cast<double>(frame.reference[3]))));
            csv << frame_index << ',' << frame.stamp_ns << ','
                << frame.reference[0] << ',' << frame.reference[1] << ','
                << frame.reference[2] << ',' << frame.reference[3] << ','
                << result.pose.t[0] << ',' << result.pose.t[1] << ','
                << result.pose.t[2] << ',' << estimated_yaw << ','
                << xy_error << ',' << result.alignment.inliers << ','
                << result.alignment.rmse << ',' << result.alignment.nn_ms << ','
                << result.map_points << '\n';
        }
        if (frame_limit == declared_frames &&
            input.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("sequence has trailing bytes");
        }
        const double wall_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();
        const double squared_error = std::inner_product(
            xy_errors.begin(), xy_errors.end(), xy_errors.begin(), 0.0);
        const double ate_rmse = std::sqrt(squared_error / xy_errors.size());
        const double final_drift_percent =
            reference_distance > 1e-6 ? 100.0 * final_error / reference_distance : 0.0;
        const int minimum_observed_inliers = inliers.empty()
            ? 0
            : static_cast<int>(*std::min_element(inliers.begin(), inliers.end()));
        const bool quality_pass =
            ate_rmse <= options.maximum_ate_rmse_m &&
            final_drift_percent <= options.maximum_final_drift_percent &&
            minimum_observed_inliers >= options.minimum_inliers;

        std::ofstream json(options.json);
        if (!json) throw std::runtime_error("cannot open JSON report");
        json << std::setprecision(10)
             << "{\n"
             << "  \"schema_version\": 1,\n"
             << "  \"algorithm\": \"cudarobotics.gpu_kiss_icp_real_sequence.v1\",\n"
             << "  \"sequence\": " << json_string(options.sequence) << ",\n"
             << "  \"trajectory_csv\": " << json_string(options.csv) << ",\n"
             << "  \"gpu\": {\n"
             << "    \"device\": " << device << ",\n"
             << "    \"name\": " << json_string(device_properties.name) << ",\n"
             << "    \"uuid\": \"" << gpu_uuid(device_properties) << "\",\n"
             << "    \"driver_version\": " << driver_version << ",\n"
             << "    \"compute_capability\": \""
             << device_properties.major << '.' << device_properties.minor
             << "\"\n"
             << "  },\n"
             << "  \"nn_backend\": \""
             << cudarobotics::kiss_icp_backend_name(config.nn_backend) << "\",\n"
             << "  \"frames\": " << frame_limit << ",\n"
             << "  \"first_stamp_ns\": " << first_stamp_ns << ",\n"
             << "  \"last_stamp_ns\": " << last_stamp_ns << ",\n"
             << "  \"duration_s\": "
             << static_cast<double>(last_stamp_ns - first_stamp_ns) / 1e9 << ",\n"
             << "  \"wall_time_ms\": " << wall_ms << ",\n"
             << "  \"mean_frame_ms\": " << wall_ms / frame_limit << ",\n"
             << "  \"reference_path_length_m\": " << reference_distance << ",\n"
             << "  \"estimated_path_length_m\": " << estimated_distance << ",\n"
             << "  \"ate_rmse_m\": " << ate_rmse << ",\n"
             << "  \"final_xy_error_m\": " << final_error << ",\n"
             << "  \"final_drift_percent\": " << final_drift_percent << ",\n"
             << "  \"yaw_error_p95_rad\": " << percentile(yaw_errors, 0.95) << ",\n"
             << "  \"inliers_min\": " << minimum_observed_inliers << ",\n"
             << "  \"inliers_median\": " << percentile(inliers, 0.5) << ",\n"
             << "  \"alignment_rmse_p95\": " << percentile(rmse, 0.95) << ",\n"
             << "  \"nn_ms_p95\": " << percentile(nn_ms, 0.95) << ",\n"
             << "  \"thresholds\": {\n"
             << "    \"maximum_ate_rmse_m\": " << options.maximum_ate_rmse_m << ",\n"
             << "    \"maximum_final_drift_percent\": "
             << options.maximum_final_drift_percent << ",\n"
             << "    \"minimum_inliers\": " << options.minimum_inliers << "\n"
             << "  },\n"
             << "  \"quality_pass\": " << (quality_pass ? "true" : "false") << "\n"
             << "}\n";
        std::printf(
            "real KISS-ICP frames=%u ATE=%.3f m final_drift=%.2f%% "
            "mean=%.3f ms quality=%s\n",
            frame_limit,
            ate_rmse,
            final_drift_percent,
            wall_ms / frame_limit,
            quality_pass ? "PASS" : "FAIL");
        return options.check && !quality_pass ? 2 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "cudanav_kiss_icp_sequence: %s\n", error.what());
        usage(argv[0]);
        return 1;
    }
}
