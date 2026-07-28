#include "cudarobotics/kiss_icp_gpu.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

bool near(float a, float b, float tolerance = 1e-5f) {
    return std::fabs(a - b) <= tolerance;
}

}  // namespace

int main() {
    using namespace cudarobotics;

    KissIcpConfig invalid;
    invalid.hash_capacity = 1000;
    if (validate_kiss_icp_config(invalid).empty()) {
        std::fprintf(stderr, "non-power-of-two hash capacity was accepted\n");
        return 1;
    }

    KissIcpConfig config;
    config.map_voxel_size = 0.1f;
    config.scan_voxel_size = 0.05f;
    config.max_scan_points = 128;
    config.max_map_points = 256;
    config.hash_capacity = 512;

    KissIcpPose initial;
    initial.t[0] = 2.0f;
    initial.t[1] = -1.0f;
    initial.t[2] = 0.5f;

    std::vector<float> scan;
    for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
            scan.push_back(0.25f * x);
            scan.push_back(0.25f * y);
            scan.push_back(0.03f * (x * x + y));
        }
    }

    KissIcpOdometry odometry(config);
    odometry.reset(initial);
    KissIcpFrameResult first = odometry.register_scan(scan);
    if (odometry.frame_count() != 1 || first.map_points == 0 ||
        !near(first.pose.t[0], initial.t[0]) ||
        !near(first.pose.t[1], initial.t[1]) ||
        !near(first.pose.t[2], initial.t[2])) {
        std::fprintf(stderr, "first scan did not preserve the explicit initial pose\n");
        return 2;
    }

    KissIcpFrameResult second = odometry.register_scan(scan);
    if (odometry.frame_count() != 2 || second.alignment.inliers < 10 ||
        !near(second.pose.t[0], initial.t[0], 1e-3f) ||
        !near(second.pose.t[1], initial.t[1], 1e-3f) ||
        !near(second.pose.t[2], initial.t[2], 1e-3f)) {
        std::fprintf(stderr, "streaming registration changed a stationary scan\n");
        return 3;
    }

    bool rejected_bad_shape = false;
    try {
        odometry.register_scan(std::vector<float>{0.0f, 1.0f});
    } catch (const std::invalid_argument&) {
        rejected_bad_shape = true;
    }
    if (!rejected_bad_shape) {
        std::fprintf(stderr, "malformed xyz vector was accepted\n");
        return 4;
    }

    odometry.reset();
    if (odometry.frame_count() != 0 || !odometry.map_snapshot().empty()) {
        std::fprintf(stderr, "reset did not clear streaming state\n");
        return 5;
    }

    std::printf("KISS-ICP reusable GPU streaming API: PASS\n");
    return 0;
}
