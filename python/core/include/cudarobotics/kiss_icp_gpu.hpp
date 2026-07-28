#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace cudarobotics {

struct KissIcpMat3 {
    float m[9] = {1.0f, 0.0f, 0.0f,
                  0.0f, 1.0f, 0.0f,
                  0.0f, 0.0f, 1.0f};
};

struct KissIcpPose {
    KissIcpMat3 R;
    float t[3] = {0.0f, 0.0f, 0.0f};
};

enum class KissIcpNnBackend {
    Voxel,
    BruteForce,
};

struct KissIcpConfig {
    float map_voxel_size = 0.35f;
    float scan_voxel_size = 0.22f;
    float map_radius = 40.0f;
    float threshold_min = 1.0f;
    float threshold_max = 3.0f;
    int max_icp_iterations = 12;
    int normal_neighbors = 12;
    std::size_t max_scan_points = 200000;
    std::size_t max_map_points = 200000;
    std::size_t hash_capacity = 1u << 19;
    KissIcpNnBackend nn_backend = KissIcpNnBackend::Voxel;
};

struct KissIcpAlignmentStats {
    int iterations = 0;
    int inliers = 0;
    float rmse = 0.0f;
    float nn_ms = 0.0f;
    float threshold = 0.0f;
};

struct KissIcpTiming {
    double index_build_ms = 0.0;
    double map_upload_ms = 0.0;
    double map_normal_ms = 0.0;
};

struct KissIcpFrameResult {
    KissIcpPose pose;
    KissIcpAlignmentStats alignment;
    std::size_t input_points = 0;
    std::size_t sampled_points = 0;
    std::size_t map_points = 0;
    bool map_initialized = false;
};

// Returns an empty string when the configuration is valid.
std::string validate_kiss_icp_config(const KissIcpConfig& config);
const char* kiss_icp_backend_name(KissIcpNnBackend backend);

class KissIcpOdometry {
public:
    explicit KissIcpOdometry(const KissIcpConfig& config = KissIcpConfig{});
    ~KissIcpOdometry();

    KissIcpOdometry(KissIcpOdometry&&) noexcept;
    KissIcpOdometry& operator=(KissIcpOdometry&&) noexcept;
    KissIcpOdometry(const KissIcpOdometry&) = delete;
    KissIcpOdometry& operator=(const KissIcpOdometry&) = delete;

    void reset(const KissIcpPose& initial_pose = KissIcpPose{});
    KissIcpFrameResult register_scan(const float* xyz, std::size_t point_count);
    KissIcpFrameResult register_scan(const std::vector<float>& xyz);

    const KissIcpConfig& config() const noexcept;
    const KissIcpPose& pose() const noexcept;
    std::size_t frame_count() const noexcept;
    std::size_t map_point_count() const noexcept;
    std::vector<float> map_snapshot() const;
    KissIcpTiming timing() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cudarobotics
