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
    VoxelLinked,  // Retained pre-optimization correspondence reference.
};

enum class KissIcpNormalBackend {
    Voxel,
    BruteForce,
};

enum class KissIcpReductionBackend {
    Block,
    Atomic,  // Retained per-point atomic reference.
};

enum class KissIcpMapBackend {
    Dense,
    Unordered,  // Retained voxel map and per-frame packing reference.
    Pooled,  // Dense points with reusable nodes and compact reference-order links.
    Validate,  // Compare pooled point bits and order with Dense every frame.
};

enum class KissIcpDownsampleBackend {
    Cached,
    Unordered,
    Pooled,
    Validate,  // Byte-compare pooled output with the cached reference.
};

enum class KissIcpNormalUpdate {
    Full,
    Incremental,
    Validate,  // Compare every incremental normal with full recomputation.
};

enum class KissIcpNormalSchedule {
    Fused,
    Split,  // Copy reusable normals, then process a compact recomputation queue.
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
    KissIcpNormalBackend normal_backend = KissIcpNormalBackend::Voxel;
    KissIcpReductionBackend reduction_backend = KissIcpReductionBackend::Block;
    KissIcpMapBackend map_backend = KissIcpMapBackend::Pooled;
    KissIcpDownsampleBackend downsample_backend = KissIcpDownsampleBackend::Pooled;
    bool normal_query_cell_order = true;  // Schedule nearby queries together; preserve point IDs.
    KissIcpNormalUpdate normal_update = KissIcpNormalUpdate::Incremental;
    KissIcpNormalSchedule normal_schedule = KissIcpNormalSchedule::Fused;
};

struct KissIcpAlignmentStats {
    int iterations = 0;
    int inliers = 0;
    float rmse = 0.0f;
    float nn_ms = 0.0f;
    float normal_equation_ms = 0.0f;
    float threshold = 0.0f;
};

struct KissIcpTiming {
    double deskew_ms = 0.0;
    double index_build_ms = 0.0;
    double map_upload_ms = 0.0;
    double map_normal_ms = 0.0;
    // Wall-clock stage durations; map_prune/insert/pack are within map_update.
    double validation_ms = 0.0;
    double deskew_wall_ms = 0.0;
    double downsample_ms = 0.0;
    double icp_ms = 0.0;
    double map_update_ms = 0.0;
    double map_prune_ms = 0.0;
    double map_insert_ms = 0.0;
    double map_pack_ms = 0.0;
    double map_reorder_ms = 0.0;  // GPU gathering into the reference point order.
    double normal_cache_prepare_ms = 0.0;  // Remapping, transfers, delta index and cache commit.
    std::size_t normal_reused_points = 0;
    std::size_t normal_recomputed_points = 0;
    std::size_t downsample_upstream_allocations = 0;  // New arena slabs since reset.
    std::size_t downsample_arena_bytes = 0;  // Current retained slab capacity.
    std::size_t host_map_upstream_allocations = 0;  // New node-pool slabs since reset.
    std::size_t host_map_pool_bytes = 0;  // Retained node slabs, excluding buckets/points.
    std::size_t host_map_order_bytes = 0;  // Reserved integer links for exact order export.
    std::size_t normal_recompute_queue_bytes = 0;  // Split queue and its device count.
};

struct KissIcpFrameResult {
    KissIcpPose pose;
    KissIcpAlignmentStats alignment;
    std::size_t input_points = 0;
    std::size_t sampled_points = 0;
    std::size_t map_points = 0;
    bool map_initialized = false;
    bool deskewed = false;
    float point_time_span_s = 0.0f;
    std::vector<float> deskewed_xyz;
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
    KissIcpFrameResult register_scan(
        const float* xyz,
        std::size_t point_count,
        const float* point_times_seconds);
    KissIcpFrameResult register_scan(
        const float* xyz,
        std::size_t point_count,
        const float* point_times_seconds,
        float scan_start_time_seconds,
        float scan_end_time_seconds);
    KissIcpFrameResult register_scan(const std::vector<float>& xyz);
    KissIcpFrameResult register_scan(
        const std::vector<float>& xyz,
        const std::vector<float>& point_times_seconds);

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
