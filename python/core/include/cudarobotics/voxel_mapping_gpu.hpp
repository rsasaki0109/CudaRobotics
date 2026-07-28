#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cudarobotics {

struct VoxelMappingConfig {
    int width = 256;
    int height = 256;
    int depth = 32;
    float resolution = 0.10f;
    float origin_z = -1.0f;
    float min_range = 0.10f;
    float max_range = 20.0f;
    float log_odds_occupied = 0.85f;
    float log_odds_free = -0.40f;
    float log_odds_min = -4.0f;
    float log_odds_max = 4.0f;
    float occupied_threshold = 0.0f;
    int rolling_margin_cells = 48;
    std::size_t max_scan_points = 200000;
};

struct VoxelGridInfo {
    int width = 0;
    int height = 0;
    int depth = 0;
    float resolution = 0.0f;
    float origin_x = 0.0f;
    float origin_y = 0.0f;
    float origin_z = 0.0f;
};

struct VoxelMappingStats {
    std::size_t input_points = 0;
    std::size_t integrated_rays = 0;
    std::size_t observed_voxels = 0;
    bool grid_shifted = false;
    int shift_x_cells = 0;
    int shift_y_cells = 0;
    float shift_ms = 0.0f;
    float raycast_ms = 0.0f;
};

struct OccupancyProjection {
    VoxelGridInfo grid;
    std::vector<std::int8_t> data;
    float gpu_ms = 0.0f;
};

struct VoxelGridSnapshot {
    VoxelGridInfo grid;
    std::vector<float> log_odds;
    std::vector<std::uint8_t> observed;
};

std::string validate_voxel_mapping_config(const VoxelMappingConfig& config);

class VoxelMapperGpu {
public:
    explicit VoxelMapperGpu(const VoxelMappingConfig& config = VoxelMappingConfig{});
    ~VoxelMapperGpu();

    VoxelMapperGpu(VoxelMapperGpu&&) noexcept;
    VoxelMapperGpu& operator=(VoxelMapperGpu&&) noexcept;
    VoxelMapperGpu(const VoxelMapperGpu&) = delete;
    VoxelMapperGpu& operator=(const VoxelMapperGpu&) = delete;

    void reset(float center_x, float center_y);
    VoxelMappingStats integrate_scan(
        const float* xyz_world,
        std::size_t point_count,
        const float sensor_origin_world[3]);
    VoxelMappingStats integrate_scan(
        const std::vector<float>& xyz_world,
        const float sensor_origin_world[3]);

    const VoxelMappingConfig& config() const noexcept;
    VoxelGridInfo grid_info() const noexcept;
    OccupancyProjection occupancy_projection();
    VoxelGridSnapshot snapshot() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cudarobotics
