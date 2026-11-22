#include "cudarobotics/voxel_mapping_gpu.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

int index_2d(
    const cudarobotics::VoxelGridInfo& grid, float world_x, float world_y)
{
    const int x = static_cast<int>((world_x - grid.origin_x) / grid.resolution);
    const int y = static_cast<int>((world_y - grid.origin_y) / grid.resolution);
    return y * grid.width + x;
}

}  // namespace

int main()
{
    using namespace cudarobotics;
    VoxelMappingConfig invalid;
    invalid.rolling_margin_cells = invalid.width;
    if (validate_voxel_mapping_config(invalid).empty()) {
        std::fprintf(stderr, "invalid rolling margin was accepted\n");
        return 1;
    }
    invalid = VoxelMappingConfig{};
    invalid.projection_min_z = 1.0f;
    invalid.projection_max_z = 1.0f;
    if (validate_voxel_mapping_config(invalid).empty()) {
        std::fprintf(stderr, "invalid projection height band was accepted\n");
        return 1;
    }

    VoxelMappingConfig config;
    config.width = 16;
    config.height = 16;
    config.depth = 4;
    config.resolution = 1.0f;
    config.origin_z = -1.0f;
    config.min_range = 0.1f;
    config.max_range = 10.0f;
    config.projection_min_z = -0.5f;
    config.projection_max_z = 0.5f;
    config.rolling_margin_cells = 3;
    config.max_scan_points = 32;

    VoxelMapperGpu mapper(config);
    mapper.reset(0.0f, 0.0f);
    const float origin[3] = {0.0f, 0.0f, 0.0f};
    const std::vector<float> scan = {3.2f, 0.0f, 0.0f};
    const VoxelMappingStats first = mapper.integrate_scan(scan, origin);
    if (first.integrated_rays != 1 || first.observed_voxels < 4) {
        std::fprintf(stderr, "ray was not integrated into the voxel map\n");
        return 2;
    }

    OccupancyProjection projection = mapper.occupancy_projection();
    const int sensor_cell = index_2d(projection.grid, 0.0f, 0.0f);
    const int free_cell = index_2d(projection.grid, 2.0f, 0.0f);
    const int hit_cell = index_2d(projection.grid, 3.2f, 0.0f);
    const int unknown_cell = index_2d(projection.grid, -7.0f, -7.0f);
    if (projection.data[sensor_cell] != 0 ||
        projection.data[free_cell] != 0 ||
        projection.data[hit_cell] != 100 ||
        projection.data[unknown_cell] != -1)
    {
        std::fprintf(stderr, "occupancy projection violated -1/0/100 semantics\n");
        return 3;
    }

    const float shifted_origin[3] = {7.0f, 0.0f, 0.0f};
    const std::vector<float> shifted_scan = {7.0f, 3.2f, 0.0f};
    const VoxelMappingStats shifted =
        mapper.integrate_scan(shifted_scan, shifted_origin);
    if (!shifted.grid_shifted || shifted.shift_x_cells != 7) {
        std::fprintf(stderr, "rolling map did not shift by the expected cells\n");
        return 4;
    }
    projection = mapper.occupancy_projection();
    const int preserved_hit = index_2d(projection.grid, 3.2f, 0.0f);
    const int new_hit = index_2d(projection.grid, 7.0f, 3.2f);
    if (projection.data[preserved_hit] != 100 ||
        projection.data[new_hit] != 100)
    {
        std::fprintf(stderr, "rolling shift did not preserve/add occupied cells\n");
        return 5;
    }

    const std::vector<float> high_scan = {7.0f, 4.0f, 2.0f};
    mapper.integrate_scan(high_scan, shifted_origin);
    projection = mapper.occupancy_projection();
    const int high_hit = index_2d(projection.grid, 7.0f, 4.0f);
    if (projection.data[high_hit] == 100) {
        std::fprintf(stderr, "projection included an obstacle above its height band\n");
        return 6;
    }

    const VoxelGridSnapshot snapshot = mapper.snapshot();
    if (snapshot.log_odds.size() != 16u * 16u * 4u ||
        snapshot.observed.size() != snapshot.log_odds.size())
    {
        std::fprintf(stderr, "voxel snapshot shape is invalid\n");
        return 7;
    }

    bool rejected_bad_shape = false;
    try {
        mapper.integrate_scan(std::vector<float>{1.0f, 2.0f}, origin);
    } catch (const std::invalid_argument&) {
        rejected_bad_shape = true;
    }
    if (!rejected_bad_shape) {
        std::fprintf(stderr, "malformed XYZ input was accepted\n");
        return 8;
    }

    // Integration must not depend on ray order: many rays share voxels near
    // the sensor and some end in the same voxel, so a clamped per-ray update
    // would race. Reversing the scan must give a bit-identical map.
    VoxelMappingConfig order_config = config;
    order_config.max_scan_points = 512;
    order_config.log_odds_max = 1.0f;  // reached within one scan
    std::vector<float> fan;
    for (int i = 0; i < 400; ++i) {
        const float angle = 0.0025f * static_cast<float>(i % 200);
        const float range = (i % 3 == 0) ? 4.3f : 6.6f;
        fan.push_back(range * std::cos(angle));
        fan.push_back(range * std::sin(angle));
        fan.push_back(0.0f);
    }
    std::vector<float> reversed;
    for (int i = static_cast<int>(fan.size()) / 3 - 1; i >= 0; --i) {
        reversed.insert(reversed.end(), fan.begin() + 3 * i, fan.begin() + 3 * i + 3);
    }
    VoxelMapperGpu forward_mapper(order_config);
    VoxelMapperGpu reverse_mapper(order_config);
    forward_mapper.reset(0.0f, 0.0f);
    reverse_mapper.reset(0.0f, 0.0f);
    for (int repeat = 0; repeat < 3; ++repeat) {
        forward_mapper.integrate_scan(fan, origin);
        reverse_mapper.integrate_scan(reversed, origin);
    }
    const VoxelGridSnapshot forward = forward_mapper.snapshot();
    const VoxelGridSnapshot backward = reverse_mapper.snapshot();
    if (forward.log_odds != backward.log_odds || forward.observed != backward.observed) {
        std::fprintf(stderr, "voxel map depends on the order of the rays\n");
        return 9;
    }
    bool reached_max = false;
    for (const float value : forward.log_odds) {
        if (value > order_config.log_odds_max || value < order_config.log_odds_min) {
            std::fprintf(stderr, "log-odds left the clamp range\n");
            return 10;
        }
        reached_max = reached_max || value == order_config.log_odds_max;
    }
    if (!reached_max) {
        std::fprintf(stderr, "repeated hits never reached log_odds_max\n");
        return 11;
    }

    std::printf("GPU rolling voxel mapping smoke: PASS\n");
    return 0;
}
