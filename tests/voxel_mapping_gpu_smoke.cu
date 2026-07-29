#include "cudarobotics/voxel_mapping_gpu.hpp"

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

    std::printf("GPU rolling voxel mapping smoke: PASS\n");
    return 0;
}
