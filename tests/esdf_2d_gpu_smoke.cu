#include "cudarobotics/esdf_2d_gpu.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

bool compare(
    const cudarobotics::Esdf2DResult& gpu,
    const cudarobotics::Esdf2DResult& cpu,
    float tolerance)
{
    if (gpu.distances.size() != cpu.distances.size() ||
        gpu.occupied_cells != cpu.occupied_cells ||
        gpu.unknown_cells != cpu.unknown_cells)
        return false;
    for (std::size_t index = 0; index < gpu.distances.size(); ++index) {
        if (!std::isfinite(gpu.distances[index]) ||
            std::fabs(gpu.distances[index] - cpu.distances[index]) > tolerance)
        {
            std::fprintf(
                stderr, "distance mismatch at %zu: gpu=%f cpu=%f\n",
                index, gpu.distances[index], cpu.distances[index]);
            return false;
        }
    }
    return true;
}

bool run_case(
    cudarobotics::UnknownSpacePolicy policy,
    int width,
    int height,
    unsigned int seed)
{
    using namespace cudarobotics;
    constexpr float resolution = 0.2f;
    constexpr float max_distance = 1.1f;
    std::mt19937 random(seed);
    std::uniform_int_distribution<int> distribution(0, 9);
    std::vector<std::int8_t> occupancy(width * height, 0);
    for (auto& value : occupancy) {
        const int sample = distribution(random);
        value = sample == 0 ? -1 : (sample <= 2 ? 100 : 0);
    }

    Esdf2DConfig config;
    config.max_width = width;
    config.max_height = height;
    config.occupancy_threshold = 50;
    config.unknown_policy = policy;
    Esdf2DGpu gpu(config);
    const auto gpu_result =
        gpu.compute(occupancy, width, height, resolution, max_distance);
    const auto cpu_result = compute_esdf_2d_cpu_reference(
        occupancy, width, height, resolution, max_distance, 50, policy);
    return compare(gpu_result, cpu_result, 1e-5f);
}

}  // namespace

int main()
{
    using namespace cudarobotics;
    const int shapes[][2] = {{1, 1}, {1, 9}, {11, 1}, {17, 13}, {32, 31}};
    for (unsigned int shape = 0; shape < 5; ++shape) {
        if (!run_case(
                UnknownSpacePolicy::Free,
                shapes[shape][0], shapes[shape][1], 42 + shape) ||
            !run_case(
                UnknownSpacePolicy::Occupied,
                shapes[shape][0], shapes[shape][1], 142 + shape))
        {
            std::fprintf(
                stderr, "GPU ESDF differs from exact CPU reference at shape %dx%d\n",
                shapes[shape][0], shapes[shape][1]);
            return 1;
        }
    }

    Esdf2DConfig config;
    config.max_width = 8;
    config.max_height = 6;
    config.unknown_policy = UnknownSpacePolicy::Free;
    Esdf2DGpu gpu(config);
    std::vector<std::int8_t> no_obstacles(8 * 6, 0);
    auto result = gpu.compute(no_obstacles, 8, 6, 0.1f, 2.0f);
    for (float distance : result.distances) {
        if (distance != 2.0f) {
            std::fprintf(stderr, "obstacle-free ESDF did not clamp to max_distance\n");
            return 2;
        }
    }

    std::vector<std::int8_t> all_occupied(8 * 6, 100);
    result = gpu.compute(all_occupied, 8, 6, 0.1f, 2.0f);
    for (float distance : result.distances) {
        if (distance != 0.0f) {
            std::fprintf(stderr, "occupied cell has non-zero ESDF distance\n");
            return 3;
        }
    }

    bool rejected_bad_shape = false;
    try {
        gpu.compute(std::vector<std::int8_t>(3, 0), 2, 2, 0.1f, 1.0f);
    } catch (const std::invalid_argument&) {
        rejected_bad_shape = true;
    }
    if (!rejected_bad_shape) {
        std::fprintf(stderr, "malformed occupancy shape was accepted\n");
        return 4;
    }
    bool rejected_bad_value = false;
    try {
        std::vector<std::int8_t> invalid_values(8 * 6, 0);
        invalid_values[7] = -2;
        gpu.compute(invalid_values, 8, 6, 0.1f, 1.0f);
    } catch (const std::invalid_argument&) {
        rejected_bad_value = true;
    }
    if (!rejected_bad_value) {
        std::fprintf(stderr, "non-standard occupancy value was accepted\n");
        return 5;
    }

    std::printf("Exact GPU ESDF vs CPU reference: PASS\n");
    return 0;
}
