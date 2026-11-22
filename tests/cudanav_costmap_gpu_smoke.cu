// Device-resident CudaNav map chain: GPU footprint clearing and costmap versus
// the CPU references, and ESDF device input versus the host-input overload.
#include "cudarobotics/cudanav_costmap_gpu.hpp"
#include "cudarobotics/esdf_2d_gpu.hpp"
#include "cudarobotics/voxel_mapping_gpu.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using namespace cudarobotics;

void check(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}

void cuda(cudaError_t error)
{
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}

template <typename T>
T* upload(const std::vector<T>& host)
{
    T* device = nullptr;
    cuda(cudaMalloc(&device, host.size() * sizeof(T)));
    cuda(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

template <typename T>
std::vector<T> download(const T* device, std::size_t count)
{
    std::vector<T> host(count);
    cuda(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

void compare_costmaps(
    const std::vector<unsigned char>& gpu,
    const std::vector<unsigned char>& cpu)
{
    check(gpu.size() == cpu.size(), "costmap sizes differ");
    std::size_t exp_band_differences = 0;
    for (std::size_t index = 0; index < cpu.size(); ++index) {
        if (gpu[index] == cpu[index]) continue;
        const bool exp_band = cpu[index] >= 1 && cpu[index] <= 252 &&
                              gpu[index] >= 1 && gpu[index] <= 252;
        if (!exp_band || std::abs(int(gpu[index]) - int(cpu[index])) > 1) {
            std::fprintf(stderr, "cost mismatch at %zu: gpu=%d cpu=%d\n",
                         index, gpu[index], cpu[index]);
            std::exit(1);
        }
        ++exp_band_differences;
    }
    check(exp_band_differences <= cpu.size() / 1000 + 1,
          "too many one-step exponential-band differences");
}

// Random occupancy plus distances sampled densely around every band edge.
void costmap_matches_cpu_reference(int width, int height, unsigned int seed)
{
    std::mt19937 random(seed);
    std::uniform_int_distribution<int> occupancy_sample(0, 9);
    std::uniform_real_distribution<float> distance_sample(0.0f, 1.3f);
    const float edges[] = {0.25f, 1.0f};
    std::vector<std::int8_t> occupancy(width * height);
    std::vector<float> distances(occupancy.size());
    for (std::size_t index = 0; index < occupancy.size(); ++index) {
        const int sample = occupancy_sample(random);
        occupancy[index] = sample == 0 ? -1 : sample == 1 ? 100 : sample == 2 ? 49 : sample == 3 ? 50 : 0;
        float distance = distance_sample(random);
        if (index % 7 == 0) {
            const float edge = edges[(index / 7) % 2];
            distance = std::nextafter(edge, (index / 14) % 2 ? 2.0f : 0.0f);
            if ((index / 28) % 2) distance = edge;
        }
        distances[index] = distance;
    }

    CostmapBuilderGpu builder(width, height);
    std::int8_t* device_occupancy = upload(occupancy);
    float* device_distances = upload(distances);
    const auto built = builder.build(device_occupancy, device_distances, width, height);
    compare_costmaps(
        download(built.costmap, occupancy.size()),
        make_costmap_cpu(occupancy, distances));
    cuda(cudaFree(device_occupancy));
    cuda(cudaFree(device_distances));
}

void footprint_matches_cpu_reference()
{
    const CostmapGridGeometry grid{97, 83, -3.1f, 2.05f, 0.05f};
    std::vector<std::int8_t> occupancy(grid.width * grid.height);
    std::mt19937 random(7);
    std::uniform_int_distribution<int> sample(0, 3);
    for (auto& value : occupancy) value = sample(random) == 0 ? 100 : (sample(random) == 0 ? -1 : 0);

    // Centres on and between cell centres, so cells sit exactly on the radius.
    const float centres[][2] = {{-1.0f, 4.0f}, {-0.975f, 4.025f}, {1.4f, 5.9f}, {-10.0f, -10.0f}};
    CostmapBuilderGpu builder(grid.width, grid.height);
    for (const auto& centre : centres) {
        for (const float radius : {0.0f, 0.10f, 0.30f, 0.45f}) {
            std::vector<std::int8_t> cpu = occupancy;
            clear_footprint_cpu(cpu, grid, centre[0], centre[1], radius);
            std::int8_t* device = upload(occupancy);
            const std::size_t occupied =
                builder.clear_footprint(device, grid, centre[0], centre[1], radius);
            const auto gpu = download(device, occupancy.size());
            cuda(cudaFree(device));
            check(gpu == cpu, "GPU footprint clearing differs from the CPU reference");
            std::size_t expected = 0;
            for (const auto value : cpu) expected += value == 100 ? 1 : 0;
            check(occupied == expected, "occupied count after clearing is wrong");
        }
    }
}

void batched_discs_match_sequential_clearing()
{
    const CostmapGridGeometry grid{97, 83, -3.1f, 2.05f, 0.05f};
    std::vector<std::int8_t> occupancy(grid.width * grid.height);
    std::mt19937 random(11);
    std::uniform_int_distribution<int> sample(0, 3);
    for (auto& value : occupancy) value = sample(random) == 0 ? 100 : (sample(random) == 0 ? -1 : 0);
    std::uniform_real_distribution<float> x(-3.5f, 2.0f), y(1.8f, 6.5f), r(0.0f, 0.5f);
    CostmapBuilderGpu builder(grid.width, grid.height);
    // Growing counts exercise the buffer reallocation; 0 only counts.
    for (const int count : {0, 1, 7, 300, 40}) {
        std::vector<float> discs;
        std::vector<std::int8_t> cpu = occupancy;
        for (int i = 0; i < count; ++i) {
            discs.insert(discs.end(), {x(random), y(random), r(random)});
            clear_footprint_cpu(cpu, grid, discs[3 * i], discs[3 * i + 1], discs[3 * i + 2]);
        }
        std::int8_t* device = upload(occupancy);
        const std::size_t occupied = builder.clear_discs(device, grid, discs.data(), count);
        const auto gpu = download(device, occupancy.size());
        cuda(cudaFree(device));
        check(gpu == cpu, "batched disc clearing differs from sequential clearing");
        std::size_t expected = 0;
        for (const auto value : cpu) expected += value == 100 ? 1 : 0;
        check(occupied == expected, "occupied count after batched clearing is wrong");
    }
    bool rejected = false;
    const float bad[3] = {0.0f, std::numeric_limits<float>::quiet_NaN(), 0.2f};
    std::int8_t* device = upload(occupancy);
    try {
        builder.clear_discs(device, grid, bad, 1);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    cuda(cudaFree(device));
    check(rejected, "a non-finite disc must be rejected");
}

void esdf_device_input_matches_host_input()
{
    const int width = 61;
    const int height = 47;
    std::vector<std::int8_t> occupancy(width * height, 0);
    std::mt19937 random(11);
    std::uniform_int_distribution<int> sample(0, 9);
    for (auto& value : occupancy) {
        const int draw = sample(random);
        value = draw == 0 ? -1 : (draw <= 2 ? 100 : 0);
    }
    for (const auto policy : {UnknownSpacePolicy::Free, UnknownSpacePolicy::Occupied}) {
        Esdf2DConfig config;
        config.max_width = width;
        config.max_height = height;
        config.unknown_policy = policy;
        Esdf2DGpu esdf(config);
        const auto host = esdf.compute(occupancy, width, height, 0.1f, 2.0f);
        std::int8_t* device_occupancy = upload(occupancy);
        const auto device = esdf.compute_device(device_occupancy, width, height, 0.1f, 2.0f);
        const auto distances = download(device.distances, occupancy.size());
        check(std::memcmp(distances.data(), host.distances.data(),
                          distances.size() * sizeof(float)) == 0,
              "device-input ESDF is not bit-identical to the host-input ESDF");
        check(device.occupied_cells == host.occupied_cells &&
              device.unknown_cells == host.unknown_cells,
              "device-input ESDF counters differ");

        std::vector<std::int8_t> invalid = occupancy;
        invalid[invalid.size() / 2] = 101;
        std::int8_t* device_invalid = upload(invalid);
        bool rejected = false;
        try {
            esdf.compute_device(device_invalid, width, height, 0.1f, 2.0f);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "device-input ESDF accepted an out-of-range cell");
        cuda(cudaFree(device_invalid));
        cuda(cudaFree(device_occupancy));
    }
}

// The device projection must hold exactly what the host projection returns.
void voxel_device_projection_matches_host()
{
    VoxelMappingConfig config;
    config.width = 64;
    config.height = 48;
    config.depth = 16;
    config.resolution = 0.1f;
    config.origin_z = -0.8f;
    config.max_range = 6.0f;
    config.rolling_margin_cells = 8;
    VoxelMapperGpu mapper(config);
    mapper.reset(0.0f, 0.0f);
    std::vector<float> points;
    for (int index = 0; index < 360; ++index) {
        const float angle = index * 3.14159265f / 180.0f;
        points.insert(points.end(), {2.0f * std::cos(angle), 1.5f * std::sin(angle), 0.1f});
    }
    const float origin[3] = {0.0f, 0.0f, 0.2f};
    mapper.integrate_scan(points, origin);
    const auto host = mapper.occupancy_projection();
    const auto device = mapper.occupancy_projection_device();
    check(device.grid.width == host.grid.width && device.grid.height == host.grid.height &&
              device.grid.origin_x == host.grid.origin_x,
          "device projection grid info differs");
    check(download(device.data, host.data.size()) == host.data,
          "device projection differs from the host projection");
    std::size_t occupied = 0;
    for (const auto value : host.data) occupied += value == 100 ? 1 : 0;
    check(occupied > 0, "projection should contain occupied cells");
}

}  // namespace

int main()
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "no CUDA device\n");
        return 1;
    }
    costmap_matches_cpu_reference(1, 1, 1);
    costmap_matches_cpu_reference(37, 29, 2);
    costmap_matches_cpu_reference(256, 256, 3);
    footprint_matches_cpu_reference();
    batched_discs_match_sequential_clearing();
    esdf_device_input_matches_host_input();
    voxel_device_projection_matches_host();

    bool rejected = false;
    try {
        CudaNavCostmapParams params;
        params.inflation_radius = 0.1f;
        CostmapBuilderGpu invalid(4, 4, params);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "unordered costmap radii must be rejected");
    std::printf("cudanav_costmap_gpu_smoke: OK\n");
    return 0;
}
