#include "cudarobotics/esdf_2d_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cudarobotics {
namespace {

#define ESDF_CUDA_CHECK(call)                                                   \
    do {                                                                        \
        const cudaError_t esdf_cuda_error = (call);                             \
        if (esdf_cuda_error != cudaSuccess) {                                   \
            throw std::runtime_error(                                           \
                std::string("CUDA error: ") + cudaGetErrorString(esdf_cuda_error)); \
        }                                                                       \
    } while (0)

constexpr float kInfinity = 1.0e20f;

__global__ void classify_occupancy_kernel(
    const signed char* occupancy,
    unsigned char* seeds,
    int cells,
    int threshold,
    bool unknown_is_occupied,
    unsigned int* occupied_count,
    unsigned int* unknown_count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cells) return;
    const int value = occupancy[index];
    const bool unknown = value < 0;
    const bool occupied = unknown ? unknown_is_occupied : value >= threshold;
    seeds[index] = occupied ? 1 : 0;
    if (occupied) atomicAdd(occupied_count, 1u);
    if (unknown) atomicAdd(unknown_count, 1u);
}

__device__ void distance_transform_1d(
    const float* input,
    int input_stride,
    float* output,
    int output_stride,
    int length,
    int* parabola_locations,
    float* boundaries)
{
    int envelope_size = 0;
    parabola_locations[0] = 0;
    boundaries[0] = -kInfinity;
    boundaries[1] = kInfinity;
    for (int q = 1; q < length; ++q) {
        float intersection;
        while (true) {
            const int p = parabola_locations[envelope_size];
            const float fq = input[q * input_stride];
            const float fp = input[p * input_stride];
            const float qf = static_cast<float>(q);
            const float pf = static_cast<float>(p);
            intersection =
                ((fq + qf * qf) - (fp + pf * pf)) /
                static_cast<float>(2 * (q - p));
            if (intersection > boundaries[envelope_size] || envelope_size == 0) break;
            --envelope_size;
        }
        ++envelope_size;
        parabola_locations[envelope_size] = q;
        boundaries[envelope_size] = intersection;
        boundaries[envelope_size + 1] = kInfinity;
    }
    envelope_size = 0;
    for (int q = 0; q < length; ++q) {
        while (boundaries[envelope_size + 1] < static_cast<float>(q)) {
            ++envelope_size;
        }
        const int p = parabola_locations[envelope_size];
        const float delta = static_cast<float>(q - p);
        output[q * output_stride] = delta * delta + input[p * input_stride];
    }
}

__global__ void row_distance_kernel(
    const unsigned char* seeds,
    float* row_distance,
    int width,
    int height,
    int* locations,
    float* boundaries)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const int input_offset = row * width;
    const int scratch_offset = row * width;
    const int boundary_offset = row * (width + 1);
    for (int x = 0; x < width; ++x) {
        row_distance[input_offset + x] = seeds[input_offset + x] ? 0.0f : kInfinity;
    }
    distance_transform_1d(
        row_distance + input_offset, 1,
        row_distance + input_offset, 1,
        width, locations + scratch_offset, boundaries + boundary_offset);
}

__global__ void column_distance_kernel(
    const float* row_distance,
    float* distances,
    int width,
    int height,
    float resolution,
    float max_distance,
    int* locations,
    float* boundaries)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= width) return;
    const int scratch_offset = column * height;
    const int boundary_offset = column * (height + 1);
    distance_transform_1d(
        row_distance + column, width,
        distances + column, width,
        height, locations + scratch_offset, boundaries + boundary_offset);
    for (int y = 0; y < height; ++y) {
        const int index = y * width + column;
        const float metric = sqrtf(fmaxf(0.0f, distances[index])) * resolution;
        distances[index] = fminf(max_distance, metric);
    }
}

bool is_occupied(
    std::int8_t value, int threshold, UnknownSpacePolicy unknown_policy)
{
    if (value < 0) return unknown_policy == UnknownSpacePolicy::Occupied;
    return value >= threshold;
}

void validate_request(
    std::size_t size,
    int width,
    int height,
    float resolution,
    float max_distance)
{
    if (width <= 0 || height <= 0) throw std::invalid_argument("ESDF dimensions must be positive");
    if (width > INT_MAX / height) throw std::length_error("ESDF cell count exceeds int range");
    if (size != static_cast<std::size_t>(width) * height)
        throw std::invalid_argument("occupancy size must equal width * height");
    if (!(resolution > 0.0f) || !std::isfinite(resolution))
        throw std::invalid_argument("ESDF resolution must be finite and positive");
    if (!(max_distance > 0.0f) || !std::isfinite(max_distance))
        throw std::invalid_argument("ESDF max_distance must be finite and positive");
}

}  // namespace

std::string validate_esdf_2d_config(const Esdf2DConfig& c)
{
    if (c.max_width <= 0 || c.max_height <= 0)
        return "maximum ESDF dimensions must be positive";
    if (c.max_width > INT_MAX / c.max_height)
        return "maximum ESDF cell count exceeds CUDA kernel index range";
    if (c.occupancy_threshold < 0 || c.occupancy_threshold > 100)
        return "occupancy_threshold must be in [0, 100]";
    if (c.unknown_policy != UnknownSpacePolicy::Free &&
        c.unknown_policy != UnknownSpacePolicy::Occupied)
        return "unknown_policy is invalid";
    return {};
}

const char* unknown_space_policy_name(UnknownSpacePolicy policy)
{
    return policy == UnknownSpacePolicy::Occupied ? "occupied" : "free";
}

struct Esdf2DGpu::Impl {
    explicit Impl(const Esdf2DConfig& value) : config(value)
    {
        const std::string error = validate_esdf_2d_config(config);
        if (!error.empty()) throw std::invalid_argument(error);
        max_cells = config.max_width * config.max_height;
        const int max_lines = std::max(config.max_width, config.max_height);
        try {
            ESDF_CUDA_CHECK(cudaMalloc(&occupancy, max_cells * sizeof(signed char)));
            ESDF_CUDA_CHECK(cudaMalloc(&seeds, max_cells * sizeof(unsigned char)));
            ESDF_CUDA_CHECK(cudaMalloc(&row_distance, max_cells * sizeof(float)));
            ESDF_CUDA_CHECK(cudaMalloc(&distances, max_cells * sizeof(float)));
            ESDF_CUDA_CHECK(cudaMalloc(&locations, max_cells * sizeof(int)));
            ESDF_CUDA_CHECK(cudaMalloc(
                &boundaries, (max_cells + max_lines) * sizeof(float)));
            ESDF_CUDA_CHECK(cudaMalloc(&occupied_count, sizeof(unsigned int)));
            ESDF_CUDA_CHECK(cudaMalloc(&unknown_count, sizeof(unsigned int)));
            ESDF_CUDA_CHECK(cudaEventCreate(&event_start));
            ESDF_CUDA_CHECK(cudaEventCreate(&event_stop));
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    void release() noexcept
    {
        if (event_start) cudaEventDestroy(event_start);
        if (event_stop) cudaEventDestroy(event_stop);
        cudaFree(occupancy);
        cudaFree(seeds);
        cudaFree(row_distance);
        cudaFree(distances);
        cudaFree(locations);
        cudaFree(boundaries);
        cudaFree(occupied_count);
        cudaFree(unknown_count);
        occupancy = nullptr;
        seeds = nullptr;
        row_distance = distances = boundaries = nullptr;
        locations = nullptr;
        occupied_count = unknown_count = nullptr;
        event_start = event_stop = nullptr;
    }

    Esdf2DResult compute(
        const std::int8_t* host_occupancy,
        int width,
        int height,
        float resolution,
        float max_distance)
    {
        if (!host_occupancy) throw std::invalid_argument("occupancy pointer must not be null");
        validate_request(
            static_cast<std::size_t>(width) * height,
            width, height, resolution, max_distance);
        if (width > config.max_width || height > config.max_height)
            throw std::length_error("occupancy dimensions exceed configured ESDF capacity");
        const int cells = width * height;
        for (int index = 0; index < cells; ++index) {
            if (host_occupancy[index] < -1 || host_occupancy[index] > 100)
                throw std::invalid_argument("occupancy values must lie in [-1, 100]");
        }
        Esdf2DResult result;
        result.width = width;
        result.height = height;
        result.resolution = resolution;
        result.max_distance = max_distance;
        result.distances.resize(cells);

        ESDF_CUDA_CHECK(cudaMemcpy(
            occupancy, host_occupancy, cells * sizeof(signed char),
            cudaMemcpyHostToDevice));
        unsigned int zero = 0;
        ESDF_CUDA_CHECK(cudaMemcpy(
            occupied_count, &zero, sizeof(zero), cudaMemcpyHostToDevice));
        ESDF_CUDA_CHECK(cudaMemcpy(
            unknown_count, &zero, sizeof(zero), cudaMemcpyHostToDevice));
        ESDF_CUDA_CHECK(cudaEventRecord(event_start));
        classify_occupancy_kernel<<<(cells + 255) / 256, 256>>>(
            occupancy, seeds, cells, config.occupancy_threshold,
            config.unknown_policy == UnknownSpacePolicy::Occupied,
            occupied_count, unknown_count);
        ESDF_CUDA_CHECK(cudaGetLastError());
        row_distance_kernel<<<(height + 127) / 128, 128>>>(
            seeds, row_distance, width, height, locations, boundaries);
        ESDF_CUDA_CHECK(cudaGetLastError());
        column_distance_kernel<<<(width + 127) / 128, 128>>>(
            row_distance, distances, width, height, resolution, max_distance,
            locations, boundaries);
        ESDF_CUDA_CHECK(cudaGetLastError());
        ESDF_CUDA_CHECK(cudaEventRecord(event_stop));
        ESDF_CUDA_CHECK(cudaEventSynchronize(event_stop));
        ESDF_CUDA_CHECK(cudaEventElapsedTime(
            &result.gpu_ms, event_start, event_stop));
        ESDF_CUDA_CHECK(cudaMemcpy(
            result.distances.data(), distances, cells * sizeof(float),
            cudaMemcpyDeviceToHost));
        ESDF_CUDA_CHECK(cudaMemcpy(
            &zero, occupied_count, sizeof(zero), cudaMemcpyDeviceToHost));
        result.occupied_cells = zero;
        ESDF_CUDA_CHECK(cudaMemcpy(
            &zero, unknown_count, sizeof(zero), cudaMemcpyDeviceToHost));
        result.unknown_cells = zero;
        return result;
    }

    Esdf2DConfig config;
    int max_cells = 0;
    signed char* occupancy = nullptr;
    unsigned char* seeds = nullptr;
    float* row_distance = nullptr;
    float* distances = nullptr;
    int* locations = nullptr;
    float* boundaries = nullptr;
    unsigned int* occupied_count = nullptr;
    unsigned int* unknown_count = nullptr;
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;
};

Esdf2DGpu::Esdf2DGpu(const Esdf2DConfig& config) : impl_(new Impl(config)) {}
Esdf2DGpu::~Esdf2DGpu() = default;
Esdf2DGpu::Esdf2DGpu(Esdf2DGpu&&) noexcept = default;
Esdf2DGpu& Esdf2DGpu::operator=(Esdf2DGpu&&) noexcept = default;

Esdf2DResult Esdf2DGpu::compute(
    const std::int8_t* occupancy,
    int width,
    int height,
    float resolution,
    float max_distance)
{
    return impl_->compute(occupancy, width, height, resolution, max_distance);
}

Esdf2DResult Esdf2DGpu::compute(
    const std::vector<std::int8_t>& occupancy,
    int width,
    int height,
    float resolution,
    float max_distance)
{
    validate_request(occupancy.size(), width, height, resolution, max_distance);
    return compute(occupancy.data(), width, height, resolution, max_distance);
}

const Esdf2DConfig& Esdf2DGpu::config() const noexcept { return impl_->config; }

Esdf2DResult compute_esdf_2d_cpu_reference(
    const std::vector<std::int8_t>& occupancy,
    int width,
    int height,
    float resolution,
    float max_distance,
    int occupancy_threshold,
    UnknownSpacePolicy unknown_policy)
{
    validate_request(occupancy.size(), width, height, resolution, max_distance);
    if (occupancy_threshold < 0 || occupancy_threshold > 100)
        throw std::invalid_argument("occupancy_threshold must be in [0, 100]");
    std::vector<std::pair<int, int>> obstacles;
    std::size_t unknown_count = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::int8_t value = occupancy[y * width + x];
            if (value < -1 || value > 100)
                throw std::invalid_argument("occupancy values must lie in [-1, 100]");
            if (value < 0) ++unknown_count;
            if (is_occupied(value, occupancy_threshold, unknown_policy)) {
                obstacles.emplace_back(x, y);
            }
        }
    }
    Esdf2DResult result;
    result.width = width;
    result.height = height;
    result.resolution = resolution;
    result.max_distance = max_distance;
    result.occupied_cells = obstacles.size();
    result.unknown_cells = unknown_count;
    result.distances.resize(occupancy.size(), max_distance);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float best_squared = kInfinity;
            for (const auto& obstacle : obstacles) {
                const float dx = static_cast<float>(x - obstacle.first);
                const float dy = static_cast<float>(y - obstacle.second);
                best_squared = std::min(best_squared, dx * dx + dy * dy);
            }
            result.distances[y * width + x] =
                std::min(max_distance, std::sqrt(best_squared) * resolution);
        }
    }
    return result;
}

}  // namespace cudarobotics
