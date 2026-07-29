#include "cudarobotics/voxel_mapping_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cudarobotics {
namespace {

#define VOXEL_CUDA_CHECK(call)                                                  \
    do {                                                                        \
        const cudaError_t voxel_cuda_error = (call);                            \
        if (voxel_cuda_error != cudaSuccess) {                                  \
            throw std::runtime_error(                                           \
                std::string("CUDA error: ") + cudaGetErrorString(voxel_cuda_error)); \
        }                                                                       \
    } while (0)

__device__ void atomic_add_clamped(float* address, float delta, float low, float high)
{
    int* address_bits = reinterpret_cast<int*>(address);
    int old_bits = *address_bits;
    while (true) {
        const int assumed_bits = old_bits;
        const float old_value = __int_as_float(assumed_bits);
        const float updated = fminf(high, fmaxf(low, old_value + delta));
        old_bits = atomicCAS(address_bits, assumed_bits, __float_as_int(updated));
        if (old_bits == assumed_bits) return;
    }
}

__global__ void clear_grid_kernel(float* log_odds, unsigned int* observed, int cells)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cells) return;
    log_odds[index] = 0.0f;
    observed[index] = 0;
}

__global__ void shift_grid_kernel(
    const float* old_log_odds,
    const unsigned int* old_observed,
    float* new_log_odds,
    unsigned int* new_observed,
    int width,
    int height,
    int depth,
    int shift_x,
    int shift_y)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int cells = width * height * depth;
    if (index >= cells) return;
    const int x = index % width;
    const int yz = index / width;
    const int y = yz % height;
    const int z = yz / height;
    const int old_x = x + shift_x;
    const int old_y = y + shift_y;
    if (old_x < 0 || old_x >= width || old_y < 0 || old_y >= height) {
        new_log_odds[index] = 0.0f;
        new_observed[index] = 0;
        return;
    }
    const int old_index = (z * height + old_y) * width + old_x;
    new_log_odds[index] = old_log_odds[old_index];
    new_observed[index] = old_observed[old_index];
}

__device__ bool inside(int x, int y, int z, int width, int height, int depth)
{
    return x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth;
}

__global__ void raycast_kernel(
    const float* points,
    int point_count,
    float sensor_x,
    float sensor_y,
    float sensor_z,
    float origin_x,
    float origin_y,
    float origin_z,
    float resolution,
    int width,
    int height,
    int depth,
    float min_range,
    float max_range,
    float occupied_delta,
    float free_delta,
    float log_min,
    float log_max,
    float* log_odds,
    unsigned int* observed,
    unsigned int* integrated_count)
{
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= point_count) return;

    float hit_x = points[ray * 3];
    float hit_y = points[ray * 3 + 1];
    float hit_z = points[ray * 3 + 2];
    float dx = hit_x - sensor_x;
    float dy = hit_y - sensor_y;
    float dz = hit_z - sensor_z;
    float distance = sqrtf(dx * dx + dy * dy + dz * dz);
    if (!(distance >= min_range) || !isfinite(distance)) return;
    const bool has_hit = distance <= max_range;
    if (!has_hit) {
        const float scale = max_range / distance;
        hit_x = sensor_x + dx * scale;
        hit_y = sensor_y + dy * scale;
        hit_z = sensor_z + dz * scale;
        dx *= scale;
        dy *= scale;
        dz *= scale;
        distance = max_range;
    }

    const float inv_distance = 1.0f / distance;
    const float ux = dx * inv_distance;
    const float uy = dy * inv_distance;
    const float uz = dz * inv_distance;
    const float grid_x = (sensor_x - origin_x) / resolution;
    const float grid_y = (sensor_y - origin_y) / resolution;
    const float grid_z = (sensor_z - origin_z) / resolution;
    int x = static_cast<int>(floorf(grid_x));
    int y = static_cast<int>(floorf(grid_y));
    int z = static_cast<int>(floorf(grid_z));
    if (!inside(x, y, z, width, height, depth)) return;
    atomicAdd(integrated_count, 1u);

    const int step_x = ux >= 0.0f ? 1 : -1;
    const int step_y = uy >= 0.0f ? 1 : -1;
    const int step_z = uz >= 0.0f ? 1 : -1;
    const float inv_x = fabsf(ux) > 1e-8f ? 1.0f / fabsf(ux) : 1.0e30f;
    const float inv_y = fabsf(uy) > 1e-8f ? 1.0f / fabsf(uy) : 1.0e30f;
    const float inv_z = fabsf(uz) > 1e-8f ? 1.0f / fabsf(uz) : 1.0e30f;
    float next_x = ux >= 0.0f ?
        (static_cast<float>(x + 1) - grid_x) * resolution * inv_x :
        (grid_x - static_cast<float>(x)) * resolution * inv_x;
    float next_y = uy >= 0.0f ?
        (static_cast<float>(y + 1) - grid_y) * resolution * inv_y :
        (grid_y - static_cast<float>(y)) * resolution * inv_y;
    float next_z = uz >= 0.0f ?
        (static_cast<float>(z + 1) - grid_z) * resolution * inv_z :
        (grid_z - static_cast<float>(z)) * resolution * inv_z;
    const float delta_x = resolution * inv_x;
    const float delta_y = resolution * inv_y;
    const float delta_z = resolution * inv_z;

    const int hit_x_index = static_cast<int>(floorf((hit_x - origin_x) / resolution));
    const int hit_y_index = static_cast<int>(floorf((hit_y - origin_y) / resolution));
    const int hit_z_index = static_cast<int>(floorf((hit_z - origin_z) / resolution));
    const int max_steps = width + height + depth + 8;
    for (int step = 0; step < max_steps && inside(x, y, z, width, height, depth); ++step) {
        if (has_hit && x == hit_x_index && y == hit_y_index && z == hit_z_index) {
            break;
        }
        const int index = (z * height + y) * width + x;
        atomicExch(&observed[index], 1u);
        atomic_add_clamped(&log_odds[index], free_delta, log_min, log_max);
        const float next = fminf(next_x, fminf(next_y, next_z));
        if (next >= distance) break;
        if (next_x <= next_y && next_x <= next_z) {
            x += step_x;
            next_x += delta_x;
        } else if (next_y <= next_z) {
            y += step_y;
            next_y += delta_y;
        } else {
            z += step_z;
            next_z += delta_z;
        }
    }

    if (has_hit && inside(
        hit_x_index, hit_y_index, hit_z_index, width, height, depth))
    {
        const int hit_index =
            (hit_z_index * height + hit_y_index) * width + hit_x_index;
        atomicExch(&observed[hit_index], 1u);
        atomic_add_clamped(
            &log_odds[hit_index], occupied_delta, log_min, log_max);
    }
}

__global__ void project_occupancy_kernel(
    const float* log_odds,
    const unsigned int* observed,
    signed char* output,
    int width,
    int height,
    int minimum_z,
    int maximum_z,
    float occupied_threshold)
{
    const int index_2d = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_2d >= width * height) return;
    const int x = index_2d % width;
    const int y = index_2d / width;
    bool any_observed = false;
    bool occupied = false;
    for (int z = minimum_z; z < maximum_z; ++z) {
        const int index = (z * height + y) * width + x;
        if (!observed[index]) continue;
        any_observed = true;
        occupied = occupied || log_odds[index] >= occupied_threshold;
    }
    output[index_2d] = !any_observed ? -1 : (occupied ? 100 : 0);
}

__global__ void count_observed_kernel(
    const unsigned int* observed, int cells, unsigned int* count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < cells && observed[index]) atomicAdd(count, 1u);
}

float snapped_origin(float center, int cells, float resolution)
{
    const float center_cell = std::floor(center / resolution);
    return (center_cell - static_cast<float>(cells / 2)) * resolution;
}

}  // namespace

std::string validate_voxel_mapping_config(const VoxelMappingConfig& c)
{
    if (c.width <= 0 || c.height <= 0 || c.depth <= 0)
        return "voxel dimensions must be positive";
    if (c.width > INT_MAX / c.height ||
        c.width * c.height > INT_MAX / c.depth)
        return "voxel cell count exceeds CUDA kernel index range";
    if (!(c.resolution > 0.0f) || !std::isfinite(c.resolution))
        return "resolution must be finite and positive";
    if (!std::isfinite(c.origin_z)) return "origin_z must be finite";
    if (!(c.min_range >= 0.0f) || !(c.max_range > c.min_range) ||
        !std::isfinite(c.min_range) || !std::isfinite(c.max_range))
        return "range limits must be finite and max_range greater than min_range";
    if (!(c.log_odds_occupied > 0.0f) || !(c.log_odds_free < 0.0f) ||
        !std::isfinite(c.log_odds_occupied) || !std::isfinite(c.log_odds_free))
        return "occupied/free log-odds increments must have opposite signs";
    if (!std::isfinite(c.log_odds_min) || !std::isfinite(c.log_odds_max) ||
        !std::isfinite(c.occupied_threshold) ||
        !(c.log_odds_min < c.log_odds_max) ||
        c.occupied_threshold < c.log_odds_min ||
        c.occupied_threshold > c.log_odds_max)
        return "invalid log-odds bounds or occupied threshold";
    const float grid_max_z = c.origin_z + c.depth * c.resolution;
    if (!std::isfinite(c.projection_min_z) ||
        !std::isfinite(c.projection_max_z) ||
        !(c.projection_min_z < c.projection_max_z) ||
        c.projection_max_z <= c.origin_z ||
        c.projection_min_z >= grid_max_z)
        return "projection height band must be finite, ordered, and overlap the grid";
    if (c.rolling_margin_cells < 1 ||
        c.rolling_margin_cells > (std::min(c.width, c.height) - 1) / 2)
        return "rolling_margin_cells must fit inside the XY grid";
    if (c.max_scan_points == 0 ||
        c.max_scan_points > static_cast<std::size_t>(INT_MAX))
        return "max_scan_points must be in the CUDA kernel index range";
    return {};
}

struct VoxelMapperGpu::Impl {
    explicit Impl(const VoxelMappingConfig& value) : config(value)
    {
        const std::string error = validate_voxel_mapping_config(config);
        if (!error.empty()) throw std::invalid_argument(error);
        cell_count = config.width * config.height * config.depth;
        try {
            VOXEL_CUDA_CHECK(cudaMalloc(&points, config.max_scan_points * 3 * sizeof(float)));
            VOXEL_CUDA_CHECK(cudaMalloc(&log_odds, cell_count * sizeof(float)));
            VOXEL_CUDA_CHECK(cudaMalloc(&observed, cell_count * sizeof(unsigned int)));
            VOXEL_CUDA_CHECK(cudaMalloc(&scratch_log_odds, cell_count * sizeof(float)));
            VOXEL_CUDA_CHECK(cudaMalloc(&scratch_observed, cell_count * sizeof(unsigned int)));
            VOXEL_CUDA_CHECK(cudaMalloc(
                &projection, config.width * config.height * sizeof(signed char)));
            VOXEL_CUDA_CHECK(cudaMalloc(&observed_count, sizeof(unsigned int)));
            VOXEL_CUDA_CHECK(cudaMalloc(&integrated_count, sizeof(unsigned int)));
            VOXEL_CUDA_CHECK(cudaEventCreate(&event_start));
            VOXEL_CUDA_CHECK(cudaEventCreate(&event_stop));
        } catch (...) {
            release();
            throw;
        }
        reset(0.0f, 0.0f);
    }

    ~Impl() { release(); }

    void release() noexcept
    {
        if (event_start) cudaEventDestroy(event_start);
        if (event_stop) cudaEventDestroy(event_stop);
        cudaFree(points);
        cudaFree(log_odds);
        cudaFree(observed);
        cudaFree(scratch_log_odds);
        cudaFree(scratch_observed);
        cudaFree(projection);
        cudaFree(observed_count);
        cudaFree(integrated_count);
        event_start = event_stop = nullptr;
        points = log_odds = scratch_log_odds = nullptr;
        observed = scratch_observed = nullptr;
        projection = nullptr;
        observed_count = nullptr;
        integrated_count = nullptr;
    }

    void reset(float center_x, float center_y)
    {
        if (!std::isfinite(center_x) || !std::isfinite(center_y))
            throw std::invalid_argument("grid center must be finite");
        origin_x = snapped_origin(center_x, config.width, config.resolution);
        origin_y = snapped_origin(center_y, config.height, config.resolution);
        const int blocks = (cell_count + 255) / 256;
        clear_grid_kernel<<<blocks, 256>>>(log_odds, observed, cell_count);
        VOXEL_CUDA_CHECK(cudaGetLastError());
        VOXEL_CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::pair<int, int> desired_shift(float sensor_x, float sensor_y) const
    {
        const int x = static_cast<int>(std::floor((sensor_x - origin_x) / config.resolution));
        const int y = static_cast<int>(std::floor((sensor_y - origin_y) / config.resolution));
        if (x >= config.rolling_margin_cells &&
            x < config.width - config.rolling_margin_cells &&
            y >= config.rolling_margin_cells &&
            y < config.height - config.rolling_margin_cells)
            return {0, 0};
        const float new_origin_x =
            snapped_origin(sensor_x, config.width, config.resolution);
        const float new_origin_y =
            snapped_origin(sensor_y, config.height, config.resolution);
        return {
            static_cast<int>(std::lround((new_origin_x - origin_x) / config.resolution)),
            static_cast<int>(std::lround((new_origin_y - origin_y) / config.resolution))};
    }

    float shift(int shift_x, int shift_y)
    {
        if (shift_x == 0 && shift_y == 0) return 0.0f;
        VOXEL_CUDA_CHECK(cudaEventRecord(event_start));
        const int blocks = (cell_count + 255) / 256;
        shift_grid_kernel<<<blocks, 256>>>(
            log_odds, observed, scratch_log_odds, scratch_observed,
            config.width, config.height, config.depth, shift_x, shift_y);
        VOXEL_CUDA_CHECK(cudaGetLastError());
        std::swap(log_odds, scratch_log_odds);
        std::swap(observed, scratch_observed);
        origin_x += shift_x * config.resolution;
        origin_y += shift_y * config.resolution;
        VOXEL_CUDA_CHECK(cudaEventRecord(event_stop));
        VOXEL_CUDA_CHECK(cudaEventSynchronize(event_stop));
        float elapsed = 0.0f;
        VOXEL_CUDA_CHECK(cudaEventElapsedTime(&elapsed, event_start, event_stop));
        return elapsed;
    }

    VoxelMappingStats integrate(
        const float* xyz, std::size_t count, const float sensor_origin[3])
    {
        if (!xyz || !sensor_origin)
            throw std::invalid_argument("scan and sensor origin pointers must not be null");
        if (count == 0) throw std::invalid_argument("scan must contain points");
        if (count > config.max_scan_points)
            throw std::length_error("scan exceeds configured max_scan_points");
        for (int axis = 0; axis < 3; ++axis)
            if (!std::isfinite(sensor_origin[axis]))
                throw std::invalid_argument("sensor origin must be finite");
        for (std::size_t i = 0; i < count * 3; ++i)
            if (!std::isfinite(xyz[i]))
                throw std::invalid_argument("scan contains non-finite coordinates");

        VoxelMappingStats stats;
        stats.input_points = count;
        const auto shift_cells = desired_shift(sensor_origin[0], sensor_origin[1]);
        stats.shift_x_cells = shift_cells.first;
        stats.shift_y_cells = shift_cells.second;
        stats.grid_shifted = shift_cells.first != 0 || shift_cells.second != 0;
        stats.shift_ms = shift(shift_cells.first, shift_cells.second);

        VOXEL_CUDA_CHECK(cudaMemcpy(
            points, xyz, count * 3 * sizeof(float), cudaMemcpyHostToDevice));
        unsigned int zero = 0;
        VOXEL_CUDA_CHECK(cudaMemcpy(
            integrated_count, &zero, sizeof(zero), cudaMemcpyHostToDevice));
        VOXEL_CUDA_CHECK(cudaEventRecord(event_start));
        raycast_kernel<<<(static_cast<int>(count) + 255) / 256, 256>>>(
            points, static_cast<int>(count),
            sensor_origin[0], sensor_origin[1], sensor_origin[2],
            origin_x, origin_y, config.origin_z, config.resolution,
            config.width, config.height, config.depth,
            config.min_range, config.max_range,
            config.log_odds_occupied, config.log_odds_free,
            config.log_odds_min, config.log_odds_max,
            log_odds, observed, integrated_count);
        VOXEL_CUDA_CHECK(cudaGetLastError());
        VOXEL_CUDA_CHECK(cudaEventRecord(event_stop));
        VOXEL_CUDA_CHECK(cudaEventSynchronize(event_stop));
        VOXEL_CUDA_CHECK(cudaEventElapsedTime(
            &stats.raycast_ms, event_start, event_stop));
        VOXEL_CUDA_CHECK(cudaMemcpy(
            &zero, integrated_count, sizeof(zero), cudaMemcpyDeviceToHost));
        stats.integrated_rays = zero;

        zero = 0;
        VOXEL_CUDA_CHECK(cudaMemcpy(
            observed_count, &zero, sizeof(zero), cudaMemcpyHostToDevice));
        count_observed_kernel<<<(cell_count + 255) / 256, 256>>>(
            observed, cell_count, observed_count);
        VOXEL_CUDA_CHECK(cudaGetLastError());
        VOXEL_CUDA_CHECK(cudaMemcpy(
            &zero, observed_count, sizeof(zero), cudaMemcpyDeviceToHost));
        stats.observed_voxels = zero;
        return stats;
    }

    VoxelGridInfo info() const noexcept
    {
        VoxelGridInfo value;
        value.width = config.width;
        value.height = config.height;
        value.depth = config.depth;
        value.resolution = config.resolution;
        value.origin_x = origin_x;
        value.origin_y = origin_y;
        value.origin_z = config.origin_z;
        return value;
    }

    VoxelMappingConfig config;
    int cell_count = 0;
    float origin_x = 0.0f;
    float origin_y = 0.0f;
    float* points = nullptr;
    float* log_odds = nullptr;
    float* scratch_log_odds = nullptr;
    unsigned int* observed = nullptr;
    unsigned int* scratch_observed = nullptr;
    signed char* projection = nullptr;
    unsigned int* observed_count = nullptr;
    unsigned int* integrated_count = nullptr;
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;
};

VoxelMapperGpu::VoxelMapperGpu(const VoxelMappingConfig& config)
    : impl_(new Impl(config)) {}
VoxelMapperGpu::~VoxelMapperGpu() = default;
VoxelMapperGpu::VoxelMapperGpu(VoxelMapperGpu&&) noexcept = default;
VoxelMapperGpu& VoxelMapperGpu::operator=(VoxelMapperGpu&&) noexcept = default;
void VoxelMapperGpu::reset(float x, float y) { impl_->reset(x, y); }

VoxelMappingStats VoxelMapperGpu::integrate_scan(
    const float* xyz, std::size_t count, const float origin[3])
{
    return impl_->integrate(xyz, count, origin);
}

VoxelMappingStats VoxelMapperGpu::integrate_scan(
    const std::vector<float>& xyz, const float origin[3])
{
    if (xyz.size() % 3 != 0)
        throw std::invalid_argument("XYZ vector size must be divisible by three");
    return integrate_scan(xyz.data(), xyz.size() / 3, origin);
}

const VoxelMappingConfig& VoxelMapperGpu::config() const noexcept
{
    return impl_->config;
}

VoxelGridInfo VoxelMapperGpu::grid_info() const noexcept
{
    return impl_->info();
}

OccupancyProjection VoxelMapperGpu::occupancy_projection()
{
    OccupancyProjection output;
    output.grid = impl_->info();
    output.data.resize(
        static_cast<std::size_t>(impl_->config.width) * impl_->config.height);
    VOXEL_CUDA_CHECK(cudaEventRecord(impl_->event_start));
    const int cells_2d = impl_->config.width * impl_->config.height;
    const int minimum_z = std::max(
        0,
        static_cast<int>(std::floor(
            (impl_->config.projection_min_z - impl_->config.origin_z) /
            impl_->config.resolution)));
    const int maximum_z = std::min(
        impl_->config.depth,
        static_cast<int>(std::ceil(
            (impl_->config.projection_max_z - impl_->config.origin_z) /
            impl_->config.resolution)));
    project_occupancy_kernel<<<(cells_2d + 255) / 256, 256>>>(
        impl_->log_odds, impl_->observed, impl_->projection,
        impl_->config.width, impl_->config.height, minimum_z, maximum_z,
        impl_->config.occupied_threshold);
    VOXEL_CUDA_CHECK(cudaGetLastError());
    VOXEL_CUDA_CHECK(cudaEventRecord(impl_->event_stop));
    VOXEL_CUDA_CHECK(cudaEventSynchronize(impl_->event_stop));
    VOXEL_CUDA_CHECK(cudaEventElapsedTime(
        &output.gpu_ms, impl_->event_start, impl_->event_stop));
    VOXEL_CUDA_CHECK(cudaMemcpy(
        output.data.data(), impl_->projection,
        output.data.size() * sizeof(std::int8_t), cudaMemcpyDeviceToHost));
    return output;
}

VoxelGridSnapshot VoxelMapperGpu::snapshot() const
{
    VoxelGridSnapshot output;
    output.grid = impl_->info();
    output.log_odds.resize(impl_->cell_count);
    output.observed.resize(impl_->cell_count);
    std::vector<unsigned int> observed_words(impl_->cell_count);
    VOXEL_CUDA_CHECK(cudaMemcpy(
        output.log_odds.data(), impl_->log_odds,
        output.log_odds.size() * sizeof(float), cudaMemcpyDeviceToHost));
    VOXEL_CUDA_CHECK(cudaMemcpy(
        observed_words.data(), impl_->observed,
        observed_words.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    for (std::size_t index = 0; index < observed_words.size(); ++index)
        output.observed[index] = observed_words[index] ? 1u : 0u;
    return output;
}

}  // namespace cudarobotics
