#include "cudarobotics/cudanav_costmap_gpu.hpp"

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace cudarobotics {
namespace {

#define COSTMAP_CUDA_CHECK(call)                                                \
    do {                                                                        \
        const cudaError_t costmap_cuda_error = (call);                          \
        if (costmap_cuda_error != cudaSuccess) {                                \
            throw std::runtime_error(                                           \
                std::string("CUDA error: ") +                                   \
                cudaGetErrorString(costmap_cuda_error));                        \
        }                                                                       \
    } while (0)

// Shared by the CPU reference and the kernels so both evaluate the same
// expressions; only expf/hypotf implementations differ between host and device.
__host__ __device__ inline bool inside_footprint(
    int row, int column, const CostmapGridGeometry& grid,
    float x, float y, float radius)
{
    const float cell_x = grid.origin_x + (column + 0.5f) * grid.resolution;
    const float cell_y = grid.origin_y + (row + 0.5f) * grid.resolution;
    return hypotf(cell_x - x, cell_y - y) <= radius;
}

__host__ __device__ inline unsigned char cell_cost(
    signed char occupancy, float distance, const CudaNavCostmapParams& params)
{
    if (occupancy < 0) return 255;
    if (occupancy >= params.occupied_threshold) return 254;
    if (distance <= params.inscribed_radius) return 253;
    if (distance < params.inflation_radius) {
        const float value = params.cost_scale *
            expf(-params.decay * (distance - params.inscribed_radius));
        return static_cast<unsigned char>(
            fmaxf(1.0f, fminf(params.cost_scale, value)));
    }
    return 0;
}

__global__ void clear_footprint_kernel(
    signed char* occupancy, CostmapGridGeometry grid,
    float x, float y, float radius, unsigned int* occupied_count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= grid.width * grid.height) return;
    const int row = index / grid.width;
    const int column = index - row * grid.width;
    if (inside_footprint(row, column, grid, x, y, radius)) occupancy[index] = 0;
    if (occupancy[index] == 100) atomicAdd(occupied_count, 1u);
}

__global__ void costmap_kernel(
    const signed char* occupancy, const float* distances, unsigned char* costmap,
    int cells, CudaNavCostmapParams params)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cells) return;
    costmap[index] = cell_cost(occupancy[index], distances[index], params);
}

void validate_geometry(int width, int height, int max_width, int max_height)
{
    if (width <= 0 || height <= 0)
        throw std::invalid_argument("costmap dimensions must be positive");
    if (width > max_width || height > max_height)
        throw std::length_error("costmap dimensions exceed configured capacity");
}

void validate_params(const CudaNavCostmapParams& p)
{
    if (p.occupied_threshold < 0 || p.occupied_threshold > 100)
        throw std::invalid_argument("occupied_threshold must be in [0, 100]");
    if (!std::isfinite(p.inscribed_radius) || p.inscribed_radius < 0.0f ||
        !std::isfinite(p.inflation_radius) ||
        p.inflation_radius < p.inscribed_radius)
        throw std::invalid_argument("costmap radii must be finite and ordered");
    if (!std::isfinite(p.cost_scale) || p.cost_scale < 1.0f || p.cost_scale > 252.0f)
        throw std::invalid_argument("cost_scale must be in [1, 252]");
    if (!std::isfinite(p.decay) || p.decay < 0.0f)
        throw std::invalid_argument("decay must be finite and non-negative");
}

}  // namespace

void clear_footprint_cpu(
    std::vector<std::int8_t>& occupancy,
    const CostmapGridGeometry& grid,
    float x,
    float y,
    float radius)
{
    if (grid.width <= 0 || grid.height <= 0 ||
        occupancy.size() != static_cast<std::size_t>(grid.width) * grid.height)
        throw std::invalid_argument("occupancy size must equal width * height");
    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            if (inside_footprint(row, column, grid, x, y, radius))
                occupancy[static_cast<std::size_t>(row) * grid.width + column] = 0;
        }
    }
}

std::vector<unsigned char> make_costmap_cpu(
    const std::vector<std::int8_t>& occupancy,
    const std::vector<float>& distances,
    const CudaNavCostmapParams& params)
{
    validate_params(params);
    if (occupancy.size() != distances.size())
        throw std::invalid_argument("occupancy and distances must have equal size");
    std::vector<unsigned char> result(occupancy.size(), 0);
    for (std::size_t index = 0; index < result.size(); ++index)
        result[index] = cell_cost(occupancy[index], distances[index], params);
    return result;
}

struct CostmapBuilderGpu::Impl {
    Impl(int width, int height, const CudaNavCostmapParams& value)
        : params(value), max_width(width), max_height(height)
    {
        validate_params(params);
        if (max_width <= 0 || max_height <= 0 || max_width > INT_MAX / max_height)
            throw std::invalid_argument("costmap capacity must be positive");
        try {
            COSTMAP_CUDA_CHECK(cudaMalloc(
                &costmap, static_cast<std::size_t>(max_width) * max_height));
            COSTMAP_CUDA_CHECK(cudaMalloc(&occupied_count, sizeof(unsigned int)));
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    void release() noexcept
    {
        cudaFree(costmap);
        cudaFree(occupied_count);
        costmap = nullptr;
        occupied_count = nullptr;
    }

    CudaNavCostmapParams params;
    int max_width = 0;
    int max_height = 0;
    unsigned char* costmap = nullptr;
    unsigned int* occupied_count = nullptr;
};

CostmapBuilderGpu::CostmapBuilderGpu(
    int max_width, int max_height, const CudaNavCostmapParams& params)
    : impl_(new Impl(max_width, max_height, params))
{
}
CostmapBuilderGpu::~CostmapBuilderGpu() = default;
CostmapBuilderGpu::CostmapBuilderGpu(CostmapBuilderGpu&&) noexcept = default;
CostmapBuilderGpu& CostmapBuilderGpu::operator=(CostmapBuilderGpu&&) noexcept = default;

std::size_t CostmapBuilderGpu::clear_footprint(
    std::int8_t* device_occupancy,
    const CostmapGridGeometry& grid,
    float x,
    float y,
    float radius)
{
    if (!device_occupancy) throw std::invalid_argument("occupancy pointer must not be null");
    validate_geometry(grid.width, grid.height, impl_->max_width, impl_->max_height);
    if (!(grid.resolution > 0.0f) || !std::isfinite(grid.resolution) ||
        !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(radius))
        throw std::invalid_argument("footprint geometry must be finite");
    const int cells = grid.width * grid.height;
    COSTMAP_CUDA_CHECK(cudaMemset(impl_->occupied_count, 0, sizeof(unsigned int)));
    clear_footprint_kernel<<<(cells + 255) / 256, 256>>>(
        reinterpret_cast<signed char*>(device_occupancy), grid, x, y, radius,
        impl_->occupied_count);
    COSTMAP_CUDA_CHECK(cudaGetLastError());
    unsigned int occupied = 0;
    COSTMAP_CUDA_CHECK(cudaMemcpy(
        &occupied, impl_->occupied_count, sizeof(occupied), cudaMemcpyDeviceToHost));
    return occupied;
}

CostmapBuildResult CostmapBuilderGpu::build(
    const std::int8_t* device_occupancy,
    const float* device_distances,
    int width,
    int height)
{
    if (!device_occupancy || !device_distances)
        throw std::invalid_argument("grid pointers must not be null");
    validate_geometry(width, height, impl_->max_width, impl_->max_height);
    const int cells = width * height;
    CostmapBuildResult result;
    costmap_kernel<<<(cells + 255) / 256, 256>>>(
        reinterpret_cast<const signed char*>(device_occupancy), device_distances,
        impl_->costmap, cells, impl_->params);
    COSTMAP_CUDA_CHECK(cudaGetLastError());
    result.costmap = impl_->costmap;
    return result;
}

const CudaNavCostmapParams& CostmapBuilderGpu::params() const noexcept
{
    return impl_->params;
}

}  // namespace cudarobotics
