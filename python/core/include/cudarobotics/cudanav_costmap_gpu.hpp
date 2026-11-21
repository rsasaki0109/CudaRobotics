#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cudarobotics {

// Row-major 2-D grid geometry shared by the occupancy, distance and cost grids.
struct CostmapGridGeometry {
    int width = 0;
    int height = 0;
    float origin_x = 0.0f;
    float origin_y = 0.0f;
    float resolution = 0.0f;
};

// The CudaNav uchar costmap bands:
//   unknown (< 0)                      -> 255
//   occupied (>= occupied_threshold)   -> 254
//   distance <= inscribed_radius       -> 253
//   distance < inflation_radius        -> clamp(scale * exp(-decay * (d - inscribed)), 1, scale)
//   otherwise                          -> 0
struct CudaNavCostmapParams {
    int occupied_threshold = 50;
    float inscribed_radius = 0.25f;
    float inflation_radius = 1.0f;
    float cost_scale = 252.0f;
    float decay = 3.0f;
};

// CPU references. The GPU builder must match these exactly except that the
// exponential band may differ by one cost step (device vs host expf rounding).
void clear_footprint_cpu(
    std::vector<std::int8_t>& occupancy,
    const CostmapGridGeometry& grid,
    float x,
    float y,
    float radius);

std::vector<unsigned char> make_costmap_cpu(
    const std::vector<std::int8_t>& occupancy,
    const std::vector<float>& distances,
    const CudaNavCostmapParams& params = CudaNavCostmapParams{});

struct CostmapBuildResult {
    // Device pointer to width * height cells owned by the builder; valid until
    // the next build call or the builder's destruction. build() only enqueues
    // the kernel on the default stream, so later work on that stream (such as
    // MppiGpu::computeWithDeviceCostmap) sees the finished costmap.
    const unsigned char* costmap = nullptr;
};

// GPU footprint clearing and costmap construction for device-resident grids,
// so the mapping -> ESDF -> costmap -> MPPI chain never leaves the GPU.
class CostmapBuilderGpu {
public:
    CostmapBuilderGpu(
        int max_width,
        int max_height,
        const CudaNavCostmapParams& params = CudaNavCostmapParams{});
    ~CostmapBuilderGpu();

    CostmapBuilderGpu(CostmapBuilderGpu&&) noexcept;
    CostmapBuilderGpu& operator=(CostmapBuilderGpu&&) noexcept;
    CostmapBuilderGpu(const CostmapBuilderGpu&) = delete;
    CostmapBuilderGpu& operator=(const CostmapBuilderGpu&) = delete;

    // Sets cells whose centre lies within `radius` of (x, y) to free, in place,
    // and returns the number of cells equal to 100 afterwards.
    std::size_t clear_footprint(
        std::int8_t* device_occupancy,
        const CostmapGridGeometry& grid,
        float x,
        float y,
        float radius);

    CostmapBuildResult build(
        const std::int8_t* device_occupancy,
        const float* device_distances,
        int width,
        int height);

    const CudaNavCostmapParams& params() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cudarobotics
