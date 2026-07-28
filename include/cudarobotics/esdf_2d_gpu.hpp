#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cudarobotics {

enum class UnknownSpacePolicy {
    Free,
    Occupied,
};

struct Esdf2DConfig {
    int max_width = 1024;
    int max_height = 1024;
    int occupancy_threshold = 50;
    UnknownSpacePolicy unknown_policy = UnknownSpacePolicy::Occupied;
};

struct Esdf2DResult {
    int width = 0;
    int height = 0;
    float resolution = 0.0f;
    float max_distance = 0.0f;
    std::vector<float> distances;
    std::size_t occupied_cells = 0;
    std::size_t unknown_cells = 0;
    float gpu_ms = 0.0f;
};

std::string validate_esdf_2d_config(const Esdf2DConfig& config);
const char* unknown_space_policy_name(UnknownSpacePolicy policy);

class Esdf2DGpu {
public:
    explicit Esdf2DGpu(const Esdf2DConfig& config = Esdf2DConfig{});
    ~Esdf2DGpu();

    Esdf2DGpu(Esdf2DGpu&&) noexcept;
    Esdf2DGpu& operator=(Esdf2DGpu&&) noexcept;
    Esdf2DGpu(const Esdf2DGpu&) = delete;
    Esdf2DGpu& operator=(const Esdf2DGpu&) = delete;

    Esdf2DResult compute(
        const std::int8_t* occupancy,
        int width,
        int height,
        float resolution,
        float max_distance);
    Esdf2DResult compute(
        const std::vector<std::int8_t>& occupancy,
        int width,
        int height,
        float resolution,
        float max_distance);

    const Esdf2DConfig& config() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Deliberately simple O(cells * occupied) oracle for tests and small maps.
Esdf2DResult compute_esdf_2d_cpu_reference(
    const std::vector<std::int8_t>& occupancy,
    int width,
    int height,
    float resolution,
    float max_distance,
    int occupancy_threshold,
    UnknownSpacePolicy unknown_policy);

}  // namespace cudarobotics
