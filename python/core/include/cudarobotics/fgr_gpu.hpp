// GPU Fast Global Registration core. Pure C++ interface — no ROS, no CUDA headers.
#ifndef CUDAROBOTICS__FGR_GPU_HPP_
#define CUDAROBOTICS__FGR_GPU_HPP_

#include <memory>

#include "cudarobotics/filterreg_gpu.hpp"

namespace cudarobotics
{

struct FgrParams
{
  int gn_levels = 24;            // graduated non-convexity levels
  int gn_steps_per_level = 2;    // twist GN steps per level
  float mu_decay = 0.7f;         // Geman-McClure scale decay per level
  float min_mu = 1e-4f;
};

using FgrResult = FilterRegResult;

class FgrGpu
{
public:
  explicit FgrGpu(const FgrParams & params = {});
  ~FgrGpu();

  FgrGpu(const FgrGpu &) = delete;
  FgrGpu & operator=(const FgrGpu &) = delete;

  // Global alignment from identity (FPFH + graduated non-convexity).
  FgrResult registerClouds(
    const float * target_xyz, int num_target,
    const float * source_xyz, int num_source);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cudarobotics

#endif  // CUDAROBOTICS__FGR_GPU_HPP_
