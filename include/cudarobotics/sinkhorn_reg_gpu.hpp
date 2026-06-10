// GPU Sinkhorn-OT registration core. Pure C++ interface — no ROS, no CUDA headers.
#ifndef CUDAROBOTICS__SINKHORN_REG_GPU_HPP_
#define CUDAROBOTICS__SINKHORN_REG_GPU_HPP_

#include <memory>

#include "cudarobotics/filterreg_gpu.hpp"

namespace cudarobotics
{

struct SinkhornRegParams
{
  float rho = 3.0f;              // unbalanced OT marginal relaxation
  int sinkhorn_iters = 60;       // log-domain scaling iterations per outer step
  int outer_iters = 8;           // OT + GN cycles per epsilon level
  int gn_iters = 3;              // weighted twist GN steps per outer cycle
};

using RegTransformResult = FilterRegResult;

class SinkhornRegGpu
{
public:
  explicit SinkhornRegGpu(const SinkhornRegParams & params = {});
  ~SinkhornRegGpu();

  SinkhornRegGpu(const SinkhornRegGpu &) = delete;
  SinkhornRegGpu & operator=(const SinkhornRegGpu &) = delete;

  RegTransformResult registerClouds(
    const float * target_xyz, int num_target,
    const float * source_xyz, int num_source,
    const float * init_rotation = nullptr,
    const float * init_translation = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cudarobotics

#endif  // CUDAROBOTICS__SINKHORN_REG_GPU_HPP_
