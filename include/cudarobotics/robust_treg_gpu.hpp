// GPU robust Student's-t point-to-point registration. Pure C++ interface.
#ifndef CUDAROBOTICS__ROBUST_TREG_GPU_HPP_
#define CUDAROBOTICS__ROBUST_TREG_GPU_HPP_

#include <memory>

#include "cudarobotics/filterreg_gpu.hpp"

namespace cudarobotics
{

struct RobustTregParams
{
  float nu = 3.0f;                 // Student's-t degrees of freedom
  float outlier_fraction = 0.05f;    // uniform outlier floor c_out
  int outer_iters_per_sigma = 6;     // EM outer steps per sigma level
  int gn_iters = 3;                  // weighted twist GN steps per outer step
};

using RobustTregResult = FilterRegResult;

class RobustTregGpu
{
public:
  explicit RobustTregGpu(const RobustTregParams & params = {});
  ~RobustTregGpu();

  RobustTregGpu(const RobustTregGpu &) = delete;
  RobustTregGpu & operator=(const RobustTregGpu &) = delete;

  RobustTregResult registerClouds(
    const float * target_xyz, int num_target,
    const float * source_xyz, int num_source,
    const float * init_rotation = nullptr,
    const float * init_translation = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cudarobotics

#endif  // CUDAROBOTICS__ROBUST_TREG_GPU_HPP_
