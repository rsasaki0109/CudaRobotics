// GPU FilterReg core. Pure C++ interface — no ROS, no CUDA headers here.
#ifndef CUDAROBOTICS__FILTERREG_GPU_HPP_
#define CUDAROBOTICS__FILTERREG_GPU_HPP_

#include <memory>

namespace cudarobotics
{

struct FilterRegParams
{
  float voxel_size = 0.07f;          // fixed grid resolution (independent of sigma)
  float bbox_margin = 2.0f;          // [m] padding around the target cloud bbox
  float outlier_fraction = 0.1f;     // c_outlier = fraction * mean_density
  int iters_per_sigma = 8;           // Gauss-Newton steps per sigma level
  float step_tol = 1e-5f;            // early stop on twist norm
};

struct FilterRegResult
{
  float rotation[9] = {             // row-major 3x3 mapping source -> target
    1.f, 0.f, 0.f,
    0.f, 1.f, 0.f,
    0.f, 0.f, 1.f};
  float translation[3] = {0.f, 0.f, 0.f};
  int iterations = 0;
  float final_rmse = 0.f;
};

class FilterRegGpu
{
public:
  explicit FilterRegGpu(const FilterRegParams & params = {});
  ~FilterRegGpu();

  FilterRegGpu(const FilterRegGpu &) = delete;
  FilterRegGpu & operator=(const FilterRegGpu &) = delete;

  // Registers source onto fixed target. Arrays are xyz interleaved (N*3 floats).
  // init_rotation / init_translation may be nullptr (identity).
  FilterRegResult registerClouds(
    const float * target_xyz, int num_target,
    const float * source_xyz, int num_source,
    const float * init_rotation = nullptr,
    const float * init_translation = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cudarobotics

#endif  // CUDAROBOTICS__FILTERREG_GPU_HPP_
