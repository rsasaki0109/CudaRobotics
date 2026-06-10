// GPU robust Student's-t point-to-plane registration. Pure C++ interface.
#ifndef CUDAROBOTICS__ROBUST_P2PLANE_GPU_HPP_
#define CUDAROBOTICS__ROBUST_P2PLANE_GPU_HPP_

#include <memory>

#include "cudarobotics/filterreg_gpu.hpp"

namespace cudarobotics
{

struct RobustP2PlaneParams
{
  float nu = 3.0f;                 // Student's-t degrees of freedom
  int knn_k = 14;                  // kNN-PCA normals on target (computed once)
  float outlier_fraction = 0.05f; // uniform outlier floor c_out
  int outer_iters_per_sigma = 6;   // EM outer steps per sigma level
  int gn_iters = 3;                // weighted twist GN steps per outer step
};

using RobustP2PlaneResult = FilterRegResult;

class RobustP2PlaneGpu
{
public:
  explicit RobustP2PlaneGpu(const RobustP2PlaneParams & params = {});
  ~RobustP2PlaneGpu();

  RobustP2PlaneGpu(const RobustP2PlaneGpu &) = delete;
  RobustP2PlaneGpu & operator=(const RobustP2PlaneGpu &) = delete;

  RobustP2PlaneResult registerClouds(
    const float * target_xyz, int num_target,
    const float * source_xyz, int num_source,
    const float * init_rotation = nullptr,
    const float * init_translation = nullptr);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cudarobotics

#endif  // CUDAROBOTICS__ROBUST_P2PLANE_GPU_HPP_
