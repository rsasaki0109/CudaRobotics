// GPU BCPD non-rigid registration core. Pure C++ interface — no ROS, no CUDA headers.
#ifndef CUDAROBOTICS__BCPD_GPU_HPP_
#define CUDAROBOTICS__BCPD_GPU_HPP_

#include <memory>
#include <vector>

namespace cudarobotics
{

struct BcpdParams
{
  float beta = 1.2f;             // GP motion-coherence length scale
  float lambda = 0.5f;           // trade-off vs data term
  int max_iters = 50;            // EM iterations
};

struct BcpdResult
{
  std::vector<float> deformed_xyz;   // M*3 aligned model control points
  int iterations = 0;
  float final_sigma = 0.f;
  float mean_surface_distance = 0.f;
};

class BcpdGpu
{
public:
  explicit BcpdGpu(const BcpdParams & params = {});
  ~BcpdGpu();

  BcpdGpu(const BcpdGpu &) = delete;
  BcpdGpu & operator=(const BcpdGpu &) = delete;

  BcpdResult registerClouds(
    const float * target_xyz, int num_target,
    const float * source_xyz, int num_source);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cudarobotics

#endif  // CUDAROBOTICS__BCPD_GPU_HPP_
