// Attribute production MPPI solve time to path search and costmap upload.
#include "cuda_mppi_controller/mppi_gpu.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using cuda_mppi_controller::MppiGpu;
using cuda_mppi_controller::MppiParams;

double run_case(int K, int path_points, bool with_costmap)
{
  MppiParams params;
  params.batch_size = K;
  MppiGpu planner(params);
  std::vector<float> path;
  for (int i = 0; i < path_points; ++i) {
    path.push_back(1.0f + 8.0f * i / (path_points > 1 ? path_points - 1 : 1));
    path.push_back(5.0f);
  }
  std::vector<unsigned char> costmap(200 * 200, 0);
  constexpr int warmup = 20, iterations = 200;
  auto compute = [&] {
    planner.compute(
      1.0f, 5.0f, 0.0f,
      with_costmap ? costmap.data() : nullptr,
      with_costmap ? 200 : 0, with_costmap ? 200 : 0,
      0.0f, 0.0f, 0.05f,
      path.empty() ? nullptr : path.data(), path_points,
      9.0f, 5.0f, 0.0f, true);
  };
  for (int i = 0; i < warmup; ++i) compute();
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) compute();
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(stop - start).count() / iterations;
}

int main(int argc, char ** argv)
{
  const int K = argc > 1 ? std::atoi(argv[1]) : 65536;
  std::printf("case,path_points,costmap,mean_ms\n");
  for (int points : {0, 32, 81, 256}) {
    std::printf("path_%d,%d,0,%.6f\n", points, points, run_case(K, points, false));
  }
  std::printf("path_81_map,81,1,%.6f\n", run_case(K, 81, true));
}
