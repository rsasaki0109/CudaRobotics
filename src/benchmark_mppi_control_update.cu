// Microbenchmark for the MPPI weighted control update reduction.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

__global__ void legacy_update(const float * perturbed, const float * weights,
                              float * nominal, int K, int controls)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= controls) return;
  float sum = 0.0f;
  for (int k = 0; k < K; ++k) sum += weights[k] * perturbed[k * controls + idx];
  nominal[idx] = sum;
}

__global__ void parallel_update(const float * perturbed, const float * weights,
                                float * nominal, int K, int controls)
{
  constexpr int tile = 32, warps = 8;
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  int idx = blockIdx.x * tile + lane;
  extern __shared__ float partial[];
  float sum = 0.0f;
  if (idx < controls) {
    for (int k = warp; k < K; k += warps) sum += weights[k] * perturbed[k * controls + idx];
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  if (warp == 0 && idx < controls) {
    float total = partial[lane];
#pragma unroll
    for (int other = 1; other < warps; ++other) total += partial[other * tile + lane];
    nominal[idx] = total;
  }
}

template<class Launch>
float measure(Launch launch, int warmup = 20, int iterations = 200)
{
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) launch();
  CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  return milliseconds / iterations;
}

int main()
{
  constexpr int controls = 56 * 3;
  constexpr int threads = 256;
  std::printf("K,legacy_ms,parallel_ms,speedup,max_abs_error\n");
  for (int K : {2048, 8192, 16384, 65536}) {
    size_t samples = static_cast<size_t>(K) * controls;
    std::vector<float> h_perturbed(samples), h_weights(K, 1.0f / K);
    for (size_t i = 0; i < samples; ++i) h_perturbed[i] = std::sin(static_cast<float>(i % 1009) * 0.01f);
    float * d_perturbed, * d_weights, * d_legacy, * d_parallel;
    CUDA_CHECK(cudaMalloc(&d_perturbed, samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_legacy, controls * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_parallel, controls * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_perturbed, h_perturbed.data(), samples * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), K * sizeof(float), cudaMemcpyHostToDevice));
    auto old_launch = [&] { legacy_update<<<(controls + threads - 1) / threads, threads>>>(d_perturbed, d_weights, d_legacy, K, controls); };
    auto new_launch = [&] { parallel_update<<<(controls + 31) / 32, threads, threads * sizeof(float)>>>(d_perturbed, d_weights, d_parallel, K, controls); };
    float old_ms = measure(old_launch), new_ms = measure(new_launch);
    std::vector<float> old_values(controls), new_values(controls);
    CUDA_CHECK(cudaMemcpy(old_values.data(), d_legacy, controls * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(new_values.data(), d_parallel, controls * sizeof(float), cudaMemcpyDeviceToHost));
    float max_error = 0.0f;
    for (int i = 0; i < controls; ++i) max_error = std::max(max_error, std::fabs(old_values[i] - new_values[i]));
    std::printf("%d,%.6f,%.6f,%.2f,%.9g\n", K, old_ms, new_ms, old_ms / new_ms, max_error);
    cudaFree(d_perturbed); cudaFree(d_weights); cudaFree(d_legacy); cudaFree(d_parallel);
  }
}
