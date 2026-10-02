// Parallel MPPI reductions shared by the demo controllers.
//
//   softmin:  w_k = exp(-(c_k - min c) / lambda) / sum_j exp(-(c_j - min c) / lambda)
//   update:   nominal[j] = sum_k w_k * perturbed[k * controls + j]   (sample-major layout)
//
// Both replace single-thread / one-thread-per-timestep kernels whose cost grew
// linearly with K on a single GPU thread.
#pragma once

#include <cfloat>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

namespace cudabot {

constexpr int kSoftminThreads = 512;
constexpr int kUpdateLanes = 32;   // control elements per block (one per lane)
constexpr int kUpdateWarps = 8;    // warps splitting the K samples

struct MinOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return fminf(a, b); }
};

// Single-block softmin over K costs. Launch with <<<1, BLOCK>>>.
// min_out may be nullptr. If the weight sum is not positive the weights are
// left unnormalized, matching the legacy serial kernel.
template <int BLOCK>
__global__ void softmin_weights_kernel(const float* __restrict__ costs,
                                       float* __restrict__ weights,
                                       float* __restrict__ min_out,
                                       int K, float lambda)
{
    using Reduce = cub::BlockReduce<float, BLOCK>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ float s_min, s_sum;

    float local_min = FLT_MAX;
    for (int k = threadIdx.x; k < K; k += BLOCK) local_min = fminf(local_min, costs[k]);
    float block_min = Reduce(temp).Reduce(local_min, MinOp());
    if (threadIdx.x == 0) s_min = block_min;
    __syncthreads();
    const float min_c = s_min;

    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += BLOCK) {
        float w = expf(-(costs[k] - min_c) / lambda);
        weights[k] = w;
        local_sum += w;
    }
    float block_sum = Reduce(temp).Sum(local_sum);
    if (threadIdx.x == 0) {
        s_sum = block_sum;
        if (min_out) *min_out = min_c;
    }
    __syncthreads();

    const float sum = s_sum;
    if (sum > 0.0f) {
        for (int k = threadIdx.x; k < K; k += BLOCK) weights[k] /= sum;
    }
}

// nominal[j] = sum_k weights[k] * perturbed[k * controls + j].
// Each block owns kUpdateLanes adjacent control elements (coalesced reads) and
// its warps split the K samples. Launch with <<<ceil(controls / 32), 32 * WARPS>>>.
template <int WARPS>
__global__ void weighted_control_update_kernel(const float* __restrict__ perturbed,
                                               const float* __restrict__ weights,
                                               float* __restrict__ nominal,
                                               int K, int controls)
{
    __shared__ float partial[WARPS][kUpdateLanes];
    const int lane = threadIdx.x % kUpdateLanes;
    const int warp = threadIdx.x / kUpdateLanes;
    const int idx = blockIdx.x * kUpdateLanes + lane;

    float sum = 0.0f;
    if (idx < controls) {
        for (int k = warp; k < K; k += WARPS) sum += weights[k] * perturbed[k * controls + idx];
    }
    partial[warp][lane] = sum;
    __syncthreads();

    if (warp == 0 && idx < controls) {
        float total = partial[0][lane];
#pragma unroll
        for (int w = 1; w < WARPS; ++w) total += partial[w][lane];
        nominal[idx] = total;
    }
}

inline void launch_softmin_weights(const float* d_costs, float* d_weights, int K, float lambda,
                                   float* d_min_out = nullptr, cudaStream_t stream = 0)
{
    softmin_weights_kernel<kSoftminThreads><<<1, kSoftminThreads, 0, stream>>>(
        d_costs, d_weights, d_min_out, K, lambda);
}

inline void launch_weighted_control_update(const float* d_perturbed, const float* d_weights,
                                           float* d_nominal, int K, int controls,
                                           cudaStream_t stream = 0)
{
    const int blocks = (controls + kUpdateLanes - 1) / kUpdateLanes;
    weighted_control_update_kernel<kUpdateWarps><<<blocks, kUpdateLanes * kUpdateWarps, 0, stream>>>(
        d_perturbed, d_weights, d_nominal, K, controls);
}

}  // namespace cudabot
