// cuda_blas.cuh
//
// Minimal CUDA BLAS-like primitives used by cudabot's GPU GN/PCG solvers
// and any kernel sequence that needs zero/copy/axpy/xpay/dot.  Drop these
// in via `using namespace cudabot::blas;` after the include, or qualify
// each call.
//
// Conventions:
//   - All kernels take an `int n` total length and the usual block/grid.
//   - dot_kernel uses block-level shared-memory reduction and atomic-add
//     to `out` (so `out` must be zero-initialised before launch).

#pragma once

#include <cuda_runtime.h>

namespace cudabot { namespace blas {

__global__ inline void zero_kernel(int n, float* arr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = 0.0f;
}

__global__ inline void copy_kernel(int n, const float* src, float* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

// y += a * x
__global__ inline void axpy_kernel(int n, float a, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] += a * x[idx];
}

// y = x + a * y
__global__ inline void xpay_kernel(int n, float a, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = x[idx] + a * y[idx];
}

// out += a . b  (caller zero-inits *out)
__global__ inline void dot_kernel(int n, const float* a, const float* b, float* out) {
    __shared__ float sm[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float v = 0.0f;
    for (int k = idx; k < n; k += gridDim.x * blockDim.x) v += a[k] * b[k];
    sm[tid] = v;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sm[0]);
}

}}  // namespace cudabot::blas
