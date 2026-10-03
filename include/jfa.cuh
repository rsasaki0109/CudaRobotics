// jfa.cuh
//
// Jump Flooding Algorithm (Rong & Tan 2006) for Euclidean distance fields on
// 2D and 3D occupancy grids, shared by the ESDF demos. One thread per cell:
// each pass looks at the 8 (2D) or 26 (3D) neighbours at step k and keeps the
// nearest occupied cell seen so far; k halves each pass, so log2(max dim)
// passes converge. Distances are to the nearest occupied cell centre, in
// metres, and 0 inside occupied cells.
//
//   cudabot::jfa2d_distance(d_occ, d_seed_a, d_seed_b, d_dist, W, H, res, max_dist);
//
// Kernels live in an anonymous namespace so several translation units of one
// target can include this header.

#pragma once

#include <algorithm>
#include <cfloat>
#include <utility>

#include <cuda_runtime.h>

namespace cudabot {
namespace {

__global__ void jfa2d_init_kernel(const unsigned char* __restrict__ occ, int* __restrict__ seed, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    seed[idx] = occ[idx] ? idx : -1;
}

__global__ void jfa2d_step_kernel(const int* __restrict__ seed_in, int* __restrict__ seed_out, int W, int H, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    int best = seed_in[idx];
    float best_d2 = FLT_MAX;
    if (best >= 0) {
        int ex = x - best % W, ey = y - best / W;
        best_d2 = static_cast<float>(ex * ex + ey * ey);
    }
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx * k, ny = y + dy * k;
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            int s = seed_in[ny * W + nx];
            if (s < 0) continue;
            int ex = x - s % W, ey = y - s / W;
            float d2 = static_cast<float>(ex * ex + ey * ey);
            if (d2 < best_d2) { best = s; best_d2 = d2; }
        }
    seed_out[idx] = best;
}

// Cells with no seed (no obstacle anywhere) get max_dist.
__global__ void jfa2d_to_dist_kernel(const int* __restrict__ seed, float* __restrict__ dist,
                                     int W, int H, float res, float max_dist) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    int s = seed[idx];
    if (s < 0) { dist[idx] = max_dist; return; }
    int dx = x - s % W, dy = y - s / W;
    dist[idx] = sqrtf(static_cast<float>(dx * dx + dy * dy)) * res;
}

__global__ void jfa3d_init_kernel(const unsigned char* __restrict__ occ, int* __restrict__ seed, int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;
    int idx = (z * H + y) * W + x;
    seed[idx] = occ[idx] ? idx : -1;
}

__global__ void jfa3d_step_kernel(const int* __restrict__ seed_in, int* __restrict__ seed_out,
                                  int W, int H, int D, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;
    int idx = (z * H + y) * W + x;
    int best = seed_in[idx];
    float best_d2 = FLT_MAX;
    if (best >= 0) {
        int ex = x - best % W, ey = y - (best / W) % H, ez = z - best / (W * H);
        best_d2 = static_cast<float>(ex * ex + ey * ey + ez * ez);
    }
    for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = x + dx * k, ny = y + dy * k, nz = z + dz * k;
                if (nx < 0 || nx >= W || ny < 0 || ny >= H || nz < 0 || nz >= D) continue;
                int s = seed_in[(nz * H + ny) * W + nx];
                if (s < 0) continue;
                int ex = x - s % W, ey = y - (s / W) % H, ez = z - s / (W * H);
                float d2 = static_cast<float>(ex * ex + ey * ey + ez * ez);
                if (d2 < best_d2) { best = s; best_d2 = d2; }
            }
    seed_out[idx] = best;
}

// Cells with no seed get max_dist; with clamp, every distance is capped at max_dist.
__global__ void jfa3d_to_dist_kernel(const int* __restrict__ seed, float* __restrict__ dist,
                                     int W, int H, int D, float res, float max_dist, bool clamp) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;
    int idx = (z * H + y) * W + x;
    int s = seed[idx];
    if (s < 0) { dist[idx] = max_dist; return; }
    int dx = x - s % W, dy = y - (s / W) % H, dz = z - s / (W * H);
    float d = sqrtf(static_cast<float>(dx * dx + dy * dy + dz * dz)) * res;
    dist[idx] = clamp ? fminf(max_dist, d) : d;
}

}  // namespace

// Full 2D pipeline: occupancy -> seeds -> log2 passes -> distances. seed_a and
// seed_b are W*H scratch buffers. Asynchronous on `stream`.
// k_start is the first step size (default max(W, H) / 2).
inline void jfa2d_distance(const unsigned char* d_occ, int* d_seed_a, int* d_seed_b, float* d_dist,
                           int W, int H, float res, float max_dist, int k_start = 0,
                           cudaStream_t stream = 0) {
    dim3 blk(16, 16), grd((W + 15) / 16, (H + 15) / 16);
    jfa2d_init_kernel<<<grd, blk, 0, stream>>>(d_occ, d_seed_a, W, H);
    int *in = d_seed_a, *out = d_seed_b;
    for (int k = k_start > 0 ? k_start : std::max(W, H) / 2; k >= 1; k /= 2) {
        jfa2d_step_kernel<<<grd, blk, 0, stream>>>(in, out, W, H, k);
        std::swap(in, out);
    }
    jfa2d_to_dist_kernel<<<grd, blk, 0, stream>>>(in, d_dist, W, H, res, max_dist);
}

// Full 3D pipeline; seed_a and seed_b are W*H*D scratch buffers.
inline void jfa3d_distance(const unsigned char* d_occ, int* d_seed_a, int* d_seed_b, float* d_dist,
                           int W, int H, int D, float res, float max_dist, bool clamp = false,
                           cudaStream_t stream = 0) {
    dim3 blk(8, 8, 4), grd((W + 7) / 8, (H + 7) / 8, (D + 3) / 4);
    jfa3d_init_kernel<<<grd, blk, 0, stream>>>(d_occ, d_seed_a, W, H, D);
    int *in = d_seed_a, *out = d_seed_b;
    for (int k = std::max(std::max(W, H), D) / 2; k >= 1; k /= 2) {
        jfa3d_step_kernel<<<grd, blk, 0, stream>>>(in, out, W, H, D, k);
        std::swap(in, out);
    }
    jfa3d_to_dist_kernel<<<grd, blk, 0, stream>>>(in, d_dist, W, H, D, res, max_dist, clamp);
}

}  // namespace cudabot
