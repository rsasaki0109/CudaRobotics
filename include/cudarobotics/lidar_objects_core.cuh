// lidar_objects_core.cuh
//
// The algorithms of the LiDAR object pipeline (header-only; every symbol has
// internal linkage, so several translation units may include it):
//
//   ground segmentation   concentric-zone model: polar bins, a PCA plane per bin
//                         (one warp per bin), an outward check per sector
//   clustering            Euclidean (0.5 m) on a voxel graph, lock-free
//                         union-find; label = the component's smallest point index
//   boxes                 L-shape fitting (90 headings, closeness criterion;
//                         one warp per (cluster, heading)), a learned class
//                         (lidar_box_classifier.h) and a size-prior completion
//   tracking              a static tracker (world-frame voxel accumulation) and a
//                         motion tracker (constant-velocity Kalman filter,
//                         object-frame accumulation, stand-still vs moving test)
//
// The points are in a frame centred at the sensor with its axes aligned to the
// world (z up). The ground model takes the sensor's height above the ground and
// the box class the scan's upper beam angle (defaults SENSOR_H, VERT_MAX: the
// sensor the class was trained on). Each GPU routine has a
// CPU twin with the same arithmetic in the same order (the boxes are
// bit-identical), which the demo src/gpu_ground_segmentation.cu checks.
//
// The public, CUDA-free interface is cudarobotics/lidar_objects_gpu.hpp.

#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cuda_check.cuh"
#include "lidar_box_classifier.h"

namespace cudabot {

// ---- sensor model the pipeline assumes ----
static constexpr float PI_F = 3.14159265f;
static constexpr float VERT_MIN = -24.8f * PI_F / 180.0f, VERT_MAX = 2.0f * PI_F / 180.0f;
static constexpr float SENSOR_H = 1.8f;   // default height above the ground

// ---- polar grid ----
static const int N_RING = 24, N_SECTOR = 72, N_BIN = N_RING * N_SECTOR;
static constexpr float R_MIN = 2.0f, R_MAX = 60.0f;
static constexpr float SEED_TH = 0.25f, DIST_TH = 0.12f;
static constexpr float UPRIGHT_COS = 0.906f;    // cos(25 deg)
static constexpr float STEP_TH = 0.25f, SLOPE_TH = 0.18f;   // height jump allowed between rings

// Ring boundaries grow geometrically so near rings are narrow and far ones wide.
__host__ __device__ static inline float ring_edge(int k) {
    return R_MIN * powf(R_MAX / R_MIN, (float)k / N_RING);
}

__host__ __device__ static inline int bin_of(float x, float y) {
    float r = sqrtf(x * x + y * y);
    if (r < R_MIN || r >= R_MAX) return -1;
    int ring = (int)(N_RING * logf(r / R_MIN) / logf(R_MAX / R_MIN));
    ring = ring < 0 ? 0 : (ring >= N_RING ? N_RING - 1 : ring);
    float a = atan2f(y, x) + PI_F;
    int sec = (int)(a / (2.0f * PI_F) * N_SECTOR);
    sec = sec >= N_SECTOR ? N_SECTOR - 1 : sec;
    return ring * N_SECTOR + sec;
}

// ---- ground model ----
struct BinPlane { float nx, ny, nz, d; int ok; float zc; };   // n . p + d = 0, height at bin centre

__host__ __device__ static inline void smallest_eigvec3(const float* C, float* v) {
    // Jacobi eigenvalue iterations on the symmetric 3x3 C; returns the eigenvector
    // of the smallest eigenvalue.
    float a[9], V[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    for (int k = 0; k < 9; ++k) a[k] = C[k];
    for (int sweep = 0; sweep < 8; ++sweep)
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) {
                float apq = a[p * 3 + q];
                if (fabsf(apq) < 1e-12f) continue;
                float theta = 0.5f * atan2f(2.0f * apq, a[q * 3 + q] - a[p * 3 + p]);
                float c = cosf(theta), s = sinf(theta);
                for (int k = 0; k < 3; ++k) {   // A <- A J
                    float akp = a[k * 3 + p], akq = a[k * 3 + q];
                    a[k * 3 + p] = c * akp - s * akq;
                    a[k * 3 + q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {   // A <- J^T A
                    float apk = a[p * 3 + k], aqk = a[q * 3 + k];
                    a[p * 3 + k] = c * apk - s * aqk;
                    a[q * 3 + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {   // V <- V J
                    float vkp = V[k * 3 + p], vkq = V[k * 3 + q];
                    V[k * 3 + p] = c * vkp - s * vkq;
                    V[k * 3 + q] = s * vkp + c * vkq;
                }
            }
    int m = 0;
    if (a[4] < a[m * 4]) m = 1;
    if (a[8] < a[m * 4]) m = 2;
    v[0] = V[0 * 3 + m]; v[1] = V[1 * 3 + m]; v[2] = V[2 * 3 + m];
}

// Fit the plane of one bin from its points pts[idx[k]] for k in [begin, end).
__host__ __device__ static inline BinPlane fit_bin(const float* pts, const int* idx, int begin, int end,
                                                   float cx, float cy) {
    BinPlane P; P.nx = 0; P.ny = 0; P.nz = 1; P.d = 0; P.ok = 0; P.zc = 0;
    if (end - begin < 5) return P;
    float zmin = 1e9f;
    for (int k = begin; k < end; ++k) zmin = fminf(zmin, pts[idx[k] * 3 + 2]);
    float n[3] = { 0, 0, 1 }, d = 0.0f;
    for (int round = 0; round < 3; ++round) {
        float cnt = 0, m[3] = { 0, 0, 0 };
        for (int k = begin; k < end; ++k) {
            const float* p = &pts[idx[k] * 3];
            bool use = round == 0 ? p[2] < zmin + SEED_TH
                                  : fabsf(n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + d) < DIST_TH;
            if (!use) continue;
            m[0] += p[0]; m[1] += p[1]; m[2] += p[2]; cnt += 1.0f;
        }
        if (cnt < 3.0f) return P;
        m[0] /= cnt; m[1] /= cnt; m[2] /= cnt;
        float C[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int k = begin; k < end; ++k) {
            const float* p = &pts[idx[k] * 3];
            bool use = round == 0 ? p[2] < zmin + SEED_TH
                                  : fabsf(n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + d) < DIST_TH;
            if (!use) continue;
            float a = p[0] - m[0], b = p[1] - m[1], c = p[2] - m[2];
            C[0] += a * a; C[1] += a * b; C[2] += a * c;
            C[4] += b * b; C[5] += b * c; C[8] += c * c;
        }
        C[3] = C[1]; C[6] = C[2]; C[7] = C[5];
        smallest_eigvec3(C, n);
        if (n[2] < 0.0f) { n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2]; }
        d = -(n[0] * m[0] + n[1] * m[1] + n[2] * m[2]);
    }
    P.nx = n[0]; P.ny = n[1]; P.nz = n[2]; P.d = d;
    P.ok = n[2] >= UPRIGHT_COS;
    P.zc = -(n[0] * cx + n[1] * cy + d) / n[2];
    return P;
}

__host__ __device__ static inline void bin_center(int b, float& cx, float& cy) {
    int ring = b / N_SECTOR, sec = b - ring * N_SECTOR;
    float r = 0.5f * (ring_edge(ring) + ring_edge(ring + 1));
    float a = (sec + 0.5f) / N_SECTOR * 2.0f * PI_F - PI_F;
    cx = r * cosf(a); cy = r * sinf(a);
}

// Walk one sector outward: keep a plane only if it continues the last kept one.
__host__ __device__ static inline void check_sector(BinPlane* planes, int sec, float sensor_h) {
    float last_z = -sensor_h, last_r = 0.0f;
    for (int ring = 0; ring < N_RING; ++ring) {
        BinPlane& P = planes[ring * N_SECTOR + sec];
        if (!P.ok) continue;
        float r = 0.5f * (ring_edge(ring) + ring_edge(ring + 1));
        float allow = STEP_TH + SLOPE_TH * (r - last_r);
        if (P.zc > last_z + allow) { P.ok = 0; continue; }
        last_z = P.zc; last_r = r;
    }
}

static __global__ void bin_kernel(const float* pts, const int* gt, int* bin, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bin[i] = gt[i] < 0 ? N_BIN : (bin_of(pts[i * 3 + 0], pts[i * 3 + 1]) < 0 ? N_BIN
                                   : bin_of(pts[i * 3 + 0], pts[i * 3 + 1]));
}

static __global__ void bin_range_kernel(const int* sorted_bin, int* start, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = sorted_bin[i];
    if (i == 0 || sorted_bin[i - 1] != b) {
        for (int k = i == 0 ? 0 : sorted_bin[i - 1] + 1; k <= b; ++k) start[k] = i;
    }
    if (i == n - 1) for (int k = b + 1; k <= N_BIN; ++k) start[k] = n;
}

__device__ static inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// fit_bin with the 32 lanes of a warp sharing a bin's points (lane-strided sums,
// a fixed butterfly reduction, so the result is deterministic). Near bins hold
// hundreds to thousands of points; one thread per bin left those few threads
// doing all the work.
__device__ static inline BinPlane fit_bin_warp(const float* pts, const int* idx, int begin, int end,
                                               float cx, float cy, int lane) {
    BinPlane P; P.nx = 0; P.ny = 0; P.nz = 1; P.d = 0; P.ok = 0; P.zc = 0;
    if (end - begin < 5) return P;
    float zmin = 1e9f;
    for (int k = begin + lane; k < end; k += 32) zmin = fminf(zmin, pts[idx[k] * 3 + 2]);
    for (int o = 16; o > 0; o >>= 1) zmin = fminf(zmin, __shfl_xor_sync(0xffffffffu, zmin, o));
    float n[3] = { 0, 0, 1 }, d = 0.0f;
    for (int round = 0; round < 3; ++round) {
        float cnt = 0, m0 = 0, m1 = 0, m2 = 0;
        for (int k = begin + lane; k < end; k += 32) {
            const float* p = &pts[idx[k] * 3];
            bool use = round == 0 ? p[2] < zmin + SEED_TH
                                  : fabsf(n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + d) < DIST_TH;
            if (!use) continue;
            m0 += p[0]; m1 += p[1]; m2 += p[2]; cnt += 1.0f;
        }
        cnt = warp_sum(cnt); m0 = warp_sum(m0); m1 = warp_sum(m1); m2 = warp_sum(m2);
        if (cnt < 3.0f) return P;
        float m[3] = { m0 / cnt, m1 / cnt, m2 / cnt };
        float C[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int k = begin + lane; k < end; k += 32) {
            const float* p = &pts[idx[k] * 3];
            bool use = round == 0 ? p[2] < zmin + SEED_TH
                                  : fabsf(n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + d) < DIST_TH;
            if (!use) continue;
            float a = p[0] - m[0], b = p[1] - m[1], c = p[2] - m[2];
            C[0] += a * a; C[1] += a * b; C[2] += a * c;
            C[4] += b * b; C[5] += b * c; C[8] += c * c;
        }
        C[0] = warp_sum(C[0]); C[1] = warp_sum(C[1]); C[2] = warp_sum(C[2]);
        C[4] = warp_sum(C[4]); C[5] = warp_sum(C[5]); C[8] = warp_sum(C[8]);
        C[3] = C[1]; C[6] = C[2]; C[7] = C[5];
        smallest_eigvec3(C, n);   // every lane computes the same 3x3 result
        if (n[2] < 0.0f) { n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2]; }
        d = -(n[0] * m[0] + n[1] * m[1] + n[2] * m[2]);
    }
    P.nx = n[0]; P.ny = n[1]; P.nz = n[2]; P.d = d;
    P.ok = n[2] >= UPRIGHT_COS;
    P.zc = -(n[0] * cx + n[1] * cy + d) / n[2];
    return P;
}

// one warp = one bin
static __global__ void fit_kernel(const float* pts, const int* idx, const int* start, BinPlane* planes) {
    int b = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (b >= N_BIN) return;
    float cx, cy;
    bin_center(b, cx, cy);
    BinPlane P = fit_bin_warp(pts, idx, start[b], start[b + 1], cx, cy, lane);
    if (lane == 0) planes[b] = P;
}

static __global__ void check_kernel(BinPlane* planes, float sensor_h) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < N_SECTOR) check_sector(planes, s, sensor_h);
}

static __global__ void label_kernel(const float* pts, const int* bin, const BinPlane* planes, int* lab, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = bin[i];
    if (b >= N_BIN) { lab[i] = 0; return; }
    const BinPlane& P = planes[b];
    const float* p = &pts[i * 3];
    lab[i] = P.ok && fabsf(P.nx * p[0] + P.ny * p[1] + P.nz * p[2] + P.d) < DIST_TH;
}

// ---- clustering ----
static constexpr float CL_EPS = 0.5f;
static const int CL_MIN = 10;              // smaller clusters are dropped

__device__ static inline int lower_bound_key(const int* key, int n, int c) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (key[mid] < c) lo = mid + 1; else hi = mid;
    }
    return lo;
}

__device__ static inline int uf_find(const int* parent, int x) {
    for (int p = parent[x]; p != x; p = parent[x]) x = p;
    return x;
}

// Hook the larger root under the smaller one; retry if another thread got there first.
__device__ static inline void uf_unite(int* parent, int a, int b) {
    for (;;) {
        a = uf_find(parent, a);
        b = uf_find(parent, b);
        if (a == b) return;
        if (a > b) { int t = a; a = b; b = t; }
        int old = atomicCAS(&parent[b], b, a);
        if (old == b) return;
        b = old;
    }
}

// ---- voxel clustering: the same partition with far fewer point tests ----
// Two points in one voxel of side CL_EPS / sqrt(3) are always within CL_EPS, so
// all points of a voxel belong to one cluster and the voxel can be the node.
// Two voxels are connected iff some pair of their points is within CL_EPS; a
// voxel pair is tested point by point and the test stops at the first such
// pair. Voxels of points within CL_EPS are at most 2 apart per axis.
static constexpr float VX_S = CL_EPS * 0.57735027f * 0.999f;
static const int VNX = 420, VNY = 420, VNZ = 44;   // x, y in [-60.6, 60.6], z in [-6.3, 6.3]

__host__ __device__ static inline int voxel_key(const float* p) {
    int ix = (int)floorf((p[0] + 60.6f) / VX_S), iy = (int)floorf((p[1] + 60.6f) / VX_S);
    int iz = (int)floorf((p[2] + 6.3f) / VX_S);
    if (ix < 0 || iy < 0 || iz < 0 || ix >= VNX || iy >= VNY || iz >= VNZ) return -1;
    return (iz * VNY + iy) * VNX + ix;
}

static __global__ void voxel_key_kernel(const float* pts, const int* active, int* key, int* idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int k = active[i] ? voxel_key(&pts[i * 3]) : -1;
    key[i] = k >= 0 ? k : 0x7fffffff;
    idx[i] = i;
}

// flag[k] = 1 where a new voxel starts in the sorted keys
static __global__ void voxel_flag_kernel(const int* key, int* flag, int n_valid) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n_valid) flag[k] = k == 0 || key[k] != key[k - 1];
}

static __global__ void voxel_build_kernel(const int* key, const int* flag, const int* vid_excl, int* vid,
                                   int* vstart, int* vkey, int* parent, int* cmin, int n_valid) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_valid) return;
    int v = vid_excl[k] + flag[k] - 1;   // inclusive id
    vid[k] = v;
    if (flag[k]) { vstart[v] = k; vkey[v] = key[k]; parent[v] = v; cmin[v] = 0x7fffffff; }
}

// The 62 neighbour offsets within 2 voxels per axis that lead to a larger key
// (dz > 0, or dz == 0 and dy > 0, or dz == dy == 0 and dx > 0): each voxel pair once.
static const int N_VOFF = 62;
static __constant__ int c_voff[N_VOFF * 3];

// Tight bounds of each voxel's points: a voxel pair whose bounds are more than
// CL_EPS apart cannot hold a close pair, and is skipped without point tests.
static __global__ void voxel_bounds_kernel(const float* pts, const int* items, const int* vstart, float* vbox, int n_vox) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vox) return;
    float lo[3] = { 1e9f, 1e9f, 1e9f }, hi[3] = { -1e9f, -1e9f, -1e9f };
    for (int a = vstart[v]; a < vstart[v + 1]; ++a)
        for (int c = 0; c < 3; ++c) {
            float x = pts[items[a] * 3 + c];
            lo[c] = fminf(lo[c], x); hi[c] = fmaxf(hi[c], x);
        }
    for (int c = 0; c < 3; ++c) { vbox[v * 6 + c] = lo[c]; vbox[v * 6 + 3 + c] = hi[c]; }
}

// one warp = one (voxel, neighbour offset) pair. Near the sensor a voxel holds
// hundreds of points, and a pair that turns out not to be connected tests every
// point pair; the 32 lanes share those tests and stop as soon as one lane finds
// a pair within CL_EPS. Every decision that ends the warp's work is taken on
// warp-uniform values (lane 0's union-find check is broadcast), so the warp
// exits together.
static __global__ void voxel_unite_kernel(const float* pts, const int* items, const int* vstart, const int* vkey,
                                   const float* vbox, int* parent, int n_vox) {
    int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (w >= n_vox * N_VOFF) return;   // whole warps
    int v = w / N_VOFF, o = w - v * N_VOFF;
    int key = vkey[v];
    int x = key % VNX + c_voff[o * 3 + 0], y = (key / VNX) % VNY + c_voff[o * 3 + 1];
    int z = key / (VNX * VNY) + c_voff[o * 3 + 2];
    if (x < 0 || y < 0 || z < 0 || x >= VNX || y >= VNY || z >= VNZ) return;
    int nk = (z * VNY + y) * VNX + x;
    int u = lower_bound_key(vkey, n_vox, nk);
    if (u >= n_vox || vkey[u] != nk) return;
    float gap2 = 0.0f;
    for (int c = 0; c < 3; ++c) {
        float g = fmaxf(0.0f, fmaxf(vbox[v * 6 + c] - vbox[u * 6 + 3 + c], vbox[u * 6 + c] - vbox[v * 6 + 3 + c]));
        gap2 += g * g;
    }
    if (gap2 > CL_EPS * CL_EPS) return;   // bounds too far apart
    int joined = lane == 0 ? uf_find(parent, v) == uf_find(parent, u) : 0;
    if (__shfl_sync(0xffffffffu, joined, 0)) return;   // already connected
    int a0 = vstart[v], nb = vstart[u + 1] - vstart[u], b0 = vstart[u];
    int n_pair = (vstart[v + 1] - a0) * nb;
    for (int base = 0; base < n_pair; base += 32) {
        int q = base + lane;
        bool close = false;
        if (q < n_pair) {
            int ia = q / nb, ib = q - ia * nb;
            const float* p = &pts[items[a0 + ia] * 3];
            const float* r = &pts[items[b0 + ib] * 3];
            float ex = p[0] - r[0], ey = p[1] - r[1], ez = p[2] - r[2];
            close = ex * ex + ey * ey + ez * ez <= CL_EPS * CL_EPS;
        }
        if (__any_sync(0xffffffffu, close)) {
            if (lane == 0) uf_unite(parent, v, u);
            return;
        }
    }
}

// Each component's label is its smallest point index: the first point of each
// voxel is its smallest (stable sort), and the root collects the minimum.
static __global__ void voxel_root_min_kernel(const int* items, const int* vstart, int* parent, int* cmin, int n_vox) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vox) return;
    int r = uf_find(parent, v);
    atomicMin(&cmin[r], items[vstart[v]]);
}

static __global__ void voxel_label_kernel(const int* items, const int* vid, const int* parent, const int* cmin,
                                   int* label, int n_valid) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_valid) return;
    label[items[k]] = cmin[uf_find(parent, vid[k])];
}

// ---- L-shape box fitting (search-based, closeness criterion) ----
// For heading th the cluster's points are projected on e1 = (cos th, sin th) and
// e2 = (-sin th, cos th). Each point's distance to the nearer edge of the
// bounding rectangle in that frame is clamped below at LS_D0, and the heading's
// score is the sum of the inverse distances: points hugging two perpendicular
// edges (the L a LiDAR sees of a car) score high. Headings repeat every 90 deg,
// so N_TH headings cover [0, 90 deg). On the GPU one warp scores one heading:
// lane l takes points l, l + 32, ... and the 32 partial sums are combined by an
// xor butterfly. The CPU runs the same lanes and the same butterfly, with the
// same heading table, so the two pick the same heading and the same box.
static const int N_TH = 90;
static constexpr float LS_D0 = 0.01f;

// sensor frame; len along the heading; tmax = tan of the highest point's elevation; n points
struct Obb { float cx, cy, len, wid, yaw; int th; float zlo, zhi, tmax; int n; };

__host__ __device__ static inline void ls_project(const float* pts, int i, float c, float s, float& p1, float& p2) {
    float x = pts[i * 3], y = pts[i * 3 + 1];
    p1 = c * x + s * y; p2 = -s * x + c * y;
}

__host__ __device__ static inline float ls_term(float p1, float p2, const float* lohi) {
    float d = fminf(fminf(p1 - lohi[0], lohi[1] - p1), fminf(p2 - lohi[2], lohi[3] - p2));
    return 1.0f / fmaxf(d, LS_D0);
}

// CPU: the GPU warp's lanes and butterfly, run serially.
static float lshape_score_cpu(const float* pts, const int* items, int a0, int a1, float c, float s) {
    float lohi[4] = { 1e9f, -1e9f, 1e9f, -1e9f }, part[32];
    for (int a = a0; a < a1; ++a) {
        float p1, p2;
        ls_project(pts, items[a], c, s, p1, p2);
        lohi[0] = fminf(lohi[0], p1); lohi[1] = fmaxf(lohi[1], p1);
        lohi[2] = fminf(lohi[2], p2); lohi[3] = fmaxf(lohi[3], p2);
    }
    for (int l = 0; l < 32; ++l) {
        part[l] = 0.0f;
        for (int a = a0 + l; a < a1; a += 32) {
            float p1, p2;
            ls_project(pts, items[a], c, s, p1, p2);
            part[l] += ls_term(p1, p2, lohi);
        }
    }
    for (int off = 16; off >= 1; off >>= 1) {
        float nxt[32];
        for (int l = 0; l < 32; ++l) nxt[l] = part[l] + part[l ^ off];
        for (int l = 0; l < 32; ++l) part[l] = nxt[l];
    }
    return part[0];
}

// The bounding rectangle of the points in the frame of heading k.
__host__ __device__ static inline Obb lshape_rect(const float* pts, const int* items, int a0, int a1,
                                                  const float* cs, int k) {
    float c = cs[k * 2], s = cs[k * 2 + 1];
    float lo1 = 1e9f, hi1 = -1e9f, lo2 = 1e9f, hi2 = -1e9f, zlo = 1e9f, zhi = -1e9f, tmax = -1e9f;
    for (int a = a0; a < a1; ++a) {
        float x = pts[items[a] * 3], y = pts[items[a] * 3 + 1], z = pts[items[a] * 3 + 2];
        float p1 = c * x + s * y, p2 = -s * x + c * y;
        lo1 = fminf(lo1, p1); hi1 = fmaxf(hi1, p1); lo2 = fminf(lo2, p2); hi2 = fmaxf(hi2, p2);
        zlo = fminf(zlo, z); zhi = fmaxf(zhi, z);
        float r = sqrtf(x * x + y * y);
        if (r > 1e-3f) tmax = fmaxf(tmax, z / r);
    }
    float m1 = 0.5f * (lo1 + hi1), m2 = 0.5f * (lo2 + hi2);
    Obb B;
    B.zlo = zlo; B.zhi = zhi; B.tmax = tmax; B.n = a1 - a0;
    B.cx = c * m1 - s * m2; B.cy = s * m1 + c * m2;
    B.len = hi1 - lo1; B.wid = hi2 - lo2;
    B.yaw = k * (0.5f * PI_F / N_TH); B.th = k;
    return B;
}

// The point indices of the clustered points come sorted by cluster label (runs).
static __global__ void lshape_key_kernel(const int* label, int* key, int* idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    key[i] = label[i] >= 0 ? label[i] : 0x7fffffff;
    idx[i] = i;
}

// one warp = one (cluster, heading) pair
static __global__ void lshape_score_kernel(const float* pts, const int* items, const int* rkey, const int* start,
                                    const int* cnt, const float* cs, float* score, int n_runs, int min_n) {
    int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (w >= n_runs * N_TH) return;   // whole warps
    int r = w / N_TH, k = w - r * N_TH;
    if (rkey[r] == 0x7fffffff || cnt[r] < min_n) return;
    int a0 = start[r], a1 = a0 + cnt[r];
    float c = cs[k * 2], s = cs[k * 2 + 1];
    float lohi[4] = { 1e9f, -1e9f, 1e9f, -1e9f };
    for (int a = a0 + lane; a < a1; a += 32) {
        float p1, p2;
        ls_project(pts, items[a], c, s, p1, p2);
        lohi[0] = fminf(lohi[0], p1); lohi[1] = fmaxf(lohi[1], p1);
        lohi[2] = fminf(lohi[2], p2); lohi[3] = fmaxf(lohi[3], p2);
    }
    for (int off = 16; off >= 1; off >>= 1) {   // min / max: exact in any order
        lohi[0] = fminf(lohi[0], __shfl_xor_sync(0xffffffffu, lohi[0], off));
        lohi[1] = fmaxf(lohi[1], __shfl_xor_sync(0xffffffffu, lohi[1], off));
        lohi[2] = fminf(lohi[2], __shfl_xor_sync(0xffffffffu, lohi[2], off));
        lohi[3] = fmaxf(lohi[3], __shfl_xor_sync(0xffffffffu, lohi[3], off));
    }
    float part = 0.0f;
    for (int a = a0 + lane; a < a1; a += 32) {
        float p1, p2;
        ls_project(pts, items[a], c, s, p1, p2);
        part += ls_term(p1, p2, lohi);
    }
    for (int off = 16; off >= 1; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
    if (lane == 0) score[w] = part;
}

// one thread = one cluster: best heading (the first on ties), then its rectangle
static __global__ void lshape_select_kernel(const float* pts, const int* items, const int* rkey, const int* start,
                                     const int* cnt, const float* cs, const float* score, Obb* obb, int n_runs,
                                     int min_n) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_runs) return;
    if (rkey[r] == 0x7fffffff || cnt[r] < min_n) { obb[r].th = -1; return; }
    int best = 0;
    for (int k = 1; k < N_TH; ++k) if (score[r * N_TH + k] > score[r * N_TH + best]) best = k;
    obb[r] = lshape_rect(pts, items, start[r], start[r] + cnt[r], cs, best);
}

static void heading_table(std::vector<float>& cs) {
    cs.resize(N_TH * 2);
    for (int k = 0; k < N_TH; ++k) {
        double th = k * (0.5 * 3.14159265358979 / N_TH);
        cs[k * 2] = (float)std::cos(th); cs[k * 2 + 1] = (float)std::sin(th);
    }
}

// GPU L-shape fitting of a batch of point sets (the trackers' refits): the same
// kernels as for the clusters, one warp per (set, heading), so the boxes are
// bit-identical to lshape_fit_cpu's.
struct GpuBoxFitter {
    float *d_pts = nullptr, *d_score = nullptr, *d_cs = nullptr;
    int *d_items = nullptr, *d_rkey = nullptr, *d_start = nullptr, *d_cnt = nullptr;
    Obb* d_obb = nullptr;
    size_t cap_pts = 0, cap_sets = 0;
    GpuBoxFitter() {
        std::vector<float> cs;
        heading_table(cs);
        CUDA_CHECK(cudaMalloc(&d_cs, cs.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_cs, cs.data(), cs.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~GpuBoxFitter() {
        cudaFree(d_pts); cudaFree(d_score); cudaFree(d_cs); cudaFree(d_items); cudaFree(d_rkey);
        cudaFree(d_start); cudaFree(d_cnt); cudaFree(d_obb);
    }
    // sets: xyz points per set; returns one box per set, and the GPU time (upload, kernels, download) in ms
    float run(const std::vector<const std::vector<float>*>& sets, std::vector<Obb>& out) {
        size_t n_sets = sets.size(), n_pts = 0;
        out.resize(n_sets);
        if (!n_sets) return 0.0f;
        std::vector<int> start(n_sets), cnt(n_sets);
        for (size_t k = 0; k < n_sets; ++k) { start[k] = (int)n_pts; cnt[k] = (int)(sets[k]->size() / 3); n_pts += cnt[k]; }
        if (n_pts > cap_pts) {
            cap_pts = n_pts * 2;
            cudaFree(d_pts); cudaFree(d_items);
            CUDA_CHECK(cudaMalloc(&d_pts, cap_pts * 3 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_items, cap_pts * sizeof(int)));
            std::vector<int> iota_items(cap_pts);
            std::iota(iota_items.begin(), iota_items.end(), 0);
            CUDA_CHECK(cudaMemcpy(d_items, iota_items.data(), cap_pts * sizeof(int), cudaMemcpyHostToDevice));
        }
        if (n_sets > cap_sets) {
            cap_sets = n_sets * 2;
            cudaFree(d_rkey); cudaFree(d_start); cudaFree(d_cnt); cudaFree(d_score); cudaFree(d_obb);
            CUDA_CHECK(cudaMalloc(&d_rkey, cap_sets * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_rkey, 0, cap_sets * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_start, cap_sets * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_cnt, cap_sets * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_score, cap_sets * N_TH * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_obb, cap_sets * sizeof(Obb)));
        }
        std::vector<float> flat(n_pts * 3);
        for (size_t k = 0; k < n_sets; ++k) std::copy(sets[k]->begin(), sets[k]->end(), flat.begin() + start[k] * 3);
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaMemcpy(d_pts, flat.data(), flat.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_start, start.data(), n_sets * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cnt, cnt.data(), n_sets * sizeof(int), cudaMemcpyHostToDevice));
        const int B = 256, n = (int)n_sets;
        lshape_score_kernel<<<(n * N_TH * 32 + B - 1) / B, B>>>(d_pts, d_items, d_rkey, d_start, d_cnt, d_cs, d_score,
                                                               n, 1);
        lshape_select_kernel<<<(n + B - 1) / B, B>>>(d_pts, d_items, d_rkey, d_start, d_cnt, d_cs, d_score, d_obb, n, 1);
        CUDA_CHECK(cudaMemcpy(out.data(), d_obb, n_sets * sizeof(Obb), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
        return ms;
    }
};

// GPU L-shape fitting of every cluster of >= CL_MIN points of a device label array.
struct GpuLShape {
    int cap;
    int *d_key, *d_idx, *d_rkey, *d_cnt, *d_start;
    float *d_score, *d_cs;
    Obb* d_obb;
    explicit GpuLShape(int cap_points) : cap(cap_points) {
        for (int** p : { &d_key, &d_idx, &d_rkey, &d_cnt, &d_start })
            CUDA_CHECK(cudaMalloc(p, cap * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_score, (size_t)cap * N_TH * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_obb, cap * sizeof(Obb)));
        std::vector<float> cs;
        heading_table(cs);
        CUDA_CHECK(cudaMalloc(&d_cs, cs.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_cs, cs.data(), cs.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~GpuLShape() {
        for (int* p : { d_key, d_idx, d_rkey, d_cnt, d_start }) cudaFree(p);
        cudaFree(d_score); cudaFree(d_cs); cudaFree(d_obb);
    }
    // n points; keys[r] = cluster label, obb[r] = its box (th = -1 for runs that are not clusters)
    float run(const float* d_pts, const int* d_label, int n, std::vector<int>& keys, std::vector<Obb>& obb) {
        const int B = 256, G = (n + B - 1) / B;
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        lshape_key_kernel<<<G, B>>>(d_label, d_key, d_idx, n);
        thrust::device_ptr<int> key(d_key), idx(d_idx), rkey(d_rkey), cnt(d_cnt);
        thrust::stable_sort_by_key(key, key + n, idx);
        int n_runs = (int)(thrust::reduce_by_key(key, key + n, thrust::constant_iterator<int>(1), rkey, cnt)
                               .first - rkey);
        thrust::exclusive_scan(cnt, cnt + n_runs, thrust::device_ptr<int>(d_start));
        lshape_score_kernel<<<(n_runs * N_TH * 32 + B - 1) / B, B>>>(d_pts, d_idx, d_rkey, d_start, d_cnt, d_cs,
                                                               d_score, n_runs, CL_MIN);
        lshape_select_kernel<<<(n_runs + B - 1) / B, B>>>(d_pts, d_idx, d_rkey, d_start, d_cnt, d_cs, d_score,
                                                          d_obb, n_runs, CL_MIN);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
        keys.resize(n_runs); obb.resize(n_runs);
        CUDA_CHECK(cudaMemcpy(keys.data(), d_rkey, n_runs * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(obb.data(), d_obb, n_runs * sizeof(Obb), cudaMemcpyDeviceToHost));
        return ms;
    }
};

// GPU voxel clustering: same partition and labels as the point-level clusterers.
struct GpuVoxelClusterer {
    int cap;
    int *d_active, *d_key, *d_items, *d_flag, *d_vexcl, *d_vid, *d_vstart, *d_vkey, *d_parent, *d_cmin, *d_label;
    float* d_vbox;
    explicit GpuVoxelClusterer(int cap_points) : cap(cap_points) {
        CUDA_CHECK(cudaMalloc(&d_vbox, (size_t)cap * 6 * sizeof(float)));
        for (int** p : { &d_active, &d_key, &d_items, &d_flag, &d_vexcl, &d_vid, &d_vstart, &d_vkey,
                          &d_parent, &d_cmin, &d_label })
            CUDA_CHECK(cudaMalloc(p, (cap + 1) * sizeof(int)));
        int off[N_VOFF * 3], n = 0;
        for (int dz = -2; dz <= 2; ++dz) for (int dy = -2; dy <= 2; ++dy) for (int dx = -2; dx <= 2; ++dx)
            if (dz > 0 || (dz == 0 && (dy > 0 || (dy == 0 && dx > 0)))) {
                off[n * 3 + 0] = dx; off[n * 3 + 1] = dy; off[n * 3 + 2] = dz; ++n;
            }
        CUDA_CHECK(cudaMemcpyToSymbol(c_voff, off, sizeof(off)));
    }
    ~GpuVoxelClusterer() {
        for (int* p : { d_active, d_key, d_items, d_flag, d_vexcl, d_vid, d_vstart, d_vkey, d_parent, d_cmin, d_label })
            cudaFree(p);
        cudaFree(d_vbox);
    }
    // active.size() points; label = the component's smallest point index, -1 for inactive points
    float run(const float* d_pts, const std::vector<int>& active, std::vector<int>& label, int& n_vox_out) {
        const int n = (int)active.size(), B = 256, G = (n + B - 1) / B;
        int n_valid = 0;
        for (int a : active) n_valid += a;
        CUDA_CHECK(cudaMemcpy(d_active, active.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        voxel_key_kernel<<<G, B>>>(d_pts, d_active, d_key, d_items, n);
        thrust::stable_sort_by_key(thrust::device_ptr<int>(d_key), thrust::device_ptr<int>(d_key) + n,
                                   thrust::device_ptr<int>(d_items));
        int Gv = (n_valid + B - 1) / B;
        voxel_flag_kernel<<<Gv > 0 ? Gv : 1, B>>>(d_key, d_flag, n_valid);
        thrust::exclusive_scan(thrust::device_ptr<int>(d_flag), thrust::device_ptr<int>(d_flag) + n_valid,
                               thrust::device_ptr<int>(d_vexcl));
        voxel_build_kernel<<<Gv > 0 ? Gv : 1, B>>>(d_key, d_flag, d_vexcl, d_vid, d_vstart, d_vkey, d_parent,
                                                  d_cmin, n_valid);
        int n_vox = 0;
        if (n_valid > 0) {
            int last_excl = 0, last_flag = 0;
            CUDA_CHECK(cudaMemcpy(&last_excl, d_vexcl + n_valid - 1, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_flag, d_flag + n_valid - 1, sizeof(int), cudaMemcpyDeviceToHost));
            n_vox = last_excl + last_flag;
        }
        CUDA_CHECK(cudaMemcpy(d_vstart + n_vox, &n_valid, sizeof(int), cudaMemcpyHostToDevice));
        int Gx = (n_vox + B - 1) / B, Gp = (int)(((long long)n_vox * N_VOFF * 32 + B - 1) / B);
        CUDA_CHECK(cudaMemset(d_label, 0xff, n * sizeof(int)));   // -1
        if (n_vox > 0) {
            voxel_bounds_kernel<<<Gx, B>>>(d_pts, d_items, d_vstart, d_vbox, n_vox);
            voxel_unite_kernel<<<Gp, B>>>(d_pts, d_items, d_vstart, d_vkey, d_vbox, d_parent, n_vox);
            voxel_root_min_kernel<<<Gx, B>>>(d_items, d_vstart, d_parent, d_cmin, n_vox);
            voxel_label_kernel<<<Gv, B>>>(d_items, d_vid, d_parent, d_cmin, d_label, n_valid);
        }
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
        label.resize(n);
        CUDA_CHECK(cudaMemcpy(label.data(), d_label, n * sizeof(int), cudaMemcpyDeviceToHost));
        n_vox_out = n_vox;
        return ms;
    }
};

// ---- size-prior completion (host; one step per cluster) ----
// A LiDAR sees only the near faces of an object, so the L-shape box covers the
// visible part. A class size prior fills in the rest: the class comes from the
// cluster's height and footprint (standing in for a classifier). The longer
// observed side is the length, unless neither side exceeds the class width: then
// the sensor sees an end of the object and its length runs away from the sensor,
// along the axis closer to the line of sight. A side shorter than the prior grows
// away from the sensor (the origin of the sensor frame), keeping the edge the
// sensor sees. A side the sensor stands across grows about its centre.
struct SizePrior { const char* name; float hmin, hmax, len, wid, min_wid; };
static const int N_CLS = 2;
static const SizePrior PRIOR[N_CLS] = {
    { "car", 1.0f, 2.0f, 4.5f, 1.8f, 0.0f },
    { "van", 2.0f, 3.2f, 6.0f, 2.0f, 0.8f },   // min_wid keeps thin walls out
};

static int classify_box(const Obb& B) {
    float h = B.zhi - B.zlo, lo = std::max(B.len, B.wid), sh = std::min(B.len, B.wid);
    for (int k = 0; k < N_CLS; ++k) {
        const SizePrior& P = PRIOR[k];
        if (h >= P.hmin && h < P.hmax && lo >= 1.2f && lo <= 1.2f * P.len && sh <= 1.2f * P.wid && sh >= P.min_wid)
            return k;
    }
    return -1;
}

// Learned class: a small MLP (scripts/train_box_classifier.py, trained on
// scenes of seeds 101-200, none of which is evaluated) on six features of the
// L-shape box. The last one says whether the top of the object may be cut off
// by the scan's upper beam: a van taller than the beam reaches looks short.
static const int N_FEAT = 7;
static const char* FEAT_NAME[N_FEAT] = { "long", "short", "height", "log_n", "range", "top_margin_deg",
                                         "sensor_height" };

// vert_max: the scan's upper beam angle (radians); sensor_h: the sensor's height above the ground. The
// last feature lets the class read the others for the sensor's height: from a low sensor a car's roof
// reaches the upper beam as a van's does from a high one.
static void box_features(const Obb& B, float* f, float vert_max = VERT_MAX, float sensor_h = SENSOR_H) {
    f[0] = std::max(B.len, B.wid);
    f[1] = std::min(B.len, B.wid);
    f[2] = B.zhi - B.zlo;
    f[3] = std::log((float)B.n);
    f[4] = std::hypot(B.cx, B.cy);
    f[5] = vert_max * (180.0f / PI_F) - std::atan(B.tmax) * (180.0f / PI_F);   // 0 when the top reaches the upper beam
    f[6] = sensor_h;
}

// One forward pass of a generated MLP (tanh hidden layer) over the first NI features; returns the arg max.
template <int NI, int NH, int NO>
static int mlp_argmax(const float* feat, const float (&mean)[NI], const float (&sd)[NI], const float (&W1)[NH][NI],
                      const float (&B1)[NH], const float (&W2)[NO][NH], const float (&B2)[NO]) {
    float f[NI], h[NH], best = -1e30f;
    for (int i = 0; i < NI; ++i) f[i] = (feat[i] - mean[i]) / sd[i];
    for (int j = 0; j < NH; ++j) {
        float a = B1[j];
        for (int i = 0; i < NI; ++i) a += W1[j][i] * f[i];
        h[j] = std::tanh(a);
    }
    int arg = 0;
    for (int k = 0; k < NO; ++k) {
        float a = B2[k];
        for (int j = 0; j < NH; ++j) a += W2[k][j] * h[j];
        if (a > best) { best = a; arg = k; }
    }
    return arg;
}

// The class: the MLP of lidar_box_classifier.h, trained on random scenes and on
// drives, from sensors 0.8-2.5 m above the ground (docs/gpu_ground_segmentation.md).
static int classify_box_mlp(const Obb& B, float vert_max = VERT_MAX, float sensor_h = SENSOR_H) {
    namespace M = lidar_box_classifier;
    float f[N_FEAT];
    box_features(B, f, vert_max, sensor_h);
    int arg = mlp_argmax(f, M::MEAN, M::STD, M::W1, M::B1, M::W2, M::B2);
    return arg < N_CLS ? arg : -1;   // classes: car, van, none
}

static Obb complete_box(const Obb& B, int cls, float scale, bool end_rule = true) {
    if (cls < 0) return B;
    float c = std::cos(B.yaw), s = std::sin(B.yaw);
    float m[2] = { c * B.cx + s * B.cy, -s * B.cx + c * B.cy }, ext[2] = { B.len, B.wid };
    bool len_first = B.len >= B.wid;
    if (end_rule && std::max(B.len, B.wid) <= 1.2f * PRIOR[cls].wid)   // end view: the length runs along the line of sight
        len_first = std::fabs(c * B.cx + s * B.cy) >= std::fabs(-s * B.cx + c * B.cy);
    float want[2] = { scale * (len_first ? PRIOR[cls].len : PRIOR[cls].wid),
                      scale * (len_first ? PRIOR[cls].wid : PRIOR[cls].len) };
    for (int a = 0; a < 2; ++a) {
        if (ext[a] >= want[a]) continue;
        float lo = m[a] - 0.5f * ext[a], hi = m[a] + 0.5f * ext[a];
        if (lo > 0.0f) hi = lo + want[a];          // sensor on the low side: the low edge is seen
        else if (hi < 0.0f) lo = hi - want[a];     // sensor on the high side
        else { lo = m[a] - 0.5f * want[a]; hi = m[a] + 0.5f * want[a]; }
        m[a] = 0.5f * (lo + hi); ext[a] = want[a];
    }
    Obb R = B;
    R.cx = c * m[0] - s * m[1]; R.cy = s * m[0] + c * m[1];
    R.len = ext[0]; R.wid = ext[1];
    return R;
}

// ---- multi-frame tracking (--sequence) ----
// The objects are static and the sensor pose is known, so clusters are
// associated in the world frame: a cluster the learned class calls a car or a
// van joins the track whose box lies within TRK_GATE of the cluster's box
// (closest pairs first, one cluster per track), or starts a new track. A track keeps the
// world points of its clusters on a TRK_VOX voxel grid, counting the scans each
// voxel was seen in, and refits its L-shape box to the voxels seen in at least
// trk_hits scans (3 by default; fewer while the track is young), so faces seen from earlier
// poses stay in the box while points that a single scan wrongly kept (ground
// left beside an object) do not pile up. Its class is the majority of its
// clusters' learned classes.
static constexpr float TRK_GATE = 1.0f, TRK_VOX = 0.1f;

struct Track {
    std::vector<float> pts;   // world x, y, z of the first point in each voxel
    std::vector<int> hits, last;
    std::unordered_map<long long, int> vox;
    int votes[N_CLS], scans = 0;
    Obb box;                  // world frame
};

static float rect_dist(const Obb& B, float x, float y) {
    float c = std::cos(B.yaw), sn = std::sin(B.yaw), dx = x - B.cx, dy = y - B.cy;
    float u = std::fabs(c * dx + sn * dy) - 0.5f * B.len, v = std::fabs(-sn * dx + c * dy) - 0.5f * B.wid;
    return std::hypot(std::max(u, 0.0f), std::max(v, 0.0f));
}

static void rect_corners(const Obb& B, float* x, float* y) {
    float c = std::cos(B.yaw), sn = std::sin(B.yaw);
    for (int k = 0; k < 4; ++k) {
        float a = (k == 0 || k == 3 ? 0.5f : -0.5f) * B.len, w = (k < 2 ? 0.5f : -0.5f) * B.wid;
        x[k] = B.cx + c * a - sn * w; y[k] = B.cy + sn * a + c * w;
    }
}

// Distance between two rectangles: 0 if they overlap, else the closest vertex-to-rectangle distance.
static float rect_rect_dist(const Obb& A, const Obb& B) {
    float ax[4], ay[4], bx[4], by[4], d = 1e30f;
    rect_corners(A, ax, ay); rect_corners(B, bx, by);
    for (int k = 0; k < 4; ++k) d = std::min(d, std::min(rect_dist(B, ax[k], ay[k]), rect_dist(A, bx[k], by[k])));
    if (d > 0.0f) {   // edges may cross with every vertex outside: separating-axis test
        bool separated = false;
        for (int r = 0; r < 4 && !separated; ++r) {   // the two axes of each rectangle
            float yaw = r < 2 ? A.yaw : B.yaw;
            float ux = (r & 1) ? -std::sin(yaw) : std::cos(yaw), uy = (r & 1) ? std::cos(yaw) : std::sin(yaw);
            float amin = 1e30f, amax = -1e30f, bmin = 1e30f, bmax = -1e30f;
            for (int k = 0; k < 4; ++k) {
                float pa = ax[k] * ux + ay[k] * uy, pb = bx[k] * ux + by[k] * uy;
                amin = std::min(amin, pa); amax = std::max(amax, pa);
                bmin = std::min(bmin, pb); bmax = std::max(bmax, pb);
            }
            separated = amax < bmin || bmax < amin;
        }
        if (!separated) d = 0.0f;
    }
    return d;
}

static Obb lshape_fit_cpu(const float* pts, const int* items, int n, const std::vector<float>& cs) {
    int best = 0;
    float best_s = lshape_score_cpu(pts, items, 0, n, cs[0], cs[1]);
    for (int k = 1; k < N_TH; ++k) {
        float sc = lshape_score_cpu(pts, items, 0, n, cs[k * 2], cs[k * 2 + 1]);
        if (sc > best_s) { best_s = sc; best = k; }
    }
    return lshape_rect(pts, items, 0, n, cs.data(), best);
}

// Add a cluster's points (sensor frame, sensor at (px, py, pz)); `fit` gets the points to refit the box to.
static void track_add(Track& T, const std::vector<float>& pts, const std::vector<int>& items, float px, float py,
                      float pz, int scan, int trk_hits, std::vector<float>& fit) {
    for (int i : items) {
        float x = pts[i * 3] + px, y = pts[i * 3 + 1] + py, z = pts[i * 3 + 2] + pz;
        long long kx = (long long)std::floor(x / TRK_VOX) + 100000, ky = (long long)std::floor(y / TRK_VOX) + 100000;
        long long kz = (long long)std::floor(z / TRK_VOX) + 1000;
        auto it = T.vox.emplace((kx * 200000 + ky) * 2000 + kz, (int)T.hits.size());
        int v = it.first->second;
        if (it.second) {
            T.pts.push_back(x); T.pts.push_back(y); T.pts.push_back(z);
            T.hits.push_back(0); T.last.push_back(-1);
        }
        if (T.last[v] != scan) { T.last[v] = scan; T.hits[v]++; }
    }
    T.scans++;
    int need = std::min(trk_hits, T.scans);
    std::vector<int> idx;
    for (size_t v = 0; v < T.hits.size(); ++v) if (T.hits[v] >= need) idx.push_back((int)v);
    if ((int)idx.size() < CL_MIN) { idx.resize(T.hits.size()); std::iota(idx.begin(), idx.end(), 0); }
    fit.clear();
    for (int v : idx) fit.insert(fit.end(), T.pts.begin() + v * 3, T.pts.begin() + v * 3 + 3);
}

// ---- motion tracking (--moving) ----
// Each track also runs a constant-velocity Kalman filter on (x, y, vx, vy). Its
// measurement is the centre of the cluster's box completed with the size prior
// of the track's majority class (so that a scan classed differently does not
// shift it), which depends less on the viewpoint than the box of the visible
// points. A cluster may join a track whose predicted centre lies within
// MV_GATE of the measurement, or whose predicted box lies within TRK_GATE of the
// cluster's box (the measured centre jumps when the view of a nearby car turns
// from its rear to its side); the pairs go closest first by the box distance,
// then by the centre distance.
//
// The track keeps every scan's points with their time. A refit tries two
// hypotheses: the object stands still, or it moves with the filter's velocity
// (only tried above MV_STATIC). Under each, the points are moved to the current
// time and counted on the voxel grid as the static tracker does; the hypothesis
// with more voxels seen in at least K scans wins (a tie: standing still). The
// filter alone cannot tell: as the sensor passes a parked car, the visible part
// and with it the measured centre drift, which reads as a velocity.
static constexpr float MV_STATIC = 1.0f, MV_SIGMA_A = 2.0f, MV_SIGMA_Z = 0.3f, MV_GATE = 2.0f, SCAN_DT = 0.1f;

struct MotionState {
    double x[4] = { 0, 0, 0, 0 }, P[4][4] = {};
    std::vector<float> pts, t;   // world x, y, z per point (one per voxel per scan), and its time
    std::vector<int> scan;       // and its scan number
    float t_box = 0.0f;
    bool moving = false;         // the hypothesis the last refit chose
    // the stand-still hypothesis's voxel grid, kept up to date as points arrive (moving the points by a
    // zero velocity leaves them where they are, so it equals a recount over all points)
    std::unordered_map<long long, int> svox;
    std::vector<float> sP;
    std::vector<int> shits, slast;
};

static long long trk_voxel_key(float x, float y, float z) {
    long long kx = (long long)std::floor(x / TRK_VOX) + 100000, ky = (long long)std::floor(y / TRK_VOX) + 100000;
    long long kz = (long long)std::floor(z / TRK_VOX) + 1000;
    return (kx * 200000 + ky) * 2000 + kz;
}

// Add a point (world frame, at scan `scan`) to the stand-still voxel grid.
static void motion_add_static(MotionState& M, float x, float y, float z, int scan) {
    auto it = M.svox.emplace(trk_voxel_key(x, y, z), (int)M.shits.size());
    int v = it.first->second;
    if (it.second) { M.sP.push_back(x); M.sP.push_back(y); M.sP.push_back(z); M.shits.push_back(0); M.slast.push_back(-1); }
    if (M.slast[v] != scan) { M.slast[v] = scan; M.shits[v]++; }
}

static void kf_predict(MotionState& M, double dt) {
    double F[4][4] = { { 1, 0, dt, 0 }, { 0, 1, 0, dt }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } }, FP[4][4] = {}, Pn[4][4] = {};
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) FP[i][j] += F[i][k] * M.P[k][j];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) Pn[i][j] += FP[i][k] * F[j][k];
    double q = (double)MV_SIGMA_A * MV_SIGMA_A, d2 = dt * dt, d3 = d2 * dt, d4 = d3 * dt;
    Pn[0][0] += q * d4 / 4; Pn[1][1] += q * d4 / 4; Pn[2][2] += q * d2; Pn[3][3] += q * d2;
    Pn[0][2] += q * d3 / 2; Pn[2][0] += q * d3 / 2; Pn[1][3] += q * d3 / 2; Pn[3][1] += q * d3 / 2;
    M.x[0] += dt * M.x[2]; M.x[1] += dt * M.x[3];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) M.P[i][j] = Pn[i][j];
}

static void kf_update(MotionState& M, double zx, double zy) {
    double r = (double)MV_SIGMA_Z * MV_SIGMA_Z;
    double S[2][2] = { { M.P[0][0] + r, M.P[0][1] }, { M.P[1][0], M.P[1][1] + r } };
    double det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    double Si[2][2] = { { S[1][1] / det, -S[0][1] / det }, { -S[1][0] / det, S[0][0] / det } };
    double K[4][2];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) K[i][j] = M.P[i][0] * Si[0][j] + M.P[i][1] * Si[1][j];
    double y0 = zx - M.x[0], y1 = zy - M.x[1];
    for (int i = 0; i < 4; ++i) M.x[i] += K[i][0] * y0 + K[i][1] * y1;
    double Pn[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) Pn[i][j] = M.P[i][j] - K[i][0] * M.P[0][j] - K[i][1] * M.P[1][j];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) M.P[i][j] = Pn[i][j];
}

// The points of a motion track moved to t_now by (vx, vy), on the voxel grid:
// returns the voxels seen in at least `need` scans (their first points in P).
static int motion_voxels(const MotionState& M, float t_now, float vx, float vy, int need, std::vector<float>& P,
                         std::vector<int>& idx) {
    std::unordered_map<long long, int> vox;
    P.clear(); idx.clear();
    std::vector<int> hits, last;
    for (size_t i = 0; i < M.t.size(); ++i) {
        float dt = t_now - M.t[i];
        float x = M.pts[i * 3] + vx * dt, y = M.pts[i * 3 + 1] + vy * dt, z = M.pts[i * 3 + 2];
        long long kx = (long long)std::floor(x / TRK_VOX) + 100000, ky = (long long)std::floor(y / TRK_VOX) + 100000;
        long long kz = (long long)std::floor(z / TRK_VOX) + 1000;
        auto it = vox.emplace((kx * 200000 + ky) * 2000 + kz, (int)hits.size());
        int v = it.first->second;
        if (it.second) { P.push_back(x); P.push_back(y); P.push_back(z); hits.push_back(0); last.push_back(-1); }
        if (last[v] != M.scan[i]) { last[v] = M.scan[i]; hits[v]++; }
    }
    for (size_t v = 0; v < hits.size(); ++v) if (hits[v] >= need) idx.push_back((int)v);
    int consistent = (int)idx.size();
    if (consistent < CL_MIN) { idx.resize(hits.size()); std::iota(idx.begin(), idx.end(), 0); }
    return consistent;
}

// The points to refit a motion track's box to at time t_now: standing still or moving, whichever is more
// consistent.
static void motion_refit(Track& T, MotionState& M, float t_now, int trk_hits, std::vector<float>& fit) {
    int need = std::min(trk_hits, T.scans);
    std::vector<float> P0, P1;
    std::vector<int> i0, i1;
    // stand still: the incremental grid
    for (size_t v = 0; v < M.shits.size(); ++v) if (M.shits[v] >= need) i0.push_back((int)v);
    int n0 = (int)i0.size();
    if (n0 < CL_MIN) { i0.resize(M.shits.size()); std::iota(i0.begin(), i0.end(), 0); }
    P0 = M.sP;
    M.moving = false;
    if (std::hypot(M.x[2], M.x[3]) > MV_STATIC) {
        int n1 = motion_voxels(M, t_now, (float)M.x[2], (float)M.x[3], need, P1, i1);
        M.moving = n1 > n0;
    }
    const std::vector<float>& P = M.moving ? P1 : P0;
    fit.clear();
    for (int v : (M.moving ? i1 : i0)) fit.insert(fit.end(), P.begin() + v * 3, P.begin() + v * 3 + 3);
    M.t_box = t_now;
}

// One tracker: static (world-frame accumulation) or motion (Kalman filter, object-frame accumulation).
struct Tracker {
    bool motion = false;
    int trk_hits = 3;   // a voxel counts once seen in this many scans
    std::vector<Track> tracks;
    std::vector<MotionState> ms;
    std::vector<int> track_of, last_track;   // per point: the track its cluster joined (-1: none); per object
    int id_switches = 0;
    float t_prev = 0.0f;
    std::vector<int> fit_track;                 // the tracks to refit after update(), and their points
    std::vector<std::vector<float>> fit_pts;

    // The track's box at time t (motion tracks: moved by the estimated velocity).
    Obb box_at(int k, float t) const {
        Obb B = tracks[k].box;
        if (motion && ms[k].moving) {
            B.cx += (float)ms[k].x[2] * (t - ms[k].t_box); B.cy += (float)ms[k].x[3] * (t - ms[k].t_box);
        }
        return B;
    }

    // cand: clusters (indices into ckeys / cobb) the learned class calls cars or vans, with their classes and points.
    void update(float t, float px, float py, float pz, int scan, const std::vector<int>& cand,
                const std::vector<int>& ccls, const std::vector<std::vector<int>>& citems, const std::vector<int>& ckeys,
                const std::vector<Obb>& cobb, const std::vector<float>& pts, const std::vector<float>& cs) {
        track_of.assign(pts.size() / 3, -1);
        fit_track.clear(); fit_pts.clear();
        if (motion) for (MotionState& M : ms) kf_predict(M, t - t_prev);
        t_prev = t;
        std::vector<std::tuple<float, int, int>> pairs;   // distance, candidate, track
        for (size_t k = 0; k < cand.size(); ++k) {
            Obb W = cobb[cand[k]];
            W.cx += px; W.cy += py;
            for (size_t j = 0; j < tracks.size(); ++j) {
                if (motion) {   // predicted centre vs measured centre
                    Obb Z = complete_box(cobb[cand[k]], track_class((int)j, ccls[cand[k]]), 1.0f);
                    float d = (float)std::hypot(Z.cx + px - ms[j].x[0], Z.cy + py - ms[j].x[1]);
                    float dr = rect_rect_dist(box_at((int)j, t), W);
                    if (d < MV_GATE || dr < TRK_GATE) pairs.emplace_back(dr + 0.01f * d, (int)k, (int)j);
                } else {
                    float d = rect_rect_dist(tracks[j].box, W);
                    if (d < TRK_GATE) pairs.emplace_back(d, (int)k, (int)j);
                }
            }
        }
        std::sort(pairs.begin(), pairs.end());
        std::vector<int> cand_track(cand.size(), -1), used(tracks.size(), 0);
        for (const auto& pr : pairs) {
            int k = std::get<1>(pr), j = std::get<2>(pr);
            if (cand_track[k] >= 0 || used[j]) continue;
            cand_track[k] = j; used[j] = 1;
        }
        for (size_t k = 0; k < cand.size(); ++k) {
            int jt = cand_track[k];
            Obb Z = complete_box(cobb[cand[k]], jt >= 0 ? track_class(jt, ccls[cand[k]]) : ccls[cand[k]], 1.0f);
            double zx = Z.cx + px, zy = Z.cy + py;   // the measurement (motion tracks)
            if (cand_track[k] < 0) {
                cand_track[k] = (int)tracks.size();
                tracks.emplace_back();
                for (int c = 0; c < N_CLS; ++c) tracks.back().votes[c] = 0;
                ms.emplace_back();
                MotionState& M = ms.back();
                M.x[0] = zx; M.x[1] = zy;
                M.P[0][0] = M.P[1][1] = (double)MV_SIGMA_Z * MV_SIGMA_Z;
                M.P[2][2] = M.P[3][3] = 100.0;
            } else if (motion) {
                kf_update(ms[cand_track[k]], zx, zy);
            }
            Track& T = tracks[cand_track[k]];
            fit_track.push_back(cand_track[k]);
            fit_pts.emplace_back();
            if (motion) {
                MotionState& M = ms[cand_track[k]];
                std::unordered_set<long long> seen;   // one point per voxel per scan
                for (int i : citems[k]) {
                    float x = pts[i * 3] + px, y = pts[i * 3 + 1] + py, z = pts[i * 3 + 2] + pz;
                    long long kx = (long long)std::floor(x / TRK_VOX) + 100000;
                    long long ky = (long long)std::floor(y / TRK_VOX) + 100000;
                    long long kz = (long long)std::floor(z / TRK_VOX) + 1000;
                    if (!seen.insert((kx * 200000 + ky) * 2000 + kz).second) continue;
                    M.pts.push_back(x); M.pts.push_back(y); M.pts.push_back(z); M.t.push_back(t); M.scan.push_back(scan);
                    motion_add_static(M, x, y, z, scan);
                }
                T.scans++;
                motion_refit(T, M, t, trk_hits, fit_pts.back());
            } else {
                track_add(T, pts, citems[k], px, py, pz, scan, trk_hits, fit_pts.back());
            }
            T.votes[ccls[cand[k]]]++;
            track_of[ckeys[cand[k]]] = cand_track[k];
        }
    }

    // Refit the boxes of the tracks update() touched: on the GPU in one batch (gpu != nullptr) or on the CPU.
    // With verify, the CPU fits too and `same` turns false on any difference. Returns the fit time in ms.
    double fit(GpuBoxFitter* gpu, const std::vector<float>& cs, bool verify, bool& same) {
        std::vector<Obb> boxes(fit_track.size());
        double ms = 0.0;
        if (gpu) {
            std::vector<const std::vector<float>*> sets;
            for (const auto& P : fit_pts) sets.push_back(&P);
            ms = gpu->run(sets, boxes);
        } else {
            auto c0 = std::chrono::high_resolution_clock::now();
            for (size_t j = 0; j < fit_track.size(); ++j) boxes[j] = fit_cpu(j, cs);
            ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - c0).count();
        }
        for (size_t j = 0; j < fit_track.size(); ++j) {
            if (gpu && verify) {
                Obb c = fit_cpu(j, cs);
                if (std::memcmp(&c, &boxes[j], sizeof(Obb)) != 0) same = false;
            }
            tracks[fit_track[j]].box = boxes[j];
        }
        return ms;
    }

    Obb fit_cpu(size_t j, const std::vector<float>& cs) const {
        std::vector<int> idx(fit_pts[j].size() / 3);
        std::iota(idx.begin(), idx.end(), 0);
        return lshape_fit_cpu(fit_pts[j].data(), idx.data(), (int)idx.size(), cs);
    }

    // The majority class of track k counting one more vote for class c.
    int track_class(int k, int c) const {
        int v[N_CLS], tc = 0;
        for (int i = 0; i < N_CLS; ++i) v[i] = tracks[k].votes[i] + (i == c);
        for (int i = 1; i < N_CLS; ++i) if (v[i] > v[tc]) tc = i;
        return tc;
    }

    // Raw and size-prior boxes of track k in the sensor frame at time t.
    void boxes(int k, float t, float px, float py, Obb& raw, Obb& done_box) const {
        const Track& T = tracks[k];
        raw = motion ? box_at(k, t) : T.box;
        raw.cx -= px; raw.cy -= py;
        int tc = 0;
        for (int c = 1; c < N_CLS; ++c) if (T.votes[c] > T.votes[tc]) tc = c;
        done_box = complete_box(raw, tc, 1.0f);
    }

    // Evaluation: object b was seen in track k (counts identity switches).
    void observe(int b, int k) {
        if ((int)last_track.size() <= b) last_track.resize(b + 1, -1);
        if (last_track[b] >= 0 && last_track[b] != k) ++id_switches;
        last_track[b] = k;
    }
};

// ---- GPU ground segmentation of n points (bin, sort, fit, check, label) ----
// valid[i] < 0 marks a point to skip (no return). lab: 1 ground, 0 not.
struct GpuGroundSegmenter {
    int cap;
    int *d_bin, *d_idx, *d_sorted_bin, *d_start, *d_lab;
    BinPlane* d_planes;
    explicit GpuGroundSegmenter(int cap_points) : cap(cap_points) {
        CUDA_CHECK(cudaMalloc(&d_bin, cap * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_idx, cap * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sorted_bin, cap * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_start, (N_BIN + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_lab, cap * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_planes, N_BIN * sizeof(BinPlane)));
    }
    ~GpuGroundSegmenter() {
        cudaFree(d_bin); cudaFree(d_idx); cudaFree(d_sorted_bin); cudaFree(d_start); cudaFree(d_lab);
        cudaFree(d_planes);
    }
    // Launches the work on the default stream; the labels are in d_lab.
    void run(const float* d_pts, const int* d_valid, int n, float sensor_h = SENSOR_H) {
        const int B = 256, G = (n + B - 1) / B;
        bin_kernel<<<G, B>>>(d_pts, d_valid, d_bin, n);
        thrust::sequence(thrust::device_ptr<int>(d_idx), thrust::device_ptr<int>(d_idx) + n);
        CUDA_CHECK(cudaMemcpy(d_sorted_bin, d_bin, n * sizeof(int), cudaMemcpyDeviceToDevice));
        thrust::stable_sort_by_key(thrust::device_ptr<int>(d_sorted_bin), thrust::device_ptr<int>(d_sorted_bin) + n,
                                   thrust::device_ptr<int>(d_idx));
        bin_range_kernel<<<G, B>>>(d_sorted_bin, d_start, n);
        fit_kernel<<<(N_BIN * 32 + 127) / 128, 128>>>(d_pts, d_idx, d_start, d_planes);
        check_kernel<<<1, N_SECTOR>>>(d_planes, sensor_h);
        label_kernel<<<G, B>>>(d_pts, d_bin, d_planes, d_lab, n);
    }
};


// ---- free-space box refinement ----
// A ray proves the space it crossed empty. The free-space grid (FS_CELL cells
// over the sensor's BEV, FS_HALF around it) keeps, per cell, the lowest height
// (mm, sensor frame) at which a ray crossed it, up to FS_MARGIN before the ray's
// end (the surface it hit). A box that reaches the ground and rises to the top
// of its cluster contradicts every cell under it that a ray crossed below that
// top. Around a completed box, FS_NCAND candidates (shifts along and across it,
// headings, lengths, widths) are scored by
//   w_free * (free area inside the box, m^2)
//   + w_out * (mean distance of the cluster's points outside the box, m)
//   + w_size * ((length / length0 - 1)^2 + (width / width0 - 1)^2),
// one GPU thread per (cluster, candidate), and the cheapest wins (the first on
// ties). The CPU runs the same arithmetic in the same order.
static constexpr float FS_CELL = 0.1f, FS_HALF = 60.0f, FS_MARGIN = 0.3f, FS_STEP = 0.05f, FS_TOL_MM = 50.0f;
static const int FS_N = 1200;   // cells per side
static const int FS_NU = 21, FS_NV = 11, FS_NYAW = 9, FS_NL = 5, FS_NW = 3;
static const int FS_NCAND = FS_NU * FS_NV * FS_NYAW * FS_NL * FS_NW;
static constexpr float FS_DU = 0.1f, FS_DV = 0.1f, FS_DYAW = 1.0f * PI_F / 180.0f;

struct FsParams { float w_free, w_out, w_size; };
static const FsParams FS_DEFAULT = { 1.0f, 10.0f, 3.0f };   // chosen on the dev scene and training seeds 101-110

// One ray (sensor at the origin) into the grid: per crossed cell, the lowest height.
__host__ __device__ static inline void fs_ray(const float* p, int* grid) {
    float r = sqrtf(p[0] * p[0] + p[1] * p[1]);
    if (r <= FS_MARGIN) return;
    float end = r - FS_MARGIN;
    for (float t = 0.0f; t < end; t += FS_STEP) {
        float f = t / r;
        float x = p[0] * f, y = p[1] * f, z = p[2] * f;
        int ix = (int)floorf((x + FS_HALF) / FS_CELL), iy = (int)floorf((y + FS_HALF) / FS_CELL);
        if (ix < 0 || iy < 0 || ix >= FS_N || iy >= FS_N) return;
        int zmm = (int)floorf(z * 1000.0f);
#ifdef __CUDA_ARCH__
        atomicMin(&grid[iy * FS_N + ix], zmm);
#else
        int& g = grid[iy * FS_N + ix];
        if (zmm < g) g = zmm;
#endif
    }
}

static __global__ void fs_grid_kernel(const float* pts, const int* valid, int* grid, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && valid[i] >= 0) fs_ray(&pts[i * 3], grid);
}

// The candidate box k around B0; cs: cos / sin of B0's heading and of the FS_NYAW headings.
__host__ __device__ static inline Obb fs_candidate(const Obb& B0, const float* cs, int k) {
    int iw = k % FS_NW; k /= FS_NW;
    int il = k % FS_NL; k /= FS_NL;
    int iy = k % FS_NYAW; k /= FS_NYAW;
    int iv = k % FS_NV; k /= FS_NV;
    int iu = k;
    float du = (iu - FS_NU / 2) * FS_DU, dv = (iv - FS_NV / 2) * FS_DV;
    Obb C = B0;
    C.cx = B0.cx + cs[0] * du - cs[1] * dv;
    C.cy = B0.cy + cs[1] * du + cs[0] * dv;
    C.yaw = B0.yaw + (iy - FS_NYAW / 2) * FS_DYAW;
    C.len = B0.len * (0.9f + 0.05f * il);
    C.wid = B0.wid * (0.9f + 0.1f * iw);
    C.th = iy;   // which heading of the table
    return C;
}

// Whether footprint sample (i, j) of box C (heading cos c, sin s) lies on a cell a ray crossed below ztop.
__host__ __device__ static inline int fs_free_sample(const int* grid, const Obb& C, float c, float s, float ztop, int i,
                                                     int j) {
    float a = -0.5f * C.len + (i + 0.5f) * FS_CELL, b = -0.5f * C.wid + (j + 0.5f) * FS_CELL;
    float x = C.cx + c * a - s * b, y = C.cy + s * a + c * b;
    int ix = (int)floorf((x + FS_HALF) / FS_CELL), iy = (int)floorf((y + FS_HALF) / FS_CELL);
    if (ix < 0 || iy < 0 || ix >= FS_N || iy >= FS_N) return 0;
    return (float)grid[iy * FS_N + ix] < ztop;
}

// The distance of point i outside box C (0 inside).
__host__ __device__ static inline float fs_outside(const float* pts, int i, const Obb& C, float c, float s) {
    float dx = pts[i * 3] - C.cx, dy = pts[i * 3 + 1] - C.cy;
    float u = fabsf(c * dx + s * dy) - 0.5f * C.len, v = fabsf(-s * dx + c * dy) - 0.5f * C.wid;
    u = fmaxf(u, 0.0f); v = fmaxf(v, 0.0f);
    return sqrtf(u * u + v * v);
}

__host__ __device__ static inline float fs_total(const Obb& B0, const Obb& C, int free_cells, float out, int n, FsParams P) {
    float dl = C.len / B0.len - 1.0f, dw = C.wid / B0.wid - 1.0f;
    return P.w_free * free_cells * (FS_CELL * FS_CELL) + P.w_out * out / (float)(n > 0 ? n : 1) +
           P.w_size * (dl * dl + dw * dw);
}

// The cost of candidate C (serial; the GPU kernel splits it over a warp with the same result).
__host__ __device__ static inline float fs_cost(const int* grid, const float* pts, const int* items, int a0, int a1,
                                                const Obb& B0, const Obb& C, const float* cs, FsParams P) {
    float c = cs[2 + C.th * 2], s = cs[3 + C.th * 2];
    float ztop = B0.zhi * 1000.0f - FS_TOL_MM;
    int nl = (int)(C.len / FS_CELL), nw = (int)(C.wid / FS_CELL);
    int free_cells = 0;
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nw; ++j) free_cells += fs_free_sample(grid, C, c, s, ztop, i, j);
    float out = 0.0f;
    for (int a = a0; a < a1; ++a) out += fs_outside(pts, items[a], C, c, s);
    return fs_total(B0, C, free_cells, out, a1 - a0, P);
}

// one thread = one candidate, one block row (blockIdx.y) = one cluster. The
// block stages its cluster's points in shared memory, FS_BLOCK at a time; every
// thread still adds their outside distances in point order, so the cost is bit
// for bit fs_cost's.
static const int FS_BLOCK = 256;
static __global__ void fs_cost_kernel(const int* grid, const float* pts, const int* items, const int* start,
                                      const Obb* B0, const float* cs, FsParams P, float* cost) {
    __shared__ float px[FS_BLOCK], py[FS_BLOCK];
    const int r = blockIdx.y, k = blockIdx.x * FS_BLOCK + threadIdx.x;
    const bool active = k < FS_NCAND;
    const float* tab = &cs[r * (2 + 2 * FS_NYAW)];
    const Obb& b0 = B0[r];
    Obb C = fs_candidate(b0, tab, active ? k : 0);
    float c = tab[2 + C.th * 2], s = tab[3 + C.th * 2];
    int free_cells = 0;
    if (active) {
        float ztop = b0.zhi * 1000.0f - FS_TOL_MM;
        int nl = (int)(C.len / FS_CELL), nw = (int)(C.wid / FS_CELL);
        for (int i = 0; i < nl; ++i)
            for (int j = 0; j < nw; ++j) free_cells += fs_free_sample(grid, C, c, s, ztop, i, j);
    }
    const int a0 = start[r], a1 = start[r + 1];
    float out = 0.0f;
    for (int base = a0; base < a1; base += FS_BLOCK) {
        int a = base + threadIdx.x;
        if (a < a1) { px[threadIdx.x] = pts[items[a] * 3]; py[threadIdx.x] = pts[items[a] * 3 + 1]; }
        __syncthreads();
        int m = a1 - base < FS_BLOCK ? a1 - base : FS_BLOCK;
        if (active)
            for (int t = 0; t < m; ++t) {   // fs_outside on the staged point
                float dx = px[t] - C.cx, dy = py[t] - C.cy;
                float u = fabsf(c * dx + s * dy) - 0.5f * C.len, v = fabsf(-s * dx + c * dy) - 0.5f * C.wid;
                u = fmaxf(u, 0.0f); v = fmaxf(v, 0.0f);
                out += sqrtf(u * u + v * v);
            }
        __syncthreads();
    }
    if (active) cost[(long long)r * FS_NCAND + k] = fs_total(b0, C, free_cells, out, a1 - a0, P);
}

// one block = one cluster: the cheapest candidate, the first on ties (the least
// (cost, index) pair, which is what a serial scan keeping the first minimum finds)
static __global__ void fs_select_kernel(const Obb* B0, const float* cs, const float* cost, Obb* out) {
    __shared__ float bc[FS_BLOCK];
    __shared__ int bk[FS_BLOCK];
    const int r = blockIdx.x;
    const float* q = &cost[(long long)r * FS_NCAND];
    float best_c = 0.0f;
    int best = -1;
    for (int k = threadIdx.x; k < FS_NCAND; k += FS_BLOCK)
        if (best < 0 || q[k] < best_c) { best_c = q[k]; best = k; }
    bc[threadIdx.x] = best_c; bk[threadIdx.x] = best;
    __syncthreads();
    for (int h = FS_BLOCK / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) {
            int o = threadIdx.x + h;
            if (bk[o] >= 0 && (bk[threadIdx.x] < 0 || bc[o] < bc[threadIdx.x] ||
                               (bc[o] == bc[threadIdx.x] && bk[o] < bk[threadIdx.x]))) {
                bc[threadIdx.x] = bc[o]; bk[threadIdx.x] = bk[o];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        Obb C = fs_candidate(B0[r], &cs[r * (2 + 2 * FS_NYAW)], bk[0]);
        C.th = B0[r].th;
        out[r] = C;
    }
}

// The cos / sin table of one box: its heading, then the FS_NYAW candidate headings.
static void fs_heading_table(const Obb& B0, float* cs) {
    cs[0] = std::cos(B0.yaw); cs[1] = std::sin(B0.yaw);
    for (int k = 0; k < FS_NYAW; ++k) {
        float y = B0.yaw + (k - FS_NYAW / 2) * FS_DYAW;
        cs[2 + 2 * k] = std::cos(y); cs[3 + 2 * k] = std::sin(y);
    }
}

// GPU free-space refinement of a batch of completed boxes (sensor frame) of one scan.
struct GpuFreeSpaceRefiner {
    int cap;
    int *d_grid, *d_items = nullptr, *d_start = nullptr;
    float *d_cs = nullptr, *d_cost = nullptr;
    Obb *d_b0 = nullptr, *d_out = nullptr;
    size_t cap_items = 0, cap_boxes = 0;
    explicit GpuFreeSpaceRefiner(int cap_points) : cap(cap_points) {
        CUDA_CHECK(cudaMalloc(&d_grid, (size_t)FS_N * FS_N * sizeof(int)));
    }
    ~GpuFreeSpaceRefiner() {
        cudaFree(d_grid); cudaFree(d_items); cudaFree(d_start); cudaFree(d_cs); cudaFree(d_cost);
        cudaFree(d_b0); cudaFree(d_out);
    }
    // d_pts / d_valid: the scan (n points, valid < 0 = no return); boxes and their clusters' point indices
    // (items, start: CSR). Returns the time in ms (grid and search).
    float run(const float* d_pts, const int* d_valid, int n, const std::vector<Obb>& boxes,
              const std::vector<int>& items, const std::vector<int>& start, FsParams P, std::vector<Obb>& out) {
        const int B = 256;
        int nb = (int)boxes.size();
        out = boxes;
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaMemset(d_grid, 0x7f, (size_t)FS_N * FS_N * sizeof(int)));
        fs_grid_kernel<<<(n + B - 1) / B, B>>>(d_pts, d_valid, d_grid, n);
        if (nb > 0) {
            if (items.size() > cap_items) {
                cap_items = items.size() * 2;
                cudaFree(d_items);
                CUDA_CHECK(cudaMalloc(&d_items, cap_items * sizeof(int)));
            }
            if ((size_t)nb > cap_boxes) {
                cap_boxes = nb * 2;
                cudaFree(d_start); cudaFree(d_cs); cudaFree(d_cost); cudaFree(d_b0); cudaFree(d_out);
                CUDA_CHECK(cudaMalloc(&d_start, (cap_boxes + 1) * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_cs, cap_boxes * (2 + 2 * FS_NYAW) * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_cost, cap_boxes * FS_NCAND * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_b0, cap_boxes * sizeof(Obb)));
                CUDA_CHECK(cudaMalloc(&d_out, cap_boxes * sizeof(Obb)));
            }
            std::vector<float> cs(nb * (2 + 2 * FS_NYAW));
            for (int r = 0; r < nb; ++r) fs_heading_table(boxes[r], &cs[r * (2 + 2 * FS_NYAW)]);
            CUDA_CHECK(cudaMemcpy(d_items, items.data(), items.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_start, start.data(), (nb + 1) * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_cs, cs.data(), cs.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_b0, boxes.data(), nb * sizeof(Obb), cudaMemcpyHostToDevice));
            dim3 grid_dim((FS_NCAND + FS_BLOCK - 1) / FS_BLOCK, nb);   // candidates x clusters
            fs_cost_kernel<<<grid_dim, FS_BLOCK>>>(d_grid, d_pts, d_items, d_start, d_b0, d_cs, P, d_cost);
            fs_select_kernel<<<nb, FS_BLOCK>>>(d_b0, d_cs, d_cost, d_out);
            CUDA_CHECK(cudaMemcpy(out.data(), d_out, nb * sizeof(Obb), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
        return ms;
    }
};

// CPU twin: the grid of a scan, and the refinement of one box.
static void fs_grid_cpu(const std::vector<float>& pts, const std::vector<int>& valid, std::vector<int>& grid) {
    grid.assign((size_t)FS_N * FS_N, 0x7f7f7f7f);
    for (size_t i = 0; i < valid.size(); ++i) if (valid[i] >= 0) fs_ray(&pts[i * 3], grid.data());
}

static Obb fs_refine_cpu(const std::vector<int>& grid, const std::vector<float>& pts, const int* items, int a0, int a1,
                         const Obb& B0, FsParams P) {
    float cs[2 + 2 * FS_NYAW];
    fs_heading_table(B0, cs);
    int best = 0;
    float best_c = 0.0f;
    for (int k = 0; k < FS_NCAND; ++k) {
        float c = fs_cost(grid.data(), pts.data(), items, a0, a1, B0, fs_candidate(B0, cs, k), cs, P);
        if (k == 0 || c < best_c) { best_c = c; best = k; }
    }
    Obb C = fs_candidate(B0, cs, best);
    C.th = B0.th;
    return C;
}
}  // namespace cudabot
