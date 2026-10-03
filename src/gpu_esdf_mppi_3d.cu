/*************************************************************************
    3D ESDF-MPPI
    - Builds a 3D occupancy grid (ground, a wall with one window, pillars,
      a suspended slab) at 128 x 128 x 64 voxels
    - Computes the 3D Euclidean distance field with GPU Jump Flooding
      (26-neighbour propagation, as in comparison_esdf_3d.cu)
    - Computes a cost-to-go field (geodesic distance to the goal) with a
      parallel GPU wavefront on a coarser grid; straight-line distance
      would leave MPPI parked under the wall instead of finding the window
    - MPPI for a 3D double integrator (acceleration control): one thread
      per sampled trajectory, trilinear lookups of both fields
    - The rollout cost is shared host/device code, so the same function
      gives a CPU reference: the demo reports CPU vs GPU rollout time and
      checks that both produce the same costs for the same controls
    Output: gif/gpu_esdf_mppi_3d.gif (top view + side view slices)
 ************************************************************************/

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cuda_check.cuh"
#include "cuda_video.h"
#include "demo_args.h"
#include "display.h"
#include "mppi_reduction.cuh"

// -------------------------------------------------------------------------
// World / ESDF grid
// -------------------------------------------------------------------------
constexpr float WORLD_X = 16.0f;
constexpr float WORLD_Y = 16.0f;
constexpr float WORLD_Z = 8.0f;
constexpr int   NX = 128;
constexpr int   NY = 128;
constexpr int   NZ = 64;
constexpr float RES = WORLD_X / NX;
constexpr float MAX_DIST = 6.0f;

// Cost-to-go (geodesic distance to the goal) on a coarser grid
constexpr int   CX = NX / 2;
constexpr int   CY = NY / 2;
constexpr int   CZ = NZ / 2;
constexpr float CRES = WORLD_X / CX;
constexpr float CTG_BLOCKED = 1.0e3f;   // value in voxels the vehicle cannot enter

// -------------------------------------------------------------------------
// Vehicle (3D double integrator) and MPPI
// -------------------------------------------------------------------------
constexpr int   CTRL_DIM  = 3;
constexpr int   T_HORIZON = 40;
constexpr float DT        = 0.1f;
constexpr float A_MAX     = 4.0f;     // per-axis acceleration limit (m/s^2)
constexpr float V_MAX     = 3.0f;     // speed limit (m/s)
constexpr float SIGMA     = 2.0f;     // acceleration noise (m/s^2)
constexpr float LAMBDA    = 1.0f;
constexpr int   ITERS_PER_STEP = 2;
constexpr float ROBOT_R   = 0.3f;     // collision radius
constexpr float CLEARANCE = 1.0f;     // barrier starts here
constexpr float GOAL_TOL  = 0.4f;

constexpr float W_GOAL    = 1.0f;
constexpr float W_CTRL    = 0.02f;
constexpr float W_OBS     = 3.0f;
constexpr float COLLIDE_PENALTY = 200.0f;
constexpr float W_TERM    = 10.0f;

struct Box { float x0, y0, z0, x1, y1, z1; };

// -------------------------------------------------------------------------
// Scene
// -------------------------------------------------------------------------
static void stamp_box(std::vector<unsigned char>& occ, const Box& b) {
    int x0 = std::max(0, static_cast<int>(std::floor(b.x0 / RES)));
    int y0 = std::max(0, static_cast<int>(std::floor(b.y0 / RES)));
    int z0 = std::max(0, static_cast<int>(std::floor(b.z0 / RES)));
    int x1 = std::min(NX - 1, static_cast<int>(std::ceil(b.x1 / RES)) - 1);
    int y1 = std::min(NY - 1, static_cast<int>(std::ceil(b.y1 / RES)) - 1);
    int z1 = std::min(NZ - 1, static_cast<int>(std::ceil(b.z1 / RES)) - 1);
    for (int z = z0; z <= z1; z++)
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++)
                occ[(static_cast<size_t>(z) * NY + y) * NX + x] = 1u;
}

static std::vector<Box> build_scene(std::vector<unsigned char>& occ) {
    std::vector<Box> boxes = {
        { 0.0f,  0.0f, 0.0f, WORLD_X, WORLD_Y, 0.3f},   // ground
        // wall across y = 8 with a single window at x 9..11, z 2.5..4.5
        { 0.0f,  7.75f, 0.0f,  9.0f,  8.25f, WORLD_Z},
        {11.0f,  7.75f, 0.0f, WORLD_X, 8.25f, WORLD_Z},
        { 9.0f,  7.75f, 0.0f, 11.0f,  8.25f, 2.5f},
        { 9.0f,  7.75f, 4.5f, 11.0f,  8.25f, WORLD_Z},
        // pillars on both sides of the wall
        { 4.0f,  3.0f, 0.0f,  4.8f,  3.8f, 6.0f},
        { 7.0f,  5.0f, 0.0f,  7.8f,  5.8f, 6.0f},
        { 2.5f,  5.5f, 0.0f,  3.3f,  6.3f, 6.0f},
        { 6.0f,  1.5f, 0.0f,  6.8f,  2.3f, 6.0f},
        {12.0f, 10.0f, 0.0f, 12.8f, 10.8f, 6.0f},
        { 9.5f, 12.0f, 0.0f, 10.3f, 12.8f, 6.0f},
        {13.5f, 12.5f, 0.0f, 14.3f, 13.3f, 3.0f},
        // suspended slab before the goal
        {10.5f, 13.5f, 3.2f, 15.5f, 15.5f, 3.6f},
    };
    occ.assign(static_cast<size_t>(NX) * NY * NZ, 0u);
    for (const Box& b : boxes) stamp_box(occ, b);
    return boxes;
}

// -------------------------------------------------------------------------
// GPU 3D Jump Flooding (from comparison_esdf_3d.cu)
// -------------------------------------------------------------------------
__device__ __forceinline__ void unflatten(int idx, int& x, int& y, int& z) {
    x = idx % NX;
    y = (idx / NX) % NY;
    z = idx / (NX * NY);
}

__global__ void jfa3d_init_kernel(const unsigned char* __restrict__ occ, int* __restrict__ seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;
    int idx = (z * NY + y) * NX + x;
    seed[idx] = occ[idx] ? idx : -1;
}

__global__ void jfa3d_step_kernel(const int* __restrict__ seed_in, int* __restrict__ seed_out, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;
    int idx = (z * NY + y) * NX + x;
    int best = seed_in[idx];
    float best_d2 = FLT_MAX;
    if (best >= 0) {
        int bx, by, bz; unflatten(best, bx, by, bz);
        int ex = x - bx, ey = y - by, ez = z - bz;
        best_d2 = static_cast<float>(ex * ex + ey * ey + ez * ez);
    }
    for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = x + dx * k, ny = y + dy * k, nz = z + dz * k;
                if (nx < 0 || nx >= NX || ny < 0 || ny >= NY || nz < 0 || nz >= NZ) continue;
                int s = seed_in[(nz * NY + ny) * NX + nx];
                if (s < 0) continue;
                int sx, sy, sz; unflatten(s, sx, sy, sz);
                int ex = x - sx, ey = y - sy, ez = z - sz;
                float d2 = static_cast<float>(ex * ex + ey * ey + ez * ez);
                if (d2 < best_d2) { best = s; best_d2 = d2; }
            }
    seed_out[idx] = best;
}

__global__ void jfa3d_to_dist_kernel(const int* __restrict__ seed, float* __restrict__ dist) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;
    int idx = (z * NY + y) * NX + x;
    int s = seed[idx];
    if (s < 0) { dist[idx] = MAX_DIST; return; }
    int sx, sy, sz; unflatten(s, sx, sy, sz);
    int dx = x - sx, dy = y - sy, dz = z - sz;
    dist[idx] = fminf(MAX_DIST, sqrtf(static_cast<float>(dx * dx + dy * dy + dz * dz)) * RES);
}

// -------------------------------------------------------------------------
// Shared host/device lookups
// -------------------------------------------------------------------------
__host__ __device__ inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Trilinear lookup of a W x H x D field with voxel size `res` at a world point.
__host__ __device__ inline float trilinear(const float* f, int W, int H, int D, float res,
                                           float x, float y, float z) {
    float fx = x / res - 0.5f, fy = y / res - 0.5f, fz = z / res - 0.5f;
    int x0 = static_cast<int>(floorf(fx)), y0 = static_cast<int>(floorf(fy)), z0 = static_cast<int>(floorf(fz));
    float ax = fx - x0, ay = fy - y0, az = fz - z0;
    int xa = clampi(x0, 0, W - 1), xb = clampi(x0 + 1, 0, W - 1);
    int ya = clampi(y0, 0, H - 1), yb = clampi(y0 + 1, 0, H - 1);
    int za = clampi(z0, 0, D - 1), zb = clampi(z0 + 1, 0, D - 1);
    auto at = [&](int i, int j, int k) { return f[(static_cast<size_t>(k) * H + j) * W + i]; };
    float c00 = at(xa, ya, za) * (1 - ax) + at(xb, ya, za) * ax;
    float c10 = at(xa, yb, za) * (1 - ax) + at(xb, yb, za) * ax;
    float c01 = at(xa, ya, zb) * (1 - ax) + at(xb, ya, zb) * ax;
    float c11 = at(xa, yb, zb) * (1 - ax) + at(xb, yb, zb) * ax;
    float c0 = c00 * (1 - ay) + c10 * ay, c1 = c01 * (1 - ay) + c11 * ay;
    return c0 * (1 - az) + c1 * az;
}

// Distance to the nearest occupied voxel centre.
__host__ __device__ inline float esdf_at(const float* esdf, float x, float y, float z) {
    return trilinear(esdf, NX, NY, NZ, RES, x, y, z);
}

// Geodesic distance to the goal around obstacles.
__host__ __device__ inline float ctg_at(const float* ctg, float x, float y, float z) {
    return trilinear(ctg, CX, CY, CZ, CRES, x, y, z);
}

// -------------------------------------------------------------------------
// GPU wavefront cost-to-go: parallel Bellman-Ford relaxation over 26
// neighbours on the coarse grid; voxels closer than the vehicle radius to an
// obstacle (per the ESDF) are blocked.
// -------------------------------------------------------------------------
__global__ void ctg_init_kernel(const float* __restrict__ esdf, float* __restrict__ ctg,
                                unsigned char* __restrict__ blocked, int goal_idx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= CX || y >= CY || z >= CZ) return;
    int idx = (z * CY + y) * CX + x;
    float d = esdf_at(esdf, (x + 0.5f) * CRES, (y + 0.5f) * CRES, (z + 0.5f) * CRES);
    blocked[idx] = d < ROBOT_R + 0.05f;
    ctg[idx] = idx == goal_idx ? 0.0f : CTG_BLOCKED;
}

__global__ void ctg_relax_kernel(const float* __restrict__ in, float* __restrict__ out,
                                 const unsigned char* __restrict__ blocked, int* __restrict__ changed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= CX || y >= CY || z >= CZ) return;
    int idx = (z * CY + y) * CX + x;
    float best = in[idx];
    if (!blocked[idx]) {
        for (int dz = -1; dz <= 1; dz++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx, ny = y + dy, nz = z + dz;
                    if ((!dx && !dy && !dz) || nx < 0 || nx >= CX || ny < 0 || ny >= CY || nz < 0 || nz >= CZ) continue;
                    float c = in[(nz * CY + ny) * CX + nx] + CRES * sqrtf(static_cast<float>(dx * dx + dy * dy + dz * dz));
                    if (c < best) best = c;
                }
        if (best < in[idx]) *changed = 1;
    }
    out[idx] = best;
}

// One double-integrator step: per-axis acceleration clamp, speed clamp.
__host__ __device__ inline void step_dynamics(float s[6], const float a_in[3]) {
    float a[3];
    for (int i = 0; i < 3; i++) a[i] = fminf(fmaxf(a_in[i], -A_MAX), A_MAX);
    for (int i = 0; i < 3; i++) s[3 + i] += a[i] * DT;
    float v = sqrtf(s[3] * s[3] + s[4] * s[4] + s[5] * s[5]);
    if (v > V_MAX) for (int i = 0; i < 3; i++) s[3 + i] *= V_MAX / v;
    for (int i = 0; i < 3; i++) s[i] += s[3 + i] * DT;
}

// Cost of a T x 3 acceleration sequence from `start` (position, velocity):
// cost-to-go for progress, ESDF barrier for clearance.
__host__ __device__ inline float rollout_cost(const float start[6], const float* controls,
                                              const float* esdf, const float* ctg) {
    float s[6];
    for (int i = 0; i < 6; i++) s[i] = start[i];
    float total = 0.0f;
    for (int t = 0; t < T_HORIZON; t++) {
        const float* a = controls + t * CTRL_DIM;
        step_dynamics(s, a);
        total += W_GOAL * ctg_at(ctg, s[0], s[1], s[2]);
        total += W_CTRL * (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        float d = esdf_at(esdf, s[0], s[1], s[2]) - ROBOT_R;
        if (d < CLEARANCE) {
            float inv = 1.0f / fmaxf(d, 0.05f) - 1.0f / CLEARANCE;
            total += W_OBS * inv * inv;
        }
        if (d < 0.0f) total += COLLIDE_PENALTY;
        if (s[0] < 0 || s[0] > WORLD_X || s[1] < 0 || s[1] > WORLD_Y || s[2] < 0 || s[2] > WORLD_Z)
            total += COLLIDE_PENALTY;
    }
    return total + W_TERM * ctg_at(ctg, s[0], s[1], s[2]);
}

// -------------------------------------------------------------------------
// MPPI kernels: one thread = one sampled trajectory
// -------------------------------------------------------------------------
__global__ void init_rng(curandState* states, int n, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, i, 0, &states[i]);
}

__global__ void rollout_kernel(const float* __restrict__ d_start, const float* __restrict__ d_ctg,
                               const float* __restrict__ d_nominal, const float* __restrict__ d_esdf,
                               float* __restrict__ d_costs, float* __restrict__ d_perturbed,
                               curandState* __restrict__ d_rng, int K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curandState rng = d_rng[k];
    float* u = d_perturbed + static_cast<size_t>(k) * T_HORIZON * CTRL_DIM;
    for (int i = 0; i < T_HORIZON * CTRL_DIM; i++)
        u[i] = fminf(fmaxf(d_nominal[i] + SIGMA * curand_normal(&rng), -A_MAX), A_MAX);
    float start[6];
    for (int i = 0; i < 6; i++) start[i] = d_start[i];
    d_costs[k] = rollout_cost(start, u, d_esdf, d_ctg);
    d_rng[k] = rng;
}

// -------------------------------------------------------------------------
// CPU reference for the same rollout batch
// -------------------------------------------------------------------------
static double cpu_rollout_ms(const std::vector<float>& esdf, const std::vector<float>& ctg, const float start[6],
                             const std::vector<float>& controls, int K, std::vector<float>& costs) {
    costs.resize(K);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < K; k++)
        costs[k] = rollout_cost(start, controls.data() + static_cast<size_t>(k) * T_HORIZON * CTRL_DIM,
                                esdf.data(), ctg.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// -------------------------------------------------------------------------
// Rendering
// -------------------------------------------------------------------------
constexpr int TOP_PX  = 480;              // top view: 16 m x 16 m
constexpr int SIDE_W  = 480;              // side view: 16 m x 8 m
constexpr int SIDE_H  = 240;

static cv::Vec3b dist_color(float d) {
    if (d <= 0.5f * RES) return cv::Vec3b(45, 45, 45);
    float t = std::min(d / 3.0f, 1.0f);
    return cv::Vec3b(static_cast<uchar>(80 + (1 - t) * 60), static_cast<uchar>(30 + t * 200),
                     static_cast<uchar>(40 + (1 - t) * 180));
}

// XY slice at height z (top view, +y up).
static cv::Mat render_top(const std::vector<float>& esdf, float z) {
    int k = clampi(static_cast<int>(z / RES), 0, NZ - 1);
    cv::Mat img(NY, NX, CV_8UC3);
    for (int j = 0; j < NY; j++)
        for (int i = 0; i < NX; i++)
            img.at<cv::Vec3b>(NY - 1 - j, i) = dist_color(esdf[(static_cast<size_t>(k) * NY + j) * NX + i]);
    cv::Mat out;
    cv::resize(img, out, cv::Size(TOP_PX, TOP_PX), 0, 0, cv::INTER_NEAREST);
    return out;
}

// XZ slice at y (side view, +z up).
static cv::Mat render_side(const std::vector<float>& esdf, float y) {
    int j = clampi(static_cast<int>(y / RES), 0, NY - 1);
    cv::Mat img(NZ, NX, CV_8UC3);
    for (int k = 0; k < NZ; k++)
        for (int i = 0; i < NX; i++)
            img.at<cv::Vec3b>(NZ - 1 - k, i) = dist_color(esdf[(static_cast<size_t>(k) * NY + j) * NX + i]);
    cv::Mat out;
    cv::resize(img, out, cv::Size(SIDE_W, SIDE_H), 0, 0, cv::INTER_NEAREST);
    return out;
}

static cv::Point top_px(float x, float y) {
    return cv::Point(static_cast<int>(x / WORLD_X * TOP_PX), static_cast<int>((1 - y / WORLD_Y) * TOP_PX));
}
static cv::Point side_px(float x, float z) {
    return cv::Point(static_cast<int>(x / WORLD_X * SIDE_W), static_cast<int>((1 - z / WORLD_Z) * SIDE_H));
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------
int main(int argc, char** argv) {
    cudabot::DemoArgs args(argc, argv, "3D ESDF-MPPI: Jump Flooding distance field + double-integrator MPPI");
    const int K = args.get_int("samples", 4096, "sampled trajectories per iteration", 1);
    const int max_steps = args.get_int("steps", 400, "maximum simulation steps", 1);
    const int seed = args.get_int("seed", 2026, "cuRAND seed", 0);
    const bool write_video = !args.flag("no-video", "skip the AVI/GIF output");
    args.finish();

    // 1. Scene and 3D ESDF
    std::vector<unsigned char> occ;
    build_scene(occ);
    const size_t cells = occ.size();
    unsigned char* d_occ = nullptr;
    int *d_seed_a = nullptr, *d_seed_b = nullptr;
    float* d_esdf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_occ, cells));
    CUDA_CHECK(cudaMalloc(&d_seed_a, cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seed_b, cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_esdf, cells * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_occ, occ.data(), cells, cudaMemcpyHostToDevice));

    dim3 blk(8, 8, 4), grd((NX + 7) / 8, (NY + 7) / 8, (NZ + 3) / 4);
    auto build_esdf = [&]() {
        jfa3d_init_kernel<<<grd, blk>>>(d_occ, d_seed_a);
        int *in_ptr = d_seed_a, *out_ptr = d_seed_b;
        for (int k = std::max(std::max(NX, NY), NZ) / 2; k >= 1; k /= 2) {
            jfa3d_step_kernel<<<grd, blk>>>(in_ptr, out_ptr, k);
            std::swap(in_ptr, out_ptr);
        }
        jfa3d_to_dist_kernel<<<grd, blk>>>(in_ptr, d_esdf);
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    build_esdf();   // warm-up (module load / JIT)
    auto e0 = std::chrono::high_resolution_clock::now();
    build_esdf();
    auto e1 = std::chrono::high_resolution_clock::now();
    std::printf("3D ESDF (JFA, %dx%dx%d = %zu voxels): %.2f ms\n", NX, NY, NZ, cells,
                std::chrono::duration<double, std::milli>(e1 - e0).count());
    std::vector<float> h_esdf(cells);
    CUDA_CHECK(cudaMemcpy(h_esdf.data(), d_esdf, cells * sizeof(float), cudaMemcpyDeviceToHost));

    // 2. Cost-to-go from the goal (wavefront on the coarse grid)
    float state[6] = {2.0f, 2.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    const float start_pos[3] = {state[0], state[1], state[2]};
    const float goal[3] = {14.0f, 14.5f, 4.5f};
    const size_t ccells = static_cast<size_t>(CX) * CY * CZ;
    float *d_ctg_a = nullptr, *d_ctg_b = nullptr;
    unsigned char* d_blocked = nullptr;
    int* d_changed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ctg_a, ccells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ctg_b, ccells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blocked, ccells));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    dim3 cgrd((CX + 7) / 8, (CY + 7) / 8, (CZ + 3) / 4);
    const int goal_idx = (static_cast<int>(goal[2] / CRES) * CY + static_cast<int>(goal[1] / CRES)) * CX +
                         static_cast<int>(goal[0] / CRES);
    auto c0 = std::chrono::high_resolution_clock::now();
    ctg_init_kernel<<<cgrd, blk>>>(d_esdf, d_ctg_a, d_blocked, goal_idx);
    float* d_ctg = d_ctg_a;
    float* d_ctg_next = d_ctg_b;
    int sweeps = 0, changed = 1;
    while (changed) {
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
        for (int i = 0; i < 16; i++, sweeps++) {   // check convergence every 16 sweeps
            ctg_relax_kernel<<<cgrd, blk>>>(d_ctg, d_ctg_next, d_blocked, d_changed);
            std::swap(d_ctg, d_ctg_next);
        }
        CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
    }
    auto c1 = std::chrono::high_resolution_clock::now();
    std::vector<float> h_ctg(ccells);
    CUDA_CHECK(cudaMemcpy(h_ctg.data(), d_ctg, ccells * sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("Cost-to-go (wavefront, %dx%dx%d, %d sweeps): %.2f ms; geodesic start->goal %.2f m (straight line %.2f m)\n",
                CX, CY, CZ, sweeps, std::chrono::duration<double, std::milli>(c1 - c0).count(),
                ctg_at(h_ctg.data(), state[0], state[1], state[2]),
                std::sqrt((goal[0]-state[0])*(goal[0]-state[0]) + (goal[1]-state[1])*(goal[1]-state[1]) + (goal[2]-state[2])*(goal[2]-state[2])));

    // 3. MPPI buffers
    const int U = T_HORIZON * CTRL_DIM;
    float *d_start, *d_nominal, *d_costs, *d_weights, *d_perturbed;
    curandState* d_rng;
    CUDA_CHECK(cudaMalloc(&d_start, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nominal, U * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_costs, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_perturbed, static_cast<size_t>(K) * U * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rng, K * sizeof(curandState)));
    CUDA_CHECK(cudaMemset(d_nominal, 0, U * sizeof(float)));
    const int threads = 256, blocks = (K + threads - 1) / threads;
    init_rng<<<blocks, threads>>>(d_rng, K, static_cast<unsigned long long>(seed));

    CUDA_CHECK(cudaMemcpy(d_start, state, sizeof(state), cudaMemcpyHostToDevice));

    // 4. CPU reference on the first batch: same controls, same cost function
    rollout_kernel<<<blocks, threads>>>(d_start, d_ctg, d_nominal, d_esdf, d_costs, d_perturbed, d_rng, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto g0 = std::chrono::high_resolution_clock::now();
    rollout_kernel<<<blocks, threads>>>(d_start, d_ctg, d_nominal, d_esdf, d_costs, d_perturbed, d_rng, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto g1 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
    std::vector<float> h_controls(static_cast<size_t>(K) * U), h_gpu_costs(K), h_cpu_costs;
    CUDA_CHECK(cudaMemcpy(h_controls.data(), d_perturbed, h_controls.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gpu_costs.data(), d_costs, K * sizeof(float), cudaMemcpyDeviceToHost));
    double cpu_ms = cpu_rollout_ms(h_esdf, h_ctg, state, h_controls, K, h_cpu_costs);
    float max_rel = 0.0f;
    for (int k = 0; k < K; k++)
        max_rel = std::max(max_rel, std::fabs(h_cpu_costs[k] - h_gpu_costs[k]) / std::max(1.0f, std::fabs(h_cpu_costs[k])));
    std::printf("Rollout batch (K=%d, T=%d): CPU %.2f ms, GPU %.3f ms (%.0fx); max relative cost diff %.2e\n",
                K, T_HORIZON, cpu_ms, gpu_ms, cpu_ms / std::max(gpu_ms, 1e-6), max_rel);
    CUDA_CHECK(cudaMemset(d_nominal, 0, U * sizeof(float)));

    // 5. Closed loop
    cv::VideoWriter video;
    if (write_video) {
        cudabot::ensure_dirs({"gif"});
        video.open("gif/gpu_esdf_mppi_3d.avi", cudabot::avi_fourcc(), 20, cv::Size(TOP_PX + SIDE_W, TOP_PX));
    }
    std::vector<float> h_nominal(U, 0.0f);
    std::vector<cv::Point3f> path = {cv::Point3f(state[0], state[1], state[2])};
    const int SHOW = std::min(K, 48);
    std::vector<float> h_samples(static_cast<size_t>(SHOW) * U);
    double mppi_ms = 0.0;
    float min_clearance = FLT_MAX, path_len = 0.0f;
    bool reached = false;
    int steps = 0;

    for (int step = 0; step < max_steps; step++) {
        CUDA_CHECK(cudaMemcpy(d_start, state, sizeof(state), cudaMemcpyHostToDevice));
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS_PER_STEP; it++) {
            rollout_kernel<<<blocks, threads>>>(d_start, d_ctg, d_nominal, d_esdf, d_costs, d_perturbed, d_rng, K);
            cudabot::launch_softmin_weights(d_costs, d_weights, K, LAMBDA);
            cudabot::launch_weighted_control_update(d_perturbed, d_weights, d_nominal, K, U);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        mppi_ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
        CUDA_CHECK(cudaMemcpy(h_nominal.data(), d_nominal, U * sizeof(float), cudaMemcpyDeviceToHost));
        if (write_video)
            CUDA_CHECK(cudaMemcpy(h_samples.data(), d_perturbed, h_samples.size() * sizeof(float), cudaMemcpyDeviceToHost));

        // Draw the plan from the pre-step state, then apply the first control.
        cv::Mat frame;
        if (write_video) {
            cv::Mat top = render_top(h_esdf, state[2]), side = render_side(h_esdf, state[1]);
            auto draw_rollout = [&](const float* u, cv::Scalar color, int width) {
                float s[6];
                std::copy(state, state + 6, s);
                cv::Point pt = top_px(s[0], s[1]), ps = side_px(s[0], s[2]);
                for (int t = 0; t < T_HORIZON; t++) {
                    step_dynamics(s, u + t * CTRL_DIM);
                    cv::Point nt = top_px(s[0], s[1]), ns = side_px(s[0], s[2]);
                    cv::line(top, pt, nt, color, width, cv::LINE_AA);
                    cv::line(side, ps, ns, color, width, cv::LINE_AA);
                    pt = nt; ps = ns;
                }
            };
            for (int k = 0; k < SHOW; k++) draw_rollout(h_samples.data() + static_cast<size_t>(k) * U, cv::Scalar(150, 150, 150), 1);
            for (size_t i = 1; i < path.size(); i++) {
                cv::line(top, top_px(path[i-1].x, path[i-1].y), top_px(path[i].x, path[i].y), cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
                cv::line(side, side_px(path[i-1].x, path[i-1].z), side_px(path[i].x, path[i].z), cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            }
            draw_rollout(h_nominal.data(), cv::Scalar(0, 230, 255), 2);
            cv::circle(top, top_px(start_pos[0], start_pos[1]), 6, cv::Scalar(255, 120, 80), cv::FILLED);
            cv::circle(top, top_px(goal[0], goal[1]), 7, cv::Scalar(80, 255, 80), cv::FILLED);
            cv::circle(side, side_px(goal[0], goal[2]), 7, cv::Scalar(80, 255, 80), cv::FILLED);
            cv::circle(top, top_px(state[0], state[1]), 6, cv::Scalar(0, 255, 255), cv::FILLED);
            cv::circle(side, side_px(state[0], state[2]), 6, cv::Scalar(0, 255, 255), cv::FILLED);
            frame = cv::Mat(TOP_PX, TOP_PX + SIDE_W, CV_8UC3, cv::Scalar(25, 25, 25));
            top.copyTo(frame(cv::Rect(0, 0, TOP_PX, TOP_PX)));
            side.copyTo(frame(cv::Rect(TOP_PX, 0, SIDE_W, SIDE_H)));
            const float dist = std::sqrt((state[0]-goal[0])*(state[0]-goal[0]) + (state[1]-goal[1])*(state[1]-goal[1]) +
                                         (state[2]-goal[2])*(state[2]-goal[2]));
            char lines[5][96];
            std::snprintf(lines[0], 96, "3D ESDF-MPPI (GPU Jump Flooding + double integrator)");
            std::snprintf(lines[1], 96, "top: slice at z=%.1f m   side: slice at y=%.1f m", state[2], state[1]);
            std::snprintf(lines[2], 96, "step %d   dist to goal %.2f m", step, dist);
            std::snprintf(lines[3], 96, "K=%d T=%d   MPPI %.2f ms/step", K, T_HORIZON, mppi_ms / (step + 1));
            std::snprintf(lines[4], 96, "min clearance %.2f m", min_clearance == FLT_MAX ? 0.0f : min_clearance);
            for (int i = 0; i < 5; i++)
                cv::putText(frame, lines[i], cv::Point(TOP_PX + 12, SIDE_H + 34 * (i + 1)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
        }

        float prev[3] = {state[0], state[1], state[2]};
        step_dynamics(state, h_nominal.data());
        path.push_back(cv::Point3f(state[0], state[1], state[2]));
        path_len += std::sqrt((state[0]-prev[0])*(state[0]-prev[0]) + (state[1]-prev[1])*(state[1]-prev[1]) + (state[2]-prev[2])*(state[2]-prev[2]));
        min_clearance = std::min(min_clearance, esdf_at(h_esdf.data(), state[0], state[1], state[2]) - ROBOT_R);
        steps = step + 1;

        // Warm start: shift the nominal sequence by one step.
        std::copy(h_nominal.begin() + CTRL_DIM, h_nominal.end(), h_nominal.begin());
        std::fill(h_nominal.end() - CTRL_DIM, h_nominal.end(), 0.0f);
        CUDA_CHECK(cudaMemcpy(d_nominal, h_nominal.data(), U * sizeof(float), cudaMemcpyHostToDevice));

        if (write_video) {
            video.write(frame);
            cudabot::imshow("gpu_esdf_mppi_3d", frame);
            cudabot::waitKey(1);
        }
        float dx = state[0] - goal[0], dy = state[1] - goal[1], dz = state[2] - goal[2];
        if (std::sqrt(dx * dx + dy * dy + dz * dz) < GOAL_TOL) { reached = true; }
        if (reached || step + 1 == max_steps) {
            if (write_video) for (int i = 0; i < 30; i++) video.write(frame);   // hold the last frame
            break;
        }
    }

    std::printf("%s after %d steps (%.1f s); path length %.2f m; min clearance %.2f m (%s)\n",
                reached ? "Goal reached" : "Goal NOT reached", steps, steps * DT, path_len, min_clearance,
                min_clearance >= 0.0f ? "collision-free" : "COLLISION");
    std::printf("MPPI: %.3f ms per control step (%d iterations of K=%d, T=%d)\n",
                mppi_ms / std::max(steps, 1), ITERS_PER_STEP, K, T_HORIZON);

    if (write_video) {
        video.release();
        std::printf("Video saved to gif/gpu_esdf_mppi_3d.avi\n");
        cudabot::avi_to_gif("gif/gpu_esdf_mppi_3d.avi", "gif/gpu_esdf_mppi_3d.gif", 20, 900);
        std::printf("GIF saved to gif/gpu_esdf_mppi_3d.gif\n");
    }

    CUDA_CHECK(cudaFree(d_occ));
    CUDA_CHECK(cudaFree(d_seed_a));
    CUDA_CHECK(cudaFree(d_seed_b));
    CUDA_CHECK(cudaFree(d_esdf));
    CUDA_CHECK(cudaFree(d_start));
    CUDA_CHECK(cudaFree(d_ctg_a));
    CUDA_CHECK(cudaFree(d_ctg_b));
    CUDA_CHECK(cudaFree(d_blocked));
    CUDA_CHECK(cudaFree(d_changed));
    CUDA_CHECK(cudaFree(d_nominal));
    CUDA_CHECK(cudaFree(d_costs));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_perturbed));
    CUDA_CHECK(cudaFree(d_rng));
    return reached && min_clearance >= 0.0f ? 0 : 1;
}
