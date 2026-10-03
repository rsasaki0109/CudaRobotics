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
    - --movers N adds moving spheres beyond the wall; --mode chooses how the
      planner sees them (ignore, rebuild the ESDF each step, or predict them
      at constant velocity in the rollout cost) and --trials compares the
      modes on the same scenarios; mode 4 updates the ESDF only in voxel
      windows around the movers instead of rerunning the full JFA
    Output: gif/gpu_esdf_mppi_3d.gif (top view + side view slices),
            gif/gpu_esdf_mppi_3d_dynamic.gif with movers
 ************************************************************************/

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cuda_check.cuh"
#include "cuda_video.h"
#include "demo_args.h"
#include "display.h"
#include "mppi_reduction.cuh"
#include "jfa.cuh"

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

// Moving spherical obstacles (--movers): constant velocity, bouncing inside a
// box beyond the wall. How the planner sees them depends on --mode.
constexpr int   MAX_MOVERS = 8;
constexpr float MOVER_R    = 0.45f;
struct Movers {
    int n = 0;
    float p[MAX_MOVERS][3], v[MAX_MOVERS][3];   // current position and velocity
    float lo[3], hi[3];                          // region they bounce inside
};
enum DynMode { MODE_STATIC = 0, MODE_REBUILD = 1, MODE_PREDICT = 2, MODE_PREDICT_BOUNCE = 3, MODE_REBUILD_LOCAL = 4 };
static const char* MODE_NAMES[] = {"static", "rebuild", "predict", "predict_bounce", "rebuild_local"};

// Mover coordinate after tau seconds: constant velocity, or with reflections at the
// region bounds (a triangle wave) when bounce is set.
__host__ __device__ inline float predict_coord(float x0, float v, float tau, float lo, float hi, bool bounce) {
    float x = x0 + v * tau;
    if (!bounce) return x;
    float L = hi - lo, y = fmodf(x - lo, 2.0f * L);
    if (y < 0.0f) y += 2.0f * L;
    return lo + (y <= L ? y : 2.0f * L - y);
}
const float MOVER_BOX[2][3] = {{1.0f, 9.0f, 1.5f}, {15.0f, 13.5f, 6.0f}};

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

// Local ESDF update for movers. Movers only add occupancy on top of the static
// map, so inside a voxel window the distance is the static ESDF or the distance to
// the nearest mover sphere, whichever is smaller; with restore the window gets its
// static values back. Windows extend past each sphere by the vehicle radius plus
// the clearance band, so outside them the movers never change the rollout cost.
__global__ void esdf_window_kernel(const float* __restrict__ esdf_static, float* __restrict__ esdf_out,
                                   int x0, int y0, int z0, int wx, int wy, int wz, Movers mv, bool restore) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= wx || j >= wy || k >= wz) return;
    int x = x0 + i, y = y0 + j, z = z0 + k;
    size_t idx = (static_cast<size_t>(z) * NY + y) * NX + x;
    float d = esdf_static[idx];
    if (!restore) {
        float px = (x + 0.5f) * RES, py = (y + 0.5f) * RES, pz = (z + 0.5f) * RES;
        for (int m = 0; m < mv.n; m++) {
            float ex = px - mv.p[m][0], ey = py - mv.p[m][1], ez = pz - mv.p[m][2];
            d = fminf(d, fmaxf(0.0f, sqrtf(ex * ex + ey * ey + ez * ez) - MOVER_R));
        }
    }
    esdf_out[idx] = d;
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
                                              const float* esdf, const float* ctg,
                                              const Movers& mv, int predict) {
    float s[6];
    for (int i = 0; i < 6; i++) s[i] = start[i];
    float total = 0.0f;
    for (int t = 0; t < T_HORIZON; t++) {
        const float* a = controls + t * CTRL_DIM;
        step_dynamics(s, a);
        total += W_GOAL * ctg_at(ctg, s[0], s[1], s[2]);
        total += W_CTRL * (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        float d = esdf_at(esdf, s[0], s[1], s[2]) - ROBOT_R;
        if (predict) {   // distance to each mover at its predicted position (1: linear, 2: bouncing)
            float tau = (t + 1) * DT;
            for (int i = 0; i < mv.n; i++) {
                float ex = s[0] - predict_coord(mv.p[i][0], mv.v[i][0], tau, mv.lo[0], mv.hi[0], predict == 2);
                float ey = s[1] - predict_coord(mv.p[i][1], mv.v[i][1], tau, mv.lo[1], mv.hi[1], predict == 2);
                float ez = s[2] - predict_coord(mv.p[i][2], mv.v[i][2], tau, mv.lo[2], mv.hi[2], predict == 2);
                d = fminf(d, sqrtf(ex * ex + ey * ey + ez * ez) - MOVER_R - ROBOT_R);
            }
        }
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
                               curandState* __restrict__ d_rng, int K, Movers mv, int predict) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curandState rng = d_rng[k];
    float* u = d_perturbed + static_cast<size_t>(k) * T_HORIZON * CTRL_DIM;
    for (int i = 0; i < T_HORIZON * CTRL_DIM; i++)
        u[i] = fminf(fmaxf(d_nominal[i] + SIGMA * curand_normal(&rng), -A_MAX), A_MAX);
    float start[6];
    for (int i = 0; i < 6; i++) start[i] = d_start[i];
    d_costs[k] = rollout_cost(start, u, d_esdf, d_ctg, mv, predict);
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
                                esdf.data(), ctg.data(), Movers(), false);
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
    const bool no_video = args.flag("no-video", "skip the AVI/GIF output");
    const int n_movers = args.get_int("movers", 0, "moving spherical obstacles beyond the wall (max 8)", 0);
    const int mode_arg = args.get_int("mode", MODE_PREDICT, "with movers: 0 static map, 1 rebuild ESDF, 2 predict, 3 predict with bounces, 4 local ESDF update", 0);
    const int trials = args.get_int("trials", 0, "with movers: run N episodes per mode and print a table", 0);
    const float mover_speed = args.get_float("mover-speed", 1.0f, "with movers: speed multiplier (base 0.8-1.5 m/s)");
    args.finish();
    if (n_movers > MAX_MOVERS || mode_arg > MODE_REBUILD_LOCAL) { std::fprintf(stderr, "--movers <= 8, --mode 0..4\n"); return 2; }
    const bool write_video = !no_video && trials == 0;

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
        cudabot::jfa3d_distance(d_occ, d_seed_a, d_seed_b, d_esdf, NX, NY, NZ, RES, MAX_DIST, true);
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
    rollout_kernel<<<blocks, threads>>>(d_start, d_ctg, d_nominal, d_esdf, d_costs, d_perturbed, d_rng, K, Movers(), false);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto g0 = std::chrono::high_resolution_clock::now();
    rollout_kernel<<<blocks, threads>>>(d_start, d_ctg, d_nominal, d_esdf, d_costs, d_perturbed, d_rng, K, Movers(), false);
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

    // 5. Closed loop (one episode; with movers, the planner's view depends on mode)
    const float start_state[6] = {state[0], state[1], state[2], 0.0f, 0.0f, 0.0f};
    unsigned char* d_occ_dyn = nullptr;
    float* d_esdf_dyn = nullptr;
    std::vector<unsigned char> occ_dyn;
    if (n_movers > 0) {
        CUDA_CHECK(cudaMalloc(&d_occ_dyn, cells));
        CUDA_CHECK(cudaMalloc(&d_esdf_dyn, cells * sizeof(float)));
    }
    auto init_movers = [&](int episode_seed) {
        Movers mv;
        mv.n = n_movers;
        for (int a = 0; a < 3; a++) { mv.lo[a] = MOVER_BOX[0][a]; mv.hi[a] = MOVER_BOX[1][a]; }
        std::mt19937 rng(static_cast<unsigned>(episode_seed));
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        for (int i = 0; i < mv.n; i++) {
            for (int a = 0; a < 3; a++)
                mv.p[i][a] = MOVER_BOX[0][a] + u01(rng) * (MOVER_BOX[1][a] - MOVER_BOX[0][a]);
            float speed = mover_speed * (0.8f + 0.7f * u01(rng)), yaw = 6.2832f * u01(rng), climb = 0.3f * (u01(rng) - 0.5f);
            mv.v[i][0] = speed * std::cos(yaw); mv.v[i][1] = speed * std::sin(yaw); mv.v[i][2] = speed * climb;
        }
        return mv;
    };
    auto step_movers = [&](Movers& mv) {
        for (int i = 0; i < mv.n; i++)
            for (int a = 0; a < 3; a++) {
                mv.p[i][a] += mv.v[i][a] * DT;
                if (mv.p[i][a] < MOVER_BOX[0][a] || mv.p[i][a] > MOVER_BOX[1][a]) {
                    mv.v[i][a] = -mv.v[i][a];
                    mv.p[i][a] = std::min(std::max(mv.p[i][a], MOVER_BOX[0][a]), MOVER_BOX[1][a]);
                }
            }
    };
    // True clearance of the vehicle: static map and the movers' actual positions.
    auto clearance = [&](const float* p, const Movers& mv) {
        float d = esdf_at(h_esdf.data(), p[0], p[1], p[2]) - ROBOT_R;
        for (int i = 0; i < mv.n; i++)
            d = std::min(d, std::sqrt((p[0]-mv.p[i][0])*(p[0]-mv.p[i][0]) + (p[1]-mv.p[i][1])*(p[1]-mv.p[i][1]) +
                                      (p[2]-mv.p[i][2])*(p[2]-mv.p[i][2])) - MOVER_R - ROBOT_R);
        return d;
    };
    // Rebuild mode: stamp the movers' current spheres into the occupancy grid and rerun JFA.
    auto rebuild_esdf = [&](const Movers& mv) {
        occ_dyn = occ;
        const int r = static_cast<int>(std::ceil(MOVER_R / RES));
        for (int i = 0; i < mv.n; i++) {
            int ci = static_cast<int>(mv.p[i][0] / RES), cj = static_cast<int>(mv.p[i][1] / RES), ck = static_cast<int>(mv.p[i][2] / RES);
            for (int k = std::max(0, ck - r); k <= std::min(NZ - 1, ck + r); k++)
                for (int j = std::max(0, cj - r); j <= std::min(NY - 1, cj + r); j++)
                    for (int ii = std::max(0, ci - r); ii <= std::min(NX - 1, ci + r); ii++) {
                        float dx = (ii + 0.5f) * RES - mv.p[i][0], dy = (j + 0.5f) * RES - mv.p[i][1], dz = (k + 0.5f) * RES - mv.p[i][2];
                        if (dx*dx + dy*dy + dz*dz <= MOVER_R * MOVER_R) occ_dyn[(static_cast<size_t>(k) * NY + j) * NX + ii] = 1u;
                    }
        }
        CUDA_CHECK(cudaMemcpy(d_occ_dyn, occ_dyn.data(), cells, cudaMemcpyHostToDevice));
        cudabot::jfa3d_distance(d_occ_dyn, d_seed_a, d_seed_b, d_esdf_dyn, NX, NY, NZ, RES, MAX_DIST, true);
    };

    // Rebuild-local mode: restore last step's windows to the static ESDF, then write
    // the movers' current spheres into new windows (see esdf_window_kernel).
    std::vector<std::array<int, 6>> prev_windows;
    auto local_update_esdf = [&](const Movers& mv) {
        dim3 wblk(8, 8, 4);
        auto launch = [&](const std::array<int, 6>& w, bool restore) {
            dim3 wgrd((w[3] + 7) / 8, (w[4] + 7) / 8, (w[5] + 3) / 4);
            esdf_window_kernel<<<wgrd, wblk>>>(d_esdf, d_esdf_dyn, w[0], w[1], w[2], w[3], w[4], w[5], mv, restore);
        };
        for (const auto& w : prev_windows) launch(w, true);
        prev_windows.clear();
        const float half = MOVER_R + ROBOT_R + CLEARANCE + 2.0f * RES;
        const int lim[3] = {NX, NY, NZ};
        for (int m = 0; m < mv.n; m++) {
            std::array<int, 6> w;
            for (int a = 0; a < 3; a++) {
                int lo = std::max(0, static_cast<int>(std::floor((mv.p[m][a] - half) / RES)));
                int hi = std::min(lim[a] - 1, static_cast<int>(std::ceil((mv.p[m][a] + half) / RES)));
                w[a] = lo; w[3 + a] = hi - lo + 1;
            }
            prev_windows.push_back(w);
        }
        for (const auto& w : prev_windows) launch(w, false);
    };

    struct Episode { bool reached = false; int steps = 0; float path_len = 0, min_clearance = FLT_MAX; double ms = 0; };
    auto run_episode = [&](int mode, int episode_seed, bool video_on) {
        Episode ep;
        float st[6];
        std::copy(start_state, start_state + 6, st);
        Movers mv = init_movers(episode_seed);
        if (n_movers > 0 && mode == MODE_REBUILD_LOCAL) {
            CUDA_CHECK(cudaMemcpy(d_esdf_dyn, d_esdf, cells * sizeof(float), cudaMemcpyDeviceToDevice));
            prev_windows.clear();
        }
        init_rng<<<blocks, threads>>>(d_rng, K, static_cast<unsigned long long>(episode_seed));
        CUDA_CHECK(cudaMemset(d_nominal, 0, U * sizeof(float)));
        const std::string tag = n_movers > 0 ? "gpu_esdf_mppi_3d_dynamic" : "gpu_esdf_mppi_3d";
        const std::string avi = "gif/" + tag + ".avi", gif = "gif/" + tag + ".gif";
        cv::VideoWriter video;
        if (video_on) {
            cudabot::ensure_dirs({"gif"});
            video.open(avi, cudabot::avi_fourcc(), 20, cv::Size(TOP_PX + SIDE_W, TOP_PX));
        }
        std::vector<float> h_nominal(U, 0.0f);
        std::vector<cv::Point3f> path = {cv::Point3f(st[0], st[1], st[2])};
        const int SHOW = std::min(K, 48);
        std::vector<float> h_samples(static_cast<size_t>(SHOW) * U);
        const int predict = n_movers == 0 ? 0 : mode == MODE_PREDICT ? 1 : mode == MODE_PREDICT_BOUNCE ? 2 : 0;

        for (int step = 0; step < max_steps; step++) {
            CUDA_CHECK(cudaMemcpy(d_start, st, sizeof(st), cudaMemcpyHostToDevice));
            auto t0 = std::chrono::high_resolution_clock::now();
            const float* esdf_now = d_esdf;
            if (n_movers > 0 && mode == MODE_REBUILD) { rebuild_esdf(mv); esdf_now = d_esdf_dyn; }
            if (n_movers > 0 && mode == MODE_REBUILD_LOCAL) { local_update_esdf(mv); esdf_now = d_esdf_dyn; }
            for (int it = 0; it < ITERS_PER_STEP; it++) {
                rollout_kernel<<<blocks, threads>>>(d_start, d_ctg, d_nominal, esdf_now, d_costs, d_perturbed, d_rng, K, mv, predict);
                cudabot::launch_softmin_weights(d_costs, d_weights, K, LAMBDA);
                cudabot::launch_weighted_control_update(d_perturbed, d_weights, d_nominal, K, U);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            ep.ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
            CUDA_CHECK(cudaMemcpy(h_nominal.data(), d_nominal, U * sizeof(float), cudaMemcpyDeviceToHost));
            if (video_on)
                CUDA_CHECK(cudaMemcpy(h_samples.data(), d_perturbed, h_samples.size() * sizeof(float), cudaMemcpyDeviceToHost));

            // Draw the plan from the pre-step state, then apply the first control.
            cv::Mat frame;
            if (video_on) {
                cv::Mat top = render_top(h_esdf, st[2]), side = render_side(h_esdf, st[1]);
                auto draw_rollout = [&](const float* u, cv::Scalar color, int width) {
                    float s2[6];
                    std::copy(st, st + 6, s2);
                    cv::Point pt = top_px(s2[0], s2[1]), ps = side_px(s2[0], s2[2]);
                    for (int t = 0; t < T_HORIZON; t++) {
                        step_dynamics(s2, u + t * CTRL_DIM);
                        cv::Point nt = top_px(s2[0], s2[1]), ns = side_px(s2[0], s2[2]);
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
                const int mr = static_cast<int>(MOVER_R / WORLD_X * TOP_PX);
                for (int i = 0; i < mv.n; i++) {
                    cv::circle(top, top_px(mv.p[i][0], mv.p[i][1]), mr, cv::Scalar(60, 60, 255), cv::FILLED, cv::LINE_AA);
                    cv::circle(side, side_px(mv.p[i][0], mv.p[i][2]), mr, cv::Scalar(60, 60, 255), cv::FILLED, cv::LINE_AA);
                    if (predict) {   // where the planner expects it over the horizon
                        cv::Point prev_pt = top_px(mv.p[i][0], mv.p[i][1]);
                        for (int t = 1; t <= T_HORIZON; t++) {
                            float tau = t * DT;
                            cv::Point pt = top_px(predict_coord(mv.p[i][0], mv.v[i][0], tau, mv.lo[0], mv.hi[0], predict == 2),
                                                  predict_coord(mv.p[i][1], mv.v[i][1], tau, mv.lo[1], mv.hi[1], predict == 2));
                            cv::line(top, prev_pt, pt, cv::Scalar(60, 60, 255), 1, cv::LINE_AA);
                            prev_pt = pt;
                        }
                    }
                }
                cv::circle(top, top_px(start_pos[0], start_pos[1]), 6, cv::Scalar(255, 120, 80), cv::FILLED);
                cv::circle(top, top_px(goal[0], goal[1]), 7, cv::Scalar(80, 255, 80), cv::FILLED);
                cv::circle(side, side_px(goal[0], goal[2]), 7, cv::Scalar(80, 255, 80), cv::FILLED);
                cv::circle(top, top_px(st[0], st[1]), 6, cv::Scalar(0, 255, 255), cv::FILLED);
                cv::circle(side, side_px(st[0], st[2]), 6, cv::Scalar(0, 255, 255), cv::FILLED);
                frame = cv::Mat(TOP_PX, TOP_PX + SIDE_W, CV_8UC3, cv::Scalar(25, 25, 25));
                top.copyTo(frame(cv::Rect(0, 0, TOP_PX, TOP_PX)));
                side.copyTo(frame(cv::Rect(TOP_PX, 0, SIDE_W, SIDE_H)));
                const float dist = std::sqrt((st[0]-goal[0])*(st[0]-goal[0]) + (st[1]-goal[1])*(st[1]-goal[1]) +
                                             (st[2]-goal[2])*(st[2]-goal[2]));
                char lines[5][96];
                if (n_movers > 0)
                    std::snprintf(lines[0], 96, "3D ESDF-MPPI, %d movers, mode: %s", n_movers, MODE_NAMES[mode]);
                else
                    std::snprintf(lines[0], 96, "3D ESDF-MPPI (GPU Jump Flooding + double integrator)");
                std::snprintf(lines[1], 96, "top: slice at z=%.1f m   side: slice at y=%.1f m", st[2], st[1]);
                std::snprintf(lines[2], 96, "step %d   dist to goal %.2f m", step, dist);
                std::snprintf(lines[3], 96, "K=%d T=%d   MPPI %.2f ms/step", K, T_HORIZON, ep.ms / (step + 1));
                std::snprintf(lines[4], 96, "min clearance %.2f m", ep.min_clearance == FLT_MAX ? 0.0f : ep.min_clearance);
                for (int i = 0; i < 5; i++)
                    cv::putText(frame, lines[i], cv::Point(TOP_PX + 12, SIDE_H + 34 * (i + 1)),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
            }

            float prev[3] = {st[0], st[1], st[2]};
            step_dynamics(st, h_nominal.data());
            step_movers(mv);
            path.push_back(cv::Point3f(st[0], st[1], st[2]));
            ep.path_len += std::sqrt((st[0]-prev[0])*(st[0]-prev[0]) + (st[1]-prev[1])*(st[1]-prev[1]) + (st[2]-prev[2])*(st[2]-prev[2]));
            ep.min_clearance = std::min(ep.min_clearance, clearance(st, mv));
            ep.steps = step + 1;

            // Warm start: shift the nominal sequence by one step.
            std::copy(h_nominal.begin() + CTRL_DIM, h_nominal.end(), h_nominal.begin());
            std::fill(h_nominal.end() - CTRL_DIM, h_nominal.end(), 0.0f);
            CUDA_CHECK(cudaMemcpy(d_nominal, h_nominal.data(), U * sizeof(float), cudaMemcpyHostToDevice));

            if (video_on) {
                video.write(frame);
                cudabot::imshow("gpu_esdf_mppi_3d", frame);
                cudabot::waitKey(1);
            }
            float dx = st[0] - goal[0], dy = st[1] - goal[1], dz = st[2] - goal[2];
            if (std::sqrt(dx * dx + dy * dy + dz * dz) < GOAL_TOL) ep.reached = true;
            if (ep.reached || step + 1 == max_steps) {
                if (video_on) for (int i = 0; i < 30; i++) video.write(frame);   // hold the last frame
                break;
            }
        }
        if (video_on) {
            video.release();
            std::printf("Video saved to %s\n", avi.c_str());
            cudabot::avi_to_gif(avi, gif, 20, 900);
            std::printf("GIF saved to %s\n", gif.c_str());
        }
        return ep;
    };

    if (n_movers > 0) {
        // Check the local update against a full rebuild on the first mover layout,
        // over voxels where either says the vehicle is inside the clearance band.
        Movers mv = init_movers(seed);
        rebuild_esdf(mv);
        std::vector<float> full(cells), local(cells);
        CUDA_CHECK(cudaMemcpy(full.data(), d_esdf_dyn, cells * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_esdf_dyn, d_esdf, cells * sizeof(float), cudaMemcpyDeviceToDevice));
        prev_windows.clear();
        auto l0 = std::chrono::high_resolution_clock::now();
        local_update_esdf(mv);
        CUDA_CHECK(cudaDeviceSynchronize());
        double local_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - l0).count();
        CUDA_CHECK(cudaMemcpy(local.data(), d_esdf_dyn, cells * sizeof(float), cudaMemcpyDeviceToHost));
        float max_diff = 0.0f;
        size_t band = 0;
        for (size_t v = 0; v < cells; v++) {
            if (std::min(full[v], local[v]) >= ROBOT_R + CLEARANCE) continue;
            band++;
            max_diff = std::max(max_diff, std::fabs(full[v] - local[v]));
        }
        std::printf("Local ESDF update (%d windows): %.3f ms; max |local - full rebuild| %.3f m over %zu voxels in the clearance band\n",
                    n_movers, local_ms, max_diff, band);
    }

    bool ok = true;
    if (trials > 0 && n_movers > 0) {
        // Same mover scenarios (episode seeds) for every mode, so rows are paired.
        std::printf("\n%d movers at %.1fx speed, %d episodes per mode (episode seeds %d..%d)\n",
                    n_movers, mover_speed, trials, seed, seed + trials - 1);
        std::printf("| mode | success | collisions | timeouts | mean steps | mean min clearance (m) | ms per step |\n|---|---:|---:|---:|---:|---:|---:|\n");
        for (int mode = MODE_STATIC; mode <= MODE_REBUILD_LOCAL; mode++) {
            int success = 0, collisions = 0, timeouts = 0;
            double steps_sum = 0, clear_sum = 0, ms_sum = 0;
            for (int t = 0; t < trials; t++) {
                Episode ep = run_episode(mode, seed + t, false);
                const bool collided = ep.min_clearance < 0.0f;
                collisions += collided;
                timeouts += !collided && !ep.reached;
                success += ep.reached && !collided;
                steps_sum += ep.steps; clear_sum += ep.min_clearance; ms_sum += ep.ms / std::max(ep.steps, 1);
                std::printf("  %s seed=%d %s steps=%d min_clearance=%.2f\n", MODE_NAMES[mode], seed + t,
                            collided ? "COLLISION" : (ep.reached ? "success" : "timeout"), ep.steps, ep.min_clearance);
            }
            std::printf("| %s | %d/%d | %d | %d | %.1f | %.2f | %.2f |\n", MODE_NAMES[mode], success, trials, collisions,
                        timeouts, steps_sum / trials, clear_sum / trials, ms_sum / trials);
        }
    } else {
        const int mode = n_movers > 0 ? mode_arg : MODE_STATIC;
        Episode ep = run_episode(mode, seed, write_video);
        std::printf("%s after %d steps (%.1f s); path length %.2f m; min clearance %.2f m (%s)\n",
                    ep.reached ? "Goal reached" : "Goal NOT reached", ep.steps, ep.steps * DT, ep.path_len, ep.min_clearance,
                    ep.min_clearance >= 0.0f ? "collision-free" : "COLLISION");
        std::printf("MPPI: %.3f ms per control step (%d iterations of K=%d, T=%d)\n",
                    ep.ms / std::max(ep.steps, 1), ITERS_PER_STEP, K, T_HORIZON);
        ok = ep.reached && ep.min_clearance >= 0.0f;
    }

    if (d_occ_dyn) CUDA_CHECK(cudaFree(d_occ_dyn));
    if (d_esdf_dyn) CUDA_CHECK(cudaFree(d_esdf_dyn));
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
    return ok ? 0 : 1;
}
