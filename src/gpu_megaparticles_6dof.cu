// gpu_megaparticles_6dof.cu
//
// MegaParticles-style 6-DoF / SE(3) Monte Carlo relocalization.
//
// This pushes the earlier SE(2) MegaParticles demos
// (gpu_megaparticles_stein_mcl.cu, gpu_megaparticles_lsh.cu) up to full 6-DoF
// poses: a flying sensor in a 3D voxel world, range scans scored against a
// precomputed 3D Euclidean distance field, Gauss-Newton-like per-particle SE(3)
// corrections, neighbor-consensus Stein motion, and recovery after a hidden
// kidnap and scan blackout.
//
// In 6-DoF the fixed-grid neighbor bucketing used by the SE(2) demos is no
// longer practical: a dense 6-D grid explodes combinatorially.  This is exactly
// where the explicit p-stable LSH neighbor index from gpu_megaparticles_lsh.cu
// becomes essential rather than optional -- here it hashes the full 6-D pose
// feature (x, y, z, s*rotation-vector) into L random tables, so neighbor
// consensus stays O(L) per particle independent of the pose-space dimension.
//
// Two filter paths run side by side: a local bootstrap MCL that tracks well
// before the kidnap but has no support at the hidden relocated pose, and the
// one-million-particle 6-DoF MegaParticles path that starts globally uniform
// over SE(3) and re-localizes as soon as scans return.
//
// This is a repo-sized SE(3) demonstration of the MegaParticles ideas, not a
// full reproduction of Koide et al.'s system (GICP distribution-to-distribution
// scoring, iterative LSH neighbor lists, full posterior backend).  The honest
// scope is: 6-DoF poses, a 3D distance-field range likelihood, an explicit
// p-stable LSH neighbor consensus, and hidden-kidnap relocalization.
//
// Output: gif/gpu_megaparticles_6dof.gif

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int K_MEGA = 1 << 20;     // 1,048,576 global 6-DoF particles
constexpr int K_LOCAL = 1 << 16;    // 65,536 local bootstrap particles
constexpr int THREADS = 256;
constexpr int N_STEPS = 120;
constexpr int KIDNAP_STEP = 54;
constexpr int OCCLUDE_STEPS = 14;
constexpr int VIDEO_EVERY = 2;
constexpr int STEIN_ITERS = 2;
constexpr int POST_PROP_ITERS = 2;
constexpr float PI_F = 3.14159265358979323846f;

constexpr float WORLD_W = 16.0f;    // x
constexpr float WORLD_H = 12.0f;    // y
constexpr float WORLD_D = 5.0f;     // z
constexpr float GRID_RES = 0.16f;
constexpr int GRID_W = static_cast<int>(WORLD_W / GRID_RES);  // 100
constexpr int GRID_H = static_cast<int>(WORLD_H / GRID_RES);  // 75
constexpr int GRID_D = static_cast<int>(WORLD_D / GRID_RES);  // 31
constexpr float DIST_CLAMP = 2.2f;

constexpr int N_RING = 4;
constexpr int N_AZ = 12;
constexpr int N_SCAN = N_RING * N_AZ;  // 48 range rays
constexpr float MAX_RANGE = 12.0f;

constexpr float DT = 0.18f;
constexpr float DIST_SIGMA = 0.28f;
constexpr float LOCAL_SIGMA_XYZ = 0.05f;
constexpr float LOCAL_SIGMA_ROT = 0.020f;
constexpr float MEGA_SIGMA_XYZ = 0.08f;
constexpr float MEGA_SIGMA_ROT = 0.030f;
constexpr float LIK_TEMP = 0.65f;

// Shared coarse position grid for representative-state readout.
constexpr int RB_X = 32;
constexpr int RB_Y = 24;
constexpr int RB_Z = 12;
constexpr int N_REPBUCK = RB_X * RB_Y * RB_Z;

// p-stable LSH neighbor index over the 6-D pose feature.
constexpr int LSH_K = 3;
constexpr int LSH_L = 8;
constexpr int LSH_FEAT = 6;          // x, y, z, s*rotvec(3)
constexpr float LSH_ROT_SCALE = 1.1f;
constexpr float LSH_R = 0.9f;
constexpr int LSH_HBITS = 14;
constexpr int LSH_NBUCK = 1 << LSH_HBITS;

constexpr int PANEL_W = 430;
constexpr int PANEL_H = 360;
constexpr int XY_H = 264;            // top-down view height
constexpr int XZ_H = PANEL_H - XY_H; // side view strip height
constexpr int INFO_W = 340;
constexpr int FRAME_W = PANEL_W * 2 + INFO_W;
constexpr int FRAME_H = PANEL_H;

__constant__ float c_lsh_a[LSH_L * LSH_K * LSH_FEAT];
__constant__ float c_lsh_b[LSH_L * LSH_K];

struct Box { float x0, y0, z0, x1, y1, z1; };

struct Pose6 {
    float x, y, z;
    float qw, qx, qy, qz;
};

struct StepSummary {
    float local_err = 0.0f;
    float mega_err = 0.0f;
    float local_rot_deg = 0.0f;
    float mega_rot_deg = 0.0f;
    bool visible = true;
    double local_ms = 0.0;
    double mega_ms = 0.0;
};

struct FinalStats {
    float local_post_rmse = 0.0f;
    float mega_post_rmse = 0.0f;
    float final_local_err = 0.0f;
    float final_mega_err = 0.0f;
    float final_mega_rot_deg = 0.0f;
    int mega_reacq_step = -1;
    double local_ms = 0.0;
    double mega_ms = 0.0;
};

__host__ __device__ static inline float clampf(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}

// ---------------------------------------------------------------------------
// Host + device quaternion helpers (unit quaternions, Hamilton convention).
// ---------------------------------------------------------------------------

__host__ __device__ static inline void quat_normalize(float& w, float& x, float& y, float& z) {
    float n = sqrtf(w * w + x * x + y * y + z * z);
    if (n < 1e-12f) { w = 1.0f; x = y = z = 0.0f; return; }
    float inv = 1.0f / n;
    w *= inv; x *= inv; y *= inv; z *= inv;
}

// q = a * b (apply b first, then a)
__host__ __device__ static inline void quat_mul(float aw, float ax, float ay, float az,
                                                float bw, float bx, float by, float bz,
                                                float& ow, float& ox, float& oy, float& oz) {
    ow = aw * bw - ax * bx - ay * by - az * bz;
    ox = aw * bx + ax * bw + ay * bz - az * by;
    oy = aw * by - ax * bz + ay * bw + az * bx;
    oz = aw * bz + ax * by - ay * bx + az * bw;
}

// rotate vector v by quaternion q
__host__ __device__ static inline void quat_rotate(float qw, float qx, float qy, float qz,
                                                   float vx, float vy, float vz,
                                                   float& ox, float& oy, float& oz) {
    // t = 2 * (q_vec x v)
    float tx = 2.0f * (qy * vz - qz * vy);
    float ty = 2.0f * (qz * vx - qx * vz);
    float tz = 2.0f * (qx * vy - qy * vx);
    ox = vx + qw * tx + (qy * tz - qz * ty);
    oy = vy + qw * ty + (qz * tx - qx * tz);
    oz = vz + qw * tz + (qx * ty - qy * tx);
}

// rotate vector v by inverse of q (conjugate)
__host__ __device__ static inline void quat_rotate_inv(float qw, float qx, float qy, float qz,
                                                       float vx, float vy, float vz,
                                                       float& ox, float& oy, float& oz) {
    quat_rotate(qw, -qx, -qy, -qz, vx, vy, vz, ox, oy, oz);
}

// exponential map: rotation vector w -> unit quaternion
__host__ __device__ static inline void quat_exp(float wx, float wy, float wz,
                                                float& ow, float& ox, float& oy, float& oz) {
    float theta = sqrtf(wx * wx + wy * wy + wz * wz);
    if (theta < 1e-7f) {
        ow = 1.0f; ox = 0.5f * wx; oy = 0.5f * wy; oz = 0.5f * wz;
        quat_normalize(ow, ox, oy, oz);
        return;
    }
    float half = 0.5f * theta;
    float s = sinf(half) / theta;
    ow = cosf(half); ox = wx * s; oy = wy * s; oz = wz * s;
}

// logarithm map: unit quaternion -> rotation vector (axis * angle)
__host__ __device__ static inline void quat_log(float qw, float qx, float qy, float qz,
                                                float& wx, float& wy, float& wz) {
    if (qw < 0.0f) { qw = -qw; qx = -qx; qy = -qy; qz = -qz; }
    float vn = sqrtf(qx * qx + qy * qy + qz * qz);
    if (vn < 1e-7f) { wx = 2.0f * qx; wy = 2.0f * qy; wz = 2.0f * qz; return; }
    float angle = 2.0f * atan2f(vn, qw);
    float scale = angle / vn;
    wx = qx * scale; wy = qy * scale; wz = qz * scale;
}

static float quat_angle_between(const Pose6& a, const Pose6& b) {
    // relative quaternion a^-1 * b, angle = 2*acos(|w|)
    float rw, rx, ry, rz;
    quat_mul(a.qw, -a.qx, -a.qy, -a.qz, b.qw, b.qx, b.qy, b.qz, rw, rx, ry, rz);
    return 2.0f * std::acos(std::min(1.0f, std::fabs(rw)));
}

static float pose_dist_xyz(const Pose6& a, const Pose6& b) {
    float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// ---------------------------------------------------------------------------
// 3D scene + occupancy.
// ---------------------------------------------------------------------------

static void stamp_box(std::vector<unsigned char>& occ, const Box& b) {
    int x0 = std::max(0, static_cast<int>(std::floor(b.x0 / GRID_RES)));
    int y0 = std::max(0, static_cast<int>(std::floor(b.y0 / GRID_RES)));
    int z0 = std::max(0, static_cast<int>(std::floor(b.z0 / GRID_RES)));
    int x1 = std::min(GRID_W - 1, static_cast<int>(std::ceil(b.x1 / GRID_RES)));
    int y1 = std::min(GRID_H - 1, static_cast<int>(std::ceil(b.y1 / GRID_RES)));
    int z1 = std::min(GRID_D - 1, static_cast<int>(std::ceil(b.z1 / GRID_RES)));
    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
                occ[(z * GRID_H + y) * GRID_W + x] = 1u;
}

struct CpuMap {
    std::vector<Box> boxes;
    std::vector<unsigned char> occ;
    std::vector<float> dist, gx, gy, gz;
};

static CpuMap make_map() {
    CpuMap m;
    std::vector<Box>& b = m.boxes;
    float t = GRID_RES * 2.0f;
    // Floor and ceiling.
    b.push_back({0, 0, 0, WORLD_W, WORLD_H, 0.35f});
    b.push_back({0, 0, WORLD_D - 0.35f, WORLD_W, WORLD_H, WORLD_D});
    // Outer walls.
    b.push_back({0, 0, 0, t, WORLD_H, WORLD_D});
    b.push_back({WORLD_W - t, 0, 0, WORLD_W, WORLD_H, WORLD_D});
    b.push_back({0, 0, 0, WORLD_W, t, WORLD_D});
    b.push_back({0, WORLD_H - t, 0, WORLD_W, WORLD_H, WORLD_D});
    // Internal pillars (full height) -- vertical structure constrains x,y,yaw.
    b.push_back({4.0f, 3.0f, 0.35f, 4.8f, 3.8f, WORLD_D - 0.35f});
    b.push_back({11.0f, 3.2f, 0.35f, 11.8f, 4.0f, WORLD_D - 0.35f});
    b.push_back({4.4f, 8.6f, 0.35f, 5.2f, 9.4f, WORLD_D - 0.35f});
    b.push_back({11.4f, 8.2f, 0.35f, 12.2f, 9.0f, WORLD_D - 0.35f});
    // Suspended slabs + steps -- structure at varying z constrains z, roll, pitch.
    b.push_back({6.2f, 5.0f, 3.4f, 9.6f, 7.4f, 3.9f});
    b.push_back({1.8f, 5.4f, 1.4f, 2.8f, 6.6f, 2.2f});
    b.push_back({13.0f, 6.0f, 2.2f, 14.0f, 7.2f, 3.0f});
    b.push_back({7.0f, 1.4f, 0.35f, 9.0f, 2.0f, 1.8f});
    b.push_back({7.2f, 10.0f, 2.6f, 9.2f, 10.7f, 3.4f});

    m.occ.assign(static_cast<size_t>(GRID_W) * GRID_H * GRID_D, 0u);
    for (const Box& bx : b) stamp_box(m.occ, bx);
    return m;
}

static bool occ_at(const CpuMap& m, int ix, int iy, int iz) {
    if (ix < 0 || ix >= GRID_W || iy < 0 || iy >= GRID_H || iz < 0 || iz >= GRID_D) return true;
    return m.occ[(iz * GRID_H + iy) * GRID_W + ix] != 0u;
}

static bool is_wall_world(const CpuMap& m, float x, float y, float z) {
    int ix = static_cast<int>(x / GRID_RES);
    int iy = static_cast<int>(y / GRID_RES);
    int iz = static_cast<int>(z / GRID_RES);
    return occ_at(m, ix, iy, iz);
}

// ---------------------------------------------------------------------------
// 3D ESDF via Jump Flooding (GPU) + gradient.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void unflatten3(int idx, int& x, int& y, int& z) {
    x = idx % GRID_W;
    y = (idx / GRID_W) % GRID_H;
    z = idx / (GRID_W * GRID_H);
}

__global__ void jfa_init_kernel(const unsigned char* occ, int* seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    seed[i] = occ[i] ? i : -1;
}

__global__ void jfa_step_kernel(const int* seed_in, int* seed_out, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= GRID_W || y >= GRID_H || z >= GRID_D) return;
    int idx = (z * GRID_H + y) * GRID_W + x;
    int best = seed_in[idx];
    float best_d2 = FLT_MAX;
    if (best >= 0) {
        int bx, by, bz; unflatten3(best, bx, by, bz);
        int ex = x - bx, ey = y - by, ez = z - bz;
        best_d2 = static_cast<float>(ex * ex + ey * ey + ez * ez);
    }
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx * k, ny = y + dy * k, nz = z + dz * k;
                if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H || nz < 0 || nz >= GRID_D) continue;
                int s = seed_in[(nz * GRID_H + ny) * GRID_W + nx];
                if (s < 0) continue;
                int sx, sy, sz; unflatten3(s, sx, sy, sz);
                int ex = x - sx, ey = y - sy, ez = z - sz;
                float d2 = static_cast<float>(ex * ex + ey * ey + ez * ez);
                if (d2 < best_d2) { best_d2 = d2; best = s; }
            }
    seed_out[idx] = best;
}

__global__ void jfa_to_dist_kernel(const int* seed, float* dist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int s = seed[i];
    if (s < 0) { dist[i] = DIST_CLAMP; return; }
    int x, y, z; unflatten3(i, x, y, z);
    int sx, sy, sz; unflatten3(s, sx, sy, sz);
    int ex = x - sx, ey = y - sy, ez = z - sz;
    float d = sqrtf(static_cast<float>(ex * ex + ey * ey + ez * ez)) * GRID_RES;
    dist[i] = fminf(d, DIST_CLAMP);
}

__global__ void grad_kernel(const float* dist, float* gx, float* gy, float* gz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= GRID_W || y >= GRID_H || z >= GRID_D) return;
    int idx = (z * GRID_H + y) * GRID_W + x;
    int x0 = max(0, x - 1), x1 = min(GRID_W - 1, x + 1);
    int y0 = max(0, y - 1), y1 = min(GRID_H - 1, y + 1);
    int z0 = max(0, z - 1), z1 = min(GRID_D - 1, z + 1);
    gx[idx] = (dist[(z * GRID_H + y) * GRID_W + x1] - dist[(z * GRID_H + y) * GRID_W + x0]) /
              ((x1 - x0) * GRID_RES + 1e-6f);
    gy[idx] = (dist[(z * GRID_H + y1) * GRID_W + x] - dist[(z * GRID_H + y0) * GRID_W + x]) /
              ((y1 - y0) * GRID_RES + 1e-6f);
    gz[idx] = (dist[(z1 * GRID_H + y) * GRID_W + x] - dist[(z0 * GRID_H + y) * GRID_W + x]) /
              ((z1 - z0) * GRID_RES + 1e-6f);
}

struct DeviceMap {
    float *dist = nullptr, *gx = nullptr, *gy = nullptr, *gz = nullptr;

    void build(const CpuMap& m) {
        int n = GRID_W * GRID_H * GRID_D;
        unsigned char* d_occ = nullptr;
        int *seed_a = nullptr, *seed_b = nullptr;
        CUDA_CHECK(cudaMalloc(&d_occ, n * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&seed_a, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&seed_b, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dist, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gx, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gy, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gz, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_occ, m.occ.data(), n * sizeof(unsigned char), cudaMemcpyHostToDevice));

        int blocks = (n + THREADS - 1) / THREADS;
        jfa_init_kernel<<<blocks, THREADS>>>(d_occ, seed_a, n);
        CUDA_CHECK(cudaGetLastError());
        dim3 blk(8, 8, 4);
        dim3 grd((GRID_W + 7) / 8, (GRID_H + 7) / 8, (GRID_D + 3) / 4);
        int kmax = 1;
        while (kmax < GRID_W || kmax < GRID_H || kmax < GRID_D) kmax <<= 1;
        int* in_ptr = seed_a; int* out_ptr = seed_b;
        for (int k = kmax / 2; k >= 1; k >>= 1) {
            jfa_step_kernel<<<grd, blk>>>(in_ptr, out_ptr, k);
            CUDA_CHECK(cudaGetLastError());
            std::swap(in_ptr, out_ptr);
        }
        jfa_to_dist_kernel<<<blocks, THREADS>>>(in_ptr, dist, n);
        CUDA_CHECK(cudaGetLastError());
        grad_kernel<<<grd, blk>>>(dist, gx, gy, gz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_occ));
        CUDA_CHECK(cudaFree(seed_a));
        CUDA_CHECK(cudaFree(seed_b));
    }
    void free_all() {
        CUDA_CHECK(cudaFree(dist));
        CUDA_CHECK(cudaFree(gx));
        CUDA_CHECK(cudaFree(gy));
        CUDA_CHECK(cudaFree(gz));
    }
};

__device__ __forceinline__ float sample3(const float* field, float wx, float wy, float wz) {
    int ix = max(0, min(GRID_W - 1, static_cast<int>(wx / GRID_RES)));
    int iy = max(0, min(GRID_H - 1, static_cast<int>(wy / GRID_RES)));
    int iz = max(0, min(GRID_D - 1, static_cast<int>(wz / GRID_RES)));
    return field[(iz * GRID_H + iy) * GRID_W + ix];
}

// ---------------------------------------------------------------------------
// Scan generation (host raycast against occupancy).
// ---------------------------------------------------------------------------

struct ScanPattern {
    std::vector<float> dx, dy, dz;  // unit ray directions in body frame
};

static ScanPattern make_pattern() {
    ScanPattern p;
    for (int r = 0; r < N_RING; ++r) {
        float elev = (-35.0f + 70.0f * r / (N_RING - 1)) * PI_F / 180.0f;
        for (int a = 0; a < N_AZ; ++a) {
            float az = (2.0f * PI_F * a) / N_AZ;
            float ce = std::cos(elev);
            p.dx.push_back(ce * std::cos(az));
            p.dy.push_back(ce * std::sin(az));
            p.dz.push_back(std::sin(elev));
        }
    }
    return p;
}

static float raycast(const CpuMap& m, const Pose6& pose, const ScanPattern& pat, int k) {
    // body-frame ray direction -> world
    float wx, wy, wz;
    quat_rotate(pose.qw, pose.qx, pose.qy, pose.qz, pat.dx[k], pat.dy[k], pat.dz[k], wx, wy, wz);
    for (float r = 0.15f; r < MAX_RANGE; r += 0.05f) {
        float x = pose.x + r * wx, y = pose.y + r * wy, z = pose.z + r * wz;
        if (is_wall_world(m, x, y, z)) return r;
    }
    return MAX_RANGE;
}

// Returns body-frame scan endpoints (range * dir + noise).
static void make_scan(const CpuMap& m, const Pose6& pose, const ScanPattern& pat, int step,
                      std::vector<float>& ex, std::vector<float>& ey, std::vector<float>& ez) {
    ex.resize(N_SCAN); ey.resize(N_SCAN); ez.resize(N_SCAN);
    std::mt19937 rng(9000 + step * 31);
    std::normal_distribution<float> noise(0.0f, 0.04f);
    for (int k = 0; k < N_SCAN; ++k) {
        float r = clampf(raycast(m, pose, pat, k) + noise(rng), 0.2f, MAX_RANGE);
        ex[k] = r * pat.dx[k];
        ey[k] = r * pat.dy[k];
        ez[k] = r * pat.dz[k];
    }
}

// ---------------------------------------------------------------------------
// Particles.
// ---------------------------------------------------------------------------

__global__ void init_rng_kernel(curandState* st, unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &st[i]);
}

__device__ __forceinline__ void random_quat(curandState* s, float& w, float& x, float& y, float& z) {
    // Marsaglia uniform random unit quaternion.
    float u1 = curand_uniform(s), u2 = curand_uniform(s), u3 = curand_uniform(s);
    float s1 = sqrtf(1.0f - u1), s2 = sqrtf(u1);
    w = s1 * sinf(2.0f * PI_F * u2);
    x = s1 * cosf(2.0f * PI_F * u2);
    y = s2 * sinf(2.0f * PI_F * u3);
    z = s2 * cosf(2.0f * PI_F * u3);
}

__global__ void init_uniform_kernel(float* x, float* y, float* z,
                                    float* qw, float* qx, float* qy, float* qz,
                                    float* wgt, curandState* rng, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState s = rng[i];
    x[i] = 0.6f + (WORLD_W - 1.2f) * curand_uniform(&s);
    y[i] = 0.6f + (WORLD_H - 1.2f) * curand_uniform(&s);
    z[i] = 0.7f + (WORLD_D - 1.4f) * curand_uniform(&s);
    float w_, x_, y_, z_;
    random_quat(&s, w_, x_, y_, z_);
    qw[i] = w_; qx[i] = x_; qy[i] = y_; qz[i] = z_;
    wgt[i] = 1.0f / n;
    rng[i] = s;
}

__global__ void init_gaussian_kernel(float* x, float* y, float* z,
                                     float* qw, float* qx, float* qy, float* qz,
                                     float* wgt, curandState* rng, int n, Pose6 pose) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState s = rng[i];
    x[i] = clampf(pose.x + 0.25f * curand_normal(&s), 0.5f, WORLD_W - 0.5f);
    y[i] = clampf(pose.y + 0.25f * curand_normal(&s), 0.5f, WORLD_H - 0.5f);
    z[i] = clampf(pose.z + 0.20f * curand_normal(&s), 0.6f, WORLD_D - 0.6f);
    float dw = 0.10f * curand_normal(&s), dx = 0.10f * curand_normal(&s), dz = 0.10f * curand_normal(&s);
    float ew, ex, ey, ez; quat_exp(dw, dx, dz, ew, ex, ey, ez);
    float ow, ox, oy, oz;
    quat_mul(pose.qw, pose.qx, pose.qy, pose.qz, ew, ex, ey, ez, ow, ox, oy, oz);
    quat_normalize(ow, ox, oy, oz);
    qw[i] = ow; qx[i] = ox; qy[i] = oy; qz[i] = oz;
    wgt[i] = 1.0f / n;
    rng[i] = s;
}

__global__ void predict_kernel(float* x, float* y, float* z,
                               float* qw, float* qx, float* qy, float* qz,
                               curandState* rng, int n,
                               float vx, float vy, float vz,
                               float wx, float wy, float wz,
                               float sxyz, float srot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState s = rng[i];
    // World-frame translation command + noise.
    x[i] = clampf(x[i] + (vx + sxyz * curand_normal(&s)) * DT, 0.45f, WORLD_W - 0.45f);
    y[i] = clampf(y[i] + (vy + sxyz * curand_normal(&s)) * DT, 0.45f, WORLD_H - 0.45f);
    z[i] = clampf(z[i] + (vz + sxyz * curand_normal(&s)) * DT, 0.55f, WORLD_D - 0.55f);
    // Body-frame angular increment + noise.
    float rwx = (wx + srot * curand_normal(&s)) * DT;
    float rwy = (wy + srot * curand_normal(&s)) * DT;
    float rwz = (wz + srot * curand_normal(&s)) * DT;
    float ew, ex, ey, ez; quat_exp(rwx, rwy, rwz, ew, ex, ey, ez);
    float ow, ox, oy, oz;
    quat_mul(qw[i], qx[i], qy[i], qz[i], ew, ex, ey, ez, ow, ox, oy, oz);
    quat_normalize(ow, ox, oy, oz);
    qw[i] = ow; qx[i] = ox; qy[i] = oy; qz[i] = oz;
    rng[i] = s;
}

// SE(3) likelihood + Gauss-Newton-like step (right-perturbation body rotation,
// world-frame translation).
__global__ void likelihood_step_kernel(const float* __restrict__ x,
                                       const float* __restrict__ y,
                                       const float* __restrict__ z,
                                       const float* __restrict__ qw,
                                       const float* __restrict__ qx,
                                       const float* __restrict__ qy,
                                       const float* __restrict__ qz,
                                       const float* __restrict__ scan_x,
                                       const float* __restrict__ scan_y,
                                       const float* __restrict__ scan_z,
                                       const float* __restrict__ dist,
                                       const float* __restrict__ gxf,
                                       const float* __restrict__ gyf,
                                       const float* __restrict__ gzf,
                                       float* __restrict__ score,
                                       float* __restrict__ stx, float* __restrict__ sty, float* __restrict__ stz,
                                       float* __restrict__ srx, float* __restrict__ sry, float* __restrict__ srz,
                                       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float px = x[i], py = y[i], pz = z[i];
    float w_ = qw[i], xq = qx[i], yq = qy[i], zq = qz[i];
    float logp = 0.0f;
    float gtx = 0.0f, gty = 0.0f, gtz = 0.0f;     // translation gradient
    float grx = 0.0f, gry = 0.0f, grz = 0.0f;     // rotation gradient
    float htx = 0.0f, hty = 0.0f, htz = 0.0f;     // translation Hessian diag
    float hrx = 0.0f, hry = 0.0f, hrz = 0.0f;     // rotation Hessian diag
    const float inv_var = 1.0f / (DIST_SIGMA * DIST_SIGMA);

    for (int k = 0; k < N_SCAN; ++k) {
        float ex = scan_x[k], ey = scan_y[k], ez = scan_z[k];
        float rx, ry, rz;
        quat_rotate(w_, xq, yq, zq, ex, ey, ez, rx, ry, rz);  // body endpoint -> world dir
        float wx = px + rx, wy = py + ry, wz = pz + rz;
        bool outside = (wx < 0.0f || wx >= WORLD_W || wy < 0.0f || wy >= WORLD_H ||
                        wz < 0.0f || wz >= WORLD_D);
        float d = outside ? DIST_CLAMP : sample3(dist, wx, wy, wz);
        float ggx = outside ? 0.0f : sample3(gxf, wx, wy, wz);
        float ggy = outside ? 0.0f : sample3(gyf, wx, wy, wz);
        float ggz = outside ? 0.0f : sample3(gzf, wx, wy, wz);
        d = fminf(d, DIST_CLAMP);
        logp += -0.5f * d * d * inv_var;
        // translation gradient (world frame)
        gtx += d * ggx * inv_var; gty += d * ggy * inv_var; gtz += d * ggz * inv_var;
        htx += ggx * ggx * inv_var; hty += ggy * ggy * inv_var; htz += ggz * ggz * inv_var;
        // rotation gradient (body frame): -d * ((R^T g) x e)
        float rgx, rgy, rgz;
        quat_rotate_inv(w_, xq, yq, zq, ggx, ggy, ggz, rgx, rgy, rgz);
        float cx = rgy * ez - rgz * ey;
        float cy = rgz * ex - rgx * ez;
        float cz = rgx * ey - rgy * ex;
        grx += -d * cx * inv_var; gry += -d * cy * inv_var; grz += -d * cz * inv_var;
        hrx += cx * cx * inv_var; hry += cy * cy * inv_var; hrz += cz * cz * inv_var;
    }
    stx[i] = clampf(-gtx / (htx + 0.20f), -0.25f, 0.25f);
    sty[i] = clampf(-gty / (hty + 0.20f), -0.25f, 0.25f);
    stz[i] = clampf(-gtz / (htz + 0.20f), -0.25f, 0.25f);
    srx[i] = clampf(-grx / (hrx + 0.30f), -0.08f, 0.08f);
    sry[i] = clampf(-gry / (hry + 0.30f), -0.08f, 0.08f);
    srz[i] = clampf(-grz / (hrz + 0.30f), -0.08f, 0.08f);
    score[i] = fmaxf(logp, -120.0f);
}

__global__ void posterior_from_score_kernel(const float* score, float* post, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    post[i] = expf(fmaxf(score[i] * LIK_TEMP, -80.0f)) + 1e-20f;
}

// Apply a per-particle SE(3) increment.
__global__ void apply_step_kernel(float* x, float* y, float* z,
                                  float* qw, float* qx, float* qy, float* qz,
                                  const float* dtx, const float* dty, const float* dtz,
                                  const float* drx, const float* dry, const float* drz,
                                  float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = clampf(x[i] + scale * dtx[i], 0.45f, WORLD_W - 0.45f);
    y[i] = clampf(y[i] + scale * dty[i], 0.45f, WORLD_H - 0.45f);
    z[i] = clampf(z[i] + scale * dtz[i], 0.55f, WORLD_D - 0.55f);
    float ew, ex, ey, ez;
    quat_exp(scale * drx[i], scale * dry[i], scale * drz[i], ew, ex, ey, ez);
    float ow, ox, oy, oz;
    quat_mul(qw[i], qx[i], qy[i], qz[i], ew, ex, ey, ez, ow, ox, oy, oz);
    quat_normalize(ow, ox, oy, oz);
    qw[i] = ow; qx[i] = ox; qy[i] = oy; qz[i] = oz;
}

// ---------------------------------------------------------------------------
// Local bootstrap MCL kernels (weights, normalize, systematic resample).
// ---------------------------------------------------------------------------

__global__ void weights_kernel(const float* score, float* wgt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    wgt[i] = expf(fmaxf(score[i] * LIK_TEMP, -80.0f));
}

__global__ void reduce_sum_kernel(const float* v, float* out, int n) {
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    float s = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) s += v[i];
    sh[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[0] = sh[0];
}

__global__ void normalize_kernel(float* wgt, float sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    wgt[i] = (sum > 1e-30f) ? wgt[i] / sum : 1.0f / n;
}

__global__ void cumsum_kernel(const float* wgt, float* wcum, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) { acc += wgt[i]; wcum[i] = acc; }
    wcum[n - 1] = 1.0f;
}

__global__ void resample_kernel(const float* x, const float* y, const float* z,
                                const float* qw, const float* qx, const float* qy, const float* qz,
                                const float* wcum,
                                float* x2, float* y2, float* z2,
                                float* qw2, float* qx2, float* qy2, float* qz2,
                                int n, float offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float target = fminf(0.99999994f, offset + i / static_cast<float>(n));
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (wcum[mid] < target) lo = mid + 1; else hi = mid;
    }
    x2[i] = x[lo]; y2[i] = y[lo]; z2[i] = z[lo];
    qw2[i] = qw[lo]; qx2[i] = qx[lo]; qy2[i] = qy[lo]; qz2[i] = qz[lo];
}

// ---------------------------------------------------------------------------
// p-stable LSH neighbor index over the 6-D pose feature.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void pose_feature(float x, float y, float z,
                                              float qw, float qx, float qy, float qz,
                                              float* f) {
    float wx, wy, wz; quat_log(qw, qx, qy, qz, wx, wy, wz);
    f[0] = x; f[1] = y; f[2] = z;
    f[3] = LSH_ROT_SCALE * wx; f[4] = LSH_ROT_SCALE * wy; f[5] = LSH_ROT_SCALE * wz;
}

__device__ __forceinline__ int lsh_bucket(const float* f, int l) {
    unsigned int key = 2166136261u ^ (static_cast<unsigned int>(l) * 0x9E3779B1u);
    #pragma unroll
    for (int j = 0; j < LSH_K; ++j) {
        const float* a = &c_lsh_a[(l * LSH_K + j) * LSH_FEAT];
        float proj = c_lsh_b[l * LSH_K + j];
        #pragma unroll
        for (int d = 0; d < LSH_FEAT; ++d) proj += a[d] * f[d];
        int bin = static_cast<int>(floorf(proj / LSH_R));
        unsigned int ub = static_cast<unsigned int>(bin + 1048576);
        key = (key ^ ub) * 16777619u;
    }
    return static_cast<int>(key & (LSH_NBUCK - 1));
}

__global__ void lsh_aggregate_kernel(const float* x, const float* y, const float* z,
                                     const float* qw, const float* qx, const float* qy, const float* qz,
                                     const float* stx, const float* sty, const float* stz,
                                     const float* srx, const float* sry, const float* srz,
                                     const float* post,
                                     float* b_tx, float* b_ty, float* b_tz,
                                     float* b_rx, float* b_ry, float* b_rz,
                                     float* b_x, float* b_y, float* b_z,
                                     float* b_post, float* b_count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float f[LSH_FEAT];
    pose_feature(x[i], y[i], z[i], qw[i], qx[i], qy[i], qz[i], f);
    float p = post[i] + 1e-20f;
    for (int l = 0; l < LSH_L; ++l) {
        int b = l * LSH_NBUCK + lsh_bucket(f, l);
        atomicAdd(&b_tx[b], p * stx[i]); atomicAdd(&b_ty[b], p * sty[i]); atomicAdd(&b_tz[b], p * stz[i]);
        atomicAdd(&b_rx[b], p * srx[i]); atomicAdd(&b_ry[b], p * sry[i]); atomicAdd(&b_rz[b], p * srz[i]);
        atomicAdd(&b_x[b], x[i]); atomicAdd(&b_y[b], y[i]); atomicAdd(&b_z[b], z[i]);
        atomicAdd(&b_post[b], p); atomicAdd(&b_count[b], 1.0f);
    }
}

// Stein-style update: combine own GN step with the LSH neighbor-consensus step,
// a mild position repulsion from the neighbor mean, and jitter.
__global__ void lsh_stein_kernel(float* x, float* y, float* z,
                                 float* qw, float* qx, float* qy, float* qz,
                                 curandState* rng,
                                 const float* stx, const float* sty, const float* stz,
                                 const float* srx, const float* sry, const float* srz,
                                 const float* b_tx, const float* b_ty, const float* b_tz,
                                 const float* b_rx, const float* b_ry, const float* b_rz,
                                 const float* b_x, const float* b_y, const float* b_z,
                                 const float* b_post, const float* b_count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float f[LSH_FEAT];
    pose_feature(x[i], y[i], z[i], qw[i], qx[i], qy[i], qz[i], f);
    float atx = 0, aty = 0, atz = 0, arx = 0, ary = 0, arz = 0, mx = 0, my = 0, mz = 0;
    for (int l = 0; l < LSH_L; ++l) {
        int b = l * LSH_NBUCK + lsh_bucket(f, l);
        float mass = fmaxf(b_post[b], 1e-20f);
        float cnt = fmaxf(b_count[b], 1.0f);
        atx += b_tx[b] / mass; aty += b_ty[b] / mass; atz += b_tz[b] / mass;
        arx += b_rx[b] / mass; ary += b_ry[b] / mass; arz += b_rz[b] / mass;
        mx += b_x[b] / cnt; my += b_y[b] / cnt; mz += b_z[b] / cnt;
    }
    float il = 1.0f / LSH_L;
    atx *= il; aty *= il; atz *= il; arx *= il; ary *= il; arz *= il;
    mx *= il; my *= il; mz *= il;
    curandState s = rng[i];
    float dtx = 0.45f * stx[i] + 0.75f * atx + 0.020f * (x[i] - mx) + 0.015f * curand_normal(&s);
    float dty = 0.45f * sty[i] + 0.75f * aty + 0.020f * (y[i] - my) + 0.015f * curand_normal(&s);
    float dtz = 0.45f * stz[i] + 0.75f * atz + 0.020f * (z[i] - mz) + 0.012f * curand_normal(&s);
    float drx = 0.45f * srx[i] + 0.75f * arx + 0.004f * curand_normal(&s);
    float dry = 0.45f * sry[i] + 0.75f * ary + 0.004f * curand_normal(&s);
    float drz = 0.45f * srz[i] + 0.75f * arz + 0.004f * curand_normal(&s);
    x[i] = clampf(x[i] + clampf(dtx, -0.20f, 0.20f), 0.45f, WORLD_W - 0.45f);
    y[i] = clampf(y[i] + clampf(dty, -0.20f, 0.20f), 0.45f, WORLD_H - 0.45f);
    z[i] = clampf(z[i] + clampf(dtz, -0.18f, 0.18f), 0.55f, WORLD_D - 0.55f);
    float ew, ex, ey, ez;
    quat_exp(clampf(drx, -0.07f, 0.07f), clampf(dry, -0.07f, 0.07f), clampf(drz, -0.07f, 0.07f),
             ew, ex, ey, ez);
    float ow, ox, oy, oz;
    quat_mul(qw[i], qx[i], qy[i], qz[i], ew, ex, ey, ez, ow, ox, oy, oz);
    quat_normalize(ow, ox, oy, oz);
    qw[i] = ow; qx[i] = ox; qy[i] = oy; qz[i] = oz;
    rng[i] = s;
}

__global__ void lsh_post_aggregate_kernel(const float* x, const float* y, const float* z,
                                          const float* qw, const float* qx, const float* qy, const float* qz,
                                          const float* post, float* b_post, float* b_count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float f[LSH_FEAT];
    pose_feature(x[i], y[i], z[i], qw[i], qx[i], qy[i], qz[i], f);
    float p = post[i];
    for (int l = 0; l < LSH_L; ++l) {
        int b = l * LSH_NBUCK + lsh_bucket(f, l);
        atomicAdd(&b_post[b], p); atomicAdd(&b_count[b], 1.0f);
    }
}

__global__ void lsh_post_smooth_kernel(const float* x, const float* y, const float* z,
                                       const float* qw, const float* qx, const float* qy, const float* qz,
                                       float* post, const float* b_post, const float* b_count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float f[LSH_FEAT];
    pose_feature(x[i], y[i], z[i], qw[i], qx[i], qy[i], qz[i], f);
    float mean = 0.0f;
    for (int l = 0; l < LSH_L; ++l) {
        int b = l * LSH_NBUCK + lsh_bucket(f, l);
        mean += b_post[b] / fmaxf(b_count[b], 1.0f);
    }
    mean /= LSH_L;
    post[i] = 0.58f * post[i] + 0.42f * mean;
}

// ---------------------------------------------------------------------------
// Representative-state readout (coarse position grid, posterior-weighted mean
// position + sign-aligned mean quaternion in the dominant bucket).
// ---------------------------------------------------------------------------

__device__ __forceinline__ int rep_bucket(float x, float y, float z) {
    int bx = max(0, min(RB_X - 1, static_cast<int>(x / WORLD_W * RB_X)));
    int by = max(0, min(RB_Y - 1, static_cast<int>(y / WORLD_H * RB_Y)));
    int bz = max(0, min(RB_Z - 1, static_cast<int>(z / WORLD_D * RB_Z)));
    return bx + RB_X * (by + RB_Y * bz);
}

__global__ void rep_aggregate_kernel(const float* x, const float* y, const float* z,
                                     const float* qw, const float* qx, const float* qy, const float* qz,
                                     const float* post,
                                     float* b_x, float* b_y, float* b_z,
                                     float* b_qw, float* b_qx, float* b_qy, float* b_qz,
                                     float* b_post, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = rep_bucket(x[i], y[i], z[i]);
    float p = post[i] + 1e-20f;
    atomicAdd(&b_x[b], p * x[i]); atomicAdd(&b_y[b], p * y[i]); atomicAdd(&b_z[b], p * z[i]);
    // sign-align to qw >= 0 hemisphere before averaging
    float qwv = qw[i], qxv = qx[i], qyv = qy[i], qzv = qz[i];
    if (qwv < 0.0f) { qwv = -qwv; qxv = -qxv; qyv = -qyv; qzv = -qzv; }
    atomicAdd(&b_qw[b], p * qwv); atomicAdd(&b_qx[b], p * qxv);
    atomicAdd(&b_qy[b], p * qyv); atomicAdd(&b_qz[b], p * qzv);
    atomicAdd(&b_post[b], p);
}

struct ParticleSet {
    int n = 0;
    float *x, *y, *z, *qw, *qx, *qy, *qz, *w, *score;
    float *stx, *sty, *stz, *srx, *sry, *srz;
    float *x2, *y2, *z2, *qw2, *qx2, *qy2, *qz2, *wcum, *redux;
    curandState* rng;
    std::vector<float> hx, hy, hz;

    void alloc(int n_, unsigned long long seed) {
        n = n_;
        int blocks = (n + THREADS - 1) / THREADS;
        auto m = [&](float** p) { CUDA_CHECK(cudaMalloc(p, n * sizeof(float))); };
        m(&x); m(&y); m(&z); m(&qw); m(&qx); m(&qy); m(&qz); m(&w); m(&score);
        m(&stx); m(&sty); m(&stz); m(&srx); m(&sry); m(&srz);
        m(&x2); m(&y2); m(&z2); m(&qw2); m(&qx2); m(&qy2); m(&qz2); m(&wcum);
        CUDA_CHECK(cudaMalloc(&redux, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rng, n * sizeof(curandState)));
        hx.resize(n); hy.resize(n); hz.resize(n);
        init_rng_kernel<<<blocks, THREADS>>>(rng, seed, n);
        CUDA_CHECK(cudaGetLastError());
    }
    void free_all() {
        for (float* p : {x, y, z, qw, qx, qy, qz, w, score, stx, sty, stz, srx, sry, srz,
                         x2, y2, z2, qw2, qx2, qy2, qz2, wcum, redux})
            CUDA_CHECK(cudaFree(p));
        CUDA_CHECK(cudaFree(rng));
    }
    void copy_pos_host() {
        CUDA_CHECK(cudaMemcpy(hx.data(), x, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hy.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hz.data(), z, n * sizeof(float), cudaMemcpyDeviceToHost));
    }
};

struct LshTables {
    float *tx, *ty, *tz, *rx, *ry, *rz, *bx, *by, *bz, *post, *count;
    size_t total = static_cast<size_t>(LSH_L) * LSH_NBUCK;
    void alloc() {
        size_t bytes = total * sizeof(float);
        for (float** p : {&tx, &ty, &tz, &rx, &ry, &rz, &bx, &by, &bz, &post, &count})
            CUDA_CHECK(cudaMalloc(p, bytes));
    }
    void clear_all() {
        size_t bytes = total * sizeof(float);
        for (float* p : {tx, ty, tz, rx, ry, rz, bx, by, bz, post, count})
            CUDA_CHECK(cudaMemset(p, 0, bytes));
    }
    void clear_post() {
        size_t bytes = total * sizeof(float);
        CUDA_CHECK(cudaMemset(post, 0, bytes));
        CUDA_CHECK(cudaMemset(count, 0, bytes));
    }
    void free_all() {
        for (float* p : {tx, ty, tz, rx, ry, rz, bx, by, bz, post, count}) CUDA_CHECK(cudaFree(p));
    }
};

struct RepBuckets {
    float *x, *y, *z, *qw, *qx, *qy, *qz, *post;
    std::vector<float> hx, hy, hz, hqw, hqx, hqy, hqz, hpost;
    void alloc() {
        size_t bytes = N_REPBUCK * sizeof(float);
        for (float** p : {&x, &y, &z, &qw, &qx, &qy, &qz, &post}) CUDA_CHECK(cudaMalloc(p, bytes));
        hx.resize(N_REPBUCK); hy.resize(N_REPBUCK); hz.resize(N_REPBUCK);
        hqw.resize(N_REPBUCK); hqx.resize(N_REPBUCK); hqy.resize(N_REPBUCK); hqz.resize(N_REPBUCK);
        hpost.resize(N_REPBUCK);
    }
    void clear() {
        size_t bytes = N_REPBUCK * sizeof(float);
        for (float* p : {x, y, z, qw, qx, qy, qz, post}) CUDA_CHECK(cudaMemset(p, 0, bytes));
    }
    void free_all() {
        for (float* p : {x, y, z, qw, qx, qy, qz, post}) CUDA_CHECK(cudaFree(p));
    }
};

static Pose6 representative(ParticleSet& p, RepBuckets& rb) {
    int blocks = (p.n + THREADS - 1) / THREADS;
    rb.clear();
    rep_aggregate_kernel<<<blocks, THREADS>>>(p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz, p.w,
                                              rb.x, rb.y, rb.z, rb.qw, rb.qx, rb.qy, rb.qz, rb.post, p.n);
    CUDA_CHECK(cudaGetLastError());
    auto dl = [&](std::vector<float>& h, float* d) {
        CUDA_CHECK(cudaMemcpy(h.data(), d, N_REPBUCK * sizeof(float), cudaMemcpyDeviceToHost));
    };
    dl(rb.hpost, rb.post); dl(rb.hx, rb.x); dl(rb.hy, rb.y); dl(rb.hz, rb.z);
    dl(rb.hqw, rb.qw); dl(rb.hqx, rb.qx); dl(rb.hqy, rb.qy); dl(rb.hqz, rb.qz);
    int best = 0;
    for (int i = 1; i < N_REPBUCK; ++i) if (rb.hpost[i] > rb.hpost[best]) best = i;
    float mass = std::max(rb.hpost[best], 1e-20f);
    Pose6 out;
    out.x = rb.hx[best] / mass; out.y = rb.hy[best] / mass; out.z = rb.hz[best] / mass;
    float qw = rb.hqw[best], qx = rb.hqx[best], qy = rb.hqy[best], qz = rb.hqz[best];
    quat_normalize(qw, qx, qy, qz);
    out.qw = qw; out.qx = qx; out.qy = qy; out.qz = qz;
    return out;
}

static LshTables make_lsh_params(unsigned int seed) {
    std::vector<float> a(LSH_L * LSH_K * LSH_FEAT), b(LSH_L * LSH_K);
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::uniform_real_distribution<float> u(0.0f, LSH_R);
    for (float& v : a) v = g(rng);
    for (float& v : b) v = u(rng);
    CUDA_CHECK(cudaMemcpyToSymbol(c_lsh_a, a.data(), a.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_lsh_b, b.data(), b.size() * sizeof(float)));
    return LshTables{};
}

// ---------------------------------------------------------------------------
// Filter steps.
// ---------------------------------------------------------------------------

static Pose6 local_step(ParticleSet& p, RepBuckets& rb, const DeviceMap& dm,
                        const float* sx, const float* sy, const float* sz,
                        float vx, float vy, float vz, float wx, float wy, float wz,
                        bool visible, std::mt19937& host_rng) {
    int blocks = (p.n + THREADS - 1) / THREADS;
    predict_kernel<<<blocks, THREADS>>>(p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz, p.rng, p.n,
                                        vx, vy, vz, wx, wy, wz, LOCAL_SIGMA_XYZ, LOCAL_SIGMA_ROT);
    CUDA_CHECK(cudaGetLastError());
    if (!visible) return representative(p, rb);
    likelihood_step_kernel<<<blocks, THREADS>>>(p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz,
                                                sx, sy, sz, dm.dist, dm.gx, dm.gy, dm.gz,
                                                p.score, p.stx, p.sty, p.stz, p.srx, p.sry, p.srz, p.n);
    CUDA_CHECK(cudaGetLastError());
    weights_kernel<<<blocks, THREADS>>>(p.score, p.w, p.n);
    CUDA_CHECK(cudaGetLastError());
    reduce_sum_kernel<<<1, THREADS, THREADS * sizeof(float)>>>(p.w, p.redux, p.n);
    CUDA_CHECK(cudaGetLastError());
    float sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&sum, p.redux, sizeof(float), cudaMemcpyDeviceToHost));
    normalize_kernel<<<blocks, THREADS>>>(p.w, sum, p.n);
    CUDA_CHECK(cudaGetLastError());
    Pose6 est = representative(p, rb);
    cumsum_kernel<<<1, 1>>>(p.w, p.wcum, p.n);
    CUDA_CHECK(cudaGetLastError());
    std::uniform_real_distribution<float> unif(0.0f, 1.0f / p.n);
    resample_kernel<<<blocks, THREADS>>>(p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz, p.wcum,
                                         p.x2, p.y2, p.z2, p.qw2, p.qx2, p.qy2, p.qz2, p.n, unif(host_rng));
    CUDA_CHECK(cudaGetLastError());
    std::swap(p.x, p.x2); std::swap(p.y, p.y2); std::swap(p.z, p.z2);
    std::swap(p.qw, p.qw2); std::swap(p.qx, p.qx2); std::swap(p.qy, p.qy2); std::swap(p.qz, p.qz2);
    return est;
}

static Pose6 mega_step(ParticleSet& p, LshTables& tbl, RepBuckets& rb, const DeviceMap& dm,
                       const float* sx, const float* sy, const float* sz,
                       float vx, float vy, float vz, float wx, float wy, float wz, bool visible) {
    int blocks = (p.n + THREADS - 1) / THREADS;
    if (!visible) {
        // No range scans: the MegaParticles path models lost localization by
        // re-seeding globally uniform over SE(3) (relocalization without a
        // reliable prior), so it can re-localize the instant scans return. The
        // local bootstrap path, by contrast, only diffuses locally and stays
        // tied to the stale pre-kidnap mode.
        init_uniform_kernel<<<blocks, THREADS>>>(p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz, p.w, p.rng, p.n);
        CUDA_CHECK(cudaGetLastError());
        return representative(p, rb);
    }
    predict_kernel<<<blocks, THREADS>>>(p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz, p.rng, p.n,
                                        vx, vy, vz, wx, wy, wz, MEGA_SIGMA_XYZ, MEGA_SIGMA_ROT);
    CUDA_CHECK(cudaGetLastError());
    for (int it = 0; it < STEIN_ITERS; ++it) {
        likelihood_step_kernel<<<blocks, THREADS>>>(p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz,
                                                    sx, sy, sz, dm.dist, dm.gx, dm.gy, dm.gz,
                                                    p.score, p.stx, p.sty, p.stz, p.srx, p.sry, p.srz, p.n);
        CUDA_CHECK(cudaGetLastError());
        posterior_from_score_kernel<<<blocks, THREADS>>>(p.score, p.w, p.n);
        CUDA_CHECK(cudaGetLastError());
        tbl.clear_all();
        lsh_aggregate_kernel<<<blocks, THREADS>>>(p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz,
                                                  p.stx, p.sty, p.stz, p.srx, p.sry, p.srz, p.w,
                                                  tbl.tx, tbl.ty, tbl.tz, tbl.rx, tbl.ry, tbl.rz,
                                                  tbl.bx, tbl.by, tbl.bz, tbl.post, tbl.count, p.n);
        CUDA_CHECK(cudaGetLastError());
        lsh_stein_kernel<<<blocks, THREADS>>>(p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz, p.rng,
                                              p.stx, p.sty, p.stz, p.srx, p.sry, p.srz,
                                              tbl.tx, tbl.ty, tbl.tz, tbl.rx, tbl.ry, tbl.rz,
                                              tbl.bx, tbl.by, tbl.bz, tbl.post, tbl.count, p.n);
        CUDA_CHECK(cudaGetLastError());
    }
    likelihood_step_kernel<<<blocks, THREADS>>>(p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz,
                                                sx, sy, sz, dm.dist, dm.gx, dm.gy, dm.gz,
                                                p.score, p.stx, p.sty, p.stz, p.srx, p.sry, p.srz, p.n);
    CUDA_CHECK(cudaGetLastError());
    posterior_from_score_kernel<<<blocks, THREADS>>>(p.score, p.w, p.n);
    CUDA_CHECK(cudaGetLastError());
    for (int it = 0; it < POST_PROP_ITERS; ++it) {
        tbl.clear_post();
        lsh_post_aggregate_kernel<<<blocks, THREADS>>>(p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz,
                                                       p.w, tbl.post, tbl.count, p.n);
        CUDA_CHECK(cudaGetLastError());
        lsh_post_smooth_kernel<<<blocks, THREADS>>>(p.x, p.y, p.z, p.qw, p.qx, p.qy, p.qz,
                                                    p.w, tbl.post, tbl.count, p.n);
        CUDA_CHECK(cudaGetLastError());
    }
    return representative(p, rb);
}

// ---------------------------------------------------------------------------
// Trajectory.
// ---------------------------------------------------------------------------

static void yaw_pitch_to_quat(float yaw, float pitch, float roll,
                              float& qw, float& qx, float& qy, float& qz) {
    float cy = std::cos(yaw * 0.5f), sy = std::sin(yaw * 0.5f);
    float cp = std::cos(pitch * 0.5f), sp = std::sin(pitch * 0.5f);
    float cr = std::cos(roll * 0.5f), sr = std::sin(roll * 0.5f);
    qw = cr * cp * cy + sr * sp * sy;
    qx = sr * cp * cy - cr * sp * sy;
    qy = cr * sp * cy + sr * cp * sy;
    qz = cr * cp * sy - sr * sp * cy;
}

static std::vector<Pose6> make_truth(std::vector<std::array<float, 6>>& cmd) {
    std::vector<Pose6> truth(N_STEPS);
    cmd.resize(N_STEPS);
    for (int t = 0; t < N_STEPS; ++t) {
        float ph = 0.10f * t;
        Pose6 p;
        if (t < KIDNAP_STEP) {
            p.x = 8.0f + 4.5f * std::cos(ph);
            p.y = 6.0f + 3.4f * std::sin(0.9f * ph);
            p.z = 2.3f + 0.8f * std::sin(0.7f * ph + 0.5f);
        } else {
            float ph2 = 0.10f * (t - KIDNAP_STEP);
            p.x = 4.0f + 3.0f * std::sin(1.1f * ph2 + 0.6f);
            p.y = 9.0f - 2.6f * std::cos(0.8f * ph2);
            p.z = 1.6f + 0.7f * std::sin(0.9f * ph2);
        }
        float yaw = 0.6f * std::sin(0.6f * ph) + (t < KIDNAP_STEP ? 0.0f : 2.2f);
        float pitch = 0.18f * std::sin(0.5f * ph + 0.3f);
        float roll = 0.12f * std::sin(0.45f * ph + 1.1f);
        yaw_pitch_to_quat(yaw, pitch, roll, p.qw, p.qx, p.qy, p.qz);
        truth[t] = p;
    }
    // velocities (world) + body angular rates via finite differences
    for (int t = 0; t < N_STEPS; ++t) {
        int tn = std::min(N_STEPS - 1, t + 1);
        bool boundary = (t == KIDNAP_STEP - 1);
        float vx = boundary ? 0.0f : (truth[tn].x - truth[t].x) / DT;
        float vy = boundary ? 0.0f : (truth[tn].y - truth[t].y) / DT;
        float vz = boundary ? 0.0f : (truth[tn].z - truth[t].z) / DT;
        float wx = 0, wy = 0, wz = 0;
        if (!boundary) {
            float rw, rx, ry, rz;
            quat_mul(truth[t].qw, -truth[t].qx, -truth[t].qy, -truth[t].qz,
                     truth[tn].qw, truth[tn].qx, truth[tn].qy, truth[tn].qz, rw, rx, ry, rz);
            float lx, ly, lz; quat_log(rw, rx, ry, rz, lx, ly, lz);
            wx = lx / DT; wy = ly / DT; wz = lz / DT;
        }
        cmd[t] = {vx, vy, vz, wx, wy, wz};
    }
    return truth;
}

// ---------------------------------------------------------------------------
// Visualization (top-down XY + side XZ strip per panel).
// ---------------------------------------------------------------------------

static cv::Point xy_px(int ox, float x, float y) {
    return cv::Point(ox + static_cast<int>(x / WORLD_W * PANEL_W),
                     static_cast<int>((WORLD_H - y) / WORLD_H * XY_H));
}
static cv::Point xz_px(int ox, float x, float z) {
    return cv::Point(ox + static_cast<int>(x / WORLD_W * PANEL_W),
                     XY_H + static_cast<int>((WORLD_D - z) / WORLD_D * XZ_H));
}

static void draw_pose_marker(cv::Mat& img, int ox, const Pose6& p, const cv::Scalar& col, int rad) {
    cv::Point c = xy_px(ox, p.x, p.y);
    cv::circle(img, c, rad, col, -1, cv::LINE_AA);
    float hx, hy, hz; quat_rotate(p.qw, p.qx, p.qy, p.qz, 1.0f, 0.0f, 0.0f, hx, hy, hz);
    cv::line(img, c, xy_px(ox, p.x + 0.9f * hx, p.y + 0.9f * hy), col, 2, cv::LINE_AA);
    cv::circle(img, xz_px(ox, p.x, p.z), std::max(2, rad - 1), col, -1, cv::LINE_AA);
}

static void draw_panel(cv::Mat& img, int ox, const std::string& title, const CpuMap& map,
                       const std::vector<Pose6>& truth_hist, const std::vector<Pose6>& est_hist,
                       const Pose6& truth, const Pose6& est,
                       const std::vector<float>& hx, const std::vector<float>& hy,
                       const std::vector<float>& hz, int stride,
                       const cv::Scalar& pcol, const cv::Scalar& ecol) {
    cv::rectangle(img, cv::Rect(ox, 0, PANEL_W, XY_H), cv::Scalar(250, 250, 247), -1);
    cv::rectangle(img, cv::Rect(ox, XY_H, PANEL_W, XZ_H), cv::Scalar(238, 240, 243), -1);
    // map boxes: XY footprint + XZ side
    for (const Box& b : map.boxes) {
        cv::rectangle(img, xy_px(ox, b.x0, b.y1), xy_px(ox, b.x1, b.y0), cv::Scalar(70, 76, 84), -1);
        cv::rectangle(img, xz_px(ox, b.x0, b.z1), xz_px(ox, b.x1, b.z0), cv::Scalar(95, 100, 108), -1);
    }
    for (int i = 0; i < static_cast<int>(hx.size()); i += stride) {
        cv::Point a = xy_px(ox, hx[i], hy[i]);
        if (a.x >= ox && a.x < ox + PANEL_W && a.y >= 0 && a.y < XY_H)
            img.at<cv::Vec3b>(a.y, a.x) = cv::Vec3b((uchar)pcol[0], (uchar)pcol[1], (uchar)pcol[2]);
        cv::Point s = xz_px(ox, hx[i], hz[i]);
        if (s.x >= ox && s.x < ox + PANEL_W && s.y >= XY_H && s.y < PANEL_H)
            img.at<cv::Vec3b>(s.y, s.x) = cv::Vec3b((uchar)pcol[0], (uchar)pcol[1], (uchar)pcol[2]);
    }
    for (size_t i = 1; i < truth_hist.size(); ++i)
        cv::line(img, xy_px(ox, truth_hist[i - 1].x, truth_hist[i - 1].y),
                 xy_px(ox, truth_hist[i].x, truth_hist[i].y), cv::Scalar(40, 40, 40), 1, cv::LINE_AA);
    for (size_t i = 1; i < est_hist.size(); ++i)
        cv::line(img, xy_px(ox, est_hist[i - 1].x, est_hist[i - 1].y),
                 xy_px(ox, est_hist[i].x, est_hist[i].y), ecol, 2, cv::LINE_AA);
    draw_pose_marker(img, ox, truth, cv::Scalar(20, 20, 20), 5);
    draw_pose_marker(img, ox, est, ecol, 6);
    cv::putText(img, title, cv::Point(ox + 12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(20, 24, 32), 2, cv::LINE_AA);
    cv::putText(img, "top-down (x,y)", cv::Point(ox + 12, XY_H - 10), cv::FONT_HERSHEY_SIMPLEX, 0.40,
                cv::Scalar(120, 124, 130), 1, cv::LINE_AA);
    cv::putText(img, "side (x,z)", cv::Point(ox + 12, XY_H + 18), cv::FONT_HERSHEY_SIMPLEX, 0.40,
                cv::Scalar(120, 124, 130), 1, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(ox, 0, PANEL_W, PANEL_H), cv::Scalar(205, 205, 200), 1);
    cv::line(img, cv::Point(ox, XY_H), cv::Point(ox + PANEL_W, XY_H), cv::Scalar(205, 205, 200), 1);
}

static void draw_info(cv::Mat& img, int ox, int step, const StepSummary& s,
                      const FinalStats& st, bool occluded) {
    cv::rectangle(img, cv::Rect(ox, 0, INFO_W, FRAME_H), cv::Scalar(244, 246, 246), -1);
    cv::putText(img, "MegaParticles 6-DoF", cv::Point(ox + 16, 32), cv::FONT_HERSHEY_SIMPLEX,
                0.60, cv::Scalar(20, 28, 35), 2, cv::LINE_AA);
    cv::putText(img, "SE(3) relocalization + LSH", cv::Point(ox + 16, 56), cv::FONT_HERSHEY_SIMPLEX,
                0.44, cv::Scalar(70, 78, 88), 1, cv::LINE_AA);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "step %03d / %03d", step, N_STEPS - 1);
    cv::putText(img, buf, cv::Point(ox + 16, 96), cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(30, 36, 44), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "scan: %s", occluded ? "blocked / hidden kidnap" : "visible");
    cv::putText(img, buf, cv::Point(ox + 16, 120), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                occluded ? cv::Scalar(40, 70, 190) : cv::Scalar(40, 120, 80), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "local bootstrap: %.2f m", s.local_err);
    cv::putText(img, buf, cv::Point(ox + 16, 158), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(180, 110, 40), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "mega 6-DoF: %.2f m / %.1f deg", s.mega_err, s.mega_rot_deg);
    cv::putText(img, buf, cv::Point(ox + 16, 182), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(55, 95, 175), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "post-kidnap RMSE:");
    cv::putText(img, buf, cv::Point(ox + 16, 222), cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(45, 50, 58), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "  local %.2f m   mega %.3f m", st.local_post_rmse, st.mega_post_rmse);
    cv::putText(img, buf, cv::Point(ox + 16, 246), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(30, 110, 90), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "mega particles: %d", K_MEGA);
    cv::putText(img, buf, cv::Point(ox + 16, 284), cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "LSH: %d tables x %d proj (6-D)", LSH_L, LSH_K);
    cv::putText(img, buf, cv::Point(ox + 16, 308), cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "avg step: %.1f / %.1f ms", st.local_ms, st.mega_ms);
    cv::putText(img, buf, cv::Point(ox + 16, 332), cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
}

static void ensure_dirs() {
    int rc = std::system("mkdir -p gif tmp");
    if (rc != 0) std::fprintf(stderr, "mkdir rc %d\n", rc);
}

static FinalStats run_demo() {
    ensure_dirs();
    CpuMap map = make_map();
    DeviceMap dm;
    dm.build(map);
    LshTables tbl = make_lsh_params(20240601u);
    tbl.alloc();
    ScanPattern pat = make_pattern();

    float *d_sx, *d_sy, *d_sz;
    CUDA_CHECK(cudaMalloc(&d_sx, N_SCAN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sy, N_SCAN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sz, N_SCAN * sizeof(float)));

    ParticleSet local, mega;
    local.alloc(K_LOCAL, 1234);
    mega.alloc(K_MEGA, 5678);
    RepBuckets rb;
    rb.alloc();

    std::vector<std::array<float, 6>> cmd;
    std::vector<Pose6> truth = make_truth(cmd);

    int lblocks = (K_LOCAL + THREADS - 1) / THREADS;
    int mblocks = (K_MEGA + THREADS - 1) / THREADS;
    init_gaussian_kernel<<<lblocks, THREADS>>>(local.x, local.y, local.z, local.qw, local.qx,
                                               local.qy, local.qz, local.w, local.rng, local.n, truth.front());
    CUDA_CHECK(cudaGetLastError());
    init_uniform_kernel<<<mblocks, THREADS>>>(mega.x, mega.y, mega.z, mega.qw, mega.qx,
                                              mega.qy, mega.qz, mega.w, mega.rng, mega.n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cv::VideoWriter video("tmp/gpu_megaparticles_6dof.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 12, cv::Size(FRAME_W, FRAME_H));
    if (!video.isOpened()) { std::fprintf(stderr, "video open failed\n"); std::exit(1); }

    std::mt19937 host_rng(7);
    std::vector<Pose6> local_hist, mega_hist, truth_hist;
    std::vector<float> ex, ey, ez;
    FinalStats st;
    StepSummary last;
    int post_count = 0;
    float local_sq = 0, mega_sq = 0;
    double local_ms_sum = 0, mega_ms_sum = 0;
    bool mega_has_track = false; Pose6 mega_track{};

    for (int t = 0; t < N_STEPS; ++t) {
        bool visible = !(t >= KIDNAP_STEP && t < KIDNAP_STEP + OCCLUDE_STEPS);
        bool just_unblocked = (t == KIDNAP_STEP + OCCLUDE_STEPS);
        if (visible) {
            make_scan(map, truth[t], pat, t, ex, ey, ez);
            CUDA_CHECK(cudaMemcpy(d_sx, ex.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_sy, ey.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_sz, ez.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));
        }
        const std::array<float, 6>& c = cmd[t];
        auto t0 = std::chrono::high_resolution_clock::now();
        Pose6 local_est = local_step(local, rb, dm, d_sx, d_sy, d_sz,
                                     c[0], c[1], c[2], c[3], c[4], c[5], visible, host_rng);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        Pose6 mega_est = mega_step(mega, tbl, rb, dm, d_sx, d_sy, d_sz,
                                   c[0], c[1], c[2], c[3], c[4], c[5], visible);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();

        // Representative-state continuity gate for the mega path (same idea as
        // the SE(2) demos): allow a global jump right after blackout, otherwise
        // reject implausible jumps relative to the odometry prediction.
        if (!mega_has_track || just_unblocked) {
            mega_track = mega_est; mega_has_track = true;
        } else if (visible) {
            Pose6 pred = mega_track;
            pred.x += c[0] * DT; pred.y += c[1] * DT; pred.z += c[2] * DT;
            float ew, ex2, ey2, ez2; quat_exp(c[3] * DT, c[4] * DT, c[5] * DT, ew, ex2, ey2, ez2);
            float ow, ox2, oy2, oz2;
            quat_mul(pred.qw, pred.qx, pred.qy, pred.qz, ew, ex2, ey2, ez2, ow, ox2, oy2, oz2);
            quat_normalize(ow, ox2, oy2, oz2);
            pred.qw = ow; pred.qx = ox2; pred.qy = oy2; pred.qz = oz2;
            if (pose_dist_xyz(mega_est, pred) > 3.0f) mega_est = pred;
            mega_track = mega_est;
        } else {
            mega_est = mega_track;
        }

        double local_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double mega_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        local_ms_sum += local_ms; mega_ms_sum += mega_ms;
        st.local_ms = local_ms_sum / (t + 1); st.mega_ms = mega_ms_sum / (t + 1);

        last.local_err = pose_dist_xyz(local_est, truth[t]);
        last.mega_err = pose_dist_xyz(mega_est, truth[t]);
        last.local_rot_deg = quat_angle_between(local_est, truth[t]) * 180.0f / PI_F;
        last.mega_rot_deg = quat_angle_between(mega_est, truth[t]) * 180.0f / PI_F;
        last.visible = visible;
        last.local_ms = local_ms; last.mega_ms = mega_ms;

        if (t >= KIDNAP_STEP + OCCLUDE_STEPS) {
            local_sq += last.local_err * last.local_err;
            mega_sq += last.mega_err * last.mega_err;
            post_count++;
            if (st.mega_reacq_step < 0 && last.mega_err < 0.8f)
                st.mega_reacq_step = t - (KIDNAP_STEP + OCCLUDE_STEPS);
        }
        st.final_local_err = last.local_err;
        st.final_mega_err = last.mega_err;
        st.final_mega_rot_deg = last.mega_rot_deg;
        st.local_post_rmse = post_count ? std::sqrt(local_sq / post_count) : 0.0f;
        st.mega_post_rmse = post_count ? std::sqrt(mega_sq / post_count) : 0.0f;

        local_hist.push_back(local_est);
        mega_hist.push_back(mega_est);
        truth_hist.push_back(truth[t]);

        if (t % VIDEO_EVERY == 0 || t == N_STEPS - 1) {
            local.copy_pos_host();
            mega.copy_pos_host();
            cv::Mat frame(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(244, 246, 246));
            draw_panel(frame, 0, "64K bootstrap MCL", map, truth_hist, local_hist, truth[t], local_est,
                       local.hx, local.hy, local.hz, std::max(1, K_LOCAL / 2500),
                       cv::Scalar(185, 195, 230), cv::Scalar(40, 95, 210));
            draw_panel(frame, PANEL_W, "1M 6-DoF MegaParticles + LSH", map, truth_hist, mega_hist, truth[t], mega_est,
                       mega.hx, mega.hy, mega.hz, std::max(1, K_MEGA / 3500),
                       cv::Scalar(190, 215, 200), cv::Scalar(30, 150, 95));
            draw_info(frame, PANEL_W * 2, t, last, st, !visible);
            video.write(frame);
        }

        std::printf("step %3d vis=%d local=%.3f mega=%.3f m / %.1f deg  local=%.1fms mega=%.1fms\n",
                    t, visible ? 1 : 0, last.local_err, last.mega_err, last.mega_rot_deg, local_ms, mega_ms);
    }

    video.release();
    avi_to_gif("tmp/gpu_megaparticles_6dof.avi", "gif/gpu_megaparticles_6dof.gif", 12, 900);

    CUDA_CHECK(cudaFree(d_sx)); CUDA_CHECK(cudaFree(d_sy)); CUDA_CHECK(cudaFree(d_sz));
    tbl.free_all(); rb.free_all(); local.free_all(); mega.free_all(); dm.free_all();
    return st;
}

}  // namespace cudabot

int main() {
    cudabot::FinalStats st = cudabot::run_demo();
    std::printf("\nMegaParticles-style 6-DoF / SE(3) relocalization\n");
    std::printf("post-kidnap RMSE: local bootstrap %.4f m, mega 6-DoF %.4f m\n",
                st.local_post_rmse, st.mega_post_rmse);
    std::printf("final error: local bootstrap %.4f m, mega 6-DoF %.4f m / %.2f deg\n",
                st.final_local_err, st.final_mega_err, st.final_mega_rot_deg);
    std::printf("mega reacquisition after blackout: %d frames\n", st.mega_reacq_step);
    std::printf("avg GPU step: local bootstrap %.3f ms, mega 6-DoF %.3f ms\n", st.local_ms, st.mega_ms);
    std::printf("Wrote gif/gpu_megaparticles_6dof.gif\n");
    return 0;
}
