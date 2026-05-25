// gpu_megaparticles_lsh.cu
//
// MegaParticles-style relocalization with an explicit p-stable LSH neighbor
// list, compared head-to-head against a fixed-grid neighbor bucketing.
//
// The earlier MegaParticles-style demo (gpu_megaparticles_stein_mcl.cu) used a
// single fixed-resolution grid to gather the neighbor statistics that drive the
// Stein-style particle motion.  That grid is a practical stand-in for the
// locality-sensitive hashing neighbor search used in Koide et al.'s line of
// work, but it has a single axis-aligned partition: particles near a cell
// boundary never aggregate with their true neighbors one cell over.
//
// This demo replaces that stand-in with an explicit p-stable (Datar et al.
// 2004) LSH neighbor index: L independent hash tables, each formed from K
// random Gaussian projections of the 4-D pose feature (x, y, s*cos th,
// s*sin th) quantised at bin width r.  Two particles are neighbors if they
// collide in at least one of the L tables.  The random offsets and multiple
// tables recover the cross-boundary neighbors the single grid misses.
//
// Both filter paths are identical except for the neighbor structure: one
// million globally distributed particles, the same range-field likelihood, the
// same Gauss-Newton-like per-particle step, the same posterior smoothing, the
// same representative-state readout, and the same hidden-kidnap blackout.  The
// only independent variable is grid-neighbor vs LSH-neighbor aggregation, so
// the reported neighbor recall (vs brute-force ground truth) and post-kidnap
// relocalization RMSE isolate the contribution of the explicit LSH index.
//
// Output: gif/gpu_megaparticles_lsh.gif

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
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

constexpr int K_MEGA = 1 << 20;
constexpr int THREADS = 256;
constexpr int N_SCAN = 30;
constexpr int N_STEPS = 128;
constexpr int KIDNAP_STEP = 56;
constexpr int OCCLUDE_STEPS = 15;
constexpr int VIDEO_EVERY = 2;
constexpr int STEIN_ITERS = 2;
constexpr int POST_PROP_ITERS = 2;
constexpr float PI_F = 3.14159265358979323846f;
constexpr float WORLD_W = 34.0f;
constexpr float WORLD_H = 24.0f;
constexpr float GRID_RES = 0.10f;
constexpr int GRID_W = static_cast<int>(WORLD_W / GRID_RES);
constexpr int GRID_H = static_cast<int>(WORLD_H / GRID_RES);
constexpr float MAX_RANGE = 18.0f;
constexpr float DT = 0.18f;
constexpr float DIST_SIGMA = 0.30f;
constexpr float MEGA_MOTION_SIGMA_XY = 0.060f;
constexpr float MEGA_MOTION_SIGMA_TH = 0.016f;
constexpr float OCCLUDED_SPREAD_XY = 0.95f;
constexpr float OCCLUDED_SPREAD_TH = 0.22f;
constexpr float LIK_TEMP = 0.72f;

// Fixed-grid neighbor bucketing (the baseline neighbor structure).
constexpr int BUCKET_X = 48;
constexpr int BUCKET_Y = 34;
constexpr int BUCKET_T = 24;
constexpr int N_BUCKETS = BUCKET_X * BUCKET_Y * BUCKET_T;

// p-stable LSH neighbor index (the explicit neighbor structure).
constexpr int LSH_K = 3;                  // Gaussian projections AND-ed per table
constexpr int LSH_L = 8;                  // independent OR tables
constexpr int LSH_FEAT = 4;               // pose feature dims
constexpr float LSH_ANG = 1.6f;           // angular feature scale (cos/sin gain)
constexpr float LSH_R = 0.85f;            // projection bin width
constexpr int LSH_HBITS = 14;
constexpr int LSH_NBUCK = 1 << LSH_HBITS; // hash buckets per table

// Neighbor-recall measurement (host-side, on a sampled particle pool).
constexpr int RECALL_POOL = 4096;
constexpr int RECALL_QUERY = 128;
constexpr float RECALL_RADIUS = LSH_R;    // feature-space neighbor radius

constexpr int PANEL_W = 470;
constexpr int PANEL_H = 360;
constexpr int INFO_W = 330;
constexpr int FRAME_W = PANEL_W * 2 + INFO_W;
constexpr int FRAME_H = PANEL_H;

__constant__ float c_lsh_a[LSH_L * LSH_K * LSH_FEAT];
__constant__ float c_lsh_b[LSH_L * LSH_K];

struct Pose2 {
    float x;
    float y;
    float th;
};

struct Rect {
    float x0;
    float y0;
    float x1;
    float y1;
};

struct StepSummary {
    Pose2 grid_est{};
    Pose2 lsh_est{};
    float grid_err = 0.0f;
    float lsh_err = 0.0f;
    bool scan_visible = true;
    double grid_ms = 0.0;
    double lsh_ms = 0.0;
    float recall_grid = 0.0f;
    float recall_lsh = 0.0f;
};

struct FinalStats {
    float grid_post_rmse = 0.0f;
    float lsh_post_rmse = 0.0f;
    float final_grid_err = 0.0f;
    float final_lsh_err = 0.0f;
    int grid_reacq_step = -1;
    int lsh_reacq_step = -1;
    double grid_ms = 0.0;
    double lsh_ms = 0.0;
    float recall_grid = 0.0f;
    float recall_lsh = 0.0f;
};

__host__ __device__ static inline float clampf(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}

__host__ __device__ static inline float wrap_angle(float a) {
    while (a > PI_F) a -= 2.0f * PI_F;
    while (a < -PI_F) a += 2.0f * PI_F;
    return a;
}

static float pose_error_xy(const Pose2& a, const Pose2& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static Pose2 integrate_pose(Pose2 p, float v, float omega) {
    p.x = clampf(p.x + v * std::cos(p.th) * DT, 0.5f, WORLD_W - 0.5f);
    p.y = clampf(p.y + v * std::sin(p.th) * DT, 0.5f, WORLD_H - 0.5f);
    p.th = wrap_angle(p.th + omega * DT);
    return p;
}

struct CpuMap {
    std::vector<Rect> rects;
    std::vector<float> dist;
    std::vector<float> gx;
    std::vector<float> gy;
    cv::Mat occ;
};

static void add_rect(std::vector<Rect>& rects, float x0, float y0, float x1, float y1) {
    rects.push_back({std::min(x0, x1), std::min(y0, y1), std::max(x0, x1), std::max(y0, y1)});
}

static bool inside_rect(const Rect& r, float x, float y) {
    return x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1;
}

static bool is_wall_world(const std::vector<Rect>& rects, float x, float y) {
    if (x < 0.0f || x > WORLD_W || y < 0.0f || y > WORLD_H) return true;
    for (const Rect& r : rects) {
        if (inside_rect(r, x, y)) return true;
    }
    return false;
}

static CpuMap make_map() {
    CpuMap map;
    std::vector<Rect>& r = map.rects;

    add_rect(r, 0.0f, 0.0f, WORLD_W, 0.25f);
    add_rect(r, 0.0f, WORLD_H - 0.25f, WORLD_W, WORLD_H);
    add_rect(r, 0.0f, 0.0f, 0.25f, WORLD_H);
    add_rect(r, WORLD_W - 0.25f, 0.0f, WORLD_W, WORLD_H);

    add_rect(r, 7.8f, 0.0f, 8.1f, 5.7f);
    add_rect(r, 7.8f, 7.1f, 8.1f, 13.6f);
    add_rect(r, 7.8f, 15.0f, 8.1f, WORLD_H);

    add_rect(r, 16.2f, 0.0f, 16.5f, 4.3f);
    add_rect(r, 16.2f, 6.1f, 16.5f, 16.8f);
    add_rect(r, 16.2f, 18.5f, 16.5f, WORLD_H);

    add_rect(r, 24.2f, 0.0f, 24.5f, 8.8f);
    add_rect(r, 24.2f, 10.6f, 24.5f, 19.1f);
    add_rect(r, 24.2f, 21.1f, 24.5f, WORLD_H);

    add_rect(r, 0.0f, 7.2f, 5.4f, 7.5f);
    add_rect(r, 7.2f, 7.2f, 18.1f, 7.5f);
    add_rect(r, 20.2f, 7.2f, WORLD_W, 7.5f);

    add_rect(r, 5.7f, 15.4f, 12.2f, 15.7f);
    add_rect(r, 14.2f, 15.4f, 27.2f, 15.7f);
    add_rect(r, 29.1f, 15.4f, WORLD_W, 15.7f);

    add_rect(r, 3.3f, 18.8f, 4.7f, 20.0f);
    add_rect(r, 12.5f, 2.3f, 14.0f, 3.5f);
    add_rect(r, 19.5f, 20.2f, 21.8f, 21.0f);
    add_rect(r, 28.7f, 4.2f, 30.2f, 5.7f);
    add_rect(r, 30.0f, 18.6f, 31.8f, 19.1f);

    map.occ = cv::Mat(GRID_H, GRID_W, CV_8U, cv::Scalar(255));
    for (int iy = 0; iy < GRID_H; ++iy) {
        for (int ix = 0; ix < GRID_W; ++ix) {
            float x = (ix + 0.5f) * GRID_RES;
            float y = (iy + 0.5f) * GRID_RES;
            if (is_wall_world(r, x, y)) map.occ.at<unsigned char>(iy, ix) = 0;
        }
    }

    cv::Mat dist_px;
    cv::distanceTransform(map.occ, dist_px, cv::DIST_L2, 5);
    map.dist.resize(GRID_W * GRID_H);
    map.gx.resize(GRID_W * GRID_H);
    map.gy.resize(GRID_W * GRID_H);
    for (int iy = 0; iy < GRID_H; ++iy) {
        for (int ix = 0; ix < GRID_W; ++ix) {
            float d = dist_px.at<float>(iy, ix) * GRID_RES;
            map.dist[iy * GRID_W + ix] = std::min(d, 2.5f);
        }
    }
    for (int iy = 0; iy < GRID_H; ++iy) {
        for (int ix = 0; ix < GRID_W; ++ix) {
            int ix0 = std::max(0, ix - 1), ix1 = std::min(GRID_W - 1, ix + 1);
            int iy0 = std::max(0, iy - 1), iy1 = std::min(GRID_H - 1, iy + 1);
            float dx = (map.dist[iy * GRID_W + ix1] - map.dist[iy * GRID_W + ix0]) /
                       ((ix1 - ix0) * GRID_RES + 1.0e-6f);
            float dy = (map.dist[iy1 * GRID_W + ix] - map.dist[iy0 * GRID_W + ix]) /
                       ((iy1 - iy0) * GRID_RES + 1.0e-6f);
            map.gx[iy * GRID_W + ix] = dx;
            map.gy[iy * GRID_W + ix] = dy;
        }
    }
    return map;
}

static float raycast_range(const std::vector<Rect>& rects, const Pose2& pose, float rel_angle) {
    float a = pose.th + rel_angle;
    for (float r = 0.12f; r < MAX_RANGE; r += 0.045f) {
        float x = pose.x + r * std::cos(a);
        float y = pose.y + r * std::sin(a);
        if (is_wall_world(rects, x, y)) return r;
    }
    return MAX_RANGE;
}

static void make_scan(const std::vector<Rect>& rects,
                      const Pose2& pose,
                      int step,
                      std::vector<float>& sx,
                      std::vector<float>& sy) {
    sx.resize(N_SCAN);
    sy.resize(N_SCAN);
    std::mt19937 rng(8000 + step * 17);
    std::normal_distribution<float> noise(0.0f, 0.035f);
    for (int i = 0; i < N_SCAN; ++i) {
        float frac = (N_SCAN == 1) ? 0.0f : static_cast<float>(i) / (N_SCAN - 1);
        float a = (-132.0f + 264.0f * frac) * PI_F / 180.0f;
        float r = clampf(raycast_range(rects, pose, a) + noise(rng), 0.2f, MAX_RANGE);
        sx[i] = r * std::cos(a);
        sy[i] = r * std::sin(a);
    }
}

__global__ void init_rng_kernel(curandState* states, unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &states[i]);
}

__global__ void init_uniform_kernel(float* x, float* y, float* th, float* w, curandState* rng, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState s = rng[i];
    x[i] = 0.6f + (WORLD_W - 1.2f) * curand_uniform(&s);
    y[i] = 0.6f + (WORLD_H - 1.2f) * curand_uniform(&s);
    th[i] = -PI_F + 2.0f * PI_F * curand_uniform(&s);
    w[i] = 1.0f / n;
    rng[i] = s;
}

__global__ void predict_kernel(float* x,
                               float* y,
                               float* th,
                               curandState* rng,
                               int n,
                               float v,
                               float omega,
                               float xy_sigma,
                               float th_sigma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState s = rng[i];
    float vt = v + xy_sigma * curand_normal(&s);
    float wt = omega + th_sigma * curand_normal(&s);
    float theta = th[i];
    x[i] = clampf(x[i] + vt * cosf(theta) * DT, 0.45f, WORLD_W - 0.45f);
    y[i] = clampf(y[i] + vt * sinf(theta) * DT, 0.45f, WORLD_H - 0.45f);
    th[i] = wrap_angle(theta + wt * DT);
    rng[i] = s;
}

__global__ void occlusion_spread_kernel(float* x, float* y, float* th, curandState* rng, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState s = rng[i];
    x[i] = clampf(x[i] + OCCLUDED_SPREAD_XY * curand_normal(&s), 0.45f, WORLD_W - 0.45f);
    y[i] = clampf(y[i] + OCCLUDED_SPREAD_XY * curand_normal(&s), 0.45f, WORLD_H - 0.45f);
    th[i] = wrap_angle(th[i] + OCCLUDED_SPREAD_TH * curand_normal(&s));
    rng[i] = s;
}

__device__ float sample_nearest(const float* field, float wx, float wy) {
    int ix = static_cast<int>(wx / GRID_RES);
    int iy = static_cast<int>(wy / GRID_RES);
    ix = max(0, min(GRID_W - 1, ix));
    iy = max(0, min(GRID_H - 1, iy));
    return field[iy * GRID_W + ix];
}

__global__ void likelihood_gradient_kernel(const float* __restrict__ x,
                                           const float* __restrict__ y,
                                           const float* __restrict__ th,
                                           const float* __restrict__ scan_x,
                                           const float* __restrict__ scan_y,
                                           const float* __restrict__ dist,
                                           const float* __restrict__ grad_x,
                                           const float* __restrict__ grad_y,
                                           float* __restrict__ score,
                                           float* __restrict__ step_x,
                                           float* __restrict__ step_y,
                                           float* __restrict__ step_th,
                                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float px = x[i], py = y[i], pth = th[i];
    float c = cosf(pth), s = sinf(pth);
    float logp = 0.0f;
    float gx_acc = 0.0f, gy_acc = 0.0f, gt_acc = 0.0f;
    float hxx = 0.0f, hyy = 0.0f, htt = 0.0f;
    const float inv_var = 1.0f / (DIST_SIGMA * DIST_SIGMA);

    #pragma unroll
    for (int k = 0; k < N_SCAN; ++k) {
        float sx = scan_x[k];
        float sy = scan_y[k];
        float wx = px + c * sx - s * sy;
        float wy = py + s * sx + c * sy;
        bool outside = (wx < 0.0f || wx >= WORLD_W || wy < 0.0f || wy >= WORLD_H);
        float d = outside ? 2.5f : sample_nearest(dist, wx, wy);
        float gxf = outside ? ((wx < 0.0f) ? -1.0f : ((wx >= WORLD_W) ? 1.0f : 0.0f))
                            : sample_nearest(grad_x, wx, wy);
        float gyf = outside ? ((wy < 0.0f) ? -1.0f : ((wy >= WORLD_H) ? 1.0f : 0.0f))
                            : sample_nearest(grad_y, wx, wy);
        d = fminf(d, 2.5f);
        float dthx = -s * sx - c * sy;
        float dthy = c * sx - s * sy;
        float jt = gxf * dthx + gyf * dthy;
        logp += -0.5f * d * d * inv_var;
        gx_acc += d * gxf * inv_var;
        gy_acc += d * gyf * inv_var;
        gt_acc += d * jt * inv_var;
        hxx += gxf * gxf * inv_var;
        hyy += gyf * gyf * inv_var;
        htt += jt * jt * inv_var;
    }
    float dx = -gx_acc / (hxx + 0.20f);
    float dy = -gy_acc / (hyy + 0.20f);
    float dt = -gt_acc / (htt + 0.40f);
    step_x[i] = clampf(dx, -0.22f, 0.22f);
    step_y[i] = clampf(dy, -0.22f, 0.22f);
    step_th[i] = clampf(dt, -0.060f, 0.060f);
    score[i] = fmaxf(logp, -90.0f);
}

__global__ void posterior_from_score_kernel(const float* score, float* posterior, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    posterior[i] = expf(fmaxf(score[i] * LIK_TEMP, -80.0f)) + 1.0e-18f;
}

// ---------------------------------------------------------------------------
// Fixed-grid neighbor structure (baseline).
// ---------------------------------------------------------------------------

__device__ int grid_bucket(float x, float y, float th) {
    int bx = max(0, min(BUCKET_X - 1, static_cast<int>(x / WORLD_W * BUCKET_X)));
    int by = max(0, min(BUCKET_Y - 1, static_cast<int>(y / WORLD_H * BUCKET_Y)));
    float tn = (wrap_angle(th) + PI_F) / (2.0f * PI_F);
    int bt = max(0, min(BUCKET_T - 1, static_cast<int>(tn * BUCKET_T)));
    return bx + BUCKET_X * (by + BUCKET_Y * bt);
}

__global__ void grid_motion_aggregate_kernel(const float* x,
                                             const float* y,
                                             const float* th,
                                             const float* step_x,
                                             const float* step_y,
                                             const float* step_th,
                                             const float* posterior,
                                             float* b_step_x,
                                             float* b_step_y,
                                             float* b_step_th,
                                             float* b_x,
                                             float* b_y,
                                             float* b_sin,
                                             float* b_cos,
                                             float* b_post,
                                             float* b_count,
                                             int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = grid_bucket(x[i], y[i], th[i]);
    float p = posterior[i] + 1.0e-18f;
    atomicAdd(&b_step_x[b], p * step_x[i]);
    atomicAdd(&b_step_y[b], p * step_y[i]);
    atomicAdd(&b_step_th[b], p * step_th[i]);
    atomicAdd(&b_x[b], x[i]);
    atomicAdd(&b_y[b], y[i]);
    atomicAdd(&b_sin[b], sinf(th[i]));
    atomicAdd(&b_cos[b], cosf(th[i]));
    atomicAdd(&b_post[b], p);
    atomicAdd(&b_count[b], 1.0f);
}

__global__ void grid_stein_update_kernel(float* x,
                                         float* y,
                                         float* th,
                                         curandState* rng,
                                         const float* step_x,
                                         const float* step_y,
                                         const float* step_th,
                                         const float* b_step_x,
                                         const float* b_step_y,
                                         const float* b_step_th,
                                         const float* b_x,
                                         const float* b_y,
                                         const float* b_sin,
                                         const float* b_cos,
                                         const float* b_post,
                                         const float* b_count,
                                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = grid_bucket(x[i], y[i], th[i]);
    float cnt = fmaxf(b_count[b], 1.0f);
    float mass = fmaxf(b_post[b], 1.0e-18f);
    float avg_dx = b_step_x[b] / mass;
    float avg_dy = b_step_y[b] / mass;
    float avg_dt = b_step_th[b] / mass;
    float mx = b_x[b] / cnt;
    float my = b_y[b] / cnt;
    float mt = atan2f(b_sin[b] / cnt, b_cos[b] / cnt);
    float rep_x = 0.020f * (x[i] - mx);
    float rep_y = 0.020f * (y[i] - my);
    float rep_t = 0.006f * wrap_angle(th[i] - mt);
    curandState s = rng[i];
    float jitter_x = 0.012f * curand_normal(&s);
    float jitter_y = 0.012f * curand_normal(&s);
    float jitter_t = 0.003f * curand_normal(&s);
    float dx = 0.45f * step_x[i] + 0.75f * avg_dx + rep_x + jitter_x;
    float dy = 0.45f * step_y[i] + 0.75f * avg_dy + rep_y + jitter_y;
    float dt = 0.45f * step_th[i] + 0.75f * avg_dt + rep_t + jitter_t;
    x[i] = clampf(x[i] + clampf(dx, -0.18f, 0.18f), 0.45f, WORLD_W - 0.45f);
    y[i] = clampf(y[i] + clampf(dy, -0.18f, 0.18f), 0.45f, WORLD_H - 0.45f);
    th[i] = wrap_angle(th[i] + clampf(dt, -0.050f, 0.050f));
    rng[i] = s;
}

__global__ void grid_posterior_aggregate_kernel(const float* x,
                                                const float* y,
                                                const float* th,
                                                const float* posterior,
                                                float* b_post,
                                                float* b_count,
                                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = grid_bucket(x[i], y[i], th[i]);
    atomicAdd(&b_post[b], posterior[i]);
    atomicAdd(&b_count[b], 1.0f);
}

__global__ void grid_posterior_smooth_kernel(const float* x,
                                             const float* y,
                                             const float* th,
                                             float* posterior,
                                             const float* b_post,
                                             const float* b_count,
                                             int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = grid_bucket(x[i], y[i], th[i]);
    float mean = b_post[b] / fmaxf(b_count[b], 1.0f);
    posterior[i] = 0.58f * posterior[i] + 0.42f * mean;
}

// ---------------------------------------------------------------------------
// p-stable LSH neighbor structure (explicit neighbor list).
// ---------------------------------------------------------------------------

__device__ int lsh_bucket(float f0, float f1, float f2, float f3, int l) {
    unsigned int key = 2166136261u ^ (static_cast<unsigned int>(l) * 0x9E3779B1u);
    #pragma unroll
    for (int j = 0; j < LSH_K; ++j) {
        const float* a = &c_lsh_a[(l * LSH_K + j) * LSH_FEAT];
        float proj = a[0] * f0 + a[1] * f1 + a[2] * f2 + a[3] * f3 + c_lsh_b[l * LSH_K + j];
        int bin = static_cast<int>(floorf(proj / LSH_R));
        unsigned int ub = static_cast<unsigned int>(bin + 1048576);
        key = (key ^ ub) * 16777619u;
    }
    return static_cast<int>(key & (LSH_NBUCK - 1));
}

__global__ void lsh_motion_aggregate_kernel(const float* x,
                                            const float* y,
                                            const float* th,
                                            const float* step_x,
                                            const float* step_y,
                                            const float* step_th,
                                            const float* posterior,
                                            float* b_step_x,
                                            float* b_step_y,
                                            float* b_step_th,
                                            float* b_x,
                                            float* b_y,
                                            float* b_sin,
                                            float* b_cos,
                                            float* b_post,
                                            float* b_count,
                                            int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float thi = th[i];
    float ct = cosf(thi), st = sinf(thi);
    float f0 = x[i], f1 = y[i], f2 = LSH_ANG * ct, f3 = LSH_ANG * st;
    float p = posterior[i] + 1.0e-18f;
    float psx = p * step_x[i], psy = p * step_y[i], pst = p * step_th[i];
    for (int l = 0; l < LSH_L; ++l) {
        int b = l * LSH_NBUCK + lsh_bucket(f0, f1, f2, f3, l);
        atomicAdd(&b_step_x[b], psx);
        atomicAdd(&b_step_y[b], psy);
        atomicAdd(&b_step_th[b], pst);
        atomicAdd(&b_x[b], f0);
        atomicAdd(&b_y[b], f1);
        atomicAdd(&b_sin[b], st);
        atomicAdd(&b_cos[b], ct);
        atomicAdd(&b_post[b], p);
        atomicAdd(&b_count[b], 1.0f);
    }
}

__global__ void lsh_stein_update_kernel(float* x,
                                        float* y,
                                        float* th,
                                        curandState* rng,
                                        const float* step_x,
                                        const float* step_y,
                                        const float* step_th,
                                        const float* b_step_x,
                                        const float* b_step_y,
                                        const float* b_step_th,
                                        const float* b_x,
                                        const float* b_y,
                                        const float* b_sin,
                                        const float* b_cos,
                                        const float* b_post,
                                        const float* b_count,
                                        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x[i], yi = y[i], thi = th[i];
    float ct = cosf(thi), st = sinf(thi);
    float f0 = xi, f1 = yi, f2 = LSH_ANG * ct, f3 = LSH_ANG * st;
    // Average the per-table neighbor statistics (union over the L OR tables).
    float avg_dx = 0.0f, avg_dy = 0.0f, avg_dt = 0.0f;
    float mx = 0.0f, my = 0.0f, msin = 0.0f, mcos = 0.0f;
    for (int l = 0; l < LSH_L; ++l) {
        int b = l * LSH_NBUCK + lsh_bucket(f0, f1, f2, f3, l);
        float cnt = fmaxf(b_count[b], 1.0f);
        float mass = fmaxf(b_post[b], 1.0e-18f);
        avg_dx += b_step_x[b] / mass;
        avg_dy += b_step_y[b] / mass;
        avg_dt += b_step_th[b] / mass;
        mx += b_x[b] / cnt;
        my += b_y[b] / cnt;
        msin += b_sin[b] / cnt;
        mcos += b_cos[b] / cnt;
    }
    float inv_l = 1.0f / LSH_L;
    avg_dx *= inv_l;
    avg_dy *= inv_l;
    avg_dt *= inv_l;
    mx *= inv_l;
    my *= inv_l;
    float mt = atan2f(msin, mcos);
    float rep_x = 0.020f * (xi - mx);
    float rep_y = 0.020f * (yi - my);
    float rep_t = 0.006f * wrap_angle(thi - mt);
    curandState s = rng[i];
    float jitter_x = 0.012f * curand_normal(&s);
    float jitter_y = 0.012f * curand_normal(&s);
    float jitter_t = 0.003f * curand_normal(&s);
    float dx = 0.45f * step_x[i] + 0.75f * avg_dx + rep_x + jitter_x;
    float dy = 0.45f * step_y[i] + 0.75f * avg_dy + rep_y + jitter_y;
    float dt = 0.45f * step_th[i] + 0.75f * avg_dt + rep_t + jitter_t;
    x[i] = clampf(xi + clampf(dx, -0.18f, 0.18f), 0.45f, WORLD_W - 0.45f);
    y[i] = clampf(yi + clampf(dy, -0.18f, 0.18f), 0.45f, WORLD_H - 0.45f);
    th[i] = wrap_angle(thi + clampf(dt, -0.050f, 0.050f));
    rng[i] = s;
}

__global__ void lsh_posterior_aggregate_kernel(const float* x,
                                               const float* y,
                                               const float* th,
                                               const float* posterior,
                                               float* b_post,
                                               float* b_count,
                                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float thi = th[i];
    float f0 = x[i], f1 = y[i], f2 = LSH_ANG * cosf(thi), f3 = LSH_ANG * sinf(thi);
    float p = posterior[i];
    for (int l = 0; l < LSH_L; ++l) {
        int b = l * LSH_NBUCK + lsh_bucket(f0, f1, f2, f3, l);
        atomicAdd(&b_post[b], p);
        atomicAdd(&b_count[b], 1.0f);
    }
}

__global__ void lsh_posterior_smooth_kernel(const float* x,
                                            const float* y,
                                            const float* th,
                                            float* posterior,
                                            const float* b_post,
                                            const float* b_count,
                                            int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float thi = th[i];
    float f0 = x[i], f1 = y[i], f2 = LSH_ANG * cosf(thi), f3 = LSH_ANG * sinf(thi);
    float mean = 0.0f;
    for (int l = 0; l < LSH_L; ++l) {
        int b = l * LSH_NBUCK + lsh_bucket(f0, f1, f2, f3, l);
        mean += b_post[b] / fmaxf(b_count[b], 1.0f);
    }
    mean /= LSH_L;
    posterior[i] = 0.58f * posterior[i] + 0.42f * mean;
}

// ---------------------------------------------------------------------------
// Shared representative-state readout (coarse posterior grid, identical for
// both neighbor structures so the only independent variable stays the Stein
// neighbor aggregation).
// ---------------------------------------------------------------------------

__global__ void representative_aggregate_kernel(const float* x,
                                                const float* y,
                                                const float* th,
                                                const float* posterior,
                                                float* b_x,
                                                float* b_y,
                                                float* b_sin,
                                                float* b_cos,
                                                float* b_post,
                                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = grid_bucket(x[i], y[i], th[i]);
    float p = posterior[i] + 1.0e-18f;
    atomicAdd(&b_x[b], p * x[i]);
    atomicAdd(&b_y[b], p * y[i]);
    atomicAdd(&b_sin[b], p * sinf(th[i]));
    atomicAdd(&b_cos[b], p * cosf(th[i]));
    atomicAdd(&b_post[b], p);
}

struct ParticleSet {
    int n = 0;
    float *x = nullptr, *y = nullptr, *th = nullptr;
    float *w = nullptr, *score = nullptr, *step_x = nullptr, *step_y = nullptr, *step_th = nullptr;
    curandState* rng = nullptr;
    std::vector<float> hx, hy, hth;

    void alloc(int n_, unsigned long long seed) {
        n = n_;
        int blocks = (n + THREADS - 1) / THREADS;
        CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&y, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&th, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&w, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&score, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&step_x, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&step_y, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&step_th, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rng, n * sizeof(curandState)));
        hx.resize(n);
        hy.resize(n);
        hth.resize(n);
        init_rng_kernel<<<blocks, THREADS>>>(rng, seed, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void free_all() {
        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
        CUDA_CHECK(cudaFree(th));
        CUDA_CHECK(cudaFree(w));
        CUDA_CHECK(cudaFree(score));
        CUDA_CHECK(cudaFree(step_x));
        CUDA_CHECK(cudaFree(step_y));
        CUDA_CHECK(cudaFree(step_th));
        CUDA_CHECK(cudaFree(rng));
    }

    void copy_pose_to_host() {
        CUDA_CHECK(cudaMemcpy(hx.data(), x, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hy.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hth.data(), th, n * sizeof(float), cudaMemcpyDeviceToHost));
    }
};

struct GridBuckets {
    float *step_x = nullptr, *step_y = nullptr, *step_th = nullptr;
    float *x = nullptr, *y = nullptr, *sin_th = nullptr, *cos_th = nullptr;
    float *post = nullptr, *count = nullptr;

    void alloc() {
        size_t bytes = static_cast<size_t>(N_BUCKETS) * sizeof(float);
        for (float** p : {&step_x, &step_y, &step_th, &x, &y, &sin_th, &cos_th, &post, &count}) {
            CUDA_CHECK(cudaMalloc(p, bytes));
        }
    }
    void clear_all() {
        size_t bytes = static_cast<size_t>(N_BUCKETS) * sizeof(float);
        for (float* p : {step_x, step_y, step_th, x, y, sin_th, cos_th, post, count}) {
            CUDA_CHECK(cudaMemset(p, 0, bytes));
        }
    }
    void clear_post() {
        size_t bytes = static_cast<size_t>(N_BUCKETS) * sizeof(float);
        CUDA_CHECK(cudaMemset(post, 0, bytes));
        CUDA_CHECK(cudaMemset(count, 0, bytes));
    }
    void free_all() {
        for (float* p : {step_x, step_y, step_th, x, y, sin_th, cos_th, post, count}) {
            CUDA_CHECK(cudaFree(p));
        }
    }
};

struct LshTables {
    float *step_x = nullptr, *step_y = nullptr, *step_th = nullptr;
    float *x = nullptr, *y = nullptr, *sin_th = nullptr, *cos_th = nullptr;
    float *post = nullptr, *count = nullptr;
    size_t total = static_cast<size_t>(LSH_L) * LSH_NBUCK;

    void alloc() {
        size_t bytes = total * sizeof(float);
        for (float** p : {&step_x, &step_y, &step_th, &x, &y, &sin_th, &cos_th, &post, &count}) {
            CUDA_CHECK(cudaMalloc(p, bytes));
        }
    }
    void clear_all() {
        size_t bytes = total * sizeof(float);
        for (float* p : {step_x, step_y, step_th, x, y, sin_th, cos_th, post, count}) {
            CUDA_CHECK(cudaMemset(p, 0, bytes));
        }
    }
    void clear_post() {
        size_t bytes = total * sizeof(float);
        CUDA_CHECK(cudaMemset(post, 0, bytes));
        CUDA_CHECK(cudaMemset(count, 0, bytes));
    }
    void free_all() {
        for (float* p : {step_x, step_y, step_th, x, y, sin_th, cos_th, post, count}) {
            CUDA_CHECK(cudaFree(p));
        }
    }
};

struct DeviceMap {
    float *dist = nullptr, *gx = nullptr, *gy = nullptr;
    void upload(const CpuMap& map) {
        size_t bytes = GRID_W * GRID_H * sizeof(float);
        CUDA_CHECK(cudaMalloc(&dist, bytes));
        CUDA_CHECK(cudaMalloc(&gx, bytes));
        CUDA_CHECK(cudaMalloc(&gy, bytes));
        CUDA_CHECK(cudaMemcpy(dist, map.dist.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gx, map.gx.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gy, map.gy.data(), bytes, cudaMemcpyHostToDevice));
    }
    void free_all() {
        CUDA_CHECK(cudaFree(dist));
        CUDA_CHECK(cudaFree(gx));
        CUDA_CHECK(cudaFree(gy));
    }
};

// Representative pose = posterior-weighted mean of the highest-mass coarse-grid
// bucket.  Uses the GridBuckets scratch (post/x/y/sin/cos) for both paths.
static Pose2 representative_pose(ParticleSet& p, GridBuckets& g) {
    int blocks = (p.n + THREADS - 1) / THREADS;
    g.clear_all();
    representative_aggregate_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.w,
                                                         g.x, g.y, g.sin_th, g.cos_th, g.post, p.n);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> h_post(N_BUCKETS), h_x(N_BUCKETS), h_y(N_BUCKETS), h_sin(N_BUCKETS), h_cos(N_BUCKETS);
    CUDA_CHECK(cudaMemcpy(h_post.data(), g.post, N_BUCKETS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_x.data(), g.x, N_BUCKETS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y.data(), g.y, N_BUCKETS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sin.data(), g.sin_th, N_BUCKETS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cos.data(), g.cos_th, N_BUCKETS * sizeof(float), cudaMemcpyDeviceToHost));
    int best = 0;
    for (int i = 1; i < N_BUCKETS; ++i) {
        if (h_post[i] > h_post[best]) best = i;
    }
    float mass = std::max(h_post[best], 1.0e-18f);
    return {h_x[best] / mass, h_y[best] / mass,
            std::atan2(h_sin[best] / mass, h_cos[best] / mass)};
}

// ---------------------------------------------------------------------------
// Host-side neighbor-recall measurement on a sampled particle pool.
// ---------------------------------------------------------------------------

struct LshHostParams {
    std::vector<float> a;  // LSH_L * LSH_K * LSH_FEAT
    std::vector<float> b;  // LSH_L * LSH_K
};

static LshHostParams make_lsh_params(unsigned int seed) {
    LshHostParams p;
    p.a.resize(LSH_L * LSH_K * LSH_FEAT);
    p.b.resize(LSH_L * LSH_K);
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> unif(0.0f, LSH_R);
    for (float& v : p.a) v = gauss(rng);
    for (float& v : p.b) v = unif(rng);
    CUDA_CHECK(cudaMemcpyToSymbol(c_lsh_a, p.a.data(), p.a.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_lsh_b, p.b.data(), p.b.size() * sizeof(float)));
    return p;
}

static int host_grid_bucket(float x, float y, float th) {
    int bx = std::max(0, std::min(BUCKET_X - 1, static_cast<int>(x / WORLD_W * BUCKET_X)));
    int by = std::max(0, std::min(BUCKET_Y - 1, static_cast<int>(y / WORLD_H * BUCKET_Y)));
    float tn = (wrap_angle(th) + PI_F) / (2.0f * PI_F);
    int bt = std::max(0, std::min(BUCKET_T - 1, static_cast<int>(tn * BUCKET_T)));
    return bx + BUCKET_X * (by + BUCKET_Y * bt);
}

static int host_lsh_bucket(const LshHostParams& lp, float x, float y, float th, int l) {
    float f0 = x, f1 = y, f2 = LSH_ANG * std::cos(th), f3 = LSH_ANG * std::sin(th);
    unsigned int key = 2166136261u ^ (static_cast<unsigned int>(l) * 0x9E3779B1u);
    for (int j = 0; j < LSH_K; ++j) {
        const float* a = &lp.a[(l * LSH_K + j) * LSH_FEAT];
        float proj = a[0] * f0 + a[1] * f1 + a[2] * f2 + a[3] * f3 + lp.b[l * LSH_K + j];
        int bin = static_cast<int>(std::floor(proj / LSH_R));
        unsigned int ub = static_cast<unsigned int>(bin + 1048576);
        key = (key ^ ub) * 16777619u;
    }
    return static_cast<int>(key & (LSH_NBUCK - 1));
}

// Returns true if the running recall average updated (pool had real neighbors).
static bool measure_recall(const ParticleSet& p,
                           const LshHostParams& lp,
                           float& recall_grid,
                           float& recall_lsh) {
    int pool = std::min(RECALL_POOL, p.n);
    int nq = std::min(RECALL_QUERY, pool);
    // Precompute grid + LSH keys for the pool.
    std::vector<int> gkey(pool);
    std::vector<int> lkey(static_cast<size_t>(pool) * LSH_L);
    for (int i = 0; i < pool; ++i) {
        gkey[i] = host_grid_bucket(p.hx[i], p.hy[i], p.hth[i]);
        for (int l = 0; l < LSH_L; ++l) {
            lkey[static_cast<size_t>(i) * LSH_L + l] = host_lsh_bucket(lp, p.hx[i], p.hy[i], p.hth[i], l);
        }
    }
    double sum_grid = 0.0, sum_lsh = 0.0;
    int counted = 0;
    float r2 = RECALL_RADIUS * RECALL_RADIUS;
    for (int q = 0; q < nq; ++q) {
        float qx = p.hx[q], qy = p.hy[q];
        float qcs = LSH_ANG * std::cos(p.hth[q]), qsn = LSH_ANG * std::sin(p.hth[q]);
        int neighbors = 0, hit_grid = 0, hit_lsh = 0;
        for (int j = 0; j < pool; ++j) {
            if (j == q) continue;
            float dx = qx - p.hx[j];
            float dy = qy - p.hy[j];
            float dc = qcs - LSH_ANG * std::cos(p.hth[j]);
            float ds = qsn - LSH_ANG * std::sin(p.hth[j]);
            if (dx * dx + dy * dy + dc * dc + ds * ds > r2) continue;
            neighbors++;
            if (gkey[j] == gkey[q]) hit_grid++;
            bool lsh_hit = false;
            for (int l = 0; l < LSH_L; ++l) {
                if (lkey[static_cast<size_t>(j) * LSH_L + l] == lkey[static_cast<size_t>(q) * LSH_L + l]) {
                    lsh_hit = true;
                    break;
                }
            }
            if (lsh_hit) hit_lsh++;
        }
        if (neighbors > 0) {
            sum_grid += static_cast<double>(hit_grid) / neighbors;
            sum_lsh += static_cast<double>(hit_lsh) / neighbors;
            counted++;
        }
    }
    if (counted == 0) return false;
    recall_grid = static_cast<float>(sum_grid / counted);
    recall_lsh = static_cast<float>(sum_lsh / counted);
    return true;
}

// ---------------------------------------------------------------------------
// Filter steps.
// ---------------------------------------------------------------------------

static Pose2 grid_filter_step(ParticleSet& p,
                              GridBuckets& g,
                              const DeviceMap& dmap,
                              const float* d_scan_x,
                              const float* d_scan_y,
                              float v,
                              float omega,
                              bool visible) {
    int blocks = (p.n + THREADS - 1) / THREADS;
    predict_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng, p.n, v, omega,
                                        MEGA_MOTION_SIGMA_XY, MEGA_MOTION_SIGMA_TH);
    CUDA_CHECK(cudaGetLastError());
    if (!visible) {
        occlusion_spread_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng, p.n);
        CUDA_CHECK(cudaGetLastError());
        return representative_pose(p, g);
    }
    for (int it = 0; it < STEIN_ITERS; ++it) {
        likelihood_gradient_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, d_scan_x, d_scan_y,
                                                        dmap.dist, dmap.gx, dmap.gy,
                                                        p.score, p.step_x, p.step_y, p.step_th, p.n);
        CUDA_CHECK(cudaGetLastError());
        posterior_from_score_kernel<<<blocks, THREADS>>>(p.score, p.w, p.n);
        CUDA_CHECK(cudaGetLastError());
        g.clear_all();
        grid_motion_aggregate_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.step_x, p.step_y, p.step_th,
                                                          p.w, g.step_x, g.step_y, g.step_th,
                                                          g.x, g.y, g.sin_th, g.cos_th,
                                                          g.post, g.count, p.n);
        CUDA_CHECK(cudaGetLastError());
        grid_stein_update_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng,
                                                      p.step_x, p.step_y, p.step_th,
                                                      g.step_x, g.step_y, g.step_th,
                                                      g.x, g.y, g.sin_th, g.cos_th,
                                                      g.post, g.count, p.n);
        CUDA_CHECK(cudaGetLastError());
    }
    likelihood_gradient_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, d_scan_x, d_scan_y,
                                                    dmap.dist, dmap.gx, dmap.gy,
                                                    p.score, p.step_x, p.step_y, p.step_th, p.n);
    CUDA_CHECK(cudaGetLastError());
    posterior_from_score_kernel<<<blocks, THREADS>>>(p.score, p.w, p.n);
    CUDA_CHECK(cudaGetLastError());
    for (int it = 0; it < POST_PROP_ITERS; ++it) {
        g.clear_post();
        grid_posterior_aggregate_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.w, g.post, g.count, p.n);
        CUDA_CHECK(cudaGetLastError());
        grid_posterior_smooth_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.w, g.post, g.count, p.n);
        CUDA_CHECK(cudaGetLastError());
    }
    return representative_pose(p, g);
}

static Pose2 lsh_filter_step(ParticleSet& p,
                             LshTables& tbl,
                             GridBuckets& rep,
                             const DeviceMap& dmap,
                             const float* d_scan_x,
                             const float* d_scan_y,
                             float v,
                             float omega,
                             bool visible) {
    int blocks = (p.n + THREADS - 1) / THREADS;
    predict_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng, p.n, v, omega,
                                        MEGA_MOTION_SIGMA_XY, MEGA_MOTION_SIGMA_TH);
    CUDA_CHECK(cudaGetLastError());
    if (!visible) {
        occlusion_spread_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng, p.n);
        CUDA_CHECK(cudaGetLastError());
        return representative_pose(p, rep);
    }
    for (int it = 0; it < STEIN_ITERS; ++it) {
        likelihood_gradient_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, d_scan_x, d_scan_y,
                                                        dmap.dist, dmap.gx, dmap.gy,
                                                        p.score, p.step_x, p.step_y, p.step_th, p.n);
        CUDA_CHECK(cudaGetLastError());
        posterior_from_score_kernel<<<blocks, THREADS>>>(p.score, p.w, p.n);
        CUDA_CHECK(cudaGetLastError());
        tbl.clear_all();
        lsh_motion_aggregate_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.step_x, p.step_y, p.step_th,
                                                         p.w, tbl.step_x, tbl.step_y, tbl.step_th,
                                                         tbl.x, tbl.y, tbl.sin_th, tbl.cos_th,
                                                         tbl.post, tbl.count, p.n);
        CUDA_CHECK(cudaGetLastError());
        lsh_stein_update_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng,
                                                     p.step_x, p.step_y, p.step_th,
                                                     tbl.step_x, tbl.step_y, tbl.step_th,
                                                     tbl.x, tbl.y, tbl.sin_th, tbl.cos_th,
                                                     tbl.post, tbl.count, p.n);
        CUDA_CHECK(cudaGetLastError());
    }
    likelihood_gradient_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, d_scan_x, d_scan_y,
                                                    dmap.dist, dmap.gx, dmap.gy,
                                                    p.score, p.step_x, p.step_y, p.step_th, p.n);
    CUDA_CHECK(cudaGetLastError());
    posterior_from_score_kernel<<<blocks, THREADS>>>(p.score, p.w, p.n);
    CUDA_CHECK(cudaGetLastError());
    for (int it = 0; it < POST_PROP_ITERS; ++it) {
        tbl.clear_post();
        lsh_posterior_aggregate_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.w, tbl.post, tbl.count, p.n);
        CUDA_CHECK(cudaGetLastError());
        lsh_posterior_smooth_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.w, tbl.post, tbl.count, p.n);
        CUDA_CHECK(cudaGetLastError());
    }
    return representative_pose(p, rep);
}

// ---------------------------------------------------------------------------
// Visualization.
// ---------------------------------------------------------------------------

static void draw_pose(cv::Mat& img, int ox, const Pose2& p, const cv::Scalar& color, int radius) {
    auto to_px = [&](float x, float y) {
        int px = ox + static_cast<int>(x / WORLD_W * PANEL_W);
        int py = static_cast<int>((WORLD_H - y) / WORLD_H * PANEL_H);
        return cv::Point(px, py);
    };
    cv::Point c = to_px(p.x, p.y);
    cv::circle(img, c, radius, color, -1, cv::LINE_AA);
    cv::Point h = to_px(p.x + 0.9f * std::cos(p.th), p.y + 0.9f * std::sin(p.th));
    cv::line(img, c, h, color, 2, cv::LINE_AA);
}

static void draw_panel(cv::Mat& img,
                       int ox,
                       const std::string& title,
                       const CpuMap& map,
                       const std::vector<Pose2>& truth_hist,
                       const std::vector<Pose2>& est_hist,
                       const Pose2& truth,
                       const Pose2& est,
                       const std::vector<float>& px,
                       const std::vector<float>& py,
                       int stride,
                       const cv::Scalar& particle_color,
                       const cv::Scalar& est_color) {
    cv::rectangle(img, cv::Rect(ox, 0, PANEL_W, PANEL_H), cv::Scalar(250, 250, 247), -1);
    for (const Rect& r : map.rects) {
        int x0 = ox + static_cast<int>(r.x0 / WORLD_W * PANEL_W);
        int x1 = ox + static_cast<int>(r.x1 / WORLD_W * PANEL_W);
        int y0 = static_cast<int>((WORLD_H - r.y1) / WORLD_H * PANEL_H);
        int y1 = static_cast<int>((WORLD_H - r.y0) / WORLD_H * PANEL_H);
        cv::rectangle(img, cv::Rect(cv::Point(x0, y0), cv::Point(x1 + 1, y1 + 1)),
                      cv::Scalar(58, 64, 72), -1);
    }
    for (int i = 0; i < static_cast<int>(px.size()); i += stride) {
        int x = ox + static_cast<int>(px[i] / WORLD_W * PANEL_W);
        int y = static_cast<int>((WORLD_H - py[i]) / WORLD_H * PANEL_H);
        if (x >= ox && x < ox + PANEL_W && y >= 0 && y < PANEL_H) {
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<unsigned char>(particle_color[0]),
                                                static_cast<unsigned char>(particle_color[1]),
                                                static_cast<unsigned char>(particle_color[2]));
        }
    }
    auto to_px = [&](const Pose2& p) {
        return cv::Point(ox + static_cast<int>(p.x / WORLD_W * PANEL_W),
                         static_cast<int>((WORLD_H - p.y) / WORLD_H * PANEL_H));
    };
    for (size_t i = 1; i < truth_hist.size(); ++i) {
        cv::line(img, to_px(truth_hist[i - 1]), to_px(truth_hist[i]), cv::Scalar(40, 40, 40), 1, cv::LINE_AA);
    }
    for (size_t i = 1; i < est_hist.size(); ++i) {
        cv::line(img, to_px(est_hist[i - 1]), to_px(est_hist[i]), est_color, 2, cv::LINE_AA);
    }
    draw_pose(img, ox, truth, cv::Scalar(20, 20, 20), 5);
    draw_pose(img, ox, est, est_color, 6);
    cv::putText(img, title, cv::Point(ox + 14, 28), cv::FONT_HERSHEY_SIMPLEX, 0.58,
                cv::Scalar(20, 24, 32), 2, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(ox, 0, PANEL_W, PANEL_H), cv::Scalar(210, 210, 205), 1);
}

static void draw_info(cv::Mat& img, int ox, int step, const StepSummary& s,
                      const FinalStats& partial, bool occluded) {
    cv::rectangle(img, cv::Rect(ox, 0, INFO_W, FRAME_H), cv::Scalar(244, 246, 246), -1);
    cv::putText(img, "MegaParticles + LSH", cv::Point(ox + 18, 34), cv::FONT_HERSHEY_SIMPLEX,
                0.62, cv::Scalar(20, 28, 35), 2, cv::LINE_AA);
    cv::putText(img, "p-stable neighbor index", cv::Point(ox + 18, 58), cv::FONT_HERSHEY_SIMPLEX,
                0.46, cv::Scalar(70, 78, 88), 1, cv::LINE_AA);

    char buf[256];
    std::snprintf(buf, sizeof(buf), "step %03d / %03d", step, N_STEPS - 1);
    cv::putText(img, buf, cv::Point(ox + 18, 96), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(30, 36, 44), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "scan: %s", occluded ? "blocked / hidden kidnap" : "visible");
    cv::putText(img, buf, cv::Point(ox + 18, 120), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                occluded ? cv::Scalar(40, 70, 190) : cv::Scalar(40, 120, 80), 1, cv::LINE_AA);

    std::snprintf(buf, sizeof(buf), "grid-neighbor err: %.2f m", s.grid_err);
    cv::putText(img, buf, cv::Point(ox + 18, 158), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(180, 110, 40), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "LSH-neighbor err: %.2f m", s.lsh_err);
    cv::putText(img, buf, cv::Point(ox + 18, 182), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(55, 95, 175), 1, cv::LINE_AA);

    std::snprintf(buf, sizeof(buf), "neighbor recall (true kNN):");
    cv::putText(img, buf, cv::Point(ox + 18, 222), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(45, 50, 58), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "  grid %.0f%%   LSH %.0f%%",
                  100.0f * partial.recall_grid, 100.0f * partial.recall_lsh);
    cv::putText(img, buf, cv::Point(ox + 18, 246), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(30, 110, 90), 1, cv::LINE_AA);

    std::snprintf(buf, sizeof(buf), "particles: %d x2", K_MEGA);
    cv::putText(img, buf, cv::Point(ox + 18, 284), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "LSH: %d tables x %d proj", LSH_L, LSH_K);
    cv::putText(img, buf, cv::Point(ox + 18, 308), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "avg step: %.2f / %.2f ms", partial.grid_ms, partial.lsh_ms);
    cv::putText(img, buf, cv::Point(ox + 18, 332), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
}

static std::vector<Pose2> make_truth(std::vector<float>& v_cmd, std::vector<float>& w_cmd) {
    std::vector<Pose2> truth(N_STEPS);
    v_cmd.resize(N_STEPS);
    w_cmd.resize(N_STEPS);
    Pose2 p{4.2f, 4.6f, 0.18f};
    for (int t = 0; t < N_STEPS; ++t) {
        if (t == KIDNAP_STEP) {
            p = {27.7f, 18.2f, -2.55f};
        }
        float v = 0.88f + 0.10f * std::sin(0.08f * t);
        float w = 0.28f * std::sin(0.105f * t + 0.4f);
        if (t > 24 && t < 42) w += 0.28f;
        if (t > 78 && t < 94) w -= 0.23f;
        v_cmd[t] = v;
        w_cmd[t] = w;
        truth[t] = p;
        p = integrate_pose(p, v, w);
    }
    return truth;
}

static void ensure_dirs() {
    int rc = std::system("mkdir -p gif tmp");
    if (rc != 0) std::fprintf(stderr, "mkdir failed with code %d\n", rc);
}

static FinalStats run_demo() {
    ensure_dirs();
    CpuMap map = make_map();
    DeviceMap dmap;
    dmap.upload(map);
    LshHostParams lp = make_lsh_params(20240517u);

    float *d_scan_x = nullptr, *d_scan_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scan_x, N_SCAN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_y, N_SCAN * sizeof(float)));

    ParticleSet grid_set, lsh_set;
    grid_set.alloc(K_MEGA, 5678);
    lsh_set.alloc(K_MEGA, 9012);
    GridBuckets grid_buckets, rep_buckets;
    grid_buckets.alloc();
    rep_buckets.alloc();
    LshTables lsh_tables;
    lsh_tables.alloc();

    std::vector<float> v_cmd, w_cmd;
    std::vector<Pose2> truth = make_truth(v_cmd, w_cmd);

    int mega_blocks = (K_MEGA + THREADS - 1) / THREADS;
    init_uniform_kernel<<<mega_blocks, THREADS>>>(grid_set.x, grid_set.y, grid_set.th,
                                                  grid_set.w, grid_set.rng, grid_set.n);
    CUDA_CHECK(cudaGetLastError());
    init_uniform_kernel<<<mega_blocks, THREADS>>>(lsh_set.x, lsh_set.y, lsh_set.th,
                                                  lsh_set.w, lsh_set.rng, lsh_set.n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cv::VideoWriter video("tmp/gpu_megaparticles_lsh.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 12,
                          cv::Size(FRAME_W, FRAME_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open output video\n");
        std::exit(1);
    }

    std::vector<Pose2> grid_hist, lsh_hist, truth_hist;
    std::vector<float> scan_x, scan_y;
    FinalStats stats;
    StepSummary last;
    int post_count = 0;
    float grid_post_sq = 0.0f, lsh_post_sq = 0.0f;
    double grid_ms_sum = 0.0, lsh_ms_sum = 0.0;
    double recall_grid_sum = 0.0, recall_lsh_sum = 0.0;
    int recall_count = 0;
    bool grid_has_track = false, lsh_has_track = false;
    Pose2 grid_track{}, lsh_track{};

    for (int t = 0; t < N_STEPS; ++t) {
        bool visible = !(t >= KIDNAP_STEP && t < KIDNAP_STEP + OCCLUDE_STEPS);
        bool just_unblocked = (t == KIDNAP_STEP + OCCLUDE_STEPS);
        if (visible) {
            make_scan(map.rects, truth[t], t, scan_x, scan_y);
            CUDA_CHECK(cudaMemcpy(d_scan_x, scan_x.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_scan_y, scan_y.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        Pose2 grid_est = grid_filter_step(grid_set, grid_buckets, dmap, d_scan_x, d_scan_y,
                                           v_cmd[t], w_cmd[t], visible);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        Pose2 lsh_est = lsh_filter_step(lsh_set, lsh_tables, rep_buckets, dmap, d_scan_x, d_scan_y,
                                        v_cmd[t], w_cmd[t], visible);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();

        // Identical representative-state continuity gate for both paths.
        auto gate = [&](bool& has_track, Pose2& track, Pose2 est) -> Pose2 {
            if (!has_track || just_unblocked) {
                track = est;
                has_track = true;
                return est;
            }
            Pose2 predicted = integrate_pose(track, v_cmd[t], w_cmd[t]);
            if (visible && pose_error_xy(est, predicted) > 2.6f) est = predicted;
            track = visible ? est : predicted;
            return visible ? est : track;
        };
        grid_est = gate(grid_has_track, grid_track, grid_est);
        lsh_est = gate(lsh_has_track, lsh_track, lsh_est);

        double grid_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double lsh_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        grid_ms_sum += grid_ms;
        lsh_ms_sum += lsh_ms;
        stats.grid_ms = grid_ms_sum / (t + 1);
        stats.lsh_ms = lsh_ms_sum / (t + 1);

        // Neighbor recall on the LSH particle pool (both indices scored on the
        // same particle population for a fair comparison).
        if (visible) {
            lsh_set.copy_pose_to_host();
            float rg = 0.0f, rl = 0.0f;
            if (measure_recall(lsh_set, lp, rg, rl)) {
                recall_grid_sum += rg;
                recall_lsh_sum += rl;
                recall_count++;
                last.recall_grid = rg;
                last.recall_lsh = rl;
            }
        }
        stats.recall_grid = recall_count ? static_cast<float>(recall_grid_sum / recall_count) : 0.0f;
        stats.recall_lsh = recall_count ? static_cast<float>(recall_lsh_sum / recall_count) : 0.0f;

        last.grid_est = grid_est;
        last.lsh_est = lsh_est;
        last.grid_err = pose_error_xy(grid_est, truth[t]);
        last.lsh_err = pose_error_xy(lsh_est, truth[t]);
        last.scan_visible = visible;
        last.grid_ms = grid_ms;
        last.lsh_ms = lsh_ms;

        if (t >= KIDNAP_STEP + OCCLUDE_STEPS) {
            grid_post_sq += last.grid_err * last.grid_err;
            lsh_post_sq += last.lsh_err * last.lsh_err;
            post_count++;
            if (stats.grid_reacq_step < 0 && last.grid_err < 0.65f)
                stats.grid_reacq_step = t - (KIDNAP_STEP + OCCLUDE_STEPS);
            if (stats.lsh_reacq_step < 0 && last.lsh_err < 0.65f)
                stats.lsh_reacq_step = t - (KIDNAP_STEP + OCCLUDE_STEPS);
        }
        stats.final_grid_err = last.grid_err;
        stats.final_lsh_err = last.lsh_err;
        stats.grid_post_rmse = post_count ? std::sqrt(grid_post_sq / post_count) : 0.0f;
        stats.lsh_post_rmse = post_count ? std::sqrt(lsh_post_sq / post_count) : 0.0f;

        grid_hist.push_back(grid_est);
        lsh_hist.push_back(lsh_est);
        truth_hist.push_back(truth[t]);

        if (t % VIDEO_EVERY == 0 || t == N_STEPS - 1) {
            grid_set.copy_pose_to_host();
            cv::Mat frame(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(244, 246, 246));
            draw_panel(frame, 0, "1M grid-neighbor SPF", map, truth_hist, grid_hist, truth[t], grid_est,
                       grid_set.hx, grid_set.hy, std::max(1, K_MEGA / 3600),
                       cv::Scalar(190, 205, 230), cv::Scalar(40, 120, 210));
            draw_panel(frame, PANEL_W, "1M LSH-neighbor SPF", map, truth_hist, lsh_hist, truth[t], lsh_est,
                       lsh_set.hx, lsh_set.hy, std::max(1, K_MEGA / 3600),
                       cv::Scalar(190, 215, 200), cv::Scalar(30, 150, 95));
            draw_info(frame, PANEL_W * 2, t, last, stats, !visible);
            video.write(frame);
        }

        std::printf("step %3d visible=%d grid_err=%.3f lsh_err=%.3f recall(g/l)=%.2f/%.2f grid=%.2fms lsh=%.2fms\n",
                    t, visible ? 1 : 0, last.grid_err, last.lsh_err,
                    last.recall_grid, last.recall_lsh, grid_ms, lsh_ms);
    }

    video.release();
    avi_to_gif("tmp/gpu_megaparticles_lsh.avi", "gif/gpu_megaparticles_lsh.gif", 12, 900);

    CUDA_CHECK(cudaFree(d_scan_x));
    CUDA_CHECK(cudaFree(d_scan_y));
    grid_buckets.free_all();
    rep_buckets.free_all();
    lsh_tables.free_all();
    grid_set.free_all();
    lsh_set.free_all();
    dmap.free_all();
    return stats;
}

}  // namespace cudabot

int main() {
    cudabot::FinalStats stats = cudabot::run_demo();
    std::printf("\nMegaParticles-style SE(2) relocalization: grid vs explicit p-stable LSH neighbors\n");
    std::printf("neighbor recall vs brute-force kNN: grid %.1f%%, LSH %.1f%%\n",
                100.0f * stats.recall_grid, 100.0f * stats.recall_lsh);
    std::printf("post-kidnap RMSE: grid-neighbor %.4f m, LSH-neighbor %.4f m\n",
                stats.grid_post_rmse, stats.lsh_post_rmse);
    std::printf("final error: grid-neighbor %.4f m, LSH-neighbor %.4f m\n",
                stats.final_grid_err, stats.final_lsh_err);
    std::printf("reacquisition after blackout: grid %d frames, LSH %d frames\n",
                stats.grid_reacq_step, stats.lsh_reacq_step);
    std::printf("avg GPU step: grid-neighbor %.4f ms, LSH-neighbor %.4f ms\n",
                stats.grid_ms, stats.lsh_ms);
    std::printf("Wrote gif/gpu_megaparticles_lsh.gif\n");
    return 0;
}
