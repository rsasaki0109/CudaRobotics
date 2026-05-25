// gpu_megaparticles_stein_mcl.cu
//
// Compact MegaParticles-inspired localization demo.
//
// This is not a full reproduction of Koide et al.'s 6-DoF system.  It keeps
// the core ideas visible in a repo-sized SE(2) demo:
//   * one million globally distributed particles,
//   * range-scan likelihood against a precomputed distance field,
//   * Gauss-Newton-like particle motion from likelihood gradients,
//   * sparse bucket-neighbor Stein-style attraction/repulsion,
//   * posterior smoothing on the same dynamic neighbor buckets,
//   * recovery after a hidden kidnap and scan blackout.
//
// Output: gif/gpu_megaparticles_stein_mcl.gif

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int K_MEGA = 1 << 20;
constexpr int K_LOCAL = 1 << 16;
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
constexpr float LOCAL_MOTION_SIGMA_XY = 0.035f;
constexpr float LOCAL_MOTION_SIGMA_TH = 0.010f;
constexpr float MEGA_MOTION_SIGMA_XY = 0.060f;
constexpr float MEGA_MOTION_SIGMA_TH = 0.016f;
constexpr float OCCLUDED_SPREAD_XY = 0.95f;
constexpr float OCCLUDED_SPREAD_TH = 0.22f;
constexpr float LIK_TEMP = 0.72f;
constexpr int BUCKET_X = 48;
constexpr int BUCKET_Y = 34;
constexpr int BUCKET_T = 24;
constexpr int N_BUCKETS = BUCKET_X * BUCKET_Y * BUCKET_T;
constexpr int PANEL_W = 470;
constexpr int PANEL_H = 360;
constexpr int INFO_W = 330;
constexpr int FRAME_W = PANEL_W * 2 + INFO_W;
constexpr int FRAME_H = PANEL_H;

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
    Pose2 local{};
    Pose2 mega{};
    float local_err = 0.0f;
    float mega_err = 0.0f;
    float mega_score = 0.0f;
    bool scan_visible = true;
    double local_ms = 0.0;
    double mega_ms = 0.0;
};

struct FinalStats {
    float local_post_rmse = 0.0f;
    float mega_post_rmse = 0.0f;
    float final_local_err = 0.0f;
    float final_mega_err = 0.0f;
    int mega_reacq_step = -1;
    double local_ms = 0.0;
    double mega_ms = 0.0;
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

__global__ void init_gaussian_kernel(float* x,
                                     float* y,
                                     float* th,
                                     float* w,
                                     curandState* rng,
                                     int n,
                                     Pose2 pose) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState s = rng[i];
    x[i] = clampf(pose.x + 0.22f * curand_normal(&s), 0.5f, WORLD_W - 0.5f);
    y[i] = clampf(pose.y + 0.22f * curand_normal(&s), 0.5f, WORLD_H - 0.5f);
    th[i] = wrap_angle(pose.th + 0.10f * curand_normal(&s));
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

__global__ void weights_from_score_kernel(const float* score, float* weights, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    weights[i] = expf(fmaxf(score[i] * LIK_TEMP, -80.0f));
}

__global__ void posterior_from_score_kernel(const float* score, float* posterior, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    posterior[i] = expf(fmaxf(score[i] * LIK_TEMP, -80.0f)) + 1.0e-18f;
}

__global__ void reduce_weight_stats_kernel(const float* weights, float* out, int n) {
    extern __shared__ float sh[];
    float* s_sum = sh;
    float* s_sq = sh + blockDim.x;
    int tid = threadIdx.x;
    float sum = 0.0f, sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = weights[i];
        sum += v;
        sq += v * v;
    }
    s_sum[tid] = sum;
    s_sq[tid] = sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq[tid] += s_sq[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = s_sum[0];
        out[1] = s_sq[0];
    }
}

__global__ void normalize_kernel(float* weights, float sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    weights[i] = (sum > 1.0e-30f) ? (weights[i] / sum) : (1.0f / n);
}

__global__ void weighted_mean_kernel(const float* x,
                                     const float* y,
                                     const float* th,
                                     const float* weights,
                                     float* out,
                                     int n) {
    extern __shared__ float sh[];
    float* sx = sh;
    float* sy = sh + blockDim.x;
    float* sc = sh + 2 * blockDim.x;
    float* ss = sh + 3 * blockDim.x;
    int tid = threadIdx.x;
    float ax = 0.0f, ay = 0.0f, ac = 0.0f, as = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float wt = weights[i];
        ax += x[i] * wt;
        ay += y[i] * wt;
        ac += cosf(th[i]) * wt;
        as += sinf(th[i]) * wt;
    }
    sx[tid] = ax;
    sy[tid] = ay;
    sc[tid] = ac;
    ss[tid] = as;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sx[tid] += sx[tid + stride];
            sy[tid] += sy[tid + stride];
            sc[tid] += sc[tid + stride];
            ss[tid] += ss[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = sx[0];
        out[1] = sy[0];
        out[2] = atan2f(ss[0], sc[0]);
    }
}

__global__ void cumsum_kernel(const float* weights, float* wcum, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += weights[i];
        wcum[i] = acc;
    }
    wcum[n - 1] = 1.0f;
}

__global__ void resample_kernel(const float* x,
                                const float* y,
                                const float* th,
                                const float* wcum,
                                float* x2,
                                float* y2,
                                float* th2,
                                int n,
                                float offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float target = fminf(0.99999994f, offset + i / static_cast<float>(n));
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (wcum[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    x2[i] = x[lo];
    y2[i] = y[lo];
    th2[i] = th[lo];
}

__global__ void copy_back_uniform_kernel(float* x,
                                         float* y,
                                         float* th,
                                         float* w,
                                         const float* x2,
                                         const float* y2,
                                         const float* th2,
                                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = x2[i];
    y[i] = y2[i];
    th[i] = th2[i];
    w[i] = 1.0f / n;
}

__device__ int bucket_id(float x, float y, float th) {
    int bx = max(0, min(BUCKET_X - 1, static_cast<int>(x / WORLD_W * BUCKET_X)));
    int by = max(0, min(BUCKET_Y - 1, static_cast<int>(y / WORLD_H * BUCKET_Y)));
    float tn = (wrap_angle(th) + PI_F) / (2.0f * PI_F);
    int bt = max(0, min(BUCKET_T - 1, static_cast<int>(tn * BUCKET_T)));
    return bx + BUCKET_X * (by + BUCKET_Y * bt);
}

__global__ void bucket_motion_aggregate_kernel(const float* x,
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
    int b = bucket_id(x[i], y[i], th[i]);
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

__global__ void stein_bucket_update_kernel(float* x,
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
    int b = bucket_id(x[i], y[i], th[i]);
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

__global__ void bucket_posterior_aggregate_kernel(const float* x,
                                                  const float* y,
                                                  const float* th,
                                                  const float* posterior,
                                                  float* b_post,
                                                  float* b_count,
                                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = bucket_id(x[i], y[i], th[i]);
    atomicAdd(&b_post[b], posterior[i]);
    atomicAdd(&b_count[b], 1.0f);
}

__global__ void posterior_smooth_kernel(const float* x,
                                        const float* y,
                                        const float* th,
                                        float* posterior,
                                        const float* b_post,
                                        const float* b_count,
                                        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = bucket_id(x[i], y[i], th[i]);
    float mean = b_post[b] / fmaxf(b_count[b], 1.0f);
    posterior[i] = 0.58f * posterior[i] + 0.42f * mean;
}

__global__ void bucket_representative_aggregate_kernel(const float* x,
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
    int b = bucket_id(x[i], y[i], th[i]);
    float p = posterior[i] + 1.0e-18f;
    atomicAdd(&b_x[b], p * x[i]);
    atomicAdd(&b_y[b], p * y[i]);
    atomicAdd(&b_sin[b], p * sinf(th[i]));
    atomicAdd(&b_cos[b], p * cosf(th[i]));
    atomicAdd(&b_post[b], p);
}

__global__ void argmax_pose_kernel(const float* score,
                                   const float* x,
                                   const float* y,
                                   const float* th,
                                   float* block_score,
                                   float* block_pose,
                                   int n) {
    __shared__ float s_score[THREADS];
    __shared__ int s_idx[THREADS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float v = (i < n) ? score[i] : -1.0e30f;
    s_score[tid] = v;
    s_idx[tid] = i;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_score[tid + stride] > s_score[tid]) {
            s_score[tid] = s_score[tid + stride];
            s_idx[tid] = s_idx[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        int bi = blockIdx.x;
        int best = s_idx[0];
        block_score[bi] = s_score[0];
        if (best >= 0 && best < n) {
            block_pose[3 * bi + 0] = x[best];
            block_pose[3 * bi + 1] = y[best];
            block_pose[3 * bi + 2] = th[best];
        } else {
            block_pose[3 * bi + 0] = 0.0f;
            block_pose[3 * bi + 1] = 0.0f;
            block_pose[3 * bi + 2] = 0.0f;
        }
    }
}

struct ParticleSet {
    int n = 0;
    float *x = nullptr, *y = nullptr, *th = nullptr;
    float *w = nullptr, *score = nullptr, *step_x = nullptr, *step_y = nullptr, *step_th = nullptr;
    float *x2 = nullptr, *y2 = nullptr, *th2 = nullptr, *wcum = nullptr;
    float *stats = nullptr, *mean = nullptr;
    float *block_score = nullptr, *block_pose = nullptr;
    curandState* rng = nullptr;
    std::vector<float> hx, hy, h_block_score, h_block_pose, h_mean, h_stats;

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
        CUDA_CHECK(cudaMalloc(&x2, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&y2, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&th2, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&wcum, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&stats, 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&mean, 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&block_score, blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&block_pose, 3 * blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rng, n * sizeof(curandState)));
        hx.resize(n);
        hy.resize(n);
        h_block_score.resize(blocks);
        h_block_pose.resize(3 * blocks);
        h_mean.resize(3);
        h_stats.resize(2);
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
        CUDA_CHECK(cudaFree(x2));
        CUDA_CHECK(cudaFree(y2));
        CUDA_CHECK(cudaFree(th2));
        CUDA_CHECK(cudaFree(wcum));
        CUDA_CHECK(cudaFree(stats));
        CUDA_CHECK(cudaFree(mean));
        CUDA_CHECK(cudaFree(block_score));
        CUDA_CHECK(cudaFree(block_pose));
        CUDA_CHECK(cudaFree(rng));
    }

    void copy_xy_to_host() {
        CUDA_CHECK(cudaMemcpy(hx.data(), x, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hy.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost));
    }

    Pose2 weighted_mean() {
        weighted_mean_kernel<<<1, THREADS, 4 * THREADS * sizeof(float)>>>(x, y, th, w, mean, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_mean.data(), mean, 3 * sizeof(float), cudaMemcpyDeviceToHost));
        return {h_mean[0], h_mean[1], h_mean[2]};
    }

    Pose2 argmax(float* score_ptr, float* best_score = nullptr) {
        int blocks = (n + THREADS - 1) / THREADS;
        argmax_pose_kernel<<<blocks, THREADS>>>(score_ptr, x, y, th, block_score, block_pose, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_block_score.data(), block_score, blocks * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_block_pose.data(), block_pose, 3 * blocks * sizeof(float), cudaMemcpyDeviceToHost));
        int best = 0;
        for (int i = 1; i < blocks; ++i) {
            if (h_block_score[i] > h_block_score[best]) best = i;
        }
        if (best_score) *best_score = h_block_score[best];
        return {h_block_pose[3 * best + 0], h_block_pose[3 * best + 1], h_block_pose[3 * best + 2]};
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

struct MegaBuckets {
    float *step_x = nullptr, *step_y = nullptr, *step_th = nullptr;
    float *x = nullptr, *y = nullptr, *sin_th = nullptr, *cos_th = nullptr;
    float *post = nullptr, *count = nullptr;
    std::vector<float> h_x, h_y, h_sin, h_cos, h_post;

    void alloc() {
        CUDA_CHECK(cudaMalloc(&step_x, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&step_y, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&step_th, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&x, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&y, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sin_th, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cos_th, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&post, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&count, N_BUCKETS * sizeof(float)));
        h_x.resize(N_BUCKETS);
        h_y.resize(N_BUCKETS);
        h_sin.resize(N_BUCKETS);
        h_cos.resize(N_BUCKETS);
        h_post.resize(N_BUCKETS);
    }

    void clear_all() {
        CUDA_CHECK(cudaMemset(step_x, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(step_y, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(step_th, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(x, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(y, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(sin_th, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(cos_th, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(post, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(count, 0, N_BUCKETS * sizeof(float)));
    }

    void clear_post() {
        CUDA_CHECK(cudaMemset(post, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(count, 0, N_BUCKETS * sizeof(float)));
    }

    void free_all() {
        CUDA_CHECK(cudaFree(step_x));
        CUDA_CHECK(cudaFree(step_y));
        CUDA_CHECK(cudaFree(step_th));
        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
        CUDA_CHECK(cudaFree(sin_th));
        CUDA_CHECK(cudaFree(cos_th));
        CUDA_CHECK(cudaFree(post));
        CUDA_CHECK(cudaFree(count));
    }
};

static Pose2 bucket_representative(ParticleSet& p, MegaBuckets& buckets, float* best_mass) {
    int blocks = (p.n + THREADS - 1) / THREADS;
    buckets.clear_all();
    bucket_representative_aggregate_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.w,
                                                                buckets.x, buckets.y,
                                                                buckets.sin_th, buckets.cos_th,
                                                                buckets.post, p.n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(buckets.h_post.data(), buckets.post, N_BUCKETS * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(buckets.h_x.data(), buckets.x, N_BUCKETS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(buckets.h_y.data(), buckets.y, N_BUCKETS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(buckets.h_sin.data(), buckets.sin_th, N_BUCKETS * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(buckets.h_cos.data(), buckets.cos_th, N_BUCKETS * sizeof(float),
                          cudaMemcpyDeviceToHost));
    int best = 0;
    for (int i = 1; i < N_BUCKETS; ++i) {
        if (buckets.h_post[i] > buckets.h_post[best]) best = i;
    }
    float mass = std::max(buckets.h_post[best], 1.0e-18f);
    if (best_mass) *best_mass = mass;
    return {buckets.h_x[best] / mass,
            buckets.h_y[best] / mass,
            std::atan2(buckets.h_sin[best] / mass, buckets.h_cos[best] / mass)};
}

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
    cv::putText(img, title, cv::Point(ox + 14, 28), cv::FONT_HERSHEY_SIMPLEX, 0.60,
                cv::Scalar(20, 24, 32), 2, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(ox, 0, PANEL_W, PANEL_H), cv::Scalar(210, 210, 205), 1);
}

static void draw_info(cv::Mat& img,
                      int ox,
                      int step,
                      const StepSummary& s,
                      const FinalStats& partial,
                      bool occluded) {
    cv::rectangle(img, cv::Rect(ox, 0, INFO_W, FRAME_H), cv::Scalar(244, 246, 246), -1);
    cv::putText(img, "MegaParticles-style SPF", cv::Point(ox + 18, 34), cv::FONT_HERSHEY_SIMPLEX,
                0.62, cv::Scalar(20, 28, 35), 2, cv::LINE_AA);
    cv::putText(img, "SE(2) range-field demo", cv::Point(ox + 18, 60), cv::FONT_HERSHEY_SIMPLEX,
                0.48, cv::Scalar(70, 78, 88), 1, cv::LINE_AA);

    char buf[256];
    std::snprintf(buf, sizeof(buf), "step %03d / %03d", step, N_STEPS - 1);
    cv::putText(img, buf, cv::Point(ox + 18, 104), cv::FONT_HERSHEY_SIMPLEX, 0.54,
                cv::Scalar(30, 36, 44), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "scan: %s", occluded ? "blocked / hidden kidnap" : "visible");
    cv::putText(img, buf, cv::Point(ox + 18, 130), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                occluded ? cv::Scalar(40, 70, 190) : cv::Scalar(40, 120, 80), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "local bootstrap err: %.2f m", s.local_err);
    cv::putText(img, buf, cv::Point(ox + 18, 176), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(180, 80, 40), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "mega stein err: %.2f m", s.mega_err);
    cv::putText(img, buf, cv::Point(ox + 18, 202), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(55, 95, 175), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "mega particles: %d", K_MEGA);
    cv::putText(img, buf, cv::Point(ox + 18, 248), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "local particles: %d", K_LOCAL);
    cv::putText(img, buf, cv::Point(ox + 18, 272), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "avg step: %.2f / %.2f ms", partial.local_ms, partial.mega_ms);
    cv::putText(img, buf, cv::Point(ox + 18, 318), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
}

static void local_filter_step(ParticleSet& p,
                              const DeviceMap& dmap,
                              const float* d_scan_x,
                              const float* d_scan_y,
                              float v,
                              float omega,
                              bool visible,
                              std::mt19937& host_rng,
                              Pose2& estimate) {
    int blocks = (p.n + THREADS - 1) / THREADS;
    predict_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng, p.n, v, omega,
                                        LOCAL_MOTION_SIGMA_XY, LOCAL_MOTION_SIGMA_TH);
    CUDA_CHECK(cudaGetLastError());
    if (visible) {
        likelihood_gradient_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, d_scan_x, d_scan_y,
                                                        dmap.dist, dmap.gx, dmap.gy,
                                                        p.score, p.step_x, p.step_y, p.step_th, p.n);
        CUDA_CHECK(cudaGetLastError());
        weights_from_score_kernel<<<blocks, THREADS>>>(p.score, p.w, p.n);
        CUDA_CHECK(cudaGetLastError());
        reduce_weight_stats_kernel<<<1, THREADS, 2 * THREADS * sizeof(float)>>>(p.w, p.stats, p.n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(p.h_stats.data(), p.stats, 2 * sizeof(float), cudaMemcpyDeviceToHost));
        normalize_kernel<<<blocks, THREADS>>>(p.w, p.h_stats[0], p.n);
        CUDA_CHECK(cudaGetLastError());
        estimate = p.weighted_mean();
        cumsum_kernel<<<1, 1>>>(p.w, p.wcum, p.n);
        CUDA_CHECK(cudaGetLastError());
        std::uniform_real_distribution<float> unif(0.0f, 1.0f / p.n);
        resample_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.wcum, p.x2, p.y2, p.th2,
                                             p.n, unif(host_rng));
        CUDA_CHECK(cudaGetLastError());
        copy_back_uniform_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.w, p.x2, p.y2, p.th2, p.n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        estimate = p.weighted_mean();
    }
}

static void mega_filter_step(ParticleSet& p,
                             MegaBuckets& buckets,
                             const DeviceMap& dmap,
                             const float* d_scan_x,
                             const float* d_scan_y,
                             float v,
                             float omega,
                             bool visible,
                             Pose2& estimate,
                             float& best_score) {
    int blocks = (p.n + THREADS - 1) / THREADS;
    predict_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng, p.n, v, omega,
                                        MEGA_MOTION_SIGMA_XY, MEGA_MOTION_SIGMA_TH);
    CUDA_CHECK(cudaGetLastError());
    if (!visible) {
        occlusion_spread_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng, p.n);
        CUDA_CHECK(cudaGetLastError());
        estimate = p.argmax(p.w, &best_score);
        return;
    }

    for (int it = 0; it < STEIN_ITERS; ++it) {
        likelihood_gradient_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, d_scan_x, d_scan_y,
                                                        dmap.dist, dmap.gx, dmap.gy,
                                                        p.score, p.step_x, p.step_y, p.step_th, p.n);
        CUDA_CHECK(cudaGetLastError());
        posterior_from_score_kernel<<<blocks, THREADS>>>(p.score, p.w, p.n);
        CUDA_CHECK(cudaGetLastError());
        buckets.clear_all();
        bucket_motion_aggregate_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.step_x, p.step_y, p.step_th,
                                                            p.w, buckets.step_x, buckets.step_y,
                                                            buckets.step_th, buckets.x, buckets.y,
                                                            buckets.sin_th, buckets.cos_th,
                                                            buckets.post, buckets.count, p.n);
        CUDA_CHECK(cudaGetLastError());
        stein_bucket_update_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng,
                                                        p.step_x, p.step_y, p.step_th,
                                                        buckets.step_x, buckets.step_y,
                                                        buckets.step_th, buckets.x, buckets.y,
                                                        buckets.sin_th, buckets.cos_th,
                                                        buckets.post, buckets.count, p.n);
        CUDA_CHECK(cudaGetLastError());
    }

    likelihood_gradient_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, d_scan_x, d_scan_y,
                                                    dmap.dist, dmap.gx, dmap.gy,
                                                    p.score, p.step_x, p.step_y, p.step_th, p.n);
    CUDA_CHECK(cudaGetLastError());
    posterior_from_score_kernel<<<blocks, THREADS>>>(p.score, p.w, p.n);
    CUDA_CHECK(cudaGetLastError());
    for (int it = 0; it < POST_PROP_ITERS; ++it) {
        buckets.clear_post();
        bucket_posterior_aggregate_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.w,
                                                               buckets.post, buckets.count, p.n);
        CUDA_CHECK(cudaGetLastError());
        posterior_smooth_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.w,
                                                     buckets.post, buckets.count, p.n);
        CUDA_CHECK(cudaGetLastError());
    }
    estimate = bucket_representative(p, buckets, &best_score);
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

    float *d_scan_x = nullptr, *d_scan_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scan_x, N_SCAN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_y, N_SCAN * sizeof(float)));

    ParticleSet local, mega;
    local.alloc(K_LOCAL, 1234);
    mega.alloc(K_MEGA, 5678);
    MegaBuckets buckets;
    buckets.alloc();

    std::vector<float> v_cmd, w_cmd;
    std::vector<Pose2> truth = make_truth(v_cmd, w_cmd);

    int local_blocks = (K_LOCAL + THREADS - 1) / THREADS;
    int mega_blocks = (K_MEGA + THREADS - 1) / THREADS;
    init_gaussian_kernel<<<local_blocks, THREADS>>>(local.x, local.y, local.th, local.w,
                                                    local.rng, local.n, truth.front());
    CUDA_CHECK(cudaGetLastError());
    init_uniform_kernel<<<mega_blocks, THREADS>>>(mega.x, mega.y, mega.th, mega.w, mega.rng, mega.n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cv::VideoWriter video("tmp/gpu_megaparticles_stein_mcl.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 12,
                          cv::Size(FRAME_W, FRAME_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open output video\n");
        std::exit(1);
    }

    std::mt19937 host_rng(42);
    std::vector<Pose2> local_hist, mega_hist, truth_hist;
    std::vector<float> scan_x, scan_y;
    FinalStats stats;
    StepSummary last;
    int post_count = 0;
    float local_post_sq = 0.0f;
    float mega_post_sq = 0.0f;
    double local_ms_sum = 0.0;
    double mega_ms_sum = 0.0;
    bool mega_has_track = false;
    Pose2 mega_track{};

    for (int t = 0; t < N_STEPS; ++t) {
        bool visible = !(t >= KIDNAP_STEP && t < KIDNAP_STEP + OCCLUDE_STEPS);
        bool just_unblocked = (t == KIDNAP_STEP + OCCLUDE_STEPS);
        if (visible) {
            make_scan(map.rects, truth[t], t, scan_x, scan_y);
            CUDA_CHECK(cudaMemcpy(d_scan_x, scan_x.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_scan_y, scan_y.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));
        }

        Pose2 local_est, mega_est;
        float mega_score = 0.0f;
        auto t0 = std::chrono::high_resolution_clock::now();
        local_filter_step(local, dmap, d_scan_x, d_scan_y, v_cmd[t], w_cmd[t], visible, host_rng, local_est);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        mega_filter_step(mega, buckets, dmap, d_scan_x, d_scan_y, v_cmd[t], w_cmd[t], visible, mega_est, mega_score);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();

        // The paper reports pose jitter when directly taking the max-posterior
        // particle.  This tiny representative-state gate keeps normal tracking
        // continuous while allowing a global jump immediately after blackout.
        if (!mega_has_track || just_unblocked) {
            mega_track = mega_est;
            mega_has_track = true;
        } else {
            Pose2 predicted_track = integrate_pose(mega_track, v_cmd[t], w_cmd[t]);
            if (visible && pose_error_xy(mega_est, predicted_track) > 2.6f) {
                mega_est = predicted_track;
            }
            mega_track = visible ? mega_est : predicted_track;
            if (!visible) mega_est = mega_track;
        }

        double local_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double mega_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        local_ms_sum += local_ms;
        mega_ms_sum += mega_ms;
        stats.local_ms = local_ms_sum / (t + 1);
        stats.mega_ms = mega_ms_sum / (t + 1);

        last.local = local_est;
        last.mega = mega_est;
        last.local_err = pose_error_xy(local_est, truth[t]);
        last.mega_err = pose_error_xy(mega_est, truth[t]);
        last.mega_score = mega_score;
        last.scan_visible = visible;
        last.local_ms = local_ms;
        last.mega_ms = mega_ms;

        if (t >= KIDNAP_STEP + OCCLUDE_STEPS) {
            local_post_sq += last.local_err * last.local_err;
            mega_post_sq += last.mega_err * last.mega_err;
            post_count++;
            if (stats.mega_reacq_step < 0 && last.mega_err < 0.65f) {
                stats.mega_reacq_step = t - (KIDNAP_STEP + OCCLUDE_STEPS);
            }
        }
        stats.final_local_err = last.local_err;
        stats.final_mega_err = last.mega_err;
        stats.local_post_rmse = post_count ? std::sqrt(local_post_sq / post_count) : 0.0f;
        stats.mega_post_rmse = post_count ? std::sqrt(mega_post_sq / post_count) : 0.0f;

        local_hist.push_back(local_est);
        mega_hist.push_back(mega_est);
        truth_hist.push_back(truth[t]);

        if (t % VIDEO_EVERY == 0 || t == N_STEPS - 1) {
            local.copy_xy_to_host();
            mega.copy_xy_to_host();
            cv::Mat frame(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(244, 246, 246));
            draw_panel(frame, 0, "64K bootstrap MCL", map, truth_hist, local_hist, truth[t], local_est,
                       local.hx, local.hy, std::max(1, K_LOCAL / 2500),
                       cv::Scalar(180, 190, 230), cv::Scalar(40, 95, 210));
            draw_panel(frame, PANEL_W, "1M MegaParticles-style SPF", map, truth_hist, mega_hist, truth[t], mega_est,
                       mega.hx, mega.hy, std::max(1, K_MEGA / 3600),
                       cv::Scalar(190, 215, 200), cv::Scalar(30, 150, 95));
            draw_info(frame, PANEL_W * 2, t, last, stats, !visible);
            video.write(frame);
        }

        std::printf("step %3d visible=%d local_err=%.3f mega_err=%.3f local=%.2fms mega=%.2fms\n",
                    t, visible ? 1 : 0, last.local_err, last.mega_err, local_ms, mega_ms);
    }

    video.release();
    avi_to_gif("tmp/gpu_megaparticles_stein_mcl.avi", "gif/gpu_megaparticles_stein_mcl.gif", 12, 900);

    CUDA_CHECK(cudaFree(d_scan_x));
    CUDA_CHECK(cudaFree(d_scan_y));
    buckets.free_all();
    local.free_all();
    mega.free_all();
    dmap.free_all();
    return stats;
}

}  // namespace cudabot

int main() {
    cudabot::FinalStats stats = cudabot::run_demo();
    std::printf("\nMegaParticles-style SE(2) range localization\n");
    std::printf("post-kidnap RMSE: local bootstrap %.4f m, mega stein %.4f m\n",
                stats.local_post_rmse, stats.mega_post_rmse);
    std::printf("final error: local bootstrap %.4f m, mega stein %.4f m\n",
                stats.final_local_err, stats.final_mega_err);
    std::printf("mega reacquisition after blackout: %d frames\n", stats.mega_reacq_step);
    std::printf("avg GPU step: local bootstrap %.4f ms, mega stein %.4f ms\n",
                stats.local_ms, stats.mega_ms);
    std::printf("Wrote gif/gpu_megaparticles_stein_mcl.gif\n");
    return 0;
}
