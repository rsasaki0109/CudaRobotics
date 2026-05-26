// gpu_megaparticles_smoother.cu
//
// Fixed-lag trajectory smoother for the MegaParticles representative state.
//
// The MegaParticles line (#86 Stein, #101 LSH, #104 6-DoF, #115 GICP D2D)
// reports its pose by picking the highest-posterior bucket each frame.  As the
// paper itself notes, that raw max-posterior estimate jitters frame-to-frame:
// the winning bucket flips between neighbours, and a single ambiguous frame can
// spike the readout.  Every demo in the line papered over this with a tiny
// hand-tuned continuity gate -- explicitly flagged as a known limitation.
//
// This demo replaces that gate with a principled robust fixed-lag smoother and
// reports the raw vs smoothed pose error SEPARATELY, so the smoothing benefit is
// measurable rather than hidden.  The GPU does the expensive part exactly as in
// #86 (one million particles, distance-field likelihood, bucket-neighbor Stein
// motion, posterior smoothing); each frame it emits a single raw representative
// pose.  A lightweight host backend keeps a sliding window of the last W frames
// and jointly optimises a smoothed pose chain with two factor types:
//
//   * MOTION (between consecutive smoothed poses, from the odometry command),
//     with a SWITCHABLE robust weight so a genuine pose discontinuity -- the
//     hidden kidnap -- breaks the link instead of being smeared across frames;
//   * MEASUREMENT (smoothed pose vs the raw representative), with a Huber-like
//     robust weight so a one-frame spurious max-posterior spike is rejected.
//
// A frame is "finalized" once it falls off the trailing edge of the window, so
// each reported pose has been refined using W future frames -- that is what
// makes it a smoother rather than a filter.  The interesting test is that the
// SAME backend must (a) cut the in-track jitter and (b) NOT smooth across the
// legitimate kidnap jump: the switchable motion factor handles both.
//
// Output: gif/gpu_megaparticles_smoother.gif

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
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
constexpr int BUCKET_X = 48;
constexpr int BUCKET_Y = 34;
constexpr int BUCKET_T = 24;
constexpr int N_BUCKETS = BUCKET_X * BUCKET_Y * BUCKET_T;

// --- Fixed-lag smoother backend ---------------------------------------------
constexpr int SMOOTH_W = 10;       // sliding-window length == smoothing lag (frames)
constexpr int GN_ITERS = 6;        // Gauss-Newton (IRLS) iterations per frame
// Diagonal information (1/variance) for the two factor families.
constexpr double MEAS_INFO_XY = 6.0;     // raw representative ~0.4 m std
constexpr double MEAS_INFO_TH = 22.0;
constexpr double MOTION_INFO_XY = 26.0;  // odometry CV prior, tighter than meas
constexpr double MOTION_INFO_TH = 160.0;
constexpr double ANCHOR_INFO_XY = 9.0;   // marginalized-past prior on window head
constexpr double ANCHOR_INFO_TH = 32.0;
// Robust break / reject thresholds (Geman-McClure-style scalar reweighting).
constexpr double MOTION_BREAK_M = 0.80;  // > this xy step => link weakens (kidnap)
constexpr double MEAS_REJECT_M = 1.20;   // raw obs this far from chain => outlier
constexpr double ROBUST_FLOOR = 0.02;    // keep a sliver of weight for conditioning
constexpr double BREAK_INNOV_M = 2.5;    // post-dropout jump that triggers a reset

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

struct FinalStats {
    float raw_track_rmse = 0.0f;
    float smooth_track_rmse = 0.0f;
    float raw_post_rmse = 0.0f;
    float smooth_post_rmse = 0.0f;
    float raw_jitter = 0.0f;
    float smooth_jitter = 0.0f;
    float truth_jitter = 0.0f;
    float final_raw_err = 0.0f;
    float final_smooth_err = 0.0f;
    int smooth_relock_step = -1;
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

struct ParticleSet {
    int n = 0;
    float *x = nullptr, *y = nullptr, *th = nullptr;
    float *w = nullptr, *score = nullptr, *step_x = nullptr, *step_y = nullptr, *step_th = nullptr;
    curandState* rng = nullptr;
    std::vector<float> hx, hy;

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

    void copy_xy_to_host() {
        CUDA_CHECK(cudaMemcpy(hx.data(), x, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hy.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost));
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

static void mega_filter_step(ParticleSet& p,
                             MegaBuckets& buckets,
                             const DeviceMap& dmap,
                             const float* d_scan_x,
                             const float* d_scan_y,
                             float v,
                             float omega,
                             bool visible,
                             Pose2& estimate,
                             float& best_mass) {
    int blocks = (p.n + THREADS - 1) / THREADS;
    predict_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng, p.n, v, omega,
                                        MEGA_MOTION_SIGMA_XY, MEGA_MOTION_SIGMA_TH);
    CUDA_CHECK(cudaGetLastError());
    if (!visible) {
        occlusion_spread_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng, p.n);
        CUDA_CHECK(cudaGetLastError());
        best_mass = 0.0f;
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
    estimate = bucket_representative(p, buckets, &best_mass);
}

// --- Fixed-lag robust pose-chain smoother (host backend) --------------------
//
// State is the window's smoothed poses; we minimise, by IRLS Gauss-Newton, the
// sum of a marginalized-past anchor on the head, switchable CV-motion factors
// between consecutive poses, and Huber-robust measurement factors pulling each
// pose to its raw MegaParticles representative.  A frame leaves the window from
// the head (oldest) and is "finalized" there, having seen SMOOTH_W future
// frames.

struct SmoothFrame {
    Pose2 obs{};       // raw MegaParticles representative for this frame
    bool has_obs = false;
    float v = 0.0f;    // odometry command carrying the PREVIOUS pose to this one
    float w = 0.0f;
    int index = -1;    // absolute time step
};

struct Finalized {
    int index;
    Pose2 pose;
};

// Solve SPD system A x = b in place (A row-major n*n, lower triangle used).
static void chol_solve(std::vector<double>& A, std::vector<double>& b, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[i * n + j];
            for (int k = 0; k < j; ++k) sum -= A[i * n + k] * A[j * n + k];
            if (i == j) {
                if (sum < 1.0e-12) sum = 1.0e-12;
                A[i * n + i] = std::sqrt(sum);
            } else {
                A[i * n + j] = sum / A[j * n + j];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= A[i * n + k] * b[k];
        b[i] = s / A[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int k = i + 1; k < n; ++k) s -= A[k * n + i] * b[k];
        b[i] = s / A[i * n + i];
    }
}

struct FixedLagSmoother {
    std::deque<SmoothFrame> win;
    std::deque<Pose2> est;       // current smoothed estimate per window slot
    Pose2 anchor{};
    bool have_anchor = false;
    bool lost_pending = false;   // saw a measurement dropout since the last reset

    // Push a new frame, re-optimise the window, and return any frame that fell
    // off the head (finalized).  `out` receives finalized poses (0, 1, or a
    // whole segment when a post-dropout relocalization resets the window).
    void push(const SmoothFrame& f, std::vector<Finalized>& out) {
        // Warm start the new slot from the CV prediction of the previous pose,
        // or from its own observation when the window is empty.
        Pose2 warm;
        if (est.empty()) {
            warm = f.has_obs ? f.obs : Pose2{WORLD_W * 0.5f, WORLD_H * 0.5f, 0.0f};
        } else {
            warm = integrate_pose(est.back(), f.v, f.w);
        }

        // A run of has_obs=false frames means the localizer lost track (scan
        // dropout / confidence collapse).  If measurements then resume FAR from
        // where we coasted, this is a genuine relocalization, not an outlier:
        // finalize the stale segment and restart a fresh window at the new
        // observation.  Crucially this only fires after a dropout -- the
        // high-confidence spurious-mode flips during normal tracking never set
        // lost_pending, so they are rejected by the robust measurement kernel
        // instead of resetting the smoother.
        if (!f.has_obs) lost_pending = true;
        if (f.has_obs && lost_pending) {
            if (!est.empty() && pose_error_xy(f.obs, warm) > BREAK_INNOV_M) {
                for (size_t k = 0; k < win.size(); ++k) out.push_back({win[k].index, est[k]});
                win.clear();
                est.clear();
                have_anchor = false;
                warm = f.obs;  // re-seed the new segment at the new observation
            }
            lost_pending = false;
        }

        win.push_back(f);
        est.push_back(warm);

        if (static_cast<int>(win.size()) > SMOOTH_W) {
            // The head leaves the window: it is now final.  Re-anchor the new
            // head to its current estimate to carry the marginalized past.
            out.push_back({win.front().index, est.front()});
            win.pop_front();
            est.pop_front();
            anchor = est.front();
            have_anchor = true;
        }
        solve();
    }

    // Current online (least-lagged) estimate: the newest window pose.  It has
    // a motion factor and its own measurement but no future frames, so it acts
    // as a robust filter -- it already rejects the spurious max-posterior modes.
    Pose2 current(const Pose2& fallback) const {
        return est.empty() ? fallback : est.back();
    }

    // Finalize everything still in the window using the last solution.
    void flush(std::vector<Finalized>& out) {
        solve();
        for (size_t k = 0; k < win.size(); ++k) {
            out.push_back({win[k].index, est[k]});
        }
        win.clear();
        est.clear();
    }

    void solve() {
        int W = static_cast<int>(win.size());
        if (W == 0) return;
        int n = 3 * W;
        std::vector<double> H(n * n);
        std::vector<double> g(n);

        for (int iter = 0; iter < GN_ITERS; ++iter) {
            std::fill(H.begin(), H.end(), 0.0);
            std::fill(g.begin(), g.end(), 0.0);

            auto add_block = [&](int R, int C, const double M[3][3]) {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        H[(3 * R + i) * n + (3 * C + j)] += M[i][j];
            };
            auto add_g = [&](int R, double a, double b, double c) {
                g[3 * R + 0] += a;
                g[3 * R + 1] += b;
                g[3 * R + 2] += c;
            };

            // Anchor on the head: pull est[0] toward the marginalized prior.
            if (have_anchor) {
                double rx = anchor.x - est[0].x;
                double ry = anchor.y - est[0].y;
                double rt = wrap_angle(anchor.th - est[0].th);
                double M[3][3] = {{ANCHOR_INFO_XY, 0, 0}, {0, ANCHOR_INFO_XY, 0}, {0, 0, ANCHOR_INFO_TH}};
                add_block(0, 0, M);
                add_g(0, ANCHOR_INFO_XY * rx, ANCHOR_INFO_XY * ry, ANCHOR_INFO_TH * rt);
            }

            // Measurement factors: est[k] vs raw representative obs[k].
            for (int k = 0; k < W; ++k) {
                if (!win[k].has_obs) continue;
                double rx = win[k].obs.x - est[k].x;
                double ry = win[k].obs.y - est[k].y;
                double rt = wrap_angle(win[k].obs.th - est[k].th);
                double dist = std::sqrt(rx * rx + ry * ry);
                double s = MEAS_REJECT_M / (MEAS_REJECT_M + dist);  // Huber-like
                double rw = std::max(s * s, ROBUST_FLOOR);
                double ix = MEAS_INFO_XY * rw, it = MEAS_INFO_TH * rw;
                double M[3][3] = {{ix, 0, 0}, {0, ix, 0}, {0, 0, it}};
                add_block(k, k, M);
                add_g(k, ix * rx, ix * ry, it * rt);
            }

            // Switchable CV-motion factors between consecutive poses.
            for (int k = 0; k + 1 < W; ++k) {
                const SmoothFrame& fb = win[k + 1];
                Pose2 pa = est[k];
                double a = -static_cast<double>(fb.v) * std::sin(pa.th) * DT;  // dgx/dth
                double b = static_cast<double>(fb.v) * std::cos(pa.th) * DT;   // dgy/dth
                Pose2 pred = integrate_pose(pa, fb.v, fb.w);
                double rx = est[k + 1].x - pred.x;
                double ry = est[k + 1].y - pred.y;
                double rt = wrap_angle(est[k + 1].th - pred.th);
                double dist = std::sqrt(rx * rx + ry * ry);
                // Switchable weight: a large step (the kidnap) lets the link break.
                double s = (MOTION_BREAK_M * MOTION_BREAK_M) /
                           (MOTION_BREAK_M * MOTION_BREAK_M + dist * dist);
                double rw = std::max(s, ROBUST_FLOOR);
                double d0 = MOTION_INFO_XY * rw, d1 = MOTION_INFO_XY * rw, d2 = MOTION_INFO_TH * rw;

                // Block (k+1,k+1) = Omega ; (k,k) = G^T Omega G ;
                // (k,k+1) = -G^T Omega ; (k+1,k) = transpose.
                double Hbb[3][3] = {{d0, 0, 0}, {0, d1, 0}, {0, 0, d2}};
                double Haa[3][3] = {{d0, 0, a * d0},
                                    {0, d1, b * d1},
                                    {a * d0, b * d1, a * a * d0 + b * b * d1 + d2}};
                double Hab[3][3] = {{-d0, 0, 0}, {0, -d1, 0}, {-a * d0, -b * d1, -d2}};
                double Hba[3][3] = {{-d0, 0, -a * d0}, {0, -d1, -b * d1}, {0, 0, -d2}};
                add_block(k + 1, k + 1, Hbb);
                add_block(k, k, Haa);
                add_block(k, k + 1, Hab);
                add_block(k + 1, k, Hba);
                // rhs: g_{k+1} += -Omega r ; g_k += G^T Omega r
                add_g(k + 1, -d0 * rx, -d1 * ry, -d2 * rt);
                add_g(k, d0 * rx, d1 * ry, a * d0 * rx + b * d1 * ry + d2 * rt);
            }

            std::vector<double> delta = g;
            chol_solve(H, delta, n);
            for (int k = 0; k < W; ++k) {
                est[k].x += static_cast<float>(delta[3 * k + 0]);
                est[k].y += static_cast<float>(delta[3 * k + 1]);
                est[k].th = wrap_angle(est[k].th + static_cast<float>(delta[3 * k + 2]));
            }
        }
    }
};

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

static void draw_info(cv::Mat& img,
                      int ox,
                      int step,
                      bool occluded,
                      float raw_err,
                      float smooth_err,
                      const FinalStats& partial) {
    cv::rectangle(img, cv::Rect(ox, 0, INFO_W, FRAME_H), cv::Scalar(244, 246, 246), -1);
    cv::putText(img, "MegaParticles smoother", cv::Point(ox + 18, 34), cv::FONT_HERSHEY_SIMPLEX,
                0.60, cv::Scalar(20, 28, 35), 2, cv::LINE_AA);
    cv::putText(img, "1M particles + fixed-lag", cv::Point(ox + 18, 60), cv::FONT_HERSHEY_SIMPLEX,
                0.46, cv::Scalar(70, 78, 88), 1, cv::LINE_AA);

    char buf[256];
    std::snprintf(buf, sizeof(buf), "step %03d / %03d", step, N_STEPS - 1);
    cv::putText(img, buf, cv::Point(ox + 18, 100), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(30, 36, 44), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "scan: %s", occluded ? "blocked / hidden kidnap" : "visible");
    cv::putText(img, buf, cv::Point(ox + 18, 124), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                occluded ? cv::Scalar(40, 70, 190) : cv::Scalar(40, 120, 80), 1, cv::LINE_AA);

    std::snprintf(buf, sizeof(buf), "raw err:    %.2f m", raw_err);
    cv::putText(img, buf, cv::Point(ox + 18, 166), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(40, 110, 215), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "smooth err: %.2f m", smooth_err);
    cv::putText(img, buf, cv::Point(ox + 18, 192), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(45, 150, 70), 1, cv::LINE_AA);

    std::snprintf(buf, sizeof(buf), "track jitter raw:  %.3f", partial.raw_jitter);
    cv::putText(img, buf, cv::Point(ox + 18, 234), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "track jitter smooth: %.3f", partial.smooth_jitter);
    cv::putText(img, buf, cv::Point(ox + 18, 256), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "lag: %d frames", SMOOTH_W);
    cv::putText(img, buf, cv::Point(ox + 18, 278), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "particles: %d", K_MEGA);
    cv::putText(img, buf, cv::Point(ox + 18, 300), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "avg GPU step: %.2f ms", partial.mega_ms);
    cv::putText(img, buf, cv::Point(ox + 18, 322), cv::FONT_HERSHEY_SIMPLEX, 0.44,
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

// Slice raw history up to step t for drawing (avoids drawing future frames).
static std::vector<Pose2> raw_hist_upto(const std::vector<Pose2>& h, int t) {
    return std::vector<Pose2>(h.begin(), h.begin() + (t + 1));
}

static FinalStats run_demo() {
    ensure_dirs();
    CpuMap map = make_map();
    DeviceMap dmap;
    dmap.upload(map);

    float *d_scan_x = nullptr, *d_scan_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scan_x, N_SCAN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_y, N_SCAN * sizeof(float)));

    ParticleSet mega;
    mega.alloc(K_MEGA, 5678);
    MegaBuckets buckets;
    buckets.alloc();

    std::vector<float> v_cmd, w_cmd;
    std::vector<Pose2> truth = make_truth(v_cmd, w_cmd);

    int mega_blocks = (K_MEGA + THREADS - 1) / THREADS;
    init_uniform_kernel<<<mega_blocks, THREADS>>>(mega.x, mega.y, mega.th, mega.w, mega.rng, mega.n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cv::VideoWriter video("tmp/gpu_megaparticles_smoother.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10,
                          cv::Size(FRAME_W, FRAME_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open output video\n");
        std::exit(1);
    }

    std::vector<float> scan_x, scan_y;
    std::vector<Pose2> raw_hist(N_STEPS), truth_hist;
    std::vector<Pose2> smooth_final(N_STEPS);
    std::vector<char> has_smooth(N_STEPS, 0);
    std::vector<char> visible_hist(N_STEPS, 0);

    FixedLagSmoother smoother;
    FinalStats stats;
    double mega_ms_sum = 0.0;
    Pose2 raw_track{};   // raw readout; coasts on CV while the scan is blocked
    bool have_raw = false;
    Pose2 smooth_marker{WORLD_W * 0.5f, WORLD_H * 0.5f, 0.0f};

    std::vector<Pose2> smooth_path_hist;  // finalized smoothed poses, in order

    for (int t = 0; t < N_STEPS; ++t) {
        bool visible = !(t >= KIDNAP_STEP && t < KIDNAP_STEP + OCCLUDE_STEPS);
        if (visible) {
            make_scan(map.rects, truth[t], t, scan_x, scan_y);
            CUDA_CHECK(cudaMemcpy(d_scan_x, scan_x.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_scan_y, scan_y.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));
        }

        Pose2 raw_est{};
        float best_mass = 0.0f;
        auto t0 = std::chrono::high_resolution_clock::now();
        mega_filter_step(mega, buckets, dmap, d_scan_x, d_scan_y, v_cmd[t], w_cmd[t], visible,
                         raw_est, best_mass);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double mega_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        mega_ms_sum += mega_ms;
        stats.mega_ms = mega_ms_sum / (t + 1);

        // While the scan is blocked the max-posterior readout is meaningless, so
        // the raw arm coasts on the odometry command; the smoother sees no
        // measurement for these frames and coasts on its motion factors.
        if (visible) {
            raw_track = raw_est;
            have_raw = true;
        } else if (have_raw) {
            raw_track = integrate_pose(raw_track, v_cmd[t], w_cmd[t]);
        } else {
            raw_track = raw_est;
        }
        raw_hist[t] = raw_track;
        visible_hist[t] = visible ? 1 : 0;
        truth_hist.push_back(truth[t]);

        SmoothFrame f;
        f.obs = raw_track;
        f.has_obs = visible;
        f.v = v_cmd[t];
        f.w = w_cmd[t];
        f.index = t;
        std::vector<Finalized> finalized;
        smoother.push(f, finalized);
        for (const Finalized& fin : finalized) {
            smooth_final[fin.index] = fin.pose;
            has_smooth[fin.index] = 1;
            smooth_path_hist.push_back(fin.pose);
        }
        // Live marker: the smoother's online (newest) estimate tracks the
        // current frame; the finalized path trails it by the fixed lag.
        smooth_marker = smoother.current(smooth_marker);

        float raw_err = pose_error_xy(raw_track, truth[t]);
        float smooth_err = pose_error_xy(smooth_marker, truth[t]);

        if (t % VIDEO_EVERY == 0 || t == N_STEPS - 1) {
            mega.copy_xy_to_host();
            cv::Mat frame(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(244, 246, 246));
            draw_panel(frame, 0, "raw max-posterior", map, truth_hist, raw_hist_upto(raw_hist, t),
                       truth[t], raw_track, mega.hx, mega.hy, std::max(1, K_MEGA / 3600),
                       cv::Scalar(225, 190, 150), cv::Scalar(215, 110, 40));
            draw_panel(frame, PANEL_W, "fixed-lag smoothed", map, truth_hist, smooth_path_hist,
                       truth[std::max(0, t)], smooth_marker, mega.hx, mega.hy,
                       std::max(1, K_MEGA / 3600), cv::Scalar(190, 215, 200), cv::Scalar(45, 150, 70));
            draw_info(frame, PANEL_W * 2, t, !visible, raw_err, smooth_err, stats);
            video.write(frame);
        }

        std::printf("step %3d visible=%d raw_err=%.3f smooth_err=%.3f mass=%.3g mega=%.2fms\n",
                    t, visible ? 1 : 0, raw_err, smooth_err, best_mass, mega_ms);
    }

    std::vector<Finalized> tail;
    smoother.flush(tail);
    for (const Finalized& fin : tail) {
        smooth_final[fin.index] = fin.pose;
        has_smooth[fin.index] = 1;
    }

    // --- Metrics over finalized frames ---
    auto rms = [](double sq, int c) { return c ? std::sqrt(sq / c) : 0.0; };
    double raw_track_sq = 0, smooth_track_sq = 0;
    int track_c = 0;
    double raw_post_sq = 0, smooth_post_sq = 0;
    int post_c = 0;
    int post_lo = KIDNAP_STEP + OCCLUDE_STEPS;
    for (int t = 0; t < N_STEPS; ++t) {
        if (!has_smooth[t]) continue;
        double re = pose_error_xy(raw_hist[t], truth[t]);
        double se = pose_error_xy(smooth_final[t], truth[t]);
        if (visible_hist[t] && t < KIDNAP_STEP) {
            raw_track_sq += re * re;
            smooth_track_sq += se * se;
            track_c++;
        }
        if (t >= post_lo) {
            raw_post_sq += re * re;
            smooth_post_sq += se * se;
            post_c++;
            if (stats.smooth_relock_step < 0 && se < 0.65) {
                stats.smooth_relock_step = t - post_lo;
            }
        }
    }
    stats.raw_track_rmse = static_cast<float>(rms(raw_track_sq, track_c));
    stats.smooth_track_rmse = static_cast<float>(rms(smooth_track_sq, track_c));
    stats.raw_post_rmse = static_cast<float>(rms(raw_post_sq, post_c));
    stats.smooth_post_rmse = static_cast<float>(rms(smooth_post_sq, post_c));
    stats.final_raw_err = pose_error_xy(raw_hist[N_STEPS - 1], truth[N_STEPS - 1]);
    stats.final_smooth_err = pose_error_xy(smooth_final[N_STEPS - 1], truth[N_STEPS - 1]);

    // Jitter = mean magnitude of the second difference of xy position (a
    // path-roughness / acceleration proxy) over the in-track visible frames.
    auto jitter = [&](const std::vector<Pose2>& seq) {
        double acc = 0;
        int c = 0;
        for (int t = 2; t < KIDNAP_STEP; ++t) {
            if (!(visible_hist[t] && visible_hist[t - 1] && visible_hist[t - 2])) continue;
            double ddx = seq[t].x - 2.0 * seq[t - 1].x + seq[t - 2].x;
            double ddy = seq[t].y - 2.0 * seq[t - 1].y + seq[t - 2].y;
            acc += std::sqrt(ddx * ddx + ddy * ddy);
            c++;
        }
        return c ? acc / c : 0.0;
    };
    stats.raw_jitter = static_cast<float>(jitter(raw_hist));
    stats.smooth_jitter = static_cast<float>(jitter(smooth_final));
    stats.truth_jitter = static_cast<float>(jitter(truth));

    video.release();
    avi_to_gif("tmp/gpu_megaparticles_smoother.avi", "gif/gpu_megaparticles_smoother.gif", 10, 760);

    CUDA_CHECK(cudaFree(d_scan_x));
    CUDA_CHECK(cudaFree(d_scan_y));
    buckets.free_all();
    mega.free_all();
    dmap.free_all();
    return stats;
}

}  // namespace cudabot

int main() {
    cudabot::FinalStats s = cudabot::run_demo();
    std::printf("\nMegaParticles representative-trajectory smoother (SE(2))\n");
    std::printf("in-track jitter (mean |d2 pos|): raw %.4f, smoothed %.4f, truth %.4f\n",
                s.raw_jitter, s.smooth_jitter, s.truth_jitter);
    std::printf("in-track RMSE: raw %.4f m, smoothed %.4f m\n", s.raw_track_rmse, s.smooth_track_rmse);
    std::printf("post-kidnap RMSE: raw %.4f m, smoothed %.4f m\n", s.raw_post_rmse, s.smooth_post_rmse);
    std::printf("final error: raw %.4f m, smoothed %.4f m\n", s.final_raw_err, s.final_smooth_err);
    std::printf("smoothed relock after blackout: %d frames\n", s.smooth_relock_step);
    std::printf("avg GPU step: %.4f ms\n", s.mega_ms);
    std::printf("Wrote gif/gpu_megaparticles_smoother.gif\n");
    return 0;
}
