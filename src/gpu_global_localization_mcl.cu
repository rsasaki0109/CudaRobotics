// gpu_global_localization_mcl.cu
//
// GPU global-localization / kidnapped-robot recovery demo.
//
// Two particle filters share the same map, controls, and landmark
// observations.  The local-only MCL filter starts correctly but has no
// particles near the hidden kidnapped pose, so it remains locked to the
// old mode.  The recovery filter detects a collapsed observation
// likelihood and replaces part of the particle set with sensor-reset
// hypotheses sampled from landmark range-bearing observations.
//
// Output: gif/gpu_global_localization_mcl.gif

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

constexpr int K_PART = 32768;
constexpr int N_LANDMARKS = 72;
constexpr int N_OBS = 10;
constexpr int N_STEPS = 180;
constexpr int KIDNAP_STEP = 70;
constexpr int THREADS = 256;
constexpr float WORLD_W = 36.0f;
constexpr float WORLD_H = 26.0f;
constexpr float DT = 0.12f;
constexpr float PI_F = 3.14159265358979323846f;
constexpr float RANGE_SIGMA = 0.18f;
constexpr float BEARING_SIGMA = 0.040f;
constexpr float MOTION_SIGMA_XY = 0.045f;
constexpr float MOTION_SIGMA_TH = 0.014f;
constexpr float RESET_RANGE_SIGMA = 0.24f;
constexpr float RESET_BEARING_SIGMA = 0.06f;
constexpr float MAX_SENSOR_RANGE = 18.0f;
constexpr float LOST_MAX_WEIGHT = 1.0e-16f;
constexpr float RESET_FRACTION = 0.42f;
constexpr int RESET_PARTICLES = static_cast<int>(K_PART * RESET_FRACTION);
constexpr int PANEL_W = 520;
constexpr int PANEL_H = 390;
constexpr int INFO_W = 330;
constexpr int FRAME_W = PANEL_W * 2 + INFO_W;
constexpr int FRAME_H = PANEL_H;
constexpr int VIDEO_FPS = 12;

struct Pose2 {
    float x;
    float y;
    float th;
};

struct ObsPack {
    int ids[N_OBS];
    float ranges[N_OBS];
    float bearings[N_OBS];
};

struct StepStats {
    float ex = 0.0f;
    float ey = 0.0f;
    float eth = 0.0f;
    float err = 0.0f;
    float max_w = 0.0f;
    float ess = 0.0f;
    float avg_lik = 0.0f;
    bool reset = false;
    double ms = 0.0;
};

struct BenchResult {
    float local_post_rmse = 0.0f;
    float recovery_post_rmse = 0.0f;
    float final_local_err = 0.0f;
    float final_recovery_err = 0.0f;
    int recovery_steps = -1;
    int reset_steps = 0;
    double local_ms = 0.0;
    double recovery_ms = 0.0;
};

__global__ void init_rng_kernel(curandState* states, unsigned long long seed);

struct ParticleSet {
    float *x = nullptr, *y = nullptr, *th = nullptr, *w = nullptr;
    float *x2 = nullptr, *y2 = nullptr, *th2 = nullptr, *wcum = nullptr;
    float *stats = nullptr, *mean = nullptr;
    curandState* rng = nullptr;
    std::vector<float> hx, hy;

    void alloc(unsigned long long seed) {
        CUDA_CHECK(cudaMalloc(&x, K_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&y, K_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&th, K_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&w, K_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&x2, K_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&y2, K_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&th2, K_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&wcum, K_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&stats, 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&mean, 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rng, K_PART * sizeof(curandState)));
        hx.resize(K_PART);
        hy.resize(K_PART);
        int blocks = (K_PART + THREADS - 1) / THREADS;
        init_rng_kernel<<<blocks, THREADS>>>(rng, seed);
        CUDA_CHECK(cudaGetLastError());
    }

    void free_all() {
        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
        CUDA_CHECK(cudaFree(th));
        CUDA_CHECK(cudaFree(w));
        CUDA_CHECK(cudaFree(x2));
        CUDA_CHECK(cudaFree(y2));
        CUDA_CHECK(cudaFree(th2));
        CUDA_CHECK(cudaFree(wcum));
        CUDA_CHECK(cudaFree(stats));
        CUDA_CHECK(cudaFree(mean));
        CUDA_CHECK(cudaFree(rng));
    }

    void copy_to_host() {
        CUDA_CHECK(cudaMemcpy(hx.data(), x, K_PART * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hy.data(), y, K_PART * sizeof(float), cudaMemcpyDeviceToHost));
    }
};

__host__ __device__ static inline float clampf(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}

__host__ __device__ static inline float wrap_angle(float a) {
    while (a > PI_F) a -= 2.0f * PI_F;
    while (a < -PI_F) a += 2.0f * PI_F;
    return a;
}

__global__ void init_rng_kernel(curandState* states, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    curand_init(seed, i, 0, &states[i]);
}

__global__ void init_gaussian_kernel(float* x,
                                     float* y,
                                     float* th,
                                     float* w,
                                     curandState* rng,
                                     float cx,
                                     float cy,
                                     float cth) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    curandState s = rng[i];
    x[i] = clampf(cx + 0.25f * curand_normal(&s), 0.4f, WORLD_W - 0.4f);
    y[i] = clampf(cy + 0.25f * curand_normal(&s), 0.4f, WORLD_H - 0.4f);
    th[i] = wrap_angle(cth + 0.12f * curand_normal(&s));
    w[i] = 1.0f / K_PART;
    rng[i] = s;
}

__global__ void predict_kernel(float* x,
                               float* y,
                               float* th,
                               curandState* rng,
                               float v,
                               float omega) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    curandState s = rng[i];
    float vt = v + MOTION_SIGMA_XY * curand_normal(&s);
    float wt = omega + MOTION_SIGMA_TH * curand_normal(&s);
    float theta = th[i];
    x[i] = clampf(x[i] + vt * cosf(theta) * DT, 0.4f, WORLD_W - 0.4f);
    y[i] = clampf(y[i] + vt * sinf(theta) * DT, 0.4f, WORLD_H - 0.4f);
    th[i] = wrap_angle(theta + wt * DT);
    rng[i] = s;
}

__global__ void weight_kernel(const float* __restrict__ x,
                              const float* __restrict__ y,
                              const float* __restrict__ th,
                              const float* __restrict__ lm_x,
                              const float* __restrict__ lm_y,
                              const int* __restrict__ obs_ids,
                              const float* __restrict__ obs_ranges,
                              const float* __restrict__ obs_bearings,
                              float* __restrict__ weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    float px = x[i], py = y[i], pth = th[i];
    float logw = 0.0f;
    #pragma unroll
    for (int k = 0; k < N_OBS; k++) {
        int id = obs_ids[k];
        float dx = lm_x[id] - px;
        float dy = lm_y[id] - py;
        float pr = sqrtf(dx * dx + dy * dy);
        float pb = wrap_angle(atan2f(dy, dx) - pth);
        float rr = (pr - obs_ranges[k]) / RANGE_SIGMA;
        float rb = wrap_angle(pb - obs_bearings[k]) / BEARING_SIGMA;
        logw += -0.5f * (rr * rr + rb * rb);
    }
    weights[i] = expf(fmaxf(logw, -80.0f));
}

__global__ void reduce_weight_stats_kernel(const float* weights, float* out) {
    extern __shared__ float sh[];
    float* s_sum = sh;
    float* s_sq = sh + blockDim.x;
    float* s_max = sh + 2 * blockDim.x;
    int tid = threadIdx.x;
    float sum = 0.0f, sq = 0.0f, mx = 0.0f;
    for (int i = tid; i < K_PART; i += blockDim.x) {
        float v = weights[i];
        sum += v;
        sq += v * v;
        mx = fmaxf(mx, v);
    }
    s_sum[tid] = sum;
    s_sq[tid] = sq;
    s_max[tid] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq[tid] += s_sq[tid + s];
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = s_sum[0];
        out[1] = s_sq[0];
        out[2] = s_max[0];
    }
}

__global__ void normalize_kernel(float* weights, float sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    if (sum > 1.0e-30f) weights[i] /= sum;
    else weights[i] = 1.0f / K_PART;
}

__global__ void weighted_mean_kernel(const float* x,
                                     const float* y,
                                     const float* th,
                                     const float* weights,
                                     float* out) {
    extern __shared__ float sh[];
    float* sx = sh;
    float* sy = sh + blockDim.x;
    float* sc = sh + 2 * blockDim.x;
    float* ss = sh + 3 * blockDim.x;
    int tid = threadIdx.x;
    float ax = 0.0f, ay = 0.0f, ac = 0.0f, as = 0.0f;
    for (int i = tid; i < K_PART; i += blockDim.x) {
        float wt = weights[i];
        ax += x[i] * wt;
        ay += y[i] * wt;
        ac += cosf(th[i]) * wt;
        as += sinf(th[i]) * wt;
    }
    sx[tid] = ax; sy[tid] = ay; sc[tid] = ac; ss[tid] = as;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sx[tid] += sx[tid + s];
            sy[tid] += sy[tid + s];
            sc[tid] += sc[tid + s];
            ss[tid] += ss[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = sx[0];
        out[1] = sy[0];
        out[2] = atan2f(ss[0], sc[0]);
    }
}

__global__ void cumsum_kernel(const float* weights, float* wcum) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float acc = 0.0f;
    for (int i = 0; i < K_PART; i++) {
        acc += weights[i];
        wcum[i] = acc;
    }
    wcum[K_PART - 1] = 1.0f;
}

__global__ void resample_kernel(const float* x,
                                const float* y,
                                const float* th,
                                const float* wcum,
                                float* x2,
                                float* y2,
                                float* th2,
                                float offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    float base = 1.0f / K_PART;
    float target = fminf(0.99999994f, offset + base * i);
    int lo = 0, hi = K_PART - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (wcum[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    x2[i] = x[lo];
    y2[i] = y[lo];
    th2[i] = th[lo];
}

__global__ void reset_uniform_weights_kernel(float* weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    weights[i] = 1.0f / K_PART;
}

__global__ void sensor_reset_kernel(float* x,
                                    float* y,
                                    float* th,
                                    const float* lm_x,
                                    const float* lm_y,
                                    const int* obs_ids,
                                    const float* obs_ranges,
                                    const float* obs_bearings,
                                    curandState* rng,
                                    int n_reset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_reset) return;
    curandState s = rng[i];
    int k = min(N_OBS - 1, static_cast<int>(curand_uniform(&s) * N_OBS));
    int id = obs_ids[k];
    float theta = curand_uniform(&s) * 2.0f * PI_F - PI_F;
    float range = fmaxf(0.1f, obs_ranges[k] + RESET_RANGE_SIGMA * curand_normal(&s));
    float bearing = obs_bearings[k] + RESET_BEARING_SIGMA * curand_normal(&s);
    float global_bearing = theta + bearing;
    float px = lm_x[id] - range * cosf(global_bearing);
    float py = lm_y[id] - range * sinf(global_bearing);
    x[i] = clampf(px + 0.08f * curand_normal(&s), 0.4f, WORLD_W - 0.4f);
    y[i] = clampf(py + 0.08f * curand_normal(&s), 0.4f, WORLD_H - 0.4f);
    th[i] = wrap_angle(theta);
    rng[i] = s;
}

static void swap_particle_buffers(ParticleSet& p) {
    std::swap(p.x, p.x2);
    std::swap(p.y, p.y2);
    std::swap(p.th, p.th2);
}

static void init_particles(ParticleSet& p, const Pose2& pose) {
    int blocks = (K_PART + THREADS - 1) / THREADS;
    init_gaussian_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.w, p.rng,
                                              pose.x, pose.y, pose.th);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

static float pose_error_xy(const Pose2& p, float ex, float ey) {
    float dx = p.x - ex;
    float dy = p.y - ey;
    return std::sqrt(dx * dx + dy * dy);
}

static void make_landmarks(std::vector<float>& lx, std::vector<float>& ly) {
    lx.clear(); ly.clear();
    lx.reserve(N_LANDMARKS); ly.reserve(N_LANDMARKS);
    for (int iy = 0; iy < 6; iy++) {
        for (int ix = 0; ix < 12; ix++) {
            float x = 2.4f + ix * 2.75f + 0.35f * std::sin(1.7f * iy + 0.4f * ix);
            float y = 2.2f + iy * 4.10f + 0.42f * std::cos(0.9f * ix - 0.5f * iy);
            lx.push_back(clampf(x, 1.0f, WORLD_W - 1.0f));
            ly.push_back(clampf(y, 1.0f, WORLD_H - 1.0f));
        }
    }
}

static void controls_at(int step, float& v, float& omega) {
    float t = step * DT;
    v = 0.58f + 0.08f * std::sin(0.37f * t);
    omega = 0.34f * std::sin(0.51f * t) + 0.12f * std::cos(0.17f * t);
}

static void advance_pose(Pose2& p, float v, float omega) {
    p.x += v * std::cos(p.th) * DT;
    p.y += v * std::sin(p.th) * DT;
    p.th = wrap_angle(p.th + omega * DT);
    p.x = clampf(p.x, 1.2f, WORLD_W - 1.2f);
    p.y = clampf(p.y, 1.2f, WORLD_H - 1.2f);
}

static ObsPack observe_pose(const Pose2& pose,
                            const std::vector<float>& lx,
                            const std::vector<float>& ly,
                            std::mt19937& rng) {
    std::vector<std::pair<float, int> > ranked;
    ranked.reserve(N_LANDMARKS);
    for (int i = 0; i < N_LANDMARKS; i++) {
        float dx = lx[i] - pose.x;
        float dy = ly[i] - pose.y;
        float d = std::sqrt(dx * dx + dy * dy);
        if (d < MAX_SENSOR_RANGE) ranked.push_back(std::make_pair(d, i));
    }
    if (ranked.empty()) {
        for (int i = 0; i < N_LANDMARKS; i++) {
            float dx = lx[i] - pose.x;
            float dy = ly[i] - pose.y;
            ranked.push_back(std::make_pair(std::sqrt(dx * dx + dy * dy), i));
        }
    }
    std::sort(ranked.begin(), ranked.end());
    std::normal_distribution<float> nr(0.0f, RANGE_SIGMA);
    std::normal_distribution<float> nb(0.0f, BEARING_SIGMA);
    ObsPack obs{};
    for (int k = 0; k < N_OBS; k++) {
        int id = ranked[std::min(k, static_cast<int>(ranked.size()) - 1)].second;
        float dx = lx[id] - pose.x;
        float dy = ly[id] - pose.y;
        obs.ids[k] = id;
        obs.ranges[k] = fmaxf(0.1f, std::sqrt(dx * dx + dy * dy) + nr(rng));
        obs.bearings[k] = wrap_angle(std::atan2(dy, dx) - pose.th + nb(rng));
    }
    return obs;
}

static StepStats run_gpu_step(ParticleSet& p,
                              const float* d_lx,
                              const float* d_ly,
                              int* d_obs_ids,
                              float* d_obs_ranges,
                              float* d_obs_bearings,
                              const ObsPack& obs,
                              const Pose2& gt,
                              float v,
                              float omega,
                              bool enable_recovery,
                              std::mt19937& rng_host) {
    int blocks = (K_PART + THREADS - 1) / THREADS;
    CUDA_CHECK(cudaMemcpy(d_obs_ids, obs.ids, N_OBS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_ranges, obs.ranges, N_OBS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_bearings, obs.bearings, N_OBS * sizeof(float), cudaMemcpyHostToDevice));

    auto t0 = std::chrono::high_resolution_clock::now();
    predict_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng, v, omega);
    weight_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, d_lx, d_ly, d_obs_ids,
                                       d_obs_ranges, d_obs_bearings, p.w);
    reduce_weight_stats_kernel<<<1, THREADS, 3 * THREADS * sizeof(float)>>>(p.w, p.stats);
    CUDA_CHECK(cudaGetLastError());

    float h_stats[3];
    CUDA_CHECK(cudaMemcpy(h_stats, p.stats, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    bool lost = enable_recovery && h_stats[2] < LOST_MAX_WEIGHT;
    if (lost) {
        int reset_blocks = (RESET_PARTICLES + THREADS - 1) / THREADS;
        sensor_reset_kernel<<<reset_blocks, THREADS>>>(p.x, p.y, p.th, d_lx, d_ly,
                                                       d_obs_ids, d_obs_ranges,
                                                       d_obs_bearings, p.rng,
                                                       RESET_PARTICLES);
        weight_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, d_lx, d_ly, d_obs_ids,
                                           d_obs_ranges, d_obs_bearings, p.w);
        reduce_weight_stats_kernel<<<1, THREADS, 3 * THREADS * sizeof(float)>>>(p.w, p.stats);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_stats, p.stats, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    }

    float sum_w = h_stats[0];
    float sum_sq = h_stats[1];
    normalize_kernel<<<blocks, THREADS>>>(p.w, sum_w);
    weighted_mean_kernel<<<1, THREADS, 4 * THREADS * sizeof(float)>>>(p.x, p.y, p.th, p.w, p.mean);
    cumsum_kernel<<<1, 1>>>(p.w, p.wcum);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f / K_PART);
    resample_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.wcum, p.x2, p.y2, p.th2,
                                         u01(rng_host));
    swap_particle_buffers(p);
    reset_uniform_weights_kernel<<<blocks, THREADS>>>(p.w);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    float h_mean[3];
    CUDA_CHECK(cudaMemcpy(h_mean, p.mean, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    StepStats s;
    s.ex = h_mean[0];
    s.ey = h_mean[1];
    s.eth = h_mean[2];
    s.err = pose_error_xy(gt, s.ex, s.ey);
    s.max_w = h_stats[2];
    s.avg_lik = sum_w / K_PART;
    s.ess = sum_sq > 1.0e-30f ? (sum_w * sum_w / sum_sq) : 1.0f;
    s.reset = lost;
    s.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return s;
}

static cv::Point2i wp(float x, float y, int offx, int offy) {
    int px = offx + static_cast<int>(x / WORLD_W * PANEL_W);
    int py = offy + PANEL_H - 1 - static_cast<int>(y / WORLD_H * PANEL_H);
    return cv::Point2i(px, py);
}

static void draw_panel(cv::Mat& img,
                       int offx,
                       const char* title,
                       const std::vector<float>& lx,
                       const std::vector<float>& ly,
                       const std::vector<Pose2>& trail,
                       const std::vector<cv::Point2f>& est_trail,
                       const std::vector<float>& px,
                       const std::vector<float>& py,
                       const Pose2& gt,
                       const StepStats& stats,
                       cv::Scalar color) {
    cv::Rect r(offx, 0, PANEL_W, PANEL_H);
    cv::rectangle(img, r, cv::Scalar(24, 27, 31), -1);
    cv::rectangle(img, r, cv::Scalar(73, 80, 88), 1);
    for (size_t i = 0; i < lx.size(); i++) {
        cv::circle(img, wp(lx[i], ly[i], offx, 0), 2, cv::Scalar(120, 175, 210), -1, cv::LINE_AA);
    }
    for (size_t i = 1; i < trail.size(); i++) {
        cv::line(img, wp(trail[i - 1].x, trail[i - 1].y, offx, 0),
                 wp(trail[i].x, trail[i].y, offx, 0), cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
    }
    for (size_t i = 1; i < est_trail.size(); i++) {
        cv::line(img, wp(est_trail[i - 1].x, est_trail[i - 1].y, offx, 0),
                 wp(est_trail[i].x, est_trail[i].y, offx, 0), color, 1, cv::LINE_AA);
    }
    for (int i = 0; i < K_PART; i += 28) {
        cv::Point2i q = wp(px[i], py[i], offx, 0);
        if (q.x >= offx && q.x < offx + PANEL_W && q.y >= 0 && q.y < PANEL_H) {
            img.at<cv::Vec3b>(q.y, q.x) = cv::Vec3b(
                static_cast<unsigned char>(color[0] * 0.62),
                static_cast<unsigned char>(color[1] * 0.62),
                static_cast<unsigned char>(color[2] * 0.62));
        }
    }
    cv::Point2i g = wp(gt.x, gt.y, offx, 0);
    cv::circle(img, g, 6, cv::Scalar(245, 245, 245), -1, cv::LINE_AA);
    cv::circle(img, wp(stats.ex, stats.ey, offx, 0), 8, color, 2, cv::LINE_AA);
    float hx = 0.55f * std::cos(gt.th);
    float hy = 0.55f * std::sin(gt.th);
    cv::line(img, g, wp(gt.x + hx, gt.y + hy, offx, 0), cv::Scalar(245, 245, 245), 2, cv::LINE_AA);
    cv::putText(img, title, cv::Point(offx + 16, 28), cv::FONT_HERSHEY_SIMPLEX, 0.62,
                cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, cv::format("err %.2f m  ESS %.0f", stats.err, stats.ess),
                cv::Point(offx + 16, 52), cv::FONT_HERSHEY_SIMPLEX, 0.43,
                cv::Scalar(220, 225, 230), 1, cv::LINE_AA);
    if (stats.reset) {
        cv::putText(img, "sensor reset", cv::Point(offx + 16, 76),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(70, 210, 255), 1,
                    cv::LINE_AA);
    }
}

static cv::Mat draw_frame(int step,
                          const std::vector<float>& lx,
                          const std::vector<float>& ly,
                          const std::vector<Pose2>& trail,
                          const std::vector<cv::Point2f>& local_trail,
                          const std::vector<cv::Point2f>& rec_trail,
                          ParticleSet& local,
                          ParticleSet& recovery,
                          const Pose2& gt,
                          const StepStats& sl,
                          const StepStats& sr,
                          const BenchResult& bench) {
    cv::Mat img(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(18, 20, 23));
    draw_panel(img, 0, "local MCL", lx, ly, trail, local_trail, local.hx, local.hy,
               gt, sl, cv::Scalar(80, 120, 235));
    draw_panel(img, PANEL_W, "global recovery MCL", lx, ly, trail, rec_trail,
               recovery.hx, recovery.hy, gt, sr, cv::Scalar(65, 210, 145));
    int ix = PANEL_W * 2;
    cv::rectangle(img, cv::Rect(ix, 0, INFO_W, FRAME_H), cv::Scalar(28, 31, 36), -1);
    cv::rectangle(img, cv::Rect(ix, 0, INFO_W, FRAME_H), cv::Scalar(73, 80, 88), 1);
    cv::putText(img, "GPU localization recovery", cv::Point(ix + 18, 32),
                cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(245, 245, 245), 1,
                cv::LINE_AA);
    cv::putText(img, cv::format("step %03d / %d", step, N_STEPS),
                cv::Point(ix + 18, 62), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(210, 216, 224), 1, cv::LINE_AA);
    cv::putText(img, cv::format("%d particles, %d landmarks", K_PART, N_LANDMARKS),
                cv::Point(ix + 18, 96), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(178, 185, 194), 1, cv::LINE_AA);
    cv::putText(img, cv::format("%d range-bearing obs", N_OBS),
                cv::Point(ix + 18, 120), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(178, 185, 194), 1, cv::LINE_AA);
    if (step >= KIDNAP_STEP) {
        cv::putText(img, "hidden kidnap active", cv::Point(ix + 18, 156),
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(75, 120, 245), 1,
                    cv::LINE_AA);
    } else {
        cv::putText(img, cv::format("kidnap at step %d", KIDNAP_STEP),
                    cv::Point(ix + 18, 156), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                    cv::Scalar(178, 185, 194), 1, cv::LINE_AA);
    }
    cv::putText(img, cv::format("local post RMSE %.2f m", bench.local_post_rmse),
                cv::Point(ix + 18, 208), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(120, 145, 245), 1, cv::LINE_AA);
    cv::putText(img, cv::format("recovery post RMSE %.2f m", bench.recovery_post_rmse),
                cv::Point(ix + 18, 236), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(90, 225, 150), 1, cv::LINE_AA);
    cv::putText(img, cv::format("reset steps %d", bench.reset_steps),
                cv::Point(ix + 18, 276), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                cv::Scalar(220, 225, 230), 1, cv::LINE_AA);
    if (bench.recovery_steps >= 0) {
        std::string msg = bench.recovery_steps == 0
            ? "same-frame reacquire"
            : cv::format("reacquired in %d steps", bench.recovery_steps);
        cv::putText(img, msg, cv::Point(ix + 18, 304), cv::FONT_HERSHEY_SIMPLEX,
                    0.46, cv::Scalar(90, 225, 150), 1, cv::LINE_AA);
    }
    cv::putText(img, cv::format("avg local %.3f ms", bench.local_ms),
                cv::Point(ix + 18, 344), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(210, 216, 224), 1, cv::LINE_AA);
    cv::putText(img, cv::format("avg recovery %.3f ms", bench.recovery_ms),
                cv::Point(ix + 18, 370), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(210, 216, 224), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

int main() {
    using namespace cudabot;
    int mkdir_rc = std::system("mkdir -p gif");
    if (mkdir_rc != 0) {
        std::fprintf(stderr, "Failed to create gif directory\n");
        return 1;
    }

    std::vector<float> lx, ly;
    make_landmarks(lx, ly);
    float *d_lx = nullptr, *d_ly = nullptr;
    int* d_obs_ids = nullptr;
    float *d_obs_ranges = nullptr, *d_obs_bearings = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lx, N_LANDMARKS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ly, N_LANDMARKS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obs_ids, N_OBS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_obs_ranges, N_OBS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obs_bearings, N_OBS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lx, lx.data(), N_LANDMARKS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ly, ly.data(), N_LANDMARKS * sizeof(float), cudaMemcpyHostToDevice));

    Pose2 gt{5.6f, 5.2f, 0.34f};
    ParticleSet local, recovery;
    local.alloc(25052026ULL);
    recovery.alloc(25052027ULL);
    init_particles(local, gt);
    init_particles(recovery, gt);

    std::mt19937 rng_obs(77);
    std::mt19937 rng_local(101);
    std::mt19937 rng_recovery(202);
    std::vector<Pose2> gt_trail;
    std::vector<cv::Point2f> local_trail, rec_trail;
    gt_trail.reserve(N_STEPS);
    local_trail.reserve(N_STEPS);
    rec_trail.reserve(N_STEPS);

    BenchResult bench;
    double local_err_post = 0.0;
    double rec_err_post = 0.0;
    double local_ms_sum = 0.0;
    double recovery_ms_sum = 0.0;
    int post_count = 0;
    bool reacquired = false;

    cv::VideoWriter video("gif/gpu_global_localization_mcl.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS,
                          cv::Size(FRAME_W, FRAME_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "Failed to open gif/gpu_global_localization_mcl.avi\n");
        return 1;
    }

    StepStats sl, sr;
    for (int step = 0; step < N_STEPS; step++) {
        float v, omega;
        controls_at(step, v, omega);
        if (step == KIDNAP_STEP) {
            gt = Pose2{28.6f, 19.0f, -2.35f};
        } else {
            advance_pose(gt, v, omega);
        }
        ObsPack obs = observe_pose(gt, lx, ly, rng_obs);
        sl = run_gpu_step(local, d_lx, d_ly, d_obs_ids, d_obs_ranges, d_obs_bearings,
                          obs, gt, v, omega, false, rng_local);
        sr = run_gpu_step(recovery, d_lx, d_ly, d_obs_ids, d_obs_ranges, d_obs_bearings,
                          obs, gt, v, omega, true, rng_recovery);
        local_ms_sum += sl.ms;
        recovery_ms_sum += sr.ms;
        bench.local_ms = local_ms_sum / (step + 1);
        bench.recovery_ms = recovery_ms_sum / (step + 1);
        if (sr.reset) bench.reset_steps++;
        if (step >= KIDNAP_STEP) {
            local_err_post += sl.err * sl.err;
            rec_err_post += sr.err * sr.err;
            post_count++;
            if (!reacquired && sr.err < 0.65f) {
                bench.recovery_steps = step - KIDNAP_STEP;
                reacquired = true;
            }
            bench.local_post_rmse = std::sqrt(local_err_post / post_count);
            bench.recovery_post_rmse = std::sqrt(rec_err_post / post_count);
        }
        bench.final_local_err = sl.err;
        bench.final_recovery_err = sr.err;
        gt_trail.push_back(gt);
        local_trail.push_back(cv::Point2f(sl.ex, sl.ey));
        rec_trail.push_back(cv::Point2f(sr.ex, sr.ey));
        if (step % 2 == 0 || step == N_STEPS - 1 || step == KIDNAP_STEP) {
            local.copy_to_host();
            recovery.copy_to_host();
            video.write(draw_frame(step, lx, ly, gt_trail, local_trail, rec_trail,
                                   local, recovery, gt, sl, sr, bench));
        }
        if (step < 6 || step == KIDNAP_STEP || step % 20 == 19) {
            std::printf("step %03d  local err %.3f m  recovery err %.3f m  "
                        "maxW %.2e reset %d\n",
                        step, sl.err, sr.err, sr.max_w, sr.reset ? 1 : 0);
        }
    }
    video.release();

    if (bench.recovery_steps < 0) bench.recovery_steps = N_STEPS - KIDNAP_STEP;

    avi_to_gif("gif/gpu_global_localization_mcl.avi",
               "gif/gpu_global_localization_mcl.gif",
               VIDEO_FPS, 900);

    std::printf("GPU global localization MCL\n");
    std::printf("particles: %d, landmarks: %d, observations: %d, kidnap step: %d\n",
                K_PART, N_LANDMARKS, N_OBS, KIDNAP_STEP);
    std::printf("post-kidnap RMSE: local %.4f m, recovery %.4f m\n",
                bench.local_post_rmse, bench.recovery_post_rmse);
    std::printf("final error: local %.4f m, recovery %.4f m\n",
                bench.final_local_err, bench.final_recovery_err);
    if (bench.recovery_steps == 0) {
        std::printf("recovery reacquired in the kidnap frame, sensor reset triggered %d/%d steps\n",
                    bench.reset_steps, N_STEPS);
    } else {
        std::printf("recovery reacquired in %d steps, sensor reset triggered %d/%d steps\n",
                    bench.recovery_steps, bench.reset_steps, N_STEPS);
    }
    std::printf("avg GPU step: local %.4f ms, recovery %.4f ms\n",
                bench.local_ms, bench.recovery_ms);
    std::printf("Wrote gif/gpu_global_localization_mcl.gif\n");

    local.free_all();
    recovery.free_all();
    CUDA_CHECK(cudaFree(d_lx));
    CUDA_CHECK(cudaFree(d_ly));
    CUDA_CHECK(cudaFree(d_obs_ids));
    CUDA_CHECK(cudaFree(d_obs_ranges));
    CUDA_CHECK(cudaFree(d_obs_bearings));
    return 0;
}
