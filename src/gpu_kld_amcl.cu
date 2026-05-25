// gpu_kld_amcl.cu
//
// GPU augmented KLD-sampling adaptive MCL with kidnap recovery.
//
// Two textbook ideas are combined on the same landmark range-bearing
// kidnap scenario used by gpu_global_localization_mcl:
//
//   1. KLD-sampling (Fox, "Adapting the Sample Size in Particle Filters
//      Through KLD-Sampling", NIPS 2001 / IJRR 2003): the number of
//      particles each step is set from the Kullback-Leibler bound on the
//      number of occupied (x, y, theta) histogram bins, so the filter uses
//      few particles when localized and many when the posterior is spread.
//
//   2. Augmented MCL (Thrun/Burgard/Fox, Probabilistic Robotics, Table
//      8.3): short- and long-term average-likelihood trackers w_fast /
//      w_slow drive a per-particle injection probability of uniform global
//      poses, so the filter recovers from a hidden kidnap without any
//      explicit sensor-reset rule.
//
// The occupied-bin count is taken from the current particle set as a
// practical parallel approximation to Fox's sequential stop rule. The same
// adaptive filter runs on the GPU (one thread per active particle) and on
// the CPU for a direct timing comparison.
//
// Output: gif/gpu_kld_amcl.gif

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int MAX_PART = 65536;
constexpr int INIT_PART = 2000;
constexpr int MIN_PART = 400;
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
constexpr float MAX_SENSOR_RANGE = 18.0f;

// KLD-sampling bound parameters.
constexpr float KLD_BIN_XY = 1.0f;
constexpr float KLD_BIN_TH = 0.3490658504f;  // 20 degrees
constexpr int KLD_NX = static_cast<int>(WORLD_W / KLD_BIN_XY) + 1;
constexpr int KLD_NY = static_cast<int>(WORLD_H / KLD_BIN_XY) + 1;
constexpr int KLD_NTH = 18;
constexpr int KLD_NBINS = KLD_NX * KLD_NY * KLD_NTH;
constexpr float KLD_EPSILON = 0.02f;
constexpr float KLD_Z = 2.32634787f;  // upper 0.99 quantile of N(0,1)

// Augmented-MCL injection trackers (0 <= alpha_slow << alpha_fast).
constexpr float ALPHA_SLOW = 0.01f;
constexpr float ALPHA_FAST = 0.10f;
constexpr float MAX_INJECT = 0.35f;
// Only inject once the short-term likelihood drops well below the long-term
// baseline, so transient resampling noise does not retrigger after recovery.
constexpr float INJECT_DEADBAND = 0.4f;

constexpr int MAP_W = 600;
constexpr int MAP_H = 440;
constexpr int INFO_W = 348;
constexpr int FRAME_W = MAP_W + INFO_W;
constexpr int FRAME_H = MAP_H;
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
    int n_part = 0;
    int occupied = 0;
    float p_inject = 0.0f;
    double ms = 0.0;
};

__host__ __device__ static inline float clampf(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}

__host__ __device__ static inline float wrap_angle(float a) {
    while (a > PI_F) a -= 2.0f * PI_F;
    while (a < -PI_F) a += 2.0f * PI_F;
    return a;
}

// Number of particles for the next step from the KLD bound on k occupied
// histogram bins (Fox 2003, eq. for the chi-square Wilson-Hilferty form).
static int kld_sample_size(int k) {
    if (k <= 1) return MIN_PART;
    double kk = static_cast<double>(k - 1);
    double inner = 1.0 - 2.0 / (9.0 * kk) + std::sqrt(2.0 / (9.0 * kk)) * KLD_Z;
    double n = (kk / (2.0 * KLD_EPSILON)) * inner * inner * inner;
    int ni = static_cast<int>(std::ceil(n));
    return std::max(MIN_PART, std::min(MAX_PART, ni));
}

__host__ __device__ static inline int bin_index(float x, float y, float th) {
    int bx = static_cast<int>(clampf(x / KLD_BIN_XY, 0.0f, KLD_NX - 1.0f));
    int by = static_cast<int>(clampf(y / KLD_BIN_XY, 0.0f, KLD_NY - 1.0f));
    int bt = static_cast<int>((wrap_angle(th) + PI_F) / KLD_BIN_TH);
    if (bt >= KLD_NTH) bt = KLD_NTH - 1;
    if (bt < 0) bt = 0;
    return (bt * KLD_NY + by) * KLD_NX + bx;
}

__global__ void init_rng_kernel(curandState* states, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MAX_PART) return;
    curand_init(seed, i, 0, &states[i]);
}

__global__ void init_gaussian_kernel(float* x, float* y, float* th, float* w,
                                     curandState* rng, int n,
                                     float cx, float cy, float cth) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState s = rng[i];
    x[i] = clampf(cx + 0.25f * curand_normal(&s), 0.4f, WORLD_W - 0.4f);
    y[i] = clampf(cy + 0.25f * curand_normal(&s), 0.4f, WORLD_H - 0.4f);
    th[i] = wrap_angle(cth + 0.12f * curand_normal(&s));
    w[i] = 1.0f / n;
    rng[i] = s;
}

__global__ void predict_kernel(float* x, float* y, float* th, curandState* rng,
                               int n, float v, float omega) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
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
                              int n,
                              float* __restrict__ weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
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

__global__ void reduce_sum_kernel(const float* weights, int n, float* out) {
    extern __shared__ float ssum[];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) sum += weights[i];
    ssum[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ssum[tid] += ssum[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = ssum[0];
}

__global__ void normalize_kernel(float* weights, int n, float sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (sum > 1.0e-30f) weights[i] /= sum;
    else weights[i] = 1.0f / n;
}

__global__ void weighted_mean_kernel(const float* x, const float* y,
                                     const float* th, const float* weights,
                                     int n, float* out) {
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

__global__ void cumsum_kernel(const float* weights, int n, float* wcum) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        acc += weights[i];
        wcum[i] = acc;
    }
    wcum[n - 1] = 1.0f;
}

__global__ void clear_bins_kernel(int* occ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= KLD_NBINS) return;
    occ[i] = 0;
}

__global__ void mark_bins_kernel(const float* x, const float* y, const float* th,
                                 int n, int* occ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    occ[bin_index(x[i], y[i], th[i])] = 1;
}

__global__ void count_bins_kernel(const int* occ, int* out) {
    extern __shared__ int sc[];
    int tid = threadIdx.x;
    int sum = 0;
    for (int i = tid; i < KLD_NBINS; i += blockDim.x) sum += occ[i];
    sc[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sc[tid] += sc[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = sc[0];
}

// Augmented resample: draw n_new particles from the old cumulative weights,
// but with probability p_inject replace a draw with a uniform global pose.
__global__ void resample_augment_kernel(const float* x, const float* y,
                                        const float* th, const float* wcum,
                                        int n_old, int n_new, float p_inject,
                                        curandState* rng,
                                        float* x2, float* y2, float* th2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_new) return;
    curandState s = rng[i];
    if (curand_uniform(&s) < p_inject) {
        x2[i] = 0.4f + curand_uniform(&s) * (WORLD_W - 0.8f);
        y2[i] = 0.4f + curand_uniform(&s) * (WORLD_H - 0.8f);
        th2[i] = curand_uniform(&s) * 2.0f * PI_F - PI_F;
    } else {
        float target = fminf(0.99999994f, curand_uniform(&s));
        int lo = 0, hi = n_old - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (wcum[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        x2[i] = x[lo];
        y2[i] = y[lo];
        th2[i] = th[lo];
    }
    rng[i] = s;
}

__global__ void reset_uniform_weights_kernel(float* weights, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    weights[i] = 1.0f / n;
}

// ----------------------------------------------------------------------
// GPU filter state
// ----------------------------------------------------------------------
struct GpuAmcl {
    float *x = nullptr, *y = nullptr, *th = nullptr, *w = nullptr;
    float *x2 = nullptr, *y2 = nullptr, *th2 = nullptr, *wcum = nullptr;
    float *scalar = nullptr, *mean = nullptr;
    int *occ = nullptr, *icount = nullptr;
    curandState* rng = nullptr;
    std::vector<float> hx, hy;
    int n_active = INIT_PART;
    float w_slow = 0.0f;
    float w_fast = 0.0f;

    void alloc(unsigned long long seed, const Pose2& pose) {
        CUDA_CHECK(cudaMalloc(&x, MAX_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&y, MAX_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&th, MAX_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&w, MAX_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&x2, MAX_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&y2, MAX_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&th2, MAX_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&wcum, MAX_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&scalar, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&mean, 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&occ, KLD_NBINS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&icount, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&rng, MAX_PART * sizeof(curandState)));
        hx.resize(MAX_PART);
        hy.resize(MAX_PART);
        int blocks = (MAX_PART + THREADS - 1) / THREADS;
        init_rng_kernel<<<blocks, THREADS>>>(rng, seed);
        int ib = (INIT_PART + THREADS - 1) / THREADS;
        init_gaussian_kernel<<<ib, THREADS>>>(x, y, th, w, rng, INIT_PART,
                                              pose.x, pose.y, pose.th);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void free_all() {
        for (void* p : {(void*)x, (void*)y, (void*)th, (void*)w, (void*)x2,
                        (void*)y2, (void*)th2, (void*)wcum, (void*)scalar,
                        (void*)mean, (void*)occ, (void*)icount, (void*)rng}) {
            CUDA_CHECK(cudaFree(p));
        }
    }

    void swap_buffers() {
        std::swap(x, x2);
        std::swap(y, y2);
        std::swap(th, th2);
    }

    void copy_to_host() {
        CUDA_CHECK(cudaMemcpy(hx.data(), x, n_active * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hy.data(), y, n_active * sizeof(float), cudaMemcpyDeviceToHost));
    }
};

static float pose_error_xy(const Pose2& p, float ex, float ey) {
    float dx = p.x - ex;
    float dy = p.y - ey;
    return std::sqrt(dx * dx + dy * dy);
}

static StepStats run_gpu_step(GpuAmcl& f, const float* d_lx, const float* d_ly,
                              int* d_obs_ids, float* d_obs_ranges, float* d_obs_bearings,
                              const ObsPack& obs, const Pose2& gt, float v, float omega) {
    CUDA_CHECK(cudaMemcpy(d_obs_ids, obs.ids, N_OBS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_ranges, obs.ranges, N_OBS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_bearings, obs.bearings, N_OBS * sizeof(float), cudaMemcpyHostToDevice));

    int n = f.n_active;
    int blocks = (n + THREADS - 1) / THREADS;
    int bin_blocks = (KLD_NBINS + THREADS - 1) / THREADS;

    auto t0 = std::chrono::high_resolution_clock::now();
    predict_kernel<<<blocks, THREADS>>>(f.x, f.y, f.th, f.rng, n, v, omega);
    weight_kernel<<<blocks, THREADS>>>(f.x, f.y, f.th, d_lx, d_ly, d_obs_ids,
                                       d_obs_ranges, d_obs_bearings, n, f.w);
    reduce_sum_kernel<<<1, THREADS, THREADS * sizeof(float)>>>(f.w, n, f.scalar);
    float sum_w = 0.0f;
    CUDA_CHECK(cudaMemcpy(&sum_w, f.scalar, sizeof(float), cudaMemcpyDeviceToHost));
    float w_avg = sum_w / n;

    // Augmented-MCL injection probability.
    if (f.w_slow == 0.0f) { f.w_slow = w_avg; f.w_fast = w_avg; }
    f.w_slow += ALPHA_SLOW * (w_avg - f.w_slow);
    f.w_fast += ALPHA_FAST * (w_avg - f.w_fast);
    float p_inject = 0.0f;
    if (f.w_slow > 1.0e-30f) {
        float drop = 1.0f - f.w_fast / f.w_slow;
        if (drop > INJECT_DEADBAND) p_inject = fminf(drop, MAX_INJECT);
    }

    normalize_kernel<<<blocks, THREADS>>>(f.w, n, sum_w);
    weighted_mean_kernel<<<1, THREADS, 4 * THREADS * sizeof(float)>>>(f.x, f.y, f.th, f.w, n, f.mean);

    // KLD bin occupancy of the current set drives the next sample size.
    clear_bins_kernel<<<bin_blocks, THREADS>>>(f.occ);
    mark_bins_kernel<<<blocks, THREADS>>>(f.x, f.y, f.th, n, f.occ);
    count_bins_kernel<<<1, THREADS, THREADS * sizeof(int)>>>(f.occ, f.icount);
    int occupied = 0;
    CUDA_CHECK(cudaMemcpy(&occupied, f.icount, sizeof(int), cudaMemcpyDeviceToHost));
    int n_new = kld_sample_size(occupied);

    cumsum_kernel<<<1, 1>>>(f.w, n, f.wcum);
    int rs_blocks = (n_new + THREADS - 1) / THREADS;
    resample_augment_kernel<<<rs_blocks, THREADS>>>(f.x, f.y, f.th, f.wcum, n, n_new,
                                                    p_inject, f.rng, f.x2, f.y2, f.th2);
    f.swap_buffers();
    reset_uniform_weights_kernel<<<rs_blocks, THREADS>>>(f.w, n_new);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    float h_mean[3];
    CUDA_CHECK(cudaMemcpy(h_mean, f.mean, 3 * sizeof(float), cudaMemcpyDeviceToHost));

    StepStats s;
    s.ex = h_mean[0];
    s.ey = h_mean[1];
    s.eth = h_mean[2];
    s.err = pose_error_xy(gt, s.ex, s.ey);
    s.n_part = n;
    s.occupied = occupied;
    s.p_inject = p_inject;
    s.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    f.n_active = n_new;
    return s;
}

// ----------------------------------------------------------------------
// CPU reference filter (identical algorithm, single threaded, for timing)
// ----------------------------------------------------------------------
struct CpuAmcl {
    std::vector<float> x, y, th, w, x2, y2, th2, wcum;
    std::vector<char> occ;
    std::mt19937 rng;
    int n_active = INIT_PART;
    float w_slow = 0.0f, w_fast = 0.0f;

    void init(unsigned long long seed, const Pose2& pose) {
        x.resize(MAX_PART); y.resize(MAX_PART); th.resize(MAX_PART);
        w.resize(MAX_PART); x2.resize(MAX_PART); y2.resize(MAX_PART);
        th2.resize(MAX_PART); wcum.resize(MAX_PART);
        occ.resize(KLD_NBINS);
        rng.seed(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int i = 0; i < INIT_PART; i++) {
            x[i] = clampf(pose.x + 0.25f * nd(rng), 0.4f, WORLD_W - 0.4f);
            y[i] = clampf(pose.y + 0.25f * nd(rng), 0.4f, WORLD_H - 0.4f);
            th[i] = wrap_angle(pose.th + 0.12f * nd(rng));
            w[i] = 1.0f / INIT_PART;
        }
    }
};

static StepStats run_cpu_step(CpuAmcl& f, const std::vector<float>& lx,
                              const std::vector<float>& ly, const ObsPack& obs,
                              const Pose2& gt, float v, float omega) {
    int n = f.n_active;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        float vt = v + MOTION_SIGMA_XY * nd(f.rng);
        float wt = omega + MOTION_SIGMA_TH * nd(f.rng);
        float theta = f.th[i];
        f.x[i] = clampf(f.x[i] + vt * std::cos(theta) * DT, 0.4f, WORLD_W - 0.4f);
        f.y[i] = clampf(f.y[i] + vt * std::sin(theta) * DT, 0.4f, WORLD_H - 0.4f);
        f.th[i] = wrap_angle(theta + wt * DT);
    }
    float sum_w = 0.0f;
    for (int i = 0; i < n; i++) {
        float logw = 0.0f;
        for (int k = 0; k < N_OBS; k++) {
            int id = obs.ids[k];
            float dx = lx[id] - f.x[i];
            float dy = ly[id] - f.y[i];
            float pr = std::sqrt(dx * dx + dy * dy);
            float pb = wrap_angle(std::atan2(dy, dx) - f.th[i]);
            float rr = (pr - obs.ranges[k]) / RANGE_SIGMA;
            float rb = wrap_angle(pb - obs.bearings[k]) / BEARING_SIGMA;
            logw += -0.5f * (rr * rr + rb * rb);
        }
        f.w[i] = std::exp(std::max(logw, -80.0f));
        sum_w += f.w[i];
    }
    float w_avg = sum_w / n;
    if (f.w_slow == 0.0f) { f.w_slow = w_avg; f.w_fast = w_avg; }
    f.w_slow += ALPHA_SLOW * (w_avg - f.w_slow);
    f.w_fast += ALPHA_FAST * (w_avg - f.w_fast);
    float p_inject = 0.0f;
    if (f.w_slow > 1.0e-30f) {
        float drop = 1.0f - f.w_fast / f.w_slow;
        if (drop > INJECT_DEADBAND) p_inject = std::min(drop, MAX_INJECT);
    }

    float ex = 0.0f, ey = 0.0f, ec = 0.0f, es = 0.0f;
    float inv = (sum_w > 1.0e-30f) ? 1.0f / sum_w : 0.0f;
    for (int i = 0; i < n; i++) {
        float wn = (sum_w > 1.0e-30f) ? f.w[i] * inv : 1.0f / n;
        f.w[i] = wn;
        ex += f.x[i] * wn; ey += f.y[i] * wn;
        ec += std::cos(f.th[i]) * wn; es += std::sin(f.th[i]) * wn;
    }

    std::fill(f.occ.begin(), f.occ.end(), 0);
    for (int i = 0; i < n; i++) f.occ[bin_index(f.x[i], f.y[i], f.th[i])] = 1;
    int occupied = 0;
    for (char c : f.occ) occupied += c;
    int n_new = kld_sample_size(occupied);

    float acc = 0.0f;
    for (int i = 0; i < n; i++) { acc += f.w[i]; f.wcum[i] = acc; }
    f.wcum[n - 1] = 1.0f;
    for (int i = 0; i < n_new; i++) {
        if (u01(f.rng) < p_inject) {
            f.x2[i] = 0.4f + u01(f.rng) * (WORLD_W - 0.8f);
            f.y2[i] = 0.4f + u01(f.rng) * (WORLD_H - 0.8f);
            f.th2[i] = u01(f.rng) * 2.0f * PI_F - PI_F;
        } else {
            float target = std::min(0.99999994f, u01(f.rng));
            int lo = 0, hi = n - 1;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (f.wcum[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            f.x2[i] = f.x[lo]; f.y2[i] = f.y[lo]; f.th2[i] = f.th[lo];
        }
    }
    f.x.swap(f.x2); f.y.swap(f.y2); f.th.swap(f.th2);
    for (int i = 0; i < n_new; i++) f.w[i] = 1.0f / n_new;
    auto t1 = std::chrono::high_resolution_clock::now();

    StepStats s;
    s.ex = ex; s.ey = ey; s.eth = std::atan2(es, ec);
    s.err = pose_error_xy(gt, ex, ey);
    s.n_part = n;
    s.occupied = occupied;
    s.p_inject = p_inject;
    s.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    f.n_active = n_new;
    return s;
}

// ----------------------------------------------------------------------
// Scenario (shared with gpu_global_localization_mcl)
// ----------------------------------------------------------------------
static void make_landmarks(std::vector<float>& lx, std::vector<float>& ly) {
    lx.clear(); ly.clear();
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

static ObsPack observe_pose(const Pose2& pose, const std::vector<float>& lx,
                            const std::vector<float>& ly, std::mt19937& rng) {
    std::vector<std::pair<float, int> > ranked;
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

// ----------------------------------------------------------------------
// Visualisation
// ----------------------------------------------------------------------
static cv::Point2i wp(float x, float y) {
    int px = static_cast<int>(x / WORLD_W * MAP_W);
    int py = MAP_H - 1 - static_cast<int>(y / WORLD_H * MAP_H);
    return cv::Point2i(px, py);
}

static void draw_count_plot(cv::Mat& img, const cv::Rect& r,
                            const std::vector<int>& counts, int upto) {
    cv::rectangle(img, r, cv::Scalar(29, 31, 36), -1);
    cv::rectangle(img, r, cv::Scalar(76, 80, 88), 1);
    cv::putText(img, "KLD particle count", cv::Point(r.x + 12, r.y + 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    for (int g = 0; g <= 4; g++) {
        int yy = r.y + r.height - 20 - g * (r.height - 48) / 4;
        cv::line(img, cv::Point(r.x + 44, yy), cv::Point(r.x + r.width - 12, yy),
                 cv::Scalar(45, 48, 55), 1);
    }
    // kidnap marker
    float kx01 = static_cast<float>(KIDNAP_STEP) / N_STEPS;
    int kxp = r.x + 44 + static_cast<int>(kx01 * (r.width - 58));
    cv::line(img, cv::Point(kxp, r.y + 30), cv::Point(kxp, r.y + r.height - 20),
             cv::Scalar(70, 90, 150), 1, cv::LINE_AA);
    int last = std::min(upto, static_cast<int>(counts.size()) - 1);
    if (last >= 1) {
        std::vector<cv::Point> pts;
        for (int i = 0; i <= last; i++) {
            float x01 = static_cast<float>(i) / N_STEPS;
            float y01 = clampf(static_cast<float>(counts[i]) / MAX_PART, 0.0f, 1.0f);
            int xx = r.x + 44 + static_cast<int>(x01 * (r.width - 58));
            int yy = r.y + r.height - 20 - static_cast<int>(y01 * (r.height - 48));
            pts.emplace_back(xx, yy);
        }
        cv::polylines(img, pts, false, cv::Scalar(250, 190, 70), 2, cv::LINE_AA);
    }
    cv::putText(img, cv::format("%dk", MAX_PART / 1000), cv::Point(r.x + 6, r.y + 44),
                cv::FONT_HERSHEY_SIMPLEX, 0.32, cv::Scalar(165, 170, 180), 1, cv::LINE_AA);
    cv::putText(img, "0", cv::Point(r.x + 28, r.y + r.height - 16),
                cv::FONT_HERSHEY_SIMPLEX, 0.32, cv::Scalar(165, 170, 180), 1, cv::LINE_AA);
}

static cv::Mat draw_frame(int step, const std::vector<float>& lx,
                          const std::vector<float>& ly,
                          const std::vector<Pose2>& trail,
                          const std::vector<cv::Point2f>& est_trail,
                          const GpuAmcl& f, const Pose2& gt, const StepStats& s,
                          const std::vector<int>& counts,
                          double gpu_ms, double cpu_ms, double speedup,
                          float gpu_rmse, float cpu_rmse, int peak_n, int reacq) {
    cv::Mat img(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(18, 20, 23));
    cv::Rect map(0, 0, MAP_W, MAP_H);
    cv::rectangle(img, map, cv::Scalar(24, 27, 31), -1);
    cv::rectangle(img, map, cv::Scalar(73, 80, 88), 1);
    for (size_t i = 0; i < lx.size(); i++)
        cv::circle(img, wp(lx[i], ly[i]), 2, cv::Scalar(120, 175, 210), -1, cv::LINE_AA);

    int stride = std::max(1, s.n_part / 1500);
    for (int i = 0; i < s.n_part; i += stride) {
        cv::Point2i q = wp(f.hx[i], f.hy[i]);
        if (q.x >= 0 && q.x < MAP_W && q.y >= 0 && q.y < MAP_H)
            img.at<cv::Vec3b>(q.y, q.x) = cv::Vec3b(60, 150, 245);
    }
    for (size_t i = 1; i < trail.size(); i++)
        cv::line(img, wp(trail[i - 1].x, trail[i - 1].y), wp(trail[i].x, trail[i].y),
                 cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
    for (size_t i = 1; i < est_trail.size(); i++)
        cv::line(img, wp(est_trail[i - 1].x, est_trail[i - 1].y),
                 wp(est_trail[i].x, est_trail[i].y), cv::Scalar(90, 225, 150), 1, cv::LINE_AA);

    cv::Point2i g = wp(gt.x, gt.y);
    cv::circle(img, g, 6, cv::Scalar(245, 245, 245), -1, cv::LINE_AA);
    cv::line(img, g, wp(gt.x + 0.55f * std::cos(gt.th), gt.y + 0.55f * std::sin(gt.th)),
             cv::Scalar(245, 245, 245), 2, cv::LINE_AA);
    cv::circle(img, wp(s.ex, s.ey), 8, cv::Scalar(90, 225, 150), 2, cv::LINE_AA);

    cv::putText(img, "GPU augmented KLD-AMCL", cv::Point(16, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, cv::format("err %.2f m   particles %d   bins %d",
                                s.err, s.n_part, s.occupied),
                cv::Point(16, 52), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(220, 225, 230), 1, cv::LINE_AA);
    if (s.p_inject > 0.02f)
        cv::putText(img, cv::format("injecting %.0f%% global poses", 100.0f * s.p_inject),
                    cv::Point(16, 76), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(70, 210, 255), 1, cv::LINE_AA);

    int ix = MAP_W;
    cv::rectangle(img, cv::Rect(ix, 0, INFO_W, FRAME_H), cv::Scalar(28, 31, 36), -1);
    cv::rectangle(img, cv::Rect(ix, 0, INFO_W, FRAME_H), cv::Scalar(73, 80, 88), 1);
    cv::putText(img, "GPU KLD-sampling AMCL", cv::Point(ix + 16, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.54, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, cv::format("step %03d / %d", step, N_STEPS), cv::Point(ix + 16, 58),
                cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(210, 216, 224), 1, cv::LINE_AA);
    if (step >= KIDNAP_STEP)
        cv::putText(img, "hidden kidnap active", cv::Point(ix + 16, 86),
                    cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(75, 120, 245), 1, cv::LINE_AA);
    else
        cv::putText(img, cv::format("kidnap at step %d", KIDNAP_STEP), cv::Point(ix + 16, 86),
                    cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(178, 185, 194), 1, cv::LINE_AA);

    draw_count_plot(img, cv::Rect(ix + 14, 104, INFO_W - 30, 150), counts, step);

    int yb = 286;
    cv::putText(img, cv::format("GPU %.3f ms/step", gpu_ms), cv::Point(ix + 16, yb),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(90, 225, 135), 1, cv::LINE_AA);
    cv::putText(img, cv::format("CPU %.3f ms/step", cpu_ms), cv::Point(ix + 16, yb + 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(165, 175, 190), 1, cv::LINE_AA);
    cv::putText(img, cv::format("speedup %.1fx", speedup), cv::Point(ix + 16, yb + 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(250, 190, 70), 1, cv::LINE_AA);
    cv::putText(img, cv::format("peak particles %d", peak_n), cv::Point(ix + 16, yb + 94),
                cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(220, 225, 232), 1, cv::LINE_AA);
    std::string reacq_msg = reacq >= 0 ? cv::format("reacquired in %d steps", reacq)
                                       : "recovering...";
    cv::putText(img, reacq_msg, cv::Point(ix + 16, yb + 122), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(90, 225, 150), 1, cv::LINE_AA);
    if (gpu_rmse > 0.0f) {
        cv::putText(img, cv::format("GPU settled RMSE %.3f m", gpu_rmse),
                    cv::Point(ix + 16, yb + 146), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                    cv::Scalar(120, 225, 160), 1, cv::LINE_AA);
        cv::putText(img, cv::format("CPU settled RMSE %.3f m", cpu_rmse),
                    cv::Point(ix + 16, yb + 170), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                    cv::Scalar(200, 205, 212), 1, cv::LINE_AA);
    } else {
        cv::putText(img, "settled RMSE: pending", cv::Point(ix + 16, yb + 146),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(170, 175, 182), 1, cv::LINE_AA);
    }
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
    GpuAmcl gpu;
    gpu.alloc(25052026ULL, gt);
    CpuAmcl cpu;
    cpu.init(25052028ULL, gt);

    std::mt19937 rng_obs(77);
    std::vector<Pose2> gt_trail;
    std::vector<cv::Point2f> est_trail;
    std::vector<int> counts;

    cv::VideoWriter video("gif/gpu_kld_amcl.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(FRAME_W, FRAME_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "Failed to open gif/gpu_kld_amcl.avi\n");
        return 1;
    }

    constexpr int SETTLE_STEPS = 30;
    double gpu_ms_sum = 0.0, cpu_ms_sum = 0.0;
    double gpu_err_post = 0.0, cpu_err_post = 0.0;
    int post_count = 0, peak_n = 0, reacq = -1;
    double gpu_ms_avg = 0.0, cpu_ms_avg = 0.0, speedup = 0.0;
    float gpu_rmse = 0.0f, cpu_rmse = 0.0f;

    for (int step = 0; step < N_STEPS; step++) {
        float v, omega;
        controls_at(step, v, omega);
        if (step == KIDNAP_STEP) gt = Pose2{28.6f, 19.0f, -2.35f};
        else advance_pose(gt, v, omega);

        ObsPack obs = observe_pose(gt, lx, ly, rng_obs);
        StepStats sg = run_gpu_step(gpu, d_lx, d_ly, d_obs_ids, d_obs_ranges,
                                    d_obs_bearings, obs, gt, v, omega);
        StepStats sc = run_cpu_step(cpu, lx, ly, obs, gt, v, omega);

        gpu_ms_sum += sg.ms;
        cpu_ms_sum += sc.ms;
        gpu_ms_avg = gpu_ms_sum / (step + 1);
        cpu_ms_avg = cpu_ms_sum / (step + 1);
        speedup = cpu_ms_avg / std::max(1.0e-9, gpu_ms_avg);
        peak_n = std::max(peak_n, std::max(sg.n_part, gpu.n_active));
        if (step >= KIDNAP_STEP && reacq < 0 && sg.err < 0.6f) reacq = step - KIDNAP_STEP;
        if (step >= N_STEPS - SETTLE_STEPS) {
            gpu_err_post += sg.err * sg.err;
            cpu_err_post += sc.err * sc.err;
            post_count++;
            gpu_rmse = std::sqrt(gpu_err_post / post_count);
            cpu_rmse = std::sqrt(cpu_err_post / post_count);
        }

        gt_trail.push_back(gt);
        est_trail.push_back(cv::Point2f(sg.ex, sg.ey));
        counts.push_back(sg.n_part);

        if (step % 3 == 0 || step == N_STEPS - 1 || step == KIDNAP_STEP) {
            gpu.copy_to_host();
            video.write(draw_frame(step, lx, ly, gt_trail, est_trail, gpu, gt, sg,
                                   counts, gpu_ms_avg, cpu_ms_avg, speedup,
                                   gpu_rmse, cpu_rmse, peak_n, reacq));
        }
        if (step < 6 || step == KIDNAP_STEP || step % 20 == 19) {
            std::printf("step %03d  err %.3f m  particles %d  bins %d  inject %.2f\n",
                        step, sg.err, sg.n_part, sg.occupied, sg.p_inject);
        }
    }
    video.release();
    avi_to_gif("gif/gpu_kld_amcl.avi", "gif/gpu_kld_amcl.gif", 10, 720);

    std::printf("GPU augmented KLD-sampling AMCL\n");
    std::printf("particle range: [%d, %d], landmarks: %d, kidnap step: %d\n",
                MIN_PART, MAX_PART, N_LANDMARKS, KIDNAP_STEP);
    std::printf("peak particles: %d\n", peak_n);
    std::printf("reacquired after kidnap in %d steps\n", reacq);
    std::printf("settled RMSE (last %d steps): GPU %.4f m, CPU %.4f m\n",
                SETTLE_STEPS, gpu_rmse, cpu_rmse);
    std::printf("avg step time: GPU %.4f ms, CPU %.4f ms\n", gpu_ms_avg, cpu_ms_avg);
    std::printf("speedup: %.1fx\n", speedup);
    std::printf("Wrote gif/gpu_kld_amcl.gif\n");

    gpu.free_all();
    CUDA_CHECK(cudaFree(d_lx));
    CUDA_CHECK(cudaFree(d_ly));
    CUDA_CHECK(cudaFree(d_obs_ids));
    CUDA_CHECK(cudaFree(d_obs_ranges));
    CUDA_CHECK(cudaFree(d_obs_bearings));
    return 0;
}
