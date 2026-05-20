/*************************************************************************
    Differentiable Particle Filter with a learnable MLP observation model.

    Extends the soft-resampling + reparameterized-motion DPF added in
    src/diff_pf.cu: the per-landmark likelihood
        log p(z_l | x) = -(||x - l|| - z_l)^2 / (2 sigma^2)
    is replaced with a small MLP h_theta(d, z) -> log_lik, where
        d = ||particle - landmark||,    z = measured distance
    Internally the two inputs are scaled as (d / range, (z - d) / 5 sigma)
    so the tiny tanh network does not spend capacity fighting raw units.
    The MLP is trained three ways:
      1. supervised against the analytic Gaussian on a synthetic dataset;
      2. fine-tuned end-to-end on tracking loss by finite-differencing the
         tiny MLP weight vector through a soft-resampling DPF rollout;
      3. trained with direct GPU backprop on calibration-learned observation
         surrogates for misspecified range sensors.

    The clean scene keeps Gaussian range noise, where the handcrafted
    likelihood is correctly specified. Hard scenes inject range outliers,
    distance-dependent range bias, occlusion short-returns, landmark dropouts,
    and a hidden pose jump, making the Gaussian likelihood intentionally
    misspecified. Learned observation models can be adapted either from
    localization performance or from a calibration trace that exposes the
    sensor residual distribution.

    Output:
      - Training curve (MSE on log-likelihood, supervised pre-training)
      - Tracking-loss finite-difference curve for the end-to-end MLP
      - Hard-scene tracking gif: handcrafted DPF vs supervised MLP-DPF vs
        tracking-loss MLP-DPF vs calibrated-surrogate MLP-DPF, same alpha
        (=3.14, learned in src/diff_pf.cu)
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "gpu_mlp.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); std::exit(EXIT_FAILURE); \
    } \
} while (0)

using cudabot::GpuMLP;
using cudabot::mlp_forward;

constexpr int   N_PARTICLES   = 1024;
constexpr int   N_LANDMARKS   = 8;
constexpr int   N_FRAMES      = 240;
constexpr float DT            = 0.1f;
constexpr float WORLD_W       = 40.0f;
constexpr float WORLD_H       = 30.0f;
constexpr float OBS_RANGE     = 22.0f;
constexpr float OBS_SIGMA     = 1.0f;
constexpr float SOFT_BETA     = 0.7f;
constexpr float TRAINED_ALPHA = 3.14f;
constexpr float HARD_OUTLIER_PROB = 0.18f;
constexpr float HARD_OUTLIER_MAG  = 9.0f;
constexpr float BIASED_RANGE_D0    = 10.0f;
constexpr float BIASED_RANGE_GAIN  = 0.35f;
constexpr float OCCLUSION_START_T = 1.0f;
constexpr float OCCLUSION_END_T   = 16.0f;
constexpr float OCCLUSION_DROP_PROB = 0.30f;
constexpr float OCCLUSION_SHORT_PROB = 0.25f;
constexpr float OCCLUSION_SHORT_MAG  = 8.0f;
constexpr float KIDNAP_T  = 3.0f;
constexpr float KIDNAP_DX = -4.0f;
constexpr float KIDNAP_DY = 3.0f;

constexpr int   MLP_INPUT     = 2;   // (d, z)
constexpr int   MLP_HIDDEN    = 16;
constexpr int   MLP_LAYERS    = 1;   // single hidden layer
constexpr int   MLP_OUTPUT    = 1;   // log-likelihood
constexpr int   MLP_ACTIV     = 1;   // tanh hidden activation

constexpr int   E2E_TRAIN_PARTICLES = 384;
constexpr int   E2E_TRAIN_FRAMES    = 48;
constexpr int   E2E_TRAIN_EPOCHS    = 24;
constexpr int   E2E_GRAD_SEEDS      = 2;
constexpr int   E2E_VAL_FRAMES      = 96;
constexpr int   E2E_VAL_SEEDS       = 2;
constexpr float E2E_FD_EPS          = 0.04f;
constexpr float E2E_LR              = 0.0035f;
constexpr float E2E_GRAD_CLIP       = 4.0f;
constexpr float E2E_WEIGHT_CLIP     = 8.0f;
constexpr int   DIRECT_TRAIN_SAMPLES = 32768;
constexpr int   DIRECT_TRAIN_EPOCHS  = 1600;
constexpr float DIRECT_TRAIN_LR      = 0.01f;
constexpr int   CALIBRATION_SAMPLES  = 8192;
constexpr int   CALIBRATION_RESIDUAL_BINS = 96;
constexpr float CALIBRATION_RESIDUAL_MAX  = HARD_OUTLIER_MAG + 4.0f * OBS_SIGMA;
constexpr float CALIBRATION_HIST_SMOOTH   = 2.0f;

constexpr int   PANEL_W       = 480;
constexpr int   PANEL_H       = 360;
constexpr float VIS_SX        = static_cast<float>(PANEL_W) / WORLD_W;
constexpr float VIS_SY        = static_cast<float>(PANEL_H) / WORLD_H;

struct Pose2 { float x, y, th; };

enum ObservationMode {
    OBS_CLEAN_GAUSSIAN,
    OBS_RANGE_OUTLIERS,
    OBS_BIASED_RANGE,
    OBS_OCCLUSION_KIDNAP
};

__host__ __device__ inline float wrap_pi(float a) {
    while (a >  static_cast<float>(M_PI)) a -= 2.0f * static_cast<float>(M_PI);
    while (a < -static_cast<float>(M_PI)) a += 2.0f * static_cast<float>(M_PI);
    return a;
}

// ---------------------------------------------------------------------------
// curand init / motion noise refresh
// ---------------------------------------------------------------------------
__global__ void init_curand(curandState* states, unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &states[i]);
}
__global__ void refresh_motion_noise(curandState* states, int n,
                                     float* d_eps_x, float* d_eps_y, float* d_eps_th) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState s = states[i];
    d_eps_x[i]  = curand_normal(&s);
    d_eps_y[i]  = curand_normal(&s);
    d_eps_th[i] = curand_normal(&s);
    states[i] = s;
}
__global__ void predict_kernel(float* px, float* py, float* pth,
                               const float* eps_x, const float* eps_y, const float* eps_th,
                               float alpha, float v, float omega,
                               float sigma_xy, float sigma_th, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float th = pth[i];
    float dx = (v * cosf(th) + alpha * sigma_xy * eps_x[i]) * dt;
    float dy = (v * sinf(th) + alpha * sigma_xy * eps_y[i]) * dt;
    float dth = (omega + alpha * sigma_th * eps_th[i]) * dt;
    px[i]  = px[i]  + dx;
    py[i]  = py[i]  + dy;
    pth[i] = wrap_pi(th + dth);
}

// ---------------------------------------------------------------------------
// Analytic likelihood kernel (handcrafted Gaussian)
// ---------------------------------------------------------------------------
__global__ void likelihood_kernel_analytic(const float* px, const float* py,
                                           const float* lx, const float* ly, int nL,
                                           const float* z_dist,
                                           const unsigned char* z_valid,
                                           float sigma, float* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float log_lik = 0.0f;
    float two_sig2 = 2.0f * sigma * sigma;
    for (int l = 0; l < nL; l++) {
        if (!z_valid[l]) continue;
        float dx = px[i] - lx[l];
        float dy = py[i] - ly[l];
        float d = sqrtf(dx * dx + dy * dy);
        float r = d - z_dist[l];
        log_lik += -(r * r) / two_sig2;
    }
    w[i] = expf(log_lik);
}

// ---------------------------------------------------------------------------
// MLP likelihood kernel (each particle, each visible landmark, MLP(d,z))
// ---------------------------------------------------------------------------
__global__ void likelihood_kernel_mlp(const float* px, const float* py,
                                      const float* lx, const float* ly, int nL,
                                      const float* z_dist,
                                      const unsigned char* z_valid,
                                      const float* __restrict__ mlp_w,
                                      int input_dim, int hidden_dim,
                                      int n_layers, int output_dim,
                                      int activation,
                                      float* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float scratch[64];  // hidden_dim * 2 — fits MLP_HIDDEN <= 32
    float mlp_in[2];
    float mlp_out[1];
    float log_lik = 0.0f;
    for (int l = 0; l < nL; l++) {
        if (!z_valid[l]) continue;
        float dx = px[i] - lx[l];
        float dy = py[i] - ly[l];
        float d = sqrtf(dx * dx + dy * dy);
        mlp_in[0] = d / OBS_RANGE;
        mlp_in[1] = (z_dist[l] - d) / (5.0f * OBS_SIGMA);
        mlp_forward(mlp_w, mlp_in, input_dim,
                    mlp_out, output_dim,
                    hidden_dim, n_layers, scratch, activation);
        log_lik += mlp_out[0];
    }
    w[i] = expf(log_lik);
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------
static void normalise(std::vector<float>& w) {
    double s = 0.0;
    for (float v : w) s += v;
    if (s < 1.0e-30) s = 1.0e-30;
    float inv = static_cast<float>(1.0 / s);
    for (float& v : w) v *= inv;
}

static void soft_resample(const std::vector<float>& w_norm,
                          std::vector<int>& indices,
                          float beta, std::mt19937& rng) {
    int N = static_cast<int>(w_norm.size());
    std::vector<float> w_mix(N);
    float u = (1.0f - beta) / N;
    float sum = 0.0f;
    for (int i = 0; i < N; i++) { w_mix[i] = beta * w_norm[i] + u; sum += w_mix[i]; }
    float inv = 1.0f / sum;
    for (float& v : w_mix) v *= inv;
    std::vector<float> cumsum(N);
    cumsum[0] = w_mix[0];
    for (int i = 1; i < N; i++) cumsum[i] = cumsum[i - 1] + w_mix[i];
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    indices.resize(N);
    for (int i = 0; i < N; i++) {
        float u2 = uni(rng);
        int lo = 0, hi = N - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (cumsum[mid] < u2) lo = mid + 1; else hi = mid;
        }
        indices[i] = lo;
    }
}

struct ParticleSet {
    float *d_px, *d_py, *d_pth, *d_w;
    float *d_eps_x, *d_eps_y, *d_eps_th;
    curandState* d_states;
    std::vector<float> h_px, h_py, h_pth, h_w;

    void alloc(int n, unsigned long long seed) {
        size_t bf = n * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_px,  bf)); CUDA_CHECK(cudaMalloc(&d_py, bf));
        CUDA_CHECK(cudaMalloc(&d_pth, bf)); CUDA_CHECK(cudaMalloc(&d_w,  bf));
        CUDA_CHECK(cudaMalloc(&d_eps_x, bf)); CUDA_CHECK(cudaMalloc(&d_eps_y, bf));
        CUDA_CHECK(cudaMalloc(&d_eps_th, bf));
        CUDA_CHECK(cudaMalloc(&d_states, n * sizeof(curandState)));
        int blk = 256, gd = (n + blk - 1) / blk;
        init_curand<<<gd, blk>>>(d_states, seed, n);
        h_px.resize(n); h_py.resize(n); h_pth.resize(n); h_w.resize(n);
    }
    void free_all() {
        cudaFree(d_px); cudaFree(d_py); cudaFree(d_pth); cudaFree(d_w);
        cudaFree(d_eps_x); cudaFree(d_eps_y); cudaFree(d_eps_th); cudaFree(d_states);
    }
};

static void upload(ParticleSet& P, int n,
                   const std::vector<float>& px,
                   const std::vector<float>& py,
                   const std::vector<float>& pth) {
    CUDA_CHECK(cudaMemcpy(P.d_px,  px.data(),  n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(P.d_py,  py.data(),  n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(P.d_pth, pth.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

static Pose2 gt_at(float t) {
    Pose2 p;
    p.x = 0.5f * WORLD_W + 8.0f * std::cos(0.5f * t);
    p.y = 0.5f * WORLD_H + 5.5f * std::sin(0.8f * t);
    p.th = wrap_pi(0.5f * t);
    return p;
}
static Pose2 scene_gt_at(float t, ObservationMode mode) {
    Pose2 p = gt_at(t);
    if (mode == OBS_OCCLUSION_KIDNAP && t >= KIDNAP_T) {
        p.x = std::max(1.0f, std::min(WORLD_W - 1.0f, p.x + KIDNAP_DX));
        p.y = std::max(1.0f, std::min(WORLD_H - 1.0f, p.y + KIDNAP_DY));
    }
    return p;
}
static float biased_range_mean(float d) {
    return d + BIASED_RANGE_GAIN * std::max(0.0f, d - BIASED_RANGE_D0);
}
static float sample_biased_range_sensor(float d, std::mt19937& rng) {
    std::normal_distribution<float> noise(0.0f, OBS_SIGMA);
    return std::max(0.0f, biased_range_mean(d) + noise(rng));
}
static float sample_range_outlier_sensor(float d, std::mt19937& rng) {
    std::normal_distribution<float> noise(0.0f, OBS_SIGMA);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::uniform_real_distribution<float> outlier(-HARD_OUTLIER_MAG, HARD_OUTLIER_MAG);
    float z = d + noise(rng);
    if (uni(rng) < HARD_OUTLIER_PROB) z += outlier(rng);
    return std::max(0.0f, z);
}
static bool sample_occlusion_sensor(float d, std::mt19937& rng, float t, float& z) {
    std::normal_distribution<float> noise(0.0f, OBS_SIGMA);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::uniform_real_distribution<float> short_hit(0.0f, OCCLUSION_SHORT_MAG);
    bool occlusion_window = (t >= OCCLUSION_START_T && t <= OCCLUSION_END_T);
    if (occlusion_window && uni(rng) < OCCLUSION_DROP_PROB) {
        return false;
    }
    bool short_return = occlusion_window && uni(rng) < OCCLUSION_SHORT_PROB;
    float zn = d + noise(rng);
    if (short_return) {
        zn -= short_hit(rng);
    }
    z = std::max(0.0f, zn);
    return true;
}
static void controls_at(float t, float& v, float& omega) {
    Pose2 p_now = gt_at(t);
    Pose2 p_next = gt_at(t + DT);
    float dx = p_next.x - p_now.x;
    float dy = p_next.y - p_now.y;
    v = std::sqrt(dx * dx + dy * dy) / DT;
    omega = wrap_pi(p_next.th - p_now.th) / DT;
}
static void observe(const Pose2& gt, const std::vector<float>& lx,
                    const std::vector<float>& ly, std::mt19937& rng,
                    std::vector<float>& z, std::vector<unsigned char>& valid,
                    ObservationMode mode = OBS_CLEAN_GAUSSIAN,
                    float t = 0.0f) {
    int L = static_cast<int>(lx.size());
    z.resize(L); valid.assign(L, 0u);
    std::normal_distribution<float> noise(0.0f, OBS_SIGMA);
    for (int l = 0; l < L; l++) {
        float dx = gt.x - lx[l];
        float dy = gt.y - ly[l];
        float d = std::sqrt(dx * dx + dy * dy);
        if (d <= OBS_RANGE) {
            if (mode == OBS_RANGE_OUTLIERS) {
                z[l] = sample_range_outlier_sensor(d, rng);
            } else if (mode == OBS_BIASED_RANGE) {
                z[l] = sample_biased_range_sensor(d, rng);
            } else if (mode == OBS_OCCLUSION_KIDNAP) {
                if (!sample_occlusion_sensor(d, rng, t, z[l])) continue;
            } else {
                z[l] = std::max(0.0f, d + noise(rng));
            }
            valid[l] = 1u;
        }
    }
}

enum LikelihoodMode { LIK_ANALYTIC, LIK_MLP };

struct PFStep { float ex, ey; };
static PFStep run_step(ParticleSet& P, int n, float alpha, float v, float omega,
                       float sigma_xy, float sigma_th,
                       const float* d_lx, const float* d_ly,
                       const float* d_z, const unsigned char* d_zv,
                       LikelihoodMode mode, const GpuMLP* mlp,
                       std::mt19937& rng) {
    int blk = 256, gd = (n + blk - 1) / blk;
    refresh_motion_noise<<<gd, blk>>>(P.d_states, n, P.d_eps_x, P.d_eps_y, P.d_eps_th);
    predict_kernel<<<gd, blk>>>(P.d_px, P.d_py, P.d_pth,
                                P.d_eps_x, P.d_eps_y, P.d_eps_th,
                                alpha, v, omega, sigma_xy, sigma_th, DT, n);
    if (mode == LIK_ANALYTIC) {
        likelihood_kernel_analytic<<<gd, blk>>>(P.d_px, P.d_py, d_lx, d_ly, N_LANDMARKS,
                                                d_z, d_zv, OBS_SIGMA, P.d_w, n);
    } else {
        auto cfg = mlp->config();
        likelihood_kernel_mlp<<<gd, blk>>>(P.d_px, P.d_py, d_lx, d_ly, N_LANDMARKS,
                                           d_z, d_zv, mlp->device_weights(),
                                           cfg.input_dim, cfg.hidden_dim,
                                           cfg.n_layers, cfg.output_dim,
                                           MLP_ACTIV, P.d_w, n);
    }
    CUDA_CHECK(cudaMemcpy(P.h_px.data(),  P.d_px,  n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P.h_py.data(),  P.d_py,  n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P.h_pth.data(), P.d_pth, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P.h_w.data(),   P.d_w,   n * sizeof(float), cudaMemcpyDeviceToHost));
    normalise(P.h_w);
    float ex = 0.0f, ey = 0.0f;
    for (int i = 0; i < n; i++) { ex += P.h_w[i] * P.h_px[i]; ey += P.h_w[i] * P.h_py[i]; }
    std::vector<int> idx;
    soft_resample(P.h_w, idx, SOFT_BETA, rng);
    std::vector<float> npx(n), npy(n), npth(n);
    for (int i = 0; i < n; i++) {
        npx[i]  = P.h_px[idx[i]];
        npy[i]  = P.h_py[idx[i]];
        npth[i] = P.h_pth[idx[i]];
    }
    upload(P, n, npx, npy, npth);
    return {ex, ey};
}

// ---------------------------------------------------------------------------
// End-to-end MLP training through tracking loss
// ---------------------------------------------------------------------------
static void reset_particle_set(ParticleSet& P, int n,
                               unsigned int init_seed,
                               unsigned long long motion_seed,
                               ObservationMode obs_mode) {
    std::vector<float> ipx(n), ipy(n), ipth(n);
    std::mt19937 rng_init(init_seed);
    std::normal_distribution<float> nxy(0.0f, 1.5f);
    Pose2 gt0 = scene_gt_at(0.0f, obs_mode);
    for (int i = 0; i < n; i++) {
        ipx[i]  = gt0.x + nxy(rng_init);
        ipy[i]  = gt0.y + nxy(rng_init);
        ipth[i] = gt0.th;
    }
    upload(P, n, ipx, ipy, ipth);

    int blk = 256, gd = (n + blk - 1) / blk;
    init_curand<<<gd, blk>>>(P.d_states, motion_seed, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static float rollout_tracking_loss_mlp(ParticleSet& P, int n,
                                       const GpuMLP& mlp,
                                       const std::vector<float>& lx,
                                       const std::vector<float>& ly,
                                       const float* d_lx,
                                       const float* d_ly,
                                       float* d_z,
                                       unsigned char* d_zv,
                                       unsigned int seed,
                                       int n_frames,
                                       ObservationMode obs_mode) {
    reset_particle_set(P, n, seed + 11u,
                       static_cast<unsigned long long>(seed) * 1009ULL + 17ULL,
                       obs_mode);

    std::mt19937 rng_obs(seed + 23u);
    std::mt19937 rng_resample(seed + 37u);
    double loss_sum = 0.0;
    for (int s = 0; s < n_frames; s++) {
        float t = s * DT;
        Pose2 gt = scene_gt_at(t, obs_mode);
        float v, omega;
        controls_at(t, v, omega);

        std::vector<float> z;
        std::vector<unsigned char> valid;
        observe(gt, lx, ly, rng_obs, z, valid, obs_mode, t);
        CUDA_CHECK(cudaMemcpy(d_z, z.data(), N_LANDMARKS * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_zv, valid.data(), N_LANDMARKS,
                              cudaMemcpyHostToDevice));

        PFStep out = run_step(P, n, TRAINED_ALPHA, v, omega,
                              0.6f, 0.1f, d_lx, d_ly, d_z, d_zv,
                              LIK_MLP, &mlp, rng_resample);
        float dx = out.ex - gt.x;
        float dy = out.ey - gt.y;
        loss_sum += dx * dx + dy * dy;
    }
    return static_cast<float>(loss_sum / n_frames);
}

struct AdamVec {
    std::vector<float> m, v;
    int t = 0;

    explicit AdamVec(int n) : m(n, 0.0f), v(n, 0.0f) {}
};

static float apply_adam(std::vector<float>& weights,
                        std::vector<float>& grad,
                        AdamVec& adam,
                        float lr) {
    double norm2 = 0.0;
    for (float g : grad) norm2 += static_cast<double>(g) * g;
    float grad_norm = std::sqrt(static_cast<float>(norm2));
    if (grad_norm > E2E_GRAD_CLIP) {
        float s = E2E_GRAD_CLIP / (grad_norm + 1.0e-12f);
        for (float& g : grad) g *= s;
        grad_norm = E2E_GRAD_CLIP;
    }

    adam.t++;
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float eps = 1.0e-8f;
    float b1_corr = 1.0f - std::pow(beta1, adam.t);
    float b2_corr = 1.0f - std::pow(beta2, adam.t);
    for (size_t i = 0; i < weights.size(); i++) {
        adam.m[i] = beta1 * adam.m[i] + (1.0f - beta1) * grad[i];
        adam.v[i] = beta2 * adam.v[i] + (1.0f - beta2) * grad[i] * grad[i];
        float mh = adam.m[i] / b1_corr;
        float vh = adam.v[i] / b2_corr;
        weights[i] -= lr * mh / (std::sqrt(vh) + eps);
        weights[i] = std::max(-E2E_WEIGHT_CLIP, std::min(E2E_WEIGHT_CLIP, weights[i]));
    }
    return grad_norm;
}

static void train_mlp_tracking_finite_difference(
    GpuMLP& mlp,
    const std::vector<float>& lx,
    const std::vector<float>& ly,
    const float* d_lx,
    const float* d_ly,
    float* d_z,
    unsigned char* d_zv,
    std::vector<float>& loss_curve,
    ObservationMode obs_mode) {
    auto cfg = mlp.config();
    std::vector<float> weights = mlp.get_weights();
    std::vector<float> grad(cfg.total_weights, 0.0f);
    std::vector<float> tmp = weights;
    AdamVec adam(cfg.total_weights);
    ParticleSet P;
    P.alloc(E2E_TRAIN_PARTICLES, 777ULL);
    loss_curve.clear();

    auto eval_loss = [&](const std::vector<float>& w,
                         unsigned int seed_base,
                         int n_frames) -> float {
        mlp.load_weights(w);
        return rollout_tracking_loss_mlp(P, E2E_TRAIN_PARTICLES, mlp,
                                         lx, ly, d_lx, d_ly, d_z, d_zv,
                                         seed_base, n_frames,
                                         obs_mode);
    };
    auto eval_loss_avg = [&](const std::vector<float>& w,
                             unsigned int seed_base,
                             int n_frames,
                             int n_seeds) -> float {
        double loss = 0.0;
        for (int k = 0; k < n_seeds; k++) {
            loss += eval_loss(w, seed_base + static_cast<unsigned int>(k) * 10007u,
                              n_frames);
        }
        return static_cast<float>(loss / n_seeds);
    };

    std::vector<float> best_weights = weights;
    float best_val = eval_loss_avg(weights, 24001u, E2E_VAL_FRAMES, E2E_VAL_SEEDS);
    std::printf("E2E MLP initial validation loss %.4f (%d seeds x %d frames)\n",
                best_val, E2E_VAL_SEEDS, E2E_VAL_FRAMES);

    for (int epoch = 0; epoch < E2E_TRAIN_EPOCHS; epoch++) {
        unsigned int seed_base = 9001u + static_cast<unsigned int>(epoch) * 131u;
        float base_loss = eval_loss_avg(weights, seed_base + 7u,
                                        E2E_TRAIN_FRAMES, E2E_GRAD_SEEDS);
        loss_curve.push_back(base_loss);

        for (int wid = 0; wid < cfg.total_weights; wid++) {
            double gsum = 0.0;
            for (int k = 0; k < E2E_GRAD_SEEDS; k++) {
                unsigned int fd_seed = seed_base + 31u + static_cast<unsigned int>(k) * 10007u;
                tmp = weights;
                tmp[wid] += E2E_FD_EPS;
                float lp = eval_loss(tmp, fd_seed, E2E_TRAIN_FRAMES);

                tmp[wid] = weights[wid] - E2E_FD_EPS;
                float lm = eval_loss(tmp, fd_seed, E2E_TRAIN_FRAMES);

                gsum += (lp - lm) / (2.0f * E2E_FD_EPS);
            }
            grad[wid] = static_cast<float>(gsum / E2E_GRAD_SEEDS);
        }

        float gnorm = apply_adam(weights, grad, adam, E2E_LR);
        mlp.load_weights(weights);
        float next_loss = eval_loss_avg(weights, seed_base + 7u,
                                        E2E_TRAIN_FRAMES, E2E_GRAD_SEEDS);
        float val_loss = eval_loss_avg(weights, 24001u, E2E_VAL_FRAMES, E2E_VAL_SEEDS);
        if (val_loss < best_val) {
            best_val = val_loss;
            best_weights = weights;
        }
        std::printf("E2E MLP epoch %02d/%02d: tracking loss %.4f -> %.4f, "
                    "val %.4f, |g|=%.3f\n",
                    epoch + 1, E2E_TRAIN_EPOCHS, base_loss, next_loss,
                    val_loss, gnorm);
    }

    mlp.load_weights(best_weights);
    std::printf("E2E MLP restored best validation loss %.4f\n", best_val);
    P.free_all();
}

struct ResidualCalibrationModel {
    std::vector<float> log_lik_by_bin;
    float residual_min = -CALIBRATION_RESIDUAL_MAX;
    float residual_max =  CALIBRATION_RESIDUAL_MAX;
    float tail_log_lik = -12.0f;
    float residual_rmse = OBS_SIGMA;
    int attempts = 0;
    int valid_samples = 0;
};

static int residual_bin(float r, const ResidualCalibrationModel& calib) {
    float u = (r - calib.residual_min) / (calib.residual_max - calib.residual_min);
    int b = static_cast<int>(u * CALIBRATION_RESIDUAL_BINS);
    return std::max(0, std::min(CALIBRATION_RESIDUAL_BINS - 1, b));
}

static float calibrated_log_lik_at(float r, const ResidualCalibrationModel& calib) {
    float bin_width = (calib.residual_max - calib.residual_min) / CALIBRATION_RESIDUAL_BINS;
    float x = (r - calib.residual_min) / bin_width - 0.5f;
    int lo = static_cast<int>(std::floor(x));
    float a = x - lo;
    lo = std::max(0, std::min(CALIBRATION_RESIDUAL_BINS - 1, lo));
    int hi = std::max(0, std::min(CALIBRATION_RESIDUAL_BINS - 1, lo + 1));
    return (1.0f - a) * calib.log_lik_by_bin[lo] + a * calib.log_lik_by_bin[hi];
}

static ResidualCalibrationModel fit_occlusion_calibration(std::mt19937& rng) {
    ResidualCalibrationModel calib;
    calib.log_lik_by_bin.assign(CALIBRATION_RESIDUAL_BINS, -12.0f);
    std::vector<float> counts(CALIBRATION_RESIDUAL_BINS, CALIBRATION_HIST_SMOOTH);
    std::uniform_real_distribution<float> ud(0.0f, OBS_RANGE);
    std::uniform_real_distribution<float> ut(OCCLUSION_START_T, OCCLUSION_END_T);
    double residual_sq = 0.0;
    while (calib.valid_samples < CALIBRATION_SAMPLES) {
        calib.attempts++;
        float d = ud(rng);
        float t = ut(rng);
        float z = 0.0f;
        if (!sample_occlusion_sensor(d, rng, t, z)) continue;
        float residual = z - d;
        counts[residual_bin(residual, calib)] += 1.0f;
        residual_sq += static_cast<double>(residual) * residual;
        calib.valid_samples++;
    }

    float max_count = *std::max_element(counts.begin(), counts.end());
    for (int b = 0; b < CALIBRATION_RESIDUAL_BINS; b++) {
        float ll = std::log(counts[b] / max_count);
        calib.log_lik_by_bin[b] = std::max(-12.0f, ll);
    }
    calib.tail_log_lik = calibrated_log_lik_at(-6.0f, calib);
    calib.residual_rmse = std::sqrt(static_cast<float>(residual_sq / calib.valid_samples));
    return calib;
}

static void train_mlp_calibrated_residual_surrogate(GpuMLP& mlp,
                                                    const ResidualCalibrationModel& calib,
                                                    std::vector<float>& loss_curve) {
    std::vector<float> h_train_in(DIRECT_TRAIN_SAMPLES * MLP_INPUT);
    std::vector<float> h_train_tgt(DIRECT_TRAIN_SAMPLES * MLP_OUTPUT);
    std::mt19937 rng_data(4444);
    std::uniform_real_distribution<float> ud(0.0f, OBS_RANGE);
    std::uniform_real_distribution<float> ur(calib.residual_min, calib.residual_max);
    for (int i = 0; i < DIRECT_TRAIN_SAMPLES; i++) {
        float d = ud(rng_data);
        float residual = ur(rng_data);
        float z = std::max(0.0f, d + residual);
        float effective_residual = z - d;
        h_train_in[i * MLP_INPUT + 0] = d / OBS_RANGE;
        h_train_in[i * MLP_INPUT + 1] = effective_residual / (5.0f * OBS_SIGMA);
        h_train_tgt[i] = calibrated_log_lik_at(effective_residual, calib);
    }

    float *d_train_in, *d_train_tgt;
    CUDA_CHECK(cudaMalloc(&d_train_in,  DIRECT_TRAIN_SAMPLES * MLP_INPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_train_tgt, DIRECT_TRAIN_SAMPLES * MLP_OUTPUT * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_train_in, h_train_in.data(),
                          DIRECT_TRAIN_SAMPLES * MLP_INPUT * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_train_tgt, h_train_tgt.data(),
                          DIRECT_TRAIN_SAMPLES * MLP_OUTPUT * sizeof(float),
                          cudaMemcpyHostToDevice));

    loss_curve.clear();
    for (int e = 0; e < DIRECT_TRAIN_EPOCHS; e++) {
        float L = mlp.train_step_backprop(d_train_in, d_train_tgt,
                                          DIRECT_TRAIN_SAMPLES,
                                          DIRECT_TRAIN_LR, MLP_ACTIV);
        loss_curve.push_back(L);
    }
    cudaFree(d_train_in);
    cudaFree(d_train_tgt);
}

// ---------------------------------------------------------------------------
// Visualisation
// ---------------------------------------------------------------------------
static cv::Point2i w2p(float x, float y) {
    return {static_cast<int>(x * VIS_SX),
            PANEL_H - 1 - static_cast<int>(y * VIS_SY)};
}
static void draw_base(cv::Mat& panel, const std::vector<float>& lx, const std::vector<float>& ly) {
    panel.setTo(cv::Scalar(248, 248, 248));
    for (size_t i = 0; i < lx.size(); i++) {
        auto p = w2p(lx[i], ly[i]);
        cv::circle(panel, p, 5, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
        cv::circle(panel, p, 5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}
static void draw_particles(cv::Mat& panel, const std::vector<float>& px,
                           const std::vector<float>& py, cv::Scalar color) {
    for (size_t i = 0; i < px.size(); i++) cv::circle(panel, w2p(px[i], py[i]), 1, color, -1);
}
static void label(cv::Mat& panel, const std::string& s) {
    cv::putText(panel, s, cv::Point(8, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "DPF with learnable MLP observation model ("
              << N_PARTICLES << " particles, " << N_LANDMARKS << " landmarks, "
              << "MLP " << MLP_INPUT << "->" << MLP_HIDDEN << "->" << MLP_OUTPUT << ")"
              << std::endl;
    ObservationMode hard_mode = OBS_OCCLUSION_KIDNAP;
    std::printf("Occlusion+kidnap scene: %.0f%% landmark dropout and %.0f%% short returns "
                "from %.1fs to %.1fs, plus hidden pose jump at %.1fs.\n",
                100.0f * OCCLUSION_DROP_PROB, 100.0f * OCCLUSION_SHORT_PROB,
                OCCLUSION_START_T, OCCLUSION_END_T, KIDNAP_T);

    // --- Landmark layout (same as src/diff_pf.cu)
    std::vector<float> lx(N_LANDMARKS), ly(N_LANDMARKS);
    std::mt19937 rng_world(7);
    std::uniform_real_distribution<float> ux(8.0f, WORLD_W - 8.0f);
    std::uniform_real_distribution<float> uy(6.0f, WORLD_H - 6.0f);
    for (int i = 0; i < N_LANDMARKS; i++) { lx[i] = ux(rng_world); ly[i] = uy(rng_world); }

    float *d_lx, *d_ly, *d_z;  unsigned char* d_zv;
    CUDA_CHECK(cudaMalloc(&d_lx, N_LANDMARKS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ly, N_LANDMARKS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z,  N_LANDMARKS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_zv, N_LANDMARKS));
    CUDA_CHECK(cudaMemcpy(d_lx, lx.data(), N_LANDMARKS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ly, ly.data(), N_LANDMARKS * sizeof(float), cudaMemcpyHostToDevice));

    // --- Step 1: pre-train MLP supervised on (d, z) -> analytic log-lik
    GpuMLP mlp(MLP_INPUT, MLP_HIDDEN, MLP_LAYERS, MLP_OUTPUT);
    mlp.init_random(42);

    // Sample d uniformly in [0, OBS_RANGE] and residual r uniformly in
    // [-5 sigma, 5 sigma], then z = d - r. Sampling r directly (rather
    // than via z = d + N(0, sigma)) gives a balanced distribution of
    // training targets including the parabolic tail, which is what the
    // MLP needs to see to learn the shape rather than collapsing to a
    // near-zero predictor.
    constexpr int N_TRAIN = 16384;
    std::vector<float> h_train_in(N_TRAIN * MLP_INPUT);
    std::vector<float> h_train_tgt(N_TRAIN * MLP_OUTPUT);
    std::mt19937 rng_data(123);
    std::uniform_real_distribution<float> ud(0.0f, OBS_RANGE);
    std::uniform_real_distribution<float> ur(-5.0f * OBS_SIGMA, 5.0f * OBS_SIGMA);
    float two_sig2 = 2.0f * OBS_SIGMA * OBS_SIGMA;
    for (int i = 0; i < N_TRAIN; i++) {
        float d = ud(rng_data);
        float r = ur(rng_data);
        float z = d - r;
        if (z < 0.0f) z = 0.0f;
        h_train_in[i * MLP_INPUT + 0] = d / OBS_RANGE;
        h_train_in[i * MLP_INPUT + 1] = (z - d) / (5.0f * OBS_SIGMA);
        // Clip extreme log-likelihoods so the MLP target range stays
        // bounded; weights = exp(log_lik) below -12 are numerically
        // indistinguishable anyway.
        float ll = -(r * r) / two_sig2;
        if (ll < -12.0f) ll = -12.0f;
        h_train_tgt[i] = ll;
    }
    float *d_train_in, *d_train_tgt;
    CUDA_CHECK(cudaMalloc(&d_train_in,  N_TRAIN * MLP_INPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_train_tgt, N_TRAIN * MLP_OUTPUT * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_train_in,  h_train_in.data(),  N_TRAIN * MLP_INPUT * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_train_tgt, h_train_tgt.data(), N_TRAIN * MLP_OUTPUT * sizeof(float),
                          cudaMemcpyHostToDevice));

    constexpr int MLP_EPOCHS = 2500;
    constexpr float MLP_LR   = 0.01f;
    std::vector<float> mlp_loss_curve;
    auto t_train_0 = std::chrono::high_resolution_clock::now();
    for (int e = 0; e < MLP_EPOCHS; e++) {
        float L = mlp.train_step_backprop(d_train_in, d_train_tgt, N_TRAIN, MLP_LR, MLP_ACTIV);
        mlp_loss_curve.push_back(L);
    }
    auto t_train_1 = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration<double, std::milli>(t_train_1 - t_train_0).count();
    std::printf("MLP supervised pre-training: %d epochs in %.1f ms, "
                "loss %.4f -> %.4f\n",
                MLP_EPOCHS, train_ms, mlp_loss_curve.front(), mlp_loss_curve.back());
    cudaFree(d_train_in); cudaFree(d_train_tgt);

    // --- Step 2: sample MLP at a few (d, z) to verify fit
    std::vector<float> probe_in(8 * MLP_INPUT);
    std::vector<float> probe_out(8 * MLP_OUTPUT);
    float probe_d[] = { 2.0f, 5.0f, 5.0f, 10.0f, 10.0f, 15.0f, 15.0f, 20.0f };
    float probe_z[] = { 2.0f, 5.0f, 6.0f, 10.0f, 12.0f, 15.0f, 17.0f, 20.0f };
    for (int i = 0; i < 8; i++) {
        probe_in[i * MLP_INPUT + 0] = probe_d[i] / OBS_RANGE;
        probe_in[i * MLP_INPUT + 1] = (probe_z[i] - probe_d[i]) / (5.0f * OBS_SIGMA);
    }
    float* d_probe_in; float* d_probe_out;
    CUDA_CHECK(cudaMalloc(&d_probe_in,  8 * MLP_INPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probe_out, 8 * MLP_OUTPUT * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_probe_in, probe_in.data(),
                          8 * MLP_INPUT * sizeof(float), cudaMemcpyHostToDevice));
    mlp.forward_batch(d_probe_in, d_probe_out, 8, MLP_ACTIV);
    CUDA_CHECK(cudaMemcpy(probe_out.data(), d_probe_out,
                          8 * MLP_OUTPUT * sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("MLP fit probe:\n  (d,z)        analytic_log_lik     mlp_log_lik\n");
    for (int i = 0; i < 8; i++) {
        float r = probe_d[i] - probe_z[i];
        float ana = -(r * r) / two_sig2;
        std::printf("  (%4.1f,%4.1f)     %9.4f       %9.4f\n",
                    probe_d[i], probe_z[i], ana, probe_out[i]);
    }
    cudaFree(d_probe_in); cudaFree(d_probe_out);

    // --- Step 3: fine-tune a second MLP directly on tracking loss.
    // Each weight gradient is a central finite difference of the full
    // soft-resampling DPF rollout loss.
    GpuMLP mlp_e2e(MLP_INPUT, MLP_HIDDEN, MLP_LAYERS, MLP_OUTPUT);
    mlp_e2e.load_weights(mlp.get_weights());
    std::vector<float> tracking_loss_curve;
    auto t_e2e_0 = std::chrono::high_resolution_clock::now();
    train_mlp_tracking_finite_difference(mlp_e2e, lx, ly, d_lx, d_ly,
                                         d_z, d_zv, tracking_loss_curve,
                                         hard_mode);
    auto t_e2e_1 = std::chrono::high_resolution_clock::now();
    double e2e_ms = std::chrono::duration<double, std::milli>(t_e2e_1 - t_e2e_0).count();
    std::printf("MLP tracking-loss fine-tuning: %d epochs in %.1f ms, "
                "loss %.4f -> %.4f\n",
                E2E_TRAIN_EPOCHS, e2e_ms,
                tracking_loss_curve.front(), tracking_loss_curve.back());

    // --- Step 4: calibrated observation-surrogate MLP for occlusion short returns.
    // A calibration trace contains known true distances and measured ranges.
    // Missing measurements are skipped just like the PF likelihood, so the
    // MLP learns the valid short-return residual tail from traces.
    GpuMLP mlp_direct(MLP_INPUT, MLP_HIDDEN, MLP_LAYERS, MLP_OUTPUT);
    mlp_direct.load_weights(mlp.get_weights());
    std::vector<float> direct_loss_curve;
    std::mt19937 rng_calib(5151);
    ResidualCalibrationModel calib = fit_occlusion_calibration(rng_calib);
    std::printf("Occlusion calibration trace: %d valid samples from %d attempts, "
                "%d residual bins, residual RMSE %.3f m, log-lik at -6m %.3f\n",
                calib.valid_samples, calib.attempts, CALIBRATION_RESIDUAL_BINS,
                calib.residual_rmse, calib.tail_log_lik);
    auto t_direct_0 = std::chrono::high_resolution_clock::now();
    train_mlp_calibrated_residual_surrogate(mlp_direct, calib, direct_loss_curve);
    auto t_direct_1 = std::chrono::high_resolution_clock::now();
    double direct_ms = std::chrono::duration<double, std::milli>(t_direct_1 - t_direct_0).count();
    std::printf("MLP calibrated observation surrogate: %d epochs in %.1f ms, "
                "loss %.4f -> %.4f\n",
                DIRECT_TRAIN_EPOCHS, direct_ms,
                direct_loss_curve.front(), direct_loss_curve.back());

    // --- Step 5: tracking eval, handcrafted vs three learned likelihoods
    ParticleSet PA, PB, PC, PD;
    PA.alloc(N_PARTICLES, 11);
    PB.alloc(N_PARTICLES, 13);
    PC.alloc(N_PARTICLES, 17);
    PD.alloc(N_PARTICLES, 19);
    std::vector<float> ipx(N_PARTICLES), ipy(N_PARTICLES), ipth(N_PARTICLES);
    std::mt19937 rng_init(42);
    std::normal_distribution<float> nxy(0.0f, 1.5f);
    Pose2 gt0 = scene_gt_at(0.0f, hard_mode);
    for (int i = 0; i < N_PARTICLES; i++) {
        ipx[i]  = gt0.x + nxy(rng_init);
        ipy[i]  = gt0.y + nxy(rng_init);
        ipth[i] = gt0.th;
    }
    upload(PA, N_PARTICLES, ipx, ipy, ipth);
    upload(PB, N_PARTICLES, ipx, ipy, ipth);
    upload(PC, N_PARTICLES, ipx, ipy, ipth);
    upload(PD, N_PARTICLES, ipx, ipy, ipth);

    cv::VideoWriter video("gif/comparison_diff_pf_mlp_occlusion_kidnap.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
                          cv::Size(PANEL_W * 4, PANEL_H));

    std::mt19937 rng_obs_eval(99);
    std::mt19937 rng_eval_A(199);
    std::mt19937 rng_eval_B(299);
    std::mt19937 rng_eval_C(399);
    std::mt19937 rng_eval_D(499);
    double rmse_A = 0.0, rmse_B = 0.0, rmse_C = 0.0, rmse_D = 0.0;
    for (int s = 0; s < N_FRAMES; s++) {
        float t = s * DT;
        Pose2 gt = scene_gt_at(t, hard_mode);
        float v, omega; controls_at(t, v, omega);
        std::vector<float> z; std::vector<unsigned char> valid;
        observe(gt, lx, ly, rng_obs_eval, z, valid, hard_mode, t);
        CUDA_CHECK(cudaMemcpy(d_z,  z.data(),     N_LANDMARKS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_zv, valid.data(), N_LANDMARKS,                 cudaMemcpyHostToDevice));

        auto out_A = run_step(PA, N_PARTICLES, TRAINED_ALPHA, v, omega,
                              0.6f, 0.1f, d_lx, d_ly, d_z, d_zv,
                              LIK_ANALYTIC, nullptr, rng_eval_A);
        auto out_B = run_step(PB, N_PARTICLES, TRAINED_ALPHA, v, omega,
                              0.6f, 0.1f, d_lx, d_ly, d_z, d_zv,
                              LIK_MLP, &mlp, rng_eval_B);
        auto out_C = run_step(PC, N_PARTICLES, TRAINED_ALPHA, v, omega,
                              0.6f, 0.1f, d_lx, d_ly, d_z, d_zv,
                              LIK_MLP, &mlp_e2e, rng_eval_C);
        auto out_D = run_step(PD, N_PARTICLES, TRAINED_ALPHA, v, omega,
                              0.6f, 0.1f, d_lx, d_ly, d_z, d_zv,
                              LIK_MLP, &mlp_direct, rng_eval_D);

        rmse_A += (out_A.ex - gt.x) * (out_A.ex - gt.x) + (out_A.ey - gt.y) * (out_A.ey - gt.y);
        rmse_B += (out_B.ex - gt.x) * (out_B.ex - gt.x) + (out_B.ey - gt.y) * (out_B.ey - gt.y);
        rmse_C += (out_C.ex - gt.x) * (out_C.ex - gt.x) + (out_C.ey - gt.y) * (out_C.ey - gt.y);
        rmse_D += (out_D.ex - gt.x) * (out_D.ex - gt.x) + (out_D.ey - gt.y) * (out_D.ey - gt.y);

        cv::Mat P0(PANEL_H, PANEL_W, CV_8UC3);
        cv::Mat P1(PANEL_H, PANEL_W, CV_8UC3);
        cv::Mat P2(PANEL_H, PANEL_W, CV_8UC3);
        cv::Mat P3(PANEL_H, PANEL_W, CV_8UC3);
        draw_base(P0, lx, ly);  draw_base(P1, lx, ly);  draw_base(P2, lx, ly);  draw_base(P3, lx, ly);
        draw_particles(P0, PA.h_px, PA.h_py, cv::Scalar(60, 60, 200));
        draw_particles(P1, PB.h_px, PB.h_py, cv::Scalar(0, 130, 60));
        draw_particles(P2, PC.h_px, PC.h_py, cv::Scalar(190, 110, 0));
        draw_particles(P3, PD.h_px, PD.h_py, cv::Scalar(150, 55, 150));
        auto gp = w2p(gt.x, gt.y);
        cv::circle(P0, gp, 5, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        cv::circle(P1, gp, 5, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        cv::circle(P2, gp, 5, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        cv::circle(P3, gp, 5, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        cv::circle(P0, w2p(out_A.ex, out_A.ey), 6, cv::Scalar(60, 60, 200), 2, cv::LINE_AA);
        cv::circle(P1, w2p(out_B.ex, out_B.ey), 6, cv::Scalar(0, 130, 60), 2, cv::LINE_AA);
        cv::circle(P2, w2p(out_C.ex, out_C.ey), 6, cv::Scalar(190, 110, 0), 2, cv::LINE_AA);
        cv::circle(P3, w2p(out_D.ex, out_D.ey), 6, cv::Scalar(150, 55, 150), 2, cv::LINE_AA);
        label(P0, "Occlusion+kidnap: Gaussian");
        label(P1, "Occlusion+kidnap: supervised MLP");
        label(P2, "Occlusion+kidnap: tracking-loss MLP");
        label(P3, "Occlusion+kidnap: calibrated surrogate MLP");
        cv::Mat row01, row23, combined;
        cv::hconcat(P0, P1, row01);
        cv::hconcat(P2, P3, row23);
        cv::hconcat(row01, row23, combined);
        video.write(combined);
    }
    video.release();
    PA.free_all(); PB.free_all(); PC.free_all(); PD.free_all();
    cudaFree(d_lx); cudaFree(d_ly); cudaFree(d_z); cudaFree(d_zv);

    rmse_A = std::sqrt(rmse_A / N_FRAMES);
    rmse_B = std::sqrt(rmse_B / N_FRAMES);
    rmse_C = std::sqrt(rmse_C / N_FRAMES);
    rmse_D = std::sqrt(rmse_D / N_FRAMES);
    float ratio_B = static_cast<float>(rmse_B / rmse_A);
    float ratio_C = static_cast<float>(rmse_C / rmse_A);
    float ratio_D = static_cast<float>(rmse_D / rmse_A);
    std::printf("Occlusion+kidnap eval RMSE (alpha=%.2f):\n"
                "  DPF + handcrafted likelihood       = %.3f m\n"
                "  DPF + supervised MLP likelihood    = %.3f m (%.2fx)\n"
                "  DPF + tracking-tuned MLP likelihood = %.3f m (%.2fx)\n"
                "  DPF + calibrated surrogate MLP likelihood = %.3f m (%.2fx)\n",
                TRAINED_ALPHA, rmse_A, rmse_B, ratio_B, rmse_C, ratio_C,
                rmse_D, ratio_D);

    std::system("ffmpeg -y -i gif/comparison_diff_pf_mlp_occlusion_kidnap.avi "
                "-vf 'fps=15,scale=1600:-1:flags=lanczos' -loop 0 "
                "gif/comparison_diff_pf_mlp_occlusion_kidnap.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_diff_pf_mlp_occlusion_kidnap.gif" << std::endl;
    return 0;
}
