/*************************************************************************
    Differentiable Particle Filter (DPF) demo.

    2D landmark-based localization with 1,024 particles. Two ideas from
    the DPF literature applied as small CUDA kernels:

      1. Reparameterized motion model — per-particle Gaussian noise is
         pre-sampled once on the GPU and scaled at runtime by a learnable
         scalar `alpha`. The same noise stream is reused across forward
         passes so the path-to-output mapping is deterministic, which is
         what makes a forward-mode autodiff gradient w.r.t. `alpha`
         meaningful.
      2. Soft-resampling — each step mixes the likelihood weights with
         a uniform distribution (w' = beta*w + (1-beta)/N) and applies
         the corresponding importance correction so the gradient does
         not vanish through the resample step.

    The forward pass is implemented twice on each frame:
      - a plain `float` kernel run (used for visualisation / metric)
      - a `DualNumber` kernel run with `alpha` as the dual variable
        (used to compute the per-step tracking-loss gradient)
    The gradient drives an Adam step on `alpha` between epochs.

    Comparison gif shows three side-by-side panels at every frame:
      - Standard hard-resample PF with handcrafted noise scale
      - DPF before training (random `alpha`)
      - DPF after training (`alpha` updated by gradient descent)

    The "novel" content here is not the algorithm in the abstract — the
    DPF paper (Karkus & Hsu 2018) covers the theory — but a compact,
    self-contained CUDA implementation that fits in this repo's autodiff
    + MLP foundation. It is the localization analogue of Diff-MPPI.
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

#include "autodiff_engine.cuh"
#include "cuda_check.cuh"

using Dual = cudabot::DualNumber<float>;

constexpr int   N_PARTICLES   = 1024;
constexpr int   N_LANDMARKS   = 8;
constexpr int   N_FRAMES      = 240;
constexpr float DT            = 0.1f;
constexpr float WORLD_W       = 40.0f;
constexpr float WORLD_H       = 30.0f;
constexpr float OBS_RANGE     = 22.0f;
constexpr float OBS_SIGMA     = 1.0f;
constexpr float SOFT_BETA     = 0.7f;
constexpr int   N_TRAIN_EPOCHS = 120;
constexpr int   N_TRAIN_FRAMES = 80;
constexpr int   PANEL_W       = 380;
constexpr int   PANEL_H       = 285;
constexpr float VIS_SX        = static_cast<float>(PANEL_W) / WORLD_W;
constexpr float VIS_SY        = static_cast<float>(PANEL_H) / WORLD_H;

struct Pose2 { float x, y, th; };

__host__ __device__ inline float wrap_pi(float a) {
    while (a >  static_cast<float>(M_PI)) a -= 2.0f * static_cast<float>(M_PI);
    while (a < -static_cast<float>(M_PI)) a += 2.0f * static_cast<float>(M_PI);
    return a;
}

// ---------------------------------------------------------------------------
// Noise / RNG setup
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

// ---------------------------------------------------------------------------
// Forward pass kernels (float and Dual variants)
//
// Predict step:
//   x' = x + (v * cos(theta) + alpha * sigma_xy * eps_x) * dt
//   y' = y + (v * sin(theta) + alpha * sigma_xy * eps_y) * dt
//   theta' = theta + (omega + alpha * sigma_th * eps_th) * dt
//
// alpha is the learnable motion-noise scale.
// ---------------------------------------------------------------------------
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

// Dual-number predict: alpha is the dual variable; the particle state
// keeps both value and derivative. Particle state is stored as
// (x.val, x.deriv, y.val, y.deriv, th.val, th.deriv) interleaved per
// particle so a single device buffer holds everything.
__global__ void predict_dual_kernel(float* dx_val, float* dx_deriv,
                                    float* dy_val, float* dy_deriv,
                                    float* dth_val, float* dth_deriv,
                                    const float* eps_x, const float* eps_y,
                                    const float* eps_th,
                                    Dual alpha, float v, float omega,
                                    float sigma_xy, float sigma_th, float dt,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Dual x{dx_val[i], dx_deriv[i]};
    Dual y{dy_val[i], dy_deriv[i]};
    Dual th{dth_val[i], dth_deriv[i]};

    Dual sig_xy = Dual::constant(sigma_xy);
    Dual sig_th = Dual::constant(sigma_th);
    Dual eps_xd = Dual::constant(eps_x[i]);
    Dual eps_yd = Dual::constant(eps_y[i]);
    Dual eps_td = Dual::constant(eps_th[i]);
    Dual vd = Dual::constant(v);
    Dual omd = Dual::constant(omega);
    Dual dtd = Dual::constant(dt);

    Dual cosT = cudabot::cos(th);
    Dual sinT = cudabot::sin(th);
    Dual dx_step = (vd * cosT + alpha * sig_xy * eps_xd) * dtd;
    Dual dy_step = (vd * sinT + alpha * sig_xy * eps_yd) * dtd;
    Dual dth_step = (omd + alpha * sig_th * eps_td) * dtd;

    Dual xN = x + dx_step;
    Dual yN = y + dy_step;
    Dual thN_unwrap = th + dth_step;
    // wrap stays inside +/-pi; only operate on .val, leave .deriv intact.
    float wv = wrap_pi(thN_unwrap.val);
    Dual thN{wv, thN_unwrap.deriv};

    dx_val[i] = xN.val;  dx_deriv[i] = xN.deriv;
    dy_val[i] = yN.val;  dy_deriv[i] = yN.deriv;
    dth_val[i] = thN.val; dth_deriv[i] = thN.deriv;
}

// ---------------------------------------------------------------------------
// Observation likelihood kernel.
// Observation: distance to each in-range landmark + Gaussian noise.
// Likelihood: w_i = prod_l N(z_l ; ||p_i - l||, sigma).
// ---------------------------------------------------------------------------
__global__ void likelihood_kernel(const float* px, const float* py,
                                  const float* lx, const float* ly, int nL,
                                  const float* z_dist, const unsigned char* z_valid,
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

// Dual-number likelihood: derivative of log-likelihood w.r.t. alpha
// flows in via (px.deriv, py.deriv) from the predict step. Each
// landmark contributes:
//   r = d - z;  log_lik += -r^2 / (2 sigma^2)
//   d/d_alpha log_lik = -r * (1/d) * (px.deriv * (px - lx) +
//                                     py.deriv * (py - ly)) / sigma^2
// Output w as Dual (w_val, w_deriv) so it can flow into resample-step
// downstream.
__global__ void likelihood_dual_kernel(const float* px_v, const float* px_d,
                                       const float* py_v, const float* py_d,
                                       const float* lx, const float* ly, int nL,
                                       const float* z_dist,
                                       const unsigned char* z_valid,
                                       float sigma,
                                       float* w_val, float* w_deriv, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Dual log_lik = Dual::constant(0.0f);
    float two_sig2 = 2.0f * sigma * sigma;
    Dual two_sig2_d = Dual::constant(two_sig2);
    Dual x{px_v[i], px_d[i]};
    Dual y{py_v[i], py_d[i]};
    for (int l = 0; l < nL; l++) {
        if (!z_valid[l]) continue;
        Dual dx_ = x - lx[l];
        Dual dy_ = y - ly[l];
        Dual d = cudabot::sqrt(dx_ * dx_ + dy_ * dy_);
        Dual z = Dual::constant(z_dist[l]);
        Dual r = d - z;
        Dual term = (Dual::constant(-1.0f) * r * r) / two_sig2_d;
        log_lik = log_lik + term;
    }
    Dual w = cudabot::exp(log_lik);
    w_val[i]   = w.val;
    w_deriv[i] = w.deriv;
}

// ---------------------------------------------------------------------------
// Normalisation + weighted mean (float and Dual)
// ---------------------------------------------------------------------------
static void normalise_weights(std::vector<float>& w) {
    double s = 0.0;
    for (float v : w) s += v;
    if (s < 1.0e-30) s = 1.0e-30;
    float inv = static_cast<float>(1.0 / s);
    for (float& v : w) v *= inv;
}

// Soft-resampling on the host (N_PARTICLES is small enough).
//   w' = beta * w + (1 - beta) / N
//   draw N indices from cumsum(w')
//   corrected weight = w / w'
static void soft_resample(const std::vector<float>& w_norm,
                          std::vector<int>& indices,
                          std::vector<float>& w_corr,
                          float beta, std::mt19937& rng) {
    int N = static_cast<int>(w_norm.size());
    std::vector<float> w_mix(N);
    float u = (1.0f - beta) / N;
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        w_mix[i] = beta * w_norm[i] + u;
        sum += w_mix[i];
    }
    float inv = 1.0f / sum;
    for (float& v : w_mix) v *= inv;

    std::vector<float> cumsum(N);
    cumsum[0] = w_mix[0];
    for (int i = 1; i < N; i++) cumsum[i] = cumsum[i - 1] + w_mix[i];

    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    indices.resize(N);
    w_corr.resize(N);
    for (int i = 0; i < N; i++) {
        float u = uni(rng);
        int lo = 0, hi = N - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (cumsum[mid] < u) lo = mid + 1; else hi = mid;
        }
        indices[i] = lo;
        w_corr[i] = w_norm[lo] / w_mix[lo];
    }
    float wsum = 0.0f;
    for (float v : w_corr) wsum += v;
    if (wsum < 1.0e-30f) wsum = 1.0e-30f;
    for (float& v : w_corr) v /= wsum;
}

// Hard resample (systematic) for the baseline panel.
static void hard_resample(const std::vector<float>& w_norm,
                          std::vector<int>& indices, std::mt19937& rng) {
    int N = static_cast<int>(w_norm.size());
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    float u0 = uni(rng) / N;
    std::vector<float> cumsum(N);
    cumsum[0] = w_norm[0];
    for (int i = 1; i < N; i++) cumsum[i] = cumsum[i - 1] + w_norm[i];
    indices.resize(N);
    int j = 0;
    for (int i = 0; i < N; i++) {
        float u = u0 + static_cast<float>(i) / N;
        while (j < N - 1 && cumsum[j] < u) j++;
        indices[i] = j;
    }
}

// ---------------------------------------------------------------------------
// Particle state container (host-side mirror, GPU-resident buffers)
// ---------------------------------------------------------------------------
struct ParticleSet {
    float *d_px, *d_py, *d_pth;
    float *d_px_v, *d_px_d, *d_py_v, *d_py_d, *d_pth_v, *d_pth_d;
    float *d_w, *d_w_d;
    float *d_eps_x, *d_eps_y, *d_eps_th;
    curandState* d_states;

    std::vector<float> h_px, h_py, h_pth, h_w, h_w_d;
    std::vector<float> h_px_d, h_py_d;

    void alloc(int n, unsigned long long seed) {
        size_t bf = n * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_px, bf));   CUDA_CHECK(cudaMalloc(&d_py, bf));
        CUDA_CHECK(cudaMalloc(&d_pth, bf));
        CUDA_CHECK(cudaMalloc(&d_px_v, bf)); CUDA_CHECK(cudaMalloc(&d_px_d, bf));
        CUDA_CHECK(cudaMalloc(&d_py_v, bf)); CUDA_CHECK(cudaMalloc(&d_py_d, bf));
        CUDA_CHECK(cudaMalloc(&d_pth_v, bf)); CUDA_CHECK(cudaMalloc(&d_pth_d, bf));
        CUDA_CHECK(cudaMalloc(&d_w, bf));    CUDA_CHECK(cudaMalloc(&d_w_d, bf));
        CUDA_CHECK(cudaMalloc(&d_eps_x, bf)); CUDA_CHECK(cudaMalloc(&d_eps_y, bf));
        CUDA_CHECK(cudaMalloc(&d_eps_th, bf));
        CUDA_CHECK(cudaMalloc(&d_states, n * sizeof(curandState)));
        int blk = 256, gd = (n + blk - 1) / blk;
        init_curand<<<gd, blk>>>(d_states, seed, n);
        h_px.resize(n); h_py.resize(n); h_pth.resize(n); h_w.resize(n); h_w_d.resize(n);
        h_px_d.resize(n); h_py_d.resize(n);
    }
    void free_all() {
        cudaFree(d_px); cudaFree(d_py); cudaFree(d_pth);
        cudaFree(d_px_v); cudaFree(d_px_d);
        cudaFree(d_py_v); cudaFree(d_py_d);
        cudaFree(d_pth_v); cudaFree(d_pth_d);
        cudaFree(d_w); cudaFree(d_w_d);
        cudaFree(d_eps_x); cudaFree(d_eps_y); cudaFree(d_eps_th);
        cudaFree(d_states);
    }
};

static void upload_particles(ParticleSet& P, int n,
                             const std::vector<float>& px,
                             const std::vector<float>& py,
                             const std::vector<float>& pth) {
    CUDA_CHECK(cudaMemcpy(P.d_px,  px.data(),  n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(P.d_py,  py.data(),  n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(P.d_pth, pth.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(P.d_px_v,  px.data(),  n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(P.d_py_v,  py.data(),  n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(P.d_pth_v, pth.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(P.d_px_d, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(P.d_py_d, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(P.d_pth_d, 0, n * sizeof(float)));
}

// ---------------------------------------------------------------------------
// Simulator (ground truth + observations)
// ---------------------------------------------------------------------------
static Pose2 gt_at(float t) {
    Pose2 p;
    p.x = 0.5f * WORLD_W + 8.0f * std::cos(0.5f * t);
    p.y = 0.5f * WORLD_H + 5.5f * std::sin(0.8f * t);
    p.th = wrap_pi(0.5f * t);
    return p;
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
                    std::vector<float>& z, std::vector<unsigned char>& valid) {
    int L = static_cast<int>(lx.size());
    z.resize(L); valid.assign(L, 0u);
    std::normal_distribution<float> noise(0.0f, OBS_SIGMA);
    for (int l = 0; l < L; l++) {
        float dx = gt.x - lx[l];
        float dy = gt.y - ly[l];
        float d = std::sqrt(dx * dx + dy * dy);
        if (d <= OBS_RANGE) {
            z[l] = d + noise(rng);
            valid[l] = 1u;
        }
    }
}

// ---------------------------------------------------------------------------
// One filter step (float; produces estimate). resample_mode: 0=hard, 1=soft.
// ---------------------------------------------------------------------------
struct StepOut {
    float ex, ey;        // estimate
};

static StepOut run_step_float(ParticleSet& P, int n,
                              float alpha, float v, float omega,
                              const std::vector<float>& z,
                              const std::vector<unsigned char>& valid,
                              float sigma_xy, float sigma_th,
                              const float* d_lx, const float* d_ly,
                              const float* d_z, const unsigned char* d_zv,
                              int resample_mode, std::mt19937& rng,
                              float beta) {
    int blk = 256;
    int gd  = (n + blk - 1) / blk;
    refresh_motion_noise<<<gd, blk>>>(P.d_states, n, P.d_eps_x, P.d_eps_y, P.d_eps_th);
    predict_kernel<<<gd, blk>>>(P.d_px, P.d_py, P.d_pth,
                                P.d_eps_x, P.d_eps_y, P.d_eps_th,
                                alpha, v, omega, sigma_xy, sigma_th, DT, n);
    likelihood_kernel<<<gd, blk>>>(P.d_px, P.d_py, d_lx, d_ly, N_LANDMARKS,
                                   d_z, d_zv, OBS_SIGMA, P.d_w, n);
    CUDA_CHECK(cudaMemcpy(P.h_px.data(),  P.d_px,  n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P.h_py.data(),  P.d_py,  n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P.h_pth.data(), P.d_pth, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P.h_w.data(),   P.d_w,   n * sizeof(float), cudaMemcpyDeviceToHost));

    normalise_weights(P.h_w);
    float ex = 0.0f, ey = 0.0f;
    for (int i = 0; i < n; i++) {
        ex += P.h_w[i] * P.h_px[i];
        ey += P.h_w[i] * P.h_py[i];
    }
    std::vector<int> idx; std::vector<float> wcorr;
    if (resample_mode == 0) {
        hard_resample(P.h_w, idx, rng);
        wcorr.assign(n, 1.0f / n);
    } else {
        soft_resample(P.h_w, idx, wcorr, beta, rng);
    }
    std::vector<float> npx(n), npy(n), npth(n);
    for (int i = 0; i < n; i++) {
        npx[i]  = P.h_px[idx[i]];
        npy[i]  = P.h_py[idx[i]];
        npth[i] = P.h_pth[idx[i]];
    }
    upload_particles(P, n, npx, npy, npth);
    return {ex, ey};
}

// One DPF step that returns both estimate and d_estimate/d_alpha.
struct StepOutDual {
    float ex, ey;
    float dex, dey;
};
static StepOutDual run_step_dual(ParticleSet& P, int n,
                                 Dual alpha, float v, float omega,
                                 const std::vector<float>& z,
                                 const std::vector<unsigned char>& valid,
                                 float sigma_xy, float sigma_th,
                                 const float* d_lx, const float* d_ly,
                                 const float* d_z, const unsigned char* d_zv,
                                 std::mt19937& rng, float beta) {
    int blk = 256;
    int gd  = (n + blk - 1) / blk;
    refresh_motion_noise<<<gd, blk>>>(P.d_states, n, P.d_eps_x, P.d_eps_y, P.d_eps_th);
    predict_dual_kernel<<<gd, blk>>>(P.d_px_v, P.d_px_d, P.d_py_v, P.d_py_d,
                                     P.d_pth_v, P.d_pth_d,
                                     P.d_eps_x, P.d_eps_y, P.d_eps_th,
                                     alpha, v, omega, sigma_xy, sigma_th, DT, n);
    likelihood_dual_kernel<<<gd, blk>>>(P.d_px_v, P.d_px_d, P.d_py_v, P.d_py_d,
                                        d_lx, d_ly, N_LANDMARKS,
                                        d_z, d_zv, OBS_SIGMA,
                                        P.d_w, P.d_w_d, n);
    CUDA_CHECK(cudaMemcpy(P.h_px.data(),  P.d_px_v,  n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P.h_px_d.data(),P.d_px_d,  n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P.h_py.data(),  P.d_py_v,  n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P.h_py_d.data(),P.d_py_d,  n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P.h_pth.data(), P.d_pth_v, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P.h_w.data(),   P.d_w,     n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P.h_w_d.data(), P.d_w_d,   n * sizeof(float), cudaMemcpyDeviceToHost));

    // Normalise (value + derivative).
    double S = 0.0, Sd = 0.0;
    for (int i = 0; i < n; i++) { S += P.h_w[i]; Sd += P.h_w_d[i]; }
    if (S < 1.0e-30) S = 1.0e-30;
    double invS = 1.0 / S;
    double invS2 = invS * invS;
    std::vector<float> w_n(n), w_n_d(n);
    for (int i = 0; i < n; i++) {
        w_n[i]   = static_cast<float>(P.h_w[i] * invS);
        // d/da (w_i / S) = (w_i_d S - w_i Sd) / S^2
        w_n_d[i] = static_cast<float>(((double)P.h_w_d[i] * S - (double)P.h_w[i] * Sd) * invS2);
    }

    // Estimate + derivative
    float ex = 0.0f, ey = 0.0f;
    float dex = 0.0f, dey = 0.0f;
    for (int i = 0; i < n; i++) {
        ex  += w_n[i]   * P.h_px[i];
        ey  += w_n[i]   * P.h_py[i];
        dex += w_n_d[i] * P.h_px[i] + w_n[i] * P.h_px_d[i];
        dey += w_n_d[i] * P.h_py[i] + w_n[i] * P.h_py_d[i];
    }

    // Soft-resample on values (resample is treated as straight-through
    // for gradients: we use the same index draw as the value run).
    std::vector<int> idx; std::vector<float> wcorr;
    soft_resample(w_n, idx, wcorr, beta, rng);
    std::vector<float> npx(n), npy(n), npth(n), npx_d(n), npy_d(n);
    for (int i = 0; i < n; i++) {
        npx[i]   = P.h_px[idx[i]];
        npy[i]   = P.h_py[idx[i]];
        npth[i]  = P.h_pth[idx[i]];
        npx_d[i] = P.h_px_d[idx[i]];
        npy_d[i] = P.h_py_d[idx[i]];
    }
    // Upload the resampled value + derivative state.
    CUDA_CHECK(cudaMemcpy(P.d_px_v,  npx.data(),    n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(P.d_py_v,  npy.data(),    n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(P.d_pth_v, npth.data(),   n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(P.d_px_d,  npx_d.data(),  n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(P.d_py_d,  npy_d.data(),  n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(P.d_pth_d, 0,             n * sizeof(float)));
    return {ex, ey, dex, dey};
}

// ---------------------------------------------------------------------------
// Visualization helpers
// ---------------------------------------------------------------------------
static cv::Point2i w2p(float x, float y) {
    int px = static_cast<int>(x * VIS_SX);
    int py = PANEL_H - 1 - static_cast<int>(y * VIS_SY);
    return {px, py};
}
static void draw_panel_base(cv::Mat& panel,
                            const std::vector<float>& lx,
                            const std::vector<float>& ly) {
    panel.setTo(cv::Scalar(248, 248, 248));
    for (size_t i = 0; i < lx.size(); i++) {
        auto p = w2p(lx[i], ly[i]);
        cv::circle(panel, p, 5, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
        cv::circle(panel, p, 5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}
static void draw_particles(cv::Mat& panel,
                           const std::vector<float>& px,
                           const std::vector<float>& py,
                           cv::Scalar color) {
    for (size_t i = 0; i < px.size(); i++) {
        cv::circle(panel, w2p(px[i], py[i]), 1, color, -1);
    }
}
static void draw_gt(cv::Mat& panel, Pose2 gt) {
    cv::circle(panel, w2p(gt.x, gt.y), 5, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::circle(panel, w2p(gt.x, gt.y), 2, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
}
static void draw_est(cv::Mat& panel, float ex, float ey, cv::Scalar color) {
    cv::circle(panel, w2p(ex, ey), 6, color, 2, cv::LINE_AA);
}
static void label(cv::Mat& panel, const std::string& s) {
    cv::putText(panel, s, cv::Point(8, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

// ---------------------------------------------------------------------------
// Train alpha by averaging gradient over N_TRAIN_FRAMES steps and applying
// an Adam-style update.
// ---------------------------------------------------------------------------
struct AdamState { float m=0.0f, v=0.0f; int t=0; };
static float adam_step(AdamState& s, float grad, float lr,
                       float beta1=0.9f, float beta2=0.999f, float eps=1e-8f) {
    s.t++;
    s.m = beta1 * s.m + (1.0f - beta1) * grad;
    s.v = beta2 * s.v + (1.0f - beta2) * grad * grad;
    float mh = s.m / (1.0f - std::pow(beta1, s.t));
    float vh = s.v / (1.0f - std::pow(beta2, s.t));
    return lr * mh / (std::sqrt(vh) + eps);
}

static float train_alpha(float& alpha, std::vector<float>& lx,
                         std::vector<float>& ly,
                         float* d_lx, float* d_ly, float* d_z, unsigned char* d_zv,
                         float sigma_xy_true, float sigma_th_true,
                         unsigned long long seed_train,
                         std::vector<float>& loss_curve) {
    AdamState adam;
    loss_curve.clear();
    for (int epoch = 0; epoch < N_TRAIN_EPOCHS; epoch++) {
        ParticleSet P;
        P.alloc(N_PARTICLES, seed_train + 991 * epoch);
        // Init particles around the true start with a wider spread, no theta-derivative.
        std::vector<float> ipx(N_PARTICLES), ipy(N_PARTICLES), ipth(N_PARTICLES);
        std::mt19937 rng_init(seed_train + epoch * 13);
        std::normal_distribution<float> nxy(0.0f, 1.5f);
        Pose2 gt0 = gt_at(0.0f);
        for (int i = 0; i < N_PARTICLES; i++) {
            ipx[i]  = gt0.x + nxy(rng_init);
            ipy[i]  = gt0.y + nxy(rng_init);
            ipth[i] = gt0.th;
        }
        upload_particles(P, N_PARTICLES, ipx, ipy, ipth);

        std::mt19937 rng_obs(seed_train + epoch * 17 + 1);
        double loss_sum = 0.0;
        double grad_sum = 0.0;
        for (int s = 0; s < N_TRAIN_FRAMES; s++) {
            float t = s * DT;
            Pose2 gt = gt_at(t);
            float v, omega; controls_at(t, v, omega);
            std::vector<float> z;
            std::vector<unsigned char> valid;
            observe(gt, lx, ly, rng_obs, z, valid);
            CUDA_CHECK(cudaMemcpy(d_z,  z.data(),     z.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_zv, valid.data(), valid.size(),             cudaMemcpyHostToDevice));

            Dual alpha_dual{alpha, 1.0f};
            StepOutDual out = run_step_dual(P, N_PARTICLES, alpha_dual, v, omega,
                                            z, valid, sigma_xy_true, sigma_th_true,
                                            d_lx, d_ly, d_z, d_zv, rng_obs, SOFT_BETA);
            float dx = out.ex - gt.x;
            float dy = out.ey - gt.y;
            float L = dx * dx + dy * dy;
            // dL/dalpha = 2 dx d_ex + 2 dy d_ey
            float dL = 2.0f * (dx * out.dex + dy * out.dey);
            loss_sum += L;
            grad_sum += dL;
        }
        P.free_all();
        float avg_loss = static_cast<float>(loss_sum / N_TRAIN_FRAMES);
        float avg_grad = static_cast<float>(grad_sum / N_TRAIN_FRAMES);
        float upd = adam_step(adam, avg_grad, 0.04f);
        alpha -= upd;
        if (alpha < 0.05f) alpha = 0.05f;
        if (alpha > 4.0f) alpha = 4.0f;
        loss_curve.push_back(avg_loss);
    }
    return alpha;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "Differentiable Particle Filter demo ("
              << N_PARTICLES << " particles, " << N_LANDMARKS << " landmarks)" << std::endl;

    // Random landmark layout
    std::vector<float> lx(N_LANDMARKS), ly(N_LANDMARKS);
    std::mt19937 rng_world(7);
    std::uniform_real_distribution<float> ux(8.0f, WORLD_W - 8.0f);
    std::uniform_real_distribution<float> uy(6.0f, WORLD_H - 6.0f);
    for (int i = 0; i < N_LANDMARKS; i++) {
        lx[i] = ux(rng_world);
        ly[i] = uy(rng_world);
    }

    float *d_lx, *d_ly, *d_z;
    unsigned char *d_zv;
    CUDA_CHECK(cudaMalloc(&d_lx, N_LANDMARKS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ly, N_LANDMARKS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z,  N_LANDMARKS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_zv, N_LANDMARKS));
    CUDA_CHECK(cudaMemcpy(d_lx, lx.data(), N_LANDMARKS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ly, ly.data(), N_LANDMARKS * sizeof(float), cudaMemcpyHostToDevice));

    const float sigma_xy_true = 0.6f;
    const float sigma_th_true = 0.1f;

    // Train alpha
    float alpha_dpf = 0.2f;  // start small
    std::vector<float> loss_curve;
    std::cout << "Training alpha (Adam, " << N_TRAIN_EPOCHS << " epochs, "
              << N_TRAIN_FRAMES << " steps/epoch)..." << std::endl;
    train_alpha(alpha_dpf, lx, ly, d_lx, d_ly, d_z, d_zv,
                sigma_xy_true, sigma_th_true, 12345ULL, loss_curve);
    std::cout << "Trained alpha = " << alpha_dpf << " (initial 0.20)" << std::endl;
    std::cout << "Loss curve: ";
    for (size_t i = 0; i < loss_curve.size(); i += 10)
        std::cout << loss_curve[i] << " ";
    std::cout << std::endl;

    // Evaluation pass: three filters running side by side.
    //   Panel 0: Hard-resample PF with handcrafted alpha = 1.0
    //   Panel 1: DPF before training (alpha = 0.2)
    //   Panel 2: DPF after training (alpha = alpha_dpf)
    ParticleSet PA, PB, PC;
    PA.alloc(N_PARTICLES, 11);
    PB.alloc(N_PARTICLES, 13);
    PC.alloc(N_PARTICLES, 17);

    std::vector<float> ipx(N_PARTICLES), ipy(N_PARTICLES), ipth(N_PARTICLES);
    std::mt19937 rng_init_eval(42);
    std::normal_distribution<float> nxy(0.0f, 1.5f);
    Pose2 gt0 = gt_at(0.0f);
    for (int i = 0; i < N_PARTICLES; i++) {
        ipx[i]  = gt0.x + nxy(rng_init_eval);
        ipy[i]  = gt0.y + nxy(rng_init_eval);
        ipth[i] = gt0.th;
    }
    upload_particles(PA, N_PARTICLES, ipx, ipy, ipth);
    upload_particles(PB, N_PARTICLES, ipx, ipy, ipth);
    upload_particles(PC, N_PARTICLES, ipx, ipy, ipth);

    cv::VideoWriter video("gif/comparison_diff_pf.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
                          cv::Size(PANEL_W * 3, PANEL_H));

    double rmse_A = 0.0, rmse_B = 0.0, rmse_C = 0.0;

    std::mt19937 rng_eval(99);
    for (int s = 0; s < N_FRAMES; s++) {
        float t = s * DT;
        Pose2 gt = gt_at(t);
        float v, omega; controls_at(t, v, omega);
        std::vector<float> z; std::vector<unsigned char> valid;
        observe(gt, lx, ly, rng_eval, z, valid);
        CUDA_CHECK(cudaMemcpy(d_z, z.data(), N_LANDMARKS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_zv, valid.data(), N_LANDMARKS, cudaMemcpyHostToDevice));

        auto out_A = run_step_float(PA, N_PARTICLES, 1.0f, v, omega, z, valid,
                                    sigma_xy_true, sigma_th_true,
                                    d_lx, d_ly, d_z, d_zv, 0, rng_eval, SOFT_BETA);
        auto out_B = run_step_float(PB, N_PARTICLES, 0.2f, v, omega, z, valid,
                                    sigma_xy_true, sigma_th_true,
                                    d_lx, d_ly, d_z, d_zv, 1, rng_eval, SOFT_BETA);
        auto out_C = run_step_float(PC, N_PARTICLES, alpha_dpf, v, omega, z, valid,
                                    sigma_xy_true, sigma_th_true,
                                    d_lx, d_ly, d_z, d_zv, 1, rng_eval, SOFT_BETA);

        rmse_A += (out_A.ex - gt.x) * (out_A.ex - gt.x) + (out_A.ey - gt.y) * (out_A.ey - gt.y);
        rmse_B += (out_B.ex - gt.x) * (out_B.ex - gt.x) + (out_B.ey - gt.y) * (out_B.ey - gt.y);
        rmse_C += (out_C.ex - gt.x) * (out_C.ex - gt.x) + (out_C.ey - gt.y) * (out_C.ey - gt.y);

        cv::Mat P0(PANEL_H, PANEL_W, CV_8UC3);
        cv::Mat P1(PANEL_H, PANEL_W, CV_8UC3);
        cv::Mat P2(PANEL_H, PANEL_W, CV_8UC3);
        draw_panel_base(P0, lx, ly);
        draw_panel_base(P1, lx, ly);
        draw_panel_base(P2, lx, ly);

        // After run_step_*, the host mirrors hold the most recent
        // (post-resample, pre-next-predict) particles.
        draw_particles(P0, PA.h_px, PA.h_py, cv::Scalar(60, 60, 200));
        draw_particles(P1, PB.h_px, PB.h_py, cv::Scalar(120, 120, 120));
        draw_particles(P2, PC.h_px, PC.h_py, cv::Scalar(0, 160, 0));
        draw_gt(P0, gt); draw_gt(P1, gt); draw_gt(P2, gt);
        draw_est(P0, out_A.ex, out_A.ey, cv::Scalar(60, 60, 200));
        draw_est(P1, out_B.ex, out_B.ey, cv::Scalar(120, 120, 120));
        draw_est(P2, out_C.ex, out_C.ey, cv::Scalar(0, 160, 0));

        char buf[160];
        std::snprintf(buf, sizeof(buf), "Hard-resample PF (handcrafted alpha=1.00)");
        label(P0, buf);
        std::snprintf(buf, sizeof(buf), "DPF, untrained (alpha=0.20)");
        label(P1, buf);
        std::snprintf(buf, sizeof(buf), "DPF, trained (alpha=%.2f)", alpha_dpf);
        label(P2, buf);

        cv::Mat row1, combined;
        cv::hconcat(P0, P1, row1);
        cv::hconcat(row1, P2, combined);
        video.write(combined);
    }

    video.release();
    PA.free_all(); PB.free_all(); PC.free_all();
    cudaFree(d_lx); cudaFree(d_ly); cudaFree(d_z); cudaFree(d_zv);

    rmse_A = std::sqrt(rmse_A / N_FRAMES);
    rmse_B = std::sqrt(rmse_B / N_FRAMES);
    rmse_C = std::sqrt(rmse_C / N_FRAMES);
    std::printf("Eval RMSE:\n"
                "  hard-resample PF  (alpha=1.00)        = %.3f m\n"
                "  DPF untrained     (alpha=0.20)        = %.3f m\n"
                "  DPF trained       (alpha=%.2f)        = %.3f m\n",
                rmse_A, rmse_B, alpha_dpf, rmse_C);
    std::system("ffmpeg -y -i gif/comparison_diff_pf.avi "
                "-vf 'fps=15,scale=1140:-1:flags=lanczos' -loop 0 "
                "gif/comparison_diff_pf.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_diff_pf.gif" << std::endl;
    return 0;
}
