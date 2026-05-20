/*************************************************************************
    Differentiable Particle Filter with a learnable MLP observation model.

    Extends the soft-resampling + reparameterized-motion DPF added in
    src/diff_pf.cu: the per-landmark likelihood
        log p(z_l | x) = -(||x - l|| - z_l)^2 / (2 sigma^2)
    is replaced with a small MLP h_theta(d, z) -> log_lik, where
        d = ||particle - landmark||,    z = measured distance
    The MLP is trained supervised against the analytic Gaussian on a
    synthetic dataset, then dropped into the DPF observation kernel.

    The point is not "neural likelihood beats Gaussian" (the analytic
    form is optimal under the assumed noise model). The point is that
    the DPF architecture accepts a swappable observation model, which
    is what enables future work where the observation is non-Gaussian
    or there is no analytic form. Demonstrating that a learned model
    achieves comparable tracking RMSE validates the framework.

    Output:
      - Training curve (MSE on log-likelihood, supervised pre-training)
      - Tracking gif: handcrafted DPF vs MLP-DPF, same scene, same
        alpha (=3.14, learned in src/diff_pf.cu)
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <chrono>

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

constexpr int   MLP_INPUT     = 2;   // (d, z)
constexpr int   MLP_HIDDEN    = 16;
constexpr int   MLP_LAYERS    = 1;   // single hidden layer
constexpr int   MLP_OUTPUT    = 1;   // log-likelihood
constexpr int   MLP_ACTIV     = 1;   // tanh hidden activation

constexpr int   PANEL_W       = 480;
constexpr int   PANEL_H       = 360;
constexpr float VIS_SX        = static_cast<float>(PANEL_W) / WORLD_W;
constexpr float VIS_SY        = static_cast<float>(PANEL_H) / WORLD_H;

struct Pose2 { float x, y, th; };

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
        mlp_in[0] = d;
        mlp_in[1] = z_dist[l];
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
        h_train_in[i * MLP_INPUT + 0] = d;
        h_train_in[i * MLP_INPUT + 1] = z;
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
        probe_in[i * MLP_INPUT + 0] = probe_d[i];
        probe_in[i * MLP_INPUT + 1] = probe_z[i];
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

    // --- Step 3: tracking eval, handcrafted vs MLP
    ParticleSet PA, PB;
    PA.alloc(N_PARTICLES, 11);
    PB.alloc(N_PARTICLES, 13);
    std::vector<float> ipx(N_PARTICLES), ipy(N_PARTICLES), ipth(N_PARTICLES);
    std::mt19937 rng_init(42);
    std::normal_distribution<float> nxy(0.0f, 1.5f);
    Pose2 gt0 = gt_at(0.0f);
    for (int i = 0; i < N_PARTICLES; i++) {
        ipx[i]  = gt0.x + nxy(rng_init);
        ipy[i]  = gt0.y + nxy(rng_init);
        ipth[i] = gt0.th;
    }
    upload(PA, N_PARTICLES, ipx, ipy, ipth);
    upload(PB, N_PARTICLES, ipx, ipy, ipth);

    cv::VideoWriter video("gif/comparison_diff_pf_mlp.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
                          cv::Size(PANEL_W * 2, PANEL_H));

    std::mt19937 rng_eval(99);
    double rmse_A = 0.0, rmse_B = 0.0;
    for (int s = 0; s < N_FRAMES; s++) {
        float t = s * DT;
        Pose2 gt = gt_at(t);
        float v, omega; controls_at(t, v, omega);
        std::vector<float> z; std::vector<unsigned char> valid;
        observe(gt, lx, ly, rng_eval, z, valid);
        CUDA_CHECK(cudaMemcpy(d_z,  z.data(),     N_LANDMARKS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_zv, valid.data(), N_LANDMARKS,                 cudaMemcpyHostToDevice));

        auto out_A = run_step(PA, N_PARTICLES, TRAINED_ALPHA, v, omega,
                              0.6f, 0.1f, d_lx, d_ly, d_z, d_zv,
                              LIK_ANALYTIC, nullptr, rng_eval);
        auto out_B = run_step(PB, N_PARTICLES, TRAINED_ALPHA, v, omega,
                              0.6f, 0.1f, d_lx, d_ly, d_z, d_zv,
                              LIK_MLP, &mlp, rng_eval);

        rmse_A += (out_A.ex - gt.x) * (out_A.ex - gt.x) + (out_A.ey - gt.y) * (out_A.ey - gt.y);
        rmse_B += (out_B.ex - gt.x) * (out_B.ex - gt.x) + (out_B.ey - gt.y) * (out_B.ey - gt.y);

        cv::Mat P0(PANEL_H, PANEL_W, CV_8UC3);
        cv::Mat P1(PANEL_H, PANEL_W, CV_8UC3);
        draw_base(P0, lx, ly);  draw_base(P1, lx, ly);
        draw_particles(P0, PA.h_px, PA.h_py, cv::Scalar(60, 60, 200));
        draw_particles(P1, PB.h_px, PB.h_py, cv::Scalar(0, 130, 60));
        auto gp = w2p(gt.x, gt.y);
        cv::circle(P0, gp, 5, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        cv::circle(P1, gp, 5, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        cv::circle(P0, w2p(out_A.ex, out_A.ey), 6, cv::Scalar(60, 60, 200), 2, cv::LINE_AA);
        cv::circle(P1, w2p(out_B.ex, out_B.ey), 6, cv::Scalar(0, 130, 60), 2, cv::LINE_AA);
        label(P0, "DPF + handcrafted Gaussian likelihood");
        label(P1, "DPF + MLP-learned likelihood (2->16->1, supervised)");
        cv::Mat combined;
        cv::hconcat(P0, P1, combined);
        video.write(combined);
    }
    video.release();
    PA.free_all(); PB.free_all();
    cudaFree(d_lx); cudaFree(d_ly); cudaFree(d_z); cudaFree(d_zv);

    rmse_A = std::sqrt(rmse_A / N_FRAMES);
    rmse_B = std::sqrt(rmse_B / N_FRAMES);
    float ratio = static_cast<float>(rmse_B / rmse_A);
    std::printf("Eval RMSE (alpha=%.2f):\n"
                "  DPF + handcrafted likelihood = %.3f m\n"
                "  DPF + MLP likelihood         = %.3f m\n"
                "  MLP/handcrafted ratio = %.2fx\n",
                TRAINED_ALPHA, rmse_A, rmse_B, ratio);

    std::system("ffmpeg -y -i gif/comparison_diff_pf_mlp.avi "
                "-vf 'fps=15,scale=960:-1:flags=lanczos' -loop 0 "
                "gif/comparison_diff_pf_mlp.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_diff_pf_mlp.gif" << std::endl;
    return 0;
}
