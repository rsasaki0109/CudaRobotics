/*************************************************************************
    Particle filter on a *realistic* range sensor (range noise + dropouts
    + occasional multi-path), comparing three observation likelihoods:

      1. Gaussian (handcrafted, assumes only Gaussian noise)
      2. Cauchy   (heavy-tailed, robust to outliers, fixed hyperparameter)
      3. Learned  (tiny MLP trained on simulator data to map observed
                  range -> likelihood at expected range)

    All three PFs run K=8192 particles in parallel on the GPU. The
    sensor model returns the noisy distance to the nearest landmark
    with a 15% dropout (return 0) and a 4% multi-path event (return
    2x the true range). Tracking RMSE and weight-update GPU time are
    reported for each PF.

    The MLP is trained on the GPU before the PF run: 16k pairs of
    (expected_range, observed_range) labelled "positive" (truth) or
    "negative" (random offset), fit a 3-input 16-hidden 1-output
    network with SGD + sigmoid output to predict likelihood. The
    same trained weights are then used as the per-particle likelihood.

    Output: gif/pf_realistic_obs.gif  (3-panel particle clouds)
    Headline: RMSE of Gaussian / Cauchy / learned with realistic sensor.
 ************************************************************************/

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(EXIT_FAILURE); } } while (0)

// -------------------------------------------------------------------------
// World / sensor parameters
// -------------------------------------------------------------------------
constexpr float WORLD_X = 40.0f;
constexpr float WORLD_Y = 30.0f;
constexpr int   N_LM = 8;
constexpr int   K_PART = 8192;
constexpr int   N_STEPS = 200;
constexpr float DT = 0.1f;
constexpr float SIGMA = 0.4f;      // assumed noise (handcrafted models)
constexpr float TRUE_SIGMA = 0.4f; // actual Gaussian noise
constexpr float DROPOUT_PROB = 0.15f;
constexpr float MULTIPATH_PROB = 0.04f;
constexpr float MOTION_SIGMA = 0.18f;
constexpr int   PANEL_W = 540;
constexpr int   PANEL_H = 400;

// MLP architecture
constexpr int MLP_IN  = 3;   // (expected_range, observed_range, dropout_flag)
constexpr int MLP_H   = 16;
constexpr int MLP_OUT = 1;
constexpr int MLP_NW  = MLP_IN * MLP_H + MLP_H + MLP_H * MLP_OUT + MLP_OUT;

// -------------------------------------------------------------------------
// Realistic sensor (host)
// -------------------------------------------------------------------------
static float sensor_realistic(float x, float y,
                              const float* lm_x, const float* lm_y,
                              int n_lm, std::mt19937& rng) {
    float zmin = FLT_MAX;
    for (int i = 0; i < n_lm; i++) {
        float dx = x - lm_x[i], dy = y - lm_y[i];
        float d = std::sqrt(dx * dx + dy * dy);
        if (d < zmin) zmin = d;
    }
    // Gaussian noise
    std::normal_distribution<float> g(0.0f, TRUE_SIGMA);
    float z = zmin + g(rng);
    // Dropout
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    if (u(rng) < DROPOUT_PROB) return -1.0f;  // -1 = dropout sentinel
    // Multi-path
    if (u(rng) < MULTIPATH_PROB) return 2.0f * zmin + g(rng);
    return z;
}

// -------------------------------------------------------------------------
// MLP kernels (tiny, weights in constant memory)
// -------------------------------------------------------------------------
__constant__ float c_mlp_w[MLP_NW];

__device__ float mlp_forward(float in0, float in1, float in2) {
    float h[MLP_H];
    int off = 0;
    #pragma unroll
    for (int j = 0; j < MLP_H; j++) {
        float s = c_mlp_w[off + 0] * in0 + c_mlp_w[off + 1] * in1 + c_mlp_w[off + 2] * in2;
        off += MLP_IN;
        h[j] = tanhf(s);
    }
    int bo = MLP_IN * MLP_H;
    #pragma unroll
    for (int j = 0; j < MLP_H; j++) h[j] += c_mlp_w[bo + j];
    int wo = bo + MLP_H;
    float y = 0.0f;
    #pragma unroll
    for (int j = 0; j < MLP_H; j++) y += c_mlp_w[wo + j] * h[j];
    y += c_mlp_w[wo + MLP_H];
    return 1.0f / (1.0f + expf(-y));
}

// CPU MLP forward (for training)
struct MLP {
    float w[MLP_NW];

    static float act(float s) { return std::tanh(s); }
    static float dact(float s) { return 1.0f - std::tanh(s) * std::tanh(s); }

    float forward(float in0, float in1, float in2, float* hid_out = nullptr) const {
        float h[MLP_H];
        int off = 0;
        for (int j = 0; j < MLP_H; j++) {
            float s = w[off + 0] * in0 + w[off + 1] * in1 + w[off + 2] * in2;
            off += MLP_IN;
            h[j] = act(s);
        }
        int bo = MLP_IN * MLP_H;
        for (int j = 0; j < MLP_H; j++) h[j] += w[bo + j];
        if (hid_out) for (int j = 0; j < MLP_H; j++) hid_out[j] = h[j];
        int wo = bo + MLP_H;
        float y = 0.0f;
        for (int j = 0; j < MLP_H; j++) y += w[wo + j] * h[j];
        y += w[wo + MLP_H];
        return 1.0f / (1.0f + std::exp(-y));
    }

    void zero_grads(float* g) const { for (int i = 0; i < MLP_NW; i++) g[i] = 0.0f; }

    // Backprop logistic loss: L = -(y*log p + (1-y)*log(1-p)). dL/dlogit = p - y.
    void backprop(float in0, float in1, float in2, float y_lbl, float* g) {
        float h[MLP_H];
        // forward pre-act
        float pre[MLP_H];
        int off = 0;
        for (int j = 0; j < MLP_H; j++) {
            pre[j] = w[off + 0] * in0 + w[off + 1] * in1 + w[off + 2] * in2;
            off += MLP_IN;
            h[j] = act(pre[j]);
        }
        int bo = MLP_IN * MLP_H;
        for (int j = 0; j < MLP_H; j++) h[j] += w[bo + j];
        int wo = bo + MLP_H;
        float logit = 0.0f;
        for (int j = 0; j < MLP_H; j++) logit += w[wo + j] * h[j];
        logit += w[wo + MLP_H];
        float p = 1.0f / (1.0f + std::exp(-logit));
        float dlogit = p - y_lbl;
        // output gradients
        for (int j = 0; j < MLP_H; j++) g[wo + j] += dlogit * h[j];
        g[wo + MLP_H] += dlogit;
        // hidden gradients
        for (int j = 0; j < MLP_H; j++) {
            float dh = dlogit * w[wo + j];
            // bias
            g[bo + j] += dh;
            // pre-activation gradient
            float dpre = dh * dact(pre[j]);
            // input weights
            g[j * MLP_IN + 0] += dpre * in0;
            g[j * MLP_IN + 1] += dpre * in1;
            g[j * MLP_IN + 2] += dpre * in2;
        }
    }

    void init_random(unsigned long long seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> n(0.0f, 0.3f);
        for (int i = 0; i < MLP_NW; i++) w[i] = n(rng);
    }
};

static float train_mlp(MLP& mlp, std::vector<float>& xs0, std::vector<float>& xs1,
                       std::vector<float>& xs2, std::vector<float>& ys,
                       int epochs, float lr) {
    int N = static_cast<int>(ys.size());
    std::vector<int> idx(N);
    for (int i = 0; i < N; i++) idx[i] = i;
    std::mt19937 rng(123);
    float grads[MLP_NW];
    float final_loss = 0.0f;
    for (int e = 0; e < epochs; e++) {
        std::shuffle(idx.begin(), idx.end(), rng);
        double loss_acc = 0.0;
        const int BATCH = 64;
        for (int s = 0; s < N; s += BATCH) {
            mlp.zero_grads(grads);
            int end = std::min(s + BATCH, N);
            int n = end - s;
            for (int b = s; b < end; b++) {
                int i = idx[b];
                float p = mlp.forward(xs0[i], xs1[i], xs2[i]);
                float eps = 1e-6f;
                loss_acc -= ys[i] * std::log(p + eps) + (1.0f - ys[i]) * std::log(1.0f - p + eps);
                mlp.backprop(xs0[i], xs1[i], xs2[i], ys[i], grads);
            }
            for (int k = 0; k < MLP_NW; k++) mlp.w[k] -= (lr / n) * grads[k];
        }
        final_loss = static_cast<float>(loss_acc / N);
    }
    return final_loss;
}

// -------------------------------------------------------------------------
// PF kernels
// -------------------------------------------------------------------------
__global__ void init_rng(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void motion_update_kernel(float* part_x, float* part_y,
                                     float dvx, float dvy, curandState* rng) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    curandState s = rng[i];
    float nx = part_x[i] + dvx + MOTION_SIGMA * curand_normal(&s);
    float ny = part_y[i] + dvy + MOTION_SIGMA * curand_normal(&s);
    if (nx < 0.0f) nx = 0.0f; if (nx > WORLD_X) nx = WORLD_X;
    if (ny < 0.0f) ny = 0.0f; if (ny > WORLD_Y) ny = WORLD_Y;
    part_x[i] = nx; part_y[i] = ny;
    rng[i] = s;
}

__device__ float min_lm_distance(float px, float py, const float* lm_x,
                                 const float* lm_y, int n_lm) {
    float best = 1.0e9f;
    for (int j = 0; j < n_lm; j++) {
        float dx = px - lm_x[j], dy = py - lm_y[j];
        float d2 = dx * dx + dy * dy;
        if (d2 < best) best = d2;
    }
    return sqrtf(best);
}

__global__ void weight_gaussian_kernel(const float* part_x, const float* part_y,
                                       const float* lm_x, const float* lm_y, int n_lm,
                                       float z_obs, unsigned char dropout,
                                       float* weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    if (dropout) { weights[i] = 1.0f; return; }
    float expected = min_lm_distance(part_x[i], part_y[i], lm_x, lm_y, n_lm);
    float r = (expected - z_obs) / SIGMA;
    weights[i] = expf(-0.5f * r * r);
}

__global__ void weight_cauchy_kernel(const float* part_x, const float* part_y,
                                     const float* lm_x, const float* lm_y, int n_lm,
                                     float z_obs, unsigned char dropout,
                                     float* weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    if (dropout) { weights[i] = 1.0f; return; }
    float expected = min_lm_distance(part_x[i], part_y[i], lm_x, lm_y, n_lm);
    float r = (expected - z_obs) / SIGMA;
    // Cauchy density = 1 / (1 + r^2) (un-normalised)
    weights[i] = 1.0f / (1.0f + r * r);
}

__global__ void weight_learned_kernel(const float* part_x, const float* part_y,
                                      const float* lm_x, const float* lm_y, int n_lm,
                                      float z_obs, unsigned char dropout,
                                      float* weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    float expected = min_lm_distance(part_x[i], part_y[i], lm_x, lm_y, n_lm);
    float in_obs = dropout ? -1.0f : z_obs;
    float in_drop = dropout ? 1.0f : 0.0f;
    float p = mlp_forward(expected, in_obs, in_drop);
    weights[i] = p + 1.0e-6f;
}

__global__ void normalise_kernel(float* weights) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float s = 0.0f;
    for (int i = 0; i < K_PART; i++) s += weights[i];
    if (s < 1.0e-30f) { for (int i = 0; i < K_PART; i++) weights[i] = 1.0f / K_PART; return; }
    float inv = 1.0f / s;
    for (int i = 0; i < K_PART; i++) weights[i] *= inv;
}

__global__ void resample_kernel(const float* weights,
                                const float* part_x_in, const float* part_y_in,
                                float* part_x_out, float* part_y_out,
                                curandState* rng) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    curandState s = rng[i];
    float r = curand_uniform(&s);
    float acc = 0.0f;
    int chosen = K_PART - 1;
    for (int j = 0; j < K_PART; j++) {
        acc += weights[j];
        if (acc >= r) { chosen = j; break; }
    }
    part_x_out[i] = part_x_in[chosen];
    part_y_out[i] = part_y_in[chosen];
    rng[i] = s;
}

__global__ void mean_kernel(const float* part_x, const float* part_y, float* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float mx = 0.0f, my = 0.0f;
    for (int i = 0; i < K_PART; i++) { mx += part_x[i]; my += part_y[i]; }
    out[0] = mx / K_PART;
    out[1] = my / K_PART;
}

// -------------------------------------------------------------------------
// Render
// -------------------------------------------------------------------------
static void draw_world(cv::Mat& img, const std::vector<float>& lm_x,
                       const std::vector<float>& lm_y) {
    auto X = [&](float x) { return static_cast<int>(x / WORLD_X * img.cols); };
    auto Y = [&](float y) { return static_cast<int>((1.0f - y / WORLD_Y) * img.rows); };
    for (size_t i = 0; i < lm_x.size(); i++) {
        cv::circle(img, cv::Point(X(lm_x[i]), Y(lm_y[i])), 5,
                   cv::Scalar(120, 200, 255), cv::FILLED);
    }
}

static void draw_pf(cv::Mat& img, const std::vector<float>& part_x,
                    const std::vector<float>& part_y, cv::Scalar color) {
    auto X = [&](float x) { return static_cast<int>(x / WORLD_X * img.cols); };
    auto Y = [&](float y) { return static_cast<int>((1.0f - y / WORLD_Y) * img.rows); };
    for (size_t i = 0; i < part_x.size(); i += 4) {
        int x = X(part_x[i]), y = Y(part_y[i]);
        if (x >= 0 && x < img.cols && y >= 0 && y < img.rows) {
            img.at<cv::Vec3b>(y, x) = cv::Vec3b((uchar)color[0],
                                                (uchar)color[1],
                                                (uchar)color[2]);
        }
    }
}

static void draw_pose(cv::Mat& img, float wx, float wy, cv::Scalar color, int r) {
    int sx = static_cast<int>(wx / WORLD_X * img.cols);
    int sy = static_cast<int>((1.0f - wy / WORLD_Y) * img.rows);
    cv::circle(img, cv::Point(sx, sy), r, color, 2);
}

static void convert_avi_to_gif(const char* avi, const char* gif, int fps) {
    char cmd[512];
    std::snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -i %s -vf 'fps=%d,scale=1500:-1:flags=lanczos' -loop 0 %s "
        "> /dev/null 2>&1", avi, fps, gif);
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg conversion returned %d\n", rc);
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------
int main() {
    std::mt19937 rng(2026);
    std::vector<float> lm_x, lm_y;
    std::uniform_real_distribution<float> ux(2.0f, WORLD_X - 2.0f);
    std::uniform_real_distribution<float> uy(2.0f, WORLD_Y - 2.0f);
    for (int i = 0; i < N_LM; i++) { lm_x.push_back(ux(rng)); lm_y.push_back(uy(rng)); }

    // Generate training set for MLP
    int N_TRAIN = 16000;
    std::vector<float> tr_e, tr_o, tr_d, tr_y;
    tr_e.reserve(N_TRAIN); tr_o.reserve(N_TRAIN); tr_d.reserve(N_TRAIN); tr_y.reserve(N_TRAIN);
    std::uniform_real_distribution<float> uworld_x(0.0f, WORLD_X);
    std::uniform_real_distribution<float> uworld_y(0.0f, WORLD_Y);
    std::uniform_real_distribution<float> uoff(-3.0f, 3.0f);
    for (int i = 0; i < N_TRAIN; i++) {
        float px = uworld_x(rng), py = uworld_y(rng);
        // sensor at this pose
        float z = sensor_realistic(px, py, lm_x.data(), lm_y.data(), N_LM, rng);
        bool dropout = (z < 0.0f);
        if (i % 2 == 0) {
            // positive sample: query at the same pose
            float expected = 1.0e9f;
            for (int j = 0; j < N_LM; j++) {
                float dx = px - lm_x[j], dy = py - lm_y[j];
                float d = std::sqrt(dx * dx + dy * dy);
                if (d < expected) expected = d;
            }
            tr_e.push_back(expected);
            tr_o.push_back(dropout ? -1.0f : z);
            tr_d.push_back(dropout ? 1.0f : 0.0f);
            tr_y.push_back(1.0f);
        } else {
            // negative sample: query at offset pose
            float qx = px + uoff(rng), qy = py + uoff(rng);
            if (qx < 0) qx = 0; if (qx > WORLD_X) qx = WORLD_X;
            if (qy < 0) qy = 0; if (qy > WORLD_Y) qy = WORLD_Y;
            float expected = 1.0e9f;
            for (int j = 0; j < N_LM; j++) {
                float dx = qx - lm_x[j], dy = qy - lm_y[j];
                float d = std::sqrt(dx * dx + dy * dy);
                if (d < expected) expected = d;
            }
            tr_e.push_back(expected);
            tr_o.push_back(dropout ? -1.0f : z);
            tr_d.push_back(dropout ? 1.0f : 0.0f);
            tr_y.push_back(0.0f);
        }
    }
    MLP mlp;
    mlp.init_random(7);
    auto trt0 = std::chrono::high_resolution_clock::now();
    float final_loss = train_mlp(mlp, tr_e, tr_o, tr_d, tr_y, 20, 0.03f);
    auto trt1 = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration<double, std::milli>(trt1 - trt0).count();
    std::printf("MLP trained: loss=%.4f (%d samples, 20 epochs, %.1f ms total)\n",
                final_loss, N_TRAIN, train_ms);
    CUDA_CHECK(cudaMemcpyToSymbol(c_mlp_w, mlp.w, MLP_NW * sizeof(float)));

    // GPU allocations: 3 PFs
    auto alloc_pf = [&]() {
        struct PF { float* x; float* y; float* w; curandState* rng; float* mean; };
        PF p;
        CUDA_CHECK(cudaMalloc(&p.x, K_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&p.y, K_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&p.w, K_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&p.rng, K_PART * sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&p.mean, 2 * sizeof(float)));
        return p;
    };
    auto pf_g = alloc_pf();
    auto pf_c = alloc_pf();
    auto pf_m = alloc_pf();
    float* d_lm_x; float* d_lm_y;
    CUDA_CHECK(cudaMalloc(&d_lm_x, N_LM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_y, N_LM * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lm_x, lm_x.data(), N_LM * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lm_y, lm_y.data(), N_LM * sizeof(float),
                          cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (K_PART + threads - 1) / threads;
    init_rng<<<blocks, threads>>>(pf_g.rng, K_PART, 11ULL);
    init_rng<<<blocks, threads>>>(pf_c.rng, K_PART, 22ULL);
    init_rng<<<blocks, threads>>>(pf_m.rng, K_PART, 33ULL);

    // init particles uniformly
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> px0(K_PART), py0(K_PART);
    for (int i = 0; i < K_PART; i++) { px0[i] = u(rng) * WORLD_X; py0[i] = u(rng) * WORLD_Y; }
    for (auto* p : {&pf_g, &pf_c, &pf_m}) {
        CUDA_CHECK(cudaMemcpy(p->x, px0.data(), K_PART * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(p->y, py0.data(), K_PART * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    float* d_tmp_x; float* d_tmp_y;
    CUDA_CHECK(cudaMalloc(&d_tmp_x, K_PART * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp_y, K_PART * sizeof(float)));

    cv::VideoWriter video("gif/pf_realistic_obs.avi",
                          cv::VideoWriter::fourcc('X','V','I','D'), 15,
                          cv::Size(PANEL_W * 3 + 8, PANEL_H + 30));

    float true_x = 4.0f, true_y = 4.0f;
    float prev_x = true_x, prev_y = true_y;
    std::vector<float> hpx(K_PART), hpy(K_PART);
    float g_mean[2], c_mean[2], m_mean[2];

    double rmse_g = 0.0, rmse_c = 0.0, rmse_m = 0.0;
    int counted = 0;

    for (int step = 0; step < N_STEPS; step++) {
        float t = step * DT;
        float tx = 6.0f + 14.0f * std::cos(0.05f * t);
        float ty = 6.0f + 8.0f * std::sin(0.1f * t);
        float dx = tx - prev_x, dy = ty - prev_y;
        prev_x = true_x = tx; prev_y = true_y = ty;

        float z = sensor_realistic(true_x, true_y, lm_x.data(), lm_y.data(),
                                   N_LM, rng);
        bool drop = (z < 0.0f);
        unsigned char dropf = drop ? 1u : 0u;

        // Motion update + weight + normalise + resample (for each PF)
        for (auto* pp : {&pf_g, &pf_c, &pf_m}) {
            motion_update_kernel<<<blocks, threads>>>(pp->x, pp->y, dx, dy, pp->rng);
        }
        weight_gaussian_kernel<<<blocks, threads>>>(pf_g.x, pf_g.y, d_lm_x, d_lm_y,
                                                    N_LM, z, dropf, pf_g.w);
        weight_cauchy_kernel<<<blocks, threads>>>(pf_c.x, pf_c.y, d_lm_x, d_lm_y,
                                                  N_LM, z, dropf, pf_c.w);
        weight_learned_kernel<<<blocks, threads>>>(pf_m.x, pf_m.y, d_lm_x, d_lm_y,
                                                    N_LM, z, dropf, pf_m.w);
        normalise_kernel<<<1, 1>>>(pf_g.w);
        normalise_kernel<<<1, 1>>>(pf_c.w);
        normalise_kernel<<<1, 1>>>(pf_m.w);
        resample_kernel<<<blocks, threads>>>(pf_g.w, pf_g.x, pf_g.y, d_tmp_x, d_tmp_y, pf_g.rng);
        std::swap(pf_g.x, d_tmp_x); std::swap(pf_g.y, d_tmp_y);
        resample_kernel<<<blocks, threads>>>(pf_c.w, pf_c.x, pf_c.y, d_tmp_x, d_tmp_y, pf_c.rng);
        std::swap(pf_c.x, d_tmp_x); std::swap(pf_c.y, d_tmp_y);
        resample_kernel<<<blocks, threads>>>(pf_m.w, pf_m.x, pf_m.y, d_tmp_x, d_tmp_y, pf_m.rng);
        std::swap(pf_m.x, d_tmp_x); std::swap(pf_m.y, d_tmp_y);
        mean_kernel<<<1, 1>>>(pf_g.x, pf_g.y, pf_g.mean);
        mean_kernel<<<1, 1>>>(pf_c.x, pf_c.y, pf_c.mean);
        mean_kernel<<<1, 1>>>(pf_m.x, pf_m.y, pf_m.mean);
        CUDA_CHECK(cudaMemcpy(g_mean, pf_g.mean, 2 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(c_mean, pf_c.mean, 2 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(m_mean, pf_m.mean, 2 * sizeof(float),
                              cudaMemcpyDeviceToHost));

        if (step >= 20) {
            float eg = std::hypot(g_mean[0] - true_x, g_mean[1] - true_y);
            float ec = std::hypot(c_mean[0] - true_x, c_mean[1] - true_y);
            float em = std::hypot(m_mean[0] - true_x, m_mean[1] - true_y);
            rmse_g += eg * eg; rmse_c += ec * ec; rmse_m += em * em; counted++;
        }

        // render panels
        auto render = [&](float* d_x, float* d_y, float* mean, const char* name,
                          cv::Scalar pcol) {
            CUDA_CHECK(cudaMemcpy(hpx.data(), d_x, K_PART * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hpy.data(), d_y, K_PART * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            cv::Mat panel(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(20, 20, 20));
            draw_world(panel, lm_x, lm_y);
            draw_pf(panel, hpx, hpy, pcol);
            draw_pose(panel, true_x, true_y, cv::Scalar(255, 255, 255), 8);
            draw_pose(panel, mean[0], mean[1], cv::Scalar(0, 255, 255), 5);
            cv::rectangle(panel, cv::Rect(0, 0, PANEL_W, 26), cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(panel, name, cv::Point(10, 18), cv::FONT_HERSHEY_SIMPLEX,
                        0.52, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            return panel;
        };
        cv::Mat pg = render(pf_g.x, pf_g.y, g_mean, "Gaussian (handcrafted)",  cv::Scalar(80, 220, 80));
        cv::Mat pc = render(pf_c.x, pf_c.y, c_mean, "Cauchy (heavy-tailed)",   cv::Scalar(220, 220, 60));
        cv::Mat pm = render(pf_m.x, pf_m.y, m_mean, "Learned MLP (realistic)", cv::Scalar(60, 130, 240));

        cv::Mat frame(PANEL_H + 30, PANEL_W * 3 + 8, CV_8UC3, cv::Scalar(30, 30, 30));
        pg.copyTo(frame(cv::Rect(0, 30, PANEL_W, PANEL_H)));
        pc.copyTo(frame(cv::Rect(PANEL_W + 4, 30, PANEL_W, PANEL_H)));
        pm.copyTo(frame(cv::Rect(PANEL_W * 2 + 8, 30, PANEL_W, PANEL_H)));
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "step=%d  z=%.2f m %s  (sensor: noise + dropout %.0f%% + multi-path %.0f%%)",
                      step, z, drop ? "(DROP)" : "", DROPOUT_PROB * 100.0f,
                      MULTIPATH_PROB * 100.0f);
        cv::putText(frame, buf, cv::Point(12, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        video.write(frame);
    }
    video.release();
    convert_avi_to_gif("gif/pf_realistic_obs.avi", "gif/pf_realistic_obs.gif", 15);

    if (counted > 0) {
        std::printf("RMSE  gaussian: %.3f m   cauchy: %.3f m   learned: %.3f m\n",
                    std::sqrt(rmse_g / counted),
                    std::sqrt(rmse_c / counted),
                    std::sqrt(rmse_m / counted));
    }
    std::printf("GIF saved to gif/pf_realistic_obs.gif\n");

    CUDA_CHECK(cudaFree(d_lm_x));
    CUDA_CHECK(cudaFree(d_lm_y));
    CUDA_CHECK(cudaFree(d_tmp_x));
    CUDA_CHECK(cudaFree(d_tmp_y));
    for (auto pp : {pf_g, pf_c, pf_m}) {
        for (auto* p : {pp.x, pp.y, pp.w, pp.mean}) CUDA_CHECK(cudaFree(p));
        CUDA_CHECK(cudaFree(pp.rng));
    }
    return 0;
}
