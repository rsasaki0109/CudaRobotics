/*************************************************************************
    > File Name: comparison_swarm.cu
    > Run PSO, DE, CMA-ES on Rastrigin D=30, plot convergence curves
    > on same graph. 800x600 output.
    > Output: gif/comparison_swarm.gif
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "benchmark_functions.cuh"

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------
static const int DIM = 30;
static const float X_MIN = -5.12f;
static const float X_MAX = 5.12f;
static const int MAX_ITER = 1000;

// PSO parameters
static const int PSO_N = 100000;
static const float W_START = 0.9f;
static const float W_END = 0.4f;
static const float C1 = 2.0f;
static const float C2 = 2.0f;
static const float V_MAX = 2.0f;

// DE parameters
static const int DE_N = 10000;
static const float DE_F = 0.8f;
static const float DE_CR = 0.9f;

// CMA-ES parameters
static const int CMA_LAMBDA = 4096;
static const int CMA_MU = CMA_LAMBDA / 2;
static const int CMA_MAX_ITER = 500;

// =========================================================================
// PSO Kernels
// =========================================================================
__global__ void pso_init_curand(curandState* states, int N, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curand_init(seed, i, 0, &states[i]);
}

__global__ void pso_init_kernel(
    float* positions, float* velocities,
    float* p_best_pos, float* p_best_fit,
    curandState* rng, int N, int D, float x_min, float x_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curandState ls = rng[i];
    for (int d = 0; d < D; d++) {
        float pos = x_min + curand_uniform(&ls) * (x_max - x_min);
        positions[i * D + d] = pos;
        p_best_pos[i * D + d] = pos;
        velocities[i * D + d] = (curand_uniform(&ls) - 0.5f) * 2.0f;
    }
    p_best_fit[i] = FLT_MAX;
    rng[i] = ls;
}

__global__ void pso_evaluate_kernel(const float* positions, float* fitness, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    fitness[i] = rastrigin(&positions[i * D], D);
}

__global__ void pso_update_pbest_kernel(
    const float* positions, const float* fitness,
    float* p_best_pos, float* p_best_fit, int N, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (fitness[i] < p_best_fit[i]) {
        p_best_fit[i] = fitness[i];
        for (int d = 0; d < D; d++)
            p_best_pos[i * D + d] = positions[i * D + d];
    }
}

__global__ void pso_find_min_kernel(
    const float* fitness, const float* positions,
    float* g_best_fit, float* g_best_pos, int N, int D)
{
    __shared__ float s_fit[256];
    __shared__ int s_idx[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s_fit[tid] = (i < N) ? fitness[i] : FLT_MAX;
    s_idx[tid] = i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_fit[tid + s] < s_fit[tid]) {
            s_fit[tid] = s_fit[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        int* g_fit_int = (int*)g_best_fit;
        int old_val = *g_fit_int;
        int new_val = __float_as_int(s_fit[0]);
        while (s_fit[0] < __int_as_float(old_val)) {
            int assumed = old_val;
            old_val = atomicCAS(g_fit_int, assumed, new_val);
            if (old_val == assumed) {
                for (int d = 0; d < D; d++)
                    g_best_pos[d] = positions[s_idx[0] * D + d];
                break;
            }
        }
    }
}

__global__ void pso_update_kernel(
    float* positions, float* velocities,
    const float* p_best_pos, const float* g_best_pos,
    float w, float c1, float c2,
    curandState* rng, int N, int D,
    float x_min, float x_max, float v_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curandState ls = rng[i];
    for (int d = 0; d < D; d++) {
        float r1 = curand_uniform(&ls);
        float r2 = curand_uniform(&ls);
        int idx = i * D + d;
        float v = w * velocities[idx]
                + c1 * r1 * (p_best_pos[idx] - positions[idx])
                + c2 * r2 * (g_best_pos[d] - positions[idx]);
        v = fminf(fmaxf(v, -v_max), v_max);
        velocities[idx] = v;
        float pos = fminf(fmaxf(positions[idx] + v, x_min), x_max);
        positions[idx] = pos;
    }
    rng[i] = ls;
}

// =========================================================================
// DE Kernels
// =========================================================================
__global__ void de_init_kernel(float* pop, curandState* rng, int N, int D, float x_min, float x_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curandState ls = rng[i];
    for (int d = 0; d < D; d++)
        pop[i * D + d] = x_min + curand_uniform(&ls) * (x_max - x_min);
    rng[i] = ls;
}

__global__ void de_evaluate_kernel(const float* pop, float* fitness, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    fitness[i] = rastrigin(&pop[i * D], D);
}

__global__ void de_mutate_crossover_kernel(
    const float* pop, const float* fitness,
    float* trial, float* trial_fitness,
    curandState* rng, int N, int D,
    float F_val, float CR_val, float x_min, float x_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curandState ls = rng[i];

    int r1, r2, r3;
    do { r1 = (int)(curand_uniform(&ls) * N) % N; } while (r1 == i);
    do { r2 = (int)(curand_uniform(&ls) * N) % N; } while (r2 == i || r2 == r1);
    do { r3 = (int)(curand_uniform(&ls) * N) % N; } while (r3 == i || r3 == r1 || r3 == r2);

    int j_rand = (int)(curand_uniform(&ls) * D) % D;
    for (int d = 0; d < D; d++) {
        if (curand_uniform(&ls) < CR_val || d == j_rand) {
            float m = pop[r1 * D + d] + F_val * (pop[r2 * D + d] - pop[r3 * D + d]);
            trial[i * D + d] = fminf(fmaxf(m, x_min), x_max);
        } else {
            trial[i * D + d] = pop[i * D + d];
        }
    }
    trial_fitness[i] = rastrigin(&trial[i * D], D);
    rng[i] = ls;
}

__global__ void de_selection_kernel(
    float* pop, float* fitness, const float* trial, const float* trial_fitness, int N, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (trial_fitness[i] <= fitness[i]) {
        fitness[i] = trial_fitness[i];
        for (int d = 0; d < D; d++)
            pop[i * D + d] = trial[i * D + d];
    }
}

__global__ void find_min_fitness_kernel(const float* fitness, float* g_best, int N) {
    __shared__ float s_fit[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s_fit[tid] = (i < N) ? fitness[i] : FLT_MAX;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_fit[tid + s] < s_fit[tid])
            s_fit[tid] = s_fit[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        int* g_int = (int*)g_best;
        int old_val = *g_int;
        int new_val = __float_as_int(s_fit[0]);
        while (s_fit[0] < __int_as_float(old_val)) {
            int assumed = old_val;
            old_val = atomicCAS(g_int, assumed, new_val);
            if (old_val == assumed) break;
        }
    }
}

// =========================================================================
// CMA-ES Kernels
// =========================================================================
__global__ void cma_sample_kernel(
    float* samples, const float* mean, const float* chol_L,
    float sigma, curandState* rng, int N, int D, float x_min, float x_max)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    curandState ls = rng[k];
    float z[30];
    for (int d = 0; d < D; d++) z[d] = curand_normal(&ls);
    for (int d = 0; d < D; d++) {
        float Lz = 0.0f;
        for (int j = 0; j <= d; j++) Lz += chol_L[d * D + j] * z[j];
        float val = mean[d] + sigma * Lz;
        samples[k * D + d] = fminf(fmaxf(val, x_min), x_max);
    }
    rng[k] = ls;
}

__global__ void cma_evaluate_kernel(const float* samples, float* fitness, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    fitness[i] = rastrigin(&samples[i * D], D);
}

// =========================================================================
// CPU CMA-ES helpers
// =========================================================================
void cholesky_decompose(const vector<float>& C, vector<float>& L, int D) {
    fill(L.begin(), L.end(), 0.0f);
    for (int i = 0; i < D; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) sum += L[i * D + k] * L[j * D + k];
            if (i == j) {
                float val = C[i * D + i] - sum;
                L[i * D + j] = (val > 0.0f) ? sqrtf(val) : 1e-6f;
            } else {
                L[i * D + j] = (C[i * D + j] - sum) / L[j * D + j];
            }
        }
    }
}

// =========================================================================
// Visualization
// =========================================================================
static const int VIS_W = 800;
static const int VIS_H = 600;

void draw_convergence_graph(cv::Mat& img,
    const vector<float>& pso_hist,
    const vector<float>& de_hist,
    const vector<float>& cma_hist,
    int current_iter)
{
    img = cv::Scalar(30, 30, 30);

    int margin_l = 80, margin_r = 20, margin_t = 50, margin_b = 60;
    int plot_w = VIS_W - margin_l - margin_r;
    int plot_h = VIS_H - margin_t - margin_b;

    // Find global range across all histories
    float global_max = 1.0f, global_min = 1e6f;
    for (auto& h : {pso_hist, de_hist, cma_hist}) {
        for (float v : h) {
            global_max = fmaxf(global_max, v);
            global_min = fminf(global_min, v);
        }
    }

    float log_max = logf(global_max + 1.0f);
    float log_min = logf(fmaxf(global_min, 1e-8f) + 1.0f);
    if (log_max <= log_min) log_max = log_min + 1.0f;

    int total_iters = MAX_ITER;

    // Draw grid
    cv::rectangle(img, cv::Point(margin_l, margin_t),
                  cv::Point(margin_l + plot_w, margin_t + plot_h), cv::Scalar(50, 50, 50), 1);

    // Y-axis grid lines
    for (int i = 0; i <= 5; i++) {
        int y = margin_t + plot_h * i / 5;
        cv::line(img, cv::Point(margin_l, y), cv::Point(margin_l + plot_w, y),
                 cv::Scalar(60, 60, 60), 1);
        float lv = log_max - (log_max - log_min) * i / 5.0f;
        float val = expf(lv) - 1.0f;
        char buf[32];
        if (val >= 100.0f) snprintf(buf, sizeof(buf), "%.0f", val);
        else if (val >= 1.0f) snprintf(buf, sizeof(buf), "%.1f", val);
        else snprintf(buf, sizeof(buf), "%.3f", val);
        cv::putText(img, buf, cv::Point(5, y + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200));
    }

    // X-axis labels
    for (int i = 0; i <= 5; i++) {
        int x = margin_l + plot_w * i / 5;
        int iter_val = total_iters * i / 5;
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", iter_val);
        cv::putText(img, buf, cv::Point(x - 10, margin_t + plot_h + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200));
    }

    // Axis labels
    cv::putText(img, "Iteration", cv::Point(VIS_W / 2 - 30, VIS_H - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200));
    cv::putText(img, "Best Fitness (log)", cv::Point(5, margin_t - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200));

    // Title
    cv::putText(img, "Swarm Optimization Comparison on Rastrigin D=30",
                cv::Point(VIS_W / 2 - 200, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255));

    // Draw curves
    auto draw_curve = [&](const vector<float>& hist, cv::Scalar color, int max_iter) {
        if (hist.empty()) return;
        cv::Point prev;
        for (int i = 0; i < (int)hist.size(); i++) {
            float lv = logf(fmaxf(hist[i], 1e-8f) + 1.0f);
            int px = margin_l + (int)(plot_w * (float)i / max(max_iter - 1, 1));
            int py = margin_t + plot_h - (int)(plot_h * (lv - log_min) / (log_max - log_min));
            py = max(margin_t, min(py, margin_t + plot_h));
            cv::Point pt(px, py);
            if (i > 0) cv::line(img, prev, pt, color, 2);
            prev = pt;
        }
    };

    draw_curve(pso_hist, cv::Scalar(0, 200, 255), MAX_ITER);    // Yellow-orange
    draw_curve(de_hist, cv::Scalar(0, 255, 128), MAX_ITER);     // Green
    draw_curve(cma_hist, cv::Scalar(255, 128, 0), CMA_MAX_ITER); // Blue-ish

    // Legend
    int ly = margin_t + 15;
    cv::line(img, cv::Point(margin_l + 10, ly), cv::Point(margin_l + 40, ly),
             cv::Scalar(0, 200, 255), 2);
    cv::putText(img, "PSO (N=100K)", cv::Point(margin_l + 45, ly + 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 200, 255));

    ly += 20;
    cv::line(img, cv::Point(margin_l + 10, ly), cv::Point(margin_l + 40, ly),
             cv::Scalar(0, 255, 128), 2);
    cv::putText(img, "DE (N=10K)", cv::Point(margin_l + 45, ly + 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 128));

    ly += 20;
    cv::line(img, cv::Point(margin_l + 10, ly), cv::Point(margin_l + 40, ly),
             cv::Scalar(255, 128, 0), 2);
    cv::putText(img, "CMA-ES (Lambda=4096)", cv::Point(margin_l + 45, ly + 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 128, 0));

    // Current values
    char buf[256];
    ly = VIS_H - margin_b + 35;
    float pso_val = pso_hist.empty() ? 0 : pso_hist.back();
    float de_val = de_hist.empty() ? 0 : de_hist.back();
    float cma_val = cma_hist.empty() ? 0 : cma_hist.back();
    snprintf(buf, sizeof(buf), "PSO: %.4f  |  DE: %.4f  |  CMA-ES: %.4f",
             pso_val, de_val, cma_val);
    cv::putText(img, buf, cv::Point(margin_l, ly),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255));
}

// =========================================================================
// Main
// =========================================================================
int main() {
    cout << "=== Swarm Optimization Comparison on Rastrigin D=30 ===" << endl;

    const int D = DIM;
    const int threads = 256;

    // ===================== PSO Setup =====================
    const int pso_blocks = (PSO_N + threads - 1) / threads;
    float *d_pso_pos, *d_pso_vel, *d_pso_fit;
    float *d_pso_pb_pos, *d_pso_pb_fit;
    float *d_pso_gb_pos, *d_pso_gb_fit;
    curandState *d_pso_rng;

    CUDA_CHECK(cudaMalloc(&d_pso_pos, PSO_N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pso_vel, PSO_N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pso_fit, PSO_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pso_pb_pos, PSO_N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pso_pb_fit, PSO_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pso_gb_pos, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pso_gb_fit, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pso_rng, PSO_N * sizeof(curandState)));

    pso_init_curand<<<pso_blocks, threads>>>(d_pso_rng, PSO_N, 42);
    CUDA_CHECK(cudaDeviceSynchronize());
    pso_init_kernel<<<pso_blocks, threads>>>(
        d_pso_pos, d_pso_vel, d_pso_pb_pos, d_pso_pb_fit,
        d_pso_rng, PSO_N, D, X_MIN, X_MAX);
    CUDA_CHECK(cudaDeviceSynchronize());
    float init_fit = FLT_MAX;
    CUDA_CHECK(cudaMemcpy(d_pso_gb_fit, &init_fit, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_pso_gb_pos, 0, D * sizeof(float)));

    // ===================== DE Setup =====================
    const int de_blocks = (DE_N + threads - 1) / threads;
    float *d_de_pop, *d_de_fit, *d_de_trial, *d_de_trial_fit, *d_de_gb;
    curandState *d_de_rng;

    CUDA_CHECK(cudaMalloc(&d_de_pop, DE_N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_de_fit, DE_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_de_trial, DE_N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_de_trial_fit, DE_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_de_gb, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_de_rng, DE_N * sizeof(curandState)));

    pso_init_curand<<<de_blocks, threads>>>(d_de_rng, DE_N, 123);
    CUDA_CHECK(cudaDeviceSynchronize());
    de_init_kernel<<<de_blocks, threads>>>(d_de_pop, d_de_rng, DE_N, D, X_MIN, X_MAX);
    CUDA_CHECK(cudaDeviceSynchronize());
    de_evaluate_kernel<<<de_blocks, threads>>>(d_de_pop, d_de_fit, DE_N, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ===================== CMA-ES Setup =====================
    const int cma_blocks = (CMA_LAMBDA + threads - 1) / threads;
    float *d_cma_samples, *d_cma_fit, *d_cma_mean, *d_cma_chol;
    curandState *d_cma_rng;

    CUDA_CHECK(cudaMalloc(&d_cma_samples, CMA_LAMBDA * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cma_fit, CMA_LAMBDA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cma_mean, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cma_chol, D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cma_rng, CMA_LAMBDA * sizeof(curandState)));

    pso_init_curand<<<cma_blocks, threads>>>(d_cma_rng, CMA_LAMBDA, 77);
    CUDA_CHECK(cudaDeviceSynchronize());

    // CMA-ES host state
    vector<float> cma_mean(D);
    srand(42);
    for (int d = 0; d < D; d++)
        cma_mean[d] = X_MIN + ((float)rand() / RAND_MAX) * (X_MAX - X_MIN);
    float cma_sigma = 2.0f;
    vector<float> cma_C(D * D, 0.0f);
    for (int d = 0; d < D; d++) cma_C[d * D + d] = 1.0f;
    vector<float> cma_L(D * D);
    cholesky_decompose(cma_C, cma_L, D);
    vector<float> cma_p_sigma(D, 0.0f);
    vector<float> cma_p_c(D, 0.0f);

    // CMA-ES weights
    vector<float> cma_weights(CMA_MU);
    float cma_mu_eff = 0.0f;
    {
        float sw = 0.0f;
        for (int i = 0; i < CMA_MU; i++) {
            cma_weights[i] = logf((float)(CMA_MU + 1)) - logf((float)(i + 1));
            sw += cma_weights[i];
        }
        float sw2 = 0.0f;
        for (int i = 0; i < CMA_MU; i++) { cma_weights[i] /= sw; sw2 += cma_weights[i] * cma_weights[i]; }
        cma_mu_eff = 1.0f / sw2;
    }
    float cma_c_sigma = (cma_mu_eff + 2.0f) / (D + cma_mu_eff + 5.0f);
    float cma_d_sigma = 1.0f + 2.0f * fmaxf(0.0f, sqrtf((cma_mu_eff - 1.0f) / (D + 1.0f)) - 1.0f) + cma_c_sigma;
    float cma_cc = (4.0f + cma_mu_eff / D) / (D + 4.0f + 2.0f * cma_mu_eff / D);
    float cma_c1 = 2.0f / ((D + 1.3f) * (D + 1.3f) + cma_mu_eff);
    float cma_c_mu = fminf(1.0f - cma_c1,
        2.0f * (cma_mu_eff - 2.0f + 1.0f / cma_mu_eff) / ((D + 2.0f) * (D + 2.0f) + cma_mu_eff));
    float cma_chi_n = sqrtf((float)D) * (1.0f - 1.0f / (4.0f * D) + 1.0f / (21.0f * D * D));

    CUDA_CHECK(cudaMemcpy(d_cma_mean, cma_mean.data(), D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cma_chol, cma_L.data(), D * D * sizeof(float), cudaMemcpyHostToDevice));

    vector<float> cma_h_samples(CMA_LAMBDA * D);
    vector<float> cma_h_fitness(CMA_LAMBDA);

    // ===================== Run all methods =====================
    vector<float> pso_history, de_history, cma_history;

    string avi_path = "gif/comparison_swarm.avi";
    string gif_path = "gif/comparison_swarm.gif";
    cv::VideoWriter video(avi_path, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 15,
                          cv::Size(VIS_W, VIS_H));

    auto t_start = chrono::high_resolution_clock::now();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // --- PSO step ---
        float w = W_START - (W_START - W_END) * (float)iter / (MAX_ITER - 1);
        pso_evaluate_kernel<<<pso_blocks, threads>>>(d_pso_pos, d_pso_fit, PSO_N, D);
        CUDA_CHECK(cudaDeviceSynchronize());
        pso_update_pbest_kernel<<<pso_blocks, threads>>>(
            d_pso_pos, d_pso_fit, d_pso_pb_pos, d_pso_pb_fit, PSO_N, D);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(d_pso_gb_fit, &init_fit, sizeof(float), cudaMemcpyHostToDevice));
        pso_find_min_kernel<<<pso_blocks, threads>>>(
            d_pso_pb_fit, d_pso_pb_pos, d_pso_gb_fit, d_pso_gb_pos, PSO_N, D);
        CUDA_CHECK(cudaDeviceSynchronize());
        float pso_best;
        CUDA_CHECK(cudaMemcpy(&pso_best, d_pso_gb_fit, sizeof(float), cudaMemcpyDeviceToHost));
        pso_history.push_back(pso_best);
        pso_update_kernel<<<pso_blocks, threads>>>(
            d_pso_pos, d_pso_vel, d_pso_pb_pos, d_pso_gb_pos,
            w, C1, C2, d_pso_rng, PSO_N, D, X_MIN, X_MAX, V_MAX);
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- DE step ---
        de_mutate_crossover_kernel<<<de_blocks, threads>>>(
            d_de_pop, d_de_fit, d_de_trial, d_de_trial_fit,
            d_de_rng, DE_N, D, DE_F, DE_CR, X_MIN, X_MAX);
        CUDA_CHECK(cudaDeviceSynchronize());
        de_selection_kernel<<<de_blocks, threads>>>(
            d_de_pop, d_de_fit, d_de_trial, d_de_trial_fit, DE_N, D);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(d_de_gb, &init_fit, sizeof(float), cudaMemcpyHostToDevice));
        find_min_fitness_kernel<<<de_blocks, threads>>>(d_de_fit, d_de_gb, DE_N);
        CUDA_CHECK(cudaDeviceSynchronize());
        float de_best;
        CUDA_CHECK(cudaMemcpy(&de_best, d_de_gb, sizeof(float), cudaMemcpyDeviceToHost));
        de_history.push_back(de_best);

        // --- CMA-ES step (500 iters only) ---
        if (iter < CMA_MAX_ITER) {
            cma_sample_kernel<<<cma_blocks, threads>>>(
                d_cma_samples, d_cma_mean, d_cma_chol, cma_sigma,
                d_cma_rng, CMA_LAMBDA, D, X_MIN, X_MAX);
            CUDA_CHECK(cudaDeviceSynchronize());
            cma_evaluate_kernel<<<cma_blocks, threads>>>(d_cma_samples, d_cma_fit, CMA_LAMBDA, D);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(cma_h_samples.data(), d_cma_samples,
                CMA_LAMBDA * D * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(cma_h_fitness.data(), d_cma_fit,
                CMA_LAMBDA * sizeof(float), cudaMemcpyDeviceToHost));

            vector<int> indices(CMA_LAMBDA);
            iota(indices.begin(), indices.end(), 0);
            sort(indices.begin(), indices.end(), [&](int a, int b) {
                return cma_h_fitness[a] < cma_h_fitness[b];
            });

            cma_history.push_back(cma_h_fitness[indices[0]]);

            // CMA-ES update (same as standalone)
            vector<float> old_mean = cma_mean;
            fill(cma_mean.begin(), cma_mean.end(), 0.0f);
            for (int i = 0; i < CMA_MU; i++) {
                int idx = indices[i];
                for (int d = 0; d < D; d++)
                    cma_mean[d] += cma_weights[i] * cma_h_samples[idx * D + d];
            }

            vector<float> mean_diff(D);
            for (int d = 0; d < D; d++)
                mean_diff[d] = (cma_mean[d] - old_mean[d]) / cma_sigma;

            vector<float> inv_L_md(D);
            for (int i = 0; i < D; i++) {
                float sum = mean_diff[i];
                for (int j = 0; j < i; j++) sum -= cma_L[i * D + j] * inv_L_md[j];
                inv_L_md[i] = sum / cma_L[i * D + i];
            }

            float sqrt_cs = sqrtf(cma_c_sigma * (2.0f - cma_c_sigma) * cma_mu_eff);
            for (int d = 0; d < D; d++)
                cma_p_sigma[d] = (1.0f - cma_c_sigma) * cma_p_sigma[d] + sqrt_cs * inv_L_md[d];

            float ps_norm = 0.0f;
            for (int d = 0; d < D; d++) ps_norm += cma_p_sigma[d] * cma_p_sigma[d];
            ps_norm = sqrtf(ps_norm);

            float thresh = (1.4f + 2.0f / (D + 1.0f)) * cma_chi_n *
                sqrtf(1.0f - powf(1.0f - cma_c_sigma, 2.0f * (iter + 1)));
            int h_sig = (ps_norm < thresh) ? 1 : 0;

            float sqrt_cc = sqrtf(cma_cc * (2.0f - cma_cc) * cma_mu_eff);
            for (int d = 0; d < D; d++)
                cma_p_c[d] = (1.0f - cma_cc) * cma_p_c[d] + h_sig * sqrt_cc * mean_diff[d];

            float delta_h = (1.0f - h_sig) * cma_cc * (2.0f - cma_cc);
            for (int i = 0; i < D; i++) {
                for (int j = 0; j <= i; j++) {
                    float r1 = cma_p_c[i] * cma_p_c[j];
                    float rmu = 0.0f;
                    for (int k = 0; k < CMA_MU; k++) {
                        int idx = indices[k];
                        float di = (cma_h_samples[idx * D + i] - old_mean[i]) / cma_sigma;
                        float dj = (cma_h_samples[idx * D + j] - old_mean[j]) / cma_sigma;
                        rmu += cma_weights[k] * di * dj;
                    }
                    cma_C[i * D + j] = (1.0f - cma_c1 - cma_c_mu + cma_c1 * delta_h) * cma_C[i * D + j]
                                      + cma_c1 * r1 + cma_c_mu * rmu;
                    cma_C[j * D + i] = cma_C[i * D + j];
                }
            }

            cma_sigma *= expf((cma_c_sigma / cma_d_sigma) * (ps_norm / cma_chi_n - 1.0f));
            cma_sigma = fminf(fmaxf(cma_sigma, 1e-10f), 10.0f);

            cholesky_decompose(cma_C, cma_L, D);
            CUDA_CHECK(cudaMemcpy(d_cma_mean, cma_mean.data(), D * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_cma_chol, cma_L.data(), D * D * sizeof(float), cudaMemcpyHostToDevice));
        }

        if (iter % 100 == 0) {
            printf("Iter %4d: PSO=%.4f  DE=%.4f  CMA-ES=%.4f\n",
                   iter, pso_history.back(), de_history.back(),
                   cma_history.empty() ? 0.0f : cma_history.back());
        }

        // Write frame every 10 iterations
        if (iter % 10 == 0) {
            cv::Mat frame(VIS_H, VIS_W, CV_8UC3);
            draw_convergence_graph(frame, pso_history, de_history, cma_history, iter);
            video.write(frame);
        }
    }

    auto t_end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(t_end - t_start).count();

    printf("\n=== Final Results (Rastrigin D=%d) ===\n", D);
    printf("PSO  (N=100K, 1000 iter): %.6f\n", pso_history.back());
    printf("DE   (N=10K,  1000 iter): %.6f\n", de_history.back());
    printf("CMA-ES (Lambda=4096, 500 iter): %.6f\n", cma_history.back());
    printf("Total time: %.3f s\n", elapsed);

    video.release();
    cout << "Video saved to " << avi_path << endl;

    string cmd = "ffmpeg -y -i " + avi_path + " -vf 'fps=15,scale=400:-1' -loop 0 " + gif_path + " 2>/dev/null";
    system(cmd.c_str());
    cout << "GIF saved to " << gif_path << endl;

    // Cleanup
    cudaFree(d_pso_pos); cudaFree(d_pso_vel); cudaFree(d_pso_fit);
    cudaFree(d_pso_pb_pos); cudaFree(d_pso_pb_fit);
    cudaFree(d_pso_gb_pos); cudaFree(d_pso_gb_fit); cudaFree(d_pso_rng);
    cudaFree(d_de_pop); cudaFree(d_de_fit);
    cudaFree(d_de_trial); cudaFree(d_de_trial_fit); cudaFree(d_de_gb); cudaFree(d_de_rng);
    cudaFree(d_cma_samples); cudaFree(d_cma_fit);
    cudaFree(d_cma_mean); cudaFree(d_cma_chol); cudaFree(d_cma_rng);

    return 0;
}
