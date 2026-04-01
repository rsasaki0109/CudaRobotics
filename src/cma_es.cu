/*************************************************************************
    > File Name: cma_es.cu
    > CUDA CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    > Lambda=4096 samples on GPU, mu=lambda/2
    > GPU: sample generation (multivariate normal via Cholesky) + fitness evaluation
    > CPU: covariance matrix update (D=30, 30x30 matrix)
    > 500 generations on Rastrigin function
    > Output: gif/cma_es.gif
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
static const int LAMBDA = 4096;
static const int DIM = 30;
static const int MU = LAMBDA / 2;
static const int MAX_GEN = 500;
static const float X_MIN = -5.12f;
static const float X_MAX = 5.12f;

// -------------------------------------------------------------------------
// Kernels
// -------------------------------------------------------------------------
__global__ void init_curand_kernel(curandState* states, int N, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curand_init(seed, i, 0, &states[i]);
}

// Generate samples: x_k = mean + sigma * L * z_k
// where L is Cholesky factor (lower triangular), z_k ~ N(0, I)
__global__ void cma_sample_kernel(
    float* samples,           // [LAMBDA * D]
    const float* mean,        // [D]
    const float* chol_L,      // [D * D] lower triangular
    float sigma,
    curandState* rng,
    int N, int D,
    float x_min, float x_max)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    curandState local_state = rng[k];

    // Generate z ~ N(0, I)
    float z[30];  // stack alloc for D=30
    for (int d = 0; d < D; d++)
        z[d] = curand_normal(&local_state);

    // x = mean + sigma * L * z
    for (int d = 0; d < D; d++) {
        float Lz = 0.0f;
        for (int j = 0; j <= d; j++)
            Lz += chol_L[d * D + j] * z[j];
        float val = mean[d] + sigma * Lz;
        val = fminf(fmaxf(val, x_min), x_max);
        samples[k * D + d] = val;
    }

    rng[k] = local_state;
}

__global__ void cma_evaluate_kernel(
    const float* samples, float* fitness,
    int N, int D, int func_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    fitness[i] = evaluate_benchmark(&samples[i * D], D, func_id);
}

// -------------------------------------------------------------------------
// CPU: Cholesky decomposition of covariance matrix
// -------------------------------------------------------------------------
void cholesky_decompose(const vector<float>& C, vector<float>& L, int D) {
    fill(L.begin(), L.end(), 0.0f);
    for (int i = 0; i < D; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++)
                sum += L[i * D + k] * L[j * D + k];
            if (i == j) {
                float val = C[i * D + i] - sum;
                L[i * D + j] = (val > 0.0f) ? sqrtf(val) : 1e-6f;
            } else {
                L[i * D + j] = (C[i * D + j] - sum) / L[j * D + j];
            }
        }
    }
}

// -------------------------------------------------------------------------
// Visualization
// -------------------------------------------------------------------------
static const int VIS_W = 800;
static const int VIS_H = 400;

void draw_rastrigin_landscape(cv::Mat& img, int w, int h, float view_min, float view_max) {
    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            float x = view_min + (view_max - view_min) * px / w;
            float y = view_min + (view_max - view_min) * py / h;
            float val = 20.0f + x*x + y*y - 10.0f*(cosf(2*M_PI*x) + cosf(2*M_PI*y));
            float norm = fminf(logf(val + 1.0f) / logf(100.0f), 1.0f);
            int b = (int)(255 * (1.0f - norm));
            int g = (int)(255 * (1.0f - norm) * 0.5f);
            int r = (int)(50 + 205 * norm);
            img.at<cv::Vec3b>(py, px) = cv::Vec3b(b, g, r);
        }
    }
}

void draw_convergence(cv::Mat& img, const vector<float>& history,
    int ox, int oy, int w, int h) {
    cv::rectangle(img, cv::Rect(ox, oy, w, h), cv::Scalar(40, 40, 40), -1);
    if (history.empty()) return;

    float max_val = *max_element(history.begin(), history.end());
    float min_val = *min_element(history.begin(), history.end());
    float log_max = logf(max_val + 1.0f);
    float log_min = logf(fmaxf(min_val, 1e-6f) + 1.0f);
    if (log_max <= log_min) log_max = log_min + 1.0f;

    int margin = 40;
    cv::line(img, cv::Point(ox + margin, oy + h - margin),
             cv::Point(ox + w - 10, oy + h - margin), cv::Scalar(200, 200, 200));
    cv::line(img, cv::Point(ox + margin, oy + 10),
             cv::Point(ox + margin, oy + h - margin), cv::Scalar(200, 200, 200));

    cv::putText(img, "Generation", cv::Point(ox + w/2 - 30, oy + h - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200));
    cv::putText(img, "log(f)", cv::Point(ox + 2, oy + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200));

    int plot_w = w - margin - 10;
    int plot_h = h - margin - 10;
    int total = (int)history.size();

    cv::Point prev;
    for (int i = 0; i < total; i++) {
        float lv = logf(fmaxf(history[i], 1e-6f) + 1.0f);
        int px = ox + margin + (int)(plot_w * (float)i / max(total - 1, 1));
        int py = oy + h - margin - (int)(plot_h * (lv - log_min) / (log_max - log_min));
        py = max(oy + 10, min(py, oy + h - margin));
        cv::Point pt(px, py);
        if (i > 0)
            cv::line(img, prev, pt, cv::Scalar(255, 128, 0), 1);
        prev = pt;
    }
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    cout << "=== CUDA CMA-ES ===" << endl;
    cout << "Lambda=" << LAMBDA << ", Mu=" << MU << ", D=" << DIM << endl;

    const int N = LAMBDA;
    const int D = DIM;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    // Device memory
    float *d_samples, *d_fitness;
    float *d_mean, *d_chol_L;
    curandState *d_rng;

    CUDA_CHECK(cudaMalloc(&d_samples, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fitness, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_chol_L, D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rng, N * sizeof(curandState)));

    init_curand_kernel<<<blocks, threads>>>(d_rng, N, 77);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Host state for CMA-ES
    vector<float> h_mean(D, 0.0f);
    // Initialize mean randomly in search space
    srand(42);
    for (int d = 0; d < D; d++)
        h_mean[d] = X_MIN + ((float)rand() / RAND_MAX) * (X_MAX - X_MIN);

    float sigma = 2.0f;  // Initial step size

    // Covariance matrix C = I
    vector<float> C(D * D, 0.0f);
    for (int d = 0; d < D; d++) C[d * D + d] = 1.0f;

    // Cholesky factor
    vector<float> L(D * D, 0.0f);
    cholesky_decompose(C, L, D);

    // Evolution paths
    vector<float> p_sigma(D, 0.0f);
    vector<float> p_c(D, 0.0f);

    // CMA-ES parameters
    float mu_eff = 0.0f;
    vector<float> weights(MU);
    {
        float sum_w = 0.0f;
        for (int i = 0; i < MU; i++) {
            weights[i] = logf((float)(MU + 1)) - logf((float)(i + 1));
            sum_w += weights[i];
        }
        float sum_w2 = 0.0f;
        for (int i = 0; i < MU; i++) {
            weights[i] /= sum_w;
            sum_w2 += weights[i] * weights[i];
        }
        mu_eff = 1.0f / sum_w2;
    }

    float c_sigma = (mu_eff + 2.0f) / (D + mu_eff + 5.0f);
    float d_sigma = 1.0f + 2.0f * fmaxf(0.0f, sqrtf((mu_eff - 1.0f) / (D + 1.0f)) - 1.0f) + c_sigma;
    float cc = (4.0f + mu_eff / D) / (D + 4.0f + 2.0f * mu_eff / D);
    float c1 = 2.0f / ((D + 1.3f) * (D + 1.3f) + mu_eff);
    float c_mu = fminf(1.0f - c1,
        2.0f * (mu_eff - 2.0f + 1.0f / mu_eff) / ((D + 2.0f) * (D + 2.0f) + mu_eff));
    float chi_n = sqrtf((float)D) * (1.0f - 1.0f / (4.0f * D) + 1.0f / (21.0f * D * D));

    // Upload initial state
    CUDA_CHECK(cudaMemcpy(d_mean, h_mean.data(), D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chol_L, L.data(), D * D * sizeof(float), cudaMemcpyHostToDevice));

    // Video
    string avi_path = "gif/cma_es.avi";
    string gif_path = "gif/cma_es.gif";
    cv::VideoWriter video(avi_path, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 15,
                          cv::Size(VIS_W, VIS_H));

    vector<float> h_samples(N * D);
    vector<float> h_fitness(N);
    vector<float> convergence_history;

    auto t_start = chrono::high_resolution_clock::now();

    for (int gen = 0; gen < MAX_GEN; gen++) {
        // GPU: Generate samples
        cma_sample_kernel<<<blocks, threads>>>(
            d_samples, d_mean, d_chol_L, sigma,
            d_rng, N, D, X_MIN, X_MAX);
        CUDA_CHECK(cudaDeviceSynchronize());

        // GPU: Evaluate fitness
        cma_evaluate_kernel<<<blocks, threads>>>(d_samples, d_fitness, N, D, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy to host for CMA update
        CUDA_CHECK(cudaMemcpy(h_samples.data(), d_samples, N * D * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_fitness.data(), d_fitness, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Sort by fitness
        vector<int> indices(N);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&](int a, int b) {
            return h_fitness[a] < h_fitness[b];
        });

        float best_fit = h_fitness[indices[0]];
        convergence_history.push_back(best_fit);

        if (gen % 50 == 0)
            printf("Gen %4d: best = %.6f, sigma = %.6f\n", gen, best_fit, sigma);

        // CPU: CMA-ES update
        // New mean
        vector<float> old_mean = h_mean;
        fill(h_mean.begin(), h_mean.end(), 0.0f);
        for (int i = 0; i < MU; i++) {
            int idx = indices[i];
            for (int d = 0; d < D; d++)
                h_mean[d] += weights[i] * h_samples[idx * D + d];
        }

        // Mean shift in transformed space: inv(L) * (new_mean - old_mean) / sigma
        vector<float> mean_diff(D);
        for (int d = 0; d < D; d++)
            mean_diff[d] = (h_mean[d] - old_mean[d]) / sigma;

        // Solve L * y = mean_diff for y (forward substitution)
        vector<float> inv_L_mean_diff(D);
        for (int i = 0; i < D; i++) {
            float sum = mean_diff[i];
            for (int j = 0; j < i; j++)
                sum -= L[i * D + j] * inv_L_mean_diff[j];
            inv_L_mean_diff[i] = sum / L[i * D + i];
        }

        // Update evolution path p_sigma
        float sqrt_c_sigma = sqrtf(c_sigma * (2.0f - c_sigma) * mu_eff);
        for (int d = 0; d < D; d++)
            p_sigma[d] = (1.0f - c_sigma) * p_sigma[d] + sqrt_c_sigma * inv_L_mean_diff[d];

        // |p_sigma|
        float ps_norm = 0.0f;
        for (int d = 0; d < D; d++)
            ps_norm += p_sigma[d] * p_sigma[d];
        ps_norm = sqrtf(ps_norm);

        // h_sigma indicator
        float threshold = (1.4f + 2.0f / (D + 1.0f)) * chi_n *
            sqrtf(1.0f - powf(1.0f - c_sigma, 2.0f * (gen + 1)));
        int h_sig = (ps_norm < threshold) ? 1 : 0;

        // Update evolution path p_c
        float sqrt_cc = sqrtf(cc * (2.0f - cc) * mu_eff);
        for (int d = 0; d < D; d++)
            p_c[d] = (1.0f - cc) * p_c[d] + h_sig * sqrt_cc * mean_diff[d];

        // Update covariance matrix
        float delta_h = (1.0f - h_sig) * cc * (2.0f - cc);
        for (int i = 0; i < D; i++) {
            for (int j = 0; j <= i; j++) {
                float rank_one = p_c[i] * p_c[j];
                float rank_mu = 0.0f;
                for (int k = 0; k < MU; k++) {
                    int idx = indices[k];
                    float di = (h_samples[idx * D + i] - old_mean[i]) / sigma;
                    float dj = (h_samples[idx * D + j] - old_mean[j]) / sigma;
                    rank_mu += weights[k] * di * dj;
                }
                C[i * D + j] = (1.0f - c1 - c_mu + c1 * delta_h) * C[i * D + j]
                              + c1 * rank_one
                              + c_mu * rank_mu;
                C[j * D + i] = C[i * D + j];
            }
        }

        // Update sigma
        sigma *= expf((c_sigma / d_sigma) * (ps_norm / chi_n - 1.0f));
        sigma = fminf(fmaxf(sigma, 1e-10f), 10.0f);

        // Recompute Cholesky
        cholesky_decompose(C, L, D);

        // Upload updated state to GPU
        CUDA_CHECK(cudaMemcpy(d_mean, h_mean.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_chol_L, L.data(), D * D * sizeof(float), cudaMemcpyHostToDevice));

        // Visualize every 5 generations
        if (gen % 5 == 0) {
            cv::Mat frame(VIS_H, VIS_W, CV_8UC3, cv::Scalar(30, 30, 30));

            cv::Mat landscape = frame(cv::Rect(0, 0, 400, 400));
            draw_rastrigin_landscape(landscape, 400, 400, X_MIN, X_MAX);

            // Draw samples (2D projection)
            int draw_n = min(N, 2000);
            for (int i = 0; i < draw_n; i++) {
                int idx = indices[i];
                float x = h_samples[idx * D + 0];
                float y = h_samples[idx * D + 1];
                int px = (int)((x - X_MIN) / (X_MAX - X_MIN) * 400);
                int py = (int)((y - X_MIN) / (X_MAX - X_MIN) * 400);
                cv::Scalar color = (i < MU) ? cv::Scalar(255, 128, 0) : cv::Scalar(100, 100, 100);
                if (px >= 0 && px < 400 && py >= 0 && py < 400)
                    cv::circle(landscape, cv::Point(px, py), 1, color, -1);
            }

            // Draw mean
            int mx = (int)((h_mean[0] - X_MIN) / (X_MAX - X_MIN) * 400);
            int my = (int)((h_mean[1] - X_MIN) / (X_MAX - X_MIN) * 400);
            cv::circle(landscape, cv::Point(mx, my), 5, cv::Scalar(0, 0, 255), -1);

            draw_convergence(frame, convergence_history, 400, 0, 400, 400);

            char buf[256];
            snprintf(buf, sizeof(buf), "CMA-ES Lambda=%d D=%d Gen=%d Best=%.4f sigma=%.4f",
                     N, D, gen, best_fit, sigma);
            cv::putText(frame, buf, cv::Point(10, VIS_H - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(255, 255, 255));

            video.write(frame);
        }
    }

    auto t_end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(t_end - t_start).count();

    printf("\n=== Results ===\n");
    printf("Final best fitness: %.6f\n", convergence_history.back());
    printf("Total time: %.3f s\n", elapsed);

    video.release();
    cout << "Video saved to " << avi_path << endl;

    string cmd = "ffmpeg -y -i " + avi_path + " -vf 'fps=15,scale=400:-1' -loop 0 " + gif_path + " 2>/dev/null";
    system(cmd.c_str());
    cout << "GIF saved to " << gif_path << endl;

    cudaFree(d_samples);
    cudaFree(d_fitness);
    cudaFree(d_mean);
    cudaFree(d_chol_L);
    cudaFree(d_rng);

    return 0;
}
