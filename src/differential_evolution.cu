/*************************************************************************
    > File Name: differential_evolution.cu
    > CUDA Differential Evolution (DE/rand/1/bin)
    > N=10000, D=30
    > Kernels:
    >   - de_evaluate_kernel: 1 thread per individual
    >   - de_mutate_crossover_kernel: 1 thread per individual (F=0.8, CR=0.9)
    > 1000 generations on Rastrigin function
    > Output: gif/differential_evolution.gif
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <algorithm>
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
static const int DE_N = 10000;
static const int DE_D = 30;
static const int MAX_GEN = 1000;
static const float F = 0.8f;       // Differential weight
static const float CR = 0.9f;      // Crossover probability
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

__global__ void de_init_kernel(
    float* population, curandState* rng,
    int N, int D, float x_min, float x_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState local_state = rng[i];
    for (int d = 0; d < D; d++) {
        population[i * D + d] = x_min + curand_uniform(&local_state) * (x_max - x_min);
    }
    rng[i] = local_state;
}

__global__ void de_evaluate_kernel(
    const float* population, float* fitness,
    int N, int D, int func_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    fitness[i] = evaluate_benchmark(&population[i * D], D, func_id);
}

__global__ void de_mutate_crossover_kernel(
    const float* population, const float* fitness,
    float* trial, float* trial_fitness,
    curandState* rng,
    int N, int D, float F_val, float CR_val,
    float x_min, float x_max, int func_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState local_state = rng[i];

    // Select 3 distinct random indices != i
    int r1, r2, r3;
    do { r1 = (int)(curand_uniform(&local_state) * N) % N; } while (r1 == i);
    do { r2 = (int)(curand_uniform(&local_state) * N) % N; } while (r2 == i || r2 == r1);
    do { r3 = (int)(curand_uniform(&local_state) * N) % N; } while (r3 == i || r3 == r1 || r3 == r2);

    // Guaranteed crossover dimension
    int j_rand = (int)(curand_uniform(&local_state) * D) % D;

    // DE/rand/1/bin
    for (int d = 0; d < D; d++) {
        float r = curand_uniform(&local_state);
        if (r < CR_val || d == j_rand) {
            float mutant = population[r1 * D + d]
                         + F_val * (population[r2 * D + d] - population[r3 * D + d]);
            // Clamp to bounds
            mutant = fminf(fmaxf(mutant, x_min), x_max);
            trial[i * D + d] = mutant;
        } else {
            trial[i * D + d] = population[i * D + d];
        }
    }

    // Evaluate trial
    trial_fitness[i] = evaluate_benchmark(&trial[i * D], D, func_id);

    rng[i] = local_state;
}

__global__ void de_selection_kernel(
    float* population, float* fitness,
    const float* trial, const float* trial_fitness,
    int N, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (trial_fitness[i] <= fitness[i]) {
        fitness[i] = trial_fitness[i];
        for (int d = 0; d < D; d++)
            population[i * D + d] = trial[i * D + d];
    }
}

// Find minimum fitness (simple reduction)
__global__ void find_min_kernel(
    const float* fitness, float* g_best_fit,
    int N)
{
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
        int* g_fit_int = (int*)g_best_fit;
        int old_val = *g_fit_int;
        int new_val = __float_as_int(s_fit[0]);
        while (s_fit[0] < __int_as_float(old_val)) {
            int assumed = old_val;
            old_val = atomicCAS(g_fit_int, assumed, new_val);
            if (old_val == assumed) break;
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
            cv::line(img, prev, pt, cv::Scalar(0, 255, 128), 1);
        prev = pt;
    }
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    cout << "=== CUDA Differential Evolution (DE/rand/1/bin) ===" << endl;
    cout << "N=" << DE_N << ", D=" << DE_D << ", F=" << F << ", CR=" << CR << endl;

    const int N = DE_N;
    const int D = DE_D;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    float *d_population, *d_fitness;
    float *d_trial, *d_trial_fitness;
    float *d_g_best_fit;
    curandState *d_rng;

    CUDA_CHECK(cudaMalloc(&d_population, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fitness, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trial, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trial_fitness, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_best_fit, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rng, N * sizeof(curandState)));

    init_curand_kernel<<<blocks, threads>>>(d_rng, N, 123);
    CUDA_CHECK(cudaDeviceSynchronize());

    de_init_kernel<<<blocks, threads>>>(d_population, d_rng, N, D, X_MIN, X_MAX);
    CUDA_CHECK(cudaDeviceSynchronize());

    de_evaluate_kernel<<<blocks, threads>>>(d_population, d_fitness, N, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Video
    string avi_path = "gif/differential_evolution.avi";
    string gif_path = "gif/differential_evolution.gif";
    cv::VideoWriter video(avi_path, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 15,
                          cv::Size(VIS_W, VIS_H));

    vector<float> h_population(N * D);
    vector<float> convergence_history;

    auto t_start = chrono::high_resolution_clock::now();

    for (int gen = 0; gen < MAX_GEN; gen++) {
        // Mutation + Crossover + Trial evaluation
        de_mutate_crossover_kernel<<<blocks, threads>>>(
            d_population, d_fitness, d_trial, d_trial_fitness,
            d_rng, N, D, F, CR, X_MIN, X_MAX, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Selection
        de_selection_kernel<<<blocks, threads>>>(
            d_population, d_fitness, d_trial, d_trial_fitness, N, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Find global best
        float init_fit = FLT_MAX;
        CUDA_CHECK(cudaMemcpy(d_g_best_fit, &init_fit, sizeof(float), cudaMemcpyHostToDevice));
        find_min_kernel<<<blocks, threads>>>(d_fitness, d_g_best_fit, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        float g_best;
        CUDA_CHECK(cudaMemcpy(&g_best, d_g_best_fit, sizeof(float), cudaMemcpyDeviceToHost));
        convergence_history.push_back(g_best);

        if (gen % 50 == 0)
            printf("Gen %4d: best = %.6f\n", gen, g_best);

        // Visualize every 10 generations
        if (gen % 10 == 0) {
            CUDA_CHECK(cudaMemcpy(h_population.data(), d_population,
                N * D * sizeof(float), cudaMemcpyDeviceToHost));

            cv::Mat frame(VIS_H, VIS_W, CV_8UC3, cv::Scalar(30, 30, 30));

            // Left: landscape with individuals
            cv::Mat landscape = frame(cv::Rect(0, 0, 400, 400));
            draw_rastrigin_landscape(landscape, 400, 400, X_MIN, X_MAX);

            int draw_n = min(N, 3000);
            for (int i = 0; i < draw_n; i++) {
                float x = h_population[i * D + 0];
                float y = h_population[i * D + 1];
                int px = (int)((x - X_MIN) / (X_MAX - X_MIN) * 400);
                int py = (int)((y - X_MIN) / (X_MAX - X_MIN) * 400);
                if (px >= 0 && px < 400 && py >= 0 && py < 400)
                    cv::circle(landscape, cv::Point(px, py), 1, cv::Scalar(0, 255, 128), -1);
            }

            // Right: convergence
            draw_convergence(frame, convergence_history, 400, 0, 400, 400);

            char buf[256];
            snprintf(buf, sizeof(buf), "DE/rand/1/bin N=%d D=%d Gen=%d Best=%.4f",
                     N, D, gen, g_best);
            cv::putText(frame, buf, cv::Point(10, VIS_H - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));

            video.write(frame);
        }
    }

    auto t_end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(t_end - t_start).count();

    float final_best;
    CUDA_CHECK(cudaMemcpy(&final_best, d_g_best_fit, sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n=== Results ===\n");
    printf("Final best fitness: %.6f\n", final_best);
    printf("Total time: %.3f s\n", elapsed);

    video.release();
    cout << "Video saved to " << avi_path << endl;

    string cmd = "ffmpeg -y -i " + avi_path + " -vf 'fps=15,scale=400:-1' -loop 0 " + gif_path + " 2>/dev/null";
    system(cmd.c_str());
    cout << "GIF saved to " << gif_path << endl;

    cudaFree(d_population);
    cudaFree(d_fitness);
    cudaFree(d_trial);
    cudaFree(d_trial_fitness);
    cudaFree(d_g_best_fit);
    cudaFree(d_rng);

    return 0;
}
