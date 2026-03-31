/*************************************************************************
    > File Name: pso.cu
    > CUDA Particle Swarm Optimization
    > N=100000 particles, D=30 dimensions
    > Kernels:
    >   - pso_evaluate_kernel: 1 thread per particle, evaluate fitness
    >   - pso_update_kernel: 1 thread per particle, update velocity & position
    >     w = 0.9 -> 0.4 linear decay, c1=c2=2.0
    > 1000 iterations on Rastrigin function
    > Visualization: 2D projection of particles on Rastrigin landscape + convergence
    > Output: gif/pso.gif
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
static const int PSO_N = 100000;
static const int PSO_D = 30;
static const int MAX_ITER = 1000;
static const float W_START = 0.9f;
static const float W_END = 0.4f;
static const float C1 = 2.0f;
static const float C2 = 2.0f;
static const float X_MIN = -5.12f;
static const float X_MAX = 5.12f;
static const float V_MAX = 2.0f;

// -------------------------------------------------------------------------
// Kernels
// -------------------------------------------------------------------------
__global__ void init_curand_kernel(curandState* states, int N, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curand_init(seed, i, 0, &states[i]);
}

__global__ void pso_init_kernel(
    float* positions, float* velocities,
    float* p_best_pos, float* p_best_fit,
    curandState* rng, int N, int D,
    float x_min, float x_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState local_state = rng[i];
    for (int d = 0; d < D; d++) {
        float r = curand_uniform(&local_state);
        float pos = x_min + r * (x_max - x_min);
        positions[i * D + d] = pos;
        p_best_pos[i * D + d] = pos;
        velocities[i * D + d] = (curand_uniform(&local_state) - 0.5f) * 2.0f;
    }
    p_best_fit[i] = FLT_MAX;
    rng[i] = local_state;
}

__global__ void pso_evaluate_kernel(
    const float* positions, float* fitness,
    int N, int D, int func_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    fitness[i] = evaluate_benchmark(&positions[i * D], D, func_id);
}

__global__ void pso_update_pbest_kernel(
    const float* positions, const float* fitness,
    float* p_best_pos, float* p_best_fit,
    int N, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (fitness[i] < p_best_fit[i]) {
        p_best_fit[i] = fitness[i];
        for (int d = 0; d < D; d++)
            p_best_pos[i * D + d] = positions[i * D + d];
    }
}

__global__ void pso_update_kernel(
    float* positions, float* velocities,
    const float* p_best_pos,
    const float* g_best_pos,
    float w, float c1, float c2,
    curandState* rng,
    int N, int D,
    float x_min, float x_max, float v_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState local_state = rng[i];
    for (int d = 0; d < D; d++) {
        float r1 = curand_uniform(&local_state);
        float r2 = curand_uniform(&local_state);
        int idx = i * D + d;

        float v = w * velocities[idx]
                + c1 * r1 * (p_best_pos[idx] - positions[idx])
                + c2 * r2 * (g_best_pos[d] - positions[idx]);

        // Clamp velocity
        v = fminf(fmaxf(v, -v_max), v_max);
        velocities[idx] = v;

        float pos = positions[idx] + v;
        // Clamp position
        pos = fminf(fmaxf(pos, x_min), x_max);
        positions[idx] = pos;
    }
    rng[i] = local_state;
}

// Simple parallel reduction to find global best
__global__ void find_min_kernel(
    const float* fitness, const float* positions,
    float* g_best_fit, float* g_best_pos,
    int N, int D)
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
        // Use atomics to compare across blocks
        // Simple approach: write block results, then reduce on CPU
        int block_best = s_idx[0];
        float block_fit = s_fit[0];
        // Atomic compare-and-swap for float min
        int* g_fit_int = (int*)g_best_fit;
        int old_val = *g_fit_int;
        int new_val = __float_as_int(block_fit);
        while (block_fit < __int_as_float(old_val)) {
            int assumed = old_val;
            old_val = atomicCAS(g_fit_int, assumed, new_val);
            if (old_val == assumed) {
                // We won the CAS, update position
                for (int d = 0; d < D; d++)
                    g_best_pos[d] = positions[block_best * D + d];
                break;
            }
        }
    }
}

// -------------------------------------------------------------------------
// Visualization
// -------------------------------------------------------------------------
static const int VIS_W = 800;
static const int VIS_H = 400;
static const int LANDSCAPE_W = 400;
static const int LANDSCAPE_H = 400;
static const int GRAPH_W = 400;
static const int GRAPH_H = 400;

void draw_rastrigin_landscape(cv::Mat& img, float view_min, float view_max) {
    for (int py = 0; py < LANDSCAPE_H; py++) {
        for (int px = 0; px < LANDSCAPE_W; px++) {
            float x = view_min + (view_max - view_min) * px / LANDSCAPE_W;
            float y = view_min + (view_max - view_min) * py / LANDSCAPE_H;
            float val = 20.0f + x*x + y*y - 10.0f*(cosf(2*M_PI*x) + cosf(2*M_PI*y));
            // Map value to color (log scale for better visibility)
            float norm = fminf(logf(val + 1.0f) / logf(100.0f), 1.0f);
            int b = (int)(255 * (1.0f - norm));
            int g = (int)(255 * (1.0f - norm) * 0.5f);
            int r = (int)(50 + 205 * norm);
            img.at<cv::Vec3b>(py, px) = cv::Vec3b(b, g, r);
        }
    }
}

void draw_particles_2d(cv::Mat& img,
    const vector<float>& positions, int N, int D,
    float view_min, float view_max) {
    // Draw first 5000 particles (x[0], x[1] projection)
    int draw_n = min(N, 5000);
    for (int i = 0; i < draw_n; i++) {
        float x = positions[i * D + 0];
        float y = positions[i * D + 1];
        int px = (int)((x - view_min) / (view_max - view_min) * LANDSCAPE_W);
        int py = (int)((y - view_min) / (view_max - view_min) * LANDSCAPE_H);
        if (px >= 0 && px < LANDSCAPE_W && py >= 0 && py < LANDSCAPE_H)
            cv::circle(img, cv::Point(px, py), 1, cv::Scalar(0, 255, 255), -1);
    }
}

void draw_convergence(cv::Mat& img, const vector<float>& history, int iter,
    int ox, int oy, int w, int h) {
    // Background
    cv::rectangle(img, cv::Rect(ox, oy, w, h), cv::Scalar(40, 40, 40), -1);

    if (history.empty()) return;

    // Find range
    float max_val = *max_element(history.begin(), history.end());
    float min_val = *min_element(history.begin(), history.end());
    if (max_val <= min_val) max_val = min_val + 1.0f;

    // Use log scale
    float log_max = logf(max_val + 1.0f);
    float log_min = logf(fmaxf(min_val, 1e-6f) + 1.0f);
    if (log_max <= log_min) log_max = log_min + 1.0f;

    int margin = 40;

    // Axes
    cv::line(img, cv::Point(ox + margin, oy + h - margin),
             cv::Point(ox + w - 10, oy + h - margin), cv::Scalar(200, 200, 200));
    cv::line(img, cv::Point(ox + margin, oy + 10),
             cv::Point(ox + margin, oy + h - margin), cv::Scalar(200, 200, 200));

    cv::putText(img, "Iteration", cv::Point(ox + w/2 - 30, oy + h - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200));
    cv::putText(img, "log(f)", cv::Point(ox + 2, oy + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200));

    int plot_w = w - margin - 10;
    int plot_h = h - margin - 10;

    int total_iters = (int)history.size();
    cv::Point prev;
    for (int i = 0; i < (int)history.size(); i++) {
        float lv = logf(fmaxf(history[i], 1e-6f) + 1.0f);
        int px = ox + margin + (int)(plot_w * (float)i / max(total_iters - 1, 1));
        int py = oy + h - margin - (int)(plot_h * (lv - log_min) / (log_max - log_min));
        py = max(oy + 10, min(py, oy + h - margin));
        cv::Point pt(px, py);
        if (i > 0)
            cv::line(img, prev, pt, cv::Scalar(0, 200, 255), 1);
        prev = pt;
    }
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    cout << "=== CUDA Particle Swarm Optimization ===" << endl;
    cout << "N=" << PSO_N << " particles, D=" << PSO_D << " dimensions" << endl;
    cout << "Function: Rastrigin" << endl;

    const int N = PSO_N;
    const int D = PSO_D;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    // Allocate device memory
    float *d_positions, *d_velocities, *d_fitness;
    float *d_p_best_pos, *d_p_best_fit;
    float *d_g_best_pos, *d_g_best_fit;
    curandState *d_rng;

    CUDA_CHECK(cudaMalloc(&d_positions, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_velocities, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fitness, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_best_pos, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_best_fit, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_best_pos, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_best_fit, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rng, N * sizeof(curandState)));

    // Initialize cuRAND
    init_curand_kernel<<<blocks, threads>>>(d_rng, N, 42);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize particles
    pso_init_kernel<<<blocks, threads>>>(
        d_positions, d_velocities, d_p_best_pos, d_p_best_fit,
        d_rng, N, D, X_MIN, X_MAX);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize global best
    float init_fit = FLT_MAX;
    CUDA_CHECK(cudaMemcpy(d_g_best_fit, &init_fit, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_g_best_pos, 0, D * sizeof(float)));

    // First evaluation
    pso_evaluate_kernel<<<blocks, threads>>>(d_positions, d_fitness, N, D, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    pso_update_pbest_kernel<<<blocks, threads>>>(d_positions, d_fitness, d_p_best_pos, d_p_best_fit, N, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    find_min_kernel<<<blocks, threads>>>(d_fitness, d_positions, d_g_best_fit, d_g_best_pos, N, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Video output
    string avi_path = "gif/pso.avi";
    string gif_path = "gif/pso.gif";
    cv::VideoWriter video(avi_path, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 15,
                          cv::Size(VIS_W, VIS_H));

    vector<float> h_positions(N * D);
    vector<float> convergence_history;

    auto t_start = chrono::high_resolution_clock::now();

    // Main PSO loop
    for (int iter = 0; iter < MAX_ITER; iter++) {
        float w = W_START - (W_START - W_END) * (float)iter / (float)(MAX_ITER - 1);

        // Evaluate fitness
        pso_evaluate_kernel<<<blocks, threads>>>(d_positions, d_fitness, N, D, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update personal best
        pso_update_pbest_kernel<<<blocks, threads>>>(
            d_positions, d_fitness, d_p_best_pos, d_p_best_fit, N, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Find global best
        CUDA_CHECK(cudaMemcpy(d_g_best_fit, &init_fit, sizeof(float), cudaMemcpyHostToDevice));
        find_min_kernel<<<blocks, threads>>>(
            d_p_best_fit, d_p_best_pos, d_g_best_fit, d_g_best_pos, N, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Get global best fitness
        float g_best_fit;
        CUDA_CHECK(cudaMemcpy(&g_best_fit, d_g_best_fit, sizeof(float), cudaMemcpyDeviceToHost));
        convergence_history.push_back(g_best_fit);

        // Update velocities and positions
        pso_update_kernel<<<blocks, threads>>>(
            d_positions, d_velocities, d_p_best_pos, d_g_best_pos,
            w, C1, C2, d_rng, N, D, X_MIN, X_MAX, V_MAX);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (iter % 50 == 0)
            printf("Iter %4d: global best = %.6f, w = %.3f\n", iter, g_best_fit, w);

        // Visualize every 10 iterations
        if (iter % 10 == 0) {
            CUDA_CHECK(cudaMemcpy(h_positions.data(), d_positions,
                N * D * sizeof(float), cudaMemcpyDeviceToHost));

            cv::Mat frame(VIS_H, VIS_W, CV_8UC3, cv::Scalar(30, 30, 30));

            // Left: Rastrigin landscape with particles
            cv::Mat landscape = frame(cv::Rect(0, 0, LANDSCAPE_W, LANDSCAPE_H));
            draw_rastrigin_landscape(landscape, X_MIN, X_MAX);
            draw_particles_2d(landscape, h_positions, N, D, X_MIN, X_MAX);

            // Draw global best
            float gbx, gby;
            float h_g_best_pos[2];
            CUDA_CHECK(cudaMemcpy(h_g_best_pos, d_g_best_pos, 2 * sizeof(float), cudaMemcpyDeviceToHost));
            int gpx = (int)((h_g_best_pos[0] - X_MIN) / (X_MAX - X_MIN) * LANDSCAPE_W);
            int gpy = (int)((h_g_best_pos[1] - X_MIN) / (X_MAX - X_MIN) * LANDSCAPE_H);
            cv::circle(landscape, cv::Point(gpx, gpy), 5, cv::Scalar(0, 0, 255), -1);

            // Right: convergence graph
            draw_convergence(frame, convergence_history, iter, LANDSCAPE_W, 0, GRAPH_W, GRAPH_H);

            // Title and info
            char buf[256];
            snprintf(buf, sizeof(buf), "PSO N=%d D=%d Iter=%d Best=%.4f",
                     N, D, iter, g_best_fit);
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
    printf("Time per iteration: %.3f ms\n", elapsed / MAX_ITER * 1000.0);

    // Timing for evaluation kernel only
    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);
    cudaEventRecord(start_ev);
    pso_evaluate_kernel<<<blocks, threads>>>(d_positions, d_fitness, N, D, 0);
    cudaEventRecord(stop_ev);
    cudaEventSynchronize(stop_ev);
    float eval_ms;
    cudaEventElapsedTime(&eval_ms, start_ev, stop_ev);
    printf("Evaluation kernel time (N=%d): %.3f ms\n", N, eval_ms);
    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);

    video.release();
    cout << "Video saved to " << avi_path << endl;

    string cmd = "ffmpeg -y -i " + avi_path + " -vf 'fps=15,scale=400:-1' -loop 0 " + gif_path + " 2>/dev/null";
    system(cmd.c_str());
    cout << "GIF saved to " << gif_path << endl;

    // Cleanup
    cudaFree(d_positions);
    cudaFree(d_velocities);
    cudaFree(d_fitness);
    cudaFree(d_p_best_pos);
    cudaFree(d_p_best_fit);
    cudaFree(d_g_best_pos);
    cudaFree(d_g_best_fit);
    cudaFree(d_rng);

    return 0;
}
