/*************************************************************************
    STOMP: CPU (K=100) vs CUDA (K=10,000) side-by-side comparison
    Left panel:  CPU with K=100 samples (sparse, slow convergence)
    Right panel: CUDA with K=10,000 samples (dense, fast convergence)
    Output: gif/comparison_stomp.gif
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cfloat>
#include <algorithm>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------
static const int K_CPU         = 100;      // CPU sample count
static const int K_GPU         = 10000;    // GPU sample count
static const int N_WAYPOINTS   = 50;
static const int DIM           = 2;
static const float LAMBDA      = 10.0f;
static const int MAX_ITER      = 100;
static const float WORKSPACE   = 50.0f;
static const float NOISE_STD   = 2.0f;
static const float OBS_COST_WEIGHT    = 100.0f;
static const float SMOOTH_COST_WEIGHT = 1.0f;
static const float OBS_CLEARANCE      = 1.5f;

static const float START_X = 5.0f,  START_Y = 5.0f;
static const float GOAL_X  = 45.0f, GOAL_Y  = 45.0f;

// Obstacles
static const int N_OBSTACLES = 8;
__constant__ float d_obs_x[N_OBSTACLES];
__constant__ float d_obs_y[N_OBSTACLES];
__constant__ float d_obs_r[N_OBSTACLES];

static float h_obs_x[N_OBSTACLES] = {10.0f, 20.0f, 30.0f, 15.0f, 25.0f, 35.0f, 22.0f, 38.0f};
static float h_obs_y[N_OBSTACLES] = {15.0f, 25.0f, 10.0f, 35.0f, 18.0f, 30.0f, 40.0f, 20.0f};
static float h_obs_r[N_OBSTACLES] = { 3.0f,  4.0f,  3.5f,  3.0f,  3.5f,  2.5f,  3.0f,  3.0f};

// -------------------------------------------------------------------------
// CUDA Kernels (same as stomp.cu)
// -------------------------------------------------------------------------
__global__ void init_curand_kernel(curandState* states, int K, unsigned long long seed)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curand_init(seed, k, 0, &states[k]);
}

__global__ void generate_noisy_trajectories_kernel(
    const float* __restrict__ d_trajectory,
    float* __restrict__ d_noisy,
    curandState* __restrict__ d_rand_states,
    int K, int N, int dim, float noise_std)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_state = d_rand_states[k];

    float raw_x[N_WAYPOINTS];
    float raw_y[N_WAYPOINTS];

    for (int i = 0; i < N; i++) {
        raw_x[i] = curand_normal(&local_state) * noise_std;
        raw_y[i] = curand_normal(&local_state) * noise_std;
    }

    int base = k * N * dim;
    for (int i = 0; i < N; i++) {
        float sx = 0.0f, sy = 0.0f;
        int count = 0;
        for (int j = max(0, i - 2); j <= min(N - 1, i + 2); j++) {
            sx += raw_x[j];
            sy += raw_y[j];
            count++;
        }
        sx /= (float)count;
        sy /= (float)count;

        d_noisy[base + i * dim + 0] = d_trajectory[i * dim + 0] + sx;
        d_noisy[base + i * dim + 1] = d_trajectory[i * dim + 1] + sy;
    }

    d_noisy[base + 0]               = d_trajectory[0];
    d_noisy[base + 1]               = d_trajectory[1];
    d_noisy[base + (N - 1) * dim + 0] = d_trajectory[(N - 1) * dim + 0];
    d_noisy[base + (N - 1) * dim + 1] = d_trajectory[(N - 1) * dim + 1];

    d_rand_states[k] = local_state;
}

__global__ void compute_costs_kernel(
    const float* __restrict__ d_noisy,
    float* __restrict__ d_costs,
    int K, int N, int dim,
    float obs_weight, float smooth_weight, float clearance)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    int base = k * N * dim;
    float cost = 0.0f;

    for (int i = 0; i < N; i++) {
        float wx = d_noisy[base + i * dim + 0];
        float wy = d_noisy[base + i * dim + 1];

        for (int o = 0; o < N_OBSTACLES; o++) {
            float dx = wx - d_obs_x[o];
            float dy = wy - d_obs_y[o];
            float dist = sqrtf(dx * dx + dy * dy);
            float margin = dist - d_obs_r[o];
            if (margin < 0.0f) {
                cost += obs_weight * 10.0f;
            } else if (margin < clearance) {
                float t = 1.0f - margin / clearance;
                cost += obs_weight * t * t;
            }
        }

        if (i >= 1 && i < N - 1) {
            float ax = d_noisy[base + (i + 1) * dim + 0] - 2.0f * wx + d_noisy[base + (i - 1) * dim + 0];
            float ay = d_noisy[base + (i + 1) * dim + 1] - 2.0f * wy + d_noisy[base + (i - 1) * dim + 1];
            cost += smooth_weight * (ax * ax + ay * ay);
        }
    }

    d_costs[k] = cost;
}

__global__ void update_trajectory_kernel(
    float* __restrict__ d_trajectory,
    const float* __restrict__ d_noisy,
    const float* __restrict__ d_weights,
    int K, int N, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (i == 0 || i == N - 1) return;

    for (int d = 0; d < dim; d++) {
        float weighted_sum = 0.0f;
        for (int k = 0; k < K; k++) {
            weighted_sum += d_weights[k] * d_noisy[k * N * dim + i * dim + d];
        }
        d_trajectory[i * dim + d] = weighted_sum;
    }
}

// -------------------------------------------------------------------------
// CPU STOMP implementation
// -------------------------------------------------------------------------
static float cpu_obstacle_cost(float wx, float wy)
{
    float cost = 0.0f;
    for (int o = 0; o < N_OBSTACLES; o++) {
        float dx = wx - h_obs_x[o];
        float dy = wy - h_obs_y[o];
        float dist = sqrtf(dx * dx + dy * dy);
        float margin = dist - h_obs_r[o];
        if (margin < 0.0f) {
            cost += OBS_COST_WEIGHT * 10.0f;
        } else if (margin < OBS_CLEARANCE) {
            float t = 1.0f - margin / OBS_CLEARANCE;
            cost += OBS_COST_WEIGHT * t * t;
        }
    }
    return cost;
}

struct CPUStomp {
    int K;
    vector<float> trajectory;       // [N * DIM]
    vector<float> noisy;            // [K * N * DIM]
    vector<float> costs;            // [K]
    vector<float> weights;          // [K]

    void init(int k_samples) {
        K = k_samples;
        int traj_size = N_WAYPOINTS * DIM;
        trajectory.resize(traj_size);
        noisy.resize(K * traj_size);
        costs.resize(K);
        weights.resize(K);

        // Straight line initialization
        for (int i = 0; i < N_WAYPOINTS; i++) {
            float t = (float)i / (float)(N_WAYPOINTS - 1);
            trajectory[i * DIM + 0] = START_X + t * (GOAL_X - START_X);
            trajectory[i * DIM + 1] = START_Y + t * (GOAL_Y - START_Y);
        }
    }

    void generate_noisy() {
        int traj_size = N_WAYPOINTS * DIM;
        for (int k = 0; k < K; k++) {
            // Generate raw noise
            float raw_x[N_WAYPOINTS], raw_y[N_WAYPOINTS];
            for (int i = 0; i < N_WAYPOINTS; i++) {
                raw_x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * NOISE_STD;
                raw_y[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * NOISE_STD;
            }

            int base = k * traj_size;
            for (int i = 0; i < N_WAYPOINTS; i++) {
                float sx = 0.0f, sy = 0.0f;
                int count = 0;
                for (int j = max(0, i - 2); j <= min(N_WAYPOINTS - 1, i + 2); j++) {
                    sx += raw_x[j]; sy += raw_y[j]; count++;
                }
                sx /= (float)count; sy /= (float)count;
                noisy[base + i * DIM + 0] = trajectory[i * DIM + 0] + sx;
                noisy[base + i * DIM + 1] = trajectory[i * DIM + 1] + sy;
            }
            // Fix endpoints
            noisy[base + 0] = trajectory[0];
            noisy[base + 1] = trajectory[1];
            noisy[base + (N_WAYPOINTS - 1) * DIM + 0] = trajectory[(N_WAYPOINTS - 1) * DIM + 0];
            noisy[base + (N_WAYPOINTS - 1) * DIM + 1] = trajectory[(N_WAYPOINTS - 1) * DIM + 1];
        }
    }

    void compute_costs_cpu() {
        int traj_size = N_WAYPOINTS * DIM;
        for (int k = 0; k < K; k++) {
            int base = k * traj_size;
            float cost = 0.0f;
            for (int i = 0; i < N_WAYPOINTS; i++) {
                float wx = noisy[base + i * DIM + 0];
                float wy = noisy[base + i * DIM + 1];
                cost += cpu_obstacle_cost(wx, wy);
                if (i >= 1 && i < N_WAYPOINTS - 1) {
                    float ax = noisy[base + (i + 1) * DIM + 0] - 2.0f * wx + noisy[base + (i - 1) * DIM + 0];
                    float ay = noisy[base + (i + 1) * DIM + 1] - 2.0f * wy + noisy[base + (i - 1) * DIM + 1];
                    cost += SMOOTH_COST_WEIGHT * (ax * ax + ay * ay);
                }
            }
            costs[k] = cost;
        }
    }

    void compute_weights_cpu() {
        float min_cost = *min_element(costs.begin(), costs.end());
        float sum_exp = 0.0f;
        for (int k = 0; k < K; k++) {
            weights[k] = expf(-1.0f / LAMBDA * (costs[k] - min_cost));
            sum_exp += weights[k];
        }
        if (sum_exp > 0.0f) {
            for (int k = 0; k < K; k++) weights[k] /= sum_exp;
        }
    }

    void update_trajectory_cpu() {
        int traj_size = N_WAYPOINTS * DIM;
        for (int i = 1; i < N_WAYPOINTS - 1; i++) {
            for (int d = 0; d < DIM; d++) {
                float ws = 0.0f;
                for (int k = 0; k < K; k++) {
                    ws += weights[k] * noisy[k * traj_size + i * DIM + d];
                }
                trajectory[i * DIM + d] = ws;
            }
        }
    }

    float iterate() {
        generate_noisy();
        compute_costs_cpu();
        compute_weights_cpu();
        update_trajectory_cpu();
        return *min_element(costs.begin(), costs.end());
    }
};

// -------------------------------------------------------------------------
// GPU STOMP wrapper
// -------------------------------------------------------------------------
struct GPUStomp {
    int K;
    float *d_trajectory, *d_noisy, *d_costs, *d_weights;
    curandState *d_rand_states;
    vector<float> h_trajectory;
    vector<float> h_costs;
    vector<float> h_weights;
    vector<float> h_noisy;
    int traj_size;

    void init(int k_samples) {
        K = k_samples;
        traj_size = N_WAYPOINTS * DIM;
        h_trajectory.resize(traj_size);
        h_costs.resize(K);
        h_weights.resize(K);
        h_noisy.resize(K * traj_size);

        for (int i = 0; i < N_WAYPOINTS; i++) {
            float t = (float)i / (float)(N_WAYPOINTS - 1);
            h_trajectory[i * DIM + 0] = START_X + t * (GOAL_X - START_X);
            h_trajectory[i * DIM + 1] = START_Y + t * (GOAL_Y - START_Y);
        }

        CUDA_CHECK(cudaMalloc(&d_trajectory, traj_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_noisy, K * traj_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs, K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights, K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rand_states, K * sizeof(curandState)));

        CUDA_CHECK(cudaMemcpy(d_trajectory, h_trajectory.data(), traj_size * sizeof(float), cudaMemcpyHostToDevice));

        int block = 256;
        int grid = (K + block - 1) / block;
        init_curand_kernel<<<grid, block>>>(d_rand_states, K, 42ULL);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    float iterate() {
        int block = 256;
        int grid_K = (K + block - 1) / block;
        int grid_N = (N_WAYPOINTS + block - 1) / block;

        generate_noisy_trajectories_kernel<<<grid_K, block>>>(
            d_trajectory, d_noisy, d_rand_states,
            K, N_WAYPOINTS, DIM, NOISE_STD);
        CUDA_CHECK(cudaGetLastError());

        compute_costs_kernel<<<grid_K, block>>>(
            d_noisy, d_costs, K, N_WAYPOINTS, DIM,
            OBS_COST_WEIGHT, SMOOTH_COST_WEIGHT, OBS_CLEARANCE);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_costs.data(), d_costs, K * sizeof(float), cudaMemcpyDeviceToHost));

        // Compute weights on host
        float min_cost = *min_element(h_costs.begin(), h_costs.end());
        float sum_exp = 0.0f;
        for (int k = 0; k < K; k++) {
            h_weights[k] = expf(-1.0f / LAMBDA * (h_costs[k] - min_cost));
            sum_exp += h_weights[k];
        }
        if (sum_exp > 0.0f) {
            for (int k = 0; k < K; k++) h_weights[k] /= sum_exp;
        }

        CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), K * sizeof(float), cudaMemcpyHostToDevice));

        update_trajectory_kernel<<<grid_N, block>>>(
            d_trajectory, d_noisy, d_weights,
            K, N_WAYPOINTS, DIM);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_trajectory.data(), d_trajectory, traj_size * sizeof(float), cudaMemcpyDeviceToHost));

        return min_cost;
    }

    void copy_noisy(int vis_K) {
        vis_K = min(vis_K, K);
        CUDA_CHECK(cudaMemcpy(h_noisy.data(), d_noisy, vis_K * traj_size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void cleanup() {
        CUDA_CHECK(cudaFree(d_trajectory));
        CUDA_CHECK(cudaFree(d_noisy));
        CUDA_CHECK(cudaFree(d_costs));
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_rand_states));
    }
};

// -------------------------------------------------------------------------
// Visualization
// -------------------------------------------------------------------------
static cv::Point world_to_pixel(float wx, float wy, int img_size, float ws)
{
    int px = (int)(wx / ws * (float)img_size);
    int py = img_size - 1 - (int)(wy / ws * (float)img_size);
    return cv::Point(px, py);
}

void draw_panel(
    cv::Mat& img,
    const vector<float>& trajectory,
    const vector<float>& noisy,
    int vis_K,
    const char* label,
    float min_cost,
    int iter,
    double ms)
{
    int H = img.rows, W = img.cols;
    img.setTo(cv::Scalar(255, 255, 255));

    // Draw obstacles
    for (int o = 0; o < N_OBSTACLES; o++) {
        cv::Point center = world_to_pixel(h_obs_x[o], h_obs_y[o], W, WORKSPACE);
        int r = (int)(h_obs_r[o] / WORKSPACE * (float)W);
        cv::circle(img, center, r, cv::Scalar(40, 40, 40), -1);
    }

    // Draw noisy trajectories
    for (int k = 0; k < vis_K; k++) {
        int base = k * N_WAYPOINTS * DIM;
        for (int i = 0; i < N_WAYPOINTS - 1; i++) {
            cv::Point p1 = world_to_pixel(noisy[base + i * DIM + 0],
                                           noisy[base + i * DIM + 1], W, WORKSPACE);
            cv::Point p2 = world_to_pixel(noisy[base + (i + 1) * DIM + 0],
                                           noisy[base + (i + 1) * DIM + 1], W, WORKSPACE);
            cv::line(img, p1, p2, cv::Scalar(220, 220, 220), 1);
        }
    }

    // Draw current trajectory (thick red)
    for (int i = 0; i < N_WAYPOINTS - 1; i++) {
        cv::Point p1 = world_to_pixel(trajectory[i * DIM + 0],
                                       trajectory[i * DIM + 1], W, WORKSPACE);
        cv::Point p2 = world_to_pixel(trajectory[(i + 1) * DIM + 0],
                                       trajectory[(i + 1) * DIM + 1], W, WORKSPACE);
        cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 3);
    }

    // Start and goal
    cv::circle(img, world_to_pixel(START_X, START_Y, W, WORKSPACE), 7, cv::Scalar(0, 200, 0), -1);
    cv::circle(img, world_to_pixel(GOAL_X, GOAL_Y, W, WORKSPACE), 7, cv::Scalar(255, 0, 0), -1);

    // Labels
    cv::putText(img, label, cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 0), 2);
    char buf[128];
    snprintf(buf, sizeof(buf), "Iter %d/%d  Cost: %.1f", iter, MAX_ITER, min_cost);
    cv::putText(img, buf, cv::Point(10, 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(80, 80, 80), 1);
    snprintf(buf, sizeof(buf), "Time: %.2f ms", ms);
    cv::putText(img, buf, cv::Point(10, 72),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(80, 80, 80), 1);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main()
{
    cout << "STOMP: CPU (K=" << K_CPU << ") vs CUDA (K=" << K_GPU << ") Comparison" << endl;

    srand(42);

    // Copy obstacles to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_x, h_obs_x, N_OBSTACLES * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_y, h_obs_y, N_OBSTACLES * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_r, h_obs_r, N_OBSTACLES * sizeof(float)));

    CPUStomp cpu_stomp;
    cpu_stomp.init(K_CPU);

    GPUStomp gpu_stomp;
    gpu_stomp.init(K_GPU);

    int W = 500, H = 500;

    cv::VideoWriter video(
        "gif/comparison_stomp.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 15,
        cv::Size(W * 2, H));

    if (!video.isOpened()) {
        cerr << "Failed to open video writer" << endl;
        return 1;
    }

    for (int iter = 1; iter <= MAX_ITER; iter++) {
        // CPU iteration
        auto t0 = chrono::high_resolution_clock::now();
        float cpu_cost = cpu_stomp.iterate();
        auto t1 = chrono::high_resolution_clock::now();
        double cpu_ms = chrono::duration<double, milli>(t1 - t0).count();

        // GPU iteration
        auto t2 = chrono::high_resolution_clock::now();
        float gpu_cost = gpu_stomp.iterate();
        auto t3 = chrono::high_resolution_clock::now();
        double gpu_ms = chrono::duration<double, milli>(t3 - t2).count();

        // Copy noisy samples for visualization
        int cpu_vis_K = min(K_CPU, 100);
        int gpu_vis_K = min(K_GPU, 500);
        gpu_stomp.copy_noisy(gpu_vis_K);

        // Draw
        cv::Mat left(H, W, CV_8UC3);
        cv::Mat right(H, W, CV_8UC3);

        char cpu_label[64], gpu_label[64];
        snprintf(cpu_label, sizeof(cpu_label), "CPU (K=%d)", K_CPU);
        snprintf(gpu_label, sizeof(gpu_label), "CUDA (K=%d)", K_GPU);

        draw_panel(left, cpu_stomp.trajectory, cpu_stomp.noisy, cpu_vis_K,
                   cpu_label, cpu_cost, iter, cpu_ms);
        draw_panel(right, gpu_stomp.h_trajectory, gpu_stomp.h_noisy, gpu_vis_K,
                   gpu_label, gpu_cost, iter, gpu_ms);

        cv::Mat combined;
        cv::hconcat(left, right, combined);

        cv::namedWindow("comparison_stomp", cv::WINDOW_AUTOSIZE);
        cv::imshow("comparison_stomp", combined);
        cv::waitKey(1);
        video.write(combined);

        if (iter % 10 == 0) {
            printf("Iter %3d: CPU cost=%.1f (%.2fms)  GPU cost=%.1f (%.2fms)\n",
                   iter, cpu_cost, cpu_ms, gpu_cost, gpu_ms);
        }
    }

    // Hold final frame
    cv::Mat left(H, W, CV_8UC3), right(H, W, CV_8UC3);
    draw_panel(left, cpu_stomp.trajectory, cpu_stomp.noisy, 0,
               "CPU (Final)", cpu_stomp.costs.empty() ? 0.0f : *min_element(cpu_stomp.costs.begin(), cpu_stomp.costs.end()),
               MAX_ITER, 0.0);
    draw_panel(right, gpu_stomp.h_trajectory, gpu_stomp.h_noisy, 0,
               "CUDA (Final)", gpu_stomp.h_costs.empty() ? 0.0f : *min_element(gpu_stomp.h_costs.begin(), gpu_stomp.h_costs.end()),
               MAX_ITER, 0.0);
    cv::Mat combined;
    cv::hconcat(left, right, combined);
    for (int i = 0; i < 30; i++) video.write(combined);

    video.release();
    cout << "Video saved to gif/comparison_stomp.avi" << endl;

    system("ffmpeg -y -i gif/comparison_stomp.avi "
           "-vf 'fps=15,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_stomp.gif 2>/dev/null");
    cout << "GIF saved to gif/comparison_stomp.gif" << endl;

    cv::imshow("comparison_stomp", combined);
    cv::waitKey(0);

    gpu_stomp.cleanup();

    return 0;
}
