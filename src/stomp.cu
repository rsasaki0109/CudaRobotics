/*************************************************************************
    > File Name: stomp.cu
    > CUDA-parallelized STOMP (Stochastic Trajectory Optimization for Motion Planning)
    > Algorithm:
    >   1. Generate K noisy trajectories by adding correlated noise
    >   2. Evaluate cost for each trajectory (obstacle + smoothness)
    >   3. Weight trajectories by exp(-cost/lambda)
    >   4. Update trajectory as cost-weighted average
    >   5. Repeat for max_iterations
    > CUDA kernels:
    >   - generate_noisy_trajectories_kernel: K threads, each generates one noisy trajectory
    >   - compute_costs_kernel: K threads, each evaluates cost for one trajectory
    >   - update_trajectory_kernel: N_WAYPOINTS threads, each computes weighted avg for one waypoint
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cfloat>
#include <algorithm>
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
static const int K_SAMPLES     = 10000;    // number of noisy trajectory samples
static const int N_WAYPOINTS   = 50;       // number of waypoints
static const int DIM           = 2;        // 2D workspace
static const float LAMBDA      = 10.0f;    // temperature parameter
static const int MAX_ITER      = 100;      // max optimization iterations
static const float WORKSPACE   = 50.0f;    // workspace size
static const float NOISE_STD   = 2.0f;     // noise standard deviation
static const float OBS_COST_WEIGHT    = 100.0f;  // obstacle cost weight
static const float SMOOTH_COST_WEIGHT = 1.0f;    // smoothness cost weight
static const float OBS_CLEARANCE      = 1.5f;    // clearance around obstacles

// Start and goal
static const float START_X = 5.0f,  START_Y = 5.0f;
static const float GOAL_X  = 45.0f, GOAL_Y  = 45.0f;

// Obstacles: (cx, cy, radius)
static const int N_OBSTACLES = 8;
__constant__ float d_obs_x[N_OBSTACLES];
__constant__ float d_obs_y[N_OBSTACLES];
__constant__ float d_obs_r[N_OBSTACLES];

// Host-side obstacle data
static float h_obs_x[N_OBSTACLES] = {10.0f, 20.0f, 30.0f, 15.0f, 25.0f, 35.0f, 22.0f, 38.0f};
static float h_obs_y[N_OBSTACLES] = {15.0f, 25.0f, 10.0f, 35.0f, 18.0f, 30.0f, 40.0f, 20.0f};
static float h_obs_r[N_OBSTACLES] = { 3.0f,  4.0f,  3.5f,  3.0f,  3.5f,  2.5f,  3.0f,  3.0f};

// -------------------------------------------------------------------------
// Kernel: Generate K noisy trajectories
// Each thread generates one trajectory by adding correlated Gaussian noise
// to the current trajectory. Noise is smoothed by a simple running average
// to create correlation (approximating covariance from finite differences).
// -------------------------------------------------------------------------
__global__ void generate_noisy_trajectories_kernel(
    const float* __restrict__ d_trajectory,   // current trajectory [N_WAYPOINTS * DIM]
    float* __restrict__ d_noisy,              // output [K * N_WAYPOINTS * DIM]
    curandState* __restrict__ d_rand_states,
    int K, int N, int dim, float noise_std)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_state = d_rand_states[k];

    // Generate raw noise and apply simple smoothing (running average of 5)
    float raw_x[N_WAYPOINTS];
    float raw_y[N_WAYPOINTS];

    for (int i = 0; i < N; i++) {
        raw_x[i] = curand_normal(&local_state) * noise_std;
        raw_y[i] = curand_normal(&local_state) * noise_std;
    }

    // Simple smoothing pass (correlated noise approximation)
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

        // Add smoothed noise to current trajectory
        d_noisy[base + i * dim + 0] = d_trajectory[i * dim + 0] + sx;
        d_noisy[base + i * dim + 1] = d_trajectory[i * dim + 1] + sy;
    }

    // Fix start and goal
    d_noisy[base + 0]               = d_trajectory[0];
    d_noisy[base + 1]               = d_trajectory[1];
    d_noisy[base + (N - 1) * dim + 0] = d_trajectory[(N - 1) * dim + 0];
    d_noisy[base + (N - 1) * dim + 1] = d_trajectory[(N - 1) * dim + 1];

    d_rand_states[k] = local_state;
}

// -------------------------------------------------------------------------
// Kernel: Compute cost for each noisy trajectory
// Cost = obstacle_cost + smoothness_cost
// -------------------------------------------------------------------------
__global__ void compute_costs_kernel(
    const float* __restrict__ d_noisy,  // [K * N_WAYPOINTS * DIM]
    float* __restrict__ d_costs,        // [K]
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

        // Obstacle cost: penalize proximity to obstacles
        for (int o = 0; o < N_OBSTACLES; o++) {
            float dx = wx - d_obs_x[o];
            float dy = wy - d_obs_y[o];
            float dist = sqrtf(dx * dx + dy * dy);
            float margin = dist - d_obs_r[o];
            if (margin < 0.0f) {
                // Inside obstacle: very high cost
                cost += obs_weight * 10.0f;
            } else if (margin < clearance) {
                // Near obstacle: smooth penalty
                float t = 1.0f - margin / clearance;
                cost += obs_weight * t * t;
            }
        }

        // Smoothness cost: finite difference acceleration
        if (i >= 1 && i < N - 1) {
            float ax = d_noisy[base + (i + 1) * dim + 0] - 2.0f * wx + d_noisy[base + (i - 1) * dim + 0];
            float ay = d_noisy[base + (i + 1) * dim + 1] - 2.0f * wy + d_noisy[base + (i - 1) * dim + 1];
            cost += smooth_weight * (ax * ax + ay * ay);
        }
    }

    d_costs[k] = cost;
}

// -------------------------------------------------------------------------
// Kernel: Initialize cuRAND states
// -------------------------------------------------------------------------
__global__ void init_curand_kernel(curandState* states, int K, unsigned long long seed)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curand_init(seed, k, 0, &states[k]);
}

// -------------------------------------------------------------------------
// Kernel: Update trajectory as cost-weighted average
// Each thread handles one waypoint
// -------------------------------------------------------------------------
__global__ void update_trajectory_kernel(
    float* __restrict__ d_trajectory,         // [N * DIM] - updated in place
    const float* __restrict__ d_noisy,        // [K * N * DIM]
    const float* __restrict__ d_weights,      // [K] - normalized weights
    int K, int N, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Skip start and goal waypoints
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
// Host: compute weights from costs (softmin)
// -------------------------------------------------------------------------
void compute_weights(const vector<float>& costs, vector<float>& weights, float lambda)
{
    int K = (int)costs.size();
    float min_cost = *min_element(costs.begin(), costs.end());

    // exp(-1/lambda * (cost - min_cost))
    float sum_exp = 0.0f;
    weights.resize(K);
    for (int k = 0; k < K; k++) {
        weights[k] = expf(-1.0f / lambda * (costs[k] - min_cost));
        sum_exp += weights[k];
    }

    // Normalize
    if (sum_exp > 0.0f) {
        for (int k = 0; k < K; k++) {
            weights[k] /= sum_exp;
        }
    }
}

// -------------------------------------------------------------------------
// Visualization helper
// -------------------------------------------------------------------------
static cv::Point world_to_pixel(float wx, float wy, int img_size, float ws)
{
    int px = (int)(wx / ws * (float)img_size);
    int py = img_size - 1 - (int)(wy / ws * (float)img_size);
    return cv::Point(px, py);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main()
{
    cout << "CUDA STOMP: Stochastic Trajectory Optimization for Motion Planning" << endl;
    cout << "K = " << K_SAMPLES << " samples, " << N_WAYPOINTS << " waypoints, "
         << MAX_ITER << " iterations" << endl;

    // Copy obstacle data to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_x, h_obs_x, N_OBSTACLES * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_y, h_obs_y, N_OBSTACLES * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_r, h_obs_r, N_OBSTACLES * sizeof(float)));

    // Initialize trajectory as straight line from start to goal
    int traj_size = N_WAYPOINTS * DIM;
    vector<float> h_trajectory(traj_size);
    for (int i = 0; i < N_WAYPOINTS; i++) {
        float t = (float)i / (float)(N_WAYPOINTS - 1);
        h_trajectory[i * DIM + 0] = START_X + t * (GOAL_X - START_X);
        h_trajectory[i * DIM + 1] = START_Y + t * (GOAL_Y - START_Y);
    }

    // Allocate device memory
    float *d_trajectory, *d_noisy, *d_costs, *d_weights;
    curandState *d_rand_states;

    CUDA_CHECK(cudaMalloc(&d_trajectory, traj_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_noisy, K_SAMPLES * traj_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_costs, K_SAMPLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, K_SAMPLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rand_states, K_SAMPLES * sizeof(curandState)));

    // Copy initial trajectory to device
    CUDA_CHECK(cudaMemcpy(d_trajectory, h_trajectory.data(), traj_size * sizeof(float), cudaMemcpyHostToDevice));

    // Init cuRAND
    int block = 256;
    int grid_rand = (K_SAMPLES + block - 1) / block;
    init_curand_kernel<<<grid_rand, block>>>(d_rand_states, K_SAMPLES, 42ULL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Visualization setup
    int IMG_SIZE = 600;
    cv::VideoWriter video(
        "gif/stomp.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 15,
        cv::Size(IMG_SIZE, IMG_SIZE));

    if (!video.isOpened()) {
        cerr << "Failed to open video writer" << endl;
        return 1;
    }

    vector<float> h_costs(K_SAMPLES);
    vector<float> h_weights(K_SAMPLES);
    vector<float> h_noisy(K_SAMPLES * traj_size);

    int grid_K = (K_SAMPLES + block - 1) / block;
    int grid_N = (N_WAYPOINTS + block - 1) / block;

    // ===================== STOMP iterations =====================
    for (int iter = 0; iter < MAX_ITER; iter++) {

        // 1. Generate noisy trajectories
        generate_noisy_trajectories_kernel<<<grid_K, block>>>(
            d_trajectory, d_noisy, d_rand_states,
            K_SAMPLES, N_WAYPOINTS, DIM, NOISE_STD);
        CUDA_CHECK(cudaGetLastError());

        // 2. Compute costs
        compute_costs_kernel<<<grid_K, block>>>(
            d_noisy, d_costs, K_SAMPLES, N_WAYPOINTS, DIM,
            OBS_COST_WEIGHT, SMOOTH_COST_WEIGHT, OBS_CLEARANCE);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3. Copy costs to host, compute weights
        CUDA_CHECK(cudaMemcpy(h_costs.data(), d_costs, K_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost));
        compute_weights(h_costs, h_weights, LAMBDA);

        // Copy weights to device
        CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), K_SAMPLES * sizeof(float), cudaMemcpyHostToDevice));

        // 4. Update trajectory
        update_trajectory_kernel<<<grid_N, block>>>(
            d_trajectory, d_noisy, d_weights,
            K_SAMPLES, N_WAYPOINTS, DIM);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy current trajectory and noisy samples for visualization
        CUDA_CHECK(cudaMemcpy(h_trajectory.data(), d_trajectory, traj_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Copy a subset of noisy trajectories for visualization (up to 500)
        int vis_K = min(K_SAMPLES, 500);
        CUDA_CHECK(cudaMemcpy(h_noisy.data(), d_noisy, vis_K * traj_size * sizeof(float), cudaMemcpyDeviceToHost));

        // --- Visualize ---
        cv::Mat img(IMG_SIZE, IMG_SIZE, CV_8UC3, cv::Scalar(255, 255, 255));

        // Draw obstacles
        for (int o = 0; o < N_OBSTACLES; o++) {
            cv::Point center = world_to_pixel(h_obs_x[o], h_obs_y[o], IMG_SIZE, WORKSPACE);
            int r = (int)(h_obs_r[o] / WORKSPACE * (float)IMG_SIZE);
            cv::circle(img, center, r, cv::Scalar(40, 40, 40), -1);
            cv::circle(img, center, (int)((h_obs_r[o] + OBS_CLEARANCE) / WORKSPACE * (float)IMG_SIZE),
                       cv::Scalar(180, 180, 180), 1);
        }

        // Draw noisy trajectories (thin gray lines)
        for (int k = 0; k < vis_K; k++) {
            int base = k * N_WAYPOINTS * DIM;
            for (int i = 0; i < N_WAYPOINTS - 1; i++) {
                cv::Point p1 = world_to_pixel(h_noisy[base + i * DIM + 0],
                                               h_noisy[base + i * DIM + 1], IMG_SIZE, WORKSPACE);
                cv::Point p2 = world_to_pixel(h_noisy[base + (i + 1) * DIM + 0],
                                               h_noisy[base + (i + 1) * DIM + 1], IMG_SIZE, WORKSPACE);
                cv::line(img, p1, p2, cv::Scalar(220, 220, 220), 1);
            }
        }

        // Draw current trajectory (thick red line)
        for (int i = 0; i < N_WAYPOINTS - 1; i++) {
            cv::Point p1 = world_to_pixel(h_trajectory[i * DIM + 0],
                                           h_trajectory[i * DIM + 1], IMG_SIZE, WORKSPACE);
            cv::Point p2 = world_to_pixel(h_trajectory[(i + 1) * DIM + 0],
                                           h_trajectory[(i + 1) * DIM + 1], IMG_SIZE, WORKSPACE);
            cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 3);
        }

        // Draw start and goal
        cv::circle(img, world_to_pixel(START_X, START_Y, IMG_SIZE, WORKSPACE),
                   8, cv::Scalar(0, 200, 0), -1);
        cv::circle(img, world_to_pixel(GOAL_X, GOAL_Y, IMG_SIZE, WORKSPACE),
                   8, cv::Scalar(255, 0, 0), -1);

        // Labels
        char buf[128];
        float min_cost = *min_element(h_costs.begin(), h_costs.end());
        snprintf(buf, sizeof(buf), "STOMP (CUDA K=%d)  Iter %d/%d  Cost: %.1f",
                 K_SAMPLES, iter + 1, MAX_ITER, min_cost);
        cv::putText(img, buf, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 2);

        cv::namedWindow("stomp", cv::WINDOW_AUTOSIZE);
        cv::imshow("stomp", img);
        cv::waitKey(1);
        video.write(img);

        if (iter % 10 == 0) {
            printf("Iter %3d: min_cost = %.2f\n", iter + 1, min_cost);
        }
    }

    // Hold final frame
    cv::Mat final_img(IMG_SIZE, IMG_SIZE, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int o = 0; o < N_OBSTACLES; o++) {
        cv::Point center = world_to_pixel(h_obs_x[o], h_obs_y[o], IMG_SIZE, WORKSPACE);
        int r = (int)(h_obs_r[o] / WORKSPACE * (float)IMG_SIZE);
        cv::circle(final_img, center, r, cv::Scalar(40, 40, 40), -1);
    }
    for (int i = 0; i < N_WAYPOINTS - 1; i++) {
        cv::Point p1 = world_to_pixel(h_trajectory[i * DIM + 0],
                                       h_trajectory[i * DIM + 1], IMG_SIZE, WORKSPACE);
        cv::Point p2 = world_to_pixel(h_trajectory[(i + 1) * DIM + 0],
                                       h_trajectory[(i + 1) * DIM + 1], IMG_SIZE, WORKSPACE);
        cv::line(final_img, p1, p2, cv::Scalar(0, 0, 255), 3);
    }
    cv::circle(final_img, world_to_pixel(START_X, START_Y, IMG_SIZE, WORKSPACE),
               8, cv::Scalar(0, 200, 0), -1);
    cv::circle(final_img, world_to_pixel(GOAL_X, GOAL_Y, IMG_SIZE, WORKSPACE),
               8, cv::Scalar(255, 0, 0), -1);
    cv::putText(final_img, "STOMP - Final Trajectory", cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    for (int i = 0; i < 30; i++) video.write(final_img);

    video.release();
    cout << "Video saved to gif/stomp.avi" << endl;

    // Convert to GIF
    system("ffmpeg -y -i gif/stomp.avi "
           "-vf 'fps=15,scale=600:-1:flags=lanczos' -loop 0 "
           "gif/stomp.gif 2>/dev/null");
    cout << "GIF saved to gif/stomp.gif" << endl;

    cv::imshow("stomp", final_img);
    cv::waitKey(0);

    // Cleanup
    CUDA_CHECK(cudaFree(d_trajectory));
    CUDA_CHECK(cudaFree(d_noisy));
    CUDA_CHECK(cudaFree(d_costs));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_rand_states));

    return 0;
}
