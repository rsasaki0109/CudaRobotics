/*************************************************************************
    > File Name: mppi.cu
    > CUDA-parallelized MPPI (Model Predictive Path Integral) controller
    > Algorithm:
    >   1. Generate K=4096 sample control sequences with cuRAND noise
    >   2. rollout_kernel: each of K threads rolls out T=30 steps of bicycle model,
    >      accumulates running cost (goal, obstacle, control effort)
    >   3. compute_weights_kernel: softmin over K costs (exp(-1/lambda * cost))
    >   4. update_controls_kernel: weighted average of K control sequences per timestep
    >   5. Apply first control, shift horizon, repeat
    > Bicycle model: state (x, y, theta, v), control (accel, steer_rate), wheelbase L=2.5
    > CUDA kernels use one thread per sample trajectory
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
static const int K_SAMPLES    = 4096;     // number of sample trajectories
static const int T_HORIZON    = 30;       // prediction horizon steps
static const float DT         = 0.05f;    // timestep
static const float WHEELBASE  = 2.5f;     // bicycle model wheelbase
static const float LAMBDA     = 10.0f;    // temperature parameter
static const float WORKSPACE  = 50.0f;    // workspace size

// Control limits
static const float MAX_ACCEL       = 5.0f;
static const float MAX_STEER_RATE  = 1.0f;    // rad/s
static const float MAX_SPEED       = 8.0f;
static const float MAX_STEER       = 0.6f;    // max steering angle

// Noise standard deviations
static const float ACCEL_NOISE_STD = 2.0f;
static const float STEER_NOISE_STD = 0.4f;

// Cost weights
static const float GOAL_WEIGHT     = 1.0f;
static const float OBS_WEIGHT      = 200.0f;
static const float SPEED_WEIGHT    = 0.1f;
static const float STEER_WEIGHT    = 5.0f;
static const float TERMINAL_WEIGHT = 10.0f;

// Start and goal
static const float START_X = 5.0f,  START_Y = 5.0f;
static const float START_THETA = 0.0f, START_V = 0.0f;
static const float GOAL_X  = 45.0f, GOAL_Y  = 45.0f;

// Obstacles: (cx, cy, radius)
static const int N_OBSTACLES = 10;
__constant__ float d_obs_x[N_OBSTACLES];
__constant__ float d_obs_y[N_OBSTACLES];
__constant__ float d_obs_r[N_OBSTACLES];

static float h_obs_x[N_OBSTACLES] = {12.0f, 20.0f, 30.0f, 15.0f, 25.0f, 35.0f, 22.0f, 38.0f, 10.0f, 32.0f};
static float h_obs_y[N_OBSTACLES] = {15.0f, 25.0f, 10.0f, 35.0f, 18.0f, 30.0f, 40.0f, 20.0f, 30.0f, 38.0f};
static float h_obs_r[N_OBSTACLES] = { 3.0f,  3.5f,  3.0f,  2.5f,  3.5f,  2.5f,  3.0f,  3.0f,  2.5f,  2.5f};

static const int MAX_STEPS = 600;     // maximum simulation steps
static const float GOAL_TOL = 2.0f;   // goal tolerance

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
// Kernel: Rollout K trajectories with bicycle model
// Each thread rolls out one trajectory, accumulates cost
// Inputs:
//   state: (x, y, theta, v) - current robot state
//   d_nominal: [T * 2] nominal control sequence (accel, steer_rate) per step
// Outputs:
//   d_costs: [K] total cost for each sample
//   d_perturbed: [K * T * 2] perturbed control sequences
//   d_trajectories: [K * T * 4] rolled out states (for visualization)
// -------------------------------------------------------------------------
__global__ void rollout_kernel(
    float sx, float sy, float stheta, float sv,
    const float* __restrict__ d_nominal,       // [T * 2]
    float* __restrict__ d_costs,               // [K]
    float* __restrict__ d_perturbed,           // [K * T * 2]
    float* __restrict__ d_trajectories,        // [K * T * 4]
    curandState* __restrict__ d_rand_states,
    int K, int T, float dt, float L)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_state = d_rand_states[k];

    float x = sx, y = sy, theta = stheta, v = sv;
    float steer = 0.0f;  // current steering angle
    float cost = 0.0f;

    for (int t = 0; t < T; t++) {
        // Sample perturbed control
        float noise_a = curand_normal(&local_state) * ACCEL_NOISE_STD;
        float noise_s = curand_normal(&local_state) * STEER_NOISE_STD;

        float accel      = d_nominal[t * 2 + 0] + noise_a;
        float steer_rate = d_nominal[t * 2 + 1] + noise_s;

        // Clamp controls
        accel      = fminf(fmaxf(accel, -MAX_ACCEL), MAX_ACCEL);
        steer_rate = fminf(fmaxf(steer_rate, -MAX_STEER_RATE), MAX_STEER_RATE);

        // Store perturbed controls
        d_perturbed[k * T * 2 + t * 2 + 0] = accel;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer_rate;

        // Bicycle model integration
        steer += steer_rate * dt;
        steer = fminf(fmaxf(steer, -MAX_STEER), MAX_STEER);

        v += accel * dt;
        v = fminf(fmaxf(v, -1.0f), MAX_SPEED);

        x += v * cosf(theta) * dt;
        y += v * sinf(theta) * dt;
        theta += v / L * tanf(steer) * dt;

        // Store trajectory for visualization
        d_trajectories[k * T * 4 + t * 4 + 0] = x;
        d_trajectories[k * T * 4 + t * 4 + 1] = y;
        d_trajectories[k * T * 4 + t * 4 + 2] = theta;
        d_trajectories[k * T * 4 + t * 4 + 3] = v;

        // Running cost
        // Distance to goal
        float dx_g = x - GOAL_X;
        float dy_g = y - GOAL_Y;
        float dist_goal = sqrtf(dx_g * dx_g + dy_g * dy_g);
        cost += GOAL_WEIGHT * dist_goal * dt;

        // Obstacle cost
        for (int o = 0; o < N_OBSTACLES; o++) {
            float dx_o = x - d_obs_x[o];
            float dy_o = y - d_obs_y[o];
            float dist = sqrtf(dx_o * dx_o + dy_o * dy_o);
            float margin = dist - d_obs_r[o];
            if (margin < 0.0f) {
                cost += OBS_WEIGHT * 10.0f;
            } else if (margin < 2.0f) {
                float pen = 1.0f - margin / 2.0f;
                cost += OBS_WEIGHT * pen * pen;
            }
        }

        // Control effort
        cost += SPEED_WEIGHT * accel * accel * dt;
        cost += STEER_WEIGHT * steer_rate * steer_rate * dt;

        // Boundary cost
        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) {
            cost += 500.0f;
        }
    }

    // Terminal cost
    float dx_t = x - GOAL_X;
    float dy_t = y - GOAL_Y;
    cost += TERMINAL_WEIGHT * sqrtf(dx_t * dx_t + dy_t * dy_t);

    d_costs[k] = cost;
    d_rand_states[k] = local_state;
}

// -------------------------------------------------------------------------
// Kernel: Compute softmin weights from costs
// Two-pass: first find min cost, then compute exp weights
// -------------------------------------------------------------------------
__global__ void compute_weights_kernel(
    const float* __restrict__ d_costs,
    float* __restrict__ d_weights,
    float* __restrict__ d_min_cost,   // [1] scratch for min cost
    int K, float lambda)
{
    // Simple single-thread pass for now (K=4096 is small)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float min_c = FLT_MAX;
    for (int k = 0; k < K; k++) {
        if (d_costs[k] < min_c) min_c = d_costs[k];
    }
    d_min_cost[0] = min_c;

    float sum_exp = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = expf(-1.0f / lambda * (d_costs[k] - min_c));
        d_weights[k] = w;
        sum_exp += w;
    }

    if (sum_exp > 0.0f) {
        for (int k = 0; k < K; k++) {
            d_weights[k] /= sum_exp;
        }
    }
}

// -------------------------------------------------------------------------
// Kernel: Update nominal controls as weighted average of perturbed sequences
// One thread per timestep
// -------------------------------------------------------------------------
__global__ void update_controls_kernel(
    float* __restrict__ d_nominal,             // [T * 2] - updated in place
    const float* __restrict__ d_perturbed,     // [K * T * 2]
    const float* __restrict__ d_weights,       // [K]
    int K, int T)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float wa = 0.0f, ws = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = d_weights[k];
        wa += w * d_perturbed[k * T * 2 + t * 2 + 0];
        ws += w * d_perturbed[k * T * 2 + t * 2 + 1];
    }

    d_nominal[t * 2 + 0] = wa;
    d_nominal[t * 2 + 1] = ws;
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
    cout << "CUDA MPPI: Model Predictive Path Integral Controller" << endl;
    cout << "K = " << K_SAMPLES << " samples, T = " << T_HORIZON
         << " steps, dt = " << DT << "s" << endl;

    // Copy obstacle data to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_x, h_obs_x, N_OBSTACLES * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_y, h_obs_y, N_OBSTACLES * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_r, h_obs_r, N_OBSTACLES * sizeof(float)));

    // Robot state
    float rx = START_X, ry = START_Y, rtheta = START_THETA, rv = START_V;
    float rsteer = 0.0f;

    // Nominal control sequence [T * 2] initialized to zero
    int ctrl_size = T_HORIZON * 2;
    vector<float> h_nominal(ctrl_size, 0.0f);

    // Allocate device memory
    float *d_nominal, *d_costs, *d_weights, *d_perturbed, *d_trajectories, *d_min_cost;
    curandState *d_rand_states;

    CUDA_CHECK(cudaMalloc(&d_nominal, ctrl_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_costs, K_SAMPLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, K_SAMPLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_perturbed, K_SAMPLES * ctrl_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trajectories, K_SAMPLES * T_HORIZON * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_min_cost, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rand_states, K_SAMPLES * sizeof(curandState)));

    // Init cuRAND
    int block = 256;
    int grid_rand = (K_SAMPLES + block - 1) / block;
    init_curand_kernel<<<grid_rand, block>>>(d_rand_states, K_SAMPLES, 42ULL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Visualization setup
    int IMG_SIZE = 800;
    cv::VideoWriter video(
        "gif/mppi.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 20,
        cv::Size(IMG_SIZE, IMG_SIZE));

    if (!video.isOpened()) {
        cerr << "Failed to open video writer" << endl;
        return 1;
    }

    // Host buffers for visualization
    int vis_K = 200;  // subsample for drawing
    vector<float> h_costs(K_SAMPLES);
    vector<float> h_trajectories(K_SAMPLES * T_HORIZON * 4);

    // Path history
    vector<float> path_x, path_y;
    path_x.push_back(rx);
    path_y.push_back(ry);

    int grid_K = (K_SAMPLES + block - 1) / block;
    int grid_T = (T_HORIZON + block - 1) / block;

    // ===================== MPPI control loop =====================
    for (int step = 0; step < MAX_STEPS; step++) {

        // Check goal reached
        float dx_g = rx - GOAL_X;
        float dy_g = ry - GOAL_Y;
        if (sqrtf(dx_g * dx_g + dy_g * dy_g) < GOAL_TOL) {
            printf("Goal reached at step %d!\n", step);
            break;
        }

        // Copy nominal controls to device
        CUDA_CHECK(cudaMemcpy(d_nominal, h_nominal.data(), ctrl_size * sizeof(float), cudaMemcpyHostToDevice));

        // 1. Rollout K trajectories
        rollout_kernel<<<grid_K, block>>>(
            rx, ry, rtheta, rv,
            d_nominal, d_costs, d_perturbed, d_trajectories, d_rand_states,
            K_SAMPLES, T_HORIZON, DT, WHEELBASE);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2. Compute weights (softmin)
        compute_weights_kernel<<<1, 1>>>(d_costs, d_weights, d_min_cost, K_SAMPLES, LAMBDA);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3. Update nominal controls
        update_controls_kernel<<<grid_T, block>>>(d_nominal, d_perturbed, d_weights, K_SAMPLES, T_HORIZON);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy back updated nominal
        CUDA_CHECK(cudaMemcpy(h_nominal.data(), d_nominal, ctrl_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Copy costs and trajectories for visualization
        CUDA_CHECK(cudaMemcpy(h_costs.data(), d_costs, K_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_trajectories.data(), d_trajectories,
                              K_SAMPLES * T_HORIZON * 4 * sizeof(float), cudaMemcpyDeviceToHost));

        // Apply first control to robot
        float accel      = h_nominal[0];
        float steer_rate = h_nominal[1];

        accel      = fminf(fmaxf(accel, -MAX_ACCEL), MAX_ACCEL);
        steer_rate = fminf(fmaxf(steer_rate, -MAX_STEER_RATE), MAX_STEER_RATE);

        rsteer += steer_rate * DT;
        rsteer = fminf(fmaxf(rsteer, -MAX_STEER), MAX_STEER);
        rv += accel * DT;
        rv = fminf(fmaxf(rv, -1.0f), MAX_SPEED);
        rx += rv * cosf(rtheta) * DT;
        ry += rv * sinf(rtheta) * DT;
        rtheta += rv / WHEELBASE * tanf(rsteer) * DT;

        path_x.push_back(rx);
        path_y.push_back(ry);

        // Shift nominal controls (warm start)
        for (int t = 0; t < T_HORIZON - 1; t++) {
            h_nominal[t * 2 + 0] = h_nominal[(t + 1) * 2 + 0];
            h_nominal[t * 2 + 1] = h_nominal[(t + 1) * 2 + 1];
        }
        h_nominal[(T_HORIZON - 1) * 2 + 0] = 0.0f;
        h_nominal[(T_HORIZON - 1) * 2 + 1] = 0.0f;

        // --- Visualization ---
        cv::Mat img(IMG_SIZE, IMG_SIZE, CV_8UC3, cv::Scalar(255, 255, 255));

        // Draw obstacles
        for (int o = 0; o < N_OBSTACLES; o++) {
            cv::Point center = world_to_pixel(h_obs_x[o], h_obs_y[o], IMG_SIZE, WORKSPACE);
            int r = (int)(h_obs_r[o] / WORKSPACE * (float)IMG_SIZE);
            cv::circle(img, center, r, cv::Scalar(40, 40, 40), -1);
            cv::circle(img, center, (int)((h_obs_r[o] + 2.0f) / WORKSPACE * (float)IMG_SIZE),
                       cv::Scalar(200, 200, 200), 1);
        }

        // Find min/max cost for color mapping
        float min_cost = FLT_MAX, max_cost = -FLT_MAX;
        for (int k = 0; k < K_SAMPLES; k++) {
            if (h_costs[k] < min_cost) min_cost = h_costs[k];
            if (h_costs[k] > max_cost) max_cost = h_costs[k];
        }
        float cost_range = max_cost - min_cost;
        if (cost_range < 1e-6f) cost_range = 1.0f;

        // Draw subsampled trajectories colored by cost (green=low, red=high)
        int stride = max(1, K_SAMPLES / vis_K);
        for (int ki = 0; ki < K_SAMPLES; ki += stride) {
            float norm_cost = (h_costs[ki] - min_cost) / cost_range;
            norm_cost = fminf(fmaxf(norm_cost, 0.0f), 1.0f);

            // Green to red gradient
            int r_col = (int)(norm_cost * 255.0f);
            int g_col = (int)((1.0f - norm_cost) * 255.0f);
            cv::Scalar color(0, g_col, r_col);

            int base = ki * T_HORIZON * 4;
            for (int t = 0; t < T_HORIZON - 1; t++) {
                cv::Point p1 = world_to_pixel(h_trajectories[base + t * 4 + 0],
                                               h_trajectories[base + t * 4 + 1], IMG_SIZE, WORKSPACE);
                cv::Point p2 = world_to_pixel(h_trajectories[base + (t + 1) * 4 + 0],
                                               h_trajectories[base + (t + 1) * 4 + 1], IMG_SIZE, WORKSPACE);
                cv::line(img, p1, p2, color, 1);
            }
        }

        // Draw best trajectory (lowest cost) as thick blue
        int best_k = 0;
        for (int k = 1; k < K_SAMPLES; k++) {
            if (h_costs[k] < h_costs[best_k]) best_k = k;
        }
        {
            int base = best_k * T_HORIZON * 4;
            for (int t = 0; t < T_HORIZON - 1; t++) {
                cv::Point p1 = world_to_pixel(h_trajectories[base + t * 4 + 0],
                                               h_trajectories[base + t * 4 + 1], IMG_SIZE, WORKSPACE);
                cv::Point p2 = world_to_pixel(h_trajectories[base + (t + 1) * 4 + 0],
                                               h_trajectories[base + (t + 1) * 4 + 1], IMG_SIZE, WORKSPACE);
                cv::line(img, p1, p2, cv::Scalar(255, 100, 0), 3);
            }
        }

        // Draw path history
        for (int i = 1; i < (int)path_x.size(); i++) {
            cv::Point p1 = world_to_pixel(path_x[i - 1], path_y[i - 1], IMG_SIZE, WORKSPACE);
            cv::Point p2 = world_to_pixel(path_x[i], path_y[i], IMG_SIZE, WORKSPACE);
            cv::line(img, p1, p2, cv::Scalar(150, 150, 0), 2);
        }

        // Draw robot as red rectangle
        {
            float half_len = 1.5f;
            float half_wid = 0.8f;
            float cs = cosf(rtheta), sn = sinf(rtheta);
            cv::Point2f corners[4];
            float lx[4] = {-half_len, half_len, half_len, -half_len};
            float ly[4] = {-half_wid, -half_wid, half_wid, half_wid};
            for (int i = 0; i < 4; i++) {
                float wx = rx + cs * lx[i] - sn * ly[i];
                float wy = ry + sn * lx[i] + cs * ly[i];
                corners[i] = cv::Point2f(
                    wx / WORKSPACE * (float)IMG_SIZE,
                    IMG_SIZE - 1 - wy / WORKSPACE * (float)IMG_SIZE);
            }
            vector<cv::Point> pts;
            for (int i = 0; i < 4; i++) pts.push_back(cv::Point((int)corners[i].x, (int)corners[i].y));
            cv::fillConvexPoly(img, pts, cv::Scalar(0, 0, 220));
        }

        // Draw goal
        cv::circle(img, world_to_pixel(GOAL_X, GOAL_Y, IMG_SIZE, WORKSPACE),
                   10, cv::Scalar(0, 180, 0), -1);
        cv::circle(img, world_to_pixel(GOAL_X, GOAL_Y, IMG_SIZE, WORKSPACE),
                   12, cv::Scalar(0, 120, 0), 2);

        // Draw start
        cv::circle(img, world_to_pixel(START_X, START_Y, IMG_SIZE, WORKSPACE),
                   8, cv::Scalar(200, 100, 0), -1);

        // Labels
        char buf[256];
        snprintf(buf, sizeof(buf), "MPPI (CUDA K=%d T=%d)  Step %d  v=%.1f",
                 K_SAMPLES, T_HORIZON, step, rv);
        cv::putText(img, buf, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 0), 2);
        snprintf(buf, sizeof(buf), "Min cost: %.1f  Dist to goal: %.1f",
                 min_cost, sqrtf(dx_g * dx_g + dy_g * dy_g));
        cv::putText(img, buf, cv::Point(10, 58),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(80, 80, 80), 1);

        cv::namedWindow("mppi", cv::WINDOW_AUTOSIZE);
        cv::imshow("mppi", img);
        cv::waitKey(1);
        video.write(img);

        if (step % 20 == 0) {
            printf("Step %3d: pos=(%.1f, %.1f) v=%.1f steer=%.2f min_cost=%.1f\n",
                   step, rx, ry, rv, rsteer, min_cost);
        }
    }

    // Hold final frame
    cv::Mat final_img(IMG_SIZE, IMG_SIZE, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int o = 0; o < N_OBSTACLES; o++) {
        cv::Point center = world_to_pixel(h_obs_x[o], h_obs_y[o], IMG_SIZE, WORKSPACE);
        int r = (int)(h_obs_r[o] / WORKSPACE * (float)IMG_SIZE);
        cv::circle(final_img, center, r, cv::Scalar(40, 40, 40), -1);
    }
    for (int i = 1; i < (int)path_x.size(); i++) {
        cv::Point p1 = world_to_pixel(path_x[i - 1], path_y[i - 1], IMG_SIZE, WORKSPACE);
        cv::Point p2 = world_to_pixel(path_x[i], path_y[i], IMG_SIZE, WORKSPACE);
        cv::line(final_img, p1, p2, cv::Scalar(255, 100, 0), 3);
    }
    cv::circle(final_img, world_to_pixel(START_X, START_Y, IMG_SIZE, WORKSPACE),
               8, cv::Scalar(200, 100, 0), -1);
    cv::circle(final_img, world_to_pixel(GOAL_X, GOAL_Y, IMG_SIZE, WORKSPACE),
               10, cv::Scalar(0, 180, 0), -1);
    cv::putText(final_img, "MPPI - Final Path", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    for (int i = 0; i < 40; i++) video.write(final_img);

    video.release();
    cout << "Video saved to gif/mppi.avi" << endl;

    // Convert to GIF
    system("ffmpeg -y -i gif/mppi.avi "
           "-vf 'fps=20,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/mppi.gif 2>/dev/null");
    cout << "GIF saved to gif/mppi.gif" << endl;

    cv::imshow("mppi", final_img);
    cv::waitKey(0);

    // Cleanup
    CUDA_CHECK(cudaFree(d_nominal));
    CUDA_CHECK(cudaFree(d_costs));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_perturbed));
    CUDA_CHECK(cudaFree(d_trajectories));
    CUDA_CHECK(cudaFree(d_min_cost));
    CUDA_CHECK(cudaFree(d_rand_states));

    return 0;
}
