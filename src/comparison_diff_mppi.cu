/*************************************************************************
    Comparison: Standard MPPI vs Differentiable MPPI
    Left : sampling-only MPPI
    Right: MPPI + gradient refinement
    Output: gif/comparison_diff_mppi.gif
 ************************************************************************/

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "diff_cost.cuh"
#include "diff_dynamics.cuh"

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;
using namespace cudabot;

static const int K_SAMPLES = 4096;
static const int T_HORIZON = 30;
static const int MAX_STEPS = 450;
static const float WORKSPACE = 50.0f;
static const float START_X = 5.0f;
static const float START_Y = 5.0f;
static const float START_THETA = 0.0f;
static const float START_V = 0.0f;
static const float GOAL_TOL = 2.0f;
static const float LAMBDA = 8.0f;
static const float GRAD_ALPHA = 0.01f;
static const int N_OBSTACLES = 10;

static const Obstacle h_obstacles[N_OBSTACLES] = {
    {12.0f, 15.0f, 3.0f}, {20.0f, 25.0f, 3.5f}, {30.0f, 10.0f, 3.0f},
    {15.0f, 35.0f, 2.5f}, {25.0f, 18.0f, 3.5f}, {35.0f, 30.0f, 2.5f},
    {22.0f, 40.0f, 3.0f}, {38.0f, 20.0f, 3.0f}, {10.0f, 30.0f, 2.5f},
    {32.0f, 38.0f, 2.5f}
};

__constant__ Obstacle d_obstacles_cmp[N_OBSTACLES];

__global__ void init_curand_kernel(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void rollout_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal,
    float* d_costs,
    float* d_perturbed,
    float* d_trajectories,
    curandState* d_rng,
    BicycleParams params,
    CostParams cost_params,
    int K, int T)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float x = sx;
    float y = sy;
    float theta = stheta;
    float v = sv;
    float total_cost = 0.0f;

    for (int t = 0; t < T; t++) {
        float accel = d_nominal[t * 2 + 0] + curand_normal(&local_rng) * 1.5f;
        float steer = d_nominal[t * 2 + 1] + curand_normal(&local_rng) * 0.18f;
        accel = clampf(accel, -4.0f, 4.0f);
        steer = clampf(steer, -params.max_steer, params.max_steer);

        d_perturbed[k * T * 2 + t * 2 + 0] = accel;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer;

        bicycle_step(x, y, theta, v, accel, steer, params);

        d_trajectories[k * T * 4 + t * 4 + 0] = x;
        d_trajectories[k * T * 4 + t * 4 + 1] = y;
        d_trajectories[k * T * 4 + t * 4 + 2] = theta;
        d_trajectories[k * T * 4 + t * 4 + 3] = v;

        float dxg = x - cost_params.goal_x;
        float dyg = y - cost_params.goal_y;
        total_cost += cost_params.goal_weight * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;
        total_cost += (accel * accel + steer * steer) * cost_params.control_weight * params.dt;
        float desired_heading = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
        float heading_err = theta - desired_heading;
        total_cost += cost_params.heading_weight * heading_err * heading_err * params.dt;
        float speed_err = v - cost_params.target_speed;
        total_cost += cost_params.speed_weight * speed_err * speed_err * params.dt;
        for (int i = 0; i < N_OBSTACLES; i++) {
            float dx = x - d_obstacles_cmp[i].x;
            float dy = y - d_obstacles_cmp[i].y;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_cmp[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }
        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) total_cost += 500.0f;
    }

    float dx = x - cost_params.goal_x;
    float dy = y - cost_params.goal_y;
    total_cost += cost_params.terminal_weight * sqrtf(dx * dx + dy * dy + 0.01f);
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

__global__ void compute_weights_kernel(const float* d_costs, float* d_weights, int K, float lambda) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float min_cost = FLT_MAX;
    for (int k = 0; k < K; k++) min_cost = fminf(min_cost, d_costs[k]);

    float sum_w = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = expf(-(d_costs[k] - min_cost) / lambda);
        d_weights[k] = w;
        sum_w += w;
    }
    if (sum_w > 0.0f) {
        for (int k = 0; k < K; k++) d_weights[k] /= sum_w;
    }
}

__global__ void update_controls_kernel(float* d_nominal, const float* d_perturbed, const float* d_weights, int K, int T) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float accel = 0.0f;
    float steer = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = d_weights[k];
        accel += w * d_perturbed[k * T * 2 + t * 2 + 0];
        steer += w * d_perturbed[k * T * 2 + t * 2 + 1];
    }
    d_nominal[t * 2 + 0] = accel;
    d_nominal[t * 2 + 1] = steer;
}

__global__ void rollout_nominal_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal, float* d_states,
    BicycleParams params, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float x = sx;
    float y = sy;
    float theta = stheta;
    float v = sv;
    d_states[0] = x;
    d_states[1] = y;
    d_states[2] = theta;
    d_states[3] = v;

    for (int t = 0; t < T; t++) {
        bicycle_step(x, y, theta, v, d_nominal[t * 2 + 0], d_nominal[t * 2 + 1], params);
        d_states[(t + 1) * 4 + 0] = x;
        d_states[(t + 1) * 4 + 1] = y;
        d_states[(t + 1) * 4 + 2] = theta;
        d_states[(t + 1) * 4 + 3] = v;
    }
}

__device__ void terminal_grad(float x, float y, const CostParams& cp, float grad[4]) {
    for (int i = 0; i < 4; i++) grad[i] = 0.0f;
    for (int var = 0; var < 4; var++) {
        Dualf dx = (var == 0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dy = (var == 1) ? Dualf::variable(y) : Dualf::constant(y);
        Dualf cost = goal_cost_diff(dx, dy, cp.goal_x, cp.goal_y, cp.terminal_weight);
        grad[var] = cost.deriv;
    }
}

__device__ void stage_cost_grad(float x, float y, float theta, float v, float accel, float steer, const CostParams& cp, float grad[6]) {
    for (int var = 0; var < 6; var++) {
        Dualf dx = (var == 0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dy = (var == 1) ? Dualf::variable(y) : Dualf::constant(y);
        Dualf dtheta = (var == 2) ? Dualf::variable(theta) : Dualf::constant(theta);
        Dualf dv = (var == 3) ? Dualf::variable(v) : Dualf::constant(v);
        Dualf da = (var == 4) ? Dualf::variable(accel) : Dualf::constant(accel);
        Dualf ds = (var == 5) ? Dualf::variable(steer) : Dualf::constant(steer);
        (void)dtheta;
        (void)dv;

        Dualf cost = goal_cost_diff(dx, dy, cp.goal_x, cp.goal_y, cp.goal_weight)
                   + obstacle_cost_diff(dx, dy, d_obstacles_cmp, N_OBSTACLES, cp.obs_influence, cp.obs_weight)
                   + control_cost_diff(da, ds, cp.control_weight)
                   + speed_cost_diff(dv, cp.target_speed, cp.speed_weight)
                   + heading_cost_diff(dx, dy, dtheta, cp.goal_x, cp.goal_y, cp.heading_weight);
        grad[var] = cost.deriv;
    }
}

__global__ void compute_gradient_kernel(
    const float* d_states, const float* d_nominal, float* d_grad,
    BicycleParams params, CostParams cost_params, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float adj[4];
    terminal_grad(d_states[T * 4 + 0], d_states[T * 4 + 1], cost_params, adj);

    for (int t = T - 1; t >= 0; t--) {
        float x = d_states[t * 4 + 0];
        float y = d_states[t * 4 + 1];
        float theta = d_states[t * 4 + 2];
        float v = d_states[t * 4 + 3];
        float accel = d_nominal[t * 2 + 0];
        float steer = d_nominal[t * 2 + 1];
        float J[4][6];
        float stage_grad[6];
        float next_adj[4];

        bicycle_jacobian(x, y, theta, v, accel, steer, params, J);
        stage_cost_grad(x, y, theta, v, accel, steer, cost_params, stage_grad);

        d_grad[t * 2 + 0] = stage_grad[4];
        d_grad[t * 2 + 1] = stage_grad[5];
        for (int row = 0; row < 4; row++) {
            d_grad[t * 2 + 0] += J[row][4] * adj[row];
            d_grad[t * 2 + 1] += J[row][5] * adj[row];
        }

        for (int col = 0; col < 4; col++) {
            next_adj[col] = stage_grad[col];
            for (int row = 0; row < 4; row++) next_adj[col] += J[row][col] * adj[row];
        }
        for (int i = 0; i < 4; i++) adj[i] = next_adj[i];
    }
}

__global__ void gradient_step_kernel(float* d_nominal, const float* d_grad, int T, float alpha, float max_steer) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    d_nominal[t * 2 + 0] = clampf(d_nominal[t * 2 + 0] - alpha * d_grad[t * 2 + 0], -4.0f, 4.0f);
    d_nominal[t * 2 + 1] = clampf(d_nominal[t * 2 + 1] - alpha * d_grad[t * 2 + 1], -max_steer, max_steer);
}

static cv::Point world_to_pixel(float x, float y, int size) {
    int px = static_cast<int>(x / WORKSPACE * size);
    int py = size - 1 - static_cast<int>(y / WORKSPACE * size);
    return cv::Point(px, py);
}

struct Planner {
    bool use_gradient;
    BicycleParams params;
    CostParams cost_params;
    float rx, ry, rtheta, rv;
    float *d_nominal, *d_costs, *d_weights, *d_perturbed, *d_trajectories, *d_states, *d_grad;
    curandState* d_rng;
    vector<float> h_nominal;
    vector<float> h_costs;
    vector<float> h_trajectories;
    vector<float> h_states;
    vector<float> h_grad;
    vector<float> path_x, path_y, cost_history;
    bool reached = false;

    void init(bool enable_gradient, unsigned long long seed) {
        use_gradient = enable_gradient;
        rx = START_X;
        ry = START_Y;
        rtheta = START_THETA;
        rv = START_V;
        h_nominal.assign(T_HORIZON * 2, 0.0f);
        h_costs.resize(K_SAMPLES);
        h_trajectories.resize(K_SAMPLES * T_HORIZON * 4);
        h_states.resize((T_HORIZON + 1) * 4);
        h_grad.resize(T_HORIZON * 2, 0.0f);
        path_x.assign(1, rx);
        path_y.assign(1, ry);

        CUDA_CHECK(cudaMalloc(&d_nominal, h_nominal.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs, h_costs.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights, K_SAMPLES * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perturbed, K_SAMPLES * T_HORIZON * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_trajectories, h_trajectories.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_states, h_states.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad, h_grad.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng, K_SAMPLES * sizeof(curandState)));
        init_curand_kernel<<<(K_SAMPLES + 255) / 256, 256>>>(d_rng, K_SAMPLES, seed);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void step() {
        if (reached) return;

        float goal_dx = rx - cost_params.goal_x;
        float goal_dy = ry - cost_params.goal_y;
        if (sqrtf(goal_dx * goal_dx + goal_dy * goal_dy) < GOAL_TOL) {
            reached = true;
            return;
        }

        CUDA_CHECK(cudaMemcpy(d_nominal, h_nominal.data(), h_nominal.size() * sizeof(float), cudaMemcpyHostToDevice));
        rollout_kernel<<<(K_SAMPLES + 255) / 256, 256>>>(
            rx, ry, rtheta, rv, d_nominal, d_costs, d_perturbed, d_trajectories, d_rng,
            params, cost_params, K_SAMPLES, T_HORIZON);
        compute_weights_kernel<<<1, 1>>>(d_costs, d_weights, K_SAMPLES, LAMBDA);
        update_controls_kernel<<<(T_HORIZON + 255) / 256, 256>>>(d_nominal, d_perturbed, d_weights, K_SAMPLES, T_HORIZON);
        if (use_gradient) {
            rollout_nominal_kernel<<<1, 1>>>(rx, ry, rtheta, rv, d_nominal, d_states, params, T_HORIZON);
            compute_gradient_kernel<<<1, 1>>>(d_states, d_nominal, d_grad, params, cost_params, T_HORIZON);
            gradient_step_kernel<<<(T_HORIZON + 255) / 256, 256>>>(d_nominal, d_grad, T_HORIZON, GRAD_ALPHA, params.max_steer);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_nominal.data(), d_nominal, h_nominal.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_costs.data(), d_costs, h_costs.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_trajectories.data(), d_trajectories, h_trajectories.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (use_gradient) {
            CUDA_CHECK(cudaMemcpy(h_states.data(), d_states, h_states.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_grad.data(), d_grad, h_grad.size() * sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            fill(h_grad.begin(), h_grad.end(), 0.0f);
        }

        bicycle_step(rx, ry, rtheta, rv, h_nominal[0], h_nominal[1], params);
        path_x.push_back(rx);
        path_y.push_back(ry);
        cost_history.push_back(*min_element(h_costs.begin(), h_costs.end()));

        for (int t = 0; t < T_HORIZON - 1; t++) {
            h_nominal[t * 2 + 0] = h_nominal[(t + 1) * 2 + 0];
            h_nominal[t * 2 + 1] = h_nominal[(t + 1) * 2 + 1];
        }
        h_nominal[(T_HORIZON - 1) * 2 + 0] = 0.0f;
        h_nominal[(T_HORIZON - 1) * 2 + 1] = 0.0f;
    }

    void cleanup() {
        CUDA_CHECK(cudaFree(d_nominal));
        CUDA_CHECK(cudaFree(d_costs));
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_perturbed));
        CUDA_CHECK(cudaFree(d_trajectories));
        CUDA_CHECK(cudaFree(d_states));
        CUDA_CHECK(cudaFree(d_grad));
        CUDA_CHECK(cudaFree(d_rng));
    }
};

static void draw_cost_plot(cv::Mat& img, const vector<float>& history) {
    if (history.size() < 2) return;
    int x0 = img.cols - 150;
    int y0 = 20;
    int w = 130;
    int h = 80;
    cv::rectangle(img, cv::Rect(x0, y0, w, h), cv::Scalar(245, 245, 245), -1);
    cv::rectangle(img, cv::Rect(x0, y0, w, h), cv::Scalar(180, 180, 180), 1);

    float min_c = *min_element(history.begin(), history.end());
    float max_c = *max_element(history.begin(), history.end());
    float range = fmaxf(1e-6f, max_c - min_c);
    for (size_t i = 1; i < history.size(); i++) {
        int x1 = x0 + static_cast<int>((i - 1) * (w - 10) / float(max<size_t>(1, history.size() - 1))) + 5;
        int x2 = x0 + static_cast<int>(i * (w - 10) / float(max<size_t>(1, history.size() - 1))) + 5;
        int y1 = y0 + h - 5 - static_cast<int>((history[i - 1] - min_c) / range * (h - 10));
        int y2 = y0 + h - 5 - static_cast<int>((history[i] - min_c) / range * (h - 10));
        cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 80, 220), 2);
    }
}

static void draw_panel(cv::Mat& img, const Planner& planner, const char* title, int step) {
    img.setTo(cv::Scalar(255, 255, 255));
    for (int i = 0; i < N_OBSTACLES; i++) {
        cv::Point c = world_to_pixel(h_obstacles[i].x, h_obstacles[i].y, img.cols);
        int r = static_cast<int>(h_obstacles[i].r / WORKSPACE * img.cols);
        cv::circle(img, c, r, cv::Scalar(40, 40, 40), -1);
    }

    float min_cost = planner.h_costs.empty() ? 0.0f : *min_element(planner.h_costs.begin(), planner.h_costs.end());
    float max_cost = planner.h_costs.empty() ? 1.0f : *max_element(planner.h_costs.begin(), planner.h_costs.end());
    float range = fmaxf(1e-6f, max_cost - min_cost);
    int stride = max(1, K_SAMPLES / 120);
    for (int k = 0; k < K_SAMPLES; k += stride) {
        float nc = (planner.h_costs[k] - min_cost) / range;
        cv::Scalar color(0, static_cast<int>((1.0f - nc) * 255.0f), static_cast<int>(nc * 255.0f));
        int base = k * T_HORIZON * 4;
        for (int t = 0; t < T_HORIZON - 1; t++) {
            cv::line(img,
                     world_to_pixel(planner.h_trajectories[base + t * 4 + 0], planner.h_trajectories[base + t * 4 + 1], img.cols),
                     world_to_pixel(planner.h_trajectories[base + (t + 1) * 4 + 0], planner.h_trajectories[base + (t + 1) * 4 + 1], img.cols),
                     color, 1);
        }
    }

    if (planner.use_gradient) {
        for (int t = 0; t < T_HORIZON - 1; t++) {
            cv::line(img,
                     world_to_pixel(planner.h_states[t * 4 + 0], planner.h_states[t * 4 + 1], img.cols),
                     world_to_pixel(planner.h_states[(t + 1) * 4 + 0], planner.h_states[(t + 1) * 4 + 1], img.cols),
                     cv::Scalar(255, 100, 0), 3);
        }
    }

    for (size_t i = 1; i < planner.path_x.size(); i++) {
        cv::line(img,
                 world_to_pixel(planner.path_x[i - 1], planner.path_y[i - 1], img.cols),
                 world_to_pixel(planner.path_x[i], planner.path_y[i], img.cols),
                 cv::Scalar(150, 150, 0), 2);
    }

    cv::circle(img, world_to_pixel(planner.cost_params.goal_x, planner.cost_params.goal_y, img.cols), 7, cv::Scalar(0, 180, 0), -1);
    cv::circle(img, world_to_pixel(START_X, START_Y, img.cols), 6, cv::Scalar(180, 120, 0), -1);
    cv::circle(img, world_to_pixel(planner.rx, planner.ry, img.cols), 7, cv::Scalar(0, 0, 220), -1);

    char buf[256];
    cv::putText(img, title, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 2);
    snprintf(buf, sizeof(buf), "step=%d cost=%.1f", step, min_cost);
    cv::putText(img, buf, cv::Point(8, 44), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(70, 70, 70), 1);
    snprintf(buf, sizeof(buf), "grad(a0,s0)=(%.3f, %.3f)", planner.h_grad[0], planner.h_grad[1]);
    cv::putText(img, buf, cv::Point(8, 62), cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(70, 70, 70), 1);
    draw_cost_plot(img, planner.cost_history);
}

int main() {
    cout << "Comparison: Standard MPPI vs Differentiable MPPI" << endl;
    CUDA_CHECK(cudaMemcpyToSymbol(d_obstacles_cmp, h_obstacles, sizeof(h_obstacles)));

    Planner standard;
    Planner diff;
    standard.init(false, 2025ULL);
    diff.init(true, 2026ULL);

    int panel = 400;
    cv::VideoWriter video(
        "gif/comparison_diff_mppi.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 20, cv::Size(panel * 2, panel));
    if (!video.isOpened()) {
        cerr << "Failed to open video writer" << endl;
        return 1;
    }

    for (int step = 0; step < MAX_STEPS; step++) {
        standard.step();
        diff.step();

        cv::Mat left(panel, panel, CV_8UC3);
        cv::Mat right(panel, panel, CV_8UC3);
        draw_panel(left, standard, "MPPI (sampling only)", step);
        draw_panel(right, diff, "Diff-MPPI", step);

        cv::Mat frame;
        cv::hconcat(left, right, frame);
        video.write(frame);
        cv::imshow("comparison_diff_mppi", frame);
        cv::waitKey(1);

        float sdx = standard.rx - standard.cost_params.goal_x;
        float sdy = standard.ry - standard.cost_params.goal_y;
        float ddx = diff.rx - diff.cost_params.goal_x;
        float ddy = diff.ry - diff.cost_params.goal_y;
        if (step % 20 == 0) {
            printf("Step %3d  standard_dist=%.2f diff_dist=%.2f\n",
                   step, sqrtf(sdx * sdx + sdy * sdy), sqrtf(ddx * ddx + ddy * ddy));
        }
        if (standard.reached && diff.reached) break;
    }

    video.release();
    system("ffmpeg -y -i gif/comparison_diff_mppi.avi "
           "-vf 'fps=15,scale=400:-1' -loop 0 "
           "gif/comparison_diff_mppi.gif 2>/dev/null");

    standard.cleanup();
    diff.cleanup();
    return 0;
}
