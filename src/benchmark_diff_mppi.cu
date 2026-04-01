/*************************************************************************
    Benchmark: MPPI vs Diff-MPPI
    - Runs multiple navigation scenarios across configurable sample sweeps
    - Compares sampling-only MPPI against gradient-refined variants
    - Supports both fixed-budget and cap-based wall-clock analyses downstream
    - Writes per-episode CSV to build/benchmark_diff_mppi.csv by default
 ************************************************************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "diff_cost.cuh"
#include "diff_dynamics.cuh"

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;
using namespace cudabot;

static const float WORKSPACE = 50.0f;
static const int MAX_OBSTACLES = 16;
static const float DEFAULT_LAMBDA = 8.0f;
static const int DEFAULT_T_HORIZON = 30;
static const int BENCH_WARMUP_ITERS = 4;

__constant__ Obstacle d_obstacles_bench[MAX_OBSTACLES];

struct Scenario {
    string name;
    float start_x = 5.0f;
    float start_y = 5.0f;
    float start_theta = 0.0f;
    float start_v = 0.0f;
    float goal_tol = 2.0f;
    int max_steps = 220;
    BicycleParams params;
    CostParams cost_params;
    int n_obs = 0;
    Obstacle obstacles[MAX_OBSTACLES];
};

struct PlannerVariant {
    string name;
    bool use_gradient = false;
    int grad_steps = 0;
    float alpha = 0.0f;
};

struct EpisodeMetrics {
    string scenario;
    string planner;
    int seed = 0;
    int k_samples = 0;
    int t_horizon = 0;
    int grad_steps = 0;
    float alpha = 0.0f;
    int reached_goal = 0;
    int collision_free = 0;
    int success = 0;
    int steps = 0;
    float final_distance = 0.0f;
    float min_goal_distance = 0.0f;
    float cumulative_cost = 0.0f;
    int collisions = 0;
    float avg_control_ms = 0.0f;
    float total_control_ms = 0.0f;
    float episode_ms = 0.0f;
    long long sample_budget = 0;
};

struct SummaryStats {
    int episodes = 0;
    int successes = 0;
    float sum_steps = 0.0f;
    float sum_final_distance = 0.0f;
    float sum_min_goal_distance = 0.0f;
    float sum_cumulative_cost = 0.0f;
    float sum_avg_control_ms = 0.0f;
    float sum_total_control_ms = 0.0f;
    float sum_collisions = 0.0f;
};

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
    curandState* d_rng,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int K,
    int T)
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

        float dxg = x - cost_params.goal_x;
        float dyg = y - cost_params.goal_y;
        total_cost += cost_params.goal_weight * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;
        total_cost += cost_params.control_weight * (accel * accel + steer * steer) * params.dt;
        float desired_heading = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
        float heading_err = theta - desired_heading;
        total_cost += cost_params.heading_weight * heading_err * heading_err * params.dt;
        float speed_err = v - cost_params.target_speed;
        total_cost += cost_params.speed_weight * speed_err * speed_err * params.dt;

        for (int i = 0; i < n_obs; i++) {
            float dx = x - d_obstacles_bench[i].x;
            float dy = y - d_obstacles_bench[i].y;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_bench[i].r;
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

__device__ void stage_cost_grad(
    float x, float y, float theta, float v, float accel, float steer,
    const CostParams& cp, int n_obs, float grad[6])
{
    for (int var = 0; var < 6; var++) {
        Dualf dx = (var == 0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dy = (var == 1) ? Dualf::variable(y) : Dualf::constant(y);
        Dualf dtheta = (var == 2) ? Dualf::variable(theta) : Dualf::constant(theta);
        Dualf dv = (var == 3) ? Dualf::variable(v) : Dualf::constant(v);
        Dualf da = (var == 4) ? Dualf::variable(accel) : Dualf::constant(accel);
        Dualf ds = (var == 5) ? Dualf::variable(steer) : Dualf::constant(steer);

        Dualf cost = goal_cost_diff(dx, dy, cp.goal_x, cp.goal_y, cp.goal_weight)
                   + obstacle_cost_diff(dx, dy, d_obstacles_bench, n_obs, cp.obs_influence, cp.obs_weight)
                   + control_cost_diff(da, ds, cp.control_weight)
                   + speed_cost_diff(dv, cp.target_speed, cp.speed_weight)
                   + heading_cost_diff(dx, dy, dtheta, cp.goal_x, cp.goal_y, cp.heading_weight);
        grad[var] = cost.deriv;
    }
}

__global__ void compute_gradient_kernel(
    const float* d_states, const float* d_nominal, float* d_grad,
    BicycleParams params, CostParams cost_params, int n_obs, int T)
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
        stage_cost_grad(x, y, theta, v, accel, steer, cost_params, n_obs, stage_grad);

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

static float host_step_cost(
    float x, float y, float theta, float v, float accel, float steer,
    const Scenario& scenario)
{
    const CostParams& cp = scenario.cost_params;
    float dxg = x - cp.goal_x;
    float dyg = y - cp.goal_y;
    float cost = cp.goal_weight * sqrtf(dxg * dxg + dyg * dyg + 0.01f);
    cost += cp.control_weight * (accel * accel + steer * steer);
    float desired_heading = atan2f(cp.goal_y - y, cp.goal_x - x);
    float heading_err = theta - desired_heading;
    cost += cp.heading_weight * heading_err * heading_err;
    float speed_err = v - cp.target_speed;
    cost += cp.speed_weight * speed_err * speed_err;

    for (int i = 0; i < scenario.n_obs; i++) {
        float dx = x - scenario.obstacles[i].x;
        float dy = y - scenario.obstacles[i].y;
        float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - scenario.obstacles[i].r;
        if (margin <= 0.1f) cost += cp.obs_weight * 100.0f;
        else if (margin < cp.obs_influence) cost += cp.obs_weight / (margin * margin);
    }

    if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) cost += 500.0f;
    return cost;
}

static float min_obstacle_margin(float x, float y, const Scenario& scenario) {
    float best = 1.0e9f;
    for (int i = 0; i < scenario.n_obs; i++) {
        float dx = x - scenario.obstacles[i].x;
        float dy = y - scenario.obstacles[i].y;
        float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - scenario.obstacles[i].r;
        best = std::min(best, margin);
    }
    return best;
}

class EpisodeRunner {
public:
    EpisodeRunner(const PlannerVariant& variant, const Scenario& scenario, int k_samples, int t_horizon, int seed)
        : variant_(variant), scenario_(scenario), k_samples_(k_samples), t_horizon_(t_horizon), seed_(seed) {
        reset_state();

        h_nominal_.assign(t_horizon_ * 2, 0.0f);
        h_costs_.assign(k_samples_, 0.0f);
        h_grad_.assign(t_horizon_ * 2, 0.0f);
        h_states_.assign((t_horizon_ + 1) * 4, 0.0f);

        CUDA_CHECK(cudaMalloc(&d_nominal_, h_nominal_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs_, h_costs_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_, k_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perturbed_, k_samples_ * t_horizon_ * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_states_, h_states_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_, h_grad_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_, k_samples_ * sizeof(curandState)));

        reset_rng();
    }

    ~EpisodeRunner() {
        CUDA_CHECK(cudaFree(d_nominal_));
        CUDA_CHECK(cudaFree(d_costs_));
        CUDA_CHECK(cudaFree(d_weights_));
        CUDA_CHECK(cudaFree(d_perturbed_));
        CUDA_CHECK(cudaFree(d_states_));
        CUDA_CHECK(cudaFree(d_grad_));
        CUDA_CHECK(cudaFree(d_rng_));
    }

    EpisodeMetrics run() {
        reset_state();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        warmup_controller();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        reset_rng();

        auto episode_begin = chrono::steady_clock::now();
        float total_control_ms = 0.0f;

        for (int step = 0; step < scenario_.max_steps; step++) {
            float goal_dx = rx_ - scenario_.cost_params.goal_x;
            float goal_dy = ry_ - scenario_.cost_params.goal_y;
            float goal_dist = sqrtf(goal_dx * goal_dx + goal_dy * goal_dy);
            min_goal_distance_ = std::min(min_goal_distance_, goal_dist);
            if (goal_dist < scenario_.goal_tol) {
                reached_goal_ = true;
                steps_taken_ = step;
                break;
            }

            auto t0 = chrono::steady_clock::now();
            controller_update(rx_, ry_, rtheta_, rv_);
            auto t1 = chrono::steady_clock::now();
            float control_ms = chrono::duration<float, milli>(t1 - t0).count();
            total_control_ms += control_ms;

            CUDA_CHECK(cudaMemcpy(h_nominal_.data(), d_nominal_, h_nominal_.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_costs_.data(), d_costs_, h_costs_.size() * sizeof(float), cudaMemcpyDeviceToHost));

            float accel = h_nominal_[0];
            float steer = h_nominal_[1];
            bicycle_step(rx_, ry_, rtheta_, rv_, accel, steer, scenario_.params);
            cumulative_cost_ += host_step_cost(rx_, ry_, rtheta_, rv_, accel, steer, scenario_);

            float margin = min_obstacle_margin(rx_, ry_, scenario_);
            if (margin <= 0.0f || rx_ < 0.0f || rx_ > WORKSPACE || ry_ < 0.0f || ry_ > WORKSPACE) collisions_++;

            for (int t = 0; t < t_horizon_ - 1; t++) {
                h_nominal_[t * 2 + 0] = h_nominal_[(t + 1) * 2 + 0];
                h_nominal_[t * 2 + 1] = h_nominal_[(t + 1) * 2 + 1];
            }
            h_nominal_[(t_horizon_ - 1) * 2 + 0] = 0.0f;
            h_nominal_[(t_horizon_ - 1) * 2 + 1] = 0.0f;
            steps_taken_ = step + 1;
        }

        auto episode_end = chrono::steady_clock::now();
        float final_dx = rx_ - scenario_.cost_params.goal_x;
        float final_dy = ry_ - scenario_.cost_params.goal_y;
        float final_distance = sqrtf(final_dx * final_dx + final_dy * final_dy);
        if (final_distance < scenario_.goal_tol) reached_goal_ = true;

        EpisodeMetrics metrics;
        metrics.scenario = scenario_.name;
        metrics.planner = variant_.name;
        metrics.seed = seed_;
        metrics.k_samples = k_samples_;
        metrics.t_horizon = t_horizon_;
        metrics.grad_steps = variant_.grad_steps;
        metrics.alpha = variant_.alpha;
        metrics.reached_goal = reached_goal_ ? 1 : 0;
        metrics.collision_free = collisions_ == 0 ? 1 : 0;
        metrics.success = (metrics.reached_goal && metrics.collision_free) ? 1 : 0;
        metrics.steps = steps_taken_;
        metrics.final_distance = final_distance;
        metrics.min_goal_distance = min_goal_distance_;
        metrics.cumulative_cost = cumulative_cost_;
        metrics.collisions = collisions_;
        metrics.total_control_ms = total_control_ms;
        metrics.avg_control_ms = steps_taken_ > 0 ? total_control_ms / steps_taken_ : 0.0f;
        metrics.episode_ms = chrono::duration<float, milli>(episode_end - episode_begin).count();
        metrics.sample_budget = static_cast<long long>(steps_taken_) * k_samples_ * t_horizon_;
        return metrics;
    }

private:
    void reset_rng() {
        int block = 256;
        init_curand_kernel<<<(k_samples_ + block - 1) / block, block>>>(d_rng_, k_samples_, static_cast<unsigned long long>(seed_));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void controller_update(float sx, float sy, float stheta, float sv) {
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size() * sizeof(float), cudaMemcpyHostToDevice));
        int block = 256;
        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
            sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rng_,
            scenario_.params, scenario_.cost_params, scenario_.n_obs, k_samples_, t_horizon_);
        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, DEFAULT_LAMBDA);
        update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
            d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);

        if (variant_.use_gradient) {
            for (int gs = 0; gs < variant_.grad_steps; gs++) {
                rollout_nominal_kernel<<<1, 1>>>(sx, sy, stheta, sv, d_nominal_, d_states_, scenario_.params, t_horizon_);
                compute_gradient_kernel<<<1, 1>>>(d_states_, d_nominal_, d_grad_, scenario_.params, scenario_.cost_params, scenario_.n_obs, t_horizon_);
                gradient_step_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                    d_nominal_, d_grad_, t_horizon_, variant_.alpha, scenario_.params.max_steer);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void warmup_controller() {
        for (int iter = 0; iter < BENCH_WARMUP_ITERS; iter++) {
            controller_update(scenario_.start_x, scenario_.start_y, scenario_.start_theta, scenario_.start_v);
        }
    }

    void reset_state() {
        rx_ = scenario_.start_x;
        ry_ = scenario_.start_y;
        rtheta_ = scenario_.start_theta;
        rv_ = scenario_.start_v;
        steps_taken_ = 0;
        collisions_ = 0;
        reached_goal_ = false;
        cumulative_cost_ = 0.0f;
        min_goal_distance_ = sqrtf((rx_ - scenario_.cost_params.goal_x) * (rx_ - scenario_.cost_params.goal_x)
                                 + (ry_ - scenario_.cost_params.goal_y) * (ry_ - scenario_.cost_params.goal_y));
    }

    PlannerVariant variant_;
    Scenario scenario_;
    int k_samples_;
    int t_horizon_;
    int seed_;

    float rx_ = 0.0f;
    float ry_ = 0.0f;
    float rtheta_ = 0.0f;
    float rv_ = 0.0f;
    int steps_taken_ = 0;
    int collisions_ = 0;
    bool reached_goal_ = false;
    float cumulative_cost_ = 0.0f;
    float min_goal_distance_ = 0.0f;

    vector<float> h_nominal_;
    vector<float> h_costs_;
    vector<float> h_grad_;
    vector<float> h_states_;

    float* d_nominal_ = nullptr;
    float* d_costs_ = nullptr;
    float* d_weights_ = nullptr;
    float* d_perturbed_ = nullptr;
    float* d_states_ = nullptr;
    float* d_grad_ = nullptr;
    curandState* d_rng_ = nullptr;
};

static Scenario make_cluttered_scene() {
    Scenario s;
    s.name = "cluttered";
    s.start_x = 5.0f;
    s.start_y = 5.0f;
    s.cost_params.goal_x = 45.0f;
    s.cost_params.goal_y = 45.0f;
    s.cost_params.goal_weight = 5.0f;
    s.cost_params.control_weight = 0.1f;
    s.cost_params.speed_weight = 0.15f;
    s.cost_params.target_speed = 3.5f;
    s.cost_params.heading_weight = 0.35f;
    s.cost_params.obs_weight = 10.0f;
    s.cost_params.obs_influence = 5.0f;
    s.cost_params.terminal_weight = 8.0f;
    const Obstacle obs[] = {
        {12.0f, 15.0f, 3.0f}, {20.0f, 25.0f, 3.5f}, {30.0f, 10.0f, 3.0f},
        {15.0f, 35.0f, 2.5f}, {25.0f, 18.0f, 3.5f}, {35.0f, 30.0f, 2.5f},
        {22.0f, 40.0f, 3.0f}, {38.0f, 20.0f, 3.0f}, {10.0f, 30.0f, 2.5f},
        {32.0f, 38.0f, 2.5f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    return s;
}

static Scenario make_narrow_passage_scene() {
    Scenario s;
    s.name = "narrow_passage";
    s.start_x = 4.0f;
    s.start_y = 8.0f;
    s.cost_params.goal_x = 46.0f;
    s.cost_params.goal_y = 42.0f;
    s.max_steps = 260;
    s.cost_params.target_speed = 3.0f;
    s.cost_params.obs_weight = 14.0f;
    s.cost_params.obs_influence = 5.5f;
    const Obstacle obs[] = {
        {22.0f, 6.0f, 2.3f}, {23.0f, 12.0f, 2.3f}, {24.0f, 18.0f, 2.3f},
        {26.0f, 32.0f, 2.3f}, {27.0f, 38.0f, 2.3f}, {28.0f, 44.0f, 2.3f},
        {36.0f, 24.0f, 2.8f}, {14.0f, 26.0f, 2.8f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    return s;
}

static Scenario make_slalom_scene() {
    Scenario s;
    s.name = "slalom";
    s.start_x = 4.0f;
    s.start_y = 6.0f;
    s.cost_params.goal_x = 46.0f;
    s.cost_params.goal_y = 44.0f;
    s.max_steps = 240;
    s.cost_params.target_speed = 3.6f;
    s.cost_params.obs_weight = 11.0f;
    const Obstacle obs[] = {
        {10.0f, 14.0f, 2.7f}, {16.0f, 32.0f, 2.8f}, {22.0f, 14.0f, 2.8f},
        {28.0f, 33.0f, 2.8f}, {34.0f, 15.0f, 2.8f}, {40.0f, 33.0f, 2.8f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    return s;
}

static Scenario make_corner_scene() {
    Scenario s;
    s.name = "corner_turn";
    s.start_x = 6.0f;
    s.start_y = 6.0f;
    s.cost_params.goal_x = 44.0f;
    s.cost_params.goal_y = 44.0f;
    s.max_steps = 240;
    s.cost_params.target_speed = 3.2f;
    s.cost_params.obs_weight = 13.0f;
    const Obstacle obs[] = {
        {18.0f, 12.0f, 3.0f}, {24.0f, 12.0f, 3.0f}, {30.0f, 12.0f, 3.0f},
        {30.0f, 18.0f, 3.0f}, {30.0f, 24.0f, 3.0f}, {30.0f, 30.0f, 3.0f},
        {18.0f, 30.0f, 2.6f}, {12.0f, 24.0f, 2.6f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    return s;
}

static void ensure_build_dir() {
    mkdir("build", 0755);
}

static vector<int> parse_int_list(const string& text) {
    vector<int> values;
    string token;
    stringstream ss(text);
    while (getline(ss, token, ',')) {
        if (token.empty()) continue;
        values.push_back(std::max(1, atoi(token.c_str())));
    }
    sort(values.begin(), values.end());
    values.erase(unique(values.begin(), values.end()), values.end());
    return values;
}

static void write_csv(const vector<EpisodeMetrics>& rows, const string& path) {
    ofstream out(path);
    out << "scenario,planner,seed,k_samples,t_horizon,grad_steps,alpha,reached_goal,collision_free,success,steps,final_distance,min_goal_distance,cumulative_cost,collisions,avg_control_ms,total_control_ms,episode_ms,sample_budget\n";
    for (const auto& r : rows) {
        out << r.scenario << ','
            << r.planner << ','
            << r.seed << ','
            << r.k_samples << ','
            << r.t_horizon << ','
            << r.grad_steps << ','
            << r.alpha << ','
            << r.reached_goal << ','
            << r.collision_free << ','
            << r.success << ','
            << r.steps << ','
            << r.final_distance << ','
            << r.min_goal_distance << ','
            << r.cumulative_cost << ','
            << r.collisions << ','
            << r.avg_control_ms << ','
            << r.total_control_ms << ','
            << r.episode_ms << ','
            << r.sample_budget << '\n';
    }
}

static void print_summary(const vector<EpisodeMetrics>& rows) {
    map<string, SummaryStats> stats;
    for (const auto& r : rows) {
        string key = r.scenario + " | " + r.planner + " | K=" + to_string(r.k_samples);
        auto& s = stats[key];
        s.episodes++;
        s.successes += r.success;
        s.sum_steps += r.steps;
        s.sum_final_distance += r.final_distance;
        s.sum_min_goal_distance += r.min_goal_distance;
        s.sum_cumulative_cost += r.cumulative_cost;
        s.sum_avg_control_ms += r.avg_control_ms;
        s.sum_total_control_ms += r.total_control_ms;
        s.sum_collisions += r.collisions;
    }

    cout << "=== benchmark_diff_mppi summary ===" << endl;
    for (const auto& kv : stats) {
        const SummaryStats& s = kv.second;
        float n = static_cast<float>(s.episodes);
        printf("%s : success=%.2f steps=%.1f final_dist=%.2f min_dist=%.2f cost=%.1f avg_ms=%.2f collisions=%.2f\n",
               kv.first.c_str(),
               s.successes / n,
               s.sum_steps / n,
               s.sum_final_distance / n,
               s.sum_min_goal_distance / n,
               s.sum_cumulative_cost / n,
               s.sum_avg_control_ms / n,
               s.sum_collisions / n);
    }
}

int main(int argc, char** argv) {
    bool quick = false;
    string csv_path = "build/benchmark_diff_mppi.csv";
    vector<int> k_values;
    int seed_count = -1;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--quick") quick = true;
        else if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
        else if (arg == "--k-values" && i + 1 < argc) k_values = parse_int_list(argv[++i]);
        else if (arg == "--seed-count" && i + 1 < argc) seed_count = std::max(1, atoi(argv[++i]));
    }

    ensure_build_dir();

    vector<Scenario> scenarios;
    scenarios.push_back(make_cluttered_scene());
    scenarios.push_back(make_narrow_passage_scene());
    if (!quick) {
        scenarios.push_back(make_slalom_scene());
        scenarios.push_back(make_corner_scene());
    }

    vector<PlannerVariant> variants;
    variants.push_back({"mppi", false, 0, 0.0f});
    variants.push_back({"diff_mppi_1", true, 1, 0.010f});
    variants.push_back({"diff_mppi_3", true, 3, 0.006f});

    if (k_values.empty()) k_values = quick ? vector<int>{1024, 4096} : vector<int>{1024, 2048, 4096};
    if (seed_count <= 0) seed_count = quick ? 2 : 4;

    vector<EpisodeMetrics> rows;
    rows.reserve(scenarios.size() * variants.size() * k_values.size() * seed_count);

    for (size_t si = 0; si < scenarios.size(); si++) {
        const Scenario& scenario = scenarios[si];
        CUDA_CHECK(cudaMemcpyToSymbol(d_obstacles_bench, scenario.obstacles, sizeof(Obstacle) * scenario.n_obs));
        for (int k_samples : k_values) {
            for (size_t vi = 0; vi < variants.size(); vi++) {
                const PlannerVariant& variant = variants[vi];
                for (int seed = 0; seed < seed_count; seed++) {
                    int run_seed = static_cast<int>(1000 + si * 100 + vi * 20 + seed * 7 + k_samples);
                    EpisodeRunner runner(variant, scenario, k_samples, DEFAULT_T_HORIZON, run_seed);
                    EpisodeMetrics metrics = runner.run();
                    rows.push_back(metrics);
                    printf("[%s] %s K=%d seed=%d success=%d steps=%d final_dist=%.2f avg_ms=%.2f collisions=%d\n",
                           scenario.name.c_str(), variant.name.c_str(), k_samples, seed,
                           metrics.success, metrics.steps, metrics.final_distance,
                           metrics.avg_control_ms, metrics.collisions);
                }
            }
        }
    }

    write_csv(rows, csv_path);
    print_summary(rows);
    cout << "CSV saved to " << csv_path << endl;
    return 0;
}
