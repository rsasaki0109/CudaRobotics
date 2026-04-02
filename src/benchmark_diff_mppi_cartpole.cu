/*************************************************************************
    Diff-MPPI CartPole Benchmark
    - Compares sampling-only MPPI against gradient-refined variants
    - Uses nonlinear CartPole dynamics to partially close the "2D-only" gap
    - Writes CSV metrics compatible with the existing Diff-MPPI summary scripts
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
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "autodiff_engine.cuh"

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;
using namespace cudabot;

static const int DEFAULT_T_HORIZON = 35;
static const int DEFAULT_MAX_STEPS = 280;
static const float DEFAULT_LAMBDA = 2.0f;
static const float DEFAULT_NOISE_SIGMA = 0.40f;

struct EpisodeMetrics {
    string scenario;
    string planner;
    int seed = 0;
    int k_samples = 0;
    int t_horizon = 0;
    int grad_steps = 0;
    float alpha = 0.0f;
    int reached_goal = 0;
    int collision_free = 1;
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
    double steps_sum = 0.0;
    double final_sum = 0.0;
    double min_sum = 0.0;
    double cost_sum = 0.0;
    double ms_sum = 0.0;
};

struct CartPoleParams {
    float gravity = 9.8f;
    float masscart = 1.0f;
    float masspole = 0.1f;
    float length = 0.5f;
    float force_mag = 10.0f;
    float tau = 0.02f;
    float x_threshold = 2.4f;

    __host__ __device__ float total_mass() const { return masscart + masspole; }
    __host__ __device__ float polemass_length() const { return masspole * length; }
};

struct CartPoleCostParams {
    float angle_weight = 0.0f;
    float x_weight = 0.0f;
    float x_dot_weight = 0.0f;
    float theta_dot_weight = 0.0f;
    float action_weight = 0.0f;
    float terminal_angle_weight = 0.0f;
    float terminal_x_weight = 0.0f;
    float terminal_x_dot_weight = 0.0f;
    float terminal_theta_dot_weight = 0.0f;
    float out_of_bounds_penalty = 0.0f;
    float success_angle = 0.0f;
    float success_x = 0.0f;
    float success_x_dot = 0.0f;
    float success_theta_dot = 0.0f;
    int success_window = 0;
};

struct Scenario {
    string name;
    CartPoleParams params;
    CartPoleCostParams cost_params;
    int max_steps = DEFAULT_MAX_STEPS;
    float x_lo = 0.0f;
    float x_hi = 0.0f;
    float x_dot_lo = 0.0f;
    float x_dot_hi = 0.0f;
    float theta_lo = 0.0f;
    float theta_hi = 0.0f;
    float theta_dot_lo = 0.0f;
    float theta_dot_hi = 0.0f;
};

struct PlannerVariant {
    string name;
    bool use_gradient = false;
    int grad_steps = 0;
    float alpha = 0.0f;
    float noise_sigma = DEFAULT_NOISE_SIGMA;
};

__host__ __device__ inline float clampf_local(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__host__ __device__ inline float wrap_anglef(float theta) {
    return atan2f(sinf(theta), cosf(theta));
}

__host__ __device__ inline float stabilization_error(float x, float x_dot, float theta, float theta_dot) {
    float angle_err = wrap_anglef(theta);
    return sqrtf(angle_err * angle_err + 0.25f * x * x + 0.05f * x_dot * x_dot + 0.08f * theta_dot * theta_dot);
}

template <typename Scalar>
__host__ __device__ Scalar cartpole_angle_error_diff(const Scalar& theta) {
    return cudabot::atan2(cudabot::sin(theta), cudabot::cos(theta));
}

__host__ __device__ inline void cartpole_step(
    float& x, float& x_dot, float& theta, float& theta_dot, float action, const CartPoleParams& p)
{
    action = clampf_local(action, -1.0f, 1.0f);
    float force = action * p.force_mag;
    float costheta = cosf(theta);
    float sintheta = sinf(theta);
    float temp = (force + p.polemass_length() * theta_dot * theta_dot * sintheta) / p.total_mass();
    float thetaacc = (p.gravity * sintheta - costheta * temp) /
        (p.length * (4.0f / 3.0f - p.masspole * costheta * costheta / p.total_mass()));
    float xacc = temp - p.polemass_length() * thetaacc * costheta / p.total_mass();

    x += p.tau * x_dot;
    x_dot += p.tau * xacc;
    theta += p.tau * theta_dot;
    theta_dot += p.tau * thetaacc;
}

__device__ inline void cartpole_step_diff(
    Dualf& x, Dualf& x_dot, Dualf& theta, Dualf& theta_dot, Dualf action, const CartPoleParams& p)
{
    Dualf total_mass = Dualf::constant(p.total_mass());
    Dualf polemass_length = Dualf::constant(p.polemass_length());
    Dualf gravity = Dualf::constant(p.gravity);
    Dualf length = Dualf::constant(p.length);
    action = clamp(action, -1.0f, 1.0f);
    Dualf force = action * p.force_mag;
    Dualf costheta = cudabot::cos(theta);
    Dualf sintheta = cudabot::sin(theta);
    Dualf temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
    Dualf thetaacc = (gravity * sintheta - costheta * temp) /
        (length * (Dualf::constant(4.0f / 3.0f) - Dualf::constant(p.masspole) * costheta * costheta / total_mass));
    Dualf xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    x = x + p.tau * x_dot;
    x_dot = x_dot + p.tau * xacc;
    theta = theta + p.tau * theta_dot;
    theta_dot = theta_dot + p.tau * thetaacc;
}

__host__ __device__ inline float stage_cost(
    float x, float x_dot, float theta, float theta_dot, float action, const CartPoleCostParams& cp)
{
    float angle_err = wrap_anglef(theta);
    return cp.angle_weight * angle_err * angle_err
         + cp.x_weight * x * x
         + cp.x_dot_weight * x_dot * x_dot
         + cp.theta_dot_weight * theta_dot * theta_dot
         + cp.action_weight * action * action;
}

__host__ __device__ inline float terminal_cost(
    float x, float x_dot, float theta, float theta_dot, const CartPoleCostParams& cp)
{
    float angle_err = wrap_anglef(theta);
    return cp.terminal_angle_weight * angle_err * angle_err
         + cp.terminal_x_weight * x * x
         + cp.terminal_x_dot_weight * x_dot * x_dot
         + cp.terminal_theta_dot_weight * theta_dot * theta_dot;
}

__device__ inline void stage_cost_grad(
    float x, float x_dot, float theta, float theta_dot, float action, const CartPoleCostParams& cp, float grad[5])
{
    for (int var = 0; var < 5; var++) {
        Dualf dx = (var == 0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dxd = (var == 1) ? Dualf::variable(x_dot) : Dualf::constant(x_dot);
        Dualf dtheta = (var == 2) ? Dualf::variable(theta) : Dualf::constant(theta);
        Dualf dthetad = (var == 3) ? Dualf::variable(theta_dot) : Dualf::constant(theta_dot);
        Dualf du = (var == 4) ? Dualf::variable(action) : Dualf::constant(action);
        Dualf angle_err = cartpole_angle_error_diff(dtheta);
        Dualf cost = cp.angle_weight * angle_err * angle_err
                   + cp.x_weight * dx * dx
                   + cp.x_dot_weight * dxd * dxd
                   + cp.theta_dot_weight * dthetad * dthetad
                   + cp.action_weight * du * du;
        grad[var] = cost.deriv;
    }
}

__device__ inline void terminal_grad(
    float x, float x_dot, float theta, float theta_dot, const CartPoleCostParams& cp, float grad[4])
{
    for (int var = 0; var < 4; var++) {
        Dualf dx = (var == 0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dxd = (var == 1) ? Dualf::variable(x_dot) : Dualf::constant(x_dot);
        Dualf dtheta = (var == 2) ? Dualf::variable(theta) : Dualf::constant(theta);
        Dualf dthetad = (var == 3) ? Dualf::variable(theta_dot) : Dualf::constant(theta_dot);
        Dualf angle_err = cartpole_angle_error_diff(dtheta);
        Dualf cost = cp.terminal_angle_weight * angle_err * angle_err
                   + cp.terminal_x_weight * dx * dx
                   + cp.terminal_x_dot_weight * dxd * dxd
                   + cp.terminal_theta_dot_weight * dthetad * dthetad;
        grad[var] = cost.deriv;
    }
}

__device__ inline void cartpole_jacobian(
    float x, float x_dot, float theta, float theta_dot, float action, const CartPoleParams& p, float J[4][5])
{
    for (int col = 0; col < 5; col++) {
        Dualf dx = (col == 0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dxd = (col == 1) ? Dualf::variable(x_dot) : Dualf::constant(x_dot);
        Dualf dtheta = (col == 2) ? Dualf::variable(theta) : Dualf::constant(theta);
        Dualf dthetad = (col == 3) ? Dualf::variable(theta_dot) : Dualf::constant(theta_dot);
        Dualf du = (col == 4) ? Dualf::variable(action) : Dualf::constant(action);
        cartpole_step_diff(dx, dxd, dtheta, dthetad, du, p);
        J[0][col] = dx.deriv;
        J[1][col] = dxd.deriv;
        J[2][col] = dtheta.deriv;
        J[3][col] = dthetad.deriv;
    }
}

__global__ void init_curand_kernel(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void rollout_kernel(
    float sx, float sx_dot, float stheta, float stheta_dot,
    const float* d_nominal,
    float* d_costs,
    float* d_perturbed,
    curandState* d_rng,
    CartPoleParams params,
    CartPoleCostParams cost_params,
    float noise_sigma,
    int K,
    int T)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float x = sx;
    float x_dot = sx_dot;
    float theta = stheta;
    float theta_dot = stheta_dot;
    float total_cost = 0.0f;

    for (int t = 0; t < T; t++) {
        float action = d_nominal[t] + curand_normal(&local_rng) * noise_sigma;
        action = clampf_local(action, -1.0f, 1.0f);
        d_perturbed[k * T + t] = action;
        cartpole_step(x, x_dot, theta, theta_dot, action, params);
        total_cost += stage_cost(x, x_dot, theta, theta_dot, action, cost_params);
        if (fabsf(x) > params.x_threshold) {
            total_cost += cost_params.out_of_bounds_penalty;
            break;
        }
    }

    total_cost += terminal_cost(x, x_dot, theta, theta_dot, cost_params);
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

    float action = 0.0f;
    for (int k = 0; k < K; k++) action += d_weights[k] * d_perturbed[k * T + t];
    d_nominal[t] = clampf_local(action, -1.0f, 1.0f);
}

__global__ void rollout_nominal_kernel(
    float sx, float sx_dot, float stheta, float stheta_dot, const float* d_nominal, float* d_states,
    CartPoleParams params, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float x = sx;
    float x_dot = sx_dot;
    float theta = stheta;
    float theta_dot = stheta_dot;

    d_states[0] = x;
    d_states[1] = x_dot;
    d_states[2] = theta;
    d_states[3] = theta_dot;

    for (int t = 0; t < T; t++) {
        cartpole_step(x, x_dot, theta, theta_dot, d_nominal[t], params);
        d_states[(t + 1) * 4 + 0] = x;
        d_states[(t + 1) * 4 + 1] = x_dot;
        d_states[(t + 1) * 4 + 2] = theta;
        d_states[(t + 1) * 4 + 3] = theta_dot;
    }
}

__global__ void compute_gradient_kernel(
    const float* d_states, const float* d_nominal, float* d_grad,
    CartPoleParams params, CartPoleCostParams cost_params, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float adj[4];
    terminal_grad(
        d_states[T * 4 + 0], d_states[T * 4 + 1], d_states[T * 4 + 2], d_states[T * 4 + 3],
        cost_params, adj);

    for (int t = T - 1; t >= 0; t--) {
        float x = d_states[t * 4 + 0];
        float x_dot = d_states[t * 4 + 1];
        float theta = d_states[t * 4 + 2];
        float theta_dot = d_states[t * 4 + 3];
        float action = d_nominal[t];

        float J[4][5];
        float stage_grad_vec[5];
        float next_adj[4];

        cartpole_jacobian(x, x_dot, theta, theta_dot, action, params, J);
        stage_cost_grad(x, x_dot, theta, theta_dot, action, cost_params, stage_grad_vec);

        d_grad[t] = stage_grad_vec[4];
        for (int row = 0; row < 4; row++) d_grad[t] += J[row][4] * adj[row];

        for (int col = 0; col < 4; col++) {
            next_adj[col] = stage_grad_vec[col];
            for (int row = 0; row < 4; row++) next_adj[col] += J[row][col] * adj[row];
        }

        for (int i = 0; i < 4; i++) adj[i] = next_adj[i];
    }
}

__global__ void gradient_step_kernel(float* d_nominal, const float* d_grad, float alpha, int T) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    d_nominal[t] = clampf_local(d_nominal[t] - alpha * d_grad[t], -1.0f, 1.0f);
}

static vector<int> parse_int_list(const string& text) {
    vector<int> values;
    string token;
    for (char c : text) {
        if (c == ',') {
            if (!token.empty()) values.push_back(atoi(token.c_str()));
            token.clear();
        } else {
            token.push_back(c);
        }
    }
    if (!token.empty()) values.push_back(atoi(token.c_str()));
    return values;
}

static vector<string> parse_string_list(const string& text) {
    vector<string> values;
    string token;
    for (char c : text) {
        if (c == ',') {
            if (!token.empty()) values.push_back(token);
            token.clear();
        } else {
            token.push_back(c);
        }
    }
    if (!token.empty()) values.push_back(token);
    return values;
}

static Scenario make_cartpole_recover() {
    Scenario s;
    s.name = "cartpole_recover";
    s.max_steps = 220;
    s.x_lo = -0.65f; s.x_hi = 0.65f;
    s.x_dot_lo = -1.0f; s.x_dot_hi = 1.0f;
    s.theta_lo = -0.80f; s.theta_hi = 0.80f;
    s.theta_dot_lo = -1.5f; s.theta_dot_hi = 1.5f;
    s.cost_params = {
        8.5f, 0.85f, 0.14f, 0.40f, 0.02f,
        20.0f, 1.40f, 0.20f, 0.55f,
        900.0f,
        0.12f, 0.40f, 0.45f, 0.80f,
        18
    };
    return s;
}

static Scenario make_cartpole_large_angle() {
    Scenario s;
    s.name = "cartpole_large_angle";
    s.max_steps = 260;
    s.x_lo = -0.35f; s.x_hi = 0.35f;
    s.x_dot_lo = -0.45f; s.x_dot_hi = 0.45f;
    s.theta_lo = 1.35f; s.theta_hi = 2.35f;
    s.theta_dot_lo = -0.60f; s.theta_dot_hi = 0.60f;
    s.cost_params = {
        7.0f, 0.28f, 0.07f, 0.16f, 0.010f,
        24.0f, 1.10f, 0.10f, 0.25f,
        900.0f,
        0.14f, 0.45f, 0.55f, 1.00f,
        20
    };
    return s;
}

class EpisodeRunner {
public:
    EpisodeRunner(const PlannerVariant& variant, const Scenario& scenario, int k_samples, int t_horizon, int seed)
        : variant_(variant), scenario_(scenario), k_samples_(k_samples), t_horizon_(t_horizon), seed_(seed)
    {
        h_nominal_.assign(t_horizon_, 0.0f);
        h_costs_.assign(k_samples_, 0.0f);
        h_grad_.assign(t_horizon_, 0.0f);

        CUDA_CHECK(cudaMalloc(&d_nominal_, t_horizon_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs_, k_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perturbed_, k_samples_ * t_horizon_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_, k_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_states_, (t_horizon_ + 1) * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_, t_horizon_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_, k_samples_ * sizeof(curandState)));

        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), t_horizon_ * sizeof(float), cudaMemcpyHostToDevice));
        init_curand();
        sample_initial_state();
        warmup_controller();
    }

    ~EpisodeRunner() {
        if (d_nominal_) cudaFree(d_nominal_);
        if (d_costs_) cudaFree(d_costs_);
        if (d_perturbed_) cudaFree(d_perturbed_);
        if (d_weights_) cudaFree(d_weights_);
        if (d_states_) cudaFree(d_states_);
        if (d_grad_) cudaFree(d_grad_);
        if (d_rng_) cudaFree(d_rng_);
    }

    EpisodeMetrics run() {
        sample_initial_state();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), t_horizon_ * sizeof(float), cudaMemcpyHostToDevice));

        auto episode_begin = chrono::steady_clock::now();
        float control_ms_total = 0.0f;
        float cumulative_cost = 0.0f;
        float best_error = stabilization_error(x_, x_dot_, theta_, theta_dot_);
        bool success = false;
        bool track_violation = false;
        int stable_steps = 0;
        int executed_steps = 0;

        for (int step = 0; step < scenario_.max_steps; step++) {
            auto control_begin = chrono::steady_clock::now();
            controller_update();
            CUDA_CHECK(cudaMemcpy(h_nominal_.data(), d_nominal_, t_horizon_ * sizeof(float), cudaMemcpyDeviceToHost));
            auto control_end = chrono::steady_clock::now();
            float control_ms = chrono::duration<float, milli>(control_end - control_begin).count();
            control_ms_total += control_ms;

            float action = h_nominal_[0];
            cartpole_step(x_, x_dot_, theta_, theta_dot_, action, scenario_.params);
            cumulative_cost += stage_cost(x_, x_dot_, theta_, theta_dot_, action, scenario_.cost_params);
            executed_steps = step + 1;

            float err = stabilization_error(x_, x_dot_, theta_, theta_dot_);
            best_error = min(best_error, err);

            bool stable = fabsf(wrap_anglef(theta_)) < scenario_.cost_params.success_angle
                       && fabsf(x_) < scenario_.cost_params.success_x
                       && fabsf(x_dot_) < scenario_.cost_params.success_x_dot
                       && fabsf(theta_dot_) < scenario_.cost_params.success_theta_dot;
            stable_steps = stable ? (stable_steps + 1) : 0;
            if (stable_steps >= scenario_.cost_params.success_window) {
                success = true;
                break;
            }

            if (fabsf(x_) > scenario_.params.x_threshold) {
                cumulative_cost += scenario_.cost_params.out_of_bounds_penalty;
                track_violation = true;
                break;
            }

            shift_nominal();
        }

        cumulative_cost += terminal_cost(x_, x_dot_, theta_, theta_dot_, scenario_.cost_params);
        auto episode_end = chrono::steady_clock::now();
        float episode_ms = chrono::duration<float, milli>(episode_end - episode_begin).count();
        float final_error = stabilization_error(x_, x_dot_, theta_, theta_dot_);

        EpisodeMetrics metrics;
        metrics.scenario = scenario_.name;
        metrics.planner = variant_.name;
        metrics.seed = seed_;
        metrics.k_samples = k_samples_;
        metrics.t_horizon = t_horizon_;
        metrics.grad_steps = variant_.grad_steps;
        metrics.alpha = variant_.alpha;
        metrics.reached_goal = success ? 1 : 0;
        metrics.collision_free = track_violation ? 0 : 1;
        metrics.success = success ? 1 : 0;
        metrics.steps = executed_steps;
        metrics.final_distance = final_error;
        metrics.min_goal_distance = best_error;
        metrics.cumulative_cost = cumulative_cost;
        metrics.collisions = track_violation ? 1 : 0;
        metrics.avg_control_ms = control_ms_total / max(1, executed_steps);
        metrics.total_control_ms = control_ms_total;
        metrics.episode_ms = episode_ms;
        metrics.sample_budget = static_cast<long long>(executed_steps) * static_cast<long long>(k_samples_);
        return metrics;
    }

private:
    void sample_initial_state() {
        mt19937 rng(seed_);
        uniform_real_distribution<float> x_dist(scenario_.x_lo, scenario_.x_hi);
        uniform_real_distribution<float> xd_dist(scenario_.x_dot_lo, scenario_.x_dot_hi);
        uniform_real_distribution<float> th_dist(scenario_.theta_lo, scenario_.theta_hi);
        uniform_real_distribution<float> thd_dist(scenario_.theta_dot_lo, scenario_.theta_dot_hi);
        x_ = x_dist(rng);
        x_dot_ = xd_dist(rng);
        theta_ = th_dist(rng);
        theta_dot_ = thd_dist(rng);
    }

    void init_curand() {
        int block = 256;
        init_curand_kernel<<<(k_samples_ + block - 1) / block, block>>>(d_rng_, k_samples_, 1000ULL + seed_);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void controller_update() {
        int block = 256;
        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
            x_, x_dot_, theta_, theta_dot_, d_nominal_, d_costs_, d_perturbed_, d_rng_,
            scenario_.params, scenario_.cost_params, variant_.noise_sigma, k_samples_, t_horizon_);
        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, DEFAULT_LAMBDA);
        update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
            d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);

        if (variant_.use_gradient) {
            for (int gs = 0; gs < variant_.grad_steps; gs++) {
                rollout_nominal_kernel<<<1, 1>>>(x_, x_dot_, theta_, theta_dot_, d_nominal_, d_states_, scenario_.params, t_horizon_);
                compute_gradient_kernel<<<1, 1>>>(d_states_, d_nominal_, d_grad_, scenario_.params, scenario_.cost_params, t_horizon_);
                gradient_step_kernel<<<(t_horizon_ + block - 1) / block, block>>>(d_nominal_, d_grad_, variant_.alpha, t_horizon_);
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void shift_nominal() {
        if (t_horizon_ <= 1) return;
        for (int t = 0; t + 1 < t_horizon_; t++) h_nominal_[t] = h_nominal_[t + 1];
        h_nominal_.back() = 0.0f;
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), t_horizon_ * sizeof(float), cudaMemcpyHostToDevice));
    }

    void warmup_controller() {
        controller_update();
        CUDA_CHECK(cudaMemcpy(h_nominal_.data(), d_nominal_, t_horizon_ * sizeof(float), cudaMemcpyDeviceToHost));
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), t_horizon_ * sizeof(float), cudaMemcpyHostToDevice));
    }

    PlannerVariant variant_;
    Scenario scenario_;
    int k_samples_ = 0;
    int t_horizon_ = 0;
    int seed_ = 0;

    float x_ = 0.0f;
    float x_dot_ = 0.0f;
    float theta_ = 0.0f;
    float theta_dot_ = 0.0f;

    vector<float> h_nominal_;
    vector<float> h_costs_;
    vector<float> h_grad_;

    float* d_nominal_ = nullptr;
    float* d_costs_ = nullptr;
    float* d_perturbed_ = nullptr;
    float* d_weights_ = nullptr;
    float* d_states_ = nullptr;
    float* d_grad_ = nullptr;
    curandState* d_rng_ = nullptr;
};

static void write_csv(const vector<EpisodeMetrics>& rows, const string& path) {
    ofstream out(path);
    out << "scenario,planner,seed,k_samples,t_horizon,grad_steps,alpha,reached_goal,collision_free,success,steps,"
           "final_distance,min_goal_distance,cumulative_cost,collisions,avg_control_ms,total_control_ms,episode_ms,sample_budget\n";
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
        s.steps_sum += r.steps;
        s.final_sum += r.final_distance;
        s.min_sum += r.min_goal_distance;
        s.cost_sum += r.cumulative_cost;
        s.ms_sum += r.avg_control_ms;
    }

    cout << "=== benchmark_diff_mppi_cartpole summary ===" << endl;
    for (const auto& kv : stats) {
        const auto& s = kv.second;
        double inv = 1.0 / max(1, s.episodes);
        cout << kv.first
             << " : success=" << s.successes * inv
             << " steps=" << s.steps_sum * inv
             << " final_err=" << s.final_sum * inv
             << " min_err=" << s.min_sum * inv
             << " cost=" << s.cost_sum * inv
             << " avg_ms=" << s.ms_sum * inv << endl;
    }
}

int main(int argc, char** argv) {
    bool quick = false;
    string csv_path = "build/benchmark_diff_mppi_cartpole.csv";
    vector<int> k_values;
    vector<string> scenario_names;
    vector<string> planner_names;
    int seed_count = -1;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--quick") quick = true;
        else if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
        else if (arg == "--k-values" && i + 1 < argc) k_values = parse_int_list(argv[++i]);
        else if (arg == "--seed-count" && i + 1 < argc) seed_count = max(1, atoi(argv[++i]));
        else if (arg == "--scenarios" && i + 1 < argc) scenario_names = parse_string_list(argv[++i]);
        else if (arg == "--planners" && i + 1 < argc) planner_names = parse_string_list(argv[++i]);
        else {
            cerr << "Unknown arg: " << arg << endl;
            return 1;
        }
    }

    if (k_values.empty()) k_values = quick ? vector<int>{256, 512} : vector<int>{256, 512, 1024, 2048};
    if (seed_count < 0) seed_count = quick ? 2 : 4;

    vector<Scenario> scenarios = {make_cartpole_recover(), make_cartpole_large_angle()};
    if (!scenario_names.empty()) {
        vector<Scenario> filtered;
        for (const auto& scenario : scenarios) {
            if (find(scenario_names.begin(), scenario_names.end(), scenario.name) != scenario_names.end()) {
                filtered.push_back(scenario);
            }
        }
        scenarios = filtered;
    }
    if (scenarios.empty()) {
        cerr << "No scenarios selected." << endl;
        return 1;
    }

    vector<PlannerVariant> variants = {
        {"mppi", false, 0, 0.0f, 0.40f},
        {"diff_mppi_1", true, 1, 0.020f, 0.36f},
        {"diff_mppi_3", true, 3, 0.010f, 0.36f},
    };
    if (!planner_names.empty()) {
        vector<PlannerVariant> filtered;
        for (const auto& variant : variants) {
            if (find(planner_names.begin(), planner_names.end(), variant.name) != planner_names.end()) {
                filtered.push_back(variant);
            }
        }
        variants = filtered;
    }
    if (variants.empty()) {
        cerr << "No planners selected." << endl;
        return 1;
    }

    vector<EpisodeMetrics> rows;
    rows.reserve(scenarios.size() * variants.size() * k_values.size() * seed_count);

    for (size_t si = 0; si < scenarios.size(); si++) {
        const auto& scenario = scenarios[si];
        for (int k_samples : k_values) {
            for (size_t vi = 0; vi < variants.size(); vi++) {
                const auto& variant = variants[vi];
                for (int seed = 0; seed < seed_count; seed++) {
                    int run_seed = static_cast<int>(4000 + si * 100 + vi * 20 + seed * 7 + k_samples);
                    EpisodeRunner runner(variant, scenario, k_samples, DEFAULT_T_HORIZON, run_seed);
                    EpisodeMetrics metrics = runner.run();
                    rows.push_back(metrics);
                    printf("[%s] %s K=%d seed=%d success=%d steps=%d final_err=%.3f avg_ms=%.2f track_loss=%d\n",
                           scenario.name.c_str(), variant.name.c_str(), k_samples, seed, metrics.success, metrics.steps,
                           metrics.final_distance, metrics.avg_control_ms, metrics.collisions);
                }
            }
        }
    }

    write_csv(rows, csv_path);
    print_summary(rows);
    cout << "CSV saved to " << csv_path << endl;
    return 0;
}
