/*************************************************************************
    Diff-MPPI Dynamic Bicycle Benchmark
    - Higher-fidelity mobile-navigation pilot with steering lag and drag
    - Compares sampling-only MPPI against gradient-refined variants
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
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "autodiff_engine.cuh"
#include "diff_cost.cuh"

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;
using namespace cudabot;

static const float WORKSPACE = 50.0f;
static const int MAX_OBSTACLES = 16;
static const int MAX_DYNAMIC_OBSTACLES = 8;
static const int DEFAULT_T_HORIZON = 36;
static const float DEFAULT_LAMBDA = 10.0f;

__constant__ Obstacle d_obstacles_dynbike[MAX_OBSTACLES];

struct DynamicObstacle {
    float x;
    float y;
    float vx;
    float vy;
    float r;
};

__constant__ DynamicObstacle d_dynamic_obstacles_dynbike[MAX_DYNAMIC_OBSTACLES];

struct BikeParams {
    float L = 2.8f;
    float max_speed = 7.5f;
    float max_steer = 0.55f;
    float max_accel = 3.5f;
    float dt = 0.06f;
    float steer_tau = 0.18f;
    float linear_drag = 0.10f;
    float quad_drag = 0.025f;
};

struct BikeCostParams {
    float goal_x = 45.0f;
    float goal_y = 45.0f;
    float goal_weight = 5.5f;
    float control_weight = 0.06f;
    float speed_weight = 0.18f;
    float target_speed = 3.8f;
    float heading_weight = 0.48f;
    float steer_weight = 0.01f;
    float obs_weight = 12.0f;
    float obs_influence = 5.5f;
    float terminal_weight = 10.0f;
    float terminal_heading_weight = 1.8f;
    float terminal_speed_weight = 0.8f;
    float terminal_steer_weight = 0.10f;
};

struct Scenario {
    string name;
    float start_x = 5.0f;
    float start_y = 5.0f;
    float start_yaw = 0.0f;
    float start_v = 0.0f;
    float start_steer = 0.0f;
    float goal_tol = 2.0f;
    int max_steps = 280;
    BikeParams params;
    BikeCostParams cost_params;
    float grad_alpha_scale = 1.0f;
    int n_obs = 0;
    Obstacle obstacles[MAX_OBSTACLES];
    int n_dyn_obs = 0;
    DynamicObstacle dynamic_obstacles[MAX_DYNAMIC_OBSTACLES];
};

struct PlannerVariant {
    string name;
    bool use_feedback = false;
    int grad_steps = 0;
    float alpha = 0.0f;
    float accel_sigma = 0.0f;
    float steer_sigma = 0.0f;
    float feedback_gain_scale = 0.0f;
    float feedback_noise_accel = 0.0f;
    float feedback_noise_steer = 0.0f;
    float feedback_longitudinal_gain = 0.0f;
    float feedback_speed_gain = 0.0f;
    float feedback_lateral_gain = 0.0f;
    float feedback_heading_gain = 0.0f;
    float feedback_steer_state_gain = 0.0f;
    float feedback_setpoint_blend = 0.0f;
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
    double steps_sum = 0.0;
    double final_sum = 0.0;
    double min_sum = 0.0;
    double cost_sum = 0.0;
    double ms_sum = 0.0;
};

__host__ __device__ inline float clampf_local(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__host__ __device__ inline float wrap_anglef(float angle) {
    while (angle > 3.14159265f) angle -= 6.28318531f;
    while (angle < -3.14159265f) angle += 6.28318531f;
    return angle;
}

template <typename Scalar>
__host__ __device__ Scalar wrap_angle_diff(const Scalar& angle) {
    return cudabot::atan2(cudabot::sin(angle), cudabot::cos(angle));
}

__host__ __device__ inline void dynamic_bicycle_step(
    float& x, float& y, float& yaw, float& v, float& steer,
    float accel_cmd, float steer_cmd, const BikeParams& p)
{
    accel_cmd = clampf_local(accel_cmd, -p.max_accel, p.max_accel);
    steer_cmd = clampf_local(steer_cmd, -p.max_steer, p.max_steer);

    float steer_rate = (steer_cmd - steer) / p.steer_tau;
    steer = clampf_local(steer + steer_rate * p.dt, -p.max_steer, p.max_steer);

    float drag = p.linear_drag * v + p.quad_drag * v * fabsf(v);
    v = clampf_local(v + (accel_cmd - drag) * p.dt, 0.0f, p.max_speed);

    yaw = wrap_anglef(yaw + v / p.L * tanf(steer) * p.dt);
    x += v * cosf(yaw) * p.dt;
    y += v * sinf(yaw) * p.dt;
}

__device__ inline void dynamic_bicycle_step_diff(
    Dualf& x, Dualf& y, Dualf& yaw, Dualf& v, Dualf& steer,
    Dualf accel_cmd, Dualf steer_cmd, const BikeParams& p)
{
    accel_cmd = clamp(accel_cmd, -p.max_accel, p.max_accel);
    steer_cmd = clamp(steer_cmd, -p.max_steer, p.max_steer);

    Dualf steer_rate = (steer_cmd - steer) / Dualf::constant(p.steer_tau);
    steer = clamp(steer + steer_rate * p.dt, -p.max_steer, p.max_steer);

    Dualf drag = Dualf::constant(p.linear_drag) * v + Dualf::constant(p.quad_drag) * v * cudabot::abs(v);
    v = clamp(v + (accel_cmd - drag) * p.dt, 0.0f, p.max_speed);

    yaw = wrap_angle_diff(yaw + v / Dualf::constant(p.L) * cudabot::tan(steer) * p.dt);
    x = x + v * cudabot::cos(yaw) * p.dt;
    y = y + v * cudabot::sin(yaw) * p.dt;
}

__device__ inline void dynamic_bicycle_jacobian(
    float x, float y, float yaw, float v, float steer, float accel_cmd, float steer_cmd,
    const BikeParams& p, float J[5][7])
{
    for (int col = 0; col < 7; col++) {
        Dualf dx = (col == 0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dy = (col == 1) ? Dualf::variable(y) : Dualf::constant(y);
        Dualf dyaw = (col == 2) ? Dualf::variable(yaw) : Dualf::constant(yaw);
        Dualf dv = (col == 3) ? Dualf::variable(v) : Dualf::constant(v);
        Dualf dsteer = (col == 4) ? Dualf::variable(steer) : Dualf::constant(steer);
        Dualf da = (col == 5) ? Dualf::variable(accel_cmd) : Dualf::constant(accel_cmd);
        Dualf ds = (col == 6) ? Dualf::variable(steer_cmd) : Dualf::constant(steer_cmd);
        dynamic_bicycle_step_diff(dx, dy, dyaw, dv, dsteer, da, ds, p);
        J[0][col] = dx.deriv;
        J[1][col] = dy.deriv;
        J[2][col] = dyaw.deriv;
        J[3][col] = dv.deriv;
        J[4][col] = dsteer.deriv;
    }
}

__host__ __device__ inline float dynamic_obstacle_margin(float x, float y, const DynamicObstacle& obs, float tau) {
    float ox = obs.x + obs.vx * tau;
    float oy = obs.y + obs.vy * tau;
    float dx = x - ox;
    float dy = y - oy;
    return sqrtf(dx * dx + dy * dy + 1e-6f) - obs.r;
}

__device__ inline Dualf dynamic_obstacle_cost_diff(
    Dualf px, Dualf py, float tau, int n_dyn_obs, float influence, float weight)
{
    Dualf cost = Dualf::constant(0.0f);
    for (int i = 0; i < n_dyn_obs; i++) {
        float ox = d_dynamic_obstacles_dynbike[i].x + d_dynamic_obstacles_dynbike[i].vx * tau;
        float oy = d_dynamic_obstacles_dynbike[i].y + d_dynamic_obstacles_dynbike[i].vy * tau;
        Dualf dx = px - Dualf::constant(ox);
        Dualf dy = py - Dualf::constant(oy);
        Dualf d = cudabot::sqrt(dx * dx + dy * dy + Dualf::constant(1e-6f))
                - Dualf::constant(d_dynamic_obstacles_dynbike[i].r);
        if (d.val < influence && d.val > 0.1f) {
            cost = cost + Dualf::constant(weight) / (d * d);
        } else if (d.val <= 0.1f) {
            cost = cost + Dualf::constant(weight * 100.0f);
        }
    }
    return cost;
}

__host__ __device__ inline float stage_cost(
    float x, float y, float yaw, float v, float steer, float accel_cmd, float steer_cmd,
    const BikeCostParams& cp, float dt)
{
    float dx = x - cp.goal_x;
    float dy = y - cp.goal_y;
    float desired_heading = atan2f(cp.goal_y - y, cp.goal_x - x);
    float heading_err = wrap_anglef(yaw - desired_heading);
    float speed_err = v - cp.target_speed;
    return cp.goal_weight * sqrtf(dx * dx + dy * dy + 0.01f) * dt
         + cp.control_weight * (accel_cmd * accel_cmd + steer_cmd * steer_cmd) * dt
         + cp.speed_weight * speed_err * speed_err * dt
         + cp.heading_weight * heading_err * heading_err * dt
         + cp.steer_weight * steer * steer * dt;
}

__host__ __device__ inline float terminal_cost(
    float x, float y, float yaw, float v, float steer, const BikeCostParams& cp)
{
    float dx = x - cp.goal_x;
    float dy = y - cp.goal_y;
    float desired_heading = atan2f(cp.goal_y - y, cp.goal_x - x);
    float heading_err = wrap_anglef(yaw - desired_heading);
    float speed_err = v - cp.target_speed;
    return cp.terminal_weight * sqrtf(dx * dx + dy * dy + 0.01f)
         + cp.terminal_heading_weight * heading_err * heading_err
         + cp.terminal_speed_weight * speed_err * speed_err
         + cp.terminal_steer_weight * steer * steer;
}

__device__ inline void terminal_grad(
    float x, float y, float yaw, float v, float steer, const BikeCostParams& cp, float grad[5])
{
    for (int var = 0; var < 5; var++) {
        Dualf dx = (var == 0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dy = (var == 1) ? Dualf::variable(y) : Dualf::constant(y);
        Dualf dyaw = (var == 2) ? Dualf::variable(yaw) : Dualf::constant(yaw);
        Dualf dv = (var == 3) ? Dualf::variable(v) : Dualf::constant(v);
        Dualf dsteer = (var == 4) ? Dualf::variable(steer) : Dualf::constant(steer);

        Dualf desired = cudabot::atan2(Dualf::constant(cp.goal_y) - dy, Dualf::constant(cp.goal_x) - dx);
        Dualf heading_err = wrap_angle_diff(dyaw - desired);
        Dualf speed_err = dv - Dualf::constant(cp.target_speed);
        Dualf cost = Dualf::constant(cp.terminal_weight)
                   * cudabot::sqrt((dx - Dualf::constant(cp.goal_x)) * (dx - Dualf::constant(cp.goal_x))
                                 + (dy - Dualf::constant(cp.goal_y)) * (dy - Dualf::constant(cp.goal_y))
                                 + Dualf::constant(0.01f))
                   + Dualf::constant(cp.terminal_heading_weight) * heading_err * heading_err
                   + Dualf::constant(cp.terminal_speed_weight) * speed_err * speed_err
                   + Dualf::constant(cp.terminal_steer_weight) * dsteer * dsteer;
        grad[var] = cost.deriv;
    }
}

__device__ inline void stage_cost_grad(
    float x, float y, float yaw, float v, float steer, float accel_cmd, float steer_cmd,
    const BikeCostParams& cp, int n_obs, int n_dyn_obs, float tau, float dt, float grad[7])
{
    for (int var = 0; var < 7; var++) {
        Dualf dx = (var == 0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dy = (var == 1) ? Dualf::variable(y) : Dualf::constant(y);
        Dualf dyaw = (var == 2) ? Dualf::variable(yaw) : Dualf::constant(yaw);
        Dualf dv = (var == 3) ? Dualf::variable(v) : Dualf::constant(v);
        Dualf dsteer = (var == 4) ? Dualf::variable(steer) : Dualf::constant(steer);
        Dualf da = (var == 5) ? Dualf::variable(accel_cmd) : Dualf::constant(accel_cmd);
        Dualf ds = (var == 6) ? Dualf::variable(steer_cmd) : Dualf::constant(steer_cmd);

        Dualf desired = cudabot::atan2(Dualf::constant(cp.goal_y) - dy, Dualf::constant(cp.goal_x) - dx);
        Dualf heading_err = wrap_angle_diff(dyaw - desired);
        Dualf speed_err = dv - Dualf::constant(cp.target_speed);
        Dualf cost = Dualf::constant(cp.goal_weight)
                   * cudabot::sqrt((dx - Dualf::constant(cp.goal_x)) * (dx - Dualf::constant(cp.goal_x))
                                 + (dy - Dualf::constant(cp.goal_y)) * (dy - Dualf::constant(cp.goal_y))
                                 + Dualf::constant(0.01f)) * dt
                   + obstacle_cost_diff(dx, dy, d_obstacles_dynbike, n_obs, cp.obs_influence, cp.obs_weight)
                   + dynamic_obstacle_cost_diff(dx, dy, tau, n_dyn_obs, cp.obs_influence, cp.obs_weight)
                   + Dualf::constant(cp.control_weight) * (da * da + ds * ds) * dt
                   + Dualf::constant(cp.speed_weight) * speed_err * speed_err * dt
                   + Dualf::constant(cp.heading_weight) * heading_err * heading_err * dt
                   + Dualf::constant(cp.steer_weight) * dsteer * dsteer * dt;
        grad[var] = cost.deriv;
    }
}

__global__ void init_curand_kernel(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void rollout_kernel(
    float sx, float sy, float syaw, float sv, float ssteer,
    const float* d_nominal, float* d_costs, float* d_perturbed, float* d_rollout_states, curandState* d_rng,
    BikeParams params, BikeCostParams cost_params, int n_obs, int n_dyn_obs, int start_step,
    float accel_sigma, float steer_sigma, int K, int T)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float x = sx;
    float y = sy;
    float yaw = syaw;
    float v = sv;
    float steer = ssteer;
    float total_cost = 0.0f;
    int state_base = k * (T + 1) * 5;

    if (d_rollout_states != nullptr) {
        d_rollout_states[state_base + 0] = x;
        d_rollout_states[state_base + 1] = y;
        d_rollout_states[state_base + 2] = yaw;
        d_rollout_states[state_base + 3] = v;
        d_rollout_states[state_base + 4] = steer;
    }

    for (int t = 0; t < T; t++) {
        float accel_cmd = d_nominal[t * 2 + 0] + curand_normal(&local_rng) * accel_sigma;
        float steer_cmd = d_nominal[t * 2 + 1] + curand_normal(&local_rng) * steer_sigma;
        accel_cmd = clampf_local(accel_cmd, -params.max_accel, params.max_accel);
        steer_cmd = clampf_local(steer_cmd, -params.max_steer, params.max_steer);

        d_perturbed[k * T * 2 + t * 2 + 0] = accel_cmd;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer_cmd;

        dynamic_bicycle_step(x, y, yaw, v, steer, accel_cmd, steer_cmd, params);
        if (d_rollout_states != nullptr) {
            int offset = state_base + (t + 1) * 5;
            d_rollout_states[offset + 0] = x;
            d_rollout_states[offset + 1] = y;
            d_rollout_states[offset + 2] = yaw;
            d_rollout_states[offset + 3] = v;
            d_rollout_states[offset + 4] = steer;
        }
        total_cost += stage_cost(x, y, yaw, v, steer, accel_cmd, steer_cmd, cost_params, params.dt);

        for (int i = 0; i < n_obs; i++) {
            float dx = x - d_obstacles_dynbike[i].x;
            float dy = y - d_obstacles_dynbike[i].y;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_dynbike[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float margin = dynamic_obstacle_margin(x, y, d_dynamic_obstacles_dynbike[i], tau);
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) {
            total_cost += 500.0f;
            break;
        }
    }

    total_cost += terminal_cost(x, y, yaw, v, steer, cost_params);
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

__global__ void compute_rollout_initial_gradients_kernel(
    const float* d_rollout_states,
    const float* d_perturbed,
    float* d_rollout_init_grads,
    BikeParams params,
    BikeCostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const float* rollout_states = &d_rollout_states[k * (T + 1) * 5];
    const float* rollout_actions = &d_perturbed[k * T * 2];

    float adj[5];
    terminal_grad(rollout_states[T * 5 + 0], rollout_states[T * 5 + 1], rollout_states[T * 5 + 2],
                  rollout_states[T * 5 + 3], rollout_states[T * 5 + 4], cost_params, adj);

    for (int t = T - 1; t >= 0; t--) {
        float x = rollout_states[t * 5 + 0];
        float y = rollout_states[t * 5 + 1];
        float yaw = rollout_states[t * 5 + 2];
        float v = rollout_states[t * 5 + 3];
        float steer = rollout_states[t * 5 + 4];
        float accel_cmd = rollout_actions[t * 2 + 0];
        float steer_cmd = rollout_actions[t * 2 + 1];

        float J[5][7];
        float stage_grad_vec[7];
        float next_adj[5];
        float tau = (start_step + t) * params.dt;

        dynamic_bicycle_jacobian(x, y, yaw, v, steer, accel_cmd, steer_cmd, params, J);
        stage_cost_grad(x, y, yaw, v, steer, accel_cmd, steer_cmd, cost_params, n_obs, n_dyn_obs, tau, params.dt, stage_grad_vec);

        for (int col = 0; col < 5; col++) {
            next_adj[col] = stage_grad_vec[col];
            for (int row = 0; row < 5; row++) next_adj[col] += J[row][col] * adj[row];
        }
        for (int i = 0; i < 5; i++) adj[i] = next_adj[i];
    }

    for (int i = 0; i < 5; i++) d_rollout_init_grads[k * 5 + i] = adj[i];
}

__global__ void compute_sensitivity_feedback_gains_kernel(
    const float* d_nominal,
    const float* d_perturbed,
    const float* d_weights,
    const float* d_rollout_init_grads,
    float* d_feedback_gains,
    float lambda,
    int K,
    int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float weighted_grad[5] = {};
    for (int k = 0; k < K; k++) {
        float w = d_weights[k];
        for (int j = 0; j < 5; j++) weighted_grad[j] += w * d_rollout_init_grads[k * 5 + j];
    }

    for (int t = 0; t < T; t++) {
        float accel_mean = d_nominal[t * 2 + 0];
        float steer_mean = d_nominal[t * 2 + 1];
        for (int j = 0; j < 5; j++) {
            float accel_cov = 0.0f;
            float steer_cov = 0.0f;
            for (int k = 0; k < K; k++) {
                float w = d_weights[k];
                float g = d_rollout_init_grads[k * 5 + j];
                accel_cov += w * d_perturbed[k * T * 2 + t * 2 + 0] * g;
                steer_cov += w * d_perturbed[k * T * 2 + t * 2 + 1] * g;
            }
            d_feedback_gains[t * 10 + 0 * 5 + j] = -(accel_cov - accel_mean * weighted_grad[j]) / lambda;
            d_feedback_gains[t * 10 + 1 * 5 + j] = -(steer_cov - steer_mean * weighted_grad[j]) / lambda;
        }
    }
}

__global__ void rollout_feedback_kernel(
    float sx, float sy, float syaw, float sv, float ssteer,
    const float* d_nominal,
    const float* d_nominal_states,
    const float* d_feedback_gains,
    float* d_costs,
    float* d_perturbed,
    curandState* d_rng,
    BikeParams params,
    BikeCostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    float gain_scale,
    float noise_accel_sigma,
    float noise_steer_sigma,
    float longitudinal_gain,
    float speed_gain,
    float lateral_gain,
    float heading_gain,
    float steer_state_gain,
    float setpoint_blend)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float x = sx;
    float y = sy;
    float yaw = syaw;
    float v = sv;
    float steer = ssteer;
    float total_cost = 0.0f;

    for (int t = 0; t < T; t++) {
        int t_next = min(t + 1, T);
        float x_nom0 = d_nominal_states[t * 5 + 0];
        float y_nom0 = d_nominal_states[t * 5 + 1];
        float yaw_nom0 = d_nominal_states[t * 5 + 2];
        float v_nom0 = d_nominal_states[t * 5 + 3];
        float steer_nom0 = d_nominal_states[t * 5 + 4];
        float x_nom1 = d_nominal_states[t_next * 5 + 0];
        float y_nom1 = d_nominal_states[t_next * 5 + 1];
        float yaw_nom1 = d_nominal_states[t_next * 5 + 2];
        float v_nom1 = d_nominal_states[t_next * 5 + 3];
        float steer_nom1 = d_nominal_states[t_next * 5 + 4];

        float x_nom = (1.0f - setpoint_blend) * x_nom0 + setpoint_blend * x_nom1;
        float y_nom = (1.0f - setpoint_blend) * y_nom0 + setpoint_blend * y_nom1;
        float yaw_nom = wrap_anglef((1.0f - setpoint_blend) * yaw_nom0 + setpoint_blend * yaw_nom1);
        float v_nom = (1.0f - setpoint_blend) * v_nom0 + setpoint_blend * v_nom1;
        float steer_nom = (1.0f - setpoint_blend) * steer_nom0 + setpoint_blend * steer_nom1;

        float dx = x_nom - x;
        float dy = y_nom - y;
        float ex = x - x_nom;
        float ey = y - y_nom;
        float eyaw = wrap_anglef(yaw - yaw_nom);
        float ev = v - v_nom;
        float esteer = steer - steer_nom;
        float ct = cosf(yaw_nom);
        float st = sinf(yaw_nom);
        float longitudinal_err = ct * dx + st * dy;
        float lateral_err = -st * dx + ct * dy;
        float heading_err = wrap_anglef(yaw_nom - yaw);
        float speed_err = v_nom - v;
        float steer_state_err = steer_nom - steer;

        const float* K_t = &d_feedback_gains[t * 10];
        float accel_feedback =
            K_t[0] * ex + K_t[1] * ey + K_t[2] * eyaw + K_t[3] * ev + K_t[4] * esteer;
        float steer_feedback =
            K_t[5] * ex + K_t[6] * ey + K_t[7] * eyaw + K_t[8] * ev + K_t[9] * esteer;

        float accel_cmd = d_nominal[t * 2 + 0]
                        + curand_normal(&local_rng) * noise_accel_sigma
                        - gain_scale * accel_feedback
                        + longitudinal_gain * longitudinal_err
                        + speed_gain * speed_err;
        float steer_cmd = d_nominal[t * 2 + 1]
                        + curand_normal(&local_rng) * noise_steer_sigma
                        - gain_scale * steer_feedback
                        + lateral_gain * lateral_err
                        + heading_gain * heading_err
                        + steer_state_gain * steer_state_err;
        accel_cmd = clampf_local(accel_cmd, -params.max_accel, params.max_accel);
        steer_cmd = clampf_local(steer_cmd, -params.max_steer, params.max_steer);

        d_perturbed[k * T * 2 + t * 2 + 0] = accel_cmd;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer_cmd;

        dynamic_bicycle_step(x, y, yaw, v, steer, accel_cmd, steer_cmd, params);
        total_cost += stage_cost(x, y, yaw, v, steer, accel_cmd, steer_cmd, cost_params, params.dt);

        for (int i = 0; i < n_obs; i++) {
            float odx = x - d_obstacles_dynbike[i].x;
            float ody = y - d_obstacles_dynbike[i].y;
            float margin = sqrtf(odx * odx + ody * ody + 1e-6f) - d_obstacles_dynbike[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float margin = dynamic_obstacle_margin(x, y, d_dynamic_obstacles_dynbike[i], tau);
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) total_cost += 500.0f;
    }

    total_cost += terminal_cost(x, y, yaw, v, steer, cost_params);
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
    float sx, float sy, float syaw, float sv, float ssteer,
    const float* d_nominal, float* d_states, BikeParams params, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float x = sx;
    float y = sy;
    float yaw = syaw;
    float v = sv;
    float steer = ssteer;
    d_states[0] = x;
    d_states[1] = y;
    d_states[2] = yaw;
    d_states[3] = v;
    d_states[4] = steer;

    for (int t = 0; t < T; t++) {
        dynamic_bicycle_step(x, y, yaw, v, steer, d_nominal[t * 2 + 0], d_nominal[t * 2 + 1], params);
        d_states[(t + 1) * 5 + 0] = x;
        d_states[(t + 1) * 5 + 1] = y;
        d_states[(t + 1) * 5 + 2] = yaw;
        d_states[(t + 1) * 5 + 3] = v;
        d_states[(t + 1) * 5 + 4] = steer;
    }
}

__global__ void compute_gradient_kernel(
    const float* d_states, const float* d_nominal, float* d_grad, BikeParams params,
    BikeCostParams cost_params, int n_obs, int n_dyn_obs, int start_step, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float adj[5];
    terminal_grad(d_states[T * 5 + 0], d_states[T * 5 + 1], d_states[T * 5 + 2],
                  d_states[T * 5 + 3], d_states[T * 5 + 4], cost_params, adj);

    for (int t = T - 1; t >= 0; t--) {
        float x = d_states[t * 5 + 0];
        float y = d_states[t * 5 + 1];
        float yaw = d_states[t * 5 + 2];
        float v = d_states[t * 5 + 3];
        float steer = d_states[t * 5 + 4];
        float accel_cmd = d_nominal[t * 2 + 0];
        float steer_cmd = d_nominal[t * 2 + 1];

        float J[5][7];
        float stage_grad_vec[7];
        float next_adj[5];
        float tau = (start_step + t) * params.dt;

        dynamic_bicycle_jacobian(x, y, yaw, v, steer, accel_cmd, steer_cmd, params, J);
        stage_cost_grad(x, y, yaw, v, steer, accel_cmd, steer_cmd, cost_params, n_obs, n_dyn_obs, tau, params.dt, stage_grad_vec);

        d_grad[t * 2 + 0] = stage_grad_vec[5];
        d_grad[t * 2 + 1] = stage_grad_vec[6];
        for (int row = 0; row < 5; row++) {
            d_grad[t * 2 + 0] += J[row][5] * adj[row];
            d_grad[t * 2 + 1] += J[row][6] * adj[row];
        }

        for (int col = 0; col < 5; col++) {
            next_adj[col] = stage_grad_vec[col];
            for (int row = 0; row < 5; row++) next_adj[col] += J[row][col] * adj[row];
        }
        for (int i = 0; i < 5; i++) adj[i] = next_adj[i];
    }
}

__global__ void gradient_step_kernel(float* d_nominal, const float* d_grad, int T, float alpha, float max_accel, float max_steer) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const float accel_grad_clip = 25.0f;
    const float steer_grad_clip = 2.5f;
    float accel_grad = clampf_local(d_grad[t * 2 + 0], -accel_grad_clip, accel_grad_clip);
    float steer_grad = clampf_local(d_grad[t * 2 + 1], -steer_grad_clip, steer_grad_clip);
    d_nominal[t * 2 + 0] = clampf_local(d_nominal[t * 2 + 0] - alpha * accel_grad, -max_accel, max_accel);
    d_nominal[t * 2 + 1] = clampf_local(d_nominal[t * 2 + 1] - alpha * steer_grad, -max_steer, max_steer);
}

static float host_step_cost(
    float x, float y, float yaw, float v, float steer, float accel_cmd, float steer_cmd,
    const Scenario& scenario, int step_index)
{
    float cost = stage_cost(x, y, yaw, v, steer, accel_cmd, steer_cmd, scenario.cost_params, scenario.params.dt);
    float tau = step_index * scenario.params.dt;
    for (int i = 0; i < scenario.n_obs; i++) {
        float dx = x - scenario.obstacles[i].x;
        float dy = y - scenario.obstacles[i].y;
        float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - scenario.obstacles[i].r;
        if (margin <= 0.1f) cost += scenario.cost_params.obs_weight * 100.0f;
        else if (margin < scenario.cost_params.obs_influence) cost += scenario.cost_params.obs_weight / (margin * margin);
    }
    for (int i = 0; i < scenario.n_dyn_obs; i++) {
        float margin = dynamic_obstacle_margin(x, y, scenario.dynamic_obstacles[i], tau);
        if (margin <= 0.1f) cost += scenario.cost_params.obs_weight * 100.0f;
        else if (margin < scenario.cost_params.obs_influence) cost += scenario.cost_params.obs_weight / (margin * margin);
    }
    if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) cost += 500.0f;
    return cost;
}

static float min_obstacle_margin(float x, float y, const Scenario& scenario, int step_index) {
    float best = 1.0e9f;
    float tau = step_index * scenario.params.dt;
    for (int i = 0; i < scenario.n_obs; i++) {
        float dx = x - scenario.obstacles[i].x;
        float dy = y - scenario.obstacles[i].y;
        best = min(best, sqrtf(dx * dx + dy * dy + 1e-6f) - scenario.obstacles[i].r);
    }
    for (int i = 0; i < scenario.n_dyn_obs; i++) {
        best = min(best, dynamic_obstacle_margin(x, y, scenario.dynamic_obstacles[i], tau));
    }
    return best;
}

static Scenario make_dynbike_crossing_scene() {
    Scenario s;
    s.name = "dynbike_crossing";
    s.start_x = 4.0f;
    s.start_y = 6.0f;
    s.start_yaw = 0.15f;
    s.start_v = 0.0f;
    s.goal_tol = 2.2f;
    s.max_steps = 300;
    s.params.max_speed = 7.0f;
    s.params.dt = 0.06f;
    s.params.steer_tau = 0.20f;
    s.cost_params.goal_x = 46.0f;
    s.cost_params.goal_y = 44.0f;
    s.cost_params.goal_weight = 5.5f;
    s.cost_params.obs_weight = 12.0f;
    s.cost_params.obs_influence = 5.6f;
    s.cost_params.heading_weight = 0.45f;
    s.cost_params.target_speed = 4.0f;
    s.cost_params.terminal_weight = 12.0f;
    s.grad_alpha_scale = 0.08f;
    const Obstacle obs[] = {
        {16.0f, 16.0f, 2.8f}, {16.0f, 34.0f, 2.8f},
        {34.0f, 14.0f, 2.6f}, {34.0f, 36.0f, 2.6f}
    };
    const DynamicObstacle dyn[] = {
        {11.0f, 24.0f, 1.45f, 0.0f, 2.5f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    s.n_dyn_obs = static_cast<int>(sizeof(dyn) / sizeof(dyn[0]));
    for (int i = 0; i < s.n_dyn_obs; i++) s.dynamic_obstacles[i] = dyn[i];
    return s;
}

static Scenario make_dynbike_slalom_scene() {
    Scenario s;
    s.name = "dynbike_slalom";
    s.start_x = 4.0f;
    s.start_y = 6.0f;
    s.start_yaw = 0.12f;
    s.start_v = 0.0f;
    s.goal_tol = 2.3f;
    s.max_steps = 320;
    s.params.max_speed = 6.8f;
    s.params.dt = 0.06f;
    s.params.steer_tau = 0.22f;
    s.cost_params.goal_x = 46.0f;
    s.cost_params.goal_y = 44.0f;
    s.cost_params.goal_weight = 5.6f;
    s.cost_params.obs_weight = 12.5f;
    s.cost_params.obs_influence = 5.8f;
    s.cost_params.heading_weight = 0.44f;
    s.cost_params.target_speed = 3.8f;
    s.cost_params.terminal_weight = 12.0f;
    s.grad_alpha_scale = 0.06f;
    const Obstacle obs[] = {
        {10.0f, 14.0f, 2.7f}, {16.0f, 32.0f, 2.8f}, {22.0f, 14.0f, 2.8f},
        {28.0f, 33.0f, 2.8f}, {34.0f, 15.0f, 2.8f}, {40.0f, 33.0f, 2.8f}
    };
    const DynamicObstacle dyn[] = {
        {24.0f, 40.0f, 0.0f, -1.35f, 2.5f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    s.n_dyn_obs = static_cast<int>(sizeof(dyn) / sizeof(dyn[0]));
    for (int i = 0; i < s.n_dyn_obs; i++) s.dynamic_obstacles[i] = dyn[i];
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
        values.push_back(max(1, atoi(token.c_str())));
    }
    sort(values.begin(), values.end());
    values.erase(unique(values.begin(), values.end()), values.end());
    return values;
}

static vector<string> parse_string_list(const string& text) {
    vector<string> values;
    string token;
    stringstream ss(text);
    while (getline(ss, token, ',')) {
        if (!token.empty()) values.push_back(token);
    }
    sort(values.begin(), values.end());
    values.erase(unique(values.begin(), values.end()), values.end());
    return values;
}

class EpisodeRunner {
public:
    EpisodeRunner(const PlannerVariant& variant, const Scenario& scenario, int k_samples, int t_horizon, int seed)
        : variant_(variant), scenario_(scenario), k_samples_(k_samples), t_horizon_(t_horizon), seed_(seed)
    {
        h_nominal_.assign(t_horizon_ * 2, 0.0f);
        h_costs_.assign(k_samples_, 0.0f);
        h_grad_.assign(t_horizon_ * 2, 0.0f);
        h_states_.assign((t_horizon_ + 1) * 5, 0.0f);

        CUDA_CHECK(cudaMalloc(&d_nominal_, h_nominal_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs_, h_costs_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_, k_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perturbed_, k_samples_ * t_horizon_ * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rollout_states_, k_samples_ * (t_horizon_ + 1) * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rollout_init_grads_, k_samples_ * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_states_, h_states_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_, h_grad_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_feedback_gains_, t_horizon_ * 2 * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_, k_samples_ * sizeof(curandState)));

        reset_rng();
        reset_state();
    }

    ~EpisodeRunner() {
        cudaFree(d_nominal_);
        cudaFree(d_costs_);
        cudaFree(d_weights_);
        cudaFree(d_perturbed_);
        cudaFree(d_rollout_states_);
        cudaFree(d_rollout_init_grads_);
        cudaFree(d_states_);
        cudaFree(d_grad_);
        cudaFree(d_feedback_gains_);
        cudaFree(d_rng_);
    }

    EpisodeMetrics run() {
        reset_state();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size() * sizeof(float), cudaMemcpyHostToDevice));

        auto episode_begin = chrono::steady_clock::now();
        float total_control_ms = 0.0f;

        for (int step = 0; step < scenario_.max_steps; step++) {
            float dx = x_ - scenario_.cost_params.goal_x;
            float dy = y_ - scenario_.cost_params.goal_y;
            float goal_dist = sqrtf(dx * dx + dy * dy);
            min_goal_distance_ = min(min_goal_distance_, goal_dist);
            if (goal_dist < scenario_.goal_tol) {
                reached_goal_ = true;
                steps_taken_ = step;
                break;
            }

            auto t0 = chrono::steady_clock::now();
            controller_update(step);
            auto t1 = chrono::steady_clock::now();
            float control_ms = chrono::duration<float, milli>(t1 - t0).count();
            total_control_ms += control_ms;

            CUDA_CHECK(cudaMemcpy(h_nominal_.data(), d_nominal_, h_nominal_.size() * sizeof(float), cudaMemcpyDeviceToHost));
            float accel_cmd = h_nominal_[0];
            float steer_cmd = h_nominal_[1];

            dynamic_bicycle_step(x_, y_, yaw_, v_, steer_, accel_cmd, steer_cmd, scenario_.params);
            cumulative_cost_ += host_step_cost(x_, y_, yaw_, v_, steer_, accel_cmd, steer_cmd, scenario_, step + 1);
            float margin = min_obstacle_margin(x_, y_, scenario_, step + 1);
            if (margin <= 0.0f || x_ < 0.0f || x_ > WORKSPACE || y_ < 0.0f || y_ > WORKSPACE) collisions_++;

            for (int t = 0; t < t_horizon_ - 1; t++) {
                h_nominal_[t * 2 + 0] = h_nominal_[(t + 1) * 2 + 0];
                h_nominal_[t * 2 + 1] = h_nominal_[(t + 1) * 2 + 1];
            }
            h_nominal_[(t_horizon_ - 1) * 2 + 0] = 0.0f;
            h_nominal_[(t_horizon_ - 1) * 2 + 1] = 0.0f;
            steps_taken_ = step + 1;
        }

        auto episode_end = chrono::steady_clock::now();
        float dx = x_ - scenario_.cost_params.goal_x;
        float dy = y_ - scenario_.cost_params.goal_y;
        float final_distance = sqrtf(dx * dx + dy * dy);
        if (final_distance < scenario_.goal_tol) reached_goal_ = true;
        cumulative_cost_ += terminal_cost(x_, y_, yaw_, v_, steer_, scenario_.cost_params);

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
        metrics.avg_control_ms = steps_taken_ > 0 ? total_control_ms / steps_taken_ : 0.0f;
        metrics.total_control_ms = total_control_ms;
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

    void reset_state() {
        x_ = scenario_.start_x;
        y_ = scenario_.start_y;
        yaw_ = scenario_.start_yaw;
        v_ = scenario_.start_v;
        steer_ = scenario_.start_steer;
        steps_taken_ = 0;
        collisions_ = 0;
        reached_goal_ = false;
        cumulative_cost_ = 0.0f;
        min_goal_distance_ = sqrtf((x_ - scenario_.cost_params.goal_x) * (x_ - scenario_.cost_params.goal_x)
                                 + (y_ - scenario_.cost_params.goal_y) * (y_ - scenario_.cost_params.goal_y));
    }

    void controller_update(int start_step) {
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size() * sizeof(float), cudaMemcpyHostToDevice));
        int block = 256;
        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
            x_, y_, yaw_, v_, steer_, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_, scenario_.params, scenario_.cost_params,
            scenario_.n_obs, scenario_.n_dyn_obs, start_step, variant_.accel_sigma, variant_.steer_sigma, k_samples_, t_horizon_);
        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, DEFAULT_LAMBDA);
        update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);

        if (variant_.use_feedback) {
            rollout_nominal_kernel<<<1, 1>>>(x_, y_, yaw_, v_, steer_, d_nominal_, d_states_, scenario_.params, t_horizon_);
            compute_rollout_initial_gradients_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                d_rollout_states_, d_perturbed_, d_rollout_init_grads_, scenario_.params, scenario_.cost_params,
                scenario_.n_obs, scenario_.n_dyn_obs, start_step, k_samples_, t_horizon_);
            compute_sensitivity_feedback_gains_kernel<<<1, 1>>>(
                d_nominal_, d_perturbed_, d_weights_, d_rollout_init_grads_, d_feedback_gains_,
                DEFAULT_LAMBDA, k_samples_, t_horizon_);
            rollout_feedback_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                x_, y_, yaw_, v_, steer_, d_nominal_, d_states_, d_feedback_gains_, d_costs_, d_perturbed_, d_rng_,
                scenario_.params, scenario_.cost_params, scenario_.n_obs, scenario_.n_dyn_obs, start_step, k_samples_, t_horizon_,
                variant_.feedback_gain_scale,
                variant_.feedback_noise_accel,
                variant_.feedback_noise_steer,
                variant_.feedback_longitudinal_gain,
                variant_.feedback_speed_gain,
                variant_.feedback_lateral_gain,
                variant_.feedback_heading_gain,
                variant_.feedback_steer_state_gain,
                variant_.feedback_setpoint_blend);
            compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, DEFAULT_LAMBDA);
            update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);
        }

        if (variant_.grad_steps > 0) {
            for (int gs = 0; gs < variant_.grad_steps; gs++) {
                rollout_nominal_kernel<<<1, 1>>>(x_, y_, yaw_, v_, steer_, d_nominal_, d_states_, scenario_.params, t_horizon_);
                compute_gradient_kernel<<<1, 1>>>(d_states_, d_nominal_, d_grad_, scenario_.params, scenario_.cost_params,
                                                  scenario_.n_obs, scenario_.n_dyn_obs, start_step, t_horizon_);
                gradient_step_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                    d_nominal_, d_grad_, t_horizon_, variant_.alpha * scenario_.grad_alpha_scale,
                    scenario_.params.max_accel, scenario_.params.max_steer);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    PlannerVariant variant_;
    Scenario scenario_;
    int k_samples_;
    int t_horizon_;
    int seed_;

    float x_ = 0.0f;
    float y_ = 0.0f;
    float yaw_ = 0.0f;
    float v_ = 0.0f;
    float steer_ = 0.0f;
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
    float* d_rollout_states_ = nullptr;
    float* d_rollout_init_grads_ = nullptr;
    float* d_states_ = nullptr;
    float* d_grad_ = nullptr;
    float* d_feedback_gains_ = nullptr;
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

    cout << "=== benchmark_diff_mppi_dynamic_bicycle summary ===" << endl;
    for (const auto& kv : stats) {
        const auto& s = kv.second;
        double inv = 1.0 / max(1, s.episodes);
        cout << kv.first
             << " : success=" << s.successes * inv
             << " steps=" << s.steps_sum * inv
             << " final_dist=" << s.final_sum * inv
             << " min_dist=" << s.min_sum * inv
             << " cost=" << s.cost_sum * inv
             << " avg_ms=" << s.ms_sum * inv << endl;
    }
}

int main(int argc, char** argv) {
    bool quick = false;
    string csv_path = "build/benchmark_diff_mppi_dynamic_bicycle.csv";
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

    ensure_build_dir();

    vector<Scenario> scenarios = {make_dynbike_crossing_scene(), make_dynbike_slalom_scene()};
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

    vector<PlannerVariant> variants;
    {
        PlannerVariant v;
        v.name = "mppi";
        v.accel_sigma = 1.0f;
        v.steer_sigma = 0.12f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_sens";
        v.use_feedback = true;
        v.accel_sigma = 1.0f;
        v.steer_sigma = 0.12f;
        v.feedback_gain_scale = 0.55f;
        v.feedback_noise_accel = 0.85f;
        v.feedback_noise_steer = 0.09f;
        v.feedback_longitudinal_gain = 0.10f;
        v.feedback_speed_gain = 0.16f;
        v.feedback_lateral_gain = 0.15f;
        v.feedback_heading_gain = 0.22f;
        v.feedback_steer_state_gain = 0.10f;
        v.feedback_setpoint_blend = 0.35f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_1";
        v.grad_steps = 1;
        v.alpha = 0.006f;
        v.accel_sigma = 1.0f;
        v.steer_sigma = 0.12f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_3";
        v.grad_steps = 3;
        v.alpha = 0.0015f;
        v.accel_sigma = 1.0f;
        v.steer_sigma = 0.12f;
        variants.push_back(v);
    }
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

    if (k_values.empty()) k_values = quick ? vector<int>{32, 64, 128} : vector<int>{32, 64, 128, 256};
    if (seed_count <= 0) seed_count = quick ? 2 : 4;

    vector<EpisodeMetrics> rows;
    rows.reserve(scenarios.size() * variants.size() * k_values.size() * seed_count);

    for (size_t si = 0; si < scenarios.size(); si++) {
        const Scenario& scenario = scenarios[si];
        CUDA_CHECK(cudaMemcpyToSymbol(d_obstacles_dynbike, scenario.obstacles, sizeof(Obstacle) * scenario.n_obs));
        if (scenario.n_dyn_obs > 0) {
            CUDA_CHECK(cudaMemcpyToSymbol(d_dynamic_obstacles_dynbike, scenario.dynamic_obstacles,
                                          sizeof(DynamicObstacle) * scenario.n_dyn_obs));
        }
        for (int k_samples : k_values) {
            for (size_t vi = 0; vi < variants.size(); vi++) {
                const auto& variant = variants[vi];
                for (int seed = 0; seed < seed_count; seed++) {
                    int run_seed = static_cast<int>(7000 + si * 100 + vi * 20 + seed * 7 + k_samples);
                    EpisodeRunner runner(variant, scenario, k_samples, DEFAULT_T_HORIZON, run_seed);
                    EpisodeMetrics metrics = runner.run();
                    rows.push_back(metrics);
                    printf("[%s] %s K=%d seed=%d success=%d steps=%d final_dist=%.2f avg_ms=%.2f collisions=%d\n",
                           scenario.name.c_str(), variant.name.c_str(), k_samples, seed, metrics.success,
                           metrics.steps, metrics.final_distance, metrics.avg_control_ms, metrics.collisions);
                }
            }
        }
    }

    write_csv(rows, csv_path);
    print_summary(rows);
    cout << "CSV saved to " << csv_path << endl;
    return 0;
}
