/*************************************************************************
    Diff-MPPI MuJoCo Reacher Benchmark
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
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mujoco/mujoco.h>

#include "autodiff_engine.cuh"

#ifndef CUDAROBOTICS_SOURCE_DIR
#define CUDAROBOTICS_SOURCE_DIR "."
#endif

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;
using namespace cudabot;

static const int STATE_DIM = 4;
static const int CTRL_DIM = 2;
static const int GAIN_DIM = STATE_DIM * CTRL_DIM;
static const int DEFAULT_T_HORIZON = 20;
static const int DEFAULT_FRAME_SKIP = 2;
static const float DEFAULT_LAMBDA = 3.0f;

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

struct ApproxParams {
    float l1 = 0.10f;
    float l2 = 0.11f;
    float dt = 0.02f;
    float ctrl_limit = 1.0f;
    float torque_scale_1 = 18.0f;
    float torque_scale_2 = 12.0f;
    float damping_1 = 4.5f;
    float damping_2 = 3.8f;
    float coupling = 0.4f;
    float max_vel_1 = 8.0f;
    float max_vel_2 = 8.0f;
};

struct CostParams {
    float goal_x = 0.0f;
    float goal_y = 0.0f;
    float goal_weight = 14.0f;
    float control_weight = 0.04f;
    float velocity_weight = 0.05f;
    float terminal_weight = 38.0f;
    float terminal_velocity_weight = 0.30f;
    float success_tol = 0.035f;
    int success_window = 4;
};

struct Scenario {
    string name;
    float q_lo = -0.10f;
    float q_hi = 0.10f;
    float dq_lo = -0.005f;
    float dq_hi = 0.005f;
    float target_r_min = 0.0f;
    float target_r_max = 0.20f;
    float target_box = 0.20f;
    int max_steps = 50;
    int t_horizon = DEFAULT_T_HORIZON;
    ApproxParams approx;
    CostParams cost_params;
};

struct PlannerVariant {
    string name;
    bool use_feedback = false;
    bool use_gradient = false;
    int grad_steps = 0;
    float alpha = 0.0f;
    float noise_sigma_1 = 0.0f;
    float noise_sigma_2 = 0.0f;
    float feedback_gain_scale = 1.0f;
};

__host__ __device__ inline float clampf_local(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__host__ __device__ inline float wrap_anglef(float a) {
    while (a > 3.14159265f) a -= 6.28318531f;
    while (a < -3.14159265f) a += 6.28318531f;
    return a;
}

template <typename Scalar>
__host__ __device__ inline Scalar wrap_angle_diff(const Scalar& a) {
    return cudabot::atan2(cudabot::sin(a), cudabot::cos(a));
}

template <typename Scalar>
__host__ __device__ inline void arm_forward_kinematics(
    const Scalar& q1, const Scalar& q2, const ApproxParams& p,
    Scalar& ee_x, Scalar& ee_y)
{
    ee_x = Scalar::constant(p.l1) * cudabot::cos(q1)
         + Scalar::constant(p.l2) * cudabot::cos(q1 + q2);
    ee_y = Scalar::constant(p.l1) * cudabot::sin(q1)
         + Scalar::constant(p.l2) * cudabot::sin(q1 + q2);
}

__host__ __device__ inline void arm_forward_kinematics(
    float q1, float q2, const ApproxParams& p,
    float& ee_x, float& ee_y)
{
    ee_x = p.l1 * cosf(q1) + p.l2 * cosf(q1 + q2);
    ee_y = p.l1 * sinf(q1) + p.l2 * sinf(q1 + q2);
}

__host__ __device__ inline float end_effector_distance(
    float q1, float q2, const ApproxParams& params, const CostParams& cp)
{
    float ee_x, ee_y;
    arm_forward_kinematics(q1, q2, params, ee_x, ee_y);
    float dx = ee_x - cp.goal_x;
    float dy = ee_y - cp.goal_y;
    return sqrtf(dx * dx + dy * dy + 1.0e-6f);
}

__host__ __device__ inline void approx_step(
    float& q1, float& q2, float& dq1, float& dq2,
    float u1, float u2, const ApproxParams& p)
{
    u1 = clampf_local(u1, -p.ctrl_limit, p.ctrl_limit);
    u2 = clampf_local(u2, -p.ctrl_limit, p.ctrl_limit);
    float ddq1 = p.torque_scale_1 * u1 - p.damping_1 * dq1 - p.coupling * sinf(q2);
    float ddq2 = p.torque_scale_2 * u2 - p.damping_2 * dq2 - 0.5f * p.coupling * sinf(q2);
    dq1 = clampf_local(dq1 + p.dt * ddq1, -p.max_vel_1, p.max_vel_1);
    dq2 = clampf_local(dq2 + p.dt * ddq2, -p.max_vel_2, p.max_vel_2);
    q1 = wrap_anglef(q1 + p.dt * dq1);
    q2 = wrap_anglef(q2 + p.dt * dq2);
}

__device__ inline void approx_step_diff(
    Dualf& q1, Dualf& q2, Dualf& dq1, Dualf& dq2,
    Dualf u1, Dualf u2, const ApproxParams& p)
{
    u1 = clamp(u1, -p.ctrl_limit, p.ctrl_limit);
    u2 = clamp(u2, -p.ctrl_limit, p.ctrl_limit);
    Dualf ddq1 = Dualf::constant(p.torque_scale_1) * u1
               - Dualf::constant(p.damping_1) * dq1
               - Dualf::constant(p.coupling) * cudabot::sin(q2);
    Dualf ddq2 = Dualf::constant(p.torque_scale_2) * u2
               - Dualf::constant(p.damping_2) * dq2
               - Dualf::constant(0.5f * p.coupling) * cudabot::sin(q2);
    dq1 = clamp(dq1 + Dualf::constant(p.dt) * ddq1, -p.max_vel_1, p.max_vel_1);
    dq2 = clamp(dq2 + Dualf::constant(p.dt) * ddq2, -p.max_vel_2, p.max_vel_2);
    q1 = wrap_angle_diff(q1 + Dualf::constant(p.dt) * dq1);
    q2 = wrap_angle_diff(q2 + Dualf::constant(p.dt) * dq2);
}

__host__ __device__ inline float stage_cost(
    float q1, float q2, float dq1, float dq2, float u1, float u2,
    const ApproxParams& params, const CostParams& cp)
{
    float ee_x, ee_y;
    arm_forward_kinematics(q1, q2, params, ee_x, ee_y);
    float dx = ee_x - cp.goal_x;
    float dy = ee_y - cp.goal_y;
    float dist = sqrtf(dx * dx + dy * dy + 1.0e-6f);
    return cp.goal_weight * dist * params.dt
         + cp.control_weight * (u1 * u1 + u2 * u2) * params.dt
         + cp.velocity_weight * (dq1 * dq1 + dq2 * dq2) * params.dt;
}

__host__ __device__ inline float terminal_cost(
    float q1, float q2, float dq1, float dq2,
    const ApproxParams& params, const CostParams& cp)
{
    return cp.terminal_weight * end_effector_distance(q1, q2, params, cp)
         + cp.terminal_velocity_weight * (dq1 * dq1 + dq2 * dq2);
}

__device__ inline void stage_cost_grad(
    float q1, float q2, float dq1, float dq2, float u1, float u2,
    const ApproxParams& params, const CostParams& cp, float grad[6])
{
    for (int var = 0; var < 6; var++) {
        Dualf q1v = (var == 0) ? Dualf::variable(q1) : Dualf::constant(q1);
        Dualf q2v = (var == 1) ? Dualf::variable(q2) : Dualf::constant(q2);
        Dualf dq1v = (var == 2) ? Dualf::variable(dq1) : Dualf::constant(dq1);
        Dualf dq2v = (var == 3) ? Dualf::variable(dq2) : Dualf::constant(dq2);
        Dualf u1v = (var == 4) ? Dualf::variable(u1) : Dualf::constant(u1);
        Dualf u2v = (var == 5) ? Dualf::variable(u2) : Dualf::constant(u2);
        Dualf ee_x, ee_y;
        arm_forward_kinematics(q1v, q2v, params, ee_x, ee_y);
        Dualf dx = ee_x - Dualf::constant(cp.goal_x);
        Dualf dy = ee_y - Dualf::constant(cp.goal_y);
        Dualf dist = cudabot::sqrt(dx * dx + dy * dy + Dualf::constant(1.0e-6f));
        Dualf cost = Dualf::constant(cp.goal_weight * params.dt) * dist
                   + Dualf::constant(cp.control_weight * params.dt) * (u1v * u1v + u2v * u2v)
                   + Dualf::constant(cp.velocity_weight * params.dt) * (dq1v * dq1v + dq2v * dq2v);
        grad[var] = cost.deriv;
    }
}

__device__ inline void terminal_grad(
    float q1, float q2, float dq1, float dq2,
    const ApproxParams& params, const CostParams& cp, float grad[4])
{
    for (int var = 0; var < 4; var++) {
        Dualf q1v = (var == 0) ? Dualf::variable(q1) : Dualf::constant(q1);
        Dualf q2v = (var == 1) ? Dualf::variable(q2) : Dualf::constant(q2);
        Dualf dq1v = (var == 2) ? Dualf::variable(dq1) : Dualf::constant(dq1);
        Dualf dq2v = (var == 3) ? Dualf::variable(dq2) : Dualf::constant(dq2);
        Dualf ee_x, ee_y;
        arm_forward_kinematics(q1v, q2v, params, ee_x, ee_y);
        Dualf dx = ee_x - Dualf::constant(cp.goal_x);
        Dualf dy = ee_y - Dualf::constant(cp.goal_y);
        Dualf dist = cudabot::sqrt(dx * dx + dy * dy + Dualf::constant(1.0e-6f));
        Dualf cost = Dualf::constant(cp.terminal_weight) * dist
                   + Dualf::constant(cp.terminal_velocity_weight) * (dq1v * dq1v + dq2v * dq2v);
        grad[var] = cost.deriv;
    }
}

__device__ inline void approx_jacobian(
    float q1, float q2, float dq1, float dq2, float u1, float u2,
    const ApproxParams& params, float J[STATE_DIM][STATE_DIM + CTRL_DIM])
{
    for (int col = 0; col < STATE_DIM + CTRL_DIM; col++) {
        Dualf q1v = (col == 0) ? Dualf::variable(q1) : Dualf::constant(q1);
        Dualf q2v = (col == 1) ? Dualf::variable(q2) : Dualf::constant(q2);
        Dualf dq1v = (col == 2) ? Dualf::variable(dq1) : Dualf::constant(dq1);
        Dualf dq2v = (col == 3) ? Dualf::variable(dq2) : Dualf::constant(dq2);
        Dualf u1v = (col == 4) ? Dualf::variable(u1) : Dualf::constant(u1);
        Dualf u2v = (col == 5) ? Dualf::variable(u2) : Dualf::constant(u2);
        approx_step_diff(q1v, q2v, dq1v, dq2v, u1v, u2v, params);
        J[0][col] = q1v.deriv;
        J[1][col] = q2v.deriv;
        J[2][col] = dq1v.deriv;
        J[3][col] = dq2v.deriv;
    }
}

__global__ void init_curand_kernel(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void rollout_kernel(
    float sq1, float sq2, float sdq1, float sdq2,
    const float* d_nominal,
    float* d_costs,
    float* d_perturbed,
    float* d_rollout_states,
    curandState* d_rng,
    ApproxParams params,
    CostParams cp,
    int K,
    int T,
    float sigma_1,
    float sigma_2)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float q1 = sq1, q2 = sq2, dq1 = sdq1, dq2 = sdq2;
    float total_cost = 0.0f;
    if (d_rollout_states != nullptr) {
        d_rollout_states[k * (T + 1) * STATE_DIM + 0] = q1;
        d_rollout_states[k * (T + 1) * STATE_DIM + 1] = q2;
        d_rollout_states[k * (T + 1) * STATE_DIM + 2] = dq1;
        d_rollout_states[k * (T + 1) * STATE_DIM + 3] = dq2;
    }

    for (int t = 0; t < T; t++) {
        float u1 = d_nominal[t * CTRL_DIM + 0] + curand_normal(&local_rng) * sigma_1;
        float u2 = d_nominal[t * CTRL_DIM + 1] + curand_normal(&local_rng) * sigma_2;
        u1 = clampf_local(u1, -params.ctrl_limit, params.ctrl_limit);
        u2 = clampf_local(u2, -params.ctrl_limit, params.ctrl_limit);
        d_perturbed[k * T * CTRL_DIM + t * CTRL_DIM + 0] = u1;
        d_perturbed[k * T * CTRL_DIM + t * CTRL_DIM + 1] = u2;
        approx_step(q1, q2, dq1, dq2, u1, u2, params);
        total_cost += stage_cost(q1, q2, dq1, dq2, u1, u2, params, cp);
        if (d_rollout_states != nullptr) {
            d_rollout_states[k * (T + 1) * STATE_DIM + (t + 1) * STATE_DIM + 0] = q1;
            d_rollout_states[k * (T + 1) * STATE_DIM + (t + 1) * STATE_DIM + 1] = q2;
            d_rollout_states[k * (T + 1) * STATE_DIM + (t + 1) * STATE_DIM + 2] = dq1;
            d_rollout_states[k * (T + 1) * STATE_DIM + (t + 1) * STATE_DIM + 3] = dq2;
        }
    }
    total_cost += terminal_cost(q1, q2, dq1, dq2, params, cp);
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
    if (sum_w > 0.0f) for (int k = 0; k < K; k++) d_weights[k] /= sum_w;
}

__global__ void update_controls_kernel(float* d_nominal, const float* d_perturbed, const float* d_weights, int K, int T) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float u1 = 0.0f, u2 = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = d_weights[k];
        u1 += w * d_perturbed[k * T * CTRL_DIM + t * CTRL_DIM + 0];
        u2 += w * d_perturbed[k * T * CTRL_DIM + t * CTRL_DIM + 1];
    }
    d_nominal[t * CTRL_DIM + 0] = u1;
    d_nominal[t * CTRL_DIM + 1] = u2;
}

__global__ void rollout_nominal_kernel(
    float sq1, float sq2, float sdq1, float sdq2,
    const float* d_nominal, float* d_states,
    ApproxParams params, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float q1 = sq1, q2 = sq2, dq1 = sdq1, dq2 = sdq2;
    d_states[0] = q1;
    d_states[1] = q2;
    d_states[2] = dq1;
    d_states[3] = dq2;
    for (int t = 0; t < T; t++) {
        approx_step(q1, q2, dq1, dq2, d_nominal[t * CTRL_DIM + 0], d_nominal[t * CTRL_DIM + 1], params);
        d_states[(t + 1) * STATE_DIM + 0] = q1;
        d_states[(t + 1) * STATE_DIM + 1] = q2;
        d_states[(t + 1) * STATE_DIM + 2] = dq1;
        d_states[(t + 1) * STATE_DIM + 3] = dq2;
    }
}

__global__ void compute_rollout_initial_gradients_kernel(
    const float* d_rollout_states,
    const float* d_perturbed,
    float* d_rollout_init_grads,
    ApproxParams params,
    CostParams cp,
    int K,
    int T)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const float* states = &d_rollout_states[k * (T + 1) * STATE_DIM];
    const float* controls = &d_perturbed[k * T * CTRL_DIM];
    float adj[STATE_DIM];
    terminal_grad(states[T * STATE_DIM + 0], states[T * STATE_DIM + 1], states[T * STATE_DIM + 2], states[T * STATE_DIM + 3], params, cp, adj);
    for (int t = T - 1; t >= 0; t--) {
        float q1 = states[t * STATE_DIM + 0];
        float q2 = states[t * STATE_DIM + 1];
        float dq1 = states[t * STATE_DIM + 2];
        float dq2 = states[t * STATE_DIM + 3];
        float u1 = controls[t * CTRL_DIM + 0];
        float u2 = controls[t * CTRL_DIM + 1];
        float J[STATE_DIM][STATE_DIM + CTRL_DIM];
        float stage_grad_vec[STATE_DIM + CTRL_DIM];
        float next_adj[STATE_DIM];
        approx_jacobian(q1, q2, dq1, dq2, u1, u2, params, J);
        stage_cost_grad(q1, q2, dq1, dq2, u1, u2, params, cp, stage_grad_vec);
        for (int col = 0; col < STATE_DIM; col++) {
            next_adj[col] = stage_grad_vec[col];
            for (int row = 0; row < STATE_DIM; row++) next_adj[col] += J[row][col] * adj[row];
        }
        for (int i = 0; i < STATE_DIM; i++) adj[i] = next_adj[i];
    }
    for (int i = 0; i < STATE_DIM; i++) d_rollout_init_grads[k * STATE_DIM + i] = adj[i];
}

__global__ void compute_reference_feedback_gain_kernel(
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
    const float inv_lambda = 1.0f / fmaxf(1.0e-6f, lambda);
    float weighted_grad[STATE_DIM] = {};
    for (int k = 0; k < K; k++) {
        float w = d_weights[k];
        for (int j = 0; j < STATE_DIM; j++) weighted_grad[j] += w * d_rollout_init_grads[k * STATE_DIM + j];
    }
    for (int i = 0; i < T * GAIN_DIM; i++) d_feedback_gains[i] = 0.0f;
    float nominal_u1 = d_nominal[0];
    float nominal_u2 = d_nominal[1];
    for (int j = 0; j < STATE_DIM; j++) {
        float gain_u1 = 0.0f, gain_u2 = 0.0f;
        for (int k = 0; k < K; k++) {
            float w = d_weights[k];
            float centered_grad = d_rollout_init_grads[k * STATE_DIM + j] - weighted_grad[j];
            float delta_u1 = d_perturbed[k * T * CTRL_DIM + 0] - nominal_u1;
            float delta_u2 = d_perturbed[k * T * CTRL_DIM + 1] - nominal_u2;
            gain_u1 += -inv_lambda * w * centered_grad * delta_u1;
            gain_u2 += -inv_lambda * w * centered_grad * delta_u2;
        }
        d_feedback_gains[0 * STATE_DIM + j] = gain_u1;
        d_feedback_gains[1 * STATE_DIM + j] = gain_u2;
    }
}

__global__ void broadcast_first_feedback_gain_kernel(float* d_feedback_gains, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * GAIN_DIM;
    if (idx >= total) return;
    d_feedback_gains[idx] = d_feedback_gains[idx % GAIN_DIM];
}

__global__ void rollout_feedback_kernel(
    float sq1, float sq2, float sdq1, float sdq2,
    const float* d_nominal,
    const float* d_nominal_states,
    const float* d_feedback_gains,
    float* d_costs,
    float* d_perturbed,
    curandState* d_rng,
    ApproxParams params,
    CostParams cp,
    int K,
    int T,
    float gain_scale,
    float sigma_1,
    float sigma_2)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curandState local_rng = d_rng[k];
    float q1 = sq1, q2 = sq2, dq1 = sdq1, dq2 = sdq2;
    float total_cost = 0.0f;
    for (int t = 0; t < T; t++) {
        const float* ref = &d_nominal_states[t * STATE_DIM];
        const float* gains = &d_feedback_gains[t * GAIN_DIM];
        float err[STATE_DIM] = {wrap_anglef(q1 - ref[0]), wrap_anglef(q2 - ref[1]), dq1 - ref[2], dq2 - ref[3]};
        float fb1 = 0.0f, fb2 = 0.0f;
        for (int j = 0; j < STATE_DIM; j++) {
            fb1 += gains[0 * STATE_DIM + j] * err[j];
            fb2 += gains[1 * STATE_DIM + j] * err[j];
        }
        float u1 = d_nominal[t * CTRL_DIM + 0] - gain_scale * fb1 + curand_normal(&local_rng) * sigma_1;
        float u2 = d_nominal[t * CTRL_DIM + 1] - gain_scale * fb2 + curand_normal(&local_rng) * sigma_2;
        u1 = clampf_local(u1, -params.ctrl_limit, params.ctrl_limit);
        u2 = clampf_local(u2, -params.ctrl_limit, params.ctrl_limit);
        d_perturbed[k * T * CTRL_DIM + t * CTRL_DIM + 0] = u1;
        d_perturbed[k * T * CTRL_DIM + t * CTRL_DIM + 1] = u2;
        approx_step(q1, q2, dq1, dq2, u1, u2, params);
        total_cost += stage_cost(q1, q2, dq1, dq2, u1, u2, params, cp);
    }
    total_cost += terminal_cost(q1, q2, dq1, dq2, params, cp);
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

__global__ void compute_gradient_kernel(
    const float* d_states, const float* d_nominal, float* d_grad,
    ApproxParams params, CostParams cp, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float adj[STATE_DIM];
    terminal_grad(d_states[T * STATE_DIM + 0], d_states[T * STATE_DIM + 1], d_states[T * STATE_DIM + 2], d_states[T * STATE_DIM + 3], params, cp, adj);
    for (int t = T - 1; t >= 0; t--) {
        float q1 = d_states[t * STATE_DIM + 0];
        float q2 = d_states[t * STATE_DIM + 1];
        float dq1 = d_states[t * STATE_DIM + 2];
        float dq2 = d_states[t * STATE_DIM + 3];
        float u1 = d_nominal[t * CTRL_DIM + 0];
        float u2 = d_nominal[t * CTRL_DIM + 1];
        float J[STATE_DIM][STATE_DIM + CTRL_DIM];
        float stage_grad_vec[STATE_DIM + CTRL_DIM];
        float next_adj[STATE_DIM];
        approx_jacobian(q1, q2, dq1, dq2, u1, u2, params, J);
        stage_cost_grad(q1, q2, dq1, dq2, u1, u2, params, cp, stage_grad_vec);
        d_grad[t * CTRL_DIM + 0] = stage_grad_vec[4];
        d_grad[t * CTRL_DIM + 1] = stage_grad_vec[5];
        for (int row = 0; row < STATE_DIM; row++) {
            d_grad[t * CTRL_DIM + 0] += J[row][4] * adj[row];
            d_grad[t * CTRL_DIM + 1] += J[row][5] * adj[row];
        }
        for (int col = 0; col < STATE_DIM; col++) {
            next_adj[col] = stage_grad_vec[col];
            for (int row = 0; row < STATE_DIM; row++) next_adj[col] += J[row][col] * adj[row];
        }
        for (int i = 0; i < STATE_DIM; i++) adj[i] = next_adj[i];
    }
}

static vector<int> parse_int_list(const string& text) {
    vector<int> values;
    string token;
    stringstream ss(text);
    while (getline(ss, token, ',')) if (!token.empty()) values.push_back(atoi(token.c_str()));
    sort(values.begin(), values.end());
    values.erase(unique(values.begin(), values.end()), values.end());
    return values;
}

static vector<string> parse_string_list(const string& text) {
    vector<string> values;
    string token;
    stringstream ss(text);
    while (getline(ss, token, ',')) if (!token.empty()) values.push_back(token);
    sort(values.begin(), values.end());
    values.erase(unique(values.begin(), values.end()), values.end());
    return values;
}

static unsigned int stable_name_hash(const string& text) {
    unsigned int hash = 2166136261u;
    for (unsigned char c : text) {
        hash ^= static_cast<unsigned int>(c);
        hash *= 16777619u;
    }
    return hash;
}

static int make_run_seed(const string& scenario_name, const string& planner_name, int k_samples, int seed_index) {
    unsigned int h = stable_name_hash(scenario_name);
    h ^= stable_name_hash(planner_name) + 0x9e3779b9u + (h << 6) + (h >> 2);
    h ^= static_cast<unsigned int>(k_samples) * 2654435761u;
    h ^= static_cast<unsigned int>(seed_index + 1) * 2246822519u;
    return static_cast<int>(h & 0x7fffffff);
}

static bool file_exists(const string& path) {
    ifstream in(path);
    return in.good();
}

static void ensure_build_dir() {
    mkdir("build", 0755);
}

static string default_model_path() {
    const string source_dir = CUDAROBOTICS_SOURCE_DIR;
    vector<string> candidates = {
        source_dir + "/mujoco_models/reacher.xml",
        "mujoco_models/reacher.xml",
        "../mujoco_models/reacher.xml",
        "../../mujoco_models/reacher.xml",
    };
    for (const auto& path : candidates) if (file_exists(path)) return path;
    return candidates.front();
}

static Scenario make_standard_reacher() {
    Scenario s;
    s.name = "reacher_v5";
    return s;
}

static Scenario make_edge_target_reacher() {
    Scenario s;
    s.name = "reacher_edge_target";
    s.q_lo = -0.35f;
    s.q_hi = 0.35f;
    s.dq_lo = -0.05f;
    s.dq_hi = 0.05f;
    s.target_r_min = 0.15f;
    s.target_r_max = 0.22f;
    s.target_box = 0.22f;
    s.max_steps = 60;
    s.t_horizon = 24;
    s.cost_params.goal_weight = 16.0f;
    s.cost_params.terminal_weight = 42.0f;
    s.cost_params.success_tol = 0.032f;
    s.cost_params.success_window = 3;
    return s;
}

static Scenario make_terminal_edge_reacher() {
    Scenario s;
    s.name = "reacher_terminal_edge";
    s.q_lo = -0.55f;
    s.q_hi = 0.55f;
    s.dq_lo = -0.08f;
    s.dq_hi = 0.08f;
    s.target_r_min = 0.16f;
    s.target_r_max = 0.22f;
    s.target_box = 0.22f;
    s.max_steps = 70;
    s.t_horizon = 32;
    s.cost_params.goal_weight = 2.5f;
    s.cost_params.control_weight = 0.03f;
    s.cost_params.velocity_weight = 0.03f;
    s.cost_params.terminal_weight = 95.0f;
    s.cost_params.terminal_velocity_weight = 0.12f;
    s.cost_params.success_tol = 0.028f;
    s.cost_params.success_window = 2;
    return s;
}

static void sample_target(mt19937& rng, const Scenario& scenario, float& gx, float& gy) {
    uniform_real_distribution<float> dist(-scenario.target_box, scenario.target_box);
    while (true) {
        gx = dist(rng);
        gy = dist(rng);
        float r = sqrtf(gx * gx + gy * gy);
        if (r < scenario.target_r_max && r >= scenario.target_r_min) break;
    }
}

static void state_from_mujoco(const mjData* data, float& q1, float& q2, float& dq1, float& dq2, float& gx, float& gy) {
    q1 = static_cast<float>(data->qpos[0]);
    q2 = static_cast<float>(data->qpos[1]);
    gx = static_cast<float>(data->qpos[2]);
    gy = static_cast<float>(data->qpos[3]);
    dq1 = static_cast<float>(data->qvel[0]);
    dq2 = static_cast<float>(data->qvel[1]);
}

static void set_mujoco_state(const mjModel* model, mjData* data, float q1, float q2, float dq1, float dq2, float gx, float gy) {
    mj_resetData(model, data);
    data->qpos[0] = q1;
    data->qpos[1] = q2;
    data->qpos[2] = gx;
    data->qpos[3] = gy;
    data->qvel[0] = dq1;
    data->qvel[1] = dq2;
    data->qvel[2] = 0.0;
    data->qvel[3] = 0.0;
    mj_forward(model, data);
}

static void mujoco_step_env(const mjModel* model, mjData* data, float u1, float u2, int frame_skip, float ctrl_limit) {
    u1 = clampf_local(u1, -ctrl_limit, ctrl_limit);
    u2 = clampf_local(u2, -ctrl_limit, ctrl_limit);
    for (int i = 0; i < frame_skip; i++) {
        data->ctrl[0] = u1;
        data->ctrl[1] = u2;
        mj_step(model, data);
    }
}

static float approx_rollout_cost_host(float sq1, float sq2, float sdq1, float sdq2, const vector<float>& nominal, const ApproxParams& params, const CostParams& cp) {
    float q1 = sq1, q2 = sq2, dq1 = sdq1, dq2 = sdq2;
    float total_cost = 0.0f;
    const int T = static_cast<int>(nominal.size() / CTRL_DIM);
    for (int t = 0; t < T; t++) {
        float u1 = nominal[t * CTRL_DIM + 0];
        float u2 = nominal[t * CTRL_DIM + 1];
        approx_step(q1, q2, dq1, dq2, u1, u2, params);
        total_cost += stage_cost(q1, q2, dq1, dq2, u1, u2, params, cp);
    }
    total_cost += terminal_cost(q1, q2, dq1, dq2, params, cp);
    return total_cost;
}

class EpisodeRunner {
public:
    EpisodeRunner(const PlannerVariant& variant, const Scenario& scenario, const mjModel* model, int fingertip_body_id, int target_body_id, int frame_skip, int k_samples, int t_horizon, int seed)
        : variant_(variant), scenario_(scenario), model_(model), fingertip_body_id_(fingertip_body_id), target_body_id_(target_body_id), frame_skip_(frame_skip), k_samples_(k_samples), t_horizon_(t_horizon), seed_(seed) {
        data_ = mj_makeData(model_);
        h_nominal_.assign(t_horizon_ * CTRL_DIM, 0.0f);
        h_grad_.assign(t_horizon_ * CTRL_DIM, 0.0f);
        h_states_.assign((t_horizon_ + 1) * STATE_DIM, 0.0f);
        CUDA_CHECK(cudaMalloc(&d_nominal_, h_nominal_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs_, k_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perturbed_, k_samples_ * t_horizon_ * CTRL_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_, k_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rollout_states_, k_samples_ * (t_horizon_ + 1) * STATE_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rollout_init_grads_, k_samples_ * STATE_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_states_, h_states_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_, h_grad_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_feedback_gains_, t_horizon_ * GAIN_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_, k_samples_ * sizeof(curandState)));
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size() * sizeof(float), cudaMemcpyHostToDevice));
        reset_rng();
    }

    ~EpisodeRunner() {
        if (d_nominal_) cudaFree(d_nominal_);
        if (d_costs_) cudaFree(d_costs_);
        if (d_perturbed_) cudaFree(d_perturbed_);
        if (d_weights_) cudaFree(d_weights_);
        if (d_rollout_states_) cudaFree(d_rollout_states_);
        if (d_rollout_init_grads_) cudaFree(d_rollout_init_grads_);
        if (d_states_) cudaFree(d_states_);
        if (d_grad_) cudaFree(d_grad_);
        if (d_feedback_gains_) cudaFree(d_feedback_gains_);
        if (d_rng_) cudaFree(d_rng_);
        if (data_) mj_deleteData(data_);
    }

    EpisodeMetrics run() {
        warmup_controller();
        reset_state();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size() * sizeof(float), cudaMemcpyHostToDevice));
        reset_rng();

        auto episode_begin = chrono::steady_clock::now();
        float control_ms_total = 0.0f;
        float cumulative_cost = 0.0f;
        float best_error = goal_distance_mujoco();
        bool success = false;
        int stable_steps = 0;
        int executed_steps = 0;

        for (int step = 0; step < scenario_.max_steps; step++) {
            auto control_begin = chrono::steady_clock::now();
            controller_update();
            CUDA_CHECK(cudaMemcpy(h_nominal_.data(), d_nominal_, h_nominal_.size() * sizeof(float), cudaMemcpyDeviceToHost));
            auto control_end = chrono::steady_clock::now();
            float control_ms = chrono::duration<float, milli>(control_end - control_begin).count();
            control_ms_total += control_ms;

            float u1 = h_nominal_[0];
            float u2 = h_nominal_[1];
            mujoco_step_env(model_, data_, u1, u2, frame_skip_, scenario_.approx.ctrl_limit);
            sync_state_from_mujoco();
            cumulative_cost += host_stage_cost_mujoco(u1, u2);
            executed_steps = step + 1;

            float err = goal_distance_mujoco();
            best_error = min(best_error, err);
            stable_steps = (err < episode_cost_params_.success_tol) ? (stable_steps + 1) : 0;
            if (stable_steps >= episode_cost_params_.success_window) {
                success = true;
                break;
            }
            shift_nominal();
        }

        float final_error = goal_distance_mujoco();
        if (!success && final_error < episode_cost_params_.success_tol) success = true;
        cumulative_cost += terminal_cost(q1_, q2_, dq1_, dq2_, scenario_.approx, episode_cost_params_);

        auto episode_end = chrono::steady_clock::now();
        float episode_ms = chrono::duration<float, milli>(episode_end - episode_begin).count();

        EpisodeMetrics metrics;
        metrics.scenario = scenario_.name;
        metrics.planner = variant_.name;
        metrics.seed = seed_;
        metrics.k_samples = k_samples_;
        metrics.t_horizon = t_horizon_;
        metrics.grad_steps = variant_.grad_steps;
        metrics.alpha = variant_.alpha;
        metrics.reached_goal = success ? 1 : 0;
        metrics.collision_free = 1;
        metrics.success = success ? 1 : 0;
        metrics.steps = executed_steps;
        metrics.final_distance = final_error;
        metrics.min_goal_distance = best_error;
        metrics.cumulative_cost = cumulative_cost;
        metrics.collisions = 0;
        metrics.avg_control_ms = control_ms_total / max(1, executed_steps);
        metrics.total_control_ms = control_ms_total;
        metrics.episode_ms = episode_ms;
        metrics.sample_budget = static_cast<long long>(executed_steps) * static_cast<long long>(k_samples_);
        return metrics;
    }

private:
    float goal_distance_mujoco() const {
        const double* finger = &data_->xpos[3 * fingertip_body_id_];
        const double* target = &data_->xpos[3 * target_body_id_];
        float dx = static_cast<float>(finger[0] - target[0]);
        float dy = static_cast<float>(finger[1] - target[1]);
        return sqrtf(dx * dx + dy * dy + 1.0e-6f);
    }

    float host_stage_cost_mujoco(float u1, float u2) const {
        float dist = goal_distance_mujoco();
        return episode_cost_params_.goal_weight * dist * scenario_.approx.dt
             + episode_cost_params_.control_weight * (u1 * u1 + u2 * u2) * scenario_.approx.dt
             + episode_cost_params_.velocity_weight * (dq1_ * dq1_ + dq2_ * dq2_) * scenario_.approx.dt;
    }

    void sample_initial_state() {
        mt19937 rng(seed_);
        uniform_real_distribution<float> q_dist(scenario_.q_lo, scenario_.q_hi);
        uniform_real_distribution<float> dq_dist(scenario_.dq_lo, scenario_.dq_hi);
        q1_ = q_dist(rng);
        q2_ = q_dist(rng);
        dq1_ = dq_dist(rng);
        dq2_ = dq_dist(rng);
        sample_target(rng, scenario_, goal_x_, goal_y_);
        episode_cost_params_ = scenario_.cost_params;
        episode_cost_params_.goal_x = goal_x_;
        episode_cost_params_.goal_y = goal_y_;
    }

    void reset_state() {
        sample_initial_state();
        set_mujoco_state(model_, data_, q1_, q2_, dq1_, dq2_, goal_x_, goal_y_);
        sync_state_from_mujoco();
    }

    void sync_state_from_mujoco() {
        state_from_mujoco(data_, q1_, q2_, dq1_, dq2_, goal_x_, goal_y_);
    }

    void reset_rng() {
        const int block = 256;
        init_curand_kernel<<<(k_samples_ + block - 1) / block, block>>>(d_rng_, k_samples_, 7000ULL + static_cast<unsigned long long>(seed_));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void compute_feedback_inputs() {
        const int block = 256;
        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(q1_, q2_, dq1_, dq2_, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_, scenario_.approx, episode_cost_params_, k_samples_, t_horizon_, variant_.noise_sigma_1, variant_.noise_sigma_2);
        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, DEFAULT_LAMBDA);
        compute_rollout_initial_gradients_kernel<<<(k_samples_ + block - 1) / block, block>>>(d_rollout_states_, d_perturbed_, d_rollout_init_grads_, scenario_.approx, episode_cost_params_, k_samples_, t_horizon_);
        compute_reference_feedback_gain_kernel<<<1, 1>>>(d_nominal_, d_perturbed_, d_weights_, d_rollout_init_grads_, d_feedback_gains_, DEFAULT_LAMBDA, k_samples_, t_horizon_);
        broadcast_first_feedback_gain_kernel<<<(t_horizon_ * GAIN_DIM + block - 1) / block, block>>>(d_feedback_gains_, t_horizon_);
        rollout_nominal_kernel<<<1, 1>>>(q1_, q2_, dq1_, dq2_, d_nominal_, d_states_, scenario_.approx, t_horizon_);
    }

    void controller_update() {
        const int block = 256;
        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(q1_, q2_, dq1_, dq2_, d_nominal_, d_costs_, d_perturbed_, nullptr, d_rng_, scenario_.approx, episode_cost_params_, k_samples_, t_horizon_, variant_.noise_sigma_1, variant_.noise_sigma_2);
        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, DEFAULT_LAMBDA);
        update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);

        if (variant_.use_feedback) {
            compute_feedback_inputs();
            rollout_feedback_kernel<<<(k_samples_ + block - 1) / block, block>>>(q1_, q2_, dq1_, dq2_, d_nominal_, d_states_, d_feedback_gains_, d_costs_, d_perturbed_, d_rng_, scenario_.approx, episode_cost_params_, k_samples_, t_horizon_, variant_.feedback_gain_scale, variant_.noise_sigma_1, variant_.noise_sigma_2);
            compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, DEFAULT_LAMBDA);
            update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);
        }

        if (variant_.use_gradient) {
            CUDA_CHECK(cudaMemcpy(h_nominal_.data(), d_nominal_, h_nominal_.size() * sizeof(float), cudaMemcpyDeviceToHost));
            float accepted_cost = approx_rollout_cost_host(q1_, q2_, dq1_, dq2_, h_nominal_, scenario_.approx, episode_cost_params_);
            for (int gs = 0; gs < variant_.grad_steps; gs++) {
                rollout_nominal_kernel<<<1, 1>>>(q1_, q2_, dq1_, dq2_, d_nominal_, d_states_, scenario_.approx, t_horizon_);
                compute_gradient_kernel<<<1, 1>>>(d_states_, d_nominal_, d_grad_, scenario_.approx, episode_cost_params_, t_horizon_);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_grad_.data(), d_grad_, h_grad_.size() * sizeof(float), cudaMemcpyDeviceToHost));
                vector<float> candidate = h_nominal_;
                for (size_t i = 0; i < candidate.size(); i++) candidate[i] = clampf_local(candidate[i] - variant_.alpha * h_grad_[i], -scenario_.approx.ctrl_limit, scenario_.approx.ctrl_limit);
                float candidate_cost = approx_rollout_cost_host(q1_, q2_, dq1_, dq2_, candidate, scenario_.approx, episode_cost_params_);
                if (candidate_cost + 1.0e-4f < accepted_cost) {
                    h_nominal_.swap(candidate);
                    accepted_cost = candidate_cost;
                    CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size() * sizeof(float), cudaMemcpyHostToDevice));
                } else {
                    break;
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void shift_nominal() {
        if (t_horizon_ <= 1) return;
        for (int t = 0; t + 1 < t_horizon_; t++) {
            h_nominal_[t * CTRL_DIM + 0] = h_nominal_[(t + 1) * CTRL_DIM + 0];
            h_nominal_[t * CTRL_DIM + 1] = h_nominal_[(t + 1) * CTRL_DIM + 1];
        }
        h_nominal_[(t_horizon_ - 1) * CTRL_DIM + 0] = 0.0f;
        h_nominal_[(t_horizon_ - 1) * CTRL_DIM + 1] = 0.0f;
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void warmup_controller() {
        reset_state();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size() * sizeof(float), cudaMemcpyHostToDevice));
        controller_update();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    PlannerVariant variant_;
    Scenario scenario_;
    const mjModel* model_ = nullptr;
    mjData* data_ = nullptr;
    int fingertip_body_id_ = -1;
    int target_body_id_ = -1;
    int frame_skip_ = DEFAULT_FRAME_SKIP;
    int k_samples_ = 0;
    int t_horizon_ = 0;
    int seed_ = 0;
    float q1_ = 0.0f, q2_ = 0.0f, dq1_ = 0.0f, dq2_ = 0.0f;
    float goal_x_ = 0.0f, goal_y_ = 0.0f;
    CostParams episode_cost_params_;

    vector<float> h_nominal_;
    vector<float> h_grad_;
    vector<float> h_states_;
    float* d_nominal_ = nullptr;
    float* d_costs_ = nullptr;
    float* d_perturbed_ = nullptr;
    float* d_weights_ = nullptr;
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
        out << r.scenario << ',' << r.planner << ',' << r.seed << ',' << r.k_samples << ',' << r.t_horizon << ','
            << r.grad_steps << ',' << r.alpha << ',' << r.reached_goal << ',' << r.collision_free << ',' << r.success << ','
            << r.steps << ',' << r.final_distance << ',' << r.min_goal_distance << ',' << r.cumulative_cost << ','
            << r.collisions << ',' << r.avg_control_ms << ',' << r.total_control_ms << ',' << r.episode_ms << ','
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
    cout << "=== benchmark_diff_mppi_mujoco_reacher summary ===" << endl;
    for (const auto& kv : stats) {
        const auto& s = kv.second;
        double inv = 1.0 / max(1, s.episodes);
        cout << kv.first << " : success=" << s.successes * inv << " steps=" << s.steps_sum * inv
             << " final_dist=" << s.final_sum * inv << " min_dist=" << s.min_sum * inv
             << " cost=" << s.cost_sum * inv << " avg_ms=" << s.ms_sum * inv << endl;
    }
}

int main(int argc, char** argv) {
    bool quick = false;
    string csv_path = "build/benchmark_diff_mppi_mujoco_reacher.csv";
    string model_path;
    vector<int> k_values;
    vector<string> scenario_names;
    vector<string> planner_names;
    int seed_count = -1;
    int frame_skip = DEFAULT_FRAME_SKIP;
    int override_t_horizon = -1;
    float override_feedback_gain_scale = -1.0f;
    int override_grad_steps = -1;
    float override_alpha = -1.0f;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--quick") quick = true;
        else if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
        else if (arg == "--model-path" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--frame-skip" && i + 1 < argc) frame_skip = max(1, atoi(argv[++i]));
        else if (arg == "--t-horizon" && i + 1 < argc) override_t_horizon = max(1, atoi(argv[++i]));
        else if (arg == "--k-values" && i + 1 < argc) k_values = parse_int_list(argv[++i]);
        else if (arg == "--seed-count" && i + 1 < argc) seed_count = max(1, atoi(argv[++i]));
        else if (arg == "--scenarios" && i + 1 < argc) scenario_names = parse_string_list(argv[++i]);
        else if (arg == "--planners" && i + 1 < argc) planner_names = parse_string_list(argv[++i]);
        else if (arg == "--override-feedback-gain-scale" && i + 1 < argc) override_feedback_gain_scale = atof(argv[++i]);
        else if (arg == "--override-grad-steps" && i + 1 < argc) override_grad_steps = atoi(argv[++i]);
        else if (arg == "--override-alpha" && i + 1 < argc) override_alpha = atof(argv[++i]);
    }

    ensure_build_dir();
    if (model_path.empty()) model_path = default_model_path();
    char load_error[1024] = {};
    mjModel* model = mj_loadXML(model_path.c_str(), nullptr, load_error, sizeof(load_error));
    if (!model) {
        cerr << "Failed to load MuJoCo model from " << model_path << endl;
        cerr << load_error << endl;
        return 1;
    }
    if (model->nq < 4 || model->nv < 4 || model->nu < 2) {
        cerr << "Unexpected MuJoCo model dimensions in " << model_path << endl;
        mj_deleteModel(model);
        return 1;
    }
    int fingertip_body_id = mj_name2id(model, mjOBJ_BODY, "fingertip");
    int target_body_id = mj_name2id(model, mjOBJ_BODY, "target");
    if (fingertip_body_id < 0 || target_body_id < 0) {
        cerr << "Failed to resolve Reacher body ids in " << model_path << endl;
        mj_deleteModel(model);
        return 1;
    }

    vector<Scenario> scenarios = {
        make_standard_reacher(),
        make_edge_target_reacher(),
        make_terminal_edge_reacher(),
    };
    if (!scenario_names.empty()) {
        vector<Scenario> filtered;
        for (const auto& scenario : scenarios) if (find(scenario_names.begin(), scenario_names.end(), scenario.name) != scenario_names.end()) filtered.push_back(scenario);
        scenarios.swap(filtered);
    }
    if (scenarios.empty()) {
        cerr << "No scenarios selected." << endl;
        mj_deleteModel(model);
        return 1;
    }

    vector<PlannerVariant> variants = {
        {"mppi", false, false, 0, 0.0f, 0.35f, 0.35f, 1.0f},
        {"feedback_mppi_ref", true, false, 0, 0.0f, 0.28f, 0.28f, 1.5f},
        {"diff_mppi_1", false, true, 1, 0.010f, 0.26f, 0.26f, 1.0f},
        {"diff_mppi_3", false, true, 3, 0.008f, 0.26f, 0.26f, 1.0f},
    };
    if (!planner_names.empty()) {
        vector<PlannerVariant> filtered;
        for (const auto& variant : variants) if (find(planner_names.begin(), planner_names.end(), variant.name) != planner_names.end()) filtered.push_back(variant);
        variants.swap(filtered);
    }
    if (variants.empty()) {
        cerr << "No planners selected." << endl;
        mj_deleteModel(model);
        return 1;
    }
    for (auto& v : variants) {
        if (override_feedback_gain_scale >= 0.0f && v.use_feedback) v.feedback_gain_scale = override_feedback_gain_scale;
        if (override_grad_steps >= 0 && v.use_gradient) v.grad_steps = override_grad_steps;
        if (override_alpha >= 0.0f && v.use_gradient) v.alpha = override_alpha;
    }

    if (k_values.empty()) k_values = quick ? vector<int>{64, 128} : vector<int>{128, 256, 512};
    if (seed_count < 0) seed_count = quick ? 2 : 4;

    vector<EpisodeMetrics> rows;
    rows.reserve(scenarios.size() * variants.size() * k_values.size() * seed_count);
    for (size_t si = 0; si < scenarios.size(); si++) {
        const auto& scenario = scenarios[si];
        for (int ks : k_values) {
            for (size_t vi = 0; vi < variants.size(); vi++) {
                const auto& variant = variants[vi];
                for (int seed = 0; seed < seed_count; seed++) {
                    int run_seed = make_run_seed(scenario.name, variant.name, ks, seed);
                    int t_horizon = override_t_horizon > 0 ? override_t_horizon : scenario.t_horizon;
                    EpisodeRunner runner(variant, scenario, model, fingertip_body_id, target_body_id, frame_skip, ks, t_horizon, run_seed);
                    EpisodeMetrics m = runner.run();
                    rows.push_back(m);
                    printf("[%s] %s K=%d seed=%d success=%d steps=%d final_dist=%.3f avg_ms=%.2f fail=%d\n",
                           scenario.name.c_str(), variant.name.c_str(), ks, seed, m.success, m.steps, m.final_distance, m.avg_control_ms, m.collisions);
                }
            }
        }
    }

    write_csv(rows, csv_path);
    print_summary(rows);
    cout << "MuJoCo model: " << model_path << endl;
    cout << "CSV saved to " << csv_path << endl;
    mj_deleteModel(model);
    return 0;
}
