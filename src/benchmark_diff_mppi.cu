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
#include <random>
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
static const int MAX_DYNAMIC_OBSTACLES = 8;
static const float DEFAULT_LAMBDA = 8.0f;
static const int DEFAULT_T_HORIZON = 30;
static const int BENCH_WARMUP_ITERS = 4;

__constant__ Obstacle d_obstacles_bench[MAX_OBSTACLES];

struct DynamicObstacle {
    float x;
    float y;
    float vx;
    float vy;
    float r;
};

__constant__ DynamicObstacle d_dynamic_obstacles_bench[MAX_DYNAMIC_OBSTACLES];

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
    float grad_alpha_scale = 1.0f;
    int n_obs = 0;
    Obstacle obstacles[MAX_OBSTACLES];
    int n_dyn_obs = 0;
    DynamicObstacle dynamic_obstacles[MAX_DYNAMIC_OBSTACLES];
    bool use_dynamic_mismatch = false;
    float dyn_time_offset_max = 0.0f;
    float dyn_speed_scale_max = 0.0f;
    float dyn_lateral_jitter = 0.0f;
};

struct PlannerVariant {
    string name;
    bool use_sampling = true;
    bool use_feedback = false;
    bool use_gradient = false;
    int feedback_mode = 0;
    int feedback_passes = 1;
    int replan_stride = 1;
    int grad_steps = 0;
    float alpha = 0.0f;
    float sampling_lambda = DEFAULT_LAMBDA;
    float feedback_gain_scale = 1.0f;
    float feedback_noise_accel = 0.9f;
    float feedback_noise_steer = 0.10f;
    float feedback_longitudinal_gain = 0.0f;
    float feedback_speed_gain = 0.0f;
    float feedback_lateral_gain = 0.0f;
    float feedback_heading_gain = 0.0f;
    float feedback_setpoint_blend = 0.0f;
    float feedback_q_position = 0.0f;
    float feedback_q_heading = 0.0f;
    float feedback_q_speed = 0.0f;
    float feedback_r_accel = 0.0f;
    float feedback_r_steer = 0.0f;
    float feedback_terminal_scale = 0.0f;
    float feedback_cov_regularization = 0.0f;
    float feedback_cov_blend = 1.0f;
    float feedback_lqr_blend = 0.0f;
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

struct TraceRow {
    string scenario;
    string planner;
    int seed = 0;
    int k_samples = 0;
    int grad_steps = 0;
    float alpha = 0.0f;
    int episode_step = 0;
    int horizon_step = 0;
    float goal_distance = 0.0f;
    float min_obstacle_margin = 0.0f;
    float control_ms = 0.0f;
    float sampled_accel = 0.0f;
    float sampled_steer = 0.0f;
    float final_accel = 0.0f;
    float final_steer = 0.0f;
    float delta_accel = 0.0f;
    float delta_steer = 0.0f;
    float delta_norm = 0.0f;
    float grad_accel = 0.0f;
    float grad_steer = 0.0f;
    float grad_norm = 0.0f;
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
    float* d_rollout_states,
    curandState* d_rng,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
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

    if (d_rollout_states != nullptr) {
        d_rollout_states[k * (T + 1) * 4 + 0] = x;
        d_rollout_states[k * (T + 1) * 4 + 1] = y;
        d_rollout_states[k * (T + 1) * 4 + 2] = theta;
        d_rollout_states[k * (T + 1) * 4 + 3] = v;
    }

    for (int t = 0; t < T; t++) {
        float accel = d_nominal[t * 2 + 0] + curand_normal(&local_rng) * 1.5f;
        float steer = d_nominal[t * 2 + 1] + curand_normal(&local_rng) * 0.18f;
        accel = clampf(accel, -4.0f, 4.0f);
        steer = clampf(steer, -params.max_steer, params.max_steer);

        d_perturbed[k * T * 2 + t * 2 + 0] = accel;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer;

        bicycle_step(x, y, theta, v, accel, steer, params);

        if (d_rollout_states != nullptr) {
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 0] = x;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 1] = y;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 2] = theta;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 3] = v;
        }

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

        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
            float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
            float dx = x - ox;
            float dy = y - oy;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_dynamic_obstacles_bench[i].r;
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

__host__ __device__ inline float wrap_angle(float angle) {
    while (angle > 3.14159265f) angle -= 6.28318531f;
    while (angle < -3.14159265f) angle += 6.28318531f;
    return angle;
}

__device__ void terminal_grad(float x, float y, const CostParams& cp, float grad[4]);
__device__ void stage_cost_grad(
    float x, float y, float theta, float v, float accel, float steer,
    const CostParams& cp, int n_obs, int n_dyn_obs, float tau, float grad[6]);

__device__ inline bool invert_2x2(const float A[2][2], float invA[2][2]) {
    float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (fabsf(det) < 1.0e-8f) return false;
    float inv_det = 1.0f / det;
    invA[0][0] = A[1][1] * inv_det;
    invA[0][1] = -A[0][1] * inv_det;
    invA[1][0] = -A[1][0] * inv_det;
    invA[1][1] = A[0][0] * inv_det;
    return true;
}

__device__ inline bool invert_4x4(const float A[4][4], float invA[4][4]) {
    float aug[4][8];
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) aug[row][col] = A[row][col];
        for (int col = 0; col < 4; col++) aug[row][4 + col] = (row == col) ? 1.0f : 0.0f;
    }

    for (int pivot = 0; pivot < 4; pivot++) {
        int best_row = pivot;
        float best_value = fabsf(aug[pivot][pivot]);
        for (int row = pivot + 1; row < 4; row++) {
            float value = fabsf(aug[row][pivot]);
            if (value > best_value) {
                best_value = value;
                best_row = row;
            }
        }
        if (best_value < 1.0e-8f) return false;
        if (best_row != pivot) {
            for (int col = 0; col < 8; col++) {
                float tmp = aug[pivot][col];
                aug[pivot][col] = aug[best_row][col];
                aug[best_row][col] = tmp;
            }
        }

        float diag = aug[pivot][pivot];
        float inv_diag = 1.0f / diag;
        for (int col = 0; col < 8; col++) aug[pivot][col] *= inv_diag;

        for (int row = 0; row < 4; row++) {
            if (row == pivot) continue;
            float factor = aug[row][pivot];
            if (fabsf(factor) < 1.0e-12f) continue;
            for (int col = 0; col < 8; col++) aug[row][col] -= factor * aug[pivot][col];
        }
    }

    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) invA[row][col] = aug[row][4 + col];
    }
    return true;
}

__global__ void compute_feedback_gains_kernel(
    const float* d_states,
    const float* d_nominal,
    float* d_feedback_gains,
    BicycleParams params,
    CostParams cost_params,
    int T,
    float q_position_scale,
    float q_heading_scale,
    float q_speed_scale,
    float r_accel_scale,
    float r_steer_scale,
    float terminal_scale)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float q_position = fmaxf(0.05f, cost_params.goal_weight * params.dt * q_position_scale);
    float q_heading = fmaxf(0.02f, cost_params.heading_weight * params.dt * q_heading_scale);
    float q_speed = fmaxf(0.02f, cost_params.speed_weight * params.dt * q_speed_scale);
    float r_accel = fmaxf(0.05f, cost_params.control_weight * params.dt * r_accel_scale);
    float r_steer = fmaxf(0.03f, cost_params.control_weight * params.dt * r_steer_scale);

    float Q[4][4] = {};
    Q[0][0] = q_position;
    Q[1][1] = q_position;
    Q[2][2] = q_heading;
    Q[3][3] = q_speed;

    float P[4][4] = {};
    P[0][0] = fmaxf(0.25f, cost_params.terminal_weight * terminal_scale);
    P[1][1] = fmaxf(0.25f, cost_params.terminal_weight * terminal_scale);
    P[2][2] = fmaxf(0.10f, cost_params.heading_weight * terminal_scale);
    P[3][3] = fmaxf(0.10f, cost_params.speed_weight * terminal_scale);

    for (int t = T - 1; t >= 0; t--) {
        float x = d_states[t * 4 + 0];
        float y = d_states[t * 4 + 1];
        float theta = d_states[t * 4 + 2];
        float v = d_states[t * 4 + 3];
        float accel = d_nominal[t * 2 + 0];
        float steer = d_nominal[t * 2 + 1];

        float J[4][6];
        bicycle_jacobian(x, y, theta, v, accel, steer, params, J);

        float A[4][4];
        float B[4][2];
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) A[row][col] = J[row][col];
            B[row][0] = J[row][4];
            B[row][1] = J[row][5];
        }

        float PB[4][2] = {};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 2; col++) {
                for (int k = 0; k < 4; k++) PB[row][col] += P[row][k] * B[k][col];
            }
        }

        float BtPB[2][2] = {};
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 2; col++) {
                for (int k = 0; k < 4; k++) BtPB[row][col] += B[k][row] * PB[k][col];
            }
        }

        float S[2][2] = {
            {BtPB[0][0] + r_accel, BtPB[0][1]},
            {BtPB[1][0], BtPB[1][1] + r_steer},
        };
        float S_inv[2][2];
        if (!invert_2x2(S, S_inv)) {
            S[0][0] += 0.10f;
            S[1][1] += 0.10f;
            invert_2x2(S, S_inv);
        }

        float PA[4][4] = {};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                for (int k = 0; k < 4; k++) PA[row][col] += P[row][k] * A[k][col];
            }
        }

        float BtPA[2][4] = {};
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 4; col++) {
                for (int k = 0; k < 4; k++) BtPA[row][col] += B[k][row] * PA[k][col];
            }
        }

        float K[2][4] = {};
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 4; col++) {
                for (int k = 0; k < 2; k++) K[row][col] += S_inv[row][k] * BtPA[k][col];
                d_feedback_gains[t * 8 + row * 4 + col] = K[row][col];
            }
        }

        float AtPA[4][4] = {};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                for (int k = 0; k < 4; k++) AtPA[row][col] += A[k][row] * PA[k][col];
            }
        }

        float KtBtPA[4][4] = {};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                for (int k = 0; k < 2; k++) KtBtPA[row][col] += K[k][row] * BtPA[k][col];
            }
        }

        float P_next[4][4] = {};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                P_next[row][col] = Q[row][col] + AtPA[row][col] - KtBtPA[row][col];
            }
        }

        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                P[row][col] = 0.5f * (P_next[row][col] + P_next[col][row]);
            }
        }
    }
}

__global__ void rollout_feedback_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal,
    const float* d_nominal_states,
    const float* d_feedback_gains,
    float* d_costs,
    float* d_perturbed,
    curandState* d_rng,
    BicycleParams params,
    CostParams cost_params,
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
    float setpoint_blend)
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
        int t_next = min(t + 1, T);
        float x_nom0 = d_nominal_states[t * 4 + 0];
        float y_nom0 = d_nominal_states[t * 4 + 1];
        float theta_nom0 = d_nominal_states[t * 4 + 2];
        float v_nom0 = d_nominal_states[t * 4 + 3];
        float x_nom1 = d_nominal_states[t_next * 4 + 0];
        float y_nom1 = d_nominal_states[t_next * 4 + 1];
        float theta_nom1 = d_nominal_states[t_next * 4 + 2];
        float v_nom1 = d_nominal_states[t_next * 4 + 3];
        float x_nom = (1.0f - setpoint_blend) * x_nom0 + setpoint_blend * x_nom1;
        float y_nom = (1.0f - setpoint_blend) * y_nom0 + setpoint_blend * y_nom1;
        float theta_nom = wrap_angle((1.0f - setpoint_blend) * theta_nom0 + setpoint_blend * theta_nom1);
        float v_nom = (1.0f - setpoint_blend) * v_nom0 + setpoint_blend * v_nom1;
        float dx = x_nom - x;
        float dy = y_nom - y;
        float ex = x - x_nom;
        float ey = y - y_nom;
        float etheta = wrap_angle(theta - theta_nom);
        float ev = v - v_nom;
        float ct = cosf(theta_nom);
        float st = sinf(theta_nom);
        float longitudinal_err = ct * dx + st * dy;
        float lateral_err = -st * dx + ct * dy;
        float heading_err = wrap_angle(theta_nom - theta);
        float speed_err = v_nom - v;

        const float* K_t = &d_feedback_gains[t * 8];
        float accel_feedback =
            K_t[0] * ex + K_t[1] * ey + K_t[2] * etheta + K_t[3] * ev;
        float steer_feedback =
            K_t[4] * ex + K_t[5] * ey + K_t[6] * etheta + K_t[7] * ev;

        float accel = d_nominal[t * 2 + 0]
                    + curand_normal(&local_rng) * noise_accel_sigma
                    - gain_scale * accel_feedback
                    + longitudinal_gain * longitudinal_err
                    + speed_gain * speed_err;
        float steer = d_nominal[t * 2 + 1]
                    + curand_normal(&local_rng) * noise_steer_sigma
                    - gain_scale * steer_feedback
                    + lateral_gain * lateral_err
                    + heading_gain * heading_err;
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
        float goal_heading_err = wrap_angle(theta - desired_heading);
        total_cost += cost_params.heading_weight * goal_heading_err * goal_heading_err * params.dt;
        float speed_goal_err = v - cost_params.target_speed;
        total_cost += cost_params.speed_weight * speed_goal_err * speed_goal_err * params.dt;

        for (int i = 0; i < n_obs; i++) {
            float odx = x - d_obstacles_bench[i].x;
            float ody = y - d_obstacles_bench[i].y;
            float margin = sqrtf(odx * odx + ody * ody + 1e-6f) - d_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
            float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
            float odx = x - ox;
            float ody = y - oy;
            float margin = sqrtf(odx * odx + ody * ody + 1e-6f) - d_dynamic_obstacles_bench[i].r;
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

__global__ void compute_rollout_initial_gradients_kernel(
    const float* d_rollout_states,
    const float* d_perturbed,
    float* d_rollout_init_grads,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const float* rollout_states = &d_rollout_states[k * (T + 1) * 4];
    const float* rollout_actions = &d_perturbed[k * T * 2];

    float adj[4];
    terminal_grad(rollout_states[T * 4 + 0], rollout_states[T * 4 + 1], cost_params, adj);

    for (int t = T - 1; t >= 0; t--) {
        float x = rollout_states[t * 4 + 0];
        float y = rollout_states[t * 4 + 1];
        float theta = rollout_states[t * 4 + 2];
        float v = rollout_states[t * 4 + 3];
        float accel = rollout_actions[t * 2 + 0];
        float steer = rollout_actions[t * 2 + 1];

        float J[4][6];
        float stage_grad_vec[6];
        float next_adj[4];
        float tau = (start_step + t) * params.dt;

        bicycle_jacobian(x, y, theta, v, accel, steer, params, J);
        stage_cost_grad(x, y, theta, v, accel, steer, cost_params, n_obs, n_dyn_obs, tau, stage_grad_vec);

        for (int col = 0; col < 4; col++) {
            next_adj[col] = stage_grad_vec[col];
            for (int row = 0; row < 4; row++) next_adj[col] += J[row][col] * adj[row];
        }
        for (int i = 0; i < 4; i++) adj[i] = next_adj[i];
    }

    for (int i = 0; i < 4; i++) d_rollout_init_grads[k * 4 + i] = adj[i];
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

    float weighted_grad[4] = {};
    for (int k = 0; k < K; k++) {
        float w = d_weights[k];
        for (int j = 0; j < 4; j++) weighted_grad[j] += w * d_rollout_init_grads[k * 4 + j];
    }

    for (int t = 0; t < T; t++) {
        float accel_mean = d_nominal[t * 2 + 0];
        float steer_mean = d_nominal[t * 2 + 1];
        for (int j = 0; j < 4; j++) {
            float accel_cov = 0.0f;
            float steer_cov = 0.0f;
            for (int k = 0; k < K; k++) {
                float w = d_weights[k];
                float g = d_rollout_init_grads[k * 4 + j];
                accel_cov += w * d_perturbed[k * T * 2 + t * 2 + 0] * g;
                steer_cov += w * d_perturbed[k * T * 2 + t * 2 + 1] * g;
            }
            d_feedback_gains[t * 8 + 0 * 4 + j] = -(accel_cov - accel_mean * weighted_grad[j]) / lambda;
            d_feedback_gains[t * 8 + 1 * 4 + j] = -(steer_cov - steer_mean * weighted_grad[j]) / lambda;
        }
    }
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
    float weighted_grad[4] = {};
    for (int k = 0; k < K; k++) {
        float w = d_weights[k];
        for (int j = 0; j < 4; j++) weighted_grad[j] += w * d_rollout_init_grads[k * 4 + j];
    }

    for (int i = 0; i < T * 8; i++) d_feedback_gains[i] = 0.0f;

    float nominal_accel = d_nominal[0];
    float nominal_steer = d_nominal[1];
    for (int j = 0; j < 4; j++) {
        float accel_gain = 0.0f;
        float steer_gain = 0.0f;
        for (int k = 0; k < K; k++) {
            float w = d_weights[k];
            float centered_grad = d_rollout_init_grads[k * 4 + j] - weighted_grad[j];
            float delta_accel = d_perturbed[k * T * 2 + 0] - nominal_accel;
            float delta_steer = d_perturbed[k * T * 2 + 1] - nominal_steer;
            accel_gain += -inv_lambda * w * centered_grad * delta_accel;
            steer_gain += -inv_lambda * w * centered_grad * delta_steer;
        }
        d_feedback_gains[0 * 4 + j] = accel_gain;
        d_feedback_gains[1 * 4 + j] = steer_gain;
    }
}

__global__ void compute_covariance_feedback_gains_kernel(
    const float* d_nominal,
    const float* d_nominal_states,
    const float* d_perturbed,
    const float* d_rollout_states,
    const float* d_weights,
    float* d_feedback_gains,
    int K,
    int T,
    float regularization)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const float eps = fmaxf(1.0e-4f, regularization);
    for (int t = 0; t < T; t++) {
        float Sigma_xx[4][4] = {};
        float Sigma_ux[2][4] = {};
        float x_nom = d_nominal_states[t * 4 + 0];
        float y_nom = d_nominal_states[t * 4 + 1];
        float theta_nom = d_nominal_states[t * 4 + 2];
        float v_nom = d_nominal_states[t * 4 + 3];
        float accel_nom = d_nominal[t * 2 + 0];
        float steer_nom = d_nominal[t * 2 + 1];

        for (int k = 0; k < K; k++) {
            float w = d_weights[k];
            const float* rollout_state = &d_rollout_states[k * (T + 1) * 4 + t * 4];
            float x_dev[4];
            x_dev[0] = rollout_state[0] - x_nom;
            x_dev[1] = rollout_state[1] - y_nom;
            x_dev[2] = wrap_angle(rollout_state[2] - theta_nom);
            x_dev[3] = rollout_state[3] - v_nom;

            float u_dev[2];
            u_dev[0] = d_perturbed[k * T * 2 + t * 2 + 0] - accel_nom;
            u_dev[1] = d_perturbed[k * T * 2 + t * 2 + 1] - steer_nom;

            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 4; col++) Sigma_xx[row][col] += w * x_dev[row] * x_dev[col];
            }
            for (int row = 0; row < 2; row++) {
                for (int col = 0; col < 4; col++) Sigma_ux[row][col] += w * u_dev[row] * x_dev[col];
            }
        }

        for (int i = 0; i < 4; i++) Sigma_xx[i][i] += eps;

        float Sigma_xx_inv[4][4];
        if (!invert_4x4(Sigma_xx, Sigma_xx_inv)) {
            for (int i = 0; i < 4; i++) Sigma_xx[i][i] += 10.0f * eps;
            invert_4x4(Sigma_xx, Sigma_xx_inv);
        }

        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 4; col++) {
                float gain = 0.0f;
                for (int k = 0; k < 4; k++) gain += Sigma_ux[row][k] * Sigma_xx_inv[k][col];
                d_feedback_gains[t * 8 + row * 4 + col] = -gain;
            }
        }
    }
}

__global__ void blend_feedback_gains_kernel(
    float* d_out,
    const float* d_aux,
    int T,
    float out_scale,
    float aux_scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * 8;
    if (idx >= total) return;
    d_out[idx] = out_scale * d_out[idx] + aux_scale * d_aux[idx];
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

__device__ inline Dualf dynamic_obstacle_cost_diff(
    Dualf px, Dualf py, float tau, int n_dyn_obs, float influence, float weight)
{
    Dualf cost = Dualf::constant(0.0f);
    for (int i = 0; i < n_dyn_obs; i++) {
        float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
        float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
        Dualf dx = px - Dualf::constant(ox);
        Dualf dy = py - Dualf::constant(oy);
        Dualf d = cudabot::sqrt(dx * dx + dy * dy + Dualf::constant(1e-6f))
                - Dualf::constant(d_dynamic_obstacles_bench[i].r);
        if (d.val < influence && d.val > 0.1f) {
            cost = cost + Dualf::constant(weight) / (d * d);
        } else if (d.val <= 0.1f) {
            cost = cost + Dualf::constant(weight * 100.0f);
        }
    }
    return cost;
}

__device__ void stage_cost_grad(
    float x, float y, float theta, float v, float accel, float steer,
    const CostParams& cp, int n_obs, int n_dyn_obs, float tau, float grad[6])
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
                   + dynamic_obstacle_cost_diff(dx, dy, tau, n_dyn_obs, cp.obs_influence, cp.obs_weight)
                   + control_cost_diff(da, ds, cp.control_weight)
                   + speed_cost_diff(dv, cp.target_speed, cp.speed_weight)
                   + heading_cost_diff(dx, dy, dtheta, cp.goal_x, cp.goal_y, cp.heading_weight);
        grad[var] = cost.deriv;
    }
}

__global__ void compute_gradient_kernel(
    const float* d_states, const float* d_nominal, float* d_grad,
    BicycleParams params, CostParams cost_params, int n_obs, int n_dyn_obs, int start_step, int T)
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
        float tau = (start_step + t) * params.dt;

        bicycle_jacobian(x, y, theta, v, accel, steer, params, J);
        stage_cost_grad(x, y, theta, v, accel, steer, cost_params, n_obs, n_dyn_obs, tau, stage_grad);

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

static float dynamic_obstacle_margin(float x, float y, const DynamicObstacle& obs, float tau) {
    float ox = obs.x + obs.vx * tau;
    float oy = obs.y + obs.vy * tau;
    float dx = x - ox;
    float dy = y - oy;
    return sqrtf(dx * dx + dy * dy + 1e-6f) - obs.r;
}

static float host_step_cost(
    float x, float y, float theta, float v, float accel, float steer,
    const Scenario& scenario, int step_index)
{
    const CostParams& cp = scenario.cost_params;
    float tau = step_index * scenario.params.dt;
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

    for (int i = 0; i < scenario.n_dyn_obs; i++) {
        float margin = dynamic_obstacle_margin(x, y, scenario.dynamic_obstacles[i], tau);
        if (margin <= 0.1f) cost += cp.obs_weight * 100.0f;
        else if (margin < cp.obs_influence) cost += cp.obs_weight / (margin * margin);
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
        float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - scenario.obstacles[i].r;
        best = std::min(best, margin);
    }
    for (int i = 0; i < scenario.n_dyn_obs; i++) {
        best = std::min(best, dynamic_obstacle_margin(x, y, scenario.dynamic_obstacles[i], tau));
    }
    return best;
}

class EpisodeRunner {
public:
    EpisodeRunner(const PlannerVariant& variant, const Scenario& planning_scenario, const Scenario& eval_scenario,
                  int k_samples, int t_horizon, int seed,
                  vector<TraceRow>* trace_rows = nullptr, int trace_max_steps = 0)
        : variant_(variant), planning_scenario_(planning_scenario), eval_scenario_(eval_scenario),
          k_samples_(k_samples), t_horizon_(t_horizon), seed_(seed),
          trace_rows_(trace_rows), trace_max_steps_(trace_max_steps) {
        reset_state();

        h_nominal_.assign(t_horizon_ * 2, 0.0f);
        h_costs_.assign(k_samples_, 0.0f);
        h_grad_.assign(t_horizon_ * 2, 0.0f);
        h_states_.assign((t_horizon_ + 1) * 4, 0.0f);
        h_feedback_gains_host_.assign(t_horizon_ * 2 * 4, 0.0f);
        h_sample_nominal_.assign(t_horizon_ * 2, 0.0f);
        h_final_nominal_.assign(t_horizon_ * 2, 0.0f);
        h_grad_snapshot_.assign(t_horizon_ * 2, 0.0f);

        CUDA_CHECK(cudaMalloc(&d_nominal_, h_nominal_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs_, h_costs_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_, k_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perturbed_, k_samples_ * t_horizon_ * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rollout_states_, k_samples_ * (t_horizon_ + 1) * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rollout_init_grads_, k_samples_ * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_states_, h_states_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_, h_grad_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_feedback_gains_, t_horizon_ * 2 * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_feedback_gains_aux_, t_horizon_ * 2 * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_, k_samples_ * sizeof(curandState)));

        reset_rng();
    }

    ~EpisodeRunner() {
        CUDA_CHECK(cudaFree(d_nominal_));
        CUDA_CHECK(cudaFree(d_costs_));
        CUDA_CHECK(cudaFree(d_weights_));
        CUDA_CHECK(cudaFree(d_perturbed_));
        CUDA_CHECK(cudaFree(d_rollout_states_));
        CUDA_CHECK(cudaFree(d_rollout_init_grads_));
        CUDA_CHECK(cudaFree(d_states_));
        CUDA_CHECK(cudaFree(d_grad_));
        CUDA_CHECK(cudaFree(d_feedback_gains_));
        CUDA_CHECK(cudaFree(d_feedback_gains_aux_));
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
        int controller_updates = 0;

        for (int step = 0; step < eval_scenario_.max_steps; step++) {
            float goal_dx = rx_ - eval_scenario_.cost_params.goal_x;
            float goal_dy = ry_ - eval_scenario_.cost_params.goal_y;
            float goal_dist = sqrtf(goal_dx * goal_dx + goal_dy * goal_dy);
            float margin_before = min_obstacle_margin(rx_, ry_, eval_scenario_, step);
            min_goal_distance_ = std::min(min_goal_distance_, goal_dist);
            if (goal_dist < eval_scenario_.goal_tol) {
                reached_goal_ = true;
                steps_taken_ = step;
                break;
            }

            auto t0 = chrono::steady_clock::now();
            bool replan = should_replan(step);
            if (replan) {
                controller_update(rx_, ry_, rtheta_, rv_, step);
                controller_updates++;
                sync_nominal_from_device();
                if (uses_feedback_local_action()) sync_feedback_policy_from_device();
                CUDA_CHECK(cudaMemcpy(h_costs_.data(), d_costs_, h_costs_.size() * sizeof(float), cudaMemcpyDeviceToHost));
            }
            float accel = h_nominal_[0];
            float steer = h_nominal_[1];
            if (uses_feedback_local_action()) {
                compute_feedback_inner_action(accel, steer);
            }
            auto t1 = chrono::steady_clock::now();
            float control_ms = chrono::duration<float, milli>(t1 - t0).count();
            total_control_ms += control_ms;

            if (trace_rows_ != nullptr && step < trace_max_steps_) {
                append_trace_rows(step, goal_dist, margin_before, control_ms);
            }

            bicycle_step(rx_, ry_, rtheta_, rv_, accel, steer, eval_scenario_.params);
            cumulative_cost_ += host_step_cost(rx_, ry_, rtheta_, rv_, accel, steer, eval_scenario_, step + 1);

            float margin = min_obstacle_margin(rx_, ry_, eval_scenario_, step + 1);
            if (margin <= 0.0f || rx_ < 0.0f || rx_ > WORKSPACE || ry_ < 0.0f || ry_ > WORKSPACE) collisions_++;

            shift_host_policy();
            steps_taken_ = step + 1;
        }

        auto episode_end = chrono::steady_clock::now();
        float final_dx = rx_ - eval_scenario_.cost_params.goal_x;
        float final_dy = ry_ - eval_scenario_.cost_params.goal_y;
        float final_distance = sqrtf(final_dx * final_dx + final_dy * final_dy);
        if (final_distance < eval_scenario_.goal_tol) reached_goal_ = true;

        EpisodeMetrics metrics;
        metrics.scenario = eval_scenario_.name;
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
        metrics.sample_budget = static_cast<long long>(controller_updates) * k_samples_ * t_horizon_;
        return metrics;
    }

private:
    void reset_rng() {
        int block = 256;
        init_curand_kernel<<<(k_samples_ + block - 1) / block, block>>>(d_rng_, k_samples_, static_cast<unsigned long long>(seed_));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    bool uses_feedback_local_action() const {
        return variant_.use_feedback && (variant_.feedback_mode == 5 || variant_.feedback_mode == 6 || variant_.feedback_mode == 7);
    }

    bool should_replan(int step) const {
        if (variant_.feedback_mode != 6) return true;
        int stride = max(1, variant_.replan_stride);
        return (step % stride) == 0;
    }

    void sync_nominal_from_device() {
        CUDA_CHECK(cudaMemcpy(h_nominal_.data(), d_nominal_, h_nominal_.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void sync_feedback_policy_from_device() {
        CUDA_CHECK(cudaMemcpy(h_states_.data(), d_states_, h_states_.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_feedback_gains_host_.data(), d_feedback_gains_, h_feedback_gains_host_.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void compute_feedback_inner_action(float& accel, float& steer) {
        int t_next = min(1, t_horizon_);
        float x_nom0 = h_states_[0];
        float y_nom0 = h_states_[1];
        float theta_nom0 = h_states_[2];
        float v_nom0 = h_states_[3];
        float x_nom1 = h_states_[t_next * 4 + 0];
        float y_nom1 = h_states_[t_next * 4 + 1];
        float theta_nom1 = h_states_[t_next * 4 + 2];
        float v_nom1 = h_states_[t_next * 4 + 3];
        float blend = variant_.feedback_setpoint_blend;
        float x_nom = (1.0f - blend) * x_nom0 + blend * x_nom1;
        float y_nom = (1.0f - blend) * y_nom0 + blend * y_nom1;
        float theta_nom = wrap_angle((1.0f - blend) * theta_nom0 + blend * theta_nom1);
        float v_nom = (1.0f - blend) * v_nom0 + blend * v_nom1;

        float dx = x_nom - rx_;
        float dy = y_nom - ry_;
        float ex = rx_ - x_nom;
        float ey = ry_ - y_nom;
        float etheta = wrap_angle(rtheta_ - theta_nom);
        float ev = rv_ - v_nom;
        float ct = cosf(theta_nom);
        float st = sinf(theta_nom);
        float longitudinal_err = ct * dx + st * dy;
        float lateral_err = -st * dx + ct * dy;
        float heading_err = wrap_angle(theta_nom - rtheta_);
        float speed_err = v_nom - rv_;

        const float* K_t = h_feedback_gains_host_.data();
        float accel_feedback = K_t[0] * ex + K_t[1] * ey + K_t[2] * etheta + K_t[3] * ev;
        float steer_feedback = K_t[4] * ex + K_t[5] * ey + K_t[6] * etheta + K_t[7] * ev;

        accel = h_nominal_[0]
              - variant_.feedback_gain_scale * accel_feedback
              + variant_.feedback_longitudinal_gain * longitudinal_err
              + variant_.feedback_speed_gain * speed_err;
        steer = h_nominal_[1]
              - variant_.feedback_gain_scale * steer_feedback
              + variant_.feedback_lateral_gain * lateral_err
              + variant_.feedback_heading_gain * heading_err;
        accel = clampf(accel, -4.0f, 4.0f);
        steer = clampf(steer, -eval_scenario_.params.max_steer, eval_scenario_.params.max_steer);

        if (trace_rows_ != nullptr) {
            h_sample_nominal_ = h_nominal_;
            h_final_nominal_ = h_nominal_;
            h_final_nominal_[0] = accel;
            h_final_nominal_[1] = steer;
            fill(h_grad_snapshot_.begin(), h_grad_snapshot_.end(), 0.0f);
        }
    }

    void shift_host_policy() {
        for (int t = 0; t < t_horizon_ - 1; t++) {
            h_nominal_[t * 2 + 0] = h_nominal_[(t + 1) * 2 + 0];
            h_nominal_[t * 2 + 1] = h_nominal_[(t + 1) * 2 + 1];
        }
        h_nominal_[(t_horizon_ - 1) * 2 + 0] = 0.0f;
        h_nominal_[(t_horizon_ - 1) * 2 + 1] = 0.0f;

        if (!uses_feedback_local_action()) return;

        for (int t = 0; t < t_horizon_; t++) {
            for (int i = 0; i < 4; i++) {
                h_states_[t * 4 + i] = h_states_[(t + 1) * 4 + i];
            }
        }
        for (int i = 0; i < 4; i++) {
            h_states_[t_horizon_ * 4 + i] = h_states_[(t_horizon_ - 1) * 4 + i];
        }

        for (int t = 0; t < t_horizon_ - 1; t++) {
            for (int i = 0; i < 8; i++) {
                h_feedback_gains_host_[t * 8 + i] = h_feedback_gains_host_[(t + 1) * 8 + i];
            }
        }
        for (int i = 0; i < 8; i++) {
            h_feedback_gains_host_[(t_horizon_ - 1) * 8 + i] = 0.0f;
        }
    }

    void controller_update(float sx, float sy, float stheta, float sv, int start_step) {
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size() * sizeof(float), cudaMemcpyHostToDevice));
        int block = 256;
        if (variant_.use_sampling) {
            int open_loop_passes = 1;
            if (variant_.use_feedback && (variant_.feedback_mode == 1 || variant_.feedback_mode == 3 || variant_.feedback_mode == 4 || variant_.feedback_mode == 6)) {
                open_loop_passes = 2;
            }
            for (int pass = 0; pass < open_loop_passes; pass++) {
                rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                    sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                    planning_scenario_.params, planning_scenario_.cost_params,
                    planning_scenario_.n_obs, planning_scenario_.n_dyn_obs, start_step, k_samples_, t_horizon_);
                compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
                update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                    d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);
            }

            if (variant_.use_feedback) {
                if (uses_feedback_local_action()) {
                    rollout_nominal_kernel<<<1, 1>>>(sx, sy, stheta, sv, d_nominal_, d_states_, planning_scenario_.params, t_horizon_);
                    if (variant_.feedback_mode == 5) {
                        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs, start_step, k_samples_, t_horizon_);
                        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
                        compute_rollout_initial_gradients_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            d_rollout_states_, d_perturbed_, d_rollout_init_grads_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                            start_step, k_samples_, t_horizon_);
                        compute_sensitivity_feedback_gains_kernel<<<1, 1>>>(
                            d_nominal_, d_perturbed_, d_weights_, d_rollout_init_grads_, d_feedback_gains_,
                            variant_.sampling_lambda, k_samples_, t_horizon_);
                    } else if (variant_.feedback_mode == 6) {
                        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs, start_step, k_samples_, t_horizon_);
                        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
                        compute_covariance_feedback_gains_kernel<<<1, 1>>>(
                            d_nominal_, d_states_, d_perturbed_, d_rollout_states_, d_weights_, d_feedback_gains_,
                            k_samples_, t_horizon_, variant_.feedback_cov_regularization);
                        compute_feedback_gains_kernel<<<1, 1>>>(
                            d_states_, d_nominal_, d_feedback_gains_aux_, planning_scenario_.params, planning_scenario_.cost_params, t_horizon_,
                            variant_.feedback_q_position, variant_.feedback_q_heading, variant_.feedback_q_speed,
                            variant_.feedback_r_accel, variant_.feedback_r_steer, variant_.feedback_terminal_scale);
                        blend_feedback_gains_kernel<<<(t_horizon_ * 8 + block - 1) / block, block>>>(
                            d_feedback_gains_, d_feedback_gains_aux_, t_horizon_,
                            variant_.feedback_cov_blend, variant_.feedback_lqr_blend);
                    } else {
                        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs, start_step, k_samples_, t_horizon_);
                        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
                        compute_rollout_initial_gradients_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            d_rollout_states_, d_perturbed_, d_rollout_init_grads_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                            start_step, k_samples_, t_horizon_);
                        update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                            d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);
                        compute_reference_feedback_gain_kernel<<<1, 1>>>(
                            d_nominal_, d_perturbed_, d_weights_, d_rollout_init_grads_, d_feedback_gains_,
                            variant_.sampling_lambda, k_samples_, t_horizon_);
                        rollout_nominal_kernel<<<1, 1>>>(sx, sy, stheta, sv, d_nominal_, d_states_, planning_scenario_.params, t_horizon_);
                    }
                } else {
                for (int fb_pass = 0; fb_pass < max(1, variant_.feedback_passes); fb_pass++) {
                    rollout_nominal_kernel<<<1, 1>>>(sx, sy, stheta, sv, d_nominal_, d_states_, planning_scenario_.params, t_horizon_);
                    if (variant_.feedback_mode == 2) {
                        compute_rollout_initial_gradients_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            d_rollout_states_, d_perturbed_, d_rollout_init_grads_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                            start_step, k_samples_, t_horizon_);
                        compute_sensitivity_feedback_gains_kernel<<<1, 1>>>(
                            d_nominal_, d_perturbed_, d_weights_, d_rollout_init_grads_, d_feedback_gains_,
                            variant_.sampling_lambda, k_samples_, t_horizon_);
                    } else if (variant_.feedback_mode == 3 || variant_.feedback_mode == 4) {
                        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs, start_step, k_samples_, t_horizon_);
                        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
                        compute_covariance_feedback_gains_kernel<<<1, 1>>>(
                            d_nominal_, d_states_, d_perturbed_, d_rollout_states_, d_weights_, d_feedback_gains_,
                            k_samples_, t_horizon_, variant_.feedback_cov_regularization);
                        if (variant_.feedback_mode == 4) {
                            compute_feedback_gains_kernel<<<1, 1>>>(
                                d_states_, d_nominal_, d_feedback_gains_aux_, planning_scenario_.params, planning_scenario_.cost_params, t_horizon_,
                                variant_.feedback_q_position, variant_.feedback_q_heading, variant_.feedback_q_speed,
                                variant_.feedback_r_accel, variant_.feedback_r_steer, variant_.feedback_terminal_scale);
                            blend_feedback_gains_kernel<<<(t_horizon_ * 8 + block - 1) / block, block>>>(
                                d_feedback_gains_, d_feedback_gains_aux_, t_horizon_,
                                variant_.feedback_cov_blend, variant_.feedback_lqr_blend);
                        }
                    } else {
                        compute_feedback_gains_kernel<<<1, 1>>>(
                            d_states_, d_nominal_, d_feedback_gains_, planning_scenario_.params, planning_scenario_.cost_params, t_horizon_,
                            variant_.feedback_q_position, variant_.feedback_q_heading, variant_.feedback_q_speed,
                            variant_.feedback_r_accel, variant_.feedback_r_steer, variant_.feedback_terminal_scale);
                    }
                    rollout_feedback_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                        sx, sy, stheta, sv, d_nominal_, d_states_, d_feedback_gains_, d_costs_, d_perturbed_, d_rng_,
                        planning_scenario_.params, planning_scenario_.cost_params,
                        planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                        start_step, k_samples_, t_horizon_,
                        variant_.feedback_gain_scale,
                        variant_.feedback_noise_accel,
                        variant_.feedback_noise_steer,
                        variant_.feedback_longitudinal_gain,
                        variant_.feedback_speed_gain,
                        variant_.feedback_lateral_gain,
                        variant_.feedback_heading_gain,
                        variant_.feedback_setpoint_blend);
                    compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
                    update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                        d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);
                }
                }
            }
        }

        if (trace_rows_ != nullptr) {
            CUDA_CHECK(cudaMemcpy(h_sample_nominal_.data(), d_nominal_, h_sample_nominal_.size() * sizeof(float), cudaMemcpyDeviceToHost));
            fill(h_grad_snapshot_.begin(), h_grad_snapshot_.end(), 0.0f);
        }

        if (variant_.use_gradient) {
            for (int gs = 0; gs < variant_.grad_steps; gs++) {
                rollout_nominal_kernel<<<1, 1>>>(sx, sy, stheta, sv, d_nominal_, d_states_, planning_scenario_.params, t_horizon_);
                compute_gradient_kernel<<<1, 1>>>(d_states_, d_nominal_, d_grad_, planning_scenario_.params, planning_scenario_.cost_params,
                                                  planning_scenario_.n_obs, planning_scenario_.n_dyn_obs, start_step, t_horizon_);
                gradient_step_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                    d_nominal_, d_grad_, t_horizon_,
                    variant_.alpha * planning_scenario_.grad_alpha_scale, planning_scenario_.params.max_steer);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        if (trace_rows_ != nullptr) {
            CUDA_CHECK(cudaMemcpy(h_final_nominal_.data(), d_nominal_, h_final_nominal_.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (variant_.use_gradient) {
                CUDA_CHECK(cudaMemcpy(h_grad_snapshot_.data(), d_grad_, h_grad_snapshot_.size() * sizeof(float), cudaMemcpyDeviceToHost));
            }
        }
    }

    void append_trace_rows(int episode_step, float goal_distance, float min_margin, float control_ms) {
        for (int t = 0; t < t_horizon_; t++) {
            TraceRow row;
            row.scenario = eval_scenario_.name;
            row.planner = variant_.name;
            row.seed = seed_;
            row.k_samples = k_samples_;
            row.grad_steps = variant_.grad_steps;
            row.alpha = variant_.alpha;
            row.episode_step = episode_step;
            row.horizon_step = t;
            row.goal_distance = goal_distance;
            row.min_obstacle_margin = min_margin;
            row.control_ms = control_ms;
            row.sampled_accel = h_sample_nominal_[t * 2 + 0];
            row.sampled_steer = h_sample_nominal_[t * 2 + 1];
            row.final_accel = h_final_nominal_[t * 2 + 0];
            row.final_steer = h_final_nominal_[t * 2 + 1];
            row.delta_accel = row.final_accel - row.sampled_accel;
            row.delta_steer = row.final_steer - row.sampled_steer;
            row.delta_norm = sqrtf(row.delta_accel * row.delta_accel + row.delta_steer * row.delta_steer);
            row.grad_accel = h_grad_snapshot_[t * 2 + 0];
            row.grad_steer = h_grad_snapshot_[t * 2 + 1];
            row.grad_norm = sqrtf(row.grad_accel * row.grad_accel + row.grad_steer * row.grad_steer);
            trace_rows_->push_back(row);
        }
    }

    void warmup_controller() {
        for (int iter = 0; iter < BENCH_WARMUP_ITERS; iter++) {
            controller_update(planning_scenario_.start_x, planning_scenario_.start_y,
                              planning_scenario_.start_theta, planning_scenario_.start_v, 0);
        }
    }

    void reset_state() {
        rx_ = eval_scenario_.start_x;
        ry_ = eval_scenario_.start_y;
        rtheta_ = eval_scenario_.start_theta;
        rv_ = eval_scenario_.start_v;
        steps_taken_ = 0;
        collisions_ = 0;
        reached_goal_ = false;
        cumulative_cost_ = 0.0f;
        min_goal_distance_ = sqrtf((rx_ - eval_scenario_.cost_params.goal_x) * (rx_ - eval_scenario_.cost_params.goal_x)
                                 + (ry_ - eval_scenario_.cost_params.goal_y) * (ry_ - eval_scenario_.cost_params.goal_y));
    }

    PlannerVariant variant_;
    Scenario planning_scenario_;
    Scenario eval_scenario_;
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
    vector<float> h_feedback_gains_host_;
    vector<float> h_sample_nominal_;
    vector<float> h_final_nominal_;
    vector<float> h_grad_snapshot_;

    float* d_nominal_ = nullptr;
    float* d_costs_ = nullptr;
    float* d_weights_ = nullptr;
    float* d_perturbed_ = nullptr;
    float* d_rollout_states_ = nullptr;
    float* d_rollout_init_grads_ = nullptr;
    float* d_states_ = nullptr;
    float* d_grad_ = nullptr;
    float* d_feedback_gains_ = nullptr;
    float* d_feedback_gains_aux_ = nullptr;
    curandState* d_rng_ = nullptr;
    vector<TraceRow>* trace_rows_ = nullptr;
    int trace_max_steps_ = 0;
};

static Scenario instantiate_eval_scenario(const Scenario& nominal, int seed) {
    Scenario eval = nominal;
    if (!nominal.use_dynamic_mismatch || nominal.n_dyn_obs <= 0) return eval;

    std::mt19937 rng(static_cast<uint32_t>(seed) * 747796405u + 2891336453u);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);

    for (int i = 0; i < eval.n_dyn_obs; i++) {
        DynamicObstacle& obs = eval.dynamic_obstacles[i];
        float speed = sqrtf(obs.vx * obs.vx + obs.vy * obs.vy);
        float nx = 1.0f;
        float ny = 0.0f;
        if (speed > 1.0e-5f) {
            nx = -obs.vy / speed;
            ny = obs.vx / speed;
        }
        float time_offset = nominal.dyn_time_offset_max * unit(rng);
        float speed_scale = 1.0f + nominal.dyn_speed_scale_max * unit(rng);
        float lateral_jitter = nominal.dyn_lateral_jitter * unit(rng);
        obs.x += obs.vx * time_offset + nx * lateral_jitter;
        obs.y += obs.vy * time_offset + ny * lateral_jitter;
        obs.vx *= speed_scale;
        obs.vy *= speed_scale;
    }
    return eval;
}

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

static Scenario make_dynamic_crossing_scene() {
    Scenario s;
    s.name = "dynamic_crossing";
    s.start_x = 4.0f;
    s.start_y = 6.0f;
    s.cost_params.goal_x = 46.0f;
    s.cost_params.goal_y = 44.0f;
    s.max_steps = 260;
    s.cost_params.target_speed = 3.2f;
    s.cost_params.goal_weight = 5.2f;
    s.cost_params.obs_weight = 11.5f;
    s.cost_params.obs_influence = 5.2f;
    s.cost_params.heading_weight = 0.40f;
    s.grad_alpha_scale = 0.20f;
    const Obstacle obs[] = {
        {16.0f, 16.0f, 2.8f}, {16.0f, 34.0f, 2.8f},
        {34.0f, 14.0f, 2.6f}, {34.0f, 36.0f, 2.6f}
    };
    const DynamicObstacle dyn[] = {
        {11.0f, 24.0f, 1.55f, 0.0f, 2.4f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    s.n_dyn_obs = static_cast<int>(sizeof(dyn) / sizeof(dyn[0]));
    for (int i = 0; i < s.n_dyn_obs; i++) s.dynamic_obstacles[i] = dyn[i];
    return s;
}

static Scenario make_dynamic_slalom_scene() {
    Scenario s;
    s.name = "dynamic_slalom";
    s.start_x = 4.0f;
    s.start_y = 6.0f;
    s.cost_params.goal_x = 46.0f;
    s.cost_params.goal_y = 44.0f;
    s.max_steps = 260;
    s.cost_params.target_speed = 3.5f;
    s.cost_params.goal_weight = 5.2f;
    s.cost_params.obs_weight = 11.5f;
    s.cost_params.obs_influence = 5.4f;
    s.cost_params.heading_weight = 0.38f;
    s.grad_alpha_scale = 0.22f;
    const Obstacle obs[] = {
        {10.0f, 14.0f, 2.7f}, {16.0f, 32.0f, 2.8f}, {22.0f, 14.0f, 2.8f},
        {28.0f, 33.0f, 2.8f}, {34.0f, 15.0f, 2.8f}, {40.0f, 33.0f, 2.8f}
    };
    const DynamicObstacle dyn[] = {
        {24.0f, 40.0f, 0.0f, -1.45f, 2.4f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    s.n_dyn_obs = static_cast<int>(sizeof(dyn) / sizeof(dyn[0]));
    for (int i = 0; i < s.n_dyn_obs; i++) s.dynamic_obstacles[i] = dyn[i];
    return s;
}

static Scenario make_uncertain_crossing_scene() {
    Scenario s = make_dynamic_crossing_scene();
    s.name = "uncertain_crossing";
    s.use_dynamic_mismatch = true;
    s.dyn_time_offset_max = 1.15f;
    s.dyn_speed_scale_max = 0.18f;
    s.dyn_lateral_jitter = 0.85f;
    return s;
}

static Scenario make_uncertain_slalom_scene() {
    Scenario s = make_dynamic_slalom_scene();
    s.name = "uncertain_slalom";
    s.use_dynamic_mismatch = true;
    s.dyn_time_offset_max = 0.95f;
    s.dyn_speed_scale_max = 0.16f;
    s.dyn_lateral_jitter = 0.75f;
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

static void write_trace_csv(const vector<TraceRow>& rows, const string& path) {
    ofstream out(path);
    out << "scenario,planner,seed,k_samples,grad_steps,alpha,episode_step,horizon_step,goal_distance,min_obstacle_margin,control_ms,"
           "sampled_accel,sampled_steer,final_accel,final_steer,delta_accel,delta_steer,delta_norm,grad_accel,grad_steer,grad_norm\n";
    for (const auto& r : rows) {
        out << r.scenario << ','
            << r.planner << ','
            << r.seed << ','
            << r.k_samples << ','
            << r.grad_steps << ','
            << r.alpha << ','
            << r.episode_step << ','
            << r.horizon_step << ','
            << r.goal_distance << ','
            << r.min_obstacle_margin << ','
            << r.control_ms << ','
            << r.sampled_accel << ','
            << r.sampled_steer << ','
            << r.final_accel << ','
            << r.final_steer << ','
            << r.delta_accel << ','
            << r.delta_steer << ','
            << r.delta_norm << ','
            << r.grad_accel << ','
            << r.grad_steer << ','
            << r.grad_norm << '\n';
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
    string trace_csv_path;
    vector<int> k_values;
    vector<string> scenario_names;
    vector<string> planner_names;
    int seed_count = -1;
    int trace_max_steps = 0;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--quick") quick = true;
        else if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
        else if (arg == "--trace-csv" && i + 1 < argc) trace_csv_path = argv[++i];
        else if (arg == "--trace-max-steps" && i + 1 < argc) trace_max_steps = std::max(0, atoi(argv[++i]));
        else if (arg == "--k-values" && i + 1 < argc) k_values = parse_int_list(argv[++i]);
        else if (arg == "--seed-count" && i + 1 < argc) seed_count = std::max(1, atoi(argv[++i]));
        else if (arg == "--scenarios" && i + 1 < argc) scenario_names = parse_string_list(argv[++i]);
        else if (arg == "--planners" && i + 1 < argc) planner_names = parse_string_list(argv[++i]);
    }

    ensure_build_dir();

    vector<Scenario> all_scenarios;
    all_scenarios.push_back(make_cluttered_scene());
    all_scenarios.push_back(make_narrow_passage_scene());
    all_scenarios.push_back(make_slalom_scene());
    all_scenarios.push_back(make_corner_scene());
    all_scenarios.push_back(make_dynamic_crossing_scene());
    all_scenarios.push_back(make_dynamic_slalom_scene());
    all_scenarios.push_back(make_uncertain_crossing_scene());
    all_scenarios.push_back(make_uncertain_slalom_scene());

    vector<Scenario> scenarios;
    if (!scenario_names.empty()) {
        for (const auto& wanted : scenario_names) {
            auto it = find_if(all_scenarios.begin(), all_scenarios.end(),
                              [&](const Scenario& s) { return s.name == wanted; });
            if (it == all_scenarios.end()) {
                fprintf(stderr, "Unknown scenario: %s\n", wanted.c_str());
                return 1;
            }
            scenarios.push_back(*it);
        }
    } else if (quick) {
        scenarios.push_back(make_cluttered_scene());
        scenarios.push_back(make_narrow_passage_scene());
    } else {
        scenarios.push_back(make_cluttered_scene());
        scenarios.push_back(make_narrow_passage_scene());
        scenarios.push_back(make_slalom_scene());
        scenarios.push_back(make_corner_scene());
    }

    vector<PlannerVariant> variants;
    {
        PlannerVariant v;
        v.name = "mppi";
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi";
        v.use_feedback = true;
        v.feedback_mode = 1;
        v.feedback_gain_scale = 1.0f;
        v.feedback_noise_accel = 0.9f;
        v.feedback_noise_steer = 0.10f;
        v.feedback_longitudinal_gain = 0.20f;
        v.feedback_speed_gain = 0.30f;
        v.feedback_lateral_gain = 0.28f;
        v.feedback_heading_gain = 0.42f;
        v.feedback_setpoint_blend = 0.0f;
        v.feedback_q_position = 1.8f;
        v.feedback_q_heading = 1.2f;
        v.feedback_q_speed = 1.0f;
        v.feedback_r_accel = 1.4f;
        v.feedback_r_steer = 1.1f;
        v.feedback_terminal_scale = 4.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_sens";
        v.use_feedback = true;
        v.feedback_mode = 2;
        v.feedback_gain_scale = 0.60f;
        v.feedback_noise_accel = 0.80f;
        v.feedback_noise_steer = 0.09f;
        v.feedback_longitudinal_gain = 0.12f;
        v.feedback_speed_gain = 0.18f;
        v.feedback_lateral_gain = 0.16f;
        v.feedback_heading_gain = 0.24f;
        v.feedback_setpoint_blend = 0.35f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_cov";
        v.use_feedback = true;
        v.feedback_mode = 3;
        v.feedback_gain_scale = 0.70f;
        v.feedback_noise_accel = 0.65f;
        v.feedback_noise_steer = 0.07f;
        v.feedback_longitudinal_gain = 0.18f;
        v.feedback_speed_gain = 0.24f;
        v.feedback_lateral_gain = 0.28f;
        v.feedback_heading_gain = 0.38f;
        v.feedback_setpoint_blend = 0.10f;
        v.feedback_cov_regularization = 0.20f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_fused";
        v.use_feedback = true;
        v.feedback_mode = 4;
        v.feedback_passes = 2;
        v.feedback_gain_scale = 0.75f;
        v.feedback_noise_accel = 0.60f;
        v.feedback_noise_steer = 0.07f;
        v.feedback_longitudinal_gain = 0.16f;
        v.feedback_speed_gain = 0.22f;
        v.feedback_lateral_gain = 0.24f;
        v.feedback_heading_gain = 0.34f;
        v.feedback_setpoint_blend = 0.15f;
        v.feedback_q_position = 1.6f;
        v.feedback_q_heading = 1.1f;
        v.feedback_q_speed = 0.9f;
        v.feedback_r_accel = 1.3f;
        v.feedback_r_steer = 1.0f;
        v.feedback_terminal_scale = 3.5f;
        v.feedback_cov_regularization = 0.18f;
        v.feedback_cov_blend = 0.75f;
        v.feedback_lqr_blend = 0.35f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_hf";
        v.use_feedback = true;
        v.feedback_mode = 6;
        v.replan_stride = 2;
        v.feedback_gain_scale = 0.55f;
        v.feedback_noise_accel = 0.0f;
        v.feedback_noise_steer = 0.0f;
        v.feedback_longitudinal_gain = 0.06f;
        v.feedback_speed_gain = 0.08f;
        v.feedback_lateral_gain = 0.10f;
        v.feedback_heading_gain = 0.14f;
        v.feedback_setpoint_blend = 0.30f;
        v.feedback_q_position = 1.6f;
        v.feedback_q_heading = 1.1f;
        v.feedback_q_speed = 0.9f;
        v.feedback_r_accel = 1.3f;
        v.feedback_r_steer = 1.0f;
        v.feedback_terminal_scale = 3.5f;
        v.feedback_cov_regularization = 0.18f;
        v.feedback_cov_blend = 0.75f;
        v.feedback_lqr_blend = 0.35f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_ref";
        v.use_feedback = true;
        v.feedback_mode = 7;
        v.feedback_gain_scale = 1.0f;
        v.feedback_noise_accel = 0.0f;
        v.feedback_noise_steer = 0.0f;
        v.feedback_longitudinal_gain = 0.0f;
        v.feedback_speed_gain = 0.0f;
        v.feedback_lateral_gain = 0.0f;
        v.feedback_heading_gain = 0.0f;
        v.feedback_setpoint_blend = 0.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_release";
        v.use_feedback = true;
        v.feedback_mode = 7;
        v.sampling_lambda = 1.0f / 5.0f;
        v.feedback_gain_scale = 1.0f;
        v.feedback_noise_accel = 0.0f;
        v.feedback_noise_steer = 0.0f;
        v.feedback_longitudinal_gain = 0.0f;
        v.feedback_speed_gain = 0.0f;
        v.feedback_lateral_gain = 0.0f;
        v.feedback_heading_gain = 0.0f;
        v.feedback_setpoint_blend = 0.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "grad_only_3";
        v.use_sampling = false;
        v.use_gradient = true;
        v.grad_steps = 3;
        v.alpha = 0.004f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_1";
        v.use_gradient = true;
        v.grad_steps = 1;
        v.alpha = 0.010f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_3";
        v.use_gradient = true;
        v.grad_steps = 3;
        v.alpha = 0.006f;
        variants.push_back(v);
    }

    if (!planner_names.empty()) {
        vector<PlannerVariant> filtered;
        for (const auto& wanted : planner_names) {
            auto it = find_if(variants.begin(), variants.end(),
                              [&](const PlannerVariant& v) { return v.name == wanted; });
            if (it == variants.end()) {
                fprintf(stderr, "Unknown planner: %s\n", wanted.c_str());
                return 1;
            }
            filtered.push_back(*it);
        }
        variants.swap(filtered);
    }

    if (k_values.empty()) k_values = quick ? vector<int>{1024, 4096} : vector<int>{1024, 2048, 4096};
    if (seed_count <= 0) seed_count = quick ? 2 : 4;

    vector<EpisodeMetrics> rows;
    rows.reserve(scenarios.size() * variants.size() * k_values.size() * seed_count);
    vector<TraceRow> trace_rows;
    bool trace_enabled = !trace_csv_path.empty();
    if (trace_enabled && trace_max_steps <= 0) trace_max_steps = 64;

    for (size_t si = 0; si < scenarios.size(); si++) {
        const Scenario& scenario = scenarios[si];
        CUDA_CHECK(cudaMemcpyToSymbol(d_obstacles_bench, scenario.obstacles, sizeof(Obstacle) * scenario.n_obs));
        if (scenario.n_dyn_obs > 0) {
            CUDA_CHECK(cudaMemcpyToSymbol(d_dynamic_obstacles_bench, scenario.dynamic_obstacles,
                                          sizeof(DynamicObstacle) * scenario.n_dyn_obs));
        }
        for (int k_samples : k_values) {
            for (size_t vi = 0; vi < variants.size(); vi++) {
                const PlannerVariant& variant = variants[vi];
                for (int seed = 0; seed < seed_count; seed++) {
                    int run_seed = static_cast<int>(1000 + si * 100 + vi * 20 + seed * 7 + k_samples);
                    Scenario eval_scenario = instantiate_eval_scenario(scenario, run_seed);
                    EpisodeRunner runner(
                        variant, scenario, eval_scenario, k_samples, DEFAULT_T_HORIZON, run_seed,
                        trace_enabled ? &trace_rows : nullptr, trace_max_steps);
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
    if (trace_enabled) write_trace_csv(trace_rows, trace_csv_path);
    print_summary(rows);
    cout << "CSV saved to " << csv_path << endl;
    if (trace_enabled) cout << "Trace CSV saved to " << trace_csv_path << endl;
    return 0;
}
