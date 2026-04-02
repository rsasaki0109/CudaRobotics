/*************************************************************************
    Diff-MPPI Planar Manipulator Benchmark
    - 2-link planar arm reaching with static and moving workspace obstacles
    - Compares MPPI, covariance-feedback MPPI, and hybrid Diff-MPPI variants
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
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "autodiff_engine.cuh"

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;
using namespace cudabot;

static const int MAX_OBSTACLES = 16;
static const int MAX_DYNAMIC_OBSTACLES = 8;
static const int DEFAULT_T_HORIZON = 36;
static const int BENCH_WARMUP_ITERS = 4;
static const float DEFAULT_LAMBDA = 3.0f;

struct WorkspaceObstacle {
    float x;
    float y;
    float r;
};

struct DynamicObstacle {
    float x;
    float y;
    float vx;
    float vy;
    float r;
};

__constant__ WorkspaceObstacle d_arm_obstacles[MAX_OBSTACLES];
__constant__ DynamicObstacle d_arm_dynamic_obstacles[MAX_DYNAMIC_OBSTACLES];

struct ArmParams {
    float l1 = 0.85f;
    float l2 = 0.75f;
    float dt = 0.05f;
    float max_vel_1 = 2.8f;
    float max_vel_2 = 3.2f;
    float max_torque_1 = 4.0f;
    float max_torque_2 = 4.0f;
    float damping_1 = 0.22f;
    float damping_2 = 0.18f;
    float gravity_1 = 0.85f;
    float gravity_2 = 0.38f;
    float coupling = 0.12f;
};

struct ArmCostParams {
    float goal_x = 1.10f;
    float goal_y = 0.60f;
    float goal_weight = 5.0f;
    float control_weight = 0.04f;
    float velocity_weight = 0.10f;
    float obstacle_weight = 10.0f;
    float obs_influence = 0.18f;
    float terminal_weight = 12.0f;
    float terminal_velocity_weight = 0.50f;
};

struct Scenario {
    string name;
    float start_q1 = -1.3f;
    float start_q2 = 1.6f;
    float start_dq1 = 0.0f;
    float start_dq2 = 0.0f;
    float goal_tol = 0.10f;
    int max_steps = 180;
    ArmParams params;
    ArmCostParams cost_params;
    float grad_alpha_scale = 1.0f;
    int n_obs = 0;
    WorkspaceObstacle obstacles[MAX_OBSTACLES];
    int n_dyn_obs = 0;
    DynamicObstacle dynamic_obstacles[MAX_DYNAMIC_OBSTACLES];
};

struct PlannerVariant {
    string name;
    bool use_feedback = false;
    bool use_gradient = false;
    int feedback_mode = 0;
    int grad_steps = 0;
    float alpha = 0.0f;
    float sampling_lambda = DEFAULT_LAMBDA;
    float torque_sigma_1 = 0.0f;
    float torque_sigma_2 = 0.0f;
    float feedback_gain_scale = 0.0f;
    float feedback_noise_torque_1 = 0.0f;
    float feedback_noise_torque_2 = 0.0f;
    float feedback_q_gain = 0.0f;
    float feedback_dq_gain = 0.0f;
    float feedback_setpoint_blend = 0.0f;
    float feedback_cov_regularization = 0.0f;
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

template <typename Scalar>
__host__ __device__ inline void arm_forward_kinematics(
    const Scalar& q1, const Scalar& q2, const ArmParams& p,
    Scalar& elbow_x, Scalar& elbow_y, Scalar& ee_x, Scalar& ee_y)
{
    elbow_x = Scalar::constant(0.0f) + Scalar::constant(p.l1) * cudabot::cos(q1);
    elbow_y = Scalar::constant(0.0f) + Scalar::constant(p.l1) * cudabot::sin(q1);
    ee_x = elbow_x + Scalar::constant(p.l2) * cudabot::cos(q1 + q2);
    ee_y = elbow_y + Scalar::constant(p.l2) * cudabot::sin(q1 + q2);
}

__host__ __device__ inline void arm_forward_kinematics(
    float q1, float q2, const ArmParams& p,
    float& elbow_x, float& elbow_y, float& ee_x, float& ee_y)
{
    elbow_x = p.l1 * cosf(q1);
    elbow_y = p.l1 * sinf(q1);
    ee_x = elbow_x + p.l2 * cosf(q1 + q2);
    ee_y = elbow_y + p.l2 * sinf(q1 + q2);
}

template <typename Scalar>
__device__ inline Scalar point_obstacle_cost(
    const Scalar& px, const Scalar& py, int n_obs, float influence, float weight)
{
    Scalar cost = Scalar::constant(0.0f);
    for (int i = 0; i < n_obs; i++) {
        Scalar dx = px - Scalar::constant(d_arm_obstacles[i].x);
        Scalar dy = py - Scalar::constant(d_arm_obstacles[i].y);
        Scalar d = cudabot::sqrt(dx * dx + dy * dy + Scalar::constant(1.0e-6f))
                 - Scalar::constant(d_arm_obstacles[i].r);
        if (d.val < influence && d.val > 0.02f) {
            cost = cost + Scalar::constant(weight) / (d * d);
        } else if (d.val <= 0.02f) {
            cost = cost + Scalar::constant(weight * 120.0f);
        }
    }
    return cost;
}

template <typename Scalar>
__device__ inline Scalar point_dynamic_obstacle_cost(
    const Scalar& px, const Scalar& py, float tau, int n_dyn_obs, float influence, float weight)
{
    Scalar cost = Scalar::constant(0.0f);
    for (int i = 0; i < n_dyn_obs; i++) {
        float ox = d_arm_dynamic_obstacles[i].x + d_arm_dynamic_obstacles[i].vx * tau;
        float oy = d_arm_dynamic_obstacles[i].y + d_arm_dynamic_obstacles[i].vy * tau;
        Scalar dx = px - Scalar::constant(ox);
        Scalar dy = py - Scalar::constant(oy);
        Scalar d = cudabot::sqrt(dx * dx + dy * dy + Scalar::constant(1.0e-6f))
                 - Scalar::constant(d_arm_dynamic_obstacles[i].r);
        if (d.val < influence && d.val > 0.02f) {
            cost = cost + Scalar::constant(weight) / (d * d);
        } else if (d.val <= 0.02f) {
            cost = cost + Scalar::constant(weight * 120.0f);
        }
    }
    return cost;
}

template <typename Scalar>
__device__ inline Scalar workspace_obstacle_cost(
    const Scalar& q1, const Scalar& q2, const ArmParams& params,
    int n_obs, int n_dyn_obs, float tau, float influence, float weight)
{
    Scalar elbow_x, elbow_y, ee_x, ee_y;
    arm_forward_kinematics(q1, q2, params, elbow_x, elbow_y, ee_x, ee_y);

    Scalar shoulder_x = Scalar::constant(0.0f);
    Scalar shoulder_y = Scalar::constant(0.0f);
    Scalar wrist_mid_x = elbow_x + Scalar::constant(0.5f) * (ee_x - elbow_x);
    Scalar wrist_mid_y = elbow_y + Scalar::constant(0.5f) * (ee_y - elbow_y);
    Scalar upper_mid_x = Scalar::constant(0.5f) * elbow_x;
    Scalar upper_mid_y = Scalar::constant(0.5f) * elbow_y;
    Scalar forearm_quarter_x = elbow_x + Scalar::constant(0.25f) * (ee_x - elbow_x);
    Scalar forearm_quarter_y = elbow_y + Scalar::constant(0.25f) * (ee_y - elbow_y);
    Scalar forearm_three_quarter_x = elbow_x + Scalar::constant(0.75f) * (ee_x - elbow_x);
    Scalar forearm_three_quarter_y = elbow_y + Scalar::constant(0.75f) * (ee_y - elbow_y);

    Scalar cost = Scalar::constant(0.0f);
    const Scalar points_x[] = {upper_mid_x, elbow_x, forearm_quarter_x, wrist_mid_x, forearm_three_quarter_x, ee_x};
    const Scalar points_y[] = {upper_mid_y, elbow_y, forearm_quarter_y, wrist_mid_y, forearm_three_quarter_y, ee_y};
    (void)shoulder_x;
    (void)shoulder_y;
    for (int i = 0; i < 6; i++) {
        cost = cost + point_obstacle_cost(points_x[i], points_y[i], n_obs, influence, weight);
        cost = cost + point_dynamic_obstacle_cost(points_x[i], points_y[i], tau, n_dyn_obs, influence, weight);
    }
    return cost;
}

__host__ __device__ inline void arm_step(
    float& q1, float& q2, float& dq1, float& dq2,
    float tau1, float tau2, const ArmParams& p)
{
    tau1 = clampf_local(tau1, -p.max_torque_1, p.max_torque_1);
    tau2 = clampf_local(tau2, -p.max_torque_2, p.max_torque_2);

    float ddq1 = tau1
               - p.damping_1 * dq1
               - p.gravity_1 * sinf(q1)
               - p.coupling * sinf(q1 + q2);
    float ddq2 = tau2
               - p.damping_2 * dq2
               - p.gravity_2 * sinf(q1 + q2);

    dq1 = clampf_local(dq1 + p.dt * ddq1, -p.max_vel_1, p.max_vel_1);
    dq2 = clampf_local(dq2 + p.dt * ddq2, -p.max_vel_2, p.max_vel_2);
    q1 = wrap_anglef(q1 + p.dt * dq1);
    q2 = wrap_anglef(q2 + p.dt * dq2);
}

__device__ inline void arm_step_diff(
    Dualf& q1, Dualf& q2, Dualf& dq1, Dualf& dq2,
    Dualf tau1, Dualf tau2, const ArmParams& p)
{
    tau1 = clamp(tau1, -p.max_torque_1, p.max_torque_1);
    tau2 = clamp(tau2, -p.max_torque_2, p.max_torque_2);

    Dualf ddq1 = tau1
               - Dualf::constant(p.damping_1) * dq1
               - Dualf::constant(p.gravity_1) * cudabot::sin(q1)
               - Dualf::constant(p.coupling) * cudabot::sin(q1 + q2);
    Dualf ddq2 = tau2
               - Dualf::constant(p.damping_2) * dq2
               - Dualf::constant(p.gravity_2) * cudabot::sin(q1 + q2);

    dq1 = clamp(dq1 + p.dt * ddq1, -p.max_vel_1, p.max_vel_1);
    dq2 = clamp(dq2 + p.dt * ddq2, -p.max_vel_2, p.max_vel_2);
    q1 = wrap_angle_diff(q1 + p.dt * dq1);
    q2 = wrap_angle_diff(q2 + p.dt * dq2);
}

__device__ inline void arm_jacobian(
    float q1, float q2, float dq1, float dq2, float tau1, float tau2,
    const ArmParams& p, float J[4][6])
{
    for (int col = 0; col < 6; col++) {
        Dualf dq1_var = (col == 0) ? Dualf::variable(q1) : Dualf::constant(q1);
        Dualf dq2_var = (col == 1) ? Dualf::variable(q2) : Dualf::constant(q2);
        Dualf ddq1_var = (col == 2) ? Dualf::variable(dq1) : Dualf::constant(dq1);
        Dualf ddq2_var = (col == 3) ? Dualf::variable(dq2) : Dualf::constant(dq2);
        Dualf dt1_var = (col == 4) ? Dualf::variable(tau1) : Dualf::constant(tau1);
        Dualf dt2_var = (col == 5) ? Dualf::variable(tau2) : Dualf::constant(tau2);
        arm_step_diff(dq1_var, dq2_var, ddq1_var, ddq2_var, dt1_var, dt2_var, p);
        J[0][col] = dq1_var.deriv;
        J[1][col] = dq2_var.deriv;
        J[2][col] = ddq1_var.deriv;
        J[3][col] = ddq2_var.deriv;
    }
}

__host__ __device__ inline float point_margin(float px, float py, const WorkspaceObstacle& obs) {
    float dx = px - obs.x;
    float dy = py - obs.y;
    return sqrtf(dx * dx + dy * dy + 1.0e-6f) - obs.r;
}

__host__ __device__ inline float point_margin(float px, float py, const DynamicObstacle& obs, float tau) {
    float ox = obs.x + obs.vx * tau;
    float oy = obs.y + obs.vy * tau;
    float dx = px - ox;
    float dy = py - oy;
    return sqrtf(dx * dx + dy * dy + 1.0e-6f) - obs.r;
}

static float host_min_margin(float q1, float q2, const Scenario& scenario, int step_index) {
    float elbow_x, elbow_y, ee_x, ee_y;
    arm_forward_kinematics(q1, q2, scenario.params, elbow_x, elbow_y, ee_x, ee_y);
    const float tau = step_index * scenario.params.dt;
    const float px[] = {0.5f * elbow_x, elbow_x, elbow_x + 0.25f * (ee_x - elbow_x),
                        elbow_x + 0.5f * (ee_x - elbow_x), elbow_x + 0.75f * (ee_x - elbow_x), ee_x};
    const float py[] = {0.5f * elbow_y, elbow_y, elbow_y + 0.25f * (ee_y - elbow_y),
                        elbow_y + 0.5f * (ee_y - elbow_y), elbow_y + 0.75f * (ee_y - elbow_y), ee_y};
    float best = 1.0e9f;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < scenario.n_obs; j++) best = min(best, point_margin(px[i], py[i], scenario.obstacles[j]));
        for (int j = 0; j < scenario.n_dyn_obs; j++) best = min(best, point_margin(px[i], py[i], scenario.dynamic_obstacles[j], tau));
    }
    return best;
}

static float host_stage_cost(
    float q1, float q2, float dq1, float dq2, float tau1, float tau2,
    const Scenario& scenario, int step_index)
{
    float elbow_x, elbow_y, ee_x, ee_y;
    arm_forward_kinematics(q1, q2, scenario.params, elbow_x, elbow_y, ee_x, ee_y);
    float dx = ee_x - scenario.cost_params.goal_x;
    float dy = ee_y - scenario.cost_params.goal_y;
    float cost = scenario.cost_params.goal_weight * sqrtf(dx * dx + dy * dy + 1.0e-6f) * scenario.params.dt;
    cost += scenario.cost_params.control_weight * (tau1 * tau1 + tau2 * tau2) * scenario.params.dt;
    cost += scenario.cost_params.velocity_weight * (dq1 * dq1 + dq2 * dq2) * scenario.params.dt;

    const float tau = step_index * scenario.params.dt;
    const float px[] = {0.5f * elbow_x, elbow_x, elbow_x + 0.25f * (ee_x - elbow_x),
                        elbow_x + 0.5f * (ee_x - elbow_x), elbow_x + 0.75f * (ee_x - elbow_x), ee_x};
    const float py[] = {0.5f * elbow_y, elbow_y, elbow_y + 0.25f * (ee_y - elbow_y),
                        elbow_y + 0.5f * (ee_y - elbow_y), elbow_y + 0.75f * (ee_y - elbow_y), ee_y};
    for (int pidx = 0; pidx < 6; pidx++) {
        for (int i = 0; i < scenario.n_obs; i++) {
            float margin = point_margin(px[pidx], py[pidx], scenario.obstacles[i]);
            if (margin <= 0.02f) cost += scenario.cost_params.obstacle_weight * 120.0f;
            else if (margin < scenario.cost_params.obs_influence) cost += scenario.cost_params.obstacle_weight / (margin * margin);
        }
        for (int i = 0; i < scenario.n_dyn_obs; i++) {
            float margin = point_margin(px[pidx], py[pidx], scenario.dynamic_obstacles[i], tau);
            if (margin <= 0.02f) cost += scenario.cost_params.obstacle_weight * 120.0f;
            else if (margin < scenario.cost_params.obs_influence) cost += scenario.cost_params.obstacle_weight / (margin * margin);
        }
    }
    return cost;
}

__host__ __device__ inline float end_effector_distance(
    float q1, float q2, const ArmParams& params, const ArmCostParams& cp)
{
    float elbow_x, elbow_y, ee_x, ee_y;
    arm_forward_kinematics(q1, q2, params, elbow_x, elbow_y, ee_x, ee_y);
    float dx = ee_x - cp.goal_x;
    float dy = ee_y - cp.goal_y;
    return sqrtf(dx * dx + dy * dy + 1.0e-6f);
}

__device__ inline float stage_cost_device(
    float q1, float q2, float dq1, float dq2, float tau1, float tau2,
    const ArmParams& params, const ArmCostParams& cp,
    int n_obs, int n_dyn_obs, float tau)
{
    float elbow_x, elbow_y, ee_x, ee_y;
    arm_forward_kinematics(q1, q2, params, elbow_x, elbow_y, ee_x, ee_y);
    float dx = ee_x - cp.goal_x;
    float dy = ee_y - cp.goal_y;
    float cost = cp.goal_weight * sqrtf(dx * dx + dy * dy + 1.0e-6f) * params.dt;
    cost += cp.control_weight * (tau1 * tau1 + tau2 * tau2) * params.dt;
    cost += cp.velocity_weight * (dq1 * dq1 + dq2 * dq2) * params.dt;

    const float px[] = {0.5f * elbow_x, elbow_x, elbow_x + 0.25f * (ee_x - elbow_x),
                        elbow_x + 0.5f * (ee_x - elbow_x), elbow_x + 0.75f * (ee_x - elbow_x), ee_x};
    const float py[] = {0.5f * elbow_y, elbow_y, elbow_y + 0.25f * (ee_y - elbow_y),
                        elbow_y + 0.5f * (ee_y - elbow_y), elbow_y + 0.75f * (ee_y - elbow_y), ee_y};
    for (int pidx = 0; pidx < 6; pidx++) {
        for (int i = 0; i < n_obs; i++) {
            float margin = point_margin(px[pidx], py[pidx], d_arm_obstacles[i]);
            if (margin <= 0.02f) cost += cp.obstacle_weight * 120.0f;
            else if (margin < cp.obs_influence) cost += cp.obstacle_weight / (margin * margin);
        }
        for (int i = 0; i < n_dyn_obs; i++) {
            float margin = point_margin(px[pidx], py[pidx], d_arm_dynamic_obstacles[i], tau);
            if (margin <= 0.02f) cost += cp.obstacle_weight * 120.0f;
            else if (margin < cp.obs_influence) cost += cp.obstacle_weight / (margin * margin);
        }
    }
    return cost;
}

__host__ __device__ inline float terminal_cost(
    float q1, float q2, float dq1, float dq2, const ArmParams& params, const ArmCostParams& cp)
{
    float ee_dist = end_effector_distance(q1, q2, params, cp);
    return cp.terminal_weight * ee_dist
         + cp.terminal_velocity_weight * (dq1 * dq1 + dq2 * dq2);
}

__device__ inline void terminal_grad(
    float q1, float q2, float dq1, float dq2, const ArmParams& params, const ArmCostParams& cp, float grad[4])
{
    for (int var = 0; var < 4; var++) {
        Dualf dq1_var = (var == 0) ? Dualf::variable(q1) : Dualf::constant(q1);
        Dualf dq2_var = (var == 1) ? Dualf::variable(q2) : Dualf::constant(q2);
        Dualf ddq1_var = (var == 2) ? Dualf::variable(dq1) : Dualf::constant(dq1);
        Dualf ddq2_var = (var == 3) ? Dualf::variable(dq2) : Dualf::constant(dq2);
        Dualf elbow_x, elbow_y, ee_x, ee_y;
        arm_forward_kinematics(dq1_var, dq2_var, params, elbow_x, elbow_y, ee_x, ee_y);
        Dualf dx = ee_x - Dualf::constant(cp.goal_x);
        Dualf dy = ee_y - Dualf::constant(cp.goal_y);
        Dualf ee_dist = cudabot::sqrt(dx * dx + dy * dy + Dualf::constant(1.0e-6f));
        Dualf cost = Dualf::constant(cp.terminal_weight) * ee_dist
                   + Dualf::constant(cp.terminal_velocity_weight) * (ddq1_var * ddq1_var + ddq2_var * ddq2_var);
        grad[var] = cost.deriv;
    }
}

__device__ inline void stage_cost_grad(
    float q1, float q2, float dq1, float dq2, float tau1, float tau2,
    const ArmParams& params, const ArmCostParams& cp,
    int n_obs, int n_dyn_obs, float tau, float grad[6])
{
    for (int var = 0; var < 6; var++) {
        Dualf dq1_var = (var == 0) ? Dualf::variable(q1) : Dualf::constant(q1);
        Dualf dq2_var = (var == 1) ? Dualf::variable(q2) : Dualf::constant(q2);
        Dualf ddq1_var = (var == 2) ? Dualf::variable(dq1) : Dualf::constant(dq1);
        Dualf ddq2_var = (var == 3) ? Dualf::variable(dq2) : Dualf::constant(dq2);
        Dualf dt1_var = (var == 4) ? Dualf::variable(tau1) : Dualf::constant(tau1);
        Dualf dt2_var = (var == 5) ? Dualf::variable(tau2) : Dualf::constant(tau2);

        Dualf elbow_x, elbow_y, ee_x, ee_y;
        arm_forward_kinematics(dq1_var, dq2_var, params, elbow_x, elbow_y, ee_x, ee_y);
        Dualf dx = ee_x - Dualf::constant(cp.goal_x);
        Dualf dy = ee_y - Dualf::constant(cp.goal_y);
        Dualf ee_dist = cudabot::sqrt(dx * dx + dy * dy + Dualf::constant(1.0e-6f));
        Dualf cost = Dualf::constant(cp.goal_weight * params.dt) * ee_dist
                   + Dualf::constant(cp.control_weight * params.dt) * (dt1_var * dt1_var + dt2_var * dt2_var)
                   + Dualf::constant(cp.velocity_weight * params.dt) * (ddq1_var * ddq1_var + ddq2_var * ddq2_var)
                   + workspace_obstacle_cost(dq1_var, dq2_var, params, n_obs, n_dyn_obs, tau,
                                             cp.obs_influence, cp.obstacle_weight);
        grad[var] = cost.deriv;
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
    ArmParams params,
    ArmCostParams cp,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    float sigma_1,
    float sigma_2)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float q1 = sq1;
    float q2 = sq2;
    float dq1 = sdq1;
    float dq2 = sdq2;
    float total_cost = 0.0f;

    if (d_rollout_states != nullptr) {
        d_rollout_states[k * (T + 1) * 4 + 0] = q1;
        d_rollout_states[k * (T + 1) * 4 + 1] = q2;
        d_rollout_states[k * (T + 1) * 4 + 2] = dq1;
        d_rollout_states[k * (T + 1) * 4 + 3] = dq2;
    }

    for (int t = 0; t < T; t++) {
        float tau1 = d_nominal[t * 2 + 0] + curand_normal(&local_rng) * sigma_1;
        float tau2 = d_nominal[t * 2 + 1] + curand_normal(&local_rng) * sigma_2;
        tau1 = clampf_local(tau1, -params.max_torque_1, params.max_torque_1);
        tau2 = clampf_local(tau2, -params.max_torque_2, params.max_torque_2);
        d_perturbed[k * T * 2 + t * 2 + 0] = tau1;
        d_perturbed[k * T * 2 + t * 2 + 1] = tau2;

        arm_step(q1, q2, dq1, dq2, tau1, tau2, params);
        if (d_rollout_states != nullptr) {
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 0] = q1;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 1] = q2;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 2] = dq1;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 3] = dq2;
        }
        float tau_world = (start_step + t + 1) * params.dt;
        total_cost += stage_cost_device(q1, q2, dq1, dq2, tau1, tau2, params, cp, n_obs, n_dyn_obs, tau_world);
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
    if (sum_w > 0.0f) {
        for (int k = 0; k < K; k++) d_weights[k] /= sum_w;
    }
}

__global__ void update_controls_kernel(float* d_nominal, const float* d_perturbed, const float* d_weights, int K, int T) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float tau1 = 0.0f;
    float tau2 = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = d_weights[k];
        tau1 += w * d_perturbed[k * T * 2 + t * 2 + 0];
        tau2 += w * d_perturbed[k * T * 2 + t * 2 + 1];
    }
    d_nominal[t * 2 + 0] = tau1;
    d_nominal[t * 2 + 1] = tau2;
}

__global__ void rollout_nominal_kernel(
    float sq1, float sq2, float sdq1, float sdq2,
    const float* d_nominal, float* d_states,
    ArmParams params, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float q1 = sq1;
    float q2 = sq2;
    float dq1 = sdq1;
    float dq2 = sdq2;
    d_states[0] = q1;
    d_states[1] = q2;
    d_states[2] = dq1;
    d_states[3] = dq2;
    for (int t = 0; t < T; t++) {
        arm_step(q1, q2, dq1, dq2, d_nominal[t * 2 + 0], d_nominal[t * 2 + 1], params);
        d_states[(t + 1) * 4 + 0] = q1;
        d_states[(t + 1) * 4 + 1] = q2;
        d_states[(t + 1) * 4 + 2] = dq1;
        d_states[(t + 1) * 4 + 3] = dq2;
    }
}

__device__ inline bool invert_4x4(const float A[4][4], float invA[4][4]) {
    float aug[4][8];
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) aug[row][col] = A[row][col];
        for (int col = 0; col < 4; col++) aug[row][4 + col] = (row == col) ? 1.0f : 0.0f;
    }
    for (int pivot = 0; pivot < 4; pivot++) {
        int best_row = pivot;
        float best_val = fabsf(aug[pivot][pivot]);
        for (int row = pivot + 1; row < 4; row++) {
            float val = fabsf(aug[row][pivot]);
            if (val > best_val) {
                best_val = val;
                best_row = row;
            }
        }
        if (best_val < 1.0e-8f) return false;
        if (best_row != pivot) {
            for (int col = 0; col < 8; col++) {
                float tmp = aug[pivot][col];
                aug[pivot][col] = aug[best_row][col];
                aug[best_row][col] = tmp;
            }
        }
        float inv_diag = 1.0f / aug[pivot][pivot];
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

    float eps = fmaxf(1.0e-4f, regularization);
    for (int t = 0; t < T; t++) {
        float Sigma_xx[4][4] = {};
        float Sigma_ux[2][4] = {};
        float q1_nom = d_nominal_states[t * 4 + 0];
        float q2_nom = d_nominal_states[t * 4 + 1];
        float dq1_nom = d_nominal_states[t * 4 + 2];
        float dq2_nom = d_nominal_states[t * 4 + 3];
        float tau1_nom = d_nominal[t * 2 + 0];
        float tau2_nom = d_nominal[t * 2 + 1];

        for (int k = 0; k < K; k++) {
            float w = d_weights[k];
            const float* state = &d_rollout_states[k * (T + 1) * 4 + t * 4];
            float x_dev[4];
            x_dev[0] = wrap_anglef(state[0] - q1_nom);
            x_dev[1] = wrap_anglef(state[1] - q2_nom);
            x_dev[2] = state[2] - dq1_nom;
            x_dev[3] = state[3] - dq2_nom;
            float u_dev[2];
            u_dev[0] = d_perturbed[k * T * 2 + t * 2 + 0] - tau1_nom;
            u_dev[1] = d_perturbed[k * T * 2 + t * 2 + 1] - tau2_nom;
            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 4; col++) Sigma_xx[row][col] += w * x_dev[row] * x_dev[col];
            }
            for (int row = 0; row < 2; row++) {
                for (int col = 0; col < 4; col++) Sigma_ux[row][col] += w * u_dev[row] * x_dev[col];
            }
        }

        for (int i = 0; i < 4; i++) Sigma_xx[i][i] += eps;
        float inv_xx[4][4];
        if (!invert_4x4(Sigma_xx, inv_xx)) {
            for (int i = 0; i < 4; i++) Sigma_xx[i][i] += 10.0f * eps;
            invert_4x4(Sigma_xx, inv_xx);
        }

        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 4; col++) {
                float gain = 0.0f;
                for (int k = 0; k < 4; k++) gain += Sigma_ux[row][k] * inv_xx[k][col];
                d_feedback_gains[t * 8 + row * 4 + col] = -gain;
            }
        }
    }
}

__global__ void compute_rollout_initial_gradients_kernel(
    const float* d_rollout_states,
    const float* d_perturbed,
    float* d_rollout_init_grads,
    ArmParams params,
    ArmCostParams cp,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const float* states = &d_rollout_states[k * (T + 1) * 4];
    const float* controls = &d_perturbed[k * T * 2];

    float adj[4];
    terminal_grad(states[T * 4 + 0], states[T * 4 + 1], states[T * 4 + 2], states[T * 4 + 3], params, cp, adj);
    for (int t = T - 1; t >= 0; t--) {
        float q1 = states[t * 4 + 0];
        float q2 = states[t * 4 + 1];
        float dq1 = states[t * 4 + 2];
        float dq2 = states[t * 4 + 3];
        float tau1 = controls[t * 2 + 0];
        float tau2 = controls[t * 2 + 1];
        float J[4][6];
        float stage_grad_vec[6];
        float next_adj[4];
        float tau_world = (start_step + t) * params.dt;
        arm_jacobian(q1, q2, dq1, dq2, tau1, tau2, params, J);
        stage_cost_grad(q1, q2, dq1, dq2, tau1, tau2, params, cp, n_obs, n_dyn_obs, tau_world, stage_grad_vec);
        for (int col = 0; col < 4; col++) {
            next_adj[col] = stage_grad_vec[col];
            for (int row = 0; row < 4; row++) next_adj[col] += J[row][col] * adj[row];
        }
        for (int i = 0; i < 4; i++) adj[i] = next_adj[i];
    }
    for (int i = 0; i < 4; i++) d_rollout_init_grads[k * 4 + i] = adj[i];
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

    float nominal_tau1 = d_nominal[0];
    float nominal_tau2 = d_nominal[1];
    for (int j = 0; j < 4; j++) {
        float gain_tau1 = 0.0f;
        float gain_tau2 = 0.0f;
        for (int k = 0; k < K; k++) {
            float w = d_weights[k];
            float centered_grad = d_rollout_init_grads[k * 4 + j] - weighted_grad[j];
            float delta_tau1 = d_perturbed[k * T * 2 + 0] - nominal_tau1;
            float delta_tau2 = d_perturbed[k * T * 2 + 1] - nominal_tau2;
            gain_tau1 += -inv_lambda * w * centered_grad * delta_tau1;
            gain_tau2 += -inv_lambda * w * centered_grad * delta_tau2;
        }
        d_feedback_gains[0 * 4 + j] = gain_tau1;
        d_feedback_gains[1 * 4 + j] = gain_tau2;
    }
}

__global__ void broadcast_first_feedback_gain_kernel(float* d_feedback_gains, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * 8;
    if (idx >= total) return;
    int base_idx = idx % 8;
    d_feedback_gains[idx] = d_feedback_gains[base_idx];
}

__global__ void rollout_feedback_kernel(
    float sq1, float sq2, float sdq1, float sdq2,
    const float* d_nominal,
    const float* d_nominal_states,
    const float* d_feedback_gains,
    float* d_costs,
    float* d_perturbed,
    curandState* d_rng,
    ArmParams params,
    ArmCostParams cp,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    float gain_scale,
    float noise_sigma_1,
    float noise_sigma_2,
    float q_gain,
    float dq_gain,
    float setpoint_blend)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float q1 = sq1;
    float q2 = sq2;
    float dq1 = sdq1;
    float dq2 = sdq2;
    float total_cost = 0.0f;

    for (int t = 0; t < T; t++) {
        int t_next = min(t + 1, T);
        float q1_nom = (1.0f - setpoint_blend) * d_nominal_states[t * 4 + 0] + setpoint_blend * d_nominal_states[t_next * 4 + 0];
        float q2_nom = (1.0f - setpoint_blend) * d_nominal_states[t * 4 + 1] + setpoint_blend * d_nominal_states[t_next * 4 + 1];
        float dq1_nom = (1.0f - setpoint_blend) * d_nominal_states[t * 4 + 2] + setpoint_blend * d_nominal_states[t_next * 4 + 2];
        float dq2_nom = (1.0f - setpoint_blend) * d_nominal_states[t * 4 + 3] + setpoint_blend * d_nominal_states[t_next * 4 + 3];
        float q_err_1 = wrap_anglef(q1 - q1_nom);
        float q_err_2 = wrap_anglef(q2 - q2_nom);
        float dq_err_1 = dq1 - dq1_nom;
        float dq_err_2 = dq2 - dq2_nom;
        const float* K_t = &d_feedback_gains[t * 8];
        float fb_1 = K_t[0] * q_err_1 + K_t[1] * q_err_2 + K_t[2] * dq_err_1 + K_t[3] * dq_err_2;
        float fb_2 = K_t[4] * q_err_1 + K_t[5] * q_err_2 + K_t[6] * dq_err_1 + K_t[7] * dq_err_2;

        float tau1 = d_nominal[t * 2 + 0]
                   + curand_normal(&local_rng) * noise_sigma_1
                   - gain_scale * fb_1
                   - q_gain * q_err_1
                   - dq_gain * dq_err_1;
        float tau2 = d_nominal[t * 2 + 1]
                   + curand_normal(&local_rng) * noise_sigma_2
                   - gain_scale * fb_2
                   - q_gain * q_err_2
                   - dq_gain * dq_err_2;
        tau1 = clampf_local(tau1, -params.max_torque_1, params.max_torque_1);
        tau2 = clampf_local(tau2, -params.max_torque_2, params.max_torque_2);
        d_perturbed[k * T * 2 + t * 2 + 0] = tau1;
        d_perturbed[k * T * 2 + t * 2 + 1] = tau2;

        arm_step(q1, q2, dq1, dq2, tau1, tau2, params);
        float tau_world = (start_step + t + 1) * params.dt;
        total_cost += stage_cost_device(q1, q2, dq1, dq2, tau1, tau2, params, cp, n_obs, n_dyn_obs, tau_world);
    }

    total_cost += terminal_cost(q1, q2, dq1, dq2, params, cp);
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

__global__ void compute_gradient_kernel(
    const float* d_states, const float* d_nominal, float* d_grad,
    ArmParams params, ArmCostParams cp, int n_obs, int n_dyn_obs, int start_step, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float adj[4];
    terminal_grad(d_states[T * 4 + 0], d_states[T * 4 + 1], d_states[T * 4 + 2], d_states[T * 4 + 3], params, cp, adj);
    for (int t = T - 1; t >= 0; t--) {
        float q1 = d_states[t * 4 + 0];
        float q2 = d_states[t * 4 + 1];
        float dq1 = d_states[t * 4 + 2];
        float dq2 = d_states[t * 4 + 3];
        float tau1 = d_nominal[t * 2 + 0];
        float tau2 = d_nominal[t * 2 + 1];
        float J[4][6];
        float stage_grad_vec[6];
        float next_adj[4];
        float tau_world = (start_step + t) * params.dt;
        arm_jacobian(q1, q2, dq1, dq2, tau1, tau2, params, J);
        stage_cost_grad(q1, q2, dq1, dq2, tau1, tau2, params, cp, n_obs, n_dyn_obs, tau_world, stage_grad_vec);
        d_grad[t * 2 + 0] = stage_grad_vec[4];
        d_grad[t * 2 + 1] = stage_grad_vec[5];
        for (int row = 0; row < 4; row++) {
            d_grad[t * 2 + 0] += J[row][4] * adj[row];
            d_grad[t * 2 + 1] += J[row][5] * adj[row];
        }
        for (int col = 0; col < 4; col++) {
            next_adj[col] = stage_grad_vec[col];
            for (int row = 0; row < 4; row++) next_adj[col] += J[row][col] * adj[row];
        }
        for (int i = 0; i < 4; i++) adj[i] = next_adj[i];
    }
}

__global__ void gradient_step_kernel(float* d_nominal, const float* d_grad, int T, float alpha, const ArmParams params) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    d_nominal[t * 2 + 0] = clampf_local(d_nominal[t * 2 + 0] - alpha * d_grad[t * 2 + 0], -params.max_torque_1, params.max_torque_1);
    d_nominal[t * 2 + 1] = clampf_local(d_nominal[t * 2 + 1] - alpha * d_grad[t * 2 + 1], -params.max_torque_2, params.max_torque_2);
}

class EpisodeRunner {
public:
    EpisodeRunner(const PlannerVariant& variant, const Scenario& scenario, int k_samples, int t_horizon, int seed)
        : variant_(variant), scenario_(scenario), k_samples_(k_samples), t_horizon_(t_horizon), seed_(seed) {
        reset_state();
        h_nominal_.assign(t_horizon_ * 2, 0.0f);
        h_costs_.assign(k_samples_, 0.0f);
        h_states_.assign((t_horizon_ + 1) * 4, 0.0f);
        CUDA_CHECK(cudaMalloc(&d_nominal_, h_nominal_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs_, h_costs_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_, k_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perturbed_, k_samples_ * t_horizon_ * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rollout_states_, k_samples_ * (t_horizon_ + 1) * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rollout_init_grads_, k_samples_ * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_states_, h_states_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_, h_nominal_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_feedback_gains_, t_horizon_ * 8 * sizeof(float)));
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
            float ee_dist = end_effector_distance(q1_, q2_, scenario_.params, scenario_.cost_params);
            min_goal_distance_ = min(min_goal_distance_, ee_dist);
            if (ee_dist < scenario_.goal_tol) {
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
            float tau1 = h_nominal_[0];
            float tau2 = h_nominal_[1];
            arm_step(q1_, q2_, dq1_, dq2_, tau1, tau2, scenario_.params);
            cumulative_cost_ += host_stage_cost(q1_, q2_, dq1_, dq2_, tau1, tau2, scenario_, step + 1);

            if (host_min_margin(q1_, q2_, scenario_, step + 1) <= 0.02f) collisions_++;

            for (int t = 0; t < t_horizon_ - 1; t++) {
                h_nominal_[t * 2 + 0] = h_nominal_[(t + 1) * 2 + 0];
                h_nominal_[t * 2 + 1] = h_nominal_[(t + 1) * 2 + 1];
            }
            h_nominal_[(t_horizon_ - 1) * 2 + 0] = 0.0f;
            h_nominal_[(t_horizon_ - 1) * 2 + 1] = 0.0f;
            steps_taken_ = step + 1;
        }

        auto episode_end = chrono::steady_clock::now();
        float final_distance = end_effector_distance(q1_, q2_, scenario_.params, scenario_.cost_params);
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

    void controller_update(int start_step) {
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size() * sizeof(float), cudaMemcpyHostToDevice));
        int block = 256;

        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
            q1_, q2_, dq1_, dq2_, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
            scenario_.params, scenario_.cost_params, scenario_.n_obs, scenario_.n_dyn_obs,
            start_step, k_samples_, t_horizon_, variant_.torque_sigma_1, variant_.torque_sigma_2);
        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
        update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
            d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);

        if (variant_.use_feedback) {
            rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                q1_, q2_, dq1_, dq2_, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                scenario_.params, scenario_.cost_params, scenario_.n_obs, scenario_.n_dyn_obs,
                start_step, k_samples_, t_horizon_, variant_.torque_sigma_1, variant_.torque_sigma_2);
            compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
            update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);

            rollout_nominal_kernel<<<1, 1>>>(q1_, q2_, dq1_, dq2_, d_nominal_, d_states_, scenario_.params, t_horizon_);
            rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                q1_, q2_, dq1_, dq2_, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                scenario_.params, scenario_.cost_params, scenario_.n_obs, scenario_.n_dyn_obs,
                start_step, k_samples_, t_horizon_, variant_.feedback_noise_torque_1, variant_.feedback_noise_torque_2);
            compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
            if (variant_.feedback_mode == 2) {
                compute_rollout_initial_gradients_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                    d_rollout_states_, d_perturbed_, d_rollout_init_grads_,
                    scenario_.params, scenario_.cost_params, scenario_.n_obs, scenario_.n_dyn_obs,
                    start_step, k_samples_, t_horizon_);
                compute_reference_feedback_gain_kernel<<<1, 1>>>(
                    d_nominal_, d_perturbed_, d_weights_, d_rollout_init_grads_, d_feedback_gains_,
                    variant_.sampling_lambda, k_samples_, t_horizon_);
                broadcast_first_feedback_gain_kernel<<<(t_horizon_ * 8 + block - 1) / block, block>>>(
                    d_feedback_gains_, t_horizon_);
            } else {
                compute_covariance_feedback_gains_kernel<<<1, 1>>>(
                    d_nominal_, d_states_, d_perturbed_, d_rollout_states_, d_weights_, d_feedback_gains_,
                    k_samples_, t_horizon_, variant_.feedback_cov_regularization);
            }
            rollout_feedback_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                q1_, q2_, dq1_, dq2_, d_nominal_, d_states_, d_feedback_gains_,
                d_costs_, d_perturbed_, d_rng_, scenario_.params, scenario_.cost_params,
                scenario_.n_obs, scenario_.n_dyn_obs, start_step, k_samples_, t_horizon_,
                variant_.feedback_gain_scale, variant_.feedback_noise_torque_1, variant_.feedback_noise_torque_2,
                variant_.feedback_q_gain, variant_.feedback_dq_gain, variant_.feedback_setpoint_blend);
            compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
            update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);
        }

        if (variant_.use_gradient) {
            for (int gs = 0; gs < variant_.grad_steps; gs++) {
                rollout_nominal_kernel<<<1, 1>>>(q1_, q2_, dq1_, dq2_, d_nominal_, d_states_, scenario_.params, t_horizon_);
                compute_gradient_kernel<<<1, 1>>>(
                    d_states_, d_nominal_, d_grad_, scenario_.params, scenario_.cost_params,
                    scenario_.n_obs, scenario_.n_dyn_obs, start_step, t_horizon_);
                gradient_step_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                    d_nominal_, d_grad_, t_horizon_,
                    variant_.alpha * scenario_.grad_alpha_scale, scenario_.params);
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void warmup_controller() {
        for (int iter = 0; iter < BENCH_WARMUP_ITERS; iter++) controller_update(0);
    }

    void reset_state() {
        q1_ = scenario_.start_q1;
        q2_ = scenario_.start_q2;
        dq1_ = scenario_.start_dq1;
        dq2_ = scenario_.start_dq2;
        steps_taken_ = 0;
        collisions_ = 0;
        reached_goal_ = false;
        cumulative_cost_ = 0.0f;
        min_goal_distance_ = end_effector_distance(q1_, q2_, scenario_.params, scenario_.cost_params);
    }

    PlannerVariant variant_;
    Scenario scenario_;
    int k_samples_;
    int t_horizon_;
    int seed_;

    float q1_ = 0.0f;
    float q2_ = 0.0f;
    float dq1_ = 0.0f;
    float dq2_ = 0.0f;
    int steps_taken_ = 0;
    int collisions_ = 0;
    bool reached_goal_ = false;
    float cumulative_cost_ = 0.0f;
    float min_goal_distance_ = 0.0f;

    vector<float> h_nominal_;
    vector<float> h_costs_;
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

static Scenario make_arm_static_shelf_scene() {
    Scenario s;
    s.name = "arm_static_shelf";
    s.start_q1 = -1.45f;
    s.start_q2 = 1.90f;
    s.goal_tol = 0.15f;
    s.max_steps = 220;
    s.grad_alpha_scale = 1.15f;
    s.cost_params.goal_x = 1.08f;
    s.cost_params.goal_y = 0.44f;
    s.cost_params.goal_weight = 6.6f;
    s.cost_params.control_weight = 0.032f;
    s.cost_params.velocity_weight = 0.09f;
    s.cost_params.obstacle_weight = 10.0f;
    s.cost_params.obs_influence = 0.18f;
    s.cost_params.terminal_weight = 17.0f;
    const WorkspaceObstacle obs[] = {
        {0.70f, 0.22f, 0.12f},
        {0.74f, 0.74f, 0.12f},
        {0.84f, 0.42f, 0.09f},
        {0.98f, 0.78f, 0.09f},
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    return s;
}

static Scenario make_arm_dynamic_sweep_scene() {
    Scenario s;
    s.name = "arm_dynamic_sweep";
    s.start_q1 = -1.25f;
    s.start_q2 = 1.72f;
    s.goal_tol = 0.15f;
    s.max_steps = 240;
    s.grad_alpha_scale = 1.25f;
    s.cost_params.goal_x = 1.09f;
    s.cost_params.goal_y = 0.46f;
    s.cost_params.goal_weight = 6.8f;
    s.cost_params.control_weight = 0.032f;
    s.cost_params.velocity_weight = 0.10f;
    s.cost_params.obstacle_weight = 11.0f;
    s.cost_params.obs_influence = 0.18f;
    s.cost_params.terminal_weight = 18.0f;
    const WorkspaceObstacle obs[] = {
        {0.60f, 0.24f, 0.13f},
        {0.84f, 0.26f, 0.10f},
        {0.90f, 0.88f, 0.10f},
    };
    const DynamicObstacle dyn[] = {
        {1.28f, 0.49f, -0.30f, 0.0f, 0.09f},
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
        s.steps_sum += r.steps;
        s.final_sum += r.final_distance;
        s.min_sum += r.min_goal_distance;
        s.cost_sum += r.cumulative_cost;
        s.ms_sum += r.avg_control_ms;
    }
    cout << "=== benchmark_diff_mppi_manipulator summary ===" << endl;
    for (const auto& kv : stats) {
        const SummaryStats& s = kv.second;
        float n = static_cast<float>(s.episodes);
        printf("%s : success=%.2f steps=%.1f final_dist=%.2f min_dist=%.2f cost=%.1f avg_ms=%.2f\n",
               kv.first.c_str(),
               s.successes / n,
               static_cast<float>(s.steps_sum / n),
               static_cast<float>(s.final_sum / n),
               static_cast<float>(s.min_sum / n),
               static_cast<float>(s.cost_sum / n),
               static_cast<float>(s.ms_sum / n));
    }
}

int main(int argc, char** argv) {
    bool quick = false;
    string csv_path = "build/benchmark_diff_mppi_manipulator.csv";
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
    }

    ensure_build_dir();

    vector<Scenario> all_scenarios = {
        make_arm_static_shelf_scene(),
        make_arm_dynamic_sweep_scene(),
    };
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
    } else {
        scenarios = all_scenarios;
    }

    vector<PlannerVariant> variants;
    {
        PlannerVariant v;
        v.name = "mppi";
        v.torque_sigma_1 = 1.10f;
        v.torque_sigma_2 = 1.10f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_cov";
        v.use_feedback = true;
        v.feedback_mode = 1;
        v.torque_sigma_1 = 1.05f;
        v.torque_sigma_2 = 1.05f;
        v.feedback_gain_scale = 0.70f;
        v.feedback_noise_torque_1 = 0.80f;
        v.feedback_noise_torque_2 = 0.80f;
        v.feedback_q_gain = 0.42f;
        v.feedback_dq_gain = 0.22f;
        v.feedback_setpoint_blend = 0.20f;
        v.feedback_cov_regularization = 0.25f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_ref";
        v.use_feedback = true;
        v.feedback_mode = 2;
        v.torque_sigma_1 = 1.05f;
        v.torque_sigma_2 = 1.05f;
        v.feedback_gain_scale = 1.00f;
        v.feedback_noise_torque_1 = 0.80f;
        v.feedback_noise_torque_2 = 0.80f;
        v.feedback_q_gain = 0.0f;
        v.feedback_dq_gain = 0.0f;
        v.feedback_setpoint_blend = 0.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_1";
        v.use_gradient = true;
        v.grad_steps = 1;
        v.alpha = 0.060f;
        v.torque_sigma_1 = 1.10f;
        v.torque_sigma_2 = 1.10f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_3";
        v.use_gradient = true;
        v.grad_steps = 3;
        v.alpha = 0.015f;
        v.torque_sigma_1 = 1.10f;
        v.torque_sigma_2 = 1.10f;
        variants.push_back(v);
    }

    if (!planner_names.empty()) {
        vector<PlannerVariant> filtered;
        for (const auto& wanted : planner_names) {
            auto it = find_if(variants.begin(), variants.end(), [&](const PlannerVariant& v) { return v.name == wanted; });
            if (it == variants.end()) {
                fprintf(stderr, "Unknown planner: %s\n", wanted.c_str());
                return 1;
            }
            filtered.push_back(*it);
        }
        variants.swap(filtered);
    }

    if (k_values.empty()) k_values = quick ? vector<int>{256, 512} : vector<int>{256, 512, 1024, 2048};
    if (seed_count <= 0) seed_count = quick ? 2 : 4;

    vector<EpisodeMetrics> rows;
    rows.reserve(scenarios.size() * variants.size() * k_values.size() * seed_count);

    for (size_t si = 0; si < scenarios.size(); si++) {
        const Scenario& scenario = scenarios[si];
        CUDA_CHECK(cudaMemcpyToSymbol(d_arm_obstacles, scenario.obstacles, sizeof(WorkspaceObstacle) * scenario.n_obs));
        if (scenario.n_dyn_obs > 0) {
            CUDA_CHECK(cudaMemcpyToSymbol(d_arm_dynamic_obstacles, scenario.dynamic_obstacles,
                                          sizeof(DynamicObstacle) * scenario.n_dyn_obs));
        }
        for (int k_samples : k_values) {
            for (size_t vi = 0; vi < variants.size(); vi++) {
                const PlannerVariant& variant = variants[vi];
                for (int seed = 0; seed < seed_count; seed++) {
                    int run_seed = static_cast<int>(2000 + si * 100 + vi * 20 + seed * 7 + k_samples);
                    EpisodeRunner runner(variant, scenario, k_samples, DEFAULT_T_HORIZON, run_seed);
                    EpisodeMetrics metrics = runner.run();
                    rows.push_back(metrics);
                    printf("[%s] %s K=%d seed=%d success=%d steps=%d final_dist=%.3f avg_ms=%.2f collisions=%d\n",
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
