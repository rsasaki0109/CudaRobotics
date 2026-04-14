/*************************************************************************
    Diff-MPPI MuJoCo Benchmark
    - Standard InvertedPendulum-v4 XML from Gymnasium / MuJoCo
    - Ground-truth dynamics are stepped with the MuJoCo C API on CPU
    - GPU rollouts use a differentiable approximate cart-pole model
    - Compares MPPI, reference-feedback MPPI, and hybrid Diff-MPPI
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
static const int CTRL_DIM = 1;
static const int DEFAULT_T_HORIZON = 35;
static const int DEFAULT_MAX_STEPS = 250;
static const int DEFAULT_FRAME_SKIP = 2;
static const float DEFAULT_LAMBDA = 2.5f;
static const float DEFAULT_NOISE_SIGMA = 0.32f;

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
    float gravity = 9.81f;
    float masscart = 1.0f;
    float masspole = 0.1f;
    float length = 0.6f;
    float tau = 0.04f;  // MuJoCo timestep 0.02 with frame_skip 2.
    float ctrl_limit = 3.0f;
    float force_scale = 8.5f;  // Tuned to match MuJoCo's one-step response near upright.
    float x_limit = 1.0f;
    float theta_limit = 0.2f;

    __host__ __device__ float total_mass() const { return masscart + masspole; }
    __host__ __device__ float polemass_length() const { return masspole * length; }
};

struct CostParams {
    float angle_weight = 18.0f;
    float x_weight = 1.5f;
    float x_dot_weight = 0.25f;
    float theta_dot_weight = 0.85f;
    float action_weight = 0.03f;
    float terminal_angle_weight = 50.0f;
    float terminal_x_weight = 3.0f;
    float terminal_x_dot_weight = 0.5f;
    float terminal_theta_dot_weight = 2.5f;
    float failure_penalty = 750.0f;
    float success_angle = 0.05f;
    float success_x = 0.10f;
    float success_x_dot = 0.20f;
    float success_theta_dot = 0.35f;
    int success_window = 20;
};

struct Scenario {
    string name;
    float x_lo = 0.0f;
    float x_hi = 0.0f;
    float x_dot_lo = 0.0f;
    float x_dot_hi = 0.0f;
    float theta_lo = 0.0f;
    float theta_hi = 0.0f;
    float theta_dot_lo = 0.0f;
    float theta_dot_hi = 0.0f;
    int max_steps = DEFAULT_MAX_STEPS;
    ApproxParams approx;
    CostParams cost_params;
};

struct PlannerVariant {
    string name;
    bool use_feedback = false;
    bool use_gradient = false;
    int grad_steps = 0;
    float alpha = 0.0f;
    float noise_sigma = DEFAULT_NOISE_SIGMA;
    float feedback_gain_scale = 1.0f;
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
__host__ __device__ Scalar angle_error_diff(const Scalar& theta) {
    return cudabot::atan2(cudabot::sin(theta), cudabot::cos(theta));
}

__host__ __device__ inline void approx_step(
    float& x, float& x_dot, float& theta, float& theta_dot, float action, const ApproxParams& p)
{
    action = clampf_local(action, -p.ctrl_limit, p.ctrl_limit);
    const float dt = 0.5f * p.tau;
    for (int substep = 0; substep < 2; substep++) {
        float force = action * p.force_scale;
        float costheta = cosf(theta);
        float sintheta = sinf(theta);
        float temp = (force + p.polemass_length() * theta_dot * theta_dot * sintheta) / p.total_mass();
        float thetaacc = (p.gravity * sintheta - costheta * temp) /
            (p.length * (4.0f / 3.0f - p.masspole * costheta * costheta / p.total_mass()));
        float xacc = temp - p.polemass_length() * thetaacc * costheta / p.total_mass();

        x_dot += dt * xacc;
        theta_dot += dt * thetaacc;
        x += dt * x_dot;
        theta += dt * theta_dot;
    }
}

__device__ inline void approx_step_diff(
    Dualf& x, Dualf& x_dot, Dualf& theta, Dualf& theta_dot, Dualf action, const ApproxParams& p)
{
    Dualf total_mass = Dualf::constant(p.total_mass());
    Dualf polemass_length = Dualf::constant(p.polemass_length());
    Dualf gravity = Dualf::constant(p.gravity);
    Dualf length = Dualf::constant(p.length);
    action = clamp(action, -p.ctrl_limit, p.ctrl_limit);
    Dualf dt = Dualf::constant(0.5f * p.tau);
    for (int substep = 0; substep < 2; substep++) {
        Dualf force = action * p.force_scale;
        Dualf costheta = cudabot::cos(theta);
        Dualf sintheta = cudabot::sin(theta);
        Dualf temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        Dualf thetaacc = (gravity * sintheta - costheta * temp) /
            (length * (Dualf::constant(4.0f / 3.0f) - Dualf::constant(p.masspole) * costheta * costheta / total_mass));
        Dualf xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        x_dot = x_dot + dt * xacc;
        theta_dot = theta_dot + dt * thetaacc;
        x = x + dt * x_dot;
        theta = theta + dt * theta_dot;
    }
}

__host__ __device__ inline float stage_cost(
    float x, float x_dot, float theta, float theta_dot, float action, const CostParams& cp)
{
    float angle_err = wrap_anglef(theta);
    return cp.angle_weight * angle_err * angle_err
         + cp.x_weight * x * x
         + cp.x_dot_weight * x_dot * x_dot
         + cp.theta_dot_weight * theta_dot * theta_dot
         + cp.action_weight * action * action;
}

__host__ __device__ inline float terminal_cost(
    float x, float x_dot, float theta, float theta_dot, const CostParams& cp)
{
    float angle_err = wrap_anglef(theta);
    return cp.terminal_angle_weight * angle_err * angle_err
         + cp.terminal_x_weight * x * x
         + cp.terminal_x_dot_weight * x_dot * x_dot
         + cp.terminal_theta_dot_weight * theta_dot * theta_dot;
}

__device__ inline void stage_cost_grad(
    float x, float x_dot, float theta, float theta_dot, float action, const CostParams& cp, float grad[5])
{
    for (int var = 0; var < 5; var++) {
        Dualf dx = (var == 0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dxd = (var == 1) ? Dualf::variable(x_dot) : Dualf::constant(x_dot);
        Dualf dtheta = (var == 2) ? Dualf::variable(theta) : Dualf::constant(theta);
        Dualf dthetad = (var == 3) ? Dualf::variable(theta_dot) : Dualf::constant(theta_dot);
        Dualf du = (var == 4) ? Dualf::variable(action) : Dualf::constant(action);
        Dualf angle_err = angle_error_diff(dtheta);
        Dualf cost = cp.angle_weight * angle_err * angle_err
                   + cp.x_weight * dx * dx
                   + cp.x_dot_weight * dxd * dxd
                   + cp.theta_dot_weight * dthetad * dthetad
                   + cp.action_weight * du * du;
        grad[var] = cost.deriv;
    }
}

__device__ inline void terminal_grad(
    float x, float x_dot, float theta, float theta_dot, const CostParams& cp, float grad[4])
{
    for (int var = 0; var < 4; var++) {
        Dualf dx = (var == 0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dxd = (var == 1) ? Dualf::variable(x_dot) : Dualf::constant(x_dot);
        Dualf dtheta = (var == 2) ? Dualf::variable(theta) : Dualf::constant(theta);
        Dualf dthetad = (var == 3) ? Dualf::variable(theta_dot) : Dualf::constant(theta_dot);
        Dualf angle_err = angle_error_diff(dtheta);
        Dualf cost = cp.terminal_angle_weight * angle_err * angle_err
                   + cp.terminal_x_weight * dx * dx
                   + cp.terminal_x_dot_weight * dxd * dxd
                   + cp.terminal_theta_dot_weight * dthetad * dthetad;
        grad[var] = cost.deriv;
    }
}

__device__ inline void approx_jacobian(
    float x, float x_dot, float theta, float theta_dot, float action, const ApproxParams& p, float J[4][5])
{
    for (int col = 0; col < 5; col++) {
        Dualf dx = (col == 0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dxd = (col == 1) ? Dualf::variable(x_dot) : Dualf::constant(x_dot);
        Dualf dtheta = (col == 2) ? Dualf::variable(theta) : Dualf::constant(theta);
        Dualf dthetad = (col == 3) ? Dualf::variable(theta_dot) : Dualf::constant(theta_dot);
        Dualf du = (col == 4) ? Dualf::variable(action) : Dualf::constant(action);
        approx_step_diff(dx, dxd, dtheta, dthetad, du, p);
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
    ApproxParams params,
    CostParams cost_params,
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
        action = clampf_local(action, -params.ctrl_limit, params.ctrl_limit);
        d_perturbed[k * T + t] = action;
        approx_step(x, x_dot, theta, theta_dot, action, params);
        total_cost += stage_cost(x, x_dot, theta, theta_dot, action, cost_params);
        if (fabsf(x) > params.x_limit || fabsf(wrap_anglef(theta)) > params.theta_limit) {
            total_cost += cost_params.failure_penalty;
            break;
        }
    }

    total_cost += terminal_cost(x, x_dot, theta, theta_dot, cost_params);
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

__global__ void rollout_feedback_kernel(
    float sx, float sx_dot, float stheta, float stheta_dot,
    const float* d_nominal,
    const float* d_reference_states,
    const float* d_feedback_gains,
    float* d_costs,
    float* d_perturbed,
    curandState* d_rng,
    ApproxParams params,
    CostParams cost_params,
    float noise_sigma,
    float gain_scale,
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
        const float* ref = &d_reference_states[t * STATE_DIM];
        const float* gains = &d_feedback_gains[t * STATE_DIM];
        float err[STATE_DIM] = {
            x - ref[0],
            x_dot - ref[1],
            wrap_anglef(theta - ref[2]),
            theta_dot - ref[3]
        };
        float feedback = 0.0f;
        for (int j = 0; j < STATE_DIM; j++) feedback += gains[j] * err[j];

        float action = d_nominal[t] - gain_scale * feedback + curand_normal(&local_rng) * noise_sigma;
        action = clampf_local(action, -params.ctrl_limit, params.ctrl_limit);
        d_perturbed[k * T + t] = action;
        approx_step(x, x_dot, theta, theta_dot, action, params);
        total_cost += stage_cost(x, x_dot, theta, theta_dot, action, cost_params);
        if (fabsf(x) > params.x_limit || fabsf(wrap_anglef(theta)) > params.theta_limit) {
            total_cost += cost_params.failure_penalty;
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
    d_nominal[t] = action;
}

__global__ void rollout_nominal_kernel(
    float sx, float sx_dot, float stheta, float stheta_dot,
    const float* d_nominal,
    float* d_states,
    ApproxParams params,
    int T)
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
        approx_step(x, x_dot, theta, theta_dot, d_nominal[t], params);
        d_states[(t + 1) * STATE_DIM + 0] = x;
        d_states[(t + 1) * STATE_DIM + 1] = x_dot;
        d_states[(t + 1) * STATE_DIM + 2] = theta;
        d_states[(t + 1) * STATE_DIM + 3] = theta_dot;
    }
}

__global__ void compute_gradient_kernel(
    const float* d_states,
    const float* d_nominal,
    float* d_grad,
    ApproxParams params,
    CostParams cost_params,
    int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float adj[STATE_DIM];
    terminal_grad(
        d_states[T * STATE_DIM + 0], d_states[T * STATE_DIM + 1],
        d_states[T * STATE_DIM + 2], d_states[T * STATE_DIM + 3],
        cost_params, adj);

    for (int t = T - 1; t >= 0; t--) {
        float x = d_states[t * STATE_DIM + 0];
        float x_dot = d_states[t * STATE_DIM + 1];
        float theta = d_states[t * STATE_DIM + 2];
        float theta_dot = d_states[t * STATE_DIM + 3];
        float action = d_nominal[t];

        float J[STATE_DIM][STATE_DIM + CTRL_DIM];
        float stage_grad_vec[STATE_DIM + CTRL_DIM];
        float next_adj[STATE_DIM];

        approx_jacobian(x, x_dot, theta, theta_dot, action, params, J);
        stage_cost_grad(x, x_dot, theta, theta_dot, action, cost_params, stage_grad_vec);

        d_grad[t] = stage_grad_vec[4];
        for (int row = 0; row < STATE_DIM; row++) d_grad[t] += J[row][4] * adj[row];

        for (int col = 0; col < STATE_DIM; col++) {
            next_adj[col] = stage_grad_vec[col];
            for (int row = 0; row < STATE_DIM; row++) next_adj[col] += J[row][col] * adj[row];
        }
        for (int i = 0; i < STATE_DIM; i++) adj[i] = next_adj[i];
    }
}

__global__ void gradient_step_kernel(float* d_nominal, const float* d_grad, float alpha, float ctrl_limit, int T) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    d_nominal[t] = clampf_local(d_nominal[t] - alpha * d_grad[t], -ctrl_limit, ctrl_limit);
}

static vector<int> parse_int_list(const string& text) {
    vector<int> values;
    string token;
    stringstream ss(text);
    while (getline(ss, token, ',')) {
        if (!token.empty()) values.push_back(atoi(token.c_str()));
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
        source_dir + "/mujoco_models/inverted_pendulum.xml",
        "mujoco_models/inverted_pendulum.xml",
        "../mujoco_models/inverted_pendulum.xml",
        "../../mujoco_models/inverted_pendulum.xml",
    };
    for (const auto& path : candidates) {
        if (file_exists(path)) return path;
    }
    return candidates.front();
}

static Scenario make_standard_balance() {
    Scenario s;
    s.name = "inverted_pendulum_v4";
    s.x_lo = -0.01f; s.x_hi = 0.01f;
    s.x_dot_lo = -0.01f; s.x_dot_hi = 0.01f;
    s.theta_lo = -0.01f; s.theta_hi = 0.01f;
    s.theta_dot_lo = -0.01f; s.theta_dot_hi = 0.01f;
    s.max_steps = 250;
    s.cost_params.success_window = 25;
    return s;
}

static Scenario make_wide_reset_balance() {
    Scenario s;
    s.name = "inverted_pendulum_wide_reset";
    s.x_lo = -0.12f; s.x_hi = 0.12f;
    s.x_dot_lo = -0.18f; s.x_dot_hi = 0.18f;
    s.theta_lo = -0.16f; s.theta_hi = 0.16f;
    s.theta_dot_lo = -0.35f; s.theta_dot_hi = 0.35f;
    s.max_steps = 250;
    s.cost_params.angle_weight = 20.0f;
    s.cost_params.terminal_angle_weight = 55.0f;
    s.cost_params.success_window = 18;
    return s;
}

static void state_from_mujoco(const mjData* data, float& x, float& x_dot, float& theta, float& theta_dot) {
    x = static_cast<float>(data->qpos[0]);
    theta = static_cast<float>(data->qpos[1]);
    x_dot = static_cast<float>(data->qvel[0]);
    theta_dot = static_cast<float>(data->qvel[1]);
}

static void set_mujoco_state(const mjModel* model, mjData* data, float x, float x_dot, float theta, float theta_dot) {
    mj_resetData(model, data);
    data->qpos[0] = x;
    data->qpos[1] = theta;
    data->qvel[0] = x_dot;
    data->qvel[1] = theta_dot;
    mj_forward(model, data);
}

static void mujoco_step_env(const mjModel* model, mjData* data, float action, int frame_skip, float ctrl_limit) {
    action = clampf_local(action, -ctrl_limit, ctrl_limit);
    for (int i = 0; i < frame_skip; i++) {
        data->ctrl[0] = action;
        mj_step(model, data);
    }
}

static void approx_rollout_host(
    float sx, float sx_dot, float stheta, float stheta_dot,
    const vector<float>& nominal, const ApproxParams& params,
    vector<float>& states)
{
    const int T = static_cast<int>(nominal.size());
    states.assign((T + 1) * STATE_DIM, 0.0f);
    float x = sx;
    float x_dot = sx_dot;
    float theta = stheta;
    float theta_dot = stheta_dot;
    states[0] = x;
    states[1] = x_dot;
    states[2] = theta;
    states[3] = theta_dot;
    for (int t = 0; t < T; t++) {
        approx_step(x, x_dot, theta, theta_dot, nominal[t], params);
        states[(t + 1) * STATE_DIM + 0] = x;
        states[(t + 1) * STATE_DIM + 1] = x_dot;
        states[(t + 1) * STATE_DIM + 2] = theta;
        states[(t + 1) * STATE_DIM + 3] = theta_dot;
    }
}

static float approx_rollout_cost_host(
    float sx, float sx_dot, float stheta, float stheta_dot,
    const vector<float>& nominal, const ApproxParams& params, const CostParams& cost_params)
{
    float x = sx;
    float x_dot = sx_dot;
    float theta = stheta;
    float theta_dot = stheta_dot;
    float total_cost = 0.0f;
    for (float action : nominal) {
        approx_step(x, x_dot, theta, theta_dot, action, params);
        total_cost += stage_cost(x, x_dot, theta, theta_dot, action, cost_params);
        if (fabsf(x) > params.x_limit || fabsf(wrap_anglef(theta)) > params.theta_limit) {
            total_cost += cost_params.failure_penalty;
            break;
        }
    }
    total_cost += terminal_cost(x, x_dot, theta, theta_dot, cost_params);
    return total_cost;
}

static void approx_linearize_fd(
    float x, float x_dot, float theta, float theta_dot, float action, const ApproxParams& params,
    float A[STATE_DIM][STATE_DIM], float B[STATE_DIM])
{
    const float eps = 1.0e-3f;

    float base_next[STATE_DIM];
    {
        float bx = x;
        float bxd = x_dot;
        float bth = theta;
        float bthd = theta_dot;
        approx_step(bx, bxd, bth, bthd, action, params);
        base_next[0] = bx;
        base_next[1] = bxd;
        base_next[2] = bth;
        base_next[3] = bthd;
    }

    for (int col = 0; col < STATE_DIM; col++) {
        float px = x;
        float pxd = x_dot;
        float pth = theta;
        float pthd = theta_dot;
        if (col == 0) px += eps;
        if (col == 1) pxd += eps;
        if (col == 2) pth += eps;
        if (col == 3) pthd += eps;
        approx_step(px, pxd, pth, pthd, action, params);
        A[0][col] = (px - base_next[0]) / eps;
        A[1][col] = (pxd - base_next[1]) / eps;
        A[2][col] = (pth - base_next[2]) / eps;
        A[3][col] = (pthd - base_next[3]) / eps;
    }

    float ux = x;
    float uxd = x_dot;
    float uth = theta;
    float uthd = theta_dot;
    approx_step(ux, uxd, uth, uthd, action + eps, params);
    B[0] = (ux - base_next[0]) / eps;
    B[1] = (uxd - base_next[1]) / eps;
    B[2] = (uth - base_next[2]) / eps;
    B[3] = (uthd - base_next[3]) / eps;
}

static void compute_lqr_feedback_gains(
    const vector<float>& reference_states,
    const vector<float>& nominal,
    const ApproxParams& params,
    const CostParams& cost_params,
    vector<float>& gains)
{
    const int T = static_cast<int>(nominal.size());
    gains.assign(T * STATE_DIM, 0.0f);

    float P[STATE_DIM][STATE_DIM] = {};
    P[0][0] = cost_params.terminal_x_weight;
    P[1][1] = cost_params.terminal_x_dot_weight;
    P[2][2] = cost_params.terminal_angle_weight;
    P[3][3] = cost_params.terminal_theta_dot_weight;

    const float Q_diag[STATE_DIM] = {
        cost_params.x_weight,
        cost_params.x_dot_weight,
        cost_params.angle_weight,
        cost_params.theta_dot_weight
    };
    const float R = fmaxf(1.0e-4f, cost_params.action_weight);

    for (int t = T - 1; t >= 0; t--) {
        const float* s = &reference_states[t * STATE_DIM];
        float A[STATE_DIM][STATE_DIM];
        float B[STATE_DIM];
        approx_linearize_fd(s[0], s[1], s[2], s[3], nominal[t], params, A, B);

        float PB[STATE_DIM] = {};
        float PA[STATE_DIM][STATE_DIM] = {};
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                PB[i] += P[i][j] * B[j];
                for (int k = 0; k < STATE_DIM; k++) {
                    PA[i][j] += P[i][k] * A[k][j];
                }
            }
        }

        float BtPB = 0.0f;
        for (int i = 0; i < STATE_DIM; i++) BtPB += B[i] * PB[i];
        float S = R + BtPB;

        float K[STATE_DIM] = {};
        for (int j = 0; j < STATE_DIM; j++) {
            float BtPA = 0.0f;
            for (int i = 0; i < STATE_DIM; i++) BtPA += B[i] * PA[i][j];
            K[j] = BtPA / fmaxf(1.0e-5f, S);
            gains[t * STATE_DIM + j] = K[j];
        }

        float AtPA[STATE_DIM][STATE_DIM] = {};
        float AtPB[STATE_DIM] = {};
        float nextP[STATE_DIM][STATE_DIM] = {};

        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                for (int k = 0; k < STATE_DIM; k++) {
                    AtPA[i][j] += A[k][i] * PA[k][j];
                }
            }
            for (int k = 0; k < STATE_DIM; k++) AtPB[i] += A[k][i] * PB[k];
        }

        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                nextP[i][j] = AtPA[i][j] - AtPB[i] * K[j];
            }
            nextP[i][i] += Q_diag[i];
        }

        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) P[i][j] = nextP[i][j];
        }
    }
}

class EpisodeRunner {
public:
    EpisodeRunner(
        const PlannerVariant& variant,
        const Scenario& scenario,
        const mjModel* model,
        int frame_skip,
        int k_samples,
        int t_horizon,
        int seed)
        : variant_(variant),
          scenario_(scenario),
          model_(model),
          frame_skip_(frame_skip),
          k_samples_(k_samples),
          t_horizon_(t_horizon),
          seed_(seed)
    {
        data_ = mj_makeData(model_);
        h_nominal_.assign(t_horizon_, 0.0f);
        h_grad_.assign(t_horizon_, 0.0f);
        h_reference_states_.assign((t_horizon_ + 1) * STATE_DIM, 0.0f);
        h_feedback_gains_.assign(t_horizon_ * STATE_DIM, 0.0f);

        CUDA_CHECK(cudaMalloc(&d_nominal_, t_horizon_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs_, k_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perturbed_, k_samples_ * t_horizon_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_, k_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_states_, (t_horizon_ + 1) * STATE_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_, t_horizon_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_reference_states_, (t_horizon_ + 1) * STATE_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_feedback_gains_, t_horizon_ * STATE_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_, k_samples_ * sizeof(curandState)));

        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), t_horizon_ * sizeof(float), cudaMemcpyHostToDevice));
        reset_rng();
    }

    ~EpisodeRunner() {
        if (d_nominal_) cudaFree(d_nominal_);
        if (d_costs_) cudaFree(d_costs_);
        if (d_perturbed_) cudaFree(d_perturbed_);
        if (d_weights_) cudaFree(d_weights_);
        if (d_states_) cudaFree(d_states_);
        if (d_grad_) cudaFree(d_grad_);
        if (d_reference_states_) cudaFree(d_reference_states_);
        if (d_feedback_gains_) cudaFree(d_feedback_gains_);
        if (d_rng_) cudaFree(d_rng_);
        if (data_) mj_deleteData(data_);
    }

    EpisodeMetrics run() {
        warmup_controller();
        reset_state();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), t_horizon_ * sizeof(float), cudaMemcpyHostToDevice));
        reset_rng();

        auto episode_begin = chrono::steady_clock::now();
        float control_ms_total = 0.0f;
        float cumulative_cost = 0.0f;
        float best_error = stabilization_error(x_, x_dot_, theta_, theta_dot_);
        bool success = false;
        bool terminated = false;
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
            mujoco_step_env(model_, data_, action, frame_skip_, scenario_.approx.ctrl_limit);
            sync_state_from_mujoco();

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

            if (!isfinite(x_) || !isfinite(x_dot_) || !isfinite(theta_) || !isfinite(theta_dot_)
                || fabsf(x_) > scenario_.approx.x_limit || fabsf(wrap_anglef(theta_)) > scenario_.approx.theta_limit) {
                cumulative_cost += scenario_.cost_params.failure_penalty;
                terminated = true;
                break;
            }

            shift_nominal();
        }

        if (!success && !terminated && executed_steps >= scenario_.max_steps) {
            success = true;
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
        metrics.collision_free = terminated ? 0 : 1;
        metrics.success = success ? 1 : 0;
        metrics.steps = executed_steps;
        metrics.final_distance = final_error;
        metrics.min_goal_distance = best_error;
        metrics.cumulative_cost = cumulative_cost;
        metrics.collisions = terminated ? 1 : 0;
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

    void reset_state() {
        sample_initial_state();
        set_mujoco_state(model_, data_, x_, x_dot_, theta_, theta_dot_);
        sync_state_from_mujoco();
    }

    void sync_state_from_mujoco() {
        state_from_mujoco(data_, x_, x_dot_, theta_, theta_dot_);
    }

    void reset_rng() {
        const int block = 256;
        init_curand_kernel<<<(k_samples_ + block - 1) / block, block>>>(d_rng_, k_samples_, 1000ULL + static_cast<unsigned long long>(seed_));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void compute_feedback_pass_inputs() {
        approx_rollout_host(x_, x_dot_, theta_, theta_dot_, h_nominal_, scenario_.approx, h_reference_states_);
        compute_lqr_feedback_gains(h_reference_states_, h_nominal_, scenario_.approx, scenario_.cost_params, h_feedback_gains_);
        CUDA_CHECK(cudaMemcpy(d_reference_states_, h_reference_states_.data(),
                              h_reference_states_.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_feedback_gains_, h_feedback_gains_.data(),
                              h_feedback_gains_.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void controller_update() {
        const int block = 256;
        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
            x_, x_dot_, theta_, theta_dot_, d_nominal_, d_costs_, d_perturbed_, d_rng_,
            scenario_.approx, scenario_.cost_params, variant_.noise_sigma, k_samples_, t_horizon_);
        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, DEFAULT_LAMBDA);
        update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
            d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);

        if (variant_.use_feedback) {
            CUDA_CHECK(cudaMemcpy(h_nominal_.data(), d_nominal_, t_horizon_ * sizeof(float), cudaMemcpyDeviceToHost));
            compute_feedback_pass_inputs();
            rollout_feedback_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                x_, x_dot_, theta_, theta_dot_, d_nominal_, d_reference_states_, d_feedback_gains_,
                d_costs_, d_perturbed_, d_rng_, scenario_.approx, scenario_.cost_params,
                variant_.noise_sigma, variant_.feedback_gain_scale, k_samples_, t_horizon_);
            compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, DEFAULT_LAMBDA);
            update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);
        }

        if (variant_.use_gradient) {
            CUDA_CHECK(cudaMemcpy(h_nominal_.data(), d_nominal_, t_horizon_ * sizeof(float), cudaMemcpyDeviceToHost));
            float accepted_cost = approx_rollout_cost_host(
                x_, x_dot_, theta_, theta_dot_, h_nominal_, scenario_.approx, scenario_.cost_params);
            for (int gs = 0; gs < variant_.grad_steps; gs++) {
                rollout_nominal_kernel<<<1, 1>>>(
                    x_, x_dot_, theta_, theta_dot_, d_nominal_, d_states_, scenario_.approx, t_horizon_);
                compute_gradient_kernel<<<1, 1>>>(
                    d_states_, d_nominal_, d_grad_, scenario_.approx, scenario_.cost_params, t_horizon_);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_grad_.data(), d_grad_, t_horizon_ * sizeof(float), cudaMemcpyDeviceToHost));

                vector<float> candidate = h_nominal_;
                for (int t = 0; t < t_horizon_; t++) {
                    candidate[t] = clampf_local(
                        candidate[t] - variant_.alpha * h_grad_[t],
                        -scenario_.approx.ctrl_limit, scenario_.approx.ctrl_limit);
                }

                float candidate_cost = approx_rollout_cost_host(
                    x_, x_dot_, theta_, theta_dot_, candidate, scenario_.approx, scenario_.cost_params);
                if (candidate_cost + 1.0e-4f < accepted_cost) {
                    h_nominal_.swap(candidate);
                    accepted_cost = candidate_cost;
                    CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(),
                                          t_horizon_ * sizeof(float), cudaMemcpyHostToDevice));
                } else {
                    break;
                }
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
        reset_state();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), t_horizon_ * sizeof(float), cudaMemcpyHostToDevice));
        controller_update();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), t_horizon_ * sizeof(float), cudaMemcpyHostToDevice));
    }

    PlannerVariant variant_;
    Scenario scenario_;
    const mjModel* model_ = nullptr;
    mjData* data_ = nullptr;
    int frame_skip_ = DEFAULT_FRAME_SKIP;
    int k_samples_ = 0;
    int t_horizon_ = 0;
    int seed_ = 0;

    float x_ = 0.0f;
    float x_dot_ = 0.0f;
    float theta_ = 0.0f;
    float theta_dot_ = 0.0f;

    vector<float> h_nominal_;
    vector<float> h_grad_;
    vector<float> h_reference_states_;
    vector<float> h_feedback_gains_;

    float* d_nominal_ = nullptr;
    float* d_costs_ = nullptr;
    float* d_perturbed_ = nullptr;
    float* d_weights_ = nullptr;
    float* d_states_ = nullptr;
    float* d_grad_ = nullptr;
    float* d_reference_states_ = nullptr;
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

    cout << "=== benchmark_diff_mppi_mujoco summary ===" << endl;
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
    string csv_path = "build/benchmark_diff_mppi_mujoco.csv";
    string model_path;
    vector<int> k_values;
    vector<string> scenario_names;
    vector<string> planner_names;
    int seed_count = -1;
    int frame_skip = DEFAULT_FRAME_SKIP;
    float override_feedback_gain_scale = -1.0f;
    int override_grad_steps = -1;
    float override_alpha = -1.0f;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--quick") quick = true;
        else if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
        else if (arg == "--model-path" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--frame-skip" && i + 1 < argc) frame_skip = max(1, atoi(argv[++i]));
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
    if (model->nq < 2 || model->nv < 2 || model->nu < 1) {
        cerr << "Unexpected MuJoCo model dimensions in " << model_path << endl;
        mj_deleteModel(model);
        return 1;
    }

    vector<Scenario> scenarios = {
        make_standard_balance(),
        make_wide_reset_balance(),
    };
    if (!scenario_names.empty()) {
        vector<Scenario> filtered;
        for (const auto& scenario : scenarios) {
            if (find(scenario_names.begin(), scenario_names.end(), scenario.name) != scenario_names.end()) {
                filtered.push_back(scenario);
            }
        }
        scenarios.swap(filtered);
    }
    if (scenarios.empty()) {
        cerr << "No scenarios selected." << endl;
        mj_deleteModel(model);
        return 1;
    }

    vector<PlannerVariant> variants = {
        {"mppi", false, false, 0, 0.0f, 0.34f, 1.0f},
        {"feedback_mppi_ref", true, false, 0, 0.0f, 0.28f, 1.0f},
        {"diff_mppi_3", false, true, 3, 0.001f, 0.28f, 1.0f},
    };
    if (!planner_names.empty()) {
        vector<PlannerVariant> filtered;
        for (const auto& variant : variants) {
            if (find(planner_names.begin(), planner_names.end(), variant.name) != planner_names.end()) {
                filtered.push_back(variant);
            }
        }
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

    if (k_values.empty()) k_values = quick ? vector<int>{256, 512} : vector<int>{256, 512, 1024};
    if (seed_count < 0) seed_count = quick ? 2 : 4;

    vector<EpisodeMetrics> rows;
    rows.reserve(scenarios.size() * variants.size() * k_values.size() * seed_count);

    for (size_t si = 0; si < scenarios.size(); si++) {
        const auto& scenario = scenarios[si];
        for (int ks : k_values) {
            for (size_t vi = 0; vi < variants.size(); vi++) {
                const auto& variant = variants[vi];
                for (int seed = 0; seed < seed_count; seed++) {
                    int run_seed = static_cast<int>(5000 + si * 100 + vi * 20 + seed * 7 + ks);
                    EpisodeRunner runner(variant, scenario, model, frame_skip, ks, DEFAULT_T_HORIZON, run_seed);
                    EpisodeMetrics m = runner.run();
                    rows.push_back(m);
                    printf("[%s] %s K=%d seed=%d success=%d steps=%d final_dist=%.3f avg_ms=%.2f fail=%d\n",
                           scenario.name.c_str(), variant.name.c_str(), ks, seed,
                           m.success, m.steps, m.final_distance, m.avg_control_ms, m.collisions);
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
