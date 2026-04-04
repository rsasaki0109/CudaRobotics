/*************************************************************************
    Diff-MPPI 7-DOF Manipulator Benchmark
    - Panda-like 7-DOF serial arm with 3D workspace obstacles
    - Second-order joint dynamics with gravity and damping
    - Compares MPPI, reference-feedback MPPI, and hybrid Diff-MPPI
    - CSV output compatible with existing Diff-MPPI summary scripts
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

static const int NDOF = 7;
static const int STATE_DIM = NDOF * 2;   // q[7] + dq[7] = 14
static const int CTRL_DIM = NDOF;         // tau[7]
static const int GAINS_PER_STEP = CTRL_DIM * STATE_DIM;  // 7 * 14 = 98

static const int MAX_OBSTACLES = 16;
static const int MAX_DYNAMIC_OBSTACLES = 8;
static const int DEFAULT_T_HORIZON = 30;
static const int BENCH_WARMUP_ITERS = 4;
static const float DEFAULT_LAMBDA = 3.0f;

struct Obstacle3D {
    float x, y, z, r;
};

struct DynamicObstacle3D {
    float x, y, z;
    float vx, vy, vz;
    float r;
};

__constant__ Obstacle3D d_obs_7dof[MAX_OBSTACLES];
__constant__ DynamicObstacle3D d_dyn_obs_7dof[MAX_DYNAMIC_OBSTACLES];

// Panda-like DH parameters (modified DH convention simplified)
// Link lengths and offsets chosen to approximate Franka Emika Panda geometry
struct ArmParams7 {
    float dt = 0.04f;
    float max_vel[NDOF]    = {2.18f, 2.18f, 2.18f, 2.18f, 2.61f, 2.61f, 2.61f};
    float max_torque[NDOF] = {87.0f, 87.0f, 87.0f, 87.0f, 12.0f, 12.0f, 12.0f};
    float damping[NDOF]    = {2.5f, 2.5f, 2.0f, 2.0f, 0.8f, 0.8f, 0.6f};
    float gravity_comp[NDOF] = {0.0f, 25.0f, 5.0f, 18.0f, 1.5f, 1.0f, 0.2f};
    // Simplified link lengths along DH chain (d and a parameters combined)
    float d[NDOF] = {0.333f, 0.0f, 0.316f, 0.0f, 0.384f, 0.0f, 0.107f};
    float a[NDOF] = {0.0f, 0.0f, 0.0825f, -0.0825f, 0.0f, 0.088f, 0.0f};
};

struct CostParams7 {
    float goal_x = 0.5f;
    float goal_y = 0.0f;
    float goal_z = 0.4f;
    float goal_weight = 8.0f;
    float control_weight = 0.0002f;
    float velocity_weight = 0.01f;
    float obstacle_weight = 15.0f;
    float obs_influence = 0.12f;
    float terminal_weight = 25.0f;
    float terminal_velocity_weight = 0.3f;
};

struct Scenario {
    string name;
    float start_q[NDOF] = {0.0f, -0.78f, 0.0f, -2.36f, 0.0f, 1.57f, 0.78f};
    float start_dq[NDOF] = {};
    float goal_tol = 0.08f;
    int max_steps = 200;
    ArmParams7 params;
    CostParams7 cost_params;
    float grad_alpha_scale = 1.0f;
    int n_obs = 0;
    Obstacle3D obstacles[MAX_OBSTACLES];
    int n_dyn_obs = 0;
    DynamicObstacle3D dynamic_obstacles[MAX_DYNAMIC_OBSTACLES];
};

struct PlannerVariant {
    string name;
    bool use_feedback = false;
    bool use_gradient = false;
    int feedback_mode = 0;
    int grad_steps = 0;
    float alpha = 0.0f;
    float grad_clip = 50.0f;
    float sampling_lambda = DEFAULT_LAMBDA;
    float torque_sigma[NDOF] = {8.0f, 8.0f, 6.0f, 6.0f, 2.0f, 2.0f, 1.5f};
    float feedback_gain_scale = 0.0f;
    float feedback_noise_sigma[NDOF] = {};
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
    double steps_sum = 0;
    double final_sum = 0;
    double min_sum = 0;
    double cost_sum = 0;
    double ms_sum = 0;
};

__host__ __device__ inline float clampf_local(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__host__ __device__ inline float wrap_anglef(float a) {
    a = fmodf(a + M_PI, 2.0f * M_PI);
    if (a < 0.0f) a += 2.0f * M_PI;
    return a - M_PI;
}

// ======================== Forward Kinematics ========================
// Simplified Panda FK: compute positions of each joint frame origin
// Uses a reduced model: alternating rotations about z and x axes
// Joint axes: z, z, z, z, z, z, z  (revolute about local z)
// We approximate with a planar-ish chain in 3D

__host__ __device__ inline void fk_joint_positions(
    const float q[NDOF], const ArmParams7& p,
    float pos[NDOF + 1][3])
{
    // Base frame at origin
    pos[0][0] = 0.0f; pos[0][1] = 0.0f; pos[0][2] = 0.0f;

    // Cumulative rotation: we model the Panda as a chain of rotations
    // Joint 1: rotate about z at base, offset d[0] along z
    // Joint 2: rotate about y (shoulder pitch), no offset
    // Joint 3: rotate about z (elbow twist), offset d[2] along z
    // Joint 4: rotate about y (elbow pitch)
    // Joint 5: rotate about z (wrist twist), offset d[4] along z
    // Joint 6: rotate about y (wrist pitch)
    // Joint 7: rotate about z (flange), offset d[6] along z

    // Simplified: accumulate a 3D position using rotation matrices
    // We use a compact approach: track position + rotation as 3x3 matrix
    float R[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    float px = 0.0f, py = 0.0f, pz = 0.0f;

    // Helper: rotate R by angle about axis (0=z, 1=y)
    // Then translate by local offset
    for (int j = 0; j < NDOF; j++) {
        float c = cosf(q[j]);
        float s = sinf(q[j]);
        float Rn[3][3];

        if (j == 0 || j == 2 || j == 4 || j == 6) {
            // Rotate about local z
            for (int row = 0; row < 3; row++) {
                Rn[row][0] = R[row][0] * c - R[row][1] * s;
                Rn[row][1] = R[row][0] * s + R[row][1] * c;
                Rn[row][2] = R[row][2];
            }
        } else {
            // Rotate about local y (joints 1, 3, 5)
            for (int row = 0; row < 3; row++) {
                Rn[row][0] = R[row][0] * c + R[row][2] * s;
                Rn[row][1] = R[row][1];
                Rn[row][2] = -R[row][0] * s + R[row][2] * c;
            }
        }

        for (int r = 0; r < 3; r++)
            for (int cc = 0; cc < 3; cc++)
                R[r][cc] = Rn[r][cc];

        // Translate along local z by d[j] and along local x by a[j]
        px += R[0][2] * p.d[j] + R[0][0] * p.a[j];
        py += R[1][2] * p.d[j] + R[1][0] * p.a[j];
        pz += R[2][2] * p.d[j] + R[2][0] * p.a[j];

        pos[j + 1][0] = px;
        pos[j + 1][1] = py;
        pos[j + 1][2] = pz;
    }
}

__host__ __device__ inline void end_effector_pos(
    const float q[NDOF], const ArmParams7& p,
    float& ex, float& ey, float& ez)
{
    float pos[NDOF + 1][3];
    fk_joint_positions(q, p, pos);
    ex = pos[NDOF][0];
    ey = pos[NDOF][1];
    ez = pos[NDOF][2];
}

__host__ __device__ inline float ee_distance(
    const float q[NDOF], const ArmParams7& p, const CostParams7& cp)
{
    float ex, ey, ez;
    end_effector_pos(q, p, ex, ey, ez);
    float dx = ex - cp.goal_x;
    float dy = ey - cp.goal_y;
    float dz = ez - cp.goal_z;
    return sqrtf(dx * dx + dy * dy + dz * dz + 1.0e-6f);
}

template <typename Scalar>
__host__ __device__ Scalar wrap_angle_diff(const Scalar& angle) {
    return cudabot::atan2(cudabot::sin(angle), cudabot::cos(angle));
}

// ======================== Dynamics ========================
__host__ __device__ inline void arm7_step(
    float q[NDOF], float dq[NDOF],
    const float tau[NDOF], const ArmParams7& p)
{
    for (int j = 0; j < NDOF; j++) {
        float t = clampf_local(tau[j], -p.max_torque[j], p.max_torque[j]);
        float ddq = t - p.damping[j] * dq[j] - p.gravity_comp[j] * sinf(q[j]);
        dq[j] = clampf_local(dq[j] + p.dt * ddq, -p.max_vel[j], p.max_vel[j]);
        q[j] = wrap_anglef(q[j] + p.dt * dq[j]);
    }
}

// Dual-number version for autodiff
__device__ inline void arm7_step_diff(
    Dualf q[NDOF], Dualf dq[NDOF],
    Dualf tau[NDOF], const ArmParams7& p)
{
    for (int j = 0; j < NDOF; j++) {
        tau[j] = clamp(tau[j], -p.max_torque[j], p.max_torque[j]);
        Dualf ddq = tau[j]
                   - Dualf::constant(p.damping[j]) * dq[j]
                   - Dualf::constant(p.gravity_comp[j]) * cudabot::sin(q[j]);
        dq[j] = clamp(dq[j] + p.dt * ddq, -p.max_vel[j], p.max_vel[j]);
        q[j] = wrap_angle_diff(q[j] + p.dt * dq[j]);
    }
}

// ======================== Cost Functions ========================
__device__ inline float stage_cost_device(
    const float q[NDOF], const float dq[NDOF], const float tau[NDOF],
    const ArmParams7& params, const CostParams7& cp,
    int n_obs, int n_dyn_obs, float time_world)
{
    // Single FK call for both goal distance and obstacle checking
    float pos[NDOF + 1][3];
    fk_joint_positions(q, params, pos);
    float dx_g = pos[NDOF][0] - cp.goal_x;
    float dy_g = pos[NDOF][1] - cp.goal_y;
    float dz_g = pos[NDOF][2] - cp.goal_z;
    float ee_dist = sqrtf(dx_g*dx_g + dy_g*dy_g + dz_g*dz_g + 1.0e-6f);
    float cost = cp.goal_weight * ee_dist * params.dt;

    float ctrl_cost = 0.0f;
    float vel_cost = 0.0f;
    for (int j = 0; j < NDOF; j++) {
        ctrl_cost += tau[j] * tau[j];
        vel_cost += dq[j] * dq[j];
    }
    cost += cp.control_weight * ctrl_cost * params.dt;
    cost += cp.velocity_weight * vel_cost * params.dt;
    // Sample 10 points along the arm chain
    for (int link = 0; link < NDOF; link++) {
        float mx = 0.5f * (pos[link][0] + pos[link + 1][0]);
        float my = 0.5f * (pos[link][1] + pos[link + 1][1]);
        float mz = 0.5f * (pos[link][2] + pos[link + 1][2]);
        // Check static obstacles
        for (int i = 0; i < n_obs; i++) {
            float dx = mx - d_obs_7dof[i].x;
            float dy = my - d_obs_7dof[i].y;
            float dz = mz - d_obs_7dof[i].z;
            float d = sqrtf(dx*dx + dy*dy + dz*dz + 1.0e-6f) - d_obs_7dof[i].r;
            if (d <= 0.02f) cost += cp.obstacle_weight * 120.0f;
            else if (d < cp.obs_influence) cost += cp.obstacle_weight / (d * d);
        }
        // Check dynamic obstacles
        for (int i = 0; i < n_dyn_obs; i++) {
            float ox = d_dyn_obs_7dof[i].x + d_dyn_obs_7dof[i].vx * time_world;
            float oy = d_dyn_obs_7dof[i].y + d_dyn_obs_7dof[i].vy * time_world;
            float oz = d_dyn_obs_7dof[i].z + d_dyn_obs_7dof[i].vz * time_world;
            float dx = mx - ox;
            float dy = my - oy;
            float dz = mz - oz;
            float d = sqrtf(dx*dx + dy*dy + dz*dz + 1.0e-6f) - d_dyn_obs_7dof[i].r;
            if (d <= 0.02f) cost += cp.obstacle_weight * 120.0f;
            else if (d < cp.obs_influence) cost += cp.obstacle_weight / (d * d);
        }
    }
    // Also check EE point
    float ee_x = pos[NDOF][0], ee_y = pos[NDOF][1], ee_z = pos[NDOF][2];
    for (int i = 0; i < n_obs; i++) {
        float dx = ee_x - d_obs_7dof[i].x;
        float dy = ee_y - d_obs_7dof[i].y;
        float dz = ee_z - d_obs_7dof[i].z;
        float d = sqrtf(dx*dx + dy*dy + dz*dz + 1.0e-6f) - d_obs_7dof[i].r;
        if (d <= 0.02f) cost += cp.obstacle_weight * 120.0f;
        else if (d < cp.obs_influence) cost += cp.obstacle_weight / (d * d);
    }
    for (int i = 0; i < n_dyn_obs; i++) {
        float ox = d_dyn_obs_7dof[i].x + d_dyn_obs_7dof[i].vx * time_world;
        float oy = d_dyn_obs_7dof[i].y + d_dyn_obs_7dof[i].vy * time_world;
        float oz = d_dyn_obs_7dof[i].z + d_dyn_obs_7dof[i].vz * time_world;
        float dx = ee_x - ox;
        float dy = ee_y - oy;
        float dz = ee_z - oz;
        float d = sqrtf(dx*dx + dy*dy + dz*dz + 1.0e-6f) - d_dyn_obs_7dof[i].r;
        if (d <= 0.02f) cost += cp.obstacle_weight * 120.0f;
        else if (d < cp.obs_influence) cost += cp.obstacle_weight / (d * d);
    }

    return cost;
}

__host__ __device__ inline float terminal_cost_fn(
    const float q[NDOF], const float dq[NDOF],
    const ArmParams7& params, const CostParams7& cp)
{
    float dist = ee_distance(q, params, cp);
    float vel_sq = 0.0f;
    for (int j = 0; j < NDOF; j++) vel_sq += dq[j] * dq[j];
    return cp.terminal_weight * dist + cp.terminal_velocity_weight * vel_sq;
}

static float host_stage_cost(
    const float q[NDOF], const float dq[NDOF], const float tau[NDOF],
    const Scenario& s, int step_idx)
{
    float ee_dist = ee_distance(q, s.params, s.cost_params);
    float cost = s.cost_params.goal_weight * ee_dist * s.params.dt;
    float ctrl_c = 0.0f, vel_c = 0.0f;
    for (int j = 0; j < NDOF; j++) { ctrl_c += tau[j]*tau[j]; vel_c += dq[j]*dq[j]; }
    cost += s.cost_params.control_weight * ctrl_c * s.params.dt;
    cost += s.cost_params.velocity_weight * vel_c * s.params.dt;

    float pos[NDOF + 1][3];
    fk_joint_positions(q, s.params, pos);
    float time_world = step_idx * s.params.dt;
    for (int link = 0; link <= NDOF; link++) {
        float px = (link < NDOF) ? 0.5f*(pos[link][0]+pos[link+1][0]) : pos[NDOF][0];
        float py = (link < NDOF) ? 0.5f*(pos[link][1]+pos[link+1][1]) : pos[NDOF][1];
        float pz = (link < NDOF) ? 0.5f*(pos[link][2]+pos[link+1][2]) : pos[NDOF][2];
        for (int i = 0; i < s.n_obs; i++) {
            float dx = px - s.obstacles[i].x;
            float dy = py - s.obstacles[i].y;
            float dz = pz - s.obstacles[i].z;
            float d = sqrtf(dx*dx + dy*dy + dz*dz + 1e-6f) - s.obstacles[i].r;
            if (d <= 0.02f) cost += s.cost_params.obstacle_weight * 120.0f;
            else if (d < s.cost_params.obs_influence) cost += s.cost_params.obstacle_weight / (d*d);
        }
        for (int i = 0; i < s.n_dyn_obs; i++) {
            float ox = s.dynamic_obstacles[i].x + s.dynamic_obstacles[i].vx * time_world;
            float oy = s.dynamic_obstacles[i].y + s.dynamic_obstacles[i].vy * time_world;
            float oz = s.dynamic_obstacles[i].z + s.dynamic_obstacles[i].vz * time_world;
            float dx = px - ox, dy = py - oy, dz = pz - oz;
            float d = sqrtf(dx*dx + dy*dy + dz*dz + 1e-6f) - s.dynamic_obstacles[i].r;
            if (d <= 0.02f) cost += s.cost_params.obstacle_weight * 120.0f;
            else if (d < s.cost_params.obs_influence) cost += s.cost_params.obstacle_weight / (d*d);
        }
    }
    return cost;
}

static float host_min_margin(const float q[NDOF], const Scenario& s, int step_idx) {
    float pos[NDOF + 1][3];
    fk_joint_positions(q, s.params, pos);
    float time_world = step_idx * s.params.dt;
    float best = 1.0e9f;
    for (int link = 0; link <= NDOF; link++) {
        float px = (link < NDOF) ? 0.5f*(pos[link][0]+pos[link+1][0]) : pos[NDOF][0];
        float py = (link < NDOF) ? 0.5f*(pos[link][1]+pos[link+1][1]) : pos[NDOF][1];
        float pz = (link < NDOF) ? 0.5f*(pos[link][2]+pos[link+1][2]) : pos[NDOF][2];
        for (int i = 0; i < s.n_obs; i++) {
            float dx = px - s.obstacles[i].x;
            float dy = py - s.obstacles[i].y;
            float dz = pz - s.obstacles[i].z;
            float d = sqrtf(dx*dx + dy*dy + dz*dz + 1e-6f) - s.obstacles[i].r;
            best = fminf(best, d);
        }
        for (int i = 0; i < s.n_dyn_obs; i++) {
            float ox = s.dynamic_obstacles[i].x + s.dynamic_obstacles[i].vx * time_world;
            float oy = s.dynamic_obstacles[i].y + s.dynamic_obstacles[i].vy * time_world;
            float oz = s.dynamic_obstacles[i].z + s.dynamic_obstacles[i].vz * time_world;
            float dx = px - ox, dy = py - oy, dz = pz - oz;
            float d = sqrtf(dx*dx + dy*dy + dz*dz + 1e-6f) - s.dynamic_obstacles[i].r;
            best = fminf(best, d);
        }
    }
    return best;
}

// Terminal cost gradient: EE distance part via FD on q (7 passes), velocity part analytical.
// terminal_cost = weight * ee_dist(q) + vel_weight * sum(dq_j^2)
__device__ inline void terminal_grad_fn(
    const float q[NDOF], const float dq[NDOF],
    const ArmParams7& params, const CostParams7& cp, float grad[STATE_DIM])
{
    float eps = 1.0e-3f;
    float cost_base = terminal_cost_fn(q, dq, params, cp);

    // d/d(q_j): only EE distance depends on q, need FD through FK (7 passes)
    for (int j = 0; j < NDOF; j++) {
        float q_plus[NDOF], dq_copy[NDOF];
        for (int i = 0; i < NDOF; i++) { q_plus[i] = q[i]; dq_copy[i] = dq[i]; }
        q_plus[j] += eps;
        float cost_plus = terminal_cost_fn(q_plus, dq_copy, params, cp);
        grad[j] = (cost_plus - cost_base) / eps;
    }

    // d/d(dq_j): analytical — vel_weight * 2 * dq_j
    for (int j = 0; j < NDOF; j++) {
        grad[NDOF + j] = cp.terminal_velocity_weight * 2.0f * dq[j];
    }
}

// Stage cost gradient: q part via FD (7 passes), dq and ctrl parts analytical.
// Goal/obstacle costs depend only on q (through FK), not dq or tau.
// Velocity cost = vel_weight * dq_j^2 * dt → analytical dq gradient.
// Control cost = ctrl_weight * tau_j^2 * dt → analytical ctrl gradient.
__device__ inline void stage_cost_grad_fn(
    const float q[NDOF], const float dq[NDOF], const float tau[NDOF],
    const ArmParams7& params, const CostParams7& cp,
    int n_obs, int n_dyn_obs, float time_world,
    float grad_state[STATE_DIM], float grad_ctrl[CTRL_DIM])
{
    float eps = 1.0e-3f;
    float cost_base = stage_cost_device(q, dq, tau, params, cp, n_obs, n_dyn_obs, time_world);

    // q gradient via finite differences (7 passes of stage_cost_device)
    for (int j = 0; j < NDOF; j++) {
        float q_tmp[NDOF];
        for (int i = 0; i < NDOF; i++) q_tmp[i] = q[i];
        q_tmp[j] += eps;
        float cost_plus = stage_cost_device(q_tmp, dq, tau, params, cp, n_obs, n_dyn_obs, time_world);
        grad_state[j] = (cost_plus - cost_base) / eps;
    }

    // dq gradient: analytical (d/d_dq_j of velocity_weight * dq_j^2 * dt)
    for (int j = 0; j < NDOF; j++) {
        grad_state[NDOF + j] = 2.0f * cp.velocity_weight * dq[j] * params.dt;
    }

    // Control gradient: analytical (d/d_tau_j of control_weight * tau_j^2 * dt)
    for (int j = 0; j < CTRL_DIM; j++) {
        grad_ctrl[j] = 2.0f * cp.control_weight * tau[j] * params.dt;
    }
}

// Dynamics Jacobian: d(q_next, dq_next)/d(q, dq, tau) via finite differences
// Analytical dynamics Jacobian — O(NDOF) instead of O(NDOF^2) finite-diff passes.
// Dynamics are decoupled per joint: ddq_j = tau_j - d_j*dq_j - g_j*sin(q_j)
// so the 14x21 Jacobian is block-diagonal with 7 blocks of size 2x3.
__device__ inline void arm7_jacobian(
    const float q[NDOF], const float dq[NDOF], const float tau[NDOF],
    const ArmParams7& params,
    float J[STATE_DIM][STATE_DIM + CTRL_DIM])  // 14 x 21
{
    // Zero the matrix
    for (int r = 0; r < STATE_DIM; r++)
        for (int c = 0; c < STATE_DIM + CTRL_DIM; c++)
            J[r][c] = 0.0f;

    float dt = params.dt;
    for (int j = 0; j < NDOF; j++) {
        float t_clamped = clampf_local(tau[j], -params.max_torque[j], params.max_torque[j]);
        float ddq = t_clamped - params.damping[j] * dq[j] - params.gravity_comp[j] * sinf(q[j]);
        float dq_raw = dq[j] + dt * ddq;
        bool dq_clamped = (dq_raw <= -params.max_vel[j] || dq_raw >= params.max_vel[j]);
        bool tau_clamped = (tau[j] <= -params.max_torque[j] || tau[j] >= params.max_torque[j]);

        // d(ddq)/d(q_j) = -g_j * cos(q_j)
        // d(ddq)/d(dq_j) = -d_j
        // d(ddq)/d(tau_j) = 1  (if tau not clamped, else 0)
        float dddq_dq = -params.gravity_comp[j] * cosf(q[j]);
        float dddq_ddq = -params.damping[j];
        float dddq_dtau = tau_clamped ? 0.0f : 1.0f;

        // d(dq_new)/d(*) = dt * d(ddq)/d(*)  (if dq not clamped, else 0)
        float ddqn_dq = dq_clamped ? 0.0f : dt * dddq_dq;
        float ddqn_ddq = dq_clamped ? 0.0f : 1.0f + dt * dddq_ddq;
        float ddqn_dtau = dq_clamped ? 0.0f : dt * dddq_dtau;

        // d(q_new)/d(*) = d(q + dt*dq_new)/d(*) = delta + dt * d(dq_new)/d(*)
        float dqn_dq = 1.0f + dt * ddqn_dq;
        float dqn_ddq = dt * ddqn_ddq;
        float dqn_dtau = dt * ddqn_dtau;

        // Row j (q_j): columns j, NDOF+j, STATE_DIM+j
        J[j][j] = dqn_dq;
        J[j][NDOF + j] = dqn_ddq;
        J[j][STATE_DIM + j] = dqn_dtau;
        // Row NDOF+j (dq_j): columns j, NDOF+j, STATE_DIM+j
        J[NDOF + j][j] = ddqn_dq;
        J[NDOF + j][NDOF + j] = ddqn_ddq;
        J[NDOF + j][STATE_DIM + j] = ddqn_dtau;
    }
}

// ======================== CUDA Kernels ========================
__global__ void init_curand_kernel(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void rollout_kernel(
    const float* d_start_state,
    const float* d_nominal,
    float* d_costs,
    float* d_perturbed,
    float* d_rollout_states,
    curandState* d_rng,
    ArmParams7 params,
    CostParams7 cp,
    int n_obs, int n_dyn_obs, int start_step,
    int K, int T,
    const float* d_sigma)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float q[NDOF], dq[NDOF];
    for (int j = 0; j < NDOF; j++) {
        q[j] = d_start_state[j];
        dq[j] = d_start_state[NDOF + j];
    }

    if (d_rollout_states != nullptr) {
        for (int j = 0; j < NDOF; j++) {
            d_rollout_states[k * (T+1) * STATE_DIM + j] = q[j];
            d_rollout_states[k * (T+1) * STATE_DIM + NDOF + j] = dq[j];
        }
    }

    float total_cost = 0.0f;
    for (int t = 0; t < T; t++) {
        float tau[NDOF];
        for (int j = 0; j < NDOF; j++) {
            tau[j] = d_nominal[t * CTRL_DIM + j] + curand_normal(&local_rng) * d_sigma[j];
            tau[j] = clampf_local(tau[j], -params.max_torque[j], params.max_torque[j]);
            d_perturbed[k * T * CTRL_DIM + t * CTRL_DIM + j] = tau[j];
        }

        arm7_step(q, dq, tau, params);
        if (d_rollout_states != nullptr) {
            for (int j = 0; j < NDOF; j++) {
                d_rollout_states[k * (T+1) * STATE_DIM + (t+1) * STATE_DIM + j] = q[j];
                d_rollout_states[k * (T+1) * STATE_DIM + (t+1) * STATE_DIM + NDOF + j] = dq[j];
            }
        }
        float tau_world = (start_step + t + 1) * params.dt;
        total_cost += stage_cost_device(q, dq, tau, params, cp, n_obs, n_dyn_obs, tau_world);
    }

    total_cost += terminal_cost_fn(q, dq, params, cp);
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

// Single-thread: sequential min-reduce + normalize (K small enough for single-thread scan).
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

__global__ void update_controls_kernel(
    float* d_nominal, const float* d_perturbed, const float* d_weights,
    int K, int T)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float tau[NDOF] = {};
    for (int k = 0; k < K; k++) {
        float w = d_weights[k];
        for (int j = 0; j < NDOF; j++) {
            tau[j] += w * d_perturbed[k * T * CTRL_DIM + t * CTRL_DIM + j];
        }
    }
    for (int j = 0; j < NDOF; j++) d_nominal[t * CTRL_DIM + j] = tau[j];
}

// Single-thread: sequential forward rollout (each state depends on the previous).
__global__ void rollout_nominal_kernel(
    const float* d_start_state, const float* d_nominal, float* d_states,
    ArmParams7 params, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float q[NDOF], dq[NDOF];
    for (int j = 0; j < NDOF; j++) {
        q[j] = d_start_state[j];
        dq[j] = d_start_state[NDOF + j];
    }
    for (int j = 0; j < STATE_DIM; j++) d_states[j] = d_start_state[j];

    for (int t = 0; t < T; t++) {
        float tau[NDOF];
        for (int j = 0; j < NDOF; j++) tau[j] = d_nominal[t * CTRL_DIM + j];
        arm7_step(q, dq, tau, params);
        for (int j = 0; j < NDOF; j++) {
            d_states[(t+1) * STATE_DIM + j] = q[j];
            d_states[(t+1) * STATE_DIM + NDOF + j] = dq[j];
        }
    }
}

// Phase 1: Parallel cost gradient + Jacobian computation (T threads)
// Each thread handles one timestep — the expensive FK-based FD runs in parallel.
__global__ void precompute_cost_gradients_kernel(
    const float* d_states, const float* d_nominal,
    float* d_stage_grads_s,   // T * STATE_DIM
    float* d_stage_grads_c,   // T * CTRL_DIM
    float* d_jacobians,       // T * STATE_DIM * (STATE_DIM + CTRL_DIM)
    ArmParams7 params, CostParams7 cp,
    int n_obs, int n_dyn_obs, int start_step, int T)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float q[NDOF], dq[NDOF], tau[NDOF];
    for (int j = 0; j < NDOF; j++) {
        q[j] = d_states[t * STATE_DIM + j];
        dq[j] = d_states[t * STATE_DIM + NDOF + j];
        tau[j] = d_nominal[t * CTRL_DIM + j];
    }

    float grad_s[STATE_DIM], grad_c[CTRL_DIM];
    float tau_world = (start_step + t) * params.dt;
    stage_cost_grad_fn(q, dq, tau, params, cp, n_obs, n_dyn_obs, tau_world, grad_s, grad_c);

    for (int j = 0; j < STATE_DIM; j++) d_stage_grads_s[t * STATE_DIM + j] = grad_s[j];
    for (int j = 0; j < CTRL_DIM; j++) d_stage_grads_c[t * CTRL_DIM + j] = grad_c[j];

    // Analytical Jacobian (cheap, no FK)
    float J[STATE_DIM][STATE_DIM + CTRL_DIM];
    arm7_jacobian(q, dq, tau, params, J);
    int stride = STATE_DIM * (STATE_DIM + CTRL_DIM);
    for (int r = 0; r < STATE_DIM; r++)
        for (int c = 0; c < STATE_DIM + CTRL_DIM; c++)
            d_jacobians[t * stride + r * (STATE_DIM + CTRL_DIM) + c] = J[r][c];
}

// Phase 2: Sequential backward adjoint pass (1 thread, matrix ops only, no FK)
__global__ void backward_adjoint_kernel(
    const float* d_states,
    const float* d_stage_grads_s,
    const float* d_stage_grads_c,
    const float* d_jacobians,
    float* d_grad,
    ArmParams7 params, CostParams7 cp, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float adj[STATE_DIM];
    {
        float q[NDOF], dq[NDOF];
        for (int j = 0; j < NDOF; j++) {
            q[j] = d_states[T * STATE_DIM + j];
            dq[j] = d_states[T * STATE_DIM + NDOF + j];
        }
        terminal_grad_fn(q, dq, params, cp, adj);
    }

    int jac_stride = STATE_DIM * (STATE_DIM + CTRL_DIM);
    for (int t = T - 1; t >= 0; t--) {
        const float* J_flat = &d_jacobians[t * jac_stride];
        const float* gs = &d_stage_grads_s[t * STATE_DIM];
        const float* gc = &d_stage_grads_c[t * CTRL_DIM];

        // Control gradient = stage_cost_grad_ctrl + J_ctrl^T * adj
        for (int j = 0; j < CTRL_DIM; j++) {
            float g = gc[j];
            for (int row = 0; row < STATE_DIM; row++) {
                g += J_flat[row * (STATE_DIM + CTRL_DIM) + STATE_DIM + j] * adj[row];
            }
            d_grad[t * CTRL_DIM + j] = g;
        }

        // Adjoint update: adj_new = stage_cost_grad_state + J_state^T * adj
        float new_adj[STATE_DIM];
        for (int col = 0; col < STATE_DIM; col++) {
            float a = gs[col];
            for (int row = 0; row < STATE_DIM; row++) {
                a += J_flat[row * (STATE_DIM + CTRL_DIM) + col] * adj[row];
            }
            new_adj[col] = a;
        }
        for (int j = 0; j < STATE_DIM; j++) adj[j] = new_adj[j];
    }
}

__global__ void gradient_step_kernel(
    float* d_nominal, const float* d_grad, int T, float alpha, float grad_clip, ArmParams7 params)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    // Per-timestep gradient norm clipping
    float norm_sq = 0.0f;
    for (int j = 0; j < CTRL_DIM; j++) {
        float g = d_grad[t * CTRL_DIM + j];
        norm_sq += g * g;
    }
    float scale = alpha;
    if (grad_clip > 0.0f && norm_sq > grad_clip * grad_clip) {
        scale = alpha * grad_clip / sqrtf(norm_sq);
    }
    for (int j = 0; j < CTRL_DIM; j++) {
        float val = d_nominal[t * CTRL_DIM + j] - scale * d_grad[t * CTRL_DIM + j];
        d_nominal[t * CTRL_DIM + j] = clampf_local(val, -params.max_torque[j], params.max_torque[j]);
    }
}

// Reference feedback gain kernel (current-action only, same as 2-link version)
__global__ void compute_rollout_initial_gradients_kernel(
    const float* d_rollout_states,
    const float* d_perturbed,
    float* d_rollout_init_grads,
    ArmParams7 params,
    CostParams7 cp,
    int n_obs, int n_dyn_obs, int start_step,
    int K, int T)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const float* states = &d_rollout_states[k * (T+1) * STATE_DIM];
    const float* controls = &d_perturbed[k * T * CTRL_DIM];

    float adj[STATE_DIM];
    {
        float q[NDOF], dq[NDOF];
        for (int j = 0; j < NDOF; j++) {
            q[j] = states[T * STATE_DIM + j];
            dq[j] = states[T * STATE_DIM + NDOF + j];
        }
        terminal_grad_fn(q, dq, params, cp, adj);
    }

    for (int t = T - 1; t >= 0; t--) {
        float q[NDOF], dq[NDOF], tau[NDOF];
        for (int j = 0; j < NDOF; j++) {
            q[j] = states[t * STATE_DIM + j];
            dq[j] = states[t * STATE_DIM + NDOF + j];
            tau[j] = controls[t * CTRL_DIM + j];
        }
        float J[STATE_DIM][STATE_DIM + CTRL_DIM];
        float grad_s[STATE_DIM], grad_c[CTRL_DIM];
        float tau_world = (start_step + t) * params.dt;
        arm7_jacobian(q, dq, tau, params, J);
        stage_cost_grad_fn(q, dq, tau, params, cp, n_obs, n_dyn_obs, tau_world, grad_s, grad_c);

        float new_adj[STATE_DIM];
        for (int col = 0; col < STATE_DIM; col++) {
            float a = grad_s[col];
            for (int row = 0; row < STATE_DIM; row++) a += J[row][col] * adj[row];
            new_adj[col] = a;
        }
        for (int j = 0; j < STATE_DIM; j++) adj[j] = new_adj[j];
    }
    for (int j = 0; j < STATE_DIM; j++) d_rollout_init_grads[k * STATE_DIM + j] = adj[j];
}

__global__ void compute_reference_feedback_gain_kernel(
    const float* d_nominal,
    const float* d_perturbed,
    const float* d_weights,
    const float* d_rollout_init_grads,
    float* d_feedback_gains,
    float lambda, int K, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float inv_lambda = 1.0f / fmaxf(1.0e-6f, lambda);
    float weighted_grad[STATE_DIM] = {};
    for (int k = 0; k < K; k++) {
        float w = d_weights[k];
        for (int j = 0; j < STATE_DIM; j++)
            weighted_grad[j] += w * d_rollout_init_grads[k * STATE_DIM + j];
    }

    for (int i = 0; i < T * GAINS_PER_STEP; i++) d_feedback_gains[i] = 0.0f;

    float nominal_ctrl[CTRL_DIM];
    for (int j = 0; j < CTRL_DIM; j++) nominal_ctrl[j] = d_nominal[j];

    for (int sj = 0; sj < STATE_DIM; sj++) {
        for (int cj = 0; cj < CTRL_DIM; cj++) {
            float gain = 0.0f;
            for (int k = 0; k < K; k++) {
                float w = d_weights[k];
                float cg = d_rollout_init_grads[k * STATE_DIM + sj] - weighted_grad[sj];
                float du = d_perturbed[k * T * CTRL_DIM + cj] - nominal_ctrl[cj];
                gain += -inv_lambda * w * cg * du;
            }
            d_feedback_gains[cj * STATE_DIM + sj] = gain;
        }
    }
}

// ======================== Episode Runner ========================
class EpisodeRunner {
public:
    EpisodeRunner(const PlannerVariant& variant, const Scenario& scenario, int k_samples, int t_horizon, int seed)
        : variant_(variant), scenario_(scenario), k_samples_(k_samples), t_horizon_(t_horizon), seed_(seed) {
        reset_state();
        h_nominal_.assign(t_horizon_ * CTRL_DIM, 0.0f);
        h_costs_.assign(k_samples_, 0.0f);
        h_states_.assign((t_horizon_ + 1) * STATE_DIM, 0.0f);
        h_start_state_.assign(STATE_DIM, 0.0f);
        h_sigma_.assign(CTRL_DIM, 0.0f);
        for (int j = 0; j < CTRL_DIM; j++) h_sigma_[j] = variant_.torque_sigma[j];

        CUDA_CHECK(cudaMalloc(&d_nominal_, h_nominal_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs_, h_costs_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_, k_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perturbed_, k_samples_ * t_horizon_ * CTRL_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rollout_states_, k_samples_ * (t_horizon_ + 1) * STATE_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rollout_init_grads_, k_samples_ * STATE_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_states_, h_states_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_, h_nominal_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_stage_grads_s_, t_horizon_ * STATE_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_stage_grads_c_, t_horizon_ * CTRL_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_jacobians_, t_horizon_ * STATE_DIM * (STATE_DIM + CTRL_DIM) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_feedback_gains_, t_horizon_ * GAINS_PER_STEP * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_start_state_, STATE_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sigma_, CTRL_DIM * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_sigma_, h_sigma_.data(), CTRL_DIM * sizeof(float), cudaMemcpyHostToDevice));
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
        CUDA_CHECK(cudaFree(d_stage_grads_s_));
        CUDA_CHECK(cudaFree(d_stage_grads_c_));
        CUDA_CHECK(cudaFree(d_jacobians_));
        CUDA_CHECK(cudaFree(d_feedback_gains_));
        CUDA_CHECK(cudaFree(d_start_state_));
        CUDA_CHECK(cudaFree(d_sigma_));
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
            float dist = ee_distance(q_, scenario_.params, scenario_.cost_params);
            min_goal_distance_ = fminf(min_goal_distance_, dist);
            if (dist < scenario_.goal_tol) {
                reached_goal_ = true;
                steps_taken_ = step;
                break;
            }

            auto t0 = chrono::steady_clock::now();
            controller_update(step);
            auto t1 = chrono::steady_clock::now();
            total_control_ms += chrono::duration<float, milli>(t1 - t0).count();

            CUDA_CHECK(cudaMemcpy(h_nominal_.data(), d_nominal_, h_nominal_.size() * sizeof(float), cudaMemcpyDeviceToHost));
            float tau[NDOF];
            for (int j = 0; j < NDOF; j++) tau[j] = h_nominal_[j];
            arm7_step(q_, dq_, tau, scenario_.params);
            cumulative_cost_ += host_stage_cost(q_, dq_, tau, scenario_, step + 1);

            if (host_min_margin(q_, scenario_, step + 1) <= 0.02f) collisions_++;

            // Shift nominal
            for (int t = 0; t < t_horizon_ - 1; t++) {
                for (int j = 0; j < CTRL_DIM; j++)
                    h_nominal_[t * CTRL_DIM + j] = h_nominal_[(t+1) * CTRL_DIM + j];
            }
            for (int j = 0; j < CTRL_DIM; j++)
                h_nominal_[(t_horizon_ - 1) * CTRL_DIM + j] = 0.0f;

            steps_taken_ = step + 1;
        }

        auto episode_end = chrono::steady_clock::now();
        float final_distance = ee_distance(q_, scenario_.params, scenario_.cost_params);
        if (final_distance < scenario_.goal_tol) reached_goal_ = true;

        EpisodeMetrics m;
        m.scenario = scenario_.name;
        m.planner = variant_.name;
        m.seed = seed_;
        m.k_samples = k_samples_;
        m.t_horizon = t_horizon_;
        m.grad_steps = variant_.grad_steps;
        m.alpha = variant_.alpha;
        m.reached_goal = reached_goal_ ? 1 : 0;
        m.collision_free = (collisions_ == 0) ? 1 : 0;
        m.success = (m.reached_goal && m.collision_free) ? 1 : 0;
        m.steps = steps_taken_;
        m.final_distance = final_distance;
        m.min_goal_distance = min_goal_distance_;
        m.cumulative_cost = cumulative_cost_;
        m.collisions = collisions_;
        m.total_control_ms = total_control_ms;
        m.avg_control_ms = steps_taken_ > 0 ? total_control_ms / steps_taken_ : 0.0f;
        m.episode_ms = chrono::duration<float, milli>(episode_end - episode_begin).count();
        m.sample_budget = static_cast<long long>(steps_taken_) * k_samples_ * t_horizon_;
        return m;
    }

private:
    void reset_rng() {
        int block = 256;
        init_curand_kernel<<<(k_samples_ + block - 1) / block, block>>>(d_rng_, k_samples_, static_cast<unsigned long long>(seed_));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void sync_start_state() {
        for (int j = 0; j < NDOF; j++) {
            h_start_state_[j] = q_[j];
            h_start_state_[NDOF + j] = dq_[j];
        }
        CUDA_CHECK(cudaMemcpy(d_start_state_, h_start_state_.data(), STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    }

    void controller_update(int start_step) {
        sync_start_state();
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size() * sizeof(float), cudaMemcpyHostToDevice));
        int block = 256;

        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
            d_start_state_, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
            scenario_.params, scenario_.cost_params, scenario_.n_obs, scenario_.n_dyn_obs,
            start_step, k_samples_, t_horizon_, d_sigma_);
        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
        update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
            d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);

        if (variant_.use_feedback && variant_.feedback_mode == 1) {
            // Reference feedback: second rollout + sensitivity gains
            float fb_sigma[NDOF];
            for (int j = 0; j < NDOF; j++) fb_sigma[j] = variant_.feedback_noise_sigma[j];
            CUDA_CHECK(cudaMemcpy(d_sigma_, fb_sigma, CTRL_DIM * sizeof(float), cudaMemcpyHostToDevice));

            rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                d_start_state_, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                scenario_.params, scenario_.cost_params, scenario_.n_obs, scenario_.n_dyn_obs,
                start_step, k_samples_, t_horizon_, d_sigma_);
            compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);

            compute_rollout_initial_gradients_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                d_rollout_states_, d_perturbed_, d_rollout_init_grads_,
                scenario_.params, scenario_.cost_params, scenario_.n_obs, scenario_.n_dyn_obs,
                start_step, k_samples_, t_horizon_);
            update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);
            compute_reference_feedback_gain_kernel<<<1, 1>>>(
                d_nominal_, d_perturbed_, d_weights_, d_rollout_init_grads_, d_feedback_gains_,
                variant_.sampling_lambda, k_samples_, t_horizon_);

            // Restore sampling sigma
            CUDA_CHECK(cudaMemcpy(d_sigma_, h_sigma_.data(), CTRL_DIM * sizeof(float), cudaMemcpyHostToDevice));
        }

        if (variant_.use_gradient) {
            for (int gs = 0; gs < variant_.grad_steps; gs++) {
                rollout_nominal_kernel<<<1, 1>>>(d_start_state_, d_nominal_, d_states_, scenario_.params, t_horizon_);
                // Phase 1: parallel cost gradient + Jacobian (T threads)
                precompute_cost_gradients_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                    d_states_, d_nominal_, d_stage_grads_s_, d_stage_grads_c_, d_jacobians_,
                    scenario_.params, scenario_.cost_params,
                    scenario_.n_obs, scenario_.n_dyn_obs, start_step, t_horizon_);
                // Phase 2: sequential backward adjoint (1 thread, no FK)
                backward_adjoint_kernel<<<1, 1>>>(
                    d_states_, d_stage_grads_s_, d_stage_grads_c_, d_jacobians_, d_grad_,
                    scenario_.params, scenario_.cost_params, t_horizon_);
                gradient_step_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                    d_nominal_, d_grad_, t_horizon_,
                    variant_.alpha * scenario_.grad_alpha_scale, variant_.grad_clip, scenario_.params);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void warmup_controller() {
        for (int iter = 0; iter < BENCH_WARMUP_ITERS; iter++) controller_update(0);
    }

    void reset_state() {
        for (int j = 0; j < NDOF; j++) {
            q_[j] = scenario_.start_q[j];
            dq_[j] = scenario_.start_dq[j];
        }
        steps_taken_ = 0;
        collisions_ = 0;
        reached_goal_ = false;
        cumulative_cost_ = 0.0f;
        min_goal_distance_ = ee_distance(q_, scenario_.params, scenario_.cost_params);
    }

    PlannerVariant variant_;
    Scenario scenario_;
    int k_samples_, t_horizon_, seed_;

    float q_[NDOF] = {};
    float dq_[NDOF] = {};
    int steps_taken_ = 0;
    int collisions_ = 0;
    bool reached_goal_ = false;
    float cumulative_cost_ = 0.0f;
    float min_goal_distance_ = 0.0f;

    vector<float> h_nominal_, h_costs_, h_states_, h_start_state_, h_sigma_;
    float* d_nominal_ = nullptr;
    float* d_costs_ = nullptr;
    float* d_weights_ = nullptr;
    float* d_perturbed_ = nullptr;
    float* d_rollout_states_ = nullptr;
    float* d_rollout_init_grads_ = nullptr;
    float* d_states_ = nullptr;
    float* d_grad_ = nullptr;
    float* d_stage_grads_s_ = nullptr;
    float* d_stage_grads_c_ = nullptr;
    float* d_jacobians_ = nullptr;
    float* d_feedback_gains_ = nullptr;
    float* d_start_state_ = nullptr;
    float* d_sigma_ = nullptr;
    curandState* d_rng_ = nullptr;
};

// ======================== Scenarios ========================
static Scenario make_7dof_shelf_reach() {
    Scenario s;
    s.name = "7dof_shelf_reach";
    // Start: Panda home configuration
    s.start_q[0] = 0.0f;   s.start_q[1] = -0.78f; s.start_q[2] = 0.0f;
    s.start_q[3] = -2.36f;  s.start_q[4] = 0.0f;   s.start_q[5] = 1.57f;
    s.start_q[6] = 0.78f;
    s.goal_tol = 0.15f;
    s.max_steps = 300;
    s.grad_alpha_scale = 0.5f;
    // Goal: reach forward, avoiding obstacle in the workspace
    s.cost_params.goal_x = 0.40f;
    s.cost_params.goal_y = 0.15f;
    s.cost_params.goal_z = 0.35f;
    s.cost_params.goal_weight = 20.0f;
    s.cost_params.control_weight = 0.0001f;
    s.cost_params.velocity_weight = 0.005f;
    s.cost_params.obstacle_weight = 10.0f;
    s.cost_params.obs_influence = 0.08f;
    s.cost_params.terminal_weight = 50.0f;
    // Obstacle blocking the direct path
    const Obstacle3D obs[] = {
        {0.32f,  0.08f, 0.28f, 0.06f},
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    return s;
}

static Scenario make_7dof_dynamic_avoid() {
    Scenario s;
    s.name = "7dof_dynamic_avoid";
    s.start_q[0] = 0.0f;   s.start_q[1] = -0.78f; s.start_q[2] = 0.0f;
    s.start_q[3] = -2.36f;  s.start_q[4] = 0.0f;   s.start_q[5] = 1.57f;
    s.start_q[6] = 0.78f;
    s.goal_tol = 0.10f;
    s.max_steps = 300;
    s.grad_alpha_scale = 0.5f;
    s.cost_params.goal_x = 0.55f;
    s.cost_params.goal_y = -0.20f;
    s.cost_params.goal_z = 0.30f;
    s.cost_params.goal_weight = 15.0f;
    s.cost_params.control_weight = 0.0001f;
    s.cost_params.velocity_weight = 0.005f;
    s.cost_params.obstacle_weight = 15.0f;
    s.cost_params.obs_influence = 0.10f;
    s.cost_params.terminal_weight = 40.0f;
    // Static obstacles
    const Obstacle3D obs[] = {
        {0.45f, -0.10f, 0.20f, 0.06f},
        {0.50f,  0.10f, 0.35f, 0.05f},
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    // Moving obstacle crossing the workspace
    const DynamicObstacle3D dyn[] = {
        {0.70f, 0.0f, 0.30f, -0.15f, -0.05f, 0.0f, 0.06f},
    };
    s.n_dyn_obs = static_cast<int>(sizeof(dyn) / sizeof(dyn[0]));
    for (int i = 0; i < s.n_dyn_obs; i++) s.dynamic_obstacles[i] = dyn[i];
    return s;
}

// ======================== Utilities ========================
static void ensure_build_dir() { mkdir("build", 0755); }

static vector<int> parse_int_list(const string& text) {
    vector<int> v; string tok; stringstream ss(text);
    while (getline(ss, tok, ',')) { if (!tok.empty()) v.push_back(max(1, atoi(tok.c_str()))); }
    sort(v.begin(), v.end()); v.erase(unique(v.begin(), v.end()), v.end());
    return v;
}

static vector<string> parse_string_list(const string& text) {
    vector<string> v; string tok; stringstream ss(text);
    while (getline(ss, tok, ',')) { if (!tok.empty()) v.push_back(tok); }
    sort(v.begin(), v.end()); v.erase(unique(v.begin(), v.end()), v.end());
    return v;
}

static void write_csv(const vector<EpisodeMetrics>& rows, const string& path) {
    ofstream out(path);
    out << "scenario,planner,seed,k_samples,t_horizon,grad_steps,alpha,reached_goal,collision_free,success,steps,final_distance,min_goal_distance,cumulative_cost,collisions,avg_control_ms,total_control_ms,episode_ms,sample_budget\n";
    for (const auto& r : rows) {
        out << r.scenario << ',' << r.planner << ',' << r.seed << ','
            << r.k_samples << ',' << r.t_horizon << ',' << r.grad_steps << ','
            << r.alpha << ',' << r.reached_goal << ',' << r.collision_free << ','
            << r.success << ',' << r.steps << ',' << r.final_distance << ','
            << r.min_goal_distance << ',' << r.cumulative_cost << ',' << r.collisions << ','
            << r.avg_control_ms << ',' << r.total_control_ms << ',' << r.episode_ms << ','
            << r.sample_budget << '\n';
    }
}

static void print_summary(const vector<EpisodeMetrics>& rows) {
    map<string, SummaryStats> stats;
    for (const auto& r : rows) {
        string key = r.scenario + " | " + r.planner + " | K=" + to_string(r.k_samples);
        auto& s = stats[key]; s.episodes++;
        s.successes += r.success; s.steps_sum += r.steps;
        s.final_sum += r.final_distance; s.min_sum += r.min_goal_distance;
        s.cost_sum += r.cumulative_cost; s.ms_sum += r.avg_control_ms;
    }
    cout << "=== benchmark_diff_mppi_manipulator_7dof summary ===" << endl;
    for (const auto& kv : stats) {
        const SummaryStats& s = kv.second;
        float n = static_cast<float>(s.episodes);
        printf("%s : success=%.2f steps=%.1f final_dist=%.3f min_dist=%.3f cost=%.1f avg_ms=%.2f\n",
               kv.first.c_str(), s.successes / n, s.steps_sum / n,
               s.final_sum / n, s.min_sum / n, s.cost_sum / n, s.ms_sum / n);
    }
}

// ======================== Main ========================
int main(int argc, char** argv) {
    bool quick = false;
    string csv_path = "build/benchmark_diff_mppi_manipulator_7dof.csv";
    vector<int> k_values;
    vector<string> scenario_names, planner_names;
    int seed_count = -1;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--quick") quick = true;
        else if (arg == "--csv" && i+1 < argc) csv_path = argv[++i];
        else if (arg == "--k-values" && i+1 < argc) k_values = parse_int_list(argv[++i]);
        else if (arg == "--seed-count" && i+1 < argc) seed_count = max(1, atoi(argv[++i]));
        else if (arg == "--scenarios" && i+1 < argc) scenario_names = parse_string_list(argv[++i]);
        else if (arg == "--planners" && i+1 < argc) planner_names = parse_string_list(argv[++i]);
    }

    ensure_build_dir();

    vector<Scenario> all_scenarios = {
        make_7dof_shelf_reach(),
        make_7dof_dynamic_avoid(),
    };
    vector<Scenario> scenarios;
    if (!scenario_names.empty()) {
        for (const auto& wanted : scenario_names) {
            auto it = find_if(all_scenarios.begin(), all_scenarios.end(),
                              [&](const Scenario& s) { return s.name == wanted; });
            if (it == all_scenarios.end()) { fprintf(stderr, "Unknown scenario: %s\n", wanted.c_str()); return 1; }
            scenarios.push_back(*it);
        }
    } else { scenarios = all_scenarios; }

    vector<PlannerVariant> variants;
    {
        PlannerVariant v;
        v.name = "mppi";
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_ref";
        v.use_feedback = true;
        v.feedback_mode = 1;
        v.feedback_gain_scale = 1.0f;
        for (int j = 0; j < NDOF; j++) v.feedback_noise_sigma[j] = v.torque_sigma[j] * 0.7f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_1";
        v.use_gradient = true;
        v.grad_steps = 1;
        v.alpha = 0.02f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_3";
        v.use_gradient = true;
        v.grad_steps = 3;
        v.alpha = 0.008f;
        variants.push_back(v);
    }

    if (!planner_names.empty()) {
        vector<PlannerVariant> filtered;
        for (const auto& wanted : planner_names) {
            auto it = find_if(variants.begin(), variants.end(),
                              [&](const PlannerVariant& v) { return v.name == wanted; });
            if (it == variants.end()) { fprintf(stderr, "Unknown planner: %s\n", wanted.c_str()); return 1; }
            filtered.push_back(*it);
        }
        variants.swap(filtered);
    }

    if (k_values.empty()) k_values = quick ? vector<int>{256, 512} : vector<int>{256, 512, 1024};
    if (seed_count <= 0) seed_count = quick ? 2 : 4;

    vector<EpisodeMetrics> rows;
    rows.reserve(scenarios.size() * variants.size() * k_values.size() * seed_count);

    for (size_t si = 0; si < scenarios.size(); si++) {
        const Scenario& scenario = scenarios[si];
        CUDA_CHECK(cudaMemcpyToSymbol(d_obs_7dof, scenario.obstacles, sizeof(Obstacle3D) * scenario.n_obs));
        if (scenario.n_dyn_obs > 0) {
            CUDA_CHECK(cudaMemcpyToSymbol(d_dyn_obs_7dof, scenario.dynamic_obstacles,
                                          sizeof(DynamicObstacle3D) * scenario.n_dyn_obs));
        }
        for (int ks : k_values) {
            for (size_t vi = 0; vi < variants.size(); vi++) {
                const PlannerVariant& variant = variants[vi];
                for (int seed = 0; seed < seed_count; seed++) {
                    int run_seed = static_cast<int>(3000 + si * 100 + vi * 20 + seed * 7 + ks);
                    EpisodeRunner runner(variant, scenario, ks, DEFAULT_T_HORIZON, run_seed);
                    EpisodeMetrics m = runner.run();
                    rows.push_back(m);
                    printf("[%s] %s K=%d seed=%d success=%d steps=%d final_dist=%.3f avg_ms=%.2f collisions=%d\n",
                           scenario.name.c_str(), variant.name.c_str(), ks, seed,
                           m.success, m.steps, m.final_distance, m.avg_control_ms, m.collisions);
                }
            }
        }
    }

    write_csv(rows, csv_path);
    print_summary(rows);
    cout << "CSV saved to " << csv_path << endl;
    return 0;
}
