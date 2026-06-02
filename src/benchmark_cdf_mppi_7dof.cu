/*************************************************************************
    One-Step CDF-MPPI 7-DOF Manipulator Benchmark  (arXiv:2509.00836)
    - Literature-faithful direct baseline for the Diff-MPPI research line.
    - A Configuration-Space Distance Field drives an angle-based one-step (H=1)
      MPPI on joint velocities. Two CDF sources:
        * cdf_mppi        : analytic margin-derived CDF (PRIMARY; works well)
        * cdf_mppi_neural : neural CDF ablation (a value-MSE MLP, include/
                            neural_cdf.cuh) — documents that learning the 7-D
                            contact gradient needs eikonal/Sobolev supervision,
                            which the repo's GpuMLP lacks. See
                            paper/cdf_mppi_baseline_results.md.
    - SAME task scaffold as benchmark_diff_mppi_manipulator_7dof.cu (FK,
      obstacles, host_min_margin collision oracle, EE-goal success test, CSV
      schema) so the two binaries' outputs concatenate for an apples-to-apples,
      matched-wall-clock-per-control-step comparison.

    Fairness: both controllers solve the identical task (identical FK +
    obstacles + success metric); only the native action space + integrator
    differ (CDF-MPPI = velocity-kinematic H=1; Diff-MPPI = torque 2nd-order
    T=30). Per-step wall-clock (avg_control_ms) is the shared budget meter;
    sample-count (N*1 vs K*T) and replan dt are disclosed as secondary columns.

    NOTE: the pure task functions below (Obstacle3D, ArmParams7, Scenario, FK,
    host_min_margin, scenario makers, CSV/summary) are COPIED verbatim from
    benchmark_diff_mppi_manipulator_7dof.cu by repo convention. Keep the two in
    sync — the matched comparison depends on identical scenario definitions.
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
#include <opencv2/opencv.hpp>

#include "neural_cdf.cuh"

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;
using namespace cudabot;

static const int NDOF = 7;
static const int MAX_OBSTACLES = 16;
static const int MAX_DYNAMIC_OBSTACLES = 8;

// CDF-MPPI control parameters (paper defaults).
static const float CDF_DT = 0.01f;          // s, one-step replan period
static const float PI_F = 3.14159265358979323846f;

// ======================== Copied task scaffold ========================
struct Obstacle3D { float x, y, z, r; };
struct DynamicObstacle3D { float x, y, z; float vx, vy, vz; float r; };

struct ArmParams7 {
    float dt = 0.04f;
    float max_vel[NDOF]    = {2.18f, 2.18f, 2.18f, 2.18f, 2.61f, 2.61f, 2.61f};
    float max_torque[NDOF] = {87.0f, 87.0f, 87.0f, 87.0f, 12.0f, 12.0f, 12.0f};
    float damping[NDOF]    = {2.5f, 2.5f, 2.0f, 2.0f, 0.8f, 0.8f, 0.6f};
    float gravity_comp[NDOF] = {0.0f, 25.0f, 5.0f, 18.0f, 1.5f, 1.0f, 0.2f};
    float d[NDOF] = {0.333f, 0.0f, 0.316f, 0.0f, 0.384f, 0.0f, 0.107f};
    float a[NDOF] = {0.0f, 0.0f, 0.0825f, -0.0825f, 0.0f, 0.088f, 0.0f};
};

struct CostParams7 {
    float goal_x = 0.5f, goal_y = 0.0f, goal_z = 0.4f;
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

struct EpisodeMetrics {
    string scenario, planner;
    int seed = 0, k_samples = 0, t_horizon = 0, grad_steps = 0;
    float alpha = 0.0f;
    int reached_goal = 0, collision_free = 0, success = 0, steps = 0;
    float final_distance = 0.0f, min_goal_distance = 0.0f, cumulative_cost = 0.0f;
    int collisions = 0;
    float avg_control_ms = 0.0f, total_control_ms = 0.0f, episode_ms = 0.0f;
    long long sample_budget = 0;
};

struct SummaryStats {
    int episodes = 0, successes = 0;
    double steps_sum = 0, final_sum = 0, min_sum = 0, cost_sum = 0, ms_sum = 0;
};

__host__ __device__ inline float clampf_local(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

// Forward kinematics — verbatim copy (joint frame origins of simplified Panda).
__host__ __device__ inline void fk_joint_positions(
    const float q[NDOF], const ArmParams7& p, float pos[NDOF + 1][3])
{
    pos[0][0] = 0.0f; pos[0][1] = 0.0f; pos[0][2] = 0.0f;
    float R[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    for (int j = 0; j < NDOF; j++) {
        float c = cosf(q[j]); float s = sinf(q[j]); float Rn[3][3];
        if (j == 0 || j == 2 || j == 4 || j == 6) {
            for (int row = 0; row < 3; row++) {
                Rn[row][0] = R[row][0] * c - R[row][1] * s;
                Rn[row][1] = R[row][0] * s + R[row][1] * c;
                Rn[row][2] = R[row][2];
            }
        } else {
            for (int row = 0; row < 3; row++) {
                Rn[row][0] = R[row][0] * c + R[row][2] * s;
                Rn[row][1] = R[row][1];
                Rn[row][2] = -R[row][0] * s + R[row][2] * c;
            }
        }
        for (int r = 0; r < 3; r++)
            for (int cc = 0; cc < 3; cc++) R[r][cc] = Rn[r][cc];
        px += R[0][2] * p.d[j] + R[0][0] * p.a[j];
        py += R[1][2] * p.d[j] + R[1][0] * p.a[j];
        pz += R[2][2] * p.d[j] + R[2][0] * p.a[j];
        pos[j + 1][0] = px; pos[j + 1][1] = py; pos[j + 1][2] = pz;
    }
}

__host__ __device__ inline void end_effector_pos(
    const float q[NDOF], const ArmParams7& p, float& ex, float& ey, float& ez)
{
    float pos[NDOF + 1][3];
    fk_joint_positions(q, p, pos);
    ex = pos[NDOF][0]; ey = pos[NDOF][1]; ez = pos[NDOF][2];
}

__host__ __device__ inline float ee_distance(
    const float q[NDOF], const ArmParams7& p, const CostParams7& cp)
{
    float ex, ey, ez; end_effector_pos(q, p, ex, ey, ez);
    float dx = ex - cp.goal_x, dy = ey - cp.goal_y, dz = ez - cp.goal_z;
    return sqrtf(dx * dx + dy * dy + dz * dz + 1.0e-6f);
}

// Workspace min signed margin over link midpoints + EE (collision oracle).
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
            float dx = px - s.obstacles[i].x, dy = py - s.obstacles[i].y, dz = pz - s.obstacles[i].z;
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

// Trajectory cost (goal + obstacle), velocity/torque-free — for cumulative_cost
// CSV parity. Uses scenario obstacles directly (host).
static float host_traj_cost(const float q[NDOF], const Scenario& s, int step_idx) {
    float ee_dist = ee_distance(q, s.params, s.cost_params);
    float cost = s.cost_params.goal_weight * ee_dist * s.params.dt;
    float pos[NDOF + 1][3];
    fk_joint_positions(q, s.params, pos);
    float time_world = step_idx * s.params.dt;
    for (int link = 0; link <= NDOF; link++) {
        float px = (link < NDOF) ? 0.5f*(pos[link][0]+pos[link+1][0]) : pos[NDOF][0];
        float py = (link < NDOF) ? 0.5f*(pos[link][1]+pos[link+1][1]) : pos[NDOF][1];
        float pz = (link < NDOF) ? 0.5f*(pos[link][2]+pos[link+1][2]) : pos[NDOF][2];
        for (int i = 0; i < s.n_obs; i++) {
            float dx = px - s.obstacles[i].x, dy = py - s.obstacles[i].y, dz = pz - s.obstacles[i].z;
            float d = sqrtf(dx*dx + dy*dy + dz*dz + 1e-6f) - s.obstacles[i].r;
            if (d <= 0.02f) cost += s.cost_params.obstacle_weight * 120.0f;
            else if (d < s.cost_params.obs_influence) cost += s.cost_params.obstacle_weight / (d*d);
        }
        (void)time_world;
    }
    return cost;
}

// The CDF is built from the workspace signed margin m(q):
//   f_c(q) = clamp( max(m,0) / ||grad_q m||, 0, 1.5 )   (Newton-step distance est.)
// with the CDF ascent direction = grad_q m (points AWAY from the obstacle).
// We regress the SMOOTH field m(q) (neural-SDF-style, easy to fit) and apply
// this analytic transform at runtime — both for the neural variant (m from MLP)
// and the oracle variant (m from host_min_margin directly). This sidesteps the
// hard problem of learning the composite CDF gradient by value-MSE alone.

// Oracle margin value + gradient (8 FK evals, host). step_idx selects the
// obstacle positions at the CURRENT control step — so on dynamic obstacles the
// analytic CDF-MPPI is REACTIVE to the obstacle's current location (but, being
// H=1, cannot anticipate its motion: that is the differentiation we test).
static void margin_value_grad(const float q[NDOF], const Scenario& s, int step_idx,
                              float& m, float grad[NDOF]) {
    const float eps = 1.0e-3f;
    m = host_min_margin(q, s, step_idx);
    for (int j = 0; j < NDOF; j++) {
        float qp[NDOF]; for (int k = 0; k < NDOF; k++) qp[k] = q[k];
        qp[j] += eps;
        grad[j] = (host_min_margin(qp, s, step_idx) - m) / eps;
    }
}

// Shared transform: (margin, grad_margin) -> (CDF value fc, unit ascent ghat).
static inline void cdf_transform(float m, const float gradm[NDOF], float& fc, float ghat[NDOF]) {
    float gn2 = 0.0f;
    for (int j = 0; j < NDOF; j++) gn2 += gradm[j] * gradm[j];
    float gn = sqrtf(gn2) + 1.0e-6f;
    fc = fminf(fmaxf(m, 0.0f) / gn, 1.5f);
    for (int j = 0; j < NDOF; j++) ghat[j] = gradm[j] / gn;
}

// ======================== CDF-MPPI specifics ========================

// Velocity-integrated kinematic plant: q <- clamp(q + dt*clamp(u, +/-vb), [lo,hi]).
__host__ __device__ inline void arm_vel_step(
    float q[NDOF], const float u[NDOF], const float vb[NDOF], float dt)
{
    const float lo[NDOF] = {-2.8973f,-1.7628f,-2.8973f,-3.0718f,-2.8973f,-0.0175f,-2.8973f};
    const float hi[NDOF] = { 2.8973f, 1.7628f, 2.8973f,-0.0698f, 2.8973f, 3.7525f, 2.8973f};
    for (int j = 0; j < NDOF; j++) {
        float uj = clampf_local(u[j], -vb[j], vb[j]);
        float qn = q[j] + dt * uj;
        q[j] = clampf_local(qn, lo[j], hi[j]);
    }
}

struct CdfVariant {
    string name;
    int n_samples = 200;
    float sigma = 1.0f;       // velocity sampling std (rad/s), isotropic
    float alpha_mu = 0.5f;    // mean update rate
    float beta = 1.0f;        // softmax temperature
    float alpha1 = 20.0f;     // obstacle (CDF) angle weight
    float alpha2 = 10.0f;     // goal angle weight
    float d_act = 1.0f;       // CDF activation threshold
    // CDF source: default = analytic margin field (primary baseline; works).
    // use_neural = true selects the neural-CDF ablation (a value-MSE MLP cannot
    // learn the 7-D contact-gradient without eikonal/Sobolev supervision — see
    // include/neural_cdf.cuh and paper/cdf_mppi_baseline_results.md).
    bool use_neural = false;
};

__global__ void init_curand_kernel(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// One-step CDF-MPPI sampling + angle-based cost. 1 thread = 1 velocity sample.
__global__ void cdf_mppi_kernel(
    const float* d_mu, float sigma,
    const float* d_ghat, const float* d_goalhat,
    const float* d_vb,
    int gate_active, float alpha1, float alpha2,
    curandState* d_rng, float* d_costs, float* d_sampled, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curandState lr = d_rng[i];
    float u[NDOF];
    float nu = 0.0f, dot1 = 0.0f, dot2 = 0.0f;
    for (int j = 0; j < NDOF; j++) {
        float uj = d_mu[j] + curand_normal(&lr) * sigma;
        uj = clampf_local(uj, -d_vb[j], d_vb[j]);
        u[j] = uj; nu += uj * uj;
    }
    nu = sqrtf(nu) + 1.0e-9f;
    for (int j = 0; j < NDOF; j++) {
        dot1 += u[j] * d_ghat[j];
        dot2 += u[j] * d_goalhat[j];
    }
    dot1 /= nu; dot2 /= nu;
    float theta1 = acosf(clampf_local(dot1, -1.0f, 1.0f));
    float theta2 = acosf(clampf_local(dot2, -1.0f, 1.0f));
    // Deactivate theta1 (no obstacle penalty) when motion already moves away
    // (theta1 < pi/2) or globally inactive (far from obstacle / goal closer).
    float t1 = (gate_active && theta1 >= 0.5f * PI_F) ? theta1 : 0.0f;
    d_costs[i] = alpha1 * t1 + alpha2 * theta2;
    for (int j = 0; j < NDOF; j++) d_sampled[i * NDOF + j] = u[j];
    d_rng[i] = lr;
}

// Single-thread min-reduce + softmax normalize (N small).
__global__ void cdf_weights_kernel(const float* d_costs, float* d_weights, int N, float beta) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float cmin = FLT_MAX;
    for (int k = 0; k < N; k++) cmin = fminf(cmin, d_costs[k]);
    float sum = 0.0f;
    for (int k = 0; k < N; k++) { float w = expf(-(d_costs[k] - cmin) / beta); d_weights[k] = w; sum += w; }
    if (sum > 0.0f) for (int k = 0; k < N; k++) d_weights[k] /= sum;
}

// Importance-weighted mean update, 1 thread = 1 joint.
__global__ void cdf_update_mean_kernel(
    float* d_mu, const float* d_sampled, const float* d_weights,
    const float* d_vb, float alpha_mu, int N)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= NDOF) return;
    float acc = 0.0f;
    for (int k = 0; k < N; k++) acc += d_weights[k] * d_sampled[k * NDOF + j];
    float mu = (1.0f - alpha_mu) * d_mu[j] + alpha_mu * acc;
    d_mu[j] = clampf_local(mu, -d_vb[j], d_vb[j]);
}

// ======================== Episode Runner ========================
class CdfEpisodeRunner {
public:
    CdfEpisodeRunner(const CdfVariant& variant, const Scenario& scenario,
                     const float q_goal[NDOF], int seed, NeuralCdf* cdf)
        : variant_(variant), scenario_(scenario), seed_(seed), cdf_(cdf)
    {
        for (int j = 0; j < NDOF; j++) q_goal_[j] = q_goal[j];
        for (int j = 0; j < NDOF; j++) vb_[j] = scenario_.params.max_vel[j];
        N_ = variant_.n_samples;

        CUDA_CHECK(cudaMalloc(&d_mu_, NDOF * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ghat_, NDOF * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_goalhat_, NDOF * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_vb_, NDOF * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs_, N_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_, N_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sampled_, N_ * NDOF * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_, N_ * sizeof(curandState)));
        CUDA_CHECK(cudaMemcpy(d_vb_, vb_, NDOF * sizeof(float), cudaMemcpyHostToDevice));
        reset_rng();
    }

    ~CdfEpisodeRunner() {
        cudaFree(d_mu_); cudaFree(d_ghat_); cudaFree(d_goalhat_); cudaFree(d_vb_);
        cudaFree(d_costs_); cudaFree(d_weights_); cudaFree(d_sampled_); cudaFree(d_rng_);
    }

    EpisodeMetrics run() {
        reset_state();
        reset_mu();
        warmup();
        reset_mu();
        reset_rng();

        auto episode_begin = chrono::steady_clock::now();
        float total_control_ms = 0.0f;

        for (int step = 0; step < scenario_.max_steps; step++) {
            float dist = ee_distance(q_, scenario_.params, scenario_.cost_params);
            min_goal_distance_ = fminf(min_goal_distance_, dist);
            if (dist < scenario_.goal_tol) { reached_goal_ = true; steps_taken_ = step; break; }

            auto t0 = chrono::steady_clock::now();
            controller_update(step);
            auto t1 = chrono::steady_clock::now();
            total_control_ms += chrono::duration<float, milli>(t1 - t0).count();

            CUDA_CHECK(cudaMemcpy(mu_, d_mu_, NDOF * sizeof(float), cudaMemcpyDeviceToHost));
            arm_vel_step(q_, mu_, vb_, CDF_DT);
            cumulative_cost_ += host_traj_cost(q_, scenario_, step + 1);
            if (host_min_margin(q_, scenario_, step + 1) <= 0.02f) collisions_++;
            steps_taken_ = step + 1;
        }

        auto episode_end = chrono::steady_clock::now();
        float final_distance = ee_distance(q_, scenario_.params, scenario_.cost_params);
        if (final_distance < scenario_.goal_tol) reached_goal_ = true;

        EpisodeMetrics m;
        m.scenario = scenario_.name; m.planner = variant_.name; m.seed = seed_;
        m.k_samples = N_; m.t_horizon = 1; m.grad_steps = 0; m.alpha = variant_.alpha_mu;
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
        m.sample_budget = static_cast<long long>(steps_taken_) * N_ * 1;
        return m;
    }

private:
    void reset_rng() {
        int block = 256;
        init_curand_kernel<<<(N_ + block - 1) / block, block>>>(
            d_rng_, N_, static_cast<unsigned long long>(seed_));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    void reset_mu() { CUDA_CHECK(cudaMemset(d_mu_, 0, NDOF * sizeof(float))); }

    // Returns the workspace margin m and its joint-space gradient from either
    // the neural margin field (MLP) or the oracle (host_min_margin FD).
    void margin_source(int step, float& m, float gradm[NDOF]) {
        if (variant_.use_neural) cdf_->value_and_grad(q_, m, gradm);          // neural (static)
        else margin_value_grad(q_, scenario_, step, m, gradm);               // analytic (primary)
    }

    void controller_update(int step) {
        float m; float gradm[NDOF];
        margin_source(step, m, gradm);
        float fc; float ghat[NDOF];
        cdf_transform(m, gradm, fc, ghat);   // -> CDF value + unit ascent dir

        float goalhat[NDOF];
        float gv = 0.0f, goalvec[NDOF];
        for (int j = 0; j < NDOF; j++) { goalvec[j] = q_goal_[j] - q_[j]; gv += goalvec[j] * goalvec[j]; }
        gv = sqrtf(gv) + 1.0e-9f;
        for (int j = 0; j < NDOF; j++) goalhat[j] = goalvec[j] / gv;
        int gate_active = (fc < variant_.d_act && fc < gv) ? 1 : 0;

        CUDA_CHECK(cudaMemcpy(d_ghat_, ghat, NDOF * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_goalhat_, goalhat, NDOF * sizeof(float), cudaMemcpyHostToDevice));

        int block = 256;
        cdf_mppi_kernel<<<(N_ + block - 1) / block, block>>>(
            d_mu_, variant_.sigma, d_ghat_, d_goalhat_, d_vb_,
            gate_active, variant_.alpha1, variant_.alpha2,
            d_rng_, d_costs_, d_sampled_, N_);
        cdf_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, N_, variant_.beta);
        cdf_update_mean_kernel<<<1, NDOF>>>(
            d_mu_, d_sampled_, d_weights_, d_vb_, variant_.alpha_mu, N_);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void warmup() { for (int i = 0; i < 4; i++) controller_update(0); }

    void reset_state() {
        for (int j = 0; j < NDOF; j++) q_[j] = scenario_.start_q[j];
        steps_taken_ = 0; collisions_ = 0; reached_goal_ = false; cumulative_cost_ = 0.0f;
        min_goal_distance_ = ee_distance(q_, scenario_.params, scenario_.cost_params);
    }

    CdfVariant variant_;
    Scenario scenario_;
    int seed_;
    NeuralCdf* cdf_;
    float q_goal_[NDOF];
    float vb_[NDOF];
    int N_;

    float q_[NDOF] = {};
    float mu_[NDOF] = {};
    int steps_taken_ = 0, collisions_ = 0;
    bool reached_goal_ = false;
    float cumulative_cost_ = 0.0f, min_goal_distance_ = 0.0f;

    float* d_mu_ = nullptr; float* d_ghat_ = nullptr; float* d_goalhat_ = nullptr;
    float* d_vb_ = nullptr; float* d_costs_ = nullptr; float* d_weights_ = nullptr;
    float* d_sampled_ = nullptr; curandState* d_rng_ = nullptr;
};

// ======================== Scenarios (copied) ========================
static Scenario make_7dof_shelf_reach() {
    Scenario s;
    s.name = "7dof_shelf_reach";
    s.start_q[0]=0.0f; s.start_q[1]=-0.78f; s.start_q[2]=0.0f;
    s.start_q[3]=-2.36f; s.start_q[4]=0.0f; s.start_q[5]=1.57f; s.start_q[6]=0.78f;
    s.goal_tol = 0.15f;
    s.max_steps = 300;
    s.grad_alpha_scale = 0.5f;
    s.cost_params.goal_x = 0.40f; s.cost_params.goal_y = 0.15f; s.cost_params.goal_z = 0.35f;
    s.cost_params.goal_weight = 20.0f;
    s.cost_params.obstacle_weight = 10.0f;
    s.cost_params.obs_influence = 0.08f;
    const Obstacle3D obs[] = { {0.32f, 0.08f, 0.28f, 0.06f} };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    return s;
}

// Static-only variant of 7dof_dynamic_avoid (dynamic obstacle dropped — CDF-MPPI
// is a static-obstacle method; this gives a second static reaching case).
static Scenario make_7dof_static_reach2() {
    Scenario s;
    s.name = "7dof_static_reach2";
    s.start_q[0]=0.0f; s.start_q[1]=-0.78f; s.start_q[2]=0.0f;
    s.start_q[3]=-2.36f; s.start_q[4]=0.0f; s.start_q[5]=1.57f; s.start_q[6]=0.78f;
    s.goal_tol = 0.15f;
    s.max_steps = 300;
    s.cost_params.goal_x = 0.55f; s.cost_params.goal_y = -0.20f; s.cost_params.goal_z = 0.30f;
    s.cost_params.goal_weight = 15.0f;
    s.cost_params.obstacle_weight = 15.0f;
    s.cost_params.obs_influence = 0.10f;
    const Obstacle3D obs[] = { {0.45f, -0.10f, 0.20f, 0.06f}, {0.50f, 0.10f, 0.35f, 0.05f} };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    return s;
}

// Dynamic-obstacle reaching (copied from the Diff-MPPI benchmark verbatim) — a
// moving obstacle crosses the workspace. CDF-MPPI is reactive (H=1) and should
// struggle to anticipate the motion; this is the differentiation test vs the
// multi-step + autodiff-refinement Diff-MPPI.
static Scenario make_7dof_dynamic_avoid() {
    Scenario s;
    s.name = "7dof_dynamic_avoid";
    s.start_q[0]=0.0f; s.start_q[1]=-0.78f; s.start_q[2]=0.0f;
    s.start_q[3]=-2.36f; s.start_q[4]=0.0f; s.start_q[5]=1.57f; s.start_q[6]=0.78f;
    s.goal_tol = 0.10f;
    s.max_steps = 300;
    s.cost_params.goal_x = 0.55f; s.cost_params.goal_y = -0.20f; s.cost_params.goal_z = 0.30f;
    s.cost_params.goal_weight = 15.0f;
    s.cost_params.obstacle_weight = 15.0f;
    s.cost_params.obs_influence = 0.10f;
    const Obstacle3D obs[] = { {0.45f, -0.10f, 0.20f, 0.06f}, {0.50f, 0.10f, 0.35f, 0.05f} };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    const DynamicObstacle3D dyn[] = { {0.70f, 0.0f, 0.30f, -0.15f, -0.05f, 0.0f, 0.06f} };
    s.n_dyn_obs = static_cast<int>(sizeof(dyn) / sizeof(dyn[0]));
    for (int i = 0; i < s.n_dyn_obs; i++) s.dynamic_obstacles[i] = dyn[i];
    return s;
}

// ======================== Utilities (copied) ========================
static void ensure_build_dir() { mkdir("build", 0755); }

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
        string key = r.scenario + " | " + r.planner + " | N=" + to_string(r.k_samples);
        auto& s = stats[key]; s.episodes++;
        s.successes += r.success; s.steps_sum += r.steps;
        s.final_sum += r.final_distance; s.min_sum += r.min_goal_distance;
        s.cost_sum += r.cumulative_cost; s.ms_sum += r.avg_control_ms;
    }
    cout << "=== benchmark_cdf_mppi_7dof summary ===" << endl;
    for (const auto& kv : stats) {
        const SummaryStats& s = kv.second;
        float n = static_cast<float>(s.episodes);
        printf("%s : success=%.2f steps=%.1f final_dist=%.3f min_dist=%.3f cost=%.1f avg_ms=%.2f\n",
               kv.first.c_str(), s.successes / n, s.steps_sum / n,
               s.final_sum / n, s.min_sum / n, s.cost_sum / n, s.ms_sum / n);
    }
}

// ======================== CDF data prep (host) ========================
static void sample_config(mt19937& rng, float q[NDOF]) {
    for (int j = 0; j < NDOF; j++) {
        uniform_real_distribution<float> d(NCDF_Q_LO[j], NCDF_Q_HI[j]);
        q[j] = d(rng);
    }
}

// Training data + goal config seed. Targets are the dense margin-based CDF.
struct CdfData {
    vector<float> query_raw;     // Nq * 7
    vector<float> target_raw;    // Nq
    float q_goal[NDOF];
    float q_goal_ee = 1e9f;
};

static CdfData build_cdf_data(const Scenario& s, int n_candidates, int n_query, unsigned int seed) {
    CdfData data;
    mt19937 rng(seed);
    float best_free_q[NDOF]; float best_free_ee = 1e9f; bool have_free = false;
    const float band = 0.30f;  // |margin| band defining "near contact"

    // 1. queries: the field is gate-relevant only near the contact manifold, so
    //    weight training there. Collect near-contact configs (|margin|<band) up
    //    to ~2/3 of the set; fill the rest with uniform far-field anchors.
    int near_cap = (2 * n_query) / 3;
    int near_count = 0;
    for (int i = 0; i < n_candidates; i++) {
        float q[NDOF]; sample_config(rng, q);
        float margin = host_min_margin(q, s, 0);
        if (margin > 0.02f) {
            float ee = ee_distance(q, s.params, s.cost_params);
            if (ee < best_free_ee) { best_free_ee = ee; for (int j=0;j<NDOF;j++) best_free_q[j]=q[j]; have_free = true; }
        }
        if (fabsf(margin) < band && near_count < near_cap) {
            for (int j=0;j<NDOF;j++) data.query_raw.push_back(q[j]);
            near_count++;
        }
    }
    int n_uniform = n_query / 3;
    for (int i = 0; i < n_uniform; i++) {
        float q[NDOF]; sample_config(rng, q);
        for (int j=0;j<NDOF;j++) data.query_raw.push_back(q[j]);
    }

    // 2. goal config: refine best free seed by gradient descent on ee_distance,
    //    staying collision-free + in limits.
    if (!have_free) { for (int j=0;j<NDOF;j++) best_free_q[j]=s.start_q[j]; }
    float qg[NDOF]; for (int j=0;j<NDOF;j++) qg[j]=best_free_q[j];
    float lr = 0.5f; const float eps = 1e-3f;
    for (int it = 0; it < 300; it++) {
        float base = ee_distance(qg, s.params, s.cost_params);
        float grad[NDOF];
        for (int j = 0; j < NDOF; j++) {
            float qp[NDOF]; for (int k=0;k<NDOF;k++) qp[k]=qg[k];
            qp[j]+=eps; grad[j]=(ee_distance(qp,s.params,s.cost_params)-base)/eps;
        }
        float qn[NDOF];
        for (int j = 0; j < NDOF; j++) qn[j]=clampf_local(qg[j]-lr*grad[j], NCDF_Q_LO[j], NCDF_Q_HI[j]);
        if (host_min_margin(qn, s, 0) > 0.02f && ee_distance(qn,s.params,s.cost_params) < base)
            for (int j=0;j<NDOF;j++) qg[j]=qn[j];
        else lr *= 0.7f;
        if (lr < 1e-3f) break;
    }
    for (int j=0;j<NDOF;j++) data.q_goal[j]=qg[j];
    data.q_goal_ee = ee_distance(qg, s.params, s.cost_params);

    // 3. labels: smooth workspace signed margin m(q) (host FK). The MLP learns
    //    this neural-SDF-style field; fc/ghat are derived analytically at runtime.
    int Nq = static_cast<int>(data.query_raw.size()) / NCDF_INPUT_DIM;
    data.target_raw.assign(Nq, 0.0f);
    for (int i = 0; i < Nq; i++)
        data.target_raw[i] = host_min_margin(&data.query_raw[i*NCDF_INPUT_DIM], s, 0);
    return data;
}

// Gradient cosine-similarity: neural CDF grad vs dense margin-field grad,
// measured ONLY near the contact manifold (|margin|<band) where the gradient is
// gate-active and actually used by the controller. Far-field gradients are ~0
// (saturated) and meaningless to compare.
static float validate_gradient_cosine(NeuralCdf& cdf, const Scenario& s, int K, unsigned int seed) {
    mt19937 rng(seed);
    double cos_sum = 0.0; int n = 0, tries = 0;
    while (n < K && tries < K * 500) {
        tries++;
        float q[NDOF]; sample_config(rng, q);
        if (fabsf(host_min_margin(q, s, 0)) > 0.30f) continue;  // near-contact only
        float mm; float gm[NDOF]; cdf.value_and_grad(q, mm, gm);     // neural margin grad
        float mt; float gt[NDOF]; margin_value_grad(q, s, 0, mt, gt);   // true margin grad
        double dot=0, nm=0, nt=0;
        for (int j=0;j<NDOF;j++){ dot+=gm[j]*gt[j]; nm+=gm[j]*gm[j]; nt+=gt[j]*gt[j]; }
        if (nm > 1e-12 && nt > 1e-12) { cos_sum += dot/(sqrt(nm)*sqrt(nt)); n++; }
    }
    return n > 0 ? static_cast<float>(cos_sum / n) : 0.0f;
}

// Render a 2D CDF slice (q1,q2 swept; q3..q7 fixed at goal config).
static void render_cdf_slice(NeuralCdf& cdf, const Scenario& s, const CdfData& data,
                             const string& path, int res = 96) {
    cv::Mat heat(res, res, CV_8UC1);
    cv::Mat collide(res, res, CV_8UC1, cv::Scalar(0));
    float fmax = 1e-3f;
    vector<float> vals(res*res);
    for (int iy = 0; iy < res; iy++) {
        for (int ix = 0; ix < res; ix++) {
            float q[NDOF]; for (int j=0;j<NDOF;j++) q[j]=data.q_goal[j];
            q[0] = NCDF_Q_LO[0] + (NCDF_Q_HI[0]-NCDF_Q_LO[0]) * (ix+0.5f)/res;
            q[1] = NCDF_Q_LO[1] + (NCDF_Q_HI[1]-NCDF_Q_LO[1]) * (iy+0.5f)/res;
            float v = cdf.value(q);
            vals[iy*res+ix] = v; fmax = max(fmax, v);
            if (host_min_margin(q, s, 0) < 0.0f) collide.at<uchar>(res-1-iy, ix) = 255;
        }
    }
    for (int iy = 0; iy < res; iy++)
        for (int ix = 0; ix < res; ix++)
            heat.at<uchar>(res-1-iy, ix) = (uchar)(255.0f * clampf_local(vals[iy*res+ix]/fmax,0,1));
    cv::Mat color; cv::applyColorMap(heat, color, cv::COLORMAP_TURBO);
    cv::resize(color, color, cv::Size(420,420), 0,0, cv::INTER_NEAREST);
    cv::Mat coll_big; cv::resize(collide, coll_big, cv::Size(420,420),0,0,cv::INTER_NEAREST);
    for (int y=0;y<420;y++) for (int x=0;x<420;x++)
        if (coll_big.at<uchar>(y,x)>128) { color.at<cv::Vec3b>(y,x)=cv::Vec3b(255,255,255); }
    cv::putText(color, "CDF slice (q1,q2) "+s.name, cv::Point(10,26),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
    cv::imwrite(path, color);
}

// ======================== Main ========================
int main(int argc, char** argv) {
    bool quick = false, do_validate = false;
    string csv_path = "build/benchmark_cdf_mppi_7dof.csv";
    vector<string> scenario_names, planner_names;
    int seed_count = -1;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--quick") quick = true;
        else if (arg == "--validate" || arg == "--train-cdf") do_validate = true;
        else if (arg == "--csv" && i+1 < argc) csv_path = argv[++i];
        else if (arg == "--seed-count" && i+1 < argc) seed_count = max(1, atoi(argv[++i]));
        else if (arg == "--scenarios" && i+1 < argc) scenario_names = parse_string_list(argv[++i]);
        else if (arg == "--planners" && i+1 < argc) planner_names = parse_string_list(argv[++i]);
    }
    ensure_build_dir();

    vector<Scenario> all_scenarios = {
        make_7dof_shelf_reach(), make_7dof_static_reach2(), make_7dof_dynamic_avoid() };
    // CDF-MPPI replans at CDF_DT; reflect it in the scenario for cost/time parity.
    for (auto& s : all_scenarios) s.params.dt = CDF_DT;

    vector<Scenario> scenarios;
    if (!scenario_names.empty()) {
        for (const auto& w : scenario_names) {
            auto it = find_if(all_scenarios.begin(), all_scenarios.end(),
                              [&](const Scenario& s){ return s.name == w; });
            if (it == all_scenarios.end()) { fprintf(stderr,"Unknown scenario: %s\n", w.c_str()); return 1; }
            scenarios.push_back(*it);
        }
    } else scenarios = all_scenarios;

    vector<CdfVariant> variants;
    { CdfVariant v; v.name = "cdf_mppi"; variants.push_back(v); }                       // analytic CDF (primary)
    { CdfVariant v; v.name = "cdf_mppi_neural"; v.use_neural = true; variants.push_back(v); }  // neural ablation
    if (!planner_names.empty()) {
        vector<CdfVariant> filt;
        for (const auto& w : planner_names) {
            auto it = find_if(variants.begin(), variants.end(),
                              [&](const CdfVariant& v){ return v.name == w; });
            if (it == variants.end()) { fprintf(stderr,"Unknown planner: %s\n", w.c_str()); return 1; }
            filt.push_back(*it);
        }
        variants.swap(filt);
    }
    if (seed_count <= 0) seed_count = quick ? 2 : 4;

    int n_candidates = quick ? 120000 : 400000;
    int n_query = quick ? 20000 : 40000;

    vector<EpisodeMetrics> rows;
    for (size_t si = 0; si < scenarios.size(); si++) {
        const Scenario& scenario = scenarios[si];
        printf("\n=== scenario %s : building CDF (candidates=%d) ===\n",
               scenario.name.c_str(), n_candidates);
        CdfData data = build_cdf_data(scenario, n_candidates, n_query, 100 + si);
        printf("  queries Nq=%d, q_goal EE dist=%.3f (goal_tol=%.3f)\n",
               (int)(data.target_raw.size()), data.q_goal_ee, scenario.goal_tol);
        // Export the goal configuration so the Diff-MPPI binary can use the
        // SAME q_goal (fair-rematch: removes CDF-MPPI's free-IK advantage).
        {
            string qg_path = "build/qgoal_" + scenario.name + ".txt";
            ofstream qg(qg_path);
            for (int j = 0; j < NDOF; j++) qg << data.q_goal[j] << (j+1<NDOF ? ' ' : '\n');
            printf("  q_goal exported -> %s\n", qg_path.c_str());
        }
        if (data.q_goal_ee > scenario.goal_tol)
            printf("  [warn] q_goal EE (%.3f) exceeds goal_tol — CDF-MPPI cannot reach success here.\n",
                   data.q_goal_ee);

        // Train neural CDF only if a neural-ablation variant is selected.
        bool need_neural = false;
        for (auto& v : variants) if (v.use_neural) need_neural = true;
        NeuralCdf cdf;
        if (need_neural) {
            printf("  training neural CDF on Nq=%d queries...\n",
                   (int)(data.target_raw.size()));
            int epochs = quick ? 400 : 1200;
            float rmse = cdf.train(data.query_raw, data.target_raw, epochs);
            if (do_validate) {
                float cosg = validate_gradient_cosine(cdf, scenario, 200, 7);
                string png = "build/cdf_slice_" + scenario.name + ".png";
                render_cdf_slice(cdf, scenario, data, png);
                printf("  [validate] held-out RMSE=%.4f rad, grad cos-sim=%.3f (need>0.7), slice -> %s\n",
                       rmse, cosg, png.c_str());
            }
        }

        for (size_t vi = 0; vi < variants.size(); vi++) {
            const CdfVariant& variant = variants[vi];
            for (int seed = 0; seed < seed_count; seed++) {
                int run_seed = static_cast<int>(3000 + si * 100 + vi * 20 + seed * 7);
                CdfEpisodeRunner runner(variant, scenario, data.q_goal, run_seed, &cdf);
                EpisodeMetrics m = runner.run();
                rows.push_back(m);
                printf("[%s] %s seed=%d success=%d steps=%d final_dist=%.3f avg_ms=%.3f collisions=%d\n",
                       scenario.name.c_str(), variant.name.c_str(), seed,
                       m.success, m.steps, m.final_distance, m.avg_control_ms, m.collisions);
            }
        }
    }

    write_csv(rows, csv_path);
    print_summary(rows);
    cout << "CSV saved to " << csv_path << endl;
    return 0;
}
