/*************************************************************************
    Differentiable Planar Pushing — Diff-MPPI vs vanilla MPPI
    ---------------------------------------------------------------------
    The decisive gap-#2 test for the Diff-MPPI research line: a task where the
    model gradient flows THROUGH CONTACT, so an autodiff refinement of the MPPI
    control mean carries information pure sampling lacks.

    Quasi-static non-prehensile pushing: a point pusher (px,py) pushes a disk
    object (ox,oy) toward a goal. The object moves ONLY when in contact, via a
    SMOOTH (softplus penetration -> normal force) contact model, so the rollout
    cost is differentiable in the controls and forward-mode dual-number autodiff
    yields the exact gradient through contact.

    Hypothesis: random velocity samples rarely sustain directed contact, so
    vanilla MPPI is inefficient; the autodiff gradient directly tells the pusher
    how to stay in contact and push toward the goal -> Diff-MPPI should win.

    CSV schema matches the other Diff-MPPI benchmarks.
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
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "autodiff_engine.cuh"

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;
using namespace cudabot;

static const int STATE_DIM = 4;   // px, py, ox, oy
static const int CTRL_DIM = 2;    // ux, uy (pusher velocity)
static const int DEFAULT_T = 12;

struct PushParams {
    float dt = 0.05f;
    float u_max = 2.0f;       // pusher velocity bound
    float obj_r = 0.30f;      // object disk radius
    float push_r = 0.10f;     // pusher radius
    float contact = 0.40f;    // obj_r + push_r
    float ksoft = 18.0f;      // softplus sharpness
    float push_gain = 9.0f;   // contact mobility (object speed per unit force)
    // cost weights
    float w_obj = 1.0f;       // stage object-to-goal
    float w_ctrl = 0.01f;     // control reg
    float w_near = 0.05f;     // pusher-near-object shaping (smooth approach grad)
    float w_term = 60.0f;     // terminal object-to-goal
};

struct PushScenario {
    string name;
    float px0, py0, ox0, oy0;
    float goal_x, goal_y;
    float goal_tol = 0.20f;
    int max_steps = 140;
    PushParams params;
};

struct Variant {
    string name;
    int grad_steps = 0;
    float alpha = 0.0f;
    float grad_clip = 30.0f;
    float sigma = 0.6f;          // velocity sampling std
    float lambda = 5.0f;         // softmax temperature
    bool use_low_pass_sampling = false;
    float lp_alpha = 0.35f;
    bool use_soppi_sampling = false;
    int soppi_svgd_iters = 1;
    float soppi_step_size = 0.06f;
    float soppi_bandwidth = 2.0f;
    int soppi_neighbor_count = 0; // 0 = all particles; >0 = deterministic particle subset
    bool use_object_informed = false;
    float oi_ref_weight = 2.0f;     // track an object-only reference trajectory
    float oi_obj_speed = 1.8f;      // direct-actuated object speed used by the reference
    float oi_seed_blend = 0.15f;    // blend nominal toward reference contact strategy
    float oi_contact_margin = 0.04f;
};

struct EpisodeMetrics {
    string scenario, planner;
    int seed = 0, k_samples = 0, t_horizon = 0, grad_steps = 0;
    float alpha = 0.0f;
    int reached_goal = 0, collision_free = 1, success = 0, steps = 0;
    float final_distance = 0.0f, min_goal_distance = 0.0f, cumulative_cost = 0.0f;
    int collisions = 0;
    float mean_control_delta = 0.0f, control_roughness = 0.0f;
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

// ---- float (rollout / plant) ----
__host__ __device__ inline float softplus_f(float x, float k) {
    float kx = k * x;
    if (kx > 20.0f) return x;          // stable: log1p(exp)->kx for large kx
    return log1pf(expf(kx)) / k;
}

__host__ __device__ inline void push_step_f(
    float& px, float& py, float& ox, float& oy, float ux, float uy, const PushParams& p)
{
    px += p.dt * ux;
    py += p.dt * uy;
    float dx = ox - px, dy = oy - py;
    float dist = sqrtf(dx*dx + dy*dy + 1e-9f);
    float nx = dx / dist, ny = dy / dist;
    float pen = p.contact - dist;                 // >0 when overlapping
    float F = softplus_f(pen, p.ksoft);
    ox += p.dt * p.push_gain * F * nx;
    oy += p.dt * p.push_gain * F * ny;
}

__host__ __device__ inline float stage_cost_f(
    float px, float py, float ox, float oy, float ux, float uy,
    float gx, float gy, const PushParams& p)
{
    float dox = ox - gx, doy = oy - gy;
    float c = p.w_obj * (dox*dox + doy*doy) * p.dt;
    c += p.w_ctrl * (ux*ux + uy*uy) * p.dt;
    float dpx = px - ox, dpy = py - oy;
    c += p.w_near * (dpx*dpx + dpy*dpy) * p.dt;   // keep pusher near object
    return c;
}

__host__ __device__ inline float terminal_cost_f(float ox, float oy, float gx, float gy, const PushParams& p) {
    float dox = ox - gx, doy = oy - gy;
    return p.w_term * (dox*dox + doy*doy);
}

__host__ __device__ inline void object_ref_disk_f(
    float ox0, float oy0, float gx, float gy, float dt, float obj_speed, int step,
    float& rx, float& ry)
{
    float dx = gx - ox0, dy = gy - oy0;
    float dist = sqrtf(dx*dx + dy*dy + 1e-9f);
    float travel = fminf(dist, fmaxf(0.0f, obj_speed) * dt * static_cast<float>(step));
    rx = ox0 + dx / dist * travel;
    ry = oy0 + dy / dist * travel;
}

// ---- Dualf (gradient through contact, forward-mode) ----
__device__ inline Dualf softplus_d(const Dualf& x, float k) {
    Dualf kx = Dualf::constant(k) * x;
    Dualf e = cudabot::exp(kx);
    return cudabot::log(Dualf::constant(1.0f) + e) * Dualf::constant(1.0f / k);
}

__device__ inline void push_step_d(
    Dualf& px, Dualf& py, Dualf& ox, Dualf& oy, Dualf ux, Dualf uy, const PushParams& p)
{
    px = px + Dualf::constant(p.dt) * ux;
    py = py + Dualf::constant(p.dt) * uy;
    Dualf dx = ox - px, dy = oy - py;
    Dualf dist = cudabot::sqrt(dx*dx + dy*dy + Dualf::constant(1e-9f));
    Dualf nx = dx / dist, ny = dy / dist;
    Dualf pen = Dualf::constant(p.contact) - dist;
    Dualf F = softplus_d(pen, p.ksoft);
    ox = ox + Dualf::constant(p.dt * p.push_gain) * F * nx;
    oy = oy + Dualf::constant(p.dt * p.push_gain) * F * ny;
}

// Forward-mode derivative of total rollout cost w.r.t. nominal control `active`.
__device__ inline float dcost_dparam(
    const float start[STATE_DIM], const float* nominal, int T, int active,
    float gx, float gy, const PushParams& p)
{
    Dualf px = Dualf::constant(start[0]);
    Dualf py = Dualf::constant(start[1]);
    Dualf ox = Dualf::constant(start[2]);
    Dualf oy = Dualf::constant(start[3]);
    Dualf cost = Dualf::constant(0.0f);
    for (int t = 0; t < T; t++) {
        Dualf ux = (active == t*2+0) ? Dualf::variable(nominal[t*2+0]) : Dualf::constant(nominal[t*2+0]);
        Dualf uy = (active == t*2+1) ? Dualf::variable(nominal[t*2+1]) : Dualf::constant(nominal[t*2+1]);
        ux = clamp(ux, -p.u_max, p.u_max);
        uy = clamp(uy, -p.u_max, p.u_max);
        push_step_d(px, py, ox, oy, ux, uy, p);
        Dualf dox = ox - Dualf::constant(gx), doy = oy - Dualf::constant(gy);
        cost = cost + Dualf::constant(p.w_obj) * (dox*dox + doy*doy) * Dualf::constant(p.dt);
        cost = cost + Dualf::constant(p.w_ctrl) * (ux*ux + uy*uy) * Dualf::constant(p.dt);
        Dualf dpx = px - ox, dpy = py - oy;
        cost = cost + Dualf::constant(p.w_near) * (dpx*dpx + dpy*dpy) * Dualf::constant(p.dt);
    }
    Dualf dox = ox - Dualf::constant(gx), doy = oy - Dualf::constant(gy);
    cost = cost + Dualf::constant(p.w_term) * (dox*dox + doy*doy);
    return cost.deriv;
}

// ======================== Kernels ========================
__global__ void init_curand_kernel(curandState* st, int n, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &st[i]);
}

__global__ void push_rollout_kernel(
    const float* d_start, const float* d_nominal, float* d_costs, float* d_perturbed,
    curandState* d_rng, PushParams p, float gx, float gy, int K, int T, float sigma)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curandState rng = d_rng[k];
    float px = d_start[0], py = d_start[1], ox = d_start[2], oy = d_start[3];
    float cost = 0.0f;
    for (int t = 0; t < T; t++) {
        float ux = d_nominal[t*2+0] + curand_normal(&rng) * sigma;
        float uy = d_nominal[t*2+1] + curand_normal(&rng) * sigma;
        ux = clampf_local(ux, -p.u_max, p.u_max);
        uy = clampf_local(uy, -p.u_max, p.u_max);
        d_perturbed[k*T*2 + t*2 + 0] = ux;
        d_perturbed[k*T*2 + t*2 + 1] = uy;
        push_step_f(px, py, ox, oy, ux, uy, p);
        cost += stage_cost_f(px, py, ox, oy, ux, uy, gx, gy, p);
    }
    cost += terminal_cost_f(ox, oy, gx, gy, p);
    d_costs[k] = cost;
    d_rng[k] = rng;
}

__global__ void push_low_pass_rollout_kernel(
    const float* d_start, const float* d_nominal, float* d_costs, float* d_perturbed,
    curandState* d_rng, PushParams p, float gx, float gy, int K, int T, float sigma, float lp_alpha)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curandState rng = d_rng[k];
    float px = d_start[0], py = d_start[1], ox = d_start[2], oy = d_start[3];
    float cost = 0.0f;
    float fx = 0.0f, fy = 0.0f;
    float alpha = clampf_local(lp_alpha, 0.02f, 1.0f);
    float beta = 1.0f - alpha;
    float variance_gain = sqrtf((2.0f - alpha) / alpha);
    for (int t = 0; t < T; t++) {
        fx = beta * fx + alpha * curand_normal(&rng);
        fy = beta * fy + alpha * curand_normal(&rng);
        float ux = d_nominal[t*2+0] + fx * variance_gain * sigma;
        float uy = d_nominal[t*2+1] + fy * variance_gain * sigma;
        ux = clampf_local(ux, -p.u_max, p.u_max);
        uy = clampf_local(uy, -p.u_max, p.u_max);
        d_perturbed[k*T*2 + t*2 + 0] = ux;
        d_perturbed[k*T*2 + t*2 + 1] = uy;
        push_step_f(px, py, ox, oy, ux, uy, p);
        cost += stage_cost_f(px, py, ox, oy, ux, uy, gx, gy, p);
    }
    cost += terminal_cost_f(ox, oy, gx, gy, p);
    d_costs[k] = cost;
    d_rng[k] = rng;
}

__global__ void push_object_informed_rollout_kernel(
    const float* d_start, const float* d_nominal, float* d_costs, float* d_perturbed,
    curandState* d_rng, PushParams p, float gx, float gy, int K, int T, float sigma,
    bool use_low_pass, float lp_alpha, float oi_ref_weight, float oi_obj_speed)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curandState rng = d_rng[k];
    const float ox0 = d_start[2], oy0 = d_start[3];
    float px = d_start[0], py = d_start[1], ox = ox0, oy = oy0;
    float cost = 0.0f;
    float fx = 0.0f, fy = 0.0f;
    float alpha = clampf_local(lp_alpha, 0.02f, 1.0f);
    float beta = 1.0f - alpha;
    float variance_gain = use_low_pass ? sqrtf((2.0f - alpha) / alpha) : 1.0f;
    for (int t = 0; t < T; t++) {
        float nx = curand_normal(&rng);
        float ny = curand_normal(&rng);
        if (use_low_pass) {
            fx = beta * fx + alpha * nx;
            fy = beta * fy + alpha * ny;
            nx = fx * variance_gain;
            ny = fy * variance_gain;
        }
        float ux = d_nominal[t*2+0] + nx * sigma;
        float uy = d_nominal[t*2+1] + ny * sigma;
        ux = clampf_local(ux, -p.u_max, p.u_max);
        uy = clampf_local(uy, -p.u_max, p.u_max);
        d_perturbed[k*T*2 + t*2 + 0] = ux;
        d_perturbed[k*T*2 + t*2 + 1] = uy;
        push_step_f(px, py, ox, oy, ux, uy, p);
        cost += stage_cost_f(px, py, ox, oy, ux, uy, gx, gy, p);
        float rx, ry;
        object_ref_disk_f(ox0, oy0, gx, gy, p.dt, oi_obj_speed, t + 1, rx, ry);
        float erx = ox - rx, ery = oy - ry;
        cost += fmaxf(0.0f, oi_ref_weight) * (erx*erx + ery*ery) * p.dt;
    }
    cost += terminal_cost_f(ox, oy, gx, gy, p);
    d_costs[k] = cost;
    d_rng[k] = rng;
}

__global__ void push_fixed_rollout_kernel(
    const float* d_start, const float* d_controls, float* d_costs,
    PushParams p, float gx, float gy, int K, int T)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    float px = d_start[0], py = d_start[1], ox = d_start[2], oy = d_start[3];
    float cost = 0.0f;
    for (int t = 0; t < T; t++) {
        float ux = clampf_local(d_controls[k*T*2 + t*2 + 0], -p.u_max, p.u_max);
        float uy = clampf_local(d_controls[k*T*2 + t*2 + 1], -p.u_max, p.u_max);
        push_step_f(px, py, ox, oy, ux, uy, p);
        cost += stage_cost_f(px, py, ox, oy, ux, uy, gx, gy, p);
    }
    cost += terminal_cost_f(ox, oy, gx, gy, p);
    d_costs[k] = cost;
}

__global__ void push_weights_kernel(const float* d_costs, float* d_w, int K, float lambda) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float cmin = FLT_MAX;
    for (int k = 0; k < K; k++) cmin = fminf(cmin, d_costs[k]);
    float sum = 0.0f;
    for (int k = 0; k < K; k++) { float w = expf(-(d_costs[k]-cmin)/lambda); d_w[k]=w; sum+=w; }
    if (sum > 0.0f) for (int k = 0; k < K; k++) d_w[k] /= sum;
}

__global__ void push_update_kernel(float* d_nominal, const float* d_perturbed, const float* d_w, int K, int T) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float ux = 0.0f, uy = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = d_w[k];
        ux += w * d_perturbed[k*T*2 + t*2 + 0];
        uy += w * d_perturbed[k*T*2 + t*2 + 1];
    }
    d_nominal[t*2+0] = ux;
    d_nominal[t*2+1] = uy;
}

// Single-thread autodiff gradient refinement of the control mean.
__global__ void push_grad_step_kernel(
    const float* d_start, float* d_nominal, PushParams p, float gx, float gy,
    int T, float alpha, float grad_clip)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float start[STATE_DIM] = { d_start[0], d_start[1], d_start[2], d_start[3] };
    // grad of total rollout cost w.r.t. each control param (forward-mode, 2T passes)
    float gnorm2 = 0.0f;
    float grad[2*32];   // T <= 32
    for (int kparam = 0; kparam < 2*T; kparam++) {
        float g = dcost_dparam(start, d_nominal, T, kparam, gx, gy, p);
        grad[kparam] = g;
        gnorm2 += g*g;
    }
    float scale = alpha;
    float gnorm = sqrtf(gnorm2);
    if (grad_clip > 0.0f && gnorm > grad_clip) scale = alpha * grad_clip / gnorm;
    for (int kparam = 0; kparam < 2*T; kparam++) {
        float v = d_nominal[kparam] - scale * grad[kparam];
        d_nominal[kparam] = clampf_local(v, -p.u_max, p.u_max);
    }
}

__global__ void push_sample_grad_kernel(
    const float* d_start, const float* d_controls, float* d_control_grads,
    PushParams p, float gx, float gy, int K, int T)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    float start[STATE_DIM] = { d_start[0], d_start[1], d_start[2], d_start[3] };
    const float* controls = &d_controls[k*T*CTRL_DIM];
    for (int param = 0; param < T*CTRL_DIM; param++) {
        d_control_grads[k*T*CTRL_DIM + param] = dcost_dparam(start, controls, T, param, gx, gy, p);
    }
}

__global__ void push_soppi_svgd_step_kernel(
    const float* d_controls, float* d_controls_next, const float* d_control_grads,
    PushParams p, int K, int T, int neighbor_count,
    float lambda, float bandwidth, float step_size, float sigma)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * T;
    if (idx >= total) return;

    const float noise = fmaxf(0.05f, sigma);
    const float h = fmaxf(0.10f, bandwidth);
    int k = idx / T;
    int t = idx - k * T;
    int base = k*T*CTRL_DIM + t*CTRL_DIM;
    float ux_i = d_controls[base + 0];
    float uy_i = d_controls[base + 1];
    int neighbor_samples = K;
    if (neighbor_count > 0 && neighbor_count < K) neighbor_samples = neighbor_count;
    int stride = K / neighbor_samples;
    if (stride < 1) stride = 1;

    float phi_x = 0.0f;
    float phi_y = 0.0f;
    for (int m = 0; m < neighbor_samples; m++) {
        int j = neighbor_count > 0 ? (k + m * stride) % K : m;
        int jbase = j*T*CTRL_DIM + t*CTRL_DIM;
        float ux_j = d_controls[jbase + 0];
        float uy_j = d_controls[jbase + 1];
        float dx = (ux_j - ux_i) / noise;
        float dy = (uy_j - uy_i) / noise;
        float k_rbf = expf(-(dx*dx + dy*dy) / h);

        float score_x = -clampf_local(d_control_grads[jbase + 0] / fmaxf(lambda, 1.0e-3f), -25.0f, 25.0f);
        float score_y = -clampf_local(d_control_grads[jbase + 1] / fmaxf(lambda, 1.0e-3f), -25.0f, 25.0f);
        float repel_x = -2.0f * k_rbf * dx / (h * noise);
        float repel_y = -2.0f * k_rbf * dy / (h * noise);
        phi_x += k_rbf * score_x + repel_x;
        phi_y += k_rbf * score_y + repel_y;
    }
    phi_x /= fmaxf(1.0f, static_cast<float>(neighbor_samples));
    phi_y /= fmaxf(1.0f, static_cast<float>(neighbor_samples));
    d_controls_next[base + 0] = clampf_local(ux_i + clampf_local(step_size * phi_x, -0.35f, 0.35f), -p.u_max, p.u_max);
    d_controls_next[base + 1] = clampf_local(uy_i + clampf_local(step_size * phi_y, -0.35f, 0.35f), -p.u_max, p.u_max);
}

// ======================== Episode Runner ========================
class EpisodeRunner {
public:
    EpisodeRunner(const Variant& v, const PushScenario& sc, int K, int T, int seed)
        : v_(v), sc_(sc), K_(K), T_(T), seed_(seed) {
        h_nominal_.assign(T_*CTRL_DIM, 0.0f);
        CUDA_CHECK(cudaMalloc(&d_start_, STATE_DIM*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nominal_, T_*CTRL_DIM*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs_, K_*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_, K_*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perturbed_, K_*T_*CTRL_DIM*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_, K_*sizeof(curandState)));
        if (v_.use_soppi_sampling) {
            CUDA_CHECK(cudaMalloc(&d_soppi_scratch_, K_*T_*CTRL_DIM*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_soppi_grad_, K_*T_*CTRL_DIM*sizeof(float)));
        }
        reset_rng();
    }
    ~EpisodeRunner() {
        cudaFree(d_start_); cudaFree(d_nominal_); cudaFree(d_costs_);
        cudaFree(d_weights_); cudaFree(d_perturbed_); cudaFree(d_rng_);
        if (d_soppi_scratch_) cudaFree(d_soppi_scratch_);
        if (d_soppi_grad_) cudaFree(d_soppi_grad_);
    }

    EpisodeMetrics run() {
        reset_state();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        warmup();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        reset_rng();

        auto ep0 = chrono::steady_clock::now();
        float ctrl_ms = 0.0f;
        float prev_ux = 0.0f, prev_uy = 0.0f;
        bool have_prev_control = false;
        float control_delta_sum = 0.0f;
        float control_roughness_sum = 0.0f;
        int control_delta_count = 0;
        for (int step = 0; step < sc_.max_steps; step++) {
            float dist = obj_goal_dist();
            min_dist_ = fminf(min_dist_, dist);
            if (dist < sc_.goal_tol) { reached_ = true; steps_ = step; break; }

            auto t0 = chrono::steady_clock::now();
            controller_update();
            auto t1 = chrono::steady_clock::now();
            ctrl_ms += chrono::duration<float, milli>(t1 - t0).count();

            CUDA_CHECK(cudaMemcpy(h_nominal_.data(), d_nominal_, h_nominal_.size()*sizeof(float), cudaMemcpyDeviceToHost));
            if (have_prev_control) {
                float dux = h_nominal_[0] - prev_ux;
                float duy = h_nominal_[1] - prev_uy;
                control_delta_sum += sqrtf(dux*dux + duy*duy);
                control_roughness_sum += dux*dux + duy*duy;
                control_delta_count++;
            }
            prev_ux = h_nominal_[0];
            prev_uy = h_nominal_[1];
            have_prev_control = true;
            push_step_f(px_, py_, ox_, oy_, h_nominal_[0], h_nominal_[1], sc_.params);
            cum_cost_ += stage_cost_f(px_, py_, ox_, oy_, h_nominal_[0], h_nominal_[1], sc_.goal_x, sc_.goal_y, sc_.params);

            // shift nominal
            for (int t = 0; t < T_-1; t++) { h_nominal_[t*2+0]=h_nominal_[(t+1)*2+0]; h_nominal_[t*2+1]=h_nominal_[(t+1)*2+1]; }
            h_nominal_[(T_-1)*2+0]=0.0f; h_nominal_[(T_-1)*2+1]=0.0f;
            steps_ = step + 1;
        }
        auto ep1 = chrono::steady_clock::now();

        EpisodeMetrics m;
        m.scenario = sc_.name; m.planner = v_.name; m.seed = seed_;
        m.k_samples = K_; m.t_horizon = T_; m.grad_steps = v_.grad_steps; m.alpha = v_.alpha;
        float fd = obj_goal_dist();
        if (fd < sc_.goal_tol) reached_ = true;
        m.reached_goal = reached_ ? 1 : 0;
        m.success = m.reached_goal;  // no collision concept here
        m.steps = steps_;
        m.final_distance = fd; m.min_goal_distance = min_dist_;
        m.cumulative_cost = cum_cost_;
        m.mean_control_delta = control_delta_count > 0 ? control_delta_sum / control_delta_count : 0.0f;
        m.control_roughness = control_delta_count > 0 ? control_roughness_sum / control_delta_count : 0.0f;
        m.total_control_ms = ctrl_ms;
        m.avg_control_ms = steps_ > 0 ? ctrl_ms / steps_ : 0.0f;
        m.episode_ms = chrono::duration<float, milli>(ep1 - ep0).count();
        m.sample_budget = static_cast<long long>(steps_) * K_ * T_;
        return m;
    }

private:
    void reset_rng() {
        int b = 256;
        init_curand_kernel<<<(K_+b-1)/b, b>>>(d_rng_, K_, (unsigned long long)seed_);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    void reset_state() {
        px_ = sc_.px0; py_ = sc_.py0; ox_ = sc_.ox0; oy_ = sc_.oy0;
        steps_ = 0; reached_ = false; cum_cost_ = 0.0f; min_dist_ = obj_goal_dist();
    }
    float obj_goal_dist() const {
        float dx = ox_ - sc_.goal_x, dy = oy_ - sc_.goal_y;
        return sqrtf(dx*dx + dy*dy);
    }
    void sync_start() {
        float s[STATE_DIM] = { px_, py_, ox_, oy_ };
        CUDA_CHECK(cudaMemcpy(d_start_, s, STATE_DIM*sizeof(float), cudaMemcpyHostToDevice));
    }
    void controller_update() {
        sync_start();
        seed_object_informed_nominal();
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size()*sizeof(float), cudaMemcpyHostToDevice));
        int b = 256;
        if (v_.use_object_informed) {
            push_object_informed_rollout_kernel<<<(K_+b-1)/b, b>>>(
                d_start_, d_nominal_, d_costs_, d_perturbed_, d_rng_,
                sc_.params, sc_.goal_x, sc_.goal_y, K_, T_, v_.sigma,
                v_.use_low_pass_sampling, v_.lp_alpha, v_.oi_ref_weight, v_.oi_obj_speed);
        } else if (v_.use_low_pass_sampling) {
            push_low_pass_rollout_kernel<<<(K_+b-1)/b, b>>>(
                d_start_, d_nominal_, d_costs_, d_perturbed_, d_rng_,
                sc_.params, sc_.goal_x, sc_.goal_y, K_, T_, v_.sigma, v_.lp_alpha);
        } else {
            push_rollout_kernel<<<(K_+b-1)/b, b>>>(
                d_start_, d_nominal_, d_costs_, d_perturbed_, d_rng_,
                sc_.params, sc_.goal_x, sc_.goal_y, K_, T_, v_.sigma);
        }
        if (v_.use_soppi_sampling) {
            int total_particles = K_ * T_;
            for (int iter = 0; iter < max(1, v_.soppi_svgd_iters); iter++) {
                push_sample_grad_kernel<<<(K_+b-1)/b, b>>>(
                    d_start_, d_perturbed_, d_soppi_grad_,
                    sc_.params, sc_.goal_x, sc_.goal_y, K_, T_);
                push_soppi_svgd_step_kernel<<<(total_particles+b-1)/b, b>>>(
                    d_perturbed_, d_soppi_scratch_, d_soppi_grad_,
                    sc_.params, K_, T_, v_.soppi_neighbor_count,
                    v_.lambda, v_.soppi_bandwidth, v_.soppi_step_size, v_.sigma);
                CUDA_CHECK(cudaMemcpy(d_perturbed_, d_soppi_scratch_,
                                      K_*T_*CTRL_DIM*sizeof(float), cudaMemcpyDeviceToDevice));
                push_fixed_rollout_kernel<<<(K_+b-1)/b, b>>>(
                    d_start_, d_perturbed_, d_costs_,
                    sc_.params, sc_.goal_x, sc_.goal_y, K_, T_);
            }
        }
        push_weights_kernel<<<1,1>>>(d_costs_, d_weights_, K_, v_.lambda);
        push_update_kernel<<<(T_+b-1)/b, b>>>(d_nominal_, d_perturbed_, d_weights_, K_, T_);
        for (int g = 0; g < v_.grad_steps; g++) {
            push_grad_step_kernel<<<1,1>>>(d_start_, d_nominal_, sc_.params, sc_.goal_x, sc_.goal_y,
                                           T_, v_.alpha, v_.grad_clip);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    void warmup() { for (int i = 0; i < 3; i++) controller_update(); }

    void seed_object_informed_nominal() {
        if (!v_.use_object_informed || v_.oi_seed_blend <= 0.0f) return;
        const PushParams& p = sc_.params;
        float sim_px = px_, sim_py = py_;
        float blend = clampf_local(v_.oi_seed_blend, 0.0f, 1.0f);
        float dxg = sc_.goal_x - ox_, dyg = sc_.goal_y - oy_;
        float dg = sqrtf(dxg*dxg + dyg*dyg + 1e-9f);
        float dirx = dxg / dg, diry = dyg / dg;
        float contact_offset = p.contact + fmaxf(0.0f, v_.oi_contact_margin);
        for (int t = 0; t < T_; t++) {
            float refx, refy;
            object_ref_disk_f(ox_, oy_, sc_.goal_x, sc_.goal_y, p.dt, v_.oi_obj_speed, t + 1, refx, refy);
            float target_px = refx - dirx * contact_offset;
            float target_py = refy - diry * contact_offset;
            float ux = clampf_local((target_px - sim_px) / p.dt, -p.u_max, p.u_max);
            float uy = clampf_local((target_py - sim_py) / p.dt, -p.u_max, p.u_max);
            int base = t * CTRL_DIM;
            h_nominal_[base + 0] = (1.0f - blend) * h_nominal_[base + 0] + blend * ux;
            h_nominal_[base + 1] = (1.0f - blend) * h_nominal_[base + 1] + blend * uy;
            sim_px += p.dt * h_nominal_[base + 0];
            sim_py += p.dt * h_nominal_[base + 1];
        }
    }

    Variant v_; PushScenario sc_; int K_, T_, seed_;
    float px_=0, py_=0, ox_=0, oy_=0;
    int steps_=0; bool reached_=false; float cum_cost_=0, min_dist_=0;
    vector<float> h_nominal_;
    float *d_start_=nullptr, *d_nominal_=nullptr, *d_costs_=nullptr, *d_weights_=nullptr, *d_perturbed_=nullptr;
    float *d_soppi_scratch_=nullptr, *d_soppi_grad_=nullptr;
    curandState* d_rng_=nullptr;
};

// ======================== Scenarios ========================
static PushScenario make_push_straight() {
    PushScenario s; s.name = "push_straight";
    s.ox0 = 1.0f; s.oy0 = 2.0f; s.px0 = 0.55f; s.py0 = 2.0f;   // pusher left of object
    s.goal_x = 3.0f; s.goal_y = 2.0f;                          // push right
    s.goal_tol = 0.20f; s.max_steps = 160;
    return s;
}
static PushScenario make_push_diagonal() {
    PushScenario s; s.name = "push_diagonal";
    s.ox0 = 1.2f; s.oy0 = 1.2f; s.px0 = 0.8f; s.py0 = 0.8f;
    s.goal_x = 3.0f; s.goal_y = 2.8f;
    s.goal_tol = 0.20f; s.max_steps = 180;
    return s;
}

// ======================== Utilities ========================
static void ensure_build_dir() { mkdir("build", 0755); }
static vector<int> parse_int_list(const string& t) {
    vector<int> v; string tok; stringstream ss(t);
    while (getline(ss, tok, ',')) if (!tok.empty()) v.push_back(max(1, atoi(tok.c_str())));
    sort(v.begin(), v.end()); v.erase(unique(v.begin(), v.end()), v.end()); return v;
}
static vector<string> parse_string_list(const string& t) {
    vector<string> v; string tok; stringstream ss(t);
    while (getline(ss, tok, ',')) if (!tok.empty()) v.push_back(tok);
    sort(v.begin(), v.end()); v.erase(unique(v.begin(), v.end()), v.end()); return v;
}
static void write_csv(const vector<EpisodeMetrics>& rows, const string& path) {
    ofstream out(path);
    out << "scenario,planner,seed,k_samples,t_horizon,grad_steps,alpha,reached_goal,collision_free,success,steps,final_distance,min_goal_distance,cumulative_cost,collisions,mean_control_delta,control_roughness,avg_control_ms,total_control_ms,episode_ms,sample_budget\n";
    for (const auto& r : rows)
        out << r.scenario<<','<<r.planner<<','<<r.seed<<','<<r.k_samples<<','<<r.t_horizon<<','
            << r.grad_steps<<','<<r.alpha<<','<<r.reached_goal<<','<<r.collision_free<<','<<r.success<<','
            << r.steps<<','<<r.final_distance<<','<<r.min_goal_distance<<','<<r.cumulative_cost<<','
            << r.collisions<<','<<r.mean_control_delta<<','<<r.control_roughness<<','
            << r.avg_control_ms<<','<<r.total_control_ms<<','<<r.episode_ms<<','<<r.sample_budget<<'\n';
}
static void print_summary(const vector<EpisodeMetrics>& rows) {
    map<string, SummaryStats> st;
    for (const auto& r : rows) {
        string key = r.scenario + " | " + r.planner + " | K=" + to_string(r.k_samples);
        auto& s = st[key]; s.episodes++; s.successes += r.success; s.steps_sum += r.steps;
        s.final_sum += r.final_distance; s.min_sum += r.min_goal_distance;
        s.cost_sum += r.cumulative_cost; s.ms_sum += r.avg_control_ms;
    }
    cout << "=== benchmark_diff_mppi_pushing summary ===" << endl;
    for (const auto& kv : st) {
        const SummaryStats& s = kv.second; float n = s.episodes;
        printf("%s : success=%.2f steps=%.1f final_dist=%.3f min_dist=%.3f cost=%.1f avg_ms=%.3f\n",
               kv.first.c_str(), s.successes/n, s.steps_sum/n, s.final_sum/n, s.min_sum/n, s.cost_sum/n, s.ms_sum/n);
    }
}

// ======================== Main ========================
int main(int argc, char** argv) {
    bool quick = false;
    string csv_path = "build/benchmark_diff_mppi_pushing.csv";
    vector<int> k_values; vector<string> scenario_names, planner_names;
    int seed_count = -1;
    float override_lp_alpha = -1.0f;
    int override_soppi_iters = -1;
    int override_soppi_neighbor_count = -1;
    float override_soppi_step_size = -1.0f;
    float override_soppi_bandwidth = -1.0f;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--quick") quick = true;
        else if (a == "--csv" && i+1<argc) csv_path = argv[++i];
        else if (a == "--k-values" && i+1<argc) k_values = parse_int_list(argv[++i]);
        else if (a == "--seed-count" && i+1<argc) seed_count = max(1, atoi(argv[++i]));
        else if (a == "--scenarios" && i+1<argc) scenario_names = parse_string_list(argv[++i]);
        else if (a == "--planners" && i+1<argc) planner_names = parse_string_list(argv[++i]);
        else if (a == "--override-lp-alpha" && i+1<argc) override_lp_alpha = atof(argv[++i]);
        else if (a == "--override-soppi-iters" && i+1<argc) override_soppi_iters = atoi(argv[++i]);
        else if (a == "--override-soppi-neighbors" && i+1<argc) override_soppi_neighbor_count = max(0, atoi(argv[++i]));
        else if (a == "--override-soppi-step-size" && i+1<argc) override_soppi_step_size = atof(argv[++i]);
        else if (a == "--override-soppi-bandwidth" && i+1<argc) override_soppi_bandwidth = atof(argv[++i]);
    }
    ensure_build_dir();

    vector<PushScenario> all_sc = { make_push_straight(), make_push_diagonal() };
    vector<PushScenario> scenarios;
    if (!scenario_names.empty()) {
        for (auto& w : scenario_names) {
            auto it = find_if(all_sc.begin(), all_sc.end(), [&](const PushScenario& s){return s.name==w;});
            if (it==all_sc.end()) { fprintf(stderr,"Unknown scenario: %s\n", w.c_str()); return 1; }
            scenarios.push_back(*it);
        }
    } else scenarios = all_sc;

    vector<Variant> variants;
    { Variant v; v.name="mppi"; variants.push_back(v); }
    { Variant v; v.name="lp_mppi"; v.use_low_pass_sampling=true; v.lp_alpha=0.35f; variants.push_back(v); }
    { Variant v; v.name="lp_mppi_smooth"; v.use_low_pass_sampling=true; v.lp_alpha=0.20f; variants.push_back(v); }
    { Variant v; v.name="oi_mppi"; v.use_object_informed=true; v.oi_ref_weight=2.0f; v.oi_obj_speed=1.8f; v.oi_seed_blend=0.15f; variants.push_back(v); }
    { Variant v; v.name="oi_lp_mppi"; v.use_object_informed=true; v.use_low_pass_sampling=true; v.lp_alpha=0.25f; v.oi_ref_weight=2.0f; v.oi_obj_speed=1.8f; v.oi_seed_blend=0.12f; variants.push_back(v); }
    { Variant v; v.name="diff_mppi_1"; v.grad_steps=1; v.alpha=0.04f; variants.push_back(v); }
    { Variant v; v.name="diff_mppi_3"; v.grad_steps=3; v.alpha=0.02f; variants.push_back(v); }
    { Variant v; v.name="soppi"; v.use_soppi_sampling=true; v.soppi_step_size=0.06f; v.soppi_bandwidth=2.0f; variants.push_back(v); }
    { Variant v; v.name="soppi_fast"; v.use_soppi_sampling=true; v.soppi_step_size=0.06f; v.soppi_bandwidth=2.0f; v.soppi_neighbor_count=32; variants.push_back(v); }
    if (!planner_names.empty()) {
        vector<Variant> f;
        for (auto& w : planner_names) {
            auto it = find_if(variants.begin(), variants.end(), [&](const Variant& v){return v.name==w;});
            if (it==variants.end()) { fprintf(stderr,"Unknown planner: %s\n", w.c_str()); return 1; }
            f.push_back(*it);
        }
        variants.swap(f);
    }
    for (auto& v : variants) {
        if (override_lp_alpha >= 0.0f && v.use_low_pass_sampling) v.lp_alpha = override_lp_alpha;
        if (override_soppi_iters >= 0 && v.use_soppi_sampling) v.soppi_svgd_iters = override_soppi_iters;
        if (override_soppi_neighbor_count >= 0 && v.use_soppi_sampling) v.soppi_neighbor_count = override_soppi_neighbor_count;
        if (override_soppi_step_size >= 0.0f && v.use_soppi_sampling) v.soppi_step_size = override_soppi_step_size;
        if (override_soppi_bandwidth >= 0.0f && v.use_soppi_sampling) v.soppi_bandwidth = override_soppi_bandwidth;
    }
    if (k_values.empty()) k_values = quick ? vector<int>{256} : vector<int>{256, 512};
    if (seed_count <= 0) seed_count = quick ? 4 : 8;

    vector<EpisodeMetrics> rows;
    for (size_t si = 0; si < scenarios.size(); si++) {
        const PushScenario& sc = scenarios[si];
        for (int ks : k_values) {
            for (size_t vi = 0; vi < variants.size(); vi++) {
                for (int seed = 0; seed < seed_count; seed++) {
                    int run_seed = (int)(5000 + si*100 + seed*7 + ks);
                    EpisodeRunner runner(variants[vi], sc, ks, DEFAULT_T, run_seed);
                    EpisodeMetrics m = runner.run();
                    rows.push_back(m);
                    printf("[%s] %s K=%d seed=%d success=%d steps=%d final=%.3f avg_ms=%.3f\n",
                           sc.name.c_str(), variants[vi].name.c_str(), ks, seed,
                           m.success, m.steps, m.final_distance, m.avg_control_ms);
                }
            }
        }
    }
    write_csv(rows, csv_path);
    print_summary(rows);
    cout << "CSV saved to " << csv_path << endl;
    return 0;
}
