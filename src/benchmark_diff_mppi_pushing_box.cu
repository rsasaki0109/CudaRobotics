/*************************************************************************
    Differentiable Box Pushing (position + ORIENTATION) — Diff-MPPI vs MPPI
    ---------------------------------------------------------------------
    Strengthened gap-#2 test. A point/disk pusher pushes a RECTANGULAR box to a
    target POSE (x, y, theta). Reaching the target orientation requires pushing
    OFF-CENTRE to generate torque — i.e. choosing the contact point. Random
    velocity sampling rarely produces the sustained off-centre contact needed to
    rotate the box precisely, whereas the autodiff gradient THROUGH the contact
    (smooth box-SDF penetration -> normal force + torque) directly informs where
    to push. Hypothesis: here the gradient should turn the modest efficiency edge
    seen on disk pushing into a SUCCESS-RATE edge on the orientation component.

    Quasi-static, smooth contact; forward-mode dual-number autodiff for the
    gradient. CSV schema matches the other Diff-MPPI benchmarks.
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

static const int STATE_DIM = 5;   // px, py, ox, oy, oth
static const int CTRL_DIM = 2;    // ux, uy
static const int DEFAULT_T = 16;

struct BoxParams {
    float dt = 0.05f;
    float u_max = 2.0f;
    float hx = 0.35f, hy = 0.18f;   // box half-extents
    float push_r = 0.08f;
    float push_gain = 11.0f;        // translation mobility
    float rot_gain = 14.0f;         // rotation mobility (1/inertia folded in)
    float pen_thresh = 0.0f;        // static-friction deadzone (off by default;
                                    // a coarse deadzone hurts fine positioning
                                    // for ALL methods, see results notes)
    // cost weights
    float w_pos = 1.5f;             // stage position
    float w_ang = 0.6f;             // stage orientation
    float w_ctrl = 0.01f;
    float w_near = 0.04f;           // pusher-near-box shaping
    float w_term_pos = 90.0f;       // terminal position
    float w_term_ang = 40.0f;       // terminal orientation
};

struct BoxScenario {
    string name;
    float px0, py0, ox0, oy0, oth0;
    float gx, gy, gth;
    float pos_tol = 0.20f, ang_tol = 0.25f;
    int max_steps = 200;
    BoxParams params;
};

struct Variant {
    string name;
    int grad_steps = 0;
    float alpha = 0.0f;
    float grad_clip = 15.0f;
    float sigma = 0.6f;
    float lambda = 5.0f;
};

struct EpisodeMetrics {
    string scenario, planner;
    int seed = 0, k_samples = 0, t_horizon = 0, grad_steps = 0;
    float alpha = 0.0f;
    int reached_goal = 0, collision_free = 1, success = 0, steps = 0;
    float final_distance = 0.0f, min_goal_distance = 0.0f, cumulative_cost = 0.0f;
    int collisions = 0;
    float avg_control_ms = 0.0f, total_control_ms = 0.0f, episode_ms = 0.0f;
    long long sample_budget = 0;
};

struct SummaryStats {
    int episodes = 0, successes = 0;
    double steps_sum = 0, final_sum = 0, min_sum = 0, cost_sum = 0, ms_sum = 0, ang_sum = 0;
};

__host__ __device__ inline float clampf_local(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
__host__ __device__ inline float wrapf(float a) { return atan2f(sinf(a), cosf(a)); }

// ===================== float dynamics (rollout / plant) =====================
__host__ __device__ inline void push_step_box_f(
    float& px, float& py, float& ox, float& oy, float& oth,
    float ux, float uy, const BoxParams& p)
{
    px += p.dt * ux; py += p.dt * uy;
    float c = cosf(oth), s = sinf(oth);
    float dx = px - ox, dy = py - oy;
    float lx =  c*dx + s*dy;          // world->box (R(-th))
    float ly = -s*dx + c*dy;
    float qx = fabsf(lx) - p.hx, qy = fabsf(ly) - p.hy;
    float rqx = fmaxf(qx, 0.0f), rqy = fmaxf(qy, 0.0f);
    float outside = sqrtf(rqx*rqx + rqy*rqy + 1e-9f);
    float inside = fminf(fmaxf(qx, qy), 0.0f);
    float sd = outside + inside;                 // box signed distance of pusher
    float pen = fmaxf(p.push_r - sd, 0.0f);       // smooth contact penetration
    float peff = fmaxf(pen - p.pen_thresh, 0.0f); // friction deadzone
    float nlx = rqx * (lx >= 0 ? 1.0f : -1.0f);
    float nly = rqy * (ly >= 0 ? 1.0f : -1.0f);
    float nlen = sqrtf(nlx*nlx + nly*nly + 1e-9f);
    nlx /= nlen; nly /= nlen;                     // outward normal (box frame)
    float nwx = c*nlx - s*nly;                    // box->world
    float nwy = s*nlx + c*nly;
    float Fx = -nwx * p.push_gain * peff;         // push box away from pusher
    float Fy = -nwy * p.push_gain * peff;
    float cxw = px - nwx * sd, cyw = py - nwy * sd;   // contact point (world)
    float rx = cxw - ox, ry = cyw - oy;
    float torque = rx*Fy - ry*Fx;
    ox += p.dt * Fx; oy += p.dt * Fy;
    oth = wrapf(oth + p.dt * p.rot_gain * torque);
}

__host__ __device__ inline float stage_cost_box_f(
    float px, float py, float ox, float oy, float oth, float ux, float uy,
    float gx, float gy, float gth, const BoxParams& p)
{
    float dpx = ox - gx, dpy = oy - gy;
    float dth = wrapf(oth - gth);
    float c = p.w_pos * (dpx*dpx + dpy*dpy) * p.dt;
    c += p.w_ang * (dth*dth) * p.dt;
    c += p.w_ctrl * (ux*ux + uy*uy) * p.dt;
    float ex = px - ox, ey = py - oy;
    c += p.w_near * (ex*ex + ey*ey) * p.dt;
    return c;
}

__host__ __device__ inline float terminal_cost_box_f(
    float ox, float oy, float oth, float gx, float gy, float gth, const BoxParams& p)
{
    float dpx = ox - gx, dpy = oy - gy, dth = wrapf(oth - gth);
    return p.w_term_pos * (dpx*dpx + dpy*dpy) + p.w_term_ang * (dth*dth);
}

// ===================== Dualf helpers + dynamics (gradient) =====================
__device__ inline Dualf d_abs(const Dualf& x) { return x.val >= 0.0f ? x : (Dualf::constant(0.0f) - x); }
__device__ inline Dualf d_relu(const Dualf& x) { return x.val > 0.0f ? x : Dualf::constant(0.0f); }
__device__ inline Dualf d_max(const Dualf& a, const Dualf& b) { return a.val >= b.val ? a : b; }
__device__ inline Dualf d_min0(const Dualf& x) { return x.val < 0.0f ? x : Dualf::constant(0.0f); }

// Forward-mode derivative of total rollout cost w.r.t. nominal control `active`.
__device__ inline float dcost_dparam_box(
    const float start[STATE_DIM], const float* nominal, int T, int active,
    float gx, float gy, float gth, const BoxParams& p)
{
    Dualf px = Dualf::constant(start[0]), py = Dualf::constant(start[1]);
    Dualf ox = Dualf::constant(start[2]), oy = Dualf::constant(start[3]);
    Dualf oth = Dualf::constant(start[4]);
    Dualf cost = Dualf::constant(0.0f);
    for (int t = 0; t < T; t++) {
        Dualf ux = (active == t*2+0) ? Dualf::variable(nominal[t*2+0]) : Dualf::constant(nominal[t*2+0]);
        Dualf uy = (active == t*2+1) ? Dualf::variable(nominal[t*2+1]) : Dualf::constant(nominal[t*2+1]);
        ux = clamp(ux, -p.u_max, p.u_max);
        uy = clamp(uy, -p.u_max, p.u_max);
        // dynamics
        px = px + Dualf::constant(p.dt) * ux;
        py = py + Dualf::constant(p.dt) * uy;
        Dualf c = cudabot::cos(oth), s = cudabot::sin(oth);
        Dualf dx = px - ox, dy = py - oy;
        Dualf lx = c*dx + s*dy;
        Dualf ly = (Dualf::constant(0.0f) - s)*dx + c*dy;
        Dualf qx = d_abs(lx) - Dualf::constant(p.hx);
        Dualf qy = d_abs(ly) - Dualf::constant(p.hy);
        Dualf rqx = d_relu(qx), rqy = d_relu(qy);
        Dualf outside = cudabot::sqrt(rqx*rqx + rqy*rqy + Dualf::constant(1e-9f));
        Dualf inside = d_min0(d_max(qx, qy));
        Dualf sd = outside + inside;
        Dualf pen = d_relu(Dualf::constant(p.push_r) - sd);
        Dualf peff = d_relu(pen - Dualf::constant(p.pen_thresh));   // friction deadzone
        Dualf nlx = rqx * Dualf::constant(lx.val >= 0.0f ? 1.0f : -1.0f);
        Dualf nly = rqy * Dualf::constant(ly.val >= 0.0f ? 1.0f : -1.0f);
        Dualf nlen = cudabot::sqrt(nlx*nlx + nly*nly + Dualf::constant(1e-9f));
        nlx = nlx / nlen; nly = nly / nlen;
        Dualf nwx = c*nlx - s*nly;
        Dualf nwy = s*nlx + c*nly;
        Dualf Fx = (Dualf::constant(0.0f) - nwx) * Dualf::constant(p.push_gain) * peff;
        Dualf Fy = (Dualf::constant(0.0f) - nwy) * Dualf::constant(p.push_gain) * peff;
        Dualf cxw = px - nwx*sd, cyw = py - nwy*sd;
        Dualf rx = cxw - ox, ry = cyw - oy;
        Dualf torque = rx*Fy - ry*Fx;
        ox = ox + Dualf::constant(p.dt) * Fx;
        oy = oy + Dualf::constant(p.dt) * Fy;
        oth = oth + Dualf::constant(p.dt * p.rot_gain) * torque;   // no wrap in grad rollout
        // stage cost
        Dualf dpx = ox - Dualf::constant(gx), dpy = oy - Dualf::constant(gy);
        Dualf dthr = oth - Dualf::constant(gth);
        Dualf dth = cudabot::atan2(cudabot::sin(dthr), cudabot::cos(dthr));
        cost = cost + Dualf::constant(p.w_pos) * (dpx*dpx + dpy*dpy) * Dualf::constant(p.dt);
        cost = cost + Dualf::constant(p.w_ang) * (dth*dth) * Dualf::constant(p.dt);
        cost = cost + Dualf::constant(p.w_ctrl) * (ux*ux + uy*uy) * Dualf::constant(p.dt);
        Dualf ex = px - ox, ey = py - oy;
        cost = cost + Dualf::constant(p.w_near) * (ex*ex + ey*ey) * Dualf::constant(p.dt);
    }
    Dualf dpx = ox - Dualf::constant(gx), dpy = oy - Dualf::constant(gy);
    Dualf dthr = oth - Dualf::constant(gth);
    Dualf dth = cudabot::atan2(cudabot::sin(dthr), cudabot::cos(dthr));
    cost = cost + Dualf::constant(p.w_term_pos) * (dpx*dpx + dpy*dpy);
    cost = cost + Dualf::constant(p.w_term_ang) * (dth*dth);
    return cost.deriv;
}

// ======================== Kernels ========================
__global__ void init_curand_kernel(curandState* st, int n, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &st[i]);
}

__global__ void rollout_kernel(
    const float* d_start, const float* d_nominal, float* d_costs, float* d_perturbed,
    curandState* d_rng, BoxParams p, float gx, float gy, float gth, int K, int T, float sigma)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curandState rng = d_rng[k];
    float px=d_start[0], py=d_start[1], ox=d_start[2], oy=d_start[3], oth=d_start[4];
    float cost = 0.0f;
    for (int t = 0; t < T; t++) {
        float ux = d_nominal[t*2+0] + curand_normal(&rng) * sigma;
        float uy = d_nominal[t*2+1] + curand_normal(&rng) * sigma;
        ux = clampf_local(ux, -p.u_max, p.u_max);
        uy = clampf_local(uy, -p.u_max, p.u_max);
        d_perturbed[k*T*2 + t*2 + 0] = ux;
        d_perturbed[k*T*2 + t*2 + 1] = uy;
        push_step_box_f(px, py, ox, oy, oth, ux, uy, p);
        cost += stage_cost_box_f(px, py, ox, oy, oth, ux, uy, gx, gy, gth, p);
    }
    cost += terminal_cost_box_f(ox, oy, oth, gx, gy, gth, p);
    d_costs[k] = cost;
    d_rng[k] = rng;
}

// Mechanism diagnostic: replay the SAME sampled controls written by rollout_kernel
// (d_perturbed) and measure, per sample, how much rotation the box actually
// underwent. d_rot = total angular path Sum|dtheta| (did this sample engage
// torque-generating off-centre contact at all), d_netrot = net signed rotation.
// Pairs 1:1 with d_costs from the same rollout, so cost-vs-rotation is exact.
__global__ void replay_rot_kernel(
    const float* d_start, const float* d_perturbed, float* d_rot, float* d_netrot,
    BoxParams p, int K, int T)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    float px=d_start[0], py=d_start[1], ox=d_start[2], oy=d_start[3], oth=d_start[4];
    float oth0 = oth, path = 0.0f;
    for (int t = 0; t < T; t++) {
        float ux = d_perturbed[k*T*2 + t*2 + 0];
        float uy = d_perturbed[k*T*2 + t*2 + 1];
        float oth_prev = oth;
        push_step_box_f(px, py, ox, oy, oth, ux, uy, p);
        path += fabsf(wrapf(oth - oth_prev));
    }
    d_rot[k] = path;
    d_netrot[k] = wrapf(oth - oth0);
}

// Host cost of an UNPERTURBED nominal rollout (the current mean the sampler
// perturbs around): the baseline against which "did a sample improve?" is judged.
// Mirrors rollout_kernel exactly with zero noise, using the controller model.
static float host_rollout_cost(const float start[STATE_DIM], const vector<float>& nominal,
                               int T, float gx, float gy, float gth, const BoxParams& p) {
    float px=start[0], py=start[1], ox=start[2], oy=start[3], oth=start[4];
    float cost = 0.0f;
    for (int t = 0; t < T; t++) {
        float ux = clampf_local(nominal[t*2+0], -p.u_max, p.u_max);
        float uy = clampf_local(nominal[t*2+1], -p.u_max, p.u_max);
        push_step_box_f(px, py, ox, oy, oth, ux, uy, p);
        cost += stage_cost_box_f(px, py, ox, oy, oth, ux, uy, gx, gy, gth, p);
    }
    cost += terminal_cost_box_f(ox, oy, oth, gx, gy, gth, p);
    return cost;
}

__global__ void weights_kernel(const float* d_costs, float* d_w, int K, float lambda) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float cmin = FLT_MAX;
    for (int k = 0; k < K; k++) cmin = fminf(cmin, d_costs[k]);
    float sum = 0.0f;
    for (int k = 0; k < K; k++) { float w = expf(-(d_costs[k]-cmin)/lambda); d_w[k]=w; sum+=w; }
    if (sum > 0.0f) for (int k = 0; k < K; k++) d_w[k] /= sum;
}

__global__ void update_kernel(float* d_nominal, const float* d_perturbed, const float* d_w, int K, int T) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float ux=0.0f, uy=0.0f;
    for (int k = 0; k < K; k++) { float w=d_w[k]; ux+=w*d_perturbed[k*T*2+t*2+0]; uy+=w*d_perturbed[k*T*2+t*2+1]; }
    d_nominal[t*2+0]=ux; d_nominal[t*2+1]=uy;
}

__global__ void grad_step_kernel(
    const float* d_start, float* d_nominal, BoxParams p, float gx, float gy, float gth,
    int T, float alpha, float grad_clip)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float start[STATE_DIM] = { d_start[0], d_start[1], d_start[2], d_start[3], d_start[4] };
    float grad[2*32];
    float gnorm2 = 0.0f;
    for (int kp = 0; kp < 2*T; kp++) {
        float g = dcost_dparam_box(start, d_nominal, T, kp, gx, gy, gth, p);
        grad[kp] = g; gnorm2 += g*g;
    }
    float scale = alpha, gnorm = sqrtf(gnorm2);
    if (grad_clip > 0.0f && gnorm > grad_clip) scale = alpha * grad_clip / gnorm;
    for (int kp = 0; kp < 2*T; kp++)
        d_nominal[kp] = clampf_local(d_nominal[kp] - scale*grad[kp], -p.u_max, p.u_max);
}

// ======================== Episode Runner ========================
class EpisodeRunner {
public:
    bool record_traj = false;            // when set, log per-step pose to traj_flat
    vector<float> traj_flat;             // [px,py,ox,oy,oth] per recorded step
    // Mechanism diagnostic (vanilla MPPI only): per control step, measure what
    // fraction of the K sampled rollouts engage torque-generating rotation and
    // improve on the current mean. Quantifies "random sampling rarely produces the
    // sustained off-centre contact needed to rotate the box". diag_flat stride 9:
    // [step, rem_ang, nominal_cost, min_cost, improve_frac, contact_frac, rot_mean,
    //  rot_p90, escape_frac]. escape_frac = fraction of K samples whose NET rotation
    // is toward the goal angle by enough to break the tolerance latch this step --
    // the sharp mechanism number (near 0 at the rotation plateau where sampling stalls).
    bool record_diag = false;
    int  diag_dump_step = -1;            // step at which to dump per-sample (cost,rot,netrot)
    vector<float> diag_flat;             // stride 9 (see above)
    vector<float> diag_scatter;          // stride 3: [cost, rot, netrot] at diag_dump_step
    // Model-mismatch robustness knob (default 1.0 => no-op, published numbers
    // reproduce byte-identically). When != 1.0, the TRUE plant's contact mobility
    // (push_gain, rot_gain) is scaled while the controller's internal rollout +
    // autodiff gradient keep the NOMINAL gains: the controller's contact model is
    // deliberately WRONG. This stress-tests whether the contact-gradient win
    // survives sim-to-real-style contact-model error.
    float plant_gain_scale = 1.0f;
    // Geometry-mismatch robustness knob (default 1.0 => no-op). When != 1.0, the
    // TRUE plant box half-extents (hx, hy) are scaled by this factor while the
    // controller's rollout + autodiff gradient keep the NOMINAL box size: the
    // controller has the wrong object dimensions. This is an axis distinct from
    // plant_gain_scale (object SHAPE error vs contact MOBILITY error) and a real
    // sim-to-real concern (exact object dimensions are rarely known).
    float plant_size_scale = 1.0f;

    EpisodeRunner(const Variant& v, const BoxScenario& sc, int K, int T, int seed)
        : v_(v), sc_(sc), K_(K), T_(T), seed_(seed) {
        h_nominal_.assign(T_*CTRL_DIM, 0.0f);
        CUDA_CHECK(cudaMalloc(&d_start_, STATE_DIM*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nominal_, T_*CTRL_DIM*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs_, K_*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_, K_*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perturbed_, K_*T_*CTRL_DIM*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_, K_*sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&d_rot_, K_*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_netrot_, K_*sizeof(float)));
        reset_rng();
    }
    ~EpisodeRunner() {
        cudaFree(d_start_); cudaFree(d_nominal_); cudaFree(d_costs_);
        cudaFree(d_weights_); cudaFree(d_perturbed_); cudaFree(d_rng_);
        cudaFree(d_rot_); cudaFree(d_netrot_);
    }

    EpisodeMetrics run() {
        reset_state();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        warmup();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        reset_rng();

        // True plant params: contact mobility scaled by plant_gain_scale (the
        // controller's model, used in rollout/grad below, keeps sc_.params).
        BoxParams plant_p = sc_.params;
        plant_p.push_gain *= plant_gain_scale;
        plant_p.rot_gain  *= plant_gain_scale;
        plant_p.hx        *= plant_size_scale;   // true object SHAPE (controller keeps nominal)
        plant_p.hy        *= plant_size_scale;

        auto ep0 = chrono::steady_clock::now();
        float ctrl_ms = 0.0f;
        if (record_traj) { traj_flat.clear();
            traj_flat.insert(traj_flat.end(), {px_,py_,ox_,oy_,oth_}); }
        for (int step = 0; step < sc_.max_steps; step++) {
            float pd = pos_dist(), ad = ang_err();
            min_dist_ = fminf(min_dist_, pd);
            if (pd < sc_.pos_tol && ad < sc_.ang_tol) { reached_ = true; steps_ = step; break; }

            auto t0 = chrono::steady_clock::now();
            controller_update();
            auto t1 = chrono::steady_clock::now();
            ctrl_ms += chrono::duration<float, milli>(t1 - t0).count();

            // h_nominal_ still holds the PRE-update mean (the sampler perturbed
            // around it); d_costs_/d_perturbed_ hold this step's K samples. Collect
            // the mechanism diagnostic before h_nominal_ is overwritten below.
            if (record_diag) collect_diag(step);

            CUDA_CHECK(cudaMemcpy(h_nominal_.data(), d_nominal_, h_nominal_.size()*sizeof(float), cudaMemcpyDeviceToHost));
            push_step_box_f(px_, py_, ox_, oy_, oth_, h_nominal_[0], h_nominal_[1], plant_p);
            cum_cost_ += stage_cost_box_f(px_, py_, ox_, oy_, oth_, h_nominal_[0], h_nominal_[1], sc_.gx, sc_.gy, sc_.gth, sc_.params);
            for (int t = 0; t < T_-1; t++) { h_nominal_[t*2+0]=h_nominal_[(t+1)*2+0]; h_nominal_[t*2+1]=h_nominal_[(t+1)*2+1]; }
            h_nominal_[(T_-1)*2+0]=0.0f; h_nominal_[(T_-1)*2+1]=0.0f;
            steps_ = step + 1;
            if (record_traj) traj_flat.insert(traj_flat.end(), {px_,py_,ox_,oy_,oth_});
        }
        auto ep1 = chrono::steady_clock::now();

        EpisodeMetrics m;
        m.scenario=sc_.name; m.planner=v_.name; m.seed=seed_;
        m.k_samples=K_; m.t_horizon=T_; m.grad_steps=v_.grad_steps; m.alpha=v_.alpha;
        float pd = pos_dist(), ad = ang_err();
        if (pd < sc_.pos_tol && ad < sc_.ang_tol) reached_ = true;
        m.reached_goal = reached_?1:0; m.success = reached_?1:0;
        m.steps=steps_; m.final_distance=pd; m.min_goal_distance=ad;  // store final ang err in min col
        m.cumulative_cost=cum_cost_;
        m.total_control_ms=ctrl_ms; m.avg_control_ms = steps_>0? ctrl_ms/steps_ : 0.0f;
        m.episode_ms = chrono::duration<float, milli>(ep1 - ep0).count();
        m.sample_budget = (long long)steps_ * K_ * T_;
        return m;
    }

private:
    void reset_rng() { int b=256; init_curand_kernel<<<(K_+b-1)/b,b>>>(d_rng_, K_, (unsigned long long)seed_); CUDA_CHECK(cudaDeviceSynchronize()); }
    void reset_state() { px_=sc_.px0; py_=sc_.py0; ox_=sc_.ox0; oy_=sc_.oy0; oth_=sc_.oth0; steps_=0; reached_=false; cum_cost_=0; min_dist_=pos_dist(); }
    float pos_dist() const { float dx=ox_-sc_.gx, dy=oy_-sc_.gy; return sqrtf(dx*dx+dy*dy); }
    float ang_err() const { return fabsf(wrapf(oth_ - sc_.gth)); }
    void sync_start() { float s[STATE_DIM]={px_,py_,ox_,oy_,oth_}; CUDA_CHECK(cudaMemcpy(d_start_, s, STATE_DIM*sizeof(float), cudaMemcpyHostToDevice)); }
    void controller_update() {
        sync_start();
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size()*sizeof(float), cudaMemcpyHostToDevice));
        int b=256;
        rollout_kernel<<<(K_+b-1)/b,b>>>(d_start_, d_nominal_, d_costs_, d_perturbed_, d_rng_, sc_.params, sc_.gx, sc_.gy, sc_.gth, K_, T_, v_.sigma);
        weights_kernel<<<1,1>>>(d_costs_, d_weights_, K_, v_.lambda);
        update_kernel<<<(T_+b-1)/b,b>>>(d_nominal_, d_perturbed_, d_weights_, K_, T_);
        for (int g = 0; g < v_.grad_steps; g++)
            grad_step_kernel<<<1,1>>>(d_start_, d_nominal_, sc_.params, sc_.gx, sc_.gy, sc_.gth, T_, v_.alpha, v_.grad_clip);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    void warmup() { for (int i = 0; i < 3; i++) controller_update(); }

    // Measure the K-sample statistics at the current decision state. Reuses this
    // step's rollout (d_costs_ from controller_update, d_perturbed_) and replays it
    // for rotation; h_nominal_ is still the pre-update mean (the sampling centre).
    void collect_diag(int step) {
        int b = 256;
        replay_rot_kernel<<<(K_+b-1)/b,b>>>(d_start_, d_perturbed_, d_rot_, d_netrot_, sc_.params, K_, T_);
        CUDA_CHECK(cudaDeviceSynchronize());
        float start[STATE_DIM] = { px_, py_, ox_, oy_, oth_ };
        float nominal_cost = host_rollout_cost(start, h_nominal_, T_, sc_.gx, sc_.gy, sc_.gth, sc_.params);
        vector<float> hc(K_), hr(K_), hn(K_);
        CUDA_CHECK(cudaMemcpy(hc.data(), d_costs_,  K_*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hr.data(), d_rot_,    K_*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hn.data(), d_netrot_, K_*sizeof(float), cudaMemcpyDeviceToHost));
        const float rot_thresh = 0.02f;   // rad of total angular path => "engaged rotation"
        // Signed rotation still needed to reach the goal angle, and how much beyond
        // tolerance: a sample "escapes" the latch if it rotates the right way by >= margin.
        float need = wrapf(sc_.gth - oth_);
        float margin = fmaxf(0.0f, fabsf(need) - sc_.ang_tol);
        int n_improve = 0, n_contact = 0, n_escape = 0; float min_cost = FLT_MAX, rot_sum = 0.0f;
        for (int k = 0; k < K_; k++) {
            if (hc[k] < nominal_cost) n_improve++;
            if (hr[k] > rot_thresh)   n_contact++;
            if (margin > 0.0f && need*hn[k] > 0.0f && fabsf(hn[k]) >= margin) n_escape++;
            min_cost = fminf(min_cost, hc[k]);
            rot_sum += hr[k];
        }
        vector<float> rs = hr; sort(rs.begin(), rs.end());
        float rot_p90 = rs[(int)(0.9f*(K_-1))];
        float improve_frac = (float)n_improve / K_;
        float contact_frac = (float)n_contact / K_;
        float escape_frac = (float)n_escape / K_;
        float rot_mean = rot_sum / K_;
        float rem_ang = ang_err();
        diag_flat.insert(diag_flat.end(),
            { (float)step, rem_ang, nominal_cost, min_cost, improve_frac, contact_frac, rot_mean, rot_p90, escape_frac });
        if (step == diag_dump_step)
            for (int k = 0; k < K_; k++) diag_scatter.insert(diag_scatter.end(), { hc[k], hr[k], hn[k] });
    }

    Variant v_; BoxScenario sc_; int K_, T_, seed_;
    float px_=0,py_=0,ox_=0,oy_=0,oth_=0;
    int steps_=0; bool reached_=false; float cum_cost_=0, min_dist_=0;
    vector<float> h_nominal_;
    float *d_start_=nullptr,*d_nominal_=nullptr,*d_costs_=nullptr,*d_weights_=nullptr,*d_perturbed_=nullptr;
    float *d_rot_=nullptr,*d_netrot_=nullptr;
    curandState* d_rng_=nullptr;
};

// ======================== Scenarios ========================
static BoxScenario make_box_turn() {
    BoxScenario s; s.name="box_turn";
    s.ox0=1.2f; s.oy0=2.0f; s.oth0=0.0f; s.px0=0.6f; s.py0=2.0f;
    s.gx=2.1f; s.gy=2.0f; s.gth=0.9f;             // translate + rotate
    s.pos_tol=0.20f; s.ang_tol=0.25f; s.max_steps=260;
    return s;
}
static BoxScenario make_box_align() {
    BoxScenario s; s.name="box_align";
    s.ox0=1.5f; s.oy0=1.5f; s.oth0=0.0f; s.px0=1.5f; s.py0=0.9f;
    s.gx=1.7f; s.gy=2.4f; s.gth=-0.7f;            // mostly rotate + small move
    s.pos_tol=0.22f; s.ang_tol=0.25f; s.max_steps=240;
    return s;
}
// Second orientation-dominant task, structurally distinct from box_align:
// opposite handedness (+1.0 rad), pusher engages the LEFT face (push +x) instead
// of the bottom, and a TIGHTER angular tolerance. Tight ang_tol is precisely the
// regime where pure sampling plateaus and the contact gradient closes the last bit,
// so this re-tests the monotone-in-gradient-steps signature on different contact
// mechanics — a guard against box_align being a single lucky scenario.
static BoxScenario make_box_pivot() {
    BoxScenario s; s.name="box_pivot";
    s.ox0=1.6f; s.oy0=1.8f; s.oth0=0.0f; s.px0=0.95f; s.py0=1.55f;
    s.gx=2.0f; s.gy=1.8f; s.gth=0.70f;            // rotate +0.7, small +x translate
    s.pos_tol=0.22f; s.ang_tol=0.11f; s.max_steps=240;
    return s;
}

// ======================== Utilities ========================
static void ensure_build_dir() { mkdir("build", 0755); }
static vector<int> parse_int_list(const string& t){ vector<int> v; string tok; stringstream ss(t); while(getline(ss,tok,',')) if(!tok.empty()) v.push_back(max(1,atoi(tok.c_str()))); sort(v.begin(),v.end()); v.erase(unique(v.begin(),v.end()),v.end()); return v; }
static vector<string> parse_string_list(const string& t){ vector<string> v; string tok; stringstream ss(t); while(getline(ss,tok,',')) if(!tok.empty()) v.push_back(tok); sort(v.begin(),v.end()); v.erase(unique(v.begin(),v.end()),v.end()); return v; }
static void write_csv(const vector<EpisodeMetrics>& rows, const string& path) {
    ofstream out(path);
    out << "scenario,planner,seed,k_samples,t_horizon,grad_steps,alpha,reached_goal,collision_free,success,steps,final_distance,min_goal_distance,cumulative_cost,collisions,avg_control_ms,total_control_ms,episode_ms,sample_budget\n";
    for (const auto& r : rows)
        out << r.scenario<<','<<r.planner<<','<<r.seed<<','<<r.k_samples<<','<<r.t_horizon<<','<<r.grad_steps<<','<<r.alpha<<','
            << r.reached_goal<<','<<r.collision_free<<','<<r.success<<','<<r.steps<<','<<r.final_distance<<','<<r.min_goal_distance<<','
            << r.cumulative_cost<<','<<r.collisions<<','<<r.avg_control_ms<<','<<r.total_control_ms<<','<<r.episode_ms<<','<<r.sample_budget<<'\n';
}
static void print_summary(const vector<EpisodeMetrics>& rows) {
    map<string, SummaryStats> st;
    for (const auto& r : rows) {
        string key = r.scenario + " | " + r.planner + " | K=" + to_string(r.k_samples);
        auto& s = st[key]; s.episodes++; s.successes+=r.success; s.steps_sum+=r.steps;
        s.final_sum+=r.final_distance; s.ang_sum+=r.min_goal_distance; s.cost_sum+=r.cumulative_cost; s.ms_sum+=r.avg_control_ms;
    }
    cout << "=== benchmark_diff_mppi_pushing_box summary ===" << endl;
    for (const auto& kv : st) {
        const SummaryStats& s = kv.second; float n=s.episodes;
        printf("%s : success=%.2f steps=%.1f pos_err=%.3f ang_err=%.3f cost=%.1f avg_ms=%.3f\n",
               kv.first.c_str(), s.successes/n, s.steps_sum/n, s.final_sum/n, s.ang_sum/n, s.cost_sum/n, s.ms_sum/n);
    }
}

// ======================== Main ========================
int main(int argc, char** argv) {
    bool quick=false; string csv_path="build/benchmark_diff_mppi_pushing_box.csv";
    vector<int> k_values; vector<string> scenario_names, planner_names; int seed_count=-1;
    int horizon=DEFAULT_T; string dump_traj_prefix=""; float plant_gain_scale=1.0f; float plant_size_scale=1.0f;
    string diag_prefix="";
    for (int i=1;i<argc;i++){ string a=argv[i];
        if (a=="--quick") quick=true;
        else if (a=="--csv"&&i+1<argc) csv_path=argv[++i];
        else if (a=="--k-values"&&i+1<argc) k_values=parse_int_list(argv[++i]);
        else if (a=="--seed-count"&&i+1<argc) seed_count=max(1,atoi(argv[++i]));
        else if (a=="--scenarios"&&i+1<argc) scenario_names=parse_string_list(argv[++i]);
        else if (a=="--planners"&&i+1<argc) planner_names=parse_string_list(argv[++i]);
        else if (a=="--horizon"&&i+1<argc) horizon=max(2,atoi(argv[++i]));
        else if (a=="--dump-traj"&&i+1<argc) dump_traj_prefix=argv[++i];
        // mechanism diagnostic: per-step K-sample rotation/improvement statistics
        // for vanilla MPPI across tasks (quantifies where sampling is contact-starved).
        else if (a=="--diag-mechanism"&&i+1<argc) diag_prefix=argv[++i];
        // model-mismatch robustness: scale the TRUE plant contact mobility while the
        // controller keeps nominal gains (default 1.0 => published numbers reproduce).
        else if (a=="--plant-gain-scale"&&i+1<argc) plant_gain_scale=(float)atof(argv[++i]);
        // true plant box-size scale vs the controller's nominal size (default 1 => no-op).
        else if (a=="--plant-size-scale"&&i+1<argc) plant_size_scale=(float)atof(argv[++i]);
    }
    ensure_build_dir();

    // Trajectory-dump mode: write per-step box poses for the figure filmstrip.
    if (!dump_traj_prefix.empty()) {
        BoxScenario sc = make_box_align();
        struct { string name; int grad; float alpha; } sel[] = {
            {"mppi", 0, 0.0f}, {"diff_mppi_5", 5, 0.008f} };
        for (auto& s : sel) {
            Variant v; v.name=s.name; v.grad_steps=s.grad; v.alpha=s.alpha;
            EpisodeRunner runner(v, sc, 1024, horizon, /*seed=*/6000+1*100+0*20+0*7+1024);
            runner.record_traj = true;
            EpisodeMetrics m = runner.run();
            string path = dump_traj_prefix + "_" + s.name + ".csv";
            ofstream out(path);
            out << "# scenario=" << sc.name << " goal=" << sc.gx << "," << sc.gy << "," << sc.gth
                << " hx=" << sc.params.hx << " hy=" << sc.params.hy
                << " success=" << m.success << " steps=" << m.steps << "\n";
            out << "px,py,ox,oy,oth\n";
            for (size_t k=0; k+4 < runner.traj_flat.size(); k+=5)
                out << runner.traj_flat[k] << "," << runner.traj_flat[k+1] << ","
                    << runner.traj_flat[k+2] << "," << runner.traj_flat[k+3] << ","
                    << runner.traj_flat[k+4] << "\n";
            printf("[dump-traj] %s -> %s (success=%d steps=%d)\n",
                   s.name.c_str(), path.c_str(), m.success, m.steps);
        }
        return 0;
    }

    // Mechanism-diagnostic mode: run vanilla MPPI (the sampler under study) at a
    // large budget on each task and log, per control step, what fraction of the K
    // sampled rollouts engage torque-generating rotation and improve on the current
    // mean. Tests the thesis that sampling is contact-starved precisely on the
    // rotation-dominant tasks where the gradient wins, and not on box_turn.
    if (!diag_prefix.empty()) {
        int Kdiag = k_values.empty() ? 4096 : k_values.back();
        vector<BoxScenario> diag_sc = { make_box_turn(), make_box_align(), make_box_pivot() };
        if (!scenario_names.empty()) {
            vector<BoxScenario> f;
            for (auto& w : scenario_names) { auto it=find_if(diag_sc.begin(),diag_sc.end(),[&](const BoxScenario&s){return s.name==w;});
                if (it!=diag_sc.end()) f.push_back(*it); }
            if (!f.empty()) diag_sc.swap(f);
        }
        printf("=== mechanism diagnostic: vanilla MPPI, K=%d, per-step sample statistics ===\n", Kdiag);
        for (auto& sc : diag_sc) {
            Variant v; v.name="mppi"; v.grad_steps=0; v.alpha=0.0f;   // the sampler under study
            EpisodeRunner runner(v, sc, Kdiag, horizon, /*seed=*/6000+0*100+0*20+0*7+Kdiag);
            runner.record_diag = true; runner.diag_dump_step = 60;
            EpisodeMetrics m = runner.run();
            string path = diag_prefix + "_" + sc.name + ".csv";
            ofstream out(path);
            out << "# scenario=" << sc.name << " K=" << Kdiag << " ang_tol=" << sc.ang_tol
                << " success=" << m.success << " steps=" << m.steps << "\n";
            out << "step,rem_ang,nominal_cost,min_cost,improve_frac,contact_frac,rot_mean,rot_p90,escape_frac\n";
            double sif=0, scf=0, sef=0; int nrows=0;
            for (size_t k=0; k+8 < runner.diag_flat.size(); k+=9) {
                for (int j=0;j<9;j++) out << (j?",":"") << runner.diag_flat[k+j];
                out << "\n";
                sif += runner.diag_flat[k+4]; scf += runner.diag_flat[k+5]; sef += runner.diag_flat[k+8]; nrows++;
            }
            // per-sample scatter (cost vs rotation) at diag_dump_step
            string spath = diag_prefix + "_" + sc.name + "_scatter.csv";
            ofstream sout(spath);
            sout << "# scenario=" << sc.name << " step=" << runner.diag_dump_step << " K=" << Kdiag << "\n";
            sout << "cost,rot,netrot\n";
            for (size_t k=0; k+2 < runner.diag_scatter.size(); k+=3)
                sout << runner.diag_scatter[k] << "," << runner.diag_scatter[k+1] << "," << runner.diag_scatter[k+2] << "\n";
            printf("[diag] %-10s success=%d steps=%d  improve_frac=%.3f  contact_frac=%.3f  escape_frac=%.4f -> %s\n",
                   sc.name.c_str(), m.success, m.steps, nrows?sif/nrows:0.0, nrows?scf/nrows:0.0, nrows?sef/nrows:0.0, path.c_str());
        }
        return 0;
    }

    vector<BoxScenario> all_sc = { make_box_turn(), make_box_align(), make_box_pivot() };
    vector<BoxScenario> scenarios;
    if (!scenario_names.empty()) {
        for (auto& w : scenario_names) { auto it=find_if(all_sc.begin(),all_sc.end(),[&](const BoxScenario&s){return s.name==w;});
            if (it==all_sc.end()){fprintf(stderr,"Unknown scenario: %s\n",w.c_str());return 1;} scenarios.push_back(*it); }
    } else scenarios = all_sc;

    vector<Variant> variants;
    { Variant v; v.name="mppi"; variants.push_back(v); }
    { Variant v; v.name="diff_mppi_1"; v.grad_steps=1; v.alpha=0.02f; variants.push_back(v); }
    { Variant v; v.name="diff_mppi_3"; v.grad_steps=3; v.alpha=0.010f; variants.push_back(v); }
    { Variant v; v.name="diff_mppi_5"; v.grad_steps=5; v.alpha=0.008f; variants.push_back(v); }
    if (!planner_names.empty()) {
        vector<Variant> f; for (auto& w : planner_names){ auto it=find_if(variants.begin(),variants.end(),[&](const Variant&v){return v.name==w;});
            if (it==variants.end()){fprintf(stderr,"Unknown planner: %s\n",w.c_str());return 1;} f.push_back(*it);} variants.swap(f);
    }
    if (k_values.empty()) k_values = quick ? vector<int>{256} : vector<int>{256, 1024};
    if (seed_count<=0) seed_count = quick ? 4 : 8;

    vector<EpisodeMetrics> rows;
    for (size_t si=0; si<scenarios.size(); si++) {
        const BoxScenario& sc = scenarios[si];
        for (int ks : k_values) for (size_t vi=0; vi<variants.size(); vi++) for (int seed=0; seed<seed_count; seed++) {
            int run_seed = (int)(6000 + si*100 + vi*20 + seed*7 + ks);
            EpisodeRunner runner(variants[vi], sc, ks, horizon, run_seed);
            runner.plant_gain_scale = plant_gain_scale;
            runner.plant_size_scale = plant_size_scale;
            EpisodeMetrics m = runner.run();
            rows.push_back(m);
            printf("[%s] %s K=%d seed=%d success=%d steps=%d pos=%.3f ang=%.3f avg_ms=%.3f\n",
                   sc.name.c_str(), variants[vi].name.c_str(), ks, seed, m.success, m.steps, m.final_distance, m.min_goal_distance, m.avg_control_ms);
        }
    }
    write_csv(rows, csv_path);
    print_summary(rows);
    cout << "CSV saved to " << csv_path << endl;
    return 0;
}
