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
    float nlx = rqx * (lx >= 0 ? 1.0f : -1.0f);
    float nly = rqy * (ly >= 0 ? 1.0f : -1.0f);
    float nlen = sqrtf(nlx*nlx + nly*nly + 1e-9f);
    nlx /= nlen; nly /= nlen;                     // outward normal (box frame)
    float nwx = c*nlx - s*nly;                    // box->world
    float nwy = s*nlx + c*nly;
    float Fx = -nwx * p.push_gain * pen;          // push box away from pusher
    float Fy = -nwy * p.push_gain * pen;
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
        Dualf nlx = rqx * Dualf::constant(lx.val >= 0.0f ? 1.0f : -1.0f);
        Dualf nly = rqy * Dualf::constant(ly.val >= 0.0f ? 1.0f : -1.0f);
        Dualf nlen = cudabot::sqrt(nlx*nlx + nly*nly + Dualf::constant(1e-9f));
        nlx = nlx / nlen; nly = nly / nlen;
        Dualf nwx = c*nlx - s*nly;
        Dualf nwy = s*nlx + c*nly;
        Dualf Fx = (Dualf::constant(0.0f) - nwx) * Dualf::constant(p.push_gain) * pen;
        Dualf Fy = (Dualf::constant(0.0f) - nwy) * Dualf::constant(p.push_gain) * pen;
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
    EpisodeRunner(const Variant& v, const BoxScenario& sc, int K, int T, int seed)
        : v_(v), sc_(sc), K_(K), T_(T), seed_(seed) {
        h_nominal_.assign(T_*CTRL_DIM, 0.0f);
        CUDA_CHECK(cudaMalloc(&d_start_, STATE_DIM*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nominal_, T_*CTRL_DIM*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs_, K_*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_, K_*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perturbed_, K_*T_*CTRL_DIM*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_, K_*sizeof(curandState)));
        reset_rng();
    }
    ~EpisodeRunner() {
        cudaFree(d_start_); cudaFree(d_nominal_); cudaFree(d_costs_);
        cudaFree(d_weights_); cudaFree(d_perturbed_); cudaFree(d_rng_);
    }

    EpisodeMetrics run() {
        reset_state();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        warmup();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        reset_rng();

        auto ep0 = chrono::steady_clock::now();
        float ctrl_ms = 0.0f;
        for (int step = 0; step < sc_.max_steps; step++) {
            float pd = pos_dist(), ad = ang_err();
            min_dist_ = fminf(min_dist_, pd);
            if (pd < sc_.pos_tol && ad < sc_.ang_tol) { reached_ = true; steps_ = step; break; }

            auto t0 = chrono::steady_clock::now();
            controller_update();
            auto t1 = chrono::steady_clock::now();
            ctrl_ms += chrono::duration<float, milli>(t1 - t0).count();

            CUDA_CHECK(cudaMemcpy(h_nominal_.data(), d_nominal_, h_nominal_.size()*sizeof(float), cudaMemcpyDeviceToHost));
            push_step_box_f(px_, py_, ox_, oy_, oth_, h_nominal_[0], h_nominal_[1], sc_.params);
            cum_cost_ += stage_cost_box_f(px_, py_, ox_, oy_, oth_, h_nominal_[0], h_nominal_[1], sc_.gx, sc_.gy, sc_.gth, sc_.params);
            for (int t = 0; t < T_-1; t++) { h_nominal_[t*2+0]=h_nominal_[(t+1)*2+0]; h_nominal_[t*2+1]=h_nominal_[(t+1)*2+1]; }
            h_nominal_[(T_-1)*2+0]=0.0f; h_nominal_[(T_-1)*2+1]=0.0f;
            steps_ = step + 1;
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

    Variant v_; BoxScenario sc_; int K_, T_, seed_;
    float px_=0,py_=0,ox_=0,oy_=0,oth_=0;
    int steps_=0; bool reached_=false; float cum_cost_=0, min_dist_=0;
    vector<float> h_nominal_;
    float *d_start_=nullptr,*d_nominal_=nullptr,*d_costs_=nullptr,*d_weights_=nullptr,*d_perturbed_=nullptr;
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
    for (int i=1;i<argc;i++){ string a=argv[i];
        if (a=="--quick") quick=true;
        else if (a=="--csv"&&i+1<argc) csv_path=argv[++i];
        else if (a=="--k-values"&&i+1<argc) k_values=parse_int_list(argv[++i]);
        else if (a=="--seed-count"&&i+1<argc) seed_count=max(1,atoi(argv[++i]));
        else if (a=="--scenarios"&&i+1<argc) scenario_names=parse_string_list(argv[++i]);
        else if (a=="--planners"&&i+1<argc) planner_names=parse_string_list(argv[++i]);
    }
    ensure_build_dir();

    vector<BoxScenario> all_sc = { make_box_turn(), make_box_align() };
    vector<BoxScenario> scenarios;
    if (!scenario_names.empty()) {
        for (auto& w : scenario_names) { auto it=find_if(all_sc.begin(),all_sc.end(),[&](const BoxScenario&s){return s.name==w;});
            if (it==all_sc.end()){fprintf(stderr,"Unknown scenario: %s\n",w.c_str());return 1;} scenarios.push_back(*it); }
    } else scenarios = all_sc;

    vector<Variant> variants;
    { Variant v; v.name="mppi"; variants.push_back(v); }
    { Variant v; v.name="diff_mppi_1"; v.grad_steps=1; v.alpha=0.02f; variants.push_back(v); }
    { Variant v; v.name="diff_mppi_3"; v.grad_steps=3; v.alpha=0.010f; variants.push_back(v); }
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
            EpisodeRunner runner(variants[vi], sc, ks, DEFAULT_T, run_seed);
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
