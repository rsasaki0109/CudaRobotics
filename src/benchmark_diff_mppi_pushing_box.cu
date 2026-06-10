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
    float w_contact_loss = 0.0f;    // squared gap penalty when pusher leaves contact
    float w_term_pos = 90.0f;       // terminal position
    float w_term_ang = 40.0f;       // terminal orientation
    int obstacle_count = 0;         // axis-aligned obstacle count (0 or 1)
    float obs_min_x = 0.0f, obs_min_y = 0.0f, obs_max_x = 0.0f, obs_max_y = 0.0f;
    float w_obs = 70.0f;            // squared-penetration barrier weight
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
    bool hard_rollout = false;   // sample with the exact hard-contact model (fidelity arm)
    bool use_low_pass_sampling = false;
    float lp_alpha = 0.35f;
    bool use_soppi_sampling = false;
    int soppi_svgd_iters = 1;
    float soppi_step_size = 0.06f;
    float soppi_bandwidth = 2.0f;
    int soppi_neighbor_count = 0; // 0 = all particles; >0 = deterministic particle subset
    bool use_object_informed = false;
    float oi_ref_weight_pos = 1.5f;
    float oi_ref_weight_ang = 3.0f;
    float oi_obj_speed = 1.2f;
    float oi_ang_speed = 1.2f;
    float oi_seed_blend = 0.10f;
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
    double steps_sum = 0, final_sum = 0, min_sum = 0, cost_sum = 0, ms_sum = 0, ang_sum = 0, du_sum = 0, rough_sum = 0;
};

__host__ __device__ inline float clampf_local(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
__host__ __device__ inline float wrapf(float a) { return atan2f(sinf(a), cosf(a)); }

__host__ __device__ inline void box_corners_f(
    float ox, float oy, float oth, float hx, float hy, float out[8])
{
    float c = cosf(oth), s = sinf(oth);
    const float lx[4] = { hx, -hx, -hx, hx };
    const float ly[4] = { hy, hy, -hy, -hy };
    for (int i = 0; i < 4; i++) {
        out[i*2+0] = ox + c*lx[i] - s*ly[i];
        out[i*2+1] = oy + s*lx[i] + c*ly[i];
    }
}

__host__ __device__ inline float point_aabb_penetration_f(
    float px, float py, float min_x, float min_y, float max_x, float max_y)
{
    if (px < min_x || px > max_x || py < min_y || py > max_y) return 0.0f;
    float pen_x = fminf(px - min_x, max_x - px);
    float pen_y = fminf(py - min_y, max_y - py);
    return fminf(pen_x, pen_y);
}

__host__ __device__ inline float box_obstacle_penetration_f(
    float ox, float oy, float oth, const BoxParams& p)
{
    if (p.obstacle_count <= 0) return 0.0f;
    float corners[8];
    box_corners_f(ox, oy, oth, p.hx, p.hy, corners);
    float max_pen = 0.0f;
    for (int i = 0; i < 4; i++) {
        float pen = point_aabb_penetration_f(
            corners[i*2+0], corners[i*2+1],
            p.obs_min_x, p.obs_min_y, p.obs_max_x, p.obs_max_y);
        max_pen = fmaxf(max_pen, pen);
    }
    return max_pen;
}

// Signed distance from pusher centre to box surface (>0 outside, <0 inside).
__host__ __device__ inline float pusher_box_sd_f(
    float px, float py, float ox, float oy, float oth, const BoxParams& p)
{
    float c = cosf(oth), s = sinf(oth);
    float dx = px - ox, dy = py - oy;
    float lx =  c*dx + s*dy;
    float ly = -s*dx + c*dy;
    float qx = fabsf(lx) - p.hx, qy = fabsf(ly) - p.hy;
    float rqx = fmaxf(qx, 0.0f), rqy = fmaxf(qy, 0.0f);
    float outside = sqrtf(rqx*rqx + rqy*rqy + 1e-9f);
    float inside = fminf(fmaxf(qx, qy), 0.0f);
    return outside + inside;
}

__host__ __device__ inline float contact_loss_stage_cost_box_f(
    float px, float py, float ox, float oy, float oth, const BoxParams& p)
{
    if (p.w_contact_loss <= 0.0f) return 0.0f;
    float gap = fmaxf(pusher_box_sd_f(px, py, ox, oy, oth, p) - p.push_r, 0.0f);
    return p.w_contact_loss * gap * gap * p.dt;
}

__host__ __device__ inline float obstacle_stage_cost_box_f(
    float ox, float oy, float oth, const BoxParams& p)
{
    float pen = box_obstacle_penetration_f(ox, oy, oth, p);
    if (pen <= 0.0f) return 0.0f;
    return p.w_obs * pen * pen * p.dt;
}

__host__ __device__ inline float obstacle_terminal_cost_box_f(
    float ox, float oy, float oth, const BoxParams& p)
{
    float pen = box_obstacle_penetration_f(ox, oy, oth, p);
    if (pen <= 0.0f) return 0.0f;
    return p.w_obs * 2.0f * pen * pen;
}

__host__ __device__ inline void resolve_box_obstacles_f(
    float& ox, float& oy, float oth, const BoxParams& p)
{
    if (p.obstacle_count <= 0) return;
    float corners[8];
    const float cx = 0.5f * (p.obs_min_x + p.obs_max_x);
    const float cy = 0.5f * (p.obs_min_y + p.obs_max_y);
    for (int iter = 0; iter < 4; iter++) {
        box_corners_f(ox, oy, oth, p.hx, p.hy, corners);
        float shift_x = 0.0f, shift_y = 0.0f;
        bool moved = false;
        for (int i = 0; i < 4; i++) {
            float px = corners[i*2+0], py = corners[i*2+1];
            float pen = point_aabb_penetration_f(
                px, py, p.obs_min_x, p.obs_min_y, p.obs_max_x, p.obs_max_y);
            if (pen <= 0.0f) continue;
            float dx = px - cx, dy = py - cy;
            float dl = sqrtf(dx*dx + dy*dy + 1e-9f);
            shift_x += (dx / dl) * pen;
            shift_y += (dy / dl) * pen;
            moved = true;
        }
        if (!moved) break;
        ox += shift_x * 0.5f;
        oy += shift_y * 0.5f;
    }
}

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
    resolve_box_obstacles_f(ox, oy, oth, p);
}

// ===================== HARD-CONTACT plant (independent, higher-fidelity) =====================
// A structurally DIFFERENT true plant for the sim-to-sim mismatch test. The
// controller's rollout and gradient stay on the smooth model above; reality is
// this rigid box. Same contact geometry (box-SDF normal + contact point), but the
// force law is replaced by: hard non-penetration + Coulomb stick-slip friction,
// resolved with sequential impulses on a momentum-carrying rigid body (mass M,
// inertia I, linear/angular damping). The smooth model has NO friction (normal
// push only) and a soft penetration ramp; this has a friction cone and exact
// non-penetration -- the mismatch is the contact model itself, not a scaled param.
struct HardParams {
    float dt = 0.05f;
    float hx = 0.35f, hy = 0.18f, push_r = 0.08f;   // geometry (matches smooth nominal)
    float mass = 1.0f;              // box mass
    float mu = 0.6f;                // Coulomb friction coefficient (the swept knob)
    float damp_lin = 6.0f;          // linear velocity damping rate (1/s)
    float damp_ang = 7.0f;          // angular velocity damping rate (1/s)
    float beta = 0.35f;             // Baumgarte penetration push-out fraction
    int   substeps = 4;             // dt subdivisions for stable impulse resolution
    int   iters = 8;                // sequential-impulse iterations per substep
};

// Advance the rigid box one control step under the hard-contact plant. Pusher is a
// velocity-controlled (kinematic) disc, identical to the smooth driver. Box pose
// (ox,oy,oth) and box velocity (vx,vy,w) are updated in place.
__host__ __device__ inline void push_step_box_hard_f(
    float& px, float& py, float& ox, float& oy, float& oth,
    float& vx, float& vy, float& w,
    float ux, float uy, const HardParams& hp)
{
    const float inertia = hp.mass * (hp.hx*hp.hx + hp.hy*hp.hy) / 3.0f;  // 2D box about COM
    const float invM = 1.0f / hp.mass, invI = 1.0f / inertia;
    const float h = hp.dt / hp.substeps;
    for (int ss = 0; ss < hp.substeps; ss++) {
        px += h * ux; py += h * uy;                  // kinematic pusher
        // --- contact geometry (identical to the smooth model) ---
        float c = cosf(oth), s = sinf(oth);
        float dx = px - ox, dy = py - oy;
        float lx =  c*dx + s*dy, ly = -s*dx + c*dy;
        float qx = fabsf(lx) - hp.hx, qy = fabsf(ly) - hp.hy;
        float rqx = fmaxf(qx, 0.0f), rqy = fmaxf(qy, 0.0f);
        float outside = sqrtf(rqx*rqx + rqy*rqy + 1e-9f);
        float inside = fminf(fmaxf(qx, qy), 0.0f);
        float sd = outside + inside;
        float pen = hp.push_r - sd;                  // >0 => overlap (hard)
        if (pen > 0.0f) {
            float nlx = rqx * (lx >= 0 ? 1.0f : -1.0f);
            float nly = rqy * (ly >= 0 ? 1.0f : -1.0f);
            float nlen = sqrtf(nlx*nlx + nly*nly);
            if (nlen < 1e-6f) {                      // pusher inside: normal from min face
                if (qx > qy) { nlx = (lx >= 0 ? 1.f : -1.f); nly = 0.f; }
                else          { nlx = 0.f; nly = (ly >= 0 ? 1.f : -1.f); }
                nlen = 1.0f;
            }
            nlx /= nlen; nly /= nlen;
            float nwx = c*nlx - s*nly, nwy = s*nlx + c*nly;   // world outward normal (box->pusher)
            float cxw = px - nwx*sd, cyw = py - nwy*sd;
            float rx = cxw - ox, ry = cyw - oy;
            float tx = -nwy, ty = nwx;                        // tangent
            float rn = rx*nwy - ry*nwx;                       // r x n  (scalar)
            float rt = rx*ty  - ry*tx;                        // r x t  (scalar)
            float kn = invM + rn*rn*invI;
            float kt = invM + rt*rt*invI;
            float vn_target = -hp.beta * pen / h;             // Baumgarte: recede to clear overlap
            float Jn_acc = 0.0f, Jt_acc = 0.0f;
            for (int it = 0; it < hp.iters; it++) {
                // contact-point velocity of box relative to (kinematic) pusher
                float cpvx = vx - w*ry, cpvy = vy + w*rx;
                float relx = cpvx - ux,  rely = cpvy - uy;
                float vn = relx*nwx + rely*nwy;               // +: box moving toward pusher (closing)
                float dJn = (vn - vn_target) / kn;            // impulse in -n drives vn down
                float newJn = fmaxf(Jn_acc + dJn, 0.0f);      // push only
                dJn = newJn - Jn_acc; Jn_acc = newJn;
                vx -= invM * dJn * nwx; vy -= invM * dJn * nwy;
                w  -= invI * dJn * rn;
                // friction: drive tangential rel-vel to zero, clamp to Coulomb cone
                cpvx = vx - w*ry; cpvy = vy + w*rx;
                relx = cpvx - ux; rely = cpvy - uy;
                float vt = relx*tx + rely*ty;
                float dJt = -vt / kt;
                float maxJt = hp.mu * Jn_acc;
                float newJt = fminf(fmaxf(Jt_acc + dJt, -maxJt), maxJt);
                dJt = newJt - Jt_acc; Jt_acc = newJt;
                vx += invM * dJt * tx; vy += invM * dJt * ty;
                w  += invI * dJt * rt;
            }
        }
        // integrate pose, then damp (semi-implicit, momentum-carrying)
        ox += h * vx; oy += h * vy; oth = wrapf(oth + h * w);
        float dl = fmaxf(0.0f, 1.0f - hp.damp_lin * h);
        float da = fmaxf(0.0f, 1.0f - hp.damp_ang * h);
        vx *= dl; vy *= dl; w *= da;
    }
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
    c += contact_loss_stage_cost_box_f(px, py, ox, oy, oth, p);
    c += obstacle_stage_cost_box_f(ox, oy, oth, p);
    return c;
}

__host__ __device__ inline float terminal_cost_box_f(
    float ox, float oy, float oth, float gx, float gy, float gth, const BoxParams& p)
{
    float dpx = ox - gx, dpy = oy - gy, dth = wrapf(oth - gth);
    return p.w_term_pos * (dpx*dpx + dpy*dpy) + p.w_term_ang * (dth*dth)
         + obstacle_terminal_cost_box_f(ox, oy, oth, p);
}

__host__ __device__ inline void object_ref_box_f(
    float ox0, float oy0, float oth0, float gx, float gy, float gth,
    float dt, float obj_speed, float ang_speed, int step,
    float& rx, float& ry, float& rth)
{
    float dx = gx - ox0, dy = gy - oy0;
    float dist = sqrtf(dx*dx + dy*dy + 1e-9f);
    float travel = fminf(dist, fmaxf(0.0f, obj_speed) * dt * static_cast<float>(step));
    rx = ox0 + dx / dist * travel;
    ry = oy0 + dy / dist * travel;
    float need = wrapf(gth - oth0);
    float astep = fminf(fabsf(need), fmaxf(0.0f, ang_speed) * dt * static_cast<float>(step));
    rth = wrapf(oth0 + (need >= 0.0f ? astep : -astep));
}

// ===================== Dualf helpers + dynamics (gradient) =====================
__device__ inline Dualf d_abs(const Dualf& x) { return x.val >= 0.0f ? x : (Dualf::constant(0.0f) - x); }
__device__ inline Dualf d_relu(const Dualf& x) { return x.val > 0.0f ? x : Dualf::constant(0.0f); }
__device__ inline Dualf d_max(const Dualf& a, const Dualf& b) { return a.val >= b.val ? a : b; }
__device__ inline Dualf d_min(const Dualf& a, const Dualf& b) { return a.val <= b.val ? a : b; }
__device__ inline Dualf d_min0(const Dualf& x) { return x.val < 0.0f ? x : Dualf::constant(0.0f); }

__device__ inline Dualf obstacle_stage_cost_dual(
    Dualf ox, Dualf oy, Dualf oth, const BoxParams& p)
{
    if (p.obstacle_count <= 0) return Dualf::constant(0.0f);
    Dualf cost = Dualf::constant(0.0f);
    Dualf c = cudabot::cos(oth), s = cudabot::sin(oth);
    const float lx[4] = { p.hx, -p.hx, -p.hx, p.hx };
    const float ly[4] = { p.hy, p.hy, -p.hy, -p.hy };
    for (int i = 0; i < 4; i++) {
        Dualf px = ox + c*Dualf::constant(lx[i]) - s*Dualf::constant(ly[i]);
        Dualf py = oy + s*Dualf::constant(lx[i]) + c*Dualf::constant(ly[i]);
        Dualf pen_x = d_min(Dualf::constant(p.obs_max_x) - px, px - Dualf::constant(p.obs_min_x));
        Dualf pen_y = d_min(Dualf::constant(p.obs_max_y) - py, py - Dualf::constant(p.obs_min_y));
        Dualf pen = d_relu(d_min(pen_x, pen_y));
        cost = cost + Dualf::constant(p.w_obs) * pen * pen * Dualf::constant(p.dt);
    }
    return cost;
}

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
        if (p.w_contact_loss > 0.0f) {
            Dualf gap = d_relu(sd - Dualf::constant(p.push_r));
            cost = cost + Dualf::constant(p.w_contact_loss) * gap * gap * Dualf::constant(p.dt);
        }
        cost = cost + obstacle_stage_cost_dual(ox, oy, oth, p);
    }
    Dualf dpx = ox - Dualf::constant(gx), dpy = oy - Dualf::constant(gy);
    Dualf dthr = oth - Dualf::constant(gth);
    Dualf dth = cudabot::atan2(cudabot::sin(dthr), cudabot::cos(dthr));
    cost = cost + Dualf::constant(p.w_term_pos) * (dpx*dpx + dpy*dpy);
    cost = cost + Dualf::constant(p.w_term_ang) * (dth*dth);
    cost = cost + obstacle_stage_cost_dual(ox, oy, oth, p) * Dualf::constant(2.0f);
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

__global__ void rollout_low_pass_kernel(
    const float* d_start, const float* d_nominal, float* d_costs, float* d_perturbed,
    curandState* d_rng, BoxParams p, float gx, float gy, float gth, int K, int T, float sigma, float lp_alpha)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curandState rng = d_rng[k];
    float px=d_start[0], py=d_start[1], ox=d_start[2], oy=d_start[3], oth=d_start[4];
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
        push_step_box_f(px, py, ox, oy, oth, ux, uy, p);
        cost += stage_cost_box_f(px, py, ox, oy, oth, ux, uy, gx, gy, gth, p);
    }
    cost += terminal_cost_box_f(ox, oy, oth, gx, gy, gth, p);
    d_costs[k] = cost;
    d_rng[k] = rng;
}

__global__ void rollout_object_informed_kernel(
    const float* d_start, const float* d_nominal, float* d_costs, float* d_perturbed,
    curandState* d_rng, BoxParams p, float gx, float gy, float gth, int K, int T,
    float sigma, bool use_low_pass, float lp_alpha, float oi_ref_weight_pos,
    float oi_ref_weight_ang, float oi_obj_speed, float oi_ang_speed)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curandState rng = d_rng[k];
    const float ox0=d_start[2], oy0=d_start[3], oth0=d_start[4];
    float px=d_start[0], py=d_start[1], ox=ox0, oy=oy0, oth=oth0;
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
        float ux = clampf_local(d_nominal[t*2+0] + nx * sigma, -p.u_max, p.u_max);
        float uy = clampf_local(d_nominal[t*2+1] + ny * sigma, -p.u_max, p.u_max);
        d_perturbed[k*T*2 + t*2 + 0] = ux;
        d_perturbed[k*T*2 + t*2 + 1] = uy;
        push_step_box_f(px, py, ox, oy, oth, ux, uy, p);
        cost += stage_cost_box_f(px, py, ox, oy, oth, ux, uy, gx, gy, gth, p);
        float rx, ry, rth;
        object_ref_box_f(ox0, oy0, oth0, gx, gy, gth, p.dt, oi_obj_speed, oi_ang_speed, t + 1, rx, ry, rth);
        float ex = ox - rx, ey = oy - ry, eth = wrapf(oth - rth);
        cost += (fmaxf(0.0f, oi_ref_weight_pos) * (ex*ex + ey*ey) +
                 fmaxf(0.0f, oi_ref_weight_ang) * eth*eth) * p.dt;
    }
    cost += terminal_cost_box_f(ox, oy, oth, gx, gy, gth, p);
    d_costs[k] = cost;
    d_rng[k] = rng;
}

__global__ void fixed_rollout_kernel(
    const float* d_start, const float* d_controls, float* d_costs,
    BoxParams p, float gx, float gy, float gth, int K, int T)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    float px=d_start[0], py=d_start[1], ox=d_start[2], oy=d_start[3], oth=d_start[4];
    float cost = 0.0f;
    for (int t = 0; t < T; t++) {
        float ux = clampf_local(d_controls[k*T*2 + t*2 + 0], -p.u_max, p.u_max);
        float uy = clampf_local(d_controls[k*T*2 + t*2 + 1], -p.u_max, p.u_max);
        push_step_box_f(px, py, ox, oy, oth, ux, uy, p);
        cost += stage_cost_box_f(px, py, ox, oy, oth, ux, uy, gx, gy, gth, p);
    }
    cost += terminal_cost_box_f(ox, oy, oth, gx, gy, gth, p);
    d_costs[k] = cost;
}

// Fidelity-arm rollout: the sampler predicts with the EXACT hard-contact plant model
// (push_step_box_hard_f) instead of the smooth model -- no model mismatch, but the
// dynamics are non-differentiable so there is no gradient to add. Used to ask whether
// giving MPPI the right model beats the smooth-but-differentiable gradient. Cost uses
// the same weights (p_cost) as every other planner. Each sample carries its own box
// velocity (vx,vy,w) initialised to rest, exactly like the episode's true plant.
__global__ void rollout_kernel_hard(
    const float* d_start, const float* d_nominal, float* d_costs, float* d_perturbed,
    curandState* d_rng, BoxParams p_cost, HardParams hp, float gx, float gy, float gth,
    int K, int T, float sigma)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curandState rng = d_rng[k];
    float px=d_start[0], py=d_start[1], ox=d_start[2], oy=d_start[3], oth=d_start[4];
    float vx=0.0f, vy=0.0f, w=0.0f;
    float cost = 0.0f;
    for (int t = 0; t < T; t++) {
        float ux = clampf_local(d_nominal[t*2+0] + curand_normal(&rng) * sigma, -p_cost.u_max, p_cost.u_max);
        float uy = clampf_local(d_nominal[t*2+1] + curand_normal(&rng) * sigma, -p_cost.u_max, p_cost.u_max);
        d_perturbed[k*T*2 + t*2 + 0] = ux;
        d_perturbed[k*T*2 + t*2 + 1] = uy;
        push_step_box_hard_f(px, py, ox, oy, oth, vx, vy, w, ux, uy, hp);
        cost += stage_cost_box_f(px, py, ox, oy, oth, ux, uy, gx, gy, gth, p_cost);
    }
    cost += terminal_cost_box_f(ox, oy, oth, gx, gy, gth, p_cost);
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

// Total cost of rolling the TRUE hard-contact plant (push_step_box_hard_f) from
// `start` under `nominal`, box initialised at rest -- the same per-step planning
// convention as the fidelity-arm sampler (rollout_kernel_hard). The smooth model is
// velocity-free, so the smooth gradient is implicitly also evaluated from rest:
// comparing the two from the same rest state is apples-to-apples. Cost uses the same
// weights (pc) as every planner. Used by the gradient-agreement capstone to obtain a
// finite-difference sensitivity of the structurally-different true plant.
static float hard_rollout_cost(const float start[STATE_DIM], const vector<float>& nominal,
                               int T, const HardParams& hp, const BoxParams& pc,
                               float gx, float gy, float gth) {
    float px=start[0], py=start[1], ox=start[2], oy=start[3], oth=start[4];
    float vx=0.0f, vy=0.0f, w=0.0f, cost=0.0f;
    for (int t = 0; t < T; t++) {
        float ux = clampf_local(nominal[t*2+0], -pc.u_max, pc.u_max);
        float uy = clampf_local(nominal[t*2+1], -pc.u_max, pc.u_max);
        push_step_box_hard_f(px, py, ox, oy, oth, vx, vy, w, ux, uy, hp);
        cost += stage_cost_box_f(px, py, ox, oy, oth, ux, uy, gx, gy, gth, pc);
    }
    cost += terminal_cost_box_f(ox, oy, oth, gx, gy, gth, pc);
    return cost;
}

// Pusher-box penetration under the smooth box-SDF (>0 => overlap, contact active).
// Instantaneous contact indicator used to gate the gradient-agreement diagnostic to
// contact-engaged steps (where u actually couples into box pose, so the gradient is
// dynamics-driven rather than a residual of the tiny control-cost term).
static float box_pen(float px, float py, float ox, float oy, float oth, const BoxParams& p) {
    float c=cosf(oth), s=sinf(oth);
    float dx=px-ox, dy=py-oy;
    float lx= c*dx + s*dy, ly=-s*dx + c*dy;
    float qx=fabsf(lx)-p.hx, qy=fabsf(ly)-p.hy;
    float rqx=fmaxf(qx,0.0f), rqy=fmaxf(qy,0.0f);
    float outside=sqrtf(rqx*rqx + rqy*rqy + 1e-9f);
    float inside=fminf(fmaxf(qx,qy),0.0f);
    return p.push_r - (outside + inside);
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

// Diagnostic: write the full 2T smooth-model forward-mode autodiff gradient vector --
// the EXACT gradient grad_step_kernel descends -- without taking a step. The
// gradient-agreement capstone compares this direction (what the controller follows)
// against the true hard-contact plant's finite-difference sensitivity.
__global__ void grad_vec_kernel(
    const float* d_start, const float* d_nominal, float* d_grad,
    BoxParams p, float gx, float gy, float gth, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float start[STATE_DIM] = { d_start[0], d_start[1], d_start[2], d_start[3], d_start[4] };
    for (int kp = 0; kp < 2*T; kp++)
        d_grad[kp] = dcost_dparam_box(start, d_nominal, T, kp, gx, gy, gth, p);
}

__global__ void soppi_timestep_score_kernel(
    const float* d_start, const float* d_controls, float* d_scores,
    BoxParams p, float gx, float gy, float gth, int K, int T, float lambda)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * T;
    if (idx >= total) return;

    int k = idx / T;
    int t = idx - k * T;
    int base = k * T * CTRL_DIM + t * CTRL_DIM;
    float start[STATE_DIM] = { d_start[0], d_start[1], d_start[2], d_start[3], d_start[4] };
    const float* controls = &d_controls[k * T * CTRL_DIM];
    float inv_lambda = 1.0f / fmaxf(lambda, 1.0e-3f);
    d_scores[base + 0] = -clampf_local(
        dcost_dparam_box(start, controls, T, t * CTRL_DIM + 0, gx, gy, gth, p) * inv_lambda,
        -25.0f, 25.0f);
    d_scores[base + 1] = -clampf_local(
        dcost_dparam_box(start, controls, T, t * CTRL_DIM + 1, gx, gy, gth, p) * inv_lambda,
        -25.0f, 25.0f);
}

__global__ void soppi_svgd_step_kernel(
    const float* d_controls, float* d_controls_next, const float* d_control_grads,
    BoxParams p, int K, int T, int neighbor_count,
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

    float phi_x = 0.0f;
    float phi_y = 0.0f;
    for (int m = 0; m < neighbor_samples; m++) {
        int j = m;
        if (neighbor_count > 0 && neighbor_count < K) {
            // Low-discrepancy hashed subset: better gradient coverage than a
            // fixed stride ring, which can miss contact-loss structure.
            unsigned int h = 1597334677u * static_cast<unsigned int>(k + 1)
                           + 3812015801u * static_cast<unsigned int>(m + 1)
                           + 2654435761u * static_cast<unsigned int>(t + 1);
            j = static_cast<int>(h % static_cast<unsigned int>(K));
        }
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
    // Gradient-agreement capstone: per control step, measure the cosine similarity
    // between (a) the smooth-model autodiff gradient the controller descends and
    // (b) the TRUE plant's finite-difference cost sensitivity, both w.r.t. the warm
    // plan at the visited state. cos > 0 => following the smooth gradient also lowers
    // the true cost (the gradient "buys" real improvement); cos -> 0/negative => the
    // gradient misleads. Sweeping the hard plant's friction explains the success
    // boundary at the level of gradient DIRECTION. grad_agree_flat stride 7:
    // [step, cos_full, cos_first, gnorm_smooth, gnorm_hard, rem_ang, engaged].
    // engaged = contact-active AND still-needs-rotation (where the mechanism lives).
    bool record_grad_agree = false;
    vector<float> grad_agree_flat;       // stride 7 (see above)
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
    // Sim-to-sim mismatch: when true, the TRUE plant is the hard-contact rigid body
    // (push_step_box_hard_f) -- a structurally different contact model (hard
    // non-penetration + Coulomb stick-slip friction + box momentum) -- while the
    // controller's rollout + gradient keep the smooth model. This is the strongest
    // test of external validity: the contact MODEL (not a scaled parameter) is wrong.
    // Default false => the smooth plant runs and published numbers are byte-identical.
    bool  true_plant_hard = false;
    float plant_mu = 0.6f;               // Coulomb friction of the hard true plant

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
        CUDA_CHECK(cudaMalloc(&d_grad_, T_*CTRL_DIM*sizeof(float)));
        if (v_.use_soppi_sampling) {
            CUDA_CHECK(cudaMalloc(&d_soppi_scratch_, K_*T_*CTRL_DIM*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_soppi_grad_, K_*T_*CTRL_DIM*sizeof(float)));
        }
        reset_rng();
    }
    ~EpisodeRunner() {
        cudaFree(d_start_); cudaFree(d_nominal_); cudaFree(d_costs_);
        cudaFree(d_weights_); cudaFree(d_perturbed_); cudaFree(d_rng_);
        cudaFree(d_rot_); cudaFree(d_netrot_); cudaFree(d_grad_);
        if (d_soppi_scratch_) cudaFree(d_soppi_scratch_);
        if (d_soppi_grad_) cudaFree(d_soppi_grad_);
    }

    EpisodeMetrics run() {
        // Hard-contact params, built BEFORE warmup so a hard_rollout controller can
        // use them. Same geometry as the controller's nominal box (size mismatch is its
        // own axis, kept off here); friction mu is the swept knob. Used both as the
        // true plant (--true-plant hard) and, for the fidelity-arm planner, in rollout.
        hard_p_ = HardParams();
        hard_p_.dt = sc_.params.dt; hard_p_.push_r = sc_.params.push_r;
        hard_p_.hx = sc_.params.hx * plant_size_scale;
        hard_p_.hy = sc_.params.hy * plant_size_scale;
        hard_p_.mu = plant_mu;

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
        const HardParams& hard_p = hard_p_;

        auto ep0 = chrono::steady_clock::now();
        float ctrl_ms = 0.0f;
        float prev_ux = 0.0f, prev_uy = 0.0f;
        bool have_prev_control = false;
        float control_delta_sum = 0.0f;
        float control_roughness_sum = 0.0f;
        int control_delta_count = 0;
        int collision_count = 0;
        bool collision_free = true;
        if (record_traj) { traj_flat.clear();
            traj_flat.insert(traj_flat.end(), {px_,py_,ox_,oy_,oth_}); }
        for (int step = 0; step < sc_.max_steps; step++) {
            float pd = pos_dist(), ad = ang_err();
            min_dist_ = fminf(min_dist_, pd);
            if (pd < sc_.pos_tol && ad < sc_.ang_tol) { reached_ = true; steps_ = step; break; }

            // Gradient-agreement diagnostic at the warm plan (before the controller
            // overwrites it); not counted in control timing (it is pure instrumentation).
            if (record_grad_agree) collect_grad_agree(step);

            auto t0 = chrono::steady_clock::now();
            controller_update();
            auto t1 = chrono::steady_clock::now();
            ctrl_ms += chrono::duration<float, milli>(t1 - t0).count();

            // h_nominal_ still holds the PRE-update mean (the sampler perturbed
            // around it); d_costs_/d_perturbed_ hold this step's K samples. Collect
            // the mechanism diagnostic before h_nominal_ is overwritten below.
            if (record_diag) collect_diag(step);

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
            if (true_plant_hard)
                push_step_box_hard_f(px_, py_, ox_, oy_, oth_, vx_, vy_, w_, h_nominal_[0], h_nominal_[1], hard_p);
            else
                push_step_box_f(px_, py_, ox_, oy_, oth_, h_nominal_[0], h_nominal_[1], plant_p);
            cum_cost_ += stage_cost_box_f(px_, py_, ox_, oy_, oth_, h_nominal_[0], h_nominal_[1], sc_.gx, sc_.gy, sc_.gth, sc_.params);
            if (box_obstacle_penetration_f(ox_, oy_, oth_, plant_p) > 0.01f) {
                collision_count++;
                collision_free = false;
            }
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
        m.reached_goal = reached_?1:0;
        m.collision_free = collision_free ? 1 : 0;
        m.collisions = collision_count;
        m.success = (reached_ && collision_free) ? 1 : 0;
        m.steps=steps_; m.final_distance=pd; m.min_goal_distance=ad;  // store final ang err in min col
        m.cumulative_cost=cum_cost_;
        m.mean_control_delta = control_delta_count > 0 ? control_delta_sum / control_delta_count : 0.0f;
        m.control_roughness = control_delta_count > 0 ? control_roughness_sum / control_delta_count : 0.0f;
        m.total_control_ms=ctrl_ms; m.avg_control_ms = steps_>0? ctrl_ms/steps_ : 0.0f;
        m.episode_ms = chrono::duration<float, milli>(ep1 - ep0).count();
        m.sample_budget = (long long)steps_ * K_ * T_;
        return m;
    }

private:
    void reset_rng() { int b=256; init_curand_kernel<<<(K_+b-1)/b,b>>>(d_rng_, K_, (unsigned long long)seed_); CUDA_CHECK(cudaDeviceSynchronize()); }
    void reset_state() { px_=sc_.px0; py_=sc_.py0; ox_=sc_.ox0; oy_=sc_.oy0; oth_=sc_.oth0; vx_=vy_=w_=0.0f; steps_=0; reached_=false; cum_cost_=0; min_dist_=pos_dist(); }
    float pos_dist() const { float dx=ox_-sc_.gx, dy=oy_-sc_.gy; return sqrtf(dx*dx+dy*dy); }
    float ang_err() const { return fabsf(wrapf(oth_ - sc_.gth)); }
    void sync_start() { float s[STATE_DIM]={px_,py_,ox_,oy_,oth_}; CUDA_CHECK(cudaMemcpy(d_start_, s, STATE_DIM*sizeof(float), cudaMemcpyHostToDevice)); }
    void controller_update() {
        sync_start();
        seed_object_informed_nominal();
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size()*sizeof(float), cudaMemcpyHostToDevice));
        int b=256;
        if (v_.hard_rollout)   // fidelity arm: sample with the exact hard-contact model (no gradient)
            rollout_kernel_hard<<<(K_+b-1)/b,b>>>(d_start_, d_nominal_, d_costs_, d_perturbed_, d_rng_, sc_.params, hard_p_, sc_.gx, sc_.gy, sc_.gth, K_, T_, v_.sigma);
        else if (v_.use_object_informed)
            rollout_object_informed_kernel<<<(K_+b-1)/b,b>>>(
                d_start_, d_nominal_, d_costs_, d_perturbed_, d_rng_,
                sc_.params, sc_.gx, sc_.gy, sc_.gth, K_, T_, v_.sigma,
                v_.use_low_pass_sampling, v_.lp_alpha, v_.oi_ref_weight_pos,
                v_.oi_ref_weight_ang, v_.oi_obj_speed, v_.oi_ang_speed);
        else if (v_.use_low_pass_sampling)
            rollout_low_pass_kernel<<<(K_+b-1)/b,b>>>(
                d_start_, d_nominal_, d_costs_, d_perturbed_, d_rng_,
                sc_.params, sc_.gx, sc_.gy, sc_.gth, K_, T_, v_.sigma, v_.lp_alpha);
        else
            rollout_kernel<<<(K_+b-1)/b,b>>>(d_start_, d_nominal_, d_costs_, d_perturbed_, d_rng_, sc_.params, sc_.gx, sc_.gy, sc_.gth, K_, T_, v_.sigma);
        if (v_.use_soppi_sampling) {
            int total_particles = K_ * T_;
            float* d_controls_src = d_perturbed_;
            float* d_controls_dst = d_soppi_scratch_;
            for (int iter = 0; iter < max(1, v_.soppi_svgd_iters); iter++) {
                soppi_timestep_score_kernel<<<(total_particles+b-1)/b,b>>>(
                    d_start_, d_controls_src, d_soppi_grad_,
                    sc_.params, sc_.gx, sc_.gy, sc_.gth, K_, T_, v_.lambda);
                soppi_svgd_step_kernel<<<(total_particles+b-1)/b,b>>>(
                    d_controls_src, d_controls_dst, d_soppi_grad_,
                    sc_.params, K_, T_, v_.soppi_neighbor_count,
                    v_.lambda, v_.soppi_bandwidth, v_.soppi_step_size, v_.sigma);
                fixed_rollout_kernel<<<(K_+b-1)/b,b>>>(
                    d_start_, d_controls_dst, d_costs_,
                    sc_.params, sc_.gx, sc_.gy, sc_.gth, K_, T_);
                float* tmp = d_controls_src;
                d_controls_src = d_controls_dst;
                d_controls_dst = tmp;
            }
            if (d_controls_src != d_perturbed_) {
                CUDA_CHECK(cudaMemcpy(d_perturbed_, d_controls_src,
                                      K_*T_*CTRL_DIM*sizeof(float), cudaMemcpyDeviceToDevice));
            }
        }
        weights_kernel<<<1,1>>>(d_costs_, d_weights_, K_, v_.lambda);
        update_kernel<<<(T_+b-1)/b,b>>>(d_nominal_, d_perturbed_, d_weights_, K_, T_);
        for (int g = 0; g < v_.grad_steps; g++)
            grad_step_kernel<<<1,1>>>(d_start_, d_nominal_, sc_.params, sc_.gx, sc_.gy, sc_.gth, T_, v_.alpha, v_.grad_clip);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    void warmup() { for (int i = 0; i < 3; i++) controller_update(); }

    void seed_object_informed_nominal() {
        if (!v_.use_object_informed || v_.oi_seed_blend <= 0.0f) return;
        const BoxParams& p = sc_.params;
        float blend = clampf_local(v_.oi_seed_blend, 0.0f, 1.0f);
        float sim_px = px_, sim_py = py_;
        float dxg = sc_.gx - ox_, dyg = sc_.gy - oy_;
        float dg = sqrtf(dxg*dxg + dyg*dyg + 1e-9f);
        float dirx = dxg / dg, diry = dyg / dg;
        float contact_offset = fmaxf(p.hx, p.hy) + p.push_r + fmaxf(0.0f, v_.oi_contact_margin);
        for (int t = 0; t < T_; t++) {
            float refx, refy, refth;
            object_ref_box_f(ox_, oy_, oth_, sc_.gx, sc_.gy, sc_.gth,
                             p.dt, v_.oi_obj_speed, v_.oi_ang_speed, t + 1, refx, refy, refth);
            float target_px, target_py;
            if (dg > fmaxf(0.25f, 1.15f * sc_.pos_tol)) {
                target_px = refx - dirx * contact_offset;
                target_py = refy - diry * contact_offset;
            } else {
                float need = wrapf(sc_.gth - oth_);
                float sign = need >= 0.0f ? 1.0f : -1.0f;
                float c = cosf(refth), s = sinf(refth);
                float lx = -p.hx - p.push_r - v_.oi_contact_margin;
                float ly = -sign * p.hy;
                target_px = refx + c*lx - s*ly;
                target_py = refy + s*lx + c*ly;
            }
            float ux = clampf_local((target_px - sim_px) / p.dt, -p.u_max, p.u_max);
            float uy = clampf_local((target_py - sim_py) / p.dt, -p.u_max, p.u_max);
            int base = t * CTRL_DIM;
            h_nominal_[base + 0] = (1.0f - blend) * h_nominal_[base + 0] + blend * ux;
            h_nominal_[base + 1] = (1.0f - blend) * h_nominal_[base + 1] + blend * uy;
            sim_px += p.dt * h_nominal_[base + 0];
            sim_py += p.dt * h_nominal_[base + 1];
        }
    }

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

    // Gradient-agreement capstone (see grad_agree_flat docs). At the current decision
    // state and warm plan, compute the smooth-model autodiff gradient (the direction
    // the controller descends) and the TRUE hard-contact plant's central-difference
    // sensitivity, then their cosine similarity. cos is scale-invariant, which is
    // exactly right: the smooth and hard dynamics have different magnitudes but the
    // mechanism question is whether they point the same way.
    void collect_grad_agree(int step) {
        // (a) smooth gradient: exactly grad_step_kernel's gradient, on GPU.
        sync_start();
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size()*sizeof(float), cudaMemcpyHostToDevice));
        grad_vec_kernel<<<1,1>>>(d_start_, d_nominal_, d_grad_, sc_.params, sc_.gx, sc_.gy, sc_.gth, T_);
        CUDA_CHECK(cudaDeviceSynchronize());
        vector<float> gs(T_*CTRL_DIM);
        CUDA_CHECK(cudaMemcpy(gs.data(), d_grad_, gs.size()*sizeof(float), cudaMemcpyDeviceToHost));
        // (b) hard-plant FD sensitivity: central difference on the true plant cost.
        float start[STATE_DIM] = { px_, py_, ox_, oy_, oth_ };
        vector<float> nom = h_nominal_, gh(T_*CTRL_DIM);
        const float eps = 1e-2f;   // velocity units; hard plant has substeps -> not too small
        for (int i = 0; i < T_*CTRL_DIM; i++) {
            float saved = nom[i];
            nom[i] = saved + eps; float cp = hard_rollout_cost(start, nom, T_, hard_p_, sc_.params, sc_.gx, sc_.gy, sc_.gth);
            nom[i] = saved - eps; float cm = hard_rollout_cost(start, nom, T_, hard_p_, sc_.params, sc_.gx, sc_.gy, sc_.gth);
            nom[i] = saved;        gh[i] = (cp - cm) / (2.0f*eps);
        }
        // cosine similarity over the full horizon and over the first applied control.
        double dot=0, ns=0, nh=0, dot1=0, ns1=0, nh1=0;
        for (int i=0;i<T_*CTRL_DIM;i++){ dot+=(double)gs[i]*gh[i]; ns+=(double)gs[i]*gs[i]; nh+=(double)gh[i]*gh[i]; }
        for (int i=0;i<CTRL_DIM;i++){ dot1+=(double)gs[i]*gh[i]; ns1+=(double)gs[i]*gs[i]; nh1+=(double)gh[i]*gh[i]; }
        float gnorm_s=(float)sqrt(ns), gnorm_h=(float)sqrt(nh);
        float cos_full  = (ns>1e-18 && nh>1e-18) ? (float)(dot /(sqrt(ns )*sqrt(nh ))) : 0.0f;
        float cos_first = (ns1>1e-18&& nh1>1e-18) ? (float)(dot1/(sqrt(ns1)*sqrt(nh1))) : 0.0f;
        float rem_ang = ang_err();
        // engaged = pusher in/near contact AND box still needs to rotate AND both
        // gradients are non-negligible (so cos is a real direction comparison, not
        // dominated by the tiny control-cost residual present when out of contact).
        float pen = box_pen(px_, py_, ox_, oy_, oth_, sc_.params);
        int engaged = (rem_ang > sc_.ang_tol && pen > -0.05f &&
                       gnorm_s > 1e-2f && gnorm_h > 1e-2f) ? 1 : 0;
        grad_agree_flat.insert(grad_agree_flat.end(),
            { (float)step, cos_full, cos_first, gnorm_s, gnorm_h, rem_ang, (float)engaged });
    }

    Variant v_; BoxScenario sc_; int K_, T_, seed_;
    HardParams hard_p_;                     // hard-contact params (true plant and/or fidelity-arm rollout)
    float px_=0,py_=0,ox_=0,oy_=0,oth_=0;
    float vx_=0,vy_=0,w_=0;                 // box velocity (hard true plant only)
    int steps_=0; bool reached_=false; float cum_cost_=0, min_dist_=0;
    vector<float> h_nominal_;
    float *d_start_=nullptr,*d_nominal_=nullptr,*d_costs_=nullptr,*d_weights_=nullptr,*d_perturbed_=nullptr;
    float *d_rot_=nullptr,*d_netrot_=nullptr;
    float *d_grad_=nullptr;                 // 2T smooth-model gradient (grad-agreement diag)
    float *d_soppi_scratch_=nullptr,*d_soppi_grad_=nullptr;
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
// Negative control for the mechanism claim: a SMALL rotation with a WIDE angular
// tolerance and a pusher that engages the box near a corner, so isotropic velocity
// noise routinely produces correctly-signed torque. Here the latch the gradient is
// supposed to break does not form — vanilla MPPI alone succeeds. The prediction is
// that this regime has a HIGH escape_frac (sampling is NOT contact-starved), so the
// gradient buys little: escape_frac becomes a predictor of where Diff-MPPI helps.
static BoxScenario make_box_swivel() {
    BoxScenario s; s.name="box_swivel";
    s.ox0=1.5f; s.oy0=1.5f; s.oth0=0.0f; s.px0=1.85f; s.py0=0.9f;
    s.gx=1.55f; s.gy=1.95f; s.gth=0.30f;          // small +0.30 rotate via off-centre push
    s.pos_tol=0.22f; s.ang_tol=0.20f; s.max_steps=240;
    return s;
}
// Orientation-binding variant of box_align: identical geometry with a slightly
// widened position gate (0.28 m) and a tighter heading gate (0.08 rad). On the
// parent task both planners finish near ~0.28 m / ~0.03 rad without reaching
// success; here the combined gate separates SOPPI-family planners from vanilla MPPI.
static BoxScenario make_box_align_strict() {
    BoxScenario s = make_box_align();
    s.name = "box_align_strict";
    s.pos_tol = 0.28f;
    s.ang_tol = 0.08f;
    return s;
}
// Obstacle-detour variant of box_align: a wall blocks the direct upward push lane,
// forcing a lateral approach before the final orientation alignment.
static BoxScenario make_box_align_detour() {
    BoxScenario s = make_box_align();
    s.name = "box_align_detour";
    s.px0 = 1.10f;
    s.py0 = 1.20f;
    s.max_steps = 280;
    s.params.obstacle_count = 1;
    // Narrow wall on the direct push lane between start and goal pose.
    s.params.obs_min_x = 1.48f;
    s.params.obs_max_x = 1.72f;
    s.params.obs_min_y = 1.98f;
    s.params.obs_max_y = 2.14f;
    s.params.w_obs = 85.0f;
    return s;
}
// Contact-loss variant of box_align_strict: orientation-binding gate plus a gap
// penalty that punishes losing face contact during the rotation arc.
static BoxScenario make_box_align_contact_loss() {
    BoxScenario s = make_box_align_strict();
    s.name = "box_align_contact_loss";
    s.params.w_near = 0.0f;
    s.params.w_contact_loss = 47.0f;
    s.params.rot_gain = 15.0f;
    s.params.pen_thresh = 0.007f;
    return s;
}
// Wider orientation gate on the contact-loss cell: same contact gradient, easier
// success criterion so pure SOPPI-family planners can reach >=0.50 on fixed seeds.
static BoxScenario make_box_align_contact_arc() {
    BoxScenario s = make_box_align_contact_loss();
    s.name = "box_align_contact_arc";
    s.pos_tol = 0.30f;
    s.ang_tol = 0.12f;
    return s;
}

// ======================== Utilities ========================
static void ensure_build_dir() { mkdir("build", 0755); }
static vector<int> parse_int_list(const string& t){ vector<int> v; string tok; stringstream ss(t); while(getline(ss,tok,',')) if(!tok.empty()) v.push_back(max(1,atoi(tok.c_str()))); sort(v.begin(),v.end()); v.erase(unique(v.begin(),v.end()),v.end()); return v; }
static vector<string> parse_string_list(const string& t){ vector<string> v; string tok; stringstream ss(t); while(getline(ss,tok,',')) if(!tok.empty()) v.push_back(tok); sort(v.begin(),v.end()); v.erase(unique(v.begin(),v.end()),v.end()); return v; }
static void write_csv(const vector<EpisodeMetrics>& rows, const string& path) {
    ofstream out(path);
    out << "scenario,planner,seed,k_samples,t_horizon,grad_steps,alpha,reached_goal,collision_free,success,steps,final_distance,min_goal_distance,cumulative_cost,collisions,mean_control_delta,control_roughness,avg_control_ms,total_control_ms,episode_ms,sample_budget\n";
    for (const auto& r : rows)
        out << r.scenario<<','<<r.planner<<','<<r.seed<<','<<r.k_samples<<','<<r.t_horizon<<','<<r.grad_steps<<','<<r.alpha<<','
            << r.reached_goal<<','<<r.collision_free<<','<<r.success<<','<<r.steps<<','<<r.final_distance<<','<<r.min_goal_distance<<','
            << r.cumulative_cost<<','<<r.collisions<<','<<r.mean_control_delta<<','<<r.control_roughness<<','
            << r.avg_control_ms<<','<<r.total_control_ms<<','<<r.episode_ms<<','<<r.sample_budget<<'\n';
}
static void print_summary(const vector<EpisodeMetrics>& rows) {
    map<string, SummaryStats> st;
    for (const auto& r : rows) {
        string key = r.scenario + " | " + r.planner + " | K=" + to_string(r.k_samples);
        auto& s = st[key]; s.episodes++; s.successes+=r.success; s.steps_sum+=r.steps;
        s.final_sum+=r.final_distance; s.ang_sum+=r.min_goal_distance; s.cost_sum+=r.cumulative_cost; s.ms_sum+=r.avg_control_ms;
        s.du_sum+=r.mean_control_delta; s.rough_sum+=r.control_roughness;
    }
    cout << "=== benchmark_diff_mppi_pushing_box summary ===" << endl;
    for (const auto& kv : st) {
        const SummaryStats& s = kv.second; float n=s.episodes;
        printf("%s : success=%.2f steps=%.1f pos_err=%.3f ang_err=%.3f cost=%.1f du=%.3f rough=%.3f avg_ms=%.3f\n",
               kv.first.c_str(), s.successes/n, s.steps_sum/n, s.final_sum/n, s.ang_sum/n, s.cost_sum/n, s.du_sum/n, s.rough_sum/n, s.ms_sum/n);
    }
}

// ======================== Main ========================
int main(int argc, char** argv) {
    bool quick=false; string csv_path="build/benchmark_diff_mppi_pushing_box.csv";
    vector<int> k_values; vector<string> scenario_names, planner_names; int seed_count=-1;
    int horizon=DEFAULT_T; string dump_traj_prefix=""; float plant_gain_scale=1.0f; float plant_size_scale=1.0f;
    string diag_prefix=""; bool true_plant_hard=false; float plant_mu=0.6f;
    string grad_agree_prefix="";
    float override_lp_alpha = -1.0f;
    int override_soppi_iters = -1;
    int override_soppi_neighbor_count = -1;
    float override_soppi_step_size = -1.0f;
    float override_soppi_bandwidth = -1.0f;
    for (int i=1;i<argc;i++){ string a=argv[i];
        if (a=="--quick") quick=true;
        else if (a=="--csv"&&i+1<argc) csv_path=argv[++i];
        else if (a=="--k-values"&&i+1<argc) k_values=parse_int_list(argv[++i]);
        else if (a=="--seed-count"&&i+1<argc) seed_count=max(1,atoi(argv[++i]));
        else if (a=="--scenarios"&&i+1<argc) scenario_names=parse_string_list(argv[++i]);
        else if (a=="--planners"&&i+1<argc) planner_names=parse_string_list(argv[++i]);
        else if (a=="--horizon"&&i+1<argc) horizon=max(2,atoi(argv[++i]));
        else if (a=="--override-lp-alpha"&&i+1<argc) override_lp_alpha=(float)atof(argv[++i]);
        else if (a=="--override-soppi-iters"&&i+1<argc) override_soppi_iters=atoi(argv[++i]);
        else if (a=="--override-soppi-neighbors"&&i+1<argc) override_soppi_neighbor_count=max(0,atoi(argv[++i]));
        else if (a=="--override-soppi-step-size"&&i+1<argc) override_soppi_step_size=(float)atof(argv[++i]);
        else if (a=="--override-soppi-bandwidth"&&i+1<argc) override_soppi_bandwidth=(float)atof(argv[++i]);
        else if (a=="--dump-traj"&&i+1<argc) dump_traj_prefix=argv[++i];
        // mechanism diagnostic: per-step K-sample rotation/improvement statistics
        // for vanilla MPPI across tasks (quantifies where sampling is contact-starved).
        else if (a=="--diag-mechanism"&&i+1<argc) diag_prefix=argv[++i];
        // model-mismatch robustness: scale the TRUE plant contact mobility while the
        // controller keeps nominal gains (default 1.0 => published numbers reproduce).
        else if (a=="--plant-gain-scale"&&i+1<argc) plant_gain_scale=(float)atof(argv[++i]);
        // true plant box-size scale vs the controller's nominal size (default 1 => no-op).
        else if (a=="--plant-size-scale"&&i+1<argc) plant_size_scale=(float)atof(argv[++i]);
        // sim-to-sim: run the STRUCTURALLY different hard-contact rigid-body plant as
        // ground truth (controller keeps the smooth model). Default smooth => byte-identical.
        else if (a=="--true-plant"&&i+1<argc) true_plant_hard = (string(argv[++i])=="hard");
        // Coulomb friction of the hard true plant (only used with --true-plant hard).
        else if (a=="--mu"&&i+1<argc) plant_mu=(float)atof(argv[++i]);
        // gradient-agreement capstone: sweep hard-plant friction and log cos(smooth
        // autodiff gradient, true-plant FD sensitivity) at the visited states.
        else if (a=="--grad-agreement"&&i+1<argc) grad_agree_prefix=argv[++i];
    }
    ensure_build_dir();

    // Physics self-test for the hard-contact plant: scripted pushes whose outcome is
    // known a priori, so the rigid-body / friction model is validated independently of
    // any controller. Asserts the qualitative invariants; prints the numbers.
    {
        bool selftest=false; for(int i=1;i<argc;i++) if(string(argv[i])=="--selftest-hard") selftest=true;
        if (selftest) {
            HardParams hp; hp.hx=0.35f; hp.hy=0.18f; hp.push_r=0.08f;
            auto run = [&](float px,float py,float oth0,float ux,float uy,float mu,int n,
                           float& ox,float& oy,float& oth){
                hp.mu=mu; ox=0; oy=0; oth=oth0; float vx=0,vy=0,w=0; float p=px,q=py;
                for(int i=0;i<n;i++) push_step_box_hard_f(p,q,ox,oy,oth,vx,vy,w,ux,uy,hp);
            };
            float ox,oy,oth;
            // (1) centred push on the left face -> translate +x, negligible rotation
            run(-0.42f, 0.0f, 0.0f, +1.0f, 0.0f, 0.5f, 40, ox,oy,oth);
            printf("[selftest] centred push:    ox=%+.3f oy=%+.3f oth=%+.3f  (expect ox>0, |oth| small)\n",ox,oy,oth);
            bool ok1 = ox>0.05f && fabsf(oth)<0.05f;
            // (2) off-centre push (pusher above centre) on left face -> rotates clockwise (oth<0)
            run(-0.42f, +0.12f, 0.0f, +1.0f, 0.0f, 0.5f, 40, ox,oy,oth);
            printf("[selftest] off-centre push: ox=%+.3f oy=%+.3f oth=%+.3f  (expect ox>0, oth<0)\n",ox,oy,oth);
            bool ok2 = ox>0.05f && oth<-0.02f;
            // (3) frictionless tangential drag: pusher slides ALONG the face, mu=0 -> box still
            float ox0,oy0,oth0_, oxm,oym,othm;
            run(-0.40f, 0.0f, 0.0f, 0.0f, +1.0f, 0.0f, 40, ox0,oy0,oth0_);   // mu=0
            run(-0.40f, 0.0f, 0.0f, 0.0f, +1.0f, 0.8f, 40, oxm,oym,othm);    // mu=0.8
            printf("[selftest] tangential mu=0: oy=%+.3f | mu=0.8: oy=%+.3f  (expect ~0 vs >0)\n",oy0,oym);
            bool ok3 = fabsf(oy0)<0.01f && oym>0.02f;   // frictionless: zero drag; friction: nonzero drag
            printf("[selftest] invariants: centred=%s  off-centre-torque=%s  coulomb-friction=%s\n",
                   ok1?"PASS":"FAIL", ok2?"PASS":"FAIL", ok3?"PASS":"FAIL");
            return (ok1&&ok2&&ok3)?0:1;
        }
    }

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
        vector<BoxScenario> diag_sc = { make_box_turn(), make_box_align(), make_box_pivot(), make_box_swivel(), make_box_align_strict(), make_box_align_detour(), make_box_align_contact_loss() };
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

    // Gradient-agreement capstone: explain the sim-to-sim success boundary at the level
    // of gradient DIRECTION. On box_pivot (the tight-tolerance rotation task with the
    // friction boundary), run the paper's Diff-MPPI (diff_mppi_3) against the hard
    // true plant while sweeping Coulomb friction mu. At each visited, contact-engaged
    // state, measure cos between the smooth autodiff gradient the controller descends
    // and the true plant's finite-difference cost sensitivity. The prediction (made
    // before looking): cos is high at low mu (the smooth gradient agrees with what the
    // true plant actually rewards, so the gradient helps) and falls as mu rises (the
    // smooth model omits stick-slip, so its gradient increasingly misleads), tracking
    // the success boundary. We report whatever the curve is -- agreement is the claim,
    // not a guarantee. Seeds reuse the box_pivot/diff_mppi_3 indices (si=2, vi=2) so
    // the trajectories are IDENTICAL to the sim-to-sim experiment.
    if (!grad_agree_prefix.empty()) {
        BoxScenario sc = make_box_pivot();
        vector<float> mus = { 0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f };
        int Kga   = k_values.empty() ? 1024 : k_values.back();
        int nseed = seed_count>0 ? seed_count : 8;
        Variant v; v.name="diff_mppi_3"; v.grad_steps=3; v.alpha=0.010f;
        string path = grad_agree_prefix + "_" + sc.name + ".csv";
        ofstream out(path);
        out << "# scenario=" << sc.name << " planner=diff_mppi_3 K=" << Kga
            << " ang_tol=" << sc.ang_tol << " seeds=" << nseed
            << "  cos(smooth-autodiff-grad, hard-plant-FD-sensitivity) over contact-engaged steps\n";
        out << "mu,cos_full_mean,cos_full_std,cos_first_mean,n_engaged,success_rate,final_ang_mean\n";
        printf("=== gradient-agreement capstone: cos(smooth grad, hard-plant FD sensitivity) vs mu ===\n");
        printf("    scenario=%s planner=diff_mppi_3 K=%d seeds=%d ang_tol=%.3f\n",
               sc.name.c_str(), Kga, nseed, sc.ang_tol);
        for (float mu : mus) {
            vector<float> cosv; double cos1_sum=0; int succ=0; double fang=0;
            for (int seed=0; seed<nseed; seed++) {
                EpisodeRunner runner(v, sc, Kga, horizon, (int)(6000 + 2*100 + 2*20 + seed*7 + Kga));
                runner.true_plant_hard = true; runner.plant_mu = mu;
                runner.record_grad_agree = true;
                EpisodeMetrics m = runner.run();
                succ += m.success; fang += m.min_goal_distance;
                for (size_t k=0; k+6 < runner.grad_agree_flat.size(); k+=7)
                    if (runner.grad_agree_flat[k+6] > 0.5f) {       // engaged
                        cosv.push_back(runner.grad_agree_flat[k+1]); // cos_full
                        cos1_sum += runner.grad_agree_flat[k+2];     // cos_first
                    }
            }
            double mean=0, var=0;
            for (float c : cosv) mean += c; if (!cosv.empty()) mean /= cosv.size();
            for (float c : cosv) var += (c-mean)*(c-mean); if (cosv.size()>1) var /= (cosv.size()-1);
            double cos1mean = cosv.empty()? 0.0 : cos1_sum/cosv.size();
            double sr = (double)succ/nseed, fa = fang/nseed, sd = sqrt(var);
            out << mu << "," << mean << "," << sd << "," << cos1mean << ","
                << cosv.size() << "," << sr << "," << fa << "\n";
            printf("  mu=%.2f  cos_full=%+.3f +/- %.3f  cos_first=%+.3f  n_engaged=%zu  success=%.2f  final_ang=%.3f\n",
                   mu, mean, sd, cos1mean, cosv.size(), sr, fa);
        }
        printf("[grad-agreement] wrote %s\n", path.c_str());
        return 0;
    }

    // box_swivel and box_align_strict are appended LAST so the existing scenarios keep
    // their indices si=0..2 (the per-run seed in the sweep loop is si-dependent);
    // published numbers stay byte-identical.
    vector<BoxScenario> all_sc = { make_box_turn(), make_box_align(), make_box_pivot(), make_box_swivel(), make_box_align_strict(), make_box_align_detour(), make_box_align_contact_loss(), make_box_align_contact_arc() };
    vector<BoxScenario> scenarios;
    if (!scenario_names.empty()) {
        for (auto& w : scenario_names) { auto it=find_if(all_sc.begin(),all_sc.end(),[&](const BoxScenario&s){return s.name==w;});
            if (it==all_sc.end()){fprintf(stderr,"Unknown scenario: %s\n",w.c_str());return 1;} scenarios.push_back(*it); }
    } else scenarios = all_sc;

    vector<Variant> variants;
    { Variant v; v.name="mppi"; variants.push_back(v); }
    { Variant v; v.name="lp_mppi"; v.use_low_pass_sampling=true; v.lp_alpha=0.35f; variants.push_back(v); }
    { Variant v; v.name="lp_mppi_smooth"; v.use_low_pass_sampling=true; v.lp_alpha=0.20f; variants.push_back(v); }
    { Variant v; v.name="oi_mppi"; v.use_object_informed=true; v.oi_ref_weight_pos=1.5f; v.oi_ref_weight_ang=3.0f; v.oi_obj_speed=1.2f; v.oi_ang_speed=1.2f; v.oi_seed_blend=0.12f; variants.push_back(v); }
    { Variant v; v.name="oi_lp_mppi"; v.use_object_informed=true; v.use_low_pass_sampling=true; v.lp_alpha=0.25f; v.oi_ref_weight_pos=1.5f; v.oi_ref_weight_ang=3.0f; v.oi_obj_speed=1.2f; v.oi_ang_speed=1.2f; v.oi_seed_blend=0.10f; variants.push_back(v); }
    { Variant v; v.name="diff_mppi_1"; v.grad_steps=1; v.alpha=0.02f; variants.push_back(v); }
    { Variant v; v.name="diff_mppi_3"; v.grad_steps=3; v.alpha=0.010f; variants.push_back(v); }
    { Variant v; v.name="diff_mppi_5"; v.grad_steps=5; v.alpha=0.008f; variants.push_back(v); }
    { Variant v; v.name="soppi"; v.use_soppi_sampling=true; v.soppi_step_size=0.06f; v.soppi_bandwidth=2.0f; variants.push_back(v); }
    { Variant v; v.name="soppi_fast"; v.use_soppi_sampling=true; v.soppi_step_size=0.06f; v.soppi_bandwidth=2.0f; v.soppi_neighbor_count=64; v.soppi_svgd_iters=2; variants.push_back(v); }
    { Variant v; v.name="soppi_g3"; v.use_soppi_sampling=true; v.soppi_step_size=0.06f; v.soppi_bandwidth=2.0f; v.grad_steps=3; v.alpha=0.010f; variants.push_back(v); }
    { Variant v; v.name="soppi_fast_g3"; v.use_soppi_sampling=true; v.soppi_step_size=0.06f; v.soppi_bandwidth=2.0f; v.soppi_neighbor_count=32; v.grad_steps=3; v.alpha=0.010f; variants.push_back(v); }
    // Fidelity arm: vanilla MPPI that ROLLS OUT with the exact hard-contact model (no
    // gradient). Only meaningful with --true-plant hard, where it is the model-exact
    // sampler to beat.
    { Variant v; v.name="mppi_hardmodel"; v.grad_steps=0; v.alpha=0.0f; v.hard_rollout=true; variants.push_back(v); }
    if (!planner_names.empty()) {
        vector<Variant> f; for (auto& w : planner_names){ auto it=find_if(variants.begin(),variants.end(),[&](const Variant&v){return v.name==w;});
            if (it==variants.end()){fprintf(stderr,"Unknown planner: %s\n",w.c_str());return 1;} f.push_back(*it);} variants.swap(f);
    }
    for (auto& v : variants) {
        if (override_lp_alpha >= 0.0f && v.use_low_pass_sampling) v.lp_alpha = override_lp_alpha;
        if (override_soppi_iters >= 0 && v.use_soppi_sampling) v.soppi_svgd_iters = override_soppi_iters;
        if (override_soppi_neighbor_count >= 0 && v.use_soppi_sampling) v.soppi_neighbor_count = override_soppi_neighbor_count;
        if (override_soppi_step_size >= 0.0f && v.use_soppi_sampling) v.soppi_step_size = override_soppi_step_size;
        if (override_soppi_bandwidth >= 0.0f && v.use_soppi_sampling) v.soppi_bandwidth = override_soppi_bandwidth;
    }
    if (k_values.empty()) k_values = quick ? vector<int>{256} : vector<int>{256, 1024};
    if (seed_count<=0) seed_count = quick ? 4 : 8;

    vector<EpisodeMetrics> rows;
    for (size_t si=0; si<scenarios.size(); si++) {
        const BoxScenario& sc = scenarios[si];
        for (int ks : k_values) for (size_t vi=0; vi<variants.size(); vi++) for (int seed=0; seed<seed_count; seed++) {
            int run_seed = (int)(6000 + si*100 + seed*7 + ks);
            EpisodeRunner runner(variants[vi], sc, ks, horizon, run_seed);
            runner.plant_gain_scale = plant_gain_scale;
            runner.plant_size_scale = plant_size_scale;
            runner.true_plant_hard = true_plant_hard;
            runner.plant_mu = plant_mu;
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
