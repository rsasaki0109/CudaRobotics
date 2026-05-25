// gpu_online_slam_3d_switchable.cu
//
// GPU online 3D SLAM with a switchable-constraint sliding-window backend.
//
// This fuses two earlier demos into the harder online setting:
//   - #63 (gpu_online_slam) gave a 2D sliding-window pose-graph optimiser
//     that runs as frames stream in, with an iSAM-style global pass when a
//     loop fires.  But every loop it ever detected was assumed correct.
//   - #98 (gpu_pose_graph_slam_3d_switchable) gave a full SE(3) backend that
//     rejects bad loops with per-loop switch variables (Suenderhauf &
//     Protzel, IROS 2012) -- but as a one-shot BATCH solve over a frozen
//     graph, where all loops are present before any optimisation starts.
//
// The interesting and honest case is the combination of both: a robot
// streams poses, true AND false loop closures arrive incrementally, and the
// back-end has to decide on each loop the moment it appears -- there is no
// chance to look at the whole graph and trim the worst loops after the fact.
// Each frame the switch variables are re-minimised in closed form alongside a
// sliding-window SE(3) Gauss-Newton solve, so a false loop is switched OFF as
// soon as its residual blows up, and a true loop earns its weight back over a
// few iterations.  An occasional global pass propagates a good loop's
// correction across the whole trajectory.
//
// Two paths run in lockstep over the SAME streamed edge set, differing only
// in the back-end:
//   - "plain online"      : every loop weighted 1 (no switches).  False loops
//                           yank the live estimate as they arrive.
//   - "switchable online" : per-loop switches optimised live.  False loops
//                           are rejected the moment they appear.
//
// SE(3) residual for edge i->j:  pred = T_i^-1 T_j,  r_t = pred_t - z_t,
//                                r_R = log(z_R^T pred_R).
// State update:                  t <- t + dt,  R <- Exp(dw) R.
// Jacobians: central finite differences on the same residual.
// Sliding window: poses in [active_lo, active_hi) accept updates; the rest are
// fixed constants, and pose active_lo is pinned as the window anchor.
//
// Output: gif/gpu_online_slam_3d_switchable.gif -- two panels (plain vs
// switchable) growing frame-by-frame, loops coloured by switch weight.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_blas.cuh"
#include "cuda_video.h"

namespace cudabot {

using blas::axpy_kernel;
using blas::copy_kernel;
using blas::dot_kernel;
using blas::xpay_kernel;
using blas::zero_kernel;

constexpr int N_FRAMES = 420;          // total poses streamed
constexpr int WINDOW = 80;             // sliding-window size
constexpr int GN_ITERS = 4;            // GN iters per frame (window)
constexpr int PCG_ITERS = 22;          // PCG iters per GN (window)
constexpr int GLOBAL_GN_ITERS = 8;     // GN iters for a global pass
constexpr int GLOBAL_PCG_ITERS = 50;   // PCG iters for a global pass
constexpr int LOOP_BURST_FRAMES = 18;  // min frames between global passes
constexpr int THREADS = 256;
constexpr float PI_F = 3.14159265358979323846f;

constexpr float ODOM_SIGMA_T = 0.015f;
constexpr float ODOM_SIGMA_R = 0.005f;
constexpr float LOOP_SIGMA_T = 0.012f;
constexpr float LOOP_SIGMA_R = 0.0040f;

constexpr float LOOP_DIST = 0.60f;     // GT proximity for a true loop
constexpr int   LOOP_MIN_GAP = 140;    // need this many frames between i and j
constexpr int   LOOP_PER_FRAME_MAX = 1;
constexpr int   FALSE_START = 90;      // first frame a false loop can appear
constexpr int   FALSE_EVERY = 16;      // inject a false loop this often

constexpr float FD_EPS_T = 1.0e-3f;
constexpr float FD_EPS_R = 2.0e-4f;
constexpr float DAMPING = 5.0e-3f;
constexpr float MAX_DT = 0.60f;        // per-iter SE(3) step clamp (translation)
constexpr float MAX_DW = 0.35f;        // per-iter SE(3) step clamp (rotation)

// Switchable-constraint hyper-parameters (see #98 for the derivation).
constexpr float SWITCH_PRIOR_XI = 8000.0f;
constexpr float SWITCH_DAMP_OFF = 0.85f;   // turn a loop OFF fast
constexpr float SWITCH_DAMP_ON = 0.18f;    // turn a loop back ON slowly
constexpr float SWITCH_REJECT_THRESH = 0.2f;

constexpr int PANEL_W = 470;
constexpr int PANEL_H = 430;
constexpr int TITLE_H = 34;
constexpr int FOOT_H = 72;
constexpr int VIDEO_FPS = 12;
constexpr int VIDEO_STRIDE = 2;
constexpr float PROJ_SCALE = 18.0f;

struct Pose {
    float t[3];
    float R[9];
};

struct Edge {
    int i, j;
    float t[3];
    float R[9];
    float wt, wr;
    float switch_weight;
    int loop;
    int outlier;
};

// -------------------------------------------------------------------------
// SE(3) math helpers (host + device).  Lifted from #98.
// -------------------------------------------------------------------------
__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline void mat3_identity(float* R) {
    R[0] = 1.0f; R[1] = 0.0f; R[2] = 0.0f;
    R[3] = 0.0f; R[4] = 1.0f; R[5] = 0.0f;
    R[6] = 0.0f; R[7] = 0.0f; R[8] = 1.0f;
}

__host__ __device__ static inline void mat3_mul(const float* A, const float* B, float* C) {
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float v = 0.0f;
            for (int k = 0; k < 3; k++) v += A[3 * r + k] * B[3 * k + c];
            C[3 * r + c] = v;
        }
    }
}

__host__ __device__ static inline void mat3_transpose_mul(const float* A,
                                                          const float* B,
                                                          float* C) {
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float v = 0.0f;
            for (int k = 0; k < 3; k++) v += A[3 * k + r] * B[3 * k + c];
            C[3 * r + c] = v;
        }
    }
}

__host__ __device__ static inline void mat3_transpose_vec(const float* R,
                                                          const float* v,
                                                          float* out) {
    out[0] = R[0] * v[0] + R[3] * v[1] + R[6] * v[2];
    out[1] = R[1] * v[0] + R[4] * v[1] + R[7] * v[2];
    out[2] = R[2] * v[0] + R[5] * v[1] + R[8] * v[2];
}

__host__ __device__ static inline void mat3_vec(const float* R, const float* v, float* out) {
    out[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
    out[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
    out[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
}

__host__ __device__ static inline void so3_exp(const float* w, float* R) {
    float theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    float theta = sqrtf(theta2);
    float A = 1.0f;
    float B = 0.5f;
    if (theta > 1.0e-7f) {
        A = sinf(theta) / theta;
        B = (1.0f - cosf(theta)) / theta2;
    }
    float K[9] = {
        0.0f, -w[2],  w[1],
        w[2],  0.0f, -w[0],
       -w[1],  w[0],  0.0f
    };
    float K2[9];
    mat3_mul(K, K, K2);
    mat3_identity(R);
    for (int k = 0; k < 9; k++) R[k] += A * K[k] + B * K2[k];
}

__host__ __device__ static inline void so3_log(const float* R, float* w) {
    float cos_theta = clampf((R[0] + R[4] + R[8] - 1.0f) * 0.5f, -1.0f, 1.0f);
    float theta = acosf(cos_theta);
    if (theta < 1.0e-6f) {
        w[0] = 0.5f * (R[7] - R[5]);
        w[1] = 0.5f * (R[2] - R[6]);
        w[2] = 0.5f * (R[3] - R[1]);
        return;
    }
    float scale = theta / (2.0f * sinf(theta));
    w[0] = scale * (R[7] - R[5]);
    w[1] = scale * (R[2] - R[6]);
    w[2] = scale * (R[3] - R[1]);
}

__host__ __device__ static inline void pose_relative(const Pose& a,
                                                     const Pose& b,
                                                     float* rel_t,
                                                     float* rel_R) {
    float dt[3] = {b.t[0] - a.t[0], b.t[1] - a.t[1], b.t[2] - a.t[2]};
    mat3_transpose_vec(a.R, dt, rel_t);
    mat3_transpose_mul(a.R, b.R, rel_R);
}

__host__ __device__ static inline void residual_edge(const Pose& pi,
                                                     const Pose& pj,
                                                     const float* et,
                                                     const float* eR,
                                                     float* r) {
    float pred_t[3];
    float pred_R[9];
    pose_relative(pi, pj, pred_t, pred_R);
    r[0] = pred_t[0] - et[0];
    r[1] = pred_t[1] - et[1];
    r[2] = pred_t[2] - et[2];
    float Rerr[9];
    mat3_transpose_mul(eR, pred_R, Rerr);
    so3_log(Rerr, r + 3);
}

__host__ __device__ static inline float edge_chi2(float wt, float wr, const float* r) {
    float trans = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    float rot = r[3] * r[3] + r[4] * r[4] + r[5] * r[5];
    return wt * trans + wr * rot;
}

__host__ __device__ static inline float switch_target(float chi2) {
    return clampf(1.0f - chi2 / (2.0f * SWITCH_PRIOR_XI), 0.0f, 1.0f);
}

__host__ __device__ static inline float switch_step(float prev, float target) {
    float damp = (target < prev) ? SWITCH_DAMP_OFF : SWITCH_DAMP_ON;
    return prev + damp * (target - prev);
}

__host__ __device__ static inline float edge_weight_scale(int loop,
                                                          int robust_enabled,
                                                          float switch_weight) {
    if (!robust_enabled || !loop) return 1.0f;
    return clampf(switch_weight, 0.0f, 1.0f);
}

__host__ __device__ static inline void perturb_pose(const Pose& in, int axis,
                                                    float eps, Pose& out) {
    out = in;
    if (axis < 3) {
        out.t[axis] += eps;
        return;
    }
    float w[3] = {0.0f, 0.0f, 0.0f};
    w[axis - 3] = eps;
    float E[9], Rnew[9];
    so3_exp(w, E);
    mat3_mul(E, in.R, Rnew);
    for (int k = 0; k < 9; k++) out.R[k] = Rnew[k];
}

static void euler_zyx(float yaw, float pitch, float roll, float* R) {
    float cy = std::cos(yaw), sy = std::sin(yaw);
    float cp = std::cos(pitch), sp = std::sin(pitch);
    float cr = std::cos(roll), sr = std::sin(roll);
    R[0] = cy * cp;
    R[1] = cy * sp * sr - sy * cr;
    R[2] = cy * sp * cr + sy * sr;
    R[3] = sy * cp;
    R[4] = sy * sp * sr + cy * cr;
    R[5] = sy * sp * cr - cy * sr;
    R[6] = -sp;
    R[7] = cp * sr;
    R[8] = cp * cr;
}

// -------------------------------------------------------------------------
// Device load helpers
// -------------------------------------------------------------------------
__device__ static inline void load_pose_device(const float* poses, int idx, Pose& p) {
    p.t[0] = poses[12 * idx + 0];
    p.t[1] = poses[12 * idx + 1];
    p.t[2] = poses[12 * idx + 2];
    for (int k = 0; k < 9; k++) p.R[k] = poses[12 * idx + 3 + k];
}

// -------------------------------------------------------------------------
// Sliding-window aware kernels.  An edge is processed when at least one of
// its endpoints lies in [active_lo, active_hi); contributions to fixed poses
// are skipped.  Lifts the SE(3) assembly of #98 into the windowed scheme
// of #63.
// -------------------------------------------------------------------------
__global__ void linearize_fd_kernel(int n_edges,
                                    const int* __restrict__ ei,
                                    const int* __restrict__ ej,
                                    const float* __restrict__ et,
                                    const float* __restrict__ eR,
                                    const float* __restrict__ poses,
                                    int active_lo, int active_hi,
                                    float* __restrict__ residuals,
                                    float* __restrict__ Ji_all,
                                    float* __restrict__ Jj_all) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    int i = ei[e], j = ej[e];
    bool ia = (i >= active_lo && i < active_hi);
    bool ja = (j >= active_lo && j < active_hi);
    if (!ia && !ja) return;

    const float* edge_t = et + 3 * e;
    const float* edge_R = eR + 9 * e;
    Pose pi, pj;
    load_pose_device(poses, i, pi);
    load_pose_device(poses, j, pj);

    float base[6];
    residual_edge(pi, pj, edge_t, edge_R, base);
    for (int r = 0; r < 6; r++) residuals[6 * e + r] = base[r];

    for (int axis = 0; axis < 6; axis++) {
        float eps = axis < 3 ? FD_EPS_T : FD_EPS_R;
        Pose pp, pm;
        float rp[6], rm[6];
        perturb_pose(pi, axis, eps, pp);
        perturb_pose(pi, axis, -eps, pm);
        residual_edge(pp, pj, edge_t, edge_R, rp);
        residual_edge(pm, pj, edge_t, edge_R, rm);
        for (int r = 0; r < 6; r++)
            Ji_all[e * 36 + r * 6 + axis] = (rp[r] - rm[r]) / (2.0f * eps);
        perturb_pose(pj, axis, eps, pp);
        perturb_pose(pj, axis, -eps, pm);
        residual_edge(pi, pp, edge_t, edge_R, rp);
        residual_edge(pi, pm, edge_t, edge_R, rm);
        for (int r = 0; r < 6; r++)
            Jj_all[e * 36 + r * 6 + axis] = (rp[r] - rm[r]) / (2.0f * eps);
    }
}

// Closed-form, damped switch update from the freshly-linearised residuals.
// Loops that no longer touch the active window keep their last weight.
__global__ void update_switch_kernel(int n_edges,
                                     const int* __restrict__ ei,
                                     const int* __restrict__ ej,
                                     const int* __restrict__ eloop,
                                     const float* __restrict__ ew,
                                     const float* __restrict__ residuals,
                                     int active_lo, int active_hi,
                                     float* __restrict__ eswitch) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    if (!eloop[e]) {
        eswitch[e] = 1.0f;
        return;
    }
    int i = ei[e], j = ej[e];
    bool touch = (i >= active_lo && i < active_hi) || (j >= active_lo && j < active_hi);
    if (!touch) return;
    const float* r = residuals + 6 * e;
    float chi2 = edge_chi2(ew[2 * e + 0], ew[2 * e + 1], r);
    float target = switch_target(chi2);
    eswitch[e] = switch_step(eswitch[e], target);
}

__global__ void assemble_kernel(int n_edges,
                                const int* __restrict__ ei,
                                const int* __restrict__ ej,
                                const int* __restrict__ eloop,
                                const float* __restrict__ eswitch,
                                const float* __restrict__ ew,
                                const float* __restrict__ residuals,
                                const float* __restrict__ Ji_all,
                                const float* __restrict__ Jj_all,
                                int robust_enabled,
                                int active_lo, int active_hi,
                                float* __restrict__ b,
                                float* __restrict__ diag) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    int i = ei[e], j = ej[e];
    bool ia = (i >= active_lo && i < active_hi);
    bool ja = (j >= active_lo && j < active_hi);
    if (!ia && !ja) return;
    const float* r = residuals + 6 * e;
    float scale = edge_weight_scale(eloop[e], robust_enabled, eswitch[e]);
    float wt[6] = {
        scale * ew[2 * e + 0], scale * ew[2 * e + 0], scale * ew[2 * e + 0],
        scale * ew[2 * e + 1], scale * ew[2 * e + 1], scale * ew[2 * e + 1],
    };
    const float* Ji = Ji_all + 36 * e;
    const float* Jj = Jj_all + 36 * e;

    for (int c = 0; c < 6; c++) {
        float bi = 0.0f, bj = 0.0f;
        for (int rr = 0; rr < 6; rr++) {
            float wr = wt[rr] * r[rr];
            bi += Ji[rr * 6 + c] * wr;
            bj += Jj[rr * 6 + c] * wr;
        }
        if (ia) atomicAdd(&b[6 * i + c], bi);
        if (ja) atomicAdd(&b[6 * j + c], bj);
    }
    for (int a = 0; a < 6; a++) {
        for (int bcol = 0; bcol < 6; bcol++) {
            float vii = 0.0f, vjj = 0.0f;
            for (int rr = 0; rr < 6; rr++) {
                vii += Ji[rr * 6 + a] * wt[rr] * Ji[rr * 6 + bcol];
                vjj += Jj[rr * 6 + a] * wt[rr] * Jj[rr * 6 + bcol];
            }
            if (ia) atomicAdd(&diag[36 * i + 6 * a + bcol], vii);
            if (ja) atomicAdd(&diag[36 * j + 6 * a + bcol], vjj);
        }
    }
}

__global__ void matvec_kernel(int n_edges,
                              const int* __restrict__ ei,
                              const int* __restrict__ ej,
                              const int* __restrict__ eloop,
                              const float* __restrict__ eswitch,
                              const float* __restrict__ ew,
                              const float* __restrict__ Ji_all,
                              const float* __restrict__ Jj_all,
                              int robust_enabled,
                              int active_lo, int active_hi,
                              const float* __restrict__ x,
                              float* __restrict__ y) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    int i = ei[e], j = ej[e];
    bool ia = (i >= active_lo && i < active_hi);
    bool ja = (j >= active_lo && j < active_hi);
    if (!ia && !ja) return;
    float scale = edge_weight_scale(eloop[e], robust_enabled, eswitch[e]);
    float wt[6] = {
        scale * ew[2 * e + 0], scale * ew[2 * e + 0], scale * ew[2 * e + 0],
        scale * ew[2 * e + 1], scale * ew[2 * e + 1], scale * ew[2 * e + 1],
    };
    const float* Ji = Ji_all + 36 * e;
    const float* Jj = Jj_all + 36 * e;
    float xi[6], xj[6];
    for (int k = 0; k < 6; k++) {
        xi[k] = ia ? x[6 * i + k] : 0.0f;
        xj[k] = ja ? x[6 * j + k] : 0.0f;
    }
    float u[6];
    for (int r = 0; r < 6; r++) {
        float v = 0.0f;
        for (int c = 0; c < 6; c++) {
            v += Ji[r * 6 + c] * xi[c];
            v += Jj[r * 6 + c] * xj[c];
        }
        u[r] = wt[r] * v;
    }
    for (int c = 0; c < 6; c++) {
        float yi = 0.0f, yj = 0.0f;
        for (int r = 0; r < 6; r++) {
            yi += Ji[r * 6 + c] * u[r];
            yj += Jj[r * 6 + c] * u[r];
        }
        if (ia) atomicAdd(&y[6 * i + c], yi);
        if (ja) atomicAdd(&y[6 * j + c], yj);
    }
}

__global__ void add_damping_kernel(int n, float damping, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] += damping * x[idx];
}

// Pin pose `idx`: zero its gradient row and force an identity Hessian block.
__global__ void anchor_pose_kernel(float* b, float* diag, int idx) {
    int k = threadIdx.x;
    if (k < 6) b[6 * idx + k] = 0.0f;
    if (k < 36) diag[36 * idx + k] = 0.0f;
    __syncthreads();
    if (k < 6) diag[36 * idx + 6 * k + k] = 1.0f;
}

__global__ void zero_anchor_kernel(float* x, int idx) {
    int k = threadIdx.x;
    if (k < 6) x[6 * idx + k] = 0.0f;
}

__device__ static bool solve6_spd_device(const float* A_in, const float* rhs,
                                         float damping, float* out) {
    float A[36], L[36];
    for (int i = 0; i < 36; i++) { A[i] = A_in[i]; L[i] = 0.0f; }
    for (int i = 0; i < 6; i++) A[6 * i + i] += damping;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j <= i; j++) {
            float s = A[6 * i + j];
            for (int k = 0; k < j; k++) s -= L[6 * i + k] * L[6 * j + k];
            if (i == j) {
                if (s <= 1.0e-12f) return false;
                L[6 * i + j] = sqrtf(s);
            } else {
                L[6 * i + j] = s / L[6 * j + j];
            }
        }
    }
    float y[6];
    for (int i = 0; i < 6; i++) {
        float s = rhs[i];
        for (int k = 0; k < i; k++) s -= L[6 * i + k] * y[k];
        y[i] = s / L[6 * i + i];
    }
    for (int i = 5; i >= 0; i--) {
        float s = y[i];
        for (int k = i + 1; k < 6; k++) s -= L[6 * k + i] * out[k];
        out[i] = s / L[6 * i + i];
    }
    return true;
}

__global__ void apply_precond_kernel(int n_poses,
                                     int active_lo, int active_hi,
                                     const float* __restrict__ diag,
                                     const float* __restrict__ r,
                                     float damping,
                                     float* __restrict__ z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_poses) return;
    if (i < active_lo || i >= active_hi) {
        for (int k = 0; k < 6; k++) z[6 * i + k] = 0.0f;
        return;
    }
    float rhs[6], sol[6] = {0, 0, 0, 0, 0, 0};
    for (int k = 0; k < 6; k++) rhs[k] = r[6 * i + k];
    bool ok = solve6_spd_device(diag + 36 * i, rhs, damping, sol);
    if (!ok) {
        for (int k = 0; k < 6; k++) {
            float d = diag[36 * i + 6 * k + k] + damping;
            sol[k] = rhs[k] / fmaxf(d, 1.0e-6f);
        }
    }
    for (int k = 0; k < 6; k++) z[6 * i + k] = sol[k];
}

// SE(3) state update over the active window, with a per-iteration step clamp
// for stability (no host-side line search in the streaming loop).
__global__ void update_poses_kernel(int n_poses,
                                    int active_lo, int active_hi,
                                    float* __restrict__ poses,
                                    const float* __restrict__ dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_poses) return;
    if (i < active_lo || i >= active_hi) return;
    float dt[3], dw[3];
    for (int k = 0; k < 3; k++) dt[k] = clampf(dx[6 * i + k], -MAX_DT, MAX_DT);
    for (int k = 0; k < 3; k++) dw[k] = clampf(dx[6 * i + 3 + k], -MAX_DW, MAX_DW);
    poses[12 * i + 0] += dt[0];
    poses[12 * i + 1] += dt[1];
    poses[12 * i + 2] += dt[2];
    float Rold[9], E[9], Rnew[9];
    for (int k = 0; k < 9; k++) Rold[k] = poses[12 * i + 3 + k];
    so3_exp(dw, E);
    mat3_mul(E, Rold, Rnew);
    for (int k = 0; k < 9; k++) poses[12 * i + 3 + k] = Rnew[k];
}

// -------------------------------------------------------------------------
// GPU solver context: holds device buffers and runs one windowed (or global)
// GN+PCG solve on a given pose buffer.
// -------------------------------------------------------------------------
struct Solver {
    int max_edges = 0;
    int *d_ei = nullptr, *d_ej = nullptr, *d_eloop = nullptr;
    float *d_et = nullptr, *d_eR = nullptr, *d_ew = nullptr, *d_eswitch = nullptr;
    float *d_residuals = nullptr, *d_Ji = nullptr, *d_Jj = nullptr;
    float *d_b = nullptr, *d_diag = nullptr, *d_dx = nullptr;
    float *d_r = nullptr, *d_z = nullptr, *d_p = nullptr, *d_Ap = nullptr, *d_scratch = nullptr;

    void alloc(int max_e) {
        max_edges = max_e;
        const int n_state = N_FRAMES * 6;
        CUDA_CHECK(cudaMalloc(&d_ei, max_e * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ej, max_e * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_eloop, max_e * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_et, max_e * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_eR, max_e * 9 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ew, max_e * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_eswitch, max_e * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_residuals, max_e * 6 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Ji, max_e * 36 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Jj, max_e * 36 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, n_state * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_diag, N_FRAMES * 36 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dx, n_state * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_r, n_state * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_z, n_state * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_p, n_state * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Ap, n_state * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scratch, sizeof(float)));
    }

    void free_all() {
        cudaFree(d_ei); cudaFree(d_ej); cudaFree(d_eloop);
        cudaFree(d_et); cudaFree(d_eR); cudaFree(d_ew); cudaFree(d_eswitch);
        cudaFree(d_residuals); cudaFree(d_Ji); cudaFree(d_Jj);
        cudaFree(d_b); cudaFree(d_diag); cudaFree(d_dx);
        cudaFree(d_r); cudaFree(d_z); cudaFree(d_p); cudaFree(d_Ap); cudaFree(d_scratch);
    }

    float dot(int n, const float* a, const float* b) {
        float out = 0.0f;
        CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
        dot_kernel<<<32, 256>>>(n, a, b, d_scratch);
        CUDA_CHECK(cudaMemcpy(&out, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
        return out;
    }

    // One full GN+PCG solve over [active_lo, active_hi) on d_poses.
    void solve(float* d_poses, int robust, int active_lo, int active_hi,
               int n_edges, int gn_iters, int pcg_iters) {
        const int n_state = N_FRAMES * 6;
        const int anchor = active_lo;
        int be = (n_edges + THREADS - 1) / THREADS;
        int bs = (n_state + THREADS - 1) / THREADS;
        int bd = (N_FRAMES * 36 + THREADS - 1) / THREADS;
        int bp = (N_FRAMES + THREADS - 1) / THREADS;

        for (int gn = 0; gn < gn_iters; gn++) {
            linearize_fd_kernel<<<be, THREADS>>>(n_edges, d_ei, d_ej, d_et, d_eR,
                                                 d_poses, active_lo, active_hi,
                                                 d_residuals, d_Ji, d_Jj);
            if (robust) {
                update_switch_kernel<<<be, THREADS>>>(n_edges, d_ei, d_ej, d_eloop,
                                                      d_ew, d_residuals,
                                                      active_lo, active_hi, d_eswitch);
            }
            zero_kernel<<<bs, THREADS>>>(n_state, d_b);
            zero_kernel<<<bd, THREADS>>>(N_FRAMES * 36, d_diag);
            assemble_kernel<<<be, THREADS>>>(n_edges, d_ei, d_ej, d_eloop, d_eswitch,
                                             d_ew, d_residuals, d_Ji, d_Jj, robust,
                                             active_lo, active_hi, d_b, d_diag);
            anchor_pose_kernel<<<1, 36>>>(d_b, d_diag, anchor);

            // PCG: solve (H + lambda I) dx = -b on the active window.
            zero_kernel<<<bs, THREADS>>>(n_state, d_dx);
            zero_kernel<<<bs, THREADS>>>(n_state, d_r);
            axpy_kernel<<<bs, THREADS>>>(n_state, -1.0f, d_b, d_r);
            zero_anchor_kernel<<<1, 6>>>(d_r, anchor);
            apply_precond_kernel<<<bp, THREADS>>>(N_FRAMES, active_lo, active_hi,
                                                  d_diag, d_r, DAMPING, d_z);
            zero_anchor_kernel<<<1, 6>>>(d_z, anchor);
            copy_kernel<<<bs, THREADS>>>(n_state, d_z, d_p);

            float rz_old = dot(n_state, d_r, d_z);
            float rr0 = fmaxf(dot(n_state, d_r, d_r), 1.0e-12f);
            if (rz_old <= 0.0f) { /* nothing to do */ }

            for (int pcg = 0; pcg < pcg_iters; pcg++) {
                zero_kernel<<<bs, THREADS>>>(n_state, d_Ap);
                matvec_kernel<<<be, THREADS>>>(n_edges, d_ei, d_ej, d_eloop, d_eswitch,
                                               d_ew, d_Ji, d_Jj, robust,
                                               active_lo, active_hi, d_p, d_Ap);
                add_damping_kernel<<<bs, THREADS>>>(n_state, DAMPING, d_p, d_Ap);
                zero_anchor_kernel<<<1, 6>>>(d_Ap, anchor);
                float pAp = dot(n_state, d_p, d_Ap);
                if (pAp <= 1.0e-20f) break;
                float alpha = rz_old / pAp;
                axpy_kernel<<<bs, THREADS>>>(n_state, alpha, d_p, d_dx);
                axpy_kernel<<<bs, THREADS>>>(n_state, -alpha, d_Ap, d_r);
                zero_anchor_kernel<<<1, 6>>>(d_r, anchor);
                float rr = dot(n_state, d_r, d_r);
                if (rr < rr0 * 1.0e-7f) break;
                apply_precond_kernel<<<bp, THREADS>>>(N_FRAMES, active_lo, active_hi,
                                                      d_diag, d_r, DAMPING, d_z);
                zero_anchor_kernel<<<1, 6>>>(d_z, anchor);
                float rz_new = dot(n_state, d_r, d_z);
                float beta = rz_new / fmaxf(1.0e-20f, rz_old);
                xpay_kernel<<<bs, THREADS>>>(n_state, beta, d_z, d_p);
                zero_anchor_kernel<<<1, 6>>>(d_p, anchor);
                rz_old = rz_new;
            }
            update_poses_kernel<<<bp, THREADS>>>(N_FRAMES, active_lo, active_hi,
                                                 d_poses, d_dx);
        }
    }
};

// -------------------------------------------------------------------------
// Host trajectory + edge construction
// -------------------------------------------------------------------------
static std::vector<Pose> make_ground_truth() {
    // Two coincident laps of a 3D figure: every point in lap 2 revisits the
    // matching lap-1 point, producing genuine SE(3) loop closures.
    std::vector<Pose> gt(N_FRAMES);
    for (int i = 0; i < N_FRAMES; i++) {
        float s = static_cast<float>(i) / N_FRAMES;
        float u = 4.0f * PI_F * s;
        float x = 7.0f * std::cos(u) + 1.1f * std::cos(3.0f * u);
        float y = 5.2f * std::sin(u);
        float z = 1.8f + 0.9f * std::sin(2.0f * u);
        float dx = -7.0f * std::sin(u) - 3.3f * std::sin(3.0f * u);
        float dy = 5.2f * std::cos(u);
        float dz = 1.8f * std::cos(2.0f * u);
        float yaw = std::atan2(dy, dx);
        float pitch = std::atan2(-dz, std::sqrt(dx * dx + dy * dy));
        float roll = 0.18f * std::sin(1.5f * u);
        gt[i].t[0] = x; gt[i].t[1] = y; gt[i].t[2] = z;
        euler_zyx(yaw, pitch, roll, gt[i].R);
    }
    return gt;
}

static void add_noise_to_edge(Edge& e, float st, float sr, std::mt19937& rng) {
    std::normal_distribution<float> nt(0.0f, st);
    std::normal_distribution<float> nr(0.0f, sr);
    for (int k = 0; k < 3; k++) e.t[k] += nt(rng);
    float w[3] = {nr(rng), nr(rng), nr(rng)};
    float E[9], Rnew[9];
    so3_exp(w, E);
    mat3_mul(E, e.R, Rnew);
    for (int k = 0; k < 9; k++) e.R[k] = Rnew[k];
}

static Edge make_edge(const std::vector<Pose>& gt, int i, int j,
                      float st, float sr, bool loop, std::mt19937& rng) {
    Edge e{};
    e.i = i; e.j = j;
    pose_relative(gt[i], gt[j], e.t, e.R);
    add_noise_to_edge(e, st, sr, rng);
    e.wt = 1.0f / (st * st);
    e.wr = 1.0f / (sr * sr);
    e.switch_weight = 1.0f;
    e.loop = loop ? 1 : 0;
    e.outlier = 0;
    return e;
}

static Edge make_false_loop(const std::vector<Pose>& gt, int i, int j,
                            int k, std::mt19937& rng) {
    Edge e = make_edge(gt, i, j, LOOP_SIGMA_T, LOOP_SIGMA_R, true, rng);
    float phase = 0.73f * static_cast<float>(k) + 0.31f;
    e.t[0] += 5.20f + 1.35f * std::sin(phase);
    e.t[1] += -4.15f + 1.10f * std::cos(1.7f * phase);
    e.t[2] += 1.85f + 0.90f * std::sin(0.6f * phase + 0.4f);
    float w[3] = {
        0.48f * std::sin(0.9f * phase),
        -0.62f + 0.16f * std::cos(phase),
        0.72f * std::sin(1.3f * phase + 0.2f),
    };
    float E[9], Rnew[9];
    so3_exp(w, E);
    mat3_mul(E, e.R, Rnew);
    for (int r = 0; r < 9; r++) e.R[r] = Rnew[r];
    e.outlier = 1;
    return e;
}

static void compose(const Pose& prev, const Edge& odom, Pose& out) {
    float Rt[3];
    mat3_vec(prev.R, odom.t, Rt);
    out.t[0] = prev.t[0] + Rt[0];
    out.t[1] = prev.t[1] + Rt[1];
    out.t[2] = prev.t[2] + Rt[2];
    mat3_mul(prev.R, odom.R, out.R);
}

static float rmse_translation(const std::vector<Pose>& poses,
                              const std::vector<Pose>& gt, int up_to) {
    double sum = 0.0;
    for (int i = 0; i <= up_to; i++) {
        double dx = poses[i].t[0] - gt[i].t[0];
        double dy = poses[i].t[1] - gt[i].t[1];
        double dz = poses[i].t[2] - gt[i].t[2];
        sum += dx * dx + dy * dy + dz * dz;
    }
    return static_cast<float>(std::sqrt(sum / (up_to + 1)));
}

// -------------------------------------------------------------------------
// Visualisation
// -------------------------------------------------------------------------
static cv::Point2i project(float x, float y, float z, float yaw, float pitch) {
    float cy = std::cos(yaw), sy = std::sin(yaw);
    float cp = std::cos(pitch), sp = std::sin(pitch);
    float x1 = cy * x + sy * y;
    float y1 = -sy * x + cy * y;
    float z1 = z - 1.7f;
    float y2 = cp * y1 - sp * z1;
    float z2 = sp * y1 + cp * z1;
    return cv::Point2i(PANEL_W / 2 + static_cast<int>(PROJ_SCALE * x1),
                       static_cast<int>(PANEL_H * 0.60f) -
                       static_cast<int>(PROJ_SCALE * z2 + 0.20f * PROJ_SCALE * y2));
}

static cv::Scalar switch_color(float s) {
    s = clampf(s, 0.0f, 1.0f);
    return cv::Scalar(70.0f, 90.0f + 130.0f * s, 70.0f + 175.0f * (1.0f - s));
}

static void draw_grid(cv::Mat& img, float yaw, float pitch) {
    for (int g = -8; g <= 8; g += 2) {
        cv::line(img, project(-9, g, 0, yaw, pitch), project(9, g, 0, yaw, pitch),
                 cv::Scalar(44, 47, 54), 1, cv::LINE_AA);
        cv::line(img, project(g, -7, 0, yaw, pitch), project(g, 7, 0, yaw, pitch),
                 cv::Scalar(44, 47, 54), 1, cv::LINE_AA);
    }
}

static cv::Mat draw_panel(const std::vector<Pose>& poses, int up_to,
                          const std::vector<Pose>& gt,
                          const std::vector<Edge>& edges,
                          const std::vector<float>& eswitch,
                          bool switchable,
                          const char* title, float rmse, int yaw_seed) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(24, 25, 30));
    float yaw = -0.72f + 0.004f * yaw_seed;
    float pitch = 0.56f;
    draw_grid(img, yaw, pitch);

    // GT (faint) over the whole trajectory.
    for (int k = 1; k < N_FRAMES; k++)
        cv::line(img, project(gt[k - 1].t[0], gt[k - 1].t[1], gt[k - 1].t[2], yaw, pitch),
                 project(gt[k].t[0], gt[k].t[1], gt[k].t[2], yaw, pitch),
                 cv::Scalar(95, 95, 95), 1, cv::LINE_AA);

    // Loop closures up to now.
    for (const Edge& e : edges) {
        if (!e.loop || e.i > up_to || e.j > up_to) continue;
        cv::Scalar color;
        int thickness;
        if (switchable) {
            float s = (static_cast<size_t>(&e - &edges[0]) < eswitch.size())
                          ? eswitch[&e - &edges[0]] : 1.0f;
            color = switch_color(s);
            thickness = (s > 0.5f) ? 1 : 2;
        } else {
            color = e.outlier ? cv::Scalar(70, 120, 235) : cv::Scalar(190, 190, 70);
            thickness = 1;
        }
        cv::line(img, project(poses[e.i].t[0], poses[e.i].t[1], poses[e.i].t[2], yaw, pitch),
                 project(poses[e.j].t[0], poses[e.j].t[1], poses[e.j].t[2], yaw, pitch),
                 color, thickness, cv::LINE_AA);
    }

    // Current estimate.
    cv::Scalar traj = switchable ? cv::Scalar(70, 210, 255) : cv::Scalar(120, 165, 250);
    for (int k = 1; k <= up_to; k++)
        cv::line(img, project(poses[k - 1].t[0], poses[k - 1].t[1], poses[k - 1].t[2], yaw, pitch),
                 project(poses[k].t[0], poses[k].t[1], poses[k].t[2], yaw, pitch),
                 traj, 2, cv::LINE_AA);
    cv::circle(img, project(poses[up_to].t[0], poses[up_to].t[1], poses[up_to].t[2], yaw, pitch),
               4, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);

    cv::putText(img, title, cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX,
                0.52, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
    cv::putText(img, cv::format("RMSE %.3f m", rmse), cv::Point(12, PANEL_H - 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                switchable ? cv::Scalar(90, 225, 135) : cv::Scalar(120, 165, 250),
                1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<Pose> gt = make_ground_truth();

    std::mt19937 rng(20260525);
    // Pre-compute odometry between consecutive GT poses.
    std::vector<Edge> odom;
    odom.reserve(N_FRAMES - 1);
    for (int k = 0; k < N_FRAMES - 1; k++)
        odom.push_back(make_edge(gt, k, k + 1, ODOM_SIGMA_T, ODOM_SIGMA_R, false, rng));

    // Two estimates, both anchored at GT pose 0.
    std::vector<Pose> plain_h(N_FRAMES), switch_h(N_FRAMES);
    plain_h[0] = gt[0];
    switch_h[0] = gt[0];

    int max_edges = (N_FRAMES - 1) + 4 * N_FRAMES;
    Solver solver;
    solver.alloc(max_edges);

    float *d_poses_plain = nullptr, *d_poses_switch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_poses_plain, N_FRAMES * 12 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_poses_switch, N_FRAMES * 12 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_poses_plain, 0, N_FRAMES * 12 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_poses_switch, 0, N_FRAMES * 12 * sizeof(float)));
    auto upload_pose = [&](float* d_poses, const Pose& p, int idx) {
        float flat[12];
        flat[0] = p.t[0]; flat[1] = p.t[1]; flat[2] = p.t[2];
        for (int k = 0; k < 9; k++) flat[3 + k] = p.R[k];
        CUDA_CHECK(cudaMemcpy(d_poses + 12 * idx, flat, 12 * sizeof(float),
                              cudaMemcpyHostToDevice));
    };
    upload_pose(d_poses_plain, plain_h[0], 0);
    upload_pose(d_poses_switch, switch_h[0], 0);

    // Host-side mirror of the edge set; appended as frames stream in.
    std::vector<Edge> edges;
    edges.reserve(max_edges);
    std::vector<float> eswitch_view;  // current device switch weights (switch path)

    if (std::system("mkdir -p gif") != 0) std::fprintf(stderr, "mkdir gif failed\n");
    const int frame_w = PANEL_W * 2 + 6;
    const int frame_h = TITLE_H + PANEL_H + FOOT_H;
    cv::VideoWriter video("gif/gpu_online_slam_3d_switchable.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(frame_w, frame_h));

    int last_global = -10000;
    int n_true_loops = 0, n_false_loops = 0, n_global_passes = 0;
    double total_step_ms = 0.0;
    int counted = 0;
    int next_false = FALSE_START;

    for (int t = 1; t < N_FRAMES; t++) {
        // 1. Dead-reckoning prediction for the new pose (both paths).
        compose(plain_h[t - 1], odom[t - 1], plain_h[t]);
        compose(switch_h[t - 1], odom[t - 1], switch_h[t]);
        upload_pose(d_poses_plain, plain_h[t], t);
        upload_pose(d_poses_switch, switch_h[t], t);

        // 2. Append the odom edge.
        int old_n = static_cast<int>(edges.size());
        edges.push_back(odom[t - 1]);

        bool loop_fired = false;
        // 3a. True loop closures from GT proximity (a good detector).
        int added = 0;
        {
            struct Cand { int j; float d2; };
            std::vector<Cand> cands;
            for (int j = 0; j + LOOP_MIN_GAP <= t; j++) {
                float dx = gt[t].t[0] - gt[j].t[0];
                float dy = gt[t].t[1] - gt[j].t[1];
                float dz = gt[t].t[2] - gt[j].t[2];
                float d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < LOOP_DIST * LOOP_DIST) cands.push_back({j, d2});
            }
            std::sort(cands.begin(), cands.end(),
                      [](const Cand& a, const Cand& b) { return a.d2 < b.d2; });
            for (const Cand& c : cands) {
                if (added >= LOOP_PER_FRAME_MAX) break;
                edges.push_back(make_edge(gt, c.j, t, LOOP_SIGMA_T, LOOP_SIGMA_R, true, rng));
                added++; n_true_loops++; loop_fired = true;
            }
        }
        // 3b. False loop closures injected on a schedule (perceptual aliasing).
        if (t >= next_false && t - 50 > 10) {
            std::uniform_int_distribution<int> pick(10, t - 50);
            int j = pick(rng);
            edges.push_back(make_false_loop(gt, j, t, n_false_loops, rng));
            n_false_loops++; loop_fired = true;
            next_false = t + FALSE_EVERY;
        }

        // 4. Upload the newly-appended edges (existing device entries persist,
        //    so the evolving switch weights are preserved).
        int n_edges = static_cast<int>(edges.size());
        int n_new = n_edges - old_n;
        if (n_new > 0) {
            std::vector<int> ei(n_new), ej(n_new), el(n_new);
            std::vector<float> et(n_new * 3), eR(n_new * 9), ew(n_new * 2), es(n_new, 1.0f);
            for (int e = 0; e < n_new; e++) {
                const Edge& E = edges[old_n + e];
                ei[e] = E.i; ej[e] = E.j; el[e] = E.loop;
                for (int k = 0; k < 3; k++) et[3 * e + k] = E.t[k];
                for (int k = 0; k < 9; k++) eR[9 * e + k] = E.R[k];
                ew[2 * e + 0] = E.wt; ew[2 * e + 1] = E.wr;
            }
            CUDA_CHECK(cudaMemcpy(solver.d_ei + old_n, ei.data(), n_new * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(solver.d_ej + old_n, ej.data(), n_new * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(solver.d_eloop + old_n, el.data(), n_new * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(solver.d_et + old_n * 3, et.data(), n_new * 3 * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(solver.d_eR + old_n * 9, eR.data(), n_new * 9 * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(solver.d_ew + old_n * 2, ew.data(), n_new * 2 * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(solver.d_eswitch + old_n, es.data(), n_new * sizeof(float), cudaMemcpyHostToDevice));
        }

        // 5. Sliding-window solves (both back-ends).
        int active_lo = std::max(0, t - WINDOW + 1);
        int active_hi = t + 1;

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);

        solver.solve(d_poses_plain, 0, active_lo, active_hi, n_edges, GN_ITERS, PCG_ITERS);
        solver.solve(d_poses_switch, 1, active_lo, active_hi, n_edges, GN_ITERS, PCG_ITERS);

        // 6. Global pass when a loop fires and we haven't run one recently.
        bool do_global = loop_fired && (t - last_global >= LOOP_BURST_FRAMES);
        if (do_global) {
            solver.solve(d_poses_plain, 0, 0, active_hi, n_edges, GLOBAL_GN_ITERS, GLOBAL_PCG_ITERS);
            solver.solve(d_poses_switch, 1, 0, active_hi, n_edges, GLOBAL_GN_ITERS, GLOBAL_PCG_ITERS);
            last_global = t;
            n_global_passes++;
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        if (t >= 5) { total_step_ms += ms; counted++; }

        // 7. Read back poses for visualisation / metrics.
        int rd_lo = do_global ? 0 : active_lo;
        std::vector<float> buf((active_hi - rd_lo) * 12);
        CUDA_CHECK(cudaMemcpy(buf.data(), d_poses_plain + 12 * rd_lo,
                              buf.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int k = rd_lo; k < active_hi; k++) {
            const float* f = buf.data() + (k - rd_lo) * 12;
            plain_h[k].t[0] = f[0]; plain_h[k].t[1] = f[1]; plain_h[k].t[2] = f[2];
            for (int m = 0; m < 9; m++) plain_h[k].R[m] = f[3 + m];
        }
        CUDA_CHECK(cudaMemcpy(buf.data(), d_poses_switch + 12 * rd_lo,
                              buf.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int k = rd_lo; k < active_hi; k++) {
            const float* f = buf.data() + (k - rd_lo) * 12;
            switch_h[k].t[0] = f[0]; switch_h[k].t[1] = f[1]; switch_h[k].t[2] = f[2];
            for (int m = 0; m < 9; m++) switch_h[k].R[m] = f[3 + m];
        }
        // 8. Read back the switch weights for the switch path.
        eswitch_view.resize(n_edges);
        CUDA_CHECK(cudaMemcpy(eswitch_view.data(), solver.d_eswitch,
                              n_edges * sizeof(float), cudaMemcpyDeviceToHost));
        int rejected = 0;
        for (int e = 0; e < n_edges; e++)
            if (edges[e].outlier && eswitch_view[e] < SWITCH_REJECT_THRESH) rejected++;

        float rmse_plain = rmse_translation(plain_h, gt, t);
        float rmse_switch = rmse_translation(switch_h, gt, t);

        // 9. Draw.
        if (t % VIDEO_STRIDE == 0 || t == N_FRAMES - 1) {
            cv::Mat left = draw_panel(plain_h, t, gt, edges, eswitch_view, false,
                                      "plain online (no switches)", rmse_plain, t);
            cv::Mat right = draw_panel(switch_h, t, gt, edges, eswitch_view, true,
                                       "switchable online", rmse_switch, t);
            cv::Mat frame(frame_h, frame_w, CV_8UC3, cv::Scalar(16, 17, 21));
            left.copyTo(frame(cv::Rect(0, TITLE_H, PANEL_W, PANEL_H)));
            right.copyTo(frame(cv::Rect(PANEL_W + 6, TITLE_H, PANEL_W, PANEL_H)));
            cv::putText(frame,
                        "GPU online 3D SLAM: switchable loop constraints (Suenderhauf 2012) in a sliding window",
                        cv::Point(12, 22), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
            int fy = TITLE_H + PANEL_H + 22;
            cv::putText(frame,
                        cv::format("frame %d / %d   window=%d   true loops %d   false loops %d   global passes %d",
                                   t + 1, N_FRAMES, WINDOW, n_true_loops, n_false_loops, n_global_passes),
                        cv::Point(12, fy), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                        cv::Scalar(205, 210, 218), 1, cv::LINE_AA);
            cv::putText(frame,
                        cv::format("RMSE  plain %.3f m  vs  switchable %.3f m      false loops rejected live %d / %d      %.1f ms/step",
                                   rmse_plain, rmse_switch, rejected, n_false_loops, ms),
                        cv::Point(12, fy + 24), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                        cv::Scalar(250, 200, 90), 1, cv::LINE_AA);
            video.write(frame);
        }
        if (t % 40 == 0 || t == N_FRAMES - 1) {
            std::printf("  t=%4d  RMSE plain=%.3f  switch=%.3f  true=%d false=%d rej=%d  step=%.1f ms\n",
                        t, rmse_plain, rmse_switch, n_true_loops, n_false_loops, rejected, ms);
        }
    }

    // Hold the last frame.
    {
        float rmse_plain = rmse_translation(plain_h, gt, N_FRAMES - 1);
        float rmse_switch = rmse_translation(switch_h, gt, N_FRAMES - 1);
        cv::Mat left = draw_panel(plain_h, N_FRAMES - 1, gt, edges, eswitch_view, false,
                                  "plain online (no switches)", rmse_plain, N_FRAMES - 1);
        cv::Mat right = draw_panel(switch_h, N_FRAMES - 1, gt, edges, eswitch_view, true,
                                   "switchable online", rmse_switch, N_FRAMES - 1);
        cv::Mat frame(frame_h, frame_w, CV_8UC3, cv::Scalar(16, 17, 21));
        left.copyTo(frame(cv::Rect(0, TITLE_H, PANEL_W, PANEL_H)));
        right.copyTo(frame(cv::Rect(PANEL_W + 6, TITLE_H, PANEL_W, PANEL_H)));
        cv::putText(frame,
                    "GPU online 3D SLAM: switchable loop constraints (Suenderhauf 2012) in a sliding window",
                    cv::Point(12, 22), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
        for (int i = 0; i < 18; i++) video.write(frame);
    }
    video.release();

    float final_plain = rmse_translation(plain_h, gt, N_FRAMES - 1);
    float final_switch = rmse_translation(switch_h, gt, N_FRAMES - 1);
    int final_rejected = 0;
    float clean_sum = 0.0f, false_sum = 0.0f; int clean_cnt = 0, false_cnt = 0;
    for (int e = 0; e < static_cast<int>(edges.size()); e++) {
        if (!edges[e].loop) continue;
        float s = e < static_cast<int>(eswitch_view.size()) ? eswitch_view[e] : 1.0f;
        if (edges[e].outlier) {
            false_sum += s; false_cnt++;
            if (s < SWITCH_REJECT_THRESH) final_rejected++;
        } else { clean_sum += s; clean_cnt++; }
    }
    std::printf("\n=== GPU online 3D switchable-constraint SLAM ===\n");
    std::printf("frames: %d, window: %d, true loops: %d, false loops: %d, global passes: %d\n",
                N_FRAMES, WINDOW, n_true_loops, n_false_loops, n_global_passes);
    std::printf("final RMSE: plain online = %.4f m,  switchable online = %.4f m\n",
                final_plain, final_switch);
    std::printf("final loop switch: clean avg %.3f, false avg %.3f, rejected false %d/%d\n",
                clean_cnt ? clean_sum / clean_cnt : 1.0f,
                false_cnt ? false_sum / false_cnt : 0.0f, final_rejected, n_false_loops);
    if (counted > 0)
        std::printf("avg step time: %.2f ms (both back-ends, window=%d, GN=%d, PCG=%d)\n",
                    total_step_ms / counted, WINDOW, GN_ITERS, PCG_ITERS);

    avi_to_gif("gif/gpu_online_slam_3d_switchable.avi",
               "gif/gpu_online_slam_3d_switchable.gif", VIDEO_FPS, 900);
    std::printf("Wrote gif/gpu_online_slam_3d_switchable.gif\n");

    cudaFree(d_poses_plain);
    cudaFree(d_poses_switch);
    solver.free_all();
    return 0;
}
