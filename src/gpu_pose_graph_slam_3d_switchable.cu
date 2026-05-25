// gpu_pose_graph_slam_3d_switchable.cu
//
// GPU 3D pose-graph SLAM with dynamic switchable constraints.
//
// The robust 3D backend (gpu_pose_graph_slam_3d_robust) rejects bad loop
// closures with a FROZEN front-end trim: it scores every loop once at the
// initial guess, kills the worst hand-picked fraction, and never revisits
// that decision.  That works, but the rejection set is fixed before any
// optimisation happens and depends on a hand-tuned trim fraction.
//
// This demo replaces the frozen trim with explicit per-loop switch
// variables s_e in [0, 1] that are optimised JOINTLY with the SE(3) poses,
// following Suenderhauf & Protzel, "Switchable Constraints for Robust Pose
// Graph SLAM" (IROS 2012).  The joint objective is
//
//   E(x, s) = sum_odom  0.5 || f_e(x) ||^2_Omega
//           + sum_loop  0.5 Psi(s_e) || f_e(x) ||^2_Omega
//           + sum_loop  0.5 Xi (1 - s_e)^2
//
// with the linear switch function Psi(s) = s.  We minimise it by block
// coordinate descent: each outer iteration linearises the SE(3) residuals,
// closed-form-minimises every switch given the current poses
//
//   s_e* = clamp(1 - chi2_e / (2 Xi), 0, 1),
//
// (damped for stability), then re-solves the pose graph with the updated
// switch weights via damped Jacobi-PCG.  No loop is removed by hand: a true
// loop keeps chi2 ~ dof so s -> 1, a false loop has huge chi2 so s -> 0, and
// every switch is free to change as the poses move.
//
// Residual for edge i->j:  pred = T_i^-1 T_j,  r_t = pred_t - z_t,
//                          r_R = log(z_R^T pred_R).
// State update:            t <- t + dt,  R <- Exp(dw) R.
// SE(3) Jacobians come from central finite differences on the same residual.
//
// Output: gif/gpu_pose_graph_slam_3d_switchable.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
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

constexpr int N_POSES = 384;
constexpr int GN_ITERS = 60;
constexpr int PCG_ITERS = 90;
constexpr int THREADS = 256;
constexpr int MAX_LOOP_EDGES = 192;
constexpr int OUTLIER_LOOP_EDGES = 36;
constexpr int SNAP_STRIDE = 1;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int VIDEO_FPS = 10;
constexpr float PI_F = 3.14159265358979323846f;
constexpr float ODOM_SIGMA_T = 0.022f;
constexpr float ODOM_SIGMA_R = 0.007f;
constexpr float LOOP_SIGMA_T = 0.010f;
constexpr float LOOP_SIGMA_R = 0.0035f;
constexpr float LOOP_DIST = 0.95f;
constexpr int LOOP_MIN_GAP = 90;
constexpr float FD_EPS_T = 1.0e-3f;
constexpr float FD_EPS_R = 2.0e-4f;
constexpr float DAMPING = 4.0e-3f;

// Switchable-constraint hyper-parameters.
// Xi is the switch-prior information: clamp(1 - chi2/(2 Xi)) keeps a loop on
// while its weighted squared error stays below ~2 Xi.  A true loop's residual
// shrinks to ~dof as poses converge so its switch climbs back to ~1; a false
// loop is off by metres (chi2 in the hundreds of thousands) so its switch is
// pinned at 0.  Xi is set well above a true loop's initial misclosure so good
// loops survive their transient error, yet far below a false loop's gross
// chi2 so outliers are rejected outright.
//
// The switch update is asymmetric: a loop is turned OFF fast (a bad loop is
// killed before the graph can bend to accommodate it) but turned back ON
// slowly (a once-rejected loop has to earn trust again over several
// iterations).  This stops the classic failure where a false loop pulls two
// poses together, lowers its own residual, and re-activates itself.
constexpr float SWITCH_PRIOR_XI = 8000.0f;
constexpr float SWITCH_DAMP_OFF = 0.85f;
constexpr float SWITCH_DAMP_ON = 0.18f;
constexpr float SWITCH_REJECT_THRESH = 0.2f;

__host__ __device__ static inline float switch_step(float prev, float target) {
    float damp = (target < prev) ? SWITCH_DAMP_OFF : SWITCH_DAMP_ON;
    return prev + damp * (target - prev);
}

static const char* DEMO_TITLE = "GPU switchable-constraint 3D pose-graph SLAM";
static const char* DEMO_SUBTITLE =
    "per-loop switch variables jointly optimised with SE(3) poses (Suenderhauf 2012)";
static const char* OUTPUT_STEM = "gpu_pose_graph_slam_3d_switchable";

struct Pose {
    float t[3];
    float R[9];
};

struct Edge {
    int i;
    int j;
    float t[3];
    float R[9];
    float wt;
    float wr;
    float switch_weight;
    int loop;
    int outlier;
};

struct EdgeLinearization {
    float r[6];
    float Ji[36];
    float Jj[36];
};

struct Snapshot {
    int iter;
    std::vector<Pose> poses;
    std::vector<float> switch_weight;  // per-edge, for loop colouring
    float trans_rmse;
    float rot_rmse_deg;
    float cost;
    float clean_switch;
    float outlier_switch;
    int rejected;
};

struct BenchResult {
    double gpu_ms = 0.0;
    double cpu_ms = 0.0;
    double speedup = 0.0;
    double plain_gpu_ms = 0.0;
    float init_trans_rmse = 0.0f;
    float final_trans_rmse = 0.0f;
    float init_rot_rmse_deg = 0.0f;
    float final_rot_rmse_deg = 0.0f;
    float final_cost = 0.0f;
    float plain_trans_rmse = 0.0f;
    float plain_rot_rmse_deg = 0.0f;
    float plain_cost = 0.0f;
    float clean_loop_weight = 1.0f;
    float outlier_loop_weight = 1.0f;
    int rejected_outliers = 0;
    int odom_edges = 0;
    int loop_edges = 0;
    int outlier_edges = 0;
};

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
                                                     const Edge& edge,
                                                     float* r) {
    float pred_t[3];
    float pred_R[9];
    pose_relative(pi, pj, pred_t, pred_R);
    r[0] = pred_t[0] - edge.t[0];
    r[1] = pred_t[1] - edge.t[1];
    r[2] = pred_t[2] - edge.t[2];
    float Rerr[9];
    mat3_transpose_mul(edge.R, pred_R, Rerr);
    so3_log(Rerr, r + 3);
}

__host__ __device__ static inline float edge_chi2(float wt,
                                                  float wr,
                                                  const float* r) {
    float trans = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    float rot = r[3] * r[3] + r[4] * r[4] + r[5] * r[5];
    return wt * trans + wr * rot;
}

// Closed-form switch minimiser of  0.5 s chi2 + 0.5 Xi (1 - s)^2  over s.
__host__ __device__ static inline float switch_target(float chi2) {
    return clampf(1.0f - chi2 / (2.0f * SWITCH_PRIOR_XI), 0.0f, 1.0f);
}

// Loop edges use Psi(s) = s as a weight on the information; non-loop edges
// (and the "plain GN" reference) always use 1.
__host__ __device__ static inline float edge_weight_scale(int loop,
                                                          int robust_enabled,
                                                          float switch_weight) {
    if (!robust_enabled || !loop) return 1.0f;
    return clampf(switch_weight, 0.0f, 1.0f);
}

__host__ __device__ static inline float edge_cost_value(int loop,
                                                        int robust_enabled,
                                                        float switch_weight,
                                                        float wt,
                                                        float wr,
                                                        const float* r) {
    float chi2 = edge_chi2(wt, wr, r);
    if (robust_enabled && loop) {
        float s = clampf(switch_weight, 0.0f, 1.0f);
        float prior = 1.0f - s;
        return 0.5f * s * chi2 + 0.5f * SWITCH_PRIOR_XI * prior * prior;
    }
    return 0.5f * chi2;
}

__host__ __device__ static inline void perturb_pose(const Pose& in,
                                                    int axis,
                                                    float eps,
                                                    Pose& out) {
    out = in;
    if (axis < 3) {
        out.t[axis] += eps;
        return;
    }
    float w[3] = {0.0f, 0.0f, 0.0f};
    w[axis - 3] = eps;
    float E[9];
    float Rnew[9];
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

static std::vector<Pose> make_ground_truth() {
    std::vector<Pose> gt(N_POSES);
    for (int i = 0; i < N_POSES; i++) {
        float s = static_cast<float>(i) / N_POSES;
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
        gt[i].t[0] = x;
        gt[i].t[1] = y;
        gt[i].t[2] = z;
        euler_zyx(yaw, pitch, roll, gt[i].R);
    }
    return gt;
}

static void add_noise_to_edge(Edge& e,
                              float sigma_t,
                              float sigma_r,
                              std::mt19937& rng) {
    std::normal_distribution<float> nt(0.0f, sigma_t);
    std::normal_distribution<float> nr(0.0f, sigma_r);
    for (int k = 0; k < 3; k++) e.t[k] += nt(rng);
    float w[3] = {nr(rng), nr(rng), nr(rng)};
    float E[9];
    float Rnew[9];
    so3_exp(w, E);
    mat3_mul(E, e.R, Rnew);
    for (int k = 0; k < 9; k++) e.R[k] = Rnew[k];
}

static Edge make_edge_from_gt(const std::vector<Pose>& gt,
                              int i,
                              int j,
                              float sigma_t,
                              float sigma_r,
                              bool loop,
                              std::mt19937& rng) {
    Edge e{};
    e.i = i;
    e.j = j;
    pose_relative(gt[i], gt[j], e.t, e.R);
    add_noise_to_edge(e, sigma_t, sigma_r, rng);
    e.wt = 1.0f / (sigma_t * sigma_t);
    e.wr = 1.0f / (sigma_r * sigma_r);
    e.switch_weight = 1.0f;
    e.loop = loop ? 1 : 0;
    e.outlier = 0;
    return e;
}

static Edge make_outlier_loop_edge(const std::vector<Pose>& gt,
                                   int i,
                                   int j,
                                   int k,
                                   std::mt19937& rng) {
    Edge e = make_edge_from_gt(gt, i, j, LOOP_SIGMA_T, LOOP_SIGMA_R, true, rng);
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

static std::vector<Edge> make_edges(const std::vector<Pose>& gt,
                                    int& odom_edges,
                                    int& loop_edges,
                                    int& outlier_edges) {
    std::mt19937 rng(25052026);
    std::vector<Edge> edges;
    for (int i = 0; i < N_POSES - 1; i++) {
        edges.push_back(make_edge_from_gt(gt, i, i + 1, ODOM_SIGMA_T, ODOM_SIGMA_R,
                                          false, rng));
    }
    odom_edges = static_cast<int>(edges.size());

    loop_edges = 0;
    outlier_edges = 0;
    int lap_gap = N_POSES / 2;
    for (int i = 0; i < lap_gap && loop_edges < MAX_LOOP_EDGES; i += 1) {
        edges.push_back(make_edge_from_gt(gt, i, i + lap_gap, LOOP_SIGMA_T, LOOP_SIGMA_R,
                                          true, rng));
        loop_edges++;
    }

    std::vector<std::pair<float, std::pair<int, int> > > candidates;
    for (int i = 0; i < N_POSES; i++) {
        for (int j = i + LOOP_MIN_GAP; j < N_POSES; j++) {
            float dx = gt[j].t[0] - gt[i].t[0];
            float dy = gt[j].t[1] - gt[i].t[1];
            float dz = gt[j].t[2] - gt[i].t[2];
            float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < LOOP_DIST * LOOP_DIST) {
                candidates.push_back(std::make_pair(d2, std::make_pair(i, j)));
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const std::pair<float, std::pair<int, int> >& a,
                 const std::pair<float, std::pair<int, int> >& b) {
                  return a.first < b.first;
              });
    std::vector<int> used(N_POSES, 0);
    for (size_t c = 0; c < candidates.size() && loop_edges < MAX_LOOP_EDGES; c++) {
        int i = candidates[c].second.first;
        int j = candidates[c].second.second;
        if (used[i] >= 2 || used[j] >= 2) continue;
        edges.push_back(make_edge_from_gt(gt, i, j, LOOP_SIGMA_T, LOOP_SIGMA_R,
                                          true, rng));
        used[i]++;
        used[j]++;
        loop_edges++;
    }

    for (int k = 0; k < OUTLIER_LOOP_EDGES; k++) {
        int i = (17 * k + 23) % (N_POSES - LOOP_MIN_GAP - 1);
        int gap = LOOP_MIN_GAP + 24 + ((31 * k) % 140);
        int j = (i + gap) % N_POSES;
        if (j == i) j = (j + N_POSES / 3) % N_POSES;
        edges.push_back(make_outlier_loop_edge(gt, i, j, k, rng));
        loop_edges++;
        outlier_edges++;
    }
    return edges;
}

static std::vector<Pose> chain_initial(const std::vector<Pose>& gt,
                                       const std::vector<Edge>& edges,
                                       int odom_edges) {
    std::vector<Pose> poses(N_POSES);
    poses[0] = gt[0];
    for (int e = 0; e < odom_edges; e++) {
        const Edge& edge = edges[e];
        const Pose& pi = poses[edge.i];
        Pose pj{};
        float Rt[3];
        mat3_vec(pi.R, edge.t, Rt);
        pj.t[0] = pi.t[0] + Rt[0];
        pj.t[1] = pi.t[1] + Rt[1];
        pj.t[2] = pi.t[2] + Rt[2];
        mat3_mul(pi.R, edge.R, pj.R);
        poses[edge.j] = pj;
    }
    return poses;
}

static void flatten_poses(const std::vector<Pose>& poses, std::vector<float>& flat) {
    flat.resize(poses.size() * 12);
    for (int i = 0; i < static_cast<int>(poses.size()); i++) {
        flat[12 * i + 0] = poses[i].t[0];
        flat[12 * i + 1] = poses[i].t[1];
        flat[12 * i + 2] = poses[i].t[2];
        for (int k = 0; k < 9; k++) flat[12 * i + 3 + k] = poses[i].R[k];
    }
}

static void flatten_edges(const std::vector<Edge>& edges,
                          std::vector<int>& ei,
                          std::vector<int>& ej,
                          std::vector<int>& eloop,
                          std::vector<float>& et,
                          std::vector<float>& eR,
                          std::vector<float>& eswitch,
                          std::vector<float>& ew) {
    int n = static_cast<int>(edges.size());
    ei.resize(n);
    ej.resize(n);
    eloop.resize(n);
    et.resize(n * 3);
    eR.resize(n * 9);
    eswitch.resize(n);
    ew.resize(n * 2);
    for (int e = 0; e < n; e++) {
        ei[e] = edges[e].i;
        ej[e] = edges[e].j;
        eloop[e] = edges[e].loop;
        for (int k = 0; k < 3; k++) et[3 * e + k] = edges[e].t[k];
        for (int k = 0; k < 9; k++) eR[9 * e + k] = edges[e].R[k];
        eswitch[e] = edges[e].switch_weight;
        ew[2 * e + 0] = edges[e].wt;
        ew[2 * e + 1] = edges[e].wr;
    }
}

static float rmse_translation(const std::vector<Pose>& poses, const std::vector<Pose>& gt) {
    double sum = 0.0;
    for (int i = 0; i < N_POSES; i++) {
        double dx = poses[i].t[0] - gt[i].t[0];
        double dy = poses[i].t[1] - gt[i].t[1];
        double dz = poses[i].t[2] - gt[i].t[2];
        sum += dx * dx + dy * dy + dz * dz;
    }
    return static_cast<float>(std::sqrt(sum / N_POSES));
}

static float rmse_rotation_deg(const std::vector<Pose>& poses, const std::vector<Pose>& gt) {
    double sum = 0.0;
    for (int i = 0; i < N_POSES; i++) {
        float Rerr[9];
        float w[3];
        mat3_transpose_mul(gt[i].R, poses[i].R, Rerr);
        so3_log(Rerr, w);
        double a = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        sum += a * a;
    }
    return static_cast<float>(std::sqrt(sum / N_POSES) * 180.0 / PI_F);
}

static float graph_cost_host(const std::vector<Pose>& poses,
                             const std::vector<Edge>& edges,
                             int robust_enabled) {
    double cost = 0.0;
    for (const Edge& e : edges) {
        float r[6];
        residual_edge(poses[e.i], poses[e.j], e, r);
        cost += edge_cost_value(e.loop, robust_enabled, e.switch_weight, e.wt, e.wr, r);
    }
    return static_cast<float>(cost);
}

static void loop_switch_stats(const std::vector<Edge>& edges,
                              float& clean_avg,
                              float& outlier_avg,
                              int& rejected) {
    double clean_sum = 0.0, outlier_sum = 0.0;
    int clean_count = 0, outlier_count = 0;
    rejected = 0;
    for (const Edge& e : edges) {
        if (!e.loop) continue;
        float w = clampf(e.switch_weight, 0.0f, 1.0f);
        if (e.outlier) {
            outlier_sum += w;
            outlier_count++;
            if (w < SWITCH_REJECT_THRESH) rejected++;
        } else {
            clean_sum += w;
            clean_count++;
        }
    }
    clean_avg = clean_count > 0 ? static_cast<float>(clean_sum / clean_count) : 1.0f;
    outlier_avg = outlier_count > 0 ? static_cast<float>(outlier_sum / outlier_count) : 0.0f;
}

static void apply_update_host(const std::vector<Pose>& in,
                              const std::vector<float>& dx,
                              float step,
                              std::vector<Pose>& out) {
    out = in;
    for (int i = 1; i < N_POSES; i++) {
        out[i].t[0] += step * dx[6 * i + 0];
        out[i].t[1] += step * dx[6 * i + 1];
        out[i].t[2] += step * dx[6 * i + 2];
        float w[3] = {
            step * dx[6 * i + 3],
            step * dx[6 * i + 4],
            step * dx[6 * i + 5],
        };
        float E[9], Rnew[9];
        so3_exp(w, E);
        mat3_mul(E, in[i].R, Rnew);
        for (int k = 0; k < 9; k++) out[i].R[k] = Rnew[k];
    }
}

__device__ static inline void load_pose_device(const float* poses, int idx, Pose& p) {
    p.t[0] = poses[12 * idx + 0];
    p.t[1] = poses[12 * idx + 1];
    p.t[2] = poses[12 * idx + 2];
    for (int k = 0; k < 9; k++) p.R[k] = poses[12 * idx + 3 + k];
}

__device__ static inline void load_edge_device(int e,
                                               const int* ei,
                                               const int* ej,
                                               const float* et,
                                               const float* eR,
                                               const float* ew,
                                               Edge& edge) {
    edge.i = ei[e];
    edge.j = ej[e];
    for (int k = 0; k < 3; k++) edge.t[k] = et[3 * e + k];
    for (int k = 0; k < 9; k++) edge.R[k] = eR[9 * e + k];
    edge.wt = ew[2 * e + 0];
    edge.wr = ew[2 * e + 1];
    edge.loop = 0;
}

__global__ void linearize_fd_kernel(int n_edges,
                                    const int* __restrict__ ei,
                                    const int* __restrict__ ej,
                                    const float* __restrict__ et,
                                    const float* __restrict__ eR,
                                    const float* __restrict__ ew,
                                    const float* __restrict__ poses,
                                    float* __restrict__ residuals,
                                    float* __restrict__ Ji_all,
                                    float* __restrict__ Jj_all) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;

    Edge edge;
    Pose pi, pj;
    load_edge_device(e, ei, ej, et, eR, ew, edge);
    load_pose_device(poses, edge.i, pi);
    load_pose_device(poses, edge.j, pj);

    float base[6];
    residual_edge(pi, pj, edge, base);
    for (int r = 0; r < 6; r++) residuals[6 * e + r] = base[r];

    for (int axis = 0; axis < 6; axis++) {
        float eps = axis < 3 ? FD_EPS_T : FD_EPS_R;
        Pose pp, pm;
        float rp[6], rm[6];
        perturb_pose(pi, axis, eps, pp);
        perturb_pose(pi, axis, -eps, pm);
        residual_edge(pp, pj, edge, rp);
        residual_edge(pm, pj, edge, rm);
        for (int r = 0; r < 6; r++) {
            Ji_all[e * 36 + r * 6 + axis] = (rp[r] - rm[r]) / (2.0f * eps);
        }
        perturb_pose(pj, axis, eps, pp);
        perturb_pose(pj, axis, -eps, pm);
        residual_edge(pi, pp, edge, rp);
        residual_edge(pi, pm, edge, rm);
        for (int r = 0; r < 6; r++) {
            Jj_all[e * 36 + r * 6 + axis] = (rp[r] - rm[r]) / (2.0f * eps);
        }
    }
}

// Closed-form, damped switch update from the freshly-linearised residuals.
// One thread per edge; non-loop edges keep switch = 1.
__global__ void update_switch_kernel(int n_edges,
                                     const int* __restrict__ eloop,
                                     const float* __restrict__ ew,
                                     const float* __restrict__ residuals,
                                     float* __restrict__ eswitch) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    if (!eloop[e]) {
        eswitch[e] = 1.0f;
        return;
    }
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
                                float* __restrict__ b,
                                float* __restrict__ diag) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    int i = ei[e], j = ej[e];
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
        atomicAdd(&b[6 * i + c], bi);
        atomicAdd(&b[6 * j + c], bj);
    }
    for (int a = 0; a < 6; a++) {
        for (int bcol = 0; bcol < 6; bcol++) {
            float vii = 0.0f, vjj = 0.0f;
            for (int rr = 0; rr < 6; rr++) {
                vii += Ji[rr * 6 + a] * wt[rr] * Ji[rr * 6 + bcol];
                vjj += Jj[rr * 6 + a] * wt[rr] * Jj[rr * 6 + bcol];
            }
            atomicAdd(&diag[36 * i + 6 * a + bcol], vii);
            atomicAdd(&diag[36 * j + 6 * a + bcol], vjj);
        }
    }
}

__global__ void matvec_kernel(int n_edges,
                              const int* __restrict__ ei,
                              const int* __restrict__ ej,
                              const int* __restrict__ eloop,
                              const float* __restrict__ eswitch,
                              const float* __restrict__ ew,
                              const float* __restrict__ residuals,
                              const float* __restrict__ Ji_all,
                              const float* __restrict__ Jj_all,
                              int robust_enabled,
                              const float* __restrict__ x,
                              float* __restrict__ y) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    int i = ei[e], j = ej[e];
    float scale = edge_weight_scale(eloop[e], robust_enabled, eswitch[e]);
    float wt[6] = {
        scale * ew[2 * e + 0], scale * ew[2 * e + 0], scale * ew[2 * e + 0],
        scale * ew[2 * e + 1], scale * ew[2 * e + 1], scale * ew[2 * e + 1],
    };
    const float* Ji = Ji_all + 36 * e;
    const float* Jj = Jj_all + 36 * e;
    float u[6];
    for (int r = 0; r < 6; r++) {
        float v = 0.0f;
        for (int c = 0; c < 6; c++) {
            v += Ji[r * 6 + c] * x[6 * i + c];
            v += Jj[r * 6 + c] * x[6 * j + c];
        }
        u[r] = wt[r] * v;
    }
    for (int c = 0; c < 6; c++) {
        float yi = 0.0f, yj = 0.0f;
        for (int r = 0; r < 6; r++) {
            yi += Ji[r * 6 + c] * u[r];
            yj += Jj[r * 6 + c] * u[r];
        }
        atomicAdd(&y[6 * i + c], yi);
        atomicAdd(&y[6 * j + c], yj);
    }
}

__global__ void add_damping_kernel(int n, float damping, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] += damping * x[idx];
}

__global__ void zero_anchor6_kernel(float* x) {
    int k = threadIdx.x;
    if (k < 6) x[k] = 0.0f;
}

__global__ void anchor_diag_kernel(float* b, float* diag) {
    int k = threadIdx.x;
    if (k < 6) b[k] = 0.0f;
    if (k < 36) diag[k] = 0.0f;
    __syncthreads();
    if (k < 6) diag[6 * k + k] = 1.0f;
}

__device__ static bool solve6_spd_device(const float* A_in,
                                         const float* rhs,
                                         float damping,
                                         float* out) {
    float A[36];
    float L[36];
    for (int i = 0; i < 36; i++) {
        A[i] = A_in[i];
        L[i] = 0.0f;
    }
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
                                     const float* __restrict__ diag,
                                     const float* __restrict__ r,
                                     float damping,
                                     float* __restrict__ z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_poses) return;
    float rhs[6];
    float sol[6] = {0, 0, 0, 0, 0, 0};
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

static void compute_cpu_linearization(const std::vector<Pose>& poses,
                                      const std::vector<Edge>& edges,
                                      std::vector<EdgeLinearization>& lin) {
    lin.resize(edges.size());
    for (int e = 0; e < static_cast<int>(edges.size()); e++) {
        const Edge& edge = edges[e];
        residual_edge(poses[edge.i], poses[edge.j], edge, lin[e].r);
        for (int axis = 0; axis < 6; axis++) {
            float eps = axis < 3 ? FD_EPS_T : FD_EPS_R;
            Pose pp, pm;
            float rp[6], rm[6];
            perturb_pose(poses[edge.i], axis, eps, pp);
            perturb_pose(poses[edge.i], axis, -eps, pm);
            residual_edge(pp, poses[edge.j], edge, rp);
            residual_edge(pm, poses[edge.j], edge, rm);
            for (int r = 0; r < 6; r++) lin[e].Ji[r * 6 + axis] = (rp[r] - rm[r]) / (2.0f * eps);
            perturb_pose(poses[edge.j], axis, eps, pp);
            perturb_pose(poses[edge.j], axis, -eps, pm);
            residual_edge(poses[edge.i], pp, edge, rp);
            residual_edge(poses[edge.i], pm, edge, rm);
            for (int r = 0; r < 6; r++) lin[e].Jj[r * 6 + axis] = (rp[r] - rm[r]) / (2.0f * eps);
        }
    }
}

static void update_switches_cpu(const std::vector<EdgeLinearization>& lin,
                                std::vector<Edge>& edges) {
    for (int e = 0; e < static_cast<int>(edges.size()); e++) {
        if (!edges[e].loop) continue;
        float chi2 = edge_chi2(edges[e].wt, edges[e].wr, lin[e].r);
        float target = switch_target(chi2);
        edges[e].switch_weight = switch_step(edges[e].switch_weight, target);
    }
}

static void cpu_matvec(const std::vector<Edge>& edges,
                       const std::vector<EdgeLinearization>& lin,
                       const std::vector<float>& x,
                       int robust_enabled,
                       std::vector<float>& y) {
    std::fill(y.begin(), y.end(), 0.0f);
    for (int e = 0; e < static_cast<int>(edges.size()); e++) {
        const Edge& edge = edges[e];
        float scale = edge_weight_scale(edge.loop, robust_enabled, edge.switch_weight);
        float wt[6] = {
            scale * edge.wt, scale * edge.wt, scale * edge.wt,
            scale * edge.wr, scale * edge.wr, scale * edge.wr
        };
        float u[6];
        for (int r = 0; r < 6; r++) {
            float v = 0.0f;
            for (int c = 0; c < 6; c++) {
                v += lin[e].Ji[r * 6 + c] * x[6 * edge.i + c];
                v += lin[e].Jj[r * 6 + c] * x[6 * edge.j + c];
            }
            u[r] = wt[r] * v;
        }
        for (int c = 0; c < 6; c++) {
            float yi = 0.0f, yj = 0.0f;
            for (int r = 0; r < 6; r++) {
                yi += lin[e].Ji[r * 6 + c] * u[r];
                yj += lin[e].Jj[r * 6 + c] * u[r];
            }
            y[6 * edge.i + c] += yi;
            y[6 * edge.j + c] += yj;
        }
    }
    for (int i = 0; i < static_cast<int>(x.size()); i++) y[i] += DAMPING * x[i];
    for (int k = 0; k < 6; k++) y[k] = 0.0f;
}

static bool solve6_spd_host(const float* A_in, const float* rhs, float damping, float* out) {
    float A[36], L[36] = {0};
    for (int i = 0; i < 36; i++) A[i] = A_in[i];
    for (int i = 0; i < 6; i++) A[6 * i + i] += damping;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j <= i; j++) {
            float s = A[6 * i + j];
            for (int k = 0; k < j; k++) s -= L[6 * i + k] * L[6 * j + k];
            if (i == j) {
                if (s <= 1.0e-12f) return false;
                L[6 * i + j] = std::sqrt(s);
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

static void assemble_cpu(const std::vector<Edge>& edges,
                         const std::vector<EdgeLinearization>& lin,
                         int robust_enabled,
                         std::vector<float>& b,
                         std::vector<float>& diag) {
    std::fill(b.begin(), b.end(), 0.0f);
    std::fill(diag.begin(), diag.end(), 0.0f);
    for (int e = 0; e < static_cast<int>(edges.size()); e++) {
        const Edge& edge = edges[e];
        float scale = edge_weight_scale(edge.loop, robust_enabled, edge.switch_weight);
        float wt[6] = {
            scale * edge.wt, scale * edge.wt, scale * edge.wt,
            scale * edge.wr, scale * edge.wr, scale * edge.wr
        };
        for (int c = 0; c < 6; c++) {
            float bi = 0.0f, bj = 0.0f;
            for (int r = 0; r < 6; r++) {
                float wr = wt[r] * lin[e].r[r];
                bi += lin[e].Ji[r * 6 + c] * wr;
                bj += lin[e].Jj[r * 6 + c] * wr;
            }
            b[6 * edge.i + c] += bi;
            b[6 * edge.j + c] += bj;
        }
        for (int a = 0; a < 6; a++) {
            for (int c = 0; c < 6; c++) {
                float vii = 0.0f, vjj = 0.0f;
                for (int r = 0; r < 6; r++) {
                    vii += lin[e].Ji[r * 6 + a] * wt[r] * lin[e].Ji[r * 6 + c];
                    vjj += lin[e].Jj[r * 6 + a] * wt[r] * lin[e].Jj[r * 6 + c];
                }
                diag[36 * edge.i + 6 * a + c] += vii;
                diag[36 * edge.j + 6 * a + c] += vjj;
            }
        }
    }
    for (int k = 0; k < 6; k++) b[k] = 0.0f;
    for (int k = 0; k < 36; k++) diag[k] = 0.0f;
    for (int k = 0; k < 6; k++) diag[6 * k + k] = 1.0f;
}

static void apply_cpu_precond(const std::vector<float>& diag,
                              const std::vector<float>& r,
                              std::vector<float>& z) {
    for (int i = 0; i < N_POSES; i++) {
        float rhs[6], sol[6] = {0, 0, 0, 0, 0, 0};
        for (int k = 0; k < 6; k++) rhs[k] = r[6 * i + k];
        bool ok = solve6_spd_host(diag.data() + 36 * i, rhs, DAMPING, sol);
        if (!ok) {
            for (int k = 0; k < 6; k++) {
                float d = diag[36 * i + 6 * k + k] + DAMPING;
                sol[k] = rhs[k] / std::max(d, 1.0e-6f);
            }
        }
        for (int k = 0; k < 6; k++) z[6 * i + k] = sol[k];
    }
}

static void cpu_pcg_solve(const std::vector<Edge>& edges,
                          const std::vector<EdgeLinearization>& lin,
                          const std::vector<float>& b,
                          const std::vector<float>& diag,
                          int robust_enabled,
                          std::vector<float>& dx) {
    const int n = N_POSES * 6;
    dx.assign(n, 0.0f);
    std::vector<float> r(n), z(n), p(n), Ap(n);
    for (int i = 0; i < n; i++) r[i] = -b[i];
    for (int k = 0; k < 6; k++) r[k] = 0.0f;
    apply_cpu_precond(diag, r, z);
    p = z;
    auto dot = [](const std::vector<float>& a, const std::vector<float>& bvec) {
        double s = 0.0;
        for (int i = 0; i < static_cast<int>(a.size()); i++) s += a[i] * bvec[i];
        return static_cast<float>(s);
    };
    float rz_old = dot(r, z);
    float rr0 = std::max(1.0e-12f, dot(r, r));
    for (int it = 0; it < PCG_ITERS; it++) {
        cpu_matvec(edges, lin, p, robust_enabled, Ap);
        float pAp = dot(p, Ap);
        if (pAp <= 1.0e-20f) break;
        float alpha = rz_old / pAp;
        for (int i = 0; i < n; i++) {
            dx[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        for (int k = 0; k < 6; k++) r[k] = 0.0f;
        float rr = dot(r, r);
        if (rr < rr0 * 1.0e-7f) break;
        apply_cpu_precond(diag, r, z);
        float rz_new = dot(r, z);
        float beta = rz_new / std::max(1.0e-20f, rz_old);
        for (int i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
        rz_old = rz_new;
    }
    for (int k = 0; k < 6; k++) dx[k] = 0.0f;
}

static double run_cpu_reference(const std::vector<Pose>& initial,
                                const std::vector<Pose>& gt,
                                const std::vector<Edge>& edges_in,
                                int robust_enabled,
                                std::vector<Pose>& out) {
    std::vector<Pose> poses = initial;
    std::vector<Edge> edges = edges_in;
    std::vector<EdgeLinearization> lin;
    std::vector<float> b(N_POSES * 6), diag(N_POSES * 36), dx;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < GN_ITERS; iter++) {
        compute_cpu_linearization(poses, edges, lin);
        if (robust_enabled) update_switches_cpu(lin, edges);
        assemble_cpu(edges, lin, robust_enabled, b, diag);
        cpu_pcg_solve(edges, lin, b, diag, robust_enabled, dx);
        float current_cost = graph_cost_host(poses, edges, robust_enabled);
        std::vector<Pose> trial;
        bool accepted = false;
        for (float step : {1.0f, 0.5f, 0.25f, 0.125f, 0.0625f}) {
            apply_update_host(poses, dx, step, trial);
            float trial_cost = graph_cost_host(trial, edges, robust_enabled);
            if (trial_cost < current_cost) {
                poses.swap(trial);
                accepted = true;
                break;
            }
        }
        if (!accepted) break;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    out = poses;
    (void)gt;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double run_gpu_solver(const std::vector<Pose>& initial,
                             const std::vector<Pose>& gt,
                             const std::vector<Edge>& edges_in,
                             int robust_enabled,
                             const char* run_label,
                             std::vector<Pose>& out,
                             std::vector<Snapshot>& snapshots) {
    std::vector<Edge> edges = edges_in;  // local copy; switch_weight evolves
    const int n_edges = static_cast<int>(edges.size());
    const int n_state = N_POSES * 6;
    const int n_pose_floats = N_POSES * 12;
    std::vector<int> ei, ej, eloop;
    std::vector<float> et, eR, eswitch, ew;
    flatten_edges(edges, ei, ej, eloop, et, eR, eswitch, ew);
    std::vector<float> poses_flat;
    flatten_poses(initial, poses_flat);
    std::vector<Pose> poses_host = initial;
    std::vector<float> dx_host(n_state);
    std::vector<float> eswitch_host = eswitch;

    int *d_ei = nullptr, *d_ej = nullptr, *d_eloop = nullptr;
    float *d_et = nullptr, *d_eR = nullptr, *d_eswitch = nullptr, *d_ew = nullptr;
    float *d_poses = nullptr, *d_residuals = nullptr, *d_Ji = nullptr, *d_Jj = nullptr;
    float *d_b = nullptr, *d_diag = nullptr, *d_dx = nullptr, *d_r = nullptr;
    float *d_z = nullptr, *d_p = nullptr, *d_Ap = nullptr, *d_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ei, n_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ej, n_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_eloop, n_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_et, n_edges * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_eR, n_edges * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_eswitch, n_edges * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ew, n_edges * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_poses, n_pose_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residuals, n_edges * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ji, n_edges * 36 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Jj, n_edges * 36 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n_state * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_diag, N_POSES * 36 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx, n_state * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r, n_state * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, n_state * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p, n_state * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n_state * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_ei, ei.data(), n_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ej, ej.data(), n_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eloop, eloop.data(), n_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_et, et.data(), n_edges * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eR, eR.data(), n_edges * 9 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eswitch, eswitch.data(), n_edges * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ew, ew.data(), n_edges * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_poses, poses_flat.data(), n_pose_floats * sizeof(float), cudaMemcpyHostToDevice));

    auto push_snapshot = [&](int iter, float cost) {
        Snapshot s;
        s.iter = iter;
        s.poses = poses_host;
        s.switch_weight.resize(n_edges);
        for (int e = 0; e < n_edges; e++) s.switch_weight[e] = edges[e].switch_weight;
        s.trans_rmse = rmse_translation(poses_host, gt);
        s.rot_rmse_deg = rmse_rotation_deg(poses_host, gt);
        s.cost = cost;
        loop_switch_stats(edges, s.clean_switch, s.outlier_switch, s.rejected);
        snapshots.push_back(std::move(s));
    };

    snapshots.clear();
    push_snapshot(0, graph_cost_host(poses_host, edges, robust_enabled));

    int blocks_e = (n_edges + THREADS - 1) / THREADS;
    int blocks_state = (n_state + THREADS - 1) / THREADS;
    int blocks_diag = (N_POSES * 36 + THREADS - 1) / THREADS;
    int blocks_pose = (N_POSES + THREADS - 1) / THREADS;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    for (int iter = 0; iter < GN_ITERS; iter++) {
        linearize_fd_kernel<<<blocks_e, THREADS>>>(n_edges, d_ei, d_ej, d_et, d_eR,
                                                   d_ew, d_poses, d_residuals,
                                                   d_Ji, d_Jj);
        if (robust_enabled) {
            // Coordinate step on the switches: closed-form minimiser given poses.
            update_switch_kernel<<<blocks_e, THREADS>>>(n_edges, d_eloop, d_ew,
                                                        d_residuals, d_eswitch);
            CUDA_CHECK(cudaMemcpy(eswitch_host.data(), d_eswitch,
                                  n_edges * sizeof(float), cudaMemcpyDeviceToHost));
            for (int e = 0; e < n_edges; e++) edges[e].switch_weight = eswitch_host[e];
        }
        zero_kernel<<<blocks_state, THREADS>>>(n_state, d_b);
        zero_kernel<<<blocks_diag, THREADS>>>(N_POSES * 36, d_diag);
        assemble_kernel<<<blocks_e, THREADS>>>(n_edges, d_ei, d_ej, d_eloop,
                                               d_eswitch, d_ew,
                                               d_residuals, d_Ji, d_Jj, robust_enabled,
                                               d_b, d_diag);
        anchor_diag_kernel<<<1, 36>>>(d_b, d_diag);

        zero_kernel<<<blocks_state, THREADS>>>(n_state, d_dx);
        zero_kernel<<<blocks_state, THREADS>>>(n_state, d_r);
        axpy_kernel<<<blocks_state, THREADS>>>(n_state, -1.0f, d_b, d_r);
        zero_anchor6_kernel<<<1, 6>>>(d_r);
        apply_precond_kernel<<<blocks_pose, THREADS>>>(N_POSES, d_diag, d_r, DAMPING, d_z);
        zero_anchor6_kernel<<<1, 6>>>(d_z);
        copy_kernel<<<blocks_state, THREADS>>>(n_state, d_z, d_p);

        float rz_old = 0.0f;
        float rr0 = 0.0f;
        CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
        dot_kernel<<<32, 256>>>(n_state, d_r, d_z, d_scratch);
        CUDA_CHECK(cudaMemcpy(&rz_old, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
        dot_kernel<<<32, 256>>>(n_state, d_r, d_r, d_scratch);
        CUDA_CHECK(cudaMemcpy(&rr0, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
        rr0 = fmaxf(rr0, 1.0e-12f);

        for (int pcg = 0; pcg < PCG_ITERS; pcg++) {
            zero_kernel<<<blocks_state, THREADS>>>(n_state, d_Ap);
            matvec_kernel<<<blocks_e, THREADS>>>(n_edges, d_ei, d_ej, d_eloop,
                                                 d_eswitch, d_ew,
                                                 d_residuals, d_Ji, d_Jj, robust_enabled,
                                                 d_p, d_Ap);
            add_damping_kernel<<<blocks_state, THREADS>>>(n_state, DAMPING, d_p, d_Ap);
            zero_anchor6_kernel<<<1, 6>>>(d_Ap);

            float pAp = 0.0f;
            CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
            dot_kernel<<<32, 256>>>(n_state, d_p, d_Ap, d_scratch);
            CUDA_CHECK(cudaMemcpy(&pAp, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
            if (pAp <= 1.0e-20f) break;
            float alpha = rz_old / pAp;
            axpy_kernel<<<blocks_state, THREADS>>>(n_state, alpha, d_p, d_dx);
            axpy_kernel<<<blocks_state, THREADS>>>(n_state, -alpha, d_Ap, d_r);
            zero_anchor6_kernel<<<1, 6>>>(d_r);

            float rr = 0.0f;
            CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
            dot_kernel<<<32, 256>>>(n_state, d_r, d_r, d_scratch);
            CUDA_CHECK(cudaMemcpy(&rr, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
            if (rr < rr0 * 1.0e-7f) break;
            apply_precond_kernel<<<blocks_pose, THREADS>>>(N_POSES, d_diag, d_r, DAMPING, d_z);
            zero_anchor6_kernel<<<1, 6>>>(d_z);
            float rz_new = 0.0f;
            CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
            dot_kernel<<<32, 256>>>(n_state, d_r, d_z, d_scratch);
            CUDA_CHECK(cudaMemcpy(&rz_new, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
            float beta = rz_new / fmaxf(1.0e-20f, rz_old);
            xpay_kernel<<<blocks_state, THREADS>>>(n_state, beta, d_z, d_p);
            zero_anchor6_kernel<<<1, 6>>>(d_p);
            rz_old = rz_new;
        }

        CUDA_CHECK(cudaMemcpy(dx_host.data(), d_dx, n_state * sizeof(float), cudaMemcpyDeviceToHost));
        float current_cost = graph_cost_host(poses_host, edges, robust_enabled);
        std::vector<Pose> trial;
        bool accepted = false;
        float accepted_cost = current_cost;
        for (float step : {1.0f, 0.5f, 0.25f, 0.125f, 0.0625f}) {
            apply_update_host(poses_host, dx_host, step, trial);
            float trial_cost = graph_cost_host(trial, edges, robust_enabled);
            if (trial_cost < current_cost) {
                poses_host.swap(trial);
                accepted_cost = trial_cost;
                accepted = true;
                break;
            }
        }
        if (!accepted) {
            std::printf("  %s iter %02d: no pose decrease (cost %.2f), switches still active\n",
                        run_label, iter + 1, current_cost);
            // Keep iterating: the switch coordinate step can still reduce E.
        } else {
            flatten_poses(poses_host, poses_flat);
            CUDA_CHECK(cudaMemcpy(d_poses, poses_flat.data(), n_pose_floats * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        if (((iter + 1) % SNAP_STRIDE == 0) || iter + 1 == GN_ITERS) {
            push_snapshot(iter + 1, accepted_cost);
        }
        if (iter < 5 || iter % 5 == 4) {
            std::printf("  %s iter %02d  cost %.2f  trans RMSE %.3f m  rot %.3f deg"
                        "  clean s %.2f  false s %.2f\n",
                        run_label, iter + 1, accepted_cost,
                        rmse_translation(poses_host, gt),
                        rmse_rotation_deg(poses_host, gt),
                        snapshots.back().clean_switch, snapshots.back().outlier_switch);
        }
    }

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t0, t1));
    out = poses_host;

    CUDA_CHECK(cudaFree(d_ei));
    CUDA_CHECK(cudaFree(d_ej));
    CUDA_CHECK(cudaFree(d_eloop));
    CUDA_CHECK(cudaFree(d_et));
    CUDA_CHECK(cudaFree(d_eR));
    CUDA_CHECK(cudaFree(d_eswitch));
    CUDA_CHECK(cudaFree(d_ew));
    CUDA_CHECK(cudaFree(d_poses));
    CUDA_CHECK(cudaFree(d_residuals));
    CUDA_CHECK(cudaFree(d_Ji));
    CUDA_CHECK(cudaFree(d_Jj));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_diag));
    CUDA_CHECK(cudaFree(d_dx));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_Ap));
    CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return elapsed_ms;
}

static cv::Point2i project(float x, float y, float z, float yaw, float pitch) {
    float cy = std::cos(yaw), sy = std::sin(yaw);
    float cp = std::cos(pitch), sp = std::sin(pitch);
    float x1 = cy * x + sy * y;
    float y1 = -sy * x + cy * y;
    float z1 = z - 1.7f;
    float y2 = cp * y1 - sp * z1;
    float z2 = sp * y1 + cp * z1;
    float scale = 42.0f;
    return cv::Point2i(PANEL_W / 2 - 110 + static_cast<int>(scale * x1),
                       350 - static_cast<int>(scale * z2 + 0.18f * scale * y2));
}

static cv::Scalar interp_color(float t) {
    t = clampf(t, 0.0f, 1.0f);
    return cv::Scalar(80 + 120 * t, 190 - 40 * t, 255 - 160 * t);
}

// Switch weight -> colour: s=1 trusted (green), s=0 rejected (red).
static cv::Scalar switch_color(float s) {
    s = clampf(s, 0.0f, 1.0f);
    return cv::Scalar(70.0f, 90.0f + 130.0f * s, 70.0f + 175.0f * (1.0f - s));
}

static void draw_traj(cv::Mat& img,
                      const std::vector<Pose>& poses,
                      cv::Scalar color,
                      int thickness,
                      float yaw,
                      float pitch) {
    for (int i = 1; i < N_POSES; i++) {
        cv::line(img,
                 project(poses[i - 1].t[0], poses[i - 1].t[1], poses[i - 1].t[2], yaw, pitch),
                 project(poses[i].t[0], poses[i].t[1], poses[i].t[2], yaw, pitch),
                 color, thickness, cv::LINE_AA);
    }
}

static cv::Mat draw_frame(const std::vector<Snapshot>& snaps,
                          int frame_idx,
                          const std::vector<Pose>& gt,
                          const std::vector<Pose>& initial,
                          const std::vector<Edge>& edges,
                          const BenchResult& bench) {
    const Snapshot& snap = snaps[frame_idx];
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 19, 23));
    float yaw = -0.72f + 0.018f * snap.iter;
    float pitch = 0.58f;
    cv::putText(img,
                cv::format("%s  iter %02d / %d", DEMO_TITLE, snap.iter, GN_ITERS),
                cv::Point(28, 34), cv::FONT_HERSHEY_SIMPLEX, 0.62,
                cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, DEMO_SUBTITLE,
                cv::Point(31, 58), cv::FONT_HERSHEY_SIMPLEX, 0.40,
                cv::Scalar(165, 170, 180), 1, cv::LINE_AA);

    cv::Rect scene(24, 76, 560, 500);
    cv::rectangle(img, scene, cv::Scalar(27, 29, 34), -1);
    cv::rectangle(img, scene, cv::Scalar(78, 82, 90), 1);
    for (int g = -8; g <= 8; g += 2) {
        cv::line(img, project(-9, g, 0, yaw, pitch), project(9, g, 0, yaw, pitch),
                 cv::Scalar(46, 49, 56), 1, cv::LINE_AA);
        cv::line(img, project(g, -7, 0, yaw, pitch), project(g, 7, 0, yaw, pitch),
                 cv::Scalar(46, 49, 56), 1, cv::LINE_AA);
    }
    draw_traj(img, gt, cv::Scalar(210, 210, 210), 1, yaw, pitch);
    draw_traj(img, initial, cv::Scalar(80, 90, 110), 1, yaw, pitch);
    // Loop closures coloured by their current switch weight.
    for (int e = 0; e < static_cast<int>(edges.size()); e++) {
        if (!edges[e].loop) continue;
        float s = e < static_cast<int>(snap.switch_weight.size()) ? snap.switch_weight[e] : 1.0f;
        cv::Scalar color = switch_color(s);
        int thickness = (s > 0.5f) ? 1 : 2;
        cv::line(img,
                 project(snap.poses[edges[e].i].t[0], snap.poses[edges[e].i].t[1],
                         snap.poses[edges[e].i].t[2], yaw, pitch),
                 project(snap.poses[edges[e].j].t[0], snap.poses[edges[e].j].t[1],
                         snap.poses[edges[e].j].t[2], yaw, pitch),
                 color, thickness, cv::LINE_AA);
    }
    draw_traj(img, snap.poses, cv::Scalar(70, 210, 255), 2, yaw, pitch);
    for (int i = 0; i < N_POSES; i += 12) {
        cv::circle(img, project(snap.poses[i].t[0], snap.poses[i].t[1], snap.poses[i].t[2], yaw, pitch),
                   2, interp_color(static_cast<float>(i) / N_POSES), -1, cv::LINE_AA);
    }
    cv::putText(img, "gray=GT  dim=odometry  cyan=optimized",
                cv::Point(scene.x + 12, scene.y + scene.height - 34),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(225, 225, 225), 1, cv::LINE_AA);
    cv::putText(img, "loop colour = switch weight (green on / red rejected)",
                cv::Point(scene.x + 12, scene.y + scene.height - 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(225, 225, 225), 1, cv::LINE_AA);

    cv::Rect info(600, 76, 336, 500);
    cv::rectangle(img, info, cv::Scalar(29, 31, 36), -1);
    cv::rectangle(img, info, cv::Scalar(76, 80, 88), 1);
    int tx = info.x + 14;
    cv::putText(img, "metrics", cv::Point(tx, info.y + 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, cv::format("%d poses, %d odom, %d loops", N_POSES,
                                bench.odom_edges, bench.loop_edges),
                cv::Point(tx, info.y + 56), cv::FONT_HERSHEY_SIMPLEX,
                0.40, cv::Scalar(205, 210, 218), 1, cv::LINE_AA);
    cv::putText(img, cv::format("%d false loop closures injected", bench.outlier_edges),
                cv::Point(tx, info.y + 78), cv::FONT_HERSHEY_SIMPLEX,
                0.40, cv::Scalar(205, 210, 218), 1, cv::LINE_AA);
    cv::putText(img, cv::format("trans %.3f -> %.3f m",
                                bench.init_trans_rmse, snap.trans_rmse),
                cv::Point(tx, info.y + 104), cv::FONT_HERSHEY_SIMPLEX,
                0.46, cv::Scalar(90, 225, 135), 1, cv::LINE_AA);
    cv::putText(img, cv::format("rot %.3f -> %.3f deg",
                                bench.init_rot_rmse_deg, snap.rot_rmse_deg),
                cv::Point(tx, info.y + 128), cv::FONT_HERSHEY_SIMPLEX,
                0.46, cv::Scalar(90, 225, 135), 1, cv::LINE_AA);
    cv::putText(img, cv::format("plain GN %.2f m / %.1f deg",
                                bench.plain_trans_rmse, bench.plain_rot_rmse_deg),
                cv::Point(tx, info.y + 152), cv::FONT_HERSHEY_SIMPLEX,
                0.43, cv::Scalar(105, 125, 235), 1, cv::LINE_AA);
    cv::putText(img, cv::format("GPU %.2f ms  CPU %.2f ms", bench.gpu_ms, bench.cpu_ms),
                cv::Point(tx, info.y + 178), cv::FONT_HERSHEY_SIMPLEX,
                0.46, cv::Scalar(165, 175, 190), 1, cv::LINE_AA);
    cv::putText(img, cv::format("speedup %.1fx", bench.speedup),
                cv::Point(tx, info.y + 202), cv::FONT_HERSHEY_SIMPLEX,
                0.50, cv::Scalar(250, 190, 70), 1, cv::LINE_AA);
    cv::putText(img, cv::format("switch: clean %.2f  false %.2f",
                                snap.clean_switch, snap.outlier_switch),
                cv::Point(tx, info.y + 228), cv::FONT_HERSHEY_SIMPLEX,
                0.43, cv::Scalar(220, 225, 232), 1, cv::LINE_AA);
    cv::putText(img, cv::format("rejected false %d / %d", snap.rejected, bench.outlier_edges),
                cv::Point(tx, info.y + 250), cv::FONT_HERSHEY_SIMPLEX,
                0.43, cv::Scalar(220, 225, 232), 1, cv::LINE_AA);

    // Switch-convergence plot: clean (green) and false (red) avg switch vs iter.
    cv::Rect plot(info.x + 14, info.y + 274, info.width - 28, 198);
    cv::rectangle(img, plot, cv::Scalar(22, 24, 28), -1);
    cv::rectangle(img, plot, cv::Scalar(70, 74, 82), 1);
    cv::putText(img, "switch weight vs iteration",
                cv::Point(plot.x + 6, plot.y + 16), cv::FONT_HERSHEY_SIMPLEX,
                0.36, cv::Scalar(180, 185, 195), 1, cv::LINE_AA);
    int p_top = plot.y + 24, p_bot = plot.y + plot.height - 18;
    int p_left = plot.x + 10, p_right = plot.x + plot.width - 10;
    // gridlines at s=0,0.5,1
    for (float gv : {0.0f, 0.5f, 1.0f}) {
        int yy = p_bot - static_cast<int>((p_bot - p_top) * gv);
        cv::line(img, cv::Point(p_left, yy), cv::Point(p_right, yy),
                 cv::Scalar(44, 47, 54), 1, cv::LINE_AA);
        cv::putText(img, cv::format("%.1f", gv), cv::Point(p_right + 2, yy + 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.30, cv::Scalar(150, 155, 165), 1, cv::LINE_AA);
    }
    int last_iter = std::max(1, snaps.back().iter);
    auto plot_xy = [&](int it, float val) {
        float fx = static_cast<float>(it) / last_iter;
        int xx = p_left + static_cast<int>((p_right - p_left) * fx);
        int yy = p_bot - static_cast<int>((p_bot - p_top) * clampf(val, 0.0f, 1.0f));
        return cv::Point(xx, yy);
    };
    for (int k = 1; k <= frame_idx; k++) {
        cv::line(img, plot_xy(snaps[k - 1].iter, snaps[k - 1].clean_switch),
                 plot_xy(snaps[k].iter, snaps[k].clean_switch),
                 cv::Scalar(90, 225, 135), 2, cv::LINE_AA);
        cv::line(img, plot_xy(snaps[k - 1].iter, snaps[k - 1].outlier_switch),
                 plot_xy(snaps[k].iter, snaps[k].outlier_switch),
                 cv::Scalar(95, 110, 240), 2, cv::LINE_AA);
    }
    cv::putText(img, "green=true loops  red=false loops",
                cv::Point(plot.x + 6, plot.y + plot.height - 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.32, cv::Scalar(180, 185, 195), 1, cv::LINE_AA);
    return img;
}

static void write_video(const std::vector<Snapshot>& snapshots,
                        const std::vector<Pose>& gt,
                        const std::vector<Pose>& initial,
                        const std::vector<Edge>& edges,
                        const BenchResult& bench) {
    int mkdir_rc = std::system("mkdir -p gif");
    if (mkdir_rc != 0) {
        std::fprintf(stderr, "Failed to create gif directory\n");
        std::exit(1);
    }
    const std::string avi_path = std::string("gif/") + OUTPUT_STEM + ".avi";
    const std::string gif_path = std::string("gif/") + OUTPUT_STEM + ".gif";
    cv::VideoWriter writer(
        avi_path,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        VIDEO_FPS,
        cv::Size(PANEL_W, PANEL_H));
    if (!writer.isOpened()) {
        std::fprintf(stderr, "Failed to open %s\n", avi_path.c_str());
        std::exit(1);
    }
    for (int k = 0; k < static_cast<int>(snapshots.size()); k++) {
        writer.write(draw_frame(snapshots, k, gt, initial, edges, bench));
    }
    for (int i = 0; i < 10; i++) {
        writer.write(draw_frame(snapshots, static_cast<int>(snapshots.size()) - 1,
                                gt, initial, edges, bench));
    }
    writer.release();
    avi_to_gif(avi_path, gif_path, VIDEO_FPS, 620);
}

}  // namespace cudabot

int main() {
    using namespace cudabot;

    std::vector<Pose> gt = make_ground_truth();
    int odom_edges = 0, loop_edges = 0, outlier_edges = 0;
    std::vector<Edge> edges = make_edges(gt, odom_edges, loop_edges, outlier_edges);
    std::vector<Pose> initial = chain_initial(gt, edges, odom_edges);

    BenchResult bench;
    bench.odom_edges = odom_edges;
    bench.loop_edges = loop_edges;
    bench.outlier_edges = outlier_edges;
    bench.init_trans_rmse = rmse_translation(initial, gt);
    bench.init_rot_rmse_deg = rmse_rotation_deg(initial, gt);

    std::printf("%s\n", DEMO_TITLE);
    std::printf("poses: %d, odom edges: %d, loop edges: %d, false loops: %d, total edges: %zu\n",
                N_POSES, odom_edges, loop_edges, outlier_edges, edges.size());
    std::printf("switch prior Xi: %.0f, damp off %.2f / on %.2f\n",
                SWITCH_PRIOR_XI, SWITCH_DAMP_OFF, SWITCH_DAMP_ON);
    std::printf("initial RMSE: %.4f m, %.4f deg, plain cost %.2f\n",
                bench.init_trans_rmse, bench.init_rot_rmse_deg,
                graph_cost_host(initial, edges, 0));

    // Plain GN on the corrupted graph (all loop weights = 1).
    std::vector<Pose> plain_gpu_out;
    std::vector<Snapshot> plain_snapshots;
    bench.plain_gpu_ms =
        run_gpu_solver(initial, gt, edges, 0, "plain", plain_gpu_out, plain_snapshots);
    bench.plain_trans_rmse = rmse_translation(plain_gpu_out, gt);
    bench.plain_rot_rmse_deg = rmse_rotation_deg(plain_gpu_out, gt);
    bench.plain_cost = graph_cost_host(plain_gpu_out, edges, 0);
    std::printf("plain GPU final RMSE: %.4f m, %.4f deg, cost %.2f\n",
                bench.plain_trans_rmse, bench.plain_rot_rmse_deg, bench.plain_cost);

    // Switchable constraints: switches optimised jointly with poses.
    std::vector<Pose> gpu_out;
    std::vector<Snapshot> snapshots;
    bench.gpu_ms = run_gpu_solver(initial, gt, edges, 1, "switch", gpu_out, snapshots);

    std::vector<Pose> cpu_out;
    bench.cpu_ms = run_cpu_reference(initial, gt, edges, 1, cpu_out);
    bench.speedup = bench.cpu_ms / std::max(1.0e-9, bench.gpu_ms);
    bench.final_trans_rmse = rmse_translation(gpu_out, gt);
    bench.final_rot_rmse_deg = rmse_rotation_deg(gpu_out, gt);

    // Recover the final switch weights from the last snapshot for stats.
    std::vector<Edge> final_edges = edges;
    if (!snapshots.empty()) {
        for (int e = 0; e < static_cast<int>(final_edges.size()) &&
                        e < static_cast<int>(snapshots.back().switch_weight.size()); e++) {
            final_edges[e].switch_weight = snapshots.back().switch_weight[e];
        }
    }
    bench.final_cost = graph_cost_host(gpu_out, final_edges, 1);
    loop_switch_stats(final_edges, bench.clean_loop_weight, bench.outlier_loop_weight,
                      bench.rejected_outliers);
    if (!snapshots.empty()) {
        snapshots.back().trans_rmse = bench.final_trans_rmse;
        snapshots.back().rot_rmse_deg = bench.final_rot_rmse_deg;
        snapshots.back().cost = bench.final_cost;
    }

    write_video(snapshots, gt, initial, final_edges, bench);

    float cpu_trans = rmse_translation(cpu_out, gt);
    float cpu_rot = rmse_rotation_deg(cpu_out, gt);
    std::printf("GPU time: %.3f ms\n", bench.gpu_ms);
    std::printf("CPU time: %.3f ms\n", bench.cpu_ms);
    std::printf("Speedup: %.1fx\n", bench.speedup);
    std::printf("GPU switchable final RMSE: %.4f m, %.4f deg, cost %.2f\n",
                bench.final_trans_rmse, bench.final_rot_rmse_deg, bench.final_cost);
    std::printf("CPU switchable final RMSE: %.4f m, %.4f deg\n", cpu_trans, cpu_rot);
    std::printf("Final loop switch weights: clean avg %.3f, false avg %.3f, rejected false %d/%d\n",
                bench.clean_loop_weight, bench.outlier_loop_weight,
                bench.rejected_outliers, bench.outlier_edges);
    std::printf("Wrote gif/%s.gif\n", OUTPUT_STEM);
    return 0;
}
