// gpu_batched_ik.cu
//
// GPU batched inverse kinematics with random restarts (CPU vs CUDA comparison).
//
// Numerical IK for a 7-DOF arm (damped least squares on the full 6-D pose
// error, joint limits enforced by clamping) converges from a good initial guess
// and stalls in a local minimum or at a joint limit from a bad one. The standard
// remedy is random restarts: solve from many initial configurations and keep
// the best. On a CPU the restarts multiply the cost; on a GPU they are close to
// free, because every (target, restart) pair is an independent problem -- one
// thread = one IK solve, the repo's canonical idiom.
//
// Setup: a Franka-Panda-like arm (the published modified-DH table and joint
// limits), P target poses made by forward kinematics of random in-limit
// configurations (so every target is reachable), S random restarts per target.
// The solver is a single __host__ __device__ routine called both by a serial
// CPU loop and by the batch CUDA kernel.
//
// Reported: success rate (position < 1 mm and orientation < 1 deg) for the
// best of the first 1, 4, 8, ... S restarts, the CPU and GPU batch times, and
// the CPU/GPU agreement.
//
// Layout: [side and top views of the arm for one target, the restarts
// converging over the iterations] | [info panel].
//
// Output: gif/gpu_batched_ik.gif
//
// Second part, collision-aware IK: sphere obstacles in the workspace, the arm
// approximated by spheres along its links, and targets taken from collision-free
// configurations. The 7-DOF arm has one redundant degree of freedom; the
// collision-aware solver pushes the arm away from the obstacles in the null space
// of the pose task, so the clearance does not cost pose accuracy. Success there
// also needs a collision-free final configuration.
//
// Options: --no-video (skip the GIF), --check (exit non-zero unless the
// best-of-S success rates are >= 99% for the pose-only and >= 95% for the
// collision-aware problem, and CPU and GPU agree).

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

static const int N_TARGET = 1024;   // target poses
static const int N_SEED = 32;       // random restarts per target
static const int N_ITER = 80;       // DLS iterations per restart
static const int NJ = 7;            // joints

static constexpr float POS_TOL = 1.0e-3f;              // 1 mm
static constexpr float ORI_TOL = 1.0f * 3.14159265f / 180.0f;   // 1 deg

// Franka Panda, modified (Craig) DH: frame i = RotX(alpha) TransX(a) RotZ(q) TransZ(d),
// followed by the flange (d = 0.107, q = 0).
__host__ __device__ static inline float dh_a(int i) {
    const float a[8] = { 0.0f, 0.0f, 0.0f, 0.0825f, -0.0825f, 0.0f, 0.088f, 0.0f };
    return a[i];
}
__host__ __device__ static inline float dh_d(int i) {
    const float d[8] = { 0.333f, 0.0f, 0.316f, 0.0f, 0.384f, 0.0f, 0.0f, 0.107f };
    return d[i];
}
__host__ __device__ static inline float dh_alpha(int i) {
    const float h = 1.57079633f;
    const float al[8] = { 0.0f, -h, h, h, -h, h, h, 0.0f };
    return al[i];
}
__host__ __device__ static inline float q_lo(int i) {
    const float lo[7] = { -2.8973f, -1.7628f, -2.8973f, -3.0718f, -2.8973f, -0.0175f, -2.8973f };
    return lo[i];
}
__host__ __device__ static inline float q_hi(int i) {
    const float hi[7] = { 2.8973f, 1.7628f, 2.8973f, -0.0698f, 2.8973f, 3.7525f, 2.8973f };
    return hi[i];
}

// 3x4 homogeneous transform, row-major (rotation | translation).
struct Tf { float m[12]; };

__host__ __device__ static inline Tf tf_identity() {
    Tf t;
    for (int k = 0; k < 12; ++k) t.m[k] = 0.0f;
    t.m[0] = t.m[5] = t.m[10] = 1.0f;
    return t;
}

__host__ __device__ static inline Tf tf_mul(const Tf& A, const Tf& B) {
    Tf C;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c)
            C.m[r * 4 + c] = A.m[r * 4 + 0] * B.m[0 * 4 + c] + A.m[r * 4 + 1] * B.m[1 * 4 + c]
                           + A.m[r * 4 + 2] * B.m[2 * 4 + c];
        C.m[r * 4 + 3] = A.m[r * 4 + 0] * B.m[3] + A.m[r * 4 + 1] * B.m[7]
                       + A.m[r * 4 + 2] * B.m[11] + A.m[r * 4 + 3];
    }
    return C;
}

// Modified-DH link transform RotX(alpha) TransX(a) RotZ(q) TransZ(d).
__host__ __device__ static inline Tf dh_link(float a, float d, float alpha, float q) {
    float ca = cosf(alpha), sa = sinf(alpha), cq = cosf(q), sq = sinf(q);
    Tf t;
    t.m[0] = cq;       t.m[1] = -sq;      t.m[2] = 0.0f;  t.m[3] = a;
    t.m[4] = sq * ca;  t.m[5] = cq * ca;  t.m[6] = -sa;   t.m[7] = -sa * d;
    t.m[8] = sq * sa;  t.m[9] = cq * sa;  t.m[10] = ca;   t.m[11] = ca * d;
    return t;
}

// Forward kinematics: the flange pose and, optionally, every joint frame's
// origin and z axis (for the geometric Jacobian and for drawing).
__host__ __device__ static inline Tf fk(const float* q, float* origins = nullptr,
                                       float* zaxes = nullptr) {
    Tf T = tf_identity();
    for (int i = 0; i < 8; ++i) {
        T = tf_mul(T, dh_link(dh_a(i), dh_d(i), dh_alpha(i), i < NJ ? q[i] : 0.0f));
        if (origins) { origins[i * 3 + 0] = T.m[3]; origins[i * 3 + 1] = T.m[7]; origins[i * 3 + 2] = T.m[11]; }
        if (zaxes && i < NJ) { zaxes[i * 3 + 0] = T.m[2]; zaxes[i * 3 + 1] = T.m[6]; zaxes[i * 3 + 2] = T.m[10]; }
    }
    return T;
}

// Pose error toward the target: position error and the rotation vector of
// R_target * R^T (exact log map, so it stays valid for large errors).
__host__ __device__ static inline void pose_error(const Tf& cur, const Tf& tgt, float* e,
                                                  float* pos_err, float* ori_err) {
    e[0] = tgt.m[3] - cur.m[3];
    e[1] = tgt.m[7] - cur.m[7];
    e[2] = tgt.m[11] - cur.m[11];
    // R_err = R_t * R_c^T
    float R[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R[r * 3 + c] = tgt.m[r * 4 + 0] * cur.m[c * 4 + 0] + tgt.m[r * 4 + 1] * cur.m[c * 4 + 1]
                         + tgt.m[r * 4 + 2] * cur.m[c * 4 + 2];
    float tr = R[0] + R[4] + R[8];
    float cang = fminf(1.0f, fmaxf(-1.0f, 0.5f * (tr - 1.0f)));
    float ang = acosf(cang);
    float vx = R[7] - R[5], vy = R[2] - R[6], vz = R[3] - R[1];
    float s = sinf(ang);
    float k = s > 1.0e-4f ? 0.5f * ang / s : 0.5f;   // small angles: vee(R - R^T) / 2
    if (ang > 3.1f) {
        // near pi the antisymmetric part vanishes: take the axis from the symmetric part
        float ax = sqrtf(fmaxf(0.0f, 0.5f * (R[0] + 1.0f)));
        float ay = sqrtf(fmaxf(0.0f, 0.5f * (R[4] + 1.0f)));
        float az = sqrtf(fmaxf(0.0f, 0.5f * (R[8] + 1.0f)));
        if (vx < 0.0f) ax = -ax;
        if (vy < 0.0f) ay = -ay;
        if (vz < 0.0f) az = -az;
        e[3] = ang * ax; e[4] = ang * ay; e[5] = ang * az;
    } else {
        e[3] = k * vx; e[4] = k * vy; e[5] = k * vz;
    }
    *pos_err = sqrtf(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    *ori_err = ang;
}

// Solve the 6x6 SPD system A x = b in place by Cholesky (A is overwritten).
__host__ __device__ static inline void chol_solve6(float* A, float* b) {
    for (int j = 0; j < 6; ++j) {
        float s = A[j * 6 + j];
        for (int k = 0; k < j; ++k) s -= A[j * 6 + k] * A[j * 6 + k];
        float d = sqrtf(fmaxf(s, 1.0e-12f));
        A[j * 6 + j] = d;
        for (int i = j + 1; i < 6; ++i) {
            float t = A[i * 6 + j];
            for (int k = 0; k < j; ++k) t -= A[i * 6 + k] * A[j * 6 + k];
            A[i * 6 + j] = t / d;
        }
    }
    for (int i = 0; i < 6; ++i) {   // L y = b
        float t = b[i];
        for (int k = 0; k < i; ++k) t -= A[i * 6 + k] * b[k];
        b[i] = t / A[i * 6 + i];
    }
    for (int i = 5; i >= 0; --i) {  // L^T x = y
        float t = b[i];
        for (int k = i + 1; k < 6; ++k) t -= A[k * 6 + i] * b[k];
        b[i] = t / A[i * 6 + i];
    }
}

// Deterministic hash -> [0, 1), shared by host and device.
__host__ __device__ static inline float hash01(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int x = a * 73856093u ^ b * 19349663u ^ c * 83492791u ^ 0x9e3779b9u;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return (float)(x & 0x00ffffffu) / 16777216.0f;
}

__host__ __device__ static inline void random_config(unsigned int salt, unsigned int idx, float* q) {
    for (int j = 0; j < NJ; ++j) {
        float u = hash01(salt, idx, (unsigned int)j);
        q[j] = q_lo(j) + u * (q_hi(j) - q_lo(j));
    }
}

// One IK solve: damped least squares from q (in/out). Returns the final
// position and orientation errors; with trace, also the per-iteration q.
__host__ __device__ static inline void ik_solve(const Tf& target, float* q,
                                                float* pos_err, float* ori_err,
                                                float* trace = nullptr) {
    const float lambda2 = 0.01f;      // damping^2
    const float w_ori = 0.5f;         // metres per radian in the error weighting
    const float max_step = 0.3f;      // rad, per-iteration joint step norm cap
    float e[6], pe = 0.0f, oe = 0.0f;
    for (int it = 0; it < N_ITER; ++it) {
        float org[24], zax[21];
        Tf cur = fk(q, org, zax);
        pose_error(cur, target, e, &pe, &oe);
        if (trace) for (int j = 0; j < NJ; ++j) trace[it * NJ + j] = q[j];
        if (pe < POS_TOL && oe < ORI_TOL) {
            if (trace) for (int k = it + 1; k < N_ITER; ++k) for (int j = 0; j < NJ; ++j) trace[k * NJ + j] = q[j];
            break;
        }
        // geometric Jacobian (6x7): linear z_i x (p_e - p_i), angular z_i
        float J[6 * NJ];
        const float* pe_w = &org[7 * 3];   // flange origin
        for (int i = 0; i < NJ; ++i) {
            const float* z = &zax[i * 3];
            const float* p = &org[i * 3];
            float rx = pe_w[0] - p[0], ry = pe_w[1] - p[1], rz = pe_w[2] - p[2];
            J[0 * NJ + i] = z[1] * rz - z[2] * ry;
            J[1 * NJ + i] = z[2] * rx - z[0] * rz;
            J[2 * NJ + i] = z[0] * ry - z[1] * rx;
            J[3 * NJ + i] = w_ori * z[0];
            J[4 * NJ + i] = w_ori * z[1];
            J[5 * NJ + i] = w_ori * z[2];
        }
        float r[6] = { e[0], e[1], e[2], w_ori * e[3], w_ori * e[4], w_ori * e[5] };
        // (J J^T + lambda^2 I) y = r ;  dq = J^T y
        float A[36];
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) {
                float s = 0.0f;
                for (int i = 0; i < NJ; ++i) s += J[a * NJ + i] * J[b * NJ + i];
                A[a * 6 + b] = s + (a == b ? lambda2 : 0.0f);
            }
        chol_solve6(A, r);
        float dq[NJ], n2 = 0.0f;
        for (int i = 0; i < NJ; ++i) {
            float s = 0.0f;
            for (int a = 0; a < 6; ++a) s += J[a * NJ + i] * r[a];
            dq[i] = s;
            n2 += s * s;
        }
        float scale = n2 > max_step * max_step ? max_step / sqrtf(n2) : 1.0f;
        for (int i = 0; i < NJ; ++i)
            q[i] = fminf(q_hi(i), fmaxf(q_lo(i), q[i] + scale * dq[i]));
    }
    Tf cur = fk(q);
    pose_error(cur, target, e, pos_err, ori_err);
}

// ============================ collision-aware IK ============================
static const int MAX_OBS = 6;
static const int N_SEG_PTS = 3;          // collision spheres per link segment
static constexpr float LINK_R = 0.06f;       // link collision radius
static constexpr float CLEAR_MARGIN = 0.02f; // clearance the avoidance aims for

__constant__ float c_obs[MAX_OBS * 4];   // sphere obstacles: cx, cy, cz, radius

// Collision cost sum max(0, margin - clearance)^2 over the link spheres and the
// obstacles, its gradient in q (grad, may be null) and the minimum clearance.
// Segment k runs from the previous frame origin (the base for k = 0) to frame
// origin k; its points move with joints 0..k-1 (joint i turns about z_i through
// origin i).
__host__ __device__ static inline float collision_cost(const float* org, const float* zax,
                                                       const float* obs, int n_obs,
                                                       float* grad, float* min_clear) {
    float cost = 0.0f, mc = 1.0e9f;
    if (grad) for (int i = 0; i < NJ; ++i) grad[i] = 0.0f;
    for (int k = 0; k < 8; ++k) {
        float ax = k ? org[(k - 1) * 3 + 0] : 0.0f, ay = k ? org[(k - 1) * 3 + 1] : 0.0f;
        float az = k ? org[(k - 1) * 3 + 2] : 0.0f;
        float bx = org[k * 3 + 0], by = org[k * 3 + 1], bz = org[k * 3 + 2];
        for (int m = 0; m < N_SEG_PTS; ++m) {
            float u = (m + 0.5f) / N_SEG_PTS;
            float px = ax + u * (bx - ax), py = ay + u * (by - ay), pz = az + u * (bz - az);
            for (int o = 0; o < n_obs; ++o) {
                float dx = px - obs[o * 4 + 0], dy = py - obs[o * 4 + 1], dz = pz - obs[o * 4 + 2];
                float dist = sqrtf(dx * dx + dy * dy + dz * dz) + 1.0e-9f;
                float cl = dist - LINK_R - obs[o * 4 + 3];
                mc = fminf(mc, cl);
                float pen = CLEAR_MARGIN - cl;
                if (pen <= 0.0f) continue;
                cost += pen * pen;
                if (!grad) continue;
                // d(cost)/dq_i = -2 pen * n . (z_i x (p - o_i)),  n = (p - c) / |p - c|
                float nx = dx / dist, ny = dy / dist, nz = dz / dist;
                for (int i = 0; i < k && i < NJ; ++i) {
                    const float* z = &zax[i * 3];
                    float rx = px - org[i * 3 + 0], ry = py - org[i * 3 + 1], rz = pz - org[i * 3 + 2];
                    float vx = z[1] * rz - z[2] * ry, vy = z[2] * rx - z[0] * rz, vz = z[0] * ry - z[1] * rx;
                    grad[i] += -2.0f * pen * (nx * vx + ny * vy + nz * vz);
                }
            }
        }
    }
    if (min_clear) *min_clear = mc;
    return cost;
}

// Damped least squares as ik_solve, plus (with avoid) a step down the collision
// cost's gradient projected into the null space of the pose task. Returns the
// final pose errors and the final minimum clearance.
__host__ __device__ static inline void ik_solve_avoid(const Tf& target, float* q,
                                                      const float* obs, int n_obs, bool avoid,
                                                      float* pos_err, float* ori_err,
                                                      float* clearance, float* trace = nullptr) {
    const float lambda2 = 0.01f, w_ori = 0.5f, max_step = 0.3f;
    const float k_null = 20.0f;       // null-space step gain on the collision gradient
    float e[6], pe = 0.0f, oe = 0.0f, cl = 0.0f;
    for (int it = 0; it < N_ITER; ++it) {
        float org[24], zax[21];
        Tf cur = fk(q, org, zax);
        pose_error(cur, target, e, &pe, &oe);
        float g[NJ];
        float cost = collision_cost(org, zax, obs, n_obs, avoid ? g : nullptr, &cl);
        if (trace) for (int j = 0; j < NJ; ++j) trace[it * NJ + j] = q[j];
        if (pe < POS_TOL && oe < ORI_TOL && (!avoid || cl > 0.0f)) {
            if (trace) for (int k = it + 1; k < N_ITER; ++k) for (int j = 0; j < NJ; ++j) trace[k * NJ + j] = q[j];
            break;
        }
        float J[6 * NJ];
        const float* pe_w = &org[7 * 3];
        for (int i = 0; i < NJ; ++i) {
            const float* z = &zax[i * 3];
            const float* p = &org[i * 3];
            float rx = pe_w[0] - p[0], ry = pe_w[1] - p[1], rz = pe_w[2] - p[2];
            J[0 * NJ + i] = z[1] * rz - z[2] * ry;
            J[1 * NJ + i] = z[2] * rx - z[0] * rz;
            J[2 * NJ + i] = z[0] * ry - z[1] * rx;
            J[3 * NJ + i] = w_ori * z[0];
            J[4 * NJ + i] = w_ori * z[1];
            J[5 * NJ + i] = w_ori * z[2];
        }
        float A[36], L[36];
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) {
                float sum = 0.0f;
                for (int i = 0; i < NJ; ++i) sum += J[a * NJ + i] * J[b * NJ + i];
                A[a * 6 + b] = sum + (a == b ? lambda2 : 0.0f);
            }
        float r[6] = { e[0], e[1], e[2], w_ori * e[3], w_ori * e[4], w_ori * e[5] };
        for (int k = 0; k < 36; ++k) L[k] = A[k];
        chol_solve6(L, r);
        float dq[NJ];
        for (int i = 0; i < NJ; ++i) {
            float sum = 0.0f;
            for (int a = 0; a < 6; ++a) sum += J[a * NJ + i] * r[a];
            dq[i] = sum;
        }
        if (avoid && cost > 0.0f) {
            // null-space projection: g - J^T (J J^T + lambda^2 I)^-1 J g
            float v[6];
            for (int a = 0; a < 6; ++a) {
                float sum = 0.0f;
                for (int i = 0; i < NJ; ++i) sum += J[a * NJ + i] * g[i];
                v[a] = sum;
            }
            for (int k = 0; k < 36; ++k) L[k] = A[k];
            chol_solve6(L, v);
            for (int i = 0; i < NJ; ++i) {
                float jw = 0.0f;
                for (int a = 0; a < 6; ++a) jw += J[a * NJ + i] * v[a];
                dq[i] -= k_null * (g[i] - jw);
            }
        }
        float n2 = 0.0f;
        for (int i = 0; i < NJ; ++i) n2 += dq[i] * dq[i];
        float scale = n2 > max_step * max_step ? max_step / sqrtf(n2) : 1.0f;
        for (int i = 0; i < NJ; ++i)
            q[i] = fminf(q_hi(i), fmaxf(q_lo(i), q[i] + scale * dq[i]));
    }
    float org[24], zax[21];
    Tf cur = fk(q, org, zax);
    pose_error(cur, target, e, pos_err, ori_err);
    collision_cost(org, zax, obs, n_obs, nullptr, clearance);
}

__global__ void ik_avoid_kernel(const Tf* __restrict__ targets, int n_obs, bool avoid,
                                float* __restrict__ pos_err, float* __restrict__ ori_err,
                                float* __restrict__ clearance, int n_target, int n_seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_target * n_seed) return;
    int t = idx / n_seed, s = idx - t * n_seed;
    float q[NJ];
    random_config(4u, (unsigned int)(t * n_seed + s), q);
    ik_solve_avoid(targets[t], q, c_obs, n_obs, avoid, &pos_err[idx], &ori_err[idx], &clearance[idx]);
}

// one thread = one (target, restart) solve
__global__ void ik_batch_kernel(const Tf* __restrict__ targets, float* __restrict__ q_out,
                                float* __restrict__ pos_err, float* __restrict__ ori_err,
                                int n_target, int n_seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_target * n_seed) return;
    int t = idx / n_seed, s = idx - t * n_seed;
    float q[NJ];
    random_config(2u, (unsigned int)(t * n_seed + s), q);
    float pe, oe;
    ik_solve(targets[t], q, &pe, &oe);
    for (int j = 0; j < NJ; ++j) q_out[idx * NJ + j] = q[j];
    pos_err[idx] = pe;
    ori_err[idx] = oe;
}

}  // namespace cudabot

using namespace cudabot;

static bool solved(float pe, float oe) { return pe < POS_TOL && oe < ORI_TOL; }

// Fraction of targets solved by the best of the first k restarts.
static double success_rate(const std::vector<float>& pe, const std::vector<float>& oe, int k) {
    int ok = 0;
    for (int t = 0; t < N_TARGET; ++t)
        for (int s = 0; s < k; ++s)
            if (solved(pe[t * N_SEED + s], oe[t * N_SEED + s])) { ++ok; break; }
    return (double)ok / N_TARGET;
}

// Fraction of targets solved collision-free by the best of the first k restarts.
static double success_rate_cf(const std::vector<float>& pe, const std::vector<float>& oe,
                              const std::vector<float>& cl, int k) {
    int ok = 0;
    for (int t = 0; t < N_TARGET; ++t)
        for (int s = 0; s < k; ++s) {
            int i = t * N_SEED + s;
            if (solved(pe[i], oe[i]) && cl[i] > 0.0f) { ++ok; break; }
        }
    return (double)ok / N_TARGET;
}

// ---- drawing ----
static const int FRAME_W = 1200, FRAME_H = 560, VIEW = 420;

static cv::Point view_px(float u, float v, int ox, int oy, float umin, float vmin, float span) {
    return cv::Point(ox + (int)((u - umin) / span * VIEW), oy + VIEW - (int)((v - vmin) / span * VIEW));
}

static void draw_arm(cv::Mat& img, const float* q, int ox, int oy, bool side,
                     cv::Scalar col, int thick) {
    float org[24];
    fk(q, org);
    std::vector<cv::Point> pts;
    pts.push_back(side ? view_px(0, 0, ox, oy, -0.9f, -0.2f, 1.8f) : view_px(0, 0, ox, oy, -0.9f, -0.9f, 1.8f));
    for (int i = 0; i < 8; ++i) {
        float x = org[i * 3 + 0], y = org[i * 3 + 1], z = org[i * 3 + 2];
        pts.push_back(side ? view_px(x, z, ox, oy, -0.9f, -0.2f, 1.8f)
                           : view_px(x, y, ox, oy, -0.9f, -0.9f, 1.8f));
    }
    cv::polylines(img, pts, false, col, thick, cv::LINE_AA);
    for (size_t i = 1; i < pts.size(); ++i) cv::circle(img, pts[i], thick + 1, col, -1, cv::LINE_AA);
}

int main(int argc, char** argv) {
    bool no_video = false, check = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--no-video")) no_video = true;
        else if (!std::strcmp(argv[i], "--check")) check = true;
    }
    std::printf("=== GPU batched inverse kinematics with random restarts (CPU vs CUDA) ===\n");

    // ---- reachable targets: FK of random in-limit configurations ----
    std::vector<Tf> targets(N_TARGET);
    std::vector<float> q_true(N_TARGET * NJ);
    for (int t = 0; t < N_TARGET; ++t) {
        random_config(1u, (unsigned int)t, &q_true[t * NJ]);
        targets[t] = fk(&q_true[t * NJ]);
    }
    const int total = N_TARGET * N_SEED;

    // ---- CPU batch (serial, same solver) ----
    std::vector<float> cpu_q(total * NJ), cpu_pe(total), cpu_oe(total);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < N_TARGET; ++t)
        for (int s = 0; s < N_SEED; ++s) {
            int idx = t * N_SEED + s;
            float* q = &cpu_q[idx * NJ];
            random_config(2u, (unsigned int)idx, q);
            ik_solve(targets[t], q, &cpu_pe[idx], &cpu_oe[idx]);
        }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- GPU batch ----
    Tf* d_targets;
    float *d_q, *d_pe, *d_oe;
    CUDA_CHECK(cudaMalloc(&d_targets, N_TARGET * sizeof(Tf)));
    CUDA_CHECK(cudaMalloc(&d_q, total * NJ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pe, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_oe, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_targets, targets.data(), N_TARGET * sizeof(Tf), cudaMemcpyHostToDevice));
    int block = 128, grid = (total + block - 1) / block;
    ik_batch_kernel<<<grid, block>>>(d_targets, d_q, d_pe, d_oe, N_TARGET, N_SEED);   // warm up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    ik_batch_kernel<<<grid, block>>>(d_targets, d_q, d_pe, d_oe, N_TARGET, N_SEED);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));
    std::vector<float> gpu_pe(total), gpu_oe(total);
    CUDA_CHECK(cudaMemcpy(gpu_pe.data(), d_pe, total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_oe.data(), d_oe, total * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- results ----
    const int ks[] = { 1, 4, 8, 16, 32 };
    std::printf("targets               : %d  (%d restarts each, %d DLS iterations max)\n",
                N_TARGET, N_SEED, N_ITER);
    std::printf("success = position < %.1f mm and orientation < %.1f deg\n",
                POS_TOL * 1e3f, ORI_TOL * 180.0f / 3.14159265f);
    double gpu_rate_k[5], cpu_rate_k[5];
    for (int i = 0; i < 5; ++i) {
        gpu_rate_k[i] = success_rate(gpu_pe, gpu_oe, ks[i]);
        cpu_rate_k[i] = success_rate(cpu_pe, cpu_oe, ks[i]);
        std::printf("best of %2d restarts   : GPU %6.2f%%   CPU %6.2f%%\n", ks[i],
                    100.0 * gpu_rate_k[i], 100.0 * cpu_rate_k[i]);
    }
    // per-solve agreement: same success outcome on the same (target, restart)
    int agree = 0;
    for (int i = 0; i < total; ++i)
        if (solved(cpu_pe[i], cpu_oe[i]) == solved(gpu_pe[i], gpu_oe[i])) ++agree;
    double agree_pct = 100.0 * agree / total;
    double speedup = cpu_ms / gpu_ms;
    std::printf("CPU serial batch      : %9.2f ms  (%d solves)\n", cpu_ms, total);
    std::printf("GPU batch kernel      : %9.2f ms  (%.0fx)\n", gpu_ms, speedup);
    std::printf("per solve CPU / GPU   : %.2f us / %.4f us\n", cpu_ms * 1e3 / total, gpu_ms * 1e3 / total);
    std::printf("CPU/GPU same outcome  : %.2f%% of solves\n", agree_pct);

    bool ok = gpu_rate_k[4] >= 0.99 && agree_pct >= 99.0;

    // ================= collision-aware IK =================
    const float h_obs[MAX_OBS * 4] = {
        0.45f,  0.00f, 0.45f, 0.15f,
        0.30f,  0.30f, 0.30f, 0.13f,
        0.30f, -0.30f, 0.55f, 0.13f,
        0.05f,  0.45f, 0.60f, 0.12f,
        0.55f,  0.25f, 0.75f, 0.12f,
        0.55f, -0.25f, 0.20f, 0.12f,
    };
    const int n_obs = MAX_OBS;
    CUDA_CHECK(cudaMemcpyToSymbol(c_obs, h_obs, sizeof(h_obs)));
    // Reachable, collision-free targets close to the obstacles: the flange within
    // 0.12 m of an obstacle surface, from a configuration with at least the
    // avoidance margin of clearance (so a collision-free solution exists).
    std::vector<Tf> cf_targets(N_TARGET);
    int drawn = 0;
    for (int t = 0; t < N_TARGET; ++t) {
        float q[NJ], org[24], zax[21], cl, near;
        do {
            random_config(3u, (unsigned int)drawn++, q);
            fk(q, org, zax);
            collision_cost(org, zax, h_obs, n_obs, nullptr, &cl);
            near = 1.0e9f;
            for (int o = 0; o < n_obs; ++o) {
                float dx = org[21] - h_obs[o * 4 + 0], dy = org[22] - h_obs[o * 4 + 1];
                float dz = org[23] - h_obs[o * 4 + 2];
                near = fminf(near, sqrtf(dx * dx + dy * dy + dz * dz) - h_obs[o * 4 + 3]);
            }
        } while (cl < CLEAR_MARGIN || near > 0.12f);
        cf_targets[t] = fk(q);
    }
    std::printf("\n--- collision-aware IK: %d sphere obstacles, %d collision-free targets near "
                "them (%d configurations drawn) ---\n", n_obs, N_TARGET, drawn);
    CUDA_CHECK(cudaMemcpy(d_targets, cf_targets.data(), N_TARGET * sizeof(Tf), cudaMemcpyHostToDevice));
    float* d_cl;
    CUDA_CHECK(cudaMalloc(&d_cl, total * sizeof(float)));
    std::vector<float> pl_pe(total), pl_oe(total), pl_cl(total);
    std::vector<float> av_pe(total), av_oe(total), av_cl(total);
    float av_ms = 0.0f;
    for (int pass = 0; pass < 2; ++pass) {
        bool avoid = pass == 1;
        ik_avoid_kernel<<<grid, block>>>(d_targets, n_obs, avoid, d_pe, d_oe, d_cl, N_TARGET, N_SEED);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(e0));
        ik_avoid_kernel<<<grid, block>>>(d_targets, n_obs, avoid, d_pe, d_oe, d_cl, N_TARGET, N_SEED);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        if (avoid) CUDA_CHECK(cudaEventElapsedTime(&av_ms, e0, e1));
        std::vector<float>& pe = avoid ? av_pe : pl_pe;
        std::vector<float>& oe = avoid ? av_oe : pl_oe;
        std::vector<float>& cl = avoid ? av_cl : pl_cl;
        CUDA_CHECK(cudaMemcpy(pe.data(), d_pe, total * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(oe.data(), d_oe, total * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cl.data(), d_cl, total * sizeof(float), cudaMemcpyDeviceToHost));
    }
    // CPU reference for the collision-aware solver (same seeds)
    std::vector<float> cav_pe(total), cav_oe(total), cav_cl(total);
    auto c0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < N_TARGET; ++t)
        for (int s = 0; s < N_SEED; ++s) {
            int idx = t * N_SEED + s;
            float q[NJ];
            random_config(4u, (unsigned int)idx, q);
            ik_solve_avoid(cf_targets[t], q, h_obs, n_obs, true, &cav_pe[idx], &cav_oe[idx], &cav_cl[idx]);
        }
    auto c1 = std::chrono::high_resolution_clock::now();
    double cav_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
    double pl_rate[5], av_rate[5];
    std::printf("solved collision-free : pose-only DLS   null-space avoidance\n");
    for (int i = 0; i < 5; ++i) {
        pl_rate[i] = success_rate_cf(pl_pe, pl_oe, pl_cl, ks[i]);
        av_rate[i] = success_rate_cf(av_pe, av_oe, av_cl, ks[i]);
        std::printf("best of %2d restarts   : %6.2f%%          %6.2f%%\n", ks[i],
                    100.0 * pl_rate[i], 100.0 * av_rate[i]);
    }
    int pl_pose_ok = 0, pl_colliding = 0;
    for (int i = 0; i < total; ++i)
        if (solved(pl_pe[i], pl_oe[i])) { ++pl_pose_ok; if (pl_cl[i] <= 0.0f) ++pl_colliding; }
    std::printf("pose-only solves that collide: %d of %d pose solutions (%.1f%%)\n",
                pl_colliding, pl_pose_ok, 100.0 * pl_colliding / std::max(1, pl_pose_ok));
    int av_agree = 0;
    for (int i = 0; i < total; ++i) {
        bool g = solved(av_pe[i], av_oe[i]) && av_cl[i] > 0.0f;
        bool c = solved(cav_pe[i], cav_oe[i]) && cav_cl[i] > 0.0f;
        if (g == c) ++av_agree;
    }
    double av_agree_pct = 100.0 * av_agree / total;
    std::printf("collision-aware CPU / GPU : %.2f ms / %.2f ms  (%.0fx), same outcome %.2f%%\n",
                cav_ms, av_ms, cav_ms / av_ms, av_agree_pct);
    CUDA_CHECK(cudaFree(d_cl));
    ok = ok && av_rate[4] >= 0.95 && av_agree_pct >= 99.0;

    if (!no_video) {
        // one target, its restarts converging over the iterations (host trace)
        const int VT = 7;
        std::vector<std::vector<float>> trace(N_SEED, std::vector<float>(N_ITER * NJ));
        std::vector<int> win(N_SEED);
        for (int s = 0; s < N_SEED; ++s) {
            float q[NJ], pe, oe;
            random_config(2u, (unsigned int)(VT * N_SEED + s), q);
            ik_solve(targets[VT], q, &pe, &oe, trace[s].data());
            win[s] = solved(pe, oe);
        }
        if (ensure_dirs({ "tmp" }) != 0) std::fprintf(stderr, "warning: mkdir tmp failed\n");
        cv::VideoWriter video("tmp/gpu_batched_ik.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 12,
                              cv::Size(FRAME_W, FRAME_H));
        const int SHOW_ITERS = 40, HOLD = 14;
        float org_t[24];
        fk(&q_true[VT * NJ], org_t);
        for (int frame = 0; frame < SHOW_ITERS + HOLD; ++frame) {
            int it = std::min(frame, SHOW_ITERS - 1);
            cv::Mat img(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(28, 28, 32));
            const int ox1 = 30, ox2 = 30 + VIEW + 30, oy = 90;
            cv::rectangle(img, cv::Rect(ox1, oy, VIEW, VIEW), cv::Scalar(70, 70, 80), 1);
            cv::rectangle(img, cv::Rect(ox2, oy, VIEW, VIEW), cv::Scalar(70, 70, 80), 1);
            cv::putText(img, "side (x-z)", cv::Point(ox1, oy - 12), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(180, 180, 200), 1, cv::LINE_AA);
            cv::putText(img, "top (x-y)", cv::Point(ox2, oy - 12), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(180, 180, 200), 1, cv::LINE_AA);
            // draw up to 4 restarts that stall (grey) under 4 that reach the target (green)
            for (int pass = 0; pass < 2; ++pass) {
                int shown = 0;
                for (int s = 0; s < N_SEED && shown < 4; ++s) {
                    if (win[s] != pass) continue;
                    ++shown;
                    cv::Scalar col = pass ? cv::Scalar(110, 210, 110) : cv::Scalar(120, 120, 130);
                    const float* q = &trace[s][it * NJ];
                    draw_arm(img, q, ox1, oy, true, col, pass ? 2 : 1);
                    draw_arm(img, q, ox2, oy, false, col, pass ? 2 : 1);
                }
            }
            cv::Point tp1 = view_px(org_t[21], org_t[23], ox1, oy, -0.9f, -0.2f, 1.8f);
            cv::Point tp2 = view_px(org_t[21], org_t[22], ox2, oy, -0.9f, -0.9f, 1.8f);
            cv::drawMarker(img, tp1, cv::Scalar(80, 220, 250), cv::MARKER_TILTED_CROSS, 16, 2);
            cv::drawMarker(img, tp2, cv::Scalar(80, 220, 250), cv::MARKER_TILTED_CROSS, 16, 2);

            int px = ox2 + VIEW + 30, py = 60;
            auto put = [&](const std::string& s, int yy, double sc, cv::Scalar col, int th) {
                cv::putText(img, s, cv::Point(px, yy), cv::FONT_HERSHEY_SIMPLEX, sc, col, th, cv::LINE_AA);
            };
            char buf[128];
            put("Batched IK", py, 0.9, cv::Scalar(235, 235, 245), 2); py += 30;
            put("7-DOF, random restarts", py, 0.6, cv::Scalar(180, 180, 200), 1); py += 36;
            std::snprintf(buf, sizeof(buf), "DLS iteration %d", it + 1);
            put(buf, py, 0.58, cv::Scalar(210, 210, 225), 1); py += 26;
            int nwin = 0; for (int s = 0; s < N_SEED; ++s) nwin += win[s];
            std::snprintf(buf, sizeof(buf), "%d restarts, %d reach", N_SEED, nwin);
            put(buf, py, 0.58, cv::Scalar(210, 210, 225), 1); py += 26;
            put("shown: 4 that reach (green)", py, 0.48, cv::Scalar(110, 210, 110), 1); py += 22;
            put("and 4 that stall (grey)", py, 0.48, cv::Scalar(150, 150, 160), 1); py += 36;
            put("Batch headline", py, 0.62, cv::Scalar(150, 220, 150), 1); py += 28;
            std::snprintf(buf, sizeof(buf), "%d targets x %d", N_TARGET, N_SEED);
            put(buf, py, 0.52, cv::Scalar(200, 200, 210), 1); py += 24;
            std::snprintf(buf, sizeof(buf), "best of 1  : %.1f%%", 100.0 * gpu_rate_k[0]);
            put(buf, py, 0.52, cv::Scalar(200, 200, 210), 1); py += 24;
            std::snprintf(buf, sizeof(buf), "best of %d : %.1f%%", N_SEED, 100.0 * gpu_rate_k[4]);
            put(buf, py, 0.52, cv::Scalar(200, 200, 210), 1); py += 24;
            std::snprintf(buf, sizeof(buf), "CPU %.0f ms", cpu_ms);
            put(buf, py, 0.52, cv::Scalar(200, 200, 210), 1); py += 24;
            std::snprintf(buf, sizeof(buf), "GPU %.2f ms", gpu_ms);
            put(buf, py, 0.52, cv::Scalar(200, 200, 210), 1); py += 24;
            std::snprintf(buf, sizeof(buf), "speedup %.0fx", speedup);
            put(buf, py, 0.6, cv::Scalar(120, 230, 250), 2);
            video.write(img);
        }
        video.release();
        avi_to_gif("tmp/gpu_batched_ik.avi", "gif/gpu_batched_ik.gif", 12, 900);
        std::printf("wrote gif/gpu_batched_ik.gif\n");
    }

    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_pe));
    CUDA_CHECK(cudaFree(d_oe));
    if (check) {
        std::printf("check: %s (best-of-%d success >= 99%% pose-only and >= 95%% collision-aware, "
                    "CPU/GPU agreement >= 99%%)\n", ok ? "PASS" : "FAIL", N_SEED);
        return ok ? 0 : 1;
    }
    return 0;
}
