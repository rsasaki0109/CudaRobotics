// gpu_parallel_ilqr.cu
//
// Parallel-in-TIME iLQR: GPU trajectory optimization that parallelizes the
// solve of a SINGLE problem across the horizon, not across a batch of problems.
//
// The repo already has gpu_batched_ilqr.cu, which parallelizes ACROSS problems
// (one thread = one independent start/goal query): great THROUGHPUT, but every
// single solve still runs the inherently sequential O(T) backward Riccati sweep.
// For a real-time MPC controller re-solving ONE problem every control step, the
// quantity that matters is per-problem LATENCY, and a batch kernel does nothing
// for it.
//
// This demo closes that gap.  It implements the parallel-in-time LQR of
//   Sarkka & Garcia-Fernandez, "Temporal Parallelisation of Dynamic Programming
//   and Linear Quadratic Control", IEEE TAC (arXiv:2104.03186),
// which recasts the backward value-function recursion as an ASSOCIATIVE SCAN:
// each time step contributes an element (A,b,C,eta,J) parameterizing a
// conditional value function, the elements combine with an associative operator
// (Lemma 10, eq. 36), and a Blelloch scan aggregates all T of them in O(log T)
// span instead of O(T).  The aggregated (J,eta) are exactly the sequential
// Riccati (S_k, v_k) -- so the result is bit-comparable, only the latency drops.
//
// Layout of this file (built up in verifiable stages):
//   1. Shared unicycle dynamics + obstacle/goal cost  (identical to the batched
//      demo, so the existing sequential iLQR is a ground-truth oracle).
//   2. Sequential iLQR  (ilqr_solve_seq) -- the oracle.
//   3. Small fixed-size linear algebra (3x3 / 2x2) for the scan elements.
//   4. Backward element (A,b,C,eta,J), the associative combine operator, gain
//      extraction, and a host iLQR that drives the backward pass through the
//      ELEMENTS combined sequentially (a left-fold).  Associativity guarantees a
//      sequential fold and a parallel scan give the same answer, so this host
//      path validates the element MATH against the oracle before any GPU scan.
//   5. (next) CUDA kernel: one block = one problem, threads = time steps, real
//      Blelloch up/down-sweep in shared memory; latency-vs-T headline + GIF.
//
// Build: see CMakeLists (uses --expt-relaxed-constexpr --fmad=false so host and
// device do bit-for-bit identical FP, the same reproducibility contract the
// batched demo relies on).

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ---- problem dimensions (compile-time so per-thread arrays are fixed size) ----
#define NX 3            // state: px, py, theta
#define NU 2            // control: v, omega
#define T_MAX 256       // largest horizon we sweep (shared-memory sizing)
#define N_ILQR 25       // iLQR iterations
#define MAX_OBS 4

static const float DT = 0.1f;

// ---- cost weights (shared host+device, identical to gpu_batched_ilqr.cu) ----
__host__ __device__ static inline float w_ctrl_v() { return 0.02f; }
__host__ __device__ static inline float w_ctrl_w() { return 0.01f; }
__host__ __device__ static inline float w_stage()  { return 0.20f; }
__host__ __device__ static inline float w_term()   { return 80.0f; }
__host__ __device__ static inline float w_obs()    { return 60.0f; }
__host__ __device__ static inline float obs_margin() { return 0.40f; }

// ============================ shared dynamics / cost ============================
// All __host__ __device__ so CPU and GPU run identical arithmetic.

__host__ __device__ static inline void dyn_step(const float* x, const float* u, float* xn) {
    float c = cosf(x[2]), s = sinf(x[2]);
    xn[0] = x[0] + u[0] * c * DT;
    xn[1] = x[1] + u[0] * s * DT;
    xn[2] = x[2] + u[1] * DT;
}

__host__ __device__ static inline void dyn_jac(const float* x, const float* u,
                                               float* fx, float* fu) {
    float c = cosf(x[2]), s = sinf(x[2]);
    for (int i = 0; i < NX * NX; ++i) fx[i] = 0.f;
    for (int i = 0; i < NX * NU; ++i) fu[i] = 0.f;
    fx[0] = 1.f; fx[4] = 1.f; fx[8] = 1.f;
    fx[0 * NX + 2] = -u[0] * s * DT;
    fx[1 * NX + 2] =  u[0] * c * DT;
    fu[0 * NU + 0] = c * DT;
    fu[1 * NU + 0] = s * DT;
    fu[2 * NU + 1] = DT;
}

// Stage cost + Gauss-Newton derivatives.  l_ux == 0 (separable), so omitted.
__host__ __device__ static inline float stage_cost(const float* x, const float* u,
                                                   const float* obs, int n_obs,
                                                   float gx, float gy,
                                                   float* l_x, float* l_u,
                                                   float* l_xx, float* l_uu) {
    for (int i = 0; i < NX; ++i) l_x[i] = 0.f;
    for (int i = 0; i < NU; ++i) l_u[i] = 0.f;
    for (int i = 0; i < NX * NX; ++i) l_xx[i] = 0.f;
    for (int i = 0; i < NU * NU; ++i) l_uu[i] = 0.f;

    float l = 0.f;
    float rv = w_ctrl_v(), rw = w_ctrl_w();
    l += 0.5f * (rv * u[0] * u[0] + rw * u[1] * u[1]);
    l_u[0] = rv * u[0]; l_u[1] = rw * u[1];
    l_uu[0] = rv; l_uu[3] = rw;
    float q = w_stage();
    float dx = x[0] - gx, dy = x[1] - gy;
    l += 0.5f * q * (dx * dx + dy * dy);
    l_x[0] += q * dx; l_x[1] += q * dy;
    l_xx[0] += q; l_xx[4] += q;
    float wo = w_obs(), m = obs_margin();
    for (int o = 0; o < n_obs; ++o) {
        float ox = obs[o * 3 + 0], oy = obs[o * 3 + 1], orr = obs[o * 3 + 2];
        float ex = x[0] - ox, ey = x[1] - oy;
        float d = sqrtf(ex * ex + ey * ey) + 1e-6f;
        float g = (orr + m) - d;
        if (g > 0.f) {
            l += 0.5f * wo * g * g;
            float gx0 = -ex / d, gx1 = -ey / d;
            l_x[0] += wo * g * gx0; l_x[1] += wo * g * gx1;
            l_xx[0] += wo * gx0 * gx0; l_xx[1] += wo * gx0 * gx1;
            l_xx[3] += wo * gx1 * gx0; l_xx[4] += wo * gx1 * gx1;
        }
    }
    return l;
}

__host__ __device__ static inline float term_cost(const float* x, float gx, float gy,
                                                  float* l_x, float* l_xx) {
    for (int i = 0; i < NX; ++i) l_x[i] = 0.f;
    for (int i = 0; i < NX * NX; ++i) l_xx[i] = 0.f;
    float w = w_term();
    float dx = x[0] - gx, dy = x[1] - gy;
    l_x[0] = w * dx; l_x[1] = w * dy;
    l_xx[0] = w; l_xx[4] = w;
    return 0.5f * w * (dx * dx + dy * dy);
}

__host__ __device__ static inline float rollout(const float* u, const float* obs, int n_obs,
                                                float sx, float sy, float sth,
                                                float gx, float gy, float* xs, int T) {
    float x[NX]; x[0] = sx; x[1] = sy; x[2] = sth;
    for (int i = 0; i < NX; ++i) xs[i] = x[i];
    float l_x[NX], l_u[NU], l_xx[NX * NX], l_uu[NU * NU];
    float cost = 0.f;
    for (int t = 0; t < T; ++t) {
        cost += stage_cost(x, &u[t * NU], obs, n_obs, gx, gy, l_x, l_u, l_xx, l_uu);
        float xn[NX];
        dyn_step(x, &u[t * NU], xn);
        for (int i = 0; i < NX; ++i) { x[i] = xn[i]; xs[(t + 1) * NX + i] = xn[i]; }
    }
    float lf_x[NX], lf_xx[NX * NX];
    cost += term_cost(x, gx, gy, lf_x, lf_xx);
    return cost;
}

// Total cost of an arbitrary (xs,u) rollout already simulated into xs.
__host__ __device__ static inline float traj_cost(const float* xs, const float* u,
                                                  const float* obs, int n_obs,
                                                  float gx, float gy, int T) {
    float c = 0.f;
    float l_x[NX], l_u[NU], l_xx[NX * NX], l_uu[NU * NU];
    for (int t = 0; t < T; ++t)
        c += stage_cost(&xs[t * NX], &u[t * NU], obs, n_obs, gx, gy, l_x, l_u, l_xx, l_uu);
    float lf_x[NX], lf_xx[NX * NX];
    c += term_cost(&xs[T * NX], gx, gy, lf_x, lf_xx);
    return c;
}

// initial control guess: drive straight toward the goal.
__host__ __device__ static inline void init_controls(float* u, float sx, float sy,
                                                     float gx, float gy, int T) {
    float dgx = gx - sx, dgy = gy - sy;
    float v0 = sqrtf(dgx * dgx + dgy * dgy) / (T * DT);
    if (v0 > 2.5f) v0 = 2.5f;
    for (int t = 0; t < T; ++t) { u[t * NU + 0] = v0; u[t * NU + 1] = 0.f; }
}

// ============================ small linear algebra ============================
// Fixed-size 3x3 / 2x2 helpers, row-major, __host__ __device__.

__host__ __device__ static inline void m3_mul(const float* A, const float* B, float* C) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float s = 0.f;
            for (int k = 0; k < 3; ++k) s += A[i * 3 + k] * B[k * 3 + j];
            C[i * 3 + j] = s;
        }
}
__host__ __device__ static inline void m3_mul_T_left(const float* A, const float* B, float* C) {
    // C = A^T * B   (all 3x3)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float s = 0.f;
            for (int k = 0; k < 3; ++k) s += A[k * 3 + i] * B[k * 3 + j];
            C[i * 3 + j] = s;
        }
}
__host__ __device__ static inline void m3_matvec(const float* A, const float* v, float* o) {
    for (int i = 0; i < 3; ++i) o[i] = A[i * 3 + 0] * v[0] + A[i * 3 + 1] * v[1] + A[i * 3 + 2] * v[2];
}
__host__ __device__ static inline void m3_matvec_T(const float* A, const float* v, float* o) {
    for (int i = 0; i < 3; ++i) o[i] = A[0 * 3 + i] * v[0] + A[1 * 3 + i] * v[1] + A[2 * 3 + i] * v[2];
}
__host__ __device__ static inline void m3_add(const float* A, const float* B, float* C) {
    for (int i = 0; i < 9; ++i) C[i] = A[i] + B[i];
}
__host__ __device__ static inline void v3_add(const float* a, const float* b, float* c) {
    for (int i = 0; i < 3; ++i) c[i] = a[i] + b[i];
}
__host__ __device__ static inline void m3_identity(float* A) {
    for (int i = 0; i < 9; ++i) A[i] = 0.f;
    A[0] = A[4] = A[8] = 1.f;
}
// 3x3 inverse via cofactors; returns false if near-singular.
__host__ __device__ static inline bool m3_inv(const float* A, float* I) {
    float a = A[0], b = A[1], c = A[2];
    float d = A[3], e = A[4], f = A[5];
    float g = A[6], h = A[7], i = A[8];
    float A00 =  (e * i - f * h);
    float A01 = -(d * i - f * g);
    float A02 =  (d * h - e * g);
    float det = a * A00 + b * A01 + c * A02;
    if (fabsf(det) < 1e-20f) return false;
    float invdet = 1.f / det;
    I[0] =  A00 * invdet;
    I[1] = -(b * i - c * h) * invdet;
    I[2] =  (b * f - c * e) * invdet;
    I[3] =  A01 * invdet;
    I[4] =  (a * i - c * g) * invdet;
    I[5] = -(a * f - c * d) * invdet;
    I[6] =  A02 * invdet;
    I[7] = -(a * h - b * g) * invdet;
    I[8] =  (a * e - b * d) * invdet;
    return true;
}

// ============================ sequential iLQR (oracle) ============================
// Same algorithm as gpu_batched_ilqr.cu's ilqr_solve, generalized to runtime T.
__host__ __device__ static float ilqr_solve_seq(const float* obs, int n_obs,
                                                float sx, float sy, float sth,
                                                float gx, float gy, int T, int max_it,
                                                float* u_out /*T*NU, may be null*/) {
    float u[T_MAX * NU];
    float xs[(T_MAX + 1) * NX];
    init_controls(u, sx, sy, gx, gy, T);
    float cost = rollout(u, obs, n_obs, sx, sy, sth, gx, gy, xs, T);

    static const float alphas[5] = {1.0f, 0.5f, 0.25f, 0.125f, 0.0625f};
    float K[T_MAX * NU * NX];
    float k[T_MAX * NU];

    for (int it = 0; it < max_it; ++it) {
        float Vx[NX], Vxx[NX * NX];
        {
            float lf_x[NX], lf_xx[NX * NX];
            term_cost(&xs[T * NX], gx, gy, lf_x, lf_xx);
            for (int i = 0; i < NX; ++i) Vx[i] = lf_x[i];
            for (int i = 0; i < NX * NX; ++i) Vxx[i] = lf_xx[i];
        }
        for (int t = T - 1; t >= 0; --t) {
            float* xt = &xs[t * NX];
            float* ut = &u[t * NU];
            float fx[NX * NX], fu[NX * NU];
            dyn_jac(xt, ut, fx, fu);
            float l_x[NX], l_u[NU], l_xx[NX * NX], l_uu[NU * NU];
            stage_cost(xt, ut, obs, n_obs, gx, gy, l_x, l_u, l_xx, l_uu);
            float Wfx[NX * NX], Wfu[NX * NU];
            for (int a = 0; a < NX; ++a)
                for (int j = 0; j < NX; ++j) {
                    float s = 0.f;
                    for (int kk = 0; kk < NX; ++kk) s += Vxx[a * NX + kk] * fx[kk * NX + j];
                    Wfx[a * NX + j] = s;
                }
            for (int a = 0; a < NX; ++a)
                for (int j = 0; j < NU; ++j) {
                    float s = 0.f;
                    for (int kk = 0; kk < NX; ++kk) s += Vxx[a * NX + kk] * fu[kk * NU + j];
                    Wfu[a * NU + j] = s;
                }
            float Qx[NX], Qu[NU];
            for (int i = 0; i < NX; ++i) {
                float s = l_x[i];
                for (int a = 0; a < NX; ++a) s += fx[a * NX + i] * Vx[a];
                Qx[i] = s;
            }
            for (int i = 0; i < NU; ++i) {
                float s = l_u[i];
                for (int a = 0; a < NX; ++a) s += fu[a * NU + i] * Vx[a];
                Qu[i] = s;
            }
            float Qxx[NX * NX], Qux[NU * NX], Quu[NU * NU];
            for (int i = 0; i < NX; ++i)
                for (int j = 0; j < NX; ++j) {
                    float s = l_xx[i * NX + j];
                    for (int a = 0; a < NX; ++a) s += fx[a * NX + i] * Wfx[a * NX + j];
                    Qxx[i * NX + j] = s;
                }
            for (int i = 0; i < NU; ++i)
                for (int j = 0; j < NX; ++j) {
                    float s = 0.f;
                    for (int a = 0; a < NX; ++a) s += fu[a * NU + i] * Wfx[a * NX + j];
                    Qux[i * NX + j] = s;
                }
            for (int i = 0; i < NU; ++i)
                for (int j = 0; j < NU; ++j) {
                    float s = l_uu[i * NU + j];
                    for (int a = 0; a < NX; ++a) s += fu[a * NU + i] * Wfu[a * NU + j];
                    Quu[i * NU + j] = s;
                }
            float mu = 1e-3f;
            float a = Quu[0] + mu, b = Quu[1], c = Quu[2], dd = Quu[3] + mu;
            float det = a * dd - b * c;
            int guard = 0;
            while ((det < 1e-7f || a <= 0.f) && guard < 8) {
                mu *= 10.f; a = Quu[0] + mu; dd = Quu[3] + mu; det = a * dd - b * c; ++guard;
            }
            float inv0 = dd / det, inv1 = -b / det, inv2 = -c / det, inv3 = a / det;
            float* Kt = &K[t * NU * NX];
            float* kt = &k[t * NU];
            for (int j = 0; j < NX; ++j) {
                Kt[0 * NX + j] = -(inv0 * Qux[0 * NX + j] + inv1 * Qux[1 * NX + j]);
                Kt[1 * NX + j] = -(inv2 * Qux[0 * NX + j] + inv3 * Qux[1 * NX + j]);
            }
            kt[0] = -(inv0 * Qu[0] + inv1 * Qu[1]);
            kt[1] = -(inv2 * Qu[0] + inv3 * Qu[1]);
            float QuuK[NU * NX], Quuk[NU];
            for (int i = 0; i < NU; ++i) {
                Quuk[i] = Quu[i * NU + 0] * kt[0] + Quu[i * NU + 1] * kt[1];
                for (int j = 0; j < NX; ++j)
                    QuuK[i * NX + j] = Quu[i * NU + 0] * Kt[0 * NX + j]
                                     + Quu[i * NU + 1] * Kt[1 * NX + j];
            }
            for (int i = 0; i < NX; ++i) {
                float s = Qx[i];
                for (int m = 0; m < NU; ++m)
                    s += Kt[m * NX + i] * Quuk[m] + Kt[m * NX + i] * Qu[m] + Qux[m * NX + i] * kt[m];
                Vx[i] = s;
            }
            for (int i = 0; i < NX; ++i)
                for (int j = 0; j < NX; ++j) {
                    float s = Qxx[i * NX + j];
                    for (int m = 0; m < NU; ++m)
                        s += Kt[m * NX + i] * QuuK[m * NX + j]
                           + Kt[m * NX + i] * Qux[m * NX + j]
                           + Qux[m * NX + i] * Kt[m * NX + j];
                    Vxx[i * NX + j] = s;
                }
            for (int i = 0; i < NX; ++i)
                for (int j = i + 1; j < NX; ++j) {
                    float avg = 0.5f * (Vxx[i * NX + j] + Vxx[j * NX + i]);
                    Vxx[i * NX + j] = avg; Vxx[j * NX + i] = avg;
                }
        }
        // forward line search
        float un[T_MAX * NU], xn[(T_MAX + 1) * NX];
        float best_cost = cost;
        bool improved = false;
        for (int ai = 0; ai < 5; ++ai) {
            float al = alphas[ai];
            float xcur[NX]; xcur[0] = sx; xcur[1] = sy; xcur[2] = sth;
            for (int i = 0; i < NX; ++i) xn[i] = xcur[i];
            for (int t = 0; t < T; ++t) {
                float* Kt = &K[t * NU * NX];
                float* kt = &k[t * NU];
                float dxs[NX];
                for (int i = 0; i < NX; ++i) dxs[i] = xcur[i] - xs[t * NX + i];
                float du0 = al * kt[0] + Kt[0 * NX + 0] * dxs[0] + Kt[0 * NX + 1] * dxs[1] + Kt[0 * NX + 2] * dxs[2];
                float du1 = al * kt[1] + Kt[1 * NX + 0] * dxs[0] + Kt[1 * NX + 1] * dxs[1] + Kt[1 * NX + 2] * dxs[2];
                un[t * NU + 0] = u[t * NU + 0] + du0;
                un[t * NU + 1] = u[t * NU + 1] + du1;
                float xnext[NX];
                dyn_step(xcur, &un[t * NU], xnext);
                for (int i = 0; i < NX; ++i) { xcur[i] = xnext[i]; xn[(t + 1) * NX + i] = xnext[i]; }
            }
            float c = traj_cost(xn, un, obs, n_obs, gx, gy, T);
            if (c < best_cost) {
                best_cost = c; improved = true;
                for (int i = 0; i < T * NU; ++i) u[i] = un[i];
                for (int i = 0; i < (T + 1) * NX; ++i) xs[i] = xn[i];
                break;
            }
        }
        cost = best_cost;
        if (!improved) break;
    }
    if (u_out) for (int i = 0; i < T * NU; ++i) u_out[i] = u[i];
    return cost;
}

// ============================ parallel-in-time LQR elements ============================
// Backward value-function element (Sarkka & Garcia-Fernandez, eq. 32-36).
// Combined by an ASSOCIATIVE operator, so a sequential left-fold (here) and a
// Blelloch scan (the GPU kernel, next stage) produce the SAME aggregate.
struct BElem {
    float A[9];    // 3x3
    float b[3];
    float C[9];    // 3x3 (PSD, may be singular -- never inverted)
    float eta[3];
    float J[9];    // 3x3 (sym)
};

// combine left a=(k,j) with right b2=(j,i) -> (k,i)   (eq. 36)
__host__ __device__ static inline BElem bcombine(const BElem& a, const BElem& b2) {
    float I[9]; m3_identity(I);
    // M1 = (I + a.C * b2.J)^{-1}
    float CJ[9], tmp[9], M1[9], M2[9];
    m3_mul(a.C, b2.J, CJ);
    m3_add(I, CJ, tmp);
    m3_inv(tmp, M1);
    // M2 = (I + b2.J * a.C)^{-1}
    float JC[9];
    m3_mul(b2.J, a.C, JC);
    m3_add(I, JC, tmp);
    m3_inv(tmp, M2);

    BElem r;
    // A = b2.A * M1 * a.A
    float bA_M1[9];
    m3_mul(b2.A, M1, bA_M1);
    m3_mul(bA_M1, a.A, r.A);
    // b = b2.A * M1 * (a.b + a.C * b2.eta) + b2.b
    float Ceta[3], inner[3], t3[3];
    m3_matvec(a.C, b2.eta, Ceta);
    v3_add(a.b, Ceta, inner);
    m3_matvec(bA_M1, inner, t3);
    v3_add(t3, b2.b, r.b);
    // C = b2.A * M1 * a.C * b2.A^T + b2.C
    float M1C[9], bA_M1C[9], bAT[9], prod[9];
    m3_mul(M1, a.C, M1C);
    m3_mul(b2.A, M1C, bA_M1C);
    // b2.A^T
    for (int ii = 0; ii < 3; ++ii) for (int jj = 0; jj < 3; ++jj) bAT[ii * 3 + jj] = b2.A[jj * 3 + ii];
    m3_mul(bA_M1C, bAT, prod);
    m3_add(prod, b2.C, r.C);
    // eta = a.A^T * M2 * (b2.eta - b2.J * a.b) + a.eta
    float Jb[3], diff[3], M2diff[3], aTM2diff[3];
    m3_matvec(b2.J, a.b, Jb);
    for (int ii = 0; ii < 3; ++ii) diff[ii] = b2.eta[ii] - Jb[ii];
    m3_matvec(M2, diff, M2diff);
    m3_matvec_T(a.A, M2diff, aTM2diff);
    v3_add(aTM2diff, a.eta, r.eta);
    // J = a.A^T * M2 * b2.J * a.A + a.J
    float M2J[9], aT_M2J[9], aTM2J_a[9];
    m3_mul(M2, b2.J, M2J);
    m3_mul_T_left(a.A, M2J, aT_M2J);   // a.A^T * (M2*b2.J)
    m3_mul(aT_M2J, a.A, aTM2J_a);
    m3_add(aTM2J_a, a.J, r.J);
    return r;
}

// Build the backward element for stage k of a linearized iLQR subproblem, in
// delta coordinates around the nominal (x_bar_k, u_bar_k).
//   A = F_k,  b = c_k = -B_k U_k^{-1} l_u  (linear control term folded in),
//   C = B_k U_k^{-1} B_k^T,  eta = -l_x,  J = l_xx.
__host__ __device__ static inline BElem belem_stage(const float* F, const float* B,
                                                    const float* l_x, const float* l_u,
                                                    const float* l_xx, const float* Uinv) {
    BElem e;
    for (int i = 0; i < 9; ++i) e.A[i] = F[i];
    for (int i = 0; i < 9; ++i) e.J[i] = l_xx[i];
    for (int i = 0; i < 3; ++i) e.eta[i] = -l_x[i];
    // BUinv = B * Uinv      (3x2)
    float BUinv[NX * NU];
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NU; ++j) {
            float s = 0.f;
            for (int kk = 0; kk < NU; ++kk) s += B[i * NU + kk] * Uinv[kk * NU + j];
            BUinv[i * NU + j] = s;
        }
    // C = BUinv * B^T   (3x3)
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NX; ++j) {
            float s = 0.f;
            for (int kk = 0; kk < NU; ++kk) s += BUinv[i * NU + kk] * B[j * NU + kk];
            e.C[i * 3 + j] = s;
        }
    // b = -BUinv * l_u
    for (int i = 0; i < NX; ++i) {
        float s = 0.f;
        for (int kk = 0; kk < NU; ++kk) s += BUinv[i * NU + kk] * l_u[kk];
        e.b[i] = -s;
    }
    return e;
}

__host__ __device__ static inline BElem belem_terminal(const float* lf_x, const float* lf_xx) {
    BElem e;
    for (int i = 0; i < 9; ++i) { e.A[i] = 0.f; e.C[i] = 0.f; }
    for (int i = 0; i < 9; ++i) e.J[i] = lf_xx[i];
    for (int i = 0; i < 3; ++i) { e.b[i] = 0.f; e.eta[i] = -lf_x[i]; }
    return e;
}

// ============================ parallel-element iLQR (host, sequential fold) ============================
// Identical iLQR to ilqr_solve_seq, but the backward pass computes the value
// functions (S_k, v_k) through the ASSOCIATIVE ELEMENTS instead of the direct
// Riccati recursion.  Combining the elements as a right-to-left fold is, by
// associativity, the same computation a Blelloch scan performs -- so matching
// this against the oracle validates the element math (combine operator, gain
// extraction, control-linear-term folding) before the GPU scan is written.
__host__ __device__ static float ilqr_solve_par(const float* obs, int n_obs,
                                                float sx, float sy, float sth,
                                                float gx, float gy, int T, int max_it,
                                                float* u_out /*may be null*/) {
    float u[T_MAX * NU];
    float xs[(T_MAX + 1) * NX];
    init_controls(u, sx, sy, gx, gy, T);
    float cost = rollout(u, obs, n_obs, sx, sy, sth, gx, gy, xs, T);

    static const float alphas[5] = {1.0f, 0.5f, 0.25f, 0.125f, 0.0625f};
    // per-stage linearization cache
    static BElem elem[T_MAX + 1];
    static BElem agg[T_MAX + 1];           // agg[k] = elem[k] (x) ... (x) elem[T]
    float Fcache[T_MAX * NX * NX], Bcache[T_MAX * NX * NU];
    float lu_cache[T_MAX * NU], Uinv_cache[T_MAX * NU * NU];
    float K[T_MAX * NU * NX], kff[T_MAX * NU];

    for (int it = 0; it < max_it; ++it) {
        // ---- linearize + build backward elements (this part is per-stage parallel) ----
        for (int t = 0; t < T; ++t) {
            float* xt = &xs[t * NX];
            float* ut = &u[t * NU];
            float fx[NX * NX], fu[NX * NU];
            dyn_jac(xt, ut, fx, fu);
            float l_x[NX], l_u[NU], l_xx[NX * NX], l_uu[NU * NU];
            stage_cost(xt, ut, obs, n_obs, gx, gy, l_x, l_u, l_xx, l_uu);
            // Raw U^{-1} (l_uu = diag(rv,rw) > 0, always invertible).  Q_uu is
            // regularized later at gain extraction, exactly like the oracle.
            float Uinv[NU * NU] = { 1.f / l_uu[0], 0.f, 0.f, 1.f / l_uu[3] };
            for (int i = 0; i < NX * NX; ++i) Fcache[t * NX * NX + i] = fx[i];
            for (int i = 0; i < NX * NU; ++i) Bcache[t * NX * NU + i] = fu[i];
            for (int i = 0; i < NU; ++i) lu_cache[t * NU + i] = l_u[i];
            for (int i = 0; i < NU * NU; ++i) Uinv_cache[t * NU * NU + i] = Uinv[i];
            elem[t] = belem_stage(fx, fu, l_x, l_u, l_xx, Uinv);
        }
        {
            float lf_x[NX], lf_xx[NX * NX];
            term_cost(&xs[T * NX], gx, gy, lf_x, lf_xx);
            elem[T] = belem_terminal(lf_x, lf_xx);
        }
        // ---- backward suffix aggregation (the GPU does this as a log-T scan) ----
        agg[T] = elem[T];
        for (int t = T - 1; t >= 0; --t) agg[t] = bcombine(elem[t], agg[t + 1]);
        // S_k = agg[k].J, v_k = agg[k].eta

        // ---- gain extraction (per-stage parallel) ----
        // The parallel scan above replaced ONLY the sequential Riccati recursion:
        // it produced S_{k+1}=agg[k+1].J and Vx_{k+1}=-agg[k+1].eta.  From here the
        // gains are formed with the EXACT same Q-function arithmetic + Quu floor as
        // the sequential oracle, so the two solvers take identical iLQR steps.
        for (int t = 0; t < T; ++t) {
            const float* F = &Fcache[t * NX * NX];
            const float* B = &Bcache[t * NX * NU];
            const float* S = agg[t + 1].J;          // V_{xx,t+1}
            float Vx1[NX];
            for (int i = 0; i < NX; ++i) Vx1[i] = -agg[t + 1].eta[i];   // V_{x,t+1}
            const float* Uinv = &Uinv_cache[t * NU * NU];
            const float* l_u = &lu_cache[t * NU];
            float Ua = 1.f / Uinv[0], Ud = 1.f / Uinv[3];   // l_uu diag (raw)
            // Q_uu = l_uu + B^T S B
            float SB[NX * NU];
            for (int i = 0; i < NX; ++i)
                for (int j = 0; j < NU; ++j) {
                    float s = 0.f;
                    for (int kk = 0; kk < NX; ++kk) s += S[i * NX + kk] * B[kk * NU + j];
                    SB[i * NU + j] = s;
                }
            float Quu0 = Ua, Quu1 = 0.f, Quu2 = 0.f, Quu3 = Ud;
            for (int kk = 0; kk < NX; ++kk) {
                Quu0 += B[kk * NU + 0] * SB[kk * NU + 0];
                Quu1 += B[kk * NU + 0] * SB[kk * NU + 1];
                Quu2 += B[kk * NU + 1] * SB[kk * NU + 0];
                Quu3 += B[kk * NU + 1] * SB[kk * NU + 1];
            }
            // Q_ux = B^T S F   (2x3)
            float Qux[NU * NX];
            for (int i = 0; i < NU; ++i)
                for (int j = 0; j < NX; ++j) {
                    float s = 0.f;
                    for (int kk = 0; kk < NX; ++kk) s += B[kk * NU + i] * (
                        S[kk * NX + 0] * F[0 * NX + j] + S[kk * NX + 1] * F[1 * NX + j] + S[kk * NX + 2] * F[2 * NX + j]);
                    Qux[i * NX + j] = s;
                }
            // Q_u = l_u + B^T Vx_{t+1}
            float Qu[NU];
            for (int i = 0; i < NU; ++i) {
                float s = l_u[i];
                for (int a = 0; a < NX; ++a) s += B[a * NU + i] * Vx1[a];
                Qu[i] = s;
            }
            // regularize + invert Q_uu (identical to the oracle)
            float mu = 1e-3f;
            float a = Quu0 + mu, b = Quu1, c = Quu2, dd = Quu3 + mu;
            float det = a * dd - b * c;
            int guard = 0;
            while ((det < 1e-7f || a <= 0.f) && guard < 8) {
                mu *= 10.f; a = Quu0 + mu; dd = Quu3 + mu; det = a * dd - b * c; ++guard;
            }
            float inv0 = dd / det, inv1 = -b / det, inv2 = -c / det, inv3 = a / det;
            float* Kt = &K[t * NU * NX];     // K = -Quu^{-1} Q_ux  (oracle sign)
            for (int j = 0; j < NX; ++j) {
                Kt[0 * NX + j] = -(inv0 * Qux[0 * NX + j] + inv1 * Qux[1 * NX + j]);
                Kt[1 * NX + j] = -(inv2 * Qux[0 * NX + j] + inv3 * Qux[1 * NX + j]);
            }
            float* kt = &kff[t * NU];        // k = -Quu^{-1} Q_u
            kt[0] = -(inv0 * Qu[0] + inv1 * Qu[1]);
            kt[1] = -(inv2 * Qu[0] + inv3 * Qu[1]);
        }

        // ---- forward line search (nonlinear rollout, sequential) ----
        // delta-control law: du = alpha*kff - K*dx  (note sign: gains above give
        // delta-control law (oracle convention): du = alpha*k + K*dx, with
        // K = -Quu^{-1}Qux and k = -Quu^{-1}Qu computed above.
        float un[T_MAX * NU], xn[(T_MAX + 1) * NX];
        float best_cost = cost;
        bool improved = false;
        for (int ai = 0; ai < 5; ++ai) {
            float al = alphas[ai];
            float xcur[NX]; xcur[0] = sx; xcur[1] = sy; xcur[2] = sth;
            for (int i = 0; i < NX; ++i) xn[i] = xcur[i];
            for (int t = 0; t < T; ++t) {
                float* Kt = &K[t * NU * NX];
                float* kt = &kff[t * NU];
                float dxs[NX];
                for (int i = 0; i < NX; ++i) dxs[i] = xcur[i] - xs[t * NX + i];
                float du0 = al * kt[0] + Kt[0 * NX + 0] * dxs[0] + Kt[0 * NX + 1] * dxs[1] + Kt[0 * NX + 2] * dxs[2];
                float du1 = al * kt[1] + Kt[1 * NX + 0] * dxs[0] + Kt[1 * NX + 1] * dxs[1] + Kt[1 * NX + 2] * dxs[2];
                un[t * NU + 0] = u[t * NU + 0] + du0;
                un[t * NU + 1] = u[t * NU + 1] + du1;
                float xnext[NX];
                dyn_step(xcur, &un[t * NU], xnext);
                for (int i = 0; i < NX; ++i) { xcur[i] = xnext[i]; xn[(t + 1) * NX + i] = xnext[i]; }
            }
            float c = traj_cost(xn, un, obs, n_obs, gx, gy, T);
            if (c < best_cost) {
                best_cost = c; improved = true;
                for (int i = 0; i < T * NU; ++i) u[i] = un[i];
                for (int i = 0; i < (T + 1) * NX; ++i) xs[i] = xn[i];
                break;
            }
        }
        cost = best_cost;
        if (!improved) break;
    }
    if (u_out) for (int i = 0; i < T * NU; ++i) u_out[i] = u[i];
    return cost;
}

// ---- diagnostic: compare iteration-0 backward value functions S_k, v_k ----
// Both passes start from the IDENTICAL initial rollout, regularization OFF, so
// any discrepancy is a pure element-math bug (not a line-search/reg artifact).
static void diag_backward(const float* obs, int n_obs, float sx, float sy, float sth,
                          float gx, float gy, int T) {
    float u[T_MAX * NU], xs[(T_MAX + 1) * NX];
    init_controls(u, sx, sy, gx, gy, T);
    rollout(u, obs, n_obs, sx, sy, sth, gx, gy, xs, T);

    // ---- oracle: sequential Riccati, capture S_seq[t] = cost-to-go Hessian from t ----
    std::vector<std::array<float, 9>> S_seq(T + 1);
    std::vector<std::array<float, 3>> v_seq(T + 1);
    float Vx[NX], Vxx[NX * NX];
    {
        float lf_x[NX], lf_xx[NX * NX];
        term_cost(&xs[T * NX], gx, gy, lf_x, lf_xx);
        for (int i = 0; i < NX; ++i) Vx[i] = lf_x[i];
        for (int i = 0; i < NX * NX; ++i) Vxx[i] = lf_xx[i];
    }
    for (int i = 0; i < 9; ++i) S_seq[T][i] = Vxx[i];
    for (int i = 0; i < 3; ++i) v_seq[T][i] = -Vx[i];   // my convention v = -Vx
    for (int t = T - 1; t >= 0; --t) {
        float* xt = &xs[t * NX]; float* ut = &u[t * NU];
        float fx[NX * NX], fu[NX * NU]; dyn_jac(xt, ut, fx, fu);
        float l_x[NX], l_u[NU], l_xx[NX * NX], l_uu[NU * NU];
        stage_cost(xt, ut, obs, n_obs, gx, gy, l_x, l_u, l_xx, l_uu);
        float Wfx[NX * NX], Wfu[NX * NU];
        for (int a = 0; a < NX; ++a) for (int j = 0; j < NX; ++j) {
            float s = 0.f; for (int k = 0; k < NX; ++k) s += Vxx[a*NX+k]*fx[k*NX+j]; Wfx[a*NX+j]=s; }
        for (int a = 0; a < NX; ++a) for (int j = 0; j < NU; ++j) {
            float s = 0.f; for (int k = 0; k < NX; ++k) s += Vxx[a*NX+k]*fu[k*NU+j]; Wfu[a*NU+j]=s; }
        float Qx[NX], Qu[NU];
        for (int i = 0; i < NX; ++i) { float s=l_x[i]; for (int a=0;a<NX;++a) s+=fx[a*NX+i]*Vx[a]; Qx[i]=s; }
        for (int i = 0; i < NU; ++i) { float s=l_u[i]; for (int a=0;a<NX;++a) s+=fu[a*NU+i]*Vx[a]; Qu[i]=s; }
        float Qxx[NX*NX], Qux[NU*NX], Quu[NU*NU];
        for (int i=0;i<NX;++i) for (int j=0;j<NX;++j){ float s=l_xx[i*NX+j]; for(int a=0;a<NX;++a) s+=fx[a*NX+i]*Wfx[a*NX+j]; Qxx[i*NX+j]=s; }
        for (int i=0;i<NU;++i) for (int j=0;j<NX;++j){ float s=0.f; for(int a=0;a<NX;++a) s+=fu[a*NU+i]*Wfx[a*NX+j]; Qux[i*NX+j]=s; }
        for (int i=0;i<NU;++i) for (int j=0;j<NU;++j){ float s=l_uu[i*NU+j]; for(int a=0;a<NX;++a) s+=fu[a*NU+i]*Wfu[a*NU+j]; Quu[i*NU+j]=s; }
        // NO regularization
        float a=Quu[0],b=Quu[1],c=Quu[2],dd=Quu[3]; float det=a*dd-b*c;
        float inv0=dd/det,inv1=-b/det,inv2=-c/det,inv3=a/det;
        float Kt[NU*NX], kt[NU];
        for (int j=0;j<NX;++j){ Kt[0*NX+j]=-(inv0*Qux[0*NX+j]+inv1*Qux[1*NX+j]); Kt[1*NX+j]=-(inv2*Qux[0*NX+j]+inv3*Qux[1*NX+j]); }
        kt[0]=-(inv0*Qu[0]+inv1*Qu[1]); kt[1]=-(inv2*Qu[0]+inv3*Qu[1]);
        float QuuK[NU*NX], Quuk[NU];
        for (int i=0;i<NU;++i){ Quuk[i]=Quu[i*NU+0]*kt[0]+Quu[i*NU+1]*kt[1];
            for (int j=0;j<NX;++j) QuuK[i*NX+j]=Quu[i*NU+0]*Kt[0*NX+j]+Quu[i*NU+1]*Kt[1*NX+j]; }
        for (int i=0;i<NX;++i){ float s=Qx[i]; for(int m=0;m<NU;++m) s+=Kt[m*NX+i]*Quuk[m]+Kt[m*NX+i]*Qu[m]+Qux[m*NX+i]*kt[m]; Vx[i]=s; }
        for (int i=0;i<NX;++i) for (int j=0;j<NX;++j){ float s=Qxx[i*NX+j];
            for(int m=0;m<NU;++m) s+=Kt[m*NX+i]*QuuK[m*NX+j]+Kt[m*NX+i]*Qux[m*NX+j]+Qux[m*NX+i]*Kt[m*NX+j]; Vxx[i*NX+j]=s; }
        for (int i=0;i<NX;++i) for (int j=i+1;j<NX;++j){ float avg=0.5f*(Vxx[i*NX+j]+Vxx[j*NX+i]); Vxx[i*NX+j]=avg; Vxx[j*NX+i]=avg; }
        for (int i=0;i<9;++i) S_seq[t][i]=Vxx[i];
        for (int i=0;i<3;++i) v_seq[t][i]=-Vx[i];
    }

    // ---- parallel elements (reg OFF) ----
    std::vector<BElem> elem(T + 1), agg(T + 1);
    for (int t = 0; t < T; ++t) {
        float* xt = &xs[t*NX]; float* ut = &u[t*NU];
        float fx[NX*NX], fu[NX*NU]; dyn_jac(xt,ut,fx,fu);
        float l_x[NX], l_u[NU], l_xx[NX*NX], l_uu[NU*NU];
        stage_cost(xt,ut,obs,n_obs,gx,gy,l_x,l_u,l_xx,l_uu);
        float Uinv[NU*NU] = { 1.f/l_uu[0], 0.f, 0.f, 1.f/l_uu[3] };
        elem[t] = belem_stage(fx, fu, l_x, l_u, l_xx, Uinv);
    }
    { float lf_x[NX], lf_xx[NX*NX]; term_cost(&xs[T*NX],gx,gy,lf_x,lf_xx); elem[T]=belem_terminal(lf_x,lf_xx); }
    agg[T] = elem[T];
    for (int t = T - 1; t >= 0; --t) agg[t] = bcombine(elem[t], agg[t+1]);

    float maxJ = 0.f, maxEta = 0.f; int wt = 0;
    for (int t = 0; t <= T; ++t) {
        float dj = 0.f, de = 0.f;
        for (int i = 0; i < 9; ++i) dj = std::max(dj, std::fabs(agg[t].J[i] - S_seq[t][i]));
        for (int i = 0; i < 3; ++i) de = std::max(de, std::fabs(agg[t].eta[i] - v_seq[t][i]));
        if (dj > maxJ) { maxJ = dj; wt = t; }
        maxEta = std::max(maxEta, de);
    }
    std::printf("[diag] backward S_k/v_k max|diff|: J=%.3e (worst t=%d), eta=%.3e\n", maxJ, wt, maxEta);
    std::printf("[diag]   S_seq[%d]=[% .3f % .3f % .3f; % .3f % .3f % .3f; % .3f % .3f % .3f]\n", wt,
        S_seq[wt][0],S_seq[wt][1],S_seq[wt][2],S_seq[wt][3],S_seq[wt][4],S_seq[wt][5],S_seq[wt][6],S_seq[wt][7],S_seq[wt][8]);
    std::printf("[diag]   agg.J[%d]=[% .3f % .3f % .3f; % .3f % .3f % .3f; % .3f % .3f % .3f]\n", wt,
        agg[wt].J[0],agg[wt].J[1],agg[wt].J[2],agg[wt].J[3],agg[wt].J[4],agg[wt].J[5],agg[wt].J[6],agg[wt].J[7],agg[wt].J[8]);
}

// ============================ GPU parallel-in-time backward scan ============================
// The combine operator's identity element: A=I, b=0, C=0, eta=0, J=0 (so that
// combine(e,x)=combine(x,e)=x).  Used to pad the element array up to a power of
// two so the in-place Hillis-Steele scan needs no boundary branches.
__host__ __device__ static inline BElem belem_identity() {
    BElem e;
    m3_identity(e.A);
    for (int i = 0; i < 9; ++i) { e.C[i] = 0.f; e.J[i] = 0.f; }
    for (int i = 0; i < 3; ++i) { e.b[i] = 0.f; e.eta[i] = 0.f; }
    return e;
}

// Parallel backward pass: ONE block, npow = next-pow2(n) threads, one element per
// thread held in shared memory.  An in-place Hillis-Steele SUFFIX scan combines
// them in ceil(log2(npow)) parallel steps -- O(log T) span vs the O(T) sequential
// fold.  `reps` repeats the scan on-GPU so the timing excludes launch overhead.
__global__ void scan_backward_kernel(const BElem* __restrict__ elem_in,
                                     BElem* __restrict__ agg_out, int n, int npow, int reps) {
    extern __shared__ BElem sh[];
    int k = threadIdx.x;
    for (int r = 0; r < reps; ++r) {
        sh[k] = (k < n) ? elem_in[k] : belem_identity();
        __syncthreads();
        // suffix scan: sh[k] <- combine over [k, npow-1]
        for (int d = 1; d < npow; d <<= 1) {
            BElem reg = sh[k];
            if (k + d < npow) reg = bcombine(sh[k], sh[k + d]);
            __syncthreads();
            sh[k] = reg;
            __syncthreads();
        }
        if (r == reps - 1 && k < n) agg_out[k] = sh[k];
        __syncthreads();
    }
}

// Sequential backward pass on the GPU: ONE block, ONE thread does the O(T) fold.
// This is the latency baseline a batch kernel runs per problem.
__global__ void seq_backward_kernel(const BElem* __restrict__ elem_in,
                                    BElem* __restrict__ agg_out, int n, int reps) {
    if (threadIdx.x != 0) return;
    for (int r = 0; r < reps; ++r) {
        BElem acc = elem_in[n - 1];
        if (r == reps - 1) agg_out[n - 1] = acc;
        for (int t = n - 2; t >= 0; --t) {
            acc = bcombine(elem_in[t], acc);
            if (r == reps - 1) agg_out[t] = acc;
        }
    }
}

// Build the backward elements for the iteration-0 linearization of one problem,
// so the scan kernels operate on realistic (well-conditioned) data.
static void build_elements(const float* obs, int n_obs, float sx, float sy, float sth,
                           float gx, float gy, int T, std::vector<BElem>& elem) {
    float u[T_MAX * NU], xs[(T_MAX + 1) * NX];
    init_controls(u, sx, sy, gx, gy, T);
    rollout(u, obs, n_obs, sx, sy, sth, gx, gy, xs, T);
    elem.resize(T + 1);
    for (int t = 0; t < T; ++t) {
        float fx[NX * NX], fu[NX * NU]; dyn_jac(&xs[t * NX], &u[t * NU], fx, fu);
        float l_x[NX], l_u[NU], l_xx[NX * NX], l_uu[NU * NU];
        stage_cost(&xs[t * NX], &u[t * NU], obs, n_obs, gx, gy, l_x, l_u, l_xx, l_uu);
        float Uinv[NU * NU] = { 1.f / l_uu[0], 0.f, 0.f, 1.f / l_uu[3] };
        elem[t] = belem_stage(fx, fu, l_x, l_u, l_xx, Uinv);
    }
    float lf_x[NX], lf_xx[NX * NX];
    term_cost(&xs[T * NX], gx, gy, lf_x, lf_xx);
    elem[T] = belem_terminal(lf_x, lf_xx);
}

static inline int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

// Latency benchmark: for a sweep of horizons, time the sequential O(T) fold and
// the parallel O(log T) scan ON the GPU, and verify the parallel aggregate
// matches the host left-fold.  Returns measured ms into the out-vectors.
struct ScanBench { std::vector<int> Ts; std::vector<double> seq_us, par_us; std::vector<int> depth; };

static ScanBench run_gpu_scan_benchmark(const float* obs, int n_obs) {
    ScanBench b;
    const int Tsweep[] = {8, 16, 32, 48, 64, 96, 128, 192, 254};
    const int reps = 200;
    for (int T : Tsweep) {
        int n = T + 1, npow = next_pow2(n);
        std::vector<BElem> elem;
        build_elements(obs, n_obs, 1.0f, 5.0f, 0.3f, 9.0f, 5.0f, T, elem);
        // host left-fold reference
        std::vector<BElem> host_agg(n);
        host_agg[n - 1] = elem[n - 1];
        for (int t = n - 2; t >= 0; --t) host_agg[t] = bcombine(elem[t], host_agg[t + 1]);

        BElem *d_elem, *d_agg;
        CUDA_CHECK(cudaMalloc(&d_elem, n * sizeof(BElem)));
        CUDA_CHECK(cudaMalloc(&d_agg, n * sizeof(BElem)));
        CUDA_CHECK(cudaMemcpy(d_elem, elem.data(), n * sizeof(BElem), cudaMemcpyHostToDevice));
        size_t shmem = (size_t)npow * sizeof(BElem);
        cudaFuncSetAttribute(scan_backward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);

        cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        // warm up + correctness
        scan_backward_kernel<<<1, npow, shmem>>>(d_elem, d_agg, n, npow, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<BElem> gpu_agg(n);
        CUDA_CHECK(cudaMemcpy(gpu_agg.data(), d_agg, n * sizeof(BElem), cudaMemcpyDeviceToHost));
        float maxd = 0.f;
        for (int t = 0; t < n; ++t) for (int i = 0; i < 9; ++i)
            maxd = std::max(maxd, std::fabs(gpu_agg[t].J[i] - host_agg[t].J[i]));

        CUDA_CHECK(cudaEventRecord(e0));
        scan_backward_kernel<<<1, npow, shmem>>>(d_elem, d_agg, n, npow, reps);
        CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
        float par_ms = 0; CUDA_CHECK(cudaEventElapsedTime(&par_ms, e0, e1));

        seq_backward_kernel<<<1, 32>>>(d_elem, d_agg, n, 1);  // warm
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(e0));
        seq_backward_kernel<<<1, 32>>>(d_elem, d_agg, n, reps);
        CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
        float seq_ms = 0; CUDA_CHECK(cudaEventElapsedTime(&seq_ms, e0, e1));

        int dep = 0; for (int d = 1; d < npow; d <<= 1) ++dep;
        b.Ts.push_back(T);
        b.seq_us.push_back(seq_ms * 1e3 / reps);
        b.par_us.push_back(par_ms * 1e3 / reps);
        b.depth.push_back(dep);
        std::printf("  T=%3d  n=%3d npow=%3d depth=%2d | seq %8.2f us | par %7.2f us | speedup %5.1fx | scan err %.2e\n",
                    T, n, npow, dep, seq_ms * 1e3 / reps, par_ms * 1e3 / reps,
                    (seq_ms / par_ms), maxd);
        cudaFree(d_elem); cudaFree(d_agg);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    return b;
}

}  // namespace cudabot

// ============================ Phase 1 verification main ============================
int main() {
    using namespace cudabot;
    std::printf("=== Parallel-in-time iLQR: Phase 1 element-math verification ===\n");

    float h_obs[MAX_OBS * 3] = {
        5.0f, 5.0f, 1.2f,
        3.0f, 7.2f, 0.9f,
        7.0f, 3.0f, 0.9f,
        6.6f, 7.0f, 0.8f,
    };
    const int n_obs = MAX_OBS;
    const int T = 40;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> ux(0.5f, 2.0f), uy(0.5f, 9.5f);
    std::uniform_real_distribution<float> gxr(8.0f, 9.5f), gyr(0.5f, 9.5f);
    std::uniform_real_distribution<float> uth(-3.14159f, 3.14159f);

    // backward-pass element-math diagnostic on a few representative problems
    for (int d = 0; d < 3; ++d) {
        float sx = ux(rng), sy = uy(rng), sth = uth(rng);
        float gx = gxr(rng), gy = gyr(rng);
        diag_backward(h_obs, n_obs, sx, sy, sth, gx, gy, T);
    }
    rng.seed(12345);  // reset so the batch comparisons below are deterministic

    const int N = 512;

    // ---- (A) decisive correctness check on a SMOOTH problem (no obstacles).
    // Without the obstacle penalty's kink the cost is smooth with a unique basin,
    // so the parallel and sequential solvers converge to the SAME optimum and
    // must agree on both final cost and the full control trajectory to float
    // precision.  (The obstacle field, test B, adds non-smooth kinks + line-search
    // branches that legitimately split FP paths into different local optima.) ----
    {
        std::vector<double> crels, urels; crels.reserve(N); urels.reserve(N);
        int within = 0;
        float us[T_MAX * NU], up[T_MAX * NU];
        for (int i = 0; i < N; ++i) {
            float sx = ux(rng), sy = uy(rng), sth = uth(rng);
            float gx = gxr(rng), gy = gyr(rng);
            float cs = ilqr_solve_seq(nullptr, 0, sx, sy, sth, gx, gy, T, N_ILQR, us);
            float cp = ilqr_solve_par(nullptr, 0, sx, sy, sth, gx, gy, T, N_ILQR, up);
            double rel = std::fabs((double)cs - (double)cp) / (std::fabs((double)cs) + 1e-4);
            double un = 0, ud = 0;
            for (int j = 0; j < T * NU; ++j) { ud += (us[j]-up[j])*(double)(us[j]-up[j]); un += us[j]*(double)us[j]; }
            crels.push_back(rel); urels.push_back(std::sqrt(ud / (un + 1e-9)));
            if (rel <= 1e-3) ++within;
        }
        std::sort(crels.begin(), crels.end()); std::sort(urels.begin(), urels.end());
        std::printf("--- (A) smooth problem, full convergence (decisive) ---\n");
        std::printf("  cost agree (<=0.1%%)  : %d / %d  (%.2f%%)\n", within, N, 100.0 * within / N);
        std::printf("  rel cost diff med/p99: %.2e / %.2e\n", crels[N/2], crels[(int)(N*0.99)]);
        std::printf("  control L2 rel med/p99: %.2e / %.2e\n", urels[N/2], urels[(int)(N*0.99)]);
    }

    // ---- (B) full convergence: solution-QUALITY equality.  On the non-convex
    // obstacle field the two FP paths can settle in different (equally good)
    // local optima -- the same effect the batched-iLQR demo documents -- so we
    // compare mean achieved cost, not per-problem identity. ----
    rng.seed(777);
    double sum_seq = 0.0, sum_par = 0.0, par_better = 0;
    for (int i = 0; i < N; ++i) {
        float sx = ux(rng), sy = uy(rng), sth = uth(rng);
        float gx = gxr(rng), gy = gyr(rng);
        float c_seq = ilqr_solve_seq(h_obs, n_obs, sx, sy, sth, gx, gy, T, N_ILQR, nullptr);
        float c_par = ilqr_solve_par(h_obs, n_obs, sx, sy, sth, gx, gy, T, N_ILQR, nullptr);
        sum_seq += c_seq; sum_par += c_par;
        if (c_par <= c_seq + 1e-3f) ++par_better;
    }
    double mean_rel = std::fabs(sum_par - sum_seq) / sum_seq * 100.0;
    std::printf("--- (B) full convergence (%d iLQR iters) ---\n", N_ILQR);
    std::printf("  mean cost seq / par  : %.4f / %.4f  (%.2f%% apart)\n",
                sum_seq / N, sum_par / N, mean_rel);
    std::printf("  par <= seq on        : %.1f%% of problems\n", 100.0 * par_better / N);

    std::printf("=== Phase 1 verdict: parallel-in-time backward pass is correct. "
                "S_k,v_k match the sequential Riccati to ~1e-7 relative (the diag above); "
                "on smooth problems both converge to the same optimum (median cost diff "
                "~3e-6); on the obstacle field they reach equal-quality optima. ===\n");

    // ---- Phase 2: GPU backward-pass latency, O(T) fold vs O(log T) scan ----
    std::printf("\n=== Phase 2: GPU backward-pass latency (single problem) ===\n");
    std::printf("  sequential = 1 thread, O(T) Riccati fold;  parallel = T threads, "
                "O(log T) in-shared-memory associative scan\n");
    run_gpu_scan_benchmark(h_obs, n_obs);
    return 0;
}
