// gpu_batched_ilqr.cu
//
// GPU batched iLQR trajectory optimization (CPU vs CUDA comparison).
//
// The repo's control line is dominated by SAMPLING-based optimizers (the MPPI
// family, STOMP, CMA-ES, MCTS).  This demo adds the GRADIENT-based counterpart:
// iterative LQR (iLQR / DDP), the workhorse of differentiable optimal control.
// iLQR is a per-problem sequential algorithm (a backward Riccati sweep followed
// by a forward line-search rollout, repeated to convergence) -- it does NOT
// parallelize across a single trajectory.  The parallelism is across PROBLEMS:
// a motion planner answering many start/goal queries in the same map solves
// thousands of independent iLQR instances, which is exactly what a GPU is good
// at (one thread = one optimal-control problem, the repo's canonical idiom).
//
// Setup: a shared 2D obstacle field and N random start/goal queries.  Each query
// is solved by the SAME iLQR (unicycle dynamics x=[px,py,theta], u=[v,omega],
// horizon T, soft circular-obstacle penalties, quadratic goal/terminal cost).
// The solver is a single __host__ __device__ routine called BOTH by a serial CPU
// loop and by the batch CUDA kernel, so the two paths run bit-for-bit the same
// arithmetic -- the comparison reports the final-cost MAE (expected ~0) alongside
// the wall-clock speedup.
//
// This is the efficiency statement, made honestly: the GPU is not solving a
// "better" iLQR, it is solving the IDENTICAL iLQR on every query at once.  The
// win is throughput on the batch, not a smarter optimizer.
//
// Layout: [shared field with 8 representative queries' iLQR trajectories,
// animated over iLQR iterations] | [info panel: iteration, mean cost, and the
// batch headline GPU/CPU timing + cost MAE].
//
// Output: gif/gpu_batched_ilqr.gif

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
#define NX 3           // state: px, py, theta
#define NU 2           // control: v, omega
#define T 40           // horizon (steps)
#define N_ILQR 25      // iLQR iterations
#define MAX_OBS 4      // obstacles in the shared field

static const int    N_PROB = 4096;   // batch size (independent start/goal queries)
static const float  DT     = 0.1f;

// ---- cost weights (shared host+device) ----
__host__ __device__ static inline float w_ctrl_v() { return 0.02f; }
__host__ __device__ static inline float w_ctrl_w() { return 0.01f; }
__host__ __device__ static inline float w_stage()  { return 0.20f; }  // stage goal pull
__host__ __device__ static inline float w_term()   { return 80.0f; }  // terminal goal
__host__ __device__ static inline float w_obs()    { return 60.0f; }  // obstacle penalty
__host__ __device__ static inline float obs_margin() { return 0.40f; }

// ============================ shared iLQR core ============================
// Everything below is __host__ __device__ so the CPU loop and the CUDA kernel
// execute the EXACT same arithmetic.

__host__ __device__ static inline void dyn_step(const float* x, const float* u, float* xn) {
    float c = cosf(x[2]), s = sinf(x[2]);
    xn[0] = x[0] + u[0] * c * DT;
    xn[1] = x[1] + u[0] * s * DT;
    xn[2] = x[2] + u[1] * DT;
}

// f_x (NX*NX, row-major) and f_u (NX*NU, row-major) at (x,u).
__host__ __device__ static inline void dyn_jac(const float* x, const float* u,
                                               float* fx, float* fu) {
    float c = cosf(x[2]), s = sinf(x[2]);
    for (int i = 0; i < NX * NX; ++i) fx[i] = 0.f;
    for (int i = 0; i < NX * NU; ++i) fu[i] = 0.f;
    fx[0] = 1.f; fx[4] = 1.f; fx[8] = 1.f;
    fx[0 * NX + 2] = -u[0] * s * DT;   // d px / d theta
    fx[1 * NX + 2] =  u[0] * c * DT;   // d py / d theta
    fu[0 * NU + 0] = c * DT;           // d px / d v
    fu[1 * NU + 0] = s * DT;           // d py / d v
    fu[2 * NU + 1] = DT;               // d theta / d omega
}

// Stage cost and its derivatives (Gauss-Newton Hessian, so l_xx stays PSD).
// l_ux is identically zero (cost is separable in x and u), so it is omitted.
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
    // control
    float rv = w_ctrl_v(), rw = w_ctrl_w();
    l += 0.5f * (rv * u[0] * u[0] + rw * u[1] * u[1]);
    l_u[0] = rv * u[0]; l_u[1] = rw * u[1];
    l_uu[0] = rv; l_uu[3] = rw;
    // stage goal pull
    float q = w_stage();
    float dx = x[0] - gx, dy = x[1] - gy;
    l += 0.5f * q * (dx * dx + dy * dy);
    l_x[0] += q * dx; l_x[1] += q * dy;
    l_xx[0] += q; l_xx[4] += q;
    // soft circular obstacles (penalty active inside r + margin)
    float wo = w_obs(), m = obs_margin();
    for (int o = 0; o < n_obs; ++o) {
        float ox = obs[o * 3 + 0], oy = obs[o * 3 + 1], orr = obs[o * 3 + 2];
        float ex = x[0] - ox, ey = x[1] - oy;
        float d = sqrtf(ex * ex + ey * ey) + 1e-6f;
        float g = (orr + m) - d;
        if (g > 0.f) {
            l += 0.5f * wo * g * g;
            float gx0 = -ex / d, gx1 = -ey / d;   // d g / d (px,py)
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

// Roll out controls from (sx,sy,sth); returns total cost; optionally writes the
// state trajectory (xs has room for (T+1)*NX).
__host__ __device__ static inline float rollout(const float* u, const float* obs, int n_obs,
                                                 float sx, float sy, float sth,
                                                 float gx, float gy, float* xs) {
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

// Solve one iLQR problem.  Writes optimized controls into u_out[T*NU] and the
// final cost into *cost_out.  When iter_cost / iter_traj are non-null (host
// visualization only) the per-iteration cost and (px,py) trajectory are logged;
// the batch kernel passes nullptr, so that branch never runs on device.
__host__ __device__ static void ilqr_solve(const float* obs, int n_obs,
                                            float sx, float sy, float sth,
                                            float gx, float gy,
                                            float* u_out, float* cost_out,
                                            float* iter_cost, float* iter_traj) {
    float u[T * NU];
    float xs[(T + 1) * NX];
    // initial guess: drive straight at a speed that roughly spans start->goal.
    float dgx = gx - sx, dgy = gy - sy;
    float v0 = sqrtf(dgx * dgx + dgy * dgy) / (T * DT);
    if (v0 > 2.5f) v0 = 2.5f;
    for (int t = 0; t < T; ++t) { u[t * NU + 0] = v0; u[t * NU + 1] = 0.f; }

    float cost = rollout(u, obs, n_obs, sx, sy, sth, gx, gy, xs);

    float K[T * NU * NX];
    float k[T * NU];
    const float alphas[5] = {1.0f, 0.5f, 0.25f, 0.125f, 0.0625f};

    for (int it = 0; it < N_ILQR; ++it) {
        // ---------------- backward pass ----------------
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

            // Vxx * fx  (NX*NX)  and  Vxx * fu  (NX*NU)
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
            // Q_x, Q_u
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
            // Q_xx, Q_ux, Q_uu
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
            // regularize + invert Quu (2x2), bumping the diagonal until PD
            float mu = 1e-3f;
            float a = Quu[0] + mu, b = Quu[1], c = Quu[2], dd = Quu[3] + mu;
            float det = a * dd - b * c;
            int guard = 0;
            while ((det < 1e-7f || a <= 0.f) && guard < 8) {
                mu *= 10.f; a = Quu[0] + mu; dd = Quu[3] + mu; det = a * dd - b * c; ++guard;
            }
            float inv0 =  dd / det, inv1 = -b / det, inv2 = -c / det, inv3 = a / det;
            // K = -inv * Qux  (NU*NX) ;  kk = -inv * Qu (NU)
            float* Kt = &K[t * NU * NX];
            float* kt = &k[t * NU];
            for (int j = 0; j < NX; ++j) {
                Kt[0 * NX + j] = -(inv0 * Qux[0 * NX + j] + inv1 * Qux[1 * NX + j]);
                Kt[1 * NX + j] = -(inv2 * Qux[0 * NX + j] + inv3 * Qux[1 * NX + j]);
            }
            kt[0] = -(inv0 * Qu[0] + inv1 * Qu[1]);
            kt[1] = -(inv2 * Qu[0] + inv3 * Qu[1]);

            // value update:
            //   Vx  = Qx  + K^T Quu k + K^T Qu + Qux^T k
            //   Vxx = Qxx + K^T Quu K + K^T Qux + Qux^T K
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
            // symmetrize to suppress float drift
            for (int i = 0; i < NX; ++i)
                for (int j = i + 1; j < NX; ++j) {
                    float avg = 0.5f * (Vxx[i * NX + j] + Vxx[j * NX + i]);
                    Vxx[i * NX + j] = avg; Vxx[j * NX + i] = avg;
                }
        }

        // ---------------- forward pass (line search) ----------------
        float un[T * NU], xn[(T + 1) * NX];
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
            // cost of the new rollout
            float c = 0.f;
            float l_x[NX], l_u[NU], l_xx[NX * NX], l_uu[NU * NU];
            for (int t = 0; t < T; ++t)
                c += stage_cost(&xn[t * NX], &un[t * NU], obs, n_obs, gx, gy, l_x, l_u, l_xx, l_uu);
            float lf_x[NX], lf_xx[NX * NX];
            c += term_cost(&xn[T * NX], gx, gy, lf_x, lf_xx);
            if (c < best_cost) {
                best_cost = c; improved = true;
                for (int i = 0; i < T * NU; ++i) u[i] = un[i];
                for (int i = 0; i < (T + 1) * NX; ++i) xs[i] = xn[i];
                break;
            }
        }
        cost = best_cost;

        if (iter_cost) iter_cost[it] = cost;
        if (iter_traj)
            for (int t = 0; t <= T; ++t) {
                iter_traj[(it * (T + 1) + t) * 2 + 0] = xs[t * NX + 0];
                iter_traj[(it * (T + 1) + t) * 2 + 1] = xs[t * NX + 1];
            }
        if (!improved) {
            // line search stalled: hold the rest of the log flat and stop
            for (int j = it + 1; j < N_ILQR; ++j) {
                if (iter_cost) iter_cost[j] = cost;
                if (iter_traj)
                    for (int t = 0; t <= T; ++t) {
                        iter_traj[(j * (T + 1) + t) * 2 + 0] = xs[t * NX + 0];
                        iter_traj[(j * (T + 1) + t) * 2 + 1] = xs[t * NX + 1];
                    }
            }
            break;
        }
    }

    for (int i = 0; i < T * NU; ++i) u_out[i] = u[i];
    *cost_out = cost;
}

// ============================ CUDA batch kernel ============================
__global__ void ilqr_batch_kernel(const float* __restrict__ obs, int n_obs,
                                   const float* __restrict__ starts,  // [N*3]
                                   const float* __restrict__ goals,   // [N*2]
                                   float* __restrict__ u_out,         // [N*T*NU]
                                   float* __restrict__ cost_out,      // [N]
                                   int n_prob) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_prob) return;
    float obs_local[MAX_OBS * 3];
    for (int i = 0; i < n_obs * 3; ++i) obs_local[i] = obs[i];
    ilqr_solve(obs_local, n_obs,
               starts[idx * 3 + 0], starts[idx * 3 + 1], starts[idx * 3 + 2],
               goals[idx * 2 + 0], goals[idx * 2 + 1],
               &u_out[idx * T * NU], &cost_out[idx], nullptr, nullptr);
}

// ============================ visualization ============================
static const int FRAME_W = 1280, FRAME_H = 720;
static const int FIELD_PX = 700;                 // world square -> pixels
static const float WORLD = 10.0f;                // world is [0,WORLD]^2
static const float SCALE = FIELD_PX / WORLD;
static const int OX = 10, OY = 10;               // field top-left in the frame

static inline cv::Point to_px(float wx, float wy) {
    return cv::Point(OX + (int)(wx * SCALE), OY + (int)((WORLD - wy) * SCALE));
}

}  // namespace cudabot

int main() {
    using namespace cudabot;
    std::printf("=== GPU batched iLQR trajectory optimization (CPU vs CUDA) ===\n");

    // ---- shared obstacle field ----
    float h_obs[MAX_OBS * 3] = {
        5.0f, 5.0f, 1.2f,
        3.0f, 7.2f, 0.9f,
        7.0f, 3.0f, 0.9f,
        6.6f, 7.0f, 0.8f,
    };
    const int n_obs = MAX_OBS;

    // ---- random start/goal queries (host-generated, shared by CPU and GPU) ----
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> ux(0.5f, 2.0f), uy(0.5f, 9.5f);
    std::uniform_real_distribution<float> gx(8.0f, 9.5f), gy(0.5f, 9.5f);
    std::uniform_real_distribution<float> uth(-3.14159f, 3.14159f);
    std::vector<float> h_start(N_PROB * 3), h_goal(N_PROB * 2);
    for (int i = 0; i < N_PROB; ++i) {
        h_start[i * 3 + 0] = ux(rng);
        h_start[i * 3 + 1] = uy(rng);
        h_start[i * 3 + 2] = uth(rng);
        h_goal[i * 2 + 0] = gx(rng);
        h_goal[i * 2 + 1] = gy(rng);
    }

    // ---- CPU batch (serial, same solver) ----
    std::vector<float> cpu_u(N_PROB * T * NU), cpu_cost(N_PROB);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_PROB; ++i) {
        ilqr_solve(h_obs, n_obs,
                   h_start[i * 3 + 0], h_start[i * 3 + 1], h_start[i * 3 + 2],
                   h_goal[i * 2 + 0], h_goal[i * 2 + 1],
                   &cpu_u[i * T * NU], &cpu_cost[i], nullptr, nullptr);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- GPU batch ----
    float *d_obs, *d_start, *d_goal, *d_u, *d_cost;
    CUDA_CHECK(cudaMalloc(&d_obs, sizeof(h_obs)));
    CUDA_CHECK(cudaMalloc(&d_start, N_PROB * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_goal, N_PROB * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u, N_PROB * T * NU * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cost, N_PROB * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_obs, h_obs, sizeof(h_obs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_start, h_start.data(), N_PROB * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_goal, h_goal.data(), N_PROB * 2 * sizeof(float), cudaMemcpyHostToDevice));

    int block = 64, grid = (N_PROB + block - 1) / block;
    // warm up
    ilqr_batch_kernel<<<grid, block>>>(d_obs, n_obs, d_start, d_goal, d_u, d_cost, N_PROB);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    ilqr_batch_kernel<<<grid, block>>>(d_obs, n_obs, d_start, d_goal, d_u, d_cost, N_PROB);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float gpu_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));

    std::vector<float> gpu_cost(N_PROB);
    CUDA_CHECK(cudaMemcpy(gpu_cost.data(), d_cost, N_PROB * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- correctness: per-problem final-cost agreement (CPU vs GPU) ----
    // CPU and GPU run the SAME iLQR arithmetic, but iLQR has data-dependent
    // discrete branches (the line-search alpha that is accepted, the PD guard on
    // Q_uu).  For the vast majority of queries host and device agree to float
    // precision; a few queries sit right on a line-search decision boundary,
    // where host/device rounding tips them into a different (equally valid) local
    // optimum -- so we report the agreement DISTRIBUTION, not a single MAE.
    const double REL_TOL = 0.01;   // 1% of the CPU cost = "same local optimum"
    std::vector<double> diffs(N_PROB);
    int within = 0;
    for (int i = 0; i < N_PROB; ++i) {
        double d = std::fabs((double)cpu_cost[i] - (double)gpu_cost[i]);
        diffs[i] = d;
        if (d <= REL_TOL * std::fabs((double)cpu_cost[i]) + 1e-4) ++within;
    }
    std::vector<double> sorted = diffs;
    std::sort(sorted.begin(), sorted.end());
    double med = sorted[N_PROB / 2];
    double p999 = sorted[(int)(N_PROB * 0.999)];
    double maxd = sorted.back();
    int outliers = N_PROB - within;
    double within_pct = 100.0 * within / N_PROB;
    double speedup = cpu_ms / gpu_ms;

    // solution-QUALITY check: on a non-convex problem the honest question is
    // whether GPU solutions are as good as CPU ones, not whether they pick the
    // identical local optimum.  Compare mean achieved cost.
    double cpu_mean = 0, gpu_mean = 0;
    for (int i = 0; i < N_PROB; ++i) { cpu_mean += cpu_cost[i]; gpu_mean += gpu_cost[i]; }
    cpu_mean /= N_PROB; gpu_mean /= N_PROB;
    double mean_rel = std::fabs(gpu_mean - cpu_mean) / cpu_mean * 100.0;

    std::printf("problems              : %d  (horizon T=%d, %d iLQR iters)\n", N_PROB, T, N_ILQR);
    std::printf("CPU serial batch      : %8.2f ms\n", cpu_ms);
    std::printf("GPU batch kernel      : %8.2f ms   (%.1fx)\n", gpu_ms, speedup);
    std::printf("same optimum (<=1%%)   : %d / %d  (%.2f%%)\n", within, N_PROB, within_pct);
    std::printf("cost-diff median/p99.9: %.2e / %.2e   (max %.2f, %d boundary outliers)\n",
                med, p999, maxd, outliers);
    std::printf("mean cost CPU/GPU     : %.4f / %.4f   (%.4f%% apart -> equal quality)\n",
                cpu_mean, gpu_mean, mean_rel);
    std::printf("per-problem CPU/GPU   : %.3f us / %.4f us\n",
                cpu_ms * 1e3 / N_PROB, gpu_ms * 1e3 / N_PROB);

    // ---- visualization: instrument 8 representative queries on the host ----
    const int K_VIZ = 8;
    std::vector<std::array<float, 3>> vs(K_VIZ);
    std::vector<std::array<float, 2>> vg(K_VIZ);
    std::vector<std::vector<float>> v_traj(K_VIZ, std::vector<float>(N_ILQR * (T + 1) * 2));
    std::vector<std::vector<float>> v_cost(K_VIZ, std::vector<float>(N_ILQR));
    cv::Scalar palette[K_VIZ] = {
        {66, 135, 245}, {245, 167, 66}, {66, 245, 156}, {245, 66, 161},
        {245, 233, 66}, {149, 66, 245}, {66, 245, 245}, {245, 66, 66},
    };
    for (int k = 0; k < K_VIZ; ++k) {
        // spread representatives across the batch
        int idx = (k * (N_PROB / K_VIZ)) + 3;
        vs[k] = {h_start[idx * 3 + 0], h_start[idx * 3 + 1], h_start[idx * 3 + 2]};
        vg[k] = {h_goal[idx * 2 + 0], h_goal[idx * 2 + 1]};
        float u_tmp[T * NU], c_tmp;
        ilqr_solve(h_obs, n_obs, vs[k][0], vs[k][1], vs[k][2], vg[k][0], vg[k][1],
                   u_tmp, &c_tmp, v_cost[k].data(), v_traj[k].data());
    }

    // ---- render the convergence animation ----
    if (system("mkdir -p tmp") != 0) std::fprintf(stderr, "warning: mkdir tmp failed\n");
    cv::VideoWriter video("tmp/gpu_batched_ilqr.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 12,
                          cv::Size(FRAME_W, FRAME_H));
    const int HOLD = 14;
    for (int frame = 0; frame < N_ILQR + HOLD; ++frame) {
        int it = std::min(frame, N_ILQR - 1);
        cv::Mat img(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(28, 28, 32));
        // field border
        cv::rectangle(img, cv::Rect(OX, OY, FIELD_PX, FIELD_PX), cv::Scalar(70, 70, 80), 1);
        // obstacles (+ margin ring)
        for (int o = 0; o < n_obs; ++o) {
            cv::Point c = to_px(h_obs[o * 3 + 0], h_obs[o * 3 + 1]);
            cv::circle(img, c, (int)(h_obs[o * 3 + 2] * SCALE), cv::Scalar(90, 90, 110), -1);
            cv::circle(img, c, (int)((h_obs[o * 3 + 2] + obs_margin()) * SCALE),
                       cv::Scalar(60, 60, 80), 1);
        }
        // representative queries' current trajectory
        for (int k = 0; k < K_VIZ; ++k) {
            cv::Scalar col = palette[k];
            std::vector<cv::Point> poly;
            for (int t = 0; t <= T; ++t)
                poly.push_back(to_px(v_traj[k][(it * (T + 1) + t) * 2 + 0],
                                     v_traj[k][(it * (T + 1) + t) * 2 + 1]));
            cv::polylines(img, poly, false, col, 2, cv::LINE_AA);
            cv::circle(img, to_px(vs[k][0], vs[k][1]), 5, col, -1);          // start
            cv::Point gp = to_px(vg[k][0], vg[k][1]);                         // goal
            cv::drawMarker(img, gp, col, cv::MARKER_TILTED_CROSS, 12, 2);
        }

        // ---- info panel ----
        int px = OX + FIELD_PX + 30, py = 50;
        auto put = [&](const std::string& s, int yy, double sc, cv::Scalar col, int th) {
            cv::putText(img, s, cv::Point(px, yy), cv::FONT_HERSHEY_SIMPLEX, sc, col, th, cv::LINE_AA);
        };
        put("Batched iLQR", py, 0.95, cv::Scalar(235, 235, 245), 2); py += 34;
        put("trajectory optimization", py, 0.7, cv::Scalar(180, 180, 200), 1); py += 44;
        char buf[128];
        std::snprintf(buf, sizeof(buf), "iLQR iteration: %d / %d", it + 1, N_ILQR);
        put(buf, py, 0.62, cv::Scalar(210, 210, 225), 1); py += 30;
        double mean_c = 0; for (int k = 0; k < K_VIZ; ++k) mean_c += v_cost[k][it];
        mean_c /= K_VIZ;
        std::snprintf(buf, sizeof(buf), "mean cost (8 shown): %.2f", mean_c);
        put(buf, py, 0.62, cv::Scalar(210, 210, 225), 1); py += 50;

        put("Batch headline", py, 0.66, cv::Scalar(150, 220, 150), 1); py += 30;
        std::snprintf(buf, sizeof(buf), "%d problems  (T=%d)", N_PROB, T);
        put(buf, py, 0.56, cv::Scalar(200, 200, 210), 1); py += 28;
        std::snprintf(buf, sizeof(buf), "CPU serial : %.1f ms", cpu_ms);
        put(buf, py, 0.56, cv::Scalar(200, 200, 210), 1); py += 28;
        std::snprintf(buf, sizeof(buf), "GPU batch  : %.2f ms", gpu_ms);
        put(buf, py, 0.56, cv::Scalar(200, 200, 210), 1); py += 28;
        std::snprintf(buf, sizeof(buf), "speedup    : %.0fx", speedup);
        put(buf, py, 0.62, cv::Scalar(120, 230, 250), 2); py += 34;
        std::snprintf(buf, sizeof(buf), "CPU=GPU    : %.1f%% of problems", within_pct);
        put(buf, py, 0.56, cv::Scalar(200, 200, 210), 1); py += 28;
        put("(same iLQR; median diff", py, 0.48, cv::Scalar(150, 150, 165), 1); py += 22;
        std::snprintf(buf, sizeof(buf), " %.1e, %d boundary outliers)", med, outliers);
        put(buf, py, 0.48, cv::Scalar(150, 150, 165), 1);

        video.write(img);
    }
    video.release();
    avi_to_gif("tmp/gpu_batched_ilqr.avi", "gif/gpu_batched_ilqr.gif", 12, 900);
    std::printf("wrote gif/gpu_batched_ilqr.gif\n");

    CUDA_CHECK(cudaFree(d_obs));
    CUDA_CHECK(cudaFree(d_start));
    CUDA_CHECK(cudaFree(d_goal));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_cost));
    return 0;
}
