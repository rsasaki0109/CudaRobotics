// gpu_cudampc.cu
//
// GPU-native nonlinear MPC: SCP + parallel-in-horizon ADMM in one fused,
// shared-memory-resident cooperative CUDA kernel with neighbor-local
// atomic-flag synchronization.
//
// Reproduction of:
//   B. Akbari, M. Greeff, "CudaMPC: A GPU-Native Solver for Model Predictive
//   Control", arXiv:2608.03051 (2026),
// which co-designs the optimization algorithm, execution model and memory
// architecture around CUDA:
//
//   * Algorithm -- sequential convex programming (SCP) wraps a parallel-in-
//     horizon ADMM splitting of the piMPC formulation (SOLARIS-JHU/piMPC.jl).
//     Copies z_{k+1} = xbar_{k+1}, v_k = Bbar_k Du_k are introduced and the
//     dynamics moved onto z, so every stagewise update depends only on its
//     immediate neighbours and the horizon parallelizes.
//   * Execution -- the complete ADMM iteration is fused into a single kernel.
//     The horizon is partitioned into sub-horizons of M stages, one per CUDA
//     block, launched cooperatively so all blocks are resident; a bounded
//     one-stage drift (partially asynchronous ADMM) is admitted.
//   * Memory -- primal/consensus/dual variables and cached stage matrices live
//     in shared memory.  Only boundary variables cross blocks, through global
//     memory, synchronized with pairwise atomic flags (no grid-wide barrier).
//
// Update equations (paper eqs. 8-13), per SCP linearization:
//   J_k = (R_k + rho Bbar_k^T Bbar_k)^{-1} Bbar_k^T
//   H_k = (Qbar_k + rho I + rho Abar_{k+1}^T Abar_{k+1})^{-1},
//   H_{N-1} = (Qbar_{N-1} + rho I)^{-1}
//   Du_k      = J_k (v_k - beta_k)
//   xbar_{k+1}= H_k h_k,  h_k = qbar_k + rho (z_{k+1}-theta_k + Abar_{k+1}^T r_{k+1})
//   r_{k+1}   = z_{k+2} - v_{k+1} - ebar_{k+1} + lambda_{k+1}
//   gamma_k   = Bbar_k Du_k + beta_k - Abar_k xbar_k - ebar_k + lambda_k
//   eta_k     = xbar_{k+1} + theta_k + Abar_k xbar_k + ebar_k - lambda_k
//   z_{k+1}   = Proj_Xbar((2 eta_k + gamma_k)/3),  v_k = (z_{k+1}+gamma_k)/2
//   theta,beta,lambda follow the scaled-dual increments.
// Nesterov acceleration with adaptive restart (piMPC default) is applied.
//
// Benchmarks (paper Table I): Pendulum (nx=2,nu=1), Cart-Pole (nx=4,nu=2) and
// Car Parking with OBCA collision-avoidance constraints (nx=4,nu=2).
//
// Build: CMakeLists, --expt-relaxed-constexpr.

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ============================== small dense LA (device) ==============================
template <int N>
__device__ __forceinline__ bool invN(const float* M, float* Mi) {
    float a[N * N], inv[N * N];
    for (int i = 0; i < N * N; ++i) { a[i] = M[i]; inv[i] = (i % (N + 1) == 0) ? 1.f : 0.f; }
    for (int c = 0; c < N; ++c) {
        int piv = c;
        for (int r = c + 1; r < N; ++r) if (fabsf(a[r * N + c]) > fabsf(a[piv * N + c])) piv = r;
        if (fabsf(a[piv * N + c]) < 1e-9f) return false;
        for (int j = 0; j < N; ++j) { float t = a[c * N + j]; a[c * N + j] = a[piv * N + j]; a[piv * N + j] = t;
                                      t = inv[c * N + j]; inv[c * N + j] = inv[piv * N + j]; inv[piv * N + j] = t; }
        float d = a[c * N + c];
        for (int j = 0; j < N; ++j) { a[c * N + j] /= d; inv[c * N + j] /= d; }
        for (int r = 0; r < N; ++r) if (r != c) {
            float f = a[r * N + c];
            for (int j = 0; j < N; ++j) { a[r * N + j] -= f * a[c * N + j]; inv[r * N + j] -= f * inv[c * N + j]; }
        }
    }
    for (int i = 0; i < N * N; ++i) Mi[i] = inv[i];
    return true;
}
__device__ __forceinline__ unsigned int ld_flag(const unsigned int* p) { return *((volatile const unsigned int*)p); }
__device__ __forceinline__ void st_flag(unsigned int* p, unsigned int v) { *((volatile unsigned int*)p) = v; }
__device__ __forceinline__ void spin_ge(const unsigned int* p, unsigned int v) {
    while (ld_flag(p) < v) {}
}

static const int MAXNX = 4, MAXHSMAX = 6;

// obstacle / ego geometry (constant memory; shared by host setup and device buildCons)
static const int MAXOBS = 6;
__constant__ float c_obs[3 * MAXOBS];
__constant__ int c_nobs;
__constant__ float CAR_A, CAR_B;

// ============================== problem models ==============================
struct Pendulum {
    static constexpr int NX = 2, NU = 1, MAXHS = 0;
    static constexpr float DT = 0.01f;
    __host__ __device__ __forceinline__ static void f(const float* x, const float* u, float* xn) {
        const float g = 9.81f, l = 1.0f, b = 0.1f;
        xn[0] = x[0] + DT * x[1];
        xn[1] = x[1] + DT * (-(g / l) * sinf(x[0]) - b * x[1] + u[0]);
    }
    __device__ __forceinline__ static void buildCons(const float*, float*, int& nhs) { nhs = 0; }
};

struct CartPole {  // x=[p, th, pdot, thdot], u=[F, tau]
    static constexpr int NX = 4, NU = 2, MAXHS = 0;
    static constexpr float DT = 0.01f;
    __host__ __device__ __forceinline__ static void f(const float* x, const float* u, float* xn) {
        const float Mc = 1.0f, Mp = 0.2f, l = 0.5f, g = 9.81f;
        const float I = (1.f / 3.f) * Mp * l * l;
        float p = x[0], th = x[1], pd = x[2], thd = x[3];
        float F = u[0], tau = u[1];
        float ct = cosf(th), st = sinf(th);
        float det = (Mc + Mp) * (Mp * l * l + I) - (Mp * l * ct) * (Mp * l * ct);
        if (fabsf(det) < 1e-6f) det = (det < 0 ? -1e-6f : 1e-6f);
        float pdd = ((Mp * l * l + I) * (F + Mp * l * st * thd * thd) - Mp * l * ct * (tau + Mp * g * l * st)) / det;
        float thdd = ((Mc + Mp) * (tau + Mp * g * l * st) - Mp * l * ct * (F + Mp * l * st * thd * thd)) / det;
        xn[0] = p + DT * pd; xn[1] = th + DT * thd;
        xn[2] = pd + DT * pdd; xn[3] = thd + DT * thdd;
    }
    __device__ __forceinline__ static void buildCons(const float*, float*, int& nhs) { nhs = 0; }
};

// Kinematic bicycle, x=[px,py,theta,v], u=[a,delta]; OBCA rectangle-vs-circle
// collision half-spaces (one per obstacle) on (px,py,theta).
struct CarParking {
    static constexpr int NX = 4, NU = 2, MAXHS = 6;
    static constexpr float DT = 0.1f;
    __host__ __device__ __forceinline__ static void f(const float* x, const float* u, float* xn) {
        float px = x[0], py = x[1], th = x[2], v = x[3];
        float a = u[0], w = u[1];
        xn[0] = px + DT * v * cosf(th);
        xn[1] = py + DT * v * sinf(th);
        xn[2] = th + DT * w;
        xn[3] = v + DT * a;
    }
    __device__ __forceinline__ static void buildCons(const float* x, float* hs, int& nhs) {
        float px = x[0], py = x[1], th = x[2];
        float c = cosf(th), s = sinf(th);
        nhs = 0;
        for (int j = 0; j < c_nobs; ++j) {
            if (nhs >= MAXHS) break;
            float ox = c_obs[2 * j], oy = c_obs[2 * j + 1], r = c_obs[2 * MAXOBS + j];
            float dx = ox - px, dy = oy - py;
            float lx = c * dx + s * dy, ly = -s * dx + c * dy;
            float qx = fminf(CAR_A, fmaxf(-CAR_A, lx));
            float qy = fminf(CAR_B, fmaxf(-CAR_B, ly));
            float ex = lx - qx, ey = ly - qy;
            float dist = sqrtf(ex * ex + ey * ey);
            if (dist < 1e-4f) { ex = 1.f; ey = 0.f; dist = 1.f; }
            float nlx = ex / dist, nly = ey / dist;
            float nx = c * nlx - s * nly, ny = s * nlx + c * nly;
            float rqx = c * qx - s * qy, rqy = s * qx + c * qy;
            float drqx = -s * qx - c * qy, drqy = c * qx - s * qy;
            float ndrq = nx * drqx + ny * drqy;
            // constraint g = n.(o - p - R(th) q) - r >= 0, linearized; stored as a^T x <= b
            // with a = -[n ; ndrq ; 0], b = -r + n.o - n.rq + ndrq*th
            float* h = &hs[nhs * (NX + 1)];
            h[0] = nx; h[1] = ny; h[2] = ndrq; h[3] = 0.f;
            h[4] = -r + nx * ox + ny * oy - nx * rqx - ny * rqy + ndrq * th;
            nhs++;
        }
        for (int j = nhs; j < MAXHS; ++j) { float* h = &hs[j * (NX + 1)]; for (int i = 0; i <= NX; ++i) h[i] = 0.f; h[NX] = 1e6f; }
    }
};

// ============================== numeric linearization ==============================
template <class Mdl>
__device__ __forceinline__ void linModel(const float* x, const float* u, float* A, float* B, float* e) {
    const int NX = Mdl::NX, NU = Mdl::NU;
    const float eps = 1e-4f;
    float f0[MAXNX], fp[MAXNX], fm[MAXNX], xp[MAXNX], xm[MAXNX], up[MAXNX], um[MAXNX];
    Mdl::f(x, u, f0);
    for (int j = 0; j < NX; ++j) {
        for (int i = 0; i < NX; ++i) { xp[i] = x[i]; xm[i] = x[i]; }
        xp[j] += eps; xm[j] -= eps;
        Mdl::f(xp, u, fp); Mdl::f(xm, u, fm);
        for (int i = 0; i < NX; ++i) A[i * NX + j] = (fp[i] - fm[i]) / (2 * eps);
    }
    for (int j = 0; j < NU; ++j) {
        for (int i = 0; i < NU; ++i) { up[i] = u[i]; um[i] = u[i]; }
        up[j] += eps; um[j] -= eps;
        Mdl::f(x, up, fp); Mdl::f(x, um, fm);
        for (int i = 0; i < NX; ++i) B[i * NU + j] = (fp[i] - fm[i]) / (2 * eps);
    }
    for (int i = 0; i < NX; ++i) {
        float s = f0[i];
        for (int j = 0; j < NX; ++j) s -= A[i * NX + j] * x[j];
        for (int j = 0; j < NU; ++j) s -= B[i * NU + j] * u[j];
        e[i] = s;
    }
}

// ============================== prep kernels ==============================
template <class Mdl>
__global__ void prep1_kernel(const float* __restrict__ X, const float* __restrict__ U, int N,
                             const float* __restrict__ Qw, const float* __restrict__ Qfw,
                             float Qu, float Rval, const float* __restrict__ xref, const float* __restrict__ uref,
                             float* __restrict__ Abar, float* __restrict__ Bbar, float* __restrict__ Ebar,
                             float* __restrict__ Qbar, float* __restrict__ qbar, float* __restrict__ Rdiag,
                             float* __restrict__ HS) {
    const int NX = Mdl::NX, NU = Mdl::NU, nn = NX + NU, MAXHS = Mdl::MAXHS;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    float x[MAXNX], u[MAXNX];
    for (int i = 0; i < NX; ++i) x[i] = X[k * NX + i];
    for (int i = 0; i < NU; ++i) u[i] = U[k * NU + i];
    float A[MAXNX * MAXNX], B[MAXNX * MAXNX], e[MAXNX];
    linModel<Mdl>(x, u, A, B, e);
    float* Ab = &Abar[k * nn * nn];
    float* Bb = &Bbar[k * nn * NU];
    float* Eb = &Ebar[k * nn];
    for (int i = 0; i < nn * nn; ++i) Ab[i] = 0.f;
    for (int i = 0; i < NX; ++i) for (int j = 0; j < NX; ++j) Ab[i * nn + j] = A[i * NX + j];
    for (int i = 0; i < NX; ++i) for (int j = 0; j < NU; ++j) Ab[i * nn + NX + j] = B[i * NU + j];
    for (int i = 0; i < NU; ++i) Ab[(NX + i) * nn + NX + i] = 1.f;
    for (int i = 0; i < nn * NU; ++i) Bb[i] = 0.f;
    for (int i = 0; i < NX; ++i) for (int j = 0; j < NU; ++j) Bb[i * NU + j] = B[i * NU + j];
    for (int i = 0; i < NU; ++i) Bb[(NX + i) * NU + i] = 1.f;
    for (int i = 0; i < NX; ++i) Eb[i] = e[i];
    for (int i = NX; i < nn; ++i) Eb[i] = 0.f;
    const float* Qx = (k == N - 1) ? Qfw : Qw;
    for (int i = 0; i < nn; ++i) Qbar[k * nn + i] = (i < NX) ? Qx[i] : Qu;
    for (int i = 0; i < NX; ++i) qbar[k * nn + i] = Qx[i] * xref[i];
    for (int i = 0; i < NU; ++i) qbar[k * nn + NX + i] = Qu * uref[i];
    for (int i = 0; i < NU; ++i) Rdiag[k * NU + i] = Rval;
    if (MAXHS > 0) { int nhs = 0; Mdl::buildCons(&X[(k + 1) * NX], &HS[k * MAXHS * (NX + 1)], nhs); }
}

template <class Mdl>
__global__ void prep2_kernel(int N, float rho,
                             const float* __restrict__ Abar, const float* __restrict__ Bbar,
                             const float* __restrict__ Qbar, const float* __restrict__ Qfw, float Qu,
                             const float* __restrict__ Rdiag,
                             float* __restrict__ J, float* __restrict__ H, float* __restrict__ HAN) {
    const int NX = Mdl::NX, NU = Mdl::NU, nn = NX + NU;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    float M[NU * NU];
    for (int i = 0; i < NU * NU; ++i) M[i] = 0.f;
    for (int a = 0; a < nn; ++a) for (int i = 0; i < NU; ++i) for (int j = 0; j < NU; ++j)
        M[i * NU + j] += rho * Bbar[(k * nn + a) * NU + i] * Bbar[(k * nn + a) * NU + j];
    for (int i = 0; i < NU; ++i) M[i * NU + i] += Rdiag[k * NU + i] + 1e-9f;
    float Mi[NU * NU]; invN<NU>(M, Mi);
    for (int i = 0; i < NU; ++i) for (int a = 0; a < nn; ++a) {
        float s = 0; for (int j = 0; j < NU; ++j) s += Mi[i * NU + j] * Bbar[(k * nn + a) * NU + j];
        J[(k * NU + i) * nn + a] = s;
    }
    if (k == N - 1) {
        float Mt[nn * nn];
        for (int i = 0; i < nn * nn; ++i) Mt[i] = 0.f;
        for (int i = 0; i < nn; ++i) Mt[i * nn + i] = ((i < NX) ? Qfw[i] : Qu) + rho;
        invN<nn>(Mt, HAN);
        return;
    }
    float Mh[nn * nn];
    for (int i = 0; i < nn * nn; ++i) Mh[i] = 0.f;
    for (int i = 0; i < nn; ++i) Mh[i * nn + i] = Qbar[k * nn + i] + rho;
    const float* An = &Abar[(k + 1) * nn * nn];
    for (int i = 0; i < nn; ++i) for (int j = 0; j < nn; ++j) {
        float s = 0; for (int t = 0; t < nn; ++t) s += An[t * nn + i] * An[t * nn + j];
        Mh[i * nn + j] += rho * s;
    }
    invN<nn>(Mh, &H[k * nn * nn]);
}

template <class Mdl>
__global__ void rollout_kernel(const float* __restrict__ x0, const float* __restrict__ U, int N,
                               float* __restrict__ Xout) {
    const int NX = Mdl::NX;
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < NX; ++i) Xout[i] = x0[i];
    for (int k = 0; k < N; ++k) Mdl::f(&Xout[k * NX], &U[k * Mdl::NU], &Xout[(k + 1) * NX]);
}

// projection of one augmented stage onto Xbar = X x U (Dykstra on the state)
template <class Mdl>
__device__ void projectStage(float* z, const float* xmin, const float* xmax,
                             const float* umin, const float* umax, const float* hs) {
    const int NX = Mdl::NX, NU = Mdl::NU, MAXHS = Mdl::MAXHS;
    for (int i = 0; i < NU; ++i) z[NX + i] = fminf(umax[i], fmaxf(umin[i], z[NX + i]));
    const int m = 2 * NX + MAXHS;
    float A[(2 * MAXNX + MAXHSMAX) * MAXNX], Bc[2 * MAXNX + MAXHSMAX], corr[(2 * MAXNX + MAXHSMAX) * MAXNX];
    int c = 0;
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NX; ++j) A[c * NX + j] = 0.f; A[c * NX + i] = -1.f; Bc[c] = -xmin[i]; c++;
        for (int j = 0; j < NX; ++j) A[c * NX + j] = 0.f; A[c * NX + i] = 1.f; Bc[c] = xmax[i]; c++;
    }
    for (int j = 0; j < MAXHS; ++j) { for (int i = 0; i < NX; ++i) A[c * NX + i] = hs[j * (NX + 1) + i]; Bc[c] = hs[j * (NX + 1) + NX]; c++; }
    for (int i = 0; i < m * NX; ++i) corr[i] = 0.f;
    for (int sw = 0; sw < 4; ++sw) {
        for (int ci = 0; ci < m; ++ci) {
            float* p = &corr[ci * NX];
            float y[MAXNX], zz[MAXNX];
            for (int i = 0; i < NX; ++i) y[i] = z[i] + p[i];
            float aty = 0, n2 = 0;
            for (int i = 0; i < NX; ++i) { aty += A[ci * NX + i] * y[i]; n2 += A[ci * NX + i] * A[ci * NX + i]; }
            if (aty > Bc[ci] && n2 > 1e-12f) { float t = (aty - Bc[ci]) / n2; for (int i = 0; i < NX; ++i) zz[i] = y[i] - t * A[ci * NX + i]; }
            else for (int i = 0; i < NX; ++i) zz[i] = y[i];
            for (int i = 0; i < NX; ++i) { p[i] = y[i] - zz[i]; z[i] = zz[i]; }
        }
    }
}

// shared-memory layout (must match the kernel)
template <class Mdl>
static int smemFloats(int M) {
    const int NX = Mdl::NX, NU = Mdl::NU, nn = NX + NU, MAXHS = Mdl::MAXHS;
    int o = 0;
    o += (M + 1) * nn;      // X
    o += 5 * M * nn;        // Z,V,Th,Be,La
    o += 5 * M * nn;        // hats
    o += 5 * M * nn;        // olds
    o += M * NU;            // DU
    o += 2 * M * nn;        // BU, AX
    o += M * nn * nn;       // Abar
    o += M * nn * NU;       // Bbar
    o += M * nn;            // Ebar
    o += M * nn;            // Qbar
    o += M * NU;            // Rdiag
    o += M * NU * nn;       // J
    o += M * nn * nn;       // H
    o += M * MAXHS * (NX + 1);
    o += nn * nn;           // HAN
    o += 3 * nn;            // boundary buffers
    o += 2 * NX + 2 * NU;   // boxes
    return o;
}

// ============================== fused parallel-in-horizon ADMM ==============================
template <class Mdl>
__global__ void cudampc_admm_kernel(
    int N, int M, float rho, int admm_iters, int B,
    const float* __restrict__ gX0,
    const float* __restrict__ gXcur, const float* __restrict__ gUcur,
    const float* __restrict__ gAbar, const float* __restrict__ gBbar, const float* __restrict__ gEbar,
    const float* __restrict__ gQbar, const float* __restrict__ gqbar, const float* __restrict__ gRdiag,
    const float* __restrict__ gJ, const float* __restrict__ gH, const float* __restrict__ gHAN,
    const float* __restrict__ gHS,
    const float* __restrict__ gXmin, const float* __restrict__ gXmax,
    const float* __restrict__ gUmin, const float* __restrict__ gUmax,
    float* __restrict__ gUout, float* __restrict__ gXout,
    float* __restrict__ gDbgRes,
    float* __restrict__ bxb, float* __restrict__ bzc, float* __restrict__ bvc, float* __restrict__ blc,
    unsigned int* __restrict__ cp, unsigned int* __restrict__ cc) {
    const int NX = Mdl::NX, NU = Mdl::NU, nn = NX + NU, MAXHS = Mdl::MAXHS;
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int gk0 = b * M;
    const int NT = M * nn;

    extern __shared__ float smem[];
    int o = 0;
    float* sX = smem + o; o += (M + 1) * nn;
    float* sZ = smem + o; o += M * nn;
    float* sV = smem + o; o += M * nn;
    float* sTh = smem + o; o += M * nn;
    float* sBe = smem + o; o += M * nn;
    float* sLa = smem + o; o += M * nn;
    float* sZh = smem + o; o += M * nn;
    float* sVh = smem + o; o += M * nn;
    float* sThh = smem + o; o += M * nn;
    float* sBeh = smem + o; o += M * nn;
    float* sLah = smem + o; o += M * nn;
    float* sZo = smem + o; o += M * nn;
    float* sVo = smem + o; o += M * nn;
    float* sTho = smem + o; o += M * nn;
    float* sBeo = smem + o; o += M * nn;
    float* sLao = smem + o; o += M * nn;
    float* sDU = smem + o; o += M * NU;
    float* sBU = smem + o; o += M * nn;
    float* sAX = smem + o; o += M * nn;
    float* sAb = smem + o; o += M * nn * nn;
    float* sBb = smem + o; o += M * nn * NU;
    float* sEb = smem + o; o += M * nn;
    float* sQb = smem + o; o += M * nn;
    float* sRd = smem + o; o += M * NU;
    float* sJ = smem + o; o += M * NU * nn;
    float* sH = smem + o; o += M * nn * nn;
    float* sHS = smem + o; o += M * MAXHS * (NX + 1);
    float* sHAN = smem + o; o += nn * nn;
    float* sZB = smem + o; o += nn;
    float* sVB = smem + o; o += nn;
    float* sLB = smem + o; o += nn;
    float* sXmin = smem + o; o += NX;
    float* sXmax = smem + o; o += NX;
    float* sUmin = smem + o; o += NU;
    float* sUmax = smem + o; o += NU;
    __shared__ float s_res, s_alpha, s_resprev, s_mom;
    __shared__ int s_reset;

    // ---- load prep data, init variables ----
    for (int k = 0; k < M; ++k) {
        int gk = gk0 + k;
        if (gk >= N) continue;
        for (int i = tid; i < nn * nn; i += blockDim.x) sAb[k * nn * nn + i] = gAbar[gk * nn * nn + i];
        for (int i = tid; i < nn * NU; i += blockDim.x) sBb[k * nn * NU + i] = gBbar[gk * nn * NU + i];
        for (int i = tid; i < nn; i += blockDim.x) sEb[k * nn + i] = gEbar[gk * nn + i];
        for (int i = tid; i < nn; i += blockDim.x) sQb[k * nn + i] = gqbar[gk * nn + i];
        for (int i = tid; i < NU; i += blockDim.x) sRd[k * NU + i] = gRdiag[gk * NU + i];
        for (int i = tid; i < NU * nn; i += blockDim.x) sJ[k * NU * nn + i] = gJ[gk * NU * nn + i];
        if (gk < N - 1) for (int i = tid; i < nn * nn; i += blockDim.x) sH[k * nn * nn + i] = gH[gk * nn * nn + i];
        if (MAXHS > 0) for (int i = tid; i < MAXHS * (NX + 1); i += blockDim.x) sHS[k * MAXHS * (NX + 1) + i] = gHS[gk * MAXHS * (NX + 1) + i];
        for (int i = tid; i < nn; i += blockDim.x) {
            sZ[k * nn + i] = 0.f; sV[k * nn + i] = 0.f; sTh[k * nn + i] = 0.f; sBe[k * nn + i] = 0.f; sLa[k * nn + i] = 0.f;
            sZh[k * nn + i] = 0.f; sVh[k * nn + i] = 0.f; sThh[k * nn + i] = 0.f; sBeh[k * nn + i] = 0.f; sLah[k * nn + i] = 0.f;
        }
        for (int i = tid; i < NU; i += blockDim.x) sDU[k * NU + i] = 0.f;
        for (int i = tid; i < nn; i += blockDim.x) {
            if (gk == 0) sX[i] = gX0[i];
            else sX[k * nn + i] = (i < NX) ? gXcur[gk * NX + i] : gUcur[(gk - 1) * NU + (i - NX)];
        }
    }
    for (int i = tid; i < nn * nn; i += blockDim.x) sHAN[i] = gHAN[i];
    for (int i = tid; i < NX; i += blockDim.x) { sXmin[i] = gXmin[i]; sXmax[i] = gXmax[i]; }
    for (int i = tid; i < NU; i += blockDim.x) { sUmin[i] = gUmin[i]; sUmax[i] = gUmax[i]; }
    if (tid == 0) { s_alpha = 1.f; s_resprev = 1e30f; s_res = 0.f; }
    __syncthreads();
    long long t_start = clock64(), spin_cyc = 0;

    for (int it = 0; it < admm_iters; ++it) {
        // ---------------- Phase A: primal ----------------
        if (b + 1 < B) {
            int par = (it + 1) & 1;  // previous iteration's consensus buffer
            if (tid == 0) { long long t0 = clock64(); spin_ge(&cc[b + 1], (unsigned)it); spin_cyc += clock64() - t0; }
            __syncthreads();
            for (int i = tid; i < nn; i += blockDim.x) {
                sZB[i] = bzc[((b + 1) * 2 + par) * nn + i];
                sVB[i] = bvc[((b + 1) * 2 + par) * nn + i];
                sLB[i] = blc[((b + 1) * 2 + par) * nn + i];
            }
            __syncthreads();
        }
        for (int t = tid; t < NT; t += blockDim.x) {
            int k = t / nn, i = t % nn, gk = gk0 + k;
            if (gk >= N) continue;
            if (i < NU) {
                float s = 0;
                for (int a = 0; a < nn; ++a) s += sJ[(k * NU + i) * nn + a] * (sVh[k * nn + a] - sBeh[k * nn + a]);
                sDU[k * NU + i] = s;
            }
            float h = sQb[k * nn + i] + rho * (sZh[k * nn + i] - sThh[k * nn + i]);
            if (gk < N - 1) {
                const float* An = (k < M - 1) ? &sAb[(k + 1) * nn * nn] : &gAbar[(gk + 1) * nn * nn];
                float Atr = 0;
                for (int tt = 0; tt < nn; ++tt) {
                    float rt = (k < M - 1)
                        ? (sZh[(k + 1) * nn + tt] - sVh[(k + 1) * nn + tt] + sLah[(k + 1) * nn + tt] - sEb[(k + 1) * nn + tt])
                        : (sZB[tt] - sVB[tt] + sLB[tt] - gEbar[(gk + 1) * nn + tt]);
                    Atr += An[tt * nn + i] * rt;
                }
                h += rho * Atr;
            }
            sBU[k * nn + i] = h;  // temp: raw h_k before H_k
        }
        __syncthreads();
        for (int t = tid; t < NT; t += blockDim.x) {
            int k = t / nn, i = t % nn, gk = gk0 + k;
            if (gk >= N) continue;
            float h[MAXNX + MAXNX];
            for (int j = 0; j < nn; ++j) h[j] = sBU[k * nn + j];
            float out = 0;
            if (gk == N - 1) { for (int j = 0; j < nn; ++j) out += sHAN[i * nn + j] * h[j]; }
            else { for (int j = 0; j < nn; ++j) out += sH[k * nn * nn + i * nn + j] * h[j]; }
            sX[(k + 1) * nn + i] = out;
        }
        __syncthreads();
        for (int i = tid; i < nn; i += blockDim.x) bxb[(b * 2 + (it & 1)) * nn + i] = sX[M * nn + i];
        __threadfence();
        __syncthreads();
        if (tid == 0) { __threadfence(); st_flag(&cp[b], (unsigned)(it + 1)); }

        // ---------------- Phase B: consensus ----------------
        if (b > 0) {
            if (tid == 0) { long long t0 = clock64(); spin_ge(&cp[b - 1], (unsigned)(it + 1)); spin_cyc += clock64() - t0; }
            __syncthreads();
            for (int i = tid; i < nn; i += blockDim.x) sX[i] = bxb[((b - 1) * 2 + (it & 1)) * nn + i];
            __syncthreads();
        }
        // save previous iterates (for momentum) and compute BU/AX/Z in one pass
        for (int t = tid; t < NT; t += blockDim.x) {
            int k = t / nn, i = t % nn, gk = gk0 + k;
            if (gk >= N) continue;
            sZo[k * nn + i] = sZ[k * nn + i]; sVo[k * nn + i] = sV[k * nn + i]; sTho[k * nn + i] = sTh[k * nn + i];
            sBeo[k * nn + i] = sBe[k * nn + i]; sLao[k * nn + i] = sLa[k * nn + i];
            float bu = 0; for (int j = 0; j < NU; ++j) bu += sBb[(k * nn + i) * NU + j] * sDU[k * NU + j];
            float ax = 0; for (int j = 0; j < nn; ++j) ax += sAb[(k * nn + i) * nn + j] * sX[k * nn + j];
            sBU[k * nn + i] = bu;
            sAX[k * nn + i] = ax;
            sZ[k * nn + i] = (2.f * (sX[(k + 1) * nn + i] + sThh[k * nn + i]) + bu + sBeh[k * nn + i] + ax + sEb[k * nn + i] - sLah[k * nn + i]) / 3.f;
        }
        __syncthreads();
        for (int kk = tid; kk < M; kk += blockDim.x) {
            int gk = gk0 + kk;
            if (gk < N) projectStage<Mdl>(&sZ[kk * nn], sXmin, sXmax, sUmin, sUmax,
                                          MAXHS > 0 ? &sHS[kk * MAXHS * (NX + 1)] : nullptr);
        }
        __syncthreads();
        // primal/dual updates + local residual accumulation
        float loc = 0.f;
        for (int t = tid; t < NT; t += blockDim.x) {
            int k = t / nn, i = t % nn, gk = gk0 + k;
            if (gk >= N) continue;
            float bu = sBU[k * nn + i], ax = sAX[k * nn + i];
            float Zn = sZ[k * nn + i];
            float Vn = 0.5f * (Zn + bu + sBeh[k * nn + i] - ax - sEb[k * nn + i] + sLah[k * nn + i]);
            float Tn = sThh[k * nn + i] + sX[(k + 1) * nn + i] - Zn;
            float Bn = sBeh[k * nn + i] + bu - Vn;
            float Ln = sLah[k * nn + i] + Zn - ax - Vn - sEb[k * nn + i];
            sZ[k * nn + i] = Zn; sV[k * nn + i] = Vn; sTh[k * nn + i] = Tn; sBe[k * nn + i] = Bn; sLa[k * nn + i] = Ln;
            float a1 = Tn - sThh[k * nn + i], a2 = Bn - sBeh[k * nn + i], a3 = Ln - sLah[k * nn + i];
            float a4 = Zn - sZh[k * nn + i], a5 = Vn - sVh[k * nn + i];
            float a6 = (Zn - Vn) - (sZh[k * nn + i] - sVh[k * nn + i]);
            loc += a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6;
        }
        __syncthreads();
        atomicAdd(&s_res, loc);
        __syncthreads();
        if (tid == 0) {
            float res = rho * s_res;
            const float eta = 0.999f;
            if (res < eta * s_resprev) {
                float alpha = 0.5f * (1.f + sqrtf(1.f + 4.f * s_alpha * s_alpha));
                s_mom = (s_alpha - 1.f) / alpha;
                s_alpha = alpha; s_resprev = res; s_reset = 0;
            } else {
                s_mom = 0.f; s_resprev = res / eta; s_alpha = 1.f; s_reset = 1;
            }
        }
        __syncthreads();
        float mom = s_mom; int reset = s_reset;
        for (int t = tid; t < NT; t += blockDim.x) {
            int k = t / nn, i = t % nn, gk = gk0 + k;
            if (gk >= N) continue;
            float Zo = sZo[k * nn + i], Vo = sVo[k * nn + i], To = sTho[k * nn + i], Bo = sBeo[k * nn + i], Lo = sLao[k * nn + i];
            float Zn = sZ[k * nn + i], Vn = sV[k * nn + i], Tn = sTh[k * nn + i], Bn = sBe[k * nn + i], Ln = sLa[k * nn + i];
            if (reset) {
                sZh[k * nn + i] = Zn; sVh[k * nn + i] = Vn; sThh[k * nn + i] = Tn; sBeh[k * nn + i] = Bn; sLah[k * nn + i] = Ln;
            } else {
                sZh[k * nn + i] = Zn + mom * (Zn - Zo); sVh[k * nn + i] = Vn + mom * (Vn - Vo);
                sThh[k * nn + i] = Tn + mom * (Tn - To); sBeh[k * nn + i] = Bn + mom * (Bn - Bo);
                sLah[k * nn + i] = Ln + mom * (Ln - Lo);
            }
        }
        if (tid == 0) s_res = 0.f;  // reset accumulator for the next iteration
        __syncthreads();
        for (int i = tid; i < nn; i += blockDim.x) {
            bzc[(b * 2 + (it & 1)) * nn + i] = sZ[i];
            bvc[(b * 2 + (it & 1)) * nn + i] = sV[i];
            blc[(b * 2 + (it & 1)) * nn + i] = sLa[i];
        }
        __threadfence();
        __syncthreads();
        if (tid == 0) { __threadfence(); st_flag(&cc[b], (unsigned)(it + 1)); }
    }

    for (int k = 0; k < M; ++k) {
        int gk = gk0 + k;
        if (gk >= N) break;
        for (int i = tid; i < NU; i += blockDim.x) gUout[gk * NU + i] = sX[(k + 1) * nn + NX + i];
        for (int i = tid; i < NX; i += blockDim.x) gXout[gk * NX + i] = sX[k * nn + i];
    }
    if (gDbgRes && tid == 0) gDbgRes[b] = rho * s_res;
    if (gDbgRes && b == 0 && tid == 0) {
        float r1 = 0, r2 = 0, r3 = 0;
        for (int k = 0; k < M; ++k) {
            if (gk0 + k >= N) break;
            for (int i = 0; i < nn; ++i) {
                r1 = fmaxf(r1, fabsf(sX[(k + 1) * nn + i] - sZ[k * nn + i]));
                r2 = fmaxf(r2, fabsf(sBU[k * nn + i] - sV[k * nn + i]));
                r3 = fmaxf(r3, fabsf(sZ[k * nn + i] - sAX[k * nn + i] - sV[k * nn + i] - sEb[k * nn + i]));
            }
        }
        gDbgRes[1] = r1; gDbgRes[2] = r2; gDbgRes[3] = r3;
        gDbgRes[4] = (float)spin_cyc; gDbgRes[5] = (float)(clock64() - t_start);
    }
}

// ============================== host problem driver ==============================
struct ProblemCfg {
    int N = 200, steps = 200, admm_iters = 2000, scp_iters = 6;
    int forceM = 0;
    float rho = 10.f, Qu = 0.f, R = 0.1f;
    std::vector<float> Q, Qf, xref, uref, xmin, xmax, umin, umax, x0;
    std::vector<float> obs;  // x,y,r triples
};

struct RunResult {
    std::vector<std::vector<float>> traj;  // [step][state]
    std::vector<float> goals;
    std::vector<float> u0_first;
    std::vector<float> dbg_res;
    float max_u = 0.f, min_clear = 1e9f, final_err = 0.f, rmse = 0.f;
    int collisions = 0, box_viol = 0;
    double solve_ms = 0.0;
    double prep_ms = 0.0, admm_ms = 0.0, rollout_ms = 0.0;
    bool reach = false;
};

// closed-loop run for a given model
template <class Mdl>
static RunResult runClosedLoop(const ProblemCfg& cfg, unsigned seed) {
    const int NX = Mdl::NX, NU = Mdl::NU, nn = NX + NU;
    const int N = cfg.N;
    RunResult res;
    res.goals = cfg.xref;

    // obstacles to constant memory (only meaningful for CarParking)
    if (!cfg.obs.empty()) {
        int nobs = (int)cfg.obs.size() / 3;
        if (nobs > MAXOBS) nobs = MAXOBS;
        std::vector<float> ob(3 * MAXOBS, 0.f);
        for (int j = 0; j < nobs; ++j) { ob[2 * j] = cfg.obs[3 * j]; ob[2 * j + 1] = cfg.obs[3 * j + 1]; ob[2 * MAXOBS + j] = cfg.obs[3 * j + 2]; }
        CUDA_CHECK(cudaMemcpyToSymbol(c_obs, ob.data(), 3 * MAXOBS * sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_nobs, &nobs, sizeof(int)));
        float ca = 0.9f, cb = 0.45f;
        CUDA_CHECK(cudaMemcpyToSymbol(CAR_A, &ca, sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(CAR_B, &cb, sizeof(float)));
    } else {
        int nobs = 0; CUDA_CHECK(cudaMemcpyToSymbol(c_nobs, &nobs, sizeof(int)));
    }

    // device buffers
    float *dX, *dU, *dX0, *dXout, *dUout, *dAb, *dBb, *dEb, *dQb, *dqb, *dRd, *dJ, *dH, *dHAN, *dHS;
    float *dQw, *dQfw, *dxref, *duref, *dXmin, *dXmax, *dUmin, *dUmax;
    unsigned int *dcp, *dcc;
    float *bxb, *bzc, *bvc, *blc, *dDbg;
    CUDA_CHECK(cudaMalloc(&dDbg, (size_t)(256) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dX, (size_t)(N + 1) * NX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dU, (size_t)N * NU * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dX0, nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dXout, (size_t)(N + 1) * NX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dUout, (size_t)N * NU * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dAb, (size_t)N * nn * nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBb, (size_t)N * nn * NU * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dEb, (size_t)N * nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQb, (size_t)N * nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dqb, (size_t)N * nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRd, (size_t)N * NU * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dJ, (size_t)N * NU * nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dH, (size_t)N * nn * nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dHAN, (size_t)nn * nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dHS, (size_t)N * Mdl::MAXHS * (NX + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQw, NX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQfw, NX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dxref, NX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&duref, NU * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dXmin, NX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dXmax, NX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dUmin, NU * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dUmax, NU * sizeof(float)));

    // choose sub-horizon M so the grid is resident and shared memory fits
    int smemBudgetBytes = 48 * 1024;
    int numSms = 0; CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, 0));
    int M = 32;
    if (cfg.forceM > 0) M = cfg.forceM;
    {
        int need = smemFloats<Mdl>(M) * (int)sizeof(float);
        while (M > 4 && need > smemBudgetBytes) { M -= 4; need = smemFloats<Mdl>(M) * (int)sizeof(float); }
    }
    int B = (N + M - 1) / M;
    if (cfg.forceM == 0 && B > numSms) { M = (N + numSms - 1) / numSms; B = (N + M - 1) / M; }
    int threads = M * nn; if (threads > 1024) threads = 1024;
    int smemBytes = smemFloats<Mdl>(M) * (int)sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute((const void*)cudampc_admm_kernel<Mdl>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
    CUDA_CHECK(cudaMalloc(&bxb, (size_t)(B + 1) * 2 * nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bzc, (size_t)(B + 1) * 2 * nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bvc, (size_t)(B + 1) * 2 * nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&blc, (size_t)(B + 1) * 2 * nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dcp, (size_t)(B + 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dcc, (size_t)(B + 1) * sizeof(unsigned int)));

    // upload problem data
    CUDA_CHECK(cudaMemcpy(dQw, cfg.Q.data(), NX * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQfw, cfg.Qf.data(), NX * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dxref, cfg.xref.data(), NX * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(duref, cfg.uref.data(), NU * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dXmin, cfg.xmin.data(), NX * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dXmax, cfg.xmax.data(), NX * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dUmin, cfg.umin.data(), NU * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dUmax, cfg.umax.data(), NU * sizeof(float), cudaMemcpyHostToDevice));

    // state
    std::vector<float> X((N + 1) * NX, 0.f), U(N * NU, 0.f), x0 = cfg.x0;
    for (int i = 0; i < NX; ++i) X[i] = x0[i];
    std::vector<float> u_prev(NU, 0.f);
    CUDA_CHECK(cudaMemcpy(dX, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dU, U.data(), U.size() * sizeof(float), cudaMemcpyHostToDevice));

    int nLin = (N + 127) / 128;
    int nPrep = (N + 127) / 128;
    int grid = B;

    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventCreate(&e3));
    double totalSolve = 0.0, prepMs = 0.0, admmMs = 0.0, rollMs = 0.0;

    res.traj.assign(cfg.steps + 1, std::vector<float>(NX));
    for (int i = 0; i < NX; ++i) res.traj[0][i] = x0[i];

    std::mt19937 rng(seed);

    for (int s = 0; s < cfg.steps; ++s) {
        CUDA_CHECK(cudaMemcpy(dX0, x0.data(), NX * sizeof(float), cudaMemcpyHostToDevice));
        // refresh trajectory from current controls
        rollout_kernel<Mdl><<<1, 1>>>(dX0, dU, N, dX);
        CUDA_CHECK(cudaMemcpy(dX0 + NX, u_prev.data(), NU * sizeof(float), cudaMemcpyHostToDevice));  // [x0; u_prev]

        CUDA_CHECK(cudaEventRecord(e0));
        for (int j = 0; j < cfg.scp_iters; ++j) {
            CUDA_CHECK(cudaEventRecord(e1));
            prep1_kernel<Mdl><<<nLin, 128>>>(dX, dU, N, dQw, dQfw, cfg.Qu, cfg.R, dxref, duref,
                                             dAb, dBb, dEb, dQb, dqb, dRd, dHS);
            prep2_kernel<Mdl><<<nPrep, 128>>>(N, cfg.rho, dAb, dBb, dQb, dQfw, cfg.Qu, dRd, dJ, dH, dHAN);
            CUDA_CHECK(cudaEventRecord(e2));
            CUDA_CHECK(cudaMemset(dcp, 0, (B + 1) * sizeof(unsigned int)));
            CUDA_CHECK(cudaMemset(dcc, 0, (B + 1) * sizeof(unsigned int)));
            CUDA_CHECK(cudaMemset(bzc, 0, (size_t)(B + 1) * 2 * nn * sizeof(float)));
            CUDA_CHECK(cudaMemset(bvc, 0, (size_t)(B + 1) * 2 * nn * sizeof(float)));
            CUDA_CHECK(cudaMemset(blc, 0, (size_t)(B + 1) * 2 * nn * sizeof(float)));
            int Narg = N, Marg = M, Barg = B, itersA = cfg.admm_iters;
            float rhoA = cfg.rho;
            void* args[] = {&Narg, &Marg, (void*)&rhoA, (void*)&itersA, &Barg,
                            (void*)&dX0, (void*)&dX, (void*)&dU, (void*)&dAb, (void*)&dBb, (void*)&dEb,
                            (void*)&dQb, (void*)&dqb, (void*)&dRd, (void*)&dJ, (void*)&dH, (void*)&dHAN,
                            (void*)&dHS, (void*)&dXmin, (void*)&dXmax, (void*)&dUmin, (void*)&dUmax,
                            (void*)&dUout, (void*)&dXout, (void*)&dDbg, (void*)&bxb, (void*)&bzc, (void*)&bvc, (void*)&blc,
                            (void*)&dcp, (void*)&dcc};
            CUDA_CHECK(cudaLaunchCooperativeKernel((const void*)cudampc_admm_kernel<Mdl>,
                                                   dim3(grid), dim3(threads), args, smemBytes, 0));
            CUDA_CHECK(cudaEventRecord(e3));
            CUDA_CHECK(cudaEventSynchronize(e3));
            float a = 0, b = 0, c = 0;
            CUDA_CHECK(cudaEventElapsedTime(&a, e1, e2));
            CUDA_CHECK(cudaEventElapsedTime(&b, e2, e3));
            prepMs += a; admmMs += b;
            // optimized controls -> refresh nonlinear trajectory
            CUDA_CHECK(cudaMemcpy(dU, dUout, (size_t)N * NU * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaEventRecord(e1));
            rollout_kernel<Mdl><<<1, 1>>>(dX0, dU, N, dX);
            CUDA_CHECK(cudaEventRecord(e2));
            CUDA_CHECK(cudaEventSynchronize(e2));
            CUDA_CHECK(cudaEventElapsedTime(&c, e1, e2));
            rollMs += c;
        }
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        totalSolve += ms;

        // apply first control, step the true plant
        std::vector<float> u0(NU);
        CUDA_CHECK(cudaMemcpy(u0.data(), dUout, NU * sizeof(float), cudaMemcpyDeviceToHost));
        if (s == 0) res.u0_first = u0;
        for (int i = 0; i < NU; ++i) {
            res.max_u = std::max(res.max_u, std::fabs(u0[i]));
            if (u0[i] > cfg.umax[i] + 1e-2f || u0[i] < cfg.umin[i] - 1e-2f) res.box_viol++;
            u0[i] = std::min(cfg.umax[i], std::max(cfg.umin[i], u0[i]));
        }
        {
            float xh[MAXNX], uh[MAXNX], xn[MAXNX];
            for (int i = 0; i < NX; ++i) xh[i] = x0[i];
            for (int i = 0; i < NU; ++i) uh[i] = u0[i];
            Mdl::f(xh, uh, xn);
            for (int i = 0; i < NX; ++i) x0[i] = xn[i];
        }
        u_prev = u0;
        for (int i = 0; i < NX; ++i) res.traj[s + 1][i] = x0[i];
        // shift controls for warm start
        std::vector<float> Ush((size_t)N * NU);
        CUDA_CHECK(cudaMemcpy(Ush.data(), dUout, (size_t)N * NU * sizeof(float), cudaMemcpyDeviceToHost));
        for (int k = 0; k + 1 < N; ++k) for (int i = 0; i < NU; ++i) Ush[k * NU + i] = Ush[(k + 1) * NU + i];
        CUDA_CHECK(cudaMemcpy(dU, Ush.data(), (size_t)N * NU * sizeof(float), cudaMemcpyHostToDevice));

        // collision / tracking metrics
        if (!cfg.obs.empty()) {
            int nobs = (int)cfg.obs.size() / 3;
            for (int j = 0; j < nobs; ++j) {
                float ox = cfg.obs[3 * j], oy = cfg.obs[3 * j + 1], r = cfg.obs[3 * j + 2];
                float d = std::sqrt((x0[0] - ox) * (x0[0] - ox) + (x0[1] - oy) * (x0[1] - oy)) - r;
                res.min_clear = std::min(res.min_clear, d);
                if (d < 0.f) res.collisions++;
            }
        }
    }
    res.solve_ms = totalSolve;
    res.prep_ms = prepMs; res.admm_ms = admmMs; res.rollout_ms = rollMs;
    res.dbg_res.resize(256);
    CUDA_CHECK(cudaMemcpy(res.dbg_res.data(), dDbg, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    // tracking error
    double se = 0; float fe = 0;
    for (int s = 0; s <= cfg.steps; ++s) {
        float e = 0;
        for (int i = 0; i < NX; ++i) { float d = res.traj[s][i] - cfg.xref[i]; e += d * d; }
        se += e; fe = std::sqrt(e);
    }
    res.rmse = (float)std::sqrt(se / (cfg.steps + 1));
    res.final_err = fe;
    res.reach = fe < 0.3f;

    cudaFree(dX); cudaFree(dU); cudaFree(dX0); cudaFree(dXout); cudaFree(dUout);
    cudaFree(dAb); cudaFree(dBb); cudaFree(dEb); cudaFree(dQb); cudaFree(dqb); cudaFree(dRd);
    cudaFree(dJ); cudaFree(dH); cudaFree(dHAN); cudaFree(dHS);
    cudaFree(dQw); cudaFree(dQfw); cudaFree(dxref); cudaFree(duref);
    cudaFree(dXmin); cudaFree(dXmax); cudaFree(dUmin); cudaFree(dUmax);
    cudaFree(bxb); cudaFree(bzc); cudaFree(bvc); cudaFree(blc); cudaFree(dcp); cudaFree(dcc); cudaFree(dDbg);
    return res;
}

}  // namespace cudabot

using namespace cudabot;

// ============================== GIF rendering ==============================
// Portable AVI->GIF (the shared avi_to_gif uses a Unix stderr redirect that
// cmd.exe cannot honour, so use -loglevel error instead).
static void gifFromAvi(const std::string& avi, const std::string& gif, int fps, int scale_w) {
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -loglevel error -i %s "
                  "-vf \"fps=%d,scale=%d:-1:flags=lanczos,"
                  "split[a][b];[a]palettegen=stats_mode=diff[p];"
                  "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle\" %s",
                  avi.c_str(), fps, scale_w, gif.c_str());
    if (std::system(cmd) != 0) std::fprintf(stderr, "ffmpeg failed for %s\n", gif.c_str());
    std::remove(avi.c_str());  // drop the large intermediate AVI
}
static void renderPendulum(const RunResult& r, int steps, float xref) {
    const int W = 640, H = 640, CX = 320, CY = 240; const float L = 220.f;
    if (cudabot::ensure_dirs({"tmp"}) != 0) std::fprintf(stderr, "warn: mkdir\n");
    cv::VideoWriter vw("tmp/gpu_cudampc_pendulum.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(W, H));
    for (int s = 0; s <= steps; ++s) {
        cv::Mat img(H, W, CV_8UC3, cv::Scalar(24, 24, 30));
        int tx = CX + (int)(L * sinf(xref)), ty = CY + (int)(L * cosf(xref));
        cv::line(img, {CX, CY}, {tx, ty}, cv::Scalar(80, 150, 80), 1, cv::LINE_AA);
        float th = r.traj[s][0];
        int px = CX + (int)(L * sinf(th)), py = CY + (int)(L * cosf(th));
        cv::line(img, {CX, CY}, {px, py}, cv::Scalar(200, 140, 50), 5, cv::LINE_AA);
        cv::circle(img, {CX, CY}, 9, cv::Scalar(200, 200, 200), -1, cv::LINE_AA);
        cv::circle(img, {px, py}, 13, cv::Scalar(60, 150, 250), -1, cv::LINE_AA);
        char b[128]; std::snprintf(b, sizeof(b), "CudaMPC pendulum  step %d  theta=%.3f", s, th);
        cv::putText(img, b, {24, H - 24}, cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(220, 220, 235), 1, cv::LINE_AA);
        vw.write(img);
    }
    vw.release(); gifFromAvi("tmp/gpu_cudampc_pendulum.avi", "gif/gpu_cudampc_pendulum.gif", 30, 700);
    std::printf("wrote gif/gpu_cudampc_pendulum.gif\n");
}

static void renderCartPole(const RunResult& r, int steps) {
    const int W = 800, H = 560, railY = 400; const float PX = 120.f, L = 180.f;
    if (cudabot::ensure_dirs({"tmp"}) != 0) std::fprintf(stderr, "warn: mkdir\n");
    cv::VideoWriter vw("tmp/gpu_cudampc_cartpole.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(W, H));
    for (int s = 0; s <= steps; ++s) {
        cv::Mat img(H, W, CV_8UC3, cv::Scalar(24, 24, 30));
        int rx0 = (int)(W / 2 - 2.4f * PX), rx1 = (int)(W / 2 + 2.4f * PX);
        cv::line(img, {rx0, railY}, {rx1, railY}, cv::Scalar(90, 90, 110), 3, cv::LINE_AA);
        float p = r.traj[s][0], th = r.traj[s][1];
        int cx = W / 2 + (int)(p * PX);
        cv::rectangle(img, {cx - 30, railY - 24}, {cx + 30, railY + 24}, cv::Scalar(200, 140, 50), -1, cv::LINE_AA);
        int jx = cx, jy = railY - 24;
        int px = jx + (int)(L * sinf(th)), py = jy - (int)(L * cosf(th));
        cv::line(img, {jx, jy}, {px, py}, cv::Scalar(60, 150, 250), 6, cv::LINE_AA);
        cv::circle(img, {px, py}, 12, cv::Scalar(90, 200, 90), -1, cv::LINE_AA);
        char b[128]; std::snprintf(b, sizeof(b), "CudaMPC cart-pole  step %d  p=%.2f  theta=%.3f", s, p, th);
        cv::putText(img, b, {24, H - 24}, cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(220, 220, 235), 1, cv::LINE_AA);
        vw.write(img);
    }
    vw.release(); gifFromAvi("tmp/gpu_cudampc_cartpole.avi", "gif/gpu_cudampc_cartpole.gif", 30, 800);
    std::printf("wrote gif/gpu_cudampc_cartpole.gif\n");
}

static void renderParking(const RunResult& r, int steps, const std::vector<float>& obs, float ca, float cb) {
    const int W = 800, H = 800; const float PX = 34.f;
    auto proj = [&](float x, float y, int& sx, int& sy) { sx = W / 2 + (int)(PX * x); sy = H / 2 - (int)(PX * y); };
    if (cudabot::ensure_dirs({"tmp"}) != 0) std::fprintf(stderr, "warn: mkdir\n");
    cv::VideoWriter vw("tmp/gpu_cudampc_parking.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(W, H));
    int nobs = (int)obs.size() / 3;
    for (int s = 0; s <= steps; ++s) {
        cv::Mat img(H, W, CV_8UC3, cv::Scalar(24, 24, 30));
        for (int j = 0; j < nobs; ++j) {
            int sx, sy; proj(obs[3 * j], obs[3 * j + 1], sx, sy);
            cv::circle(img, {sx, sy}, (int)(PX * obs[3 * j + 2]), cv::Scalar(60, 60, 90), -1, cv::LINE_AA);
            cv::circle(img, {sx, sy}, (int)(PX * obs[3 * j + 2]), cv::Scalar(100, 100, 140), 1, cv::LINE_AA);
        }
        int gx, gy; proj(r.goals[0], r.goals[1], gx, gy);
        cv::drawMarker(img, {gx, gy}, cv::Scalar(90, 200, 90), cv::MARKER_TILTED_CROSS, 14, 2, cv::LINE_AA);
        for (int i = 1; i <= s; ++i) {
            int ax, ay, bx, by; proj(r.traj[i - 1][0], r.traj[i - 1][1], ax, ay); proj(r.traj[i][0], r.traj[i][1], bx, by);
            cv::line(img, {ax, ay}, {bx, by}, cv::Scalar(120, 90, 50), 1, cv::LINE_AA);
        }
        float px = r.traj[s][0], py = r.traj[s][1], th = r.traj[s][2];
        float c = cosf(th), sn = sinf(th);
        cv::Point2f corners[4] = {
            {ca * c - cb * sn, ca * sn + cb * c}, {-ca * c - cb * sn, -ca * sn + cb * c},
            {-ca * c + cb * sn, -ca * sn - cb * c}, {ca * c + cb * sn, ca * sn - cb * c}};
        cv::Point pts[4]; for (int i = 0; i < 4; ++i) { int sx, sy; proj(px + corners[i].x, py + corners[i].y, sx, sy); pts[i] = {sx, sy}; }
        cv::fillConvexPoly(img, pts, 4, cv::Scalar(200, 140, 50), cv::LINE_AA);
        char b[128]; std::snprintf(b, sizeof(b), "CudaMPC car parking (OBCA)  step %d", s);
        cv::putText(img, b, {24, H - 24}, cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(220, 220, 235), 1, cv::LINE_AA);
        vw.write(img);
    }
    vw.release(); gifFromAvi("tmp/gpu_cudampc_parking.avi", "gif/gpu_cudampc_parking.gif", 30, 800);
    std::printf("wrote gif/gpu_cudampc_parking.gif\n");
}

static void report(const char* name, const RunResult& r, int NX, int steps) {
    std::printf("  RMSE=%.4f  final-err=%.4f  max|u|=%.3f  box-viol=%d  solve=%.2f ms (%.3f ms/step)\n",
                r.rmse, r.final_err, r.max_u, r.box_viol, r.solve_ms, r.solve_ms / steps);
    if (r.collisions >= 0) {
        if (r.min_clear < 1e8f) std::printf("  min clearance=%+.3f m  collisions=%d\n", r.min_clear, r.collisions);
    }
    if (r.dbg_res.size() >= 4)
        std::printf("  ADMM primal residual: |x-z|=%.2e |Bdu-v|=%.2e |z-Ax-v-e|=%.2e\n",
                    r.dbg_res[1], r.dbg_res[2], r.dbg_res[3]);
    std::printf("  breakdown: prep=%.1f ms (%.3f/step)  ADMM=%.1f ms (%.3f/step)  rollout=%.1f ms\n",
                r.prep_ms, r.prep_ms / steps, r.admm_ms, r.admm_ms / steps, r.rollout_ms);
    if (r.dbg_res.size() >= 6 && r.dbg_res[5] > 0)
        std::printf("  block0 spin fraction: %.1f%%  (spin=%.3g cyc, total=%.3g cyc)\n",
                    100.0 * r.dbg_res[4] / r.dbg_res[5], r.dbg_res[4], r.dbg_res[5]);
    std::printf("  RESULT: %s\n", (r.reach && r.collisions == 0 && r.box_viol == 0) ? "PASS" : "CHECK");
}

int main(int argc, char** argv) {
    int ovAdmm = 0, ovScp = 0;
    if (argc > 1) ovAdmm = std::atoi(argv[1]);
    if (argc > 2) ovScp = std::atoi(argv[2]);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== CudaMPC: GPU-native SCP + parallel-in-horizon ADMM ===\n");
    std::printf("Device: %s (sm_%d%d, %d SMs, optin shared/block=%zu KB)\n\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.sharedMemPerBlockOptin / 1024);

    // ---------------- Pendulum ----------------
    {
        ProblemCfg c;
        c.N = 200; c.steps = 300; c.admm_iters = 1000; c.scp_iters = 6; c.rho = 10.f;
        c.Q = {10.f, 1.f}; c.Qf = {100.f, 10.f}; c.Qu = 0.f; c.R = 0.1f;
        c.xref = {0.5f, 0.f}; c.uref = {0.f};
        c.x0 = {-1.0f, 0.f};
        c.xmin = {-1e6f, -1e6f}; c.xmax = {1e6f, 1e6f};
        c.umin = {-10.f}; c.umax = {10.f};
        if (ovAdmm) c.admm_iters = ovAdmm; if (ovScp) c.scp_iters = ovScp;
        std::printf("[Pendulum] nx=2 nu=1 N=%d dt=%.2f scp=%d admm=%d rho=%.1f\n",
                    c.N, Pendulum::DT, c.scp_iters, c.admm_iters, c.rho);
        RunResult r = runClosedLoop<Pendulum>(c, 7);        std::printf("  final state: theta=%.4f (ref %.2f) omega=%.4f\n", r.traj[c.steps][0], c.xref[0], r.traj[c.steps][1]);
        report("Pendulum", r, 2, c.steps);
        renderPendulum(r, c.steps, c.xref[0]);
        std::printf("\n");
    }

    // ---------------- Cart-Pole ----------------
    {
        ProblemCfg c;
        c.N = 200; c.steps = 500; c.admm_iters = 2000; c.scp_iters = 6; c.rho = 10.f;
        c.Q = {10.f, 60.f, 1.f, 8.f}; c.Qf = {100.f, 400.f, 10.f, 40.f}; c.Qu = 0.f; c.R = 0.1f;
        c.xref = {0.f, 0.f, 0.f, 0.f}; c.uref = {0.f, 0.f};
        c.x0 = {0.f, 0.35f, 0.f, 0.f};
        c.xmin = {-2.4f, -1e6f, -1e6f, -1e6f}; c.xmax = {2.4f, 1e6f, 1e6f, 1e6f};
        c.umin = {-30.f, -5.f}; c.umax = {30.f, 5.f};
        if (ovAdmm) c.admm_iters = ovAdmm; if (ovScp) c.scp_iters = ovScp;
        std::printf("[Cart-Pole] nx=4 nu=2 N=%d dt=%.2f scp=%d admm=%d rho=%.1f\n",
                    c.N, CartPole::DT, c.scp_iters, c.admm_iters, c.rho);
        RunResult r = runClosedLoop<CartPole>(c, 7);
        std::printf("  final state: p=%.4f th=%.4f pd=%.4f thd=%.4f\n",
                    r.traj[c.steps][0], r.traj[c.steps][1], r.traj[c.steps][2], r.traj[c.steps][3]);
        report("CartPole", r, 4, c.steps);
        renderCartPole(r, c.steps);
        std::printf("\n");
    }

    // ---------------- Car Parking (OBCA) ----------------
    {
        ProblemCfg c;
        c.N = 1000; c.steps = 250; c.admm_iters = 1200; c.scp_iters = 6; c.rho = 10.f;
        c.Q = {8.f, 8.f, 4.f, 2.f}; c.Qf = {80.f, 80.f, 40.f, 20.f}; c.Qu = 0.f; c.R = 0.5f;
        c.xref = {5.f, 4.f, 0.f, 0.f}; c.uref = {0.f, 0.f};
        c.x0 = {-6.f, -4.f, 0.f, 0.f};
        c.xmin = {-1e6f, -1e6f, -1e6f, -3.f}; c.xmax = {1e6f, 1e6f, 1e6f, 3.f};
        c.umin = {-2.f, -1.5f}; c.umax = {2.f, 1.5f};
        c.obs = {-2.f, -1.0f, 1.0f,   0.5f, 1.2f, 1.0f,   2.8f, -0.5f, 1.0f};
        if (ovAdmm) c.admm_iters = ovAdmm; if (ovScp) c.scp_iters = ovScp;
        std::printf("[Car Parking] nx=4 nu=2 N=%d dt=%.2f scp=%d admm=%d rho=%.1f  obstacles=%d\n",
                    c.N, CarParking::DT, c.scp_iters, c.admm_iters, c.rho, (int)c.obs.size() / 3);
        RunResult r = runClosedLoop<CarParking>(c, 7);
        std::printf("  final pose: px=%.3f py=%.3f th=%.3f v=%.3f  (goal %.2f,%.2f)\n",
                    r.traj[c.steps][0], r.traj[c.steps][1], r.traj[c.steps][2], r.traj[c.steps][3],
                    c.xref[0], c.xref[1]);
        report("CarParking", r, 4, c.steps);
        renderParking(r, c.steps, c.obs, 0.9f, 0.45f);
        std::printf("\n");
    }
    return 0;
}
