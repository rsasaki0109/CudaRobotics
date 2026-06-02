// gpu_filterreg.cu
//
// GPU FilterReg: probabilistic rigid point-cloud registration.
//
// The repo's registration line is ICP / GICP / NDT.  FilterReg
//   (Gao & Tedrake, "FilterReg: Robust and Efficient Probabilistic
//    Point-Set Registration using Gaussian Filter and Twist
//    Parameterization", CVPR 2019, arXiv:1811.10136)
// is the modern probabilistic registrant that the probreg package never had a
// fast GPU version of.  Its two ideas:
//
//   1. E-step as a GAUSSIAN FILTER.  Classical GMM/EM registration (CPD,
//      GMMReg) forms a dense N x M responsibility matrix -- the cost bottleneck.
//      FilterReg instead computes the EM correspondence by Gaussian FILTERING,
//      which is O(N+M) not O(NM).  Crucially this is a PROPER EM posterior, not a
//      raw mean-shift: each observation x_n is normalised by the model density it
//      sees, Z_n = sum_j K(x_n, T y_j), so mass is conserved.  Two filters/iter:
//         A) splat the transformed model {T y_j}, blur, SLICE at the observations
//            -> Z_n ; set a_n = 1/(Z_n + c).
//         B) splat the observations weighted by a_n (channels a_n and a_n x_n),
//            blur, SLICE at the model points -> W_j = sum_n a_n K, (Mu)_j = sum_n
//            a_n x_n K ; the filtered correspondence is mu_j = (Mu)_j / W_j.
//      Without the a_n normaliser (filter A) the "correspondence" is a density
//      mode-seeking mean-shift that drifts/shrinks even for identical clouds.
//      The paper uses a permutohedral lattice (best in high dim); for 3D position
//      a regular voxel grid filter -- SPLAT, separable Gaussian BLUR, trilinear
//      SLICE -- is the same idea, simpler, and very GPU-friendly.  (Splat indexes
//      cells while slice samples cell CENTRES, so the trilinear lookup is offset
//      by half a voxel to match; otherwise a systematic h/2 bias walks the
//      solution off.)  The permutohedral lattice is the natural extension for
//      point-to-plane / FPFH channels; point-to-plane would also remove the
//      residual O(sigma^2 * curvature) soft-mean bias.  Noted as future work.
//
//   2. M-step as a TWIST Gauss-Newton step.  The incremental rigid motion is a
//      twist xi in se(3), T <- exp(xi^) T.  With filtered correspondence
//      mu_j = M1(p_j)/M0(p_j), the weighted point-to-point cost
//         sum_j  M0_j || T y_j - mu_j ||^2
//      is minimised by one Gauss-Newton step: J_j = [I, -[p_j]_x] (3x6),
//      H = sum_j M0_j J_j^T J_j (6x6),  g = sum_j M0_j J_j^T r_j,  solve H d = -g,
//      T <- exp(d) T.  sigma is annealed (coarse-to-fine).
//
// This file builds up in verifiable stages, correctness first:
//   1. procedural 3D cloud + a known SE(3) perturbation (+ noise, partial overlap)
//   2. voxel Gaussian filter (splat / separable blur / slice) on the GPU
//   3. twist Gauss-Newton M-step (6x6 normal equations, SE(3) exp)
//   4. numeric verification: recover the known transform (rotation/translation err)
//   5. (next) the rendered convergence GIF
//
// Build: CMakeLists adds the target with --expt-relaxed-constexpr.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ============================ small SE(3) / SO(3) helpers (host) ============================
struct Mat3 { float m[9]; };           // row-major
struct Pose { Mat3 R; float t[3]; };

static inline void mat3_vec(const Mat3& R, const float* v, float* o) {
    o[0] = R.m[0]*v[0] + R.m[1]*v[1] + R.m[2]*v[2];
    o[1] = R.m[3]*v[0] + R.m[4]*v[1] + R.m[5]*v[2];
    o[2] = R.m[6]*v[0] + R.m[7]*v[1] + R.m[8]*v[2];
}
static inline void pose_apply(const Pose& T, const float* y, float* p) {
    mat3_vec(T.R, y, p);
    p[0] += T.t[0]; p[1] += T.t[1]; p[2] += T.t[2];
}
static inline Mat3 mat3_mul(const Mat3& A, const Mat3& B) {
    Mat3 C;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float s = 0.f;
            for (int k = 0; k < 3; ++k) s += A.m[i*3+k] * B.m[k*3+j];
            C.m[i*3+j] = s;
        }
    return C;
}
// SO(3) exponential via Rodrigues.
static inline Mat3 so3_exp(const float* w) {
    float th = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    Mat3 R;
    if (th < 1e-9f) { R = {1,0,0, 0,1,0, 0,0,1}; return R; }
    float a = w[0]/th, b = w[1]/th, c = w[2]/th;
    float s = std::sin(th), co = std::cos(th), v = 1.f - co;
    R.m[0] = a*a*v + co;   R.m[1] = a*b*v - c*s;  R.m[2] = a*c*v + b*s;
    R.m[3] = a*b*v + c*s;  R.m[4] = b*b*v + co;   R.m[5] = b*c*v - a*s;
    R.m[6] = a*c*v - b*s;  R.m[7] = b*c*v + a*s;  R.m[8] = c*c*v + co;
    return R;
}
// SE(3) exponential of twist xi = [v(0:3), w(3:6)]  ->  Pose (left increment).
static inline Pose se3_exp(const float* xi) {
    const float* v = xi; const float* w = xi + 3;
    Pose T; T.R = so3_exp(w);
    float th = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    // V = I + (1-cos)/th^2 [w]_x + (th-sin)/th^3 [w]_x^2
    float Vm[9];
    if (th < 1e-6f) { for (int i=0;i<9;++i) Vm[i] = (i%4==0)?1.f:0.f; }
    else {
        float A = (1.f - std::cos(th)) / (th*th);
        float B = (th - std::sin(th)) / (th*th*th);
        float wx[9] = {0,-w[2],w[1], w[2],0,-w[0], -w[1],w[0],0};
        float wx2[9];
        for (int i=0;i<3;++i) for (int j=0;j<3;++j){ float s=0; for(int k=0;k<3;++k) s+=wx[i*3+k]*wx[k*3+j]; wx2[i*3+j]=s; }
        for (int i=0;i<9;++i) Vm[i] = ((i%4==0)?1.f:0.f) + A*wx[i] + B*wx2[i];
    }
    T.t[0] = Vm[0]*v[0]+Vm[1]*v[1]+Vm[2]*v[2];
    T.t[1] = Vm[3]*v[0]+Vm[4]*v[1]+Vm[5]*v[2];
    T.t[2] = Vm[6]*v[0]+Vm[7]*v[1]+Vm[8]*v[2];
    return T;
}
// Compose: out = A * B  (apply B then A).
static inline Pose pose_mul(const Pose& A, const Pose& B) {
    Pose C; C.R = mat3_mul(A.R, B.R);
    float Rb_t[3]; mat3_vec(A.R, B.t, Rb_t);
    C.t[0] = Rb_t[0] + A.t[0]; C.t[1] = Rb_t[1] + A.t[1]; C.t[2] = Rb_t[2] + A.t[2];
    return C;
}

// ============================ procedural test cloud ============================
// A trefoil-knot tube: a distinctive, asymmetric 3D shape so that an unknown
// rotation is unambiguously observable (unlike a sphere/plane).
// A "lumpy" closed surface: a sphere modulated by several fixed sinusoidal lobes
// plus a few localized Gaussian bumps.  The asymmetric bumps break every rotation
// and translation symmetry, so registration is fully 6-DOF constrained (unlike a
// sphere or a 1-D tube, which admit free sliding).  Also makes a distinctive GIF.
static std::vector<float> make_lumpy(int n, unsigned seed) {
    std::vector<float> pts(n * 3);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uu(-1.f, 1.f), up(0.f, 6.2831853f);
    // a handful of fixed localized bumps (direction on unit sphere, height, width)
    const float bumps[][5] = {
        { 0.8f,  0.2f,  0.5f,  0.9f, 0.25f},
        {-0.3f,  0.9f,  0.2f,  0.7f, 0.30f},
        { 0.1f, -0.6f,  0.8f,  0.8f, 0.22f},
        {-0.7f, -0.4f, -0.5f,  1.0f, 0.28f},
        { 0.4f,  0.3f, -0.85f, 0.6f, 0.20f},
    };
    int nb = sizeof(bumps)/sizeof(bumps[0]);
    for (int i = 0; i < n; ++i) {
        // uniform direction on the sphere
        float z = uu(rng), phi = up(rng), r2 = std::sqrt(std::max(0.f, 1.f - z*z));
        float dx = r2*std::cos(phi), dy = r2*std::sin(phi), dz = z;
        // base radius with low-order lobes (smooth global asymmetry)
        float R = 2.0f + 0.35f*std::sin(3.f*phi) * (1.f - z*z)
                       + 0.30f*dz*dx + 0.20f*std::cos(2.f*phi);
        // localized bumps
        for (int b = 0; b < nb; ++b) {
            float d = dx*bumps[b][0] + dy*bumps[b][1] + dz*bumps[b][2];  // cos angle
            float ang = 1.f - d;                                         // 0 at bump center
            R += bumps[b][3] * std::exp(-ang*ang / (2.f*bumps[b][4]*bumps[b][4]));
        }
        pts[i*3+0] = R*dx; pts[i*3+1] = R*dy; pts[i*3+2] = R*dz;
    }
    return pts;
}

// ============================ GPU voxel Gaussian filter ============================
// Grid metadata (origin + inverse voxel size + dims), shared by all kernels.
struct Grid { float ox, oy, oz; float inv_h; int nx, ny, nz; };

__host__ __device__ static inline int grid_idx(const Grid& g, int ix, int iy, int iz) {
    return (iz * g.ny + iy) * g.nx + ix;
}

// Apply the current pose to the model cloud: P = R Y + t (one thread per point).
__global__ void transform_kernel(const float* __restrict__ Y, int M,
                                 const float* __restrict__ R, const float* __restrict__ t,
                                 float* __restrict__ P) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= M) return;
    float y0 = Y[j*3+0], y1 = Y[j*3+1], y2 = Y[j*3+2];
    P[j*3+0] = R[0]*y0 + R[1]*y1 + R[2]*y2 + t[0];
    P[j*3+1] = R[3]*y0 + R[4]*y1 + R[5]*y2 + t[1];
    P[j*3+2] = R[6]*y0 + R[7]*y1 + R[8]*y2 + t[2];
}

// Weighted SPLAT: scatter point i with weight A[i] into its voxel, accumulating
//   m0 += A[i],  m1 += A[i] * pos_i   (atomic).  A == nullptr means weight 1.
__global__ void splat_w_kernel(const float* __restrict__ P, const float* __restrict__ A,
                               int N, Grid g, float* __restrict__ m0, float* __restrict__ m1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x = P[i*3+0], y = P[i*3+1], z = P[i*3+2];
    int ix = (int)floorf((x - g.ox) * g.inv_h);
    int iy = (int)floorf((y - g.oy) * g.inv_h);
    int iz = (int)floorf((z - g.oz) * g.inv_h);
    if (ix < 0 || iy < 0 || iz < 0 || ix >= g.nx || iy >= g.ny || iz >= g.nz) return;
    int idx = grid_idx(g, ix, iy, iz);
    float a = A ? A[i] : 1.f;
    atomicAdd(&m0[idx], a);
    if (m1) { atomicAdd(&m1[idx*3+0], a*x); atomicAdd(&m1[idx*3+1], a*y); atomicAdd(&m1[idx*3+2], a*z); }
}

// Trilinear sample of a SCALAR grid at world points Q (no transform).
__global__ void slice_scalar_kernel(const float* __restrict__ Q, int M, Grid g,
                                    const float* __restrict__ bm0, float* __restrict__ out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= M) return;
    float px = Q[j*3+0], py = Q[j*3+1], pz = Q[j*3+2];
    float fx = (px-g.ox)*g.inv_h-0.5f, fy = (py-g.oy)*g.inv_h-0.5f, fz = (pz-g.oz)*g.inv_h-0.5f;
    int ix = (int)floorf(fx), iy = (int)floorf(fy), iz = (int)floorf(fz);
    float tx = fx-ix, ty = fy-iy, tz = fz-iz;
    float a = 0.f;
    for (int dz=0;dz<2;++dz) for (int dy=0;dy<2;++dy) for (int dx=0;dx<2;++dx) {
        int jx=ix+dx, jy=iy+dy, jz=iz+dz;
        if (jx<0||jy<0||jz<0||jx>=g.nx||jy>=g.ny||jz>=g.nz) continue;
        float w = (dx?tx:1-tx)*(dy?ty:1-ty)*(dz?tz:1-tz);
        a += w * bm0[grid_idx(g,jx,jy,jz)];
    }
    out[j] = a;
}

// Per-observation EM normalizer weight a_n = 1 / (Z_n + c_outlier).
__global__ void compute_a_kernel(const float* __restrict__ Z, int N, float c_out, float* __restrict__ A) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    A[n] = 1.f / (Z[n] + c_out);
}

// Separable Gaussian blur along one axis (axis = 0/1/2), radius R, weights w[].
// One thread per voxel; reads neighbours along the axis.  Applied to both the
// scalar field m0 and the 3-vector field m1 (comp = number of components).
__global__ void blur_axis_kernel(const float* __restrict__ in, float* __restrict__ out,
                                 Grid g, int axis, int R, const float* __restrict__ w, int comp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = g.nx * g.ny * g.nz;
    if (idx >= total) return;
    int ix = idx % g.nx;
    int iy = (idx / g.nx) % g.ny;
    int iz = idx / (g.nx * g.ny);
    for (int c = 0; c < comp; ++c) {
        float acc = 0.f;
        for (int d = -R; d <= R; ++d) {
            int jx = ix, jy = iy, jz = iz;
            if (axis == 0) jx += d; else if (axis == 1) jy += d; else jz += d;
            if (jx < 0 || jy < 0 || jz < 0 || jx >= g.nx || jy >= g.ny || jz >= g.nz) continue;
            int j = grid_idx(g, jx, jy, jz);
            acc += w[d + R] * in[j*comp + c];
        }
        out[idx*comp + c] = acc;
    }
}

// SLICE: trilinear sample of the blurred m0 / m1 grids at the transformed model
// points p_j = R y_j + t, producing per-model-point moments (M0_j, M1_j).
__global__ void slice_kernel(const float* __restrict__ Y, int M, Grid g,
                             const float* __restrict__ R, const float* __restrict__ t,
                             const float* __restrict__ bm0, const float* __restrict__ bm1,
                             float* __restrict__ outM0, float* __restrict__ outM1,
                             float* __restrict__ outP) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= M) return;
    float y0 = Y[j*3+0], y1 = Y[j*3+1], y2 = Y[j*3+2];
    float px = R[0]*y0 + R[1]*y1 + R[2]*y2 + t[0];
    float py = R[3]*y0 + R[4]*y1 + R[5]*y2 + t[1];
    float pz = R[6]*y0 + R[7]*y1 + R[8]*y2 + t[2];
    outP[j*3+0] = px; outP[j*3+1] = py; outP[j*3+2] = pz;
    float fx = (px - g.ox) * g.inv_h - 0.5f, fy = (py - g.oy) * g.inv_h - 0.5f, fz = (pz - g.oz) * g.inv_h - 0.5f;
    int ix = (int)floorf(fx), iy = (int)floorf(fy), iz = (int)floorf(fz);
    float tx = fx - ix, ty = fy - iy, tz = fz - iz;
    float a0 = 0.f, a1x = 0.f, a1y = 0.f, a1z = 0.f;
    for (int dz = 0; dz < 2; ++dz)
    for (int dy = 0; dy < 2; ++dy)
    for (int dx = 0; dx < 2; ++dx) {
        int jx = ix+dx, jy = iy+dy, jz = iz+dz;
        if (jx < 0 || jy < 0 || jz < 0 || jx >= g.nx || jy >= g.ny || jz >= g.nz) continue;
        float wx = dx ? tx : (1.f-tx), wy = dy ? ty : (1.f-ty), wz = dz ? tz : (1.f-tz);
        float wgt = wx*wy*wz;
        int idx = grid_idx(g, jx, jy, jz);
        a0  += wgt * bm0[idx];
        a1x += wgt * bm1[idx*3+0];
        a1y += wgt * bm1[idx*3+1];
        a1z += wgt * bm1[idx*3+2];
    }
    outM0[j] = a0;
    outM1[j*3+0] = a1x; outM1[j*3+1] = a1y; outM1[j*3+2] = a1z;
}

// M-STEP accumulation: per model point, form the weighted twist normal equations.
//   mu_j = M1_j / M0_j,  r_j = p_j - mu_j,  J_j = [I, -[p_j]_x]
//   H += M0_j J_j^T J_j (21 upper-tri entries),  g += M0_j J_j^T r_j (6)
// Accumulated into global H(21)/g(6)/cost(1)/wsum(1) with atomics.
__global__ void mstep_kernel(const float* __restrict__ P, const float* __restrict__ M0,
                             const float* __restrict__ M1, int M, float m0_floor,
                             float* __restrict__ Hg /* 21+6+2 */) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= M) return;
    float w = M0[j];
    if (w < m0_floor) return;                 // no support -> treat as outlier, skip
    float px = P[j*3+0], py = P[j*3+1], pz = P[j*3+2];
    float mux = M1[j*3+0]/w, muy = M1[j*3+1]/w, muz = M1[j*3+2]/w;
    float rx = px - mux, ry = py - muy, rz = pz - muz;
    // J rows (3x6): [ I | -[p]_x ],  -[p]_x = [[0,pz,-py],[-pz,0,px],[py,-px,0]]
    // columns: 0:vx 1:vy 2:vz 3:wx 4:wy 5:wz
    float J[18] = {
        1,0,0,   0,  pz, -py,
        0,1,0, -pz,   0,  px,
        0,0,1,  py, -px,   0
    };
    float Hl[21]; int c = 0;
    for (int a = 0; a < 6; ++a)
        for (int b = a; b < 6; ++b) {
            float s = J[0*6+a]*J[0*6+b] + J[1*6+a]*J[1*6+b] + J[2*6+a]*J[2*6+b];
            Hl[c++] = w * s;
        }
    float gl[6];
    for (int a = 0; a < 6; ++a)
        gl[a] = w * (J[0*6+a]*rx + J[1*6+a]*ry + J[2*6+a]*rz);
    for (int k = 0; k < 21; ++k) atomicAdd(&Hg[k], Hl[k]);
    for (int k = 0; k < 6; ++k)  atomicAdd(&Hg[21+k], gl[k]);
    atomicAdd(&Hg[27], w * (rx*rx + ry*ry + rz*rz));   // weighted cost
    atomicAdd(&Hg[28], w);                              // weight sum
}

// Solve the 6x6 SPD system H d = -g (H stored as 21 upper-tri) via Cholesky.
static bool solve6(const float* Hut, const float* g, float* d) {
    float H[36];
    int c = 0;
    for (int a = 0; a < 6; ++a)
        for (int b = a; b < 6; ++b) { H[a*6+b] = H[b*6+a] = Hut[c++]; }
    for (int i = 0; i < 6; ++i) H[i*6+i] += 1e-6f;     // tiny damping
    float L[36] = {0};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            float s = H[i*6+j];
            for (int k = 0; k < j; ++k) s -= L[i*6+k]*L[j*6+k];
            if (i == j) { if (s <= 0) return false; L[i*6+i] = std::sqrt(s); }
            else L[i*6+j] = s / L[j*6+j];
        }
    }
    float y[6];
    for (int i = 0; i < 6; ++i) { float s = -g[i]; for (int k = 0; k < i; ++k) s -= L[i*6+k]*y[k]; y[i] = s / L[i*6+i]; }
    for (int i = 5; i >= 0; --i) { float s = y[i]; for (int k = i+1; k < 6; ++k) s -= L[k*6+i]*d[k]; d[i] = s / L[i*6+i]; }
    return true;
}

// ============================ FilterReg driver ============================
struct FilterRegResult { Pose T; int iters; float final_rmse; };

// Registers source Y onto fixed X.  Builds the grid filter from X once per sigma
// level; runs Gauss-Newton twist steps, re-slicing each iteration.
static FilterRegResult filterreg(const std::vector<float>& X, const std::vector<float>& Y,
                                 Pose T0, std::vector<Pose>* traj = nullptr) {
    int N = X.size() / 3, M = Y.size() / 3;
    // bounding box of X (+margin) -> grid
    float lo[3] = {1e9f,1e9f,1e9f}, hi[3] = {-1e9f,-1e9f,-1e9f};
    for (int i = 0; i < N; ++i) for (int k = 0; k < 3; ++k) {
        lo[k] = std::min(lo[k], X[i*3+k]); hi[k] = std::max(hi[k], X[i*3+k]); }
    float margin = 2.0f;
    for (int k = 0; k < 3; ++k) { lo[k] -= margin; hi[k] += margin; }

    // device buffers for the clouds
    float *dX, *dY; CUDA_CHECK(cudaMalloc(&dX, N*3*sizeof(float))); CUDA_CHECK(cudaMalloc(&dY, M*3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, X.data(), N*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY, Y.data(), M*3*sizeof(float), cudaMemcpyHostToDevice));
    float *dR, *dt; CUDA_CHECK(cudaMalloc(&dR, 9*sizeof(float))); CUDA_CHECK(cudaMalloc(&dt, 3*sizeof(float)));
    float *dP, *dM0, *dM1; CUDA_CHECK(cudaMalloc(&dP, M*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dM0, M*sizeof(float))); CUDA_CHECK(cudaMalloc(&dM1, M*3*sizeof(float)));
    float *dZ, *dA; CUDA_CHECK(cudaMalloc(&dZ, N*sizeof(float))); CUDA_CHECK(cudaMalloc(&dA, N*sizeof(float)));
    float *dHg; CUDA_CHECK(cudaMalloc(&dHg, 29*sizeof(float)));
    float Iden[9] = {1,0,0, 0,1,0, 0,0,1}, Zero[3] = {0,0,0};
    float *dIden, *dZero; CUDA_CHECK(cudaMalloc(&dIden, 9*sizeof(float))); CUDA_CHECK(cudaMalloc(&dZero, 3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIden, Iden, 9*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dZero, Zero, 3*sizeof(float), cudaMemcpyHostToDevice));

    Pose T = T0;
    FilterRegResult res; res.iters = 0; res.final_rmse = 0;
    if (traj) traj->push_back(T);

    // Fixed FINE voxel size that resolves the structure, INDEPENDENT of sigma.
    // (Coupling h to sigma was the bug: a coarse h>=structure averages distinct
    // surface branches into one voxel and the filtered centroid is meaningless.)
    // The Gaussian weighting is applied entirely through the blur radius R = sigma/h0.
    float h0 = 0.07f;
    Grid g;
    g.ox = lo[0]; g.oy = lo[1]; g.oz = lo[2]; g.inv_h = 1.f / h0;
    g.nx = (int)std::ceil((hi[0]-lo[0])/h0) + 1;
    g.ny = (int)std::ceil((hi[1]-lo[1])/h0) + 1;
    g.nz = (int)std::ceil((hi[2]-lo[2])/h0) + 1;
    int total = g.nx * g.ny * g.nz, gb = (total+255)/256;
    // grid filter scratch (1-channel m0*, 3-channel m1*), reused every iteration
    float *m0a,*m0b,*m1a,*m1b;
    CUDA_CHECK(cudaMalloc(&m0a, total*sizeof(float)));   CUDA_CHECK(cudaMalloc(&m0b, total*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m1a, total*3*sizeof(float))); CUDA_CHECK(cudaMalloc(&m1b, total*3*sizeof(float)));

    // coarse-to-fine sigma annealing
    const float sigmas[] = {0.7f, 0.5f, 0.35f, 0.25f, 0.17f, 0.11f, 0.07f, 0.05f};
    for (float sigma : sigmas) {
        // Gaussian blur with std = sigma in voxel units, truncated at 3 std.
        float sv = sigma / h0;
        int Rk = std::max(1, (int)std::ceil(3.f * sv));
        std::vector<float> wk(2*Rk+1); float wsum = 0;
        for (int d = -Rk; d <= Rk; ++d) { float wv = std::exp(-0.5f*d*d/(sv*sv)); wk[d+Rk] = wv; wsum += wv; }
        for (auto& v : wk) v /= wsum;
        float* dW; CUDA_CHECK(cudaMalloc(&dW, wk.size()*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dW, wk.data(), wk.size()*sizeof(float), cudaMemcpyHostToDevice));

        const int iters_per_level = 8;
        for (int it = 0; it < iters_per_level; ++it) {
            CUDA_CHECK(cudaMemcpy(dR, T.R.m, 9*sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dt, T.t, 3*sizeof(float), cudaMemcpyHostToDevice));
            // p_j = T y_j
            transform_kernel<<<(M+255)/256,256>>>(dY, M, dR, dt, dP);

            // --- E-step filter A: model density Z_n = sum_j K(x_n, p_j) ---
            // (the per-observation normaliser that makes this proper EM, not a
            // density-mode mean-shift -- the fix for the migration/shrink bug)
            CUDA_CHECK(cudaMemset(m0a, 0, total*sizeof(float)));
            splat_w_kernel<<<(M+255)/256,256>>>(dP, nullptr, M, g, m0a, nullptr);
            blur_axis_kernel<<<gb,256>>>(m0a, m0b, g, 0, Rk, dW, 1);
            blur_axis_kernel<<<gb,256>>>(m0b, m0a, g, 1, Rk, dW, 1);
            blur_axis_kernel<<<gb,256>>>(m0a, m0b, g, 2, Rk, dW, 1);
            slice_scalar_kernel<<<(N+255)/256,256>>>(dX, N, g, m0b, dZ);
            // a_n = 1/(Z_n + c) ; c set to a small fraction of the mean density
            float meanZ; {
                std::vector<float> hZ(N); CUDA_CHECK(cudaMemcpy(hZ.data(), dZ, N*sizeof(float), cudaMemcpyDeviceToHost));
                double s = 0; for (float z : hZ) s += z; meanZ = (float)(s / N);
            }
            compute_a_kernel<<<(N+255)/256,256>>>(dZ, N, 0.1f*meanZ + 1e-9f, dA);

            // --- E-step filter B: normalised correspondence at model points ---
            //   W_j = sum_n a_n K(p_j,x_n),  (Mu)_j = sum_n a_n x_n K(p_j,x_n)
            CUDA_CHECK(cudaMemset(m0a, 0, total*sizeof(float)));
            CUDA_CHECK(cudaMemset(m1a, 0, total*3*sizeof(float)));
            splat_w_kernel<<<(N+255)/256,256>>>(dX, dA, N, g, m0a, m1a);
            blur_axis_kernel<<<gb,256>>>(m0a, m0b, g, 0, Rk, dW, 1);
            blur_axis_kernel<<<gb,256>>>(m0b, m0a, g, 1, Rk, dW, 1);
            blur_axis_kernel<<<gb,256>>>(m0a, m0b, g, 2, Rk, dW, 1);
            blur_axis_kernel<<<gb,256>>>(m1a, m1b, g, 0, Rk, dW, 3);
            blur_axis_kernel<<<gb,256>>>(m1b, m1a, g, 1, Rk, dW, 3);
            blur_axis_kernel<<<gb,256>>>(m1a, m1b, g, 2, Rk, dW, 3);
            // slice at the (already transformed) model points -> dM0=W_j, dM1=(Mu)_j
            slice_kernel<<<(M+255)/256,256>>>(dP, M, g, dIden, dZero, m0b, m1b, dM0, dM1, dP);

            // --- M-step: weighted twist Gauss-Newton (mu = M1/M0, weight = M0) ---
            CUDA_CHECK(cudaMemset(dHg, 0, 29*sizeof(float)));
            mstep_kernel<<<(M+255)/256,256>>>(dP, dM0, dM1, M, 1e-12f, dHg);
            float Hg[29]; CUDA_CHECK(cudaMemcpy(Hg, dHg, 29*sizeof(float), cudaMemcpyDeviceToHost));
            float wsum2 = Hg[28];
            res.final_rmse = wsum2 > 0 ? std::sqrt(Hg[27]/wsum2) : 0.f;
            float d[6];
            if (!solve6(Hg, Hg+21, d)) break;
            Pose dT = se3_exp(d);
            T = pose_mul(dT, T);
            ++res.iters;
            if (traj) traj->push_back(T);
            float step = 0; for (int k=0;k<6;++k) step += d[k]*d[k];
            if (std::getenv("FR_DBG"))
                std::printf("[dbg] s=%.2f it=%2d rmse=%.4f |d|=%.4f d=(% .3f % .3f % .3f|% .3f % .3f % .3f)\n",
                    sigma, it, res.final_rmse, std::sqrt(step), d[0],d[1],d[2],d[3],d[4],d[5]);
            if (std::sqrt(step) < 1e-5f) break;
        }
        cudaFree(dW);
    }
    cudaFree(m0a); cudaFree(m0b); cudaFree(m1a); cudaFree(m1b);
    cudaFree(dZ); cudaFree(dA); cudaFree(dIden); cudaFree(dZero);
    res.T = T;
    cudaFree(dX); cudaFree(dY); cudaFree(dR); cudaFree(dt);
    cudaFree(dP); cudaFree(dM0); cudaFree(dM1); cudaFree(dHg);
    return res;
}

// ============================ convergence GIF ============================
// Orbiting 3D view of the source cloud (orange) locking onto the fixed cloud
// (cyan) as the pose trajectory plays back.  Painter's-algorithm depth sorting
// with depth-cued brightness gives a clean 3D look.
static void render_gif(const std::vector<float>& X, const std::vector<float>& Y,
                       const std::vector<Pose>& traj) {
    const int W = 1280, H = 720;
    const int CX = 380, CY = 360;           // 3D viewport centre
    const float SCALE = 78.f;
    // subsample for clean, fast rendering
    auto sub = [](const std::vector<float>& P, int stride) {
        std::vector<float> q; for (size_t i = 0; i < P.size()/3; i += stride) {
            q.push_back(P[i*3]); q.push_back(P[i*3+1]); q.push_back(P[i*3+2]); } return q; };
    std::vector<float> Xs = sub(X, 4), Ys = sub(Y, 4);

    if (system("mkdir -p tmp") != 0) std::fprintf(stderr, "warning: mkdir tmp failed\n");
    cv::VideoWriter video("tmp/gpu_filterreg.avi", cv::VideoWriter::fourcc('M','J','P','G'), 20,
                          cv::Size(W, H));
    const float elev = 0.42f;
    int ntraj = (int)traj.size();
    const int HOLD = 26;
    int nframes = ntraj + HOLD;
    struct Splat { float sx, sy, depth; cv::Scalar col; };
    for (int f = 0; f < nframes; ++f) {
        int k = std::min(f, ntraj - 1);
        float az = 0.6f + f * 0.018f;       // slow orbit
        cv::Mat img(H, W, CV_8UC3, cv::Scalar(26, 26, 32));
        const Pose& T = traj[k];
        float ca = std::cos(az), sa = std::sin(az), ce = std::cos(elev), se = std::sin(elev);
        auto project = [&](float x, float y, float z, float& sx, float& sy, float& depth) {
            float x1 = x*ca - y*sa, y1 = x*sa + y*ca, z1 = z;
            sx = CX + SCALE * x1;
            sy = CY - SCALE * (z1*ce - y1*se);
            depth = y1*ce + z1*se;
        };
        std::vector<Splat> sp; sp.reserve(Xs.size()/3 + Ys.size()/3);
        for (size_t i = 0; i < Xs.size()/3; ++i) {
            Splat s; project(Xs[i*3], Xs[i*3+1], Xs[i*3+2], s.sx, s.sy, s.depth);
            s.col = cv::Scalar(210, 180, 60); sp.push_back(s);     // fixed: cyan-ish (BGR)
        }
        for (size_t i = 0; i < Ys.size()/3; ++i) {
            float y0[3] = {Ys[i*3], Ys[i*3+1], Ys[i*3+2]}, p[3]; pose_apply(T, y0, p);
            Splat s; project(p[0], p[1], p[2], s.sx, s.sy, s.depth);
            s.col = cv::Scalar(40, 130, 240); sp.push_back(s);     // source: orange (BGR)
        }
        std::sort(sp.begin(), sp.end(), [](const Splat&a, const Splat&b){ return a.depth < b.depth; });
        float dmin=1e9f, dmax=-1e9f; for (auto&s:sp){ dmin=std::min(dmin,s.depth); dmax=std::max(dmax,s.depth);}
        for (auto& s : sp) {
            float t = (s.depth - dmin) / (dmax - dmin + 1e-6f);    // 0 far .. 1 near
            float b = 0.45f + 0.55f*t;
            cv::circle(img, cv::Point((int)s.sx, (int)s.sy), 2, s.col*b, -1, cv::LINE_AA);
        }
        // info panel
        int px = 800, py = 70;
        auto put = [&](const std::string& s, int yy, double sc, cv::Scalar c, int th){
            cv::putText(img, s, cv::Point(px, yy), cv::FONT_HERSHEY_SIMPLEX, sc, c, th, cv::LINE_AA); };
        put("GPU FilterReg", py, 1.0, cv::Scalar(235,235,245), 2); py += 38;
        put("probabilistic registration", py, 0.62, cv::Scalar(180,180,200), 1); py += 50;
        cv::circle(img, cv::Point(px+8, py-6), 6, cv::Scalar(210,180,60), -1);
        cv::putText(img, "fixed cloud", cv::Point(px+26, py), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200,200,210), 1, cv::LINE_AA); py += 30;
        cv::circle(img, cv::Point(px+8, py-6), 6, cv::Scalar(40,130,240), -1);
        cv::putText(img, "source (aligning)", cv::Point(px+26, py), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200,200,210), 1, cv::LINE_AA); py += 52;
        char buf[128];
        std::snprintf(buf, sizeof(buf), "iteration %d / %d", k, ntraj-1);
        put(buf, py, 0.62, cv::Scalar(210,210,225), 1); py += 40;
        put("E-step: Gaussian filter (O(N+M))", py, 0.5, cv::Scalar(150,200,150), 1); py += 26;
        put("M-step: SE(3) twist Gauss-Newton", py, 0.5, cv::Scalar(150,200,150), 1); py += 26;
        put("coarse-to-fine sigma annealing", py, 0.5, cv::Scalar(150,200,150), 1); py += 44;
        if (f >= nframes - HOLD) put("ALIGNED", py, 0.8, cv::Scalar(120,230,250), 2);
        video.write(img);
    }
    video.release();
    avi_to_gif("tmp/gpu_filterreg.avi", "gif/gpu_filterreg.gif", 20, 900);
    std::printf("wrote gif/gpu_filterreg.gif\n");
}

}  // namespace cudabot

// ============================ verification main ============================
int main() {
    using namespace cudabot;
    std::printf("=== GPU FilterReg: probabilistic rigid registration (verification) ===\n");

    const int N = 12000;
    std::vector<float> X = make_lumpy(N, 1);            // fixed cloud

    // build a source cloud = X under a KNOWN rigid transform + noise + partial overlap
    std::mt19937 rng(7);
    const char* envp = std::getenv("FR_EASY");
    bool easy = envp && envp[0] == '1';
    float gt_w[3] = {0.25f, -0.35f, 0.20f};               // ground-truth rotation (rad)
    float gt_t[3] = {0.7f, -0.5f, 0.4f};                  // ground-truth translation
    if (easy) { gt_w[0]=0.05f; gt_w[1]=-0.04f; gt_w[2]=0.03f; gt_t[0]=0.15f; gt_t[1]=-0.1f; gt_t[2]=0.08f; }
    Mat3 Rgt = so3_exp(gt_w);
    Pose Tgt; Tgt.R = Rgt; for (int k=0;k<3;++k) Tgt.t[k]=gt_t[k];
    bool zero = std::getenv("FR_ZERO") && std::getenv("FR_ZERO")[0]=='1';
    if (zero) { for (int k=0;k<3;++k){gt_w[k]=0;gt_t[k]=0;} Rgt=so3_exp(gt_w); Tgt.R=Rgt; for(int k=0;k<3;++k)Tgt.t[k]=0; }
    std::normal_distribution<float> noise(0.f, zero?0.f:0.02f);
    std::uniform_real_distribution<float> keep(0.f, 1.f);
    std::vector<float> Y;
    for (int i = 0; i < N; ++i) {
        if (!zero && keep(rng) > 0.85f) continue;         // drop 15% (partial overlap)
        float y[3] = {X[i*3+0], X[i*3+1], X[i*3+2]}, p[3];
        pose_apply(Tgt, y, p);
        Y.push_back(p[0]+noise(rng)); Y.push_back(p[1]+noise(rng)); Y.push_back(p[2]+noise(rng));
    }
    int M = Y.size()/3;
    std::printf("fixed N=%d  source M=%d  (15%% dropped, sigma noise 0.02)\n", N, M);
    std::printf("ground-truth  rot=(% .3f % .3f % .3f) rad   trans=(% .3f % .3f % .3f)\n",
                gt_w[0],gt_w[1],gt_w[2], gt_t[0],gt_t[1],gt_t[2]);

    // we recover the transform that maps SOURCE onto FIXED, i.e. inverse of Tgt.
    Pose T0; T0.R = {1,0,0, 0,1,0, 0,0,1}; T0.t[0]=T0.t[1]=T0.t[2]=0;
    std::vector<Pose> traj;
    auto t0 = std::chrono::high_resolution_clock::now();
    FilterRegResult res = filterreg(X, Y, T0, &traj);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // expected recovered transform = Tgt^{-1}: R_exp = Rgt^T, t_exp = -Rgt^T t
    Mat3 RgtT; for (int i=0;i<3;++i) for (int j=0;j<3;++j) RgtT.m[i*3+j]=Rgt.m[j*3+i];
    float texp[3]; mat3_vec(RgtT, gt_t, texp); for (int k=0;k<3;++k) texp[k]=-texp[k];
    // rotation error = angle of R_exp^T R_result
    Mat3 Re = mat3_mul(RgtT, res.T.R);   // should be ~ R_exp^{-1} R_res ... compute err below
    // err rotation: angle of (RgtT^T * res.R) = (Rgt * res.R)
    Mat3 Rerr = mat3_mul(Rgt, res.T.R);
    float tr = Rerr.m[0]+Rerr.m[4]+Rerr.m[8];
    float ang = std::acos(std::min(1.f, std::max(-1.f, (tr-1.f)*0.5f)));
    float terr = 0; for (int k=0;k<3;++k){ float e=res.T.t[k]-texp[k]; terr+=e*e; } terr=std::sqrt(terr);

    std::printf("recovered     rot-matrix vs Tgt^{-1}: angle err = %.4f rad (%.3f deg)\n", ang, ang*57.2958f);
    std::printf("              trans = (% .3f % .3f % .3f)  expected (% .3f % .3f % .3f)  err=%.4f\n",
                res.T.t[0],res.T.t[1],res.T.t[2], texp[0],texp[1],texp[2], terr);
    std::printf("iters=%d  final weighted RMSE=%.4f  wall=%.1f ms\n", res.iters, res.final_rmse, ms);
    if (ang < 0.02f && terr < 0.05f)
        std::printf("RESULT: PASS -- FilterReg recovered the known transform.\n");
    else
        std::printf("RESULT: CHECK -- transform not recovered within tolerance.\n");

    if (!zero && !easy) render_gif(X, Y, traj);
    return 0;
}
