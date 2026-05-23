// gpu_gicp_3d.cu
//
// GPU 3D GICP (Generalized ICP) point cloud registration.
//
// Per-point local covariance is estimated from the k-nearest neighbours in
// the same cloud, eigendecomposed, and regularised to a *disk* shape:
//   C_reg = I - (1 - eps) * n n^T
// where n is the eigenvector of the smallest eigenvalue (i.e. the surface
// normal).  This yields effective point-to-plane behaviour: displacement
// along the normal is penalised by ~1/eps, displacement along the tangent
// plane by ~1.
//
// Matching: for each source point find the nearest target point by brute
// force.  Per match the weight matrix is
//   M = (C_t + R C_s R^T)^{-1}                  (3x3 SPD)
// and the residual r = R p_s + t - p_t.  We accumulate
//   H = sum J^T M J,  b = sum J^T M r
// with J = [ I_3 | -R hat(p_s) ] (3x6) using atomicAdd, then take a
// Levenberg-Marquardt damped Gauss-Newton step on SE(3) via 6x6 Cholesky.
//
// State update uses the right-perturbation convention:
//   t_new = t + delta_t,   R_new = R * Exp(delta_w)
//
// 3x3 eigendecomposition uses the classical Cardano / trig formulation;
// the smallest eigenvector is recovered as the column-of-largest-norm of
// (A - lambda_mid I)(A - lambda_max I).
//
// Demo: same room as gpu_ndt_3d.cu but downsampled to ~2.5 k points so
// brute-force NN search is tractable.  60 random pose perturbations.

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr float X_HALF   = 10.0f;
constexpr float Y_HALF   = 10.0f;
constexpr float Z_LO     = 0.0f;
constexpr float Z_HI     = 6.0f;

constexpr int   N_POINTS   = 2500;
constexpr int   K_NEIGHBORS = 15;
constexpr float GICP_EPS   = 1e-2f;
constexpr float MATCH_MAX_D2 = 1.5f * 1.5f;
constexpr int   N_FRAMES   = 60;
constexpr int   GN_ITERS   = 20;
constexpr int   PANEL_W    = 720;
constexpr int   PANEL_H    = 560;

// -------------------------------------------------------------------------
// Cloud generation (compact version of the NDT 3D scene).
// -------------------------------------------------------------------------
struct Box { float xmin, ymin, zmin, xmax, ymax, zmax; };

static void sample_plane(int n, float ux, float uy, float uz,
                         float vx, float vy, float vz,
                         float ox, float oy, float oz,
                         std::vector<float>& out, std::mt19937& rng) {
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    for (int i = 0; i < n; i++) {
        float s = uni(rng), t = uni(rng);
        out.push_back(ox + s * ux + t * vx);
        out.push_back(oy + s * uy + t * vy);
        out.push_back(oz + s * uz + t * vz);
    }
}

static void make_cloud(std::vector<float>& out, std::mt19937& rng) {
    out.clear();
    sample_plane(340, 2 * X_HALF, 0, 0,  0, 2 * Y_HALF, 0,
                 -X_HALF, -Y_HALF, 0.0f, out, rng);        // floor
    sample_plane(220, 2 * X_HALF, 0, 0,  0, 2 * Y_HALF, 0,
                 -X_HALF, -Y_HALF, Z_HI, out, rng);        // ceiling
    sample_plane(330, 2 * X_HALF, 0, 0,  0, 0, Z_HI,
                 -X_HALF, -Y_HALF, 0.0f, out, rng);
    sample_plane(330, 2 * X_HALF, 0, 0,  0, 0, Z_HI,
                 -X_HALF,  Y_HALF, 0.0f, out, rng);
    sample_plane(330, 0, 2 * Y_HALF, 0, 0, 0, Z_HI,
                 -X_HALF, -Y_HALF, 0.0f, out, rng);
    sample_plane(330, 0, 2 * Y_HALF, 0, 0, 0, Z_HI,
                  X_HALF, -Y_HALF, 0.0f, out, rng);
    std::vector<Box> boxes = {
        {-4.0f, -3.0f, 0.0f, -1.5f, 0.0f, 2.0f},
        { 2.0f,  1.0f, 0.0f,  4.5f, 4.0f, 2.8f},
        {-2.0f,  3.0f, 0.0f,  1.0f, 6.0f, 1.5f},
    };
    for (const auto& b : boxes) {
        float dx = b.xmax - b.xmin;
        float dy = b.ymax - b.ymin;
        float dz = b.zmax - b.zmin;
        sample_plane(45, dx, 0, 0, 0, dy, 0, b.xmin, b.ymin, b.zmax, out, rng);
        sample_plane(30, dx, 0, 0, 0, 0, dz, b.xmin, b.ymin, b.zmin, out, rng);
        sample_plane(30, dx, 0, 0, 0, 0, dz, b.xmin, b.ymax, b.zmin, out, rng);
        sample_plane(30, 0, dy, 0, 0, 0, dz, b.xmin, b.ymin, b.zmin, out, rng);
        sample_plane(30, 0, dy, 0, 0, 0, dz, b.xmax, b.ymin, b.zmin, out, rng);
    }
}

static void add_noise(std::vector<float>& cloud, float sigma, std::mt19937& rng) {
    std::normal_distribution<float> n(0.0f, sigma);
    for (size_t i = 0; i < cloud.size(); i++) cloud[i] += n(rng);
}

static void mat3_mul(const float* A, const float* B, float* C) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            float v = 0.0f;
            for (int k = 0; k < 3; k++) v += A[3*r + k] * B[3*k + c];
            C[3*r + c] = v;
        }
}

static void so3_exp(const float* w, float* R) {
    float theta = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    if (theta < 1e-9f) {
        R[0] = 1.0f; R[1] = -w[2]; R[2] = w[1];
        R[3] = w[2]; R[4] = 1.0f; R[5] = -w[0];
        R[6] = -w[1]; R[7] = w[0]; R[8] = 1.0f;
        return;
    }
    float s = std::sin(theta) / theta;
    float c = (1.0f - std::cos(theta)) / (theta * theta);
    float K[9] = {
        0.0f, -w[2],  w[1],
        w[2],  0.0f, -w[0],
       -w[1],  w[0],  0.0f
    };
    float K2[9];
    for (int r = 0; r < 3; r++)
        for (int cc = 0; cc < 3; cc++) {
            float v = 0.0f;
            for (int kk = 0; kk < 3; kk++) v += K[3 * r + kk] * K[3 * kk + cc];
            K2[3 * r + cc] = v;
        }
    R[0] = 1.0f; R[1] = 0.0f; R[2] = 0.0f;
    R[3] = 0.0f; R[4] = 1.0f; R[5] = 0.0f;
    R[6] = 0.0f; R[7] = 0.0f; R[8] = 1.0f;
    for (int k = 0; k < 9; k++) R[k] += s * K[k] + c * K2[k];
}

static void apply_transform(const std::vector<float>& in, const float* R,
                            const float* t, std::vector<float>& out) {
    out.resize(in.size());
    int n = (int)in.size() / 3;
    for (int i = 0; i < n; i++) {
        float p[3] = { in[3*i+0], in[3*i+1], in[3*i+2] };
        out[3*i+0] = R[0]*p[0] + R[1]*p[1] + R[2]*p[2] + t[0];
        out[3*i+1] = R[3]*p[0] + R[4]*p[1] + R[5]*p[2] + t[1];
        out[3*i+2] = R[6]*p[0] + R[7]*p[1] + R[8]*p[2] + t[2];
    }
}

// -------------------------------------------------------------------------
// 3x3 symmetric eigendecomposition (Cardano) — return smallest eigenvector.
// Input C stored as 6 floats: c00, c01, c02, c11, c12, c22.
// -------------------------------------------------------------------------
__device__ inline void smallest_eigvec_3x3_sym(const float* C, float* n_out) {
    float c00 = C[0], c01 = C[1], c02 = C[2];
    float c11 = C[3], c12 = C[4], c22 = C[5];
    float eig[3];
    float p1 = c01*c01 + c02*c02 + c12*c12;
    if (p1 < 1e-12f) {
        eig[0] = c00; eig[1] = c11; eig[2] = c22;
    } else {
        float q = (c00 + c11 + c22) / 3.0f;
        float p2 = (c00 - q)*(c00 - q) + (c11 - q)*(c11 - q) + (c22 - q)*(c22 - q)
                  + 2.0f * p1;
        float p = sqrtf(p2 / 6.0f);
        float ip = 1.0f / p;
        float b00 = (c00 - q) * ip, b01 = c01 * ip, b02 = c02 * ip;
        float b11 = (c11 - q) * ip, b12 = c12 * ip, b22 = (c22 - q) * ip;
        float detB = b00 * (b11 * b22 - b12 * b12)
                   - b01 * (b01 * b22 - b12 * b02)
                   + b02 * (b01 * b12 - b11 * b02);
        float r = detB * 0.5f;
        if (r < -1.0f) r = -1.0f;
        if (r >  1.0f) r =  1.0f;
        float phi = acosf(r) / 3.0f;
        float e1 = q + 2.0f * p * cosf(phi);
        float e3 = q + 2.0f * p * cosf(phi + 2.0f * (float)M_PI / 3.0f);
        float e2 = 3.0f * q - e1 - e3;
        eig[0] = e1; eig[1] = e2; eig[2] = e3;
    }
    // Identify min, mid, max.
    float lmin = fminf(fminf(eig[0], eig[1]), eig[2]);
    float lmax = fmaxf(fmaxf(eig[0], eig[1]), eig[2]);
    float lmid = eig[0] + eig[1] + eig[2] - lmin - lmax;
    // Eigvec for lmin: column of (A - lmid I)(A - lmax I) with largest norm.
    float A0 = c00 - lmid, A4 = c11 - lmid, A8 = c22 - lmid;
    float B0 = c00 - lmax, B4 = c11 - lmax, B8 = c22 - lmax;
    // Full 3x3 multiplication, row-major.  A and B are symmetric so off-diag
    // entries are c01, c02, c12 unchanged.
    // (A * B)[i, j] = sum_k A[i, k] * B[k, j]
    float prod[9];
    // Helper rows of A, B
    float Ar[9] = { A0,  c01, c02,
                    c01, A4,  c12,
                    c02, c12, A8  };
    float Br[9] = { B0,  c01, c02,
                    c01, B4,  c12,
                    c02, c12, B8  };
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float s = 0.0f;
            for (int k = 0; k < 3; k++) s += Ar[3*i + k] * Br[3*k + j];
            prod[3*i + j] = s;
        }
    int best_col = 0;
    float best_norm = -1.0f;
    #pragma unroll
    for (int j = 0; j < 3; j++) {
        float nn = prod[j]*prod[j] + prod[3+j]*prod[3+j] + prod[6+j]*prod[6+j];
        if (nn > best_norm) { best_norm = nn; best_col = j; }
    }
    float nx = prod[best_col];
    float ny = prod[3 + best_col];
    float nz = prod[6 + best_col];
    float nn = sqrtf(nx*nx + ny*ny + nz*nz);
    if (nn < 1e-9f) {
        // Degenerate: fall back to z-axis.
        n_out[0] = 0.0f; n_out[1] = 0.0f; n_out[2] = 1.0f;
    } else {
        n_out[0] = nx / nn; n_out[1] = ny / nn; n_out[2] = nz / nn;
    }
}

__device__ inline float invert_sym3_dev(const float* C, float* inv) {
    float a = C[0], b = C[1], c = C[2];
    float d = C[3], e = C[4], f = C[5];
    float det = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c);
    if (fabsf(det) < 1e-12f) det = (det < 0 ? -1e-12f : 1e-12f);
    float inv_det = 1.0f / det;
    inv[0] = (d * f - e * e) * inv_det;
    inv[1] = (c * e - b * f) * inv_det;
    inv[2] = (b * e - c * d) * inv_det;
    inv[3] = (a * f - c * c) * inv_det;
    inv[4] = (b * c - a * e) * inv_det;
    inv[5] = (a * d - b * b) * inv_det;
    return det;
}

// -------------------------------------------------------------------------
// Per-point covariance kernel: brute-force k-NN, sample covariance,
// regularise to disk shape.
// Output cov stored as 6 floats per point (c00, c01, c02, c11, c12, c22).
// -------------------------------------------------------------------------
__global__ void compute_cov_kernel(int n, const float* __restrict__ pts,
                                   float* __restrict__ cov6,
                                   unsigned char* __restrict__ ok) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = pts[3*i + 0];
    float yi = pts[3*i + 1];
    float zi = pts[3*i + 2];
    // Top-K smallest squared distances.
    float bd[K_NEIGHBORS];
    int   bi[K_NEIGHBORS];
    #pragma unroll
    for (int k = 0; k < K_NEIGHBORS; k++) { bd[k] = 1e30f; bi[k] = -1; }
    float worst = 1e30f;
    int worst_pos = 0;
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        float dx = pts[3*j + 0] - xi;
        float dy = pts[3*j + 1] - yi;
        float dz = pts[3*j + 2] - zi;
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < worst) {
            bd[worst_pos] = d2;
            bi[worst_pos] = j;
            worst = bd[0]; worst_pos = 0;
            #pragma unroll
            for (int k = 1; k < K_NEIGHBORS; k++) {
                if (bd[k] > worst) { worst = bd[k]; worst_pos = k; }
            }
        }
    }
    // Mean
    float mx = xi, my = yi, mz = zi;
    int cnt = 1;
    #pragma unroll
    for (int k = 0; k < K_NEIGHBORS; k++) {
        if (bi[k] < 0) continue;
        mx += pts[3*bi[k] + 0];
        my += pts[3*bi[k] + 1];
        mz += pts[3*bi[k] + 2];
        cnt++;
    }
    if (cnt < 4) { ok[i] = 0; return; }
    float inv_n = 1.0f / (float)cnt;
    mx *= inv_n; my *= inv_n; mz *= inv_n;
    // Covariance
    float c00 = (xi - mx)*(xi - mx);
    float c11 = (yi - my)*(yi - my);
    float c22 = (zi - mz)*(zi - mz);
    float c01 = (xi - mx)*(yi - my);
    float c02 = (xi - mx)*(zi - mz);
    float c12 = (yi - my)*(zi - mz);
    #pragma unroll
    for (int k = 0; k < K_NEIGHBORS; k++) {
        if (bi[k] < 0) continue;
        float dxk = pts[3*bi[k] + 0] - mx;
        float dyk = pts[3*bi[k] + 1] - my;
        float dzk = pts[3*bi[k] + 2] - mz;
        c00 += dxk*dxk; c11 += dyk*dyk; c22 += dzk*dzk;
        c01 += dxk*dyk; c02 += dxk*dzk; c12 += dyk*dzk;
    }
    c00 *= inv_n; c11 *= inv_n; c22 *= inv_n;
    c01 *= inv_n; c02 *= inv_n; c12 *= inv_n;
    // Eigendecompose -> normal -> disk regularisation.
    float Cs[6] = { c00, c01, c02, c11, c12, c22 };
    float nv[3];
    smallest_eigvec_3x3_sym(Cs, nv);
    float w = 1.0f - GICP_EPS;
    // C_reg = I - w * n n^T
    cov6[6*i + 0] = 1.0f - w * nv[0] * nv[0];
    cov6[6*i + 1] =       - w * nv[0] * nv[1];
    cov6[6*i + 2] =       - w * nv[0] * nv[2];
    cov6[6*i + 3] = 1.0f - w * nv[1] * nv[1];
    cov6[6*i + 4] =       - w * nv[1] * nv[2];
    cov6[6*i + 5] = 1.0f - w * nv[2] * nv[2];
    ok[i] = 1;
}

// -------------------------------------------------------------------------
// Match + GN accumulator kernel.
// One thread per source point: brute-force NN in target, compute weight
// matrix M and accumulate per-correspondence H and b via atomicAdd.
// -------------------------------------------------------------------------
__global__ void gicp_accum_kernel(int n,
                                  const float* __restrict__ src_pts,
                                  const float* __restrict__ src_cov,
                                  const unsigned char* __restrict__ src_ok,
                                  const float* __restrict__ tgt_pts,
                                  const float* __restrict__ tgt_cov,
                                  const unsigned char* __restrict__ tgt_ok,
                                  const float* __restrict__ R,
                                  const float* __restrict__ t,
                                  float* g, float* H,
                                  float* cost_out, int* match_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!src_ok[i]) return;
    float ps[3] = { src_pts[3*i + 0], src_pts[3*i + 1], src_pts[3*i + 2] };
    // pw = R * ps + t
    float pw[3];
    pw[0] = R[0]*ps[0] + R[1]*ps[1] + R[2]*ps[2] + t[0];
    pw[1] = R[3]*ps[0] + R[4]*ps[1] + R[5]*ps[2] + t[1];
    pw[2] = R[6]*ps[0] + R[7]*ps[1] + R[8]*ps[2] + t[2];
    // Brute-force NN in target.
    float best_d2 = MATCH_MAX_D2;
    int   best_j = -1;
    for (int j = 0; j < n; j++) {
        if (!tgt_ok[j]) continue;
        float dx = tgt_pts[3*j + 0] - pw[0];
        float dy = tgt_pts[3*j + 1] - pw[1];
        float dz = tgt_pts[3*j + 2] - pw[2];
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best_d2) { best_d2 = d2; best_j = j; }
    }
    if (best_j < 0) return;
    // Source cov in world: A = R * C_s * R^T (3x3 sym).
    float a = src_cov[6*i + 0], b = src_cov[6*i + 1], c = src_cov[6*i + 2];
    float d = src_cov[6*i + 3], e = src_cov[6*i + 4], f = src_cov[6*i + 5];
    // RC = R * C_s (row-major 3x3, C_s is sym).  Then A = RC * R^T.
    float RC[9];
    RC[0] = R[0]*a + R[1]*b + R[2]*c;
    RC[1] = R[0]*b + R[1]*d + R[2]*e;
    RC[2] = R[0]*c + R[1]*e + R[2]*f;
    RC[3] = R[3]*a + R[4]*b + R[5]*c;
    RC[4] = R[3]*b + R[4]*d + R[5]*e;
    RC[5] = R[3]*c + R[4]*e + R[5]*f;
    RC[6] = R[6]*a + R[7]*b + R[8]*c;
    RC[7] = R[6]*b + R[7]*d + R[8]*e;
    RC[8] = R[6]*c + R[7]*e + R[8]*f;
    // A = RC * R^T: A_ij = sum_k RC[i,k] * R[j,k]
    float A00 = RC[0]*R[0] + RC[1]*R[1] + RC[2]*R[2];
    float A01 = RC[0]*R[3] + RC[1]*R[4] + RC[2]*R[5];
    float A02 = RC[0]*R[6] + RC[1]*R[7] + RC[2]*R[8];
    float A11 = RC[3]*R[3] + RC[4]*R[4] + RC[5]*R[5];
    float A12 = RC[3]*R[6] + RC[4]*R[7] + RC[5]*R[8];
    float A22 = RC[6]*R[6] + RC[7]*R[7] + RC[8]*R[8];
    // Combined cov: K = C_t + A.
    float K[6] = {
        tgt_cov[6*best_j + 0] + A00,
        tgt_cov[6*best_j + 1] + A01,
        tgt_cov[6*best_j + 2] + A02,
        tgt_cov[6*best_j + 3] + A11,
        tgt_cov[6*best_j + 4] + A12,
        tgt_cov[6*best_j + 5] + A22,
    };
    float Mi[6];
    invert_sym3_dev(K, Mi);
    float M00 = Mi[0], M01 = Mi[1], M02 = Mi[2];
    float M11 = Mi[3], M12 = Mi[4], M22 = Mi[5];
    // Residual r = pw - p_t
    float rx = pw[0] - tgt_pts[3*best_j + 0];
    float ry = pw[1] - tgt_pts[3*best_j + 1];
    float rz = pw[2] - tgt_pts[3*best_j + 2];
    // u = M r
    float ux = M00*rx + M01*ry + M02*rz;
    float uy = M01*rx + M11*ry + M12*rz;
    float uz = M02*rx + M12*ry + M22*rz;
    float cost = 0.5f * (rx*ux + ry*uy + rz*uz);
    atomicAdd(cost_out, cost);
    atomicAdd(match_count, 1);

    // J (3x6) = [ I | -R * hat(ps) ]  -- same as NDT 3D.
    // hat(ps) = | 0    -ps_z  ps_y |
    //           | ps_z  0    -ps_x |
    //           |-ps_y  ps_x  0    |
    // -R * hat(ps) -- per column.
    float Jr[9];   // 3x3 rotation block of J, row-major (rows: dr_x, dr_y, dr_z; cols: dwx, dwy, dwz)
    Jr[0] = -(R[0]*0.0f       + R[1]*ps[2]    + R[2]*(-ps[1]));
    Jr[1] = -(R[0]*(-ps[2])   + R[1]*0.0f     + R[2]*ps[0]);
    Jr[2] = -(R[0]*ps[1]      + R[1]*(-ps[0]) + R[2]*0.0f);
    Jr[3] = -(R[3]*0.0f       + R[4]*ps[2]    + R[5]*(-ps[1]));
    Jr[4] = -(R[3]*(-ps[2])   + R[4]*0.0f     + R[5]*ps[0]);
    Jr[5] = -(R[3]*ps[1]      + R[4]*(-ps[0]) + R[5]*0.0f);
    Jr[6] = -(R[6]*0.0f       + R[7]*ps[2]    + R[8]*(-ps[1]));
    Jr[7] = -(R[6]*(-ps[2])   + R[7]*0.0f     + R[8]*ps[0]);
    Jr[8] = -(R[6]*ps[1]      + R[7]*(-ps[0]) + R[8]*0.0f);

    // b = J^T M r = J^T u
    //   For p in {0,1,2}: J col p = e_p ⇒ b[p] = u[p]
    //   For p in {3,4,5}: J col p = Jr[:, p-3] ⇒ b[p] = Jr_col^T * u
    float b0 = ux, b1 = uy, b2 = uz;
    float b3 = Jr[0]*ux + Jr[3]*uy + Jr[6]*uz;
    float b4 = Jr[1]*ux + Jr[4]*uy + Jr[7]*uz;
    float b5 = Jr[2]*ux + Jr[5]*uy + Jr[8]*uz;
    atomicAdd(&g[0], b0); atomicAdd(&g[1], b1); atomicAdd(&g[2], b2);
    atomicAdd(&g[3], b3); atomicAdd(&g[4], b4); atomicAdd(&g[5], b5);

    // H = J^T M J.  Compute U = M * J (3x6): U[:, 0..2] = M, U[:, 3..5] = M * Jr.
    float MJ[9];
    MJ[0] = M00*Jr[0] + M01*Jr[3] + M02*Jr[6];
    MJ[1] = M00*Jr[1] + M01*Jr[4] + M02*Jr[7];
    MJ[2] = M00*Jr[2] + M01*Jr[5] + M02*Jr[8];
    MJ[3] = M01*Jr[0] + M11*Jr[3] + M12*Jr[6];
    MJ[4] = M01*Jr[1] + M11*Jr[4] + M12*Jr[7];
    MJ[5] = M01*Jr[2] + M11*Jr[5] + M12*Jr[8];
    MJ[6] = M02*Jr[0] + M12*Jr[3] + M22*Jr[6];
    MJ[7] = M02*Jr[1] + M12*Jr[4] + M22*Jr[7];
    MJ[8] = M02*Jr[2] + M12*Jr[5] + M22*Jr[8];
    // Upper-tri 6x6 H layout matches NDT 3D.
    float Hl[21];
    // row 0 (J col 0 = e_0): H[0, q] = U[0, q]
    Hl[0]  = M00;
    Hl[1]  = M01;
    Hl[2]  = M02;
    Hl[3]  = MJ[0];
    Hl[4]  = MJ[1];
    Hl[5]  = MJ[2];
    // row 1
    Hl[6]  = M11;
    Hl[7]  = M12;
    Hl[8]  = MJ[3];
    Hl[9]  = MJ[4];
    Hl[10] = MJ[5];
    // row 2
    Hl[11] = M22;
    Hl[12] = MJ[6];
    Hl[13] = MJ[7];
    Hl[14] = MJ[8];
    // rows 3..5: H[p, q] = Jr_col_{p-3}^T * MJ_col_{q-3}
    auto Jdot = [&](int a, int bb) {
        return Jr[0 + a]*MJ[0 + bb] + Jr[3 + a]*MJ[3 + bb] + Jr[6 + a]*MJ[6 + bb];
    };
    Hl[15] = Jdot(0, 0);
    Hl[16] = Jdot(0, 1);
    Hl[17] = Jdot(0, 2);
    Hl[18] = Jdot(1, 1);
    Hl[19] = Jdot(1, 2);
    Hl[20] = Jdot(2, 2);
    for (int k = 0; k < 21; k++) atomicAdd(&H[k], Hl[k]);
}

// -------------------------------------------------------------------------
// Host: 6x6 SPD Cholesky solve (same as gpu_ndt_3d.cu).
// -------------------------------------------------------------------------
static const int H_OFF[6][6] = {
    { 0,  1,  2,  3,  4,  5},
    { 1,  6,  7,  8,  9, 10},
    { 2,  7, 11, 12, 13, 14},
    { 3,  8, 12, 15, 16, 17},
    { 4,  9, 13, 16, 18, 19},
    { 5, 10, 14, 17, 19, 20},
};

static bool cholesky_solve_6(const float* H_packed, const float* g, float lambda,
                              float* dx) {
    float A[36];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
            A[6*i + j] = H_packed[H_OFF[i][j]];
            if (i == j) A[6*i + j] += lambda;
        }
    float L[36] = {0};
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j <= i; j++) {
            float s = A[6*i + j];
            for (int k = 0; k < j; k++) s -= L[6*i + k] * L[6*j + k];
            if (i == j) {
                if (s <= 0.0f) return false;
                L[6*i + j] = std::sqrt(s);
            } else {
                L[6*i + j] = s / L[6*j + j];
            }
        }
    }
    float y[6];
    for (int i = 0; i < 6; i++) {
        float s = g[i];
        for (int k = 0; k < i; k++) s -= L[6*i + k] * y[k];
        y[i] = s / L[6*i + i];
    }
    for (int i = 5; i >= 0; i--) {
        float s = y[i];
        for (int k = i + 1; k < 6; k++) s -= L[6*k + i] * dx[k];
        dx[i] = s / L[6*i + i];
    }
    return true;
}

// -------------------------------------------------------------------------
// Visualisation: same simple projection as NDT 3D.
// -------------------------------------------------------------------------
struct Cam { float yaw, pitch, dist; };
static cv::Point2i project(float x, float y, float z, const Cam& c, int W, int H) {
    float cy = std::cos(c.yaw), sy = std::sin(c.yaw);
    float cp = std::cos(c.pitch), sp = std::sin(c.pitch);
    float x1 =  cy * x + sy * y;
    float y1 = -sy * x + cy * y;
    float z1 =  z - 1.5f;
    float y2 =  cp * y1 - sp * z1;
    float z2 =  sp * y1 + cp * z1;
    float xc = x1, yc = z2;
    float zc = c.dist - y2;
    if (zc < 0.1f) zc = 0.1f;
    float f = 1.0f * H;
    return cv::Point2i(static_cast<int>(W * 0.5f + f * xc / zc),
                       static_cast<int>(H * 0.6f - f * yc / zc));
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::mt19937 rng(123);
    std::vector<float> map_cloud_h;
    make_cloud(map_cloud_h, rng);
    int n_total = (int)map_cloud_h.size() / 3;
    if (n_total > N_POINTS) {
        map_cloud_h.resize(N_POINTS * 3);
        n_total = N_POINTS;
    }
    add_noise(map_cloud_h, 0.015f, rng);
    int n = n_total;
    std::printf("Cloud: %d points\n", n);

    // Device buffers.
    float *d_map = nullptr, *d_live = nullptr;
    float *d_map_cov = nullptr, *d_live_cov = nullptr;
    unsigned char *d_map_ok = nullptr, *d_live_ok = nullptr;
    CUDA_CHECK(cudaMalloc(&d_map,      n * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_live,     n * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_map_cov,  n * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_live_cov, n * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_map_ok,   n));
    CUDA_CHECK(cudaMalloc(&d_live_ok,  n));

    float *d_g, *d_H, *d_cost, *d_R, *d_t;
    int *d_match;
    CUDA_CHECK(cudaMalloc(&d_g, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H, 21 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cost, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_match, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_R, 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_t, 3 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_map, map_cloud_h.data(), n * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));

    int blk = 256;
    int blocks_pts = (n + blk - 1) / blk;

    // Map covariance once (target stays fixed).
    compute_cov_kernel<<<blocks_pts, blk>>>(n, d_map, d_map_cov, d_map_ok);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_gicp_3d.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          10, cv::Size(PANEL_W, PANEL_H + 60));

    std::uniform_real_distribution<float> per_xy(-1.0f, 1.0f);
    std::uniform_real_distribution<float> per_z(-0.4f, 0.4f);
    std::uniform_real_distribution<float> per_w(-0.20f, 0.20f);

    double total_t_err = 0.0, total_w_err = 0.0, total_ms = 0.0;
    int counted = 0;

    for (int frame = 0; frame < N_FRAMES; frame++) {
        float gt_t[3] = { per_xy(rng), per_xy(rng), per_z(rng) };
        float gt_w[3] = { per_w(rng), per_w(rng), per_w(rng) };
        float gt_R[9];
        so3_exp(gt_w, gt_R);
        std::vector<float> live_cloud_h;
        apply_transform(map_cloud_h, gt_R, gt_t, live_cloud_h);
        add_noise(live_cloud_h, 0.015f, rng);
        // Estimator target: maps live back to map = (gt_R^T, -gt_R^T gt_t).
        float Rgt_inv[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                Rgt_inv[3*r + c] = gt_R[3*c + r];
        float tgt_inv[3];
        tgt_inv[0] = -(Rgt_inv[0]*gt_t[0] + Rgt_inv[1]*gt_t[1] + Rgt_inv[2]*gt_t[2]);
        tgt_inv[1] = -(Rgt_inv[3]*gt_t[0] + Rgt_inv[4]*gt_t[1] + Rgt_inv[5]*gt_t[2]);
        tgt_inv[2] = -(Rgt_inv[6]*gt_t[0] + Rgt_inv[7]*gt_t[1] + Rgt_inv[8]*gt_t[2]);

        CUDA_CHECK(cudaMemcpy(d_live, live_cloud_h.data(), n * 3 * sizeof(float),
                              cudaMemcpyHostToDevice));
        compute_cov_kernel<<<blocks_pts, blk>>>(n, d_live, d_live_cov, d_live_ok);

        float R_est[9] = {1,0,0, 0,1,0, 0,0,1};
        float t_est[3] = {0,0,0};
        float lambda = 1e-2f;
        float prev_cost = 1e30f;

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int it = 0; it < GN_ITERS; it++) {
            CUDA_CHECK(cudaMemset(d_g, 0, 6 * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_H, 0, 21 * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_cost, 0, sizeof(float)));
            CUDA_CHECK(cudaMemset(d_match, 0, sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_R, R_est, 9 * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_t, t_est, 3 * sizeof(float), cudaMemcpyHostToDevice));
            gicp_accum_kernel<<<blocks_pts, blk>>>(n,
                                                    d_live, d_live_cov, d_live_ok,
                                                    d_map,  d_map_cov,  d_map_ok,
                                                    d_R, d_t,
                                                    d_g, d_H, d_cost, d_match);
            float g_h[6], H_h[21], cost_h = 0.0f;
            int   match_h = 0;
            CUDA_CHECK(cudaMemcpy(g_h, d_g, 6 * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(H_h, d_H, 21 * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&cost_h, d_cost, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&match_h, d_match, sizeof(int), cudaMemcpyDeviceToHost));
            if (match_h < 30) break;
            if (cost_h < prev_cost) lambda = fmaxf(lambda * 0.5f, 1e-5f);
            else                    lambda = fminf(lambda * 4.0f, 1e4f);
            prev_cost = cost_h;
            float dx[6] = {0};
            if (!cholesky_solve_6(H_h, g_h, lambda, dx)) {
                lambda = fminf(lambda * 4.0f, 1e4f);
                continue;
            }
            float delta_t[3] = { -dx[0], -dx[1], -dx[2] };
            float delta_w[3] = { -dx[3], -dx[4], -dx[5] };
            float nt = std::sqrt(delta_t[0]*delta_t[0] + delta_t[1]*delta_t[1] + delta_t[2]*delta_t[2]);
            if (nt > 0.4f) { float k = 0.4f / nt; for (int q=0;q<3;q++) delta_t[q] *= k; }
            float nw = std::sqrt(delta_w[0]*delta_w[0] + delta_w[1]*delta_w[1] + delta_w[2]*delta_w[2]);
            if (nw > 0.15f) { float k = 0.15f / nw; for (int q=0;q<3;q++) delta_w[q] *= k; }
            t_est[0] += delta_t[0]; t_est[1] += delta_t[1]; t_est[2] += delta_t[2];
            float E[9]; so3_exp(delta_w, E);
            float Rn[9]; mat3_mul(R_est, E, Rn);
            for (int q = 0; q < 9; q++) R_est[q] = Rn[q];
        }
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);

        float err_t = std::sqrt((t_est[0]-tgt_inv[0])*(t_est[0]-tgt_inv[0])
                              + (t_est[1]-tgt_inv[1])*(t_est[1]-tgt_inv[1])
                              + (t_est[2]-tgt_inv[2])*(t_est[2]-tgt_inv[2]));
        float Rd[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
                float v = 0.0f;
                for (int k = 0; k < 3; k++) v += R_est[3*k + r] * Rgt_inv[3*k + c];
                Rd[3*r + c] = v;
            }
        float tr = Rd[0] + Rd[4] + Rd[8];
        float ct = 0.5f * (tr - 1.0f);
        if (ct >  1.0f) ct =  1.0f;
        if (ct < -1.0f) ct = -1.0f;
        float w_err = std::acos(ct);
        total_t_err += err_t; total_w_err += w_err; total_ms += ms; counted++;
        if (frame < 6 || frame % 10 == 0)
            std::printf("frame %2d  gt_t=(%+.2f,%+.2f,%+.2f) gt_w=%.2f  err_t=%.3f m  err_w=%.3f rad  %.2f ms\n",
                        frame, gt_t[0], gt_t[1], gt_t[2],
                        std::sqrt(gt_w[0]*gt_w[0]+gt_w[1]*gt_w[1]+gt_w[2]*gt_w[2]),
                        err_t, w_err, ms);

        // Visualisation
        cv::Mat img(PANEL_H + 60, PANEL_W, CV_8UC3, cv::Scalar(20, 20, 20));
        Cam cam{ 0.6f + 0.04f * frame, 0.55f, 28.0f };
        for (int i = 0; i < n; i += 2) {
            cv::Point2i p = project(map_cloud_h[3*i+0], map_cloud_h[3*i+1], map_cloud_h[3*i+2],
                                    cam, PANEL_W, PANEL_H);
            if (p.x < 0 || p.x >= PANEL_W || p.y < 0 || p.y >= PANEL_H) continue;
            img.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(150, 150, 150);
        }
        for (int i = 0; i < n; i += 2) {
            cv::Point2i p = project(live_cloud_h[3*i+0], live_cloud_h[3*i+1], live_cloud_h[3*i+2],
                                    cam, PANEL_W, PANEL_H);
            if (p.x < 0 || p.x >= PANEL_W || p.y < 0 || p.y >= PANEL_H) continue;
            img.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(50, 50, 220);
        }
        for (int i = 0; i < n; i++) {
            float ps[3] = { live_cloud_h[3*i+0], live_cloud_h[3*i+1], live_cloud_h[3*i+2] };
            float px = R_est[0]*ps[0] + R_est[1]*ps[1] + R_est[2]*ps[2] + t_est[0];
            float py = R_est[3]*ps[0] + R_est[4]*ps[1] + R_est[5]*ps[2] + t_est[1];
            float pz = R_est[6]*ps[0] + R_est[7]*ps[1] + R_est[8]*ps[2] + t_est[2];
            cv::Point2i p = project(px, py, pz, cam, PANEL_W, PANEL_H);
            if (p.x < 0 || p.x >= PANEL_W || p.y < 0 || p.y >= PANEL_H) continue;
            img.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(60, 220, 90);
        }
        cv::putText(img, cv::format("GPU GICP 3D registration  frame %d / %d", frame, N_FRAMES),
                    cv::Point(10, 24),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
        cv::putText(img, "grey=target  red=source(unaligned)  green=source(aligned)",
                    cv::Point(10, 46),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(190, 190, 190), 1);
        cv::putText(img, cv::format("gt_t=(%+.2f, %+.2f, %+.2f)   gt_|w|=%+.2f rad",
                                    gt_t[0], gt_t[1], gt_t[2],
                                    std::sqrt(gt_w[0]*gt_w[0]+gt_w[1]*gt_w[1]+gt_w[2]*gt_w[2])),
                    cv::Point(10, PANEL_H + 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220), 1);
        cv::putText(img, cv::format("err_t=%.3f m   err_R=%.3f rad   %.2f ms / scenario",
                                    err_t, w_err, ms),
                    cv::Point(10, PANEL_H + 42),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(180, 220, 180), 1);
        video.write(img);
    }
    video.release();

    std::printf("Avg err_t = %.4f m   avg err_R = %.4f rad   avg %.2f ms/frame\n",
                total_t_err / counted, total_w_err / counted, total_ms / counted);
    cudabot::avi_to_gif("gif/gpu_gicp_3d.avi", "gif/gpu_gicp_3d.gif", 10, 600);
    std::printf("GIF saved to gif/gpu_gicp_3d.gif\n");

    cudaFree(d_map); cudaFree(d_live);
    cudaFree(d_map_cov); cudaFree(d_live_cov);
    cudaFree(d_map_ok); cudaFree(d_live_ok);
    cudaFree(d_g); cudaFree(d_H); cudaFree(d_cost); cudaFree(d_match);
    cudaFree(d_R); cudaFree(d_t);
    return 0;
}
