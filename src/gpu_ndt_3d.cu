// gpu_ndt_3d.cu
//
// GPU 3D NDT (Normal Distributions Transform) point cloud registration.
//
// State per scenario:
//   - R (3x3 row-major) and t (3) representing the live -> map transform.
//   - Updates use the right-perturbation convention:
//       t_new = t + delta_t          (world-frame translation)
//       R_new = R * Exp(delta_w)     (body-frame rotation)
//   - 6-DOF delta = (delta_tx, delta_ty, delta_tz, delta_wx, delta_wy, delta_wz)
//
// For each map cell we store mu (3), Sigma^-1 (upper-tri 6 floats), valid.
// Score per live point: s = exp(-0.5 d^T Sigma^-1 d)  with d = R p + t - mu.
// Negative log score:    f = -ln(eta + s)
//
// Gauss-Newton on the sum of f over live points.  The per-point Jacobian of
// the residual r = d w.r.t. delta is:
//     J_r = [ I_3  |  -R * hat(p_sensor) ]            (3x6)
// Gauss-Newton Hessian approximation:
//     H ~= w * J_r^T Sigma^-1 J_r              with w = s / (eta + s)
// Gradient:
//     g  =  w * J_r^T Sigma^-1 d
//
// We accumulate H (21 floats, full 6x6 upper-tri symmetric) and g (6 floats)
// over all live points with atomicAdd.

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
#include "solve6_packed.cuh"

namespace cudabot {

// World volume: 20 x 20 x 6 m, centred near origin.
constexpr float X_HALF = 10.0f;
constexpr float Y_HALF = 10.0f;
constexpr float Z_LO   = 0.0f;
constexpr float Z_HI   = 6.0f;

constexpr int GX = 16;
constexpr int GY = 16;
constexpr int GZ = 6;
constexpr float CELL_XY = (2.0f * X_HALF) / GX;        // 0.625 m
constexpr float CELL_Z  = (Z_HI - Z_LO) / GZ;          // 0.5 m

constexpr int N_POINTS = 16000;     // points per cloud
constexpr int N_FRAMES = 60;
constexpr int GN_ITERS = 24;
constexpr int MIN_PTS_PER_CELL = 6;
constexpr float REG_VAR = 0.4f;
constexpr float NDT_EPS = 0.05f;

constexpr int PANEL_W = 720;
constexpr int PANEL_H = 560;

// -------------------------------------------------------------------------
// Cloud generation: a 'room' (4 walls + floor + ceiling) + 3 boxes.
// -------------------------------------------------------------------------
struct Box { float xmin, ymin, zmin, xmax, ymax, zmax; };

static void sample_plane(int n, float ux, float uy, float uz,
                         float vx, float vy, float vz,
                         float ox, float oy, float oz,
                         std::vector<float>& out,
                         std::mt19937& rng) {
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
    // Floor (z = 0) and ceiling (z = Z_HI), spanning the room.
    sample_plane(2200, 2 * X_HALF, 0, 0,  0, 2 * Y_HALF, 0,
                 -X_HALF, -Y_HALF, 0.0f, out, rng);
    sample_plane(1600, 2 * X_HALF, 0, 0,  0, 2 * Y_HALF, 0,
                 -X_HALF, -Y_HALF, Z_HI, out, rng);
    // Four walls.
    sample_plane(2200, 2 * X_HALF, 0, 0,  0, 0, Z_HI,
                 -X_HALF, -Y_HALF, 0.0f, out, rng);   // y = -Y_HALF
    sample_plane(2200, 2 * X_HALF, 0, 0,  0, 0, Z_HI,
                 -X_HALF,  Y_HALF, 0.0f, out, rng);   // y =  Y_HALF
    sample_plane(2200, 0, 2 * Y_HALF, 0, 0, 0, Z_HI,
                 -X_HALF, -Y_HALF, 0.0f, out, rng);   // x = -X_HALF
    sample_plane(2200, 0, 2 * Y_HALF, 0, 0, 0, Z_HI,
                  X_HALF, -Y_HALF, 0.0f, out, rng);   // x =  X_HALF
    // A couple of boxes inside.
    std::vector<Box> boxes = {
        {-4.0f, -3.0f, 0.0f, -1.5f, 0.0f, 2.0f},
        { 2.0f,  1.0f, 0.0f,  4.5f, 4.0f, 2.8f},
        {-2.0f,  3.0f, 0.0f,  1.0f, 6.0f, 1.5f},
    };
    for (const auto& b : boxes) {
        float dx = b.xmax - b.xmin;
        float dy = b.ymax - b.ymin;
        float dz = b.zmax - b.zmin;
        sample_plane(350, dx, 0, 0, 0, dy, 0, b.xmin, b.ymin, b.zmax, out, rng); // top
        sample_plane(220, dx, 0, 0, 0, 0, dz, b.xmin, b.ymin, b.zmin, out, rng); // y- face
        sample_plane(220, dx, 0, 0, 0, 0, dz, b.xmin, b.ymax, b.zmin, out, rng); // y+ face
        sample_plane(220, 0, dy, 0, 0, 0, dz, b.xmin, b.ymin, b.zmin, out, rng); // x- face
        sample_plane(220, 0, dy, 0, 0, 0, dz, b.xmax, b.ymin, b.zmin, out, rng); // x+ face
    }
}

static void add_noise(std::vector<float>& cloud, float sigma, std::mt19937& rng) {
    std::normal_distribution<float> n(0.0f, sigma);
    for (size_t i = 0; i < cloud.size(); i++) cloud[i] += n(rng);
}

// Apply a (R, t) transform to a point cloud (in-place).
static void mat3_mul(const float* A, const float* B, float* C) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            float v = 0.0f;
            for (int k = 0; k < 3; k++) v += A[3 * r + k] * B[3 * k + c];
            C[3 * r + c] = v;
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

static void apply_transform(const std::vector<float>& in, const float* R, const float* t,
                            std::vector<float>& out) {
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
// GPU kernels
// -------------------------------------------------------------------------
__device__ inline int voxel_index(float x, float y, float z) {
    int cx = (int)((x + X_HALF) / CELL_XY);
    int cy = (int)((y + Y_HALF) / CELL_XY);
    int cz = (int)((z - Z_LO)   / CELL_Z);
    if (cx < 0 || cx >= GX || cy < 0 || cy >= GY || cz < 0 || cz >= GZ) return -1;
    return (cz * GY + cy) * GX + cx;
}

__global__ void accum_grid_kernel(int n,
                                  const float* pts,           // n * 3
                                  float* sum_x, float* sum_y, float* sum_z,
                                  float* sum_xx, float* sum_yy, float* sum_zz,
                                  float* sum_xy, float* sum_xz, float* sum_yz,
                                  int* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = pts[3*i+0], y = pts[3*i+1], z = pts[3*i+2];
    int idx = voxel_index(x, y, z);
    if (idx < 0) return;
    atomicAdd(&sum_x[idx],  x);
    atomicAdd(&sum_y[idx],  y);
    atomicAdd(&sum_z[idx],  z);
    atomicAdd(&sum_xx[idx], x*x);
    atomicAdd(&sum_yy[idx], y*y);
    atomicAdd(&sum_zz[idx], z*z);
    atomicAdd(&sum_xy[idx], x*y);
    atomicAdd(&sum_xz[idx], x*z);
    atomicAdd(&sum_yz[idx], y*z);
    atomicAdd(&count[idx], 1);
}

// Invert 3x3 symmetric matrix C (stored as 6 floats: C00, C01, C02, C11, C12, C22)
// into inv (same layout).  Returns det.
__device__ inline float invert_sym3(const float* C, float* inv) {
    float a = C[0], b = C[1], c = C[2];
    float d = C[3], e = C[4], f = C[5];
    float det = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c);
    if (fabsf(det) < 1e-12f) det = (det < 0 ? -1e-12f : 1e-12f);
    float inv_det = 1.0f / det;
    inv[0] = (d * f - e * e) * inv_det;            // (0,0)
    inv[1] = (c * e - b * f) * inv_det;            // (0,1)
    inv[2] = (b * e - c * d) * inv_det;            // (0,2)
    inv[3] = (a * f - c * c) * inv_det;            // (1,1)
    inv[4] = (b * c - a * e) * inv_det;            // (1,2)
    inv[5] = (a * d - b * b) * inv_det;            // (2,2)
    return det;
}

__global__ void finalize_grid_kernel(int n_cells,
                                     const float* sum_x, const float* sum_y, const float* sum_z,
                                     const float* sum_xx, const float* sum_yy, const float* sum_zz,
                                     const float* sum_xy, const float* sum_xz, const float* sum_yz,
                                     const int* count,
                                     float* mu,           // n_cells * 3
                                     float* sinv,         // n_cells * 6 (upper-tri of Sigma^-1)
                                     unsigned char* valid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    int c = count[idx];
    if (c < MIN_PTS_PER_CELL) { valid[idx] = 0; return; }
    float inv_n = 1.0f / (float)c;
    float mx = sum_x[idx] * inv_n;
    float my = sum_y[idx] * inv_n;
    float mz = sum_z[idx] * inv_n;
    float vxx = sum_xx[idx] * inv_n - mx * mx + REG_VAR;
    float vyy = sum_yy[idx] * inv_n - my * my + REG_VAR;
    float vzz = sum_zz[idx] * inv_n - mz * mz + REG_VAR;
    float vxy = sum_xy[idx] * inv_n - mx * my;
    float vxz = sum_xz[idx] * inv_n - mx * mz;
    float vyz = sum_yz[idx] * inv_n - my * mz;
    float Sigma6[6] = { vxx, vxy, vxz, vyy, vyz, vzz };
    float Sinv[6];
    float det = invert_sym3(Sigma6, Sinv);
    if (fabsf(det) < 1e-10f) { valid[idx] = 0; return; }
    mu[3*idx+0] = mx; mu[3*idx+1] = my; mu[3*idx+2] = mz;
    for (int k = 0; k < 6; k++) sinv[6*idx+k] = Sinv[k];
    valid[idx] = 1;
}

// Compute g (6) and H (21 floats upper-tri 6x6) for current (R, t).
__global__ void ndt_grad_hess_kernel(int n,
                                     const float* pts_sensor,   // live points in sensor frame
                                     const float* R, const float* t,
                                     const float* mu, const float* sinv,
                                     const unsigned char* valid,
                                     float* g, float* H, float* score_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float ps[3] = { pts_sensor[3*i+0], pts_sensor[3*i+1], pts_sensor[3*i+2] };
    // pw = R * ps + t
    float pw[3];
    pw[0] = R[0]*ps[0] + R[1]*ps[1] + R[2]*ps[2] + t[0];
    pw[1] = R[3]*ps[0] + R[4]*ps[1] + R[5]*ps[2] + t[1];
    pw[2] = R[6]*ps[0] + R[7]*ps[1] + R[8]*ps[2] + t[2];
    int idx = voxel_index(pw[0], pw[1], pw[2]);
    if (idx < 0) return;
    if (!valid[idx]) return;
    float d[3] = { pw[0] - mu[3*idx+0], pw[1] - mu[3*idx+1], pw[2] - mu[3*idx+2] };
    float S00 = sinv[6*idx+0], S01 = sinv[6*idx+1], S02 = sinv[6*idx+2];
    float S11 = sinv[6*idx+3], S12 = sinv[6*idx+4], S22 = sinv[6*idx+5];
    // q = Sigma^-1 * d
    float q[3];
    q[0] = S00*d[0] + S01*d[1] + S02*d[2];
    q[1] = S01*d[0] + S11*d[1] + S12*d[2];
    q[2] = S02*d[0] + S12*d[1] + S22*d[2];
    float quad = d[0]*q[0] + d[1]*q[1] + d[2]*q[2];
    float score = expf(-0.5f * quad);
    atomicAdd(score_out, score);
    float w = score / (NDT_EPS + score);

    // J_r (3x6) = [ I | -R * hat(ps) ] (rows: dr_x, dr_y, dr_z; cols: dtx, dty, dtz, dwx, dwy, dwz)
    // hat(ps) = | 0    -ps_z  ps_y |
    //           | ps_z  0    -ps_x |
    //           |-ps_y  ps_x  0    |
    // -R * hat(ps) (3x3): compute row-by-row.
    // J3..J5 (the rotation columns) is a 3x3 matrix, let's call it M.
    float M[9];
    // M = -R * hat(ps)
    // Let h0 = R*(0, -ps_z, ps_y) etc.
    M[0] = -(R[0]*0.0f      + R[1]*ps[2] + R[2]*(-ps[1]));  // col dwx
    M[1] = -(R[0]*(-ps[2])  + R[1]*0.0f  + R[2]*ps[0]);     // col dwy
    M[2] = -(R[0]*ps[1]     + R[1]*(-ps[0]) + R[2]*0.0f);   // col dwz
    M[3] = -(R[3]*0.0f      + R[4]*ps[2] + R[5]*(-ps[1]));
    M[4] = -(R[3]*(-ps[2])  + R[4]*0.0f  + R[5]*ps[0]);
    M[5] = -(R[3]*ps[1]     + R[4]*(-ps[0]) + R[5]*0.0f);
    M[6] = -(R[6]*0.0f      + R[7]*ps[2] + R[8]*(-ps[1]));
    M[7] = -(R[6]*(-ps[2])  + R[7]*0.0f  + R[8]*ps[0]);
    M[8] = -(R[6]*ps[1]     + R[7]*(-ps[0]) + R[8]*0.0f);
    // Full J (3x6), row-major:
    //   row r col c < 3 : (r == c ? 1 : 0)
    //   row r col c >= 3: M[3*r + (c-3)]
    // Gradient g_k = w * d^T Sigma^-1 J[:, k] = w * q^T J[:, k]
    // For k in {0, 1, 2}: J[:, k] = e_k ⇒ q^T e_k = q[k]
    // For k in {3, 4, 5}: J[:, k] = M[:, k-3] ⇒ q^T M_col
    float g_loc[6];
    g_loc[0] = w * q[0];
    g_loc[1] = w * q[1];
    g_loc[2] = w * q[2];
    g_loc[3] = w * (q[0] * M[0] + q[1] * M[3] + q[2] * M[6]);
    g_loc[4] = w * (q[0] * M[1] + q[1] * M[4] + q[2] * M[7]);
    g_loc[5] = w * (q[0] * M[2] + q[1] * M[5] + q[2] * M[8]);
    atomicAdd(&g[0], g_loc[0]);
    atomicAdd(&g[1], g_loc[1]);
    atomicAdd(&g[2], g_loc[2]);
    atomicAdd(&g[3], g_loc[3]);
    atomicAdd(&g[4], g_loc[4]);
    atomicAdd(&g[5], g_loc[5]);

    // H = w * J^T Sigma^-1 J  (6x6 PSD)
    // Compute U = Sigma^-1 * J (3x6), then H = w * J^T * U.
    // U[:, 0..2] = Sigma^-1 * I = Sigma^-1
    // U[:, 3..5] = Sigma^-1 * M (3x3)
    float Sm[9];
    Sm[0] = S00*M[0] + S01*M[3] + S02*M[6];
    Sm[1] = S00*M[1] + S01*M[4] + S02*M[7];
    Sm[2] = S00*M[2] + S01*M[5] + S02*M[8];
    Sm[3] = S01*M[0] + S11*M[3] + S12*M[6];
    Sm[4] = S01*M[1] + S11*M[4] + S12*M[7];
    Sm[5] = S01*M[2] + S11*M[5] + S12*M[8];
    Sm[6] = S02*M[0] + S12*M[3] + S22*M[6];
    Sm[7] = S02*M[1] + S12*M[4] + S22*M[7];
    Sm[8] = S02*M[2] + S12*M[5] + S22*M[8];

    // H_{p,q} = w * sum_r J[r, p] * U[r, q]
    // For p in {0,1,2}: J[r, p] = (r == p ? 1 : 0); so H[p, q] = w * U[p, q]
    //   for q < 3: U[p, q] = Sigma^-1[p, q]
    //   for q in {3,4,5}: U[p, q] = Sm[3*p + (q-3)]
    // For p in {3,4,5}: J[r, p] = M[r, p-3]; H[p, q] = w * (M[0, p-3]*U[0,q] + M[1, p-3]*U[1,q] + M[2, p-3]*U[2,q])
    //
    // We accumulate upper-tri 6x6 = 21 entries.  Index layout:
    //   H[0]=H(0,0) H[1]=H(0,1) H[2]=H(0,2) H[3]=H(0,3) H[4]=H(0,4) H[5]=H(0,5)
    //   H[6]=H(1,1) H[7]=H(1,2) H[8]=H(1,3) H[9]=H(1,4) H[10]=H(1,5)
    //   H[11]=H(2,2) H[12]=H(2,3) H[13]=H(2,4) H[14]=H(2,5)
    //   H[15]=H(3,3) H[16]=H(3,4) H[17]=H(3,5)
    //   H[18]=H(4,4) H[19]=H(4,5)
    //   H[20]=H(5,5)
    float Hl[21];
    // row 0
    Hl[0] = w * S00;
    Hl[1] = w * S01;
    Hl[2] = w * S02;
    Hl[3] = w * Sm[0];
    Hl[4] = w * Sm[1];
    Hl[5] = w * Sm[2];
    // row 1
    Hl[6] = w * S11;
    Hl[7] = w * S12;
    Hl[8] = w * Sm[3];
    Hl[9] = w * Sm[4];
    Hl[10] = w * Sm[5];
    // row 2
    Hl[11] = w * S22;
    Hl[12] = w * Sm[6];
    Hl[13] = w * Sm[7];
    Hl[14] = w * Sm[8];
    // rows 3..5 with rotation block
    // H(p, q) for p,q in {3,4,5}: w * M[:, p-3]^T * U[:, q]
    //   where U[:, q] for q in {3,4,5} = Sm[:, q-3]
    // H(3, 3) = w * M_col0 . Sm_col0
    auto Mcol_dot = [&](int a, int b) {
        // M[:, a] = (M[0+a], M[3+a], M[6+a]) since M is row-major 3x3.
        // We want M_col_a^T * Sm_col_b = M[0+a]*Sm[0+b] + M[3+a]*Sm[3+b] + M[6+a]*Sm[6+b]
        return M[0 + a] * Sm[0 + b] + M[3 + a] * Sm[3 + b] + M[6 + a] * Sm[6 + b];
    };
    Hl[15] = w * Mcol_dot(0, 0);
    Hl[16] = w * Mcol_dot(0, 1);
    Hl[17] = w * Mcol_dot(0, 2);
    Hl[18] = w * Mcol_dot(1, 1);
    Hl[19] = w * Mcol_dot(1, 2);
    Hl[20] = w * Mcol_dot(2, 2);
    for (int k = 0; k < 21; k++) atomicAdd(&H[k], Hl[k]);
}

// The packed-6x6 SPD Cholesky solve (H_OFF + cholesky_solve_6) is shared with
// gpu_gicp_3d.cu / gpu_ndt_3d_multires.cu via include/solve6_packed.cuh.

// -------------------------------------------------------------------------
// Visualisation: simple perspective camera projection.
// -------------------------------------------------------------------------
struct Cam {
    float yaw, pitch, dist;
};
static cv::Point2i project(float x, float y, float z, const Cam& c, int W, int H) {
    float cy = std::cos(c.yaw), sy = std::sin(c.yaw);
    float cp = std::cos(c.pitch), sp = std::sin(c.pitch);
    float x1 =  cy * x + sy * y;
    float y1 = -sy * x + cy * y;
    float z1 =  z - 1.5f;
    float y2 =  cp * y1 - sp * z1;
    float z2 =  sp * y1 + cp * z1;
    float xc = x1;
    float yc = z2;
    float zc = c.dist - y2;
    if (zc < 0.1f) zc = 0.1f;
    float f = 1.0f * H;
    int px = (int)(W * 0.5f + f * xc / zc);
    int py = (int)(H * 0.6f - f * yc / zc);
    return cv::Point2i(px, py);
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::mt19937 rng(123);
    std::vector<float> map_cloud_h;
    make_cloud(map_cloud_h, rng);
    int n_map = (int)map_cloud_h.size() / 3;
    if (n_map > N_POINTS) {
        map_cloud_h.resize(N_POINTS * 3);
        n_map = N_POINTS;
    }
    add_noise(map_cloud_h, 0.015f, rng);
    std::printf("Map cloud: %d points\n", n_map);

    int n_cells = GX * GY * GZ;

    // Device buffers
    float *d_map = nullptr, *d_live = nullptr;
    CUDA_CHECK(cudaMalloc(&d_map,  n_map * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_live, n_map * 3 * sizeof(float)));

    float *d_sum_x, *d_sum_y, *d_sum_z;
    float *d_sum_xx, *d_sum_yy, *d_sum_zz;
    float *d_sum_xy, *d_sum_xz, *d_sum_yz;
    int   *d_count;
    float *d_mu, *d_sinv;
    unsigned char *d_valid;
    CUDA_CHECK(cudaMalloc(&d_sum_x, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_y, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_z, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_xx, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_yy, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_zz, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_xy, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_xz, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_yz, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_count, n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mu, n_cells * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sinv, n_cells * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid, n_cells));

    float *d_g, *d_H, *d_score, *d_Rt, *d_tt;
    CUDA_CHECK(cudaMalloc(&d_g, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H, 21 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_score, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Rt, 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tt, 3 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_map, map_cloud_h.data(), n_map * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Build NDT grid from the map.
    int blk = 256;
    int blocks_pts = (n_map + blk - 1) / blk;
    int blocks_cells = (n_cells + blk - 1) / blk;

    CUDA_CHECK(cudaMemset(d_sum_x, 0, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_y, 0, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_z, 0, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_xx, 0, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_yy, 0, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_zz, 0, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_xy, 0, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_xz, 0, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum_yz, 0, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_count, 0, n_cells * sizeof(int)));

    accum_grid_kernel<<<blocks_pts, blk>>>(n_map, d_map,
                                            d_sum_x, d_sum_y, d_sum_z,
                                            d_sum_xx, d_sum_yy, d_sum_zz,
                                            d_sum_xy, d_sum_xz, d_sum_yz,
                                            d_count);
    finalize_grid_kernel<<<blocks_cells, blk>>>(n_cells,
                                                 d_sum_x, d_sum_y, d_sum_z,
                                                 d_sum_xx, d_sum_yy, d_sum_zz,
                                                 d_sum_xy, d_sum_xz, d_sum_yz,
                                                 d_count, d_mu, d_sinv, d_valid);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Visualisation setup
    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_ndt_3d.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          10, cv::Size(PANEL_W, PANEL_H + 60));

    std::uniform_real_distribution<float> per_xy(-1.2f, 1.2f);
    std::uniform_real_distribution<float> per_z(-0.6f, 0.6f);
    std::uniform_real_distribution<float> per_w(-0.25f, 0.25f);

    double total_t_err = 0.0, total_w_err = 0.0, total_ms = 0.0;
    int counted = 0;

    for (int frame = 0; frame < N_FRAMES; frame++) {
        // Ground-truth perturbation
        float gt_t[3] = { per_xy(rng), per_xy(rng), per_z(rng) };
        float gt_w[3] = { per_w(rng), per_w(rng), per_w(rng) };
        float gt_R[9];
        so3_exp(gt_w, gt_R);
        // Live cloud = GT_R * map + GT_t  (so the "true" alignment transform
        // taking live back to map is R^T, -R^T t).  We solve for (R_est, t_est)
        // that maps LIVE points back to MAP frame, so the GT for the estimator
        // is (R_est_gt, t_est_gt) = (gt_R^T, -gt_R^T * gt_t).
        std::vector<float> live_cloud_h;
        apply_transform(map_cloud_h, gt_R, gt_t, live_cloud_h);
        add_noise(live_cloud_h, 0.015f, rng);

        // Target for estimator
        float Rgt_inv[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                Rgt_inv[3*r + c] = gt_R[3*c + r];
        float tgt_inv[3];
        tgt_inv[0] = -(Rgt_inv[0]*gt_t[0] + Rgt_inv[1]*gt_t[1] + Rgt_inv[2]*gt_t[2]);
        tgt_inv[1] = -(Rgt_inv[3]*gt_t[0] + Rgt_inv[4]*gt_t[1] + Rgt_inv[5]*gt_t[2]);
        tgt_inv[2] = -(Rgt_inv[6]*gt_t[0] + Rgt_inv[7]*gt_t[1] + Rgt_inv[8]*gt_t[2]);

        CUDA_CHECK(cudaMemcpy(d_live, live_cloud_h.data(), n_map * 3 * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Initialise estimate at identity.
        float R_est[9] = {1,0,0, 0,1,0, 0,0,1};
        float t_est[3] = {0,0,0};
        float lambda = 1.0f;
        float prev_neg_score = 1e30f;

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int it = 0; it < GN_ITERS; it++) {
            CUDA_CHECK(cudaMemset(d_g, 0, 6 * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_H, 0, 21 * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_score, 0, sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_Rt, R_est, 9 * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_tt, t_est, 3 * sizeof(float), cudaMemcpyHostToDevice));
            ndt_grad_hess_kernel<<<blocks_pts, blk>>>(n_map, d_live, d_Rt, d_tt,
                                                       d_mu, d_sinv, d_valid,
                                                       d_g, d_H, d_score);
            float g_h[6], H_h[21], score_h = 0.0f;
            CUDA_CHECK(cudaMemcpy(g_h, d_g, 6 * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(H_h, d_H, 21 * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&score_h, d_score, sizeof(float), cudaMemcpyDeviceToHost));
            float neg = -score_h;
            // Adaptive damping: if improved, decrease; else increase.
            if (neg < prev_neg_score) lambda = fmaxf(lambda * 0.5f, 1e-3f);
            else                      lambda = fminf(lambda * 4.0f, 1e6f);
            prev_neg_score = neg;
            float dx[6] = {0};
            // GN step is -H^-1 g  (we want gradient descent of negative log score)
            // g_h was accumulated as +grad of -log score (since g_loc = w * q^T J,
            // matching d(-ln(eta+s))/dx in absolute value, and we negate inside the
            // GN update by computing dx = -H^-1 g).
            if (!cholesky_solve_6(H_h, g_h, lambda, dx)) {
                lambda = fminf(lambda * 4.0f, 1e6f);
                continue;
            }
            // dx is H^-1 g; we want -H^-1 g
            float delta_t[3] = { -dx[0], -dx[1], -dx[2] };
            float delta_w[3] = { -dx[3], -dx[4], -dx[5] };
            // Cap step
            float nt = std::sqrt(delta_t[0]*delta_t[0] + delta_t[1]*delta_t[1] + delta_t[2]*delta_t[2]);
            if (nt > 0.5f) { float k = 0.5f / nt; for (int q=0;q<3;q++) delta_t[q] *= k; }
            float nw = std::sqrt(delta_w[0]*delta_w[0] + delta_w[1]*delta_w[1] + delta_w[2]*delta_w[2]);
            if (nw > 0.2f) { float k = 0.2f / nw; for (int q=0;q<3;q++) delta_w[q] *= k; }
            // Apply
            t_est[0] += delta_t[0];
            t_est[1] += delta_t[1];
            t_est[2] += delta_t[2];
            float E[9]; so3_exp(delta_w, E);
            float Rn[9]; mat3_mul(R_est, E, Rn);
            for (int q = 0; q < 9; q++) R_est[q] = Rn[q];
        }
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);

        // Errors: compare (R_est, t_est) with (Rgt_inv, tgt_inv).
        float err_t = std::sqrt((t_est[0]-tgt_inv[0])*(t_est[0]-tgt_inv[0])
                              + (t_est[1]-tgt_inv[1])*(t_est[1]-tgt_inv[1])
                              + (t_est[2]-tgt_inv[2])*(t_est[2]-tgt_inv[2]));
        // Rotation error: log(R_est^T Rgt_inv)
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
        total_t_err += err_t;
        total_w_err += w_err;
        total_ms += ms;
        counted++;
        if (frame < 6 || frame % 10 == 0)
            std::printf("frame %2d  gt_t=(%+.2f,%+.2f,%+.2f) gt_w=%.2f rad  err_t=%.3f m  err_w=%.3f rad  %.2f ms\n",
                        frame, gt_t[0], gt_t[1], gt_t[2],
                        std::sqrt(gt_w[0]*gt_w[0]+gt_w[1]*gt_w[1]+gt_w[2]*gt_w[2]),
                        err_t, w_err, ms);

        // ---- Visualisation ----
        cv::Mat img(PANEL_H + 60, PANEL_W, CV_8UC3, cv::Scalar(20, 20, 20));
        Cam cam{ 0.6f + 0.04f * frame, 0.55f, 28.0f };

        // map cloud (light grey)
        for (int i = 0; i < n_map; i += 6) {
            cv::Point2i p = project(map_cloud_h[3*i+0], map_cloud_h[3*i+1], map_cloud_h[3*i+2],
                                    cam, PANEL_W, PANEL_H);
            if (p.x < 0 || p.x >= PANEL_W || p.y < 0 || p.y >= PANEL_H) continue;
            img.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(150, 150, 150);
        }
        // live cloud BEFORE align (red)
        for (int i = 0; i < n_map; i += 6) {
            cv::Point2i p = project(live_cloud_h[3*i+0], live_cloud_h[3*i+1], live_cloud_h[3*i+2],
                                    cam, PANEL_W, PANEL_H);
            if (p.x < 0 || p.x >= PANEL_W || p.y < 0 || p.y >= PANEL_H) continue;
            img.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(50, 50, 220);
        }
        // live cloud AFTER align (green): apply (R_est, t_est) to live cloud
        for (int i = 0; i < n_map; i += 4) {
            float ps[3] = { live_cloud_h[3*i+0], live_cloud_h[3*i+1], live_cloud_h[3*i+2] };
            float px = R_est[0]*ps[0] + R_est[1]*ps[1] + R_est[2]*ps[2] + t_est[0];
            float py = R_est[3]*ps[0] + R_est[4]*ps[1] + R_est[5]*ps[2] + t_est[1];
            float pz = R_est[6]*ps[0] + R_est[7]*ps[1] + R_est[8]*ps[2] + t_est[2];
            cv::Point2i p = project(px, py, pz, cam, PANEL_W, PANEL_H);
            if (p.x < 0 || p.x >= PANEL_W || p.y < 0 || p.y >= PANEL_H) continue;
            img.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(60, 220, 90);
        }

        cv::putText(img, cv::format("GPU NDT 3D registration  frame %d / %d", frame, N_FRAMES),
                    cv::Point(10, 24),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
        cv::putText(img, "grey=map  red=live(unaligned)  green=live(aligned)",
                    cv::Point(10, 46),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(190, 190, 190), 1);
        cv::putText(img, cv::format("gt_t=(%+.2f, %+.2f, %+.2f)   gt_yaw=%+.2f rad",
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
    cudabot::avi_to_gif("gif/gpu_ndt_3d.avi", "gif/gpu_ndt_3d.gif", 10, 600);
    std::printf("GIF saved to gif/gpu_ndt_3d.gif\n");

    cudaFree(d_map); cudaFree(d_live);
    cudaFree(d_sum_x); cudaFree(d_sum_y); cudaFree(d_sum_z);
    cudaFree(d_sum_xx); cudaFree(d_sum_yy); cudaFree(d_sum_zz);
    cudaFree(d_sum_xy); cudaFree(d_sum_xz); cudaFree(d_sum_yz);
    cudaFree(d_count); cudaFree(d_mu); cudaFree(d_sinv); cudaFree(d_valid);
    cudaFree(d_g); cudaFree(d_H); cudaFree(d_score); cudaFree(d_Rt); cudaFree(d_tt);
    return 0;
}
