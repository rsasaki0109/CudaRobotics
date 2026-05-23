// gpu_pose_graph_slam.cu
//
// GPU 2D pose-graph SLAM backend (Gauss-Newton + Jacobi-PCG).
//
// Problem setup:
//   - N nodes, each SE(2) pose (x, y, theta).
//   - Sequential odometry edges + extra loop-closure edges.
//   - Each edge has a measured relative pose Z_ij and an information matrix.
//
// Residual for edge (i, j) with measurement (zx, zy, zt):
//   T_ij = T_i^-1 * T_j  (in the rotated frame of i)
//     dx = (xj - xi) cos(ti) + (yj - yi) sin(ti)
//     dy = -(xj - xi) sin(ti) + (yj - yi) cos(ti)
//     dt = wrap(tj - ti)
//   r = (dx - zx, dy - zy, wrap(dt - zt))
//
// Jacobians (Euclidean parameterization of the pose vector):
//   dr/d(xi_i) is 3x3, dr/d(xi_j) is 3x3.
//
// Normal equations: H dx = -b
//   H_ii += J_i^T Omega J_i
//   H_jj += J_j^T Omega J_j
//   H_ij += J_i^T Omega J_j  (and H_ji = H_ij^T)
//   b_i  += J_i^T Omega r
//   b_j  += J_j^T Omega r
//
// We never materialize H; we recompute the edge-wise multiplies inside the
// PCG matvec. Block-diagonal Jacobi preconditioner (3x3 inverse per node).
//
// Pose 0 is anchored (delta = 0) to fix gauge.
//
// Visualization: two panels (ground truth + noisy initial, ground truth + optimized).

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_blas.cuh"
#include "cuda_video.h"

namespace cudabot {

using blas::axpy_kernel;
using blas::xpay_kernel;
using blas::copy_kernel;
using blas::zero_kernel;
using blas::dot_kernel;

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------
constexpr int N_POSES = 200;
constexpr int GN_ITERS = 30;
constexpr int PCG_ITERS = 100;
constexpr float PCG_TOL = 1.0e-7f;

constexpr float ODOM_SIGMA_XY = 0.06f;
constexpr float ODOM_SIGMA_TH = 0.020f;
constexpr float LC_SIGMA_XY = 0.05f;
constexpr float LC_SIGMA_TH = 0.015f;

constexpr float LC_DIST = 2.5f;
constexpr int   LC_MIN_GAP = 25;

constexpr int   PANEL_W = 540;
constexpr int   PANEL_H = 540;

// -------------------------------------------------------------------------
// Host helpers
// -------------------------------------------------------------------------
struct Edge {
    int i, j;
    float zx, zy, zt;
};

static inline float wrap_angle(float a) {
    while (a >  M_PI) a -= 2.0f * M_PI;
    while (a < -M_PI) a += 2.0f * M_PI;
    return a;
}

static void make_ground_truth(std::vector<float>& gt) {
    gt.assign(N_POSES * 3, 0.0f);
    // figure-8 trajectory
    for (int k = 0; k < N_POSES; k++) {
        float s = static_cast<float>(k) / (N_POSES - 1);
        float u = s * 2.0f * static_cast<float>(M_PI);
        float x = 10.0f * std::sin(u);
        float y = 6.0f * std::sin(2.0f * u);
        float dxds = 10.0f * std::cos(u);
        float dyds = 12.0f * std::cos(2.0f * u);
        float th = std::atan2(dyds, dxds);
        gt[3 * k + 0] = x;
        gt[3 * k + 1] = y;
        gt[3 * k + 2] = th;
    }
}

static void chain_initial_from_odometry(const std::vector<Edge>& odom,
                                        const std::vector<float>& gt,
                                        std::vector<float>& poses) {
    poses.assign(N_POSES * 3, 0.0f);
    // Anchor pose 0 to GT (gauge fix).
    poses[0] = gt[0];
    poses[1] = gt[1];
    poses[2] = gt[2];
    for (const auto& e : odom) {
        float xi = poses[3 * e.i + 0];
        float yi = poses[3 * e.i + 1];
        float ti = poses[3 * e.i + 2];
        float c = std::cos(ti), s = std::sin(ti);
        poses[3 * e.j + 0] = xi + c * e.zx - s * e.zy;
        poses[3 * e.j + 1] = yi + s * e.zx + c * e.zy;
        poses[3 * e.j + 2] = wrap_angle(ti + e.zt);
    }
}

// -------------------------------------------------------------------------
// CUDA kernels
// -------------------------------------------------------------------------

// For each edge compute residual r and contribution to b.
// Also compute diagonal blocks H_ii, H_jj for the Jacobi preconditioner.
__global__ void assemble_kernel(int n_edges,
                                const int* __restrict__ ei,
                                const int* __restrict__ ej,
                                const float* __restrict__ ez,    // n_edges * 3
                                const float* __restrict__ poses, // n_poses * 3
                                float omega_xy, float omega_th,
                                float* __restrict__ b,           // n_poses * 3
                                float* __restrict__ diag) {      // n_poses * 9
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;

    int i = ei[e], j = ej[e];
    float zx = ez[3 * e + 0];
    float zy = ez[3 * e + 1];
    float zt = ez[3 * e + 2];

    float xi = poses[3 * i + 0];
    float yi = poses[3 * i + 1];
    float ti = poses[3 * i + 2];
    float xj = poses[3 * j + 0];
    float yj = poses[3 * j + 1];
    float tj = poses[3 * j + 2];

    float c = cosf(ti), s = sinf(ti);
    float dxw = xj - xi;
    float dyw = yj - yi;
    float dx = dxw * c + dyw * s;
    float dy = -dxw * s + dyw * c;
    float dt = tj - ti;
    while (dt >  M_PI) dt -= 2.0f * M_PI;
    while (dt < -M_PI) dt += 2.0f * M_PI;

    float rx = dx - zx;
    float ry = dy - zy;
    float rt = dt - zt;
    while (rt >  M_PI) rt -= 2.0f * M_PI;
    while (rt < -M_PI) rt += 2.0f * M_PI;

    // Jacobians J_i (3x3), J_j (3x3)
    //   dr_x/d xi_i = (-c, -s, -dxw*s + dyw*c)
    //   dr_y/d xi_i = ( s, -c, -dxw*c - dyw*s)
    //   dr_t/d xi_i = ( 0,  0, -1)
    //   dr_x/d xi_j = ( c,  s, 0)
    //   dr_y/d xi_j = (-s,  c, 0)
    //   dr_t/d xi_j = ( 0,  0, 1)
    float Ji[9] = {
        -c, -s, -dxw * s + dyw * c,
         s, -c, -dxw * c - dyw * s,
         0.0f, 0.0f, -1.0f
    };
    float Jj[9] = {
         c,  s, 0.0f,
        -s,  c, 0.0f,
         0.0f, 0.0f, 1.0f
    };
    float Wr[3] = { omega_xy * rx, omega_xy * ry, omega_th * rt };

    // b += J^T W r
    for (int k = 0; k < 3; k++) {
        float bi_k = Ji[3 * 0 + k] * Wr[0] + Ji[3 * 1 + k] * Wr[1] + Ji[3 * 2 + k] * Wr[2];
        atomicAdd(&b[3 * i + k], bi_k);
    }
    for (int k = 0; k < 3; k++) {
        float bj_k = Jj[3 * 0 + k] * Wr[0] + Jj[3 * 1 + k] * Wr[1] + Jj[3 * 2 + k] * Wr[2];
        atomicAdd(&b[3 * j + k], bj_k);
    }

    // diag block: J^T W J  (only diagonal 3x3 blocks)
    float w[3] = { omega_xy, omega_xy, omega_th };
    // For each pair of cols (p, q) in {0..2}: sum_k Ji[k,p] * w[k] * Ji[k,q]
    // store as 9 floats row-major (3x3)
    for (int p = 0; p < 3; p++) {
        for (int q = 0; q < 3; q++) {
            float vi = 0.0f, vj = 0.0f;
            for (int kk = 0; kk < 3; kk++) {
                vi += Ji[3 * kk + p] * w[kk] * Ji[3 * kk + q];
                vj += Jj[3 * kk + p] * w[kk] * Jj[3 * kk + q];
            }
            atomicAdd(&diag[9 * i + 3 * p + q], vi);
            atomicAdd(&diag[9 * j + 3 * p + q], vj);
        }
    }
}

// Anchor: zero out b[0..2] and force diag[0] = identity (so block solve does nothing).
__global__ void anchor_kernel(float* b, float* diag) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        b[0] = 0.0f; b[1] = 0.0f; b[2] = 0.0f;
        for (int p = 0; p < 9; p++) diag[p] = 0.0f;
        diag[0] = 1.0f; diag[4] = 1.0f; diag[8] = 1.0f;
    }
}

// matvec: y = H * x  (via per-edge accumulation)
__global__ void matvec_kernel(int n_edges,
                              const int* __restrict__ ei,
                              const int* __restrict__ ej,
                              const float* __restrict__ ez,
                              const float* __restrict__ poses,
                              float omega_xy, float omega_th,
                              const float* __restrict__ x,
                              float* __restrict__ y) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    int i = ei[e], j = ej[e];
    float ti = poses[3 * i + 2];
    float dxw = poses[3 * j + 0] - poses[3 * i + 0];
    float dyw = poses[3 * j + 1] - poses[3 * i + 1];
    float c = cosf(ti), s = sinf(ti);
    float Ji[9] = {
        -c, -s, -dxw * s + dyw * c,
         s, -c, -dxw * c - dyw * s,
         0.0f, 0.0f, -1.0f
    };
    float Jj[9] = {
         c,  s, 0.0f,
        -s,  c, 0.0f,
         0.0f, 0.0f, 1.0f
    };

    float xi[3] = { x[3 * i + 0], x[3 * i + 1], x[3 * i + 2] };
    float xj[3] = { x[3 * j + 0], x[3 * j + 1], x[3 * j + 2] };

    // u = J_i * xi + J_j * xj  (3-vector)
    float u[3] = {0.0f, 0.0f, 0.0f};
    for (int r = 0; r < 3; r++) {
        u[r] = Ji[3 * r + 0] * xi[0] + Ji[3 * r + 1] * xi[1] + Ji[3 * r + 2] * xi[2]
             + Jj[3 * r + 0] * xj[0] + Jj[3 * r + 1] * xj[1] + Jj[3 * r + 2] * xj[2];
    }
    float w[3] = { omega_xy, omega_xy, omega_th };
    float Wu[3] = { w[0] * u[0], w[1] * u[1], w[2] * u[2] };

    // y_i += J_i^T * Wu;  y_j += J_j^T * Wu;
    for (int k = 0; k < 3; k++) {
        float yi_k = Ji[3 * 0 + k] * Wu[0] + Ji[3 * 1 + k] * Wu[1] + Ji[3 * 2 + k] * Wu[2];
        float yj_k = Jj[3 * 0 + k] * Wu[0] + Jj[3 * 1 + k] * Wu[1] + Jj[3 * 2 + k] * Wu[2];
        atomicAdd(&y[3 * i + k], yi_k);
        atomicAdd(&y[3 * j + k], yj_k);
    }
}

// Apply 3x3 block inverse preconditioner per node.  z[i] = D[i]^-1 * r[i]
__global__ void apply_precond_kernel(int n_poses,
                                     const float* __restrict__ diag,
                                     const float* __restrict__ r,
                                     float* __restrict__ z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_poses) return;
    const float* D = diag + 9 * i;
    float m[9];
    for (int k = 0; k < 9; k++) m[k] = D[k];
    // Add a small lambda to the diagonal to keep positive definite.
    m[0] += 1.0e-6f; m[4] += 1.0e-6f; m[8] += 1.0e-6f;
    // det
    float det = m[0] * (m[4] * m[8] - m[5] * m[7])
              - m[1] * (m[3] * m[8] - m[5] * m[6])
              + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (fabsf(det) < 1.0e-12f) {
        z[3 * i + 0] = r[3 * i + 0];
        z[3 * i + 1] = r[3 * i + 1];
        z[3 * i + 2] = r[3 * i + 2];
        return;
    }
    float inv_det = 1.0f / det;
    float inv[9];
    inv[0] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
    inv[1] = -(m[1] * m[8] - m[2] * m[7]) * inv_det;
    inv[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
    inv[3] = -(m[3] * m[8] - m[5] * m[6]) * inv_det;
    inv[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
    inv[5] = -(m[0] * m[5] - m[2] * m[3]) * inv_det;
    inv[6] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
    inv[7] = -(m[0] * m[7] - m[1] * m[6]) * inv_det;
    inv[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;

    float rx = r[3 * i + 0], ry = r[3 * i + 1], rt = r[3 * i + 2];
    z[3 * i + 0] = inv[0] * rx + inv[1] * ry + inv[2] * rt;
    z[3 * i + 1] = inv[3] * rx + inv[4] * ry + inv[5] * rt;
    z[3 * i + 2] = inv[6] * rx + inv[7] * ry + inv[8] * rt;
}

__global__ void zero_anchor_dx(float* dx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dx[0] = 0.0f; dx[1] = 0.0f; dx[2] = 0.0f;
    }
}

__global__ void update_poses_kernel(int n_poses,
                                    float* __restrict__ poses,
                                    const float* __restrict__ dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_poses) return;
    if (i == 0) return;  // anchor
    poses[3 * i + 0] += dx[3 * i + 0];
    poses[3 * i + 1] += dx[3 * i + 1];
    float th = poses[3 * i + 2] + dx[3 * i + 2];
    while (th >  M_PI) th -= 2.0f * M_PI;
    while (th < -M_PI) th += 2.0f * M_PI;
    poses[3 * i + 2] = th;
}

// Reductions and BLAS-like primitives come from include/cuda_blas.cuh

// -------------------------------------------------------------------------
// Host driver
// -------------------------------------------------------------------------

static float pose_rmse(const std::vector<float>& a, const std::vector<float>& gt) {
    double s = 0.0;
    for (int k = 0; k < N_POSES; k++) {
        double dx = a[3 * k + 0] - gt[3 * k + 0];
        double dy = a[3 * k + 1] - gt[3 * k + 1];
        s += dx * dx + dy * dy;
    }
    return std::sqrt(s / N_POSES);
}

// Project world->pixel for visualization.
static cv::Point2i to_pixel(float x, float y, float scale, int cx, int cy) {
    int px = static_cast<int>(cx + scale * x);
    int py = static_cast<int>(cy - scale * y);
    return cv::Point2i(px, py);
}

static cv::Mat draw_panel(const std::vector<float>& poses,
                          const std::vector<float>& gt,
                          const std::vector<Edge>& loops,
                          const std::string& title,
                          float rmse) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(20, 20, 20));
    float scale = 18.0f;
    int cx = PANEL_W / 2;
    int cy = PANEL_H / 2 + 20;

    // grid
    for (int g = -10; g <= 10; g++) {
        cv::line(img, to_pixel(g * 2, -12, scale, cx, cy),
                       to_pixel(g * 2,  12, scale, cx, cy),
                       cv::Scalar(40, 40, 40), 1);
        cv::line(img, to_pixel(-12, g * 2, scale, cx, cy),
                       to_pixel( 12, g * 2, scale, cx, cy),
                       cv::Scalar(40, 40, 40), 1);
    }

    // ground truth (white)
    for (int k = 1; k < N_POSES; k++) {
        cv::line(img,
                 to_pixel(gt[3 * (k - 1) + 0], gt[3 * (k - 1) + 1], scale, cx, cy),
                 to_pixel(gt[3 * k + 0],       gt[3 * k + 1],       scale, cx, cy),
                 cv::Scalar(180, 180, 180), 1);
    }

    // loop closure edges (cyan, faint)
    for (const auto& lc : loops) {
        cv::line(img,
                 to_pixel(poses[3 * lc.i + 0], poses[3 * lc.i + 1], scale, cx, cy),
                 to_pixel(poses[3 * lc.j + 0], poses[3 * lc.j + 1], scale, cx, cy),
                 cv::Scalar(200, 200, 60), 1);
    }

    // current poses (orange)
    for (int k = 1; k < N_POSES; k++) {
        cv::line(img,
                 to_pixel(poses[3 * (k - 1) + 0], poses[3 * (k - 1) + 1], scale, cx, cy),
                 to_pixel(poses[3 * k + 0],       poses[3 * k + 1],       scale, cx, cy),
                 cv::Scalar(0, 140, 255), 2);
    }
    for (int k = 0; k < N_POSES; k += 8) {
        cv::circle(img, to_pixel(poses[3 * k + 0], poses[3 * k + 1], scale, cx, cy),
                   2, cv::Scalar(50, 200, 255), -1);
    }

    cv::putText(img, title, cv::Point(10, 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "RMSE xy = %.3f m   loops = %zu", rmse, loops.size());
    cv::putText(img, buf, cv::Point(10, PANEL_H - 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1);
    return img;
}

// convert_avi_to_gif moved to include/cuda_video.h (avi_to_gif).

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<float> gt;
    make_ground_truth(gt);

    // Build odometry edges from ground truth + Gaussian noise.
    std::mt19937 rng(7);
    std::normal_distribution<float> n_xy(0.0f, ODOM_SIGMA_XY);
    std::normal_distribution<float> n_th(0.0f, ODOM_SIGMA_TH);

    std::vector<Edge> edges;
    for (int k = 0; k < N_POSES - 1; k++) {
        float ti = gt[3 * k + 2];
        float dxw = gt[3 * (k + 1) + 0] - gt[3 * k + 0];
        float dyw = gt[3 * (k + 1) + 1] - gt[3 * k + 1];
        float zx = dxw * std::cos(ti) + dyw * std::sin(ti) + n_xy(rng);
        float zy = -dxw * std::sin(ti) + dyw * std::cos(ti) + n_xy(rng);
        float zt = wrap_angle(gt[3 * (k + 1) + 2] - gt[3 * k + 2]) + n_th(rng);
        edges.push_back({k, k + 1, zx, zy, zt});
    }
    int n_odom = static_cast<int>(edges.size());

    // Loop-closure detection: ground-truth spatial proximity.
    std::normal_distribution<float> nlc_xy(0.0f, LC_SIGMA_XY);
    std::normal_distribution<float> nlc_th(0.0f, LC_SIGMA_TH);
    std::vector<Edge> loops;
    for (int i = 0; i < N_POSES; i++) {
        for (int j = i + LC_MIN_GAP; j < N_POSES; j++) {
            float dxw = gt[3 * j + 0] - gt[3 * i + 0];
            float dyw = gt[3 * j + 1] - gt[3 * i + 1];
            if (dxw * dxw + dyw * dyw < LC_DIST * LC_DIST) {
                float ti = gt[3 * i + 2];
                float zx = dxw * std::cos(ti) + dyw * std::sin(ti) + nlc_xy(rng);
                float zy = -dxw * std::sin(ti) + dyw * std::cos(ti) + nlc_xy(rng);
                float zt = wrap_angle(gt[3 * j + 2] - gt[3 * i + 2]) + nlc_th(rng);
                Edge e{i, j, zx, zy, zt};
                edges.push_back(e);
                loops.push_back(e);
                if (loops.size() >= 80) goto done_lc;
            }
        }
    }
done_lc:
    int n_edges = static_cast<int>(edges.size());
    int n_lc = static_cast<int>(loops.size());
    std::printf("Pose-graph: %d nodes, %d odom edges, %d loop edges (total %d)\n",
                N_POSES, n_odom, n_lc, n_edges);

    // Initial poses by chaining noisy odometry.
    std::vector<float> initial;
    chain_initial_from_odometry(std::vector<Edge>(edges.begin(), edges.begin() + n_odom),
                                gt, initial);
    std::vector<float> poses_h = initial;

    float rmse_init = pose_rmse(initial, gt);
    std::printf("Initial chained RMSE xy = %.3f m\n", rmse_init);

    // Device buffers
    int n_pose_floats = N_POSES * 3;
    int n_diag_floats = N_POSES * 9;

    int *d_ei = nullptr, *d_ej = nullptr;
    float *d_ez = nullptr, *d_poses = nullptr, *d_b = nullptr, *d_diag = nullptr;
    float *d_dx = nullptr, *d_r = nullptr, *d_z = nullptr, *d_p = nullptr, *d_Ap = nullptr;
    float *d_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ei, n_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ej, n_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ez, n_edges * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_poses, n_pose_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n_pose_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_diag, n_diag_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx, n_pose_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r, n_pose_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, n_pose_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p, n_pose_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n_pose_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch, sizeof(float)));

    std::vector<int> ei(n_edges), ej(n_edges);
    std::vector<float> ez(n_edges * 3);
    for (int e = 0; e < n_edges; e++) {
        ei[e] = edges[e].i;
        ej[e] = edges[e].j;
        ez[3 * e + 0] = edges[e].zx;
        ez[3 * e + 1] = edges[e].zy;
        ez[3 * e + 2] = edges[e].zt;
    }
    CUDA_CHECK(cudaMemcpy(d_ei, ei.data(), n_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ej, ej.data(), n_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ez, ez.data(), n_edges * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_poses, poses_h.data(), n_pose_floats * sizeof(float), cudaMemcpyHostToDevice));

    float omega_xy = 1.0f / (ODOM_SIGMA_XY * ODOM_SIGMA_XY);
    float omega_th = 1.0f / (ODOM_SIGMA_TH * ODOM_SIGMA_TH);
    (void)omega_xy; (void)omega_th;

    // Video writer
    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_pose_graph_slam.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          12, cv::Size(PANEL_W * 2, PANEL_H));

    auto draw_frame = [&](const std::string& title, float rmse) {
        CUDA_CHECK(cudaMemcpy(poses_h.data(), d_poses, n_pose_floats * sizeof(float), cudaMemcpyDeviceToHost));
        cv::Mat left  = draw_panel(initial, gt, loops, "initial chained odometry", rmse_init);
        cv::Mat right = draw_panel(poses_h, gt, loops, title, rmse);
        cv::Mat combined(PANEL_H, PANEL_W * 2, CV_8UC3);
        left.copyTo(combined(cv::Rect(0, 0, PANEL_W, PANEL_H)));
        right.copyTo(combined(cv::Rect(PANEL_W, 0, PANEL_W, PANEL_H)));
        video.write(combined);
    };

    draw_frame("iter 00 (= initial)", rmse_init);

    // Gauss-Newton iterations
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    int blk = 256;
    int blocks_e = (n_edges + blk - 1) / blk;
    int blocks_n = (N_POSES + blk - 1) / blk;
    int blocks_vec = (n_pose_floats + blk - 1) / blk;

    for (int it = 0; it < GN_ITERS; it++) {
        // Compute b, diag (omega_xy = 1, omega_th = 1 for simplicity; both edge types use same weight)
        zero_kernel<<<blocks_vec, blk>>>(n_pose_floats, d_b);
        zero_kernel<<<(n_diag_floats + blk - 1) / blk, blk>>>(n_diag_floats, d_diag);
        assemble_kernel<<<blocks_e, blk>>>(n_edges, d_ei, d_ej, d_ez, d_poses,
                                           omega_xy, omega_th, d_b, d_diag);
        anchor_kernel<<<1, 1>>>(d_b, d_diag);

        // PCG to solve H * dx = -b. Initial dx = 0.
        // r = -b - H * 0 = -b
        zero_kernel<<<blocks_vec, blk>>>(n_pose_floats, d_dx);
        // r = -b
        // (compute as r = 0 - b)
        zero_kernel<<<blocks_vec, blk>>>(n_pose_floats, d_r);
        // r += -1 * b
        axpy_kernel<<<blocks_vec, blk>>>(n_pose_floats, -1.0f, d_b, d_r);
        // anchor row 0 of r
        zero_anchor_dx<<<1, 1>>>(d_r);
        // z = M^-1 * r
        apply_precond_kernel<<<blocks_n, blk>>>(N_POSES, d_diag, d_r, d_z);
        // p = z
        copy_kernel<<<blocks_vec, blk>>>(n_pose_floats, d_z, d_p);

        // rz_old = r . z
        float rz_old = 0.0f;
        CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
        dot_kernel<<<32, 256>>>(n_pose_floats, d_r, d_z, d_scratch);
        CUDA_CHECK(cudaMemcpy(&rz_old, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));

        for (int k = 0; k < PCG_ITERS; k++) {
            zero_kernel<<<blocks_vec, blk>>>(n_pose_floats, d_Ap);
            matvec_kernel<<<blocks_e, blk>>>(n_edges, d_ei, d_ej, d_ez, d_poses,
                                              omega_xy, omega_th, d_p, d_Ap);
            zero_anchor_dx<<<1, 1>>>(d_Ap);  // anchor: row 0 of Ap = 0 contribution

            float pAp = 0.0f;
            CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
            dot_kernel<<<32, 256>>>(n_pose_floats, d_p, d_Ap, d_scratch);
            CUDA_CHECK(cudaMemcpy(&pAp, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
            if (pAp <= 0.0f) break;

            float alpha = rz_old / pAp;
            axpy_kernel<<<blocks_vec, blk>>>(n_pose_floats, alpha, d_p, d_dx);
            axpy_kernel<<<blocks_vec, blk>>>(n_pose_floats, -alpha, d_Ap, d_r);

            float rr = 0.0f;
            CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
            dot_kernel<<<32, 256>>>(n_pose_floats, d_r, d_r, d_scratch);
            CUDA_CHECK(cudaMemcpy(&rr, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
            if (rr < PCG_TOL) break;

            apply_precond_kernel<<<blocks_n, blk>>>(N_POSES, d_diag, d_r, d_z);
            float rz_new = 0.0f;
            CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
            dot_kernel<<<32, 256>>>(n_pose_floats, d_r, d_z, d_scratch);
            CUDA_CHECK(cudaMemcpy(&rz_new, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
            float beta = rz_new / rz_old;
            xpay_kernel<<<blocks_vec, blk>>>(n_pose_floats, beta, d_z, d_p);
            rz_old = rz_new;
        }

        zero_anchor_dx<<<1, 1>>>(d_dx);
        update_poses_kernel<<<blocks_n, blk>>>(N_POSES, d_poses, d_dx);

        // Frame + log
        CUDA_CHECK(cudaMemcpy(poses_h.data(), d_poses, n_pose_floats * sizeof(float), cudaMemcpyDeviceToHost));
        float rmse = pose_rmse(poses_h, gt);
        if (it < 5 || it % 5 == 0) std::printf("  iter %02d  RMSE=%.3f m\n", it + 1, rmse);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "GN iter %02d", it + 1);
        draw_frame(buf, rmse);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0.0f; cudaEventElapsedTime(&ms, t0, t1);

    CUDA_CHECK(cudaMemcpy(poses_h.data(), d_poses, n_pose_floats * sizeof(float), cudaMemcpyDeviceToHost));
    float rmse_final = pose_rmse(poses_h, gt);
    std::printf("GN done.  RMSE init=%.3f -> final=%.3f m  (%d iters, total %.2f ms = %.2f ms/iter)\n",
                rmse_init, rmse_final, GN_ITERS, ms, ms / GN_ITERS);
    // Debug: print a few poses
    for (int idx : {0, 50, 100, 150, 199}) {
        std::printf("  pose %3d: opt (%.2f, %.2f, %.2f)  GT (%.2f, %.2f, %.2f)  init (%.2f, %.2f, %.2f)\n",
                    idx,
                    poses_h[3 * idx + 0], poses_h[3 * idx + 1], poses_h[3 * idx + 2],
                    gt[3 * idx + 0],      gt[3 * idx + 1],      gt[3 * idx + 2],
                    initial[3 * idx + 0], initial[3 * idx + 1], initial[3 * idx + 2]);
    }

    // Final frames
    for (int k = 0; k < 20; k++) draw_frame("converged", rmse_final);
    video.release();
    cudabot::avi_to_gif("gif/gpu_pose_graph_slam.avi", "gif/gpu_pose_graph_slam.gif", 12, 1080);
    std::printf("GIF saved to gif/gpu_pose_graph_slam.gif\n");

    cudaFree(d_ei); cudaFree(d_ej); cudaFree(d_ez); cudaFree(d_poses);
    cudaFree(d_b); cudaFree(d_diag); cudaFree(d_dx);
    cudaFree(d_r); cudaFree(d_z); cudaFree(d_p); cudaFree(d_Ap);
    cudaFree(d_scratch);
    return 0;
}
