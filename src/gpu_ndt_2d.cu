// gpu_ndt_2d.cu
//
// GPU 2D NDT (Normal Distributions Transform) scan matching.
//
// Algorithm:
//   - Build NDT grid from a "map" scan: per grid cell, store mean mu and
//     2x2 covariance Sigma of the points falling in that cell.
//   - For a "live" scan, optimize the 2D pose (tx, ty, theta) so that
//     transformed live points fall on high-likelihood NDT cells.
//   - Score per point: s = exp(-0.5 d^T Sigma^-1 d) where d = T(p) - mu.
//   - Gauss-Newton on negative log-likelihood:
//       g(p) = -ln(eta + s)  (with small eta for numerical floor)
//       gradient + Hessian computed analytically over (tx, ty, theta).
//
// Demo:
//   - Generate a ground-truth scan from a room with several rectangular
//     obstacles plus the outer walls.
//   - Apply a random pose perturbation to generate the "live" scan.
//   - Run NDT to recover the perturbation pose.
//   - Visualise: map scan (light), live scan before align (red), live scan
//     after align (green), NDT cell ellipses, GT vs estimated pose.
//
// Constants are picked so the demo runs 60+ scenarios per second on GPU.

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

// World extent: a 40 x 40 m room centred at (0, 0).
constexpr float WORLD_HALF = 20.0f;
constexpr int   N_RAYS     = 720;
constexpr float MAX_RANGE  = 25.0f;
constexpr int   N_CELLS_X  = 20;   // 2 m cells
constexpr int   N_CELLS_Y  = 20;
constexpr float CELL_SIZE  = (2.0f * WORLD_HALF) / N_CELLS_X;
constexpr int   N_FRAMES   = 80;
constexpr int   GN_ITERS   = 20;
constexpr int   PANEL_W    = 620;
constexpr int   PANEL_H    = 620;
constexpr float NDT_EPS    = 0.10f;
constexpr int   MIN_PTS_PER_CELL = 4;

// Obstacle: axis-aligned rectangles inside the room.
struct Rect { float xmin, ymin, xmax, ymax; };
__constant__ Rect d_rects[8];
__constant__ int  d_n_rects = 0;  // overwritten via cudaMemcpyToSymbol

// --- Scan generation ----------------------------------------------------
__device__ float ray_distance_to_rect(float ox, float oy, float dx, float dy,
                                      const Rect& r) {
    // Slab method for AABB ray intersection.
    float tmin = 0.0f, tmax = MAX_RANGE;
    if (fabsf(dx) > 1e-6f) {
        float t1 = (r.xmin - ox) / dx;
        float t2 = (r.xmax - ox) / dx;
        if (t1 > t2) { float t = t1; t1 = t2; t2 = t; }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
    } else if (ox < r.xmin || ox > r.xmax) {
        return MAX_RANGE;
    }
    if (fabsf(dy) > 1e-6f) {
        float t1 = (r.ymin - oy) / dy;
        float t2 = (r.ymax - oy) / dy;
        if (t1 > t2) { float t = t1; t1 = t2; t2 = t; }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
    } else if (oy < r.ymin || oy > r.ymax) {
        return MAX_RANGE;
    }
    if (tmin > tmax || tmax < 0.0f) return MAX_RANGE;
    return fmaxf(tmin, 0.0f);
}

__global__ void raycast_kernel(float ox, float oy, float yaw,
                               int n_rects, float world_half,
                               float* __restrict__ out_x,
                               float* __restrict__ out_y,
                               unsigned char* __restrict__ out_hit) {
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= N_RAYS) return;
    float a = (float)rid / N_RAYS * 2.0f * (float)M_PI + yaw;
    float dx = cosf(a), dy = sinf(a);
    // Outer walls: 4 rectangles surrounding the room (thin slabs outside walls).
    float best = MAX_RANGE;
    // Outer wall: just clip to world bounds.
    if (fabsf(dx) > 1e-6f) {
        float tx = (dx > 0 ? (world_half - ox) : (-world_half - ox)) / dx;
        if (tx > 0) best = fminf(best, tx);
    }
    if (fabsf(dy) > 1e-6f) {
        float ty = (dy > 0 ? (world_half - oy) : (-world_half - oy)) / dy;
        if (ty > 0) best = fminf(best, ty);
    }
    // Inner obstacles.
    for (int k = 0; k < n_rects; k++) {
        float d = ray_distance_to_rect(ox, oy, dx, dy, d_rects[k]);
        best = fminf(best, d);
    }
    if (best >= MAX_RANGE) {
        out_hit[rid] = 0;
        out_x[rid] = 0.0f; out_y[rid] = 0.0f;
    } else {
        out_hit[rid] = 1;
        // Express the hit in SENSOR frame, not world.
        // sensor frame = world rotated by -yaw and translated by -(ox, oy).
        float hx_w = ox + best * dx;
        float hy_w = oy + best * dy;
        float cx = hx_w - ox, cy = hy_w - oy;
        float c = cosf(-yaw), s = sinf(-yaw);
        out_x[rid] = c * cx - s * cy;
        out_y[rid] = s * cx + c * cy;
    }
}

__global__ void add_noise_kernel(int n, unsigned long long seed,
                                 float sigma,
                                 float* x, float* y, unsigned char* hit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!hit[i]) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, i, 0, &st);
    float nx = curand_normal(&st) * sigma;
    float ny = curand_normal(&st) * sigma;
    x[i] += nx; y[i] += ny;
}

// --- NDT grid build -----------------------------------------------------
__global__ void accum_grid_kernel(int n, const float* x, const float* y,
                                  const unsigned char* hit, float world_half,
                                  float cell_size, int gw, int gh,
                                  // accumulators (per cell):
                                  //   sum_x, sum_y, sum_xx, sum_yy, sum_xy, count
                                  float* sum_x, float* sum_y,
                                  float* sum_xx, float* sum_yy, float* sum_xy,
                                  int* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!hit[i]) return;
    float px = x[i], py = y[i];
    int cx = static_cast<int>((px + world_half) / cell_size);
    int cy = static_cast<int>((py + world_half) / cell_size);
    if (cx < 0 || cx >= gw || cy < 0 || cy >= gh) return;
    int idx = cy * gw + cx;
    atomicAdd(&sum_x[idx], px);
    atomicAdd(&sum_y[idx], py);
    atomicAdd(&sum_xx[idx], px * px);
    atomicAdd(&sum_yy[idx], py * py);
    atomicAdd(&sum_xy[idx], px * py);
    atomicAdd(&count[idx], 1);
}

// Compute per-cell (mu, Sigma, Sigma^-1, valid flag).  Cells with < MIN_PTS
// are marked invalid.
__global__ void finalize_grid_kernel(int n_cells,
                                     const float* sum_x, const float* sum_y,
                                     const float* sum_xx, const float* sum_yy,
                                     const float* sum_xy, const int* count,
                                     float* mu_x, float* mu_y,
                                     float* inv00, float* inv01, float* inv11,
                                     unsigned char* valid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    int c = count[idx];
    if (c < MIN_PTS_PER_CELL) { valid[idx] = 0; return; }
    float inv_n = 1.0f / static_cast<float>(c);
    float mx = sum_x[idx] * inv_n;
    float my = sum_y[idx] * inv_n;
    float vxx = sum_xx[idx] * inv_n - mx * mx;
    float vyy = sum_yy[idx] * inv_n - my * my;
    float vxy = sum_xy[idx] * inv_n - mx * my;
    // Regularise: add small isotropic noise.
    vxx += 0.15f;
    vyy += 0.15f;
    float det = vxx * vyy - vxy * vxy;
    if (det < 1e-9f) { valid[idx] = 0; return; }
    float inv_det = 1.0f / det;
    mu_x[idx] = mx; mu_y[idx] = my;
    inv00[idx] =  vyy * inv_det;
    inv01[idx] = -vxy * inv_det;
    inv11[idx] =  vxx * inv_det;
    valid[idx] = 1;
}

// --- NDT score + Gauss-Newton accumulators ------------------------------
// One thread per scan point.  Accumulates b (3) and H (3x3 upper-triangular,
// stored as 6 floats: H00, H01, H02, H11, H12, H22) plus negative log-likelihood.
//
// Pose: x = (tx, ty, theta).  Live point in sensor frame -> world:
//     w = R(theta) * p + t,  J_w_x = (1, 0, -sin(theta)*px - cos(theta)*py)
//                             J_w_y = (0, 1,  cos(theta)*px - sin(theta)*py)
// Residual d = w - mu, score s = exp(-0.5 d^T S d).
// f = -ln(eta + s),  df/dx = -1/(eta+s) * d s/dx
// ds/dx = s * (-S d)^T * J_w
// So df/dx = (s/(eta+s)) * d^T S J_w  (gradient column vector of length 3)
// We approximate H with the Gauss-Newton form for negative log-likelihood:
//     H ~ (s/(eta+s))^2 * (J_w^T S^T d) (d^T S J_w) + ...
// For simplicity and robustness we use a positive-semi-definite approximation:
//     H ~ (s/(eta+s)) * J_w^T S J_w
// which corresponds to the classical NDT Newton Hessian dropped to its PSD core.

__global__ void ndt_grad_hess_kernel(int n,
                                     const float* px, const float* py,
                                     const unsigned char* hit,
                                     float tx, float ty, float theta,
                                     float world_half, float cell_size,
                                     int gw, int gh,
                                     const float* mu_x, const float* mu_y,
                                     const float* inv00, const float* inv01,
                                     const float* inv11,
                                     const unsigned char* valid,
                                     float* g, float* H, float* score_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!hit[i]) return;
    float c = cosf(theta), s = sinf(theta);
    float wpx = c * px[i] - s * py[i] + tx;
    float wpy = s * px[i] + c * py[i] + ty;
    int cx = static_cast<int>((wpx + world_half) / cell_size);
    int cy = static_cast<int>((wpy + world_half) / cell_size);
    if (cx < 0 || cx >= gw || cy < 0 || cy >= gh) return;
    int idx = cy * gw + cx;
    if (!valid[idx]) return;
    float dx = wpx - mu_x[idx];
    float dy = wpy - mu_y[idx];
    float S00 = inv00[idx], S01 = inv01[idx], S11 = inv11[idx];
    // q = S d
    float qx = S00 * dx + S01 * dy;
    float qy = S01 * dx + S11 * dy;
    float quad = dx * qx + dy * qy;
    float score = expf(-0.5f * quad);
    atomicAdd(score_out, score);
    float w = score / (NDT_EPS + score);
    // J_w (2x3): cols (tx, ty, theta)
    //   row x: ( 1, 0, -s*px - c*py )
    //   row y: ( 0, 1,  c*px - s*py )
    float Jx_theta = -s * px[i] - c * py[i];
    float Jy_theta =  c * px[i] - s * py[i];
    // gradient g_k = w * d^T S J_w[:, k] = w * (qx * J_x_k + qy * J_y_k)
    float gtx = w * qx;
    float gty = w * qy;
    float gth = w * (qx * Jx_theta + qy * Jy_theta);
    atomicAdd(&g[0], gtx);
    atomicAdd(&g[1], gty);
    atomicAdd(&g[2], gth);
    // PSD GN Hessian: H = w * J_w^T S J_w
    // Compute u = S * J_w[:, k] for each k, then H[i, j] = J_w[:, i]^T * u_j.
    // Equivalent expansion:
    //   S J_w[:, 0] = (S00, S01)
    //   S J_w[:, 1] = (S01, S11)
    //   S J_w[:, 2] = (S00*Jx_theta + S01*Jy_theta, S01*Jx_theta + S11*Jy_theta)
    float u20x = S00 * Jx_theta + S01 * Jy_theta;
    float u20y = S01 * Jx_theta + S11 * Jy_theta;
    float H00 = w * S00;
    float H01 = w * S01;
    float H02 = w * u20x;            // J_w[:, 0] = (1, 0)
    float H11 = w * S11;
    float H12 = w * u20y;            // J_w[:, 1] = (0, 1)
    float H22 = w * (Jx_theta * u20x + Jy_theta * u20y);
    atomicAdd(&H[0], H00);
    atomicAdd(&H[1], H01);
    atomicAdd(&H[2], H02);
    atomicAdd(&H[3], H11);
    atomicAdd(&H[4], H12);
    atomicAdd(&H[5], H22);
}

// --- Host helpers --------------------------------------------------------
static void invert3x3_sym(const float* H6, float lambda, float* out9) {
    // H6 = (H00, H01, H02, H11, H12, H22) stored upper-tri.
    float a = H6[0] + lambda;
    float b = H6[1];
    float c = H6[2];
    float d = H6[3] + lambda;
    float e = H6[4];
    float f = H6[5] + lambda;
    float det = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c);
    if (fabsf(det) < 1e-12f) det = (det < 0 ? -1e-12f : 1e-12f);
    float inv_det = 1.0f / det;
    out9[0] = (d * f - e * e) * inv_det;
    out9[1] = (c * e - b * f) * inv_det;
    out9[2] = (b * e - c * d) * inv_det;
    out9[3] = out9[1];
    out9[4] = (a * f - c * c) * inv_det;
    out9[5] = (b * c - a * e) * inv_det;
    out9[6] = out9[2];
    out9[7] = out9[5];
    out9[8] = (a * d - b * b) * inv_det;
}

// --- Visualisation -------------------------------------------------------
static cv::Point2i world_to_panel(float x, float y) {
    float scale = PANEL_W / (2.0f * WORLD_HALF);
    int px = static_cast<int>(PANEL_W / 2 + scale * x);
    int py = static_cast<int>(PANEL_H / 2 - scale * y);
    return cv::Point2i(px, py);
}

static void draw_ellipse(cv::Mat& img, float mx, float my,
                         float vxx, float vyy, float vxy, cv::Scalar color) {
    // Eigen decomposition of 2x2 covariance.
    float tr = vxx + vyy;
    float det = vxx * vyy - vxy * vxy;
    float disc = std::sqrt(std::max(0.0f, 0.25f * tr * tr - det));
    float l1 = 0.5f * tr + disc;
    float l2 = 0.5f * tr - disc;
    if (l1 < 1e-6f) l1 = 1e-6f;
    if (l2 < 1e-6f) l2 = 1e-6f;
    float ang = 0.5f * std::atan2(2.0f * vxy, vxx - vyy);
    float scale = PANEL_W / (2.0f * WORLD_HALF);
    cv::Point2i c = world_to_panel(mx, my);
    cv::Size axes(static_cast<int>(scale * std::sqrt(l1)),
                  static_cast<int>(scale * std::sqrt(l2)));
    if (axes.width < 1 || axes.height < 1) return;
    cv::ellipse(img, c, axes, -ang * 180.0f / static_cast<float>(M_PI),
                0.0f, 360.0f, color, 1);
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    // Set up rectangles (obstacle layout).
    std::vector<Rect> rects = {
        {-12.0f, -10.0f, -7.0f, -3.0f},
        { -2.0f,  -8.0f,  3.0f, -2.0f},
        {  6.0f,   2.0f, 11.0f,  6.0f},
        { -9.0f,   4.0f, -3.0f, 10.0f},
        {  9.0f,  -8.0f, 13.0f, -3.0f},
        {-15.0f,  10.0f, -8.0f, 14.0f},
        {  4.0f, -14.0f,  9.0f, -10.0f},
        {-14.0f,  -4.0f, -11.0f,  2.0f}
    };
    int n_rects = static_cast<int>(rects.size());
    CUDA_CHECK(cudaMemcpyToSymbol(d_rects, rects.data(),
                                  n_rects * sizeof(Rect)));

    // Device buffers
    float *d_map_x, *d_map_y;     unsigned char *d_map_hit;
    float *d_live_x, *d_live_y;   unsigned char *d_live_hit;
    CUDA_CHECK(cudaMalloc(&d_map_x,   N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_map_y,   N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_map_hit, N_RAYS));
    CUDA_CHECK(cudaMalloc(&d_live_x,  N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_live_y,  N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_live_hit, N_RAYS));

    int n_cells = N_CELLS_X * N_CELLS_Y;
    float *d_sum_x, *d_sum_y, *d_sum_xx, *d_sum_yy, *d_sum_xy;
    int   *d_count;
    float *d_mu_x, *d_mu_y, *d_inv00, *d_inv01, *d_inv11;
    unsigned char* d_valid;
    CUDA_CHECK(cudaMalloc(&d_sum_x, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_y, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_xx, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_yy, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_xy, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_count, n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mu_x, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mu_y, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv00, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv01, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv11, n_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid, n_cells));

    float *d_g, *d_H, *d_score;
    CUDA_CHECK(cudaMalloc(&d_g, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_score, sizeof(float)));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uni_xy(-1.0f, 1.0f);
    std::uniform_real_distribution<float> uni_th(-0.35f, 0.35f);

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_ndt_2d.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          10, cv::Size(PANEL_W, PANEL_H + 50));

    int blocks_rays = (N_RAYS + 255) / 256;
    int blocks_cells = (n_cells + 255) / 256;

    double total_ms = 0.0;
    int count_frames = 0;
    double total_pose_err = 0.0;
    double total_yaw_err = 0.0;

    std::vector<float> h_map_x(N_RAYS), h_map_y(N_RAYS);
    std::vector<unsigned char> h_map_hit(N_RAYS);
    std::vector<float> h_live_x(N_RAYS), h_live_y(N_RAYS);
    std::vector<unsigned char> h_live_hit(N_RAYS);
    std::vector<float> h_mu_x(n_cells), h_mu_y(n_cells);
    std::vector<float> h_v00(n_cells), h_v11(n_cells), h_v01(n_cells);
    std::vector<unsigned char> h_valid(n_cells);

    for (int frame = 0; frame < N_FRAMES; frame++) {
        // Choose a sensor position somewhere inside the room (avoiding obstacles).
        float ox = 0.0f, oy = 0.0f;
        for (int tries = 0; tries < 64; tries++) {
            float t = static_cast<float>(frame) / N_FRAMES * 2.0f * static_cast<float>(M_PI);
            float jitter_x = std::cos(t * 1.7f + tries) * 10.0f;
            float jitter_y = std::sin(t * 1.2f + tries * 0.3f) * 10.0f;
            ox = jitter_x;
            oy = jitter_y;
            bool inside = false;
            for (const auto& r : rects) {
                if (ox > r.xmin && ox < r.xmax && oy > r.ymin && oy < r.ymax) {
                    inside = true; break;
                }
            }
            if (!inside) break;
        }

        // 1. Map scan at (ox, oy, 0).
        raycast_kernel<<<blocks_rays, 256>>>(ox, oy, 0.0f, n_rects, WORLD_HALF,
                                              d_map_x, d_map_y, d_map_hit);
        add_noise_kernel<<<blocks_rays, 256>>>(N_RAYS, frame * 13ULL + 1,
                                               0.02f, d_map_x, d_map_y, d_map_hit);

        // 2. Convert map scan to WORLD points (apply identity for map, since
        //    we'll align the live scan into the map's world frame).
        //    The raycast already returns sensor-frame coords, but for the map
        //    we want world-frame -> rotate by 0 + translate by (ox, oy).
        CUDA_CHECK(cudaMemset(d_sum_x, 0, n_cells * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sum_y, 0, n_cells * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sum_xx, 0, n_cells * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sum_yy, 0, n_cells * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sum_xy, 0, n_cells * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_count, 0, n_cells * sizeof(int)));

        // Treat the map sensor frame as the world frame for matching, so we
        // can accumulate the NDT grid in sensor coords too.  The "live" scan
        // will be expressed in its OWN sensor frame and we recover the
        // sensor->map relative pose.
        accum_grid_kernel<<<blocks_rays, 256>>>(N_RAYS, d_map_x, d_map_y, d_map_hit,
                                                 WORLD_HALF, CELL_SIZE,
                                                 N_CELLS_X, N_CELLS_Y,
                                                 d_sum_x, d_sum_y, d_sum_xx,
                                                 d_sum_yy, d_sum_xy, d_count);
        finalize_grid_kernel<<<blocks_cells, 256>>>(n_cells, d_sum_x, d_sum_y,
                                                     d_sum_xx, d_sum_yy, d_sum_xy,
                                                     d_count, d_mu_x, d_mu_y,
                                                     d_inv00, d_inv01, d_inv11,
                                                     d_valid);

        // 3. Live scan: same sensor but with a random ground-truth perturbation.
        float gt_dx = uni_xy(rng);
        float gt_dy = uni_xy(rng);
        float gt_dth = uni_th(rng);
        // The "true" live sensor pose is (ox + dx', oy + dy', dth) where the
        // world translation is rotated by the ROBOT's yaw - we'll use a model
        // where the perturbation is purely an additive sensor-frame offset:
        //   live_sensor_world_pose = map_sensor_world_pose * Exp(perturb)
        // For simplicity, use additive world-frame translation:
        float live_ox = ox + gt_dx;
        float live_oy = oy + gt_dy;
        float live_yaw = gt_dth;
        raycast_kernel<<<blocks_rays, 256>>>(live_ox, live_oy, live_yaw,
                                              n_rects, WORLD_HALF,
                                              d_live_x, d_live_y, d_live_hit);
        add_noise_kernel<<<blocks_rays, 256>>>(N_RAYS, frame * 17ULL + 7,
                                               0.02f, d_live_x, d_live_y, d_live_hit);

        // 4. NDT GN: optimize (tx, ty, theta) to transform live scan into map
        //    sensor frame.  The transform takes live SENSOR-frame point to
        //    map SENSOR-frame point:
        //       world: w = T_live_world * p_live = (R(live_yaw) p + (live_ox, live_oy))
        //       map sensor frame: m = R(0)^T * (w - (ox, oy)) = w - (ox, oy)
        //    so the optimal (tx, ty, theta) corresponds to:
        //       theta_opt = live_yaw
        //       (tx_opt, ty_opt) = (live_ox - ox, live_oy - oy) = (gt_dx, gt_dy)
        //    GT theta = live_yaw = gt_dth, GT tx/ty = (gt_dx, gt_dy).

        float tx = 0.0f, ty = 0.0f, th = 0.0f;
        float lambda = 0.0f;

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int it = 0; it < GN_ITERS; it++) {
            CUDA_CHECK(cudaMemset(d_g, 0, 3 * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_H, 0, 6 * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_score, 0, sizeof(float)));
            ndt_grad_hess_kernel<<<blocks_rays, 256>>>(N_RAYS, d_live_x, d_live_y,
                                                       d_live_hit,
                                                       tx, ty, th,
                                                       WORLD_HALF, CELL_SIZE,
                                                       N_CELLS_X, N_CELLS_Y,
                                                       d_mu_x, d_mu_y,
                                                       d_inv00, d_inv01, d_inv11,
                                                       d_valid,
                                                       d_g, d_H, d_score);
            float h_g[3], h_H[6], h_score = 0.0f;
            CUDA_CHECK(cudaMemcpy(h_g, d_g, 3 * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_H, d_H, 6 * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_score, d_score, sizeof(float), cudaMemcpyDeviceToHost));
            float H_inv[9];
            invert3x3_sym(h_H, lambda + 1e-3f, H_inv);
            float dtx = -(H_inv[0] * h_g[0] + H_inv[1] * h_g[1] + H_inv[2] * h_g[2]);
            float dty = -(H_inv[3] * h_g[0] + H_inv[4] * h_g[1] + H_inv[5] * h_g[2]);
            float dth = -(H_inv[6] * h_g[0] + H_inv[7] * h_g[1] + H_inv[8] * h_g[2]);
            // Clip step magnitude.
            float step_norm = std::sqrt(dtx * dtx + dty * dty);
            if (step_norm > 1.0f) {
                dtx *= 1.0f / step_norm;
                dty *= 1.0f / step_norm;
            }
            if (std::fabs(dth) > 0.3f) dth = (dth > 0 ? 0.3f : -0.3f);
            tx += dtx; ty += dty; th += dth;
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms = 0.0f; cudaEventElapsedTime(&ms, t0, t1);

        float err_t = std::sqrt((tx - gt_dx) * (tx - gt_dx) + (ty - gt_dy) * (ty - gt_dy));
        float err_th = std::fabs(th - gt_dth);
        if (err_th > static_cast<float>(M_PI))
            err_th = 2.0f * static_cast<float>(M_PI) - err_th;
        total_ms += ms;
        total_pose_err += err_t;
        total_yaw_err += err_th;
        count_frames++;
        if (frame < 5 || frame % 20 == 0)
            std::printf("frame %3d  gt=(%+.2f, %+.2f, %+.2f)  est=(%+.2f, %+.2f, %+.2f)  err=(%.3f m, %.3f rad)  %.2f ms\n",
                        frame, gt_dx, gt_dy, gt_dth, tx, ty, th, err_t, err_th, ms);

        // --- Visualisation ---
        CUDA_CHECK(cudaMemcpy(h_map_x.data(),  d_map_x,  N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_map_y.data(),  d_map_y,  N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_map_hit.data(),d_map_hit,N_RAYS, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_live_x.data(), d_live_x, N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_live_y.data(), d_live_y, N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_live_hit.data(),d_live_hit,N_RAYS, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mu_x.data(), d_mu_x, n_cells * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mu_y.data(), d_mu_y, n_cells * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_v00.data(), d_inv00, n_cells * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_v01.data(), d_inv01, n_cells * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_v11.data(), d_inv11, n_cells * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_valid.data(), d_valid, n_cells, cudaMemcpyDeviceToHost));

        cv::Mat img(PANEL_H + 50, PANEL_W, CV_8UC3, cv::Scalar(18, 18, 18));
        // Outer wall + obstacles
        for (const auto& r : rects) {
            cv::rectangle(img, world_to_panel(r.xmin, r.ymax),
                          world_to_panel(r.xmax, r.ymin),
                          cv::Scalar(75, 75, 75), -1);
        }
        cv::rectangle(img, world_to_panel(-WORLD_HALF, WORLD_HALF),
                      world_to_panel(WORLD_HALF, -WORLD_HALF),
                      cv::Scalar(70, 70, 70), 1);

        // NDT cell ellipses (mu, Sigma).  We need covariance, but stored is
        // inverse covariance — recover by inverting 2x2 again.
        for (int k = 0; k < n_cells; k++) {
            if (!h_valid[k]) continue;
            float a = h_v00[k], b = h_v01[k], c = h_v11[k];
            float det = a * c - b * b;
            if (det < 1e-6f) continue;
            float inv_det = 1.0f / det;
            float vxx =  c * inv_det;
            float vxy = -b * inv_det;
            float vyy =  a * inv_det;
            draw_ellipse(img, h_mu_x[k], h_mu_y[k], vxx, vyy, vxy,
                         cv::Scalar(70, 160, 230));
        }

        // Map scan: world coords = (ox, oy) + sensor
        for (int k = 0; k < N_RAYS; k++) {
            if (!h_map_hit[k]) continue;
            cv::circle(img, world_to_panel(ox + h_map_x[k], oy + h_map_y[k]),
                       1, cv::Scalar(180, 180, 180), -1);
        }

        // Live scan in INITIAL (unaligned) frame: live sensor pose is the GT,
        // but the NDT is fitting in the MAP sensor frame.  So we draw the live
        // points at (ox + live_sensor_point) — i.e., assume identity transform.
        for (int k = 0; k < N_RAYS; k++) {
            if (!h_live_hit[k]) continue;
            cv::circle(img, world_to_panel(ox + h_live_x[k], oy + h_live_y[k]),
                       1, cv::Scalar(50, 50, 220), -1);
        }
        // Live scan after NDT alignment: apply T(tx, ty, theta) to each point.
        float ce = std::cos(th), se = std::sin(th);
        for (int k = 0; k < N_RAYS; k++) {
            if (!h_live_hit[k]) continue;
            float wx = ce * h_live_x[k] - se * h_live_y[k] + tx;
            float wy = se * h_live_x[k] + ce * h_live_y[k] + ty;
            cv::circle(img, world_to_panel(ox + wx, oy + wy),
                       1, cv::Scalar(60, 220, 80), -1);
        }

        // Sensor markers
        cv::circle(img, world_to_panel(ox, oy), 4, cv::Scalar(255, 255, 255), -1);
        cv::Point2i s_est = world_to_panel(ox + tx, oy + ty);
        cv::circle(img, s_est, 4, cv::Scalar(60, 220, 80), -1);
        cv::Point2i s_gt = world_to_panel(ox + gt_dx, oy + gt_dy);
        cv::drawMarker(img, s_gt, cv::Scalar(255, 200, 80),
                       cv::MARKER_CROSS, 12, 2);

        cv::putText(img, cv::format("GPU NDT 2D scan matching  frame %d", frame),
                    cv::Point(10, 24),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
        cv::putText(img, cv::format("gt = (%+.2f, %+.2f, %+.2f)   est = (%+.2f, %+.2f, %+.2f)",
                                    gt_dx, gt_dy, gt_dth, tx, ty, th),
                    cv::Point(10, PANEL_H + 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220), 1);
        cv::putText(img, cv::format("err t = %.3f m   err yaw = %.3f rad   %.2f ms / scenario",
                                    err_t, err_th, ms),
                    cv::Point(10, PANEL_H + 38),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(180, 220, 180), 1);
        cv::putText(img, "white=map  red=live(unaligned)  green=live(aligned)  blue=NDT cells",
                    cv::Point(10, 46),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180, 180, 180), 1);
        video.write(img);
    }
    video.release();

    std::printf("Avg pose err = %.4f m   avg yaw err = %.4f rad   avg %.2f ms/frame\n",
                total_pose_err / count_frames,
                total_yaw_err / count_frames,
                total_ms / count_frames);

    cudabot::avi_to_gif("gif/gpu_ndt_2d.avi", "gif/gpu_ndt_2d.gif", 12, 720);
    std::printf("GIF saved to gif/gpu_ndt_2d.gif\n");

    cudaFree(d_map_x); cudaFree(d_map_y); cudaFree(d_map_hit);
    cudaFree(d_live_x); cudaFree(d_live_y); cudaFree(d_live_hit);
    cudaFree(d_sum_x); cudaFree(d_sum_y); cudaFree(d_sum_xx);
    cudaFree(d_sum_yy); cudaFree(d_sum_xy); cudaFree(d_count);
    cudaFree(d_mu_x); cudaFree(d_mu_y); cudaFree(d_inv00);
    cudaFree(d_inv01); cudaFree(d_inv11); cudaFree(d_valid);
    cudaFree(d_g); cudaFree(d_H); cudaFree(d_score);
    return 0;
}
