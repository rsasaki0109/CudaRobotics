// gpu_gicp_2d.cu
//
// GPU 2D GICP (Generalized ICP) scan matching.
//
// Algorithm (Segal et al., "Generalized-ICP", RSS 2009, 2D specialisation):
//   - For each scan point compute the local sample covariance from its
//     k-nearest neighbours in the same scan.
//   - Eigendecompose the 2x2 covariance, identify the surface normal
//     direction (smallest eigenvalue), and replace the covariance with
//     I - (1 - eps) * n n^T.  This yields a "disk" covariance whose
//     inverse penalises displacement along the normal much more than
//     along the tangent — i.e. effective point-to-line cost.
//   - Match each source point to its nearest neighbour in the target.
//   - Per match the per-correspondence weight matrix is
//         M = (C_t + R C_s R^T)^{-1}                 (2x2 SPD)
//     and the residual r = R(theta) p_s + t - p_t.
//   - Gauss-Newton on
//         F = sum_i 0.5 r_i^T M_i r_i
//     with J_i = [ I_2 | R'(theta) p_s ]               (2x3)
//     accumulating H = sum J^T M J and b = sum J^T M r.
//   - 3x3 SPD solve with Levenberg-Marquardt damping.
//
// Compared with NDT (gpu_ndt_2d.cu) GICP forms one correspondence per
// source point against the nearest target point, rather than against a
// grid cell.  The covariance regularisation is per-point and reflects the
// actual local geometry, which makes the cost behave like a point-to-line
// metric on the underlying surface.
//
// Demo: same room as gpu_ndt_2d.cu (40 x 40 m, 8 obstacles), 80 frames,
// random ground-truth perturbation, recover the relative pose.

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

constexpr float WORLD_HALF   = 20.0f;
constexpr int   N_RAYS       = 720;
constexpr float MAX_RANGE    = 25.0f;
constexpr int   K_NEIGHBORS  = 10;
constexpr float GICP_EPS     = 1e-2f;   // normal-direction "squash"
constexpr float MATCH_MAX_D2 = 2.0f * 2.0f;
constexpr int   N_FRAMES     = 80;
constexpr int   GN_ITERS     = 15;
constexpr int   PANEL_W      = 620;
constexpr int   PANEL_H      = 620;

struct Rect { float xmin, ymin, xmax, ymax; };
__constant__ Rect d_rects[8];

// --- Scan generation (identical to NDT 2D) ------------------------------
__device__ float ray_distance_to_rect(float ox, float oy, float dx, float dy,
                                      const Rect& r) {
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

__global__ void raycast_kernel(float ox, float oy, float yaw, int n_rects,
                               float world_half,
                               float* __restrict__ out_x,
                               float* __restrict__ out_y,
                               unsigned char* __restrict__ out_hit) {
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= N_RAYS) return;
    float a = (float)rid / N_RAYS * 2.0f * (float)M_PI + yaw;
    float dx = cosf(a), dy = sinf(a);
    float best = MAX_RANGE;
    if (fabsf(dx) > 1e-6f) {
        float tx = (dx > 0 ? (world_half - ox) : (-world_half - ox)) / dx;
        if (tx > 0) best = fminf(best, tx);
    }
    if (fabsf(dy) > 1e-6f) {
        float ty = (dy > 0 ? (world_half - oy) : (-world_half - oy)) / dy;
        if (ty > 0) best = fminf(best, ty);
    }
    for (int k = 0; k < n_rects; k++) {
        float d = ray_distance_to_rect(ox, oy, dx, dy, d_rects[k]);
        best = fminf(best, d);
    }
    if (best >= MAX_RANGE) {
        out_hit[rid] = 0;
        out_x[rid] = 0.0f; out_y[rid] = 0.0f;
    } else {
        out_hit[rid] = 1;
        float hx_w = ox + best * dx;
        float hy_w = oy + best * dy;
        float cx = hx_w - ox, cy = hy_w - oy;
        float c = cosf(-yaw), s = sinf(-yaw);
        out_x[rid] = c * cx - s * cy;
        out_y[rid] = s * cx + c * cy;
    }
}

__global__ void add_noise_kernel(int n, unsigned long long seed, float sigma,
                                 float* x, float* y, unsigned char* hit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!hit[i]) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, i, 0, &st);
    x[i] += curand_normal(&st) * sigma;
    y[i] += curand_normal(&st) * sigma;
}

// --- Per-point covariance (k-NN in same scan, regularised to disk) ------
//
// For each hit point i:
//   1. Brute-force scan all other hits, keep the K smallest squared distances.
//   2. Compute 2x2 sample covariance over those K neighbours (+ self).
//   3. Eigendecompose closed-form, normal n = eigenvector of smaller eigval.
//   4. Output C_reg = I - (1 - eps) * n n^T  (eigenvalues become (eps, 1)).
//
// Stored per-point: (c00, c01, c11) for the regularised covariance.
__global__ void compute_cov_kernel(int n,
                                   const float* __restrict__ px,
                                   const float* __restrict__ py,
                                   const unsigned char* __restrict__ hit,
                                   float* __restrict__ c00,
                                   float* __restrict__ c01,
                                   float* __restrict__ c11,
                                   unsigned char* __restrict__ ok) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!hit[i]) { ok[i] = 0; return; }
    float xi = px[i], yi = py[i];
    // Top-K smallest-distance neighbours by insertion into a local buffer.
    float bd[K_NEIGHBORS];
    int   bi[K_NEIGHBORS];
    #pragma unroll
    for (int k = 0; k < K_NEIGHBORS; k++) { bd[k] = 1e30f; bi[k] = -1; }
    float worst = 1e30f;
    int worst_pos = 0;
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        if (!hit[j]) continue;
        float dxj = px[j] - xi;
        float dyj = py[j] - yi;
        float d2 = dxj * dxj + dyj * dyj;
        if (d2 < worst) {
            bd[worst_pos] = d2;
            bi[worst_pos] = j;
            // recompute worst.
            worst = bd[0]; worst_pos = 0;
            #pragma unroll
            for (int k = 1; k < K_NEIGHBORS; k++) {
                if (bd[k] > worst) { worst = bd[k]; worst_pos = k; }
            }
        }
    }
    // Compute mean + 2x2 sample covariance including self.
    float mx = xi, my = yi;
    int cnt = 1;
    #pragma unroll
    for (int k = 0; k < K_NEIGHBORS; k++) {
        if (bi[k] < 0) continue;
        mx += px[bi[k]]; my += py[bi[k]]; cnt++;
    }
    if (cnt < 3) { ok[i] = 0; return; }
    float inv_n = 1.0f / static_cast<float>(cnt);
    mx *= inv_n; my *= inv_n;
    float sxx = (xi - mx) * (xi - mx);
    float syy = (yi - my) * (yi - my);
    float sxy = (xi - mx) * (yi - my);
    #pragma unroll
    for (int k = 0; k < K_NEIGHBORS; k++) {
        if (bi[k] < 0) continue;
        float dx = px[bi[k]] - mx;
        float dy = py[bi[k]] - my;
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    sxx *= inv_n; syy *= inv_n; sxy *= inv_n;
    // Eigendecomposition of 2x2 symmetric.  Smaller eigenvalue's eigenvector
    // is the normal direction.
    float tr  = sxx + syy;
    float det = sxx * syy - sxy * sxy;
    float disc = sqrtf(fmaxf(0.0f, 0.25f * tr * tr - det));
    float l_small = 0.5f * tr - disc;
    float l_large = 0.5f * tr + disc;
    if (l_large < 1e-9f) { ok[i] = 0; return; }
    // Eigenvector for l_small:  (C - l_small I) v = 0
    float nx, ny;
    if (fabsf(sxy) > 1e-9f) {
        nx = l_small - syy;
        ny = sxy;
    } else {
        // Diagonal: normal is along the axis with smaller eigenvalue.
        if (sxx < syy) { nx = 1.0f; ny = 0.0f; }
        else           { nx = 0.0f; ny = 1.0f; }
    }
    float nn = sqrtf(nx * nx + ny * ny);
    if (nn < 1e-9f) { nx = 1.0f; ny = 0.0f; }
    else            { nx /= nn;  ny /= nn; }
    // C_reg = I - (1 - eps) * n n^T  (eigenvalues become (eps, 1))
    float w = 1.0f - GICP_EPS;
    c00[i] = 1.0f - w * nx * nx;
    c01[i] =      - w * nx * ny;
    c11[i] = 1.0f - w * ny * ny;
    ok[i] = 1;
}

// --- Match + Gauss-Newton accumulators ----------------------------------
//
// For each source point i, find the nearest target point j (brute force).
// Reject if distance^2 > MATCH_MAX_D2 in WORLD coords (current iterate
// applied to the source).  Build per-correspondence M = (C_t + R C_s R^T)^{-1}
// and accumulate J^T M r and J^T M J.
__global__ void gicp_accum_kernel(int n,
                                  const float* __restrict__ spx,  // source (live) in sensor frame
                                  const float* __restrict__ spy,
                                  const unsigned char* __restrict__ shit,
                                  const float* __restrict__ sc00,
                                  const float* __restrict__ sc01,
                                  const float* __restrict__ sc11,
                                  const unsigned char* __restrict__ sok,
                                  const float* __restrict__ tpx,  // target (map) in map sensor frame
                                  const float* __restrict__ tpy,
                                  const unsigned char* __restrict__ thit,
                                  const float* __restrict__ tc00,
                                  const float* __restrict__ tc01,
                                  const float* __restrict__ tc11,
                                  const unsigned char* __restrict__ tok,
                                  float tx, float ty, float theta,
                                  float* g, float* H, float* cost_out,
                                  int* match_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!shit[i] || !sok[i]) return;
    float c = cosf(theta), s = sinf(theta);
    float qx = c * spx[i] - s * spy[i] + tx;
    float qy = s * spx[i] + c * spy[i] + ty;
    // Nearest neighbour search in target.
    float best_d2 = MATCH_MAX_D2;
    int best_j = -1;
    for (int j = 0; j < n; j++) {
        if (!thit[j] || !tok[j]) continue;
        float dx = tpx[j] - qx;
        float dy = tpy[j] - qy;
        float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) { best_d2 = d2; best_j = j; }
    }
    if (best_j < 0) return;
    // Source covariance rotated to world: C_s_w = R * C_s * R^T.
    float a = sc00[i], b = sc01[i], d = sc11[i];
    // R C R^T for R = [[c, -s], [s, c]]:
    //   m00 = c^2*a - 2 c s b + s^2 d
    //   m01 = c s (a - d) + (c^2 - s^2) b
    //   m11 = s^2 a + 2 c s b + c^2 d
    float cs = c * s;
    float c2 = c * c, s2 = s * s;
    float sw00 = c2 * a - 2.0f * cs * b + s2 * d;
    float sw01 = cs * (a - d) + (c2 - s2) * b;
    float sw11 = s2 * a + 2.0f * cs * b + c2 * d;
    float ta = tc00[best_j], tb = tc01[best_j], tdd = tc11[best_j];
    // Combined covariance.
    float K00 = ta + sw00;
    float K01 = tb + sw01;
    float K11 = tdd + sw11;
    float Kdet = K00 * K11 - K01 * K01;
    if (Kdet < 1e-9f) return;
    float inv_det = 1.0f / Kdet;
    float M00 =  K11 * inv_det;
    float M01 = -K01 * inv_det;
    float M11 =  K00 * inv_det;
    // Residual r = (qx, qy) - (tpx, tpy).
    float rx = qx - tpx[best_j];
    float ry = qy - tpy[best_j];
    // u = M r
    float ux = M00 * rx + M01 * ry;
    float uy = M01 * rx + M11 * ry;
    float cost = 0.5f * (rx * ux + ry * uy);
    atomicAdd(cost_out, cost);
    atomicAdd(match_count, 1);
    // J = [ I  | R'(theta) p_s ]
    //   R'(theta) = [[-s, -c], [c, -s]]
    //   J col theta = (-s*px - c*py,  c*px - s*py)
    float Jx_th = -s * spx[i] - c * spy[i];
    float Jy_th =  c * spx[i] - s * spy[i];
    // b = J^T M r = J^T u
    //   b[0] = ux
    //   b[1] = uy
    //   b[2] = Jx_th * ux + Jy_th * uy
    atomicAdd(&g[0], ux);
    atomicAdd(&g[1], uy);
    atomicAdd(&g[2], Jx_th * ux + Jy_th * uy);
    // H = J^T M J.  Compute v = M * J[:, 2] = (M00*Jx_th + M01*Jy_th, M01*Jx_th + M11*Jy_th)
    float vx = M00 * Jx_th + M01 * Jy_th;
    float vy = M01 * Jx_th + M11 * Jy_th;
    // H[0,0] = M00,  H[0,1] = M01,  H[0,2] = vx
    // H[1,1] = M11,  H[1,2] = vy
    // H[2,2] = Jx_th * vx + Jy_th * vy
    atomicAdd(&H[0], M00);
    atomicAdd(&H[1], M01);
    atomicAdd(&H[2], vx);
    atomicAdd(&H[3], M11);
    atomicAdd(&H[4], vy);
    atomicAdd(&H[5], Jx_th * vx + Jy_th * vy);
}

// --- Host helpers --------------------------------------------------------
static void invert3x3_sym(const float* H6, float lambda, float* out9) {
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

static cv::Point2i world_to_panel(float x, float y) {
    float scale = PANEL_W / (2.0f * WORLD_HALF);
    return cv::Point2i(static_cast<int>(PANEL_W / 2 + scale * x),
                       static_cast<int>(PANEL_H / 2 - scale * y));
}

}  // namespace cudabot

using namespace cudabot;

int main() {
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
    CUDA_CHECK(cudaMemcpyToSymbol(d_rects, rects.data(), n_rects * sizeof(Rect)));

    float *d_src_x, *d_src_y;     unsigned char *d_src_hit;
    float *d_tgt_x, *d_tgt_y;     unsigned char *d_tgt_hit;
    CUDA_CHECK(cudaMalloc(&d_src_x,  N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_y,  N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_hit, N_RAYS));
    CUDA_CHECK(cudaMalloc(&d_tgt_x,  N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt_y,  N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt_hit, N_RAYS));

    float *d_src_c00, *d_src_c01, *d_src_c11;   unsigned char* d_src_ok;
    float *d_tgt_c00, *d_tgt_c01, *d_tgt_c11;   unsigned char* d_tgt_ok;
    CUDA_CHECK(cudaMalloc(&d_src_c00, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_c01, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_c11, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_ok,  N_RAYS));
    CUDA_CHECK(cudaMalloc(&d_tgt_c00, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt_c01, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt_c11, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt_ok,  N_RAYS));

    float *d_g, *d_H, *d_cost;  int* d_matches;
    CUDA_CHECK(cudaMalloc(&d_g, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cost, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_matches, sizeof(int)));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uni_xy(-1.0f, 1.0f);
    std::uniform_real_distribution<float> uni_th(-0.35f, 0.35f);

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_gicp_2d.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          10, cv::Size(PANEL_W, PANEL_H + 50));

    int blocks_rays = (N_RAYS + 255) / 256;

    double total_ms = 0.0;
    int    count_frames = 0;
    double total_pose_err = 0.0;
    double total_yaw_err  = 0.0;

    std::vector<float> h_src_x(N_RAYS), h_src_y(N_RAYS);
    std::vector<float> h_tgt_x(N_RAYS), h_tgt_y(N_RAYS);
    std::vector<unsigned char> h_src_hit(N_RAYS), h_tgt_hit(N_RAYS);
    std::vector<float> h_tgt_c00(N_RAYS), h_tgt_c01(N_RAYS), h_tgt_c11(N_RAYS);
    std::vector<unsigned char> h_tgt_ok(N_RAYS);

    for (int frame = 0; frame < N_FRAMES; frame++) {
        float ox = 0.0f, oy = 0.0f;
        for (int tries = 0; tries < 64; tries++) {
            float t = static_cast<float>(frame) / N_FRAMES * 2.0f * static_cast<float>(M_PI);
            ox = std::cos(t * 1.7f + tries) * 10.0f;
            oy = std::sin(t * 1.2f + tries * 0.3f) * 10.0f;
            bool inside = false;
            for (const auto& r : rects) {
                if (ox > r.xmin && ox < r.xmax && oy > r.ymin && oy < r.ymax) {
                    inside = true; break;
                }
            }
            if (!inside) break;
        }

        // Target scan at (ox, oy, 0).
        raycast_kernel<<<blocks_rays, 256>>>(ox, oy, 0.0f, n_rects, WORLD_HALF,
                                              d_tgt_x, d_tgt_y, d_tgt_hit);
        add_noise_kernel<<<blocks_rays, 256>>>(N_RAYS, frame * 13ULL + 1,
                                               0.02f, d_tgt_x, d_tgt_y, d_tgt_hit);
        compute_cov_kernel<<<blocks_rays, 256>>>(N_RAYS, d_tgt_x, d_tgt_y, d_tgt_hit,
                                                  d_tgt_c00, d_tgt_c01, d_tgt_c11,
                                                  d_tgt_ok);

        // Source (live) scan at perturbed pose.
        float gt_dx = uni_xy(rng);
        float gt_dy = uni_xy(rng);
        float gt_dth = uni_th(rng);
        float live_ox = ox + gt_dx;
        float live_oy = oy + gt_dy;
        float live_yaw = gt_dth;
        raycast_kernel<<<blocks_rays, 256>>>(live_ox, live_oy, live_yaw,
                                              n_rects, WORLD_HALF,
                                              d_src_x, d_src_y, d_src_hit);
        add_noise_kernel<<<blocks_rays, 256>>>(N_RAYS, frame * 17ULL + 7,
                                               0.02f, d_src_x, d_src_y, d_src_hit);
        compute_cov_kernel<<<blocks_rays, 256>>>(N_RAYS, d_src_x, d_src_y, d_src_hit,
                                                  d_src_c00, d_src_c01, d_src_c11,
                                                  d_src_ok);

        // GICP Gauss-Newton with adaptive LM damping.
        float tx = 0.0f, ty = 0.0f, th = 0.0f;
        float lambda = 1e-3f;
        float last_cost = 1e30f;

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int it = 0; it < GN_ITERS; it++) {
            CUDA_CHECK(cudaMemset(d_g, 0, 3 * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_H, 0, 6 * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_cost, 0, sizeof(float)));
            CUDA_CHECK(cudaMemset(d_matches, 0, sizeof(int)));
            gicp_accum_kernel<<<blocks_rays, 256>>>(N_RAYS,
                                                     d_src_x, d_src_y, d_src_hit,
                                                     d_src_c00, d_src_c01, d_src_c11,
                                                     d_src_ok,
                                                     d_tgt_x, d_tgt_y, d_tgt_hit,
                                                     d_tgt_c00, d_tgt_c01, d_tgt_c11,
                                                     d_tgt_ok,
                                                     tx, ty, th,
                                                     d_g, d_H, d_cost, d_matches);
            float h_g[3], h_H[6], h_cost = 0.0f;
            int   h_match = 0;
            CUDA_CHECK(cudaMemcpy(h_g, d_g, 3 * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_H, d_H, 6 * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_match, d_matches, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_match < 10) break;
            // Adaptive LM: if cost dropped, shrink lambda; else grow and retry.
            if (h_cost < last_cost) lambda = fmaxf(1e-5f, lambda * 0.5f);
            else                    lambda = fminf(1e3f,  lambda * 4.0f);
            last_cost = h_cost;
            float H_inv[9];
            invert3x3_sym(h_H, lambda, H_inv);
            float dtx = -(H_inv[0] * h_g[0] + H_inv[1] * h_g[1] + H_inv[2] * h_g[2]);
            float dty = -(H_inv[3] * h_g[0] + H_inv[4] * h_g[1] + H_inv[5] * h_g[2]);
            float dth = -(H_inv[6] * h_g[0] + H_inv[7] * h_g[1] + H_inv[8] * h_g[2]);
            float step_norm = std::sqrt(dtx * dtx + dty * dty);
            if (step_norm > 0.8f) {
                dtx *= 0.8f / step_norm;
                dty *= 0.8f / step_norm;
            }
            if (std::fabs(dth) > 0.25f) dth = (dth > 0 ? 0.25f : -0.25f);
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
        total_yaw_err  += err_th;
        count_frames++;
        if (frame < 5 || frame % 20 == 0)
            std::printf("frame %3d  gt=(%+.2f,%+.2f,%+.2f) est=(%+.2f,%+.2f,%+.2f) err=(%.3f m, %.3f rad) %.2f ms\n",
                        frame, gt_dx, gt_dy, gt_dth, tx, ty, th, err_t, err_th, ms);

        // --- Visualisation ---
        CUDA_CHECK(cudaMemcpy(h_src_x.data(),  d_src_x,  N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_src_y.data(),  d_src_y,  N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_src_hit.data(),d_src_hit,N_RAYS, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tgt_x.data(),  d_tgt_x,  N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tgt_y.data(),  d_tgt_y,  N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tgt_hit.data(),d_tgt_hit,N_RAYS, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tgt_c00.data(),d_tgt_c00,N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tgt_c01.data(),d_tgt_c01,N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tgt_c11.data(),d_tgt_c11,N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tgt_ok.data(), d_tgt_ok, N_RAYS, cudaMemcpyDeviceToHost));

        cv::Mat img(PANEL_H + 50, PANEL_W, CV_8UC3, cv::Scalar(18, 18, 18));
        for (const auto& r : rects) {
            cv::rectangle(img, world_to_panel(r.xmin, r.ymax),
                          world_to_panel(r.xmax, r.ymin),
                          cv::Scalar(75, 75, 75), -1);
        }
        cv::rectangle(img, world_to_panel(-WORLD_HALF, WORLD_HALF),
                      world_to_panel(WORLD_HALF, -WORLD_HALF),
                      cv::Scalar(70, 70, 70), 1);

        // Surface normals on target points (every 8th).
        for (int k = 0; k < N_RAYS; k += 8) {
            if (!h_tgt_hit[k] || !h_tgt_ok[k]) continue;
            // Normal direction is the smaller-eigenvalue eigenvector of the
            // regularised cov.  Since C_reg = I - w n n^T, we recover n by
            // diagonalising C_reg again.
            float a = h_tgt_c00[k], b = h_tgt_c01[k], c = h_tgt_c11[k];
            float tr = a + c;
            float det = a * c - b * b;
            float disc = std::sqrt(std::max(0.0f, 0.25f * tr * tr - det));
            float l_small = 0.5f * tr - disc;
            float nx, ny;
            if (std::fabs(b) > 1e-6f) { nx = l_small - c; ny = b; }
            else                       { nx = (a < c ? 1.0f : 0.0f);
                                         ny = (a < c ? 0.0f : 1.0f); }
            float nn = std::sqrt(nx * nx + ny * ny);
            if (nn < 1e-6f) continue;
            nx /= nn; ny /= nn;
            float wx = ox + h_tgt_x[k], wy = oy + h_tgt_y[k];
            cv::line(img, world_to_panel(wx, wy),
                     world_to_panel(wx + 0.4f * nx, wy + 0.4f * ny),
                     cv::Scalar(120, 200, 220), 1);
        }

        // Target points
        for (int k = 0; k < N_RAYS; k++) {
            if (!h_tgt_hit[k]) continue;
            cv::circle(img, world_to_panel(ox + h_tgt_x[k], oy + h_tgt_y[k]),
                       1, cv::Scalar(200, 200, 200), -1);
        }
        // Source (initial / unaligned)
        for (int k = 0; k < N_RAYS; k++) {
            if (!h_src_hit[k]) continue;
            cv::circle(img, world_to_panel(ox + h_src_x[k], oy + h_src_y[k]),
                       1, cv::Scalar(50, 50, 220), -1);
        }
        // Source after GICP
        float ce = std::cos(th), se = std::sin(th);
        for (int k = 0; k < N_RAYS; k++) {
            if (!h_src_hit[k]) continue;
            float wx = ce * h_src_x[k] - se * h_src_y[k] + tx;
            float wy = se * h_src_x[k] + ce * h_src_y[k] + ty;
            cv::circle(img, world_to_panel(ox + wx, oy + wy),
                       1, cv::Scalar(60, 220, 80), -1);
        }
        // Sensor markers
        cv::circle(img, world_to_panel(ox, oy), 4, cv::Scalar(255, 255, 255), -1);
        cv::circle(img, world_to_panel(ox + tx, oy + ty), 4, cv::Scalar(60, 220, 80), -1);
        cv::drawMarker(img, world_to_panel(ox + gt_dx, oy + gt_dy),
                       cv::Scalar(255, 200, 80), cv::MARKER_CROSS, 12, 2);

        cv::putText(img, cv::format("GPU GICP 2D scan matching  frame %d", frame),
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
        cv::putText(img, "white=target  red=source(unaligned)  green=source(aligned)  cyan=normals",
                    cv::Point(10, 46),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180, 180, 180), 1);
        video.write(img);
    }
    video.release();

    std::printf("Avg pose err = %.4f m   avg yaw err = %.4f rad   avg %.2f ms/frame\n",
                total_pose_err / count_frames,
                total_yaw_err  / count_frames,
                total_ms       / count_frames);

    cudabot::avi_to_gif("gif/gpu_gicp_2d.avi", "gif/gpu_gicp_2d.gif", 12, 720);
    std::printf("GIF saved to gif/gpu_gicp_2d.gif\n");

    cudaFree(d_src_x); cudaFree(d_src_y); cudaFree(d_src_hit);
    cudaFree(d_tgt_x); cudaFree(d_tgt_y); cudaFree(d_tgt_hit);
    cudaFree(d_src_c00); cudaFree(d_src_c01); cudaFree(d_src_c11); cudaFree(d_src_ok);
    cudaFree(d_tgt_c00); cudaFree(d_tgt_c01); cudaFree(d_tgt_c11); cudaFree(d_tgt_ok);
    cudaFree(d_g); cudaFree(d_H); cudaFree(d_cost); cudaFree(d_matches);
    return 0;
}
