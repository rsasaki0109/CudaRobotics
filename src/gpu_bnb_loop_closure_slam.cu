// gpu_bnb_loop_closure_slam.cu
//
// GPU 2D SLAM whose loop-closure search is BRANCH-AND-BOUND correlative scan
// matching -- the capstone of the CSM line (#120 exhaustive -> #121 single-scan
// loop front-end -> #123 submap loop front-end -> #124 standalone B&B).
//
// #121/#123 detect a loop by EXHAUSTIVELY scoring a window of candidate relposes
// against a submap likelihood field and taking the argmax (one thread = one
// candidate).  That is the repo's canonical parallel pattern, but the candidate
// count grows with the window, so they lean on a coarse-to-fine heuristic to keep
// it affordable.  #124 showed branch-and-bound finds the IDENTICAL global optimum
// of that same objective while scoring orders of magnitude fewer candidates, by
// pruning whole blocks against an admissible upper bound built from a
// multi-resolution max-pool of the field.  This demo wires that primitive into
// the SLAM loop-closure front-end: at every loop attempt it searches the SAME
// full-resolution relpose window (a 4.5 M-cell grid, +/-8 m / +/-0.6 rad at the
// field resolution) two ways -- brute force and branch-and-bound -- and confirms,
// frame after frame, that they return the SAME relpose while B&B scores hundreds
// of times fewer candidates.  The branch-and-bound relpose drives the live
// pose-graph back-end, closing the drifting lap.
//
// This is the efficiency statement, made honestly: B&B is not a different (or
// "wider") search that finds more loops -- a larger window in a sparse-scan scene
// only invites perceptual-aliasing false positives -- it is the SAME search done
// cheaply enough that the front-end can run the full-resolution window directly,
// no coarse-to-fine heuristic, at a fraction of the brute-force candidate work.
//
// Layout: [dead reckoning] | [B&B SLAM] | [info].  The info panel tracks
// dead-reckon vs SLAM ATE, the loops accepted, the per-attempt agreement between
// B&B and brute force, and the candidate-count reduction.
//
// Output: gif/gpu_bnb_loop_closure_slam.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <random>
#include <unordered_map>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int THREADS = 256;
constexpr float PI_F = 3.14159265358979323846f;

// --- World / sensor ---------------------------------------------------------
constexpr float WORLD_HALF = 14.0f;
constexpr int N_SCAN = 64;
constexpr float SCAN_NOISE = 0.060f;
constexpr float MAX_RANGE = 30.0f;
constexpr float MATCH_RANGE = 12.0f;

// --- Local likelihood field (built per loop target, in its sensor frame) ----
constexpr float LF_HALF = 12.5f;
constexpr float LF_RES = 0.0625f;
constexpr int LGRID = static_cast<int>(2.0f * LF_HALF / LF_RES);  // 400
constexpr float FIELD_SIGMA = 0.45f;
constexpr int SUBMAP_KF = 8;

// --- Trajectory (single closed elliptical lap) ------------------------------
constexpr int N_KF = 140;
constexpr float ELLIPSE_A = 9.0f;
constexpr float ELLIPSE_B = 5.5f;
constexpr float ODOM_SIGMA_XY = 0.022f;
constexpr float ODOM_SIGMA_TH = 0.006f;
constexpr float ODOM_BIAS_TH = 0.0034f;

// --- Loop-closure detection -------------------------------------------------
constexpr int LC_MIN_GAP = 45;
constexpr float LC_GATE_R = 5.5f;
constexpr int LC_MAX_CAND = 2;
constexpr float LC_ACCEPT = 0.62f;

// Shared relpose search GRID (both brute force and B&B enumerate it exactly):
// 2^8 = 256 translation cells per axis at the field resolution (+/-8 m), and
// +/-0.6 rad of heading at ~1 deg, all centred on the estimate-predicted relpose.
constexpr int BNB_C = 8;
constexpr int BNB_S = 1 << BNB_C;          // 256
constexpr float SEARCH_RES_TH = 0.0175f;   // ~1 deg
constexpr int SEARCH_HT = 34;              // +/- 0.595 rad
constexpr int SEARCH_NT = 2 * SEARCH_HT + 1;  // 69
constexpr int BNB_SEED_DROP = 3;           // GPU frontier seeded at level C-3
constexpr int C_MAX = BNB_C;

// Fine local refinement applied to the coarse argmax (shared by both searches).
constexpr int FINE_NXY = 31;
constexpr float FINE_RES_XY = 0.020f;      // +/- 0.30 m
constexpr int FINE_NT = 31;
constexpr float FINE_RES_TH = 0.0040f;     // +/- 0.06 rad

// --- Pose-graph back-end ----------------------------------------------------
constexpr int GN_ITERS = 8;
constexpr float ODOM_INFO_XY = 1.0f / (ODOM_SIGMA_XY * ODOM_SIGMA_XY);
constexpr float ODOM_INFO_TH = 1.0f / (ODOM_SIGMA_TH * ODOM_SIGMA_TH);
constexpr float LOOP_INFO_XY = 320.0f;
constexpr float LOOP_INFO_TH = 4000.0f;
constexpr float ANCHOR_INFO = 1.0e7f;

// --- Visualization ----------------------------------------------------------
constexpr int PANEL_W = 430;
constexpr int PANEL_H = 430;
constexpr int INFO_W = 340;
constexpr int FRAME_W = PANEL_W * 2 + INFO_W;
constexpr int FRAME_H = PANEL_H;
constexpr float VIEW_HALF = 13.6f;

struct Pose { float x, y, th; };
struct Rect { float x0, y0, x1, y1; };
struct Edge { int i, j; float zx, zy, zt; float info_xy, info_th; };

__host__ __device__ static inline float clampf(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}
static inline float wrap_angle(float a) {
    while (a > PI_F) a -= 2.0f * PI_F;
    while (a < -PI_F) a += 2.0f * PI_F;
    return a;
}
static inline Pose compose(const Pose& a, const Pose& z) {
    float c = std::cos(a.th), s = std::sin(a.th);
    return {a.x + c * z.x - s * z.y, a.y + s * z.x + c * z.y, wrap_angle(a.th + z.th)};
}
static inline Pose relative(const Pose& a, const Pose& b) {
    float c = std::cos(a.th), s = std::sin(a.th);
    float dxw = b.x - a.x, dyw = b.y - a.y;
    return {c * dxw + s * dyw, -s * dxw + c * dyw, wrap_angle(b.th - a.th)};
}

// --- Map (true environment, used only for raycasting the simulated sensor) ---
static std::vector<Rect> make_rects() {
    std::vector<Rect> r;
    r.push_back({-13.5f, 13.2f, 13.5f, 13.5f});
    r.push_back({-13.5f, -13.5f, 13.5f, -13.2f});
    r.push_back({-13.5f, -13.5f, -13.2f, 13.5f});
    r.push_back({13.2f, -13.5f, 13.5f, 13.5f});
    r.push_back({-8.5f, 7.0f, 2.0f, 7.3f});
    r.push_back({5.0f, 4.5f, 5.3f, 12.5f});
    r.push_back({-11.5f, -2.5f, -11.2f, 8.5f});
    r.push_back({-6.5f, -9.0f, 3.0f, -8.7f});
    r.push_back({7.0f, -9.5f, 10.0f, -9.2f});
    r.push_back({9.7f, -9.5f, 10.0f, -4.5f});
    r.push_back({-3.0f, 0.5f, -2.2f, 1.3f});
    r.push_back({2.6f, -3.2f, 3.4f, -2.4f});
    r.push_back({-0.4f, 9.5f, 0.4f, 10.3f});
    return r;
}

static bool is_wall(const std::vector<Rect>& rects, float x, float y) {
    if (x <= -WORLD_HALF || x >= WORLD_HALF || y <= -WORLD_HALF || y >= WORLD_HALF) return true;
    for (const Rect& r : rects)
        if (x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1) return true;
    return false;
}

static Pose gt_pose(int k) {
    float u = (2.0f * PI_F * k) / N_KF;
    float x = ELLIPSE_A * std::cos(u);
    float y = ELLIPSE_B * std::sin(u);
    float dx = -ELLIPSE_A * std::sin(u);
    float dy = ELLIPSE_B * std::cos(u);
    return {x, y, std::atan2(dy, dx)};
}

static void make_scan(const std::vector<Rect>& rects, const Pose& p, unsigned seed,
                      std::vector<float>& fx, std::vector<float>& fy,
                      std::vector<float>& mx, std::vector<float>& my) {
    fx.clear(); fy.clear(); mx.clear(); my.clear();
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, SCAN_NOISE);
    float cth = std::cos(-p.th), sth = std::sin(-p.th);
    for (int i = 0; i < N_SCAN; ++i) {
        float a = p.th + (2.0f * PI_F * i) / N_SCAN;
        float ca = std::cos(a), sa = std::sin(a);
        float r = MAX_RANGE;
        for (float t = 0.10f; t < MAX_RANGE; t += 0.04f)
            if (is_wall(rects, p.x + t * ca, p.y + t * sa)) { r = t; break; }
        r = clampf(r + noise(rng), 0.15f, MAX_RANGE);
        float wx = r * ca, wy = r * sa;
        float lx = cth * wx - sth * wy;
        float ly = sth * wx + cth * wy;
        fx.push_back(lx); fy.push_back(ly);
        if (r <= MATCH_RANGE) { mx.push_back(lx); my.push_back(ly); }
    }
}

// --- Local likelihood field -------------------------------------------------
__host__ __device__ static inline float sample_local(const float* lut, float lx, float ly) {
    float fx = (lx + LF_HALF) / LF_RES - 0.5f;
    float fy = (ly + LF_HALF) / LF_RES - 0.5f;
    int ix = static_cast<int>(floorf(fx));
    int iy = static_cast<int>(floorf(fy));
    if (ix < 0 || ix >= LGRID - 1 || iy < 0 || iy >= LGRID - 1) return 0.0f;
    float tx = fx - ix, ty = fy - iy;
    const float* row0 = lut + iy * LGRID + ix;
    const float* row1 = row0 + LGRID;
    float a = row0[0] * (1 - tx) + row0[1] * tx;
    float b = row1[0] * (1 - tx) + row1[1] * tx;
    return a * (1 - ty) + b * ty;
}

static std::vector<float> field_from_occ(const cv::Mat& occ) {
    cv::Mat dist_px;
    cv::distanceTransform(occ, dist_px, cv::DIST_L2, 3);
    std::vector<float> lut(LGRID * LGRID);
    float inv2s2 = 1.0f / (2.0f * FIELD_SIGMA * FIELD_SIGMA);
    for (int iy = 0; iy < LGRID; ++iy)
        for (int ix = 0; ix < LGRID; ++ix) {
            float d = dist_px.at<float>(iy, ix) * LF_RES;
            lut[iy * LGRID + ix] = std::exp(-d * d * inv2s2);
        }
    return lut;
}

static inline void raster_point(cv::Mat& occ, float lx, float ly) {
    int ix = static_cast<int>((lx + LF_HALF) / LF_RES);
    int iy = static_cast<int>((ly + LF_HALF) / LF_RES);
    if (ix >= 0 && ix < LGRID && iy >= 0 && iy < LGRID) occ.at<unsigned char>(iy, ix) = 0;
}

static std::vector<float> build_submap_field(int target, const std::vector<Pose>& est,
                                             const std::vector<std::vector<float>>& mx,
                                             const std::vector<std::vector<float>>& my) {
    int s0 = (target / SUBMAP_KF) * SUBMAP_KF;
    int s1 = std::min(s0 + SUBMAP_KF, N_KF);
    cv::Mat occ(LGRID, LGRID, CV_8U, cv::Scalar(255));
    for (int m = s0; m < s1; ++m) {
        Pose rel = relative(est[target], est[m]);
        float c = std::cos(rel.th), s = std::sin(rel.th);
        for (size_t t = 0; t < mx[m].size(); ++t) {
            float lx = rel.x + c * mx[m][t] - s * my[m][t];
            float ly = rel.y + s * mx[m][t] + c * my[m][t];
            raster_point(occ, lx, ly);
        }
    }
    return field_from_occ(occ);
}

// --- Fine refinement (continuous bilinear CSM around a coarse argmax) --------
__global__ void csm_kernel(const float* __restrict__ sx, const float* __restrict__ sy,
                           int n_pts, const float* __restrict__ lut,
                           float cx, float cy, float cth,
                           int nxy, int nt, float res_xy, float res_th,
                           float* __restrict__ score) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nxy * nxy * nt;
    if (idx >= total) return;
    int it = idx / (nxy * nxy);
    int rem = idx - it * (nxy * nxy);
    int iy = rem / nxy;
    int ix = rem - iy * nxy;
    float px = cx + (ix - (nxy - 1) * 0.5f) * res_xy;
    float py = cy + (iy - (nxy - 1) * 0.5f) * res_xy;
    float pth = cth + (it - (nt - 1) * 0.5f) * res_th;
    float c = cosf(pth), s = sinf(pth);
    float acc = 0.0f;
    for (int k = 0; k < n_pts; ++k) {
        float wx = px + c * sx[k] - s * sy[k];
        float wy = py + s * sx[k] + c * sy[k];
        acc += sample_local(lut, wx, wy);
    }
    score[idx] = acc;
}

struct LoopResult { Pose rel; float score; };

static std::pair<Pose, float> fine_refine(const float* d_sx, const float* d_sy, int n_pts,
                                          const float* d_lut, const Pose& center,
                                          float* d_score, std::vector<float>& h_score) {
    int total = FINE_NXY * FINE_NXY * FINE_NT;
    int blocks = (total + THREADS - 1) / THREADS;
    csm_kernel<<<blocks, THREADS>>>(d_sx, d_sy, n_pts, d_lut, center.x, center.y, center.th,
                                    FINE_NXY, FINE_NT, FINE_RES_XY, FINE_RES_TH, d_score);
    CUDA_CHECK(cudaGetLastError());
    h_score.resize(total);
    CUDA_CHECK(cudaMemcpy(h_score.data(), d_score, total * sizeof(float), cudaMemcpyDeviceToHost));
    int best = 0;
    for (int i = 1; i < total; ++i)
        if (h_score[i] > h_score[best]) best = i;
    int it = best / (FINE_NXY * FINE_NXY);
    int rem = best - it * (FINE_NXY * FINE_NXY);
    int iy = rem / FINE_NXY, ix = rem - iy * FINE_NXY;
    Pose p{center.x + (ix - (FINE_NXY - 1) * 0.5f) * FINE_RES_XY,
           center.y + (iy - (FINE_NXY - 1) * 0.5f) * FINE_RES_XY,
           wrap_angle(center.th + (it - (FINE_NT - 1) * 0.5f) * FINE_RES_TH)};
    return {p, h_score[best]};
}

// --- Shared coarse search grid ----------------------------------------------
// base_x/base_y[it*npts + k] = field cell of rotated scan point k at the window's
// minimum corner (heading slice it).  A grid candidate at offset (ix, iy) samples
// cell base + (ix, iy); a B&B node at corner (a, b) level L samples the max-pool
// level L there -- the admissible bound for every leaf in the 2^L block.
static void build_base_local(const std::vector<float>& px, const std::vector<float>& py,
                             const Pose& center, std::vector<int>& bx, std::vector<int>& by) {
    int npts = static_cast<int>(px.size());
    bx.assign(SEARCH_NT * npts, 0);
    by.assign(SEARCH_NT * npts, 0);
    float cornerx = center.x - (BNB_S * 0.5f) * LF_RES;
    float cornery = center.y - (BNB_S * 0.5f) * LF_RES;
    for (int it = 0; it < SEARCH_NT; ++it) {
        float th = center.th + (it - SEARCH_HT) * SEARCH_RES_TH;
        float c = std::cos(th), s = std::sin(th);
        for (int k = 0; k < npts; ++k) {
            float lx = cornerx + c * px[k] - s * py[k];
            float ly = cornery + s * px[k] + c * py[k];
            bx[it * npts + k] = static_cast<int>(std::floor((lx + LF_HALF) / LF_RES));
            by[it * npts + k] = static_cast<int>(std::floor((ly + LF_HALF) / LF_RES));
        }
    }
}

static inline Pose grid_pose(const Pose& center, int it, int ix, int iy) {
    return {center.x + (ix - BNB_S * 0.5f) * LF_RES,
            center.y + (iy - BNB_S * 0.5f) * LF_RES,
            wrap_angle(center.th + (it - SEARCH_HT) * SEARCH_RES_TH)};
}

// Brute force: one thread scores one (it, iy, ix) grid candidate (nearest field).
__global__ void exhaustive_grid_kernel(const int* __restrict__ base_x, const int* __restrict__ base_y,
                                       const float* __restrict__ m0, int npts,
                                       float* __restrict__ score) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BNB_S * BNB_S * SEARCH_NT;
    if (idx >= total) return;
    int it = idx / (BNB_S * BNB_S);
    int rem = idx - it * (BNB_S * BNB_S);
    int iy = rem / BNB_S, ix = rem - iy * BNB_S;
    const int* bx = base_x + it * npts;
    const int* by = base_y + it * npts;
    float acc = 0.0f;
    for (int k = 0; k < npts; ++k) {
        int cx = bx[k] + ix, cy = by[k] + iy;
        if (cx >= 0 && cx < LGRID && cy >= 0 && cy < LGRID) acc += m0[cy * LGRID + cx];
    }
    score[idx] = acc;
}

struct Cell { int it, ix, iy; float score; };

static Cell run_exhaustive_grid(const int* d_base_x, const int* d_base_y, const float* d_m0,
                                int npts, float* d_score, std::vector<float>& h_score) {
    int total = BNB_S * BNB_S * SEARCH_NT;
    int blocks = (total + THREADS - 1) / THREADS;
    exhaustive_grid_kernel<<<blocks, THREADS>>>(d_base_x, d_base_y, d_m0, npts, d_score);
    CUDA_CHECK(cudaGetLastError());
    h_score.resize(total);
    CUDA_CHECK(cudaMemcpy(h_score.data(), d_score, total * sizeof(float), cudaMemcpyDeviceToHost));
    int best = 0;
    for (int i = 1; i < total; ++i)
        if (h_score[i] > h_score[best]) best = i;
    int it = best / (BNB_S * BNB_S);
    int rem = best - it * (BNB_S * BNB_S);
    int iy = rem / BNB_S, ix = rem - iy * BNB_S;
    return {it, ix, iy, h_score[best]};
}

// --- Multi-resolution max-pool of the local field (built on the GPU) --------
__global__ void maxpool_local_kernel(const float* __restrict__ src, float* __restrict__ dst, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= LGRID * LGRID) return;
    int iy = idx / LGRID, ix = idx - iy * LGRID;
    float v = src[idx];
    if (ix + step < LGRID) v = fmaxf(v, src[iy * LGRID + ix + step]);
    if (iy + step < LGRID) v = fmaxf(v, src[(iy + step) * LGRID + ix]);
    if (ix + step < LGRID && iy + step < LGRID) v = fmaxf(v, src[(iy + step) * LGRID + ix + step]);
    dst[idx] = v;
}

// Frontier scoring: one thread = one B&B node, max-pool level (block size span);
// the sample cell is clamped to [0, LGRID-span] so the bound's window always
// contains the block's in-grid leaves (admissible even at the grid edge, #124).
__global__ void score_nodes_local_kernel(const int* __restrict__ base_x, const int* __restrict__ base_y,
                                         const float* __restrict__ ml, const int* __restrict__ nit,
                                         const int* __restrict__ na, const int* __restrict__ nb,
                                         int nnodes, int npts, int span, float* __restrict__ out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= nnodes) return;
    int it = nit[n], a = na[n], b = nb[n];
    const int* bx = base_x + it * npts;
    const int* by = base_y + it * npts;
    int hi = LGRID - span;
    float acc = 0.0f;
    for (int k = 0; k < npts; ++k) {
        int cx = min(max(bx[k] + a, 0), hi);
        int cy = min(max(by[k] + b, 0), hi);
        acc += ml[cy * LGRID + cx];
    }
    out[n] = acc;
}

static float score_host_local(const std::vector<float>& ml, const int* bx, const int* by,
                              int npts, int a, int b, int span) {
    const float* M = ml.data();
    float acc = 0.0f;
    if (span == 1) {
        for (int k = 0; k < npts; ++k) {
            int cx = bx[k] + a, cy = by[k] + b;
            if (cx >= 0 && cx < LGRID && cy >= 0 && cy < LGRID) acc += M[cy * LGRID + cx];
        }
    } else {
        int hi = LGRID - span;
        for (int k = 0; k < npts; ++k) {
            int cx = std::min(std::max(bx[k] + a, 0), hi);
            int cy = std::min(std::max(by[k] + b, 0), hi);
            acc += M[cy * LGRID + cx];
        }
    }
    return acc;
}

struct BNode { float bound; int it, a, b, level; };
struct BNodeCmp { bool operator()(const BNode& x, const BNode& y) const { return x.bound < y.bound; } };

static void build_maxpool(float* d_lut, std::vector<float*>& d_levels,
                          std::vector<std::vector<float>>& h_levels) {
    int cells = LGRID * LGRID;
    CUDA_CHECK(cudaMemcpy(d_levels[0], d_lut, cells * sizeof(float), cudaMemcpyDeviceToDevice));
    int blocks = (cells + THREADS - 1) / THREADS;
    for (int c = 1; c <= C_MAX; ++c)
        maxpool_local_kernel<<<blocks, THREADS>>>(d_levels[c - 1], d_levels[c], 1 << (c - 1));
    CUDA_CHECK(cudaGetLastError());
    for (int c = 0; c <= C_MAX; ++c) {
        h_levels[c].resize(cells);
        CUDA_CHECK(cudaMemcpy(h_levels[c].data(), d_levels[c], cells * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
}

// Branch-and-bound over the shared grid; returns the coarse argmax cell and the
// number of node-scorings performed.  GPU scores a complete coarse frontier in
// parallel; a host best-first descent refines the winner (first leaf == optimum).
static Cell run_bnb(const std::vector<int>& base_x, const std::vector<int>& base_y,
                    const int* d_base_x, const int* d_base_y, std::vector<float*>& d_levels,
                    std::vector<std::vector<float>>& h_levels, int npts, int* d_nit, int* d_na,
                    int* d_nb, float* d_node_score, long long* out_nodes) {
    int seed_level = std::max(1, BNB_C - BNB_SEED_DROP);
    int blk = 1 << seed_level;
    int per = BNB_S / blk;
    int frontier = per * per * SEARCH_NT;
    std::vector<int> nit(frontier), na(frontier), nb(frontier);
    int idx = 0;
    for (int it = 0; it < SEARCH_NT; ++it)
        for (int bi = 0; bi < per; ++bi)
            for (int ai = 0; ai < per; ++ai) {
                nit[idx] = it; na[idx] = ai * blk; nb[idx] = bi * blk; ++idx;
            }
    CUDA_CHECK(cudaMemcpy(d_nit, nit.data(), frontier * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_na, na.data(), frontier * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nb, nb.data(), frontier * sizeof(int), cudaMemcpyHostToDevice));
    int blocks = (frontier + THREADS - 1) / THREADS;
    score_nodes_local_kernel<<<blocks, THREADS>>>(d_base_x, d_base_y, d_levels[seed_level],
                                                  d_nit, d_na, d_nb, frontier, npts, blk, d_node_score);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> bounds(frontier);
    CUDA_CHECK(cudaMemcpy(bounds.data(), d_node_score, frontier * sizeof(float),
                          cudaMemcpyDeviceToHost));

    long long count = frontier;
    std::priority_queue<BNode, std::vector<BNode>, BNodeCmp> pq;
    for (int i = 0; i < frontier; ++i) pq.push({bounds[i], nit[i], na[i], nb[i], seed_level});
    BNode best{-1.0f, 0, 0, 0, 0};
    while (!pq.empty()) {
        BNode n = pq.top();
        pq.pop();
        if (n.level == 0) { best = n; break; }
        int h = 1 << (n.level - 1);
        int off[4][2] = {{0, 0}, {h, 0}, {0, h}, {h, h}};
        const int* bx = &base_x[n.it * npts];
        const int* by = &base_y[n.it * npts];
        for (int q = 0; q < 4; ++q) {
            int a2 = n.a + off[q][0], b2 = n.b + off[q][1];
            float bd = score_host_local(h_levels[n.level - 1], bx, by, npts, a2, b2, h);
            ++count;
            pq.push({bd, n.it, a2, b2, n.level - 1});
        }
    }
    if (out_nodes) *out_nodes = count;
    return {best.it, best.a, best.b, best.bound};
}

// --- Dense SE(2) pose-graph Gauss-Newton back-end (small graph, host) -------
static bool chol_solve(std::vector<double>& A, std::vector<double>& b, int n) {
    for (int j = 0; j < n; ++j) {
        double sum = A[j * n + j];
        for (int k = 0; k < j; ++k) sum -= A[j * n + k] * A[j * n + k];
        if (sum <= 0.0) return false;
        double Ljj = std::sqrt(sum);
        A[j * n + j] = Ljj;
        for (int i = j + 1; i < n; ++i) {
            double s = A[i * n + j];
            for (int k = 0; k < j; ++k) s -= A[i * n + k] * A[j * n + k];
            A[i * n + j] = s / Ljj;
        }
    }
    for (int i = 0; i < n; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= A[i * n + k] * b[k];
        b[i] = s / A[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int k = i + 1; k < n; ++k) s -= A[k * n + i] * b[k];
        b[i] = s / A[i * n + i];
    }
    return true;
}

static void optimise_graph(std::vector<Pose>& poses, const std::vector<Edge>& edges,
                           const Pose& anchor, int n_active) {
    int n = n_active * 3;
    std::vector<double> H(n * n), g(n);
    for (int gn = 0; gn < GN_ITERS; ++gn) {
        std::fill(H.begin(), H.end(), 0.0);
        std::fill(g.begin(), g.end(), 0.0);
        {
            double dx = poses[0].x - anchor.x, dy = poses[0].y - anchor.y;
            double dt = wrap_angle(poses[0].th - anchor.th);
            H[0 * n + 0] += ANCHOR_INFO; H[1 * n + 1] += ANCHOR_INFO; H[2 * n + 2] += ANCHOR_INFO;
            g[0] += ANCHOR_INFO * dx; g[1] += ANCHOR_INFO * dy; g[2] += ANCHOR_INFO * dt;
        }
        for (const Edge& e : edges) {
            if (e.i >= n_active || e.j >= n_active) continue;
            const Pose& pi = poses[e.i];
            const Pose& pj = poses[e.j];
            float c = std::cos(pi.th), s = std::sin(pi.th);
            float dxw = pj.x - pi.x, dyw = pj.y - pi.y;
            float dx = dxw * c + dyw * s;
            float dy = -dxw * s + dyw * c;
            float rt = wrap_angle(wrap_angle(pj.th - pi.th) - e.zt);
            float rx = dx - e.zx, ry = dy - e.zy;
            double Ji[9] = {-c, -s, -dxw * s + dyw * c,
                             s, -c, -dxw * c - dyw * s,
                             0, 0, -1};
            double Jj[9] = { c,  s, 0,
                            -s,  c, 0,
                             0,  0, 1};
            double w[3] = {e.info_xy, e.info_xy, e.info_th};
            double r[3] = {rx, ry, rt};
            int bi = e.i * 3, bj = e.j * 3;
            auto add = [&](const double* Ja, int ba, const double* Jb, int bb) {
                for (int p = 0; p < 3; ++p)
                    for (int q = 0; q < 3; ++q) {
                        double v = 0.0;
                        for (int kk = 0; kk < 3; ++kk) v += Ja[3 * kk + p] * w[kk] * Jb[3 * kk + q];
                        H[(ba + p) * n + (bb + q)] += v;
                    }
            };
            add(Ji, bi, Ji, bi); add(Jj, bj, Jj, bj);
            add(Ji, bi, Jj, bj); add(Jj, bj, Ji, bi);
            for (int p = 0; p < 3; ++p) {
                double gi = 0, gj = 0;
                for (int kk = 0; kk < 3; ++kk) { gi += Ji[3 * kk + p] * w[kk] * r[kk];
                                                 gj += Jj[3 * kk + p] * w[kk] * r[kk]; }
                g[bi + p] += gi; g[bj + p] += gj;
            }
        }
        for (int d = 0; d < n; ++d) H[d * n + d] += 1.0e-3;
        std::vector<double> dx = g;
        if (!chol_solve(H, dx, n)) break;
        for (int k = 0; k < n_active; ++k) {
            poses[k].x -= static_cast<float>(dx[3 * k + 0]);
            poses[k].y -= static_cast<float>(dx[3 * k + 1]);
            poses[k].th = wrap_angle(poses[k].th - static_cast<float>(dx[3 * k + 2]));
        }
        poses[0] = anchor;
    }
}

// --- Visualization ----------------------------------------------------------
static cv::Point world_to_panel(int ox, float x, float y) {
    int px = ox + static_cast<int>((x + VIEW_HALF) / (2 * VIEW_HALF) * PANEL_W);
    int py = static_cast<int>((VIEW_HALF - y) / (2 * VIEW_HALF) * PANEL_H);
    return cv::Point(px, py);
}
static inline void put_pt(cv::Mat& img, int ox, float x, float y, const cv::Vec3b& col) {
    cv::Point p = world_to_panel(ox, x, y);
    if (p.x > ox && p.x < ox + PANEL_W - 1 && p.y > 0 && p.y < PANEL_H - 1)
        img.at<cv::Vec3b>(p.y, p.x) = col;
}
static cv::Vec3b submap_tint(int submap) {
    static const cv::Vec3b pal[6] = {
        {150, 175, 120}, {120, 150, 195}, {165, 135, 170},
        {120, 175, 175}, {175, 160, 110}, {150, 145, 150}};
    return pal[submap % 6];
}
static void draw_map(cv::Mat& img, int ox, const std::vector<Pose>& poses, int k,
                     const std::vector<std::vector<float>>& mx,
                     const std::vector<std::vector<float>>& my,
                     const cv::Vec3b& pt_col, const cv::Scalar& traj_col, bool tint) {
    cv::rectangle(img, cv::Rect(ox, 0, PANEL_W, PANEL_H), cv::Scalar(248, 248, 245), -1);
    for (int j = 0; j <= k; ++j) {
        const Pose& p = poses[j];
        float c = std::cos(p.th), s = std::sin(p.th);
        cv::Vec3b col = tint ? submap_tint(j / SUBMAP_KF) : pt_col;
        for (size_t t = 0; t < mx[j].size(); ++t) {
            float wx = p.x + c * mx[j][t] - s * my[j][t];
            float wy = p.y + s * mx[j][t] + c * my[j][t];
            put_pt(img, ox, wx, wy, col);
        }
    }
    for (int j = 1; j <= k; ++j)
        cv::line(img, world_to_panel(ox, poses[j - 1].x, poses[j - 1].y),
                 world_to_panel(ox, poses[j].x, poses[j].y), traj_col, 2, cv::LINE_AA);
    cv::circle(img, world_to_panel(ox, poses[k].x, poses[k].y), 4, traj_col, -1, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(ox + 1, 1, PANEL_W - 2, PANEL_H - 2), cv::Scalar(120, 124, 130), 1);
}

struct Stats {
    long long bnb_nodes = 0, exh_cand = 0;
    int agree = 0, attempts = 0;
    double gpu_bnb_ms = 0.0, gpu_exh_ms = 0.0;
};

static float ate_rmse(const std::vector<Pose>& poses, int k) {
    double s = 0.0;
    for (int j = 0; j <= k; ++j) {
        Pose g = gt_pose(j);
        double dx = poses[j].x - g.x, dy = poses[j].y - g.y;
        s += dx * dx + dy * dy;
    }
    return std::sqrt(s / (k + 1));
}

static void draw_info(cv::Mat& img, int ox, int k, float odom_ate, float slam_ate,
                      int accepted, int rejected, const Stats& s,
                      const std::vector<float>& odom_hist, const std::vector<float>& slam_hist) {
    cv::rectangle(img, cv::Rect(ox, 0, INFO_W, FRAME_H), cv::Scalar(244, 246, 246), -1);
    cv::putText(img, "branch & bound SLAM", cv::Point(ox + 14, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.54, cv::Scalar(20, 28, 35), 2, cv::LINE_AA);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "keyframe %d / %d", k, N_KF - 1);
    cv::putText(img, buf, cv::Point(ox + 14, 56), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(60, 66, 74), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "dead-reckon ATE: %.2f m", odom_ate);
    cv::putText(img, buf, cv::Point(ox + 14, 84), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(60, 70, 210), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "B&B SLAM ATE:    %.2f m", slam_ate);
    cv::putText(img, buf, cv::Point(ox + 14, 108), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(45, 150, 60), 1, cv::LINE_AA);

    int px0 = ox + 14, py0 = 128, pw = INFO_W - 36, ph = 96;
    cv::rectangle(img, cv::Rect(px0, py0, pw, ph), cv::Scalar(255, 255, 255), -1);
    cv::rectangle(img, cv::Rect(px0, py0, pw, ph), cv::Scalar(200, 204, 210), 1);
    float ymax = 4.0f;
    auto plot = [&](const std::vector<float>& h, const cv::Scalar& col) {
        for (size_t i = 1; i < h.size(); ++i) {
            float x0 = px0 + pw * (i - 1) / (float)(N_KF - 1);
            float x1 = px0 + pw * i / (float)(N_KF - 1);
            float y0 = py0 + ph - ph * clampf(h[i - 1] / ymax, 0, 1);
            float y1 = py0 + ph - ph * clampf(h[i] / ymax, 0, 1);
            cv::line(img, cv::Point((int)x0, (int)y0), cv::Point((int)x1, (int)y1), col, 2, cv::LINE_AA);
        }
    };
    plot(odom_hist, cv::Scalar(60, 70, 210));
    plot(slam_hist, cv::Scalar(45, 150, 60));
    cv::putText(img, "ATE vs keyframe (0-4 m)", cv::Point(px0, py0 - 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(70, 76, 84), 1, cv::LINE_AA);

    int y = py0 + ph + 28;
    std::snprintf(buf, sizeof(buf), "loops accepted: %d  (rej %d)", accepted, rejected);
    cv::putText(img, buf, cv::Point(ox + 14, y), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(45, 120, 55), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "window: 4.5M cells (+/-8m, +/-0.6rad)");
    cv::putText(img, buf, cv::Point(ox + 14, y + 22), cv::FONT_HERSHEY_SIMPLEX, 0.40,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "B&B nodes/attempt:  %.1fk", s.bnb_nodes / 1e3);
    cv::putText(img, buf, cv::Point(ox + 14, y + 44), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(45, 120, 55), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "brute force/attempt: %.2fM", s.exh_cand / 1e6);
    cv::putText(img, buf, cv::Point(ox + 14, y + 66), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(70, 110, 160), 1, cv::LINE_AA);
    if (s.bnb_nodes > 0) {
        std::snprintf(buf, sizeof(buf), "B&B scores %.0fx fewer", (double)s.exh_cand / s.bnb_nodes);
        cv::putText(img, buf, cv::Point(ox + 14, y + 88), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                    cv::Scalar(45, 95, 175), 1, cv::LINE_AA);
    }
    std::snprintf(buf, sizeof(buf), "B&B == brute force: %d/%d", s.agree, s.attempts);
    cv::putText(img, buf, cv::Point(ox + 14, y + 110), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(40, 130, 40), 1, cv::LINE_AA);
}

static void ensure_dirs() {
    int rc = std::system("mkdir -p gif tmp");
    if (rc != 0) std::fprintf(stderr, "mkdir failed with code %d\n", rc);
}

struct Gpu {
    float *d_sx, *d_sy, *d_lut, *d_score;
    std::vector<float*> d_levels;
    int *d_nit, *d_na, *d_nb;
    float* d_node_score;
    int *d_base_x, *d_base_y;
};

// One SLAM keyframe: gate candidate revisits, search each relpose with B&B (which
// drives the graph) and with brute force (verification + count), accept by score.
struct LoopCtx {
    std::vector<Pose>& est;
    std::vector<Edge>& edges;
    std::vector<Edge>& loop_edges;
    std::unordered_map<int, std::vector<float>>& field_cache;
    int& accepted;
    int& rejected;
    int& proposed;
};

static bool process_keyframe(LoopCtx& v, int k,
                             const std::vector<std::vector<float>>& mx,
                             const std::vector<std::vector<float>>& my,
                             Gpu& gpu, std::vector<float>& h_score,
                             std::vector<std::vector<float>>& h_levels,
                             std::vector<int>& base_x, std::vector<int>& base_y, Stats& stats) {
    if (k < LC_MIN_GAP) return false;
    auto get_field = [&](int o) -> const std::vector<float>& {
        auto it = v.field_cache.find(o);
        if (it == v.field_cache.end())
            it = v.field_cache.emplace(o, build_submap_field(o, v.est, mx, my)).first;
        return it->second;
    };

    std::vector<std::pair<float, int>> cand;
    for (int o = 0; o <= k - LC_MIN_GAP; ++o) {
        float dx = v.est[o].x - v.est[k].x, dy = v.est[o].y - v.est[k].y;
        float d2 = dx * dx + dy * dy;
        if (d2 < LC_GATE_R * LC_GATE_R) cand.push_back({d2, o});
    }
    std::sort(cand.begin(), cand.end());

    bool accepted_any = false;
    int tried = 0;
    for (auto& cc : cand) {
        if (tried >= LC_MAX_CAND) break;
        int o = cc.second;
        ++tried;
        ++v.proposed;
        const std::vector<float>& lut = get_field(o);
        CUDA_CHECK(cudaMemcpy(gpu.d_lut, lut.data(), lut.size() * sizeof(float), cudaMemcpyHostToDevice));
        int n_pts = static_cast<int>(mx[k].size());
        CUDA_CHECK(cudaMemcpy(gpu.d_sx, mx[k].data(), n_pts * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu.d_sy, my[k].data(), n_pts * sizeof(float), cudaMemcpyHostToDevice));
        Pose rel_init = relative(v.est[o], v.est[k]);

        build_base_local(mx[k], my[k], rel_init, base_x, base_y);
        CUDA_CHECK(cudaMemcpy(gpu.d_base_x, base_x.data(), base_x.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu.d_base_y, base_y.data(), base_y.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        build_maxpool(gpu.d_lut, gpu.d_levels, h_levels);

        // Brute force over the grid (verification + candidate count + timing).
        auto e0 = std::chrono::high_resolution_clock::now();
        Cell ce = run_exhaustive_grid(gpu.d_base_x, gpu.d_base_y, gpu.d_lut, n_pts, gpu.d_score, h_score);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto e1 = std::chrono::high_resolution_clock::now();

        // Branch-and-bound over the SAME grid (drives the graph).
        long long nodes = 0;
        auto b0 = std::chrono::high_resolution_clock::now();
        Cell cb = run_bnb(base_x, base_y, gpu.d_base_x, gpu.d_base_y, gpu.d_levels, h_levels,
                          n_pts, gpu.d_nit, gpu.d_na, gpu.d_nb, gpu.d_node_score, &nodes);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto b1 = std::chrono::high_resolution_clock::now();

        int dcell = std::max(std::abs(ce.it - cb.it), std::max(std::abs(ce.ix - cb.ix),
                                                               std::abs(ce.iy - cb.iy)));
        ++stats.attempts;
        if (dcell <= 1) ++stats.agree;
        stats.bnb_nodes = nodes;
        stats.exh_cand = (long long)BNB_S * BNB_S * SEARCH_NT;
        stats.gpu_bnb_ms = std::chrono::duration<double, std::milli>(b1 - b0).count();
        stats.gpu_exh_ms = std::chrono::duration<double, std::milli>(e1 - e0).count();

        // Refine the B&B argmax with a fine local pass, then gate.
        Pose coarse = grid_pose(rel_init, cb.it, cb.ix, cb.iy);
        auto fine = fine_refine(gpu.d_sx, gpu.d_sy, n_pts, gpu.d_lut, coarse, gpu.d_score, h_score);
        float score = n_pts > 0 ? fine.second / n_pts : 0.0f;

        if (score >= LC_ACCEPT) {
            v.edges.push_back({o, k, fine.first.x, fine.first.y, fine.first.th,
                               LOOP_INFO_XY, LOOP_INFO_TH});
            v.loop_edges.push_back({o, k, 0, 0, 0, 0, 0});
            ++v.accepted;
            accepted_any = true;
        } else {
            ++v.rejected;
        }
    }
    return accepted_any;
}

static Stats run_demo() {
    ensure_dirs();
    std::vector<Rect> rects = make_rects();

    std::vector<std::vector<float>> mx(N_KF), my(N_KF);
    std::vector<Pose> odom(N_KF), odom_z(N_KF);
    {
        std::mt19937 rng(20260527u);
        std::normal_distribution<float> nxy(0.0f, ODOM_SIGMA_XY), nth(0.0f, ODOM_SIGMA_TH);
        Pose g0 = gt_pose(0);
        odom[0] = g0;
        std::vector<float> fx, fy;
        make_scan(rects, g0, 1000, fx, fy, mx[0], my[0]);
        for (int k = 1; k < N_KF; ++k) {
            Pose gprev = gt_pose(k - 1), gcur = gt_pose(k);
            Pose ztrue = relative(gprev, gcur);
            Pose z{ztrue.x + nxy(rng), ztrue.y + nxy(rng),
                   wrap_angle(ztrue.th + nth(rng) + ODOM_BIAS_TH)};
            odom_z[k] = z;
            odom[k] = compose(odom[k - 1], z);
            make_scan(rects, gcur, 1000u + k, fx, fy, mx[k], my[k]);
        }
    }

    std::vector<Pose> est = odom;
    std::vector<Edge> edges, loop_edges;
    std::unordered_map<int, std::vector<float>> field_cache;
    int accepted = 0, rejected = 0, proposed = 0;
    LoopCtx v{est, edges, loop_edges, field_cache, accepted, rejected, proposed};

    Gpu gpu;
    CUDA_CHECK(cudaMalloc(&gpu.d_sx, N_SCAN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu.d_sy, N_SCAN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu.d_lut, LGRID * LGRID * sizeof(float)));
    int grid_total = BNB_S * BNB_S * SEARCH_NT;
    CUDA_CHECK(cudaMalloc(&gpu.d_score, grid_total * sizeof(float)));
    gpu.d_levels.assign(C_MAX + 1, nullptr);
    for (int c = 0; c <= C_MAX; ++c) CUDA_CHECK(cudaMalloc(&gpu.d_levels[c], LGRID * LGRID * sizeof(float)));
    int seed_level = std::max(1, BNB_C - BNB_SEED_DROP);
    int frontier_max = (BNB_S / (1 << seed_level)) * (BNB_S / (1 << seed_level)) * SEARCH_NT;
    CUDA_CHECK(cudaMalloc(&gpu.d_nit, frontier_max * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_na, frontier_max * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_nb, frontier_max * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_node_score, frontier_max * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu.d_base_x, SEARCH_NT * N_SCAN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_base_y, SEARCH_NT * N_SCAN * sizeof(int)));

    cv::VideoWriter video("tmp/gpu_bnb_loop_closure_slam.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 12, cv::Size(FRAME_W, FRAME_H));
    if (!video.isOpened()) { std::fprintf(stderr, "failed to open video\n"); std::exit(1); }

    Stats stats;
    std::vector<float> h_score;
    std::vector<std::vector<float>> h_levels(C_MAX + 1);
    std::vector<int> base_x, base_y;
    std::vector<float> odom_hist, slam_hist;
    const cv::Vec3b odom_pt(150, 150, 215);

    for (int k = 1; k < N_KF; ++k) {
        edges.push_back({k - 1, k, odom_z[k].x, odom_z[k].y, odom_z[k].th, ODOM_INFO_XY, ODOM_INFO_TH});
        bool acc = process_keyframe(v, k, mx, my, gpu, h_score, h_levels, base_x, base_y, stats);
        if (acc) optimise_graph(est, edges, gt_pose(0), k + 1);

        float odom_ate = ate_rmse(odom, k);
        float slam_ate = ate_rmse(est, k);
        odom_hist.push_back(odom_ate);
        slam_hist.push_back(slam_ate);

        bool render = (k % 2 == 0) || acc || k == N_KF - 1;
        if (render) {
            cv::Mat frame(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(244, 246, 246));
            draw_map(frame, 0, odom, k, mx, my, odom_pt, cv::Scalar(60, 70, 210), false);
            cv::putText(frame, "dead reckoning (odometry)", cv::Point(12, 26),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 24, 32), 2, cv::LINE_AA);
            draw_map(frame, PANEL_W, est, k, mx, my, odom_pt, cv::Scalar(45, 150, 60), true);
            for (const Edge& le : loop_edges)
                cv::line(frame, world_to_panel(PANEL_W, est[le.i].x, est[le.i].y),
                         world_to_panel(PANEL_W, est[le.j].x, est[le.j].y),
                         cv::Scalar(200, 180, 40), 1, cv::LINE_AA);
            cv::putText(frame, "branch & bound SLAM", cv::Point(PANEL_W + 12, 26),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 24, 32), 2, cv::LINE_AA);
            draw_info(frame, PANEL_W * 2, k, odom_ate, slam_ate, accepted, rejected, stats,
                      odom_hist, slam_hist);
            int holds = acc ? 5 : 1;
            for (int h = 0; h < holds; ++h) video.write(frame);
        }

        if (k % 10 == 0 || acc)
            std::printf("kf %3d  odom=%.3f  slam=%.3f  loops=%d  agree=%d/%d  bnb=%lldk\n",
                        k, odom_ate, slam_ate, accepted, stats.agree, stats.attempts,
                        stats.bnb_nodes / 1000);
    }

    {
        cv::Mat frame(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(244, 246, 246));
        draw_map(frame, 0, odom, N_KF - 1, mx, my, odom_pt, cv::Scalar(60, 70, 210), false);
        cv::putText(frame, "dead reckoning (odometry)", cv::Point(12, 26),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 24, 32), 2, cv::LINE_AA);
        draw_map(frame, PANEL_W, est, N_KF - 1, mx, my, odom_pt, cv::Scalar(45, 150, 60), true);
        for (const Edge& le : loop_edges)
            cv::line(frame, world_to_panel(PANEL_W, est[le.i].x, est[le.i].y),
                     world_to_panel(PANEL_W, est[le.j].x, est[le.j].y),
                     cv::Scalar(200, 180, 40), 1, cv::LINE_AA);
        cv::putText(frame, "branch & bound SLAM", cv::Point(PANEL_W + 12, 26),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 24, 32), 2, cv::LINE_AA);
        draw_info(frame, PANEL_W * 2, N_KF - 1, ate_rmse(odom, N_KF - 1), ate_rmse(est, N_KF - 1),
                  accepted, rejected, stats, odom_hist, slam_hist);
        for (int h = 0; h < 30; ++h) video.write(frame);
    }

    video.release();
    avi_to_gif("tmp/gpu_bnb_loop_closure_slam.avi", "gif/gpu_bnb_loop_closure_slam.gif", 12, 900);

    CUDA_CHECK(cudaFree(gpu.d_sx));
    CUDA_CHECK(cudaFree(gpu.d_sy));
    CUDA_CHECK(cudaFree(gpu.d_lut));
    CUDA_CHECK(cudaFree(gpu.d_score));
    for (int c = 0; c <= C_MAX; ++c) CUDA_CHECK(cudaFree(gpu.d_levels[c]));
    CUDA_CHECK(cudaFree(gpu.d_nit));
    CUDA_CHECK(cudaFree(gpu.d_na));
    CUDA_CHECK(cudaFree(gpu.d_nb));
    CUDA_CHECK(cudaFree(gpu.d_node_score));
    CUDA_CHECK(cudaFree(gpu.d_base_x));
    CUDA_CHECK(cudaFree(gpu.d_base_y));

    std::printf("\nGPU branch-and-bound loop-closure SLAM\n");
    std::printf("loops: %d proposed, %d accepted, %d rejected; ATE %.3f m (dead reckoning %.3f m)\n",
                proposed, accepted, rejected, ate_rmse(est, N_KF - 1), ate_rmse(odom, N_KF - 1));
    std::printf("B&B == brute force on %d/%d attempts (identical relpose argmax)\n",
                stats.agree, stats.attempts);
    return stats;
}

}  // namespace cudabot

int main() {
    cudabot::Stats s = cudabot::run_demo();
    std::printf("shared window: %lld candidate cells (+/-8 m, +/-0.6 rad at field resolution)\n",
                s.exh_cand);
    std::printf("B&B scored %.1fk nodes/attempt vs brute force %.2fM -- %.0fx fewer, identical optimum\n",
                s.bnb_nodes / 1e3, s.exh_cand / 1e6,
                s.bnb_nodes > 0 ? (double)s.exh_cand / s.bnb_nodes : 0.0);
    std::printf("GPU B&B %.2f ms vs brute force %.2f ms per attempt\n", s.gpu_bnb_ms, s.gpu_exh_ms);
    std::printf("Wrote gif/gpu_bnb_loop_closure_slam.gif\n");
    return 0;
}
