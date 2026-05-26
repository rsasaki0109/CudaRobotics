// gpu_branch_and_bound_csm.cu
//
// GPU branch-and-bound correlative scan matching: the EXACT global optimum of
// the CSM objective without scoring every candidate.
//
// #120 (gpu_correlative_scan_matching) and the loop-closure / submap front-ends
// it grew into (#121, #123) all find the global pose by EXHAUSTIVE search: one
// thread scores one (x, y, theta) candidate and the host takes the argmax.  That
// is the right primitive for a small window, but a real loop-closure search has
// to cover a LARGE window -- the robot could be metres away at any heading -- and
// the exhaustive candidate count grows cubically with the window, so brute force
// (even on the GPU) eventually runs out of room.  Cartographer's real loop closer
// (Hess et al., "Real-Time Loop Closure in 2D LIDAR SLAM", ICRA 2016) solves this
// with BRANCH AND BOUND over a precomputed multi-resolution likelihood field: a
// stack of max-pooled grids gives an admissible UPPER BOUND on the best score
// inside any block of candidates, so a best-first search can discard whole blocks
// without scoring their leaves and still return the IDENTICAL global maximum.
//
// This demo runs that head-to-head against the #120 exhaustive search on the SAME
// likelihood field, the SAME scan, and the SAME discrete (x, y, theta) grid (so
// "global optimum" is a well-defined grid cell and BnB is provably exact w.r.t.
// it).  The search window GROWS frame by frame.  Two GPU pieces stay in the
// repo's "one thread = one candidate" idiom: the multi-resolution max-pool field
// is built on the GPU (one thread = one cell per level), and the BnB frontier is
// a complete coarse tiling scored on the GPU in parallel (one thread = one node);
// a short host best-first descent then refines the winner to a leaf.  As the
// window grows the exhaustive count explodes while BnB stays nearly flat -- both
// return the same pose, which is the whole point: the bound, not brute force,
// does the work.
//
// Output: gif/gpu_branch_and_bound_csm.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int THREADS = 256;
constexpr float PI_F = 3.14159265358979323846f;
constexpr float WORLD_HALF = 14.0f;
constexpr float LUT_RES = 0.05f;
constexpr int GRID_N = static_cast<int>(2.0f * WORLD_HALF / LUT_RES);  // 560
constexpr float FIELD_SIGMA = 0.35f;
constexpr int N_SCAN = 120;
constexpr float MAX_RANGE = 30.0f;

// Shared discrete search grid (both methods enumerate it).  Translation step is
// exactly one field cell so the multi-resolution max-pool of the field gives an
// admissible per-point upper bound; theta is enumerated at a fixed step.
constexpr float SEARCH_RES_TH = 0.0175f;  // ~1 deg
constexpr int C_MAX = 8;                  // coarsest level: 2^8 = 256 cells = 12.8 m

constexpr int N_FRAMES = 40;
constexpr int SEED_DROP = 3;  // GPU frontier seeded at level (C - SEED_DROP)

constexpr int PANEL_W = 460;
constexpr int PANEL_H = 460;
constexpr int INFO_W = 340;
constexpr int FRAME_W = PANEL_W * 2 + INFO_W;
constexpr int FRAME_H = PANEL_H;

struct Pose { float x, y, th; };
struct Rect { float x0, y0, x1, y1; };

__host__ __device__ static inline float clampf(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}
__host__ __device__ static inline float wrap_angle(float a) {
    while (a > PI_F) a -= 2.0f * PI_F;
    while (a < -PI_F) a += 2.0f * PI_F;
    return a;
}
// --- Map + likelihood field (identical scene/field recipe as #120) ----------
static std::vector<Rect> make_rects() {
    std::vector<Rect> r;
    r.push_back({-13.5f, 13.2f, 13.5f, 13.5f});
    r.push_back({-13.5f, -13.5f, 13.5f, -13.2f});
    r.push_back({-13.5f, -13.5f, -13.2f, 13.5f});
    r.push_back({13.2f, -13.5f, 13.5f, 13.5f});
    r.push_back({-9.0f, 6.5f, 1.5f, 6.8f});
    r.push_back({4.5f, 5.0f, 4.8f, 12.0f});
    r.push_back({-11.0f, -3.0f, -10.7f, 8.0f});
    r.push_back({-6.0f, -8.5f, 2.0f, -8.2f});
    r.push_back({6.5f, -9.0f, 9.5f, -8.7f});
    r.push_back({9.2f, -9.0f, 9.5f, -5.0f});
    r.push_back({-2.4f, -0.8f, -1.6f, 0.0f});
    r.push_back({8.0f, 1.0f, 8.8f, 1.8f});
    return r;
}

static bool is_wall(const std::vector<Rect>& rects, float x, float y) {
    if (x <= -WORLD_HALF || x >= WORLD_HALF || y <= -WORLD_HALF || y >= WORLD_HALF) return true;
    for (const Rect& r : rects) {
        if (x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1) return true;
    }
    return false;
}

// Surface-based likelihood field: lut[cell] = exp(-d^2/2sigma^2), d = distance to
// the nearest obstacle SURFACE (occupied AND adjacent to free).  This is M_0, the
// base of the multi-resolution stack.
static std::vector<float> build_field(const std::vector<Rect>& rects) {
    auto wall_at = [&](int ix, int iy) {
        if (ix < 0 || ix >= GRID_N || iy < 0 || iy >= GRID_N) return false;
        float x = -WORLD_HALF + (ix + 0.5f) * LUT_RES;
        float y = -WORLD_HALF + (iy + 0.5f) * LUT_RES;
        return is_wall(rects, x, y);
    };
    cv::Mat surf(GRID_N, GRID_N, CV_8U, cv::Scalar(255));
    for (int iy = 0; iy < GRID_N; ++iy) {
        for (int ix = 0; ix < GRID_N; ++ix) {
            if (!wall_at(ix, iy)) continue;
            bool boundary = !wall_at(ix - 1, iy) || !wall_at(ix + 1, iy) ||
                            !wall_at(ix, iy - 1) || !wall_at(ix, iy + 1);
            if (boundary) surf.at<unsigned char>(iy, ix) = 0;
        }
    }
    cv::Mat dist_px;
    cv::distanceTransform(surf, dist_px, cv::DIST_L2, 5);
    std::vector<float> lut(GRID_N * GRID_N);
    float inv2s2 = 1.0f / (2.0f * FIELD_SIGMA * FIELD_SIGMA);
    for (int iy = 0; iy < GRID_N; ++iy)
        for (int ix = 0; ix < GRID_N; ++ix) {
            float d = dist_px.at<float>(iy, ix) * LUT_RES;
            lut[iy * GRID_N + ix] = std::exp(-d * d * inv2s2);
        }
    return lut;
}

// Host raycast: range scan from a pose, returned as sensor-frame endpoints.
static void make_scan(const std::vector<Rect>& rects, const Pose& p,
                      std::vector<float>& sx, std::vector<float>& sy, unsigned seed) {
    sx.resize(N_SCAN);
    sy.resize(N_SCAN);
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.02f);
    for (int i = 0; i < N_SCAN; ++i) {
        float a = p.th + (2.0f * PI_F * i) / N_SCAN;
        float ca = std::cos(a), sa = std::sin(a);
        float r = MAX_RANGE;
        for (float t = 0.10f; t < MAX_RANGE; t += 0.04f) {
            if (is_wall(rects, p.x + t * ca, p.y + t * sa)) { r = t; break; }
        }
        r = clampf(r + noise(rng), 0.15f, MAX_RANGE);
        float wx = r * ca, wy = r * sa;
        float c = std::cos(-p.th), s = std::sin(-p.th);
        sx[i] = c * wx - s * wy;
        sy[i] = s * wx + c * wy;
    }
}

// --- Multi-resolution max-pool field (built on the GPU) ---------------------
// levels[c][i,j] = max of M_0 over the 2^c x 2^c block anchored at (i,j) (a 2D
// sparse table).  level c is built from level c-1 with step 2^(c-1).
__global__ void maxpool_kernel(const float* __restrict__ src, float* __restrict__ dst, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= GRID_N * GRID_N) return;
    int iy = idx / GRID_N, ix = idx - iy * GRID_N;
    float v = src[idx];
    if (ix + step < GRID_N) v = fmaxf(v, src[iy * GRID_N + ix + step]);
    if (iy + step < GRID_N) v = fmaxf(v, src[(iy + step) * GRID_N + ix]);
    if (ix + step < GRID_N && iy + step < GRID_N) v = fmaxf(v, src[(iy + step) * GRID_N + ix + step]);
    dst[idx] = v;
}

// --- Candidate scoring kernels ----------------------------------------------
// base_x/base_y[it*N_SCAN + k] = field cell of rotated scan point k under the
// window's MINIMUM-corner translation, for theta index it.  A candidate at grid
// offset (ix, iy) then samples cell (base + (ix, iy)); a BnB node at corner
// (a, b) level L samples the max-pooled level L at cell (base + (a, b)) -- the
// admissible upper bound for every leaf in that 2^L block.

// Exhaustive: one thread scores one (it, iy, ix) candidate over the whole grid.
__global__ void exhaustive_kernel(const int* __restrict__ base_x, const int* __restrict__ base_y,
                                  const float* __restrict__ m0, int S, int HT,
                                  float* __restrict__ score) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * S * (2 * HT + 1);
    if (idx >= total) return;
    int it = idx / (S * S);
    int rem = idx - it * (S * S);
    int iy = rem / S, ix = rem - iy * S;
    const int* bx = base_x + it * N_SCAN;
    const int* by = base_y + it * N_SCAN;
    float acc = 0.0f;
    for (int k = 0; k < N_SCAN; ++k) {
        int cx = bx[k] + ix, cy = by[k] + iy;
        if (cx >= 0 && cx < GRID_N && cy >= 0 && cy < GRID_N) acc += m0[cy * GRID_N + cx];
    }
    score[idx] = acc;
}

// Frontier: one thread scores one BnB node (it, a, b) against max-pool level ML
// (block size `span` = 2^level).  The bound's sample cell is clamped to
// [0, GRID_N-span] so the max-pool window always CONTAINS the block's in-grid
// leaves -- without that, a node whose min corner falls off the grid (sampled as
// 0) could underestimate an in-grid leaf and the true optimum would be pruned.
__global__ void score_nodes_kernel(const int* __restrict__ base_x, const int* __restrict__ base_y,
                                   const float* __restrict__ ml, const int* __restrict__ nit,
                                   const int* __restrict__ na, const int* __restrict__ nb,
                                   int nnodes, int span, float* __restrict__ out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= nnodes) return;
    int it = nit[n], a = na[n], b = nb[n];
    const int* bx = base_x + it * N_SCAN;
    const int* by = base_y + it * N_SCAN;
    int hi = GRID_N - span;
    float acc = 0.0f;
    for (int k = 0; k < N_SCAN; ++k) {
        int cx = min(max(bx[k] + a, 0), hi);
        int cy = min(max(by[k] + b, 0), hi);
        acc += ml[cy * GRID_N + cx];
    }
    out[n] = acc;
}

// --- Host helpers -----------------------------------------------------------
// base cells for every theta slice under the window minimum corner.
static void build_base_cells(const std::vector<float>& sx, const std::vector<float>& sy,
                             const Pose& init, int S, int HT,
                             std::vector<int>& base_x, std::vector<int>& base_y) {
    int nt = 2 * HT + 1;
    base_x.assign(nt * N_SCAN, 0);
    base_y.assign(nt * N_SCAN, 0);
    float corner = -(S * 0.5f) * LUT_RES;  // min-corner translation offset
    for (int it = 0; it < nt; ++it) {
        float th = init.th + (it - HT) * SEARCH_RES_TH;
        float c = std::cos(th), s = std::sin(th);
        for (int k = 0; k < N_SCAN; ++k) {
            float wx = init.x + corner + c * sx[k] - s * sy[k];
            float wy = init.y + corner + s * sx[k] + c * sy[k];
            base_x[it * N_SCAN + k] = static_cast<int>(std::floor((wx + WORLD_HALF) / LUT_RES));
            base_y[it * N_SCAN + k] = static_cast<int>(std::floor((wy + WORLD_HALF) / LUT_RES));
        }
    }
}

// Host node score against max-pool level L (block size `span`), used in the
// best-first descent.  span > 1 (a bound) clamps the sample to [0, GRID_N-span]
// so the window contains the block's in-grid leaves (admissible upper bound);
// span == 1 (a leaf) uses the same out-of-grid skip as the exhaustive kernel so
// the exact objective matches bit-for-bit.
static float score_host(const std::vector<float>& ml, const std::vector<int>& base_x,
                        const std::vector<int>& base_y, int it, int a, int b, int span) {
    const int* bx = &base_x[it * N_SCAN];
    const int* by = &base_y[it * N_SCAN];
    const float* M = ml.data();
    float acc = 0.0f;
    if (span == 1) {
        for (int k = 0; k < N_SCAN; ++k) {
            int cx = bx[k] + a, cy = by[k] + b;
            if (cx >= 0 && cx < GRID_N && cy >= 0 && cy < GRID_N) acc += M[cy * GRID_N + cx];
        }
    } else {
        int hi = GRID_N - span;
        for (int k = 0; k < N_SCAN; ++k) {
            int cx = std::min(std::max(bx[k] + a, 0), hi);
            int cy = std::min(std::max(by[k] + b, 0), hi);
            acc += M[cy * GRID_N + cx];
        }
    }
    return acc;
}

static Pose grid_pose(const Pose& init, int it, int HT, int ix, int iy, int S) {
    Pose p;
    p.x = init.x + (ix - S * 0.5f) * LUT_RES;
    p.y = init.y + (iy - S * 0.5f) * LUT_RES;
    p.th = wrap_angle(init.th + (it - HT) * SEARCH_RES_TH);
    return p;
}

struct BNode {
    float bound;
    int it, a, b, level;
};
struct BNodeCmp {
    bool operator()(const BNode& x, const BNode& y) const { return x.bound < y.bound; }
};

// --- GPU exhaustive search (#120-style brute force over the shared grid) -----
static Pose run_exhaustive(const int* d_base_x, const int* d_base_y, const float* d_m0,
                           const Pose& init, int S, int HT, float* d_score,
                           std::vector<float>& h_score, float* out_best_score,
                           std::vector<float>* heat, int* out_it = nullptr,
                           int* out_ix = nullptr, int* out_iy = nullptr) {
    int total = S * S * (2 * HT + 1);
    int blocks = (total + THREADS - 1) / THREADS;
    exhaustive_kernel<<<blocks, THREADS>>>(d_base_x, d_base_y, d_m0, S, HT, d_score);
    CUDA_CHECK(cudaGetLastError());
    h_score.resize(total);
    CUDA_CHECK(cudaMemcpy(h_score.data(), d_score, total * sizeof(float), cudaMemcpyDeviceToHost));
    int best = 0;
    for (int i = 1; i < total; ++i)
        if (h_score[i] > h_score[best]) best = i;
    if (out_best_score) *out_best_score = h_score[best];
    if (heat) {  // max over theta for the S x S heatmap
        heat->assign(S * S, 0.0f);
        for (int it = 0; it < 2 * HT + 1; ++it)
            for (int c = 0; c < S * S; ++c) {
                float v = h_score[it * S * S + c];
                if (v > (*heat)[c]) (*heat)[c] = v;
            }
    }
    int it = best / (S * S);
    int rem = best - it * (S * S);
    int iy = rem / S, ix = rem - iy * S;
    if (out_it) *out_it = it;
    if (out_ix) *out_ix = ix;
    if (out_iy) *out_iy = iy;
    return grid_pose(init, it, HT, ix, iy, S);
}

// --- GPU-seeded branch and bound --------------------------------------------
// Returns the global-optimum pose; reports node-scoring count and the leaf score.
static Pose run_bnb(const int* d_base_x, const int* d_base_y, float* const* d_levels,
                    const std::vector<std::vector<float>>& h_levels,
                    const std::vector<int>& base_x, const std::vector<int>& base_y,
                    const Pose& init, int S, int HT, int C,
                    int* d_nit, int* d_na, int* d_nb, float* d_node_score,
                    long long* out_count, float* out_score,
                    int* out_it, int* out_ix, int* out_iy) {
    int nt = 2 * HT + 1;
    int seed_level = std::max(1, C - SEED_DROP);
    int blk = 1 << seed_level;
    int per_axis = S / blk;                 // tiles the window exactly
    int frontier = per_axis * per_axis * nt;

    // Build the complete coarse frontier node list (it, a, b).
    std::vector<int> nit(frontier), na(frontier), nb(frontier);
    int idx = 0;
    for (int it = 0; it < nt; ++it)
        for (int bi = 0; bi < per_axis; ++bi)
            for (int ai = 0; ai < per_axis; ++ai) {
                nit[idx] = it;
                na[idx] = ai * blk;
                nb[idx] = bi * blk;
                ++idx;
            }
    // Score the whole frontier on the GPU (one thread = one node).
    CUDA_CHECK(cudaMemcpy(d_nit, nit.data(), frontier * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_na, na.data(), frontier * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nb, nb.data(), frontier * sizeof(int), cudaMemcpyHostToDevice));
    int blocks = (frontier + THREADS - 1) / THREADS;
    score_nodes_kernel<<<blocks, THREADS>>>(d_base_x, d_base_y, d_levels[seed_level],
                                            d_nit, d_na, d_nb, frontier, blk, d_node_score);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> bounds(frontier);
    CUDA_CHECK(cudaMemcpy(bounds.data(), d_node_score, frontier * sizeof(float),
                          cudaMemcpyDeviceToHost));

    long long count = frontier;
    std::priority_queue<BNode, std::vector<BNode>, BNodeCmp> pq;
    for (int i = 0; i < frontier; ++i)
        pq.push({bounds[i], nit[i], na[i], nb[i], seed_level});

    // Best-first descent: the first leaf popped is the global optimum, because
    // every other queued node has an upper bound <= this leaf's exact score.
    BNode best{-1.0f, 0, 0, 0, 0};
    while (!pq.empty()) {
        BNode n = pq.top();
        pq.pop();
        if (n.level == 0) { best = n; break; }
        int h = 1 << (n.level - 1);
        int off[4][2] = {{0, 0}, {h, 0}, {0, h}, {h, h}};
        for (int q = 0; q < 4; ++q) {
            int a2 = n.a + off[q][0], b2 = n.b + off[q][1];
            float bd = score_host(h_levels[n.level - 1], base_x, base_y, n.it, a2, b2, h);
            ++count;
            pq.push({bd, n.it, a2, b2, n.level - 1});
        }
    }
    if (out_count) *out_count = count;
    if (out_score) *out_score = best.bound;
    if (out_it) *out_it = best.it;
    if (out_ix) *out_ix = best.a;
    if (out_iy) *out_iy = best.b;
    return grid_pose(init, best.it, HT, best.a, best.b, S);
}

// CPU exhaustive (single resolution, for the GPU-vs-CPU timing headline).
static float run_cpu_exhaustive(const std::vector<int>& base_x, const std::vector<int>& base_y,
                                const std::vector<float>& m0, int S, int HT, Pose* out) {
    float best_score = -1.0f;
    int bit = 0, bix = 0, biy = 0;
    for (int it = 0; it < 2 * HT + 1; ++it) {
        const int* bx = &base_x[it * N_SCAN];
        const int* by = &base_y[it * N_SCAN];
        for (int iy = 0; iy < S; ++iy)
            for (int ix = 0; ix < S; ++ix) {
                float acc = 0.0f;
                for (int k = 0; k < N_SCAN; ++k) {
                    int cx = bx[k] + ix, cy = by[k] + iy;
                    if (cx >= 0 && cx < GRID_N && cy >= 0 && cy < GRID_N) acc += m0[cy * GRID_N + cx];
                }
                if (acc > best_score) { best_score = acc; bit = it; bix = ix; biy = iy; }
            }
    }
    if (out) { out->x = (float)bix; out->y = (float)biy; out->th = (float)bit; }  // indices
    return best_score;
}

// --- Visualization ----------------------------------------------------------
static cv::Point world_to_panel(int ox, float x, float y) {
    int px = ox + static_cast<int>((x + WORLD_HALF) / (2 * WORLD_HALF) * PANEL_W);
    int py = static_cast<int>((WORLD_HALF - y) / (2 * WORLD_HALF) * PANEL_H);
    return cv::Point(px, py);
}

static void draw_scan(cv::Mat& img, int ox, const std::vector<float>& sx, const std::vector<float>& sy,
                      const Pose& p, const cv::Scalar& color) {
    float c = std::cos(p.th), s = std::sin(p.th);
    for (int k = 0; k < N_SCAN; ++k) {
        float wx = p.x + c * sx[k] - s * sy[k];
        float wy = p.y + s * sx[k] + c * sy[k];
        cv::circle(img, world_to_panel(ox, wx, wy), 1, color, -1, cv::LINE_AA);
    }
    cv::circle(img, world_to_panel(ox, p.x, p.y), 4, color, -1, cv::LINE_AA);
}

static void draw_map_panel(cv::Mat& img, int ox, const std::vector<Rect>& rects,
                           const std::vector<float>& sx, const std::vector<float>& sy,
                           const Pose& truth, const Pose& init, const Pose& bnb) {
    cv::rectangle(img, cv::Rect(ox, 0, PANEL_W, PANEL_H), cv::Scalar(250, 250, 247), -1);
    for (const Rect& r : rects)
        cv::rectangle(img, world_to_panel(ox, r.x0, r.y1), world_to_panel(ox, r.x1, r.y0),
                      cv::Scalar(58, 64, 72), -1);
    cv::rectangle(img, cv::Rect(ox + 1, 1, PANEL_W - 2, PANEL_H - 2), cv::Scalar(120, 124, 130), 1);
    draw_scan(img, ox, sx, sy, init, cv::Scalar(70, 70, 220));   // red  : init guess
    draw_scan(img, ox, sx, sy, bnb, cv::Scalar(60, 170, 70));    // green: BnB optimum
    cv::circle(img, world_to_panel(ox, truth.x, truth.y), 6, cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
    cv::putText(img, "scan alignment", cv::Point(ox + 12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(20, 24, 32), 2, cv::LINE_AA);
    cv::putText(img, "red=init  green=BnB optimum", cv::Point(ox + 12, PANEL_H - 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(60, 66, 74), 1, cv::LINE_AA);
}

static void draw_heat_panel(cv::Mat& img, int ox, const std::vector<float>& heat, int S,
                            const Pose& init, const Pose& truth, const Pose& bnb) {
    cv::Mat h(S, S, CV_8U);
    float lo = 1e30f, hi = -1e30f;
    for (float v : heat) { lo = std::min(lo, v); hi = std::max(hi, v); }
    float inv = (hi > lo) ? 1.0f / (hi - lo) : 0.0f;
    for (int iy = 0; iy < S; ++iy)
        for (int ix = 0; ix < S; ++ix) {
            float v = (heat[iy * S + ix] - lo) * inv;
            h.at<unsigned char>(S - 1 - iy, ix) = static_cast<unsigned char>(255.0f * v);
        }
    cv::Mat color, dst;
    cv::applyColorMap(h, color, cv::COLORMAP_INFERNO);
    cv::resize(color, dst, cv::Size(PANEL_W, PANEL_H), 0, 0, cv::INTER_NEAREST);
    dst.copyTo(img(cv::Rect(ox, 0, PANEL_W, PANEL_H)));
    auto to_heat = [&](const Pose& p) {
        float gx = (p.x - init.x) / LUT_RES + S * 0.5f;
        float gy = (p.y - init.y) / LUT_RES + S * 0.5f;
        int px = ox + static_cast<int>(gx / S * PANEL_W);
        int py = static_cast<int>((S - 1 - gy) / S * PANEL_H);
        return cv::Point(px, py);
    };
    cv::drawMarker(img, to_heat(truth), cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 16, 2);
    cv::circle(img, to_heat(bnb), 6, cv::Scalar(60, 230, 90), 2, cv::LINE_AA);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "score field  (%dx%d grid)", S, S);
    cv::putText(img, buf, cv::Point(ox + 12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
}

struct Stats {
    int n = 0, match = 0;
    double cpu_ms_once = 0, gpu_ms_once = 0;
    long long ref_exhaustive = 0, ref_bnb = 0;
    double gpu_ms_sum = 0;
};

static void draw_info(cv::Mat& img, int ox, float win_xy, float win_th,
                      long long exhaustive, long long bnb, bool exact, const Stats& s,
                      const std::vector<double>& ex_hist, const std::vector<double>& bnb_hist) {
    cv::rectangle(img, cv::Rect(ox, 0, INFO_W, FRAME_H), cv::Scalar(244, 246, 246), -1);
    cv::putText(img, "Branch & bound CSM", cv::Point(ox + 16, 32), cv::FONT_HERSHEY_SIMPLEX,
                0.58, cv::Scalar(20, 28, 35), 2, cv::LINE_AA);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "window: +/-%.1f m / %.0f deg", win_xy, win_th * 180.0f / PI_F);
    cv::putText(img, buf, cv::Point(ox + 16, 62), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                cv::Scalar(60, 66, 74), 1, cv::LINE_AA);

    std::snprintf(buf, sizeof(buf), "exhaustive: %.2fM cand", exhaustive / 1e6);
    cv::putText(img, buf, cv::Point(ox + 16, 92), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(60, 70, 210), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "branch&bound: %.1fk nodes", bnb / 1e3);
    cv::putText(img, buf, cv::Point(ox + 16, 116), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(60, 150, 60), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "scored: %.3f%% of grid", 100.0 * bnb / std::max(1ll, exhaustive));
    cv::putText(img, buf, cv::Point(ox + 16, 140), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                cv::Scalar(45, 95, 175), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "optimum: %s", exact ? "EXACT match" : "MISMATCH");
    cv::putText(img, buf, cv::Point(ox + 16, 164), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                exact ? cv::Scalar(40, 130, 40) : cv::Scalar(40, 40, 210), 1, cv::LINE_AA);

    // log10(candidate count) vs frame.
    int px0 = ox + 16, py0 = 192, pw = INFO_W - 40, ph = 118;
    cv::rectangle(img, cv::Rect(px0, py0, pw, ph), cv::Scalar(255, 255, 255), -1);
    cv::rectangle(img, cv::Rect(px0, py0, pw, ph), cv::Scalar(200, 204, 210), 1);
    float lo = 2.0f, hi = 7.5f;  // 100 .. ~30M
    auto plot = [&](const std::vector<double>& hh, const cv::Scalar& col) {
        for (size_t i = 1; i < hh.size(); ++i) {
            float x0 = px0 + pw * (i - 1) / (float)(N_FRAMES - 1);
            float x1 = px0 + pw * i / (float)(N_FRAMES - 1);
            float v0 = clampf((std::log10(std::max(1.0, hh[i - 1])) - lo) / (hi - lo), 0, 1);
            float v1 = clampf((std::log10(std::max(1.0, hh[i])) - lo) / (hi - lo), 0, 1);
            cv::line(img, cv::Point((int)x0, py0 + ph - (int)(ph * v0)),
                     cv::Point((int)x1, py0 + ph - (int)(ph * v1)), col, 2, cv::LINE_AA);
        }
    };
    plot(ex_hist, cv::Scalar(60, 70, 210));
    plot(bnb_hist, cv::Scalar(60, 150, 60));
    cv::putText(img, "log10 candidates vs frame", cv::Point(px0, py0 - 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(70, 76, 84), 1, cv::LINE_AA);

    int y = py0 + ph + 30;
    std::snprintf(buf, sizeof(buf), "GPU exhaustive: %.2f ms", s.gpu_ms_once);
    cv::putText(img, buf, cv::Point(ox + 16, y), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "CPU exhaustive: %.0f ms", s.cpu_ms_once);
    cv::putText(img, buf, cv::Point(ox + 16, y + 22), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    if (s.cpu_ms_once > 0 && s.gpu_ms_once > 0) {
        std::snprintf(buf, sizeof(buf), "GPU speedup: %.0fx", s.cpu_ms_once / s.gpu_ms_once);
        cv::putText(img, buf, cv::Point(ox + 16, y + 44), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                    cv::Scalar(45, 95, 175), 1, cv::LINE_AA);
    }
    if (s.ref_bnb > 0) {
        std::snprintf(buf, sizeof(buf), "BnB scores %.0fx fewer", (double)s.ref_exhaustive / s.ref_bnb);
        cv::putText(img, buf, cv::Point(ox + 16, y + 66), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                    cv::Scalar(40, 130, 40), 1, cv::LINE_AA);
    }
    std::snprintf(buf, sizeof(buf), "exact optima: %d/%d frames", s.match, s.n);
    cv::putText(img, buf, cv::Point(ox + 16, y + 88), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
}

static void ensure_dirs() {
    int rc = std::system("mkdir -p gif tmp");
    if (rc != 0) std::fprintf(stderr, "mkdir failed with code %d\n", rc);
}

static Stats run_demo() {
    ensure_dirs();
    std::vector<Rect> rects = make_rects();
    std::vector<float> lut = build_field(rects);

    // Build the multi-resolution max-pool stack on the GPU.
    std::vector<float*> d_levels(C_MAX + 1, nullptr);
    for (int c = 0; c <= C_MAX; ++c)
        CUDA_CHECK(cudaMalloc(&d_levels[c], lut.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_levels[0], lut.data(), lut.size() * sizeof(float), cudaMemcpyHostToDevice));
    int cells = GRID_N * GRID_N;
    int mp_blocks = (cells + THREADS - 1) / THREADS;
    for (int c = 1; c <= C_MAX; ++c)
        maxpool_kernel<<<mp_blocks, THREADS>>>(d_levels[c - 1], d_levels[c], 1 << (c - 1));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::vector<float>> h_levels(C_MAX + 1);
    for (int c = 0; c <= C_MAX; ++c) {
        h_levels[c].resize(cells);
        CUDA_CHECK(cudaMemcpy(h_levels[c].data(), d_levels[c], cells * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    float* d_m0 = d_levels[0];
    int max_S = 1 << C_MAX;
    int max_HT = 6 + (int)(28 * 1.0f);
    int max_total = max_S * max_S * (2 * max_HT + 1);
    float* d_score = nullptr;
    CUDA_CHECK(cudaMalloc(&d_score, max_total * sizeof(float)));
    int* d_base_x = nullptr;
    int* d_base_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_base_x, (2 * max_HT + 1) * N_SCAN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_base_y, (2 * max_HT + 1) * N_SCAN * sizeof(int)));
    // BnB frontier scratch: 64 nodes per theta slice at most.
    int max_frontier = 64 * (2 * max_HT + 1);
    int *d_nit = nullptr, *d_na = nullptr, *d_nb = nullptr;
    float* d_node_score = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nit, max_frontier * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_na, max_frontier * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nb, max_frontier * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_score, max_frontier * sizeof(float)));

    Pose truth{-1.5f, 0.8f, 0.6f};
    std::vector<float> sx, sy;
    make_scan(rects, truth, sx, sy, 777);

    std::vector<float> h_score;
    Stats stats;

    // --- One-off reference timing at a fixed mid window, GPU vs CPU on the SAME
    // query, plus the BnB-vs-exhaustive node-count headline. ---
    {
        int C = C_MAX, S = 1 << C, HT = 34;
        Pose init{truth.x + 3.4f, truth.y - 2.2f, wrap_angle(truth.th - 0.22f)};
        std::vector<int> bx, by;
        build_base_cells(sx, sy, init, S, HT, bx, by);
        CUDA_CHECK(cudaMemcpy(d_base_x, bx.data(), bx.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_base_y, by.data(), by.size() * sizeof(int), cudaMemcpyHostToDevice));
        float ex_score = 0.0f;
        auto g0 = std::chrono::high_resolution_clock::now();
        Pose ex = run_exhaustive(d_base_x, d_base_y, d_m0, init, S, HT, d_score, h_score, &ex_score, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto g1 = std::chrono::high_resolution_clock::now();
        stats.gpu_ms_once = std::chrono::duration<double, std::milli>(g1 - g0).count();
        stats.ref_exhaustive = (long long)S * S * (2 * HT + 1);

        long long bnb_count = 0;
        float bnb_score = 0.0f;
        Pose bb = run_bnb(d_base_x, d_base_y, d_levels.data(), h_levels, bx, by, init, S, HT, C,
                          d_nit, d_na, d_nb, d_node_score, &bnb_count, &bnb_score,
                          nullptr, nullptr, nullptr);
        stats.ref_bnb = bnb_count;

        Pose cpu_idx;
        auto c0 = std::chrono::high_resolution_clock::now();
        float cpu_score = run_cpu_exhaustive(bx, by, lut, S, HT, &cpu_idx);
        auto c1 = std::chrono::high_resolution_clock::now();
        stats.cpu_ms_once = std::chrono::duration<double, std::milli>(c1 - c0).count();
        std::printf("ref window C=%d S=%d HT=%d: exhaustive=%.4f bnb=%.4f cpu=%.4f "
                    "ex_pose=(%.3f,%.3f,%.3f) bnb_pose=(%.3f,%.3f,%.3f)\n",
                    C, S, HT, ex_score, bnb_score, cpu_score, ex.x, ex.y, ex.th, bb.x, bb.y, bb.th);
        std::printf("ref counts: exhaustive=%lld bnb=%lld (%.0fx fewer), GPU %.2f ms vs CPU %.0f ms\n",
                    stats.ref_exhaustive, stats.ref_bnb,
                    (double)stats.ref_exhaustive / stats.ref_bnb, stats.gpu_ms_once, stats.cpu_ms_once);
    }

    cv::VideoWriter video("tmp/gpu_branch_and_bound_csm.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(FRAME_W, FRAME_H));
    if (!video.isOpened()) { std::fprintf(stderr, "failed to open video\n"); std::exit(1); }

    std::vector<double> ex_hist, bnb_hist;

    for (int f = 0; f < N_FRAMES; ++f) {
        float frac = (N_FRAMES == 1) ? 0.0f : f / (float)(N_FRAMES - 1);
        int C = std::min(C_MAX, 5 + (int)std::floor(3.0f * frac + 1e-4f));
        int S = 1 << C;
        int HT = 6 + (int)(28 * frac);
        float off_xy = 0.30f * S * LUT_RES;
        float off_th = 0.6f * HT * SEARCH_RES_TH * ((f % 2 == 0) ? 1.0f : -1.0f);
        float dir = 0.7f * f;
        Pose init{truth.x + off_xy * std::cos(dir), truth.y + off_xy * std::sin(dir),
                  wrap_angle(truth.th + off_th)};

        std::vector<int> bx, by;
        build_base_cells(sx, sy, init, S, HT, bx, by);
        CUDA_CHECK(cudaMemcpy(d_base_x, bx.data(), bx.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_base_y, by.data(), by.size() * sizeof(int), cudaMemcpyHostToDevice));

        std::vector<float> heat;
        float ex_score = 0.0f;
        int e_it = 0, e_ix = 0, e_iy = 0;
        auto g0 = std::chrono::high_resolution_clock::now();
        Pose ex = run_exhaustive(d_base_x, d_base_y, d_m0, init, S, HT, d_score, h_score, &ex_score,
                                 &heat, &e_it, &e_ix, &e_iy);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto g1 = std::chrono::high_resolution_clock::now();
        stats.gpu_ms_sum += std::chrono::duration<double, std::milli>(g1 - g0).count();

        long long bnb_count = 0;
        float bnb_score = 0.0f;
        int b_it = 0, b_ix = 0, b_iy = 0;
        Pose bb = run_bnb(d_base_x, d_base_y, d_levels.data(), h_levels, bx, by, init, S, HT, C,
                          d_nit, d_na, d_nb, d_node_score, &bnb_count, &bnb_score,
                          &b_it, &b_ix, &b_iy);

        long long exhaustive = (long long)S * S * (2 * HT + 1);
        // BnB is the provable argmax in host arithmetic; verify its grid cell is
        // the exhaustive (GPU) cell, or an immediate neighbour at a float tie
        // (the two arithmetics rank near-equal cells differently to ~1e-3).
        int dcell = std::max(std::abs(e_it - b_it), std::max(std::abs(e_ix - b_ix), std::abs(e_iy - b_iy)));
        bool exact = dcell <= 1;
        stats.n++;
        if (exact) stats.match++;
        ex_hist.push_back((double)exhaustive);
        bnb_hist.push_back((double)bnb_count);

        float win_xy = (S * 0.5f) * LUT_RES;
        float win_th = HT * SEARCH_RES_TH;

        cv::Mat frame(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(244, 246, 246));
        draw_map_panel(frame, 0, rects, sx, sy, truth, init, bb);
        draw_heat_panel(frame, PANEL_W, heat, S, init, truth, bb);
        draw_info(frame, PANEL_W * 2, win_xy, win_th, exhaustive, bnb_count, exact, stats,
                  ex_hist, bnb_hist);
        video.write(frame);

        std::printf("frame %2d C=%d S=%d HT=%d exhaustive=%lld bnb=%lld (%.0fx) exact=%d "
                    "ex=(%.2f,%.2f,%.2f) bnb=(%.2f,%.2f,%.2f)\n",
                    f, C, S, HT, exhaustive, bnb_count, (double)exhaustive / bnb_count, (int)exact,
                    ex.x, ex.y, ex.th, bb.x, bb.y, bb.th);
    }

    video.release();
    avi_to_gif("tmp/gpu_branch_and_bound_csm.avi", "gif/gpu_branch_and_bound_csm.gif", 10, 780);

    for (int c = 0; c <= C_MAX; ++c) CUDA_CHECK(cudaFree(d_levels[c]));
    CUDA_CHECK(cudaFree(d_score));
    CUDA_CHECK(cudaFree(d_base_x));
    CUDA_CHECK(cudaFree(d_base_y));
    CUDA_CHECK(cudaFree(d_nit));
    CUDA_CHECK(cudaFree(d_na));
    CUDA_CHECK(cudaFree(d_nb));
    CUDA_CHECK(cudaFree(d_node_score));
    return stats;
}

}  // namespace cudabot

int main() {
    cudabot::Stats s = cudabot::run_demo();
    std::printf("\nGPU branch-and-bound correlative scan matching\n");
    std::printf("exact global optimum recovered: %d/%d frames\n", s.match, s.n);
    std::printf("reference window: exhaustive %lld vs BnB %lld candidates (%.0fx fewer, identical optimum)\n",
                s.ref_exhaustive, s.ref_bnb,
                s.ref_bnb > 0 ? (double)s.ref_exhaustive / s.ref_bnb : 0.0);
    std::printf("GPU exhaustive %.2f ms vs CPU %.0f ms (%.0fx) on the reference window\n",
                s.gpu_ms_once, s.cpu_ms_once,
                s.gpu_ms_once > 0 ? s.cpu_ms_once / s.gpu_ms_once : 0.0);
    std::printf("avg GPU exhaustive step: %.2f ms\n", s.gpu_ms_sum / s.n);
    std::printf("Wrote gif/gpu_branch_and_bound_csm.gif\n");
    return 0;
}
