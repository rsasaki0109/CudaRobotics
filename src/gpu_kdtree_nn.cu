// gpu_kdtree_nn.cu
//
// GPU KD-tree nearest-neighbour search - the foundational spatial-index query
// that the repo's point-cloud stack (voxel/statistical filtering, normal
// estimation, GICP, RANSAC) leans on but never demonstrated as a primitive.
//
// A balanced KD-tree is built once on the host (recursive median split) and
// uploaded as flat arrays.  The queries are embarrassingly parallel, so the GPU
// map is:
//
//   one thread = one query point
//
// Each thread descends the tree to the query's leaf, then backtracks, pruning
// any subtree whose splitting-plane distance already exceeds the best distance
// found - the textbook exact NN search, with an explicit stack (no recursion).
//
// The nearest neighbour is an exact argmin over squared distances, and the
// KD-tree returns the SAME neighbour as an exhaustive brute-force scan (it
// prunes only provably-farther subtrees).  With random float coordinates ties
// are measure-zero, so the GPU KD-tree and the CPU brute force agree on the
// neighbour index for 100% of queries - we report that match exactly.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ----------------------------------------------------------------- constants
#define N_PTS   40000
#define N_QUERY 40000
static const float WORLD = 30.0f;
static const int   STACK_MAX = 64;

static const int PANEL_W = 720;
static const int PANEL_H = 720;

// KD-tree node (flat arrays): the point it stores, split axis, children
struct KdTree {
    std::vector<float> px, py;          // points (original order)
    std::vector<int>   node_pt;         // node -> point index
    std::vector<int>   node_ax;         // split axis (0/1)
    std::vector<int>   node_l, node_r;  // children (-1 = none)
    int root = -1;
};

// recursive median-split build over an index range
static int build_rec(KdTree& t, std::vector<int>& idx, int lo, int hi, int depth) {
    if (lo >= hi) return -1;
    int axis = depth & 1;
    int mid = (lo + hi) / 2;
    std::nth_element(idx.begin() + lo, idx.begin() + mid, idx.begin() + hi,
                     [&](int a, int b) {
                         return (axis == 0 ? t.px[a] : t.py[a]) < (axis == 0 ? t.px[b] : t.py[b]);
                     });
    int pt = idx[mid];
    int n = (int)t.node_pt.size();
    t.node_pt.push_back(pt); t.node_ax.push_back(axis);
    t.node_l.push_back(-1);  t.node_r.push_back(-1);
    int l = build_rec(t, idx, lo, mid, depth + 1);
    int r = build_rec(t, idx, mid + 1, hi, depth + 1);
    t.node_l[n] = l; t.node_r[n] = r;
    return n;
}

// ----------------------------------------------------------- NN search (shared)
__host__ __device__ static inline int nn_search(
        float qx, float qy, const float* px, const float* py,
        const int* npt, const int* nax, const int* nl, const int* nr, int root) {
    int stack[STACK_MAX]; float gate[STACK_MAX]; int sp = 0;
    stack[sp] = root; gate[sp] = 0.0f; ++sp;
    float best = 1e30f; int best_idx = -1;
    while (sp > 0) {
        --sp;
        int node = stack[sp]; float g = gate[sp];
        if (node < 0 || g >= best) continue;
        int pt = npt[node];
        float dx = px[pt] - qx, dy = py[pt] - qy;
        float d = dx * dx + dy * dy;
        if (d < best) { best = d; best_idx = pt; }
        int ax = nax[node];
        float sd = (ax == 0 ? qx - px[pt] : qy - py[pt]);
        int near = sd <= 0.0f ? nl[node] : nr[node];
        int far  = sd <= 0.0f ? nr[node] : nl[node];
        // push far first (gated by plane distance), near last (explored first)
        if (sp < STACK_MAX - 1) { stack[sp] = far;  gate[sp] = sd * sd; ++sp; }
        if (sp < STACK_MAX - 1) { stack[sp] = near; gate[sp] = 0.0f;    ++sp; }
    }
    return best_idx;
}

__global__ void nn_kernel(const float* qx, const float* qy, int nq,
                          const float* px, const float* py,
                          const int* npt, const int* nax, const int* nl, const int* nr,
                          int root, int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nq) return;
    out[i] = nn_search(qx[i], qy[i], px, py, npt, nax, nl, nr, root);
}

// brute-force exact NN (CPU reference)
static int brute_nn(float qx, float qy, const std::vector<float>& px,
                    const std::vector<float>& py) {
    float best = 1e30f; int bi = -1;
    for (int j = 0; j < N_PTS; ++j) {
        float dx = px[j] - qx, dy = py[j] - qy, d = dx * dx + dy * dy;
        if (d < best) { best = d; bi = j; }
    }
    return bi;
}

// ------------------------------------------------------------- visualisation
static void draw_partition(cv::Mat& img, const KdTree& t, int node,
                           float xmin, float xmax, float ymin, float ymax,
                           int depth, int maxdepth,
                           std::function<cv::Point(float,float)> P) {
    if (node < 0 || depth > maxdepth) return;
    int pt = t.node_pt[node], ax = t.node_ax[node];
    if (ax == 0) {
        float sx = t.px[pt];
        cv::line(img, P(sx, ymin), P(sx, ymax), cv::Scalar(70, 70, 80), 1, cv::LINE_AA);
        draw_partition(img, t, t.node_l[node], xmin, sx, ymin, ymax, depth + 1, maxdepth, P);
        draw_partition(img, t, t.node_r[node], sx, xmax, ymin, ymax, depth + 1, maxdepth, P);
    } else {
        float sy = t.py[pt];
        cv::line(img, P(xmin, sy), P(xmax, sy), cv::Scalar(70, 70, 80), 1, cv::LINE_AA);
        draw_partition(img, t, t.node_l[node], xmin, xmax, ymin, sy, depth + 1, maxdepth, P);
        draw_partition(img, t, t.node_r[node], xmin, xmax, sy, ymax, depth + 1, maxdepth, P);
    }
}

// ===========================================================================
int main() {
    // ---------- points + queries
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(1.0f, WORLD - 1.0f);
    KdTree t;
    t.px.resize(N_PTS); t.py.resize(N_PTS);
    for (int i = 0; i < N_PTS; ++i) { t.px[i] = U(rng); t.py[i] = U(rng); }
    std::vector<float> qx(N_QUERY), qy(N_QUERY);
    for (int i = 0; i < N_QUERY; ++i) { qx[i] = U(rng); qy[i] = U(rng); }

    // ---------- build tree (host)
    std::vector<int> idx(N_PTS);
    for (int i = 0; i < N_PTS; ++i) idx[i] = i;
    t.node_pt.reserve(N_PTS); t.node_ax.reserve(N_PTS);
    t.node_l.reserve(N_PTS);  t.node_r.reserve(N_PTS);
    auto tb0 = std::chrono::high_resolution_clock::now();
    t.root = build_rec(t, idx, 0, N_PTS, 0);
    auto tb1 = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();

    // ---------- CPU brute force (reference)
    std::vector<int> nn_brute(N_QUERY);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_QUERY; ++i) nn_brute[i] = brute_nn(qx[i], qy[i], t.px, t.py);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_brute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------- CPU KD-tree (reference for the same algorithm)
    std::vector<int> nn_cpu(N_QUERY);
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_QUERY; ++i)
        nn_cpu[i] = nn_search(qx[i], qy[i], t.px.data(), t.py.data(),
                              t.node_pt.data(), t.node_ax.data(),
                              t.node_l.data(), t.node_r.data(), t.root);
    auto t3 = std::chrono::high_resolution_clock::now();
    double cpu_kd_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // ---------- GPU KD-tree
    int nnodes = (int)t.node_pt.size();
    float *d_px, *d_py, *d_qx, *d_qy; int *d_npt, *d_nax, *d_nl, *d_nr, *d_out;
    CUDA_CHECK(cudaMalloc(&d_px, N_PTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py, N_PTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qx, N_QUERY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qy, N_QUERY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_npt, nnodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nax, nnodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nl, nnodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nr, nnodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, N_QUERY * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_px, t.px.data(), N_PTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, t.py.data(), N_PTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qx, qx.data(), N_QUERY * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qy, qy.data(), N_QUERY * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_npt, t.node_pt.data(), nnodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nax, t.node_ax.data(), nnodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nl, t.node_l.data(), nnodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nr, t.node_r.data(), nnodes * sizeof(int), cudaMemcpyHostToDevice));

    int blk = 128, grid = (N_QUERY + blk - 1) / blk;
    auto run_gpu = [&](float* ms_out) {
        cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        nn_kernel<<<grid, blk>>>(d_qx, d_qy, N_QUERY, d_px, d_py,
                                 d_npt, d_nax, d_nl, d_nr, t.root, d_out);
        CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
        if (ms_out) CUDA_CHECK(cudaEventElapsedTime(ms_out, e0, e1));
    };
    float gpu_ms = 0.0f;
    run_gpu(nullptr);                                // warm-up
    run_gpu(&gpu_ms);                                // timed

    std::vector<int> nn_gpu(N_QUERY);
    CUDA_CHECK(cudaMemcpy(nn_gpu.data(), d_out, N_QUERY * sizeof(int), cudaMemcpyDeviceToHost));

    // ---------- compare
    int mism_kd = 0, mism_brute = 0;
    for (int i = 0; i < N_QUERY; ++i) {
        if (nn_gpu[i] != nn_cpu[i]) ++mism_kd;
        if (nn_gpu[i] != nn_brute[i]) ++mism_brute;
    }
    double sp_brute = cpu_brute_ms / gpu_ms;
    double sp_kd    = cpu_kd_ms / gpu_ms;

    std::printf("build %.2f ms (%d nodes)\n", build_ms, nnodes);
    std::printf("CPU brute %.2f ms, CPU kd %.2f ms, GPU kd %.3f ms\n",
                cpu_brute_ms, cpu_kd_ms, gpu_ms);
    std::printf("GPU kd vs brute force: %d / %d mismatches (exact-NN agreement %.4f%%)\n",
                mism_brute, N_QUERY, 100.0 * (N_QUERY - mism_brute) / N_QUERY);
    std::printf("GPU kd vs CPU kd: %d mismatches   speedup %.0fx (vs brute), %.0fx (vs CPU kd)\n",
                mism_kd, sp_brute, sp_kd);

    // ---------- animation: sweeping query + its nearest neighbour
    if (system("mkdir -p tmp") != 0) std::fprintf(stderr, "warning: mkdir tmp failed\n");
    cv::VideoWriter video("tmp/gpu_kdtree_nn.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          12, cv::Size(PANEL_W, PANEL_H));
    const int GX = 20, GY = 70, GW = 680, GH = 620;
    auto P = [&](float x, float y) {
        return cv::Point(GX + (int)(x / WORLD * GW), GY + (int)(y / WORLD * GH));
    };
    // base image: partition lines + points
    cv::Mat base(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(24, 24, 28));
    draw_partition(base, t, t.root, 0, WORLD, 0, WORLD, 0, 7, P);
    // subsample the drawn points so the KD-tree partition is visible (the
    // benchmark + queries still use all N_PTS points).
    for (int j = 0; j < N_PTS; j += 13)
        cv::circle(base, P(t.px[j], t.py[j]), 2, cv::Scalar(120, 150, 170), -1, cv::LINE_AA);

    int NF = 48;
    for (int f = 0; f < NF; ++f) {
        double a = 2.0 * M_PI * f / NF;
        float cqx = WORLD * 0.5f + 9.0f * (float)std::cos(a);
        float cqy = WORLD * 0.5f + 9.0f * (float)std::sin(1.7 * a);
        int nn = nn_search(cqx, cqy, t.px.data(), t.py.data(), t.node_pt.data(),
                           t.node_ax.data(), t.node_l.data(), t.node_r.data(), t.root);
        cv::Mat img = base.clone();
        if (nn >= 0) {
            cv::line(img, P(cqx, cqy), P(t.px[nn], t.py[nn]), cv::Scalar(80, 255, 255), 2, cv::LINE_AA);
            cv::circle(img, P(t.px[nn], t.py[nn]), 6, cv::Scalar(80, 255, 80), 2, cv::LINE_AA);
        }
        cv::circle(img, P(cqx, cqy), 6, cv::Scalar(80, 80, 255), -1, cv::LINE_AA);
        char l1[200], l2[200];
        std::snprintf(l1, sizeof(l1),
                      "GPU KD-tree nearest-neighbour (one thread = one query)  %d points  %d queries",
                      N_PTS, N_QUERY);
        std::snprintf(l2, sizeof(l2),
                      "exact NN: GPU kd == CPU brute force (%.2f%%)   GPU %.2f ms vs brute %.0f ms -> %.0fx",
                      100.0 * (N_QUERY - mism_brute) / N_QUERY, gpu_ms, cpu_brute_ms, sp_brute);
        cv::putText(img, l1, {16, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {235,235,235}, 1, cv::LINE_AA);
        cv::putText(img, l2, {16, 52}, cv::FONT_HERSHEY_SIMPLEX, 0.46, {180,255,180}, 1, cv::LINE_AA);
        video.write(img);
    }
    video.release();
    cudabot::avi_to_gif("tmp/gpu_kdtree_nn.avi", "gif/gpu_kdtree_nn.gif", 12, 720);
    std::printf("wrote gif/gpu_kdtree_nn.gif\n");

    CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_qx)); CUDA_CHECK(cudaFree(d_qy));
    CUDA_CHECK(cudaFree(d_npt)); CUDA_CHECK(cudaFree(d_nax));
    CUDA_CHECK(cudaFree(d_nl)); CUDA_CHECK(cudaFree(d_nr)); CUDA_CHECK(cudaFree(d_out));
    return 0;
}

}  // namespace cudabot

int main() { return cudabot::main(); }
