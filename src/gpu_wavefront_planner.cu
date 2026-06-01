// gpu_wavefront_planner.cu
//
// GPU parallel wavefront planner - a Bellman-Ford-style cost-to-go relaxation
// over an occupancy grid that yields the exact shortest-path field from a goal,
// then extracts a path by greedy descent.  This is the parallel counterpart to
// the serial Dijkstra wavefront used in classic grid planners; it complements
// the repo's A*/Dijkstra demos with the throughput-oriented relaxation form.
//
// The map onto the canonical 2D idiom is:
//
//   one thread = one cell
//
// Each sweep, every free cell pulls its cost-to-go down to the cheapest
// neighbour-plus-edge:
//
//   D(p) = min( D(p), min_{q in N8(p), free} D(q) + w(p,q) )
//
// with integer edge weights (10 orthogonal, 14 diagonal ~ 10*sqrt(2)).  The
// goal is pinned to 0 and obstacles to INF.  Iterating to a fixpoint gives the
// exact single-source shortest-path field (label-correcting Bellman-Ford); the
// GPU batches several sweeps between host sync checks to avoid the per-iter
// changed-flag round-trip from dominating.
//
// Integer arithmetic + a deterministic min => the CPU and GPU cost fields and
// the extracted path are bit-identical.  We report the field agreement and that
// both ends pick the same path.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ----------------------------------------------------------------- constants
#define GRID_W 384
#define GRID_H 384
static const int N_CELLS = GRID_W * GRID_H;
static const int W_ORTHO = 10;
static const int W_DIAG  = 14;
static const int INF     = 1 << 28;
static const int MAX_ITERS   = 4096;
static const int CHECK_EVERY = 16;          // GPU sweeps between host sync checks

static const int PANEL_W = 620;
static const int PANEL_H = 660;

__host__ __device__ static inline int idx_of(int x, int y) { return y * GRID_W + x; }

__constant__ int  DCX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
__constant__ int  DCY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
__constant__ int  DCW[8] = {W_DIAG, W_ORTHO, W_DIAG, W_ORTHO, W_ORTHO, W_DIAG, W_ORTHO, W_DIAG};
static const int  HCX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
static const int  HCY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
static const int  HCW[8] = {W_DIAG, W_ORTHO, W_DIAG, W_ORTHO, W_ORTHO, W_DIAG, W_ORTHO, W_DIAG};

// --------------------------------------------------------------- environment
// 1 = obstacle (occupied), 0 = free.  A few rooms/walls with gaps so the
// shortest path has to wind around.
static void make_map(std::vector<unsigned char>& occ) {
    occ.assign(N_CELLS, 0);
    auto wall = [&](int x0, int y0, int x1, int y1) {
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
                if (x >= 0 && x < GRID_W && y >= 0 && y < GRID_H) occ[idx_of(x, y)] = 1;
    };
    int b = 3;
    wall(0, 0, GRID_W - 1, b); wall(0, GRID_H - 1 - b, GRID_W - 1, GRID_H - 1);
    wall(0, 0, b, GRID_H - 1); wall(GRID_W - 1 - b, 0, GRID_W - 1, GRID_H - 1);
    // vertical walls with a gap each (serpentine corridor)
    wall(100, 0, 112, 285);                 // gap at bottom
    wall(200, 100, 212, GRID_H - 1);        // gap at top
    wall(300, 0, 312, 270);                 // gap at bottom
    // a couple of blocks
    wall(135, 320, 190, 350);
    wall(240, 45, 285, 82);
}

// --------------------------------------------------------------- relaxation
__global__ void relax_kernel(const unsigned char* occ, int* D, int* changed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_CELLS) return;
    if (occ[i]) return;
    int x = i % GRID_W, y = i / GRID_W;
    int best = D[i];
    for (int k = 0; k < 8; ++k) {
        int nx = x + DCX[k], ny = y + DCY[k];
        if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H) continue;
        int j = idx_of(nx, ny);
        if (occ[j]) continue;
        int dq = D[j];
        if (dq >= INF) continue;
        int v = dq + DCW[k];
        if (v < best) best = v;
    }
    if (best < D[i]) { atomicMin(&D[i], best); atomicExch(changed, 1); }
}

// --------------------------------------------------------------- CPU pipeline
static void wavefront_cpu(const std::vector<unsigned char>& occ, int goal,
                          std::vector<int>& D, int& iters_out) {
    D.assign(N_CELLS, INF);
    D[goal] = 0;
    // in-place (Gauss-Seidel) relaxation: monotone min-plus updates converge to
    // the same unique shortest-path fixpoint as the GPU's atomicMin sweeps, so
    // the final field is bit-identical; in-place just gets there in fewer
    // sweeps and avoids a per-sweep copy.
    int iters = 0;
    for (iters = 0; iters < MAX_ITERS; ++iters) {
        bool changed = false;
        for (int i = 0; i < N_CELLS; ++i) {
            if (occ[i]) continue;
            int x = i % GRID_W, y = i / GRID_W, best = D[i];
            for (int k = 0; k < 8; ++k) {
                int nx = x + HCX[k], ny = y + HCY[k];
                if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H) continue;
                int j = idx_of(nx, ny);
                if (occ[j]) continue;
                int dq = D[j];
                if (dq >= INF) continue;
                int v = dq + HCW[k];
                if (v < best) best = v;
            }
            if (best < D[i]) { D[i] = best; changed = true; }
        }
        if (!changed) break;
    }
    iters_out = iters;
}

// greedy descent from start to goal along decreasing cost-to-go
static bool extract_path(const std::vector<unsigned char>& occ,
                         const std::vector<int>& D, int start, int goal,
                         std::vector<int>& path) {
    path.clear();
    int cur = start;
    if (D[start] >= INF) return false;
    for (int guard = 0; guard < N_CELLS; ++guard) {
        path.push_back(cur);
        if (cur == goal) return true;
        int x = cur % GRID_W, y = cur / GRID_W, best = D[cur], bn = -1;
        for (int k = 0; k < 8; ++k) {
            int nx = x + HCX[k], ny = y + HCY[k];
            if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H) continue;
            int j = idx_of(nx, ny);
            if (occ[j]) continue;
            if (D[j] < best) { best = D[j]; bn = j; }
        }
        if (bn < 0) return false;
        cur = bn;
    }
    return false;
}

// ------------------------------------------------------------- visualisation
static void draw(cv::Mat& out, const std::vector<unsigned char>& occ,
                 const std::vector<int>& D, const std::vector<int>& path,
                 int start, int goal, int maxfin,
                 const char* l1, const char* l2, const char* l3) {
    out = cv::Mat(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(28, 28, 32));
    const int GX = 54, GY = 70, GW = 512, GH = 512;
    cv::Mat gray(GRID_H, GRID_W, CV_8UC1);
    for (int i = 0; i < N_CELLS; ++i)
        gray.data[i] = (uint8_t)((maxfin > 0 && D[i] < INF) ? std::min(255, D[i] * 255 / maxfin) : 0);
    cv::Mat g; cv::applyColorMap(gray, g, cv::COLORMAP_JET);     // one colormap pass
    for (int i = 0; i < N_CELLS; ++i) {
        if (occ[i])            g.at<cv::Vec3b>(i / GRID_W, i % GRID_W) = {35, 35, 35};
        else if (D[i] >= INF)  g.at<cv::Vec3b>(i / GRID_W, i % GRID_W) = {88, 88, 92};
    }
    cv::Mat scaled; cv::resize(g, scaled, cv::Size(GW, GH), 0, 0, cv::INTER_NEAREST);
    scaled.copyTo(out(cv::Rect(GX, GY, GW, GH)));
    auto P = [&](int id) {
        return cv::Point(GX + (id % GRID_W) * GW / GRID_W, GY + (id / GRID_W) * GH / GRID_H);
    };
    for (size_t k = 1; k < path.size(); ++k)
        cv::line(out, P(path[k - 1]), P(path[k]), cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::circle(out, P(goal),  7, cv::Scalar(80, 255, 80), -1, cv::LINE_AA);
    cv::circle(out, P(start), 7, cv::Scalar(80, 80, 255), -1, cv::LINE_AA);
    cv::putText(out, l1, {14, 26}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {235, 235, 235}, 1, cv::LINE_AA);
    cv::putText(out, l2, {14, 48}, cv::FONT_HERSHEY_SIMPLEX, 0.46, {180, 220, 255}, 1, cv::LINE_AA);
    cv::putText(out, l3, {14, PANEL_H - 16}, cv::FONT_HERSHEY_SIMPLEX, 0.46, {180, 255, 180}, 1, cv::LINE_AA);
}

static int max_finite(const std::vector<int>& D) {
    int m = 1;
    for (int v : D) if (v < INF && v > m) m = v;
    return m;
}

// ===========================================================================
int main() {
    std::vector<unsigned char> occ;
    make_map(occ);
    int start = idx_of(30, 30), goal = idx_of(GRID_W - 30, GRID_H - 30);
    if (occ[start] || occ[goal]) { std::fprintf(stderr, "start/goal blocked\n"); return 1; }

    // ---------- CPU
    std::vector<int> D_cpu; int it_cpu = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    wavefront_cpu(occ, goal, D_cpu, it_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------- GPU
    unsigned char* d_occ; int *d_D, *d_changed;
    CUDA_CHECK(cudaMalloc(&d_occ, N_CELLS));
    CUDA_CHECK(cudaMalloc(&d_D, N_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_occ, occ.data(), N_CELLS, cudaMemcpyHostToDevice));
    int blk = 128, grid = (N_CELLS + blk - 1) / blk;

    std::vector<int> D_init(N_CELLS, INF); D_init[goal] = 0;
    auto run_gpu = [&](float* ms_out, int* it_out) {
        CUDA_CHECK(cudaMemcpy(d_D, D_init.data(), N_CELLS * sizeof(int), cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        int it = 0;
        while (it < MAX_ITERS) {
            int z = 0; CUDA_CHECK(cudaMemcpy(d_changed, &z, sizeof(int), cudaMemcpyHostToDevice));
            int batch = std::min(CHECK_EVERY, MAX_ITERS - it);
            for (int b = 0; b < batch; ++b) { relax_kernel<<<grid, blk>>>(d_occ, d_D, d_changed); ++it; }
            int ch; CUDA_CHECK(cudaMemcpy(&ch, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
            if (!ch) break;
        }
        CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
        if (ms_out) CUDA_CHECK(cudaEventElapsedTime(ms_out, e0, e1));
        if (it_out) *it_out = it;
    };
    float gpu_ms = 0.0f; int it_gpu = 0;
    run_gpu(nullptr, nullptr);                       // warm-up
    run_gpu(&gpu_ms, &it_gpu);                        // timed

    std::vector<int> D_gpu(N_CELLS);
    CUDA_CHECK(cudaMemcpy(D_gpu.data(), d_D, N_CELLS * sizeof(int), cudaMemcpyDeviceToHost));

    // ---------- compare
    int mism = 0, reached = 0;
    for (int i = 0; i < N_CELLS; ++i) {
        if (D_cpu[i] != D_gpu[i]) ++mism;
        if (!occ[i] && D_gpu[i] < INF) ++reached;
    }
    std::vector<int> path_cpu, path_gpu;
    bool ok_cpu = extract_path(occ, D_cpu, start, goal, path_cpu);
    bool ok_gpu = extract_path(occ, D_gpu, start, goal, path_gpu);
    bool path_same = (path_cpu.size() == path_gpu.size());
    if (path_same) for (size_t k = 0; k < path_cpu.size(); ++k) if (path_cpu[k] != path_gpu[k]) { path_same = false; break; }
    double speedup = cpu_ms / gpu_ms;

    std::printf("CPU %.2f ms (%d sweeps), GPU %.3f ms (%d sweeps)  -> %.0fx\n",
                cpu_ms, it_cpu, gpu_ms, it_gpu, speedup);
    std::printf("cost-field cell mismatches CPU vs GPU: %d / %d   reached cells %d\n",
                mism, N_CELLS, reached);
    std::printf("path: CPU %s (%zu), GPU %s (%zu), identical %s   goal cost %d\n",
                ok_cpu ? "ok" : "FAIL", path_cpu.size(), ok_gpu ? "ok" : "FAIL",
                path_gpu.size(), path_same ? "YES" : "NO", D_gpu[start]);

    // ---------- animation: wavefront expanding, then the path
    if (system("mkdir -p tmp") != 0) std::fprintf(stderr, "warning: mkdir tmp failed\n");
    cv::VideoWriter video("tmp/gpu_wavefront_planner.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          10, cv::Size(PANEL_W, PANEL_H));
    int maxfin = max_finite(D_gpu);
    CUDA_CHECK(cudaMemcpy(d_D, D_init.data(), N_CELLS * sizeof(int), cudaMemcpyHostToDevice));
    int it = 0; bool done = false;
    std::vector<int> empty_path;
    while (!done && it < MAX_ITERS) {
        int z = 0; CUDA_CHECK(cudaMemcpy(d_changed, &z, sizeof(int), cudaMemcpyHostToDevice));
        for (int b = 0; b < CHECK_EVERY; ++b) { relax_kernel<<<grid, blk>>>(d_occ, d_D, d_changed); ++it; }
        int ch; CUDA_CHECK(cudaMemcpy(&ch, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        if (!ch) done = true;
        std::vector<int> Dsnap(N_CELLS);
        CUDA_CHECK(cudaMemcpy(Dsnap.data(), d_D, N_CELLS * sizeof(int), cudaMemcpyDeviceToHost));
        int rc = 0; for (int i = 0; i < N_CELLS; ++i) if (!occ[i] && Dsnap[i] < INF) ++rc;
        char l1[200], l2[200], l3[200];
        std::snprintf(l1, sizeof(l1),
                      "GPU wavefront planner (one thread = one cell, Bellman-Ford relaxation)  %dx%d",
                      GRID_W, GRID_H);
        std::snprintf(l2, sizeof(l2), "cost-to-go from goal   sweeps: %d   reached: %.1f%%",
                      it, 100.0 * rc / N_CELLS);
        std::snprintf(l3, sizeof(l3),
                      "CPU %.0f ms vs GPU %.2f ms -> %.0fx   CPU/GPU field bit-identical (mismatch %d)",
                      cpu_ms, gpu_ms, speedup, mism);
        cv::Mat img; draw(img, occ, Dsnap, empty_path, start, goal, maxfin, l1, l2, l3);
        video.write(img);
    }
    // final frames: full field + extracted path
    {
        char l1[200], l2[200], l3[200];
        std::snprintf(l1, sizeof(l1),
                      "GPU wavefront planner (one thread = one cell, Bellman-Ford relaxation)  %dx%d",
                      GRID_W, GRID_H);
        std::snprintf(l2, sizeof(l2),
                      "shortest path extracted by greedy descent   length %zu   goal cost %d",
                      path_gpu.size(), D_gpu[start]);
        std::snprintf(l3, sizeof(l3),
                      "CPU %.0f ms vs GPU %.2f ms -> %.0fx   path CPU==GPU: %s",
                      cpu_ms, gpu_ms, speedup, path_same ? "yes" : "no");
        cv::Mat img; draw(img, occ, D_gpu, path_gpu, start, goal, maxfin, l1, l2, l3);
        for (int r = 0; r < 18; ++r) video.write(img);
    }
    video.release();
    cudabot::avi_to_gif("tmp/gpu_wavefront_planner.avi", "gif/gpu_wavefront_planner.gif", 10, 620);
    std::printf("wrote gif/gpu_wavefront_planner.gif\n");

    CUDA_CHECK(cudaFree(d_occ)); CUDA_CHECK(cudaFree(d_D)); CUDA_CHECK(cudaFree(d_changed));
    return 0;
}

}  // namespace cudabot

int main() { return cudabot::main(); }
