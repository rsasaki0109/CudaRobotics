/*************************************************************************
    Massively-parallel RRT* rewire comparison.

    Setup: a random forest of N nodes (uniform in a 40x40 world with
    static disk obstacles) is built sequentially, each node connecting
    to its nearest predecessor that is collision-free. This produces a
    valid (but sub-optimal) tree rooted at node 0.

    Rewire step (the focus of this benchmark):
      For each node i, search all neighbours j with euclidean distance
      < REWIRE_RADIUS. If cost[j] + dist(i, j) < cost[i] and the
      segment i->j is collision-free, set parent[i] = j and propagate
      the cost update.

    CPU baseline: 1 thread, two nested loops, with chase-the-parent
    propagation after each adoption. N_CPU = 2,000.
    GPU kernel:   1 thread per node, candidate parent selection done in
                  parallel over a uniform grid (cell = rewire radius, so
                  a node only scans its 3x3 cells); cost propagation done
                  via 4 fixed-point iterations (more nodes => more
                  iterations needed). N_GPU = 200,000 (100x larger forest).

    Headline metric: per-node rewire throughput.
 ************************************************************************/

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include "cuda_check.cuh"
#include "cuda_video.h"

constexpr float WORLD = 40.0f;
constexpr int   N_CPU = 2000;
constexpr int   N_GPU = 200000;
constexpr float REWIRE_RADIUS = 2.5f;
constexpr int   N_OBS = 12;
constexpr float OBS_RADIUS = 1.6f;
constexpr int   PANEL = 540;
constexpr int   ITERS = 4;
// Neighbour grid for the GPU rewire. Cells are slightly larger than the
// radius so every neighbour within it is in the 3x3 cells around a node.
constexpr float GRID_CELL = REWIRE_RADIUS * 1.001f;
constexpr int   GRID_DIM = static_cast<int>(WORLD / GRID_CELL) + 1;
constexpr int   GRID_CELLS = GRID_DIM * GRID_DIM;

struct Disk { float cx, cy, r; };

// -------------------------------------------------------------------------
// Scene
// -------------------------------------------------------------------------
static std::vector<Disk> make_obstacles(std::mt19937& rng) {
    std::uniform_real_distribution<float> u(3.0f, WORLD - 3.0f);
    std::vector<Disk> ob;
    ob.reserve(N_OBS);
    for (int i = 0; i < N_OBS; i++) ob.push_back({u(rng), u(rng), OBS_RADIUS});
    return ob;
}

__host__ __device__ static inline bool seg_clear(float ax, float ay, float bx, float by,
                                                 const Disk* obs, int n_ob) {
    float dx = bx - ax, dy = by - ay;
    float len = sqrtf(dx * dx + dy * dy);
    int steps = static_cast<int>(len / 0.15f) + 1;
    if (steps > 64) steps = 64;
    for (int s = 0; s <= steps; s++) {
        float t = static_cast<float>(s) / steps;
        float x = ax + t * dx;
        float y = ay + t * dy;
        for (int i = 0; i < n_ob; i++) {
            float ex = x - obs[i].cx, ey = y - obs[i].cy;
            if (ex * ex + ey * ey <= obs[i].r * obs[i].r) return false;
        }
    }
    return true;
}

// -------------------------------------------------------------------------
// Sampling: produce N valid nodes outside any obstacle
// -------------------------------------------------------------------------
static std::vector<float> sample_nodes(int N, const std::vector<Disk>& obs,
                                       std::mt19937& rng) {
    std::uniform_real_distribution<float> u(0.5f, WORLD - 0.5f);
    std::vector<float> pts;
    pts.reserve(N * 2);
    pts.push_back(1.0f); pts.push_back(1.0f);  // root
    while (static_cast<int>(pts.size() / 2) < N) {
        float x = u(rng), y = u(rng);
        bool inside = false;
        for (const auto& o : obs) {
            float dx = x - o.cx, dy = y - o.cy;
            if (dx * dx + dy * dy <= o.r * o.r) { inside = true; break; }
        }
        if (inside) continue;
        pts.push_back(x); pts.push_back(y);
    }
    return pts;
}

// -------------------------------------------------------------------------
// Build initial tree (sequential, host)
// -------------------------------------------------------------------------
static void build_initial_tree(const std::vector<float>& pts,
                               const std::vector<Disk>& obs,
                               std::vector<int>& parent,
                               std::vector<float>& cost) {
    int N = static_cast<int>(pts.size() / 2);
    parent.assign(N, -1);
    cost.assign(N, 0.0f);
    for (int i = 1; i < N; i++) {
        float xi = pts[i * 2], yi = pts[i * 2 + 1];
        int best = -1;
        float best_d = FLT_MAX;
        // connect to nearest predecessor (collision-free)
        for (int j = 0; j < i; j++) {
            float xj = pts[j * 2], yj = pts[j * 2 + 1];
            float dx = xi - xj, dy = yi - yj;
            float d2 = dx * dx + dy * dy;
            if (d2 < best_d && seg_clear(xi, yi, xj, yj, obs.data(),
                                         static_cast<int>(obs.size()))) {
                best_d = d2;
                best = j;
            }
            if (best >= 0 && d2 > 4.0f * REWIRE_RADIUS * REWIRE_RADIUS) {
                // Early exit heuristic on large i — we already have a candidate
                // and this one is too far to beat by much.
            }
        }
        if (best < 0) best = 0;  // fallback to root
        parent[i] = best;
        cost[i] = cost[best] + std::sqrt(best_d);
    }
}

// -------------------------------------------------------------------------
// CPU rewire (O(N^2))
// -------------------------------------------------------------------------
static double cpu_rewire_ms(const std::vector<float>& pts,
                            const std::vector<Disk>& obs,
                            std::vector<int>& parent,
                            std::vector<float>& cost,
                            int iters) {
    int N = static_cast<int>(pts.size() / 2);
    auto t0 = std::chrono::high_resolution_clock::now();
    int n_ob = static_cast<int>(obs.size());
    for (int it = 0; it < iters; it++) {
        for (int i = 1; i < N; i++) {
            float xi = pts[i * 2], yi = pts[i * 2 + 1];
            float best_cost = cost[i];
            int best_par = parent[i];
            for (int j = 0; j < N; j++) {
                if (j == i) continue;
                float xj = pts[j * 2], yj = pts[j * 2 + 1];
                float dx = xi - xj, dy = yi - yj;
                float d2 = dx * dx + dy * dy;
                if (d2 > REWIRE_RADIUS * REWIRE_RADIUS) continue;
                float d = std::sqrt(d2);
                float c = cost[j] + d;
                if (c < best_cost && seg_clear(xi, yi, xj, yj, obs.data(), n_ob)) {
                    best_cost = c;
                    best_par = j;
                }
            }
            if (best_par != parent[i]) {
                parent[i] = best_par;
                cost[i] = best_cost;
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// -------------------------------------------------------------------------
// GPU rewire kernels
// -------------------------------------------------------------------------
__host__ __device__ static inline int grid_coord(float v) {
    int c = static_cast<int>(v / GRID_CELL);
    return c < 0 ? 0 : (c >= GRID_DIM ? GRID_DIM - 1 : c);
}

__global__ void grid_count_kernel(const float* __restrict__ pts, int N,
                                  int* __restrict__ cell_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int c = grid_coord(pts[i * 2 + 1]) * GRID_DIM + grid_coord(pts[i * 2]);
    atomicAdd(&cell_count[c], 1);
}

// Exclusive scan of the few hundred cell counts; one thread is enough.
__global__ void grid_scan_kernel(const int* __restrict__ cell_count,
                                 int* __restrict__ cell_start,
                                 int* __restrict__ cell_fill) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int sum = 0;
    for (int c = 0; c < GRID_CELLS; c++) {
        cell_start[c] = sum;
        cell_fill[c] = 0;
        sum += cell_count[c];
    }
    cell_start[GRID_CELLS] = sum;
}

__global__ void grid_scatter_kernel(const float* __restrict__ pts, int N,
                                    const int* __restrict__ cell_start,
                                    int* __restrict__ cell_fill,
                                    int* __restrict__ cell_node,
                                    float2* __restrict__ cell_pts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x = pts[i * 2], y = pts[i * 2 + 1];
    int c = grid_coord(y) * GRID_DIM + grid_coord(x);
    int k = cell_start[c] + atomicAdd(&cell_fill[c], 1);
    cell_node[k] = i;
    cell_pts[k] = make_float2(x, y);
}

// Same choice as a scan over all j in index order: the cheapest collision-free
// parent below the current cost, the smallest index on ties (the grid visits
// nodes in a different order, so ties are broken explicitly).
__global__ void rewire_kernel(const float* __restrict__ cost_in,
                              int* __restrict__ parent_out,
                              float* __restrict__ cost_out,
                              const Disk* __restrict__ obs, int n_ob,
                              const int* __restrict__ cell_start,
                              const int* __restrict__ cell_node,
                              const float2* __restrict__ cell_pts,
                              int N, float radius) {
    // Threads take nodes in grid order: the lanes of a warp then scan the same
    // cells in step, so their loads of neighbour data are shared.
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    int i = cell_node[t];
    if (i == 0) {
        cost_out[i] = cost_in[i];
        return;
    }
    float xi = cell_pts[t].x, yi = cell_pts[t].y;
    float best_cost = cost_in[i];
    int   best_par  = -1;
    float r2 = radius * radius;
    int gx = grid_coord(xi), gy = grid_coord(yi);
    for (int cy = max(gy - 1, 0); cy <= min(gy + 1, GRID_DIM - 1); cy++) {
        for (int cx = max(gx - 1, 0); cx <= min(gx + 1, GRID_DIM - 1); cx++) {
            int c = cy * GRID_DIM + cx;
            for (int k = cell_start[c]; k < cell_start[c + 1]; k++) {
                int j = cell_node[k];
                if (j == i) continue;
                float2 pj = cell_pts[k];
                float dx = xi - pj.x, dy = yi - pj.y;
                float d2 = dx * dx + dy * dy;
                if (d2 > r2) continue;
                float d = sqrtf(d2);
                float c_j = cost_in[j] + d;
                bool better = c_j < best_cost
                           || (c_j == best_cost && best_par >= 0 && j < best_par);
                if (better && seg_clear(xi, yi, pj.x, pj.y, obs, n_ob)) {
                    best_cost = c_j;
                    best_par  = j;
                }
            }
        }
    }
    if (best_par >= 0) {
        parent_out[i] = best_par;
        cost_out[i]   = best_cost;
    } else {
        cost_out[i] = cost_in[i];
    }
}

__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i];
}

static double gpu_rewire_ms(const std::vector<float>& pts,
                            const std::vector<Disk>& obs,
                            std::vector<int>& parent,
                            std::vector<float>& cost,
                            int iters) {
    int N = static_cast<int>(pts.size() / 2);
    int n_ob = static_cast<int>(obs.size());

    float* d_pts = nullptr;
    Disk*  d_obs = nullptr;
    int*   d_parent = nullptr;
    float* d_cost_a = nullptr;
    float* d_cost_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pts,    N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obs,    n_ob * sizeof(Disk)));
    CUDA_CHECK(cudaMalloc(&d_parent, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cost_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cost_b, N * sizeof(float)));
    int* d_cell_count = nullptr;
    int* d_cell_start = nullptr;
    int* d_cell_fill = nullptr;
    int* d_cell_node = nullptr;
    float2* d_cell_pts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cell_count, GRID_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_start, (GRID_CELLS + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_fill, GRID_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_node, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_pts, N * sizeof(float2)));
    CUDA_CHECK(cudaMemcpy(d_pts, pts.data(), pts.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs, obs.data(), n_ob * sizeof(Disk),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_parent, parent.data(), N * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cost_a, cost.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    // The grid build is part of the timed rewire.
    CUDA_CHECK(cudaMemsetAsync(d_cell_count, 0, GRID_CELLS * sizeof(int)));
    grid_count_kernel<<<blocks, threads>>>(d_pts, N, d_cell_count);
    grid_scan_kernel<<<1, 1>>>(d_cell_count, d_cell_start, d_cell_fill);
    grid_scatter_kernel<<<blocks, threads>>>(d_pts, N, d_cell_start, d_cell_fill,
                                             d_cell_node, d_cell_pts);
    float* in_ptr = d_cost_a;
    float* out_ptr = d_cost_b;
    for (int it = 0; it < iters; it++) {
        rewire_kernel<<<blocks, threads>>>(in_ptr, d_parent, out_ptr,
                                           d_obs, n_ob, d_cell_start, d_cell_node,
                                           d_cell_pts, N, REWIRE_RADIUS);
        std::swap(in_ptr, out_ptr);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(parent.data(), d_parent, N * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cost.data(), in_ptr, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_pts));
    CUDA_CHECK(cudaFree(d_obs));
    CUDA_CHECK(cudaFree(d_parent));
    CUDA_CHECK(cudaFree(d_cost_a));
    CUDA_CHECK(cudaFree(d_cost_b));
    CUDA_CHECK(cudaFree(d_cell_count));
    CUDA_CHECK(cudaFree(d_cell_start));
    CUDA_CHECK(cudaFree(d_cell_fill));
    CUDA_CHECK(cudaFree(d_cell_node));
    CUDA_CHECK(cudaFree(d_cell_pts));
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// -------------------------------------------------------------------------
// Rendering
// -------------------------------------------------------------------------
static cv::Mat render_tree(const std::vector<float>& pts,
                           const std::vector<int>& parent,
                           const std::vector<float>& cost,
                           const std::vector<Disk>& obs,
                           const char* title, double ms, int N_draw) {
    cv::Mat img(PANEL, PANEL, CV_8UC3, cv::Scalar(20, 20, 20));
    auto W = [&](float x) { return static_cast<int>(x / WORLD * PANEL); };
    auto Hh = [&](float y) { return static_cast<int>((1.0f - y / WORLD) * PANEL); };
    // obstacles
    for (const auto& o : obs) {
        cv::circle(img, cv::Point(W(o.cx), Hh(o.cy)),
                   static_cast<int>(o.r / WORLD * PANEL), cv::Scalar(60, 60, 200),
                   cv::FILLED);
    }
    // edges (limit drawing to first N_draw for perf)
    int N = static_cast<int>(pts.size() / 2);
    int N_d = std::min(N, N_draw);
    float max_cost = 0.0f;
    for (int i = 0; i < N_d; i++) max_cost = std::max(max_cost, cost[i]);
    if (max_cost < 1.0f) max_cost = 1.0f;
    for (int i = 1; i < N_d; i++) {
        int p = parent[i];
        if (p < 0) continue;
        float t = cost[i] / max_cost;
        cv::Scalar col(
            static_cast<int>(200.0f * (1.0f - t) + 30.0f),
            static_cast<int>(200.0f * t + 30.0f),
            static_cast<int>(60.0f));
        cv::line(img,
                 cv::Point(W(pts[i * 2]),     Hh(pts[i * 2 + 1])),
                 cv::Point(W(pts[p * 2]),     Hh(pts[p * 2 + 1])),
                 col, 1, cv::LINE_AA);
    }
    cv::rectangle(img, cv::Rect(0, 0, PANEL, 30), cv::Scalar(0, 0, 0), cv::FILLED);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s  N=%d  %.2f ms (rewire %d iters)",
                  title, N, ms, ITERS);
    cv::putText(img, buf, cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    return img;
}


// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------
int main() {
    std::printf("RRT* rewire: CPU N=%d (O(N^2) %d iters) vs GPU N=%d (parallel %d iters)\n",
                N_CPU, ITERS, N_GPU, ITERS);

    std::mt19937 rng(2026);
    auto obs = make_obstacles(rng);

    auto pts_cpu = sample_nodes(N_CPU, obs, rng);
    auto pts_gpu = sample_nodes(N_GPU, obs, rng);
    std::vector<int> parent_cpu, parent_gpu;
    std::vector<float> cost_cpu, cost_gpu;
    build_initial_tree(pts_cpu, obs, parent_cpu, cost_cpu);
    build_initial_tree(pts_gpu, obs, parent_gpu, cost_gpu);

    // Snapshot initial tree
    cv::Mat init_cpu = render_tree(pts_cpu, parent_cpu, cost_cpu, obs,
                                   "CPU initial", 0.0, N_CPU);
    cv::Mat init_gpu = render_tree(pts_gpu, parent_gpu, cost_gpu, obs,
                                   "GPU initial", 0.0, 6000);

    double cpu_ms = cpu_rewire_ms(pts_cpu, obs, parent_cpu, cost_cpu, ITERS);
    double gpu_ms = gpu_rewire_ms(pts_gpu, obs, parent_gpu, cost_gpu, ITERS);

    cv::Mat after_cpu = render_tree(pts_cpu, parent_cpu, cost_cpu, obs,
                                    "CPU after rewire", cpu_ms, N_CPU);
    cv::Mat after_gpu = render_tree(pts_gpu, parent_gpu, cost_gpu, obs,
                                    "GPU after rewire", gpu_ms, 6000);

    cv::Mat top(PANEL, PANEL * 2 + 4, CV_8UC3, cv::Scalar(30, 30, 30));
    init_cpu.copyTo(top(cv::Rect(0, 0, PANEL, PANEL)));
    init_gpu.copyTo(top(cv::Rect(PANEL + 4, 0, PANEL, PANEL)));
    cv::Mat bot(PANEL, PANEL * 2 + 4, CV_8UC3, cv::Scalar(30, 30, 30));
    after_cpu.copyTo(bot(cv::Rect(0, 0, PANEL, PANEL)));
    after_gpu.copyTo(bot(cv::Rect(PANEL + 4, 0, PANEL, PANEL)));
    cv::Mat combined(PANEL * 2 + 4, PANEL * 2 + 4, CV_8UC3, cv::Scalar(30, 30, 30));
    top.copyTo(combined(cv::Rect(0, 0, PANEL * 2 + 4, PANEL)));
    bot.copyTo(combined(cv::Rect(0, PANEL + 4, PANEL * 2 + 4, PANEL)));

    cv::VideoWriter video("gif/comparison_rrtstar_rewire.avi",
                          cudabot::avi_fourcc(), 4,
                          cv::Size(combined.cols, combined.rows));
    if (!video.isOpened()) {
        std::fprintf(stderr, "Failed to open gif/comparison_rrtstar_rewire.avi\n");
        return 1;
    }
    for (int f = 0; f < 12; f++) video.write(combined);
    video.release();
    cudabot::avi_to_gif_simple("gif/comparison_rrtstar_rewire.avi", "gif/comparison_rrtstar_rewire.gif", 4, 900);

    double cpu_per_us = cpu_ms * 1.0e3 / (static_cast<double>(N_CPU) * ITERS);
    double gpu_per_us = gpu_ms * 1.0e3 / (static_cast<double>(N_GPU) * ITERS);
    std::printf("Avg CPU %.2f ms / rewire (N=%d, %d iters)\n"
                "Avg GPU %.3f ms / rewire (N=%d, %d iters)\n"
                "Per-node: CPU %.4f us, GPU %.4f us "
                "(%.0fx faster per node)\n",
                cpu_ms, N_CPU, ITERS, gpu_ms, N_GPU, ITERS,
                cpu_per_us, gpu_per_us, cpu_per_us / gpu_per_us);
    std::printf("GIF saved to gif/comparison_rrtstar_rewire.gif\n");
    return 0;
}
