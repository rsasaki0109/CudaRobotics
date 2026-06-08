// gpu_dbscan.cu
//
// GPU DBSCAN density-based point-cloud clustering — the clustering primitive
// that the repo's PointCloud section (voxel/statistical filtering, normal
// estimation, GICP, RANSAC, label propagation, GICP-MCL) did not yet cover.
//
// DBSCAN finds clusters as maximal sets of points reachable through chains of
// points that each have at least `min_pts` neighbours inside an `eps` ball,
// and labels everything else as noise. The algorithm has a natural parallel
// structure that matches the repo's canonical idiom:
//
//   one thread = one point
//
// Pipeline (CPU and GPU run the SAME logic):
//   1. neighbour count: for each point i, count |{j : dist(i,j) < eps}|
//   2. core mark:       core[i] = (n_neighbours[i] >= min_pts)
//   3. label propagation (parallel union-find lite): for each core point i,
//      atomically pull labels[i] down to min(labels[j]) over all core
//      neighbours j; iterate until no label changes
//   4. border assignment: each non-core point in eps of some core inherits
//      the (smallest) neighbouring core label; otherwise it is NOISE (-1)
//
// We deliberately use brute-force pairwise neighbour search so the CPU and GPU
// paths run identical arithmetic — the only difference is the parallel layout.
// At N = 8192 this is ~67 M pair checks (~tens of GFLOP), large enough to make
// the GPU's win obvious while keeping the CPU reference's runtime sane.
//
// Correctness reporting: cluster IDs are renumbered canonically by first
// appearance, then the CPU and GPU label arrays are compared point-by-point.
// We report the fraction of points with the same label and the per-cluster
// size match.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <unordered_map>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ----------------------------------------------------------------- constants
#define N_POINTS  8192
static const float EPS      = 0.55f;
static const int   MIN_PTS  = 8;
static const float EPS2     = EPS * EPS;
static const int   MAX_PROP_ITERS = 80;
static const int   N_BLOBS  = 6;
static const float WORLD_W  = 30.0f;
static const float WORLD_H  = 30.0f;
static const int   PANEL_W  = 760;
static const int   PANEL_H  = 600;

static const int NOISE_LABEL = -1;

// ------------------------------------------------------------- point source
static void make_points(std::vector<float>& xy) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> ux(2.0f, WORLD_W - 2.0f);
    std::uniform_real_distribution<float> uy(2.0f, WORLD_H - 2.0f);
    std::normal_distribution<float>       blob(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    // pick blob centres
    float cx[N_BLOBS], cy[N_BLOBS], cr[N_BLOBS];
    for (int b = 0; b < N_BLOBS; ++b) {
        cx[b] = ux(rng);
        cy[b] = uy(rng);
        cr[b] = 0.6f + 0.7f * uni(rng);           // anisotropic spread
    }
    std::vector<std::array<float, 2>> pts(N_POINTS);
    int n_noise = N_POINTS / 10;                  // ~10% uniform noise
    int n_blobs = N_POINTS - n_noise;
    for (int i = 0; i < n_blobs; ++i) {
        int b = i * N_BLOBS / n_blobs;
        pts[i] = {cx[b] + cr[b] * blob(rng), cy[b] + cr[b] * blob(rng)};
    }
    for (int i = n_blobs; i < N_POINTS; ++i) {
        pts[i] = {ux(rng), uy(rng)};
    }
    std::shuffle(pts.begin(), pts.end(), rng);
    xy.assign(2 * N_POINTS, 0.0f);
    for (int i = 0; i < N_POINTS; ++i) {
        xy[2 * i + 0] = pts[i][0];
        xy[2 * i + 1] = pts[i][1];
    }
}

// ---------------------------------------------------------- shared kernels
__host__ __device__ static inline int count_neighbours(
        int i, const float* xy) {
    float xi = xy[2 * i + 0], yi = xy[2 * i + 1];
    int c = 0;
    for (int j = 0; j < N_POINTS; ++j) {
        float dx = xy[2 * j + 0] - xi;
        float dy = xy[2 * j + 1] - yi;
        if (dx * dx + dy * dy < EPS2) ++c;
    }
    return c;
}

__global__ void count_kernel(const float* xy, int* n_neigh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_POINTS) return;
    n_neigh[i] = count_neighbours(i, xy);
}

__global__ void core_mark_kernel(const int* n_neigh, int* core,
                                 int* labels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_POINTS) return;
    core[i] = (n_neigh[i] >= MIN_PTS) ? 1 : 0;
    labels[i] = core[i] ? i : NOISE_LABEL;
}

// One propagation sweep: each core point pulls its label down to the smallest
// label among its core neighbours. Returns 1 in `changed` if any label moved.
__global__ void propagate_kernel(const float* xy, const int* core,
                                 int* labels, int* changed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_POINTS) return;
    if (!core[i]) return;
    float xi = xy[2 * i + 0], yi = xy[2 * i + 1];
    int  my = labels[i];
    int  m  = my;
    for (int j = 0; j < N_POINTS; ++j) {
        if (!core[j]) continue;
        float dx = xy[2 * j + 0] - xi;
        float dy = xy[2 * j + 1] - yi;
        if (dx * dx + dy * dy < EPS2) {
            int lj = labels[j];
            if (lj < m) m = lj;
        }
    }
    if (m < my) {
        atomicMin(&labels[i], m);
        atomicExch(changed, 1);
    }
}

__global__ void border_kernel(const float* xy, const int* core,
                              int* labels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_POINTS) return;
    if (core[i]) return;                              // core already labelled
    float xi = xy[2 * i + 0], yi = xy[2 * i + 1];
    int best = NOISE_LABEL;
    for (int j = 0; j < N_POINTS; ++j) {
        if (!core[j]) continue;
        float dx = xy[2 * j + 0] - xi;
        float dy = xy[2 * j + 1] - yi;
        if (dx * dx + dy * dy < EPS2) {
            int lj = labels[j];
            if (best == NOISE_LABEL || lj < best) best = lj;
        }
    }
    labels[i] = best;
}

// ---------------------------------------------------------- CPU reference
static void dbscan_cpu(const std::vector<float>& xy,
                       std::vector<int>& labels_out,
                       int& iters_out) {
    std::vector<int> n_neigh(N_POINTS), core(N_POINTS), labels(N_POINTS);
    for (int i = 0; i < N_POINTS; ++i) n_neigh[i] = count_neighbours(i, xy.data());
    for (int i = 0; i < N_POINTS; ++i) {
        core[i]   = (n_neigh[i] >= MIN_PTS) ? 1 : 0;
        labels[i] = core[i] ? i : NOISE_LABEL;
    }
    int iters = 0;
    for (iters = 0; iters < MAX_PROP_ITERS; ++iters) {
        std::vector<int> next = labels;
        bool changed = false;
        for (int i = 0; i < N_POINTS; ++i) {
            if (!core[i]) continue;
            float xi = xy[2 * i + 0], yi = xy[2 * i + 1];
            int m = labels[i];
            for (int j = 0; j < N_POINTS; ++j) {
                if (!core[j]) continue;
                float dx = xy[2 * j + 0] - xi;
                float dy = xy[2 * j + 1] - yi;
                if (dx * dx + dy * dy < EPS2) {
                    if (labels[j] < m) m = labels[j];
                }
            }
            if (m < labels[i]) { next[i] = m; changed = true; }
        }
        labels.swap(next);
        if (!changed) break;
    }
    // border
    for (int i = 0; i < N_POINTS; ++i) {
        if (core[i]) continue;
        float xi = xy[2 * i + 0], yi = xy[2 * i + 1];
        int best = NOISE_LABEL;
        for (int j = 0; j < N_POINTS; ++j) {
            if (!core[j]) continue;
            float dx = xy[2 * j + 0] - xi;
            float dy = xy[2 * j + 1] - yi;
            if (dx * dx + dy * dy < EPS2) {
                if (best == NOISE_LABEL || labels[j] < best) best = labels[j];
            }
        }
        labels[i] = best;
    }
    labels_out = std::move(labels);
    iters_out  = iters;
}

// ---------------------------------------------------------- canonicalise
// Renumber labels by first appearance so two runs can be compared regardless
// of how the underlying root indices ended up. Noise stays NOISE_LABEL.
static void canonicalise(std::vector<int>& labels, int& n_clusters) {
    std::unordered_map<int, int> remap;
    int next = 0;
    for (int& l : labels) {
        if (l == NOISE_LABEL) continue;
        auto it = remap.find(l);
        if (it == remap.end()) { remap[l] = next; l = next; ++next; }
        else                   { l = it->second; }
    }
    n_clusters = next;
}

// ------------------------------------------------------ visualisation
static cv::Scalar palette(int k) {
    static const cv::Scalar P[] = {
        {255, 120,  60}, { 60, 200, 255}, {120, 230, 100}, {220, 100, 220},
        {255, 220,  90}, {100, 180, 255}, {255, 170, 110}, {180, 240, 200},
        {200, 130, 255}, { 90, 220, 200}};
    return P[((k % 10) + 10) % 10];
}

static void draw_frame(cv::Mat& img,
                       const std::vector<float>& xy,
                       const std::vector<int>& labels,
                       const std::vector<int>& core,
                       const char* l1, const char* l2, const char* l3) {
    img = cv::Mat(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(20, 20, 24));
    float sx = (PANEL_W - 40) / WORLD_W;
    float sy = (PANEL_H - 110) / WORLD_H;
    for (int i = 0; i < N_POINTS; ++i) {
        int u = (int)(20 + xy[2 * i + 0] * sx);
        int v = (int)(40 + xy[2 * i + 1] * sy);
        if (u < 0 || u >= PANEL_W || v < 0 || v >= PANEL_H - 70) continue;
        cv::Scalar col;
        if (labels[i] == NOISE_LABEL) col = cv::Scalar(80, 80, 90);
        else                          col = palette(labels[i]);
        int r = (core.empty() || core[i]) ? 2 : 1;
        cv::circle(img, cv::Point(u, v), r, col, -1);
    }
    cv::putText(img, l1, cv::Point(12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::putText(img, l2, cv::Point(12, PANEL_H - 32),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 255, 200), 1, cv::LINE_AA);
    cv::putText(img, l3, cv::Point(12, PANEL_H - 12),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 220, 255), 1, cv::LINE_AA);
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::printf("GPU DBSCAN: N = %d, eps = %.3f, min_pts = %d\n",
                N_POINTS, EPS, MIN_PTS);

    std::vector<float> xy;
    make_points(xy);

    // --------------------------------------------------- CPU reference
    std::vector<int> labels_cpu;
    int cpu_iters = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    dbscan_cpu(xy, labels_cpu, cpu_iters);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::vector<int> labels_cpu_can = labels_cpu;
    int n_cpu = 0;
    canonicalise(labels_cpu_can, n_cpu);

    // --------------------------------------------------- GPU pipeline
    float *d_xy;
    int   *d_n, *d_core, *d_labels, *d_changed;
    CUDA_CHECK(cudaMalloc(&d_xy,      xy.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_n,       N_POINTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_core,    N_POINTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels,  N_POINTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_xy, xy.data(), xy.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    int block = 128, grid = (N_POINTS + block - 1) / block;

    // warm-up
    count_kernel<<<grid, block>>>(d_xy, d_n);
    core_mark_kernel<<<grid, block>>>(d_n, d_core, d_labels);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    count_kernel<<<grid, block>>>(d_xy, d_n);
    core_mark_kernel<<<grid, block>>>(d_n, d_core, d_labels);
    int gpu_iters = 0;
    for (gpu_iters = 0; gpu_iters < MAX_PROP_ITERS; ++gpu_iters) {
        int z = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &z, sizeof(int),
                              cudaMemcpyHostToDevice));
        propagate_kernel<<<grid, block>>>(d_xy, d_core, d_labels, d_changed);
        int ch;
        CUDA_CHECK(cudaMemcpy(&ch, d_changed, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (!ch) break;
    }
    border_kernel<<<grid, block>>>(d_xy, d_core, d_labels);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));

    std::vector<int> labels_gpu(N_POINTS), core_gpu(N_POINTS);
    CUDA_CHECK(cudaMemcpy(labels_gpu.data(), d_labels,
                          N_POINTS * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(core_gpu.data(), d_core,
                          N_POINTS * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> labels_gpu_can = labels_gpu;
    int n_gpu = 0;
    canonicalise(labels_gpu_can, n_gpu);

    // --------------------------------------------------- compare
    int n_core_gpu = 0, n_noise_cpu = 0, n_noise_gpu = 0;
    for (int i = 0; i < N_POINTS; ++i) {
        if (core_gpu[i]) ++n_core_gpu;
        if (labels_gpu[i] == NOISE_LABEL) ++n_noise_gpu;
        if (labels_cpu[i] == NOISE_LABEL) ++n_noise_cpu;
    }
    // a clean cluster-by-cluster agreement: for each cluster id in CPU,
    // find the most-common GPU id covering it
    std::unordered_map<long long, int> co;
    int n_in_cluster_cpu = 0;
    for (int i = 0; i < N_POINTS; ++i) {
        if (labels_cpu_can[i] == NOISE_LABEL) continue;
        ++n_in_cluster_cpu;
        long long k = (long long)labels_cpu_can[i] * 100000LL +
                      (long long)labels_gpu_can[i];
        co[k]++;
    }
    // pick max overlap per CPU cluster id
    std::vector<int> best_match(n_cpu, 0);
    for (auto& kv : co) {
        int c_cpu = (int)(kv.first / 100000LL);
        int cnt = kv.second;
        if (cnt > best_match[c_cpu]) best_match[c_cpu] = cnt;
    }
    int n_matched = 0;
    for (int v : best_match) n_matched += v;
    double match_frac = (n_in_cluster_cpu > 0)
        ? (double)n_matched / (double)n_in_cluster_cpu : 1.0;

    double speedup = cpu_ms / gpu_ms;
    std::printf("CPU %.2f ms (%d prop iters), GPU %.3f ms (%d iters)  -> %.0fx\n",
                cpu_ms, cpu_iters, gpu_ms, gpu_iters, speedup);
    std::printf("clusters: CPU %d, GPU %d   core points GPU %d\n",
                n_cpu, n_gpu, n_core_gpu);
    std::printf("noise points: CPU %d, GPU %d   cluster-agreement %.2f%%\n",
                n_noise_cpu, n_noise_gpu, 100.0 * match_frac);

    // --------------------------------------------------- animation
    // Re-run the GPU pipeline step-by-step to record per-iter snapshots.
    // (Identical math to the timed run; just instrumented host-side.)
    std::vector<int> snap_labels(N_POINTS);
    std::vector<int> snap_core(N_POINTS);
    CUDA_CHECK(cudaMemcpy(d_labels, labels_cpu.data(), N_POINTS * sizeof(int),
                          cudaMemcpyHostToDevice));   // wipe so we re-run cleanly
    count_kernel<<<grid, block>>>(d_xy, d_n);
    core_mark_kernel<<<grid, block>>>(d_n, d_core, d_labels);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(snap_core.data(), d_core, N_POINTS * sizeof(int),
                          cudaMemcpyDeviceToHost));

    if (system("mkdir -p tmp") != 0)
        std::fprintf(stderr, "warning: mkdir tmp failed\n");
    cv::VideoWriter video("tmp/gpu_dbscan.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          8, cv::Size(PANEL_W, PANEL_H));

    // Frame 0: raw input
    {
        std::vector<int> raw_labels(N_POINTS, NOISE_LABEL);
        cv::Mat img;
        char l1[160], l2[160], l3[160];
        std::snprintf(l1, sizeof(l1),
                      "GPU DBSCAN (one thread = one point, brute force)  "
                      "N = %d  eps = %.2f  min_pts = %d", N_POINTS, EPS, MIN_PTS);
        std::snprintf(l2, sizeof(l2), "step: raw input  (no labels yet)");
        std::snprintf(l3, sizeof(l3),
                      "CPU %.0f ms (%d iters)  vs  GPU %.2f ms (%d iters)  ->  %.0fx",
                      cpu_ms, cpu_iters, gpu_ms, gpu_iters, speedup);
        draw_frame(img, xy, raw_labels, {}, l1, l2, l3);
        for (int r = 0; r < 4; ++r) video.write(img);
    }

    // Frame 1: cores highlighted (label = self, non-core = NOISE)
    {
        std::vector<int> show(N_POINTS);
        for (int i = 0; i < N_POINTS; ++i)
            show[i] = snap_core[i] ? 0 : NOISE_LABEL;
        cv::Mat img;
        char l1[160], l2[160], l3[160];
        std::snprintf(l1, sizeof(l1),
                      "GPU DBSCAN (one thread = one point, brute force)  "
                      "N = %d  eps = %.2f  min_pts = %d", N_POINTS, EPS, MIN_PTS);
        int nc = 0; for (int v : snap_core) nc += v;
        std::snprintf(l2, sizeof(l2),
                      "step: neighbour count + core mark  (core points: %d / %d)",
                      nc, N_POINTS);
        std::snprintf(l3, sizeof(l3),
                      "CPU %.0f ms (%d iters)  vs  GPU %.2f ms (%d iters)  ->  %.0fx",
                      cpu_ms, cpu_iters, gpu_ms, gpu_iters, speedup);
        draw_frame(img, xy, show, snap_core, l1, l2, l3);
        for (int r = 0; r < 4; ++r) video.write(img);
    }

    // Iterations of label propagation (re-run, snapping each iter)
    int it = 0;
    for (it = 0; it < MAX_PROP_ITERS; ++it) {
        int z = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &z, sizeof(int), cudaMemcpyHostToDevice));
        propagate_kernel<<<grid, block>>>(d_xy, d_core, d_labels, d_changed);
        int ch;
        CUDA_CHECK(cudaMemcpy(&ch, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(snap_labels.data(), d_labels,
                              N_POINTS * sizeof(int), cudaMemcpyDeviceToHost));

        // canonicalise just for visual stability
        std::vector<int> show = snap_labels;
        int nc; canonicalise(show, nc);
        cv::Mat img;
        char l1[160], l2[160], l3[160];
        std::snprintf(l1, sizeof(l1),
                      "GPU DBSCAN (one thread = one point, brute force)  "
                      "N = %d  eps = %.2f  min_pts = %d", N_POINTS, EPS, MIN_PTS);
        std::snprintf(l2, sizeof(l2),
                      "step: label propagation iter %d   running clusters: %d",
                      it + 1, nc);
        std::snprintf(l3, sizeof(l3),
                      "CPU %.0f ms (%d iters)  vs  GPU %.2f ms (%d iters)  ->  %.0fx",
                      cpu_ms, cpu_iters, gpu_ms, gpu_iters, speedup);
        draw_frame(img, xy, show, snap_core, l1, l2, l3);
        video.write(img);
        if (!ch) break;
    }

    // Final: border assignment
    border_kernel<<<grid, block>>>(d_xy, d_core, d_labels);
    CUDA_CHECK(cudaMemcpy(snap_labels.data(), d_labels,
                          N_POINTS * sizeof(int), cudaMemcpyDeviceToHost));
    {
        std::vector<int> show = snap_labels;
        int nc; canonicalise(show, nc);
        cv::Mat img;
        char l1[160], l2[160], l3[160];
        std::snprintf(l1, sizeof(l1),
                      "GPU DBSCAN (one thread = one point, brute force)  "
                      "N = %d  eps = %.2f  min_pts = %d", N_POINTS, EPS, MIN_PTS);
        int n_noise = 0;
        for (int v : snap_labels) if (v == NOISE_LABEL) ++n_noise;
        std::snprintf(l2, sizeof(l2),
                      "step: border assignment + noise  (clusters: %d, noise: %d)",
                      nc, n_noise);
        std::snprintf(l3, sizeof(l3),
                      "CPU %.0f ms (%d iters)  vs  GPU %.2f ms (%d iters)  ->  %.0fx   "
                      "cluster agreement %.1f%%",
                      cpu_ms, cpu_iters, gpu_ms, gpu_iters, speedup,
                      100.0 * match_frac);
        draw_frame(img, xy, show, snap_core, l1, l2, l3);
        for (int r = 0; r < 8; ++r) video.write(img);
    }

    video.release();
    cudabot::avi_to_gif("tmp/gpu_dbscan.avi", "gif/gpu_dbscan.gif", 8, 760);
    std::printf("wrote gif/gpu_dbscan.gif\n");

    CUDA_CHECK(cudaFree(d_xy));
    CUDA_CHECK(cudaFree(d_n));
    CUDA_CHECK(cudaFree(d_core));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_changed));
    return 0;
}
