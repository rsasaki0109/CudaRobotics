/*************************************************************************
    Euclidean Signed Distance Field (ESDF) comparison: CPU brute-force at
    100x100 vs GPU Jump Flooding Algorithm at 800x800. Same 80m x 80m
    static scene, obstacle disks added every frame so the ESDF is
    recomputed each frame on both sides. The CPU panel renders a chunky
    distance heatmap; the GPU panel renders a smooth one.

    Algorithm:
      CPU:  for each grid cell, scan the obstacle-cell list, take the
            minimum Euclidean distance.
      GPU:  Jump Flooding. Each cell stores a seed (nearest obstacle
            coordinate). log2(W) passes with step sizes W/2, W/4, ..., 1
            propagate the nearest-seed information across the grid; a
            final pass converts seed coordinates into Euclidean distance.

    The headline metric is per-cell ESDF throughput on the same scene.
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <cfloat>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include "cuda_check.cuh"

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------
constexpr float WORLD_W = 80.0f;
constexpr float WORLD_H = 80.0f;

constexpr int   CPU_GRID = 100;
constexpr float CPU_RES  = WORLD_W / CPU_GRID;

constexpr int   GPU_GRID = 800;
constexpr float GPU_RES  = WORLD_W / GPU_GRID;

constexpr float MAX_DIST = 30.0f;  // for colormap normalisation

constexpr int PANEL_W = 600;
constexpr int PANEL_H = 600;

constexpr int SIM_FRAMES = 90;
constexpr int N_NEW_DISKS_PER_FRAME = 1;
constexpr float NEW_DISK_RADIUS_MIN = 0.6f;
constexpr float NEW_DISK_RADIUS_MAX = 1.4f;

// -------------------------------------------------------------------------
// Scene
// -------------------------------------------------------------------------
struct Disk { float cx, cy, r; };

static void build_walls(std::vector<unsigned char>& occ, int W, int H, float res) {
    occ.assign(W * H, 0u);
    auto set = [&](int gx, int gy) {
        if (gx >= 0 && gx < W && gy >= 0 && gy < H) occ[gy * W + gx] = 1u;
    };
    auto fill_rect_world = [&](float x0, float y0, float x1, float y1) {
        int gx0 = static_cast<int>(std::floor(x0 / res));
        int gy0 = static_cast<int>(std::floor(y0 / res));
        int gx1 = static_cast<int>(std::ceil(x1 / res));
        int gy1 = static_cast<int>(std::ceil(y1 / res));
        for (int gy = gy0; gy <= gy1; gy++)
            for (int gx = gx0; gx <= gx1; gx++) set(gx, gy);
    };
    // outer walls (0.4 m thick)
    fill_rect_world(0.0f, 0.0f, WORLD_W, 0.4f);
    fill_rect_world(0.0f, WORLD_H - 0.4f, WORLD_W, WORLD_H);
    fill_rect_world(0.0f, 0.0f, 0.4f, WORLD_H);
    fill_rect_world(WORLD_W - 0.4f, 0.0f, WORLD_W, WORLD_H);
    // a few interior corridor walls
    fill_rect_world(20.0f, 18.0f, 22.0f, 50.0f);
    fill_rect_world(58.0f, 30.0f, 60.0f, 62.0f);
    fill_rect_world(30.0f, 55.0f, 52.0f, 57.0f);
}

static void stamp_disk(std::vector<unsigned char>& occ, int W, int H,
                       float res, const Disk& d) {
    int gx0 = std::max(0, static_cast<int>(std::floor((d.cx - d.r) / res)));
    int gy0 = std::max(0, static_cast<int>(std::floor((d.cy - d.r) / res)));
    int gx1 = std::min(W - 1, static_cast<int>(std::ceil((d.cx + d.r) / res)));
    int gy1 = std::min(H - 1, static_cast<int>(std::ceil((d.cy + d.r) / res)));
    float r2 = d.r * d.r;
    for (int gy = gy0; gy <= gy1; gy++) {
        float wy = (gy + 0.5f) * res;
        for (int gx = gx0; gx <= gx1; gx++) {
            float wx = (gx + 0.5f) * res;
            float dx = wx - d.cx, dy = wy - d.cy;
            if (dx * dx + dy * dy <= r2) occ[gy * W + gx] = 1u;
        }
    }
}

// -------------------------------------------------------------------------
// CPU brute-force ESDF
// -------------------------------------------------------------------------
static double cpu_esdf_ms(const std::vector<unsigned char>& occ,
                          int W, int H, float res,
                          std::vector<float>& dist) {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> obstacle_idx;
    obstacle_idx.reserve(W * H / 8);
    for (int i = 0; i < W * H; i++) if (occ[i]) obstacle_idx.push_back(i);
    dist.assign(W * H, MAX_DIST);
    if (obstacle_idx.empty()) {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    for (int gy = 0; gy < H; gy++) {
        for (int gx = 0; gx < W; gx++) {
            float best2 = MAX_DIST * MAX_DIST / (res * res);
            for (int oi : obstacle_idx) {
                int ox = oi % W;
                int oy = oi / W;
                int dx = gx - ox, dy = gy - oy;
                float d2 = static_cast<float>(dx * dx + dy * dy);
                if (d2 < best2) best2 = d2;
            }
            dist[gy * W + gx] = std::sqrt(best2) * res;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// -------------------------------------------------------------------------
// GPU Jump Flooding
// -------------------------------------------------------------------------
__global__ void jfa_init_kernel(const unsigned char* __restrict__ occ,
                                int* __restrict__ seed, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    seed[idx] = occ[idx] ? idx : -1;
}

__global__ void jfa_step_kernel(const int* __restrict__ seed_in,
                                int* __restrict__ seed_out,
                                int W, int H, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    int best = seed_in[idx];
    float best_d2 = FLT_MAX;
    if (best >= 0) {
        int bx = best % W, by = best / W;
        int ex = x - bx, ey = y - by;
        best_d2 = static_cast<float>(ex * ex + ey * ey);
    }
    #pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        #pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx * k;
            int ny = y + dy * k;
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            int s = seed_in[ny * W + nx];
            if (s < 0) continue;
            int sx = s % W, sy = s / W;
            int ex = x - sx, ey = y - sy;
            float d2 = static_cast<float>(ex * ex + ey * ey);
            if (d2 < best_d2) { best = s; best_d2 = d2; }
        }
    }
    seed_out[idx] = best;
}

__global__ void jfa_to_dist_kernel(const int* __restrict__ seed,
                                   float* __restrict__ dist,
                                   int W, int H, float res) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    int s = seed[idx];
    if (s < 0) { dist[idx] = MAX_DIST; return; }
    int sx = s % W, sy = s / W;
    int dx = x - sx, dy = y - sy;
    dist[idx] = sqrtf(static_cast<float>(dx * dx + dy * dy)) * res;
}

struct GpuEsdf {
    unsigned char* d_occ = nullptr;
    int* d_seed_a = nullptr;
    int* d_seed_b = nullptr;
    float* d_dist = nullptr;
    int W = 0, H = 0;
    void alloc(int w, int h) {
        W = w; H = h;
        CUDA_CHECK(cudaMalloc(&d_occ, W * H * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_seed_a, W * H * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_seed_b, W * H * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_dist, W * H * sizeof(float)));
    }
    void free_all() {
        if (d_occ) cudaFree(d_occ);
        if (d_seed_a) cudaFree(d_seed_a);
        if (d_seed_b) cudaFree(d_seed_b);
        if (d_dist) cudaFree(d_dist);
    }
};

static double gpu_esdf_ms(GpuEsdf& g, const std::vector<unsigned char>& occ,
                          float res, std::vector<float>& dist_host) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    CUDA_CHECK(cudaMemcpy(g.d_occ, occ.data(), g.W * g.H * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((g.W + 15) / 16, (g.H + 15) / 16);
    cudaEventRecord(e0);
    jfa_init_kernel<<<grid, block>>>(g.d_occ, g.d_seed_a, g.W, g.H);
    int* in = g.d_seed_a;
    int* out = g.d_seed_b;
    int k = 1;
    while (k * 2 < std::max(g.W, g.H)) k *= 2;
    for (; k >= 1; k /= 2) {
        jfa_step_kernel<<<grid, block>>>(in, out, g.W, g.H, k);
        std::swap(in, out);
    }
    jfa_to_dist_kernel<<<grid, block>>>(in, g.d_dist, g.W, g.H, res);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    dist_host.resize(g.W * g.H);
    CUDA_CHECK(cudaMemcpy(dist_host.data(), g.d_dist,
                          g.W * g.H * sizeof(float), cudaMemcpyDeviceToHost));
    return static_cast<double>(ms);
}

// -------------------------------------------------------------------------
// Rendering
// -------------------------------------------------------------------------
static cv::Vec3b heatmap(float v) {
    // turbo-ish: blue (far) -> teal -> green -> yellow -> red (close to obstacle)
    float t = std::min(1.0f, std::max(0.0f, 1.0f - v / MAX_DIST));
    float r, g, b;
    if (t < 0.25f) { r = 0.0f; g = 4.0f * t; b = 1.0f; }
    else if (t < 0.5f) { r = 0.0f; g = 1.0f; b = 1.0f - 4.0f * (t - 0.25f); }
    else if (t < 0.75f) { r = 4.0f * (t - 0.5f); g = 1.0f; b = 0.0f; }
    else { r = 1.0f; g = 1.0f - 4.0f * (t - 0.75f); b = 0.0f; }
    return cv::Vec3b(static_cast<unsigned char>(b * 255),
                     static_cast<unsigned char>(g * 255),
                     static_cast<unsigned char>(r * 255));
}

static void draw_esdf(cv::Mat& panel,
                      const std::vector<unsigned char>& occ,
                      const std::vector<float>& dist,
                      int W, int H) {
    panel.create(PANEL_H, PANEL_W, CV_8UC3);
    for (int py = 0; py < PANEL_H; py++) {
        float wy = (PANEL_H - 1 - py) * (WORLD_H / PANEL_H);
        int gy = std::min(H - 1, std::max(0, static_cast<int>(wy * H / WORLD_H)));
        for (int px = 0; px < PANEL_W; px++) {
            float wx = px * (WORLD_W / PANEL_W);
            int gx = std::min(W - 1, std::max(0, static_cast<int>(wx * W / WORLD_W)));
            int idx = gy * W + gx;
            if (occ[idx]) {
                panel.at<cv::Vec3b>(py, px) = cv::Vec3b(30, 30, 30);
            } else {
                panel.at<cv::Vec3b>(py, px) = heatmap(dist[idx]);
            }
        }
    }
}

static void draw_label(cv::Mat& panel, const std::string& text, int y_offset) {
    cv::Point pt(12, y_offset);
    cv::putText(panel, text, pt, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
    cv::putText(panel, text, pt, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

// -------------------------------------------------------------------------
// Validation: ensure GPU JFA matches CPU brute force at small grid
// -------------------------------------------------------------------------
static void validate_jfa_vs_brute(unsigned seed) {
    constexpr int VW = 64;
    std::vector<unsigned char> occ(VW * VW, 0);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> ux(2, VW - 3), uy(2, VW - 3);
    for (int i = 0; i < 30; i++) occ[uy(rng) * VW + ux(rng)] = 1;
    float res = 0.5f;

    std::vector<float> dist_cpu;
    cpu_esdf_ms(occ, VW, VW, res, dist_cpu);

    GpuEsdf g; g.alloc(VW, VW);
    std::vector<float> dist_gpu;
    gpu_esdf_ms(g, occ, res, dist_gpu);
    g.free_all();

    float max_err = 0.0f;
    double sum_err = 0.0;
    for (int i = 0; i < VW * VW; i++) {
        float e = std::fabs(dist_cpu[i] - dist_gpu[i]);
        if (e > max_err) max_err = e;
        sum_err += e;
    }
    std::printf("JFA validation against CPU brute force on %dx%d grid: "
                "max err %.4f m, mean err %.5f m\n",
                VW, VW, max_err, sum_err / (VW * VW));
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    std::printf("ESDF comparison: CPU brute force %dx%d vs GPU JFA %dx%d on "
                "%.0f m x %.0f m scene\n",
                CPU_GRID, CPU_GRID, GPU_GRID, GPU_GRID, WORLD_W, WORLD_H);
    validate_jfa_vs_brute(1234u);

    std::vector<unsigned char> cpu_occ, gpu_occ;
    build_walls(cpu_occ, CPU_GRID, CPU_GRID, CPU_RES);
    build_walls(gpu_occ, GPU_GRID, GPU_GRID, GPU_RES);

    GpuEsdf g; g.alloc(GPU_GRID, GPU_GRID);
    std::vector<float> dist_cpu, dist_gpu;

    cv::VideoWriter video("gif/comparison_esdf.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
                          cv::Size(PANEL_W * 2, PANEL_H));

    std::mt19937 rng(7u);
    std::uniform_real_distribution<float> ux(4.0f, WORLD_W - 4.0f);
    std::uniform_real_distribution<float> uy(4.0f, WORLD_H - 4.0f);
    std::uniform_real_distribution<float> ur(NEW_DISK_RADIUS_MIN, NEW_DISK_RADIUS_MAX);

    double cpu_ms_sum = 0.0, gpu_ms_sum = 0.0;
    int timed_frames = 0;

    for (int f = 0; f < SIM_FRAMES; f++) {
        for (int k = 0; k < N_NEW_DISKS_PER_FRAME; k++) {
            Disk d{ux(rng), uy(rng), ur(rng)};
            stamp_disk(cpu_occ, CPU_GRID, CPU_GRID, CPU_RES, d);
            stamp_disk(gpu_occ, GPU_GRID, GPU_GRID, GPU_RES, d);
        }

        double cpu_ms = cpu_esdf_ms(cpu_occ, CPU_GRID, CPU_GRID, CPU_RES, dist_cpu);
        double gpu_ms = gpu_esdf_ms(g, gpu_occ, GPU_RES, dist_gpu);

        if (f >= 5) {
            cpu_ms_sum += cpu_ms;
            gpu_ms_sum += gpu_ms;
            timed_frames++;
        }

        cv::Mat left, right;
        draw_esdf(left, cpu_occ, dist_cpu, CPU_GRID, CPU_GRID);
        draw_esdf(right, gpu_occ, dist_gpu, GPU_GRID, GPU_GRID);

        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "CPU brute force  %dx%d  %.1f ms", CPU_GRID, CPU_GRID, cpu_ms);
        draw_label(left, buf, 28);
        std::snprintf(buf, sizeof(buf),
                      "GPU JFA  %dx%d  %.3f ms", GPU_GRID, GPU_GRID, gpu_ms);
        draw_label(right, buf, 28);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        video.write(combined);
    }
    video.release();
    g.free_all();

    if (timed_frames > 0) {
        double cpu_ms = cpu_ms_sum / timed_frames;
        double gpu_ms = gpu_ms_sum / timed_frames;
        double cpu_per_cell_us = cpu_ms * 1.0e3 / (CPU_GRID * CPU_GRID);
        double gpu_per_cell_us = gpu_ms * 1.0e3 / (GPU_GRID * GPU_GRID);
        std::printf("Avg CPU %.2f ms / ESDF (%d cells)\n"
                    "Avg GPU %.3f ms / ESDF (%d cells)\n"
                    "Per-cell throughput: GPU %.4f us/cell vs CPU %.3f us/cell "
                    "(%.0fx faster per cell)\n",
                    cpu_ms, CPU_GRID * CPU_GRID,
                    gpu_ms, GPU_GRID * GPU_GRID,
                    gpu_per_cell_us, cpu_per_cell_us,
                    cpu_per_cell_us / gpu_per_cell_us);
    }

    std::system("ffmpeg -y -i gif/comparison_esdf.avi "
                "-vf 'fps=15,scale=900:-1:flags=lanczos' -loop 0 "
                "gif/comparison_esdf.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_esdf.gif" << std::endl;
    return 0;
}
