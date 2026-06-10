/*************************************************************************
    3D Euclidean Signed Distance Field comparison: CPU brute-force at
    32x32x16 vs GPU 3D Jump Flooding Algorithm at 128x128x64. Same
    16m x 16m x 8m static scene with axis-aligned boxes; each frame
    adds one disk-stack obstacle so the ESDF is rebuilt every frame on
    both sides.

    Algorithm:
      CPU:  for each voxel, scan the obstacle-voxel list, take minimum
            Euclidean distance.
      GPU:  3D Jump Flooding. Each voxel stores a seed (linear index of
            nearest obstacle voxel). Steps with k = max(W,H,D)/2, ...,
            1 propagate nearest-seed info across the 3D grid via 26
            neighbours. A final pass converts seeds into distances.

    Visualization: two horizontal slices (z = 0.25*D, z = 0.5*D) shown
    side-by-side with a colormap. CPU panel renders the chunky 32x32
    slice; GPU panel renders the smooth 128x128 slice.

    Headline metric: per-voxel ESDF throughput on the same scene.
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

#include <cuda_runtime.h>
#include "cuda_check.cuh"

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------
constexpr float WORLD_W = 16.0f;
constexpr float WORLD_H = 16.0f;
constexpr float WORLD_D = 8.0f;

constexpr int CPU_W = 32;
constexpr int CPU_H = 32;
constexpr int CPU_D = 16;
constexpr float CPU_RES = WORLD_W / CPU_W;

constexpr int GPU_W = 128;
constexpr int GPU_H = 128;
constexpr int GPU_D = 64;
constexpr float GPU_RES = WORLD_W / GPU_W;

constexpr float MAX_DIST = 6.0f;

constexpr int PANEL_W = 360;
constexpr int PANEL_H = 360;

constexpr int SIM_FRAMES = 60;

struct Box { float x0, y0, z0, x1, y1, z1; };

// -------------------------------------------------------------------------
// Scene
// -------------------------------------------------------------------------
static void stamp_box(std::vector<unsigned char>& occ, int W, int H, int D,
                      float res, const Box& b) {
    int x0 = std::max(0, static_cast<int>(std::floor(b.x0 / res)));
    int y0 = std::max(0, static_cast<int>(std::floor(b.y0 / res)));
    int z0 = std::max(0, static_cast<int>(std::floor(b.z0 / res)));
    int x1 = std::min(W - 1, static_cast<int>(std::ceil(b.x1 / res)));
    int y1 = std::min(H - 1, static_cast<int>(std::ceil(b.y1 / res)));
    int z1 = std::min(D - 1, static_cast<int>(std::ceil(b.z1 / res)));
    for (int z = z0; z <= z1; z++)
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++)
                occ[(z * H + y) * W + x] = 1u;
}

static void build_scene(std::vector<unsigned char>& occ, int W, int H, int D,
                        float res) {
    occ.assign(static_cast<size_t>(W) * H * D, 0u);
    // Ground (z = 0 .. 0.4 m)
    stamp_box(occ, W, H, D, res, {0.0f, 0.0f, 0.0f, WORLD_W, WORLD_H, 0.4f});
    // Four vertical pillars
    stamp_box(occ, W, H, D, res, { 3.0f,  3.0f, 0.4f,  4.0f,  4.0f, 7.0f});
    stamp_box(occ, W, H, D, res, {12.0f,  3.0f, 0.4f, 13.0f,  4.0f, 7.0f});
    stamp_box(occ, W, H, D, res, { 3.0f, 12.0f, 0.4f,  4.0f, 13.0f, 7.0f});
    stamp_box(occ, W, H, D, res, {12.0f, 12.0f, 0.4f, 13.0f, 13.0f, 7.0f});
    // Suspended slab
    stamp_box(occ, W, H, D, res, {6.0f, 6.0f, 5.5f, 10.0f, 10.0f, 6.0f});
}

static void add_random_disk_stack(std::vector<unsigned char>& occ,
                                  int W, int H, int D, float res,
                                  std::mt19937& rng) {
    std::uniform_real_distribution<float> ux(2.0f, WORLD_W - 2.0f);
    std::uniform_real_distribution<float> uy(2.0f, WORLD_H - 2.0f);
    std::uniform_real_distribution<float> ur(0.4f, 0.9f);
    float cx = ux(rng), cy = uy(rng);
    float r = ur(rng);
    Box b = {cx - r, cy - r, 0.4f, cx + r, cy + r, 0.4f + 2.5f * r};
    stamp_box(occ, W, H, D, res, b);
}

// -------------------------------------------------------------------------
// CPU brute-force 3D ESDF
// -------------------------------------------------------------------------
static double cpu_esdf3d_ms(const std::vector<unsigned char>& occ,
                            int W, int H, int D, float res,
                            std::vector<float>& dist) {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> obs;
    obs.reserve(W * H * D / 8);
    for (int i = 0; i < W * H * D; i++) if (occ[i]) obs.push_back(i);
    dist.assign(static_cast<size_t>(W) * H * D, MAX_DIST);
    if (obs.empty()) {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    for (int z = 0; z < D; z++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float best2 = MAX_DIST * MAX_DIST / (res * res);
                for (int oi : obs) {
                    int ox = oi % W;
                    int oy = (oi / W) % H;
                    int oz = oi / (W * H);
                    int dx = x - ox, dy = y - oy, dz = z - oz;
                    float d2 = static_cast<float>(dx * dx + dy * dy + dz * dz);
                    if (d2 < best2) best2 = d2;
                }
                dist[(z * H + y) * W + x] = std::sqrt(best2) * res;
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// -------------------------------------------------------------------------
// GPU 3D Jump Flooding
// -------------------------------------------------------------------------
__device__ __forceinline__ void unflatten(int idx, int W, int H,
                                          int& x, int& y, int& z) {
    x = idx % W;
    y = (idx / W) % H;
    z = idx / (W * H);
}

__global__ void jfa3d_init_kernel(const unsigned char* __restrict__ occ,
                                  int* __restrict__ seed,
                                  int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;
    int idx = (z * H + y) * W + x;
    seed[idx] = occ[idx] ? idx : -1;
}

__global__ void jfa3d_step_kernel(const int* __restrict__ seed_in,
                                  int* __restrict__ seed_out,
                                  int W, int H, int D, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;
    int idx = (z * H + y) * W + x;
    int best = seed_in[idx];
    float best_d2 = FLT_MAX;
    if (best >= 0) {
        int bx, by, bz; unflatten(best, W, H, bx, by, bz);
        int ex = x - bx, ey = y - by, ez = z - bz;
        best_d2 = static_cast<float>(ex * ex + ey * ey + ez * ez);
    }
    #pragma unroll
    for (int dz = -1; dz <= 1; dz++) {
        #pragma unroll
        for (int dy = -1; dy <= 1; dy++) {
            #pragma unroll
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = x + dx * k;
                int ny = y + dy * k;
                int nz = z + dz * k;
                if (nx < 0 || nx >= W || ny < 0 || ny >= H || nz < 0 || nz >= D) continue;
                int s = seed_in[(nz * H + ny) * W + nx];
                if (s < 0) continue;
                int sx, sy, sz; unflatten(s, W, H, sx, sy, sz);
                int ex = x - sx, ey = y - sy, ez = z - sz;
                float d2 = static_cast<float>(ex * ex + ey * ey + ez * ez);
                if (d2 < best_d2) { best = s; best_d2 = d2; }
            }
        }
    }
    seed_out[idx] = best;
}

__global__ void jfa3d_to_dist_kernel(const int* __restrict__ seed,
                                     float* __restrict__ dist,
                                     int W, int H, int D, float res) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) return;
    int idx = (z * H + y) * W + x;
    int s = seed[idx];
    if (s < 0) { dist[idx] = MAX_DIST; return; }
    int sx, sy, sz; unflatten(s, W, H, sx, sy, sz);
    int dx = x - sx, dy = y - sy, dz = z - sz;
    dist[idx] = sqrtf(static_cast<float>(dx * dx + dy * dy + dz * dz)) * res;
}

static double gpu_esdf3d_ms(const std::vector<unsigned char>& occ,
                            int W, int H, int D, float res,
                            std::vector<float>& dist) {
    size_t cells = static_cast<size_t>(W) * H * D;
    unsigned char* d_occ = nullptr;
    int* d_seed_a = nullptr;
    int* d_seed_b = nullptr;
    float* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_occ,    cells * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_seed_a, cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seed_b, cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist,   cells * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_occ, occ.data(), cells, cudaMemcpyHostToDevice));

    dim3 blk(8, 8, 4);
    dim3 grd((W + 7) / 8, (H + 7) / 8, (D + 3) / 4);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    jfa3d_init_kernel<<<grd, blk>>>(d_occ, d_seed_a, W, H, D);
    int* in_ptr = d_seed_a;
    int* out_ptr = d_seed_b;
    int kmax = std::max(std::max(W, H), D) / 2;
    for (int k = kmax; k >= 1; k /= 2) {
        jfa3d_step_kernel<<<grd, blk>>>(in_ptr, out_ptr, W, H, D, k);
        std::swap(in_ptr, out_ptr);
    }
    jfa3d_to_dist_kernel<<<grd, blk>>>(in_ptr, d_dist, W, H, D, res);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    dist.resize(cells);
    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, cells * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_occ));
    CUDA_CHECK(cudaFree(d_seed_a));
    CUDA_CHECK(cudaFree(d_seed_b));
    CUDA_CHECK(cudaFree(d_dist));
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// -------------------------------------------------------------------------
// Render one z-slice
// -------------------------------------------------------------------------
static cv::Mat render_slice(const std::vector<float>& dist,
                            int W, int H, int D, int z, const char* title,
                            float ms) {
    cv::Mat img(H, W, CV_8UC3);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float d = dist[(z * H + y) * W + x];
            cv::Vec3b& px = img.at<cv::Vec3b>(H - 1 - y, x);
            float t = std::min(d / MAX_DIST, 1.0f);
            int r = static_cast<int>((1.0f - t) * 180.0f + 40.0f);
            int g = static_cast<int>(t * 200.0f + 30.0f);
            int b = static_cast<int>(80.0f + (1.0f - t) * 60.0f);
            px = cv::Vec3b(b, g, r);
        }
    }
    cv::Mat out;
    cv::resize(img, out, cv::Size(PANEL_W, PANEL_H), 0, 0, cv::INTER_NEAREST);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s  %.2f ms", title, ms);
    cv::rectangle(out, cv::Rect(0, 0, PANEL_W, 22), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(out, buf, cv::Point(8, 16), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    char buf2[64];
    std::snprintf(buf2, sizeof(buf2), "z=%.2f m", (z + 0.5f) * (WORLD_D / D));
    cv::putText(out, buf2, cv::Point(8, PANEL_H - 8), cv::FONT_HERSHEY_SIMPLEX,
                0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return out;
}

static cv::Mat make_panel(const std::vector<float>& dist, int W, int H, int D,
                          const char* title, float ms) {
    int z_lo = D / 4;
    int z_hi = D / 2;
    cv::Mat lo = render_slice(dist, W, H, D, z_lo, title, ms);
    cv::Mat hi = render_slice(dist, W, H, D, z_hi, "", ms);
    cv::Mat combined(PANEL_H, PANEL_W * 2, CV_8UC3);
    lo.copyTo(combined(cv::Rect(0, 0, PANEL_W, PANEL_H)));
    hi.copyTo(combined(cv::Rect(PANEL_W, 0, PANEL_W, PANEL_H)));
    return combined;
}

// -------------------------------------------------------------------------
// AVI -> GIF
// -------------------------------------------------------------------------
static void convert_avi_to_gif(const char* avi_path, const char* gif_path,
                               int fps) {
    char cmd[512];
    std::snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -i %s -vf 'fps=%d,scale=900:-1:flags=lanczos' -loop 0 %s "
        "> /dev/null 2>&1", avi_path, fps, gif_path);
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg conversion returned %d\n", rc);
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------
int main() {
    std::printf("3D ESDF comparison: CPU brute force %dx%dx%d vs GPU JFA %dx%dx%d "
                "on %.0fx%.0fx%.0f m\n",
                CPU_W, CPU_H, CPU_D, GPU_W, GPU_H, GPU_D,
                WORLD_W, WORLD_H, WORLD_D);

    std::vector<unsigned char> occ_cpu;
    std::vector<unsigned char> occ_gpu;
    build_scene(occ_cpu, CPU_W, CPU_H, CPU_D, CPU_RES);
    build_scene(occ_gpu, GPU_W, GPU_H, GPU_D, GPU_RES);

    // Validation: 32x32x16 CPU vs 32x32x16 GPU
    std::vector<unsigned char> occ_val;
    build_scene(occ_val, CPU_W, CPU_H, CPU_D, CPU_RES);
    std::vector<float> dist_cpu_val, dist_gpu_val;
    cpu_esdf3d_ms(occ_val, CPU_W, CPU_H, CPU_D, CPU_RES, dist_cpu_val);
    gpu_esdf3d_ms(occ_val, CPU_W, CPU_H, CPU_D, CPU_RES, dist_gpu_val);
    double max_err = 0.0;
    for (size_t i = 0; i < dist_cpu_val.size(); i++) {
        double e = std::abs(static_cast<double>(dist_cpu_val[i]) -
                            static_cast<double>(dist_gpu_val[i]));
        if (e > max_err) max_err = e;
    }
    std::printf("Validation on %dx%dx%d: max |CPU - GPU| = %.4f m\n",
                CPU_W, CPU_H, CPU_D, max_err);

    cv::VideoWriter video("gif/comparison_esdf_3d.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 15,
                          cv::Size(PANEL_W * 4 + 4, PANEL_H + 36));
    if (!video.isOpened()) {
        std::fprintf(stderr, "Failed to open gif/comparison_esdf_3d.avi\n");
        return 1;
    }

    std::mt19937 rng(2026);
    double cpu_ms_sum = 0.0, gpu_ms_sum = 0.0;
    int counted = 0;

    for (int f = 0; f < SIM_FRAMES; f++) {
        if (f > 0) {
            add_random_disk_stack(occ_cpu, CPU_W, CPU_H, CPU_D, CPU_RES, rng);
            std::mt19937 rng2(2026 + f);
            add_random_disk_stack(occ_gpu, GPU_W, GPU_H, GPU_D, GPU_RES, rng2);
        }

        std::vector<float> dist_cpu, dist_gpu;
        double cpu_ms = cpu_esdf3d_ms(occ_cpu, CPU_W, CPU_H, CPU_D, CPU_RES, dist_cpu);
        double gpu_ms = gpu_esdf3d_ms(occ_gpu, GPU_W, GPU_H, GPU_D, GPU_RES, dist_gpu);

        if (f >= 2) { cpu_ms_sum += cpu_ms; gpu_ms_sum += gpu_ms; counted++; }

        cv::Mat cpu_panel = make_panel(dist_cpu, CPU_W, CPU_H, CPU_D,
                                       "CPU brute force", cpu_ms);
        cv::Mat gpu_panel = make_panel(dist_gpu, GPU_W, GPU_H, GPU_D,
                                       "GPU JFA 3D", gpu_ms);

        cv::Mat frame(PANEL_H + 36, PANEL_W * 4 + 4, CV_8UC3,
                      cv::Scalar(30, 30, 30));
        cpu_panel.copyTo(frame(cv::Rect(0, 36, PANEL_W * 2, PANEL_H)));
        gpu_panel.copyTo(frame(cv::Rect(PANEL_W * 2 + 4, 36, PANEL_W * 2, PANEL_H)));
        char title[128];
        std::snprintf(title, sizeof(title),
                      "3D ESDF  CPU %dx%dx%d   |   GPU JFA %dx%dx%d",
                      CPU_W, CPU_H, CPU_D, GPU_W, GPU_H, GPU_D);
        cv::putText(frame, title, cv::Point(12, 26), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        video.write(frame);
    }
    video.release();

    if (counted > 0) {
        double cpu_ms = cpu_ms_sum / counted;
        double gpu_ms = gpu_ms_sum / counted;
        double cpu_per_us = cpu_ms * 1.0e3 / (static_cast<double>(CPU_W) * CPU_H * CPU_D);
        double gpu_per_us = gpu_ms * 1.0e3 / (static_cast<double>(GPU_W) * GPU_H * GPU_D);
        std::printf("Avg CPU %.2f ms / ESDF (%d voxels)\n"
                    "Avg GPU %.3f ms / ESDF (%d voxels)\n"
                    "Per-voxel: GPU %.4f us, CPU %.4f us "
                    "(%.0fx faster per voxel)\n",
                    cpu_ms, CPU_W * CPU_H * CPU_D,
                    gpu_ms, GPU_W * GPU_H * GPU_D,
                    gpu_per_us, cpu_per_us, cpu_per_us / gpu_per_us);
    }

    convert_avi_to_gif("gif/comparison_esdf_3d.avi",
                       "gif/comparison_esdf_3d.gif", 15);
    std::printf("GIF saved to gif/comparison_esdf_3d.gif\n");
    return 0;
}
