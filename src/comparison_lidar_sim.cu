/*************************************************************************
    Massive Lidar Simulator: CPU 1,024 rays/scan vs GPU 1,048,576 rays/scan.
    Same 2D scene, same sensor pose, same ray-step (DDA-style traversal of
    an occupancy grid). The visual contrast at GPU scale is a continuous
    outline of every visible surface vs sparse dots on the CPU side.
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int GRID_W = 400;
constexpr int GRID_H = 400;
constexpr float GRID_RES = 0.20f;  // 0.20 m/cell -> 80m x 80m world
constexpr float WORLD_W = GRID_W * GRID_RES;
constexpr float WORLD_H = GRID_H * GRID_RES;
constexpr float MAX_RANGE = 30.0f;
constexpr int   N_RAYS_CPU = 1024;
constexpr int   N_RAYS_GPU = 1 << 20;  // 1,048,576

constexpr int PANEL_W = 600;
constexpr int PANEL_H = 600;
constexpr float VIS_SCALE = static_cast<float>(PANEL_W) / WORLD_W;

constexpr int SIM_FRAMES = 90;

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------
static void build_scene(std::vector<unsigned char>& grid) {
    grid.assign(GRID_W * GRID_H, 0u);
    auto set = [&](int gx, int gy) {
        if (gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H)
            grid[gy * GRID_W + gx] = 1u;
    };
    auto fill_rect = [&](int x0, int y0, int x1, int y1) {
        for (int gy = y0; gy <= y1; gy++)
            for (int gx = x0; gx <= x1; gx++)
                set(gx, gy);
    };
    auto fill_disk = [&](int cx, int cy, int r) {
        for (int gy = cy - r; gy <= cy + r; gy++)
            for (int gx = cx - r; gx <= cx + r; gx++) {
                int dx = gx - cx, dy = gy - cy;
                if (dx * dx + dy * dy <= r * r) set(gx, gy);
            }
    };

    // Outer walls
    fill_rect(0, 0, GRID_W - 1, 1);
    fill_rect(0, GRID_H - 2, GRID_W - 1, GRID_H - 1);
    fill_rect(0, 0, 1, GRID_H - 1);
    fill_rect(GRID_W - 2, 0, GRID_W - 1, GRID_H - 1);

    // Inner walls (corridor structure)
    fill_rect(80, 100, 90, 300);
    fill_rect(310, 100, 320, 250);
    fill_rect(150, 240, 260, 250);
    fill_rect(150, 50,  160, 180);
    fill_rect(220, 60, 300, 70);

    // Pillars
    fill_disk(140, 110, 8);
    fill_disk(140, 320, 8);
    fill_disk(220, 320, 8);
    fill_disk(290, 180, 8);
    fill_disk(360, 320, 8);
    fill_disk( 50,  60, 8);
    fill_disk( 50, 260, 8);
    fill_disk(360,  80, 8);
}

// ---------------------------------------------------------------------------
// CPU raycast (DDA along ray direction in grid coordinates)
// ---------------------------------------------------------------------------
static float cpu_raycast(const unsigned char* grid, float sx, float sy,
                         float theta, float max_range) {
    float cx = sx / GRID_RES;
    float cy = sy / GRID_RES;
    float dx = std::cos(theta);
    float dy = std::sin(theta);
    int   step_x = (dx > 0.0f) ? 1 : -1;
    int   step_y = (dy > 0.0f) ? 1 : -1;
    int   gx = static_cast<int>(std::floor(cx));
    int   gy = static_cast<int>(std::floor(cy));
    float inv_dx = (dx != 0.0f) ? std::fabs(1.0f / dx) : 1e30f;
    float inv_dy = (dy != 0.0f) ? std::fabs(1.0f / dy) : 1e30f;
    float t_max_x = (dx > 0.0f) ? (gx + 1 - cx) * inv_dx
                                : (cx - gx) * inv_dx;
    float t_max_y = (dy > 0.0f) ? (gy + 1 - cy) * inv_dy
                                : (cy - gy) * inv_dy;
    float t_grid = max_range / GRID_RES;

    while (true) {
        if (gx < 0 || gx >= GRID_W || gy < 0 || gy >= GRID_H) return max_range;
        if (grid[gy * GRID_W + gx]) {
            float t_cell = std::min(t_max_x, t_max_y);
            return std::min(t_cell, t_grid) * GRID_RES;
        }
        if (t_max_x < t_max_y) {
            t_max_x += inv_dx;
            gx += step_x;
            if (t_max_x * GRID_RES > max_range) return max_range;
        } else {
            t_max_y += inv_dy;
            gy += step_y;
            if (t_max_y * GRID_RES > max_range) return max_range;
        }
    }
}

// ---------------------------------------------------------------------------
// GPU raycast
// ---------------------------------------------------------------------------
__global__ void raycast_kernel(const unsigned char* __restrict__ grid,
                               int gridW, int gridH, float gridRes,
                               float sx, float sy, float angle0,
                               float angle_step, int n_rays, float max_range,
                               float* __restrict__ d_dist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) return;
    float theta = angle0 + i * angle_step;
    float cx = sx / gridRes;
    float cy = sy / gridRes;
    float dx, dy;
    sincosf(theta, &dy, &dx);
    int step_x = (dx > 0.0f) ? 1 : -1;
    int step_y = (dy > 0.0f) ? 1 : -1;
    int gx = static_cast<int>(floorf(cx));
    int gy = static_cast<int>(floorf(cy));
    float inv_dx = (dx != 0.0f) ? fabsf(1.0f / dx) : 1e30f;
    float inv_dy = (dy != 0.0f) ? fabsf(1.0f / dy) : 1e30f;
    float t_max_x = (dx > 0.0f) ? (gx + 1 - cx) * inv_dx : (cx - gx) * inv_dx;
    float t_max_y = (dy > 0.0f) ? (gy + 1 - cy) * inv_dy : (cy - gy) * inv_dy;
    float t_grid_limit = max_range / gridRes;
    float dist = max_range;
    for (int it = 0; it < gridW + gridH; it++) {
        if (gx < 0 || gx >= gridW || gy < 0 || gy >= gridH) break;
        if (grid[gy * gridW + gx]) {
            float t_cell = fminf(t_max_x, t_max_y);
            dist = fminf(t_cell, t_grid_limit) * gridRes;
            break;
        }
        if (t_max_x < t_max_y) {
            t_max_x += inv_dx;
            gx += step_x;
            if (t_max_x > t_grid_limit) break;
        } else {
            t_max_y += inv_dy;
            gy += step_y;
            if (t_max_y > t_grid_limit) break;
        }
    }
    d_dist[i] = dist;
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------
static cv::Point2i world_to_px(float x, float y) {
    int px = static_cast<int>(x * VIS_SCALE);
    int py = PANEL_H - 1 - static_cast<int>(y * VIS_SCALE);
    return {px, py};
}

static void draw_scene(cv::Mat& panel, const std::vector<unsigned char>& grid) {
    panel.setTo(cv::Scalar(245, 245, 245));
    for (int gy = 0; gy < GRID_H; gy++) {
        for (int gx = 0; gx < GRID_W; gx++) {
            if (grid[gy * GRID_W + gx]) {
                int x0 = static_cast<int>(gx * GRID_RES * VIS_SCALE);
                int y0 = PANEL_H - 1 - static_cast<int>((gy + 1) * GRID_RES * VIS_SCALE);
                int x1 = static_cast<int>((gx + 1) * GRID_RES * VIS_SCALE);
                int y1 = PANEL_H - 1 - static_cast<int>(gy * GRID_RES * VIS_SCALE);
                cv::rectangle(panel, cv::Point(x0, y0), cv::Point(x1, y1),
                              cv::Scalar(70, 70, 70), -1);
            }
        }
    }
}

static void draw_sparse_rays(cv::Mat& panel, float sx, float sy,
                             float angle0, float angle_step, int n_rays,
                             const float* dist, cv::Scalar hit_color,
                             int hit_radius) {
    auto sensor_px = world_to_px(sx, sy);
    for (int i = 0; i < n_rays; i++) {
        float theta = angle0 + i * angle_step;
        float d = dist[i];
        float hx = sx + d * std::cos(theta);
        float hy = sy + d * std::sin(theta);
        auto hit_px = world_to_px(hx, hy);
        cv::line(panel, sensor_px, hit_px, cv::Scalar(200, 220, 240), 1,
                 cv::LINE_AA);
        if (d < MAX_RANGE - 1e-3f) {
            cv::circle(panel, hit_px, hit_radius, hit_color, -1, cv::LINE_AA);
        }
    }
    cv::circle(panel, sensor_px, 7, cv::Scalar(0, 100, 200), -1, cv::LINE_AA);
    cv::circle(panel, sensor_px, 7, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

// Direct-pixel splatting for the 1M-ray panel. Each hit writes a
// single pixel; cv::circle would be hundreds of milliseconds per
// frame at 1M points and dominate runtime.
static void draw_dense_hits(cv::Mat& panel, float sx, float sy,
                            float angle0, float angle_step, int n_rays,
                            const float* dist, cv::Vec3b hit_color) {
    int stride = panel.step;
    unsigned char* data = panel.data;
    for (int i = 0; i < n_rays; i++) {
        float d = dist[i];
        if (d >= MAX_RANGE - 1e-3f) continue;
        float theta = angle0 + i * angle_step;
        float hx = sx + d * std::cos(theta);
        float hy = sy + d * std::sin(theta);
        int px = static_cast<int>(hx * VIS_SCALE);
        int py = PANEL_H - 1 - static_cast<int>(hy * VIS_SCALE);
        if (px < 0 || px >= PANEL_W || py < 0 || py >= PANEL_H) continue;
        unsigned char* p = data + py * stride + px * 3;
        p[0] = hit_color[0];
        p[1] = hit_color[1];
        p[2] = hit_color[2];
    }
    auto sensor_px = world_to_px(sx, sy);
    cv::circle(panel, sensor_px, 7, cv::Scalar(0, 100, 200), -1, cv::LINE_AA);
    cv::circle(panel, sensor_px, 7, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

static void draw_label(cv::Mat& panel, const std::string& text, int y_offset) {
    cv::putText(panel, text, cv::Point(12, y_offset),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "Massive Lidar Simulator comparison: CPU "
              << N_RAYS_CPU << " rays vs GPU " << N_RAYS_GPU << " rays per scan"
              << std::endl;

    std::vector<unsigned char> h_grid;
    build_scene(h_grid);

    unsigned char* d_grid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid, GRID_W * GRID_H * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_grid, h_grid.data(),
                          GRID_W * GRID_H * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    float* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist, N_RAYS_GPU * sizeof(float)));
    std::vector<float> h_dist_gpu(N_RAYS_GPU);
    std::vector<float> h_dist_cpu(N_RAYS_CPU);

    cv::VideoWriter video("gif/comparison_lidar_sim.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
                          cv::Size(PANEL_W * 2, PANEL_H));

    // Sensor follows an elliptic trajectory through the scene
    float cx_w = WORLD_W * 0.5f;
    float cy_w = WORLD_H * 0.5f;
    float a = WORLD_W * 0.22f;
    float b = WORLD_H * 0.22f;

    double cpu_ms_sum = 0.0;
    double gpu_ms_sum = 0.0;
    int    timed_frames = 0;

    for (int f = 0; f < SIM_FRAMES; f++) {
        float u = static_cast<float>(f) / SIM_FRAMES;
        float traj_t = 2.0f * static_cast<float>(M_PI) * u;
        float sx = cx_w + a * std::cos(traj_t);
        float sy = cy_w + b * std::sin(traj_t * 1.3f);
        float scan_angle = traj_t * 0.7f;  // sweep angle offset

        // CPU raycast
        auto cpu_t0 = std::chrono::high_resolution_clock::now();
        float angle_step_cpu = 2.0f * static_cast<float>(M_PI) / N_RAYS_CPU;
        for (int i = 0; i < N_RAYS_CPU; i++) {
            float theta = scan_angle + i * angle_step_cpu;
            h_dist_cpu[i] = cpu_raycast(h_grid.data(), sx, sy, theta, MAX_RANGE);
        }
        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

        // GPU raycast
        float angle_step_gpu = 2.0f * static_cast<float>(M_PI) / N_RAYS_GPU;
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        int block = 256;
        int grid = (N_RAYS_GPU + block - 1) / block;
        raycast_kernel<<<grid, block>>>(d_grid, GRID_W, GRID_H, GRID_RES,
                                        sx, sy, scan_angle, angle_step_gpu,
                                        N_RAYS_GPU, MAX_RANGE, d_dist);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        CUDA_CHECK(cudaMemcpy(h_dist_gpu.data(), d_dist,
                              N_RAYS_GPU * sizeof(float), cudaMemcpyDeviceToHost));

        if (f >= 5) {  // skip warmup
            cpu_ms_sum += cpu_ms;
            gpu_ms_sum += gpu_ms;
            timed_frames++;
        }

        // Visualization
        cv::Mat left(PANEL_H, PANEL_W, CV_8UC3);
        cv::Mat right(PANEL_H, PANEL_W, CV_8UC3);
        draw_scene(left, h_grid);
        draw_scene(right, h_grid);
        draw_sparse_rays(left, sx, sy,
                         scan_angle, angle_step_cpu, N_RAYS_CPU,
                         h_dist_cpu.data(),
                         cv::Scalar(0, 0, 220), 2);
        draw_dense_hits(right, sx, sy,
                        scan_angle, angle_step_gpu, N_RAYS_GPU,
                        h_dist_gpu.data(),
                        cv::Vec3b(0, 140, 0));

        char buf[128];
        std::snprintf(buf, sizeof(buf), "CPU 1,024 rays  %.1f ms", cpu_ms);
        draw_label(left, buf, 28);
        std::snprintf(buf, sizeof(buf), "GPU 1,048,576 rays  %.2f ms", gpu_ms);
        draw_label(right, buf, 28);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        video.write(combined);
    }

    video.release();
    cudaFree(d_grid);
    cudaFree(d_dist);

    if (timed_frames > 0) {
        double cpu_ms = cpu_ms_sum / timed_frames;
        double gpu_ms = gpu_ms_sum / timed_frames;
        double cpu_per_ray_us = cpu_ms * 1.0e3 / N_RAYS_CPU;
        double gpu_per_ray_us = gpu_ms * 1.0e3 / N_RAYS_GPU;
        std::printf("Avg CPU %.2f ms / scan (%d rays)\n"
                    "Avg GPU %.2f ms / scan (%d rays)\n"
                    "Per-ray throughput: GPU %.4f us/ray vs CPU %.3f us/ray "
                    "(%.0fx faster per ray)\n",
                    cpu_ms, N_RAYS_CPU, gpu_ms, N_RAYS_GPU,
                    gpu_per_ray_us, cpu_per_ray_us,
                    cpu_per_ray_us / gpu_per_ray_us);
    }

    std::system("ffmpeg -y -i gif/comparison_lidar_sim.avi "
                "-vf 'fps=15,scale=900:-1:flags=lanczos' -loop 0 "
                "gif/comparison_lidar_sim.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_lidar_sim.gif" << std::endl;
    return 0;
}
