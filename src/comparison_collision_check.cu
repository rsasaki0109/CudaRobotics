/*************************************************************************
    Massive Parallel Collision Checker:
      CPU 1,024 candidate segments / scan vs CUDA 1,048,576 segments / scan.

    Each candidate is a straight-line motion from the current robot pose
    to a goal sampled uniformly inside the world. A candidate is feasible
    when no cell along the swept line (2D DDA traversal) is occupied. The
    same deterministic per-(frame, idx) hash produces goal samples on both
    sides, so the only difference is the candidate count and execution
    platform.

    GPU output is a per-segment feasibility flag plus the goal coordinate;
    the visualisation paints reachable goals as a dense green stipple,
    showing the actual free-space reachable from the robot at the current
    pose. CPU panel shows the same algorithm at 1024 candidates.

    Headline metric is per-candidate collision-check throughput.
 ************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------
constexpr int GRID_W = 400;
constexpr int GRID_H = 400;
constexpr float GRID_RES = 0.20f;            // 0.20 m/cell -> 80m x 80m world
constexpr float WORLD_W = GRID_W * GRID_RES;
constexpr float WORLD_H = GRID_H * GRID_RES;

constexpr int N_CAND_CPU = 1024;
constexpr int N_CAND_GPU = 1 << 20;          // 1,048,576

constexpr int PANEL_W = 600;
constexpr int PANEL_H = 600;
constexpr float VIS_SCALE = static_cast<float>(PANEL_W) / WORLD_W;

constexpr int SIM_FRAMES = 90;

// -------------------------------------------------------------------------
// Scene (same style as comparison_lidar_sim.cu so the demos read together)
// -------------------------------------------------------------------------
static void build_scene(std::vector<unsigned char>& grid) {
    grid.assign(GRID_W * GRID_H, 0u);
    auto set = [&](int gx, int gy) {
        if (gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H)
            grid[gy * GRID_W + gx] = 1u;
    };
    auto fill_rect = [&](int x0, int y0, int x1, int y1) {
        for (int gy = y0; gy <= y1; gy++)
            for (int gx = x0; gx <= x1; gx++) set(gx, gy);
    };
    auto fill_disk = [&](int cx, int cy, int r) {
        for (int gy = cy - r; gy <= cy + r; gy++)
            for (int gx = cx - r; gx <= cx + r; gx++) {
                int dx = gx - cx, dy = gy - cy;
                if (dx * dx + dy * dy <= r * r) set(gx, gy);
            }
    };
    fill_rect(0, 0, GRID_W - 1, 1);
    fill_rect(0, GRID_H - 2, GRID_W - 1, GRID_H - 1);
    fill_rect(0, 0, 1, GRID_H - 1);
    fill_rect(GRID_W - 2, 0, GRID_W - 1, GRID_H - 1);
    fill_rect(80, 100, 90, 300);
    fill_rect(310, 100, 320, 250);
    fill_rect(150, 240, 260, 250);
    fill_rect(150, 50,  160, 180);
    fill_rect(220, 60, 300, 70);
    fill_disk(140, 110, 8);
    fill_disk(140, 320, 8);
    fill_disk(220, 320, 8);
    fill_disk(290, 180, 8);
    fill_disk(360, 320, 8);
    fill_disk( 50,  60, 8);
    fill_disk( 50, 260, 8);
    fill_disk(360,  80, 8);
}

// -------------------------------------------------------------------------
// Deterministic goal hash. Same on CPU and GPU.
// -------------------------------------------------------------------------
__host__ __device__ static float u01(unsigned int h) {
    return (h & 0x7fffffffu) * (1.0f / 2147483647.0f);
}

__host__ __device__ static void sample_goal(int frame, int idx,
                                            float& gx, float& gy) {
    unsigned int h = static_cast<unsigned int>(idx) * 73856093u
                   ^ static_cast<unsigned int>(frame) * 19349663u;
    h = (h ^ (h >> 16)) * 2654435761u;
    float u = u01(h);
    h = (h ^ (h >> 13)) * 1597334677u;
    float v = u01(h);
    gx = u * WORLD_W;
    gy = v * WORLD_H;
}

// -------------------------------------------------------------------------
// CPU collision check via 2D DDA
// -------------------------------------------------------------------------
static bool cpu_segment_free(const unsigned char* grid,
                             float sx, float sy, float ex, float ey) {
    float dx = ex - sx;
    float dy = ey - sy;
    float fx = sx / GRID_RES;
    float fy = sy / GRID_RES;
    int gx = static_cast<int>(std::floor(fx));
    int gy = static_cast<int>(std::floor(fy));
    int step_x = (dx > 0.0f) ? 1 : -1;
    int step_y = (dy > 0.0f) ? 1 : -1;
    float inv_dx = (std::fabs(dx) > 1e-7f) ? 1.0f / std::fabs(dx) : 1e30f;
    float inv_dy = (std::fabs(dy) > 1e-7f) ? 1.0f / std::fabs(dy) : 1e30f;
    float t_max_x = (dx > 0.0f) ? (gx + 1 - fx) * GRID_RES * inv_dx
                                : (fx - gx) * GRID_RES * inv_dx;
    float t_max_y = (dy > 0.0f) ? (gy + 1 - fy) * GRID_RES * inv_dy
                                : (fy - gy) * GRID_RES * inv_dy;
    float dt_x = GRID_RES * inv_dx;
    float dt_y = GRID_RES * inv_dy;
    float seg_len = std::sqrt(dx * dx + dy * dy);

    while (true) {
        if (gx < 0 || gx >= GRID_W || gy < 0 || gy >= GRID_H) return true;
        if (grid[gy * GRID_W + gx]) return false;
        float t_next = std::min(t_max_x, t_max_y);
        if (t_next >= seg_len) return true;
        if (t_max_x < t_max_y) { t_max_x += dt_x; gx += step_x; }
        else                   { t_max_y += dt_y; gy += step_y; }
    }
}

// -------------------------------------------------------------------------
// GPU collision check
// -------------------------------------------------------------------------
__global__ void check_kernel(const unsigned char* __restrict__ grid,
                             float sx, float sy, int frame, int n_cand,
                             unsigned char* __restrict__ out_feasible,
                             float2* __restrict__ out_goal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cand) return;
    float ex, ey;
    sample_goal(frame, i, ex, ey);
    out_goal[i] = make_float2(ex, ey);

    float dx = ex - sx;
    float dy = ey - sy;
    float fx = sx / GRID_RES;
    float fy = sy / GRID_RES;
    int gx = static_cast<int>(floorf(fx));
    int gy = static_cast<int>(floorf(fy));
    int step_x = (dx > 0.0f) ? 1 : -1;
    int step_y = (dy > 0.0f) ? 1 : -1;
    float inv_dx = (fabsf(dx) > 1e-7f) ? 1.0f / fabsf(dx) : 1e30f;
    float inv_dy = (fabsf(dy) > 1e-7f) ? 1.0f / fabsf(dy) : 1e30f;
    float t_max_x = (dx > 0.0f) ? (gx + 1 - fx) * GRID_RES * inv_dx
                                : (fx - gx) * GRID_RES * inv_dx;
    float t_max_y = (dy > 0.0f) ? (gy + 1 - fy) * GRID_RES * inv_dy
                                : (fy - gy) * GRID_RES * inv_dy;
    float dt_x = GRID_RES * inv_dx;
    float dt_y = GRID_RES * inv_dy;
    float seg_len = sqrtf(dx * dx + dy * dy);
    unsigned char free_flag = 1u;

    #pragma unroll 4
    for (int it = 0; it < GRID_W + GRID_H; it++) {
        if (gx < 0 || gx >= GRID_W || gy < 0 || gy >= GRID_H) break;
        if (grid[gy * GRID_W + gx]) { free_flag = 0u; break; }
        float t_next = fminf(t_max_x, t_max_y);
        if (t_next >= seg_len) break;
        if (t_max_x < t_max_y) { t_max_x += dt_x; gx += step_x; }
        else                   { t_max_y += dt_y; gy += step_y; }
    }
    out_feasible[i] = free_flag;
}

// -------------------------------------------------------------------------
// Rendering
// -------------------------------------------------------------------------
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

static void draw_cpu_lines(cv::Mat& panel, float sx, float sy,
                           const std::vector<float>& gx,
                           const std::vector<float>& gy,
                           const std::vector<unsigned char>& feas) {
    auto sensor = world_to_px(sx, sy);
    for (size_t i = 0; i < feas.size(); i++) {
        auto endp = world_to_px(gx[i], gy[i]);
        cv::Scalar col = feas[i] ? cv::Scalar(40, 160, 40)
                                 : cv::Scalar(60, 60, 200);
        cv::line(panel, sensor, endp, col, 1, cv::LINE_AA);
        cv::circle(panel, endp, 2, col, -1, cv::LINE_AA);
    }
    cv::circle(panel, sensor, 7, cv::Scalar(0, 100, 200), -1, cv::LINE_AA);
    cv::circle(panel, sensor, 7, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

// Direct pixel splat for 1M endpoints. Only paint reachable goals so the
// resulting blob is the actually free-space-reachable region from the
// sensor pose; blocked candidates contribute nothing.
static void draw_gpu_endpoints(cv::Mat& panel,
                               const std::vector<float2>& goals,
                               const std::vector<unsigned char>& feas) {
    int stride = panel.step;
    unsigned char* data = panel.data;
    cv::Vec3b green(40, 200, 80);
    for (size_t i = 0; i < feas.size(); i++) {
        if (!feas[i]) continue;
        int px = static_cast<int>(goals[i].x * VIS_SCALE);
        int py = PANEL_H - 1 - static_cast<int>(goals[i].y * VIS_SCALE);
        if (px < 0 || px >= PANEL_W - 1 || py < 0 || py >= PANEL_H - 1) continue;
        for (int dy = 0; dy < 2; dy++) {
            unsigned char* row = data + (py + dy) * stride + px * 3;
            row[0] = green[0]; row[1] = green[1]; row[2] = green[2];
            row[3] = green[0]; row[4] = green[1]; row[5] = green[2];
        }
    }
}

static void draw_label(cv::Mat& panel, const std::string& text, int y_offset) {
    cv::putText(panel, text, cv::Point(12, y_offset),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
    cv::putText(panel, text, cv::Point(12, y_offset),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

// -------------------------------------------------------------------------
// Validation: GPU and CPU agree per-candidate at a small N
// -------------------------------------------------------------------------
static void validate_consistency(const unsigned char* h_grid,
                                 const unsigned char* d_grid) {
    constexpr int N = 4096;
    float sx = WORLD_W * 0.5f;
    float sy = WORLD_H * 0.5f;
    unsigned char* d_feas;
    float2* d_goal;
    CUDA_CHECK(cudaMalloc(&d_feas, N));
    CUDA_CHECK(cudaMalloc(&d_goal, N * sizeof(float2)));
    int blk = 256, grd = (N + blk - 1) / blk;
    check_kernel<<<grd, blk>>>(d_grid, sx, sy, /*frame=*/0, N, d_feas, d_goal);
    std::vector<unsigned char> hf(N);
    std::vector<float2> hg(N);
    CUDA_CHECK(cudaMemcpy(hf.data(), d_feas, N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hg.data(), d_goal, N * sizeof(float2),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_feas); cudaFree(d_goal);

    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        bool cpu = cpu_segment_free(h_grid, sx, sy, hg[i].x, hg[i].y);
        bool gpu = hf[i] != 0;
        if (cpu != gpu) mismatches++;
    }
    std::printf("Validation: %d / %d agreement (%.3f%%) between CPU DDA and "
                "CUDA DDA at central pose\n",
                N - mismatches, N, 100.0 * (N - mismatches) / N);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    std::printf("Massive collision check: CPU %d segments vs CUDA %d segments per scan\n",
                N_CAND_CPU, N_CAND_GPU);
    std::vector<unsigned char> h_grid;
    build_scene(h_grid);

    unsigned char* d_grid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid, GRID_W * GRID_H));
    CUDA_CHECK(cudaMemcpy(d_grid, h_grid.data(), GRID_W * GRID_H,
                          cudaMemcpyHostToDevice));
    unsigned char* d_feas = nullptr;
    float2* d_goal = nullptr;
    CUDA_CHECK(cudaMalloc(&d_feas, N_CAND_GPU));
    CUDA_CHECK(cudaMalloc(&d_goal, N_CAND_GPU * sizeof(float2)));
    std::vector<unsigned char> h_feas(N_CAND_GPU);
    std::vector<float2> h_goal(N_CAND_GPU);

    validate_consistency(h_grid.data(), d_grid);

    cv::VideoWriter video("gif/comparison_collision_check.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
                          cv::Size(PANEL_W * 2, PANEL_H));

    double cpu_ms_sum = 0.0, gpu_ms_sum = 0.0;
    long cpu_n_feas_sum = 0, gpu_n_feas_sum = 0;
    int  timed_frames = 0;

    float cx_w = WORLD_W * 0.5f;
    float cy_w = WORLD_H * 0.5f;
    float a = WORLD_W * 0.22f;
    float b = WORLD_H * 0.22f;

    std::vector<float> cpu_gx(N_CAND_CPU), cpu_gy(N_CAND_CPU);
    std::vector<unsigned char> cpu_feas(N_CAND_CPU);

    for (int f = 0; f < SIM_FRAMES; f++) {
        float u = static_cast<float>(f) / SIM_FRAMES;
        float traj_t = 2.0f * static_cast<float>(M_PI) * u;
        float sx = cx_w + a * std::cos(traj_t);
        float sy = cy_w + b * std::sin(traj_t * 1.3f);

        // CPU sweep
        auto cpu_t0 = std::chrono::high_resolution_clock::now();
        int cpu_n_feas = 0;
        for (int i = 0; i < N_CAND_CPU; i++) {
            sample_goal(f, i, cpu_gx[i], cpu_gy[i]);
            bool ok = cpu_segment_free(h_grid.data(), sx, sy,
                                       cpu_gx[i], cpu_gy[i]);
            cpu_feas[i] = ok ? 1u : 0u;
            if (ok) cpu_n_feas++;
        }
        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

        // GPU sweep
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        int blk = 256, grd = (N_CAND_GPU + blk - 1) / blk;
        check_kernel<<<grd, blk>>>(d_grid, sx, sy, f, N_CAND_GPU,
                                   d_feas, d_goal);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        CUDA_CHECK(cudaMemcpy(h_feas.data(), d_feas, N_CAND_GPU,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_goal.data(), d_goal,
                              N_CAND_GPU * sizeof(float2),
                              cudaMemcpyDeviceToHost));
        int gpu_n_feas = 0;
        for (int i = 0; i < N_CAND_GPU; i++) if (h_feas[i]) gpu_n_feas++;

        if (f >= 5) {
            cpu_ms_sum += cpu_ms;
            gpu_ms_sum += gpu_ms;
            cpu_n_feas_sum += cpu_n_feas;
            gpu_n_feas_sum += gpu_n_feas;
            timed_frames++;
        }

        cv::Mat left(PANEL_H, PANEL_W, CV_8UC3);
        cv::Mat right(PANEL_H, PANEL_W, CV_8UC3);
        draw_scene(left, h_grid);
        draw_scene(right, h_grid);
        draw_cpu_lines(left, sx, sy, cpu_gx, cpu_gy, cpu_feas);
        draw_gpu_endpoints(right, h_goal, h_feas);
        auto sensor_px = world_to_px(sx, sy);
        cv::circle(right, sensor_px, 7, cv::Scalar(0, 100, 200), -1, cv::LINE_AA);
        cv::circle(right, sensor_px, 7, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        char buf[160];
        std::snprintf(buf, sizeof(buf), "CPU %d segments  %.1f ms  (%d feas)",
                      N_CAND_CPU, cpu_ms, cpu_n_feas);
        draw_label(left, buf, 28);
        std::snprintf(buf, sizeof(buf), "GPU %d segments  %.3f ms  (%d feas)",
                      N_CAND_GPU, gpu_ms, gpu_n_feas);
        draw_label(right, buf, 28);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        video.write(combined);
    }

    video.release();
    cudaFree(d_grid); cudaFree(d_feas); cudaFree(d_goal);

    if (timed_frames > 0) {
        double cpu_ms = cpu_ms_sum / timed_frames;
        double gpu_ms = gpu_ms_sum / timed_frames;
        double cpu_per_cand_us = cpu_ms * 1.0e3 / N_CAND_CPU;
        double gpu_per_cand_us = gpu_ms * 1.0e3 / N_CAND_GPU;
        double cpu_feas_frac = static_cast<double>(cpu_n_feas_sum) /
                               static_cast<double>(timed_frames) / N_CAND_CPU;
        double gpu_feas_frac = static_cast<double>(gpu_n_feas_sum) /
                               static_cast<double>(timed_frames) / N_CAND_GPU;
        std::printf("Avg CPU %.2f ms / scan (%d segments, %.2f%% feasible)\n"
                    "Avg GPU %.3f ms / scan (%d segments, %.2f%% feasible)\n"
                    "Per-candidate throughput: GPU %.4f us/cand vs CPU %.3f us/cand "
                    "(%.0fx faster per candidate)\n",
                    cpu_ms, N_CAND_CPU, 100.0 * cpu_feas_frac,
                    gpu_ms, N_CAND_GPU, 100.0 * gpu_feas_frac,
                    gpu_per_cand_us, cpu_per_cand_us,
                    cpu_per_cand_us / gpu_per_cand_us);
    }

    std::system("ffmpeg -y -i gif/comparison_collision_check.avi "
                "-vf 'fps=15,scale=900:-1:flags=lanczos' -loop 0 "
                "gif/comparison_collision_check.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_collision_check.gif" << std::endl;
    return 0;
}
