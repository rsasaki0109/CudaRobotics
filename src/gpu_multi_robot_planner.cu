// gpu_multi_robot_planner.cu
//
// Massive multi-robot path planning on GPU.
//
//   - N_ROBOTS independent agents share a single 2D occupancy grid.
//   - Per robot we compute a Bellman-Ford "distance to goal" field on the
//     GPU (one thread per (robot, cell)), so all N fields are built in
//     parallel in a single kernel sweep that we iterate to convergence.
//   - At simulation time each robot greedily descends its own field while
//     applying a short-range repulsion from its neighbors.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "cuda_check.cuh"

namespace cudabot {

constexpr int N_ROBOTS = 200;
constexpr int GRID = 256;
constexpr float RES = 0.1f;
constexpr int   BF_ITERS = 180;
constexpr int   SIM_STEPS = 220;
constexpr float ROBOT_SPEED = 0.18f;
constexpr float REPULSION_RADIUS = 0.7f;
constexpr float REPULSION_STRENGTH = 0.55f;
constexpr float ARRIVAL_DIST = 0.35f;
constexpr float INF_DIST = 1.0e6f;

static void fill_rect(std::vector<unsigned char>& occ, int x0, int y0, int x1, int y1) {
    x0 = std::max(0, x0); y0 = std::max(0, y0);
    x1 = std::min(GRID - 1, x1); y1 = std::min(GRID - 1, y1);
    for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++) occ[y * GRID + x] = 1u;
}

static void build_occupancy(std::vector<unsigned char>& occ) {
    occ.assign(GRID * GRID, 0u);
    fill_rect(occ, 0, 0, GRID - 1, 1);
    fill_rect(occ, 0, GRID - 2, GRID - 1, GRID - 1);
    fill_rect(occ, 0, 0, 1, GRID - 1);
    fill_rect(occ, GRID - 2, 0, GRID - 1, GRID - 1);
    int pillars[][2] = {
        {40, 60}, {40, 130}, {40, 200},
        {120, 40}, {120, 110}, {120, 180},
        {200, 70}, {200, 150}, {200, 220},
        {170, 30}, {80, 220}
    };
    for (auto& p : pillars) fill_rect(occ, p[0], p[1], p[0] + 12, p[1] + 12);
    fill_rect(occ, 70, 35, 100, 38);
    fill_rect(occ, 150, 90, 180, 93);
    fill_rect(occ, 60, 160, 90, 163);
    fill_rect(occ, 150, 200, 200, 203);
}

__global__ void init_dist_kernel(int n_robots, int grid,
                                 const unsigned char* __restrict__ occ,
                                 const int* __restrict__ goal_cells,
                                 float* __restrict__ dist) {
    int r = blockIdx.y;
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.z * blockDim.y + threadIdx.y;
    if (cx >= grid || cy >= grid || r >= n_robots) return;
    int cell = cy * grid + cx;
    float v = INF_DIST;
    if (occ[cell] == 0u && cell == goal_cells[r]) v = 0.0f;
    dist[r * grid * grid + cell] = v;
}

__global__ void bf_relax_kernel(int n_robots, int grid,
                                const unsigned char* __restrict__ occ,
                                const float* __restrict__ dist_in,
                                float* __restrict__ dist_out) {
    int r = blockIdx.y;
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.z * blockDim.y + threadIdx.y;
    if (cx >= grid || cy >= grid || r >= n_robots) return;
    int cell = cy * grid + cx;
    int base = r * grid * grid;
    float best = dist_in[base + cell];
    if (occ[cell] != 0u) { dist_out[base + cell] = INF_DIST; return; }
    #pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        #pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = cx + dx;
            int ny = cy + dy;
            if (nx < 0 || nx >= grid || ny < 0 || ny >= grid) continue;
            float v = dist_in[base + ny * grid + nx];
            float step = (dx != 0 && dy != 0) ? 1.4142136f : 1.0f;
            float cand = v + step;
            if (cand < best) best = cand;
        }
    }
    dist_out[base + cell] = best;
}

__global__ void simulate_step_kernel(int n_robots, int grid, float res,
                                     const unsigned char* __restrict__ occ,
                                     const float* __restrict__ dist,
                                     const float* __restrict__ pos_in,
                                     const unsigned char* __restrict__ arrived,
                                     float* __restrict__ pos_out) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_robots) return;
    if (arrived[r] != 0u) {
        pos_out[2 * r + 0] = pos_in[2 * r + 0];
        pos_out[2 * r + 1] = pos_in[2 * r + 1];
        return;
    }
    float x = pos_in[2 * r + 0];
    float y = pos_in[2 * r + 1];
    int cx = static_cast<int>(x / res);
    int cy = static_cast<int>(y / res);
    if (cx < 0 || cx >= grid || cy < 0 || cy >= grid) {
        pos_out[2 * r + 0] = x; pos_out[2 * r + 1] = y; return;
    }
    const float* df = dist + r * grid * grid;
    float best = df[cy * grid + cx];
    int bdx = 0, bdy = 0;
    #pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        #pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = cx + dx;
            int ny = cy + dy;
            if (nx < 0 || nx >= grid || ny < 0 || ny >= grid) continue;
            if (occ[ny * grid + nx] != 0u) continue;
            float v = df[ny * grid + nx];
            if (v < best) { best = v; bdx = dx; bdy = dy; }
        }
    }
    float dxg = static_cast<float>(bdx);
    float dyg = static_cast<float>(bdy);
    float gl = sqrtf(dxg * dxg + dyg * dyg);
    if (gl > 1.0e-6f) { dxg /= gl; dyg /= gl; }

    float rx = 0.0f, ry = 0.0f;
    for (int k = 0; k < n_robots; k++) {
        if (k == r) continue;
        float dx = x - pos_in[2 * k + 0];
        float dy = y - pos_in[2 * k + 1];
        float d2 = dx * dx + dy * dy;
        if (d2 > REPULSION_RADIUS * REPULSION_RADIUS) continue;
        float d = sqrtf(d2) + 1.0e-4f;
        float w = (REPULSION_RADIUS - d) / REPULSION_RADIUS;
        rx += w * w * dx / d;
        ry += w * w * dy / d;
    }
    float vx = dxg + REPULSION_STRENGTH * rx;
    float vy = dyg + REPULSION_STRENGTH * ry;
    float vl = sqrtf(vx * vx + vy * vy);
    if (vl < 1.0e-6f) { vx = dxg; vy = dyg; vl = 1.0f; }
    vx /= vl; vy /= vl;

    float nxw = x + vx * ROBOT_SPEED;
    float nyw = y + vy * ROBOT_SPEED;
    int ncx = static_cast<int>(nxw / res);
    int ncy = static_cast<int>(nyw / res);
    if (ncx < 0 || ncx >= grid || ncy < 0 || ncy >= grid ||
        occ[ncy * grid + ncx] != 0u) {
        nxw = x; nyw = y;
    }
    pos_out[2 * r + 0] = nxw;
    pos_out[2 * r + 1] = nyw;
}

static cv::Vec3b hsv_to_bgr(float h, float s, float v) {
    float c = v * s;
    float x = c * (1.0f - std::fabs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    float r, g, b;
    int hi = static_cast<int>(h * 6.0f) % 6;
    switch (hi) {
        case 0: r = c; g = x; b = 0; break;
        case 1: r = x; g = c; b = 0; break;
        case 2: r = 0; g = c; b = x; break;
        case 3: r = 0; g = x; b = c; break;
        case 4: r = x; g = 0; b = c; break;
        default: r = c; g = 0; b = x; break;
    }
    return cv::Vec3b(static_cast<unsigned char>((b + m) * 255),
                     static_cast<unsigned char>((g + m) * 255),
                     static_cast<unsigned char>((r + m) * 255));
}

static cv::Mat draw_frame(const std::vector<unsigned char>& occ,
                          const std::vector<float>& pos,
                          const std::vector<float>& goals,
                          int n_arrived, int step, float ms) {
    int px = 3;
    int W = GRID * px;
    int H = GRID * px;
    cv::Mat img(H, W + 240, CV_8UC3, cv::Scalar(20, 20, 26));
    for (int y = 0; y < GRID; y++) {
        for (int x = 0; x < GRID; x++) {
            if (occ[y * GRID + x] != 0u) {
                cv::rectangle(img,
                              cv::Rect(x * px, (GRID - 1 - y) * px, px, px),
                              cv::Scalar(60, 60, 70), cv::FILLED);
            }
        }
    }
    for (int r = 0; r < N_ROBOTS; r++) {
        float x = pos[2 * r + 0], y = pos[2 * r + 1];
        int cx = static_cast<int>(x / RES);
        int cy = static_cast<int>(y / RES);
        cv::Vec3b col = hsv_to_bgr(static_cast<float>(r) / N_ROBOTS, 0.85f, 0.95f);
        cv::circle(img, cv::Point(cx * px + px / 2, (GRID - 1 - cy) * px + px / 2),
                   3, cv::Scalar(col[0], col[1], col[2]), cv::FILLED);
    }
    for (int r = 0; r < N_ROBOTS; r += 4) {
        float gx = goals[2 * r + 0], gy = goals[2 * r + 1];
        int cx = static_cast<int>(gx / RES);
        int cy = static_cast<int>(gy / RES);
        cv::drawMarker(img, cv::Point(cx * px + px / 2, (GRID - 1 - cy) * px + px / 2),
                       cv::Scalar(220, 220, 220), cv::MARKER_TILTED_CROSS, 5, 1);
    }
    char buf[128];
    int x0 = W + 12;
    cv::putText(img, "GPU multi-robot planner", cv::Point(x0, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(240, 240, 240), 1);
    std::snprintf(buf, sizeof(buf), "robots:     %d", N_ROBOTS);
    cv::putText(img, buf, cv::Point(x0, 60), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(220, 220, 220), 1);
    std::snprintf(buf, sizeof(buf), "grid:       %d x %d", GRID, GRID);
    cv::putText(img, buf, cv::Point(x0, 84), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(220, 220, 220), 1);
    std::snprintf(buf, sizeof(buf), "fields built once on GPU");
    cv::putText(img, buf, cv::Point(x0, 108), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(180, 180, 180), 1);
    std::snprintf(buf, sizeof(buf), "step:       %d", step);
    cv::putText(img, buf, cv::Point(x0, 144), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(220, 220, 220), 1);
    std::snprintf(buf, sizeof(buf), "arrived:    %d / %d", n_arrived, N_ROBOTS);
    cv::putText(img, buf, cv::Point(x0, 168), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(220, 220, 220), 1);
    std::snprintf(buf, sizeof(buf), "step time:  %.2f ms", ms);
    cv::putText(img, buf, cv::Point(x0, 192), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(220, 220, 220), 1);
    return img;
}

static void convert_avi_to_gif(const std::string& avi, const std::string& gif, int fps) {
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -i %s -vf \"fps=%d,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle\" %s 2>/dev/null",
                  avi.c_str(), fps, gif.c_str());
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg failed (%d)\n", rc);
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<unsigned char> occ;
    build_occupancy(occ);

    std::mt19937 rng(11);
    auto rand_free = [&]() {
        std::uniform_int_distribution<int> ux(2, GRID - 3);
        std::uniform_int_distribution<int> uy(2, GRID - 3);
        for (int t = 0; t < 1000; t++) {
            int x = ux(rng), y = uy(rng);
            if (occ[y * GRID + x] == 0u) return cv::Point(x, y);
        }
        return cv::Point(2, 2);
    };

    std::vector<float> starts(2 * N_ROBOTS), goals(2 * N_ROBOTS);
    std::vector<int> goal_cells(N_ROBOTS);
    for (int r = 0; r < N_ROBOTS; r++) {
        auto s = rand_free();
        auto g = rand_free();
        for (int t = 0; t < 80; t++) {
            int dx = s.x - g.x, dy = s.y - g.y;
            if (dx * dx + dy * dy > 60 * 60) break;
            g = rand_free();
        }
        starts[2 * r + 0] = (s.x + 0.5f) * RES;
        starts[2 * r + 1] = (s.y + 0.5f) * RES;
        goals[2 * r + 0]  = (g.x + 0.5f) * RES;
        goals[2 * r + 1]  = (g.y + 0.5f) * RES;
        goal_cells[r] = g.y * GRID + g.x;
    }

    unsigned char* d_occ = nullptr;
    int* d_goal_cells = nullptr;
    float* d_dist_a = nullptr;
    float* d_dist_b = nullptr;
    float* d_pos_a = nullptr;
    float* d_pos_b = nullptr;
    unsigned char* d_arrived = nullptr;
    size_t dist_floats = static_cast<size_t>(N_ROBOTS) * GRID * GRID;
    CUDA_CHECK(cudaMalloc(&d_occ, GRID * GRID));
    CUDA_CHECK(cudaMalloc(&d_goal_cells, N_ROBOTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist_a, dist_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist_b, dist_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_a, 2 * N_ROBOTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_b, 2 * N_ROBOTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_arrived, N_ROBOTS));
    CUDA_CHECK(cudaMemset(d_arrived, 0, N_ROBOTS));

    CUDA_CHECK(cudaMemcpy(d_occ, occ.data(), GRID * GRID, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_goal_cells, goal_cells.data(), N_ROBOTS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_a, starts.data(), 2 * N_ROBOTS * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    dim3 blockDim(8, 8);
    dim3 gridDim((GRID + 7) / 8, N_ROBOTS, (GRID + 7) / 8);
    init_dist_kernel<<<gridDim, blockDim>>>(N_ROBOTS, GRID, d_occ, d_goal_cells, d_dist_a);
    for (int it = 0; it < BF_ITERS; it++) {
        if (it % 2 == 0)
            bf_relax_kernel<<<gridDim, blockDim>>>(N_ROBOTS, GRID, d_occ, d_dist_a, d_dist_b);
        else
            bf_relax_kernel<<<gridDim, blockDim>>>(N_ROBOTS, GRID, d_occ, d_dist_b, d_dist_a);
    }
    float* d_dist_final = (BF_ITERS % 2 == 0) ? d_dist_a : d_dist_b;
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_fields = 0.0f; cudaEventElapsedTime(&ms_fields, t0, t1);
    std::printf("Distance fields built: %d robots x %d x %d cells, %d sweeps -> %.2f ms\n",
                N_ROBOTS, GRID, GRID, BF_ITERS, ms_fields);

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_multi_robot_planner.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          15, cv::Size(GRID * 3 + 240, GRID * 3));

    std::vector<float> pos_h(2 * N_ROBOTS);
    std::vector<unsigned char> arr_h(N_ROBOTS, 0);
    pos_h = starts;

    float ms_step_total = 0.0f;
    int frame_count = 0;
    for (int step = 0; step < SIM_STEPS; step++) {
        cudaEventRecord(t0);
        int blk = 256;
        int blocks = (N_ROBOTS + blk - 1) / blk;
        if (step % 2 == 0)
            simulate_step_kernel<<<blocks, blk>>>(N_ROBOTS, GRID, RES,
                                                   d_occ, d_dist_final,
                                                   d_pos_a, d_arrived, d_pos_b);
        else
            simulate_step_kernel<<<blocks, blk>>>(N_ROBOTS, GRID, RES,
                                                   d_occ, d_dist_final,
                                                   d_pos_b, d_arrived, d_pos_a);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms = 0.0f; cudaEventElapsedTime(&ms, t0, t1);
        ms_step_total += ms;

        float* d_pos_now = (step % 2 == 0) ? d_pos_b : d_pos_a;
        CUDA_CHECK(cudaMemcpy(pos_h.data(), d_pos_now, 2 * N_ROBOTS * sizeof(float),
                              cudaMemcpyDeviceToHost));
        int n_arrived = 0;
        for (int r = 0; r < N_ROBOTS; r++) {
            float dx = pos_h[2 * r + 0] - goals[2 * r + 0];
            float dy = pos_h[2 * r + 1] - goals[2 * r + 1];
            if (dx * dx + dy * dy < ARRIVAL_DIST * ARRIVAL_DIST) arr_h[r] = 1u;
            if (arr_h[r]) n_arrived++;
        }
        CUDA_CHECK(cudaMemcpy(d_arrived, arr_h.data(), N_ROBOTS, cudaMemcpyHostToDevice));
        cv::Mat frame = draw_frame(occ, pos_h, goals, n_arrived, step + 1, ms);
        video.write(frame);
        frame_count++;
        if (n_arrived == N_ROBOTS) break;
    }
    video.release();
    int final_arr = 0; for (auto a : arr_h) if (a) final_arr++;
    std::printf("Sim done.  %d steps, %d / %d arrived  (avg %.2f ms / step)\n",
                frame_count, final_arr, N_ROBOTS, ms_step_total / std::max(1, frame_count));
    convert_avi_to_gif("gif/gpu_multi_robot_planner.avi",
                       "gif/gpu_multi_robot_planner.gif", 12);
    std::printf("GIF saved to gif/gpu_multi_robot_planner.gif\n");

    cudaFree(d_occ); cudaFree(d_goal_cells);
    cudaFree(d_dist_a); cudaFree(d_dist_b);
    cudaFree(d_pos_a); cudaFree(d_pos_b);
    cudaFree(d_arrived);
    return 0;
}
