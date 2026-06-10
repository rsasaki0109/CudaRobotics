/*************************************************************************
    ESDF-MPPI
    - Builds a 2D occupancy grid for a static scene
    - Computes a Euclidean Signed Distance Field with Jump Flooding (JFA)
      on the GPU
    - MPPI rollout uses bilinear ESDF lookup as the obstacle clearance
      cost term (cheap, exact, no MLP training required)
    Output: gif/esdf_mppi.gif

    Contrast with sdf_mppi.cu (Neural-SDF MPPI):
      - Neural SDF: MLP forward pass per cost evaluation, requires offline
        training, smooth gradients but approximate.
      - ESDF (this file): single bilinear lookup per cost evaluation,
        no training, numerically exact on the grid.
 ************************************************************************/

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cuda_check.cuh"

using namespace std;

// -------------------------------------------------------------------------
// Scene / ESDF grid
// -------------------------------------------------------------------------
constexpr float WORLD_W = 40.0f;
constexpr float WORLD_H = 40.0f;
constexpr int   GRID    = 400;
constexpr float RES     = WORLD_W / GRID;
constexpr float MAX_DIST = 20.0f;

// -------------------------------------------------------------------------
// MPPI hyperparameters
// -------------------------------------------------------------------------
constexpr int   K_SAMPLES   = 4096;
constexpr int   T_HORIZON   = 30;
constexpr int   MAX_STEPS   = 300;
constexpr float DT          = 0.16f;
constexpr float MAX_SPEED   = 1.6f;
constexpr float LAMBDA      = 2.0f;
constexpr float SIGMA       = 0.55f;
constexpr float CLEARANCE   = 0.8f;
constexpr int   ITERS_PER_STEP = 4;

// Cost weights
constexpr float W_GOAL      = 1.6f;
constexpr float W_CTRL      = 0.15f;
constexpr float W_OBS       = 4.0f;
constexpr float COLLIDE_PENALTY = 150.0f;

// -------------------------------------------------------------------------
// Scene builders
// -------------------------------------------------------------------------
struct Disk { float cx, cy, r; };

static void fill_rect_world(std::vector<unsigned char>& occ, int W, int H, float res,
                            float x0, float y0, float x1, float y1) {
    int gx0 = std::max(0, static_cast<int>(std::floor(x0 / res)));
    int gy0 = std::max(0, static_cast<int>(std::floor(y0 / res)));
    int gx1 = std::min(W - 1, static_cast<int>(std::ceil(x1 / res)));
    int gy1 = std::min(H - 1, static_cast<int>(std::ceil(y1 / res)));
    for (int gy = gy0; gy <= gy1; gy++)
        for (int gx = gx0; gx <= gx1; gx++) occ[gy * W + gx] = 1u;
}

static void stamp_disk(std::vector<unsigned char>& occ, int W, int H, float res,
                       const Disk& d) {
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

static void build_scene(std::vector<unsigned char>& occ) {
    occ.assign(GRID * GRID, 0u);
    fill_rect_world(occ, GRID, GRID, RES, 0.0f, 0.0f, WORLD_W, 0.4f);
    fill_rect_world(occ, GRID, GRID, RES, 0.0f, WORLD_H - 0.4f, WORLD_W, WORLD_H);
    fill_rect_world(occ, GRID, GRID, RES, 0.0f, 0.0f, 0.4f, WORLD_H);
    fill_rect_world(occ, GRID, GRID, RES, WORLD_W - 0.4f, 0.0f, WORLD_W, WORLD_H);

    // Two interior corridor walls forming an S-curve
    fill_rect_world(occ, GRID, GRID, RES, 10.0f,  8.0f, 11.0f, 28.0f);
    fill_rect_world(occ, GRID, GRID, RES, 23.0f, 12.0f, 24.0f, 32.0f);

    Disk disks[] = {
        { 6.0f, 33.0f, 1.4f}, {14.0f, 6.0f, 1.2f}, {18.0f, 22.0f, 1.6f},
        {28.0f, 10.0f, 1.5f}, {32.0f, 22.0f, 1.3f}, {34.0f, 34.0f, 1.8f},
        {20.0f, 36.0f, 1.0f}, {30.0f, 30.0f, 1.1f},
    };
    for (const auto& d : disks) stamp_disk(occ, GRID, GRID, RES, d);
}

// -------------------------------------------------------------------------
// Jump Flooding kernels (reused from comparison_esdf)
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
                                   const unsigned char* __restrict__ occ,
                                   float* __restrict__ dist,
                                   int W, int H, float res) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    int s = seed[idx];
    float d_signed;
    if (s < 0) { d_signed = MAX_DIST; }
    else {
        int sx = s % W, sy = s / W;
        int dx = x - sx, dy = y - sy;
        d_signed = sqrtf(static_cast<float>(dx * dx + dy * dy)) * res;
    }
    // Inside occupied cells get negative distance (use cell-center offset)
    if (occ[idx]) d_signed = -d_signed;
    dist[idx] = d_signed;
}

// -------------------------------------------------------------------------
// MPPI kernels with ESDF cost
// -------------------------------------------------------------------------
__device__ __forceinline__ float bilinear_esdf(const float* __restrict__ esdf,
                                               float wx, float wy) {
    float fx = wx / RES - 0.5f;
    float fy = wy / RES - 0.5f;
    int x0 = static_cast<int>(floorf(fx));
    int y0 = static_cast<int>(floorf(fy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    if (x0 < 0) x0 = 0; if (x0 > GRID - 1) x0 = GRID - 1;
    if (y0 < 0) y0 = 0; if (y0 > GRID - 1) y0 = GRID - 1;
    if (x1 < 0) x1 = 0; if (x1 > GRID - 1) x1 = GRID - 1;
    if (y1 < 0) y1 = 0; if (y1 > GRID - 1) y1 = GRID - 1;
    float ax = fx - floorf(fx);
    float ay = fy - floorf(fy);
    float d00 = esdf[y0 * GRID + x0];
    float d10 = esdf[y0 * GRID + x1];
    float d01 = esdf[y1 * GRID + x0];
    float d11 = esdf[y1 * GRID + x1];
    float d0 = d00 * (1.0f - ax) + d10 * ax;
    float d1 = d01 * (1.0f - ax) + d11 * ax;
    return d0 * (1.0f - ay) + d1 * ay;
}

__global__ void init_rng(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void rollout_kernel(
    float sx, float sy, float gx, float gy,
    const float* __restrict__ d_nominal,
    const float* __restrict__ d_esdf,
    float* __restrict__ d_costs,
    float* __restrict__ d_perturbed,
    curandState* __restrict__ d_rng)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K_SAMPLES) return;
    curandState rng = d_rng[k];
    float x = sx, y = sy;
    float total = 0.0f;
    for (int t = 0; t < T_HORIZON; t++) {
        float ux = d_nominal[t * 2 + 0] + SIGMA * curand_normal(&rng);
        float uy = d_nominal[t * 2 + 1] + SIGMA * curand_normal(&rng);
        ux = fminf(fmaxf(ux, -MAX_SPEED), MAX_SPEED);
        uy = fminf(fmaxf(uy, -MAX_SPEED), MAX_SPEED);
        d_perturbed[k * T_HORIZON * 2 + t * 2 + 0] = ux;
        d_perturbed[k * T_HORIZON * 2 + t * 2 + 1] = uy;
        x += ux * DT;
        y += uy * DT;
        float dx = x - gx;
        float dy = y - gy;
        total += W_GOAL * sqrtf(dx * dx + dy * dy + 1.0e-4f);
        total += W_CTRL * (ux * ux + uy * uy);
        float esdf = bilinear_esdf(d_esdf, x, y);
        if (esdf < CLEARANCE) {
            float margin = fmaxf(esdf, 0.05f);
            float inv = 1.0f / margin - 1.0f / CLEARANCE;
            total += W_OBS * inv * inv;
        }
        if (esdf < 0.0f) total += COLLIDE_PENALTY;
        if (x < 0.0f || x > WORLD_W || y < 0.0f || y > WORLD_H) total += 100.0f;
    }
    float dx = x - gx, dy = y - gy;
    total += 10.0f * sqrtf(dx * dx + dy * dy + 1.0e-4f);
    d_costs[k] = total;
    d_rng[k] = rng;
}

__global__ void compute_weights_kernel(const float* __restrict__ d_costs,
                                       float* __restrict__ d_weights) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float min_cost = FLT_MAX;
    for (int k = 0; k < K_SAMPLES; k++) min_cost = fminf(min_cost, d_costs[k]);
    float sum_w = 0.0f;
    for (int k = 0; k < K_SAMPLES; k++) {
        float w = expf(-(d_costs[k] - min_cost) / LAMBDA);
        d_weights[k] = w;
        sum_w += w;
    }
    if (sum_w > 0.0f) for (int k = 0; k < K_SAMPLES; k++) d_weights[k] /= sum_w;
}

__global__ void update_controls_kernel(float* __restrict__ d_nominal,
                                       const float* __restrict__ d_perturbed,
                                       const float* __restrict__ d_weights) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T_HORIZON) return;
    float ux = 0.0f, uy = 0.0f;
    for (int k = 0; k < K_SAMPLES; k++) {
        float w = d_weights[k];
        ux += w * d_perturbed[k * T_HORIZON * 2 + t * 2 + 0];
        uy += w * d_perturbed[k * T_HORIZON * 2 + t * 2 + 1];
    }
    d_nominal[t * 2 + 0] = ux;
    d_nominal[t * 2 + 1] = uy;
}

// -------------------------------------------------------------------------
// Rendering
// -------------------------------------------------------------------------
static cv::Mat render_esdf_heatmap(const std::vector<float>& esdf) {
    cv::Mat img(GRID, GRID, CV_8UC3);
    for (int gy = 0; gy < GRID; gy++) {
        for (int gx = 0; gx < GRID; gx++) {
            float d = esdf[gy * GRID + gx];
            cv::Vec3b& px = img.at<cv::Vec3b>(GRID - 1 - gy, gx);
            if (d < 0.0f) { px = cv::Vec3b(40, 40, 40); continue; }
            float t = std::min(d / 6.0f, 1.0f);
            int r = static_cast<int>((1.0f - t) * 180.0f + 40.0f);
            int g = static_cast<int>(t * 200.0f + 30.0f);
            int b = static_cast<int>(80.0f + (1.0f - t) * 60.0f);
            px = cv::Vec3b(b, g, r);
        }
    }
    int W = 700, H = 700;
    cv::Mat out;
    cv::resize(img, out, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);
    return out;
}

static void draw_point(cv::Mat& img, float wx, float wy, cv::Scalar color, int radius = 6) {
    float sx = wx / WORLD_W * img.cols;
    float sy = (1.0f - wy / WORLD_H) * img.rows;
    cv::circle(img, cv::Point(static_cast<int>(sx), static_cast<int>(sy)),
               radius, color, cv::FILLED);
}

static void draw_path(cv::Mat& img, const std::vector<cv::Point2f>& path, cv::Scalar color) {
    for (size_t i = 1; i < path.size(); i++) {
        float ax = path[i - 1].x / WORLD_W * img.cols;
        float ay = (1.0f - path[i - 1].y / WORLD_H) * img.rows;
        float bx = path[i].x / WORLD_W * img.cols;
        float by = (1.0f - path[i].y / WORLD_H) * img.rows;
        cv::line(img,
                 cv::Point(static_cast<int>(ax), static_cast<int>(ay)),
                 cv::Point(static_cast<int>(bx), static_cast<int>(by)),
                 color, 2);
    }
}

// -------------------------------------------------------------------------
// AVI -> GIF via system ffmpeg
// -------------------------------------------------------------------------
static void convert_avi_to_gif(const char* avi_path, const char* gif_path, int fps) {
    char cmd[512];
    std::snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -i %s -vf 'fps=%d,scale=700:-1:flags=lanczos' -loop 0 %s "
        "> /dev/null 2>&1",
        avi_path, fps, gif_path);
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg conversion returned %d\n", rc);
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------
int main() {
    // 1. Build the scene
    std::vector<unsigned char> occ;
    build_scene(occ);

    // 2. Build ESDF via JFA
    unsigned char* d_occ = nullptr;
    int*           d_seed_a = nullptr;
    int*           d_seed_b = nullptr;
    float*         d_esdf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_occ,    GRID * GRID * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_seed_a, GRID * GRID * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seed_b, GRID * GRID * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_esdf,   GRID * GRID * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_occ, occ.data(), occ.size(), cudaMemcpyHostToDevice));

    dim3 blk2d(16, 16);
    dim3 grd2d((GRID + 15) / 16, (GRID + 15) / 16);

    auto t0 = std::chrono::high_resolution_clock::now();
    jfa_init_kernel<<<grd2d, blk2d>>>(d_occ, d_seed_a, GRID, GRID);
    int* in_ptr = d_seed_a;
    int* out_ptr = d_seed_b;
    int k = GRID / 2;
    while (k >= 1) {
        jfa_step_kernel<<<grd2d, blk2d>>>(in_ptr, out_ptr, GRID, GRID, k);
        std::swap(in_ptr, out_ptr);
        k /= 2;
    }
    jfa_to_dist_kernel<<<grd2d, blk2d>>>(in_ptr, d_occ, d_esdf, GRID, GRID, RES);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double esdf_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("ESDF build (JFA, %dx%d): %.3f ms\n", GRID, GRID, esdf_ms);

    std::vector<float> h_esdf(GRID * GRID);
    CUDA_CHECK(cudaMemcpy(h_esdf.data(), d_esdf, h_esdf.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // 3. MPPI setup
    float* d_nominal = nullptr;
    float* d_costs = nullptr;
    float* d_perturbed = nullptr;
    float* d_weights = nullptr;
    curandState* d_rng = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nominal,   T_HORIZON * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_costs,     K_SAMPLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_perturbed, K_SAMPLES * T_HORIZON * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights,   K_SAMPLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rng,       K_SAMPLES * sizeof(curandState)));
    CUDA_CHECK(cudaMemset(d_nominal, 0, T_HORIZON * 2 * sizeof(float)));
    int threads = 256;
    int blocks = (K_SAMPLES + threads - 1) / threads;
    init_rng<<<blocks, threads>>>(d_rng, K_SAMPLES, 2026ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    cv::Point2f state(3.0f, 3.0f);
    cv::Point2f goal(36.0f, 36.0f);
    std::vector<cv::Point2f> path = { state };
    std::vector<float> h_nominal(T_HORIZON * 2, 0.0f);

    cv::Mat background = render_esdf_heatmap(h_esdf);

    const char* AVI_PATH = "gif/esdf_mppi.avi";
    const char* GIF_PATH = "gif/esdf_mppi.gif";
    cv::VideoWriter video(
        AVI_PATH, cv::VideoWriter::fourcc('X','V','I','D'), 15,
        cv::Size(background.cols, background.rows));
    if (!video.isOpened()) {
        std::fprintf(stderr, "Failed to open %s\n", AVI_PATH);
        return 1;
    }

    double rollout_ms_total = 0.0;
    int rollout_calls = 0;

    for (int step = 0; step < MAX_STEPS; step++) {
        for (int iter = 0; iter < ITERS_PER_STEP; iter++) {
            auto rt0 = std::chrono::high_resolution_clock::now();
            rollout_kernel<<<blocks, threads>>>(
                state.x, state.y, goal.x, goal.y,
                d_nominal, d_esdf, d_costs, d_perturbed, d_rng);
            compute_weights_kernel<<<1, 1>>>(d_costs, d_weights);
            update_controls_kernel<<<1, T_HORIZON>>>(d_nominal, d_perturbed, d_weights);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto rt1 = std::chrono::high_resolution_clock::now();
            rollout_ms_total += std::chrono::duration<double, std::milli>(rt1 - rt0).count();
            rollout_calls++;
        }

        CUDA_CHECK(cudaMemcpy(h_nominal.data(), d_nominal,
                              h_nominal.size() * sizeof(float), cudaMemcpyDeviceToHost));
        float ux = std::min(std::max(h_nominal[0], -MAX_SPEED), MAX_SPEED);
        float uy = std::min(std::max(h_nominal[1], -MAX_SPEED), MAX_SPEED);
        state.x += ux * DT;
        state.y += uy * DT;
        state.x = std::min(std::max(state.x, 0.0f), WORLD_W);
        state.y = std::min(std::max(state.y, 0.0f), WORLD_H);
        path.push_back(state);

        // shift the nominal control sequence
        for (int t = 0; t < T_HORIZON - 1; t++) {
            h_nominal[t * 2 + 0] = h_nominal[(t + 1) * 2 + 0];
            h_nominal[t * 2 + 1] = h_nominal[(t + 1) * 2 + 1];
        }
        h_nominal[(T_HORIZON - 1) * 2 + 0] = 0.0f;
        h_nominal[(T_HORIZON - 1) * 2 + 1] = 0.0f;
        CUDA_CHECK(cudaMemcpy(d_nominal, h_nominal.data(),
                              h_nominal.size() * sizeof(float), cudaMemcpyHostToDevice));

        cv::Mat frame = background.clone();
        draw_path(frame, path, cv::Scalar(255, 255, 255));
        draw_point(frame, 3.0f, 3.0f, cv::Scalar(255, 100, 100), 7);
        draw_point(frame, goal.x, goal.y, cv::Scalar(100, 255, 100), 7);
        draw_point(frame, state.x, state.y, cv::Scalar(0, 255, 255), 6);

        char buf[128];
        float dist = std::hypot(state.x - goal.x, state.y - goal.y);
        std::snprintf(buf, sizeof(buf), "step=%d dist=%.2fm K=%d T=%d",
                      step, dist, K_SAMPLES, T_HORIZON);
        cv::putText(frame, buf, cv::Point(10, frame.rows - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
        cv::putText(frame, "ESDF-MPPI (Jump Flooding + bilinear lookup cost)",
                    cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255, 255, 255), 1);
        video.write(frame);

        if (dist < 0.5f) {
            std::printf("Goal reached at step %d (dist=%.3f m)\n", step, dist);
            break;
        }
    }

    video.release();
    std::printf("Avg rollout time (K=%d, T=%d): %.3f ms / iter (%d iters total)\n",
                K_SAMPLES, T_HORIZON, rollout_ms_total / rollout_calls, rollout_calls);

    CUDA_CHECK(cudaFree(d_occ));
    CUDA_CHECK(cudaFree(d_seed_a));
    CUDA_CHECK(cudaFree(d_seed_b));
    CUDA_CHECK(cudaFree(d_esdf));
    CUDA_CHECK(cudaFree(d_nominal));
    CUDA_CHECK(cudaFree(d_costs));
    CUDA_CHECK(cudaFree(d_perturbed));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_rng));

    std::printf("Video saved to %s\n", AVI_PATH);
    convert_avi_to_gif(AVI_PATH, GIF_PATH, 15);
    std::printf("GIF saved to %s\n", GIF_PATH);
    return 0;
}
