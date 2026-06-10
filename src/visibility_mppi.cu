/*************************************************************************
    Visibility-aware Diff-MPPI

    Standard MPPI minimizes (goal distance + collision penalty + control
    magnitude). It is "blind" — the planner is happy to cut close behind
    a corner that hides a landmark or chase a goal through a pinhole that
    will leave the robot without sensor coverage at the next step.

    Visibility-aware MPPI adds a single cost term that REWARDS poses
    from which the robot's 360 deg 2D LiDAR can see many landmarks
    (line-of-sight unblocked, within sensor range). A visibility field
    V(x, y) is precomputed once on the GPU as the count of visible
    landmarks from each cell. The rollout uses a bilinear lookup of V
    in the per-sample cost term, so the cost remains autodiff-friendly
    (smooth in pose).

    Two MPPI controllers run side-by-side on the same K=4096 samples,
    T=30 horizon, ESDF clearance cost:

      baseline:           cost = goal + control + ESDF clearance
      visibility-aware:   cost = baseline - W_VIS * V(x, y)

    Output: gif/visibility_mppi.gif (two trajectories, visibility heatmap
            background, side-by-side); CSV of visible-landmark count
            along each trajectory printed at end.
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
#include <curand_kernel.h>
#include "cuda_check.cuh"

// -------------------------------------------------------------------------
// World / planner parameters
// -------------------------------------------------------------------------
constexpr float WORLD_X = 30.0f;
constexpr float WORLD_Y = 30.0f;
constexpr int   GRID = 300;
constexpr float RES = WORLD_X / GRID;
constexpr float MAX_DIST = 12.0f;
constexpr int   N_LM = 12;
constexpr float SENSOR_RANGE = 9.0f;

constexpr int   K_SAMPLES = 4096;
constexpr int   T_HORIZON = 30;
constexpr int   MAX_STEPS = 220;
constexpr float DT = 0.16f;
constexpr float MAX_SPEED = 1.5f;
constexpr float LAMBDA = 2.0f;
constexpr float SIGMA = 0.55f;
constexpr float CLEARANCE = 0.7f;
constexpr int   ITERS_PER_STEP = 4;

constexpr float W_GOAL = 1.4f;
constexpr float W_CTRL = 0.12f;
constexpr float W_OBS = 4.0f;
constexpr float COLLIDE_PENALTY = 150.0f;
constexpr float W_VIS = 1.2f;

constexpr int PANEL_W = 540;
constexpr int PANEL_H = 540;

struct Disk { float cx, cy, r; };

// -------------------------------------------------------------------------
// Scene
// -------------------------------------------------------------------------
static void fill_rect(std::vector<unsigned char>& occ, float x0, float y0,
                      float x1, float y1) {
    int gx0 = std::max(0, static_cast<int>(x0 / RES));
    int gy0 = std::max(0, static_cast<int>(y0 / RES));
    int gx1 = std::min(GRID - 1, static_cast<int>(x1 / RES));
    int gy1 = std::min(GRID - 1, static_cast<int>(y1 / RES));
    for (int gy = gy0; gy <= gy1; gy++)
        for (int gx = gx0; gx <= gx1; gx++) occ[gy * GRID + gx] = 1u;
}

static void stamp_disk(std::vector<unsigned char>& occ, const Disk& d) {
    int gx0 = std::max(0, static_cast<int>((d.cx - d.r) / RES));
    int gy0 = std::max(0, static_cast<int>((d.cy - d.r) / RES));
    int gx1 = std::min(GRID - 1, static_cast<int>((d.cx + d.r) / RES));
    int gy1 = std::min(GRID - 1, static_cast<int>((d.cy + d.r) / RES));
    float r2 = d.r * d.r;
    for (int gy = gy0; gy <= gy1; gy++) {
        float wy = (gy + 0.5f) * RES;
        for (int gx = gx0; gx <= gx1; gx++) {
            float wx = (gx + 0.5f) * RES;
            float dx = wx - d.cx, dy = wy - d.cy;
            if (dx * dx + dy * dy <= r2) occ[gy * GRID + gx] = 1u;
        }
    }
}

static void build_scene(std::vector<unsigned char>& occ,
                        std::vector<float>& lm_x, std::vector<float>& lm_y) {
    occ.assign(GRID * GRID, 0u);
    fill_rect(occ, 0.0f, 0.0f, WORLD_X, 0.4f);
    fill_rect(occ, 0.0f, WORLD_Y - 0.4f, WORLD_X, WORLD_Y);
    fill_rect(occ, 0.0f, 0.0f, 0.4f, WORLD_Y);
    fill_rect(occ, WORLD_X - 0.4f, 0.0f, WORLD_X, WORLD_Y);
    // wall pieces that create occlusions
    fill_rect(occ,  8.0f,  5.0f,  9.0f, 14.0f);
    fill_rect(occ, 16.0f, 12.0f, 17.0f, 24.0f);
    fill_rect(occ, 21.0f,  4.0f, 22.0f, 14.0f);
    Disk disks[] = {
        {12.0f, 18.0f, 1.6f}, {22.0f, 22.0f, 1.4f},
        { 5.0f, 22.0f, 1.0f}, {26.0f, 10.0f, 1.5f},
    };
    for (const auto& d : disks) stamp_disk(occ, d);
    lm_x = { 3.5f, 14.0f, 26.0f, 18.5f, 11.0f, 5.5f, 22.5f,  9.0f,
             14.0f, 27.0f, 4.0f, 18.0f };
    lm_y = { 3.5f, 25.5f,  4.0f, 18.0f,  8.0f, 13.0f, 13.5f, 23.0f,
              4.0f,  20.0f, 28.0f, 27.0f };
}

// -------------------------------------------------------------------------
// JFA ESDF kernels
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
    float d;
    if (s < 0) { d = MAX_DIST; }
    else {
        int sx = s % W, sy = s / W;
        int dx = x - sx, dy = y - sy;
        d = sqrtf(static_cast<float>(dx * dx + dy * dy)) * res;
    }
    if (occ[idx]) d = -d;
    dist[idx] = d;
}

// -------------------------------------------------------------------------
// Visibility field kernel
// -------------------------------------------------------------------------
__device__ bool los_clear(const unsigned char* occ, float ax, float ay,
                          float bx, float by) {
    float dx = bx - ax, dy = by - ay;
    float len = sqrtf(dx * dx + dy * dy);
    if (len < 1e-3f) return true;
    float ux = dx / len, uy = dy / len;
    // DDA
    float fx = ax / RES, fy = ay / RES;
    int gx = (int)floorf(fx), gy = (int)floorf(fy);
    int step_x = (ux > 0.0f) ? 1 : -1;
    int step_y = (uy > 0.0f) ? 1 : -1;
    float inv_dx = (fabsf(ux) > 1e-7f) ? 1.0f / fabsf(ux) : 1e30f;
    float inv_dy = (fabsf(uy) > 1e-7f) ? 1.0f / fabsf(uy) : 1e30f;
    float t_max_x = (ux > 0.0f) ? (gx + 1 - fx) * RES * inv_dx
                                : (fx - gx) * RES * inv_dx;
    float t_max_y = (uy > 0.0f) ? (gy + 1 - fy) * RES * inv_dy
                                : (fy - gy) * RES * inv_dy;
    float dt_x = RES * inv_dx, dt_y = RES * inv_dy;
    int max_iter = GRID + GRID;
    for (int it = 0; it < max_iter; it++) {
        if (gx < 0 || gx >= GRID || gy < 0 || gy >= GRID) return true;
        if (occ[gy * GRID + gx] != 0u) return false;
        float t_next = fminf(t_max_x, t_max_y);
        if (t_next >= len) return true;
        if (t_max_x < t_max_y) { gx += step_x; t_max_x += dt_x; }
        else                    { gy += step_y; t_max_y += dt_y; }
    }
    return true;
}

__global__ void visibility_field_kernel(const unsigned char* __restrict__ occ,
                                        const float* __restrict__ lm_x,
                                        const float* __restrict__ lm_y,
                                        int n_lm,
                                        float* __restrict__ vis) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= GRID || y >= GRID) return;
    int idx = y * GRID + x;
    if (occ[idx]) { vis[idx] = 0.0f; return; }
    float wx = (x + 0.5f) * RES;
    float wy = (y + 0.5f) * RES;
    int count = 0;
    for (int i = 0; i < n_lm; i++) {
        float dx = wx - lm_x[i], dy = wy - lm_y[i];
        float d2 = dx * dx + dy * dy;
        if (d2 > SENSOR_RANGE * SENSOR_RANGE) continue;
        if (los_clear(occ, wx, wy, lm_x[i], lm_y[i])) count++;
    }
    vis[idx] = (float)count;
}

// -------------------------------------------------------------------------
// MPPI kernels with optional visibility cost
// -------------------------------------------------------------------------
__device__ __forceinline__ float bilinear(const float* f, float wx, float wy) {
    float fx = wx / RES - 0.5f;
    float fy = wy / RES - 0.5f;
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    if (x0 < 0) x0 = 0; if (x0 > GRID - 1) x0 = GRID - 1;
    if (y0 < 0) y0 = 0; if (y0 > GRID - 1) y0 = GRID - 1;
    if (x1 < 0) x1 = 0; if (x1 > GRID - 1) x1 = GRID - 1;
    if (y1 < 0) y1 = 0; if (y1 > GRID - 1) y1 = GRID - 1;
    float ax = fx - floorf(fx);
    float ay = fy - floorf(fy);
    float v00 = f[y0 * GRID + x0];
    float v10 = f[y0 * GRID + x1];
    float v01 = f[y1 * GRID + x0];
    float v11 = f[y1 * GRID + x1];
    float v0 = v00 * (1.0f - ax) + v10 * ax;
    float v1 = v01 * (1.0f - ax) + v11 * ax;
    return v0 * (1.0f - ay) + v1 * ay;
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
    const float* __restrict__ d_vis,
    float w_vis,
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
        float dx = x - gx, dy = y - gy;
        total += W_GOAL * sqrtf(dx * dx + dy * dy + 1.0e-4f);
        total += W_CTRL * (ux * ux + uy * uy);
        float esdf = bilinear(d_esdf, x, y);
        if (esdf < CLEARANCE) {
            float margin = fmaxf(esdf, 0.05f);
            float inv = 1.0f / margin - 1.0f / CLEARANCE;
            total += W_OBS * inv * inv;
        }
        if (esdf < 0.0f) total += COLLIDE_PENALTY;
        if (w_vis > 0.0f) {
            float v = bilinear(d_vis, x, y);
            total -= w_vis * v;
        }
        if (x < 0.0f || x > WORLD_X || y < 0.0f || y > WORLD_Y) total += 100.0f;
    }
    float dx = x - gx, dy = y - gy;
    total += 10.0f * sqrtf(dx * dx + dy * dy + 1.0e-4f);
    d_costs[k] = total;
    d_rng[k] = rng;
}

__global__ void compute_weights_kernel(const float* costs, float* weights) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float min_cost = FLT_MAX;
    for (int k = 0; k < K_SAMPLES; k++) min_cost = fminf(min_cost, costs[k]);
    float sum_w = 0.0f;
    for (int k = 0; k < K_SAMPLES; k++) {
        float w = expf(-(costs[k] - min_cost) / LAMBDA);
        weights[k] = w;
        sum_w += w;
    }
    if (sum_w > 0.0f) for (int k = 0; k < K_SAMPLES; k++) weights[k] /= sum_w;
}

__global__ void update_controls_kernel(float* nominal, const float* perturbed,
                                       const float* weights) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T_HORIZON) return;
    float ux = 0.0f, uy = 0.0f;
    for (int k = 0; k < K_SAMPLES; k++) {
        float w = weights[k];
        ux += w * perturbed[k * T_HORIZON * 2 + t * 2 + 0];
        uy += w * perturbed[k * T_HORIZON * 2 + t * 2 + 1];
    }
    nominal[t * 2 + 0] = ux;
    nominal[t * 2 + 1] = uy;
}

// -------------------------------------------------------------------------
// Rendering
// -------------------------------------------------------------------------
static cv::Mat render_panel(const std::vector<float>& vis,
                            const std::vector<unsigned char>& occ,
                            const std::vector<cv::Point2f>& path,
                            cv::Point2f goal, cv::Point2f cur,
                            const std::vector<float>& lm_x,
                            const std::vector<float>& lm_y,
                            const char* title, int visible_count) {
    cv::Mat img(GRID, GRID, CV_8UC3, cv::Scalar(20, 20, 20));
    float vmax = 0.0f;
    for (float v : vis) if (v > vmax) vmax = v;
    if (vmax < 1.0f) vmax = 1.0f;
    for (int gy = 0; gy < GRID; gy++) {
        for (int gx = 0; gx < GRID; gx++) {
            int idx = gy * GRID + gx;
            cv::Vec3b col;
            if (occ[idx]) col = cv::Vec3b(120, 60, 60);
            else {
                float t = vis[idx] / vmax;
                col = cv::Vec3b(static_cast<uchar>(60.0f + t * 40.0f),
                                static_cast<uchar>(60.0f + t * 180.0f),
                                static_cast<uchar>(60.0f + t * 30.0f));
            }
            img.at<cv::Vec3b>(GRID - 1 - gy, gx) = col;
        }
    }
    auto to_pt = [&](cv::Point2f p) {
        return cv::Point(static_cast<int>(p.x / WORLD_X * img.cols),
                         static_cast<int>((1.0f - p.y / WORLD_Y) * img.rows));
    };
    for (size_t i = 0; i < lm_x.size(); i++) {
        cv::Point2f lm(lm_x[i], lm_y[i]);
        cv::circle(img, to_pt(lm), 4, cv::Scalar(240, 240, 60), cv::FILLED);
    }
    for (size_t i = 1; i < path.size(); i++) {
        cv::line(img, to_pt(path[i - 1]), to_pt(path[i]),
                 cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
    cv::circle(img, to_pt(cv::Point2f(2.0f, 2.0f)), 5, cv::Scalar(120, 120, 255), 2);
    cv::circle(img, to_pt(goal), 5, cv::Scalar(120, 255, 120), 2);
    cv::circle(img, to_pt(cur), 5, cv::Scalar(0, 255, 255), cv::FILLED);
    cv::Mat out;
    cv::resize(img, out, cv::Size(PANEL_W, PANEL_H), 0, 0, cv::INTER_NEAREST);
    cv::rectangle(out, cv::Rect(0, 0, PANEL_W, 26), cv::Scalar(0, 0, 0), cv::FILLED);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s   visible LMs at current pose: %d",
                  title, visible_count);
    cv::putText(out, buf, cv::Point(10, 18), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return out;
}

static int count_visible_host(const std::vector<unsigned char>& occ,
                              float wx, float wy,
                              const std::vector<float>& lm_x,
                              const std::vector<float>& lm_y) {
    auto los = [&](float ax, float ay, float bx, float by) {
        float dx = bx - ax, dy = by - ay;
        float len = std::sqrt(dx * dx + dy * dy);
        if (len < 1e-3f) return true;
        int steps = static_cast<int>(len / (RES * 0.5f)) + 1;
        for (int s = 1; s < steps; s++) {
            float t = (float)s / steps;
            float x = ax + t * dx, y = ay + t * dy;
            int gx = (int)(x / RES), gy = (int)(y / RES);
            if (gx < 0 || gx >= GRID || gy < 0 || gy >= GRID) return true;
            if (occ[gy * GRID + gx]) return false;
        }
        return true;
    };
    int c = 0;
    for (size_t i = 0; i < lm_x.size(); i++) {
        float dx = wx - lm_x[i], dy = wy - lm_y[i];
        if (dx * dx + dy * dy > SENSOR_RANGE * SENSOR_RANGE) continue;
        if (los(wx, wy, lm_x[i], lm_y[i])) c++;
    }
    return c;
}

static void convert_avi_to_gif(const char* avi, const char* gif, int fps) {
    char cmd[512];
    std::snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -i %s -vf 'fps=%d,scale=1200:-1:flags=lanczos' -loop 0 %s "
        "> /dev/null 2>&1", avi, fps, gif);
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg conversion returned %d\n", rc);
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------
int main() {
    std::vector<unsigned char> occ;
    std::vector<float> lm_x, lm_y;
    build_scene(occ, lm_x, lm_y);
    int n_lm = static_cast<int>(lm_x.size());

    // Device buffers shared
    unsigned char* d_occ = nullptr;
    int *d_seed_a = nullptr, *d_seed_b = nullptr;
    float *d_esdf = nullptr, *d_vis = nullptr;
    float *d_lm_x = nullptr, *d_lm_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_occ,   occ.size()));
    CUDA_CHECK(cudaMalloc(&d_seed_a, GRID * GRID * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seed_b, GRID * GRID * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_esdf,  GRID * GRID * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vis,   GRID * GRID * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_x,  n_lm * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_y,  n_lm * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_occ, occ.data(), occ.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lm_x, lm_x.data(), n_lm * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lm_y, lm_y.data(), n_lm * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blk2d(16, 16);
    dim3 grd2d((GRID + 15) / 16, (GRID + 15) / 16);

    auto t0 = std::chrono::high_resolution_clock::now();
    jfa_init_kernel<<<grd2d, blk2d>>>(d_occ, d_seed_a, GRID, GRID);
    int* in_ptr = d_seed_a; int* out_ptr = d_seed_b;
    for (int k = GRID / 2; k >= 1; k /= 2) {
        jfa_step_kernel<<<grd2d, blk2d>>>(in_ptr, out_ptr, GRID, GRID, k);
        std::swap(in_ptr, out_ptr);
    }
    jfa_to_dist_kernel<<<grd2d, blk2d>>>(in_ptr, d_occ, d_esdf, GRID, GRID, RES);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    std::printf("ESDF build: %.2f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count());

    auto v0 = std::chrono::high_resolution_clock::now();
    visibility_field_kernel<<<grd2d, blk2d>>>(d_occ, d_lm_x, d_lm_y, n_lm, d_vis);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto v1 = std::chrono::high_resolution_clock::now();
    std::printf("Visibility field build: %.2f ms\n",
                std::chrono::duration<double, std::milli>(v1 - v0).count());

    std::vector<float> h_vis(GRID * GRID);
    CUDA_CHECK(cudaMemcpy(h_vis.data(), d_vis, h_vis.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Two parallel MPPI controllers
    auto alloc = [&]() {
        struct M { float *nominal, *costs, *perturbed, *weights;
                   curandState* rng; };
        M m;
        CUDA_CHECK(cudaMalloc(&m.nominal,   T_HORIZON * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m.costs,     K_SAMPLES * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m.perturbed, K_SAMPLES * T_HORIZON * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m.weights,   K_SAMPLES * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m.rng,       K_SAMPLES * sizeof(curandState)));
        CUDA_CHECK(cudaMemset(m.nominal, 0, T_HORIZON * 2 * sizeof(float)));
        return m;
    };
    auto m_base = alloc();
    auto m_vis  = alloc();
    int threads = 256;
    int blocks = (K_SAMPLES + threads - 1) / threads;
    init_rng<<<blocks, threads>>>(m_base.rng, K_SAMPLES, 1ULL);
    init_rng<<<blocks, threads>>>(m_vis.rng,  K_SAMPLES, 2ULL);

    cv::Point2f goal(27.0f, 27.0f);
    cv::Point2f base_state(2.0f, 2.0f), vis_state(2.0f, 2.0f);
    std::vector<cv::Point2f> base_path = { base_state }, vis_path = { vis_state };
    std::vector<float> base_nom(T_HORIZON * 2, 0.0f);
    std::vector<float> vis_nom(T_HORIZON * 2, 0.0f);

    cv::VideoWriter video("gif/visibility_mppi.avi",
                          cv::VideoWriter::fourcc('X','V','I','D'), 15,
                          cv::Size(PANEL_W * 2 + 4, PANEL_H + 30));

    int sum_base_vis = 0, sum_vis_vis = 0, vis_counted = 0;
    int base_done_step = -1, vis_done_step = -1;
    double total_iter_ms = 0.0; int counted_iters = 0;

    for (int step = 0; step < MAX_STEPS; step++) {
        bool base_done = base_done_step >= 0;
        bool vis_done = vis_done_step >= 0;
        auto run = [&](decltype(m_base)& m, std::vector<float>& nom,
                       cv::Point2f& state, float w_vis_arg) {
            for (int iter = 0; iter < ITERS_PER_STEP; iter++) {
                rollout_kernel<<<blocks, threads>>>(
                    state.x, state.y, goal.x, goal.y,
                    m.nominal, d_esdf, d_vis, w_vis_arg,
                    m.costs, m.perturbed, m.rng);
                compute_weights_kernel<<<1, 1>>>(m.costs, m.weights);
                update_controls_kernel<<<1, T_HORIZON>>>(m.nominal, m.perturbed, m.weights);
            }
            CUDA_CHECK(cudaMemcpy(nom.data(), m.nominal,
                                  nom.size() * sizeof(float), cudaMemcpyDeviceToHost));
            float ux = std::min(std::max(nom[0], -MAX_SPEED), MAX_SPEED);
            float uy = std::min(std::max(nom[1], -MAX_SPEED), MAX_SPEED);
            state.x = std::min(std::max(state.x + ux * DT, 0.0f), WORLD_X);
            state.y = std::min(std::max(state.y + uy * DT, 0.0f), WORLD_Y);
            for (int t = 0; t < T_HORIZON - 1; t++) {
                nom[t * 2 + 0] = nom[(t + 1) * 2 + 0];
                nom[t * 2 + 1] = nom[(t + 1) * 2 + 1];
            }
            nom[(T_HORIZON - 1) * 2 + 0] = 0.0f;
            nom[(T_HORIZON - 1) * 2 + 1] = 0.0f;
            CUDA_CHECK(cudaMemcpy(m.nominal, nom.data(),
                                  nom.size() * sizeof(float), cudaMemcpyHostToDevice));
        };
        auto it0 = std::chrono::high_resolution_clock::now();
        if (!base_done) run(m_base, base_nom, base_state, 0.0f);
        if (!vis_done) run(m_vis, vis_nom, vis_state, W_VIS);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto it1 = std::chrono::high_resolution_clock::now();
        if (step >= 4) {
            total_iter_ms += std::chrono::duration<double, std::milli>(it1 - it0).count();
            counted_iters++;
        }
        if (!base_done) base_path.push_back(base_state);
        if (!vis_done)  vis_path.push_back(vis_state);

        int vis_base = count_visible_host(occ, base_state.x, base_state.y, lm_x, lm_y);
        int vis_vis  = count_visible_host(occ, vis_state.x,  vis_state.y,  lm_x, lm_y);
        if (step >= 5) { sum_base_vis += vis_base; sum_vis_vis += vis_vis; vis_counted++; }

        cv::Mat pb = render_panel(h_vis, occ, base_path, goal, base_state,
                                  lm_x, lm_y, "baseline MPPI (no visibility cost)", vis_base);
        cv::Mat pv = render_panel(h_vis, occ, vis_path, goal, vis_state,
                                  lm_x, lm_y, "visibility-aware MPPI", vis_vis);
        cv::Mat frame(PANEL_H + 30, PANEL_W * 2 + 4, CV_8UC3, cv::Scalar(30, 30, 30));
        pb.copyTo(frame(cv::Rect(0, 30, PANEL_W, PANEL_H)));
        pv.copyTo(frame(cv::Rect(PANEL_W + 4, 30, PANEL_W, PANEL_H)));
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "Visibility-aware MPPI  step=%d  base_dist=%.2f  vis_dist=%.2f",
                      step,
                      (float)std::hypot(base_state.x - goal.x, base_state.y - goal.y),
                      (float)std::hypot(vis_state.x  - goal.x, vis_state.y  - goal.y));
        cv::putText(frame, buf, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        video.write(frame);

        if (!base_done && std::hypot(base_state.x - goal.x, base_state.y - goal.y) < 0.6f)
            base_done_step = step;
        if (!vis_done && std::hypot(vis_state.x - goal.x, vis_state.y - goal.y) < 0.6f)
            vis_done_step = step;
        if (base_done_step >= 0 && vis_done_step >= 0) break;
    }
    video.release();
    convert_avi_to_gif("gif/visibility_mppi.avi", "gif/visibility_mppi.gif", 15);

    std::printf("Goal reached: baseline step=%d, visibility-aware step=%d\n",
                base_done_step, vis_done_step);
    if (vis_counted > 0) {
        std::printf("Avg visible LMs along path:  baseline=%.2f   visibility-aware=%.2f\n",
                    (double)sum_base_vis / vis_counted,
                    (double)sum_vis_vis / vis_counted);
    }
    if (counted_iters > 0) {
        std::printf("Avg per-step MPPI time (both controllers): %.2f ms\n",
                    total_iter_ms / counted_iters);
    }
    std::printf("GIF saved to gif/visibility_mppi.gif\n");

    CUDA_CHECK(cudaFree(d_seed_a));
    CUDA_CHECK(cudaFree(d_seed_b));
    CUDA_CHECK(cudaFree(d_esdf));
    CUDA_CHECK(cudaFree(d_vis));
    CUDA_CHECK(cudaFree(d_lm_x));
    CUDA_CHECK(cudaFree(d_lm_y));
    CUDA_CHECK(cudaFree(d_occ));
    cudaFree(m_base.nominal); cudaFree(m_base.costs);
    cudaFree(m_base.perturbed); cudaFree(m_base.weights);
    cudaFree(m_base.rng);
    cudaFree(m_vis.nominal); cudaFree(m_vis.costs);
    cudaFree(m_vis.perturbed); cudaFree(m_vis.weights);
    cudaFree(m_vis.rng);
    return 0;
}
