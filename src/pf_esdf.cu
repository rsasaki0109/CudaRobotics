/*************************************************************************
    PF with ESDF observation likelihood.

    Setup: 2D 40 x 30 m world with L landmarks at fixed positions. The
    robot has a single sensor that returns the (noisy) Euclidean
    distance to the *nearest* landmark — no IDs, no bearings.

    Two observation models compared:
      handcrafted: for each particle, scan all L landmarks (O(K * L))
                   to recompute the nearest-landmark distance, then
                   evaluate a Gaussian likelihood on the residual.
      ESDF lookup: precompute an ESDF of the landmark set once on the
                   GPU via JFA. Each particle weight is one bilinear
                   ESDF lookup (O(K)) plus a Gaussian.

    Both PFs use K = 10,000 particles, the same motion model, and the
    same noise. The ESDF lookup version becomes increasingly attractive
    as L grows; the headline metric is per-particle observation cost.

    Visualization: two panels side-by-side (handcrafted vs ESDF) showing
    particle cloud, true robot pose, and estimate per frame.
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

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(EXIT_FAILURE); } } while (0)

// -------------------------------------------------------------------------
// World / sensor
// -------------------------------------------------------------------------
constexpr float WORLD_X = 40.0f;
constexpr float WORLD_Y = 30.0f;
constexpr int   N_LANDMARKS = 64;

constexpr int   GRID_W = 400;
constexpr int   GRID_H = 300;
constexpr float RES = WORLD_X / GRID_W;
constexpr float MAX_DIST = 30.0f;

constexpr int   K_PART = 10000;
constexpr int   N_STEPS = 200;
constexpr float DT = 0.1f;
constexpr float SENSOR_SIGMA = 0.4f;
constexpr float MOTION_SIGMA = 0.15f;
constexpr int   PANEL_W = 600;
constexpr int   PANEL_H = 450;

// -------------------------------------------------------------------------
// JFA ESDF kernels (mirror src/comparison_esdf.cu, fixed grid)
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

// -------------------------------------------------------------------------
// PF kernels
// -------------------------------------------------------------------------
__device__ __forceinline__ float bilinear_lookup(const float* esdf,
                                                 float wx, float wy) {
    float fx = wx / RES - 0.5f;
    float fy = wy / RES - 0.5f;
    int x0 = static_cast<int>(floorf(fx));
    int y0 = static_cast<int>(floorf(fy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    if (x0 < 0) x0 = 0; if (x0 > GRID_W - 1) x0 = GRID_W - 1;
    if (y0 < 0) y0 = 0; if (y0 > GRID_H - 1) y0 = GRID_H - 1;
    if (x1 < 0) x1 = 0; if (x1 > GRID_W - 1) x1 = GRID_W - 1;
    if (y1 < 0) y1 = 0; if (y1 > GRID_H - 1) y1 = GRID_H - 1;
    float ax = fx - floorf(fx);
    float ay = fy - floorf(fy);
    float d00 = esdf[y0 * GRID_W + x0];
    float d10 = esdf[y0 * GRID_W + x1];
    float d01 = esdf[y1 * GRID_W + x0];
    float d11 = esdf[y1 * GRID_W + x1];
    float d0 = d00 * (1.0f - ax) + d10 * ax;
    float d1 = d01 * (1.0f - ax) + d11 * ax;
    return d0 * (1.0f - ay) + d1 * ay;
}

__global__ void init_rng(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void motion_update_kernel(float* __restrict__ part_x,
                                     float* __restrict__ part_y,
                                     float dvx, float dvy,
                                     curandState* __restrict__ rng) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    curandState s = rng[i];
    float nx = part_x[i] + dvx + MOTION_SIGMA * curand_normal(&s);
    float ny = part_y[i] + dvy + MOTION_SIGMA * curand_normal(&s);
    if (nx < 0.0f) nx = 0.0f; if (nx > WORLD_X) nx = WORLD_X;
    if (ny < 0.0f) ny = 0.0f; if (ny > WORLD_Y) ny = WORLD_Y;
    part_x[i] = nx;
    part_y[i] = ny;
    rng[i] = s;
}

__global__ void weight_handcrafted_kernel(const float* __restrict__ part_x,
                                          const float* __restrict__ part_y,
                                          const float* __restrict__ lm_x,
                                          const float* __restrict__ lm_y,
                                          int n_lm, float z_obs,
                                          float* __restrict__ weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    float px = part_x[i], py = part_y[i];
    float min_d2 = FLT_MAX;
    for (int j = 0; j < n_lm; j++) {
        float dx = px - lm_x[j], dy = py - lm_y[j];
        float d2 = dx * dx + dy * dy;
        if (d2 < min_d2) min_d2 = d2;
    }
    float predicted = sqrtf(min_d2);
    float r = (predicted - z_obs) / SENSOR_SIGMA;
    weights[i] = expf(-0.5f * r * r);
}

__global__ void weight_esdf_kernel(const float* __restrict__ part_x,
                                   const float* __restrict__ part_y,
                                   const float* __restrict__ esdf,
                                   float z_obs,
                                   float* __restrict__ weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    float px = part_x[i], py = part_y[i];
    float predicted = bilinear_lookup(esdf, px, py);
    float r = (predicted - z_obs) / SENSOR_SIGMA;
    weights[i] = expf(-0.5f * r * r);
}

__global__ void normalise_kernel(float* __restrict__ weights) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float s = 0.0f;
    for (int i = 0; i < K_PART; i++) s += weights[i];
    if (s < 1.0e-30f) {
        for (int i = 0; i < K_PART; i++) weights[i] = 1.0f / K_PART;
        return;
    }
    float inv = 1.0f / s;
    for (int i = 0; i < K_PART; i++) weights[i] *= inv;
}

__global__ void resample_kernel(const float* __restrict__ weights,
                                const float* __restrict__ part_x_in,
                                const float* __restrict__ part_y_in,
                                float* __restrict__ part_x_out,
                                float* __restrict__ part_y_out,
                                curandState* __restrict__ rng) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K_PART) return;
    curandState s = rng[i];
    float r = curand_uniform(&s);
    float acc = 0.0f;
    int chosen = K_PART - 1;
    for (int j = 0; j < K_PART; j++) {
        acc += weights[j];
        if (acc >= r) { chosen = j; break; }
    }
    part_x_out[i] = part_x_in[chosen];
    part_y_out[i] = part_y_in[chosen];
    rng[i] = s;
}

__global__ void mean_kernel(const float* __restrict__ part_x,
                            const float* __restrict__ part_y,
                            float* __restrict__ out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float mx = 0.0f, my = 0.0f;
    for (int i = 0; i < K_PART; i++) { mx += part_x[i]; my += part_y[i]; }
    out[0] = mx / K_PART;
    out[1] = my / K_PART;
}

// -------------------------------------------------------------------------
// Host helpers
// -------------------------------------------------------------------------
static void build_landmarks(std::vector<float>& lm_x,
                            std::vector<float>& lm_y,
                            std::mt19937& rng) {
    std::uniform_real_distribution<float> ux(2.0f, WORLD_X - 2.0f);
    std::uniform_real_distribution<float> uy(2.0f, WORLD_Y - 2.0f);
    lm_x.clear(); lm_y.clear();
    lm_x.reserve(N_LANDMARKS); lm_y.reserve(N_LANDMARKS);
    for (int i = 0; i < N_LANDMARKS; i++) { lm_x.push_back(ux(rng)); lm_y.push_back(uy(rng)); }
}

static void draw_world(cv::Mat& img, const std::vector<float>& lm_x,
                       const std::vector<float>& lm_y) {
    auto X = [&](float x) { return static_cast<int>(x / WORLD_X * img.cols); };
    auto Y = [&](float y) { return static_cast<int>((1.0f - y / WORLD_Y) * img.rows); };
    for (size_t i = 0; i < lm_x.size(); i++) {
        cv::circle(img, cv::Point(X(lm_x[i]), Y(lm_y[i])), 4,
                   cv::Scalar(120, 200, 255), cv::FILLED);
    }
}

static void draw_pf(cv::Mat& img,
                    const std::vector<float>& part_x,
                    const std::vector<float>& part_y,
                    cv::Scalar color) {
    auto X = [&](float x) { return static_cast<int>(x / WORLD_X * img.cols); };
    auto Y = [&](float y) { return static_cast<int>((1.0f - y / WORLD_Y) * img.rows); };
    for (size_t i = 0; i < part_x.size(); i += 5) {
        img.at<cv::Vec3b>(Y(part_y[i]), X(part_x[i])) =
            cv::Vec3b(static_cast<uchar>(color[0]),
                      static_cast<uchar>(color[1]),
                      static_cast<uchar>(color[2]));
    }
}

static void draw_pose(cv::Mat& img, float wx, float wy, cv::Scalar color, int r) {
    int sx = static_cast<int>(wx / WORLD_X * img.cols);
    int sy = static_cast<int>((1.0f - wy / WORLD_Y) * img.rows);
    cv::circle(img, cv::Point(sx, sy), r, color, 2);
}

static void convert_avi_to_gif(const char* avi, const char* gif, int fps) {
    char cmd[512];
    std::snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -i %s -vf 'fps=%d,scale=900:-1:flags=lanczos' -loop 0 %s "
        "> /dev/null 2>&1", avi, fps, gif);
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg conversion returned %d\n", rc);
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------
int main() {
    std::mt19937 rng(2026);
    std::vector<float> lm_x, lm_y;
    build_landmarks(lm_x, lm_y, rng);

    // Build the landmark occupancy grid and ESDF on the GPU
    std::vector<unsigned char> occ(GRID_W * GRID_H, 0u);
    for (int i = 0; i < N_LANDMARKS; i++) {
        int gx = static_cast<int>(lm_x[i] / RES);
        int gy = static_cast<int>(lm_y[i] / RES);
        if (gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H)
            occ[gy * GRID_W + gx] = 1u;
    }
    unsigned char* d_occ = nullptr;
    int *d_seed_a = nullptr, *d_seed_b = nullptr;
    float* d_esdf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_occ,    occ.size()));
    CUDA_CHECK(cudaMalloc(&d_seed_a, GRID_W * GRID_H * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seed_b, GRID_W * GRID_H * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_esdf,   GRID_W * GRID_H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_occ, occ.data(), occ.size(), cudaMemcpyHostToDevice));
    dim3 blk2d(16, 16);
    dim3 grd2d((GRID_W + 15) / 16, (GRID_H + 15) / 16);
    jfa_init_kernel<<<grd2d, blk2d>>>(d_occ, d_seed_a, GRID_W, GRID_H);
    int *in_ptr = d_seed_a, *out_ptr = d_seed_b;
    int kmax = std::max(GRID_W, GRID_H) / 2;
    for (int k = kmax; k >= 1; k /= 2) {
        jfa_step_kernel<<<grd2d, blk2d>>>(in_ptr, out_ptr, GRID_W, GRID_H, k);
        std::swap(in_ptr, out_ptr);
    }
    jfa_to_dist_kernel<<<grd2d, blk2d>>>(in_ptr, d_esdf, GRID_W, GRID_H, RES);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate two parallel PFs (handcrafted + esdf-lookup)
    auto alloc_pf = [&]() {
        struct PF { float* x; float* y; float* w; curandState* rng; float* mean; };
        PF p;
        CUDA_CHECK(cudaMalloc(&p.x,   K_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&p.y,   K_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&p.w,   K_PART * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&p.rng, K_PART * sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&p.mean, 2 * sizeof(float)));
        return p;
    };
    auto pf_h = alloc_pf();
    auto pf_e = alloc_pf();

    int threads = 256;
    int blocks = (K_PART + threads - 1) / threads;
    init_rng<<<blocks, threads>>>(pf_h.rng, K_PART, 100ULL);
    init_rng<<<blocks, threads>>>(pf_e.rng, K_PART, 200ULL);

    // Initial: uniform over world
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> px0(K_PART), py0(K_PART);
    for (int i = 0; i < K_PART; i++) {
        px0[i] = u(rng) * WORLD_X;
        py0[i] = u(rng) * WORLD_Y;
    }
    CUDA_CHECK(cudaMemcpy(pf_h.x, px0.data(), K_PART * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pf_h.y, py0.data(), K_PART * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pf_e.x, px0.data(), K_PART * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pf_e.y, py0.data(), K_PART * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Landmarks on the GPU
    float* d_lm_x = nullptr; float* d_lm_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lm_x, N_LANDMARKS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_y, N_LANDMARKS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lm_x, lm_x.data(), N_LANDMARKS * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lm_y, lm_y.data(), N_LANDMARKS * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Truth trajectory (sinusoidal)
    std::normal_distribution<float> sensor_noise(0.0f, SENSOR_SIGMA);

    cv::VideoWriter video("gif/pf_esdf.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 15,
                          cv::Size(PANEL_W * 2 + 4, PANEL_H + 30));
    if (!video.isOpened()) {
        std::fprintf(stderr, "Failed to open gif/pf_esdf.avi\n");
        return 1;
    }

    float true_x = 5.0f, true_y = 5.0f;
    float prev_x = true_x, prev_y = true_y;
    std::vector<float> h_part_x(K_PART), h_part_y(K_PART);
    std::vector<float> e_part_x(K_PART), e_part_y(K_PART);
    float h_mean[2], e_mean[2];

    double rmse_h_sum = 0.0, rmse_e_sum = 0.0;
    int rmse_counted = 0;
    double time_h_us_total = 0.0, time_e_us_total = 0.0;
    int time_calls = 0;

    float* tmp_x = nullptr; float* tmp_y = nullptr;
    CUDA_CHECK(cudaMalloc(&tmp_x, K_PART * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tmp_y, K_PART * sizeof(float)));

    for (int step = 0; step < N_STEPS; step++) {
        float t = step * DT;
        float tx = 8.0f + 14.0f * std::cos(0.05f * t);
        float ty = 8.0f + 8.0f * std::sin(0.1f * t);
        float dx = tx - prev_x;
        float dy = ty - prev_y;
        prev_x = true_x = tx; prev_y = true_y = ty;

        // Sensor: nearest landmark distance + noise
        float zmin = FLT_MAX;
        for (int j = 0; j < N_LANDMARKS; j++) {
            float ex = true_x - lm_x[j], ey = true_y - lm_y[j];
            float d = std::sqrt(ex * ex + ey * ey);
            if (d < zmin) zmin = d;
        }
        float z_obs = zmin + sensor_noise(rng);

        // motion
        motion_update_kernel<<<blocks, threads>>>(pf_h.x, pf_h.y, dx, dy, pf_h.rng);
        motion_update_kernel<<<blocks, threads>>>(pf_e.x, pf_e.y, dx, dy, pf_e.rng);

        // weights (timed)
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        weight_handcrafted_kernel<<<blocks, threads>>>(
            pf_h.x, pf_h.y, d_lm_x, d_lm_y, N_LANDMARKS, z_obs, pf_h.w);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        weight_esdf_kernel<<<blocks, threads>>>(
            pf_e.x, pf_e.y, d_esdf, z_obs, pf_e.w);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();
        if (step >= 5) {
            time_h_us_total += std::chrono::duration<double, std::micro>(t1 - t0).count();
            time_e_us_total += std::chrono::duration<double, std::micro>(t2 - t1).count();
            time_calls++;
        }

        normalise_kernel<<<1, 1>>>(pf_h.w);
        normalise_kernel<<<1, 1>>>(pf_e.w);

        resample_kernel<<<blocks, threads>>>(pf_h.w, pf_h.x, pf_h.y, tmp_x, tmp_y, pf_h.rng);
        std::swap(pf_h.x, tmp_x); std::swap(pf_h.y, tmp_y);
        resample_kernel<<<blocks, threads>>>(pf_e.w, pf_e.x, pf_e.y, tmp_x, tmp_y, pf_e.rng);
        std::swap(pf_e.x, tmp_x); std::swap(pf_e.y, tmp_y);

        mean_kernel<<<1, 1>>>(pf_h.x, pf_h.y, pf_h.mean);
        mean_kernel<<<1, 1>>>(pf_e.x, pf_e.y, pf_e.mean);
        CUDA_CHECK(cudaMemcpy(h_mean, pf_h.mean, 2 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(e_mean, pf_e.mean, 2 * sizeof(float), cudaMemcpyDeviceToHost));

        if (step >= 20) {
            float eh = std::hypot(h_mean[0] - true_x, h_mean[1] - true_y);
            float ee = std::hypot(e_mean[0] - true_x, e_mean[1] - true_y);
            rmse_h_sum += eh * eh; rmse_e_sum += ee * ee; rmse_counted++;
        }

        // render
        CUDA_CHECK(cudaMemcpy(h_part_x.data(), pf_h.x, K_PART * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_part_y.data(), pf_h.y, K_PART * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(e_part_x.data(), pf_e.x, K_PART * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(e_part_y.data(), pf_e.y, K_PART * sizeof(float),
                              cudaMemcpyDeviceToHost));
        cv::Mat panel_h(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(20, 20, 20));
        cv::Mat panel_e(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(20, 20, 20));
        draw_world(panel_h, lm_x, lm_y);
        draw_world(panel_e, lm_x, lm_y);
        draw_pf(panel_h, h_part_x, h_part_y, cv::Scalar(80, 220, 80));
        draw_pf(panel_e, e_part_x, e_part_y, cv::Scalar(220, 120, 60));
        draw_pose(panel_h, true_x, true_y, cv::Scalar(255, 255, 255), 7);
        draw_pose(panel_e, true_x, true_y, cv::Scalar(255, 255, 255), 7);
        draw_pose(panel_h, h_mean[0], h_mean[1], cv::Scalar(0, 255, 255), 5);
        draw_pose(panel_e, e_mean[0], e_mean[1], cv::Scalar(0, 255, 255), 5);

        cv::Mat frame(PANEL_H + 30, PANEL_W * 2 + 4, CV_8UC3, cv::Scalar(30, 30, 30));
        panel_h.copyTo(frame(cv::Rect(0, 30, PANEL_W, PANEL_H)));
        panel_e.copyTo(frame(cv::Rect(PANEL_W + 4, 30, PANEL_W, PANEL_H)));
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "handcrafted O(K*L=%d)   ESDF lookup O(K)   step=%d  L=%d  K=%d",
                      K_PART * N_LANDMARKS, step, N_LANDMARKS, K_PART);
        cv::putText(frame, buf, cv::Point(12, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        video.write(frame);
    }

    video.release();
    convert_avi_to_gif("gif/pf_esdf.avi", "gif/pf_esdf.gif", 15);

    if (rmse_counted > 0) {
        double rmse_h = std::sqrt(rmse_h_sum / rmse_counted);
        double rmse_e = std::sqrt(rmse_e_sum / rmse_counted);
        std::printf("RMSE  handcrafted: %.3f m   esdf-lookup: %.3f m\n",
                    rmse_h, rmse_e);
    }
    if (time_calls > 0) {
        double t_h = time_h_us_total / time_calls;
        double t_e = time_e_us_total / time_calls;
        std::printf("Per-step weight time  handcrafted (K*L=%d): %.2f us   "
                    "esdf-lookup (K=%d): %.2f us  (%.1fx faster)\n",
                    K_PART * N_LANDMARKS, t_h, K_PART, t_e, t_h / t_e);
    }
    std::printf("GIF saved to gif/pf_esdf.gif\n");

    CUDA_CHECK(cudaFree(d_occ));
    CUDA_CHECK(cudaFree(d_seed_a));
    CUDA_CHECK(cudaFree(d_seed_b));
    CUDA_CHECK(cudaFree(d_esdf));
    CUDA_CHECK(cudaFree(d_lm_x));
    CUDA_CHECK(cudaFree(d_lm_y));
    CUDA_CHECK(cudaFree(tmp_x));
    CUDA_CHECK(cudaFree(tmp_y));
    for (auto* p : {pf_h.x, pf_h.y, pf_h.w, pf_h.mean,
                    pf_e.x, pf_e.y, pf_e.w, pf_e.mean}) {
        CUDA_CHECK(cudaFree(p));
    }
    CUDA_CHECK(cudaFree(pf_h.rng));
    CUDA_CHECK(cudaFree(pf_e.rng));
    return 0;
}
