// gpu_diffusion_policy.cu
//
// Behavior-cloned diffusion policy demo.
//
// The existing gpu_diffusion_planner demo uses an analytic score function for
// Langevin trajectory denoising. This file takes the next step toward a learned
// planner: it builds a small synthetic expert dataset, trains a GPU MLP to
// predict local waypoint corrections, then uses that learned policy inside the
// same massively parallel denoising loop.
//
// Output: gif/gpu_diffusion_policy.gif

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"
#include "gpu_mlp.cuh"

namespace cudabot {

constexpr int N_TRAJ = 512;
constexpr int N_WAYPOINTS = 64;
constexpr int DENOISE_STEPS = 88;
constexpr int TRAIN_SAMPLES = 768;
constexpr int TRAIN_STEPS = 260;
constexpr int INPUT_DIM = 8;
constexpr int OUTPUT_DIM = 2;
constexpr int HIDDEN_DIM = 24;
constexpr int MLP_LAYERS = 2;
constexpr int MAX_OBS = 8;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 600;
constexpr int VIDEO_FPS = 14;

constexpr float WORLD_W = 16.0f;
constexpr float WORLD_H = 10.0f;
constexpr float START_X = 1.0f;
constexpr float START_Y = 1.0f;
constexpr float GOAL_X = 15.0f;
constexpr float GOAL_Y = 9.0f;
constexpr float INIT_NOISE = 1.55f;
constexpr float NOISE_START = 0.26f;
constexpr float NOISE_END = 0.018f;
constexpr float POLICY_GAIN_START = 0.86f;
constexpr float POLICY_GAIN_END = 0.30f;
constexpr float SMOOTH_GAIN = 0.16f;
constexpr float MAX_STEP = 0.68f;
constexpr float SAFE_MARGIN = 0.72f;
constexpr float PI_F = 3.14159265358979323846f;

struct Circle {
    float x;
    float y;
    float r;
};

struct Point2 {
    float x;
    float y;
};

struct TrainingSet {
    std::vector<float> input;
    std::vector<float> target;
};

__constant__ Circle c_obs[MAX_OBS];
__constant__ int c_n_obs;

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

__host__ __device__ static inline float sqr(float x) {
    return x * x;
}

__host__ __device__ static inline Point2 straight_anchor(float s) {
    return {lerpf(START_X, GOAL_X, s), lerpf(START_Y, GOAL_Y, s)};
}

__device__ static void features_device(float x, float y, float s, float* feat) {
    float best_clearance = 1.0e9f;
    float best_ux = 1.0f;
    float best_uy = 0.0f;
    for (int i = 0; i < c_n_obs; i++) {
        Circle o = c_obs[i];
        float dx = x - o.x;
        float dy = y - o.y;
        float d = sqrtf(dx * dx + dy * dy) + 1.0e-6f;
        float clearance = d - o.r;
        if (clearance < best_clearance) {
            best_clearance = clearance;
            best_ux = dx / d;
            best_uy = dy / d;
        }
    }

    Point2 anchor = straight_anchor(s);
    feat[0] = 2.0f * x / WORLD_W - 1.0f;
    feat[1] = 2.0f * y / WORLD_H - 1.0f;
    feat[2] = 2.0f * s - 1.0f;
    feat[3] = clampf((anchor.x - x) / 4.0f, -1.0f, 1.0f);
    feat[4] = clampf((anchor.y - y) / 3.0f, -1.0f, 1.0f);
    feat[5] = best_ux;
    feat[6] = best_uy;
    feat[7] = clampf(best_clearance / 2.5f, -1.0f, 1.0f);
}

__global__ void init_trajectories_kernel(float* xs,
                                         float* ys,
                                         unsigned long long seed) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= N_WAYPOINTS || i >= N_TRAJ) return;

    float s = static_cast<float>(t) / (N_WAYPOINTS - 1);
    Point2 anchor = straight_anchor(s);
    int idx = i * N_WAYPOINTS + t;

    if (t == 0) {
        xs[idx] = START_X;
        ys[idx] = START_Y;
        return;
    }
    if (t == N_WAYPOINTS - 1) {
        xs[idx] = GOAL_X;
        ys[idx] = GOAL_Y;
        return;
    }

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, idx, 0, &rng);
    float spread = INIT_NOISE * (0.65f + 0.7f * sinf(PI_F * s));
    xs[idx] = clampf(anchor.x + spread * curand_normal(&rng), 0.25f, WORLD_W - 0.25f);
    ys[idx] = clampf(anchor.y + spread * curand_normal(&rng), 0.25f, WORLD_H - 0.25f);
}

__global__ void learned_denoise_kernel(const float* __restrict__ weights,
                                       const float* __restrict__ xs_in,
                                       const float* __restrict__ ys_in,
                                       float* __restrict__ xs_out,
                                       float* __restrict__ ys_out,
                                       float policy_gain,
                                       float noise_scale,
                                       int step_index,
                                       unsigned long long seed) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= N_WAYPOINTS || i >= N_TRAJ) return;

    int idx = i * N_WAYPOINTS + t;
    if (t == 0) {
        xs_out[idx] = START_X;
        ys_out[idx] = START_Y;
        return;
    }
    if (t == N_WAYPOINTS - 1) {
        xs_out[idx] = GOAL_X;
        ys_out[idx] = GOAL_Y;
        return;
    }

    float x = xs_in[idx];
    float y = ys_in[idx];
    float s = static_cast<float>(t) / (N_WAYPOINTS - 1);

    float feat[INPUT_DIM];
    float out[OUTPUT_DIM];
    float scratch[HIDDEN_DIM * 2];
    features_device(x, y, s, feat);
    mlp_forward(weights, feat, INPUT_DIM, out, OUTPUT_DIM,
                HIDDEN_DIM, MLP_LAYERS, scratch, 1);

    float smooth_x = xs_in[idx - 1] + xs_in[idx + 1] - 2.0f * x;
    float smooth_y = ys_in[idx - 1] + ys_in[idx + 1] - 2.0f * y;

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, idx + step_index * N_TRAJ * N_WAYPOINTS, 0, &rng);

    float dx = policy_gain * clampf(out[0], -1.2f, 1.2f) + SMOOTH_GAIN * smooth_x
             + noise_scale * curand_normal(&rng);
    float dy = policy_gain * clampf(out[1], -1.2f, 1.2f) + SMOOTH_GAIN * smooth_y
             + noise_scale * curand_normal(&rng);
    dx = clampf(dx, -MAX_STEP, MAX_STEP);
    dy = clampf(dy, -MAX_STEP, MAX_STEP);

    xs_out[idx] = clampf(x + dx, 0.15f, WORLD_W - 0.15f);
    ys_out[idx] = clampf(y + dy, 0.15f, WORLD_H - 0.15f);
}

__global__ void path_cost_kernel(const float* __restrict__ xs,
                                 const float* __restrict__ ys,
                                 float* __restrict__ cost) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_TRAJ) return;

    float c = 0.0f;
    for (int t = 0; t < N_WAYPOINTS - 1; t++) {
        int idx = i * N_WAYPOINTS + t;
        float dx = xs[idx + 1] - xs[idx];
        float dy = ys[idx + 1] - ys[idx];
        c += 0.20f * (dx * dx + dy * dy);
    }
    for (int t = 0; t < N_WAYPOINTS; t++) {
        int idx = i * N_WAYPOINTS + t;
        float x = xs[idx];
        float y = ys[idx];
        for (int o = 0; o < c_n_obs; o++) {
            Circle ob = c_obs[o];
            float d = sqrtf(sqr(x - ob.x) + sqr(y - ob.y)) - ob.r;
            if (d < 0.0f) c += 180.0f * (-d + 0.05f);
            else if (d < SAFE_MARGIN) c += 4.0f * sqr(SAFE_MARGIN - d);
        }
    }
    cost[i] = c;
}

static float clearance_host(float x,
                            float y,
                            const std::vector<Circle>& obs,
                            float* ux,
                            float* uy) {
    float best = 1.0e9f;
    *ux = 1.0f;
    *uy = 0.0f;
    for (const Circle& o : obs) {
        float dx = x - o.x;
        float dy = y - o.y;
        float d = std::sqrt(dx * dx + dy * dy) + 1.0e-6f;
        float c = d - o.r;
        if (c < best) {
            best = c;
            *ux = dx / d;
            *uy = dy / d;
        }
    }
    return best;
}

static void features_host(float x,
                          float y,
                          float s,
                          const std::vector<Circle>& obs,
                          float* feat) {
    float ux = 1.0f;
    float uy = 0.0f;
    float c = clearance_host(x, y, obs, &ux, &uy);
    Point2 anchor = straight_anchor(s);
    feat[0] = 2.0f * x / WORLD_W - 1.0f;
    feat[1] = 2.0f * y / WORLD_H - 1.0f;
    feat[2] = 2.0f * s - 1.0f;
    feat[3] = clampf((anchor.x - x) / 4.0f, -1.0f, 1.0f);
    feat[4] = clampf((anchor.y - y) / 3.0f, -1.0f, 1.0f);
    feat[5] = ux;
    feat[6] = uy;
    feat[7] = clampf(c / 2.5f, -1.0f, 1.0f);
}

static std::vector<Point2> expert_waypoints() {
    return {
        {START_X, START_Y},
        {2.7f, 1.2f},
        {5.0f, 1.6f},
        {6.7f, 3.0f},
        {8.9f, 7.2f},
        {10.9f, 8.7f},
        {13.2f, 9.0f},
        {GOAL_X, GOAL_Y},
    };
}

static Point2 interpolate_polyline(const std::vector<Point2>& pts, float s) {
    if (pts.empty()) return {0.0f, 0.0f};
    if (s <= 0.0f) return pts.front();
    if (s >= 1.0f) return pts.back();

    std::vector<float> prefix(pts.size(), 0.0f);
    for (size_t i = 1; i < pts.size(); i++) {
        float dx = pts[i].x - pts[i - 1].x;
        float dy = pts[i].y - pts[i - 1].y;
        prefix[i] = prefix[i - 1] + std::sqrt(dx * dx + dy * dy);
    }
    float total = prefix.back();
    float target = s * total;
    for (size_t i = 1; i < pts.size(); i++) {
        if (target <= prefix[i]) {
            float local = (target - prefix[i - 1]) / std::max(1.0e-6f, prefix[i] - prefix[i - 1]);
            return {lerpf(pts[i - 1].x, pts[i].x, local),
                    lerpf(pts[i - 1].y, pts[i].y, local)};
        }
    }
    return pts.back();
}

static TrainingSet make_training_set(const std::vector<Circle>& obs) {
    TrainingSet set;
    set.input.resize(TRAIN_SAMPLES * INPUT_DIM);
    set.target.resize(TRAIN_SAMPLES * OUTPUT_DIM);

    std::mt19937 rng(25052026);
    std::uniform_real_distribution<float> uni_s(0.025f, 0.975f);
    std::normal_distribution<float> noise(0.0f, 1.15f);
    std::uniform_real_distribution<float> extra(-0.55f, 0.55f);
    std::vector<Point2> expert = expert_waypoints();

    for (int i = 0; i < TRAIN_SAMPLES; i++) {
        float s = uni_s(rng);
        Point2 p = interpolate_polyline(expert, s);
        float x = clampf(p.x + noise(rng) + extra(rng), 0.25f, WORLD_W - 0.25f);
        float y = clampf(p.y + noise(rng) + extra(rng), 0.25f, WORLD_H - 0.25f);

        features_host(x, y, s, obs, &set.input[i * INPUT_DIM]);

        float ux = 1.0f;
        float uy = 0.0f;
        float clear = clearance_host(x, y, obs, &ux, &uy);
        float repulse = clear < SAFE_MARGIN ? (SAFE_MARGIN - clear) : 0.0f;
        float dx = 0.35f * (p.x - x) + 0.34f * repulse * ux;
        float dy = 0.35f * (p.y - y) + 0.34f * repulse * uy;
        set.target[i * OUTPUT_DIM + 0] = clampf(dx, -0.95f, 0.95f);
        set.target[i * OUTPUT_DIM + 1] = clampf(dy, -0.95f, 0.95f);
    }
    return set;
}

static cv::Point to_px(float x, float y) {
    int px = static_cast<int>(x / WORLD_W * PANEL_W);
    int py = static_cast<int>((1.0f - y / WORLD_H) * PANEL_H);
    return cv::Point(px, py);
}

static void draw_polyline(cv::Mat& img,
                          const std::vector<Point2>& path,
                          const cv::Scalar& color,
                          int thickness) {
    for (size_t i = 1; i < path.size(); i++) {
        cv::line(img, to_px(path[i - 1].x, path[i - 1].y),
                 to_px(path[i].x, path[i].y), color, thickness, cv::LINE_AA);
    }
}

static cv::Mat draw_frame(const std::vector<float>& xs,
                          const std::vector<float>& ys,
                          const std::vector<Circle>& obs,
                          const std::vector<Point2>& expert,
                          int step,
                          float loss,
                          float ms,
                          int best_idx,
                          const std::vector<float>* costs) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(22, 23, 25));

    for (int gy = 0; gy <= 10; gy++) {
        int y = static_cast<int>(gy * PANEL_H / 10.0f);
        cv::line(img, cv::Point(0, y), cv::Point(PANEL_W, y), cv::Scalar(34, 34, 38), 1);
    }
    for (int gx = 0; gx <= 16; gx += 2) {
        int x = static_cast<int>(gx * PANEL_W / 16.0f);
        cv::line(img, cv::Point(x, 0), cv::Point(x, PANEL_H), cv::Scalar(34, 34, 38), 1);
    }

    for (const Circle& o : obs) {
        int r = static_cast<int>(o.r / WORLD_W * PANEL_W);
        int margin = static_cast<int>((o.r + SAFE_MARGIN) / WORLD_W * PANEL_W);
        cv::circle(img, to_px(o.x, o.y), margin, cv::Scalar(35, 46, 54), 1, cv::LINE_AA);
        cv::circle(img, to_px(o.x, o.y), r, cv::Scalar(48, 62, 130), cv::FILLED);
        cv::circle(img, to_px(o.x, o.y), r, cv::Scalar(95, 120, 230), 2, cv::LINE_AA);
    }

    draw_polyline(img, expert, cv::Scalar(84, 155, 255), 2);

    int stride = std::max(1, N_TRAJ / 192);
    for (int i = 0; i < N_TRAJ; i += stride) {
        cv::Scalar col(70, 170, 185);
        for (int t = 0; t < N_WAYPOINTS - 1; t++) {
            int idx = i * N_WAYPOINTS + t;
            cv::line(img, to_px(xs[idx], ys[idx]),
                     to_px(xs[idx + 1], ys[idx + 1]), col, 1, cv::LINE_AA);
        }
    }

    if (best_idx >= 0) {
        for (int t = 0; t < N_WAYPOINTS - 1; t++) {
            int idx = best_idx * N_WAYPOINTS + t;
            cv::line(img, to_px(xs[idx], ys[idx]),
                     to_px(xs[idx + 1], ys[idx + 1]),
                     cv::Scalar(90, 245, 130), 3, cv::LINE_AA);
        }
    }

    cv::circle(img, to_px(START_X, START_Y), 7, cv::Scalar(245, 245, 245), cv::FILLED);
    cv::circle(img, to_px(GOAL_X, GOAL_Y), 7, cv::Scalar(70, 80, 245), cv::FILLED);

    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, 34), cv::Scalar(5, 7, 10), cv::FILLED);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "GPU diffusion policy  BC loss %.4f  denoise %3d/%d  %.2f ms/step",
                  loss, step, DENOISE_STEPS, ms);
    cv::putText(img, buf, cv::Point(12, 23),
                cv::FONT_HERSHEY_SIMPLEX, 0.56, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);

    if (costs != nullptr && best_idx >= 0) {
        std::snprintf(buf, sizeof(buf), "best path cost %.2f", (*costs)[best_idx]);
        cv::putText(img, buf, cv::Point(12, PANEL_H - 14),
                    cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(170, 240, 180), 1, cv::LINE_AA);
    }
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<Circle> obs = {
        {4.0f, 3.0f, 1.1f},
        {6.5f, 5.8f, 1.2f},
        {9.5f, 4.2f, 1.4f},
        {12.2f, 7.0f, 1.2f},
        {8.0f, 8.0f, 0.6f},
        {11.0f, 1.9f, 1.1f},
    };
    int n_obs = static_cast<int>(obs.size());
    CUDA_CHECK(cudaMemcpyToSymbol(c_obs, obs.data(), n_obs * sizeof(Circle)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_n_obs, &n_obs, sizeof(int)));

    TrainingSet train = make_training_set(obs);
    float* d_train_x = nullptr;
    float* d_train_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_train_x, train.input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_train_y, train.target.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_train_x, train.input.data(), train.input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_train_y, train.target.data(), train.target.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    GpuMLP policy(INPUT_DIM, HIDDEN_DIM, MLP_LAYERS, OUTPUT_DIM);
    policy.init_random(90617ULL);

    float loss = 0.0f;
    auto train_begin = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < TRAIN_STEPS; step++) {
        float lr = 0.018f * (1.0f - 0.75f * static_cast<float>(step) / TRAIN_STEPS);
        loss = policy.train_step_backprop(d_train_x, d_train_y, TRAIN_SAMPLES, lr, 1);
        if (step % 65 == 0) {
            std::printf("  train step %3d  loss %.5f  lr %.4f\n", step, loss, lr);
        }
    }
    auto train_end = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration<double, std::milli>(train_end - train_begin).count();
    std::printf("Behavior cloning: %d samples, %d steps, final loss %.5f, %.1f ms\n",
                TRAIN_SAMPLES, TRAIN_STEPS, loss, train_ms);

    CUDA_CHECK(cudaFree(d_train_x));
    CUDA_CHECK(cudaFree(d_train_y));

    int n_values = N_TRAJ * N_WAYPOINTS;
    float* d_x_a = nullptr;
    float* d_y_a = nullptr;
    float* d_x_b = nullptr;
    float* d_y_b = nullptr;
    float* d_cost = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x_a, n_values * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_a, n_values * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_b, n_values * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_b, n_values * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cost, N_TRAJ * sizeof(float)));

    dim3 block(32, 8);
    dim3 grid((N_WAYPOINTS + block.x - 1) / block.x,
              (N_TRAJ + block.y - 1) / block.y);
    init_trajectories_kernel<<<grid, block>>>(d_x_a, d_y_a, 72511ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_x(n_values);
    std::vector<float> h_y(n_values);
    std::vector<float> h_cost(N_TRAJ);
    std::vector<Point2> expert = expert_waypoints();

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_diffusion_policy.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_diffusion_policy.avi\n");
        return 1;
    }

    auto write_frame = [&](int step, float ms, int best_idx, const std::vector<float>* costs) {
        CUDA_CHECK(cudaMemcpy(h_x.data(), d_x_a, n_values * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y_a, n_values * sizeof(float), cudaMemcpyDeviceToHost));
        video.write(draw_frame(h_x, h_y, obs, expert, step, loss, ms, best_idx, costs));
    };

    write_frame(0, 0.0f, -1, nullptr);

    float* in_x = d_x_a;
    float* in_y = d_y_a;
    float* out_x = d_x_b;
    float* out_y = d_y_b;
    double total_ms = 0.0;
    int measured_steps = 0;
    float last_ms = 0.0f;

    for (int step = 0; step < DENOISE_STEPS; step++) {
        float u = static_cast<float>(step) / (DENOISE_STEPS - 1);
        float policy_gain = lerpf(POLICY_GAIN_START, POLICY_GAIN_END, u);
        float noise = lerpf(NOISE_START, NOISE_END, u);

        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0));
        learned_denoise_kernel<<<grid, block>>>(policy.device_weights(),
                                                in_x, in_y, out_x, out_y,
                                                policy_gain, noise, step, 11627ULL);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaEventElapsedTime(&last_ms, ev0, ev1));
        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));

        std::swap(in_x, out_x);
        std::swap(in_y, out_y);
        if (in_x != d_x_a) {
            CUDA_CHECK(cudaMemcpy(d_x_a, in_x, n_values * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_y_a, in_y, n_values * sizeof(float), cudaMemcpyDeviceToDevice));
            in_x = d_x_a;
            in_y = d_y_a;
            out_x = d_x_b;
            out_y = d_y_b;
        }

        if (step >= 5) {
            total_ms += last_ms;
            measured_steps++;
        }
        if (step % 3 == 0 || step == DENOISE_STEPS - 1) {
            write_frame(step + 1, last_ms, -1, nullptr);
        }
        if (step % 22 == 0) {
            std::printf("  denoise step %3d  gain %.3f  noise %.3f  %.3f ms\n",
                        step, policy_gain, noise, last_ms);
        }
    }

    path_cost_kernel<<<(N_TRAJ + 127) / 128, 128>>>(d_x_a, d_y_a, d_cost);
    CUDA_CHECK(cudaMemcpy(h_cost.data(), d_cost, N_TRAJ * sizeof(float), cudaMemcpyDeviceToHost));
    int best = static_cast<int>(std::min_element(h_cost.begin(), h_cost.end()) - h_cost.begin());
    std::printf("Best learned trajectory: idx %d  cost %.3f\n", best, h_cost[best]);

    for (int i = 0; i < 20; i++) {
        write_frame(DENOISE_STEPS, 0.0f, best, &h_cost);
    }
    video.release();

    double avg_ms = measured_steps > 0 ? total_ms / measured_steps : 0.0;
    std::printf("Avg learned denoise step: %.3f ms (%d trajectories x %d waypoints, MLP %d-%d-%d-%d)\n",
                avg_ms, N_TRAJ, N_WAYPOINTS, INPUT_DIM, HIDDEN_DIM, HIDDEN_DIM, OUTPUT_DIM);

    avi_to_gif("gif/gpu_diffusion_policy.avi", "gif/gpu_diffusion_policy.gif", VIDEO_FPS, 720);
    std::printf("GIF saved to gif/gpu_diffusion_policy.gif\n");

    CUDA_CHECK(cudaFree(d_x_a));
    CUDA_CHECK(cudaFree(d_y_a));
    CUDA_CHECK(cudaFree(d_x_b));
    CUDA_CHECK(cudaFree(d_y_b));
    CUDA_CHECK(cudaFree(d_cost));
    return 0;
}
