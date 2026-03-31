#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "gpu_mlp.cuh"

namespace cudabot {

static constexpr float NSDF_WORLD_MIN = 0.0f;
static constexpr float NSDF_WORLD_MAX = 10.0f;
static constexpr float NSDF_WORLD_SIZE = NSDF_WORLD_MAX - NSDF_WORLD_MIN;
static constexpr int NSDF_GRID_RES = 64;
static constexpr int NSDF_TRAIN_SAMPLES = 10000;
static constexpr int NSDF_INPUT_DIM = 2;
static constexpr int NSDF_HIDDEN_DIM = 64;
static constexpr int NSDF_HIDDEN_LAYERS = 3;
static constexpr int NSDF_OUTPUT_DIM = 1;
static constexpr int NSDF_ACTIVATION = 0;  // ReLU

struct CirclePrimitive {
    float x;
    float y;
    float r;
};

struct BoxPrimitive {
    float cx;
    float cy;
    float hx;
    float hy;
};

enum class NeuralSceneKind {
    DemoWorld,
    LShapeWorld
};

inline float host_clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

__host__ __device__ inline float sdf_circle_2d(float x, float y, const CirclePrimitive& c) {
    float dx = x - c.x;
    float dy = y - c.y;
    return sqrtf(dx * dx + dy * dy) - c.r;
}

__host__ __device__ inline float sdf_box_2d(float x, float y, const BoxPrimitive& b) {
    float qx = fabsf(x - b.cx) - b.hx;
    float qy = fabsf(y - b.cy) - b.hy;
    float ax = fmaxf(qx, 0.0f);
    float ay = fmaxf(qy, 0.0f);
    float outside = sqrtf(ax * ax + ay * ay);
    float inside = fminf(fmaxf(qx, qy), 0.0f);
    return outside + inside;
}

__host__ __device__ inline float sdf_union_2d(float a, float b) {
    return fminf(a, b);
}

__host__ __device__ inline float demo_world_sdf(float x, float y) {
    float sdf = 1.0e6f;
    sdf = sdf_union_2d(sdf, sdf_circle_2d(x, y, {3.0f, 3.2f, 1.0f}));
    sdf = sdf_union_2d(sdf, sdf_circle_2d(x, y, {6.7f, 4.1f, 1.1f}));
    sdf = sdf_union_2d(sdf, sdf_circle_2d(x, y, {5.2f, 7.6f, 0.9f}));
    sdf = sdf_union_2d(sdf, sdf_box_2d(x, y, {4.8f, 2.1f, 1.2f, 0.28f}));
    sdf = sdf_union_2d(sdf, sdf_box_2d(x, y, {7.6f, 7.1f, 0.28f, 1.5f}));
    return sdf;
}

__host__ __device__ inline float lshape_world_sdf(float x, float y) {
    float vertical = sdf_box_2d(x, y, {4.0f, 6.2f, 0.65f, 2.7f});
    float horizontal = sdf_box_2d(x, y, {6.3f, 4.0f, 2.95f, 0.65f});
    float upper_block = sdf_box_2d(x, y, {7.9f, 7.6f, 0.85f, 0.85f});
    return sdf_union_2d(sdf_union_2d(vertical, horizontal), upper_block);
}

__host__ __device__ inline float scene_sdf(NeuralSceneKind scene, float x, float y) {
    return scene == NeuralSceneKind::DemoWorld ? demo_world_sdf(x, y) : lshape_world_sdf(x, y);
}

inline void encode_sdf_input(float x, float y, float* out) {
    out[0] = 2.0f * (x - NSDF_WORLD_MIN) / NSDF_WORLD_SIZE - 1.0f;
    out[1] = 2.0f * (y - NSDF_WORLD_MIN) / NSDF_WORLD_SIZE - 1.0f;
}

inline std::vector<float> make_eval_inputs(int res) {
    std::vector<float> inputs(res * res * NSDF_INPUT_DIM);
    for (int iy = 0; iy < res; iy++) {
        for (int ix = 0; ix < res; ix++) {
            float x = NSDF_WORLD_MIN + NSDF_WORLD_SIZE * (static_cast<float>(ix) + 0.5f) / res;
            float y = NSDF_WORLD_MIN + NSDF_WORLD_SIZE * (static_cast<float>(iy) + 0.5f) / res;
            float* ptr = &inputs[(iy * res + ix) * NSDF_INPUT_DIM];
            encode_sdf_input(x, y, ptr);
        }
    }
    return inputs;
}

inline std::vector<float> make_true_sdf_grid(NeuralSceneKind scene, int res) {
    std::vector<float> grid(res * res);
    for (int iy = 0; iy < res; iy++) {
        for (int ix = 0; ix < res; ix++) {
            float x = NSDF_WORLD_MIN + NSDF_WORLD_SIZE * (static_cast<float>(ix) + 0.5f) / res;
            float y = NSDF_WORLD_MIN + NSDF_WORLD_SIZE * (static_cast<float>(iy) + 0.5f) / res;
            grid[iy * res + ix] = scene_sdf(scene, x, y);
        }
    }
    return grid;
}

inline void make_training_set(
    NeuralSceneKind scene,
    std::vector<float>& inputs,
    std::vector<float>& targets,
    int n_samples = NSDF_TRAIN_SAMPLES,
    int grid_res = NSDF_GRID_RES,
    unsigned int seed = 7)
{
    inputs.resize(n_samples * NSDF_INPUT_DIM);
    targets.resize(n_samples * NSDF_OUTPUT_DIM);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> cell_dist(0, grid_res - 1);
    std::uniform_real_distribution<float> jitter(-0.45f, 0.45f);
    float cell = NSDF_WORLD_SIZE / grid_res;

    for (int n = 0; n < n_samples; n++) {
        int gx = cell_dist(rng);
        int gy = cell_dist(rng);
        float x = NSDF_WORLD_MIN + (gx + 0.5f + jitter(rng)) * cell;
        float y = NSDF_WORLD_MIN + (gy + 0.5f + jitter(rng)) * cell;
        x = host_clampf(x, NSDF_WORLD_MIN, NSDF_WORLD_MAX);
        y = host_clampf(y, NSDF_WORLD_MIN, NSDF_WORLD_MAX);
        encode_sdf_input(x, y, &inputs[n * NSDF_INPUT_DIM]);
        targets[n] = scene_sdf(scene, x, y);
    }
}

inline float train_neural_sdf(
    GpuMLP& mlp,
    const std::vector<float>& inputs,
    const std::vector<float>& targets,
    int epochs = 500,
    int batch_size = 256,
    float lr = 0.001f,
    int activation = NSDF_ACTIVATION,
    int batches_per_epoch = 2,
    std::vector<float>* epoch_losses = nullptr)
{
    std::vector<int> indices(targets.size());
    for (size_t i = 0; i < indices.size(); i++) indices[i] = static_cast<int>(i);
    std::mt19937 rng(17);

    std::vector<float> batch_in(batch_size * NSDF_INPUT_DIM);
    std::vector<float> batch_out(batch_size * NSDF_OUTPUT_DIM);

    float* d_input = nullptr;
    float* d_target = nullptr;
    cudaMalloc(&d_input, batch_in.size() * sizeof(float));
    cudaMalloc(&d_target, batch_out.size() * sizeof(float));

    float last_loss = 0.0f;
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::shuffle(indices.begin(), indices.end(), rng);
        float epoch_loss = 0.0f;

        for (int b = 0; b < batches_per_epoch; b++) {
            int base = (b * batch_size) % static_cast<int>(indices.size());
            for (int i = 0; i < batch_size; i++) {
                int src = indices[(base + i) % indices.size()];
                batch_in[i * NSDF_INPUT_DIM + 0] = inputs[src * NSDF_INPUT_DIM + 0];
                batch_in[i * NSDF_INPUT_DIM + 1] = inputs[src * NSDF_INPUT_DIM + 1];
                batch_out[i] = targets[src];
            }
            cudaMemcpy(d_input, batch_in.data(), batch_in.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_target, batch_out.data(), batch_out.size() * sizeof(float), cudaMemcpyHostToDevice);
            last_loss = mlp.train_step_backprop(d_input, d_target, batch_size, lr, activation);
            epoch_loss += last_loss;
        }

        epoch_loss /= static_cast<float>(batches_per_epoch);
        if (epoch_losses) epoch_losses->push_back(epoch_loss);
    }

    cudaFree(d_input);
    cudaFree(d_target);
    return last_loss;
}

inline std::vector<float> predict_sdf_grid(GpuMLP& mlp, int res, int activation = NSDF_ACTIVATION) {
    std::vector<float> inputs = make_eval_inputs(res);
    std::vector<float> outputs(res * res * NSDF_OUTPUT_DIM);

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, inputs.size() * sizeof(float));
    cudaMalloc(&d_output, outputs.size() * sizeof(float));
    cudaMemcpy(d_input, inputs.data(), inputs.size() * sizeof(float), cudaMemcpyHostToDevice);
    mlp.forward_batch(d_input, d_output, res * res, activation);
    cudaMemcpy(outputs.data(), d_output, outputs.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    return outputs;
}

inline cv::Point world_to_pixel_nsdf(float x, float y, int size) {
    int px = static_cast<int>((x - NSDF_WORLD_MIN) / NSDF_WORLD_SIZE * size);
    int py = size - 1 - static_cast<int>((y - NSDF_WORLD_MIN) / NSDF_WORLD_SIZE * size);
    px = std::max(0, std::min(size - 1, px));
    py = std::max(0, std::min(size - 1, py));
    return cv::Point(px, py);
}

inline void draw_scene_overlay(cv::Mat& img, NeuralSceneKind scene) {
    if (scene == NeuralSceneKind::DemoWorld) {
        const CirclePrimitive circles[] = {
            {3.0f, 3.2f, 1.0f},
            {6.7f, 4.1f, 1.1f},
            {5.2f, 7.6f, 0.9f}
        };
        const BoxPrimitive boxes[] = {
            {4.8f, 2.1f, 1.2f, 0.28f},
            {7.6f, 7.1f, 0.28f, 1.5f}
        };
        for (const auto& c : circles) {
            cv::circle(img, world_to_pixel_nsdf(c.x, c.y, img.cols),
                       static_cast<int>(c.r / NSDF_WORLD_SIZE * img.cols),
                       cv::Scalar(0, 0, 0), 2);
        }
        for (const auto& b : boxes) {
            cv::Point p0 = world_to_pixel_nsdf(b.cx - b.hx, b.cy - b.hy, img.cols);
            cv::Point p1 = world_to_pixel_nsdf(b.cx + b.hx, b.cy + b.hy, img.cols);
            cv::rectangle(img, cv::Rect(std::min(p0.x, p1.x), std::min(p0.y, p1.y),
                                        std::abs(p1.x - p0.x), std::abs(p1.y - p0.y)),
                          cv::Scalar(0, 0, 0), 2);
        }
    } else {
        const BoxPrimitive boxes[] = {
            {4.0f, 6.2f, 0.65f, 2.7f},
            {6.3f, 4.0f, 2.95f, 0.65f},
            {7.9f, 7.6f, 0.85f, 0.85f}
        };
        for (const auto& b : boxes) {
            cv::Point p0 = world_to_pixel_nsdf(b.cx - b.hx, b.cy - b.hy, img.cols);
            cv::Point p1 = world_to_pixel_nsdf(b.cx + b.hx, b.cy + b.hy, img.cols);
            cv::rectangle(img, cv::Rect(std::min(p0.x, p1.x), std::min(p0.y, p1.y),
                                        std::abs(p1.x - p0.x), std::abs(p1.y - p0.y)),
                          cv::Scalar(0, 0, 0), 2);
        }
    }
}

inline cv::Mat render_sdf_heatmap(
    const std::vector<float>& grid,
    int res,
    NeuralSceneKind scene,
    const std::string& title,
    int img_size = 420)
{
    float max_abs = 1.0e-3f;
    for (float v : grid) max_abs = std::max(max_abs, std::fabs(v));

    cv::Mat gray(res, res, CV_8UC1);
    for (int iy = 0; iy < res; iy++) {
        for (int ix = 0; ix < res; ix++) {
            float v = grid[iy * res + ix];
            float norm = 0.5f + 0.5f * (v / max_abs);
            norm = host_clampf(norm, 0.0f, 1.0f);
            gray.at<unsigned char>(res - 1 - iy, ix) = static_cast<unsigned char>(255.0f * norm);
        }
    }

    cv::Mat color;
    cv::applyColorMap(gray, color, cv::COLORMAP_TURBO);
    cv::resize(color, color, cv::Size(img_size, img_size), 0.0, 0.0, cv::INTER_NEAREST);
    draw_scene_overlay(color, scene);

    cv::putText(color, title, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 255), 2);
    return color;
}

inline float sample_grid_bilinear(const std::vector<float>& grid, int res, float x, float y) {
    float fx = host_clampf((x - NSDF_WORLD_MIN) / NSDF_WORLD_SIZE * (res - 1), 0.0f, res - 1.0f);
    float fy = host_clampf((y - NSDF_WORLD_MIN) / NSDF_WORLD_SIZE * (res - 1), 0.0f, res - 1.0f);

    int x0 = static_cast<int>(fx);
    int y0 = static_cast<int>(fy);
    int x1 = std::min(x0 + 1, res - 1);
    int y1 = std::min(y0 + 1, res - 1);
    float tx = fx - x0;
    float ty = fy - y0;

    float v00 = grid[y0 * res + x0];
    float v10 = grid[y0 * res + x1];
    float v01 = grid[y1 * res + x0];
    float v11 = grid[y1 * res + x1];

    float v0 = v00 * (1.0f - tx) + v10 * tx;
    float v1 = v01 * (1.0f - tx) + v11 * tx;
    return v0 * (1.0f - ty) + v1 * ty;
}

inline void draw_path(
    cv::Mat& img,
    const std::vector<cv::Point2f>& path,
    const cv::Scalar& color,
    float radius = 0.12f)
{
    for (size_t i = 1; i < path.size(); i++) {
        cv::line(img, world_to_pixel_nsdf(path[i - 1].x, path[i - 1].y, img.cols),
                 world_to_pixel_nsdf(path[i].x, path[i].y, img.cols), color, 2);
    }
    if (!path.empty()) {
        cv::circle(img, world_to_pixel_nsdf(path.back().x, path.back().y, img.cols),
                   std::max(2, static_cast<int>(radius / NSDF_WORLD_SIZE * img.cols)),
                   color, -1);
    }
}

inline void draw_start_goal(cv::Mat& img, cv::Point2f start, cv::Point2f goal) {
    cv::circle(img, world_to_pixel_nsdf(start.x, start.y, img.cols), 7, cv::Scalar(0, 255, 0), -1);
    cv::circle(img, world_to_pixel_nsdf(goal.x, goal.y, img.cols), 7, cv::Scalar(255, 255, 255), -1);
}

inline void convert_avi_to_gif(const std::string& avi_path, const std::string& gif_path, int fps = 15) {
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -i %s -vf 'fps=%d,scale=480:-1' -loop 0 %s 2>/dev/null",
                  avi_path.c_str(), fps, gif_path.c_str());
    std::system(cmd);
}

}  // namespace cudabot
