/*************************************************************************
    Neural-SDF Potential Field
    - Train a neural SDF and build a 100x100 potential map on GPU
    - CPU gradient descent follows the learned field
    Output: gif/sdf_potential_field.gif
 ************************************************************************/

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "neural_sdf_nav.cuh"

using namespace std;
using namespace cudabot;

static const char* AVI_PATH = "gif/sdf_potential_field.avi";
static const char* GIF_PATH = "gif/sdf_potential_field.gif";
static const int FIELD_RES = 100;

__device__ float eval_sdf_network(const float* weights, float px, float py) {
    float input[2];
    float output[1];
    float scratch[NSDF_HIDDEN_DIM * 2];
    input[0] = 2.0f * (px - NSDF_WORLD_MIN) / NSDF_WORLD_SIZE - 1.0f;
    input[1] = 2.0f * (py - NSDF_WORLD_MIN) / NSDF_WORLD_SIZE - 1.0f;
    mlp_forward(weights, input, NSDF_INPUT_DIM, output, NSDF_OUTPUT_DIM,
                NSDF_HIDDEN_DIM, NSDF_HIDDEN_LAYERS, scratch, NSDF_ACTIVATION);
    return output[0];
}

__global__ void compute_sdf_potential_kernel(
    const float* weights,
    float* d_field,
    int res,
    float goal_x,
    float goal_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= res * res) return;

    int ix = idx % res;
    int iy = idx / res;
    float x = NSDF_WORLD_MIN + NSDF_WORLD_SIZE * (static_cast<float>(ix) + 0.5f) / res;
    float y = NSDF_WORLD_MIN + NSDF_WORLD_SIZE * (static_cast<float>(iy) + 0.5f) / res;
    float h = 0.01f;

    float sdf = eval_sdf_network(weights, x, y);
    float dsdx = (eval_sdf_network(weights, fminf(x + h, NSDF_WORLD_MAX), y)
                - eval_sdf_network(weights, fmaxf(x - h, NSDF_WORLD_MIN), y)) / (2.0f * h);
    float dsdy = (eval_sdf_network(weights, x, fminf(y + h, NSDF_WORLD_MAX))
                - eval_sdf_network(weights, x, fmaxf(y - h, NSDF_WORLD_MIN))) / (2.0f * h);
    float grad_norm = sqrtf(dsdx * dsdx + dsdy * dsdy);

    float dx = x - goal_x;
    float dy = y - goal_y;
    float attractive = 0.55f * (dx * dx + dy * dy);

    float repulsive = 0.0f;
    float safe = 1.2f;
    float margin = fmaxf(sdf, 0.03f);
    if (sdf < safe) {
        float inv = 1.0f / margin - 1.0f / safe;
        repulsive = 10.0f * inv * inv * (1.0f + 0.25f * grad_norm);
    }
    if (sdf < 0.0f) repulsive += 120.0f;

    d_field[idx] = attractive + repulsive;
}

static cv::Mat render_field(const vector<float>& field) {
    float vmin = 1.0e9f;
    float vmax = -1.0e9f;
    for (float v : field) {
        vmin = min(vmin, v);
        vmax = max(vmax, v);
    }
    if (vmax <= vmin) vmax = vmin + 1.0f;

    cv::Mat gray(FIELD_RES, FIELD_RES, CV_8UC1);
    for (int iy = 0; iy < FIELD_RES; iy++) {
        for (int ix = 0; ix < FIELD_RES; ix++) {
            float v = field[iy * FIELD_RES + ix];
            float norm = (v - vmin) / (vmax - vmin);
            norm = host_clampf(norm, 0.0f, 1.0f);
            gray.at<unsigned char>(FIELD_RES - 1 - iy, ix) = static_cast<unsigned char>(255.0f * (1.0f - norm));
        }
    }

    cv::Mat color;
    cv::applyColorMap(gray, color, cv::COLORMAP_JET);
    cv::resize(color, color, cv::Size(480, 480), 0.0, 0.0, cv::INTER_NEAREST);
    draw_scene_overlay(color, NeuralSceneKind::DemoWorld);
    cv::putText(color, "Neural SDF Potential Field", cv::Point(10, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.68, cv::Scalar(255, 255, 255), 2);
    return color;
}

int main() {
    vector<float> train_inputs;
    vector<float> train_targets;
    make_training_set(NeuralSceneKind::DemoWorld, train_inputs, train_targets);

    GpuMLP mlp(NSDF_INPUT_DIM, NSDF_HIDDEN_DIM, NSDF_HIDDEN_LAYERS, NSDF_OUTPUT_DIM);
    mlp.init_random(123);
    float train_loss = train_neural_sdf(mlp, train_inputs, train_targets, 500, 256, 0.001f,
                                        NSDF_ACTIVATION, 2, nullptr);
    cout << "Training loss: " << train_loss << endl;

    float* d_field = nullptr;
    cudaMalloc(&d_field, FIELD_RES * FIELD_RES * sizeof(float));
    int threads = 256;
    int blocks = (FIELD_RES * FIELD_RES + threads - 1) / threads;
    cv::Point2f start(0.8f, 0.9f);
    cv::Point2f goal(9.1f, 9.0f);
    compute_sdf_potential_kernel<<<blocks, threads>>>(mlp.device_weights(), d_field, FIELD_RES, goal.x, goal.y);
    cudaDeviceSynchronize();

    vector<float> field(FIELD_RES * FIELD_RES);
    cudaMemcpy(field.data(), d_field, field.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_field);

    vector<cv::Point2f> path;
    path.push_back(start);
    cv::Point2f p = start;
    const float eps = 0.08f;
    const float step = 0.11f;
    for (int iter = 0; iter < 180; iter++) {
        float dx = sample_grid_bilinear(field, FIELD_RES, p.x + eps, p.y)
                 - sample_grid_bilinear(field, FIELD_RES, p.x - eps, p.y);
        float dy = sample_grid_bilinear(field, FIELD_RES, p.x, p.y + eps)
                 - sample_grid_bilinear(field, FIELD_RES, p.x, p.y - eps);
        float norm = sqrtf(dx * dx + dy * dy);
        if (norm < 1.0e-5f) break;

        cv::Point2f goal_dir(goal.x - p.x, goal.y - p.y);
        float goal_norm = sqrtf(goal_dir.x * goal_dir.x + goal_dir.y * goal_dir.y);
        if (goal_norm > 1.0e-6f) {
            goal_dir.x /= goal_norm;
            goal_dir.y /= goal_norm;
        }

        p.x -= step * dx / norm;
        p.y -= step * dy / norm;
        p.x += 0.02f * goal_dir.x;
        p.y += 0.02f * goal_dir.y;
        p.x = host_clampf(p.x, NSDF_WORLD_MIN, NSDF_WORLD_MAX);
        p.y = host_clampf(p.y, NSDF_WORLD_MIN, NSDF_WORLD_MAX);

        if (scene_sdf(NeuralSceneKind::DemoWorld, p.x, p.y) < 0.02f) {
            p.x -= 0.08f * dx / norm;
            p.y -= 0.08f * dy / norm;
        }

        path.push_back(p);
        float dist_goal = sqrtf((p.x - goal.x) * (p.x - goal.x) + (p.y - goal.y) * (p.y - goal.y));
        if (dist_goal < 0.25f) break;
    }

    cv::VideoWriter video(
        AVI_PATH,
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
        15,
        cv::Size(480, 480));

    if (!video.isOpened()) {
        fprintf(stderr, "Failed to open %s\n", AVI_PATH);
        return 1;
    }

    for (size_t i = 1; i <= path.size(); i++) {
        cv::Mat frame = render_field(field);
        vector<cv::Point2f> partial(path.begin(), path.begin() + i);
        draw_path(frame, partial, cv::Scalar(0, 0, 255));
        draw_start_goal(frame, start, goal);

        char buf[128];
        std::snprintf(buf, sizeof(buf), "Path steps: %zu  Final dist: %.2f", i - 1,
                      hypotf(path[i - 1].x - goal.x, path[i - 1].y - goal.y));
        cv::putText(frame, buf, cv::Point(10, 460), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(255, 255, 255), 1);
        video.write(frame);
    }

    video.release();
    cout << "Video saved to gif/sdf_potential_field.avi" << endl;
    convert_avi_to_gif(AVI_PATH, GIF_PATH, 15);
    cout << "GIF saved to gif/sdf_potential_field.gif" << endl;
    return 0;
}
