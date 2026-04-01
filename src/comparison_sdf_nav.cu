/*************************************************************************
    SDF Navigation Comparison
    - Left: circle-approximation MPPI on an L-shaped obstacle world
    - Right: neural-SDF MPPI on the same world
    Output: gif/comparison_sdf_nav.gif
 ************************************************************************/

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "neural_sdf_nav.cuh"

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;
using namespace cudabot;

static const char* AVI_PATH = "gif/comparison_sdf_nav.avi";
static const char* GIF_PATH = "gif/comparison_sdf_nav.gif";

static const int K_SAMPLES = 4096;
static const int T_HORIZON = 30;
static const int MAX_STEPS = 170;
static const float DT = 0.16f;
static const float MAX_SPEED = 1.45f;
static const float LAMBDA = 2.0f;
static const int SDF_WEIGHT_CAP = 12000;

__constant__ float d_compare_sdf_weights[SDF_WEIGHT_CAP];

__device__ float eval_neural_lshape(float x, float y) {
    float input[2];
    float output[1];
    float scratch[NSDF_HIDDEN_DIM * 2];
    input[0] = 2.0f * (x - NSDF_WORLD_MIN) / NSDF_WORLD_SIZE - 1.0f;
    input[1] = 2.0f * (y - NSDF_WORLD_MIN) / NSDF_WORLD_SIZE - 1.0f;
    mlp_forward(d_compare_sdf_weights, input, NSDF_INPUT_DIM, output, NSDF_OUTPUT_DIM,
                NSDF_HIDDEN_DIM, NSDF_HIDDEN_LAYERS, scratch, NSDF_ACTIVATION);
    return output[0];
}

__device__ float circle_approx_cost(float x, float y) {
    const CirclePrimitive approx[] = {
        {4.2f, 5.0f, 1.0f},
        {6.3f, 4.2f, 1.1f},
        {7.8f, 7.5f, 1.0f}
    };
    float cost = 0.0f;
    for (int i = 0; i < 3; i++) {
        const CirclePrimitive& c = approx[i];
        float dx = x - c.x;
        float dy = y - c.y;
        float d = sqrtf(dx * dx + dy * dy) - c.r;
        if (d < 0.8f) {
            float margin = fmaxf(d, 0.03f);
            float inv = 1.0f / margin - 1.0f / 0.8f;
            cost += 4.0f * inv * inv;
        }
        if (d < 0.0f) cost += 120.0f;
    }
    return cost;
}

__device__ float neural_sdf_cost(float x, float y) {
    float sdf = eval_neural_lshape(x, y);
    float cost = 0.0f;
    if (sdf < 0.8f) {
        float margin = fmaxf(sdf, 0.03f);
        float inv = 1.0f / margin - 1.0f / 0.8f;
        cost += 4.0f * inv * inv;
    }
    if (sdf < 0.0f) cost += 120.0f;
    return cost;
}

__global__ void init_rng(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

template <bool UseNeural>
__global__ void rollout_kernel(
    float sx,
    float sy,
    const float* d_nominal,
    float* d_costs,
    float* d_perturbed,
    curandState* d_rng)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K_SAMPLES) return;

    curandState rng = d_rng[k];
    float x = sx;
    float y = sy;
    float total_cost = 0.0f;

    for (int t = 0; t < T_HORIZON; t++) {
        float ux = d_nominal[t * 2 + 0] + 0.48f * curand_normal(&rng);
        float uy = d_nominal[t * 2 + 1] + 0.48f * curand_normal(&rng);
        ux = fminf(fmaxf(ux, -MAX_SPEED), MAX_SPEED);
        uy = fminf(fmaxf(uy, -MAX_SPEED), MAX_SPEED);
        d_perturbed[k * T_HORIZON * 2 + t * 2 + 0] = ux;
        d_perturbed[k * T_HORIZON * 2 + t * 2 + 1] = uy;

        x += ux * DT;
        y += uy * DT;
        float dx = x - 9.2f;
        float dy = y - 9.0f;
        total_cost += 1.7f * sqrtf(dx * dx + dy * dy + 1.0e-4f);
        total_cost += 0.18f * (ux * ux + uy * uy);
        total_cost += UseNeural ? neural_sdf_cost(x, y) : circle_approx_cost(x, y);

        if (x < NSDF_WORLD_MIN || x > NSDF_WORLD_MAX || y < NSDF_WORLD_MIN || y > NSDF_WORLD_MAX) {
            total_cost += 80.0f;
        }
    }

    float dx = x - 9.2f;
    float dy = y - 9.0f;
    total_cost += 12.0f * sqrtf(dx * dx + dy * dy + 1.0e-4f);
    d_costs[k] = total_cost;
    d_rng[k] = rng;
}

__global__ void compute_weights_kernel(const float* d_costs, float* d_weights) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float min_cost = FLT_MAX;
    for (int k = 0; k < K_SAMPLES; k++) min_cost = fminf(min_cost, d_costs[k]);
    float sum_w = 0.0f;
    for (int k = 0; k < K_SAMPLES; k++) {
        float w = expf(-(d_costs[k] - min_cost) / LAMBDA);
        d_weights[k] = w;
        sum_w += w;
    }
    if (sum_w > 0.0f) {
        for (int k = 0; k < K_SAMPLES; k++) d_weights[k] /= sum_w;
    }
}

__global__ void update_controls_kernel(float* d_nominal, const float* d_perturbed, const float* d_weights) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T_HORIZON) return;
    float ux = 0.0f;
    float uy = 0.0f;
    for (int k = 0; k < K_SAMPLES; k++) {
        float w = d_weights[k];
        ux += w * d_perturbed[k * T_HORIZON * 2 + t * 2 + 0];
        uy += w * d_perturbed[k * T_HORIZON * 2 + t * 2 + 1];
    }
    d_nominal[t * 2 + 0] = ux;
    d_nominal[t * 2 + 1] = uy;
}

static void shift_nominal(vector<float>& nominal) {
    for (int t = 0; t < T_HORIZON - 1; t++) {
        nominal[t * 2 + 0] = nominal[(t + 1) * 2 + 0];
        nominal[t * 2 + 1] = nominal[(t + 1) * 2 + 1];
    }
    nominal[(T_HORIZON - 1) * 2 + 0] = 0.0f;
    nominal[(T_HORIZON - 1) * 2 + 1] = 0.0f;
}

int main() {
    vector<float> train_inputs;
    vector<float> train_targets;
    make_training_set(NeuralSceneKind::LShapeWorld, train_inputs, train_targets);

    GpuMLP mlp(NSDF_INPUT_DIM, NSDF_HIDDEN_DIM, NSDF_HIDDEN_LAYERS, NSDF_OUTPUT_DIM);
    mlp.init_random(888);
    float train_loss = train_neural_sdf(mlp, train_inputs, train_targets, 500, 256, 0.001f,
                                        NSDF_ACTIVATION, 2, nullptr);
    cout << "Training loss: " << train_loss << endl;

    vector<float> weights = mlp.get_weights();
    CUDA_CHECK(cudaMemcpyToSymbol(d_compare_sdf_weights, weights.data(), weights.size() * sizeof(float)));

    vector<float> true_grid = make_true_sdf_grid(NeuralSceneKind::LShapeWorld, NSDF_GRID_RES);
    vector<float> pred_grid = predict_sdf_grid(mlp, NSDF_GRID_RES);
    cv::Mat left_bg = render_sdf_heatmap(true_grid, NSDF_GRID_RES, NeuralSceneKind::LShapeWorld, "Circle Approx MPPI");
    cv::Mat right_bg = render_sdf_heatmap(pred_grid, NSDF_GRID_RES, NeuralSceneKind::LShapeWorld, "Neural SDF MPPI");

    float* d_nominal_circle = nullptr;
    float* d_nominal_neural = nullptr;
    float* d_costs_circle = nullptr;
    float* d_costs_neural = nullptr;
    float* d_pert_circle = nullptr;
    float* d_pert_neural = nullptr;
    float* d_weights_circle = nullptr;
    float* d_weights_neural = nullptr;
    curandState* d_rng_circle = nullptr;
    curandState* d_rng_neural = nullptr;

    CUDA_CHECK(cudaMalloc(&d_nominal_circle, T_HORIZON * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nominal_neural, T_HORIZON * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_costs_circle, K_SAMPLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_costs_neural, K_SAMPLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pert_circle, K_SAMPLES * T_HORIZON * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pert_neural, K_SAMPLES * T_HORIZON * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights_circle, K_SAMPLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights_neural, K_SAMPLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rng_circle, K_SAMPLES * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_rng_neural, K_SAMPLES * sizeof(curandState)));
    CUDA_CHECK(cudaMemset(d_nominal_circle, 0, T_HORIZON * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_nominal_neural, 0, T_HORIZON * 2 * sizeof(float)));

    int threads = 256;
    int blocks = (K_SAMPLES + threads - 1) / threads;
    init_rng<<<blocks, threads>>>(d_rng_circle, K_SAMPLES, 1001ULL);
    init_rng<<<blocks, threads>>>(d_rng_neural, K_SAMPLES, 2002ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<float> h_nom_circle(T_HORIZON * 2);
    vector<float> h_nom_neural(T_HORIZON * 2);
    cv::Point2f circle_state(0.9f, 0.9f);
    cv::Point2f neural_state(0.9f, 0.9f);
    cv::Point2f goal(9.2f, 9.0f);
    vector<cv::Point2f> circle_path(1, circle_state);
    vector<cv::Point2f> neural_path(1, neural_state);

    cv::VideoWriter video(
        AVI_PATH,
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
        15,
        cv::Size(left_bg.cols * 2, left_bg.rows));

    if (!video.isOpened()) {
        fprintf(stderr, "Failed to open %s\n", AVI_PATH);
        return 1;
    }

    for (int step = 0; step < MAX_STEPS; step++) {
        for (int iter = 0; iter < 6; iter++) {
            rollout_kernel<false><<<blocks, threads>>>(circle_state.x, circle_state.y, d_nominal_circle,
                                                       d_costs_circle, d_pert_circle, d_rng_circle);
            rollout_kernel<true><<<blocks, threads>>>(neural_state.x, neural_state.y, d_nominal_neural,
                                                      d_costs_neural, d_pert_neural, d_rng_neural);
            compute_weights_kernel<<<1, 1>>>(d_costs_circle, d_weights_circle);
            compute_weights_kernel<<<1, 1>>>(d_costs_neural, d_weights_neural);
            update_controls_kernel<<<1, T_HORIZON>>>(d_nominal_circle, d_pert_circle, d_weights_circle);
            update_controls_kernel<<<1, T_HORIZON>>>(d_nominal_neural, d_pert_neural, d_weights_neural);
        }

        CUDA_CHECK(cudaMemcpy(h_nom_circle.data(), d_nominal_circle, h_nom_circle.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_nom_neural.data(), d_nominal_neural, h_nom_neural.size() * sizeof(float), cudaMemcpyDeviceToHost));

        circle_state.x += host_clampf(h_nom_circle[0], -MAX_SPEED, MAX_SPEED) * DT;
        circle_state.y += host_clampf(h_nom_circle[1], -MAX_SPEED, MAX_SPEED) * DT;
        neural_state.x += host_clampf(h_nom_neural[0], -MAX_SPEED, MAX_SPEED) * DT;
        neural_state.y += host_clampf(h_nom_neural[1], -MAX_SPEED, MAX_SPEED) * DT;

        circle_state.x = host_clampf(circle_state.x, NSDF_WORLD_MIN, NSDF_WORLD_MAX);
        circle_state.y = host_clampf(circle_state.y, NSDF_WORLD_MIN, NSDF_WORLD_MAX);
        neural_state.x = host_clampf(neural_state.x, NSDF_WORLD_MIN, NSDF_WORLD_MAX);
        neural_state.y = host_clampf(neural_state.y, NSDF_WORLD_MIN, NSDF_WORLD_MAX);

        if (scene_sdf(NeuralSceneKind::LShapeWorld, circle_state.x, circle_state.y) < 0.02f && !circle_path.empty()) {
            circle_state = circle_path.back();
        }
        if (scene_sdf(NeuralSceneKind::LShapeWorld, neural_state.x, neural_state.y) < 0.02f && !neural_path.empty()) {
            neural_state = neural_path.back();
        }

        circle_path.push_back(circle_state);
        neural_path.push_back(neural_state);

        shift_nominal(h_nom_circle);
        shift_nominal(h_nom_neural);
        CUDA_CHECK(cudaMemcpy(d_nominal_circle, h_nom_circle.data(), h_nom_circle.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nominal_neural, h_nom_neural.data(), h_nom_neural.size() * sizeof(float), cudaMemcpyHostToDevice));

        cv::Mat left = left_bg.clone();
        cv::Mat right = right_bg.clone();
        draw_path(left, circle_path, cv::Scalar(255, 255, 255));
        draw_path(right, neural_path, cv::Scalar(255, 255, 255));
        draw_start_goal(left, cv::Point2f(0.9f, 0.9f), goal);
        draw_start_goal(right, cv::Point2f(0.9f, 0.9f), goal);

        char left_buf[96];
        char right_buf[96];
        std::snprintf(left_buf, sizeof(left_buf), "dist=%.2f", hypotf(circle_state.x - goal.x, circle_state.y - goal.y));
        std::snprintf(right_buf, sizeof(right_buf), "dist=%.2f", hypotf(neural_state.x - goal.x, neural_state.y - goal.y));
        cv::putText(left, left_buf, cv::Point(10, left.rows - 12), cv::FONT_HERSHEY_SIMPLEX,
                    0.55, cv::Scalar(255, 255, 255), 1);
        cv::putText(right, right_buf, cv::Point(10, right.rows - 12), cv::FONT_HERSHEY_SIMPLEX,
                    0.55, cv::Scalar(255, 255, 255), 1);

        cv::Mat frame;
        cv::hconcat(left, right, frame);
        video.write(frame);

        if (hypotf(circle_state.x - goal.x, circle_state.y - goal.y) < 0.35f &&
            hypotf(neural_state.x - goal.x, neural_state.y - goal.y) < 0.35f) {
            break;
        }
    }

    video.release();
    CUDA_CHECK(cudaFree(d_nominal_circle));
    CUDA_CHECK(cudaFree(d_nominal_neural));
    CUDA_CHECK(cudaFree(d_costs_circle));
    CUDA_CHECK(cudaFree(d_costs_neural));
    CUDA_CHECK(cudaFree(d_pert_circle));
    CUDA_CHECK(cudaFree(d_pert_neural));
    CUDA_CHECK(cudaFree(d_weights_circle));
    CUDA_CHECK(cudaFree(d_weights_neural));
    CUDA_CHECK(cudaFree(d_rng_circle));
    CUDA_CHECK(cudaFree(d_rng_neural));

    cout << "Video saved to gif/comparison_sdf_nav.avi" << endl;
    convert_avi_to_gif(AVI_PATH, GIF_PATH, 15);
    cout << "GIF saved to gif/comparison_sdf_nav.gif" << endl;
    return 0;
}
