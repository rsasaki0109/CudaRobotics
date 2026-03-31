/*************************************************************************
    Neural-SDF MPPI
    - MPPI cost uses a learned neural signed distance field
    - MLP weights are copied into constant memory for rollout evaluation
    Output: gif/sdf_mppi.gif
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

static const char* AVI_PATH = "gif/sdf_mppi.avi";
static const char* GIF_PATH = "gif/sdf_mppi.gif";

static const int K_SAMPLES = 4096;
static const int T_HORIZON = 30;
static const int MAX_STEPS = 170;
static const float DT = 0.16f;
static const float MAX_SPEED = 1.4f;
static const float LAMBDA = 2.2f;
static const int SDF_WEIGHT_CAP = 12000;

__constant__ float d_sdf_weights[SDF_WEIGHT_CAP];

__device__ float eval_constant_sdf(float x, float y) {
    float input[2];
    float output[1];
    float scratch[NSDF_HIDDEN_DIM * 2];
    input[0] = 2.0f * (x - NSDF_WORLD_MIN) / NSDF_WORLD_SIZE - 1.0f;
    input[1] = 2.0f * (y - NSDF_WORLD_MIN) / NSDF_WORLD_SIZE - 1.0f;
    mlp_forward(d_sdf_weights, input, NSDF_INPUT_DIM, output, NSDF_OUTPUT_DIM,
                NSDF_HIDDEN_DIM, NSDF_HIDDEN_LAYERS, scratch, NSDF_ACTIVATION);
    return output[0];
}

__global__ void init_rng(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

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
        float ux = d_nominal[t * 2 + 0] + 0.45f * curand_normal(&rng);
        float uy = d_nominal[t * 2 + 1] + 0.45f * curand_normal(&rng);
        ux = fminf(fmaxf(ux, -MAX_SPEED), MAX_SPEED);
        uy = fminf(fmaxf(uy, -MAX_SPEED), MAX_SPEED);
        d_perturbed[k * T_HORIZON * 2 + t * 2 + 0] = ux;
        d_perturbed[k * T_HORIZON * 2 + t * 2 + 1] = uy;

        x += ux * DT;
        y += uy * DT;
        float sdf = eval_constant_sdf(x, y);
        float dx = x - 9.1f;
        float dy = y - 9.0f;
        total_cost += 1.7f * sqrtf(dx * dx + dy * dy + 1.0e-4f);
        total_cost += 0.18f * (ux * ux + uy * uy);
        if (sdf < 0.8f) {
            float margin = fmaxf(sdf, 0.03f);
            float inv = 1.0f / margin - 1.0f / 0.8f;
            total_cost += 3.8f * inv * inv;
        }
        if (sdf < 0.0f) total_cost += 120.0f;
        if (x < NSDF_WORLD_MIN || x > NSDF_WORLD_MAX || y < NSDF_WORLD_MIN || y > NSDF_WORLD_MAX) {
            total_cost += 80.0f;
        }
    }

    float dx = x - 9.1f;
    float dy = y - 9.0f;
    total_cost += 10.0f * sqrtf(dx * dx + dy * dy + 1.0e-4f);
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

int main() {
    vector<float> train_inputs;
    vector<float> train_targets;
    make_training_set(NeuralSceneKind::DemoWorld, train_inputs, train_targets);

    GpuMLP mlp(NSDF_INPUT_DIM, NSDF_HIDDEN_DIM, NSDF_HIDDEN_LAYERS, NSDF_OUTPUT_DIM);
    mlp.init_random(77);
    float train_loss = train_neural_sdf(mlp, train_inputs, train_targets, 500, 256, 0.001f,
                                        NSDF_ACTIVATION, 2, nullptr);
    cout << "Training loss: " << train_loss << endl;

    vector<float> weights = mlp.get_weights();
    CUDA_CHECK(cudaMemcpyToSymbol(d_sdf_weights, weights.data(), weights.size() * sizeof(float)));

    vector<float> background_grid = predict_sdf_grid(mlp, NSDF_GRID_RES);
    cv::Mat background = render_sdf_heatmap(background_grid, NSDF_GRID_RES, NeuralSceneKind::DemoWorld,
                                            "Neural SDF MPPI");

    float* d_nominal = nullptr;
    float* d_costs = nullptr;
    float* d_perturbed = nullptr;
    float* d_weights = nullptr;
    curandState* d_rng = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nominal, T_HORIZON * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_costs, K_SAMPLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_perturbed, K_SAMPLES * T_HORIZON * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, K_SAMPLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rng, K_SAMPLES * sizeof(curandState)));
    CUDA_CHECK(cudaMemset(d_nominal, 0, T_HORIZON * 2 * sizeof(float)));

    int threads = 256;
    int blocks = (K_SAMPLES + threads - 1) / threads;
    init_rng<<<blocks, threads>>>(d_rng, K_SAMPLES, 2026ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    cv::Point2f state(0.9f, 0.8f);
    cv::Point2f goal(9.1f, 9.0f);
    vector<cv::Point2f> path;
    path.push_back(state);
    vector<float> h_nominal(T_HORIZON * 2);

    cv::VideoWriter video(
        AVI_PATH,
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
        15,
        cv::Size(background.cols, background.rows));

    if (!video.isOpened()) {
        fprintf(stderr, "Failed to open %s\n", AVI_PATH);
        return 1;
    }

    for (int step = 0; step < MAX_STEPS; step++) {
        for (int iter = 0; iter < 6; iter++) {
            rollout_kernel<<<blocks, threads>>>(state.x, state.y, d_nominal, d_costs, d_perturbed, d_rng);
            compute_weights_kernel<<<1, 1>>>(d_costs, d_weights);
            update_controls_kernel<<<1, T_HORIZON>>>(d_nominal, d_perturbed, d_weights);
        }

        CUDA_CHECK(cudaMemcpy(h_nominal.data(), d_nominal, h_nominal.size() * sizeof(float), cudaMemcpyDeviceToHost));
        state.x += host_clampf(h_nominal[0], -MAX_SPEED, MAX_SPEED) * DT;
        state.y += host_clampf(h_nominal[1], -MAX_SPEED, MAX_SPEED) * DT;
        state.x = host_clampf(state.x, NSDF_WORLD_MIN, NSDF_WORLD_MAX);
        state.y = host_clampf(state.y, NSDF_WORLD_MIN, NSDF_WORLD_MAX);
        if (scene_sdf(NeuralSceneKind::DemoWorld, state.x, state.y) < 0.02f && path.size() >= 2) {
            state = path.back();
        }
        path.push_back(state);

        for (int t = 0; t < T_HORIZON - 1; t++) {
            h_nominal[t * 2 + 0] = h_nominal[(t + 1) * 2 + 0];
            h_nominal[t * 2 + 1] = h_nominal[(t + 1) * 2 + 1];
        }
        h_nominal[(T_HORIZON - 1) * 2 + 0] = 0.0f;
        h_nominal[(T_HORIZON - 1) * 2 + 1] = 0.0f;
        CUDA_CHECK(cudaMemcpy(d_nominal, h_nominal.data(), h_nominal.size() * sizeof(float), cudaMemcpyHostToDevice));

        cv::Mat frame = background.clone();
        draw_path(frame, path, cv::Scalar(255, 255, 255));
        draw_start_goal(frame, cv::Point2f(0.9f, 0.8f), goal);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "step=%d dist=%.2f", step,
                      hypotf(state.x - goal.x, state.y - goal.y));
        cv::putText(frame, buf, cv::Point(10, frame.rows - 12), cv::FONT_HERSHEY_SIMPLEX,
                    0.55, cv::Scalar(255, 255, 255), 1);
        video.write(frame);

        if (hypotf(state.x - goal.x, state.y - goal.y) < 0.35f) break;
    }

    video.release();
    CUDA_CHECK(cudaFree(d_nominal));
    CUDA_CHECK(cudaFree(d_costs));
    CUDA_CHECK(cudaFree(d_perturbed));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_rng));

    cout << "Video saved to gif/sdf_mppi.avi" << endl;
    convert_avi_to_gif(AVI_PATH, GIF_PATH, 15);
    cout << "GIF saved to gif/sdf_mppi.gif" << endl;
    return 0;
}
