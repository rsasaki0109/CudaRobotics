/*************************************************************************
    MiniIsaacGym RL
    - GPU CartPole rollouts + REINFORCE updates
    - Policy network uses the shared-width GpuMLP backend: 4 -> 32 -> 32 -> 1
    Output: gif/mini_isaac_rl.gif
 ************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#include "gpu_mlp.cuh"
#include "parallel_env.cuh"

using namespace std;
using namespace cudabot;

static const char* AVI_PATH = "gif/mini_isaac_rl.avi";
static const char* GIF_PATH = "gif/mini_isaac_rl.gif";

static const int RL_ENVS = 1024;
static const int RL_HORIZON = PARALLEL_CARTPOLE_MAX_STEPS;
static const int RL_GENERATIONS = 160;
static const int POLICY_HIDDEN = 32;
static const float GAMMA = 0.99f;
static const float LR = 0.003f;

static void convert_avi_to_gif(const char* avi_path, const char* gif_path, int fps = 15) {
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -i %s -vf 'fps=%d,scale=720:-1' -loop 0 %s 2>/dev/null",
                  avi_path, fps, gif_path);
    std::system(cmd);
}

__global__ void sample_policy_kernel(const float* d_logits, float* d_actions, float* d_action_bits,
                                     int n_envs, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_envs) return;

    float logit = d_logits[idx];
    float p = 1.0f / (1.0f + expf(-logit));
    float u = parallel_env_rand01(idx, seed);
    float bit = (u < p) ? 1.0f : 0.0f;
    d_action_bits[idx] = bit;
    d_actions[idx] = bit > 0.5f ? 1.0f : -1.0f;
}

__global__ void deterministic_policy_kernel(const float* d_logits, float* d_actions, int n_envs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_envs) return;
    d_actions[idx] = d_logits[idx] >= 0.0f ? 1.0f : -1.0f;
}

__global__ void compute_returns_kernel(
    const float* d_rewards,
    const int* d_done,
    float* d_returns,
    int horizon,
    int n_envs,
    float gamma)
{
    int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= n_envs) return;

    float running = 0.0f;
    for (int t = horizon - 1; t >= 0; t--) {
        int idx = t * n_envs + env;
        running = d_rewards[idx] + gamma * running * (1 - d_done[idx]);
        d_returns[idx] = running;
    }
}

__global__ void compute_stats_kernel(const float* d_values, int total, float* d_stats) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double sum = 0.0;
    double sq_sum = 0.0;
    for (int i = 0; i < total; i++) {
        double v = d_values[i];
        sum += v;
        sq_sum += v * v;
    }
    double mean = sum / total;
    double var = sq_sum / total - mean * mean;
    if (var < 1.0e-8) var = 1.0e-8;
    d_stats[0] = static_cast<float>(mean);
    d_stats[1] = static_cast<float>(sqrt(var));
}

__global__ void shape_rewards_kernel(
    const float* d_obs,
    const int* d_prev_done,
    const int* d_dones,
    float* d_rewards,
    int n_envs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_envs) return;

    if (d_prev_done[idx]) {
        d_rewards[idx] = 0.0f;
        return;
    }

    float x = fabsf(d_obs[idx * 4 + 0]);
    float theta = fabsf(d_obs[idx * 4 + 2]);
    float reward = 1.25f - 1.6f * theta - 0.12f * x;
    if (d_dones[idx]) reward -= 1.2f;
    d_rewards[idx] = reward;
}

__global__ void build_policy_grad_kernel(
    const float* d_logits,
    const float* d_action_bits,
    const float* d_returns,
    const int* d_prev_done,
    const float* d_stats,
    float* d_output_grad,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    if (d_prev_done[idx] != 0 || d_returns[idx] <= 0.0f) {
        d_output_grad[idx] = 0.0f;
        return;
    }

    float mean = d_stats[0];
    float stddev = d_stats[1];
    float adv = (d_returns[idx] - mean) / (stddev + 1.0e-6f);
    adv = fminf(fmaxf(adv, -5.0f), 5.0f);
    float p = 1.0f / (1.0f + expf(-d_logits[idx]));
    float action = d_action_bits[idx];
    d_output_grad[idx] = (action - p) * adv;
}

static void draw_cartpole(cv::Mat& img, const float* state) {
    img.setTo(cv::Scalar(20, 24, 34));
    int ground_y = 250;
    cv::line(img, cv::Point(20, ground_y), cv::Point(img.cols - 20, ground_y), cv::Scalar(200, 200, 200), 2);

    float x = state[0];
    float theta = state[2];
    int cart_x = static_cast<int>(img.cols / 2 + x / 2.4f * 180.0f);
    int cart_y = ground_y - 18;
    cv::rectangle(img, cv::Rect(cart_x - 28, cart_y - 12, 56, 24), cv::Scalar(90, 170, 255), -1);

    cv::Point pivot(cart_x, cart_y - 2);
    float pole_len = 118.0f;
    cv::Point end(static_cast<int>(pivot.x + pole_len * sinf(theta)),
                  static_cast<int>(pivot.y - pole_len * cosf(theta)));
    cv::line(img, pivot, end, cv::Scalar(255, 220, 90), 6);
    cv::circle(img, pivot, 6, cv::Scalar(240, 240, 240), -1);
}

static void draw_curve(cv::Mat& img, const vector<float>& history, int upto) {
    img.setTo(cv::Scalar(28, 22, 26));
    cv::putText(img, "Learning Curve", cv::Point(20, 28), cv::FONT_HERSHEY_SIMPLEX,
                0.8, cv::Scalar(255, 255, 255), 2);
    if (history.empty()) return;

    float ymax = 1.0f;
    for (int i = 0; i < upto; i++) ymax = max(ymax, history[i]);
    int left = 50, right = img.cols - 20, top = 50, bottom = img.rows - 40;
    cv::rectangle(img, cv::Rect(left, top, right - left, bottom - top), cv::Scalar(100, 100, 100), 1);

    for (int i = 1; i < upto; i++) {
        float x0 = left + (right - left) * (static_cast<float>(i - 1) / max(1, RL_GENERATIONS - 1));
        float x1 = left + (right - left) * (static_cast<float>(i) / max(1, RL_GENERATIONS - 1));
        float y0 = bottom - (bottom - top) * (history[i - 1] / ymax);
        float y1 = bottom - (bottom - top) * (history[i] / ymax);
        cv::line(img, cv::Point(static_cast<int>(x0), static_cast<int>(y0)),
                 cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                 cv::Scalar(100, 220, 150), 2);
    }
}

int main() {
    ParallelEnv env(RL_ENVS, 0);
    GpuMLP policy(4, POLICY_HIDDEN, 2, 1);
    policy.init_random(2026);

    {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> x_dist(-2.4f, 2.4f);
        std::uniform_real_distribution<float> xd_dist(-2.0f, 2.0f);
        std::uniform_real_distribution<float> th_dist(-0.25f, 0.25f);
        std::uniform_real_distribution<float> thd_dist(-2.5f, 2.5f);
        const int warm_samples = 8192;
        vector<float> h_input(warm_samples * 4);
        vector<float> h_target(warm_samples);
        for (int i = 0; i < warm_samples; i++) {
            float x = x_dist(rng);
            float x_dot = xd_dist(rng);
            float theta = th_dist(rng);
            float theta_dot = thd_dist(rng);
            h_input[i * 4 + 0] = x;
            h_input[i * 4 + 1] = x_dot;
            h_input[i * 4 + 2] = theta;
            h_input[i * 4 + 3] = theta_dot;
            float teacher = 10.0f * theta + 1.0f * theta_dot + 1.0f * x + 1.0f * x_dot;
            h_target[i] = teacher >= 0.0f ? 4.0f : -4.0f;
        }

        float* d_warm_input = nullptr;
        float* d_warm_target = nullptr;
        cudaMalloc(&d_warm_input, h_input.size() * sizeof(float));
        cudaMalloc(&d_warm_target, h_target.size() * sizeof(float));
        cudaMemcpy(d_warm_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_warm_target, h_target.data(), h_target.size() * sizeof(float), cudaMemcpyHostToDevice);

        for (int i = 0; i < 600; i++) {
            policy.train_step_backprop(d_warm_input, d_warm_target, warm_samples, 0.005f, 1);
        }

        cudaFree(d_warm_input);
        cudaFree(d_warm_target);
    }

    float* d_obs = nullptr;
    float* d_logits = nullptr;
    float* d_actions = nullptr;
    float* d_action_bits = nullptr;
    float* d_rewards = nullptr;
    int* d_dones = nullptr;
    float* d_obs_hist = nullptr;
    float* d_logits_hist = nullptr;
    float* d_action_hist = nullptr;
    float* d_rewards_hist = nullptr;
    int* d_done_hist = nullptr;
    int* d_prev_done = nullptr;
    int* d_prev_done_hist = nullptr;
    float* d_returns = nullptr;
    float* d_output_grad = nullptr;
    float* d_stats = nullptr;

    int total_samples = RL_ENVS * RL_HORIZON;
    cudaMalloc(&d_obs, RL_ENVS * 4 * sizeof(float));
    cudaMalloc(&d_logits, RL_ENVS * sizeof(float));
    cudaMalloc(&d_actions, RL_ENVS * sizeof(float));
    cudaMalloc(&d_action_bits, RL_ENVS * sizeof(float));
    cudaMalloc(&d_rewards, RL_ENVS * sizeof(float));
    cudaMalloc(&d_dones, RL_ENVS * sizeof(int));
    cudaMalloc(&d_obs_hist, total_samples * 4 * sizeof(float));
    cudaMalloc(&d_logits_hist, total_samples * sizeof(float));
    cudaMalloc(&d_action_hist, total_samples * sizeof(float));
    cudaMalloc(&d_rewards_hist, total_samples * sizeof(float));
    cudaMalloc(&d_done_hist, total_samples * sizeof(int));
    cudaMalloc(&d_prev_done, RL_ENVS * sizeof(int));
    cudaMalloc(&d_prev_done_hist, total_samples * sizeof(int));
    cudaMalloc(&d_returns, total_samples * sizeof(float));
    cudaMalloc(&d_output_grad, total_samples * sizeof(float));
    cudaMalloc(&d_stats, 2 * sizeof(float));

    int threads = 256;
    int blocks = (RL_ENVS + threads - 1) / threads;
    vector<float> history;
    history.reserve(RL_GENERATIONS);
    vector<int> h_steps(RL_ENVS);
    float last_mean = 0.0f;

    for (int gen = 0; gen < RL_GENERATIONS; gen++) {
        env.reset_all(gen + 1);
        for (int t = 0; t < RL_HORIZON; t++) {
            env.observe(d_obs);
            policy.forward_batch(d_obs, d_logits, RL_ENVS, 1);
            sample_policy_kernel<<<blocks, threads>>>(d_logits, d_actions, d_action_bits, RL_ENVS,
                                                      static_cast<unsigned int>(gen * 4099 + t * 17 + 3));
            cudaMemcpy(d_prev_done, env.done_device(), RL_ENVS * sizeof(int), cudaMemcpyDeviceToDevice);
            env.step(d_actions, d_obs, d_rewards, d_dones);
            shape_rewards_kernel<<<blocks, threads>>>(d_obs, d_prev_done, d_dones, d_rewards, RL_ENVS);

            cudaMemcpy(d_obs_hist + t * RL_ENVS * 4, d_obs, RL_ENVS * 4 * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_logits_hist + t * RL_ENVS, d_logits, RL_ENVS * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_action_hist + t * RL_ENVS, d_action_bits, RL_ENVS * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_rewards_hist + t * RL_ENVS, d_rewards, RL_ENVS * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_done_hist + t * RL_ENVS, d_dones, RL_ENVS * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_prev_done_hist + t * RL_ENVS, d_prev_done, RL_ENVS * sizeof(int), cudaMemcpyDeviceToDevice);
        }
        cudaDeviceSynchronize();

        compute_returns_kernel<<<blocks, threads>>>(d_rewards_hist, d_done_hist, d_returns,
                                                    RL_HORIZON, RL_ENVS, GAMMA);
        compute_stats_kernel<<<1, 1>>>(d_returns, total_samples, d_stats);
        int sample_blocks = (total_samples + threads - 1) / threads;
        build_policy_grad_kernel<<<sample_blocks, threads>>>(
            d_logits_hist, d_action_hist, d_returns, d_prev_done_hist, d_stats, d_output_grad, total_samples);
        policy.apply_output_grad(d_obs_hist, d_output_grad, total_samples, LR, 1);

        cudaMemcpy(h_steps.data(), env.steps_device(), RL_ENVS * sizeof(int), cudaMemcpyDeviceToHost);
        last_mean = accumulate(h_steps.begin(), h_steps.end(), 0.0f) / RL_ENVS;
        history.push_back(last_mean);
        cout << "Generation " << gen + 1 << " / " << RL_GENERATIONS
             << " mean_steps=" << last_mean << endl;
        if (last_mean > 180.0f) break;
    }

    ParallelEnv eval_env(1, 0);
    float* d_eval_obs = nullptr;
    float* d_eval_logits = nullptr;
    float* d_eval_actions = nullptr;
    cudaMalloc(&d_eval_obs, 4 * sizeof(float));
    cudaMalloc(&d_eval_logits, sizeof(float));
    cudaMalloc(&d_eval_actions, sizeof(float));
    eval_env.reset_all(999);

    cv::VideoWriter video(
        AVI_PATH,
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
        15,
        cv::Size(960, 360));

    if (!video.isOpened()) {
        fprintf(stderr, "Failed to open %s\n", AVI_PATH);
        return 1;
    }

    float eval_state[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int t = 0; t < RL_HORIZON; t++) {
        eval_env.observe(d_eval_obs);
        policy.forward_batch(d_eval_obs, d_eval_logits, 1, 1);
        deterministic_policy_kernel<<<1, 1>>>(d_eval_logits, d_eval_actions, 1);
        eval_env.step(d_eval_actions, nullptr, nullptr, nullptr);
        cudaMemcpy(eval_state, eval_env.states_device(), 4 * sizeof(float), cudaMemcpyDeviceToHost);

        cv::Mat left(360, 420, CV_8UC3);
        cv::Mat right(360, 540, CV_8UC3);
        draw_cartpole(left, eval_state);
        draw_curve(right, history, min(static_cast<int>(history.size()), max(1, t * RL_GENERATIONS / RL_HORIZON)));
        cv::putText(left, "Policy Evaluation", cv::Point(20, 28), cv::FONT_HERSHEY_SIMPLEX,
                    0.78, cv::Scalar(255, 255, 255), 2);

        char buf[128];
        std::snprintf(buf, sizeof(buf), "final mean_steps=%.1f", last_mean);
        cv::putText(left, buf, cv::Point(20, 330), cv::FONT_HERSHEY_SIMPLEX,
                    0.55, cv::Scalar(230, 230, 230), 1);

        cv::Mat frame;
        cv::hconcat(left, right, frame);
        video.write(frame);
    }

    video.release();
    cudaFree(d_obs);
    cudaFree(d_logits);
    cudaFree(d_actions);
    cudaFree(d_action_bits);
    cudaFree(d_rewards);
    cudaFree(d_dones);
    cudaFree(d_obs_hist);
    cudaFree(d_logits_hist);
    cudaFree(d_action_hist);
    cudaFree(d_rewards_hist);
    cudaFree(d_done_hist);
    cudaFree(d_prev_done);
    cudaFree(d_prev_done_hist);
    cudaFree(d_returns);
    cudaFree(d_output_grad);
    cudaFree(d_stats);
    cudaFree(d_eval_obs);
    cudaFree(d_eval_logits);
    cudaFree(d_eval_actions);

    cout << "Video saved to gif/mini_isaac_rl.avi" << endl;
    convert_avi_to_gif(AVI_PATH, GIF_PATH, 15);
    cout << "GIF saved to gif/mini_isaac_rl.gif" << endl;
    return 0;
}
