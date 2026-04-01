/*************************************************************************
    MiniIsaacGym
    - Simulate 4096 CartPole environments in parallel on GPU
    - Visualize one representative environment and the reward/step histogram
    Output: gif/mini_isaac.gif
 ************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <vector>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#include "parallel_env.cuh"

using namespace std;
using namespace cudabot;

static const char* AVI_PATH = "gif/mini_isaac.avi";
static const char* GIF_PATH = "gif/mini_isaac.gif";
static const int N_ENVS = 4096;
static const int FRAMES = 220;
static const int STEPS_PER_FRAME = 3;

static void convert_avi_to_gif(const char* avi_path, const char* gif_path, int fps = 15) {
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -i %s -vf 'fps=%d,scale=640:-1' -loop 0 %s 2>/dev/null",
                  avi_path, fps, gif_path);
    std::system(cmd);
}

__global__ void heuristic_actions_kernel(const float* d_states, float* d_actions, int n_envs, int frame_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_envs) return;

    float x = d_states[idx * 4 + 0];
    float x_dot = d_states[idx * 4 + 1];
    float theta = d_states[idx * 4 + 2];
    float theta_dot = d_states[idx * 4 + 3];
    float u = 2.6f * theta + 0.9f * theta_dot + 0.18f * x + 0.10f * x_dot;
    float dither = 0.10f * sinf(0.03f * (frame_idx + idx));
    d_actions[idx] = fminf(fmaxf(u + dither, -1.0f), 1.0f);
}

static void draw_cartpole(cv::Mat& img, const float* state) {
    img.setTo(cv::Scalar(18, 24, 32));
    int ground_y = 250;
    cv::line(img, cv::Point(20, ground_y), cv::Point(img.cols - 20, ground_y), cv::Scalar(180, 180, 180), 2);

    float x = state[0];
    float theta = state[2];
    int cart_x = static_cast<int>(img.cols / 2 + x / 2.4f * 180.0f);
    int cart_y = ground_y - 18;
    cv::Rect cart(cart_x - 28, cart_y - 12, 56, 24);
    cv::rectangle(img, cart, cv::Scalar(70, 160, 255), -1);

    cv::Point pivot(cart_x, cart_y - 2);
    float pole_len = 120.0f;
    cv::Point pole_end(
        static_cast<int>(pivot.x + pole_len * sinf(theta)),
        static_cast<int>(pivot.y - pole_len * cosf(theta)));
    cv::line(img, pivot, pole_end, cv::Scalar(255, 220, 80), 6);
    cv::circle(img, pivot, 6, cv::Scalar(240, 240, 240), -1);
}

static void draw_histogram(cv::Mat& img, const vector<int>& steps) {
    img.setTo(cv::Scalar(26, 20, 24));
    const int bins = 20;
    vector<int> counts(bins, 0);
    for (int step : steps) {
        int bin = min(bins - 1, step * bins / PARALLEL_CARTPOLE_MAX_STEPS);
        counts[bin]++;
    }
    int max_count = *max_element(counts.begin(), counts.end());
    max_count = max(max_count, 1);

    for (int i = 0; i < bins; i++) {
        int x0 = 20 + i * (img.cols - 40) / bins;
        int x1 = 20 + (i + 1) * (img.cols - 40) / bins - 4;
        int h = static_cast<int>((img.rows - 70) * (static_cast<float>(counts[i]) / max_count));
        cv::rectangle(img, cv::Rect(x0, img.rows - 30 - h, max(2, x1 - x0), h),
                      cv::Scalar(90, 200, 140), -1);
    }

    cv::putText(img, "Survival Histogram", cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX,
                0.75, cv::Scalar(255, 255, 255), 2);
    cv::putText(img, "episode steps", cv::Point(20, img.rows - 8), cv::FONT_HERSHEY_SIMPLEX,
                0.55, cv::Scalar(220, 220, 220), 1);
}

int main() {
    ParallelEnv env(N_ENVS, 0);

    float* d_actions = nullptr;
    float* d_obs = nullptr;
    float* d_rewards = nullptr;
    int* d_dones = nullptr;
    cudaMalloc(&d_actions, N_ENVS * sizeof(float));
    cudaMalloc(&d_obs, N_ENVS * 4 * sizeof(float));
    cudaMalloc(&d_rewards, N_ENVS * sizeof(float));
    cudaMalloc(&d_dones, N_ENVS * sizeof(int));

    vector<int> h_steps(N_ENVS);
    vector<int> h_dones(N_ENVS);
    float h_state[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    cv::VideoWriter video(
        AVI_PATH,
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
        15,
        cv::Size(960, 360));

    if (!video.isOpened()) {
        fprintf(stderr, "Failed to open %s\n", AVI_PATH);
        return 1;
    }

    int threads = 256;
    int blocks = (N_ENVS + threads - 1) / threads;

    for (int frame_idx = 0; frame_idx < FRAMES; frame_idx++) {
        for (int sub = 0; sub < STEPS_PER_FRAME; sub++) {
            heuristic_actions_kernel<<<blocks, threads>>>(env.states_device(), d_actions, N_ENVS,
                                                          frame_idx * STEPS_PER_FRAME + sub);
            env.step(d_actions, d_obs, d_rewards, d_dones);
        }
        cudaDeviceSynchronize();

        cudaMemcpy(h_state, env.states_device(), 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_steps.data(), env.steps_device(), N_ENVS * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dones.data(), env.done_device(), N_ENVS * sizeof(int), cudaMemcpyDeviceToHost);

        int done_count = accumulate(h_dones.begin(), h_dones.end(), 0);
        float mean_steps = accumulate(h_steps.begin(), h_steps.end(), 0.0f) / N_ENVS;

        cv::Mat left(360, 420, CV_8UC3);
        cv::Mat right(360, 540, CV_8UC3);
        draw_cartpole(left, h_state);
        draw_histogram(right, h_steps);

        char buf[128];
        std::snprintf(buf, sizeof(buf), "envs=%d  mean_steps=%.1f  done=%d", N_ENVS, mean_steps, done_count);
        cv::putText(left, "Representative CartPole", cv::Point(20, 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.72, cv::Scalar(255, 255, 255), 2);
        cv::putText(left, buf, cv::Point(20, 330), cv::FONT_HERSHEY_SIMPLEX,
                    0.52, cv::Scalar(230, 230, 230), 1);

        cv::Mat frame;
        cv::hconcat(left, right, frame);
        video.write(frame);

        if (done_count > static_cast<int>(0.88f * N_ENVS)) {
            env.reset_all(frame_idx + 99);
        }
    }

    video.release();
    cudaFree(d_actions);
    cudaFree(d_obs);
    cudaFree(d_rewards);
    cudaFree(d_dones);

    cout << "Video saved to gif/mini_isaac.avi" << endl;
    convert_avi_to_gif(AVI_PATH, GIF_PATH, 15);
    cout << "GIF saved to gif/mini_isaac.gif" << endl;
    return 0;
}
