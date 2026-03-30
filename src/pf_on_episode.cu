/*************************************************************************
    > File Name: pf_on_episode.cu
    > CUDA-parallelized Particle Filter on Episode (Ueda arXiv 2019)
    > TEACH phase: record figure-8 trajectory with (x,y,yaw,lidar,v,omega)
    > REPLAY phase: particle filter on episode timeline
    > CUDA kernels: predict_episode, compute_weight, weighted_action, resample
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

// ---------------------------------------------------------------------------
// CUDA error check macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
#define PI 3.141592653f
#define DT 0.2f

#define N_TEACH_STEPS 200
#define N_PARTICLES 500

// Lidar
#define NUM_BEAMS 36
#define MAX_RANGE 10.0f

// Motion noise
#define NOISE_V 0.15f
#define NOISE_OMEGA 0.1f

// Visualization
#define IMG_W 800
#define IMG_H 800
#define SCALE 30.0f
#define OX 400
#define OY 400

// ---------------------------------------------------------------------------
// Episode step data
// ---------------------------------------------------------------------------
struct EpisodeStep {
    float x, y, yaw;
    float v, omega;
    float beams[NUM_BEAMS];
};

// ---------------------------------------------------------------------------
// Kernel: init cuRAND
// ---------------------------------------------------------------------------
__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}

// ---------------------------------------------------------------------------
// Kernel: predict episode - each particle is an index into the episode timeline
// Particle state: (episode_index as float, offset_x, offset_y, offset_yaw)
// For simplicity, each particle stores a float episode time index
// ---------------------------------------------------------------------------
__global__ void predict_episode_kernel(
    float* p_time,      // particle's episode time index [NP]
    float* p_offset_x,  // position offset from episode [NP]
    float* p_offset_y,
    float* p_offset_yaw,
    float dt_advance,    // how much to advance in episode time
    curandState* rng_states, int np, int max_steps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;

    curandState local_rng = rng_states[idx];

    // Advance episode time with noise
    float time_noise = curand_normal(&local_rng) * 0.5f;
    p_time[idx] += dt_advance + time_noise;

    // Clamp
    if (p_time[idx] < 0.0f) p_time[idx] = 0.0f;
    if (p_time[idx] >= (float)(max_steps - 1)) p_time[idx] = (float)(max_steps - 2);

    // Add position noise
    p_offset_x[idx] += curand_normal(&local_rng) * 0.05f;
    p_offset_y[idx] += curand_normal(&local_rng) * 0.05f;
    p_offset_yaw[idx] += curand_normal(&local_rng) * 0.02f;

    // Decay offsets
    p_offset_x[idx] *= 0.95f;
    p_offset_y[idx] *= 0.95f;
    p_offset_yaw[idx] *= 0.95f;

    rng_states[idx] = local_rng;
}

// ---------------------------------------------------------------------------
// Kernel: compute weight based on lidar similarity to episode
// ---------------------------------------------------------------------------
__global__ void compute_weight_kernel(
    const float* p_time,
    const float* p_offset_x, const float* p_offset_y,
    float* pw,
    const float* episode_beams,  // [N_TEACH_STEPS * NUM_BEAMS]
    const float* current_beams,  // [NUM_BEAMS]
    int np, int max_steps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;

    int t_idx = (int)p_time[idx];
    if (t_idx < 0) t_idx = 0;
    if (t_idx >= max_steps) t_idx = max_steps - 1;

    // Compare current lidar with episode lidar at particle's time
    float log_w = 0.0f;
    float sigma = 1.0f;

    for (int b = 0; b < NUM_BEAMS; b++) {
        float ep_range = episode_beams[t_idx * NUM_BEAMS + b];
        float cur_range = current_beams[b];
        float diff = ep_range - cur_range;
        log_w += -0.5f * diff * diff / (sigma * sigma);
    }

    // Penalize position offset
    float offset_penalty = p_offset_x[idx] * p_offset_x[idx] +
                           p_offset_y[idx] * p_offset_y[idx];
    log_w -= 0.5f * offset_penalty;

    pw[idx] = expf(log_w);
}

// ---------------------------------------------------------------------------
// Kernel: compute weighted action from episode
// Each particle votes for the action at its episode time
// ---------------------------------------------------------------------------
__global__ void weighted_action_kernel(
    const float* p_time, const float* pw,
    const float* episode_v, const float* episode_omega,
    float* out_v, float* out_omega,
    int np, int max_steps)
{
    extern __shared__ float sdata[];
    float* sv = sdata;
    float* so = sdata + blockDim.x;
    float* sw = sdata + 2 * blockDim.x;
    int tid = threadIdx.x;

    float wv = 0, wo = 0, ww = 0;
    for (int i = tid; i < np; i += blockDim.x) {
        int t_idx = (int)p_time[i];
        if (t_idx < 0) t_idx = 0;
        if (t_idx >= max_steps - 1) t_idx = max_steps - 2;

        float w = pw[i];
        wv += w * episode_v[t_idx];
        wo += w * episode_omega[t_idx];
        ww += w;
    }
    sv[tid] = wv; so[tid] = wo; sw[tid] = ww;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sv[tid] += sv[tid + s];
            so[tid] += so[tid + s];
            sw[tid] += sw[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_w = sw[0];
        if (total_w < 1e-30f) total_w = 1e-30f;
        *out_v = sv[0] / total_w;
        *out_omega = so[0] / total_w;
    }
}

// ---------------------------------------------------------------------------
// Kernel: normalize weights
// ---------------------------------------------------------------------------
__global__ void normalize_weights_kernel(float* pw, int np) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float val = 0.0f;
    for (int i = tid; i < np; i += blockDim.x) val += pw[i];
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float total = sdata[0];
    if (total < 1e-30f) total = 1e-30f;
    for (int i = tid; i < np; i += blockDim.x) pw[i] /= total;
}

// ---------------------------------------------------------------------------
// Kernel: cumulative sum
// ---------------------------------------------------------------------------
__global__ void cumsum_kernel(const float* pw, float* wcum, int np) {
    wcum[0] = pw[0];
    for (int i = 1; i < np; i++) wcum[i] = wcum[i - 1] + pw[i];
}

// ---------------------------------------------------------------------------
// Kernel: systematic resampling
// ---------------------------------------------------------------------------
__global__ void resample_kernel(
    const float* p_time_in, const float* ox_in, const float* oy_in, const float* oyaw_in,
    float* p_time_out, float* ox_out, float* oy_out, float* oyaw_out,
    const float* wcum, float base_step, float rand_offset, int np)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    float target = base_step * ip + rand_offset;
    int lo = 0, hi = np - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (wcum[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    p_time_out[ip] = p_time_in[lo];
    ox_out[ip] = ox_in[lo];
    oy_out[ip] = oy_in[lo];
    oyaw_out[ip] = oyaw_in[lo];
}

// ---------------------------------------------------------------------------
// Host: visualization
// ---------------------------------------------------------------------------
cv::Point2i to_px(float x, float y) {
    return cv::Point2i((int)(x * SCALE) + OX, IMG_H - (int)(y * SCALE) - OY);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "Particle Filter on Episode (Ueda arXiv 2019)" << std::endl;

    std::mt19937 gen(42);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    // ======================================================================
    // TEACH phase: record figure-8 trajectory
    // ======================================================================
    std::vector<EpisodeStep> episode(N_TEACH_STEPS);
    {
        float x = 0, y = 0, yaw = 0;
        for (int i = 0; i < N_TEACH_STEPS; i++) {
            float t = (float)i / N_TEACH_STEPS * 2.0f * PI;
            // Figure-8 velocities
            float v = 1.0f;
            float omega = 1.2f * cosf(2.0f * t);

            yaw += omega * DT;
            x += v * cosf(yaw) * DT;
            y += v * sinf(yaw) * DT;

            episode[i].x = x;
            episode[i].y = y;
            episode[i].yaw = yaw;
            episode[i].v = v;
            episode[i].omega = omega;

            // Simulate simple distance beams (from origin)
            for (int b = 0; b < NUM_BEAMS; b++) {
                float angle = yaw + (float)b * (2.0f * PI / NUM_BEAMS) - PI;
                // Simple environment: walls at +/-8m
                float min_r = MAX_RANGE;
                float ca = cosf(angle), sa = sinf(angle);
                for (float r = 0.1f; r < MAX_RANGE; r += 0.1f) {
                    float wx = x + r * ca, wy = y + r * sa;
                    if (fabsf(wx) > 8.0f || fabsf(wy) > 8.0f) {
                        min_r = r;
                        break;
                    }
                }
                episode[i].beams[b] = min_r + gauss(gen) * 0.05f;
            }
        }
    }

    // Flatten episode data for GPU
    std::vector<float> h_ep_beams(N_TEACH_STEPS * NUM_BEAMS);
    std::vector<float> h_ep_v(N_TEACH_STEPS), h_ep_omega(N_TEACH_STEPS);
    for (int i = 0; i < N_TEACH_STEPS; i++) {
        h_ep_v[i] = episode[i].v;
        h_ep_omega[i] = episode[i].omega;
        for (int b = 0; b < NUM_BEAMS; b++)
            h_ep_beams[i * NUM_BEAMS + b] = episode[i].beams[b];
    }

    // GPU memory
    const int np = N_PARTICLES;
    const int threads = 256;
    const int blocks = (np + threads - 1) / threads;

    float *d_ep_beams, *d_ep_v, *d_ep_omega;
    CUDA_CHECK(cudaMalloc(&d_ep_beams, N_TEACH_STEPS * NUM_BEAMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ep_v, N_TEACH_STEPS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ep_omega, N_TEACH_STEPS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ep_beams, h_ep_beams.data(), N_TEACH_STEPS * NUM_BEAMS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ep_v, h_ep_v.data(), N_TEACH_STEPS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ep_omega, h_ep_omega.data(), N_TEACH_STEPS * sizeof(float), cudaMemcpyHostToDevice));

    float *d_cur_beams;
    CUDA_CHECK(cudaMalloc(&d_cur_beams, NUM_BEAMS * sizeof(float)));

    // Particle states
    float *d_p_time, *d_ox, *d_oy, *d_oyaw, *d_pw;
    float *d_p_time_tmp, *d_ox_tmp, *d_oy_tmp, *d_oyaw_tmp;
    CUDA_CHECK(cudaMalloc(&d_p_time, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ox, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_oy, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_oyaw, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pw, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_time_tmp, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ox_tmp, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_oy_tmp, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_oyaw_tmp, np * sizeof(float)));

    float *d_wcum;
    CUDA_CHECK(cudaMalloc(&d_wcum, np * sizeof(float)));

    float *d_out_v, *d_out_omega;
    CUDA_CHECK(cudaMalloc(&d_out_v, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_omega, sizeof(float)));

    curandState* d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, np * sizeof(curandState)));
    init_curand_kernel<<<blocks, threads>>>(d_rng_states, 42ULL, np);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize particles spread across episode time
    std::vector<float> h_p_time(np), h_ox(np, 0), h_oy(np, 0), h_oyaw(np, 0), h_pw(np, 1.0f / np);
    for (int i = 0; i < np; i++) {
        h_p_time[i] = gauss(gen) * 5.0f;
        if (h_p_time[i] < 0) h_p_time[i] = 0;
        if (h_p_time[i] >= N_TEACH_STEPS - 1) h_p_time[i] = N_TEACH_STEPS - 2;
    }
    CUDA_CHECK(cudaMemcpy(d_p_time, h_p_time.data(), np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ox, h_ox.data(), np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oy, h_oy.data(), np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oyaw, h_oyaw.data(), np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pw, h_pw.data(), np * sizeof(float), cudaMemcpyHostToDevice));

    // ======================================================================
    // REPLAY phase
    // ======================================================================
    cv::namedWindow("pf_on_episode", cv::WINDOW_NORMAL);
    cv::VideoWriter video(
        "gif/pf_on_episode.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, cv::Size(IMG_W, IMG_H));

    // Replay robot follows the taught trajectory with noise
    float replay_x = 0, replay_y = 0, replay_yaw = 0;

    for (int step = 0; step < N_TEACH_STEPS; step++) {
        // Replay robot follows episode with noise
        float cmd_v = episode[step].v + gauss(gen) * NOISE_V;
        float cmd_omega = episode[step].omega + gauss(gen) * NOISE_OMEGA;
        replay_yaw += cmd_omega * DT;
        replay_x += cmd_v * cosf(replay_yaw) * DT;
        replay_y += cmd_v * sinf(replay_yaw) * DT;

        // Simulate current lidar for replay robot
        float cur_beams[NUM_BEAMS];
        for (int b = 0; b < NUM_BEAMS; b++) {
            float angle = replay_yaw + (float)b * (2.0f * PI / NUM_BEAMS) - PI;
            float min_r = MAX_RANGE;
            float ca = cosf(angle), sa = sinf(angle);
            for (float r = 0.1f; r < MAX_RANGE; r += 0.1f) {
                float wx = replay_x + r * ca, wy = replay_y + r * sa;
                if (fabsf(wx) > 8.0f || fabsf(wy) > 8.0f) {
                    min_r = r;
                    break;
                }
            }
            cur_beams[b] = min_r + gauss(gen) * 0.05f;
        }
        CUDA_CHECK(cudaMemcpy(d_cur_beams, cur_beams, NUM_BEAMS * sizeof(float), cudaMemcpyHostToDevice));

        // --- GPU: predict ---
        predict_episode_kernel<<<blocks, threads>>>(
            d_p_time, d_ox, d_oy, d_oyaw,
            1.0f, d_rng_states, np, N_TEACH_STEPS);

        // --- GPU: compute weights ---
        compute_weight_kernel<<<blocks, threads>>>(
            d_p_time, d_ox, d_oy, d_pw,
            d_ep_beams, d_cur_beams, np, N_TEACH_STEPS);

        // --- GPU: normalize ---
        normalize_weights_kernel<<<1, threads, threads * sizeof(float)>>>(d_pw, np);

        // --- GPU: weighted action ---
        weighted_action_kernel<<<1, threads, 3 * threads * sizeof(float)>>>(
            d_p_time, d_pw, d_ep_v, d_ep_omega,
            d_out_v, d_out_omega, np, N_TEACH_STEPS);

        // --- GPU: resample ---
        CUDA_CHECK(cudaMemcpy(h_pw.data(), d_pw, np * sizeof(float), cudaMemcpyDeviceToHost));
        float Neff_denom = 0;
        for (int i = 0; i < np; i++) Neff_denom += h_pw[i] * h_pw[i];
        if (1.0f / Neff_denom < np / 2) {
            cumsum_kernel<<<1, 1>>>(d_pw, d_wcum, np);
            float rand_off = uni(gen) / np;
            resample_kernel<<<blocks, threads>>>(
                d_p_time, d_ox, d_oy, d_oyaw,
                d_p_time_tmp, d_ox_tmp, d_oy_tmp, d_oyaw_tmp,
                d_wcum, 1.0f / np, rand_off, np);
            CUDA_CHECK(cudaMemcpy(d_p_time, d_p_time_tmp, np * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_ox, d_ox_tmp, np * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_oy, d_oy_tmp, np * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_oyaw, d_oyaw_tmp, np * sizeof(float), cudaMemcpyDeviceToDevice));
            std::vector<float> pw_uni(np, 1.0f / np);
            CUDA_CHECK(cudaMemcpy(d_pw, pw_uni.data(), np * sizeof(float), cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // Read back particles
        CUDA_CHECK(cudaMemcpy(h_p_time.data(), d_p_time, np * sizeof(float), cudaMemcpyDeviceToHost));

        // --- Visualization ---
        cv::Mat bg(IMG_H, IMG_W, CV_8UC3, cv::Scalar(255, 255, 255));

        // Draw taught path (green)
        for (int i = 1; i < N_TEACH_STEPS; i++) {
            cv::line(bg, to_px(episode[i - 1].x, episode[i - 1].y),
                     to_px(episode[i].x, episode[i].y),
                     cv::Scalar(0, 200, 0), 2);
        }

        // Draw particles (red) at their episode positions
        for (int i = 0; i < np; i++) {
            int t_idx = (int)h_p_time[i];
            if (t_idx < 0) t_idx = 0;
            if (t_idx >= N_TEACH_STEPS) t_idx = N_TEACH_STEPS - 1;
            float px = episode[t_idx].x;
            float py = episode[t_idx].y;
            cv::circle(bg, to_px(px, py), 2, cv::Scalar(0, 0, 255), -1);
        }

        // Draw replay robot (blue)
        cv::circle(bg, to_px(replay_x, replay_y), 6, cv::Scalar(255, 0, 0), -1);

        // Draw episode time histogram (yellow bar at bottom)
        {
            int hist[50] = {};
            for (int i = 0; i < np; i++) {
                int bin = (int)(h_p_time[i] / N_TEACH_STEPS * 50);
                if (bin < 0) bin = 0; if (bin >= 50) bin = 49;
                hist[bin]++;
            }
            int max_h = 1;
            for (int i = 0; i < 50; i++) if (hist[i] > max_h) max_h = hist[i];
            for (int i = 0; i < 50; i++) {
                int bar_h = hist[i] * 60 / max_h;
                cv::rectangle(bg, cv::Point(i * 16, IMG_H - bar_h),
                    cv::Point(i * 16 + 14, IMG_H),
                    cv::Scalar(0, 200, 255), -1);
            }
            cv::putText(bg, "Episode Time Histogram", cv::Point(10, IMG_H - 65),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        // Current step indicator
        {
            int bar_x = (int)((float)step / N_TEACH_STEPS * IMG_W);
            cv::line(bg, cv::Point(bar_x, IMG_H - 70), cv::Point(bar_x, IMG_H),
                     cv::Scalar(255, 0, 0), 2);
        }

        char buf[64];
        snprintf(buf, sizeof(buf), "Step %d/%d", step, N_TEACH_STEPS);
        cv::putText(bg, buf, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

        cv::imshow("pf_on_episode", bg);
        video.write(bg);
        cv::waitKey(5);
    }

    video.release();
    system("ffmpeg -y -i gif/pf_on_episode.avi "
           "-vf 'fps=15,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/pf_on_episode.gif 2>/dev/null");
    std::cout << "GIF saved to gif/pf_on_episode.gif" << std::endl;

    cudaFree(d_ep_beams); cudaFree(d_ep_v); cudaFree(d_ep_omega);
    cudaFree(d_cur_beams); cudaFree(d_p_time); cudaFree(d_ox);
    cudaFree(d_oy); cudaFree(d_oyaw); cudaFree(d_pw);
    cudaFree(d_p_time_tmp); cudaFree(d_ox_tmp); cudaFree(d_oy_tmp);
    cudaFree(d_oyaw_tmp); cudaFree(d_wcum); cudaFree(d_rng_states);
    cudaFree(d_out_v); cudaFree(d_out_omega);

    return 0;
}
