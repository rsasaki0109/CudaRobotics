/*************************************************************************
    > File Name: pf_on_episode.cu
    > CUDA-parallelized Particle Filter on Episode (PFoE)
    > Based on: Ryuichi Ueda, "Particle Filter on Episode" (arXiv:1904.08761)
    >
    > CUDA kernels for:
    >   - predict_episode_kernel: advance particles along episode timeline
    >   - compute_weight_kernel: weight by sensor similarity
    >   - weighted_action_kernel: compute weighted average action
    >   - resample_kernel: systematic resampling in episode time space
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
#define DT 0.1f
#define SIM_TIME 25.0f
#define PI 3.141592653f

#define N_PARTICLES 500
#define EPISODE_LEN 200
#define N_BEAMS 36
#define LIDAR_MAX_RANGE 10.0f

// Map dimensions (meters)
#define MAP_XMIN -2.0f
#define MAP_XMAX 12.0f
#define MAP_YMIN -2.0f
#define MAP_YMAX 12.0f

// Image
#define IMG_W 1200
#define IMG_H 1000

// ---------------------------------------------------------------------------
// CUDA error check macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

// ---------------------------------------------------------------------------
// Episode step: recorded (x, y, yaw, lidar[N_BEAMS], v, omega)
// ---------------------------------------------------------------------------
struct EpisodeStep {
    float x, y, yaw;
    float lidar[N_BEAMS];
    float v, omega;
};

// ---------------------------------------------------------------------------
// Wall segment for lidar simulation
// ---------------------------------------------------------------------------
struct Wall {
    float x1, y1, x2, y2;
};

#define MAX_WALLS 20

// ---------------------------------------------------------------------------
// Device constants
// ---------------------------------------------------------------------------
__constant__ Wall d_walls[MAX_WALLS];
__constant__ int d_n_walls;

// ---------------------------------------------------------------------------
// Device: ray-segment intersection, returns distance or LIDAR_MAX_RANGE
// ---------------------------------------------------------------------------
__device__ float ray_segment_intersect(float ox, float oy, float angle,
                                       float wx1, float wy1, float wx2, float wy2) {
    float dx = cosf(angle);
    float dy = sinf(angle);
    float sx = wx2 - wx1;
    float sy = wy2 - wy1;

    float denom = dx * sy - dy * sx;
    if (fabsf(denom) < 1e-8f) return LIDAR_MAX_RANGE;

    float t = ((wx1 - ox) * sy - (wy1 - oy) * sx) / denom;
    float u = ((wx1 - ox) * dy - (wy1 - oy) * dx) / denom;

    if (t > 0.0f && t < LIDAR_MAX_RANGE && u >= 0.0f && u <= 1.0f)
        return t;
    return LIDAR_MAX_RANGE;
}

// ---------------------------------------------------------------------------
// Device: simulate one lidar beam against all walls
// ---------------------------------------------------------------------------
__device__ float simulate_lidar_beam(float ox, float oy, float angle) {
    float min_d = LIDAR_MAX_RANGE;
    for (int i = 0; i < d_n_walls; i++) {
        float d = ray_segment_intersect(ox, oy, angle,
                                        d_walls[i].x1, d_walls[i].y1,
                                        d_walls[i].x2, d_walls[i].y2);
        if (d < min_d) min_d = d;
    }
    return min_d;
}

// ---------------------------------------------------------------------------
// Kernel: initialize cuRAND states
// ---------------------------------------------------------------------------
__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// ---------------------------------------------------------------------------
// Kernel: predict - advance particles along episode timeline with noise
//   d_episode_idx[i] is the (float) episode time index for particle i
// ---------------------------------------------------------------------------
__global__ void predict_episode_kernel(float* d_episode_idx,
                                       curandState* rng_states,
                                       int np, int episode_len) {
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    curandState local_rng = rng_states[ip];

    // Advance by 1 step + Gaussian noise
    float noise = curand_normal(&local_rng) * 0.5f;
    float new_idx = d_episode_idx[ip] + 1.0f + noise;

    // Clamp to valid range [0, episode_len - 1]
    if (new_idx < 0.0f) new_idx = 0.0f;
    if (new_idx > (float)(episode_len - 1)) new_idx = (float)(episode_len - 1);

    d_episode_idx[ip] = new_idx;
    rng_states[ip] = local_rng;
}

// ---------------------------------------------------------------------------
// Kernel: compute weight - compare current sensor readings with episode readings
//   at each particle's episode time position
//   d_current_lidar[N_BEAMS]: current sensor reading
//   d_episode[EPISODE_LEN]: full episode data on device
//   d_weights[np]: output weights
// ---------------------------------------------------------------------------
__global__ void compute_weight_kernel(const float* d_episode_idx,
                                      const EpisodeStep* d_episode,
                                      const float* d_current_lidar,
                                      float* d_weights,
                                      int np, int episode_len) {
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    // Get episode index (nearest integer)
    int eidx = (int)(d_episode_idx[ip] + 0.5f);
    if (eidx < 0) eidx = 0;
    if (eidx >= episode_len) eidx = episode_len - 1;

    // Compute sum of squared differences
    float ssd = 0.0f;
    for (int b = 0; b < N_BEAMS; b++) {
        float diff = d_current_lidar[b] - d_episode[eidx].lidar[b];
        ssd += diff * diff;
    }

    // Weight = exp(-ssd / (2 * sigma^2)), sigma^2 = 1.0
    float sigma2 = 1.0f;
    d_weights[ip] = expf(-ssd / (2.0f * sigma2));
}

// ---------------------------------------------------------------------------
// Kernel: normalize weights (single-block reduction)
// ---------------------------------------------------------------------------
__global__ void normalize_weights_kernel(float* d_weights, int np) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float val = 0.0f;
    for (int i = tid; i < np; i += blockDim.x) {
        val += d_weights[i];
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float total = sdata[0];
    if (total < 1e-30f) total = 1e-30f;

    for (int i = tid; i < np; i += blockDim.x) {
        d_weights[i] /= total;
    }
}

// ---------------------------------------------------------------------------
// Kernel: compute weighted average action from episode
//   Outputs: weighted v and omega
// ---------------------------------------------------------------------------
__global__ void weighted_action_kernel(const float* d_episode_idx,
                                       const float* d_weights,
                                       const EpisodeStep* d_episode,
                                       float* d_action_out,  // [2]: v, omega
                                       int np, int episode_len) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float sv = 0.0f, sw = 0.0f;
    for (int i = tid; i < np; i += blockDim.x) {
        int eidx = (int)(d_episode_idx[i] + 0.5f);
        if (eidx < 0) eidx = 0;
        if (eidx >= episode_len) eidx = episode_len - 1;

        float w = d_weights[i];
        sv += w * d_episode[eidx].v;
        sw += w * d_episode[eidx].omega;
    }
    sdata[tid * 2 + 0] = sv;
    sdata[tid * 2 + 1] = sw;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid * 2 + 0] += sdata[(tid + s) * 2 + 0];
            sdata[tid * 2 + 1] += sdata[(tid + s) * 2 + 1];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_action_out[0] = sdata[0];
        d_action_out[1] = sdata[1];
    }
}

// ---------------------------------------------------------------------------
// Kernel: cumulative sum for resampling (single thread)
// ---------------------------------------------------------------------------
__global__ void cumsum_kernel(const float* d_weights, float* d_wcum, int np) {
    d_wcum[0] = d_weights[0];
    for (int i = 1; i < np; i++) {
        d_wcum[i] = d_wcum[i - 1] + d_weights[i];
    }
}

// ---------------------------------------------------------------------------
// Kernel: systematic resampling in episode time space
// ---------------------------------------------------------------------------
__global__ void resample_kernel(const float* d_episode_idx_in,
                                float* d_episode_idx_out,
                                const float* d_wcum,
                                float base_step,
                                float rand_offset,
                                int np) {
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    float target = base_step * ip + rand_offset;

    // Binary search in wcum
    int lo = 0, hi = np - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (d_wcum[mid] < target) lo = mid + 1;
        else hi = mid;
    }

    d_episode_idx_out[ip] = d_episode_idx_in[lo];
}

// ---------------------------------------------------------------------------
// Host: simulate all lidar beams for a given pose
// ---------------------------------------------------------------------------
void simulate_lidar_host(float px, float py, float yaw,
                         const std::vector<Wall>& walls,
                         float* lidar_out) {
    for (int b = 0; b < N_BEAMS; b++) {
        float angle = yaw + 2.0f * PI * b / N_BEAMS;
        float min_d = LIDAR_MAX_RANGE;
        for (size_t w = 0; w < walls.size(); w++) {
            float dx = cosf(angle);
            float dy = sinf(angle);
            float sx = walls[w].x2 - walls[w].x1;
            float sy = walls[w].y2 - walls[w].y1;
            float denom = dx * sy - dy * sx;
            if (fabsf(denom) < 1e-8f) continue;
            float t = ((walls[w].x1 - px) * sy - (walls[w].y1 - py) * sx) / denom;
            float u = ((walls[w].x1 - px) * dy - (walls[w].y1 - py) * dx) / denom;
            if (t > 0.0f && t < LIDAR_MAX_RANGE && u >= 0.0f && u <= 1.0f) {
                if (t < min_d) min_d = t;
            }
        }
        lidar_out[b] = min_d;
    }
}

// ---------------------------------------------------------------------------
// Host: coordinate to pixel
// ---------------------------------------------------------------------------
cv::Point2i to_pixel(float x, float y) {
    // Map [MAP_XMIN, MAP_XMAX] x [MAP_YMIN, MAP_YMAX] -> [0, IMG_W-1] x [0, IMG_H-1]
    // Leave margin for histogram on right side
    int map_area_w = IMG_W - 200;  // reserve 200px for histogram
    int px = (int)((x - MAP_XMIN) / (MAP_XMAX - MAP_XMIN) * map_area_w);
    int py = IMG_H - 1 - (int)((y - MAP_YMIN) / (MAP_YMAX - MAP_YMIN) * IMG_H);
    return cv::Point2i(px, py);
}

// ---------------------------------------------------------------------------
// Host: generate figure-8 trajectory for TEACH phase
// ---------------------------------------------------------------------------
void generate_figure8_episode(std::vector<EpisodeStep>& episode,
                              const std::vector<Wall>& walls) {
    // Figure-8 centered at (5, 5), radii ~3
    float cx = 5.0f, cy = 5.0f;
    float radius = 3.0f;

    float x = cx + radius, y = cy, yaw = PI / 2.0f;

    for (int t = 0; t < EPISODE_LEN; t++) {
        EpisodeStep step;
        step.x = x;
        step.y = y;
        step.yaw = yaw;

        // Simulate lidar at this pose
        simulate_lidar_host(x, y, yaw, walls, step.lidar);

        // Figure-8 parametric: two circles
        // First half: circle around (cx, cy+radius) CCW
        // Second half: circle around (cx, cy-radius) CW
        float progress = (float)t / EPISODE_LEN;
        float v, omega;

        if (progress < 0.5f) {
            // Upper circle, CCW
            v = 2.0f * PI * radius / (EPISODE_LEN * DT * 0.5f);
            omega = v / radius;
        } else {
            // Lower circle, CW
            v = 2.0f * PI * radius / (EPISODE_LEN * DT * 0.5f);
            omega = -v / radius;
        }

        step.v = v;
        step.omega = omega;

        episode.push_back(step);

        // Advance state with motion model
        x += DT * cosf(yaw) * v;
        y += DT * sinf(yaw) * v;
        yaw += DT * omega;

        // Normalize yaw
        while (yaw > PI) yaw -= 2.0f * PI;
        while (yaw < -PI) yaw += 2.0f * PI;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "Particle Filter on Episode (PFoE) with CUDA" << std::endl;
    std::cout << "Particles: " << N_PARTICLES << ", Episode length: " << EPISODE_LEN << std::endl;

    // ------------------------------------------
    // Define map walls (rectangular room with inner obstacle)
    // ------------------------------------------
    std::vector<Wall> walls;
    // Outer walls
    walls.push_back({0.0f, 0.0f, 10.0f, 0.0f});   // bottom
    walls.push_back({10.0f, 0.0f, 10.0f, 10.0f});  // right
    walls.push_back({10.0f, 10.0f, 0.0f, 10.0f});  // top
    walls.push_back({0.0f, 10.0f, 0.0f, 0.0f});    // left
    // Inner obstacles
    walls.push_back({3.0f, 3.0f, 3.0f, 4.5f});
    walls.push_back({3.0f, 4.5f, 4.5f, 4.5f});
    walls.push_back({7.0f, 5.5f, 7.0f, 7.0f});
    walls.push_back({7.0f, 7.0f, 5.5f, 7.0f});

    int n_walls = (int)walls.size();
    CUDA_CHECK(cudaMemcpyToSymbol(d_walls, walls.data(), n_walls * sizeof(Wall)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_n_walls, &n_walls, sizeof(int)));

    // ------------------------------------------
    // TEACH phase: record episode
    // ------------------------------------------
    std::vector<EpisodeStep> episode;
    generate_figure8_episode(episode, walls);
    std::cout << "TEACH phase complete: " << episode.size() << " steps recorded." << std::endl;

    // Copy episode to device
    EpisodeStep* d_episode;
    CUDA_CHECK(cudaMalloc(&d_episode, EPISODE_LEN * sizeof(EpisodeStep)));
    CUDA_CHECK(cudaMemcpy(d_episode, episode.data(), EPISODE_LEN * sizeof(EpisodeStep),
                          cudaMemcpyHostToDevice));

    // ------------------------------------------
    // CUDA memory allocation for particle filter
    // ------------------------------------------
    const int np = N_PARTICLES;
    const int threads = 256;
    const int blocks = (np + threads - 1) / threads;

    // Particle episode indices (float)
    float *d_episode_idx, *d_episode_idx_tmp;
    CUDA_CHECK(cudaMalloc(&d_episode_idx, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_episode_idx_tmp, np * sizeof(float)));

    // Initialize particles uniformly across first portion of episode
    {
        std::vector<float> init_idx(np);
        for (int i = 0; i < np; i++) {
            init_idx[i] = (float)i / np * 10.0f;  // Start clustered near beginning
        }
        CUDA_CHECK(cudaMemcpy(d_episode_idx, init_idx.data(), np * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Weights
    float* d_weights;
    CUDA_CHECK(cudaMalloc(&d_weights, np * sizeof(float)));
    {
        std::vector<float> w_init(np, 1.0f / np);
        CUDA_CHECK(cudaMemcpy(d_weights, w_init.data(), np * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Current lidar on device
    float* d_current_lidar;
    CUDA_CHECK(cudaMalloc(&d_current_lidar, N_BEAMS * sizeof(float)));

    // Action output on device
    float* d_action_out;
    CUDA_CHECK(cudaMalloc(&d_action_out, 2 * sizeof(float)));

    // Cumulative sum for resampling
    float* d_wcum;
    CUDA_CHECK(cudaMalloc(&d_wcum, np * sizeof(float)));

    // cuRAND states
    curandState* d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, np * sizeof(curandState)));
    init_curand_kernel<<<blocks, threads>>>(d_rng_states, 42ULL, np);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Host buffers
    std::vector<float> h_episode_idx(np);
    std::vector<float> h_weights(np);
    float h_current_lidar[N_BEAMS];
    float h_action[2];

    // ------------------------------------------
    // REPLAY phase: robot state
    // ------------------------------------------
    // Start near episode start with slight offset
    float robot_x = episode[0].x + 0.3f;
    float robot_y = episode[0].y + 0.3f;
    float robot_yaw = episode[0].yaw + 0.1f;

    std::vector<float> replay_x_hist, replay_y_hist;
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> uni_d(0.0f, 1.0f);
    std::normal_distribution<float> gauss_d(0.0f, 1.0f);

    // ------------------------------------------
    // Visualization
    // ------------------------------------------
    cv::namedWindow("pf_on_episode", cv::WINDOW_NORMAL);
    std::string video_path = "gif/pf_on_episode.avi";
    cv::VideoWriter video(video_path, cv::VideoWriter::fourcc('X','V','I','D'), 30,
                          cv::Size(IMG_W, IMG_H));

    float time_val = 0.0f;
    int step_count = 0;

    while (time_val <= SIM_TIME) {
        time_val += DT;
        step_count++;

        // --- 1. Simulate lidar at robot's current position ---
        simulate_lidar_host(robot_x, robot_y, robot_yaw, walls, h_current_lidar);
        CUDA_CHECK(cudaMemcpy(d_current_lidar, h_current_lidar, N_BEAMS * sizeof(float),
                              cudaMemcpyHostToDevice));

        // --- 2. Predict: advance particles along episode timeline ---
        predict_episode_kernel<<<blocks, threads>>>(
            d_episode_idx, d_rng_states, np, EPISODE_LEN);

        // --- 3. Compute weights by sensor similarity ---
        compute_weight_kernel<<<blocks, threads>>>(
            d_episode_idx, d_episode, d_current_lidar, d_weights,
            np, EPISODE_LEN);

        // --- 4. Normalize weights ---
        normalize_weights_kernel<<<1, threads, threads * sizeof(float)>>>(
            d_weights, np);

        // --- 5. Compute weighted average action ---
        weighted_action_kernel<<<1, threads, threads * 2 * sizeof(float)>>>(
            d_episode_idx, d_weights, d_episode, d_action_out,
            np, EPISODE_LEN);

        CUDA_CHECK(cudaMemcpy(h_action, d_action_out, 2 * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // --- 6. Resampling ---
        CUDA_CHECK(cudaMemcpy(h_weights.data(), d_weights, np * sizeof(float),
                              cudaMemcpyDeviceToHost));

        float neff_denom = 0.0f;
        for (int i = 0; i < np; i++) neff_denom += h_weights[i] * h_weights[i];
        float neff = 1.0f / neff_denom;

        if (neff < np / 2.0f) {
            cumsum_kernel<<<1, 1>>>(d_weights, d_wcum, np);

            float rand_offset = uni_d(gen) / np;
            float base_step = 1.0f / np;

            resample_kernel<<<blocks, threads>>>(
                d_episode_idx, d_episode_idx_tmp, d_wcum,
                base_step, rand_offset, np);

            CUDA_CHECK(cudaMemcpy(d_episode_idx, d_episode_idx_tmp, np * sizeof(float),
                                  cudaMemcpyDeviceToDevice));

            // Reset weights
            std::vector<float> pw_uniform(np, 1.0f / np);
            CUDA_CHECK(cudaMemcpy(d_weights, pw_uniform.data(), np * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // --- 7. Apply action to robot (with occasional perturbation) ---
        float cmd_v = h_action[0];
        float cmd_omega = h_action[1];

        // Add perturbation every 50 steps to test recovery
        if (step_count % 50 == 0 && step_count > 0) {
            robot_x += gauss_d(gen) * 0.3f;
            robot_y += gauss_d(gen) * 0.3f;
            robot_yaw += gauss_d(gen) * 0.15f;
        }

        robot_x += DT * cosf(robot_yaw) * cmd_v;
        robot_y += DT * sinf(robot_yaw) * cmd_v;
        robot_yaw += DT * cmd_omega;
        while (robot_yaw > PI) robot_yaw -= 2.0f * PI;
        while (robot_yaw < -PI) robot_yaw += 2.0f * PI;

        // Clamp to map
        robot_x = fmaxf(MAP_XMIN + 0.1f, fminf(MAP_XMAX - 0.1f, robot_x));
        robot_y = fmaxf(MAP_YMIN + 0.1f, fminf(MAP_YMAX - 0.1f, robot_y));

        replay_x_hist.push_back(robot_x);
        replay_y_hist.push_back(robot_y);

        // --- 8. Visualization ---
        // Read back particle indices
        CUDA_CHECK(cudaMemcpy(h_episode_idx.data(), d_episode_idx, np * sizeof(float),
                              cudaMemcpyDeviceToHost));

        cv::Mat bg(IMG_H, IMG_W, CV_8UC3, cv::Scalar(255, 255, 255));

        // Draw walls
        for (size_t w = 0; w < walls.size(); w++) {
            cv::line(bg, to_pixel(walls[w].x1, walls[w].y1),
                     to_pixel(walls[w].x2, walls[w].y2),
                     cv::Scalar(0, 0, 0), 2);
        }

        // Draw taught trajectory (green)
        for (int t = 1; t < EPISODE_LEN; t++) {
            cv::line(bg, to_pixel(episode[t-1].x, episode[t-1].y),
                     to_pixel(episode[t].x, episode[t].y),
                     cv::Scalar(0, 180, 0), 2);
        }

        // Draw replay trajectory (blue)
        for (size_t t = 1; t < replay_x_hist.size(); t++) {
            cv::line(bg, to_pixel(replay_x_hist[t-1], replay_y_hist[t-1]),
                     to_pixel(replay_x_hist[t], replay_y_hist[t]),
                     cv::Scalar(255, 100, 0), 2);
        }

        // Draw particles projected to physical space (red dots)
        for (int i = 0; i < np; i++) {
            int eidx = (int)(h_episode_idx[i] + 0.5f);
            if (eidx < 0) eidx = 0;
            if (eidx >= EPISODE_LEN) eidx = EPISODE_LEN - 1;
            cv::circle(bg, to_pixel(episode[eidx].x, episode[eidx].y),
                       3, cv::Scalar(0, 0, 255), -1);
        }

        // Draw robot current position (blue filled circle)
        cv::circle(bg, to_pixel(robot_x, robot_y), 8, cv::Scalar(255, 50, 0), -1);

        // Draw robot direction
        float arrow_len = 0.5f;
        cv::arrowedLine(bg, to_pixel(robot_x, robot_y),
                        to_pixel(robot_x + arrow_len * cosf(robot_yaw),
                                 robot_y + arrow_len * sinf(robot_yaw)),
                        cv::Scalar(255, 50, 0), 2);

        // Draw lidar beams (light gray, from robot)
        for (int b = 0; b < N_BEAMS; b++) {
            float angle = robot_yaw + 2.0f * PI * b / N_BEAMS;
            float ex = robot_x + h_current_lidar[b] * cosf(angle);
            float ey = robot_y + h_current_lidar[b] * sinf(angle);
            cv::line(bg, to_pixel(robot_x, robot_y), to_pixel(ex, ey),
                     cv::Scalar(200, 200, 200), 1);
        }

        // --- Episode time histogram on the right side ---
        int hist_x_start = IMG_W - 190;
        int hist_w = 170;
        int hist_h = IMG_H - 40;
        int hist_y_start = 20;

        // Background for histogram
        cv::rectangle(bg, cv::Point(hist_x_start - 5, hist_y_start - 5),
                      cv::Point(hist_x_start + hist_w + 5, hist_y_start + hist_h + 5),
                      cv::Scalar(240, 240, 240), -1);
        cv::rectangle(bg, cv::Point(hist_x_start - 5, hist_y_start - 5),
                      cv::Point(hist_x_start + hist_w + 5, hist_y_start + hist_h + 5),
                      cv::Scalar(0, 0, 0), 1);

        // Compute histogram bins
        const int n_bins = 40;
        std::vector<int> bins(n_bins, 0);
        for (int i = 0; i < np; i++) {
            int bin = (int)(h_episode_idx[i] / EPISODE_LEN * n_bins);
            if (bin < 0) bin = 0;
            if (bin >= n_bins) bin = n_bins - 1;
            bins[bin]++;
        }
        int max_bin = *std::max_element(bins.begin(), bins.end());
        if (max_bin < 1) max_bin = 1;

        // Draw bars
        int bar_h = hist_h / n_bins;
        for (int b = 0; b < n_bins; b++) {
            int bar_w = (int)((float)bins[b] / max_bin * hist_w);
            int y_top = hist_y_start + b * bar_h;
            cv::rectangle(bg, cv::Point(hist_x_start, y_top),
                          cv::Point(hist_x_start + bar_w, y_top + bar_h - 1),
                          cv::Scalar(0, 200, 255), -1);
        }

        // Labels
        cv::putText(bg, "Episode Time", cv::Point(hist_x_start, hist_y_start - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        cv::putText(bg, "0", cv::Point(hist_x_start - 15, hist_y_start + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1);
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", EPISODE_LEN);
        cv::putText(bg, buf, cv::Point(hist_x_start - 30, hist_y_start + hist_h),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1);

        // Info text
        snprintf(buf, sizeof(buf), "t=%.1f Neff=%.0f", time_val, neff);
        cv::putText(bg, buf, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        snprintf(buf, sizeof(buf), "v=%.2f w=%.2f", cmd_v, cmd_omega);
        cv::putText(bg, buf, cv::Point(10, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

        // Legend
        cv::circle(bg, cv::Point(15, 75), 5, cv::Scalar(0, 180, 0), -1);
        cv::putText(bg, "Taught trajectory", cv::Point(25, 80),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        cv::circle(bg, cv::Point(15, 95), 5, cv::Scalar(255, 100, 0), -1);
        cv::putText(bg, "Replay trajectory", cv::Point(25, 100),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        cv::circle(bg, cv::Point(15, 115), 5, cv::Scalar(0, 0, 255), -1);
        cv::putText(bg, "Particles (episode->space)", cv::Point(25, 120),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

        cv::imshow("pf_on_episode", bg);
        video.write(bg);
        cv::waitKey(5);
    }

    video.release();
    std::cout << "Video saved to " << video_path << std::endl;

    // Convert to gif
    std::string gif_path = "gif/pf_on_episode.gif";
    std::string cmd = "ffmpeg -y -i " + video_path + " -vf \"fps=15,scale=600:-1\" " + gif_path + " 2>/dev/null";
    int ret = system(cmd.c_str());
    if (ret == 0) {
        std::cout << "GIF saved to " << gif_path << std::endl;
    }

    // Cleanup
    cudaFree(d_episode);
    cudaFree(d_episode_idx);
    cudaFree(d_episode_idx_tmp);
    cudaFree(d_weights);
    cudaFree(d_current_lidar);
    cudaFree(d_action_out);
    cudaFree(d_wcum);
    cudaFree(d_rng_states);

    return 0;
}
