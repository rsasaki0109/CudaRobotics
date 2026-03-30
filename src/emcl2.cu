/*************************************************************************
    > File Name: emcl2.cu
    > CUDA-parallelized Expansion Resetting MCL (emcl2)
    > Based on Ueda (IROS 2004) - Expansion Resetting for kidnapped robot
    > CUDA kernels: predict, compute_likelihood, check_reset,
    >               expansion_reset, sensor_reset, resample
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
#define DT 0.1f
#define SIM_TIME 30.0f

// Grid
#define GRID_W 200
#define GRID_H 200
#define GRID_RES 0.1f

// Particles
#define N_PARTICLES 500

// Lidar
#define NUM_BEAMS 36
#define BEAM_ANGLE_STEP (2.0f * PI / NUM_BEAMS)
#define MAX_RANGE 10.0f

// Likelihood field
#define SIGMA_HIT 0.2f
#define Z_HIT 0.9f
#define Z_RAND 0.1f

// Motion noise
#define ALPHA1 0.1f
#define ALPHA2 0.1f
#define ALPHA3 0.1f
#define ALPHA4 0.1f

// Expansion reset thresholds
#define EXPANSION_RESET_THRESHOLD 0.001f
#define EXPANSION_NOISE_XY 2.0f
#define EXPANSION_NOISE_TH 1.0f

// Visualization
#define VIS_SCALE 4
#define IMG_W (GRID_W * VIS_SCALE)
#define IMG_H (GRID_H * VIS_SCALE)

// Kidnap time
#define KIDNAP_TIME 10.0f

// ---------------------------------------------------------------------------
// Kernel: init cuRAND
// ---------------------------------------------------------------------------
__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}

// ---------------------------------------------------------------------------
// Kernel: predict particles (velocity motion model)
// ---------------------------------------------------------------------------
__global__ void predict_kernel(
    float* px, float* py, float* ptheta,
    float v, float omega, float dt,
    curandState* rng_states, int np)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;

    curandState local_rng = rng_states[idx];

    float v_hat = v + curand_normal(&local_rng) * (ALPHA1 * fabsf(v) + ALPHA2 * fabsf(omega));
    float omega_hat = omega + curand_normal(&local_rng) * (ALPHA3 * fabsf(v) + ALPHA4 * fabsf(omega));

    float theta = ptheta[idx];
    if (fabsf(omega_hat) < 1e-6f) {
        px[idx] += v_hat * cosf(theta) * dt;
        py[idx] += v_hat * sinf(theta) * dt;
    } else {
        float r = v_hat / omega_hat;
        px[idx] += r * (sinf(theta + omega_hat * dt) - sinf(theta));
        py[idx] += r * (cosf(theta) - cosf(theta + omega_hat * dt));
    }
    ptheta[idx] += omega_hat * dt;

    float th = ptheta[idx];
    while (th > PI) th -= 2.0f * PI;
    while (th < -PI) th += 2.0f * PI;
    ptheta[idx] = th;

    rng_states[idx] = local_rng;
}

// ---------------------------------------------------------------------------
// Kernel: compute likelihood using likelihood field model
// ---------------------------------------------------------------------------
__global__ void compute_likelihood_kernel(
    float* px, float* py, float* ptheta, float* pw,
    const float* likelihood_field, const float* beam_ranges,
    int width, int height, float resolution,
    float origin_x, float origin_y, int np)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;

    float x = px[idx], y = py[idx], theta = ptheta[idx];
    float log_w = 0.0f;

    for (int b = 0; b < NUM_BEAMS; b++) {
        float range = beam_ranges[b];
        if (range >= MAX_RANGE) continue;

        float beam_angle = theta + (float)b * BEAM_ANGLE_STEP - PI;
        float ex = x + range * cosf(beam_angle);
        float ey = y + range * sinf(beam_angle);

        int gx = (int)((ex - origin_x) / resolution);
        int gy = (int)((ey - origin_y) / resolution);

        if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
            float lf = likelihood_field[gy * width + gx];
            log_w += logf(fmaxf(lf, 1e-10f));
        } else {
            log_w += logf(Z_RAND / MAX_RANGE);
        }
    }
    pw[idx] = expf(log_w);
}

// ---------------------------------------------------------------------------
// Kernel: check if expansion reset is needed
// Single-block reduction to compute max weight and average weight
// ---------------------------------------------------------------------------
__global__ void check_reset_kernel(
    const float* pw, int np,
    float* out_max_w, float* out_avg_w, float* out_sum_w)
{
    extern __shared__ float sdata[];
    float* s_max = sdata;
    float* s_sum = sdata + blockDim.x;
    int tid = threadIdx.x;

    float local_max = 0.0f, local_sum = 0.0f;
    for (int i = tid; i < np; i += blockDim.x) {
        float w = pw[i];
        if (w > local_max) local_max = w;
        local_sum += w;
    }
    s_max[tid] = local_max;
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_max_w = s_max[0];
        *out_avg_w = s_sum[0] / (float)np;
        *out_sum_w = s_sum[0];
    }
}

// ---------------------------------------------------------------------------
// Kernel: expansion reset - add noise to particles
// Expands particle distribution when kidnapped
// ---------------------------------------------------------------------------
__global__ void expansion_reset_kernel(
    float* px, float* py, float* ptheta, float* pw,
    float noise_xy, float noise_th,
    const int* occupancy, int grid_w, int grid_h,
    float origin_x, float origin_y, float resolution,
    curandState* rng_states, int np)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;

    curandState local_rng = rng_states[idx];

    // Add Gaussian noise to position and orientation
    px[idx] += curand_normal(&local_rng) * noise_xy;
    py[idx] += curand_normal(&local_rng) * noise_xy;
    ptheta[idx] += curand_normal(&local_rng) * noise_th;

    // Clamp to free space
    int gx = (int)((px[idx] - origin_x) / resolution);
    int gy = (int)((py[idx] - origin_y) / resolution);
    if (gx < 2) gx = 2; if (gx >= grid_w - 2) gx = grid_w - 3;
    if (gy < 2) gy = 2; if (gy >= grid_h - 2) gy = grid_h - 3;

    // If in obstacle, random free cell
    if (occupancy[gy * grid_w + gx] == 1) {
        // Generate random free cell
        for (int attempt = 0; attempt < 100; attempt++) {
            int rx = (int)(curand_uniform(&local_rng) * (grid_w - 4)) + 2;
            int ry = (int)(curand_uniform(&local_rng) * (grid_h - 4)) + 2;
            if (occupancy[ry * grid_w + rx] == 0) {
                px[idx] = origin_x + (rx + 0.5f) * resolution;
                py[idx] = origin_y + (ry + 0.5f) * resolution;
                break;
            }
        }
    }

    float th = ptheta[idx];
    while (th > PI) th -= 2.0f * PI;
    while (th < -PI) th += 2.0f * PI;
    ptheta[idx] = th;

    pw[idx] = 1.0f / (float)np;
    rng_states[idx] = local_rng;
}

// ---------------------------------------------------------------------------
// Kernel: sensor reset - scatter some particles near high-likelihood areas
// ---------------------------------------------------------------------------
__global__ void sensor_reset_kernel(
    float* px, float* py, float* ptheta, float* pw,
    const float* beam_ranges,
    const int* occupancy, int grid_w, int grid_h,
    float origin_x, float origin_y, float resolution,
    curandState* rng_states, int np, int n_reset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_reset) return;

    curandState local_rng = rng_states[idx];

    // Place particle at random free cell
    for (int attempt = 0; attempt < 200; attempt++) {
        int rx = (int)(curand_uniform(&local_rng) * (grid_w - 4)) + 2;
        int ry = (int)(curand_uniform(&local_rng) * (grid_h - 4)) + 2;
        if (occupancy[ry * grid_w + rx] == 0) {
            px[idx] = origin_x + (rx + 0.5f) * resolution;
            py[idx] = origin_y + (ry + 0.5f) * resolution;
            ptheta[idx] = curand_uniform(&local_rng) * 2.0f * PI - PI;
            pw[idx] = 1.0f / (float)np;
            break;
        }
    }

    rng_states[idx] = local_rng;
}

// ---------------------------------------------------------------------------
// Kernel: normalize weights
// ---------------------------------------------------------------------------
__global__ void normalize_weights_kernel(float* pw, int np, float total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;
    if (total > 1e-30f) pw[idx] /= total;
    else pw[idx] = 1.0f / (float)np;
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
    const float* px_in, const float* py_in, const float* ptheta_in,
    float* px_out, float* py_out, float* ptheta_out,
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
    px_out[ip] = px_in[lo];
    py_out[ip] = py_in[lo];
    ptheta_out[ip] = ptheta_in[lo];
}

// ---------------------------------------------------------------------------
// Kernel: weighted mean
// ---------------------------------------------------------------------------
__global__ void weighted_mean_kernel(
    const float* px, const float* py, const float* ptheta,
    const float* pw, float* out_x, float* out_y, float* out_theta, int np)
{
    extern __shared__ float sdata[];
    float* sx = sdata;
    float* sy = sdata + blockDim.x;
    float* sc = sdata + 2 * blockDim.x;
    float* ss = sdata + 3 * blockDim.x;
    int tid = threadIdx.x;

    float vx = 0, vy = 0, vc = 0, vs = 0;
    for (int i = tid; i < np; i += blockDim.x) {
        float w = pw[i];
        vx += px[i] * w; vy += py[i] * w;
        vc += cosf(ptheta[i]) * w; vs += sinf(ptheta[i]) * w;
    }
    sx[tid] = vx; sy[tid] = vy; sc[tid] = vc; ss[tid] = vs;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sx[tid] += sx[tid + s]; sy[tid] += sy[tid + s];
            sc[tid] += sc[tid + s]; ss[tid] += ss[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out_x = sx[0]; *out_y = sy[0];
        *out_theta = atan2f(ss[0], sc[0]);
    }
}

// ---------------------------------------------------------------------------
// Kernel: build likelihood field
// ---------------------------------------------------------------------------
__global__ void build_likelihood_field_kernel(
    const int* occupancy, float* likelihood_field,
    int width, int height, float sigma_hit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int cy = idx / width, cx = idx % width;
    float min_dist = 1e6f;
    int search_radius = (int)(3.0f * sigma_hit / GRID_RES) + 1;
    if (search_radius > 50) search_radius = 50;

    for (int dy = -search_radius; dy <= search_radius; dy++) {
        for (int dx = -search_radius; dx <= search_radius; dx++) {
            int nx = cx + dx, ny = cy + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (occupancy[ny * width + nx] == 1) {
                    float dist = sqrtf((float)(dx * dx + dy * dy)) * GRID_RES;
                    if (dist < min_dist) min_dist = dist;
                }
            }
        }
    }
    likelihood_field[idx] = Z_HIT * expf(-0.5f * min_dist * min_dist / (sigma_hit * sigma_hit))
                            + Z_RAND / MAX_RANGE;
}

// ---------------------------------------------------------------------------
// Host: build occupancy map
// ---------------------------------------------------------------------------
void build_map(std::vector<int>& occupancy, int w, int h) {
    occupancy.assign(w * h, 0);
    // Boundary walls
    for (int x = 0; x < w; x++) {
        occupancy[0 * w + x] = 1; occupancy[1 * w + x] = 1;
        occupancy[(h - 1) * w + x] = 1; occupancy[(h - 2) * w + x] = 1;
    }
    for (int y = 0; y < h; y++) {
        occupancy[y * w + 0] = 1; occupancy[y * w + 1] = 1;
        occupancy[y * w + (w - 1)] = 1; occupancy[y * w + (w - 2)] = 1;
    }
    // Internal walls
    for (int x = 40; x < 80; x++)
        for (int t = 0; t < 3; t++) occupancy[(60 + t) * w + x] = 1;
    for (int y = 80; y < 140; y++)
        for (int t = 0; t < 3; t++) occupancy[y * w + (130 + t)] = 1;
    for (int x = 30; x < 55; x++)
        for (int t = 0; t < 3; t++) occupancy[(130 + t) * w + x] = 1;
    for (int y = 130; y < 165; y++)
        for (int t = 0; t < 3; t++) occupancy[y * w + (30 + t)] = 1;
    for (int x = 90; x < 110; x++) {
        occupancy[140 * w + x] = 1; occupancy[141 * w + x] = 1;
        occupancy[159 * w + x] = 1; occupancy[160 * w + x] = 1;
    }
    for (int y = 140; y < 161; y++) {
        occupancy[y * w + 90] = 1; occupancy[y * w + 91] = 1;
        occupancy[y * w + 109] = 1; occupancy[y * w + 110] = 1;
    }
}

// ---------------------------------------------------------------------------
// Host: simulate lidar
// ---------------------------------------------------------------------------
void simulate_lidar(const std::vector<int>& occupancy,
    float rx, float ry, float rtheta,
    float ox, float oy, float res, int gw, int gh,
    float* beam_ranges)
{
    float step = res * 0.5f;
    for (int b = 0; b < NUM_BEAMS; b++) {
        float angle = rtheta + (float)b * BEAM_ANGLE_STEP - PI;
        float ca = cosf(angle), sa = sinf(angle);
        float range = 0.0f;
        bool hit = false;
        while (range < MAX_RANGE) {
            range += step;
            int gx = (int)((rx + range * ca - ox) / res);
            int gy = (int)((ry + range * sa - oy) / res);
            if (gx < 0 || gx >= gw || gy < 0 || gy >= gh) { hit = true; break; }
            if (occupancy[gy * gw + gx] == 1) { hit = true; break; }
        }
        beam_ranges[b] = hit ? range : MAX_RANGE;
    }
}

// ---------------------------------------------------------------------------
// Host: visualization helpers
// ---------------------------------------------------------------------------
cv::Point2i grid_to_pixel(int gx, int gy) {
    return cv::Point2i(gx * VIS_SCALE, (GRID_H - 1 - gy) * VIS_SCALE);
}

cv::Point2i world_to_pixel(float wx, float wy, float ox, float oy, float res) {
    int gx = (int)((wx - ox) / res);
    int gy = (int)((wy - oy) / res);
    return grid_to_pixel(gx, gy);
}

void draw_arrow(cv::Mat& img, cv::Point2i pt, float theta, cv::Scalar color, int length, int thickness) {
    int dx = (int)(length * cosf(theta));
    int dy = (int)(-length * sinf(theta));
    cv::arrowedLine(img, pt, cv::Point2i(pt.x + dx, pt.y + dy), color, thickness, cv::LINE_AA, 0, 0.3);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "emcl2: Expansion Resetting MCL (Ueda IROS 2004)" << std::endl;

    float origin_x = 0.0f, origin_y = 0.0f;

    // Build occupancy map
    std::vector<int> h_occupancy;
    build_map(h_occupancy, GRID_W, GRID_H);

    int* d_occupancy;
    CUDA_CHECK(cudaMalloc(&d_occupancy, GRID_W * GRID_H * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_occupancy, h_occupancy.data(),
                          GRID_W * GRID_H * sizeof(int), cudaMemcpyHostToDevice));

    // Build likelihood field
    float* d_likelihood_field;
    CUDA_CHECK(cudaMalloc(&d_likelihood_field, GRID_W * GRID_H * sizeof(float)));
    {
        int total = GRID_W * GRID_H;
        build_likelihood_field_kernel<<<(total + 255) / 256, 256>>>(
            d_occupancy, d_likelihood_field, GRID_W, GRID_H, SIGMA_HIT);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Particle memory
    const int np = N_PARTICLES;
    const int threads = 256;
    const int blocks = (np + threads - 1) / threads;

    float *d_px, *d_py, *d_ptheta, *d_pw;
    float *d_px_tmp, *d_py_tmp, *d_ptheta_tmp;
    CUDA_CHECK(cudaMalloc(&d_px, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ptheta, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pw, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_px_tmp, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py_tmp, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ptheta_tmp, np * sizeof(float)));

    float *d_wcum;
    CUDA_CHECK(cudaMalloc(&d_wcum, np * sizeof(float)));

    curandState* d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, np * sizeof(curandState)));
    init_curand_kernel<<<blocks, threads>>>(d_rng_states, 42ULL, np);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *d_max_w, *d_avg_w, *d_sum_w;
    CUDA_CHECK(cudaMalloc(&d_max_w, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_avg_w, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_w, sizeof(float)));

    float *d_est_x, *d_est_y, *d_est_theta;
    CUDA_CHECK(cudaMalloc(&d_est_x, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_est_y, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_est_theta, sizeof(float)));

    float *d_beam_ranges;
    CUDA_CHECK(cudaMalloc(&d_beam_ranges, NUM_BEAMS * sizeof(float)));

    // Host buffers
    std::vector<float> h_px(np), h_py(np), h_ptheta(np), h_pw(np);
    float h_beam_ranges[NUM_BEAMS];

    // Ground truth
    float gt_x = 5.0f, gt_y = 5.0f, gt_theta = 0.0f;
    float robot_v = 1.0f, robot_omega = 0.0f;

    std::mt19937 gen(12345);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    // Initialize particles near ground truth
    for (int i = 0; i < np; i++) {
        h_px[i] = gt_x + gauss(gen) * 1.0f;
        h_py[i] = gt_y + gauss(gen) * 1.0f;
        h_ptheta[i] = gt_theta + gauss(gen) * 0.5f;
        h_pw[i] = 1.0f / np;
    }
    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ptheta, h_ptheta.data(), np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pw, h_pw.data(), np * sizeof(float), cudaMemcpyHostToDevice));

    // Pre-render map
    cv::Mat map_img(IMG_H, IMG_W, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int gy = 0; gy < GRID_H; gy++)
        for (int gx = 0; gx < GRID_W; gx++)
            if (h_occupancy[gy * GRID_W + gx] == 1) {
                int px_x = gx * VIS_SCALE, px_y = (GRID_H - 1 - gy) * VIS_SCALE;
                cv::rectangle(map_img, cv::Point(px_x, px_y),
                    cv::Point(px_x + VIS_SCALE - 1, px_y + VIS_SCALE - 1),
                    cv::Scalar(0, 0, 0), -1);
            }

    // Visualization
    cv::namedWindow("emcl2", cv::WINDOW_NORMAL);
    cv::resizeWindow("emcl2", IMG_W, IMG_H);
    cv::VideoWriter video(
        "gif/emcl2.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, cv::Size(IMG_W, IMG_H));

    bool kidnapped = false;
    bool recovering = false;
    bool recovered = false;
    int recovery_count = 0;

    float time_val = 0.0f;
    int step = 0;

    while (time_val <= SIM_TIME) {
        time_val += DT;
        step++;

        // Robot trajectory: navigate through the map
        float map_w = GRID_W * GRID_RES;
        float map_h_m = GRID_H * GRID_RES;
        if (gt_x > map_w - 3.0f || gt_x < 3.0f) robot_omega = 0.3f;
        if (gt_y > map_h_m - 3.0f || gt_y < 3.0f) robot_omega = 0.3f;

        // Gentle curves
        if (step % 50 == 0) robot_omega = (uni(gen) - 0.5f) * 0.6f;

        // KIDNAP at t=10s: teleport robot
        if (time_val >= KIDNAP_TIME && !kidnapped) {
            kidnapped = true;
            recovering = true;
            recovered = false;
            recovery_count = 0;
            gt_x = 15.0f;
            gt_y = 15.0f;
            gt_theta = PI / 2.0f;
            robot_omega = 0.1f;
            printf("[%.1fs] KIDNAPPED! Robot teleported to (%.1f, %.1f)\n", time_val, gt_x, gt_y);
        }

        // Update ground truth
        gt_theta += robot_omega * DT;
        gt_x += robot_v * cosf(gt_theta) * DT;
        gt_y += robot_v * sinf(gt_theta) * DT;

        // Clamp to map
        if (gt_x < 1.0f) { gt_x = 1.0f; robot_omega = 0.5f; }
        if (gt_x > map_w - 1.0f) { gt_x = map_w - 1.0f; robot_omega = 0.5f; }
        if (gt_y < 1.0f) { gt_y = 1.0f; robot_omega = 0.5f; }
        if (gt_y > map_h_m - 1.0f) { gt_y = map_h_m - 1.0f; robot_omega = 0.5f; }

        // Simulate lidar
        simulate_lidar(h_occupancy, gt_x, gt_y, gt_theta,
                       origin_x, origin_y, GRID_RES, GRID_W, GRID_H, h_beam_ranges);
        CUDA_CHECK(cudaMemcpy(d_beam_ranges, h_beam_ranges,
                              NUM_BEAMS * sizeof(float), cudaMemcpyHostToDevice));

        // --- GPU: predict ---
        predict_kernel<<<blocks, threads>>>(
            d_px, d_py, d_ptheta, robot_v, robot_omega, DT, d_rng_states, np);

        // --- GPU: compute likelihood ---
        compute_likelihood_kernel<<<blocks, threads>>>(
            d_px, d_py, d_ptheta, d_pw,
            d_likelihood_field, d_beam_ranges,
            GRID_W, GRID_H, GRID_RES, origin_x, origin_y, np);

        // --- GPU: check if reset needed ---
        check_reset_kernel<<<1, threads, 2 * threads * sizeof(float)>>>(
            d_pw, np, d_max_w, d_avg_w, d_sum_w);

        float h_max_w, h_avg_w, h_sum_w;
        CUDA_CHECK(cudaMemcpy(&h_max_w, d_max_w, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_avg_w, d_avg_w, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_sum_w, d_sum_w, sizeof(float), cudaMemcpyDeviceToHost));

        // Expansion resetting: if max likelihood is very low, particles are lost
        if (h_max_w < EXPANSION_RESET_THRESHOLD) {
            // Expansion reset: spread particles with noise
            expansion_reset_kernel<<<blocks, threads>>>(
                d_px, d_py, d_ptheta, d_pw,
                EXPANSION_NOISE_XY, EXPANSION_NOISE_TH,
                d_occupancy, GRID_W, GRID_H,
                origin_x, origin_y, GRID_RES,
                d_rng_states, np);

            // Also scatter some particles randomly (sensor reset)
            int n_reset = np / 4;
            sensor_reset_kernel<<<(n_reset + 255) / 256, 256>>>(
                d_px, d_py, d_ptheta, d_pw,
                d_beam_ranges, d_occupancy, GRID_W, GRID_H,
                origin_x, origin_y, GRID_RES,
                d_rng_states, np, n_reset);

            // Recompute weights after reset
            compute_likelihood_kernel<<<blocks, threads>>>(
                d_px, d_py, d_ptheta, d_pw,
                d_likelihood_field, d_beam_ranges,
                GRID_W, GRID_H, GRID_RES, origin_x, origin_y, np);

            check_reset_kernel<<<1, threads, 2 * threads * sizeof(float)>>>(
                d_pw, np, d_max_w, d_avg_w, d_sum_w);
            CUDA_CHECK(cudaMemcpy(&h_sum_w, d_sum_w, sizeof(float), cudaMemcpyDeviceToHost));
        }

        // Normalize weights
        normalize_weights_kernel<<<blocks, threads>>>(d_pw, np, h_sum_w);

        // Weighted mean
        weighted_mean_kernel<<<1, threads, 4 * threads * sizeof(float)>>>(
            d_px, d_py, d_ptheta, d_pw, d_est_x, d_est_y, d_est_theta, np);

        float est_x, est_y, est_theta;
        CUDA_CHECK(cudaMemcpy(&est_x, d_est_x, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&est_y, d_est_y, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&est_theta, d_est_theta, sizeof(float), cudaMemcpyDeviceToHost));

        // Check recovery
        if (recovering) {
            float err = sqrtf((est_x - gt_x) * (est_x - gt_x) + (est_y - gt_y) * (est_y - gt_y));
            if (err < 1.5f) {
                recovery_count++;
                if (recovery_count > 20) {
                    recovered = true;
                    recovering = false;
                    printf("[%.1fs] RECOVERED! Error: %.2f m\n", time_val, err);
                }
            } else {
                recovery_count = 0;
            }
        }

        // Resample
        CUDA_CHECK(cudaMemcpy(h_pw.data(), d_pw, np * sizeof(float), cudaMemcpyDeviceToHost));
        float Neff_denom = 0.0f;
        for (int i = 0; i < np; i++) Neff_denom += h_pw[i] * h_pw[i];
        float Neff = 1.0f / Neff_denom;

        if (Neff < np / 2) {
            cumsum_kernel<<<1, 1>>>(d_pw, d_wcum, np);
            float rand_offset = uni(gen) / np;
            resample_kernel<<<blocks, threads>>>(
                d_px, d_py, d_ptheta,
                d_px_tmp, d_py_tmp, d_ptheta_tmp,
                d_wcum, 1.0f / np, rand_offset, np);
            CUDA_CHECK(cudaMemcpy(d_px, d_px_tmp, np * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_py, d_py_tmp, np * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_ptheta, d_ptheta_tmp, np * sizeof(float), cudaMemcpyDeviceToDevice));
            // Reset weights
            std::vector<float> pw_uni(np, 1.0f / np);
            CUDA_CHECK(cudaMemcpy(d_pw, pw_uni.data(), np * sizeof(float), cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Visualization ---
        CUDA_CHECK(cudaMemcpy(h_px.data(), d_px, np * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_py.data(), d_py, np * sizeof(float), cudaMemcpyDeviceToHost));

        cv::Mat bg = map_img.clone();

        // Draw particles (red)
        for (int i = 0; i < np; i++) {
            cv::Point2i pt = world_to_pixel(h_px[i], h_py[i], origin_x, origin_y, GRID_RES);
            if (pt.x >= 0 && pt.x < IMG_W && pt.y >= 0 && pt.y < IMG_H)
                cv::circle(bg, pt, 2, cv::Scalar(0, 0, 255), -1);
        }

        // Estimate (blue)
        cv::Point2i est_pt = world_to_pixel(est_x, est_y, origin_x, origin_y, GRID_RES);
        cv::circle(bg, est_pt, 8, cv::Scalar(255, 0, 0), -1);
        draw_arrow(bg, est_pt, est_theta, cv::Scalar(255, 0, 0), 20, 2);

        // Ground truth (green)
        cv::Point2i gt_pt = world_to_pixel(gt_x, gt_y, origin_x, origin_y, GRID_RES);
        cv::circle(bg, gt_pt, 8, cv::Scalar(0, 200, 0), -1);
        draw_arrow(bg, gt_pt, gt_theta, cv::Scalar(0, 200, 0), 20, 2);

        // Status text
        char buf[128];
        snprintf(buf, sizeof(buf), "t=%.1fs  N=%d", time_val, np);
        cv::putText(bg, buf, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

        if (kidnapped && recovering && !recovered) {
            if (h_max_w < EXPANSION_RESET_THRESHOLD) {
                cv::putText(bg, "KIDNAPPED!", cv::Point(IMG_W / 2 - 120, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
            } else {
                cv::putText(bg, "RECOVERING!", cv::Point(IMG_W / 2 - 140, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 165, 255), 3);
            }
        }
        if (recovered) {
            cv::putText(bg, "RECOVERED!", cv::Point(IMG_W / 2 - 130, 60),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 200, 0), 3);
        }

        cv::imshow("emcl2", bg);
        video.write(bg);
        cv::waitKey(5);
    }

    video.release();
    system("ffmpeg -y -i gif/emcl2.avi "
           "-vf 'fps=15,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/emcl2.gif 2>/dev/null");
    std::cout << "GIF saved to gif/emcl2.gif" << std::endl;

    cudaFree(d_px); cudaFree(d_py); cudaFree(d_ptheta); cudaFree(d_pw);
    cudaFree(d_px_tmp); cudaFree(d_py_tmp); cudaFree(d_ptheta_tmp);
    cudaFree(d_wcum); cudaFree(d_rng_states);
    cudaFree(d_max_w); cudaFree(d_avg_w); cudaFree(d_sum_w);
    cudaFree(d_est_x); cudaFree(d_est_y); cudaFree(d_est_theta);
    cudaFree(d_beam_ranges); cudaFree(d_likelihood_field); cudaFree(d_occupancy);

    return 0;
}
