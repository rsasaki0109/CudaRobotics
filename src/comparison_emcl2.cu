/*************************************************************************
    comparison_emcl2.cu
    Side-by-side comparison: Standard MCL vs emcl2 (Expansion Resetting MCL)
    Left panel:  Standard MCL (no expansion reset -- fails after kidnapping)
    Right panel: emcl2 (expansion + sensor resetting -- recovers)
    Demonstrates the advantage of expansion resetting for kidnapped robot
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
#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
#define PI 3.141592653f
#define DT 0.1f
#define SIM_TIME 35.0f

// Grid
#define GRID_W 200
#define GRID_H 200
#define GRID_RES 0.1f

// Particles
#define N_PARTICLES 500

// Likelihood field
#define SIGMA_HIT 0.2f
#define Z_HIT 0.9f
#define Z_RAND 0.1f

// Motion model
#define ALPHA1 0.1f
#define ALPHA2 0.1f
#define ALPHA3 0.1f
#define ALPHA4 0.1f

// Lidar
#define NUM_BEAMS 36
#define BEAM_ANGLE_STEP (2.0f * PI / NUM_BEAMS)
#define MAX_RANGE 10.0f

// Expansion resetting
#define EXPANSION_RATE 0.1f
#define RESET_THRESHOLD 0.5f
#define SENSOR_RESET_RATIO 0.2f
#define LIKELIHOOD_THRESHOLD 0.001f

// Visualization
#define VIS_SCALE 4
#define IMG_W (GRID_W * VIS_SCALE)
#define IMG_H (GRID_H * VIS_SCALE)
#define PANEL_W IMG_W
#define PANEL_H IMG_H
#define COMBINED_W (PANEL_W * 2)
#define COMBINED_H PANEL_H

// Kidnap timing
#define KIDNAP_TIME 10.0f
#define KIDNAP_X 15.0f
#define KIDNAP_Y 15.0f
#define KIDNAP_THETA 2.0f

#define RECOVERY_DIST_THRESHOLD 1.5f

// ===========================================================================
// CUDA Kernels (shared by both MCL and emcl2)
// ===========================================================================

__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void build_likelihood_field_kernel(
    const int* occupancy, float* likelihood_field,
    int width, int height, float sigma_hit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int cy = idx / width;
    int cx = idx % width;

    float min_dist = 1e6f;
    int search_radius = (int)(3.0f * sigma_hit / GRID_RES) + 1;
    if (search_radius > 50) search_radius = 50;

    for (int dy = -search_radius; dy <= search_radius; dy++) {
        for (int dx = -search_radius; dx <= search_radius; dx++) {
            int nx = cx + dx;
            int ny = cy + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (occupancy[ny * width + nx] == 1) {
                    float dist = sqrtf((float)(dx * dx + dy * dy)) * GRID_RES;
                    if (dist < min_dist) min_dist = dist;
                }
            }
        }
    }

    float prob = Z_HIT * expf(-0.5f * (min_dist * min_dist) / (sigma_hit * sigma_hit))
                 + Z_RAND / MAX_RANGE;
    likelihood_field[idx] = prob;
}

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
    float gamma_hat = curand_normal(&local_rng) * (ALPHA1 * fabsf(v) + ALPHA2 * fabsf(omega)) * 0.1f;

    float theta = ptheta[idx];

    if (fabsf(omega_hat) < 1e-6f) {
        px[idx] += v_hat * cosf(theta) * dt;
        py[idx] += v_hat * sinf(theta) * dt;
    } else {
        float r = v_hat / omega_hat;
        px[idx] += r * (sinf(theta + omega_hat * dt) - sinf(theta));
        py[idx] += r * (cosf(theta) - cosf(theta + omega_hat * dt));
    }
    ptheta[idx] += omega_hat * dt + gamma_hat * dt;

    float th = ptheta[idx];
    while (th > PI) th -= 2.0f * PI;
    while (th < -PI) th += 2.0f * PI;
    ptheta[idx] = th;

    rng_states[idx] = local_rng;
}

__global__ void compute_likelihood_kernel(
    float* px, float* py, float* ptheta, float* pw,
    const float* likelihood_field, const float* beam_ranges,
    int width, int height, float resolution,
    float origin_x, float origin_y, int np)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;

    float x = px[idx];
    float y = py[idx];
    float theta = ptheta[idx];
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
            if (lf > 1e-10f) log_w += logf(lf);
            else log_w += logf(1e-10f);
        } else {
            log_w += logf(Z_RAND / MAX_RANGE);
        }
    }

    pw[idx] = expf(log_w);
}

__global__ void check_reset_kernel(
    const float* pw, int np, int* out_bad_count, float* out_sum)
{
    extern __shared__ float sdata[];
    int* s_bad = (int*)sdata;
    float* s_sum = (float*)(s_bad + blockDim.x);

    int tid = threadIdx.x;
    int bad = 0;
    float wsum = 0.0f;

    for (int i = tid; i < np; i += blockDim.x) {
        float w = pw[i];
        wsum += w;
        if (w < LIKELIHOOD_THRESHOLD) bad++;
    }
    s_bad[tid] = bad;
    s_sum[tid] = wsum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_bad[tid] += s_bad[tid + s];
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_bad_count = s_bad[0];
        *out_sum = s_sum[0];
    }
}

__global__ void sum_weights_kernel(
    const float* pw, int np, float* out_sum)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float val = 0.0f;

    for (int i = tid; i < np; i += blockDim.x) {
        val += pw[i];
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) *out_sum = sdata[0];
}

__global__ void normalize_weights_kernel(float* pw, int np, float total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;
    if (total > 1e-30f) pw[idx] /= total;
    else pw[idx] = 1.0f / (float)np;
}

__global__ void expansion_reset_kernel(
    float* px, float* py, float* ptheta,
    curandState* rng_states, float expansion_scale, int np)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;

    curandState local_rng = rng_states[idx];
    px[idx] += curand_normal(&local_rng) * expansion_scale;
    py[idx] += curand_normal(&local_rng) * expansion_scale;
    ptheta[idx] += curand_normal(&local_rng) * expansion_scale * 0.5f;

    float th = ptheta[idx];
    while (th > PI) th -= 2.0f * PI;
    while (th < -PI) th += 2.0f * PI;
    ptheta[idx] = th;

    rng_states[idx] = local_rng;
}

__global__ void sensor_reset_kernel(
    float* px, float* py, float* ptheta,
    const float* beam_ranges,
    const float* likelihood_field,
    const int* occupancy,
    curandState* rng_states,
    float origin_x, float origin_y,
    float resolution,
    int width, int height,
    int start_idx, int count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    int idx = start_idx + tid;
    curandState local_rng = rng_states[idx];

    float best_x = 0.0f, best_y = 0.0f, best_theta = 0.0f;
    float best_lh = -1.0f;

    for (int attempt = 0; attempt < 20; attempt++) {
        float rx = origin_x + curand_uniform(&local_rng) * width * resolution;
        float ry = origin_y + curand_uniform(&local_rng) * height * resolution;
        float rtheta = (curand_uniform(&local_rng) - 0.5f) * 2.0f * PI;

        int gx = (int)((rx - origin_x) / resolution);
        int gy = (int)((ry - origin_y) / resolution);

        if (gx < 0 || gx >= width || gy < 0 || gy >= height) continue;
        if (occupancy[gy * width + gx] == 1) continue;

        float log_w = 0.0f;
        int valid_beams = 0;
        for (int b = 0; b < NUM_BEAMS; b += 4) {
            float range = beam_ranges[b];
            if (range >= MAX_RANGE) continue;

            float beam_angle = rtheta + (float)b * BEAM_ANGLE_STEP - PI;
            float ex = rx + range * cosf(beam_angle);
            float ey = ry + range * sinf(beam_angle);

            int egx = (int)((ex - origin_x) / resolution);
            int egy = (int)((ey - origin_y) / resolution);

            if (egx >= 0 && egx < width && egy >= 0 && egy < height) {
                float lf = likelihood_field[egy * width + egx];
                if (lf > 1e-10f) log_w += logf(lf);
                else log_w += logf(1e-10f);
                valid_beams++;
            }
        }

        if (valid_beams > 0) {
            float avg_lh = log_w / (float)valid_beams;
            if (avg_lh > best_lh) {
                best_lh = avg_lh;
                best_x = rx;
                best_y = ry;
                best_theta = rtheta;
            }
        }
    }

    if (best_lh > -1.0f) {
        px[idx] = best_x;
        py[idx] = best_y;
        ptheta[idx] = best_theta;
    } else {
        for (int attempt = 0; attempt < 50; attempt++) {
            float rx = origin_x + curand_uniform(&local_rng) * width * resolution;
            float ry = origin_y + curand_uniform(&local_rng) * height * resolution;
            int gx = (int)((rx - origin_x) / resolution);
            int gy = (int)((ry - origin_y) / resolution);
            if (gx >= 0 && gx < width && gy >= 0 && gy < height &&
                occupancy[gy * width + gx] == 0) {
                px[idx] = rx;
                py[idx] = ry;
                ptheta[idx] = (curand_uniform(&local_rng) - 0.5f) * 2.0f * PI;
                break;
            }
        }
    }

    rng_states[idx] = local_rng;
}

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
        vx += px[i] * w;
        vy += py[i] * w;
        vc += cosf(ptheta[i]) * w;
        vs += sinf(ptheta[i]) * w;
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

__global__ void cumsum_kernel(const float* pw, float* wcum, int np) {
    wcum[0] = pw[0];
    for (int i = 1; i < np; i++) wcum[i] = wcum[i - 1] + pw[i];
}

__global__ void resample_kernel(
    const float* px_in, const float* py_in, const float* ptheta_in,
    float* px_out, float* py_out, float* ptheta_out,
    const float* wcum, float base_step, float rand_offset,
    int np_in, int np_out)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np_out) return;

    float target = base_step * ip + rand_offset;
    int lo = 0, hi = np_in - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (wcum[mid] < target) lo = mid + 1;
        else hi = mid;
    }

    px_out[ip] = px_in[lo];
    py_out[ip] = py_in[lo];
    ptheta_out[ip] = ptheta_in[lo];
}

// ===========================================================================
// Host helpers
// ===========================================================================

void build_map(std::vector<int>& occupancy, int w, int h) {
    occupancy.assign(w * h, 0);
    for (int x = 0; x < w; x++) {
        occupancy[0 * w + x] = 1; occupancy[1 * w + x] = 1;
        occupancy[(h - 1) * w + x] = 1; occupancy[(h - 2) * w + x] = 1;
    }
    for (int y = 0; y < h; y++) {
        occupancy[y * w + 0] = 1; occupancy[y * w + 1] = 1;
        occupancy[y * w + (w - 1)] = 1; occupancy[y * w + (w - 2)] = 1;
    }
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

void simulate_lidar(
    const std::vector<int>& occupancy,
    float robot_x, float robot_y, float robot_theta,
    float origin_x, float origin_y, float resolution,
    int grid_w, int grid_h, float* beam_ranges)
{
    float step = resolution * 0.5f;
    for (int b = 0; b < NUM_BEAMS; b++) {
        float angle = robot_theta + (float)b * BEAM_ANGLE_STEP - PI;
        float ca = cosf(angle);
        float sa = sinf(angle);
        float range = 0.0f;
        bool hit = false;
        while (range < MAX_RANGE) {
            range += step;
            float wx = robot_x + range * ca;
            float wy = robot_y + range * sa;
            int gx = (int)((wx - origin_x) / resolution);
            int gy = (int)((wy - origin_y) / resolution);
            if (gx < 0 || gx >= grid_w || gy < 0 || gy >= grid_h) { hit = true; break; }
            if (occupancy[gy * grid_w + gx] == 1) { hit = true; break; }
        }
        beam_ranges[b] = hit ? range : MAX_RANGE;
    }
}

cv::Point2i grid_to_pixel(int gx, int gy) {
    return cv::Point2i(gx * VIS_SCALE, (GRID_H - 1 - gy) * VIS_SCALE);
}

cv::Point2i world_to_pixel(float wx, float wy, float origin_x, float origin_y, float resolution) {
    int gx = (int)((wx - origin_x) / resolution);
    int gy = (int)((wy - origin_y) / resolution);
    return grid_to_pixel(gx, gy);
}

void draw_arrow(cv::Mat& img, cv::Point2i pt, float theta, cv::Scalar color, int length, int thickness) {
    int dx = (int)(length * cosf(theta));
    int dy = (int)(-length * sinf(theta));
    cv::Point2i tip(pt.x + dx, pt.y + dy);
    cv::arrowedLine(img, pt, tip, color, thickness, cv::LINE_AA, 0, 0.3);
}

// ===========================================================================
// Per-filter state on GPU
// ===========================================================================
struct FilterGPU {
    float *d_px, *d_py, *d_ptheta, *d_pw;
    float *d_px_tmp, *d_py_tmp, *d_ptheta_tmp;
    float *d_wcum;
    curandState *d_rng_states;
    int *d_bad_count;
    float *d_sum;
    float *d_est_x, *d_est_y, *d_est_theta;

    void allocate(int np, unsigned long long seed) {
        CUDA_CHECK(cudaMalloc(&d_px,         np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_py,         np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ptheta,     np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pw,         np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_px_tmp,     np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_py_tmp,     np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ptheta_tmp, np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_wcum,       np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_states, np * sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&d_bad_count,  sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sum,        sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_est_x,      sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_est_y,      sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_est_theta,  sizeof(float)));

        int threads = 256;
        int blocks = (np + threads - 1) / threads;
        init_curand_kernel<<<blocks, threads>>>(d_rng_states, seed, np);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void free_all() {
        cudaFree(d_px); cudaFree(d_py); cudaFree(d_ptheta); cudaFree(d_pw);
        cudaFree(d_px_tmp); cudaFree(d_py_tmp); cudaFree(d_ptheta_tmp);
        cudaFree(d_wcum); cudaFree(d_rng_states);
        cudaFree(d_bad_count); cudaFree(d_sum);
        cudaFree(d_est_x); cudaFree(d_est_y); cudaFree(d_est_theta);
    }
};

// ===========================================================================
// Main
// ===========================================================================
int main() {
    std::cout << "Comparison: Standard MCL vs emcl2 (Expansion Resetting MCL)" << std::endl;

    float origin_x = 0.0f, origin_y = 0.0f;

    // Build map
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
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        build_likelihood_field_kernel<<<blocks, threads>>>(
            d_occupancy, d_likelihood_field, GRID_W, GRID_H, SIGMA_HIT);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<float> h_likelihood_field(GRID_W * GRID_H);
    CUDA_CHECK(cudaMemcpy(h_likelihood_field.data(), d_likelihood_field,
                          GRID_W * GRID_H * sizeof(float), cudaMemcpyDeviceToHost));

    // Beam ranges on device
    float *d_beam_ranges;
    CUDA_CHECK(cudaMalloc(&d_beam_ranges, NUM_BEAMS * sizeof(float)));

    // Allocate two filters: standard MCL and emcl2
    FilterGPU mcl, emcl;
    mcl.allocate(N_PARTICLES, 42ULL);
    emcl.allocate(N_PARTICLES, 123ULL);

    // Host buffers
    std::vector<float> h_px(N_PARTICLES), h_py(N_PARTICLES), h_ptheta(N_PARTICLES);
    float h_beam_ranges[NUM_BEAMS];

    // Initialize particles
    float gt_x = 5.0f, gt_y = 5.0f, gt_theta = 0.0f;

    std::mt19937 gen(12345);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    float init_spread_xy = 1.0f, init_spread_th = 0.5f;

    // Initialize both filters identically
    std::vector<float> init_px(N_PARTICLES), init_py(N_PARTICLES), init_pth(N_PARTICLES), init_pw(N_PARTICLES);
    for (int i = 0; i < N_PARTICLES; i++) {
        init_px[i]  = gt_x + gauss(gen) * init_spread_xy;
        init_py[i]  = gt_y + gauss(gen) * init_spread_xy;
        init_pth[i] = gt_theta + gauss(gen) * init_spread_th;
        init_pw[i]  = 1.0f / N_PARTICLES;
    }

    // Upload to both filters
    FilterGPU* filters[2] = {&mcl, &emcl};
    for (int f = 0; f < 2; f++) {
        CUDA_CHECK(cudaMemcpy(filters[f]->d_px,     init_px.data(),  N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(filters[f]->d_py,     init_py.data(),  N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(filters[f]->d_ptheta, init_pth.data(), N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(filters[f]->d_pw,     init_pw.data(),  N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Robot trajectory
    float robot_v = 1.0f, robot_omega = 0.0f;

    // Visualization
    cv::namedWindow("comparison_emcl2", cv::WINDOW_NORMAL);
    cv::resizeWindow("comparison_emcl2", COMBINED_W, COMBINED_H);

    cv::VideoWriter video(
        "gif/comparison_emcl2.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
        cv::Size(COMBINED_W, COMBINED_H));

    // Pre-render map
    cv::Mat map_img(IMG_H, IMG_W, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int gy = 0; gy < GRID_H; gy++) {
        for (int gx = 0; gx < GRID_W; gx++) {
            if (h_occupancy[gy * GRID_W + gx] == 1) {
                int px_x = gx * VIS_SCALE;
                int px_y = (GRID_H - 1 - gy) * VIS_SCALE;
                cv::rectangle(map_img,
                    cv::Point(px_x, px_y),
                    cv::Point(px_x + VIS_SCALE - 1, px_y + VIS_SCALE - 1),
                    cv::Scalar(0, 0, 0), -1);
            }
        }
    }

    // Simulation state
    float time_val = 0.0f;
    int step_count = 0;
    bool kidnapped = false;
    int mcl_reset_count = 0, emcl_reset_count = 0;
    bool mcl_recovered = false, emcl_recovered = false;
    int mcl_consec = 0, emcl_consec = 0;
    float mcl_recovery_time = -1.0f, emcl_recovery_time = -1.0f;
    float kidnap_actual_time = -1.0f;

    int threads = 256;
    int pblocks = (N_PARTICLES + threads - 1) / threads;

    std::cout << "Starting simulation... Kidnap at t=" << KIDNAP_TIME << "s" << std::endl;

    while (time_val <= SIM_TIME) {
        time_val += DT;
        step_count++;

        // --- Robot control ---
        float targets[][2] = {
            {15.0f, 5.0f}, {15.0f, 15.0f}, {5.0f, 15.0f},
            {5.0f, 8.0f}, {12.0f, 8.0f}, {12.0f, 17.0f},
            {3.0f, 17.0f}, {3.0f, 5.0f}
        };
        int n_targets = 8;
        int wp_idx = ((int)(time_val / 5.0f)) % n_targets;

        float target_x = targets[wp_idx][0];
        float target_y = targets[wp_idx][1];
        float dx_t = target_x - gt_x;
        float dy_t = target_y - gt_y;
        float target_angle = atan2f(dy_t, dx_t);

        float angle_diff = target_angle - gt_theta;
        while (angle_diff > PI) angle_diff -= 2.0f * PI;
        while (angle_diff < -PI) angle_diff += 2.0f * PI;

        robot_v = 1.0f;
        robot_omega = angle_diff * 2.0f;
        if (robot_omega > 1.5f) robot_omega = 1.5f;
        if (robot_omega < -1.5f) robot_omega = -1.5f;

        // --- Kidnap ---
        if (!kidnapped && time_val >= KIDNAP_TIME) {
            kidnapped = true;
            kidnap_actual_time = time_val;
            gt_x = KIDNAP_X;
            gt_y = KIDNAP_Y;
            gt_theta = KIDNAP_THETA;
            std::cout << "KIDNAPPED at t=" << time_val << "s!" << std::endl;
        }

        // --- Update ground truth ---
        if (fabsf(robot_omega) < 1e-6f) {
            gt_x += robot_v * cosf(gt_theta) * DT;
            gt_y += robot_v * sinf(gt_theta) * DT;
        } else {
            float r = robot_v / robot_omega;
            gt_x += r * (sinf(gt_theta + robot_omega * DT) - sinf(gt_theta));
            gt_y += r * (cosf(gt_theta) - cosf(gt_theta + robot_omega * DT));
        }
        gt_theta += robot_omega * DT;
        while (gt_theta > PI) gt_theta -= 2.0f * PI;
        while (gt_theta < -PI) gt_theta += 2.0f * PI;

        float map_min = 0.5f;
        float map_max_x = GRID_W * GRID_RES - 0.5f;
        float map_max_y = GRID_H * GRID_RES - 0.5f;
        gt_x = std::max(map_min, std::min(map_max_x, gt_x));
        gt_y = std::max(map_min, std::min(map_max_y, gt_y));

        // --- Simulate lidar ---
        simulate_lidar(h_occupancy, gt_x, gt_y, gt_theta,
                       origin_x, origin_y, GRID_RES, GRID_W, GRID_H, h_beam_ranges);
        for (int b = 0; b < NUM_BEAMS; b++) {
            h_beam_ranges[b] += gauss(gen) * 0.05f;
            if (h_beam_ranges[b] < 0.0f) h_beam_ranges[b] = 0.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_beam_ranges, h_beam_ranges,
                              NUM_BEAMS * sizeof(float), cudaMemcpyHostToDevice));

        // =====================================================
        // Run both filters
        // =====================================================
        float est_x[2], est_y[2], est_theta_arr[2];
        float err[2];
        bool did_reset[2] = {false, false};
        float bad_frac[2] = {0, 0};

        // Particle data for visualization
        std::vector<float> vis_px[2], vis_py[2];
        vis_px[0].resize(N_PARTICLES); vis_py[0].resize(N_PARTICLES);
        vis_px[1].resize(N_PARTICLES); vis_py[1].resize(N_PARTICLES);

        for (int f = 0; f < 2; f++) {
            FilterGPU& filt = *filters[f];
            bool is_emcl = (f == 1);

            // Predict
            predict_kernel<<<pblocks, threads>>>(
                filt.d_px, filt.d_py, filt.d_ptheta,
                robot_v, robot_omega, DT,
                filt.d_rng_states, N_PARTICLES);

            // Compute likelihood
            compute_likelihood_kernel<<<pblocks, threads>>>(
                filt.d_px, filt.d_py, filt.d_ptheta, filt.d_pw,
                d_likelihood_field, d_beam_ranges,
                GRID_W, GRID_H, GRID_RES,
                origin_x, origin_y, N_PARTICLES);

            // Check reset condition
            int shared_size = threads * sizeof(int) + threads * sizeof(float);
            check_reset_kernel<<<1, threads, shared_size>>>(
                filt.d_pw, N_PARTICLES, filt.d_bad_count, filt.d_sum);
            CUDA_CHECK(cudaDeviceSynchronize());

            int h_bad_count;
            float h_sum;
            CUDA_CHECK(cudaMemcpy(&h_bad_count, filt.d_bad_count, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_sum, filt.d_sum, sizeof(float), cudaMemcpyDeviceToHost));

            bad_frac[f] = (float)h_bad_count / (float)N_PARTICLES;
            bool need_reset = (bad_frac[f] > RESET_THRESHOLD);

            // Normalize
            normalize_weights_kernel<<<pblocks, threads>>>(filt.d_pw, N_PARTICLES, h_sum);
            CUDA_CHECK(cudaDeviceSynchronize());

            if (is_emcl && need_reset) {
                // emcl2: expansion + sensor reset
                did_reset[f] = true;
                emcl_reset_count++;

                float expansion_scale = EXPANSION_RATE * (1.0f + bad_frac[f] * 5.0f);
                int n_sensor = (int)(N_PARTICLES * SENSOR_RESET_RATIO);
                int n_expansion = N_PARTICLES - n_sensor;

                expansion_reset_kernel<<<pblocks, threads>>>(
                    filt.d_px, filt.d_py, filt.d_ptheta,
                    filt.d_rng_states, expansion_scale, n_expansion);

                if (n_sensor > 0) {
                    int sr_blocks = (n_sensor + threads - 1) / threads;
                    sensor_reset_kernel<<<sr_blocks, threads>>>(
                        filt.d_px, filt.d_py, filt.d_ptheta,
                        d_beam_ranges, d_likelihood_field, d_occupancy,
                        filt.d_rng_states,
                        origin_x, origin_y, GRID_RES, GRID_W, GRID_H,
                        n_expansion, n_sensor);
                }

                std::vector<float> pw_uniform(N_PARTICLES, 1.0f / (float)N_PARTICLES);
                CUDA_CHECK(cudaMemcpy(filt.d_pw, pw_uniform.data(), N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaDeviceSynchronize());
            } else {
                // Standard resampling
                cumsum_kernel<<<1, 1>>>(filt.d_pw, filt.d_wcum, N_PARTICLES);
                CUDA_CHECK(cudaDeviceSynchronize());

                float rand_offset = uni(gen) / (float)N_PARTICLES;
                float base_step = 1.0f / (float)N_PARTICLES;

                resample_kernel<<<pblocks, threads>>>(
                    filt.d_px, filt.d_py, filt.d_ptheta,
                    filt.d_px_tmp, filt.d_py_tmp, filt.d_ptheta_tmp,
                    filt.d_wcum, base_step, rand_offset,
                    N_PARTICLES, N_PARTICLES);
                CUDA_CHECK(cudaDeviceSynchronize());

                CUDA_CHECK(cudaMemcpy(filt.d_px,     filt.d_px_tmp,     N_PARTICLES * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(filt.d_py,     filt.d_py_tmp,     N_PARTICLES * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(filt.d_ptheta, filt.d_ptheta_tmp, N_PARTICLES * sizeof(float), cudaMemcpyDeviceToDevice));

                std::vector<float> pw_uniform(N_PARTICLES, 1.0f / (float)N_PARTICLES);
                CUDA_CHECK(cudaMemcpy(filt.d_pw, pw_uniform.data(), N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
            }

            // Recompute weights for estimation
            compute_likelihood_kernel<<<pblocks, threads>>>(
                filt.d_px, filt.d_py, filt.d_ptheta, filt.d_pw,
                d_likelihood_field, d_beam_ranges,
                GRID_W, GRID_H, GRID_RES,
                origin_x, origin_y, N_PARTICLES);

            sum_weights_kernel<<<1, threads, threads * sizeof(float)>>>(
                filt.d_pw, N_PARTICLES, filt.d_sum);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_sum, filt.d_sum, sizeof(float), cudaMemcpyDeviceToHost));
            normalize_weights_kernel<<<pblocks, threads>>>(filt.d_pw, N_PARTICLES, h_sum);

            weighted_mean_kernel<<<1, threads, 4 * threads * sizeof(float)>>>(
                filt.d_px, filt.d_py, filt.d_ptheta, filt.d_pw,
                filt.d_est_x, filt.d_est_y, filt.d_est_theta, N_PARTICLES);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(&est_x[f],         filt.d_est_x,     sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&est_y[f],         filt.d_est_y,     sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&est_theta_arr[f], filt.d_est_theta, sizeof(float), cudaMemcpyDeviceToHost));

            float ex = gt_x - est_x[f];
            float ey = gt_y - est_y[f];
            err[f] = sqrtf(ex * ex + ey * ey);

            // Read particles for visualization
            CUDA_CHECK(cudaMemcpy(vis_px[f].data(), filt.d_px, N_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(vis_py[f].data(), filt.d_py, N_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // Check recovery
        if (kidnapped) {
            if (!mcl_recovered) {
                if (err[0] < RECOVERY_DIST_THRESHOLD) { mcl_consec++; if (mcl_consec > 20) { mcl_recovered = true; mcl_recovery_time = time_val; } }
                else mcl_consec = 0;
            }
            if (!emcl_recovered) {
                if (err[1] < RECOVERY_DIST_THRESHOLD) { emcl_consec++; if (emcl_consec > 20) { emcl_recovered = true; emcl_recovery_time = time_val; } }
                else emcl_consec = 0;
            }
        }

        // =====================================================
        // Visualization
        // =====================================================
        cv::Mat combined(COMBINED_H, COMBINED_W, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Mat left  = combined(cv::Rect(0, 0, PANEL_W, PANEL_H));
        cv::Mat right = combined(cv::Rect(PANEL_W, 0, PANEL_W, PANEL_H));

        const char* titles[2] = {"Standard MCL", "emcl2 (Expansion Reset)"};
        cv::Mat* panels[2] = {&left, &right};

        for (int f = 0; f < 2; f++) {
            cv::Mat& frame = *panels[f];
            map_img.copyTo(frame);

            // Draw particles (red)
            for (int i = 0; i < N_PARTICLES; i++) {
                cv::Point2i pt = world_to_pixel(vis_px[f][i], vis_py[f][i], origin_x, origin_y, GRID_RES);
                cv::circle(frame, pt, 2, cv::Scalar(0, 0, 255), -1);
            }

            // Ground truth (green)
            cv::Point2i gt_pt = world_to_pixel(gt_x, gt_y, origin_x, origin_y, GRID_RES);
            draw_arrow(frame, gt_pt, gt_theta, cv::Scalar(0, 200, 0), 20, 2);
            cv::circle(frame, gt_pt, 4, cv::Scalar(0, 200, 0), -1);

            // Estimate (blue)
            cv::Point2i est_pt = world_to_pixel(est_x[f], est_y[f], origin_x, origin_y, GRID_RES);
            draw_arrow(frame, est_pt, est_theta_arr[f], cv::Scalar(255, 0, 0), 20, 2);
            cv::circle(frame, est_pt, 4, cv::Scalar(255, 0, 0), -1);

            // Title
            char buf[128];
            snprintf(buf, sizeof(buf), "%s", titles[f]);
            cv::putText(frame, buf, cv::Point(10, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

            snprintf(buf, sizeof(buf), "err=%.2fm", err[f]);
            cv::putText(frame, buf, cv::Point(10, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

            // Status
            if (!kidnapped) {
                cv::putText(frame, "Tracking", cv::Point(10, IMG_H - 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 150, 0), 2);
            } else {
                bool rec = (f == 0) ? mcl_recovered : emcl_recovered;
                if (!rec) {
                    cv::putText(frame, "KIDNAPPED!", cv::Point(PANEL_W / 2 - 80, PANEL_H / 2),
                                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
                    if (f == 1) {
                        cv::putText(frame, "RECOVERING...", cv::Point(10, IMG_H - 20),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 140, 255), 2);
                    } else {
                        cv::putText(frame, "LOST", cv::Point(10, IMG_H - 20),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 200), 2);
                    }
                } else {
                    cv::putText(frame, "RECOVERED!", cv::Point(PANEL_W / 2 - 80, PANEL_H / 2),
                                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 200, 0), 2);
                }
            }
        }

        // Divider
        cv::line(combined, cv::Point(PANEL_W, 0), cv::Point(PANEL_W, COMBINED_H),
                 cv::Scalar(100, 100, 100), 2);

        // Time
        char buf[128];
        snprintf(buf, sizeof(buf), "t=%.1fs  NP=%d", time_val, N_PARTICLES);
        cv::putText(combined, buf, cv::Point(COMBINED_W / 2 - 80, COMBINED_H - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);

        cv::imshow("comparison_emcl2", combined);
        video.write(combined);
        cv::waitKey(5);

        if (step_count % 50 == 0) {
            printf("t=%.1f  MCL err=%.2f  emcl2 err=%.2f  emcl2 resets=%d\n",
                   time_val, err[0], err[1], emcl_reset_count);
        }
    }

    video.release();
    cv::destroyAllWindows();

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Standard MCL: " << (mcl_recovered ? "Recovered" : "FAILED to recover") << std::endl;
    std::cout << "  emcl2:        " << (emcl_recovered ? "Recovered" : "Did not recover in time") << std::endl;
    if (emcl_recovered)
        std::cout << "  emcl2 recovery time: " << (emcl_recovery_time - kidnap_actual_time) << "s (" << emcl_reset_count << " resets)" << std::endl;

    // Convert to gif
    int ret = system("which ffmpeg > /dev/null 2>&1 && "
        "ffmpeg -y -i gif/comparison_emcl2.avi "
        "-vf \"fps=15,scale=800:-1:flags=lanczos\" "
        "-gifflags +transdiff "
        "gif/comparison_emcl2.gif 2>/dev/null && "
        "echo 'GIF saved to gif/comparison_emcl2.gif' || echo 'ffmpeg not available, skipping gif'");
    (void)ret;

    // Cleanup
    cudaFree(d_occupancy);
    cudaFree(d_likelihood_field);
    cudaFree(d_beam_ranges);
    mcl.free_all();
    emcl.free_all();

    return 0;
}
