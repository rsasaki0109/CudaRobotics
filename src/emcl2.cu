/*************************************************************************
    > File Name: emcl2.cu
    > CUDA-parallelized Expansion Resetting MCL (emcl2)
    > Based on: Ryuichi Ueda, "Expansion resetting for recovery from
    >   fatal error in Monte Carlo localization" (IROS 2004)
    > Features: Fixed particle count (no KLD), expansion resetting,
    >           sensor resetting, blended recovery
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
#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
#define PI 3.141592653f
#define DT 0.1f
#define SIM_TIME 40.0f

// Grid
#define GRID_W 200
#define GRID_H 200
#define GRID_RES 0.1f  // meters per cell

// Particles (fixed count -- no KLD-sampling)
#define N_PARTICLES 500

// Likelihood field
#define SIGMA_HIT 0.2f
#define Z_HIT 0.9f
#define Z_RAND 0.1f

// Motion model (velocity model noise params)
#define ALPHA1 0.1f
#define ALPHA2 0.1f
#define ALPHA3 0.1f
#define ALPHA4 0.1f

// Lidar
#define NUM_BEAMS 36
#define BEAM_ANGLE_STEP (2.0f * PI / NUM_BEAMS)
#define MAX_RANGE 10.0f

// Expansion resetting parameters
#define EXPANSION_RATE 0.1f       // noise scaling for expansion
#define RESET_THRESHOLD 0.5f      // fraction of bad particles to trigger reset
#define SENSOR_RESET_RATIO 0.2f   // fraction of particles from sensor reset
#define LIKELIHOOD_THRESHOLD 0.001f // "bad" particle likelihood threshold

// Visualization
#define VIS_SCALE 4
#define IMG_W (GRID_W * VIS_SCALE)
#define IMG_H (GRID_H * VIS_SCALE)

// Kidnap timing
#define KIDNAP_TIME 10.0f
#define KIDNAP_X 15.0f
#define KIDNAP_Y 15.0f
#define KIDNAP_THETA 2.0f

// Recovery detection
#define RECOVERY_DIST_THRESHOLD 1.5f

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
// Kernel: build likelihood field
// ---------------------------------------------------------------------------
__global__ void build_likelihood_field_kernel(
    const int* occupancy,
    float* likelihood_field,
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

// ---------------------------------------------------------------------------
// Kernel 1: predict_kernel - motion model with cuRAND noise
// 1 thread per particle
// ---------------------------------------------------------------------------
__global__ void predict_kernel(
    float* px, float* py, float* ptheta,
    float v, float omega, float dt,
    curandState* rng_states,
    int np)
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

// ---------------------------------------------------------------------------
// Kernel 2: compute_likelihood_kernel - lookup likelihood field per beam
// 1 thread per particle
// ---------------------------------------------------------------------------
__global__ void compute_likelihood_kernel(
    float* px, float* py, float* ptheta,
    float* pw,
    const float* likelihood_field,
    const float* beam_ranges,
    int width, int height,
    float resolution,
    float origin_x, float origin_y,
    int np)
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
            if (lf > 1e-10f) {
                log_w += logf(lf);
            } else {
                log_w += logf(1e-10f);
            }
        } else {
            log_w += logf(Z_RAND / MAX_RANGE);
        }
    }

    pw[idx] = expf(log_w);
}

// ---------------------------------------------------------------------------
// Kernel 3: check_reset_kernel - parallel reduction to count "bad" particles
// A particle is "bad" if its likelihood is below LIKELIHOOD_THRESHOLD
// Also computes total weight sum for normalization
// ---------------------------------------------------------------------------
__global__ void check_reset_kernel(
    const float* pw, int np,
    int* out_bad_count, float* out_sum)
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

// ---------------------------------------------------------------------------
// Kernel: normalize weights
// ---------------------------------------------------------------------------
__global__ void normalize_weights_kernel(float* pw, int np, float total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;
    if (total > 1e-30f) {
        pw[idx] /= total;
    } else {
        pw[idx] = 1.0f / (float)np;
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: expansion_reset_kernel
// For each particle, add scaled noise proportional to error severity.
// Particles "spread outward" from their current positions.
// ---------------------------------------------------------------------------
__global__ void expansion_reset_kernel(
    float* px, float* py, float* ptheta,
    curandState* rng_states,
    float expansion_scale,
    int np)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;

    curandState local_rng = rng_states[idx];

    // Spread outward: add Gaussian noise scaled by expansion_scale
    px[idx] += curand_normal(&local_rng) * expansion_scale;
    py[idx] += curand_normal(&local_rng) * expansion_scale;
    ptheta[idx] += curand_normal(&local_rng) * expansion_scale * 0.5f;

    // Wrap angle
    float th = ptheta[idx];
    while (th > PI) th -= 2.0f * PI;
    while (th < -PI) th += 2.0f * PI;
    ptheta[idx] = th;

    rng_states[idx] = local_rng;
}

// ---------------------------------------------------------------------------
// Kernel 5: sensor_reset_kernel
// Generate particles at positions where sensor readings match well.
// Each thread generates one candidate particle by sampling from the
// likelihood field: pick a random beam, place particle at beam endpoint
// shifted to match map features.
// ---------------------------------------------------------------------------
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

    // Try to find a good position by sampling from high-likelihood regions
    float best_x = 0.0f, best_y = 0.0f, best_theta = 0.0f;
    float best_lh = -1.0f;

    for (int attempt = 0; attempt < 20; attempt++) {
        // Random position in free space
        float rx = origin_x + curand_uniform(&local_rng) * width * resolution;
        float ry = origin_y + curand_uniform(&local_rng) * height * resolution;
        float rtheta = (curand_uniform(&local_rng) - 0.5f) * 2.0f * PI;

        int gx = (int)((rx - origin_x) / resolution);
        int gy = (int)((ry - origin_y) / resolution);

        if (gx < 0 || gx >= width || gy < 0 || gy >= height) continue;
        if (occupancy[gy * width + gx] == 1) continue;

        // Evaluate likelihood at this position
        float log_w = 0.0f;
        int valid_beams = 0;
        for (int b = 0; b < NUM_BEAMS; b += 4) { // Sample every 4th beam for speed
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
        // Fallback: random free-space position
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

// ---------------------------------------------------------------------------
// Kernel: weighted mean estimation (single-block reduction)
// ---------------------------------------------------------------------------
__global__ void weighted_mean_kernel(
    const float* px, const float* py, const float* ptheta,
    const float* pw, float* out_x, float* out_y, float* out_theta,
    int np)
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
    sx[tid] = vx;
    sy[tid] = vy;
    sc[tid] = vc;
    ss[tid] = vs;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sx[tid] += sx[tid + s];
            sy[tid] += sy[tid + s];
            sc[tid] += sc[tid + s];
            ss[tid] += ss[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_x = sx[0];
        *out_y = sy[0];
        *out_theta = atan2f(ss[0], sc[0]);
    }
}

// ---------------------------------------------------------------------------
// Kernel: cumulative sum (sequential, single thread)
// ---------------------------------------------------------------------------
__global__ void cumsum_kernel(const float* pw, float* wcum, int np) {
    wcum[0] = pw[0];
    for (int i = 1; i < np; i++) {
        wcum[i] = wcum[i - 1] + pw[i];
    }
}

// ---------------------------------------------------------------------------
// Kernel 6: resample_kernel - systematic resampling
// ---------------------------------------------------------------------------
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

    px_out[ip]     = px_in[lo];
    py_out[ip]     = py_in[lo];
    ptheta_out[ip] = ptheta_in[lo];
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
void simulate_lidar(
    const std::vector<int>& occupancy,
    float robot_x, float robot_y, float robot_theta,
    float origin_x, float origin_y, float resolution,
    int grid_w, int grid_h,
    float* beam_ranges)
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

            if (gx < 0 || gx >= grid_w || gy < 0 || gy >= grid_h) {
                hit = true;
                break;
            }
            if (occupancy[gy * grid_w + gx] == 1) {
                hit = true;
                break;
            }
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "CUDA emcl2 (Expansion Resetting Monte Carlo Localization)" << std::endl;
    std::cout << "Based on: Ueda, 'Expansion resetting for recovery from fatal error" << std::endl;
    std::cout << "          in Monte Carlo localization' (IROS 2004)" << std::endl;

    float origin_x = 0.0f;
    float origin_y = 0.0f;

    // ------------------------------------------
    // Build occupancy map
    // ------------------------------------------
    std::vector<int> h_occupancy;
    build_map(h_occupancy, GRID_W, GRID_H);

    int* d_occupancy;
    CUDA_CHECK(cudaMalloc(&d_occupancy, GRID_W * GRID_H * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_occupancy, h_occupancy.data(),
                          GRID_W * GRID_H * sizeof(int), cudaMemcpyHostToDevice));

    // ------------------------------------------
    // Build likelihood field on GPU
    // ------------------------------------------
    float* d_likelihood_field;
    CUDA_CHECK(cudaMalloc(&d_likelihood_field, GRID_W * GRID_H * sizeof(float)));

    {
        int total_cells = GRID_W * GRID_H;
        int threads = 256;
        int blocks = (total_cells + threads - 1) / threads;
        build_likelihood_field_kernel<<<blocks, threads>>>(
            d_occupancy, d_likelihood_field,
            GRID_W, GRID_H, SIGMA_HIT);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<float> h_likelihood_field(GRID_W * GRID_H);
    CUDA_CHECK(cudaMemcpy(h_likelihood_field.data(), d_likelihood_field,
                          GRID_W * GRID_H * sizeof(float), cudaMemcpyDeviceToHost));

    // ------------------------------------------
    // Particle memory (fixed N_PARTICLES)
    // ------------------------------------------
    float *d_px, *d_py, *d_ptheta, *d_pw;
    float *d_px_tmp, *d_py_tmp, *d_ptheta_tmp;
    CUDA_CHECK(cudaMalloc(&d_px,         N_PARTICLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py,         N_PARTICLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ptheta,     N_PARTICLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pw,         N_PARTICLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_px_tmp,     N_PARTICLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py_tmp,     N_PARTICLES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ptheta_tmp, N_PARTICLES * sizeof(float)));

    float *d_wcum;
    CUDA_CHECK(cudaMalloc(&d_wcum, N_PARTICLES * sizeof(float)));

    // cuRAND states
    curandState* d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, N_PARTICLES * sizeof(curandState)));
    {
        int threads = 256;
        int blocks = (N_PARTICLES + threads - 1) / threads;
        init_curand_kernel<<<blocks, threads>>>(d_rng_states, 42ULL, N_PARTICLES);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Reduction outputs
    int *d_bad_count;
    float *d_sum;
    CUDA_CHECK(cudaMalloc(&d_bad_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));

    // Weighted mean outputs
    float *d_est_x, *d_est_y, *d_est_theta;
    CUDA_CHECK(cudaMalloc(&d_est_x,     sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_est_y,     sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_est_theta, sizeof(float)));

    // Beam ranges on device
    float *d_beam_ranges;
    CUDA_CHECK(cudaMalloc(&d_beam_ranges, NUM_BEAMS * sizeof(float)));

    // Host buffers
    std::vector<float> h_px(N_PARTICLES), h_py(N_PARTICLES), h_ptheta(N_PARTICLES), h_pw(N_PARTICLES);
    float h_beam_ranges[NUM_BEAMS];

    // ------------------------------------------
    // Initialize particles around ground truth
    // ------------------------------------------
    float gt_x = 5.0f;
    float gt_y = 5.0f;
    float gt_theta = 0.0f;

    std::mt19937 gen(12345);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    float init_spread_xy = 1.0f;
    float init_spread_th = 0.5f;
    for (int i = 0; i < N_PARTICLES; i++) {
        h_px[i]     = gt_x + gauss(gen) * init_spread_xy;
        h_py[i]     = gt_y + gauss(gen) * init_spread_xy;
        h_ptheta[i] = gt_theta + gauss(gen) * init_spread_th;
        h_pw[i]     = 1.0f / N_PARTICLES;
    }

    CUDA_CHECK(cudaMemcpy(d_px,     h_px.data(),     N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py,     h_py.data(),     N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ptheta, h_ptheta.data(), N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pw,     h_pw.data(),     N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));

    // ------------------------------------------
    // Robot trajectory
    // ------------------------------------------
    float robot_v = 1.0f;
    float robot_omega = 0.0f;

    // ------------------------------------------
    // Visualization setup
    // ------------------------------------------
    cv::namedWindow("emcl2", cv::WINDOW_NORMAL);
    cv::resizeWindow("emcl2", IMG_W, IMG_H);

    cv::VideoWriter video(
        "gif/emcl2.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
        cv::Size(IMG_W, IMG_H));

    // Pre-render map image
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

    // ------------------------------------------
    // Simulation loop
    // ------------------------------------------
    float time_val = 0.0f;
    int step_count = 0;
    bool kidnapped = false;
    bool recovered = false;
    float kidnap_actual_time = -1.0f;
    float recovery_time = -1.0f;
    int reset_count = 0;
    int consecutive_good = 0;

    int threads = 256;
    int pblocks = (N_PARTICLES + threads - 1) / threads;

    std::cout << "Starting simulation..." << std::endl;
    std::cout << "Robot will be kidnapped at t=" << KIDNAP_TIME << "s" << std::endl;

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
        float dx = target_x - gt_x;
        float dy = target_y - gt_y;
        float target_angle = atan2f(dy, dx);

        float angle_diff = target_angle - gt_theta;
        while (angle_diff > PI) angle_diff -= 2.0f * PI;
        while (angle_diff < -PI) angle_diff += 2.0f * PI;

        robot_v = 1.0f;
        robot_omega = angle_diff * 2.0f;
        if (robot_omega > 1.5f) robot_omega = 1.5f;
        if (robot_omega < -1.5f) robot_omega = -1.5f;

        // --- Kidnap the robot ---
        if (!kidnapped && time_val >= KIDNAP_TIME) {
            kidnapped = true;
            kidnap_actual_time = time_val;
            gt_x = KIDNAP_X;
            gt_y = KIDNAP_Y;
            gt_theta = KIDNAP_THETA;
            std::cout << "KIDNAPPED at t=" << time_val << "s! "
                      << "Teleported to (" << KIDNAP_X << ", " << KIDNAP_Y << ")" << std::endl;
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
                       origin_x, origin_y, GRID_RES,
                       GRID_W, GRID_H, h_beam_ranges);

        for (int b = 0; b < NUM_BEAMS; b++) {
            h_beam_ranges[b] += gauss(gen) * 0.05f;
            if (h_beam_ranges[b] < 0.0f) h_beam_ranges[b] = 0.0f;
        }

        CUDA_CHECK(cudaMemcpy(d_beam_ranges, h_beam_ranges,
                              NUM_BEAMS * sizeof(float), cudaMemcpyHostToDevice));

        // --- GPU: predict ---
        predict_kernel<<<pblocks, threads>>>(
            d_px, d_py, d_ptheta,
            robot_v, robot_omega, DT,
            d_rng_states, N_PARTICLES);

        // --- GPU: compute likelihood ---
        compute_likelihood_kernel<<<pblocks, threads>>>(
            d_px, d_py, d_ptheta, d_pw,
            d_likelihood_field, d_beam_ranges,
            GRID_W, GRID_H, GRID_RES,
            origin_x, origin_y, N_PARTICLES);

        // --- GPU: check for reset condition ---
        {
            int shared_size = threads * sizeof(int) + threads * sizeof(float);
            check_reset_kernel<<<1, threads, shared_size>>>(
                d_pw, N_PARTICLES, d_bad_count, d_sum);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_bad_count;
        float h_sum;
        CUDA_CHECK(cudaMemcpy(&h_bad_count, d_bad_count, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

        float bad_fraction = (float)h_bad_count / (float)N_PARTICLES;
        bool do_reset = (bad_fraction > RESET_THRESHOLD);

        // --- Normalize weights ---
        normalize_weights_kernel<<<pblocks, threads>>>(d_pw, N_PARTICLES, h_sum);
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Expansion + sensor resetting if triggered ---
        if (do_reset) {
            reset_count++;

            // Scale expansion noise by error severity
            float expansion_scale = EXPANSION_RATE * (1.0f + bad_fraction * 5.0f);

            // Number of particles for sensor reset
            int n_sensor = (int)(N_PARTICLES * SENSOR_RESET_RATIO);
            int n_expansion = N_PARTICLES - n_sensor;

            // Expansion reset: spread existing particles outward
            expansion_reset_kernel<<<pblocks, threads>>>(
                d_px, d_py, d_ptheta,
                d_rng_states, expansion_scale,
                n_expansion);

            // Sensor reset: generate new particles from high-likelihood positions
            if (n_sensor > 0) {
                int sr_threads = 256;
                int sr_blocks = (n_sensor + sr_threads - 1) / sr_threads;
                sensor_reset_kernel<<<sr_blocks, sr_threads>>>(
                    d_px, d_py, d_ptheta,
                    d_beam_ranges,
                    d_likelihood_field,
                    d_occupancy,
                    d_rng_states,
                    origin_x, origin_y,
                    GRID_RES, GRID_W, GRID_H,
                    n_expansion, n_sensor);
            }

            // Reset weights to uniform after reset
            std::vector<float> pw_uniform(N_PARTICLES, 1.0f / (float)N_PARTICLES);
            CUDA_CHECK(cudaMemcpy(d_pw, pw_uniform.data(), N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaDeviceSynchronize());
        } else {
            // --- Standard systematic resampling ---
            cumsum_kernel<<<1, 1>>>(d_pw, d_wcum, N_PARTICLES);
            CUDA_CHECK(cudaDeviceSynchronize());

            float rand_offset = uni(gen) / (float)N_PARTICLES;
            float base_step = 1.0f / (float)N_PARTICLES;

            resample_kernel<<<pblocks, threads>>>(
                d_px, d_py, d_ptheta,
                d_px_tmp, d_py_tmp, d_ptheta_tmp,
                d_wcum, base_step, rand_offset,
                N_PARTICLES, N_PARTICLES);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(d_px,     d_px_tmp,     N_PARTICLES * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_py,     d_py_tmp,     N_PARTICLES * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_ptheta, d_ptheta_tmp, N_PARTICLES * sizeof(float), cudaMemcpyDeviceToDevice));

            std::vector<float> pw_uniform(N_PARTICLES, 1.0f / (float)N_PARTICLES);
            CUDA_CHECK(cudaMemcpy(d_pw, pw_uniform.data(), N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
        }

        // --- GPU: weighted mean estimate ---
        // Recompute likelihood after reset for proper estimation
        compute_likelihood_kernel<<<pblocks, threads>>>(
            d_px, d_py, d_ptheta, d_pw,
            d_likelihood_field, d_beam_ranges,
            GRID_W, GRID_H, GRID_RES,
            origin_x, origin_y, N_PARTICLES);

        {
            int shared_size = threads * sizeof(int) + threads * sizeof(float);
            check_reset_kernel<<<1, threads, shared_size>>>(
                d_pw, N_PARTICLES, d_bad_count, d_sum);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
        normalize_weights_kernel<<<pblocks, threads>>>(d_pw, N_PARTICLES, h_sum);

        weighted_mean_kernel<<<1, threads, 4 * threads * sizeof(float)>>>(
            d_px, d_py, d_ptheta, d_pw,
            d_est_x, d_est_y, d_est_theta,
            N_PARTICLES);
        CUDA_CHECK(cudaDeviceSynchronize());

        float est_x, est_y, est_theta;
        CUDA_CHECK(cudaMemcpy(&est_x,     d_est_x,     sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&est_y,     d_est_y,     sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&est_theta, d_est_theta, sizeof(float), cudaMemcpyDeviceToHost));

        // --- Check recovery ---
        float err_x = gt_x - est_x;
        float err_y = gt_y - est_y;
        float err = sqrtf(err_x * err_x + err_y * err_y);

        if (kidnapped && !recovered) {
            if (err < RECOVERY_DIST_THRESHOLD) {
                consecutive_good++;
                if (consecutive_good > 20) {
                    recovered = true;
                    recovery_time = time_val;
                    std::cout << "RECOVERED at t=" << time_val << "s! "
                              << "(took " << (time_val - kidnap_actual_time) << "s, "
                              << reset_count << " resets)" << std::endl;
                }
            } else {
                consecutive_good = 0;
            }
        }

        // --- Read back particles for visualization ---
        CUDA_CHECK(cudaMemcpy(h_px.data(),     d_px,     N_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_py.data(),     d_py,     N_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ptheta.data(), d_ptheta, N_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pw.data(),     d_pw,     N_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost));

        // --- Visualization ---
        cv::Mat frame = map_img.clone();

        // Draw particles (red dots)
        float max_w = *std::max_element(h_pw.begin(), h_pw.begin() + N_PARTICLES);
        for (int i = 0; i < N_PARTICLES; i++) {
            cv::Point2i pt = world_to_pixel(h_px[i], h_py[i], origin_x, origin_y, GRID_RES);
            float w_norm = (max_w > 1e-10f) ? h_pw[i] / max_w : 0.0f;
            int radius = 1 + (int)(w_norm * 3.0f);
            cv::circle(frame, pt, radius, cv::Scalar(0, 0, 255), -1);
        }

        // Draw lidar beams
        cv::Point2i gt_pt = world_to_pixel(gt_x, gt_y, origin_x, origin_y, GRID_RES);
        for (int b = 0; b < NUM_BEAMS; b++) {
            float angle = gt_theta + (float)b * BEAM_ANGLE_STEP - PI;
            float ex = gt_x + h_beam_ranges[b] * cosf(angle);
            float ey = gt_y + h_beam_ranges[b] * sinf(angle);
            cv::Point2i ep = world_to_pixel(ex, ey, origin_x, origin_y, GRID_RES);
            cv::line(frame, gt_pt, ep, cv::Scalar(0, 200, 200), 1, cv::LINE_AA);
        }

        // Draw ground truth (green arrow)
        draw_arrow(frame, gt_pt, gt_theta, cv::Scalar(0, 200, 0), 20, 2);
        cv::circle(frame, gt_pt, 4, cv::Scalar(0, 200, 0), -1);

        // Draw estimated pose (blue arrow)
        cv::Point2i est_pt = world_to_pixel(est_x, est_y, origin_x, origin_y, GRID_RES);
        draw_arrow(frame, est_pt, est_theta, cv::Scalar(255, 0, 0), 20, 2);
        cv::circle(frame, est_pt, 4, cv::Scalar(255, 0, 0), -1);

        // Text overlay
        char text_buf[128];
        snprintf(text_buf, sizeof(text_buf), "emcl2 - Particles: %d", N_PARTICLES);
        cv::putText(frame, text_buf, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

        snprintf(text_buf, sizeof(text_buf), "t=%.1fs  error=%.2fm", time_val, err);
        cv::putText(frame, text_buf, cv::Point(10, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

        if (do_reset) {
            snprintf(text_buf, sizeof(text_buf), "RESET #%d (bad=%.0f%%)",
                     reset_count, bad_fraction * 100.0f);
            cv::putText(frame, text_buf, cv::Point(10, 75),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 200), 2);
        }

        // Status text
        if (!kidnapped) {
            cv::putText(frame, "Normal tracking", cv::Point(10, IMG_H - 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 150, 0), 2);
        } else if (!recovered) {
            cv::putText(frame, "KIDNAPPED!", cv::Point(IMG_W / 2 - 100, IMG_H / 2 - 20),
                        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
            cv::putText(frame, "RECOVERING...", cv::Point(10, IMG_H - 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 140, 255), 2);
        } else {
            cv::putText(frame, "RECOVERED!", cv::Point(IMG_W / 2 - 100, IMG_H / 2 - 20),
                        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 200, 0), 3);
            snprintf(text_buf, sizeof(text_buf), "Recovery: %.1fs (%d resets)",
                     recovery_time - kidnap_actual_time, reset_count);
            cv::putText(frame, text_buf, cv::Point(10, IMG_H - 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 0), 2);
        }

        cv::imshow("emcl2", frame);
        video.write(frame);
        cv::waitKey(5);

        if (step_count % 50 == 0) {
            printf("t=%.1f  bad=%.0f%%  resets=%d  error=%.3fm%s\n",
                   time_val, bad_fraction * 100.0f, reset_count, err,
                   do_reset ? " [RESET]" : "");
        }
    }

    video.release();
    cv::destroyAllWindows();

    std::cout << "\nSimulation complete." << std::endl;
    std::cout << "Total resets: " << reset_count << std::endl;
    if (recovered) {
        std::cout << "Recovery time: " << (recovery_time - kidnap_actual_time) << "s" << std::endl;
    } else {
        std::cout << "Did not fully recover within simulation time." << std::endl;
    }
    std::cout << "Video saved to gif/emcl2.avi" << std::endl;

    // Convert to gif
    int ret = system("which ffmpeg > /dev/null 2>&1 && "
        "ffmpeg -y -i gif/emcl2.avi "
        "-vf \"fps=15,scale=400:-1:flags=lanczos\" "
        "-gifflags +transdiff "
        "gif/emcl2.gif 2>/dev/null && "
        "echo 'GIF saved to gif/emcl2.gif' || echo 'ffmpeg not available, skipping gif'");
    (void)ret;

    // --- Cleanup ---
    cudaFree(d_occupancy);
    cudaFree(d_likelihood_field);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_ptheta);
    cudaFree(d_pw);
    cudaFree(d_px_tmp);
    cudaFree(d_py_tmp);
    cudaFree(d_ptheta_tmp);
    cudaFree(d_wcum);
    cudaFree(d_rng_states);
    cudaFree(d_bad_count);
    cudaFree(d_sum);
    cudaFree(d_est_x);
    cudaFree(d_est_y);
    cudaFree(d_est_theta);
    cudaFree(d_beam_ranges);

    return 0;
}
