/*************************************************************************
    > File Name: amcl.cu
    > CUDA-parallelized Adaptive Monte Carlo Localization (AMCL)
    > Features: KLD-sampling, likelihood field model, augmented MCL
    > CUDA kernels: likelihood field build, predict, update weights,
    >               Neff computation, systematic resampling
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
#define SIM_TIME 60.0f

// Grid
#define GRID_W 200
#define GRID_H 200
#define GRID_RES 0.1f  // meters per cell

// Particles
#define INIT_NP 500
#define MIN_NP 100
#define MAX_NP 2000

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

// KLD-sampling
#define KLD_EPS 0.05f
#define KLD_Z 2.33f

// Augmented MCL
#define ALPHA_SLOW 0.001f
#define ALPHA_FAST 0.1f

// Visualization
#define VIS_SCALE 4  // pixels per grid cell
#define IMG_W (GRID_W * VIS_SCALE)
#define IMG_H (GRID_H * VIS_SCALE)

// KLD bin resolution (in grid cells)
#define KLD_BIN_SIZE 2

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
// Each thread handles one grid cell. Computes distance to nearest obstacle
// cell and converts to Gaussian likelihood.
// ---------------------------------------------------------------------------
__global__ void build_likelihood_field_kernel(
    const int* occupancy,     // [GRID_H * GRID_W], 1=obstacle, 0=free
    float* likelihood_field,  // [GRID_H * GRID_W], output likelihood values
    int width, int height, float sigma_hit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int cy = idx / width;
    int cx = idx % width;

    // Find minimum distance to any obstacle cell (brute force within radius)
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

    // Gaussian likelihood based on distance
    float prob = Z_HIT * expf(-0.5f * (min_dist * min_dist) / (sigma_hit * sigma_hit))
                 + Z_RAND / MAX_RANGE;
    likelihood_field[idx] = prob;
}

// ---------------------------------------------------------------------------
// Kernel: predict particles (velocity motion model with noise)
// State: [x, y, theta] per particle
// ---------------------------------------------------------------------------
__global__ void predict_particles_kernel(
    float* px, float* py, float* ptheta,
    float v, float omega, float dt,
    curandState* rng_states,
    int np)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;

    curandState local_rng = rng_states[idx];

    // Sample noisy velocity
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

    // Wrap angle
    float th = ptheta[idx];
    while (th > PI) th -= 2.0f * PI;
    while (th < -PI) th += 2.0f * PI;
    ptheta[idx] = th;

    rng_states[idx] = local_rng;
}

// ---------------------------------------------------------------------------
// Kernel: update weights using likelihood field
// For each particle, cast NUM_BEAMS rays, find endpoints, lookup likelihood
// ---------------------------------------------------------------------------
__global__ void update_weights_kernel(
    float* px, float* py, float* ptheta,
    float* pw,
    const float* likelihood_field,
    const float* beam_ranges,   // [NUM_BEAMS] from simulated lidar
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
        if (range >= MAX_RANGE) continue;  // skip max-range readings

        float beam_angle = theta + (float)b * BEAM_ANGLE_STEP - PI;
        float ex = x + range * cosf(beam_angle);
        float ey = y + range * sinf(beam_angle);

        // Convert to grid coordinates
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

    pw[idx] *= expf(log_w);
}

// ---------------------------------------------------------------------------
// Kernel: compute sum of weights and sum of squared weights (for Neff)
//         Single-block reduction
// ---------------------------------------------------------------------------
__global__ void compute_neff_kernel(
    const float* pw, int np,
    float* out_sum, float* out_sum_sq)
{
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sq  = sdata + blockDim.x;

    int tid = threadIdx.x;
    float val = 0.0f, val_sq = 0.0f;

    for (int i = tid; i < np; i += blockDim.x) {
        float w = pw[i];
        val += w;
        val_sq += w * w;
    }
    s_sum[tid] = val;
    s_sq[tid]  = val_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq[tid]  += s_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_sum    = s_sum[0];
        *out_sum_sq = s_sq[0];
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
// Kernel: cumulative sum (sequential, single thread)
// ---------------------------------------------------------------------------
__global__ void cumsum_kernel(const float* pw, float* wcum, int np) {
    wcum[0] = pw[0];
    for (int i = 1; i < np; i++) {
        wcum[i] = wcum[i - 1] + pw[i];
    }
}

// ---------------------------------------------------------------------------
// Kernel: systematic resampling
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

    // binary search
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
// Host: build occupancy map
// Creates walls around boundary and some internal obstacles
// ---------------------------------------------------------------------------
void build_map(std::vector<int>& occupancy, int w, int h) {
    occupancy.assign(w * h, 0);

    // Boundary walls
    for (int x = 0; x < w; x++) {
        occupancy[0 * w + x] = 1;
        occupancy[1 * w + x] = 1;
        occupancy[(h - 1) * w + x] = 1;
        occupancy[(h - 2) * w + x] = 1;
    }
    for (int y = 0; y < h; y++) {
        occupancy[y * w + 0] = 1;
        occupancy[y * w + 1] = 1;
        occupancy[y * w + (w - 1)] = 1;
        occupancy[y * w + (w - 2)] = 1;
    }

    // Internal walls / obstacles
    // Horizontal wall segment
    for (int x = 40; x < 80; x++) {
        for (int t = 0; t < 3; t++) {
            occupancy[(60 + t) * w + x] = 1;
        }
    }
    // Vertical wall segment
    for (int y = 80; y < 140; y++) {
        for (int t = 0; t < 3; t++) {
            occupancy[y * w + (130 + t)] = 1;
        }
    }
    // L-shaped obstacle
    for (int x = 30; x < 55; x++) {
        for (int t = 0; t < 3; t++) {
            occupancy[(130 + t) * w + x] = 1;
        }
    }
    for (int y = 130; y < 165; y++) {
        for (int t = 0; t < 3; t++) {
            occupancy[y * w + (30 + t)] = 1;
        }
    }
    // Box obstacle
    for (int x = 90; x < 110; x++) {
        occupancy[140 * w + x] = 1;
        occupancy[141 * w + x] = 1;
        occupancy[159 * w + x] = 1;
        occupancy[160 * w + x] = 1;
    }
    for (int y = 140; y < 161; y++) {
        occupancy[y * w + 90] = 1;
        occupancy[y * w + 91] = 1;
        occupancy[y * w + 109] = 1;
        occupancy[y * w + 110] = 1;
    }
}

// ---------------------------------------------------------------------------
// Host: simulate lidar from ground truth position on the occupancy grid
// ---------------------------------------------------------------------------
void simulate_lidar(
    const std::vector<int>& occupancy,
    float robot_x, float robot_y, float robot_theta,
    float origin_x, float origin_y, float resolution,
    int grid_w, int grid_h,
    float* beam_ranges)
{
    float step = resolution * 0.5f;  // ray-march step

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
// Host: KLD-sampling -- compute required number of particles
// ---------------------------------------------------------------------------
int kld_sample_count(
    const float* h_px, const float* h_py, int np,
    float origin_x, float origin_y, float resolution,
    int grid_w, int grid_h)
{
    int bin_w = (grid_w + KLD_BIN_SIZE - 1) / KLD_BIN_SIZE;
    int bin_h = (grid_h + KLD_BIN_SIZE - 1) / KLD_BIN_SIZE;
    std::vector<bool> bins(bin_w * bin_h, false);
    int k = 0;

    for (int i = 0; i < np; i++) {
        int gx = (int)((h_px[i] - origin_x) / resolution);
        int gy = (int)((h_py[i] - origin_y) / resolution);
        int bx = gx / KLD_BIN_SIZE;
        int by = gy / KLD_BIN_SIZE;
        if (bx >= 0 && bx < bin_w && by >= 0 && by < bin_h) {
            int bidx = by * bin_w + bx;
            if (!bins[bidx]) {
                bins[bidx] = true;
                k++;
            }
        }
    }

    if (k <= 1) return MAX_NP;

    // Wilson-Hilferty approximation
    float kf = (float)(k - 1);
    float term = 1.0f - 2.0f / (9.0f * kf) + sqrtf(2.0f / (9.0f * kf)) * KLD_Z;
    float n_kld = kf / (2.0f * KLD_EPS) * term * term * term;

    int n = (int)ceilf(n_kld);
    if (n < MIN_NP) n = MIN_NP;
    if (n > MAX_NP) n = MAX_NP;
    return n;
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
    // Flip y for screen coordinates
    int dx = (int)(length * cosf(theta));
    int dy = (int)(-length * sinf(theta));
    cv::Point2i tip(pt.x + dx, pt.y + dy);
    cv::arrowedLine(img, pt, tip, color, thickness, cv::LINE_AA, 0, 0.3);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "CUDA AMCL (Adaptive Monte Carlo Localization)" << std::endl;

    // Grid origin: bottom-left corner in world coordinates
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

    // Read back likelihood field for visualization
    std::vector<float> h_likelihood_field(GRID_W * GRID_H);
    CUDA_CHECK(cudaMemcpy(h_likelihood_field.data(), d_likelihood_field,
                          GRID_W * GRID_H * sizeof(float), cudaMemcpyDeviceToHost));

    // ------------------------------------------
    // Particle memory (allocate for MAX_NP)
    // ------------------------------------------
    float *d_px, *d_py, *d_ptheta, *d_pw;
    float *d_px_tmp, *d_py_tmp, *d_ptheta_tmp;
    CUDA_CHECK(cudaMalloc(&d_px,         MAX_NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py,         MAX_NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ptheta,     MAX_NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pw,         MAX_NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_px_tmp,     MAX_NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py_tmp,     MAX_NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ptheta_tmp, MAX_NP * sizeof(float)));

    float *d_wcum;
    CUDA_CHECK(cudaMalloc(&d_wcum, MAX_NP * sizeof(float)));

    // cuRAND states
    curandState* d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, MAX_NP * sizeof(curandState)));
    {
        int threads = 256;
        int blocks = (MAX_NP + threads - 1) / threads;
        init_curand_kernel<<<blocks, threads>>>(d_rng_states, 42ULL, MAX_NP);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Neff reduction outputs
    float *d_sum, *d_sum_sq;
    CUDA_CHECK(cudaMalloc(&d_sum,    sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_sq, sizeof(float)));

    // Weighted mean outputs
    float *d_est_x, *d_est_y, *d_est_theta;
    CUDA_CHECK(cudaMalloc(&d_est_x,     sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_est_y,     sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_est_theta, sizeof(float)));

    // Beam ranges on device
    float *d_beam_ranges;
    CUDA_CHECK(cudaMalloc(&d_beam_ranges, NUM_BEAMS * sizeof(float)));

    // Host buffers
    std::vector<float> h_px(MAX_NP), h_py(MAX_NP), h_ptheta(MAX_NP), h_pw(MAX_NP);
    float h_beam_ranges[NUM_BEAMS];

    // ------------------------------------------
    // Initialize particles: spread around the map center with uncertainty
    // ------------------------------------------
    int current_np = INIT_NP;

    // Ground truth start
    float gt_x = 5.0f;
    float gt_y = 5.0f;
    float gt_theta = 0.0f;

    std::mt19937 gen(12345);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    // Spread particles with large initial uncertainty
    float init_spread_xy = 3.0f;
    float init_spread_th = PI;
    for (int i = 0; i < current_np; i++) {
        h_px[i]     = gt_x + gauss(gen) * init_spread_xy;
        h_py[i]     = gt_y + gauss(gen) * init_spread_xy;
        h_ptheta[i] = gt_theta + gauss(gen) * init_spread_th;
        h_pw[i]     = 1.0f / current_np;
    }

    CUDA_CHECK(cudaMemcpy(d_px,     h_px.data(),     current_np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py,     h_py.data(),     current_np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ptheta, h_ptheta.data(), current_np * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pw,     h_pw.data(),     current_np * sizeof(float), cudaMemcpyHostToDevice));

    // ------------------------------------------
    // Augmented MCL state
    // ------------------------------------------
    float w_slow = 0.0f;
    float w_fast = 0.0f;

    // ------------------------------------------
    // Robot trajectory: drive a path through the map
    // ------------------------------------------
    float robot_v = 1.0f;       // m/s
    float robot_omega = 0.0f;   // rad/s

    // ------------------------------------------
    // Visualization setup
    // ------------------------------------------
    cv::namedWindow("amcl", cv::WINDOW_NORMAL);
    cv::resizeWindow("amcl", IMG_W, IMG_H);

    cv::VideoWriter video(
        "gif/amcl.avi",
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
    int step = 0;

    std::cout << "Starting simulation..." << std::endl;

    while (time_val <= SIM_TIME) {
        time_val += DT;
        step++;

        // --- Robot control: navigate through the map ---
        // Change direction periodically to create interesting trajectory
        float map_center_x = GRID_W * GRID_RES * 0.5f;
        float map_center_y = GRID_H * GRID_RES * 0.5f;

        // Simple waypoint navigation
        float targets[][2] = {
            {15.0f, 5.0f}, {15.0f, 15.0f}, {5.0f, 15.0f},
            {5.0f, 8.0f}, {12.0f, 8.0f}, {12.0f, 17.0f},
            {3.0f, 17.0f}, {3.0f, 5.0f}
        };
        int n_targets = 8;
        int wp_idx = ((int)(time_val / 7.5f)) % n_targets;

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

        // Clamp to map bounds
        float map_min = 0.5f;
        float map_max_x = GRID_W * GRID_RES - 0.5f;
        float map_max_y = GRID_H * GRID_RES - 0.5f;
        gt_x = std::max(map_min, std::min(map_max_x, gt_x));
        gt_y = std::max(map_min, std::min(map_max_y, gt_y));

        // --- Simulate lidar ---
        simulate_lidar(h_occupancy, gt_x, gt_y, gt_theta,
                       origin_x, origin_y, GRID_RES,
                       GRID_W, GRID_H, h_beam_ranges);

        // Add noise to lidar
        for (int b = 0; b < NUM_BEAMS; b++) {
            h_beam_ranges[b] += gauss(gen) * 0.05f;
            if (h_beam_ranges[b] < 0.0f) h_beam_ranges[b] = 0.0f;
        }

        CUDA_CHECK(cudaMemcpy(d_beam_ranges, h_beam_ranges,
                              NUM_BEAMS * sizeof(float), cudaMemcpyHostToDevice));

        // --- GPU: predict ---
        {
            int threads = 256;
            int blocks = (current_np + threads - 1) / threads;
            predict_particles_kernel<<<blocks, threads>>>(
                d_px, d_py, d_ptheta,
                robot_v, robot_omega, DT,
                d_rng_states, current_np);
        }

        // --- GPU: update weights ---
        {
            int threads = 256;
            int blocks = (current_np + threads - 1) / threads;
            update_weights_kernel<<<blocks, threads>>>(
                d_px, d_py, d_ptheta, d_pw,
                d_likelihood_field, d_beam_ranges,
                GRID_W, GRID_H, GRID_RES,
                origin_x, origin_y, current_np);
        }

        // --- GPU: compute Neff ---
        {
            int threads = 256;
            compute_neff_kernel<<<1, threads, 2 * threads * sizeof(float)>>>(
                d_pw, current_np, d_sum, d_sum_sq);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        float h_sum, h_sum_sq;
        CUDA_CHECK(cudaMemcpy(&h_sum,    d_sum,    sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(float), cudaMemcpyDeviceToHost));

        // Normalize weights on GPU
        {
            int threads = 256;
            int blocks = (current_np + threads - 1) / threads;
            normalize_weights_kernel<<<blocks, threads>>>(d_pw, current_np, h_sum);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute Neff from normalized weights
        float neff;
        if (h_sum > 1e-30f) {
            float norm_sum_sq = h_sum_sq / (h_sum * h_sum);
            neff = 1.0f / norm_sum_sq;
        } else {
            neff = 0.0f;
        }

        // --- GPU: weighted mean estimate ---
        {
            int threads = 256;
            weighted_mean_kernel<<<1, threads, 4 * threads * sizeof(float)>>>(
                d_px, d_py, d_ptheta, d_pw,
                d_est_x, d_est_y, d_est_theta,
                current_np);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        float est_x, est_y, est_theta;
        CUDA_CHECK(cudaMemcpy(&est_x,     d_est_x,     sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&est_y,     d_est_y,     sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&est_theta,  d_est_theta, sizeof(float), cudaMemcpyDeviceToHost));

        // --- Read back particles for KLD and visualization ---
        CUDA_CHECK(cudaMemcpy(h_px.data(),     d_px,     current_np * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_py.data(),     d_py,     current_np * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ptheta.data(), d_ptheta, current_np * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pw.data(),     d_pw,     current_np * sizeof(float), cudaMemcpyDeviceToHost));

        // --- Augmented MCL: track average weight for random injection ---
        float w_avg = h_sum / (float)current_np;
        w_slow += ALPHA_SLOW * (w_avg - w_slow);
        w_fast += ALPHA_FAST * (w_avg - w_fast);

        // --- Resampling with KLD and augmented MCL ---
        float neff_threshold = current_np * 0.5f;
        if (neff < neff_threshold) {
            // KLD: determine new particle count
            int new_np = kld_sample_count(h_px.data(), h_py.data(), current_np,
                                          origin_x, origin_y, GRID_RES,
                                          GRID_W, GRID_H);

            // Systematic resampling on GPU
            cumsum_kernel<<<1, 1>>>(d_pw, d_wcum, current_np);
            CUDA_CHECK(cudaDeviceSynchronize());

            float rand_offset = uni(gen) / (float)new_np;
            float base_step = 1.0f / (float)new_np;

            {
                int threads = 256;
                int blocks = (new_np + threads - 1) / threads;
                resample_kernel<<<blocks, threads>>>(
                    d_px, d_py, d_ptheta,
                    d_px_tmp, d_py_tmp, d_ptheta_tmp,
                    d_wcum, base_step, rand_offset,
                    current_np, new_np);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // Augmented MCL: inject random particles to recover from kidnapping
            float prob_random = std::max(0.0f, 1.0f - w_fast / (w_slow + 1e-10f));
            int n_random = (int)(prob_random * new_np * 0.1f);
            if (n_random > 0) {
                // Read back resampled particles, inject random ones, write back
                std::vector<float> tmp_px(new_np), tmp_py(new_np), tmp_ptheta(new_np);
                CUDA_CHECK(cudaMemcpy(tmp_px.data(),     d_px_tmp,     new_np * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(tmp_py.data(),     d_py_tmp,     new_np * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(tmp_ptheta.data(), d_ptheta_tmp, new_np * sizeof(float), cudaMemcpyDeviceToHost));

                std::uniform_real_distribution<float> uni_x(map_min, map_max_x);
                std::uniform_real_distribution<float> uni_y(map_min, map_max_y);
                std::uniform_real_distribution<float> uni_th(-PI, PI);

                for (int i = 0; i < n_random && i < new_np; i++) {
                    // Place random particle in free space
                    for (int attempt = 0; attempt < 20; attempt++) {
                        float rx = uni_x(gen);
                        float ry = uni_y(gen);
                        int gx = (int)((rx - origin_x) / GRID_RES);
                        int gy = (int)((ry - origin_y) / GRID_RES);
                        if (gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H &&
                            h_occupancy[gy * GRID_W + gx] == 0) {
                            tmp_px[i] = rx;
                            tmp_py[i] = ry;
                            tmp_ptheta[i] = uni_th(gen);
                            break;
                        }
                    }
                }

                CUDA_CHECK(cudaMemcpy(d_px_tmp,     tmp_px.data(),     new_np * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_py_tmp,     tmp_py.data(),     new_np * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ptheta_tmp, tmp_ptheta.data(), new_np * sizeof(float), cudaMemcpyHostToDevice));
            }

            // Copy resampled particles back
            CUDA_CHECK(cudaMemcpy(d_px,     d_px_tmp,     new_np * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_py,     d_py_tmp,     new_np * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_ptheta, d_ptheta_tmp, new_np * sizeof(float), cudaMemcpyDeviceToDevice));

            // Reset weights
            std::vector<float> pw_uniform(new_np, 1.0f / (float)new_np);
            CUDA_CHECK(cudaMemcpy(d_pw, pw_uniform.data(), new_np * sizeof(float), cudaMemcpyHostToDevice));

            current_np = new_np;

            // Re-read for visualization
            CUDA_CHECK(cudaMemcpy(h_px.data(),     d_px,     current_np * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_py.data(),     d_py,     current_np * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ptheta.data(), d_ptheta, current_np * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_pw.data(),     d_pw,     current_np * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // --- Visualization ---
        cv::Mat frame = map_img.clone();

        // Draw particles (red dots, size proportional to weight)
        float max_w = *std::max_element(h_pw.begin(), h_pw.begin() + current_np);
        for (int i = 0; i < current_np; i++) {
            cv::Point2i pt = world_to_pixel(h_px[i], h_py[i], origin_x, origin_y, GRID_RES);
            float w_norm = (max_w > 1e-10f) ? h_pw[i] / max_w : 0.0f;
            int radius = 1 + (int)(w_norm * 3.0f);
            cv::circle(frame, pt, radius, cv::Scalar(0, 0, 255), -1);
        }

        // Draw lidar beams (thin yellow lines)
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
        snprintf(text_buf, sizeof(text_buf), "Particles: %d", current_np);
        cv::putText(frame, text_buf, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

        snprintf(text_buf, sizeof(text_buf), "Neff: %.0f", neff);
        cv::putText(frame, text_buf, cv::Point(10, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

        snprintf(text_buf, sizeof(text_buf), "t=%.1fs", time_val);
        cv::putText(frame, text_buf, cv::Point(10, 75),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

        cv::imshow("amcl", frame);
        video.write(frame);
        cv::waitKey(5);

        if (step % 50 == 0) {
            float err_x = gt_x - est_x;
            float err_y = gt_y - est_y;
            float err = sqrtf(err_x * err_x + err_y * err_y);
            printf("t=%.1f  particles=%d  neff=%.0f  error=%.3fm\n",
                   time_val, current_np, neff, err);
        }
    }

    video.release();
    cv::destroyAllWindows();

    std::cout << "Video saved to gif/amcl.avi" << std::endl;

    // Convert to gif
    int ret = system("which ffmpeg > /dev/null 2>&1 && "
        "ffmpeg -y -i gif/amcl.avi "
        "-vf \"fps=15,scale=400:-1:flags=lanczos\" "
        "-gifflags +transdiff "
        "gif/amcl.gif 2>/dev/null && "
        "echo 'GIF saved to gif/amcl.gif' || echo 'ffmpeg not available, skipping gif'");
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
    cudaFree(d_sum);
    cudaFree(d_sum_sq);
    cudaFree(d_est_x);
    cudaFree(d_est_y);
    cudaFree(d_est_theta);
    cudaFree(d_beam_ranges);

    return 0;
}
