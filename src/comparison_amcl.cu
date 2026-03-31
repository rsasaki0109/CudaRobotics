/*************************************************************************
    AMCL: CPU vs CUDA side-by-side comparison GIF generator
    Left panel: CPU (sequential predict + likelihood field weight update)
    Right panel: CUDA (parallel GPU kernels)
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <chrono>
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
#define SIM_TIME 15.0f

// Grid
#define GRID_W 200
#define GRID_H 200
#define GRID_RES 0.1f

// Particles (fixed)
#define NP 500

// Likelihood field
#define SIGMA_HIT 0.2f
#define Z_HIT 0.9f
#define Z_RAND 0.1f

// Motion model noise
#define ALPHA1 0.1f
#define ALPHA2 0.1f
#define ALPHA3 0.1f
#define ALPHA4 0.1f

// Lidar
#define NUM_BEAMS 36
#define BEAM_ANGLE_STEP (2.0f * PI / NUM_BEAMS)
#define MAX_RANGE 10.0f

// Visualization
#define VIS_SCALE 4
#define IMG_W (GRID_W * VIS_SCALE)
#define IMG_H (GRID_H * VIS_SCALE)
#define PANEL_W IMG_W
#define PANEL_H IMG_H
#define COMBINED_W PANEL_W
#define COMBINED_H (PANEL_H * 2)

// ===========================================================================
// Host: build occupancy map (same as amcl.cu)
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

// ===========================================================================
// Host: build likelihood field on CPU
// ===========================================================================
void build_likelihood_field_cpu(const std::vector<int>& occupancy,
                                 std::vector<float>& lf, int w, int h) {
    lf.resize(w * h);
    for (int idx = 0; idx < w * h; idx++) {
        int cy = idx / w, cx = idx % w;
        float min_dist = 1e6f;
        int search_radius = (int)(3.0f * SIGMA_HIT / GRID_RES) + 1;
        if (search_radius > 50) search_radius = 50;
        for (int dy = -search_radius; dy <= search_radius; dy++) {
            for (int dx = -search_radius; dx <= search_radius; dx++) {
                int nx = cx + dx, ny = cy + dy;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    if (occupancy[ny * w + nx] == 1) {
                        float dist = sqrtf((float)(dx * dx + dy * dy)) * GRID_RES;
                        if (dist < min_dist) min_dist = dist;
                    }
                }
            }
        }
        lf[idx] = Z_HIT * expf(-0.5f * (min_dist * min_dist) / (SIGMA_HIT * SIGMA_HIT))
                   + Z_RAND / MAX_RANGE;
    }
}

// ===========================================================================
// Host: simulate lidar
// ===========================================================================
void simulate_lidar(const std::vector<int>& occupancy,
                    float rx, float ry, float rtheta,
                    float ox, float oy, float res,
                    int gw, int gh, float* beam_ranges) {
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

// ===========================================================================
// Visualization helpers
// ===========================================================================
cv::Point2i grid_to_pixel(int gx, int gy) {
    return cv::Point2i(gx * VIS_SCALE, (GRID_H - 1 - gy) * VIS_SCALE);
}

cv::Point2i world_to_pixel(float wx, float wy, float ox, float oy, float res) {
    int gx = (int)((wx - ox) / res);
    int gy = (int)((wy - oy) / res);
    return grid_to_pixel(gx, gy);
}

void draw_arrow(cv::Mat& img, cv::Point2i pt, float theta,
                cv::Scalar color, int length, int thickness) {
    int dx = (int)(length * cosf(theta));
    int dy = (int)(-length * sinf(theta));
    cv::arrowedLine(img, pt, cv::Point2i(pt.x + dx, pt.y + dy),
                    color, thickness, cv::LINE_AA, 0, 0.3);
}

// ===========================================================================
// CPU Implementation
// ===========================================================================

struct CPUParticle { float x, y, theta, weight; };

static void cpu_predict(CPUParticle* p, int np, float v, float omega,
                        std::mt19937& gen) {
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (int i = 0; i < np; i++) {
        float v_hat = v + gauss(gen) * (ALPHA1 * fabsf(v) + ALPHA2 * fabsf(omega));
        float omega_hat = omega + gauss(gen) * (ALPHA3 * fabsf(v) + ALPHA4 * fabsf(omega));
        float gamma_hat = gauss(gen) * (ALPHA1 * fabsf(v) + ALPHA2 * fabsf(omega)) * 0.1f;
        float th = p[i].theta;
        if (fabsf(omega_hat) < 1e-6f) {
            p[i].x += v_hat * cosf(th) * DT;
            p[i].y += v_hat * sinf(th) * DT;
        } else {
            float r = v_hat / omega_hat;
            p[i].x += r * (sinf(th + omega_hat * DT) - sinf(th));
            p[i].y += r * (cosf(th) - cosf(th + omega_hat * DT));
        }
        p[i].theta += omega_hat * DT + gamma_hat * DT;
        while (p[i].theta > PI) p[i].theta -= 2.0f * PI;
        while (p[i].theta < -PI) p[i].theta += 2.0f * PI;
    }
}

static void cpu_update_weights(CPUParticle* p, int np,
                                const std::vector<float>& lf,
                                const float* beams,
                                float ox, float oy, float res,
                                int gw, int gh) {
    for (int i = 0; i < np; i++) {
        float log_w = 0.0f;
        for (int b = 0; b < NUM_BEAMS; b++) {
            float range = beams[b];
            if (range >= MAX_RANGE) continue;
            float angle = p[i].theta + (float)b * BEAM_ANGLE_STEP - PI;
            float ex = p[i].x + range * cosf(angle);
            float ey = p[i].y + range * sinf(angle);
            int gx = (int)((ex - ox) / res);
            int gy = (int)((ey - oy) / res);
            if (gx >= 0 && gx < gw && gy >= 0 && gy < gh) {
                float val = lf[gy * gw + gx];
                log_w += (val > 1e-10f) ? logf(val) : logf(1e-10f);
            } else {
                log_w += logf(Z_RAND / MAX_RANGE);
            }
        }
        p[i].weight *= expf(log_w);
    }
}

static void cpu_normalize_and_neff(CPUParticle* p, int np, float& neff) {
    float total = 0.0f;
    for (int i = 0; i < np; i++) total += p[i].weight;
    if (total < 1e-30f) total = 1e-30f;
    float sum_sq = 0.0f;
    for (int i = 0; i < np; i++) {
        p[i].weight /= total;
        sum_sq += p[i].weight * p[i].weight;
    }
    neff = 1.0f / (sum_sq + 1e-30f);
}

static void cpu_weighted_mean(const CPUParticle* p, int np,
                               float& ex, float& ey, float& etheta) {
    float sx = 0, sy = 0, sc = 0, ss = 0;
    for (int i = 0; i < np; i++) {
        sx += p[i].x * p[i].weight;
        sy += p[i].y * p[i].weight;
        sc += cosf(p[i].theta) * p[i].weight;
        ss += sinf(p[i].theta) * p[i].weight;
    }
    ex = sx; ey = sy; etheta = atan2f(ss, sc);
}

static void cpu_resample(CPUParticle* p, int np, std::mt19937& gen) {
    std::vector<float> wcum(np);
    wcum[0] = p[0].weight;
    for (int i = 1; i < np; i++) wcum[i] = wcum[i-1] + p[i].weight;

    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    float r = uni(gen) / np;
    float step = 1.0f / np;

    std::vector<CPUParticle> newp(np);
    for (int i = 0; i < np; i++) {
        float target = step * i + r;
        int lo = 0, hi = np - 1;
        while (lo < hi) { int mid = (lo + hi) / 2; if (wcum[mid] < target) lo = mid + 1; else hi = mid; }
        newp[i] = p[lo];
        newp[i].weight = 1.0f / np;
    }
    for (int i = 0; i < np; i++) p[i] = newp[i];
}

// ===========================================================================
// CUDA Kernels
// ===========================================================================

__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void build_likelihood_field_kernel(
    const int* occupancy, float* likelihood_field,
    int width, int height, float sigma_hit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    int cy = idx / width, cx = idx % width;
    float min_dist = 1e6f;
    int sr = (int)(3.0f * sigma_hit / GRID_RES) + 1;
    if (sr > 50) sr = 50;
    for (int dy = -sr; dy <= sr; dy++) {
        for (int dx = -sr; dx <= sr; dx++) {
            int nx = cx + dx, ny = cy + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (occupancy[ny * width + nx] == 1) {
                    float dist = sqrtf((float)(dx * dx + dy * dy)) * GRID_RES;
                    if (dist < min_dist) min_dist = dist;
                }
            }
        }
    }
    likelihood_field[idx] = Z_HIT * expf(-0.5f * (min_dist * min_dist) / (sigma_hit * sigma_hit))
                            + Z_RAND / MAX_RANGE;
}

__global__ void predict_particles_kernel(
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

__global__ void update_weights_kernel(
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
        float angle = theta + (float)b * BEAM_ANGLE_STEP - PI;
        float ex = x + range * cosf(angle);
        float ey = y + range * sinf(angle);
        int gx = (int)((ex - origin_x) / resolution);
        int gy = (int)((ey - origin_y) / resolution);
        if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
            float lf = likelihood_field[gy * width + gx];
            log_w += (lf > 1e-10f) ? logf(lf) : logf(1e-10f);
        } else {
            log_w += logf(Z_RAND / MAX_RANGE);
        }
    }
    pw[idx] *= expf(log_w);
}

__global__ void compute_neff_kernel(const float* pw, int np,
                                     float* out_sum, float* out_sum_sq) {
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sq  = sdata + blockDim.x;
    int tid = threadIdx.x;
    float val = 0, val_sq = 0;
    for (int i = tid; i < np; i += blockDim.x) {
        float w = pw[i]; val += w; val_sq += w * w;
    }
    s_sum[tid] = val; s_sq[tid] = val_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s_sum[tid] += s_sum[tid + s]; s_sq[tid] += s_sq[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { *out_sum = s_sum[0]; *out_sum_sq = s_sq[0]; }
}

__global__ void normalize_weights_kernel(float* pw, int np, float total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;
    pw[idx] = (total > 1e-30f) ? pw[idx] / total : 1.0f / (float)np;
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
    while (lo < hi) { int mid = (lo + hi) / 2; if (wcum[mid] < target) lo = mid + 1; else hi = mid; }
    px_out[ip] = px_in[lo];
    py_out[ip] = py_in[lo];
    ptheta_out[ip] = ptheta_in[lo];
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
        vx += px[i] * w; vy += py[i] * w;
        vc += cosf(ptheta[i]) * w; vs += sinf(ptheta[i]) * w;
    }
    sx[tid] = vx; sy[tid] = vy; sc[tid] = vc; ss[tid] = vs;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sx[tid] += sx[tid+s]; sy[tid] += sy[tid+s]; sc[tid] += sc[tid+s]; ss[tid] += ss[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) { *out_x = sx[0]; *out_y = sy[0]; *out_theta = atan2f(ss[0], sc[0]); }
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    std::cout << "AMCL comparison: CPU vs CUDA (" << NP << " particles, "
              << NUM_BEAMS << " beams)" << std::endl;

    float origin_x = 0.0f, origin_y = 0.0f;

    // Build map
    std::vector<int> h_occupancy;
    build_map(h_occupancy, GRID_W, GRID_H);

    // Build likelihood field on CPU (for CPU path)
    std::vector<float> h_likelihood_field;
    build_likelihood_field_cpu(h_occupancy, h_likelihood_field, GRID_W, GRID_H);

    // Build likelihood field on GPU
    int* d_occupancy;
    CUDA_CHECK(cudaMalloc(&d_occupancy, GRID_W * GRID_H * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_occupancy, h_occupancy.data(), GRID_W * GRID_H * sizeof(int), cudaMemcpyHostToDevice));

    float* d_likelihood_field;
    CUDA_CHECK(cudaMalloc(&d_likelihood_field, GRID_W * GRID_H * sizeof(float)));
    {
        int total = GRID_W * GRID_H;
        int t = 256, b = (total + t - 1) / t;
        build_likelihood_field_kernel<<<b, t>>>(d_occupancy, d_likelihood_field, GRID_W, GRID_H, SIGMA_HIT);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Ground truth start
    float gt_x = 5.0f, gt_y = 5.0f, gt_theta = 0.0f;

    std::mt19937 gen(12345);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    float init_spread_xy = 3.0f, init_spread_th = PI;

    // ---- CPU particles ----
    std::vector<CPUParticle> cpu_p(NP);
    std::mt19937 gen_cpu(12345);
    for (int i = 0; i < NP; i++) {
        cpu_p[i].x = gt_x + gauss(gen_cpu) * init_spread_xy;
        cpu_p[i].y = gt_y + gauss(gen_cpu) * init_spread_xy;
        cpu_p[i].theta = gt_theta + gauss(gen_cpu) * init_spread_th;
        cpu_p[i].weight = 1.0f / NP;
    }
    double cpu_total_ms = 0.0;

    // ---- CUDA particles ----
    const int threads = 256;
    const int pblocks = (NP + threads - 1) / threads;

    float *d_px, *d_py, *d_ptheta, *d_pw;
    float *d_px_tmp, *d_py_tmp, *d_ptheta_tmp;
    CUDA_CHECK(cudaMalloc(&d_px, NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py, NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ptheta, NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pw, NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_px_tmp, NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py_tmp, NP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ptheta_tmp, NP * sizeof(float)));

    float *d_wcum;
    CUDA_CHECK(cudaMalloc(&d_wcum, NP * sizeof(float)));

    curandState* d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, NP * sizeof(curandState)));
    init_curand_kernel<<<pblocks, threads>>>(d_rng_states, 42ULL, NP);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *d_sum, *d_sum_sq;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_sq, sizeof(float)));

    float *d_est_x, *d_est_y, *d_est_theta;
    CUDA_CHECK(cudaMalloc(&d_est_x, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_est_y, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_est_theta, sizeof(float)));

    float *d_beam_ranges;
    CUDA_CHECK(cudaMalloc(&d_beam_ranges, NUM_BEAMS * sizeof(float)));

    // Init GPU particles (same distribution as CPU with gen_cpu seed)
    {
        std::mt19937 gen_init(12345);
        std::vector<float> hpx(NP), hpy(NP), hpth(NP), hpw(NP);
        for (int i = 0; i < NP; i++) {
            hpx[i] = gt_x + gauss(gen_init) * init_spread_xy;
            hpy[i] = gt_y + gauss(gen_init) * init_spread_xy;
            hpth[i] = gt_theta + gauss(gen_init) * init_spread_th;
            hpw[i] = 1.0f / NP;
        }
        CUDA_CHECK(cudaMemcpy(d_px, hpx.data(), NP * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_py, hpy.data(), NP * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ptheta, hpth.data(), NP * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pw, hpw.data(), NP * sizeof(float), cudaMemcpyHostToDevice));
    }
    double cuda_total_ms = 0.0;

    // Host readback buffers
    std::vector<float> h_px(NP), h_py(NP), h_ptheta(NP), h_pw(NP);
    float h_beam_ranges[NUM_BEAMS];

    // Pre-render map image
    cv::Mat map_img(IMG_H, IMG_W, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int gy = 0; gy < GRID_H; gy++)
        for (int gx = 0; gx < GRID_W; gx++)
            if (h_occupancy[gy * GRID_W + gx] == 1) {
                int px_x = gx * VIS_SCALE;
                int px_y = (GRID_H - 1 - gy) * VIS_SCALE;
                cv::rectangle(map_img, cv::Point(px_x, px_y),
                    cv::Point(px_x + VIS_SCALE - 1, px_y + VIS_SCALE - 1),
                    cv::Scalar(0, 0, 0), -1);
            }

    // Video
    cv::namedWindow("comparison_amcl", cv::WINDOW_NORMAL);
    cv::VideoWriter video(
        "gif/comparison_amcl.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
        cv::Size(COMBINED_W, COMBINED_H));

    float time_val = 0.0f;
    int step_count = 0;
    float robot_v = 1.0f, robot_omega = 0.0f;

    while (time_val <= SIM_TIME) {
        time_val += DT;
        step_count++;

        // Waypoint navigation
        float targets[][2] = {
            {15.0f, 5.0f}, {15.0f, 15.0f}, {5.0f, 15.0f},
            {5.0f, 8.0f}, {12.0f, 8.0f}, {12.0f, 17.0f},
            {3.0f, 17.0f}, {3.0f, 5.0f}
        };
        int wp_idx = ((int)(time_val / 7.5f)) % 8;
        float dx = targets[wp_idx][0] - gt_x;
        float dy = targets[wp_idx][1] - gt_y;
        float target_angle = atan2f(dy, dx);
        float angle_diff = target_angle - gt_theta;
        while (angle_diff > PI) angle_diff -= 2.0f * PI;
        while (angle_diff < -PI) angle_diff += 2.0f * PI;
        robot_v = 1.0f;
        robot_omega = angle_diff * 2.0f;
        if (robot_omega > 1.5f) robot_omega = 1.5f;
        if (robot_omega < -1.5f) robot_omega = -1.5f;

        // Update ground truth
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

        // Simulate lidar
        simulate_lidar(h_occupancy, gt_x, gt_y, gt_theta,
                       origin_x, origin_y, GRID_RES, GRID_W, GRID_H, h_beam_ranges);
        for (int b = 0; b < NUM_BEAMS; b++) {
            h_beam_ranges[b] += gauss(gen) * 0.05f;
            if (h_beam_ranges[b] < 0.0f) h_beam_ranges[b] = 0.0f;
        }

        // ===================== CPU step =====================
        float cpu_est_x, cpu_est_y, cpu_est_theta;
        {
            auto t0 = std::chrono::high_resolution_clock::now();

            cpu_predict(cpu_p.data(), NP, robot_v, robot_omega, gen_cpu);
            cpu_update_weights(cpu_p.data(), NP, h_likelihood_field, h_beam_ranges,
                               origin_x, origin_y, GRID_RES, GRID_W, GRID_H);
            float neff;
            cpu_normalize_and_neff(cpu_p.data(), NP, neff);
            cpu_weighted_mean(cpu_p.data(), NP, cpu_est_x, cpu_est_y, cpu_est_theta);
            if (neff < NP * 0.5f) {
                cpu_resample(cpu_p.data(), NP, gen_cpu);
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            cpu_total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        // ===================== CUDA step =====================
        float cuda_est_x, cuda_est_y, cuda_est_theta;
        {
            auto t0 = std::chrono::high_resolution_clock::now();

            CUDA_CHECK(cudaMemcpy(d_beam_ranges, h_beam_ranges, NUM_BEAMS * sizeof(float), cudaMemcpyHostToDevice));

            predict_particles_kernel<<<pblocks, threads>>>(
                d_px, d_py, d_ptheta, robot_v, robot_omega, DT, d_rng_states, NP);

            update_weights_kernel<<<pblocks, threads>>>(
                d_px, d_py, d_ptheta, d_pw, d_likelihood_field, d_beam_ranges,
                GRID_W, GRID_H, GRID_RES, origin_x, origin_y, NP);

            compute_neff_kernel<<<1, threads, 2 * threads * sizeof(float)>>>(d_pw, NP, d_sum, d_sum_sq);
            CUDA_CHECK(cudaDeviceSynchronize());

            float h_sum, h_sum_sq;
            CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(float), cudaMemcpyDeviceToHost));

            normalize_weights_kernel<<<pblocks, threads>>>(d_pw, NP, h_sum);

            weighted_mean_kernel<<<1, threads, 4 * threads * sizeof(float)>>>(
                d_px, d_py, d_ptheta, d_pw, d_est_x, d_est_y, d_est_theta, NP);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(&cuda_est_x, d_est_x, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&cuda_est_y, d_est_y, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&cuda_est_theta, d_est_theta, sizeof(float), cudaMemcpyDeviceToHost));

            // Resampling
            float neff;
            if (h_sum > 1e-30f) {
                neff = 1.0f / (h_sum_sq / (h_sum * h_sum));
            } else {
                neff = 0.0f;
            }

            if (neff < NP * 0.5f) {
                cumsum_kernel<<<1, 1>>>(d_pw, d_wcum, NP);
                CUDA_CHECK(cudaDeviceSynchronize());
                float rand_offset = uni(gen) / (float)NP;
                float base_step = 1.0f / (float)NP;
                resample_kernel<<<pblocks, threads>>>(
                    d_px, d_py, d_ptheta,
                    d_px_tmp, d_py_tmp, d_ptheta_tmp,
                    d_wcum, base_step, rand_offset, NP, NP);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(d_px, d_px_tmp, NP * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_py, d_py_tmp, NP * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_ptheta, d_ptheta_tmp, NP * sizeof(float), cudaMemcpyDeviceToDevice));

                std::vector<float> pw_uniform(NP, 1.0f / (float)NP);
                CUDA_CHECK(cudaMemcpy(d_pw, pw_uniform.data(), NP * sizeof(float), cudaMemcpyHostToDevice));
            }

            CUDA_CHECK(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            cuda_total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        // Read back CUDA particles for visualization
        CUDA_CHECK(cudaMemcpy(h_px.data(), d_px, NP * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_py.data(), d_py, NP * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ptheta.data(), d_ptheta, NP * sizeof(float), cudaMemcpyDeviceToHost));

        // ===================== Visualization =====================
        cv::Mat combined(COMBINED_H, COMBINED_W, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Mat top    = combined(cv::Rect(0, 0, PANEL_W, PANEL_H));
        cv::Mat bottom = combined(cv::Rect(0, PANEL_H, PANEL_W, PANEL_H));

        // Draw both panels
        for (int panel = 0; panel < 2; panel++) {
            cv::Mat& frame = (panel == 0) ? top : bottom;
            map_img.copyTo(frame);

            // Draw particles (red)
            for (int i = 0; i < NP; i++) {
                float wx, wy;
                if (panel == 0) { wx = cpu_p[i].x; wy = cpu_p[i].y; }
                else { wx = h_px[i]; wy = h_py[i]; }
                cv::Point2i pt = world_to_pixel(wx, wy, origin_x, origin_y, GRID_RES);
                cv::circle(frame, pt, 2, cv::Scalar(0, 0, 255), -1);
            }

            // Ground truth (green arrow)
            cv::Point2i gt_pt = world_to_pixel(gt_x, gt_y, origin_x, origin_y, GRID_RES);
            draw_arrow(frame, gt_pt, gt_theta, cv::Scalar(0, 200, 0), 20, 2);
            cv::circle(frame, gt_pt, 4, cv::Scalar(0, 200, 0), -1);

            // Estimate (blue arrow)
            float ex, ey, eth;
            if (panel == 0) { ex = cpu_est_x; ey = cpu_est_y; eth = cpu_est_theta; }
            else { ex = cuda_est_x; ey = cuda_est_y; eth = cuda_est_theta; }
            cv::Point2i est_pt = world_to_pixel(ex, ey, origin_x, origin_y, GRID_RES);
            draw_arrow(frame, est_pt, eth, cv::Scalar(255, 0, 0), 20, 2);
            cv::circle(frame, est_pt, 4, cv::Scalar(255, 0, 0), -1);
        }

        // Divider
        cv::line(combined, cv::Point(0, PANEL_H), cv::Point(COMBINED_W, PANEL_H),
                 cv::Scalar(100, 100, 100), 2);

        // Timing text
        double cpu_avg = cpu_total_ms / step_count;
        double cuda_avg = cuda_total_ms / step_count;
        char buf[128];

        snprintf(buf, sizeof(buf), "CPU AMCL  -  CPU: %.2f ms/step", cpu_avg);
        cv::putText(combined, buf, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

        snprintf(buf, sizeof(buf), "CUDA AMCL  -  CUDA: %.2f ms/step", cuda_avg);
        cv::putText(combined, buf, cv::Point(10, PANEL_H + 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

        snprintf(buf, sizeof(buf), "t=%.1fs  NP=%d", time_val, NP);
        cv::putText(combined, buf, cv::Point(10, COMBINED_H - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);

        cv::imshow("comparison_amcl", combined);
        video.write(combined);
        cv::waitKey(5);
    }

    video.release();
    cv::destroyAllWindows();

    printf("\nFinal timing: CPU=%.2f ms/step, CUDA=%.2f ms/step\n",
           cpu_total_ms / step_count, cuda_total_ms / step_count);

    // Convert to gif
    int ret = system("which ffmpeg > /dev/null 2>&1 && "
        "ffmpeg -y -i gif/comparison_amcl.avi "
        "-vf \"fps=15,scale=400:-1:flags=lanczos\" "
        "-gifflags +transdiff "
        "gif/comparison_amcl.gif 2>/dev/null && "
        "echo 'GIF saved' || echo 'ffmpeg not available, skipping gif'");
    (void)ret;

    // Cleanup
    cudaFree(d_occupancy); cudaFree(d_likelihood_field);
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_ptheta); cudaFree(d_pw);
    cudaFree(d_px_tmp); cudaFree(d_py_tmp); cudaFree(d_ptheta_tmp);
    cudaFree(d_wcum); cudaFree(d_rng_states);
    cudaFree(d_sum); cudaFree(d_sum_sq);
    cudaFree(d_est_x); cudaFree(d_est_y); cudaFree(d_est_theta);
    cudaFree(d_beam_ranges);

    return 0;
}
