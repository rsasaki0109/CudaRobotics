/*************************************************************************
    FastSLAM 1.0: CPU vs CUDA side-by-side comparison GIF generator
    Left panel: CPU (sequential per-particle predict + EKF update)
    Right panel: CUDA (parallel GPU kernels)
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
#define SIM_TIME 20.0f
#define DT 0.1f
#define PI 3.141592653f
#define MAX_RANGE 20.0f
#define NP 100
#define NTh (NP / 2)
#define MAX_LANDMARKS 8

// Noise parameters
#define SIGMA_V 0.5f
#define SIGMA_OMEGA 0.1f
#define SIGMA_RANGE 0.5f
#define SIGMA_BEARING 0.1f

// Observation noise covariance Q_obs (2x2 diagonal)
#define Q_RANGE  (SIGMA_RANGE * SIGMA_RANGE)
#define Q_BEARING (SIGMA_BEARING * SIGMA_BEARING)

// Visualization
#define PANEL_W 1750
#define PANEL_H 1750
#define COMBINED_W (PANEL_W * 2)
#define COMBINED_H PANEL_H

// ---------------------------------------------------------------------------
// CUDA error check macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

// ---------------------------------------------------------------------------
// Observation structure
// ---------------------------------------------------------------------------
struct Observation {
    float range;
    float bearing;
    int lm_id;
};

// ---------------------------------------------------------------------------
// Host helper: normalize angle to [-pi, pi]
// ---------------------------------------------------------------------------
static float h_normalize_angle(float a) {
    a = fmodf(a + PI, 2.0f * PI);
    if (a < 0.0f) a += 2.0f * PI;
    return a - PI;
}

// ---------------------------------------------------------------------------
// Coordinate transform for visualization
// ---------------------------------------------------------------------------
static cv::Point2i cv_offset(float ex, float ey, int image_width, int image_height) {
    cv::Point2i output;
    output.x = int(ex * 50) + image_width / 2;
    output.y = image_height - int(ey * 50) - image_height / 3;
    return output;
}

// ===========================================================================
// CPU Implementation
// ===========================================================================

struct CPUParticle {
    float x, y, yaw;
    float weight;
    float lm_mean[MAX_LANDMARKS * 2];    // [lm_id * 2 + {0,1}] = {x, y}
    float lm_cov[MAX_LANDMARKS * 4];     // [lm_id * 4 + {0,1,2,3}] = 2x2 row-major
    int   lm_seen[MAX_LANDMARKS];
};

static void cpu_predict(CPUParticle* particles, int np, float u_v, float u_omega,
                        std::mt19937& gen, std::normal_distribution<float>& gauss) {
    for (int i = 0; i < np; i++) {
        float nv = u_v + gauss(gen) * SIGMA_V;
        float no = u_omega + gauss(gen) * SIGMA_OMEGA;
        particles[i].x   += DT * cosf(particles[i].yaw) * nv;
        particles[i].y   += DT * sinf(particles[i].yaw) * nv;
        particles[i].yaw += DT * no;
        particles[i].yaw  = h_normalize_angle(particles[i].yaw);
    }
}

static void cpu_update_landmarks(CPUParticle* particles, int np,
                                  const std::vector<Observation>& obs) {
    for (int ip = 0; ip < np; ip++) {
        float px = particles[ip].x;
        float py = particles[ip].y;
        float pyaw = particles[ip].yaw;
        float w = 1.0f;

        for (int io = 0; io < (int)obs.size(); io++) {
            float z_range   = obs[io].range;
            float z_bearing = obs[io].bearing;
            int   lm_id     = obs[io].lm_id;

            float* mu = &particles[ip].lm_mean[lm_id * 2];
            float* P  = &particles[ip].lm_cov[lm_id * 4];

            if (particles[ip].lm_seen[lm_id] == 0) {
                // Initialize landmark
                float cs = cosf(z_bearing + pyaw);
                float sn = sinf(z_bearing + pyaw);
                mu[0] = px + z_range * cs;
                mu[1] = py + z_range * sn;

                // Jacobian of inverse obs
                float g00 = cs, g01 = -z_range * sn;
                float g10 = sn, g11 =  z_range * cs;
                P[0] = g00 * g00 * Q_RANGE + g01 * g01 * Q_BEARING;
                P[1] = g00 * g10 * Q_RANGE + g01 * g11 * Q_BEARING;
                P[2] = P[1];
                P[3] = g10 * g10 * Q_RANGE + g11 * g11 * Q_BEARING;

                particles[ip].lm_seen[lm_id] = 1;
            } else {
                // EKF update
                float mu_x = mu[0], mu_y = mu[1];
                float P00 = P[0], P01 = P[1], P10 = P[2], P11 = P[3];

                float dx = mu_x - px;
                float dy = mu_y - py;
                float q = dx * dx + dy * dy;
                float sq = sqrtf(q);

                float innov_r = z_range - sq;
                float innov_b = h_normalize_angle(z_bearing - (atan2f(dy, dx) - pyaw));

                // Jacobian H
                float H00 = dx / sq, H01 = dy / sq;
                float H10 = -dy / q,  H11 = dx / q;

                // HP = H * P
                float HP00 = H00 * P00 + H01 * P10;
                float HP01 = H00 * P01 + H01 * P11;
                float HP10 = H10 * P00 + H11 * P10;
                float HP11 = H10 * P01 + H11 * P11;

                // S = HP * H^T + Q
                float S00 = HP00 * H00 + HP01 * H01 + Q_RANGE;
                float S01 = HP00 * H10 + HP01 * H11;
                float S10 = HP10 * H00 + HP11 * H01;
                float S11 = HP10 * H10 + HP11 * H11 + Q_BEARING;

                // S^{-1}
                float det_S = S00 * S11 - S01 * S10;
                if (fabsf(det_S) < 1e-10f) det_S = 1e-10f;
                float inv_det = 1.0f / det_S;
                float Si00 =  S11 * inv_det;
                float Si01 = -S01 * inv_det;
                float Si10 = -S10 * inv_det;
                float Si11 =  S00 * inv_det;

                // K = P * H^T * S^{-1}
                float PHt00 = P00 * H00 + P01 * H01;
                float PHt01 = P00 * H10 + P01 * H11;
                float PHt10 = P10 * H00 + P11 * H01;
                float PHt11 = P10 * H10 + P11 * H11;

                float K00 = PHt00 * Si00 + PHt01 * Si10;
                float K01 = PHt00 * Si01 + PHt01 * Si11;
                float K10 = PHt10 * Si00 + PHt11 * Si10;
                float K11 = PHt10 * Si01 + PHt11 * Si11;

                // Update mean
                mu[0] = mu_x + K00 * innov_r + K01 * innov_b;
                mu[1] = mu_y + K10 * innov_r + K11 * innov_b;

                // Update covariance: P = (I - K*H) * P
                float KH00 = K00 * H00 + K01 * H10;
                float KH01 = K00 * H01 + K01 * H11;
                float KH10 = K10 * H00 + K11 * H10;
                float KH11 = K10 * H01 + K11 * H11;

                P[0] = (1.0f - KH00) * P00 + (-KH01) * P10;
                P[1] = (1.0f - KH00) * P01 + (-KH01) * P11;
                P[2] = (-KH10) * P00 + (1.0f - KH11) * P10;
                P[3] = (-KH10) * P01 + (1.0f - KH11) * P11;

                // Weight
                float det_2piS = (2.0f * PI) * (2.0f * PI) * fabsf(det_S);
                float inv_sqrt_det = 1.0f / sqrtf(det_2piS + 1e-30f);
                float maha = innov_r * (Si00 * innov_r + Si01 * innov_b)
                           + innov_b * (Si10 * innov_r + Si11 * innov_b);
                w *= inv_sqrt_det * expf(-0.5f * maha);
            }
        }
        particles[ip].weight = w;
    }
}

static void cpu_normalize_weights(CPUParticle* particles, int np) {
    float total = 0.0f;
    for (int i = 0; i < np; i++) total += particles[i].weight;
    if (total < 1e-30f) total = 1e-30f;
    for (int i = 0; i < np; i++) particles[i].weight /= total;
}

static void cpu_weighted_mean(const CPUParticle* particles, int np, float* xEst) {
    float sx = 0, sy = 0, syaw = 0;
    for (int i = 0; i < np; i++) {
        sx += particles[i].x * particles[i].weight;
        sy += particles[i].y * particles[i].weight;
        syaw += particles[i].yaw * particles[i].weight;
    }
    xEst[0] = sx; xEst[1] = sy; xEst[2] = syaw;
}

static void cpu_weighted_mean_landmarks(const CPUParticle* particles, int np,
                                         float* lm_est, int* lm_valid) {
    for (int lm = 0; lm < MAX_LANDMARKS; lm++) {
        float sx = 0, sy = 0, sw = 0;
        for (int ip = 0; ip < np; ip++) {
            if (particles[ip].lm_seen[lm]) {
                float w = particles[ip].weight;
                sx += w * particles[ip].lm_mean[lm * 2 + 0];
                sy += w * particles[ip].lm_mean[lm * 2 + 1];
                sw += w;
            }
        }
        if (sw > 1e-10f) {
            lm_est[lm * 2 + 0] = sx / sw;
            lm_est[lm * 2 + 1] = sy / sw;
            lm_valid[lm] = 1;
        } else {
            lm_valid[lm] = 0;
        }
    }
}

static void cpu_resample(CPUParticle* particles, int np, std::mt19937& gen) {
    // Compute Neff
    float sum_sq = 0.0f;
    for (int i = 0; i < np; i++) sum_sq += particles[i].weight * particles[i].weight;
    float neff = 1.0f / (sum_sq + 1e-30f);
    if (neff >= NTh) return;

    // Build cumulative sum
    std::vector<float> wcum(np);
    wcum[0] = particles[0].weight;
    for (int i = 1; i < np; i++) wcum[i] = wcum[i-1] + particles[i].weight;

    // Systematic resampling
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    float r = uni(gen) / np;
    float step = 1.0f / np;

    std::vector<CPUParticle> new_particles(np);
    for (int i = 0; i < np; i++) {
        float target = step * i + r;
        int lo = 0, hi = np - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (wcum[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        new_particles[i] = particles[lo];
        new_particles[i].weight = 1.0f / np;
    }
    for (int i = 0; i < np; i++) particles[i] = new_particles[i];
}

// ===========================================================================
// CUDA Kernels
// ===========================================================================

__device__ float normalize_angle(float a) {
    a = fmodf(a + PI, 2.0f * PI);
    if (a < 0.0f) a += 2.0f * PI;
    return a - PI;
}

__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void predict_particles_kernel(
    float* pose, float u_v, float u_omega,
    curandState* rng_states, int np)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    curandState local_rng = rng_states[ip];
    float noisy_v     = u_v     + curand_normal(&local_rng) * SIGMA_V;
    float noisy_omega = u_omega + curand_normal(&local_rng) * SIGMA_OMEGA;

    float x   = pose[ip * 3 + 0];
    float y   = pose[ip * 3 + 1];
    float yaw = pose[ip * 3 + 2];

    x   += DT * cosf(yaw) * noisy_v;
    y   += DT * sinf(yaw) * noisy_v;
    yaw += DT * noisy_omega;
    yaw  = normalize_angle(yaw);

    pose[ip * 3 + 0] = x;
    pose[ip * 3 + 1] = y;
    pose[ip * 3 + 2] = yaw;
    rng_states[ip] = local_rng;
}

__global__ void update_landmarks_kernel(
    float* pose, float* weights,
    float* landmark_mean, float* landmark_cov, int* landmark_seen,
    const Observation* obs, int n_obs, int np)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    float px = pose[ip * 3 + 0];
    float py = pose[ip * 3 + 1];
    float pyaw = pose[ip * 3 + 2];
    float w = 1.0f;

    for (int io = 0; io < n_obs; io++) {
        float z_range   = obs[io].range;
        float z_bearing = obs[io].bearing;
        int   lm_id     = obs[io].lm_id;

        int mean_base = ip * MAX_LANDMARKS * 2 + lm_id * 2;
        int cov_base  = ip * MAX_LANDMARKS * 4 + lm_id * 4;
        int seen_idx  = ip * MAX_LANDMARKS + lm_id;

        if (landmark_seen[seen_idx] == 0) {
            float cs = cosf(z_bearing + pyaw);
            float sn = sinf(z_bearing + pyaw);
            landmark_mean[mean_base + 0] = px + z_range * cs;
            landmark_mean[mean_base + 1] = py + z_range * sn;

            float g00 = cs, g01 = -z_range * sn;
            float g10 = sn, g11 =  z_range * cs;
            landmark_cov[cov_base + 0] = g00 * g00 * Q_RANGE + g01 * g01 * Q_BEARING;
            landmark_cov[cov_base + 1] = g00 * g10 * Q_RANGE + g01 * g11 * Q_BEARING;
            landmark_cov[cov_base + 2] = landmark_cov[cov_base + 1];
            landmark_cov[cov_base + 3] = g10 * g10 * Q_RANGE + g11 * g11 * Q_BEARING;
            landmark_seen[seen_idx] = 1;
        } else {
            float mu_x = landmark_mean[mean_base + 0];
            float mu_y = landmark_mean[mean_base + 1];
            float P00 = landmark_cov[cov_base + 0];
            float P01 = landmark_cov[cov_base + 1];
            float P10 = landmark_cov[cov_base + 2];
            float P11 = landmark_cov[cov_base + 3];

            float dx = mu_x - px, dy = mu_y - py;
            float q = dx * dx + dy * dy;
            float sq = sqrtf(q);
            float innov_r = z_range - sq;
            float innov_b = normalize_angle(z_bearing - (atan2f(dy, dx) - pyaw));

            float H00 = dx / sq, H01 = dy / sq;
            float H10 = -dy / q,  H11 = dx / q;

            float HP00 = H00 * P00 + H01 * P10;
            float HP01 = H00 * P01 + H01 * P11;
            float HP10 = H10 * P00 + H11 * P10;
            float HP11 = H10 * P01 + H11 * P11;

            float S00 = HP00 * H00 + HP01 * H01 + Q_RANGE;
            float S01 = HP00 * H10 + HP01 * H11;
            float S10 = HP10 * H00 + HP11 * H01;
            float S11 = HP10 * H10 + HP11 * H11 + Q_BEARING;

            float det_S = S00 * S11 - S01 * S10;
            if (fabsf(det_S) < 1e-10f) det_S = 1e-10f;
            float inv_det = 1.0f / det_S;
            float Si00 =  S11 * inv_det, Si01 = -S01 * inv_det;
            float Si10 = -S10 * inv_det, Si11 =  S00 * inv_det;

            float PHt00 = P00 * H00 + P01 * H01;
            float PHt01 = P00 * H10 + P01 * H11;
            float PHt10 = P10 * H00 + P11 * H01;
            float PHt11 = P10 * H10 + P11 * H11;

            float K00 = PHt00 * Si00 + PHt01 * Si10;
            float K01 = PHt00 * Si01 + PHt01 * Si11;
            float K10 = PHt10 * Si00 + PHt11 * Si10;
            float K11 = PHt10 * Si01 + PHt11 * Si11;

            landmark_mean[mean_base + 0] = mu_x + K00 * innov_r + K01 * innov_b;
            landmark_mean[mean_base + 1] = mu_y + K10 * innov_r + K11 * innov_b;

            float KH00 = K00 * H00 + K01 * H10;
            float KH01 = K00 * H01 + K01 * H11;
            float KH10 = K10 * H00 + K11 * H10;
            float KH11 = K10 * H01 + K11 * H11;

            landmark_cov[cov_base + 0] = (1.0f - KH00) * P00 + (-KH01) * P10;
            landmark_cov[cov_base + 1] = (1.0f - KH00) * P01 + (-KH01) * P11;
            landmark_cov[cov_base + 2] = (-KH10) * P00 + (1.0f - KH11) * P10;
            landmark_cov[cov_base + 3] = (-KH10) * P01 + (1.0f - KH11) * P11;

            float det_2piS = (2.0f * PI) * (2.0f * PI) * fabsf(det_S);
            float inv_sqrt_det = 1.0f / sqrtf(det_2piS + 1e-30f);
            float maha = innov_r * (Si00 * innov_r + Si01 * innov_b)
                       + innov_b * (Si10 * innov_r + Si11 * innov_b);
            w *= inv_sqrt_det * expf(-0.5f * maha);
        }
    }
    weights[ip] = w;
}

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

__global__ void weighted_mean_kernel(const float* pose, const float* pw,
                                     float* x_est, int np) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float s0 = 0, s1 = 0, s2 = 0;
    for (int i = tid; i < np; i += blockDim.x) {
        float w = pw[i];
        s0 += pose[i * 3 + 0] * w;
        s1 += pose[i * 3 + 1] * w;
        s2 += pose[i * 3 + 2] * w;
    }
    sdata[tid * 3 + 0] = s0;
    sdata[tid * 3 + 1] = s1;
    sdata[tid * 3 + 2] = s2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid * 3 + 0] += sdata[(tid + s) * 3 + 0];
            sdata[tid * 3 + 1] += sdata[(tid + s) * 3 + 1];
            sdata[tid * 3 + 2] += sdata[(tid + s) * 3 + 2];
        }
        __syncthreads();
    }
    if (tid == 0) { x_est[0] = sdata[0]; x_est[1] = sdata[1]; x_est[2] = sdata[2]; }
}

__global__ void cumsum_kernel(const float* pw, float* wcum, int np) {
    wcum[0] = pw[0];
    for (int i = 1; i < np; i++) wcum[i] = wcum[i - 1] + pw[i];
}

__global__ void resample_kernel(
    const float* pose_in, float* pose_out,
    const float* lm_mean_in, float* lm_mean_out,
    const float* lm_cov_in, float* lm_cov_out,
    const int* lm_seen_in, int* lm_seen_out,
    const float* wcum, float base_step, float rand_offset, int np)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    float target = base_step * ip + rand_offset;
    int lo = 0, hi = np - 1;
    while (lo < hi) { int mid = (lo + hi) / 2; if (wcum[mid] < target) lo = mid + 1; else hi = mid; }

    pose_out[ip * 3 + 0] = pose_in[lo * 3 + 0];
    pose_out[ip * 3 + 1] = pose_in[lo * 3 + 1];
    pose_out[ip * 3 + 2] = pose_in[lo * 3 + 2];

    int src_mean = lo * MAX_LANDMARKS * 2, dst_mean = ip * MAX_LANDMARKS * 2;
    int src_cov  = lo * MAX_LANDMARKS * 4, dst_cov  = ip * MAX_LANDMARKS * 4;
    int src_seen = lo * MAX_LANDMARKS,     dst_seen = ip * MAX_LANDMARKS;
    for (int lm = 0; lm < MAX_LANDMARKS; lm++) {
        lm_mean_out[dst_mean + lm * 2 + 0] = lm_mean_in[src_mean + lm * 2 + 0];
        lm_mean_out[dst_mean + lm * 2 + 1] = lm_mean_in[src_mean + lm * 2 + 1];
        lm_cov_out[dst_cov + lm * 4 + 0] = lm_cov_in[src_cov + lm * 4 + 0];
        lm_cov_out[dst_cov + lm * 4 + 1] = lm_cov_in[src_cov + lm * 4 + 1];
        lm_cov_out[dst_cov + lm * 4 + 2] = lm_cov_in[src_cov + lm * 4 + 2];
        lm_cov_out[dst_cov + lm * 4 + 3] = lm_cov_in[src_cov + lm * 4 + 3];
        lm_seen_out[dst_seen + lm] = lm_seen_in[src_seen + lm];
    }
}

__global__ void weighted_mean_landmarks_kernel(
    const float* pw, const float* lm_mean, const int* lm_seen,
    float* lm_est, int* lm_est_valid, int np)
{
    int lm_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (lm_id >= MAX_LANDMARKS) return;

    float sum_x = 0, sum_y = 0, sum_w = 0;
    for (int ip = 0; ip < np; ip++) {
        if (lm_seen[ip * MAX_LANDMARKS + lm_id]) {
            float w = pw[ip];
            sum_x += w * lm_mean[ip * MAX_LANDMARKS * 2 + lm_id * 2 + 0];
            sum_y += w * lm_mean[ip * MAX_LANDMARKS * 2 + lm_id * 2 + 1];
            sum_w += w;
        }
    }
    if (sum_w > 1e-10f) {
        lm_est[lm_id * 2 + 0] = sum_x / sum_w;
        lm_est[lm_id * 2 + 1] = sum_y / sum_w;
        lm_est_valid[lm_id] = 1;
    } else {
        lm_est_valid[lm_id] = 0;
    }
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    std::cout << "FastSLAM 1.0 comparison: CPU vs CUDA (" << NP << " particles, "
              << MAX_LANDMARKS << " landmarks)" << std::endl;

    // RFID landmarks
    const int N_TRUE_LM = 8;
    float RFID[N_TRUE_LM][2] = {
        {10.0f,  0.0f}, {15.0f, 10.0f}, {15.0f, 15.0f}, {10.0f, 20.0f},
        { 3.0f, 15.0f}, {-5.0f, 20.0f}, {-5.0f,  5.0f}, { 0.0f, 10.0f}
    };

    float u_v = 1.0f, u_omega = 0.1f;

    // Ground truth and dead reckoning (shared)
    float xTrue[3] = {0, 0, 0};
    float xDR[3]   = {0, 0, 0};

    std::vector<float> hxTrue_x, hxTrue_y, hxDR_x, hxDR_y;

    std::mt19937 gen_shared(42);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    // ---- CPU state ----
    std::vector<CPUParticle> cpu_particles(NP);
    for (int i = 0; i < NP; i++) {
        cpu_particles[i].x = 0; cpu_particles[i].y = 0; cpu_particles[i].yaw = 0;
        cpu_particles[i].weight = 1.0f / NP;
        memset(cpu_particles[i].lm_mean, 0, sizeof(cpu_particles[i].lm_mean));
        memset(cpu_particles[i].lm_cov, 0, sizeof(cpu_particles[i].lm_cov));
        memset(cpu_particles[i].lm_seen, 0, sizeof(cpu_particles[i].lm_seen));
    }
    float cpu_xEst[3] = {0, 0, 0};
    float cpu_lm_est[MAX_LANDMARKS * 2];
    int   cpu_lm_valid[MAX_LANDMARKS];
    std::vector<float> hxEst_cpu_x, hxEst_cpu_y;
    std::mt19937 gen_cpu(42);
    double cpu_total_ms = 0.0;

    // ---- CUDA state ----
    const int np = NP;
    const int threads = 256;
    const int blocks = (np + threads - 1) / threads;

    float *d_pose, *d_pose_tmp, *d_pw;
    CUDA_CHECK(cudaMalloc(&d_pose,     np * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pose_tmp, np * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pose, 0,   np * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pw, np * sizeof(float)));
    { std::vector<float> pw_init(np, 1.0f / np); CUDA_CHECK(cudaMemcpy(d_pw, pw_init.data(), np * sizeof(float), cudaMemcpyHostToDevice)); }

    float *d_lm_mean, *d_lm_mean_tmp, *d_lm_cov, *d_lm_cov_tmp;
    int *d_lm_seen, *d_lm_seen_tmp;
    CUDA_CHECK(cudaMalloc(&d_lm_mean,     np * MAX_LANDMARKS * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_mean_tmp, np * MAX_LANDMARKS * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_lm_mean, 0,   np * MAX_LANDMARKS * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_cov,     np * MAX_LANDMARKS * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_cov_tmp, np * MAX_LANDMARKS * 4 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_lm_cov, 0,   np * MAX_LANDMARKS * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_seen,     np * MAX_LANDMARKS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lm_seen_tmp, np * MAX_LANDMARKS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_lm_seen, 0,   np * MAX_LANDMARKS * sizeof(int)));

    Observation *d_obs;
    CUDA_CHECK(cudaMalloc(&d_obs, MAX_LANDMARKS * sizeof(Observation)));

    curandState *d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, np * sizeof(curandState)));
    init_curand_kernel<<<blocks, threads>>>(d_rng_states, 42ULL, np);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *d_xEst, *d_wcum, *d_lm_est;
    int *d_lm_est_valid;
    CUDA_CHECK(cudaMalloc(&d_xEst, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wcum, np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_est, MAX_LANDMARKS * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_est_valid, MAX_LANDMARKS * sizeof(int)));

    std::vector<float> h_pose(np * 3), h_pw(np);
    float h_xEst[3];
    float h_lm_est[MAX_LANDMARKS * 2];
    int h_lm_est_valid[MAX_LANDMARKS];
    float cuda_xEst[3] = {0, 0, 0};
    std::vector<float> hxEst_cuda_x, hxEst_cuda_y;
    double cuda_total_ms = 0.0;

    // ---- Video ----
    cv::namedWindow("comparison_fastslam", cv::WINDOW_NORMAL);
    cv::VideoWriter video(
        "gif/comparison_fastslam.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
        cv::Size(COMBINED_W, COMBINED_H));

    int step_count = 0;
    float time_val = 0.0f;

    while (time_val <= SIM_TIME) {
        time_val += DT;
        step_count++;

        // --- Ground truth ---
        xTrue[0] += DT * cosf(xTrue[2]) * u_v;
        xTrue[1] += DT * sinf(xTrue[2]) * u_v;
        xTrue[2] += DT * u_omega;

        // --- Dead reckoning ---
        float nv = u_v + gauss(gen_shared) * SIGMA_V;
        float no = u_omega + gauss(gen_shared) * SIGMA_OMEGA;
        xDR[0] += DT * cosf(xDR[2]) * nv;
        xDR[1] += DT * sinf(xDR[2]) * nv;
        xDR[2] += DT * no;

        // --- Generate observations ---
        std::vector<Observation> z_host;
        for (int i = 0; i < N_TRUE_LM; i++) {
            float dx = xTrue[0] - RFID[i][0];
            float dy = xTrue[1] - RFID[i][1];
            float d = sqrtf(dx * dx + dy * dy);
            if (d <= MAX_RANGE) {
                Observation ob;
                ob.range   = d + gauss(gen_shared) * SIGMA_RANGE;
                ob.bearing = atan2f(dy, dx) - xTrue[2] + gauss(gen_shared) * SIGMA_BEARING;
                ob.lm_id   = i;
                z_host.push_back(ob);
            }
        }
        int n_obs = (int)z_host.size();

        // ===================== CPU step =====================
        {
            auto t0 = std::chrono::high_resolution_clock::now();

            cpu_predict(cpu_particles.data(), NP, u_v, u_omega, gen_cpu, gauss);
            if (n_obs > 0) {
                cpu_update_landmarks(cpu_particles.data(), NP, z_host);
                cpu_normalize_weights(cpu_particles.data(), NP);
            }
            cpu_weighted_mean(cpu_particles.data(), NP, cpu_xEst);
            cpu_weighted_mean_landmarks(cpu_particles.data(), NP, cpu_lm_est, cpu_lm_valid);
            cpu_resample(cpu_particles.data(), NP, gen_cpu);

            auto t1 = std::chrono::high_resolution_clock::now();
            cpu_total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        // ===================== CUDA step =====================
        {
            auto t0 = std::chrono::high_resolution_clock::now();

            if (n_obs > 0) {
                CUDA_CHECK(cudaMemcpy(d_obs, z_host.data(), n_obs * sizeof(Observation), cudaMemcpyHostToDevice));
            }

            predict_particles_kernel<<<blocks, threads>>>(d_pose, u_v, u_omega, d_rng_states, np);

            if (n_obs > 0) {
                update_landmarks_kernel<<<blocks, threads>>>(
                    d_pose, d_pw, d_lm_mean, d_lm_cov, d_lm_seen, d_obs, n_obs, np);
                normalize_weights_kernel<<<1, threads, threads * sizeof(float)>>>(d_pw, np);
            }

            weighted_mean_kernel<<<1, threads, threads * 3 * sizeof(float)>>>(d_pose, d_pw, d_xEst, np);

            int lm_blocks = (MAX_LANDMARKS + threads - 1) / threads;
            weighted_mean_landmarks_kernel<<<lm_blocks, threads>>>(
                d_pw, d_lm_mean, d_lm_seen, d_lm_est, d_lm_est_valid, np);

            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_xEst, d_xEst, 3 * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_lm_est, d_lm_est, MAX_LANDMARKS * 2 * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_lm_est_valid, d_lm_est_valid, MAX_LANDMARKS * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_pw.data(), d_pw, np * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_pose.data(), d_pose, np * 3 * sizeof(float), cudaMemcpyDeviceToHost));

            // Resampling
            float Neff_denom = 0.0f;
            for (int i = 0; i < np; i++) Neff_denom += h_pw[i] * h_pw[i];
            float Neff = 1.0f / (Neff_denom + 1e-30f);

            if (Neff < NTh) {
                cumsum_kernel<<<1, 1>>>(d_pw, d_wcum, np);
                float rand_offset = uni(gen_shared) / np;
                float base_step = 1.0f / np;
                resample_kernel<<<blocks, threads>>>(
                    d_pose, d_pose_tmp, d_lm_mean, d_lm_mean_tmp,
                    d_lm_cov, d_lm_cov_tmp, d_lm_seen, d_lm_seen_tmp,
                    d_wcum, base_step, rand_offset, np);
                CUDA_CHECK(cudaMemcpy(d_pose, d_pose_tmp, np * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_lm_mean, d_lm_mean_tmp, np * MAX_LANDMARKS * 2 * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_lm_cov, d_lm_cov_tmp, np * MAX_LANDMARKS * 4 * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_lm_seen, d_lm_seen_tmp, np * MAX_LANDMARKS * sizeof(int), cudaMemcpyDeviceToDevice));

                std::vector<float> pw_uniform(np, 1.0f / np);
                CUDA_CHECK(cudaMemcpy(d_pw, pw_uniform.data(), np * sizeof(float), cudaMemcpyHostToDevice));
            }

            CUDA_CHECK(cudaDeviceSynchronize());

            auto t1 = std::chrono::high_resolution_clock::now();
            cuda_total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

            cuda_xEst[0] = h_xEst[0]; cuda_xEst[1] = h_xEst[1]; cuda_xEst[2] = h_xEst[2];
        }

        // Store histories
        hxTrue_x.push_back(xTrue[0]); hxTrue_y.push_back(xTrue[1]);
        hxDR_x.push_back(xDR[0]);     hxDR_y.push_back(xDR[1]);
        hxEst_cpu_x.push_back(cpu_xEst[0]); hxEst_cpu_y.push_back(cpu_xEst[1]);
        hxEst_cuda_x.push_back(cuda_xEst[0]); hxEst_cuda_y.push_back(cuda_xEst[1]);

        // ===================== Visualization =====================
        cv::Mat combined(COMBINED_H, COMBINED_W, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Mat left  = combined(cv::Rect(0, 0, PANEL_W, PANEL_H));
        cv::Mat right = combined(cv::Rect(PANEL_W, 0, PANEL_W, PANEL_H));

        // Draw both panels
        for (int panel = 0; panel < 2; panel++) {
            cv::Mat& bg = (panel == 0) ? left : right;
            std::vector<float>& hxEst_x = (panel == 0) ? hxEst_cpu_x : hxEst_cuda_x;
            std::vector<float>& hxEst_y = (panel == 0) ? hxEst_cpu_y : hxEst_cuda_y;

            // Trajectories
            for (unsigned int j = 0; j < hxTrue_x.size(); j++) {
                cv::circle(bg, cv_offset(hxTrue_x[j], hxTrue_y[j], PANEL_W, PANEL_H),
                           5, cv::Scalar(0, 255, 0), -1);
                cv::circle(bg, cv_offset(hxDR_x[j], hxDR_y[j], PANEL_W, PANEL_H),
                           5, cv::Scalar(0, 0, 0), -1);
                cv::circle(bg, cv_offset(hxEst_x[j], hxEst_y[j], PANEL_W, PANEL_H),
                           7, cv::Scalar(255, 0, 0), 3);
            }

            // True landmarks (purple)
            for (int i = 0; i < N_TRUE_LM; i++) {
                cv::circle(bg, cv_offset(RFID[i][0], RFID[i][1], PANEL_W, PANEL_H),
                           14, cv::Scalar(128, 0, 128), -1);
            }

            if (panel == 0) {
                // CPU: particles (red) and estimated landmarks (cyan)
                for (int i = 0; i < NP; i++) {
                    cv::circle(bg, cv_offset(cpu_particles[i].x, cpu_particles[i].y, PANEL_W, PANEL_H),
                               3, cv::Scalar(0, 0, 255), -1);
                }
                for (int i = 0; i < MAX_LANDMARKS; i++) {
                    if (cpu_lm_valid[i]) {
                        cv::circle(bg, cv_offset(cpu_lm_est[i * 2], cpu_lm_est[i * 2 + 1], PANEL_W, PANEL_H),
                                   10, cv::Scalar(255, 255, 0), -1);
                    }
                }
            } else {
                // CUDA: particles (red) and estimated landmarks (cyan)
                for (int i = 0; i < np; i++) {
                    cv::circle(bg, cv_offset(h_pose[i * 3], h_pose[i * 3 + 1], PANEL_W, PANEL_H),
                               3, cv::Scalar(0, 0, 255), -1);
                }
                for (int i = 0; i < MAX_LANDMARKS; i++) {
                    if (h_lm_est_valid[i]) {
                        cv::circle(bg, cv_offset(h_lm_est[i * 2], h_lm_est[i * 2 + 1], PANEL_W, PANEL_H),
                                   10, cv::Scalar(255, 255, 0), -1);
                    }
                }
            }
        }

        // Divider line
        cv::line(combined, cv::Point(PANEL_W, 0), cv::Point(PANEL_W, COMBINED_H),
                 cv::Scalar(100, 100, 100), 2);

        // Timing text
        double cpu_avg = cpu_total_ms / step_count;
        double cuda_avg = cuda_total_ms / step_count;

        char buf[128];
        snprintf(buf, sizeof(buf), "CPU FastSLAM 1.0");
        cv::putText(combined, buf, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 0), 2);
        snprintf(buf, sizeof(buf), "CPU: %.2f ms/step", cpu_avg);
        cv::putText(combined, buf, cv::Point(20, 85), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(200, 0, 0), 2);

        snprintf(buf, sizeof(buf), "CUDA FastSLAM 1.0");
        cv::putText(combined, buf, cv::Point(PANEL_W + 20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 0), 2);
        snprintf(buf, sizeof(buf), "CUDA: %.2f ms/step", cuda_avg);
        cv::putText(combined, buf, cv::Point(PANEL_W + 20, 85), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(200, 0, 0), 2);

        snprintf(buf, sizeof(buf), "t=%.1fs  particles=%d  landmarks=%d", time_val, NP, N_TRUE_LM);
        cv::putText(combined, buf, cv::Point(20, COMBINED_H - 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

        cv::imshow("comparison_fastslam", combined);
        video.write(combined);
        cv::waitKey(5);
    }

    video.release();
    cv::destroyAllWindows();

    printf("\nFinal timing: CPU=%.2f ms/step, CUDA=%.2f ms/step\n",
           cpu_total_ms / step_count, cuda_total_ms / step_count);

    // Convert to gif
    int ret = system("which ffmpeg > /dev/null 2>&1 && "
        "ffmpeg -y -i gif/comparison_fastslam.avi "
        "-vf \"fps=15,scale=700:-1:flags=lanczos\" "
        "-gifflags +transdiff "
        "gif/comparison_fastslam.gif 2>/dev/null && "
        "echo 'GIF saved' || echo 'ffmpeg not available, skipping gif'");
    (void)ret;

    // Cleanup
    cudaFree(d_pose); cudaFree(d_pose_tmp); cudaFree(d_pw);
    cudaFree(d_lm_mean); cudaFree(d_lm_mean_tmp);
    cudaFree(d_lm_cov); cudaFree(d_lm_cov_tmp);
    cudaFree(d_lm_seen); cudaFree(d_lm_seen_tmp);
    cudaFree(d_obs); cudaFree(d_rng_states);
    cudaFree(d_xEst); cudaFree(d_wcum);
    cudaFree(d_lm_est); cudaFree(d_lm_est_valid);

    return 0;
}
