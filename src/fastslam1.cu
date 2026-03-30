/*************************************************************************
    > File Name: fastslam1.cu
    > CUDA-parallelized FastSLAM 1.0
    > Based on PythonRobotics FastSLAM1 by Atsushi Sakai
    > Combines particle filter for pose estimation with per-particle EKF
    > for landmark position estimation.
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
#define SIM_TIME 50.0f
#define DT 0.1f
#define PI 3.141592653f
#define MAX_RANGE 20.0f
#define N_PARTICLES 100
#define NTh (N_PARTICLES / 2)
#define MAX_LANDMARKS 20

// Noise parameters
#define SIGMA_V 0.5f
#define SIGMA_OMEGA 0.1f
#define SIGMA_RANGE 0.5f
#define SIGMA_BEARING 0.1f

// Observation noise covariance Q_obs (2x2 diagonal)
#define Q_RANGE  (SIGMA_RANGE * SIGMA_RANGE)
#define Q_BEARING (SIGMA_BEARING * SIGMA_BEARING)

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
// Observation: range and bearing to a landmark with known ID
// ---------------------------------------------------------------------------
struct Observation {
    float range;
    float bearing;
    int lm_id;  // landmark index
};

// ---------------------------------------------------------------------------
// Device helper: normalize angle to [-pi, pi]
// ---------------------------------------------------------------------------
__device__ float normalize_angle(float a) {
    a = fmodf(a + PI, 2.0f * PI);
    if (a < 0.0f) a += 2.0f * PI;
    return a - PI;
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
// Kernel: predict particles using motion model + noise
//   pose: [N_PARTICLES * 3] = {x, y, yaw} per particle
// ---------------------------------------------------------------------------
__global__ void predict_particles_kernel(
    float* pose,          // [N_PARTICLES * 3]: x, y, yaw
    float u_v,
    float u_omega,
    curandState* rng_states,
    int np)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    curandState local_rng = rng_states[ip];

    // Add noise to control input
    float noisy_v     = u_v     + curand_normal(&local_rng) * SIGMA_V;
    float noisy_omega = u_omega + curand_normal(&local_rng) * SIGMA_OMEGA;

    float x   = pose[ip * 3 + 0];
    float y   = pose[ip * 3 + 1];
    float yaw = pose[ip * 3 + 2];

    // Motion model
    x   += DT * cosf(yaw) * noisy_v;
    y   += DT * sinf(yaw) * noisy_v;
    yaw += DT * noisy_omega;
    yaw  = normalize_angle(yaw);

    pose[ip * 3 + 0] = x;
    pose[ip * 3 + 1] = y;
    pose[ip * 3 + 2] = yaw;

    rng_states[ip] = local_rng;
}

// ---------------------------------------------------------------------------
// Kernel: update landmarks via EKF and compute weights
//   Each thread handles one particle, iterating over all observations.
//   For each observation:
//     - If landmark not yet seen: initialize
//     - If landmark seen: EKF update and accumulate weight
//
//   landmark_mean: [np * MAX_LANDMARKS * 2]
//   landmark_cov:  [np * MAX_LANDMARKS * 4] (2x2 flattened row-major)
//   landmark_seen: [np * MAX_LANDMARKS]
// ---------------------------------------------------------------------------
__global__ void update_landmarks_kernel(
    float* pose,            // [np * 3]
    float* weights,         // [np]
    float* landmark_mean,   // [np * MAX_LANDMARKS * 2]
    float* landmark_cov,    // [np * MAX_LANDMARKS * 4]
    int*   landmark_seen,   // [np * MAX_LANDMARKS]
    const Observation* obs,
    int n_obs,
    int np)
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
            // Initialize landmark from observation
            float lm_x = px + z_range * cosf(z_bearing + pyaw);
            float lm_y = py + z_range * sinf(z_bearing + pyaw);

            landmark_mean[mean_base + 0] = lm_x;
            landmark_mean[mean_base + 1] = lm_y;

            // Initial covariance: large uncertainty
            // Use inverse of Jacobian to map observation noise to world frame
            // For simplicity, use a reasonably large initial covariance
            float cs = cosf(z_bearing + pyaw);
            float sn = sinf(z_bearing + pyaw);

            // Jacobian of observation inverse: d(lm_x, lm_y)/d(range, bearing)
            // G = [cos(b+yaw),  -r*sin(b+yaw)]
            //     [sin(b+yaw),   r*cos(b+yaw)]
            // P0 = G * Q_obs * G^T
            float g00 = cs;
            float g01 = -z_range * sn;
            float g10 = sn;
            float g11 =  z_range * cs;

            float p00 = g00 * g00 * Q_RANGE + g01 * g01 * Q_BEARING;
            float p01 = g00 * g10 * Q_RANGE + g01 * g11 * Q_BEARING;
            float p10 = p01;
            float p11 = g10 * g10 * Q_RANGE + g11 * g11 * Q_BEARING;

            landmark_cov[cov_base + 0] = p00;
            landmark_cov[cov_base + 1] = p01;
            landmark_cov[cov_base + 2] = p10;
            landmark_cov[cov_base + 3] = p11;

            landmark_seen[seen_idx] = 1;
        } else {
            // EKF update for known landmark
            float mu_x = landmark_mean[mean_base + 0];
            float mu_y = landmark_mean[mean_base + 1];

            float P00 = landmark_cov[cov_base + 0];
            float P01 = landmark_cov[cov_base + 1];
            float P10 = landmark_cov[cov_base + 2];
            float P11 = landmark_cov[cov_base + 3];

            // Predicted observation
            float dx = mu_x - px;
            float dy = mu_y - py;
            float q = dx * dx + dy * dy;
            float sq = sqrtf(q);

            float z_range_pred   = sq;
            float z_bearing_pred = atan2f(dy, dx) - pyaw;

            // Innovation
            float innov_r = z_range - z_range_pred;
            float innov_b = normalize_angle(z_bearing - z_bearing_pred);

            // Jacobian H: d(range, bearing)/d(lm_x, lm_y)
            // H = [ dx/sqrt(q),   dy/sqrt(q)  ]
            //     [ -dy/q,        dx/q         ]
            float H00 = dx / sq;
            float H01 = dy / sq;
            float H10 = -dy / q;
            float H11 = dx / q;

            // S = H * P * H^T + Q_obs
            // First compute HP = H * P
            float HP00 = H00 * P00 + H01 * P10;
            float HP01 = H00 * P01 + H01 * P11;
            float HP10 = H10 * P00 + H11 * P10;
            float HP11 = H10 * P01 + H11 * P11;

            // S = HP * H^T + Q
            float S00 = HP00 * H00 + HP01 * H01 + Q_RANGE;
            float S01 = HP00 * H10 + HP01 * H11;
            float S10 = HP10 * H00 + HP11 * H01;
            float S11 = HP10 * H10 + HP11 * H11 + Q_BEARING;

            // S^{-1} (2x2 inverse)
            float det_S = S00 * S11 - S01 * S10;
            if (fabsf(det_S) < 1e-10f) det_S = 1e-10f;
            float inv_det = 1.0f / det_S;

            float Si00 =  S11 * inv_det;
            float Si01 = -S01 * inv_det;
            float Si10 = -S10 * inv_det;
            float Si11 =  S00 * inv_det;

            // Kalman gain K = P * H^T * S^{-1}
            // First P * H^T
            float PHt00 = P00 * H00 + P01 * H01;
            float PHt01 = P00 * H10 + P01 * H11;
            float PHt10 = P10 * H00 + P11 * H01;
            float PHt11 = P10 * H10 + P11 * H11;

            // K = PHt * Si
            float K00 = PHt00 * Si00 + PHt01 * Si10;
            float K01 = PHt00 * Si01 + PHt01 * Si11;
            float K10 = PHt10 * Si00 + PHt11 * Si10;
            float K11 = PHt10 * Si01 + PHt11 * Si11;

            // Update mean
            landmark_mean[mean_base + 0] = mu_x + K00 * innov_r + K01 * innov_b;
            landmark_mean[mean_base + 1] = mu_y + K10 * innov_r + K11 * innov_b;

            // Update covariance: P = (I - K*H) * P
            float KH00 = K00 * H00 + K01 * H10;
            float KH01 = K00 * H01 + K01 * H11;
            float KH10 = K10 * H00 + K11 * H10;
            float KH11 = K10 * H01 + K11 * H11;

            float IKH00 = 1.0f - KH00;
            float IKH01 = -KH01;
            float IKH10 = -KH10;
            float IKH11 = 1.0f - KH11;

            landmark_cov[cov_base + 0] = IKH00 * P00 + IKH01 * P10;
            landmark_cov[cov_base + 1] = IKH00 * P01 + IKH01 * P11;
            landmark_cov[cov_base + 2] = IKH10 * P00 + IKH11 * P10;
            landmark_cov[cov_base + 3] = IKH10 * P01 + IKH11 * P11;

            // Weight contribution: det(2*pi*S)^{-0.5} * exp(-0.5 * innov^T * S^{-1} * innov)
            float det_2piS = (2.0f * PI) * (2.0f * PI) * fabsf(det_S);
            float inv_sqrt_det = 1.0f / sqrtf(det_2piS + 1e-30f);

            float maha = innov_r * (Si00 * innov_r + Si01 * innov_b)
                       + innov_b * (Si10 * innov_r + Si11 * innov_b);

            w *= inv_sqrt_det * expf(-0.5f * maha);
        }
    }

    weights[ip] = w;
}

// ---------------------------------------------------------------------------
// Kernel: normalize weights (single-block reduction)
// ---------------------------------------------------------------------------
__global__ void normalize_weights_kernel(float* pw, int np) {
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

    float total = sdata[0];
    if (total < 1e-30f) total = 1e-30f;

    for (int i = tid; i < np; i += blockDim.x) {
        pw[i] /= total;
    }
}

// ---------------------------------------------------------------------------
// Kernel: compute weighted mean pose (single-block reduction)
// ---------------------------------------------------------------------------
__global__ void weighted_mean_kernel(const float* pose, const float* pw,
                                     float* x_est, int np) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
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

    if (tid == 0) {
        x_est[0] = sdata[0];
        x_est[1] = sdata[1];
        x_est[2] = sdata[2];
    }
}

// ---------------------------------------------------------------------------
// Kernel: inclusive prefix sum for resampling (sequential, 1 thread)
// ---------------------------------------------------------------------------
__global__ void cumsum_kernel(const float* pw, float* wcum, int np) {
    wcum[0] = pw[0];
    for (int i = 1; i < np; i++) {
        wcum[i] = wcum[i - 1] + pw[i];
    }
}

// ---------------------------------------------------------------------------
// Kernel: systematic resampling (pose + landmarks)
// ---------------------------------------------------------------------------
__global__ void resample_kernel(
    const float* pose_in,
    float* pose_out,
    const float* lm_mean_in,
    float* lm_mean_out,
    const float* lm_cov_in,
    float* lm_cov_out,
    const int* lm_seen_in,
    int* lm_seen_out,
    const float* wcum,
    float base_step,
    float rand_offset,
    int np)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    float target = base_step * ip + rand_offset;

    // Binary search in wcum
    int lo = 0, hi = np - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (wcum[mid] < target) lo = mid + 1;
        else hi = mid;
    }

    // Copy pose
    pose_out[ip * 3 + 0] = pose_in[lo * 3 + 0];
    pose_out[ip * 3 + 1] = pose_in[lo * 3 + 1];
    pose_out[ip * 3 + 2] = pose_in[lo * 3 + 2];

    // Copy all landmarks for this particle
    int src_mean = lo * MAX_LANDMARKS * 2;
    int dst_mean = ip * MAX_LANDMARKS * 2;
    int src_cov  = lo * MAX_LANDMARKS * 4;
    int dst_cov  = ip * MAX_LANDMARKS * 4;
    int src_seen = lo * MAX_LANDMARKS;
    int dst_seen = ip * MAX_LANDMARKS;

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

// ---------------------------------------------------------------------------
// Kernel: compute weighted mean landmark positions (for visualization)
// ---------------------------------------------------------------------------
__global__ void weighted_mean_landmarks_kernel(
    const float* pw,
    const float* lm_mean,
    const int* lm_seen,
    float* lm_est,        // [MAX_LANDMARKS * 2]
    int* lm_est_valid,    // [MAX_LANDMARKS]
    int np)
{
    int lm_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (lm_id >= MAX_LANDMARKS) return;

    float sum_x = 0.0f, sum_y = 0.0f;
    float sum_w = 0.0f;

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

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
cv::Point2i cv_offset(float ex, float ey,
                      int image_width = 2000, int image_height = 2000) {
    cv::Point2i output;
    output.x = int(ex * 100) + image_width / 2;
    output.y = image_height - int(ey * 100) - image_height / 3;
    return output;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    // ------------------------------------------
    // Simulation parameters
    // ------------------------------------------
    float time_val = 0.0f;

    // Control input: velocity, yaw rate
    float u_v = 1.0f;
    float u_omega = 0.1f;

    // RFID landmarks (true positions)
    const int N_TRUE_LM = 8;
    float RFID[N_TRUE_LM][2] = {
        {10.0f,  0.0f},
        {15.0f, 10.0f},
        {15.0f, 15.0f},
        {10.0f, 20.0f},
        { 3.0f, 15.0f},
        {-5.0f, 20.0f},
        {-5.0f,  5.0f},
        { 0.0f, 10.0f}
    };

    // Ground truth and dead reckoning states
    float xTrue[3] = {0.0f, 0.0f, 0.0f};  // x, y, yaw
    float xDR[3]   = {0.0f, 0.0f, 0.0f};
    float xEst[3]  = {0.0f, 0.0f, 0.0f};

    std::vector<float> hxTrue_x, hxTrue_y;
    std::vector<float> hxDR_x, hxDR_y;
    std::vector<float> hxEst_x, hxEst_y;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> gaussian_d(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni_d(0.0f, 1.0f);

    // ------------------------------------------
    // CUDA memory allocation
    // ------------------------------------------
    const int np = N_PARTICLES;
    const int threads = 256;
    const int blocks = (np + threads - 1) / threads;

    // Particle poses [np * 3]
    float *d_pose, *d_pose_tmp;
    CUDA_CHECK(cudaMalloc(&d_pose,     np * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pose_tmp, np * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pose, 0,   np * 3 * sizeof(float)));

    // Particle weights [np]
    float *d_pw;
    CUDA_CHECK(cudaMalloc(&d_pw, np * sizeof(float)));
    {
        std::vector<float> pw_init(np, 1.0f / np);
        CUDA_CHECK(cudaMemcpy(d_pw, pw_init.data(), np * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Landmark means [np * MAX_LANDMARKS * 2]
    float *d_lm_mean, *d_lm_mean_tmp;
    CUDA_CHECK(cudaMalloc(&d_lm_mean,     np * MAX_LANDMARKS * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_mean_tmp, np * MAX_LANDMARKS * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_lm_mean, 0,   np * MAX_LANDMARKS * 2 * sizeof(float)));

    // Landmark covariances [np * MAX_LANDMARKS * 4]
    float *d_lm_cov, *d_lm_cov_tmp;
    CUDA_CHECK(cudaMalloc(&d_lm_cov,     np * MAX_LANDMARKS * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_cov_tmp, np * MAX_LANDMARKS * 4 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_lm_cov, 0,   np * MAX_LANDMARKS * 4 * sizeof(float)));

    // Landmark seen flags [np * MAX_LANDMARKS]
    int *d_lm_seen, *d_lm_seen_tmp;
    CUDA_CHECK(cudaMalloc(&d_lm_seen,     np * MAX_LANDMARKS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lm_seen_tmp, np * MAX_LANDMARKS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_lm_seen, 0,   np * MAX_LANDMARKS * sizeof(int)));

    // Observations
    Observation *d_obs;
    CUDA_CHECK(cudaMalloc(&d_obs, MAX_LANDMARKS * sizeof(Observation)));

    // cuRAND states
    curandState *d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, np * sizeof(curandState)));
    init_curand_kernel<<<blocks, threads>>>(d_rng_states, 42ULL, np);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Estimated pose
    float *d_xEst;
    CUDA_CHECK(cudaMalloc(&d_xEst, 3 * sizeof(float)));

    // Cumulative sum for resampling
    float *d_wcum;
    CUDA_CHECK(cudaMalloc(&d_wcum, np * sizeof(float)));

    // Estimated landmark positions (for visualization)
    float *d_lm_est;
    int *d_lm_est_valid;
    CUDA_CHECK(cudaMalloc(&d_lm_est, MAX_LANDMARKS * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_est_valid, MAX_LANDMARKS * sizeof(int)));

    // Host buffers
    std::vector<float> h_pose(np * 3);
    std::vector<float> h_pw(np);
    float h_xEst[3];
    float h_lm_est[MAX_LANDMARKS * 2];
    int h_lm_est_valid[MAX_LANDMARKS];

    // ------------------------------------------
    // Visualization
    // ------------------------------------------
    cv::namedWindow("fastslam1", cv::WINDOW_NORMAL);
    cv::VideoWriter video("gif/fastslam1.avi", cv::VideoWriter::fourcc('X','V','I','D'), 30, cv::Size(3500, 3500));

    std::cout << "FastSLAM 1.0 with CUDA (" << N_PARTICLES << " particles, "
              << N_TRUE_LM << " landmarks)" << std::endl;

    while (time_val <= SIM_TIME) {
        time_val += DT;

        // --- Ground truth update ---
        xTrue[0] += DT * std::cos(xTrue[2]) * u_v;
        xTrue[1] += DT * std::sin(xTrue[2]) * u_v;
        xTrue[2] += DT * u_omega;

        // --- Dead reckoning update (with noise) ---
        float noisy_v = u_v + gaussian_d(gen) * SIGMA_V;
        float noisy_omega = u_omega + gaussian_d(gen) * SIGMA_OMEGA;
        xDR[0] += DT * std::cos(xDR[2]) * noisy_v;
        xDR[1] += DT * std::sin(xDR[2]) * noisy_v;
        xDR[2] += DT * noisy_omega;

        // --- Generate observations ---
        std::vector<Observation> z_host;
        for (int i = 0; i < N_TRUE_LM; i++) {
            float dx = xTrue[0] - RFID[i][0];
            float dy = xTrue[1] - RFID[i][1];
            float d = std::sqrt(dx * dx + dy * dy);
            if (d <= MAX_RANGE) {
                Observation ob;
                ob.range   = d + gaussian_d(gen) * SIGMA_RANGE;
                ob.bearing = std::atan2(dy, dx) - xTrue[2] + gaussian_d(gen) * SIGMA_BEARING;
                ob.lm_id   = i;
                z_host.push_back(ob);
            }
        }

        int n_obs = (int)z_host.size();
        if (n_obs > 0) {
            CUDA_CHECK(cudaMemcpy(d_obs, z_host.data(),
                                  n_obs * sizeof(Observation),
                                  cudaMemcpyHostToDevice));
        }

        // --- GPU: predict particles ---
        predict_particles_kernel<<<blocks, threads>>>(
            d_pose, u_v, u_omega, d_rng_states, np);

        // --- GPU: update landmarks and compute weights ---
        if (n_obs > 0) {
            update_landmarks_kernel<<<blocks, threads>>>(
                d_pose, d_pw, d_lm_mean, d_lm_cov, d_lm_seen,
                d_obs, n_obs, np);

            // --- GPU: normalize weights ---
            normalize_weights_kernel<<<1, threads, threads * sizeof(float)>>>(
                d_pw, np);
        }

        // --- GPU: weighted mean pose ---
        weighted_mean_kernel<<<1, threads, threads * 3 * sizeof(float)>>>(
            d_pose, d_pw, d_xEst, np);

        CUDA_CHECK(cudaMemcpy(h_xEst, d_xEst, 3 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        xEst[0] = h_xEst[0];
        xEst[1] = h_xEst[1];
        xEst[2] = h_xEst[2];

        // --- GPU: weighted mean landmarks ---
        int lm_blocks = (MAX_LANDMARKS + threads - 1) / threads;
        weighted_mean_landmarks_kernel<<<lm_blocks, threads>>>(
            d_pw, d_lm_mean, d_lm_seen, d_lm_est, d_lm_est_valid, np);

        CUDA_CHECK(cudaMemcpy(h_lm_est, d_lm_est, MAX_LANDMARKS * 2 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_lm_est_valid, d_lm_est_valid, MAX_LANDMARKS * sizeof(int),
                              cudaMemcpyDeviceToHost));

        // --- Read back particles for visualization and resampling check ---
        CUDA_CHECK(cudaMemcpy(h_pw.data(), d_pw, np * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pose.data(), d_pose, np * 3 * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // --- Resampling ---
        float Neff_denom = 0.0f;
        for (int i = 0; i < np; i++) Neff_denom += h_pw[i] * h_pw[i];
        float Neff = 1.0f / (Neff_denom + 1e-30f);

        if (Neff < NTh) {
            cumsum_kernel<<<1, 1>>>(d_pw, d_wcum, np);

            float rand_offset = uni_d(gen) / np;
            float base_step = 1.0f / np;

            resample_kernel<<<blocks, threads>>>(
                d_pose, d_pose_tmp,
                d_lm_mean, d_lm_mean_tmp,
                d_lm_cov, d_lm_cov_tmp,
                d_lm_seen, d_lm_seen_tmp,
                d_wcum, base_step, rand_offset, np);

            // Swap buffers
            CUDA_CHECK(cudaMemcpy(d_pose, d_pose_tmp, np * 3 * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_lm_mean, d_lm_mean_tmp, np * MAX_LANDMARKS * 2 * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_lm_cov, d_lm_cov_tmp, np * MAX_LANDMARKS * 4 * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_lm_seen, d_lm_seen_tmp, np * MAX_LANDMARKS * sizeof(int),
                                  cudaMemcpyDeviceToDevice));

            // Reset weights
            std::vector<float> pw_uniform(np, 1.0f / np);
            CUDA_CHECK(cudaMemcpy(d_pw, pw_uniform.data(), np * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Visualization ---
        hxTrue_x.push_back(xTrue[0]); hxTrue_y.push_back(xTrue[1]);
        hxDR_x.push_back(xDR[0]);     hxDR_y.push_back(xDR[1]);
        hxEst_x.push_back(xEst[0]);   hxEst_y.push_back(xEst[1]);

        cv::Mat bg(3500, 3500, CV_8UC3, cv::Scalar(255, 255, 255));

        // Draw trajectories
        for (unsigned int j = 0; j < hxTrue_x.size(); j++) {
            // Green: ground truth
            cv::circle(bg,
                       cv_offset(hxTrue_x[j], hxTrue_y[j], bg.cols, bg.rows),
                       7, cv::Scalar(0, 255, 0), -1);
            // Blue: estimated
            cv::circle(bg,
                       cv_offset(hxEst_x[j], hxEst_y[j], bg.cols, bg.rows),
                       10, cv::Scalar(255, 0, 0), 5);
            // Black: dead reckoning
            cv::circle(bg,
                       cv_offset(hxDR_x[j], hxDR_y[j], bg.cols, bg.rows),
                       7, cv::Scalar(0, 0, 0), -1);
        }

        // Red dots: particle positions
        for (int j = 0; j < np; j++) {
            cv::circle(bg,
                       cv_offset(h_pose[j * 3 + 0], h_pose[j * 3 + 1], bg.cols, bg.rows),
                       3, cv::Scalar(0, 0, 255), -1);
        }

        // Purple: true landmark positions
        for (int i = 0; i < N_TRUE_LM; i++) {
            cv::circle(bg,
                       cv_offset(RFID[i][0], RFID[i][1], bg.cols, bg.rows),
                       20, cv::Scalar(128, 0, 128), -1);
        }

        // Cyan: estimated landmark positions
        for (int i = 0; i < MAX_LANDMARKS; i++) {
            if (h_lm_est_valid[i]) {
                cv::circle(bg,
                           cv_offset(h_lm_est[i * 2 + 0], h_lm_est[i * 2 + 1], bg.cols, bg.rows),
                           15, cv::Scalar(255, 255, 0), -1);
            }
        }

        cv::imshow("fastslam1", bg);
        video.write(bg);
        cv::waitKey(5);
    }

    video.release();
    std::cout << "Video saved to videos/fastslam1.avi" << std::endl;

    // --- Cleanup ---
    cudaFree(d_pose);
    cudaFree(d_pose_tmp);
    cudaFree(d_pw);
    cudaFree(d_lm_mean);
    cudaFree(d_lm_mean_tmp);
    cudaFree(d_lm_cov);
    cudaFree(d_lm_cov_tmp);
    cudaFree(d_lm_seen);
    cudaFree(d_lm_seen_tmp);
    cudaFree(d_obs);
    cudaFree(d_rng_states);
    cudaFree(d_xEst);
    cudaFree(d_wcum);
    cudaFree(d_lm_est);
    cudaFree(d_lm_est_valid);

    return 0;
}
