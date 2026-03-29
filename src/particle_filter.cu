/*************************************************************************
    > File Name: particle_filter.cu
    > CUDA-parallelized Particle Filter
    > Based on original C++ implementation by TAI Lei
    > CUDA kernels for: particle prediction, weight update, resampling
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define SIM_TIME 50.0f
#define DT 0.1f
#define PI 3.141592653f
#define MAX_RANGE 20.0f
#define NP 1000        // increased from 100 to show GPU advantage
#define NTh (NP / 2)
#define MAX_LANDMARKS 16

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
// Device: observation stored as (distance, lx, ly)
// ---------------------------------------------------------------------------
struct Observation {
    float d;
    float lx;
    float ly;
};

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
// Kernel: predict particles + compute weights (fused)
//   px: [4 x NP] column-major  (x, y, yaw, v per particle)
//   pw: [NP]
// ---------------------------------------------------------------------------
__global__ void predict_and_weight_kernel(
    float* px,            // [4 * NP], column-major
    float* pw,            // [NP]
    const float u_v,
    const float u_omega,
    const float rsim_0,   // Rsim(0,0)
    const float rsim_1,   // Rsim(1,1)
    const Observation* obs,
    const int n_obs,
    const float Q,
    curandState* rng_states,
    const int np)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    curandState local_rng = rng_states[ip];

    // --- noisy control input ---
    float ud_v     = u_v     + curand_normal(&local_rng) * rsim_0;
    float ud_omega = u_omega + curand_normal(&local_rng) * rsim_1;

    // --- load particle state ---
    float x   = px[0 * np + ip];
    float y   = px[1 * np + ip];
    float yaw = px[2 * np + ip];
    float v   = px[3 * np + ip];

    // --- motion model: x_{t+1} = F*x + B*u ---
    x   += DT * cosf(yaw) * ud_v;
    y   += DT * sinf(yaw) * ud_v;
    yaw += DT * ud_omega;
    v    = ud_v;

    // --- store updated state ---
    px[0 * np + ip] = x;
    px[1 * np + ip] = y;
    px[2 * np + ip] = yaw;
    px[3 * np + ip] = v;

    // --- weight update ---
    float w = pw[ip];
    float sigma = sqrtf(Q);
    float inv_coeff = 1.0f / sqrtf(2.0f * PI * Q);

    for (int i = 0; i < n_obs; i++) {
        float dx = x - obs[i].lx;
        float dy = y - obs[i].ly;
        float prez = sqrtf(dx * dx + dy * dy);
        float dz = prez - obs[i].d;
        w *= inv_coeff * expf(-dz * dz / (2.0f * Q));
    }

    pw[ip] = w;
    rng_states[ip] = local_rng;
}

// ---------------------------------------------------------------------------
// Kernel: normalize weights (single-block reduction for simplicity)
// ---------------------------------------------------------------------------
__global__ void normalize_weights_kernel(float* pw, int np) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // sum reduction
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

    // normalize
    for (int i = tid; i < np; i += blockDim.x) {
        pw[i] /= total;
    }
}

// ---------------------------------------------------------------------------
// Kernel: compute weighted mean state  (single-block reduction)
// ---------------------------------------------------------------------------
__global__ void weighted_mean_kernel(const float* px, const float* pw,
                                     float* x_est, int np) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // 4 accumulators per thread
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = tid; i < np; i += blockDim.x) {
        float w = pw[i];
        s0 += px[0 * np + i] * w;
        s1 += px[1 * np + i] * w;
        s2 += px[2 * np + i] * w;
        s3 += px[3 * np + i] * w;
    }
    sdata[tid * 4 + 0] = s0;
    sdata[tid * 4 + 1] = s1;
    sdata[tid * 4 + 2] = s2;
    sdata[tid * 4 + 3] = s3;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid * 4 + 0] += sdata[(tid + s) * 4 + 0];
            sdata[tid * 4 + 1] += sdata[(tid + s) * 4 + 1];
            sdata[tid * 4 + 2] += sdata[(tid + s) * 4 + 2];
            sdata[tid * 4 + 3] += sdata[(tid + s) * 4 + 3];
        }
        __syncthreads();
    }

    if (tid == 0) {
        x_est[0] = sdata[0];
        x_est[1] = sdata[1];
        x_est[2] = sdata[2];
        x_est[3] = sdata[3];
    }
}

// ---------------------------------------------------------------------------
// Kernel: inclusive prefix sum for resampling (sequential, runs on 1 thread)
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
__global__ void resample_kernel(const float* px_in, float* px_out,
                                const float* wcum, float base_step,
                                float rand_offset, int np) {
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    float target = base_step * ip + rand_offset;

    // binary search in wcum
    int lo = 0, hi = np - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (wcum[mid] < target) lo = mid + 1;
        else hi = mid;
    }

    px_out[0 * np + ip] = px_in[0 * np + lo];
    px_out[1 * np + ip] = px_in[1 * np + lo];
    px_out[2 * np + ip] = px_in[2 * np + lo];
    px_out[3 * np + ip] = px_in[3 * np + lo];
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

void ellipse_drawing(cv::Mat bg_img, Eigen::Matrix2f pest,
                     Eigen::Vector2f center,
                     cv::Scalar ellipse_color = {0, 0, 255}) {
    Eigen::EigenSolver<Eigen::Matrix2f> ces(pest);
    Eigen::Matrix2f e_value  = ces.pseudoEigenvalueMatrix();
    Eigen::Matrix2f e_vector = ces.pseudoEigenvectors();

    double angle = std::atan2(e_vector(0, 1), e_vector(0, 0));
    cv::ellipse(bg_img,
                cv_offset(center(0), center(1), bg_img.cols, bg_img.rows),
                cv::Size(e_value(0, 0) * 1000, e_value(1, 1) * 1000),
                angle / PI * 180, 0, 360, ellipse_color, 2, 4);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    // ------------------------------------------
    // Simulation parameters (same as original)
    // ------------------------------------------
    float time_val = 0.0f;

    Eigen::Vector2f u;
    u << 1.0f, 0.1f;

    Eigen::Matrix<float, 4, 2> RFID;
    RFID << 10.0f,  0.0f,
            10.0f, 10.0f,
             0.0f, 15.0f,
            -5.0f, 20.0f;

    Eigen::Vector4f xDR   = Eigen::Vector4f::Zero();
    Eigen::Vector4f xTrue = Eigen::Vector4f::Zero();
    Eigen::Vector4f xEst  = Eigen::Vector4f::Zero();

    std::vector<Eigen::Vector4f> hxDR, hxTrue, hxEst;
    Eigen::Matrix4f PEst = Eigen::Matrix4f::Identity();

    float Q = 0.01f;

    Eigen::Matrix2f Rsim = Eigen::Matrix2f::Identity();
    Rsim(0, 0) = 1.0f;
    Rsim(1, 1) = (30.0f / 180.0f * PI) * (30.0f / 180.0f * PI);

    float Qsim = 0.04f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> gaussian_d(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni_d(0.0f, 1.0f);

    // ------------------------------------------
    // CUDA memory allocation
    // ------------------------------------------
    const int np = NP;
    const int threads = 256;
    const int blocks  = (np + threads - 1) / threads;

    // device particle states [4 * NP] column-major
    float *d_px, *d_px_tmp;
    CUDA_CHECK(cudaMalloc(&d_px,     4 * np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_px_tmp, 4 * np * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_px, 0,   4 * np * sizeof(float)));

    // device weights [NP]
    float *d_pw;
    CUDA_CHECK(cudaMalloc(&d_pw, np * sizeof(float)));
    {
        std::vector<float> pw_init(np, 1.0f / np);
        CUDA_CHECK(cudaMemcpy(d_pw, pw_init.data(), np * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // device observations
    Observation *d_obs;
    CUDA_CHECK(cudaMalloc(&d_obs, MAX_LANDMARKS * sizeof(Observation)));

    // cuRAND states
    curandState *d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, np * sizeof(curandState)));
    init_curand_kernel<<<blocks, threads>>>(d_rng_states, 42ULL, np);
    CUDA_CHECK(cudaDeviceSynchronize());

    // device estimated state
    float *d_xEst;
    CUDA_CHECK(cudaMalloc(&d_xEst, 4 * sizeof(float)));

    // device cumulative sum for resampling
    float *d_wcum;
    CUDA_CHECK(cudaMalloc(&d_wcum, np * sizeof(float)));

    // host buffers for readback
    std::vector<float> h_px(4 * np);
    std::vector<float> h_pw(np);
    float h_xEst[4];

    // ------------------------------------------
    // Visualization
    // ------------------------------------------
    cv::namedWindow("pf", cv::WINDOW_NORMAL);
    int count = 0;

    std::cout << "Particle Filter with CUDA (" << NP << " particles)" << std::endl;

    while (time_val <= SIM_TIME) {
        time_val += DT;

        // --- noisy input for dead reckoning ---
        Eigen::Vector2f ud;
        ud(0) = u(0) + gaussian_d(gen) * Rsim(0, 0);
        ud(1) = u(1) + gaussian_d(gen) * Rsim(1, 1);

        xTrue = [&]() {
            Eigen::Vector4f x = xTrue;
            x(0) += DT * std::cos(x(2)) * u(0);
            x(1) += DT * std::sin(x(2)) * u(0);
            x(2) += DT * u(1);
            x(3)  = u(0);
            return x;
        }();

        xDR = [&]() {
            Eigen::Vector4f x = xDR;
            x(0) += DT * std::cos(x(2)) * ud(0);
            x(1) += DT * std::sin(x(2)) * ud(0);
            x(2) += DT * ud(1);
            x(3)  = ud(0);
            return x;
        }();

        // --- generate observations ---
        std::vector<Observation> z_host;
        for (int i = 0; i < RFID.rows(); i++) {
            float dx = xTrue(0) - RFID(i, 0);
            float dy = xTrue(1) - RFID(i, 1);
            float d  = std::sqrt(dx * dx + dy * dy);
            if (d <= MAX_RANGE) {
                Observation ob;
                ob.d  = d + gaussian_d(gen) * Qsim;
                ob.lx = RFID(i, 0);
                ob.ly = RFID(i, 1);
                z_host.push_back(ob);
            }
        }

        int n_obs = (int)z_host.size();
        if (n_obs > 0) {
            CUDA_CHECK(cudaMemcpy(d_obs, z_host.data(),
                                  n_obs * sizeof(Observation),
                                  cudaMemcpyHostToDevice));
        }

        // --- GPU: predict + weight ---
        predict_and_weight_kernel<<<blocks, threads>>>(
            d_px, d_pw,
            u(0), u(1),
            Rsim(0, 0), Rsim(1, 1),
            d_obs, n_obs, Q,
            d_rng_states, np);

        // --- GPU: normalize weights ---
        normalize_weights_kernel<<<1, threads, threads * sizeof(float)>>>(
            d_pw, np);

        // --- GPU: weighted mean ---
        weighted_mean_kernel<<<1, threads, threads * 4 * sizeof(float)>>>(
            d_px, d_pw, d_xEst, np);

        CUDA_CHECK(cudaMemcpy(h_xEst, d_xEst, 4 * sizeof(float),
                              cudaMemcpyDeviceToHost));

        xEst << h_xEst[0], h_xEst[1], h_xEst[2], h_xEst[3];

        // --- covariance (host-side, lightweight) ---
        CUDA_CHECK(cudaMemcpy(h_px.data(), d_px, 4 * np * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pw.data(), d_pw, np * sizeof(float),
                              cudaMemcpyDeviceToHost));

        PEst = Eigen::Matrix4f::Zero();
        for (int i = 0; i < np; i++) {
            Eigen::Vector4f dx;
            dx << h_px[0 * np + i] - xEst(0),
                  h_px[1 * np + i] - xEst(1),
                  h_px[2 * np + i] - xEst(2),
                  h_px[3 * np + i] - xEst(3);
            PEst += h_pw[i] * dx * dx.transpose();
        }

        // --- GPU: resampling ---
        float Neff_denom = 0.0f;
        for (int i = 0; i < np; i++) Neff_denom += h_pw[i] * h_pw[i];
        float Neff = 1.0f / Neff_denom;

        if (Neff < NTh) {
            cumsum_kernel<<<1, 1>>>(d_pw, d_wcum, np);

            float rand_offset = uni_d(gen) / np;
            float base_step   = 1.0f / np;

            resample_kernel<<<blocks, threads>>>(
                d_px, d_px_tmp, d_wcum, base_step, rand_offset, np);

            // swap buffers
            CUDA_CHECK(cudaMemcpy(d_px, d_px_tmp, 4 * np * sizeof(float),
                                  cudaMemcpyDeviceToDevice));

            // reset weights
            std::vector<float> pw_uniform(np, 1.0f / np);
            CUDA_CHECK(cudaMemcpy(d_pw, pw_uniform.data(),
                                  np * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // --- visualization ---
        hxDR.push_back(xDR);
        hxTrue.push_back(xTrue);
        hxEst.push_back(xEst);

        cv::Mat bg(3500, 3500, CV_8UC3, cv::Scalar(255, 255, 255));

        for (unsigned int j = 0; j < hxDR.size(); j++) {
            cv::circle(bg,
                       cv_offset(hxTrue[j](0), hxTrue[j](1), bg.cols, bg.rows),
                       7, cv::Scalar(0, 255, 0), -1);
            cv::circle(bg,
                       cv_offset(hxEst[j](0), hxEst[j](1), bg.cols, bg.rows),
                       10, cv::Scalar(255, 0, 0), 5);
            cv::circle(bg,
                       cv_offset(hxDR[j](0), hxDR[j](1), bg.cols, bg.rows),
                       7, cv::Scalar(0, 0, 0), -1);
        }

        // draw particles (read back already done above)
        for (int j = 0; j < np; j++) {
            cv::circle(
                bg,
                cv_offset(h_px[0 * np + j], h_px[1 * np + j], bg.cols, bg.rows),
                3, cv::Scalar(0, 0, 255), -1);
        }

        for (int i = 0; i < RFID.rows(); i++) {
            cv::circle(
                bg,
                cv_offset(RFID(i, 0), RFID(i, 1), bg.cols, bg.rows),
                20, cv::Scalar(127, 0, 255), -1);
        }

        for (unsigned int i = 0; i < z_host.size(); i++) {
            cv::line(
                bg,
                cv_offset(z_host[i].lx, z_host[i].ly, bg.cols, bg.rows),
                cv_offset(hxEst.back()(0), hxEst.back()(1), bg.cols, bg.rows),
                cv::Scalar(0, 0, 0), 5);
        }

        ellipse_drawing(bg, PEst.block(0, 0, 2, 2), xEst.head(2));

        cv::imshow("pf", bg);
        cv::waitKey(5);
        count++;
    }

    // --- cleanup ---
    cudaFree(d_px);
    cudaFree(d_px_tmp);
    cudaFree(d_pw);
    cudaFree(d_obs);
    cudaFree(d_rng_states);
    cudaFree(d_xEst);
    cudaFree(d_wcum);

    return 0;
}
