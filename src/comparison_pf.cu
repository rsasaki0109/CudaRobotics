/*************************************************************************
    Particle Filter: CPU vs CUDA side-by-side comparison GIF generator
    Left panel: CPU (Eigen-based), Right panel: CUDA (GPU kernels)
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <chrono>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define SIM_TIME 20.0f
#define DT 0.1f
#define PI 3.141592653f
#define MAX_RANGE 20.0f
#define NP 1000
#define NTh (NP / 2)
#define MAX_LANDMARKS 16

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Observation structure (shared by CPU and GPU)
// ---------------------------------------------------------------------------
struct Observation {
    float d;
    float lx;
    float ly;
};

// ===========================================================================
// CUDA Kernels
// ===========================================================================

__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void predict_and_weight_kernel(
    float* px, float* pw,
    const float u_v, const float u_omega,
    const float rsim_0, const float rsim_1,
    const Observation* obs, const int n_obs,
    const float Q,
    curandState* rng_states, const int np)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    curandState local_rng = rng_states[ip];

    float ud_v     = u_v     + curand_normal(&local_rng) * rsim_0;
    float ud_omega = u_omega + curand_normal(&local_rng) * rsim_1;

    float x   = px[0 * np + ip];
    float y   = px[1 * np + ip];
    float yaw = px[2 * np + ip];
    float v   = px[3 * np + ip];

    x   += DT * cosf(yaw) * ud_v;
    y   += DT * sinf(yaw) * ud_v;
    yaw += DT * ud_omega;
    v    = ud_v;

    px[0 * np + ip] = x;
    px[1 * np + ip] = y;
    px[2 * np + ip] = yaw;
    px[3 * np + ip] = v;

    float w = pw[ip];
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

__global__ void weighted_mean_kernel(const float* px, const float* pw,
                                     float* x_est, int np) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

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
        x_est[0] = sdata[0]; x_est[1] = sdata[1];
        x_est[2] = sdata[2]; x_est[3] = sdata[3];
    }
}

__global__ void cumsum_kernel(const float* pw, float* wcum, int np) {
    wcum[0] = pw[0];
    for (int i = 1; i < np; i++) wcum[i] = wcum[i - 1] + pw[i];
}

__global__ void resample_kernel(const float* px_in, float* px_out,
                                const float* wcum, float base_step,
                                float rand_offset, int np) {
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    float target = base_step * ip + rand_offset;
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

// ===========================================================================
// CPU Particle Filter (Eigen-based)
// ===========================================================================

struct CpuPF {
    int np;
    Eigen::MatrixXf px;   // 4 x NP
    Eigen::VectorXf pw;   // NP
    std::mt19937 gen;
    std::normal_distribution<float> gauss;
    std::uniform_real_distribution<float> uni;

    CpuPF(int np_, unsigned int seed)
        : np(np_), px(Eigen::MatrixXf::Zero(4, np_)),
          pw(Eigen::VectorXf::Constant(np_, 1.0f / np_)),
          gen(seed), gauss(0.0f, 1.0f), uni(0.0f, 1.0f) {}

    void predict_and_weight(const Eigen::Vector2f& u,
                            const Eigen::Matrix2f& Rsim,
                            const std::vector<Observation>& obs,
                            float Q) {
        float inv_coeff = 1.0f / std::sqrt(2.0f * PI * Q);
        for (int i = 0; i < np; i++) {
            float ud_v     = u(0) + gauss(gen) * Rsim(0, 0);
            float ud_omega = u(1) + gauss(gen) * Rsim(1, 1);

            float x   = px(0, i);
            float y   = px(1, i);
            float yaw = px(2, i);

            x   += DT * std::cos(yaw) * ud_v;
            y   += DT * std::sin(yaw) * ud_v;
            yaw += DT * ud_omega;

            px(0, i) = x;
            px(1, i) = y;
            px(2, i) = yaw;
            px(3, i) = ud_v;

            float w = pw(i);
            for (auto& ob : obs) {
                float dx = x - ob.lx;
                float dy = y - ob.ly;
                float prez = std::sqrt(dx * dx + dy * dy);
                float dz = prez - ob.d;
                w *= inv_coeff * std::exp(-dz * dz / (2.0f * Q));
            }
            pw(i) = w;
        }
    }

    void normalize() {
        float total = pw.sum();
        if (total < 1e-30f) total = 1e-30f;
        pw /= total;
    }

    Eigen::Vector4f weighted_mean() {
        Eigen::Vector4f est = Eigen::Vector4f::Zero();
        for (int i = 0; i < np; i++) {
            est += pw(i) * px.col(i);
        }
        return est;
    }

    Eigen::Matrix4f covariance(const Eigen::Vector4f& xEst) {
        Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
        for (int i = 0; i < np; i++) {
            Eigen::Vector4f dx = px.col(i) - xEst;
            P += pw(i) * dx * dx.transpose();
        }
        return P;
    }

    void resample() {
        float Neff_denom = pw.dot(pw);
        float Neff = 1.0f / Neff_denom;
        if (Neff >= NTh) return;

        // Cumulative sum
        Eigen::VectorXf wcum(np);
        wcum(0) = pw(0);
        for (int i = 1; i < np; i++) wcum(i) = wcum(i - 1) + pw(i);

        float rand_offset = uni(gen) / np;
        float base_step = 1.0f / np;

        Eigen::MatrixXf new_px(4, np);
        for (int i = 0; i < np; i++) {
            float target = base_step * i + rand_offset;
            int lo = 0, hi = np - 1;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (wcum(mid) < target) lo = mid + 1;
                else hi = mid;
            }
            new_px.col(i) = px.col(lo);
        }
        px = new_px;
        pw.setConstant(1.0f / np);
    }
};

// ===========================================================================
// Visualization helpers
// ===========================================================================

cv::Point2i cv_offset(float ex, float ey, int w, int h) {
    return cv::Point2i(int(ex * 100) + w / 2, h - int(ey * 100) - h / 3);
}

void ellipse_drawing(cv::Mat& bg, Eigen::Matrix2f pest, Eigen::Vector2f center,
                     cv::Scalar color = {0, 0, 255}) {
    Eigen::EigenSolver<Eigen::Matrix2f> ces(pest);
    Eigen::Matrix2f e_value  = ces.pseudoEigenvalueMatrix();
    Eigen::Matrix2f e_vector = ces.pseudoEigenvectors();

    double angle = std::atan2(e_vector(0, 1), e_vector(0, 0));
    cv::ellipse(bg,
                cv_offset(center(0), center(1), bg.cols, bg.rows),
                cv::Size(e_value(0, 0) * 1000, e_value(1, 1) * 1000),
                angle / PI * 180, 0, 360, color, 2, 4);
}

void draw_pf_scene(cv::Mat& img,
                   const std::vector<Eigen::Vector4f>& hxTrue,
                   const std::vector<Eigen::Vector4f>& hxEst,
                   const std::vector<Eigen::Vector4f>& hxDR,
                   const float* particles_x, const float* particles_y, int np,
                   const Eigen::Matrix<float, 4, 2>& RFID,
                   const std::vector<Observation>& z_host,
                   const Eigen::Vector4f& xEst,
                   Eigen::Matrix4f& PEst,
                   const char* label, double ms) {
    int W = img.cols, H = img.rows;

    // Draw trajectory history
    for (unsigned int j = 0; j < hxTrue.size(); j++) {
        cv::circle(img, cv_offset(hxTrue[j](0), hxTrue[j](1), W, H),
                   7, cv::Scalar(0, 255, 0), -1);
        cv::circle(img, cv_offset(hxEst[j](0), hxEst[j](1), W, H),
                   10, cv::Scalar(255, 0, 0), 5);
        cv::circle(img, cv_offset(hxDR[j](0), hxDR[j](1), W, H),
                   7, cv::Scalar(0, 0, 0), -1);
    }

    // Draw particles
    for (int j = 0; j < np; j++) {
        cv::circle(img, cv_offset(particles_x[j], particles_y[j], W, H),
                   3, cv::Scalar(0, 0, 255), -1);
    }

    // Draw landmarks
    for (int i = 0; i < RFID.rows(); i++) {
        cv::circle(img, cv_offset(RFID(i, 0), RFID(i, 1), W, H),
                   20, cv::Scalar(127, 0, 255), -1);
    }

    // Draw observation lines
    for (unsigned int i = 0; i < z_host.size(); i++) {
        cv::line(img,
                 cv_offset(z_host[i].lx, z_host[i].ly, W, H),
                 cv_offset(hxEst.back()(0), hxEst.back()(1), W, H),
                 cv::Scalar(0, 0, 0), 5);
    }

    // Draw covariance ellipse
    ellipse_drawing(img, PEst.block(0, 0, 2, 2), xEst.head(2));

    // Labels
    cv::putText(img, label, cv::Point(20, 60),
                cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 0), 3);
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f ms", ms);
    cv::putText(img, buf, cv::Point(20, 120),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2);
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    // Simulation parameters
    Eigen::Vector2f u;
    u << 1.0f, 0.1f;

    Eigen::Matrix<float, 4, 2> RFID;
    RFID << 10.0f,  0.0f,
            10.0f, 10.0f,
             0.0f, 15.0f,
            -5.0f, 20.0f;

    float Q = 0.01f;
    float Qsim = 0.04f;

    Eigen::Matrix2f Rsim = Eigen::Matrix2f::Identity();
    Rsim(0, 0) = 1.0f;
    Rsim(1, 1) = (30.0f / 180.0f * PI) * (30.0f / 180.0f * PI);

    // Shared ground truth / dead reckoning (same random seed for observations)
    unsigned int shared_seed = 42;
    std::mt19937 obs_gen(shared_seed);
    std::normal_distribution<float> gaussian_d(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni_d(0.0f, 1.0f);

    // Dead reckoning noise generator (shared)
    std::mt19937 dr_gen(shared_seed + 1);

    Eigen::Vector4f xTrue = Eigen::Vector4f::Zero();
    Eigen::Vector4f xDR   = Eigen::Vector4f::Zero();

    // CPU PF
    CpuPF cpu_pf(NP, shared_seed + 100);
    Eigen::Vector4f cpu_xEst = Eigen::Vector4f::Zero();
    Eigen::Matrix4f cpu_PEst = Eigen::Matrix4f::Identity();
    std::vector<Eigen::Vector4f> cpu_hxTrue, cpu_hxEst, cpu_hxDR;

    // CUDA PF
    const int np = NP;
    const int threads = 256;
    const int blocks  = (np + threads - 1) / threads;

    float *d_px, *d_px_tmp, *d_pw;
    CUDA_CHECK(cudaMalloc(&d_px,     4 * np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_px_tmp, 4 * np * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_px, 0,   4 * np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pw, np * sizeof(float)));
    {
        std::vector<float> pw_init(np, 1.0f / np);
        CUDA_CHECK(cudaMemcpy(d_pw, pw_init.data(), np * sizeof(float), cudaMemcpyHostToDevice));
    }

    Observation *d_obs;
    CUDA_CHECK(cudaMalloc(&d_obs, MAX_LANDMARKS * sizeof(Observation)));

    curandState *d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, np * sizeof(curandState)));
    init_curand_kernel<<<blocks, threads>>>(d_rng_states, shared_seed + 200, np);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *d_xEst;
    CUDA_CHECK(cudaMalloc(&d_xEst, 4 * sizeof(float)));
    float *d_wcum;
    CUDA_CHECK(cudaMalloc(&d_wcum, np * sizeof(float)));

    std::vector<float> h_px(4 * np);
    std::vector<float> h_pw(np);
    float h_xEst[4];

    Eigen::Vector4f cuda_xEst = Eigen::Vector4f::Zero();
    Eigen::Matrix4f cuda_PEst = Eigen::Matrix4f::Identity();
    std::vector<Eigen::Vector4f> cuda_hxTrue, cuda_hxEst, cuda_hxDR;

    // Random generator for CUDA resampling offset
    std::mt19937 resample_gen(shared_seed + 300);

    // Video
    int W = 1750, H = 1750;
    cv::VideoWriter video("gif/comparison_pf.avi",
                          cv::VideoWriter::fourcc('X','V','I','D'), 30, cv::Size(W * 2, H));

    std::cout << "Particle Filter comparison: CPU vs CUDA (" << NP << " particles)" << std::endl;

    float time_val = 0.0f;
    while (time_val <= SIM_TIME) {
        time_val += DT;

        // --- Shared: compute ground truth and dead reckoning ---
        Eigen::Vector2f ud;
        ud(0) = u(0) + gaussian_d(dr_gen) * Rsim(0, 0);
        ud(1) = u(1) + gaussian_d(dr_gen) * Rsim(1, 1);

        xTrue(0) += DT * std::cos(xTrue(2)) * u(0);
        xTrue(1) += DT * std::sin(xTrue(2)) * u(0);
        xTrue(2) += DT * u(1);
        xTrue(3)  = u(0);

        xDR(0) += DT * std::cos(xDR(2)) * ud(0);
        xDR(1) += DT * std::sin(xDR(2)) * ud(0);
        xDR(2) += DT * ud(1);
        xDR(3)  = ud(0);

        // --- Shared: generate observations ---
        std::vector<Observation> z_host;
        for (int i = 0; i < RFID.rows(); i++) {
            float dx = xTrue(0) - RFID(i, 0);
            float dy = xTrue(1) - RFID(i, 1);
            float d  = std::sqrt(dx * dx + dy * dy);
            if (d <= MAX_RANGE) {
                Observation ob;
                ob.d  = d + gaussian_d(obs_gen) * Qsim;
                ob.lx = RFID(i, 0);
                ob.ly = RFID(i, 1);
                z_host.push_back(ob);
            }
        }
        int n_obs = (int)z_host.size();

        // ================= CPU PF =================
        auto t0 = std::chrono::high_resolution_clock::now();

        cpu_pf.predict_and_weight(u, Rsim, z_host, Q);
        cpu_pf.normalize();
        cpu_xEst = cpu_pf.weighted_mean();
        cpu_PEst = cpu_pf.covariance(cpu_xEst);
        cpu_pf.resample();

        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        cpu_hxTrue.push_back(xTrue);
        cpu_hxEst.push_back(cpu_xEst);
        cpu_hxDR.push_back(xDR);

        // Extract CPU particles for drawing
        std::vector<float> cpu_particles_x(np), cpu_particles_y(np);
        for (int i = 0; i < np; i++) {
            cpu_particles_x[i] = cpu_pf.px(0, i);
            cpu_particles_y[i] = cpu_pf.px(1, i);
        }

        // ================= CUDA PF =================
        if (n_obs > 0) {
            CUDA_CHECK(cudaMemcpy(d_obs, z_host.data(),
                                  n_obs * sizeof(Observation), cudaMemcpyHostToDevice));
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        predict_and_weight_kernel<<<blocks, threads>>>(
            d_px, d_pw, u(0), u(1),
            Rsim(0, 0), Rsim(1, 1),
            d_obs, n_obs, Q, d_rng_states, np);

        normalize_weights_kernel<<<1, threads, threads * sizeof(float)>>>(d_pw, np);

        weighted_mean_kernel<<<1, threads, threads * 4 * sizeof(float)>>>(
            d_px, d_pw, d_xEst, np);

        // Read back for resampling decision and covariance
        CUDA_CHECK(cudaMemcpy(h_xEst, d_xEst, 4 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_px.data(), d_px, 4 * np * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pw.data(), d_pw, np * sizeof(float), cudaMemcpyDeviceToHost));

        // Resampling
        float Neff_denom = 0.0f;
        for (int i = 0; i < np; i++) Neff_denom += h_pw[i] * h_pw[i];
        float Neff = 1.0f / Neff_denom;

        if (Neff < NTh) {
            cumsum_kernel<<<1, 1>>>(d_pw, d_wcum, np);
            float rand_offset = uni_d(resample_gen) / np;
            float base_step   = 1.0f / np;
            resample_kernel<<<blocks, threads>>>(
                d_px, d_px_tmp, d_wcum, base_step, rand_offset, np);
            CUDA_CHECK(cudaMemcpy(d_px, d_px_tmp, 4 * np * sizeof(float), cudaMemcpyDeviceToDevice));
            std::vector<float> pw_uniform(np, 1.0f / np);
            CUDA_CHECK(cudaMemcpy(d_pw, pw_uniform.data(), np * sizeof(float), cudaMemcpyHostToDevice));
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cuda_ms;
        cudaEventElapsedTime(&cuda_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cuda_xEst << h_xEst[0], h_xEst[1], h_xEst[2], h_xEst[3];

        // Covariance for CUDA side
        cuda_PEst = Eigen::Matrix4f::Zero();
        for (int i = 0; i < np; i++) {
            Eigen::Vector4f dx;
            dx << h_px[0 * np + i] - cuda_xEst(0),
                  h_px[1 * np + i] - cuda_xEst(1),
                  h_px[2 * np + i] - cuda_xEst(2),
                  h_px[3 * np + i] - cuda_xEst(3);
            cuda_PEst += h_pw[i] * dx * dx.transpose();
        }

        cuda_hxTrue.push_back(xTrue);
        cuda_hxEst.push_back(cuda_xEst);
        cuda_hxDR.push_back(xDR);

        // Re-read particles after possible resampling for drawing
        CUDA_CHECK(cudaMemcpy(h_px.data(), d_px, 4 * np * sizeof(float), cudaMemcpyDeviceToHost));

        // ================= Visualization =================
        cv::Mat left(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Mat right(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

        draw_pf_scene(left, cpu_hxTrue, cpu_hxEst, cpu_hxDR,
                      cpu_particles_x.data(), cpu_particles_y.data(), np,
                      RFID, z_host, cpu_xEst, cpu_PEst,
                      "CPU (Eigen)", cpu_ms);

        // Prepare GPU particles as separate x/y arrays
        std::vector<float> cuda_particles_x(np), cuda_particles_y(np);
        for (int i = 0; i < np; i++) {
            cuda_particles_x[i] = h_px[0 * np + i];
            cuda_particles_y[i] = h_px[1 * np + i];
        }

        draw_pf_scene(right, cuda_hxTrue, cuda_hxEst, cuda_hxDR,
                      cuda_particles_x.data(), cuda_particles_y.data(), np,
                      RFID, z_host, cuda_xEst, cuda_PEst,
                      "CUDA (GPU)", cuda_ms);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        video.write(combined);
    }

    video.release();
    std::cout << "Video saved to gif/comparison_pf.avi" << std::endl;

    // Convert to gif
    system("ffmpeg -y -i gif/comparison_pf.avi "
           "-vf 'fps=15,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_pf.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_pf.gif" << std::endl;

    // Cleanup
    cudaFree(d_px);
    cudaFree(d_px_tmp);
    cudaFree(d_pw);
    cudaFree(d_obs);
    cudaFree(d_rng_states);
    cudaFree(d_xEst);
    cudaFree(d_wcum);

    return 0;
}
