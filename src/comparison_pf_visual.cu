/*************************************************************************
    Particle Filter Visual Comparison:
    Left: CPU 100 particles (large red dots, radius 5) - noisy estimation
    Right: CUDA 10,000 particles (small red dots, radius 2) - smooth estimation
    Same ground truth, RFID landmarks at (10,0),(10,10),(0,15),(-5,20)
    800x800 per side
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

#define SIM_TIME 30.0f
#define DT 0.1f
#define PI 3.141592653f
#define MAX_RANGE 20.0f
#define NP_CPU 100
#define NP_CUDA 10000
#define MAX_LANDMARKS 16

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

struct Observation { float d, lx, ly; };

// ===========================================================================
// CUDA Kernels
// ===========================================================================
__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void predict_and_weight_kernel(
    float* px, float* pw, const float u_v, const float u_omega,
    const float rsim_0, const float rsim_1,
    const Observation* obs, const int n_obs, const float Q,
    curandState* rng_states, const int np)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;
    curandState lr = rng_states[ip];
    float ud_v = u_v + curand_normal(&lr) * rsim_0;
    float ud_o = u_omega + curand_normal(&lr) * rsim_1;
    float x = px[0 * np + ip], y = px[1 * np + ip], yaw = px[2 * np + ip];
    x += DT * cosf(yaw) * ud_v; y += DT * sinf(yaw) * ud_v; yaw += DT * ud_o;
    px[0 * np + ip] = x; px[1 * np + ip] = y; px[2 * np + ip] = yaw; px[3 * np + ip] = ud_v;
    float w = pw[ip], ic = 1.0f / sqrtf(2.0f * PI * Q);
    for (int i = 0; i < n_obs; i++) {
        float dx = x - obs[i].lx, dy = y - obs[i].ly;
        float dz = sqrtf(dx * dx + dy * dy) - obs[i].d;
        w *= ic * expf(-dz * dz / (2.0f * Q));
    }
    pw[ip] = w; rng_states[ip] = lr;
}

__global__ void normalize_weights_kernel(float* pw, int np) {
    extern __shared__ float sd[];
    int tid = threadIdx.x;
    float val = 0; for (int i = tid; i < np; i += blockDim.x) val += pw[i];
    sd[tid] = val; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sd[tid] += sd[tid + s]; __syncthreads(); }
    float total = sd[0]; if (total < 1e-30f) total = 1e-30f;
    for (int i = tid; i < np; i += blockDim.x) pw[i] /= total;
}

__global__ void weighted_mean_kernel(const float* px, const float* pw, float* xe, int np) {
    extern __shared__ float sd[];
    int tid = threadIdx.x;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = tid; i < np; i += blockDim.x) {
        float w = pw[i]; s0 += px[0 * np + i] * w; s1 += px[1 * np + i] * w;
        s2 += px[2 * np + i] * w; s3 += px[3 * np + i] * w;
    }
    sd[tid * 4] = s0; sd[tid * 4 + 1] = s1; sd[tid * 4 + 2] = s2; sd[tid * 4 + 3] = s3;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { for (int k = 0; k < 4; k++) sd[tid * 4 + k] += sd[(tid + s) * 4 + k]; }
        __syncthreads();
    }
    if (tid == 0) { xe[0] = sd[0]; xe[1] = sd[1]; xe[2] = sd[2]; xe[3] = sd[3]; }
}

__global__ void cumsum_kernel(const float* pw, float* wc, int np) {
    wc[0] = pw[0]; for (int i = 1; i < np; i++) wc[i] = wc[i - 1] + pw[i];
}

__global__ void resample_kernel(const float* pin, float* pout, const float* wc, float bs, float ro, int np) {
    int ip = blockIdx.x * blockDim.x + threadIdx.x; if (ip >= np) return;
    float t = bs * ip + ro; int lo = 0, hi = np - 1;
    while (lo < hi) { int m = (lo + hi) / 2; if (wc[m] < t) lo = m + 1; else hi = m; }
    for (int k = 0; k < 4; k++) pout[k * np + ip] = pin[k * np + lo];
}

// ===========================================================================
// CPU Particle Filter (Eigen-based)
// ===========================================================================
struct CpuPF {
    int np;
    Eigen::MatrixXf px;
    Eigen::VectorXf pw;
    std::mt19937 gen;
    std::normal_distribution<float> gauss;
    std::uniform_real_distribution<float> uni;

    CpuPF(int n, unsigned s) : np(n), px(Eigen::MatrixXf::Zero(4, n)),
        pw(Eigen::VectorXf::Constant(n, 1.0f / n)), gen(s), gauss(0, 1), uni(0, 1) {}

    void step(const Eigen::Vector2f& u, const Eigen::Matrix2f& Rsim,
              const std::vector<Observation>& obs, float Q) {
        float ic = 1.0f / std::sqrt(2.0f * PI * Q);
        for (int i = 0; i < np; i++) {
            float uv = u(0) + gauss(gen) * Rsim(0, 0);
            float uo = u(1) + gauss(gen) * Rsim(1, 1);
            float x = px(0, i), y = px(1, i), yaw = px(2, i);
            x += DT * std::cos(yaw) * uv; y += DT * std::sin(yaw) * uv; yaw += DT * uo;
            px(0, i) = x; px(1, i) = y; px(2, i) = yaw; px(3, i) = uv;
            float w = pw(i);
            for (auto& ob : obs) {
                float dz = std::sqrt((x - ob.lx) * (x - ob.lx) + (y - ob.ly) * (y - ob.ly)) - ob.d;
                w *= ic * std::exp(-dz * dz / (2.0f * Q));
            }
            pw(i) = w;
        }
    }

    void normalize() { float t = pw.sum(); if (t < 1e-30f) t = 1e-30f; pw /= t; }

    Eigen::Vector4f mean() {
        Eigen::Vector4f e = Eigen::Vector4f::Zero();
        for (int i = 0; i < np; i++) e += pw(i) * px.col(i);
        return e;
    }

    void resample() {
        float nd = pw.dot(pw);
        if (1.0f / nd >= np / 2) return;
        Eigen::VectorXf wc(np); wc(0) = pw(0);
        for (int i = 1; i < np; i++) wc(i) = wc(i - 1) + pw(i);
        float ro = uni(gen) / np, bs = 1.0f / np;
        Eigen::MatrixXf npx(4, np);
        for (int i = 0; i < np; i++) {
            float t = bs * i + ro; int lo = 0, hi = np - 1;
            while (lo < hi) { int m = (lo + hi) / 2; if (wc(m) < t) lo = m + 1; else hi = m; }
            npx.col(i) = px.col(lo);
        }
        px = npx; pw.setConstant(1.0f / np);
    }
};

// ===========================================================================
// Visualization
// ===========================================================================
cv::Point2i cv_offset(float x, float y, int w, int h) {
    return cv::Point2i(int(x * 80) + w / 2, h - int(y * 80) - h / 3);
}

void draw_panel(cv::Mat& img,
    const std::vector<Eigen::Vector4f>& hTrue,
    const std::vector<Eigen::Vector4f>& hEst,
    const std::vector<Eigen::Vector4f>& hDR,
    const float* part_x, const float* part_y, int np, int dot_r,
    const Eigen::Matrix<float, 4, 2>& RFID,
    const std::vector<Observation>& z,
    const char* label, int n_particles)
{
    int W = img.cols, H = img.rows;
    for (unsigned j = 0; j < hTrue.size(); j++) {
        cv::circle(img, cv_offset(hTrue[j](0), hTrue[j](1), W, H), 5, cv::Scalar(0, 200, 0), -1);
        cv::circle(img, cv_offset(hEst[j](0), hEst[j](1), W, H), 6, cv::Scalar(255, 0, 0), 3);
        cv::circle(img, cv_offset(hDR[j](0), hDR[j](1), W, H), 4, cv::Scalar(0, 0, 0), -1);
    }
    for (int j = 0; j < np; j++)
        cv::circle(img, cv_offset(part_x[j], part_y[j], W, H), dot_r, cv::Scalar(0, 0, 255), -1);
    for (int i = 0; i < RFID.rows(); i++)
        cv::circle(img, cv_offset(RFID(i, 0), RFID(i, 1), W, H), 15, cv::Scalar(127, 0, 255), -1);
    for (unsigned i = 0; i < z.size(); i++)
        cv::line(img, cv_offset(z[i].lx, z[i].ly, W, H),
            cv_offset(hEst.back()(0), hEst.back()(1), W, H), cv::Scalar(0, 0, 0), 3);

    cv::putText(img, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    char buf[64]; snprintf(buf, sizeof(buf), "%d particles", n_particles);
    cv::putText(img, buf, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 200), 2);
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    std::cout << "PF Visual: CPU 100 vs CUDA 10000 particles" << std::endl;

    Eigen::Vector2f u; u << 1.0f, 0.1f;
    Eigen::Matrix<float, 4, 2> RFID;
    RFID << 10.0f, 0.0f, 10.0f, 10.0f, 0.0f, 15.0f, -5.0f, 20.0f;

    float Q = 0.01f, Qsim = 0.04f;
    Eigen::Matrix2f Rsim = Eigen::Matrix2f::Identity();
    Rsim(0, 0) = 1.0f;
    Rsim(1, 1) = (30.0f / 180.0f * PI) * (30.0f / 180.0f * PI);

    unsigned seed = 42;
    std::mt19937 obs_gen(seed), dr_gen(seed + 1);
    std::normal_distribution<float> gauss(0, 1);
    std::uniform_real_distribution<float> uni_d(0, 1);

    Eigen::Vector4f xTrue = Eigen::Vector4f::Zero(), xDR = Eigen::Vector4f::Zero();

    // CPU PF (100 particles)
    CpuPF cpu_pf(NP_CPU, seed + 100);
    Eigen::Vector4f cpu_xEst = Eigen::Vector4f::Zero();
    std::vector<Eigen::Vector4f> cpu_hTrue, cpu_hEst, cpu_hDR;

    // CUDA PF (10000 particles)
    const int np_cuda = NP_CUDA;
    const int thr = 256, blk_cuda = (np_cuda + thr - 1) / thr;

    float *d_px, *d_px2, *d_pw, *d_xEst, *d_wcum;
    Observation *d_obs;
    curandState *d_rng;
    CUDA_CHECK(cudaMalloc(&d_px, 4 * np_cuda * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_px2, 4 * np_cuda * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_px, 0, 4 * np_cuda * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pw, np_cuda * sizeof(float)));
    {
        std::vector<float> pw(np_cuda, 1.0f / np_cuda);
        CUDA_CHECK(cudaMemcpy(d_pw, pw.data(), np_cuda * sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_obs, MAX_LANDMARKS * sizeof(Observation)));
    CUDA_CHECK(cudaMalloc(&d_rng, np_cuda * sizeof(curandState)));
    init_curand_kernel<<<blk_cuda, thr>>>(d_rng, seed + 200, np_cuda);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMalloc(&d_xEst, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wcum, np_cuda * sizeof(float)));

    std::vector<float> h_px(4 * np_cuda), h_pw(np_cuda);
    float h_xEst[4];
    Eigen::Vector4f cuda_xEst = Eigen::Vector4f::Zero();
    std::vector<Eigen::Vector4f> cuda_hTrue, cuda_hEst, cuda_hDR;
    std::mt19937 resample_gen(seed + 300);

    int W = 800, H = 800;
    cv::VideoWriter video("gif/comparison_pf_visual.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, cv::Size(W * 2, H));

    float time_val = 0;
    while (time_val <= SIM_TIME) {
        time_val += DT;

        Eigen::Vector2f ud;
        ud(0) = u(0) + gauss(dr_gen) * Rsim(0, 0);
        ud(1) = u(1) + gauss(dr_gen) * Rsim(1, 1);

        xTrue(0) += DT * std::cos(xTrue(2)) * u(0);
        xTrue(1) += DT * std::sin(xTrue(2)) * u(0);
        xTrue(2) += DT * u(1); xTrue(3) = u(0);

        xDR(0) += DT * std::cos(xDR(2)) * ud(0);
        xDR(1) += DT * std::sin(xDR(2)) * ud(0);
        xDR(2) += DT * ud(1); xDR(3) = ud(0);

        std::vector<Observation> z;
        for (int i = 0; i < RFID.rows(); i++) {
            float dx = xTrue(0) - RFID(i, 0), dy = xTrue(1) - RFID(i, 1);
            float d = std::sqrt(dx * dx + dy * dy);
            if (d <= MAX_RANGE) {
                Observation ob; ob.d = d + gauss(obs_gen) * Qsim;
                ob.lx = RFID(i, 0); ob.ly = RFID(i, 1); z.push_back(ob);
            }
        }
        int nobs = z.size();

        // CPU PF
        cpu_pf.step(u, Rsim, z, Q);
        cpu_pf.normalize();
        cpu_xEst = cpu_pf.mean();
        cpu_pf.resample();
        cpu_hTrue.push_back(xTrue); cpu_hEst.push_back(cpu_xEst); cpu_hDR.push_back(xDR);

        std::vector<float> cpu_px(NP_CPU), cpu_py(NP_CPU);
        for (int i = 0; i < NP_CPU; i++) { cpu_px[i] = cpu_pf.px(0, i); cpu_py[i] = cpu_pf.px(1, i); }

        // CUDA PF
        if (nobs > 0)
            CUDA_CHECK(cudaMemcpy(d_obs, z.data(), nobs * sizeof(Observation), cudaMemcpyHostToDevice));

        predict_and_weight_kernel<<<blk_cuda, thr>>>(
            d_px, d_pw, u(0), u(1), Rsim(0, 0), Rsim(1, 1), d_obs, nobs, Q, d_rng, np_cuda);
        normalize_weights_kernel<<<1, thr, thr * sizeof(float)>>>(d_pw, np_cuda);
        weighted_mean_kernel<<<1, thr, thr * 4 * sizeof(float)>>>(d_px, d_pw, d_xEst, np_cuda);

        CUDA_CHECK(cudaMemcpy(h_xEst, d_xEst, 4 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pw.data(), d_pw, np_cuda * sizeof(float), cudaMemcpyDeviceToHost));

        float nd = 0; for (int i = 0; i < np_cuda; i++) nd += h_pw[i] * h_pw[i];
        if (1.0f / nd < np_cuda / 2) {
            cumsum_kernel<<<1, 1>>>(d_pw, d_wcum, np_cuda);
            float ro = uni_d(resample_gen) / np_cuda;
            resample_kernel<<<blk_cuda, thr>>>(d_px, d_px2, d_wcum, 1.0f / np_cuda, ro, np_cuda);
            CUDA_CHECK(cudaMemcpy(d_px, d_px2, 4 * np_cuda * sizeof(float), cudaMemcpyDeviceToDevice));
            std::vector<float> uw(np_cuda, 1.0f / np_cuda);
            CUDA_CHECK(cudaMemcpy(d_pw, uw.data(), np_cuda * sizeof(float), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cuda_xEst << h_xEst[0], h_xEst[1], h_xEst[2], h_xEst[3];
        cuda_hTrue.push_back(xTrue); cuda_hEst.push_back(cuda_xEst); cuda_hDR.push_back(xDR);

        CUDA_CHECK(cudaMemcpy(h_px.data(), d_px, 4 * np_cuda * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> cuda_px(np_cuda), cuda_py(np_cuda);
        for (int i = 0; i < np_cuda; i++) { cuda_px[i] = h_px[i]; cuda_py[i] = h_px[np_cuda + i]; }

        // Visualization
        cv::Mat left(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Mat right(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

        draw_panel(left, cpu_hTrue, cpu_hEst, cpu_hDR,
            cpu_px.data(), cpu_py.data(), NP_CPU, 5, RFID, z,
            "CPU (100 particles)", NP_CPU);
        draw_panel(right, cuda_hTrue, cuda_hEst, cuda_hDR,
            cuda_px.data(), cuda_py.data(), np_cuda, 2, RFID, z,
            "CUDA (10,000 particles)", np_cuda);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        video.write(combined);
    }

    video.release();
    system("ffmpeg -y -i gif/comparison_pf_visual.avi "
           "-vf 'fps=15,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_pf_visual.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_pf_visual.gif" << std::endl;

    cudaFree(d_px); cudaFree(d_px2); cudaFree(d_pw); cudaFree(d_obs);
    cudaFree(d_rng); cudaFree(d_xEst); cudaFree(d_wcum);
    return 0;
}
