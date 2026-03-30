/*************************************************************************
    Particle Filter: CPU (100 particles) vs CUDA (10,000 particles)
    Visual comparison showing the quality difference between sparse and
    dense particle sets running in real-time side-by-side.
    Left panel:  CPU - 100 particles (sparse, noisy estimation)
    Right panel: CUDA - 10,000 particles (dense, smooth estimation)
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

#define CPU_NP 100
#define CUDA_NP 10000
#define CPU_NTh (CPU_NP / 2)
#define CUDA_NTh (CUDA_NP / 2)
#define MAX_LANDMARKS 16

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

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

    x   += DT * cosf(yaw) * ud_v;
    y   += DT * sinf(yaw) * ud_v;
    yaw += DT * ud_omega;

    px[0 * np + ip] = x;
    px[1 * np + ip] = y;
    px[2 * np + ip] = yaw;
    px[3 * np + ip] = ud_v;

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
// CPU Particle Filter (Eigen-based, 100 particles)
// ===========================================================================

struct CpuPF {
    int np;
    Eigen::MatrixXf px;   // 4 x np
    Eigen::VectorXf pw;   // np
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
        if (Neff >= np / 2) return;

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

static cv::Point2i cv_offset(float ex, float ey, int w, int h) {
    return cv::Point2i(int(ex * 100) + w / 2, h - int(ey * 100) - h / 3);
}

static void ellipse_drawing(cv::Mat& bg, Eigen::Matrix2f pest, Eigen::Vector2f center,
                            cv::Scalar color = {0, 0, 255}) {
    Eigen::EigenSolver<Eigen::Matrix2f> ces(pest);
    Eigen::Matrix2f e_value  = ces.pseudoEigenvalueMatrix();
    Eigen::Matrix2f e_vector = ces.pseudoEigenvectors();

    double angle = std::atan2(e_vector(0, 1), e_vector(0, 0));

    int ax_a = std::max(1, (int)(e_value(0, 0) * 1000));
    int ax_b = std::max(1, (int)(e_value(1, 1) * 1000));

    cv::ellipse(bg,
                cv_offset(center(0), center(1), bg.cols, bg.rows),
                cv::Size(ax_a, ax_b),
                angle / PI * 180, 0, 360, color, 2, 4);
}

// Draw a single panel for one PF
static void draw_panel(cv::Mat& img,
                       const std::vector<Eigen::Vector4f>& hxTrue,
                       const std::vector<Eigen::Vector4f>& hxEst,
                       const std::vector<Eigen::Vector4f>& hxDR,
                       const float* particles_x, const float* particles_y, int np,
                       int particle_radius,
                       const Eigen::Matrix<float, 4, 2>& RFID,
                       const std::vector<Observation>& z_host,
                       const Eigen::Vector4f& xEst,
                       Eigen::Matrix4f& PEst,
                       const char* label) {
    int W = img.cols, H = img.rows;

    // Draw landmarks as purple circles
    for (int i = 0; i < RFID.rows(); i++) {
        cv::circle(img, cv_offset(RFID(i, 0), RFID(i, 1), W, H),
                   15, cv::Scalar(127, 0, 255), -1);
    }

    // Draw observation lines
    if (!hxEst.empty()) {
        for (unsigned int i = 0; i < z_host.size(); i++) {
            cv::line(img,
                     cv_offset(z_host[i].lx, z_host[i].ly, W, H),
                     cv_offset(hxEst.back()(0), hxEst.back()(1), W, H),
                     cv::Scalar(180, 180, 180), 1);
        }
    }

    // Draw particles (red)
    for (int j = 0; j < np; j++) {
        cv::circle(img, cv_offset(particles_x[j], particles_y[j], W, H),
                   particle_radius, cv::Scalar(0, 0, 255), -1);
    }

    // Draw trajectory history
    for (unsigned int j = 1; j < hxTrue.size(); j++) {
        // Ground truth (green line)
        cv::line(img,
                 cv_offset(hxTrue[j-1](0), hxTrue[j-1](1), W, H),
                 cv_offset(hxTrue[j](0), hxTrue[j](1), W, H),
                 cv::Scalar(0, 200, 0), 3);
        // Dead reckoning (gray line)
        cv::line(img,
                 cv_offset(hxDR[j-1](0), hxDR[j-1](1), W, H),
                 cv_offset(hxDR[j](0), hxDR[j](1), W, H),
                 cv::Scalar(160, 160, 160), 2);
        // Estimation (blue line)
        cv::line(img,
                 cv_offset(hxEst[j-1](0), hxEst[j-1](1), W, H),
                 cv_offset(hxEst[j](0), hxEst[j](1), W, H),
                 cv::Scalar(255, 0, 0), 3);
    }

    // Draw covariance ellipse
    if (!hxEst.empty()) {
        ellipse_drawing(img, PEst.block(0, 0, 2, 2), xEst.head(2));
    }

    // Label
    cv::putText(img, label, cv::Point(15, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 0), 3);
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

    unsigned int shared_seed = 42;
    std::mt19937 obs_gen(shared_seed);
    std::normal_distribution<float> gaussian_d(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni_d(0.0f, 1.0f);

    std::mt19937 dr_gen(shared_seed + 1);

    Eigen::Vector4f xTrue = Eigen::Vector4f::Zero();
    Eigen::Vector4f xDR   = Eigen::Vector4f::Zero();

    // ===== CPU PF (100 particles) =====
    CpuPF cpu_pf(CPU_NP, shared_seed + 100);
    Eigen::Vector4f cpu_xEst = Eigen::Vector4f::Zero();
    Eigen::Matrix4f cpu_PEst = Eigen::Matrix4f::Identity();
    std::vector<Eigen::Vector4f> cpu_hxTrue, cpu_hxEst, cpu_hxDR;

    // ===== CUDA PF (10,000 particles) =====
    const int cuda_np = CUDA_NP;
    const int threads = 256;
    const int blocks  = (cuda_np + threads - 1) / threads;

    float *d_px, *d_px_tmp, *d_pw;
    CUDA_CHECK(cudaMalloc(&d_px,     4 * cuda_np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_px_tmp, 4 * cuda_np * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_px, 0,   4 * cuda_np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pw, cuda_np * sizeof(float)));
    {
        std::vector<float> pw_init(cuda_np, 1.0f / cuda_np);
        CUDA_CHECK(cudaMemcpy(d_pw, pw_init.data(), cuda_np * sizeof(float), cudaMemcpyHostToDevice));
    }

    Observation *d_obs;
    CUDA_CHECK(cudaMalloc(&d_obs, MAX_LANDMARKS * sizeof(Observation)));

    curandState *d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, cuda_np * sizeof(curandState)));
    init_curand_kernel<<<blocks, threads>>>(d_rng_states, shared_seed + 200, cuda_np);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *d_xEst;
    CUDA_CHECK(cudaMalloc(&d_xEst, 4 * sizeof(float)));
    float *d_wcum;
    CUDA_CHECK(cudaMalloc(&d_wcum, cuda_np * sizeof(float)));

    std::vector<float> h_px(4 * cuda_np);
    std::vector<float> h_pw(cuda_np);
    float h_xEst[4];

    Eigen::Vector4f cuda_xEst = Eigen::Vector4f::Zero();
    Eigen::Matrix4f cuda_PEst = Eigen::Matrix4f::Identity();
    std::vector<Eigen::Vector4f> cuda_hxTrue, cuda_hxEst, cuda_hxDR;

    std::mt19937 resample_gen(shared_seed + 300);

    // Video: 800x800 per side = 1600x800 combined
    const int PW = 800, PH = 800;
    cv::VideoWriter video("gif/comparison_pf_visual.avi",
                          cv::VideoWriter::fourcc('X','V','I','D'), 30, cv::Size(PW * 2, PH));

    std::cout << "Particle Filter visual comparison: CPU (" << CPU_NP
              << " particles) vs CUDA (" << CUDA_NP << " particles)" << std::endl;

    float time_val = 0.0f;
    while (time_val <= SIM_TIME) {
        time_val += DT;

        // --- Shared: ground truth and dead reckoning ---
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

        // ================= CPU PF (100 particles) =================
        cpu_pf.predict_and_weight(u, Rsim, z_host, Q);
        cpu_pf.normalize();
        cpu_xEst = cpu_pf.weighted_mean();
        cpu_PEst = cpu_pf.covariance(cpu_xEst);
        cpu_pf.resample();

        cpu_hxTrue.push_back(xTrue);
        cpu_hxEst.push_back(cpu_xEst);
        cpu_hxDR.push_back(xDR);

        std::vector<float> cpu_particles_x(CPU_NP), cpu_particles_y(CPU_NP);
        for (int i = 0; i < CPU_NP; i++) {
            cpu_particles_x[i] = cpu_pf.px(0, i);
            cpu_particles_y[i] = cpu_pf.px(1, i);
        }

        // ================= CUDA PF (10,000 particles) =================
        if (n_obs > 0) {
            CUDA_CHECK(cudaMemcpy(d_obs, z_host.data(),
                                  n_obs * sizeof(Observation), cudaMemcpyHostToDevice));
        }

        predict_and_weight_kernel<<<blocks, threads>>>(
            d_px, d_pw, u(0), u(1),
            Rsim(0, 0), Rsim(1, 1),
            d_obs, n_obs, Q, d_rng_states, cuda_np);

        normalize_weights_kernel<<<1, threads, threads * sizeof(float)>>>(d_pw, cuda_np);

        weighted_mean_kernel<<<1, threads, threads * 4 * sizeof(float)>>>(
            d_px, d_pw, d_xEst, cuda_np);

        CUDA_CHECK(cudaMemcpy(h_xEst, d_xEst, 4 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_px.data(), d_px, 4 * cuda_np * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pw.data(), d_pw, cuda_np * sizeof(float), cudaMemcpyDeviceToHost));

        // Resampling
        float Neff_denom = 0.0f;
        for (int i = 0; i < cuda_np; i++) Neff_denom += h_pw[i] * h_pw[i];
        float Neff = 1.0f / Neff_denom;

        if (Neff < CUDA_NTh) {
            cumsum_kernel<<<1, 1>>>(d_pw, d_wcum, cuda_np);
            float rand_offset = uni_d(resample_gen) / cuda_np;
            float base_step   = 1.0f / cuda_np;
            resample_kernel<<<blocks, threads>>>(
                d_px, d_px_tmp, d_wcum, base_step, rand_offset, cuda_np);
            CUDA_CHECK(cudaMemcpy(d_px, d_px_tmp, 4 * cuda_np * sizeof(float), cudaMemcpyDeviceToDevice));
            std::vector<float> pw_uniform(cuda_np, 1.0f / cuda_np);
            CUDA_CHECK(cudaMemcpy(d_pw, pw_uniform.data(), cuda_np * sizeof(float), cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        cuda_xEst << h_xEst[0], h_xEst[1], h_xEst[2], h_xEst[3];

        // Covariance for CUDA side
        cuda_PEst = Eigen::Matrix4f::Zero();
        for (int i = 0; i < cuda_np; i++) {
            Eigen::Vector4f dx;
            dx << h_px[0 * cuda_np + i] - cuda_xEst(0),
                  h_px[1 * cuda_np + i] - cuda_xEst(1),
                  h_px[2 * cuda_np + i] - cuda_xEst(2),
                  h_px[3 * cuda_np + i] - cuda_xEst(3);
            cuda_PEst += h_pw[i] * dx * dx.transpose();
        }

        cuda_hxTrue.push_back(xTrue);
        cuda_hxEst.push_back(cuda_xEst);
        cuda_hxDR.push_back(xDR);

        // Re-read particles after possible resampling
        CUDA_CHECK(cudaMemcpy(h_px.data(), d_px, 4 * cuda_np * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> cuda_particles_x(cuda_np), cuda_particles_y(cuda_np);
        for (int i = 0; i < cuda_np; i++) {
            cuda_particles_x[i] = h_px[0 * cuda_np + i];
            cuda_particles_y[i] = h_px[1 * cuda_np + i];
        }

        // ================= Visualization =================
        cv::Mat left(PH, PW, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Mat right(PH, PW, CV_8UC3, cv::Scalar(255, 255, 255));

        // CPU: 100 particles, radius 5 (large, scattered)
        draw_panel(left, cpu_hxTrue, cpu_hxEst, cpu_hxDR,
                   cpu_particles_x.data(), cpu_particles_y.data(), CPU_NP,
                   5, RFID, z_host, cpu_xEst, cpu_PEst,
                   "CPU: 100 particles");

        // CUDA: 10,000 particles, radius 2 (dense cloud)
        draw_panel(right, cuda_hxTrue, cuda_hxEst, cuda_hxDR,
                   cuda_particles_x.data(), cuda_particles_y.data(), cuda_np,
                   2, RFID, z_host, cuda_xEst, cuda_PEst,
                   "CUDA: 10,000 particles");

        // Separator line
        cv::Mat combined;
        cv::hconcat(left, right, combined);
        cv::line(combined, cv::Point(PW, 0), cv::Point(PW, PH),
                 cv::Scalar(0, 0, 0), 2);

        video.write(combined);
    }

    video.release();
    std::cout << "Video saved to gif/comparison_pf_visual.avi" << std::endl;

    // Convert to gif
    system("ffmpeg -y -i gif/comparison_pf_visual.avi "
           "-vf 'fps=15,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_pf_visual.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_pf_visual.gif" << std::endl;

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
