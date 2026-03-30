/*************************************************************************
    Benchmark: Particle Filter - CPU vs CUDA
    Measures computation time excluding visualization
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <Eigen/Eigen>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define DT 0.1f
#define PI 3.141592653f
#define MAX_RANGE 20.0f
#define SIM_TIME 10.0f  // reduced for benchmark speed

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================
// CPU Implementation
// ============================================================
Eigen::Vector4f motion_model_cpu(Eigen::Vector4f x, Eigen::Vector2f u) {
    Eigen::Matrix4f F = Eigen::Matrix4f::Identity();
    Eigen::Matrix<float, 4, 2> B;
    B << DT * std::cos(x(2)), 0,
         DT * std::sin(x(2)), 0,
         0.0, DT,
         1.0, 0.0;
    return F * x + B * u;
}

float gauss_likelihood(float x, float sigma) {
    return 1.0f / std::sqrt(2.0f * PI * sigma * sigma) *
           std::exp(-x * x / (2 * sigma * sigma));
}

struct CPUResult {
    double total_ms;
    int steps;
};

template <int NP>
CPUResult run_cpu_pf() {
    float time_val = 0.0f;
    Eigen::Vector2f u; u << 1.0f, 0.1f;
    Eigen::Matrix<float, 4, 2> RFID;
    RFID << 10.0f, 0.0f, 10.0f, 10.0f, 0.0f, 15.0f, -5.0f, 20.0f;
    Eigen::Vector4f xTrue = Eigen::Vector4f::Zero();
    float Q = 0.01f;
    Eigen::Matrix2f Rsim = Eigen::Matrix2f::Identity();
    Rsim(0,0) = 1.0f; Rsim(1,1) = (30.0f/180.0f*PI)*(30.0f/180.0f*PI);
    float Qsim = 0.04f;

    Eigen::Matrix<float, 4, Eigen::Dynamic> px = Eigen::Matrix<float, 4, Eigen::Dynamic>::Zero(4, NP);
    Eigen::VectorXf pw = Eigen::VectorXf::Ones(NP) / NP;

    std::mt19937 gen(42);
    std::normal_distribution<float> gd(0, 1);
    std::uniform_real_distribution<float> ud(0, 1);

    int steps = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    while (time_val <= SIM_TIME) {
        time_val += DT;
        xTrue = motion_model_cpu(xTrue, u);

        // generate observations
        struct Obs { float d, lx, ly; };
        std::vector<Obs> z;
        for (int i = 0; i < RFID.rows(); i++) {
            float dx = xTrue(0) - RFID(i,0);
            float dy = xTrue(1) - RFID(i,1);
            float d = std::sqrt(dx*dx + dy*dy);
            if (d <= MAX_RANGE) z.push_back({d + gd(gen)*Qsim, RFID(i,0), RFID(i,1)});
        }

        // PF update
        for (int ip = 0; ip < NP; ip++) {
            Eigen::Vector4f x = px.col(ip);
            float w = pw(ip);
            Eigen::Vector2f ud_v;
            ud_v(0) = u(0) + gd(gen) * Rsim(0,0);
            ud_v(1) = u(1) + gd(gen) * Rsim(1,1);
            x = motion_model_cpu(x, ud_v);
            for (auto& obs : z) {
                float ddx = x(0) - obs.lx;
                float ddy = x(1) - obs.ly;
                float prez = std::sqrt(ddx*ddx + ddy*ddy);
                w *= gauss_likelihood(prez - obs.d, std::sqrt(Q));
            }
            px.col(ip) = x;
            pw(ip) = w;
        }
        pw /= pw.sum();

        // resampling
        float Neff = 1.0f / pw.dot(pw);
        if (Neff < NP / 2) {
            Eigen::VectorXf wcum(NP);
            wcum(0) = pw(0);
            for (int i = 1; i < NP; i++) wcum(i) = wcum(i-1) + pw(i);
            Eigen::Matrix<float, 4, Eigen::Dynamic> output(4, NP);
            for (int i = 0; i < NP; i++) {
                float target = (float)i / NP + ud(gen) / NP;
                int idx = 0;
                while (idx < NP - 1 && wcum(idx) < target) idx++;
                output.col(i) = px.col(idx);
            }
            px = output;
            pw = Eigen::VectorXf::Ones(NP) / NP;
        }
        steps++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {ms, steps};
}

// ============================================================
// CUDA Kernels (same as particle_filter.cu)
// ============================================================
struct Observation { float d, lx, ly; };

__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void predict_and_weight_kernel(
    float* px, float* pw, float u_v, float u_omega,
    float rsim_0, float rsim_1, const Observation* obs, int n_obs,
    float Q, curandState* rng_states, int np)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;
    curandState rng = rng_states[ip];
    float ud_v = u_v + curand_normal(&rng) * rsim_0;
    float ud_o = u_omega + curand_normal(&rng) * rsim_1;
    float x = px[0*np+ip], y = px[1*np+ip], yaw = px[2*np+ip];
    x += DT * cosf(yaw) * ud_v;
    y += DT * sinf(yaw) * ud_v;
    yaw += DT * ud_o;
    px[0*np+ip] = x; px[1*np+ip] = y; px[2*np+ip] = yaw; px[3*np+ip] = ud_v;

    float w = pw[ip];
    float sigma = sqrtf(Q);
    float inv_c = 1.0f / sqrtf(2.0f * PI * Q);
    for (int i = 0; i < n_obs; i++) {
        float dx = x - obs[i].lx, dy = y - obs[i].ly;
        float dz = sqrtf(dx*dx+dy*dy) - obs[i].d;
        w *= inv_c * expf(-dz*dz/(2.0f*Q));
    }
    pw[ip] = w;
    rng_states[ip] = rng;
}

__global__ void normalize_weights_kernel(float* pw, int np) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float val = 0;
    for (int i = tid; i < np; i += blockDim.x) val += pw[i];
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    float total = sdata[0];
    if (total < 1e-30f) total = 1e-30f;
    for (int i = tid; i < np; i += blockDim.x) pw[i] /= total;
}

__global__ void cumsum_kernel(const float* pw, float* wcum, int np) {
    wcum[0] = pw[0];
    for (int i = 1; i < np; i++) wcum[i] = wcum[i-1] + pw[i];
}

__global__ void resample_kernel(const float* px_in, float* px_out,
                                const float* wcum, float base_step,
                                float rand_offset, int np) {
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;
    float target = base_step * ip + rand_offset;
    int lo = 0, hi = np - 1;
    while (lo < hi) { int mid = (lo+hi)/2; if (wcum[mid] < target) lo = mid+1; else hi = mid; }
    for (int d = 0; d < 4; d++) px_out[d*np+ip] = px_in[d*np+lo];
}

template <int NP>
CPUResult run_cuda_pf() {
    const int np = NP;
    const int threads = 256;
    const int blocks = (np + threads - 1) / threads;

    float *d_px, *d_px_tmp, *d_pw, *d_wcum;
    Observation *d_obs;
    curandState *d_rng;
    CUDA_CHECK(cudaMalloc(&d_px, 4*np*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_px_tmp, 4*np*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pw, np*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wcum, np*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obs, 16*sizeof(Observation)));
    CUDA_CHECK(cudaMalloc(&d_rng, np*sizeof(curandState)));

    CUDA_CHECK(cudaMemset(d_px, 0, 4*np*sizeof(float)));
    std::vector<float> pw_init(np, 1.0f/np);
    CUDA_CHECK(cudaMemcpy(d_pw, pw_init.data(), np*sizeof(float), cudaMemcpyHostToDevice));
    init_curand_kernel<<<blocks, threads>>>(d_rng, 42ULL, np);
    CUDA_CHECK(cudaDeviceSynchronize());

    Eigen::Vector2f u; u << 1.0f, 0.1f;
    Eigen::Matrix<float, 4, 2> RFID;
    RFID << 10.0f, 0.0f, 10.0f, 10.0f, 0.0f, 15.0f, -5.0f, 20.0f;
    Eigen::Vector4f xTrue = Eigen::Vector4f::Zero();
    float Q = 0.01f;
    Eigen::Matrix2f Rsim = Eigen::Matrix2f::Identity();
    Rsim(0,0) = 1.0f; Rsim(1,1) = (30.0f/180.0f*PI)*(30.0f/180.0f*PI);
    float Qsim = 0.04f;

    std::mt19937 gen(42);
    std::normal_distribution<float> gd(0, 1);
    std::uniform_real_distribution<float> ud(0, 1);

    float time_val = 0.0f;
    int steps = 0;
    std::vector<float> h_pw(np);

    auto t0 = std::chrono::high_resolution_clock::now();

    while (time_val <= SIM_TIME) {
        time_val += DT;
        // motion model for ground truth on host
        Eigen::Vector4f xt = xTrue;
        xt(0) += DT * std::cos(xt(2)) * u(0);
        xt(1) += DT * std::sin(xt(2)) * u(0);
        xt(2) += DT * u(1);
        xt(3) = u(0);
        xTrue = xt;

        std::vector<Observation> z_host;
        for (int i = 0; i < RFID.rows(); i++) {
            float dx = xTrue(0)-RFID(i,0), dy = xTrue(1)-RFID(i,1);
            float d = std::sqrt(dx*dx+dy*dy);
            if (d <= MAX_RANGE) z_host.push_back({d+gd(gen)*Qsim, RFID(i,0), RFID(i,1)});
        }
        int n_obs = (int)z_host.size();
        if (n_obs > 0)
            CUDA_CHECK(cudaMemcpy(d_obs, z_host.data(), n_obs*sizeof(Observation), cudaMemcpyHostToDevice));

        predict_and_weight_kernel<<<blocks, threads>>>(
            d_px, d_pw, u(0), u(1), Rsim(0,0), Rsim(1,1),
            d_obs, n_obs, Q, d_rng, np);
        normalize_weights_kernel<<<1, threads, threads*sizeof(float)>>>(d_pw, np);

        // resampling check
        CUDA_CHECK(cudaMemcpy(h_pw.data(), d_pw, np*sizeof(float), cudaMemcpyDeviceToHost));
        float neff_d = 0;
        for (int i = 0; i < np; i++) neff_d += h_pw[i]*h_pw[i];
        if (1.0f/neff_d < np/2) {
            cumsum_kernel<<<1,1>>>(d_pw, d_wcum, np);
            resample_kernel<<<blocks, threads>>>(d_px, d_px_tmp, d_wcum, 1.0f/np, ud(gen)/np, np);
            CUDA_CHECK(cudaMemcpy(d_px, d_px_tmp, 4*np*sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_pw, pw_init.data(), np*sizeof(float), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        steps++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cudaFree(d_px); cudaFree(d_px_tmp); cudaFree(d_pw);
    cudaFree(d_wcum); cudaFree(d_obs); cudaFree(d_rng);
    return {ms, steps};
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Particle Filter Benchmark: CPU vs CUDA" << std::endl;
    std::cout << "========================================" << std::endl;

    // Warmup CUDA
    { float *tmp; cudaMalloc(&tmp, 1024); cudaFree(tmp); cudaDeviceSynchronize(); }

    auto run_bench = [](const char* label, int np, double cpu_ms, int cpu_steps, double cuda_ms, int cuda_steps) {
        printf("\n  NP = %d  (%d steps)\n", np, cpu_steps);
        printf("    CPU:  %8.2f ms\n", cpu_ms);
        printf("    CUDA: %8.2f ms\n", cuda_ms);
        printf("    Speedup: %.2fx\n", cpu_ms / cuda_ms);
    };

    {
        auto cpu = run_cpu_pf<100>();
        auto cuda = run_cuda_pf<100>();
        run_bench("PF", 100, cpu.total_ms, cpu.steps, cuda.total_ms, cuda.steps);
    }
    {
        auto cpu = run_cpu_pf<1000>();
        auto cuda = run_cuda_pf<1000>();
        run_bench("PF", 1000, cpu.total_ms, cpu.steps, cuda.total_ms, cuda.steps);
    }
    {
        auto cpu = run_cpu_pf<5000>();
        auto cuda = run_cuda_pf<5000>();
        run_bench("PF", 5000, cpu.total_ms, cpu.steps, cuda.total_ms, cuda.steps);
    }
    {
        auto cpu = run_cpu_pf<10000>();
        auto cuda = run_cuda_pf<10000>();
        run_bench("PF", 10000, cpu.total_ms, cpu.steps, cuda.total_ms, cuda.steps);
    }

    std::cout << "\n========================================" << std::endl;
    return 0;
}
