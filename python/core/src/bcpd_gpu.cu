// GPU BCPD non-rigid registration library (shared by demo + Python bindings).
#include "cudarobotics/bcpd_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cuda_check.cuh"

namespace cudarobotics {
namespace {

__global__ void estep_denom_kernel(const float* __restrict__ X, int N,
                                   const float* __restrict__ P, int M,
                                   float inv2s2, float c_out, float* __restrict__ D) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= N) return;
    float xn0 = X[n*3+0], xn1 = X[n*3+1], xn2 = X[n*3+2];
    float s = 0.f;
    for (int m = 0; m < M; ++m) {
        float dx = xn0-P[m*3+0], dy = xn1-P[m*3+1], dz = xn2-P[m*3+2];
        s += __expf(-(dx*dx+dy*dy+dz*dz)*inv2s2);
    }
    D[n] = s + c_out;
}

__global__ void estep_moments_kernel(const float* __restrict__ X, int N,
                                     const float* __restrict__ P, int M,
                                     const float* __restrict__ D, float inv2s2,
                                     float* __restrict__ P1, float* __restrict__ PX) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    if (m >= M) return;
    float p0 = P[m*3+0], p1 = P[m*3+1], p2 = P[m*3+2];
    float s0 = 0.f, sx = 0.f, sy = 0.f, sz = 0.f;
    for (int n = 0; n < N; ++n) {
        float dx = X[n*3+0]-p0, dy = X[n*3+1]-p1, dz = X[n*3+2]-p2;
        float k = __expf(-(dx*dx+dy*dy+dz*dz)*inv2s2) / D[n];
        s0 += k; sx += k*X[n*3+0]; sy += k*X[n*3+1]; sz += k*X[n*3+2];
    }
    P1[m] = s0;
    PX[m*3+0] = sx; PX[m*3+1] = sy; PX[m*3+2] = sz;
}

__global__ void estep_pt1_kernel(const float* __restrict__ X, int N,
                                 const float* __restrict__ P, int M,
                                 const float* __restrict__ D, float inv2s2,
                                 float* __restrict__ Pt1) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= N) return;
    float xn0 = X[n*3+0], xn1 = X[n*3+1], xn2 = X[n*3+2];
    float s = 0.f;
    for (int m = 0; m < M; ++m) {
        float dx = xn0-P[m*3+0], dy = xn1-P[m*3+1], dz = xn2-P[m*3+2];
        s += __expf(-(dx*dx+dy*dy+dz*dz)*inv2s2);
    }
    Pt1[n] = s / D[n];
}

static bool chol_solve(std::vector<double>& A, int M, std::vector<double>& B) {
    std::vector<double> L((size_t)M*M, 0.0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[(size_t)i*M+j];
            for (int k = 0; k < j; ++k) s -= L[(size_t)i*M+k]*L[(size_t)j*M+k];
            if (i == j) { if (s <= 0) return false; L[(size_t)i*M+i] = std::sqrt(s); }
            else L[(size_t)i*M+j] = s / L[(size_t)j*M+j];
        }
    }
    for (int c = 0; c < 3; ++c) {
        std::vector<double> y(M);
        for (int i = 0; i < M; ++i) {
            double s = B[(size_t)i*3+c];
            for (int k = 0; k < i; ++k) s -= L[(size_t)i*M+k]*y[k];
            y[i] = s / L[(size_t)i*M+i];
        }
        for (int i = M-1; i >= 0; --i) {
            double s = y[i];
            for (int k = i+1; k < M; ++k) s -= L[(size_t)k*M+i]*B[(size_t)k*3+c];
            B[(size_t)i*3+c] = s / L[(size_t)i*M+i];
        }
    }
    return true;
}

static float mean_surface_distance(
  const std::vector<float>& X, const std::vector<float>& P)
{
    int N = X.size()/3, M = P.size()/3;
    double sum = 0;
    for (int m = 0; m < M; ++m) {
        float best = 1e9f;
        for (int n = 0; n < N; ++n) {
            float dx = P[m*3]-X[n*3], dy = P[m*3+1]-X[n*3+1], dz = P[m*3+2]-X[n*3+2];
            float d = dx*dx + dy*dy + dz*dz;
            if (d < best) best = d;
        }
        sum += std::sqrt(best);
    }
    return M > 0 ? (float)(sum / M) : 0.f;
}

struct InternalResult {
  std::vector<float> P;
  int iters;
  float final_sigma;
  float mean_surface_distance;
};

static InternalResult bcpd(
  const std::vector<float>& X, const std::vector<float>& Y, const BcpdParams& params)
{
    int N = X.size()/3, M = Y.size()/3;

    std::vector<double> G((size_t)M*M);
    float inv2b2 = 1.f/(2.f*params.beta*params.beta);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j) {
            float dx = Y[i*3+0]-Y[j*3+0], dy = Y[i*3+1]-Y[j*3+1], dz = Y[i*3+2]-Y[j*3+2];
            G[(size_t)i*M+j] = std::exp(-(dx*dx+dy*dy+dz*dz)*inv2b2);
        }

    double s2 = 0;
    {
        double mx[3] = {0, 0, 0}, my[3] = {0, 0, 0};
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < 3; ++k) mx[k] += X[n*3+k];
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < 3; ++k) my[k] += Y[m*3+k];
        for (int k = 0; k < 3; ++k) { mx[k] /= N; my[k] /= M; }
        double vx = 0;
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < 3; ++k) {
                double d = X[n*3+k]-mx[k];
                vx += d*d;
            }
        s2 = vx/(3.0*N) + 0.3;
    }
    float w_out = 0.1f;

    float *dX, *dP, *dD, *dP1, *dPX, *dPt1;
    CUDA_CHECK(cudaMalloc(&dX, N*3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, X.data(), N*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dP, M*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dD, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPt1, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dP1, M*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPX, M*3*sizeof(float)));

    std::vector<float> P = Y;
    CUDA_CHECK(cudaMemcpy(dP, P.data(), M*3*sizeof(float), cudaMemcpyHostToDevice));
    InternalResult res; res.iters = 0;

    std::vector<float> hP1(M), hPX(M*3), hPt1(N);
    for (int it = 0; it < params.max_iters; ++it) {
        float inv2s2 = 1.f/(2.f*(float)s2);
        float c_out = std::pow(2.f*3.14159265f*(float)s2, 1.5f)
                    * (w_out/(1.f-w_out)) * (float)M/(float)N;
        estep_denom_kernel<<<(N+255)/256, 256>>>(dX, N, dP, M, inv2s2, c_out, dD);
        estep_moments_kernel<<<(M+255)/256, 256>>>(dX, N, dP, M, dD, inv2s2, dP1, dPX);
        estep_pt1_kernel<<<(N+255)/256, 256>>>(dX, N, dP, M, dD, inv2s2, dPt1);
        CUDA_CHECK(cudaMemcpy(hP1.data(), dP1, M*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hPX.data(), dPX, M*3*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hPt1.data(), dPt1, N*sizeof(float), cudaMemcpyDeviceToHost));

        double Np = 0;
        for (int m = 0; m < M; ++m) Np += hP1[m];

        std::vector<double> A((size_t)M*M), B((size_t)M*3);
        for (int i = 0; i < M; ++i) {
            double p1 = std::max(1e-6f, hP1[i]);
            for (int j = 0; j < M; ++j) A[(size_t)i*M+j] = G[(size_t)i*M+j];
            A[(size_t)i*M+i] += params.lambda * s2 / p1;
            for (int k = 0; k < 3; ++k) B[(size_t)i*3+k] = hPX[i*3+k]/p1 - Y[i*3+k];
        }
        if (!chol_solve(A, M, B)) break;
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < 3; ++k) {
                double v = 0;
                for (int j = 0; j < M; ++j) v += G[(size_t)i*M+j]*B[(size_t)j*3+k];
                P[i*3+k] = Y[i*3+k] + (float)v;
            }
        CUDA_CHECK(cudaMemcpy(dP, P.data(), M*3*sizeof(float), cudaMemcpyHostToDevice));

        double term1 = 0, term2 = 0, term3 = 0;
        for (int n = 0; n < N; ++n) {
            double r2 = X[n*3]*X[n*3] + X[n*3+1]*X[n*3+1] + X[n*3+2]*X[n*3+2];
            term1 += hPt1[n]*r2;
        }
        for (int m = 0; m < M; ++m) {
            term2 += hPX[m*3]*P[m*3] + hPX[m*3+1]*P[m*3+1] + hPX[m*3+2]*P[m*3+2];
            double pp = P[m*3]*P[m*3] + P[m*3+1]*P[m*3+1] + P[m*3+2]*P[m*3+2];
            term3 += hP1[m]*pp;
        }
        s2 = (term1 - 2*term2 + term3) / (3.0*Np);
        if (s2 < 1e-5) s2 = 1e-5;
        ++res.iters;
    }

    res.P = P;
    res.final_sigma = std::sqrt((float)s2);
    res.mean_surface_distance = mean_surface_distance(X, P);
    cudaFree(dX); cudaFree(dP); cudaFree(dD); cudaFree(dPt1);
    cudaFree(dP1); cudaFree(dPX);
    return res;
}

BcpdResult runBcpd(
  const BcpdParams& params,
  const float* target_xyz, int num_target,
  const float* source_xyz, int num_source)
{
    if (num_target <= 0 || num_source <= 0) {
        throw std::invalid_argument("BcpdGpu: point clouds must be non-empty");
    }
    std::vector<float> X(target_xyz, target_xyz + num_target * 3);
    std::vector<float> Y(source_xyz, source_xyz + num_source * 3);
    InternalResult internal = bcpd(X, Y, params);
    BcpdResult out;
    out.deformed_xyz = std::move(internal.P);
    out.iterations = internal.iters;
    out.final_sigma = internal.final_sigma;
    out.mean_surface_distance = internal.mean_surface_distance;
    return out;
}

}  // namespace

struct BcpdGpu::Impl
{
  BcpdParams params;
};

BcpdGpu::BcpdGpu(const BcpdParams& params)
: impl_(std::make_unique<Impl>())
{
  impl_->params = params;
}

BcpdGpu::~BcpdGpu() = default;

BcpdResult BcpdGpu::registerClouds(
  const float* target_xyz, int num_target,
  const float* source_xyz, int num_source)
{
  return runBcpd(impl_->params, target_xyz, num_target, source_xyz, num_source);
}

}  // namespace cudarobotics
