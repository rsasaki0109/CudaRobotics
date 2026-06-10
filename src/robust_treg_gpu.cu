// GPU robust Student's-t point-to-point registration library.
#include "cudarobotics/robust_treg_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cuda_check.cuh"

namespace cudarobotics {
namespace {

struct Mat3 { float m[9]; };
struct Pose { Mat3 R; float t[3]; };

static inline void mat3_vec(const Mat3& R, const float* v, float* o) {
    o[0] = R.m[0]*v[0] + R.m[1]*v[1] + R.m[2]*v[2];
    o[1] = R.m[3]*v[0] + R.m[4]*v[1] + R.m[5]*v[2];
    o[2] = R.m[6]*v[0] + R.m[7]*v[1] + R.m[8]*v[2];
}
static inline void pose_apply(const Pose& T, const float* y, float* p) {
    mat3_vec(T.R, y, p);
    p[0] += T.t[0]; p[1] += T.t[1]; p[2] += T.t[2];
}
static inline Mat3 mat3_mul(const Mat3& A, const Mat3& B) {
    Mat3 C;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float s = 0.f;
            for (int k = 0; k < 3; ++k) s += A.m[i*3+k] * B.m[k*3+j];
            C.m[i*3+j] = s;
        }
    return C;
}
static inline Mat3 so3_exp(const float* w) {
    float th = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    Mat3 R;
    if (th < 1e-9f) { R = {1,0,0, 0,1,0, 0,0,1}; return R; }
    float a = w[0]/th, b = w[1]/th, c = w[2]/th;
    float s = std::sin(th), co = std::cos(th), v = 1.f - co;
    R.m[0] = a*a*v + co;   R.m[1] = a*b*v - c*s;  R.m[2] = a*c*v + b*s;
    R.m[3] = a*b*v + c*s;  R.m[4] = b*b*v + co;   R.m[5] = b*c*v - a*s;
    R.m[6] = a*c*v - b*s;  R.m[7] = b*c*v + a*s;  R.m[8] = c*c*v + co;
    return R;
}
static inline Pose se3_exp(const float* xi) {
    const float* v = xi; const float* w = xi + 3;
    Pose T; T.R = so3_exp(w);
    float th = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    float Vm[9];
    if (th < 1e-6f) { for (int i = 0; i < 9; ++i) Vm[i] = (i%4==0)?1.f:0.f; }
    else {
        float A = (1.f - std::cos(th)) / (th*th);
        float B = (th - std::sin(th)) / (th*th*th);
        float wx[9] = {0,-w[2],w[1], w[2],0,-w[0], -w[1],w[0],0}, wx2[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                float s = 0;
                for (int k = 0; k < 3; ++k) s += wx[i*3+k]*wx[k*3+j];
                wx2[i*3+j] = s;
            }
        for (int i = 0; i < 9; ++i) Vm[i] = ((i%4==0)?1.f:0.f) + A*wx[i] + B*wx2[i];
    }
    T.t[0] = Vm[0]*v[0] + Vm[1]*v[1] + Vm[2]*v[2];
    T.t[1] = Vm[3]*v[0] + Vm[4]*v[1] + Vm[5]*v[2];
    T.t[2] = Vm[6]*v[0] + Vm[7]*v[1] + Vm[8]*v[2];
    return T;
}
static inline Pose pose_mul(const Pose& A, const Pose& B) {
    Pose C; C.R = mat3_mul(A.R, B.R);
    float Rt[3]; mat3_vec(A.R, B.t, Rt);
    for (int k = 0; k < 3; ++k) C.t[k] = Rt[k] + A.t[k];
    return C;
}

__device__ __forceinline__ float comp_K(float d2, float s2, float nu) {
    return __powf(1.f + d2/(nu*s2), -0.5f*(nu+3.f));
}

__global__ void estep_denom_kernel(const float* __restrict__ P, int M,
                                   const float* __restrict__ X, int N,
                                   float s2, float nu, float c_out,
                                   float* __restrict__ Dn) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= N) return;
    float x0 = X[n*3], x1 = X[n*3+1], x2 = X[n*3+2], s = 0.f;
    for (int m = 0; m < M; ++m) {
        float dx = P[m*3]-x0, dy = P[m*3+1]-x1, dz = P[m*3+2]-x2;
        s += comp_K(dx*dx+dy*dy+dz*dz, s2, nu);
    }
    Dn[n] = s + c_out;
}

__global__ void estep_moments_kernel(const float* __restrict__ P, int M,
                                     const float* __restrict__ X, int N,
                                     const float* __restrict__ Dn,
                                     float s2, float nu,
                                     float* __restrict__ MU, float* __restrict__ Wm) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    if (m >= M) return;
    float p0 = P[m*3], p1 = P[m*3+1], p2 = P[m*3+2];
    float sw = 0, sx = 0, sy = 0, sz = 0;
    for (int n = 0; n < N; ++n) {
        float dx = p0-X[n*3], dy = p1-X[n*3+1], dz = p2-X[n*3+2];
        float d2 = dx*dx+dy*dy+dz*dz;
        float pmn = comp_K(d2, s2, nu)/Dn[n];
        float u = (nu+3.f)/(nu + d2/s2);
        float w = pmn*u;
        sw += w; sx += w*X[n*3]; sy += w*X[n*3+1]; sz += w*X[n*3+2];
    }
    Wm[m] = sw;
    float inv = 1.f/(sw+1e-20f);
    MU[m*3] = sx*inv; MU[m*3+1] = sy*inv; MU[m*3+2] = sz*inv;
}

__global__ void transform_kernel(const float* __restrict__ Y, int M,
                                 const float* __restrict__ R, const float* __restrict__ t,
                                 float* __restrict__ P) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= M) return;
    float y0 = Y[j*3], y1 = Y[j*3+1], y2 = Y[j*3+2];
    P[j*3] = R[0]*y0+R[1]*y1+R[2]*y2+t[0];
    P[j*3+1] = R[3]*y0+R[4]*y1+R[5]*y2+t[1];
    P[j*3+2] = R[6]*y0+R[7]*y1+R[8]*y2+t[2];
}

__global__ void mstep_kernel(const float* __restrict__ P, const float* __restrict__ MU,
                             const float* __restrict__ W, int M,
                             float* __restrict__ Hg) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= M) return;
    float w = W[j];
    if (w < 1e-12f) return;
    float px = P[j*3], py = P[j*3+1], pz = P[j*3+2];
    float rx = px-MU[j*3], ry = py-MU[j*3+1], rz = pz-MU[j*3+2];
    float J[18] = {1,0,0,0,pz,-py, 0,1,0,-pz,0,px, 0,0,1,py,-px,0};
    float Hl[21]; int c = 0;
    for (int a = 0; a < 6; ++a)
        for (int b = a; b < 6; ++b) {
            float s = J[0*6+a]*J[0*6+b]+J[1*6+a]*J[1*6+b]+J[2*6+a]*J[2*6+b];
            Hl[c++] = w*s;
        }
    float gl[6];
    for (int a = 0; a < 6; ++a) gl[a] = w*(J[0*6+a]*rx+J[1*6+a]*ry+J[2*6+a]*rz);
    for (int k = 0; k < 21; ++k) atomicAdd(&Hg[k], Hl[k]);
    for (int k = 0; k < 6; ++k) atomicAdd(&Hg[21+k], gl[k]);
    atomicAdd(&Hg[27], w*(rx*rx+ry*ry+rz*rz));
    atomicAdd(&Hg[28], w);
}

static bool solve6(const float* Hut, const float* gg, float* d) {
    float H[36]; int c = 0;
    for (int a = 0; a < 6; ++a)
        for (int b = a; b < 6; ++b) { H[a*6+b] = H[b*6+a] = Hut[c++]; }
    for (int i = 0; i < 6; ++i) H[i*6+i] += 1e-6f;
    float L[36] = {0};
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j) {
            float s = H[i*6+j];
            for (int k = 0; k < j; ++k) s -= L[i*6+k]*L[j*6+k];
            if (i == j) { if (s <= 0) return false; L[i*6+i] = std::sqrt(s); }
            else L[i*6+j] = s / L[j*6+j];
        }
    float y[6];
    for (int i = 0; i < 6; ++i) {
        float s = -gg[i];
        for (int k = 0; k < i; ++k) s -= L[i*6+k]*y[k];
        y[i] = s / L[i*6+i];
    }
    for (int i = 5; i >= 0; --i) {
        float s = y[i];
        for (int k = i+1; k < 6; ++k) s -= L[k*6+i]*d[k];
        d[i] = s / L[i*6+i];
    }
    return true;
}

static float trimmed_nn_rmse(const std::vector<float>& X, const std::vector<float>& Y, const Pose& T) {
    int N = X.size()/3, M = Y.size()/3;
    std::vector<float> ds;
    ds.reserve(M);
    for (int j = 0; j < M; ++j) {
        float y[3] = {Y[j*3], Y[j*3+1], Y[j*3+2]}, p[3];
        pose_apply(T, y, p);
        float best = 1e30f;
        for (int i = 0; i < N; ++i) {
            float dx = p[0]-X[i*3], dy = p[1]-X[i*3+1], dz = p[2]-X[i*3+2];
            float d = dx*dx + dy*dy + dz*dz;
            if (d < best) best = d;
        }
        ds.push_back(std::sqrt(best));
    }
    std::sort(ds.begin(), ds.end());
    int keep = std::max(1, (int)(0.6f * ds.size()));
    double sum = 0;
    for (int i = 0; i < keep; ++i) sum += ds[i];
    return (float)(sum / keep);
}

struct InternalResult { Pose T; int iters; float final_rmse; };

static InternalResult robust_treg_register(
  const std::vector<float>& X, const std::vector<float>& Y,
  Pose T0, const RobustTregParams& params)
{
    int N = X.size()/3, M = Y.size()/3;
    float *dX, *dY, *dP, *dR, *dt, *dDn, *dMU, *dWm, *dHg;
    CUDA_CHECK(cudaMalloc(&dX, N*3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, X.data(), N*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dY, M*3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dY, Y.data(), M*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dP, M*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dR, 9*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dt, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDn, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dMU, M*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWm, M*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dHg, 29*sizeof(float)));

    Pose T = T0;
    InternalResult res; res.iters = 0; res.final_rmse = 0.f;
    const float nu = params.nu;
    const float c_out = params.outlier_fraction;

    const float sigmas[] = {
        0.6f, 0.5f, 0.4f, 0.32f, 0.27f, 0.25f,
        0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f};
    for (float sig : sigmas) {
        float s2 = sig*sig;
        for (int outer = 0; outer < params.outer_iters_per_sigma; ++outer) {
            CUDA_CHECK(cudaMemcpy(dR, T.R.m, 9*sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dt, T.t, 3*sizeof(float), cudaMemcpyHostToDevice));
            transform_kernel<<<(M+255)/256, 256>>>(dY, M, dR, dt, dP);
            estep_denom_kernel<<<(N+255)/256, 256>>>(dP, M, dX, N, s2, nu, c_out, dDn);
            estep_moments_kernel<<<(M+255)/256, 256>>>(dP, M, dX, N, dDn, s2, nu, dMU, dWm);
            for (int gn = 0; gn < params.gn_iters; ++gn) {
                if (gn > 0) {
                    CUDA_CHECK(cudaMemcpy(dR, T.R.m, 9*sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(dt, T.t, 3*sizeof(float), cudaMemcpyHostToDevice));
                    transform_kernel<<<(M+255)/256, 256>>>(dY, M, dR, dt, dP);
                }
                CUDA_CHECK(cudaMemset(dHg, 0, 29*sizeof(float)));
                mstep_kernel<<<(M+255)/256, 256>>>(dP, dMU, dWm, M, dHg);
                float Hg[29];
                CUDA_CHECK(cudaMemcpy(Hg, dHg, 29*sizeof(float), cudaMemcpyDeviceToHost));
                float d[6];
                if (!solve6(Hg, Hg+21, d)) break;
                T = pose_mul(se3_exp(d), T);
            }
            ++res.iters;
        }
    }

    res.T = T;
    res.final_rmse = trimmed_nn_rmse(X, Y, T);
    cudaFree(dX); cudaFree(dY); cudaFree(dP); cudaFree(dR); cudaFree(dt);
    cudaFree(dDn); cudaFree(dMU); cudaFree(dWm); cudaFree(dHg);
    return res;
}

RobustTregResult runRobustTreg(
  const RobustTregParams& params,
  const float* target_xyz, int num_target,
  const float* source_xyz, int num_source,
  const float* init_rotation,
  const float* init_translation)
{
    if (num_target <= 0 || num_source <= 0) {
        throw std::invalid_argument("RobustTregGpu: point clouds must be non-empty");
    }
    std::vector<float> X(target_xyz, target_xyz + num_target * 3);
    std::vector<float> Y(source_xyz, source_xyz + num_source * 3);

    Pose T0;
    if (init_rotation) {
        for (int i = 0; i < 9; ++i) T0.R.m[i] = init_rotation[i];
    } else {
        T0.R = {1,0,0, 0,1,0, 0,0,1};
    }
    if (init_translation) {
        for (int k = 0; k < 3; ++k) T0.t[k] = init_translation[k];
    } else {
        T0.t[0] = T0.t[1] = T0.t[2] = 0.f;
    }

    InternalResult internal = robust_treg_register(X, Y, T0, params);
    RobustTregResult out;
    for (int i = 0; i < 9; ++i) out.rotation[i] = internal.T.R.m[i];
    for (int k = 0; k < 3; ++k) out.translation[k] = internal.T.t[k];
    out.iterations = internal.iters;
    out.final_rmse = internal.final_rmse;
    return out;
}

}  // namespace

struct RobustTregGpu::Impl
{
  RobustTregParams params;
};

RobustTregGpu::RobustTregGpu(const RobustTregParams& params)
: impl_(std::make_unique<Impl>())
{
  impl_->params = params;
}

RobustTregGpu::~RobustTregGpu() = default;

RobustTregResult RobustTregGpu::registerClouds(
  const float* target_xyz, int num_target,
  const float* source_xyz, int num_source,
  const float* init_rotation,
  const float* init_translation)
{
  return runRobustTreg(
    impl_->params, target_xyz, num_target, source_xyz, num_source,
    init_rotation, init_translation);
}

}  // namespace cudarobotics
