// GPU robust Student's-t point-to-plane registration library.
#include "cudarobotics/robust_p2plane_gpu.hpp"

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

__device__ static void sym3_smallest_evec(const float C[6], float n[3]) {
    float c00 = C[0], c01 = C[1], c02 = C[2], c11 = C[3], c12 = C[4], c22 = C[5];
    float p1 = c01*c01 + c02*c02 + c12*c12;
    if (p1 < 1e-20f) { n[0] = 0; n[1] = 0; n[2] = 1; return; }
    float q = (c00+c11+c22)/3.f;
    float b00 = c00-q, b11 = c11-q, b22 = c22-q;
    float p2 = b00*b00 + b11*b11 + b22*b22 + 2.f*p1;
    float p = sqrtf(p2/6.f), i_p = 1.f/p;
    float d00 = b00*i_p, d01 = c01*i_p, d02 = c02*i_p;
    float d11 = b11*i_p, d12 = c12*i_p, d22 = b22*i_p;
    float detB = d00*(d11*d22-d12*d12) - d01*(d01*d22-d12*d02) + d02*(d01*d12-d11*d02);
    float r = detB*0.5f; r = fminf(1.f, fmaxf(-1.f, r));
    float phi = acosf(r)/3.f;
    float e0 = q + 2.f*p*cosf(phi+2.0943951f);
    float r0[3] = {c00-e0, c01, c02};
    float r1[3] = {c01, c11-e0, c12};
    float r2[3] = {c02, c12, c22-e0};
    float x0[3] = {r0[1]*r1[2]-r0[2]*r1[1], r0[2]*r1[0]-r0[0]*r1[2], r0[0]*r1[1]-r0[1]*r1[0]};
    float x1[3] = {r0[1]*r2[2]-r0[2]*r2[1], r0[2]*r2[0]-r0[0]*r2[2], r0[0]*r2[1]-r0[1]*r2[0]};
    float x2[3] = {r1[1]*r2[2]-r1[2]*r2[1], r1[2]*r2[0]-r1[0]*r2[2], r1[0]*r2[1]-r1[1]*r2[0]};
    float n0 = x0[0]*x0[0]+x0[1]*x0[1]+x0[2]*x0[2];
    float n1 = x1[0]*x1[0]+x1[1]*x1[1]+x1[2]*x1[2];
    float n2 = x2[0]*x2[0]+x2[1]*x2[1]+x2[2]*x2[2];
    const float* best = x0; float bn = n0;
    if (n1 > bn) { best = x1; bn = n1; }
    if (n2 > bn) { best = x2; bn = n2; }
    float inv = rsqrtf(bn+1e-20f);
    n[0] = best[0]*inv; n[1] = best[1]*inv; n[2] = best[2]*inv;
}

__global__ void knn_normal_kernel(const float* __restrict__ X, int N, int K,
                                    float cx, float cy, float cz,
                                    float* __restrict__ NX) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    float xi = X[i*3], yi = X[i*3+1], zi = X[i*3+2];
    const int KMAX = 20;
    float dk[KMAX]; int ik[KMAX];
    int kk = K < KMAX ? K : KMAX;
    for (int a = 0; a < kk; ++a) { dk[a] = 1e30f; ik[a] = -1; }
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float dx = X[j*3]-xi, dy = X[j*3+1]-yi, dz = X[j*3+2]-zi;
        float d = dx*dx + dy*dy + dz*dz;
        if (d < dk[kk-1]) {
            int p = kk-1;
            while (p > 0 && dk[p-1] > d) { dk[p] = dk[p-1]; ik[p] = ik[p-1]; --p; }
            dk[p] = d; ik[p] = j;
        }
    }
    float mx = xi, my = yi, mz = zi;
    int cnt = 1;
    for (int a = 0; a < kk; ++a) {
        int j = ik[a];
        if (j < 0) continue;
        mx += X[j*3]; my += X[j*3+1]; mz += X[j*3+2]; ++cnt;
    }
    float invc = 1.f/cnt;
    mx *= invc; my *= invc; mz *= invc;
    float C[6] = {0,0,0,0,0,0};
    auto acc = [&](float px, float py, float pz) {
        float ex = px-mx, ey = py-my, ez = pz-mz;
        C[0] += ex*ex; C[1] += ex*ey; C[2] += ex*ez;
        C[3] += ey*ey; C[4] += ey*ez; C[5] += ez*ez;
    };
    acc(xi, yi, zi);
    for (int a = 0; a < kk; ++a) {
        int j = ik[a];
        if (j < 0) continue;
        acc(X[j*3], X[j*3+1], X[j*3+2]);
    }
    float nrm[3];
    sym3_smallest_evec(C, nrm);
    float ox = xi-cx, oy = yi-cy, oz = zi-cz;
    if (nrm[0]*ox + nrm[1]*oy + nrm[2]*oz < 0) {
        nrm[0] = -nrm[0]; nrm[1] = -nrm[1]; nrm[2] = -nrm[2];
    }
    NX[i*3] = nrm[0]; NX[i*3+1] = nrm[1]; NX[i*3+2] = nrm[2];
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
                                     const float* __restrict__ X,
                                     const float* __restrict__ NX, int N,
                                     const float* __restrict__ Dn,
                                     float s2, float nu,
                                     float* __restrict__ MU,
                                     float* __restrict__ NRM,
                                     float* __restrict__ Wm) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    if (m >= M) return;
    float p0 = P[m*3], p1 = P[m*3+1], p2 = P[m*3+2];
    float sw = 0, sx = 0, sy = 0, sz = 0, nx = 0, ny = 0, nz = 0;
    for (int n = 0; n < N; ++n) {
        float dx = p0-X[n*3], dy = p1-X[n*3+1], dz = p2-X[n*3+2];
        float d2 = dx*dx+dy*dy+dz*dz;
        float pmn = comp_K(d2, s2, nu)/Dn[n];
        float u = (nu+3.f)/(nu + d2/s2);
        float w = pmn*u;
        sw += w;
        sx += w*X[n*3]; sy += w*X[n*3+1]; sz += w*X[n*3+2];
        nx += w*NX[n*3]; ny += w*NX[n*3+1]; nz += w*NX[n*3+2];
    }
    Wm[m] = sw;
    float inv = 1.f/(sw+1e-20f);
    MU[m*3] = sx*inv; MU[m*3+1] = sy*inv; MU[m*3+2] = sz*inv;
    float nn = rsqrtf(nx*nx+ny*ny+nz*nz+1e-20f);
    NRM[m*3] = nx*nn; NRM[m*3+1] = ny*nn; NRM[m*3+2] = nz*nn;
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

__global__ void mstep_plane_kernel(const float* __restrict__ P,
                                   const float* __restrict__ MU,
                                   const float* __restrict__ NRM,
                                   const float* __restrict__ W, int M,
                                   float* __restrict__ Hg) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= M) return;
    float w = W[j];
    if (w < 1e-12f) return;
    float px = P[j*3], py = P[j*3+1], pz = P[j*3+2];
    float ex = px-MU[j*3], ey = py-MU[j*3+1], ez = pz-MU[j*3+2];
    float J[18] = {1,0,0,0,pz,-py, 0,1,0,-pz,0,px, 0,0,1,py,-px,0};
    float nx = NRM[j*3], ny = NRM[j*3+1], nz = NRM[j*3+2];
    float rs = nx*ex + ny*ey + nz*ez;
    float jp[6];
    for (int a = 0; a < 6; ++a) jp[a] = nx*J[a] + ny*J[6+a] + nz*J[12+a];
    float Hl[21]; int c = 0;
    for (int a = 0; a < 6; ++a)
        for (int b = a; b < 6; ++b) Hl[c++] = w*jp[a]*jp[b];
    for (int k = 0; k < 21; ++k) atomicAdd(&Hg[k], Hl[k]);
    for (int a = 0; a < 6; ++a) atomicAdd(&Hg[21+a], w*jp[a]*rs);
    atomicAdd(&Hg[27], w*rs*rs);
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

static void compute_target_normals(const std::vector<float>& X, int knn_k, std::vector<float>& NX) {
    int N = X.size()/3;
    NX.resize(N*3);
    float c[3] = {0,0,0};
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < 3; ++k) c[k] += X[i*3+k];
    for (int k = 0; k < 3; ++k) c[k] /= N;
    float *dX, *dNX;
    CUDA_CHECK(cudaMalloc(&dX, N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dNX, N*3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, X.data(), N*3*sizeof(float), cudaMemcpyHostToDevice));
    knn_normal_kernel<<<(N+127)/128, 128>>>(dX, N, knn_k, c[0], c[1], c[2], dNX);
    CUDA_CHECK(cudaMemcpy(NX.data(), dNX, N*3*sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dX);
    cudaFree(dNX);
}

struct InternalResult { Pose T; int iters; float final_rmse; };

static InternalResult robust_p2plane_register(
  const std::vector<float>& X, const std::vector<float>& Y,
  Pose T0, const RobustP2PlaneParams& params)
{
    int N = X.size()/3, M = Y.size()/3;
    std::vector<float> NX;
    compute_target_normals(X, params.knn_k, NX);

    float *dX, *dNX, *dY, *dP, *dR, *dt, *dDn, *dMU, *dNRM, *dWm, *dHg;
    CUDA_CHECK(cudaMalloc(&dX, N*3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, X.data(), N*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dNX, N*3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dNX, NX.data(), N*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dY, M*3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dY, Y.data(), M*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dP, M*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dR, 9*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dt, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDn, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dMU, M*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dNRM, M*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWm, M*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dHg, 29*sizeof(float)));

    Pose T = T0;
    InternalResult res; res.iters = 0; res.final_rmse = 0.f;
    const float nu = params.nu;
    const float c_out = params.outlier_fraction;

    const float floor_sigma = 0.45f;
    std::vector<float> sigmas = {0.8f, 0.6f, 0.5f};
    for (int i = 0; i < 6; ++i) sigmas.push_back(floor_sigma);

    for (float sig : sigmas) {
        float s2 = sig*sig;
        for (int outer = 0; outer < params.outer_iters_per_sigma; ++outer) {
            CUDA_CHECK(cudaMemcpy(dR, T.R.m, 9*sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dt, T.t, 3*sizeof(float), cudaMemcpyHostToDevice));
            transform_kernel<<<(M+255)/256, 256>>>(dY, M, dR, dt, dP);
            estep_denom_kernel<<<(N+255)/256, 256>>>(dP, M, dX, N, s2, nu, c_out, dDn);
            estep_moments_kernel<<<(M+255)/256, 256>>>(
                dP, M, dX, dNX, N, dDn, s2, nu, dMU, dNRM, dWm);
            for (int gn = 0; gn < params.gn_iters; ++gn) {
                if (gn > 0) {
                    CUDA_CHECK(cudaMemcpy(dR, T.R.m, 9*sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(dt, T.t, 3*sizeof(float), cudaMemcpyHostToDevice));
                    transform_kernel<<<(M+255)/256, 256>>>(dY, M, dR, dt, dP);
                }
                CUDA_CHECK(cudaMemset(dHg, 0, 29*sizeof(float)));
                mstep_plane_kernel<<<(M+255)/256, 256>>>(dP, dMU, dNRM, dWm, M, dHg);
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
    cudaFree(dX); cudaFree(dNX); cudaFree(dY); cudaFree(dP);
    cudaFree(dR); cudaFree(dt); cudaFree(dDn); cudaFree(dMU);
    cudaFree(dNRM); cudaFree(dWm); cudaFree(dHg);
    return res;
}

RobustP2PlaneResult runRobustP2Plane(
  const RobustP2PlaneParams& params,
  const float* target_xyz, int num_target,
  const float* source_xyz, int num_source,
  const float* init_rotation,
  const float* init_translation)
{
    if (num_target <= 0 || num_source <= 0) {
        throw std::invalid_argument("RobustP2PlaneGpu: point clouds must be non-empty");
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

    InternalResult internal = robust_p2plane_register(X, Y, T0, params);
    RobustP2PlaneResult out;
    for (int i = 0; i < 9; ++i) out.rotation[i] = internal.T.R.m[i];
    for (int k = 0; k < 3; ++k) out.translation[k] = internal.T.t[k];
    out.iterations = internal.iters;
    out.final_rmse = internal.final_rmse;
    return out;
}

}  // namespace

struct RobustP2PlaneGpu::Impl
{
  RobustP2PlaneParams params;
};

RobustP2PlaneGpu::RobustP2PlaneGpu(const RobustP2PlaneParams& params)
: impl_(std::make_unique<Impl>())
{
  impl_->params = params;
}

RobustP2PlaneGpu::~RobustP2PlaneGpu() = default;

RobustP2PlaneResult RobustP2PlaneGpu::registerClouds(
  const float* target_xyz, int num_target,
  const float* source_xyz, int num_source,
  const float* init_rotation,
  const float* init_translation)
{
  return runRobustP2Plane(
    impl_->params, target_xyz, num_target, source_xyz, num_source,
    init_rotation, init_translation);
}

}  // namespace cudarobotics
