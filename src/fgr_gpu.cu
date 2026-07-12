// GPU Fast Global Registration library (shared by demo + Python bindings).
#include "cudarobotics/fgr_gpu.hpp"

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

static const int KNN = 18;
static const int FDIM = 33;

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
    float r = detB*0.5f;
    r = fminf(1.f, fmaxf(-1.f, r));
    float phi = acosf(r)/3.f;
    float e0 = q + 2.f*p*cosf(phi+2.0943951f);
    float r0[3] = {c00-e0, c01, c02}, r1[3] = {c01, c11-e0, c12}, r2[3] = {c02, c12, c22-e0};
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

__global__ void knn_kernel(const float* __restrict__ P, int N, float cx, float cy, float cz,
                           float* __restrict__ NRM, int* __restrict__ IDX) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    float xi = P[i*3], yi = P[i*3+1], zi = P[i*3+2];
    float dk[KNN]; int ik[KNN];
    for (int a = 0; a < KNN; ++a) { dk[a] = 1e30f; ik[a] = -1; }
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float dx = P[j*3]-xi, dy = P[j*3+1]-yi, dz = P[j*3+2]-zi;
        float d = dx*dx + dy*dy + dz*dz;
        if (d < dk[KNN-1]) {
            int p = KNN-1;
            while (p > 0 && dk[p-1] > d) { dk[p] = dk[p-1]; ik[p] = ik[p-1]; --p; }
            dk[p] = d; ik[p] = j;
        }
    }
    for (int a = 0; a < KNN; ++a) IDX[i*KNN+a] = ik[a];
    float mx = xi, my = yi, mz = zi; int cnt = 1;
    for (int a = 0; a < KNN; ++a) {
        int j = ik[a]; if (j < 0) continue;
        mx += P[j*3]; my += P[j*3+1]; mz += P[j*3+2]; ++cnt;
    }
    float inv = 1.f/cnt; mx *= inv; my *= inv; mz *= inv;
    float C[6] = {0,0,0,0,0,0};
    auto acc = [&](float px, float py, float pz) {
        float ex = px-mx, ey = py-my, ez = pz-mz;
        C[0] += ex*ex; C[1] += ex*ey; C[2] += ex*ez;
        C[3] += ey*ey; C[4] += ey*ez; C[5] += ez*ez;
    };
    acc(xi, yi, zi);
    for (int a = 0; a < KNN; ++a) {
        int j = ik[a]; if (j < 0) continue;
        acc(P[j*3], P[j*3+1], P[j*3+2]);
    }
    float n[3]; sym3_smallest_evec(C, n);
    float ox = xi-cx, oy = yi-cy, oz = zi-cz;
    if (n[0]*ox + n[1]*oy + n[2]*oz < 0) { n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2]; }
    NRM[i*3] = n[0]; NRM[i*3+1] = n[1]; NRM[i*3+2] = n[2];
}

__global__ void spfh_kernel(const float* __restrict__ P, const float* __restrict__ NRM,
                            const int* __restrict__ IDX, int N, float* __restrict__ SPFH) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    float* h = SPFH + i*FDIM;
    for (int b = 0; b < FDIM; ++b) h[b] = 0;
    float pi[3] = {P[i*3], P[i*3+1], P[i*3+2]};
    float ni[3] = {NRM[i*3], NRM[i*3+1], NRM[i*3+2]};
    int cnt = 0;
    for (int a = 0; a < KNN; ++a) {
        int j = IDX[i*KNN+a]; if (j < 0) continue;
        float d[3] = {P[j*3]-pi[0], P[j*3+1]-pi[1], P[j*3+2]-pi[2]};
        float dn = sqrtf(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
        if (dn < 1e-9f) continue;
        float dh[3] = {d[0]/dn, d[1]/dn, d[2]/dn};
        float u[3] = {ni[0], ni[1], ni[2]};
        float v[3] = {dh[1]*u[2]-dh[2]*u[1], dh[2]*u[0]-dh[0]*u[2], dh[0]*u[1]-dh[1]*u[0]};
        float vn = sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        if (vn < 1e-9f) continue;
        v[0] /= vn; v[1] /= vn; v[2] /= vn;
        float w[3] = {u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]};
        float nj[3] = {NRM[j*3], NRM[j*3+1], NRM[j*3+2]};
        float alpha = v[0]*nj[0] + v[1]*nj[1] + v[2]*nj[2];
        float phi   = u[0]*dh[0] + u[1]*dh[1] + u[2]*dh[2];
        float theta = atan2f(w[0]*nj[0]+w[1]*nj[1]+w[2]*nj[2],
                             u[0]*nj[0]+u[1]*nj[1]+u[2]*nj[2]);
        int ba = min(10, max(0, (int)((alpha+1.f)*0.5f*11.f)));
        int bp = min(10, max(0, (int)((phi  +1.f)*0.5f*11.f)));
        int bt = min(10, max(0, (int)((theta+3.14159265f)/6.2831853f*11.f)));
        h[ba] += 1.f; h[11+bp] += 1.f; h[22+bt] += 1.f; ++cnt;
    }
    if (cnt > 0) { float inv = 100.f/cnt; for (int b = 0; b < FDIM; ++b) h[b] *= inv; }
}

__global__ void fpfh_kernel(const float* __restrict__ P, const int* __restrict__ IDX,
                            const float* __restrict__ SPFH, int N, float* __restrict__ FPFH) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    float f[FDIM];
    for (int b = 0; b < FDIM; ++b) f[b] = SPFH[i*FDIM+b];
    float pi[3] = {P[i*3], P[i*3+1], P[i*3+2]};
    float wsum = 0;
    for (int a = 0; a < KNN; ++a) {
        int j = IDX[i*KNN+a]; if (j < 0) continue;
        float dx = P[j*3]-pi[0], dy = P[j*3+1]-pi[1], dz = P[j*3+2]-pi[2];
        float dn = sqrtf(dx*dx+dy*dy+dz*dz);
        if (dn < 1e-9f) continue;
        float wgt = 1.f/dn; wsum += wgt;
        for (int b = 0; b < FDIM; ++b) f[b] += wgt*SPFH[j*FDIM+b];
    }
    if (wsum > 0) {
        for (int b = 0; b < FDIM; ++b) f[b] = SPFH[i*FDIM+b] + f[b]/wsum;
    }
    float s = 0; for (int b = 0; b < FDIM; ++b) s += f[b];
    if (s > 0) { float inv = 1.f/s; for (int b = 0; b < FDIM; ++b) f[b] *= inv; }
    for (int b = 0; b < FDIM; ++b) FPFH[i*FDIM+b] = f[b];
}

__global__ void match_kernel(const float* __restrict__ FS, int M, const float* __restrict__ FT,
                             int N, int* __restrict__ corr) {
    constexpr int TILE = 32;
    __shared__ float ftile[TILE * FDIM];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    bool active = i < M;
    const float* fs = active ? FS + i*FDIM : FS;
    float best = 1e30f; int bj = -1;
    for (int base = 0; base < N; base += TILE) {
        int count = min(TILE, N - base);
        for (int q = threadIdx.x; q < count * FDIM; q += blockDim.x)
            ftile[q] = FT[base * FDIM + q];
        __syncthreads();
        if (active) for (int j = 0; j < count; ++j) {
            float d = 0;
            for (int b = 0; b < FDIM; ++b) { float e = fs[b]-ftile[j*FDIM+b]; d += e*e; }
            if (d < best) { best = d; bj = base + j; }
        }
        __syncthreads();
    }
    if (active) corr[i] = bj;
}

__global__ void fgr_gn_kernel(const float* __restrict__ Sw, const float* __restrict__ X,
                              const int* __restrict__ corr, int M, float mu,
                              float* __restrict__ Hg) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= M) return;
    int j = corr[i]; if (j < 0) return;
    float px = Sw[i*3], py = Sw[i*3+1], pz = Sw[i*3+2];
    float rx = px-X[j*3], ry = py-X[j*3+1], rz = pz-X[j*3+2];
    float r2 = rx*rx + ry*ry + rz*rz;
    float gm = mu/(mu+r2); float w = gm*gm;
    float J[18] = {1,0,0,0,pz,-py, 0,1,0,-pz,0,px, 0,0,1,py,-px,0};
    float Hl[21]; int c = 0;
    for (int a = 0; a < 6; ++a)
        for (int b = a; b < 6; ++b) {
            float s = J[a]*J[b] + J[6+a]*J[6+b] + J[12+a]*J[12+b];
            Hl[c++] = w*s;
        }
    float gl[6];
    for (int a = 0; a < 6; ++a) gl[a] = w*(J[a]*rx + J[6+a]*ry + J[12+a]*rz);
    for (int k = 0; k < 21; ++k) atomicAdd(&Hg[k], Hl[k]);
    for (int k = 0; k < 6; ++k) atomicAdd(&Hg[21+k], gl[k]);
    atomicAdd(&Hg[27], w*r2);
    atomicAdd(&Hg[28], w);
}

__global__ void transform_kernel(const float* __restrict__ Y, int M,
                                 const float* __restrict__ R, const float* __restrict__ t,
                                 float* __restrict__ P) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= M) return;
    float y0 = Y[j*3], y1 = Y[j*3+1], y2 = Y[j*3+2];
    P[j*3]   = R[0]*y0 + R[1]*y1 + R[2]*y2 + t[0];
    P[j*3+1] = R[3]*y0 + R[4]*y1 + R[5]*y2 + t[1];
    P[j*3+2] = R[6]*y0 + R[7]*y1 + R[8]*y2 + t[2];
}

static bool solve6(const float* Hut, const float* g, float* d) {
    float H[36]; int c = 0;
    for (int a = 0; a < 6; ++a)
        for (int b = a; b < 6; ++b) { H[a*6+b] = H[b*6+a] = Hut[c++]; }
    for (int i = 0; i < 6; ++i) H[i*6+i] += 1e-5f;
    float L[36] = {0};
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j) {
            float s = H[i*6+j];
            for (int k = 0; k < j; ++k) s -= L[i*6+k]*L[j*6+k];
            if (i == j) { if (s <= 0) return false; L[i*6+i] = sqrtf(s); }
            else L[i*6+j] = s / L[j*6+j];
        }
    float y[6];
    for (int i = 0; i < 6; ++i) {
        float s = -g[i];
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

static void compute_fpfh(float* dP, int N, const float* cen, float*& dNRM, float*& dFPFH) {
    int* dIDX; float* dSPFH;
    CUDA_CHECK(cudaMalloc(&dNRM, N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dIDX, N*KNN*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dSPFH, N*FDIM*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dFPFH, N*FDIM*sizeof(float)));
    knn_kernel<<<(N+127)/128, 128>>>(dP, N, cen[0], cen[1], cen[2], dNRM, dIDX);
    spfh_kernel<<<(N+127)/128, 128>>>(dP, dNRM, dIDX, N, dSPFH);
    fpfh_kernel<<<(N+127)/128, 128>>>(dP, dIDX, dSPFH, N, dFPFH);
    cudaFree(dIDX); cudaFree(dSPFH);
}

static float combined_bbox_diameter(const std::vector<float>& X, const std::vector<float>& Y) {
    float lo[3] = {1e9f, 1e9f, 1e9f}, hi[3] = {-1e9f, -1e9f, -1e9f};
    auto acc = [&](const std::vector<float>& P) {
        for (size_t i = 0; i < P.size()/3; ++i)
            for (int k = 0; k < 3; ++k) {
                lo[k] = std::min(lo[k], P[i*3+k]);
                hi[k] = std::max(hi[k], P[i*3+k]);
            }
    };
    acc(X); acc(Y);
    float dx = hi[0]-lo[0], dy = hi[1]-lo[1], dz = hi[2]-lo[2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

static float surf_residual(const std::vector<float>& X, const std::vector<float>& Y, const Pose& T) {
    int N = X.size()/3, M = Y.size()/3;
    std::vector<float> ds;
    for (int j = 0; j < M; j += 3) {
        float y[3] = {Y[j*3], Y[j*3+1], Y[j*3+2]}, p[3];
        pose_apply(T, y, p);
        float best = 1e30f;
        for (int i = 0; i < N; i += 2) {
            float dx = p[0]-X[i*3], dy = p[1]-X[i*3+1], dz = p[2]-X[i*3+2];
            float d = dx*dx + dy*dy + dz*dz;
            if (d < best) best = d;
        }
        ds.push_back(std::sqrt(best));
    }
    std::sort(ds.begin(), ds.end());
    int keep = std::max(1, (int)(0.6f*ds.size()));
    double s = 0;
    for (int i = 0; i < keep; ++i) s += ds[i];
    return (float)(s / keep);
}

struct InternalResult { Pose T; int iters; float final_rmse; };

static InternalResult fgr_align(
  const std::vector<float>& X, const std::vector<float>& Y, const FgrParams& params)
{
    int N = X.size()/3, M = Y.size()/3;
    float cen[3] = {0, 0, 0};
    float *dX, *dY;
    CUDA_CHECK(cudaMalloc(&dX, N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, M*3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, X.data(), N*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY, Y.data(), M*3*sizeof(float), cudaMemcpyHostToDevice));
    float *dNX, *dFX, *dNY, *dFY;
    compute_fpfh(dX, N, cen, dNX, dFX);
    compute_fpfh(dY, M, cen, dNY, dFY);
    int* dcorr;
    CUDA_CHECK(cudaMalloc(&dcorr, M*sizeof(int)));
    match_kernel<<<(M+127)/128, 128>>>(dFY, M, dFX, N, dcorr);
    float *dSw, *dR, *dt, *dHg;
    CUDA_CHECK(cudaMalloc(&dSw, M*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dR, 9*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dt, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dHg, 29*sizeof(float)));

    Pose T;
    T.R = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    T.t[0] = T.t[1] = T.t[2] = 0.f;
    float diam = combined_bbox_diameter(X, Y);
    float mu = diam*diam;
    InternalResult res; res.iters = 0; res.final_rmse = 0.f;

    for (int level = 0; level < params.gn_levels; ++level) {
        for (int gn = 0; gn < params.gn_steps_per_level; ++gn) {
            CUDA_CHECK(cudaMemcpy(dR, T.R.m, 9*sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dt, T.t, 3*sizeof(float), cudaMemcpyHostToDevice));
            transform_kernel<<<(M+127)/128, 128>>>(dY, M, dR, dt, dSw);
            CUDA_CHECK(cudaMemset(dHg, 0, 29*sizeof(float)));
            fgr_gn_kernel<<<(M+127)/128, 128>>>(dSw, dX, dcorr, M, mu, dHg);
            float Hg[29];
            CUDA_CHECK(cudaMemcpy(Hg, dHg, 29*sizeof(float), cudaMemcpyDeviceToHost));
            float d[6];
            if (!solve6(Hg, Hg+21, d)) break;
            T = pose_mul(se3_exp(d), T);
            ++res.iters;
        }
        mu *= params.mu_decay;
        if (mu < params.min_mu) mu = params.min_mu;
    }
    res.T = T;
    res.final_rmse = surf_residual(X, Y, T);
    cudaFree(dX); cudaFree(dY); cudaFree(dNX); cudaFree(dFX);
    cudaFree(dNY); cudaFree(dFY); cudaFree(dcorr);
    cudaFree(dSw); cudaFree(dR); cudaFree(dt); cudaFree(dHg);
    return res;
}

FgrResult runFgr(
  const FgrParams& params,
  const float* target_xyz, int num_target,
  const float* source_xyz, int num_source)
{
    if (num_target <= 0 || num_source <= 0) {
        throw std::invalid_argument("FgrGpu: point clouds must be non-empty");
    }
    std::vector<float> X(target_xyz, target_xyz + num_target * 3);
    std::vector<float> Y(source_xyz, source_xyz + num_source * 3);
    InternalResult internal = fgr_align(X, Y, params);
    FgrResult out;
    for (int i = 0; i < 9; ++i) out.rotation[i] = internal.T.R.m[i];
    for (int k = 0; k < 3; ++k) out.translation[k] = internal.T.t[k];
    out.iterations = internal.iters;
    out.final_rmse = internal.final_rmse;
    return out;
}

}  // namespace

struct FgrGpu::Impl
{
  FgrParams params;
};

FgrGpu::FgrGpu(const FgrParams& params)
: impl_(std::make_unique<Impl>())
{
  impl_->params = params;
}

FgrGpu::~FgrGpu() = default;

FgrResult FgrGpu::registerClouds(
  const float* target_xyz, int num_target,
  const float* source_xyz, int num_source)
{
  return runFgr(impl_->params, target_xyz, num_target, source_xyz, num_source);
}

}  // namespace cudarobotics
