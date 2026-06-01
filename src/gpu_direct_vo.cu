// gpu_direct_vo.cu
//
// GPU dense direct (photometric) visual odometry - the front-end that DSO /
// LSD-SLAM popularised: recover camera motion by minimising the photometric
// error over ALL pixels directly, no feature extraction.  This is the dense
// counterpart to the repo's sparse Lucas-Kanade demo (gpu_lk_optical_flow):
// instead of one independent flow per feature, the whole image votes into a
// single shared pose via Gauss-Newton.
//
// Demo: a planar SE(2) warp (tx, ty, theta) between a reference image I and a
// moved frame J.  Each Gauss-Newton iteration:
//
//   r(x) = J(W(x;p)) - I(x)            photometric residual
//   Jrow = grad J . dW/dp              3-vector
//   H += Jrow Jrow^T (3x3),  b += Jrow r,   dp = -H^-1 b,  p += dp
//
// Map onto the canonical idiom: **one thread = one pixel** computing its
// residual + Jacobian row, then a block-level shared-memory reduction sums them
// into the shared 3x3 system (one global atomic per block, not per pixel).
//
// The reduction is a cross-thread sum, so - unlike the per-feature LK demo -
// this is NOT bit-identical: floating-point addition is non-associative and the
// GPU's block-tree + atomic order differs from the CPU's sequential sum.  We
// accumulate in double precision, so the per-term values match and only the
// summation order differs: the CPU and GPU recover the SAME pose to ~1e-9, and
// both converge to the ground-truth SE(2).  Honest framing: near-identical, not
// bit-exact.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ----------------------------------------------------------------- constants
#define IMG_W 512
#define IMG_H 512
#define BLK   128
static const int    N_PIX   = IMG_W * IMG_H;
static const int    N_ITERS = 25;
static const double CXC     = IMG_W * 0.5;
static const double CYC     = IMG_H * 0.5;

// ground-truth SE(2) the estimator must recover
static const double GT_TX  = 6.0;
static const double GT_TY  = -4.0;
static const double GT_TH  = 0.020;

static const int PANEL_W = 760;
static const int PANEL_H = 330;

__host__ __device__ static inline int idx2(int x, int y) { return y * IMG_W + x; }

// ----------------------------------------------------- synthetic image source
static inline float hashf(int x, int y) {
    unsigned int h = ((unsigned int)x * 73856093u) ^ ((unsigned int)y * 19349663u);
    h = (h ^ (h >> 13)) * 1274126177u;
    return (float)((h ^ (h >> 16)) & 0xffff) / 65535.0f;
}
// smooth, well-conditioned texture (meaningful gradients -> sub-pixel alignment)
static float scene(int x, int y) {
    // smooth band-limited texture: analytic gradients are well-approximated by
    // finite differences, so dense alignment converges to the true pose without
    // the bias an unfiltered per-pixel noise term would inject.
    float v = 128.0f
            + 55.0f * std::sin(0.018f * x + 0.011f * y)
            + 42.0f * std::sin(0.022f * x - 0.016f * y)
            + 35.0f * std::sin(0.040f * x + 0.050f * y)
            + 22.0f * std::sin(0.071f * x - 0.063f * y);
    (void)hashf;
    if (v < 0) v = 0; if (v > 255) v = 255;
    return v;
}

__host__ __device__ static inline float sample(const float* img, double u, double v) {
    if (u < 0) u = 0; if (u > IMG_W - 1) u = IMG_W - 1;
    if (v < 0) v = 0; if (v > IMG_H - 1) v = IMG_H - 1;
    int x0 = (int)u, y0 = (int)v;
    int x1 = x0 + 1 < IMG_W ? x0 + 1 : x0;
    int y1 = y0 + 1 < IMG_H ? y0 + 1 : y0;
    double ax = u - x0, ay = v - y0;
    float a = img[idx2(x0, y0)], b = img[idx2(x1, y0)];
    float c = img[idx2(x0, y1)], d = img[idx2(x1, y1)];
    double top = a + ax * (b - a), bot = c + ax * (d - c);
    return (float)(top + ay * (bot - top));
}

// SE(2) warp of reference pixel (x,y) under p=(tx,ty,theta), about image centre
__host__ __device__ static inline void warp(int x, int y, const double* p,
                                            double& u, double& v) {
    double c = cos(p[2]), s = sin(p[2]);
    double dx = x - CXC, dy = y - CYC;
    u = CXC + c * dx - s * dy + p[0];
    v = CYC + s * dx + c * dy + p[1];
}

// per-pixel contribution into out[11] = {H6(6), b3(3), ssd, cnt}; CPU path
__host__ __device__ static inline void pixel_terms(int x, int y, const float* I,
        const float* J, const double* p, double* out) {
    double u, v;
    warp(x, y, p, u, v);
    if (u < 1 || u > IMG_W - 2 || v < 1 || v > IMG_H - 2) return;
    double r = (double)sample(J, u, v) - (double)I[idx2(x, y)];
    double gx = 0.5 * (sample(J, u + 1, v) - sample(J, u - 1, v));
    double gy = 0.5 * (sample(J, u, v + 1) - sample(J, u, v - 1));
    double c = cos(p[2]), s = sin(p[2]);
    double ddx = x - CXC, ddy = y - CYC;
    double dwx = -s * ddx - c * ddy, dwy = c * ddx - s * ddy;
    double j0 = gx, j1 = gy, j2 = gx * dwx + gy * dwy;
    out[0] += j0*j0; out[1] += j0*j1; out[2] += j0*j2;
    out[3] += j1*j1; out[4] += j1*j2; out[5] += j2*j2;
    out[6] += j0*r;  out[7] += j1*r;  out[8] += j2*r;
    out[9] += r*r;   out[10] += 1.0;
}

// double atomic add via CAS (works on all compute capabilities)
__device__ static inline double atomicAddD(double* addr, double val) {
    unsigned long long* a = (unsigned long long*)addr;
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        old = atomicCAS(a, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// one thread = one pixel -> block-reduce 11 partials -> one atomic flush/block
__global__ void gn_kernel(const float* I, const float* J, const double* p, double* out) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    double a[11];
    for (int k = 0; k < 11; ++k) a[k] = 0.0;
    if (i < N_PIX) {
        int x = i % IMG_W, y = i / IMG_W;
        double u, v; warp(x, y, p, u, v);
        if (u >= 1 && u <= IMG_W - 2 && v >= 1 && v <= IMG_H - 2) {
            double r = (double)sample(J, u, v) - (double)I[idx2(x, y)];
            double gx = 0.5 * (sample(J, u + 1, v) - sample(J, u - 1, v));
            double gy = 0.5 * (sample(J, u, v + 1) - sample(J, u, v - 1));
            double c = cos(p[2]), s = sin(p[2]);
            double ddx = x - CXC, ddy = y - CYC;
            double dwx = -s * ddx - c * ddy, dwy = c * ddx - s * ddy;
            double j0 = gx, j1 = gy, j2 = gx * dwx + gy * dwy;
            a[0]=j0*j0; a[1]=j0*j1; a[2]=j0*j2; a[3]=j1*j1; a[4]=j1*j2; a[5]=j2*j2;
            a[6]=j0*r;  a[7]=j1*r;  a[8]=j2*r;  a[9]=r*r;   a[10]=1.0;
        }
    }
    __shared__ double sh[BLK * 11];
    for (int k = 0; k < 11; ++k) sh[tid * 11 + k] = a[k];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            for (int k = 0; k < 11; ++k) sh[tid * 11 + k] += sh[(tid + s) * 11 + k];
        __syncthreads();
    }
    if (tid == 0)
        for (int k = 0; k < 11; ++k) atomicAddD(&out[k], sh[k]);
}

// solve 3x3 symmetric H dp = -b  (H6 = [h00,h01,h02,h11,h12,h22])
static inline bool solve3(const double* H6, const double* b3, double* dp) {
    double a00 = H6[0] + 1e-3, a01 = H6[1], a02 = H6[2];
    double a11 = H6[3] + 1e-3, a12 = H6[4], a22 = H6[5] + 1e-3;
    double det = a00 * (a11 * a22 - a12 * a12)
               - a01 * (a01 * a22 - a12 * a02)
               + a02 * (a01 * a12 - a11 * a02);
    if (std::fabs(det) < 1e-12) return false;
    double i00 =  (a11 * a22 - a12 * a12), i01 = -(a01 * a22 - a02 * a12);
    double i02 =  (a01 * a12 - a02 * a11), i11 =  (a00 * a22 - a02 * a02);
    double i12 = -(a00 * a12 - a02 * a01), i22 =  (a00 * a11 - a01 * a01);
    double g0 = -b3[0], g1 = -b3[1], g2 = -b3[2];
    dp[0] = (i00*g0 + i01*g1 + i02*g2) / det;
    dp[1] = (i01*g0 + i11*g1 + i12*g2) / det;
    dp[2] = (i02*g0 + i12*g1 + i22*g2) / det;
    return true;
}

// --------------------------------------------------------------- CPU pipeline
static void vo_cpu(const std::vector<float>& I, const std::vector<float>& J,
                   double* p, std::vector<double>& rms_hist) {
    p[0] = p[1] = p[2] = 0.0; rms_hist.clear();
    for (int it = 0; it < N_ITERS; ++it) {
        double o[11] = {0,0,0,0,0,0,0,0,0,0,0};
        for (int y = 0; y < IMG_H; ++y)
            for (int x = 0; x < IMG_W; ++x)
                pixel_terms(x, y, I.data(), J.data(), p, o);
        rms_hist.push_back(o[10] > 0 ? std::sqrt(o[9] / o[10]) : 0.0);
        double dp[3];
        if (!solve3(o, o + 6, dp)) break;
        p[0] += dp[0]; p[1] += dp[1]; p[2] += dp[2];
    }
}

// ------------------------------------------------------------- visualisation
static cv::Mat gray3(const std::vector<float>& img) {
    cv::Mat g(IMG_H, IMG_W, CV_8UC1);
    for (int i = 0; i < N_PIX; ++i) g.data[i] = (uint8_t)std::min(255.f, std::max(0.f, img[i]));
    cv::Mat c; cv::cvtColor(g, c, cv::COLOR_GRAY2BGR); return c;
}
static cv::Mat residual_heat(const std::vector<float>& I, const std::vector<float>& J,
                             const double* p) {
    cv::Mat g(IMG_H, IMG_W, CV_8UC1);
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
            double u, v; warp(x, y, p, u, v);
            float r = std::fabs(sample(J.data(), u, v) - I[idx2(x, y)]);
            g.at<uint8_t>(y, x) = (uint8_t)std::min(255.f, r * 3.0f);
        }
    cv::Mat c; cv::applyColorMap(g, c, cv::COLORMAP_INFERNO); return c;
}
static void draw(cv::Mat& out, const cv::Mat& Iimg, const cv::Mat& Jimg,
                 const cv::Mat& res, const char* l1, const char* l2, const char* l3) {
    out = cv::Mat(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(28, 28, 32));
    const int IW = 230, IH = 230, Y = 56;
    int xs[3] = {18, 265, 512};
    cv::Mat a, b, c;
    cv::resize(Iimg, a, cv::Size(IW, IH)); cv::resize(Jimg, b, cv::Size(IW, IH));
    cv::resize(res,  c, cv::Size(IW, IH));
    a.copyTo(out(cv::Rect(xs[0], Y, IW, IH)));
    b.copyTo(out(cv::Rect(xs[1], Y, IW, IH)));
    c.copyTo(out(cv::Rect(xs[2], Y, IW, IH)));
    const char* lab[3] = {"reference I", "current J", "photometric residual"};
    for (int k = 0; k < 3; ++k)
        cv::putText(out, lab[k], {xs[k], Y - 8}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {220,220,220}, 1, cv::LINE_AA);
    cv::putText(out, l1, {16, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {235,235,235}, 1, cv::LINE_AA);
    cv::putText(out, l2, {16, PANEL_H - 34}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {180,220,255}, 1, cv::LINE_AA);
    cv::putText(out, l3, {16, PANEL_H - 12}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {180,255,180}, 1, cv::LINE_AA);
}

// ===========================================================================
int main() {
    std::vector<float> I(N_PIX), J(N_PIX);
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) I[idx2(x, y)] = scene(x, y);
    // J is the reference seen after moving by GT: J(x) = I(Winv_GT(x)), the
    // exact SE(2) inverse warp, so minimising I(x)-J(W(x;p)) recovers p = GT.
    double c = std::cos(GT_TH), s = std::sin(GT_TH);
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
            double qx = (x - CXC) - GT_TX, qy = (y - CYC) - GT_TY;
            double rx = CXC + c * qx + s * qy;
            double ry = CYC - s * qx + c * qy;
            J[idx2(x, y)] = sample(I.data(), rx, ry);
        }

    // ---------- CPU
    double p_cpu[3]; std::vector<double> rms_cpu;
    auto t0 = std::chrono::high_resolution_clock::now();
    vo_cpu(I, J, p_cpu, rms_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------- GPU
    float *d_I, *d_J; double *d_p, *d_out;
    CUDA_CHECK(cudaMalloc(&d_I, N_PIX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_J, N_PIX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p, 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, 11 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_I, I.data(), N_PIX * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_J, J.data(), N_PIX * sizeof(float), cudaMemcpyHostToDevice));
    int grid = (N_PIX + BLK - 1) / BLK;

    double p_gpu[3]; std::vector<double> rms_gpu;
    auto run_gpu = [&](float* ms_out) {
        double p[3] = {0,0,0}; rms_gpu.clear();
        cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        for (int it = 0; it < N_ITERS; ++it) {
            double z[11] = {0,0,0,0,0,0,0,0,0,0,0};
            CUDA_CHECK(cudaMemcpy(d_p, p, 3 * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_out, z, 11 * sizeof(double), cudaMemcpyHostToDevice));
            gn_kernel<<<grid, BLK>>>(d_I, d_J, d_p, d_out);
            double o[11];
            CUDA_CHECK(cudaMemcpy(o, d_out, 11 * sizeof(double), cudaMemcpyDeviceToHost));
            rms_gpu.push_back(o[10] > 0 ? std::sqrt(o[9] / o[10]) : 0.0);
            double dp[3];
            if (!solve3(o, o + 6, dp)) break;
            p[0] += dp[0]; p[1] += dp[1]; p[2] += dp[2];
        }
        CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
        if (ms_out) CUDA_CHECK(cudaEventElapsedTime(ms_out, e0, e1));
        p_gpu[0] = p[0]; p_gpu[1] = p[1]; p_gpu[2] = p[2];
    };
    float gpu_ms = 0.0f;
    run_gpu(nullptr);
    run_gpu(&gpu_ms);

    double dpose = std::max(std::max(std::fabs(p_cpu[0]-p_gpu[0]), std::fabs(p_cpu[1]-p_gpu[1])),
                            std::fabs(p_cpu[2]-p_gpu[2]));
    double gt_terr = std::sqrt((p_gpu[0]-GT_TX)*(p_gpu[0]-GT_TX) + (p_gpu[1]-GT_TY)*(p_gpu[1]-GT_TY));
    double gt_aerr = std::fabs(p_gpu[2]-GT_TH);
    double speedup = cpu_ms / gpu_ms;

    std::printf("CPU %.2f ms, GPU %.3f ms  -> %.0fx  (%d GN iters)\n", cpu_ms, gpu_ms, speedup, N_ITERS);
    std::printf("pose CPU (%.4f, %.4f, %.5f)  GPU (%.4f, %.4f, %.5f)  max|diff| %.2e\n",
                p_cpu[0], p_cpu[1], p_cpu[2], p_gpu[0], p_gpu[1], p_gpu[2], dpose);
    std::printf("ground truth (%.4f, %.4f, %.5f)  -> trans err %.4f px, rot err %.2e rad\n",
                GT_TX, GT_TY, GT_TH, gt_terr, gt_aerr);
    std::printf("photometric RMS: %.3f -> %.3f\n",
                rms_gpu.empty() ? 0.0 : rms_gpu.front(), rms_gpu.empty() ? 0.0 : rms_gpu.back());

    // ---------- animation
    if (system("mkdir -p tmp") != 0) std::fprintf(stderr, "warning: mkdir tmp failed\n");
    cv::VideoWriter video("tmp/gpu_direct_vo.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          6, cv::Size(PANEL_W, PANEL_H));
    cv::Mat Iimg = gray3(I), Jimg = gray3(J);
    double p[3] = {0,0,0};
    for (int it = 0; it <= N_ITERS; ++it) {
        double o[11] = {0,0,0,0,0,0,0,0,0,0,0};
        for (int y = 0; y < IMG_H; ++y)
            for (int x = 0; x < IMG_W; ++x)
                pixel_terms(x, y, I.data(), J.data(), p, o);
        double rms = o[10] > 0 ? std::sqrt(o[9] / o[10]) : 0.0;
        cv::Mat res = residual_heat(I, J, p);
        char l1[200], l2[200], l3[200];
        std::snprintf(l1, sizeof(l1),
                      "GPU direct (photometric) visual odometry (one thread = one pixel)  %dx%d  SE(2)",
                      IMG_W, IMG_H);
        std::snprintf(l2, sizeof(l2),
                      "GN iter %d/%d   pose (%.2f, %.2f, %.4f)   photometric RMS %.2f",
                      it, N_ITERS, p[0], p[1], p[2], rms);
        std::snprintf(l3, sizeof(l3),
                      "CPU %.0f ms vs GPU %.2f ms -> %.0fx   CPU/GPU pose agree to %.0e   GT err %.3f px",
                      cpu_ms, gpu_ms, speedup, dpose, gt_terr);
        cv::Mat img; draw(img, Iimg, Jimg, res, l1, l2, l3);
        for (int r = 0; r < (it == N_ITERS ? 8 : 1); ++r) video.write(img);
        if (it < N_ITERS) { double dp[3]; if (!solve3(o, o + 6, dp)) break; p[0]+=dp[0]; p[1]+=dp[1]; p[2]+=dp[2]; }
    }
    video.release();
    cudabot::avi_to_gif("tmp/gpu_direct_vo.avi", "gif/gpu_direct_vo.gif", 6, 760);
    std::printf("wrote gif/gpu_direct_vo.gif\n");

    CUDA_CHECK(cudaFree(d_I)); CUDA_CHECK(cudaFree(d_J));
    CUDA_CHECK(cudaFree(d_p)); CUDA_CHECK(cudaFree(d_out));
    return 0;
}

}  // namespace cudabot

int main() { return cudabot::main(); }
