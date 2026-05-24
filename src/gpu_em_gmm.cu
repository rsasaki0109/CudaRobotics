// gpu_em_gmm.cu
//
// GPU Gaussian-mixture EM clustering demo.
//
// A synthetic 2D point cloud is generated from five full-covariance
// Gaussians. CUDA runs the E-step over all points, reduces sufficient
// statistics per component, and updates mixture weights, means, and
// covariances in the M-step. The CPU path uses the same EM equations for
// a direct timing comparison.
//
// Output: gif/gpu_em_gmm.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int N_POINTS = 262144;
constexpr int K = 5;
constexpr int EM_ITERS = 42;
constexpr int SNAP_STRIDE = 2;
constexpr int THREADS = 256;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int VIDEO_FPS = 10;
constexpr float PI_F = 3.14159265358979323846f;
constexpr float COV_FLOOR = 0.05f;
constexpr int STATS_PER_K = 6;  // count, sx, sy, sxx, sxy, syy
constexpr int STATS_N = K * STATS_PER_K + 1;

struct Point2 {
    float x;
    float y;
};

struct GmmParams {
    std::array<float, K> w;
    std::array<float, K> mx;
    std::array<float, K> my;
    std::array<float, K> cxx;
    std::array<float, K> cxy;
    std::array<float, K> cyy;
};

struct BenchResult {
    double gpu_ms = 0.0;
    double cpu_ms = 0.0;
    double speedup = 0.0;
    float gpu_nll = 0.0f;
    float cpu_nll = 0.0f;
    float mean_rmse = 0.0f;
};

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline int stat_idx(int k, int s) {
    return k * STATS_PER_K + s;
}

__host__ __device__ static inline float log_gauss2(float x,
                                                   float y,
                                                   float mx,
                                                   float my,
                                                   float cxx,
                                                   float cxy,
                                                   float cyy) {
    cxx = fmaxf(cxx, COV_FLOOR);
    cyy = fmaxf(cyy, COV_FLOOR);
    float det = fmaxf(cxx * cyy - cxy * cxy, COV_FLOOR * COV_FLOOR);
    float dx = x - mx;
    float dy = y - my;
    float q = (cyy * dx * dx - 2.0f * cxy * dx * dy + cxx * dy * dy) / det;
    return -0.5f * (q + logf(det) + 2.0f * logf(2.0f * PI_F));
}

__global__ void expectation_kernel(const Point2* __restrict__ points,
                                   int n,
                                   const float* __restrict__ w,
                                   const float* __restrict__ mx,
                                   const float* __restrict__ my,
                                   const float* __restrict__ cxx,
                                   const float* __restrict__ cxy,
                                   const float* __restrict__ cyy,
                                   float* __restrict__ stats) {
    extern __shared__ float sh[];
    for (int i = threadIdx.x; i < STATS_N; i += blockDim.x) sh[i] = 0.0f;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Point2 p = points[i];
        float lp[K];
        float best = -1.0e30f;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            lp[k] = logf(fmaxf(w[k], 1.0e-12f))
                  + log_gauss2(p.x, p.y, mx[k], my[k], cxx[k], cxy[k], cyy[k]);
            best = fmaxf(best, lp[k]);
        }
        float denom = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; k++) denom += expf(lp[k] - best);
        float nll = -(best + logf(fmaxf(denom, 1.0e-20f)));
        atomicAdd(&sh[K * STATS_PER_K], nll);
        float inv_denom = 1.0f / denom;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            float r = expf(lp[k] - best) * inv_denom;
            atomicAdd(&sh[stat_idx(k, 0)], r);
            atomicAdd(&sh[stat_idx(k, 1)], r * p.x);
            atomicAdd(&sh[stat_idx(k, 2)], r * p.y);
            atomicAdd(&sh[stat_idx(k, 3)], r * p.x * p.x);
            atomicAdd(&sh[stat_idx(k, 4)], r * p.x * p.y);
            atomicAdd(&sh[stat_idx(k, 5)], r * p.y * p.y);
        }
    }

    __syncthreads();
    for (int s = threadIdx.x; s < STATS_N; s += blockDim.x) {
        atomicAdd(&stats[s], sh[s]);
    }
}

__global__ void maximization_kernel(int n,
                                    const float* __restrict__ stats,
                                    float* __restrict__ w,
                                    float* __restrict__ mx,
                                    float* __restrict__ my,
                                    float* __restrict__ cxx,
                                    float* __restrict__ cxy,
                                    float* __restrict__ cyy) {
    int k = threadIdx.x;
    if (k >= K) return;
    float count = fmaxf(stats[stat_idx(k, 0)], 1.0e-6f);
    float inv_count = 1.0f / count;
    float x = stats[stat_idx(k, 1)] * inv_count;
    float y = stats[stat_idx(k, 2)] * inv_count;
    float xx = stats[stat_idx(k, 3)] * inv_count - x * x;
    float xy = stats[stat_idx(k, 4)] * inv_count - x * y;
    float yy = stats[stat_idx(k, 5)] * inv_count - y * y;
    xx = fmaxf(xx, COV_FLOOR);
    yy = fmaxf(yy, COV_FLOOR);
    float max_xy = 0.92f * sqrtf(xx * yy);
    xy = clampf(xy, -max_xy, max_xy);
    w[k] = count / n;
    mx[k] = x;
    my[k] = y;
    cxx[k] = xx;
    cxy[k] = xy;
    cyy[k] = yy;
}

static GmmParams make_truth() {
    GmmParams g;
    g.w = {0.22f, 0.18f, 0.24f, 0.17f, 0.19f};
    g.mx = {-4.0f, -1.7f, 1.6f, 3.55f, 0.05f};
    g.my = {-2.0f, 2.45f, -1.35f, 2.05f, 0.35f};
    g.cxx = {0.52f, 0.72f, 0.48f, 0.66f, 1.10f};
    g.cxy = {0.18f, -0.24f, 0.10f, 0.28f, -0.38f};
    g.cyy = {0.36f, 0.50f, 0.80f, 0.44f, 0.82f};
    return g;
}

static GmmParams make_initial() {
    GmmParams g;
    g.w = {0.20f, 0.20f, 0.20f, 0.20f, 0.20f};
    g.mx = {-4.85f, -2.30f, 0.80f, 4.15f, 0.95f};
    g.my = {-1.20f, 1.70f, -2.25f, 1.25f, 1.25f};
    g.cxx = {2.40f, 2.20f, 2.50f, 2.20f, 2.80f};
    g.cxy = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    g.cyy = {2.00f, 2.30f, 2.20f, 2.10f, 2.70f};
    return g;
}

static std::vector<Point2> make_points(const GmmParams& truth) {
    std::vector<Point2> points(N_POINTS);
    std::mt19937 rng(25052026);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    std::array<float, K> cdf{};
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += truth.w[k];
        cdf[k] = acc;
    }
    cdf[K - 1] = 1.0f;

    for (int i = 0; i < N_POINTS; i++) {
        float u = uni(rng);
        int k = 0;
        while (k + 1 < K && u > cdf[k]) k++;
        float a = truth.cxx[k];
        float b = truth.cxy[k];
        float c = truth.cyy[k];
        float l00 = std::sqrt(std::max(COV_FLOOR, a));
        float l10 = b / l00;
        float l11 = std::sqrt(std::max(COV_FLOOR, c - l10 * l10));
        float z0 = normal(rng);
        float z1 = normal(rng);
        points[i] = {truth.mx[k] + l00 * z0,
                     truth.my[k] + l10 * z0 + l11 * z1};
    }
    return points;
}

static float nll_cpu(const GmmParams& g, const std::vector<Point2>& points) {
    double total = 0.0;
    for (const auto& p : points) {
        float lp[K];
        float best = -1.0e30f;
        for (int k = 0; k < K; k++) {
            lp[k] = std::log(std::max(1.0e-12f, g.w[k]))
                  + log_gauss2(p.x, p.y, g.mx[k], g.my[k], g.cxx[k], g.cxy[k], g.cyy[k]);
            best = std::max(best, lp[k]);
        }
        float denom = 0.0f;
        for (int k = 0; k < K; k++) denom += std::exp(lp[k] - best);
        total += -(best + std::log(std::max(1.0e-20f, denom)));
    }
    return static_cast<float>(total / points.size());
}

static void m_step_cpu(const std::array<double, STATS_N>& stats, int n, GmmParams& g) {
    for (int k = 0; k < K; k++) {
        double count = std::max(1.0e-9, stats[stat_idx(k, 0)]);
        double x = stats[stat_idx(k, 1)] / count;
        double y = stats[stat_idx(k, 2)] / count;
        double xx = stats[stat_idx(k, 3)] / count - x * x;
        double xy = stats[stat_idx(k, 4)] / count - x * y;
        double yy = stats[stat_idx(k, 5)] / count - y * y;
        xx = std::max<double>(COV_FLOOR, xx);
        yy = std::max<double>(COV_FLOOR, yy);
        double max_xy = 0.92 * std::sqrt(xx * yy);
        xy = std::min(max_xy, std::max(-max_xy, xy));
        g.w[k] = static_cast<float>(count / n);
        g.mx[k] = static_cast<float>(x);
        g.my[k] = static_cast<float>(y);
        g.cxx[k] = static_cast<float>(xx);
        g.cxy[k] = static_cast<float>(xy);
        g.cyy[k] = static_cast<float>(yy);
    }
}

static BenchResult run_cpu_em(const std::vector<Point2>& points,
                              GmmParams init,
                              GmmParams truth,
                              GmmParams& out,
                              std::vector<float>& history) {
    history.clear();
    auto t0 = std::chrono::high_resolution_clock::now();
    GmmParams g = init;
    for (int it = 0; it < EM_ITERS; it++) {
        std::array<double, STATS_N> stats{};
        for (const auto& p : points) {
            float lp[K];
            float best = -1.0e30f;
            for (int k = 0; k < K; k++) {
                lp[k] = std::log(std::max(1.0e-12f, g.w[k]))
                      + log_gauss2(p.x, p.y, g.mx[k], g.my[k], g.cxx[k], g.cxy[k], g.cyy[k]);
                best = std::max(best, lp[k]);
            }
            float denom = 0.0f;
            for (int k = 0; k < K; k++) denom += std::exp(lp[k] - best);
            stats[K * STATS_PER_K] += -(best + std::log(std::max(1.0e-20f, denom)));
            float inv_denom = 1.0f / denom;
            for (int k = 0; k < K; k++) {
                double r = std::exp(lp[k] - best) * inv_denom;
                stats[stat_idx(k, 0)] += r;
                stats[stat_idx(k, 1)] += r * p.x;
                stats[stat_idx(k, 2)] += r * p.y;
                stats[stat_idx(k, 3)] += r * p.x * p.x;
                stats[stat_idx(k, 4)] += r * p.x * p.y;
                stats[stat_idx(k, 5)] += r * p.y * p.y;
            }
        }
        history.push_back(static_cast<float>(stats[K * STATS_PER_K] / points.size()));
        m_step_cpu(stats, static_cast<int>(points.size()), g);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    out = g;
    history.push_back(nll_cpu(g, points));

    double err = 0.0;
    for (int k = 0; k < K; k++) {
        err += (g.mx[k] - truth.mx[k]) * (g.mx[k] - truth.mx[k])
             + (g.my[k] - truth.my[k]) * (g.my[k] - truth.my[k]);
    }
    BenchResult br;
    br.cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    br.cpu_nll = history.back();
    br.mean_rmse = static_cast<float>(std::sqrt(err / K));
    return br;
}

static void upload_params(const GmmParams& g,
                          float* d_w,
                          float* d_mx,
                          float* d_my,
                          float* d_cxx,
                          float* d_cxy,
                          float* d_cyy) {
    CUDA_CHECK(cudaMemcpy(d_w, g.w.data(), K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx, g.mx.data(), K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my, g.my.data(), K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cxx, g.cxx.data(), K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cxy, g.cxy.data(), K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cyy, g.cyy.data(), K * sizeof(float), cudaMemcpyHostToDevice));
}

static void download_params(GmmParams& g,
                            float* d_w,
                            float* d_mx,
                            float* d_my,
                            float* d_cxx,
                            float* d_cxy,
                            float* d_cyy) {
    CUDA_CHECK(cudaMemcpy(g.w.data(), d_w, K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g.mx.data(), d_mx, K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g.my.data(), d_my, K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g.cxx.data(), d_cxx, K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g.cxy.data(), d_cxy, K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g.cyy.data(), d_cyy, K * sizeof(float), cudaMemcpyDeviceToHost));
}

static BenchResult run_gpu_em(const std::vector<Point2>& points,
                              GmmParams init,
                              GmmParams truth,
                              GmmParams& out,
                              std::vector<float>& history,
                              std::vector<GmmParams>& snapshots) {
    Point2* d_points = nullptr;
    float *d_w = nullptr, *d_mx = nullptr, *d_my = nullptr;
    float *d_cxx = nullptr, *d_cxy = nullptr, *d_cyy = nullptr;
    float* d_stats = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, points.size() * sizeof(Point2)));
    CUDA_CHECK(cudaMalloc(&d_w, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mx, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_my, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cxx, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cxy, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cyy, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stats, STATS_N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), points.size() * sizeof(Point2),
                          cudaMemcpyHostToDevice));
    upload_params(init, d_w, d_mx, d_my, d_cxx, d_cxy, d_cyy);

    int blocks = (static_cast<int>(points.size()) + THREADS - 1) / THREADS;
    size_t shmem = STATS_N * sizeof(float);
    history.clear();
    snapshots.clear();
    snapshots.push_back(init);

    auto t0 = std::chrono::high_resolution_clock::now();
    GmmParams snap = init;
    float nll_sum = 0.0f;
    for (int it = 0; it < EM_ITERS; it++) {
        CUDA_CHECK(cudaMemset(d_stats, 0, STATS_N * sizeof(float)));
        expectation_kernel<<<blocks, THREADS, shmem>>>(d_points, static_cast<int>(points.size()),
                                                       d_w, d_mx, d_my,
                                                       d_cxx, d_cxy, d_cyy,
                                                       d_stats);
        CUDA_CHECK(cudaMemcpy(&nll_sum, d_stats + K * STATS_PER_K, sizeof(float),
                              cudaMemcpyDeviceToHost));
        history.push_back(nll_sum / points.size());
        maximization_kernel<<<1, K>>>(static_cast<int>(points.size()), d_stats,
                                      d_w, d_mx, d_my, d_cxx, d_cxy, d_cyy);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (((it + 1) % SNAP_STRIDE == 0) || it + 1 == EM_ITERS) {
            download_params(snap, d_w, d_mx, d_my, d_cxx, d_cxy, d_cyy);
            snapshots.push_back(snap);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    download_params(snap, d_w, d_mx, d_my, d_cxx, d_cxy, d_cyy);
    out = snap;
    history.push_back(nll_cpu(out, points));

    double err = 0.0;
    for (int k = 0; k < K; k++) {
        err += (out.mx[k] - truth.mx[k]) * (out.mx[k] - truth.mx[k])
             + (out.my[k] - truth.my[k]) * (out.my[k] - truth.my[k]);
    }

    BenchResult br;
    br.gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    br.gpu_nll = history.back();
    br.mean_rmse = static_cast<float>(std::sqrt(err / K));

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_mx));
    CUDA_CHECK(cudaFree(d_my));
    CUDA_CHECK(cudaFree(d_cxx));
    CUDA_CHECK(cudaFree(d_cxy));
    CUDA_CHECK(cudaFree(d_cyy));
    CUDA_CHECK(cudaFree(d_stats));
    return br;
}

static int assign_component(const GmmParams& g, Point2 p) {
    int best_k = 0;
    float best = -1.0e30f;
    for (int k = 0; k < K; k++) {
        float lp = std::log(std::max(1.0e-12f, g.w[k]))
                 + log_gauss2(p.x, p.y, g.mx[k], g.my[k], g.cxx[k], g.cxy[k], g.cyy[k]);
        if (lp > best) {
            best = lp;
            best_k = k;
        }
    }
    return best_k;
}

static cv::Scalar color_for_k(int k) {
    static const cv::Scalar c[K] = {
        cv::Scalar(80, 170, 255), cv::Scalar(90, 225, 135),
        cv::Scalar(250, 190, 70), cv::Scalar(210, 120, 245),
        cv::Scalar(80, 220, 225)
    };
    return c[k % K];
}

static cv::Point to_px(float x, float y, const cv::Rect& r) {
    float x01 = clampf((x + 6.2f) / 12.4f, 0.0f, 1.0f);
    float y01 = clampf((y + 4.8f) / 9.6f, 0.0f, 1.0f);
    return cv::Point(r.x + static_cast<int>(x01 * r.width),
                     r.y + r.height - static_cast<int>(y01 * r.height));
}

static void draw_ellipse(cv::Mat& img,
                         const GmmParams& g,
                         int k,
                         const cv::Rect& r,
                         cv::Scalar color,
                         int thickness) {
    float a = g.cxx[k];
    float b = g.cxy[k];
    float c = g.cyy[k];
    float tr = a + c;
    float disc = std::sqrt(std::max(0.0f, (a - c) * (a - c) + 4.0f * b * b));
    float l1 = std::max(COV_FLOOR, 0.5f * (tr + disc));
    float l2 = std::max(COV_FLOOR, 0.5f * (tr - disc));
    float angle = 0.5f * std::atan2(2.0f * b, a - c) * 180.0f / PI_F;
    float sx = r.width / 12.4f;
    float sy = r.height / 9.6f;
    float scale = 2.45f;
    cv::Point center = to_px(g.mx[k], g.my[k], r);
    cv::Size axes(static_cast<int>(scale * std::sqrt(l1) * sx),
                  static_cast<int>(scale * std::sqrt(l2) * sy));
    axes.width = std::max(4, axes.width);
    axes.height = std::max(4, axes.height);
    cv::ellipse(img, center, axes, -angle, 0.0, 360.0, color, thickness, cv::LINE_AA);
    cv::circle(img, center, 4, color, -1, cv::LINE_AA);
}

static void draw_history(cv::Mat& img,
                         const std::vector<float>& gpu_hist,
                         const std::vector<float>& cpu_hist,
                         const cv::Rect& r,
                         int upto_iter) {
    cv::rectangle(img, r, cv::Scalar(29, 31, 36), -1);
    cv::rectangle(img, r, cv::Scalar(76, 80, 88), 1);
    cv::putText(img, "negative log-likelihood / point", cv::Point(r.x + 12, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(235, 235, 235), 1,
                cv::LINE_AA);
    float hi = std::max(gpu_hist.front(), cpu_hist.front());
    float lo = std::min(gpu_hist.back(), cpu_hist.back());
    float span = std::max(1.0e-4f, hi - lo);
    for (int g = 0; g <= 4; g++) {
        int y = r.y + r.height - 22 - g * (r.height - 56) / 4;
        cv::line(img, cv::Point(r.x + 42, y), cv::Point(r.x + r.width - 14, y),
                 cv::Scalar(45, 48, 55), 1);
    }
    auto draw_one = [&](const std::vector<float>& h, cv::Scalar color, int limit) {
        int last = std::min<int>(limit, static_cast<int>(h.size()) - 1);
        if (last < 1) return;
        std::vector<cv::Point> pts;
        for (int i = 0; i <= last; i++) {
            float x01 = static_cast<float>(i) / EM_ITERS;
            float y01 = clampf((h[i] - lo) / span, 0.0f, 1.0f);
            int x = r.x + 42 + static_cast<int>(x01 * (r.width - 58));
            int y = r.y + r.height - 22 - static_cast<int>(y01 * (r.height - 56));
            pts.emplace_back(x, y);
        }
        cv::polylines(img, pts, false, color, 2, cv::LINE_AA);
    };
    draw_one(cpu_hist, cv::Scalar(160, 170, 185), static_cast<int>(cpu_hist.size()) - 1);
    draw_one(gpu_hist, cv::Scalar(90, 225, 135), upto_iter);
    cv::putText(img, "GPU", cv::Point(r.x + 252, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(90, 225, 135), 1);
    cv::putText(img, "CPU", cv::Point(r.x + 304, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(170, 180, 195), 1);
}

static cv::Mat draw_frame(const std::vector<Point2>& points,
                          const GmmParams& truth,
                          const GmmParams& fit,
                          const std::vector<float>& gpu_hist,
                          const std::vector<float>& cpu_hist,
                          const BenchResult& bench,
                          int iter) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 19, 23));
    cv::putText(img, cv::format("GPU EM GMM clustering  iter %02d / %d", iter, EM_ITERS),
                cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.70,
                cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img,
                cv::format("%d points x %d full-cov Gaussians   GPU %.2f ms   CPU %.2f ms   %.1fx",
                           N_POINTS, K, bench.gpu_ms, bench.cpu_ms, bench.speedup),
                cv::Point(20, 56), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(210, 214, 220), 1, cv::LINE_AA);

    cv::Rect scatter(28, 86, 592, 500);
    cv::rectangle(img, scatter, cv::Scalar(25, 27, 31), -1);
    cv::rectangle(img, scatter, cv::Scalar(80, 84, 92), 1);
    for (int gx = -6; gx <= 6; gx += 2) {
        cv::line(img, to_px(static_cast<float>(gx), -4.8f, scatter),
                 to_px(static_cast<float>(gx), 4.8f, scatter),
                 cv::Scalar(42, 45, 50), 1);
    }
    for (int gy = -4; gy <= 4; gy += 2) {
        cv::line(img, to_px(-6.2f, static_cast<float>(gy), scatter),
                 to_px(6.2f, static_cast<float>(gy), scatter),
                 cv::Scalar(42, 45, 50), 1);
    }
    for (int i = 0; i < N_POINTS; i += 26) {
        int k = assign_component(fit, points[i]);
        cv::circle(img, to_px(points[i].x, points[i].y, scatter), 1, color_for_k(k), -1);
    }
    for (int k = 0; k < K; k++) draw_ellipse(img, truth, k, scatter, cv::Scalar(135, 138, 145), 1);
    for (int k = 0; k < K; k++) draw_ellipse(img, fit, k, scatter, color_for_k(k), 2);
    cv::putText(img, "points colored by max responsibility; gray = GT, color = fitted",
                cv::Point(scatter.x + 12, scatter.y + 24), cv::FONT_HERSHEY_SIMPLEX,
                0.42, cv::Scalar(220, 224, 230), 1, cv::LINE_AA);

    cv::Rect stats(650, 92, 282, 168);
    cv::rectangle(img, stats, cv::Scalar(29, 31, 36), -1);
    cv::rectangle(img, stats, cv::Scalar(76, 80, 88), 1);
    cv::putText(img, cv::format("GPU NLL %.4f", gpu_hist[std::min(iter, (int)gpu_hist.size() - 1)]),
                cv::Point(stats.x + 14, stats.y + 34), cv::FONT_HERSHEY_SIMPLEX,
                0.50, cv::Scalar(90, 225, 135), 1, cv::LINE_AA);
    cv::putText(img, cv::format("CPU NLL %.4f", bench.cpu_nll),
                cv::Point(stats.x + 14, stats.y + 66), cv::FONT_HERSHEY_SIMPLEX,
                0.50, cv::Scalar(170, 180, 195), 1, cv::LINE_AA);
    cv::putText(img, cv::format("mean RMSE %.4f", bench.mean_rmse),
                cv::Point(stats.x + 14, stats.y + 98), cv::FONT_HERSHEY_SIMPLEX,
                0.50, cv::Scalar(90, 170, 255), 1, cv::LINE_AA);
    cv::putText(img, cv::format("CPU/GPU %.1fx", bench.speedup),
                cv::Point(stats.x + 14, stats.y + 130), cv::FONT_HERSHEY_SIMPLEX,
                0.50, cv::Scalar(220, 224, 230), 1, cv::LINE_AA);

    draw_history(img, gpu_hist, cpu_hist, cv::Rect(650, 306, 282, 208), iter);
    cv::putText(img, "E-step: responsibilities + sufficient stats, M-step: full covariance update",
                cv::Point(650, 558), cv::FONT_HERSHEY_SIMPLEX, 0.38,
                cv::Scalar(185, 190, 198), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    GmmParams truth = make_truth();
    GmmParams init = make_initial();
    std::vector<Point2> points = make_points(truth);

    GmmParams cpu_fit;
    std::vector<float> cpu_hist;
    BenchResult cpu = run_cpu_em(points, init, truth, cpu_fit, cpu_hist);

    GmmParams gpu_fit;
    std::vector<float> gpu_hist;
    std::vector<GmmParams> snapshots;
    BenchResult gpu = run_gpu_em(points, init, truth, gpu_fit, gpu_hist, snapshots);
    gpu.cpu_ms = cpu.cpu_ms;
    gpu.cpu_nll = cpu.cpu_nll;
    gpu.speedup = gpu.cpu_ms / std::max(1.0e-9, gpu.gpu_ms);

    std::printf("GPU EM GMM: %d points, %d components, %d iterations\n", N_POINTS, K, EM_ITERS);
    std::printf("GPU %.3f ms, CPU %.3f ms, speedup %.1fx\n",
                gpu.gpu_ms, gpu.cpu_ms, gpu.speedup);
    std::printf("GPU NLL %.6f, CPU NLL %.6f, mean RMSE %.6f\n",
                gpu.gpu_nll, gpu.cpu_nll, gpu.mean_rmse);

    int mkdir_ret = std::system("mkdir -p gif");
    (void)mkdir_ret;
    cv::VideoWriter video("gif/gpu_em_gmm.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_em_gmm.avi\n");
        return 1;
    }

    for (int hold = 0; hold < 8; hold++) {
        video.write(draw_frame(points, truth, snapshots.front(), gpu_hist, cpu_hist, gpu, 0));
    }
    for (size_t i = 0; i < snapshots.size(); i++) {
        int iter = std::min(EM_ITERS, static_cast<int>(i) * SNAP_STRIDE);
        cv::Mat frame = draw_frame(points, truth, snapshots[i], gpu_hist, cpu_hist, gpu, iter);
        video.write(frame);
        if (i + 1 == snapshots.size()) {
            for (int hold = 0; hold < 12; hold++) video.write(frame);
        }
    }
    video.release();
    cudabot::avi_to_gif("gif/gpu_em_gmm.avi", "gif/gpu_em_gmm.gif", 10, 720);
    std::printf("GIF saved to gif/gpu_em_gmm.gif\n");
    return 0;
}
