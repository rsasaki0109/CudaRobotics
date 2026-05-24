// gpu_spectral_clustering.cu
//
// GPU spectral clustering demo.
//
// A synthetic 2D point cloud with two moons plus a compact island is
// clustered through a normalized RBF affinity graph. CUDA evaluates the
// dense graph degree and repeated normalized-affinity matvecs without
// materializing the NxN matrix; a tiny host QR step keeps the three-vector
// subspace orthonormal. The CPU path runs the same on-the-fly affinity
// iteration for a direct timing comparison.
//
// Output: gif/gpu_spectral_clustering.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int N_POINTS = 3072;
constexpr int K = 3;
constexpr int SPECTRAL_ITERS = 40;
constexpr int SNAP_STRIDE = 2;
constexpr int KMEANS_ITERS = 32;
constexpr int THREADS = 128;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int VIDEO_FPS = 10;
constexpr float PI_F = 3.14159265358979323846f;
constexpr float SIGMA = 0.22f;
constexpr float INV_TWO_SIGMA2 = 1.0f / (2.0f * SIGMA * SIGMA);
constexpr float X_MIN = -1.35f;
constexpr float X_MAX = 3.75f;
constexpr float Y_MIN = -1.70f;
constexpr float Y_MAX = 1.45f;

struct Point2 {
    float x;
    float y;
};

struct Dataset {
    std::vector<Point2> points;
    std::vector<int> labels;
};

struct ClusterResult {
    std::vector<int> raw_labels;
    std::vector<int> mapped_labels;
    std::array<int, K> pred_to_truth{};
    float accuracy = 0.0f;
    float objective = 0.0f;
};

struct Snapshot {
    int iter = 0;
    ClusterResult clusters;
};

struct BenchResult {
    double gpu_ms = 0.0;
    double cpu_ms = 0.0;
    double speedup = 0.0;
    float gpu_accuracy = 0.0f;
    float cpu_accuracy = 0.0f;
    float label_agreement = 0.0f;
};

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline float affinity(Point2 a, Point2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return expf(-(dx * dx + dy * dy) * INV_TWO_SIGMA2);
}

__global__ void degree_kernel(const Point2* __restrict__ points,
                              int n,
                              float* __restrict__ degree) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Point2 pi = points[i];
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        sum += affinity(pi, points[j]);
    }
    degree[i] = fmaxf(sum, 1.0e-8f);
}

__global__ void inv_sqrt_kernel(const float* __restrict__ degree,
                                int n,
                                float* __restrict__ inv_sqrt_degree) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    inv_sqrt_degree[i] = rsqrtf(fmaxf(degree[i], 1.0e-8f));
}

__global__ void normalized_affinity_matvec_kernel(
    const Point2* __restrict__ points,
    const float* __restrict__ inv_sqrt_degree,
    const float* __restrict__ v,
    int n,
    float* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Point2 pi = points[i];
    float inv_i = inv_sqrt_degree[i];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        float w = affinity(pi, points[j]) * inv_i * inv_sqrt_degree[j];
        acc0 += w * v[j * K + 0];
        acc1 += w * v[j * K + 1];
        acc2 += w * v[j * K + 2];
    }
    y[i * K + 0] = acc0;
    y[i * K + 1] = acc1;
    y[i * K + 2] = acc2;
}

static Dataset make_dataset() {
    Dataset data;
    data.points.reserve(N_POINTS);
    data.labels.reserve(N_POINTS);

    std::mt19937 rng(25052026);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.035f);

    int per_cluster = N_POINTS / K;
    for (int i = 0; i < per_cluster; i++) {
        float t = PI_F * (static_cast<float>(i) + 0.5f) / per_cluster;
        data.points.push_back({std::cos(t) + noise(rng),
                               std::sin(t) + noise(rng)});
        data.labels.push_back(0);
    }
    for (int i = 0; i < per_cluster; i++) {
        float t = PI_F * (static_cast<float>(i) + 0.5f) / per_cluster;
        data.points.push_back({1.0f - std::cos(t) + noise(rng),
                               -std::sin(t) - 0.55f + noise(rng)});
        data.labels.push_back(1);
    }
    for (int i = 0; i < per_cluster; i++) {
        float t = 2.0f * PI_F * (static_cast<float>(i) + uni(rng)) / per_cluster;
        float r = 0.43f + 0.065f * std::sin(3.0f * t) + noise(rng);
        data.points.push_back({3.00f + r * std::cos(t) + noise(rng),
                               0.55f + r * std::sin(t) + noise(rng)});
        data.labels.push_back(2);
    }

    std::vector<int> order(data.points.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    Dataset shuffled;
    shuffled.points.resize(data.points.size());
    shuffled.labels.resize(data.labels.size());
    for (int i = 0; i < static_cast<int>(order.size()); i++) {
        shuffled.points[i] = data.points[order[i]];
        shuffled.labels[i] = data.labels[order[i]];
    }
    return shuffled;
}

static void orthonormalize(std::vector<float>& basis) {
    int n = static_cast<int>(basis.size()) / K;
    for (int c = 0; c < K; c++) {
        for (int p = 0; p < c; p++) {
            double dot = 0.0;
            for (int i = 0; i < n; i++) {
                dot += static_cast<double>(basis[i * K + c]) * basis[i * K + p];
            }
            for (int i = 0; i < n; i++) {
                basis[i * K + c] -= static_cast<float>(dot) * basis[i * K + p];
            }
        }

        double norm2 = 0.0;
        for (int i = 0; i < n; i++) {
            float x = basis[i * K + c];
            norm2 += static_cast<double>(x) * x;
        }
        float inv_norm = 1.0f / std::sqrt(std::max(1.0e-20, norm2));
        for (int i = 0; i < n; i++) {
            basis[i * K + c] *= inv_norm;
        }

        int sign_idx = 0;
        while (sign_idx < n && std::fabs(basis[sign_idx * K + c]) < 1.0e-6f) sign_idx++;
        if (sign_idx < n && basis[sign_idx * K + c] < 0.0f) {
            for (int i = 0; i < n; i++) basis[i * K + c] = -basis[i * K + c];
        }
    }
}

static std::vector<float> make_initial_basis(int n) {
    std::mt19937 rng(12345);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> basis(n * K);
    for (float& x : basis) x = normal(rng);
    orthonormalize(basis);
    return basis;
}

static std::vector<float> row_normalized_embedding(const std::vector<float>& basis) {
    int n = static_cast<int>(basis.size()) / K;
    std::vector<float> z(basis.size());
    for (int i = 0; i < n; i++) {
        float norm2 = 0.0f;
        for (int k = 0; k < K; k++) {
            float x = basis[i * K + k];
            norm2 += x * x;
        }
        float inv_norm = 1.0f / std::sqrt(std::max(1.0e-12f, norm2));
        for (int k = 0; k < K; k++) z[i * K + k] = basis[i * K + k] * inv_norm;
    }
    return z;
}

static float dist2_row(const std::vector<float>& z, int a, const std::array<float, K>& c) {
    float d2 = 0.0f;
    for (int k = 0; k < K; k++) {
        float d = z[a * K + k] - c[k];
        d2 += d * d;
    }
    return d2;
}

static ClusterResult kmeans_embedding(const std::vector<float>& basis,
                                      const std::vector<int>& truth) {
    const int n = static_cast<int>(truth.size());
    std::vector<float> z = row_normalized_embedding(basis);
    std::array<std::array<float, K>, K> centroids{};
    std::array<int, K> seeds{};

    std::array<float, K> mean{};
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < K; k++) mean[k] += z[i * K + k];
    }
    for (int k = 0; k < K; k++) mean[k] /= n;

    float best = -1.0f;
    for (int i = 0; i < n; i++) {
        float d2 = 0.0f;
        for (int k = 0; k < K; k++) {
            float d = z[i * K + k] - mean[k];
            d2 += d * d;
        }
        if (d2 > best) {
            best = d2;
            seeds[0] = i;
        }
    }
    for (int k = 0; k < K; k++) centroids[0][k] = z[seeds[0] * K + k];

    for (int c = 1; c < K; c++) {
        best = -1.0f;
        for (int i = 0; i < n; i++) {
            float min_d2 = 1.0e30f;
            for (int p = 0; p < c; p++) {
                min_d2 = std::min(min_d2, dist2_row(z, i, centroids[p]));
            }
            if (min_d2 > best) {
                best = min_d2;
                seeds[c] = i;
            }
        }
        for (int k = 0; k < K; k++) centroids[c][k] = z[seeds[c] * K + k];
    }

    std::vector<int> pred(n, 0);
    for (int iter = 0; iter < KMEANS_ITERS; iter++) {
        for (int i = 0; i < n; i++) {
            int best_k = 0;
            float best_d2 = 1.0e30f;
            for (int c = 0; c < K; c++) {
                float d2 = dist2_row(z, i, centroids[c]);
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_k = c;
                }
            }
            pred[i] = best_k;
        }

        std::array<std::array<float, K>, K> sums{};
        std::array<int, K> counts{};
        for (int i = 0; i < n; i++) {
            counts[pred[i]]++;
            for (int k = 0; k < K; k++) sums[pred[i]][k] += z[i * K + k];
        }
        for (int c = 0; c < K; c++) {
            if (counts[c] == 0) continue;
            float inv_count = 1.0f / counts[c];
            for (int k = 0; k < K; k++) centroids[c][k] = sums[c][k] * inv_count;
        }
    }

    float objective = 0.0f;
    for (int i = 0; i < n; i++) objective += dist2_row(z, i, centroids[pred[i]]);

    int confusion[K][K] = {};
    for (int i = 0; i < n; i++) confusion[pred[i]][truth[i]]++;

    std::array<int, K> perm = {0, 1, 2};
    std::array<int, K> best_perm = perm;
    int best_hits = -1;
    do {
        int hits = 0;
        for (int c = 0; c < K; c++) hits += confusion[c][perm[c]];
        if (hits > best_hits) {
            best_hits = hits;
            best_perm = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    ClusterResult result;
    result.raw_labels = pred;
    result.mapped_labels.resize(n);
    result.pred_to_truth = best_perm;
    result.accuracy = static_cast<float>(best_hits) / n;
    result.objective = objective / n;
    for (int i = 0; i < n; i++) result.mapped_labels[i] = best_perm[pred[i]];
    return result;
}

static void compute_degree_cpu(const std::vector<Point2>& points,
                               std::vector<float>& degree) {
    int n = static_cast<int>(points.size());
    degree.assign(n, 0.0f);
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            sum += affinity(points[i], points[j]);
        }
        degree[i] = std::max(sum, 1.0e-8f);
    }
}

static void normalized_affinity_matvec_cpu(const std::vector<Point2>& points,
                                           const std::vector<float>& inv_sqrt_degree,
                                           const std::vector<float>& v,
                                           std::vector<float>& y) {
    int n = static_cast<int>(points.size());
    y.assign(n * K, 0.0f);
    for (int i = 0; i < n; i++) {
        float inv_i = inv_sqrt_degree[i];
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            float w = affinity(points[i], points[j]) * inv_i * inv_sqrt_degree[j];
            acc0 += w * v[j * K + 0];
            acc1 += w * v[j * K + 1];
            acc2 += w * v[j * K + 2];
        }
        y[i * K + 0] = acc0;
        y[i * K + 1] = acc1;
        y[i * K + 2] = acc2;
    }
}

static double run_cpu_spectral(const Dataset& data,
                               const std::vector<float>& init,
                               std::vector<float>& out_basis,
                               ClusterResult& out_clusters) {
    std::vector<float> degree;
    std::vector<float> inv_sqrt_degree(data.points.size());
    std::vector<float> v = init;
    std::vector<float> y(data.points.size() * K);

    auto t0 = std::chrono::high_resolution_clock::now();
    compute_degree_cpu(data.points, degree);
    for (int i = 0; i < static_cast<int>(degree.size()); i++) {
        inv_sqrt_degree[i] = 1.0f / std::sqrt(degree[i]);
    }
    for (int it = 0; it < SPECTRAL_ITERS; it++) {
        normalized_affinity_matvec_cpu(data.points, inv_sqrt_degree, v, y);
        orthonormalize(y);
        v.swap(y);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    out_basis = v;
    out_clusters = kmeans_embedding(out_basis, data.labels);
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double run_gpu_spectral(const Dataset& data,
                               const std::vector<float>& init,
                               std::vector<float>& out_basis,
                               std::vector<Snapshot>& snapshots) {
    const int n = static_cast<int>(data.points.size());
    Point2* d_points = nullptr;
    float* d_degree = nullptr;
    float* d_inv_sqrt_degree = nullptr;
    float* d_v = nullptr;
    float* d_y = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points, n * sizeof(Point2)));
    CUDA_CHECK(cudaMalloc(&d_degree, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv_sqrt_degree, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, n * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_points, data.points.data(), n * sizeof(Point2),
                          cudaMemcpyHostToDevice));

    std::vector<float> v = init;
    std::vector<float> y(n * K);
    std::vector<std::vector<float>> snapshot_basis;
    snapshot_basis.push_back(v);

    int blocks = (n + THREADS - 1) / THREADS;
    auto t0 = std::chrono::high_resolution_clock::now();
    degree_kernel<<<blocks, THREADS>>>(d_points, n, d_degree);
    inv_sqrt_kernel<<<blocks, THREADS>>>(d_degree, n, d_inv_sqrt_degree);
    CUDA_CHECK(cudaMemcpy(d_v, v.data(), n * K * sizeof(float), cudaMemcpyHostToDevice));
    for (int it = 0; it < SPECTRAL_ITERS; it++) {
        normalized_affinity_matvec_kernel<<<blocks, THREADS>>>(
            d_points, d_inv_sqrt_degree, d_v, n, d_y);
        CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * K * sizeof(float), cudaMemcpyDeviceToHost));
        orthonormalize(y);
        v.swap(y);
        CUDA_CHECK(cudaMemcpy(d_v, v.data(), n * K * sizeof(float), cudaMemcpyHostToDevice));
        if (((it + 1) % SNAP_STRIDE == 0) || it + 1 == SPECTRAL_ITERS) {
            snapshot_basis.push_back(v);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    out_basis = v;
    snapshots.clear();
    for (int s = 0; s < static_cast<int>(snapshot_basis.size()); s++) {
        Snapshot snap;
        snap.iter = std::min(SPECTRAL_ITERS, s * SNAP_STRIDE);
        if (s == static_cast<int>(snapshot_basis.size()) - 1) snap.iter = SPECTRAL_ITERS;
        snap.clusters = kmeans_embedding(snapshot_basis[s], data.labels);
        snapshots.push_back(std::move(snap));
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_degree));
    CUDA_CHECK(cudaFree(d_inv_sqrt_degree));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_y));
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static cv::Scalar color_for_label(int label) {
    static const cv::Scalar colors[K] = {
        cv::Scalar(80, 170, 255),
        cv::Scalar(90, 225, 135),
        cv::Scalar(220, 130, 245),
    };
    return colors[label % K];
}

static cv::Point to_px(Point2 p, const cv::Rect& r) {
    float x01 = clampf((p.x - X_MIN) / (X_MAX - X_MIN), 0.0f, 1.0f);
    float y01 = clampf((p.y - Y_MIN) / (Y_MAX - Y_MIN), 0.0f, 1.0f);
    return cv::Point(r.x + static_cast<int>(x01 * r.width),
                     r.y + r.height - static_cast<int>(y01 * r.height));
}

static void splat(cv::Mat& img, cv::Point p, cv::Scalar color) {
    for (int dy = 0; dy < 2; dy++) {
        int y = p.y + dy;
        if (y < 0 || y >= img.rows) continue;
        for (int dx = 0; dx < 2; dx++) {
            int x = p.x + dx;
            if (x < 0 || x >= img.cols) continue;
            cv::Vec3b& px = img.at<cv::Vec3b>(y, x);
            px[0] = static_cast<uchar>(color[0]);
            px[1] = static_cast<uchar>(color[1]);
            px[2] = static_cast<uchar>(color[2]);
        }
    }
}

static void draw_accuracy_plot(cv::Mat& img,
                               const std::vector<Snapshot>& snapshots,
                               const cv::Rect& r,
                               int upto_index) {
    cv::rectangle(img, r, cv::Scalar(29, 31, 36), -1);
    cv::rectangle(img, r, cv::Scalar(76, 80, 88), 1);
    cv::putText(img, "mapped cluster accuracy", cv::Point(r.x + 12, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(235, 235, 235), 1,
                cv::LINE_AA);
    for (int g = 0; g <= 4; g++) {
        int y = r.y + r.height - 22 - g * (r.height - 56) / 4;
        cv::line(img, cv::Point(r.x + 42, y), cv::Point(r.x + r.width - 14, y),
                 cv::Scalar(45, 48, 55), 1);
    }
    int last = std::min(upto_index, static_cast<int>(snapshots.size()) - 1);
    if (last >= 1) {
        std::vector<cv::Point> pts;
        for (int i = 0; i <= last; i++) {
            float x01 = static_cast<float>(snapshots[i].iter) / SPECTRAL_ITERS;
            float y01 = clampf(snapshots[i].clusters.accuracy, 0.0f, 1.0f);
            int x = r.x + 42 + static_cast<int>(x01 * (r.width - 58));
            int y = r.y + r.height - 22 - static_cast<int>(y01 * (r.height - 56));
            pts.emplace_back(x, y);
        }
        cv::polylines(img, pts, false, cv::Scalar(90, 225, 135), 2, cv::LINE_AA);
    }
    cv::putText(img, "100%", cv::Point(r.x + 7, r.y + 51),
                cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(165, 170, 180), 1,
                cv::LINE_AA);
    cv::putText(img, "0", cv::Point(r.x + 23, r.y + r.height - 17),
                cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(165, 170, 180), 1,
                cv::LINE_AA);
}

static float label_agreement(const std::vector<int>& a, const std::vector<int>& b) {
    int n = static_cast<int>(a.size());
    int same = 0;
    for (int i = 0; i < n; i++) same += (a[i] == b[i]);
    return static_cast<float>(same) / n;
}

static cv::Mat draw_frame(const Dataset& data,
                          const std::vector<Snapshot>& snapshots,
                          const BenchResult& bench,
                          int snap_index) {
    const Snapshot& snap = snapshots[snap_index];
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 19, 23));
    cv::putText(img,
                cv::format("GPU spectral clustering  iter %02d / %d",
                           snap.iter, SPECTRAL_ITERS),
                cv::Point(28, 36), cv::FONT_HERSHEY_SIMPLEX, 0.82,
                cv::Scalar(245, 245, 245), 2, cv::LINE_AA);
    cv::putText(img,
                "normalized RBF graph, subspace iteration, row k-means",
                cv::Point(31, 60), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(165, 170, 180), 1, cv::LINE_AA);

    cv::Rect scatter(34, 84, 560, 488);
    cv::rectangle(img, scatter, cv::Scalar(27, 29, 34), -1);
    cv::rectangle(img, scatter, cv::Scalar(78, 82, 90), 1);
    for (int i = 0; i < static_cast<int>(data.points.size()); i++) {
        int label = snap.clusters.mapped_labels[i];
        splat(img, to_px(data.points[i], scatter), color_for_label(label));
    }
    cv::putText(img, "spectral labels on non-convex data",
                cv::Point(scatter.x + 14, scatter.y + 24), cv::FONT_HERSHEY_SIMPLEX,
                0.45, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);

    for (int k = 0; k < K; k++) {
        int x = scatter.x + 16 + k * 132;
        int y = scatter.y + scatter.height - 18;
        cv::circle(img, cv::Point(x, y - 4), 5, color_for_label(k), -1, cv::LINE_AA);
        cv::putText(img, cv::format("cluster %d", k), cv::Point(x + 12, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(220, 220, 220), 1,
                    cv::LINE_AA);
    }

    cv::Rect plot(626, 84, 296, 178);
    draw_accuracy_plot(img, snapshots, plot, snap_index);

    cv::Rect info(626, 282, 296, 290);
    cv::rectangle(img, info, cv::Scalar(29, 31, 36), -1);
    cv::rectangle(img, info, cv::Scalar(76, 80, 88), 1);
    cv::putText(img, "benchmark", cv::Point(info.x + 14, info.y + 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(245, 245, 245), 1,
                cv::LINE_AA);
    cv::putText(img, cv::format("%d points, K=%d, sigma=%.2f", N_POINTS, K, SIGMA),
                cv::Point(info.x + 14, info.y + 60), cv::FONT_HERSHEY_SIMPLEX,
                0.40, cv::Scalar(205, 210, 218), 1, cv::LINE_AA);
    cv::putText(img, cv::format("GPU %.3f ms", bench.gpu_ms),
                cv::Point(info.x + 14, info.y + 98), cv::FONT_HERSHEY_SIMPLEX,
                0.55, cv::Scalar(90, 225, 135), 1, cv::LINE_AA);
    cv::putText(img, cv::format("CPU %.3f ms", bench.cpu_ms),
                cv::Point(info.x + 14, info.y + 130), cv::FONT_HERSHEY_SIMPLEX,
                0.55, cv::Scalar(165, 175, 190), 1, cv::LINE_AA);
    cv::putText(img, cv::format("speedup %.1fx", bench.speedup),
                cv::Point(info.x + 14, info.y + 162), cv::FONT_HERSHEY_SIMPLEX,
                0.55, cv::Scalar(250, 190, 70), 1, cv::LINE_AA);
    cv::putText(img, cv::format("GPU accuracy %.2f%%", 100.0f * bench.gpu_accuracy),
                cv::Point(info.x + 14, info.y + 206), cv::FONT_HERSHEY_SIMPLEX,
                0.46, cv::Scalar(220, 225, 232), 1, cv::LINE_AA);
    cv::putText(img, cv::format("CPU accuracy %.2f%%", 100.0f * bench.cpu_accuracy),
                cv::Point(info.x + 14, info.y + 232), cv::FONT_HERSHEY_SIMPLEX,
                0.46, cv::Scalar(220, 225, 232), 1, cv::LINE_AA);
    cv::putText(img, cv::format("GPU/CPU labels %.2f%%", 100.0f * bench.label_agreement),
                cv::Point(info.x + 14, info.y + 258), cv::FONT_HERSHEY_SIMPLEX,
                0.46, cv::Scalar(220, 225, 232), 1, cv::LINE_AA);

    return img;
}

static void write_video(const Dataset& data,
                        const std::vector<Snapshot>& snapshots,
                        const BenchResult& bench) {
    int mkdir_rc = std::system("mkdir -p gif");
    if (mkdir_rc != 0) {
        std::fprintf(stderr, "Failed to create gif directory\n");
        std::exit(1);
    }
    const std::string avi_path = "gif/gpu_spectral_clustering.avi";
    const std::string gif_path = "gif/gpu_spectral_clustering.gif";
    cv::VideoWriter writer(
        avi_path,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        VIDEO_FPS,
        cv::Size(PANEL_W, PANEL_H));
    if (!writer.isOpened()) {
        std::fprintf(stderr, "Failed to open %s\n", avi_path.c_str());
        std::exit(1);
    }
    for (int i = 0; i < static_cast<int>(snapshots.size()); i++) {
        writer.write(draw_frame(data, snapshots, bench, i));
    }
    for (int i = 0; i < 8; i++) {
        writer.write(draw_frame(data, snapshots, bench,
                                static_cast<int>(snapshots.size()) - 1));
    }
    writer.release();
    avi_to_gif(avi_path, gif_path, VIDEO_FPS, 720);
}

}  // namespace cudabot

int main() {
    using namespace cudabot;

    Dataset data = make_dataset();
    std::vector<float> init = make_initial_basis(static_cast<int>(data.points.size()));

    std::vector<float> gpu_basis;
    std::vector<float> cpu_basis;
    std::vector<Snapshot> snapshots;
    ClusterResult cpu_clusters;

    BenchResult bench;
    bench.gpu_ms = run_gpu_spectral(data, init, gpu_basis, snapshots);
    bench.cpu_ms = run_cpu_spectral(data, init, cpu_basis, cpu_clusters);
    ClusterResult gpu_clusters = kmeans_embedding(gpu_basis, data.labels);
    bench.gpu_accuracy = gpu_clusters.accuracy;
    bench.cpu_accuracy = cpu_clusters.accuracy;
    bench.label_agreement = label_agreement(gpu_clusters.mapped_labels,
                                            cpu_clusters.mapped_labels);
    bench.speedup = bench.cpu_ms / std::max(1.0e-9, bench.gpu_ms);

    if (!snapshots.empty()) {
        snapshots.back().clusters = gpu_clusters;
        snapshots.back().iter = SPECTRAL_ITERS;
    }

    write_video(data, snapshots, bench);

    std::printf("GPU spectral clustering demo\n");
    std::printf("points: %d, clusters: %d, iterations: %d, sigma: %.3f\n",
                N_POINTS, K, SPECTRAL_ITERS, SIGMA);
    std::printf("GPU spectral time: %.3f ms\n", bench.gpu_ms);
    std::printf("CPU spectral time: %.3f ms\n", bench.cpu_ms);
    std::printf("Speedup: %.1fx\n", bench.speedup);
    std::printf("GPU accuracy: %.4f\n", bench.gpu_accuracy);
    std::printf("CPU accuracy: %.4f\n", bench.cpu_accuracy);
    std::printf("GPU/CPU label agreement: %.4f\n", bench.label_agreement);
    std::printf("Wrote gif/gpu_spectral_clustering.gif\n");

    return 0;
}
