// gpu_label_propagation.cu
//
// GPU semi-supervised label propagation demo.
//
// A synthetic 2D point cloud with two interlocking moons plus a compact
// island carries only a handful of labeled seed nodes per class; every
// other node is unlabeled. Clamped label propagation (Zhu & Ghahramani,
// "Learning from Labeled and Unlabeled Data with Label Propagation",
// CMU-CALD-02-107, 2002) floods the seed labels across a normalized RBF
// affinity graph: each iteration replaces every unlabeled node's class
// distribution with the degree-normalized affinity-weighted average of its
// neighbours, while seed rows stay clamped to their one-hot label. The
// harmonic fixed point follows the data manifolds, so labels wrap around
// the non-convex moons where a nearest-seed Euclidean rule would fail.
//
// CUDA evaluates the dense graph propagation matvec (one thread per node,
// no materialized NxN matrix); the CPU path runs the identical on-the-fly
// iteration for a direct timing comparison.
//
// Output: gif/gpu_label_propagation.gif

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

constexpr int N_NODES = 3072;
constexpr int K = 3;
constexpr int SEEDS_PER_CLASS = 4;
constexpr int LP_ITERS = 50;
constexpr int SNAP_STRIDE = 2;
constexpr int THREADS = 128;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int VIDEO_FPS = 10;
constexpr float PI_F = 3.14159265358979323846f;
constexpr float SIGMA = 0.20f;
constexpr float INV_TWO_SIGMA2 = 1.0f / (2.0f * SIGMA * SIGMA);
constexpr float INV_K = 1.0f / static_cast<float>(K);
constexpr float X_MIN = -1.35f;
constexpr float X_MAX = 3.75f;
constexpr float Y_MIN = -1.70f;
constexpr float Y_MAX = 1.45f;

struct Point2 {
    float x;
    float y;
};

struct Dataset {
    std::vector<Point2> points;     // node coordinates
    std::vector<int> labels;        // ground-truth class per node
    std::vector<int> seed_label;    // class if seed, else -1
    int seed_count = 0;
};

// One captured propagation state: argmax label and confidence per node plus
// the accuracy measured over the unlabeled nodes only.
struct Snapshot {
    int iter = 0;
    std::vector<int> pred;
    std::vector<float> conf;
    float accuracy = 0.0f;
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
                              float* __restrict__ inv_degree) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Point2 pi = points[i];
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        sum += affinity(pi, points[j]);
    }
    inv_degree[i] = 1.0f / fmaxf(sum, 1.0e-8f);
}

// One clamped label-propagation step. Seed rows are forced back to their
// one-hot label; every other row becomes the degree-normalized affinity
// average of all neighbour distributions.
__global__ void propagate_kernel(const Point2* __restrict__ points,
                                 const float* __restrict__ inv_degree,
                                 const int* __restrict__ seed_label,
                                 const float* __restrict__ f_in,
                                 int n,
                                 float* __restrict__ f_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int sl = seed_label[i];
    if (sl >= 0) {
        for (int k = 0; k < K; k++) f_out[i * K + k] = (k == sl) ? 1.0f : 0.0f;
        return;
    }

    Point2 pi = points[i];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        float w = affinity(pi, points[j]);
        acc0 += w * f_in[j * K + 0];
        acc1 += w * f_in[j * K + 1];
        acc2 += w * f_in[j * K + 2];
    }
    float invd = inv_degree[i];
    f_out[i * K + 0] = acc0 * invd;
    f_out[i * K + 1] = acc1 * invd;
    f_out[i * K + 2] = acc2 * invd;
}

static Dataset make_dataset() {
    Dataset data;
    data.points.reserve(N_NODES);
    data.labels.reserve(N_NODES);

    std::mt19937 rng(25052026);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.035f);

    int per_cluster = N_NODES / K;
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

    // Pick a few labeled seeds per class. Spreading them within each class
    // keeps the demo honest: propagation must still travel along the
    // manifold to reach the far ends.
    shuffled.seed_label.assign(shuffled.points.size(), -1);
    std::array<std::vector<int>, K> by_class;
    for (int i = 0; i < static_cast<int>(shuffled.labels.size()); i++) {
        by_class[shuffled.labels[i]].push_back(i);
    }
    for (int c = 0; c < K; c++) {
        std::shuffle(by_class[c].begin(), by_class[c].end(), rng);
        int take = std::min(SEEDS_PER_CLASS, static_cast<int>(by_class[c].size()));
        for (int s = 0; s < take; s++) {
            shuffled.seed_label[by_class[c][s]] = c;
            shuffled.seed_count++;
        }
    }
    return shuffled;
}

static std::vector<float> make_initial_distribution(const Dataset& data) {
    int n = static_cast<int>(data.points.size());
    std::vector<float> f(n * K, INV_K);
    for (int i = 0; i < n; i++) {
        int sl = data.seed_label[i];
        if (sl < 0) continue;
        for (int k = 0; k < K; k++) f[i * K + k] = (k == sl) ? 1.0f : 0.0f;
    }
    return f;
}

// Turn a class-distribution matrix into per-node argmax + confidence and
// score accuracy over the unlabeled nodes only (seeds are given, so they do
// not count toward the propagation quality).
static Snapshot evaluate(const std::vector<float>& f, const Dataset& data, int iter) {
    int n = static_cast<int>(data.points.size());
    Snapshot snap;
    snap.iter = iter;
    snap.pred.resize(n);
    snap.conf.resize(n);
    int correct = 0;
    int total = 0;
    for (int i = 0; i < n; i++) {
        int best_k = 0;
        float best_p = f[i * K + 0];
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float p = f[i * K + k];
            sum += p;
            if (p > best_p) {
                best_p = p;
                best_k = k;
            }
        }
        float norm = (best_p) / std::max(1.0e-12f, sum);
        snap.pred[i] = best_k;
        snap.conf[i] = clampf((norm - INV_K) / (1.0f - INV_K), 0.0f, 1.0f);
        if (data.seed_label[i] < 0) {
            total++;
            correct += (best_k == data.labels[i]);
        }
    }
    snap.accuracy = total > 0 ? static_cast<float>(correct) / total : 0.0f;
    return snap;
}

static void degree_cpu(const std::vector<Point2>& points, std::vector<float>& inv_degree) {
    int n = static_cast<int>(points.size());
    inv_degree.assign(n, 0.0f);
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            sum += affinity(points[i], points[j]);
        }
        inv_degree[i] = 1.0f / std::max(sum, 1.0e-8f);
    }
}

static void propagate_cpu(const Dataset& data,
                          const std::vector<float>& inv_degree,
                          const std::vector<float>& f_in,
                          std::vector<float>& f_out) {
    int n = static_cast<int>(data.points.size());
    f_out.assign(n * K, 0.0f);
    for (int i = 0; i < n; i++) {
        int sl = data.seed_label[i];
        if (sl >= 0) {
            for (int k = 0; k < K; k++) f_out[i * K + k] = (k == sl) ? 1.0f : 0.0f;
            continue;
        }
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            float w = affinity(data.points[i], data.points[j]);
            acc0 += w * f_in[j * K + 0];
            acc1 += w * f_in[j * K + 1];
            acc2 += w * f_in[j * K + 2];
        }
        float invd = inv_degree[i];
        f_out[i * K + 0] = acc0 * invd;
        f_out[i * K + 1] = acc1 * invd;
        f_out[i * K + 2] = acc2 * invd;
    }
}

static double run_cpu_propagation(const Dataset& data,
                                  const std::vector<float>& init,
                                  std::vector<float>& out_f) {
    std::vector<float> inv_degree;
    std::vector<float> f = init;
    std::vector<float> f_next(f.size());

    auto t0 = std::chrono::high_resolution_clock::now();
    degree_cpu(data.points, inv_degree);
    for (int it = 0; it < LP_ITERS; it++) {
        propagate_cpu(data, inv_degree, f, f_next);
        f.swap(f_next);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    out_f = f;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double run_gpu_propagation(const Dataset& data,
                                  const std::vector<float>& init,
                                  std::vector<float>& out_f,
                                  std::vector<Snapshot>& snapshots) {
    const int n = static_cast<int>(data.points.size());
    Point2* d_points = nullptr;
    float* d_inv_degree = nullptr;
    int* d_seed_label = nullptr;
    float* d_f = nullptr;
    float* d_f_next = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points, n * sizeof(Point2)));
    CUDA_CHECK(cudaMalloc(&d_inv_degree, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_seed_label, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_f, n * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_f_next, n * K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_points, data.points.data(), n * sizeof(Point2),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seed_label, data.seed_label.data(), n * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f, init.data(), n * K * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::vector<float> host_f = init;
    snapshots.clear();
    snapshots.push_back(evaluate(host_f, data, 0));

    int blocks = (n + THREADS - 1) / THREADS;
    auto t0 = std::chrono::high_resolution_clock::now();
    degree_kernel<<<blocks, THREADS>>>(d_points, n, d_inv_degree);
    for (int it = 0; it < LP_ITERS; it++) {
        propagate_kernel<<<blocks, THREADS>>>(d_points, d_inv_degree, d_seed_label,
                                              d_f, n, d_f_next);
        std::swap(d_f, d_f_next);
        if (((it + 1) % SNAP_STRIDE == 0) || it + 1 == LP_ITERS) {
            CUDA_CHECK(cudaMemcpy(host_f.data(), d_f, n * K * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            snapshots.push_back(evaluate(host_f, data, it + 1));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(host_f.data(), d_f, n * K * sizeof(float),
                          cudaMemcpyDeviceToHost));
    out_f = host_f;

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_inv_degree));
    CUDA_CHECK(cudaFree(d_seed_label));
    CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_f_next));
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
    cv::putText(img, "unlabeled-node accuracy", cv::Point(r.x + 12, r.y + 24),
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
            float x01 = static_cast<float>(snapshots[i].iter) / LP_ITERS;
            float y01 = clampf(snapshots[i].accuracy, 0.0f, 1.0f);
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
                cv::format("GPU label propagation  iter %02d / %d",
                           snap.iter, LP_ITERS),
                cv::Point(28, 36), cv::FONT_HERSHEY_SIMPLEX, 0.82,
                cv::Scalar(245, 245, 245), 2, cv::LINE_AA);
    cv::putText(img,
                "clamped seeds flood class labels along an RBF graph",
                cv::Point(31, 60), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(165, 170, 180), 1, cv::LINE_AA);

    cv::Rect scatter(34, 84, 560, 488);
    cv::rectangle(img, scatter, cv::Scalar(27, 29, 34), -1);
    cv::rectangle(img, scatter, cv::Scalar(78, 82, 90), 1);

    const cv::Scalar gray(70, 72, 78);
    for (int i = 0; i < static_cast<int>(data.points.size()); i++) {
        cv::Scalar c = color_for_label(snap.pred[i]);
        float a = snap.conf[i];
        cv::Scalar blended(gray[0] * (1.0f - a) + c[0] * a,
                           gray[1] * (1.0f - a) + c[1] * a,
                           gray[2] * (1.0f - a) + c[2] * a);
        splat(img, to_px(data.points[i], scatter), blended);
    }
    // Seeds drawn last with a white ring so the few labeled anchors stand out.
    for (int i = 0; i < static_cast<int>(data.points.size()); i++) {
        if (data.seed_label[i] < 0) continue;
        cv::Point px = to_px(data.points[i], scatter);
        cv::circle(img, px, 5, color_for_label(data.seed_label[i]), -1, cv::LINE_AA);
        cv::circle(img, px, 5, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    }
    cv::putText(img, "propagated labels (faded = low confidence)",
                cv::Point(scatter.x + 14, scatter.y + 24), cv::FONT_HERSHEY_SIMPLEX,
                0.45, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);

    for (int k = 0; k < K; k++) {
        int x = scatter.x + 16 + k * 132;
        int y = scatter.y + scatter.height - 18;
        cv::circle(img, cv::Point(x, y - 4), 5, color_for_label(k), -1, cv::LINE_AA);
        cv::putText(img, cv::format("class %d", k), cv::Point(x + 12, y),
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
    cv::putText(img, cv::format("%d nodes, K=%d, %d seeds", N_NODES, K, data.seed_count),
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
    const std::string avi_path = "gif/gpu_label_propagation.avi";
    const std::string gif_path = "gif/gpu_label_propagation.gif";
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
    std::vector<float> init = make_initial_distribution(data);

    std::vector<float> gpu_f;
    std::vector<float> cpu_f;
    std::vector<Snapshot> snapshots;

    BenchResult bench;
    bench.gpu_ms = run_gpu_propagation(data, init, gpu_f, snapshots);
    bench.cpu_ms = run_cpu_propagation(data, init, cpu_f);

    Snapshot gpu_final = evaluate(gpu_f, data, LP_ITERS);
    Snapshot cpu_final = evaluate(cpu_f, data, LP_ITERS);
    bench.gpu_accuracy = gpu_final.accuracy;
    bench.cpu_accuracy = cpu_final.accuracy;
    bench.label_agreement = label_agreement(gpu_final.pred, cpu_final.pred);
    bench.speedup = bench.cpu_ms / std::max(1.0e-9, bench.gpu_ms);

    if (!snapshots.empty()) snapshots.back() = gpu_final;

    write_video(data, snapshots, bench);

    std::printf("GPU label propagation demo\n");
    std::printf("nodes: %d, classes: %d, seeds: %d, iterations: %d, sigma: %.3f\n",
                N_NODES, K, data.seed_count, LP_ITERS, SIGMA);
    std::printf("GPU propagation time: %.3f ms\n", bench.gpu_ms);
    std::printf("CPU propagation time: %.3f ms\n", bench.cpu_ms);
    std::printf("Speedup: %.1fx\n", bench.speedup);
    std::printf("GPU accuracy (unlabeled): %.4f\n", bench.gpu_accuracy);
    std::printf("CPU accuracy (unlabeled): %.4f\n", bench.cpu_accuracy);
    std::printf("GPU/CPU label agreement: %.4f\n", bench.label_agreement);
    std::printf("Wrote gif/gpu_label_propagation.gif\n");

    return 0;
}
