// gpu_label_propagation_traversability.cu
//
// GPU semi-supervised traversability label propagation.
//
// A sparse set of labeled terrain waypoints is expanded over an implicit
// graph using RBF edge weights in position + terrain-feature space.  This is a
// compact graph-ML primitive for robotics: turn a handful of inspected cells
// into dense free/caution/blocked labels that a planner can consume.
//
// Output: gif/gpu_label_propagation_traversability.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
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
constexpr int PROP_ITERS = 40;
constexpr int SNAP_STRIDE = 2;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int VIDEO_FPS = 10;
constexpr int THREADS = 128;

constexpr float WORLD_W = 18.0f;
constexpr float WORLD_H = 11.0f;
constexpr float GRAPH_R = 1.55f;
constexpr float GRAPH_R2 = GRAPH_R * GRAPH_R;
constexpr float INV_TWO_SIGMA2 = 1.0f / (2.0f * 0.72f * 0.72f);

struct Node {
    float x;
    float y;
    float roughness;
    float clearance;
    float height;
    int truth;
    int seed;
};

struct LabelVec {
    float free_p;
    float caution_p;
    float blocked_p;
};

struct Metrics {
    float accuracy = 0.0f;
    float entropy = 0.0f;
    int seeds = 0;
    int free_count = 0;
    int caution_count = 0;
    int blocked_count = 0;
};

struct Snapshot {
    int iter = 0;
    std::vector<LabelVec> scores;
    Metrics metrics;
};

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline float sqr(float x) {
    return x * x;
}

__host__ __device__ static inline float circle_clearance(float x,
                                                         float y,
                                                         float cx,
                                                         float cy,
                                                         float r) {
    float dx = x - cx;
    float dy = y - cy;
    return sqrtf(dx * dx + dy * dy) - r;
}

__host__ __device__ static inline float terrain_height(float x, float y) {
    return 0.36f * sinf(0.52f * x + 0.35f * y)
         + 0.23f * cosf(0.72f * x - 0.40f * y)
         + 0.16f * sinf(1.16f * y);
}

__host__ __device__ static inline float raw_clearance(float x, float y) {
    float d = circle_clearance(x, y, 4.4f, 3.1f, 1.05f);
    d = fminf(d, circle_clearance(x, y, 7.2f, 7.8f, 1.15f));
    d = fminf(d, circle_clearance(x, y, 11.5f, 4.6f, 1.25f));
    d = fminf(d, circle_clearance(x, y, 14.0f, 8.4f, 0.92f));
    float ridge = fabsf(y - (5.3f + 0.75f * sinf(0.62f * x))) - 0.25f;
    return fminf(d, ridge);
}

__host__ __device__ static inline float terrain_roughness(float x, float y) {
    float h0 = terrain_height(x, y);
    float hx = terrain_height(x + 0.18f, y);
    float hy = terrain_height(x, y + 0.18f);
    float slope = sqrtf(sqr(hx - h0) + sqr(hy - h0)) / 0.18f;
    float rough_patch = expf(-0.26f * (sqr(x - 13.7f) + sqr(y - 2.8f)))
                      + 0.85f * expf(-0.22f * (sqr(x - 2.8f) + sqr(y - 8.8f)));
    return clampf(0.15f + 0.55f * slope + 0.36f * rough_patch, 0.0f, 1.0f);
}

__host__ __device__ static inline int terrain_truth(float x,
                                                    float y,
                                                    float roughness,
                                                    float clearance,
                                                    float height) {
    float signed_clearance = raw_clearance(x, y);
    if (signed_clearance < -0.04f || roughness > 0.82f) return 2;
    if (signed_clearance < 0.58f || roughness > 0.52f || fabsf(height) > 0.46f) return 1;
    if (clearance < 0.28f) return 1;
    return 0;
}

__host__ __device__ static inline LabelVec normalize(LabelVec v) {
    v.free_p = fmaxf(v.free_p, 1.0e-6f);
    v.caution_p = fmaxf(v.caution_p, 1.0e-6f);
    v.blocked_p = fmaxf(v.blocked_p, 1.0e-6f);
    float inv = 1.0f / (v.free_p + v.caution_p + v.blocked_p);
    v.free_p *= inv;
    v.caution_p *= inv;
    v.blocked_p *= inv;
    return v;
}

__host__ __device__ static inline LabelVec one_hot(int label) {
    LabelVec v{0.02f, 0.02f, 0.02f};
    if (label == 0) v.free_p = 1.0f;
    if (label == 1) v.caution_p = 1.0f;
    if (label == 2) v.blocked_p = 1.0f;
    return normalize(v);
}

__host__ __device__ static inline LabelVec feature_prior(const Node& n) {
    float clear = clampf(n.clearance, 0.0f, 1.0f);
    float rough = clampf(n.roughness, 0.0f, 1.0f);
    float abs_height = fabsf(n.height);
    float height_caution = clampf((abs_height - 0.32f) / 0.30f, 0.0f, 1.0f);
    float height_free_gate = 1.0f - clampf((abs_height - 0.42f) / 0.35f, 0.0f, 1.0f);
    float free_score = 0.14f + 1.08f * clear * (1.0f - rough) * height_free_gate;
    float caution_band = 1.0f - clampf(fabsf(clear - 0.44f) / 0.44f, 0.0f, 1.0f);
    float caution_score = 0.18f + 0.88f * caution_band + 0.52f * rough + 0.92f * height_caution;
    float blocked_score = 0.10f + 1.18f * sqr(1.0f - clear) + 0.88f * sqr(rough);
    return normalize({free_score, caution_score, blocked_score});
}

__host__ __device__ static inline float graph_weight(const Node& a, const Node& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float d2 = dx * dx + dy * dy;
    if (d2 > GRAPH_R2 || d2 < 1.0e-8f) return 0.0f;
    float feature = 1.35f * sqr(a.roughness - b.roughness)
                  + 1.05f * sqr(a.clearance - b.clearance)
                  + 0.80f * sqr(a.height - b.height);
    return expf(-d2 * INV_TWO_SIGMA2 - feature);
}

__host__ __device__ static inline int argmax_label(LabelVec v) {
    if (v.blocked_p > v.free_p && v.blocked_p > v.caution_p) return 2;
    if (v.caution_p > v.free_p) return 1;
    return 0;
}

__host__ __device__ static inline float entropy(LabelVec v) {
    return -(v.free_p * logf(fmaxf(v.free_p, 1.0e-6f))
           + v.caution_p * logf(fmaxf(v.caution_p, 1.0e-6f))
           + v.blocked_p * logf(fmaxf(v.blocked_p, 1.0e-6f))) / 1.0986122887f;
}

__global__ void init_scores_kernel(const Node* __restrict__ nodes,
                                   LabelVec* __restrict__ scores) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_NODES) return;
    scores[i] = nodes[i].seed ? one_hot(nodes[i].truth) : LabelVec{0.34f, 0.33f, 0.33f};
}

__global__ void propagate_kernel(const Node* __restrict__ nodes,
                                 const LabelVec* __restrict__ in,
                                 LabelVec* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_NODES) return;
    Node ni = nodes[i];
    if (ni.seed) {
        out[i] = one_hot(ni.truth);
        return;
    }

    float wsum = 0.0f;
    LabelVec acc{0.0f, 0.0f, 0.0f};
    for (int j = 0; j < N_NODES; j++) {
        if (j == i) continue;
        float w = graph_weight(ni, nodes[j]);
        if (w <= 0.0f) continue;
        acc.free_p += w * in[j].free_p;
        acc.caution_p += w * in[j].caution_p;
        acc.blocked_p += w * in[j].blocked_p;
        wsum += w;
    }

    LabelVec prior = feature_prior(ni);
    if (wsum > 1.0e-6f) {
        float inv = 1.0f / wsum;
        acc.free_p *= inv;
        acc.caution_p *= inv;
        acc.blocked_p *= inv;
    } else {
        acc = prior;
    }

    LabelVec blended;
    blended.free_p = 0.66f * acc.free_p + 0.34f * prior.free_p;
    blended.caution_p = 0.66f * acc.caution_p + 0.34f * prior.caution_p;
    blended.blocked_p = 0.66f * acc.blocked_p + 0.34f * prior.blocked_p;
    out[i] = normalize(blended);
}

static std::vector<Node> make_nodes() {
    std::vector<Node> nodes(N_NODES);
    std::mt19937 rng(25052026);
    std::uniform_real_distribution<float> ux(0.0f, WORLD_W);
    std::uniform_real_distribution<float> uy(0.0f, WORLD_H);
    std::uniform_real_distribution<float> jitter(-0.10f, 0.10f);

    for (int i = 0; i < N_NODES; i++) {
        float x = ux(rng);
        float y = uy(rng);
        if (i < N_NODES / 3) {
            float t = static_cast<float>(i) / static_cast<float>(N_NODES / 3 - 1);
            x = 1.2f + t * (WORLD_W - 2.4f) + jitter(rng);
            y = 2.0f + 6.8f * t + 0.55f * sinf(8.0f * t) + jitter(rng);
        }

        float h = terrain_height(x, y);
        float rough = terrain_roughness(x, y);
        float clear = clampf(raw_clearance(x, y) / 1.15f, 0.0f, 1.0f);
        int truth = terrain_truth(x, y, rough, clear, h);
        bool seed = (i % 13 == 0) || (truth == 2 && i % 29 == 0) || (truth == 1 && i % 31 == 0);
        nodes[i] = {x, y, rough, clear, h, truth, seed ? 1 : 0};
    }
    return nodes;
}

static void init_scores_host(const std::vector<Node>& nodes, std::vector<LabelVec>& scores) {
    for (int i = 0; i < N_NODES; i++) {
        scores[i] = nodes[i].seed ? one_hot(nodes[i].truth) : LabelVec{0.34f, 0.33f, 0.33f};
    }
}

static void propagate_host(const std::vector<Node>& nodes,
                           const std::vector<LabelVec>& in,
                           std::vector<LabelVec>& out) {
    for (int i = 0; i < N_NODES; i++) {
        const Node& ni = nodes[i];
        if (ni.seed) {
            out[i] = one_hot(ni.truth);
            continue;
        }

        float wsum = 0.0f;
        LabelVec acc{0.0f, 0.0f, 0.0f};
        for (int j = 0; j < N_NODES; j++) {
            if (j == i) continue;
            float w = graph_weight(ni, nodes[j]);
            if (w <= 0.0f) continue;
            acc.free_p += w * in[j].free_p;
            acc.caution_p += w * in[j].caution_p;
            acc.blocked_p += w * in[j].blocked_p;
            wsum += w;
        }

        LabelVec prior = feature_prior(ni);
        if (wsum > 1.0e-6f) {
            float inv = 1.0f / wsum;
            acc.free_p *= inv;
            acc.caution_p *= inv;
            acc.blocked_p *= inv;
        } else {
            acc = prior;
        }

        LabelVec blended;
        blended.free_p = 0.66f * acc.free_p + 0.34f * prior.free_p;
        blended.caution_p = 0.66f * acc.caution_p + 0.34f * prior.caution_p;
        blended.blocked_p = 0.66f * acc.blocked_p + 0.34f * prior.blocked_p;
        out[i] = normalize(blended);
    }
}

static Metrics evaluate(const std::vector<Node>& nodes, const std::vector<LabelVec>& scores) {
    Metrics m;
    int correct = 0;
    for (int i = 0; i < N_NODES; i++) {
        int pred = argmax_label(scores[i]);
        if (pred == nodes[i].truth) correct++;
        if (nodes[i].seed) m.seeds++;
        if (pred == 0) m.free_count++;
        if (pred == 1) m.caution_count++;
        if (pred == 2) m.blocked_count++;
        m.entropy += entropy(scores[i]);
    }
    m.accuracy = static_cast<float>(correct) / static_cast<float>(N_NODES);
    m.entropy /= static_cast<float>(N_NODES);
    return m;
}

static double cpu_propagation_ms(const std::vector<Node>& nodes, Metrics& out_metrics) {
    std::vector<LabelVec> a(N_NODES);
    std::vector<LabelVec> b(N_NODES);
    init_scores_host(nodes, a);
    std::vector<LabelVec>* in = &a;
    std::vector<LabelVec>* out = &b;

    auto begin = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < PROP_ITERS; iter++) {
        propagate_host(nodes, *in, *out);
        std::swap(in, out);
    }
    auto end = std::chrono::high_resolution_clock::now();
    out_metrics = evaluate(nodes, *in);
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

static cv::Point to_px(float x, float y) {
    int px = static_cast<int>(x / WORLD_W * PANEL_W);
    int py = static_cast<int>((1.0f - y / WORLD_H) * PANEL_H);
    return cv::Point(px, py);
}

static cv::Scalar label_color(int label) {
    if (label == 0) return cv::Scalar(104, 222, 154);
    if (label == 1) return cv::Scalar(68, 186, 244);
    return cv::Scalar(92, 90, 248);
}

static cv::Scalar truth_tint(int label) {
    if (label == 0) return cv::Scalar(38, 55, 45);
    if (label == 1) return cv::Scalar(49, 55, 36);
    return cv::Scalar(54, 36, 40);
}

static void draw_background(cv::Mat& img) {
    img = cv::Mat(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(20, 22, 25));
    for (int iy = 0; iy < PANEL_H; iy += 4) {
        for (int ix = 0; ix < PANEL_W; ix += 4) {
            float x = static_cast<float>(ix) / PANEL_W * WORLD_W;
            float y = (1.0f - static_cast<float>(iy) / PANEL_H) * WORLD_H;
            float h = terrain_height(x, y);
            float rough = terrain_roughness(x, y);
            float clear = clampf(raw_clearance(x, y) / 1.15f, 0.0f, 1.0f);
            int truth = terrain_truth(x, y, rough, clear, h);
            cv::rectangle(img, cv::Rect(ix, iy, 4, 4), truth_tint(truth), cv::FILLED);
        }
    }
}

static void draw_seed_legend(cv::Mat& img) {
    cv::rectangle(img, cv::Rect(PANEL_W - 222, 44, 204, 92), cv::Scalar(8, 10, 13), cv::FILLED);
    cv::circle(img, cv::Point(PANEL_W - 202, 68), 5, label_color(0), cv::FILLED, cv::LINE_AA);
    cv::putText(img, "free", cv::Point(PANEL_W - 186, 73),
                cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::circle(img, cv::Point(PANEL_W - 202, 94), 5, label_color(1), cv::FILLED, cv::LINE_AA);
    cv::putText(img, "caution", cv::Point(PANEL_W - 186, 99),
                cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::circle(img, cv::Point(PANEL_W - 202, 120), 5, label_color(2), cv::FILLED, cv::LINE_AA);
    cv::putText(img, "blocked", cv::Point(PANEL_W - 186, 125),
                cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
}

static cv::Mat draw_frame(const std::vector<Node>& nodes,
                          const Snapshot& snap,
                          double gpu_ms,
                          double cpu_ms,
                          const Metrics& cpu_metrics) {
    cv::Mat img;
    draw_background(img);

    for (int i = 0; i < N_NODES; i++) {
        int pred = argmax_label(snap.scores[i]);
        cv::Point p = to_px(nodes[i].x, nodes[i].y);
        int radius = nodes[i].seed ? 4 : 2;
        cv::circle(img, p, radius, label_color(pred), cv::FILLED, cv::LINE_AA);
        if (nodes[i].seed) {
            cv::circle(img, p, 6, cv::Scalar(242, 242, 242), 1, cv::LINE_AA);
        } else if (pred != nodes[i].truth && i % 3 == 0) {
            cv::circle(img, p, 4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }

    draw_seed_legend(img);
    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, 36), cv::Scalar(5, 7, 10), cv::FILLED);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "GPU label propagation traversability  nodes=%d  iters=%d  gpu=%.2f ms  cpu=%.1f ms",
                  N_NODES, PROP_ITERS, gpu_ms, cpu_ms);
    cv::putText(img, buf, cv::Point(12, 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.53, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);

    std::snprintf(buf, sizeof(buf),
                  "iter %02d  acc=%.1f%%  cpu acc=%.1f%%  seeds=%d  entropy=%.3f",
                  snap.iter, 100.0f * snap.metrics.accuracy,
                  100.0f * cpu_metrics.accuracy, snap.metrics.seeds, snap.metrics.entropy);
    cv::putText(img, buf, cv::Point(12, PANEL_H - 16),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(225, 238, 245), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<Node> nodes = make_nodes();
    Metrics cpu_metrics;
    double cpu_ms = cpu_propagation_ms(nodes, cpu_metrics);

    Node* d_nodes = nullptr;
    LabelVec* d_a = nullptr;
    LabelVec* d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nodes, N_NODES * sizeof(Node)));
    CUDA_CHECK(cudaMalloc(&d_a, N_NODES * sizeof(LabelVec)));
    CUDA_CHECK(cudaMalloc(&d_b, N_NODES * sizeof(LabelVec)));
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes.data(), N_NODES * sizeof(Node), cudaMemcpyHostToDevice));

    int blocks = (N_NODES + THREADS - 1) / THREADS;
    init_scores_kernel<<<blocks, THREADS>>>(d_nodes, d_a);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    LabelVec* in = d_a;
    LabelVec* out = d_b;
    for (int iter = 0; iter < PROP_ITERS; iter++) {
        propagate_kernel<<<blocks, THREADS>>>(d_nodes, in, out);
        LabelVec* tmp = in;
        in = out;
        out = tmp;
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float gpu_ms_f = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms_f, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaGetLastError());
    double gpu_ms = static_cast<double>(gpu_ms_f);

    std::vector<Snapshot> snapshots;
    std::vector<LabelVec> h_scores(N_NODES);
    init_scores_kernel<<<blocks, THREADS>>>(d_nodes, d_a);
    CUDA_CHECK(cudaDeviceSynchronize());
    in = d_a;
    out = d_b;
    CUDA_CHECK(cudaMemcpy(h_scores.data(), in, N_NODES * sizeof(LabelVec), cudaMemcpyDeviceToHost));
    snapshots.push_back({0, h_scores, evaluate(nodes, h_scores)});
    for (int iter = 1; iter <= PROP_ITERS; iter++) {
        propagate_kernel<<<blocks, THREADS>>>(d_nodes, in, out);
        CUDA_CHECK(cudaDeviceSynchronize());
        LabelVec* tmp = in;
        in = out;
        out = tmp;
        if (iter % SNAP_STRIDE == 0 || iter == PROP_ITERS) {
            CUDA_CHECK(cudaMemcpy(h_scores.data(), in, N_NODES * sizeof(LabelVec),
                                  cudaMemcpyDeviceToHost));
            snapshots.push_back({iter, h_scores, evaluate(nodes, h_scores)});
        }
    }

    double speedup = cpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0;
    const Metrics& final_gpu = snapshots.back().metrics;
    std::printf("CPU label propagation: %.3f ms, accuracy %.2f%%\n",
                cpu_ms, 100.0f * cpu_metrics.accuracy);
    std::printf("GPU label propagation: %.3f ms (%d nodes x %d iters, %.1fx vs CPU, accuracy %.2f%%)\n",
                gpu_ms, N_NODES, PROP_ITERS, speedup, 100.0f * final_gpu.accuracy);

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_label_propagation_traversability.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_label_propagation_traversability.avi\n");
        return 1;
    }
    for (const Snapshot& s : snapshots) {
        video.write(draw_frame(nodes, s, gpu_ms, cpu_ms, cpu_metrics));
    }
    for (int i = 0; i < 14; i++) {
        video.write(draw_frame(nodes, snapshots.back(), gpu_ms, cpu_ms, cpu_metrics));
    }
    video.release();

    avi_to_gif("gif/gpu_label_propagation_traversability.avi",
               "gif/gpu_label_propagation_traversability.gif", 8, 560);
    std::printf("GIF saved to gif/gpu_label_propagation_traversability.gif\n");

    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
