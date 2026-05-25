// gpu_graph_crf_traversability.cu
//
// GPU graph CRF traversability refinement.
//
// A synthetic terrain classifier emits noisy free/caution/blocked unary logits.
// A Potts-style CRF mean-field pass then refines those logits on an implicit
// bilateral graph over position, roughness, clearance, and height.  This is a
// robotics graph-ML primitive for turning noisy local perception into a cleaner
// traversability layer before planning.
//
// Output: gif/gpu_graph_crf_traversability.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int GRID_X = 64;
constexpr int GRID_Y = 48;
constexpr int N_NODES = GRID_X * GRID_Y;
constexpr int CRF_ITERS = 32;
constexpr int SNAP_STRIDE = 2;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int HEADER_H = 44;
constexpr int FOOTER_H = 40;
constexpr int MAP_H = PANEL_H - HEADER_H - FOOTER_H;
constexpr int HALF_W = PANEL_W / 2;
constexpr int VIDEO_FPS = 10;
constexpr int THREADS = 128;

constexpr float WORLD_W = 18.0f;
constexpr float WORLD_H = 11.0f;
constexpr float GRAPH_R = 1.45f;
constexpr float GRAPH_R2 = GRAPH_R * GRAPH_R;
constexpr float INV_TWO_SIGMA2 = 1.0f / (2.0f * 0.66f * 0.66f);
constexpr float UNARY_WEIGHT = 0.76f;
constexpr float PAIRWISE_WEIGHT = 1.65f;
constexpr float DAMPING = 0.12f;

struct Node {
    float x;
    float y;
    float roughness;
    float clearance;
    float height;
    int truth;
};

struct ClassVec {
    float free_v;
    float caution_v;
    float blocked_v;
};

struct Metrics {
    float accuracy = 0.0f;
    float entropy = 0.0f;
    int free_count = 0;
    int caution_count = 0;
    int blocked_count = 0;
};

struct Snapshot {
    int iter = 0;
    std::vector<ClassVec> probs;
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

__host__ __device__ static inline ClassVec normalize(ClassVec v) {
    v.free_v = fmaxf(v.free_v, 1.0e-6f);
    v.caution_v = fmaxf(v.caution_v, 1.0e-6f);
    v.blocked_v = fmaxf(v.blocked_v, 1.0e-6f);
    float inv = 1.0f / (v.free_v + v.caution_v + v.blocked_v);
    v.free_v *= inv;
    v.caution_v *= inv;
    v.blocked_v *= inv;
    return v;
}

__host__ __device__ static inline ClassVec softmax(ClassVec logits) {
    float m = fmaxf(logits.free_v, fmaxf(logits.caution_v, logits.blocked_v));
    ClassVec p{expf(logits.free_v - m),
               expf(logits.caution_v - m),
               expf(logits.blocked_v - m)};
    return normalize(p);
}

__host__ __device__ static inline ClassVec mix(ClassVec a, ClassVec b, float wa) {
    float wb = 1.0f - wa;
    return normalize({wa * a.free_v + wb * b.free_v,
                      wa * a.caution_v + wb * b.caution_v,
                      wa * a.blocked_v + wb * b.blocked_v});
}

__host__ __device__ static inline ClassVec sensor_logits(const Node& n) {
    float clear = clampf(n.clearance, 0.0f, 1.0f);
    float rough = clampf(n.roughness, 0.0f, 1.0f);
    float abs_height = fabsf(n.height);
    float low_clear = clampf((0.58f - clear) / 0.58f, 0.0f, 1.0f);
    float collision = clampf((0.18f - clear) / 0.18f, 0.0f, 1.0f);
    float boundary = 1.0f - clampf(fabsf(clear - 0.44f) / 0.44f, 0.0f, 1.0f);
    float height_caution = clampf((abs_height - 0.30f) / 0.34f, 0.0f, 1.0f);
    float height_free_gate = 1.0f - clampf((abs_height - 0.40f) / 0.34f, 0.0f, 1.0f);

    ClassVec logits;
    logits.free_v = 0.35f + 1.90f * clear * height_free_gate
                  - 1.22f * rough - 0.98f * height_caution - 1.16f * collision;
    logits.caution_v = 0.10f + 1.38f * boundary + 1.12f * rough
                     + 0.78f * height_caution - 0.18f * collision;
    logits.blocked_v = -0.24f + 1.95f * low_clear + 1.58f * sqr(rough)
                     + 1.12f * collision;
    return logits;
}

__host__ __device__ static inline float graph_weight(const Node& a, const Node& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float d2 = dx * dx + dy * dy;
    if (d2 > GRAPH_R2 || d2 < 1.0e-8f) return 0.0f;
    float feature = 1.60f * sqr(a.roughness - b.roughness)
                  + 1.30f * sqr(a.clearance - b.clearance)
                  + 0.95f * sqr(a.height - b.height);
    return expf(-d2 * INV_TWO_SIGMA2 - feature);
}

__host__ __device__ static inline int argmax_label(ClassVec v) {
    if (v.blocked_v > v.free_v && v.blocked_v > v.caution_v) return 2;
    if (v.caution_v > v.free_v) return 1;
    return 0;
}

__host__ __device__ static inline float entropy(ClassVec v) {
    return -(v.free_v * logf(fmaxf(v.free_v, 1.0e-6f))
           + v.caution_v * logf(fmaxf(v.caution_v, 1.0e-6f))
           + v.blocked_v * logf(fmaxf(v.blocked_v, 1.0e-6f))) / 1.0986122887f;
}

static float hash01(int i, int salt) {
    std::uint32_t x = static_cast<std::uint32_t>(i) * 747796405u
                    + static_cast<std::uint32_t>(salt) * 2891336453u
                    + 0x9e3779b9u;
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    return static_cast<float>(x & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

static void bump_label(ClassVec& v, int label, float amount) {
    if (label == 0) v.free_v += amount;
    if (label == 1) v.caution_v += amount;
    if (label == 2) v.blocked_v += amount;
}

static ClassVec make_noisy_unary(const Node& n, int i) {
    ClassVec logits = sensor_logits(n);
    logits.free_v += 0.34f * (hash01(i, 1) - 0.5f);
    logits.caution_v += 0.34f * (hash01(i, 2) - 0.5f);
    logits.blocked_v += 0.34f * (hash01(i, 3) - 0.5f);

    float corrupt = hash01(i, 4);
    if (corrupt < 0.16f) {
        int wrong = (n.truth + 1 + static_cast<int>(hash01(i, 5) > 0.48f)) % 3;
        bump_label(logits, wrong, 1.25f + 0.26f * hash01(i, 6));
        bump_label(logits, n.truth, -0.34f);
    } else {
        bump_label(logits, n.truth, 0.66f);
    }

    return logits;
}

__global__ void init_prob_kernel(const ClassVec* __restrict__ unary,
                                 ClassVec* __restrict__ q) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_NODES) return;
    q[i] = softmax(unary[i]);
}

__global__ void mean_field_kernel(const Node* __restrict__ nodes,
                                  const ClassVec* __restrict__ unary,
                                  const ClassVec* __restrict__ q_in,
                                  ClassVec* __restrict__ q_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_NODES) return;

    Node ni = nodes[i];
    float wsum = 0.0f;
    ClassVec msg{0.0f, 0.0f, 0.0f};
    for (int j = 0; j < N_NODES; j++) {
        if (j == i) continue;
        float w = graph_weight(ni, nodes[j]);
        if (w <= 0.0f) continue;
        msg.free_v += w * q_in[j].free_v;
        msg.caution_v += w * q_in[j].caution_v;
        msg.blocked_v += w * q_in[j].blocked_v;
        wsum += w;
    }

    if (wsum > 1.0e-6f) {
        float inv = 1.0f / wsum;
        msg.free_v *= inv;
        msg.caution_v *= inv;
        msg.blocked_v *= inv;
    } else {
        msg = q_in[i];
    }

    ClassVec logits;
    logits.free_v = UNARY_WEIGHT * unary[i].free_v + PAIRWISE_WEIGHT * msg.free_v;
    logits.caution_v = UNARY_WEIGHT * unary[i].caution_v + PAIRWISE_WEIGHT * msg.caution_v;
    logits.blocked_v = UNARY_WEIGHT * unary[i].blocked_v + PAIRWISE_WEIGHT * msg.blocked_v;
    q_out[i] = mix(softmax(logits), q_in[i], DAMPING);
}

static std::vector<Node> make_nodes() {
    std::vector<Node> nodes(N_NODES);
    std::mt19937 rng(25052026);
    std::uniform_real_distribution<float> jitter(-0.36f, 0.36f);

    for (int gy = 0; gy < GRID_Y; gy++) {
        for (int gx = 0; gx < GRID_X; gx++) {
            int i = gy * GRID_X + gx;
            float x = (static_cast<float>(gx) + 0.5f + jitter(rng)) / GRID_X * WORLD_W;
            float y = (static_cast<float>(gy) + 0.5f + jitter(rng)) / GRID_Y * WORLD_H;
            x = clampf(x, 0.04f, WORLD_W - 0.04f);
            y = clampf(y, 0.04f, WORLD_H - 0.04f);
            float h = terrain_height(x, y);
            float rough = terrain_roughness(x, y);
            float clear = clampf(raw_clearance(x, y) / 1.15f, 0.0f, 1.0f);
            int truth = terrain_truth(x, y, rough, clear, h);
            nodes[i] = {x, y, rough, clear, h, truth};
        }
    }
    return nodes;
}

static std::vector<ClassVec> make_unary(const std::vector<Node>& nodes) {
    std::vector<ClassVec> unary(N_NODES);
    for (int i = 0; i < N_NODES; i++) {
        unary[i] = make_noisy_unary(nodes[i], i);
    }
    return unary;
}

static void init_prob_host(const std::vector<ClassVec>& unary, std::vector<ClassVec>& q) {
    for (int i = 0; i < N_NODES; i++) {
        q[i] = softmax(unary[i]);
    }
}

static void mean_field_host(const std::vector<Node>& nodes,
                            const std::vector<ClassVec>& unary,
                            const std::vector<ClassVec>& q_in,
                            std::vector<ClassVec>& q_out) {
    for (int i = 0; i < N_NODES; i++) {
        const Node& ni = nodes[i];
        float wsum = 0.0f;
        ClassVec msg{0.0f, 0.0f, 0.0f};
        for (int j = 0; j < N_NODES; j++) {
            if (j == i) continue;
            float w = graph_weight(ni, nodes[j]);
            if (w <= 0.0f) continue;
            msg.free_v += w * q_in[j].free_v;
            msg.caution_v += w * q_in[j].caution_v;
            msg.blocked_v += w * q_in[j].blocked_v;
            wsum += w;
        }

        if (wsum > 1.0e-6f) {
            float inv = 1.0f / wsum;
            msg.free_v *= inv;
            msg.caution_v *= inv;
            msg.blocked_v *= inv;
        } else {
            msg = q_in[i];
        }

        ClassVec logits;
        logits.free_v = UNARY_WEIGHT * unary[i].free_v + PAIRWISE_WEIGHT * msg.free_v;
        logits.caution_v = UNARY_WEIGHT * unary[i].caution_v + PAIRWISE_WEIGHT * msg.caution_v;
        logits.blocked_v = UNARY_WEIGHT * unary[i].blocked_v + PAIRWISE_WEIGHT * msg.blocked_v;
        q_out[i] = mix(softmax(logits), q_in[i], DAMPING);
    }
}

static Metrics evaluate(const std::vector<Node>& nodes, const std::vector<ClassVec>& q) {
    Metrics m;
    int correct = 0;
    for (int i = 0; i < N_NODES; i++) {
        int pred = argmax_label(q[i]);
        if (pred == nodes[i].truth) correct++;
        if (pred == 0) m.free_count++;
        if (pred == 1) m.caution_count++;
        if (pred == 2) m.blocked_count++;
        m.entropy += entropy(q[i]);
    }
    m.accuracy = static_cast<float>(correct) / static_cast<float>(N_NODES);
    m.entropy /= static_cast<float>(N_NODES);
    return m;
}

static double cpu_crf_ms(const std::vector<Node>& nodes,
                         const std::vector<ClassVec>& unary,
                         Metrics& out_metrics) {
    std::vector<ClassVec> a(N_NODES);
    std::vector<ClassVec> b(N_NODES);
    init_prob_host(unary, a);
    std::vector<ClassVec>* in = &a;
    std::vector<ClassVec>* out = &b;

    auto begin = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < CRF_ITERS; iter++) {
        mean_field_host(nodes, unary, *in, *out);
        std::swap(in, out);
    }
    auto end = std::chrono::high_resolution_clock::now();
    out_metrics = evaluate(nodes, *in);
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

static cv::Point to_px(float x, float y, int x0) {
    int px = x0 + static_cast<int>(x / WORLD_W * (HALF_W - 1));
    int py = HEADER_H + static_cast<int>((1.0f - y / WORLD_H) * (MAP_H - 1));
    return cv::Point(px, py);
}

static cv::Scalar label_color(int label) {
    if (label == 0) return cv::Scalar(104, 222, 154);
    if (label == 1) return cv::Scalar(68, 186, 244);
    return cv::Scalar(92, 90, 248);
}

static cv::Scalar truth_tint(int label) {
    if (label == 0) return cv::Scalar(37, 53, 44);
    if (label == 1) return cv::Scalar(50, 55, 36);
    return cv::Scalar(54, 36, 41);
}

static void draw_background_panel(cv::Mat& img, int x0) {
    for (int iy = 0; iy < MAP_H; iy += 4) {
        for (int ix = 0; ix < HALF_W; ix += 4) {
            float x = static_cast<float>(ix) / HALF_W * WORLD_W;
            float y = (1.0f - static_cast<float>(iy) / MAP_H) * WORLD_H;
            float h = terrain_height(x, y);
            float rough = terrain_roughness(x, y);
            float clear = clampf(raw_clearance(x, y) / 1.15f, 0.0f, 1.0f);
            int truth = terrain_truth(x, y, rough, clear, h);
            cv::rectangle(img, cv::Rect(x0 + ix, HEADER_H + iy, 4, 4),
                          truth_tint(truth), cv::FILLED);
        }
    }
}

static void draw_panel_points(cv::Mat& img,
                              const std::vector<Node>& nodes,
                              const std::vector<ClassVec>& q,
                              int x0,
                              bool mark_errors) {
    for (int i = 0; i < N_NODES; i++) {
        int pred = argmax_label(q[i]);
        cv::Point p = to_px(nodes[i].x, nodes[i].y, x0);
        cv::circle(img, p, 2, label_color(pred), cv::FILLED, cv::LINE_AA);
        if (mark_errors && pred != nodes[i].truth && i % 2 == 0) {
            cv::circle(img, p, 4, cv::Scalar(246, 246, 246), 1, cv::LINE_AA);
        }
    }
}

static void draw_legend(cv::Mat& img) {
    cv::rectangle(img, cv::Rect(PANEL_W - 250, 48, 232, 88), cv::Scalar(8, 10, 13), cv::FILLED);
    cv::circle(img, cv::Point(PANEL_W - 230, 70), 5, label_color(0), cv::FILLED, cv::LINE_AA);
    cv::putText(img, "free", cv::Point(PANEL_W - 214, 75),
                cv::FONT_HERSHEY_SIMPLEX, 0.43, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::circle(img, cv::Point(PANEL_W - 230, 96), 5, label_color(1), cv::FILLED, cv::LINE_AA);
    cv::putText(img, "caution", cv::Point(PANEL_W - 214, 101),
                cv::FONT_HERSHEY_SIMPLEX, 0.43, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::circle(img, cv::Point(PANEL_W - 230, 122), 5, label_color(2), cv::FILLED, cv::LINE_AA);
    cv::putText(img, "blocked", cv::Point(PANEL_W - 214, 127),
                cv::FONT_HERSHEY_SIMPLEX, 0.43, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
}

static cv::Mat draw_frame(const std::vector<Node>& nodes,
                          const std::vector<ClassVec>& unary_probs,
                          const Metrics& unary_metrics,
                          const Snapshot& snap,
                          double gpu_ms,
                          double cpu_ms,
                          const Metrics& cpu_metrics) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 20, 24));
    draw_background_panel(img, 0);
    draw_background_panel(img, HALF_W);
    cv::line(img, cv::Point(HALF_W, HEADER_H), cv::Point(HALF_W, PANEL_H - FOOTER_H),
             cv::Scalar(18, 18, 20), 1, cv::LINE_AA);

    draw_panel_points(img, nodes, unary_probs, 0, true);
    draw_panel_points(img, nodes, snap.probs, HALF_W, true);
    draw_legend(img);

    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, HEADER_H), cv::Scalar(5, 7, 10), cv::FILLED);
    cv::rectangle(img, cv::Rect(0, PANEL_H - FOOTER_H, PANEL_W, FOOTER_H),
                  cv::Scalar(5, 7, 10), cv::FILLED);

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "GPU graph CRF traversability  nodes=%d  iters=%d  gpu=%.2f ms  cpu=%.1f ms",
                  N_NODES, CRF_ITERS, gpu_ms, cpu_ms);
    cv::putText(img, buf, cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);

    cv::putText(img, "noisy unary", cv::Point(14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.54, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "CRF refined", cv::Point(HALF_W + 14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.54, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);

    std::snprintf(buf, sizeof(buf),
                  "iter %02d  unary acc=%.1f%%  refined acc=%.1f%%  cpu acc=%.1f%%  entropy=%.3f",
                  snap.iter, 100.0f * unary_metrics.accuracy,
                  100.0f * snap.metrics.accuracy,
                  100.0f * cpu_metrics.accuracy,
                  snap.metrics.entropy);
    cv::putText(img, buf, cv::Point(12, PANEL_H - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.51, cv::Scalar(225, 238, 245), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<Node> nodes = make_nodes();
    std::vector<ClassVec> unary = make_unary(nodes);
    std::vector<ClassVec> unary_probs(N_NODES);
    init_prob_host(unary, unary_probs);
    Metrics unary_metrics = evaluate(nodes, unary_probs);

    Metrics cpu_metrics;
    double cpu_ms = cpu_crf_ms(nodes, unary, cpu_metrics);

    Node* d_nodes = nullptr;
    ClassVec* d_unary = nullptr;
    ClassVec* d_a = nullptr;
    ClassVec* d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nodes, N_NODES * sizeof(Node)));
    CUDA_CHECK(cudaMalloc(&d_unary, N_NODES * sizeof(ClassVec)));
    CUDA_CHECK(cudaMalloc(&d_a, N_NODES * sizeof(ClassVec)));
    CUDA_CHECK(cudaMalloc(&d_b, N_NODES * sizeof(ClassVec)));
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes.data(), N_NODES * sizeof(Node), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_unary, unary.data(), N_NODES * sizeof(ClassVec), cudaMemcpyHostToDevice));

    int blocks = (N_NODES + THREADS - 1) / THREADS;
    init_prob_kernel<<<blocks, THREADS>>>(d_unary, d_a);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    ClassVec* in = d_a;
    ClassVec* out = d_b;
    for (int iter = 0; iter < CRF_ITERS; iter++) {
        mean_field_kernel<<<blocks, THREADS>>>(d_nodes, d_unary, in, out);
        ClassVec* tmp = in;
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
    std::vector<ClassVec> h_probs(N_NODES);
    init_prob_kernel<<<blocks, THREADS>>>(d_unary, d_a);
    CUDA_CHECK(cudaDeviceSynchronize());
    in = d_a;
    out = d_b;
    CUDA_CHECK(cudaMemcpy(h_probs.data(), in, N_NODES * sizeof(ClassVec), cudaMemcpyDeviceToHost));
    snapshots.push_back({0, h_probs, evaluate(nodes, h_probs)});
    for (int iter = 1; iter <= CRF_ITERS; iter++) {
        mean_field_kernel<<<blocks, THREADS>>>(d_nodes, d_unary, in, out);
        CUDA_CHECK(cudaDeviceSynchronize());
        ClassVec* tmp = in;
        in = out;
        out = tmp;
        if (iter % SNAP_STRIDE == 0 || iter == CRF_ITERS) {
            CUDA_CHECK(cudaMemcpy(h_probs.data(), in, N_NODES * sizeof(ClassVec),
                                  cudaMemcpyDeviceToHost));
            snapshots.push_back({iter, h_probs, evaluate(nodes, h_probs)});
        }
    }

    double speedup = cpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0;
    const Metrics& final_gpu = snapshots.back().metrics;
    std::printf("Noisy unary traversability: accuracy %.2f%%, entropy %.3f\n",
                100.0f * unary_metrics.accuracy, unary_metrics.entropy);
    std::printf("CPU graph CRF: %.3f ms, accuracy %.2f%%\n",
                cpu_ms, 100.0f * cpu_metrics.accuracy);
    std::printf("GPU graph CRF: %.3f ms (%d nodes x %d iters, %.1fx vs CPU, accuracy %.2f%%, entropy %.3f)\n",
                gpu_ms, N_NODES, CRF_ITERS, speedup, 100.0f * final_gpu.accuracy,
                final_gpu.entropy);

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_graph_crf_traversability.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_graph_crf_traversability.avi\n");
        return 1;
    }
    for (const Snapshot& s : snapshots) {
        video.write(draw_frame(nodes, unary_probs, unary_metrics, s, gpu_ms, cpu_ms, cpu_metrics));
    }
    for (int i = 0; i < 14; i++) {
        video.write(draw_frame(nodes, unary_probs, unary_metrics, snapshots.back(),
                               gpu_ms, cpu_ms, cpu_metrics));
    }
    video.release();

    avi_to_gif("gif/gpu_graph_crf_traversability.avi",
               "gif/gpu_graph_crf_traversability.gif", 8, 560);
    std::printf("GIF saved to gif/gpu_graph_crf_traversability.gif\n");

    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_unary));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
