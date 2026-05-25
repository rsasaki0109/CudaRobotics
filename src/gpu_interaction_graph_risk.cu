// gpu_interaction_graph_risk.cu
//
// GPU interaction-graph risk propagation.
//
// Multi-agent planners often need a compact answer to "where will conflict
// spread next?"  This demo builds an implicit graph over robots/pedestrians:
// edge weights combine distance, closing speed, and time-to-collision.  A few
// high-risk seeds then diffuse through the graph with message passing, giving
// a per-agent risk label that can feed a local planner, assignment tracker, or
// crowd controller.
//
// Output: gif/gpu_interaction_graph_risk.gif

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

constexpr int N_AGENTS = 2048;
constexpr int PROP_ITERS = 10;
constexpr int N_FRAMES = 96;
constexpr int N_HOLD_FRAMES = 18;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int VIDEO_FPS = 12;

constexpr float WORLD_W = 18.0f;
constexpr float WORLD_H = 11.0f;
constexpr float DT = 0.10f;
constexpr float EDGE_R = 2.25f;
constexpr float EDGE_R2 = EDGE_R * EDGE_R;
constexpr float TTC_HORIZON = 2.8f;
constexpr float COLLISION_R = 0.34f;
constexpr float RISK_DECAY = 0.62f;
constexpr float PI_F = 3.14159265358979323846f;

struct Agent {
    float x;
    float y;
    float vx;
    float vy;
    float speed;
    float lane;
    float phase;
    int route;
    int kind;
};

struct Metrics {
    float max_risk = 0.0f;
    float mean_risk = 0.0f;
    int hot_agents = 0;
};

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline float interaction_weight(const Agent& a, const Agent& b) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float d2 = dx * dx + dy * dy;
    if (d2 > EDGE_R2 || d2 < 1.0e-8f) return 0.0f;

    float d = sqrtf(d2);
    float rvx = b.vx - a.vx;
    float rvy = b.vy - a.vy;
    float closing = -(dx * rvx + dy * rvy) / (d + 1.0e-5f);
    float ttc = closing > 0.04f ? d / closing : 1.0e6f;

    bool same_axis = (a.route & 1) == (b.route & 1);
    float proximity = expf(-d2 / (2.0f * EDGE_R2));
    float ttc_w = ttc < TTC_HORIZON ? (TTC_HORIZON - ttc) / TTC_HORIZON : 0.0f;
    if (same_axis && (d > 0.45f || ttc_w < 0.80f)) return 0.0f;
    if (!same_axis && ttc_w <= 0.0f && d > 0.95f) return 0.0f;
    float robot_boost = (a.kind == 1 || b.kind == 1) ? 1.18f : 1.0f;
    float route_boost = same_axis ? 0.04f : 1.0f;
    return clampf(robot_boost * route_boost * (0.08f * proximity + 0.82f * ttc_w), 0.0f, 1.0f);
}

__host__ __device__ static inline float seed_pair_risk(const Agent& a, const Agent& b) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float d2 = dx * dx + dy * dy;
    if (d2 > EDGE_R2 || d2 < 1.0e-8f) return 0.0f;
    float d = sqrtf(d2);
    float rvx = b.vx - a.vx;
    float rvy = b.vy - a.vy;
    float closing = -(dx * rvx + dy * rvy) / (d + 1.0e-5f);
    float ttc = closing > 0.04f ? d / closing : 1.0e6f;
    bool same_axis = (a.route & 1) == (b.route & 1);
    float near_radius = same_axis ? 0.34f : 0.95f;
    float near = d < COLLISION_R ? 1.0f : clampf((near_radius - d) / near_radius, 0.0f, 1.0f);
    float future = ttc < TTC_HORIZON ? (TTC_HORIZON - ttc) / TTC_HORIZON : 0.0f;
    float route_boost = same_axis ? 0.02f : 1.0f;
    return clampf(route_boost * (0.92f * future + 0.50f * near), 0.0f, 1.0f);
}

__host__ __device__ static inline void route_velocity(int route,
                                                       float speed,
                                                       float phase,
                                                       float x,
                                                       float y,
                                                       float& vx,
                                                       float& vy) {
    float wiggle = 0.10f * sinf(0.35f * x + 0.27f * y + phase);
    if (route == 0) {
        vx = speed;
        vy = wiggle;
    } else if (route == 1) {
        vx = wiggle;
        vy = speed;
    } else if (route == 2) {
        vx = -speed;
        vy = wiggle;
    } else {
        vx = wiggle;
        vy = -speed;
    }
}

__global__ void update_agents_kernel(Agent* agents, int frame) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_AGENTS) return;
    Agent a = agents[i];
    route_velocity(a.route, a.speed, a.phase + 0.015f * frame, a.x, a.y, a.vx, a.vy);
    a.x += a.vx * DT;
    a.y += a.vy * DT;

    if (a.route == 0 && a.x > WORLD_W + 0.4f) a.x -= WORLD_W + 0.8f;
    if (a.route == 2 && a.x < -0.4f) a.x += WORLD_W + 0.8f;
    if (a.route == 1 && a.y > WORLD_H + 0.4f) a.y -= WORLD_H + 0.8f;
    if (a.route == 3 && a.y < -0.4f) a.y += WORLD_H + 0.8f;

    a.x = clampf(a.x, -0.45f, WORLD_W + 0.45f);
    a.y = clampf(a.y, -0.45f, WORLD_H + 0.45f);
    agents[i] = a;
}

__global__ void seed_risk_kernel(const Agent* __restrict__ agents,
                                 float* __restrict__ base,
                                 float* __restrict__ risk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_AGENTS) return;
    Agent a = agents[i];
    float r = 0.0f;
    for (int j = 0; j < N_AGENTS; j++) {
        if (j == i) continue;
        float p = seed_pair_risk(a, agents[j]);
        if (p > r) r = p;
    }
    base[i] = r;
    risk[i] = r;
}

__global__ void propagate_risk_kernel(const Agent* __restrict__ agents,
                                      const float* __restrict__ base,
                                      const float* __restrict__ risk_in,
                                      float* __restrict__ risk_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_AGENTS) return;
    Agent a = agents[i];
    float weighted = 0.0f;
    float wsum = 0.0f;
    float max_msg = 0.0f;
    for (int j = 0; j < N_AGENTS; j++) {
        if (j == i) continue;
        float w = interaction_weight(a, agents[j]);
        if (w <= 0.015f) continue;
        float msg = w * risk_in[j];
        weighted += msg;
        wsum += w;
        if (msg > max_msg) max_msg = msg;
    }
    float mean_msg = wsum > 1.0e-6f ? weighted / wsum : 0.0f;
    float propagated = RISK_DECAY * (0.58f * max_msg + 0.42f * mean_msg);
    risk_out[i] = clampf(fmaxf(base[i], propagated), 0.0f, 1.0f);
}

static std::vector<Agent> make_agents() {
    std::vector<Agent> agents(N_AGENTS);
    std::mt19937 rng(25052026);
    std::uniform_real_distribution<float> ux(0.0f, WORLD_W);
    std::uniform_real_distribution<float> uy(0.0f, WORLD_H);
    std::uniform_real_distribution<float> up(0.0f, 2.0f * PI_F);
    std::normal_distribution<float> lane_noise(0.0f, 0.34f);
    std::uniform_real_distribution<float> speed_jitter(-0.18f, 0.22f);

    for (int i = 0; i < N_AGENTS; i++) {
        int route = i % 4;
        Agent a{};
        a.route = route;
        a.kind = (i % 17 == 0) ? 1 : 0;
        a.speed = (a.kind == 1 ? 1.52f : 1.08f) + speed_jitter(rng);
        a.phase = up(rng);
        if (route == 0) {
            a.x = ux(rng);
            a.y = 5.10f + lane_noise(rng);
            a.lane = a.y;
        } else if (route == 1) {
            a.x = 8.72f + lane_noise(rng);
            a.y = uy(rng);
            a.lane = a.x;
        } else if (route == 2) {
            a.x = ux(rng);
            a.y = 5.92f + lane_noise(rng);
            a.lane = a.y;
        } else {
            a.x = 7.88f + lane_noise(rng);
            a.y = uy(rng);
            a.lane = a.x;
        }
        route_velocity(route, a.speed, a.phase, a.x, a.y, a.vx, a.vy);
        agents[i] = a;
    }
    return agents;
}

static float seed_pair_risk_host(const Agent& a, const Agent& b) {
    return seed_pair_risk(a, b);
}

static float interaction_weight_host(const Agent& a, const Agent& b) {
    return interaction_weight(a, b);
}

static Metrics summarize(const std::vector<float>& risk) {
    Metrics m{};
    for (float r : risk) {
        m.max_risk = std::max(m.max_risk, r);
        m.mean_risk += r;
        if (r > 0.55f) m.hot_agents++;
    }
    m.mean_risk /= std::max(1, (int)risk.size());
    return m;
}

static double cpu_propagation_ms(const std::vector<Agent>& agents) {
    std::vector<float> base(N_AGENTS, 0.0f);
    std::vector<float> risk_a(N_AGENTS, 0.0f);
    std::vector<float> risk_b(N_AGENTS, 0.0f);

    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_AGENTS; i++) {
        float r = 0.0f;
        for (int j = 0; j < N_AGENTS; j++) {
            if (i == j) continue;
            r = std::max(r, seed_pair_risk_host(agents[i], agents[j]));
        }
        base[i] = r;
        risk_a[i] = r;
    }

    std::vector<float>* in = &risk_a;
    std::vector<float>* out = &risk_b;
    for (int iter = 0; iter < PROP_ITERS; iter++) {
        for (int i = 0; i < N_AGENTS; i++) {
            float weighted = 0.0f;
            float wsum = 0.0f;
            float max_msg = 0.0f;
            for (int j = 0; j < N_AGENTS; j++) {
                if (i == j) continue;
                float w = interaction_weight_host(agents[i], agents[j]);
                if (w <= 0.015f) continue;
                float msg = w * (*in)[j];
                weighted += msg;
                wsum += w;
                max_msg = std::max(max_msg, msg);
            }
            float mean_msg = wsum > 1.0e-6f ? weighted / wsum : 0.0f;
            float propagated = RISK_DECAY * (0.58f * max_msg + 0.42f * mean_msg);
            (*out)[i] = clampf(std::max(base[i], propagated), 0.0f, 1.0f);
        }
        std::swap(in, out);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

static cv::Point to_px(float x, float y) {
    int px = static_cast<int>(x / WORLD_W * PANEL_W);
    int py = static_cast<int>((1.0f - y / WORLD_H) * PANEL_H);
    return cv::Point(px, py);
}

static cv::Scalar risk_color(float r, int kind) {
    r = clampf(r, 0.0f, 1.0f);
    if (kind == 1 && r < 0.35f) return cv::Scalar(245, 210, 80);
    int blue = static_cast<int>(clampf(160.0f * (1.0f - r), 25.0f, 170.0f));
    int green = static_cast<int>(clampf(220.0f * (1.0f - 0.55f * r), 40.0f, 230.0f));
    int red = static_cast<int>(clampf(60.0f + 205.0f * r, 60.0f, 255.0f));
    return cv::Scalar(blue, green, red);
}

static void draw_roads(cv::Mat& img) {
    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, PANEL_H), cv::Scalar(22, 23, 26), cv::FILLED);

    int y0 = to_px(0.0f, 6.35f).y;
    int y1 = to_px(0.0f, 4.65f).y;
    int x0 = to_px(7.05f, 0.0f).x;
    int x1 = to_px(9.45f, 0.0f).x;
    cv::rectangle(img, cv::Rect(0, y0, PANEL_W, y1 - y0), cv::Scalar(43, 45, 49), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0, 0, x1 - x0, PANEL_H), cv::Scalar(43, 45, 49), cv::FILLED);

    cv::line(img, to_px(0.0f, 5.50f), to_px(WORLD_W, 5.50f),
             cv::Scalar(90, 90, 95), 1, cv::LINE_AA);
    cv::line(img, to_px(8.25f, 0.0f), to_px(8.25f, WORLD_H),
             cv::Scalar(90, 90, 95), 1, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(x0, y0, x1 - x0, y1 - y0),
                  cv::Scalar(54, 56, 61), 1, cv::LINE_AA);
}

static cv::Mat draw_frame(const std::vector<Agent>& agents,
                          const std::vector<float>& risk,
                          float gpu_ms,
                          double cpu_ms,
                          int frame) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3);
    draw_roads(img);

    std::vector<int> order(N_AGENTS);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return risk[a] > risk[b]; });

    int edge_sources = 0;
    for (int idx : order) {
        if (risk[idx] < 0.62f || edge_sources >= 36) break;
        int drawn = 0;
        for (int j = 0; j < N_AGENTS && drawn < 3; j++) {
            if (j == idx) continue;
            float w = interaction_weight_host(agents[idx], agents[j]);
            if (w < 0.42f || risk[j] < 0.35f) continue;
            cv::line(img, to_px(agents[idx].x, agents[idx].y),
                     to_px(agents[j].x, agents[j].y),
                     cv::Scalar(40, 120, 255), 1, cv::LINE_AA);
            drawn++;
        }
        edge_sources++;
    }

    for (int i = 0; i < N_AGENTS; i++) {
        const Agent& a = agents[i];
        int radius = a.kind == 1 ? 3 : 2;
        cv::circle(img, to_px(a.x, a.y), radius, risk_color(risk[i], a.kind), cv::FILLED, cv::LINE_AA);
    }

    for (int k = 0; k < 16 && k < N_AGENTS; k++) {
        int i = order[k];
        if (risk[i] < 0.45f) continue;
        cv::circle(img, to_px(agents[i].x, agents[i].y),
                   8, cv::Scalar(60, 80, 255), 1, cv::LINE_AA);
    }

    Metrics m = summarize(risk);
    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, 36), cv::Scalar(5, 7, 10), cv::FILLED);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "GPU interaction-graph risk  agents=%d  message passes=%d  gpu=%.2f ms  cpu=%.1f ms",
                  N_AGENTS, PROP_ITERS, gpu_ms, cpu_ms);
    cv::putText(img, buf, cv::Point(12, 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);

    std::snprintf(buf, sizeof(buf),
                  "frame %02d  hot agents=%d  mean risk=%.3f  max risk=%.3f",
                  frame, m.hot_agents, m.mean_risk, m.max_risk);
    cv::putText(img, buf, cv::Point(12, PANEL_H - 16),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(220, 235, 245), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<Agent> h_agents = make_agents();
    std::vector<float> h_risk(N_AGENTS);

    Agent* d_agents = nullptr;
    float* d_base = nullptr;
    float* d_risk_a = nullptr;
    float* d_risk_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_agents, N_AGENTS * sizeof(Agent)));
    CUDA_CHECK(cudaMalloc(&d_base, N_AGENTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_risk_a, N_AGENTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_risk_b, N_AGENTS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_agents, h_agents.data(), N_AGENTS * sizeof(Agent),
                          cudaMemcpyHostToDevice));

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_interaction_graph_risk.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_interaction_graph_risk.avi\n");
        return 1;
    }

    int threads = 128;
    int blocks = (N_AGENTS + threads - 1) / threads;
    double total_gpu_ms = 0.0;
    int measured = 0;
    double cpu_ms = 0.0;
    float last_gpu_ms = 0.0f;

    for (int frame = 0; frame < N_FRAMES; frame++) {
        update_agents_kernel<<<blocks, threads>>>(d_agents, frame);

        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0));
        seed_risk_kernel<<<blocks, threads>>>(d_agents, d_base, d_risk_a);
        float* in = d_risk_a;
        float* out = d_risk_b;
        for (int iter = 0; iter < PROP_ITERS; iter++) {
            propagate_risk_kernel<<<blocks, threads>>>(d_agents, d_base, in, out);
            std::swap(in, out);
        }
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaEventElapsedTime(&last_gpu_ms, ev0, ev1));
        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));
        CUDA_CHECK(cudaGetLastError());

        if (frame >= 5) {
            total_gpu_ms += last_gpu_ms;
            measured++;
        }

        CUDA_CHECK(cudaMemcpy(h_agents.data(), d_agents, N_AGENTS * sizeof(Agent),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_risk.data(), in, N_AGENTS * sizeof(float),
                              cudaMemcpyDeviceToHost));

        if (frame == 8) {
            cpu_ms = cpu_propagation_ms(h_agents);
        }

        if (frame % 2 == 0 || frame == N_FRAMES - 1) {
            video.write(draw_frame(h_agents, h_risk, last_gpu_ms, cpu_ms, frame));
        }
        if (frame % 24 == 0) {
            Metrics m = summarize(h_risk);
            std::printf("  frame %3d  gpu %.3f ms  hot %d  mean %.3f  max %.3f\n",
                        frame, last_gpu_ms, m.hot_agents, m.mean_risk, m.max_risk);
        }
    }

    for (int i = 0; i < N_HOLD_FRAMES; i++) {
        video.write(draw_frame(h_agents, h_risk, last_gpu_ms, cpu_ms, N_FRAMES));
    }
    video.release();

    double avg_gpu = measured > 0 ? total_gpu_ms / measured : 0.0;
    double speedup = cpu_ms > 0.0 ? cpu_ms / avg_gpu : 0.0;
    std::printf("CPU propagation: %.3f ms\n", cpu_ms);
    std::printf("Avg GPU propagation: %.3f ms (%d agents, %d message passes, %.1fx vs CPU)\n",
                avg_gpu, N_AGENTS, PROP_ITERS, speedup);

    avi_to_gif("gif/gpu_interaction_graph_risk.avi",
               "gif/gpu_interaction_graph_risk.gif", 8, 560);
    std::printf("GIF saved to gif/gpu_interaction_graph_risk.gif\n");

    CUDA_CHECK(cudaFree(d_agents));
    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_risk_a));
    CUDA_CHECK(cudaFree(d_risk_b));
    return 0;
}
