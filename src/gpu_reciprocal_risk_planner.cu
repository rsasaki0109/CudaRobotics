// gpu_reciprocal_risk_planner.cu
//
// GPU reciprocal collision-risk planner.
//
// This turns interaction-graph risk into control.  Each agent evaluates a
// small action set (speed choice x lateral bias) against all other agents
// under a short constant-velocity prediction.  The collision cost is weighted
// by time-to-collision and by reciprocal responsibility, so robots yield more
// to pedestrians and same-lane followers do not dominate intersection risk.
//
// Output: gif/gpu_reciprocal_risk_planner.gif

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

constexpr int N_AGENTS = 1024;
constexpr int N_ACTIONS = 9;
constexpr int HORIZON = 16;
constexpr int N_FRAMES = 104;
constexpr int HOLD_FRAMES = 18;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int VIDEO_FPS = 12;

constexpr float WORLD_W = 36.0f;
constexpr float WORLD_H = 22.0f;
constexpr float CENTER_X = 0.5f * WORLD_W;
constexpr float CENTER_Y = 0.5f * WORLD_H;
constexpr float H_LANE_0 = CENTER_Y - 0.42f;
constexpr float H_LANE_1 = CENTER_Y + 0.42f;
constexpr float V_LANE_0 = CENTER_X + 0.42f;
constexpr float V_LANE_1 = CENTER_X - 0.42f;
constexpr float ROAD_HALF_H = 1.22f;
constexpr float ROAD_HALF_W = 1.36f;
constexpr float DT = 0.12f;
constexpr float ROBOT_R = 0.16f;
constexpr float PED_R = 0.12f;
constexpr float SAFE_EXTRA = 0.30f;
constexpr float LATERAL_SPEED = 0.72f;
constexpr float MAX_SPEED = 1.75f;
constexpr float RISK_DISPLAY_SCALE = 20.0f;
constexpr float PI_F = 3.14159265358979323846f;

struct Agent {
    float x;
    float y;
    float vx;
    float vy;
    float pref_speed;
    float lane;
    float phase;
    int route;
    int kind;  // 0 pedestrian, 1 robot
};

struct ActionChoice {
    int action;
    float vx;
    float vy;
    float risk;
};

struct Metrics {
    int close_pairs = 0;
    int hot_agents = 0;
    float mean_risk = 0.0f;
    float max_risk = 0.0f;
};

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline float sqr(float x) {
    return x * x;
}

__host__ __device__ static inline void route_basis(int route,
                                                   float& dx,
                                                   float& dy,
                                                   float& lx,
                                                   float& ly) {
    if (route == 0) {
        dx = 1.0f; dy = 0.0f; lx = 0.0f; ly = 1.0f;
    } else if (route == 1) {
        dx = 0.0f; dy = 1.0f; lx = -1.0f; ly = 0.0f;
    } else if (route == 2) {
        dx = -1.0f; dy = 0.0f; lx = 0.0f; ly = -1.0f;
    } else {
        dx = 0.0f; dy = -1.0f; lx = 1.0f; ly = 0.0f;
    }
}

__host__ __device__ static inline float lane_position(const Agent& a, float x, float y) {
    return (a.route & 1) == 0 ? y : x;
}

__host__ __device__ static inline float agent_radius(const Agent& a) {
    return a.kind == 1 ? ROBOT_R : PED_R;
}

__host__ __device__ static inline float responsibility(const Agent& ego, const Agent& other) {
    if (ego.kind == 1 && other.kind == 0) return 1.30f;
    if (ego.kind == 0 && other.kind == 1) return 0.70f;
    return 1.0f;
}

__host__ __device__ static inline float same_axis_factor(const Agent& a, const Agent& b) {
    bool same_axis = (a.route & 1) == (b.route & 1);
    return same_axis ? 0.12f : 1.0f;
}

__host__ __device__ static inline float speed_scale_for_action(int action) {
    int s = action / 3;
    if (s == 0) return 0.48f;
    if (s == 1) return 0.78f;
    return 1.06f;
}

__host__ __device__ static inline float lateral_for_action(int action) {
    int l = action % 3;
    return (float)(l - 1) * LATERAL_SPEED;
}

__host__ __device__ static inline void candidate_velocity(const Agent& a,
                                                          int action,
                                                          float& vx,
                                                          float& vy) {
    float dx, dy, lx, ly;
    route_basis(a.route, dx, dy, lx, ly);
    float lane_err = lane_position(a, a.x, a.y) - a.lane;
    float lane_restore = clampf(-0.38f * lane_err, -0.34f, 0.34f);
    float speed = clampf(a.pref_speed * speed_scale_for_action(action), 0.20f, MAX_SPEED);
    float lateral = lateral_for_action(action) + lane_restore;
    vx = dx * speed + lx * lateral;
    vy = dy * speed + ly * lateral;
}

__host__ __device__ static inline float pair_risk_cost(const Agent& ego,
                                                       const Agent& other,
                                                       float ego_vx,
                                                       float ego_vy) {
    float total = 0.0f;
    float rr = agent_radius(ego) + agent_radius(other) + SAFE_EXTRA;
    float rr2 = rr * rr;
    float axis = same_axis_factor(ego, other);
    float resp = responsibility(ego, other);

    for (int h = 1; h <= HORIZON; h++) {
        float t = DT * (float)h;
        float ex = ego.x + ego_vx * t;
        float ey = ego.y + ego_vy * t;
        float ox = other.x + other.vx * t;
        float oy = other.y + other.vy * t;
        float dx = ox - ex;
        float dy = oy - ey;
        float d2 = dx * dx + dy * dy;
        if (d2 > 3.4f * 3.4f) continue;

        float rvx = other.vx - ego_vx;
        float rvy = other.vy - ego_vy;
        float d = sqrtf(d2) + 1.0e-5f;
        float closing = -(dx * rvx + dy * rvy) / d;
        float ttc_w = closing > 0.03f ? clampf((2.6f - d / closing) / 2.6f, 0.0f, 1.0f) : 0.0f;
        float overlap = d2 < rr2 ? (rr2 - d2) / rr2 : 0.0f;
        float near = d2 < 1.10f * 1.10f ? (1.10f * 1.10f - d2) / (1.10f * 1.10f) : 0.0f;
        float temporal = 1.0f / (0.35f + (float)h);
        total += resp * axis * temporal * (7.0f * overlap + 2.6f * ttc_w + 0.55f * near);
    }
    return total;
}

__host__ __device__ static inline float action_cost(const Agent* agents, int ego_idx, int action) {
    Agent ego = agents[ego_idx];
    float vx, vy;
    candidate_velocity(ego, action, vx, vy);

    float collision = 0.0f;
    for (int j = 0; j < N_AGENTS; j++) {
        if (j == ego_idx) continue;
        collision += pair_risk_cost(ego, agents[j], vx, vy);
    }

    float dx, dy, lx, ly;
    route_basis(ego.route, dx, dy, lx, ly);
    float speed = vx * dx + vy * dy;
    float lateral = vx * lx + vy * ly;
    float next_lane = lane_position(ego, ego.x + vx * DT, ego.y + vy * DT);
    float lane_err = next_lane - ego.lane;
    float comfort = 0.32f * fabsf(lateral) + 0.28f * sqr(lane_err);
    float progress = 0.26f * sqr(ego.pref_speed - speed);
    return collision + comfort + progress;
}

__global__ void evaluate_actions_kernel(const Agent* __restrict__ agents,
                                        float* __restrict__ costs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N_AGENTS * N_ACTIONS;
    if (idx >= total) return;
    int agent = idx / N_ACTIONS;
    int action = idx - agent * N_ACTIONS;
    costs[idx] = action_cost(agents, agent, action);
}

__global__ void choose_actions_kernel(const Agent* __restrict__ agents,
                                      const float* __restrict__ costs,
                                      ActionChoice* __restrict__ choices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_AGENTS) return;

    int best_action = 0;
    float best_cost = costs[i * N_ACTIONS];
    for (int a = 1; a < N_ACTIONS; a++) {
        float c = costs[i * N_ACTIONS + a];
        if (c < best_cost) {
            best_cost = c;
            best_action = a;
        }
    }

    float vx, vy;
    candidate_velocity(agents[i], best_action, vx, vy);
    choices[i] = {best_action, vx, vy, clampf(best_cost / RISK_DISPLAY_SCALE, 0.0f, 1.0f)};
}

__global__ void apply_actions_kernel(Agent* agents,
                                     const ActionChoice* choices,
                                     int frame) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_AGENTS) return;
    Agent a = agents[i];
    ActionChoice c = choices[i];

    float dx, dy, lx, ly;
    route_basis(a.route, dx, dy, lx, ly);
    float wiggle = 0.03f * sinf(a.phase + 0.07f * (float)frame);
    a.vx = c.vx + lx * wiggle;
    a.vy = c.vy + ly * wiggle;
    a.x += a.vx * DT;
    a.y += a.vy * DT;

    if (a.route == 0 && a.x > WORLD_W + 0.5f) a.x -= WORLD_W + 1.0f;
    if (a.route == 2 && a.x < -0.5f) a.x += WORLD_W + 1.0f;
    if (a.route == 1 && a.y > WORLD_H + 0.5f) a.y -= WORLD_H + 1.0f;
    if (a.route == 3 && a.y < -0.5f) a.y += WORLD_H + 1.0f;
    a.x = clampf(a.x, -0.55f, WORLD_W + 0.55f);
    a.y = clampf(a.y, -0.55f, WORLD_H + 0.55f);
    agents[i] = a;
}

static void route_velocity_host(Agent& a) {
    float dx, dy, lx, ly;
    route_basis(a.route, dx, dy, lx, ly);
    float speed = a.pref_speed;
    a.vx = dx * speed;
    a.vy = dy * speed;
}

static std::vector<Agent> make_agents() {
    std::vector<Agent> agents(N_AGENTS);
    std::mt19937 rng(25052026);
    std::uniform_real_distribution<float> ux(0.0f, WORLD_W);
    std::uniform_real_distribution<float> uy(0.0f, WORLD_H);
    std::uniform_real_distribution<float> up(0.0f, 2.0f * PI_F);
    std::normal_distribution<float> lane_noise(0.0f, 0.42f);
    std::uniform_real_distribution<float> speed_jitter(-0.08f, 0.16f);

    for (int i = 0; i < N_AGENTS; i++) {
        Agent a{};
        a.route = i % 4;
        a.kind = (i % 13 == 0) ? 1 : 0;
        a.pref_speed = (a.kind == 1 ? 1.35f : 1.02f) + speed_jitter(rng);
        a.phase = up(rng);
        if (a.route == 0) {
            a.x = ux(rng);
            a.y = H_LANE_0 + lane_noise(rng);
            a.lane = H_LANE_0;
        } else if (a.route == 1) {
            a.x = V_LANE_0 + lane_noise(rng);
            a.y = uy(rng);
            a.lane = V_LANE_0;
        } else if (a.route == 2) {
            a.x = ux(rng);
            a.y = H_LANE_1 + lane_noise(rng);
            a.lane = H_LANE_1;
        } else {
            a.x = V_LANE_1 + lane_noise(rng);
            a.y = uy(rng);
            a.lane = V_LANE_1;
        }
        route_velocity_host(a);
        agents[i] = a;
    }
    return agents;
}

static double cpu_plan_ms(const std::vector<Agent>& agents) {
    std::vector<ActionChoice> choices(N_AGENTS);
    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_AGENTS; i++) {
        int best_action = 0;
        float best_cost = action_cost(agents.data(), i, 0);
        for (int a = 1; a < N_ACTIONS; a++) {
            float c = action_cost(agents.data(), i, a);
            if (c < best_cost) {
                best_cost = c;
                best_action = a;
            }
        }
        float vx, vy;
        candidate_velocity(agents[i], best_action, vx, vy);
        choices[i] = {best_action, vx, vy, clampf(best_cost / RISK_DISPLAY_SCALE, 0.0f, 1.0f)};
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

static Metrics summarize(const std::vector<Agent>& agents,
                         const std::vector<ActionChoice>& choices) {
    Metrics m{};
    for (int i = 0; i < N_AGENTS; i++) {
        float r = choices[i].risk;
        m.max_risk = std::max(m.max_risk, r);
        m.mean_risk += r;
        if (r > 0.45f) m.hot_agents++;
        for (int j = i + 1; j < N_AGENTS; j++) {
            float rr = agent_radius(agents[i]) + agent_radius(agents[j]) + 0.12f;
            float dx = agents[i].x - agents[j].x;
            float dy = agents[i].y - agents[j].y;
            if (dx * dx + dy * dy < rr * rr) m.close_pairs++;
        }
    }
    m.mean_risk /= (float)N_AGENTS;
    return m;
}

static cv::Point to_px(float x, float y) {
    int px = static_cast<int>(x / WORLD_W * PANEL_W);
    int py = static_cast<int>((1.0f - y / WORLD_H) * PANEL_H);
    return cv::Point(px, py);
}

static cv::Scalar risk_color(float r, int kind) {
    r = clampf(r, 0.0f, 1.0f);
    if (kind == 1 && r < 0.30f) return cv::Scalar(245, 205, 80);
    int blue = static_cast<int>(clampf(165.0f * (1.0f - r), 30.0f, 170.0f));
    int green = static_cast<int>(clampf(230.0f * (1.0f - 0.58f * r), 45.0f, 230.0f));
    int red = static_cast<int>(clampf(70.0f + 190.0f * r, 70.0f, 255.0f));
    return cv::Scalar(blue, green, red);
}

static void draw_roads(cv::Mat& img) {
    img = cv::Mat(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(22, 23, 26));
    int y0 = to_px(0.0f, CENTER_Y + ROAD_HALF_H).y;
    int y1 = to_px(0.0f, CENTER_Y - ROAD_HALF_H).y;
    int x0 = to_px(CENTER_X - ROAD_HALF_W, 0.0f).x;
    int x1 = to_px(CENTER_X + ROAD_HALF_W, 0.0f).x;
    cv::rectangle(img, cv::Rect(0, y0, PANEL_W, y1 - y0), cv::Scalar(43, 45, 49), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0, 0, x1 - x0, PANEL_H), cv::Scalar(43, 45, 49), cv::FILLED);
    cv::line(img, to_px(0.0f, CENTER_Y), to_px(WORLD_W, CENTER_Y),
             cv::Scalar(92, 92, 96), 1, cv::LINE_AA);
    cv::line(img, to_px(CENTER_X, 0.0f), to_px(CENTER_X, WORLD_H),
             cv::Scalar(92, 92, 96), 1, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(x0, y0, x1 - x0, y1 - y0),
                  cv::Scalar(58, 60, 64), 1, cv::LINE_AA);
}

static cv::Mat draw_frame(const std::vector<Agent>& agents,
                          const std::vector<ActionChoice>& choices,
                          float gpu_ms,
                          double cpu_ms,
                          int frame) {
    cv::Mat img;
    draw_roads(img);

    std::vector<int> order(N_AGENTS);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return choices[a].risk > choices[b].risk; });

    for (int k = 0; k < 42 && k < N_AGENTS; k++) {
        int i = order[k];
        if (choices[i].risk < 0.42f) continue;
        cv::Point p0 = to_px(agents[i].x, agents[i].y);
        cv::Point p1 = to_px(agents[i].x + 0.55f * choices[i].vx,
                             agents[i].y + 0.55f * choices[i].vy);
        cv::arrowedLine(img, p0, p1, cv::Scalar(70, 210, 255), 1, cv::LINE_AA, 0, 0.25);
    }

    for (int i = 0; i < N_AGENTS; i++) {
        int radius = agents[i].kind == 1 ? 4 : 2;
        cv::circle(img, to_px(agents[i].x, agents[i].y),
                   radius, risk_color(choices[i].risk, agents[i].kind), cv::FILLED, cv::LINE_AA);
    }

    for (int k = 0; k < 12 && k < N_AGENTS; k++) {
        int i = order[k];
        if (choices[i].risk < 0.40f) continue;
        cv::circle(img, to_px(agents[i].x, agents[i].y),
                   9, cv::Scalar(60, 95, 255), 1, cv::LINE_AA);
    }

    Metrics m = summarize(agents, choices);
    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, 36), cv::Scalar(5, 7, 10), cv::FILLED);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "GPU reciprocal risk planner  agents=%d  actions=%d  horizon=%d  gpu=%.2f ms  cpu=%.1f ms",
                  N_AGENTS, N_ACTIONS, HORIZON, gpu_ms, cpu_ms);
    cv::putText(img, buf, cv::Point(12, 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.53, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);

    std::snprintf(buf, sizeof(buf),
                  "frame %02d  hot agents=%d  close pairs=%d  mean risk=%.3f  max risk=%.3f",
                  frame, m.hot_agents, m.close_pairs, m.mean_risk, m.max_risk);
    cv::putText(img, buf, cv::Point(12, PANEL_H - 16),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(225, 238, 245), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<Agent> h_agents = make_agents();
    std::vector<ActionChoice> h_choices(N_AGENTS);

    Agent* d_agents = nullptr;
    float* d_costs = nullptr;
    ActionChoice* d_choices = nullptr;
    CUDA_CHECK(cudaMalloc(&d_agents, N_AGENTS * sizeof(Agent)));
    CUDA_CHECK(cudaMalloc(&d_costs, N_AGENTS * N_ACTIONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_choices, N_AGENTS * sizeof(ActionChoice)));
    CUDA_CHECK(cudaMemcpy(d_agents, h_agents.data(), N_AGENTS * sizeof(Agent),
                          cudaMemcpyHostToDevice));

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_reciprocal_risk_planner.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_reciprocal_risk_planner.avi\n");
        return 1;
    }

    int eval_threads = 128;
    int eval_blocks = (N_AGENTS * N_ACTIONS + eval_threads - 1) / eval_threads;
    int agent_threads = 128;
    int agent_blocks = (N_AGENTS + agent_threads - 1) / agent_threads;
    double total_gpu_ms = 0.0;
    int measured = 0;
    float last_gpu_ms = 0.0f;
    double cpu_ms = 0.0;

    for (int frame = 0; frame < N_FRAMES; frame++) {
        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0));
        evaluate_actions_kernel<<<eval_blocks, eval_threads>>>(d_agents, d_costs);
        choose_actions_kernel<<<agent_blocks, agent_threads>>>(d_agents, d_costs, d_choices);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaEventElapsedTime(&last_gpu_ms, ev0, ev1));
        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));
        CUDA_CHECK(cudaGetLastError());

        apply_actions_kernel<<<agent_blocks, agent_threads>>>(d_agents, d_choices, frame);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (frame >= 5) {
            total_gpu_ms += last_gpu_ms;
            measured++;
        }

        CUDA_CHECK(cudaMemcpy(h_agents.data(), d_agents, N_AGENTS * sizeof(Agent),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_choices.data(), d_choices, N_AGENTS * sizeof(ActionChoice),
                              cudaMemcpyDeviceToHost));

        if (frame == 8) {
            cpu_ms = cpu_plan_ms(h_agents);
        }
        if ((frame >= 8 && frame % 2 == 0) || frame == N_FRAMES - 1) {
            video.write(draw_frame(h_agents, h_choices, last_gpu_ms, cpu_ms, frame));
        }
        if (frame % 26 == 0) {
            Metrics m = summarize(h_agents, h_choices);
            std::printf("  frame %3d  gpu %.3f ms  hot %d  close %d  mean %.3f\n",
                        frame, last_gpu_ms, m.hot_agents, m.close_pairs, m.mean_risk);
        }
    }

    for (int i = 0; i < HOLD_FRAMES; i++) {
        video.write(draw_frame(h_agents, h_choices, last_gpu_ms, cpu_ms, N_FRAMES));
    }
    video.release();

    double avg_gpu = measured > 0 ? total_gpu_ms / measured : 0.0;
    double speedup = cpu_ms > 0.0 ? cpu_ms / avg_gpu : 0.0;
    std::printf("CPU planner: %.3f ms\n", cpu_ms);
    std::printf("Avg GPU planner: %.3f ms (%d agents x %d actions x H=%d, %.1fx vs CPU)\n",
                avg_gpu, N_AGENTS, N_ACTIONS, HORIZON, speedup);

    avi_to_gif("gif/gpu_reciprocal_risk_planner.avi",
               "gif/gpu_reciprocal_risk_planner.gif", 8, 560);
    std::printf("GIF saved to gif/gpu_reciprocal_risk_planner.gif\n");

    CUDA_CHECK(cudaFree(d_agents));
    CUDA_CHECK(cudaFree(d_costs));
    CUDA_CHECK(cudaFree(d_choices));
    return 0;
}
