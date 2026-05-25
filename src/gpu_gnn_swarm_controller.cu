// gpu_gnn_swarm_controller.cu
//
// GPU GNN-style swarm controller.
//
// This demo treats a dense crowd of robots and pedestrians as an implicit
// radius graph.  Each node runs a small fixed-weight message-passing policy:
// neighbors contribute separation, flow, alignment, and collision-risk
// messages, then a goal-conditioned controller turns the final node embedding
// into velocity commands.
//
// Output: gif/gpu_gnn_swarm_controller.gif

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
constexpr int MESSAGE_PASSES = 3;
constexpr int N_FRAMES = 122;
constexpr int HOLD_FRAMES = 16;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int VIDEO_FPS = 12;
constexpr int THREADS = 128;

constexpr float WORLD_W = 36.0f;
constexpr float WORLD_H = 22.0f;
constexpr float CENTER_X = 0.5f * WORLD_W;
constexpr float CENTER_Y = 0.5f * WORLD_H;
constexpr float H_LANE_0 = CENTER_Y - 0.46f;
constexpr float H_LANE_1 = CENTER_Y + 0.46f;
constexpr float V_LANE_0 = CENTER_X + 0.46f;
constexpr float V_LANE_1 = CENTER_X - 0.46f;
constexpr float ROAD_HALF_H = 1.35f;
constexpr float ROAD_HALF_W = 1.55f;

constexpr float DT = 0.095f;
constexpr float GRAPH_R = 1.55f;
constexpr float GRAPH_R2 = GRAPH_R * GRAPH_R;
constexpr float INV_TWO_SIGMA2 = 1.0f / (2.0f * 0.82f * 0.82f);
constexpr float MAX_SPEED = 1.85f;
constexpr float ROBOT_R = 0.16f;
constexpr float PED_R = 0.11f;
constexpr float PI_F = 3.14159265358979323846f;

struct Agent {
    float x;
    float y;
    float vx;
    float vy;
    float gx;
    float gy;
    float lane;
    float phase;
    int route;
    int kind;  // 0 pedestrian, 1 robot
};

struct NodeState {
    float density;
    float sep_x;
    float sep_y;
    float align_x;
    float align_y;
    float flow_x;
    float flow_y;
    float urgency;
    float risk;
};

struct Action {
    float vx;
    float vy;
    float risk;
    float density;
};

struct PairTerms {
    float w;
    float sep_x;
    float sep_y;
    float align_x;
    float align_y;
    float risk;
};

struct Metrics {
    int hot_agents = 0;
    int close_pairs = 0;
    float mean_risk = 0.0f;
    float max_risk = 0.0f;
    float mean_density = 0.0f;
    float progress = 0.0f;
};

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
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

__host__ __device__ static inline float agent_radius(const Agent& a) {
    return a.kind == 1 ? ROBOT_R : PED_R;
}

__host__ __device__ static inline float pref_speed(const Agent& a) {
    return a.kind == 1 ? 1.46f : 1.08f;
}

__host__ __device__ static inline float lane_position(const Agent& a) {
    return (a.route & 1) == 0 ? a.y : a.x;
}

__host__ __device__ static inline void goal_unit(const Agent& a, float& ux, float& uy, float& dist) {
    float dx = a.gx - a.x;
    float dy = a.gy - a.y;
    dist = sqrtf(dx * dx + dy * dy) + 1.0e-5f;
    ux = dx / dist;
    uy = dy / dist;
}

__host__ __device__ static inline PairTerms pair_terms(const Agent& a, const Agent& b) {
    PairTerms t;
    t.w = 0.0f;
    t.sep_x = 0.0f;
    t.sep_y = 0.0f;
    t.align_x = 0.0f;
    t.align_y = 0.0f;
    t.risk = 0.0f;

    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float d2 = dx * dx + dy * dy;
    if (d2 > GRAPH_R2 || d2 < 1.0e-8f) return t;

    float d = sqrtf(d2);
    float w = expf(-d2 * INV_TWO_SIGMA2);
    float rr = agent_radius(a) + agent_radius(b) + 0.20f;
    float close = clampf((rr + 0.52f - d) / 0.52f, 0.0f, 1.0f);
    float rvx = b.vx - a.vx;
    float rvy = b.vy - a.vy;
    float closing = -(dx * rvx + dy * rvy) / (d + 1.0e-5f);
    float ttc = closing > 0.04f ? d / closing : 1.0e6f;
    float ttc_w = ttc < 2.2f ? (2.2f - ttc) / 2.2f : 0.0f;
    bool crossing = (a.route & 1) != (b.route & 1);
    float cross_boost = crossing ? 1.0f : 0.08f;
    float robot_boost = (a.kind == 1 || b.kind == 1) ? 1.14f : 1.0f;

    t.w = w;
    t.sep_x = -dx / (d + 0.08f) * close * robot_boost;
    t.sep_y = -dy / (d + 0.08f) * close * robot_boost;
    t.align_x = (b.vx - a.vx) * w;
    t.align_y = (b.vy - a.vy) * w;
    t.risk = clampf(cross_boost * (0.40f * close + 0.76f * ttc_w), 0.0f, 1.0f);
    return t;
}

__host__ __device__ static inline void initialize_node(const Agent& a, NodeState& n) {
    float ux, uy, dist;
    goal_unit(a, ux, uy, dist);
    n.density = 0.0f;
    n.sep_x = 0.0f;
    n.sep_y = 0.0f;
    n.align_x = 0.0f;
    n.align_y = 0.0f;
    n.flow_x = ux;
    n.flow_y = uy;
    n.urgency = clampf(dist / 8.5f, 0.0f, 1.0f);
    n.risk = 0.0f;
}

__host__ __device__ static inline Action policy_from_node(const Agent& a, const NodeState& n) {
    float gx, gy, dist;
    goal_unit(a, gx, gy, dist);
    float dx, dy, lx, ly;
    route_basis(a.route, dx, dy, lx, ly);
    float lane_err = lane_position(a) - a.lane;
    float lane_restore = clampf(-0.64f * lane_err, -0.48f, 0.48f);

    float flow_norm = rsqrtf(fmaxf(n.flow_x * n.flow_x + n.flow_y * n.flow_y, 1.0e-6f));
    float flow_x = n.flow_x * flow_norm;
    float flow_y = n.flow_y * flow_norm;
    float risk_brake = 1.0f - 0.42f * clampf(n.risk, 0.0f, 1.0f);
    float speed = clampf(pref_speed(a) * risk_brake + 0.18f * n.urgency, 0.20f, MAX_SPEED);

    float vx = 1.10f * gx + 0.32f * flow_x + 0.74f * n.sep_x + 0.14f * n.align_x
             + lx * lane_restore;
    float vy = 1.10f * gy + 0.32f * flow_y + 0.74f * n.sep_y + 0.14f * n.align_y
             + ly * lane_restore;
    float norm = rsqrtf(fmaxf(vx * vx + vy * vy, 1.0e-6f));

    Action out;
    out.vx = vx * norm * speed;
    out.vy = vy * norm * speed;
    out.risk = clampf(n.risk, 0.0f, 1.0f);
    out.density = clampf(n.density, 0.0f, 1.0f);
    return out;
}

__global__ void init_nodes_kernel(const Agent* __restrict__ agents,
                                  NodeState* __restrict__ nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_AGENTS) return;
    initialize_node(agents[i], nodes[i]);
}

__global__ void message_pass_kernel(const Agent* __restrict__ agents,
                                    const NodeState* __restrict__ in,
                                    NodeState* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_AGENTS) return;

    Agent a = agents[i];
    NodeState base = in[i];
    float density = 0.0f;
    float sep_x = 0.0f;
    float sep_y = 0.0f;
    float align_x = 0.0f;
    float align_y = 0.0f;
    float flow_x = 0.0f;
    float flow_y = 0.0f;
    float wsum = 0.0f;
    float risk = base.risk;

    for (int j = 0; j < N_AGENTS; j++) {
        if (j == i) continue;
        PairTerms t = pair_terms(a, agents[j]);
        if (t.w <= 0.0f) continue;
        density += t.w;
        sep_x += t.sep_x * t.w;
        sep_y += t.sep_y * t.w;
        align_x += t.align_x;
        align_y += t.align_y;
        flow_x += t.w * in[j].flow_x;
        flow_y += t.w * in[j].flow_y;
        wsum += t.w;
        risk = fmaxf(risk, t.risk);
    }

    float inv = wsum > 1.0e-5f ? 1.0f / wsum : 0.0f;
    NodeState next;
    next.density = clampf(0.025f * density, 0.0f, 1.0f);
    next.sep_x = tanhf(0.58f * base.sep_x + 0.52f * sep_x * inv);
    next.sep_y = tanhf(0.58f * base.sep_y + 0.52f * sep_y * inv);
    next.align_x = clampf(0.55f * base.align_x + 0.45f * align_x * inv, -1.0f, 1.0f);
    next.align_y = clampf(0.55f * base.align_y + 0.45f * align_y * inv, -1.0f, 1.0f);
    next.flow_x = 0.60f * base.flow_x + 0.40f * flow_x * inv;
    next.flow_y = 0.60f * base.flow_y + 0.40f * flow_y * inv;
    next.urgency = base.urgency;
    next.risk = clampf(0.60f * base.risk + 0.42f * risk + 0.10f * next.density, 0.0f, 1.0f);
    out[i] = next;
}

__global__ void policy_kernel(const Agent* __restrict__ agents,
                              const NodeState* __restrict__ nodes,
                              Action* __restrict__ actions) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_AGENTS) return;
    actions[i] = policy_from_node(agents[i], nodes[i]);
}

__global__ void step_agents_kernel(Agent* agents,
                                   const Action* actions,
                                   int frame) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_AGENTS) return;
    Agent a = agents[i];
    Action u = actions[i];
    float dx, dy, lx, ly;
    route_basis(a.route, dx, dy, lx, ly);
    float jitter = 0.025f * sinf(a.phase + 0.045f * (float)frame);

    a.vx = 0.66f * a.vx + 0.34f * u.vx + lx * jitter;
    a.vy = 0.66f * a.vy + 0.34f * u.vy + ly * jitter;
    float speed = sqrtf(a.vx * a.vx + a.vy * a.vy);
    if (speed > MAX_SPEED) {
        float scale = MAX_SPEED / (speed + 1.0e-5f);
        a.vx *= scale;
        a.vy *= scale;
    }
    a.x += a.vx * DT;
    a.y += a.vy * DT;

    float lane_wiggle = 0.22f * sinf(a.phase + 0.19f * (float)frame);
    if (a.route == 0 && a.x > WORLD_W + 0.6f) {
        a.x = -0.6f;
        a.y = a.lane + lane_wiggle;
    } else if (a.route == 2 && a.x < -0.6f) {
        a.x = WORLD_W + 0.6f;
        a.y = a.lane + lane_wiggle;
    } else if (a.route == 1 && a.y > WORLD_H + 0.6f) {
        a.y = -0.6f;
        a.x = a.lane + lane_wiggle;
    } else if (a.route == 3 && a.y < -0.6f) {
        a.y = WORLD_H + 0.6f;
        a.x = a.lane + lane_wiggle;
    }

    a.x = clampf(a.x, -0.75f, WORLD_W + 0.75f);
    a.y = clampf(a.y, -0.75f, WORLD_H + 0.75f);
    agents[i] = a;
}

static void set_goal(Agent& a) {
    if (a.route == 0) {
        a.gx = WORLD_W + 1.2f;
        a.gy = a.lane;
    } else if (a.route == 1) {
        a.gx = a.lane;
        a.gy = WORLD_H + 1.2f;
    } else if (a.route == 2) {
        a.gx = -1.2f;
        a.gy = a.lane;
    } else {
        a.gx = a.lane;
        a.gy = -1.2f;
    }
}

static void set_route_velocity(Agent& a) {
    float dx, dy, lx, ly;
    route_basis(a.route, dx, dy, lx, ly);
    a.vx = dx * pref_speed(a);
    a.vy = dy * pref_speed(a);
}

static std::vector<Agent> make_agents() {
    std::vector<Agent> agents(N_AGENTS);
    std::mt19937 rng(25052026);
    std::uniform_real_distribution<float> ux(0.0f, WORLD_W);
    std::uniform_real_distribution<float> uy(0.0f, WORLD_H);
    std::uniform_real_distribution<float> phase(0.0f, 2.0f * PI_F);
    std::normal_distribution<float> lane_noise(0.0f, 0.35f);

    for (int i = 0; i < N_AGENTS; i++) {
        Agent a{};
        a.route = i % 4;
        a.kind = (i % 11 == 0) ? 1 : 0;
        a.phase = phase(rng);
        if (a.route == 0) {
            a.x = ux(rng);
            a.lane = H_LANE_0;
            a.y = a.lane + lane_noise(rng);
        } else if (a.route == 1) {
            a.y = uy(rng);
            a.lane = V_LANE_0;
            a.x = a.lane + lane_noise(rng);
        } else if (a.route == 2) {
            a.x = ux(rng);
            a.lane = H_LANE_1;
            a.y = a.lane + lane_noise(rng);
        } else {
            a.y = uy(rng);
            a.lane = V_LANE_1;
            a.x = a.lane + lane_noise(rng);
        }
        set_goal(a);
        set_route_velocity(a);
        agents[i] = a;
    }
    return agents;
}

static void init_nodes_host(const std::vector<Agent>& agents, std::vector<NodeState>& nodes) {
    for (int i = 0; i < N_AGENTS; i++) {
        initialize_node(agents[i], nodes[i]);
    }
}

static void message_pass_host(const std::vector<Agent>& agents,
                              const std::vector<NodeState>& in,
                              std::vector<NodeState>& out) {
    for (int i = 0; i < N_AGENTS; i++) {
        const Agent& a = agents[i];
        NodeState base = in[i];
        float density = 0.0f;
        float sep_x = 0.0f;
        float sep_y = 0.0f;
        float align_x = 0.0f;
        float align_y = 0.0f;
        float flow_x = 0.0f;
        float flow_y = 0.0f;
        float wsum = 0.0f;
        float risk = base.risk;

        for (int j = 0; j < N_AGENTS; j++) {
            if (j == i) continue;
            PairTerms t = pair_terms(a, agents[j]);
            if (t.w <= 0.0f) continue;
            density += t.w;
            sep_x += t.sep_x * t.w;
            sep_y += t.sep_y * t.w;
            align_x += t.align_x;
            align_y += t.align_y;
            flow_x += t.w * in[j].flow_x;
            flow_y += t.w * in[j].flow_y;
            wsum += t.w;
            risk = std::max(risk, t.risk);
        }

        float inv = wsum > 1.0e-5f ? 1.0f / wsum : 0.0f;
        NodeState next;
        next.density = clampf(0.025f * density, 0.0f, 1.0f);
        next.sep_x = std::tanh(0.58f * base.sep_x + 0.52f * sep_x * inv);
        next.sep_y = std::tanh(0.58f * base.sep_y + 0.52f * sep_y * inv);
        next.align_x = clampf(0.55f * base.align_x + 0.45f * align_x * inv, -1.0f, 1.0f);
        next.align_y = clampf(0.55f * base.align_y + 0.45f * align_y * inv, -1.0f, 1.0f);
        next.flow_x = 0.60f * base.flow_x + 0.40f * flow_x * inv;
        next.flow_y = 0.60f * base.flow_y + 0.40f * flow_y * inv;
        next.urgency = base.urgency;
        next.risk = clampf(0.60f * base.risk + 0.42f * risk + 0.10f * next.density, 0.0f, 1.0f);
        out[i] = next;
    }
}

static void policy_host(const std::vector<Agent>& agents,
                        const std::vector<NodeState>& nodes,
                        std::vector<Action>& actions) {
    for (int i = 0; i < N_AGENTS; i++) {
        actions[i] = policy_from_node(agents[i], nodes[i]);
    }
}

static double cpu_controller_ms(const std::vector<Agent>& agents) {
    std::vector<NodeState> a(N_AGENTS);
    std::vector<NodeState> b(N_AGENTS);
    std::vector<Action> actions(N_AGENTS);
    auto begin = std::chrono::high_resolution_clock::now();
    init_nodes_host(agents, a);
    std::vector<NodeState>* in = &a;
    std::vector<NodeState>* out = &b;
    for (int pass = 0; pass < MESSAGE_PASSES; pass++) {
        message_pass_host(agents, *in, *out);
        std::swap(in, out);
    }
    policy_host(agents, *in, actions);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

static Metrics summarize(const std::vector<Agent>& agents,
                         const std::vector<NodeState>& nodes,
                         const std::vector<Action>& actions) {
    Metrics m;
    for (int i = 0; i < N_AGENTS; i++) {
        float r = actions[i].risk;
        m.mean_risk += r;
        m.max_risk = std::max(m.max_risk, r);
        m.mean_density += actions[i].density;
        if (r > 0.42f) m.hot_agents++;
        float dx, dy, lx, ly;
        route_basis(agents[i].route, dx, dy, lx, ly);
        m.progress += std::max(0.0f, agents[i].vx * dx + agents[i].vy * dy);
    }
    for (int i = 0; i < N_AGENTS; i++) {
        for (int j = i + 1; j < N_AGENTS; j++) {
            float rr = agent_radius(agents[i]) + agent_radius(agents[j]) + 0.10f;
            float dx = agents[i].x - agents[j].x;
            float dy = agents[i].y - agents[j].y;
            if (dx * dx + dy * dy < rr * rr) m.close_pairs++;
        }
    }
    m.mean_risk /= (float)N_AGENTS;
    m.mean_density /= (float)N_AGENTS;
    m.progress /= (float)N_AGENTS;
    (void)nodes;
    return m;
}

static cv::Point to_px(float x, float y) {
    int px = static_cast<int>(x / WORLD_W * PANEL_W);
    int py = static_cast<int>((1.0f - y / WORLD_H) * PANEL_H);
    return cv::Point(px, py);
}

static cv::Scalar node_color(float risk, float density, int kind) {
    risk = clampf(risk, 0.0f, 1.0f);
    density = clampf(density, 0.0f, 1.0f);
    if (kind == 1 && risk < 0.35f) return cv::Scalar(250, 214, 82);
    int blue = static_cast<int>(clampf(185.0f * (1.0f - risk) + 40.0f * density, 25.0f, 210.0f));
    int green = static_cast<int>(clampf(235.0f * (1.0f - 0.44f * risk), 65.0f, 235.0f));
    int red = static_cast<int>(clampf(72.0f + 178.0f * risk + 35.0f * density, 72.0f, 255.0f));
    return cv::Scalar(blue, green, red);
}

static void draw_roads(cv::Mat& img) {
    img = cv::Mat(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(21, 22, 25));
    int y0 = to_px(0.0f, CENTER_Y + ROAD_HALF_H).y;
    int y1 = to_px(0.0f, CENTER_Y - ROAD_HALF_H).y;
    int x0 = to_px(CENTER_X - ROAD_HALF_W, 0.0f).x;
    int x1 = to_px(CENTER_X + ROAD_HALF_W, 0.0f).x;
    cv::rectangle(img, cv::Rect(0, y0, PANEL_W, y1 - y0), cv::Scalar(44, 46, 50), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0, 0, x1 - x0, PANEL_H), cv::Scalar(44, 46, 50), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0, y0, x1 - x0, y1 - y0),
                  cv::Scalar(60, 62, 68), cv::FILLED);
    cv::line(img, to_px(0.0f, CENTER_Y), to_px(WORLD_W, CENTER_Y),
             cv::Scalar(96, 97, 102), 1, cv::LINE_AA);
    cv::line(img, to_px(CENTER_X, 0.0f), to_px(CENTER_X, WORLD_H),
             cv::Scalar(96, 97, 102), 1, cv::LINE_AA);
}

static void draw_graph_edges(cv::Mat& img,
                             const std::vector<Agent>& agents,
                             const std::vector<Action>& actions,
                             const std::vector<int>& order) {
    int drawn = 0;
    for (int k = 0; k < std::min(80, N_AGENTS) && drawn < 38; k++) {
        int i = order[k];
        if (actions[i].risk < 0.38f && actions[i].density < 0.50f) continue;
        int best = -1;
        float best_score = 0.0f;
        for (int j = 0; j < N_AGENTS; j++) {
            if (j == i) continue;
            PairTerms t = pair_terms(agents[i], agents[j]);
            float score = t.w * (0.55f + actions[j].risk) + 0.45f * t.risk;
            if (score > best_score) {
                best_score = score;
                best = j;
            }
        }
        if (best >= 0 && best_score > 0.28f) {
            cv::line(img, to_px(agents[i].x, agents[i].y), to_px(agents[best].x, agents[best].y),
                     cv::Scalar(76, 198, 255), 1, cv::LINE_AA);
            drawn++;
        }
    }
}

static cv::Mat draw_frame(const std::vector<Agent>& agents,
                          const std::vector<NodeState>& nodes,
                          const std::vector<Action>& actions,
                          float gpu_ms,
                          double cpu_ms,
                          int frame) {
    cv::Mat img;
    draw_roads(img);

    std::vector<int> order(N_AGENTS);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        float sa = actions[a].risk + 0.55f * actions[a].density;
        float sb = actions[b].risk + 0.55f * actions[b].density;
        return sa > sb;
    });
    draw_graph_edges(img, agents, actions, order);

    for (int i = 0; i < N_AGENTS; i++) {
        cv::Point p = to_px(agents[i].x, agents[i].y);
        int radius = agents[i].kind == 1 ? 4 : 2;
        cv::circle(img, p, radius, node_color(actions[i].risk, actions[i].density, agents[i].kind),
                   cv::FILLED, cv::LINE_AA);
    }
    for (int k = 0; k < 20 && k < N_AGENTS; k++) {
        int i = order[k];
        if (actions[i].risk < 0.42f) continue;
        cv::circle(img, to_px(agents[i].x, agents[i].y), 8,
                   cv::Scalar(67, 98, 255), 1, cv::LINE_AA);
    }

    Metrics m = summarize(agents, nodes, actions);
    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, 36), cv::Scalar(5, 7, 10), cv::FILLED);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "GPU GNN swarm controller  agents=%d  passes=%d  gpu=%.2f ms  cpu=%.1f ms",
                  N_AGENTS, MESSAGE_PASSES, gpu_ms, cpu_ms);
    cv::putText(img, buf, cv::Point(12, 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.53, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);

    std::snprintf(buf, sizeof(buf),
                  "frame %02d  hot=%d  close pairs=%d  density=%.3f  risk=%.3f  progress=%.2f m/s",
                  frame, m.hot_agents, m.close_pairs, m.mean_density, m.mean_risk, m.progress);
    cv::putText(img, buf, cv::Point(12, PANEL_H - 16),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(225, 238, 245), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<Agent> h_agents = make_agents();
    std::vector<NodeState> h_nodes(N_AGENTS);
    std::vector<Action> h_actions(N_AGENTS);

    Agent* d_agents = nullptr;
    NodeState* d_nodes_a = nullptr;
    NodeState* d_nodes_b = nullptr;
    Action* d_actions = nullptr;
    CUDA_CHECK(cudaMalloc(&d_agents, N_AGENTS * sizeof(Agent)));
    CUDA_CHECK(cudaMalloc(&d_nodes_a, N_AGENTS * sizeof(NodeState)));
    CUDA_CHECK(cudaMalloc(&d_nodes_b, N_AGENTS * sizeof(NodeState)));
    CUDA_CHECK(cudaMalloc(&d_actions, N_AGENTS * sizeof(Action)));
    CUDA_CHECK(cudaMemcpy(d_agents, h_agents.data(), N_AGENTS * sizeof(Agent),
                          cudaMemcpyHostToDevice));

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_gnn_swarm_controller.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_gnn_swarm_controller.avi\n");
        return 1;
    }

    int blocks = (N_AGENTS + THREADS - 1) / THREADS;
    double total_gpu_ms = 0.0;
    int measured = 0;
    float last_gpu_ms = 0.0f;
    double cpu_ms = 0.0;

    for (int frame = 0; frame < N_FRAMES; frame++) {
        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0));
        init_nodes_kernel<<<blocks, THREADS>>>(d_agents, d_nodes_a);
        NodeState* in = d_nodes_a;
        NodeState* out = d_nodes_b;
        for (int pass = 0; pass < MESSAGE_PASSES; pass++) {
            message_pass_kernel<<<blocks, THREADS>>>(d_agents, in, out);
            NodeState* tmp = in;
            in = out;
            out = tmp;
        }
        policy_kernel<<<blocks, THREADS>>>(d_agents, in, d_actions);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaEventElapsedTime(&last_gpu_ms, ev0, ev1));
        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));
        CUDA_CHECK(cudaGetLastError());

        step_agents_kernel<<<blocks, THREADS>>>(d_agents, d_actions, frame);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (frame >= 5) {
            total_gpu_ms += last_gpu_ms;
            measured++;
        }

        CUDA_CHECK(cudaMemcpy(h_agents.data(), d_agents, N_AGENTS * sizeof(Agent),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_nodes.data(), in, N_AGENTS * sizeof(NodeState),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_actions.data(), d_actions, N_AGENTS * sizeof(Action),
                              cudaMemcpyDeviceToHost));

        if (frame == 8) {
            cpu_ms = cpu_controller_ms(h_agents);
        }
        if ((frame >= 8 && frame % 2 == 0) || frame == N_FRAMES - 1) {
            video.write(draw_frame(h_agents, h_nodes, h_actions, last_gpu_ms, cpu_ms, frame));
        }
        if (frame % 30 == 0) {
            Metrics m = summarize(h_agents, h_nodes, h_actions);
            std::printf("  frame %3d  gpu %.3f ms  hot %d  close %d  density %.3f\n",
                        frame, last_gpu_ms, m.hot_agents, m.close_pairs, m.mean_density);
        }
    }

    for (int i = 0; i < HOLD_FRAMES; i++) {
        video.write(draw_frame(h_agents, h_nodes, h_actions, last_gpu_ms, cpu_ms, N_FRAMES));
    }
    video.release();

    double avg_gpu = measured > 0 ? total_gpu_ms / measured : 0.0;
    double speedup = cpu_ms > 0.0 ? cpu_ms / avg_gpu : 0.0;
    std::printf("CPU GNN controller: %.3f ms\n", cpu_ms);
    std::printf("Avg GPU GNN controller: %.3f ms (%d agents x %d passes, %.1fx vs CPU)\n",
                avg_gpu, N_AGENTS, MESSAGE_PASSES, speedup);

    avi_to_gif("gif/gpu_gnn_swarm_controller.avi",
               "gif/gpu_gnn_swarm_controller.gif", 8, 560);
    std::printf("GIF saved to gif/gpu_gnn_swarm_controller.gif\n");

    CUDA_CHECK(cudaFree(d_agents));
    CUDA_CHECK(cudaFree(d_nodes_a));
    CUDA_CHECK(cudaFree(d_nodes_b));
    CUDA_CHECK(cudaFree(d_actions));
    return 0;
}
