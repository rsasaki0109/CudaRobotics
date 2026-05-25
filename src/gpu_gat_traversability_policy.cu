// gpu_gat_traversability_policy.cu
//
// GPU graph-attention traversability policy.
//
// A noisy local terrain classifier produces free/caution/blocked unary logits
// and a weak route prior.  A small fixed-weight, multi-head graph-attention
// network then refines both traversability and a goal-conditioned corridor
// score over an implicit terrain graph.  This is a robotics graph-ML primitive:
// terrain perception becomes a policy layer a planner can follow.
//
// Output: gif/gpu_gat_traversability_policy.gif

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
constexpr int GAT_LAYERS = 4;
constexpr int N_HEADS = 3;
constexpr int SNAP_STRIDE = 1;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int HEADER_H = 44;
constexpr int FOOTER_H = 40;
constexpr int MAP_H = PANEL_H - HEADER_H - FOOTER_H;
constexpr int HALF_W = PANEL_W / 2;
constexpr int VIDEO_FPS = 9;
constexpr int THREADS = 128;

constexpr float WORLD_W = 18.0f;
constexpr float WORLD_H = 11.0f;
constexpr float START_X = 0.95f;
constexpr float START_Y = 1.15f;
constexpr float GOAL_X = 17.15f;
constexpr float GOAL_Y = 9.55f;
constexpr float GRAPH_R = 1.55f;
constexpr float GRAPH_R2 = GRAPH_R * GRAPH_R;
constexpr float INV_TWO_SIGMA2 = 1.0f / (2.0f * 0.72f * 0.72f);
constexpr float DAMPING = 0.12f;

struct Node {
    float x;
    float y;
    float roughness;
    float clearance;
    float height;
    float goal_dx;
    float goal_dy;
    float goal_dist;
    float route_prior;
    int truth;
};

struct ClassVec {
    float free_v;
    float caution_v;
    float blocked_v;
};

struct State {
    float free_p;
    float caution_p;
    float blocked_p;
    float corridor;
    float risk;
};

struct Metrics {
    float accuracy = 0.0f;
    float entropy = 0.0f;
    float corridor_passable = 0.0f;
    float path_blocked = 0.0f;
    float path_cost = 0.0f;
    int path_steps = 0;
    int reached = 0;
};

struct Snapshot {
    int layer = 0;
    std::vector<State> states;
    Metrics metrics;
    std::vector<int> path;
};

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline float sqr(float x) {
    return x * x;
}

__host__ __device__ static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
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

__host__ __device__ static inline State mix(State a, State b, float wa) {
    float wb = 1.0f - wa;
    State out;
    out.free_p = wa * a.free_p + wb * b.free_p;
    out.caution_p = wa * a.caution_p + wb * b.caution_p;
    out.blocked_p = wa * a.blocked_p + wb * b.blocked_p;
    float inv = 1.0f / fmaxf(out.free_p + out.caution_p + out.blocked_p, 1.0e-6f);
    out.free_p *= inv;
    out.caution_p *= inv;
    out.blocked_p *= inv;
    out.corridor = clampf(wa * a.corridor + wb * b.corridor, 0.0f, 1.0f);
    out.risk = clampf(wa * a.risk + wb * b.risk, 0.0f, 1.0f);
    return out;
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
    logits.free_v = 0.34f + 1.86f * clear * height_free_gate
                  - 1.18f * rough - 0.96f * height_caution - 1.10f * collision;
    logits.caution_v = 0.12f + 1.34f * boundary + 1.08f * rough
                     + 0.76f * height_caution - 0.16f * collision;
    logits.blocked_v = -0.23f + 1.88f * low_clear + 1.52f * sqr(rough)
                     + 1.10f * collision;
    return logits;
}

__host__ __device__ static inline int argmax_label(State v) {
    if (v.blocked_p > v.free_p && v.blocked_p > v.caution_p) return 2;
    if (v.caution_p > v.free_p) return 1;
    return 0;
}

__host__ __device__ static inline float entropy(State v) {
    return -(v.free_p * logf(fmaxf(v.free_p, 1.0e-6f))
           + v.caution_p * logf(fmaxf(v.caution_p, 1.0e-6f))
           + v.blocked_p * logf(fmaxf(v.blocked_p, 1.0e-6f))) / 1.0986122887f;
}

__host__ __device__ static inline float attention_logit(const Node& a,
                                                        const Node& b,
                                                        const State& qj,
                                                        int head) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float d2 = dx * dx + dy * dy;
    if (d2 > GRAPH_R2 || d2 < 1.0e-8f) return -1.0e20f;

    float d = sqrtf(d2) + 1.0e-6f;
    float terrain_sim = -1.55f * sqr(a.roughness - b.roughness)
                      - 1.25f * sqr(a.clearance - b.clearance)
                      - 0.90f * sqr(a.height - b.height);
    float base = -d2 * INV_TWO_SIGMA2 + terrain_sim;
    float goal_step = (dx * a.goal_dx + dy * a.goal_dy) / d;
    float progress = clampf((a.goal_dist - b.goal_dist) / GRAPH_R, -1.0f, 1.0f);

    if (head == 0) {
        return base + 1.55f * qj.free_p - 1.85f * qj.blocked_p
             - 0.45f * b.roughness + 0.42f * b.clearance;
    }
    if (head == 1) {
        return base + 1.22f * qj.caution_p + 0.72f * qj.free_p
             - 0.62f * fabsf(a.clearance - b.clearance);
    }
    return base + 1.35f * qj.corridor + 1.08f * goal_step
         + 0.96f * progress - 1.28f * qj.blocked_p;
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
    logits.free_v += 0.38f * (hash01(i, 1) - 0.5f);
    logits.caution_v += 0.38f * (hash01(i, 2) - 0.5f);
    logits.blocked_v += 0.38f * (hash01(i, 3) - 0.5f);

    float corrupt = hash01(i, 4);
    if (corrupt < 0.18f) {
        int wrong = (n.truth + 1 + static_cast<int>(hash01(i, 5) > 0.50f)) % 3;
        bump_label(logits, wrong, 1.28f + 0.28f * hash01(i, 6));
        bump_label(logits, n.truth, -0.38f);
    } else {
        bump_label(logits, n.truth, 0.58f);
    }

    return logits;
}

__global__ void init_state_kernel(const Node* __restrict__ nodes,
                                  const ClassVec* __restrict__ unary,
                                  State* __restrict__ states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_NODES) return;
    ClassVec p = softmax(unary[i]);
    float corridor_logit = 1.90f * p.free_v - 1.55f * p.blocked_v - 0.42f * p.caution_v
                         + 1.20f * nodes[i].route_prior - 0.50f * nodes[i].roughness
                         + 0.34f * nodes[i].clearance;
    states[i] = {p.free_v, p.caution_v, p.blocked_v, sigmoid(corridor_logit),
                 clampf(0.62f * p.blocked_v + 0.25f * nodes[i].roughness
                      + 0.13f * (1.0f - nodes[i].clearance), 0.0f, 1.0f)};
}

__global__ void gat_layer_kernel(const Node* __restrict__ nodes,
                                 const ClassVec* __restrict__ unary,
                                 const State* __restrict__ in,
                                 State* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_NODES) return;

    Node ni = nodes[i];
    State heads[N_HEADS];
    for (int h = 0; h < N_HEADS; h++) {
        float max_logit = -1.0e20f;
        for (int j = 0; j < N_NODES; j++) {
            if (j == i) continue;
            max_logit = fmaxf(max_logit, attention_logit(ni, nodes[j], in[j], h));
        }

        State acc{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float denom = 0.0f;
        for (int j = 0; j < N_NODES; j++) {
            if (j == i) continue;
            float logit = attention_logit(ni, nodes[j], in[j], h);
            if (logit < -1.0e10f) continue;
            float a = expf(logit - max_logit);
            acc.free_p += a * in[j].free_p;
            acc.caution_p += a * in[j].caution_p;
            acc.blocked_p += a * in[j].blocked_p;
            acc.corridor += a * in[j].corridor;
            acc.risk += a * in[j].risk;
            denom += a;
        }

        if (denom > 1.0e-6f) {
            float inv = 1.0f / denom;
            heads[h] = {acc.free_p * inv, acc.caution_p * inv, acc.blocked_p * inv,
                        acc.corridor * inv, acc.risk * inv};
        } else {
            heads[h] = in[i];
        }
    }

    State safety = heads[0];
    State boundary = heads[1];
    State corridor = heads[2];
    ClassVec logits;
    logits.free_v = 0.86f * unary[i].free_v + 0.90f * safety.free_p
                  + 0.46f * corridor.free_p + 0.56f * corridor.corridor
                  - 0.66f * safety.blocked_p;
    logits.caution_v = 0.82f * unary[i].caution_v + 0.72f * boundary.caution_p
                     + 0.30f * safety.caution_p + 0.22f * safety.blocked_p;
    logits.blocked_v = 0.86f * unary[i].blocked_v + 0.80f * safety.blocked_p
                     + 0.36f * boundary.blocked_p - 0.32f * corridor.corridor;
    ClassVec p = softmax(logits);

    float corridor_logit = 2.05f * p.free_v - 1.72f * p.blocked_v - 0.58f * p.caution_v
                         + 1.18f * corridor.corridor + 0.70f * ni.route_prior
                         - 0.58f * ni.roughness + 0.36f * ni.clearance
                         - 0.40f * corridor.risk;
    State next;
    next.free_p = p.free_v;
    next.caution_p = p.caution_v;
    next.blocked_p = p.blocked_v;
    next.corridor = sigmoid(corridor_logit);
    next.risk = clampf(0.54f * p.blocked_v + 0.24f * ni.roughness
                     + 0.12f * (1.0f - ni.clearance) + 0.24f * safety.risk,
                       0.0f, 1.0f);
    out[i] = mix(next, in[i], 1.0f - DAMPING);
}

static std::vector<Node> make_nodes() {
    std::vector<Node> nodes(N_NODES);
    std::mt19937 rng(26052026);
    std::uniform_real_distribution<float> jitter(-0.34f, 0.34f);
    float route_x = GOAL_X - START_X;
    float route_y = GOAL_Y - START_Y;
    float route_len2 = route_x * route_x + route_y * route_y;
    float route_len = sqrtf(route_len2);

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
            float gx_to_goal = GOAL_X - x;
            float gy_to_goal = GOAL_Y - y;
            float goal_dist = sqrtf(gx_to_goal * gx_to_goal + gy_to_goal * gy_to_goal) + 1.0e-6f;
            float goal_dx = gx_to_goal / goal_dist;
            float goal_dy = gy_to_goal / goal_dist;
            float sx = x - START_X;
            float sy = y - START_Y;
            float t = clampf((sx * route_x + sy * route_y) / route_len2, 0.0f, 1.0f);
            float proj_x = START_X + t * route_x;
            float proj_y = START_Y + t * route_y;
            float off_route = sqrtf(sqr(x - proj_x) + sqr(y - proj_y));
            float route_prior = expf(-0.5f * sqr(off_route / 1.70f))
                              * (0.34f + 0.66f * t)
                              * clampf((route_len - goal_dist + 2.0f) / route_len, 0.0f, 1.0f);
            nodes[i] = {x, y, rough, clear, h, goal_dx, goal_dy, goal_dist,
                        route_prior, truth};
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

static void init_state_host(const std::vector<Node>& nodes,
                            const std::vector<ClassVec>& unary,
                            std::vector<State>& states) {
    for (int i = 0; i < N_NODES; i++) {
        ClassVec p = softmax(unary[i]);
        float corridor_logit = 1.90f * p.free_v - 1.55f * p.blocked_v - 0.42f * p.caution_v
                             + 1.20f * nodes[i].route_prior - 0.50f * nodes[i].roughness
                             + 0.34f * nodes[i].clearance;
        states[i] = {p.free_v, p.caution_v, p.blocked_v, sigmoid(corridor_logit),
                     clampf(0.62f * p.blocked_v + 0.25f * nodes[i].roughness
                          + 0.13f * (1.0f - nodes[i].clearance), 0.0f, 1.0f)};
    }
}

static void gat_layer_host(const std::vector<Node>& nodes,
                           const std::vector<ClassVec>& unary,
                           const std::vector<State>& in,
                           std::vector<State>& out) {
    for (int i = 0; i < N_NODES; i++) {
        const Node& ni = nodes[i];
        State heads[N_HEADS];
        for (int h = 0; h < N_HEADS; h++) {
            float max_logit = -1.0e20f;
            for (int j = 0; j < N_NODES; j++) {
                if (j == i) continue;
                max_logit = fmaxf(max_logit, attention_logit(ni, nodes[j], in[j], h));
            }

            State acc{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            float denom = 0.0f;
            for (int j = 0; j < N_NODES; j++) {
                if (j == i) continue;
                float logit = attention_logit(ni, nodes[j], in[j], h);
                if (logit < -1.0e10f) continue;
                float a = expf(logit - max_logit);
                acc.free_p += a * in[j].free_p;
                acc.caution_p += a * in[j].caution_p;
                acc.blocked_p += a * in[j].blocked_p;
                acc.corridor += a * in[j].corridor;
                acc.risk += a * in[j].risk;
                denom += a;
            }

            if (denom > 1.0e-6f) {
                float inv = 1.0f / denom;
                heads[h] = {acc.free_p * inv, acc.caution_p * inv, acc.blocked_p * inv,
                            acc.corridor * inv, acc.risk * inv};
            } else {
                heads[h] = in[i];
            }
        }

        State safety = heads[0];
        State boundary = heads[1];
        State corridor = heads[2];
        ClassVec logits;
        logits.free_v = 0.86f * unary[i].free_v + 0.90f * safety.free_p
                      + 0.46f * corridor.free_p + 0.56f * corridor.corridor
                      - 0.66f * safety.blocked_p;
        logits.caution_v = 0.82f * unary[i].caution_v + 0.72f * boundary.caution_p
                         + 0.30f * safety.caution_p + 0.22f * safety.blocked_p;
        logits.blocked_v = 0.86f * unary[i].blocked_v + 0.80f * safety.blocked_p
                         + 0.36f * boundary.blocked_p - 0.32f * corridor.corridor;
        ClassVec p = softmax(logits);

        float corridor_logit = 2.05f * p.free_v - 1.72f * p.blocked_v - 0.58f * p.caution_v
                             + 1.18f * corridor.corridor + 0.70f * ni.route_prior
                             - 0.58f * ni.roughness + 0.36f * ni.clearance
                             - 0.40f * corridor.risk;
        State next;
        next.free_p = p.free_v;
        next.caution_p = p.caution_v;
        next.blocked_p = p.blocked_v;
        next.corridor = sigmoid(corridor_logit);
        next.risk = clampf(0.54f * p.blocked_v + 0.24f * ni.roughness
                         + 0.12f * (1.0f - ni.clearance) + 0.24f * safety.risk,
                           0.0f, 1.0f);
        out[i] = mix(next, in[i], 1.0f - DAMPING);
    }
}

static int nearest_node(const std::vector<Node>& nodes, float x, float y) {
    int best = 0;
    float best_d2 = 1.0e20f;
    for (int i = 0; i < N_NODES; i++) {
        float d2 = sqr(nodes[i].x - x) + sqr(nodes[i].y - y);
        if (d2 < best_d2) {
            best_d2 = d2;
            best = i;
        }
    }
    return best;
}

static std::vector<int> extract_path(const std::vector<Node>& nodes,
                                     const std::vector<State>& states) {
    std::vector<int> path;
    std::vector<unsigned char> used(N_NODES, 0);
    int current = nearest_node(nodes, START_X, START_Y);
    int goal = nearest_node(nodes, GOAL_X, GOAL_Y);
    path.push_back(current);
    used[current] = 1;

    for (int step = 0; step < 72 && current != goal; step++) {
        const Node& nc = nodes[current];
        int best = -1;
        float best_cost = 1.0e20f;
        for (int j = 0; j < N_NODES; j++) {
            if (used[j]) continue;
            float dx = nodes[j].x - nc.x;
            float dy = nodes[j].y - nc.y;
            float d = sqrtf(dx * dx + dy * dy);
            if (d < 0.18f || d > 0.96f) continue;
            float progress = nc.goal_dist - nodes[j].goal_dist;
            if (progress < -0.20f) continue;
            float cost = nodes[j].goal_dist + 1.30f * states[j].risk
                       + 1.95f * states[j].blocked_p + 0.58f * states[j].caution_p
                       + 0.46f * nodes[j].roughness - 1.30f * states[j].corridor
                       - 0.28f * progress + 0.18f * d;
            if (cost < best_cost) {
                best_cost = cost;
                best = j;
            }
        }
        if (best < 0) break;
        current = best;
        used[current] = 1;
        path.push_back(current);
        if (nodes[current].goal_dist < 0.58f) break;
    }
    return path;
}

static Metrics evaluate(const std::vector<Node>& nodes, const std::vector<State>& states) {
    Metrics m;
    int correct = 0;
    std::vector<float> corridor_values(N_NODES);
    for (int i = 0; i < N_NODES; i++) {
        int pred = argmax_label(states[i]);
        if (pred == nodes[i].truth) correct++;
        m.entropy += entropy(states[i]);
        corridor_values[i] = states[i].corridor;
    }
    m.accuracy = static_cast<float>(correct) / static_cast<float>(N_NODES);
    m.entropy /= static_cast<float>(N_NODES);

    int kth = static_cast<int>(0.82f * static_cast<float>(N_NODES));
    std::nth_element(corridor_values.begin(), corridor_values.begin() + kth, corridor_values.end());
    float threshold = corridor_values[kth];
    int passable = 0;
    int selected = 0;
    for (int i = 0; i < N_NODES; i++) {
        if (states[i].corridor >= threshold) {
            selected++;
            if (nodes[i].truth != 2) passable++;
        }
    }
    m.corridor_passable = selected > 0
                        ? static_cast<float>(passable) / static_cast<float>(selected)
                        : 0.0f;

    std::vector<int> path = extract_path(nodes, states);
    float blocked = 0.0f;
    float cost = 0.0f;
    for (int idx : path) {
        blocked += nodes[idx].truth == 2 ? 1.0f : 0.0f;
        cost += 1.0f + 2.2f * states[idx].risk + 4.5f * states[idx].blocked_p
              + 0.9f * nodes[idx].roughness - 1.4f * states[idx].corridor;
    }
    m.path_steps = static_cast<int>(path.size());
    m.path_blocked = path.empty() ? 0.0f : blocked / static_cast<float>(path.size());
    m.path_cost = path.empty() ? 0.0f : cost / static_cast<float>(path.size());
    m.reached = (!path.empty() && nodes[path.back()].goal_dist < 0.72f) ? 1 : 0;
    return m;
}

static double cpu_gat_ms(const std::vector<Node>& nodes,
                         const std::vector<ClassVec>& unary,
                         Metrics& out_metrics) {
    std::vector<State> a(N_NODES);
    std::vector<State> b(N_NODES);
    init_state_host(nodes, unary, a);
    std::vector<State>* in = &a;
    std::vector<State>* out = &b;

    auto begin = std::chrono::high_resolution_clock::now();
    for (int layer = 0; layer < GAT_LAYERS; layer++) {
        gat_layer_host(nodes, unary, *in, *out);
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

static cv::Scalar corridor_color(float v) {
    v = clampf(v, 0.0f, 1.0f);
    return cv::Scalar(55 + 120 * v, 90 + 115 * v, 80 + 80 * v);
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
                              const std::vector<State>& states,
                              int x0,
                              bool show_corridor,
                              bool mark_errors) {
    for (int i = 0; i < N_NODES; i++) {
        cv::Point p = to_px(nodes[i].x, nodes[i].y, x0);
        if (show_corridor && states[i].corridor > 0.66f) {
            int r = states[i].corridor > 0.82f ? 5 : 4;
            cv::circle(img, p, r, corridor_color(states[i].corridor), cv::FILLED, cv::LINE_AA);
        }
    }
    for (int i = 0; i < N_NODES; i++) {
        int pred = argmax_label(states[i]);
        cv::Point p = to_px(nodes[i].x, nodes[i].y, x0);
        cv::circle(img, p, 2, label_color(pred), cv::FILLED, cv::LINE_AA);
        if (mark_errors && pred != nodes[i].truth && i % 2 == 0) {
            cv::circle(img, p, 4, cv::Scalar(246, 246, 246), 1, cv::LINE_AA);
        }
    }
}

static void draw_path(cv::Mat& img,
                      const std::vector<Node>& nodes,
                      const std::vector<int>& path,
                      int x0) {
    for (size_t k = 1; k < path.size(); k++) {
        cv::line(img, to_px(nodes[path[k - 1]].x, nodes[path[k - 1]].y, x0),
                 to_px(nodes[path[k]].x, nodes[path[k]].y, x0),
                 cv::Scalar(246, 246, 246), 2, cv::LINE_AA);
    }
    cv::circle(img, to_px(START_X, START_Y, x0), 7, cv::Scalar(255, 245, 125),
               cv::FILLED, cv::LINE_AA);
    cv::circle(img, to_px(GOAL_X, GOAL_Y, x0), 8, cv::Scalar(245, 120, 255),
               2, cv::LINE_AA);
}

static void draw_legend(cv::Mat& img) {
    constexpr int x0 = PANEL_W - 262;
    constexpr int y0 = HEADER_H + 52;
    cv::rectangle(img, cv::Rect(x0, y0, 244, 112), cv::Scalar(8, 10, 13), cv::FILLED);
    cv::circle(img, cv::Point(x0 + 20, y0 + 22), 5, label_color(0), cv::FILLED, cv::LINE_AA);
    cv::putText(img, "free", cv::Point(x0 + 36, y0 + 27),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::circle(img, cv::Point(x0 + 20, y0 + 48), 5, label_color(1), cv::FILLED, cv::LINE_AA);
    cv::putText(img, "caution", cv::Point(x0 + 36, y0 + 53),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::circle(img, cv::Point(x0 + 20, y0 + 74), 5, label_color(2), cv::FILLED, cv::LINE_AA);
    cv::putText(img, "blocked", cv::Point(x0 + 36, y0 + 79),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::circle(img, cv::Point(x0 + 20, y0 + 100), 5, corridor_color(0.90f),
               cv::FILLED, cv::LINE_AA);
    cv::putText(img, "policy corridor", cv::Point(x0 + 36, y0 + 105),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
}

static cv::Mat draw_frame(const std::vector<Node>& nodes,
                          const Snapshot& initial,
                          const Snapshot& snap,
                          double gpu_ms,
                          double cpu_ms,
                          const Metrics& cpu_metrics) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 20, 24));
    draw_background_panel(img, 0);
    draw_background_panel(img, HALF_W);
    cv::line(img, cv::Point(HALF_W, HEADER_H), cv::Point(HALF_W, PANEL_H - FOOTER_H),
             cv::Scalar(18, 18, 20), 1, cv::LINE_AA);

    draw_panel_points(img, nodes, initial.states, 0, false, true);
    draw_panel_points(img, nodes, snap.states, HALF_W, true, true);
    draw_path(img, nodes, snap.path, HALF_W);
    draw_legend(img);

    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, HEADER_H), cv::Scalar(5, 7, 10), cv::FILLED);
    cv::rectangle(img, cv::Rect(0, PANEL_H - FOOTER_H, PANEL_W, FOOTER_H),
                  cv::Scalar(5, 7, 10), cv::FILLED);

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "GPU GAT traversability policy  nodes=%d  heads=%d  layers=%d  gpu=%.2f ms  cpu=%.1f ms",
                  N_NODES, N_HEADS, GAT_LAYERS, gpu_ms, cpu_ms);
    cv::putText(img, buf, cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);

    cv::putText(img, "noisy unary", cv::Point(14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.54, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "GAT policy corridor", cv::Point(HALF_W + 14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.54, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);

    std::snprintf(buf, sizeof(buf),
                  "layer %d  unary acc=%.1f%%  GAT acc=%.1f%%  corridor pass=%.1f%%  path blocked=%.1f%%  cpu acc=%.1f%%",
                  snap.layer, 100.0f * initial.metrics.accuracy,
                  100.0f * snap.metrics.accuracy,
                  100.0f * snap.metrics.corridor_passable,
                  100.0f * snap.metrics.path_blocked,
                  100.0f * cpu_metrics.accuracy);
    cv::putText(img, buf, cv::Point(12, PANEL_H - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(225, 238, 245), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<Node> nodes = make_nodes();
    std::vector<ClassVec> unary = make_unary(nodes);
    std::vector<State> initial_states(N_NODES);
    init_state_host(nodes, unary, initial_states);
    Snapshot initial{0, initial_states, evaluate(nodes, initial_states),
                     extract_path(nodes, initial_states)};

    Metrics cpu_metrics;
    double cpu_ms = cpu_gat_ms(nodes, unary, cpu_metrics);

    Node* d_nodes = nullptr;
    ClassVec* d_unary = nullptr;
    State* d_a = nullptr;
    State* d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nodes, N_NODES * sizeof(Node)));
    CUDA_CHECK(cudaMalloc(&d_unary, N_NODES * sizeof(ClassVec)));
    CUDA_CHECK(cudaMalloc(&d_a, N_NODES * sizeof(State)));
    CUDA_CHECK(cudaMalloc(&d_b, N_NODES * sizeof(State)));
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes.data(), N_NODES * sizeof(Node), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_unary, unary.data(), N_NODES * sizeof(ClassVec), cudaMemcpyHostToDevice));

    int blocks = (N_NODES + THREADS - 1) / THREADS;
    init_state_kernel<<<blocks, THREADS>>>(d_nodes, d_unary, d_a);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    State* in = d_a;
    State* out = d_b;
    for (int layer = 0; layer < GAT_LAYERS; layer++) {
        gat_layer_kernel<<<blocks, THREADS>>>(d_nodes, d_unary, in, out);
        State* tmp = in;
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
    std::vector<State> h_states(N_NODES);
    init_state_kernel<<<blocks, THREADS>>>(d_nodes, d_unary, d_a);
    CUDA_CHECK(cudaDeviceSynchronize());
    in = d_a;
    out = d_b;
    CUDA_CHECK(cudaMemcpy(h_states.data(), in, N_NODES * sizeof(State), cudaMemcpyDeviceToHost));
    snapshots.push_back({0, h_states, evaluate(nodes, h_states), extract_path(nodes, h_states)});
    for (int layer = 1; layer <= GAT_LAYERS; layer++) {
        gat_layer_kernel<<<blocks, THREADS>>>(d_nodes, d_unary, in, out);
        CUDA_CHECK(cudaDeviceSynchronize());
        State* tmp = in;
        in = out;
        out = tmp;
        if (layer % SNAP_STRIDE == 0 || layer == GAT_LAYERS) {
            CUDA_CHECK(cudaMemcpy(h_states.data(), in, N_NODES * sizeof(State),
                                  cudaMemcpyDeviceToHost));
            snapshots.push_back({layer, h_states, evaluate(nodes, h_states),
                                 extract_path(nodes, h_states)});
        }
    }

    double speedup = cpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0;
    const Metrics& final_gpu = snapshots.back().metrics;
    std::printf("Noisy unary policy: accuracy %.2f%%, corridor passable %.2f%%, path blocked %.2f%%\n",
                100.0f * initial.metrics.accuracy,
                100.0f * initial.metrics.corridor_passable,
                100.0f * initial.metrics.path_blocked);
    std::printf("CPU GAT policy: %.3f ms, accuracy %.2f%%, corridor passable %.2f%%\n",
                cpu_ms, 100.0f * cpu_metrics.accuracy,
                100.0f * cpu_metrics.corridor_passable);
    std::printf("GPU GAT policy: %.3f ms (%d nodes x %d heads x %d layers, %.1fx vs CPU, accuracy %.2f%%, corridor passable %.2f%%, path blocked %.2f%%)\n",
                gpu_ms, N_NODES, N_HEADS, GAT_LAYERS, speedup,
                100.0f * final_gpu.accuracy,
                100.0f * final_gpu.corridor_passable,
                100.0f * final_gpu.path_blocked);

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_gat_traversability_policy.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_gat_traversability_policy.avi\n");
        return 1;
    }
    for (const Snapshot& s : snapshots) {
        video.write(draw_frame(nodes, initial, s, gpu_ms, cpu_ms, cpu_metrics));
    }
    for (int i = 0; i < 15; i++) {
        video.write(draw_frame(nodes, initial, snapshots.back(), gpu_ms, cpu_ms, cpu_metrics));
    }
    video.release();

    avi_to_gif("gif/gpu_gat_traversability_policy.avi",
               "gif/gpu_gat_traversability_policy.gif", 8, 560);
    std::printf("GIF saved to gif/gpu_gat_traversability_policy.gif\n");

    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_unary));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
