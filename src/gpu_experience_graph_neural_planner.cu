// gpu_experience_graph_neural_planner.cu
//
// Batched GPU learned experience-graph planner.
//
// The demo turns a learned traversability field into a sparse waypoint graph
// with stored experience-corridor priors.  A fixed-weight MLP-style edge model
// scores terrain, clearance, route alignment, dynamic risk, and experience
// reuse.  Each CUDA block runs one graph A* query over a 48x32 waypoint graph,
// letting 128 start/goal queries run in parallel.
//
// Output: gif/gpu_experience_graph_neural_planner.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int GRAPH_W = 48;
constexpr int GRAPH_H = 32;
constexpr int N_NODES = GRAPH_W * GRAPH_H;
constexpr int MAX_DEGREE = 12;
constexpr int BATCH_QUERIES = 128;
constexpr int MAX_EXPANSIONS = N_NODES;
constexpr int SNAP_STRIDE = 18;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int HEADER_H = 44;
constexpr int FOOTER_H = 40;
constexpr int MAP_H = PANEL_H - HEADER_H - FOOTER_H;
constexpr int HALF_W = PANEL_W / 2;
constexpr int VIDEO_FPS = 9;
constexpr int THREADS = 256;

constexpr float WORLD_W = 18.0f;
constexpr float WORLD_H = 11.0f;
constexpr float INF_COST = 1.0e20f;
constexpr float BLOCK_COST = 20.0f;
constexpr float HEURISTIC_WEIGHT = 0.70f;

struct Node {
    float x;
    float y;
    float roughness;
    float clearance;
    float height;
    float experience;
    float cost;
    int truth;
    int blocked;
};

struct Metrics {
    float path_cost = 0.0f;
    float edge_risk = 0.0f;
    float experience_mean = 0.0f;
    float goal_cost = INF_COST;
    int path_steps = 0;
    int reached = 0;
    int expanded = 0;
    int opened = 0;
    int goal_idx = -1;
};

struct SearchResult {
    std::vector<float> g;
    std::vector<int> parent;
    std::vector<unsigned char> open;
    std::vector<unsigned char> closed;
    std::vector<int> path;
    Metrics metrics;
};

struct Snapshot {
    int expanded = 0;
    int current = -1;
    std::vector<unsigned char> open;
    std::vector<unsigned char> closed;
    std::vector<int> path;
    Metrics metrics;
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

__host__ __device__ static inline int node_index(int ix, int iy) {
    return iy * GRAPH_W + ix;
}

__host__ __device__ static inline float graph_x(int ix) {
    return (static_cast<float>(ix) + 0.5f) / static_cast<float>(GRAPH_W) * WORLD_W;
}

__host__ __device__ static inline float graph_y(int iy) {
    return (static_cast<float>(iy) + 0.5f) / static_cast<float>(GRAPH_H) * WORLD_H;
}

__host__ __device__ static inline int clamp_graph_x(float x) {
    return static_cast<int>(clampf(x / WORLD_W * static_cast<float>(GRAPH_W), 0.0f,
                                   static_cast<float>(GRAPH_W - 1)));
}

__host__ __device__ static inline int clamp_graph_y(float y) {
    return static_cast<int>(clampf(y / WORLD_H * static_cast<float>(GRAPH_H), 0.0f,
                                   static_cast<float>(GRAPH_H - 1)));
}

__host__ __device__ static inline void start_goal_world(int query,
                                                        float& sx,
                                                        float& sy,
                                                        float& gx,
                                                        float& gy) {
    int q = query % 8;
    if (q == 0) { sx = 0.95f; sy = 1.15f; gx = 17.15f; gy = 9.55f; return; }
    if (q == 1) { sx = 0.95f; sy = 8.95f; gx = 16.35f; gy = 1.95f; return; }
    if (q == 2) { sx = 1.70f; sy = 2.70f; gx = 13.85f; gy = 9.25f; return; }
    if (q == 3) { sx = 1.35f; sy = 9.30f; gx = 15.80f; gy = 5.80f; return; }
    if (q == 4) { sx = 3.10f; sy = 1.25f; gx = 17.00f; gy = 6.85f; return; }
    if (q == 5) { sx = 0.85f; sy = 5.55f; gx = 14.15f; gy = 9.15f; return; }
    if (q == 6) { sx = 4.05f; sy = 9.55f; gx = 16.75f; gy = 3.05f; return; }
    sx = 1.10f; sy = 3.90f; gx = 15.55f; gy = 8.30f;
}

__host__ __device__ static inline int start_node(int query) {
    float sx, sy, gx, gy;
    start_goal_world(query, sx, sy, gx, gy);
    return node_index(clamp_graph_x(sx), clamp_graph_y(sy));
}

__host__ __device__ static inline int goal_node(int query) {
    float sx, sy, gx, gy;
    start_goal_world(query, sx, sy, gx, gy);
    return node_index(clamp_graph_x(gx), clamp_graph_y(gy));
}

__host__ __device__ static inline bool is_goal_node(int idx, int query) {
    int g = goal_node(query);
    int ix = idx % GRAPH_W;
    int iy = idx / GRAPH_W;
    int gx = g % GRAPH_W;
    int gy = g / GRAPH_W;
    return abs(ix - gx) <= 1 && abs(iy - gy) <= 1;
}

__host__ __device__ static inline void edge_delta(int edge, int& dx, int& dy, float& len) {
    if (edge == 0) { dx = 1; dy = 0; len = 1.0f; return; }
    if (edge == 1) { dx = -1; dy = 0; len = 1.0f; return; }
    if (edge == 2) { dx = 0; dy = 1; len = 1.0f; return; }
    if (edge == 3) { dx = 0; dy = -1; len = 1.0f; return; }
    if (edge == 4) { dx = 1; dy = 1; len = 1.41421356f; return; }
    if (edge == 5) { dx = 1; dy = -1; len = 1.41421356f; return; }
    if (edge == 6) { dx = -1; dy = 1; len = 1.41421356f; return; }
    if (edge == 7) { dx = -1; dy = -1; len = 1.41421356f; return; }
    if (edge == 8) { dx = 2; dy = 0; len = 2.0f; return; }
    if (edge == 9) { dx = -2; dy = 0; len = 2.0f; return; }
    if (edge == 10) { dx = 0; dy = 2; len = 2.0f; return; }
    dx = 0; dy = -2; len = 2.0f;
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

__host__ __device__ static inline float line_corridor_score(float x,
                                                            float y,
                                                            float ax,
                                                            float ay,
                                                            float bx,
                                                            float by,
                                                            float width) {
    float vx = bx - ax;
    float vy = by - ay;
    float len2 = vx * vx + vy * vy;
    float t = clampf(((x - ax) * vx + (y - ay) * vy) / len2, 0.0f, 1.0f);
    float px = ax + t * vx;
    float py = ay + t * vy;
    float off = sqrtf(sqr(x - px) + sqr(y - py));
    return expf(-0.5f * sqr(off / width)) * (0.25f + 0.75f * t);
}

__host__ __device__ static inline float experience_prior(float x, float y) {
    float e0 = line_corridor_score(x, y, 0.95f, 1.15f, 17.15f, 9.55f, 1.20f);
    float e1a = line_corridor_score(x, y, 0.95f, 1.15f, 7.9f, 3.45f, 1.05f);
    float e1b = line_corridor_score(x, y, 7.9f, 3.45f, 17.15f, 9.55f, 1.05f);
    float e2a = line_corridor_score(x, y, 0.95f, 8.95f, 8.2f, 6.65f, 1.05f);
    float e2b = line_corridor_score(x, y, 8.2f, 6.65f, 16.35f, 1.95f, 1.05f);
    float e = fmaxf(e0, fmaxf(fmaxf(e1a, e1b), fmaxf(e2a, e2b)));
    return clampf(e, 0.0f, 1.0f);
}

__host__ __device__ static inline float route_prior_query(float x, float y, int query) {
    float sx, sy, gx, gy;
    start_goal_world(query, sx, sy, gx, gy);
    float route = line_corridor_score(x, y, sx, sy, gx, gy, 1.65f);
    return clampf(route, 0.0f, 1.0f);
}

__host__ __device__ static inline void moving_obstacle(int obstacle_id,
                                                       float t,
                                                       float& x,
                                                       float& y,
                                                       float& radius) {
    if (obstacle_id == 0) {
        x = 5.0f + 2.8f * sinf(0.50f * t + 0.4f);
        y = 3.0f + 0.8f * cosf(0.82f * t);
        radius = 0.44f;
        return;
    }
    if (obstacle_id == 1) {
        x = 10.2f + 0.7f * sinf(0.64f * t + 1.1f);
        y = 2.6f + 4.1f * (0.5f + 0.5f * sinf(0.46f * t));
        radius = 0.48f;
        return;
    }
    x = 12.9f + 2.1f * sinf(0.38f * t + 2.2f);
    y = 8.0f + 0.9f * cosf(0.72f * t + 0.3f);
    radius = 0.46f;
}

__host__ __device__ static inline float dynamic_edge_risk(float x, float y, float route_progress) {
    float t = clampf(12.0f * route_progress, 0.0f, 12.0f);
    float risk = 0.0f;
    for (int k = 0; k < 3; k++) {
        float ox, oy, radius;
        moving_obstacle(k, t, ox, oy, radius);
        float margin = sqrtf(sqr(x - ox) + sqr(y - oy)) - radius;
        risk += 0.65f * expf(-0.5f * sqr(margin / 0.58f))
              + 1.25f * sigmoid((0.20f - margin) * 7.0f);
    }
    return clampf(risk, 0.0f, 3.0f);
}

__host__ __device__ static inline float learned_node_cost(float rough,
                                                          float clear,
                                                          float height,
                                                          float experience,
                                                          int truth) {
    float low_clear = clampf((0.62f - clear) / 0.62f, 0.0f, 1.0f);
    float abs_h = fabsf(height);
    float height_risk = clampf((abs_h - 0.28f) / 0.42f, 0.0f, 1.0f);
    float latent0 = tanhf(1.45f * rough + 1.12f * low_clear + 0.72f * height_risk
                        - 0.95f * experience - 0.18f);
    float latent1 = tanhf(0.68f * rough - 0.86f * clear + 0.76f * abs_h
                        - 0.58f * experience + 0.25f);
    float caution = sigmoid(-0.35f + 1.38f * latent0 + 0.78f * latent1
                          + 1.12f * rough + 1.16f * low_clear
                          - 0.74f * experience);
    float cost = 0.06f + 0.92f * caution + 0.55f * rough + 0.42f * height_risk
               - 0.38f * experience;
    if (truth == 2) cost += 6.2f;
    if (truth == 1) cost += 0.74f;
    return clampf(cost, 0.05f, BLOCK_COST);
}

__host__ __device__ static inline float edge_route_progress(float x, float y, int query) {
    float sx, sy, gx, gy;
    start_goal_world(query, sx, sy, gx, gy);
    float vx = gx - sx;
    float vy = gy - sy;
    float len2 = vx * vx + vy * vy;
    return clampf(((x - sx) * vx + (y - sy) * vy) / len2, 0.0f, 1.0f);
}

__host__ __device__ static inline float learned_edge_cost(const Node* nodes,
                                                          int from,
                                                          int to,
                                                          float len_cells,
                                                          int query) {
    const Node& a = nodes[from];
    const Node& b = nodes[to];
    float mx = 0.5f * (a.x + b.x);
    float my = 0.5f * (a.y + b.y);
    float route = route_prior_query(mx, my, query);
    float experience = experience_prior(mx, my);
    float progress = edge_route_progress(mx, my, query);
    float risk = dynamic_edge_risk(mx, my, progress);
    float low_clear = clampf((0.58f - 0.5f * (a.clearance + b.clearance)) / 0.58f, 0.0f, 1.0f);
    float rough = 0.5f * (a.roughness + b.roughness);
    float slope = fabsf(b.height - a.height);
    float latent0 = tanhf(1.35f * rough + 1.20f * low_clear + 0.72f * risk
                        - 1.04f * experience - 0.42f * route + 0.10f);
    float latent1 = tanhf(0.78f * slope + 0.88f * risk - 0.64f * route
                        - 0.72f * experience + 0.18f);
    float learned = sigmoid(-0.52f + 1.18f * latent0 + 0.82f * latent1
                          + 1.05f * rough + 0.78f * low_clear + 0.82f * risk
                          - 1.05f * experience - 0.55f * route);
    float cell_size = WORLD_W / GRAPH_W;
    float dist = len_cells * cell_size;
    float cost = dist * (0.18f + 0.48f * (a.cost + b.cost) + 1.65f * learned)
               + 1.15f * risk + 0.24f * slope - 0.52f * experience - 0.22f * route;
    return clampf(cost, 0.025f, 60.0f);
}

__host__ __device__ static inline float learned_node_heuristic(const Node* nodes,
                                                               int node,
                                                               int query) {
    float sx, sy, gx, gy;
    start_goal_world(query, sx, sy, gx, gy);
    const Node& n = nodes[node];
    float dist = sqrtf(sqr(gx - n.x) + sqr(gy - n.y));
    float route = route_prior_query(n.x, n.y, query);
    float progress = edge_route_progress(n.x, n.y, query);
    float risk = dynamic_edge_risk(n.x, n.y, progress);
    float experience = experience_prior(n.x, n.y);
    float mult = clampf(0.22f + 0.34f * n.cost + 0.52f * risk
                      - 0.20f * experience - 0.14f * route, 0.12f, 1.60f);
    return dist * static_cast<float>(GRAPH_W) / WORLD_W * mult;
}

static std::vector<Node> make_nodes() {
    std::vector<Node> nodes(N_NODES);
    for (int iy = 0; iy < GRAPH_H; iy++) {
        for (int ix = 0; ix < GRAPH_W; ix++) {
            int idx = node_index(ix, iy);
            float x = graph_x(ix);
            float y = graph_y(iy);
            float h = terrain_height(x, y);
            float rough = terrain_roughness(x, y);
            float clear = clampf(raw_clearance(x, y) / 1.15f, 0.0f, 1.0f);
            int truth = terrain_truth(x, y, rough, clear, h);
            float exp_prior = experience_prior(x, y);
            float cost = learned_node_cost(rough, clear, h, exp_prior, truth);
            int blocked = (truth == 2 && cost > 7.8f && exp_prior < 0.26f) ? 1 : 0;
            for (int q = 0; q < 8; q++) {
                int s = start_node(q);
                int g = goal_node(q);
                if (abs(ix - (s % GRAPH_W)) <= 1 && abs(iy - (s / GRAPH_W)) <= 1) blocked = 0;
                if (abs(ix - (g % GRAPH_W)) <= 1 && abs(iy - (g / GRAPH_W)) <= 1) blocked = 0;
            }
            nodes[idx] = {x, y, rough, clear, h, exp_prior, cost, truth, blocked};
        }
    }
    return nodes;
}

static std::vector<int> trace_parent(const std::vector<int>& parent, int end_idx, int query) {
    std::vector<int> rev;
    if (end_idx < 0 || end_idx >= N_NODES) return rev;
    std::vector<unsigned char> used(N_NODES, 0);
    int cur = end_idx;
    int sidx = start_node(query);
    for (int steps = 0; steps < N_NODES && cur >= 0 && cur < N_NODES; steps++) {
        if (used[cur]) break;
        used[cur] = 1;
        rev.push_back(cur);
        if (cur == sidx) break;
        cur = parent[cur];
    }
    if (rev.empty() || rev.back() != sidx) return {};
    std::reverse(rev.begin(), rev.end());
    return rev;
}

static Metrics evaluate_path(const std::vector<Node>& nodes,
                             const std::vector<float>& g,
                             const std::vector<int>& path,
                             int reached,
                             int expanded,
                             int opened,
                             int goal_idx,
                             int query) {
    Metrics m;
    m.reached = reached;
    m.expanded = expanded;
    m.opened = opened;
    m.goal_idx = goal_idx;
    m.goal_cost = (goal_idx >= 0 && goal_idx < N_NODES) ? g[goal_idx] : INF_COST;
    m.path_steps = static_cast<int>(path.size());
    if (path.empty()) return m;

    float edge_sum = 0.0f;
    float risk_sum = 0.0f;
    float experience_sum = 0.0f;
    for (size_t k = 0; k < path.size(); k++) {
        const Node& n = nodes[path[k]];
        experience_sum += n.experience;
        float progress = edge_route_progress(n.x, n.y, query);
        risk_sum += dynamic_edge_risk(n.x, n.y, progress);
        if (k > 0) {
            int a = path[k - 1];
            int b = path[k];
            int dx = (b % GRAPH_W) - (a % GRAPH_W);
            int dy = (b / GRAPH_W) - (a / GRAPH_W);
            float len = sqrtf(static_cast<float>(dx * dx + dy * dy));
            edge_sum += learned_edge_cost(nodes.data(), a, b, len, query);
        }
    }
    m.path_cost = edge_sum;
    m.edge_risk = risk_sum / static_cast<float>(path.size());
    m.experience_mean = experience_sum / static_cast<float>(path.size());
    return m;
}

static int best_partial_idx(const std::vector<Node>& nodes,
                            const std::vector<float>& g,
                            const std::vector<unsigned char>& closed,
                            int query) {
    float best = INF_COST;
    int best_idx = -1;
    for (int i = 0; i < N_NODES; i++) {
        if (!closed[i]) continue;
        float score = learned_node_heuristic(nodes.data(), i, query) + 0.018f * g[i];
        if (score < best) {
            best = score;
            best_idx = i;
        }
    }
    return best_idx;
}

static SearchResult cpu_search(const std::vector<Node>& nodes,
                               int query,
                               int use_heuristic,
                               std::vector<Snapshot>* snapshots) {
    SearchResult result;
    result.g.assign(N_NODES, INF_COST);
    result.parent.assign(N_NODES, -1);
    result.open.assign(N_NODES, 0);
    result.closed.assign(N_NODES, 0);

    int sidx = start_node(query);
    result.g[sidx] = 0.0f;
    result.open[sidx] = 1;
    int opened = 1;
    int expanded = 0;
    int reached = 0;
    int goal_idx_found = -1;
    int current = sidx;

    auto maybe_snapshot = [&]() {
        if (!snapshots) return;
        int trace_idx = reached ? goal_idx_found : best_partial_idx(nodes, result.g, result.closed, query);
        std::vector<int> path = trace_parent(result.parent, trace_idx, query);
        Metrics m = evaluate_path(nodes, result.g, path, reached, expanded, opened,
                                  reached ? goal_idx_found : trace_idx, query);
        snapshots->push_back({expanded, current, result.open, result.closed, path, m});
    };

    maybe_snapshot();
    for (int iter = 0; iter < MAX_EXPANSIONS; iter++) {
        float best = INF_COST;
        int best_idx = -1;
        for (int i = 0; i < N_NODES; i++) {
            if (!result.open[i] || result.closed[i]) continue;
            float h = use_heuristic ? learned_node_heuristic(nodes.data(), i, query) : 0.0f;
            float score = result.g[i] + HEURISTIC_WEIGHT * h;
            if (score < best) {
                best = score;
                best_idx = i;
            }
        }
        if (best_idx < 0) break;

        current = best_idx;
        result.open[best_idx] = 0;
        result.closed[best_idx] = 1;
        expanded++;

        if (is_goal_node(best_idx, query)) {
            reached = 1;
            goal_idx_found = best_idx;
            if (snapshots) maybe_snapshot();
            break;
        }

        int ix = best_idx % GRAPH_W;
        int iy = best_idx / GRAPH_W;
        for (int e = 0; e < MAX_DEGREE; e++) {
            int dx, dy;
            float len;
            edge_delta(e, dx, dy, len);
            int nx = ix + dx;
            int ny = iy + dy;
            if (nx < 0 || nx >= GRAPH_W || ny < 0 || ny >= GRAPH_H) continue;
            int ni = node_index(nx, ny);
            if (nodes[ni].blocked || result.closed[ni]) continue;
            float tentative = result.g[best_idx] + learned_edge_cost(nodes.data(), best_idx, ni, len, query);
            if (tentative < result.g[ni]) {
                if (!result.open[ni]) opened++;
                result.g[ni] = tentative;
                result.parent[ni] = best_idx;
                result.open[ni] = 1;
            }
        }
        if (snapshots && (expanded % SNAP_STRIDE == 0)) maybe_snapshot();
    }

    int trace_idx = reached ? goal_idx_found : best_partial_idx(nodes, result.g, result.closed, query);
    result.path = trace_parent(result.parent, trace_idx, query);
    result.metrics = evaluate_path(nodes, result.g, result.path, reached, expanded, opened,
                                   reached ? goal_idx_found : trace_idx, query);
    if (snapshots && (snapshots->empty() || snapshots->back().expanded != expanded)) {
        snapshots->push_back({expanded, current, result.open, result.closed, result.path,
                              result.metrics});
    }
    return result;
}

static double timed_cpu_batch(const std::vector<Node>& nodes,
                              int use_heuristic,
                              std::vector<SearchResult>& results) {
    results.resize(BATCH_QUERIES);
    auto begin = std::chrono::high_resolution_clock::now();
    for (int q = 0; q < BATCH_QUERIES; q++) {
        results[q] = cpu_search(nodes, q, use_heuristic, nullptr);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

__global__ void init_search_kernel(float* __restrict__ g,
                                   int* __restrict__ parent,
                                   unsigned char* __restrict__ open,
                                   unsigned char* __restrict__ closed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BATCH_QUERIES * N_NODES;
    if (idx >= total) return;
    int q = idx / N_NODES;
    int node = idx % N_NODES;
    g[idx] = INF_COST;
    parent[idx] = -1;
    open[idx] = 0;
    closed[idx] = 0;
    if (node == start_node(q)) {
        g[idx] = 0.0f;
        open[idx] = 1;
    }
}

__global__ void graph_astar_kernel(const Node* __restrict__ nodes,
                                   int use_heuristic,
                                   float* __restrict__ g,
                                   int* __restrict__ parent,
                                   unsigned char* __restrict__ open,
                                   unsigned char* __restrict__ closed,
                                   int* __restrict__ stats) {
    __shared__ float best_score[THREADS];
    __shared__ int best_index[THREADS];
    __shared__ int done;
    __shared__ int current;
    __shared__ int expanded;
    __shared__ int opened;
    __shared__ int reached;

    int tid = threadIdx.x;
    int q = blockIdx.x;
    int base = q * N_NODES;
    int stats_base = q * 4;
    if (tid == 0) {
        done = 0;
        current = start_node(q);
        expanded = 0;
        opened = 1;
        reached = 0;
    }
    __syncthreads();

    for (int iter = 0; iter < MAX_EXPANSIONS; iter++) {
        float local_best = INF_COST;
        int local_idx = -1;
        for (int idx = tid; idx < N_NODES; idx += blockDim.x) {
            int gi = base + idx;
            if (open[gi] && !closed[gi]) {
                float h = use_heuristic ? learned_node_heuristic(nodes, idx, q) : 0.0f;
                float score = g[gi] + HEURISTIC_WEIGHT * h;
                if (score < local_best) {
                    local_best = score;
                    local_idx = idx;
                }
            }
        }
        best_score[tid] = local_best;
        best_index[tid] = local_idx;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float rhs = best_score[tid + stride];
                int rhs_idx = best_index[tid + stride];
                if (rhs < best_score[tid]) {
                    best_score[tid] = rhs;
                    best_index[tid] = rhs_idx;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            current = best_index[0];
            if (current < 0) {
                done = 1;
            } else {
                open[base + current] = 0;
                closed[base + current] = 1;
                expanded++;
                if (is_goal_node(current, q)) {
                    reached = 1;
                    done = 1;
                }
            }
        }
        __syncthreads();
        if (done) break;

        if (tid < MAX_DEGREE) {
            int ix = current % GRAPH_W;
            int iy = current / GRAPH_W;
            int dx, dy;
            float len;
            edge_delta(tid, dx, dy, len);
            int nx = ix + dx;
            int ny = iy + dy;
            if (nx >= 0 && nx < GRAPH_W && ny >= 0 && ny < GRAPH_H) {
                int ni = node_index(nx, ny);
                int ngi = base + ni;
                if (!nodes[ni].blocked && !closed[ngi]) {
                    float tentative = g[base + current] + learned_edge_cost(nodes, current, ni, len, q);
                    if (tentative < g[ngi]) {
                        if (!open[ngi]) atomicAdd(&opened, 1);
                        g[ngi] = tentative;
                        parent[ngi] = current;
                        open[ngi] = 1;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        stats[stats_base + 0] = expanded;
        stats[stats_base + 1] = opened;
        stats[stats_base + 2] = reached;
        stats[stats_base + 3] = current;
    }
}

static cv::Point to_px(const Node& n, int x0) {
    int px = x0 + static_cast<int>(n.x / WORLD_W * static_cast<float>(HALF_W - 1));
    int py = HEADER_H + static_cast<int>((1.0f - n.y / WORLD_H) * static_cast<float>(MAP_H - 1));
    return cv::Point(px, py);
}

static cv::Scalar blend(cv::Scalar a, cv::Scalar b, float wb) {
    float wa = 1.0f - wb;
    return cv::Scalar(wa * a[0] + wb * b[0],
                      wa * a[1] + wb * b[1],
                      wa * a[2] + wb * b[2]);
}

static cv::Scalar terrain_color(const Node& n) {
    if (n.blocked) return cv::Scalar(42, 34, 106);
    float c = clampf(n.cost / 5.0f, 0.0f, 1.0f);
    cv::Scalar base(42 + 40 * c, 76 + 95 * (1.0f - c), 55 + 170 * c);
    cv::Scalar expc(52, 142 + 58 * n.experience, 118 + 80 * n.experience);
    return blend(base, expc, 0.28f * n.experience);
}

static cv::Scalar edge_color(const std::vector<Node>& nodes, int a, int b, int query) {
    float len = sqrtf(static_cast<float>(sqr((b % GRAPH_W) - (a % GRAPH_W))
                                       + sqr((b / GRAPH_W) - (a / GRAPH_W))));
    float cost = learned_edge_cost(nodes.data(), a, b, len, query);
    float e = experience_prior(0.5f * (nodes[a].x + nodes[b].x),
                               0.5f * (nodes[a].y + nodes[b].y));
    float t = clampf(cost / 8.0f, 0.0f, 1.0f);
    cv::Scalar costly(56 + 125 * t, 86 + 64 * (1.0f - t), 88 + 110 * t);
    cv::Scalar experienced(72, 190, 172);
    return blend(costly, experienced, 0.45f * e);
}

static void draw_graph_background(cv::Mat& img,
                                  const std::vector<Node>& nodes,
                                  int x0,
                                  int query,
                                  int draw_edges) {
    int cw = std::max(1, HALF_W / GRAPH_W + 1);
    int ch = std::max(1, MAP_H / GRAPH_H + 1);
    for (int iy = 0; iy < GRAPH_H; iy++) {
        for (int ix = 0; ix < GRAPH_W; ix++) {
            int idx = node_index(ix, iy);
            cv::Point p = to_px(nodes[idx], x0);
            cv::rectangle(img, cv::Rect(p.x - cw / 2, p.y - ch / 2, cw, ch),
                          terrain_color(nodes[idx]), cv::FILLED);
        }
    }
    if (!draw_edges) return;
    for (int iy = 0; iy < GRAPH_H; iy++) {
        for (int ix = 0; ix < GRAPH_W; ix++) {
            int a = node_index(ix, iy);
            if (nodes[a].blocked) continue;
            for (int e = 0; e < MAX_DEGREE; e++) {
                int dx, dy;
                float len;
                edge_delta(e, dx, dy, len);
                if (dx < 0 || dy < 0) continue;
                int nx = ix + dx;
                int ny = iy + dy;
                if (nx < 0 || nx >= GRAPH_W || ny < 0 || ny >= GRAPH_H) continue;
                int b = node_index(nx, ny);
                if (nodes[b].blocked) continue;
                cv::line(img, to_px(nodes[a], x0), to_px(nodes[b], x0),
                         edge_color(nodes, a, b, query), 1, cv::LINE_AA);
            }
        }
    }
}

static void draw_search_graph(cv::Mat& img,
                              const std::vector<Node>& nodes,
                              const Snapshot& snap,
                              int x0,
                              int query) {
    draw_graph_background(img, nodes, x0, query, 0);
    for (int i = 0; i < N_NODES; i++) {
        if (nodes[i].blocked) continue;
        cv::Point p = to_px(nodes[i], x0);
        if (snap.closed[i]) {
            cv::circle(img, p, 2, cv::Scalar(94, 83, 154), cv::FILLED, cv::LINE_AA);
        } else if (snap.open[i]) {
            cv::circle(img, p, 2, cv::Scalar(84, 166, 230), cv::FILLED, cv::LINE_AA);
        }
    }
    for (size_t k = 1; k < snap.path.size(); k++) {
        int a = snap.path[k - 1];
        int b = snap.path[k];
        cv::line(img, to_px(nodes[a], x0), to_px(nodes[b], x0),
                 cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
        cv::line(img, to_px(nodes[a], x0), to_px(nodes[b], x0),
                 cv::Scalar(86, 230, 205), 1, cv::LINE_AA);
    }
    if (snap.current >= 0) {
        cv::circle(img, to_px(nodes[snap.current], x0), 5,
                   cv::Scalar(255, 245, 125), cv::FILLED, cv::LINE_AA);
    }
}

static void draw_markers(cv::Mat& img,
                         const std::vector<Node>& nodes,
                         int x0,
                         int query) {
    int s = start_node(query);
    int g = goal_node(query);
    cv::circle(img, to_px(nodes[s], x0), 7, cv::Scalar(255, 245, 125),
               cv::FILLED, cv::LINE_AA);
    cv::circle(img, to_px(nodes[g], x0), 8, cv::Scalar(245, 120, 255),
               2, cv::LINE_AA);
}

static void draw_legend(cv::Mat& img) {
    constexpr int x0 = PANEL_W - 246;
    constexpr int y0 = HEADER_H + 48;
    cv::rectangle(img, cv::Rect(x0, y0, 226, 118), cv::Scalar(8, 10, 13), cv::FILLED);
    cv::line(img, cv::Point(x0 + 14, y0 + 20), cv::Point(x0 + 32, y0 + 20),
             cv::Scalar(72, 190, 172), 2, cv::LINE_AA);
    cv::putText(img, "experience edge", cv::Point(x0 + 40, y0 + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::circle(img, cv::Point(x0 + 23, y0 + 48), 4, cv::Scalar(84, 166, 230), cv::FILLED);
    cv::putText(img, "open node", cv::Point(x0 + 40, y0 + 53),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::circle(img, cv::Point(x0 + 23, y0 + 74), 4, cv::Scalar(94, 83, 154), cv::FILLED);
    cv::putText(img, "expanded", cv::Point(x0 + 40, y0 + 79),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::line(img, cv::Point(x0 + 14, y0 + 102), cv::Point(x0 + 32, y0 + 102),
             cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
    cv::putText(img, "selected route", cv::Point(x0 + 40, y0 + 107),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
}

static cv::Mat draw_frame(const std::vector<Node>& nodes,
                          const Snapshot& snap,
                          const Metrics& dijkstra_metrics,
                          double gpu_ms,
                          double cpu_seq_ms,
                          int query) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 20, 24));
    draw_graph_background(img, nodes, 0, query, 1);
    draw_search_graph(img, nodes, snap, HALF_W, query);
    draw_markers(img, nodes, 0, query);
    draw_markers(img, nodes, HALF_W, query);
    draw_legend(img);

    cv::line(img, cv::Point(HALF_W, HEADER_H), cv::Point(HALF_W, PANEL_H - FOOTER_H),
             cv::Scalar(18, 18, 20), 1, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, HEADER_H), cv::Scalar(5, 7, 10), cv::FILLED);
    cv::rectangle(img, cv::Rect(0, PANEL_H - FOOTER_H, PANEL_W, FOOTER_H),
                  cv::Scalar(5, 7, 10), cv::FILLED);

    double speedup = gpu_ms > 0.0 ? cpu_seq_ms / gpu_ms : 0.0;
    float reduction = 100.0f * (1.0f - static_cast<float>(snap.metrics.expanded)
                                      / std::max(1.0f, static_cast<float>(dijkstra_metrics.expanded)));
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "GPU learned experience graph A*  %d queries x %d nodes  gpu=%.2f ms  cpu_seq=%.0f ms  %.1fx",
                  BATCH_QUERIES, N_NODES, gpu_ms, cpu_seq_ms, speedup);
    cv::putText(img, buf, cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "learned sparse experience graph", cv::Point(14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "batched graph A* frontier", cv::Point(HALF_W + 14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf),
                  "expanded %d / dijkstra %d (-%.1f%%)  steps=%d  g=%.2f  risk=%.2f  experience=%.2f",
                  snap.metrics.expanded, dijkstra_metrics.expanded, reduction,
                  snap.metrics.path_steps, snap.metrics.goal_cost, snap.metrics.edge_risk,
                  snap.metrics.experience_mean);
    cv::putText(img, buf, cv::Point(12, PANEL_H - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.49, cv::Scalar(225, 238, 245), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<Node> nodes = make_nodes();

    std::vector<SearchResult> cpu_astar;
    std::vector<SearchResult> cpu_dijkstra_batch;
    double cpu_astar_ms = timed_cpu_batch(nodes, 1, cpu_astar);
    double cpu_dijkstra_ms = timed_cpu_batch(nodes, 0, cpu_dijkstra_batch);

    Node* d_nodes = nullptr;
    float* d_g = nullptr;
    int* d_parent = nullptr;
    int* d_stats = nullptr;
    unsigned char* d_open = nullptr;
    unsigned char* d_closed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nodes, N_NODES * sizeof(Node)));
    size_t batch_nodes = static_cast<size_t>(BATCH_QUERIES) * N_NODES;
    CUDA_CHECK(cudaMalloc(&d_g, batch_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_parent, batch_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_open, batch_nodes * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_closed, batch_nodes * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_stats, BATCH_QUERIES * 4 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes.data(), N_NODES * sizeof(Node), cudaMemcpyHostToDevice));

    int blocks = static_cast<int>((batch_nodes + THREADS - 1) / THREADS);
    init_search_kernel<<<blocks, THREADS>>>(d_g, d_parent, d_open, d_closed);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    graph_astar_kernel<<<BATCH_QUERIES, THREADS>>>(d_nodes, 1, d_g, d_parent,
                                                   d_open, d_closed, d_stats);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float gpu_ms_f = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms_f, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaGetLastError());
    double gpu_ms = static_cast<double>(gpu_ms_f);

    std::vector<float> gpu_g(N_NODES);
    std::vector<int> gpu_parent(N_NODES);
    std::vector<unsigned char> gpu_open(N_NODES);
    std::vector<unsigned char> gpu_closed(N_NODES);
    std::vector<int> all_stats(BATCH_QUERIES * 4, 0);
    CUDA_CHECK(cudaMemcpy(gpu_g.data(), d_g, N_NODES * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_parent.data(), d_parent, N_NODES * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_open.data(), d_open, N_NODES * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_closed.data(), d_closed, N_NODES * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(all_stats.data(), d_stats, BATCH_QUERIES * 4 * sizeof(int),
                          cudaMemcpyDeviceToHost));

    const int visual_query = 0;
    const int* stats = all_stats.data();
    int gpu_goal_idx = stats[2] ? stats[3] : best_partial_idx(nodes, gpu_g, gpu_closed, visual_query);
    std::vector<int> gpu_path = trace_parent(gpu_parent, gpu_goal_idx, visual_query);
    Metrics gpu_metrics = evaluate_path(nodes, gpu_g, gpu_path, stats[2], stats[0],
                                        stats[1], gpu_goal_idx, visual_query);

    int reached_count = 0;
    float avg_expanded = 0.0f;
    for (int q = 0; q < BATCH_QUERIES; q++) {
        const int* qstats = all_stats.data() + q * 4;
        reached_count += qstats[2] ? 1 : 0;
        avg_expanded += static_cast<float>(qstats[0]);
    }
    avg_expanded /= static_cast<float>(BATCH_QUERIES);

    std::vector<Snapshot> snapshots;
    SearchResult visual = cpu_search(nodes, visual_query, 1, &snapshots);
    if (snapshots.empty() || snapshots.back().metrics.expanded != visual.metrics.expanded) {
        snapshots.push_back({visual.metrics.expanded, visual.metrics.goal_idx, visual.open,
                             visual.closed, visual.path, visual.metrics});
    }

    const Metrics& dijkstra_metrics = cpu_dijkstra_batch[visual_query].metrics;
    double speedup = gpu_ms > 0.0 ? cpu_astar_ms / gpu_ms : 0.0;
    float expansion_reduction = 100.0f * (1.0f - static_cast<float>(gpu_metrics.expanded)
                                                / std::max(1.0f, static_cast<float>(dijkstra_metrics.expanded)));

    std::printf("CPU graph Dijkstra batch: %.3f ms, q0 expanded %d, path cost %.3f, reached %d\n",
                cpu_dijkstra_ms, dijkstra_metrics.expanded, dijkstra_metrics.goal_cost,
                dijkstra_metrics.reached);
    std::printf("CPU learned experience graph A* batch: %.3f ms, q0 expanded %d, path cost %.3f, risk %.3f, experience %.3f, reached %d\n",
                cpu_astar_ms, visual.metrics.expanded, visual.metrics.goal_cost,
                visual.metrics.edge_risk, visual.metrics.experience_mean, visual.metrics.reached);
    std::printf("GPU learned experience graph A*: %.3f ms (%d queries x %d nodes x %d max edges, q0 expanded %d, %.1f%% fewer than graph Dijkstra, avg expanded/query %.1f, reached %d/%d, %.1fx vs CPU sequential graph A*)\n",
                gpu_ms, BATCH_QUERIES, N_NODES, MAX_DEGREE, gpu_metrics.expanded,
                expansion_reduction, avg_expanded, reached_count, BATCH_QUERIES, speedup);

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_experience_graph_neural_planner.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_experience_graph_neural_planner.avi\n");
        return 1;
    }
    for (const Snapshot& s : snapshots) {
        video.write(draw_frame(nodes, s, dijkstra_metrics, gpu_ms, cpu_astar_ms, visual_query));
    }
    Snapshot final_snap{gpu_metrics.expanded, gpu_metrics.goal_idx, gpu_open, gpu_closed,
                        gpu_path, gpu_metrics};
    for (int i = 0; i < 14; i++) {
        video.write(draw_frame(nodes, final_snap, dijkstra_metrics, gpu_ms,
                               cpu_astar_ms, visual_query));
    }
    video.release();

    avi_to_gif("gif/gpu_experience_graph_neural_planner.avi",
               "gif/gpu_experience_graph_neural_planner.gif", 8, 560);
    std::printf("GIF saved to gif/gpu_experience_graph_neural_planner.gif\n");

    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_parent));
    CUDA_CHECK(cudaFree(d_open));
    CUDA_CHECK(cudaFree(d_closed));
    CUDA_CHECK(cudaFree(d_stats));
    return 0;
}
