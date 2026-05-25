// gpu_anytime_neural_astar_traversability.cu
//
// Batched GPU anytime neural A* on learned traversability cost.
//
// A fixed-weight MLP-style heuristic predicts terrain-aware cost-to-go from
// roughness, clearance, height, route prior, and goal geometry.  Each query
// gets one CUDA block that parallelizes the open-set min-f reduction over a
// 192x128 grid, then expands the best node and relaxes its neighbours.  An
// anytime schedule starts with an aggressive heuristic for a fast first path,
// then anneals the weight to refine path cost when more time is available.
//
// Output: gif/gpu_anytime_neural_astar_traversability.gif

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

constexpr int GRID_W = 192;
constexpr int GRID_H = 128;
constexpr int N_CELLS = GRID_W * GRID_H;
constexpr int BATCH_QUERIES = 64;
constexpr int ANYTIME_PASSES = 4;
constexpr int MAX_EXPANSIONS = N_CELLS;
constexpr int SNAP_STRIDE = 96;
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
constexpr float START_X = 0.95f;
constexpr float START_Y = 1.15f;
constexpr float GOAL_X = 17.15f;
constexpr float GOAL_Y = 9.55f;
constexpr float BLOCK_COST = 18.0f;
constexpr float INF_COST = 1.0e20f;

static const float PASS_WEIGHTS[ANYTIME_PASSES] = {5.60f, 4.80f, 4.20f, 2.80f};

struct Cell {
    float x;
    float y;
    float roughness;
    float clearance;
    float height;
    float route_prior;
    float cost;
    float heuristic;
    int truth;
    int blocked;
};

struct Metrics {
    float path_cost = 0.0f;
    float path_blocked = 0.0f;
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
    int pass = 0;
    float weight = 0.0f;
    float best_goal_cost = INF_COST;
};

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline float sqr(float x) {
    return x * x;
}

__host__ __device__ static inline int index_of(int ix, int iy) {
    return iy * GRID_W + ix;
}

__host__ __device__ static inline int goal_ix() {
    return static_cast<int>(GOAL_X / WORLD_W * static_cast<float>(GRID_W));
}

__host__ __device__ static inline int goal_iy() {
    return static_cast<int>(GOAL_Y / WORLD_H * static_cast<float>(GRID_H));
}

__host__ __device__ static inline int start_ix() {
    return static_cast<int>(START_X / WORLD_W * static_cast<float>(GRID_W));
}

__host__ __device__ static inline int start_iy() {
    return static_cast<int>(START_Y / WORLD_H * static_cast<float>(GRID_H));
}

__host__ __device__ static inline bool is_goal_cell(int ix, int iy) {
    return abs(ix - goal_ix()) <= 2 && abs(iy - goal_iy()) <= 2;
}

__host__ __device__ static inline void action_delta(int a, int& dx, int& dy, float& len) {
    if (a == 0) { dx = 1; dy = 0; len = 1.0f; return; }
    if (a == 1) { dx = -1; dy = 0; len = 1.0f; return; }
    if (a == 2) { dx = 0; dy = 1; len = 1.0f; return; }
    if (a == 3) { dx = 0; dy = -1; len = 1.0f; return; }
    if (a == 4) { dx = 1; dy = 1; len = 1.41421356f; return; }
    if (a == 5) { dx = 1; dy = -1; len = 1.41421356f; return; }
    if (a == 6) { dx = -1; dy = 1; len = 1.41421356f; return; }
    dx = -1; dy = -1; len = 1.41421356f;
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

__host__ __device__ static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__host__ __device__ static inline float transition_cost(float cost_a,
                                                        float cost_b,
                                                        float len) {
    float uphill = fmaxf(cost_b - cost_a, 0.0f);
    return len * (0.08f + 0.42f * (cost_a + cost_b)) + 0.08f * uphill;
}

static float route_prior(float x, float y) {
    float route_x = GOAL_X - START_X;
    float route_y = GOAL_Y - START_Y;
    float route_len2 = route_x * route_x + route_y * route_y;
    float route_len = std::sqrt(route_len2);
    float sx = x - START_X;
    float sy = y - START_Y;
    float t = clampf((sx * route_x + sy * route_y) / route_len2, 0.0f, 1.0f);
    float px = START_X + t * route_x;
    float py = START_Y + t * route_y;
    float off = std::sqrt(sqr(x - px) + sqr(y - py));
    float dist_goal = std::sqrt(sqr(GOAL_X - x) + sqr(GOAL_Y - y));
    return std::exp(-0.5f * sqr(off / 1.85f))
         * (0.32f + 0.68f * t)
         * clampf((route_len - dist_goal + 2.0f) / route_len, 0.0f, 1.0f);
}

static float learned_traversability_cost(float rough,
                                         float clear,
                                         float height,
                                         float route,
                                         int truth) {
    float abs_h = std::fabs(height);
    float low_clear = clampf((0.62f - clear) / 0.62f, 0.0f, 1.0f);
    float height_risk = clampf((abs_h - 0.28f) / 0.42f, 0.0f, 1.0f);
    float learned_block = sigmoid(-1.10f + 2.40f * low_clear + 2.05f * rough
                                + 1.28f * height_risk - 1.05f * route);
    float learned_caution = sigmoid(-0.22f + 1.45f * rough + 1.22f * low_clear
                                  + 0.72f * height_risk - 0.58f * route);
    float cost = 0.06f + 0.74f * learned_caution + 4.60f * learned_block
               + 0.45f * rough + 0.38f * height_risk - 0.58f * route;
    if (truth == 2) cost += 5.8f;
    if (truth == 1) cost += 0.62f;
    return clampf(cost, 0.05f, BLOCK_COST);
}

static float neural_heuristic(float x,
                              float y,
                              float rough,
                              float clear,
                              float height,
                              float route,
                              float cost) {
    float dx = (GOAL_X - x) / WORLD_W;
    float dy = (GOAL_Y - y) / WORLD_H;
    float dist_world = std::sqrt(sqr(GOAL_X - x) + sqr(GOAL_Y - y));
    float dist_grid = dist_world * static_cast<float>(GRID_W) / WORLD_W;
    float low_clear = clampf((0.62f - clear) / 0.62f, 0.0f, 1.0f);
    float abs_h = std::fabs(height);
    float h0 = std::tanh(1.60f * dx + 0.74f * dy + 0.92f * rough
                       - 0.68f * clear + 0.42f * route - 0.18f);
    float h1 = std::tanh(-0.58f * dx + 1.18f * dy + 1.35f * low_clear
                       + 0.74f * abs_h - 0.52f * route + 0.08f);
    float h2 = std::tanh(0.36f * dx - 0.44f * dy + 0.95f * cost
                       - 1.16f * route + 0.34f);
    float risk = sigmoid(-0.72f + 1.28f * h0 + 1.04f * h1 + 0.66f * h2
                       + 1.34f * rough + 1.22f * low_clear + 0.76f * abs_h
                       - 1.18f * route);
    float corridor = sigmoid(1.05f * route - 0.48f * risk + 0.36f * h0 - 0.28f * h2);
    float mult = clampf(0.16f + 0.52f * risk + 0.12f * cost - 0.18f * corridor,
                        0.12f, 1.20f);
    return dist_grid * mult;
}

static std::vector<Cell> make_cells() {
    std::vector<Cell> cells(N_CELLS);
    for (int iy = 0; iy < GRID_H; iy++) {
        for (int ix = 0; ix < GRID_W; ix++) {
            int idx = index_of(ix, iy);
            float x = (static_cast<float>(ix) + 0.5f) / GRID_W * WORLD_W;
            float y = (static_cast<float>(iy) + 0.5f) / GRID_H * WORLD_H;
            float h = terrain_height(x, y);
            float rough = terrain_roughness(x, y);
            float clear = clampf(raw_clearance(x, y) / 1.15f, 0.0f, 1.0f);
            int truth = terrain_truth(x, y, rough, clear, h);
            float route = route_prior(x, y);
            float cost = learned_traversability_cost(rough, clear, h, route, truth);
            int blocked = (truth == 2 && cost > 8.0f && route < 0.20f) ? 1 : 0;
            if (abs(ix - start_ix()) <= 2 && abs(iy - start_iy()) <= 2) blocked = 0;
            if (abs(ix - goal_ix()) <= 2 && abs(iy - goal_iy()) <= 2) blocked = 0;
            float heuristic = neural_heuristic(x, y, rough, clear, h, route, cost);
            cells[idx] = {x, y, rough, clear, h, route, cost, heuristic, truth, blocked};
        }
    }
    return cells;
}

static std::vector<int> trace_parent(const std::vector<int>& parent, int end_idx) {
    std::vector<int> rev;
    if (end_idx < 0 || end_idx >= N_CELLS) return rev;
    std::vector<unsigned char> used(N_CELLS, 0);
    int cur = end_idx;
    for (int steps = 0; steps < N_CELLS && cur >= 0 && cur < N_CELLS; steps++) {
        if (used[cur]) break;
        used[cur] = 1;
        rev.push_back(cur);
        int sidx = index_of(start_ix(), start_iy());
        if (cur == sidx) break;
        cur = parent[cur];
    }
    if (rev.empty() || rev.back() != index_of(start_ix(), start_iy())) return {};
    std::reverse(rev.begin(), rev.end());
    return rev;
}

static Metrics evaluate_path(const std::vector<Cell>& cells,
                             const std::vector<float>& g,
                             const std::vector<int>& path,
                             int reached,
                             int expanded,
                             int opened,
                             int goal_idx) {
    Metrics m;
    m.reached = reached;
    m.expanded = expanded;
    m.opened = opened;
    m.goal_idx = goal_idx;
    m.goal_cost = (goal_idx >= 0 && goal_idx < N_CELLS) ? g[goal_idx] : INF_COST;
    m.path_steps = static_cast<int>(path.size());
    float cost_sum = 0.0f;
    float blocked_sum = 0.0f;
    for (int idx : path) {
        cost_sum += cells[idx].cost;
        blocked_sum += cells[idx].blocked ? 1.0f : 0.0f;
    }
    m.path_cost = path.empty() ? 0.0f : cost_sum / static_cast<float>(path.size());
    m.path_blocked = path.empty() ? 0.0f : blocked_sum / static_cast<float>(path.size());
    return m;
}

static int best_partial_idx(const std::vector<float>& g,
                            const std::vector<unsigned char>& closed,
                            const std::vector<float>& heuristic) {
    float best = INF_COST;
    int best_idx = -1;
    for (int i = 0; i < N_CELLS; i++) {
        if (!closed[i]) continue;
        float score = heuristic[i] + 0.008f * g[i];
        if (score < best) {
            best = score;
            best_idx = i;
        }
    }
    return best_idx;
}

static SearchResult cpu_search(const std::vector<Cell>& cells,
                               const std::vector<float>& heuristic,
                               float heuristic_weight,
                               std::vector<Snapshot>* snapshots) {
    SearchResult result;
    result.g.assign(N_CELLS, INF_COST);
    result.parent.assign(N_CELLS, -1);
    result.open.assign(N_CELLS, 0);
    result.closed.assign(N_CELLS, 0);

    int sidx = index_of(start_ix(), start_iy());
    result.g[sidx] = 0.0f;
    result.open[sidx] = 1;
    int opened = 1;
    int expanded = 0;
    int reached = 0;
    int goal_idx_found = -1;
    int current = sidx;

    auto maybe_snapshot = [&]() {
        if (!snapshots) return;
        int trace_idx = reached ? goal_idx_found : best_partial_idx(result.g, result.closed, heuristic);
        std::vector<int> path = trace_parent(result.parent, trace_idx);
        Metrics m = evaluate_path(cells, result.g, path, reached, expanded, opened,
                                  reached ? goal_idx_found : trace_idx);
        snapshots->push_back({expanded, current, result.open, result.closed, path, m});
    };

    maybe_snapshot();
    for (int iter = 0; iter < MAX_EXPANSIONS; iter++) {
        float best = INF_COST;
        int best_idx = -1;
        for (int i = 0; i < N_CELLS; i++) {
            if (!result.open[i] || result.closed[i]) continue;
            float score = result.g[i] + heuristic_weight * heuristic[i];
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

        int ix = best_idx % GRID_W;
        int iy = best_idx / GRID_W;
        if (is_goal_cell(ix, iy)) {
            reached = 1;
            goal_idx_found = best_idx;
            if (snapshots) maybe_snapshot();
            break;
        }

        for (int a = 0; a < 8; a++) {
            int dx, dy;
            float len;
            action_delta(a, dx, dy, len);
            int nx = ix + dx;
            int ny = iy + dy;
            if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H) continue;
            int ni = index_of(nx, ny);
            if (cells[ni].blocked || result.closed[ni]) continue;
            float tentative = result.g[best_idx] + transition_cost(cells[best_idx].cost,
                                                                   cells[ni].cost,
                                                                   len);
            if (tentative < result.g[ni]) {
                if (!result.open[ni]) opened++;
                result.g[ni] = tentative;
                result.parent[ni] = best_idx;
                result.open[ni] = 1;
            }
        }
        if (snapshots && (expanded % SNAP_STRIDE == 0)) maybe_snapshot();
    }

    int trace_idx = reached ? goal_idx_found : best_partial_idx(result.g, result.closed, heuristic);
    result.path = trace_parent(result.parent, trace_idx);
    result.metrics = evaluate_path(cells, result.g, result.path, reached, expanded, opened,
                                   reached ? goal_idx_found : trace_idx);
    if (snapshots && (snapshots->empty() || snapshots->back().expanded != expanded)) {
        std::vector<int> path = result.path;
        snapshots->push_back({expanded, current, result.open, result.closed, path,
                              result.metrics});
    }
    return result;
}

static double timed_cpu_search(const std::vector<Cell>& cells,
                               const std::vector<float>& heuristic,
                               float heuristic_weight,
                               SearchResult& result) {
    auto begin = std::chrono::high_resolution_clock::now();
    result = cpu_search(cells, heuristic, heuristic_weight, nullptr);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

__global__ void init_search_kernel(float* __restrict__ g,
                                   int* __restrict__ parent,
                                   unsigned char* __restrict__ open,
                                   unsigned char* __restrict__ closed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BATCH_QUERIES * N_CELLS;
    if (idx >= total) return;
    int cell = idx % N_CELLS;
    g[idx] = INF_COST;
    parent[idx] = -1;
    open[idx] = 0;
    closed[idx] = 0;
    if (cell == index_of(start_ix(), start_iy())) {
        g[idx] = 0.0f;
        open[idx] = 1;
    }
}

__global__ void neural_astar_kernel(const float* __restrict__ cost,
                                    const int* __restrict__ blocked,
                                    const float* __restrict__ heuristic,
                                    float heuristic_weight,
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
    int base = q * N_CELLS;
    int stats_base = q * 4;
    if (tid == 0) {
        done = 0;
        current = index_of(start_ix(), start_iy());
        expanded = 0;
        opened = 1;
        reached = 0;
    }
    __syncthreads();

    for (int iter = 0; iter < MAX_EXPANSIONS; iter++) {
        float local_best = INF_COST;
        int local_idx = -1;
        for (int idx = tid; idx < N_CELLS; idx += blockDim.x) {
            int gi = base + idx;
            if (open[gi] && !closed[gi]) {
                float score = g[gi] + heuristic_weight * heuristic[idx];
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
                int ix = current % GRID_W;
                int iy = current / GRID_W;
                if (is_goal_cell(ix, iy)) {
                    reached = 1;
                    done = 1;
                }
            }
        }
        __syncthreads();
        if (done) break;

        if (tid < 8) {
            int ix = current % GRID_W;
            int iy = current / GRID_W;
            int dx, dy;
            float len;
            action_delta(tid, dx, dy, len);
            int nx = ix + dx;
            int ny = iy + dy;
            if (nx >= 0 && nx < GRID_W && ny >= 0 && ny < GRID_H) {
                int ni = index_of(nx, ny);
                int ngi = base + ni;
                if (!blocked[ni] && !closed[ngi]) {
                    float tentative = g[base + current] + transition_cost(cost[current], cost[ni], len);
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

static cv::Point to_px(int ix, int iy, int x0) {
    int px = x0 + static_cast<int>(static_cast<float>(ix) / (GRID_W - 1) * (HALF_W - 1));
    int py = HEADER_H + static_cast<int>((1.0f - static_cast<float>(iy) / (GRID_H - 1)) * (MAP_H - 1));
    return cv::Point(px, py);
}

static cv::Scalar truth_tint(int label) {
    if (label == 0) return cv::Scalar(37, 53, 44);
    if (label == 1) return cv::Scalar(50, 55, 36);
    return cv::Scalar(54, 36, 41);
}

static cv::Scalar cost_color(float cost, int blocked) {
    if (blocked) return cv::Scalar(44, 35, 108);
    float v = clampf(cost / 7.0f, 0.0f, 1.0f);
    return cv::Scalar(45 + 38 * v, 82 + 95 * (1.0f - v), 55 + 180 * v);
}

static cv::Scalar heuristic_color(float heuristic, float max_h) {
    float t = clampf(heuristic / fmaxf(max_h, 1.0e-6f), 0.0f, 1.0f);
    return cv::Scalar(42 + 84 * (1.0f - t), 56 + 136 * (1.0f - t), 72 + 158 * t);
}

static cv::Scalar blend(cv::Scalar a, cv::Scalar b, float wb) {
    float wa = 1.0f - wb;
    return cv::Scalar(wa * a[0] + wb * b[0],
                      wa * a[1] + wb * b[1],
                      wa * a[2] + wb * b[2]);
}

static void heuristic_range(const std::vector<Cell>& cells, float& max_h) {
    max_h = 1.0f;
    for (const Cell& c : cells) {
        if (!c.blocked) max_h = std::max(max_h, c.heuristic);
    }
}

static void draw_cost_heuristic_panel(cv::Mat& img,
                                      const std::vector<Cell>& cells,
                                      int x0) {
    float max_h;
    heuristic_range(cells, max_h);
    int cw = std::max(1, HALF_W / GRID_W + 1);
    int ch = std::max(1, MAP_H / GRID_H + 1);
    for (int iy = 0; iy < GRID_H; iy++) {
        for (int ix = 0; ix < GRID_W; ix++) {
            int idx = index_of(ix, iy);
            cv::Point p = to_px(ix, iy, x0);
            cv::Scalar base = cost_color(cells[idx].cost, cells[idx].blocked);
            cv::Scalar h = heuristic_color(cells[idx].heuristic, max_h);
            cv::Scalar c = cells[idx].blocked ? base : blend(base, h, 0.34f);
            cv::rectangle(img, cv::Rect(p.x, p.y, cw, ch), c, cv::FILLED);
        }
    }
}

static void draw_search_panel(cv::Mat& img,
                              const std::vector<Cell>& cells,
                              const Snapshot& snap,
                              int x0) {
    int cw = std::max(1, HALF_W / GRID_W + 1);
    int ch = std::max(1, MAP_H / GRID_H + 1);
    for (int iy = 0; iy < GRID_H; iy++) {
        for (int ix = 0; ix < GRID_W; ix++) {
            int idx = index_of(ix, iy);
            cv::Point p = to_px(ix, iy, x0);
            cv::Scalar c = cells[idx].blocked ? truth_tint(2) : cv::Scalar(27, 31, 37);
            if (!cells[idx].blocked && snap.closed[idx]) {
                float t = clampf(cells[idx].heuristic / 85.0f, 0.0f, 1.0f);
                c = cv::Scalar(78 + 86 * (1.0f - t), 58 + 66 * t, 42 + 122 * t);
            } else if (!cells[idx].blocked && snap.open[idx]) {
                c = cv::Scalar(84, 158, 224);
            }
            cv::rectangle(img, cv::Rect(p.x, p.y, cw, ch), c, cv::FILLED);
        }
    }

    for (size_t k = 1; k < snap.path.size(); k++) {
        int a = snap.path[k - 1];
        int b = snap.path[k];
        cv::line(img, to_px(a % GRID_W, a / GRID_W, x0),
                 to_px(b % GRID_W, b / GRID_W, x0),
                 cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
    if (snap.current >= 0) {
        cv::circle(img, to_px(snap.current % GRID_W, snap.current / GRID_W, x0),
                   4, cv::Scalar(255, 245, 125), cv::FILLED, cv::LINE_AA);
    }
}

static void draw_markers(cv::Mat& img, int x0) {
    cv::circle(img, to_px(start_ix(), start_iy(), x0), 7, cv::Scalar(255, 245, 125),
               cv::FILLED, cv::LINE_AA);
    cv::circle(img, to_px(goal_ix(), goal_iy(), x0), 8, cv::Scalar(245, 120, 255),
               2, cv::LINE_AA);
}

static void draw_legend(cv::Mat& img) {
    constexpr int x0 = PANEL_W - 246;
    constexpr int y0 = HEADER_H + 48;
    cv::rectangle(img, cv::Rect(x0, y0, 226, 108), cv::Scalar(8, 10, 13), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + 14, y0 + 16, 16, 10), cv::Scalar(84, 158, 224), cv::FILLED);
    cv::putText(img, "open set", cv::Point(x0 + 38, y0 + 26),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(x0 + 14, y0 + 42, 16, 10), cv::Scalar(142, 76, 82), cv::FILLED);
    cv::putText(img, "expanded", cv::Point(x0 + 38, y0 + 52),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(x0 + 14, y0 + 68, 16, 10), cost_color(10.0f, 1), cv::FILLED);
    cv::putText(img, "blocked", cv::Point(x0 + 38, y0 + 78),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::line(img, cv::Point(x0 + 14, y0 + 96), cv::Point(x0 + 30, y0 + 96),
             cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(img, "parent path", cv::Point(x0 + 38, y0 + 101),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
}

static cv::Mat draw_frame(const std::vector<Cell>& cells,
                          const Snapshot& snap,
                          double gpu_ms,
                          double cpu_batch_ms,
                          const Metrics& dijkstra_metrics,
                          double dijkstra_ms,
                          const Metrics& first_metrics,
                          const Metrics& final_metrics) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 20, 24));
    draw_cost_heuristic_panel(img, cells, 0);
    draw_search_panel(img, cells, snap, HALF_W);
    draw_markers(img, 0);
    draw_markers(img, HALF_W);
    draw_legend(img);

    cv::line(img, cv::Point(HALF_W, HEADER_H), cv::Point(HALF_W, PANEL_H - FOOTER_H),
             cv::Scalar(18, 18, 20), 1, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, HEADER_H), cv::Scalar(5, 7, 10), cv::FILLED);
    cv::rectangle(img, cv::Rect(0, PANEL_H - FOOTER_H, PANEL_W, FOOTER_H),
                  cv::Scalar(5, 7, 10), cv::FILLED);

    float reduction = 100.0f * (1.0f - static_cast<float>(snap.metrics.expanded)
                                      / std::max(1.0f, static_cast<float>(dijkstra_metrics.expanded)));
    double speedup = gpu_ms > 0.0 ? cpu_batch_ms / gpu_ms : 0.0;
    float refinement = 100.0f * (1.0f - final_metrics.goal_cost
                                       / fmaxf(first_metrics.goal_cost, 1.0e-6f));
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "GPU anytime neural A*  %d passes x %dx%dx%d  gpu=%.2f ms  cpu_seq=%.0f ms  %.1fx",
                  ANYTIME_PASSES, BATCH_QUERIES, GRID_W, GRID_H, gpu_ms,
                  cpu_batch_ms, speedup);
    cv::putText(img, buf, cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "learned cost + MLP heuristic", cv::Point(14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "anytime refinement frontier", cv::Point(HALF_W + 14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf),
                  "pass %d/%d  w=%.1f  expanded %d (-%.1f%%)  g=%.2f  best=%.2f  refine=%.1f%%  dijkstra cpu=%.1f ms",
                  snap.pass + 1, ANYTIME_PASSES, snap.weight, snap.metrics.expanded,
                  reduction, snap.metrics.goal_cost, snap.best_goal_cost, refinement,
                  dijkstra_ms);
    cv::putText(img, buf, cv::Point(12, PANEL_H - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.49, cv::Scalar(225, 238, 245), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<Cell> cells = make_cells();
    std::vector<float> cost(N_CELLS);
    std::vector<int> blocked(N_CELLS);
    std::vector<float> heuristic(N_CELLS);
    std::vector<float> zero_heuristic(N_CELLS, 0.0f);
    for (int i = 0; i < N_CELLS; i++) {
        cost[i] = cells[i].cost;
        blocked[i] = cells[i].blocked;
        heuristic[i] = cells[i].heuristic;
    }

    SearchResult cpu_dijkstra;
    double dijkstra_ms = timed_cpu_search(cells, zero_heuristic, 0.0f, cpu_dijkstra);
    std::vector<SearchResult> cpu_passes(ANYTIME_PASSES);
    std::vector<double> cpu_pass_ms(ANYTIME_PASSES, 0.0);
    double cpu_single_query_ms = 0.0;
    int best_cpu_pass = 0;
    for (int p = 0; p < ANYTIME_PASSES; p++) {
        cpu_pass_ms[p] = timed_cpu_search(cells, heuristic, PASS_WEIGHTS[p], cpu_passes[p]);
        cpu_single_query_ms += cpu_pass_ms[p];
        if (p == 0 || cpu_passes[p].metrics.goal_cost < cpu_passes[best_cpu_pass].metrics.goal_cost) {
            best_cpu_pass = p;
        }
    }

    float* d_cost = nullptr;
    float* d_heuristic = nullptr;
    float* d_g = nullptr;
    int* d_blocked = nullptr;
    int* d_parent = nullptr;
    int* d_stats = nullptr;
    unsigned char* d_open = nullptr;
    unsigned char* d_closed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cost, N_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_heuristic, N_CELLS * sizeof(float)));
    size_t batch_cells = static_cast<size_t>(BATCH_QUERIES) * N_CELLS;
    CUDA_CHECK(cudaMalloc(&d_g, batch_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blocked, N_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_parent, batch_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_open, batch_cells * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_closed, batch_cells * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_stats, BATCH_QUERIES * 4 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cost, cost.data(), N_CELLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_heuristic, heuristic.data(), N_CELLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_blocked, blocked.data(), N_CELLS * sizeof(int), cudaMemcpyHostToDevice));

    int blocks = static_cast<int>((batch_cells + THREADS - 1) / THREADS);
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int p = 0; p < ANYTIME_PASSES; p++) {
        init_search_kernel<<<blocks, THREADS>>>(d_g, d_parent, d_open, d_closed);
        neural_astar_kernel<<<BATCH_QUERIES, THREADS>>>(d_cost, d_blocked, d_heuristic,
                                                       PASS_WEIGHTS[p], d_g, d_parent,
                                                       d_open, d_closed, d_stats);
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float gpu_ms_f = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms_f, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaGetLastError());
    double gpu_ms = static_cast<double>(gpu_ms_f);

    std::vector<float> gpu_g(N_CELLS);
    std::vector<int> gpu_parent(N_CELLS);
    std::vector<unsigned char> gpu_open(N_CELLS);
    std::vector<unsigned char> gpu_closed(N_CELLS);
    std::vector<int> all_stats(BATCH_QUERIES * 4, 0);
    CUDA_CHECK(cudaMemcpy(gpu_g.data(), d_g, N_CELLS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_parent.data(), d_parent, N_CELLS * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_open.data(), d_open, N_CELLS * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_closed.data(), d_closed, N_CELLS * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(all_stats.data(), d_stats, BATCH_QUERIES * 4 * sizeof(int),
                          cudaMemcpyDeviceToHost));

    const int* stats = all_stats.data();
    int gpu_goal_idx = stats[2] ? stats[3] : best_partial_idx(gpu_g, gpu_closed, heuristic);
    std::vector<int> gpu_path = trace_parent(gpu_parent, gpu_goal_idx);
    Metrics gpu_metrics = evaluate_path(cells, gpu_g, gpu_path, stats[2], stats[0],
                                        stats[1], gpu_goal_idx);

    std::vector<Snapshot> snapshots;
    float best_goal_cost = INF_COST;
    for (int p = 0; p < ANYTIME_PASSES; p++) {
        const SearchResult& pass = cpu_passes[p];
        best_goal_cost = std::min(best_goal_cost, pass.metrics.goal_cost);
        snapshots.push_back({pass.metrics.expanded, pass.metrics.goal_idx, pass.open,
                             pass.closed, pass.path, pass.metrics, p, PASS_WEIGHTS[p],
                             best_goal_cost});
    }

    const Metrics& first_cpu = cpu_passes.front().metrics;
    const Metrics& best_cpu = cpu_passes[best_cpu_pass].metrics;
    double cpu_batch_ms = cpu_single_query_ms * static_cast<double>(BATCH_QUERIES);
    double speedup = gpu_ms > 0.0 ? cpu_batch_ms / gpu_ms : 0.0;
    float expansion_reduction = 100.0f * (1.0f - static_cast<float>(gpu_metrics.expanded)
                                                / std::max(1.0f, static_cast<float>(cpu_dijkstra.metrics.expanded)));
    float cost_improvement = 100.0f * (1.0f - best_cpu.goal_cost / first_cpu.goal_cost);
    std::printf("CPU Dijkstra: %.3f ms, expanded %d, path cost %.3f, reached %d\n",
                dijkstra_ms, cpu_dijkstra.metrics.expanded, cpu_dijkstra.metrics.goal_cost,
                cpu_dijkstra.metrics.reached);
    for (int p = 0; p < ANYTIME_PASSES; p++) {
        const Metrics& m = cpu_passes[p].metrics;
        std::printf("CPU pass %d w=%.1f: %.3f ms, expanded %d, path cost %.3f, reached %d\n",
                    p + 1, PASS_WEIGHTS[p], cpu_pass_ms[p], m.expanded, m.goal_cost,
                    m.reached);
    }
    std::printf("GPU anytime neural A*: %.3f ms (%d passes x %d x %dx%d queries, final expanded/query %d, %.1f%% fewer than Dijkstra, cost %.3f -> %.3f, %.1f%% refinement, %.1fx vs CPU sequential anytime, reached %d)\n",
                gpu_ms, ANYTIME_PASSES, BATCH_QUERIES, GRID_W, GRID_H,
                gpu_metrics.expanded, expansion_reduction, first_cpu.goal_cost,
                best_cpu.goal_cost, cost_improvement, speedup, gpu_metrics.reached);

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_anytime_neural_astar_traversability.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_anytime_neural_astar_traversability.avi\n");
        return 1;
    }
    for (const Snapshot& s : snapshots) {
        for (int i = 0; i < 8; i++) {
            video.write(draw_frame(cells, s, gpu_ms, cpu_batch_ms, cpu_dijkstra.metrics,
                                   dijkstra_ms, first_cpu, best_cpu));
        }
    }
    Snapshot final_snap{gpu_metrics.expanded, gpu_metrics.goal_idx, gpu_open, gpu_closed,
                        gpu_path, gpu_metrics, ANYTIME_PASSES - 1,
                        PASS_WEIGHTS[ANYTIME_PASSES - 1], best_cpu.goal_cost};
    for (int i = 0; i < 14; i++) {
        video.write(draw_frame(cells, final_snap, gpu_ms, cpu_batch_ms,
                               cpu_dijkstra.metrics, dijkstra_ms, first_cpu, best_cpu));
    }
    video.release();

    avi_to_gif("gif/gpu_anytime_neural_astar_traversability.avi",
               "gif/gpu_anytime_neural_astar_traversability.gif", 8, 560);
    std::printf("GIF saved to gif/gpu_anytime_neural_astar_traversability.gif\n");

    CUDA_CHECK(cudaFree(d_cost));
    CUDA_CHECK(cudaFree(d_heuristic));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_blocked));
    CUDA_CHECK(cudaFree(d_parent));
    CUDA_CHECK(cudaFree(d_open));
    CUDA_CHECK(cudaFree(d_closed));
    CUDA_CHECK(cudaFree(d_stats));
    return 0;
}
