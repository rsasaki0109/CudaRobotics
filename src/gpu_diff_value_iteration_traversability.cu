// gpu_diff_value_iteration_traversability.cu
//
// GPU differentiable value iteration on learned traversability cost.
//
// A learned-style traversability layer is synthesized from terrain roughness,
// clearance, height, and a weak route prior.  A soft Bellman backup then runs
// value iteration over the cost field, producing a differentiable value field
// and a planner policy that can be followed by a robot.
//
// Output: gif/gpu_diff_value_iteration_traversability.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int GRID_W = 192;
constexpr int GRID_H = 128;
constexpr int N_CELLS = GRID_W * GRID_H;
constexpr int VI_ITERS = 220;
constexpr int SNAP_STRIDE = 20;
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
constexpr float GAMMA = 1.0f;
constexpr float SOFT_TAU = 0.12f;
constexpr float BLOCK_COST = 18.0f;
constexpr float INF_VALUE = 1.0e6f;

struct Cell {
    float x;
    float y;
    float roughness;
    float clearance;
    float height;
    float route_prior;
    float cost;
    int truth;
    int blocked;
};

struct Metrics {
    float mean_cost = 0.0f;
    float path_cost = 0.0f;
    float path_blocked = 0.0f;
    float start_value = 0.0f;
    int path_steps = 0;
    int reached = 0;
};

struct Snapshot {
    int iter = 0;
    std::vector<float> values;
    std::vector<int> policy;
    std::vector<int> path;
    Metrics metrics;
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

__global__ void init_value_kernel(const float* __restrict__ cost,
                                  const int* __restrict__ blocked,
                                  float* __restrict__ values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) return;
    int ix = idx % GRID_W;
    int iy = idx / GRID_W;
    if (blocked[idx]) {
        values[idx] = INF_VALUE;
        return;
    }
    if (is_goal_cell(ix, iy)) {
        values[idx] = 0.0f;
        return;
    }
    float dx = static_cast<float>(ix - goal_ix());
    float dy = static_cast<float>(iy - goal_iy());
    values[idx] = sqrtf(dx * dx + dy * dy) * (0.20f + 0.25f * cost[idx]);
}

__global__ void soft_vi_kernel(const float* __restrict__ cost,
                               const int* __restrict__ blocked,
                               const float* __restrict__ v_in,
                               float* __restrict__ v_out,
                               int* __restrict__ policy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) return;
    int ix = idx % GRID_W;
    int iy = idx / GRID_W;
    if (blocked[idx]) {
        v_out[idx] = INF_VALUE;
        policy[idx] = -1;
        return;
    }
    if (is_goal_cell(ix, iy)) {
        v_out[idx] = 0.0f;
        policy[idx] = -1;
        return;
    }

    float vals[8];
    float min_val = INF_VALUE;
    int best_action = -1;
    for (int a = 0; a < 8; a++) {
        int dx, dy;
        float len;
        action_delta(a, dx, dy, len);
        int nx = ix + dx;
        int ny = iy + dy;
        float val = INF_VALUE;
        if (nx >= 0 && nx < GRID_W && ny >= 0 && ny < GRID_H) {
            int ni = index_of(nx, ny);
            if (!blocked[ni]) {
                float step_cost = len * (0.08f + 0.50f * (cost[idx] + cost[ni]));
                val = step_cost + GAMMA * v_in[ni];
            }
        }
        vals[a] = val;
        if (val < min_val) {
            min_val = val;
            best_action = a;
        }
    }

    if (best_action < 0 || min_val >= 0.5f * INF_VALUE) {
        v_out[idx] = INF_VALUE;
        policy[idx] = -1;
        return;
    }

    float sum_exp = 0.0f;
    int valid_count = 0;
    for (int a = 0; a < 8; a++) {
        if (vals[a] < 0.5f * INF_VALUE) {
            sum_exp += expf(-(vals[a] - min_val) / SOFT_TAU);
            valid_count++;
        }
    }
    float inv_valid = 1.0f / fmaxf(static_cast<float>(valid_count), 1.0f);
    float soft_value = min_val - SOFT_TAU * logf(fmaxf(sum_exp * inv_valid, 1.0e-8f));
    v_out[idx] = soft_value;
    policy[idx] = best_action;
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
            cells[idx] = {x, y, rough, clear, h, route, cost, truth, blocked};
        }
    }
    return cells;
}

static void init_value_host(const std::vector<float>& cost,
                            const std::vector<int>& blocked,
                            std::vector<float>& values) {
    for (int iy = 0; iy < GRID_H; iy++) {
        for (int ix = 0; ix < GRID_W; ix++) {
            int idx = index_of(ix, iy);
            if (blocked[idx]) {
                values[idx] = INF_VALUE;
            } else if (is_goal_cell(ix, iy)) {
                values[idx] = 0.0f;
            } else {
                float dx = static_cast<float>(ix - goal_ix());
                float dy = static_cast<float>(iy - goal_iy());
                values[idx] = std::sqrt(dx * dx + dy * dy) * (0.20f + 0.25f * cost[idx]);
            }
        }
    }
}

static void soft_vi_host(const std::vector<float>& cost,
                         const std::vector<int>& blocked,
                         const std::vector<float>& v_in,
                         std::vector<float>& v_out,
                         std::vector<int>& policy) {
    for (int iy = 0; iy < GRID_H; iy++) {
        for (int ix = 0; ix < GRID_W; ix++) {
            int idx = index_of(ix, iy);
            if (blocked[idx]) {
                v_out[idx] = INF_VALUE;
                policy[idx] = -1;
                continue;
            }
            if (is_goal_cell(ix, iy)) {
                v_out[idx] = 0.0f;
                policy[idx] = -1;
                continue;
            }

            float vals[8];
            float min_val = INF_VALUE;
            int best_action = -1;
            for (int a = 0; a < 8; a++) {
                int dx, dy;
                float len;
                action_delta(a, dx, dy, len);
                int nx = ix + dx;
                int ny = iy + dy;
                float val = INF_VALUE;
                if (nx >= 0 && nx < GRID_W && ny >= 0 && ny < GRID_H) {
                    int ni = index_of(nx, ny);
                    if (!blocked[ni]) {
                        float step_cost = len * (0.08f + 0.50f * (cost[idx] + cost[ni]));
                        val = step_cost + GAMMA * v_in[ni];
                    }
                }
                vals[a] = val;
                if (val < min_val) {
                    min_val = val;
                    best_action = a;
                }
            }
            if (best_action < 0 || min_val >= 0.5f * INF_VALUE) {
                v_out[idx] = INF_VALUE;
                policy[idx] = -1;
                continue;
            }
            float sum_exp = 0.0f;
            int valid_count = 0;
            for (int a = 0; a < 8; a++) {
                if (vals[a] < 0.5f * INF_VALUE) {
                    sum_exp += std::exp(-(vals[a] - min_val) / SOFT_TAU);
                    valid_count++;
                }
            }
            float inv_valid = 1.0f / std::max(static_cast<float>(valid_count), 1.0f);
            v_out[idx] = min_val - SOFT_TAU * std::log(std::max(sum_exp * inv_valid, 1.0e-8f));
            policy[idx] = best_action;
        }
    }
}

static std::vector<int> trace_path(const std::vector<Cell>& cells,
                                   const std::vector<float>& values) {
    std::vector<int> path;
    std::vector<unsigned char> used(N_CELLS, 0);
    int ix = start_ix();
    int iy = start_iy();
    for (int step = 0; step < 620; step++) {
        int idx = index_of(ix, iy);
        path.push_back(idx);
        used[idx] = 1;
        if (is_goal_cell(ix, iy)) break;

        float best = INF_VALUE;
        int best_x = ix;
        int best_y = iy;
        for (int a = 0; a < 8; a++) {
            int dx, dy;
            float len;
            action_delta(a, dx, dy, len);
            int nx = ix + dx;
            int ny = iy + dy;
            if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H) continue;
            int ni = index_of(nx, ny);
            if (used[ni] || cells[ni].blocked) continue;
            float progress = values[idx] - values[ni];
            float val = len * (0.08f + 0.50f * (cells[idx].cost + cells[ni].cost))
                      + values[ni] - 0.12f * progress - 0.18f * cells[ni].route_prior;
            if (val < best) {
                best = val;
                best_x = nx;
                best_y = ny;
            }
        }
        if (best_x == ix && best_y == iy) break;
        ix = best_x;
        iy = best_y;
    }
    return path;
}

static Metrics evaluate(const std::vector<Cell>& cells,
                        const std::vector<float>& values) {
    Metrics m;
    int passable = 0;
    for (const Cell& c : cells) {
        m.mean_cost += c.cost;
        if (!c.blocked) passable++;
    }
    m.mean_cost /= static_cast<float>(N_CELLS);
    int sidx = index_of(start_ix(), start_iy());
    m.start_value = values[sidx];

    std::vector<int> path = trace_path(cells, values);
    float cost_sum = 0.0f;
    float blocked_sum = 0.0f;
    for (int idx : path) {
        cost_sum += cells[idx].cost;
        blocked_sum += cells[idx].blocked ? 1.0f : 0.0f;
    }
    m.path_steps = static_cast<int>(path.size());
    m.path_cost = path.empty() ? 0.0f : cost_sum / static_cast<float>(path.size());
    m.path_blocked = path.empty() ? 0.0f : blocked_sum / static_cast<float>(path.size());
    m.reached = (!path.empty() && is_goal_cell(path.back() % GRID_W, path.back() / GRID_W)) ? 1 : 0;
    (void)passable;
    return m;
}

static double cpu_soft_vi_ms(const std::vector<float>& cost,
                             const std::vector<int>& blocked,
                             const std::vector<Cell>& cells,
                             Metrics& metrics) {
    std::vector<float> a(N_CELLS);
    std::vector<float> b(N_CELLS);
    std::vector<int> policy(N_CELLS);
    init_value_host(cost, blocked, a);
    std::vector<float>* in = &a;
    std::vector<float>* out = &b;
    auto begin = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < VI_ITERS; iter++) {
        soft_vi_host(cost, blocked, *in, *out, policy);
        std::swap(in, out);
    }
    auto end = std::chrono::high_resolution_clock::now();
    metrics = evaluate(cells, *in);
    return std::chrono::duration<double, std::milli>(end - begin).count();
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

static cv::Scalar value_color(float v, float lo, float hi) {
    if (v >= 0.5f * INF_VALUE) return cv::Scalar(38, 35, 45);
    float t = clampf((v - lo) / fmaxf(hi - lo, 1.0e-6f), 0.0f, 1.0f);
    float good = 1.0f - t;
    return cv::Scalar(58 + 150 * good, 68 + 118 * good, 86 + 92 * t);
}

static void value_range(const std::vector<float>& values, float& lo, float& hi) {
    std::vector<float> finite;
    finite.reserve(N_CELLS);
    for (float v : values) {
        if (v < 0.5f * INF_VALUE) finite.push_back(v);
    }
    if (finite.empty()) {
        lo = 0.0f;
        hi = 1.0f;
        return;
    }
    size_t i5 = finite.size() / 20;
    size_t i95 = finite.size() * 19 / 20;
    std::nth_element(finite.begin(), finite.begin() + i5, finite.end());
    lo = finite[i5];
    std::nth_element(finite.begin(), finite.begin() + i95, finite.end());
    hi = finite[i95];
}

static void draw_cost_panel(cv::Mat& img, const std::vector<Cell>& cells, int x0) {
    int cw = std::max(1, HALF_W / GRID_W + 1);
    int ch = std::max(1, MAP_H / GRID_H + 1);
    for (int iy = 0; iy < GRID_H; iy++) {
        for (int ix = 0; ix < GRID_W; ix++) {
            int idx = index_of(ix, iy);
            cv::Point p = to_px(ix, iy, x0);
            cv::Scalar c = cost_color(cells[idx].cost, cells[idx].blocked);
            cv::rectangle(img, cv::Rect(p.x, p.y, cw, ch), c, cv::FILLED);
        }
    }
}

static void draw_value_panel(cv::Mat& img,
                             const std::vector<Cell>& cells,
                             const std::vector<float>& values,
                             const std::vector<int>& policy,
                             const std::vector<int>& path,
                             int x0) {
    float lo, hi;
    value_range(values, lo, hi);
    int cw = std::max(1, HALF_W / GRID_W + 1);
    int ch = std::max(1, MAP_H / GRID_H + 1);
    for (int iy = 0; iy < GRID_H; iy++) {
        for (int ix = 0; ix < GRID_W; ix++) {
            int idx = index_of(ix, iy);
            cv::Point p = to_px(ix, iy, x0);
            cv::Scalar base = cells[idx].blocked ? truth_tint(2) : value_color(values[idx], lo, hi);
            cv::rectangle(img, cv::Rect(p.x, p.y, cw, ch), base, cv::FILLED);
        }
    }

    for (int iy = 6; iy < GRID_H; iy += 9) {
        for (int ix = 6; ix < GRID_W; ix += 9) {
            int idx = index_of(ix, iy);
            if (cells[idx].blocked || policy[idx] < 0) continue;
            int dx, dy;
            float len;
            action_delta(policy[idx], dx, dy, len);
            cv::Point p = to_px(ix, iy, x0);
            cv::Point q = to_px(clampf(ix + 3 * dx, 0, GRID_W - 1),
                                clampf(iy + 3 * dy, 0, GRID_H - 1), x0);
            cv::arrowedLine(img, p, q, cv::Scalar(235, 235, 235), 1, cv::LINE_AA, 0, 0.26);
        }
    }

    for (size_t k = 1; k < path.size(); k++) {
        int a = path[k - 1];
        int b = path[k];
        cv::line(img, to_px(a % GRID_W, a / GRID_W, x0),
                 to_px(b % GRID_W, b / GRID_W, x0),
                 cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
}

static void draw_markers(cv::Mat& img, int x0) {
    cv::circle(img, to_px(start_ix(), start_iy(), x0), 7, cv::Scalar(255, 245, 125),
               cv::FILLED, cv::LINE_AA);
    cv::circle(img, to_px(goal_ix(), goal_iy(), x0), 8, cv::Scalar(245, 120, 255),
               2, cv::LINE_AA);
}

static void draw_legend(cv::Mat& img) {
    constexpr int x0 = PANEL_W - 250;
    constexpr int y0 = HEADER_H + 48;
    cv::rectangle(img, cv::Rect(x0, y0, 232, 86), cv::Scalar(8, 10, 13), cv::FILLED);
    cv::rectangle(img, cv::Rect(x0 + 16, y0 + 16, 16, 10), cost_color(0.2f, 0), cv::FILLED);
    cv::putText(img, "low cost / low value", cv::Point(x0 + 40, y0 + 26),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(x0 + 16, y0 + 42, 16, 10), cost_color(6.8f, 0), cv::FILLED);
    cv::putText(img, "rough / caution", cv::Point(x0 + 40, y0 + 52),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(x0 + 16, y0 + 66, 16, 10), cost_color(10.0f, 1), cv::FILLED);
    cv::putText(img, "blocked", cv::Point(x0 + 40, y0 + 76),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
}

static cv::Mat draw_frame(const std::vector<Cell>& cells,
                          const Snapshot& snap,
                          double gpu_ms,
                          double cpu_ms,
                          const Metrics& cpu_metrics) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 20, 24));
    draw_cost_panel(img, cells, 0);
    draw_value_panel(img, cells, snap.values, snap.policy, snap.path, HALF_W);
    draw_markers(img, 0);
    draw_markers(img, HALF_W);
    draw_legend(img);

    cv::line(img, cv::Point(HALF_W, HEADER_H), cv::Point(HALF_W, PANEL_H - FOOTER_H),
             cv::Scalar(18, 18, 20), 1, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, HEADER_H), cv::Scalar(5, 7, 10), cv::FILLED);
    cv::rectangle(img, cv::Rect(0, PANEL_H - FOOTER_H, PANEL_W, FOOTER_H),
                  cv::Scalar(5, 7, 10), cv::FILLED);

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "GPU soft value iteration traversability  grid=%dx%d  iters=%d  gpu=%.2f ms  cpu=%.1f ms",
                  GRID_W, GRID_H, VI_ITERS, gpu_ms, cpu_ms);
    cv::putText(img, buf, cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "learned traversability cost", cv::Point(14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "soft value / policy route", cv::Point(HALF_W + 14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf),
                  "iter %03d  start V=%.2f  path steps=%d  path cost=%.2f  reached=%d  cpu path cost=%.2f",
                  snap.iter, snap.metrics.start_value, snap.metrics.path_steps,
                  snap.metrics.path_cost, snap.metrics.reached, cpu_metrics.path_cost);
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
    for (int i = 0; i < N_CELLS; i++) {
        cost[i] = cells[i].cost;
        blocked[i] = cells[i].blocked;
    }

    Metrics cpu_metrics;
    double cpu_ms = cpu_soft_vi_ms(cost, blocked, cells, cpu_metrics);

    float* d_cost = nullptr;
    int* d_blocked = nullptr;
    float* d_a = nullptr;
    float* d_b = nullptr;
    int* d_policy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cost, N_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blocked, N_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_a, N_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_policy, N_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cost, cost.data(), N_CELLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_blocked, blocked.data(), N_CELLS * sizeof(int), cudaMemcpyHostToDevice));

    int blocks = (N_CELLS + THREADS - 1) / THREADS;
    init_value_kernel<<<blocks, THREADS>>>(d_cost, d_blocked, d_a);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    float* in = d_a;
    float* out = d_b;
    for (int iter = 0; iter < VI_ITERS; iter++) {
        soft_vi_kernel<<<blocks, THREADS>>>(d_cost, d_blocked, in, out, d_policy);
        float* tmp = in;
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
    std::vector<float> h_values(N_CELLS);
    std::vector<int> h_policy(N_CELLS);
    init_value_kernel<<<blocks, THREADS>>>(d_cost, d_blocked, d_a);
    CUDA_CHECK(cudaDeviceSynchronize());
    in = d_a;
    out = d_b;
    CUDA_CHECK(cudaMemcpy(h_values.data(), in, N_CELLS * sizeof(float), cudaMemcpyDeviceToHost));
    std::fill(h_policy.begin(), h_policy.end(), -1);
    snapshots.push_back({0, h_values, h_policy, trace_path(cells, h_values),
                         evaluate(cells, h_values)});
    for (int iter = 1; iter <= VI_ITERS; iter++) {
        soft_vi_kernel<<<blocks, THREADS>>>(d_cost, d_blocked, in, out, d_policy);
        CUDA_CHECK(cudaDeviceSynchronize());
        float* tmp = in;
        in = out;
        out = tmp;
        if (iter % SNAP_STRIDE == 0 || iter == VI_ITERS) {
            CUDA_CHECK(cudaMemcpy(h_values.data(), in, N_CELLS * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_policy.data(), d_policy, N_CELLS * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            snapshots.push_back({iter, h_values, h_policy, trace_path(cells, h_values),
                                 evaluate(cells, h_values)});
        }
    }

    double speedup = cpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0;
    const Metrics& final_gpu = snapshots.back().metrics;
    std::printf("CPU soft value iteration: %.3f ms, path cost %.3f, reached %d\n",
                cpu_ms, cpu_metrics.path_cost, cpu_metrics.reached);
    std::printf("GPU soft value iteration: %.3f ms (%dx%d grid x %d iters, %.1fx vs CPU, path cost %.3f, steps %d, reached %d)\n",
                gpu_ms, GRID_W, GRID_H, VI_ITERS, speedup, final_gpu.path_cost,
                final_gpu.path_steps, final_gpu.reached);

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_diff_value_iteration_traversability.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_diff_value_iteration_traversability.avi\n");
        return 1;
    }
    for (const Snapshot& s : snapshots) {
        video.write(draw_frame(cells, s, gpu_ms, cpu_ms, cpu_metrics));
    }
    for (int i = 0; i < 14; i++) {
        video.write(draw_frame(cells, snapshots.back(), gpu_ms, cpu_ms, cpu_metrics));
    }
    video.release();

    avi_to_gif("gif/gpu_diff_value_iteration_traversability.avi",
               "gif/gpu_diff_value_iteration_traversability.gif", 8, 560);
    std::printf("GIF saved to gif/gpu_diff_value_iteration_traversability.gif\n");

    CUDA_CHECK(cudaFree(d_cost));
    CUDA_CHECK(cudaFree(d_blocked));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_policy));
    return 0;
}
