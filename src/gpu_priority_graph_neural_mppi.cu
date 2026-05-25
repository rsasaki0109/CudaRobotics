// gpu_priority_graph_neural_mppi.cu
//
// GPU priority-aware graph-neural MPPI planner.
//
// A batch of robots crosses a shared interaction zone.  The baseline uses the
// coordinated graph-neural MPPI from the multi-agent demo.  The priority-aware
// variant adds learned right-of-way messages: high-priority robots keep goal
// progress through the yield zone, while lower-priority robots briefly slow and
// re-accelerate after the crossing conflict clears.
//
// Output: gif/gpu_priority_graph_neural_mppi.gif

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

constexpr int N_ROBOTS = 48;
constexpr int ROLLOUTS_PER_ROBOT = 768;
constexpr int HORIZON = 72;
constexpr int SAMPLE_ROBOTS = 12;
constexpr int SAMPLE_ROLLOUTS = 8;
constexpr int THREADS = 256;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int HEADER_H = 44;
constexpr int FOOTER_H = 52;
constexpr int MAP_H = PANEL_H - HEADER_H - FOOTER_H;
constexpr int HALF_W = PANEL_W / 2;
constexpr int VIDEO_FPS = 10;

constexpr float WORLD_W = 18.0f;
constexpr float WORLD_H = 11.0f;
constexpr float DT = 0.22f;
constexpr float PI_F = 3.14159265358979323846f;
constexpr float MIN_SPEED = 0.10f;
constexpr float MAX_SPEED = 1.38f;
constexpr float MAX_STEER = 0.66f;
constexpr float WHEEL_BASE = 1.45f;
constexpr float ROBOT_R = 0.22f;
constexpr float COLLISION_MARGIN = 0.58f;
constexpr float INF_COST = 1.0e20f;

struct RobotSpec {
    float sx;
    float sy;
    float gx;
    float gy;
    float theta0;
    float priority;
    int route;
};

struct Pose2 {
    float x;
    float y;
    float theta;
};

struct RolloutResult {
    float select_cost;
    float full_cost;
    float terminal_error;
    float mean_social_risk;
    float max_social_risk;
    float min_separation;
    float route_error;
    int robot_id;
    int rollout_id;
};

struct TeamMetrics {
    int collisions;
    int reached;
    int deadlocks;
    float min_separation;
    float mean_terminal;
    float mean_social_risk;
    float max_social_risk;
};

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline float sqr(float x) {
    return x * x;
}

__host__ __device__ static inline float fast_tanh(float x) {
    return clampf(x / (1.0f + fabsf(x)), -1.0f, 1.0f);
}

__host__ __device__ static inline float fast_sigmoid(float x) {
    return clampf(0.5f + 0.5f * fast_tanh(0.5f * x), 0.0f, 1.0f);
}

__host__ __device__ static inline float wrap_angle(float a) {
    while (a > PI_F) a -= 2.0f * PI_F;
    while (a < -PI_F) a += 2.0f * PI_F;
    return a;
}

__host__ __device__ static inline unsigned int hash_u32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__host__ __device__ static inline float hash_unit(int seed, int k, int channel) {
    unsigned int x = static_cast<unsigned int>(seed + 1) * 73856093u
                   ^ static_cast<unsigned int>(k + 3) * 19349663u
                   ^ static_cast<unsigned int>(channel + 5) * 83492791u;
    return static_cast<float>(hash_u32(x) & 0x00ffffffu) / 16777215.0f;
}

__host__ __device__ static inline float hash_signed(int seed, int k, int channel) {
    return 2.0f * hash_unit(seed, k, channel) - 1.0f;
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

__host__ __device__ static inline float static_clearance(float x, float y) {
    float d = circle_clearance(x, y, 4.2f, 3.0f, 0.82f);
    d = fminf(d, circle_clearance(x, y, 7.0f, 8.0f, 0.95f));
    d = fminf(d, circle_clearance(x, y, 11.2f, 4.4f, 1.02f));
    d = fminf(d, circle_clearance(x, y, 14.2f, 8.2f, 0.78f));
    float ridge = fabsf(y - (5.55f + 0.58f * sinf(0.58f * x))) - 0.22f;
    return fminf(d, ridge);
}

__host__ __device__ static inline float terrain_cost(float x, float y) {
    float h = 0.28f * sinf(0.45f * x + 0.22f * y)
            + 0.18f * cosf(0.62f * x - 0.35f * y);
    float rough = fabsf(h) + 0.36f * expf(-0.22f * (sqr(x - 13.5f) + sqr(y - 2.7f)));
    float clear = static_clearance(x, y);
    float cost = 0.18f + 0.82f * rough;
    if (clear < 0.0f) cost += 8.0f + 18.0f * sqr(clear);
    if (clear < 0.36f) cost += 1.7f * sqr(0.36f - clear);
    return clampf(cost, 0.05f, 18.0f);
}

__host__ __device__ static inline void route_point(const RobotSpec& r,
                                                   float progress,
                                                   float& x,
                                                   float& y) {
    float p = clampf(progress, 0.0f, 1.0f);
    float sx = r.sx;
    float sy = r.sy;
    float gx = r.gx;
    float gy = r.gy;
    float cx = 8.9f + 0.35f * sinf(0.91f * static_cast<float>(r.route + 1));
    float cy = 5.55f + 0.30f * cosf(0.73f * static_cast<float>(r.route + 2));
    float q0 = (1.0f - p) * (1.0f - p);
    float q1 = 2.0f * (1.0f - p) * p;
    float q2 = p * p;
    x = q0 * sx + q1 * cx + q2 * gx;
    y = q0 * sy + q1 * cy + q2 * gy;
}

__host__ __device__ static inline void nominal_pose(const RobotSpec& r,
                                                    int step,
                                                    float& x,
                                                    float& y) {
    float progress = static_cast<float>(step) / static_cast<float>(HORIZON - 1);
    route_point(r, progress, x, y);
}

__host__ __device__ static inline float route_distance(const RobotSpec& r,
                                                       float x,
                                                       float y,
                                                       float* progress_out) {
    float best = 1.0e9f;
    float best_p = 0.0f;
    float prev_x;
    float prev_y;
    route_point(r, 0.0f, prev_x, prev_y);
    constexpr int SEGMENTS = 18;
    for (int i = 1; i <= SEGMENTS; i++) {
        float p1 = static_cast<float>(i) / static_cast<float>(SEGMENTS);
        float bx;
        float by;
        route_point(r, p1, bx, by);
        float vx = bx - prev_x;
        float vy = by - prev_y;
        float len2 = fmaxf(vx * vx + vy * vy, 1.0e-6f);
        float u = clampf(((x - prev_x) * vx + (y - prev_y) * vy) / len2, 0.0f, 1.0f);
        float px = prev_x + u * vx;
        float py = prev_y + u * vy;
        float d = sqrtf(sqr(x - px) + sqr(y - py));
        if (d < best) {
            best = d;
            best_p = (static_cast<float>(i - 1) + u) / static_cast<float>(SEGMENTS);
        }
        prev_x = bx;
        prev_y = by;
    }
    if (progress_out) *progress_out = best_p;
    return best;
}

__host__ __device__ static inline void graph_message(const RobotSpec* robots,
                                                     int robot_id,
                                                     float x,
                                                     float y,
                                                     int step,
                                                     float& mx,
                                                     float& my,
                                                     float& risk,
                                                     float& min_sep) {
    mx = 0.0f;
    my = 0.0f;
    risk = 0.0f;
    min_sep = 1.0e6f;
    const RobotSpec& self = robots[robot_id];
    for (int j = 0; j < N_ROBOTS; j++) {
        if (j == robot_id) continue;
        float px;
        float py;
        nominal_pose(robots[j], step, px, py);
        float dx = x - px;
        float dy = y - py;
        float d2 = dx * dx + dy * dy;
        float d = sqrtf(fmaxf(d2, 1.0e-8f));
        float sep = d - 2.0f * ROBOT_R;
        min_sep = fminf(min_sep, sep);
        if (d > 3.1f) continue;
        float closing = 1.0f - clampf(d / 3.1f, 0.0f, 1.0f);
        float crossing = (self.route == robots[j].route) ? 0.38f : 1.0f;
        float priority = 0.75f + 0.50f * robots[j].priority;
        float w = crossing * priority * (0.18f * expf(-0.5f * d2 / 1.65f)
                + 1.45f * closing * closing);
        risk += w;
        float inv_d = rsqrtf(fmaxf(d2, 1.0e-6f));
        mx += w * dx * inv_d;
        my += w * dy * inv_d;
    }
    float mag = sqrtf(mx * mx + my * my);
    if (mag > 1.0e-5f) {
        float scale = 1.0f / mag;
        mx *= scale;
        my *= scale;
    }
    risk = clampf(risk, 0.0f, 6.0f);
}

__host__ __device__ static inline float route_precedence(int route) {
    if (route == 0) return 0.18f;
    if (route == 2) return 0.10f;
    if (route == 1) return 0.02f;
    return -0.05f;
}

__host__ __device__ static inline float route_green_step(int route) {
    if (route == 0) return 28.0f;
    if (route == 2) return 32.0f;
    if (route == 1) return 36.0f;
    return 40.0f;
}

__host__ __device__ static inline float yield_zone(float x, float y) {
    return expf(-0.5f * (sqr((x - 9.0f) / 3.2f) + sqr((y - 5.5f) / 2.1f)));
}

__host__ __device__ static inline void priority_signal(const RobotSpec* robots,
                                                       int robot_id,
                                                       float progress,
                                                       int step,
                                                       float& yield_pressure,
                                                       float& go_pressure,
                                                       float& zone_pressure) {
    yield_pressure = 0.0f;
    go_pressure = 0.0f;
    zone_pressure = 0.0f;
    const RobotSpec& self = robots[robot_id];
    float fx;
    float fy;
    route_point(self, progress + 0.13f, fx, fy);
    float zone = yield_zone(fx, fy);
    if (zone < 0.055f) return;
    int look_step = (step + 8 < HORIZON) ? step + 8 : HORIZON - 1;
    float self_score = self.priority + route_precedence(self.route);

    for (int j = 0; j < N_ROBOTS; j++) {
        if (j == robot_id || robots[j].route == self.route) continue;
        float px;
        float py;
        nominal_pose(robots[j], look_step, px, py);
        float dx = fx - px;
        float dy = fy - py;
        float d2 = dx * dx + dy * dy;
        float d = sqrtf(fmaxf(d2, 1.0e-8f));
        if (d > 3.85f) continue;

        float closing = 1.0f - clampf(d / 3.85f, 0.0f, 1.0f);
        float w = zone * (0.24f * expf(-0.5f * d2 / 2.45f) + 1.18f * closing * closing);
        float peer_score = robots[j].priority + route_precedence(robots[j].route);
        float delta = peer_score - self_score;
        if (delta > 0.04f) {
            yield_pressure += w * clampf(0.52f + 1.20f * delta, 0.0f, 1.35f);
        } else {
            go_pressure += w * clampf(0.36f - 0.80f * delta, 0.0f, 1.10f);
        }
        zone_pressure = fmaxf(zone_pressure, zone * closing);
    }

    yield_pressure = clampf(yield_pressure, 0.0f, 1.75f);
    go_pressure = clampf(go_pressure, 0.0f, 1.35f);
    zone_pressure = clampf(zone_pressure, 0.0f, 1.0f);
}

__host__ __device__ static inline void rollout_step(const RobotSpec* robots,
                                                    int robot_id,
                                                    int rollout,
                                                    int step,
                                                    int priority_aware,
                                                    Pose2& s,
                                                    float& prev_speed,
                                                    float& prev_steer,
                                                    float& social_risk,
                                                    float& min_sep,
                                                    float& route_error,
                                                    float& smooth,
                                                    float& terrain) {
    const RobotSpec& r = robots[robot_id];
    float progress = 0.0f;
    route_error = route_distance(r, s.x, s.y, &progress);
    float yield_pressure = 0.0f;
    float go_pressure = 0.0f;
    float zone_pressure = 0.0f;
    if (priority_aware) {
        priority_signal(robots, robot_id, progress, step,
                        yield_pressure, go_pressure, zone_pressure);
        float gx;
        float gy;
        route_point(r, progress + 0.10f, gx, gy);
        float gate_zone = yield_zone(gx, gy);
        float green = route_green_step(r.route) - 3.0f * (r.priority - 0.65f);
        float before_cross = clampf((0.66f - progress) / 0.30f, 0.0f, 1.0f);
        float early = clampf((green - static_cast<float>(step)) / 12.0f, 0.0f, 1.0f)
                    * gate_zone * before_cross;
        float released = clampf((static_cast<float>(step) - green) / 14.0f, 0.0f, 1.0f)
                       * gate_zone * before_cross;
        yield_pressure = clampf(yield_pressure + 0.40f * early, 0.0f, 1.75f);
        go_pressure = clampf(go_pressure + 0.55f * released, 0.0f, 1.35f);
        zone_pressure = fmaxf(zone_pressure, gate_zone * before_cross);
    }
    float priority_phase = 0.08f * (r.priority - 0.62f);
    if (priority_aware) {
        priority_phase = 0.04f * (r.priority - 0.62f)
                       + 0.080f * go_pressure
                       - 0.040f * yield_pressure;
    }
    float lookahead = progress + 0.14f + priority_phase + 0.04f * hash_unit(rollout, step / 6, 3);
    float tx;
    float ty;
    route_point(r, lookahead, tx, ty);

    float mx;
    float my;
    graph_message(robots, robot_id, s.x, s.y, step, mx, my, social_risk, min_sep);
    float avoid_gain = clampf(0.62f + 0.48f * social_risk, 0.0f, 2.15f);
    if (priority_aware) {
        avoid_gain = clampf(0.44f + 0.34f * social_risk
                                  + 0.62f * yield_pressure
                                  - 0.18f * go_pressure,
                            0.0f, 2.20f);
    }
    tx += avoid_gain * mx;
    ty += avoid_gain * my;

    terrain = terrain_cost(s.x, s.y);
    float dx = tx - s.x;
    float dy = ty - s.y;
    float target_heading = atan2f(dy, dx);
    float heading_error = wrap_angle(target_heading - s.theta);
    float dist = sqrtf(dx * dx + dy * dy);
    float social_slow = clampf(social_risk / 3.4f, 0.0f, 1.0f);
    if (priority_aware) {
        social_slow = clampf(0.50f * social_slow
                           + 0.50f * yield_pressure
                           - 0.35f * go_pressure
                           + 0.10f * zone_pressure,
                             0.0f, 1.0f);
    }
    float terrain_slow = clampf((terrain - 0.7f) / 4.0f, 0.0f, 1.0f);
    float base_speed = clampf(0.38f + 1.06f * fast_sigmoid(1.2f * cosf(heading_error)
                                  + 0.18f * dist - 1.15f * terrain_slow
                                  - 1.25f * social_slow),
                              MIN_SPEED, MAX_SPEED);
    if (priority_aware) {
        base_speed = clampf(base_speed * (1.04f + 0.20f * go_pressure
                                                - 0.28f * yield_pressure),
                            MIN_SPEED, MAX_SPEED);
    } else {
        base_speed = clampf(base_speed * (0.74f + 0.38f * r.priority), MIN_SPEED, MAX_SPEED);
    }
    float base_steer = clampf(0.58f * fast_tanh(1.62f * heading_error
                                  + 0.34f * social_slow * (mx * -sinf(s.theta)
                                                          + my * cosf(s.theta))),
                              -MAX_STEER, MAX_STEER);
    float n0 = 0.48f * hash_signed(rollout, step, 0)
             + 0.20f * hash_signed(rollout, step / 4, robot_id + 7);
    float n1 = 0.70f * hash_signed(rollout, step, 1)
             + 0.24f * hash_signed(rollout, step / 5, robot_id + 19);
    float speed_sigma = 0.17f;
    float steer_sigma = 0.17f;
    float speed = clampf(base_speed + speed_sigma * n0, MIN_SPEED, MAX_SPEED);
    float steer = clampf(base_steer + steer_sigma * n1, -MAX_STEER, MAX_STEER);
    smooth = 0.05f * sqr(speed - prev_speed) + 0.16f * sqr(steer - prev_steer);

    float yaw_rate = speed * steer / WHEEL_BASE;
    s.x += speed * cosf(s.theta) * DT;
    s.y += speed * sinf(s.theta) * DT;
    s.theta = wrap_angle(s.theta + yaw_rate * DT);
    s.x = clampf(s.x, 0.05f, WORLD_W - 0.05f);
    s.y = clampf(s.y, 0.05f, WORLD_H - 0.05f);
    prev_speed = speed;
    prev_steer = steer;
}

__host__ __device__ static inline RolloutResult evaluate_rollout(const RobotSpec* robots,
                                                                 int robot_id,
                                                                 int rollout,
                                                                 int priority_aware) {
    const RobotSpec& r = robots[robot_id];
    Pose2 s{r.sx, r.sy, r.theta0};
    float blind_cost = 0.0f;
    float full_cost = 0.0f;
    float risk_sum = 0.0f;
    float risk_max = 0.0f;
    float min_sep = 1.0e6f;
    float route_sum = 0.0f;
    float prev_speed = 0.70f;
    float prev_steer = 0.0f;

    for (int k = 0; k < HORIZON; k++) {
        float social_risk;
        float step_min_sep;
        float route_error;
        float smooth;
        float terrain;
        rollout_step(robots, robot_id, rollout, k, priority_aware, s, prev_speed, prev_steer,
                     social_risk, step_min_sep, route_error, smooth, terrain);
        float obstacle = terrain > 5.0f ? 8.0f * sqr(terrain - 5.0f) : 0.0f;
        float base_step = 0.62f * terrain + 1.35f * route_error * route_error
                        + 0.20f * sqr(prev_steer / MAX_STEER) + smooth + obstacle;
        float close_weight = priority_aware ? 90.0f : 82.0f;
        float social_weight = priority_aware ? 23.0f : 22.0f;
        float close_penalty = step_min_sep < COLLISION_MARGIN
                            ? close_weight * sqr(COLLISION_MARGIN - step_min_sep)
                            : 0.0f;
        float social_step = social_weight * social_risk + close_penalty;
        blind_cost += base_step;
        full_cost += base_step + social_step;
        risk_sum += social_risk;
        risk_max = fmaxf(risk_max, social_risk);
        min_sep = fminf(min_sep, step_min_sep);
        route_sum += route_error;
    }

    float terminal = sqrtf(sqr(r.gx - s.x) + sqr(r.gy - s.y));
    float final_route = route_distance(r, s.x, s.y, nullptr);
    float terminal_weight = priority_aware ? 124.0f : 118.0f;
    float terminal_cost = terminal_weight * terminal * terminal + 8.5f * final_route;
    blind_cost += terminal_cost;
    full_cost += terminal_cost;

    RolloutResult out;
    out.select_cost = full_cost;
    out.full_cost = full_cost;
    out.terminal_error = terminal;
    out.mean_social_risk = risk_sum / static_cast<float>(HORIZON);
    out.max_social_risk = risk_max;
    out.min_separation = min_sep;
    out.route_error = route_sum / static_cast<float>(HORIZON);
    out.robot_id = robot_id;
    out.rollout_id = rollout;
    return out;
}

__global__ void rollout_kernel(const RobotSpec* __restrict__ robots,
                               int priority_aware,
                               RolloutResult* __restrict__ results,
                               float* __restrict__ sample_x,
                               float* __restrict__ sample_y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N_ROBOTS * ROLLOUTS_PER_ROBOT;
    if (idx >= total) return;
    int robot_id = idx / ROLLOUTS_PER_ROBOT;
    int rollout = idx - robot_id * ROLLOUTS_PER_ROBOT;
    results[idx] = evaluate_rollout(robots, robot_id, rollout, priority_aware);

    if (!priority_aware || robot_id >= SAMPLE_ROBOTS || rollout >= SAMPLE_ROLLOUTS) return;
    Pose2 s{robots[robot_id].sx, robots[robot_id].sy, robots[robot_id].theta0};
    float prev_speed = 0.70f;
    float prev_steer = 0.0f;
    int sample_base = (robot_id * SAMPLE_ROLLOUTS + rollout) * HORIZON;
    for (int k = 0; k < HORIZON; k++) {
        float social_risk;
        float min_sep;
        float route_error;
        float smooth;
        float terrain;
        rollout_step(robots, robot_id, rollout, k, priority_aware, s, prev_speed, prev_steer,
                     social_risk, min_sep, route_error, smooth, terrain);
        sample_x[sample_base + k] = s.x;
        sample_y[sample_base + k] = s.y;
    }
}

static std::vector<RobotSpec> make_robots() {
    std::vector<RobotSpec> robots(N_ROBOTS);
    for (int i = 0; i < N_ROBOTS; i++) {
        int group = i % 4;
        int lane = i / 4;
        float lane_f = static_cast<float>(lane);
        float jitter = 0.13f * std::sin(1.37f * static_cast<float>(i));
        RobotSpec r{};
        r.route = group;
        r.priority = 0.35f + 0.65f * static_cast<float>((i * 7) % 11) / 10.0f;
        if (group == 0) {
            r.sx = 0.40f + 0.055f * lane_f;
            r.sy = 3.95f + 0.15f * lane_f + jitter;
            r.gx = 17.38f;
            r.gy = 7.05f - 0.14f * lane_f - jitter;
            r.theta0 = 0.05f;
        } else if (group == 1) {
            r.sx = 7.45f + 0.16f * lane_f + jitter;
            r.sy = 0.40f + 0.045f * lane_f;
            r.gx = 10.62f - 0.16f * lane_f - jitter;
            r.gy = 10.36f;
            r.theta0 = 1.50f;
        } else if (group == 2) {
            r.sx = 17.60f - 0.055f * lane_f;
            r.sy = 7.12f - 0.15f * lane_f + jitter;
            r.gx = 0.62f;
            r.gy = 3.98f + 0.14f * lane_f - jitter;
            r.theta0 = PI_F - 0.05f;
        } else {
            r.sx = 10.62f - 0.16f * lane_f + jitter;
            r.sy = 10.60f - 0.045f * lane_f;
            r.gx = 7.45f + 0.16f * lane_f - jitter;
            r.gy = 0.62f;
            r.theta0 = -1.58f;
        }
        robots[i] = r;
    }
    return robots;
}

static double evaluate_cpu_rollouts(const std::vector<RobotSpec>& robots,
                                    int coordinated,
                                    std::vector<RolloutResult>& out) {
    out.resize(N_ROBOTS * ROLLOUTS_PER_ROBOT);
    auto begin = std::chrono::high_resolution_clock::now();
    for (int robot = 0; robot < N_ROBOTS; robot++) {
        for (int rollout = 0; rollout < ROLLOUTS_PER_ROBOT; rollout++) {
            int idx = robot * ROLLOUTS_PER_ROBOT + rollout;
            out[idx] = evaluate_rollout(robots.data(), robot, rollout, coordinated);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

static std::vector<int> select_best_by_robot(const std::vector<RolloutResult>& results) {
    std::vector<int> best(N_ROBOTS, 0);
    std::vector<float> best_cost(N_ROBOTS, INF_COST);
    for (int robot = 0; robot < N_ROBOTS; robot++) {
        for (int rollout = 0; rollout < ROLLOUTS_PER_ROBOT; rollout++) {
            int idx = robot * ROLLOUTS_PER_ROBOT + rollout;
            if (results[idx].select_cost < best_cost[robot]) {
                best_cost[robot] = results[idx].select_cost;
                best[robot] = rollout;
            }
        }
    }
    return best;
}

static void reconstruct_one_path(const std::vector<RobotSpec>& robots,
                                 int robot,
                                 int rollout,
                                 int priority_aware,
                                 std::vector<float>& path_x,
                                 std::vector<float>& path_y) {
    path_x.assign(HORIZON, 0.0f);
    path_y.assign(HORIZON, 0.0f);
    Pose2 s{robots[robot].sx, robots[robot].sy, robots[robot].theta0};
    float prev_speed = 0.70f;
    float prev_steer = 0.0f;
    for (int k = 0; k < HORIZON; k++) {
        float social_risk;
        float min_sep;
        float route_error;
        float smooth;
        float terrain;
        rollout_step(robots.data(), robot, rollout, k, priority_aware,
                     s, prev_speed, prev_steer, social_risk, min_sep,
                     route_error, smooth, terrain);
        path_x[k] = s.x;
        path_y[k] = s.y;
    }
}

static std::vector<int> select_priority_by_robot(const std::vector<RobotSpec>& robots,
                                                 const std::vector<RolloutResult>& results) {
    std::vector<int> order(N_ROBOTS);
    for (int i = 0; i < N_ROBOTS; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        float sa = robots[a].priority + route_precedence(robots[a].route);
        float sb = robots[b].priority + route_precedence(robots[b].route);
        if (sa == sb) return a < b;
        return sa > sb;
    });

    std::vector<int> selected(N_ROBOTS, 0);
    std::vector<unsigned char> fixed(N_ROBOTS, 0);
    std::vector<float> fixed_x(N_ROBOTS * HORIZON, -100.0f);
    std::vector<float> fixed_y(N_ROBOTS * HORIZON, -100.0f);
    std::vector<float> cand_x;
    std::vector<float> cand_y;
    std::vector<float> best_x;
    std::vector<float> best_y;

    for (int robot : order) {
        float best_score = INF_COST;
        int best_rollout = 0;
        for (int rollout = 0; rollout < ROLLOUTS_PER_ROBOT; rollout++) {
            reconstruct_one_path(robots, robot, rollout, 1, cand_x, cand_y);
            float conflict = 0.0f;
            for (int peer = 0; peer < N_ROBOTS; peer++) {
                if (!fixed[peer] || robots[peer].route == robots[robot].route) continue;
                float peer_score = robots[peer].priority + route_precedence(robots[peer].route);
                float self_score = robots[robot].priority + route_precedence(robots[robot].route);
                float priority_gap = clampf(peer_score - self_score, 0.0f, 1.0f);
                for (int k = 0; k < HORIZON; k++) {
                    float dx = cand_x[k] - fixed_x[peer * HORIZON + k];
                    float dy = cand_y[k] - fixed_y[peer * HORIZON + k];
                    float sep = std::sqrt(dx * dx + dy * dy) - 2.0f * ROBOT_R;
                    if (sep < COLLISION_MARGIN) {
                        float margin = COLLISION_MARGIN - sep;
                        conflict += (280.0f + 260.0f * priority_gap) * margin * margin;
                    }
                    if (sep < 0.06f) {
                        float overlap = 0.06f - sep;
                        conflict += (2200.0f + 1800.0f * priority_gap) * overlap * overlap;
                    }
                }
            }
            int idx = robot * ROLLOUTS_PER_ROBOT + rollout;
            float score = results[idx].select_cost + conflict;
            if (score < best_score) {
                best_score = score;
                best_rollout = rollout;
                best_x = cand_x;
                best_y = cand_y;
            }
        }
        selected[robot] = best_rollout;
        fixed[robot] = 1;
        for (int k = 0; k < HORIZON; k++) {
            fixed_x[robot * HORIZON + k] = best_x[k];
            fixed_y[robot * HORIZON + k] = best_y[k];
        }
    }
    return selected;
}

static void reconstruct_paths(const std::vector<RobotSpec>& robots,
                              const std::vector<int>& selected,
                              int coordinated,
                              std::vector<float>& path_x,
                              std::vector<float>& path_y) {
    path_x.assign(N_ROBOTS * HORIZON, 0.0f);
    path_y.assign(N_ROBOTS * HORIZON, 0.0f);
    for (int robot = 0; robot < N_ROBOTS; robot++) {
        Pose2 s{robots[robot].sx, robots[robot].sy, robots[robot].theta0};
        float prev_speed = 0.70f;
        float prev_steer = 0.0f;
        for (int k = 0; k < HORIZON; k++) {
            float social_risk;
            float min_sep;
            float route_error;
            float smooth;
            float terrain;
            rollout_step(robots.data(), robot, selected[robot], k, coordinated,
                         s, prev_speed, prev_steer, social_risk, min_sep,
                         route_error, smooth, terrain);
            path_x[robot * HORIZON + k] = s.x;
            path_y[robot * HORIZON + k] = s.y;
        }
    }
}

static TeamMetrics compute_team_metrics(const std::vector<RobotSpec>& robots,
                                        const std::vector<RolloutResult>& results,
                                        const std::vector<int>& selected,
                                        const std::vector<float>& path_x,
                                        const std::vector<float>& path_y) {
    TeamMetrics m{};
    m.min_separation = 1.0e6f;
    float terminal_sum = 0.0f;
    float risk_sum = 0.0f;
    for (int robot = 0; robot < N_ROBOTS; robot++) {
        const RolloutResult& r = results[robot * ROLLOUTS_PER_ROBOT + selected[robot]];
        terminal_sum += r.terminal_error;
        risk_sum += r.mean_social_risk;
        m.max_social_risk = std::max(m.max_social_risk, r.max_social_risk);
        if (r.terminal_error < 2.25f) m.reached++;
        if (r.terminal_error > 4.0f) m.deadlocks++;
    }
    for (int k = 0; k < HORIZON; k++) {
        for (int i = 0; i < N_ROBOTS; i++) {
            float xi = path_x[i * HORIZON + k];
            float yi = path_y[i * HORIZON + k];
            for (int j = i + 1; j < N_ROBOTS; j++) {
                if (robots[i].route == robots[j].route) continue;
                float dx = xi - path_x[j * HORIZON + k];
                float dy = yi - path_y[j * HORIZON + k];
                float sep = sqrtf(dx * dx + dy * dy) - 2.0f * ROBOT_R;
                m.min_separation = std::min(m.min_separation, sep);
                if (sep < 0.06f) m.collisions++;
            }
        }
    }
    m.mean_terminal = terminal_sum / static_cast<float>(N_ROBOTS);
    m.mean_social_risk = risk_sum / static_cast<float>(N_ROBOTS);
    return m;
}

static cv::Point to_px(float x, float y, int x0) {
    int px = x0 + static_cast<int>(x / WORLD_W * static_cast<float>(HALF_W - 1));
    int py = HEADER_H + static_cast<int>((1.0f - y / WORLD_H) * static_cast<float>(MAP_H - 1));
    return cv::Point(px, py);
}

static cv::Scalar robot_color(int robot) {
    static const cv::Scalar colors[] = {
        cv::Scalar(82, 220, 255),
        cv::Scalar(120, 235, 128),
        cv::Scalar(255, 180, 82),
        cv::Scalar(226, 122, 255),
    };
    return colors[robot % 4];
}

static cv::Scalar blend(cv::Scalar a, cv::Scalar b, float wb) {
    float wa = 1.0f - wb;
    return cv::Scalar(wa * a[0] + wb * b[0],
                      wa * a[1] + wb * b[1],
                      wa * a[2] + wb * b[2]);
}

static cv::Scalar field_color(float x, float y) {
    float c = clampf(terrain_cost(x, y) / 5.0f, 0.0f, 1.0f);
    cv::Scalar base(46 + 50 * c, 118 + 82 * (1.0f - c), 66 + 146 * c);
    cv::Scalar center(60, 160, 118);
    if (static_clearance(x, y) < 0.0f) base = cv::Scalar(54, 42, 124);
    float cross = expf(-0.5f * (sqr((x - 9.2f) / 3.3f) + sqr((y - 5.5f) / 2.2f)));
    return blend(base, center, 0.22f * cross);
}

static void draw_field(cv::Mat& img, int x0) {
    constexpr int GRID_W = 160;
    constexpr int GRID_H = 100;
    int cw = std::max(2, HALF_W / GRID_W + 1);
    int ch = std::max(2, MAP_H / GRID_H + 1);
    for (int iy = 0; iy < GRID_H; iy++) {
        for (int ix = 0; ix < GRID_W; ix++) {
            float x = (static_cast<float>(ix) + 0.5f) / GRID_W * WORLD_W;
            float y = (static_cast<float>(iy) + 0.5f) / GRID_H * WORLD_H;
            cv::Point p = to_px(x, y, x0);
            cv::rectangle(img, cv::Rect(p.x, p.y, cw, ch), field_color(x, y), cv::FILLED);
        }
    }
}

static void draw_routes(cv::Mat& img, const std::vector<RobotSpec>& robots, int x0) {
    for (int robot = 0; robot < N_ROBOTS; robot += 2) {
        cv::Point prev;
        for (int i = 0; i <= 24; i++) {
            float x;
            float y;
            route_point(robots[robot], static_cast<float>(i) / 24.0f, x, y);
            cv::Point p = to_px(x, y, x0);
            if (i > 0) cv::line(img, prev, p, cv::Scalar(55, 128, 106), 1, cv::LINE_AA);
            prev = p;
        }
    }
}

static void draw_graph_edges(cv::Mat& img,
                             const std::vector<RobotSpec>& robots,
                             const std::vector<float>& path_x,
                             const std::vector<float>& path_y,
                             int x0,
                             int step) {
    int k = std::max(0, std::min(step, HORIZON - 1));
    for (int i = 0; i < N_ROBOTS; i++) {
        float xi = path_x[i * HORIZON + k];
        float yi = path_y[i * HORIZON + k];
        for (int j = i + 1; j < N_ROBOTS; j++) {
            if (robots[i].route == robots[j].route) continue;
            float xj = path_x[j * HORIZON + k];
            float yj = path_y[j * HORIZON + k];
            float d = std::sqrt(sqr(xi - xj) + sqr(yi - yj));
            if (d > 1.55f) continue;
            float w = 1.0f - d / 1.55f;
            cv::line(img, to_px(xi, yi, x0), to_px(xj, yj, x0),
                     cv::Scalar(60, 130 + 90 * w, 250), 1, cv::LINE_AA);
        }
    }
}

static void draw_samples(cv::Mat& img,
                         const std::vector<float>& sample_x,
                         const std::vector<float>& sample_y,
                         int x0,
                         int step) {
    int limit = std::max(1, std::min(step, HORIZON));
    for (int robot = 0; robot < SAMPLE_ROBOTS; robot++) {
        for (int rollout = 0; rollout < SAMPLE_ROLLOUTS; rollout += 2) {
            int base = (robot * SAMPLE_ROLLOUTS + rollout) * HORIZON;
            cv::Scalar color = blend(robot_color(robot), cv::Scalar(220, 220, 220), 0.45f);
            for (int k = 1; k < limit; k++) {
                cv::line(img, to_px(sample_x[base + k - 1], sample_y[base + k - 1], x0),
                         to_px(sample_x[base + k], sample_y[base + k], x0),
                         color, 1, cv::LINE_AA);
            }
        }
    }
}

static void draw_paths(cv::Mat& img,
                       const std::vector<RobotSpec>& robots,
                       const std::vector<float>& path_x,
                       const std::vector<float>& path_y,
                       int x0,
                       int step,
                       bool strong) {
    int limit = std::max(1, std::min(step, HORIZON));
    for (int robot = 0; robot < N_ROBOTS; robot++) {
        cv::Scalar color = robot_color(robot);
        int thickness = strong ? 2 : 1;
        for (int k = 1; k < limit; k++) {
            cv::line(img, to_px(path_x[robot * HORIZON + k - 1], path_y[robot * HORIZON + k - 1], x0),
                     to_px(path_x[robot * HORIZON + k], path_y[robot * HORIZON + k], x0),
                     color, thickness, cv::LINE_AA);
        }
        int idx = robot * HORIZON + limit - 1;
        cv::circle(img, to_px(path_x[idx], path_y[idx], x0), 4, cv::Scalar(18, 20, 24),
                   cv::FILLED, cv::LINE_AA);
        cv::circle(img, to_px(path_x[idx], path_y[idx], x0), 3, color, cv::FILLED, cv::LINE_AA);
        if (step >= HORIZON - 1) {
            cv::circle(img, to_px(robots[robot].gx, robots[robot].gy, x0), 3,
                       color, 1, cv::LINE_AA);
        }
    }
}

static cv::Mat draw_frame(const std::vector<RobotSpec>& robots,
                          const std::vector<float>& independent_x,
                          const std::vector<float>& independent_y,
                          const std::vector<float>& coordinated_x,
                          const std::vector<float>& coordinated_y,
                          const std::vector<float>& sample_x,
                          const std::vector<float>& sample_y,
                          const TeamMetrics& independent,
                          const TeamMetrics& coordinated,
                          double gpu_ms,
                          double cpu_ms,
                          int step) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 20, 24));
    draw_field(img, 0);
    draw_field(img, HALF_W);
    draw_routes(img, robots, 0);
    draw_routes(img, robots, HALF_W);
    draw_graph_edges(img, robots, independent_x, independent_y, 0, step);
    draw_graph_edges(img, robots, coordinated_x, coordinated_y, HALF_W, step);
    draw_paths(img, robots, independent_x, independent_y, 0, step, false);
    draw_samples(img, sample_x, sample_y, HALF_W, step);
    draw_paths(img, robots, coordinated_x, coordinated_y, HALF_W, step, true);

    cv::line(img, cv::Point(HALF_W, HEADER_H), cv::Point(HALF_W, PANEL_H - FOOTER_H),
             cv::Scalar(18, 18, 20), 1, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, HEADER_H), cv::Scalar(5, 7, 10), cv::FILLED);
    cv::rectangle(img, cv::Rect(0, PANEL_H - FOOTER_H, PANEL_W, FOOTER_H),
                  cv::Scalar(5, 7, 10), cv::FILLED);

    double speedup = gpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0;
    float collision_drop = 100.0f * (1.0f - static_cast<float>(coordinated.collisions)
                                           / static_cast<float>(std::max(independent.collisions, 1)));
    float risk_drop = 100.0f * (1.0f - coordinated.mean_social_risk
                                      / std::max(independent.mean_social_risk, 1.0e-6f));
    float collision_delta = -collision_drop;
    float risk_delta = -risk_drop;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "GPU priority graph-neural MPPI  %d robots x %d rollouts x H=%d  gpu=%.2f ms  %.1fx",
                  N_ROBOTS, ROLLOUTS_PER_ROBOT, HORIZON, gpu_ms, speedup);
    cv::putText(img, buf, cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "coordinated graph baseline", cv::Point(14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "priority-aware right-of-way MPPI", cv::Point(HALF_W + 14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf),
                  "collisions %d -> %d (%+.1f%%)  social %.3f -> %.3f (%+.1f%%)  min sep %.2f -> %.2f",
                  independent.collisions, coordinated.collisions, collision_delta,
                  independent.mean_social_risk, coordinated.mean_social_risk, risk_delta,
                  independent.min_separation, coordinated.min_separation);
    cv::putText(img, buf, cv::Point(12, PANEL_H - 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(225, 238, 245), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf),
                  "reach %d/%d -> %d/%d   terminal %.2f -> %.2f",
                  independent.reached, N_ROBOTS, coordinated.reached, N_ROBOTS,
                  independent.mean_terminal, coordinated.mean_terminal);
    cv::putText(img, buf, cv::Point(12, PANEL_H - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(225, 238, 245), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<RobotSpec> robots = make_robots();

    std::vector<RolloutResult> cpu_independent;
    std::vector<RolloutResult> cpu_coordinated;
    double cpu_independent_ms = evaluate_cpu_rollouts(robots, 0, cpu_independent);
    double cpu_coordinated_ms = evaluate_cpu_rollouts(robots, 1, cpu_coordinated);
    double cpu_ms = cpu_independent_ms + cpu_coordinated_ms;

    RobotSpec* d_robots = nullptr;
    RolloutResult* d_independent = nullptr;
    RolloutResult* d_coordinated = nullptr;
    float* d_sample_x = nullptr;
    float* d_sample_y = nullptr;
    int total = N_ROBOTS * ROLLOUTS_PER_ROBOT;
    int sample_total = SAMPLE_ROBOTS * SAMPLE_ROLLOUTS * HORIZON;
    CUDA_CHECK(cudaMalloc(&d_robots, N_ROBOTS * sizeof(RobotSpec)));
    CUDA_CHECK(cudaMalloc(&d_independent, total * sizeof(RolloutResult)));
    CUDA_CHECK(cudaMalloc(&d_coordinated, total * sizeof(RolloutResult)));
    CUDA_CHECK(cudaMalloc(&d_sample_x, sample_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sample_y, sample_total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_robots, robots.data(), N_ROBOTS * sizeof(RobotSpec),
                          cudaMemcpyHostToDevice));

    int blocks = (total + THREADS - 1) / THREADS;
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    rollout_kernel<<<blocks, THREADS>>>(d_robots, 0, d_independent, d_sample_x, d_sample_y);
    rollout_kernel<<<blocks, THREADS>>>(d_robots, 1, d_coordinated, d_sample_x, d_sample_y);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float gpu_ms_f = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms_f, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaGetLastError());
    double gpu_ms = static_cast<double>(gpu_ms_f);

    std::vector<RolloutResult> independent_results(total);
    std::vector<RolloutResult> coordinated_results(total);
    std::vector<float> sample_x(sample_total);
    std::vector<float> sample_y(sample_total);
    CUDA_CHECK(cudaMemcpy(independent_results.data(), d_independent, total * sizeof(RolloutResult),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(coordinated_results.data(), d_coordinated, total * sizeof(RolloutResult),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sample_x.data(), d_sample_x, sample_total * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sample_y.data(), d_sample_y, sample_total * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::vector<int> independent_selected = select_best_by_robot(independent_results);
    std::vector<int> coordinated_selected = select_priority_by_robot(robots, coordinated_results);
    std::vector<float> independent_x;
    std::vector<float> independent_y;
    std::vector<float> coordinated_x;
    std::vector<float> coordinated_y;
    reconstruct_paths(robots, independent_selected, 0, independent_x, independent_y);
    reconstruct_paths(robots, coordinated_selected, 1, coordinated_x, coordinated_y);
    TeamMetrics independent = compute_team_metrics(robots, independent_results, independent_selected,
                                                   independent_x, independent_y);
    TeamMetrics coordinated = compute_team_metrics(robots, coordinated_results, coordinated_selected,
                                                   coordinated_x, coordinated_y);

    double speedup = gpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0;
    float collision_drop = 100.0f * (1.0f - static_cast<float>(coordinated.collisions)
                                           / static_cast<float>(std::max(independent.collisions, 1)));
    float risk_drop = 100.0f * (1.0f - coordinated.mean_social_risk
                                      / std::max(independent.mean_social_risk, 1.0e-6f));
    float sep_gain = coordinated.min_separation - independent.min_separation;
    std::printf("CPU baseline+priority-aware MPPI: %.3f ms (%d robots x %d rollouts x H=%d x 2 modes; priority %.3f ms)\n",
                cpu_ms, N_ROBOTS, ROLLOUTS_PER_ROBOT, HORIZON, cpu_coordinated_ms);
    std::printf("GPU priority graph-neural MPPI: %.3f ms (baseline+priority batches, %.1fx vs CPU equivalent rollout eval)\n",
                gpu_ms, speedup);
    std::printf("Baseline coordinated team: collisions %d, reached %d/%d, deadlocks %d, min sep %.3f, terminal avg %.3f, social risk avg/max %.3f/%.3f\n",
                independent.collisions, independent.reached, N_ROBOTS, independent.deadlocks,
                independent.min_separation, independent.mean_terminal,
                independent.mean_social_risk, independent.max_social_risk);
    std::printf("Priority-aware team: collisions %d, reached %d/%d, deadlocks %d, min sep %.3f, terminal avg %.3f, social risk avg/max %.3f/%.3f, collision change %.1f%%, risk change %.1f%%, separation gain %.3f\n",
                coordinated.collisions, coordinated.reached, N_ROBOTS, coordinated.deadlocks,
                coordinated.min_separation, coordinated.mean_terminal,
                coordinated.mean_social_risk, coordinated.max_social_risk,
                collision_drop, risk_drop, sep_gain);

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_priority_graph_neural_mppi.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_priority_graph_neural_mppi.avi\n");
        return 1;
    }
    for (int k = 2; k <= HORIZON; k += 2) {
        video.write(draw_frame(robots, independent_x, independent_y, coordinated_x, coordinated_y,
                               sample_x, sample_y, independent, coordinated, gpu_ms, cpu_ms, k));
    }
    for (int i = 0; i < 12; i++) {
        video.write(draw_frame(robots, independent_x, independent_y, coordinated_x, coordinated_y,
                               sample_x, sample_y, independent, coordinated, gpu_ms, cpu_ms, HORIZON));
    }
    video.release();

    avi_to_gif("gif/gpu_priority_graph_neural_mppi.avi",
               "gif/gpu_priority_graph_neural_mppi.gif", 8, 560);
    std::printf("GIF saved to gif/gpu_priority_graph_neural_mppi.gif\n");

    CUDA_CHECK(cudaFree(d_robots));
    CUDA_CHECK(cudaFree(d_independent));
    CUDA_CHECK(cudaFree(d_coordinated));
    CUDA_CHECK(cudaFree(d_sample_x));
    CUDA_CHECK(cudaFree(d_sample_y));
    return 0;
}
