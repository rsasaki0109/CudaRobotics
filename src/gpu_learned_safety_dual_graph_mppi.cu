// gpu_learned_safety_dual_graph_mppi.cu
//
// GPU learned safety-dual prior graph-neural MPPI game planner.
//
// A batch of robots crosses a shared interaction zone.  The baseline commits a
// one-shot MPPI plan from intent-belief graph messages.  The game variant first
// evaluates a raw best-response pass, then runs both vanilla no-regret and a
// learned-prior safety-dual update.  The learned prior is a tiny fixed-weight
// MLP distilled from the hand-tuned safe no-regret controller: it predicts
// per-robot safety duals, alpha corrections, and a scale hint from graph-risk
// features, while the final CVaR/reach check remains an explicit planner guard.
//
// Output: gif/gpu_learned_safety_dual_graph_mppi.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int N_ROBOTS = 48;
constexpr int N_INTENTS = 4;
constexpr int ROLLOUTS_PER_ROBOT = 768;
constexpr int HORIZON = 72;
constexpr int SAMPLE_ROBOTS = 12;
constexpr int SAMPLE_ROLLOUTS = 8;
constexpr int THREADS = 256;
constexpr int N_GAME_PASSES = 4;
constexpr int PANEL_W = 1920;
constexpr int PANEL_H = 620;
constexpr int HEADER_H = 44;
constexpr int FOOTER_H = 52;
constexpr int MAP_H = PANEL_H - HEADER_H - FOOTER_H;
constexpr int HALF_W = PANEL_W / 4;
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
constexpr float GAME_COLLISION_MARGIN = 0.72f;
constexpr float REGRET_MIN_ALPHA = 0.16f;
constexpr float REGRET_MAX_ALPHA = 0.68f;
constexpr float SAFETY_MIN_ALPHA = 0.05f;
constexpr float SAFETY_MAX_ALPHA = 0.94f;
constexpr float SAFETY_CVAR_TARGET = 38.0f;
constexpr int SAFETY_COLLISION_TARGET = 154;
constexpr float INF_COST = 1.0e20f;

struct RobotSpec {
    float sx;
    float sy;
    float gx;
    float gy;
    float theta0;
    float priority;
    float lane;
    float jitter;
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
    float mean_tail_risk;
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
    float mean_tail_risk;
    float max_social_risk;
    float collision_cvar;
};

struct IntentStats {
    float top1_accuracy;
    float mean_confidence;
    float mean_true_probability;
};

struct GameStats {
    float mean_unilateral_gain;
    float normalized_gain;
    float max_unilateral_gain;
};

struct IterationStats {
    float mean_path_delta;
    float max_path_delta;
    float mean_unilateral_gain;
    float normalized_gain;
    float mean_alpha;
    float min_alpha;
    float max_alpha;
    float mean_positive_regret;
    float max_positive_regret;
    float mean_safety_dual;
    float max_safety_dual;
    float mean_safety_violation;
    float max_safety_violation;
    float safety_scale;
    float mean_prior_scale;
    float max_prior_scale;
    float mean_prior_margin;
    float cvar_before;
    float cvar_after;
};

struct SafetyDualPrior {
    float dual;
    float alpha_multiplier;
    float scale_hint;
    float margin;
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

__host__ __device__ static inline void intent_goal(const RobotSpec& r,
                                                   int intent,
                                                   float& gx,
                                                   float& gy) {
    float lane_f = r.lane;
    float jitter = r.jitter;
    if (intent == 0) {
        gx = 17.38f;
        gy = 7.05f - 0.14f * lane_f - jitter;
    } else if (intent == 1) {
        gx = 10.62f - 0.16f * lane_f - jitter;
        gy = 10.36f;
    } else if (intent == 2) {
        gx = 0.62f;
        gy = 3.98f + 0.14f * lane_f - jitter;
    } else {
        gx = 7.45f + 0.16f * lane_f - jitter;
        gy = 0.62f;
    }
}

__host__ __device__ static inline void intent_route_point(const RobotSpec& r,
                                                          int intent,
                                                          float progress,
                                                          float& x,
                                                          float& y) {
    float p = clampf(progress, 0.0f, 1.0f);
    float sx = r.sx;
    float sy = r.sy;
    float gx;
    float gy;
    intent_goal(r, intent, gx, gy);
    float cx = 8.9f + 0.35f * sinf(0.91f * static_cast<float>(intent + 1));
    float cy = 5.55f + 0.30f * cosf(0.73f * static_cast<float>(intent + 2));
    float q0 = (1.0f - p) * (1.0f - p);
    float q1 = 2.0f * (1.0f - p) * p;
    float q2 = p * p;
    x = q0 * sx + q1 * cx + q2 * gx;
    y = q0 * sy + q1 * cy + q2 * gy;
}

__host__ __device__ static inline void route_point(const RobotSpec& r,
                                                   float progress,
                                                   float& x,
                                                   float& y) {
    intent_route_point(r, r.route, progress, x, y);
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

__host__ __device__ static inline int stale_intent_prior(int robot_id) {
    return (robot_id + 1) & 3;
}

__host__ __device__ static inline void planned_graph_message(const RobotSpec* robots,
                                                             const float* peer_x,
                                                             const float* peer_y,
                                                             int robot_id,
                                                             float x,
                                                             float y,
                                                             int step,
                                                             float& mx,
                                                             float& my,
                                                             float& mean_risk,
                                                             float& tail_risk,
                                                             float& uncertainty,
                                                             float& min_sep) {
    mx = 0.0f;
    my = 0.0f;
    mean_risk = 0.0f;
    tail_risk = 0.0f;
    uncertainty = 0.0f;
    min_sep = 1.0e6f;
    int k = step < HORIZON ? step : HORIZON - 1;
    int kn = step + 3 < HORIZON ? step + 3 : HORIZON - 1;
    const RobotSpec& self = robots[robot_id];
    for (int j = 0; j < N_ROBOTS; j++) {
        if (j == robot_id) continue;
        float px = peer_x[j * HORIZON + k];
        float py = peer_y[j * HORIZON + k];
        float nx = peer_x[j * HORIZON + kn];
        float ny = peer_y[j * HORIZON + kn];
        float dx = x - px;
        float dy = y - py;
        float d2 = dx * dx + dy * dy;
        float d = sqrtf(fmaxf(d2, 1.0e-8f));
        float sep = d - 2.0f * ROBOT_R;
        min_sep = fminf(min_sep, sep);
        if (d > 3.4f) continue;

        float inv_d = rsqrtf(fmaxf(d2, 1.0e-6f));
        float ux = dx * inv_d;
        float uy = dy * inv_d;
        float pvx = nx - px;
        float pvy = ny - py;
        float plen = sqrtf(fmaxf(pvx * pvx + pvy * pvy, 1.0e-6f));
        pvx /= plen;
        pvy /= plen;
        float closing_axis = clampf(-(ux * pvx + uy * pvy), 0.0f, 1.0f);
        float closing = 1.0f - clampf(d / 3.4f, 0.0f, 1.0f);
        float crossing = (self.route == robots[j].route) ? 0.40f : 1.0f;
        float priority = 0.82f + 0.42f * robots[j].priority;
        float hard_close = sep < GAME_COLLISION_MARGIN
                         ? 1.25f * sqr(GAME_COLLISION_MARGIN - sep)
                         : 0.0f;
        float w = crossing * priority
                * (0.20f * expf(-0.5f * d2 / 1.45f)
                   + 1.80f * closing * closing
                   + 0.55f * closing_axis * closing
                   + hard_close);
        float tail = w * (1.0f + 0.58f * closing + 0.34f * closing_axis);
        mean_risk += w;
        tail_risk += tail;
        uncertainty += fmaxf(tail - w, 0.0f);
        mx += tail * ux;
        my += tail * uy;
    }
    float mag = sqrtf(mx * mx + my * my);
    if (mag > 1.0e-5f) {
        float scale = 1.0f / mag;
        mx *= scale;
        my *= scale;
    }
    mean_risk = clampf(mean_risk, 0.0f, 7.0f);
    tail_risk = clampf(tail_risk, 0.0f, 9.0f);
    uncertainty = clampf(uncertainty, 0.0f, 5.0f);
}

__host__ __device__ static inline void rollout_step(const RobotSpec* robots,
                                                    const float* beliefs,
                                                    const float* peer_x,
                                                    const float* peer_y,
                                                    int robot_id,
                                                    int rollout,
                                                    int step,
                                                    int best_response,
                                                    Pose2& s,
                                                    float& prev_speed,
                                                    float& prev_steer,
                                                    float& social_risk,
                                                    float& tail_risk,
                                                    float& belief_uncertainty,
                                                    float& min_sep,
                                                    float& route_error,
                                                    float& smooth,
                                                    float& terrain) {
    const RobotSpec& r = robots[robot_id];
    float progress = 0.0f;
    route_error = route_distance(r, s.x, s.y, &progress);
    float priority_phase = best_response ? 0.05f * (r.priority - 0.62f) : 0.0f;
    float lookahead = progress + 0.14f + priority_phase + 0.04f * hash_unit(rollout, step / 6, 3);
    float tx;
    float ty;
    route_point(r, lookahead, tx, ty);

    float mx;
    float my;
    if (best_response && peer_x && peer_y) {
        planned_graph_message(robots, peer_x, peer_y, robot_id, s.x, s.y, step,
                              mx, my, social_risk, tail_risk, belief_uncertainty, min_sep);
    } else {
        mx = 0.0f;
        my = 0.0f;
        social_risk = 0.0f;
        tail_risk = 0.0f;
        belief_uncertainty = 0.0f;
        min_sep = 1.0e6f;
    }
    float threat = best_response
                 ? 0.46f * social_risk + 0.18f * tail_risk + 0.04f * belief_uncertainty
                 : social_risk;
    float avoid_gain = best_response
                     ? clampf(0.18f + 0.16f * threat, 0.0f, 0.92f)
                     : 0.0f;
    tx += avoid_gain * mx;
    ty += avoid_gain * my;

    terrain = terrain_cost(s.x, s.y);
    float dx = tx - s.x;
    float dy = ty - s.y;
    float target_heading = atan2f(dy, dx);
    float heading_error = wrap_angle(target_heading - s.theta);
    float dist = sqrtf(dx * dx + dy * dy);
    float social_slow = clampf(threat / (best_response ? 18.0f : 6.4f), 0.0f,
                               best_response ? 0.18f : 0.58f);
    float terrain_slow = clampf((terrain - 0.7f) / 4.0f, 0.0f, 1.0f);
    float base_speed = clampf(0.38f + 1.06f * fast_sigmoid(1.2f * cosf(heading_error)
                                  + 0.18f * dist - 1.15f * terrain_slow
                                  - 1.25f * social_slow),
                              MIN_SPEED, MAX_SPEED);
    if (best_response) {
        base_speed = clampf(base_speed * (0.98f + 0.10f * r.priority), MIN_SPEED, MAX_SPEED);
    }
    float base_steer = clampf(0.58f * fast_tanh(1.62f * heading_error
                                  + 0.34f * social_slow * (mx * -sinf(s.theta)
                                                          + my * cosf(s.theta))),
                              -MAX_STEER, MAX_STEER);
    float n0 = 0.48f * hash_signed(rollout, step, 0)
             + 0.20f * hash_signed(rollout, step / 4, robot_id + 7);
    float n1 = 0.70f * hash_signed(rollout, step, 1)
             + 0.24f * hash_signed(rollout, step / 5, robot_id + 19);
    float speed_sigma = best_response ? 0.18f : 0.22f;
    float steer_sigma = best_response ? 0.21f : 0.22f;
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
                                                                 const float* beliefs,
                                                                 const float* peer_x,
                                                                 const float* peer_y,
                                                                 int robot_id,
                                                                 int rollout,
                                                                 int best_response) {
    const RobotSpec& r = robots[robot_id];
    Pose2 s{r.sx, r.sy, r.theta0};
    float blind_cost = 0.0f;
    float full_cost = 0.0f;
    float risk_sum = 0.0f;
    float tail_sum = 0.0f;
    float risk_max = 0.0f;
    float min_sep = 1.0e6f;
    float route_sum = 0.0f;
    float prev_speed = 0.70f;
    float prev_steer = 0.0f;

    for (int k = 0; k < HORIZON; k++) {
        float social_risk;
        float tail_risk;
        float belief_uncertainty;
        float step_min_sep;
        float route_error;
        float smooth;
        float terrain;
        rollout_step(robots, beliefs, peer_x, peer_y, robot_id, rollout, k, best_response,
                     s, prev_speed, prev_steer, social_risk, tail_risk, belief_uncertainty,
                     step_min_sep, route_error, smooth, terrain);
        float obstacle = terrain > 5.0f ? 8.0f * sqr(terrain - 5.0f) : 0.0f;
        float base_step = 0.62f * terrain + 1.35f * route_error * route_error
                        + 0.20f * sqr(prev_steer / MAX_STEER) + smooth + obstacle;
        float close_weight = best_response ? 42.0f : 62.0f;
        float social_weight = best_response ? 7.0f : 18.0f;
        float tail_weight = best_response ? 1.6f : 0.0f;
        float close_penalty = step_min_sep < COLLISION_MARGIN
                            ? close_weight * sqr(COLLISION_MARGIN - step_min_sep)
                            : 0.0f;
        float tail_close = best_response && step_min_sep < 0.76f
                         ? 1.0f * sqr(0.76f - step_min_sep)
                         : 0.0f;
        float uncertainty_step = best_response ? 0.15f * belief_uncertainty : 0.0f;
        float social_step = social_weight * social_risk + tail_weight * tail_risk
                          + uncertainty_step + close_penalty + tail_close;
        blind_cost += base_step;
        full_cost += base_step + social_step;
        risk_sum += social_risk;
        tail_sum += tail_risk;
        risk_max = fmaxf(risk_max, fmaxf(social_risk, tail_risk));
        min_sep = fminf(min_sep, step_min_sep);
        route_sum += route_error;
    }

    float terminal = sqrtf(sqr(r.gx - s.x) + sqr(r.gy - s.y));
    float final_route = route_distance(r, s.x, s.y, nullptr);
    float terminal_weight = best_response ? 460.0f : 118.0f;
    float terminal_cost = terminal_weight * terminal * terminal + 8.5f * final_route;
    blind_cost += terminal_cost;
    full_cost += terminal_cost;

    RolloutResult out;
    out.select_cost = full_cost;
    out.full_cost = full_cost;
    out.terminal_error = terminal;
    out.mean_social_risk = risk_sum / static_cast<float>(HORIZON);
    out.mean_tail_risk = tail_sum / static_cast<float>(HORIZON);
    out.max_social_risk = risk_max;
    out.min_separation = min_sep;
    out.route_error = route_sum / static_cast<float>(HORIZON);
    out.robot_id = robot_id;
    out.rollout_id = rollout;
    return out;
}

__global__ void rollout_kernel(const RobotSpec* __restrict__ robots,
                               const float* __restrict__ beliefs,
                               const float* __restrict__ peer_x,
                               const float* __restrict__ peer_y,
                               int best_response,
                               RolloutResult* __restrict__ results,
                               float* __restrict__ sample_x,
                               float* __restrict__ sample_y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N_ROBOTS * ROLLOUTS_PER_ROBOT;
    if (idx >= total) return;
    int robot_id = idx / ROLLOUTS_PER_ROBOT;
    int rollout = idx - robot_id * ROLLOUTS_PER_ROBOT;
    results[idx] = evaluate_rollout(robots, beliefs, peer_x, peer_y, robot_id, rollout,
                                    best_response);

    if (!best_response || robot_id >= SAMPLE_ROBOTS || rollout >= SAMPLE_ROLLOUTS) return;
    Pose2 s{robots[robot_id].sx, robots[robot_id].sy, robots[robot_id].theta0};
    float prev_speed = 0.70f;
    float prev_steer = 0.0f;
    int sample_base = (robot_id * SAMPLE_ROLLOUTS + rollout) * HORIZON;
    for (int k = 0; k < HORIZON; k++) {
        float social_risk;
        float tail_risk;
        float belief_uncertainty;
        float min_sep;
        float route_error;
        float smooth;
        float terrain;
        rollout_step(robots, beliefs, peer_x, peer_y, robot_id, rollout, k, best_response,
                     s, prev_speed, prev_steer, social_risk, tail_risk,
                     belief_uncertainty, min_sep, route_error, smooth, terrain);
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
        r.lane = lane_f;
        r.jitter = jitter;
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
        intent_goal(r, r.route, r.gx, r.gy);
        robots[i] = r;
    }
    return robots;
}

static std::vector<float> infer_intent_beliefs(const std::vector<RobotSpec>& robots,
                                               std::vector<int>& top_intents,
                                               IntentStats& stats) {
    std::vector<float> beliefs(N_ROBOTS * N_INTENTS, 0.0f);
    top_intents.assign(N_ROBOTS, 0);
    int correct = 0;
    float confidence_sum = 0.0f;
    float true_prob_sum = 0.0f;
    constexpr float P0 = 0.18f;
    constexpr float P1 = 0.36f;
    for (int robot = 0; robot < N_ROBOTS; robot++) {
        const RobotSpec& r = robots[robot];
        float ox0;
        float oy0;
        float ox1;
        float oy1;
        route_point(r, P0, ox0, oy0);
        route_point(r, P1, ox1, oy1);
        float noise_x = 0.035f * std::sin(1.73f * static_cast<float>(robot + 1));
        float noise_y = 0.030f * std::cos(2.11f * static_cast<float>(robot + 3));
        ox1 += noise_x;
        oy1 += noise_y;
        float ovx = ox1 - ox0;
        float ovy = oy1 - oy0;
        float olen = std::sqrt(std::max(ovx * ovx + ovy * ovy, 1.0e-6f));
        ovx /= olen;
        ovy /= olen;

        float raw[N_INTENTS];
        float max_raw = -1.0e9f;
        for (int intent = 0; intent < N_INTENTS; intent++) {
            float cx0;
            float cy0;
            float cx1;
            float cy1;
            intent_route_point(r, intent, P0, cx0, cy0);
            intent_route_point(r, intent, P1, cx1, cy1);
            float cvx = cx1 - cx0;
            float cvy = cy1 - cy0;
            float clen = std::sqrt(std::max(cvx * cvx + cvy * cvy, 1.0e-6f));
            cvx /= clen;
            cvy /= clen;
            float pos_cost = sqr(ox1 - cx1) + sqr(oy1 - cy1);
            float dir_cost = 1.0f - clampf(ovx * cvx + ovy * cvy, -1.0f, 1.0f);
            float prior = (intent == stale_intent_prior(robot)) ? 0.10f : 0.0f;
            raw[intent] = -2.25f * pos_cost - 1.35f * dir_cost + prior;
            max_raw = std::max(max_raw, raw[intent]);
        }

        float denom = 0.0f;
        for (int intent = 0; intent < N_INTENTS; intent++) {
            float p = std::exp(raw[intent] - max_raw);
            beliefs[robot * N_INTENTS + intent] = p;
            denom += p;
        }
        float best_p = -1.0f;
        int best = 0;
        for (int intent = 0; intent < N_INTENTS; intent++) {
            float p = beliefs[robot * N_INTENTS + intent] / std::max(denom, 1.0e-6f);
            p = 0.025f + 0.900f * p;
            beliefs[robot * N_INTENTS + intent] = p;
            if (p > best_p) {
                best_p = p;
                best = intent;
            }
        }
        top_intents[robot] = best;
        if (best == r.route) correct++;
        confidence_sum += best_p;
        true_prob_sum += beliefs[robot * N_INTENTS + r.route];
    }
    stats.top1_accuracy = 100.0f * static_cast<float>(correct) / static_cast<float>(N_ROBOTS);
    stats.mean_confidence = confidence_sum / static_cast<float>(N_ROBOTS);
    stats.mean_true_probability = true_prob_sum / static_cast<float>(N_ROBOTS);
    return beliefs;
}

static double evaluate_cpu_rollouts(const std::vector<RobotSpec>& robots,
                                    const std::vector<float>& beliefs,
                                    const float* peer_x,
                                    const float* peer_y,
                                    int best_response,
                                    std::vector<RolloutResult>& out) {
    out.resize(N_ROBOTS * ROLLOUTS_PER_ROBOT);
    auto begin = std::chrono::high_resolution_clock::now();
    for (int robot = 0; robot < N_ROBOTS; robot++) {
        for (int rollout = 0; rollout < ROLLOUTS_PER_ROBOT; rollout++) {
            int idx = robot * ROLLOUTS_PER_ROBOT + rollout;
            out[idx] = evaluate_rollout(robots.data(), beliefs.data(), peer_x, peer_y,
                                        robot, rollout, best_response);
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

static void reconstruct_paths(const std::vector<RobotSpec>& robots,
                              const std::vector<float>& beliefs,
                              const float* peer_x,
                              const float* peer_y,
                              const std::vector<int>& selected,
                              int best_response,
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
            float tail_risk;
            float belief_uncertainty;
            float min_sep;
            float route_error;
            float smooth;
            float terrain;
            rollout_step(robots.data(), beliefs.data(), peer_x, peer_y, robot, selected[robot],
                         k, best_response, s, prev_speed, prev_steer, social_risk,
                         tail_risk, belief_uncertainty, min_sep, route_error, smooth, terrain);
            path_x[robot * HORIZON + k] = s.x;
            path_y[robot * HORIZON + k] = s.y;
        }
    }
}

static TeamMetrics compute_path_team_metrics(const std::vector<RobotSpec>& robots,
                                             const std::vector<float>& path_x,
                                             const std::vector<float>& path_y) {
    TeamMetrics m{};
    m.min_separation = 1.0e6f;
    float terminal_sum = 0.0f;
    for (int robot = 0; robot < N_ROBOTS; robot++) {
        int last = robot * HORIZON + HORIZON - 1;
        float terminal = std::sqrt(sqr(robots[robot].gx - path_x[last])
                                 + sqr(robots[robot].gy - path_y[last]));
        terminal_sum += terminal;
        if (terminal < 2.25f) m.reached++;
        if (terminal > 4.0f) m.deadlocks++;
    }
    std::vector<float> step_hazards;
    step_hazards.reserve(HORIZON);
    float social_sum = 0.0f;
    float tail_sum = 0.0f;
    for (int k = 0; k < HORIZON; k++) {
        float step_hazard = 0.0f;
        float step_social = 0.0f;
        float step_tail = 0.0f;
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
                if (sep < 2.75f) {
                    float closing = 1.0f - clampf(sep / 2.75f, 0.0f, 1.0f);
                    float w = 0.15f * expf(-0.5f * sqr(sep / 1.05f))
                            + 1.15f * closing * closing;
                    step_social += w;
                    step_tail += w * (1.0f + 0.45f * closing);
                }
                if (sep < 0.92f) step_hazard += sqr(0.92f - sep);
            }
        }
        social_sum += step_social;
        tail_sum += step_tail;
        m.max_social_risk = std::max(m.max_social_risk, step_tail);
        step_hazards.push_back(step_hazard);
    }
    std::sort(step_hazards.begin(), step_hazards.end(), std::greater<float>());
    int tail_n = std::max(1, static_cast<int>(step_hazards.size() * 0.10f));
    float hazard_sum = 0.0f;
    for (int i = 0; i < tail_n; i++) hazard_sum += step_hazards[i];
    m.collision_cvar = hazard_sum / static_cast<float>(tail_n);
    m.mean_terminal = terminal_sum / static_cast<float>(N_ROBOTS);
    m.mean_social_risk = social_sum / static_cast<float>(HORIZON * N_ROBOTS);
    m.mean_tail_risk = tail_sum / static_cast<float>(HORIZON * N_ROBOTS);
    return m;
}

static GameStats compute_game_stats(const std::vector<RolloutResult>& response_results,
                                    const std::vector<int>& one_shot_selected,
                                    const std::vector<int>& response_selected) {
    GameStats stats{};
    float gain_sum = 0.0f;
    float baseline_cost_sum = 0.0f;
    for (int robot = 0; robot < N_ROBOTS; robot++) {
        const RolloutResult& one_shot_under_game =
            response_results[robot * ROLLOUTS_PER_ROBOT + one_shot_selected[robot]];
        const RolloutResult& best_response =
            response_results[robot * ROLLOUTS_PER_ROBOT + response_selected[robot]];
        float gain = fmaxf(one_shot_under_game.select_cost - best_response.select_cost, 0.0f);
        gain_sum += gain;
        baseline_cost_sum += fmaxf(one_shot_under_game.select_cost, 1.0e-3f);
        stats.max_unilateral_gain = std::max(stats.max_unilateral_gain, gain);
    }
    stats.mean_unilateral_gain = gain_sum / static_cast<float>(N_ROBOTS);
    stats.normalized_gain = 100.0f * gain_sum / std::max(baseline_cost_sum, 1.0e-3f);
    return stats;
}

static void mix_paths_regret_aware(const std::vector<float>& previous_x,
                                   const std::vector<float>& previous_y,
                                   const std::vector<float>& response_x,
                                   const std::vector<float>& response_y,
                                   const std::vector<RolloutResult>& response_results,
                                   const std::vector<int>& previous_selected,
                                   const std::vector<int>& response_selected,
                                   int pass,
                                   std::vector<float>& mixed_x,
                                   std::vector<float>& mixed_y,
                                   IterationStats& stats) {
    mixed_x.resize(previous_x.size());
    mixed_y.resize(previous_y.size());
    stats = IterationStats{};
    stats.min_alpha = 1.0e6f;
    float gain_sum = 0.0f;
    float baseline_cost_sum = 0.0f;
    float pass_decay = 1.0f / (1.0f + 0.10f * static_cast<float>(std::max(0, pass - 1)));
    for (int robot = 0; robot < N_ROBOTS; robot++) {
        const RolloutResult& previous =
            response_results[robot * ROLLOUTS_PER_ROBOT + previous_selected[robot]];
        const RolloutResult& best_response =
            response_results[robot * ROLLOUTS_PER_ROBOT + response_selected[robot]];
        float gain = fmaxf(previous.select_cost - best_response.select_cost, 0.0f);
        float baseline_cost = fmaxf(previous.select_cost, 1.0e-3f);
        float positive_regret = gain / baseline_cost;
        float terminal_guard = clampf(1.08f - 0.24f * fmaxf(best_response.terminal_error
                                                           - previous.terminal_error, 0.0f),
                                      0.62f, 1.08f);
        float separation_guard = best_response.min_separation >= previous.min_separation
                               ? 1.06f : 0.90f;
        float alpha_unit = clampf(positive_regret / 0.32f, 0.0f, 1.0f);
        float alpha = (REGRET_MIN_ALPHA
                    + (REGRET_MAX_ALPHA - REGRET_MIN_ALPHA) * alpha_unit)
                    * pass_decay * terminal_guard * separation_guard;
        alpha = clampf(alpha, 0.08f, REGRET_MAX_ALPHA);
        float beta = 1.0f - alpha;

        gain_sum += gain;
        baseline_cost_sum += baseline_cost;
        stats.mean_alpha += alpha;
        stats.min_alpha = std::min(stats.min_alpha, alpha);
        stats.max_alpha = std::max(stats.max_alpha, alpha);
        stats.mean_positive_regret += positive_regret;
        stats.max_positive_regret = std::max(stats.max_positive_regret, positive_regret);
        stats.mean_unilateral_gain += gain;

        for (int k = 0; k < HORIZON; k++) {
            int idx = robot * HORIZON + k;
            mixed_x[idx] = beta * previous_x[idx] + alpha * response_x[idx];
            mixed_y[idx] = beta * previous_y[idx] + alpha * response_y[idx];
            float d = std::sqrt(sqr(mixed_x[idx] - previous_x[idx])
                              + sqr(mixed_y[idx] - previous_y[idx]));
            stats.mean_path_delta += d;
            stats.max_path_delta = std::max(stats.max_path_delta, d);
        }
    }
    stats.mean_path_delta /= static_cast<float>(N_ROBOTS * HORIZON);
    stats.mean_unilateral_gain /= static_cast<float>(N_ROBOTS);
    stats.normalized_gain = 100.0f * gain_sum / std::max(baseline_cost_sum, 1.0e-3f);
    stats.mean_alpha /= static_cast<float>(N_ROBOTS);
    stats.mean_positive_regret /= static_cast<float>(N_ROBOTS);
}

static SafetyDualPrior learned_safety_dual_prior(const RobotSpec& robot,
                                                 const RolloutResult& previous,
                                                 const RolloutResult& best_response,
                                                 float positive_regret,
                                                 int pass) {
    float tail_relief = clampf((previous.mean_tail_risk - best_response.mean_tail_risk)
                             / fmaxf(previous.mean_tail_risk, 0.25f),
                               -1.0f, 1.0f);
    float sep_relief = clampf((best_response.min_separation - previous.min_separation)
                            / 0.30f, -1.0f, 1.0f);
    float terminal_delta = best_response.terminal_error - previous.terminal_error;
    float tail_violation = fmaxf(best_response.mean_tail_risk
                               - 1.04f * previous.mean_tail_risk - 0.10f, 0.0f);
    float sep_violation = fmaxf(previous.min_separation
                              - best_response.min_separation - 0.03f, 0.0f);
    float terminal_violation = fmaxf(terminal_delta - 0.24f, 0.0f);
    float crowd = clampf(best_response.mean_tail_risk / 3.2f
                       + fmaxf(0.72f - best_response.min_separation, 0.0f), 0.0f, 2.2f);
    float pass_phase = static_cast<float>(std::max(0, pass - 1));

    float h0 = fast_tanh(1.34f * positive_regret + 0.72f * tail_relief
                       + 0.54f * sep_relief - 0.56f * terminal_violation
                       + 0.16f * (robot.priority - 0.65f));
    float h1 = fast_tanh(0.92f * crowd + 0.42f * fmaxf(tail_relief, 0.0f)
                       + 0.36f * fmaxf(sep_relief, 0.0f) - 0.30f * pass_phase);
    float h2 = fast_tanh(1.20f * tail_violation + 1.18f * sep_violation
                       + 0.50f * terminal_violation
                       + 0.30f * fmaxf(terminal_delta, 0.0f));
    float h3 = fast_tanh(0.76f * positive_regret - 0.36f * crowd
                       + 0.28f * (0.65f - robot.priority)
                       - 0.20f * pass_phase);

    SafetyDualPrior p{};
    p.dual = clampf(1.00f + 0.30f * h0 + 0.18f * h1 - 0.28f * h2 + 0.08f * h3,
                    0.52f, 1.44f);
    p.alpha_multiplier = clampf(0.98f + 0.20f * h0 + 0.12f * h1 - 0.24f * h2
                              - 0.05f * pass_phase,
                                0.58f, 1.22f);
    p.scale_hint = clampf(0.96f + 0.28f * h1 + 0.16f * h0 - 0.18f * h2
                        - 0.04f * pass_phase,
                          0.68f, 1.46f);
    p.margin = tail_relief + 0.62f * sep_relief - 1.18f * tail_violation
             - 1.24f * sep_violation - 0.30f * terminal_violation;
    return p;
}

static void mix_paths_learned_safety_prior(const std::vector<RobotSpec>& robots,
                                         const std::vector<float>& previous_x,
                                         const std::vector<float>& previous_y,
                                         const std::vector<float>& response_x,
                                         const std::vector<float>& response_y,
                                         const std::vector<RolloutResult>& response_results,
                                         const std::vector<int>& previous_selected,
                                         const std::vector<int>& response_selected,
                                         int pass,
                                         std::vector<float>& mixed_x,
                                         std::vector<float>& mixed_y,
                                         IterationStats& stats) {
    std::vector<float> alpha_base(N_ROBOTS, 0.0f);
    std::vector<float> positive_regrets(N_ROBOTS, 0.0f);
    std::vector<float> gains(N_ROBOTS, 0.0f);
    std::vector<float> baselines(N_ROBOTS, 1.0f);
    std::vector<float> safety_duals(N_ROBOTS, 1.0f);
    std::vector<float> safety_violations(N_ROBOTS, 0.0f);
    std::vector<float> prior_scale_hints(N_ROBOTS, 1.0f);
    std::vector<float> prior_margins(N_ROBOTS, 0.0f);
    float pass_decay = 1.0f / (1.0f + 0.10f * static_cast<float>(std::max(0, pass - 1)));
    float gain_sum = 0.0f;
    float baseline_cost_sum = 0.0f;

    for (int robot = 0; robot < N_ROBOTS; robot++) {
        const RolloutResult& previous =
            response_results[robot * ROLLOUTS_PER_ROBOT + previous_selected[robot]];
        const RolloutResult& best_response =
            response_results[robot * ROLLOUTS_PER_ROBOT + response_selected[robot]];
        float gain = fmaxf(previous.select_cost - best_response.select_cost, 0.0f);
        float baseline_cost = fmaxf(previous.select_cost, 1.0e-3f);
        float positive_regret = gain / baseline_cost;
        float terminal_guard = clampf(1.08f - 0.24f * fmaxf(best_response.terminal_error
                                                           - previous.terminal_error, 0.0f),
                                      0.62f, 1.08f);
        float separation_guard = best_response.min_separation >= previous.min_separation
                               ? 1.06f : 0.90f;
        float alpha_unit = clampf(positive_regret / 0.32f, 0.0f, 1.0f);
        float alpha = (REGRET_MIN_ALPHA
                    + (REGRET_MAX_ALPHA - REGRET_MIN_ALPHA) * alpha_unit)
                    * pass_decay * terminal_guard * separation_guard;

        float tail_violation = fmaxf(best_response.mean_tail_risk
                                   - 1.04f * previous.mean_tail_risk - 0.10f, 0.0f);
        float sep_violation = fmaxf(previous.min_separation
                                  - best_response.min_separation - 0.03f, 0.0f);
        float terminal_violation = fmaxf(best_response.terminal_error
                                       - previous.terminal_error - 0.24f, 0.0f);
        float safety_violation = tail_violation + 1.8f * sep_violation
                               + 0.20f * terminal_violation;
        SafetyDualPrior prior =
            learned_safety_dual_prior(robots[robot], previous, best_response,
                                      positive_regret, pass);
        float safety_dual = prior.dual;

        alpha_base[robot] = clampf(alpha * safety_dual * prior.alpha_multiplier,
                                   SAFETY_MIN_ALPHA, SAFETY_MAX_ALPHA);
        positive_regrets[robot] = positive_regret;
        gains[robot] = gain;
        baselines[robot] = baseline_cost;
        safety_duals[robot] = safety_dual;
        safety_violations[robot] = safety_violation;
        prior_scale_hints[robot] = prior.scale_hint;
        prior_margins[robot] = prior.margin;
        gain_sum += gain;
        baseline_cost_sum += baseline_cost;
    }

    auto mix_for_scale = [&](float scale,
                             std::vector<float>& out_x,
                             std::vector<float>& out_y,
                             std::vector<float>* alpha_out) {
        out_x.resize(previous_x.size());
        out_y.resize(previous_y.size());
        if (alpha_out) alpha_out->assign(N_ROBOTS, 0.0f);
        for (int robot = 0; robot < N_ROBOTS; robot++) {
            float alpha = clampf(alpha_base[robot] * scale,
                                 SAFETY_MIN_ALPHA, SAFETY_MAX_ALPHA);
            int last = robot * HORIZON + HORIZON - 1;
            float beta0 = 1.0f - alpha;
            float final_x = beta0 * previous_x[last] + alpha * response_x[last];
            float final_y = beta0 * previous_y[last] + alpha * response_y[last];
            float terminal = std::sqrt(sqr(robots[robot].gx - final_x)
                                     + sqr(robots[robot].gy - final_y));
            if (pass == N_GAME_PASSES - 1 && terminal > 2.25f) {
                float reach_keep = clampf(1.0f - 0.42f * (terminal - 2.25f),
                                          0.62f, 1.0f);
                alpha = clampf(alpha * reach_keep, SAFETY_MIN_ALPHA, SAFETY_MAX_ALPHA);
            }
            if (alpha_out) (*alpha_out)[robot] = alpha;
            float beta = 1.0f - alpha;
            for (int k = 0; k < HORIZON; k++) {
                int idx = robot * HORIZON + k;
                out_x[idx] = beta * previous_x[idx] + alpha * response_x[idx];
                out_y[idx] = beta * previous_y[idx] + alpha * response_y[idx];
            }
        }
    };

    TeamMetrics previous_metrics = compute_path_team_metrics(robots, previous_x, previous_y);
    float prior_center = 0.0f;
    for (float hint : prior_scale_hints) prior_center += hint;
    prior_center = clampf(prior_center / static_cast<float>(N_ROBOTS), 0.70f, 1.45f);
    const float scale_offsets[] = {0.68f, 0.84f, 1.00f, 1.16f, 1.34f, 1.56f, 1.78f};
    float best_score = INF_COST;
    float best_scale = 1.0f;
    TeamMetrics best_metrics{};
    std::vector<float> best_x;
    std::vector<float> best_y;
    for (float offset : scale_offsets) {
        float scale = clampf(prior_center * offset, 0.50f, 1.82f);
        std::vector<float> candidate_x;
        std::vector<float> candidate_y;
        mix_for_scale(scale, candidate_x, candidate_y, nullptr);
        TeamMetrics metrics = compute_path_team_metrics(robots, candidate_x, candidate_y);
        float cvar_over = fmaxf(metrics.collision_cvar - SAFETY_CVAR_TARGET, 0.0f);
        float collision_over = static_cast<float>(
            std::max(metrics.collisions - SAFETY_COLLISION_TARGET, 0));
        float reach_loss = static_cast<float>(N_ROBOTS - metrics.reached);
        float terminal_over = fmaxf(metrics.mean_terminal - 0.82f, 0.0f);
        float sep_over = fmaxf(-0.42f - metrics.min_separation, 0.0f);
        float previous_cvar_over = fmaxf(metrics.collision_cvar
                                       - 0.96f * previous_metrics.collision_cvar, 0.0f);
        float final_pass = pass >= N_GAME_PASSES - 1 ? 1.0f : 0.0f;
        float cvar_weight = 3.0f + 1.2f * final_pass;
        float collision_weight = 0.28f + 0.06f * final_pass;
        float previous_cvar_weight = 0.65f + 0.10f * final_pass;
        float cvar_floor_weight = 0.18f + 0.03f * final_pass;
        float score = cvar_weight * cvar_over
                    + collision_weight * collision_over
                    + 24.0f * reach_loss
                    + 18.0f * static_cast<float>(metrics.deadlocks)
                    + 8.0f * terminal_over
                    + 3.0f * sep_over
                    + previous_cvar_weight * previous_cvar_over
                    + 0.08f * static_cast<float>(metrics.collisions)
                    + cvar_floor_weight * metrics.collision_cvar;
        if (score < best_score) {
            best_score = score;
            best_scale = scale;
            best_metrics = metrics;
            best_x.swap(candidate_x);
            best_y.swap(candidate_y);
        }
    }

    std::vector<float> chosen_alpha;
    mix_for_scale(best_scale, mixed_x, mixed_y, &chosen_alpha);
    if (!best_x.empty()) {
        mixed_x.swap(best_x);
        mixed_y.swap(best_y);
    }

    stats = IterationStats{};
    stats.min_alpha = 1.0e6f;
    stats.safety_scale = best_scale;
    stats.cvar_before = previous_metrics.collision_cvar;
    stats.cvar_after = best_metrics.collision_cvar;
    for (int robot = 0; robot < N_ROBOTS; robot++) {
        float alpha = chosen_alpha[robot];
        gain_sum += 0.0f;
        baseline_cost_sum += 0.0f;
        stats.mean_alpha += alpha;
        stats.min_alpha = std::min(stats.min_alpha, alpha);
        stats.max_alpha = std::max(stats.max_alpha, alpha);
        stats.mean_positive_regret += positive_regrets[robot];
        stats.max_positive_regret = std::max(stats.max_positive_regret,
                                             positive_regrets[robot]);
        stats.mean_unilateral_gain += gains[robot];
        stats.mean_safety_dual += safety_duals[robot];
        stats.max_safety_dual = std::max(stats.max_safety_dual, safety_duals[robot]);
        stats.mean_safety_violation += safety_violations[robot];
        stats.max_safety_violation = std::max(stats.max_safety_violation,
                                              safety_violations[robot]);
        stats.mean_prior_scale += prior_scale_hints[robot];
        stats.max_prior_scale = std::max(stats.max_prior_scale, prior_scale_hints[robot]);
        stats.mean_prior_margin += prior_margins[robot];
        for (int k = 0; k < HORIZON; k++) {
            int idx = robot * HORIZON + k;
            float d = std::sqrt(sqr(mixed_x[idx] - previous_x[idx])
                              + sqr(mixed_y[idx] - previous_y[idx]));
            stats.mean_path_delta += d;
            stats.max_path_delta = std::max(stats.max_path_delta, d);
        }
    }
    stats.mean_path_delta /= static_cast<float>(N_ROBOTS * HORIZON);
    stats.mean_unilateral_gain /= static_cast<float>(N_ROBOTS);
    stats.normalized_gain = 100.0f * gain_sum / std::max(baseline_cost_sum, 1.0e-3f);
    stats.mean_alpha /= static_cast<float>(N_ROBOTS);
    stats.mean_positive_regret /= static_cast<float>(N_ROBOTS);
    stats.mean_safety_dual /= static_cast<float>(N_ROBOTS);
    stats.mean_safety_violation /= static_cast<float>(N_ROBOTS);
    stats.mean_prior_scale /= static_cast<float>(N_ROBOTS);
    stats.mean_prior_margin /= static_cast<float>(N_ROBOTS);
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
                          const std::vector<float>& one_shot_x,
                          const std::vector<float>& one_shot_y,
                          const std::vector<float>& raw_x,
                          const std::vector<float>& raw_y,
                          const std::vector<float>& noregret_x,
                          const std::vector<float>& noregret_y,
                          const std::vector<float>& safe_x,
                          const std::vector<float>& safe_y,
                          const std::vector<float>& sample_x,
                          const std::vector<float>& sample_y,
                          const TeamMetrics& one_shot,
                          const TeamMetrics& raw_response,
                          const TeamMetrics& noregret,
                          const TeamMetrics& safe_noregret,
                          const IntentStats& intent_stats,
                          const GameStats& raw_game,
                          const GameStats& noregret_game,
                          const GameStats& safe_game,
                          const std::vector<IterationStats>& noregret_stats,
                          const std::vector<IterationStats>& safe_stats,
                          double gpu_ms,
                          double cpu_ms,
                          int step) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 20, 24));
    draw_field(img, 0);
    draw_field(img, HALF_W);
    draw_field(img, 2 * HALF_W);
    draw_field(img, 3 * HALF_W);
    draw_routes(img, robots, 0);
    draw_routes(img, robots, HALF_W);
    draw_routes(img, robots, 2 * HALF_W);
    draw_routes(img, robots, 3 * HALF_W);
    draw_graph_edges(img, robots, one_shot_x, one_shot_y, 0, step);
    draw_graph_edges(img, robots, raw_x, raw_y, HALF_W, step);
    draw_graph_edges(img, robots, noregret_x, noregret_y, 2 * HALF_W, step);
    draw_graph_edges(img, robots, safe_x, safe_y, 3 * HALF_W, step);
    draw_paths(img, robots, one_shot_x, one_shot_y, 0, step, false);
    draw_paths(img, robots, raw_x, raw_y, HALF_W, step, true);
    draw_paths(img, robots, noregret_x, noregret_y, 2 * HALF_W, step, true);
    draw_samples(img, sample_x, sample_y, 3 * HALF_W, step);
    draw_paths(img, robots, safe_x, safe_y, 3 * HALF_W, step, true);

    cv::line(img, cv::Point(HALF_W, HEADER_H), cv::Point(HALF_W, PANEL_H - FOOTER_H),
             cv::Scalar(18, 18, 20), 1, cv::LINE_AA);
    cv::line(img, cv::Point(2 * HALF_W, HEADER_H), cv::Point(2 * HALF_W, PANEL_H - FOOTER_H),
             cv::Scalar(18, 18, 20), 1, cv::LINE_AA);
    cv::line(img, cv::Point(3 * HALF_W, HEADER_H), cv::Point(3 * HALF_W, PANEL_H - FOOTER_H),
             cv::Scalar(18, 18, 20), 1, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, HEADER_H), cv::Scalar(5, 7, 10), cv::FILLED);
    cv::rectangle(img, cv::Rect(0, PANEL_H - FOOTER_H, PANEL_W, FOOTER_H),
                  cv::Scalar(5, 7, 10), cv::FILLED);

    double speedup = gpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0;
    float raw_collision_drop = 100.0f * (1.0f - static_cast<float>(raw_response.collisions)
                                               / static_cast<float>(std::max(one_shot.collisions, 1)));
    float noregret_collision_drop = 100.0f * (1.0f - static_cast<float>(noregret.collisions)
                                                    / static_cast<float>(std::max(one_shot.collisions, 1)));
    float safe_collision_drop = 100.0f * (1.0f - static_cast<float>(safe_noregret.collisions)
                                                / static_cast<float>(std::max(one_shot.collisions, 1)));
    float safe_cvar_drop = 100.0f * (1.0f - safe_noregret.collision_cvar
                                           / std::max(one_shot.collision_cvar, 1.0e-6f));
    IterationStats last_safe{};
    if (!safe_stats.empty()) last_safe = safe_stats.back();
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "GPU learned safety-dual prior graph MPPI  %d robots x %d rollouts x H=%d x %d passes  alpha %.2f..%.2f  gpu=%.2f ms  %.1fx",
                  N_ROBOTS, ROLLOUTS_PER_ROBOT, HORIZON, N_GAME_PASSES,
                  last_safe.min_alpha, last_safe.max_alpha, gpu_ms, speedup);
    cv::putText(img, buf, cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "selfish one-shot MPPI", cv::Point(14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "raw best response", cv::Point(HALF_W + 14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "regret-matched no-regret", cv::Point(2 * HALF_W + 14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "learned safety-dual prior", cv::Point(3 * HALF_W + 14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf),
                  "collisions %d -> %d -> %d -> %d (-%.1f%% raw, -%.1f%% nr, -%.1f%% safe)  CVaR %.2f -> %.2f (-%.1f%%)",
                  one_shot.collisions, raw_response.collisions, noregret.collisions,
                  safe_noregret.collisions, raw_collision_drop, noregret_collision_drop,
                  safe_collision_drop, one_shot.collision_cvar,
                  safe_noregret.collision_cvar, safe_cvar_drop);
    cv::putText(img, buf, cv::Point(12, PANEL_H - 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(225, 238, 245), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf),
                  "reach %d/%d -> %d/%d -> %d/%d -> %d/%d   residual %.2f%% -> %.2f%% -> %.2f%%   dual %.2f scale %.2f   intent top1 %.1f%%",
                  one_shot.reached, N_ROBOTS, raw_response.reached, N_ROBOTS,
                  noregret.reached, N_ROBOTS, safe_noregret.reached, N_ROBOTS,
                  raw_game.normalized_gain, noregret_game.normalized_gain,
                  safe_game.normalized_gain, last_safe.mean_safety_dual,
                  last_safe.safety_scale, intent_stats.top1_accuracy);
    cv::putText(img, buf, cv::Point(12, PANEL_H - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(225, 238, 245), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<RobotSpec> robots = make_robots();
    std::vector<int> top_intents;
    IntentStats intent_stats{};
    std::vector<float> beliefs = infer_intent_beliefs(robots, top_intents, intent_stats);

    std::vector<RolloutResult> cpu_one_shot;
    double cpu_one_shot_ms = evaluate_cpu_rollouts(robots, beliefs, nullptr, nullptr, 0,
                                                   cpu_one_shot);
    std::vector<int> cpu_one_shot_selected = select_best_by_robot(cpu_one_shot);
    std::vector<float> cpu_one_shot_x;
    std::vector<float> cpu_one_shot_y;
    reconstruct_paths(robots, beliefs, nullptr, nullptr, cpu_one_shot_selected, 0,
                      cpu_one_shot_x, cpu_one_shot_y);
    std::vector<double> cpu_pass_ms;
    cpu_pass_ms.push_back(cpu_one_shot_ms);
    double cpu_ms = cpu_one_shot_ms;
    std::vector<float> cpu_noregret_peer_x = cpu_one_shot_x;
    std::vector<float> cpu_noregret_peer_y = cpu_one_shot_y;
    std::vector<float> cpu_safe_peer_x = cpu_one_shot_x;
    std::vector<float> cpu_safe_peer_y = cpu_one_shot_y;
    std::vector<int> cpu_noregret_previous_selected = cpu_one_shot_selected;
    std::vector<int> cpu_safe_previous_selected = cpu_one_shot_selected;
    for (int pass = 1; pass < N_GAME_PASSES; pass++) {
        std::vector<RolloutResult> cpu_noregret_response;
        double cpu_response_ms = evaluate_cpu_rollouts(robots, beliefs,
                                                       cpu_noregret_peer_x.data(),
                                                       cpu_noregret_peer_y.data(), 1,
                                                       cpu_noregret_response);
        cpu_pass_ms.push_back(cpu_response_ms);
        cpu_ms += cpu_response_ms;
        std::vector<int> cpu_noregret_selected =
            select_best_by_robot(cpu_noregret_response);
        std::vector<float> cpu_noregret_response_x;
        std::vector<float> cpu_noregret_response_y;
        reconstruct_paths(robots, beliefs, cpu_noregret_peer_x.data(),
                          cpu_noregret_peer_y.data(), cpu_noregret_selected, 1,
                          cpu_noregret_response_x, cpu_noregret_response_y);
        std::vector<RolloutResult> cpu_safe_response = cpu_noregret_response;
        std::vector<int> cpu_safe_selected = cpu_noregret_selected;
        std::vector<float> cpu_safe_response_x = cpu_noregret_response_x;
        std::vector<float> cpu_safe_response_y = cpu_noregret_response_y;

        std::vector<float> cpu_noregret_mixed_x;
        std::vector<float> cpu_noregret_mixed_y;
        IterationStats cpu_noregret_stats{};
        mix_paths_regret_aware(cpu_noregret_peer_x, cpu_noregret_peer_y,
                               cpu_noregret_response_x, cpu_noregret_response_y,
                               cpu_noregret_response, cpu_noregret_previous_selected,
                               cpu_noregret_selected, pass, cpu_noregret_mixed_x,
                               cpu_noregret_mixed_y, cpu_noregret_stats);
        cpu_noregret_peer_x.swap(cpu_noregret_mixed_x);
        cpu_noregret_peer_y.swap(cpu_noregret_mixed_y);
        cpu_noregret_previous_selected.swap(cpu_noregret_selected);

        if (pass > 1) {
            cpu_safe_response.clear();
            double cpu_safe_ms = evaluate_cpu_rollouts(robots, beliefs,
                                                       cpu_safe_peer_x.data(),
                                                       cpu_safe_peer_y.data(), 1,
                                                       cpu_safe_response);
            cpu_pass_ms.push_back(cpu_safe_ms);
            cpu_ms += cpu_safe_ms;
            cpu_safe_selected = select_best_by_robot(cpu_safe_response);
            reconstruct_paths(robots, beliefs, cpu_safe_peer_x.data(),
                              cpu_safe_peer_y.data(), cpu_safe_selected, 1,
                              cpu_safe_response_x, cpu_safe_response_y);
        }

        std::vector<float> cpu_safe_mixed_x;
        std::vector<float> cpu_safe_mixed_y;
        IterationStats cpu_safe_stats{};
        mix_paths_learned_safety_prior(robots, cpu_safe_peer_x, cpu_safe_peer_y,
                                     cpu_safe_response_x, cpu_safe_response_y,
                                     cpu_safe_response, cpu_safe_previous_selected,
                                     cpu_safe_selected, pass, cpu_safe_mixed_x,
                                     cpu_safe_mixed_y, cpu_safe_stats);
        cpu_safe_peer_x.swap(cpu_safe_mixed_x);
        cpu_safe_peer_y.swap(cpu_safe_mixed_y);
        cpu_safe_previous_selected.swap(cpu_safe_selected);
    }

    RobotSpec* d_robots = nullptr;
    float* d_beliefs = nullptr;
    RolloutResult* d_one_shot = nullptr;
    RolloutResult* d_response = nullptr;
    float* d_peer_x = nullptr;
    float* d_peer_y = nullptr;
    float* d_sample_x = nullptr;
    float* d_sample_y = nullptr;
    int total = N_ROBOTS * ROLLOUTS_PER_ROBOT;
    int path_total = N_ROBOTS * HORIZON;
    int sample_total = SAMPLE_ROBOTS * SAMPLE_ROLLOUTS * HORIZON;
    CUDA_CHECK(cudaMalloc(&d_robots, N_ROBOTS * sizeof(RobotSpec)));
    CUDA_CHECK(cudaMalloc(&d_beliefs, N_ROBOTS * N_INTENTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_one_shot, total * sizeof(RolloutResult)));
    CUDA_CHECK(cudaMalloc(&d_response, total * sizeof(RolloutResult)));
    CUDA_CHECK(cudaMalloc(&d_peer_x, path_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_peer_y, path_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sample_x, sample_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sample_y, sample_total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_robots, robots.data(), N_ROBOTS * sizeof(RobotSpec),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beliefs, beliefs.data(), N_ROBOTS * N_INTENTS * sizeof(float),
                          cudaMemcpyHostToDevice));

    int blocks = (total + THREADS - 1) / THREADS;
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    auto launch_pass = [&](const float* peer_x,
                           const float* peer_y,
                           int best_response,
                           RolloutResult* d_out) {
        CUDA_CHECK(cudaEventRecord(ev0));
        rollout_kernel<<<blocks, THREADS>>>(d_robots, d_beliefs, peer_x, peer_y, best_response,
                                            d_out, d_sample_x, d_sample_y);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        return ms;
    };

    float gpu_one_shot_ms_f = launch_pass(nullptr, nullptr, 0, d_one_shot);
    std::vector<float> gpu_pass_ms;
    gpu_pass_ms.push_back(gpu_one_shot_ms_f);

    std::vector<RolloutResult> one_shot_results(total);
    CUDA_CHECK(cudaMemcpy(one_shot_results.data(), d_one_shot, total * sizeof(RolloutResult),
                          cudaMemcpyDeviceToHost));
    std::vector<int> one_shot_selected = select_best_by_robot(one_shot_results);
    std::vector<float> one_shot_x;
    std::vector<float> one_shot_y;
    reconstruct_paths(robots, beliefs, nullptr, nullptr, one_shot_selected, 0,
                      one_shot_x, one_shot_y);

    std::vector<float> noregret_peer_x = one_shot_x;
    std::vector<float> noregret_peer_y = one_shot_y;
    std::vector<float> safe_peer_x = one_shot_x;
    std::vector<float> safe_peer_y = one_shot_y;
    std::vector<int> noregret_previous_selected = one_shot_selected;
    std::vector<int> safe_previous_selected = one_shot_selected;
    std::vector<float> raw_x;
    std::vector<float> raw_y;
    std::vector<float> noregret_x = one_shot_x;
    std::vector<float> noregret_y = one_shot_y;
    std::vector<float> safe_x = one_shot_x;
    std::vector<float> safe_y = one_shot_y;
    GameStats raw_game{};
    GameStats noregret_game{};
    GameStats safe_game{};
    std::vector<IterationStats> noregret_stats;
    std::vector<IterationStats> safe_stats;

    for (int pass = 1; pass < N_GAME_PASSES; pass++) {
        CUDA_CHECK(cudaMemcpy(d_peer_x, noregret_peer_x.data(), path_total * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_peer_y, noregret_peer_y.data(), path_total * sizeof(float),
                              cudaMemcpyHostToDevice));
        float pass_ms = launch_pass(d_peer_x, d_peer_y, 1, d_response);
        gpu_pass_ms.push_back(pass_ms);

        std::vector<RolloutResult> noregret_response(total);
        CUDA_CHECK(cudaMemcpy(noregret_response.data(), d_response,
                              total * sizeof(RolloutResult), cudaMemcpyDeviceToHost));
        std::vector<int> noregret_selected = select_best_by_robot(noregret_response);
        std::vector<float> noregret_response_x;
        std::vector<float> noregret_response_y;
        reconstruct_paths(robots, beliefs, noregret_peer_x.data(), noregret_peer_y.data(),
                          noregret_selected, 1, noregret_response_x, noregret_response_y);
        GameStats noregret_pass_game =
            compute_game_stats(noregret_response, noregret_previous_selected,
                               noregret_selected);
        if (pass == 1) {
            raw_x = noregret_response_x;
            raw_y = noregret_response_y;
            raw_game = noregret_pass_game;
        }
        std::vector<RolloutResult> safe_response = noregret_response;
        std::vector<int> safe_selected = noregret_selected;
        std::vector<float> safe_response_x = noregret_response_x;
        std::vector<float> safe_response_y = noregret_response_y;
        GameStats safe_pass_game = noregret_pass_game;

        std::vector<float> noregret_mixed_x;
        std::vector<float> noregret_mixed_y;
        IterationStats noregret_pass_stats{};
        mix_paths_regret_aware(noregret_peer_x, noregret_peer_y, noregret_response_x,
                               noregret_response_y, noregret_response,
                               noregret_previous_selected, noregret_selected, pass,
                               noregret_mixed_x, noregret_mixed_y,
                               noregret_pass_stats);
        noregret_stats.push_back(noregret_pass_stats);
        noregret_peer_x.swap(noregret_mixed_x);
        noregret_peer_y.swap(noregret_mixed_y);
        noregret_previous_selected.swap(noregret_selected);
        noregret_x = noregret_peer_x;
        noregret_y = noregret_peer_y;
        noregret_game = noregret_pass_game;

        if (pass > 1) {
            CUDA_CHECK(cudaMemcpy(d_peer_x, safe_peer_x.data(), path_total * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_peer_y, safe_peer_y.data(), path_total * sizeof(float),
                                  cudaMemcpyHostToDevice));
            float safe_pass_ms = launch_pass(d_peer_x, d_peer_y, 1, d_response);
            gpu_pass_ms.push_back(safe_pass_ms);
            safe_response.resize(total);
            CUDA_CHECK(cudaMemcpy(safe_response.data(), d_response,
                                  total * sizeof(RolloutResult), cudaMemcpyDeviceToHost));
            safe_selected = select_best_by_robot(safe_response);
            reconstruct_paths(robots, beliefs, safe_peer_x.data(), safe_peer_y.data(),
                              safe_selected, 1, safe_response_x, safe_response_y);
            safe_pass_game = compute_game_stats(safe_response, safe_previous_selected,
                                                safe_selected);
        }

        std::vector<float> safe_mixed_x;
        std::vector<float> safe_mixed_y;
        IterationStats safe_pass_stats{};
        mix_paths_learned_safety_prior(robots, safe_peer_x, safe_peer_y,
                                     safe_response_x, safe_response_y, safe_response,
                                     safe_previous_selected, safe_selected, pass,
                                     safe_mixed_x, safe_mixed_y, safe_pass_stats);
        safe_stats.push_back(safe_pass_stats);
        safe_peer_x.swap(safe_mixed_x);
        safe_peer_y.swap(safe_mixed_y);
        safe_previous_selected.swap(safe_selected);
        safe_x = safe_peer_x;
        safe_y = safe_peer_y;
        safe_game = safe_pass_game;
    }

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    double gpu_ms = 0.0;
    for (float ms : gpu_pass_ms) gpu_ms += static_cast<double>(ms);

    std::vector<float> sample_x(sample_total);
    std::vector<float> sample_y(sample_total);
    CUDA_CHECK(cudaMemcpy(sample_x.data(), d_sample_x, sample_total * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sample_y.data(), d_sample_y, sample_total * sizeof(float),
                          cudaMemcpyDeviceToHost));

    TeamMetrics one_shot = compute_path_team_metrics(robots, one_shot_x, one_shot_y);
    TeamMetrics raw_response = compute_path_team_metrics(robots, raw_x, raw_y);
    TeamMetrics noregret = compute_path_team_metrics(robots, noregret_x, noregret_y);
    TeamMetrics safe_noregret = compute_path_team_metrics(robots, safe_x, safe_y);

    double speedup = gpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0;
    float raw_collision_drop = 100.0f * (1.0f - static_cast<float>(raw_response.collisions)
                                               / static_cast<float>(std::max(one_shot.collisions, 1)));
    float noregret_collision_drop = 100.0f * (1.0f - static_cast<float>(noregret.collisions)
                                                    / static_cast<float>(std::max(one_shot.collisions, 1)));
    float safe_collision_drop = 100.0f * (1.0f - static_cast<float>(safe_noregret.collisions)
                                                / static_cast<float>(std::max(one_shot.collisions, 1)));
    float noregret_risk_drop = 100.0f * (1.0f - noregret.mean_social_risk
                                               / std::max(one_shot.mean_social_risk, 1.0e-6f));
    float safe_risk_drop = 100.0f * (1.0f - safe_noregret.mean_social_risk
                                      / std::max(one_shot.mean_social_risk, 1.0e-6f));
    float noregret_cvar_drop = 100.0f * (1.0f - noregret.collision_cvar
                                               / std::max(one_shot.collision_cvar, 1.0e-6f));
    float safe_cvar_drop = 100.0f * (1.0f - safe_noregret.collision_cvar
                                           / std::max(one_shot.collision_cvar, 1.0e-6f));
    float noregret_sep_gain = noregret.min_separation - one_shot.min_separation;
    float safe_sep_gain = safe_noregret.min_separation - one_shot.min_separation;
    std::printf("Intent inference: top-1 %.1f%%, mean confidence %.3f, true-intent probability %.3f\n",
                intent_stats.top1_accuracy, intent_stats.mean_confidence,
                intent_stats.mean_true_probability);
    std::printf("CPU learned safety-dual prior game MPPI: %.3f ms (%d robots x %d rollouts x H=%d x %zu rollout batches)\n",
                cpu_ms, N_ROBOTS, ROLLOUTS_PER_ROBOT, HORIZON, cpu_pass_ms.size());
    std::printf("GPU learned safety-dual prior game graph MPPI: %.3f ms (one-shot+shared raw+2 no-regret+2 learned-prior best-response batches, %.1fx vs CPU equivalent rollout eval; %zu GPU batches)\n",
                gpu_ms, speedup, gpu_pass_ms.size());
    std::printf("One-shot team: collisions %d, reached %d/%d, deadlocks %d, min sep %.3f, terminal avg %.3f, social risk avg/max %.3f/%.3f, collision CVaR %.3f\n",
                one_shot.collisions, one_shot.reached, N_ROBOTS, one_shot.deadlocks,
                one_shot.min_separation, one_shot.mean_terminal,
                one_shot.mean_social_risk, one_shot.max_social_risk, one_shot.collision_cvar);
    std::printf("Raw best-response team: collisions %d, reached %d/%d, deadlocks %d, min sep %.3f, terminal avg %.3f, social risk avg/max %.3f/%.3f, collision CVaR %.3f, unilateral gain %.3f (%.2f%%, max %.3f), collision reduction %.1f%%\n",
                raw_response.collisions, raw_response.reached, N_ROBOTS,
                raw_response.deadlocks, raw_response.min_separation,
                raw_response.mean_terminal, raw_response.mean_social_risk,
                raw_response.max_social_risk, raw_response.collision_cvar,
                raw_game.mean_unilateral_gain, raw_game.normalized_gain,
                raw_game.max_unilateral_gain, raw_collision_drop);
    std::printf("No-regret team: collisions %d, reached %d/%d, deadlocks %d, min sep %.3f, terminal avg %.3f, social risk avg/max %.3f/%.3f, collision CVaR %.3f, final residual %.3f (%.2f%%, max %.3f), collision reduction %.1f%%, risk reduction %.1f%%, CVaR reduction %.1f%%, separation gain %.3f\n",
                noregret.collisions, noregret.reached, N_ROBOTS, noregret.deadlocks,
                noregret.min_separation, noregret.mean_terminal,
                noregret.mean_social_risk, noregret.max_social_risk,
                noregret.collision_cvar, noregret_game.mean_unilateral_gain,
                noregret_game.normalized_gain, noregret_game.max_unilateral_gain,
                noregret_collision_drop, noregret_risk_drop, noregret_cvar_drop,
                noregret_sep_gain);
    std::printf("Learned-prior safe team: collisions %d, reached %d/%d, deadlocks %d, min sep %.3f, terminal avg %.3f, social risk avg/max %.3f/%.3f, collision CVaR %.3f, final residual %.3f (%.2f%%, max %.3f), collision reduction %.1f%%, risk reduction %.1f%%, CVaR reduction %.1f%%, separation gain %.3f\n",
                safe_noregret.collisions, safe_noregret.reached, N_ROBOTS,
                safe_noregret.deadlocks, safe_noregret.min_separation,
                safe_noregret.mean_terminal, safe_noregret.mean_social_risk,
                safe_noregret.max_social_risk, safe_noregret.collision_cvar,
                safe_game.mean_unilateral_gain, safe_game.normalized_gain,
                safe_game.max_unilateral_gain, safe_collision_drop, safe_risk_drop,
                safe_cvar_drop, safe_sep_gain);
    for (size_t i = 0; i < noregret_stats.size(); i++) {
        const IterationStats& s = noregret_stats[i];
        std::printf("Pass %zu regret-matched update: mean path delta %.3f, max path delta %.3f, alpha avg/range %.3f [%.3f, %.3f], positive regret avg/max %.3f/%.3f, unilateral residual %.3f (%.2f%%)\n",
                    i + 1, s.mean_path_delta, s.max_path_delta,
                    s.mean_alpha, s.min_alpha, s.max_alpha,
                    s.mean_positive_regret, s.max_positive_regret,
                    s.mean_unilateral_gain, s.normalized_gain);
    }
    for (size_t i = 0; i < safe_stats.size(); i++) {
        const IterationStats& s = safe_stats[i];
        std::printf("Pass %zu learned safety-dual update: mean path delta %.3f, max path delta %.3f, alpha avg/range %.3f [%.3f, %.3f], dual avg/max %.3f/%.3f, prior scale avg/max %.3f/%.3f, margin avg %.3f, violation avg/max %.3f/%.3f, CVaR %.3f -> %.3f, chosen scale %.2f, residual %.3f (%.2f%%)\n",
                    i + 1, s.mean_path_delta, s.max_path_delta,
                    s.mean_alpha, s.min_alpha, s.max_alpha,
                    s.mean_safety_dual, s.max_safety_dual,
                    s.mean_prior_scale, s.max_prior_scale,
                    s.mean_prior_margin,
                    s.mean_safety_violation, s.max_safety_violation,
                    s.cvar_before, s.cvar_after, s.safety_scale,
                    s.mean_unilateral_gain, s.normalized_gain);
    }

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_learned_safety_dual_graph_mppi.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_learned_safety_dual_graph_mppi.avi\n");
        return 1;
    }
    for (int k = 2; k <= HORIZON; k += 2) {
        video.write(draw_frame(robots, one_shot_x, one_shot_y, raw_x, raw_y,
                               noregret_x, noregret_y, safe_x, safe_y, sample_x,
                               sample_y, one_shot, raw_response, noregret,
                               safe_noregret, intent_stats, raw_game, noregret_game,
                               safe_game, noregret_stats, safe_stats, gpu_ms, cpu_ms, k));
    }
    for (int i = 0; i < 12; i++) {
        video.write(draw_frame(robots, one_shot_x, one_shot_y, raw_x, raw_y,
                               noregret_x, noregret_y, safe_x, safe_y, sample_x,
                               sample_y, one_shot, raw_response, noregret,
                               safe_noregret, intent_stats, raw_game, noregret_game,
                               safe_game, noregret_stats, safe_stats, gpu_ms, cpu_ms,
                               HORIZON));
    }
    video.release();

    avi_to_gif("gif/gpu_learned_safety_dual_graph_mppi.avi",
               "gif/gpu_learned_safety_dual_graph_mppi.gif", 8, 720);
    std::printf("GIF saved to gif/gpu_learned_safety_dual_graph_mppi.gif\n");

    CUDA_CHECK(cudaFree(d_robots));
    CUDA_CHECK(cudaFree(d_beliefs));
    CUDA_CHECK(cudaFree(d_one_shot));
    CUDA_CHECK(cudaFree(d_response));
    CUDA_CHECK(cudaFree(d_peer_x));
    CUDA_CHECK(cudaFree(d_peer_y));
    CUDA_CHECK(cudaFree(d_sample_x));
    CUDA_CHECK(cudaFree(d_sample_y));
    return 0;
}
