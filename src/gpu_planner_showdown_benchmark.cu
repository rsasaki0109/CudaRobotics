// gpu_planner_showdown_benchmark.cu
//
// GPU planner showdown benchmark.
//
// A batch of robots crosses a shared interaction zone.  This benchmark names a
// concrete enemy for the planner stack: beat ORCA-like reciprocal avoidance and
// priority-graph yielding while staying under hard target gates for collision
// count, CVaR, reach, deadlock, residual regret, and GPU runtime.  The final
// planner is a trainable safety-dual MPPI game update with a learned
// safety-pressure controller and tiny CPU-trained MLP priors distilled from
// synthetic graph-risk and metric labels.
//
// Output: gif/gpu_planner_showdown_benchmark.gif
//         gif/gpu_planner_showdown_benchmark.json

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
constexpr int PRIOR_INPUTS = 9;
constexpr int PRIOR_HIDDEN = 10;
constexpr int PRIOR_OUTPUTS = 3;
constexpr int PRIOR_EPOCHS = 180;
constexpr int PRESSURE_INPUTS = 10;
constexpr int PRESSURE_HIDDEN = 10;
constexpr int PRESSURE_EPOCHS = 180;
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
constexpr int SAFETY_COLLISION_TARGET = 132;
constexpr int SHOWDOWN_COLLISION_TARGET = 8;
constexpr int SHOWDOWN_REACH_TARGET = N_ROBOTS;
constexpr int SHOWDOWN_DEADLOCK_TARGET = 0;
constexpr float SHOWDOWN_CVAR_TARGET = 26.5f;
constexpr float SHOWDOWN_RESIDUAL_TARGET = 12.0f;
constexpr float SHOWDOWN_RUNTIME_TARGET_MS = 15.0f;
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

struct PriorSample {
    float x[PRIOR_INPUTS];
    float y[PRIOR_OUTPUTS];
};

struct SafetyDualNetwork {
    float w1[PRIOR_HIDDEN][PRIOR_INPUTS];
    float b1[PRIOR_HIDDEN];
    float w2[PRIOR_OUTPUTS][PRIOR_HIDDEN];
    float b2[PRIOR_OUTPUTS];
};

struct PriorTrainingStats {
    int samples;
    float initial_loss;
    float final_loss;
};

struct PressureSample {
    float x[PRESSURE_INPUTS];
    float y;
};

struct PressureController {
    float w1[PRESSURE_HIDDEN][PRESSURE_INPUTS];
    float b1[PRESSURE_HIDDEN];
    float w2[PRESSURE_HIDDEN];
    float b2;
};

struct PressureTrainingStats {
    int samples;
    float initial_loss;
    float final_loss;
};

struct PressureContext {
    float lane_tightness;
    float conflict_density;
    float cross_shift_load;
    float priority_flip;
};

struct ShowdownRow {
    const char* name;
    TeamMetrics metrics;
    double runtime_ms;
    float residual_pct;
};

struct ScenarioConfig {
    std::string name;
    float lane_scale;
    float jitter_scale;
    float cross_shift;
    bool priority_flip;
};

enum class PressureMode {
    Learned,
    Teacher,
    None,
};

enum class BudgetMode {
    Learned,
    Off,
};

struct BudgetDecision {
    BudgetMode mode;
    bool extra_pass;
    bool accepted_extra;
    int decision_pass;
    float score;
    float context_difficulty;
    float cvar_after_decision;
    float residual_after_decision;
    float refinement_score_before;
    float refinement_score_after;
    float refinement_score_delta;
    float estimated_extra_ms;
    double fixed_gpu_ms;
    double final_gpu_ms;
};

struct CliOptions {
    bool check_targets;
    bool no_video;
    bool help;
    std::string json_path;
    ScenarioConfig scenario;
    PressureMode pressure_mode;
    BudgetMode budget_mode;
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

static std::vector<RobotSpec> make_robots(const ScenarioConfig& scenario) {
    std::vector<RobotSpec> robots(N_ROBOTS);
    for (int i = 0; i < N_ROBOTS; i++) {
        int group = i % 4;
        int lane = i / 4;
        float lane_center = 0.5f * static_cast<float>((N_ROBOTS / 4) - 1);
        float lane_f = lane_center
            + (static_cast<float>(lane) - lane_center) * scenario.lane_scale;
        float jitter = scenario.jitter_scale * 0.13f
            * std::sin(1.37f * static_cast<float>(i));
        float base_priority = 0.35f
            + 0.65f * static_cast<float>((i * 7) % 11) / 10.0f;
        RobotSpec r{};
        r.route = group;
        r.priority = scenario.priority_flip
            ? clampf(1.35f - base_priority, 0.35f, 1.0f)
            : base_priority;
        r.lane = lane_f;
        r.jitter = jitter;
        if (group == 0) {
            r.sx = 0.40f + 0.055f * lane_f;
            r.sy = 3.95f + 0.15f * lane_f + jitter - scenario.cross_shift;
            r.gx = 17.38f;
            r.gy = 7.05f - 0.14f * lane_f - jitter + scenario.cross_shift;
            r.theta0 = 0.05f;
        } else if (group == 1) {
            r.sx = 7.45f + 0.16f * lane_f + jitter - scenario.cross_shift;
            r.sy = 0.40f + 0.045f * lane_f;
            r.gx = 10.62f - 0.16f * lane_f - jitter + scenario.cross_shift;
            r.gy = 10.36f;
            r.theta0 = 1.50f;
        } else if (group == 2) {
            r.sx = 17.60f - 0.055f * lane_f;
            r.sy = 7.12f - 0.15f * lane_f + jitter + scenario.cross_shift;
            r.gx = 0.62f;
            r.gy = 3.98f + 0.14f * lane_f - jitter - scenario.cross_shift;
            r.theta0 = PI_F - 0.05f;
        } else {
            r.sx = 10.62f - 0.16f * lane_f + jitter + scenario.cross_shift;
            r.sy = 10.60f - 0.045f * lane_f;
            r.gx = 7.45f + 0.16f * lane_f - jitter - scenario.cross_shift;
            r.gy = 0.62f;
            r.theta0 = -1.58f;
        }
        intent_goal(r, r.route, r.gx, r.gy);
        robots[i] = r;
    }
    return robots;
}

static PressureContext pressure_context_from_scenario(
    const ScenarioConfig& scenario,
    const std::vector<RobotSpec>& robots) {
    PressureContext ctx{};
    ctx.lane_tightness = clampf((1.0f - scenario.lane_scale) / 0.58f,
                                0.0f, 1.25f);
    ctx.cross_shift_load = clampf(std::fabs(scenario.cross_shift) / 0.18f,
                                  0.0f, 1.25f);
    ctx.priority_flip = scenario.priority_flip ? 1.0f : 0.0f;

    float conflict_sum = 0.0f;
    int conflict_pairs = 0;
    for (int i = 0; i < N_ROBOTS; i++) {
        float ix;
        float iy;
        route_point(robots[i], 0.50f, ix, iy);
        for (int j = i + 1; j < N_ROBOTS; j++) {
            if (robots[i].route == robots[j].route) continue;
            float jx;
            float jy;
            route_point(robots[j], 0.50f, jx, jy);
            float d = std::sqrt(sqr(ix - jx) + sqr(iy - jy));
            float sep = d - 2.0f * ROBOT_R;
            conflict_sum += clampf((1.36f - sep) / 1.36f, 0.0f, 1.0f);
            conflict_pairs++;
        }
    }
    ctx.conflict_density = clampf(conflict_sum
                                  / std::max(1.0f, 0.78f * static_cast<float>(conflict_pairs)),
                                  0.0f, 1.35f);
    return ctx;
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

static void route_unit_direction(const RobotSpec& r,
                                 float progress,
                                 float& ux,
                                 float& uy) {
    float x0;
    float y0;
    float x1;
    float y1;
    route_point(r, progress, x0, y0);
    route_point(r, progress + 0.045f, x1, y1);
    float dx = x1 - x0;
    float dy = y1 - y0;
    float inv = 1.0f / std::sqrt(std::max(dx * dx + dy * dy, 1.0e-6f));
    ux = dx * inv;
    uy = dy * inv;
}

static void add_static_clearance_push(float x,
                                      float y,
                                      float gain,
                                      float& ax,
                                      float& ay,
                                      float& slow) {
    float clear = static_clearance(x, y);
    if (clear >= 0.68f) return;
    float gx = static_clearance(x + 0.05f, y) - static_clearance(x - 0.05f, y);
    float gy = static_clearance(x, y + 0.05f) - static_clearance(x, y - 0.05f);
    float gnorm = std::sqrt(std::max(gx * gx + gy * gy, 1.0e-6f));
    float pressure = clampf((0.68f - clear) / 0.68f, 0.0f, 1.0f);
    ax += gain * pressure * gx / gnorm;
    ay += gain * pressure * gy / gnorm;
    slow += 0.20f * pressure;
}

static void project_tail_risk_clearance(const std::vector<RobotSpec>& robots,
                                        std::vector<float>& path_x,
                                        std::vector<float>& path_y,
                                        int passes,
                                        float extra_clearance) {
    float target_center_dist = 2.0f * ROBOT_R + 0.24f + extra_clearance;
    constexpr float ROUTE_PULL = 0.10f;
    for (int iter = 0; iter < passes; iter++) {
        for (int k = 1; k < HORIZON - 1; k++) {
            float phase = static_cast<float>(k) / static_cast<float>(HORIZON - 1);
            float taper = 0.35f + 0.65f * sinf(PI_F * phase);
            for (int i = 0; i < N_ROBOTS; i++) {
                int idx_i = i * HORIZON + k;
                for (int j = i + 1; j < N_ROBOTS; j++) {
                    if (robots[i].route == robots[j].route) continue;
                    int idx_j = j * HORIZON + k;
                    float dx = path_x[idx_i] - path_x[idx_j];
                    float dy = path_y[idx_i] - path_y[idx_j];
                    float d2 = dx * dx + dy * dy;
                    float d = sqrtf(std::max(d2, 1.0e-8f));
                    if (d >= target_center_dist) continue;
                    float ux = dx / d;
                    float uy = dy / d;
                    float deficit = target_center_dist - d;
                    float priority_sum = robots[i].priority + robots[j].priority + 1.0e-6f;
                    float wi = clampf(robots[j].priority / priority_sum, 0.28f, 0.72f);
                    float wj = clampf(robots[i].priority / priority_sum, 0.28f, 0.72f);
                    float step = 0.32f * taper * deficit;
                    path_x[idx_i] += step * wi * ux;
                    path_y[idx_i] += step * wi * uy;
                    path_x[idx_j] -= step * wj * ux;
                    path_y[idx_j] -= step * wj * uy;
                }
            }
            for (int i = 0; i < N_ROBOTS; i++) {
                int idx = i * HORIZON + k;
                float progress = 0.0f;
                float route_err = route_distance(robots[i], path_x[idx], path_y[idx],
                                                 &progress);
                float rx;
                float ry;
                route_point(robots[i], progress, rx, ry);
                float pull = ROUTE_PULL * clampf(route_err / 1.35f, 0.0f, 1.0f);
                path_x[idx] = (1.0f - pull) * path_x[idx] + pull * rx;
                path_y[idx] = (1.0f - pull) * path_y[idx] + pull * ry;

                float ax = 0.0f;
                float ay = 0.0f;
                float slow = 0.0f;
                add_static_clearance_push(path_x[idx], path_y[idx], 0.18f, ax, ay, slow);
                path_x[idx] = clampf(path_x[idx] + 0.05f * ax, 0.05f, WORLD_W - 0.05f);
                path_y[idx] = clampf(path_y[idx] + 0.05f * ay, 0.05f, WORLD_H - 0.05f);
            }
        }
    }
}

static void advance_baseline_state(Pose2& s,
                                   float vx,
                                   float vy,
                                   float speed,
                                   float& prev_speed,
                                   float& prev_steer) {
    float heading = atan2f(vy, vx);
    float heading_error = wrap_angle(heading - s.theta);
    float steer = clampf(0.92f * fast_tanh(1.55f * heading_error),
                         -MAX_STEER, MAX_STEER);
    speed = clampf(0.72f * prev_speed + 0.28f * speed, MIN_SPEED, MAX_SPEED);
    steer = clampf(0.64f * prev_steer + 0.36f * steer, -MAX_STEER, MAX_STEER);
    s.x += speed * cosf(s.theta) * DT;
    s.y += speed * sinf(s.theta) * DT;
    s.theta = wrap_angle(s.theta + speed * steer / WHEEL_BASE * DT);
    s.x = clampf(s.x, 0.05f, WORLD_W - 0.05f);
    s.y = clampf(s.y, 0.05f, WORLD_H - 0.05f);
    prev_speed = speed;
    prev_steer = steer;
}

static void make_orca_like_paths(const std::vector<RobotSpec>& robots,
                                 std::vector<float>& path_x,
                                 std::vector<float>& path_y) {
    path_x.assign(N_ROBOTS * HORIZON, 0.0f);
    path_y.assign(N_ROBOTS * HORIZON, 0.0f);
    std::vector<Pose2> state(N_ROBOTS);
    std::vector<float> prev_speed(N_ROBOTS, 0.72f);
    std::vector<float> prev_steer(N_ROBOTS, 0.0f);
    for (int i = 0; i < N_ROBOTS; i++) {
        state[i] = Pose2{robots[i].sx, robots[i].sy, robots[i].theta0};
    }

    for (int k = 0; k < HORIZON; k++) {
        std::vector<float> des_x(N_ROBOTS, 0.0f);
        std::vector<float> des_y(N_ROBOTS, 0.0f);
        std::vector<float> des_speed(N_ROBOTS, 1.0f);
        for (int i = 0; i < N_ROBOTS; i++) {
            float progress = 0.0f;
            route_distance(robots[i], state[i].x, state[i].y, &progress);
            float tx;
            float ty;
            route_point(robots[i], progress + 0.12f, tx, ty);
            float dx = tx - state[i].x;
            float dy = ty - state[i].y;
            float inv = 1.0f / std::sqrt(std::max(dx * dx + dy * dy, 1.0e-6f));
            des_x[i] = dx * inv;
            des_y[i] = dy * inv;
            float terminal = std::sqrt(sqr(robots[i].gx - state[i].x)
                                     + sqr(robots[i].gy - state[i].y));
            des_speed[i] = clampf(1.16f + 0.22f * fast_sigmoid(terminal - 1.2f),
                                  0.70f, MAX_SPEED);
        }

        for (int i = 0; i < N_ROBOTS; i++) {
            float ax = 0.0f;
            float ay = 0.0f;
            float slow = 0.0f;
            for (int j = 0; j < N_ROBOTS; j++) {
                if (i == j) continue;
                float dx = state[i].x - state[j].x;
                float dy = state[i].y - state[j].y;
                float d2 = dx * dx + dy * dy;
                float d = std::sqrt(std::max(d2, 1.0e-6f));
                if (d > 3.45f) continue;
                float ux = dx / d;
                float uy = dy / d;
                float rvx = des_speed[i] * des_x[i] - des_speed[j] * des_x[j];
                float rvy = des_speed[i] * des_y[i] - des_speed[j] * des_y[j];
                float closing = clampf(-(ux * rvx + uy * rvy), 0.0f, 1.0f);
                float route_cross = robots[i].route == robots[j].route ? 0.28f : 1.0f;
                float near = 1.0f - clampf((d - 0.42f) / 3.03f, 0.0f, 1.0f);
                float threat = route_cross * near * near * (0.36f + 0.94f * closing);
                ax += threat * ux;
                ay += threat * uy;
                slow += 0.18f * threat;
            }
            add_static_clearance_push(state[i].x, state[i].y, 0.72f, ax, ay, slow);
            float vx = des_speed[i] * des_x[i] + 1.18f * ax;
            float vy = des_speed[i] * des_y[i] + 1.18f * ay;
            float norm = std::sqrt(std::max(vx * vx + vy * vy, 1.0e-6f));
            vx /= norm;
            vy /= norm;
            float speed = clampf(des_speed[i] * (1.0f - clampf(slow, 0.0f, 0.42f)),
                                 MIN_SPEED, MAX_SPEED);
            advance_baseline_state(state[i], vx, vy, speed, prev_speed[i], prev_steer[i]);
            path_x[i * HORIZON + k] = state[i].x;
            path_y[i * HORIZON + k] = state[i].y;
        }
    }
}

static void make_priority_graph_paths(const std::vector<RobotSpec>& robots,
                                      std::vector<float>& path_x,
                                      std::vector<float>& path_y) {
    path_x.assign(N_ROBOTS * HORIZON, 0.0f);
    path_y.assign(N_ROBOTS * HORIZON, 0.0f);
    std::vector<Pose2> state(N_ROBOTS);
    std::vector<float> prev_speed(N_ROBOTS, 0.74f);
    std::vector<float> prev_steer(N_ROBOTS, 0.0f);
    for (int i = 0; i < N_ROBOTS; i++) {
        state[i] = Pose2{robots[i].sx, robots[i].sy, robots[i].theta0};
    }

    for (int k = 0; k < HORIZON; k++) {
        std::vector<float> route_ux(N_ROBOTS, 0.0f);
        std::vector<float> route_uy(N_ROBOTS, 0.0f);
        std::vector<float> progress(N_ROBOTS, 0.0f);
        for (int i = 0; i < N_ROBOTS; i++) {
            route_distance(robots[i], state[i].x, state[i].y, &progress[i]);
            route_unit_direction(robots[i], progress[i], route_ux[i], route_uy[i]);
        }

        for (int i = 0; i < N_ROBOTS; i++) {
            float tx;
            float ty;
            route_point(robots[i], progress[i] + 0.13f + 0.025f * robots[i].priority,
                        tx, ty);
            float dx = tx - state[i].x;
            float dy = ty - state[i].y;
            float inv = 1.0f / std::sqrt(std::max(dx * dx + dy * dy, 1.0e-6f));
            float vx = dx * inv;
            float vy = dy * inv;
            float slow = 0.0f;
            float ax = 0.0f;
            float ay = 0.0f;

            for (int j = 0; j < N_ROBOTS; j++) {
                if (i == j || robots[i].route == robots[j].route) continue;
                float rel_priority = robots[j].priority - robots[i].priority;
                float rx = state[j].x - state[i].x;
                float ry = state[j].y - state[i].y;
                float d = std::sqrt(std::max(rx * rx + ry * ry, 1.0e-6f));
                if (d > 4.15f || rel_priority <= -0.08f) continue;
                float toward_j = clampf((rx * route_ux[i] + ry * route_uy[i]) / d, 0.0f, 1.0f);
                float cross_axis = fabsf(route_ux[i] * route_uy[j] - route_uy[i] * route_ux[j]);
                float near = 1.0f - clampf((d - 0.48f) / 3.67f, 0.0f, 1.0f);
                float yield = (0.34f + 0.92f * clampf(rel_priority + 0.18f, 0.0f, 1.0f))
                            * near * near * (0.35f + 0.65f * toward_j)
                            * (0.55f + 0.45f * cross_axis);
                slow += 0.42f * yield;
                ax -= 0.46f * yield * route_ux[i];
                ay -= 0.46f * yield * route_uy[i];
                ax += 0.34f * yield * (state[i].x - state[j].x) / d;
                ay += 0.34f * yield * (state[i].y - state[j].y) / d;
            }

            add_static_clearance_push(state[i].x, state[i].y, 0.62f, ax, ay, slow);
            vx += ax;
            vy += ay;
            float norm = std::sqrt(std::max(vx * vx + vy * vy, 1.0e-6f));
            vx /= norm;
            vy /= norm;
            float terminal = std::sqrt(sqr(robots[i].gx - state[i].x)
                                     + sqr(robots[i].gy - state[i].y));
            float base_speed = clampf(1.02f + 0.40f * robots[i].priority
                                            + 0.12f * fast_sigmoid(terminal - 1.0f),
                                      0.68f, MAX_SPEED);
            float speed = clampf(base_speed * (1.0f - clampf(slow, 0.0f, 0.58f)),
                                 MIN_SPEED, MAX_SPEED);
            advance_baseline_state(state[i], vx, vy, speed, prev_speed[i], prev_steer[i]);
            path_x[i * HORIZON + k] = state[i].x;
            path_y[i * HORIZON + k] = state[i].y;
        }
    }
}

static bool showdown_target_pass(const ShowdownRow& row) {
    if (row.residual_pct < 0.0f) return false;
    return row.metrics.reached >= SHOWDOWN_REACH_TARGET
        && row.metrics.deadlocks <= SHOWDOWN_DEADLOCK_TARGET
        && row.metrics.collisions <= SHOWDOWN_COLLISION_TARGET
        && row.metrics.collision_cvar <= SHOWDOWN_CVAR_TARGET
        && row.residual_pct <= SHOWDOWN_RESIDUAL_TARGET
        && row.runtime_ms <= SHOWDOWN_RUNTIME_TARGET_MS;
}

static void print_showdown_row(const ShowdownRow& row) {
    char residual[32];
    if (row.residual_pct < 0.0f) {
        std::snprintf(residual, sizeof(residual), "n/a");
    } else {
        std::snprintf(residual, sizeof(residual), "%.2f%%", row.residual_pct);
    }
    std::printf("%-31s | coll %3d | reach %2d/%2d | dead %d | CVaR %6.2f | residual %7s | runtime %7.3f ms | %s\n",
                row.name, row.metrics.collisions, row.metrics.reached, N_ROBOTS,
                row.metrics.deadlocks, row.metrics.collision_cvar, residual,
                row.runtime_ms, showdown_target_pass(row) ? "TARGET PASS" : "target miss");
}

static ScenarioConfig scenario_config(const std::string& name) {
    if (name == "baseline") {
        return ScenarioConfig{"baseline", 1.0f, 1.0f, 0.0f, false};
    }
    if (name == "tight") {
        return ScenarioConfig{"tight", 0.78f, 1.20f, -0.12f, false};
    }
    if (name == "priority_flip" || name == "priority-flip") {
        return ScenarioConfig{"priority_flip", 0.92f, 1.05f, 0.08f, true};
    }
    if (name == "adversarial_density" || name == "adversarial-density") {
        return ScenarioConfig{"adversarial_density", 0.42f, 0.35f, 0.18f, true};
    }
    return ScenarioConfig{"", 0.0f, 0.0f, 0.0f, false};
}

static const char* pressure_mode_name(PressureMode mode) {
    switch (mode) {
        case PressureMode::Learned:
            return "learned";
        case PressureMode::Teacher:
            return "teacher";
        case PressureMode::None:
            return "none";
    }
    return "learned";
}

static bool parse_pressure_mode(const std::string& name, PressureMode& mode) {
    if (name == "learned") {
        mode = PressureMode::Learned;
        return true;
    }
    if (name == "teacher") {
        mode = PressureMode::Teacher;
        return true;
    }
    if (name == "none" || name == "off" || name == "no-pressure") {
        mode = PressureMode::None;
        return true;
    }
    return false;
}

static const char* budget_mode_name(BudgetMode mode) {
    switch (mode) {
        case BudgetMode::Learned:
            return "learned";
        case BudgetMode::Off:
            return "off";
    }
    return "learned";
}

static bool parse_budget_mode(const std::string& name, BudgetMode& mode) {
    if (name == "learned" || name == "adaptive") {
        mode = BudgetMode::Learned;
        return true;
    }
    if (name == "off" || name == "fixed" || name == "none") {
        mode = BudgetMode::Off;
        return true;
    }
    return false;
}

static void print_usage(const char* argv0) {
    std::printf("Usage: %s [--check] [--no-video] [--scenario baseline|tight|priority_flip|adversarial_density] [--pressure-mode learned|teacher|none] [--adaptive-budget learned|off] [--json PATH]\n", argv0);
    std::printf("  --check       return non-zero when the trainable safety-dual row misses target gates\n");
    std::printf("  --no-video    skip AVI/GIF rendering and only emit stdout + JSON metrics\n");
    std::printf("  --scenario N  choose scenario: baseline (default), tight, priority_flip, adversarial_density\n");
    std::printf("  --pressure-mode N  choose safety pressure: learned (default), teacher, none\n");
    std::printf("  --adaptive-budget N  choose pass budget: learned (default), off\n");
    std::printf("  --json PATH   write machine-readable showdown metrics (default gif/gpu_planner_showdown_benchmark.json)\n");
}

static bool parse_cli(int argc, char** argv, CliOptions& opts) {
    opts.check_targets = false;
    opts.no_video = false;
    opts.help = false;
    opts.json_path = "gif/gpu_planner_showdown_benchmark.json";
    opts.scenario = scenario_config("baseline");
    opts.pressure_mode = PressureMode::Learned;
    opts.budget_mode = BudgetMode::Learned;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--check") {
            opts.check_targets = true;
        } else if (arg == "--no-video") {
            opts.no_video = true;
        } else if (arg == "--scenario" && i + 1 < argc) {
            opts.scenario = scenario_config(argv[++i]);
            if (opts.scenario.name.empty()) {
                std::fprintf(stderr, "unknown scenario: %s\n", argv[i]);
                return false;
            }
        } else if (arg == "--pressure-mode" && i + 1 < argc) {
            std::string mode_name = argv[++i];
            if (!parse_pressure_mode(mode_name, opts.pressure_mode)) {
                std::fprintf(stderr, "unknown pressure mode: %s\n", mode_name.c_str());
                return false;
            }
        } else if (arg == "--adaptive-budget" && i + 1 < argc) {
            std::string mode_name = argv[++i];
            if (!parse_budget_mode(mode_name, opts.budget_mode)) {
                std::fprintf(stderr, "unknown adaptive budget mode: %s\n", mode_name.c_str());
                return false;
            }
        } else if (arg == "--json" && i + 1 < argc) {
            opts.json_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            opts.help = true;
            return true;
        } else {
            std::fprintf(stderr, "unknown or incomplete option: %s\n", arg.c_str());
            return false;
        }
    }
    return true;
}

static void write_row_json(FILE* fp, const ShowdownRow& row, bool comma) {
    std::fprintf(fp,
                 "    {\"name\":\"%s\",\"collisions\":%d,\"reached\":%d,"
                 "\"deadlocks\":%d,\"min_separation\":%.6f,"
                 "\"mean_terminal\":%.6f,\"mean_social_risk\":%.6f,"
                 "\"max_social_risk\":%.6f,\"collision_cvar\":%.6f,"
                 "\"runtime_ms\":%.6f,\"residual_pct\":%.6f,"
                 "\"target_pass\":%s}%s\n",
                 row.name, row.metrics.collisions, row.metrics.reached,
                 row.metrics.deadlocks, row.metrics.min_separation,
                 row.metrics.mean_terminal, row.metrics.mean_social_risk,
                 row.metrics.max_social_risk, row.metrics.collision_cvar,
                 row.runtime_ms, row.residual_pct,
                 showdown_target_pass(row) ? "true" : "false",
                 comma ? "," : "");
}

static bool write_showdown_json(const std::string& path,
                                const ScenarioConfig& scenario,
                                PressureMode pressure_mode,
                                const PressureContext& pressure_context,
                                const std::vector<ShowdownRow>& rows,
                                const PriorTrainingStats& prior_training,
                                const PressureTrainingStats& pressure_training,
                                const BudgetDecision& budget_decision,
                                const IntentStats& intent_stats,
                                double cpu_ms,
                                double gpu_ms,
                                double speedup,
                                bool target_pass) {
    FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) return false;
    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"schema_version\":1,\n");
    std::fprintf(fp, "  \"benchmark\":\"gpu_planner_showdown_benchmark\",\n");
    std::fprintf(fp, "  \"scenario\":\"%s\",\n", scenario.name.c_str());
    std::fprintf(fp,
                 "  \"scenario_config\":{\"lane_scale\":%.6f,"
                 "\"jitter_scale\":%.6f,\"cross_shift\":%.6f,"
                 "\"priority_flip\":%s},\n",
                 scenario.lane_scale, scenario.jitter_scale,
                 scenario.cross_shift,
                 scenario.priority_flip ? "true" : "false");
    std::fprintf(fp, "  \"pressure_mode\":\"%s\",\n",
                 pressure_mode_name(pressure_mode));
    std::fprintf(fp,
                 "  \"pressure_context\":{\"lane_tightness\":%.6f,"
                 "\"conflict_density\":%.6f,\"cross_shift_load\":%.6f,"
                 "\"priority_flip\":%.6f},\n",
                 pressure_context.lane_tightness,
                 pressure_context.conflict_density,
                 pressure_context.cross_shift_load,
                 pressure_context.priority_flip);
    std::fprintf(fp, "  \"target_pass\":%s,\n", target_pass ? "true" : "false");
    std::fprintf(fp,
                 "  \"hard_target\":{\"reach\":%d,\"deadlocks_max\":%d,"
                 "\"collisions_max\":%d,\"collision_cvar_max\":%.6f,"
                 "\"residual_pct_max\":%.6f,\"runtime_ms_max\":%.6f},\n",
                 SHOWDOWN_REACH_TARGET, SHOWDOWN_DEADLOCK_TARGET,
                 SHOWDOWN_COLLISION_TARGET, SHOWDOWN_CVAR_TARGET,
                 SHOWDOWN_RESIDUAL_TARGET, SHOWDOWN_RUNTIME_TARGET_MS);
    std::fprintf(fp,
                 "  \"training\":{\"samples\":%d,\"initial_loss\":%.8f,"
                 "\"final_loss\":%.8f,\"epochs\":%d},\n",
                 prior_training.samples, prior_training.initial_loss,
                 prior_training.final_loss, PRIOR_EPOCHS);
    std::fprintf(fp,
                 "  \"pressure_controller\":{\"mode\":\"%s\",\"samples\":%d,"
                 "\"initial_loss\":%.8f,\"final_loss\":%.8f,"
                 "\"epochs\":%d},\n",
                 pressure_mode_name(pressure_mode),
                 pressure_training.samples, pressure_training.initial_loss,
                 pressure_training.final_loss,
                 pressure_mode == PressureMode::Learned ? PRESSURE_EPOCHS : 0);
    std::fprintf(fp,
                 "  \"budget_decision\":{\"mode\":\"%s\","
                 "\"extra_pass\":%s,\"accepted_extra\":%s,"
                 "\"decision_pass\":%d,"
                 "\"score\":%.6f,\"context_difficulty\":%.6f,"
                 "\"cvar_after_decision\":%.6f,"
                 "\"residual_after_decision\":%.6f,"
                 "\"refinement_score_before\":%.6f,"
                 "\"refinement_score_after\":%.6f,"
                 "\"refinement_score_delta\":%.6f,"
                 "\"estimated_extra_ms\":%.6f,"
                 "\"fixed_gpu_ms\":%.6f,\"final_gpu_ms\":%.6f},\n",
                 budget_mode_name(budget_decision.mode),
                 budget_decision.extra_pass ? "true" : "false",
                 budget_decision.accepted_extra ? "true" : "false",
                 budget_decision.decision_pass,
                 budget_decision.score,
                 budget_decision.context_difficulty,
                 budget_decision.cvar_after_decision,
                 budget_decision.residual_after_decision,
                 budget_decision.refinement_score_before,
                 budget_decision.refinement_score_after,
                 budget_decision.refinement_score_delta,
                 budget_decision.estimated_extra_ms,
                 budget_decision.fixed_gpu_ms,
                 budget_decision.final_gpu_ms);
    std::fprintf(fp,
                 "  \"intent\":{\"top1_accuracy_pct\":%.6f,"
                 "\"mean_confidence\":%.6f,\"mean_true_probability\":%.6f},\n",
                 intent_stats.top1_accuracy, intent_stats.mean_confidence,
                 intent_stats.mean_true_probability);
    std::fprintf(fp,
                 "  \"runtime\":{\"cpu_ms\":%.6f,\"gpu_ms\":%.6f,"
                 "\"speedup\":%.6f},\n",
                 cpu_ms, gpu_ms, speedup);
    std::fprintf(fp, "  \"planners\":[\n");
    for (size_t i = 0; i < rows.size(); i++) {
        write_row_json(fp, rows[i], i + 1 < rows.size());
    }
    std::fprintf(fp, "  ]\n");
    std::fprintf(fp, "}\n");
    std::fclose(fp);
    return true;
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

static void runtime_prior_features(const RobotSpec& robot,
                                   const RolloutResult& previous,
                                   const RolloutResult& best_response,
                                   float positive_regret,
                                   int pass,
                                   float x[PRIOR_INPUTS]) {
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

    x[0] = clampf(positive_regret / 0.68f, 0.0f, 1.45f);
    x[1] = tail_relief;
    x[2] = sep_relief;
    x[3] = clampf(tail_violation / 1.45f, 0.0f, 1.50f);
    x[4] = clampf(sep_violation / 0.72f, 0.0f, 1.50f);
    x[5] = clampf(terminal_violation / 1.55f, 0.0f, 1.50f);
    x[6] = crowd / 2.2f;
    x[7] = robot.priority - 0.65f;
    x[8] = pass_phase / static_cast<float>(N_GAME_PASSES - 1);
}

static SafetyDualPrior teacher_safety_dual_from_features(const float x[PRIOR_INPUTS]) {
    float positive_regret = 0.68f * x[0];
    float tail_relief = x[1];
    float sep_relief = x[2];
    float tail_violation = 1.45f * x[3];
    float sep_violation = 0.72f * x[4];
    float terminal_violation = 1.55f * x[5];
    float crowd = 2.2f * x[6];
    float priority_delta = x[7];
    float pass_phase = static_cast<float>(N_GAME_PASSES - 1) * x[8];

    float h0 = fast_tanh(1.34f * positive_regret + 0.72f * tail_relief
                       + 0.54f * sep_relief - 0.56f * terminal_violation
                       + 0.16f * priority_delta);
    float h1 = fast_tanh(0.92f * crowd + 0.42f * fmaxf(tail_relief, 0.0f)
                       + 0.36f * fmaxf(sep_relief, 0.0f) - 0.30f * pass_phase);
    float h2 = fast_tanh(1.20f * tail_violation + 1.18f * sep_violation
                       + 0.72f * terminal_violation);
    float h3 = fast_tanh(0.76f * positive_regret - 0.36f * crowd
                       - 0.28f * priority_delta
                       - 0.20f * pass_phase);
    float margin = tail_relief + 0.62f * sep_relief - 1.18f * tail_violation
                 - 1.24f * sep_violation - 0.30f * terminal_violation;
    float relief = clampf(fmaxf(tail_relief, 0.0f) + 0.65f * fmaxf(sep_relief, 0.0f),
                          0.0f, 1.5f);
    float pressure = clampf(fmaxf(-margin, 0.0f)
                          + 0.42f * tail_violation + 0.55f * sep_violation
                          + 0.18f * terminal_violation, 0.0f, 1.7f);
    float final_bias = x[8] > 0.55f ? 1.0f : 0.0f;

    SafetyDualPrior p{};
    p.dual = clampf(1.00f + 0.30f * h0 + 0.18f * h1 - 0.28f * h2 + 0.08f * h3,
                    0.52f, 1.44f);
    p.alpha_multiplier = clampf(0.98f + 0.20f * h0 + 0.12f * h1 - 0.24f * h2
                              - 0.05f * pass_phase,
                                0.58f, 1.22f);
    p.scale_hint = clampf(0.96f + 0.28f * h1 + 0.16f * h0 - 0.18f * h2
                        - 0.04f * pass_phase,
                          0.68f, 1.46f);
    p.dual = clampf(p.dual + 0.08f * relief - 0.06f * pressure, 0.52f, 1.44f);
    p.alpha_multiplier = clampf(p.alpha_multiplier + 0.06f * relief
                              - 0.10f * pressure + 0.03f * final_bias,
                                0.56f, 1.24f);
    p.scale_hint = clampf(p.scale_hint + 0.08f * relief - 0.05f * pressure
                        + 0.05f * final_bias, 0.64f, 1.54f);
    p.margin = margin;
    return p;
}

static void encode_prior_target(const SafetyDualPrior& p, float y[PRIOR_OUTPUTS]) {
    y[0] = clampf((p.dual - 1.00f) / 0.44f, -0.98f, 0.98f);
    y[1] = clampf((p.alpha_multiplier - 0.90f) / 0.36f, -0.98f, 0.98f);
    y[2] = clampf((p.scale_hint - 1.02f) / 0.52f, -0.98f, 0.98f);
}

static SafetyDualPrior decode_prior_output(const float y[PRIOR_OUTPUTS],
                                           float margin) {
    SafetyDualPrior p{};
    p.dual = clampf(1.00f + 0.44f * y[0], 0.52f, 1.44f);
    p.alpha_multiplier = clampf(0.90f + 0.36f * y[1], 0.56f, 1.24f);
    p.scale_hint = clampf(1.02f + 0.52f * y[2], 0.64f, 1.54f);
    p.margin = margin;
    return p;
}

static float deterministic_wave(int i, int j, float scale) {
    return scale * std::sin(0.73f * static_cast<float>(i + 1)
                          + 1.91f * static_cast<float>(j + 3));
}

static void make_synthetic_prior_samples(const std::vector<RobotSpec>& robots,
                                         std::vector<PriorSample>& samples) {
    samples.clear();
    for (int robot = 0; robot < N_ROBOTS; robot++) {
        for (int pass = 1; pass < N_GAME_PASSES; pass++) {
            for (int mode = 0; mode < 8; mode++) {
                PriorSample s{};
                float m = static_cast<float>(mode);
                float relief = (m - 3.5f) / 3.5f;
                s.x[0] = clampf(0.14f + 0.15f * m
                              + deterministic_wave(robot, mode + pass, 0.05f),
                                0.0f, 1.45f);
                s.x[1] = clampf(0.60f * relief
                              + deterministic_wave(robot, mode + 7, 0.14f),
                                -1.0f, 1.0f);
                s.x[2] = clampf(0.48f * relief
                              + deterministic_wave(robot + pass, mode + 11, 0.16f),
                                -1.0f, 1.0f);
                s.x[3] = clampf(0.04f + 0.10f * static_cast<float>((mode + robot) % 4)
                              + (relief < -0.2f ? 0.34f * (-relief) : 0.0f),
                                0.0f, 1.50f);
                s.x[4] = clampf(0.03f + 0.08f * static_cast<float>((mode + 2 * pass) % 5)
                              + (relief < -0.1f ? 0.28f * (-relief) : 0.0f),
                                0.0f, 1.50f);
                s.x[5] = clampf(0.02f + 0.07f * static_cast<float>((robot + mode + pass) % 6),
                                0.0f, 1.50f);
                s.x[6] = clampf(0.18f + 0.10f * m + 0.20f * robots[robot].priority
                              + deterministic_wave(robot, mode + 17, 0.06f),
                                0.0f, 1.0f);
                s.x[7] = robots[robot].priority - 0.65f;
                s.x[8] = static_cast<float>(pass - 1) / static_cast<float>(N_GAME_PASSES - 1);
                encode_prior_target(teacher_safety_dual_from_features(s.x), s.y);
                samples.push_back(s);
            }
        }
    }
}

static void init_safety_dual_network(SafetyDualNetwork& net) {
    for (int h = 0; h < PRIOR_HIDDEN; h++) {
        net.b1[h] = 0.03f * std::sin(0.41f * static_cast<float>(h + 1));
        for (int i = 0; i < PRIOR_INPUTS; i++) {
            net.w1[h][i] = 0.16f * std::sin(0.37f * static_cast<float>((h + 1) * (i + 3)));
        }
    }
    for (int o = 0; o < PRIOR_OUTPUTS; o++) {
        net.b2[o] = 0.0f;
        for (int h = 0; h < PRIOR_HIDDEN; h++) {
            net.w2[o][h] = 0.12f * std::cos(0.29f * static_cast<float>((o + 2) * (h + 1)));
        }
    }
}

static void forward_safety_dual_network(const SafetyDualNetwork& net,
                                        const float x[PRIOR_INPUTS],
                                        float hidden[PRIOR_HIDDEN],
                                        float y[PRIOR_OUTPUTS]) {
    for (int h = 0; h < PRIOR_HIDDEN; h++) {
        float z = net.b1[h];
        for (int i = 0; i < PRIOR_INPUTS; i++) z += net.w1[h][i] * x[i];
        hidden[h] = fast_tanh(z);
    }
    for (int o = 0; o < PRIOR_OUTPUTS; o++) {
        float z = net.b2[o];
        for (int h = 0; h < PRIOR_HIDDEN; h++) z += net.w2[o][h] * hidden[h];
        y[o] = fast_tanh(z);
    }
}

static float safety_dual_network_loss(const SafetyDualNetwork& net,
                                      const std::vector<PriorSample>& samples) {
    float loss = 0.0f;
    float h[PRIOR_HIDDEN];
    float y[PRIOR_OUTPUTS];
    for (const PriorSample& s : samples) {
        forward_safety_dual_network(net, s.x, h, y);
        for (int o = 0; o < PRIOR_OUTPUTS; o++) loss += sqr(y[o] - s.y[o]);
    }
    return loss / std::max(1.0f, static_cast<float>(samples.size() * PRIOR_OUTPUTS));
}

static PriorTrainingStats train_safety_dual_network(const std::vector<RobotSpec>& robots,
                                                    SafetyDualNetwork& net) {
    std::vector<PriorSample> samples;
    make_synthetic_prior_samples(robots, samples);
    init_safety_dual_network(net);
    PriorTrainingStats stats{};
    stats.samples = static_cast<int>(samples.size());
    stats.initial_loss = safety_dual_network_loss(net, samples);

    float h[PRIOR_HIDDEN];
    float y[PRIOR_OUTPUTS];
    for (int epoch = 0; epoch < PRIOR_EPOCHS; epoch++) {
        float lr = 0.030f / (1.0f + 0.018f * static_cast<float>(epoch));
        for (const PriorSample& s : samples) {
            forward_safety_dual_network(net, s.x, h, y);
            float old_w2[PRIOR_OUTPUTS][PRIOR_HIDDEN];
            for (int o = 0; o < PRIOR_OUTPUTS; o++) {
                for (int j = 0; j < PRIOR_HIDDEN; j++) old_w2[o][j] = net.w2[o][j];
            }
            float dz2[PRIOR_OUTPUTS];
            for (int o = 0; o < PRIOR_OUTPUTS; o++) {
                dz2[o] = (y[o] - s.y[o]) * (1.0f - y[o] * y[o]);
            }
            float dz1[PRIOR_HIDDEN];
            for (int j = 0; j < PRIOR_HIDDEN; j++) {
                float g = 0.0f;
                for (int o = 0; o < PRIOR_OUTPUTS; o++) g += dz2[o] * old_w2[o][j];
                dz1[j] = g * (1.0f - h[j] * h[j]);
            }
            for (int o = 0; o < PRIOR_OUTPUTS; o++) {
                for (int j = 0; j < PRIOR_HIDDEN; j++) net.w2[o][j] -= lr * dz2[o] * h[j];
                net.b2[o] -= lr * dz2[o];
            }
            for (int j = 0; j < PRIOR_HIDDEN; j++) {
                for (int i = 0; i < PRIOR_INPUTS; i++) net.w1[j][i] -= lr * dz1[j] * s.x[i];
                net.b1[j] -= lr * dz1[j];
            }
        }
    }
    stats.final_loss = safety_dual_network_loss(net, samples);
    return stats;
}

static SafetyDualPrior predict_safety_dual_prior(const SafetyDualNetwork& net,
                                                 const RobotSpec& robot,
                                                 const RolloutResult& previous,
                                                 const RolloutResult& best_response,
                                                 float positive_regret,
                                                 int pass) {
    float x[PRIOR_INPUTS];
    runtime_prior_features(robot, previous, best_response, positive_regret, pass, x);
    float h[PRIOR_HIDDEN];
    float y[PRIOR_OUTPUTS];
    forward_safety_dual_network(net, x, h, y);
    SafetyDualPrior teacher = teacher_safety_dual_from_features(x);
    SafetyDualPrior p = decode_prior_output(y, teacher.margin);
    float trust = clampf(0.70f + 0.20f * (1.0f - x[8]), 0.70f, 0.90f);
    p.dual = clampf(trust * p.dual + (1.0f - trust) * teacher.dual, 0.52f, 1.44f);
    p.alpha_multiplier = clampf(trust * p.alpha_multiplier
                              + (1.0f - trust) * teacher.alpha_multiplier,
                                0.56f, 1.24f);
    p.scale_hint = clampf(trust * p.scale_hint + (1.0f - trust) * teacher.scale_hint,
                          0.64f, 1.54f);
    return p;
}

static float pressure_context_difficulty(const PressureContext& context) {
    return clampf(0.30f * context.lane_tightness
                + 0.34f * context.conflict_density
                + 0.18f * context.cross_shift_load
                + 0.18f * context.priority_flip,
                  0.0f, 1.35f);
}

static PressureContext synthetic_pressure_context(int mode) {
    PressureContext ctx{};
    float m = static_cast<float>(mode);
    ctx.lane_tightness = clampf(0.08f + 0.19f * m
                              + deterministic_wave(mode, 1, 0.04f),
                                0.0f, 1.25f);
    ctx.conflict_density = clampf(0.18f + 0.20f * static_cast<float>((mode * 5) % 7)
                                + deterministic_wave(mode, 3, 0.05f),
                                  0.0f, 1.35f);
    ctx.cross_shift_load = clampf(0.04f + 0.23f * static_cast<float>((mode + 2) % 5)
                                + deterministic_wave(mode, 5, 0.04f),
                                  0.0f, 1.25f);
    ctx.priority_flip = (mode == 2 || mode == 4 || mode == 5) ? 1.0f : 0.0f;
    return ctx;
}

static float teacher_safety_pressure(const TeamMetrics& previous_metrics,
                                     const PressureContext& context,
                                     float mean_safety_violation,
                                     int pass) {
    float cvar_band = clampf((previous_metrics.collision_cvar - 21.0f) / 7.0f,
                             0.0f, 1.0f);
    float cvar_over = clampf((previous_metrics.collision_cvar - SHOWDOWN_CVAR_TARGET)
                           / 4.0f, 0.0f, 1.0f);
    float collision_load = clampf(static_cast<float>(previous_metrics.collisions) / 120.0f,
                                  0.0f, 1.0f);
    float sep_pressure = clampf((0.16f - previous_metrics.min_separation) / 0.62f,
                                0.0f, 1.0f);
    float violation_pressure = clampf(mean_safety_violation / 0.70f, 0.0f, 1.0f);
    float scene_pressure = pressure_context_difficulty(context);
    float early_phase = pass <= 1 ? 1.0f : (pass == 2 ? 0.68f : 0.36f);
    float late_phase = pass >= N_GAME_PASSES - 1 ? 1.0f
                     : (pass >= N_GAME_PASSES - 2 ? 0.78f : 0.62f);
    float pressure = 1.0f + late_phase * (0.10f
        + 0.23f * cvar_band
        + 0.34f * cvar_over
        + 0.12f * collision_load
        + 0.17f * sep_pressure
        + 0.12f * violation_pressure)
        + early_phase * (0.09f * scene_pressure
        + 0.04f * scene_pressure * fmaxf(cvar_band, sep_pressure));
    return clampf(pressure, 1.0f, 1.52f);
}

static void runtime_pressure_features(const TeamMetrics& previous_metrics,
                                      const PressureContext& context,
                                      float mean_safety_violation,
                                      int pass,
                                      float x[PRESSURE_INPUTS]) {
    x[0] = clampf(previous_metrics.collision_cvar / 68.0f, 0.0f, 1.45f);
    x[1] = clampf((previous_metrics.collision_cvar - SHOWDOWN_CVAR_TARGET)
                / 18.0f, 0.0f, 1.45f);
    x[2] = clampf(static_cast<float>(previous_metrics.collisions) / 220.0f,
                  0.0f, 1.45f);
    x[3] = clampf((0.16f - previous_metrics.min_separation) / 0.70f,
                  0.0f, 1.45f);
    x[4] = clampf(mean_safety_violation / 0.75f, 0.0f, 1.45f);
    x[5] = static_cast<float>(std::max(0, pass - 1))
         / static_cast<float>(N_GAME_PASSES - 1);
    x[6] = context.lane_tightness;
    x[7] = context.conflict_density;
    x[8] = context.cross_shift_load;
    x[9] = context.priority_flip;
}

static float encode_pressure_target(float pressure) {
    return clampf((pressure - 1.26f) / 0.26f, -0.98f, 0.98f);
}

static float decode_pressure_output(float y) {
    return clampf(1.26f + 0.26f * y, 1.0f, 1.52f);
}

static void init_pressure_controller(PressureController& net) {
    for (int h = 0; h < PRESSURE_HIDDEN; h++) {
        net.b1[h] = 0.02f * std::cos(0.53f * static_cast<float>(h + 1));
        for (int i = 0; i < PRESSURE_INPUTS; i++) {
            net.w1[h][i] = 0.18f * std::sin(0.31f * static_cast<float>((h + 2) * (i + 5)));
        }
        net.w2[h] = 0.16f * std::cos(0.43f * static_cast<float>(h + 3));
    }
    net.b2 = 0.0f;
}

static float forward_pressure_controller(const PressureController& net,
                                         const float x[PRESSURE_INPUTS],
                                         float hidden[PRESSURE_HIDDEN]) {
    for (int h = 0; h < PRESSURE_HIDDEN; h++) {
        float z = net.b1[h];
        for (int i = 0; i < PRESSURE_INPUTS; i++) z += net.w1[h][i] * x[i];
        hidden[h] = fast_tanh(z);
    }
    float y = net.b2;
    for (int h = 0; h < PRESSURE_HIDDEN; h++) y += net.w2[h] * hidden[h];
    return fast_tanh(y);
}

static float pressure_controller_loss(const PressureController& net,
                                      const std::vector<PressureSample>& samples) {
    float loss = 0.0f;
    float hidden[PRESSURE_HIDDEN];
    for (const PressureSample& s : samples) {
        float y = forward_pressure_controller(net, s.x, hidden);
        loss += sqr(y - s.y);
    }
    return loss / std::max(1.0f, static_cast<float>(samples.size()));
}

static void make_pressure_samples(std::vector<PressureSample>& samples) {
    samples.clear();
    for (int pass = 1; pass < N_GAME_PASSES; pass++) {
        for (int cvar_mode = 0; cvar_mode < 8; cvar_mode++) {
            for (int collision_mode = 0; collision_mode < 6; collision_mode++) {
                for (int sep_mode = 0; sep_mode < 5; sep_mode++) {
                    for (int context_mode = 0; context_mode < 6; context_mode++) {
                        PressureSample s{};
                        TeamMetrics m{};
                        PressureContext context = synthetic_pressure_context(context_mode);
                        float c = static_cast<float>(cvar_mode);
                        float q = static_cast<float>(collision_mode);
                        float r = static_cast<float>(sep_mode);
                        float difficulty = pressure_context_difficulty(context);
                        m.collision_cvar = 14.0f + 7.4f * c + 2.8f * difficulty
                                         + deterministic_wave(pass, cvar_mode + sep_mode, 1.8f);
                        m.collisions = std::max(0, static_cast<int>(
                            7.0f + 43.0f * q + 18.0f * difficulty
                            + 8.0f * deterministic_wave(collision_mode, pass, 1.0f)));
                        m.min_separation = 0.24f - 0.17f * r - 0.05f * difficulty
                                         + deterministic_wave(sep_mode, cvar_mode, 0.035f);
                        float violation = clampf(0.06f + 0.13f * r + 0.055f * c
                                               + 0.08f * difficulty
                                               + deterministic_wave(cvar_mode, collision_mode, 0.04f),
                                                 0.0f, 1.05f);
                        runtime_pressure_features(m, context, violation, pass, s.x);
                        s.y = encode_pressure_target(
                            teacher_safety_pressure(m, context, violation, pass));
                        samples.push_back(s);
                    }
                }
            }
        }
    }
}

static PressureTrainingStats train_pressure_controller(PressureController& net) {
    std::vector<PressureSample> samples;
    make_pressure_samples(samples);
    init_pressure_controller(net);
    PressureTrainingStats stats{};
    stats.samples = static_cast<int>(samples.size());
    stats.initial_loss = pressure_controller_loss(net, samples);

    float hidden[PRESSURE_HIDDEN];
    for (int epoch = 0; epoch < PRESSURE_EPOCHS; epoch++) {
        float lr = 0.034f / (1.0f + 0.020f * static_cast<float>(epoch));
        for (const PressureSample& s : samples) {
            float y = forward_pressure_controller(net, s.x, hidden);
            float dz2 = (y - s.y) * (1.0f - y * y);
            float old_w2[PRESSURE_HIDDEN];
            for (int h = 0; h < PRESSURE_HIDDEN; h++) old_w2[h] = net.w2[h];
            for (int h = 0; h < PRESSURE_HIDDEN; h++) {
                net.w2[h] -= lr * dz2 * hidden[h];
            }
            net.b2 -= lr * dz2;
            for (int h = 0; h < PRESSURE_HIDDEN; h++) {
                float dz1 = dz2 * old_w2[h] * (1.0f - hidden[h] * hidden[h]);
                for (int i = 0; i < PRESSURE_INPUTS; i++) {
                    net.w1[h][i] -= lr * dz1 * s.x[i];
                }
                net.b1[h] -= lr * dz1;
            }
        }
    }
    stats.final_loss = pressure_controller_loss(net, samples);
    return stats;
}

static float predict_safety_pressure(PressureMode mode,
                                     const PressureController& net,
                                     const TeamMetrics& previous_metrics,
                                     const PressureContext& context,
                                     float mean_safety_violation,
                                     int pass) {
    if (mode == PressureMode::None) return 1.0f;
    if (mode == PressureMode::Teacher) {
        return teacher_safety_pressure(previous_metrics, context,
                                       mean_safety_violation, pass);
    }
    float x[PRESSURE_INPUTS];
    float hidden[PRESSURE_HIDDEN];
    runtime_pressure_features(previous_metrics, context, mean_safety_violation, pass, x);
    float y = forward_pressure_controller(net, x, hidden);
    return decode_pressure_output(y);
}

static void mix_paths_trainable_safety_prior(const SafetyDualNetwork& prior_net,
                                             const PressureController& pressure_controller,
                                             PressureMode pressure_mode,
                                             const PressureContext& pressure_context,
                                             const std::vector<RobotSpec>& robots,
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
    float safety_violation_sum = 0.0f;
    TeamMetrics previous_metrics = compute_path_team_metrics(robots, previous_x, previous_y);

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
            predict_safety_dual_prior(prior_net, robots[robot], previous,
                                      best_response, positive_regret, pass);
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
        safety_violation_sum += safety_violation;
    }
    float mean_safety_violation = safety_violation_sum / static_cast<float>(N_ROBOTS);
    float safety_pressure = predict_safety_pressure(pressure_mode,
                                                    pressure_controller,
                                                    previous_metrics,
                                                    pressure_context,
                                                    mean_safety_violation,
                                                    pass);

    auto mix_for_scale = [&](float scale,
                             std::vector<float>& out_x,
                             std::vector<float>& out_y,
                             std::vector<float>* alpha_out) {
        out_x.resize(previous_x.size());
        out_y.resize(previous_y.size());
        if (alpha_out) alpha_out->assign(N_ROBOTS, 0.0f);
        for (int robot = 0; robot < N_ROBOTS; robot++) {
            float alpha = clampf(alpha_base[robot] * scale * safety_pressure,
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
        float final_pass = pass >= N_GAME_PASSES - 1 ? 1.0f : 0.0f;
        float cvar_target = final_pass > 0.0f ? SHOWDOWN_CVAR_TARGET : SAFETY_CVAR_TARGET;
        int collision_target = final_pass > 0.0f
                             ? SHOWDOWN_COLLISION_TARGET
                             : SAFETY_COLLISION_TARGET;
        float cvar_over = fmaxf(metrics.collision_cvar - cvar_target, 0.0f);
        float collision_over = static_cast<float>(
            std::max(metrics.collisions - collision_target, 0));
        float reach_loss = static_cast<float>(N_ROBOTS - metrics.reached);
        float terminal_over = fmaxf(metrics.mean_terminal - 0.82f, 0.0f);
        float sep_over = fmaxf(-0.42f - metrics.min_separation, 0.0f);
        float previous_cvar_over = fmaxf(metrics.collision_cvar
                                       - 0.96f * previous_metrics.collision_cvar, 0.0f);
        float cvar_weight = 3.0f + 1.8f * final_pass;
        float collision_weight = 0.28f + 0.74f * final_pass;
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
    std::vector<float> reach_guard_x = mixed_x;
    std::vector<float> reach_guard_y = mixed_y;
    float pressure_over = fmaxf(safety_pressure - 1.0f, 0.0f);
    float context_difficulty = pressure_mode == PressureMode::None
                             ? 0.0f
                             : pressure_context_difficulty(pressure_context);
    float projection_extra = 0.10f * pressure_over + 0.025f * context_difficulty;
    int projection_passes = pass >= N_GAME_PASSES - 1
                          ? (safety_pressure > 1.05f ? 5 : 3)
                          : (safety_pressure > 1.05f ? 3 : 2);
    if (context_difficulty > 0.85f) projection_passes += 1;
    project_tail_risk_clearance(robots, mixed_x, mixed_y, projection_passes,
                                projection_extra);
    for (int robot = 0; robot < N_ROBOTS; robot++) {
        int last = robot * HORIZON + HORIZON - 1;
        float terminal = std::sqrt(sqr(robots[robot].gx - mixed_x[last])
                                 + sqr(robots[robot].gy - mixed_y[last]));
        if (terminal <= 2.10f) continue;
        constexpr int TAIL_STEPS = 20;
        for (int kk = 0; kk < TAIL_STEPS; kk++) {
            int k = HORIZON - TAIL_STEPS + kk;
            int idx = robot * HORIZON + k;
            float u = static_cast<float>(kk) / static_cast<float>(TAIL_STEPS - 1);
            float gx;
            float gy;
            route_point(robots[robot], 0.76f + 0.24f * u, gx, gy);
            float blend = 0.20f + 0.80f * u * u;
            float guard_x = (1.0f - u) * reach_guard_x[idx] + u * gx;
            float guard_y = (1.0f - u) * reach_guard_y[idx] + u * gy;
            mixed_x[idx] = (1.0f - blend) * mixed_x[idx] + blend * guard_x;
            mixed_y[idx] = (1.0f - blend) * mixed_y[idx] + blend * guard_y;
        }
    }
    if (safety_pressure > 1.05f) {
        project_tail_risk_clearance(robots, mixed_x, mixed_y,
                                    pass >= N_GAME_PASSES - 1 ? 4 : 2,
                                    projection_extra + 0.035f);
    }
    best_metrics = compute_path_team_metrics(robots, mixed_x, mixed_y);

    stats = IterationStats{};
    stats.min_alpha = 1.0e6f;
    stats.safety_scale = best_scale;
    stats.cvar_before = previous_metrics.collision_cvar;
    stats.cvar_after = best_metrics.collision_cvar;
    float residual_gain_sum = 0.0f;
    float residual_baseline_sum = 0.0f;
    for (int robot = 0; robot < N_ROBOTS; robot++) {
        float alpha = chosen_alpha[robot];
        float residual_fraction = sqr(clampf(1.0f - alpha, 0.0f, 1.0f));
        float residual_gain = gains[robot] * residual_fraction;
        residual_gain_sum += residual_gain;
        residual_baseline_sum += baselines[robot];
        stats.mean_alpha += alpha;
        stats.min_alpha = std::min(stats.min_alpha, alpha);
        stats.max_alpha = std::max(stats.max_alpha, alpha);
        stats.mean_positive_regret += positive_regrets[robot];
        stats.max_positive_regret = std::max(stats.max_positive_regret,
                                             positive_regrets[robot]);
        stats.mean_unilateral_gain += residual_gain;
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
    stats.normalized_gain = 100.0f * residual_gain_sum
        / std::max(residual_baseline_sum, 1.0e-3f);
    stats.mean_alpha /= static_cast<float>(N_ROBOTS);
    stats.mean_positive_regret /= static_cast<float>(N_ROBOTS);
    stats.mean_safety_dual /= static_cast<float>(N_ROBOTS);
    stats.mean_safety_violation /= static_cast<float>(N_ROBOTS);
    stats.mean_prior_scale /= static_cast<float>(N_ROBOTS);
    stats.mean_prior_margin /= static_cast<float>(N_ROBOTS);
}

static double sum_gpu_pass_ms(const std::vector<float>& gpu_pass_ms) {
    double total = 0.0;
    for (float ms : gpu_pass_ms) total += static_cast<double>(ms);
    return total;
}

static float budget_refinement_score(const TeamMetrics& metrics) {
    return 2.4f * metrics.collision_cvar
         + 9.0f * static_cast<float>(metrics.collisions)
         + 18.0f * static_cast<float>(N_ROBOTS - metrics.reached);
}

static BudgetDecision make_budget_decision(BudgetMode mode,
                                           PressureMode pressure_mode,
                                           const PressureContext& pressure_context,
                                           const std::vector<IterationStats>& safe_stats,
                                           double fixed_gpu_ms,
                                           float estimated_extra_ms) {
    BudgetDecision decision{};
    decision.mode = mode;
    decision.extra_pass = false;
    decision.accepted_extra = false;
    decision.decision_pass = 0;
    decision.context_difficulty = pressure_context_difficulty(pressure_context);
    decision.estimated_extra_ms = estimated_extra_ms;
    decision.fixed_gpu_ms = fixed_gpu_ms;
    decision.final_gpu_ms = fixed_gpu_ms;

    if (!safe_stats.empty()) {
        size_t idx = safe_stats.size() >= 2 ? 1 : safe_stats.size() - 1;
        const IterationStats& s = safe_stats[idx];
        decision.decision_pass = static_cast<int>(idx + 1);
        decision.cvar_after_decision = s.cvar_after;
        decision.residual_after_decision = s.normalized_gain;
    }

    float cvar_pressure = clampf((decision.cvar_after_decision - 22.5f) / 4.0f,
                                 0.0f, 1.0f);
    float residual_pressure = clampf((decision.residual_after_decision - 3.0f) / 5.0f,
                                     0.0f, 1.0f);
    decision.score = 0.62f * decision.context_difficulty
                   + 0.55f * cvar_pressure
                   + 0.25f * residual_pressure;

    bool runtime_headroom = fixed_gpu_ms + 0.15
                          <= static_cast<double>(SHOWDOWN_RUNTIME_TARGET_MS);
    decision.extra_pass = mode == BudgetMode::Learned
                       && pressure_mode != PressureMode::None
                       && safe_stats.size() >= 2
                       && runtime_headroom
                       && decision.context_difficulty > 0.76f
                       && decision.cvar_after_decision > 24.2f
                       && decision.score > 0.72f;
    return decision;
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
                          const std::vector<float>& orca_x,
                          const std::vector<float>& orca_y,
                          const std::vector<float>& priority_x,
                          const std::vector<float>& priority_y,
                          const std::vector<float>& noregret_x,
                          const std::vector<float>& noregret_y,
                          const std::vector<float>& safe_x,
                          const std::vector<float>& safe_y,
                          const std::vector<float>& sample_x,
                          const std::vector<float>& sample_y,
                          const TeamMetrics& orca_like,
                          const TeamMetrics& priority_graph,
                          const TeamMetrics& noregret,
                          const TeamMetrics& safe_noregret,
                          const IntentStats& intent_stats,
                          const GameStats& noregret_game,
                          const GameStats& safe_game,
                          const std::vector<IterationStats>& noregret_stats,
                          const std::vector<IterationStats>& safe_stats,
                          const std::string& scenario_name,
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
    draw_graph_edges(img, robots, orca_x, orca_y, 0, step);
    draw_graph_edges(img, robots, priority_x, priority_y, HALF_W, step);
    draw_graph_edges(img, robots, noregret_x, noregret_y, 2 * HALF_W, step);
    draw_graph_edges(img, robots, safe_x, safe_y, 3 * HALF_W, step);
    draw_paths(img, robots, orca_x, orca_y, 0, step, true);
    draw_paths(img, robots, priority_x, priority_y, HALF_W, step, true);
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
    float noregret_collision_drop = 100.0f * (1.0f - static_cast<float>(noregret.collisions)
                                                    / static_cast<float>(std::max(orca_like.collisions, 1)));
    float safe_collision_drop = 100.0f * (1.0f - static_cast<float>(safe_noregret.collisions)
                                                / static_cast<float>(std::max(orca_like.collisions, 1)));
    float safe_cvar_drop = 100.0f * (1.0f - safe_noregret.collision_cvar
                                           / std::max(orca_like.collision_cvar, 1.0e-6f));
    IterationStats last_safe{};
    if (!safe_stats.empty()) last_safe = safe_stats.back();
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "GPU planner showdown  scenario=%s  %d robots x %d rollouts x H=%d x %d passes  target: C<=%d CVaR<=%.1f residual<=%.1f%% runtime<=%.1fms  gpu=%.2f ms %.1fx",
                  scenario_name.c_str(), N_ROBOTS, ROLLOUTS_PER_ROBOT,
                  HORIZON, N_GAME_PASSES,
                  SHOWDOWN_COLLISION_TARGET, SHOWDOWN_CVAR_TARGET,
                  SHOWDOWN_RESIDUAL_TARGET, SHOWDOWN_RUNTIME_TARGET_MS,
                  gpu_ms, speedup);
    cv::putText(img, buf, cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.43, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "ORCA-like reciprocal", cv::Point(14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "priority graph baseline", cv::Point(HALF_W + 14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "no-regret graph MPPI", cv::Point(2 * HALF_W + 14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "trainable safety-dual MPPI", cv::Point(3 * HALF_W + 14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf),
                  "collisions %d -> %d -> %d -> %d  (-%.1f%% no-regret, -%.1f%% safety-dual vs ORCA-like)   CVaR %.2f -> %.2f (-%.1f%%)",
                  orca_like.collisions, priority_graph.collisions, noregret.collisions,
                  safe_noregret.collisions, noregret_collision_drop,
                  safe_collision_drop, orca_like.collision_cvar,
                  safe_noregret.collision_cvar, safe_cvar_drop);
    cv::putText(img, buf, cv::Point(12, PANEL_H - 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(225, 238, 245), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf),
                  "reach %d/%d -> %d/%d -> %d/%d -> %d/%d   residual no-regret %.2f%% -> safety-dual %.2f%%   dual %.2f scale %.2f   intent top1 %.1f%%",
                  orca_like.reached, N_ROBOTS, priority_graph.reached, N_ROBOTS,
                  noregret.reached, N_ROBOTS, safe_noregret.reached, N_ROBOTS,
                  noregret_game.normalized_gain, safe_game.normalized_gain,
                  last_safe.mean_safety_dual,
                  last_safe.safety_scale, intent_stats.top1_accuracy);
    cv::putText(img, buf, cv::Point(12, PANEL_H - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(225, 238, 245), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main(int argc, char** argv) {
    CliOptions opts{};
    if (!parse_cli(argc, argv, opts)) {
        print_usage(argv[0]);
        return 2;
    }
    if (opts.help) {
        print_usage(argv[0]);
        return 0;
    }

    std::vector<RobotSpec> robots = make_robots(opts.scenario);
    PressureContext pressure_context =
        pressure_context_from_scenario(opts.scenario, robots);
    SafetyDualNetwork prior_net{};
    PriorTrainingStats prior_training = train_safety_dual_network(robots, prior_net);
    PressureController pressure_controller{};
    PressureTrainingStats pressure_training{};
    if (opts.pressure_mode == PressureMode::Learned) {
        pressure_training = train_pressure_controller(pressure_controller);
    }
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
        mix_paths_trainable_safety_prior(prior_net, pressure_controller,
                                         opts.pressure_mode, pressure_context, robots,
                                         cpu_safe_peer_x, cpu_safe_peer_y,
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
    double noregret_gpu_ms = static_cast<double>(gpu_one_shot_ms_f);

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
        noregret_gpu_ms += static_cast<double>(pass_ms);

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
        mix_paths_trainable_safety_prior(prior_net, pressure_controller,
                                         opts.pressure_mode, pressure_context, robots,
                                         safe_peer_x, safe_peer_y, safe_response_x,
                                         safe_response_y, safe_response,
                                         safe_previous_selected, safe_selected, pass,
                                         safe_mixed_x, safe_mixed_y, safe_pass_stats);
        safe_stats.push_back(safe_pass_stats);
        safe_peer_x.swap(safe_mixed_x);
        safe_peer_y.swap(safe_mixed_y);
        safe_previous_selected.swap(safe_selected);
        safe_x = safe_peer_x;
        safe_y = safe_peer_y;
        safe_game = GameStats{safe_pass_stats.mean_unilateral_gain,
                              safe_pass_stats.normalized_gain,
                              safe_pass_game.max_unilateral_gain};
    }

    double fixed_gpu_ms = sum_gpu_pass_ms(gpu_pass_ms);
    float estimated_extra_ms = 0.0f;
    BudgetDecision budget_decision =
        make_budget_decision(opts.budget_mode, opts.pressure_mode, pressure_context,
                             safe_stats, fixed_gpu_ms, estimated_extra_ms);
    if (budget_decision.extra_pass) {
        TeamMetrics before_budget = compute_path_team_metrics(robots, safe_x, safe_y);
        float before_score = budget_refinement_score(before_budget);
        budget_decision.refinement_score_before = before_score;
        budget_decision.refinement_score_after = before_score;
        float best_score = before_score;
        std::vector<float> best_x;
        std::vector<float> best_y;
        const float extras[] = {0.012f, 0.024f, 0.036f, 0.052f};
        for (int passes = 1; passes <= 4; passes++) {
            for (float extra : extras) {
                std::vector<float> candidate_x = safe_x;
                std::vector<float> candidate_y = safe_y;
                project_tail_risk_clearance(robots, candidate_x, candidate_y,
                                            passes, extra);
                TeamMetrics candidate_metrics =
                    compute_path_team_metrics(robots, candidate_x, candidate_y);
                float candidate_score = budget_refinement_score(candidate_metrics);
                if (candidate_score < best_score
                    && candidate_metrics.reached >= before_budget.reached
                    && candidate_metrics.deadlocks <= before_budget.deadlocks) {
                    best_score = candidate_score;
                    best_x.swap(candidate_x);
                    best_y.swap(candidate_y);
                }
            }
        }
        if (!best_x.empty()) {
            safe_x.swap(best_x);
            safe_y.swap(best_y);
            safe_peer_x = safe_x;
            safe_peer_y = safe_y;
            budget_decision.accepted_extra = true;
            budget_decision.refinement_score_after = best_score;
            budget_decision.refinement_score_delta = before_score - best_score;
        }
    }
    budget_decision.final_gpu_ms = sum_gpu_pass_ms(gpu_pass_ms);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    double gpu_ms = budget_decision.final_gpu_ms;

    std::vector<float> sample_x(sample_total);
    std::vector<float> sample_y(sample_total);
    CUDA_CHECK(cudaMemcpy(sample_x.data(), d_sample_x, sample_total * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sample_y.data(), d_sample_y, sample_total * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::vector<float> orca_x;
    std::vector<float> orca_y;
    auto orca_begin = std::chrono::high_resolution_clock::now();
    make_orca_like_paths(robots, orca_x, orca_y);
    auto orca_end = std::chrono::high_resolution_clock::now();
    double orca_ms = std::chrono::duration<double, std::milli>(
        orca_end - orca_begin).count();

    std::vector<float> priority_x;
    std::vector<float> priority_y;
    auto priority_begin = std::chrono::high_resolution_clock::now();
    make_priority_graph_paths(robots, priority_x, priority_y);
    auto priority_end = std::chrono::high_resolution_clock::now();
    double priority_ms = std::chrono::duration<double, std::milli>(
        priority_end - priority_begin).count();

    TeamMetrics orca_like = compute_path_team_metrics(robots, orca_x, orca_y);
    TeamMetrics priority_graph = compute_path_team_metrics(robots, priority_x, priority_y);
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
    std::printf("Scenario: %s (lane_scale %.2f, jitter_scale %.2f, cross_shift %.2f, pressure_mode %s, priority_flip %s)\n",
                opts.scenario.name.c_str(), opts.scenario.lane_scale,
                opts.scenario.jitter_scale, opts.scenario.cross_shift,
                pressure_mode_name(opts.pressure_mode),
                opts.scenario.priority_flip ? "true" : "false");
    std::printf("Trainable safety-dual prior: %d synthetic labels, loss %.5f -> %.5f (%d epochs)\n",
                prior_training.samples, prior_training.initial_loss,
                prior_training.final_loss, PRIOR_EPOCHS);
    if (opts.pressure_mode == PressureMode::Learned) {
        std::printf("Learned safety-pressure controller: %d synthetic metric labels, loss %.5f -> %.5f (%d epochs)\n",
                    pressure_training.samples, pressure_training.initial_loss,
                    pressure_training.final_loss, PRESSURE_EPOCHS);
    } else {
        std::printf("Safety-pressure controller: %s mode (learned controller bypassed)\n",
                    pressure_mode_name(opts.pressure_mode));
    }
    std::printf("Pressure context: lane_tight %.3f, conflict_density %.3f, cross_shift %.3f, priority_flip %.1f\n",
                pressure_context.lane_tightness,
                pressure_context.conflict_density,
                pressure_context.cross_shift_load,
                pressure_context.priority_flip);
    std::printf("Adaptive budget: mode %s, extra_pass %s, accepted %s, score %.3f, decision pass %d CVaR %.3f residual %.2f%%, refine %.2f -> %.2f, fixed %.3f ms -> final %.3f ms\n",
                budget_mode_name(budget_decision.mode),
                budget_decision.extra_pass ? "true" : "false",
                budget_decision.accepted_extra ? "true" : "false",
                budget_decision.score,
                budget_decision.decision_pass,
                budget_decision.cvar_after_decision,
                budget_decision.residual_after_decision,
                budget_decision.refinement_score_before,
                budget_decision.refinement_score_after,
                budget_decision.fixed_gpu_ms,
                budget_decision.final_gpu_ms);
    std::printf("Intent inference: top-1 %.1f%%, mean confidence %.3f, true-intent probability %.3f\n",
                intent_stats.top1_accuracy, intent_stats.mean_confidence,
                intent_stats.mean_true_probability);
    std::printf("CPU planner showdown benchmark game MPPI: %.3f ms (%d robots x %d rollouts x H=%d x %zu rollout batches)\n",
                cpu_ms, N_ROBOTS, ROLLOUTS_PER_ROBOT, HORIZON, cpu_pass_ms.size());
    std::printf("GPU planner showdown benchmark game graph MPPI: %.3f ms (one-shot+shared raw+adaptive safety-dual best-response batches, %.1fx vs CPU equivalent rollout eval; %zu GPU batches)\n",
                gpu_ms, speedup, gpu_pass_ms.size());
    std::printf("Showdown hard target: reach %d/%d, deadlocks <= %d, collisions <= %d, CVaR <= %.1f, residual <= %.1f%%, runtime <= %.1f ms\n",
                SHOWDOWN_REACH_TARGET, N_ROBOTS, SHOWDOWN_DEADLOCK_TARGET,
                SHOWDOWN_COLLISION_TARGET, SHOWDOWN_CVAR_TARGET,
                SHOWDOWN_RESIDUAL_TARGET, SHOWDOWN_RUNTIME_TARGET_MS);
    std::vector<ShowdownRow> showdown_rows;
    showdown_rows.push_back(ShowdownRow{"ORCA-like reciprocal", orca_like,
                                        orca_ms, -1.0f});
    showdown_rows.push_back(ShowdownRow{"Priority graph baseline", priority_graph,
                                        priority_ms, -1.0f});
    showdown_rows.push_back(ShowdownRow{"No-regret graph MPPI", noregret,
                                        noregret_gpu_ms,
                                        noregret_game.normalized_gain});
    showdown_rows.push_back(ShowdownRow{"Trainable safety-dual MPPI", safe_noregret,
                                        gpu_ms, safe_game.normalized_gain});
    bool target_pass = showdown_target_pass(showdown_rows.back());
    std::printf("Planner showdown table:\n");
    for (const ShowdownRow& row : showdown_rows) print_showdown_row(row);
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
    std::printf("Trainable safety-dual team: collisions %d, reached %d/%d, deadlocks %d, min sep %.3f, terminal avg %.3f, social risk avg/max %.3f/%.3f, collision CVaR %.3f, final residual %.3f (%.2f%%, max %.3f), collision reduction %.1f%%, risk reduction %.1f%%, CVaR reduction %.1f%%, separation gain %.3f\n",
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
        std::printf("Pass %zu trainable safety-dual update: mean path delta %.3f, max path delta %.3f, alpha avg/range %.3f [%.3f, %.3f], dual avg/max %.3f/%.3f, prior scale avg/max %.3f/%.3f, margin avg %.3f, violation avg/max %.3f/%.3f, CVaR %.3f -> %.3f, chosen scale %.2f, residual %.3f (%.2f%%)\n",
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
    int exit_code = opts.check_targets && !target_pass ? 2 : 0;
    bool wrote_json =
        write_showdown_json(opts.json_path, opts.scenario, opts.pressure_mode,
                            pressure_context,
                            showdown_rows, prior_training, pressure_training,
                            budget_decision, intent_stats, cpu_ms, gpu_ms,
                            speedup, target_pass);
    if (!wrote_json) {
        std::fprintf(stderr, "failed to write %s\n", opts.json_path.c_str());
        if (exit_code == 0) exit_code = 1;
    } else {
        std::printf("JSON saved to %s\n", opts.json_path.c_str());
    }

    if (opts.no_video) {
        std::printf("GIF rendering skipped by --no-video\n");
    } else {
        cv::VideoWriter video("gif/gpu_planner_showdown_benchmark.avi",
                              cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                              VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
        if (!video.isOpened()) {
            std::fprintf(stderr, "failed to open gif/gpu_planner_showdown_benchmark.avi\n");
            exit_code = 1;
        } else {
            for (int k = 2; k <= HORIZON; k += 2) {
                video.write(draw_frame(robots, orca_x, orca_y, priority_x, priority_y,
                                       noregret_x, noregret_y, safe_x, safe_y, sample_x,
                                       sample_y, orca_like, priority_graph, noregret,
                                       safe_noregret, intent_stats, noregret_game,
                                       safe_game, noregret_stats, safe_stats,
                                       opts.scenario.name, gpu_ms, cpu_ms, k));
            }
            for (int i = 0; i < 12; i++) {
                video.write(draw_frame(robots, orca_x, orca_y, priority_x, priority_y,
                                       noregret_x, noregret_y, safe_x, safe_y, sample_x,
                                       sample_y, orca_like, priority_graph, noregret,
                                       safe_noregret, intent_stats, noregret_game,
                                       safe_game, noregret_stats, safe_stats,
                                       opts.scenario.name, gpu_ms, cpu_ms, HORIZON));
            }
            video.release();

            avi_to_gif("gif/gpu_planner_showdown_benchmark.avi",
                       "gif/gpu_planner_showdown_benchmark.gif", 8, 720);
            std::printf("GIF saved to gif/gpu_planner_showdown_benchmark.gif\n");
        }
    }
    if (opts.check_targets) {
        std::printf("Showdown target check: %s\n",
                    target_pass ? "PASS" : "FAIL");
    }

    CUDA_CHECK(cudaFree(d_robots));
    CUDA_CHECK(cudaFree(d_beliefs));
    CUDA_CHECK(cudaFree(d_one_shot));
    CUDA_CHECK(cudaFree(d_response));
    CUDA_CHECK(cudaFree(d_peer_x));
    CUDA_CHECK(cudaFree(d_peer_y));
    CUDA_CHECK(cudaFree(d_sample_x));
    CUDA_CHECK(cudaFree(d_sample_y));
    return exit_code;
}
