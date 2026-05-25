// gpu_interaction_graph_neural_mppi.cu
//
// GPU interaction-graph neural MPPI planner.
//
// A learned experience-graph route supplies a sparse global prior to an
// Ackermann-like MPPI rollout optimizer, while a GPU interaction graph predicts
// social risk around moving agents.  Pairwise closing-speed seeds are diffused
// with message passing, rasterized into a spatiotemporal risk field, then used
// by the planner.  The demo compares an interaction-aware graph-neural MPPI
// rollout batch against an interaction-blind batch on the same scene.
//
// Output: gif/gpu_interaction_graph_neural_mppi.gif

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

constexpr int FIELD_W = 192;
constexpr int FIELD_H = 128;
constexpr int FIELD_CELLS = FIELD_W * FIELD_H;
constexpr int GRAPH_W = 48;
constexpr int GRAPH_H = 32;
constexpr int ROLLOUTS = 32768;
constexpr int HORIZON = 72;
constexpr int SAMPLE_TRAJS = 96;
constexpr int GUIDE_POINTS = 18;
constexpr int SOCIAL_AGENTS = 48;
constexpr int MSG_ITERS = 4;
constexpr int THREADS = 256;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int HEADER_H = 44;
constexpr int FOOTER_H = 44;
constexpr int MAP_H = PANEL_H - HEADER_H - FOOTER_H;
constexpr int HALF_W = PANEL_W / 2;
constexpr int VIDEO_FPS = 10;

constexpr float WORLD_W = 18.0f;
constexpr float WORLD_H = 11.0f;
constexpr float START_X = 0.85f;
constexpr float START_Y = 1.10f;
constexpr float START_THETA = 0.12f;
constexpr float GOAL_X = 17.15f;
constexpr float GOAL_Y = 9.55f;
constexpr float DT = 0.22f;
constexpr float PI_F = 3.14159265358979323846f;
constexpr float MIN_SPEED = 0.12f;
constexpr float MAX_SPEED = 1.48f;
constexpr float MAX_STEER = 0.62f;
constexpr float WHEEL_BASE = 1.62f;
constexpr float INF_COST = 1.0e20f;

struct RolloutResult {
    float select_cost;
    float cost;
    float terminal_error;
    float mean_risk;
    float max_risk;
    float mean_interaction;
    float max_interaction;
    float min_clearance;
    float mean_route_error;
    float final_x;
    float final_y;
    int rollout_id;
};

struct Pose2 {
    float x;
    float y;
    float theta;
};

struct SocialAgent {
    float x0;
    float y0;
    float speed;
    float radius;
    float phase;
    int route;
    int kind;
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

__host__ __device__ static inline int field_index_from_xy(float x, float y) {
    int ix = static_cast<int>(clampf(x / WORLD_W * static_cast<float>(FIELD_W), 0.0f,
                                     static_cast<float>(FIELD_W - 1)));
    int iy = static_cast<int>(clampf(y / WORLD_H * static_cast<float>(FIELD_H), 0.0f,
                                     static_cast<float>(FIELD_H - 1)));
    return iy * FIELD_W + ix;
}

__host__ __device__ static inline float lookup_field(const float* field, float x, float y) {
    return field[field_index_from_xy(x, y)];
}

__host__ __device__ static inline float lookup_time_field(const float* field,
                                                          float x,
                                                          float y,
                                                          int step) {
    int k = max(0, min(step, HORIZON - 1));
    return field[k * FIELD_CELLS + field_index_from_xy(x, y)];
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
    float e0 = line_corridor_score(x, y, START_X, START_Y, GOAL_X, GOAL_Y, 1.25f);
    float e1a = line_corridor_score(x, y, START_X, START_Y, 7.7f, 3.30f, 1.08f);
    float e1b = line_corridor_score(x, y, 7.7f, 3.30f, GOAL_X, GOAL_Y, 1.08f);
    float e2a = line_corridor_score(x, y, 0.95f, 8.95f, 8.2f, 6.65f, 1.05f);
    float e2b = line_corridor_score(x, y, 8.2f, 6.65f, 16.35f, 1.95f, 1.05f);
    return clampf(fmaxf(e0, fmaxf(fmaxf(e1a, e1b), fmaxf(e2a, e2b))), 0.0f, 1.0f);
}

__host__ __device__ static inline float learned_traversability_cost(float x, float y) {
    float h = terrain_height(x, y);
    float rough = terrain_roughness(x, y);
    float clear = clampf(raw_clearance(x, y) / 1.15f, 0.0f, 1.0f);
    float exp_prior = experience_prior(x, y);
    float low_clear = clampf((0.62f - clear) / 0.62f, 0.0f, 1.0f);
    float abs_h = fabsf(h);
    float height_risk = clampf((abs_h - 0.28f) / 0.42f, 0.0f, 1.0f);
    float latent0 = tanhf(1.38f * rough + 1.16f * low_clear + 0.82f * height_risk
                        - 0.86f * exp_prior - 0.18f);
    float latent1 = tanhf(0.82f * rough + 0.72f * abs_h - 0.74f * clear
                        - 0.54f * exp_prior + 0.08f);
    float caution = sigmoid(-0.35f + 1.24f * latent0 + 0.88f * latent1
                          + 0.92f * rough + 1.08f * low_clear - 0.62f * exp_prior);
    float cost = 0.08f + 0.88f * caution + 0.44f * rough + 0.34f * height_risk
               - 0.24f * exp_prior;
    if (raw_clearance(x, y) < -0.02f || rough > 0.82f) cost += 7.0f;
    return clampf(cost, 0.04f, 18.0f);
}

__host__ __device__ static inline void moving_obstacle(int obstacle_id,
                                                       float t,
                                                       float& x,
                                                       float& y,
                                                       float& radius) {
    if (obstacle_id == 0) {
        x = 3.2f + 3.4f * (0.5f + 0.5f * sinf(0.62f * t + 0.35f));
        y = 2.35f + 0.40f * sinf(0.86f * t + 1.40f);
        radius = 0.42f;
        return;
    }
    if (obstacle_id == 1) {
        x = 9.7f + 0.55f * sinf(0.72f * t);
        y = 2.1f + 4.6f * (0.5f + 0.5f * sinf(0.52f * t + 1.05f));
        radius = 0.50f;
        return;
    }
    if (obstacle_id == 2) {
        x = 12.8f + 2.55f * sinf(0.46f * t + 2.10f);
        y = 8.1f + 0.82f * cosf(0.78f * t + 0.20f);
        radius = 0.46f;
        return;
    }
    x = 6.4f + 3.1f * (0.5f + 0.5f * sinf(0.43f * t + 2.55f));
    y = 6.75f + 1.05f * sinf(0.64f * t + 0.80f);
    radius = 0.44f;
}

__host__ __device__ static inline float dynamic_risk(float x, float y, float t) {
    float risk = 0.0f;
    for (int k = 0; k < 4; k++) {
        float ox, oy, radius;
        moving_obstacle(k, t, ox, oy, radius);
        float margin = sqrtf(sqr(x - ox) + sqr(y - oy)) - radius;
        risk += 0.70f * expf(-0.5f * sqr(margin / 0.55f))
              + 1.30f * sigmoid((0.18f - margin) * 7.5f);
    }
    return clampf(risk, 0.0f, 4.0f);
}

__host__ __device__ static inline void social_state(const SocialAgent& a,
                                                    int step,
                                                    float& x,
                                                    float& y,
                                                    float& vx,
                                                    float& vy) {
    float t = static_cast<float>(step) * DT;
    float wiggle = 0.10f * sinf(0.65f * t + a.phase);
    if (a.route == 0) {
        x = a.x0 + a.speed * t;
        y = a.y0 + 0.18f * sinf(0.80f * t + a.phase);
        vx = a.speed;
        vy = 0.14f * cosf(0.80f * t + a.phase);
    } else if (a.route == 1) {
        x = a.x0 + wiggle;
        y = a.y0 + a.speed * t;
        vx = 0.065f * cosf(0.65f * t + a.phase);
        vy = a.speed;
    } else if (a.route == 2) {
        x = a.x0 - a.speed * t;
        y = a.y0 + 0.18f * sinf(0.70f * t + a.phase);
        vx = -a.speed;
        vy = 0.13f * cosf(0.70f * t + a.phase);
    } else {
        x = a.x0 + wiggle;
        y = a.y0 - a.speed * t;
        vx = 0.065f * cosf(0.65f * t + a.phase);
        vy = -a.speed;
    }
}

__host__ __device__ static inline float pair_seed_risk(float ax,
                                                       float ay,
                                                       float avx,
                                                       float avy,
                                                       float ar,
                                                       float bx,
                                                       float by,
                                                       float bvx,
                                                       float bvy,
                                                       float br,
                                                       int ak,
                                                       int bk) {
    float dx = bx - ax;
    float dy = by - ay;
    float d2 = dx * dx + dy * dy;
    if (d2 > 10.0f || d2 < 1.0e-8f) return 0.0f;
    float d = sqrtf(d2);
    float rvx = bvx - avx;
    float rvy = bvy - avy;
    float closing = -(dx * rvx + dy * rvy) / (d + 1.0e-5f);
    float ttc = closing > 0.04f ? d / closing : 1.0e6f;
    float combined = ar + br + 0.30f;
    float near = clampf((combined - d) / fmaxf(combined, 1.0e-4f), 0.0f, 1.0f);
    float future = ttc < 3.0f ? (3.0f - ttc) / 3.0f : 0.0f;
    float crossing = (ak != bk) ? 1.0f : 0.18f;
    return clampf(crossing * (0.70f * future + 0.55f * near), 0.0f, 1.0f);
}

__host__ __device__ static inline float pair_message_weight(float ax,
                                                            float ay,
                                                            float avx,
                                                            float avy,
                                                            float bx,
                                                            float by,
                                                            float bvx,
                                                            float bvy,
                                                            int ak,
                                                            int bk) {
    float dx = bx - ax;
    float dy = by - ay;
    float d2 = dx * dx + dy * dy;
    if (d2 > 8.0f || d2 < 1.0e-8f) return 0.0f;
    float d = sqrtf(d2);
    float rvx = bvx - avx;
    float rvy = bvy - avy;
    float closing = -(dx * rvx + dy * rvy) / (d + 1.0e-5f);
    float ttc = closing > 0.04f ? d / closing : 1.0e6f;
    float proximity = expf(-0.5f * d2 / 3.2f);
    float future = ttc < 3.2f ? (3.2f - ttc) / 3.2f : 0.0f;
    float crossing = (ak != bk) ? 1.0f : 0.22f;
    return clampf(crossing * (0.14f * proximity + 0.88f * future), 0.0f, 1.0f);
}

__global__ void predict_social_kernel(const SocialAgent* __restrict__ agents,
                                      float* __restrict__ ax,
                                      float* __restrict__ ay,
                                      float* __restrict__ avx,
                                      float* __restrict__ avy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = HORIZON * SOCIAL_AGENTS;
    if (idx >= total) return;
    int step = idx / SOCIAL_AGENTS;
    int agent = idx - step * SOCIAL_AGENTS;
    social_state(agents[agent], step, ax[idx], ay[idx], avx[idx], avy[idx]);
}

__global__ void seed_social_risk_kernel(const SocialAgent* __restrict__ agents,
                                        const float* __restrict__ ax,
                                        const float* __restrict__ ay,
                                        const float* __restrict__ avx,
                                        const float* __restrict__ avy,
                                        float* __restrict__ base,
                                        float* __restrict__ risk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = HORIZON * SOCIAL_AGENTS;
    if (idx >= total) return;
    int step = idx / SOCIAL_AGENTS;
    int i = idx - step * SOCIAL_AGENTS;
    float r = 0.0f;
    for (int j = 0; j < SOCIAL_AGENTS; j++) {
        if (i == j) continue;
        int jidx = step * SOCIAL_AGENTS + j;
        float p = pair_seed_risk(ax[idx], ay[idx], avx[idx], avy[idx], agents[i].radius,
                                 ax[jidx], ay[jidx], avx[jidx], avy[jidx], agents[j].radius,
                                 agents[i].route, agents[j].route);
        r = fmaxf(r, p);
    }
    base[idx] = r;
    risk[idx] = r;
}

__global__ void propagate_social_risk_kernel(const SocialAgent* __restrict__ agents,
                                             const float* __restrict__ ax,
                                             const float* __restrict__ ay,
                                             const float* __restrict__ avx,
                                             const float* __restrict__ avy,
                                             const float* __restrict__ base,
                                             const float* __restrict__ risk_in,
                                             float* __restrict__ risk_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = HORIZON * SOCIAL_AGENTS;
    if (idx >= total) return;
    int step = idx / SOCIAL_AGENTS;
    int i = idx - step * SOCIAL_AGENTS;
    float weighted = 0.0f;
    float wsum = 0.0f;
    float max_msg = 0.0f;
    for (int j = 0; j < SOCIAL_AGENTS; j++) {
        if (i == j) continue;
        int jidx = step * SOCIAL_AGENTS + j;
        float w = pair_message_weight(ax[idx], ay[idx], avx[idx], avy[idx],
                                      ax[jidx], ay[jidx], avx[jidx], avy[jidx],
                                      agents[i].route, agents[j].route);
        if (w <= 0.010f) continue;
        float msg = w * risk_in[jidx];
        weighted += msg;
        wsum += w;
        max_msg = fmaxf(max_msg, msg);
    }
    float mean_msg = wsum > 1.0e-6f ? weighted / wsum : 0.0f;
    float propagated = 0.72f * (0.62f * max_msg + 0.38f * mean_msg);
    risk_out[idx] = clampf(fmaxf(base[idx], propagated), 0.0f, 1.0f);
}

__global__ void rasterize_interaction_field_kernel(const SocialAgent* __restrict__ agents,
                                                   const float* __restrict__ ax,
                                                   const float* __restrict__ ay,
                                                   const float* __restrict__ avx,
                                                   const float* __restrict__ avy,
                                                   const float* __restrict__ risk,
                                                   float* __restrict__ field,
                                                   float* __restrict__ clearance_field) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = HORIZON * FIELD_CELLS;
    if (idx >= total) return;
    int step = idx / FIELD_CELLS;
    int cell = idx - step * FIELD_CELLS;
    int ix = cell % FIELD_W;
    int iy = cell / FIELD_W;
    float x = (static_cast<float>(ix) + 0.5f) / FIELD_W * WORLD_W;
    float y = (static_cast<float>(iy) + 0.5f) / FIELD_H * WORLD_H;

    float social = 0.0f;
    float min_clearance = 1.0e6f;
    for (int i = 0; i < SOCIAL_AGENTS; i++) {
        int aidx = step * SOCIAL_AGENTS + i;
        float dx = x - ax[aidx];
        float dy = y - ay[aidx];
        float d2 = dx * dx + dy * dy;
        float d = sqrtf(d2);
        min_clearance = fminf(min_clearance, d - agents[i].radius);
        float sigma = 0.42f + agents[i].radius + 0.48f * risk[aidx];
        float radial = expf(-0.5f * d2 / fmaxf(sigma * sigma, 1.0e-4f));
        float lane = expf(-0.5f * sqr((dy * avx[aidx] - dx * avy[aidx])
                                      / (sqrtf(avx[aidx] * avx[aidx] + avy[aidx] * avy[aidx]) + 1.0e-4f))
                          / 0.55f);
        social += (0.42f + 2.55f * risk[aidx]) * radial + 0.30f * risk[aidx] * lane;
    }
    field[idx] = clampf(social, 0.0f, 4.5f);
    clearance_field[idx] = min_clearance;
}

__host__ __device__ static inline void guide_point(const float* rx,
                                                   const float* ry,
                                                   float progress,
                                                   float& tx,
                                                   float& ty) {
    float u = clampf(progress, 0.0f, 1.0f) * static_cast<float>(GUIDE_POINTS - 1);
    int i0 = static_cast<int>(floorf(u));
    int i1 = min(i0 + 1, GUIDE_POINTS - 1);
    float a = u - static_cast<float>(i0);
    tx = (1.0f - a) * rx[i0] + a * rx[i1];
    ty = (1.0f - a) * ry[i0] + a * ry[i1];
}

__host__ __device__ static inline float route_distance(const float* rx,
                                                       const float* ry,
                                                       float x,
                                                       float y,
                                                       float* progress_out) {
    float best = 1.0e9f;
    float best_progress = 0.0f;
    for (int i = 0; i < GUIDE_POINTS - 1; i++) {
        float ax = rx[i];
        float ay = ry[i];
        float bx = rx[i + 1];
        float by = ry[i + 1];
        float vx = bx - ax;
        float vy = by - ay;
        float len2 = vx * vx + vy * vy;
        float t = clampf(((x - ax) * vx + (y - ay) * vy) / len2, 0.0f, 1.0f);
        float px = ax + t * vx;
        float py = ay + t * vy;
        float d = sqrtf(sqr(x - px) + sqr(y - py));
        if (d < best) {
            best = d;
            best_progress = (static_cast<float>(i) + t) / static_cast<float>(GUIDE_POINTS - 1);
        }
    }
    if (progress_out) *progress_out = best_progress;
    return best;
}

__host__ __device__ static inline void learned_control_prior(const Pose2& s,
                                                             float tx,
                                                             float ty,
                                                             float expv,
                                                             float terrain,
                                                             float risk,
                                                             float interaction,
                                                             int aware,
                                                             float& speed,
                                                             float& steer,
                                                             float& heading_error) {
    float dx = tx - s.x;
    float dy = ty - s.y;
    float target_heading = atan2f(dy, dx);
    heading_error = wrap_angle(target_heading - s.theta);
    float dist = sqrtf(dx * dx + dy * dy);
    float rough_slow = clampf((terrain - 0.72f) / 4.2f, 0.0f, 1.0f);
    float risk_slow = clampf(risk / 2.2f, 0.0f, 1.0f);
    float social_slow = clampf(interaction / 2.4f, 0.0f, 1.0f);
    float route_boost = expv;
    float align = cosf(heading_error);
    float latent0 = fast_tanh(1.20f * align + 0.74f * route_boost
                            - 0.62f * rough_slow - 0.70f * risk_slow
                            - 1.16f * social_slow + 0.18f * dist - 0.10f);
    float latent1 = fast_tanh(1.48f * heading_error + 0.34f * sinf(0.55f * s.x)
                            - 0.22f * cosf(0.80f * s.y) + 0.46f * route_boost);
    if (aware) {
        speed = clampf(0.52f + 1.18f * fast_sigmoid(1.75f * latent0 + 0.34f * dist
                                                    - 0.98f * fabsf(heading_error)
                                                    - 1.48f * social_slow),
                       MIN_SPEED, MAX_SPEED);
        float detour = 0.0f;
        steer = clampf(0.58f * fast_tanh(1.70f * heading_error + 0.42f * latent1 + detour),
                       -MAX_STEER, MAX_STEER);
    } else {
        speed = clampf(0.42f + 1.02f * fast_sigmoid(1.18f * latent0 + 0.26f * dist
                                                    - 0.82f * fabsf(heading_error)),
                       MIN_SPEED, MAX_SPEED);
        steer = clampf(0.45f * fast_tanh(1.38f * heading_error + 0.24f * latent1),
                       -MAX_STEER, MAX_STEER);
    }
}

__host__ __device__ static inline void rollout_step(int rollout,
                                                    int k,
                                                    int aware,
                                                    const float* rx,
                                                    const float* ry,
                                                    const float* cost_field,
                                                    const float* risk_field,
                                                    const float* route_field,
                                                    const float* interaction_field,
                                                    const float* clearance_field,
                                                    const float* exp_field,
                                                    Pose2& s,
                                                    float& prev_speed,
                                                    float& prev_steer,
                                                    float& route_d,
                                                    float& terrain,
                                                    float& risk,
                                                    float& interaction,
                                                    float& clearance,
                                                    float& heading_error,
                                                    float& steer,
                                                    float& smooth) {
    route_d = lookup_field(route_field, s.x, s.y);
    float tx, ty;
    float progress = static_cast<float>(k) / static_cast<float>(HORIZON - 1);
    float lookahead = progress + (aware ? 0.18f : 0.24f) + 0.04f * hash_unit(rollout, k / 8, 8);
    guide_point(rx, ry, lookahead, tx, ty);

    terrain = lookup_field(cost_field, s.x, s.y);
    risk = lookup_time_field(risk_field, s.x, s.y, k);
    interaction = aware ? lookup_time_field(interaction_field, s.x, s.y, k) : 0.0f;
    clearance = lookup_time_field(clearance_field, s.x, s.y, k);
    float expv = lookup_field(exp_field, s.x, s.y);
    float base_speed, base_steer;
    learned_control_prior(s, tx, ty, expv, terrain, risk, interaction, aware,
                          base_speed, base_steer, heading_error);

    float nv = 0.54f * hash_signed(rollout, k, 0) + 0.24f * hash_signed(rollout, k / 4, 2);
    float ns = 0.76f * hash_signed(rollout, k, 1) + 0.28f * hash_signed(rollout, k / 5, 4);
    float speed_sigma = aware ? 0.18f : 0.25f;
    float steer_sigma = aware ? 0.14f : 0.23f;
    float speed = clampf(base_speed + speed_sigma * nv, MIN_SPEED, MAX_SPEED);
    steer = clampf(base_steer + steer_sigma * ns, -MAX_STEER, MAX_STEER);
    if (aware) {
        float nx = -sinf(s.theta);
        float ny = cosf(s.theta);
        float left_risk = lookup_time_field(interaction_field, s.x + 0.55f * nx,
                                            s.y + 0.55f * ny, k);
        float right_risk = lookup_time_field(interaction_field, s.x - 0.55f * nx,
                                             s.y - 0.55f * ny, k);
        float avoid = 0.30f * fast_tanh(right_risk - left_risk);
        steer = clampf(steer + avoid, -MAX_STEER, MAX_STEER);
    }
    smooth = 0.05f * sqr(speed - prev_speed) + 0.19f * sqr(steer - prev_steer);

    float yaw_rate = speed * steer / WHEEL_BASE;
    s.x += speed * cosf(s.theta) * DT;
    s.y += speed * sinf(s.theta) * DT;
    s.theta = wrap_angle(s.theta + yaw_rate * DT);
    s.x = clampf(s.x, 0.05f, WORLD_W - 0.05f);
    s.y = clampf(s.y, 0.05f, WORLD_H - 0.05f);

    route_d = lookup_field(route_field, s.x, s.y);
    terrain = lookup_field(cost_field, s.x, s.y);
    risk = lookup_time_field(risk_field, s.x, s.y, k);
    interaction = lookup_time_field(interaction_field, s.x, s.y, k);
    clearance = lookup_time_field(clearance_field, s.x, s.y, k);
    prev_speed = speed;
    prev_steer = steer;
}

__host__ __device__ static inline RolloutResult evaluate_rollout(int rollout,
                                                                 int aware,
                                                                 const float* rx,
                                                                 const float* ry,
                                                                 const float* cost_field,
                                                                 const float* risk_field,
                                                                 const float* route_field,
                                                                 const float* interaction_field,
                                                                 const float* clearance_field,
                                                                 const float* exp_field) {
    Pose2 s{START_X, START_Y, START_THETA};
    float blind_cost = 0.0f;
    float full_cost = 0.0f;
    float risk_sum = 0.0f;
    float risk_max = 0.0f;
    float interaction_sum = 0.0f;
    float interaction_max = 0.0f;
    float min_clearance = 1.0e6f;
    float route_error_sum = 0.0f;
    float prev_speed = 0.72f;
    float prev_steer = 0.0f;

    for (int k = 0; k < HORIZON; k++) {
        float route_d, terrain, risk, interaction, clearance, heading_error, steer, smooth;
        rollout_step(rollout, k, aware, rx, ry, cost_field, risk_field, route_field,
                     interaction_field, clearance_field, exp_field,
                     s, prev_speed, prev_steer, route_d, terrain, risk, interaction, clearance,
                     heading_error, steer, smooth);
        float collision = terrain > 5.0f ? 10.0f * sqr(terrain - 5.0f) : 0.0f;
        float route_cost = 1.20f * route_d * route_d;
        float heading_cost = 0.24f * heading_error * heading_error;
        float steer_cost = 0.16f * sqr(steer / MAX_STEER);
        float base_step = 0.58f * terrain + 12.0f * risk + collision
                        + route_cost + heading_cost + steer_cost + smooth;
        float close_penalty = clearance < 0.42f ? 45.0f * sqr(0.42f - clearance) : 0.0f;
        float social_step = 17.0f * interaction + close_penalty;
        blind_cost += base_step;
        full_cost += base_step + social_step;
        risk_sum += risk;
        risk_max = fmaxf(risk_max, risk);
        interaction_sum += interaction;
        interaction_max = fmaxf(interaction_max, interaction);
        min_clearance = fminf(min_clearance, clearance);
        route_error_sum += route_d;
    }

    float terminal = sqrtf(sqr(GOAL_X - s.x) + sqr(GOAL_Y - s.y));
    float route_terminal = lookup_field(route_field, s.x, s.y);
    float goal_heading = atan2f(GOAL_Y - s.y, GOAL_X - s.x);
    float terminal_heading = wrap_angle(goal_heading - s.theta);
    float terminal_cost = 38.0f * terminal * terminal + 4.0f * route_terminal
                        + 2.5f * terminal_heading * terminal_heading;
    blind_cost += terminal_cost;
    full_cost += terminal_cost;

    RolloutResult r;
    r.select_cost = aware ? full_cost : blind_cost;
    r.cost = full_cost;
    r.terminal_error = terminal;
    r.mean_risk = risk_sum / static_cast<float>(HORIZON);
    r.max_risk = risk_max;
    r.mean_interaction = interaction_sum / static_cast<float>(HORIZON);
    r.max_interaction = interaction_max;
    r.min_clearance = min_clearance;
    r.mean_route_error = route_error_sum / static_cast<float>(HORIZON);
    r.final_x = s.x;
    r.final_y = s.y;
    r.rollout_id = rollout;
    return r;
}

__global__ void rollout_kernel(int aware,
                               const float* __restrict__ rx,
                               const float* __restrict__ ry,
                               const float* __restrict__ cost_field,
                               const float* __restrict__ risk_field,
                               const float* __restrict__ route_field,
                               const float* __restrict__ interaction_field,
                               const float* __restrict__ clearance_field,
                               const float* __restrict__ exp_field,
                               RolloutResult* __restrict__ results,
                               float* __restrict__ sample_x,
                               float* __restrict__ sample_y) {
    int rollout = blockIdx.x * blockDim.x + threadIdx.x;
    if (rollout >= ROLLOUTS) return;
    results[rollout] = evaluate_rollout(rollout, aware, rx, ry, cost_field, risk_field,
                                        route_field, interaction_field, clearance_field, exp_field);

    if (rollout >= SAMPLE_TRAJS) return;
    Pose2 s{START_X, START_Y, START_THETA};
    float prev_speed = 0.72f;
    float prev_steer = 0.0f;
    for (int k = 0; k < HORIZON; k++) {
        float route_d, terrain, risk, interaction, clearance, heading_error, steer, smooth;
        rollout_step(rollout, k, aware, rx, ry, cost_field, risk_field, route_field,
                     interaction_field, clearance_field, exp_field,
                     s, prev_speed, prev_steer, route_d, terrain, risk, interaction, clearance,
                     heading_error, steer, smooth);
        int idx = rollout * HORIZON + k;
        sample_x[idx] = s.x;
        sample_y[idx] = s.y;
    }
}

static std::vector<cv::Point2f> make_guide_route() {
    return {
        {START_X, START_Y},
        {1.95f, 1.18f},
        {3.15f, 1.28f},
        {4.30f, 1.62f},
        {5.35f, 2.15f},
        {6.42f, 2.90f},
        {7.50f, 3.68f},
        {8.60f, 4.38f},
        {9.58f, 5.12f},
        {10.58f, 5.92f},
        {11.70f, 6.70f},
        {12.82f, 7.32f},
        {13.92f, 7.75f},
        {14.95f, 8.20f},
        {15.82f, 8.68f},
        {16.52f, 9.12f},
        {16.92f, 9.38f},
        {GOAL_X, GOAL_Y},
    };
}

static std::vector<SocialAgent> make_social_agents() {
    std::vector<SocialAgent> agents(SOCIAL_AGENTS);
    for (int i = 0; i < SOCIAL_AGENTS; i++) {
        SocialAgent a{};
        a.route = i % 4;
        a.kind = (i % 9 == 0) ? 1 : 0;
        a.radius = a.kind ? 0.36f : 0.28f;
        a.phase = 0.37f * static_cast<float>(i);
        float lane_jitter = 0.18f * sinf(1.71f * static_cast<float>(i));
        float speed_jitter = 0.10f * sinf(0.93f * static_cast<float>(i));
        if (a.route == 0) {
            a.x0 = -7.0f + 0.62f * static_cast<float>(i / 4);
            a.y0 = 5.42f + lane_jitter;
            a.speed = 0.92f + speed_jitter;
        } else if (a.route == 1) {
            a.x0 = 8.95f + lane_jitter;
            a.y0 = -4.0f + 0.58f * static_cast<float>(i / 4);
            a.speed = 0.74f + 0.08f * sinf(0.61f * static_cast<float>(i));
        } else if (a.route == 2) {
            a.x0 = 23.0f - 0.58f * static_cast<float>(i / 4);
            a.y0 = 6.12f + lane_jitter;
            a.speed = 0.84f + speed_jitter;
        } else {
            a.x0 = 10.85f + lane_jitter;
            a.y0 = 15.2f - 0.54f * static_cast<float>(i / 4);
            a.speed = 0.82f + 0.07f * cosf(0.72f * static_cast<float>(i));
        }
        agents[i] = a;
    }
    return agents;
}

static void make_interaction_fields_gpu(const std::vector<SocialAgent>& agents,
                                        std::vector<float>& interaction_field,
                                        std::vector<float>& clearance_field) {
    const int social_total = HORIZON * SOCIAL_AGENTS;
    const int field_total = HORIZON * FIELD_CELLS;
    interaction_field.resize(field_total);
    clearance_field.resize(field_total);

    SocialAgent* d_agents = nullptr;
    float* d_ax = nullptr;
    float* d_ay = nullptr;
    float* d_avx = nullptr;
    float* d_avy = nullptr;
    float* d_base = nullptr;
    float* d_risk_a = nullptr;
    float* d_risk_b = nullptr;
    float* d_interaction = nullptr;
    float* d_clearance = nullptr;
    CUDA_CHECK(cudaMalloc(&d_agents, SOCIAL_AGENTS * sizeof(SocialAgent)));
    CUDA_CHECK(cudaMalloc(&d_ax, social_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ay, social_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_avx, social_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_avy, social_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_base, social_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_risk_a, social_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_risk_b, social_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_interaction, field_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_clearance, field_total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_agents, agents.data(), SOCIAL_AGENTS * sizeof(SocialAgent),
                          cudaMemcpyHostToDevice));

    int social_blocks = (social_total + THREADS - 1) / THREADS;
    int field_blocks = (field_total + THREADS - 1) / THREADS;
    predict_social_kernel<<<social_blocks, THREADS>>>(d_agents, d_ax, d_ay, d_avx, d_avy);
    seed_social_risk_kernel<<<social_blocks, THREADS>>>(d_agents, d_ax, d_ay, d_avx, d_avy,
                                                        d_base, d_risk_a);
    for (int iter = 0; iter < MSG_ITERS; iter++) {
        propagate_social_risk_kernel<<<social_blocks, THREADS>>>(d_agents, d_ax, d_ay, d_avx, d_avy,
                                                                 d_base, d_risk_a, d_risk_b);
        std::swap(d_risk_a, d_risk_b);
    }
    rasterize_interaction_field_kernel<<<field_blocks, THREADS>>>(d_agents, d_ax, d_ay, d_avx, d_avy,
                                                                  d_risk_a, d_interaction,
                                                                  d_clearance);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(interaction_field.data(), d_interaction, field_total * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(clearance_field.data(), d_clearance, field_total * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_agents));
    CUDA_CHECK(cudaFree(d_ax));
    CUDA_CHECK(cudaFree(d_ay));
    CUDA_CHECK(cudaFree(d_avx));
    CUDA_CHECK(cudaFree(d_avy));
    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_risk_a));
    CUDA_CHECK(cudaFree(d_risk_b));
    CUDA_CHECK(cudaFree(d_interaction));
    CUDA_CHECK(cudaFree(d_clearance));
}

static void make_cost_fields(const std::vector<float>& route_x,
                             const std::vector<float>& route_y,
                             std::vector<float>& cost_field,
                             std::vector<float>& risk_field,
                             std::vector<float>& route_field,
                             std::vector<float>& exp_field) {
    cost_field.resize(FIELD_CELLS);
    route_field.resize(FIELD_CELLS);
    exp_field.resize(FIELD_CELLS);
    risk_field.resize(static_cast<size_t>(HORIZON) * FIELD_CELLS);
    for (int iy = 0; iy < FIELD_H; iy++) {
        for (int ix = 0; ix < FIELD_W; ix++) {
            int idx = iy * FIELD_W + ix;
            float x = (static_cast<float>(ix) + 0.5f) / FIELD_W * WORLD_W;
            float y = (static_cast<float>(iy) + 0.5f) / FIELD_H * WORLD_H;
            cost_field[idx] = learned_traversability_cost(x, y);
            route_field[idx] = route_distance(route_x.data(), route_y.data(), x, y, nullptr);
            exp_field[idx] = experience_prior(x, y);
            for (int k = 0; k < HORIZON; k++) {
                risk_field[static_cast<size_t>(k) * FIELD_CELLS + idx] =
                    dynamic_risk(x, y, static_cast<float>(k) * DT);
            }
        }
    }
}

static double evaluate_cpu_rollouts(int guided,
                                    const std::vector<float>& rx,
                                    const std::vector<float>& ry,
                                    const std::vector<float>& cost_field,
                                    const std::vector<float>& risk_field,
                                    const std::vector<float>& route_field,
                                    const std::vector<float>& interaction_field,
                                    const std::vector<float>& clearance_field,
                                    const std::vector<float>& exp_field,
                                    std::vector<RolloutResult>& out) {
    out.resize(ROLLOUTS);
    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ROLLOUTS; i++) {
        out[i] = evaluate_rollout(i, guided, rx.data(), ry.data(), cost_field.data(),
                                  risk_field.data(), route_field.data(), interaction_field.data(),
                                  clearance_field.data(), exp_field.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

static int best_rollout_index(const std::vector<RolloutResult>& results) {
    float best = INF_COST;
    int best_idx = 0;
    for (size_t i = 0; i < results.size(); i++) {
        if (results[i].select_cost < best) {
            best = results[i].select_cost;
            best_idx = static_cast<int>(i);
        }
    }
    return best_idx;
}

static std::vector<cv::Point2f> reconstruct_trajectory(int rollout,
                                                       int guided,
                                                       const std::vector<float>& rx,
                                                       const std::vector<float>& ry,
                                                       const std::vector<float>& cost_field,
                                                       const std::vector<float>& risk_field,
                                                       const std::vector<float>& route_field,
                                                       const std::vector<float>& interaction_field,
                                                       const std::vector<float>& clearance_field,
                                                       const std::vector<float>& exp_field) {
    std::vector<cv::Point2f> traj;
    traj.reserve(HORIZON);
    Pose2 s{START_X, START_Y, START_THETA};
    float prev_speed = 0.72f;
    float prev_steer = 0.0f;
    for (int k = 0; k < HORIZON; k++) {
        float route_d, terrain, risk, interaction, clearance, heading_error, steer, smooth;
        rollout_step(rollout, k, guided, rx.data(), ry.data(), cost_field.data(), risk_field.data(),
                     route_field.data(), interaction_field.data(), clearance_field.data(),
                     exp_field.data(),
                     s, prev_speed, prev_steer, route_d, terrain, risk, interaction, clearance,
                     heading_error, steer, smooth);
        traj.push_back(cv::Point2f(s.x, s.y));
    }
    return traj;
}

static cv::Point to_px(float x, float y, int x0) {
    int px = x0 + static_cast<int>(x / WORLD_W * static_cast<float>(HALF_W - 1));
    int py = HEADER_H + static_cast<int>((1.0f - y / WORLD_H) * static_cast<float>(MAP_H - 1));
    return cv::Point(px, py);
}

static cv::Scalar blend(cv::Scalar a, cv::Scalar b, float wb) {
    float wa = 1.0f - wb;
    return cv::Scalar(wa * a[0] + wb * b[0],
                      wa * a[1] + wb * b[1],
                      wa * a[2] + wb * b[2]);
}

static cv::Scalar field_color(float x, float y) {
    float c = clampf(learned_traversability_cost(x, y) / 5.0f, 0.0f, 1.0f);
    float exp_prior = experience_prior(x, y);
    cv::Scalar terrain(42 + 40 * c, 76 + 98 * (1.0f - c), 55 + 174 * c);
    cv::Scalar expc(52, 150 + 52 * exp_prior, 120 + 78 * exp_prior);
    if (raw_clearance(x, y) < -0.02f) terrain = cv::Scalar(50, 39, 116);
    return blend(terrain, expc, 0.22f * exp_prior);
}

static void draw_field(cv::Mat& img, int x0) {
    int cw = std::max(1, HALF_W / FIELD_W + 1);
    int ch = std::max(1, MAP_H / FIELD_H + 1);
    for (int iy = 0; iy < FIELD_H; iy++) {
        for (int ix = 0; ix < FIELD_W; ix++) {
            float x = (static_cast<float>(ix) + 0.5f) / FIELD_W * WORLD_W;
            float y = (static_cast<float>(iy) + 0.5f) / FIELD_H * WORLD_H;
            cv::Point p = to_px(x, y, x0);
            cv::rectangle(img, cv::Rect(p.x, p.y, cw, ch), field_color(x, y), cv::FILLED);
        }
    }
}

static void draw_experience_graph(cv::Mat& img, int x0) {
    for (int iy = 0; iy < GRAPH_H; iy++) {
        for (int ix = 0; ix < GRAPH_W; ix++) {
            float x = (static_cast<float>(ix) + 0.5f) / GRAPH_W * WORLD_W;
            float y = (static_cast<float>(iy) + 0.5f) / GRAPH_H * WORLD_H;
            if (raw_clearance(x, y) < -0.02f) continue;
            cv::Point p = to_px(x, y, x0);
            cv::circle(img, p, 1, cv::Scalar(76, 164, 132), cv::FILLED, cv::LINE_AA);
            if (ix + 1 < GRAPH_W) {
                float nx = (static_cast<float>(ix + 1) + 0.5f) / GRAPH_W * WORLD_W;
                float ny = y;
                cv::line(img, p, to_px(nx, ny, x0), cv::Scalar(52, 118, 94), 1, cv::LINE_AA);
            }
            if (iy + 1 < GRAPH_H) {
                float nx = x;
                float ny = (static_cast<float>(iy + 1) + 0.5f) / GRAPH_H * WORLD_H;
                cv::line(img, p, to_px(nx, ny, x0), cv::Scalar(52, 118, 94), 1, cv::LINE_AA);
            }
        }
    }
}

static void draw_route(cv::Mat& img,
                       const std::vector<cv::Point2f>& route,
                       int x0,
                       cv::Scalar color,
                       int thickness) {
    for (size_t i = 1; i < route.size(); i++) {
        cv::line(img, to_px(route[i - 1].x, route[i - 1].y, x0),
                 to_px(route[i].x, route[i].y, x0), color, thickness, cv::LINE_AA);
    }
}

static void draw_dynamic_obstacles(cv::Mat& img, int x0, float t) {
    for (int k = 0; k < 4; k++) {
        float ox, oy, radius;
        moving_obstacle(k, t, ox, oy, radius);
        int px_radius = std::max(7, static_cast<int>(radius / WORLD_W * HALF_W));
        cv::circle(img, to_px(ox, oy, x0), px_radius + 8,
                   cv::Scalar(32, 72, 138), 1, cv::LINE_AA);
        cv::circle(img, to_px(ox, oy, x0), px_radius,
                   cv::Scalar(50, 118, 242), cv::FILLED, cv::LINE_AA);
        cv::circle(img, to_px(ox, oy, x0), px_radius,
                   cv::Scalar(245, 232, 166), 1, cv::LINE_AA);
    }
}

static void draw_interaction_overlay(cv::Mat& img,
                                     const std::vector<float>& interaction_field,
                                     int x0,
                                     int step) {
    int k = std::max(0, std::min(step, HORIZON - 1));
    int cw = std::max(2, HALF_W / FIELD_W + 1);
    int ch = std::max(2, MAP_H / FIELD_H + 1);
    for (int iy = 0; iy < FIELD_H; iy += 2) {
        for (int ix = 0; ix < FIELD_W; ix += 2) {
            int idx = k * FIELD_CELLS + iy * FIELD_W + ix;
            float v = clampf(interaction_field[idx] / 3.0f, 0.0f, 1.0f);
            if (v < 0.08f) continue;
            float x = (static_cast<float>(ix) + 0.5f) / FIELD_W * WORLD_W;
            float y = (static_cast<float>(iy) + 0.5f) / FIELD_H * WORLD_H;
            cv::Point p = to_px(x, y, x0);
            cv::Scalar c(32 + 70 * v, 40 + 72 * (1.0f - v), 180 + 65 * v);
            cv::rectangle(img, cv::Rect(p.x, p.y, cw * 2, ch * 2), c, cv::FILLED);
        }
    }
}

static void draw_social_agents(cv::Mat& img,
                               const std::vector<SocialAgent>& agents,
                               int x0,
                               int step) {
    std::vector<cv::Point2f> pos(agents.size());
    std::vector<cv::Point2f> vel(agents.size());
    for (size_t i = 0; i < agents.size(); i++) {
        float x, y, vx, vy;
        social_state(agents[i], step, x, y, vx, vy);
        pos[i] = cv::Point2f(x, y);
        vel[i] = cv::Point2f(vx, vy);
    }
    for (size_t i = 0; i < agents.size(); i++) {
        for (size_t j = i + 1; j < agents.size(); j++) {
            float risk = pair_seed_risk(pos[i].x, pos[i].y, vel[i].x, vel[i].y, agents[i].radius,
                                        pos[j].x, pos[j].y, vel[j].x, vel[j].y, agents[j].radius,
                                        agents[i].route, agents[j].route);
            if (risk < 0.28f) continue;
            cv::line(img, to_px(pos[i].x, pos[i].y, x0), to_px(pos[j].x, pos[j].y, x0),
                     cv::Scalar(80, 120 + 80 * risk, 245), 1, cv::LINE_AA);
        }
    }
    for (size_t i = 0; i < agents.size(); i++) {
        if (pos[i].x < -0.2f || pos[i].x > WORLD_W + 0.2f ||
            pos[i].y < -0.2f || pos[i].y > WORLD_H + 0.2f) {
            continue;
        }
        int radius = std::max(4, static_cast<int>(agents[i].radius / WORLD_W * HALF_W));
        cv::Scalar fill = agents[i].kind ? cv::Scalar(55, 220, 255) : cv::Scalar(78, 170, 245);
        cv::circle(img, to_px(pos[i].x, pos[i].y, x0), radius + 3,
                   cv::Scalar(10, 22, 46), cv::FILLED, cv::LINE_AA);
        cv::circle(img, to_px(pos[i].x, pos[i].y, x0), radius, fill, cv::FILLED, cv::LINE_AA);
    }
}

static void draw_samples(cv::Mat& img,
                         const std::vector<float>& sx,
                         const std::vector<float>& sy,
                         int x0,
                         int upto,
                         cv::Scalar color) {
    int step_limit = std::max(1, std::min(upto, HORIZON));
    for (int r = 0; r < SAMPLE_TRAJS; r += 2) {
        for (int k = 1; k < step_limit; k++) {
            int a = r * HORIZON + k - 1;
            int b = r * HORIZON + k;
            cv::line(img, to_px(sx[a], sy[a], x0), to_px(sx[b], sy[b], x0),
                     color, 1, cv::LINE_AA);
        }
    }
}

static void draw_path(cv::Mat& img,
                      const std::vector<cv::Point2f>& path,
                      int x0,
                      int upto,
                      cv::Scalar outer,
                      cv::Scalar inner) {
    int step_limit = std::max(1, std::min(upto, static_cast<int>(path.size())));
    for (int k = 1; k < step_limit; k++) {
        cv::line(img, to_px(path[k - 1].x, path[k - 1].y, x0),
                 to_px(path[k].x, path[k].y, x0), outer, 4, cv::LINE_AA);
        cv::line(img, to_px(path[k - 1].x, path[k - 1].y, x0),
                 to_px(path[k].x, path[k].y, x0), inner, 2, cv::LINE_AA);
    }
}

static void draw_markers(cv::Mat& img, int x0) {
    cv::circle(img, to_px(START_X, START_Y, x0), 7, cv::Scalar(255, 245, 125),
               cv::FILLED, cv::LINE_AA);
    cv::circle(img, to_px(GOAL_X, GOAL_Y, x0), 8, cv::Scalar(245, 120, 255),
               2, cv::LINE_AA);
}

static void draw_legend(cv::Mat& img) {
    constexpr int x0 = PANEL_W - 252;
    constexpr int y0 = HEADER_H + 48;
    cv::rectangle(img, cv::Rect(x0, y0, 232, 124), cv::Scalar(8, 10, 13), cv::FILLED);
    cv::line(img, cv::Point(x0 + 14, y0 + 20), cv::Point(x0 + 36, y0 + 20),
             cv::Scalar(92, 230, 190), 2, cv::LINE_AA);
    cv::putText(img, "graph guide", cv::Point(x0 + 44, y0 + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::line(img, cv::Point(x0 + 14, y0 + 48), cv::Point(x0 + 36, y0 + 48),
             cv::Scalar(102, 154, 235), 1, cv::LINE_AA);
    cv::putText(img, "aware rollouts", cv::Point(x0 + 44, y0 + 53),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::line(img, cv::Point(x0 + 14, y0 + 76), cv::Point(x0 + 36, y0 + 76),
             cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
    cv::putText(img, "aware best", cv::Point(x0 + 44, y0 + 81),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::line(img, cv::Point(x0 + 14, y0 + 104), cv::Point(x0 + 36, y0 + 104),
             cv::Scalar(210, 116, 255), 2, cv::LINE_AA);
    cv::putText(img, "blind best", cv::Point(x0 + 44, y0 + 109),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
}

static cv::Mat draw_frame(const std::vector<cv::Point2f>& route,
                          const std::vector<SocialAgent>& social_agents,
                          const std::vector<float>& interaction_field,
                          const std::vector<float>& sample_x,
                          const std::vector<float>& sample_y,
                          const std::vector<cv::Point2f>& guided_best_path,
                          const std::vector<cv::Point2f>& unguided_best_path,
                          const RolloutResult& guided_best,
                          const RolloutResult& unguided_best,
                          double gpu_ms,
                          double cpu_ms,
                          int step) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 20, 24));
    draw_field(img, 0);
    draw_field(img, HALF_W);
    draw_interaction_overlay(img, interaction_field, HALF_W, step);
    draw_experience_graph(img, 0);
    draw_route(img, route, 0, cv::Scalar(92, 230, 190), 2);
    draw_route(img, route, HALF_W, cv::Scalar(92, 230, 190), 2);
    float t = static_cast<float>(step) * DT;
    draw_dynamic_obstacles(img, 0, t);
    draw_dynamic_obstacles(img, HALF_W, t);
    draw_social_agents(img, social_agents, 0, step);
    draw_social_agents(img, social_agents, HALF_W, step);
    draw_samples(img, sample_x, sample_y, HALF_W, step, cv::Scalar(102, 154, 235));
    draw_path(img, unguided_best_path, HALF_W, step, cv::Scalar(42, 20, 65),
              cv::Scalar(210, 116, 255));
    draw_path(img, guided_best_path, HALF_W, step, cv::Scalar(255, 255, 255),
              cv::Scalar(92, 230, 190));
    draw_path(img, guided_best_path, 0, step, cv::Scalar(255, 255, 255),
              cv::Scalar(92, 230, 190));
    draw_markers(img, 0);
    draw_markers(img, HALF_W);
    draw_legend(img);

    cv::line(img, cv::Point(HALF_W, HEADER_H), cv::Point(HALF_W, PANEL_H - FOOTER_H),
             cv::Scalar(18, 18, 20), 1, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, HEADER_H), cv::Scalar(5, 7, 10), cv::FILLED);
    cv::rectangle(img, cv::Rect(0, PANEL_H - FOOTER_H, PANEL_W, FOOTER_H),
                  cv::Scalar(5, 7, 10), cv::FILLED);

    double speedup = gpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0;
    float cost_drop = 100.0f * (1.0f - guided_best.cost / fmaxf(unguided_best.cost, 1.0e-6f));
    float interaction_drop = 100.0f * (1.0f - guided_best.mean_interaction
                                             / fmaxf(unguided_best.mean_interaction, 1.0e-6f));
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "GPU interaction-graph neural MPPI  %d rollouts x H=%d  gpu=%.2f ms  cpu=%.0f ms  %.1fx",
                  ROLLOUTS, HORIZON, gpu_ms, cpu_ms, speedup);
    cv::putText(img, buf, cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "agent interaction graph + route prior", cv::Point(14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img, "interaction-aware MPPI rollouts", cv::Point(HALF_W + 14, HEADER_H + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf),
                  "cost %.1f -> %.1f (-%.1f%%)  social %.3f -> %.3f (-%.1f%%)  clearance %.2f -> %.2f",
                  unguided_best.cost, guided_best.cost, cost_drop,
                  unguided_best.mean_interaction, guided_best.mean_interaction, interaction_drop,
                  unguided_best.min_clearance, guided_best.min_clearance);
    cv::putText(img, buf, cv::Point(12, PANEL_H - 16),
                cv::FONT_HERSHEY_SIMPLEX, 0.49, cv::Scalar(225, 238, 245), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<cv::Point2f> route = make_guide_route();
    std::vector<float> route_x(GUIDE_POINTS);
    std::vector<float> route_y(GUIDE_POINTS);
    for (int i = 0; i < GUIDE_POINTS; i++) {
        route_x[i] = route[i].x;
        route_y[i] = route[i].y;
    }

    std::vector<float> cost_field;
    std::vector<float> risk_field;
    std::vector<float> route_field;
    std::vector<float> exp_field;
    make_cost_fields(route_x, route_y, cost_field, risk_field, route_field, exp_field);
    std::vector<SocialAgent> social_agents = make_social_agents();
    std::vector<float> interaction_field;
    std::vector<float> clearance_field;
    make_interaction_fields_gpu(social_agents, interaction_field, clearance_field);

    std::vector<RolloutResult> cpu_guided_results;
    std::vector<RolloutResult> cpu_unguided_results;
    double cpu_unguided_ms = evaluate_cpu_rollouts(0, route_x, route_y, cost_field, risk_field,
                                                   route_field, interaction_field, clearance_field,
                                                   exp_field, cpu_unguided_results);
    double cpu_guided_ms = evaluate_cpu_rollouts(1, route_x, route_y, cost_field, risk_field,
                                                 route_field, interaction_field, clearance_field,
                                                 exp_field, cpu_guided_results);
    double cpu_ms = cpu_unguided_ms + cpu_guided_ms;

    float* d_route_x = nullptr;
    float* d_route_y = nullptr;
    float* d_cost_field = nullptr;
    float* d_risk_field = nullptr;
    float* d_route_field = nullptr;
    float* d_interaction_field = nullptr;
    float* d_clearance_field = nullptr;
    float* d_exp_field = nullptr;
    float* d_sample_x = nullptr;
    float* d_sample_y = nullptr;
    RolloutResult* d_guided = nullptr;
    RolloutResult* d_unguided = nullptr;
    CUDA_CHECK(cudaMalloc(&d_route_x, GUIDE_POINTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_route_y, GUIDE_POINTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cost_field, FIELD_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_risk_field, static_cast<size_t>(HORIZON) * FIELD_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_route_field, FIELD_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_interaction_field, static_cast<size_t>(HORIZON) * FIELD_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_clearance_field, static_cast<size_t>(HORIZON) * FIELD_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_exp_field, FIELD_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sample_x, SAMPLE_TRAJS * HORIZON * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sample_y, SAMPLE_TRAJS * HORIZON * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_guided, ROLLOUTS * sizeof(RolloutResult)));
    CUDA_CHECK(cudaMalloc(&d_unguided, ROLLOUTS * sizeof(RolloutResult)));
    CUDA_CHECK(cudaMemcpy(d_route_x, route_x.data(), GUIDE_POINTS * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_route_y, route_y.data(), GUIDE_POINTS * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cost_field, cost_field.data(), FIELD_CELLS * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_risk_field, risk_field.data(),
                          static_cast<size_t>(HORIZON) * FIELD_CELLS * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_route_field, route_field.data(), FIELD_CELLS * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interaction_field, interaction_field.data(),
                          static_cast<size_t>(HORIZON) * FIELD_CELLS * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clearance_field, clearance_field.data(),
                          static_cast<size_t>(HORIZON) * FIELD_CELLS * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_exp_field, exp_field.data(), FIELD_CELLS * sizeof(float),
                          cudaMemcpyHostToDevice));

    int blocks = (ROLLOUTS + THREADS - 1) / THREADS;
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    rollout_kernel<<<blocks, THREADS>>>(0, d_route_x, d_route_y, d_cost_field,
                                        d_risk_field, d_route_field, d_interaction_field,
                                        d_clearance_field, d_exp_field, d_unguided,
                                        d_sample_x, d_sample_y);
    rollout_kernel<<<blocks, THREADS>>>(1, d_route_x, d_route_y, d_cost_field,
                                        d_risk_field, d_route_field, d_interaction_field,
                                        d_clearance_field, d_exp_field, d_guided,
                                        d_sample_x, d_sample_y);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float gpu_ms_f = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms_f, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaGetLastError());
    double gpu_ms = static_cast<double>(gpu_ms_f);

    std::vector<RolloutResult> guided_results(ROLLOUTS);
    std::vector<RolloutResult> unguided_results(ROLLOUTS);
    std::vector<float> sample_x(SAMPLE_TRAJS * HORIZON);
    std::vector<float> sample_y(SAMPLE_TRAJS * HORIZON);
    CUDA_CHECK(cudaMemcpy(guided_results.data(), d_guided, ROLLOUTS * sizeof(RolloutResult),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(unguided_results.data(), d_unguided, ROLLOUTS * sizeof(RolloutResult),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sample_x.data(), d_sample_x, SAMPLE_TRAJS * HORIZON * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sample_y.data(), d_sample_y, SAMPLE_TRAJS * HORIZON * sizeof(float),
                          cudaMemcpyDeviceToHost));

    int guided_idx = best_rollout_index(guided_results);
    int unguided_idx = best_rollout_index(unguided_results);
    RolloutResult guided_best = guided_results[guided_idx];
    RolloutResult unguided_best = unguided_results[unguided_idx];
    std::vector<cv::Point2f> guided_path = reconstruct_trajectory(guided_idx, 1, route_x, route_y,
                                                                  cost_field, risk_field, route_field,
                                                                  interaction_field, clearance_field,
                                                                  exp_field);
    std::vector<cv::Point2f> unguided_path = reconstruct_trajectory(unguided_idx, 0, route_x, route_y,
                                                                    cost_field, risk_field, route_field,
                                                                    interaction_field, clearance_field,
                                                                    exp_field);

    double speedup = gpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0;
    float cost_drop = 100.0f * (1.0f - guided_best.cost / fmaxf(unguided_best.cost, 1.0e-6f));
    float terminal_drop = 100.0f * (1.0f - guided_best.terminal_error
                                          / fmaxf(unguided_best.terminal_error, 1.0e-6f));
    float interaction_drop = 100.0f * (1.0f - guided_best.mean_interaction
                                             / fmaxf(unguided_best.mean_interaction, 1.0e-6f));
    float clearance_gain = guided_best.min_clearance - unguided_best.min_clearance;
    std::printf("CPU blind+aware interaction-graph MPPI: %.3f ms (%d rollouts x H=%d x 2 batches; aware %.3f ms)\n",
                cpu_ms, ROLLOUTS, HORIZON, cpu_guided_ms);
    std::printf("GPU interaction-graph neural MPPI: %.3f ms (aware+blind batches, %.1fx vs CPU equivalent rollout eval)\n",
                gpu_ms, speedup);
    std::printf("Blind best: cost %.3f, terminal %.3f, dynamic risk avg/max %.3f/%.3f, interaction avg/max %.3f/%.3f, min clearance %.3f, route error %.3f\n",
                unguided_best.cost, unguided_best.terminal_error, unguided_best.mean_risk,
                unguided_best.max_risk, unguided_best.mean_interaction,
                unguided_best.max_interaction, unguided_best.min_clearance,
                unguided_best.mean_route_error);
    std::printf("Aware best: cost %.3f, terminal %.3f, dynamic risk avg/max %.3f/%.3f, interaction avg/max %.3f/%.3f, min clearance %.3f, route error %.3f, cost improvement %.1f%%, terminal improvement %.1f%%, interaction improvement %.1f%%, clearance gain %.3f\n",
                guided_best.cost, guided_best.terminal_error, guided_best.mean_risk,
                guided_best.max_risk, guided_best.mean_interaction, guided_best.max_interaction,
                guided_best.min_clearance, guided_best.mean_route_error, cost_drop,
                terminal_drop, interaction_drop, clearance_gain);

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_interaction_graph_neural_mppi.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_interaction_graph_neural_mppi.avi\n");
        return 1;
    }
    for (int k = 2; k <= HORIZON; k += 2) {
        video.write(draw_frame(route, social_agents, interaction_field,
                               sample_x, sample_y, guided_path, unguided_path,
                               guided_best, unguided_best, gpu_ms, cpu_ms, k));
    }
    for (int i = 0; i < 12; i++) {
        video.write(draw_frame(route, social_agents, interaction_field,
                               sample_x, sample_y, guided_path, unguided_path,
                               guided_best, unguided_best, gpu_ms, cpu_ms, HORIZON));
    }
    video.release();

    avi_to_gif("gif/gpu_interaction_graph_neural_mppi.avi",
               "gif/gpu_interaction_graph_neural_mppi.gif", 8, 560);
    std::printf("GIF saved to gif/gpu_interaction_graph_neural_mppi.gif\n");

    CUDA_CHECK(cudaFree(d_route_x));
    CUDA_CHECK(cudaFree(d_route_y));
    CUDA_CHECK(cudaFree(d_cost_field));
    CUDA_CHECK(cudaFree(d_risk_field));
    CUDA_CHECK(cudaFree(d_route_field));
    CUDA_CHECK(cudaFree(d_interaction_field));
    CUDA_CHECK(cudaFree(d_clearance_field));
    CUDA_CHECK(cudaFree(d_exp_field));
    CUDA_CHECK(cudaFree(d_sample_x));
    CUDA_CHECK(cudaFree(d_sample_y));
    CUDA_CHECK(cudaFree(d_guided));
    CUDA_CHECK(cudaFree(d_unguided));
    return 0;
}
