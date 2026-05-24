// gpu_mcts_planner.cu
//
// Root-parallel GPU Monte Carlo Tree Search for a small 2D kinodynamic
// planning problem.  Each CUDA thread evaluates one stochastic rollout from
// a root action; root action statistics are accumulated with atomics and used
// by later rollout batches through a UCB score.
//
// Output: gif/gpu_mcts_planner.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int N_SCENARIOS = 64;
constexpr int N_ACTIONS = 7;
constexpr int MCTS_BATCHES = 8;
constexpr int ROLLOUTS_PER_BATCH = 512;
constexpr int ROLLOUTS_PER_SCENE = MCTS_BATCHES * ROLLOUTS_PER_BATCH;
constexpr int HORIZON = 48;
constexpr int MAX_OBS = 8;
constexpr int N_FRAMES = 92;
constexpr int VIDEO_FPS = 12;

constexpr float WORLD_W = 12.0f;
constexpr float WORLD_H = 8.0f;
constexpr float DT = 0.13f;
constexpr float BASE_SPEED = 1.34f;
constexpr float ACTION_DW = 0.46f;
constexpr float ROBOT_R = 0.15f;
constexpr float GOAL_R = 0.32f;
constexpr float PI_F = 3.14159265358979323846f;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;

struct Pose {
    float x;
    float y;
    float yaw;
};

struct Circle {
    float x;
    float y;
    float r;
};

struct Scene {
    Pose start;
    float goal_x;
    float goal_y;
    Circle obs[MAX_OBS];
    int n_obs;
};

struct RootStats {
    float sum[N_ACTIONS];
    int visits[N_ACTIONS];
};

struct PlanBench {
    float gpu_ms = 0.0f;
    double cpu_ms = 0.0;
    double speedup = 0.0;
};

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline int clampi(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__host__ __device__ static inline float wrap_angle(float a) {
    while (a > PI_F) a -= 2.0f * PI_F;
    while (a < -PI_F) a += 2.0f * PI_F;
    return a;
}

__host__ __device__ static inline float sqr(float x) {
    return x * x;
}

__host__ __device__ static inline float dist_to_goal(const Scene& scene, const Pose& p) {
    return sqrtf(sqr(scene.goal_x - p.x) + sqr(scene.goal_y - p.y));
}

__host__ __device__ static inline float action_w(int action) {
    return ((float)action - 0.5f * (float)(N_ACTIONS - 1)) * ACTION_DW;
}

__host__ __device__ static inline Pose integrate(const Pose& p, int action) {
    Pose q = p;
    float w = action_w(action);
    float speed_scale = 1.0f - 0.06f * fabsf((float)action - 3.0f);
    float v = BASE_SPEED * speed_scale;
    q.yaw = wrap_angle(q.yaw + w * DT);
    q.x += v * cosf(q.yaw) * DT;
    q.y += v * sinf(q.yaw) * DT;
    return q;
}

__host__ __device__ static inline float clearance(const Scene& scene, float x, float y) {
    float m = fminf(fminf(x, WORLD_W - x), fminf(y, WORLD_H - y)) - ROBOT_R;
    for (int i = 0; i < scene.n_obs; i++) {
        float d = sqrtf(sqr(x - scene.obs[i].x) + sqr(y - scene.obs[i].y))
                - scene.obs[i].r - ROBOT_R;
        m = fminf(m, d);
    }
    return m;
}

__host__ __device__ static inline uint32_t mix_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__host__ __device__ static inline float rand01(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return (float)((state >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
}

__host__ __device__ static inline int greedy_action_to_goal(const Scene& scene,
                                                            const Pose& p) {
    float desired = atan2f(scene.goal_y - p.y, scene.goal_x - p.x);
    float err = wrap_angle(desired - p.yaw);
    int offset = (int)floorf(err / ACTION_DW + (err >= 0.0f ? 0.5f : -0.5f));
    return clampi(3 + offset, 0, N_ACTIONS - 1);
}

__host__ __device__ static inline int rollout_policy(const Scene& scene,
                                                     const Pose& p,
                                                     uint32_t& rng) {
    int greedy = greedy_action_to_goal(scene, p);
    if (rand01(rng) > 0.82f) {
        return (int)(rand01(rng) * (float)N_ACTIONS) % N_ACTIONS;
    }

    float best_score = -1.0e30f;
    int best_action = greedy;
    float d0 = dist_to_goal(scene, p);
    for (int da = -2; da <= 2; da++) {
        int action = clampi(greedy + da, 0, N_ACTIONS - 1);
        Pose q = integrate(p, action);
        float d1 = dist_to_goal(scene, q);
        float c = clearance(scene, q.x, q.y);
        float score = 6.0f * (d0 - d1) + 1.3f * clampf(c, -0.4f, 0.9f)
                    - 0.018f * fabsf((float)action - 3.0f);
        if (score > best_score) {
            best_score = score;
            best_action = action;
        }
    }

    if (rand01(rng) < 0.22f) {
        best_action = clampi(best_action + ((int)(rand01(rng) * 3.0f) - 1),
                             0, N_ACTIONS - 1);
    }
    return best_action;
}

__host__ __device__ static inline float root_prior(const Scene& scene,
                                                   const Pose& p,
                                                   int action) {
    Pose q = integrate(p, action);
    float progress = dist_to_goal(scene, p) - dist_to_goal(scene, q);
    float c = clearance(scene, q.x, q.y);
    return 8.0f * progress + 1.5f * clampf(c, -0.4f, 0.9f)
         - 0.03f * fabsf((float)action - 3.0f);
}

__host__ __device__ static inline float simulate_rollout(const Scene& scene,
                                                         const Pose& start,
                                                         int first_action,
                                                         uint32_t seed) {
    Pose p = start;
    float prev_dist = dist_to_goal(scene, p);
    float reward = 0.0f;

    for (int step = 0; step < HORIZON; step++) {
        int action = (step == 0) ? first_action : rollout_policy(scene, p, seed);
        Pose q = integrate(p, action);
        float c = clearance(scene, q.x, q.y);
        float d = dist_to_goal(scene, q);

        reward += 9.5f * (prev_dist - d) - 0.025f;
        reward -= 0.012f * fabsf((float)action - 3.0f);
        if (c < 0.0f) {
            reward -= 70.0f + 1.5f * (float)(HORIZON - step);
            return reward;
        }
        if (c < 0.40f) reward -= 2.2f * (0.40f - c);
        if (d < GOAL_R) {
            reward += 120.0f + 2.0f * (float)(HORIZON - step);
            return reward;
        }

        p = q;
        prev_dist = d;
    }

    float final_clearance = clearance(scene, p.x, p.y);
    reward -= 2.7f * prev_dist;
    reward += 0.25f * clampf(final_clearance, -0.5f, 1.0f);
    return reward;
}

__host__ __device__ static inline int select_root_action(const Scene& scene,
                                                         const Pose& pose,
                                                         const float* sums,
                                                         const int* visits,
                                                         int local_sim) {
    int total = 0;
    for (int a = 0; a < N_ACTIONS; a++) total += visits[a];
    if (total < N_ACTIONS * 2) {
        return (local_sim + total) % N_ACTIONS;
    }

    float log_total = logf((float)total + 2.0f);
    float best_score = -1.0e30f;
    int best_action = 3;
    for (int a = 0; a < N_ACTIONS; a++) {
        int n = visits[a];
        float mean = (n > 0) ? sums[a] / (float)n : -1.0e20f;
        float explore = (n > 0) ? 6.5f * sqrtf(log_total / (float)n) : 1.0e6f;
        float score = mean + explore + 0.18f * root_prior(scene, pose, a);
        if (score > best_score) {
            best_score = score;
            best_action = a;
        }
    }
    return best_action;
}

__global__ void reset_stats_kernel(float* sums, int* visits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = N_SCENARIOS * N_ACTIONS;
    if (idx >= n) return;
    sums[idx] = 0.0f;
    visits[idx] = 0;
}

__global__ void mcts_rollout_kernel(const Scene* __restrict__ scenes,
                                    const Pose* __restrict__ poses,
                                    float* __restrict__ sums,
                                    int* __restrict__ visits,
                                    int batch_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = N_SCENARIOS * ROLLOUTS_PER_BATCH;
    if (idx >= n) return;

    int scenario = idx / ROLLOUTS_PER_BATCH;
    int local = idx - scenario * ROLLOUTS_PER_BATCH;
    const Scene& scene = scenes[scenario];
    Pose pose = poses[scenario];
    int base = scenario * N_ACTIONS;

    int root_action = select_root_action(scene, pose, sums + base, visits + base,
                                         local + batch_id * ROLLOUTS_PER_BATCH);
    uint32_t seed = mix_u32((uint32_t)(0x9e3779b9u
                      ^ (uint32_t)(scenario * 4099)
                      ^ (uint32_t)(batch_id * 65537)
                      ^ (uint32_t)(local * 17)));
    float reward = simulate_rollout(scene, pose, root_action, seed);
    atomicAdd(sums + base + root_action, reward);
    atomicAdd(visits + base + root_action, 1);
}

static std::vector<Scene> make_scenes() {
    std::vector<Scene> scenes(N_SCENARIOS);
    const Circle base_obs[MAX_OBS] = {
        {3.0f, 1.8f, 0.68f},
        {4.7f, 3.8f, 0.86f},
        {3.4f, 6.2f, 0.60f},
        {6.4f, 2.1f, 0.70f},
        {7.1f, 5.2f, 0.95f},
        {9.0f, 3.3f, 0.72f},
        {9.2f, 6.2f, 0.56f},
        {5.6f, 6.9f, 0.44f},
    };

    for (int s = 0; s < N_SCENARIOS; s++) {
        float a = (float)(s % 8) - 3.5f;
        float b = (float)(s / 8) - 3.5f;
        scenes[s].start = {0.85f, 1.05f + 0.055f * a, 0.20f + 0.025f * b};
        scenes[s].goal_x = 11.05f - 0.025f * b;
        scenes[s].goal_y = 7.05f + 0.045f * a;
        scenes[s].n_obs = MAX_OBS;
        for (int i = 0; i < MAX_OBS; i++) {
            scenes[s].obs[i] = base_obs[i];
            scenes[s].obs[i].x += 0.040f * b * sinf(0.7f * (float)(i + 1));
            scenes[s].obs[i].y += 0.035f * a * cosf(0.8f * (float)(i + 2));
        }
    }
    return scenes;
}

static float gpu_plan(const std::vector<Scene>& scenes,
                      const std::vector<Pose>& poses,
                      Scene* d_scenes,
                      Pose* d_poses,
                      float* d_sums,
                      int* d_visits,
                      std::vector<RootStats>& stats) {
    CUDA_CHECK(cudaMemcpy(d_scenes, scenes.data(),
                          scenes.size() * sizeof(Scene), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_poses, poses.data(),
                          poses.size() * sizeof(Pose), cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));

    int stats_n = N_SCENARIOS * N_ACTIONS;
    reset_stats_kernel<<<(stats_n + 255) / 256, 256>>>(d_sums, d_visits);
    for (int batch = 0; batch < MCTS_BATCHES; batch++) {
        int n = N_SCENARIOS * ROLLOUTS_PER_BATCH;
        mcts_rollout_kernel<<<(n + 255) / 256, 256>>>(d_scenes, d_poses,
                                                      d_sums, d_visits, batch);
    }

    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    std::vector<float> sums(stats_n);
    std::vector<int> visits(stats_n);
    CUDA_CHECK(cudaMemcpy(sums.data(), d_sums, stats_n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(visits.data(), d_visits, stats_n * sizeof(int),
                          cudaMemcpyDeviceToHost));

    stats.resize(N_SCENARIOS);
    for (int s = 0; s < N_SCENARIOS; s++) {
        for (int a = 0; a < N_ACTIONS; a++) {
            stats[s].sum[a] = sums[s * N_ACTIONS + a];
            stats[s].visits[a] = visits[s * N_ACTIONS + a];
        }
    }
    return ms;
}

static double cpu_plan(const std::vector<Scene>& scenes,
                       const std::vector<Pose>& poses,
                       std::vector<RootStats>& stats) {
    stats.assign(N_SCENARIOS, RootStats{});
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int batch = 0; batch < MCTS_BATCHES; batch++) {
        for (int s = 0; s < N_SCENARIOS; s++) {
            const Scene& scene = scenes[s];
            const Pose& pose = poses[s];
            for (int local = 0; local < ROLLOUTS_PER_BATCH; local++) {
                int sim = local + batch * ROLLOUTS_PER_BATCH;
                int action = select_root_action(scene, pose, stats[s].sum,
                                                stats[s].visits, sim);
                uint32_t seed = mix_u32((uint32_t)(0x9e3779b9u
                                  ^ (uint32_t)(s * 4099)
                                  ^ (uint32_t)(batch * 65537)
                                  ^ (uint32_t)(local * 17)));
                float reward = simulate_rollout(scene, pose, action, seed);
                stats[s].sum[action] += reward;
                stats[s].visits[action] += 1;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static int best_action_from_stats(const RootStats& stats) {
    int best = 3;
    float best_mean = -1.0e30f;
    for (int a = 0; a < N_ACTIONS; a++) {
        float mean = stats.visits[a] > 0 ? stats.sum[a] / (float)stats.visits[a]
                                         : -1.0e30f;
        if (mean > best_mean) {
            best_mean = mean;
            best = a;
        }
    }
    return best;
}

static int safe_best_action(const Scene& scene, const Pose& pose,
                            const RootStats& stats) {
    std::vector<int> order(N_ACTIONS);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&stats](int a, int b) {
        float ma = stats.visits[a] > 0 ? stats.sum[a] / (float)stats.visits[a]
                                       : -1.0e30f;
        float mb = stats.visits[b] > 0 ? stats.sum[b] / (float)stats.visits[b]
                                       : -1.0e30f;
        return ma > mb;
    });

    for (int a : order) {
        Pose q = integrate(pose, a);
        if (clearance(scene, q.x, q.y) >= 0.0f) return a;
    }
    return 3;
}

static cv::Point world_to_px(float x, float y, const cv::Rect& r) {
    float u = clampf(x / WORLD_W, 0.0f, 1.0f);
    float v = clampf(y / WORLD_H, 0.0f, 1.0f);
    return cv::Point(r.x + (int)(u * (float)r.width),
                     r.y + r.height - (int)(v * (float)r.height));
}

static cv::Scalar action_color(int action) {
    static const cv::Scalar colors[N_ACTIONS] = {
        cv::Scalar(88, 132, 255),
        cv::Scalar(74, 185, 255),
        cv::Scalar(88, 225, 226),
        cv::Scalar(112, 230, 132),
        cv::Scalar(212, 220, 90),
        cv::Scalar(244, 159, 82),
        cv::Scalar(245, 102, 112),
    };
    return colors[action];
}

static void draw_rollout_fan(cv::Mat& img,
                             const Scene& scene,
                             const Pose& pose,
                             const RootStats& stats,
                             const cv::Rect& map_rect,
                             int frame_idx) {
    float min_mean = 1.0e30f;
    float max_mean = -1.0e30f;
    for (int a = 0; a < N_ACTIONS; a++) {
        if (stats.visits[a] <= 0) continue;
        float m = stats.sum[a] / (float)stats.visits[a];
        min_mean = std::min(min_mean, m);
        max_mean = std::max(max_mean, m);
    }
    float denom = std::max(1.0f, max_mean - min_mean);

    for (int a = 0; a < N_ACTIONS; a++) {
        float mean = stats.visits[a] > 0 ? stats.sum[a] / (float)stats.visits[a]
                                         : min_mean;
        float q = clampf((mean - min_mean) / denom, 0.0f, 1.0f);
        cv::Scalar c = action_color(a);
        c *= 0.25 + 0.55 * q;

        int samples = 5 + (int)(5.0f * q);
        for (int k = 0; k < samples; k++) {
            Pose p = pose;
            uint32_t seed = mix_u32((uint32_t)(frame_idx * 8191 + a * 977 + k * 43));
            std::vector<cv::Point> pts;
            pts.push_back(world_to_px(p.x, p.y, map_rect));
            for (int t = 0; t < 22; t++) {
                int action = (t == 0) ? a : rollout_policy(scene, p, seed);
                p = integrate(p, action);
                if (clearance(scene, p.x, p.y) < 0.0f) break;
                pts.push_back(world_to_px(p.x, p.y, map_rect));
                if (dist_to_goal(scene, p) < GOAL_R) break;
            }
            if (pts.size() > 1) cv::polylines(img, pts, false, c, 1, cv::LINE_AA);
        }
    }
}

static void draw_stats_bars(cv::Mat& img,
                            const RootStats& stats,
                            const cv::Rect& r,
                            int best_action) {
    cv::rectangle(img, r, cv::Scalar(28, 30, 34), -1);
    cv::rectangle(img, r, cv::Scalar(78, 82, 90), 1);
    cv::putText(img, "root action values", cv::Point(r.x + 12, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(235, 235, 235),
                1, cv::LINE_AA);

    float min_mean = 1.0e30f;
    float max_mean = -1.0e30f;
    int max_visits = 1;
    for (int a = 0; a < N_ACTIONS; a++) {
        if (stats.visits[a] <= 0) continue;
        float mean = stats.sum[a] / (float)stats.visits[a];
        min_mean = std::min(min_mean, mean);
        max_mean = std::max(max_mean, mean);
        max_visits = std::max(max_visits, stats.visits[a]);
    }
    float denom = std::max(1.0f, max_mean - min_mean);

    for (int a = 0; a < N_ACTIONS; a++) {
        int y = r.y + 52 + a * 32;
        float mean = stats.visits[a] > 0 ? stats.sum[a] / (float)stats.visits[a]
                                         : min_mean;
        float q = clampf((mean - min_mean) / denom, 0.0f, 1.0f);
        float vq = (float)stats.visits[a] / (float)max_visits;
        cv::Scalar c = action_color(a);
        cv::rectangle(img, cv::Rect(r.x + 44, y - 13, (int)(150.0f * q), 12),
                      c, -1);
        cv::rectangle(img, cv::Rect(r.x + 44, y + 3, (int)(150.0f * vq), 7),
                      cv::Scalar(90, 95, 106), -1);
        cv::putText(img, cv::format("%+.2f", action_w(a)),
                    cv::Point(r.x + 8, y - 3), cv::FONT_HERSHEY_SIMPLEX,
                    0.35, cv::Scalar(190, 195, 202), 1, cv::LINE_AA);
        cv::putText(img, cv::format("%.1f", mean),
                    cv::Point(r.x + 202, y - 3), cv::FONT_HERSHEY_SIMPLEX,
                    0.35, (a == best_action) ? cv::Scalar(255, 255, 255)
                                             : cv::Scalar(170, 175, 182),
                    1, cv::LINE_AA);
    }
}

static cv::Mat draw_frame(const Scene& scene,
                          const Pose& pose,
                          const RootStats& stats,
                          const std::vector<Pose>& trajectory,
                          int frame_idx,
                          int best_action,
                          float gpu_ms,
                          const PlanBench& bench) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 19, 22));
    cv::putText(img, cv::format("GPU MCTS planner  frame %02d / %d",
                                frame_idx + 1, N_FRAMES),
                cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.72,
                cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img,
                cv::format("%d scenes x %d rollouts x %d horizon   GPU %.3f ms/plan   CPU %.3f ms   %.1fx",
                           N_SCENARIOS, ROLLOUTS_PER_SCENE, HORIZON,
                           gpu_ms, bench.cpu_ms, bench.speedup),
                cv::Point(18, 54), cv::FONT_HERSHEY_SIMPLEX, 0.47,
                cv::Scalar(210, 214, 220), 1, cv::LINE_AA);

    cv::Rect map_rect(36, 78, 660, 505);
    cv::rectangle(img, map_rect, cv::Scalar(25, 27, 30), -1);
    cv::rectangle(img, map_rect, cv::Scalar(82, 86, 94), 1);
    for (int i = 1; i < 12; i++) {
        int x = map_rect.x + (int)((float)i / 12.0f * map_rect.width);
        cv::line(img, cv::Point(x, map_rect.y), cv::Point(x, map_rect.y + map_rect.height),
                 cv::Scalar(38, 41, 46), 1);
    }
    for (int i = 1; i < 8; i++) {
        int y = map_rect.y + (int)((float)i / 8.0f * map_rect.height);
        cv::line(img, cv::Point(map_rect.x, y), cv::Point(map_rect.x + map_rect.width, y),
                 cv::Scalar(38, 41, 46), 1);
    }

    for (int i = 0; i < scene.n_obs; i++) {
        cv::Point c = world_to_px(scene.obs[i].x, scene.obs[i].y, map_rect);
        int rr = (int)(scene.obs[i].r / WORLD_W * map_rect.width);
        cv::circle(img, c, rr + 8, cv::Scalar(40, 50, 58), -1, cv::LINE_AA);
        cv::circle(img, c, rr, cv::Scalar(86, 100, 116), -1, cv::LINE_AA);
        cv::circle(img, c, rr, cv::Scalar(140, 156, 170), 1, cv::LINE_AA);
    }

    draw_rollout_fan(img, scene, pose, stats, map_rect, frame_idx);

    std::vector<cv::Point> path_pts;
    for (const Pose& p : trajectory) path_pts.push_back(world_to_px(p.x, p.y, map_rect));
    if (path_pts.size() > 1) {
        cv::polylines(img, path_pts, false, cv::Scalar(245, 245, 245), 3,
                      cv::LINE_AA);
    }

    cv::Point goal = world_to_px(scene.goal_x, scene.goal_y, map_rect);
    cv::circle(img, goal, 12, cv::Scalar(80, 210, 130), 2, cv::LINE_AA);
    cv::circle(img, goal, 4, cv::Scalar(80, 210, 130), -1, cv::LINE_AA);

    cv::Point robot = world_to_px(pose.x, pose.y, map_rect);
    cv::circle(img, robot, 9, cv::Scalar(248, 248, 248), -1, cv::LINE_AA);
    cv::Point nose(robot.x + (int)(18.0f * cosf(pose.yaw)),
                   robot.y - (int)(18.0f * sinf(pose.yaw)));
    cv::line(img, robot, nose, action_color(best_action), 3, cv::LINE_AA);

    cv::Rect stats_rect(720, 88, 218, 300);
    draw_stats_bars(img, stats, stats_rect, best_action);
    float d = dist_to_goal(scene, pose);
    float c = clearance(scene, pose.x, pose.y);
    cv::rectangle(img, cv::Rect(720, 414, 218, 120), cv::Scalar(28, 30, 34), -1);
    cv::rectangle(img, cv::Rect(720, 414, 218, 120), cv::Scalar(78, 82, 90), 1);
    cv::putText(img, cv::format("distance %.2f m", d), cv::Point(734, 444),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(235, 235, 235), 1,
                cv::LINE_AA);
    cv::putText(img, cv::format("clearance %.2f m", c), cv::Point(734, 474),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(210, 214, 220), 1,
                cv::LINE_AA);
    cv::putText(img, cv::format("chosen w %+.2f", action_w(best_action)),
                cv::Point(734, 504), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                action_color(best_action), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<Scene> scenes = make_scenes();
    std::vector<Pose> poses(N_SCENARIOS);
    for (int s = 0; s < N_SCENARIOS; s++) poses[s] = scenes[s].start;

    Scene* d_scenes = nullptr;
    Pose* d_poses = nullptr;
    float* d_sums = nullptr;
    int* d_visits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scenes, N_SCENARIOS * sizeof(Scene)));
    CUDA_CHECK(cudaMalloc(&d_poses, N_SCENARIOS * sizeof(Pose)));
    CUDA_CHECK(cudaMalloc(&d_sums, N_SCENARIOS * N_ACTIONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_visits, N_SCENARIOS * N_ACTIONS * sizeof(int)));

    std::vector<RootStats> gpu_stats;
    std::vector<RootStats> cpu_stats;
    (void)gpu_plan(scenes, poses, d_scenes, d_poses, d_sums, d_visits, gpu_stats);
    CUDA_CHECK(cudaDeviceSynchronize());
    float first_gpu_ms = gpu_plan(scenes, poses, d_scenes, d_poses,
                                  d_sums, d_visits, gpu_stats);
    double cpu_ms = cpu_plan(scenes, poses, cpu_stats);
    PlanBench bench;
    bench.gpu_ms = first_gpu_ms;
    bench.cpu_ms = cpu_ms;
    bench.speedup = cpu_ms / std::max(1.0e-9, (double)first_gpu_ms);

    std::printf("GPU MCTS planner: %d scenes x %d rollouts x %d horizon\n",
                N_SCENARIOS, ROLLOUTS_PER_SCENE, HORIZON);
    std::printf("GPU plan %.3f ms, CPU %.3f ms, speedup %.1fx\n",
                first_gpu_ms, cpu_ms, bench.speedup);
    std::printf("first-scene best action GPU %+0.2f, CPU %+0.2f\n",
                action_w(best_action_from_stats(gpu_stats[0])),
                action_w(best_action_from_stats(cpu_stats[0])));

    int mkdir_ret = std::system("mkdir -p gif");
    (void)mkdir_ret;
    cv::VideoWriter video("gif/gpu_mcts_planner.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_mcts_planner.avi\n");
        return 1;
    }

    std::vector<Pose> trajectory;
    trajectory.push_back(poses[0]);
    double total_gpu_ms = 0.0;
    int reached_frame = -1;

    for (int frame = 0; frame < N_FRAMES; frame++) {
        float gpu_ms = gpu_plan(scenes, poses, d_scenes, d_poses,
                                d_sums, d_visits, gpu_stats);
        total_gpu_ms += gpu_ms;

        std::vector<int> chosen(N_SCENARIOS);
        for (int s = 0; s < N_SCENARIOS; s++) {
            if (dist_to_goal(scenes[s], poses[s]) < GOAL_R) {
                chosen[s] = 3;
                continue;
            }
            chosen[s] = safe_best_action(scenes[s], poses[s], gpu_stats[s]);
            Pose q = integrate(poses[s], chosen[s]);
            if (clearance(scenes[s], q.x, q.y) >= 0.0f) poses[s] = q;
        }

        trajectory.push_back(poses[0]);
        if (reached_frame < 0 && dist_to_goal(scenes[0], poses[0]) < GOAL_R) {
            reached_frame = frame + 1;
        }

        cv::Mat frame_img = draw_frame(scenes[0], poses[0], gpu_stats[0],
                                       trajectory, frame, chosen[0], gpu_ms,
                                       bench);
        video.write(frame_img);

        if (frame % 12 == 0 || frame == N_FRAMES - 1) {
            std::printf("frame %02d  gpu %.3f ms  dist %.3f  action %+0.2f\n",
                        frame + 1, gpu_ms, dist_to_goal(scenes[0], poses[0]),
                        action_w(chosen[0]));
        }
    }

    for (int hold = 0; hold < 16; hold++) {
        int best = best_action_from_stats(gpu_stats[0]);
        cv::Mat frame_img = draw_frame(scenes[0], poses[0], gpu_stats[0],
                                       trajectory, N_FRAMES - 1, best,
                                       (float)(total_gpu_ms / N_FRAMES), bench);
        video.write(frame_img);
    }
    video.release();

    float avg_gpu_ms = (float)(total_gpu_ms / N_FRAMES);
    std::printf("Average GPU MCTS plan %.3f ms/frame\n", avg_gpu_ms);
    std::printf("Scenario 0 final distance %.3f m, reached frame %d\n",
                dist_to_goal(scenes[0], poses[0]), reached_frame);

    cudabot::avi_to_gif("gif/gpu_mcts_planner.avi",
                        "gif/gpu_mcts_planner.gif", 8, 640);
    std::printf("GIF saved to gif/gpu_mcts_planner.gif\n");

    CUDA_CHECK(cudaFree(d_scenes));
    CUDA_CHECK(cudaFree(d_poses));
    CUDA_CHECK(cudaFree(d_sums));
    CUDA_CHECK(cudaFree(d_visits));
    return 0;
}
