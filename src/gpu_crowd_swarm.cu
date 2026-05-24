// gpu_crowd_swarm.cu
//
// 10K-agent GPU crowd / boids simulation.
//
// Each frame builds a uniform-grid neighbour index, then one CUDA thread per
// agent applies separation, alignment, cohesion, group goal, boundary, and
// obstacle-avoidance forces.  The CPU benchmark uses the same grid-capped
// neighbour search for an apples-to-apples update comparison.
//
// Output: gif/gpu_crowd_swarm.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
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

constexpr int N_AGENTS = 10000;
constexpr int N_GROUPS = 5;
constexpr int GRID_X = 120;
constexpr int GRID_Y = 80;
constexpr int N_CELLS = GRID_X * GRID_Y;
constexpr int MAX_PER_CELL = 64;
constexpr float CELL_SIZE = 0.75f;
constexpr float WORLD_W = GRID_X * CELL_SIZE;
constexpr float WORLD_H = GRID_Y * CELL_SIZE;
constexpr float DT = 0.045f;
constexpr float NEIGHBOR_R = 1.75f;
constexpr float SEP_R = 0.58f;
constexpr float MAX_SPEED = 3.2f;
constexpr float MAX_ACCEL = 8.5f;
constexpr int N_FRAMES = 120;
constexpr int N_BENCH = 80;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int VIDEO_FPS = 12;

struct Agent {
    float x;
    float y;
    float vx;
    float vy;
    int group;
};

struct Obstacle {
    float x;
    float y;
    float r;
};

struct SimMetrics {
    float gpu_ms = 0.0f;
    double cpu_ms = 0.0;
    double speedup = 0.0;
    float avg_speed = 0.0f;
    float min_clearance = 0.0f;
    int overflow = 0;
};

constexpr int N_OBS = 6;
static const Obstacle HOST_OBS[N_OBS] = {
    {22.0f, 16.0f, 4.2f},
    {43.0f, 31.0f, 5.0f},
    {68.0f, 44.0f, 4.6f},
    {24.0f, 45.0f, 3.7f},
    {70.0f, 18.0f, 3.8f},
    {50.0f, 12.0f, 2.8f},
};

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline int clampi(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__host__ __device__ static inline float sqr(float x) {
    return x * x;
}

__host__ __device__ static inline void normalize(float& x, float& y) {
    float l = sqrtf(x * x + y * y);
    if (l > 1.0e-6f) {
        x /= l;
        y /= l;
    }
}

__host__ __device__ static inline int cell_index(float x, float y) {
    int cx = clampi((int)(x / CELL_SIZE), 0, GRID_X - 1);
    int cy = clampi((int)(y / CELL_SIZE), 0, GRID_Y - 1);
    return cy * GRID_X + cx;
}

__host__ __device__ static inline void group_goal(int group, float t,
                                                  float& gx, float& gy) {
    float a = 0.72f * t + 6.28318530718f * (float)group / (float)N_GROUPS;
    float orbit = 0.55f + 0.12f * sinf(0.31f * t + (float)group);
    gx = WORLD_W * (0.5f + orbit * 0.42f * cosf(a));
    gy = WORLD_H * (0.5f + orbit * 0.36f * sinf(a));
}

__host__ __device__ static inline void apply_obstacle_force(float x, float y,
                                                            float& ax, float& ay) {
    const Obstacle obs[N_OBS] = {
        {22.0f, 16.0f, 4.2f},
        {43.0f, 31.0f, 5.0f},
        {68.0f, 44.0f, 4.6f},
        {24.0f, 45.0f, 3.7f},
        {70.0f, 18.0f, 3.8f},
        {50.0f, 12.0f, 2.8f},
    };
    for (int i = 0; i < N_OBS; i++) {
        float dx = x - obs[i].x;
        float dy = y - obs[i].y;
        float d = sqrtf(dx * dx + dy * dy) + 1.0e-5f;
        float margin = obs[i].r + 2.2f;
        if (d < margin) {
            float w = (margin - d) / margin;
            ax += 9.5f * w * w * dx / d;
            ay += 9.5f * w * w * dy / d;
        }
    }
}

__host__ __device__ static inline void resolve_obstacles(Agent& a) {
    const Obstacle obs[N_OBS] = {
        {22.0f, 16.0f, 4.2f},
        {43.0f, 31.0f, 5.0f},
        {68.0f, 44.0f, 4.6f},
        {24.0f, 45.0f, 3.7f},
        {70.0f, 18.0f, 3.8f},
        {50.0f, 12.0f, 2.8f},
    };
    for (int i = 0; i < N_OBS; i++) {
        float dx = a.x - obs[i].x;
        float dy = a.y - obs[i].y;
        float d = sqrtf(dx * dx + dy * dy);
        float safe = obs[i].r + 0.18f;
        if (d < safe) {
            float nx = (d > 1.0e-5f) ? dx / d : 1.0f;
            float ny = (d > 1.0e-5f) ? dy / d : 0.0f;
            a.x = obs[i].x + nx * safe;
            a.y = obs[i].y + ny * safe;
            float vn = a.vx * nx + a.vy * ny;
            if (vn < 0.0f) {
                a.vx -= 1.35f * vn * nx;
                a.vy -= 1.35f * vn * ny;
            }
        }
    }
}

__global__ void clear_grid_kernel(int* counts, int* overflow) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_CELLS) counts[idx] = 0;
    if (idx == 0) *overflow = 0;
}

__global__ void bin_agents_kernel(const Agent* agents, int* counts,
                                  int* indices, int* overflow) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_AGENTS) return;
    int cell = cell_index(agents[i].x, agents[i].y);
    int slot = atomicAdd(counts + cell, 1);
    if (slot < MAX_PER_CELL) {
        indices[cell * MAX_PER_CELL + slot] = i;
    } else {
        atomicAdd(overflow, 1);
    }
}

__global__ void update_agents_kernel(const Agent* in,
                                     Agent* out,
                                     const int* counts,
                                     const int* indices,
                                     float t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_AGENTS) return;
    Agent a = in[i];
    int cx = clampi((int)(a.x / CELL_SIZE), 0, GRID_X - 1);
    int cy = clampi((int)(a.y / CELL_SIZE), 0, GRID_Y - 1);

    float sep_x = 0.0f;
    float sep_y = 0.0f;
    float align_x = 0.0f;
    float align_y = 0.0f;
    float coh_x = 0.0f;
    float coh_y = 0.0f;
    int count = 0;
    const float neigh2 = NEIGHBOR_R * NEIGHBOR_R;
    const float sep2 = SEP_R * SEP_R;

    for (int dy = -3; dy <= 3; dy++) {
        int yy = cy + dy;
        if (yy < 0 || yy >= GRID_Y) continue;
        for (int dx = -3; dx <= 3; dx++) {
            int xx = cx + dx;
            if (xx < 0 || xx >= GRID_X) continue;
            int cell = yy * GRID_X + xx;
            int n = min(counts[cell], MAX_PER_CELL);
            for (int k = 0; k < n; k++) {
                int j = indices[cell * MAX_PER_CELL + k];
                if (j == i) continue;
                Agent b = in[j];
                float rx = b.x - a.x;
                float ry = b.y - a.y;
                float d2 = rx * rx + ry * ry;
                if (d2 > neigh2 || d2 < 1.0e-8f) continue;
                float inv = rsqrtf(d2 + 1.0e-6f);
                float w = 1.0f - d2 / neigh2;
                if (d2 < sep2) {
                    float sw = 1.0f - d2 / sep2;
                    sep_x -= sw * sw * rx * inv;
                    sep_y -= sw * sw * ry * inv;
                }
                align_x += w * b.vx;
                align_y += w * b.vy;
                coh_x += w * b.x;
                coh_y += w * b.y;
                count++;
            }
        }
    }

    float ax = 0.0f;
    float ay = 0.0f;
    if (count > 0) {
        float inv_n = 1.0f / (float)count;
        align_x *= inv_n;
        align_y *= inv_n;
        normalize(align_x, align_y);
        float vx_n = a.vx;
        float vy_n = a.vy;
        normalize(vx_n, vy_n);
        ax += 1.25f * (align_x - vx_n);
        ay += 1.25f * (align_y - vy_n);

        coh_x = coh_x * inv_n - a.x;
        coh_y = coh_y * inv_n - a.y;
        normalize(coh_x, coh_y);
        ax += 0.55f * coh_x;
        ay += 0.55f * coh_y;
    }
    ax += 3.2f * sep_x;
    ay += 3.2f * sep_y;

    float gx, gy;
    group_goal(a.group, t, gx, gy);
    float goal_x = gx - a.x;
    float goal_y = gy - a.y;
    normalize(goal_x, goal_y);
    ax += 1.45f * goal_x;
    ay += 1.45f * goal_y;

    float center_x = a.x - WORLD_W * 0.5f;
    float center_y = a.y - WORLD_H * 0.5f;
    float tangent_x = -center_y;
    float tangent_y = center_x;
    normalize(tangent_x, tangent_y);
    ax += 0.42f * tangent_x;
    ay += 0.42f * tangent_y;

    apply_obstacle_force(a.x, a.y, ax, ay);
    if (a.x < 3.0f) ax += 4.8f * (3.0f - a.x);
    if (a.x > WORLD_W - 3.0f) ax -= 4.8f * (a.x - (WORLD_W - 3.0f));
    if (a.y < 3.0f) ay += 4.8f * (3.0f - a.y);
    if (a.y > WORLD_H - 3.0f) ay -= 4.8f * (a.y - (WORLD_H - 3.0f));

    float al = sqrtf(ax * ax + ay * ay);
    if (al > MAX_ACCEL) {
        ax *= MAX_ACCEL / al;
        ay *= MAX_ACCEL / al;
    }

    a.vx += ax * DT;
    a.vy += ay * DT;
    float sp = sqrtf(a.vx * a.vx + a.vy * a.vy);
    if (sp > MAX_SPEED) {
        a.vx *= MAX_SPEED / sp;
        a.vy *= MAX_SPEED / sp;
    }
    if (sp < 0.45f) {
        float nudge = 0.10f + 0.02f * (float)(a.group + 1);
        a.vx += nudge * cosf(t + 0.017f * i);
        a.vy += nudge * sinf(0.7f * t + 0.013f * i);
    }

    a.x += a.vx * DT;
    a.y += a.vy * DT;
    if (a.x < 0.5f || a.x > WORLD_W - 0.5f) {
        a.vx = -0.72f * a.vx;
        a.x = clampf(a.x, 0.5f, WORLD_W - 0.5f);
    }
    if (a.y < 0.5f || a.y > WORLD_H - 0.5f) {
        a.vy = -0.72f * a.vy;
        a.y = clampf(a.y, 0.5f, WORLD_H - 0.5f);
    }
    resolve_obstacles(a);
    out[i] = a;
}

static cv::Scalar group_color(int group) {
    static const cv::Scalar colors[N_GROUPS] = {
        cv::Scalar(88, 165, 255),
        cv::Scalar(82, 225, 138),
        cv::Scalar(255, 188, 75),
        cv::Scalar(235, 106, 126),
        cv::Scalar(178, 136, 255),
    };
    return colors[group % N_GROUPS];
}

static std::vector<Agent> make_agents() {
    std::vector<Agent> agents(N_AGENTS);
    std::mt19937 rng(24052026);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    for (int i = 0; i < N_AGENTS; i++) {
        int g = i % N_GROUPS;
        float a = 6.28318530718f * (float)g / (float)N_GROUPS;
        float ring = 0.25f + 0.62f * uni(rng);
        float jitter = 3.2f * normal(rng);
        float cx = WORLD_W * (0.5f + 0.34f * std::cos(a));
        float cy = WORLD_H * (0.5f + 0.30f * std::sin(a));
        float theta = a + 2.2f * (uni(rng) - 0.5f);
        float x = clampf(cx + ring * 9.5f * std::cos(theta) + jitter, 1.0f, WORLD_W - 1.0f);
        float y = clampf(cy + ring * 7.0f * std::sin(theta) + 0.7f * jitter, 1.0f, WORLD_H - 1.0f);
        float tvx = -std::sin(theta);
        float tvy = std::cos(theta);
        float speed = 1.25f + 0.55f * uni(rng);
        agents[i] = {x, y, speed * tvx, speed * tvy, g};
        resolve_obstacles(agents[i]);
    }
    return agents;
}

static void cpu_step(const std::vector<Agent>& in,
                     std::vector<Agent>& out,
                     std::vector<int>& counts,
                     std::vector<int>& indices,
                     int& overflow,
                     float t) {
    counts.assign(N_CELLS, 0);
    std::fill(indices.begin(), indices.end(), -1);
    overflow = 0;
    for (int i = 0; i < N_AGENTS; i++) {
        int cell = cell_index(in[i].x, in[i].y);
        int slot = counts[cell]++;
        if (slot < MAX_PER_CELL) {
            indices[cell * MAX_PER_CELL + slot] = i;
        } else {
            overflow++;
        }
    }
    out.resize(N_AGENTS);
    const float neigh2 = NEIGHBOR_R * NEIGHBOR_R;
    const float sep2 = SEP_R * SEP_R;

    for (int i = 0; i < N_AGENTS; i++) {
        Agent a = in[i];
        int cx = clampi((int)(a.x / CELL_SIZE), 0, GRID_X - 1);
        int cy = clampi((int)(a.y / CELL_SIZE), 0, GRID_Y - 1);
        float sep_x = 0.0f;
        float sep_y = 0.0f;
        float align_x = 0.0f;
        float align_y = 0.0f;
        float coh_x = 0.0f;
        float coh_y = 0.0f;
        int count = 0;
        for (int dy = -3; dy <= 3; dy++) {
            int yy = cy + dy;
            if (yy < 0 || yy >= GRID_Y) continue;
            for (int dx = -3; dx <= 3; dx++) {
                int xx = cx + dx;
                if (xx < 0 || xx >= GRID_X) continue;
                int cell = yy * GRID_X + xx;
                int n = std::min(counts[cell], MAX_PER_CELL);
                for (int k = 0; k < n; k++) {
                    int j = indices[cell * MAX_PER_CELL + k];
                    if (j == i || j < 0) continue;
                    Agent b = in[j];
                    float rx = b.x - a.x;
                    float ry = b.y - a.y;
                    float d2 = rx * rx + ry * ry;
                    if (d2 > neigh2 || d2 < 1.0e-8f) continue;
                    float inv = 1.0f / std::sqrt(d2 + 1.0e-6f);
                    float w = 1.0f - d2 / neigh2;
                    if (d2 < sep2) {
                        float sw = 1.0f - d2 / sep2;
                        sep_x -= sw * sw * rx * inv;
                        sep_y -= sw * sw * ry * inv;
                    }
                    align_x += w * b.vx;
                    align_y += w * b.vy;
                    coh_x += w * b.x;
                    coh_y += w * b.y;
                    count++;
                }
            }
        }
        float ax = 0.0f;
        float ay = 0.0f;
        if (count > 0) {
            float inv_n = 1.0f / (float)count;
            align_x *= inv_n;
            align_y *= inv_n;
            normalize(align_x, align_y);
            float vx_n = a.vx;
            float vy_n = a.vy;
            normalize(vx_n, vy_n);
            ax += 1.25f * (align_x - vx_n);
            ay += 1.25f * (align_y - vy_n);
            coh_x = coh_x * inv_n - a.x;
            coh_y = coh_y * inv_n - a.y;
            normalize(coh_x, coh_y);
            ax += 0.55f * coh_x;
            ay += 0.55f * coh_y;
        }
        ax += 3.2f * sep_x;
        ay += 3.2f * sep_y;
        float gx, gy;
        group_goal(a.group, t, gx, gy);
        float goal_x = gx - a.x;
        float goal_y = gy - a.y;
        normalize(goal_x, goal_y);
        ax += 1.45f * goal_x;
        ay += 1.45f * goal_y;
        float center_x = a.x - WORLD_W * 0.5f;
        float center_y = a.y - WORLD_H * 0.5f;
        float tangent_x = -center_y;
        float tangent_y = center_x;
        normalize(tangent_x, tangent_y);
        ax += 0.42f * tangent_x;
        ay += 0.42f * tangent_y;
        apply_obstacle_force(a.x, a.y, ax, ay);
        if (a.x < 3.0f) ax += 4.8f * (3.0f - a.x);
        if (a.x > WORLD_W - 3.0f) ax -= 4.8f * (a.x - (WORLD_W - 3.0f));
        if (a.y < 3.0f) ay += 4.8f * (3.0f - a.y);
        if (a.y > WORLD_H - 3.0f) ay -= 4.8f * (a.y - (WORLD_H - 3.0f));
        float al = std::sqrt(ax * ax + ay * ay);
        if (al > MAX_ACCEL) {
            ax *= MAX_ACCEL / al;
            ay *= MAX_ACCEL / al;
        }
        a.vx += ax * DT;
        a.vy += ay * DT;
        float sp = std::sqrt(a.vx * a.vx + a.vy * a.vy);
        if (sp > MAX_SPEED) {
            a.vx *= MAX_SPEED / sp;
            a.vy *= MAX_SPEED / sp;
        }
        if (sp < 0.45f) {
            float nudge = 0.10f + 0.02f * (float)(a.group + 1);
            a.vx += nudge * std::cos(t + 0.017f * i);
            a.vy += nudge * std::sin(0.7f * t + 0.013f * i);
        }
        a.x += a.vx * DT;
        a.y += a.vy * DT;
        if (a.x < 0.5f || a.x > WORLD_W - 0.5f) {
            a.vx = -0.72f * a.vx;
            a.x = clampf(a.x, 0.5f, WORLD_W - 0.5f);
        }
        if (a.y < 0.5f || a.y > WORLD_H - 0.5f) {
            a.vy = -0.72f * a.vy;
            a.y = clampf(a.y, 0.5f, WORLD_H - 0.5f);
        }
        resolve_obstacles(a);
        out[i] = a;
    }
}

static float gpu_step(Agent* d_in,
                      Agent* d_out,
                      int* d_counts,
                      int* d_indices,
                      int* d_overflow,
                      float t) {
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    clear_grid_kernel<<<(N_CELLS + 255) / 256, 256>>>(d_counts, d_overflow);
    bin_agents_kernel<<<(N_AGENTS + 255) / 256, 256>>>(d_in, d_counts,
                                                       d_indices, d_overflow);
    update_agents_kernel<<<(N_AGENTS + 255) / 256, 256>>>(d_in, d_out,
                                                         d_counts, d_indices, t);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return ms;
}

static SimMetrics benchmark(const std::vector<Agent>& initial,
                            Agent* d_a,
                            Agent* d_b,
                            int* d_counts,
                            int* d_indices,
                            int* d_overflow) {
    SimMetrics m;
    CUDA_CHECK(cudaMemcpy(d_a, initial.data(), initial.size() * sizeof(Agent),
                          cudaMemcpyHostToDevice));
    (void)gpu_step(d_a, d_b, d_counts, d_indices, d_overflow, 0.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_a, initial.data(), initial.size() * sizeof(Agent),
                          cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    Agent* src = d_a;
    Agent* dst = d_b;
    for (int i = 0; i < N_BENCH; i++) {
        clear_grid_kernel<<<(N_CELLS + 255) / 256, 256>>>(d_counts, d_overflow);
        bin_agents_kernel<<<(N_AGENTS + 255) / 256, 256>>>(src, d_counts,
                                                           d_indices, d_overflow);
        update_agents_kernel<<<(N_AGENTS + 255) / 256, 256>>>(src, dst,
                                                              d_counts, d_indices,
                                                              0.04f * i);
        std::swap(src, dst);
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float total_gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_gpu_ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    m.gpu_ms = total_gpu_ms / N_BENCH;

    std::vector<Agent> cpu_a = initial;
    std::vector<Agent> cpu_b;
    std::vector<int> counts(N_CELLS);
    std::vector<int> indices(N_CELLS * MAX_PER_CELL);
    int overflow = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 6; i++) {
        cpu_step(cpu_a, cpu_b, counts, indices, overflow, 0.04f * i);
        cpu_a.swap(cpu_b);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    m.cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 6.0;
    m.speedup = m.cpu_ms / std::max(1.0e-9, (double)m.gpu_ms);
    return m;
}

static SimMetrics summarize(const std::vector<Agent>& agents,
                            float gpu_ms,
                            double cpu_ms,
                            double speedup,
                            int overflow) {
    SimMetrics m;
    m.gpu_ms = gpu_ms;
    m.cpu_ms = cpu_ms;
    m.speedup = speedup;
    double speed_sum = 0.0;
    float min_clear = 1.0e9f;
    for (const Agent& a : agents) {
        speed_sum += std::sqrt(a.vx * a.vx + a.vy * a.vy);
        min_clear = std::min(min_clear, std::min(std::min(a.x, WORLD_W - a.x),
                                                 std::min(a.y, WORLD_H - a.y)));
        for (int i = 0; i < N_OBS; i++) {
            float d = std::sqrt(sqr(a.x - HOST_OBS[i].x) + sqr(a.y - HOST_OBS[i].y))
                    - HOST_OBS[i].r;
            min_clear = std::min(min_clear, d);
        }
    }
    m.avg_speed = (float)(speed_sum / agents.size());
    m.min_clearance = min_clear;
    m.overflow = overflow;
    return m;
}

static cv::Point world_to_px(float x, float y, const cv::Rect& r) {
    int px = r.x + (int)(clampf(x / WORLD_W, 0.0f, 1.0f) * r.width);
    int py = r.y + r.height - (int)(clampf(y / WORLD_H, 0.0f, 1.0f) * r.height);
    return cv::Point(px, py);
}

static void splat(cv::Mat& img, int x, int y, const cv::Scalar& c) {
    if ((unsigned)x >= (unsigned)img.cols || (unsigned)y >= (unsigned)img.rows) return;
    cv::Vec3b& p = img.at<cv::Vec3b>(y, x);
    p[0] = (unsigned char)std::min(255.0, 0.45 * p[0] + 0.75 * c[0]);
    p[1] = (unsigned char)std::min(255.0, 0.45 * p[1] + 0.75 * c[1]);
    p[2] = (unsigned char)std::min(255.0, 0.45 * p[2] + 0.75 * c[2]);
}

static void draw_history(cv::Mat& img,
                         const std::vector<float>& speed_hist,
                         const std::vector<float>& clear_hist,
                         const cv::Rect& r) {
    cv::rectangle(img, r, cv::Scalar(29, 31, 36), -1);
    cv::rectangle(img, r, cv::Scalar(76, 80, 88), 1);
    cv::putText(img, "swarm health", cv::Point(r.x + 12, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(235, 235, 235),
                1, cv::LINE_AA);

    auto draw_curve = [&](const std::vector<float>& h, float ymax, cv::Scalar color) {
        if (h.size() < 2) return;
        std::vector<cv::Point> pts;
        for (size_t i = 0; i < h.size(); i++) {
            float x01 = (float)i / std::max<size_t>(1, h.size() - 1);
            float y01 = clampf(h[i] / ymax, 0.0f, 1.0f);
            int x = r.x + 34 + (int)(x01 * (r.width - 46));
            int y = r.y + r.height - 18 - (int)(y01 * (r.height - 50));
            pts.emplace_back(x, y);
        }
        cv::polylines(img, pts, false, color, 2, cv::LINE_AA);
    };
    for (int g = 0; g <= 4; g++) {
        int y = r.y + r.height - 18 - g * (r.height - 50) / 4;
        cv::line(img, cv::Point(r.x + 34, y), cv::Point(r.x + r.width - 12, y),
                 cv::Scalar(45, 48, 54), 1);
    }
    draw_curve(speed_hist, MAX_SPEED, cv::Scalar(90, 170, 255));
    draw_curve(clear_hist, 4.0f, cv::Scalar(95, 230, 135));
    cv::putText(img, "speed", cv::Point(r.x + 118, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(90, 170, 255), 1);
    cv::putText(img, "clear", cv::Point(r.x + 178, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(95, 230, 135), 1);
}

static cv::Mat draw_frame(const std::vector<Agent>& agents,
                          int frame,
                          const SimMetrics& metrics,
                          const std::vector<float>& speed_hist,
                          const std::vector<float>& clear_hist) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(17, 19, 23));
    cv::putText(img, cv::format("GPU crowd swarm  frame %03d / %d",
                                frame + 1, N_FRAMES),
                cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.72,
                cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img,
                cv::format("%d agents   uniform grid %dx%d   GPU %.3f ms/step   CPU %.3f ms   %.1fx",
                           N_AGENTS, GRID_X, GRID_Y, metrics.gpu_ms,
                           metrics.cpu_ms, metrics.speedup),
                cv::Point(18, 54), cv::FONT_HERSHEY_SIMPLEX, 0.47,
                cv::Scalar(210, 214, 220), 1, cv::LINE_AA);

    cv::Rect map_rect(30, 78, 690, 500);
    cv::rectangle(img, map_rect, cv::Scalar(23, 25, 29), -1);
    cv::rectangle(img, map_rect, cv::Scalar(78, 82, 90), 1);
    for (int i = 1; i < 12; i++) {
        int x = map_rect.x + i * map_rect.width / 12;
        cv::line(img, cv::Point(x, map_rect.y), cv::Point(x, map_rect.y + map_rect.height),
                 cv::Scalar(36, 39, 45), 1);
    }
    for (int i = 1; i < 8; i++) {
        int y = map_rect.y + i * map_rect.height / 8;
        cv::line(img, cv::Point(map_rect.x, y), cv::Point(map_rect.x + map_rect.width, y),
                 cv::Scalar(36, 39, 45), 1);
    }

    for (int i = 0; i < N_OBS; i++) {
        cv::Point c = world_to_px(HOST_OBS[i].x, HOST_OBS[i].y, map_rect);
        int rr = (int)(HOST_OBS[i].r / WORLD_W * map_rect.width);
        cv::circle(img, c, rr + 7, cv::Scalar(37, 46, 55), -1, cv::LINE_AA);
        cv::circle(img, c, rr, cv::Scalar(82, 96, 112), -1, cv::LINE_AA);
        cv::circle(img, c, rr, cv::Scalar(138, 154, 170), 1, cv::LINE_AA);
    }

    for (int g = 0; g < N_GROUPS; g++) {
        float gx, gy;
        group_goal(g, 0.045f * frame, gx, gy);
        cv::Point p = world_to_px(gx, gy, map_rect);
        cv::drawMarker(img, p, group_color(g), cv::MARKER_TILTED_CROSS, 13, 2,
                       cv::LINE_AA);
    }

    for (const Agent& a : agents) {
        cv::Point p = world_to_px(a.x, a.y, map_rect);
        cv::Scalar c = group_color(a.group);
        splat(img, p.x, p.y, c);
        splat(img, p.x + 1, p.y, c * 0.75);
        splat(img, p.x, p.y + 1, c * 0.75);
    }

    cv::Rect stat_rect(742, 88, 196, 158);
    cv::rectangle(img, stat_rect, cv::Scalar(29, 31, 36), -1);
    cv::rectangle(img, stat_rect, cv::Scalar(76, 80, 88), 1);
    cv::putText(img, cv::format("avg speed %.2f", metrics.avg_speed),
                cv::Point(756, 122), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::putText(img, cv::format("min clear %.2f", metrics.min_clearance),
                cv::Point(756, 154), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                metrics.min_clearance > 0.0f ? cv::Scalar(92, 230, 132)
                                             : cv::Scalar(100, 120, 255),
                1, cv::LINE_AA);
    cv::putText(img, cv::format("cell overflow %d", metrics.overflow),
                cv::Point(756, 186), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(210, 214, 220), 1, cv::LINE_AA);
    cv::putText(img, cv::format("groups %d", N_GROUPS),
                cv::Point(756, 218), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(210, 214, 220), 1, cv::LINE_AA);

    draw_history(img, speed_hist, clear_hist, cv::Rect(742, 276, 196, 198));
    cv::putText(img, "boids: separation", cv::Point(742, 518),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(185, 190, 198), 1);
    cv::putText(img, "alignment, cohesion", cv::Point(742, 542),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(185, 190, 198), 1);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<Agent> agents = make_agents();

    Agent* d_a = nullptr;
    Agent* d_b = nullptr;
    int* d_counts = nullptr;
    int* d_indices = nullptr;
    int* d_overflow = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N_AGENTS * sizeof(Agent)));
    CUDA_CHECK(cudaMalloc(&d_b, N_AGENTS * sizeof(Agent)));
    CUDA_CHECK(cudaMalloc(&d_counts, N_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_indices, N_CELLS * MAX_PER_CELL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_overflow, sizeof(int)));

    SimMetrics bench = benchmark(agents, d_a, d_b, d_counts, d_indices, d_overflow);
    std::printf("GPU crowd swarm: %d agents, grid %dx%d, cap %d/cell\n",
                N_AGENTS, GRID_X, GRID_Y, MAX_PER_CELL);
    std::printf("GPU step %.3f ms, CPU %.3f ms, speedup %.1fx\n",
                bench.gpu_ms, bench.cpu_ms, bench.speedup);

    CUDA_CHECK(cudaMemcpy(d_a, agents.data(), agents.size() * sizeof(Agent),
                          cudaMemcpyHostToDevice));
    int mkdir_ret = std::system("mkdir -p gif");
    (void)mkdir_ret;
    cv::VideoWriter video("gif/gpu_crowd_swarm.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_crowd_swarm.avi\n");
        return 1;
    }

    std::vector<float> speed_hist;
    std::vector<float> clear_hist;
    double total_gpu_ms = 0.0;
    double total_clear = 0.0;
    int max_overflow = 0;
    Agent* src = d_a;
    Agent* dst = d_b;

    for (int frame = 0; frame < N_FRAMES; frame++) {
        float t = 0.045f * frame;
        float gpu_ms = gpu_step(src, dst, d_counts, d_indices, d_overflow, t);
        std::swap(src, dst);
        CUDA_CHECK(cudaMemcpy(agents.data(), src, agents.size() * sizeof(Agent),
                              cudaMemcpyDeviceToHost));
        int overflow = 0;
        CUDA_CHECK(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
        SimMetrics metrics = summarize(agents, gpu_ms, bench.cpu_ms, bench.speedup, overflow);
        total_gpu_ms += gpu_ms;
        total_clear += metrics.min_clearance;
        max_overflow = std::max(max_overflow, overflow);
        speed_hist.push_back(metrics.avg_speed);
        clear_hist.push_back(std::max(0.0f, metrics.min_clearance));
        if (speed_hist.size() > 74) {
            speed_hist.erase(speed_hist.begin());
            clear_hist.erase(clear_hist.begin());
        }

        cv::Mat frame_img = draw_frame(agents, frame, metrics, speed_hist, clear_hist);
        video.write(frame_img);

        if (frame % 15 == 0 || frame == N_FRAMES - 1) {
            std::printf("frame %03d  gpu %.3f ms  speed %.2f  min_clear %.2f  overflow %d\n",
                        frame + 1, gpu_ms, metrics.avg_speed,
                        metrics.min_clearance, overflow);
        }
    }

    SimMetrics final_metrics = summarize(agents, (float)(total_gpu_ms / N_FRAMES),
                                         bench.cpu_ms, bench.speedup, max_overflow);
    for (int hold = 0; hold < 14; hold++) {
        cv::Mat frame_img = draw_frame(agents, N_FRAMES - 1, final_metrics,
                                       speed_hist, clear_hist);
        video.write(frame_img);
    }
    video.release();

    std::printf("Average GPU step %.3f ms/frame\n", total_gpu_ms / N_FRAMES);
    std::printf("Average min clearance %.3f m, max cell overflow %d\n",
                total_clear / N_FRAMES, max_overflow);

    cudabot::avi_to_gif("gif/gpu_crowd_swarm.avi",
                        "gif/gpu_crowd_swarm.gif", 8, 640);
    std::printf("GIF saved to gif/gpu_crowd_swarm.gif\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_overflow));
    return 0;
}
