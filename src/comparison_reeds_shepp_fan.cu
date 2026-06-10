/*************************************************************************
    Massive Reeds-Shepp Fan: CPU 1,024 vs GPU 1,048,572 RS candidate paths.
    Each thread samples a (primitive_id, l1, l2, l3) triple, simulates the
    3-segment motion from the vehicle's start pose, performs a collision
    check against a static parking-lot occupancy grid, and reports the
    resulting end pose and Reeds-Shepp arc length. A reduction picks the
    minimum-cost collision-free candidate that lands closest to the
    target parking pose. On the GPU side the surviving end-pose cloud
    looks like a dense "lotus" fan around the vehicle; on the CPU side
    only sparse spokes are visible.
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <limits>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include "cuda_check.cuh"

    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int GRID_W = 320;
constexpr int GRID_H = 240;
constexpr float GRID_RES = 0.20f;  // -> 64 m x 48 m parking lot
constexpr float WORLD_W = GRID_W * GRID_RES;
constexpr float WORLD_H = GRID_H * GRID_RES;

constexpr float TURN_R = 3.5f;          // vehicle minimum turning radius
constexpr float MAX_TURN_ANGLE = 2.6f;  // ~150 deg per turn segment
constexpr float MAX_STRAIGHT = 9.0f;    // meters per straight segment

constexpr int N_CAND_PER_PRIM_GPU = 174762;            // ~6 * 174762 ≈ 1.05M
constexpr int N_PRIMITIVES = 6;
constexpr int N_CAND_GPU = N_CAND_PER_PRIM_GPU * N_PRIMITIVES;
constexpr int N_CAND_CPU = 1024;  // matching ray-cast demo

constexpr int PANEL_W = 700;
constexpr int PANEL_H = 525;
constexpr float VIS_SCALE_X = static_cast<float>(PANEL_W) / WORLD_W;
constexpr float VIS_SCALE_Y = static_cast<float>(PANEL_H) / WORLD_H;

constexpr int SIM_FRAMES = 60;

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------
static void build_parking_lot(std::vector<unsigned char>& grid) {
    grid.assign(GRID_W * GRID_H, 0u);
    auto fill_rect = [&](int x0, int y0, int x1, int y1) {
        for (int gy = std::max(0, y0); gy <= std::min(GRID_H - 1, y1); gy++)
            for (int gx = std::max(0, x0); gx <= std::min(GRID_W - 1, x1); gx++)
                grid[gy * GRID_W + gx] = 1u;
    };
    // Outer perimeter
    fill_rect(0, 0, GRID_W - 1, 1);
    fill_rect(0, GRID_H - 2, GRID_W - 1, GRID_H - 1);
    fill_rect(0, 0, 1, GRID_H - 1);
    fill_rect(GRID_W - 2, 0, GRID_W - 1, GRID_H - 1);

    // Parked cars at the top row (size ~ 4 m x 2 m at 0.2 m/cell -> 20 x 10)
    int top_y0 = 200, top_y1 = 218;
    for (int k = 0; k < 6; k++) {
        int x0 = 30 + k * 50;
        fill_rect(x0, top_y0, x0 + 22, top_y1);
    }
    // Parked cars at the bottom row
    int bot_y0 = 22, bot_y1 = 40;
    for (int k = 0; k < 6; k++) {
        int x0 = 30 + k * 50;
        if (k == 3) continue;  // gap = target slot
        fill_rect(x0, bot_y0, x0 + 22, bot_y1);
    }
}

// ---------------------------------------------------------------------------
// Reeds-Shepp 3-segment kinematics
// ---------------------------------------------------------------------------
struct Pose { float x, y, th; };

__host__ __device__ inline float wrap_pi(float a) {
    while (a >  static_cast<float>(M_PI)) a -= 2.0f * static_cast<float>(M_PI);
    while (a < -static_cast<float>(M_PI)) a += 2.0f * static_cast<float>(M_PI);
    return a;
}

// motion: 0 = L (left turn), 1 = R (right turn), 2 = S (straight)
// len: positive arc length for L/R (angle = len / R); meters for S.
__host__ __device__ inline Pose apply_segment(Pose p, int motion, float len, float R) {
    if (motion == 2) {
        p.x += len * cosf(p.th);
        p.y += len * sinf(p.th);
        return p;
    }
    float sign = (motion == 0) ? +1.0f : -1.0f;
    float dth  = sign * (len / R);
    float cx = p.x - sign * R * sinf(p.th);
    float cy = p.y + sign * R * cosf(p.th);
    float dx = p.x - cx, dy = p.y - cy;
    float c = cosf(dth), s = sinf(dth);
    p.x  = cx + c * dx - s * dy;
    p.y  = cy + s * dx + c * dy;
    p.th = wrap_pi(p.th + dth);
    return p;
}

__device__ inline bool segment_collides(Pose start, int motion, float len, float R,
                                        const unsigned char* grid,
                                        int gridW, int gridH, float gridRes) {
    int n_check = (motion == 2) ? max(2, (int)(len / 0.8f) + 1)
                                : max(2, (int)(len / 0.4f) + 1);
    for (int s = 1; s <= n_check; s++) {
        float u = static_cast<float>(s) / n_check;
        Pose p = apply_segment(start, motion, len * u, R);
        int gx = (int)floorf(p.x / gridRes);
        int gy = (int)floorf(p.y / gridRes);
        if (gx < 0 || gx >= gridW || gy < 0 || gy >= gridH) return true;
        if (grid[gy * gridW + gx]) return true;
    }
    return false;
}

// Map candidate index to (prim_id, t1, t2, t3) and run the RS kinematics.
__device__ inline void candidate_motions(int prim_id, int& m1, int& m2, int& m3) {
    switch (prim_id) {
        case 0: m1=0; m2=2; m3=0; break;  // LSL
        case 1: m1=1; m2=2; m3=1; break;  // RSR
        case 2: m1=0; m2=2; m3=1; break;  // LSR
        case 3: m1=1; m2=2; m3=0; break;  // RSL
        case 4: m1=0; m2=1; m3=0; break;  // LRL
        default:m1=1; m2=0; m3=1; break;  // RLR
    }
}

__global__ void rs_fan_kernel(Pose start, Pose target, float R,
                              const unsigned char* __restrict__ grid,
                              int gridW, int gridH, float gridRes,
                              float max_turn, float max_straight,
                              int n_per_prim, int n_primitives,
                              float* __restrict__ end_x,
                              float* __restrict__ end_y,
                              unsigned char* __restrict__ collide,
                              float* __restrict__ cost,
                              float* __restrict__ length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_per_prim * n_primitives;
    if (i >= total) return;
    int prim = i % n_primitives;
    int flat = i / n_primitives;
    // 56^3 = 175616 ≈ n_per_prim
    int nx = 56;
    int t1_idx = flat % nx;
    int t2_idx = (flat / nx) % nx;
    int t3_idx = (flat / (nx * nx)) % nx;

    int m1, m2, m3;
    candidate_motions(prim, m1, m2, m3);

    float l1 = (m1 == 2 ? max_straight : R * max_turn) * (t1_idx + 0.5f) / nx;
    float l2 = (m2 == 2 ? max_straight : R * max_turn) * (t2_idx + 0.5f) / nx;
    float l3 = (m3 == 2 ? max_straight : R * max_turn) * (t3_idx + 0.5f) / nx;

    Pose p0 = start;
    bool coll = segment_collides(p0, m1, l1, R, grid, gridW, gridH, gridRes);
    Pose p1 = apply_segment(p0, m1, l1, R);
    if (!coll) coll = segment_collides(p1, m2, l2, R, grid, gridW, gridH, gridRes);
    Pose p2 = apply_segment(p1, m2, l2, R);
    if (!coll) coll = segment_collides(p2, m3, l3, R, grid, gridW, gridH, gridRes);
    Pose p3 = apply_segment(p2, m3, l3, R);

    float len = l1 + l2 + l3;
    end_x[i] = p3.x;
    end_y[i] = p3.y;
    collide[i] = coll ? 1u : 0u;
    length[i] = len;

    float dx = p3.x - target.x;
    float dy = p3.y - target.y;
    float dth = wrap_pi(p3.th - target.th);
    float cost_val = sqrtf(dx * dx + dy * dy) + 1.5f * fabsf(dth) + 0.05f * len;
    if (coll) cost_val = 1.0e9f;
    cost[i] = cost_val;
}

// Per-block reduction to find argmin cost.
__global__ void argmin_reduce_kernel(const float* cost, int n,
                                     float* block_min_val, int* block_min_idx) {
    extern __shared__ unsigned char smem[];
    float* sv = reinterpret_cast<float*>(smem);
    int*   si = reinterpret_cast<int*>(smem + blockDim.x * sizeof(float));
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float v = (gid < n) ? cost[gid] : 1.0e30f;
    int   k = gid;
    sv[tid] = v;
    si[tid] = k;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sv[tid + s] < sv[tid]) {
                sv[tid] = sv[tid + s];
                si[tid] = si[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_min_val[blockIdx.x] = sv[0];
        block_min_idx[blockIdx.x] = si[0];
    }
}

// CPU fan: small N, sequential
static void cpu_fan(Pose start, Pose target, float R,
                    const unsigned char* grid,
                    float max_turn, float max_straight,
                    int n_cand, std::mt19937& rng,
                    std::vector<float>& out_ex, std::vector<float>& out_ey,
                    std::vector<unsigned char>& out_coll, int& out_best_idx,
                    float& out_best_cost,
                    int& best_prim, float& best_l1, float& best_l2, float& best_l3) {
    std::uniform_int_distribution<int> prim_dist(0, N_PRIMITIVES - 1);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    out_ex.resize(n_cand);
    out_ey.resize(n_cand);
    out_coll.assign(n_cand, 1u);
    out_best_idx = -1;
    out_best_cost = std::numeric_limits<float>::infinity();
    best_prim = 0; best_l1 = best_l2 = best_l3 = 0.0f;

    auto seg_coll = [&](Pose s, int m, float len) -> bool {
        int n_check = (m == 2) ? std::max(2, (int)(len / 0.8f) + 1)
                               : std::max(2, (int)(len / 0.4f) + 1);
        for (int k = 1; k <= n_check; k++) {
            float u = static_cast<float>(k) / n_check;
            Pose p = apply_segment(s, m, len * u, R);
            int gx = (int)std::floor(p.x / GRID_RES);
            int gy = (int)std::floor(p.y / GRID_RES);
            if (gx < 0 || gx >= GRID_W || gy < 0 || gy >= GRID_H) return true;
            if (grid[gy * GRID_W + gx]) return true;
        }
        return false;
    };

    for (int i = 0; i < n_cand; i++) {
        int prim = prim_dist(rng);
        int m1, m2, m3;
        switch (prim) {
            case 0: m1=0; m2=2; m3=0; break;
            case 1: m1=1; m2=2; m3=1; break;
            case 2: m1=0; m2=2; m3=1; break;
            case 3: m1=1; m2=2; m3=0; break;
            case 4: m1=0; m2=1; m3=0; break;
            default:m1=1; m2=0; m3=1; break;
        }
        float l1 = (m1 == 2 ? max_straight : R * max_turn) * uni(rng);
        float l2 = (m2 == 2 ? max_straight : R * max_turn) * uni(rng);
        float l3 = (m3 == 2 ? max_straight : R * max_turn) * uni(rng);

        bool coll = seg_coll(start, m1, l1);
        Pose p1 = apply_segment(start, m1, l1, R);
        if (!coll) coll = seg_coll(p1, m2, l2);
        Pose p2 = apply_segment(p1, m2, l2, R);
        if (!coll) coll = seg_coll(p2, m3, l3);
        Pose p3 = apply_segment(p2, m3, l3, R);

        out_ex[i] = p3.x;
        out_ey[i] = p3.y;
        out_coll[i] = coll ? 1u : 0u;
        if (coll) continue;

        float dx = p3.x - target.x;
        float dy = p3.y - target.y;
        float dth = wrap_pi(p3.th - target.th);
        float cost = std::sqrt(dx * dx + dy * dy) + 1.5f * std::fabs(dth)
                     + 0.05f * (l1 + l2 + l3);
        if (cost < out_best_cost) {
            out_best_cost = cost;
            out_best_idx = i;
            best_prim = prim; best_l1 = l1; best_l2 = l2; best_l3 = l3;
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
static cv::Point2i world_to_px(float x, float y) {
    int px = static_cast<int>(x * VIS_SCALE_X);
    int py = PANEL_H - 1 - static_cast<int>(y * VIS_SCALE_Y);
    return {px, py};
}

static void draw_scene(cv::Mat& panel, const std::vector<unsigned char>& grid) {
    panel.setTo(cv::Scalar(245, 245, 245));
    for (int gy = 0; gy < GRID_H; gy++) {
        for (int gx = 0; gx < GRID_W; gx++) {
            if (!grid[gy * GRID_W + gx]) continue;
            int x0 = static_cast<int>(gx * GRID_RES * VIS_SCALE_X);
            int y0 = PANEL_H - 1 - static_cast<int>((gy + 1) * GRID_RES * VIS_SCALE_Y);
            int x1 = static_cast<int>((gx + 1) * GRID_RES * VIS_SCALE_X);
            int y1 = PANEL_H - 1 - static_cast<int>(gy * GRID_RES * VIS_SCALE_Y);
            cv::rectangle(panel, cv::Point(x0, y0), cv::Point(x1, y1),
                          cv::Scalar(90, 90, 90), -1);
        }
    }
}

static void draw_vehicle(cv::Mat& panel, Pose p, cv::Scalar body_color,
                         cv::Scalar border) {
    float L = 4.0f, W = 1.9f;
    cv::Point2f corners[4] = {
        {-L * 0.4f, -W * 0.5f},
        { L * 0.6f, -W * 0.5f},
        { L * 0.6f,  W * 0.5f},
        {-L * 0.4f,  W * 0.5f},
    };
    cv::Point pts[4];
    for (int i = 0; i < 4; i++) {
        float rx = corners[i].x * std::cos(p.th) - corners[i].y * std::sin(p.th);
        float ry = corners[i].x * std::sin(p.th) + corners[i].y * std::cos(p.th);
        pts[i] = world_to_px(p.x + rx, p.y + ry);
    }
    const cv::Point* poly[1] = { pts };
    int n_pts[] = { 4 };
    cv::fillPoly(panel, poly, n_pts, 1, body_color, cv::LINE_AA);
    cv::polylines(panel, poly, n_pts, 1, true, border, 1, cv::LINE_AA);
    auto hd = world_to_px(p.x + L * 0.6f * std::cos(p.th),
                          p.y + L * 0.6f * std::sin(p.th));
    cv::circle(panel, hd, 3, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
}

static void draw_dense_endpoints(cv::Mat& panel, const float* ex, const float* ey,
                                 const unsigned char* coll, int n,
                                 cv::Vec3b feas_color) {
    int stride = panel.step;
    unsigned char* data = panel.data;
    for (int i = 0; i < n; i++) {
        if (coll[i]) continue;
        int px = static_cast<int>(ex[i] * VIS_SCALE_X);
        int py = PANEL_H - 1 - static_cast<int>(ey[i] * VIS_SCALE_Y);
        if (px < 0 || px >= PANEL_W || py < 0 || py >= PANEL_H) continue;
        unsigned char* pp = data + py * stride + px * 3;
        pp[0] = feas_color[0];
        pp[1] = feas_color[1];
        pp[2] = feas_color[2];
    }
}

static void draw_sparse_endpoints(cv::Mat& panel, const float* ex, const float* ey,
                                  const unsigned char* coll, int n,
                                  cv::Scalar feas_color, cv::Scalar coll_color) {
    for (int i = 0; i < n; i++) {
        auto p = world_to_px(ex[i], ey[i]);
        cv::circle(panel, p, 3, coll[i] ? coll_color : feas_color, -1, cv::LINE_AA);
    }
}

static void draw_path(cv::Mat& panel, Pose start, int prim,
                      float l1, float l2, float l3, float R, cv::Scalar color) {
    int m1, m2, m3;
    switch (prim) {
        case 0: m1=0; m2=2; m3=0; break;
        case 1: m1=1; m2=2; m3=1; break;
        case 2: m1=0; m2=2; m3=1; break;
        case 3: m1=1; m2=2; m3=0; break;
        case 4: m1=0; m2=1; m3=0; break;
        default:m1=1; m2=0; m3=1; break;
    }
    auto draw_seg = [&](Pose s, int m, float len) {
        int N = std::max(8, (int)(len / 0.15f));
        cv::Point prev = world_to_px(s.x, s.y);
        for (int k = 1; k <= N; k++) {
            float u = static_cast<float>(k) / N;
            Pose p = apply_segment(s, m, len * u, R);
            cv::Point cur = world_to_px(p.x, p.y);
            cv::line(panel, prev, cur, color, 3, cv::LINE_AA);
            prev = cur;
        }
        return apply_segment(s, m, len, R);
    };
    Pose p1 = draw_seg(start, m1, l1);
    Pose p2 = draw_seg(p1, m2, l2);
    draw_seg(p2, m3, l3);
}

static void draw_label(cv::Mat& panel, const std::string& text, int y_offset) {
    cv::putText(panel, text, cv::Point(12, y_offset),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "Massive Reeds-Shepp Fan comparison: CPU "
              << N_CAND_CPU << " vs GPU " << N_CAND_GPU
              << " 3-segment candidate paths per frame" << std::endl;

    std::vector<unsigned char> h_grid;
    build_parking_lot(h_grid);

    unsigned char* d_grid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid, GRID_W * GRID_H));
    CUDA_CHECK(cudaMemcpy(d_grid, h_grid.data(), GRID_W * GRID_H,
                          cudaMemcpyHostToDevice));

    float* d_ex; float* d_ey;
    unsigned char* d_coll;
    float* d_cost; float* d_len;
    CUDA_CHECK(cudaMalloc(&d_ex,   N_CAND_GPU * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ey,   N_CAND_GPU * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_coll, N_CAND_GPU));
    CUDA_CHECK(cudaMalloc(&d_cost, N_CAND_GPU * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_len,  N_CAND_GPU * sizeof(float)));

    int block = 256;
    int grid_dim = (N_CAND_GPU + block - 1) / block;
    float* d_block_min_val; int* d_block_min_idx;
    CUDA_CHECK(cudaMalloc(&d_block_min_val, grid_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_min_idx, grid_dim * sizeof(int)));

    std::vector<float>         h_ex_gpu(N_CAND_GPU);
    std::vector<float>         h_ey_gpu(N_CAND_GPU);
    std::vector<unsigned char> h_coll_gpu(N_CAND_GPU);
    std::vector<float>         h_block_min_val(grid_dim);
    std::vector<int>           h_block_min_idx(grid_dim);

    cv::VideoWriter video("gif/comparison_reeds_shepp_fan.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
                          cv::Size(PANEL_W * 2, PANEL_H));

    // Target = the open parking slot in the bottom row, facing +Y.
    Pose target;
    target.x = (30 + 3 * 50 + 11) * GRID_RES;
    target.y = (22 + 18) * GRID_RES;
    target.th = static_cast<float>(M_PI) * 0.5f;

    std::mt19937 rng(42);
    double cpu_ms_sum = 0.0, gpu_ms_sum = 0.0;
    int timed = 0;

    for (int f = 0; f < SIM_FRAMES; f++) {
        float u = static_cast<float>(f) / SIM_FRAMES;
        float t = 2.0f * static_cast<float>(M_PI) * u;
        Pose start;
        start.x  = WORLD_W * 0.5f + 8.0f * std::cos(t);
        start.y  = WORLD_H * 0.5f + 5.0f * std::sin(t * 1.3f);
        start.th = t * 0.7f;

        // GPU
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        rs_fan_kernel<<<grid_dim, block>>>(start, target, TURN_R, d_grid,
                                           GRID_W, GRID_H, GRID_RES,
                                           MAX_TURN_ANGLE, MAX_STRAIGHT,
                                           N_CAND_PER_PRIM_GPU, N_PRIMITIVES,
                                           d_ex, d_ey, d_coll, d_cost, d_len);
        size_t smem = block * (sizeof(float) + sizeof(int));
        argmin_reduce_kernel<<<grid_dim, block, smem>>>(d_cost, N_CAND_GPU,
                                                       d_block_min_val,
                                                       d_block_min_idx);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);

        CUDA_CHECK(cudaMemcpy(h_ex_gpu.data(), d_ex,
                              N_CAND_GPU * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ey_gpu.data(), d_ey,
                              N_CAND_GPU * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_coll_gpu.data(), d_coll, N_CAND_GPU,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_block_min_val.data(), d_block_min_val,
                              grid_dim * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_block_min_idx.data(), d_block_min_idx,
                              grid_dim * sizeof(int), cudaMemcpyDeviceToHost));
        int    best_idx = -1;
        float  best_val = 1.0e30f;
        for (int b = 0; b < grid_dim; b++) {
            if (h_block_min_val[b] < best_val) {
                best_val = h_block_min_val[b];
                best_idx = h_block_min_idx[b];
            }
        }
        int best_prim_gpu = 0, best_t1_gpu = 0, best_t2_gpu = 0, best_t3_gpu = 0;
        float best_l1_gpu = 0, best_l2_gpu = 0, best_l3_gpu = 0;
        if (best_idx >= 0 && best_val < 1.0e8f) {
            best_prim_gpu = best_idx % N_PRIMITIVES;
            int flat = best_idx / N_PRIMITIVES;
            int nx = 56;
            best_t1_gpu = flat % nx;
            best_t2_gpu = (flat / nx) % nx;
            best_t3_gpu = (flat / (nx * nx)) % nx;
            int m1, m2, m3;
            switch (best_prim_gpu) {
                case 0: m1=0; m2=2; m3=0; break;
                case 1: m1=1; m2=2; m3=1; break;
                case 2: m1=0; m2=2; m3=1; break;
                case 3: m1=1; m2=2; m3=0; break;
                case 4: m1=0; m2=1; m3=0; break;
                default:m1=1; m2=0; m3=1; break;
            }
            best_l1_gpu = (m1 == 2 ? MAX_STRAIGHT : TURN_R * MAX_TURN_ANGLE) * (best_t1_gpu + 0.5f) / nx;
            best_l2_gpu = (m2 == 2 ? MAX_STRAIGHT : TURN_R * MAX_TURN_ANGLE) * (best_t2_gpu + 0.5f) / nx;
            best_l3_gpu = (m3 == 2 ? MAX_STRAIGHT : TURN_R * MAX_TURN_ANGLE) * (best_t3_gpu + 0.5f) / nx;
        }

        // CPU
        std::vector<float> h_ex_cpu, h_ey_cpu;
        std::vector<unsigned char> h_coll_cpu;
        int   cpu_best_idx; float cpu_best_cost; int cpu_best_prim;
        float cpu_best_l1, cpu_best_l2, cpu_best_l3;
        auto cpu_t0 = std::chrono::high_resolution_clock::now();
        cpu_fan(start, target, TURN_R, h_grid.data(), MAX_TURN_ANGLE, MAX_STRAIGHT,
                N_CAND_CPU, rng, h_ex_cpu, h_ey_cpu, h_coll_cpu,
                cpu_best_idx, cpu_best_cost, cpu_best_prim,
                cpu_best_l1, cpu_best_l2, cpu_best_l3);
        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

        if (f >= 3) { cpu_ms_sum += cpu_ms; gpu_ms_sum += gpu_ms; timed++; }

        // Render
        cv::Mat left(PANEL_H, PANEL_W, CV_8UC3);
        cv::Mat right(PANEL_H, PANEL_W, CV_8UC3);
        draw_scene(left, h_grid);
        draw_scene(right, h_grid);
        draw_vehicle(left,  target, cv::Scalar(170, 220, 170), cv::Scalar(20, 110, 20));
        draw_vehicle(right, target, cv::Scalar(170, 220, 170), cv::Scalar(20, 110, 20));
        draw_sparse_endpoints(left, h_ex_cpu.data(), h_ey_cpu.data(),
                              h_coll_cpu.data(), N_CAND_CPU,
                              cv::Scalar(0, 0, 220), cv::Scalar(200, 200, 200));
        draw_dense_endpoints(right, h_ex_gpu.data(), h_ey_gpu.data(),
                             h_coll_gpu.data(), N_CAND_GPU,
                             cv::Vec3b(40, 160, 40));
        if (cpu_best_idx >= 0) {
            draw_path(left, start, cpu_best_prim, cpu_best_l1, cpu_best_l2,
                      cpu_best_l3, TURN_R, cv::Scalar(0, 0, 220));
        }
        if (best_idx >= 0) {
            draw_path(right, start, best_prim_gpu, best_l1_gpu, best_l2_gpu,
                      best_l3_gpu, TURN_R, cv::Scalar(0, 110, 220));
        }
        draw_vehicle(left,  start, cv::Scalar(220, 200, 170), cv::Scalar(80, 50, 30));
        draw_vehicle(right, start, cv::Scalar(220, 200, 170), cv::Scalar(80, 50, 30));

        char buf[128];
        std::snprintf(buf, sizeof(buf), "CPU %d paths   %.1f ms", N_CAND_CPU, cpu_ms);
        draw_label(left, buf, 28);
        std::snprintf(buf, sizeof(buf), "GPU %d paths   %.2f ms", N_CAND_GPU, gpu_ms);
        draw_label(right, buf, 28);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        video.write(combined);
    }

    video.release();
    cudaFree(d_grid); cudaFree(d_ex); cudaFree(d_ey); cudaFree(d_coll);
    cudaFree(d_cost); cudaFree(d_len);
    cudaFree(d_block_min_val); cudaFree(d_block_min_idx);

    if (timed > 0) {
        double cpu_ms = cpu_ms_sum / timed;
        double gpu_ms = gpu_ms_sum / timed;
        double cpu_per = cpu_ms * 1.0e3 / N_CAND_CPU;
        double gpu_per = gpu_ms * 1.0e3 / N_CAND_GPU;
        std::printf("Avg CPU %.2f ms / frame (%d paths)\n"
                    "Avg GPU %.2f ms / frame (%d paths)\n"
                    "Per-candidate throughput: GPU %.4f us/path vs CPU %.3f us/path "
                    "(%.0fx faster per candidate)\n",
                    cpu_ms, N_CAND_CPU, gpu_ms, N_CAND_GPU,
                    gpu_per, cpu_per, cpu_per / gpu_per);
    }

    std::system("ffmpeg -y -i gif/comparison_reeds_shepp_fan.avi "
                "-vf 'fps=15,scale=900:-1:flags=lanczos' -loop 0 "
                "gif/comparison_reeds_shepp_fan.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_reeds_shepp_fan.gif" << std::endl;
    return 0;
}
