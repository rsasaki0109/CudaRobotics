/*************************************************************************
    > File Name: informed_rrt_star.cu
    > CUDA-parallelized Informed RRT* path planning
    > After finding an initial path, samples from an ellipsoidal subset
    > of the state space to accelerate convergence to the optimal path.
    > Reference: Gammell et al., "Informed RRT*"
 ************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <limits>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

// ---------------------------------------------------------------------------
// Node (host-side tree structure)
// ---------------------------------------------------------------------------
struct Node {
    float x;
    float y;
    int parent_idx;   // index into node list (-1 = no parent)
    float cost;
    std::vector<float> path_x;
    std::vector<float> path_y;

    Node() : x(0), y(0), parent_idx(-1), cost(0) {}
    Node(float x_, float y_) : x(x_), y(y_), parent_idx(-1), cost(0) {}
};

// ---------------------------------------------------------------------------
// Obstacle (host-side, for transfer to device)
// ---------------------------------------------------------------------------
struct Obstacle {
    float cx, cy, r;
};

// ---------------------------------------------------------------------------
// CUDA Kernel: find nearest node to a query point
//   Each thread computes distance for one node; then block-level reduction.
//   Result: d_min_dist[blockIdx], d_min_idx[blockIdx]
// ---------------------------------------------------------------------------
__global__ void find_nearest_kernel(
    const float* __restrict__ d_node_x,
    const float* __restrict__ d_node_y,
    int num_nodes,
    float qx, float qy,
    float* d_min_dist,
    int* d_min_idx)
{
    extern __shared__ char shared_mem[];
    float* s_dist = (float*)shared_mem;
    int*   s_idx  = (int*)(s_dist + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float my_dist = FLT_MAX;
    int   my_idx  = -1;

    if (gid < num_nodes) {
        float dx = d_node_x[gid] - qx;
        float dy = d_node_y[gid] - qy;
        my_dist = dx * dx + dy * dy;
        my_idx  = gid;
    }

    s_dist[tid] = my_dist;
    s_idx[tid]  = my_idx;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_dist[tid + s] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + s];
                s_idx[tid]  = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_min_dist[blockIdx.x] = s_dist[0];
        d_min_idx[blockIdx.x]  = s_idx[0];
    }
}

// ---------------------------------------------------------------------------
// CUDA Kernel: find all nodes within radius r
//   Each thread checks one node. Writes 1/0 to d_near_mask.
// ---------------------------------------------------------------------------
__global__ void find_near_nodes_kernel(
    const float* __restrict__ d_node_x,
    const float* __restrict__ d_node_y,
    int num_nodes,
    float qx, float qy,
    float radius_sq,
    int* d_near_mask)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < num_nodes) {
        float dx = d_node_x[gid] - qx;
        float dy = d_node_y[gid] - qy;
        d_near_mask[gid] = (dx * dx + dy * dy < radius_sq) ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// CUDA Kernel: collision check for a set of line segments vs obstacles
//   Each thread checks one candidate parent -> new_node path against all
//   obstacles. d_results[tid] = 1 means collision-free, 0 means collision.
// ---------------------------------------------------------------------------
__global__ void collision_check_kernel(
    const float* __restrict__ d_from_x,
    const float* __restrict__ d_from_y,
    int num_candidates,
    float to_x, float to_y,
    const float* __restrict__ d_obs_cx,
    const float* __restrict__ d_obs_cy,
    const float* __restrict__ d_obs_r,
    int num_obs,
    float path_resolution,
    int* d_results)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_candidates) return;

    float fx = d_from_x[gid];
    float fy = d_from_y[gid];
    float dx = to_x - fx;
    float dy = to_y - fy;
    float dist = sqrtf(dx * dx + dy * dy);

    if (dist < 1e-6f) {
        for (int o = 0; o < num_obs; o++) {
            float ox = d_obs_cx[o] - fx;
            float oy = d_obs_cy[o] - fy;
            if (sqrtf(ox * ox + oy * oy) <= d_obs_r[o]) {
                d_results[gid] = 0;
                return;
            }
        }
        d_results[gid] = 1;
        return;
    }

    float theta = atan2f(dy, dx);
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    int n_steps = (int)(dist / path_resolution) + 1;

    for (int s = 0; s <= n_steps; s++) {
        float len = fminf(s * path_resolution, dist);
        float px = fx + len * cos_t;
        float py = fy + len * sin_t;
        for (int o = 0; o < num_obs; o++) {
            float ox = d_obs_cx[o] - px;
            float oy = d_obs_cy[o] - py;
            if (sqrtf(ox * ox + oy * oy) <= d_obs_r[o]) {
                d_results[gid] = 0;
                return;
            }
        }
    }
    d_results[gid] = 1;
}

// ---------------------------------------------------------------------------
// InformedRRTStar class (host-side, manages tree and device memory)
// ---------------------------------------------------------------------------
class InformedRRTStar {
public:
    InformedRRTStar(float start_x, float start_y,
                    float goal_x, float goal_y,
                    const std::vector<Obstacle>& obstacles,
                    float rand_min, float rand_max,
                    float expand_dis,
                    float path_resolution,
                    int goal_sample_rate,
                    int max_iter,
                    float connect_circle_dist);
    ~InformedRRTStar();

    std::vector<Node> planning();

private:
    // Parameters
    float start_x_, start_y_;
    float goal_x_, goal_y_;
    std::vector<Obstacle> obstacles_;
    float rand_min_, rand_max_;
    float expand_dis_;
    float path_resolution_;
    int goal_sample_rate_;
    int max_iter_;
    float connect_circle_dist_;

    // Tree
    std::vector<Node> node_list_;

    // Device memory for node positions (resized as tree grows)
    float* d_node_x_;
    float* d_node_y_;
    int    d_node_capacity_;

    // Device memory for obstacles (fixed)
    float* d_obs_cx_;
    float* d_obs_cy_;
    float* d_obs_r_;
    int    num_obs_;

    // Device memory for kernel outputs
    float* d_min_dist_;
    int*   d_min_idx_;
    int*   d_near_mask_;
    int*   d_collision_results_;
    float* d_cand_x_;
    float* d_cand_y_;

    int max_blocks_;

    // Random
    std::mt19937 rng_;
    std::uniform_int_distribution<int> goal_dist_;
    std::uniform_real_distribution<float> area_dist_;
    std::uniform_real_distribution<float> unit_dist_;  // [0,1)

    // Informed sampling state
    float c_min_;         // distance from start to goal
    float x_center_[2];   // midpoint of start and goal
    float C_[2][2];       // rotation matrix from world frame to ellipse frame

    // Methods
    void ensure_device_capacity(int needed);
    void sync_node_to_device(int idx);
    int  find_nearest_gpu(float qx, float qy);
    std::vector<int> find_near_nodes_gpu(float qx, float qy, float radius);
    std::vector<int> collision_check_gpu(const std::vector<int>& from_indices, float to_x, float to_y);
    bool single_collision_check(float x, float y);
    Node steer(int from_idx, float to_x, float to_y, float extend_length);
    float calc_dist(float x1, float y1, float x2, float y2);
    float calc_new_cost(int from_idx, float to_x, float to_y);
    void propagate_cost_to_leaves(int parent_idx);

    // Informed RRT* specific
    void compute_sampling_space();
    void informed_sample(float c_best, float& out_x, float& out_y);
};

InformedRRTStar::InformedRRTStar(float start_x, float start_y,
                                  float goal_x, float goal_y,
                                  const std::vector<Obstacle>& obstacles,
                                  float rand_min, float rand_max,
                                  float expand_dis,
                                  float path_resolution,
                                  int goal_sample_rate,
                                  int max_iter,
                                  float connect_circle_dist)
    : start_x_(start_x), start_y_(start_y)
    , goal_x_(goal_x), goal_y_(goal_y)
    , obstacles_(obstacles)
    , rand_min_(rand_min), rand_max_(rand_max)
    , expand_dis_(expand_dis)
    , path_resolution_(path_resolution)
    , goal_sample_rate_(goal_sample_rate)
    , max_iter_(max_iter)
    , connect_circle_dist_(connect_circle_dist)
    , d_node_x_(nullptr), d_node_y_(nullptr), d_node_capacity_(0)
    , d_obs_cx_(nullptr), d_obs_cy_(nullptr), d_obs_r_(nullptr)
    , d_min_dist_(nullptr), d_min_idx_(nullptr)
    , d_near_mask_(nullptr), d_collision_results_(nullptr)
    , d_cand_x_(nullptr), d_cand_y_(nullptr)
    , rng_(std::random_device{}())
    , goal_dist_(0, 100)
    , area_dist_(rand_min, rand_max)
    , unit_dist_(0.0f, 1.0f)
{
    num_obs_ = (int)obstacles_.size();

    // Upload obstacles to device
    std::vector<float> ocx(num_obs_), ocy(num_obs_), orr(num_obs_);
    for (int i = 0; i < num_obs_; i++) {
        ocx[i] = obstacles_[i].cx;
        ocy[i] = obstacles_[i].cy;
        orr[i] = obstacles_[i].r;
    }
    CUDA_CHECK(cudaMalloc(&d_obs_cx_, num_obs_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obs_cy_, num_obs_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obs_r_,  num_obs_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_obs_cx_, ocx.data(), num_obs_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_cy_, ocy.data(), num_obs_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_r_,  orr.data(), num_obs_ * sizeof(float), cudaMemcpyHostToDevice));

    // Pre-allocate workspace
    int init_cap = max_iter + 16;
    max_blocks_ = (init_cap + 255) / 256;

    CUDA_CHECK(cudaMalloc(&d_min_dist_, max_blocks_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_min_idx_,  max_blocks_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_near_mask_, init_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_collision_results_, init_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cand_x_, init_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cand_y_, init_cap * sizeof(float)));

    ensure_device_capacity(init_cap);

    // Precompute informed sampling parameters
    compute_sampling_space();
}

InformedRRTStar::~InformedRRTStar() {
    cudaFree(d_node_x_);
    cudaFree(d_node_y_);
    cudaFree(d_obs_cx_);
    cudaFree(d_obs_cy_);
    cudaFree(d_obs_r_);
    cudaFree(d_min_dist_);
    cudaFree(d_min_idx_);
    cudaFree(d_near_mask_);
    cudaFree(d_collision_results_);
    cudaFree(d_cand_x_);
    cudaFree(d_cand_y_);
}

// ---------------------------------------------------------------------------
// Compute the rotation matrix and parameters for ellipsoidal sampling
// ---------------------------------------------------------------------------
void InformedRRTStar::compute_sampling_space() {
    // c_min = distance from start to goal
    float dx = goal_x_ - start_x_;
    float dy = goal_y_ - start_y_;
    c_min_ = std::sqrt(dx * dx + dy * dy);

    // Center of the ellipse = midpoint of start and goal
    x_center_[0] = (start_x_ + goal_x_) / 2.0f;
    x_center_[1] = (start_y_ + goal_y_) / 2.0f;

    // Rotation matrix C that aligns the x-axis with the start-goal line
    // The angle from the x-axis to the start-goal vector
    float theta = std::atan2(dy, dx);
    C_[0][0] = std::cos(theta);
    C_[0][1] = -std::sin(theta);
    C_[1][0] = std::sin(theta);
    C_[1][1] = std::cos(theta);
}

// ---------------------------------------------------------------------------
// Sample from the informed ellipsoidal region
// Uses rejection-free method: sample uniformly in unit disk, scale to ellipse
// ---------------------------------------------------------------------------
void InformedRRTStar::informed_sample(float c_best, float& out_x, float& out_y) {
    if (c_best >= std::numeric_limits<float>::max()) {
        // No solution yet: sample uniformly
        out_x = area_dist_(rng_);
        out_y = area_dist_(rng_);
        return;
    }

    // Semi-major axis a = c_best / 2
    // Semi-minor axis b = sqrt(c_best^2 - c_min^2) / 2
    float a = c_best / 2.0f;
    float b = std::sqrt(std::max(0.0f, c_best * c_best - c_min_ * c_min_)) / 2.0f;

    // Sample uniformly in the unit disk using rejection
    float ux, uy;
    do {
        ux = unit_dist_(rng_) * 2.0f - 1.0f;
        uy = unit_dist_(rng_) * 2.0f - 1.0f;
    } while (ux * ux + uy * uy > 1.0f);

    // Scale to ellipse (in local frame)
    float ex = a * ux;
    float ey = b * uy;

    // Rotate to world frame and translate to center
    out_x = C_[0][0] * ex + C_[0][1] * ey + x_center_[0];
    out_y = C_[1][0] * ex + C_[1][1] * ey + x_center_[1];
}

void InformedRRTStar::ensure_device_capacity(int needed) {
    if (needed <= d_node_capacity_) return;
    int new_cap = std::max(needed, d_node_capacity_ * 2);

    float* new_x = nullptr;
    float* new_y = nullptr;
    CUDA_CHECK(cudaMalloc(&new_x, new_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&new_y, new_cap * sizeof(float)));

    if (d_node_x_ && d_node_capacity_ > 0) {
        CUDA_CHECK(cudaMemcpy(new_x, d_node_x_, d_node_capacity_ * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_y, d_node_y_, d_node_capacity_ * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaFree(d_node_x_);
        cudaFree(d_node_y_);
    }

    d_node_x_ = new_x;
    d_node_y_ = new_y;
    d_node_capacity_ = new_cap;

    // Also resize workspace arrays if needed
    int new_blocks = (new_cap + 255) / 256;
    if (new_blocks > max_blocks_) {
        cudaFree(d_min_dist_);
        cudaFree(d_min_idx_);
        CUDA_CHECK(cudaMalloc(&d_min_dist_, new_blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_min_idx_,  new_blocks * sizeof(int)));
        max_blocks_ = new_blocks;
    }

    cudaFree(d_near_mask_);
    cudaFree(d_collision_results_);
    cudaFree(d_cand_x_);
    cudaFree(d_cand_y_);
    CUDA_CHECK(cudaMalloc(&d_near_mask_, new_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_collision_results_, new_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cand_x_, new_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cand_y_, new_cap * sizeof(float)));
}

void InformedRRTStar::sync_node_to_device(int idx) {
    ensure_device_capacity(idx + 1);
    float x = node_list_[idx].x;
    float y = node_list_[idx].y;
    CUDA_CHECK(cudaMemcpy(d_node_x_ + idx, &x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_y_ + idx, &y, sizeof(float), cudaMemcpyHostToDevice));
}

int InformedRRTStar::find_nearest_gpu(float qx, float qy) {
    int n = (int)node_list_.size();
    if (n == 0) return -1;

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    size_t shared_mem = block_size * (sizeof(float) + sizeof(int));

    find_nearest_kernel<<<num_blocks, block_size, shared_mem>>>(
        d_node_x_, d_node_y_, n, qx, qy, d_min_dist_, d_min_idx_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Final reduction on host
    std::vector<float> h_dist(num_blocks);
    std::vector<int>   h_idx(num_blocks);
    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_min_dist_, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_idx.data(),  d_min_idx_,  num_blocks * sizeof(int),   cudaMemcpyDeviceToHost));

    float best_dist = FLT_MAX;
    int   best_idx  = -1;
    for (int b = 0; b < num_blocks; b++) {
        if (h_dist[b] < best_dist) {
            best_dist = h_dist[b];
            best_idx  = h_idx[b];
        }
    }
    return best_idx;
}

std::vector<int> InformedRRTStar::find_near_nodes_gpu(float qx, float qy, float radius) {
    int n = (int)node_list_.size();
    if (n == 0) return {};

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    float radius_sq = radius * radius;

    find_near_nodes_kernel<<<num_blocks, block_size>>>(
        d_node_x_, d_node_y_, n, qx, qy, radius_sq, d_near_mask_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_mask(n);
    CUDA_CHECK(cudaMemcpy(h_mask.data(), d_near_mask_, n * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> near_indices;
    for (int i = 0; i < n; i++) {
        if (h_mask[i]) near_indices.push_back(i);
    }
    return near_indices;
}

std::vector<int> InformedRRTStar::collision_check_gpu(const std::vector<int>& from_indices, float to_x, float to_y) {
    int nc = (int)from_indices.size();
    if (nc == 0) return {};

    std::vector<float> h_fx(nc), h_fy(nc);
    for (int i = 0; i < nc; i++) {
        h_fx[i] = node_list_[from_indices[i]].x;
        h_fy[i] = node_list_[from_indices[i]].y;
    }

    CUDA_CHECK(cudaMemcpy(d_cand_x_, h_fx.data(), nc * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cand_y_, h_fy.data(), nc * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 256;
    int num_blocks = (nc + block_size - 1) / block_size;

    collision_check_kernel<<<num_blocks, block_size>>>(
        d_cand_x_, d_cand_y_, nc, to_x, to_y,
        d_obs_cx_, d_obs_cy_, d_obs_r_, num_obs_,
        path_resolution_, d_collision_results_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_results(nc);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_collision_results_, nc * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> free_indices;
    for (int i = 0; i < nc; i++) {
        if (h_results[i]) free_indices.push_back(i);
    }
    return free_indices;
}

bool InformedRRTStar::single_collision_check(float x, float y) {
    for (const auto& ob : obstacles_) {
        float dx = ob.cx - x;
        float dy = ob.cy - y;
        if (std::sqrt(dx * dx + dy * dy) <= ob.r) return false;
    }
    return true;
}

float InformedRRTStar::calc_dist(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

Node InformedRRTStar::steer(int from_idx, float to_x, float to_y, float extend_length) {
    Node& from = node_list_[from_idx];
    Node new_node(from.x, from.y);
    new_node.parent_idx = from_idx;
    new_node.cost = from.cost;

    float dx = to_x - from.x;
    float dy = to_y - from.y;
    float d = std::sqrt(dx * dx + dy * dy);
    float theta = std::atan2(dy, dx);

    float elen = std::min(extend_length, d);

    new_node.path_x.push_back(new_node.x);
    new_node.path_y.push_back(new_node.y);

    int n_expand = (int)std::floor(elen / path_resolution_);
    for (int i = 0; i < n_expand; i++) {
        new_node.x += path_resolution_ * std::cos(theta);
        new_node.y += path_resolution_ * std::sin(theta);
        new_node.path_x.push_back(new_node.x);
        new_node.path_y.push_back(new_node.y);
    }

    // Snap to target if close enough
    float remaining = calc_dist(new_node.x, new_node.y, to_x, to_y);
    if (remaining <= path_resolution_) {
        new_node.x = to_x;
        new_node.y = to_y;
        if (!new_node.path_x.empty()) {
            new_node.path_x.back() = to_x;
            new_node.path_y.back() = to_y;
        }
    }

    new_node.cost = from.cost + calc_dist(from.x, from.y, new_node.x, new_node.y);
    return new_node;
}

float InformedRRTStar::calc_new_cost(int from_idx, float to_x, float to_y) {
    return node_list_[from_idx].cost + calc_dist(node_list_[from_idx].x, node_list_[from_idx].y, to_x, to_y);
}

void InformedRRTStar::propagate_cost_to_leaves(int parent_idx) {
    for (int i = 0; i < (int)node_list_.size(); i++) {
        if (node_list_[i].parent_idx == parent_idx) {
            node_list_[i].cost = node_list_[node_list_[i].parent_idx].cost
                + calc_dist(node_list_[node_list_[i].parent_idx].x,
                            node_list_[node_list_[i].parent_idx].y,
                            node_list_[i].x, node_list_[i].y);
            propagate_cost_to_leaves(i);
        }
    }
}

// ---------------------------------------------------------------------------
// Extract path from goal back to start
// ---------------------------------------------------------------------------
static std::vector<std::pair<float,float>> extract_path(
    const std::vector<Node>& node_list, int best_goal_idx,
    float goal_x, float goal_y, float best_goal_cost)
{
    std::vector<std::pair<float,float>> path;
    path.push_back({goal_x, goal_y});

    int idx = best_goal_idx;
    while (idx >= 0) {
        path.push_back({node_list[idx].x, node_list[idx].y});
        idx = node_list[idx].parent_idx;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

// ---------------------------------------------------------------------------
// Main planning loop
// ---------------------------------------------------------------------------
std::vector<Node> InformedRRTStar::planning() {
    // Visualization setup
    cv::namedWindow("informed_rrt_star", cv::WINDOW_NORMAL);
    int img_size = (int)(rand_max_ - rand_min_);
    int img_reso = 50;
    cv::Mat bg(img_size * img_reso, img_size * img_reso,
               CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw obstacles on background
    for (const auto& ob : obstacles_) {
        cv::circle(bg,
            cv::Point((int)((ob.cx - rand_min_) * img_reso),
                      (int)((ob.cy - rand_min_) * img_reso)),
            (int)(ob.r * img_reso), cv::Scalar(0, 0, 0), -1);
    }

    // Initialize tree with start node
    Node start_node(start_x_, start_y_);
    start_node.cost = 0.0f;
    start_node.parent_idx = -1;
    node_list_.clear();
    node_list_.push_back(start_node);
    sync_node_to_device(0);

    int best_goal_idx = -1;
    float best_goal_cost = std::numeric_limits<float>::max();

    for (int iter = 0; iter < max_iter_; iter++) {
        // --- Informed sampling ---
        float rnd_x, rnd_y;
        if (goal_dist_(rng_) <= goal_sample_rate_) {
            rnd_x = goal_x_;
            rnd_y = goal_y_;
        } else {
            informed_sample(best_goal_cost, rnd_x, rnd_y);
        }

        // Find nearest node (GPU)
        int nearest_idx = find_nearest_gpu(rnd_x, rnd_y);
        if (nearest_idx < 0) continue;

        // Steer toward random point
        Node new_node = steer(nearest_idx, rnd_x, rnd_y, expand_dis_);

        // Quick collision check on the new node position
        if (!single_collision_check(new_node.x, new_node.y)) continue;

        // Check the path from parent to new node
        bool path_ok = true;
        for (size_t p = 0; p < new_node.path_x.size(); p++) {
            if (!single_collision_check(new_node.path_x[p], new_node.path_y[p])) {
                path_ok = false;
                break;
            }
        }
        if (!path_ok) continue;

        // --- RRT* specific: find near nodes, choose parent, rewire ---

        int nnode = (int)node_list_.size() + 1;
        float r = connect_circle_dist_ * std::sqrt(std::log((float)nnode) / (float)nnode);
        r = std::min(r, expand_dis_ * 10.0f);

        // Find near nodes (GPU)
        std::vector<int> near_indices = find_near_nodes_gpu(new_node.x, new_node.y, r);

        // Choose best parent: parallel collision check for all near nodes -> new_node
        if (!near_indices.empty()) {
            std::vector<int> free_local_indices = collision_check_gpu(near_indices, new_node.x, new_node.y);

            float min_cost = new_node.cost;
            int best_parent = new_node.parent_idx;

            for (int li : free_local_indices) {
                int ni = near_indices[li];
                float c = calc_new_cost(ni, new_node.x, new_node.y);
                if (c < min_cost) {
                    min_cost = c;
                    best_parent = ni;
                }
            }

            new_node.parent_idx = best_parent;
            new_node.cost = min_cost;

            // Rebuild path from chosen parent
            if (best_parent != nearest_idx) {
                Node& par = node_list_[best_parent];
                float dx = new_node.x - par.x;
                float dy = new_node.y - par.y;
                float d = std::sqrt(dx * dx + dy * dy);
                float theta = std::atan2(dy, dx);

                new_node.path_x.clear();
                new_node.path_y.clear();
                float cx = par.x, cy = par.y;
                new_node.path_x.push_back(cx);
                new_node.path_y.push_back(cy);
                int ns = (int)std::floor(d / path_resolution_);
                for (int s = 0; s < ns; s++) {
                    cx += path_resolution_ * std::cos(theta);
                    cy += path_resolution_ * std::sin(theta);
                    new_node.path_x.push_back(cx);
                    new_node.path_y.push_back(cy);
                }
                new_node.path_x.push_back(new_node.x);
                new_node.path_y.push_back(new_node.y);
            }
        }

        // Add new node to tree
        int new_idx = (int)node_list_.size();
        node_list_.push_back(new_node);
        sync_node_to_device(new_idx);

        // Rewire near nodes through new_node if cheaper
        if (!near_indices.empty()) {
            for (size_t i = 0; i < near_indices.size(); i++) {
                int ni = near_indices[i];
                if (ni == new_node.parent_idx) continue;

                float improved_cost = new_node.cost + calc_dist(new_node.x, new_node.y,
                                                                 node_list_[ni].x, node_list_[ni].y);
                if (improved_cost < node_list_[ni].cost) {
                    // Check collision along path from new_node to near[i]
                    float dx = node_list_[ni].x - new_node.x;
                    float dy = node_list_[ni].y - new_node.y;
                    float d = std::sqrt(dx * dx + dy * dy);
                    float theta = std::atan2(dy, dx);
                    bool ok = true;
                    int ns = (int)(d / path_resolution_) + 1;
                    for (int s = 0; s <= ns && ok; s++) {
                        float len = std::min(s * path_resolution_, d);
                        float px = new_node.x + len * std::cos(theta);
                        float py = new_node.y + len * std::sin(theta);
                        if (!single_collision_check(px, py)) ok = false;
                    }

                    if (ok) {
                        node_list_[ni].parent_idx = new_idx;
                        node_list_[ni].cost = improved_cost;
                        propagate_cost_to_leaves(ni);
                    }
                }
            }
        }

        // Check if we reached the goal
        float dist_to_goal = calc_dist(new_node.x, new_node.y, goal_x_, goal_y_);
        if (dist_to_goal <= expand_dis_) {
            float goal_cost = new_node.cost + dist_to_goal;
            if (goal_cost < best_goal_cost) {
                best_goal_cost = goal_cost;
                best_goal_idx = new_idx;
                std::cout << "Iter " << iter << ": found path, cost = " << best_goal_cost << std::endl;
            }
        }

        // --- Visualization ---
        cv::Mat frame = bg.clone();

        // Draw all tree edges (green)
        for (int i = 1; i < (int)node_list_.size(); i++) {
            int par = node_list_[i].parent_idx;
            if (par >= 0) {
                cv::line(frame,
                    cv::Point((int)((node_list_[i].x - rand_min_) * img_reso),
                              (int)((node_list_[i].y - rand_min_) * img_reso)),
                    cv::Point((int)((node_list_[par].x - rand_min_) * img_reso),
                              (int)((node_list_[par].y - rand_min_) * img_reso)),
                    cv::Scalar(0, 255, 0), 5);
            }
        }

        // Draw sampling ellipse (yellow) if we have a solution
        if (best_goal_cost < std::numeric_limits<float>::max()) {
            float a = best_goal_cost / 2.0f;
            float b_axis = std::sqrt(std::max(0.0f, best_goal_cost * best_goal_cost - c_min_ * c_min_)) / 2.0f;
            float angle_deg = std::atan2(goal_y_ - start_y_, goal_x_ - start_x_) * 180.0f / (float)M_PI;

            cv::ellipse(frame,
                cv::Point((int)((x_center_[0] - rand_min_) * img_reso),
                          (int)((x_center_[1] - rand_min_) * img_reso)),
                cv::Size((int)(a * img_reso), (int)(b_axis * img_reso)),
                angle_deg, 0, 360,
                cv::Scalar(0, 255, 255), 5);
        }

        // Draw current best path (magenta)
        if (best_goal_idx >= 0) {
            // Draw from goal to best_goal_idx
            auto path_pts = extract_path(node_list_, best_goal_idx, goal_x_, goal_y_, best_goal_cost);
            for (size_t p = 1; p < path_pts.size(); p++) {
                cv::line(frame,
                    cv::Point((int)((path_pts[p-1].first - rand_min_) * img_reso),
                              (int)((path_pts[p-1].second - rand_min_) * img_reso)),
                    cv::Point((int)((path_pts[p].first - rand_min_) * img_reso),
                              (int)((path_pts[p].second - rand_min_) * img_reso)),
                    cv::Scalar(255, 0, 255), 10);
            }
        }

        // Draw start (red) and goal (blue)
        cv::circle(frame,
            cv::Point((int)((start_x_ - rand_min_) * img_reso),
                      (int)((start_y_ - rand_min_) * img_reso)),
            20, cv::Scalar(0, 0, 255), -1);
        cv::circle(frame,
            cv::Point((int)((goal_x_ - rand_min_) * img_reso),
                      (int)((goal_y_ - rand_min_) * img_reso)),
            20, cv::Scalar(255, 0, 0), -1);

        cv::imshow("informed_rrt_star", frame);
        cv::waitKey(5);
    }

    // Extract final result
    std::vector<Node> path;
    if (best_goal_idx >= 0) {
        std::cout << "Final path cost: " << best_goal_cost << std::endl;

        Node goal_node(goal_x_, goal_y_);
        goal_node.cost = best_goal_cost;
        goal_node.parent_idx = best_goal_idx;
        path.push_back(goal_node);

        int idx = best_goal_idx;
        while (idx >= 0) {
            path.push_back(node_list_[idx]);
            idx = node_list_[idx].parent_idx;
        }

        std::reverse(path.begin(), path.end());

        // Final visualization with path highlighted
        cv::Mat frame = bg.clone();

        // Tree edges
        for (int i = 1; i < (int)node_list_.size(); i++) {
            int par = node_list_[i].parent_idx;
            if (par >= 0) {
                cv::line(frame,
                    cv::Point((int)((node_list_[i].x - rand_min_) * img_reso),
                              (int)((node_list_[i].y - rand_min_) * img_reso)),
                    cv::Point((int)((node_list_[par].x - rand_min_) * img_reso),
                              (int)((node_list_[par].y - rand_min_) * img_reso)),
                    cv::Scalar(0, 255, 0), 5);
            }
        }

        // Final path (magenta)
        for (size_t i = 1; i < path.size(); i++) {
            cv::line(frame,
                cv::Point((int)((path[i-1].x - rand_min_) * img_reso),
                          (int)((path[i-1].y - rand_min_) * img_reso)),
                cv::Point((int)((path[i].x - rand_min_) * img_reso),
                          (int)((path[i].y - rand_min_) * img_reso)),
                cv::Scalar(255, 0, 255), 10);
        }

        // Start and goal
        cv::circle(frame,
            cv::Point((int)((start_x_ - rand_min_) * img_reso),
                      (int)((start_y_ - rand_min_) * img_reso)),
            20, cv::Scalar(0, 0, 255), -1);
        cv::circle(frame,
            cv::Point((int)((goal_x_ - rand_min_) * img_reso),
                      (int)((goal_y_ - rand_min_) * img_reso)),
            20, cv::Scalar(255, 0, 0), -1);

        cv::imshow("informed_rrt_star", frame);
        cv::waitKey(0);
    } else {
        std::cout << "No path found within " << max_iter_ << " iterations." << std::endl;
        cv::imshow("informed_rrt_star", bg);
        cv::waitKey(0);
    }

    return path;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::vector<Obstacle> obstacles = {
        {5.0f, 5.0f, 1.0f},
        {3.0f, 6.0f, 2.0f},
        {3.0f, 8.0f, 2.0f},
        {3.0f, 10.0f, 2.0f},
        {7.0f, 5.0f, 2.0f},
        {9.0f, 5.0f, 2.0f}
    };

    float start_x = 0.0f, start_y = 0.0f;
    float goal_x  = 6.0f, goal_y  = 9.0f;
    float rand_min = -2.0f, rand_max = 15.0f;
    float expand_dis = 1.0f;
    float path_resolution = 0.1f;
    int   goal_sample_rate = 10;
    int   max_iter = 500;
    float connect_circle_dist = 50.0f;

    InformedRRTStar planner(start_x, start_y, goal_x, goal_y,
                            obstacles, rand_min, rand_max,
                            expand_dis, path_resolution,
                            goal_sample_rate, max_iter,
                            connect_circle_dist);

    std::vector<Node> path = planner.planning();

    if (!path.empty()) {
        std::cout << "Path nodes: " << path.size() << std::endl;
        for (const auto& n : path) {
            std::cout << "  (" << n.x << ", " << n.y << ") cost=" << n.cost << std::endl;
        }
    }

    return 0;
}
