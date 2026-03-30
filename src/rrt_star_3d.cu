/*************************************************************************
    > File Name: rrt_star_3d.cu
    > CUDA-parallelized RRT* 3D path planning for drone/UAV navigation
    > Extends RRT* to 3D space with spherical obstacles
    > Parallelizes nearest-neighbor, radius search, and collision checking
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
    float x, y, z;
    int parent_idx;
    float cost;
    std::vector<float> path_x;
    std::vector<float> path_y;
    std::vector<float> path_z;

    Node() : x(0), y(0), z(0), parent_idx(-1), cost(0) {}
    Node(float x_, float y_, float z_) : x(x_), y(y_), z(z_), parent_idx(-1), cost(0) {}
};

// ---------------------------------------------------------------------------
// Spherical obstacle
// ---------------------------------------------------------------------------
struct Obstacle {
    float cx, cy, cz, r;
};

// ---------------------------------------------------------------------------
// CUDA Kernel: find nearest node in 3D (shared memory reduction)
// ---------------------------------------------------------------------------
__global__ void find_nearest_3d_kernel(
    const float* __restrict__ d_node_x,
    const float* __restrict__ d_node_y,
    const float* __restrict__ d_node_z,
    int num_nodes,
    float qx, float qy, float qz,
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
        float dz = d_node_z[gid] - qz;
        my_dist = dx * dx + dy * dy + dz * dz;
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
// CUDA Kernel: find all nodes within radius r in 3D
// ---------------------------------------------------------------------------
__global__ void find_near_nodes_3d_kernel(
    const float* __restrict__ d_node_x,
    const float* __restrict__ d_node_y,
    const float* __restrict__ d_node_z,
    int num_nodes,
    float qx, float qy, float qz,
    float radius_sq,
    int* d_near_mask)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < num_nodes) {
        float dx = d_node_x[gid] - qx;
        float dy = d_node_y[gid] - qy;
        float dz = d_node_z[gid] - qz;
        d_near_mask[gid] = (dx * dx + dy * dy + dz * dz < radius_sq) ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// CUDA Kernel: collision check for a single 3D line segment vs spherical obstacles
//   Checks path from (fx, fy, fz) to (to_x, to_y, to_z) against all obstacles.
//   d_results[gid] = 1 means collision-free, 0 means collision.
// ---------------------------------------------------------------------------
__global__ void collision_check_3d_kernel(
    float fx, float fy, float fz,
    float to_x, float to_y, float to_z,
    const float* __restrict__ d_obs_cx,
    const float* __restrict__ d_obs_cy,
    const float* __restrict__ d_obs_cz,
    const float* __restrict__ d_obs_r,
    int num_obs,
    float path_resolution,
    int* d_result)
{
    // Single-thread collision check for one segment
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid != 0) return;

    float dx = to_x - fx;
    float dy = to_y - fy;
    float dz = to_z - fz;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);

    if (dist < 1e-6f) {
        for (int o = 0; o < num_obs; o++) {
            float ox = d_obs_cx[o] - fx;
            float oy = d_obs_cy[o] - fy;
            float oz = d_obs_cz[o] - fz;
            if (sqrtf(ox * ox + oy * oy + oz * oz) <= d_obs_r[o]) {
                d_result[0] = 0;
                return;
            }
        }
        d_result[0] = 1;
        return;
    }

    float inv_dist = 1.0f / dist;
    float ux = dx * inv_dist;
    float uy = dy * inv_dist;
    float uz = dz * inv_dist;
    int n_steps = (int)(dist / path_resolution) + 1;

    for (int s = 0; s <= n_steps; s++) {
        float len = fminf(s * path_resolution, dist);
        float px = fx + len * ux;
        float py = fy + len * uy;
        float pz = fz + len * uz;
        for (int o = 0; o < num_obs; o++) {
            float ox = d_obs_cx[o] - px;
            float oy = d_obs_cy[o] - py;
            float oz = d_obs_cz[o] - pz;
            if (sqrtf(ox * ox + oy * oy + oz * oz) <= d_obs_r[o]) {
                d_result[0] = 0;
                return;
            }
        }
    }
    d_result[0] = 1;
}

// ---------------------------------------------------------------------------
// CUDA Kernel: batch collision check for multiple candidate parents in 3D
//   Each thread checks one candidate parent -> (to_x, to_y, to_z) path.
// ---------------------------------------------------------------------------
__global__ void batch_collision_check_kernel(
    const float* __restrict__ d_from_x,
    const float* __restrict__ d_from_y,
    const float* __restrict__ d_from_z,
    int num_candidates,
    float to_x, float to_y, float to_z,
    const float* __restrict__ d_obs_cx,
    const float* __restrict__ d_obs_cy,
    const float* __restrict__ d_obs_cz,
    const float* __restrict__ d_obs_r,
    int num_obs,
    float path_resolution,
    int* d_results)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_candidates) return;

    float fx = d_from_x[gid];
    float fy = d_from_y[gid];
    float fz = d_from_z[gid];
    float dx = to_x - fx;
    float dy = to_y - fy;
    float dz = to_z - fz;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);

    if (dist < 1e-6f) {
        for (int o = 0; o < num_obs; o++) {
            float ox = d_obs_cx[o] - fx;
            float oy = d_obs_cy[o] - fy;
            float oz = d_obs_cz[o] - fz;
            if (sqrtf(ox * ox + oy * oy + oz * oz) <= d_obs_r[o]) {
                d_results[gid] = 0;
                return;
            }
        }
        d_results[gid] = 1;
        return;
    }

    float inv_dist = 1.0f / dist;
    float ux = dx * inv_dist;
    float uy = dy * inv_dist;
    float uz = dz * inv_dist;
    int n_steps = (int)(dist / path_resolution) + 1;

    for (int s = 0; s <= n_steps; s++) {
        float len = fminf(s * path_resolution, dist);
        float px = fx + len * ux;
        float py = fy + len * uy;
        float pz = fz + len * uz;
        for (int o = 0; o < num_obs; o++) {
            float ox = d_obs_cx[o] - px;
            float oy = d_obs_cy[o] - py;
            float oz = d_obs_cz[o] - pz;
            if (sqrtf(ox * ox + oy * oy + oz * oz) <= d_obs_r[o]) {
                d_results[gid] = 0;
                return;
            }
        }
    }
    d_results[gid] = 1;
}

// ---------------------------------------------------------------------------
// RRTStar3D class
// ---------------------------------------------------------------------------
class RRTStar3D {
public:
    RRTStar3D(float start_x, float start_y, float start_z,
              float goal_x, float goal_y, float goal_z,
              const std::vector<Obstacle>& obstacles,
              float rand_min, float rand_max,
              float expand_dis,
              float path_resolution,
              int goal_sample_rate,
              int max_iter,
              float connect_circle_dist);
    ~RRTStar3D();

    std::vector<Node> planning();

private:
    // Parameters
    float start_x_, start_y_, start_z_;
    float goal_x_, goal_y_, goal_z_;
    std::vector<Obstacle> obstacles_;
    float rand_min_, rand_max_;
    float expand_dis_;
    float path_resolution_;
    int goal_sample_rate_;
    int max_iter_;
    float connect_circle_dist_;

    // Tree
    std::vector<Node> node_list_;

    // Device memory for node positions
    float* d_node_x_;
    float* d_node_y_;
    float* d_node_z_;
    int    d_node_capacity_;

    // Device memory for obstacles
    float* d_obs_cx_;
    float* d_obs_cy_;
    float* d_obs_cz_;
    float* d_obs_r_;
    int    num_obs_;

    // Device memory for kernel outputs
    float* d_min_dist_;
    int*   d_min_idx_;
    int*   d_near_mask_;
    int*   d_collision_results_;
    int*   d_single_result_;
    float* d_cand_x_;
    float* d_cand_y_;
    float* d_cand_z_;

    int max_blocks_;

    // Random
    std::mt19937 rng_;
    std::uniform_int_distribution<int> goal_dist_;
    std::uniform_real_distribution<float> area_dist_;

    // Methods
    void ensure_device_capacity(int needed);
    void sync_node_to_device(int idx);
    int  find_nearest_gpu(float qx, float qy, float qz);
    std::vector<int> find_near_nodes_gpu(float qx, float qy, float qz, float radius);
    std::vector<int> batch_collision_check_gpu(const std::vector<int>& from_indices,
                                               float to_x, float to_y, float to_z);
    bool single_collision_check_gpu(float fx, float fy, float fz,
                                    float tx, float ty, float tz);
    bool point_collision_check(float x, float y, float z);
    Node steer(int from_idx, float to_x, float to_y, float to_z, float extend_length);
    float calc_dist(float x1, float y1, float z1, float x2, float y2, float z2);
    float calc_new_cost(int from_idx, float to_x, float to_y, float to_z);
    void propagate_cost_to_leaves(int parent_idx);
};

RRTStar3D::RRTStar3D(float start_x, float start_y, float start_z,
                      float goal_x, float goal_y, float goal_z,
                      const std::vector<Obstacle>& obstacles,
                      float rand_min, float rand_max,
                      float expand_dis,
                      float path_resolution,
                      int goal_sample_rate,
                      int max_iter,
                      float connect_circle_dist)
    : start_x_(start_x), start_y_(start_y), start_z_(start_z)
    , goal_x_(goal_x), goal_y_(goal_y), goal_z_(goal_z)
    , obstacles_(obstacles)
    , rand_min_(rand_min), rand_max_(rand_max)
    , expand_dis_(expand_dis)
    , path_resolution_(path_resolution)
    , goal_sample_rate_(goal_sample_rate)
    , max_iter_(max_iter)
    , connect_circle_dist_(connect_circle_dist)
    , d_node_x_(nullptr), d_node_y_(nullptr), d_node_z_(nullptr), d_node_capacity_(0)
    , d_obs_cx_(nullptr), d_obs_cy_(nullptr), d_obs_cz_(nullptr), d_obs_r_(nullptr)
    , d_min_dist_(nullptr), d_min_idx_(nullptr)
    , d_near_mask_(nullptr), d_collision_results_(nullptr), d_single_result_(nullptr)
    , d_cand_x_(nullptr), d_cand_y_(nullptr), d_cand_z_(nullptr)
    , rng_(std::random_device{}())
    , goal_dist_(0, 100)
    , area_dist_(rand_min, rand_max)
{
    num_obs_ = (int)obstacles_.size();

    // Upload obstacles to device
    std::vector<float> ocx(num_obs_), ocy(num_obs_), ocz(num_obs_), orr(num_obs_);
    for (int i = 0; i < num_obs_; i++) {
        ocx[i] = obstacles_[i].cx;
        ocy[i] = obstacles_[i].cy;
        ocz[i] = obstacles_[i].cz;
        orr[i] = obstacles_[i].r;
    }
    CUDA_CHECK(cudaMalloc(&d_obs_cx_, num_obs_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obs_cy_, num_obs_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obs_cz_, num_obs_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obs_r_,  num_obs_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_obs_cx_, ocx.data(), num_obs_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_cy_, ocy.data(), num_obs_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_cz_, ocz.data(), num_obs_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_r_,  orr.data(), num_obs_ * sizeof(float), cudaMemcpyHostToDevice));

    // Pre-allocate workspace
    int init_cap = max_iter + 16;
    max_blocks_ = (init_cap + 255) / 256;

    CUDA_CHECK(cudaMalloc(&d_min_dist_, max_blocks_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_min_idx_,  max_blocks_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_near_mask_, init_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_collision_results_, init_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_single_result_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cand_x_, init_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cand_y_, init_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cand_z_, init_cap * sizeof(float)));

    ensure_device_capacity(init_cap);
}

RRTStar3D::~RRTStar3D() {
    cudaFree(d_node_x_);
    cudaFree(d_node_y_);
    cudaFree(d_node_z_);
    cudaFree(d_obs_cx_);
    cudaFree(d_obs_cy_);
    cudaFree(d_obs_cz_);
    cudaFree(d_obs_r_);
    cudaFree(d_min_dist_);
    cudaFree(d_min_idx_);
    cudaFree(d_near_mask_);
    cudaFree(d_collision_results_);
    cudaFree(d_single_result_);
    cudaFree(d_cand_x_);
    cudaFree(d_cand_y_);
    cudaFree(d_cand_z_);
}

void RRTStar3D::ensure_device_capacity(int needed) {
    if (needed <= d_node_capacity_) return;
    int new_cap = std::max(needed, d_node_capacity_ * 2);

    float* new_x = nullptr;
    float* new_y = nullptr;
    float* new_z = nullptr;
    CUDA_CHECK(cudaMalloc(&new_x, new_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&new_y, new_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&new_z, new_cap * sizeof(float)));

    if (d_node_x_ && d_node_capacity_ > 0) {
        CUDA_CHECK(cudaMemcpy(new_x, d_node_x_, d_node_capacity_ * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_y, d_node_y_, d_node_capacity_ * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_z, d_node_z_, d_node_capacity_ * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaFree(d_node_x_);
        cudaFree(d_node_y_);
        cudaFree(d_node_z_);
    }

    d_node_x_ = new_x;
    d_node_y_ = new_y;
    d_node_z_ = new_z;
    d_node_capacity_ = new_cap;

    // Resize workspace arrays if needed
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
    cudaFree(d_cand_z_);
    CUDA_CHECK(cudaMalloc(&d_near_mask_, new_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_collision_results_, new_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cand_x_, new_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cand_y_, new_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cand_z_, new_cap * sizeof(float)));
}

void RRTStar3D::sync_node_to_device(int idx) {
    ensure_device_capacity(idx + 1);
    float x = node_list_[idx].x;
    float y = node_list_[idx].y;
    float z = node_list_[idx].z;
    CUDA_CHECK(cudaMemcpy(d_node_x_ + idx, &x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_y_ + idx, &y, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_z_ + idx, &z, sizeof(float), cudaMemcpyHostToDevice));
}

int RRTStar3D::find_nearest_gpu(float qx, float qy, float qz) {
    int n = (int)node_list_.size();
    if (n == 0) return -1;

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    size_t shared_mem = block_size * (sizeof(float) + sizeof(int));

    find_nearest_3d_kernel<<<num_blocks, block_size, shared_mem>>>(
        d_node_x_, d_node_y_, d_node_z_, n, qx, qy, qz, d_min_dist_, d_min_idx_);
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

std::vector<int> RRTStar3D::find_near_nodes_gpu(float qx, float qy, float qz, float radius) {
    int n = (int)node_list_.size();
    if (n == 0) return {};

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    float radius_sq = radius * radius;

    find_near_nodes_3d_kernel<<<num_blocks, block_size>>>(
        d_node_x_, d_node_y_, d_node_z_, n, qx, qy, qz, radius_sq, d_near_mask_);
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

std::vector<int> RRTStar3D::batch_collision_check_gpu(
    const std::vector<int>& from_indices, float to_x, float to_y, float to_z)
{
    int nc = (int)from_indices.size();
    if (nc == 0) return {};

    std::vector<float> h_fx(nc), h_fy(nc), h_fz(nc);
    for (int i = 0; i < nc; i++) {
        h_fx[i] = node_list_[from_indices[i]].x;
        h_fy[i] = node_list_[from_indices[i]].y;
        h_fz[i] = node_list_[from_indices[i]].z;
    }

    CUDA_CHECK(cudaMemcpy(d_cand_x_, h_fx.data(), nc * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cand_y_, h_fy.data(), nc * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cand_z_, h_fz.data(), nc * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 256;
    int num_blocks = (nc + block_size - 1) / block_size;

    batch_collision_check_kernel<<<num_blocks, block_size>>>(
        d_cand_x_, d_cand_y_, d_cand_z_, nc, to_x, to_y, to_z,
        d_obs_cx_, d_obs_cy_, d_obs_cz_, d_obs_r_, num_obs_,
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

bool RRTStar3D::single_collision_check_gpu(float fx, float fy, float fz,
                                            float tx, float ty, float tz) {
    collision_check_3d_kernel<<<1, 1>>>(
        fx, fy, fz, tx, ty, tz,
        d_obs_cx_, d_obs_cy_, d_obs_cz_, d_obs_r_, num_obs_,
        path_resolution_, d_single_result_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int result;
    CUDA_CHECK(cudaMemcpy(&result, d_single_result_, sizeof(int), cudaMemcpyDeviceToHost));
    return result == 1;
}

bool RRTStar3D::point_collision_check(float x, float y, float z) {
    for (const auto& ob : obstacles_) {
        float dx = ob.cx - x;
        float dy = ob.cy - y;
        float dz = ob.cz - z;
        if (std::sqrt(dx * dx + dy * dy + dz * dz) <= ob.r) return false;
    }
    return true;
}

float RRTStar3D::calc_dist(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

Node RRTStar3D::steer(int from_idx, float to_x, float to_y, float to_z, float extend_length) {
    Node& from = node_list_[from_idx];
    Node new_node(from.x, from.y, from.z);
    new_node.parent_idx = from_idx;
    new_node.cost = from.cost;

    float dx = to_x - from.x;
    float dy = to_y - from.y;
    float dz = to_z - from.z;
    float d = std::sqrt(dx * dx + dy * dy + dz * dz);

    if (d < 1e-6f) return new_node;

    float elen = std::min(extend_length, d);
    float inv_d = 1.0f / d;
    float ux = dx * inv_d;
    float uy = dy * inv_d;
    float uz = dz * inv_d;

    new_node.path_x.push_back(new_node.x);
    new_node.path_y.push_back(new_node.y);
    new_node.path_z.push_back(new_node.z);

    int n_expand = (int)std::floor(elen / path_resolution_);
    for (int i = 0; i < n_expand; i++) {
        new_node.x += path_resolution_ * ux;
        new_node.y += path_resolution_ * uy;
        new_node.z += path_resolution_ * uz;
        new_node.path_x.push_back(new_node.x);
        new_node.path_y.push_back(new_node.y);
        new_node.path_z.push_back(new_node.z);
    }

    // Snap to target if close enough
    float remaining = calc_dist(new_node.x, new_node.y, new_node.z, to_x, to_y, to_z);
    if (remaining <= path_resolution_) {
        new_node.x = to_x;
        new_node.y = to_y;
        new_node.z = to_z;
        if (!new_node.path_x.empty()) {
            new_node.path_x.back() = to_x;
            new_node.path_y.back() = to_y;
            new_node.path_z.back() = to_z;
        }
    }

    new_node.cost = from.cost + calc_dist(from.x, from.y, from.z, new_node.x, new_node.y, new_node.z);
    return new_node;
}

float RRTStar3D::calc_new_cost(int from_idx, float to_x, float to_y, float to_z) {
    return node_list_[from_idx].cost +
           calc_dist(node_list_[from_idx].x, node_list_[from_idx].y, node_list_[from_idx].z,
                     to_x, to_y, to_z);
}

void RRTStar3D::propagate_cost_to_leaves(int parent_idx) {
    for (int i = 0; i < (int)node_list_.size(); i++) {
        if (node_list_[i].parent_idx == parent_idx) {
            node_list_[i].cost = node_list_[node_list_[i].parent_idx].cost
                + calc_dist(node_list_[node_list_[i].parent_idx].x,
                            node_list_[node_list_[i].parent_idx].y,
                            node_list_[node_list_[i].parent_idx].z,
                            node_list_[i].x, node_list_[i].y, node_list_[i].z);
            propagate_cost_to_leaves(i);
        }
    }
}

std::vector<Node> RRTStar3D::planning() {
    // Visualization setup: XY (top view) and XZ (side view) side by side
    cv::namedWindow("rrt_star_3d", cv::WINDOW_NORMAL);
    float range = rand_max_ - rand_min_;
    int img_reso = 12;
    int view_w = (int)(range * img_reso);
    int view_h = (int)(range * img_reso);
    int gap = 20;
    cv::Mat bg(view_h, view_w * 2 + gap, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw separator
    cv::rectangle(bg, cv::Rect(view_w, 0, gap, view_h), cv::Scalar(200, 200, 200), -1);

    // Lambda to convert 3D coords to pixel positions in each view
    auto xy_pt = [&](float x, float y) -> cv::Point {
        return cv::Point((int)((x - rand_min_) * img_reso),
                         (int)((y - rand_min_) * img_reso));
    };
    auto xz_pt = [&](float x, float z) -> cv::Point {
        return cv::Point(view_w + gap + (int)((x - rand_min_) * img_reso),
                         (int)((z - rand_min_) * img_reso));
    };

    // Draw obstacles in both views
    for (const auto& ob : obstacles_) {
        int r_px = (int)(ob.r * img_reso);
        cv::circle(bg, xy_pt(ob.cx, ob.cy), r_px, cv::Scalar(0, 0, 0), -1);
        cv::circle(bg, xz_pt(ob.cx, ob.cz), r_px, cv::Scalar(0, 0, 0), -1);
    }

    // Draw start and goal
    int marker_r = 8;
    cv::circle(bg, xy_pt(start_x_, start_y_), marker_r, cv::Scalar(0, 0, 255), -1);
    cv::circle(bg, xz_pt(start_x_, start_z_), marker_r, cv::Scalar(0, 0, 255), -1);
    cv::circle(bg, xy_pt(goal_x_, goal_y_),  marker_r, cv::Scalar(255, 0, 0), -1);
    cv::circle(bg, xz_pt(goal_x_, goal_z_),  marker_r, cv::Scalar(255, 0, 0), -1);

    // Labels
    cv::putText(bg, "XY (top)", cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    cv::putText(bg, "XZ (side)", cv::Point(view_w + gap + 10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    // Initialize tree with start node
    Node start_node(start_x_, start_y_, start_z_);
    start_node.cost = 0.0f;
    start_node.parent_idx = -1;
    node_list_.clear();
    node_list_.push_back(start_node);
    sync_node_to_device(0);

    int best_goal_idx = -1;
    float best_goal_cost = std::numeric_limits<float>::max();

    for (int iter = 0; iter < max_iter_; iter++) {
        // Sample random point (or goal)
        float rnd_x, rnd_y, rnd_z;
        if (goal_dist_(rng_) > goal_sample_rate_) {
            rnd_x = area_dist_(rng_);
            rnd_y = area_dist_(rng_);
            rnd_z = area_dist_(rng_);
        } else {
            rnd_x = goal_x_;
            rnd_y = goal_y_;
            rnd_z = goal_z_;
        }

        // Find nearest node (GPU)
        int nearest_idx = find_nearest_gpu(rnd_x, rnd_y, rnd_z);
        if (nearest_idx < 0) continue;

        // Steer toward random point
        Node new_node = steer(nearest_idx, rnd_x, rnd_y, rnd_z, expand_dis_);

        // Quick collision check on the new node position
        if (!point_collision_check(new_node.x, new_node.y, new_node.z)) continue;

        // Check the path from parent to new node on GPU
        if (!single_collision_check_gpu(node_list_[nearest_idx].x,
                                        node_list_[nearest_idx].y,
                                        node_list_[nearest_idx].z,
                                        new_node.x, new_node.y, new_node.z))
            continue;

        // --- RRT* specific: find near nodes, choose parent, rewire ---

        int nnode = (int)node_list_.size() + 1;
        float r = connect_circle_dist_ * std::sqrt(std::log((float)nnode) / (float)nnode);
        r = std::min(r, expand_dis_ * 10.0f);

        // Find near nodes (GPU)
        std::vector<int> near_indices = find_near_nodes_gpu(new_node.x, new_node.y, new_node.z, r);

        // Choose best parent: parallel collision check for all near nodes -> new_node
        if (!near_indices.empty()) {
            std::vector<int> free_local_indices = batch_collision_check_gpu(
                near_indices, new_node.x, new_node.y, new_node.z);

            float min_cost = new_node.cost;
            int best_parent = new_node.parent_idx;

            for (int li : free_local_indices) {
                int ni = near_indices[li];
                float c = calc_new_cost(ni, new_node.x, new_node.y, new_node.z);
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
                float dz = new_node.z - par.z;
                float d = std::sqrt(dx * dx + dy * dy + dz * dz);

                if (d > 1e-6f) {
                    float inv_d = 1.0f / d;
                    float ux = dx * inv_d;
                    float uy = dy * inv_d;
                    float uz = dz * inv_d;

                    new_node.path_x.clear();
                    new_node.path_y.clear();
                    new_node.path_z.clear();
                    float cx = par.x, cy = par.y, cz = par.z;
                    new_node.path_x.push_back(cx);
                    new_node.path_y.push_back(cy);
                    new_node.path_z.push_back(cz);
                    int ns = (int)std::floor(d / path_resolution_);
                    for (int s = 0; s < ns; s++) {
                        cx += path_resolution_ * ux;
                        cy += path_resolution_ * uy;
                        cz += path_resolution_ * uz;
                        new_node.path_x.push_back(cx);
                        new_node.path_y.push_back(cy);
                        new_node.path_z.push_back(cz);
                    }
                    new_node.path_x.push_back(new_node.x);
                    new_node.path_y.push_back(new_node.y);
                    new_node.path_z.push_back(new_node.z);
                }
            }
        }

        // Add new node to tree
        int new_idx = (int)node_list_.size();
        node_list_.push_back(new_node);
        sync_node_to_device(new_idx);

        // Rewire: check if routing through new_node is cheaper for near nodes
        if (!near_indices.empty()) {
            for (size_t i = 0; i < near_indices.size(); i++) {
                int ni = near_indices[i];
                if (ni == new_node.parent_idx) continue;

                float improved_cost = new_node.cost +
                    calc_dist(new_node.x, new_node.y, new_node.z,
                              node_list_[ni].x, node_list_[ni].y, node_list_[ni].z);

                if (improved_cost < node_list_[ni].cost) {
                    if (single_collision_check_gpu(new_node.x, new_node.y, new_node.z,
                                                   node_list_[ni].x, node_list_[ni].y, node_list_[ni].z)) {
                        node_list_[ni].parent_idx = new_idx;
                        node_list_[ni].cost = improved_cost;
                        propagate_cost_to_leaves(ni);
                    }
                }
            }
        }

        // Visualization: draw edge from new_node to its parent
        int par = new_node.parent_idx;
        if (par >= 0) {
            cv::line(bg, xy_pt(new_node.x, new_node.y),
                     xy_pt(node_list_[par].x, node_list_[par].y),
                     cv::Scalar(0, 255, 0), 1);
            cv::line(bg, xz_pt(new_node.x, new_node.z),
                     xz_pt(node_list_[par].x, node_list_[par].z),
                     cv::Scalar(0, 255, 0), 1);
        }

        if (iter % 50 == 0) {
            cv::imshow("rrt_star_3d", bg);
            cv::waitKey(1);
        }

        // Check if we reached the goal
        float dist_to_goal = calc_dist(new_node.x, new_node.y, new_node.z,
                                        goal_x_, goal_y_, goal_z_);
        if (dist_to_goal <= expand_dis_) {
            float goal_cost = new_node.cost + dist_to_goal;
            if (goal_cost < best_goal_cost) {
                best_goal_cost = goal_cost;
                best_goal_idx = new_idx;
            }
        }
    }

    // Show final tree
    cv::imshow("rrt_star_3d", bg);
    cv::waitKey(1);

    // Extract path
    std::vector<Node> path;
    if (best_goal_idx >= 0) {
        std::cout << "Found path! Cost: " << best_goal_cost << std::endl;

        Node goal_node(goal_x_, goal_y_, goal_z_);
        goal_node.cost = best_goal_cost;
        goal_node.parent_idx = best_goal_idx;
        path.push_back(goal_node);

        int idx = best_goal_idx;
        while (idx >= 0) {
            path.push_back(node_list_[idx]);

            // Draw final path in magenta
            if (node_list_[idx].parent_idx >= 0) {
                int pidx = node_list_[idx].parent_idx;
                cv::line(bg, xy_pt(node_list_[idx].x, node_list_[idx].y),
                         xy_pt(node_list_[pidx].x, node_list_[pidx].y),
                         cv::Scalar(255, 0, 255), 3);
                cv::line(bg, xz_pt(node_list_[idx].x, node_list_[idx].z),
                         xz_pt(node_list_[pidx].x, node_list_[pidx].z),
                         cv::Scalar(255, 0, 255), 3);
            }

            idx = node_list_[idx].parent_idx;
        }

        // Redraw start/goal on top
        cv::circle(bg, xy_pt(start_x_, start_y_), marker_r, cv::Scalar(0, 0, 255), -1);
        cv::circle(bg, xz_pt(start_x_, start_z_), marker_r, cv::Scalar(0, 0, 255), -1);
        cv::circle(bg, xy_pt(goal_x_, goal_y_),  marker_r, cv::Scalar(255, 0, 0), -1);
        cv::circle(bg, xz_pt(goal_x_, goal_z_),  marker_r, cv::Scalar(255, 0, 0), -1);

        cv::imshow("rrt_star_3d", bg);
        cv::waitKey(0);

        std::reverse(path.begin(), path.end());
    } else {
        std::cout << "No path found within " << max_iter_ << " iterations." << std::endl;
        cv::imshow("rrt_star_3d", bg);
        cv::waitKey(0);
    }

    return path;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    // Spherical obstacles: (cx, cy, cz, radius)
    std::vector<Obstacle> obstacles = {
        {15.0f, 15.0f, 15.0f, 5.0f},
        {25.0f, 25.0f, 25.0f, 5.0f},
        {35.0f, 15.0f, 25.0f, 4.0f},
        {10.0f, 30.0f, 30.0f, 4.0f},
        {30.0f, 10.0f, 15.0f, 3.0f},
        {20.0f, 35.0f, 20.0f, 4.0f},
        {40.0f, 25.0f, 35.0f, 3.0f}
    };

    float start_x = 0.0f, start_y = 0.0f, start_z = 0.0f;
    float goal_x  = 45.0f, goal_y = 45.0f, goal_z = 45.0f;
    float rand_min = 0.0f, rand_max = 50.0f;
    float expand_dis = 3.0f;
    float path_resolution = 0.5f;
    int   goal_sample_rate = 10;
    int   max_iter = 1000;
    float connect_circle_dist = 30.0f;

    RRTStar3D rrt_star(start_x, start_y, start_z,
                       goal_x, goal_y, goal_z,
                       obstacles, rand_min, rand_max,
                       expand_dis, path_resolution,
                       goal_sample_rate, max_iter,
                       connect_circle_dist);

    std::vector<Node> path = rrt_star.planning();

    if (!path.empty()) {
        std::cout << "Path nodes: " << path.size() << std::endl;
        for (const auto& n : path) {
            printf("  (%.2f, %.2f, %.2f) cost=%.2f\n", n.x, n.y, n.z, n.cost);
        }
    }

    return 0;
}
