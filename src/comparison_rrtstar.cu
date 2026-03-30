/*************************************************************************
    RRT*: CPU vs CUDA side-by-side comparison GIF generator
    Left panel:  CPU (linear nearest + linear near-nodes + sequential collision)
    Right panel: CUDA (find_nearest_kernel + find_near_nodes_kernel + collision_check_kernel)
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

// ---------------------------------------------------------------------------
// CUDA Kernels (from rrt_star.cu)
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
// Node structure
// ---------------------------------------------------------------------------
struct Node {
    float x, y;
    int parent_idx;
    float cost;
    std::vector<float> path_x, path_y;
    Node() : x(0), y(0), parent_idx(-1), cost(0) {}
    Node(float x_, float y_) : x(x_), y(y_), parent_idx(-1), cost(0) {}
};

// ---------------------------------------------------------------------------
// Common helpers
// ---------------------------------------------------------------------------
static float calc_dist(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1, dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

static bool point_collision_free(float px, float py,
                                 const std::vector<float>& ob_x,
                                 const std::vector<float>& ob_y,
                                 const std::vector<float>& ob_r) {
    for (int i = 0; i < (int)ob_x.size(); i++) {
        float dx = ob_x[i] - px;
        float dy = ob_y[i] - py;
        if (std::sqrt(dx * dx + dy * dy) <= ob_r[i]) return false;
    }
    return true;
}

static bool path_collision_free(float fx, float fy, float tx, float ty,
                                float path_resolution,
                                const std::vector<float>& ob_x,
                                const std::vector<float>& ob_y,
                                const std::vector<float>& ob_r) {
    float dx = tx - fx, dy = ty - fy;
    float d = std::sqrt(dx * dx + dy * dy);
    if (d < 1e-6f) return point_collision_free(fx, fy, ob_x, ob_y, ob_r);
    float theta = std::atan2(dy, dx);
    int n_steps = (int)(d / path_resolution) + 1;
    for (int s = 0; s <= n_steps; s++) {
        float len = std::min(s * path_resolution, d);
        float px = fx + len * std::cos(theta);
        float py = fy + len * std::sin(theta);
        if (!point_collision_free(px, py, ob_x, ob_y, ob_r)) return false;
    }
    return true;
}

static Node steer(const std::vector<Node>& nodes, int from_idx,
                   float to_x, float to_y, float expand_dis, float path_resolution) {
    const Node& from = nodes[from_idx];
    Node nn(from.x, from.y);
    nn.parent_idx = from_idx;
    nn.cost = from.cost;

    float dx = to_x - from.x;
    float dy = to_y - from.y;
    float d = std::sqrt(dx * dx + dy * dy);
    float theta = std::atan2(dy, dx);
    float elen = std::min(expand_dis, d);

    nn.path_x.push_back(nn.x);
    nn.path_y.push_back(nn.y);
    int n_expand = (int)std::floor(elen / path_resolution);
    for (int i = 0; i < n_expand; i++) {
        nn.x += path_resolution * std::cos(theta);
        nn.y += path_resolution * std::sin(theta);
        nn.path_x.push_back(nn.x);
        nn.path_y.push_back(nn.y);
    }

    float remaining = calc_dist(nn.x, nn.y, to_x, to_y);
    if (remaining <= path_resolution && d <= expand_dis) {
        nn.x = to_x;
        nn.y = to_y;
        if (!nn.path_x.empty()) {
            nn.path_x.back() = to_x;
            nn.path_y.back() = to_y;
        }
    }

    nn.cost = from.cost + calc_dist(from.x, from.y, nn.x, nn.y);
    return nn;
}

static void propagate_cost(std::vector<Node>& nodes, int parent_idx, float path_resolution) {
    for (int i = 0; i < (int)nodes.size(); i++) {
        if (nodes[i].parent_idx == parent_idx) {
            nodes[i].cost = nodes[nodes[i].parent_idx].cost
                + calc_dist(nodes[nodes[i].parent_idx].x, nodes[nodes[i].parent_idx].y,
                            nodes[i].x, nodes[i].y);
            propagate_cost(nodes, i, path_resolution);
        }
    }
}

// ---------------------------------------------------------------------------
// CPU RRT* functions
// ---------------------------------------------------------------------------
static int cpu_find_nearest(const std::vector<Node>& nodes, float qx, float qy) {
    int best = 0;
    float best_dist = FLT_MAX;
    for (int i = 0; i < (int)nodes.size(); i++) {
        float dx = nodes[i].x - qx;
        float dy = nodes[i].y - qy;
        float d = dx * dx + dy * dy;
        if (d < best_dist) { best_dist = d; best = i; }
    }
    return best;
}

static std::vector<int> cpu_find_near_nodes(const std::vector<Node>& nodes, float qx, float qy, float radius) {
    float r2 = radius * radius;
    std::vector<int> result;
    for (int i = 0; i < (int)nodes.size(); i++) {
        float dx = nodes[i].x - qx;
        float dy = nodes[i].y - qy;
        if (dx * dx + dy * dy < r2) result.push_back(i);
    }
    return result;
}

// ---------------------------------------------------------------------------
// GPU helper
// ---------------------------------------------------------------------------
struct GpuRRTStarHelper {
    float *d_node_x, *d_node_y;
    int d_node_capacity;

    float *d_obs_cx, *d_obs_cy, *d_obs_r;
    int num_obs;

    float *d_min_dist;
    int   *d_min_idx;
    int    max_blocks;

    int *d_near_mask;
    int *d_collision_results;
    float *d_cand_x, *d_cand_y;

    GpuRRTStarHelper(const std::vector<float>& ob_x,
                     const std::vector<float>& ob_y,
                     const std::vector<float>& ob_r,
                     int init_cap)
        : d_node_x(nullptr), d_node_y(nullptr), d_node_capacity(0),
          d_min_dist(nullptr), d_min_idx(nullptr), max_blocks(0),
          d_near_mask(nullptr), d_collision_results(nullptr),
          d_cand_x(nullptr), d_cand_y(nullptr)
    {
        num_obs = (int)ob_x.size();
        CUDA_CHECK(cudaMalloc(&d_obs_cx, num_obs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_obs_cy, num_obs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_obs_r,  num_obs * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_obs_cx, ob_x.data(), num_obs * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_obs_cy, ob_y.data(), num_obs * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_obs_r,  ob_r.data(), num_obs * sizeof(float), cudaMemcpyHostToDevice));
        ensureCapacity(init_cap);
    }

    ~GpuRRTStarHelper() {
        if (d_node_x) cudaFree(d_node_x);
        if (d_node_y) cudaFree(d_node_y);
        if (d_obs_cx) cudaFree(d_obs_cx);
        if (d_obs_cy) cudaFree(d_obs_cy);
        if (d_obs_r)  cudaFree(d_obs_r);
        if (d_min_dist) cudaFree(d_min_dist);
        if (d_min_idx) cudaFree(d_min_idx);
        if (d_near_mask) cudaFree(d_near_mask);
        if (d_collision_results) cudaFree(d_collision_results);
        if (d_cand_x) cudaFree(d_cand_x);
        if (d_cand_y) cudaFree(d_cand_y);
    }

    void ensureCapacity(int needed) {
        if (needed <= d_node_capacity) return;
        int new_cap = std::max(needed, (d_node_capacity == 0) ? 1024 : d_node_capacity * 2);

        float *new_x, *new_y;
        CUDA_CHECK(cudaMalloc(&new_x, new_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&new_y, new_cap * sizeof(float)));
        if (d_node_x && d_node_capacity > 0) {
            CUDA_CHECK(cudaMemcpy(new_x, d_node_x, d_node_capacity * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(new_y, d_node_y, d_node_capacity * sizeof(float), cudaMemcpyDeviceToDevice));
            cudaFree(d_node_x);
            cudaFree(d_node_y);
        }
        d_node_x = new_x;
        d_node_y = new_y;
        d_node_capacity = new_cap;

        int new_blocks = (new_cap + 255) / 256;
        if (new_blocks > max_blocks) {
            if (d_min_dist) cudaFree(d_min_dist);
            if (d_min_idx)  cudaFree(d_min_idx);
            CUDA_CHECK(cudaMalloc(&d_min_dist, new_blocks * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_min_idx,  new_blocks * sizeof(int)));
            max_blocks = new_blocks;
        }

        if (d_near_mask) cudaFree(d_near_mask);
        if (d_collision_results) cudaFree(d_collision_results);
        if (d_cand_x) cudaFree(d_cand_x);
        if (d_cand_y) cudaFree(d_cand_y);
        CUDA_CHECK(cudaMalloc(&d_near_mask, new_cap * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_collision_results, new_cap * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cand_x, new_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cand_y, new_cap * sizeof(float)));
    }

    void uploadNode(int idx, float x, float y) {
        ensureCapacity(idx + 1);
        CUDA_CHECK(cudaMemcpy(d_node_x + idx, &x, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_node_y + idx, &y, sizeof(float), cudaMemcpyHostToDevice));
    }

    int findNearest(int num_nodes, float qx, float qy) {
        if (num_nodes <= 1) return 0;
        int block_size = 256;
        int num_blocks = (num_nodes + block_size - 1) / block_size;
        size_t smem = block_size * (sizeof(float) + sizeof(int));

        find_nearest_kernel<<<num_blocks, block_size, smem>>>(
            d_node_x, d_node_y, num_nodes, qx, qy, d_min_dist, d_min_idx);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_dist(num_blocks);
        std::vector<int>   h_idx(num_blocks);
        CUDA_CHECK(cudaMemcpy(h_dist.data(), d_min_dist, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_idx.data(),  d_min_idx,  num_blocks * sizeof(int),   cudaMemcpyDeviceToHost));

        int best = h_idx[0];
        float best_d = h_dist[0];
        for (int i = 1; i < num_blocks; i++) {
            if (h_dist[i] < best_d) { best_d = h_dist[i]; best = h_idx[i]; }
        }
        return best;
    }

    std::vector<int> findNearNodes(int num_nodes, float qx, float qy, float radius) {
        if (num_nodes == 0) return {};
        int block_size = 256;
        int num_blocks = (num_nodes + block_size - 1) / block_size;
        float r2 = radius * radius;

        find_near_nodes_kernel<<<num_blocks, block_size>>>(
            d_node_x, d_node_y, num_nodes, qx, qy, r2, d_near_mask);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> h_mask(num_nodes);
        CUDA_CHECK(cudaMemcpy(h_mask.data(), d_near_mask, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<int> near;
        for (int i = 0; i < num_nodes; i++) {
            if (h_mask[i]) near.push_back(i);
        }
        return near;
    }

    std::vector<int> collisionCheckBatch(const std::vector<Node>& nodes,
                                         const std::vector<int>& from_indices,
                                         float to_x, float to_y,
                                         float path_resolution) {
        int nc = (int)from_indices.size();
        if (nc == 0) return {};

        std::vector<float> h_fx(nc), h_fy(nc);
        for (int i = 0; i < nc; i++) {
            h_fx[i] = nodes[from_indices[i]].x;
            h_fy[i] = nodes[from_indices[i]].y;
        }

        CUDA_CHECK(cudaMemcpy(d_cand_x, h_fx.data(), nc * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cand_y, h_fy.data(), nc * sizeof(float), cudaMemcpyHostToDevice));

        int block_size = 256;
        int num_blocks = (nc + block_size - 1) / block_size;

        collision_check_kernel<<<num_blocks, block_size>>>(
            d_cand_x, d_cand_y, nc, to_x, to_y,
            d_obs_cx, d_obs_cy, d_obs_r, num_obs,
            path_resolution, d_collision_results);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> h_res(nc);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_collision_results, nc * sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<int> free_local;
        for (int i = 0; i < nc; i++) {
            if (h_res[i]) free_local.push_back(i);
        }
        return free_local;
    }
};

// ---------------------------------------------------------------------------
// Visualization helper
// ---------------------------------------------------------------------------
static cv::Point toPixel(float wx, float wy, float rand_min, int img_reso) {
    return cv::Point((int)((wx - rand_min) * img_reso),
                     (int)((wy - rand_min) * img_reso));
}

static void draw_static(cv::Mat& img,
                         float start_x, float start_y,
                         float goal_x, float goal_y,
                         const std::vector<float>& ob_x,
                         const std::vector<float>& ob_y,
                         const std::vector<float>& ob_r,
                         float rand_min, int img_reso) {
    cv::circle(img, toPixel(start_x, start_y, rand_min, img_reso), 20, cv::Scalar(0, 0, 255), -1);
    cv::circle(img, toPixel(goal_x,  goal_y,  rand_min, img_reso), 20, cv::Scalar(255, 0, 0), -1);
    for (int i = 0; i < (int)ob_x.size(); i++) {
        cv::circle(img, toPixel(ob_x[i], ob_y[i], rand_min, img_reso),
                   (int)(ob_r[i] * img_reso), cv::Scalar(0, 0, 0), -1);
    }
}

// ---------------------------------------------------------------------------
// RRT* iteration (templated for CPU/CUDA via lambdas)
// ---------------------------------------------------------------------------

// One full RRT* iteration. Returns true if a new node was added.
// Updates best_goal_idx and best_goal_cost if goal reached.
template<typename FindNearestFn, typename FindNearNodesFn, typename CollisionCheckBatchFn>
bool rrtstar_step(
    std::vector<Node>& nodes,
    float rnd_x, float rnd_y,
    float goal_x, float goal_y,
    float expand_dis, float path_resolution,
    float connect_circle_dist,
    const std::vector<float>& ob_x,
    const std::vector<float>& ob_y,
    const std::vector<float>& ob_r,
    int& best_goal_idx, float& best_goal_cost,
    FindNearestFn find_nearest_fn,
    FindNearNodesFn find_near_fn,
    CollisionCheckBatchFn collision_batch_fn)
{
    int nearest_idx = find_nearest_fn(nodes, rnd_x, rnd_y);
    if (nearest_idx < 0) return false;

    Node new_node = steer(nodes, nearest_idx, rnd_x, rnd_y, expand_dis, path_resolution);

    // Check node and path collision
    if (!point_collision_free(new_node.x, new_node.y, ob_x, ob_y, ob_r)) return false;
    for (size_t p = 0; p < new_node.path_x.size(); p++) {
        if (!point_collision_free(new_node.path_x[p], new_node.path_y[p], ob_x, ob_y, ob_r))
            return false;
    }

    // RRT* near nodes
    int nnode = (int)nodes.size() + 1;
    float r = connect_circle_dist * std::sqrt(std::log((float)nnode) / (float)nnode);
    r = std::min(r, expand_dis * 10.0f);

    std::vector<int> near_indices = find_near_fn(nodes, new_node.x, new_node.y, r);

    // Choose best parent
    if (!near_indices.empty()) {
        std::vector<int> free_local = collision_batch_fn(nodes, near_indices, new_node.x, new_node.y);

        float min_cost = new_node.cost;
        int best_parent = new_node.parent_idx;

        for (int li : free_local) {
            int ni = near_indices[li];
            float c = nodes[ni].cost + calc_dist(nodes[ni].x, nodes[ni].y, new_node.x, new_node.y);
            if (c < min_cost) {
                min_cost = c;
                best_parent = ni;
            }
        }

        new_node.parent_idx = best_parent;
        new_node.cost = min_cost;
    }

    int new_idx = (int)nodes.size();
    nodes.push_back(new_node);

    // Rewire
    for (size_t i = 0; i < near_indices.size(); i++) {
        int ni = near_indices[i];
        if (ni == new_node.parent_idx) continue;
        float improved_cost = new_node.cost + calc_dist(new_node.x, new_node.y, nodes[ni].x, nodes[ni].y);
        if (improved_cost < nodes[ni].cost) {
            if (path_collision_free(new_node.x, new_node.y, nodes[ni].x, nodes[ni].y,
                                    path_resolution, ob_x, ob_y, ob_r)) {
                nodes[ni].parent_idx = new_idx;
                nodes[ni].cost = improved_cost;
                propagate_cost(nodes, ni, path_resolution);
            }
        }
    }

    // Check goal
    float dist_to_goal = calc_dist(new_node.x, new_node.y, goal_x, goal_y);
    if (dist_to_goal <= expand_dis) {
        float goal_cost = new_node.cost + dist_to_goal;
        if (goal_cost < best_goal_cost) {
            best_goal_cost = goal_cost;
            best_goal_idx = new_idx;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    // Problem setup (same as rrt_star.cu)
    std::vector<std::vector<float>> obstacle_list{
        {5, 5, 1}, {3, 6, 2}, {3, 8, 2}, {3, 10, 2}, {7, 5, 2}, {9, 5, 2}
    };

    float start_x = 0.0f, start_y = 0.0f;
    float goal_x  = 6.0f, goal_y  = 9.0f;
    float rand_min = -2.0f, rand_max = 15.0f;
    float expand_dis = 0.5f;
    float path_resolution = 0.1f;
    int goal_sample_rate = 5;
    int max_iter = 300;
    float connect_circle_dist = 50.0f;

    int num_obstacles = (int)obstacle_list.size();
    std::vector<float> ob_x(num_obstacles), ob_y(num_obstacles), ob_r(num_obstacles);
    for (int i = 0; i < num_obstacles; i++) {
        ob_x[i] = obstacle_list[i][0];
        ob_y[i] = obstacle_list[i][1];
        ob_r[i] = obstacle_list[i][2];
    }

    unsigned int shared_seed = 42;

    // --- CPU state ---
    std::vector<Node> cpu_nodes;
    {
        Node s(start_x, start_y); s.cost = 0; s.parent_idx = -1;
        cpu_nodes.push_back(s);
    }
    std::mt19937 cpu_gen(shared_seed);
    std::uniform_int_distribution<int> cpu_goal_dis(0, 100);
    std::uniform_real_distribution<float> cpu_area_dis(rand_min, rand_max);
    int cpu_best_goal_idx = -1;
    float cpu_best_goal_cost = FLT_MAX;
    bool cpu_found = false;

    // --- CUDA state ---
    std::vector<Node> cuda_nodes;
    {
        Node s(start_x, start_y); s.cost = 0; s.parent_idx = -1;
        cuda_nodes.push_back(s);
    }
    std::mt19937 cuda_gen(shared_seed);
    std::uniform_int_distribution<int> cuda_goal_dis(0, 100);
    std::uniform_real_distribution<float> cuda_area_dis(rand_min, rand_max);
    int cuda_best_goal_idx = -1;
    float cuda_best_goal_cost = FLT_MAX;
    bool cuda_found = false;

    GpuRRTStarHelper gpu(ob_x, ob_y, ob_r, max_iter + 16);
    gpu.uploadNode(0, start_x, start_y);

    // Visualization
    int img_size = (int)(rand_max - rand_min);
    int img_reso = 50;
    int S = img_size * img_reso; // 850

    cv::VideoWriter video("gif/comparison_rrtstar.avi",
                          cv::VideoWriter::fourcc('X','V','I','D'), 30, cv::Size(S * 2, S));

    cv::Mat bg_cpu(S, S, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat bg_cuda(S, S, CV_8UC3, cv::Scalar(255, 255, 255));

    draw_static(bg_cpu,  start_x, start_y, goal_x, goal_y, ob_x, ob_y, ob_r, rand_min, img_reso);
    draw_static(bg_cuda, start_x, start_y, goal_x, goal_y, ob_x, ob_y, ob_r, rand_min, img_reso);

    double cpu_total_ms = 0.0, cuda_total_ms = 0.0;
    int cpu_iter_count = 0, cuda_iter_count = 0;

    std::cout << "RRT* comparison: CPU vs CUDA" << std::endl;

    for (int iter = 0; iter < max_iter; iter++) {
        bool drew_something = false;

        // ============ CPU iteration ============
        if (!cpu_found) {
            float rnd_x, rnd_y;
            if (cpu_goal_dis(cpu_gen) > goal_sample_rate) {
                rnd_x = cpu_area_dis(cpu_gen);
                rnd_y = cpu_area_dis(cpu_gen);
            } else {
                rnd_x = goal_x; rnd_y = goal_y;
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            bool added = rrtstar_step(
                cpu_nodes, rnd_x, rnd_y, goal_x, goal_y,
                expand_dis, path_resolution, connect_circle_dist,
                ob_x, ob_y, ob_r,
                cpu_best_goal_idx, cpu_best_goal_cost,
                // CPU find_nearest
                [](const std::vector<Node>& nodes, float qx, float qy) -> int {
                    return cpu_find_nearest(nodes, qx, qy);
                },
                // CPU find_near_nodes
                [](const std::vector<Node>& nodes, float qx, float qy, float r) -> std::vector<int> {
                    return cpu_find_near_nodes(nodes, qx, qy, r);
                },
                // CPU collision check batch (sequential)
                [&](const std::vector<Node>& nodes, const std::vector<int>& from_indices,
                    float to_x, float to_y) -> std::vector<int> {
                    std::vector<int> free_local;
                    for (int i = 0; i < (int)from_indices.size(); i++) {
                        if (path_collision_free(nodes[from_indices[i]].x, nodes[from_indices[i]].y,
                                                to_x, to_y, path_resolution, ob_x, ob_y, ob_r))
                            free_local.push_back(i);
                    }
                    return free_local;
                }
            );
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            cpu_total_ms += ms;
            cpu_iter_count++;

            if (added) {
                int new_idx = (int)cpu_nodes.size() - 1;
                Node& nn = cpu_nodes[new_idx];
                int par = nn.parent_idx;
                if (par >= 0) {
                    cv::line(bg_cpu,
                             toPixel(nn.x, nn.y, rand_min, img_reso),
                             toPixel(cpu_nodes[par].x, cpu_nodes[par].y, rand_min, img_reso),
                             cv::Scalar(0, 255, 0), 5);
                    drew_something = true;
                }
            }

            if (cpu_best_goal_idx >= 0 && !cpu_found) {
                cpu_found = true;
                // Draw best path
                int idx = cpu_best_goal_idx;
                while (idx >= 0 && cpu_nodes[idx].parent_idx >= 0) {
                    int pidx = cpu_nodes[idx].parent_idx;
                    cv::line(bg_cpu,
                             toPixel(cpu_nodes[idx].x, cpu_nodes[idx].y, rand_min, img_reso),
                             toPixel(cpu_nodes[pidx].x, cpu_nodes[pidx].y, rand_min, img_reso),
                             cv::Scalar(255, 0, 255), 5);
                    idx = pidx;
                }
                std::cout << "CPU found path at iter " << iter
                          << " cost=" << cpu_best_goal_cost << std::endl;
            }
        }

        // ============ CUDA iteration ============
        if (!cuda_found) {
            float rnd_x, rnd_y;
            if (cuda_goal_dis(cuda_gen) > goal_sample_rate) {
                rnd_x = cuda_area_dis(cuda_gen);
                rnd_y = cuda_area_dis(cuda_gen);
            } else {
                rnd_x = goal_x; rnd_y = goal_y;
            }

            cudaEvent_t ev_start, ev_stop;
            cudaEventCreate(&ev_start);
            cudaEventCreate(&ev_stop);
            cudaEventRecord(ev_start);

            bool added = rrtstar_step(
                cuda_nodes, rnd_x, rnd_y, goal_x, goal_y,
                expand_dis, path_resolution, connect_circle_dist,
                ob_x, ob_y, ob_r,
                cuda_best_goal_idx, cuda_best_goal_cost,
                // CUDA find_nearest
                [&](const std::vector<Node>& nodes, float qx, float qy) -> int {
                    return gpu.findNearest((int)nodes.size(), qx, qy);
                },
                // CUDA find_near_nodes
                [&](const std::vector<Node>& nodes, float qx, float qy, float r) -> std::vector<int> {
                    return gpu.findNearNodes((int)nodes.size(), qx, qy, r);
                },
                // CUDA collision check batch
                [&](const std::vector<Node>& nodes, const std::vector<int>& from_indices,
                    float to_x, float to_y) -> std::vector<int> {
                    return gpu.collisionCheckBatch(nodes, from_indices, to_x, to_y, path_resolution);
                }
            );

            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            float ms;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);
            cuda_total_ms += ms;
            cuda_iter_count++;

            if (added) {
                int new_idx = (int)cuda_nodes.size() - 1;
                Node& nn = cuda_nodes[new_idx];
                gpu.uploadNode(new_idx, nn.x, nn.y);
                int par = nn.parent_idx;
                if (par >= 0) {
                    cv::line(bg_cuda,
                             toPixel(nn.x, nn.y, rand_min, img_reso),
                             toPixel(cuda_nodes[par].x, cuda_nodes[par].y, rand_min, img_reso),
                             cv::Scalar(0, 255, 0), 5);
                    drew_something = true;
                }
            }

            if (cuda_best_goal_idx >= 0 && !cuda_found) {
                cuda_found = true;
                int idx = cuda_best_goal_idx;
                while (idx >= 0 && cuda_nodes[idx].parent_idx >= 0) {
                    int pidx = cuda_nodes[idx].parent_idx;
                    cv::line(bg_cuda,
                             toPixel(cuda_nodes[idx].x, cuda_nodes[idx].y, rand_min, img_reso),
                             toPixel(cuda_nodes[pidx].x, cuda_nodes[pidx].y, rand_min, img_reso),
                             cv::Scalar(255, 0, 255), 5);
                    idx = pidx;
                }
                std::cout << "CUDA found path at iter " << iter
                          << " cost=" << cuda_best_goal_cost << std::endl;
            }
        }

        // Write frame
        if (drew_something || iter % 5 == 0) {
            cv::Mat left = bg_cpu.clone();
            cv::Mat right = bg_cuda.clone();
            char buf[128];

            // CPU label
            cv::putText(left, "CPU (linear search)", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
            if (cpu_iter_count > 0) {
                snprintf(buf, sizeof(buf), "CPU: %.2f ms avg", cpu_total_ms / cpu_iter_count);
                cv::putText(left, buf, cv::Point(20, 80),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);
            }
            snprintf(buf, sizeof(buf), "Nodes: %d", (int)cpu_nodes.size());
            cv::putText(left, buf, cv::Point(20, 115),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(100, 100, 100), 2);
            if (cpu_found) {
                cv::putText(left, "PATH FOUND", cv::Point(20, 150),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 180, 0), 2);
            }

            // CUDA label
            cv::putText(right, "CUDA (GPU parallel)", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
            if (cuda_iter_count > 0) {
                snprintf(buf, sizeof(buf), "CUDA: %.2f ms avg", cuda_total_ms / cuda_iter_count);
                cv::putText(right, buf, cv::Point(20, 80),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);
            }
            snprintf(buf, sizeof(buf), "Nodes: %d", (int)cuda_nodes.size());
            cv::putText(right, buf, cv::Point(20, 115),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(100, 100, 100), 2);
            if (cuda_found) {
                cv::putText(right, "PATH FOUND", cv::Point(20, 150),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 180, 0), 2);
            }

            cv::Mat combined;
            cv::hconcat(left, right, combined);
            video.write(combined);
        }

        if (cpu_found && cuda_found) {
            // Hold final frame
            cv::Mat left = bg_cpu.clone();
            cv::Mat right = bg_cuda.clone();
            char buf[128];

            cv::putText(left, "CPU (linear search)", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
            snprintf(buf, sizeof(buf), "CPU: %.2f ms avg", cpu_total_ms / cpu_iter_count);
            cv::putText(left, buf, cv::Point(20, 80),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);
            cv::putText(left, "PATH FOUND", cv::Point(20, 150),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 180, 0), 2);

            cv::putText(right, "CUDA (GPU parallel)", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
            snprintf(buf, sizeof(buf), "CUDA: %.2f ms avg", cuda_total_ms / cuda_iter_count);
            cv::putText(right, buf, cv::Point(20, 80),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);
            cv::putText(right, "PATH FOUND", cv::Point(20, 150),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 180, 0), 2);

            cv::Mat combined;
            cv::hconcat(left, right, combined);
            for (int f = 0; f < 60; f++) video.write(combined);
            break;
        }
    }

    // If iterations exhausted without both finding paths, draw final state
    if (!cpu_found || !cuda_found) {
        cv::Mat left = bg_cpu.clone();
        cv::Mat right = bg_cuda.clone();
        char buf[128];

        cv::putText(left, "CPU (linear search)", cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
        if (cpu_iter_count > 0) {
            snprintf(buf, sizeof(buf), "CPU: %.2f ms avg", cpu_total_ms / cpu_iter_count);
            cv::putText(left, buf, cv::Point(20, 80),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);
        }
        cv::putText(left, cpu_found ? "PATH FOUND" : "NO PATH", cv::Point(20, 150),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cpu_found ? cv::Scalar(0, 180, 0) : cv::Scalar(0, 0, 200), 2);

        cv::putText(right, "CUDA (GPU parallel)", cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
        if (cuda_iter_count > 0) {
            snprintf(buf, sizeof(buf), "CUDA: %.2f ms avg", cuda_total_ms / cuda_iter_count);
            cv::putText(right, buf, cv::Point(20, 80),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);
        }
        cv::putText(right, cuda_found ? "PATH FOUND" : "NO PATH", cv::Point(20, 150),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cuda_found ? cv::Scalar(0, 180, 0) : cv::Scalar(0, 0, 200), 2);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        for (int f = 0; f < 60; f++) video.write(combined);
    }

    video.release();
    std::cout << "Video saved to gif/comparison_rrtstar.avi" << std::endl;

    if (cpu_iter_count > 0)
        std::cout << "CPU avg: " << cpu_total_ms / cpu_iter_count << " ms/iter" << std::endl;
    if (cuda_iter_count > 0)
        std::cout << "CUDA avg: " << cuda_total_ms / cuda_iter_count << " ms/iter" << std::endl;

    system("ffmpeg -y -i gif/comparison_rrtstar.avi "
           "-vf 'fps=15,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_rrtstar.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_rrtstar.gif" << std::endl;

    return 0;
}
