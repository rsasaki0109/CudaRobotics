/*************************************************************************
    > File Name: rrt.cu
    > CUDA-parallelized RRT implementation
    > Based on original by TAI Lei (ltai@ust.hk)
    > Parallelizes nearest-node search and collision checking via CUDA
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Parallel reduction to find the nearest node in the tree to a query point.
// Each thread computes the squared distance for one node, then a shared-memory
// reduction finds the minimum distance and corresponding index.
__global__ void find_nearest_kernel(const float* __restrict__ nodes_x,
                                    const float* __restrict__ nodes_y,
                                    int num_nodes,
                                    float qx, float qy,
                                    float* block_min_dist,
                                    int*   block_min_idx)
{
    extern __shared__ char smem[];
    float* s_dist = (float*)smem;
    int*   s_idx  = (int*)(s_dist + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Each thread loads one element (or infinity if out of range)
    if (gid < num_nodes) {
        float dx = nodes_x[gid] - qx;
        float dy = nodes_y[gid] - qy;
        s_dist[tid] = dx * dx + dy * dy;
        s_idx[tid]  = gid;
    } else {
        s_dist[tid] = FLT_MAX;
        s_idx[tid]  = -1;
    }
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_dist[tid + stride] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + stride];
                s_idx[tid]  = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    // Block winner writes to global arrays
    if (tid == 0) {
        block_min_dist[blockIdx.x] = s_dist[0];
        block_min_idx[blockIdx.x]  = s_idx[0];
    }
}

// Check a candidate node against all obstacles in parallel.
// Each thread checks one obstacle. result[tid] = 1 if collision, 0 otherwise.
// Host can then reduce (any nonzero means collision).
__global__ void collision_check_kernel(const float* __restrict__ ob_x,
                                       const float* __restrict__ ob_y,
                                       const float* __restrict__ ob_r,
                                       int num_obstacles,
                                       float node_x, float node_y,
                                       int* result)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < num_obstacles) {
        float dx = ob_x[gid] - node_x;
        float dy = ob_y[gid] - node_y;
        float dist = sqrtf(dx * dx + dy * dy);
        result[gid] = (dist <= ob_r[gid]) ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// Host-side node representation (simple, no Eigen)
// ---------------------------------------------------------------------------

struct Node {
    float x;
    float y;
    int parent_idx; // index into the node list, -1 for root
    Node(float x_, float y_, int p = -1) : x(x_), y(y_), parent_idx(p) {}
};

// ---------------------------------------------------------------------------
// CUDA RRT class
// ---------------------------------------------------------------------------

class CudaRRT {
public:
    CudaRRT(float sx, float sy, float gx, float gy,
            const std::vector<std::vector<float>>& obstacles,
            float rand_min, float rand_max,
            float expand_dis,
            int goal_sample_rate = 5,
            int max_iter = 5000);
    ~CudaRRT();

    std::vector<Node> planning();

private:
    // Parameters
    float start_x, start_y;
    float goal_x, goal_y;
    float expand_dis;
    float rand_min, rand_max;
    int goal_sample_rate;
    int max_iter;

    // Obstacles (host)
    int num_obstacles;
    std::vector<float> h_ob_x, h_ob_y, h_ob_r;

    // Obstacles (device)
    float *d_ob_x, *d_ob_y, *d_ob_r;
    int   *d_collision_result;

    // Tree node coordinates (device) – dynamically grown
    float *d_nodes_x, *d_nodes_y;
    int    d_nodes_capacity;

    // Block-level reduction outputs (device)
    float *d_block_min_dist;
    int   *d_block_min_idx;
    int    max_blocks;

    // Tree (host)
    std::vector<Node> node_list;

    // Random number generation
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<int> goal_dis;
    std::uniform_real_distribution<float> area_dis;

    // Internal helpers
    int  findNearest(float qx, float qy);
    bool collisionFree(float nx, float ny);
    void ensureDeviceCapacity(int needed);
    void uploadNode(int idx);
};

CudaRRT::CudaRRT(float sx, float sy, float gx, float gy,
                  const std::vector<std::vector<float>>& obstacles,
                  float rand_min_, float rand_max_,
                  float expand_dis_,
                  int goal_sample_rate_,
                  int max_iter_)
    : start_x(sx), start_y(sy), goal_x(gx), goal_y(gy),
      expand_dis(expand_dis_), rand_min(rand_min_), rand_max(rand_max_),
      goal_sample_rate(goal_sample_rate_), max_iter(max_iter_),
      gen(rd()), goal_dis(0, 100), area_dis(rand_min_, rand_max_),
      d_nodes_x(nullptr), d_nodes_y(nullptr), d_nodes_capacity(0),
      d_block_min_dist(nullptr), d_block_min_idx(nullptr), max_blocks(0)
{
    num_obstacles = (int)obstacles.size();
    h_ob_x.resize(num_obstacles);
    h_ob_y.resize(num_obstacles);
    h_ob_r.resize(num_obstacles);
    for (int i = 0; i < num_obstacles; i++) {
        h_ob_x[i] = obstacles[i][0];
        h_ob_y[i] = obstacles[i][1];
        h_ob_r[i] = obstacles[i][2];
    }

    // Allocate obstacle arrays on device
    CUDA_CHECK(cudaMalloc(&d_ob_x, num_obstacles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ob_y, num_obstacles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ob_r, num_obstacles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_collision_result, num_obstacles * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_ob_x, h_ob_x.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ob_y, h_ob_y.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ob_r, h_ob_r.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));

    // Pre-allocate node device arrays
    ensureDeviceCapacity(1024);
}

CudaRRT::~CudaRRT() {
    if (d_ob_x) cudaFree(d_ob_x);
    if (d_ob_y) cudaFree(d_ob_y);
    if (d_ob_r) cudaFree(d_ob_r);
    if (d_collision_result) cudaFree(d_collision_result);
    if (d_nodes_x) cudaFree(d_nodes_x);
    if (d_nodes_y) cudaFree(d_nodes_y);
    if (d_block_min_dist) cudaFree(d_block_min_dist);
    if (d_block_min_idx) cudaFree(d_block_min_idx);
}

void CudaRRT::ensureDeviceCapacity(int needed) {
    if (needed <= d_nodes_capacity) return;

    int new_cap = (d_nodes_capacity == 0) ? 1024 : d_nodes_capacity;
    while (new_cap < needed) new_cap *= 2;

    float *new_x, *new_y;
    CUDA_CHECK(cudaMalloc(&new_x, new_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&new_y, new_cap * sizeof(float)));

    if (d_nodes_x && d_nodes_capacity > 0) {
        CUDA_CHECK(cudaMemcpy(new_x, d_nodes_x, d_nodes_capacity * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_y, d_nodes_y, d_nodes_capacity * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaFree(d_nodes_x);
        cudaFree(d_nodes_y);
    }
    d_nodes_x = new_x;
    d_nodes_y = new_y;
    d_nodes_capacity = new_cap;

    // Recompute max blocks needed
    int block_size = 256;
    int new_max_blocks = (new_cap + block_size - 1) / block_size;
    if (new_max_blocks > max_blocks) {
        if (d_block_min_dist) cudaFree(d_block_min_dist);
        if (d_block_min_idx) cudaFree(d_block_min_idx);
        CUDA_CHECK(cudaMalloc(&d_block_min_dist, new_max_blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_block_min_idx,  new_max_blocks * sizeof(int)));
        max_blocks = new_max_blocks;
    }
}

void CudaRRT::uploadNode(int idx) {
    ensureDeviceCapacity(idx + 1);
    float x = node_list[idx].x;
    float y = node_list[idx].y;
    CUDA_CHECK(cudaMemcpy(d_nodes_x + idx, &x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes_y + idx, &y, sizeof(float), cudaMemcpyHostToDevice));
}

int CudaRRT::findNearest(float qx, float qy) {
    int n = (int)node_list.size();
    if (n == 0) return -1;
    if (n == 1) return 0;

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    size_t smem_size = block_size * (sizeof(float) + sizeof(int));

    find_nearest_kernel<<<num_blocks, block_size, smem_size>>>(
        d_nodes_x, d_nodes_y, n, qx, qy, d_block_min_dist, d_block_min_idx);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy block results back to host and do final reduction
    std::vector<float> h_dist(num_blocks);
    std::vector<int>   h_idx(num_blocks);
    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_block_min_dist, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_idx.data(),  d_block_min_idx,  num_blocks * sizeof(int),   cudaMemcpyDeviceToHost));

    int   best_idx  = h_idx[0];
    float best_dist = h_dist[0];
    for (int i = 1; i < num_blocks; i++) {
        if (h_dist[i] < best_dist) {
            best_dist = h_dist[i];
            best_idx  = h_idx[i];
        }
    }
    return best_idx;
}

bool CudaRRT::collisionFree(float nx, float ny) {
    if (num_obstacles == 0) return true;

    int block_size = 256;
    int num_blocks = (num_obstacles + block_size - 1) / block_size;

    collision_check_kernel<<<num_blocks, block_size>>>(
        d_ob_x, d_ob_y, d_ob_r, num_obstacles, nx, ny, d_collision_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_result(num_obstacles);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_collision_result, num_obstacles * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_obstacles; i++) {
        if (h_result[i]) return false;
    }
    return true;
}

std::vector<Node> CudaRRT::planning() {
    // Visualization setup
    cv::namedWindow("rrt", cv::WINDOW_NORMAL);
    int img_size = (int)(rand_max - rand_min);
    int img_reso = 50;
    cv::Mat bg(img_size * img_reso, img_size * img_reso, CV_8UC3, cv::Scalar(255, 255, 255));

    auto toPixel = [&](float wx, float wy) -> cv::Point {
        return cv::Point((int)((wx - rand_min) * img_reso),
                         (int)((wy - rand_min) * img_reso));
    };

    // Draw start, goal, obstacles
    cv::circle(bg, toPixel(start_x, start_y), 20, cv::Scalar(0, 0, 255), -1);
    cv::circle(bg, toPixel(goal_x, goal_y),   20, cv::Scalar(255, 0, 0), -1);
    for (int i = 0; i < num_obstacles; i++) {
        cv::circle(bg, toPixel(h_ob_x[i], h_ob_y[i]),
                   (int)(h_ob_r[i] * img_reso), cv::Scalar(0, 0, 0), -1);
    }

    // Add start node
    node_list.push_back(Node(start_x, start_y));
    uploadNode(0);

    int count = 0;
    bool found = false;

    while (count < max_iter) {
        // Sample random point (biased toward goal)
        float rnd_x, rnd_y;
        if (goal_dis(gen) > goal_sample_rate) {
            rnd_x = area_dis(gen);
            rnd_y = area_dis(gen);
        } else {
            rnd_x = goal_x;
            rnd_y = goal_y;
        }

        // Find nearest node using CUDA
        int nearest_idx = findNearest(rnd_x, rnd_y);
        Node& nearest = node_list[nearest_idx];

        // Extend toward random point
        float theta = std::atan2(rnd_y - nearest.y, rnd_x - nearest.x);
        float new_x = nearest.x + expand_dis * std::cos(theta);
        float new_y = nearest.y + expand_dis * std::sin(theta);

        // Collision check using CUDA
        if (!collisionFree(new_x, new_y)) {
            count++;
            continue;
        }

        // Add new node
        int new_idx = (int)node_list.size();
        node_list.push_back(Node(new_x, new_y, nearest_idx));
        uploadNode(new_idx);

        // Visualization: draw edge
        cv::line(bg, toPixel(new_x, new_y), toPixel(nearest.x, nearest.y),
                 cv::Scalar(0, 255, 0), 10);
        cv::imshow("rrt", bg);
        cv::waitKey(5);
        count++;

        // Check if goal is reached
        float dx = new_x - goal_x;
        float dy = new_y - goal_y;
        if (std::sqrt(dx * dx + dy * dy) <= expand_dis) {
            std::cout << "find path" << std::endl;
            found = true;
            break;
        }
    }

    // Trace path back
    std::vector<Node> path;
    if (found) {
        path.push_back(Node(goal_x, goal_y));
        int idx = (int)node_list.size() - 1;
        while (idx != -1) {
            Node& n = node_list[idx];
            path.push_back(n);

            // Draw path
            if (n.parent_idx != -1) {
                Node& p = node_list[n.parent_idx];
                cv::line(bg, toPixel(n.x, n.y), toPixel(p.x, p.y),
                         cv::Scalar(255, 0, 255), 10);
            }
            idx = n.parent_idx;
        }

        cv::imshow("rrt", bg);
        cv::waitKey(0);
    } else {
        std::cout << "no path found within " << max_iter << " iterations" << std::endl;
    }

    return path;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::vector<std::vector<float>> obstacle_list{
        {5, 5, 1},
        {3, 6, 2},
        {3, 8, 2},
        {3, 10, 2},
        {7, 5, 2},
        {9, 5, 2}
    };

    float start_x = 0.0f, start_y = 0.0f;
    float goal_x  = 6.0f, goal_y  = 9.0f;
    float rand_min = -2.0f, rand_max = 15.0f;
    float expand_dis = 0.5f;

    CudaRRT rrt(start_x, start_y, goal_x, goal_y,
                obstacle_list, rand_min, rand_max, expand_dis);

    std::vector<Node> path = rrt.planning();

    return 0;
}
