/*************************************************************************
    RRT* 3D: CPU vs CUDA side-by-side comparison GIF generator
    Left panel:  CPU (linear 3D nearest neighbor search)
    Right panel: CUDA (find_nearest_3d_kernel with shared memory reduction)
    Both use the same obstacles, start, goal as rrt_star_3d.cu
    Shows XY projection on both sides.
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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

// ---------------------------------------------------------------------------
// Node structure (3D)
// ---------------------------------------------------------------------------
struct Node {
    float x, y, z;
    int parent_idx;
    float cost;
    Node() : x(0), y(0), z(0), parent_idx(-1), cost(0) {}
    Node(float x_, float y_, float z_, int p = -1)
        : x(x_), y(y_), z(z_), parent_idx(p), cost(0) {}
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
// CUDA Kernel: parallel collision check against spherical obstacles
// ---------------------------------------------------------------------------
__global__ void collision_check_3d_kernel(
    const float* __restrict__ d_obs_cx,
    const float* __restrict__ d_obs_cy,
    const float* __restrict__ d_obs_cz,
    const float* __restrict__ d_obs_r,
    int num_obs,
    float node_x, float node_y, float node_z,
    int* d_result)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < num_obs) {
        float dx = d_obs_cx[gid] - node_x;
        float dy = d_obs_cy[gid] - node_y;
        float dz = d_obs_cz[gid] - node_z;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        d_result[gid] = (dist <= d_obs_r[gid]) ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// CPU: linear 3D nearest neighbor search
// ---------------------------------------------------------------------------
int cpu_find_nearest_3d(const std::vector<Node>& nodes, float qx, float qy, float qz) {
    int best = 0;
    float best_dist = FLT_MAX;
    for (int i = 0; i < (int)nodes.size(); i++) {
        float dx = nodes[i].x - qx;
        float dy = nodes[i].y - qy;
        float dz = nodes[i].z - qz;
        float d = dx * dx + dy * dy + dz * dz;
        if (d < best_dist) { best_dist = d; best = i; }
    }
    return best;
}

// CPU: linear collision check (3D spherical obstacles)
bool cpu_collision_free_3d(float nx, float ny, float nz,
                           const std::vector<Obstacle>& obstacles) {
    for (int i = 0; i < (int)obstacles.size(); i++) {
        float dx = obstacles[i].cx - nx;
        float dy = obstacles[i].cy - ny;
        float dz = obstacles[i].cz - nz;
        if (std::sqrt(dx * dx + dy * dy + dz * dz) <= obstacles[i].r) return false;
    }
    return true;
}

// CPU: path collision check (walk along segment)
bool cpu_path_collision_free_3d(float fx, float fy, float fz,
                                float tx, float ty, float tz,
                                const std::vector<Obstacle>& obstacles,
                                float path_resolution) {
    float dx = tx - fx, dy = ty - fy, dz = tz - fz;
    float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < 1e-6f) return cpu_collision_free_3d(fx, fy, fz, obstacles);
    int n_steps = (int)(dist / path_resolution) + 1;
    for (int s = 0; s <= n_steps; s++) {
        float t = (float)s / (float)n_steps;
        float px = fx + t * dx, py = fy + t * dy, pz = fz + t * dz;
        if (!cpu_collision_free_3d(px, py, pz, obstacles)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// GPU helper class
// ---------------------------------------------------------------------------
struct GpuRRT3DHelper {
    float *d_nodes_x, *d_nodes_y, *d_nodes_z;
    int d_nodes_capacity;

    float *d_obs_cx, *d_obs_cy, *d_obs_cz, *d_obs_r;
    int *d_collision_result;
    int num_obstacles;

    float *d_block_min_dist;
    int   *d_block_min_idx;
    int    max_blocks;

    GpuRRT3DHelper(const std::vector<Obstacle>& obstacles)
        : d_nodes_x(nullptr), d_nodes_y(nullptr), d_nodes_z(nullptr),
          d_nodes_capacity(0),
          d_block_min_dist(nullptr), d_block_min_idx(nullptr), max_blocks(0)
    {
        num_obstacles = (int)obstacles.size();
        std::vector<float> ocx(num_obstacles), ocy(num_obstacles),
                           ocz(num_obstacles), orr(num_obstacles);
        for (int i = 0; i < num_obstacles; i++) {
            ocx[i] = obstacles[i].cx;
            ocy[i] = obstacles[i].cy;
            ocz[i] = obstacles[i].cz;
            orr[i] = obstacles[i].r;
        }
        CUDA_CHECK(cudaMalloc(&d_obs_cx, num_obstacles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_obs_cy, num_obstacles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_obs_cz, num_obstacles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_obs_r,  num_obstacles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_collision_result, num_obstacles * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_obs_cx, ocx.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_obs_cy, ocy.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_obs_cz, ocz.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_obs_r,  orr.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));
        ensureCapacity(1024);
    }

    ~GpuRRT3DHelper() {
        if (d_obs_cx) cudaFree(d_obs_cx);
        if (d_obs_cy) cudaFree(d_obs_cy);
        if (d_obs_cz) cudaFree(d_obs_cz);
        if (d_obs_r)  cudaFree(d_obs_r);
        if (d_collision_result) cudaFree(d_collision_result);
        if (d_nodes_x) cudaFree(d_nodes_x);
        if (d_nodes_y) cudaFree(d_nodes_y);
        if (d_nodes_z) cudaFree(d_nodes_z);
        if (d_block_min_dist) cudaFree(d_block_min_dist);
        if (d_block_min_idx)  cudaFree(d_block_min_idx);
    }

    void ensureCapacity(int needed) {
        if (needed <= d_nodes_capacity) return;
        int new_cap = (d_nodes_capacity == 0) ? 1024 : d_nodes_capacity;
        while (new_cap < needed) new_cap *= 2;

        float *new_x, *new_y, *new_z;
        CUDA_CHECK(cudaMalloc(&new_x, new_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&new_y, new_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&new_z, new_cap * sizeof(float)));
        if (d_nodes_x && d_nodes_capacity > 0) {
            CUDA_CHECK(cudaMemcpy(new_x, d_nodes_x, d_nodes_capacity * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(new_y, d_nodes_y, d_nodes_capacity * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(new_z, d_nodes_z, d_nodes_capacity * sizeof(float), cudaMemcpyDeviceToDevice));
            cudaFree(d_nodes_x);
            cudaFree(d_nodes_y);
            cudaFree(d_nodes_z);
        }
        d_nodes_x = new_x;
        d_nodes_y = new_y;
        d_nodes_z = new_z;
        d_nodes_capacity = new_cap;

        int block_size = 256;
        int new_max_blocks = (new_cap + block_size - 1) / block_size;
        if (new_max_blocks > max_blocks) {
            if (d_block_min_dist) cudaFree(d_block_min_dist);
            if (d_block_min_idx)  cudaFree(d_block_min_idx);
            CUDA_CHECK(cudaMalloc(&d_block_min_dist, new_max_blocks * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_block_min_idx,  new_max_blocks * sizeof(int)));
            max_blocks = new_max_blocks;
        }
    }

    void uploadNode(int idx, float x, float y, float z) {
        ensureCapacity(idx + 1);
        CUDA_CHECK(cudaMemcpy(d_nodes_x + idx, &x, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nodes_y + idx, &y, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nodes_z + idx, &z, sizeof(float), cudaMemcpyHostToDevice));
    }

    int findNearest(int num_nodes, float qx, float qy, float qz) {
        if (num_nodes <= 1) return 0;
        int block_size = 256;
        int num_blocks = (num_nodes + block_size - 1) / block_size;
        size_t smem_size = block_size * (sizeof(float) + sizeof(int));

        find_nearest_3d_kernel<<<num_blocks, block_size, smem_size>>>(
            d_nodes_x, d_nodes_y, d_nodes_z, num_nodes, qx, qy, qz,
            d_block_min_dist, d_block_min_idx);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_dist(num_blocks);
        std::vector<int>   h_idx(num_blocks);
        CUDA_CHECK(cudaMemcpy(h_dist.data(), d_block_min_dist, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_idx.data(),  d_block_min_idx,  num_blocks * sizeof(int),   cudaMemcpyDeviceToHost));

        int best = h_idx[0];
        float best_dist = h_dist[0];
        for (int i = 1; i < num_blocks; i++) {
            if (h_dist[i] < best_dist) { best_dist = h_dist[i]; best = h_idx[i]; }
        }
        return best;
    }

    bool collisionFree(float nx, float ny, float nz) {
        if (num_obstacles == 0) return true;
        int block_size = 256;
        int num_blocks = (num_obstacles + block_size - 1) / block_size;

        collision_check_3d_kernel<<<num_blocks, block_size>>>(
            d_obs_cx, d_obs_cy, d_obs_cz, d_obs_r, num_obstacles,
            nx, ny, nz, d_collision_result);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> h_result(num_obstacles);
        CUDA_CHECK(cudaMemcpy(h_result.data(), d_collision_result, num_obstacles * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < num_obstacles; i++) {
            if (h_result[i]) return false;
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// Visualization helper: world to pixel (XY projection)
// ---------------------------------------------------------------------------
cv::Point toPixel(float wx, float wy, float rand_min, float rand_max, int img_size) {
    float scale = (float)img_size / (rand_max - rand_min);
    return cv::Point((int)((wx - rand_min) * scale),
                     (int)((wy - rand_min) * scale));
}

void draw_static_3d(cv::Mat& img,
                    float sx, float sy,
                    float gx, float gy,
                    const std::vector<Obstacle>& obstacles,
                    float rand_min, float rand_max, int img_size) {
    float scale = (float)img_size / (rand_max - rand_min);

    // Draw obstacles as filled circles (XY projection)
    for (int i = 0; i < (int)obstacles.size(); i++) {
        cv::circle(img, toPixel(obstacles[i].cx, obstacles[i].cy, rand_min, rand_max, img_size),
                   (int)(obstacles[i].r * scale), cv::Scalar(0, 0, 0), -1);
    }

    // Start (red) and goal (blue)
    cv::circle(img, toPixel(sx, sy, rand_min, rand_max, img_size), 8, cv::Scalar(0, 0, 255), -1);
    cv::circle(img, toPixel(gx, gy, rand_min, rand_max, img_size), 8, cv::Scalar(255, 0, 0), -1);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    // Same obstacles/start/goal as rrt_star_3d.cu
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
    int goal_sample_rate = 10;
    int max_iter = 2000;

    unsigned int shared_seed = 42;

    // --- CPU RRT* 3D state ---
    std::vector<Node> cpu_nodes;
    cpu_nodes.push_back(Node(start_x, start_y, start_z));
    std::mt19937 cpu_gen(shared_seed);
    std::uniform_int_distribution<int> cpu_goal_dis(0, 100);
    std::uniform_real_distribution<float> cpu_area_dis(rand_min, rand_max);
    bool cpu_found = false;

    // --- CUDA RRT* 3D state ---
    std::vector<Node> cuda_nodes;
    cuda_nodes.push_back(Node(start_x, start_y, start_z));
    std::mt19937 cuda_gen(shared_seed);
    std::uniform_int_distribution<int> cuda_goal_dis(0, 100);
    std::uniform_real_distribution<float> cuda_area_dis(rand_min, rand_max);
    bool cuda_found = false;

    GpuRRT3DHelper gpu(obstacles);
    gpu.uploadNode(0, start_x, start_y, start_z);

    // --- Visualization: 400x400 per side = 800x400 ---
    int S = 400;
    std::string avi_path = "gif/comparison_rrt3d.avi";
    std::string gif_path = "gif/comparison_rrt3d.gif";

    cv::VideoWriter video(avi_path,
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                          30, cv::Size(S * 2, S));
    if (!video.isOpened()) {
        std::cerr << "Failed to open video writer at " << avi_path << std::endl;
        return 1;
    }

    // Persistent background images for incremental drawing
    cv::Mat bg_cpu(S, S, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat bg_cuda(S, S, CV_8UC3, cv::Scalar(255, 255, 255));

    draw_static_3d(bg_cpu,  start_x, start_y, goal_x, goal_y, obstacles, rand_min, rand_max, S);
    draw_static_3d(bg_cuda, start_x, start_y, goal_x, goal_y, obstacles, rand_min, rand_max, S);

    double cpu_total_ms = 0.0, cuda_total_ms = 0.0;
    int cpu_iter_count = 0, cuda_iter_count = 0;

    std::cout << "RRT* 3D comparison: CPU vs CUDA (XY projection)" << std::endl;

    for (int iter = 0; iter < max_iter; iter++) {
        bool drew_something = false;

        // ============ CPU iteration ============
        if (!cpu_found) {
            float rnd_x, rnd_y, rnd_z;
            if (cpu_goal_dis(cpu_gen) > goal_sample_rate) {
                rnd_x = cpu_area_dis(cpu_gen);
                rnd_y = cpu_area_dis(cpu_gen);
                rnd_z = cpu_area_dis(cpu_gen);
            } else {
                rnd_x = goal_x; rnd_y = goal_y; rnd_z = goal_z;
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            int nearest_idx = cpu_find_nearest_3d(cpu_nodes, rnd_x, rnd_y, rnd_z);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            cpu_total_ms += ms;
            cpu_iter_count++;

            Node& nearest = cpu_nodes[nearest_idx];
            float dx = rnd_x - nearest.x;
            float dy = rnd_y - nearest.y;
            float dz = rnd_z - nearest.z;
            float d = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (d < 1e-6f) continue;
            float scale = expand_dis / d;
            float new_x = nearest.x + dx * scale;
            float new_y = nearest.y + dy * scale;
            float new_z = nearest.z + dz * scale;

            if (cpu_collision_free_3d(new_x, new_y, new_z, obstacles) &&
                cpu_path_collision_free_3d(nearest.x, nearest.y, nearest.z,
                                           new_x, new_y, new_z, obstacles, path_resolution)) {
                int new_idx = (int)cpu_nodes.size();
                cpu_nodes.push_back(Node(new_x, new_y, new_z, nearest_idx));

                cv::line(bg_cpu,
                         toPixel(new_x, new_y, rand_min, rand_max, S),
                         toPixel(nearest.x, nearest.y, rand_min, rand_max, S),
                         cv::Scalar(0, 255, 0), 1);
                drew_something = true;

                float gd = std::sqrt((new_x - goal_x) * (new_x - goal_x) +
                                     (new_y - goal_y) * (new_y - goal_y) +
                                     (new_z - goal_z) * (new_z - goal_z));
                if (gd <= expand_dis) {
                    cpu_found = true;
                    int idx = new_idx;
                    while (idx != -1) {
                        Node& n = cpu_nodes[idx];
                        if (n.parent_idx != -1) {
                            Node& p = cpu_nodes[n.parent_idx];
                            cv::line(bg_cpu,
                                     toPixel(n.x, n.y, rand_min, rand_max, S),
                                     toPixel(p.x, p.y, rand_min, rand_max, S),
                                     cv::Scalar(255, 0, 255), 3);
                        }
                        idx = n.parent_idx;
                    }
                    std::cout << "CPU found path at iter " << iter << std::endl;
                }
            }
        }

        // ============ CUDA iteration ============
        if (!cuda_found) {
            float rnd_x, rnd_y, rnd_z;
            if (cuda_goal_dis(cuda_gen) > goal_sample_rate) {
                rnd_x = cuda_area_dis(cuda_gen);
                rnd_y = cuda_area_dis(cuda_gen);
                rnd_z = cuda_area_dis(cuda_gen);
            } else {
                rnd_x = goal_x; rnd_y = goal_y; rnd_z = goal_z;
            }

            cudaEvent_t ev_start, ev_stop;
            cudaEventCreate(&ev_start);
            cudaEventCreate(&ev_stop);
            cudaEventRecord(ev_start);
            int nearest_idx = gpu.findNearest((int)cuda_nodes.size(), rnd_x, rnd_y, rnd_z);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            float ms;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);
            cuda_total_ms += ms;
            cuda_iter_count++;

            Node& nearest = cuda_nodes[nearest_idx];
            float dx = rnd_x - nearest.x;
            float dy = rnd_y - nearest.y;
            float dz = rnd_z - nearest.z;
            float d = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (d < 1e-6f) continue;
            float sc = expand_dis / d;
            float new_x = nearest.x + dx * sc;
            float new_y = nearest.y + dy * sc;
            float new_z = nearest.z + dz * sc;

            if (gpu.collisionFree(new_x, new_y, new_z)) {
                int new_idx = (int)cuda_nodes.size();
                cuda_nodes.push_back(Node(new_x, new_y, new_z, nearest_idx));
                gpu.uploadNode(new_idx, new_x, new_y, new_z);

                cv::line(bg_cuda,
                         toPixel(new_x, new_y, rand_min, rand_max, S),
                         toPixel(nearest.x, nearest.y, rand_min, rand_max, S),
                         cv::Scalar(0, 255, 0), 1);
                drew_something = true;

                float gd = std::sqrt((new_x - goal_x) * (new_x - goal_x) +
                                     (new_y - goal_y) * (new_y - goal_y) +
                                     (new_z - goal_z) * (new_z - goal_z));
                if (gd <= expand_dis) {
                    cuda_found = true;
                    int idx = new_idx;
                    while (idx != -1) {
                        Node& n = cuda_nodes[idx];
                        if (n.parent_idx != -1) {
                            Node& p = cuda_nodes[n.parent_idx];
                            cv::line(bg_cuda,
                                     toPixel(n.x, n.y, rand_min, rand_max, S),
                                     toPixel(p.x, p.y, rand_min, rand_max, S),
                                     cv::Scalar(255, 0, 255), 3);
                        }
                        idx = n.parent_idx;
                    }
                    std::cout << "CUDA found path at iter " << iter << std::endl;
                }
            }
        }

        // Write frame
        if (drew_something || iter % 5 == 0) {
            cv::Mat left = bg_cpu.clone();
            cv::Mat right = bg_cuda.clone();
            char buf[128];

            // CPU labels
            cv::putText(left, "CPU (linear 3D search)", cv::Point(10, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 2);
            if (cpu_iter_count > 0) {
                snprintf(buf, sizeof(buf), "%.4f ms/iter (avg)", cpu_total_ms / cpu_iter_count);
                cv::putText(left, buf, cv::Point(10, 50),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);
            }
            snprintf(buf, sizeof(buf), "Nodes: %d", (int)cpu_nodes.size());
            cv::putText(left, buf, cv::Point(10, 72),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(100, 100, 100), 1);
            if (cpu_found) {
                cv::putText(left, "PATH FOUND", cv::Point(10, 95),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 180, 0), 2);
            }
            cv::putText(left, "XY Projection", cv::Point(10, S - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(100, 100, 100), 1);

            // CUDA labels
            cv::putText(right, "CUDA (shared mem reduction)", cv::Point(10, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 2);
            if (cuda_iter_count > 0) {
                snprintf(buf, sizeof(buf), "%.4f ms/iter (avg)", cuda_total_ms / cuda_iter_count);
                cv::putText(right, buf, cv::Point(10, 50),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);
            }
            snprintf(buf, sizeof(buf), "Nodes: %d", (int)cuda_nodes.size());
            cv::putText(right, buf, cv::Point(10, 72),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(100, 100, 100), 1);
            if (cuda_found) {
                cv::putText(right, "PATH FOUND", cv::Point(10, 95),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 180, 0), 2);
            }
            cv::putText(right, "XY Projection", cv::Point(10, S - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(100, 100, 100), 1);

            cv::Mat combined;
            cv::hconcat(left, right, combined);
            video.write(combined);
        }

        if (cpu_found && cuda_found) {
            // Hold final frame
            cv::Mat left = bg_cpu.clone();
            cv::Mat right = bg_cuda.clone();
            char buf[128];

            cv::putText(left, "CPU (linear 3D search)", cv::Point(10, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 2);
            snprintf(buf, sizeof(buf), "%.4f ms/iter (avg)", cpu_total_ms / cpu_iter_count);
            cv::putText(left, buf, cv::Point(10, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);
            cv::putText(left, "PATH FOUND", cv::Point(10, 95),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 180, 0), 2);

            cv::putText(right, "CUDA (shared mem reduction)", cv::Point(10, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 2);
            snprintf(buf, sizeof(buf), "%.4f ms/iter (avg)", cuda_total_ms / cuda_iter_count);
            cv::putText(right, buf, cv::Point(10, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);
            cv::putText(right, "PATH FOUND", cv::Point(10, 95),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 180, 0), 2);

            cv::Mat combined;
            cv::hconcat(left, right, combined);
            for (int f = 0; f < 60; f++) video.write(combined);
            break;
        }
    }

    video.release();
    std::cout << "Video saved to " << avi_path << std::endl;

    // Convert to GIF
    std::string cmd = "ffmpeg -y -i " + avi_path +
        " -vf 'fps=15,scale=800:-1:flags=lanczos' -loop 0 " + gif_path + " 2>/dev/null";
    int ret = system(cmd.c_str());
    if (ret == 0) {
        std::cout << "GIF saved to " << gif_path << std::endl;
    } else {
        std::cout << "ffmpeg conversion failed (ffmpeg may not be installed)" << std::endl;
    }

    if (cpu_iter_count > 0)
        printf("CPU  avg nearest-neighbor: %.4f ms/iter (%d iters)\n",
               cpu_total_ms / cpu_iter_count, cpu_iter_count);
    if (cuda_iter_count > 0)
        printf("CUDA avg nearest-neighbor: %.4f ms/iter (%d iters)\n",
               cuda_total_ms / cuda_iter_count, cuda_iter_count);

    return 0;
}
