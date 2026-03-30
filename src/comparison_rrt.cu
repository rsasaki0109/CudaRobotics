/*************************************************************************
    RRT: CPU vs CUDA side-by-side comparison GIF generator
    Left panel: CPU (linear nearest neighbor), Right panel: CUDA (GPU parallel)
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// CUDA kernel: parallel nearest neighbor with shared memory reduction
// ---------------------------------------------------------------------------
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

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_dist[tid + stride] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + stride];
                s_idx[tid]  = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_min_dist[blockIdx.x] = s_dist[0];
        block_min_idx[blockIdx.x]  = s_idx[0];
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel: parallel collision check
// ---------------------------------------------------------------------------
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
// Node structure
// ---------------------------------------------------------------------------
struct Node {
    float x, y;
    int parent_idx;
    Node(float x_, float y_, int p = -1) : x(x_), y(y_), parent_idx(p) {}
};

// ---------------------------------------------------------------------------
// CPU: linear nearest neighbor search
// ---------------------------------------------------------------------------
int cpu_find_nearest(const std::vector<Node>& nodes, float qx, float qy) {
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

// CPU: linear collision check
bool cpu_collision_free(float nx, float ny,
                        const std::vector<float>& ob_x,
                        const std::vector<float>& ob_y,
                        const std::vector<float>& ob_r) {
    for (int i = 0; i < (int)ob_x.size(); i++) {
        float dx = ob_x[i] - nx;
        float dy = ob_y[i] - ny;
        if (std::sqrt(dx * dx + dy * dy) <= ob_r[i]) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// GPU helper class for nearest neighbor + collision
// ---------------------------------------------------------------------------
struct GpuRRTHelper {
    float *d_nodes_x, *d_nodes_y;
    int d_nodes_capacity;

    float *d_ob_x, *d_ob_y, *d_ob_r;
    int *d_collision_result;
    int num_obstacles;

    float *d_block_min_dist;
    int   *d_block_min_idx;
    int    max_blocks;

    GpuRRTHelper(const std::vector<float>& ob_x,
                 const std::vector<float>& ob_y,
                 const std::vector<float>& ob_r)
        : d_nodes_x(nullptr), d_nodes_y(nullptr), d_nodes_capacity(0),
          d_block_min_dist(nullptr), d_block_min_idx(nullptr), max_blocks(0)
    {
        num_obstacles = (int)ob_x.size();
        CUDA_CHECK(cudaMalloc(&d_ob_x, num_obstacles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ob_y, num_obstacles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ob_r, num_obstacles * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_collision_result, num_obstacles * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_ob_x, ob_x.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ob_y, ob_y.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ob_r, ob_r.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));
        ensureCapacity(1024);
    }

    ~GpuRRTHelper() {
        if (d_ob_x) cudaFree(d_ob_x);
        if (d_ob_y) cudaFree(d_ob_y);
        if (d_ob_r) cudaFree(d_ob_r);
        if (d_collision_result) cudaFree(d_collision_result);
        if (d_nodes_x) cudaFree(d_nodes_x);
        if (d_nodes_y) cudaFree(d_nodes_y);
        if (d_block_min_dist) cudaFree(d_block_min_dist);
        if (d_block_min_idx) cudaFree(d_block_min_idx);
    }

    void ensureCapacity(int needed) {
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

    void uploadNode(int idx, float x, float y) {
        ensureCapacity(idx + 1);
        CUDA_CHECK(cudaMemcpy(d_nodes_x + idx, &x, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nodes_y + idx, &y, sizeof(float), cudaMemcpyHostToDevice));
    }

    int findNearest(int num_nodes, float qx, float qy) {
        if (num_nodes <= 1) return 0;
        int block_size = 256;
        int num_blocks = (num_nodes + block_size - 1) / block_size;
        size_t smem_size = block_size * (sizeof(float) + sizeof(int));

        find_nearest_kernel<<<num_blocks, block_size, smem_size>>>(
            d_nodes_x, d_nodes_y, num_nodes, qx, qy, d_block_min_dist, d_block_min_idx);
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

    bool collisionFree(float nx, float ny) {
        if (num_obstacles == 0) return true;
        int block_size = 256;
        int num_blocks = (num_obstacles + block_size - 1) / block_size;

        collision_check_kernel<<<num_blocks, block_size>>>(
            d_ob_x, d_ob_y, d_ob_r, num_obstacles, nx, ny, d_collision_result);
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
// Visualization helper
// ---------------------------------------------------------------------------
cv::Point toPixel(float wx, float wy, float rand_min, int img_reso) {
    return cv::Point((int)((wx - rand_min) * img_reso),
                     (int)((wy - rand_min) * img_reso));
}

void draw_static(cv::Mat& img,
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
// Main
// ---------------------------------------------------------------------------
int main() {
    // Problem setup (same as rrt.cu)
    std::vector<std::vector<float>> obstacle_list{
        {5, 5, 1}, {3, 6, 2}, {3, 8, 2}, {3, 10, 2}, {7, 5, 2}, {9, 5, 2}
    };

    float start_x = 0.0f, start_y = 0.0f;
    float goal_x  = 6.0f, goal_y  = 9.0f;
    float rand_min = -2.0f, rand_max = 15.0f;
    float expand_dis = 0.5f;
    int goal_sample_rate = 5;
    int max_iter = 5000;

    int num_obstacles = (int)obstacle_list.size();
    std::vector<float> ob_x(num_obstacles), ob_y(num_obstacles), ob_r(num_obstacles);
    for (int i = 0; i < num_obstacles; i++) {
        ob_x[i] = obstacle_list[i][0];
        ob_y[i] = obstacle_list[i][1];
        ob_r[i] = obstacle_list[i][2];
    }

    // Shared random sequence (same seed for both)
    unsigned int shared_seed = 42;

    // CPU RRT state
    std::vector<Node> cpu_nodes;
    cpu_nodes.push_back(Node(start_x, start_y));
    std::mt19937 cpu_gen(shared_seed);
    std::uniform_int_distribution<int> cpu_goal_dis(0, 100);
    std::uniform_real_distribution<float> cpu_area_dis(rand_min, rand_max);
    bool cpu_found = false;

    // CUDA RRT state
    std::vector<Node> cuda_nodes;
    cuda_nodes.push_back(Node(start_x, start_y));
    std::mt19937 cuda_gen(shared_seed);
    std::uniform_int_distribution<int> cuda_goal_dis(0, 100);
    std::uniform_real_distribution<float> cuda_area_dis(rand_min, rand_max);
    bool cuda_found = false;

    GpuRRTHelper gpu(ob_x, ob_y, ob_r);
    gpu.uploadNode(0, start_x, start_y);

    // Visualization
    int img_size = (int)(rand_max - rand_min);
    int img_reso = 50;
    int S = img_size * img_reso; // 850
    cv::VideoWriter video("gif/comparison_rrt.avi",
                          cv::VideoWriter::fourcc('X','V','I','D'), 30, cv::Size(S * 2, S));

    // Persistent background images for incremental drawing
    cv::Mat bg_cpu(S, S, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat bg_cuda(S, S, CV_8UC3, cv::Scalar(255, 255, 255));

    draw_static(bg_cpu,  start_x, start_y, goal_x, goal_y, ob_x, ob_y, ob_r, rand_min, img_reso);
    draw_static(bg_cuda, start_x, start_y, goal_x, goal_y, ob_x, ob_y, ob_r, rand_min, img_reso);

    // Running average timing
    double cpu_avg_ms = 0.0, cuda_avg_ms = 0.0;
    int cpu_iter_count = 0, cuda_iter_count = 0;

    std::cout << "RRT comparison: CPU vs CUDA" << std::endl;

    for (int iter = 0; iter < max_iter; iter++) {
        bool drew_something = false;

        // ============ CPU iteration ============
        if (!cpu_found) {
            float rnd_x, rnd_y;
            if (cpu_goal_dis(cpu_gen) > goal_sample_rate) {
                rnd_x = cpu_area_dis(cpu_gen);
                rnd_y = cpu_area_dis(cpu_gen);
            } else {
                rnd_x = goal_x;
                rnd_y = goal_y;
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            int nearest_idx = cpu_find_nearest(cpu_nodes, rnd_x, rnd_y);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            cpu_avg_ms += ms;
            cpu_iter_count++;

            Node& nearest = cpu_nodes[nearest_idx];
            float theta = std::atan2(rnd_y - nearest.y, rnd_x - nearest.x);
            float new_x = nearest.x + expand_dis * std::cos(theta);
            float new_y = nearest.y + expand_dis * std::sin(theta);

            if (cpu_collision_free(new_x, new_y, ob_x, ob_y, ob_r)) {
                int new_idx = (int)cpu_nodes.size();
                cpu_nodes.push_back(Node(new_x, new_y, nearest_idx));

                cv::line(bg_cpu, toPixel(new_x, new_y, rand_min, img_reso),
                         toPixel(nearest.x, nearest.y, rand_min, img_reso),
                         cv::Scalar(0, 255, 0), 5);
                drew_something = true;

                float dx = new_x - goal_x;
                float dy = new_y - goal_y;
                if (std::sqrt(dx * dx + dy * dy) <= expand_dis) {
                    cpu_found = true;
                    // Draw path
                    int idx = new_idx;
                    while (idx != -1) {
                        Node& n = cpu_nodes[idx];
                        if (n.parent_idx != -1) {
                            Node& p = cpu_nodes[n.parent_idx];
                            cv::line(bg_cpu, toPixel(n.x, n.y, rand_min, img_reso),
                                     toPixel(p.x, p.y, rand_min, img_reso),
                                     cv::Scalar(255, 0, 255), 5);
                        }
                        idx = n.parent_idx;
                    }
                    std::cout << "CPU found path at iter " << iter << std::endl;
                }
            }
        }

        // ============ CUDA iteration ============
        if (!cuda_found) {
            float rnd_x, rnd_y;
            if (cuda_goal_dis(cuda_gen) > goal_sample_rate) {
                rnd_x = cuda_area_dis(cuda_gen);
                rnd_y = cuda_area_dis(cuda_gen);
            } else {
                rnd_x = goal_x;
                rnd_y = goal_y;
            }

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            int nearest_idx = gpu.findNearest((int)cuda_nodes.size(), rnd_x, rnd_y);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cuda_avg_ms += ms;
            cuda_iter_count++;

            Node& nearest = cuda_nodes[nearest_idx];
            float theta = std::atan2(rnd_y - nearest.y, rnd_x - nearest.x);
            float new_x = nearest.x + expand_dis * std::cos(theta);
            float new_y = nearest.y + expand_dis * std::sin(theta);

            if (gpu.collisionFree(new_x, new_y)) {
                int new_idx = (int)cuda_nodes.size();
                cuda_nodes.push_back(Node(new_x, new_y, nearest_idx));
                gpu.uploadNode(new_idx, new_x, new_y);

                cv::line(bg_cuda, toPixel(new_x, new_y, rand_min, img_reso),
                         toPixel(nearest.x, nearest.y, rand_min, img_reso),
                         cv::Scalar(0, 255, 0), 5);
                drew_something = true;

                float dx = new_x - goal_x;
                float dy = new_y - goal_y;
                if (std::sqrt(dx * dx + dy * dy) <= expand_dis) {
                    cuda_found = true;
                    int idx = new_idx;
                    while (idx != -1) {
                        Node& n = cuda_nodes[idx];
                        if (n.parent_idx != -1) {
                            Node& p = cuda_nodes[n.parent_idx];
                            cv::line(bg_cuda, toPixel(n.x, n.y, rand_min, img_reso),
                                     toPixel(p.x, p.y, rand_min, img_reso),
                                     cv::Scalar(255, 0, 255), 5);
                        }
                        idx = n.parent_idx;
                    }
                    std::cout << "CUDA found path at iter " << iter << std::endl;
                }
            }
        }

        // Write frame (every iteration that added a node, or every 5th iter)
        if (drew_something || iter % 5 == 0) {
            // Add timing labels to copies
            cv::Mat left = bg_cpu.clone();
            cv::Mat right = bg_cuda.clone();

            char buf[128];

            // CPU label
            cv::putText(left, "CPU (linear search)", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
            if (cpu_iter_count > 0) {
                snprintf(buf, sizeof(buf), "%.3f ms/iter (avg)", cpu_avg_ms / cpu_iter_count);
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
                snprintf(buf, sizeof(buf), "%.3f ms/iter (avg)", cuda_avg_ms / cuda_iter_count);
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
            // Write a few extra frames so the final path is visible
            cv::Mat left = bg_cpu.clone();
            cv::Mat right = bg_cuda.clone();
            char buf[128];

            cv::putText(left, "CPU (linear search)", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
            snprintf(buf, sizeof(buf), "%.3f ms/iter (avg)", cpu_avg_ms / cpu_iter_count);
            cv::putText(left, buf, cv::Point(20, 80),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);
            cv::putText(left, "PATH FOUND", cv::Point(20, 150),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 180, 0), 2);

            cv::putText(right, "CUDA (GPU parallel)", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
            snprintf(buf, sizeof(buf), "%.3f ms/iter (avg)", cuda_avg_ms / cuda_iter_count);
            cv::putText(right, buf, cv::Point(20, 80),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 200), 2);
            cv::putText(right, "PATH FOUND", cv::Point(20, 150),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 180, 0), 2);

            cv::Mat combined;
            cv::hconcat(left, right, combined);
            for (int f = 0; f < 60; f++) video.write(combined); // 2s hold
            break;
        }
    }

    video.release();
    std::cout << "Video saved to gif/comparison_rrt.avi" << std::endl;

    // Convert to gif
    system("ffmpeg -y -i gif/comparison_rrt.avi "
           "-vf 'fps=15,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_rrt.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_rrt.gif" << std::endl;

    return 0;
}
