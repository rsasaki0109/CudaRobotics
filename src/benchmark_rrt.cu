/*************************************************************************
    Benchmark: RRT Nearest Neighbor Search - CPU vs CUDA
    Measures nearest neighbor query time (the RRT bottleneck)
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ============================================================
// CUDA Kernel: shared memory block reduction for nearest neighbor
// ============================================================

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

// ============================================================
// CPU: linear scan nearest neighbor
// ============================================================

int cpu_find_nearest(const float* nodes_x, const float* nodes_y,
                     int num_nodes, float qx, float qy)
{
    float best_dist = FLT_MAX;
    int best_idx = -1;
    for (int i = 0; i < num_nodes; i++) {
        float dx = nodes_x[i] - qx;
        float dy = nodes_y[i] - qy;
        float d2 = dx * dx + dy * dy;
        if (d2 < best_dist) {
            best_dist = d2;
            best_idx = i;
        }
    }
    return best_idx;
}

// ============================================================
// CUDA nearest neighbor (host wrapper)
// ============================================================

int cuda_find_nearest(const float* d_nodes_x, const float* d_nodes_y,
                      int num_nodes, float qx, float qy,
                      float* d_block_min_dist, int* d_block_min_idx,
                      float* h_block_dist, int* h_block_idx,
                      int num_blocks)
{
    int block_size = 256;
    size_t smem_size = block_size * (sizeof(float) + sizeof(int));

    find_nearest_kernel<<<num_blocks, block_size, smem_size>>>(
        d_nodes_x, d_nodes_y, num_nodes, qx, qy,
        d_block_min_dist, d_block_min_idx);

    CUDA_CHECK(cudaMemcpy(h_block_dist, d_block_min_dist,
                          num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block_idx, d_block_min_idx,
                          num_blocks * sizeof(int), cudaMemcpyDeviceToHost));

    int best_idx = h_block_idx[0];
    float best_dist = h_block_dist[0];
    for (int i = 1; i < num_blocks; i++) {
        if (h_block_dist[i] < best_dist) {
            best_dist = h_block_dist[i];
            best_idx = h_block_idx[i];
        }
    }
    return best_idx;
}

// ============================================================
// Benchmark runner
// ============================================================

struct BenchResult {
    double total_ms;
    int num_queries;
};

BenchResult run_cpu_bench(int tree_size, int num_queries,
                          const float* nodes_x, const float* nodes_y,
                          const float* queries_x, const float* queries_y)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    volatile int sink = 0;  // prevent optimization
    for (int q = 0; q < num_queries; q++) {
        int idx = cpu_find_nearest(nodes_x, nodes_y, tree_size,
                                   queries_x[q], queries_y[q]);
        sink = idx;
    }
    (void)sink;

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {ms, num_queries};
}

BenchResult run_cuda_bench(int tree_size, int num_queries,
                           const float* nodes_x, const float* nodes_y,
                           const float* queries_x, const float* queries_y)
{
    int block_size = 256;
    int num_blocks = (tree_size + block_size - 1) / block_size;

    // Allocate device memory
    float *d_nodes_x, *d_nodes_y;
    float *d_block_min_dist;
    int   *d_block_min_idx;

    CUDA_CHECK(cudaMalloc(&d_nodes_x, tree_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nodes_y, tree_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_min_dist, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_min_idx, num_blocks * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_nodes_x, nodes_x, tree_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes_y, nodes_y, tree_size * sizeof(float), cudaMemcpyHostToDevice));

    // Host buffers for block results
    std::vector<float> h_block_dist(num_blocks);
    std::vector<int>   h_block_idx(num_blocks);

    // Warmup
    cuda_find_nearest(d_nodes_x, d_nodes_y, tree_size,
                      queries_x[0], queries_y[0],
                      d_block_min_dist, d_block_min_idx,
                      h_block_dist.data(), h_block_idx.data(),
                      num_blocks);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int q = 0; q < num_queries; q++) {
        cuda_find_nearest(d_nodes_x, d_nodes_y, tree_size,
                          queries_x[q], queries_y[q],
                          d_block_min_dist, d_block_min_idx,
                          h_block_dist.data(), h_block_idx.data(),
                          num_blocks);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cudaFree(d_nodes_x);
    cudaFree(d_nodes_y);
    cudaFree(d_block_min_dist);
    cudaFree(d_block_min_idx);

    return {ms, num_queries};
}

// ============================================================
// Validation: verify CPU and CUDA produce the same result
// ============================================================

bool validate(int tree_size, const float* nodes_x, const float* nodes_y,
              const float* queries_x, const float* queries_y, int num_check)
{
    int block_size = 256;
    int num_blocks = (tree_size + block_size - 1) / block_size;

    float *d_nodes_x, *d_nodes_y;
    float *d_block_min_dist;
    int   *d_block_min_idx;

    CUDA_CHECK(cudaMalloc(&d_nodes_x, tree_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nodes_y, tree_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_min_dist, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_min_idx, num_blocks * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_nodes_x, nodes_x, tree_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes_y, nodes_y, tree_size * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_block_dist(num_blocks);
    std::vector<int>   h_block_idx(num_blocks);

    bool ok = true;
    for (int q = 0; q < num_check; q++) {
        int cpu_idx = cpu_find_nearest(nodes_x, nodes_y, tree_size,
                                       queries_x[q], queries_y[q]);
        int cuda_idx = cuda_find_nearest(d_nodes_x, d_nodes_y, tree_size,
                                         queries_x[q], queries_y[q],
                                         d_block_min_dist, d_block_min_idx,
                                         h_block_dist.data(), h_block_idx.data(),
                                         num_blocks);
        if (cpu_idx != cuda_idx) {
            fprintf(stderr, "  MISMATCH query %d: CPU=%d CUDA=%d\n", q, cpu_idx, cuda_idx);
            ok = false;
        }
    }

    cudaFree(d_nodes_x);
    cudaFree(d_nodes_y);
    cudaFree(d_block_min_dist);
    cudaFree(d_block_min_idx);
    return ok;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  RRT Nearest Neighbor Benchmark: CPU vs CUDA" << std::endl;
    std::cout << "================================================" << std::endl;

    // Warmup CUDA context
    { float *tmp; cudaMalloc(&tmp, 1024); cudaFree(tmp); cudaDeviceSynchronize(); }

    const int num_queries = 1000;
    const int tree_sizes[] = {100, 1000, 5000, 10000, 50000};
    const int num_sizes = sizeof(tree_sizes) / sizeof(tree_sizes[0]);

    // Find max tree size for allocation
    int max_tree = 0;
    for (int i = 0; i < num_sizes; i++) {
        if (tree_sizes[i] > max_tree) max_tree = tree_sizes[i];
    }

    // Generate random tree nodes in [0,100]x[0,100]
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    std::vector<float> all_nodes_x(max_tree);
    std::vector<float> all_nodes_y(max_tree);
    for (int i = 0; i < max_tree; i++) {
        all_nodes_x[i] = dist(gen);
        all_nodes_y[i] = dist(gen);
    }

    // Generate random query points
    std::vector<float> queries_x(num_queries);
    std::vector<float> queries_y(num_queries);
    for (int i = 0; i < num_queries; i++) {
        queries_x[i] = dist(gen);
        queries_y[i] = dist(gen);
    }

    // Validation
    std::cout << "\n  Validating CPU vs CUDA results..." << std::endl;
    bool valid = validate(max_tree, all_nodes_x.data(), all_nodes_y.data(),
                          queries_x.data(), queries_y.data(),
                          std::min(num_queries, 100));
    if (valid) {
        std::cout << "  Validation PASSED" << std::endl;
    } else {
        std::cout << "  Validation FAILED" << std::endl;
        return 1;
    }

    // Benchmark
    for (int s = 0; s < num_sizes; s++) {
        int tree_size = tree_sizes[s];

        auto cpu = run_cpu_bench(tree_size, num_queries,
                                 all_nodes_x.data(), all_nodes_y.data(),
                                 queries_x.data(), queries_y.data());

        auto cuda = run_cuda_bench(tree_size, num_queries,
                                   all_nodes_x.data(), all_nodes_y.data(),
                                   queries_x.data(), queries_y.data());

        printf("\n  Tree = %d nodes  (%d queries)\n", tree_size, num_queries);
        printf("    CPU:  %8.2f ms\n", cpu.total_ms);
        printf("    CUDA: %8.2f ms\n", cuda.total_ms);
        printf("    Speedup: %.2fx\n", cpu.total_ms / cuda.total_ms);
    }

    std::cout << "\n================================================" << std::endl;
    return 0;
}
