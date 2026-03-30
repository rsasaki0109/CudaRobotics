/*************************************************************************
    PRM: CPU vs CUDA side-by-side comparison GIF generator
    Left panel:  CPU (sequential sample collision, k-NN, edge collision)
    Right panel: CUDA (collision_check_samples_kernel + find_knn_kernel + edge_collision_check_kernel)
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <queue>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const int   N_SAMPLE         = 500;
static const int   N_KNN            = 10;
static const float MAX_EDGE_LEN     = 30.0f;
static const float ROBOT_SIZE       = 5.0f;
static const int   EDGE_CHECK_STEPS = 100;

// ---------------------------------------------------------------------------
// CUDA Kernels (from prm.cu)
// ---------------------------------------------------------------------------

__global__ void collision_check_samples_kernel(
    const float* __restrict__ sample_x,
    const float* __restrict__ sample_y,
    int num_samples,
    const float* __restrict__ ob_x,
    const float* __restrict__ ob_y,
    const float* __restrict__ ob_r,
    int num_obstacles,
    float robot_size,
    int* result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;

    float sx = sample_x[idx];
    float sy = sample_y[idx];
    int collides = 0;

    for (int i = 0; i < num_obstacles; i++) {
        float dx = sx - ob_x[i];
        float dy = sy - ob_y[i];
        float dist = sqrtf(dx * dx + dy * dy);
        if (dist <= ob_r[i] + robot_size) {
            collides = 1;
            break;
        }
    }
    result[idx] = collides;
}

__global__ void find_knn_kernel(
    const float* __restrict__ sample_x,
    const float* __restrict__ sample_y,
    int num_samples,
    int k,
    float max_edge_len,
    int* knn_indices,
    float* knn_dists,
    int* knn_counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;

    float mx = sample_x[idx];
    float my = sample_y[idx];

    float top_dist[10];
    int   top_idx[10];

    for (int i = 0; i < k; i++) {
        top_dist[i] = FLT_MAX;
        top_idx[i] = -1;
    }

    for (int j = 0; j < num_samples; j++) {
        if (j == idx) continue;
        float dx = mx - sample_x[j];
        float dy = my - sample_y[j];
        float d = sqrtf(dx * dx + dy * dy);
        if (d > max_edge_len) continue;

        int worst_pos = 0;
        float worst_dist = top_dist[0];
        for (int p = 1; p < k; p++) {
            if (top_dist[p] > worst_dist) {
                worst_dist = top_dist[p];
                worst_pos = p;
            }
        }

        if (d < worst_dist) {
            top_dist[worst_pos] = d;
            top_idx[worst_pos] = j;
        }
    }

    int count = 0;
    for (int i = 0; i < k; i++) {
        knn_indices[idx * k + i] = top_idx[i];
        knn_dists[idx * k + i] = top_dist[i];
        if (top_idx[i] >= 0) count++;
    }
    knn_counts[idx] = count;
}

__global__ void edge_collision_check_kernel(
    const float* __restrict__ edge_x1,
    const float* __restrict__ edge_y1,
    const float* __restrict__ edge_x2,
    const float* __restrict__ edge_y2,
    int num_edges,
    const float* __restrict__ ob_x,
    const float* __restrict__ ob_y,
    const float* __restrict__ ob_r,
    int num_obstacles,
    float robot_size,
    int num_steps,
    int* result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;

    float x1 = edge_x1[idx];
    float y1 = edge_y1[idx];
    float x2 = edge_x2[idx];
    float y2 = edge_y2[idx];

    int collides = 0;

    for (int s = 0; s <= num_steps; s++) {
        float t = (float)s / (float)num_steps;
        float px = x1 + t * (x2 - x1);
        float py = y1 + t * (y2 - y1);

        for (int i = 0; i < num_obstacles; i++) {
            float dx = px - ob_x[i];
            float dy = py - ob_y[i];
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist <= ob_r[i] + robot_size) {
                collides = 1;
                break;
            }
        }
        if (collides) break;
    }
    result[idx] = collides;
}

// ---------------------------------------------------------------------------
// Edge / adjacency structures
// ---------------------------------------------------------------------------
struct Edge {
    int from, to;
    float cost;
};

// ---------------------------------------------------------------------------
// CPU PRM functions
// ---------------------------------------------------------------------------

static void cpu_collision_check_samples(
    const std::vector<float>& sx, const std::vector<float>& sy,
    const std::vector<float>& ob_x, const std::vector<float>& ob_y,
    const std::vector<float>& ob_r, float robot_size,
    std::vector<int>& result)
{
    int n = (int)sx.size();
    result.resize(n);
    for (int i = 0; i < n; i++) {
        result[i] = 0;
        for (int j = 0; j < (int)ob_x.size(); j++) {
            float dx = sx[i] - ob_x[j];
            float dy = sy[i] - ob_y[j];
            if (std::sqrt(dx * dx + dy * dy) <= ob_r[j] + robot_size) {
                result[i] = 1;
                break;
            }
        }
    }
}

static void cpu_find_knn(
    const std::vector<float>& nx, const std::vector<float>& ny,
    int k, float max_edge_len,
    std::vector<int>& knn_indices, std::vector<float>& knn_dists, std::vector<int>& knn_counts)
{
    int n = (int)nx.size();
    knn_indices.resize(n * k, -1);
    knn_dists.resize(n * k, FLT_MAX);
    knn_counts.resize(n, 0);

    for (int i = 0; i < n; i++) {
        float top_dist[10];
        int   top_idx[10];
        for (int p = 0; p < k; p++) { top_dist[p] = FLT_MAX; top_idx[p] = -1; }

        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            float dx = nx[i] - nx[j];
            float dy = ny[i] - ny[j];
            float d = std::sqrt(dx * dx + dy * dy);
            if (d > max_edge_len) continue;

            int worst_pos = 0;
            float worst_d = top_dist[0];
            for (int p = 1; p < k; p++) {
                if (top_dist[p] > worst_d) { worst_d = top_dist[p]; worst_pos = p; }
            }
            if (d < worst_d) {
                top_dist[worst_pos] = d;
                top_idx[worst_pos] = j;
            }
        }

        int count = 0;
        for (int p = 0; p < k; p++) {
            knn_indices[i * k + p] = top_idx[p];
            knn_dists[i * k + p] = top_dist[p];
            if (top_idx[p] >= 0) count++;
        }
        knn_counts[i] = count;
    }
}

static void cpu_edge_collision_check(
    const std::vector<float>& ex1, const std::vector<float>& ey1,
    const std::vector<float>& ex2, const std::vector<float>& ey2,
    const std::vector<float>& ob_x, const std::vector<float>& ob_y,
    const std::vector<float>& ob_r, float robot_size, int num_steps,
    std::vector<int>& result)
{
    int n = (int)ex1.size();
    result.resize(n);
    for (int i = 0; i < n; i++) {
        result[i] = 0;
        for (int s = 0; s <= num_steps; s++) {
            float t = (float)s / (float)num_steps;
            float px = ex1[i] + t * (ex2[i] - ex1[i]);
            float py = ey1[i] + t * (ey2[i] - ey1[i]);
            for (int j = 0; j < (int)ob_x.size(); j++) {
                float dx = px - ob_x[j];
                float dy = py - ob_y[j];
                if (std::sqrt(dx * dx + dy * dy) <= ob_r[j] + robot_size) {
                    result[i] = 1;
                    break;
                }
            }
            if (result[i]) break;
        }
    }
}

// ---------------------------------------------------------------------------
// Dijkstra (shared by both)
// ---------------------------------------------------------------------------
static std::vector<std::pair<float,float>> dijkstra(
    const std::vector<float>& node_x, const std::vector<float>& node_y,
    const std::vector<std::vector<Edge>>& adjacency,
    int start_id, int goal_id)
{
    int n = (int)node_x.size();
    std::vector<float> dist(n, FLT_MAX);
    std::vector<int> prev(n, -1);
    std::vector<bool> visited(n, false);

    typedef std::pair<float,int> PQE;
    std::priority_queue<PQE, std::vector<PQE>, std::greater<PQE>> pq;

    dist[start_id] = 0.0f;
    pq.push({0.0f, start_id});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (visited[u]) continue;
        visited[u] = true;
        if (u == goal_id) break;

        for (const auto& e : adjacency[u]) {
            float nd = dist[u] + e.cost;
            if (nd < dist[e.to]) {
                dist[e.to] = nd;
                prev[e.to] = u;
                pq.push({nd, e.to});
            }
        }
    }

    std::vector<std::pair<float,float>> path;
    if (!visited[goal_id]) return path;
    int cur = goal_id;
    while (cur != -1) {
        path.push_back({node_x[cur], node_y[cur]});
        cur = prev[cur];
    }
    std::reverse(path.begin(), path.end());
    return path;
}

// ---------------------------------------------------------------------------
// PRM builder: returns nodes, adjacency, path, and build time
// ---------------------------------------------------------------------------

struct PRMResult {
    std::vector<float> node_x, node_y;
    std::vector<std::vector<Edge>> adjacency;
    std::vector<std::pair<float,float>> path;
    double build_ms;
};

// CPU PRM
static PRMResult build_prm_cpu(
    float sx, float sy, float gx, float gy,
    const std::vector<float>& ob_x, const std::vector<float>& ob_y,
    const std::vector<float>& ob_r,
    float area_min_x, float area_max_x, float area_min_y, float area_max_y)
{
    PRMResult res;

    auto t0 = std::chrono::high_resolution_clock::now();

    // Generate samples
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dx(area_min_x, area_max_x);
    std::uniform_real_distribution<float> dy(area_min_y, area_max_y);

    res.node_x.push_back(sx);
    res.node_y.push_back(sy);
    res.node_x.push_back(gx);
    res.node_y.push_back(gy);

    std::vector<float> cand_x(N_SAMPLE), cand_y(N_SAMPLE);
    for (int i = 0; i < N_SAMPLE; i++) {
        cand_x[i] = dx(gen);
        cand_y[i] = dy(gen);
    }

    // CPU collision check on samples
    std::vector<int> col_result;
    cpu_collision_check_samples(cand_x, cand_y, ob_x, ob_y, ob_r, ROBOT_SIZE, col_result);
    for (int i = 0; i < N_SAMPLE; i++) {
        if (col_result[i] == 0) {
            res.node_x.push_back(cand_x[i]);
            res.node_y.push_back(cand_y[i]);
        }
    }

    int num_nodes = (int)res.node_x.size();
    res.adjacency.resize(num_nodes);

    // CPU k-NN
    std::vector<int> knn_indices;
    std::vector<float> knn_dists;
    std::vector<int> knn_counts;
    cpu_find_knn(res.node_x, res.node_y, N_KNN, MAX_EDGE_LEN, knn_indices, knn_dists, knn_counts);

    // Collect candidate edges
    struct CandEdge { int from, to; float dist; };
    std::vector<CandEdge> candidates;
    for (int i = 0; i < num_nodes; i++) {
        for (int k = 0; k < N_KNN; k++) {
            int j = knn_indices[i * N_KNN + k];
            if (j < 0) continue;
            candidates.push_back({i, j, knn_dists[i * N_KNN + k]});
        }
    }

    int num_edges = (int)candidates.size();
    if (num_edges > 0) {
        std::vector<float> ex1(num_edges), ey1(num_edges), ex2(num_edges), ey2(num_edges);
        for (int i = 0; i < num_edges; i++) {
            ex1[i] = res.node_x[candidates[i].from];
            ey1[i] = res.node_y[candidates[i].from];
            ex2[i] = res.node_x[candidates[i].to];
            ey2[i] = res.node_y[candidates[i].to];
        }

        std::vector<int> edge_result;
        cpu_edge_collision_check(ex1, ey1, ex2, ey2, ob_x, ob_y, ob_r,
                                 ROBOT_SIZE, EDGE_CHECK_STEPS, edge_result);

        for (int i = 0; i < num_edges; i++) {
            if (edge_result[i] == 0) {
                int f = candidates[i].from;
                int t = candidates[i].to;
                float c = candidates[i].dist;
                res.adjacency[f].push_back({f, t, c});
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Dijkstra
    res.path = dijkstra(res.node_x, res.node_y, res.adjacency, 0, 1);
    return res;
}

// CUDA PRM
static PRMResult build_prm_cuda(
    float sx, float sy, float gx, float gy,
    const std::vector<float>& ob_x, const std::vector<float>& ob_y,
    const std::vector<float>& ob_r,
    float area_min_x, float area_max_x, float area_min_y, float area_max_y)
{
    PRMResult res;

    int num_obstacles = (int)ob_x.size();

    // Upload obstacles
    float *d_ob_x, *d_ob_y, *d_ob_r;
    CUDA_CHECK(cudaMalloc(&d_ob_x, num_obstacles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ob_y, num_obstacles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ob_r, num_obstacles * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ob_x, ob_x.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ob_y, ob_y.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ob_r, ob_r.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    // Generate samples (same seed)
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dx(area_min_x, area_max_x);
    std::uniform_real_distribution<float> dy(area_min_y, area_max_y);

    res.node_x.push_back(sx);
    res.node_y.push_back(sy);
    res.node_x.push_back(gx);
    res.node_y.push_back(gy);

    std::vector<float> cand_x(N_SAMPLE), cand_y(N_SAMPLE);
    for (int i = 0; i < N_SAMPLE; i++) {
        cand_x[i] = dx(gen);
        cand_y[i] = dy(gen);
    }

    // GPU collision check on samples
    float *d_cand_x, *d_cand_y;
    int *d_result;
    CUDA_CHECK(cudaMalloc(&d_cand_x, N_SAMPLE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cand_y, N_SAMPLE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, N_SAMPLE * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cand_x, cand_x.data(), N_SAMPLE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cand_y, cand_y.data(), N_SAMPLE * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 256;
    int num_blocks = (N_SAMPLE + block_size - 1) / block_size;
    collision_check_samples_kernel<<<num_blocks, block_size>>>(
        d_cand_x, d_cand_y, N_SAMPLE,
        d_ob_x, d_ob_y, d_ob_r, num_obstacles,
        ROBOT_SIZE, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_result(N_SAMPLE);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, N_SAMPLE * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_cand_x); cudaFree(d_cand_y); cudaFree(d_result);

    for (int i = 0; i < N_SAMPLE; i++) {
        if (h_result[i] == 0) {
            res.node_x.push_back(cand_x[i]);
            res.node_y.push_back(cand_y[i]);
        }
    }

    int num_nodes = (int)res.node_x.size();
    res.adjacency.resize(num_nodes);

    // GPU k-NN
    float *d_node_x, *d_node_y;
    CUDA_CHECK(cudaMalloc(&d_node_x, num_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_node_y, num_nodes * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_node_x, res.node_x.data(), num_nodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_y, res.node_y.data(), num_nodes * sizeof(float), cudaMemcpyHostToDevice));

    int knn_total = num_nodes * N_KNN;
    int *d_knn_indices, *d_knn_counts;
    float *d_knn_dists;
    CUDA_CHECK(cudaMalloc(&d_knn_indices, knn_total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_knn_dists, knn_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_knn_counts, num_nodes * sizeof(int)));

    num_blocks = (num_nodes + block_size - 1) / block_size;
    find_knn_kernel<<<num_blocks, block_size>>>(
        d_node_x, d_node_y, num_nodes, N_KNN, MAX_EDGE_LEN,
        d_knn_indices, d_knn_dists, d_knn_counts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_knn_indices(knn_total);
    std::vector<float> h_knn_dists(knn_total);
    std::vector<int> h_knn_counts(num_nodes);
    CUDA_CHECK(cudaMemcpy(h_knn_indices.data(), d_knn_indices, knn_total * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_knn_dists.data(), d_knn_dists, knn_total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_knn_counts.data(), d_knn_counts, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_knn_indices); cudaFree(d_knn_dists); cudaFree(d_knn_counts);
    cudaFree(d_node_x); cudaFree(d_node_y);

    // Collect candidate edges
    struct CandEdge { int from, to; float dist; };
    std::vector<CandEdge> candidates;
    for (int i = 0; i < num_nodes; i++) {
        for (int k = 0; k < N_KNN; k++) {
            int j = h_knn_indices[i * N_KNN + k];
            if (j < 0) continue;
            candidates.push_back({i, j, h_knn_dists[i * N_KNN + k]});
        }
    }

    int num_edges = (int)candidates.size();
    if (num_edges > 0) {
        std::vector<float> ex1(num_edges), ey1(num_edges), ex2(num_edges), ey2(num_edges);
        for (int i = 0; i < num_edges; i++) {
            ex1[i] = res.node_x[candidates[i].from];
            ey1[i] = res.node_y[candidates[i].from];
            ex2[i] = res.node_x[candidates[i].to];
            ey2[i] = res.node_y[candidates[i].to];
        }

        float *d_ex1, *d_ey1, *d_ex2, *d_ey2;
        int *d_edge_result;
        CUDA_CHECK(cudaMalloc(&d_ex1, num_edges * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ey1, num_edges * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ex2, num_edges * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ey2, num_edges * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_edge_result, num_edges * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_ex1, ex1.data(), num_edges * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ey1, ey1.data(), num_edges * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ex2, ex2.data(), num_edges * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ey2, ey2.data(), num_edges * sizeof(float), cudaMemcpyHostToDevice));

        num_blocks = (num_edges + block_size - 1) / block_size;
        edge_collision_check_kernel<<<num_blocks, block_size>>>(
            d_ex1, d_ey1, d_ex2, d_ey2, num_edges,
            d_ob_x, d_ob_y, d_ob_r, num_obstacles,
            ROBOT_SIZE, EDGE_CHECK_STEPS, d_edge_result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> h_edge_result(num_edges);
        CUDA_CHECK(cudaMemcpy(h_edge_result.data(), d_edge_result, num_edges * sizeof(int), cudaMemcpyDeviceToHost));

        cudaFree(d_ex1); cudaFree(d_ey1); cudaFree(d_ex2); cudaFree(d_ey2); cudaFree(d_edge_result);

        for (int i = 0; i < num_edges; i++) {
            if (h_edge_result[i] == 0) {
                int f = candidates[i].from;
                int t = candidates[i].to;
                float c = candidates[i].dist;
                res.adjacency[f].push_back({f, t, c});
            }
        }
    }

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    res.build_ms = (double)gpu_ms;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_ob_x); cudaFree(d_ob_y); cudaFree(d_ob_r);

    // Dijkstra
    res.path = dijkstra(res.node_x, res.node_y, res.adjacency, 0, 1);
    return res;
}

// ---------------------------------------------------------------------------
// Visualization
// ---------------------------------------------------------------------------

static void draw_prm(cv::Mat& img, const PRMResult& prm,
                     const std::vector<float>& ob_x, const std::vector<float>& ob_y,
                     const std::vector<float>& ob_r,
                     float sx, float sy, float gx, float gy,
                     float area_min_x, float area_min_y, int scale,
                     const char* label, double build_ms)
{
    auto toPixel = [&](float wx, float wy) -> cv::Point {
        return cv::Point((int)((wx - area_min_x) * scale),
                         (int)((wy - area_min_y) * scale));
    };

    // Obstacles
    for (int i = 0; i < (int)ob_x.size(); i++) {
        cv::circle(img, toPixel(ob_x[i], ob_y[i]),
                   (int)(ob_r[i] * scale), cv::Scalar(0, 0, 0), -1);
    }

    // Roadmap edges (gray)
    int num_nodes = (int)prm.node_x.size();
    for (int i = 0; i < num_nodes; i++) {
        for (const auto& e : prm.adjacency[i]) {
            cv::line(img, toPixel(prm.node_x[e.from], prm.node_y[e.from]),
                     toPixel(prm.node_x[e.to], prm.node_y[e.to]),
                     cv::Scalar(200, 200, 200), 1);
        }
    }

    // Sample nodes (small gray dots)
    for (int i = 2; i < num_nodes; i++) {
        cv::circle(img, toPixel(prm.node_x[i], prm.node_y[i]),
                   2, cv::Scalar(150, 150, 150), -1);
    }

    // Final path (blue)
    for (int i = 0; i + 1 < (int)prm.path.size(); i++) {
        cv::line(img, toPixel(prm.path[i].first, prm.path[i].second),
                 toPixel(prm.path[i + 1].first, prm.path[i + 1].second),
                 cv::Scalar(255, 0, 0), 3);
    }

    // Start (green), goal (red)
    cv::circle(img, toPixel(sx, sy), 8, cv::Scalar(0, 200, 0), -1);
    cv::circle(img, toPixel(gx, gy), 8, cv::Scalar(0, 0, 255), -1);

    // Labels
    cv::putText(img, label, cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    char buf[128];
    snprintf(buf, sizeof(buf), "Build: %.2f ms", build_ms);
    cv::putText(img, buf, cv::Point(10, 55),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 200), 2);

    if (!prm.path.empty()) {
        cv::putText(img, "PATH FOUND", cv::Point(10, 85),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 180, 0), 2);
    } else {
        cv::putText(img, "NO PATH", cv::Point(10, 85),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 200), 2);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    // Obstacles (same as prm.cu)
    std::vector<float> ob_x, ob_y, ob_r;

    // Central obstacle
    ob_x.push_back(25.0f); ob_y.push_back(25.0f); ob_r.push_back(10.0f);

    // Border walls
    for (float y = 0.0f; y <= 60.0f; y += 1.0f) {
        ob_x.push_back(0.0f);  ob_y.push_back(y);    ob_r.push_back(0.5f);
        ob_x.push_back(60.0f); ob_y.push_back(y);    ob_r.push_back(0.5f);
    }
    for (float x = 0.0f; x <= 60.0f; x += 1.0f) {
        ob_x.push_back(x);    ob_y.push_back(0.0f);  ob_r.push_back(0.5f);
        ob_x.push_back(x);    ob_y.push_back(60.0f); ob_r.push_back(0.5f);
    }

    float start_x = 5.0f,  start_y = 5.0f;
    float goal_x  = 50.0f, goal_y  = 50.0f;
    float area_min_x = -2.0f, area_max_x = 62.0f;
    float area_min_y = -2.0f, area_max_y = 62.0f;

    std::cout << "PRM comparison: CPU vs CUDA" << std::endl;
    std::cout << "Number of obstacles: " << ob_x.size() << std::endl;

    // Build CPU PRM
    std::cout << "Building CPU PRM..." << std::endl;
    PRMResult cpu_prm = build_prm_cpu(start_x, start_y, goal_x, goal_y,
                                       ob_x, ob_y, ob_r,
                                       area_min_x, area_max_x, area_min_y, area_max_y);
    std::cout << "CPU build: " << cpu_prm.build_ms << " ms, nodes: " << cpu_prm.node_x.size()
              << ", path: " << (cpu_prm.path.empty() ? "NOT FOUND" : "FOUND") << std::endl;

    // Build CUDA PRM
    std::cout << "Building CUDA PRM..." << std::endl;
    PRMResult cuda_prm = build_prm_cuda(start_x, start_y, goal_x, goal_y,
                                         ob_x, ob_y, ob_r,
                                         area_min_x, area_max_x, area_min_y, area_max_y);
    std::cout << "CUDA build: " << cuda_prm.build_ms << " ms, nodes: " << cuda_prm.node_x.size()
              << ", path: " << (cuda_prm.path.empty() ? "NOT FOUND" : "FOUND") << std::endl;

    // Visualization: 400x400 per side
    int img_w = (int)(area_max_x - area_min_x);  // 64
    int img_h = (int)(area_max_y - area_min_y);  // 64
    // scale to get ~400 pixels: 400/64 ~ 6.25 -> use 6
    int scale = 6;
    int W = img_w * scale;  // 384
    int H = img_h * scale;  // 384

    // Pad to exactly 400x400
    int PAD_W = 400, PAD_H = 400;

    cv::Mat left(PAD_H, PAD_W, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat right(PAD_H, PAD_W, CV_8UC3, cv::Scalar(255, 255, 255));

    draw_prm(left, cpu_prm, ob_x, ob_y, ob_r,
             start_x, start_y, goal_x, goal_y,
             area_min_x, area_min_y, scale,
             "CPU (sequential)", cpu_prm.build_ms);

    draw_prm(right, cuda_prm, ob_x, ob_y, ob_r,
             start_x, start_y, goal_x, goal_y,
             area_min_x, area_min_y, scale,
             "CUDA (GPU parallel)", cuda_prm.build_ms);

    cv::Mat combined;
    cv::hconcat(left, right, combined);  // 800x400

    // Write as video (repeated frames for visibility as gif)
    cv::VideoWriter video("gif/comparison_prm.avi",
                          cv::VideoWriter::fourcc('X','V','I','D'), 10, cv::Size(PAD_W * 2, PAD_H));

    for (int f = 0; f < 50; f++) video.write(combined);  // 5s at 10fps

    video.release();
    std::cout << "Video saved to gif/comparison_prm.avi" << std::endl;

    system("ffmpeg -y -i gif/comparison_prm.avi "
           "-vf 'fps=10,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_prm.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_prm.gif" << std::endl;

    return 0;
}
