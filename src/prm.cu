/*************************************************************************
    > File Name: prm.cu
    > CUDA-parallelized Probabilistic Road Map (PRM) planner
    > Based on PythonRobotics PRM implementation by Atsushi Sakai
    > Parallelizes sample collision check, KNN search, and edge collision
    >   check via CUDA kernels
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <queue>
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
// Constants
// ---------------------------------------------------------------------------

static const int   N_SAMPLE      = 500;
static const int   N_KNN         = 10;
static const float MAX_EDGE_LEN  = 30.0f;
static const float ROBOT_SIZE    = 5.0f;
static const int   EDGE_CHECK_STEPS = 100;  // discretization steps for edge collision

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Kernel 1: Check each sample point against all obstacles.
// One thread per sample. result[i] = 1 if sample i collides, 0 otherwise.
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

// Kernel 2: For each sample, find k-nearest neighbors among all other samples.
// One thread per sample. Writes N_KNN neighbor indices per sample.
// Uses a simple insertion-sort approach to maintain top-k.
__global__ void find_knn_kernel(
    const float* __restrict__ sample_x,
    const float* __restrict__ sample_y,
    int num_samples,
    int k,
    float max_edge_len,
    int* knn_indices,   // output: num_samples * k
    float* knn_dists,   // output: num_samples * k
    int* knn_counts)    // output: actual neighbor count per sample
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;

    float mx = sample_x[idx];
    float my = sample_y[idx];

    // Local arrays for top-k (stored in registers/local memory)
    float top_dist[10];  // N_KNN = 10
    int   top_idx[10];
    int   count = 0;

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

        // Check if this is closer than the current worst in top-k
        // Find the position of maximum distance in top-k
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

    // Write results
    count = 0;
    for (int i = 0; i < k; i++) {
        knn_indices[idx * k + i] = top_idx[i];
        knn_dists[idx * k + i] = top_dist[i];
        if (top_idx[i] >= 0) count++;
    }
    knn_counts[idx] = count;
}

// Kernel 3: Check collision along candidate edges.
// One thread per edge. Discretize the edge and check each step against all obstacles.
// result[i] = 1 if edge i has collision, 0 otherwise.
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
// Host-side PRM class
// ---------------------------------------------------------------------------

struct Edge {
    int from;
    int to;
    float cost;
};

class CudaPRM {
public:
    CudaPRM(float sx, float sy, float gx, float gy,
            const std::vector<float>& ob_x,
            const std::vector<float>& ob_y,
            const std::vector<float>& ob_r,
            float area_min_x, float area_max_x,
            float area_min_y, float area_max_y);
    ~CudaPRM();

    std::vector<std::pair<float,float>> planning();

private:
    // Start / goal
    float start_x, start_y;
    float goal_x, goal_y;

    // Area bounds
    float area_min_x, area_max_x;
    float area_min_y, area_max_y;

    // Obstacles (host)
    int num_obstacles;
    std::vector<float> h_ob_x, h_ob_y, h_ob_r;

    // Obstacles (device)
    float *d_ob_x, *d_ob_y, *d_ob_r;

    // Roadmap
    std::vector<float> node_x, node_y;  // includes start(0) and goal(1)
    std::vector<std::vector<Edge>> adjacency;

    // Methods
    void generateSamples();
    void buildRoadmap();
    std::vector<std::pair<float,float>> dijkstraSearch(int start_id, int goal_id);
};

CudaPRM::CudaPRM(float sx, float sy, float gx, float gy,
                 const std::vector<float>& ox,
                 const std::vector<float>& oy,
                 const std::vector<float>& or_,
                 float amin_x, float amax_x,
                 float amin_y, float amax_y)
    : start_x(sx), start_y(sy), goal_x(gx), goal_y(gy),
      area_min_x(amin_x), area_max_x(amax_x),
      area_min_y(amin_y), area_max_y(amax_y),
      h_ob_x(ox), h_ob_y(oy), h_ob_r(or_),
      d_ob_x(nullptr), d_ob_y(nullptr), d_ob_r(nullptr)
{
    num_obstacles = (int)h_ob_x.size();

    // Upload obstacles to device
    CUDA_CHECK(cudaMalloc(&d_ob_x, num_obstacles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ob_y, num_obstacles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ob_r, num_obstacles * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ob_x, h_ob_x.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ob_y, h_ob_y.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ob_r, h_ob_r.data(), num_obstacles * sizeof(float), cudaMemcpyHostToDevice));
}

CudaPRM::~CudaPRM() {
    if (d_ob_x) cudaFree(d_ob_x);
    if (d_ob_y) cudaFree(d_ob_y);
    if (d_ob_r) cudaFree(d_ob_r);
}

void CudaPRM::generateSamples() {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist_x(area_min_x, area_max_x);
    std::uniform_real_distribution<float> dist_y(area_min_y, area_max_y);

    // Node 0 = start, Node 1 = goal
    node_x.clear();
    node_y.clear();
    node_x.push_back(start_x);
    node_y.push_back(start_y);
    node_x.push_back(goal_x);
    node_y.push_back(goal_y);

    // Generate candidate samples
    std::vector<float> cand_x(N_SAMPLE), cand_y(N_SAMPLE);
    for (int i = 0; i < N_SAMPLE; i++) {
        cand_x[i] = dist_x(gen);
        cand_y[i] = dist_y(gen);
    }

    // Upload candidates to device
    float *d_cand_x, *d_cand_y;
    int *d_result;
    CUDA_CHECK(cudaMalloc(&d_cand_x, N_SAMPLE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cand_y, N_SAMPLE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, N_SAMPLE * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cand_x, cand_x.data(), N_SAMPLE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cand_y, cand_y.data(), N_SAMPLE * sizeof(float), cudaMemcpyHostToDevice));

    // Launch collision check kernel
    int block_size = 256;
    int num_blocks = (N_SAMPLE + block_size - 1) / block_size;
    collision_check_samples_kernel<<<num_blocks, block_size>>>(
        d_cand_x, d_cand_y, N_SAMPLE,
        d_ob_x, d_ob_y, d_ob_r, num_obstacles,
        ROBOT_SIZE, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    std::vector<int> h_result(N_SAMPLE);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, N_SAMPLE * sizeof(int), cudaMemcpyDeviceToHost));

    // Keep collision-free samples
    for (int i = 0; i < N_SAMPLE; i++) {
        if (h_result[i] == 0) {
            node_x.push_back(cand_x[i]);
            node_y.push_back(cand_y[i]);
        }
    }

    cudaFree(d_cand_x);
    cudaFree(d_cand_y);
    cudaFree(d_result);

    std::cout << "Valid samples (including start/goal): " << node_x.size() << std::endl;
}

void CudaPRM::buildRoadmap() {
    int num_nodes = (int)node_x.size();
    adjacency.resize(num_nodes);

    // Upload all node positions to device
    float *d_node_x, *d_node_y;
    CUDA_CHECK(cudaMalloc(&d_node_x, num_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_node_y, num_nodes * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_node_x, node_x.data(), num_nodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_y, node_y.data(), num_nodes * sizeof(float), cudaMemcpyHostToDevice));

    // KNN output arrays
    int knn_total = num_nodes * N_KNN;
    int *d_knn_indices, *d_knn_counts;
    float *d_knn_dists;
    CUDA_CHECK(cudaMalloc(&d_knn_indices, knn_total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_knn_dists, knn_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_knn_counts, num_nodes * sizeof(int)));

    // Launch KNN kernel
    int block_size = 256;
    int num_blocks = (num_nodes + block_size - 1) / block_size;
    find_knn_kernel<<<num_blocks, block_size>>>(
        d_node_x, d_node_y, num_nodes, N_KNN, MAX_EDGE_LEN,
        d_knn_indices, d_knn_dists, d_knn_counts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy KNN results back
    std::vector<int> h_knn_indices(knn_total);
    std::vector<float> h_knn_dists(knn_total);
    std::vector<int> h_knn_counts(num_nodes);
    CUDA_CHECK(cudaMemcpy(h_knn_indices.data(), d_knn_indices, knn_total * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_knn_dists.data(), d_knn_dists, knn_total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_knn_counts.data(), d_knn_counts, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_knn_indices);
    cudaFree(d_knn_dists);
    cudaFree(d_knn_counts);
    cudaFree(d_node_x);
    cudaFree(d_node_y);

    // Collect all candidate edges (avoid duplicates by requiring from < to)
    struct CandEdge {
        int from, to;
        float dist;
    };
    std::vector<CandEdge> candidates;

    for (int i = 0; i < num_nodes; i++) {
        for (int k = 0; k < N_KNN; k++) {
            int j = h_knn_indices[i * N_KNN + k];
            if (j < 0) continue;
            float d = h_knn_dists[i * N_KNN + k];
            // Store both directions but deduplicate later in adjacency
            candidates.push_back({i, j, d});
        }
    }

    int num_edges = (int)candidates.size();
    if (num_edges == 0) {
        std::cout << "No candidate edges found." << std::endl;
        return;
    }

    std::cout << "Candidate edges: " << num_edges << std::endl;

    // Prepare edge endpoints for GPU collision check
    std::vector<float> ex1(num_edges), ey1(num_edges), ex2(num_edges), ey2(num_edges);
    for (int i = 0; i < num_edges; i++) {
        ex1[i] = node_x[candidates[i].from];
        ey1[i] = node_y[candidates[i].from];
        ex2[i] = node_x[candidates[i].to];
        ey2[i] = node_y[candidates[i].to];
    }

    // Upload edge data to device
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

    // Launch edge collision check kernel
    num_blocks = (num_edges + block_size - 1) / block_size;
    edge_collision_check_kernel<<<num_blocks, block_size>>>(
        d_ex1, d_ey1, d_ex2, d_ey2, num_edges,
        d_ob_x, d_ob_y, d_ob_r, num_obstacles,
        ROBOT_SIZE, EDGE_CHECK_STEPS, d_edge_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    std::vector<int> h_edge_result(num_edges);
    CUDA_CHECK(cudaMemcpy(h_edge_result.data(), d_edge_result, num_edges * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_ex1);
    cudaFree(d_ey1);
    cudaFree(d_ex2);
    cudaFree(d_ey2);
    cudaFree(d_edge_result);

    // Build adjacency list from collision-free edges
    int valid_edges = 0;
    for (int i = 0; i < num_edges; i++) {
        if (h_edge_result[i] == 0) {
            int f = candidates[i].from;
            int t = candidates[i].to;
            float c = candidates[i].dist;
            adjacency[f].push_back({f, t, c});
            valid_edges++;
        }
    }

    std::cout << "Valid edges in roadmap: " << valid_edges << std::endl;
}

std::vector<std::pair<float,float>> CudaPRM::dijkstraSearch(int start_id, int goal_id) {
    int num_nodes = (int)node_x.size();
    std::vector<float> dist(num_nodes, FLT_MAX);
    std::vector<int> prev(num_nodes, -1);
    std::vector<bool> visited(num_nodes, false);

    // Min-heap: (distance, node_id)
    typedef std::pair<float, int> PQEntry;
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;

    dist[start_id] = 0.0f;
    pq.push({0.0f, start_id});

    while (!pq.empty()) {
        float d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (visited[u]) continue;
        visited[u] = true;

        if (u == goal_id) break;

        for (const auto& edge : adjacency[u]) {
            int v = edge.to;
            float new_dist = dist[u] + edge.cost;
            if (new_dist < dist[v]) {
                dist[v] = new_dist;
                prev[v] = u;
                pq.push({new_dist, v});
            }
        }
    }

    // Reconstruct path
    std::vector<std::pair<float,float>> path;
    if (!visited[goal_id]) {
        std::cout << "No path found!" << std::endl;
        return path;
    }

    int cur = goal_id;
    while (cur != -1) {
        path.push_back({node_x[cur], node_y[cur]});
        cur = prev[cur];
    }
    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<std::pair<float,float>> CudaPRM::planning() {
    // Phase 1: Learning - generate samples and build roadmap
    std::cout << "Generating samples..." << std::endl;
    generateSamples();

    std::cout << "Building roadmap..." << std::endl;
    buildRoadmap();

    // Phase 2: Query - Dijkstra from start(0) to goal(1)
    std::cout << "Running Dijkstra..." << std::endl;
    auto path = dijkstraSearch(0, 1);

    if (path.empty()) {
        std::cout << "Failed to find a path." << std::endl;
        return path;
    }

    std::cout << "Path found with " << path.size() << " waypoints." << std::endl;

    // --- Visualization ---
    int img_w = (int)(area_max_x - area_min_x);
    int img_h = (int)(area_max_y - area_min_y);
    int scale = 10;
    cv::Mat img(img_h * scale, img_w * scale, CV_8UC3, cv::Scalar(255, 255, 255));

    auto toPixel = [&](float wx, float wy) -> cv::Point {
        return cv::Point((int)((wx - area_min_x) * scale),
                         (int)((wy - area_min_y) * scale));
    };

    // Draw obstacles
    for (int i = 0; i < num_obstacles; i++) {
        cv::circle(img, toPixel(h_ob_x[i], h_ob_y[i]),
                   (int)(h_ob_r[i] * scale), cv::Scalar(0, 0, 0), -1);
    }

    // Draw roadmap edges (thin gray lines)
    int num_nodes = (int)node_x.size();
    for (int i = 0; i < num_nodes; i++) {
        for (const auto& edge : adjacency[i]) {
            cv::line(img, toPixel(node_x[edge.from], node_y[edge.from]),
                     toPixel(node_x[edge.to], node_y[edge.to]),
                     cv::Scalar(200, 200, 200), 1);
        }
    }

    // Draw sample nodes as small dots
    for (int i = 2; i < num_nodes; i++) {
        cv::circle(img, toPixel(node_x[i], node_y[i]),
                   2, cv::Scalar(150, 150, 150), -1);
    }

    // Draw final path (thick blue line)
    for (int i = 0; i + 1 < (int)path.size(); i++) {
        cv::line(img, toPixel(path[i].first, path[i].second),
                 toPixel(path[i + 1].first, path[i + 1].second),
                 cv::Scalar(255, 0, 0), 3);
    }

    // Draw start (green) and goal (red)
    cv::circle(img, toPixel(start_x, start_y), 8, cv::Scalar(0, 200, 0), -1);
    cv::circle(img, toPixel(goal_x, goal_y),   8, cv::Scalar(0, 0, 255), -1);

    cv::namedWindow("prm", cv::WINDOW_NORMAL);
    cv::imshow("prm", img);
    cv::waitKey(0);

    return path;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    // Obstacles: one big circle in the middle + border walls
    std::vector<float> ob_x, ob_y, ob_r;

    // Central obstacle
    ob_x.push_back(25.0f); ob_y.push_back(25.0f); ob_r.push_back(10.0f);

    // Border walls: approximate with small circles along the boundary
    // x = 0 wall and x = 60 wall
    for (float y = 0.0f; y <= 60.0f; y += 1.0f) {
        ob_x.push_back(0.0f);  ob_y.push_back(y);    ob_r.push_back(0.5f);
        ob_x.push_back(60.0f); ob_y.push_back(y);    ob_r.push_back(0.5f);
    }
    // y = 0 wall and y = 60 wall
    for (float x = 0.0f; x <= 60.0f; x += 1.0f) {
        ob_x.push_back(x);    ob_y.push_back(0.0f);  ob_r.push_back(0.5f);
        ob_x.push_back(x);    ob_y.push_back(60.0f); ob_r.push_back(0.5f);
    }

    std::cout << "Number of obstacles: " << ob_x.size() << std::endl;

    float start_x = 5.0f,  start_y = 5.0f;
    float goal_x  = 50.0f, goal_y  = 50.0f;

    // Area bounds (slightly beyond the walls for sampling)
    float area_min_x = -2.0f, area_max_x = 62.0f;
    float area_min_y = -2.0f, area_max_y = 62.0f;

    CudaPRM prm(start_x, start_y, goal_x, goal_y,
                ob_x, ob_y, ob_r,
                area_min_x, area_max_x,
                area_min_y, area_max_y);

    auto path = prm.planning();

    if (!path.empty()) {
        std::cout << "Path:" << std::endl;
        for (const auto& p : path) {
            std::cout << "  (" << p.first << ", " << p.second << ")" << std::endl;
        }
    }

    return 0;
}
