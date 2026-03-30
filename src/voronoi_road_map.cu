/*************************************************************************
    > File Name: voronoi_road_map.cu
    > CUDA-parallelized Voronoi Road Map Planner
    > Based on PythonRobotics VoronoiRoadMap by Atsushi Sakai
    > Voronoi diagram built on GPU using Jump Flooding Algorithm (JFA)
    > Dijkstra search on extracted road map runs on CPU
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <limits>
#include <queue>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

// -------------------------------------------------------------------------
// JFA Init Kernel
// Seed cells (obstacle points) get their own index; others get -1.
// -------------------------------------------------------------------------
__global__ void jfa_init_clear_kernel(int* grid, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        grid[idx] = -1;
    }
}

__global__ void jfa_init_seed_kernel(int* grid, int grid_w, int grid_h,
                                     const float* obs_x, const float* obs_y,
                                     int num_obs, int min_x, int min_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_obs) return;

    int gx = (int)roundf(obs_x[idx]) - min_x;
    int gy = (int)roundf(obs_y[idx]) - min_y;
    if (gx >= 0 && gx < grid_w && gy >= 0 && gy < grid_h) {
        grid[gy * grid_w + gx] = idx;
    }
}

// -------------------------------------------------------------------------
// JFA Step Kernel
// Each cell checks 8 neighbors at distance `step` and adopts nearest seed.
// -------------------------------------------------------------------------
__global__ void jfa_step_kernel(int* grid_out, const int* grid_in,
                                int grid_w, int grid_h,
                                const float* seed_x, const float* seed_y,
                                int num_seeds, int min_x, int min_y,
                                int step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = grid_w * grid_h;
    if (idx >= total) return;

    int cx = idx % grid_w;
    int cy = idx / grid_w;

    float px = (float)(cx + min_x);
    float py = (float)(cy + min_y);

    int best_seed = grid_in[idx];
    float best_dist = 1e30f;

    if (best_seed >= 0 && best_seed < num_seeds) {
        float dx = px - seed_x[best_seed];
        float dy = py - seed_y[best_seed];
        best_dist = dx * dx + dy * dy;
    }

    // Check 9 neighbors (including self) at offset `step`
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cx + dx * step;
            int ny = cy + dy * step;
            if (nx < 0 || nx >= grid_w || ny < 0 || ny >= grid_h) continue;

            int neighbor_seed = grid_in[ny * grid_w + nx];
            if (neighbor_seed < 0 || neighbor_seed >= num_seeds) continue;

            float sx = px - seed_x[neighbor_seed];
            float sy = py - seed_y[neighbor_seed];
            float d = sx * sx + sy * sy;

            if (d < best_dist) {
                best_dist = d;
                best_seed = neighbor_seed;
            }
        }
    }

    grid_out[idx] = best_seed;
}

// -------------------------------------------------------------------------
// Extract Voronoi Edges Kernel
// A cell is a Voronoi edge if adjacent cells have different nearest seeds
// and the cell is far enough from obstacles (> robot_radius).
// -------------------------------------------------------------------------
__global__ void extract_voronoi_edges_kernel(int* edge_map, const int* voronoi_grid,
                                             const float* seed_x, const float* seed_y,
                                             int num_seeds, int min_x, int min_y,
                                             int grid_w, int grid_h,
                                             float robot_radius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = grid_w * grid_h;
    if (idx >= total) return;

    int cx = idx % grid_w;
    int cy = idx / grid_w;

    edge_map[idx] = 0;

    int my_seed = voronoi_grid[idx];
    if (my_seed < 0) return;

    // Check distance to nearest obstacle (the seed itself)
    float px = (float)(cx + min_x);
    float py = (float)(cy + min_y);
    float dx = px - seed_x[my_seed];
    float dy = py - seed_y[my_seed];
    float dist_to_obs = sqrtf(dx * dx + dy * dy);

    if (dist_to_obs <= robot_radius) return;

    // Check 4-connected neighbors for different seeds
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int i = 0; i < 4; i++) {
        int nx = cx + offsets[i][0];
        int ny = cy + offsets[i][1];
        if (nx < 0 || nx >= grid_w || ny < 0 || ny >= grid_h) continue;

        int neighbor_seed = voronoi_grid[ny * grid_w + nx];
        if (neighbor_seed >= 0 && neighbor_seed != my_seed) {
            edge_map[idx] = 1;
            return;
        }
    }
}

// -------------------------------------------------------------------------
// Build road map graph from edge points + start/goal
// -------------------------------------------------------------------------
struct GraphNode {
    float x, y;
    vector<pair<int, float>> neighbors;  // (node_id, distance)
};

float dist2d(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return sqrtf(dx * dx + dy * dy);
}

// -------------------------------------------------------------------------
// Dijkstra on road map graph (CPU)
// -------------------------------------------------------------------------
vector<int> dijkstra_road_map(vector<GraphNode>& graph, int start_id, int goal_id) {
    int n = (int)graph.size();
    vector<float> dist(n, 1e30f);
    vector<int> prev(n, -1);
    vector<bool> visited(n, false);

    auto cmp = [](const pair<float, int>& a, const pair<float, int>& b) { return a.first > b.first; };
    priority_queue<pair<float, int>, vector<pair<float, int>>, decltype(cmp)> pq(cmp);

    dist[start_id] = 0.0f;
    pq.push(make_pair(0.0f, start_id));

    while (!pq.empty()) {
        pair<float, int> top = pq.top();
        pq.pop();
        int u = top.second;

        if (visited[u]) continue;
        visited[u] = true;

        if (u == goal_id) break;

        for (int k = 0; k < (int)graph[u].neighbors.size(); k++) {
            int v = graph[u].neighbors[k].first;
            float w = graph[u].neighbors[k].second;
            float nd = dist[u] + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                prev[v] = u;
                pq.push(make_pair(nd, v));
            }
        }
    }

    // Reconstruct path
    vector<int> path;
    if (!visited[goal_id]) return path;

    int cur = goal_id;
    while (cur != -1) {
        path.push_back(cur);
        cur = prev[cur];
    }
    reverse(path.begin(), path.end());
    return path;
}

// -------------------------------------------------------------------------
// Check if line segment from (x1,y1) to (x2,y2) is obstacle-free
// -------------------------------------------------------------------------
bool line_of_sight(float x1, float y1, float x2, float y2,
                   const vector<int>& h_voronoi_grid,
                   const float* h_seed_x, const float* h_seed_y,
                   int grid_w, int grid_h, int min_x, int min_y,
                   float robot_radius)
{
    float d = dist2d(x1, y1, x2, y2);
    int steps = (int)(d / 0.5f) + 1;
    for (int i = 0; i <= steps; i++) {
        float t = (steps == 0) ? 0.0f : (float)i / (float)steps;
        float px = x1 + t * (x2 - x1);
        float py = y1 + t * (y2 - y1);

        int gx = (int)roundf(px) - min_x;
        int gy = (int)roundf(py) - min_y;
        if (gx < 0 || gx >= grid_w || gy < 0 || gy >= grid_h) return false;

        int seed = h_voronoi_grid[gy * grid_w + gx];
        if (seed < 0) return false;

        float dx = px - h_seed_x[seed];
        float dy = py - h_seed_y[seed];
        if (sqrtf(dx * dx + dy * dy) <= robot_radius) return false;
    }
    return true;
}

// -------------------------------------------------------------------------
// Main Voronoi Road Map Planner
// -------------------------------------------------------------------------
void voronoi_road_map_planning(float sx, float sy, float gx, float gy,
                               vector<float>& ox, vector<float>& oy,
                               float robot_radius)
{
    int num_obs = (int)ox.size();

    // Compute grid bounds
    float fmin_x = *min_element(ox.begin(), ox.end());
    float fmax_x = *max_element(ox.begin(), ox.end());
    float fmin_y = *min_element(oy.begin(), oy.end());
    float fmax_y = *max_element(oy.begin(), oy.end());

    int min_x = (int)floorf(fmin_x) - 2;
    int max_x = (int)ceilf(fmax_x) + 2;
    int min_y = (int)floorf(fmin_y) - 2;
    int max_y = (int)ceilf(fmax_y) + 2;

    int grid_w = max_x - min_x + 1;
    int grid_h = max_y - min_y + 1;
    int total = grid_w * grid_h;

    printf("Grid: %d x %d (%d cells), obstacles: %d\n", grid_w, grid_h, total, num_obs);

    // ------ Allocate device memory ------
    float *d_ox, *d_oy;
    int *d_grid_a, *d_grid_b;
    int *d_edge_map;

    CUDA_CHECK(cudaMalloc(&d_ox, num_obs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_oy, num_obs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_a, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_b, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_map, total * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_ox, ox.data(), num_obs * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oy, oy.data(), num_obs * sizeof(float), cudaMemcpyHostToDevice));

    // ------ JFA Init ------
    int blockSize = 256;
    int clearGridSize = (total + blockSize - 1) / blockSize;
    int seedGridSize = (num_obs + blockSize - 1) / blockSize;

    jfa_init_clear_kernel<<<clearGridSize, blockSize>>>(d_grid_a, total);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    jfa_init_seed_kernel<<<seedGridSize, blockSize>>>(d_grid_a, grid_w, grid_h,
                                                       d_ox, d_oy, num_obs,
                                                       min_x, min_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------ JFA Passes ------
    int max_dim = max(grid_w, grid_h);
    int step = 1;
    while (step < max_dim) step <<= 1;
    step >>= 1;  // start at largest power of 2 <= max_dim

    int gridSizeJFA = (total + blockSize - 1) / blockSize;
    int* src = d_grid_a;
    int* dst = d_grid_b;

    while (step >= 1) {
        jfa_step_kernel<<<gridSizeJFA, blockSize>>>(dst, src,
                                                     grid_w, grid_h,
                                                     d_ox, d_oy, num_obs,
                                                     min_x, min_y, step);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap buffers
        int* tmp = src;
        src = dst;
        dst = tmp;

        step >>= 1;
    }

    // src now has the final Voronoi grid

    // ------ Extract Voronoi Edges ------
    extract_voronoi_edges_kernel<<<gridSizeJFA, blockSize>>>(d_edge_map, src,
                                                              d_ox, d_oy, num_obs,
                                                              min_x, min_y,
                                                              grid_w, grid_h,
                                                              robot_radius);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------ Copy results to host ------
    vector<int> h_voronoi_grid(total);
    vector<int> h_edge_map(total);

    CUDA_CHECK(cudaMemcpy(h_voronoi_grid.data(), src, total * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_edge_map.data(), d_edge_map, total * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_ox));
    CUDA_CHECK(cudaFree(d_oy));
    CUDA_CHECK(cudaFree(d_grid_a));
    CUDA_CHECK(cudaFree(d_grid_b));
    CUDA_CHECK(cudaFree(d_edge_map));

    // ------ Collect Voronoi edge points ------
    vector<pair<float, float>> edge_points;
    for (int y = 0; y < grid_h; y++) {
        for (int x = 0; x < grid_w; x++) {
            if (h_edge_map[y * grid_w + x]) {
                edge_points.push_back({(float)(x + min_x), (float)(y + min_y)});
            }
        }
    }
    printf("Voronoi edge points: %d\n", (int)edge_points.size());

    // ------ Build road map graph ------
    // Subsample edge points to reduce graph size (take every Nth point)
    vector<pair<float, float>> sample_points;
    int subsample = max(1, (int)edge_points.size() / 500);
    for (int i = 0; i < (int)edge_points.size(); i += subsample) {
        sample_points.push_back(edge_points[i]);
    }

    // Add start and goal
    int start_id = (int)sample_points.size();
    sample_points.push_back({sx, sy});
    int goal_id = (int)sample_points.size();
    sample_points.push_back({gx, gy});

    int n_nodes = (int)sample_points.size();
    printf("Road map nodes: %d (subsampled from %d edge points)\n",
           n_nodes, (int)edge_points.size());

    // Build graph with KNN-style connectivity
    float connect_radius = 10.0f;  // connect nodes within this radius

    vector<GraphNode> graph(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        graph[i].x = sample_points[i].first;
        graph[i].y = sample_points[i].second;
    }

    // Connect nearby nodes if line-of-sight is clear
    for (int i = 0; i < n_nodes; i++) {
        for (int j = i + 1; j < n_nodes; j++) {
            float d = dist2d(graph[i].x, graph[i].y, graph[j].x, graph[j].y);

            // For start/goal, use larger radius to ensure connectivity
            float radius = connect_radius;
            if (i == start_id || i == goal_id || j == start_id || j == goal_id) {
                radius = 20.0f;
            }

            if (d <= radius) {
                if (line_of_sight(graph[i].x, graph[i].y, graph[j].x, graph[j].y,
                                  h_voronoi_grid, ox.data(), oy.data(),
                                  grid_w, grid_h, min_x, min_y, robot_radius)) {
                    graph[i].neighbors.push_back({j, d});
                    graph[j].neighbors.push_back({i, d});
                }
            }
        }
    }

    // ------ Dijkstra on road map ------
    vector<int> path = dijkstra_road_map(graph, start_id, goal_id);
    printf("Path length: %d nodes\n", (int)path.size());

    // ------ Visualization ------
    int img_scale = 10;
    int img_w = grid_w * img_scale;
    int img_h = grid_h * img_scale;
    cv::Mat img(img_h, img_w, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw Voronoi diagram as colored regions
    // Create a color map for seeds
    vector<cv::Vec3b> seed_colors(num_obs);
    srand(42);
    for (int i = 0; i < num_obs; i++) {
        seed_colors[i] = cv::Vec3b(100 + rand() % 130, 100 + rand() % 130, 100 + rand() % 130);
    }

    for (int y = 0; y < grid_h; y++) {
        for (int x = 0; x < grid_w; x++) {
            int seed = h_voronoi_grid[y * grid_w + x];
            if (seed >= 0 && seed < num_obs) {
                cv::rectangle(img,
                              cv::Point(x * img_scale, y * img_scale),
                              cv::Point((x + 1) * img_scale, (y + 1) * img_scale),
                              cv::Scalar(seed_colors[seed][0], seed_colors[seed][1], seed_colors[seed][2]),
                              -1);
            }
        }
    }

    // Draw Voronoi edges as thin dark lines
    for (auto& pt : edge_points) {
        int gx = (int)roundf(pt.first) - min_x;
        int gy = (int)roundf(pt.second) - min_y;
        cv::rectangle(img,
                      cv::Point(gx * img_scale, gy * img_scale),
                      cv::Point((gx + 1) * img_scale, (gy + 1) * img_scale),
                      cv::Scalar(50, 50, 50), -1);
    }

    // Draw obstacles as black
    for (int i = 0; i < num_obs; i++) {
        int gx = (int)roundf(ox[i]) - min_x;
        int gy = (int)roundf(oy[i]) - min_y;
        if (gx >= 0 && gx < grid_w && gy >= 0 && gy < grid_h) {
            cv::rectangle(img,
                          cv::Point(gx * img_scale, gy * img_scale),
                          cv::Point((gx + 1) * img_scale, (gy + 1) * img_scale),
                          cv::Scalar(0, 0, 0), -1);
        }
    }

    // Draw road map edges (light gray)
    for (int i = 0; i < n_nodes; i++) {
        for (int k = 0; k < (int)graph[i].neighbors.size(); k++) {
            int j = graph[i].neighbors[k].first;
            if (j > i) {
                cv::Point p1((int)((graph[i].x - min_x + 0.5f) * img_scale),
                             (int)((graph[i].y - min_y + 0.5f) * img_scale));
                cv::Point p2((int)((graph[j].x - min_x + 0.5f) * img_scale),
                             (int)((graph[j].y - min_y + 0.5f) * img_scale));
                cv::line(img, p1, p2, cv::Scalar(200, 200, 200), 1);
            }
        }
    }

    // Draw final path as thick red line
    if (path.size() >= 2) {
        for (int i = 0; i < (int)path.size() - 1; i++) {
            int a = path[i];
            int b = path[i + 1];
            cv::Point p1((int)((graph[a].x - min_x + 0.5f) * img_scale),
                         (int)((graph[a].y - min_y + 0.5f) * img_scale));
            cv::Point p2((int)((graph[b].x - min_x + 0.5f) * img_scale),
                         (int)((graph[b].y - min_y + 0.5f) * img_scale));
            cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 3);
        }
    } else {
        printf("No path found!\n");
    }

    // Draw start and goal
    cv::circle(img,
               cv::Point((int)((sx - min_x + 0.5f) * img_scale),
                          (int)((sy - min_y + 0.5f) * img_scale)),
               img_scale, cv::Scalar(0, 255, 0), -1);
    cv::circle(img,
               cv::Point((int)((gx - min_x + 0.5f) * img_scale),
                          (int)((gy - min_y + 0.5f) * img_scale)),
               img_scale, cv::Scalar(255, 0, 0), -1);

    cv::namedWindow("voronoi", cv::WINDOW_NORMAL);
    cv::imshow("voronoi", img);
    cv::waitKey(0);
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------
int main() {
    float sx = 5.0f;
    float sy = 5.0f;
    float gx = 55.0f;
    float gy = 55.0f;
    float robot_radius = 2.0f;

    vector<float> ox;
    vector<float> oy;

    // Boundary obstacles (rectangular area 0-60 x 0-60)
    for (float i = 0; i <= 60; i++) {
        ox.push_back(i); oy.push_back(0.0f);   // bottom
    }
    for (float i = 0; i <= 60; i++) {
        ox.push_back(i); oy.push_back(60.0f);  // top
    }
    for (float i = 0; i <= 60; i++) {
        ox.push_back(0.0f); oy.push_back(i);   // left
    }
    for (float i = 0; i <= 60; i++) {
        ox.push_back(60.0f); oy.push_back(i);  // right
    }

    // Internal obstacle: rectangle at (20,10)-(25,35)
    for (float x = 20; x <= 25; x++) {
        for (float y = 10; y <= 35; y++) {
            ox.push_back(x);
            oy.push_back(y);
        }
    }

    // Internal obstacle: rectangle at (35,25)-(40,50)
    for (float x = 35; x <= 40; x++) {
        for (float y = 25; y <= 50; y++) {
            ox.push_back(x);
            oy.push_back(y);
        }
    }

    voronoi_road_map_planning(sx, sy, gx, gy, ox, oy, robot_radius);

    return 0;
}
