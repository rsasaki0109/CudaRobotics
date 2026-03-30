/*************************************************************************
    Hybrid A*: CPU vs CUDA side-by-side comparison GIF generator
    Left panel:  CPU (sequential expansion of all steer angles)
    Right panel: CUDA (GPU-parallel expansion of all steer angles)
    Timing compares per-node expansion cost.
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;

// -------------------------------------------------------------------------
// Parameters
// -------------------------------------------------------------------------
static const int GRID_X = 100;
static const int GRID_Y = 100;
static const int N_THETA = 72;
static const float CELL_SIZE = 1.0f;
static const float WHEELBASE = 2.5f;
static const float MAX_STEER = 30.0f * M_PI / 180.0f;
static const int N_STEER = 21;
static const float STEP_SIZE = 3.0f;
static const int N_SIM_STEPS = 10;
static const float VEHICLE_LENGTH = 4.0f;
static const float VEHICLE_WIDTH = 2.0f;
static const float INF_COST = 1e9f;

// -------------------------------------------------------------------------
// Hybrid A* Node
// -------------------------------------------------------------------------
struct HybridNode {
    float x, y, theta;
    float g_cost;
    float f_cost;
    int parent_idx;
    int grid_idx;

    bool operator>(const HybridNode& o) const { return f_cost > o.f_cost; }
};

struct ChildResult {
    float x, y, theta;
    float cost;
    float f_cost;
    int valid;
    int grid_idx;
};

// -------------------------------------------------------------------------
// Discretize
// -------------------------------------------------------------------------
__host__ __device__ int discretize(float x, float y, float theta, int gx, int gy, int n_theta) {
    int ix = (int)floorf(x / CELL_SIZE);
    int iy = (int)floorf(y / CELL_SIZE);
    float t = fmodf(theta, 2.0f * M_PI);
    if (t < 0) t += 2.0f * M_PI;
    int it = (int)(t / (2.0f * M_PI) * n_theta) % n_theta;
    if (ix < 0 || ix >= gx || iy < 0 || iy >= gy) return -1;
    return (it * gy + iy) * gx + ix;
}

// -------------------------------------------------------------------------
// GPU kernels
// -------------------------------------------------------------------------
__global__ void build_obstacle_grid_kernel(
    int* d_obstacle_grid,
    const float* d_ox,
    const float* d_oy,
    int n_obstacles,
    int grid_x, int grid_y,
    float cell_size,
    float inflation_radius)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= grid_x || iy >= grid_y) return;

    float cx = (ix + 0.5f) * cell_size;
    float cy = (iy + 0.5f) * cell_size;
    float r2 = inflation_radius * inflation_radius;

    int blocked = 0;
    for (int k = 0; k < n_obstacles; k++) {
        float dx = d_ox[k] - cx;
        float dy = d_oy[k] - cy;
        if (dx * dx + dy * dy <= r2) {
            blocked = 1;
            break;
        }
    }
    d_obstacle_grid[iy * grid_x + ix] = blocked;
}

__global__ void compute_heuristic_kernel(
    float* d_heuristic,
    const int* d_obstacle_grid,
    const int* d_wavefront,
    int* d_wavefront_next,
    int* d_changed,
    int grid_x, int grid_y,
    float cell_size,
    int iteration)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= grid_x || iy >= grid_y) return;

    int idx = iy * grid_x + ix;
    if (d_wavefront[idx] != iteration) return;
    if (d_obstacle_grid[idx]) return;

    float current_cost = d_heuristic[idx];
    const int ddx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int ddy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const float dc[] = {1.414f, 1.0f, 1.414f, 1.0f, 1.0f, 1.414f, 1.0f, 1.414f};

    for (int d = 0; d < 8; d++) {
        int nx = ix + ddx[d];
        int ny = iy + ddy[d];
        if (nx < 0 || nx >= grid_x || ny < 0 || ny >= grid_y) continue;
        int nidx = ny * grid_x + nx;
        if (d_obstacle_grid[nidx]) continue;

        float new_cost = current_cost + dc[d] * cell_size;
        if (new_cost < d_heuristic[nidx]) {
            d_heuristic[nidx] = new_cost;
            d_wavefront_next[nidx] = iteration + 1;
            atomicExch(d_changed, 1);
        }
    }
}

__global__ void expand_node_batch_kernel(
    ChildResult* d_children,
    float parent_x, float parent_y, float parent_theta,
    float parent_g,
    const int* d_obstacle_grid,
    const float* d_heuristic,
    int grid_x, int grid_y,
    float cell_size,
    float wheelbase,
    float max_steer,
    int n_steer,
    float step_size,
    int n_sim_steps,
    float vehicle_length,
    float vehicle_width,
    int n_theta)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= n_steer) return;

    float steer = -max_steer + (2.0f * max_steer * sid) / (n_steer - 1);

    float x = parent_x;
    float y = parent_y;
    float theta = parent_theta;
    float ds = step_size / n_sim_steps;
    int collision = 0;
    float traveled = 0.0f;

    for (int s = 0; s < n_sim_steps; s++) {
        x += ds * cosf(theta);
        y += ds * sinf(theta);
        theta += ds * tanf(steer) / wheelbase;
        traveled += ds;
        theta = fmodf(theta, 2.0f * M_PI);
        if (theta < 0) theta += 2.0f * M_PI;

        float cos_t = cosf(theta);
        float sin_t = sinf(theta);
        float hl = vehicle_length * 0.5f;
        float hw = vehicle_width * 0.5f;
        float check_x[5], check_y[5];
        check_x[0] = x + hl * cos_t - hw * sin_t;
        check_y[0] = y + hl * sin_t + hw * cos_t;
        check_x[1] = x + hl * cos_t + hw * sin_t;
        check_y[1] = y + hl * sin_t - hw * cos_t;
        check_x[2] = x - hl * cos_t - hw * sin_t;
        check_y[2] = y - hl * sin_t + hw * cos_t;
        check_x[3] = x - hl * cos_t + hw * sin_t;
        check_y[3] = y - hl * sin_t - hw * cos_t;
        check_x[4] = x;
        check_y[4] = y;

        for (int c = 0; c < 5; c++) {
            int gxi = (int)floorf(check_x[c] / cell_size);
            int gyi = (int)floorf(check_y[c] / cell_size);
            if (gxi < 0 || gxi >= grid_x || gyi < 0 || gyi >= grid_y) { collision = 1; break; }
            if (d_obstacle_grid[gyi * grid_x + gxi]) { collision = 1; break; }
        }
        if (collision) break;
    }

    d_children[sid].x = x;
    d_children[sid].y = y;
    d_children[sid].theta = theta;
    d_children[sid].cost = parent_g + traveled;
    d_children[sid].valid = collision ? 0 : 1;

    int gix = (int)floorf(x / cell_size);
    int giy = (int)floorf(y / cell_size);
    float h = 1e9f;
    if (gix >= 0 && gix < grid_x && giy >= 0 && giy < grid_y) {
        h = d_heuristic[giy * grid_x + gix];
    }
    d_children[sid].f_cost = d_children[sid].cost + h;
    d_children[sid].grid_idx = discretize(x, y, theta, grid_x, grid_y, n_theta);
}

// -------------------------------------------------------------------------
// CPU expansion: sequential evaluation of all steer angles
// -------------------------------------------------------------------------
void expand_node_cpu(
    ChildResult* children,
    float parent_x, float parent_y, float parent_theta,
    float parent_g,
    const int* obstacle_grid,
    const float* heuristic,
    int grid_x, int grid_y)
{
    for (int sid = 0; sid < N_STEER; sid++) {
        float steer = -MAX_STEER + (2.0f * MAX_STEER * sid) / (N_STEER - 1);

        float x = parent_x;
        float y = parent_y;
        float theta = parent_theta;
        float ds = STEP_SIZE / N_SIM_STEPS;
        int collision = 0;
        float traveled = 0.0f;

        for (int s = 0; s < N_SIM_STEPS; s++) {
            x += ds * cosf(theta);
            y += ds * sinf(theta);
            theta += ds * tanf(steer) / WHEELBASE;
            traveled += ds;
            theta = fmodf(theta, 2.0f * (float)M_PI);
            if (theta < 0) theta += 2.0f * (float)M_PI;

            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            float hl = VEHICLE_LENGTH * 0.5f;
            float hw = VEHICLE_WIDTH * 0.5f;
            float check_x[5], check_y[5];
            check_x[0] = x + hl * cos_t - hw * sin_t;
            check_y[0] = y + hl * sin_t + hw * cos_t;
            check_x[1] = x + hl * cos_t + hw * sin_t;
            check_y[1] = y + hl * sin_t - hw * cos_t;
            check_x[2] = x - hl * cos_t - hw * sin_t;
            check_y[2] = y - hl * sin_t + hw * cos_t;
            check_x[3] = x - hl * cos_t + hw * sin_t;
            check_y[3] = y - hl * sin_t - hw * cos_t;
            check_x[4] = x;
            check_y[4] = y;

            for (int c = 0; c < 5; c++) {
                int gxi = (int)floorf(check_x[c] / CELL_SIZE);
                int gyi = (int)floorf(check_y[c] / CELL_SIZE);
                if (gxi < 0 || gxi >= grid_x || gyi < 0 || gyi >= grid_y) { collision = 1; break; }
                if (obstacle_grid[gyi * grid_x + gxi]) { collision = 1; break; }
            }
            if (collision) break;
        }

        children[sid].x = x;
        children[sid].y = y;
        children[sid].theta = theta;
        children[sid].cost = parent_g + traveled;
        children[sid].valid = collision ? 0 : 1;

        int gix = (int)floorf(x / CELL_SIZE);
        int giy = (int)floorf(y / CELL_SIZE);
        float h = 1e9f;
        if (gix >= 0 && gix < grid_x && giy >= 0 && giy < grid_y) {
            h = heuristic[giy * grid_x + gix];
        }
        children[sid].f_cost = children[sid].cost + h;
        children[sid].grid_idx = discretize(x, y, theta, grid_x, grid_y, N_THETA);
    }
}

// -------------------------------------------------------------------------
// Obstacles
// -------------------------------------------------------------------------
void setup_obstacles(vector<float>& ox, vector<float>& oy) {
    for (float i = 0; i <= 100; i += 0.5f) {
        ox.push_back(i); oy.push_back(0.0f);
        ox.push_back(i); oy.push_back(100.0f);
        ox.push_back(0.0f); oy.push_back(i);
        ox.push_back(100.0f); oy.push_back(i);
    }
    for (float x = 0; x <= 70; x += 0.5f) { ox.push_back(x); oy.push_back(30.0f); }
    for (float x = 30; x <= 100; x += 0.5f) { ox.push_back(x); oy.push_back(60.0f); }
    for (float y = 0; y <= 20; y += 0.5f) { ox.push_back(50.0f); oy.push_back(y); }
    for (float y = 40; y <= 60; y += 0.5f) { ox.push_back(30.0f); oy.push_back(y); }
    for (float y = 60; y <= 85; y += 0.5f) { ox.push_back(70.0f); oy.push_back(y); }
}

// -------------------------------------------------------------------------
// Compute heuristic on GPU
// -------------------------------------------------------------------------
void compute_heuristic_gpu(
    float* d_heuristic,
    const int* d_obstacle_grid,
    float goal_x, float goal_y)
{
    int total = GRID_X * GRID_Y;
    vector<float> h_heuristic(total, INF_COST);
    int gx = (int)floorf(goal_x / CELL_SIZE);
    int gy = (int)floorf(goal_y / CELL_SIZE);
    if (gx >= 0 && gx < GRID_X && gy >= 0 && gy < GRID_Y)
        h_heuristic[gy * GRID_X + gx] = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_heuristic, h_heuristic.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    int* d_wavefront = nullptr;
    int* d_wavefront_next = nullptr;
    int* d_changed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_wavefront, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_wavefront_next, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    vector<int> h_wave(total, -1);
    if (gx >= 0 && gx < GRID_X && gy >= 0 && gy < GRID_Y)
        h_wave[gy * GRID_X + gx] = 0;
    CUDA_CHECK(cudaMemcpy(d_wavefront, h_wave.data(), total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wavefront_next, h_wave.data(), total * sizeof(int), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((GRID_X + 15) / 16, (GRID_Y + 15) / 16);

    for (int iter = 0; iter < GRID_X + GRID_Y; iter++) {
        int h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        compute_heuristic_kernel<<<grid, block>>>(
            d_heuristic, d_obstacle_grid,
            d_wavefront, d_wavefront_next, d_changed,
            GRID_X, GRID_Y, CELL_SIZE, iter);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        if (!h_changed) break;
        CUDA_CHECK(cudaMemcpy(d_wavefront, d_wavefront_next, total * sizeof(int), cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_wavefront));
    CUDA_CHECK(cudaFree(d_wavefront_next));
    CUDA_CHECK(cudaFree(d_changed));
}

// -------------------------------------------------------------------------
// Draw car
// -------------------------------------------------------------------------
void draw_car(cv::Mat& img, float x, float y, float theta,
              int img_reso, cv::Scalar color, int thickness = 1) {
    float hl = VEHICLE_LENGTH * 0.5f;
    float hw = VEHICLE_WIDTH * 0.5f;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    cv::Point2f corners[4];
    corners[0] = cv::Point2f((x + hl * cos_t - hw * sin_t) * img_reso,
                             (y + hl * sin_t + hw * cos_t) * img_reso);
    corners[1] = cv::Point2f((x + hl * cos_t + hw * sin_t) * img_reso,
                             (y + hl * sin_t - hw * cos_t) * img_reso);
    corners[2] = cv::Point2f((x - hl * cos_t + hw * sin_t) * img_reso,
                             (y - hl * sin_t - hw * cos_t) * img_reso);
    corners[3] = cv::Point2f((x - hl * cos_t - hw * sin_t) * img_reso,
                             (y - hl * sin_t + hw * cos_t) * img_reso);
    for (int i = 0; i < 4; i++)
        cv::line(img, corners[i], corners[(i + 1) % 4], color, thickness);
}

// -------------------------------------------------------------------------
// Generic Hybrid A* search that takes an expansion function
// Returns: (closed_list, goal_closed_idx, total_expand_time_ms)
// -------------------------------------------------------------------------
struct SearchResult {
    vector<HybridNode> closed_list;
    int goal_idx;
    bool found;
    int expanded;
    double expand_time_ms;
    vector<pair<float, float>> explored_points;  // for drawing
};

typedef void (*ExpandFunc)(
    ChildResult* children,
    float px, float py, float pt, float pg,
    const int* obs_grid, const float* heuristic,
    int gx, int gy, void* extra);

// -------------------------------------------------------------------------
// Run Hybrid A* with given expand function
// -------------------------------------------------------------------------
SearchResult run_hybrid_astar(
    float sx, float sy, float stheta,
    float goal_x, float goal_y, float goal_theta,
    const int* obstacle_grid,
    const float* heuristic,
    ExpandFunc expand_fn,
    void* expand_extra)
{
    SearchResult result;
    result.found = false;
    result.goal_idx = -1;
    result.expanded = 0;
    result.expand_time_ms = 0.0;

    int total_3d = GRID_X * GRID_Y * N_THETA;
    vector<float> best_cost(total_3d, INF_COST);
    vector<int> visited(total_3d, 0);

    auto cmp = [](const HybridNode& a, const HybridNode& b) { return a.f_cost > b.f_cost; };
    priority_queue<HybridNode, vector<HybridNode>, decltype(cmp)> open(cmp);

    HybridNode start_node;
    start_node.x = sx; start_node.y = sy; start_node.theta = stheta;
    start_node.g_cost = 0.0f; start_node.parent_idx = -1;
    int six = (int)floorf(sx / CELL_SIZE);
    int siy = (int)floorf(sy / CELL_SIZE);
    start_node.f_cost = (six >= 0 && six < GRID_X && siy >= 0 && siy < GRID_Y)
                        ? heuristic[siy * GRID_X + six] : INF_COST;
    start_node.grid_idx = discretize(sx, sy, stheta, GRID_X, GRID_Y, N_THETA);

    open.push(start_node);
    if (start_node.grid_idx >= 0) best_cost[start_node.grid_idx] = 0.0f;

    ChildResult children[N_STEER];
    float goal_dist_thresh = 3.0f;
    float goal_theta_thresh = 15.0f * M_PI / 180.0f;

    while (!open.empty()) {
        HybridNode current = open.top();
        open.pop();

        if (current.grid_idx >= 0 && current.grid_idx < total_3d) {
            if (visited[current.grid_idx]) continue;
            visited[current.grid_idx] = 1;
        }

        int current_closed_idx = (int)result.closed_list.size();
        result.closed_list.push_back(current);
        result.expanded++;
        result.explored_points.push_back({current.x, current.y});

        // Check goal
        float dx = current.x - goal_x;
        float dy = current.y - goal_y;
        if (sqrtf(dx * dx + dy * dy) < goal_dist_thresh) {
            float dtheta = fabsf(current.theta - goal_theta);
            if (dtheta > M_PI) dtheta = 2.0f * M_PI - dtheta;
            if (dtheta < goal_theta_thresh) {
                result.found = true;
                result.goal_idx = current_closed_idx;
                break;
            }
        }

        // Expand
        auto t0 = chrono::high_resolution_clock::now();
        expand_fn(children, current.x, current.y, current.theta, current.g_cost,
                  obstacle_grid, heuristic, GRID_X, GRID_Y, expand_extra);
        auto t1 = chrono::high_resolution_clock::now();
        result.expand_time_ms += chrono::duration<double, milli>(t1 - t0).count();

        for (int i = 0; i < N_STEER; i++) {
            if (!children[i].valid) continue;
            int gidx = children[i].grid_idx;
            if (gidx < 0 || gidx >= total_3d) continue;
            if (visited[gidx]) continue;
            if (children[i].cost >= best_cost[gidx]) continue;
            best_cost[gidx] = children[i].cost;

            HybridNode child;
            child.x = children[i].x; child.y = children[i].y; child.theta = children[i].theta;
            child.g_cost = children[i].cost; child.f_cost = children[i].f_cost;
            child.parent_idx = current_closed_idx;
            child.grid_idx = gidx;
            open.push(child);
        }
    }

    return result;
}

// -------------------------------------------------------------------------
// CPU expand wrapper
// -------------------------------------------------------------------------
void cpu_expand_wrapper(
    ChildResult* children,
    float px, float py, float pt, float pg,
    const int* obs_grid, const float* heuristic,
    int gx, int gy, void* extra)
{
    expand_node_cpu(children, px, py, pt, pg, obs_grid, heuristic, gx, gy);
}

// -------------------------------------------------------------------------
// CUDA expand wrapper
// -------------------------------------------------------------------------
struct CudaExpandData {
    ChildResult* d_children;
    int* d_obstacle_grid;
    float* d_heuristic;
};

void cuda_expand_wrapper(
    ChildResult* children,
    float px, float py, float pt, float pg,
    const int* obs_grid, const float* heuristic,
    int gx, int gy, void* extra)
{
    CudaExpandData* data = (CudaExpandData*)extra;

    expand_node_batch_kernel<<<1, N_STEER>>>(
        data->d_children,
        px, py, pt, pg,
        data->d_obstacle_grid, data->d_heuristic,
        gx, gy, CELL_SIZE,
        WHEELBASE, MAX_STEER, N_STEER,
        STEP_SIZE, N_SIM_STEPS,
        VEHICLE_LENGTH, VEHICLE_WIDTH,
        N_THETA);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(children, data->d_children,
                          N_STEER * sizeof(ChildResult), cudaMemcpyDeviceToHost));
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    float sx = 10.0f, sy = 10.0f, stheta = 0.0f;
    float gx = 85.0f, gy = 85.0f, gtheta = M_PI / 2.0f;

    printf("Hybrid A* Comparison: CPU vs CUDA\n");

    // Setup obstacles
    vector<float> ox, oy;
    setup_obstacles(ox, oy);
    int n_obs = (int)ox.size();

    // Allocate GPU memory
    float* d_ox = nullptr;
    float* d_oy = nullptr;
    int* d_obstacle_grid = nullptr;
    float* d_heuristic = nullptr;
    ChildResult* d_children = nullptr;

    CUDA_CHECK(cudaMalloc(&d_ox, n_obs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_oy, n_obs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obstacle_grid, GRID_X * GRID_Y * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_heuristic, GRID_X * GRID_Y * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_children, N_STEER * sizeof(ChildResult)));

    CUDA_CHECK(cudaMemcpy(d_ox, ox.data(), n_obs * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oy, oy.data(), n_obs * sizeof(float), cudaMemcpyHostToDevice));

    // Build obstacle grid
    {
        dim3 block(16, 16);
        dim3 grid((GRID_X + 15) / 16, (GRID_Y + 15) / 16);
        build_obstacle_grid_kernel<<<grid, block>>>(
            d_obstacle_grid, d_ox, d_oy, n_obs,
            GRID_X, GRID_Y, CELL_SIZE, VEHICLE_WIDTH);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    vector<int> h_obstacle_grid(GRID_X * GRID_Y);
    CUDA_CHECK(cudaMemcpy(h_obstacle_grid.data(), d_obstacle_grid,
                          GRID_X * GRID_Y * sizeof(int), cudaMemcpyDeviceToHost));

    // Compute heuristic
    compute_heuristic_gpu(d_heuristic, d_obstacle_grid, gx, gy);

    vector<float> h_heuristic(GRID_X * GRID_Y);
    CUDA_CHECK(cudaMemcpy(h_heuristic.data(), d_heuristic,
                          GRID_X * GRID_Y * sizeof(float), cudaMemcpyDeviceToHost));

    // ========== Run CPU search ==========
    printf("Running CPU Hybrid A*...\n");
    auto t0_cpu = chrono::high_resolution_clock::now();
    SearchResult cpu_result = run_hybrid_astar(
        sx, sy, stheta, gx, gy, gtheta,
        h_obstacle_grid.data(), h_heuristic.data(),
        cpu_expand_wrapper, nullptr);
    auto t1_cpu = chrono::high_resolution_clock::now();
    double cpu_total_ms = chrono::duration<double, milli>(t1_cpu - t0_cpu).count();

    // ========== Run CUDA search ==========
    printf("Running CUDA Hybrid A*...\n");
    CudaExpandData cuda_data;
    cuda_data.d_children = d_children;
    cuda_data.d_obstacle_grid = d_obstacle_grid;
    cuda_data.d_heuristic = d_heuristic;

    auto t0_cuda = chrono::high_resolution_clock::now();
    SearchResult cuda_result = run_hybrid_astar(
        sx, sy, stheta, gx, gy, gtheta,
        h_obstacle_grid.data(), h_heuristic.data(),
        cuda_expand_wrapper, &cuda_data);
    auto t1_cuda = chrono::high_resolution_clock::now();
    double cuda_total_ms = chrono::duration<double, milli>(t1_cuda - t0_cuda).count();

    // Print results
    printf("\n=== Results ===\n");
    printf("CPU:  %s, %d nodes expanded, total %.2f ms, expansion %.2f ms\n",
           cpu_result.found ? "FOUND" : "NOT FOUND",
           cpu_result.expanded, cpu_total_ms, cpu_result.expand_time_ms);
    printf("CUDA: %s, %d nodes expanded, total %.2f ms, expansion %.2f ms\n",
           cuda_result.found ? "FOUND" : "NOT FOUND",
           cuda_result.expanded, cuda_total_ms, cuda_result.expand_time_ms);
    printf("Expansion speedup: %.2fx\n", cpu_result.expand_time_ms / cuda_result.expand_time_ms);
    printf("Total speedup:     %.2fx\n", cpu_total_ms / cuda_total_ms);

    // ========== Visualization: side-by-side animated GIF ==========
    int img_reso = 6;
    int panel_w = GRID_X * img_reso;
    int panel_h = GRID_Y * img_reso;

    cv::Mat bg_cpu(panel_h, panel_w, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat bg_cuda(panel_h, panel_w, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw obstacles on both panels
    for (int iy = 0; iy < GRID_Y; iy++) {
        for (int ix = 0; ix < GRID_X; ix++) {
            if (h_obstacle_grid[iy * GRID_X + ix]) {
                cv::rectangle(bg_cpu,
                              cv::Point(ix * img_reso, iy * img_reso),
                              cv::Point((ix + 1) * img_reso, (iy + 1) * img_reso),
                              cv::Scalar(0, 0, 0), -1);
                cv::rectangle(bg_cuda,
                              cv::Point(ix * img_reso, iy * img_reso),
                              cv::Point((ix + 1) * img_reso, (iy + 1) * img_reso),
                              cv::Scalar(0, 0, 0), -1);
            }
        }
    }

    // Draw start/goal on both
    for (auto* bg : {&bg_cpu, &bg_cuda}) {
        cv::circle(*bg, cv::Point((int)(sx * img_reso), (int)(sy * img_reso)),
                   img_reso * 2, cv::Scalar(0, 200, 0), -1);
        cv::circle(*bg, cv::Point((int)(gx * img_reso), (int)(gy * img_reso)),
                   img_reso * 2, cv::Scalar(255, 0, 0), -1);
    }

    cv::VideoWriter video("gif/comparison_hybrid_astar.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
                          cv::Size(panel_w * 2, panel_h));

    int max_explored = max((int)cpu_result.explored_points.size(),
                           (int)cuda_result.explored_points.size());
    int points_per_frame = max(1, max_explored / 300);  // ~300 frames

    int cpu_idx = 0, cuda_idx = 0;
    int frame_count = 0;

    while (cpu_idx < (int)cpu_result.explored_points.size() ||
           cuda_idx < (int)cuda_result.explored_points.size()) {

        for (int p = 0; p < points_per_frame; p++) {
            if (cpu_idx < (int)cpu_result.explored_points.size()) {
                auto& pt = cpu_result.explored_points[cpu_idx++];
                int px = (int)(pt.first * img_reso);
                int py = (int)(pt.second * img_reso);
                if (px >= 0 && px < panel_w && py >= 0 && py < panel_h)
                    cv::circle(bg_cpu, cv::Point(px, py), 1, cv::Scalar(0, 200, 0), -1);
            }
            if (cuda_idx < (int)cuda_result.explored_points.size()) {
                auto& pt = cuda_result.explored_points[cuda_idx++];
                int px = (int)(pt.first * img_reso);
                int py = (int)(pt.second * img_reso);
                if (px >= 0 && px < panel_w && py >= 0 && py < panel_h)
                    cv::circle(bg_cuda, cv::Point(px, py), 1, cv::Scalar(0, 200, 0), -1);
            }
        }

        cv::Mat left = bg_cpu.clone();
        cv::Mat right = bg_cuda.clone();
        char buf[128];

        cv::putText(left, "CPU", cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        snprintf(buf, sizeof(buf), "Expand: %.2f ms", cpu_result.expand_time_ms);
        cv::putText(left, buf, cv::Point(10, 42),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);
        snprintf(buf, sizeof(buf), "Total: %.2f ms", cpu_total_ms);
        cv::putText(left, buf, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);

        cv::putText(right, "CUDA", cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        snprintf(buf, sizeof(buf), "Expand: %.2f ms", cuda_result.expand_time_ms);
        cv::putText(right, buf, cv::Point(10, 42),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);
        snprintf(buf, sizeof(buf), "Total: %.2f ms", cuda_total_ms);
        cv::putText(right, buf, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        video.write(combined);
        frame_count++;
    }

    // Draw final paths
    auto draw_path = [&](cv::Mat& img, SearchResult& res) {
        if (!res.found) return;
        vector<HybridNode> path;
        int idx = res.goal_idx;
        while (idx >= 0) {
            path.push_back(res.closed_list[idx]);
            idx = res.closed_list[idx].parent_idx;
        }
        for (int i = (int)path.size() - 1; i >= 0; i--) {
            if (i < (int)path.size() - 1) {
                cv::line(img,
                         cv::Point((int)(path[i].x * img_reso), (int)(path[i].y * img_reso)),
                         cv::Point((int)(path[i + 1].x * img_reso), (int)(path[i + 1].y * img_reso)),
                         cv::Scalar(0, 0, 255), 2);
            }
            if (i % 3 == 0)
                draw_car(img, path[i].x, path[i].y, path[i].theta, img_reso, cv::Scalar(0, 0, 200), 1);
        }
    };

    draw_path(bg_cpu, cpu_result);
    draw_path(bg_cuda, cuda_result);

    // Hold final frame
    {
        cv::Mat left = bg_cpu.clone();
        cv::Mat right = bg_cuda.clone();
        char buf[128];

        cv::putText(left, "CPU", cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        snprintf(buf, sizeof(buf), "Expand: %.2f ms", cpu_result.expand_time_ms);
        cv::putText(left, buf, cv::Point(10, 42),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);
        snprintf(buf, sizeof(buf), "Total: %.2f ms", cpu_total_ms);
        cv::putText(left, buf, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);
        if (cpu_result.found)
            cv::putText(left, "PATH FOUND", cv::Point(10, 80),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 180, 0), 1);

        cv::putText(right, "CUDA", cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        snprintf(buf, sizeof(buf), "Expand: %.2f ms", cuda_result.expand_time_ms);
        cv::putText(right, buf, cv::Point(10, 42),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);
        snprintf(buf, sizeof(buf), "Total: %.2f ms", cuda_total_ms);
        cv::putText(right, buf, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);
        snprintf(buf, sizeof(buf), "Speedup: %.2fx", cpu_total_ms / cuda_total_ms);
        cv::putText(right, buf, cv::Point(10, 80),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 0, 0), 1);
        if (cuda_result.found)
            cv::putText(right, "PATH FOUND", cv::Point(10, 100),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 180, 0), 1);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        for (int f = 0; f < 60; f++) video.write(combined);
    }

    video.release();
    printf("Video saved to gif/comparison_hybrid_astar.avi (%d frames)\n", frame_count);

    system("ffmpeg -y -i gif/comparison_hybrid_astar.avi "
           "-vf 'fps=15,scale=1200:-1:flags=lanczos' -loop 0 "
           "gif/comparison_hybrid_astar.gif 2>/dev/null");
    printf("GIF saved to gif/comparison_hybrid_astar.gif\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_ox));
    CUDA_CHECK(cudaFree(d_oy));
    CUDA_CHECK(cudaFree(d_obstacle_grid));
    CUDA_CHECK(cudaFree(d_heuristic));
    CUDA_CHECK(cudaFree(d_children));

    return 0;
}
