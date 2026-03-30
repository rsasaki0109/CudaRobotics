/*************************************************************************
    > File Name: hybrid_astar.cu
    > CUDA-parallelized Hybrid A* path planning
    > Searches in (x, y, theta) continuous space with vehicle kinematics
    > GPU kernels:
    >   - build_obstacle_grid_kernel: build 2D obstacle grid
    >   - compute_heuristic_kernel: BFS heuristic from goal
    >   - expand_node_batch_kernel: evaluate all steer angles in parallel
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
static const int N_THETA = 72;           // 5 degree bins
static const float CELL_SIZE = 1.0f;     // 1m per cell
static const float WHEELBASE = 2.5f;
static const float MAX_STEER = 30.0f * M_PI / 180.0f;
static const int N_STEER = 21;
static const float STEP_SIZE = 3.0f;     // forward step per expansion
static const int N_SIM_STEPS = 10;       // sub-steps for bicycle model simulation
static const float VEHICLE_LENGTH = 4.0f;
static const float VEHICLE_WIDTH = 2.0f;
static const float INF_COST = 1e9f;

// -------------------------------------------------------------------------
// Hybrid A* Node (CPU side)
// -------------------------------------------------------------------------
struct HybridNode {
    float x, y, theta;
    float g_cost;
    float f_cost;
    int parent_idx;       // index in closed list, -1 for start
    int grid_idx;         // discretized index for visited checking

    bool operator>(const HybridNode& o) const { return f_cost > o.f_cost; }
};

// -------------------------------------------------------------------------
// Discretize a continuous state to grid index
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
// CUDA kernel: build 2D obstacle grid (1 thread per cell)
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

// -------------------------------------------------------------------------
// CUDA kernel: compute 2D heuristic via BFS from goal (iterative wavefront)
// Each thread processes one cell per iteration
// -------------------------------------------------------------------------
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

    // 8-connected neighbors
    const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const float dc[] = {1.414f, 1.0f, 1.414f, 1.0f, 1.0f, 1.414f, 1.0f, 1.414f};

    for (int d = 0; d < 8; d++) {
        int nx = ix + dx[d];
        int ny = iy + dy[d];
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

// -------------------------------------------------------------------------
// CUDA kernel: expand one node with all steer angles in parallel
// Each thread: one steer angle, simulate bicycle model, check collision
// -------------------------------------------------------------------------
struct ChildResult {
    float x, y, theta;
    float cost;           // g-cost of child
    float f_cost;         // f = g + h
    int valid;            // 1 if collision-free
    int grid_idx;
};

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

    // Compute steer angle for this thread
    float steer = -max_steer + (2.0f * max_steer * sid) / (n_steer - 1);

    float x = parent_x;
    float y = parent_y;
    float theta = parent_theta;
    float ds = step_size / n_sim_steps;
    int collision = 0;
    float traveled = 0.0f;

    for (int s = 0; s < n_sim_steps; s++) {
        // Bicycle model
        x += ds * cosf(theta);
        y += ds * sinf(theta);
        theta += ds * tanf(steer) / wheelbase;
        traveled += ds;

        // Normalize theta
        theta = fmodf(theta, 2.0f * M_PI);
        if (theta < 0) theta += 2.0f * M_PI;

        // Check collision: check rectangle footprint of vehicle
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        // Check 4 corners + center of vehicle
        float check_x[5], check_y[5];
        float hl = vehicle_length * 0.5f;
        float hw = vehicle_width * 0.5f;
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
            int gx = (int)floorf(check_x[c] / cell_size);
            int gy = (int)floorf(check_y[c] / cell_size);
            if (gx < 0 || gx >= grid_x || gy < 0 || gy >= grid_y) {
                collision = 1;
                break;
            }
            if (d_obstacle_grid[gy * grid_x + gx]) {
                collision = 1;
                break;
            }
        }
        if (collision) break;
    }

    d_children[sid].x = x;
    d_children[sid].y = y;
    d_children[sid].theta = theta;
    d_children[sid].cost = parent_g + traveled;
    d_children[sid].valid = collision ? 0 : 1;

    // Compute heuristic from 2D map
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
// Build obstacle list for maze-like parking lot
// -------------------------------------------------------------------------
void setup_obstacles(vector<float>& ox, vector<float>& oy) {
    // Boundary walls
    for (float i = 0; i <= 100; i += 0.5f) {
        ox.push_back(i); oy.push_back(0.0f);
        ox.push_back(i); oy.push_back(100.0f);
        ox.push_back(0.0f); oy.push_back(i);
        ox.push_back(100.0f); oy.push_back(i);
    }
    // Internal walls creating maze-like corridors
    // Wall 1: horizontal at y=30, from x=0 to x=70
    for (float x = 0; x <= 70; x += 0.5f) {
        ox.push_back(x); oy.push_back(30.0f);
    }
    // Wall 2: horizontal at y=60, from x=30 to x=100
    for (float x = 30; x <= 100; x += 0.5f) {
        ox.push_back(x); oy.push_back(60.0f);
    }
    // Wall 3: vertical at x=50, from y=0 to y=20
    for (float y = 0; y <= 20; y += 0.5f) {
        ox.push_back(50.0f); oy.push_back(y);
    }
    // Wall 4: vertical at x=30, from y=40 to y=60
    for (float y = 40; y <= 60; y += 0.5f) {
        ox.push_back(30.0f); oy.push_back(y);
    }
    // Wall 5: vertical at x=70, from y=60 to y=85
    for (float y = 60; y <= 85; y += 0.5f) {
        ox.push_back(70.0f); oy.push_back(y);
    }
}

// -------------------------------------------------------------------------
// Compute 2D heuristic using GPU BFS
// -------------------------------------------------------------------------
void compute_heuristic_gpu(
    float* d_heuristic,
    const int* d_obstacle_grid,
    float goal_x, float goal_y)
{
    int total = GRID_X * GRID_Y;

    // Initialize heuristic to INF
    vector<float> h_heuristic(total, INF_COST);
    int gx = (int)floorf(goal_x / CELL_SIZE);
    int gy = (int)floorf(goal_y / CELL_SIZE);
    if (gx >= 0 && gx < GRID_X && gy >= 0 && gy < GRID_Y) {
        h_heuristic[gy * GRID_X + gx] = 0.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_heuristic, h_heuristic.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    // Wavefront arrays
    int* d_wavefront = nullptr;
    int* d_wavefront_next = nullptr;
    int* d_changed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_wavefront, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_wavefront_next, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    // Initialize wavefront: goal cell = iteration 0, rest = -1
    vector<int> h_wave(total, -1);
    if (gx >= 0 && gx < GRID_X && gy >= 0 && gy < GRID_Y) {
        h_wave[gy * GRID_X + gx] = 0;
    }
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

        // Swap wavefronts
        CUDA_CHECK(cudaMemcpy(d_wavefront, d_wavefront_next, total * sizeof(int), cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_wavefront));
    CUDA_CHECK(cudaFree(d_wavefront_next));
    CUDA_CHECK(cudaFree(d_changed));
}

// -------------------------------------------------------------------------
// Draw car rectangle on image
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

    for (int i = 0; i < 4; i++) {
        cv::line(img, corners[i], corners[(i + 1) % 4], color, thickness);
    }
    // Draw heading arrow
    cv::Point2f center(x * img_reso, y * img_reso);
    cv::Point2f front(center.x + hl * cos_t * img_reso * 0.5f,
                      center.y + hl * sin_t * img_reso * 0.5f);
    cv::arrowedLine(img, center, front, color, thickness);
}

// -------------------------------------------------------------------------
// Hybrid A* planning with CUDA-parallel expansion
// -------------------------------------------------------------------------
void hybrid_astar_planning(float sx, float sy, float stheta,
                           float gx, float gy, float gtheta) {
    // Setup obstacles
    vector<float> ox, oy;
    setup_obstacles(ox, oy);
    int n_obs = (int)ox.size();

    // Allocate GPU obstacle data
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

    // Build obstacle grid on GPU
    {
        dim3 block(16, 16);
        dim3 grid((GRID_X + 15) / 16, (GRID_Y + 15) / 16);
        build_obstacle_grid_kernel<<<grid, block>>>(
            d_obstacle_grid, d_ox, d_oy, n_obs,
            GRID_X, GRID_Y, CELL_SIZE, VEHICLE_WIDTH);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy obstacle grid to host for visualization
    vector<int> h_obstacle_grid(GRID_X * GRID_Y);
    CUDA_CHECK(cudaMemcpy(h_obstacle_grid.data(), d_obstacle_grid,
                          GRID_X * GRID_Y * sizeof(int), cudaMemcpyDeviceToHost));

    // Compute 2D heuristic on GPU
    compute_heuristic_gpu(d_heuristic, d_obstacle_grid, gx, gy);

    // Visualization setup
    int img_reso = 6;
    int img_w = GRID_X * img_reso;
    int img_h = GRID_Y * img_reso;
    cv::Mat bg(img_h, img_w, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw obstacles
    for (int iy = 0; iy < GRID_Y; iy++) {
        for (int ix = 0; ix < GRID_X; ix++) {
            if (h_obstacle_grid[iy * GRID_X + ix]) {
                cv::rectangle(bg,
                              cv::Point(ix * img_reso, iy * img_reso),
                              cv::Point((ix + 1) * img_reso, (iy + 1) * img_reso),
                              cv::Scalar(0, 0, 0), -1);
            }
        }
    }

    // Draw start and goal
    cv::circle(bg, cv::Point((int)(sx * img_reso), (int)(sy * img_reso)),
               img_reso * 2, cv::Scalar(0, 200, 0), -1);
    cv::circle(bg, cv::Point((int)(gx * img_reso), (int)(gy * img_reso)),
               img_reso * 2, cv::Scalar(255, 0, 0), -1);

    cv::namedWindow("hybrid_astar", cv::WINDOW_NORMAL);
    cv::VideoWriter video("gif/hybrid_astar.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
                          cv::Size(img_w, img_h));

    // Hybrid A* search
    int total_3d = GRID_X * GRID_Y * N_THETA;
    vector<float> best_cost(total_3d, INF_COST);
    vector<int> visited(total_3d, 0);

    // Closed list stores all expanded nodes
    vector<HybridNode> closed_list;
    closed_list.reserve(100000);

    // Priority queue
    auto cmp = [](const HybridNode& a, const HybridNode& b) { return a.f_cost > b.f_cost; };
    priority_queue<HybridNode, vector<HybridNode>, decltype(cmp)> open(cmp);

    HybridNode start_node;
    start_node.x = sx;
    start_node.y = sy;
    start_node.theta = stheta;
    start_node.g_cost = 0.0f;
    start_node.parent_idx = -1;

    // Get heuristic for start
    vector<float> h_heuristic(GRID_X * GRID_Y);
    CUDA_CHECK(cudaMemcpy(h_heuristic.data(), d_heuristic,
                          GRID_X * GRID_Y * sizeof(float), cudaMemcpyDeviceToHost));

    int six = (int)floorf(sx / CELL_SIZE);
    int siy = (int)floorf(sy / CELL_SIZE);
    start_node.f_cost = (six >= 0 && six < GRID_X && siy >= 0 && siy < GRID_Y)
                        ? h_heuristic[siy * GRID_X + six] : INF_COST;
    start_node.grid_idx = discretize(sx, sy, stheta, GRID_X, GRID_Y, N_THETA);

    open.push(start_node);
    if (start_node.grid_idx >= 0) {
        best_cost[start_node.grid_idx] = 0.0f;
    }

    bool found = false;
    int goal_closed_idx = -1;
    int expanded = 0;
    int frame_count = 0;

    vector<ChildResult> h_children(N_STEER);

    float goal_dist_thresh = 3.0f;
    float goal_theta_thresh = 15.0f * M_PI / 180.0f;

    while (!open.empty()) {
        HybridNode current = open.top();
        open.pop();

        // Check if already visited
        if (current.grid_idx >= 0 && current.grid_idx < total_3d) {
            if (visited[current.grid_idx]) continue;
            visited[current.grid_idx] = 1;
        }

        // Add to closed list
        int current_closed_idx = (int)closed_list.size();
        closed_list.push_back(current);
        expanded++;

        // Check goal
        float dx = current.x - gx;
        float dy = current.y - gy;
        float dist = sqrtf(dx * dx + dy * dy);
        if (dist < goal_dist_thresh) {
            float dtheta = fabsf(current.theta - gtheta);
            if (dtheta > M_PI) dtheta = 2.0f * M_PI - dtheta;
            if (dtheta < goal_theta_thresh) {
                found = true;
                goal_closed_idx = current_closed_idx;
                break;
            }
        }

        // Draw explored cell
        int draw_ix = (int)(current.x * img_reso);
        int draw_iy = (int)(current.y * img_reso);
        if (draw_ix >= 0 && draw_ix < img_w && draw_iy >= 0 && draw_iy < img_h) {
            cv::circle(bg, cv::Point(draw_ix, draw_iy), 1, cv::Scalar(0, 200, 0), -1);
        }

        if (expanded % 20 == 0) {
            cv::imshow("hybrid_astar", bg);
            video.write(bg);
            cv::waitKey(1);
            frame_count++;
        }

        // GPU-parallel expansion: evaluate all N_STEER children
        expand_node_batch_kernel<<<1, N_STEER>>>(
            d_children,
            current.x, current.y, current.theta,
            current.g_cost,
            d_obstacle_grid, d_heuristic,
            GRID_X, GRID_Y, CELL_SIZE,
            WHEELBASE, MAX_STEER, N_STEER,
            STEP_SIZE, N_SIM_STEPS,
            VEHICLE_LENGTH, VEHICLE_WIDTH,
            N_THETA);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy children back
        CUDA_CHECK(cudaMemcpy(h_children.data(), d_children,
                              N_STEER * sizeof(ChildResult), cudaMemcpyDeviceToHost));

        // Process children
        for (int i = 0; i < N_STEER; i++) {
            if (!h_children[i].valid) continue;
            int gidx = h_children[i].grid_idx;
            if (gidx < 0 || gidx >= total_3d) continue;
            if (visited[gidx]) continue;
            if (h_children[i].cost >= best_cost[gidx]) continue;

            best_cost[gidx] = h_children[i].cost;

            HybridNode child;
            child.x = h_children[i].x;
            child.y = h_children[i].y;
            child.theta = h_children[i].theta;
            child.g_cost = h_children[i].cost;
            child.f_cost = h_children[i].f_cost;
            child.parent_idx = current_closed_idx;
            child.grid_idx = gidx;
            open.push(child);
        }
    }

    printf("Expanded %d nodes\n", expanded);

    if (found) {
        printf("Path found!\n");
        // Trace path
        vector<HybridNode> path;
        int idx = goal_closed_idx;
        while (idx >= 0) {
            path.push_back(closed_list[idx]);
            idx = closed_list[idx].parent_idx;
        }

        // Draw path
        for (int i = (int)path.size() - 1; i >= 0; i--) {
            cv::Point pt((int)(path[i].x * img_reso), (int)(path[i].y * img_reso));
            if (i < (int)path.size() - 1) {
                cv::Point pt_next((int)(path[i + 1].x * img_reso), (int)(path[i + 1].y * img_reso));
                cv::line(bg, pt, pt_next, cv::Scalar(0, 0, 255), 2);
            }
            // Draw car orientation every few nodes
            if (i % 3 == 0) {
                draw_car(bg, path[i].x, path[i].y, path[i].theta,
                         img_reso, cv::Scalar(0, 0, 200), 1);
            }
        }

        // Redraw start and goal on top
        cv::circle(bg, cv::Point((int)(sx * img_reso), (int)(sy * img_reso)),
                   img_reso * 2, cv::Scalar(0, 200, 0), -1);
        cv::circle(bg, cv::Point((int)(gx * img_reso), (int)(gy * img_reso)),
                   img_reso * 2, cv::Scalar(255, 0, 0), -1);

        printf("Path length: %d nodes\n", (int)path.size());
    } else {
        printf("No path found!\n");
    }

    // Write final frames
    for (int f = 0; f < 60; f++) video.write(bg);

    cv::imshow("hybrid_astar", bg);
    video.write(bg);
    video.release();
    printf("Video saved to gif/hybrid_astar.avi (%d frames)\n", frame_count);

    // Convert to gif
    system("ffmpeg -y -i gif/hybrid_astar.avi "
           "-vf 'fps=15,scale=600:-1:flags=lanczos' -loop 0 "
           "gif/hybrid_astar.gif 2>/dev/null");
    printf("GIF saved to gif/hybrid_astar.gif\n");

    cv::waitKey(0);

    // Cleanup
    CUDA_CHECK(cudaFree(d_ox));
    CUDA_CHECK(cudaFree(d_oy));
    CUDA_CHECK(cudaFree(d_obstacle_grid));
    CUDA_CHECK(cudaFree(d_heuristic));
    CUDA_CHECK(cudaFree(d_children));
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    float sx = 10.0f, sy = 10.0f, stheta = 0.0f;
    float gx = 85.0f, gy = 85.0f, gtheta = M_PI / 2.0f;

    printf("Hybrid A* Planner (CUDA)\n");
    printf("Start: (%.1f, %.1f, %.1f deg)\n", sx, sy, stheta * 180.0f / M_PI);
    printf("Goal:  (%.1f, %.1f, %.1f deg)\n", gx, gy, gtheta * 180.0f / M_PI);

    hybrid_astar_planning(sx, sy, stheta, gx, gy, gtheta);

    return 0;
}
