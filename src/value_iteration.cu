/*************************************************************************
    Value Iteration for Mobile Robot Path Planning (CUDA)
    Based on: Ryuichi Ueda, "Implementation of brute-force value iteration
    for mobile robot path planning and obstacle bypassing" (JRM 2023)

    State space: (x, y, theta) discretized into a 3D grid
    Actions: forward, forward-left, forward-right, turn-left, turn-right
    CUDA kernel: one thread per state cell for parallel Bellman update
    Visualization: OpenCV heatmap of value function + optimal path
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------------------------------------------------------------------------
// Parameters
// -------------------------------------------------------------------------
static const int   NX          = 100;      // grid cells in x
static const int   NY          = 100;      // grid cells in y
static const int   NTHETA      = 36;       // heading bins (10 deg each)
static const int   TOTAL_CELLS = NX * NY * NTHETA;  // 360,000
static const float CELL_SIZE   = 0.5f;     // meters per cell
#define THETA_RES_VAL (2.0f * 3.14159265f / 36.0f)
static const float THETA_RES   = THETA_RES_VAL;  // ~10 degrees
static const float STEP_SIZE   = 0.5f;     // forward step in meters (= 1 cell)
static const float TURN_STEP   = THETA_RES;// turn step = 10 degrees
#define INF_COST    1.0e6f
#define EPSILON_CONV 0.01f                  // convergence threshold
static const int   MAX_ITER    = 2000;
#define SLIP_PROB   0.1f                    // probability of slipping to adjacent heading

// Goal cell (in grid coords)
#define GOAL_X 45
#define GOAL_Y 45

// Start cell (in grid coords)
#define START_X 5
#define START_Y 5
#define START_THETA 0

// Actions: 0=forward, 1=forward-left, 2=forward-right, 3=turn-left, 4=turn-right
#define NUM_ACTIONS 5

// -------------------------------------------------------------------------
// Obstacle map setup
// -------------------------------------------------------------------------
void setup_obstacle_map(int* obmap) {
    // Initialize all free
    for (int i = 0; i < NX * NY; i++) obmap[i] = 0;

    // Boundary walls
    for (int x = 0; x < NX; x++) {
        obmap[x * NY + 0]        = 1;  // bottom wall
        obmap[x * NY + (NY - 1)] = 1;  // top wall
    }
    for (int y = 0; y < NY; y++) {
        obmap[0 * NY + y]        = 1;  // left wall
        obmap[(NX - 1) * NY + y] = 1;  // right wall
    }

    // Vertical wall at x=15 from y=0 to y=30 (grid coords: x=30, y=0..60)
    {
        int gx = (int)(15.0f / CELL_SIZE);
        int y_start = 0;
        int y_end = (int)(30.0f / CELL_SIZE);
        for (int gy = y_start; gy < y_end && gy < NY; gy++) {
            if (gx >= 0 && gx < NX)
                obmap[gx * NY + gy] = 1;
        }
    }

    // Horizontal wall at y=35 from x=25 to x=45 (grid coords: y=70, x=50..90)
    {
        int gy = (int)(35.0f / CELL_SIZE);
        int x_start = (int)(25.0f / CELL_SIZE);
        int x_end = (int)(45.0f / CELL_SIZE);
        for (int gx = x_start; gx < x_end && gx < NX; gx++) {
            if (gy >= 0 && gy < NY)
                obmap[gx * NY + gy] = 1;
        }
    }

    // Box obstacle at (30,10)-(35,20) in meters -> grid (60,20)-(70,40)
    {
        int x0 = (int)(30.0f / CELL_SIZE);
        int x1 = (int)(35.0f / CELL_SIZE);
        int y0 = (int)(10.0f / CELL_SIZE);
        int y1 = (int)(20.0f / CELL_SIZE);
        for (int gx = x0; gx < x1 && gx < NX; gx++) {
            for (int gy = y0; gy < y1 && gy < NY; gy++) {
                obmap[gx * NY + gy] = 1;
            }
        }
    }
}

// -------------------------------------------------------------------------
// Transition model: compute next state for action
// Returns (nx, ny, ntheta) and whether valid
// -------------------------------------------------------------------------
__host__ __device__
void compute_next_state(int x, int y, int theta, int action,
                        int* nx, int* ny, int* ntheta) {
    // Heading angle
    float theta_res = THETA_RES_VAL;
    int dt = 0;  // delta theta in bins

    switch (action) {
        case 0: // forward
            dt = 0;
            break;
        case 1: // forward-left
            dt = 1;
            break;
        case 2: // forward-right
            dt = -1;
            break;
        case 3: // turn-left (in place)
            *nx = x;
            *ny = y;
            *ntheta = (theta + 1) % NTHETA;
            return;
        case 4: // turn-right (in place)
            *nx = x;
            *ny = y;
            *ntheta = (theta - 1 + NTHETA) % NTHETA;
            return;
    }

    // For forward actions, compute new heading then move
    int new_theta = (theta + dt + NTHETA) % NTHETA;
    float new_angle = new_theta * theta_res;
    *nx = x + (int)roundf(cosf(new_angle));
    *ny = y + (int)roundf(sinf(new_angle));
    *ntheta = new_theta;
}

// -------------------------------------------------------------------------
// CUDA Kernel: one Bellman update sweep
// Each thread handles one state (x, y, theta)
// -------------------------------------------------------------------------
__global__ void value_iteration_kernel(
    const float* V_in,
    float* V_out,
    const int* obmap,
    float* delta_out,  // per-block max delta for convergence check
    int nx_grid, int ny_grid, int ntheta_grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx_grid * ny_grid * ntheta_grid;
    if (idx >= total) return;

    // Decode index -> (x, y, theta)
    int theta = idx % ntheta_grid;
    int rem   = idx / ntheta_grid;
    int y     = rem % ny_grid;
    int x     = rem / ny_grid;

    // Skip obstacles and goal
    if (obmap[x * ny_grid + y]) {
        V_out[idx] = INF_COST;
        return;
    }

    // Goal: any theta at goal position
    if (x == GOAL_X && y == GOAL_Y) {
        V_out[idx] = 0.0f;
        return;
    }

    float best_val = INF_COST;

    // Try each action
    for (int a = 0; a < NUM_ACTIONS; a++) {
        int snx, sny, sntheta;
        compute_next_state(x, y, theta, a, &snx, &sny, &sntheta);

        // Bounds check
        if (snx < 0 || snx >= nx_grid || sny < 0 || sny >= ny_grid)
            continue;

        // Obstacle check for next state
        if (obmap[snx * ny_grid + sny])
            continue;

        // Stochastic transition: with SLIP_PROB, heading slips +/- 1
        float val = 0.0f;
        float main_prob = 1.0f - SLIP_PROB;

        // Main transition
        int main_idx = (snx * ny_grid + sny) * ntheta_grid + sntheta;
        val += main_prob * V_in[main_idx];

        // Slip left
        int slip_left_theta = (sntheta + 1) % ntheta_grid;
        int slip_left_idx = (snx * ny_grid + sny) * ntheta_grid + slip_left_theta;
        val += (SLIP_PROB / 2.0f) * V_in[slip_left_idx];

        // Slip right
        int slip_right_theta = (sntheta - 1 + ntheta_grid) % ntheta_grid;
        int slip_right_idx = (snx * ny_grid + sny) * ntheta_grid + slip_right_theta;
        val += (SLIP_PROB / 2.0f) * V_in[slip_right_idx];

        // cost(s,a) = 1 for all actions
        float total_cost = 1.0f + val;

        if (total_cost < best_val)
            best_val = total_cost;
    }

    V_out[idx] = best_val;

    // Compute per-thread delta for convergence
    float d = fabsf(V_out[idx] - V_in[idx]);

    // Use shared memory reduction for block-level max delta
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = d;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid])
            sdata[tid] = sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicMax((int*)delta_out, __float_as_int(sdata[0]));
}

// -------------------------------------------------------------------------
// CPU Value Iteration (single sweep)
// -------------------------------------------------------------------------
float cpu_value_iteration_sweep(float* V, const int* obmap) {
    float max_delta = 0.0f;

    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            for (int theta = 0; theta < NTHETA; theta++) {
                int idx = (x * NY + y) * NTHETA + theta;

                // Skip obstacles
                if (obmap[x * NY + y]) {
                    V[idx] = INF_COST;
                    continue;
                }

                // Skip goal
                if (x == GOAL_X && y == GOAL_Y) {
                    V[idx] = 0.0f;
                    continue;
                }

                float best_val = INF_COST;

                for (int a = 0; a < NUM_ACTIONS; a++) {
                    int snx, sny, sntheta;
                    compute_next_state(x, y, theta, a, &snx, &sny, &sntheta);

                    if (snx < 0 || snx >= NX || sny < 0 || sny >= NY) continue;
                    if (obmap[snx * NY + sny]) continue;

                    float val = 0.0f;
                    float main_prob = 1.0f - SLIP_PROB;

                    int main_idx = (snx * NY + sny) * NTHETA + sntheta;
                    val += main_prob * V[main_idx];

                    int slip_left_theta = (sntheta + 1) % NTHETA;
                    int slip_left_idx = (snx * NY + sny) * NTHETA + slip_left_theta;
                    val += (SLIP_PROB / 2.0f) * V[slip_left_idx];

                    int slip_right_theta = (sntheta - 1 + NTHETA) % NTHETA;
                    int slip_right_idx = (snx * NY + sny) * NTHETA + slip_right_theta;
                    val += (SLIP_PROB / 2.0f) * V[slip_right_idx];

                    float total_cost = 1.0f + val;
                    if (total_cost < best_val)
                        best_val = total_cost;
                }

                float old_val = V[idx];
                V[idx] = best_val;
                float d = fabsf(V[idx] - old_val);
                if (d > max_delta) max_delta = d;
            }
        }
    }
    return max_delta;
}

// -------------------------------------------------------------------------
// Extract optimal path via greedy policy (gradient descent on V)
// -------------------------------------------------------------------------
struct PathPoint {
    int x, y, theta;
};

std::vector<PathPoint> extract_path(const float* V, const int* obmap,
                                    int sx, int sy, int stheta, int max_steps = 500) {
    std::vector<PathPoint> path;
    int cx = sx, cy = sy, ct = stheta;
    path.push_back({cx, cy, ct});

    for (int step = 0; step < max_steps; step++) {
        if (cx == GOAL_X && cy == GOAL_Y) break;

        float best_val = INF_COST;
        int best_a = -1;

        for (int a = 0; a < NUM_ACTIONS; a++) {
            int snx, sny, sntheta;
            compute_next_state(cx, cy, ct, a, &snx, &sny, &sntheta);

            if (snx < 0 || snx >= NX || sny < 0 || sny >= NY) continue;
            if (obmap[snx * NY + sny]) continue;

            int next_idx = (snx * NY + sny) * NTHETA + sntheta;
            float val = 1.0f + V[next_idx];
            if (val < best_val) {
                best_val = val;
                best_a = a;
            }
        }

        if (best_a < 0) break;  // stuck

        int nx, ny, nt;
        compute_next_state(cx, cy, ct, best_a, &nx, &ny, &nt);
        cx = nx; cy = ny; ct = nt;
        path.push_back({cx, cy, ct});
    }
    return path;
}

// -------------------------------------------------------------------------
// Visualization: draw value function heatmap for a specific theta slice
// -------------------------------------------------------------------------
cv::Mat draw_value_heatmap(const float* V, const int* obmap, int theta_slice,
                           int scale = 5) {
    cv::Mat img(NY * scale, NX * scale, CV_8UC3, cv::Scalar(255, 255, 255));

    // Find max finite value for normalization
    float max_val = 0.0f;
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            int idx = (x * NY + y) * NTHETA + theta_slice;
            if (V[idx] < INF_COST * 0.5f && V[idx] > max_val)
                max_val = V[idx];
        }
    }
    if (max_val < 1.0f) max_val = 1.0f;

    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            cv::Rect cell(x * scale, (NY - 1 - y) * scale, scale, scale);

            if (obmap[x * NY + y]) {
                cv::rectangle(img, cell, cv::Scalar(0, 0, 0), -1);
            } else {
                int idx = (x * NY + y) * NTHETA + theta_slice;
                float v = V[idx];
                if (v >= INF_COST * 0.5f) {
                    cv::rectangle(img, cell, cv::Scalar(200, 200, 200), -1);
                } else {
                    // Color map: low value (near goal) = blue, high = red
                    float ratio = v / max_val;
                    ratio = fminf(ratio, 1.0f);
                    // Use OpenCV colormap approach: create a grayscale then apply
                    int gray = (int)(ratio * 255.0f);
                    // Manual jet-like colormap
                    int r, g, b;
                    if (gray < 64) {
                        r = 0; g = gray * 4; b = 255;
                    } else if (gray < 128) {
                        r = 0; g = 255; b = 255 - (gray - 64) * 4;
                    } else if (gray < 192) {
                        r = (gray - 128) * 4; g = 255; b = 0;
                    } else {
                        r = 255; g = 255 - (gray - 192) * 4; b = 0;
                    }
                    cv::rectangle(img, cell, cv::Scalar(b, g, r), -1);
                }
            }

            // Goal marker
            if (x == GOAL_X && y == GOAL_Y) {
                cv::rectangle(img, cell, cv::Scalar(0, 0, 255), -1);
            }
        }
    }
    return img;
}

// -------------------------------------------------------------------------
// Draw path on image
// -------------------------------------------------------------------------
void draw_path_on_image(cv::Mat& img, const std::vector<PathPoint>& path, int scale = 5) {
    for (size_t i = 1; i < path.size(); i++) {
        cv::Point p1(path[i-1].x * scale + scale / 2,
                     (NY - 1 - path[i-1].y) * scale + scale / 2);
        cv::Point p2(path[i].x * scale + scale / 2,
                     (NY - 1 - path[i].y) * scale + scale / 2);
        cv::line(img, p1, p2, cv::Scalar(255, 255, 255), 2);
    }

    // Draw start marker
    if (!path.empty()) {
        cv::Point s(path[0].x * scale + scale / 2,
                    (NY - 1 - path[0].y) * scale + scale / 2);
        cv::circle(img, s, scale, cv::Scalar(0, 255, 0), -1);
    }
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    printf("Value Iteration for Mobile Robot Path Planning (Ueda, JRM 2023)\n");
    printf("Grid: %d x %d x %d = %d states\n", NX, NY, NTHETA, TOTAL_CELLS);
    printf("Cell size: %.1f m, World: %.0f x %.0f m\n", CELL_SIZE, NX * CELL_SIZE, NY * CELL_SIZE);
    printf("Actions: forward, forward-left, forward-right, turn-left, turn-right\n");
    printf("Slip probability: %.0f%%\n", SLIP_PROB * 100);
    printf("Goal: (%d, %d), Start: (%d, %d, theta=%d)\n\n",
           GOAL_X, GOAL_Y, START_X, START_Y, START_THETA);

    // Setup obstacle map
    std::vector<int> obmap(NX * NY);
    setup_obstacle_map(obmap.data());

    // Count obstacles
    int obs_count = 0;
    for (int i = 0; i < NX * NY; i++) if (obmap[i]) obs_count++;
    printf("Obstacle cells: %d / %d (%.1f%%)\n\n", obs_count, NX * NY,
           100.0f * obs_count / (NX * NY));

    // =====================================================================
    // CUDA Value Iteration
    // =====================================================================
    CUDA_CHECK(cudaFree(0));  // warm up GPU

    // Allocate host memory
    std::vector<float> V_cuda(TOTAL_CELLS);

    // Initialize value function
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            for (int t = 0; t < NTHETA; t++) {
                int idx = (x * NY + y) * NTHETA + t;
                if (x == GOAL_X && y == GOAL_Y) {
                    V_cuda[idx] = 0.0f;
                } else if (obmap[x * NY + y]) {
                    V_cuda[idx] = INF_COST;
                } else {
                    V_cuda[idx] = INF_COST;
                }
            }
        }
    }

    // Allocate device memory
    float* d_V_in  = nullptr;
    float* d_V_out = nullptr;
    int*   d_obmap = nullptr;
    float* d_delta = nullptr;

    CUDA_CHECK(cudaMalloc(&d_V_in,  TOTAL_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V_out, TOTAL_CELLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obmap, NX * NY * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_delta, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_V_in, V_cuda.data(), TOTAL_CELLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obmap, obmap.data(), NX * NY * sizeof(int), cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (TOTAL_CELLS + block_size - 1) / block_size;
    int shared_mem = block_size * sizeof(float);

    // Video writer for convergence animation
    int vis_scale = 5;
    int img_w = NX * vis_scale;
    int img_h = NY * vis_scale;
    cv::VideoWriter video("gif/value_iteration.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
                          cv::Size(img_w, img_h));
    cv::namedWindow("value_iteration", cv::WINDOW_NORMAL);

    printf("Running CUDA Value Iteration...\n");
    auto t_cuda_start = std::chrono::high_resolution_clock::now();

    int cuda_iters = 0;
    float cuda_delta = INF_COST;
    int vis_theta = 0;  // visualize theta=0 slice

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Reset delta
        float zero = 0.0f;
        CUDA_CHECK(cudaMemcpy(d_delta, &zero, sizeof(float), cudaMemcpyHostToDevice));

        // Run kernel
        value_iteration_kernel<<<grid_size, block_size, shared_mem>>>(
            d_V_in, d_V_out, d_obmap, d_delta, NX, NY, NTHETA);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read back delta
        int delta_bits;
        CUDA_CHECK(cudaMemcpy(&delta_bits, d_delta, sizeof(int), cudaMemcpyDeviceToHost));
        // Reinterpret as float (matches __float_as_int in kernel)
        cuda_delta = *reinterpret_cast<float*>(&delta_bits);

        // Swap buffers
        float* tmp = d_V_in;
        d_V_in = d_V_out;
        d_V_out = tmp;

        cuda_iters++;

        // Visualize every 10 iterations
        if (iter % 10 == 0 || cuda_delta < EPSILON_CONV) {
            CUDA_CHECK(cudaMemcpy(V_cuda.data(), d_V_in, TOTAL_CELLS * sizeof(float), cudaMemcpyDeviceToHost));
            cv::Mat frame = draw_value_heatmap(V_cuda.data(), obmap.data(), vis_theta, vis_scale);

            char buf[128];
            snprintf(buf, sizeof(buf), "Iter: %d  Delta: %.4f", iter, cuda_delta);
            cv::putText(frame, buf, cv::Point(10, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            cv::putText(frame, "CUDA Value Iteration", cv::Point(10, img_h - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            video.write(frame);
            cv::imshow("value_iteration", frame);
            cv::waitKey(1);
        }

        if (cuda_delta < EPSILON_CONV) {
            printf("CUDA converged at iteration %d (delta=%.6f)\n", iter, cuda_delta);
            break;
        }
    }

    auto t_cuda_end = std::chrono::high_resolution_clock::now();
    double cuda_total_ms = std::chrono::duration<double, std::milli>(t_cuda_end - t_cuda_start).count();
    double cuda_per_iter = cuda_total_ms / cuda_iters;

    // Copy final result
    CUDA_CHECK(cudaMemcpy(V_cuda.data(), d_V_in, TOTAL_CELLS * sizeof(float), cudaMemcpyDeviceToHost));

    // Extract and draw optimal path
    std::vector<PathPoint> path_cuda = extract_path(V_cuda.data(), obmap.data(),
                                                     START_X, START_Y, START_THETA);

    cv::Mat final_frame = draw_value_heatmap(V_cuda.data(), obmap.data(), vis_theta, vis_scale);
    draw_path_on_image(final_frame, path_cuda, vis_scale);

    char buf[256];
    snprintf(buf, sizeof(buf), "Converged: %d iters  Path: %d steps", cuda_iters, (int)path_cuda.size());
    cv::putText(final_frame, buf, cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(final_frame, "CUDA Value Iteration", cv::Point(10, img_h - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    // Hold final frame for 3 seconds
    for (int f = 0; f < 90; f++) video.write(final_frame);
    cv::imshow("value_iteration", final_frame);
    cv::waitKey(1);

    video.release();
    printf("Video saved to gif/value_iteration.avi\n");

    // Convert to gif
    system("ffmpeg -y -i gif/value_iteration.avi "
           "-vf 'fps=15,scale=500:-1:flags=lanczos' -loop 0 "
           "gif/value_iteration.gif 2>/dev/null");
    printf("GIF saved to gif/value_iteration.gif\n\n");

    // =====================================================================
    // CPU Value Iteration (for comparison)
    // =====================================================================
    printf("Running CPU Value Iteration...\n");

    std::vector<float> V_cpu(TOTAL_CELLS);
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            for (int t = 0; t < NTHETA; t++) {
                int idx = (x * NY + y) * NTHETA + t;
                if (x == GOAL_X && y == GOAL_Y)
                    V_cpu[idx] = 0.0f;
                else
                    V_cpu[idx] = INF_COST;
            }
        }
    }

    auto t_cpu_start = std::chrono::high_resolution_clock::now();

    int cpu_iters = 0;
    float cpu_delta = INF_COST;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        cpu_delta = cpu_value_iteration_sweep(V_cpu.data(), obmap.data());
        cpu_iters++;

        if (cpu_delta < EPSILON_CONV) {
            printf("CPU converged at iteration %d (delta=%.6f)\n", iter, cpu_delta);
            break;
        }
    }

    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_total_ms = std::chrono::duration<double, std::milli>(t_cpu_end - t_cpu_start).count();
    double cpu_per_iter = cpu_total_ms / cpu_iters;

    // =====================================================================
    // Comparison video: CPU vs CUDA side by side
    // =====================================================================
    printf("\nGenerating comparison video...\n");

    // Re-run both with visualization
    // Re-initialize V arrays
    for (int i = 0; i < TOTAL_CELLS; i++) {
        int x = i / (NY * NTHETA);
        int y = (i / NTHETA) % NY;
        if (x == GOAL_X && y == GOAL_Y)
            V_cpu[i] = 0.0f;
        else
            V_cpu[i] = INF_COST;
    }

    std::vector<float> V_cuda2(TOTAL_CELLS);
    for (int i = 0; i < TOTAL_CELLS; i++) V_cuda2[i] = V_cpu[i];

    CUDA_CHECK(cudaMemcpy(d_V_in, V_cuda2.data(), TOTAL_CELLS * sizeof(float), cudaMemcpyHostToDevice));

    cv::VideoWriter comp_video("gif/comparison_value_iteration.avi",
                               cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
                               cv::Size(img_w * 2, img_h));

    int comp_iters = (cuda_iters > cpu_iters) ? cuda_iters : cpu_iters;
    // Use the actual iteration count, recording frames periodically
    int frame_interval = 5;

    for (int iter = 0; iter <= comp_iters; iter++) {
        // CPU sweep
        if (iter > 0) {
            cpu_value_iteration_sweep(V_cpu.data(), obmap.data());
        }

        // CUDA sweep
        if (iter > 0) {
            float zero = 0.0f;
            CUDA_CHECK(cudaMemcpy(d_delta, &zero, sizeof(float), cudaMemcpyHostToDevice));
            value_iteration_kernel<<<grid_size, block_size, shared_mem>>>(
                d_V_in, d_V_out, d_obmap, d_delta, NX, NY, NTHETA);
            CUDA_CHECK(cudaDeviceSynchronize());
            float* tmp = d_V_in; d_V_in = d_V_out; d_V_out = tmp;
        }

        if (iter % frame_interval == 0 || iter == comp_iters) {
            CUDA_CHECK(cudaMemcpy(V_cuda2.data(), d_V_in, TOTAL_CELLS * sizeof(float), cudaMemcpyDeviceToHost));

            cv::Mat left = draw_value_heatmap(V_cpu.data(), obmap.data(), vis_theta, vis_scale);
            cv::Mat right = draw_value_heatmap(V_cuda2.data(), obmap.data(), vis_theta, vis_scale);

            char lbuf[128], rbuf[128];
            snprintf(lbuf, sizeof(lbuf), "CPU  Iter: %d  (%.2f ms/iter)", iter, cpu_per_iter);
            snprintf(rbuf, sizeof(rbuf), "CUDA  Iter: %d  (%.2f ms/iter)", iter, cuda_per_iter);

            cv::putText(left, lbuf, cv::Point(10, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
            cv::putText(right, rbuf, cv::Point(10, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

            cv::Mat combined;
            cv::hconcat(left, right, combined);
            comp_video.write(combined);
        }
    }

    // Final frame with paths
    std::vector<PathPoint> path_cpu = extract_path(V_cpu.data(), obmap.data(),
                                                    START_X, START_Y, START_THETA);
    std::vector<PathPoint> path_cuda2 = extract_path(V_cuda2.data(), obmap.data(),
                                                      START_X, START_Y, START_THETA);

    cv::Mat left_final = draw_value_heatmap(V_cpu.data(), obmap.data(), vis_theta, vis_scale);
    cv::Mat right_final = draw_value_heatmap(V_cuda2.data(), obmap.data(), vis_theta, vis_scale);
    draw_path_on_image(left_final, path_cpu, vis_scale);
    draw_path_on_image(right_final, path_cuda2, vis_scale);

    char lbuf[256], rbuf[256];
    snprintf(lbuf, sizeof(lbuf), "CPU: %.2f ms/iter  Path: %d steps",
             cpu_per_iter, (int)path_cpu.size());
    snprintf(rbuf, sizeof(rbuf), "CUDA: %.2f ms/iter  Path: %d steps",
             cuda_per_iter, (int)path_cuda2.size());
    cv::putText(left_final, lbuf, cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    cv::putText(right_final, rbuf, cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

    snprintf(lbuf, sizeof(lbuf), "Speedup: %.2fx", cpu_per_iter / cuda_per_iter);
    cv::putText(right_final, lbuf, cv::Point(10, img_h - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

    cv::Mat combined_final;
    cv::hconcat(left_final, right_final, combined_final);
    for (int f = 0; f < 90; f++) comp_video.write(combined_final);

    comp_video.release();
    printf("Comparison video saved to gif/comparison_value_iteration.avi\n");

    system("ffmpeg -y -i gif/comparison_value_iteration.avi "
           "-vf 'fps=15,scale=1000:-1:flags=lanczos' -loop 0 "
           "gif/comparison_value_iteration.gif 2>/dev/null");
    printf("GIF saved to gif/comparison_value_iteration.gif\n\n");

    // =====================================================================
    // Print summary
    // =====================================================================
    printf("========================================\n");
    printf("Value Iteration Results\n");
    printf("========================================\n");
    printf("CPU:  %.2f ms per iteration, CUDA: %.2f ms per iteration, Speedup: %.2fx\n",
           cpu_per_iter, cuda_per_iter, cpu_per_iter / cuda_per_iter);
    printf("Converged in %d iterations (CUDA), %d iterations (CPU)\n",
           cuda_iters, cpu_iters);
    printf("CPU total:  %.2f ms\n", cpu_total_ms);
    printf("CUDA total: %.2f ms\n", cuda_total_ms);
    printf("CUDA path length: %d steps\n", (int)path_cuda.size());
    printf("CPU  path length: %d steps\n", (int)path_cpu.size());
    printf("========================================\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_V_in));
    CUDA_CHECK(cudaFree(d_V_out));
    CUDA_CHECK(cudaFree(d_obmap));
    CUDA_CHECK(cudaFree(d_delta));

    cv::destroyAllWindows();
    return 0;
}
