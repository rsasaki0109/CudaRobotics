/*************************************************************************
    > File Name: value_iteration.cu
    > CUDA-parallelized Brute-Force Value Iteration (Ueda JRM 2023)
    > State: 100x100x36 grid (x, y, theta), cell 0.5m, 10deg heading
    > Kernel: value_iteration_kernel (1 thread per state, Bellman update, 5 actions)
    > CPU version for comparison
    > Generates both value_iteration.gif and comparison_value_iteration.gif
 ************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cfloat>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA error check macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
#define PI 3.141592653f

#define NX 100
#define NY 100
#define NTHETA 36
#define N_STATES (NX * NY * NTHETA)

#define CELL_SIZE 0.5f
#define THETA_RES (10.0f * PI / 180.0f)

#define N_ACTIONS 5  // forward, turn_left, turn_right, forward_left, forward_right
#define GAMMA 0.95f
#define MAX_ITER 200
#define CONVERGENCE_THRESHOLD 0.01f

// Goal
#define GOAL_X 80
#define GOAL_Y 80

// Visualization
#define VIS_SCALE 6
#define IMG_W (NX * VIS_SCALE)
#define IMG_H (NY * VIS_SCALE)

// ---------------------------------------------------------------------------
// State indexing
// ---------------------------------------------------------------------------
__host__ __device__ int state_index(int ix, int iy, int itheta) {
    return (itheta * NY + iy) * NX + ix;
}

// ---------------------------------------------------------------------------
// Obstacle map (host and device)
// ---------------------------------------------------------------------------
__host__ __device__ bool is_obstacle(int ix, int iy) {
    // Boundary
    if (ix <= 0 || ix >= NX - 1 || iy <= 0 || iy >= NY - 1) return true;
    // Internal wall 1: horizontal at y=30, x=[20,60]
    if (iy >= 28 && iy <= 32 && ix >= 20 && ix <= 60) return true;
    // Internal wall 2: vertical at x=70, y=[40,80]
    if (ix >= 68 && ix <= 72 && iy >= 40 && iy <= 80) return true;
    // Internal wall 3: horizontal at y=60, x=[10,40]
    if (iy >= 58 && iy <= 62 && ix >= 10 && ix <= 40) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Transition model: deterministic motion
// Returns reward and next state
// ---------------------------------------------------------------------------
__host__ __device__ void transition(int ix, int iy, int itheta, int action,
    int* nx, int* ny, int* ntheta, float* reward)
{
    int dt = 0;
    float move_x = 0, move_y = 0;

    switch (action) {
        case 0: // forward
            move_x = cosf(itheta * THETA_RES);
            move_y = sinf(itheta * THETA_RES);
            break;
        case 1: // turn left
            dt = 1;
            break;
        case 2: // turn right
            dt = -1;
            break;
        case 3: // forward + turn left
            move_x = cosf(itheta * THETA_RES) * 0.7f;
            move_y = sinf(itheta * THETA_RES) * 0.7f;
            dt = 1;
            break;
        case 4: // forward + turn right
            move_x = cosf(itheta * THETA_RES) * 0.7f;
            move_y = sinf(itheta * THETA_RES) * 0.7f;
            dt = -1;
            break;
    }

    int nix = ix + (int)roundf(move_x);
    int niy = iy + (int)roundf(move_y);
    int nit = (itheta + dt + NTHETA) % NTHETA;

    // Clamp
    if (nix < 0) nix = 0; if (nix >= NX) nix = NX - 1;
    if (niy < 0) niy = 0; if (niy >= NY) niy = NY - 1;

    // Check obstacle
    if (is_obstacle(nix, niy)) {
        nix = ix; niy = iy; nit = itheta;
        *reward = -10.0f;
    } else if (nix == GOAL_X && niy == GOAL_Y) {
        *reward = 100.0f;
    } else {
        *reward = -0.1f;
    }

    *nx = nix; *ny = niy; *ntheta = nit;
}

// ---------------------------------------------------------------------------
// Kernel: one iteration of value iteration (Bellman update)
// ---------------------------------------------------------------------------
__global__ void value_iteration_kernel(
    const float* V_in, float* V_out, int* policy, float gamma, int n_states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) return;

    int itheta = idx / (NY * NX);
    int rem = idx % (NY * NX);
    int iy = rem / NX;
    int ix = rem % NX;

    if (is_obstacle(ix, iy)) {
        V_out[idx] = -100.0f;
        policy[idx] = 0;
        return;
    }

    // Goal state
    if (ix == GOAL_X && iy == GOAL_Y) {
        V_out[idx] = 100.0f;
        policy[idx] = 0;
        return;
    }

    float best_val = -FLT_MAX;
    int best_action = 0;

    for (int a = 0; a < N_ACTIONS; a++) {
        int nx, ny, ntheta;
        float reward;
        transition(ix, iy, itheta, a, &nx, &ny, &ntheta, &reward);

        float val = reward + gamma * V_in[state_index(nx, ny, ntheta)];
        if (val > best_val) {
            best_val = val;
            best_action = a;
        }
    }

    V_out[idx] = best_val;
    policy[idx] = best_action;
}

// ---------------------------------------------------------------------------
// CPU: one iteration of value iteration
// ---------------------------------------------------------------------------
void cpu_value_iteration(const float* V_in, float* V_out, int* policy, float gamma) {
    for (int itheta = 0; itheta < NTHETA; itheta++) {
        for (int iy = 0; iy < NY; iy++) {
            for (int ix = 0; ix < NX; ix++) {
                int idx = state_index(ix, iy, itheta);

                if (is_obstacle(ix, iy)) { V_out[idx] = -100.0f; policy[idx] = 0; continue; }
                if (ix == GOAL_X && iy == GOAL_Y) { V_out[idx] = 100.0f; policy[idx] = 0; continue; }

                float best_val = -FLT_MAX;
                int best_action = 0;
                for (int a = 0; a < N_ACTIONS; a++) {
                    int nx, ny, ntheta;
                    float reward;
                    transition(ix, iy, itheta, a, &nx, &ny, &ntheta, &reward);
                    float val = reward + gamma * V_in[state_index(nx, ny, ntheta)];
                    if (val > best_val) { best_val = val; best_action = a; }
                }
                V_out[idx] = best_val;
                policy[idx] = best_action;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host: trace optimal path from start using policy
// ---------------------------------------------------------------------------
std::vector<std::pair<int, int>> trace_path(const int* policy, int sx, int sy, int stheta) {
    std::vector<std::pair<int, int>> path;
    int ix = sx, iy = sy, itheta = stheta;
    for (int step = 0; step < 500; step++) {
        path.push_back({ix, iy});
        if (ix == GOAL_X && iy == GOAL_Y) break;

        int action = policy[state_index(ix, iy, itheta)];
        int nx, ny, ntheta;
        float reward;
        transition(ix, iy, itheta, action, &nx, &ny, &ntheta, &reward);

        if (nx == ix && ny == iy && ntheta == itheta) break; // stuck
        ix = nx; iy = ny; itheta = ntheta;
    }
    return path;
}

// ---------------------------------------------------------------------------
// Host: draw value function heatmap
// ---------------------------------------------------------------------------
void draw_value_heatmap(cv::Mat& img, const float* V, int theta_slice,
    const std::vector<std::pair<int, int>>* path, const char* label, double ms)
{
    // Find min/max for this theta slice
    float vmin = FLT_MAX, vmax = -FLT_MAX;
    for (int iy = 0; iy < NY; iy++)
        for (int ix = 0; ix < NX; ix++) {
            int idx = state_index(ix, iy, theta_slice);
            float v = V[idx];
            if (v > -90.0f) { // skip obstacle cells
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
        }
    if (vmax <= vmin) vmax = vmin + 1.0f;

    for (int iy = 0; iy < NY; iy++) {
        for (int ix = 0; ix < NX; ix++) {
            int idx = state_index(ix, iy, theta_slice);
            float v = V[idx];

            cv::Scalar color;
            if (is_obstacle(ix, iy)) {
                color = cv::Scalar(50, 50, 50);
            } else {
                float norm = (v - vmin) / (vmax - vmin);
                if (norm < 0) norm = 0; if (norm > 1) norm = 1;
                // Blue -> Red colormap
                int r = (int)(255 * norm);
                int b = (int)(255 * (1 - norm));
                color = cv::Scalar(b, 0, r);
            }

            int px = ix * VIS_SCALE;
            int py = (NY - 1 - iy) * VIS_SCALE;
            cv::rectangle(img, cv::Point(px, py),
                cv::Point(px + VIS_SCALE - 1, py + VIS_SCALE - 1), color, -1);
        }
    }

    // Goal marker
    cv::circle(img, cv::Point(GOAL_X * VIS_SCALE + VIS_SCALE / 2,
        (NY - 1 - GOAL_Y) * VIS_SCALE + VIS_SCALE / 2), 10, cv::Scalar(0, 255, 0), -1);

    // Draw path
    if (path) {
        for (size_t i = 1; i < path->size(); i++) {
            cv::Point p1((*path)[i - 1].first * VIS_SCALE + VIS_SCALE / 2,
                         (NY - 1 - (*path)[i - 1].second) * VIS_SCALE + VIS_SCALE / 2);
            cv::Point p2((*path)[i].first * VIS_SCALE + VIS_SCALE / 2,
                         (NY - 1 - (*path)[i].second) * VIS_SCALE + VIS_SCALE / 2);
            cv::line(img, p1, p2, cv::Scalar(0, 255, 255), 2);
        }
    }

    cv::putText(img, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    if (ms >= 0) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.1f ms", ms);
        cv::putText(img, buf, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 2);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "Value Iteration: Brute-Force (Ueda JRM 2023)" << std::endl;
    std::cout << "State space: " << NX << " x " << NY << " x " << NTHETA
              << " = " << N_STATES << " states" << std::endl;

    // GPU memory
    float *d_V_in, *d_V_out;
    int *d_policy;
    CUDA_CHECK(cudaMalloc(&d_V_in, N_STATES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V_out, N_STATES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_policy, N_STATES * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_V_in, 0, N_STATES * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_V_out, 0, N_STATES * sizeof(float)));

    const int threads = 256;
    const int blocks = (N_STATES + threads - 1) / threads;

    // CPU memory
    std::vector<float> cpu_V_in(N_STATES, 0), cpu_V_out(N_STATES, 0);
    std::vector<int> cpu_policy(N_STATES, 0);

    // Host readback
    std::vector<float> h_V(N_STATES);
    std::vector<int> h_policy(N_STATES);

    // Video for value iteration progress
    cv::namedWindow("value_iteration", cv::WINDOW_NORMAL);
    cv::VideoWriter video(
        "gif/value_iteration.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 10, cv::Size(IMG_W, IMG_H));

    // Video for comparison
    cv::VideoWriter comp_video(
        "gif/comparison_value_iteration.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 10, cv::Size(IMG_W * 2, IMG_H));

    double total_cpu_ms = 0, total_cuda_ms = 0;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // --- CPU ---
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_value_iteration(cpu_V_in.data(), cpu_V_out.data(), cpu_policy.data(), GAMMA);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_cpu_ms += cpu_ms;
        std::swap(cpu_V_in, cpu_V_out);

        // --- CUDA ---
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);

        value_iteration_kernel<<<blocks, threads>>>(d_V_in, d_V_out, d_policy, GAMMA, N_STATES);

        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float cuda_ms;
        cudaEventElapsedTime(&cuda_ms, start, stop);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        total_cuda_ms += cuda_ms;

        // Swap GPU buffers
        std::swap(d_V_in, d_V_out);

        // Convergence check every 10 iterations
        if (iter % 5 == 0 || iter == MAX_ITER - 1) {
            CUDA_CHECK(cudaMemcpy(h_V.data(), d_V_in, N_STATES * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_policy.data(), d_policy, N_STATES * sizeof(int), cudaMemcpyDeviceToHost));

            // Draw CUDA result
            auto path = trace_path(h_policy.data(), 10, 10, 0);
            cv::Mat frame(IMG_H, IMG_W, CV_8UC3, cv::Scalar(0, 0, 0));
            char label[64];
            snprintf(label, sizeof(label), "CUDA iter=%d", iter);
            draw_value_heatmap(frame, h_V.data(), 0, &path, label, cuda_ms);
            video.write(frame);
            cv::imshow("value_iteration", frame);
            cv::waitKey(5);

            // Comparison frame
            auto cpu_path = trace_path(cpu_policy.data(), 10, 10, 0);
            cv::Mat left(IMG_H, IMG_W, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::Mat right(IMG_H, IMG_W, CV_8UC3, cv::Scalar(0, 0, 0));
            char cpu_label[64], cuda_label[64];
            snprintf(cpu_label, sizeof(cpu_label), "CPU iter=%d", iter);
            snprintf(cuda_label, sizeof(cuda_label), "CUDA iter=%d", iter);
            draw_value_heatmap(left, cpu_V_in.data(), 0, &cpu_path, cpu_label, cpu_ms);
            draw_value_heatmap(right, h_V.data(), 0, &path, cuda_label, cuda_ms);
            cv::Mat combined;
            cv::hconcat(left, right, combined);
            comp_video.write(combined);

            printf("Iter %3d: CPU %.1fms, CUDA %.2fms\n", iter, cpu_ms, cuda_ms);
        }
    }

    video.release();
    comp_video.release();

    printf("\nTotal: CPU %.1fms, CUDA %.1fms, Speedup: %.1fx\n",
           total_cpu_ms, total_cuda_ms, total_cpu_ms / total_cuda_ms);

    system("ffmpeg -y -i gif/value_iteration.avi "
           "-vf 'fps=10,scale=600:-1:flags=lanczos' -loop 0 "
           "gif/value_iteration.gif 2>/dev/null");
    system("ffmpeg -y -i gif/comparison_value_iteration.avi "
           "-vf 'fps=10,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_value_iteration.gif 2>/dev/null");
    std::cout << "GIF saved to gif/value_iteration.gif and gif/comparison_value_iteration.gif" << std::endl;

    cudaFree(d_V_in); cudaFree(d_V_out); cudaFree(d_policy);
    return 0;
}
