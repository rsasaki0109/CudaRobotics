/*************************************************************************
    > File Name: dynamic_window_approach.cu
    > CUDA-parallelized Dynamic Window Approach
    > Based on original C++ implementation by TAI Lei
    > Each (v, yawrate) sample = 1 GPU thread
 ************************************************************************/

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cfloat>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#define PI 3.141592653f

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
// Config (mirrored on host and device)
// ---------------------------------------------------------------------------
struct DWAConfig {
    float max_speed      = 1.0f;
    float min_speed      = -0.5f;
    float max_yawrate    = 40.0f * PI / 180.0f;
    float max_accel      = 0.2f;
    float robot_radius   = 1.0f;
    float max_dyawrate   = 40.0f * PI / 180.0f;
    float v_reso         = 0.01f;
    float yawrate_reso   = 0.1f * PI / 180.0f;
    float dt             = 0.1f;
    float predict_time   = 3.0f;
    float to_goal_cost_gain = 1.0f;
    float speed_cost_gain   = 1.0f;
};

using State    = std::array<float, 5>;  // x, y, yaw, v, omega
using Control  = std::array<float, 2>;
using Point    = std::array<float, 2>;
using Obstacle = std::vector<std::array<float, 2>>;
using Traj     = std::vector<std::array<float, 5>>;

// ---------------------------------------------------------------------------
// Host: motion model (for main loop state update)
// ---------------------------------------------------------------------------
State motion(State x, Control u, float dt) {
    x[2] += u[1] * dt;
    x[0] += u[0] * cosf(x[2]) * dt;
    x[1] += u[0] * sinf(x[2]) * dt;
    x[3] = u[0];
    x[4] = u[1];
    return x;
}

// ---------------------------------------------------------------------------
// Kernel: evaluate all (v, yawrate) samples in parallel
//   Each thread: simulate trajectory → compute cost → store result
// ---------------------------------------------------------------------------
__global__ void dwa_eval_kernel(
    // current state
    float sx, float sy, float syaw, float sv, float somega,
    // dynamic window bounds
    float v_min, float v_max, float yr_min, float yr_max,
    // config
    float v_reso, float yr_reso, float dt, float predict_time,
    float max_speed, float to_goal_cost_gain, float speed_cost_gain,
    float robot_radius,
    // goal
    float gx, float gy,
    // obstacles [n_ob x 2], row-major
    const float* ob, int n_ob,
    // grid dimensions
    int n_v, int n_yr,
    // output: per-sample cost and control
    float* costs,     // [n_v * n_yr]
    float* ctrl_v,    // [n_v * n_yr]
    float* ctrl_yr,   // [n_v * n_yr]
    // output: best trajectory for each sample (last point only for goal cost,
    //         but we store full traj of the best — done on host after reduction)
    float* traj_end_x,  // [n_v * n_yr]
    float* traj_end_y   // [n_v * n_yr]
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_v * n_yr;
    if (idx >= total) return;

    int iv  = idx / n_yr;
    int iyr = idx % n_yr;

    float v  = v_min + iv  * v_reso;
    float yr = yr_min + iyr * yr_reso;

    // clamp
    if (v > v_max) v = v_max;
    if (yr > yr_max) yr = yr_max;

    // store control
    ctrl_v[idx]  = v;
    ctrl_yr[idx] = yr;

    // --- simulate trajectory ---
    float px = sx, py = sy, pyaw = syaw;
    float time_sim = 0.0f;
    float minr = FLT_MAX;
    bool collision = false;
    int skip_n = 2;
    int step = 0;

    while (time_sim <= predict_time) {
        pyaw += yr * dt;
        px   += v * cosf(pyaw) * dt;
        py   += v * sinf(pyaw) * dt;
        time_sim += dt;
        step++;

        // obstacle cost (check every skip_n steps)
        if (step % skip_n == 0) {
            for (int i = 0; i < n_ob; i++) {
                float dx = px - ob[i * 2 + 0];
                float dy = py - ob[i * 2 + 1];
                float r  = sqrtf(dx * dx + dy * dy);
                if (r <= robot_radius) {
                    collision = true;
                }
                if (r < minr) minr = r;
            }
        }
    }

    // also check last point against obstacles
    for (int i = 0; i < n_ob; i++) {
        float dx = px - ob[i * 2 + 0];
        float dy = py - ob[i * 2 + 1];
        float r  = sqrtf(dx * dx + dy * dy);
        if (r <= robot_radius) collision = true;
        if (r < minr) minr = r;
    }

    traj_end_x[idx] = px;
    traj_end_y[idx] = py;

    if (collision) {
        costs[idx] = FLT_MAX;
        return;
    }

    // --- to_goal_cost ---
    float goal_mag = sqrtf(gx * gx + gy * gy);
    float traj_mag = sqrtf(px * px + py * py);
    float dot = gx * px + gy * py;
    float cos_angle = dot / (goal_mag * traj_mag + 1e-10f);
    // clamp for acos
    if (cos_angle > 1.0f) cos_angle = 1.0f;
    if (cos_angle < -1.0f) cos_angle = -1.0f;
    float to_goal_cost = to_goal_cost_gain * acosf(cos_angle);

    // --- speed_cost ---
    float speed_cost = speed_cost_gain * (max_speed - v);

    // --- obstacle_cost ---
    float ob_cost = 1.0f / minr;

    costs[idx] = to_goal_cost + speed_cost + ob_cost;
}

// ---------------------------------------------------------------------------
// Kernel: parallel reduction to find minimum cost index
// ---------------------------------------------------------------------------
__global__ void find_min_kernel(const float* costs, int* min_idx, int n) {
    extern __shared__ char smem[];
    float* sval = (float*)smem;
    int*   sidx = (int*)(smem + blockDim.x * sizeof(float));

    int tid = threadIdx.x;

    float best_val = FLT_MAX;
    int   best_idx = 0;

    for (int i = tid; i < n; i += blockDim.x) {
        if (costs[i] < best_val) {
            best_val = costs[i];
            best_idx = i;
        }
    }
    sval[tid] = best_val;
    sidx[tid] = best_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sval[tid + s] < sval[tid]) {
                sval[tid] = sval[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *min_idx = sidx[0];
    }
}

// ---------------------------------------------------------------------------
// Host: simulate trajectory for visualization (given best v, yr)
// ---------------------------------------------------------------------------
Traj calc_trajectory_host(State x, float v, float yr, const DWAConfig& cfg) {
    Traj traj;
    traj.push_back(x);
    float t = 0.0f;
    while (t <= cfg.predict_time) {
        x = motion(x, {{v, yr}}, cfg.dt);
        traj.push_back(x);
        t += cfg.dt;
    }
    return traj;
}

// ---------------------------------------------------------------------------
// Host: visualization helpers
// ---------------------------------------------------------------------------
cv::Point2i cv_offset(float x, float y,
                      int image_width = 2000, int image_height = 2000) {
    cv::Point2i output;
    output.x = int(x * 100) + image_width / 2;
    output.y = image_height - int(y * 100) - image_height / 3;
    return output;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    State x = {{0.0f, 0.0f, PI / 8.0f, 0.0f, 0.0f}};
    Point goal = {{10.0f, 10.0f}};
    Obstacle ob = {
        {{-1.0f, -1.0f}},
        {{ 0.0f,  2.0f}},
        {{ 4.0f,  2.0f}},
        {{ 5.0f,  4.0f}},
        {{ 5.0f,  5.0f}},
        {{ 5.0f,  6.0f}},
        {{ 5.0f,  9.0f}},
        {{ 8.0f,  9.0f}},
        {{ 7.0f,  9.0f}},
        {{12.0f, 12.0f}}
    };

    Control u = {{0.0f, 0.0f}};
    DWAConfig config;
    Traj traj;
    traj.push_back(x);

    // ------------------------------------------
    // GPU memory allocation
    // ------------------------------------------
    int n_ob = (int)ob.size();

    // Max possible grid size for (v, yr) samples
    int max_nv  = (int)((config.max_speed - config.min_speed) / config.v_reso) + 2;
    int max_nyr = (int)((2.0f * config.max_yawrate) / config.yawrate_reso) + 2;
    int max_samples = max_nv * max_nyr;

    float *d_ob;
    CUDA_CHECK(cudaMalloc(&d_ob, n_ob * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ob, ob.data(), n_ob * 2 * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_costs, *d_ctrl_v, *d_ctrl_yr, *d_traj_end_x, *d_traj_end_y;
    CUDA_CHECK(cudaMalloc(&d_costs,      max_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ctrl_v,     max_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ctrl_yr,    max_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_traj_end_x, max_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_traj_end_y, max_samples * sizeof(float)));

    int *d_min_idx;
    CUDA_CHECK(cudaMalloc(&d_min_idx, sizeof(int)));

    // host buffers for readback
    std::vector<float> h_ctrl_v(max_samples), h_ctrl_yr(max_samples);

    const int threads = 256;

    bool terminal = false;
    cv::namedWindow("dwa", cv::WINDOW_NORMAL);
    int count = 0;

    std::cout << "DWA with CUDA (max " << max_samples << " samples/frame)" << std::endl;

    for (int i = 0; i < 1000 && !terminal; i++) {
        // --- calc dynamic window ---
        float dw_vmin  = std::max(x[3] - config.max_accel * config.dt, config.min_speed);
        float dw_vmax  = std::min(x[3] + config.max_accel * config.dt, config.max_speed);
        float dw_yrmin = std::max(x[4] - config.max_dyawrate * config.dt, -config.max_yawrate);
        float dw_yrmax = std::min(x[4] + config.max_dyawrate * config.dt, config.max_yawrate);

        int n_v  = (int)((dw_vmax - dw_vmin) / config.v_reso) + 1;
        int n_yr = (int)((dw_yrmax - dw_yrmin) / config.yawrate_reso) + 1;
        int n_samples = n_v * n_yr;

        int blocks = (n_samples + threads - 1) / threads;

        // --- GPU: evaluate all samples ---
        dwa_eval_kernel<<<blocks, threads>>>(
            x[0], x[1], x[2], x[3], x[4],
            dw_vmin, dw_vmax, dw_yrmin, dw_yrmax,
            config.v_reso, config.yawrate_reso,
            config.dt, config.predict_time,
            config.max_speed,
            config.to_goal_cost_gain, config.speed_cost_gain,
            config.robot_radius,
            goal[0], goal[1],
            d_ob, n_ob,
            n_v, n_yr,
            d_costs, d_ctrl_v, d_ctrl_yr,
            d_traj_end_x, d_traj_end_y);

        // --- GPU: find min cost ---
        int red_threads = 256;
        size_t smem_size = red_threads * (sizeof(float) + sizeof(int));
        find_min_kernel<<<1, red_threads, smem_size>>>(d_costs, d_min_idx, n_samples);

        // --- readback best control ---
        int h_min_idx;
        CUDA_CHECK(cudaMemcpy(&h_min_idx, d_min_idx, sizeof(int),
                              cudaMemcpyDeviceToHost));

        float best_v, best_yr;
        CUDA_CHECK(cudaMemcpy(&best_v,  d_ctrl_v  + h_min_idx, sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&best_yr, d_ctrl_yr + h_min_idx, sizeof(float),
                              cudaMemcpyDeviceToHost));

        u = {{best_v, best_yr}};

        // --- host: generate best trajectory for visualization ---
        Traj ltraj = calc_trajectory_host(x, best_v, best_yr, config);

        // --- update state ---
        x = motion(x, u, config.dt);
        traj.push_back(x);

        // --- visualization ---
        cv::Mat bg(3500, 3500, CV_8UC3, cv::Scalar(255, 255, 255));

        cv::circle(bg, cv_offset(goal[0], goal[1], bg.cols, bg.rows),
                   30, cv::Scalar(255, 0, 0), 5);

        for (int j = 0; j < n_ob; j++) {
            cv::circle(bg, cv_offset(ob[j][0], ob[j][1], bg.cols, bg.rows),
                       20, cv::Scalar(0, 0, 0), -1);
        }

        for (unsigned int j = 0; j < ltraj.size(); j++) {
            cv::circle(bg, cv_offset(ltraj[j][0], ltraj[j][1], bg.cols, bg.rows),
                       7, cv::Scalar(0, 255, 0), -1);
        }

        cv::circle(bg, cv_offset(x[0], x[1], bg.cols, bg.rows),
                   30, cv::Scalar(0, 0, 255), 5);

        cv::arrowedLine(
            bg,
            cv_offset(x[0], x[1], bg.cols, bg.rows),
            cv_offset(x[0] + cosf(x[2]), x[1] + sinf(x[2]), bg.cols, bg.rows),
            cv::Scalar(255, 0, 255), 7);

        if (sqrtf((x[0] - goal[0]) * (x[0] - goal[0]) +
                  (x[1] - goal[1]) * (x[1] - goal[1])) <= config.robot_radius) {
            terminal = true;
            for (unsigned int j = 0; j < traj.size(); j++) {
                cv::circle(bg, cv_offset(traj[j][0], traj[j][1], bg.cols, bg.rows),
                           7, cv::Scalar(0, 0, 255), -1);
            }
        }

        cv::imshow("dwa", bg);
        cv::waitKey(5);
        count++;
    }

    // --- cleanup ---
    cudaFree(d_ob);
    cudaFree(d_costs);
    cudaFree(d_ctrl_v);
    cudaFree(d_ctrl_yr);
    cudaFree(d_traj_end_x);
    cudaFree(d_traj_end_y);
    cudaFree(d_min_idx);

    return 0;
}
