/*************************************************************************
    MPPI: CPU (K=32) vs CUDA (K=4096) side-by-side comparison
    Left panel:  CPU with K=32 samples (sparse, jerky control)
    Right panel: CUDA with K=4096 samples (smooth, optimal control)
    Output: gif/comparison_mppi.gif
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cfloat>
#include <algorithm>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------
static const int K_CPU        = 32;       // CPU sample count (sparse)
static const int K_GPU        = 4096;     // GPU sample count (dense)
static const int T_HORIZON    = 30;
static const float DT         = 0.05f;
static const float WHEELBASE  = 2.5f;
static const float LAMBDA     = 10.0f;
static const float WORKSPACE  = 50.0f;

// Control limits
static const float MAX_ACCEL       = 5.0f;
static const float MAX_STEER_RATE  = 1.0f;
static const float MAX_SPEED       = 8.0f;
static const float MAX_STEER       = 0.6f;

// Noise standard deviations
static const float ACCEL_NOISE_STD = 2.0f;
static const float STEER_NOISE_STD = 0.4f;

// Cost weights
static const float GOAL_WEIGHT     = 1.0f;
static const float OBS_WEIGHT      = 200.0f;
static const float SPEED_WEIGHT    = 0.1f;
static const float STEER_WEIGHT    = 5.0f;
static const float TERMINAL_WEIGHT = 10.0f;

static const float START_X = 5.0f,  START_Y = 5.0f;
static const float START_THETA = 0.0f, START_V = 0.0f;
static const float GOAL_X  = 45.0f, GOAL_Y  = 45.0f;

// Obstacles
static const int N_OBSTACLES = 10;
__constant__ float d_obs_x[N_OBSTACLES];
__constant__ float d_obs_y[N_OBSTACLES];
__constant__ float d_obs_r[N_OBSTACLES];

static float h_obs_x[N_OBSTACLES] = {12.0f, 20.0f, 30.0f, 15.0f, 25.0f, 35.0f, 22.0f, 38.0f, 10.0f, 32.0f};
static float h_obs_y[N_OBSTACLES] = {15.0f, 25.0f, 10.0f, 35.0f, 18.0f, 30.0f, 40.0f, 20.0f, 30.0f, 38.0f};
static float h_obs_r[N_OBSTACLES] = { 3.0f,  3.5f,  3.0f,  2.5f,  3.5f,  2.5f,  3.0f,  3.0f,  2.5f,  2.5f};

static const int MAX_STEPS = 600;
static const float GOAL_TOL = 2.0f;

// -------------------------------------------------------------------------
// CUDA Kernels
// -------------------------------------------------------------------------
__global__ void init_curand_kernel(curandState* states, int K, unsigned long long seed)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curand_init(seed, k, 0, &states[k]);
}

__global__ void rollout_kernel(
    float sx, float sy, float stheta, float sv,
    const float* __restrict__ d_nominal,
    float* __restrict__ d_costs,
    float* __restrict__ d_perturbed,
    float* __restrict__ d_trajectories,
    curandState* __restrict__ d_rand_states,
    int K, int T, float dt, float L)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_state = d_rand_states[k];

    float x = sx, y = sy, theta = stheta, v = sv;
    float steer = 0.0f;
    float cost = 0.0f;

    for (int t = 0; t < T; t++) {
        float noise_a = curand_normal(&local_state) * ACCEL_NOISE_STD;
        float noise_s = curand_normal(&local_state) * STEER_NOISE_STD;

        float accel      = d_nominal[t * 2 + 0] + noise_a;
        float steer_rate = d_nominal[t * 2 + 1] + noise_s;

        accel      = fminf(fmaxf(accel, -MAX_ACCEL), MAX_ACCEL);
        steer_rate = fminf(fmaxf(steer_rate, -MAX_STEER_RATE), MAX_STEER_RATE);

        d_perturbed[k * T * 2 + t * 2 + 0] = accel;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer_rate;

        steer += steer_rate * dt;
        steer = fminf(fmaxf(steer, -MAX_STEER), MAX_STEER);
        v += accel * dt;
        v = fminf(fmaxf(v, -1.0f), MAX_SPEED);
        x += v * cosf(theta) * dt;
        y += v * sinf(theta) * dt;
        theta += v / L * tanf(steer) * dt;

        d_trajectories[k * T * 4 + t * 4 + 0] = x;
        d_trajectories[k * T * 4 + t * 4 + 1] = y;
        d_trajectories[k * T * 4 + t * 4 + 2] = theta;
        d_trajectories[k * T * 4 + t * 4 + 3] = v;

        float dx_g = x - GOAL_X;
        float dy_g = y - GOAL_Y;
        cost += GOAL_WEIGHT * sqrtf(dx_g * dx_g + dy_g * dy_g) * dt;

        for (int o = 0; o < N_OBSTACLES; o++) {
            float dx_o = x - d_obs_x[o];
            float dy_o = y - d_obs_y[o];
            float dist = sqrtf(dx_o * dx_o + dy_o * dy_o);
            float margin = dist - d_obs_r[o];
            if (margin < 0.0f) {
                cost += OBS_WEIGHT * 10.0f;
            } else if (margin < 2.0f) {
                float pen = 1.0f - margin / 2.0f;
                cost += OBS_WEIGHT * pen * pen;
            }
        }

        cost += SPEED_WEIGHT * accel * accel * dt;
        cost += STEER_WEIGHT * steer_rate * steer_rate * dt;

        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) {
            cost += 500.0f;
        }
    }

    float dx_t = x - GOAL_X;
    float dy_t = y - GOAL_Y;
    cost += TERMINAL_WEIGHT * sqrtf(dx_t * dx_t + dy_t * dy_t);

    d_costs[k] = cost;
    d_rand_states[k] = local_state;
}

__global__ void compute_weights_kernel(
    const float* __restrict__ d_costs,
    float* __restrict__ d_weights,
    float* __restrict__ d_min_cost,
    int K, float lambda)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float min_c = FLT_MAX;
    for (int k = 0; k < K; k++) {
        if (d_costs[k] < min_c) min_c = d_costs[k];
    }
    d_min_cost[0] = min_c;

    float sum_exp = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = expf(-1.0f / lambda * (d_costs[k] - min_c));
        d_weights[k] = w;
        sum_exp += w;
    }
    if (sum_exp > 0.0f) {
        for (int k = 0; k < K; k++) d_weights[k] /= sum_exp;
    }
}

__global__ void update_controls_kernel(
    float* __restrict__ d_nominal,
    const float* __restrict__ d_perturbed,
    const float* __restrict__ d_weights,
    int K, int T)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float wa = 0.0f, ws = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = d_weights[k];
        wa += w * d_perturbed[k * T * 2 + t * 2 + 0];
        ws += w * d_perturbed[k * T * 2 + t * 2 + 1];
    }
    d_nominal[t * 2 + 0] = wa;
    d_nominal[t * 2 + 1] = ws;
}

// -------------------------------------------------------------------------
// CPU MPPI implementation
// -------------------------------------------------------------------------
static float cpu_obstacle_cost(float x, float y)
{
    float cost = 0.0f;
    for (int o = 0; o < N_OBSTACLES; o++) {
        float dx = x - h_obs_x[o];
        float dy = y - h_obs_y[o];
        float dist = sqrtf(dx * dx + dy * dy);
        float margin = dist - h_obs_r[o];
        if (margin < 0.0f) {
            cost += OBS_WEIGHT * 10.0f;
        } else if (margin < 2.0f) {
            float pen = 1.0f - margin / 2.0f;
            cost += OBS_WEIGHT * pen * pen;
        }
    }
    return cost;
}

struct CpuMPPI {
    int K;
    float rx, ry, rtheta, rv, rsteer;
    vector<float> nominal;       // [T * 2]
    vector<float> perturbed;     // [K * T * 2]
    vector<float> costs;         // [K]
    vector<float> weights;       // [K]
    vector<float> trajectories;  // [K * T * 4]
    vector<float> path_x, path_y;
    bool reached;

    void init(int k_samples) {
        K = k_samples;
        rx = START_X; ry = START_Y; rtheta = START_THETA; rv = START_V;
        rsteer = 0.0f;
        reached = false;
        nominal.assign(T_HORIZON * 2, 0.0f);
        perturbed.resize(K * T_HORIZON * 2);
        costs.resize(K);
        weights.resize(K);
        trajectories.resize(K * T_HORIZON * 4);
        path_x.push_back(rx);
        path_y.push_back(ry);
    }

    void step() {
        if (reached) return;

        float dx_g = rx - GOAL_X, dy_g = ry - GOAL_Y;
        if (sqrtf(dx_g * dx_g + dy_g * dy_g) < GOAL_TOL) {
            reached = true;
            return;
        }

        // Rollout K samples
        for (int k = 0; k < K; k++) {
            float x = rx, y = ry, theta = rtheta, v = rv;
            float steer = 0.0f;
            float cost = 0.0f;

            for (int t = 0; t < T_HORIZON; t++) {
                float noise_a = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * ACCEL_NOISE_STD * 1.5f;
                float noise_s = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * STEER_NOISE_STD * 1.5f;

                float accel      = nominal[t * 2 + 0] + noise_a;
                float steer_rate = nominal[t * 2 + 1] + noise_s;

                accel      = fminf(fmaxf(accel, -MAX_ACCEL), MAX_ACCEL);
                steer_rate = fminf(fmaxf(steer_rate, -MAX_STEER_RATE), MAX_STEER_RATE);

                perturbed[k * T_HORIZON * 2 + t * 2 + 0] = accel;
                perturbed[k * T_HORIZON * 2 + t * 2 + 1] = steer_rate;

                steer += steer_rate * DT;
                steer = fminf(fmaxf(steer, -MAX_STEER), MAX_STEER);
                v += accel * DT;
                v = fminf(fmaxf(v, -1.0f), MAX_SPEED);
                x += v * cosf(theta) * DT;
                y += v * sinf(theta) * DT;
                theta += v / WHEELBASE * tanf(steer) * DT;

                trajectories[k * T_HORIZON * 4 + t * 4 + 0] = x;
                trajectories[k * T_HORIZON * 4 + t * 4 + 1] = y;
                trajectories[k * T_HORIZON * 4 + t * 4 + 2] = theta;
                trajectories[k * T_HORIZON * 4 + t * 4 + 3] = v;

                float dxg = x - GOAL_X, dyg = y - GOAL_Y;
                cost += GOAL_WEIGHT * sqrtf(dxg * dxg + dyg * dyg) * DT;
                cost += cpu_obstacle_cost(x, y);
                cost += SPEED_WEIGHT * accel * accel * DT;
                cost += STEER_WEIGHT * steer_rate * steer_rate * DT;

                if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE)
                    cost += 500.0f;
            }

            float dxt = x - GOAL_X, dyt = y - GOAL_Y;
            cost += TERMINAL_WEIGHT * sqrtf(dxt * dxt + dyt * dyt);
            costs[k] = cost;
        }

        // Compute weights (softmin)
        float min_c = *min_element(costs.begin(), costs.end());
        float sum_exp = 0.0f;
        for (int k = 0; k < K; k++) {
            weights[k] = expf(-1.0f / LAMBDA * (costs[k] - min_c));
            sum_exp += weights[k];
        }
        if (sum_exp > 0.0f) {
            for (int k = 0; k < K; k++) weights[k] /= sum_exp;
        }

        // Update nominal
        for (int t = 0; t < T_HORIZON; t++) {
            float wa = 0.0f, ws = 0.0f;
            for (int k = 0; k < K; k++) {
                wa += weights[k] * perturbed[k * T_HORIZON * 2 + t * 2 + 0];
                ws += weights[k] * perturbed[k * T_HORIZON * 2 + t * 2 + 1];
            }
            nominal[t * 2 + 0] = wa;
            nominal[t * 2 + 1] = ws;
        }

        // Apply first control
        float accel      = fminf(fmaxf(nominal[0], -MAX_ACCEL), MAX_ACCEL);
        float steer_rate = fminf(fmaxf(nominal[1], -MAX_STEER_RATE), MAX_STEER_RATE);

        rsteer += steer_rate * DT;
        rsteer = fminf(fmaxf(rsteer, -MAX_STEER), MAX_STEER);
        rv += accel * DT;
        rv = fminf(fmaxf(rv, -1.0f), MAX_SPEED);
        rx += rv * cosf(rtheta) * DT;
        ry += rv * sinf(rtheta) * DT;
        rtheta += rv / WHEELBASE * tanf(rsteer) * DT;

        path_x.push_back(rx);
        path_y.push_back(ry);

        // Shift nominal (warm start)
        for (int t = 0; t < T_HORIZON - 1; t++) {
            nominal[t * 2 + 0] = nominal[(t + 1) * 2 + 0];
            nominal[t * 2 + 1] = nominal[(t + 1) * 2 + 1];
        }
        nominal[(T_HORIZON - 1) * 2 + 0] = 0.0f;
        nominal[(T_HORIZON - 1) * 2 + 1] = 0.0f;
    }

    float min_cost() const {
        return costs.empty() ? 0.0f : *min_element(costs.begin(), costs.end());
    }
};

// -------------------------------------------------------------------------
// GPU MPPI wrapper
// -------------------------------------------------------------------------
struct GpuMPPI {
    int K;
    float rx, ry, rtheta, rv, rsteer;
    float *d_nominal, *d_costs, *d_weights, *d_perturbed, *d_trajectories, *d_min_cost;
    curandState *d_rand_states;
    vector<float> h_nominal;
    vector<float> h_costs;
    vector<float> h_trajectories;
    vector<float> path_x, path_y;
    bool reached;
    int ctrl_size;

    void init(int k_samples) {
        K = k_samples;
        rx = START_X; ry = START_Y; rtheta = START_THETA; rv = START_V;
        rsteer = 0.0f;
        reached = false;
        ctrl_size = T_HORIZON * 2;
        h_nominal.assign(ctrl_size, 0.0f);
        h_costs.resize(K);
        h_trajectories.resize(K * T_HORIZON * 4);
        path_x.push_back(rx);
        path_y.push_back(ry);

        CUDA_CHECK(cudaMalloc(&d_nominal, ctrl_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs, K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights, K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perturbed, K * ctrl_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_trajectories, K * T_HORIZON * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_min_cost, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rand_states, K * sizeof(curandState)));

        int block = 256;
        int grid = (K + block - 1) / block;
        init_curand_kernel<<<grid, block>>>(d_rand_states, K, 42ULL);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void step() {
        if (reached) return;

        float dx_g = rx - GOAL_X, dy_g = ry - GOAL_Y;
        if (sqrtf(dx_g * dx_g + dy_g * dy_g) < GOAL_TOL) {
            reached = true;
            return;
        }

        int block = 256;
        int grid_K = (K + block - 1) / block;
        int grid_T = (T_HORIZON + block - 1) / block;

        CUDA_CHECK(cudaMemcpy(d_nominal, h_nominal.data(), ctrl_size * sizeof(float), cudaMemcpyHostToDevice));

        rollout_kernel<<<grid_K, block>>>(
            rx, ry, rtheta, rv,
            d_nominal, d_costs, d_perturbed, d_trajectories, d_rand_states,
            K, T_HORIZON, DT, WHEELBASE);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        compute_weights_kernel<<<1, 1>>>(d_costs, d_weights, d_min_cost, K, LAMBDA);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        update_controls_kernel<<<grid_T, block>>>(d_nominal, d_perturbed, d_weights, K, T_HORIZON);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_nominal.data(), d_nominal, ctrl_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_costs.data(), d_costs, K * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_trajectories.data(), d_trajectories,
                              K * T_HORIZON * 4 * sizeof(float), cudaMemcpyDeviceToHost));

        // Apply first control
        float accel      = fminf(fmaxf(h_nominal[0], -MAX_ACCEL), MAX_ACCEL);
        float steer_rate = fminf(fmaxf(h_nominal[1], -MAX_STEER_RATE), MAX_STEER_RATE);

        rsteer += steer_rate * DT;
        rsteer = fminf(fmaxf(rsteer, -MAX_STEER), MAX_STEER);
        rv += accel * DT;
        rv = fminf(fmaxf(rv, -1.0f), MAX_SPEED);
        rx += rv * cosf(rtheta) * DT;
        ry += rv * sinf(rtheta) * DT;
        rtheta += rv / WHEELBASE * tanf(rsteer) * DT;

        path_x.push_back(rx);
        path_y.push_back(ry);

        // Shift nominal
        for (int t = 0; t < T_HORIZON - 1; t++) {
            h_nominal[t * 2 + 0] = h_nominal[(t + 1) * 2 + 0];
            h_nominal[t * 2 + 1] = h_nominal[(t + 1) * 2 + 1];
        }
        h_nominal[(T_HORIZON - 1) * 2 + 0] = 0.0f;
        h_nominal[(T_HORIZON - 1) * 2 + 1] = 0.0f;
    }

    float min_cost() const {
        return h_costs.empty() ? 0.0f : *min_element(h_costs.begin(), h_costs.end());
    }

    void cleanup() {
        CUDA_CHECK(cudaFree(d_nominal));
        CUDA_CHECK(cudaFree(d_costs));
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_perturbed));
        CUDA_CHECK(cudaFree(d_trajectories));
        CUDA_CHECK(cudaFree(d_min_cost));
        CUDA_CHECK(cudaFree(d_rand_states));
    }
};

// -------------------------------------------------------------------------
// Visualization
// -------------------------------------------------------------------------
static cv::Point world_to_pixel(float wx, float wy, int img_size, float ws)
{
    int px = (int)(wx / ws * (float)img_size);
    int py = img_size - 1 - (int)(wy / ws * (float)img_size);
    return cv::Point(px, py);
}

void draw_panel(
    cv::Mat& img,
    float rx, float ry, float rtheta,
    const vector<float>& path_x,
    const vector<float>& path_y,
    const vector<float>& trajectories,
    const vector<float>& costs,
    int K, int vis_K,
    const char* label,
    float mc, int step_num, double ms)
{
    int H = img.rows, W = img.cols;
    img.setTo(cv::Scalar(255, 255, 255));

    // Draw obstacles
    for (int o = 0; o < N_OBSTACLES; o++) {
        cv::Point center = world_to_pixel(h_obs_x[o], h_obs_y[o], W, WORKSPACE);
        int r = (int)(h_obs_r[o] / WORKSPACE * (float)W);
        cv::circle(img, center, r, cv::Scalar(40, 40, 40), -1);
    }

    // Draw sample trajectories colored by cost
    if (K > 0 && !costs.empty()) {
        float min_c = FLT_MAX, max_c = -FLT_MAX;
        for (int k = 0; k < K; k++) {
            if (costs[k] < min_c) min_c = costs[k];
            if (costs[k] > max_c) max_c = costs[k];
        }
        float range = max_c - min_c;
        if (range < 1e-6f) range = 1.0f;

        int stride = max(1, K / vis_K);
        for (int ki = 0; ki < K; ki += stride) {
            float nc = (costs[ki] - min_c) / range;
            nc = fminf(fmaxf(nc, 0.0f), 1.0f);
            int r_col = (int)(nc * 255.0f);
            int g_col = (int)((1.0f - nc) * 255.0f);
            cv::Scalar color(0, g_col, r_col);

            int base = ki * T_HORIZON * 4;
            for (int t = 0; t < T_HORIZON - 1; t++) {
                cv::Point p1 = world_to_pixel(trajectories[base + t * 4 + 0],
                                               trajectories[base + t * 4 + 1], W, WORKSPACE);
                cv::Point p2 = world_to_pixel(trajectories[base + (t + 1) * 4 + 0],
                                               trajectories[base + (t + 1) * 4 + 1], W, WORKSPACE);
                cv::line(img, p1, p2, color, 1);
            }
        }
    }

    // Draw path history
    for (int i = 1; i < (int)path_x.size(); i++) {
        cv::Point p1 = world_to_pixel(path_x[i - 1], path_y[i - 1], W, WORKSPACE);
        cv::Point p2 = world_to_pixel(path_x[i], path_y[i], W, WORKSPACE);
        cv::line(img, p1, p2, cv::Scalar(150, 150, 0), 2);
    }

    // Draw robot as red rectangle
    {
        float half_len = 1.2f, half_wid = 0.6f;
        float cs = cosf(rtheta), sn = sinf(rtheta);
        float lx[4] = {-half_len, half_len, half_len, -half_len};
        float ly[4] = {-half_wid, -half_wid, half_wid, half_wid};
        vector<cv::Point> pts;
        for (int i = 0; i < 4; i++) {
            float wx = rx + cs * lx[i] - sn * ly[i];
            float wy = ry + sn * lx[i] + cs * ly[i];
            pts.push_back(cv::Point(
                (int)(wx / WORKSPACE * (float)W),
                W - 1 - (int)(wy / WORKSPACE * (float)W)));
        }
        cv::fillConvexPoly(img, pts, cv::Scalar(0, 0, 220));
    }

    // Goal and start
    cv::circle(img, world_to_pixel(GOAL_X, GOAL_Y, W, WORKSPACE), 7, cv::Scalar(0, 180, 0), -1);
    cv::circle(img, world_to_pixel(START_X, START_Y, W, WORKSPACE), 6, cv::Scalar(200, 100, 0), -1);

    // Labels
    cv::putText(img, label, cv::Point(8, 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 2);
    char buf[128];
    snprintf(buf, sizeof(buf), "Step %d  Cost: %.1f", step_num, mc);
    cv::putText(img, buf, cv::Point(8, 44),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(80, 80, 80), 1);
    snprintf(buf, sizeof(buf), "Time: %.2f ms", ms);
    cv::putText(img, buf, cv::Point(8, 62),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(80, 80, 80), 1);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main()
{
    cout << "MPPI: CPU (K=" << K_CPU << ") vs CUDA (K=" << K_GPU << ") Comparison" << endl;

    srand(42);

    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_x, h_obs_x, N_OBSTACLES * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_y, h_obs_y, N_OBSTACLES * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_r, h_obs_r, N_OBSTACLES * sizeof(float)));

    CpuMPPI cpu_mppi;
    cpu_mppi.init(K_CPU);

    GpuMPPI gpu_mppi;
    gpu_mppi.init(K_GPU);

    int W = 400, H = 400;

    cv::VideoWriter video(
        "gif/comparison_mppi.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 20,
        cv::Size(W * 2, H));

    if (!video.isOpened()) {
        cerr << "Failed to open video writer" << endl;
        return 1;
    }

    for (int step = 0; step < MAX_STEPS; step++) {
        // CPU step
        auto t0 = chrono::high_resolution_clock::now();
        cpu_mppi.step();
        auto t1 = chrono::high_resolution_clock::now();
        double cpu_ms = chrono::duration<double, milli>(t1 - t0).count();

        // GPU step
        auto t2 = chrono::high_resolution_clock::now();
        gpu_mppi.step();
        auto t3 = chrono::high_resolution_clock::now();
        double gpu_ms = chrono::duration<double, milli>(t3 - t2).count();

        // Draw panels
        cv::Mat left(H, W, CV_8UC3);
        cv::Mat right(H, W, CV_8UC3);

        char cpu_label[64], gpu_label[64];
        snprintf(cpu_label, sizeof(cpu_label), "CPU (K=%d)", K_CPU);
        snprintf(gpu_label, sizeof(gpu_label), "CUDA (K=%d)", K_GPU);

        draw_panel(left, cpu_mppi.rx, cpu_mppi.ry, cpu_mppi.rtheta,
                   cpu_mppi.path_x, cpu_mppi.path_y,
                   cpu_mppi.trajectories, cpu_mppi.costs,
                   K_CPU, K_CPU, cpu_label, cpu_mppi.min_cost(), step, cpu_ms);

        draw_panel(right, gpu_mppi.rx, gpu_mppi.ry, gpu_mppi.rtheta,
                   gpu_mppi.path_x, gpu_mppi.path_y,
                   gpu_mppi.h_trajectories, gpu_mppi.h_costs,
                   K_GPU, 200, gpu_label, gpu_mppi.min_cost(), step, gpu_ms);

        cv::Mat combined;
        cv::hconcat(left, right, combined);

        cv::namedWindow("comparison_mppi", cv::WINDOW_AUTOSIZE);
        cv::imshow("comparison_mppi", combined);
        cv::waitKey(1);
        video.write(combined);

        if (step % 30 == 0) {
            printf("Step %3d: CPU cost=%.1f (%.2fms)  GPU cost=%.1f (%.2fms)\n",
                   step, cpu_mppi.min_cost(), cpu_ms, gpu_mppi.min_cost(), gpu_ms);
        }

        // Stop if both reached goal
        if (cpu_mppi.reached && gpu_mppi.reached) {
            printf("Both reached goal at step %d\n", step);
            break;
        }
    }

    // Hold final frame
    cv::Mat left(H, W, CV_8UC3), right(H, W, CV_8UC3);
    draw_panel(left, cpu_mppi.rx, cpu_mppi.ry, cpu_mppi.rtheta,
               cpu_mppi.path_x, cpu_mppi.path_y,
               cpu_mppi.trajectories, cpu_mppi.costs,
               0, 0, "CPU (Final)", cpu_mppi.min_cost(), MAX_STEPS, 0.0);
    draw_panel(right, gpu_mppi.rx, gpu_mppi.ry, gpu_mppi.rtheta,
               gpu_mppi.path_x, gpu_mppi.path_y,
               gpu_mppi.h_trajectories, gpu_mppi.h_costs,
               0, 0, "CUDA (Final)", gpu_mppi.min_cost(), MAX_STEPS, 0.0);
    cv::Mat combined;
    cv::hconcat(left, right, combined);
    for (int i = 0; i < 40; i++) video.write(combined);

    video.release();
    cout << "Video saved to gif/comparison_mppi.avi" << endl;

    system("ffmpeg -y -i gif/comparison_mppi.avi "
           "-vf 'fps=20,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_mppi.gif 2>/dev/null");
    cout << "GIF saved to gif/comparison_mppi.gif" << endl;

    cv::imshow("comparison_mppi", combined);
    cv::waitKey(0);

    gpu_mppi.cleanup();

    return 0;
}
