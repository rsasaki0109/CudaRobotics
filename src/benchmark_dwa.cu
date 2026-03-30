/*************************************************************************
    > File Name: benchmark_dwa.cu
    > DWA Benchmark: CPU vs CUDA
    > Measures core computation time for evaluating all (v, yawrate) samples
    > in the dynamic window at various resolution settings.
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <vector>
#include <array>
#include <algorithm>

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
// Config
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

// ---------------------------------------------------------------------------
// Obstacles (same as original)
// ---------------------------------------------------------------------------
static float h_obstacles[][2] = {
    {-1.0f, -1.0f},
    { 0.0f,  2.0f},
    { 4.0f,  2.0f},
    { 5.0f,  4.0f},
    { 5.0f,  5.0f},
    { 5.0f,  6.0f},
    { 5.0f,  9.0f},
    { 8.0f,  9.0f},
    { 7.0f,  9.0f},
    {12.0f, 12.0f}
};
static const int N_OB = sizeof(h_obstacles) / sizeof(h_obstacles[0]);

// ---------------------------------------------------------------------------
// CPU implementation: evaluate all (v, yawrate) samples sequentially
// ---------------------------------------------------------------------------
void cpu_dwa_eval(
    float sx, float sy, float syaw, float sv, float somega,
    float v_min, float v_max, float yr_min, float yr_max,
    const DWAConfig& cfg,
    float gx, float gy,
    const float* ob, int n_ob,
    int n_v, int n_yr,
    float* costs, float* ctrl_v, float* ctrl_yr)
{
    int total = n_v * n_yr;
    for (int idx = 0; idx < total; idx++) {
        int iv  = idx / n_yr;
        int iyr = idx % n_yr;

        float v  = v_min + iv  * cfg.v_reso;
        float yr = yr_min + iyr * cfg.yawrate_reso;

        if (v > v_max) v = v_max;
        if (yr > yr_max) yr = yr_max;

        ctrl_v[idx]  = v;
        ctrl_yr[idx] = yr;

        // simulate trajectory
        float px = sx, py = sy, pyaw = syaw;
        float time_sim = 0.0f;
        float minr = FLT_MAX;
        bool collision = false;
        int skip_n = 2;
        int step = 0;

        while (time_sim <= cfg.predict_time) {
            pyaw += yr * cfg.dt;
            px   += v * cosf(pyaw) * cfg.dt;
            py   += v * sinf(pyaw) * cfg.dt;
            time_sim += cfg.dt;
            step++;

            if (step % skip_n == 0) {
                for (int i = 0; i < n_ob; i++) {
                    float dx = px - ob[i * 2 + 0];
                    float dy = py - ob[i * 2 + 1];
                    float r  = sqrtf(dx * dx + dy * dy);
                    if (r <= cfg.robot_radius) collision = true;
                    if (r < minr) minr = r;
                }
            }
        }

        // check last point
        for (int i = 0; i < n_ob; i++) {
            float dx = px - ob[i * 2 + 0];
            float dy = py - ob[i * 2 + 1];
            float r  = sqrtf(dx * dx + dy * dy);
            if (r <= cfg.robot_radius) collision = true;
            if (r < minr) minr = r;
        }

        if (collision) {
            costs[idx] = FLT_MAX;
            continue;
        }

        // to_goal_cost
        float goal_mag = sqrtf(gx * gx + gy * gy);
        float traj_mag = sqrtf(px * px + py * py);
        float dot = gx * px + gy * py;
        float cos_angle = dot / (goal_mag * traj_mag + 1e-10f);
        if (cos_angle > 1.0f) cos_angle = 1.0f;
        if (cos_angle < -1.0f) cos_angle = -1.0f;
        float to_goal_cost = cfg.to_goal_cost_gain * acosf(cos_angle);

        // speed_cost
        float speed_cost = cfg.speed_cost_gain * (cfg.max_speed - v);

        // obstacle_cost
        float ob_cost = 1.0f / minr;

        costs[idx] = to_goal_cost + speed_cost + ob_cost;
    }
}

int cpu_find_min(const float* costs, int n) {
    int best = 0;
    float best_val = costs[0];
    for (int i = 1; i < n; i++) {
        if (costs[i] < best_val) {
            best_val = costs[i];
            best = i;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// CUDA kernel: evaluate all (v, yawrate) samples in parallel
// (same as dynamic_window_approach.cu)
// ---------------------------------------------------------------------------
__global__ void dwa_eval_kernel(
    float sx, float sy, float syaw, float sv, float somega,
    float v_min, float v_max, float yr_min, float yr_max,
    float v_reso, float yr_reso, float dt, float predict_time,
    float max_speed, float to_goal_cost_gain, float speed_cost_gain,
    float robot_radius,
    float gx, float gy,
    const float* ob, int n_ob,
    int n_v, int n_yr,
    float* costs, float* ctrl_v, float* ctrl_yr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_v * n_yr;
    if (idx >= total) return;

    int iv  = idx / n_yr;
    int iyr = idx % n_yr;

    float v  = v_min + iv  * v_reso;
    float yr = yr_min + iyr * yr_reso;

    if (v > v_max) v = v_max;
    if (yr > yr_max) yr = yr_max;

    ctrl_v[idx]  = v;
    ctrl_yr[idx] = yr;

    // simulate trajectory
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

        if (step % skip_n == 0) {
            for (int i = 0; i < n_ob; i++) {
                float dx = px - ob[i * 2 + 0];
                float dy = py - ob[i * 2 + 1];
                float r  = sqrtf(dx * dx + dy * dy);
                if (r <= robot_radius) collision = true;
                if (r < minr) minr = r;
            }
        }
    }

    // check last point
    for (int i = 0; i < n_ob; i++) {
        float dx = px - ob[i * 2 + 0];
        float dy = py - ob[i * 2 + 1];
        float r  = sqrtf(dx * dx + dy * dy);
        if (r <= robot_radius) collision = true;
        if (r < minr) minr = r;
    }

    if (collision) {
        costs[idx] = FLT_MAX;
        return;
    }

    float goal_mag = sqrtf(gx * gx + gy * gy);
    float traj_mag = sqrtf(px * px + py * py);
    float dot = gx * px + gy * py;
    float cos_angle = dot / (goal_mag * traj_mag + 1e-10f);
    if (cos_angle > 1.0f) cos_angle = 1.0f;
    if (cos_angle < -1.0f) cos_angle = -1.0f;
    float to_goal_cost = to_goal_cost_gain * acosf(cos_angle);

    float speed_cost = speed_cost_gain * (max_speed - v);
    float ob_cost = 1.0f / minr;

    costs[idx] = to_goal_cost + speed_cost + ob_cost;
}

// ---------------------------------------------------------------------------
// CUDA kernel: parallel reduction to find minimum cost index
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
// Resolution preset
// ---------------------------------------------------------------------------
struct ResolutionPreset {
    const char* name;
    float v_reso;
    float yr_reso;
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    // Initial state: x, y, yaw, v, omega
    float sx = 0.0f, sy = 0.0f, syaw = PI / 8.0f, sv = 0.0f, somega = 0.0f;
    float gx = 10.0f, gy = 10.0f;

    DWAConfig cfg;

    // Dynamic window (use full range for benchmarking: v=0, omega=0 initial)
    float dw_vmin  = std::max(sv - cfg.max_accel * cfg.dt, cfg.min_speed);
    float dw_vmax  = std::min(sv + cfg.max_accel * cfg.dt, cfg.max_speed);
    float dw_yrmin = std::max(somega - cfg.max_dyawrate * cfg.dt, -cfg.max_yawrate);
    float dw_yrmax = std::min(somega + cfg.max_dyawrate * cfg.dt, cfg.max_yawrate);

    ResolutionPreset presets[] = {
        {"Low",    0.05f,  1.0f  * PI / 180.0f},
        {"Medium", 0.01f,  0.1f  * PI / 180.0f},
        {"High",   0.005f, 0.05f * PI / 180.0f},
        {"Ultra",  0.002f, 0.02f * PI / 180.0f},
    };
    int n_presets = sizeof(presets) / sizeof(presets[0]);
    int n_iterations = 100;

    // Upload obstacles to GPU once
    float* d_ob;
    CUDA_CHECK(cudaMalloc(&d_ob, N_OB * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ob, h_obstacles, N_OB * 2 * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Print header
    printf("========================================\n");
    printf("  DWA Benchmark: CPU vs CUDA\n");
    printf("========================================\n\n");

    // Storage for summary
    struct Result {
        const char* name;
        int samples;
        double cpu_ms;
        double cuda_ms;
        double speedup;
    };
    std::vector<Result> results;

    for (int p = 0; p < n_presets; p++) {
        cfg.v_reso = presets[p].v_reso;
        cfg.yawrate_reso = presets[p].yr_reso;

        int n_v  = (int)((dw_vmax - dw_vmin) / cfg.v_reso) + 1;
        int n_yr = (int)((dw_yrmax - dw_yrmin) / cfg.yawrate_reso) + 1;
        int n_samples = n_v * n_yr;

        printf("  [%s] v_reso=%.4f, yr_reso=%.5f rad\n",
               presets[p].name, cfg.v_reso, cfg.yawrate_reso);
        printf("  n_v=%d, n_yr=%d, samples=%d\n", n_v, n_yr, n_samples);

        // ---- Allocate CPU buffers ----
        std::vector<float> h_costs(n_samples);
        std::vector<float> h_ctrl_v(n_samples);
        std::vector<float> h_ctrl_yr(n_samples);

        // ---- Allocate GPU buffers ----
        float *d_costs, *d_ctrl_v, *d_ctrl_yr;
        CUDA_CHECK(cudaMalloc(&d_costs,   n_samples * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ctrl_v,  n_samples * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ctrl_yr, n_samples * sizeof(float)));

        int *d_min_idx;
        CUDA_CHECK(cudaMalloc(&d_min_idx, sizeof(int)));

        const int threads = 256;
        int blocks = (n_samples + threads - 1) / threads;

        // ---- Warm up CUDA ----
        dwa_eval_kernel<<<blocks, threads>>>(
            sx, sy, syaw, sv, somega,
            dw_vmin, dw_vmax, dw_yrmin, dw_yrmax,
            cfg.v_reso, cfg.yawrate_reso,
            cfg.dt, cfg.predict_time,
            cfg.max_speed,
            cfg.to_goal_cost_gain, cfg.speed_cost_gain,
            cfg.robot_radius,
            gx, gy,
            d_ob, N_OB,
            n_v, n_yr,
            d_costs, d_ctrl_v, d_ctrl_yr);
        {
            int red_threads = 256;
            size_t smem_size = red_threads * (sizeof(float) + sizeof(int));
            find_min_kernel<<<1, red_threads, smem_size>>>(d_costs, d_min_idx, n_samples);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- Benchmark CPU ----
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < n_iterations; iter++) {
            cpu_dwa_eval(
                sx, sy, syaw, sv, somega,
                dw_vmin, dw_vmax, dw_yrmin, dw_yrmax,
                cfg, gx, gy,
                (const float*)h_obstacles, N_OB,
                n_v, n_yr,
                h_costs.data(), h_ctrl_v.data(), h_ctrl_yr.data());
            cpu_find_min(h_costs.data(), n_samples);
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        // ---- Benchmark CUDA ----
        // Use CUDA events for accurate GPU timing
        cudaEvent_t start_ev, stop_ev;
        CUDA_CHECK(cudaEventCreate(&start_ev));
        CUDA_CHECK(cudaEventCreate(&stop_ev));

        CUDA_CHECK(cudaEventRecord(start_ev));
        for (int iter = 0; iter < n_iterations; iter++) {
            dwa_eval_kernel<<<blocks, threads>>>(
                sx, sy, syaw, sv, somega,
                dw_vmin, dw_vmax, dw_yrmin, dw_yrmax,
                cfg.v_reso, cfg.yawrate_reso,
                cfg.dt, cfg.predict_time,
                cfg.max_speed,
                cfg.to_goal_cost_gain, cfg.speed_cost_gain,
                cfg.robot_radius,
                gx, gy,
                d_ob, N_OB,
                n_v, n_yr,
                d_costs, d_ctrl_v, d_ctrl_yr);

            int red_threads = 256;
            size_t smem_size = red_threads * (sizeof(float) + sizeof(int));
            find_min_kernel<<<1, red_threads, smem_size>>>(d_costs, d_min_idx, n_samples);
        }
        CUDA_CHECK(cudaEventRecord(stop_ev));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));

        float cuda_ms_f = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&cuda_ms_f, start_ev, stop_ev));
        double cuda_ms = (double)cuda_ms_f;

        CUDA_CHECK(cudaEventDestroy(start_ev));
        CUDA_CHECK(cudaEventDestroy(stop_ev));

        double speedup = cpu_ms / cuda_ms;

        printf("  Samples: %-8d (%d iterations)\n", n_samples, n_iterations);
        printf("    CPU:    %10.2f ms\n", cpu_ms);
        printf("    CUDA:   %10.2f ms\n", cuda_ms);
        printf("    Speedup: %7.2fx\n", speedup);
        printf("\n");

        results.push_back({presets[p].name, n_samples, cpu_ms, cuda_ms, speedup});

        // Cleanup per-preset GPU buffers
        cudaFree(d_costs);
        cudaFree(d_ctrl_v);
        cudaFree(d_ctrl_yr);
        cudaFree(d_min_idx);
    }

    // ---- Summary table ----
    printf("========================================\n");
    printf("  Summary\n");
    printf("========================================\n");
    printf("  %-8s %10s %12s %12s %10s\n",
           "Preset", "Samples", "CPU (ms)", "CUDA (ms)", "Speedup");
    printf("  %-8s %10s %12s %12s %10s\n",
           "------", "-------", "--------", "---------", "-------");
    for (auto& r : results) {
        printf("  %-8s %10d %12.2f %12.2f %9.2fx\n",
               r.name, r.samples, r.cpu_ms, r.cuda_ms, r.speedup);
    }
    printf("========================================\n");

    // Cleanup
    cudaFree(d_ob);

    return 0;
}
