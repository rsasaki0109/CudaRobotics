/*************************************************************************
    CUDA Streams Demo: Concurrent Robotics Algorithm Execution
    Demonstrates running Particle Filter, DWA, and Potential Field
    simultaneously on the same GPU using CUDA Streams.
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

#define PI 3.141592653f
#define DT 0.1f
#define MAX_RANGE 20.0f

// =====================================================================
// Stream 1: Particle Filter kernels (localization)
// =====================================================================

struct Observation { float d, lx, ly; };

__global__ void pf_init_curand(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void pf_predict_and_weight(
    float* px, float* pw, float u_v, float u_omega,
    float rsim_0, float rsim_1, const Observation* obs, int n_obs,
    float Q, curandState* rng_states, int np)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    curandState rng = rng_states[ip];
    float ud_v = u_v + curand_normal(&rng) * rsim_0;
    float ud_o = u_omega + curand_normal(&rng) * rsim_1;

    float x = px[0 * np + ip], y = px[1 * np + ip], yaw = px[2 * np + ip];
    x += DT * cosf(yaw) * ud_v;
    y += DT * sinf(yaw) * ud_v;
    yaw += DT * ud_o;
    px[0 * np + ip] = x;
    px[1 * np + ip] = y;
    px[2 * np + ip] = yaw;
    px[3 * np + ip] = ud_v;

    float w = pw[ip];
    float sigma = sqrtf(Q);
    float inv_c = 1.0f / sqrtf(2.0f * PI * Q);
    for (int i = 0; i < n_obs; i++) {
        float dx = x - obs[i].lx, dy = y - obs[i].ly;
        float dz = sqrtf(dx * dx + dy * dy) - obs[i].d;
        w *= inv_c * expf(-dz * dz / (2.0f * Q));
    }
    pw[ip] = w;
    rng_states[ip] = rng;
}

__global__ void pf_normalize_weights(float* pw, int np) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float val = 0;
    for (int i = tid; i < np; i += blockDim.x) val += pw[i];
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float total = sdata[0];
    if (total < 1e-30f) total = 1e-30f;
    for (int i = tid; i < np; i += blockDim.x) pw[i] /= total;
}

// =====================================================================
// Stream 2: DWA kernels (path planning)
// =====================================================================

__global__ void dwa_eval_kernel(
    float sx, float sy, float syaw, float sv, float somega,
    float v_min, float v_max, float yr_min, float yr_max,
    float v_reso, float yr_reso, float dt, float predict_time,
    float max_speed, float to_goal_cost_gain, float speed_cost_gain,
    float robot_radius,
    float gx, float gy,
    const float* ob, int n_ob,
    int n_v, int n_yr,
    float* costs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_v * n_yr;
    if (idx >= total) return;

    int iv  = idx / n_yr;
    int iyr = idx % n_yr;

    float v  = v_min + iv * v_reso;
    float yr = yr_min + iyr * yr_reso;
    if (v > v_max) v = v_max;
    if (yr > yr_max) yr = yr_max;

    // simulate trajectory
    float px = sx, py = sy, pyaw = syaw;
    float time_sim = 0.0f;
    float minr = FLT_MAX;
    bool collision = false;
    int step = 0;

    while (time_sim <= predict_time) {
        pyaw += yr * dt;
        px += v * cosf(pyaw) * dt;
        py += v * sinf(pyaw) * dt;
        time_sim += dt;
        step++;

        if (step % 2 == 0) {
            for (int i = 0; i < n_ob; i++) {
                float dx = px - ob[i * 2 + 0];
                float dy = py - ob[i * 2 + 1];
                float r = sqrtf(dx * dx + dy * dy);
                if (r <= robot_radius) collision = true;
                if (r < minr) minr = r;
            }
        }
    }

    for (int i = 0; i < n_ob; i++) {
        float dx = px - ob[i * 2 + 0];
        float dy = py - ob[i * 2 + 1];
        float r = sqrtf(dx * dx + dy * dy);
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
    cos_angle = fminf(fmaxf(cos_angle, -1.0f), 1.0f);
    float to_goal_cost = to_goal_cost_gain * acosf(cos_angle);
    float speed_cost = speed_cost_gain * (max_speed - v);
    float ob_cost = 1.0f / minr;

    costs[idx] = to_goal_cost + speed_cost + ob_cost;
}

// =====================================================================
// Stream 3: Potential Field kernel
// =====================================================================

__global__ void potential_field_kernel(
    float* d_pmap,
    int xwidth, int ywidth,
    float min_x, float min_y,
    float gx, float gy,
    const float* d_ox, const float* d_oy,
    int n_obs,
    float reso, float kp, float eta, float robot_radius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = xwidth * ywidth;
    if (idx >= total) return;

    int ix = idx / ywidth;
    int iy = idx % ywidth;

    float wx = (float)ix * reso + min_x;
    float wy = (float)iy * reso + min_y;

    // Attractive potential
    float dx = wx - gx;
    float dy = wy - gy;
    float dist_goal = sqrtf(dx * dx + dy * dy);
    float u_att = 0.5f * kp * dist_goal;

    // Repulsive potential
    float u_rep = 0.0f;
    for (int k = 0; k < n_obs; k++) {
        float odx = wx - d_ox[k];
        float ody = wy - d_oy[k];
        float d = sqrtf(odx * odx + ody * ody);
        if (d <= 0.001f) {
            u_rep = 1.0e6f;
            break;
        }
        if (d <= robot_radius) {
            float inv_diff = 1.0f / d - 1.0f / robot_radius;
            u_rep += 0.5f * eta * inv_diff * inv_diff;
        }
    }

    d_pmap[idx] = u_att + u_rep;
}

// =====================================================================
// Data structures for each algorithm
// =====================================================================

struct PFData {
    static const int NP = 100;
    static const int THREADS = 128;
    static const int BLOCKS = (NP + THREADS - 1) / THREADS;

    float* d_px;
    float* d_pw;
    Observation* d_obs;
    curandState* d_rng;

    // Observations (pre-generated)
    Observation h_obs[4];
    int n_obs;

    void alloc() {
        CUDA_CHECK(cudaMalloc(&d_px, 4 * NP * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pw, NP * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_obs, 16 * sizeof(Observation)));
        CUDA_CHECK(cudaMalloc(&d_rng, NP * sizeof(curandState)));
    }

    void init(cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(d_px, 0, 4 * NP * sizeof(float), stream));
        std::vector<float> pw_init(NP, 1.0f / NP);
        CUDA_CHECK(cudaMemcpyAsync(d_pw, pw_init.data(), NP * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));
        pf_init_curand<<<BLOCKS, THREADS, 0, stream>>>(d_rng, 42ULL, NP);

        // Pre-generate some observations
        h_obs[0] = {5.2f, 10.0f, 0.0f};
        h_obs[1] = {8.1f, 10.0f, 10.0f};
        h_obs[2] = {12.3f, 0.0f, 15.0f};
        h_obs[3] = {9.7f, -5.0f, 20.0f};
        n_obs = 4;
        CUDA_CHECK(cudaMemcpyAsync(d_obs, h_obs, n_obs * sizeof(Observation),
                                    cudaMemcpyHostToDevice, stream));
    }

    void run_iteration(cudaStream_t stream) {
        pf_predict_and_weight<<<BLOCKS, THREADS, 0, stream>>>(
            d_px, d_pw, 1.0f, 0.1f,
            1.0f, 0.274f,
            d_obs, n_obs, 0.01f, d_rng, NP);
        pf_normalize_weights<<<1, THREADS, THREADS * sizeof(float), stream>>>(d_pw, NP);
    }

    void free_mem() {
        cudaFree(d_px);
        cudaFree(d_pw);
        cudaFree(d_obs);
        cudaFree(d_rng);
    }
};

struct DWAData {
    float v_reso = 0.01f;
    float yr_reso = 0.1f * PI / 180.0f;
    float dt = 0.1f;
    float predict_time = 3.0f;
    float max_speed = 1.0f;
    float robot_radius = 1.0f;

    float v_min, v_max, yr_min, yr_max;
    int n_v, n_yr, n_samples;

    float* d_ob;
    float* d_costs;
    int n_ob;

    static constexpr float obstacles[][2] = {
        {-1.0f, -1.0f}, {0.0f, 2.0f}, {4.0f, 2.0f}, {5.0f, 4.0f},
        {5.0f, 5.0f}, {5.0f, 6.0f}, {5.0f, 9.0f}, {8.0f, 9.0f},
        {7.0f, 9.0f}, {12.0f, 12.0f}
    };

    void alloc() {
        n_ob = 10;

        // Dynamic window around initial state (v=0, omega=0)
        float max_accel = 0.2f;
        float max_dyawrate = 40.0f * PI / 180.0f;
        float max_yawrate = 40.0f * PI / 180.0f;
        v_min = fmaxf(0.0f - max_accel * dt, -0.5f);
        v_max = fminf(0.0f + max_accel * dt, max_speed);
        yr_min = fmaxf(0.0f - max_dyawrate * dt, -max_yawrate);
        yr_max = fminf(0.0f + max_dyawrate * dt, max_yawrate);

        n_v = (int)((v_max - v_min) / v_reso) + 1;
        n_yr = (int)((yr_max - yr_min) / yr_reso) + 1;
        n_samples = n_v * n_yr;

        CUDA_CHECK(cudaMalloc(&d_ob, n_ob * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs, n_samples * sizeof(float)));
    }

    void init(cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync(d_ob, obstacles, n_ob * 2 * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));
    }

    void run_iteration(cudaStream_t stream) {
        int threads = 256;
        int blocks = (n_samples + threads - 1) / threads;
        dwa_eval_kernel<<<blocks, threads, 0, stream>>>(
            0.0f, 0.0f, PI / 8.0f, 0.0f, 0.0f,
            v_min, v_max, yr_min, yr_max,
            v_reso, yr_reso, dt, predict_time,
            max_speed, 1.0f, 1.0f, robot_radius,
            10.0f, 10.0f,
            d_ob, n_ob,
            n_v, n_yr,
            d_costs);
    }

    void free_mem() {
        cudaFree(d_ob);
        cudaFree(d_costs);
    }
};

constexpr float DWAData::obstacles[][2];

struct PFData2 {
    // Potential field data
    static const int GRID_X = 50;
    static const int GRID_Y = 50;
    static const int GRID_TOTAL = GRID_X * GRID_Y;

    float* d_pmap;
    float* d_ox;
    float* d_oy;
    int n_obs;

    float h_ox[4] = {15.0f, 5.0f, 20.0f, 25.0f};
    float h_oy[4] = {25.0f, 15.0f, 26.0f, 25.0f};

    void alloc() {
        n_obs = 4;
        CUDA_CHECK(cudaMalloc(&d_pmap, GRID_TOTAL * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ox, n_obs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_oy, n_obs * sizeof(float)));
    }

    void init(cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync(d_ox, h_ox, n_obs * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_oy, h_oy, n_obs * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));
    }

    void run_iteration(cudaStream_t stream) {
        int threads = 256;
        int blocks = (GRID_TOTAL + threads - 1) / threads;
        potential_field_kernel<<<blocks, threads, 0, stream>>>(
            d_pmap, GRID_X, GRID_Y,
            -10.0f, 0.0f,        // min_x, min_y
            30.0f, 30.0f,        // goal
            d_ox, d_oy, n_obs,
            0.5f,                 // resolution
            5.0f,                 // KP
            100.0f,               // ETA
            5.0f);                // robot radius
    }

    void free_mem() {
        cudaFree(d_pmap);
        cudaFree(d_ox);
        cudaFree(d_oy);
    }
};

// =====================================================================
// Main
// =====================================================================

int main() {
    const int N_ITER = 100;

    // Warmup
    {
        float* tmp;
        cudaMalloc(&tmp, 1024);
        cudaFree(tmp);
        cudaDeviceSynchronize();
    }

    // Allocate algorithm data
    PFData pf;
    DWAData dwa;
    PFData2 potfield;

    pf.alloc();
    dwa.alloc();
    potfield.alloc();

    // Create 3 streams
    cudaStream_t stream1, stream2, stream3;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    CUDA_CHECK(cudaStreamCreate(&stream3));

    // Initialize data in respective streams
    pf.init(stream1);
    dwa.init(stream2);
    potfield.init(stream3);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create events for timing
    cudaEvent_t ev_start, ev_stop;
    cudaEvent_t ev_pf_start, ev_pf_stop;
    cudaEvent_t ev_dwa_start, ev_dwa_stop;
    cudaEvent_t ev_pf2_start, ev_pf2_stop;

    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventCreate(&ev_pf_start));
    CUDA_CHECK(cudaEventCreate(&ev_pf_stop));
    CUDA_CHECK(cudaEventCreate(&ev_dwa_start));
    CUDA_CHECK(cudaEventCreate(&ev_dwa_stop));
    CUDA_CHECK(cudaEventCreate(&ev_pf2_start));
    CUDA_CHECK(cudaEventCreate(&ev_pf2_stop));

    // ==================================================================
    // Sequential execution: run each algorithm on default stream
    // ==================================================================
    float seq_pf_ms, seq_dwa_ms, seq_pf2_ms;

    // PF sequential
    CUDA_CHECK(cudaEventRecord(ev_pf_start));
    for (int i = 0; i < N_ITER; i++) {
        pf.run_iteration(0);  // default stream
    }
    CUDA_CHECK(cudaEventRecord(ev_pf_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_pf_stop));
    CUDA_CHECK(cudaEventElapsedTime(&seq_pf_ms, ev_pf_start, ev_pf_stop));

    // DWA sequential
    CUDA_CHECK(cudaEventRecord(ev_dwa_start));
    for (int i = 0; i < N_ITER; i++) {
        dwa.run_iteration(0);
    }
    CUDA_CHECK(cudaEventRecord(ev_dwa_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_dwa_stop));
    CUDA_CHECK(cudaEventElapsedTime(&seq_dwa_ms, ev_dwa_start, ev_dwa_stop));

    // Potential Field sequential
    CUDA_CHECK(cudaEventRecord(ev_pf2_start));
    for (int i = 0; i < N_ITER; i++) {
        potfield.run_iteration(0);
    }
    CUDA_CHECK(cudaEventRecord(ev_pf2_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_pf2_stop));
    CUDA_CHECK(cudaEventElapsedTime(&seq_pf2_ms, ev_pf2_start, ev_pf2_stop));

    float seq_total_ms = seq_pf_ms + seq_dwa_ms + seq_pf2_ms;

    // Re-init for concurrent run
    pf.init(stream1);
    dwa.init(stream2);
    potfield.init(stream3);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ==================================================================
    // Concurrent execution: each algorithm in its own stream
    // ==================================================================
    float conc_total_ms;

    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int i = 0; i < N_ITER; i++) {
        pf.run_iteration(stream1);
        dwa.run_iteration(stream2);
        potfield.run_iteration(stream3);
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&conc_total_ms, ev_start, ev_stop));

    // Compute metrics
    float speedup = seq_total_ms / conc_total_ms;
    float max_single = fmaxf(seq_pf_ms, fmaxf(seq_dwa_ms, seq_pf2_ms));
    // Overlap: how much time was saved vs sequential, as fraction of
    // the theoretically saveable time (seq_total - max_single)
    float saved = seq_total_ms - conc_total_ms;
    float saveable = seq_total_ms - max_single;
    float overlap_pct = (saveable > 0.001f) ? (saved / saveable) * 100.0f : 0.0f;
    if (overlap_pct > 100.0f) overlap_pct = 100.0f;
    if (overlap_pct < 0.0f) overlap_pct = 0.0f;

    // ==================================================================
    // Print results
    // ==================================================================
    printf("========================================\n");
    printf("  CUDA Streams: Concurrent Algorithm Execution\n");
    printf("========================================\n");
    printf("\n");
    printf("  Sequential execution:\n");
    printf("    PF:              %6.2f ms\n", seq_pf_ms);
    printf("    DWA:             %6.2f ms\n", seq_dwa_ms);
    printf("    Potential Field: %6.2f ms\n", seq_pf2_ms);
    printf("    Total:           %6.2f ms\n", seq_total_ms);
    printf("\n");
    printf("  Concurrent execution (3 streams):\n");
    printf("    Total:           %6.2f ms\n", conc_total_ms);
    printf("    Speedup:         %5.2fx\n", speedup);
    printf("\n");
    printf("  Stream overlap achieved: %.0f%%\n", overlap_pct);
    printf("========================================\n");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaEventDestroy(ev_pf_start));
    CUDA_CHECK(cudaEventDestroy(ev_pf_stop));
    CUDA_CHECK(cudaEventDestroy(ev_dwa_start));
    CUDA_CHECK(cudaEventDestroy(ev_dwa_stop));
    CUDA_CHECK(cudaEventDestroy(ev_pf2_start));
    CUDA_CHECK(cudaEventDestroy(ev_pf2_stop));

    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaStreamDestroy(stream3));

    pf.free_mem();
    dwa.free_mem();
    potfield.free_mem();

    return 0;
}
