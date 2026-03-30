/*************************************************************************
    > File Name: state_lattice_planner.cu
    > CUDA-parallelized State Lattice Planner
    > Based on original C++ implementation by TAI Lei
    > Parallelizes lookup table search and trajectory optimization on GPU
 ************************************************************************/

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>

#include <cuda_runtime.h>

#define PI_F 3.141592653589793f

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
#define L        1.0f
#define DS       0.1f
#define CONST_V  3.0f
#define MAX_TRAJ_STEPS 512
#define MAX_ITER 100

// ---------------------------------------------------------------------------
// Device helper: yaw angle wrapping to [-pi, pi]
// ---------------------------------------------------------------------------
__device__ __host__ inline float yaw_p2p(float angle) {
    float a = fmodf(fmodf(angle + PI_F, 2.0f * PI_F) - 2.0f * PI_F, 2.0f * PI_F) + PI_F;
    return a;
}

// ---------------------------------------------------------------------------
// Host-side structs (for sampling and result collection)
// ---------------------------------------------------------------------------
struct TrajState {
    float x, y, yaw;
};

// ---------------------------------------------------------------------------
// Lookup table row: [x, y, yaw, ?, s1, s2]  (6 floats per row)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Device: quadratic interpolation (solve 3x3 system for parabola coefficients)
// Given 3 x-values and 3 y-values, returns coefficients a,b,c such that
//   y = a*x^2 + b*x + c
// Uses direct Cramer's rule for the Vandermonde-like system.
// ---------------------------------------------------------------------------
__device__ inline void quadratic_interp(float x0, float x1, float x2,
                                        float y0, float y1, float y2,
                                        float &a, float &b, float &c) {
    // A = [[x0^2, x0, 1], [x1^2, x1, 1], [x2^2, x2, 1]]
    // Solve A * [a,b,c]^T = [y0,y1,y2]^T via explicit inverse of 3x3
    float a00 = x0*x0, a01 = x0, a02 = 1.0f;
    float a10 = x1*x1, a11 = x1, a12 = 1.0f;
    float a20 = x2*x2, a21 = x2, a22 = 1.0f;

    float det = a00*(a11*a22 - a12*a21)
              - a01*(a10*a22 - a12*a20)
              + a02*(a10*a21 - a11*a20);

    if (fabsf(det) < 1e-12f) {
        a = 0.0f; b = 0.0f; c = 0.0f;
        return;
    }
    float inv_det = 1.0f / det;

    // cofactor matrix (transposed = adjugate)
    float c00 =  (a11*a22 - a12*a21);
    float c01 = -(a01*a22 - a02*a21);
    float c02 =  (a01*a12 - a02*a11);
    float c10 = -(a10*a22 - a12*a20);
    float c11 =  (a00*a22 - a02*a20);
    float c12 = -(a00*a12 - a02*a10);
    float c20 =  (a10*a21 - a11*a20);
    float c21 = -(a00*a21 - a01*a20);
    float c22 =  (a00*a11 - a01*a10);

    a = inv_det * (c00*y0 + c01*y1 + c02*y2);
    b = inv_det * (c10*y0 + c11*y1 + c12*y2);
    c = inv_det * (c20*y0 + c21*y1 + c22*y2);
}

__device__ inline float interp_eval(float a, float b, float c, float x) {
    return a*x*x + b*x + c;
}

// ---------------------------------------------------------------------------
// Device: simulate bicycle model and return last state (x, y, yaw)
// Parameters: distance, steering[3] (s0 fixed, s1, s2 optimized)
// ---------------------------------------------------------------------------
__device__ inline void generate_last_state(float distance, float s0, float s1, float s2,
                                           float &ox, float &oy, float &oyaw) {
    float n = distance / DS;
    float horizon = distance / CONST_V;
    if (n < 1.0f) n = 1.0f;
    float dt = horizon / n;

    // quadratic interpolation for steering spline
    float qa, qb, qc;
    quadratic_interp(0.0f, horizon * 0.5f, horizon, s0, s1, s2, qa, qb, qc);

    float sx = 0.0f, sy = 0.0f, syaw = 0.0f;
    for (float t = 0.0f; t < horizon; t += dt) {
        float kp = interp_eval(qa, qb, qc, t);
        sx   += CONST_V * cosf(syaw) * dt;
        sy   += CONST_V * sinf(syaw) * dt;
        syaw += CONST_V / L * tanf(kp) * dt;
        syaw  = yaw_p2p(syaw);
    }
    ox = sx; oy = sy; oyaw = syaw;
}

// ---------------------------------------------------------------------------
// Device: generate full trajectory into provided arrays, return length
// ---------------------------------------------------------------------------
__device__ inline int generate_trajectory(float distance, float s0, float s1, float s2,
                                          float *traj_x, float *traj_y, float *traj_yaw) {
    float n = distance / DS;
    float horizon = distance / CONST_V;
    if (n < 1.0f) n = 1.0f;
    float dt = horizon / n;

    float qa, qb, qc;
    quadratic_interp(0.0f, horizon * 0.5f, horizon, s0, s1, s2, qa, qb, qc);

    float sx = 0.0f, sy = 0.0f, syaw = 0.0f;
    int idx = 0;
    for (float t = 0.0f; t < horizon && idx < MAX_TRAJ_STEPS; t += dt) {
        float kp = interp_eval(qa, qb, qc, t);
        sx   += CONST_V * cosf(syaw) * dt;
        sy   += CONST_V * sinf(syaw) * dt;
        syaw += CONST_V / L * tanf(kp) * dt;
        syaw  = yaw_p2p(syaw);
        traj_x[idx]   = sx;
        traj_y[idx]   = sy;
        traj_yaw[idx] = syaw;
        idx++;
    }
    return idx;
}

// ---------------------------------------------------------------------------
// Device: compute error vector between target and generated last state
// ---------------------------------------------------------------------------
__device__ inline void compute_error(float distance, float s0, float s1, float s2,
                                     float tx, float ty, float tyaw,
                                     float &ex, float &ey, float &eyaw) {
    float lx, ly, lyaw;
    generate_last_state(distance, s0, s1, s2, lx, ly, lyaw);
    ex = tx - lx;
    ey = ty - ly;
    eyaw = yaw_p2p(tyaw - lyaw);
}

// ---------------------------------------------------------------------------
// Device: 3x3 matrix inverse (column-major storage)
// M[row + col*3]
// ---------------------------------------------------------------------------
__device__ inline bool invert3x3(const float *M, float *Minv) {
    float a00=M[0], a10=M[1], a20=M[2];
    float a01=M[3], a11=M[4], a21=M[5];
    float a02=M[6], a12=M[7], a22=M[8];

    float det = a00*(a11*a22 - a12*a21)
              - a01*(a10*a22 - a12*a20)
              + a02*(a10*a21 - a11*a20);

    if (fabsf(det) < 1e-15f) return false;
    float inv = 1.0f / det;

    Minv[0] = inv *  (a11*a22 - a12*a21);
    Minv[1] = inv * -(a10*a22 - a12*a20);
    Minv[2] = inv *  (a10*a21 - a11*a20);
    Minv[3] = inv * -(a01*a22 - a02*a21);
    Minv[4] = inv *  (a00*a22 - a02*a20);
    Minv[5] = inv * -(a00*a21 - a01*a20);
    Minv[6] = inv *  (a01*a12 - a02*a11);
    Minv[7] = inv * -(a00*a12 - a02*a10);
    Minv[8] = inv *  (a00*a11 - a01*a10);
    return true;
}

// ---------------------------------------------------------------------------
// Kernel 1: search_nearest_lookup_kernel
// Each thread handles one target state.
// Performs linear scan of lookup table to find nearest entry.
// Output: initial parameter (distance, s1, s2) per target.
// ---------------------------------------------------------------------------
__global__ void search_nearest_lookup_kernel(
        const float *d_targets,   // [N * 3]: x, y, yaw per target
        const float *d_lookup,    // [M * 6]: x, y, yaw, ?, s1, s2 per row
        int M,                    // number of lookup table rows
        float *d_params,          // output [N * 4]: distance, s0, s1, s2
        int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float tx = d_targets[tid * 3 + 0];
    float ty = d_targets[tid * 3 + 1];
    float tyaw = d_targets[tid * 3 + 2];

    float min_d = 1e30f;
    int min_id = 0;

    for (int i = 0; i < M; i++) {
        float dx = tx - d_lookup[i * 6 + 0];
        float dy = ty - d_lookup[i * 6 + 1];
        float dyaw = tyaw - d_lookup[i * 6 + 2];
        float dist = sqrtf(dx*dx + dy*dy + dyaw*dyaw);
        if (dist < min_d) {
            min_d = dist;
            min_id = i;
        }
    }

    // distance = euclidean distance from origin to target position
    float dist_to_target = sqrtf(tx*tx + ty*ty);
    // steering_sequence: s0=0 (will be set by host), s1=lookup[4], s2=lookup[5]
    d_params[tid * 4 + 0] = dist_to_target;          // distance
    d_params[tid * 4 + 1] = 0.0f;                    // s0 (k0, set later)
    d_params[tid * 4 + 2] = d_lookup[min_id * 6 + 4]; // s1
    d_params[tid * 4 + 3] = d_lookup[min_id * 6 + 5]; // s2
}

// ---------------------------------------------------------------------------
// Kernel 2: trajectory_optimization_kernel
// Each thread optimizes one trajectory independently.
// Uses numerical Jacobian + Newton step with learning rate selection.
// ---------------------------------------------------------------------------
__global__ void trajectory_optimization_kernel(
        const float *d_targets,  // [N * 3]
        float *d_params,         // [N * 4]: distance, s0, s1, s2 (in/out)
        float *d_traj_x,        // [N * MAX_TRAJ_STEPS]
        float *d_traj_y,        // [N * MAX_TRAJ_STEPS]
        float *d_traj_yaw,      // [N * MAX_TRAJ_STEPS]
        int *d_traj_len,         // [N]
        float cost_th,
        int max_iter,
        int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float tx = d_targets[tid * 3 + 0];
    float ty = d_targets[tid * 3 + 1];
    float tyaw = d_targets[tid * 3 + 2];

    float distance = d_params[tid * 4 + 0];
    float s0       = d_params[tid * 4 + 1];
    float s1       = d_params[tid * 4 + 2];
    float s2       = d_params[tid * 4 + 3];

    float h0 = 0.5f, h1 = 0.02f, h2 = 0.02f;

    for (int iter = 0; iter < max_iter; iter++) {
        // Compute error at current parameters
        float ex, ey, eyaw;
        compute_error(distance, s0, s1, s2, tx, ty, tyaw, ex, ey, eyaw);
        float cost = sqrtf(ex*ex + ey*ey + eyaw*eyaw);
        if (cost < cost_th) break;

        // Compute Jacobian via central differences
        // J is 3x3, column-major: J[row + col*3]
        float J[9], Jinv[9];

        // Column 0: d(error)/d(distance)
        float e0x, e0y, e0yaw, e1x, e1y, e1yaw;
        compute_error(distance + h0, s0, s1, s2, tx, ty, tyaw, e0x, e0y, e0yaw);
        compute_error(distance - h0, s0, s1, s2, tx, ty, tyaw, e1x, e1y, e1yaw);
        J[0] = (e0x - e1x) / (2.0f * h0);
        J[1] = (e0y - e1y) / (2.0f * h0);
        J[2] = (e0yaw - e1yaw) / (2.0f * h0);

        // Column 1: d(error)/d(s1)
        compute_error(distance, s0, s1 + h1, s2, tx, ty, tyaw, e0x, e0y, e0yaw);
        compute_error(distance, s0, s1 - h1, s2, tx, ty, tyaw, e1x, e1y, e1yaw);
        J[3] = (e0x - e1x) / (2.0f * h1);
        J[4] = (e0y - e1y) / (2.0f * h1);
        J[5] = (e0yaw - e1yaw) / (2.0f * h1);

        // Column 2: d(error)/d(s2)
        compute_error(distance, s0, s1, s2 + h2, tx, ty, tyaw, e0x, e0y, e0yaw);
        compute_error(distance, s0, s1, s2 - h2, tx, ty, tyaw, e1x, e1y, e1yaw);
        J[6] = (e0x - e1x) / (2.0f * h2);
        J[7] = (e0y - e1y) / (2.0f * h2);
        J[8] = (e0yaw - e1yaw) / (2.0f * h2);

        // dp = -J^{-1} * error
        if (!invert3x3(J, Jinv)) break;

        float dp0 = -(Jinv[0]*ex + Jinv[3]*ey + Jinv[6]*eyaw);
        float dp1 = -(Jinv[1]*ex + Jinv[4]*ey + Jinv[7]*eyaw);
        float dp2 = -(Jinv[2]*ex + Jinv[5]*ey + Jinv[8]*eyaw);

        // Learning rate selection: try alpha in [1.0, 1.5] with step 0.5
        float best_alpha = 1.0f;
        float best_cost = 1e30f;
        for (float alpha = 1.0f; alpha < 2.0f; alpha += 0.5f) {
            float nd = distance + alpha * dp0;
            float ns1 = s1 + alpha * dp1;
            float ns2 = s2 + alpha * dp2;
            float tex, tey, teyaw;
            compute_error(nd, s0, ns1, ns2, tx, ty, tyaw, tex, tey, teyaw);
            float tc = sqrtf(tex*tex + tey*tey + teyaw*teyaw);
            if (tc < best_cost) {
                best_cost = tc;
                best_alpha = alpha;
            }
        }

        distance += best_alpha * dp0;
        s1       += best_alpha * dp1;
        s2       += best_alpha * dp2;
    }

    // Store optimized parameters back
    d_params[tid * 4 + 0] = distance;
    d_params[tid * 4 + 2] = s1;
    d_params[tid * 4 + 3] = s2;

    // Generate final trajectory
    float *tx_arr   = d_traj_x   + tid * MAX_TRAJ_STEPS;
    float *ty_arr   = d_traj_y   + tid * MAX_TRAJ_STEPS;
    float *tyaw_arr = d_traj_yaw + tid * MAX_TRAJ_STEPS;
    int len = generate_trajectory(distance, s0, s1, s2, tx_arr, ty_arr, tyaw_arr);
    d_traj_len[tid] = len;
}

// ===========================================================================
// Host-side sampling functions (identical logic to the original CPU code)
// ===========================================================================

std::vector<TrajState> sample_states(const std::vector<float> &angle_samples,
                                     float a_min, float a_max,
                                     int d, float p_max, float p_min, int nh) {
    std::vector<TrajState> states;
    for (float item : angle_samples) {
        float a = a_min + (a_max - a_min) * item;
        for (int j = 0; j < nh; j++) {
            float xf = d * cosf(a);
            float yf = d * sinf(a);
            float yawf;
            if (nh == 1) yawf = (p_max - p_min) / 2.0f + a;
            else yawf = p_min + (p_max - p_min) * j / (nh - 1) + a;
            states.push_back({xf, yf, yawf});
        }
    }
    return states;
}

std::vector<TrajState> calc_uniform_polar_states(int nxy, int nh, int d,
                                                  float a_min, float a_max,
                                                  float p_min, float p_max) {
    std::vector<float> angle_samples;
    for (int i = 0; i < nxy; i++) {
        angle_samples.push_back(i * 1.0f / (nxy - 1));
    }
    return sample_states(angle_samples, a_min, a_max, d, p_max, p_min, nh);
}

std::vector<TrajState> calc_biased_polar_states(float goal_angle, int ns, int nxy,
                                                 int nh, int d,
                                                 float a_min, float a_max,
                                                 float p_min, float p_max) {
    std::vector<float> asi;
    std::vector<float> cnav;
    float cnav_max = -1e30f;
    float cnav_sum = 0.0f;
    for (int i = 0; i < ns - 1; i++) {
        float asi_sample = a_min + (a_max - a_min) * i / (ns - 1);
        asi.push_back(asi_sample);
        float cnav_sample = PI_F - fabsf(asi_sample - goal_angle);
        cnav.push_back(cnav_sample);
        cnav_sum += cnav_sample;
        if (cnav_max < cnav_sample) cnav_max = cnav_sample;
    }

    std::vector<float> csumnav;
    float cum_temp = 0.0f;
    for (int i = 0; i < ns - 1; i++) {
        cnav[i] = (cnav_max - cnav[i]) / (cnav_max * ns - cnav_sum);
        cum_temp += cnav[i];
        csumnav.push_back(cum_temp);
    }

    int li = 0;
    std::vector<float> angle_samples;
    for (int i = 0; i < nxy; i++) {
        for (int j = li; j < ns - 1; j++) {
            if (j * 1.0f / ns >= i * 1.0f / (nxy - 1)) {
                angle_samples.push_back(csumnav[j]);
                li = j - 1;
                break;
            }
        }
    }

    return sample_states(angle_samples, a_min, a_max, d, p_max, p_min, nh);
}

std::vector<TrajState> calc_lane_states(float l_center, float l_heading,
                                         float l_width, float v_width,
                                         float d, int nxy) {
    float xc = cosf(l_heading) * d + sinf(l_heading) * l_center;
    float yc = sinf(l_heading) * d + cosf(l_heading) * l_center;

    std::vector<TrajState> states;
    for (int i = 0; i < nxy; i++) {
        float delta = -0.5f * (l_width - v_width) + (l_width - v_width) * i / (nxy - 1);
        float xf = xc - delta * sinf(l_heading);
        float yf = yc + delta * cosf(l_heading);
        states.push_back({xf, yf, l_heading});
    }
    return states;
}

// ===========================================================================
// Host-side CSV reader (simple inline implementation)
// ===========================================================================

std::vector<std::vector<float>> read_lookup_table(const char *path) {
    std::vector<std::vector<float>> table;
    std::ifstream file(path);
    if (!file.is_open()) {
        return table;
    }
    std::string line;
    // skip header
    std::getline(file, line);
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stof(cell));
        }
        if (row.size() >= 6) table.push_back(row);
    }
    return table;
}

// ===========================================================================
// GPU path generation: lookup + optimize + collect trajectories
// ===========================================================================

struct TrajectoryResult {
    std::vector<float> x, y, yaw;
};

std::vector<TrajectoryResult> generate_path_cuda(
        const std::vector<TrajState> &states,
        const float *d_lookup, int lookup_rows,
        float k0 = 0.0f)
{
    int N = (int)states.size();
    if (N == 0) return {};

    // Pack targets into flat array
    std::vector<float> h_targets(N * 3);
    for (int i = 0; i < N; i++) {
        h_targets[i * 3 + 0] = states[i].x;
        h_targets[i * 3 + 1] = states[i].y;
        h_targets[i * 3 + 2] = states[i].yaw;
    }

    // Allocate device memory
    float *d_targets, *d_params;
    CUDA_CHECK(cudaMalloc(&d_targets, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_params, N * 4 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel 1: find nearest lookup entry for each target
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    search_nearest_lookup_kernel<<<gridSize, blockSize>>>(
        d_targets, d_lookup, lookup_rows, d_params, N);
    CUDA_CHECK(cudaGetLastError());

    // Set k0 for all targets (s0 = k0)
    if (k0 != 0.0f) {
        // Copy params to host, set s0, copy back
        std::vector<float> h_params(N * 4);
        CUDA_CHECK(cudaMemcpy(h_params.data(), d_params, N * 4 * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; i++) h_params[i * 4 + 1] = k0;
        CUDA_CHECK(cudaMemcpy(d_params, h_params.data(), N * 4 * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Allocate trajectory output arrays
    float *d_traj_x, *d_traj_y, *d_traj_yaw;
    int *d_traj_len;
    size_t traj_bytes = (size_t)N * MAX_TRAJ_STEPS * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_traj_x, traj_bytes));
    CUDA_CHECK(cudaMalloc(&d_traj_y, traj_bytes));
    CUDA_CHECK(cudaMalloc(&d_traj_yaw, traj_bytes));
    CUDA_CHECK(cudaMalloc(&d_traj_len, N * sizeof(int)));

    // Kernel 2: trajectory optimization
    float cost_th = 0.1f;
    trajectory_optimization_kernel<<<gridSize, blockSize>>>(
        d_targets, d_params,
        d_traj_x, d_traj_y, d_traj_yaw, d_traj_len,
        cost_th, MAX_ITER, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    std::vector<float> h_traj_x(N * MAX_TRAJ_STEPS);
    std::vector<float> h_traj_y(N * MAX_TRAJ_STEPS);
    std::vector<float> h_traj_yaw(N * MAX_TRAJ_STEPS);
    std::vector<int> h_traj_len(N);
    CUDA_CHECK(cudaMemcpy(h_traj_x.data(), d_traj_x, traj_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_traj_y.data(), d_traj_y, traj_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_traj_yaw.data(), d_traj_yaw, traj_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_traj_len.data(), d_traj_len, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Build result
    std::vector<TrajectoryResult> results(N);
    for (int i = 0; i < N; i++) {
        int len = h_traj_len[i];
        results[i].x.resize(len);
        results[i].y.resize(len);
        results[i].yaw.resize(len);
        int base = i * MAX_TRAJ_STEPS;
        for (int j = 0; j < len; j++) {
            results[i].x[j]   = h_traj_x[base + j];
            results[i].y[j]   = h_traj_y[base + j];
            results[i].yaw[j] = h_traj_yaw[base + j];
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_params));
    CUDA_CHECK(cudaFree(d_traj_x));
    CUDA_CHECK(cudaFree(d_traj_y));
    CUDA_CHECK(cudaFree(d_traj_yaw));
    CUDA_CHECK(cudaFree(d_traj_len));

    return results;
}

// ===========================================================================
// Test functions (mirror the original CPU test cases)
// ===========================================================================

std::vector<TrajectoryResult> uniform_terminal_state_sample_test(const float *d_lookup, int lookup_rows) {
    float k0 = 0.0f;
    int nxy = 5;
    int nh = 3;
    int d = 20;
    float a_min = -45.0f / 180.0f * PI_F;
    float a_max = +45.0f / 180.0f * PI_F;
    float p_min = -45.0f / 180.0f * PI_F;
    float p_max = +45.0f / 180.0f * PI_F;

    auto states = calc_uniform_polar_states(nxy, nh, d, a_min, a_max, p_min, p_max);
    return generate_path_cuda(states, d_lookup, lookup_rows, k0);
}

std::vector<TrajectoryResult> biased_terminal_state_sample_test(const float *d_lookup, int lookup_rows) {
    float k0 = 0.0f;
    int nxy = 30;
    int nh = 2;
    int d = 20;
    float a_min = -45.0f / 180.0f * PI_F;
    float a_max = +45.0f / 180.0f * PI_F;
    float p_min = -20.0f / 180.0f * PI_F;
    float p_max = +20.0f / 180.0f * PI_F;
    int ns = 100;
    float goal_angle = 0.0f;

    auto states = calc_biased_polar_states(goal_angle, ns, nxy, nh, d, a_min, a_max, p_min, p_max);
    return generate_path_cuda(states, d_lookup, lookup_rows, k0);
}

std::vector<TrajectoryResult> lane_state_sample_test(const float *d_lookup, int lookup_rows) {
    float k0 = 0.0f;
    float l_center = 10.0f;
    float l_heading = 90.0f / 180.0f * PI_F;
    float l_width = 3.0f;
    float v_width = 1.0f;
    int d = 10;
    int nxy = 5;

    auto states = calc_lane_states(l_center, l_heading, l_width, v_width, (float)d, nxy);
    return generate_path_cuda(states, d_lookup, lookup_rows, k0);
}

// ===========================================================================
// Main
// ===========================================================================

int main() {
    // Load lookup table from CSV
    auto lookup_table = read_lookup_table("../../lookuptable.csv");
    if (lookup_table.empty())
        lookup_table = read_lookup_table("../lookuptable.csv");
    if (lookup_table.empty())
        lookup_table = read_lookup_table("lookuptable.csv");
    int M = (int)lookup_table.size();
    printf("Loaded lookup table with %d entries\n", M);

    // Flatten to contiguous array (6 floats per row)
    std::vector<float> h_lookup(M * 6);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < 6; j++) {
            h_lookup[i * 6 + j] = lookup_table[i][j];
        }
    }

    // Transfer lookup table to device (persists across all test cases)
    float *d_lookup;
    CUDA_CHECK(cudaMalloc(&d_lookup, M * 6 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lookup, h_lookup.data(), M * 6 * sizeof(float), cudaMemcpyHostToDevice));

    // Test 1: Uniform sampling
    printf("\n=== Uniform Terminal State Sampling ===\n");
    auto traj1 = uniform_terminal_state_sample_test(d_lookup, M);
    printf("Generated %d trajectories\n", (int)traj1.size());
    for (int i = 0; i < (int)traj1.size(); i++) {
        if (traj1[i].x.size() > 0) {
            int last = (int)traj1[i].x.size() - 1;
            printf("  Traj %2d: %3d pts, end=(%.2f, %.2f, %.2f)\n",
                   i, (int)traj1[i].x.size(),
                   traj1[i].x[last], traj1[i].y[last], traj1[i].yaw[last]);
        }
    }

    // Test 2: Biased sampling
    printf("\n=== Biased Terminal State Sampling ===\n");
    auto traj2 = biased_terminal_state_sample_test(d_lookup, M);
    printf("Generated %d trajectories\n", (int)traj2.size());
    for (int i = 0; i < (int)traj2.size(); i++) {
        if (traj2[i].x.size() > 0) {
            int last = (int)traj2[i].x.size() - 1;
            printf("  Traj %2d: %3d pts, end=(%.2f, %.2f, %.2f)\n",
                   i, (int)traj2[i].x.size(),
                   traj2[i].x[last], traj2[i].y[last], traj2[i].yaw[last]);
        }
    }

    // Test 3: Lane sampling
    printf("\n=== Lane State Sampling ===\n");
    auto traj3 = lane_state_sample_test(d_lookup, M);
    printf("Generated %d trajectories\n", (int)traj3.size());
    for (int i = 0; i < (int)traj3.size(); i++) {
        if (traj3[i].x.size() > 0) {
            int last = (int)traj3[i].x.size() - 1;
            printf("  Traj %2d: %3d pts, end=(%.2f, %.2f, %.2f)\n",
                   i, (int)traj3[i].x.size(),
                   traj3[i].x[last], traj3[i].y[last], traj3[i].yaw[last]);
        }
    }

    CUDA_CHECK(cudaFree(d_lookup));
    printf("\nDone.\n");
    return 0;
}
