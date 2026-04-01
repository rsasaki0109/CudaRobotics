/*************************************************************************
    > File Name: aco_tsp.cu
    > CUDA Ant Colony Optimization for TSP
    > N_ANTS=4096, N_CITIES=50 (random in [0,100]x[0,100])
    > Kernels:
    >   - construct_tour_kernel: 1 thread per ant
    >   - update_pheromone_kernel: update pheromone matrix
    > 500 iterations
    > Visualization: best tour on city map
    > Output: gif/aco_tsp.gif
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------
static const int N_ANTS = 4096;
static const int N_CITIES = 50;
static const int MAX_ITER = 500;
static const float ALPHA = 1.0f;      // Pheromone importance
static const float BETA = 3.0f;       // Heuristic importance
static const float RHO = 0.1f;        // Evaporation rate
static const float Q = 100.0f;        // Pheromone deposit constant
static const float TAU_MIN = 0.01f;
static const float TAU_MAX = 10.0f;
static const float CITY_RANGE = 100.0f;

// -------------------------------------------------------------------------
// Kernels
// -------------------------------------------------------------------------
__global__ void init_curand_kernel(curandState* states, int N, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curand_init(seed, i, 0, &states[i]);
}

// Distance matrix computation
__global__ void compute_distances_kernel(
    const float* city_x, const float* city_y,
    float* dist_matrix, int n_cities)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n_cities || j >= n_cities) return;

    float dx = city_x[i] - city_x[j];
    float dy = city_y[i] - city_y[j];
    dist_matrix[i * n_cities + j] = sqrtf(dx * dx + dy * dy);
}

// Construct tour for each ant
__global__ void construct_tour_kernel(
    const float* pheromone,       // [N_CITIES * N_CITIES]
    const float* dist_matrix,     // [N_CITIES * N_CITIES]
    int* tours,                   // [N_ANTS * N_CITIES]
    float* tour_lengths,          // [N_ANTS]
    curandState* rng,
    int n_ants, int n_cities,
    float alpha, float beta)
{
    int ant = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant >= n_ants) return;

    curandState local_state = rng[ant];

    // Visited array (bitmask for n_cities <= 64)
    // For N_CITIES=50, use a simple bool array in local memory
    bool visited[50];
    for (int i = 0; i < n_cities; i++) visited[i] = false;

    // Start from random city
    int current = (int)(curand_uniform(&local_state) * n_cities) % n_cities;
    tours[ant * n_cities + 0] = current;
    visited[current] = true;

    float total_dist = 0.0f;

    for (int step = 1; step < n_cities; step++) {
        // Compute probabilities
        float prob_sum = 0.0f;
        float probs[50];

        for (int j = 0; j < n_cities; j++) {
            if (visited[j]) {
                probs[j] = 0.0f;
            } else {
                float tau = pheromone[current * n_cities + j];
                float eta = 1.0f / fmaxf(dist_matrix[current * n_cities + j], 1e-6f);
                probs[j] = powf(tau, alpha) * powf(eta, beta);
                prob_sum += probs[j];
            }
        }

        // Roulette wheel selection
        float r = curand_uniform(&local_state) * prob_sum;
        float cumsum = 0.0f;
        int next = -1;
        for (int j = 0; j < n_cities; j++) {
            if (!visited[j]) {
                cumsum += probs[j];
                if (cumsum >= r) {
                    next = j;
                    break;
                }
            }
        }
        // Fallback: pick first unvisited
        if (next < 0) {
            for (int j = 0; j < n_cities; j++) {
                if (!visited[j]) { next = j; break; }
            }
        }

        tours[ant * n_cities + step] = next;
        visited[next] = true;
        total_dist += dist_matrix[current * n_cities + next];
        current = next;
    }

    // Return to start
    total_dist += dist_matrix[current * n_cities + tours[ant * n_cities + 0]];
    tour_lengths[ant] = total_dist;

    rng[ant] = local_state;
}

// Evaporate pheromone
__global__ void evaporate_pheromone_kernel(
    float* pheromone, int n_cities, float rho, float tau_min)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n_cities || j >= n_cities) return;
    float val = pheromone[i * n_cities + j] * (1.0f - rho);
    pheromone[i * n_cities + j] = fmaxf(val, tau_min);
}

// Deposit pheromone from best ant
__global__ void deposit_pheromone_kernel(
    float* pheromone, const int* best_tour,
    float best_length, float Q_val,
    int n_cities, float tau_max)
{
    int step = blockIdx.x * blockDim.x + threadIdx.x;
    if (step >= n_cities) return;

    int from = best_tour[step];
    int to = best_tour[(step + 1) % n_cities];
    float deposit = Q_val / best_length;
    // Atomic add for both directions
    float new_val = atomicAdd(&pheromone[from * n_cities + to], deposit);
    if (new_val + deposit > tau_max)
        pheromone[from * n_cities + to] = tau_max;
    new_val = atomicAdd(&pheromone[to * n_cities + from], deposit);
    if (new_val + deposit > tau_max)
        pheromone[to * n_cities + from] = tau_max;
}

// Find minimum tour length
__global__ void find_min_tour_kernel(
    const float* tour_lengths, int* best_idx, float* best_length, int N)
{
    __shared__ float s_len[256];
    __shared__ int s_idx[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    s_len[tid] = (i < N) ? tour_lengths[i] : FLT_MAX;
    s_idx[tid] = i;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_len[tid + s] < s_len[tid]) {
            s_len[tid] = s_len[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Atomic min across blocks
        int* g_len_int = (int*)best_length;
        int old_val = *g_len_int;
        int new_val = __float_as_int(s_len[0]);
        while (s_len[0] < __int_as_float(old_val)) {
            int assumed = old_val;
            old_val = atomicCAS(g_len_int, assumed, new_val);
            if (old_val == assumed) {
                *best_idx = s_idx[0];
                break;
            }
        }
    }
}

// -------------------------------------------------------------------------
// Visualization
// -------------------------------------------------------------------------
static const int VIS_W = 600;
static const int VIS_H = 600;

void draw_tour(cv::Mat& img, const vector<float>& cx, const vector<float>& cy,
               const vector<int>& tour, int n_cities, float best_len, int iter) {
    img = cv::Scalar(30, 30, 30);

    float margin = 50.0f;
    float scale = (VIS_W - 2 * margin) / CITY_RANGE;

    // Draw edges
    for (int i = 0; i < n_cities; i++) {
        int from = tour[i];
        int to = tour[(i + 1) % n_cities];
        cv::Point p1(margin + cx[from] * scale, margin + cy[from] * scale);
        cv::Point p2(margin + cx[to] * scale, margin + cy[to] * scale);
        cv::line(img, p1, p2, cv::Scalar(0, 200, 255), 2);
    }

    // Draw cities
    for (int i = 0; i < n_cities; i++) {
        cv::Point p(margin + cx[i] * scale, margin + cy[i] * scale);
        cv::circle(img, p, 5, cv::Scalar(255, 255, 255), -1);
        cv::circle(img, p, 5, cv::Scalar(0, 128, 255), 1);
    }

    char buf[256];
    snprintf(buf, sizeof(buf), "ACO TSP: %d cities, %d ants, Iter=%d, Best=%.1f",
             n_cities, N_ANTS, iter, best_len);
    cv::putText(img, buf, cv::Point(10, VIS_H - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255));
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    cout << "=== CUDA Ant Colony Optimization for TSP ===" << endl;
    cout << "N_ANTS=" << N_ANTS << ", N_CITIES=" << N_CITIES << endl;

    // Generate random cities
    srand(42);
    vector<float> h_city_x(N_CITIES), h_city_y(N_CITIES);
    for (int i = 0; i < N_CITIES; i++) {
        h_city_x[i] = ((float)rand() / RAND_MAX) * CITY_RANGE;
        h_city_y[i] = ((float)rand() / RAND_MAX) * CITY_RANGE;
    }

    const int threads = 256;
    const int ant_blocks = (N_ANTS + threads - 1) / threads;

    // Device memory
    float *d_city_x, *d_city_y, *d_dist_matrix;
    float *d_pheromone;
    int *d_tours;
    float *d_tour_lengths;
    int *d_best_idx;
    float *d_best_length;
    curandState *d_rng;

    CUDA_CHECK(cudaMalloc(&d_city_x, N_CITIES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_city_y, N_CITIES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist_matrix, N_CITIES * N_CITIES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pheromone, N_CITIES * N_CITIES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tours, N_ANTS * N_CITIES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tour_lengths, N_ANTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_best_idx, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_length, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rng, N_ANTS * sizeof(curandState)));

    CUDA_CHECK(cudaMemcpy(d_city_x, h_city_x.data(), N_CITIES * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_city_y, h_city_y.data(), N_CITIES * sizeof(float), cudaMemcpyHostToDevice));

    // Compute distance matrix
    dim3 grid2d((N_CITIES + 15) / 16, (N_CITIES + 15) / 16);
    dim3 block2d(16, 16);
    compute_distances_kernel<<<grid2d, block2d>>>(d_city_x, d_city_y, d_dist_matrix, N_CITIES);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize pheromone to 1.0
    {
        vector<float> init_pheromone(N_CITIES * N_CITIES, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_pheromone, init_pheromone.data(),
            N_CITIES * N_CITIES * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Initialize cuRAND
    init_curand_kernel<<<ant_blocks, threads>>>(d_rng, N_ANTS, 99);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Video
    string avi_path = "gif/aco_tsp.avi";
    string gif_path = "gif/aco_tsp.gif";
    cv::VideoWriter video(avi_path, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 15,
                          cv::Size(VIS_W, VIS_H));

    float global_best_length = FLT_MAX;
    vector<int> global_best_tour(N_CITIES);

    auto t_start = chrono::high_resolution_clock::now();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Construct tours
        construct_tour_kernel<<<ant_blocks, threads>>>(
            d_pheromone, d_dist_matrix, d_tours, d_tour_lengths,
            d_rng, N_ANTS, N_CITIES, ALPHA, BETA);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Find best tour this iteration
        float init_len = FLT_MAX;
        CUDA_CHECK(cudaMemcpy(d_best_length, &init_len, sizeof(float), cudaMemcpyHostToDevice));
        int init_idx = 0;
        CUDA_CHECK(cudaMemcpy(d_best_idx, &init_idx, sizeof(int), cudaMemcpyHostToDevice));

        find_min_tour_kernel<<<ant_blocks, threads>>>(
            d_tour_lengths, d_best_idx, d_best_length, N_ANTS);
        CUDA_CHECK(cudaDeviceSynchronize());

        float iter_best_length;
        int iter_best_idx;
        CUDA_CHECK(cudaMemcpy(&iter_best_length, d_best_length, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&iter_best_idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));

        // Update global best
        if (iter_best_length < global_best_length) {
            global_best_length = iter_best_length;
            CUDA_CHECK(cudaMemcpy(global_best_tour.data(),
                &d_tours[iter_best_idx * N_CITIES],
                N_CITIES * sizeof(int), cudaMemcpyDeviceToHost));
        }

        // Evaporate pheromone
        evaporate_pheromone_kernel<<<grid2d, block2d>>>(
            d_pheromone, N_CITIES, RHO, TAU_MIN);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Deposit pheromone from global best tour
        {
            int* d_global_best_tour;
            CUDA_CHECK(cudaMalloc(&d_global_best_tour, N_CITIES * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_global_best_tour, global_best_tour.data(),
                N_CITIES * sizeof(int), cudaMemcpyHostToDevice));

            int pher_blocks = (N_CITIES + threads - 1) / threads;
            deposit_pheromone_kernel<<<pher_blocks, threads>>>(
                d_pheromone, d_global_best_tour, global_best_length, Q,
                N_CITIES, TAU_MAX);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaFree(d_global_best_tour);
        }

        if (iter % 50 == 0)
            printf("Iter %4d: best = %.2f, iter_best = %.2f\n",
                   iter, global_best_length, iter_best_length);

        // Visualize every 5 iterations
        if (iter % 5 == 0) {
            cv::Mat frame(VIS_H, VIS_W, CV_8UC3);
            draw_tour(frame, h_city_x, h_city_y, global_best_tour,
                      N_CITIES, global_best_length, iter);
            video.write(frame);
        }
    }

    auto t_end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(t_end - t_start).count();

    printf("\n=== Results ===\n");
    printf("Best tour length: %.2f\n", global_best_length);
    printf("Total time: %.3f s\n", elapsed);

    video.release();
    cout << "Video saved to " << avi_path << endl;

    string cmd = "ffmpeg -y -i " + avi_path + " -vf 'fps=15,scale=400:-1' -loop 0 " + gif_path + " 2>/dev/null";
    system(cmd.c_str());
    cout << "GIF saved to " << gif_path << endl;

    cudaFree(d_city_x);
    cudaFree(d_city_y);
    cudaFree(d_dist_matrix);
    cudaFree(d_pheromone);
    cudaFree(d_tours);
    cudaFree(d_tour_lengths);
    cudaFree(d_best_idx);
    cudaFree(d_best_length);
    cudaFree(d_rng);

    return 0;
}
