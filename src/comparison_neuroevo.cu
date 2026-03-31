/*************************************************************************
    Neuroevolution: CPU (sequential, 100 individuals) vs GPU (4096 individuals)
    Side-by-side evolution speed comparison
    Left panel:  CPU evolution progress
    Right panel: GPU evolution progress
    Output: gif/comparison_neuroevo.gif
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "gpu_neural_net.cuh"
#include "gpu_environments.cuh"
#include "gpu_genetic.cuh"

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;
using namespace std::chrono;

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------
static const int CPU_POP    = 100;
static const int GPU_POP    = 4096;
static const int N_GEN      = 500;
static const int INPUT_DIM  = 4;
static const int OUTPUT_DIM = 1;
static const int TOURN_K    = 5;
static const float CROSS_R  = 0.8f;
static const float MUT_S    = 0.1f;
static const int ELITE_N    = 10;

static const int IMG_W = 800;
static const int IMG_H = 600;

// -------------------------------------------------------------------------
// CPU Neural Network forward pass
// -------------------------------------------------------------------------
void cpu_nn_forward(const float* weights, const float* input, float* output,
                    int in_dim, int out_dim)
{
    float h1[NN_HIDDEN1], h2[NN_HIDDEN2];

    int offset = 0;
    for (int j = 0; j < NN_HIDDEN1; j++) {
        float sum = 0;
        for (int i = 0; i < in_dim; i++)
            sum += weights[offset + j * in_dim + i] * input[i];
        sum += weights[in_dim * NN_HIDDEN1 + j];
        h1[j] = tanhf(sum);
    }
    offset = in_dim * NN_HIDDEN1 + NN_HIDDEN1;

    for (int j = 0; j < NN_HIDDEN2; j++) {
        float sum = 0;
        for (int i = 0; i < NN_HIDDEN1; i++)
            sum += weights[offset + j * NN_HIDDEN1 + i] * h1[i];
        sum += weights[offset + NN_HIDDEN1 * NN_HIDDEN2 + j];
        h2[j] = tanhf(sum);
    }
    offset += NN_HIDDEN1 * NN_HIDDEN2 + NN_HIDDEN2;

    for (int j = 0; j < out_dim; j++) {
        float sum = 0;
        for (int i = 0; i < NN_HIDDEN2; i++)
            sum += weights[offset + j * NN_HIDDEN2 + i] * h2[i];
        sum += weights[offset + NN_HIDDEN2 * out_dim + j];
        output[j] = sum;
    }
}

// -------------------------------------------------------------------------
// CPU Cart-Pole evaluation
// -------------------------------------------------------------------------
float cpu_evaluate(const float* weights, int n_weights)
{
    float x = 0, x_dot = 0, theta = 0.05f, theta_dot = 0;
    int steps = 0;
    bool done = false;

    while (!done) {
        float obs[4] = {x, x_dot, theta, theta_dot};
        float out[1];
        cpu_nn_forward(weights, obs, out, INPUT_DIM, OUTPUT_DIM);
        float action = tanhf(out[0]);

        float force = action > 0 ? 10.0f : -10.0f;
        float total_mass = 1.1f;
        float pole_ml = 0.05f;
        float costh = cosf(theta), sinth = sinf(theta);
        float temp = (force + pole_ml * theta_dot * theta_dot * sinth) / total_mass;
        float th_acc = (9.8f * sinth - costh * temp) /
                       (0.5f * (4.0f/3.0f - 0.1f * costh * costh / total_mass));
        float x_acc = temp - pole_ml * th_acc * costh / total_mass;

        x += 0.02f * x_dot;
        x_dot += 0.02f * x_acc;
        theta += 0.02f * theta_dot;
        theta_dot += 0.02f * th_acc;
        steps++;

        if (fabsf(x) > 2.4f || fabsf(theta) > 12.0f * M_PI / 180.0f || steps >= 200)
            done = true;
    }
    return (float)steps;
}

// -------------------------------------------------------------------------
// CPU tournament selection
// -------------------------------------------------------------------------
int cpu_tournament(const vector<float>& fitness, int pop_size, int k, mt19937& rng_gen)
{
    uniform_int_distribution<int> dist(0, pop_size - 1);
    int best = dist(rng_gen);
    for (int i = 1; i < k; i++) {
        int c = dist(rng_gen);
        if (fitness[c] > fitness[best]) best = c;
    }
    return best;
}

// -------------------------------------------------------------------------
// CPU one generation
// -------------------------------------------------------------------------
struct CPUEvolver {
    int pop_size, n_weights;
    vector<float> population;   // [pop_size * n_weights]
    vector<float> new_pop;
    vector<float> fitness;
    mt19937 rng_gen;
    normal_distribution<float> normal_dist;
    uniform_real_distribution<float> uniform_dist;

    CPUEvolver(int ps, int nw) : pop_size(ps), n_weights(nw),
        population(ps * nw), new_pop(ps * nw), fitness(ps),
        rng_gen(42), normal_dist(0.0f, 0.5f), uniform_dist(0.0f, 1.0f) {
        for (auto& w : population) w = normal_dist(rng_gen);
    }

    float run_generation() {
        // Evaluate
        for (int i = 0; i < pop_size; i++) {
            fitness[i] = cpu_evaluate(population.data() + i * n_weights, n_weights);
        }

        // Find best
        float best = *max_element(fitness.begin(), fitness.end());

        // Elite
        vector<int> indices(pop_size);
        iota(indices.begin(), indices.end(), 0);
        partial_sort(indices.begin(), indices.begin() + min(ELITE_N, pop_size),
                     indices.end(),
                     [&](int a, int b) { return fitness[a] > fitness[b]; });

        int elite = min(ELITE_N, pop_size);
        for (int i = 0; i < elite; i++) {
            memcpy(new_pop.data() + i * n_weights,
                   population.data() + indices[i] * n_weights,
                   n_weights * sizeof(float));
        }

        // Reproduce
        normal_distribution<float> mut_dist(0.0f, MUT_S);
        for (int i = elite; i < pop_size; i++) {
            int p1 = cpu_tournament(fitness, pop_size, TOURN_K, rng_gen);
            int p2 = cpu_tournament(fitness, pop_size, TOURN_K, rng_gen);

            for (int w = 0; w < n_weights; w++) {
                float gene;
                if (uniform_dist(rng_gen) < CROSS_R) {
                    gene = (uniform_dist(rng_gen) < 0.5f)
                           ? population[p1 * n_weights + w]
                           : population[p2 * n_weights + w];
                } else {
                    gene = population[p1 * n_weights + w];
                }
                gene += mut_dist(rng_gen);
                new_pop[i * n_weights + w] = gene;
            }
        }

        swap(population, new_pop);
        return best;
    }
};

// -------------------------------------------------------------------------
// Draw panel (one side)
// -------------------------------------------------------------------------
void draw_panel(cv::Mat& img, int x_offset, int panel_w, const char* title,
                const vector<float>& best_hist, const vector<float>& avg_hist,
                int gen, double elapsed_ms, int pop_size)
{
    // Clear panel
    img(cv::Rect(x_offset, 0, panel_w, IMG_H)) = cv::Scalar(255, 255, 255);

    // Title
    cv::putText(img, title, cv::Point(x_offset + 20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

    // Stats
    char buf[128];
    snprintf(buf, sizeof(buf), "Pop: %d | Gen: %d", pop_size, gen);
    cv::putText(img, buf, cv::Point(x_offset + 20, 55),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(80, 80, 80), 1);

    snprintf(buf, sizeof(buf), "Time: %.1f ms/gen", gen > 0 ? elapsed_ms / gen : 0.0);
    cv::putText(img, buf, cv::Point(x_offset + 20, 75),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(80, 80, 80), 1);

    if (!best_hist.empty()) {
        snprintf(buf, sizeof(buf), "Best: %.0f | Avg: %.1f",
                 best_hist.back(), avg_hist.back());
        cv::putText(img, buf, cv::Point(x_offset + 20, 95),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 180), 1);
    }

    // Graph area
    int gx = x_offset + 40;
    int gy = 120;
    int gw = panel_w - 80;
    int gh = IMG_H - 200;

    // Axes
    cv::line(img, cv::Point(gx, gy), cv::Point(gx, gy + gh), cv::Scalar(0, 0, 0), 1);
    cv::line(img, cv::Point(gx, gy + gh), cv::Point(gx + gw, gy + gh),
             cv::Scalar(0, 0, 0), 1);

    // Y gridlines
    for (int v = 0; v <= 200; v += 50) {
        int y = gy + gh - (int)(v / 200.0f * gh);
        snprintf(buf, sizeof(buf), "%d", v);
        cv::putText(img, buf, cv::Point(x_offset + 10, y + 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1);
        cv::line(img, cv::Point(gx, y), cv::Point(gx + gw, y),
                 cv::Scalar(230, 230, 230), 1);
    }

    if (best_hist.empty()) return;

    int n = (int)best_hist.size();
    float max_gen = fmaxf((float)N_GEN, 1.0f);

    for (int i = 1; i < n; i++) {
        int x1 = gx + (int)((i - 1) / max_gen * gw);
        int x2 = gx + (int)(i / max_gen * gw);

        int y1b = gy + gh - (int)(best_hist[i-1] / 200.0f * gh);
        int y2b = gy + gh - (int)(best_hist[i] / 200.0f * gh);
        cv::line(img, cv::Point(x1, y1b), cv::Point(x2, y2b), cv::Scalar(0, 0, 200), 2);

        int y1a = gy + gh - (int)(avg_hist[i-1] / 200.0f * gh);
        int y2a = gy + gh - (int)(avg_hist[i] / 200.0f * gh);
        cv::line(img, cv::Point(x1, y1a), cv::Point(x2, y2a), cv::Scalar(200, 100, 0), 1);
    }

    // Legend
    cv::line(img, cv::Point(gx + gw - 100, gy + 10), cv::Point(gx + gw - 80, gy + 10),
             cv::Scalar(0, 0, 200), 2);
    cv::putText(img, "Best", cv::Point(gx + gw - 75, gy + 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 0, 200), 1);
    cv::line(img, cv::Point(gx + gw - 100, gy + 25), cv::Point(gx + gw - 80, gy + 25),
             cv::Scalar(200, 100, 0), 1);
    cv::putText(img, "Avg", cv::Point(gx + gw - 75, gy + 29),
                cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 100, 0), 1);

    // Speed bar at bottom
    snprintf(buf, sizeof(buf), "Total: %.1f s", elapsed_ms / 1000.0);
    cv::putText(img, buf, cv::Point(x_offset + 20, IMG_H - 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main()
{
    int n_weights = nn_total_weights(INPUT_DIM, OUTPUT_DIM);
    printf("Neuroevolution Comparison: CPU (pop=%d) vs GPU (pop=%d)\n",
           CPU_POP, GPU_POP);
    printf("Weights per individual: %d\n", n_weights);

    // ----- GPU setup -----
    float *d_pop1, *d_pop2, *d_fitness;
    curandState *d_rng;
    int *d_elite_indices;

    CUDA_CHECK(cudaMalloc(&d_pop1, GPU_POP * n_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pop2, GPU_POP * n_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fitness, GPU_POP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rng, GPU_POP * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_elite_indices, ELITE_N * sizeof(int)));

    int block = 256;
    int grid = (GPU_POP + block - 1) / block;

    init_curand_kernel<<<grid, block>>>(d_rng, GPU_POP, 42ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Init GPU population
    // Reuse init kernel from neuroevo.cu (inline here)
    {
        // Simple init: small random weights
        vector<float> h_init(GPU_POP * n_weights);
        mt19937 rng(42);
        normal_distribution<float> dist(0.0f, 0.5f);
        for (auto& w : h_init) w = dist(rng);
        CUDA_CHECK(cudaMemcpy(d_pop1, h_init.data(),
                              GPU_POP * n_weights * sizeof(float), cudaMemcpyHostToDevice));
    }

    vector<float> h_gpu_fitness(GPU_POP);

    // ----- CPU setup -----
    CPUEvolver cpu_evolver(CPU_POP, n_weights);

    // ----- Video -----
    const string avi_path = "gif/comparison_neuroevo.avi";
    const string gif_path = "gif/comparison_neuroevo.gif";
    cv::VideoWriter writer(avi_path, cv::VideoWriter::fourcc('X','V','I','D'),
                           15, cv::Size(IMG_W, IMG_H));
    if (!writer.isOpened()) {
        fprintf(stderr, "Failed to open video writer\n");
        return 1;
    }

    cv::Mat frame(IMG_H, IMG_W, CV_8UC3, cv::Scalar(255, 255, 255));

    vector<float> cpu_best_hist, cpu_avg_hist;
    vector<float> gpu_best_hist, gpu_avg_hist;

    float *d_cur = d_pop1, *d_next = d_pop2;

    double cpu_total_ms = 0, gpu_total_ms = 0;

    // ----- Evolution loop -----
    for (int gen = 0; gen < N_GEN; gen++) {
        // --- CPU generation ---
        auto t0 = high_resolution_clock::now();
        float cpu_best = cpu_evolver.run_generation();
        auto t1 = high_resolution_clock::now();
        double cpu_ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
        cpu_total_ms += cpu_ms;

        float cpu_avg = 0;
        for (int i = 0; i < CPU_POP; i++) cpu_avg += cpu_evolver.fitness[i];
        cpu_avg /= CPU_POP;
        cpu_best_hist.push_back(cpu_best);
        cpu_avg_hist.push_back(cpu_avg);

        // --- GPU generation ---
        auto t2 = high_resolution_clock::now();
        evaluate_fitness_kernel<<<grid, block>>>(
            d_cur, d_fitness, d_rng, GPU_POP, n_weights, INPUT_DIM, OUTPUT_DIM);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_gpu_fitness.data(), d_fitness,
                              GPU_POP * sizeof(float), cudaMemcpyDeviceToHost));

        float gpu_best = -1, gpu_avg_sum = 0;
        int gpu_best_idx = 0;
        for (int i = 0; i < GPU_POP; i++) {
            gpu_avg_sum += h_gpu_fitness[i];
            if (h_gpu_fitness[i] > gpu_best) { gpu_best = h_gpu_fitness[i]; gpu_best_idx = i; }
        }

        // Elite indices
        vector<int> indices(GPU_POP);
        iota(indices.begin(), indices.end(), 0);
        partial_sort(indices.begin(), indices.begin() + ELITE_N, indices.end(),
                     [&](int a, int b) { return h_gpu_fitness[a] > h_gpu_fitness[b]; });
        CUDA_CHECK(cudaMemcpy(d_elite_indices, indices.data(),
                              ELITE_N * sizeof(int), cudaMemcpyHostToDevice));

        reproduce_kernel<<<grid, block>>>(
            d_cur, d_fitness, d_next, d_elite_indices, d_rng,
            GPU_POP, n_weights, TOURN_K, CROSS_R, MUT_S, ELITE_N);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto t3 = high_resolution_clock::now();
        double gpu_ms = duration_cast<microseconds>(t3 - t2).count() / 1000.0;
        gpu_total_ms += gpu_ms;

        float gpu_avg = gpu_avg_sum / GPU_POP;
        gpu_best_hist.push_back(gpu_best);
        gpu_avg_hist.push_back(gpu_avg);

        // Swap GPU buffers
        float* tmp = d_cur; d_cur = d_next; d_next = tmp;

        // Print progress
        if (gen % 20 == 0 || gen == N_GEN - 1) {
            printf("Gen %3d | CPU: best=%.0f avg=%.1f (%.1fms) | GPU: best=%.0f avg=%.1f (%.1fms)\n",
                   gen, cpu_best, cpu_avg, cpu_ms, gpu_best, gpu_avg, gpu_ms);
        }

        // Visualization every 5 generations
        if (gen % 5 == 0 || gen == N_GEN - 1) {
            int panel_w = IMG_W / 2;

            char cpu_title[64], gpu_title[64];
            snprintf(cpu_title, sizeof(cpu_title), "CPU (Sequential, pop=%d)", CPU_POP);
            snprintf(gpu_title, sizeof(gpu_title), "CUDA (Parallel, pop=%d)", GPU_POP);

            draw_panel(frame, 0, panel_w, cpu_title,
                       cpu_best_hist, cpu_avg_hist, gen, cpu_total_ms, CPU_POP);
            draw_panel(frame, panel_w, panel_w, gpu_title,
                       gpu_best_hist, gpu_avg_hist, gen, gpu_total_ms, GPU_POP);

            // Separator
            cv::line(frame, cv::Point(panel_w, 0), cv::Point(panel_w, IMG_H),
                     cv::Scalar(100, 100, 100), 2);

            writer.write(frame);
        }
    }

    // Hold final frame
    for (int i = 0; i < 30; i++) writer.write(frame);

    writer.release();

    // Summary
    printf("\n=== Summary ===\n");
    printf("CPU: %.1f ms total, %.2f ms/gen (pop=%d)\n",
           cpu_total_ms, cpu_total_ms / N_GEN, CPU_POP);
    printf("GPU: %.1f ms total, %.2f ms/gen (pop=%d)\n",
           gpu_total_ms, gpu_total_ms / N_GEN, GPU_POP);
    printf("GPU speedup: %.1fx (with %.1fx more individuals)\n",
           cpu_total_ms / gpu_total_ms * GPU_POP / CPU_POP,
           (float)GPU_POP / CPU_POP);
    printf("Per-individual speedup: %.1fx\n",
           (cpu_total_ms / CPU_POP) / (gpu_total_ms / GPU_POP));

    // Convert to gif
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -y -i %s -vf 'fps=15,scale=400:-1' -loop 0 %s 2>/dev/null",
             avi_path.c_str(), gif_path.c_str());
    system(cmd);
    printf("Saved: %s\n", gif_path.c_str());

    // Cleanup
    CUDA_CHECK(cudaFree(d_pop1));
    CUDA_CHECK(cudaFree(d_pop2));
    CUDA_CHECK(cudaFree(d_fitness));
    CUDA_CHECK(cudaFree(d_rng));
    CUDA_CHECK(cudaFree(d_elite_indices));

    return 0;
}
