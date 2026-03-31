/*************************************************************************
    GPU-Parallel Neuroevolution for Cart-Pole
    - Population of 4096 neural networks evolved on GPU
    - Fixed topology: 4 -> 32 -> 16 -> 1 (tanh hidden, linear output)
    - Genetic operators: tournament selection, uniform crossover, Gaussian mutation
    - Visualization: left = best individual cart-pole replay, right = fitness graph
    Output: gif/neuroevo.gif
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
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

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------
static const int POP_SIZE       = 4096;
static const int N_GENERATIONS  = 500;
static const int INPUT_DIM      = 4;   // Cart-Pole: x, x_dot, theta, theta_dot
static const int OUTPUT_DIM     = 1;   // force direction
static const int TOURNAMENT_K   = 5;
static const float CROSSOVER_RATE = 0.8f;
static const float MUTATION_SIGMA = 0.1f;
static const int ELITE_COUNT    = 10;

static const int IMG_W = 800;
static const int IMG_H = 600;

// -------------------------------------------------------------------------
// Initialize population with small random weights
// -------------------------------------------------------------------------
__global__ void init_population_kernel(float* population, curandState* rng,
                                       int pop_size, int n_weights)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    curandState local_rng = rng[idx];
    for (int w = 0; w < n_weights; w++) {
        population[idx * n_weights + w] = curand_normal(&local_rng) * 0.5f;
    }
    rng[idx] = local_rng;
}

// -------------------------------------------------------------------------
// Draw Cart-Pole visualization on left half of image
// -------------------------------------------------------------------------
void draw_cartpole(cv::Mat& img, float x, float theta, int gen, float best_fit, int step)
{
    int left_w = IMG_W / 2;
    int cx = left_w / 2;
    int cy = IMG_H * 2 / 3;

    // Clear left half
    img(cv::Rect(0, 0, left_w, IMG_H)) = cv::Scalar(255, 255, 255);

    // Title
    char buf[128];
    snprintf(buf, sizeof(buf), "Gen %d | Best Fitness: %.0f | Step: %d", gen, best_fit, step);
    cv::putText(img, buf, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1);

    // Ground line
    cv::line(img, cv::Point(10, cy + 25), cv::Point(left_w - 10, cy + 25),
             cv::Scalar(100, 100, 100), 2);

    // Track limits
    float scale = 80.0f;  // pixels per meter
    int x_left  = cx + (int)(-2.4f * scale);
    int x_right = cx + (int)(2.4f * scale);
    cv::line(img, cv::Point(x_left, cy + 20), cv::Point(x_left, cy + 30),
             cv::Scalar(0, 0, 200), 2);
    cv::line(img, cv::Point(x_right, cy + 20), cv::Point(x_right, cy + 30),
             cv::Scalar(0, 0, 200), 2);

    // Cart
    int cart_x = cx + (int)(x * scale);
    int cart_w = 60, cart_h = 30;
    cv::rectangle(img, cv::Point(cart_x - cart_w/2, cy - cart_h/2),
                  cv::Point(cart_x + cart_w/2, cy + cart_h/2),
                  cv::Scalar(200, 100, 50), -1);
    cv::rectangle(img, cv::Point(cart_x - cart_w/2, cy - cart_h/2),
                  cv::Point(cart_x + cart_w/2, cy + cart_h/2),
                  cv::Scalar(0, 0, 0), 2);

    // Wheels
    cv::circle(img, cv::Point(cart_x - 18, cy + cart_h/2 + 5), 6, cv::Scalar(80, 80, 80), -1);
    cv::circle(img, cv::Point(cart_x + 18, cy + cart_h/2 + 5), 6, cv::Scalar(80, 80, 80), -1);

    // Pole
    float pole_len = 120.0f;
    int pole_ex = cart_x + (int)(pole_len * sinf(theta));
    int pole_ey = cy - (int)(pole_len * cosf(theta));
    cv::line(img, cv::Point(cart_x, cy - cart_h/2),
             cv::Point(pole_ex, pole_ey), cv::Scalar(0, 0, 200), 4);
    cv::circle(img, cv::Point(pole_ex, pole_ey), 6, cv::Scalar(0, 0, 255), -1);

    // Pivot
    cv::circle(img, cv::Point(cart_x, cy - cart_h/2), 4, cv::Scalar(0, 0, 0), -1);
}

// -------------------------------------------------------------------------
// Draw fitness graph on right half of image
// -------------------------------------------------------------------------
void draw_fitness_graph(cv::Mat& img, const vector<float>& best_hist,
                        const vector<float>& avg_hist, int gen)
{
    int left_w = IMG_W / 2;
    int graph_x = left_w + 30;
    int graph_y = 50;
    int graph_w = IMG_W - left_w - 60;
    int graph_h = IMG_H - 120;

    // Clear right half
    img(cv::Rect(left_w, 0, IMG_W - left_w, IMG_H)) = cv::Scalar(245, 245, 245);

    // Title
    cv::putText(img, "Fitness over Generations", cv::Point(left_w + 50, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    // Axes
    cv::line(img, cv::Point(graph_x, graph_y), cv::Point(graph_x, graph_y + graph_h),
             cv::Scalar(0, 0, 0), 1);
    cv::line(img, cv::Point(graph_x, graph_y + graph_h),
             cv::Point(graph_x + graph_w, graph_y + graph_h), cv::Scalar(0, 0, 0), 1);

    // Y-axis labels
    for (int v = 0; v <= 200; v += 50) {
        int y = graph_y + graph_h - (int)(v / 200.0f * graph_h);
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", v);
        cv::putText(img, buf, cv::Point(left_w + 5, y + 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 0, 0), 1);
        cv::line(img, cv::Point(graph_x - 3, y), cv::Point(graph_x + graph_w, y),
                 cv::Scalar(220, 220, 220), 1);
    }

    // X-axis label
    cv::putText(img, "Generation", cv::Point(graph_x + graph_w / 2 - 30, graph_y + graph_h + 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

    if (best_hist.empty()) return;

    int n = (int)best_hist.size();
    float max_gen = fmaxf((float)gen, 1.0f);

    // Draw curves
    for (int i = 1; i < n; i++) {
        int x1 = graph_x + (int)((i - 1) / max_gen * graph_w);
        int x2 = graph_x + (int)(i / max_gen * graph_w);

        // Best fitness (red)
        int y1b = graph_y + graph_h - (int)(best_hist[i-1] / 200.0f * graph_h);
        int y2b = graph_y + graph_h - (int)(best_hist[i] / 200.0f * graph_h);
        cv::line(img, cv::Point(x1, y1b), cv::Point(x2, y2b), cv::Scalar(0, 0, 200), 2);

        // Average fitness (blue)
        int y1a = graph_y + graph_h - (int)(avg_hist[i-1] / 200.0f * graph_h);
        int y2a = graph_y + graph_h - (int)(avg_hist[i] / 200.0f * graph_h);
        cv::line(img, cv::Point(x1, y1a), cv::Point(x2, y2a), cv::Scalar(200, 100, 0), 1);
    }

    // Legend
    cv::line(img, cv::Point(graph_x + 10, graph_y + 15), cv::Point(graph_x + 30, graph_y + 15),
             cv::Scalar(0, 0, 200), 2);
    cv::putText(img, "Best", cv::Point(graph_x + 35, graph_y + 19),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 200), 1);
    cv::line(img, cv::Point(graph_x + 10, graph_y + 30), cv::Point(graph_x + 30, graph_y + 30),
             cv::Scalar(200, 100, 0), 1);
    cv::putText(img, "Average", cv::Point(graph_x + 35, graph_y + 34),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 100, 0), 1);

    // Current stats
    char buf[128];
    snprintf(buf, sizeof(buf), "Gen %d: Best=%.0f Avg=%.1f",
             gen, best_hist.back(), avg_hist.back());
    cv::putText(img, buf, cv::Point(left_w + 20, IMG_H - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
}

// -------------------------------------------------------------------------
// Replay best individual on CPU (for visualization)
// -------------------------------------------------------------------------
struct ReplayFrame {
    float x, theta;
    int step;
};

vector<ReplayFrame> replay_best(const float* h_weights, int n_weights)
{
    vector<ReplayFrame> frames;

    // CPU cart-pole simulation
    float cx = 0.0f, cx_dot = 0.0f, ctheta = 0.05f, ctheta_dot = 0.0f;
    bool done = false;
    int steps = 0;

    while (!done) {
        frames.push_back({cx, ctheta, steps});

        // NN forward on CPU
        float obs[4] = {cx, cx_dot, ctheta, ctheta_dot};
        float h1[NN_HIDDEN1], h2[NN_HIDDEN2], out[1];

        int offset = 0;
        for (int j = 0; j < NN_HIDDEN1; j++) {
            float sum = 0;
            for (int i = 0; i < INPUT_DIM; i++)
                sum += h_weights[offset + j * INPUT_DIM + i] * obs[i];
            sum += h_weights[INPUT_DIM * NN_HIDDEN1 + j];
            h1[j] = tanhf(sum);
        }
        offset = INPUT_DIM * NN_HIDDEN1 + NN_HIDDEN1;

        for (int j = 0; j < NN_HIDDEN2; j++) {
            float sum = 0;
            for (int i = 0; i < NN_HIDDEN1; i++)
                sum += h_weights[offset + j * NN_HIDDEN1 + i] * h1[i];
            sum += h_weights[offset + NN_HIDDEN1 * NN_HIDDEN2 + j];
            h2[j] = tanhf(sum);
        }
        offset += NN_HIDDEN1 * NN_HIDDEN2 + NN_HIDDEN2;

        float sum = 0;
        for (int i = 0; i < NN_HIDDEN2; i++)
            sum += h_weights[offset + i] * h2[i];
        sum += h_weights[offset + NN_HIDDEN2];
        out[0] = sum;

        float action = tanhf(out[0]);

        // Physics step
        float force = action > 0 ? 10.0f : -10.0f;
        float total_mass = 1.1f;
        float pole_ml = 0.1f * 0.5f;
        float costh = cosf(ctheta), sinth = sinf(ctheta);
        float temp = (force + pole_ml * ctheta_dot * ctheta_dot * sinth) / total_mass;
        float th_acc = (9.8f * sinth - costh * temp) /
                       (0.5f * (4.0f/3.0f - 0.1f * costh * costh / total_mass));
        float x_acc = temp - pole_ml * th_acc * costh / total_mass;

        cx += 0.02f * cx_dot;
        cx_dot += 0.02f * x_acc;
        ctheta += 0.02f * ctheta_dot;
        ctheta_dot += 0.02f * th_acc;
        steps++;

        if (fabsf(cx) > 2.4f || fabsf(ctheta) > 12.0f * M_PI / 180.0f || steps >= 200)
            done = true;
    }
    return frames;
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main()
{
    int n_weights = nn_total_weights(INPUT_DIM, OUTPUT_DIM);
    printf("Cart-Pole Neuroevolution: pop=%d, generations=%d, weights=%d\n",
           POP_SIZE, N_GENERATIONS, n_weights);

    // Allocate GPU memory
    float *d_pop1, *d_pop2, *d_fitness;
    curandState *d_rng;
    int *d_elite_indices;

    CUDA_CHECK(cudaMalloc(&d_pop1, POP_SIZE * n_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pop2, POP_SIZE * n_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fitness, POP_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rng, POP_SIZE * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_elite_indices, ELITE_COUNT * sizeof(int)));

    // Host buffers
    vector<float> h_fitness(POP_SIZE);
    vector<float> h_best_weights(n_weights);

    // Init cuRAND
    int block = 256;
    int grid = (POP_SIZE + block - 1) / block;
    init_curand_kernel<<<grid, block>>>(d_rng, POP_SIZE, 42ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Init population
    init_population_kernel<<<grid, block>>>(d_pop1, d_rng, POP_SIZE, n_weights);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Video writer
    const string avi_path = "gif/neuroevo.avi";
    const string gif_path = "gif/neuroevo.gif";
    cv::VideoWriter writer(avi_path, cv::VideoWriter::fourcc('X','V','I','D'),
                           15, cv::Size(IMG_W, IMG_H));
    if (!writer.isOpened()) {
        fprintf(stderr, "Failed to open video writer: %s\n", avi_path.c_str());
        return 1;
    }

    cv::Mat frame(IMG_H, IMG_W, CV_8UC3, cv::Scalar(255, 255, 255));

    vector<float> best_history, avg_history;
    float *d_cur = d_pop1, *d_next = d_pop2;

    // -------------------------------------------------------------------------
    // Evolution loop
    // -------------------------------------------------------------------------
    for (int gen = 0; gen < N_GENERATIONS; gen++) {
        // Evaluate fitness
        evaluate_fitness_kernel<<<grid, block>>>(
            d_cur, d_fitness, d_rng, POP_SIZE, n_weights, INPUT_DIM, OUTPUT_DIM);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy fitness to host
        CUDA_CHECK(cudaMemcpy(h_fitness.data(), d_fitness,
                              POP_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        // Find best & compute avg
        float best_fit = -1.0f;
        int best_idx = 0;
        float sum_fit = 0.0f;
        for (int i = 0; i < POP_SIZE; i++) {
            sum_fit += h_fitness[i];
            if (h_fitness[i] > best_fit) {
                best_fit = h_fitness[i];
                best_idx = i;
            }
        }
        float avg_fit = sum_fit / POP_SIZE;
        best_history.push_back(best_fit);
        avg_history.push_back(avg_fit);

        if (gen % 10 == 0 || gen == N_GENERATIONS - 1) {
            printf("Gen %3d: best=%.0f, avg=%.1f\n", gen, best_fit, avg_fit);
        }

        // Copy best individual weights to host
        CUDA_CHECK(cudaMemcpy(h_best_weights.data(),
                              d_cur + best_idx * n_weights,
                              n_weights * sizeof(float), cudaMemcpyDeviceToHost));

        // Find elite indices (top ELITE_COUNT)
        vector<int> indices(POP_SIZE);
        iota(indices.begin(), indices.end(), 0);
        partial_sort(indices.begin(), indices.begin() + ELITE_COUNT, indices.end(),
                     [&](int a, int b) { return h_fitness[a] > h_fitness[b]; });
        CUDA_CHECK(cudaMemcpy(d_elite_indices, indices.data(),
                              ELITE_COUNT * sizeof(int), cudaMemcpyHostToDevice));

        // Visualization: every 5 generations or last
        if (gen % 5 == 0 || gen == N_GENERATIONS - 1) {
            // Replay best individual
            vector<ReplayFrame> replay = replay_best(h_best_weights.data(), n_weights);

            // Write frames for this generation's replay (subsample to keep video reasonable)
            int n_replay = (int)replay.size();
            int step_inc = max(1, n_replay / 20);  // ~20 frames per generation
            for (int f = 0; f < n_replay; f += step_inc) {
                draw_cartpole(frame, replay[f].x, replay[f].theta,
                              gen, best_fit, replay[f].step);
                draw_fitness_graph(frame, best_history, avg_history, gen);
                // Separator line
                cv::line(frame, cv::Point(IMG_W/2, 0), cv::Point(IMG_W/2, IMG_H),
                         cv::Scalar(150, 150, 150), 2);
                writer.write(frame);
            }
        }

        // Early stop
        if (best_fit >= 200.0f && gen >= 20) {
            printf("Solved at generation %d!\n", gen);
            // Write a few more frames to show the solved state
            vector<ReplayFrame> replay = replay_best(h_best_weights.data(), n_weights);
            for (int f = 0; f < (int)replay.size(); f += 2) {
                draw_cartpole(frame, replay[f].x, replay[f].theta,
                              gen, best_fit, replay[f].step);
                draw_fitness_graph(frame, best_history, avg_history, gen);
                cv::line(frame, cv::Point(IMG_W/2, 0), cv::Point(IMG_W/2, IMG_H),
                         cv::Scalar(150, 150, 150), 2);
                writer.write(frame);
            }
            // Hold final frame
            for (int i = 0; i < 30; i++) writer.write(frame);
            break;
        }

        // Reproduce
        reproduce_kernel<<<grid, block>>>(
            d_cur, d_fitness, d_next, d_elite_indices, d_rng,
            POP_SIZE, n_weights, TOURNAMENT_K,
            CROSSOVER_RATE, MUTATION_SIGMA, ELITE_COUNT);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap buffers
        float* tmp = d_cur;
        d_cur = d_next;
        d_next = tmp;
    }

    writer.release();

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
