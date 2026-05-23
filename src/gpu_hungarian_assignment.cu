// gpu_hungarian_assignment.cu
//
// GPU batched linear assignment for multi-target tracking.
//
// Each batch item is a dense 64x64 track-to-detection assignment problem.  A
// CUDA block solves one problem using a parallel auction algorithm in shared
// memory; a CPU Hungarian solver computes exact optima for validation.  This
// is the layout used by practical trackers when many cameras, gates, sensors,
// particles, or hypotheses need independent association at the same time.
//
// Output: gif/gpu_hungarian_assignment.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int N_ASSIGN = 64;
constexpr int N_BATCH = 512;
constexpr int N_FRAMES = 72;
constexpr int N_BENCH = 10;
constexpr int MAX_ROUNDS = 2048;
constexpr float AUCTION_EPS = 1.0e-5f;
constexpr float BENEFIT_BASE = 4.0f;
constexpr float BID_SCALE = 1000000.0f;
constexpr int PANEL_W = 900;
constexpr int PANEL_H = 560;

struct Point2 {
    float x, y;
};

// -------------------------------------------------------------------------
// Device-side batched auction solver
// -------------------------------------------------------------------------

__device__ inline float assignment_cost(const Point2& a, const Point2& b,
                                        int i, int j, int batch) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dist2 = dx * dx + dy * dy;
    float appearance = 0.0025f * fabsf(sinf(0.071f * i + 0.013f * batch)
                                     - cosf(0.067f * j - 0.011f * batch));
    return dist2 + appearance;
}

__device__ inline unsigned long long encode_bid(float bid, int bidder) {
    float clamped = fminf(fmaxf(bid, 0.0f), 4000.0f);
    unsigned int fixed = (unsigned int)(clamped * BID_SCALE + 0.5f) + 1U;
    return (((unsigned long long)fixed) << 32) | (unsigned int)(bidder + 1);
}

__device__ inline float decode_bid(unsigned long long key) {
    unsigned int fixed = (unsigned int)(key >> 32);
    if (fixed == 0U) return 0.0f;
    return ((float)(fixed - 1U)) / BID_SCALE;
}

__global__ void batched_auction_kernel(const Point2* tracks,
                                       const Point2* detections,
                                       int* assignments,
                                       float* total_costs,
                                       int* rounds_out,
                                       int* unresolved_out) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float price[N_ASSIGN];
    __shared__ int owner[N_ASSIGN];
    __shared__ int assign[N_ASSIGN];
    __shared__ unsigned long long bid_key[N_ASSIGN];
    __shared__ int done;
    __shared__ int unresolved;

    if (tid < N_ASSIGN) {
        price[tid] = 0.0f;
        owner[tid] = -1;
        assign[tid] = -1;
    }
    __syncthreads();

    const Point2* batch_tracks = tracks + batch * N_ASSIGN;
    const Point2* batch_dets = detections + batch * N_ASSIGN;

    int last_round = MAX_ROUNDS;
    for (int round = 0; round < MAX_ROUNDS; round++) {
        if (tid < N_ASSIGN) bid_key[tid] = 0ULL;
        __syncthreads();

        if (tid < N_ASSIGN && assign[tid] < 0) {
            Point2 tr = batch_tracks[tid];
            float best = -1.0e30f;
            float second = -1.0e30f;
            int best_obj = 0;
            for (int j = 0; j < N_ASSIGN; j++) {
                float c = assignment_cost(tr, batch_dets[j], tid, j, batch);
                float utility = BENEFIT_BASE - c - price[j];
                if (utility > best) {
                    second = best;
                    best = utility;
                    best_obj = j;
                } else if (utility > second) {
                    second = utility;
                }
            }
            float increment = best - second + AUCTION_EPS;
            float bid = price[best_obj] + increment;
            atomicMax(&bid_key[best_obj], encode_bid(bid, tid));
        }
        __syncthreads();

        if (tid < N_ASSIGN) {
            unsigned long long key = bid_key[tid];
            if (key != 0ULL) {
                int bidder = (int)(key & 0xffffffffU) - 1;
                int old_owner = owner[tid];
                if (old_owner >= 0) assign[old_owner] = -1;
                owner[tid] = bidder;
                assign[bidder] = tid;
                price[tid] = decode_bid(key);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int count = 0;
            for (int i = 0; i < N_ASSIGN; i++) {
                if (assign[i] < 0) count++;
            }
            unresolved = count;
            done = (count == 0);
            if (done) last_round = round + 1;
        }
        __syncthreads();
        if (done) break;
    }

    if (tid < N_ASSIGN) {
        assignments[batch * N_ASSIGN + tid] = assign[tid];
    }
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < N_ASSIGN; i++) {
            int j = assign[i];
            if (j >= 0) sum += assignment_cost(batch_tracks[i], batch_dets[j], i, j, batch);
        }
        total_costs[batch] = sum;
        rounds_out[batch] = last_round;
        unresolved_out[batch] = unresolved;
    }
}

// -------------------------------------------------------------------------
// Host reference solver and scene generation
// -------------------------------------------------------------------------

static inline float host_cost(const Point2& a, const Point2& b, int i, int j, int batch) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dist2 = dx * dx + dy * dy;
    float appearance = 0.0025f * std::fabs(std::sin(0.071f * i + 0.013f * batch)
                                         - std::cos(0.067f * j - 0.011f * batch));
    return dist2 + appearance;
}

static void make_batch_scene(int frame, std::vector<Point2>& tracks,
                             std::vector<Point2>& detections) {
    tracks.resize(N_BATCH * N_ASSIGN);
    detections.resize(N_BATCH * N_ASSIGN);
    int side = 8;
    float time = 0.050f * frame;

    for (int b = 0; b < N_BATCH; b++) {
        float phase = time + 0.019f * b;
        std::vector<Point2> local_tracks(N_ASSIGN);
        for (int i = 0; i < N_ASSIGN; i++) {
            int gx = i % side;
            int gy = i / side;
            float u = (gx + 0.5f) / (float)side;
            float v = (gy + 0.5f) / (float)side;
            float swirl = 0.040f * std::sin(5.0f * u + 4.0f * v + phase);
            float x = 0.08f + 0.84f * u + swirl * std::cos(phase + 0.11f * i);
            float y = 0.08f + 0.84f * v + swirl * std::sin(0.8f * phase + 0.13f * i);
            local_tracks[i] = {std::min(0.98f, std::max(0.02f, x)),
                               std::min(0.98f, std::max(0.02f, y))};
            tracks[b * N_ASSIGN + i] = local_tracks[i];
        }

        for (int i = 0; i < N_ASSIGN; i++) {
            int perm = (17 * i + 7 * b + 3 * frame) % N_ASSIGN;
            float flow_x = 0.030f * std::sin(phase + 0.37f * i);
            float flow_y = 0.030f * std::cos(0.7f * phase + 0.29f * i);
            detections[b * N_ASSIGN + perm] = {
                std::min(0.98f, std::max(0.02f, local_tracks[i].x + flow_x)),
                std::min(0.98f, std::max(0.02f, local_tracks[i].y + flow_y))
            };
        }
    }
}

static std::vector<float> make_cost_matrix(const Point2* tracks, const Point2* detections,
                                           int batch) {
    std::vector<float> cost(N_ASSIGN * N_ASSIGN);
    for (int i = 0; i < N_ASSIGN; i++) {
        for (int j = 0; j < N_ASSIGN; j++) {
            cost[i * N_ASSIGN + j] = host_cost(tracks[i], detections[j], i, j, batch);
        }
    }
    return cost;
}

static std::vector<int> hungarian_cpu(const std::vector<float>& cost) {
    constexpr double INF = 1.0e100;
    int n = N_ASSIGN;
    std::vector<double> u(n + 1), v(n + 1), minv(n + 1);
    std::vector<int> p(n + 1), way(n + 1);
    std::vector<char> used(n + 1);

    for (int i = 1; i <= n; i++) {
        p[0] = i;
        int j0 = 0;
        std::fill(minv.begin(), minv.end(), INF);
        std::fill(used.begin(), used.end(), 0);
        do {
            used[j0] = 1;
            int i0 = p[j0];
            double delta = INF;
            int j1 = 0;
            for (int j = 1; j <= n; j++) {
                if (used[j]) continue;
                double cur = (double)cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> assignment(n, -1);
    for (int j = 1; j <= n; j++) assignment[p[j] - 1] = j - 1;
    return assignment;
}

static double assignment_cost_host(const std::vector<float>& cost,
                                   const std::vector<int>& assignment) {
    double total = 0.0;
    for (int i = 0; i < N_ASSIGN; i++) {
        int j = assignment[i];
        if (j >= 0) total += cost[i * N_ASSIGN + j];
    }
    return total;
}

struct GpuRun {
    std::vector<int> assignments;
    std::vector<float> costs;
    std::vector<int> rounds;
    std::vector<int> unresolved;
    float ms;
};

static GpuRun solve_gpu(const std::vector<Point2>& tracks,
                        const std::vector<Point2>& detections,
                        Point2* d_tracks, Point2* d_detections,
                        int* d_assignments, float* d_costs,
                        int* d_rounds, int* d_unresolved) {
    size_t points_bytes = tracks.size() * sizeof(Point2);
    size_t assign_bytes = N_BATCH * N_ASSIGN * sizeof(int);

    CUDA_CHECK(cudaMemcpy(d_tracks, tracks.data(), points_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_detections, detections.data(), points_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    batched_auction_kernel<<<N_BATCH, N_ASSIGN>>>(d_tracks, d_detections,
                                                  d_assignments, d_costs,
                                                  d_rounds, d_unresolved);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaGetLastError());

    GpuRun run;
    run.assignments.resize(N_BATCH * N_ASSIGN);
    run.costs.resize(N_BATCH);
    run.rounds.resize(N_BATCH);
    run.unresolved.resize(N_BATCH);
    CUDA_CHECK(cudaMemcpy(run.assignments.data(), d_assignments, assign_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(run.costs.data(), d_costs, N_BATCH * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(run.rounds.data(), d_rounds, N_BATCH * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(run.unresolved.data(), d_unresolved, N_BATCH * sizeof(int), cudaMemcpyDeviceToHost));
    run.ms = ms;
    return run;
}

static cv::Point2i project(const Point2& p) {
    int x = (int)(45 + p.x * (PANEL_W - 90));
    int y = (int)(70 + p.y * (PANEL_H - 135));
    return cv::Point2i(x, y);
}

static void draw_scene(cv::Mat& img, const std::vector<Point2>& tracks,
                       const std::vector<Point2>& detections,
                       const std::vector<int>& assignments,
                       int frame, float gpu_ms, float avg_rounds,
                       double cpu_ms, double speedup, double gap_pct) {
    img.setTo(cv::Scalar(18, 18, 18));
    cv::rectangle(img, cv::Point(38, 62), cv::Point(PANEL_W - 38, PANEL_H - 56),
                  cv::Scalar(55, 55, 55), 1);

    const Point2* tr = tracks.data();
    const Point2* det = detections.data();
    for (int i = 0; i < N_ASSIGN; i++) {
        int j = assignments[i];
        if (j < 0) continue;
        cv::Point2i a = project(tr[i]);
        cv::Point2i b = project(det[j]);
        float len = std::hypot((float)(a.x - b.x), (float)(a.y - b.y));
        cv::Scalar c = len > 55.0f ? cv::Scalar(70, 110, 230) : cv::Scalar(70, 180, 110);
        cv::line(img, a, b, c, 1, cv::LINE_AA);
    }
    for (int i = 0; i < N_ASSIGN; i++) {
        cv::circle(img, project(det[i]), 4, cv::Scalar(60, 165, 245), -1, cv::LINE_AA);
    }
    for (int i = 0; i < N_ASSIGN; i++) {
        cv::circle(img, project(tr[i]), 3, cv::Scalar(235, 235, 235), -1, cv::LINE_AA);
    }

    cv::putText(img, cv::format("GPU batched Hungarian-class assignment  frame %d / %d",
                                frame, N_FRAMES),
                cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.60,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(img, "white=tracks  orange=detections  green=accepted assignment",
                cv::Point(18, 51), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(195, 195, 195), 1, cv::LINE_AA);
    cv::putText(img, cv::format("%d independent %dx%d dense assignments in one launch",
                                N_BATCH, N_ASSIGN, N_ASSIGN),
                cv::Point(18, PANEL_H - 52), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    cv::putText(img, cv::format("GPU auction %.3f ms/batch   avg rounds %.1f",
                                gpu_ms, avg_rounds),
                cv::Point(18, PANEL_H - 30), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(200, 235, 205), 1, cv::LINE_AA);
    cv::putText(img, cv::format("CPU Hungarian %.3f ms/batch   %.1fx   gap %.4f%%",
                                cpu_ms, speedup, gap_pct),
                cv::Point(18, PANEL_H - 10), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(205, 220, 255), 1, cv::LINE_AA);
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<Point2> tracks, detections;

    Point2 *d_tracks = nullptr, *d_detections = nullptr;
    int *d_assignments = nullptr, *d_rounds = nullptr, *d_unresolved = nullptr;
    float* d_costs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tracks, N_BATCH * N_ASSIGN * sizeof(Point2)));
    CUDA_CHECK(cudaMalloc(&d_detections, N_BATCH * N_ASSIGN * sizeof(Point2)));
    CUDA_CHECK(cudaMalloc(&d_assignments, N_BATCH * N_ASSIGN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_costs, N_BATCH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rounds, N_BATCH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_unresolved, N_BATCH * sizeof(int)));

    make_batch_scene(0, tracks, detections);
    (void)solve_gpu(tracks, detections,
                    d_tracks, d_detections, d_assignments,
                    d_costs, d_rounds, d_unresolved);
    CUDA_CHECK(cudaDeviceSynchronize());

    int mkdir_ret = std::system("mkdir -p gif");
    (void)mkdir_ret;
    cv::VideoWriter video("gif/gpu_hungarian_assignment.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          12, cv::Size(PANEL_W, PANEL_H));

    double total_gpu_ms = 0.0;
    double total_cpu_ms = 0.0;
    double total_gap = 0.0;
    double max_gap = 0.0;
    double total_speedup = 0.0;
    double total_rounds = 0.0;

    double last_cpu_ms = 0.0;
    double last_speedup = 0.0;
    double last_gap = 0.0;

    for (int frame = 0; frame < N_FRAMES; frame++) {
        make_batch_scene(frame, tracks, detections);
        GpuRun gpu = solve_gpu(tracks, detections,
                               d_tracks, d_detections, d_assignments,
                               d_costs, d_rounds, d_unresolved);

        double avg_rounds = 0.0;
        int unresolved_total = 0;
        for (int b = 0; b < N_BATCH; b++) {
            avg_rounds += gpu.rounds[b];
            unresolved_total += gpu.unresolved[b];
        }
        avg_rounds /= N_BATCH;
        total_rounds += avg_rounds;
        total_gpu_ms += gpu.ms;

        if (frame < N_BENCH) {
            auto t0 = std::chrono::high_resolution_clock::now();
            double exact_total = 0.0;
            double gpu_total = 0.0;
            for (int b = 0; b < N_BATCH; b++) {
                const Point2* tr = tracks.data() + b * N_ASSIGN;
                const Point2* det = detections.data() + b * N_ASSIGN;
                auto cost = make_cost_matrix(tr, det, b);
                auto cpu_assignment = hungarian_cpu(cost);
                exact_total += assignment_cost_host(cost, cpu_assignment);
                gpu_total += gpu.costs[b];
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            last_cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            last_speedup = last_cpu_ms / std::max(1.0e-6f, gpu.ms);
            last_gap = 100.0 * (gpu_total - exact_total) / std::max(1.0e-12, exact_total);
            if (last_gap < 0.0 && last_gap > -1.0e-4) last_gap = 0.0;
            total_cpu_ms += last_cpu_ms;
            total_speedup += last_speedup;
            total_gap += last_gap;
            max_gap = std::max(max_gap, last_gap);
            std::printf("bench %2d  gpu %.3f ms  cpu %.3f ms  speedup %.1fx  avg rounds %.1f  gap %.4f%%  unresolved %d\n",
                        frame, gpu.ms, last_cpu_ms, last_speedup, avg_rounds,
                        last_gap, unresolved_total);
        }

        cv::Mat img(PANEL_H, PANEL_W, CV_8UC3);
        draw_scene(img, tracks, detections, gpu.assignments,
                   frame, gpu.ms, (float)avg_rounds,
                   last_cpu_ms, last_speedup, last_gap);
        video.write(img);
    }
    video.release();

    std::printf("Avg GPU batched auction = %.3f ms for %d x %dx%d assignments\n",
                total_gpu_ms / N_FRAMES, N_BATCH, N_ASSIGN, N_ASSIGN);
    std::printf("CPU Hungarian benchmark = %.3f ms, speedup %.1fx, avg gap %.4f%%, max gap %.4f%%, avg rounds %.1f\n",
                total_cpu_ms / N_BENCH, total_speedup / N_BENCH,
                total_gap / N_BENCH, max_gap, total_rounds / N_FRAMES);

    cudabot::avi_to_gif("gif/gpu_hungarian_assignment.avi",
                        "gif/gpu_hungarian_assignment.gif", 8, 560);
    std::printf("GIF saved to gif/gpu_hungarian_assignment.gif\n");

    cudaFree(d_tracks);
    cudaFree(d_detections);
    cudaFree(d_assignments);
    cudaFree(d_costs);
    cudaFree(d_rounds);
    cudaFree(d_unresolved);
    return 0;
}
