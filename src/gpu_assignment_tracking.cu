// gpu_assignment_tracking.cu
//
// GPU multi-frame assignment tracking demo.
//
// This extends the dense assignment demo into a robotics-shaped tracker:
// each CUDA block owns one scene, predicts 48 constant-velocity tracks,
// gates 72 noisy detections with clutter and missed detections, solves a
// rectangular track-to-detection association with an auction-style update,
// then updates the track states and reports identity accuracy.
//
// Output: gif/gpu_assignment_tracking.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int N_SCENES = 128;
constexpr int N_TRACKS = 48;
constexpr int N_DETS = 72;
constexpr int N_FRAMES = 96;
constexpr int N_BENCH = 30;
constexpr int MAX_ROUNDS = 96;
constexpr float WORLD_W = 18.0f;
constexpr float WORLD_H = 11.0f;
constexpr float DT = 0.16f;
constexpr float GATE_R = 0.82f;
constexpr float GATE_COST = 3.0f;
constexpr float BENEFIT_BASE = 6.0f;
constexpr float AUCTION_EPS = 1.0e-4f;
constexpr float BID_SCALE = 1000000.0f;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int VIDEO_FPS = 12;

struct TrackState {
    float x;
    float y;
    float vx;
    float vy;
    float app;
    int id;
    int age;
    int miss;
    int active;
};

struct Detection {
    float x;
    float y;
    float app;
    float conf;
    int truth;
};

struct TruthState {
    float x;
    float y;
    float vx;
    float vy;
    float app;
};

struct FrameMetrics {
    int matched;
    int correct;
    int clutter;
    int missed;
    int rounds;
    float mean_cost;
};

struct BenchResult {
    float gpu_ms = 0.0f;
    double cpu_ms = 0.0;
    double speedup = 0.0;
};

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline float sqr(float x) {
    return x * x;
}

__host__ __device__ static inline float dist2(float ax, float ay, float bx, float by) {
    return sqr(ax - bx) + sqr(ay - by);
}

__host__ __device__ static inline float wrap_app(float x) {
    while (x > 1.0f) x -= 1.0f;
    while (x < 0.0f) x += 1.0f;
    return x;
}

__host__ __device__ static inline float app_delta(float a, float b) {
    float d = fabsf(a - b);
    return fminf(d, 1.0f - d);
}

__host__ __device__ static inline unsigned long long encode_bid(float bid, int bidder) {
    float clamped = fminf(fmaxf(bid, 0.0f), 2048.0f);
    unsigned int fixed = (unsigned int)(clamped * BID_SCALE + 0.5f) + 1U;
    return (((unsigned long long)fixed) << 32) | (unsigned int)(bidder + 1);
}

__host__ __device__ static inline float decode_bid(unsigned long long key) {
    unsigned int fixed = (unsigned int)(key >> 32);
    if (fixed == 0U) return 0.0f;
    return ((float)(fixed - 1U)) / BID_SCALE;
}

__host__ __device__ static inline float association_cost(const TrackState& tr,
                                                         const Detection& det) {
    if (det.conf <= 0.05f) return 1.0e20f;
    float px = tr.x + tr.vx * DT;
    float py = tr.y + tr.vy * DT;
    float d2 = dist2(px, py, det.x, det.y);
    if (d2 > GATE_R * GATE_R) return 1.0e20f;
    float app = app_delta(tr.app, det.app);
    return 3.4f * d2 + 18.0f * app * app - 0.25f * det.conf;
}

__global__ void assignment_tracking_kernel(TrackState* tracks,
                                           const Detection* detections,
                                           int* assignments,
                                           FrameMetrics* metrics) {
    int scene = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float price[N_DETS];
    __shared__ int owner[N_DETS];
    __shared__ int assign[N_TRACKS];
    __shared__ unsigned long long bid_key[N_DETS];
    __shared__ int any_bid;
    __shared__ int unresolved;
    __shared__ int last_round;

    TrackState* scene_tracks = tracks + scene * N_TRACKS;
    const Detection* scene_dets = detections + scene * N_DETS;
    int* scene_assign = assignments + scene * N_TRACKS;

    if (tid < N_DETS) {
        price[tid] = 0.0f;
        owner[tid] = -1;
    }
    if (tid < N_TRACKS) assign[tid] = -1;
    if (tid == 0) last_round = MAX_ROUNDS;
    __syncthreads();

    for (int round = 0; round < MAX_ROUNDS; round++) {
        if (tid < N_DETS) bid_key[tid] = 0ULL;
        if (tid == 0) any_bid = 0;
        __syncthreads();

        if (tid < N_TRACKS) {
            TrackState tr = scene_tracks[tid];
            if (tr.active && assign[tid] < 0) {
                float best = -1.0e30f;
                float second = -1.0e30f;
                int best_det = -1;
                for (int j = 0; j < N_DETS; j++) {
                    float c = association_cost(tr, scene_dets[j]);
                    if (c >= 1.0e19f || c > GATE_COST) continue;
                    float utility = BENEFIT_BASE - c - price[j];
                    if (utility > best) {
                        second = best;
                        best = utility;
                        best_det = j;
                    } else if (utility > second) {
                        second = utility;
                    }
                }
                if (best_det >= 0 && best > 0.0f) {
                    if (second < 0.0f) second = 0.0f;
                    float bid = price[best_det] + best - second + AUCTION_EPS;
                    atomicMax(&bid_key[best_det], encode_bid(bid, tid));
                    atomicExch(&any_bid, 1);
                }
            }
        }
        __syncthreads();

        if (tid < N_DETS) {
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
            for (int i = 0; i < N_TRACKS; i++) {
                TrackState tr = scene_tracks[i];
                if (!tr.active || assign[i] >= 0) continue;
                float best = -1.0e30f;
                for (int j = 0; j < N_DETS; j++) {
                    float c = association_cost(tr, scene_dets[j]);
                    if (c >= 1.0e19f || c > GATE_COST) continue;
                    best = fmaxf(best, BENEFIT_BASE - c - price[j]);
                }
                if (best > 0.0f) count++;
            }
            unresolved = count;
            if (!any_bid || count == 0) last_round = round + 1;
        }
        __syncthreads();
        if (!any_bid || unresolved == 0) break;
    }

    if (tid < N_TRACKS) {
        TrackState tr = scene_tracks[tid];
        int j = assign[tid];
        float old_x = tr.x;
        float old_y = tr.y;
        float px = clampf(tr.x + tr.vx * DT, 0.25f, WORLD_W - 0.25f);
        float py = clampf(tr.y + tr.vy * DT, 0.25f, WORLD_H - 0.25f);

        if (tr.active && j >= 0) {
            Detection det = scene_dets[j];
            float k = 0.72f;
            tr.x = px + k * (det.x - px);
            tr.y = py + k * (det.y - py);
            tr.vx = 0.66f * tr.vx + 0.34f * ((tr.x - old_x) / DT);
            tr.vy = 0.66f * tr.vy + 0.34f * ((tr.y - old_y) / DT);
            tr.app = wrap_app(0.93f * tr.app + 0.07f * det.app);
            tr.age++;
            tr.miss = 0;
        } else if (tr.active) {
            tr.x = px;
            tr.y = py;
            tr.vx *= 0.985f;
            tr.vy *= 0.985f;
            tr.age++;
            tr.miss++;
        }
        scene_tracks[tid] = tr;
        scene_assign[tid] = j;
    }
    __syncthreads();

    if (tid == 0) {
        FrameMetrics m{};
        float cost_sum = 0.0f;
        for (int i = 0; i < N_TRACKS; i++) {
            int j = assign[i];
            if (j >= 0) {
                m.matched++;
                Detection det = scene_dets[j];
                if (det.truth == i) m.correct++;
                if (det.truth < 0) m.clutter++;
                cost_sum += association_cost(scene_tracks[i], det);
            } else {
                m.missed++;
            }
        }
        m.rounds = last_round;
        m.mean_cost = (m.matched > 0) ? cost_sum / (float)m.matched : 0.0f;
        metrics[scene] = m;
    }
}

static float randn(std::mt19937& rng, float sigma) {
    std::normal_distribution<float> normal(0.0f, sigma);
    return normal(rng);
}

static float randu(std::mt19937& rng) {
    return std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
}

static cv::Scalar track_color(int id) {
    static const cv::Scalar palette[12] = {
        cv::Scalar(88, 161, 255), cv::Scalar(82, 220, 136),
        cv::Scalar(255, 184, 84), cv::Scalar(235, 105, 126),
        cv::Scalar(180, 136, 255), cv::Scalar(85, 216, 220),
        cv::Scalar(238, 228, 95), cv::Scalar(255, 128, 204),
        cv::Scalar(128, 205, 255), cv::Scalar(120, 238, 170),
        cv::Scalar(210, 156, 92), cv::Scalar(210, 132, 226),
    };
    return palette[id % 12];
}

static void init_truth_and_tracks(std::vector<TruthState>& truth,
                                  std::vector<TrackState>& tracks) {
    truth.resize(N_SCENES * N_TRACKS);
    tracks.resize(N_SCENES * N_TRACKS);
    std::mt19937 rng(24052026);

    for (int s = 0; s < N_SCENES; s++) {
        float phase = 0.071f * (float)s;
        for (int i = 0; i < N_TRACKS; i++) {
            int gx = i % 8;
            int gy = i / 8;
            float x = 1.15f + 2.05f * (float)gx + 0.23f * std::sin(phase + 0.31f * i);
            float y = 1.25f + 1.55f * (float)gy + 0.19f * std::cos(phase + 0.37f * i);
            float vx = 0.19f * std::sin(0.42f * i + phase)
                     + 0.08f * std::cos(0.11f * i);
            float vy = 0.17f * std::cos(0.39f * i - phase)
                     + 0.07f * std::sin(0.17f * i);
            float app = wrap_app(0.381966f * (float)i + 0.007f * (float)s);
            truth[s * N_TRACKS + i] = {x, y, vx, vy, app};
            tracks[s * N_TRACKS + i] = {
                x + randn(rng, 0.05f), y + randn(rng, 0.05f),
                vx + randn(rng, 0.018f), vy + randn(rng, 0.018f),
                app + randn(rng, 0.006f), i, 1, 0, 1
            };
        }
    }
}

static void step_truth(std::vector<TruthState>& truth, int frame) {
    for (int s = 0; s < N_SCENES; s++) {
        for (int i = 0; i < N_TRACKS; i++) {
            TruthState& q = truth[s * N_TRACKS + i];
            float steer_x = 0.011f * std::sin(0.053f * frame + 0.17f * i + 0.011f * s);
            float steer_y = 0.010f * std::cos(0.047f * frame + 0.19f * i - 0.013f * s);
            q.vx = clampf(q.vx + steer_x, -0.42f, 0.42f);
            q.vy = clampf(q.vy + steer_y, -0.38f, 0.38f);
            q.x += q.vx * DT;
            q.y += q.vy * DT;
            if (q.x < 0.7f || q.x > WORLD_W - 0.7f) {
                q.vx = -0.86f * q.vx;
                q.x = clampf(q.x, 0.7f, WORLD_W - 0.7f);
            }
            if (q.y < 0.7f || q.y > WORLD_H - 0.7f) {
                q.vy = -0.86f * q.vy;
                q.y = clampf(q.y, 0.7f, WORLD_H - 0.7f);
            }
            q.app = wrap_app(q.app + 0.0012f * std::sin(0.05f * frame + 0.21f * i));
        }
    }
}

static void make_detections(const std::vector<TruthState>& truth,
                            int frame,
                            std::vector<Detection>& detections) {
    detections.assign(N_SCENES * N_DETS, Detection{0.0f, 0.0f, 0.0f, 0.0f, -1});
    std::mt19937 rng(9001 + frame * 31);

    for (int s = 0; s < N_SCENES; s++) {
        std::array<int, N_DETS> slots;
        std::iota(slots.begin(), slots.end(), 0);
        std::shuffle(slots.begin(), slots.end(), rng);
        int slot_idx = 0;

        for (int i = 0; i < N_TRACKS; i++) {
            float miss_wave = 0.5f + 0.5f * std::sin(0.19f * frame + 0.37f * i + 0.023f * s);
            float p_detect = 0.94f - 0.10f * miss_wave;
            if (randu(rng) > p_detect) continue;

            TruthState q = truth[s * N_TRACKS + i];
            int slot = slots[slot_idx++];
            detections[s * N_DETS + slot] = {
                clampf(q.x + randn(rng, 0.072f), 0.1f, WORLD_W - 0.1f),
                clampf(q.y + randn(rng, 0.072f), 0.1f, WORLD_H - 0.1f),
                wrap_app(q.app + randn(rng, 0.016f)),
                clampf(0.82f + randn(rng, 0.075f), 0.35f, 1.0f),
                i
            };
        }

        int clutter_count = 8 + ((frame + 3 * s) % 7);
        for (int c = 0; c < clutter_count && slot_idx < N_DETS; c++) {
            int slot = slots[slot_idx++];
            detections[s * N_DETS + slot] = {
                0.25f + randu(rng) * (WORLD_W - 0.5f),
                0.25f + randu(rng) * (WORLD_H - 0.5f),
                randu(rng),
                clampf(0.34f + randn(rng, 0.11f), 0.08f, 0.65f),
                -1
            };
        }
    }
}

static void cpu_association_update(std::vector<TrackState>& tracks,
                                   const std::vector<Detection>& detections,
                                   std::vector<int>& assignments,
                                   std::vector<FrameMetrics>& metrics) {
    assignments.assign(N_SCENES * N_TRACKS, -1);
    metrics.assign(N_SCENES, FrameMetrics{});

    for (int s = 0; s < N_SCENES; s++) {
        TrackState* scene_tracks = tracks.data() + s * N_TRACKS;
        const Detection* scene_dets = detections.data() + s * N_DETS;
        std::array<float, N_DETS> price{};
        std::array<int, N_DETS> owner;
        std::array<int, N_TRACKS> assign;
        owner.fill(-1);
        assign.fill(-1);
        int last_round = MAX_ROUNDS;

        for (int round = 0; round < MAX_ROUNDS; round++) {
            std::array<float, N_DETS> best_bid{};
            std::array<int, N_DETS> bidder;
            bidder.fill(-1);
            bool any_bid = false;

            for (int i = 0; i < N_TRACKS; i++) {
                TrackState tr = scene_tracks[i];
                if (!tr.active || assign[i] >= 0) continue;
                float best = -1.0e30f;
                float second = -1.0e30f;
                int best_det = -1;
                for (int j = 0; j < N_DETS; j++) {
                    float c = association_cost(tr, scene_dets[j]);
                    if (c >= 1.0e19f || c > GATE_COST) continue;
                    float utility = BENEFIT_BASE - c - price[j];
                    if (utility > best) {
                        second = best;
                        best = utility;
                        best_det = j;
                    } else if (utility > second) {
                        second = utility;
                    }
                }
                if (best_det >= 0 && best > 0.0f) {
                    if (second < 0.0f) second = 0.0f;
                    float bid = price[best_det] + best - second + AUCTION_EPS;
                    if (bid > best_bid[best_det]) {
                        best_bid[best_det] = bid;
                        bidder[best_det] = i;
                    }
                    any_bid = true;
                }
            }

            for (int j = 0; j < N_DETS; j++) {
                if (bidder[j] >= 0) {
                    int old_owner = owner[j];
                    if (old_owner >= 0) assign[old_owner] = -1;
                    owner[j] = bidder[j];
                    assign[bidder[j]] = j;
                    price[j] = best_bid[j];
                }
            }

            int unresolved = 0;
            for (int i = 0; i < N_TRACKS; i++) {
                TrackState tr = scene_tracks[i];
                if (!tr.active || assign[i] >= 0) continue;
                float best = -1.0e30f;
                for (int j = 0; j < N_DETS; j++) {
                    float c = association_cost(tr, scene_dets[j]);
                    if (c >= 1.0e19f || c > GATE_COST) continue;
                    best = std::max(best, BENEFIT_BASE - c - price[j]);
                }
                if (best > 0.0f) unresolved++;
            }
            if (!any_bid || unresolved == 0) {
                last_round = round + 1;
                break;
            }
        }

        for (int i = 0; i < N_TRACKS; i++) {
            TrackState& tr = scene_tracks[i];
            int j = assign[i];
            float old_x = tr.x;
            float old_y = tr.y;
            float px = clampf(tr.x + tr.vx * DT, 0.25f, WORLD_W - 0.25f);
            float py = clampf(tr.y + tr.vy * DT, 0.25f, WORLD_H - 0.25f);
            if (tr.active && j >= 0) {
                Detection det = scene_dets[j];
                float k = 0.72f;
                tr.x = px + k * (det.x - px);
                tr.y = py + k * (det.y - py);
                tr.vx = 0.66f * tr.vx + 0.34f * ((tr.x - old_x) / DT);
                tr.vy = 0.66f * tr.vy + 0.34f * ((tr.y - old_y) / DT);
                tr.app = wrap_app(0.93f * tr.app + 0.07f * det.app);
                tr.age++;
                tr.miss = 0;
            } else if (tr.active) {
                tr.x = px;
                tr.y = py;
                tr.vx *= 0.985f;
                tr.vy *= 0.985f;
                tr.age++;
                tr.miss++;
            }
            assignments[s * N_TRACKS + i] = j;
        }

        FrameMetrics m{};
        float cost_sum = 0.0f;
        for (int i = 0; i < N_TRACKS; i++) {
            int j = assign[i];
            if (j >= 0) {
                m.matched++;
                Detection det = scene_dets[j];
                if (det.truth == i) m.correct++;
                if (det.truth < 0) m.clutter++;
                cost_sum += association_cost(scene_tracks[i], det);
            } else {
                m.missed++;
            }
        }
        m.rounds = last_round;
        m.mean_cost = (m.matched > 0) ? cost_sum / (float)m.matched : 0.0f;
        metrics[s] = m;
    }
}

static float gpu_update(TrackState* d_tracks,
                        Detection* d_dets,
                        int* d_assignments,
                        FrameMetrics* d_metrics,
                        const std::vector<Detection>& detections,
                        std::vector<TrackState>& tracks,
                        std::vector<int>& assignments,
                        std::vector<FrameMetrics>& metrics) {
    CUDA_CHECK(cudaMemcpy(d_dets, detections.data(),
                          N_SCENES * N_DETS * sizeof(Detection),
                          cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    assignment_tracking_kernel<<<N_SCENES, 128>>>(d_tracks, d_dets,
                                                 d_assignments, d_metrics);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    tracks.resize(N_SCENES * N_TRACKS);
    assignments.resize(N_SCENES * N_TRACKS);
    metrics.resize(N_SCENES);
    CUDA_CHECK(cudaMemcpy(tracks.data(), d_tracks,
                          tracks.size() * sizeof(TrackState),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(assignments.data(), d_assignments,
                          assignments.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(metrics.data(), d_metrics,
                          metrics.size() * sizeof(FrameMetrics),
                          cudaMemcpyDeviceToHost));
    return ms;
}

static BenchResult benchmark_once(std::vector<TrackState> tracks,
                                  const std::vector<Detection>& detections,
                                  TrackState* d_tracks,
                                  Detection* d_dets,
                                  int* d_assignments,
                                  FrameMetrics* d_metrics) {
    BenchResult br;
    CUDA_CHECK(cudaMemcpy(d_tracks, tracks.data(),
                          tracks.size() * sizeof(TrackState),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dets, detections.data(),
                          detections.size() * sizeof(Detection),
                          cudaMemcpyHostToDevice));
    assignment_tracking_kernel<<<N_SCENES, 128>>>(d_tracks, d_dets,
                                                 d_assignments, d_metrics);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < N_BENCH; i++) {
        CUDA_CHECK(cudaMemcpy(d_tracks, tracks.data(),
                              tracks.size() * sizeof(TrackState),
                              cudaMemcpyHostToDevice));
        assignment_tracking_kernel<<<N_SCENES, 128>>>(d_tracks, d_dets,
                                                     d_assignments, d_metrics);
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float total_gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_gpu_ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    br.gpu_ms = total_gpu_ms / N_BENCH;

    std::vector<int> cpu_assign;
    std::vector<FrameMetrics> cpu_metrics;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 3; i++) {
        std::vector<TrackState> cpu_tracks = tracks;
        cpu_association_update(cpu_tracks, detections, cpu_assign, cpu_metrics);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    br.cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 3.0;
    br.speedup = br.cpu_ms / std::max(1.0e-9, (double)br.gpu_ms);
    return br;
}

static cv::Point world_to_px(float x, float y, const cv::Rect& r) {
    float u = clampf(x / WORLD_W, 0.0f, 1.0f);
    float v = clampf(y / WORLD_H, 0.0f, 1.0f);
    return cv::Point(r.x + (int)(u * r.width),
                     r.y + r.height - (int)(v * r.height));
}

static float scene_accuracy(const FrameMetrics& m) {
    return (m.matched > 0) ? (float)m.correct / (float)m.matched : 0.0f;
}

static void draw_history(cv::Mat& img,
                         const std::vector<float>& accuracy_hist,
                         const std::vector<float>& match_hist,
                         const cv::Rect& r) {
    cv::rectangle(img, r, cv::Scalar(29, 31, 36), -1);
    cv::rectangle(img, r, cv::Scalar(76, 80, 88), 1);
    cv::putText(img, "scene-0 tracking", cv::Point(r.x + 12, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(235, 235, 235),
                1, cv::LINE_AA);
    for (int g = 0; g <= 4; g++) {
        float y01 = (float)g / 4.0f;
        int y = r.y + r.height - 20 - (int)(y01 * (r.height - 52));
        cv::line(img, cv::Point(r.x + 42, y), cv::Point(r.x + r.width - 12, y),
                 cv::Scalar(46, 49, 55), 1);
        cv::putText(img, cv::format("%.1f", y01), cv::Point(r.x + 9, y + 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.34, cv::Scalar(160, 166, 174), 1);
    }

    auto draw_curve = [&](const std::vector<float>& h, cv::Scalar color) {
        if (h.size() < 2) return;
        std::vector<cv::Point> pts;
        for (size_t i = 0; i < h.size(); i++) {
            float x01 = (float)i / std::max<size_t>(1, h.size() - 1);
            float y01 = clampf(h[i], 0.0f, 1.0f);
            int x = r.x + 42 + (int)(x01 * (r.width - 58));
            int y = r.y + r.height - 20 - (int)(y01 * (r.height - 52));
            pts.emplace_back(x, y);
        }
        cv::polylines(img, pts, false, color, 2, cv::LINE_AA);
    };
    draw_curve(match_hist, cv::Scalar(90, 160, 255));
    draw_curve(accuracy_hist, cv::Scalar(90, 230, 132));
    cv::putText(img, "accuracy", cv::Point(r.x + 126, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(90, 230, 132), 1);
    cv::putText(img, "matched", cv::Point(r.x + 220, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(90, 160, 255), 1);
}

static cv::Mat draw_frame(const std::vector<TruthState>& truth,
                          const std::vector<TrackState>& tracks,
                          const std::vector<Detection>& detections,
                          const std::vector<int>& assignments,
                          const FrameMetrics& metrics,
                          const std::vector<std::vector<cv::Point>>& trails,
                          const std::vector<float>& accuracy_hist,
                          const std::vector<float>& match_hist,
                          int frame,
                          float gpu_ms,
                          const BenchResult& bench) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 19, 22));
    cv::putText(img, cv::format("GPU assignment tracking  frame %02d / %d",
                                frame + 1, N_FRAMES),
                cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.72,
                cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img,
                cv::format("%d scenes x %d tracks x %d detections   GPU %.3f ms/frame   CPU %.3f ms   %.1fx",
                           N_SCENES, N_TRACKS, N_DETS, gpu_ms, bench.cpu_ms,
                           bench.speedup),
                cv::Point(18, 54), cv::FONT_HERSHEY_SIMPLEX, 0.47,
                cv::Scalar(210, 214, 220), 1, cv::LINE_AA);

    cv::Rect map_rect(34, 80, 660, 502);
    cv::rectangle(img, map_rect, cv::Scalar(25, 27, 31), -1);
    cv::rectangle(img, map_rect, cv::Scalar(80, 84, 92), 1);
    for (int i = 1; i < 9; i++) {
        int x = map_rect.x + i * map_rect.width / 9;
        cv::line(img, cv::Point(x, map_rect.y), cv::Point(x, map_rect.y + map_rect.height),
                 cv::Scalar(38, 41, 47), 1);
    }
    for (int i = 1; i < 6; i++) {
        int y = map_rect.y + i * map_rect.height / 6;
        cv::line(img, cv::Point(map_rect.x, y), cv::Point(map_rect.x + map_rect.width, y),
                 cv::Scalar(38, 41, 47), 1);
    }

    const TruthState* scene_truth = truth.data();
    const TrackState* scene_tracks = tracks.data();
    const Detection* scene_dets = detections.data();
    const int* scene_assign = assignments.data();

    for (int j = 0; j < N_DETS; j++) {
        Detection det = scene_dets[j];
        if (det.conf <= 0.05f) continue;
        cv::Point p = world_to_px(det.x, det.y, map_rect);
        if (det.truth >= 0) {
            cv::circle(img, p, 3, track_color(det.truth), -1, cv::LINE_AA);
        } else {
            cv::drawMarker(img, p, cv::Scalar(120, 124, 132), cv::MARKER_CROSS,
                           7, 1, cv::LINE_AA);
        }
    }

    for (int i = 0; i < N_TRACKS; i++) {
        cv::Scalar c = track_color(i);
        if (trails[i].size() > 1) cv::polylines(img, trails[i], false, c * 0.55, 1, cv::LINE_AA);
        cv::Point truth_pt = world_to_px(scene_truth[i].x, scene_truth[i].y, map_rect);
        cv::circle(img, truth_pt, 5, c * 0.35, 1, cv::LINE_AA);
        cv::Point tr_pt = world_to_px(scene_tracks[i].x, scene_tracks[i].y, map_rect);
        cv::circle(img, tr_pt, (i < 12) ? 7 : 5, c, 2, cv::LINE_AA);
        cv::Point vel_pt(tr_pt.x + (int)(16.0f * scene_tracks[i].vx),
                         tr_pt.y - (int)(16.0f * scene_tracks[i].vy));
        cv::line(img, tr_pt, vel_pt, c, 1, cv::LINE_AA);
        int j = scene_assign[i];
        if (j >= 0) {
            cv::Point det_pt = world_to_px(scene_dets[j].x, scene_dets[j].y, map_rect);
            cv::Scalar line_color = (scene_dets[j].truth == i)
                                  ? cv::Scalar(80, 210, 130)
                                  : cv::Scalar(90, 105, 255);
            cv::line(img, tr_pt, det_pt, line_color, 1, cv::LINE_AA);
        }
        if (i < 12) {
            cv::putText(img, cv::format("%02d", i), tr_pt + cv::Point(6, -6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.32, c, 1, cv::LINE_AA);
        }
    }

    cv::Rect stat_rect(720, 84, 218, 160);
    cv::rectangle(img, stat_rect, cv::Scalar(29, 31, 36), -1);
    cv::rectangle(img, stat_rect, cv::Scalar(76, 80, 88), 1);
    float acc = scene_accuracy(metrics);
    cv::putText(img, cv::format("matched %d / %d", metrics.matched, N_TRACKS),
                cv::Point(736, 118), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::putText(img, cv::format("ID accuracy %.1f%%", 100.0f * acc),
                cv::Point(736, 150), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(90, 230, 132), 1, cv::LINE_AA);
    cv::putText(img, cv::format("clutter grabs %d", metrics.clutter),
                cv::Point(736, 182), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(220, 160, 95), 1, cv::LINE_AA);
    cv::putText(img, cv::format("auction rounds %d", metrics.rounds),
                cv::Point(736, 214), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(205, 210, 218), 1, cv::LINE_AA);

    draw_history(img, accuracy_hist, match_hist, cv::Rect(720, 270, 218, 200));
    cv::putText(img, "solid: tracks  small: detections  x: clutter",
                cv::Point(720, 518), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(185, 190, 198), 1, cv::LINE_AA);
    cv::putText(img, "green links are correct ID associations",
                cv::Point(720, 544), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(185, 190, 198), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<TruthState> truth;
    std::vector<TrackState> tracks;
    init_truth_and_tracks(truth, tracks);
    std::vector<Detection> detections;
    make_detections(truth, 0, detections);

    TrackState* d_tracks = nullptr;
    Detection* d_dets = nullptr;
    int* d_assignments = nullptr;
    FrameMetrics* d_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tracks, N_SCENES * N_TRACKS * sizeof(TrackState)));
    CUDA_CHECK(cudaMalloc(&d_dets, N_SCENES * N_DETS * sizeof(Detection)));
    CUDA_CHECK(cudaMalloc(&d_assignments, N_SCENES * N_TRACKS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_metrics, N_SCENES * sizeof(FrameMetrics)));
    CUDA_CHECK(cudaMemcpy(d_tracks, tracks.data(), tracks.size() * sizeof(TrackState),
                          cudaMemcpyHostToDevice));

    BenchResult bench = benchmark_once(tracks, detections, d_tracks, d_dets,
                                       d_assignments, d_metrics);
    CUDA_CHECK(cudaMemcpy(d_tracks, tracks.data(), tracks.size() * sizeof(TrackState),
                          cudaMemcpyHostToDevice));
    std::printf("GPU assignment tracking: %d scenes x %d tracks x %d detections\n",
                N_SCENES, N_TRACKS, N_DETS);
    std::printf("GPU update %.3f ms, CPU %.3f ms, speedup %.1fx\n",
                bench.gpu_ms, bench.cpu_ms, bench.speedup);

    int mkdir_ret = std::system("mkdir -p gif");
    (void)mkdir_ret;
    cv::VideoWriter video("gif/gpu_assignment_tracking.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_assignment_tracking.avi\n");
        return 1;
    }

    std::vector<int> assignments(N_SCENES * N_TRACKS, -1);
    std::vector<FrameMetrics> metrics(N_SCENES);
    std::vector<std::vector<cv::Point>> trails(N_TRACKS);
    std::vector<float> accuracy_hist;
    std::vector<float> match_hist;
    double total_gpu_ms = 0.0;
    double total_acc = 0.0;
    double total_match = 0.0;

    for (int frame = 0; frame < N_FRAMES; frame++) {
        step_truth(truth, frame);
        make_detections(truth, frame, detections);
        float gpu_ms = gpu_update(d_tracks, d_dets, d_assignments, d_metrics,
                                  detections, tracks, assignments, metrics);
        total_gpu_ms += gpu_ms;

        for (int i = 0; i < N_TRACKS; i++) {
            cv::Point p = world_to_px(tracks[i].x, tracks[i].y,
                                      cv::Rect(34, 80, 660, 502));
            trails[i].push_back(p);
            if (trails[i].size() > 34) trails[i].erase(trails[i].begin());
        }
        float acc = scene_accuracy(metrics[0]);
        float match_rate = (float)metrics[0].matched / (float)N_TRACKS;
        accuracy_hist.push_back(acc);
        match_hist.push_back(match_rate);
        total_acc += acc;
        total_match += match_rate;

        cv::Mat img = draw_frame(truth, tracks, detections, assignments,
                                 metrics[0], trails, accuracy_hist, match_hist,
                                 frame, gpu_ms, bench);
        video.write(img);

        if (frame % 12 == 0 || frame == N_FRAMES - 1) {
            std::printf("frame %02d  gpu %.3f ms  matched %d  acc %.1f%%  clutter %d\n",
                        frame + 1, gpu_ms, metrics[0].matched,
                        100.0f * acc, metrics[0].clutter);
        }
    }

    for (int hold = 0; hold < 14; hold++) {
        cv::Mat img = draw_frame(truth, tracks, detections, assignments,
                                 metrics[0], trails, accuracy_hist, match_hist,
                                 N_FRAMES - 1, (float)(total_gpu_ms / N_FRAMES),
                                 bench);
        video.write(img);
    }
    video.release();

    std::printf("Average GPU tracking update %.3f ms/frame\n",
                total_gpu_ms / N_FRAMES);
    std::printf("Scene 0 average matched %.1f%%, ID accuracy %.1f%%\n",
                100.0 * total_match / N_FRAMES,
                100.0 * total_acc / N_FRAMES);

    cudabot::avi_to_gif("gif/gpu_assignment_tracking.avi",
                        "gif/gpu_assignment_tracking.gif", 8, 640);
    std::printf("GIF saved to gif/gpu_assignment_tracking.gif\n");

    CUDA_CHECK(cudaFree(d_tracks));
    CUDA_CHECK(cudaFree(d_dets));
    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFree(d_metrics));
    return 0;
}
