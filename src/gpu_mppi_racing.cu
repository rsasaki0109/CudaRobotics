// gpu_mppi_racing.cu
//
// GPU MPPI autonomous racing.  A kinematic-bicycle race car drives a closed
// circuit by Model-Predictive Path Integral control: every control step the
// GPU rolls out thousands of noisy candidate trajectories in parallel, scores
// each by how far it advances along the track (heavily penalising leaving the
// asphalt), and the softmax-weighted average of the perturbations updates the
// nominal control sequence.  One GPU thread = one sample trajectory -- the
// classic MPPI parallel pattern.
//
// Track geometry is baked once on the host into two grid look-ups: a *progress*
// field (arc-length along the centreline, metres) and a signed *distance* field
// (metres from the centreline).  The rollout cost is then an O(1) texture-style
// lookup per predicted state, so the per-step work is dominated by the K x T
// model integration -- exactly what the GPU parallelises.
//
// Thesis (validated, not assumed): the win is the parallel rollout.  We time
// the identical K-sample x T-horizon rollout+cost on the CPU and the GPU each
// step and report the measured speed-up, plus the lap times the controller
// actually achieves.
//
// One demo = one .cu; reuses include/cuda_check.cuh and include/cuda_video.h.

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"
#include "demo_args.h"
#include "mppi_reduction.cuh"

namespace cudabot {

// ---- MPPI ----------------------------------------------------------------
constexpr int   K_SAMPLES = 2048;   // candidate trajectories / step
constexpr int   T_HORIZON = 40;     // prediction steps
constexpr float DT        = 0.06f;
constexpr float WHEELBASE = 2.0f;
constexpr float LAMBDA    = 8.0f;

constexpr float MAX_ACCEL = 9.0f;
constexpr float MAX_STEER = 0.50f;
constexpr float MAX_SPEED = 16.0f;
constexpr float ACCEL_NOISE = 3.5f;
constexpr float STEER_NOISE = 0.22f;

// cost weights
constexpr float PROGRESS_W = 6.0f;   // reward per metre of track progress
constexpr float OFFTRACK_W = 60.0f;  // penalty per (metre beyond edge)^2
constexpr float SPEED_W    = 0.02f;  // mild reward for speed
constexpr float STEER_W    = 1.5f;   // steering effort

// ---- track / world -------------------------------------------------------
constexpr float WS    = 28.0f;       // half world extent (m)
constexpr float HALFW = 3.6f;        // track half width (m)
constexpr int   GRID  = 256;         // field grid resolution
constexpr int   NS    = 2048;        // centreline samples
constexpr int   IMG   = 760;
constexpr int   MAX_STEPS = 900;
constexpr int   TARGET_LAPS = 3;

// device copies of the baked fields
__constant__ float c_track_len;
float* d_progress = nullptr;   // [GRID*GRID] arc-length (m)
float* d_dist = nullptr;       // [GRID*GRID] distance to centreline (m)
float* d_curvature = nullptr;

// A bounded bicycle approximation: steering is limited by available lateral
// acceleration. The slippery quadrant is a known surface map, not an estimator.
__host__ __device__ inline float surface_grip(float x, float y, float scale) {
    return scale * (x < -3.0f && y > 3.0f ? 0.30f : 0.90f);
}
__host__ __device__ inline void advance_car(float& x, float& y, float& th, float& v,
                                           float accel, float steer, bool limited, float scale) {
    float yaw_rate = v / WHEELBASE * tanf(steer);
    if (limited) {
        const float capacity = surface_grip(x, y, scale) * 9.81f;
        accel = fminf(fmaxf(accel, -capacity), capacity);
        const float lateral = sqrtf(fmaxf(0.0f, capacity * capacity - accel * accel));
        const float max_yaw = lateral / fmaxf(v, 0.5f);
        yaw_rate = fminf(fmaxf(yaw_rate, -max_yaw), max_yaw);
    }
    x += v * cosf(th) * DT; y += v * sinf(th) * DT;
    th += yaw_rate * DT; v = fminf(fmaxf(v + accel * DT, 0.0f), MAX_SPEED);
}

// centreline parametric curve (smooth, non-self-intersecting circuit)
__host__ __device__ inline void centerline(float s, float& x, float& y) {
    // s in [0, 2pi)
    x = 21.0f * cosf(s) + 3.0f * cosf(3.0f * s);
    y = 15.0f * sinf(s) + 4.0f * sinf(2.0f * s);
}

// =========================================================================
// Kernels
// =========================================================================
__global__ void init_curand(curandState* st, int K, unsigned long long seed) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K) curand_init(seed, k, 0, &st[k]);
}

__device__ inline int grid_idx(float wx, float wy) {
    int gx = (int)((wx + WS) / (2.0f * WS) * GRID);
    int gy = (int)((wy + WS) / (2.0f * WS) * GRID);
    gx = min(max(gx, 0), GRID - 1);
    gy = min(max(gy, 0), GRID - 1);
    return gy * GRID + gx;
}

// One thread rolls out one trajectory; returns total cost + perturbations + xy path.
__global__ void rollout_kernel(float sx, float sy, float sth, float sv,
                               const float* __restrict__ nominal,    // [T*2]
                               const float* __restrict__ prog, const float* __restrict__ dist,
                               float* __restrict__ costs,            // [K]
                               float* __restrict__ pert,             // [K*T*2]
                               float* __restrict__ traj,             // [K*T*2]
                               curandState* __restrict__ rng, int K, const float* curvature,
                               bool grip_aware, float grip_scale) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curandState ls = rng[k];
    float x = sx, y = sy, th = sth, v = sv;
    float cost = 0.0f;
    float prev_s = prog[grid_idx(x, y)];
    const float TL = c_track_len;

    for (int t = 0; t < T_HORIZON; t++) {
        float na = nominal[t * 2 + 0] + curand_normal(&ls) * (grip_aware ? 0.8f : ACCEL_NOISE);
        float nd = nominal[t * 2 + 1] + curand_normal(&ls) * (grip_aware ? 0.07f : STEER_NOISE);
        na = fminf(fmaxf(na, -MAX_ACCEL), MAX_ACCEL);
        nd = fminf(fmaxf(nd, -MAX_STEER), MAX_STEER);
        pert[(k * T_HORIZON + t) * 2 + 0] = na;
        pert[(k * T_HORIZON + t) * 2 + 1] = nd;

        if (grip_aware) {
            const float capacity = surface_grip(x, y, grip_scale) * 9.81f;
            const float lat = v * v / WHEELBASE * tanf(nd);
            const float demand = sqrtf(lat * lat + na * na);
            const float excess = fmaxf(0.0f, demand - 0.85f * capacity);
            cost += 4.0f * excess * excess;
            const float target = sqrtf(0.75f * capacity / fmaxf(curvature[grid_idx(x, y)], 0.01f));
            const float over_speed = fmaxf(0.0f, v - target);
            cost += 3.0f * over_speed * over_speed;
        }
        advance_car(x, y, th, v, na, nd, grip_aware, grip_scale);

        traj[(k * T_HORIZON + t) * 2 + 0] = x;
        traj[(k * T_HORIZON + t) * 2 + 1] = y;

        int gi = grid_idx(x, y);
        float s = prog[gi];
        float d = dist[gi];
        // progress reward with wrap handling
        float ds = s - prev_s;
        if (ds < -0.5f * TL) ds += TL;
        if (ds >  0.5f * TL) ds -= TL;
        ds = fminf(fmaxf(ds, -2.0f), 2.0f);
        cost -= PROGRESS_W * ds;
        prev_s = s;
        // off-track penalty
        float over = d - HALFW;
        if (over > 0.0f) cost += OFFTRACK_W * over * over;
        // mild shaping
        cost -= SPEED_W * v;
        cost += STEER_W * nd * nd;
        rng[k] = ls;  // not strictly needed but keeps states advancing
    }
    rng[k] = ls;
    costs[k] = cost;
}

// softmax weights from costs (single block, K <= blockDim*?) -> use two passes on host instead.
__global__ void weighted_update_kernel(const float* __restrict__ costs,
                                        const float* __restrict__ pert,
                                        float beta, float eta,          // beta=1/lambda, eta=1/sum_w
                                        float min_cost,
                                        float* __restrict__ nominal, int K) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;   // one thread = one horizon step
    if (t >= T_HORIZON) return;
    float acc_a = 0.0f, acc_d = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = __expf(-beta * (costs[k] - min_cost)) * eta;
        acc_a += w * pert[(k * T_HORIZON + t) * 2 + 0];
        acc_d += w * pert[(k * T_HORIZON + t) * 2 + 1];
    }
    nominal[t * 2 + 0] = acc_a;
    nominal[t * 2 + 1] = acc_d;
}

// =========================================================================
// CPU reference rollout (same math) -- used only to time the parallel win.
// =========================================================================
static void rollout_cpu(float sx, float sy, float sth, float sv,
                        const std::vector<float>& nominal,
                        const std::vector<float>& prog, const std::vector<float>& dist,
                        float track_len, std::vector<float>& costs, int K, unsigned seed) {
    auto gidx = [](float wx, float wy) {
        int gx = (int)((wx + WS) / (2.0f * WS) * GRID);
        int gy = (int)((wy + WS) / (2.0f * WS) * GRID);
        gx = std::min(std::max(gx, 0), GRID - 1);
        gy = std::min(std::max(gy, 0), GRID - 1);
        return gy * GRID + gx;
    };
    for (int k = 0; k < K; k++) {
        unsigned st = seed + k * 2654435761u + 1u;
        auto nrm = [&]() {            // cheap gaussian (Box-Muller-ish via uniforms)
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            float u1 = (st & 0xffffff) / 16777216.0f + 1e-7f;
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            float u2 = (st & 0xffffff) / 16777216.0f;
            return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
        };
        float x = sx, y = sy, th = sth, v = sv, cost = 0.0f;
        float prev_s = prog[gidx(x, y)];
        for (int t = 0; t < T_HORIZON; t++) {
            float na = nominal[t * 2 + 0] + nrm() * ACCEL_NOISE;
            float nd = nominal[t * 2 + 1] + nrm() * STEER_NOISE;
            na = std::min(std::max(na, -MAX_ACCEL), MAX_ACCEL);
            nd = std::min(std::max(nd, -MAX_STEER), MAX_STEER);
            x += v * std::cos(th) * DT; y += v * std::sin(th) * DT;
            th += v / WHEELBASE * std::tan(nd) * DT; v += na * DT;
            v = std::min(std::max(v, 0.0f), MAX_SPEED);
            int gi = gidx(x, y);
            float ds = prog[gi] - prev_s;
            if (ds < -0.5f * track_len) ds += track_len;
            if (ds >  0.5f * track_len) ds -= track_len;
            ds = std::min(std::max(ds, -2.0f), 2.0f);
            cost -= PROGRESS_W * ds; prev_s = prog[gi];
            float over = dist[gi] - HALFW;
            if (over > 0.0f) cost += OFFTRACK_W * over * over;
            cost -= SPEED_W * v; cost += STEER_W * nd * nd;
        }
        costs[k] = cost;
    }
}

// =========================================================================
static cv::Point w2p(float wx, float wy) {
    int px = (int)((wx + WS) / (2.0f * WS) * IMG);
    int py = (int)((1.0f - (wy + WS) / (2.0f * WS)) * IMG);
    return cv::Point(px, py);
}

}  // namespace cudabot
using namespace cudabot;

int main(int argc, char** argv) {
    DemoArgs args(argc, argv, "GPU MPPI autonomous racing on a closed circuit");
    const int K = args.get_int("samples", K_SAMPLES, "candidate trajectories per step", 1);
    const int target_laps = args.get_int("laps", TARGET_LAPS, "stop after this many laps", 1);
    const int max_steps = args.get_int("steps", MAX_STEPS, "maximum control steps", 1);
    const int seed = args.get_int("seed", 1234, "cuRAND seed", 0);
    const bool write_video = !args.flag("no-video", "skip the AVI/GIF output");
    const bool grip_plant = args.flag("grip-plant", "apply friction limits to the true vehicle");
    const bool grip_aware = args.flag("grip-aware", "plan using the known surface grip and friction circle");
    const float grip_scale = args.get_float("grip-scale", 1.0f, "multiply dry and slippery surface grip");
    const bool cpu_reference = args.flag("cpu-reference", "time the legacy single-thread CPU reference (kinematic mode only)");
    const bool require_success = args.flag("require-success", "exit nonzero unless all laps finish without leaving the track");
    args.finish();
    if (!std::isfinite(grip_scale) || grip_scale <= 0 || (grip_aware && !grip_plant) || (cpu_reference && grip_aware)) {
        std::fprintf(stderr, "positive grip required; --grip-aware requires --grip-plant; CPU reference supports kinematic mode\n"); return 2;
    }

    // ---- bake the centreline + progress/distance fields -----------------
    std::vector<float> cs_x(NS), cs_y(NS), cs_s(NS);
    float track_len = 0.0f;
    float px = 0, py = 0;
    centerline(0.0f, px, py);
    cs_x[0] = px; cs_y[0] = py; cs_s[0] = 0.0f;
    for (int i = 1; i < NS; i++) {
        float s = 2.0f * float(M_PI) * i / NS, x, y;
        centerline(s, x, y);
        track_len += std::sqrt((x - px) * (x - px) + (y - py) * (y - py));
        cs_x[i] = x; cs_y[i] = y; cs_s[i] = track_len;
        px = x; py = y;
    }
    { float x0, y0; centerline(0.0f, x0, y0);
      track_len += std::sqrt((x0 - px) * (x0 - px) + (y0 - py) * (y0 - py)); }
    std::printf("Track: closed circuit, length %.1f m, half-width %.1f m\n", track_len, HALFW);

    std::vector<float> h_prog(GRID * GRID), h_dist(GRID * GRID), h_curvature(GRID * GRID);
    for (int gy = 0; gy < GRID; gy++)
        for (int gx = 0; gx < GRID; gx++) {
            float wx = (gx + 0.5f) / GRID * 2.0f * WS - WS;
            float wy = (gy + 0.5f) / GRID * 2.0f * WS - WS;
            float best = 1e18f; int bi = 0;
            for (int i = 0; i < NS; i++) {
                float dx = wx - cs_x[i], dy = wy - cs_y[i];
                float d2 = dx * dx + dy * dy;
                if (d2 < best) { best = d2; bi = i; }
            }
            h_prog[gy * GRID + gx] = cs_s[bi];
            h_dist[gy * GRID + gx] = std::sqrt(best);
            const int prev = (bi + NS - 4) % NS, next = (bi + 4) % NS;
            const float ax = cs_x[bi] - cs_x[prev], ay = cs_y[bi] - cs_y[prev];
            const float bx = cs_x[next] - cs_x[bi], by = cs_y[next] - cs_y[bi];
            h_curvature[gy * GRID + gx] = 2.0f * std::fabs(ax * by - ay * bx) /
                std::max(1e-6f, std::hypot(ax, ay) * std::hypot(bx, by) * std::hypot(ax + bx, ay + by));
        }

    CUDA_CHECK(cudaMalloc(&d_progress, GRID * GRID * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist, GRID * GRID * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_progress, h_prog.data(), GRID * GRID * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dist, h_dist.data(), GRID * GRID * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_track_len, &track_len, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_curvature, GRID * GRID * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_curvature, h_curvature.data(), GRID * GRID * sizeof(float), cudaMemcpyHostToDevice));

    // ---- device MPPI buffers --------------------------------------------
    float *d_nominal, *d_costs, *d_pert, *d_traj, *d_weights;
    curandState* d_rng;
    CUDA_CHECK(cudaMalloc(&d_nominal, T_HORIZON * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_costs, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pert, K * T_HORIZON * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_traj, K * T_HORIZON * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rng, K * sizeof(curandState)));
    init_curand<<<(K + 127) / 128, 128>>>(d_rng, K, (unsigned long long)seed);

    std::vector<float> nominal(T_HORIZON * 2, 0.0f);
    for (int t = 0; t < T_HORIZON; t++) nominal[t * 2 + 0] = 3.0f;   // warm start: accelerate
    std::vector<float> h_costs(K), h_traj(K * T_HORIZON * 2);

    // ---- true car state: start on the track at s=0 ----------------------
    float x0, y0; centerline(0.0f, x0, y0);
    float x1, y1; centerline(0.03f, x1, y1);
    float car_x = x0, car_y = y0, car_th = std::atan2(y1 - y0, x1 - x0), car_v = 4.0f;

    // ---- precompute track background image ------------------------------
    cv::Mat bg(IMG, IMG, CV_8UC3, cv::Scalar(28, 32, 30));
    for (int yy = 0; yy < IMG; yy++)
        for (int xx = 0; xx < IMG; xx++) {
            float wx = (xx + 0.5f) / IMG * 2.0f * WS - WS;
            float wy = (1.0f - (yy + 0.5f) / IMG) * 2.0f * WS - WS;
            int gx = std::min(std::max((int)((wx + WS) / (2 * WS) * GRID), 0), GRID - 1);
            int gy = std::min(std::max((int)((wy + WS) / (2 * WS) * GRID), 0), GRID - 1);
            float d = h_dist[gy * GRID + gx];
            if (d <= HALFW) bg.at<cv::Vec3b>(yy, xx) = grip_plant && wx < -3 && wy > 3 ? cv::Vec3b(105, 80, 45) : cv::Vec3b(60, 60, 62);
            else if (d <= HALFW + 0.5f) bg.at<cv::Vec3b>(yy, xx) = cv::Vec3b(40, 220, 240); // kerb
        }
    // dashed centreline + start/finish
    for (int i = 0; i < NS; i += 24) {
        cv::line(bg, w2p(cs_x[i], cs_y[i]), w2p(cs_x[(i + 12) % NS], cs_y[(i + 12) % NS]),
                 cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
    }
    cv::line(bg, w2p(x0, y0 + HALFW), w2p(x0, y0 - HALFW), cv::Scalar(255, 255, 255), 3, cv::LINE_AA);

    cv::VideoWriter video;
    const std::string video_tag = grip_plant ? (grip_aware ? "gpu_mppi_racing_grip_aware" : "gpu_mppi_racing_grip_blind") : "gpu_mppi_racing";
    if (write_video) {
        cudabot::ensure_dirs({"gif"});
        video.open("gif/" + video_tag + ".avi",
                   cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 1.0 / (3 * DT), cv::Size(IMG, IMG));
    }

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float gpu_ms_total = 0.0f, cpu_ms_total = 0.0f; int timing_n = 0;

    int laps = 0; float last_s = 0.0f; float best_lap = 1e9f; float lap_start_t = 0.0f;
    std::vector<float> lap_times;
    float top_speed = 0.0f;
    int steps_run = 0;
    int offtrack_steps = 0; float max_offset = 0.0f;
    std::vector<double> solve_times;

    for (int step = 0; step < max_steps && laps < target_laps; step++) {
        steps_run++;
        const auto solve_start = std::chrono::steady_clock::now();
        if (grip_aware) {
            // Feasible path-following proposals avoid wasting the reduced sampling
            // budget on acceleration/steering combinations outside the friction circle.
            float x = car_x, y = car_y, th = car_th, v = car_v;
            for (int t = 0; t < T_HORIZON; t++) {
                const int gx = std::max(0, std::min(GRID-1, (int)((x+WS)/(2*WS)*GRID)));
                const int gy = std::max(0, std::min(GRID-1, (int)((y+WS)/(2*WS)*GRID)));
                const int gi = gy * GRID + gx;
                const float look = std::max(2.0f, 0.55f * v);
                const float wanted_s = std::fmod(h_prog[gi] + look, track_len);
                int target = (int)(std::lower_bound(cs_s.begin(), cs_s.end(), wanted_s) - cs_s.begin());
                target = std::min(target, NS-1);
                const float dx = cs_x[target]-x, dy = cs_y[target]-y;
                const float error = std::remainder(std::atan2(dy, dx)-th, 2.0f*(float)M_PI);
                float steer = std::atan2(2.0f*WHEELBASE*std::sin(error), std::max(1.0f, std::hypot(dx, dy)));
                steer = std::max(-MAX_STEER, std::min(MAX_STEER, steer));
                const float capacity = surface_grip(x, y, grip_scale) * 9.81f;
                const float curve = std::max(h_curvature[gi], std::fabs(std::tan(steer)/WHEELBASE));
                const float target_v = std::min(MAX_SPEED, std::sqrt(0.65f*capacity/std::max(curve, 0.01f)));
                float accel = std::max(-0.5f*capacity, std::min(0.5f*capacity, 1.8f*(target_v-v)));
                nominal[2*t] = accel; nominal[2*t+1] = steer;
                advance_car(x, y, th, v, accel, steer, true, grip_scale);
            }
        }
        CUDA_CHECK(cudaMemcpy(d_nominal, nominal.data(), T_HORIZON * 2 * sizeof(float), cudaMemcpyHostToDevice));

        // --- GPU rollout (timed) ---
        cudaEventRecord(e0);
        rollout_kernel<<<(K + 127) / 128, 128>>>(
            car_x, car_y, car_th, car_v, d_nominal, d_progress, d_dist,
            d_costs, d_pert, d_traj, d_rng, K, d_curvature, grip_aware, grip_scale);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float gms = 0; cudaEventElapsedTime(&gms, e0, e1); gpu_ms_total += gms;

        cudabot::launch_softmin_weights(d_costs, d_weights, K, LAMBDA);
        cudabot::launch_weighted_control_update(d_pert, d_weights, d_nominal, K, T_HORIZON * 2);
        CUDA_CHECK(cudaMemcpy(nominal.data(), d_nominal, T_HORIZON * 2 * sizeof(float), cudaMemcpyDeviceToHost));
        solve_times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - solve_start).count());

        // --- CPU reference rollout (timed, identical work) ---
        if (cpu_reference && step % 12 == 0) {
            std::vector<float> cc(K);
            auto t0 = std::chrono::high_resolution_clock::now();
            rollout_cpu(car_x, car_y, car_th, car_v, nominal, h_prog, h_dist, track_len, cc, K, 99u + step);
            auto t1 = std::chrono::high_resolution_clock::now();
            cpu_ms_total += std::chrono::duration<float, std::milli>(t1 - t0).count();
            gpu_ms_total += 0; timing_n++;   // pair with the gpu time recorded this step
        }

        // --- apply first control, advance the true car ---
        float a = std::min(std::max(nominal[0], -MAX_ACCEL), MAX_ACCEL);
        float d = std::min(std::max(nominal[1], -MAX_STEER), MAX_STEER);
        advance_car(car_x, car_y, car_th, car_v, a, d, grip_plant, grip_scale);
        top_speed = std::max(top_speed, car_v);

        // shift nominal (receding horizon)
        for (int t = 0; t < T_HORIZON - 1; t++) {
            nominal[t * 2 + 0] = nominal[(t + 1) * 2 + 0];
            nominal[t * 2 + 1] = nominal[(t + 1) * 2 + 1];
        }

        // --- lap counting (progress wrap across the start line) ---
        int gx = std::min(std::max((int)((car_x + WS) / (2 * WS) * GRID), 0), GRID - 1);
        int gy = std::min(std::max((int)((car_y + WS) / (2 * WS) * GRID), 0), GRID - 1);
        float cur_s = h_prog[gy * GRID + gx];
        const float offset = h_dist[gy * GRID + gx];
        max_offset = std::max(max_offset, offset); offtrack_steps += offset > HALFW;
        if (last_s > 0.75f * track_len && cur_s < 0.25f * track_len) {
            float lt = (step * DT) - lap_start_t;
            if (step * DT > 1.0f) { lap_times.push_back(lt); best_lap = std::min(best_lap, lt); laps++; }
            lap_start_t = step * DT;
        }
        last_s = cur_s;
        if (!write_video) { if (offset > HALFW + 1.0f) break; continue; }

        // --- draw ---
        CUDA_CHECK(cudaMemcpy(h_traj.data(), d_traj, K * T_HORIZON * 2 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_costs.data(), d_costs, K * sizeof(float), cudaMemcpyDeviceToHost));
        cv::Mat img = bg.clone();
        // a sample of candidate rollouts (faint)
        for (int k = 0; k < K; k += 40) {
            cv::Point prev = w2p(car_x, car_y);
            for (int t = 0; t < T_HORIZON; t += 2) {
                cv::Point p = w2p(h_traj[(k * T_HORIZON + t) * 2], h_traj[(k * T_HORIZON + t) * 2 + 1]);
                cv::line(img, prev, p, cv::Scalar(70, 120, 90), 1, cv::LINE_AA);
                prev = p;
            }
        }
        // chosen (lowest-cost) rollout, bright
        int kbest = (int)(std::min_element(h_costs.begin(), h_costs.end()) - h_costs.begin());
        cv::Point prev = w2p(car_x, car_y);
        for (int t = 0; t < T_HORIZON; t++) {
            cv::Point p = w2p(h_traj[(kbest * T_HORIZON + t) * 2], h_traj[(kbest * T_HORIZON + t) * 2 + 1]);
            cv::line(img, prev, p, cv::Scalar(60, 200, 255), 2, cv::LINE_AA);
            prev = p;
        }
        // the car (oriented triangle)
        {
            float c = std::cos(car_th), s = std::sin(car_th);
            cv::Point2f nose(car_x + 1.4f * c, car_y + 1.4f * s);
            cv::Point2f l(car_x - 1.0f * c - 0.8f * s, car_y - 1.0f * s + 0.8f * c);
            cv::Point2f r(car_x - 1.0f * c + 0.8f * s, car_y - 1.0f * s - 0.8f * c);
            std::vector<cv::Point> tri = {w2p(nose.x, nose.y), w2p(l.x, l.y), w2p(r.x, r.y)};
            cv::fillConvexPoly(img, tri, cv::Scalar(40, 60, 230), cv::LINE_AA);
        }
        // HUD
        float speedup = (cpu_ms_total > 0 && timing_n > 0) ? (cpu_ms_total / timing_n) / (gpu_ms_total / (step + 1)) : 0.0f;
        char hud[200];
        std::snprintf(hud, sizeof(hud), "MPPI racing  K=%d x T=%d   speed=%.1f m/s   lap %d/%d   GPU %.2f ms",
                      K, T_HORIZON, car_v, laps, target_laps, gpu_ms_total / (step + 1));
        cv::putText(img, hud, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        char hud2[200];
        std::snprintf(hud2, sizeof(hud2), "%s  grip %.2f  off-track %d  best lap %.2f s",
                      grip_aware ? "GRIP AWARE" : "KINEMATIC PLAN", surface_grip(car_x, car_y, grip_scale), offtrack_steps,
                      (best_lap < 1e8 ? best_lap : 0.0f));
        cv::putText(img, hud2, cv::Point(12, 52), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 230, 255), 1, cv::LINE_AA);

        if (step % 3 == 0) video.write(img);
        if (offset > HALFW + 1.0f) break;
    }
    video.release();

    float gpu_avg = steps_run > 0 ? gpu_ms_total / steps_run : 0.0f;
    float cpu_avg = (timing_n > 0) ? cpu_ms_total / timing_n : 0.0f;
    std::printf("\n=== GPU MPPI racing ===\n");
    std::printf("rollouts/step:    %d trajectories x %d horizon, %d control steps\n", K, T_HORIZON, steps_run);
    std::printf("GPU rollout:      %.3f ms / step\n", gpu_avg);
    if (cpu_reference) {
        std::printf("CPU rollout:      %.3f ms / step (same dimensions, independently sampled controls)\n", cpu_avg);
        std::printf("speed-up:         %.1fx (GPU parallel rollout vs single-thread CPU)\n",
                    gpu_avg > 0 ? cpu_avg / gpu_avg : 0.0f);
    }
    std::printf("laps completed:   %d\n", (int)lap_times.size());
    for (size_t i = 0; i < lap_times.size(); i++) std::printf("  lap %zu: %.2f s\n", i + 1, lap_times[i]);
    std::printf("best lap:         %.2f s   top speed %.1f m/s\n", (best_lap < 1e8 ? best_lap : 0.0f), top_speed);
    std::sort(solve_times.begin(), solve_times.end());
    double solve_mean = 0; for (double t : solve_times) solve_mean += t;
    std::printf("final pose: %.2f %.2f, speed %.2f\n", car_x, car_y, car_v);
    std::printf("RESULT racing mode=%s seed=%d laps=%d target=%d offtrack=%d max_offset=%.4f sim_s=%.4f best_lap=%.4f mean_ms=%.4f p95_ms=%.4f\n",
                grip_aware ? "aware" : "blind", seed, laps, target_laps, offtrack_steps, max_offset, steps_run * DT,
                best_lap < 1e8 ? best_lap : 0.0f, solve_mean / std::max(1, steps_run), solve_times[(solve_times.size()-1)*95/100]);

    if (write_video) {
        avi_to_gif("gif/" + video_tag + ".avi", "gif/" + video_tag + ".gif", 12, 600);
        std::printf("GIF saved to gif/%s.gif\n", video_tag.c_str());
    }

    cudaFree(d_progress); cudaFree(d_dist); cudaFree(d_nominal); cudaFree(d_costs);
    cudaFree(d_pert); cudaFree(d_traj); cudaFree(d_rng);
    cudaFree(d_weights); cudaFree(d_curvature); cudaEventDestroy(e0); cudaEventDestroy(e1);
    return require_success && (laps < target_laps || offtrack_steps) ? 1 : 0;
}
