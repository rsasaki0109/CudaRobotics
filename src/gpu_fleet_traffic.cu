// GPU intersection reservations: serial access versus compatible platoons.
// Four straight lanes, identical car-following plant and arrival queues.
// A reservation remains held until every admitted vehicle has cleared it.
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include "cuda_check.cuh"
#include "cuda_video.h"
#include "demo_args.h"

constexpr float FDT = 0.1f, GATE = -2.0f, EXIT = 2.0f, DEST = 5.0f;
constexpr float GAP = 0.70f, RADIUS = 0.20f, SPEED = 1.8f, ACCEL = 1.2f;
struct Agent { float s, v, waiting, completion; int lane, admitted, done; };
struct Reservation { int group, owner; float opened; };

__host__ __device__ inline void position(const Agent& a, float& x, float& y) {
    if (a.lane == 0) { x = a.s; y = -0.65f; }
    else if (a.lane == 1) { x = -a.s; y = 0.65f; }
    else if (a.lane == 2) { x = 0.65f; y = a.s; }
    else { x = -0.65f; y = -a.s; }
}

__global__ void reserve(const Agent* agents, int n, Reservation* reservation, float time, bool platoon) {
    if (blockIdx.x || threadIdx.x) return;
    Reservation r = *reservation;
    bool occupied = false;
    for (int i = 0; i < n; i++)
        occupied |= !agents[i].done && agents[i].admitted && agents[i].s <= EXIT;
    if (!occupied && (r.group < 0 || !platoon || time - r.opened >= 3.0f)) {
        int best = -1; float oldest = -1;
        for (int i = 0; i < n; i++) {
            const Agent a = agents[i];
            if (a.done || a.admitted || a.s < GATE - 0.15f) continue;
            if (a.waiting > oldest) { oldest = a.waiting; best = i; }
        }
        r.owner = best; r.group = best < 0 ? -1 : agents[best].lane / 2; r.opened = time;
    }
    *reservation = r;
}

__global__ void advance(const Agent* in, Agent* out, int n, const Reservation* reservation,
                        float time, bool platoon) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Agent a = in[i];
    if (a.done) { out[i] = a; return; }
    const Reservation r = *reservation;
    const bool permission = platoon ? a.lane / 2 == r.group && time - r.opened < 3.0f : i == r.owner;
    float limit = 1e6f;
    for (int j = 0; j < n; j++)
        if (i != j && !in[j].done && in[j].lane == a.lane && in[j].s > a.s)
            limit = fminf(limit, in[j].s - GAP);
    if (!a.admitted && !permission) limit = fminf(limit, GATE);
    float next = fminf(a.s + fminf(SPEED, a.v + ACCEL * FDT) * FDT, limit);
    next = fmaxf(a.s, next);
    a.v = (next - a.s) / FDT; a.s = next;
    if (a.s > GATE) a.admitted = 1;
    if (a.v < 0.1f) a.waiting += FDT;
    if (a.s >= DEST) { a.done = 1; a.completion = time + FDT; }
    out[i] = a;
}

static cv::Point pixel(float x, float y) { return cv::Point((int)(420 + x * 11), (int)(420 - y * 11)); }
static cv::Mat frame(const std::vector<Agent>& agents, const Reservation& r, float time, bool platoon, int arrived) {
    cv::Mat img(840, 1040, CV_8UC3, cv::Scalar(24, 29, 32));
    cv::rectangle(img, cv::Rect(0, 397, 840, 46), cv::Scalar(61, 66, 70), cv::FILLED);
    cv::rectangle(img, cv::Rect(397, 0, 46, 840), cv::Scalar(61, 66, 70), cv::FILLED);
    cv::rectangle(img, pixel(-2, 2), pixel(2, -2), cv::Scalar(100, 150, 180), 1);
    for (const Agent& a : agents) {
        if (a.done) continue;
        float x, y; position(a, x, y);
        cv::circle(img, pixel(x, y), 3, a.admitted ? cv::Scalar(80, 230, 110) : cv::Scalar(255, 170, 60), cv::FILLED, cv::LINE_AA);
    }
    char text[160];
    cv::putText(img, platoon ? "GPU COMPATIBLE PLATOONS" : "SERIAL RESERVATIONS", cv::Point(15, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(240, 240, 240), 2);
    std::snprintf(text, sizeof(text), "time %.1f s   delivered %d / %zu", time, arrived, agents.size());
    cv::putText(img, text, cv::Point(15, 58), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(220, 220, 220), 1);
    std::snprintf(text, sizeof(text), "reservation: %s", r.group == 0 ? "east / west" : r.group == 1 ? "north / south" : "waiting");
    cv::putText(img, text, cv::Point(15, 82), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(80, 230, 110), 1);
    return img;
}

int main(int argc, char** argv) {
    cudabot::DemoArgs args(argc, argv, "GPU fleet intersection: fair serial reservations or compatible platoons");
    const int n = args.get_int("robots", 200, "number of agents (multiple of four)", 4);
    const int steps = args.get_int("steps", 2400, "maximum simulation steps", 1);
    const int seed = args.get_int("seed", 1, "arrival-spacing seed", 0);
    const bool platoon = args.flag("platoon", "batch compatible lanes under a bounded reservation");
    const bool video_on = !args.flag("no-video", "skip AVI/GIF");
    const bool require_completion = args.flag("require-completion", "exit nonzero unless every agent is delivered");
    args.finish();
    if (n % 4 || n > 2000) { std::fprintf(stderr, "--robots must be a multiple of four, at most 2000\n"); return 2; }
    std::mt19937 rng(seed); std::uniform_real_distribution<float> jitter(0, 0.12f);
    std::vector<Agent> agents(n), previous;
    for (int lane = 0; lane < 4; lane++) {
        float s = GATE - 0.05f;
        for (int k = 0; k < n / 4; k++) {
            Agent a{}; a.lane = lane; a.s = s; a.completion = -1;
            agents[lane + 4 * k] = a; s -= GAP + 0.05f + jitter(rng);
        }
    }
    Agent *a, *b; Reservation *reservation;
    CUDA_CHECK(cudaMalloc(&a, n * sizeof(Agent))); CUDA_CHECK(cudaMalloc(&b, n * sizeof(Agent)));
    CUDA_CHECK(cudaMalloc(&reservation, sizeof(Reservation)));
    Reservation r{-1, -1, 0};
    CUDA_CHECK(cudaMemcpy(a, agents.data(), n * sizeof(Agent), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reservation, &r, sizeof(r), cudaMemcpyHostToDevice));
    const std::string tag = platoon ? "gpu_fleet_traffic_platoon" : "gpu_fleet_traffic_serial";
    cv::VideoWriter video;
    if (video_on) {
        cudabot::ensure_dirs({"gif"});
        video.open("gif/" + tag + ".avi", cudabot::avi_fourcc(), 10, cv::Size(1040, 840));
        if (!video.isOpened()) { std::fprintf(stderr, "cannot open video writer\n"); return 2; }
    }
    std::vector<double> times; int arrived = 0, collisions = 0, last_step = 0;
    for (int step = 0; step < steps; step++) {
        previous = agents;
        auto t0 = std::chrono::steady_clock::now();
        reserve<<<1, 1>>>(a, n, reservation, step * FDT, platoon);
        advance<<<(n + 127) / 128, 128>>>(a, b, n, reservation, step * FDT, platoon);
        CUDA_CHECK(cudaMemcpy(agents.data(), b, n * sizeof(Agent), cudaMemcpyDeviceToHost));
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        std::swap(a, b); arrived = 0;
        for (int i = 0; i < n; i++) {
            arrived += agents[i].done;
            // Closest approach along swept segments, independently of the scheduler.
            for (int j = 0; j < i; j++) {
                if (previous[i].done || previous[j].done) continue;
                float ix, iy, jx, jy, ox, oy, px, py;
                position(previous[i], ix, iy); position(previous[j], jx, jy);
                position(agents[i], ox, oy); position(agents[j], px, py);
                const float dx = ix-jx, dy = iy-jy, vx = ox-ix-px+jx, vy = oy-iy-py+jy;
                const float f = std::max(0.0f, std::min(1.0f, -(dx*vx+dy*vy) / std::max(1e-9f, vx*vx+vy*vy)));
                collisions += (dx+f*vx)*(dx+f*vx)+(dy+f*vy)*(dy+f*vy) < 4*RADIUS*RADIUS;
            }
        }
        last_step = step + 1;
        if (video_on && step % 2 == 0) {
            CUDA_CHECK(cudaMemcpy(&r, reservation, sizeof(r), cudaMemcpyDeviceToHost));
            video.write(frame(agents, r, last_step * FDT, platoon, arrived));
        }
        if (arrived == n) break;
    }
    double mean = 0; for (double t : times) mean += t; mean /= times.size();
    std::sort(times.begin(), times.end());
    std::vector<float> waits; for (const Agent& a : agents) waits.push_back(a.waiting);
    std::sort(waits.begin(), waits.end());
    std::printf("RESULT fleet mode=%s seed=%d robots=%d arrived=%d collisions=%d sim_s=%.4f horizon_s=%.4f throughput=%.4f wait_p95=%.4f mean_ms=%.4f p95_ms=%.4f\n",
                platoon ? "platoon" : "serial", seed, n, arrived, collisions, last_step * FDT,
                steps * FDT, arrived / (steps * FDT), waits[(n-1)*95/100], mean, times[(times.size()-1)*95/100]);
    if (video_on) { video.release(); cudabot::avi_to_gif("gif/"+tag+".avi", "gif/"+tag+".gif", 10, 840); }
    cudaFree(a); cudaFree(b); cudaFree(reservation);
    return collisions || (require_completion && arrived < n) ? 1 : 0;
}
