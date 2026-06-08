// gpu_diffusion_planner.cu
//
// GPU diffusion-based motion planner.
//
// Diffusion planners (Janner et al., "Diffuser"; Chi et al., "Diffusion
// Policy") sample trajectories by Langevin-style denoising of a sequence
// of waypoints, starting from pure noise:
//      τ_{k-1} = τ_k + α_k s(τ_k) + √(2α_k) ε,    ε ~ N(0, I).
// The score s(τ) = ∇_τ log p(τ) is normally learned from data.  This
// program replaces the learned score with an analytic stand-in that
// encodes the three things a planner cares about:
//      (1) trajectory smoothness (Laplacian along the time axis)
//      (2) endpoint attraction (boundary terms holding τ_0=start, τ_T=goal)
//      (3) obstacle repulsion (gradient of an ESDF-like distance field)
// Mathematically the inference loop is identical to a learned diffusion
// planner: same Langevin step, same noise schedule.
//
// Per denoising step we update N_TRAJ trajectories of N_WAYPOINTS each
// in parallel — one CUDA thread per (trajectory, waypoint), so
// N_TRAJ x N_WAYPOINTS threads fire at once.
//
// Output: gif/gpu_diffusion_planner.gif — animated denoising of 512
//         trajectories from noise to obstacle-free paths.

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                         cudaGetErrorString(err), __FILE__, __LINE__);    \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

namespace cudabot {

constexpr int N_TRAJ = 512;
constexpr int N_WAYPOINTS = 64;
constexpr int N_STEPS = 96;
constexpr float ALPHA_START = 0.18f;
constexpr float ALPHA_END = 0.004f;
constexpr float NOISE_SCALE_START = 0.55f;
constexpr float NOISE_SCALE_END = 0.025f;

constexpr float W_SMOOTH = 1.8f;
constexpr float W_GOAL = 0.55f;
constexpr float W_REPEL = 3.5f;
constexpr float REPEL_RADIUS = 1.4f;

constexpr float START_X = -8.0f, START_Y = -5.5f;
constexpr float GOAL_X  =  8.0f, GOAL_Y =  5.5f;

constexpr int MAX_OBSTACLES = 8;
struct Obstacle { float cx, cy, r; };

__constant__ Obstacle c_obs[MAX_OBSTACLES];
__constant__ int c_n_obs;

constexpr int PANEL_W = 900;
constexpr int PANEL_H = 520;
constexpr float WORLD_HALF_X = 11.0f;
constexpr float WORLD_HALF_Y = 7.5f;
constexpr int VIDEO_FPS = 24;

// -------------------------------------------------------------------------
// Kernels
// -------------------------------------------------------------------------

// Initialize trajectories from pure Gaussian noise around the straight-line
// interpolant between start and goal (so they fan out from a sensible prior).
__global__ void init_trajectories_kernel(int n_traj, int T,
                                         float sx, float sy,
                                         float gx, float gy,
                                         float noise_scale,
                                         unsigned long long seed,
                                         float* traj_x, float* traj_y) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= T || i >= n_traj) return;
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, i * T + t, 0, &rng);
    float s = static_cast<float>(t) / (T - 1);
    float prior_x = (1.0f - s) * sx + s * gx;
    float prior_y = (1.0f - s) * sy + s * gy;
    if (t == 0) {
        traj_x[i * T + t] = sx; traj_y[i * T + t] = sy;
    } else if (t == T - 1) {
        traj_x[i * T + t] = gx; traj_y[i * T + t] = gy;
    } else {
        traj_x[i * T + t] = prior_x + noise_scale * curand_normal(&rng);
        traj_y[i * T + t] = prior_y + noise_scale * curand_normal(&rng);
    }
}

// Compute analytic score for trajectory point (i, t) and apply Langevin step.
// One thread per (trajectory, waypoint).
__global__ void langevin_step_kernel(int n_traj, int T,
                                     float sx, float sy,
                                     float gx, float gy,
                                     float alpha, float noise_scale,
                                     unsigned long long seed, int step_index,
                                     const float* __restrict__ traj_x_in,
                                     const float* __restrict__ traj_y_in,
                                     float* __restrict__ traj_x_out,
                                     float* __restrict__ traj_y_out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= T || i >= n_traj) return;

    // Endpoints are clamped (Dirichlet boundary).
    if (t == 0) {
        traj_x_out[i * T + t] = sx;
        traj_y_out[i * T + t] = sy;
        return;
    }
    if (t == T - 1) {
        traj_x_out[i * T + t] = gx;
        traj_y_out[i * T + t] = gy;
        return;
    }

    int idx = i * T + t;
    float x = traj_x_in[idx];
    float y = traj_y_in[idx];

    // (1) Smoothness score: ∇ log p_smooth(τ) ∝ -(τ_i - τ_{i-1}) - (τ_i - τ_{i+1})
    //                                       =  τ_{i-1} - 2τ_i + τ_{i+1}    (Laplacian)
    float xm = traj_x_in[idx - 1], ym = traj_y_in[idx - 1];
    float xp = traj_x_in[idx + 1], yp = traj_y_in[idx + 1];
    float s_smooth_x = xm + xp - 2.0f * x;
    float s_smooth_y = ym + yp - 2.0f * y;

    // (2) Endpoint pull (weaker — endpoints are pinned, but interior is
    //     softly attracted to the linear interpolant).
    float s = static_cast<float>(t) / (T - 1);
    float anchor_x = (1.0f - s) * sx + s * gx;
    float anchor_y = (1.0f - s) * sy + s * gy;
    float s_goal_x = anchor_x - x;
    float s_goal_y = anchor_y - y;

    // (3) Obstacle repulsion: gradient of -W_REPEL * Σ exp(-d²/r_o²)
    float s_rep_x = 0.0f, s_rep_y = 0.0f;
    for (int o = 0; o < c_n_obs; o++) {
        Obstacle ob = c_obs[o];
        float dx = x - ob.cx, dy = y - ob.cy;
        float reff = ob.r + REPEL_RADIUS;
        float inv_r2 = 1.0f / (reff * reff);
        float d2 = dx * dx + dy * dy;
        float w = expf(-d2 * inv_r2);  // bell centered on obstacle
        // gradient: ∇ exp(-d²/r²) = -2/r² * (x - c) * w
        s_rep_x += w * dx * 2.0f * inv_r2;
        s_rep_y += w * dy * 2.0f * inv_r2;
    }

    // Score sum
    float sx_total = W_SMOOTH * s_smooth_x + W_GOAL * s_goal_x + W_REPEL * s_rep_x;
    float sy_total = W_SMOOTH * s_smooth_y + W_GOAL * s_goal_y + W_REPEL * s_rep_y;

    // Langevin update: τ_{k-1} = τ_k + α * score + √(2α) * noise_scale * ε
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, idx + step_index * n_traj * T, 0, &rng);
    float nx = curand_normal(&rng);
    float ny = curand_normal(&rng);
    float kick = sqrtf(2.0f * alpha) * noise_scale;
    traj_x_out[idx] = x + alpha * sx_total + kick * nx;
    traj_y_out[idx] = y + alpha * sy_total + kick * ny;
}

// Per-trajectory cost so we can highlight the best path at the end.
__global__ void cost_kernel(int n_traj, int T,
                            const float* __restrict__ traj_x,
                            const float* __restrict__ traj_y,
                            float* __restrict__ cost) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_traj) return;
    float c = 0.0f;
    for (int t = 0; t < T - 1; t++) {
        float dx = traj_x[i * T + t + 1] - traj_x[i * T + t];
        float dy = traj_y[i * T + t + 1] - traj_y[i * T + t];
        c += dx * dx + dy * dy;
    }
    // Penalty for being inside any obstacle.
    for (int t = 0; t < T; t++) {
        float x = traj_x[i * T + t], y = traj_y[i * T + t];
        for (int o = 0; o < c_n_obs; o++) {
            Obstacle ob = c_obs[o];
            float dx = x - ob.cx, dy = y - ob.cy;
            float d = sqrtf(dx * dx + dy * dy) - ob.r;
            if (d < 0.0f) c += 60.0f * (-d);
        }
    }
    cost[i] = c;
}

// -------------------------------------------------------------------------
// Host helpers
// -------------------------------------------------------------------------
static cv::Point to_px(float x, float y) {
    int px = static_cast<int>((x + WORLD_HALF_X) / (2.0f * WORLD_HALF_X) * PANEL_W);
    int py = static_cast<int>((1.0f - (y + WORLD_HALF_Y) / (2.0f * WORLD_HALF_Y)) * PANEL_H);
    return cv::Point(px, py);
}

static cv::Mat draw(const std::vector<float>& xs,
                    const std::vector<float>& ys,
                    int n_traj, int T,
                    const std::vector<Obstacle>& obs,
                    const std::vector<float>* cost,
                    int best_idx,
                    int step, float alpha, float noise_scale, float ms_step) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(20, 20, 24));

    // obstacles
    for (const auto& o : obs) {
        cv::circle(img, to_px(o.cx, o.cy),
                   static_cast<int>(o.r / (2.0f * WORLD_HALF_X) * PANEL_W),
                   cv::Scalar(40, 40, 80), cv::FILLED);
        cv::circle(img, to_px(o.cx, o.cy),
                   static_cast<int>(o.r / (2.0f * WORLD_HALF_X) * PANEL_W),
                   cv::Scalar(80, 80, 160), 2);
    }

    // trajectories (faint cyan)
    int stride = std::max(1, n_traj / 256);
    cv::Scalar col(180, 180, 60);
    for (int i = 0; i < n_traj; i += stride) {
        for (int t = 0; t < T - 1; t++) {
            cv::line(img,
                     to_px(xs[i * T + t],     ys[i * T + t]),
                     to_px(xs[i * T + t + 1], ys[i * T + t + 1]),
                     col, 1, cv::LINE_AA);
        }
    }
    if (best_idx >= 0) {
        for (int t = 0; t < T - 1; t++) {
            cv::line(img,
                     to_px(xs[best_idx * T + t],     ys[best_idx * T + t]),
                     to_px(xs[best_idx * T + t + 1], ys[best_idx * T + t + 1]),
                     cv::Scalar(80, 240, 80), 3, cv::LINE_AA);
        }
    }
    cv::circle(img, to_px(START_X, START_Y), 7, cv::Scalar(255, 255, 255), cv::FILLED);
    cv::circle(img, to_px(GOAL_X, GOAL_Y), 7, cv::Scalar(60, 60, 240), cv::FILLED);

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "GPU diffusion planner  step %3d / %d   alpha=%.4f   sigma=%.3f   step=%.2f ms",
                  step, N_STEPS, alpha, noise_scale, ms_step);
    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, 30), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(img, buf, cv::Point(10, 21),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    if (cost && best_idx >= 0) {
        std::snprintf(buf, sizeof(buf), "best cost = %.2f", (*cost)[best_idx]);
        cv::putText(img, buf, cv::Point(10, PANEL_H - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 240, 180), 1, cv::LINE_AA);
    }
    return img;
}

static void convert_frames_to_gif(const std::string& frame_pattern, const std::string& gif, int fps) {
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -framerate %d -i %s "
                  "-vf \"fps=%d,scale=900:-1:flags=lanczos,split[a][b];"
                  "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle\" "
                  "%s 2>/dev/null",
                  fps, frame_pattern.c_str(), fps, gif.c_str());
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg failed (%d)\n", rc);
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<Obstacle> obs = {
        { -3.5f,  1.5f, 1.4f},
        { -1.0f, -3.0f, 1.2f},
        {  1.8f,  2.5f, 1.5f},
        {  4.5f, -1.0f, 1.6f},
        {  0.0f,  0.0f, 0.9f},
        {  6.5f,  3.0f, 1.0f},
        { -6.5f, -2.5f, 1.0f},
        {  3.0f,  5.0f, 0.7f},
    };
    int n_obs = static_cast<int>(obs.size());
    CUDA_CHECK(cudaMemcpyToSymbol(c_obs, obs.data(), n_obs * sizeof(Obstacle)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_n_obs, &n_obs, sizeof(int)));

    int n_floats = N_TRAJ * N_WAYPOINTS;
    float *d_x_a, *d_y_a, *d_x_b, *d_y_b, *d_cost;
    CUDA_CHECK(cudaMalloc(&d_x_a, n_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_a, n_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_b, n_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_b, n_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cost, N_TRAJ * sizeof(float)));

    dim3 block(32, 8);
    dim3 grid((N_WAYPOINTS + block.x - 1) / block.x,
              (N_TRAJ + block.y - 1) / block.y);

    init_trajectories_kernel<<<grid, block>>>(N_TRAJ, N_WAYPOINTS,
                                              START_X, START_Y, GOAL_X, GOAL_Y,
                                              NOISE_SCALE_START, 2028ULL,
                                              d_x_a, d_y_a);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_x(n_floats), h_y(n_floats), h_cost(N_TRAJ);

    int frame_setup_rc = std::system(
        "mkdir -p gif tmp/gpu_diffusion_planner_frames && rm -f tmp/gpu_diffusion_planner_frames/frame_*.png");
    if (frame_setup_rc != 0) std::fprintf(stderr, "frame setup failed (%d)\n", frame_setup_rc);
    int frame_id = 0;

    auto frame_now = [&](const float* frame_x, const float* frame_y,
                         int step, float alpha, float ns, float ms,
                         int best_idx, const std::vector<float>* cost) {
        CUDA_CHECK(cudaMemcpy(h_x.data(), frame_x, n_floats * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y.data(), frame_y, n_floats * sizeof(float),
                              cudaMemcpyDeviceToHost));
        cv::Mat img = draw(h_x, h_y, N_TRAJ, N_WAYPOINTS, obs, cost, best_idx,
                           step, alpha, ns, ms);
        char path[256];
        std::snprintf(path, sizeof(path), "tmp/gpu_diffusion_planner_frames/frame_%03d.png", frame_id++);
        cv::imwrite(path, img);
    };

    float* cur_x = d_x_a;
    float* cur_y = d_y_a;
    float* next_x = d_x_b;
    float* next_y = d_y_b;

    // Initial frame
    frame_now(cur_x, cur_y, 0, ALPHA_START, NOISE_SCALE_START, 0.0f, -1, nullptr);

    double total_ms = 0.0;
    int counted = 0;

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    for (int k = 0; k < N_STEPS; k++) {
        // Linearly anneal alpha and noise scale.
        float t01 = static_cast<float>(k) / (N_STEPS - 1);
        float alpha = (1.0f - t01) * ALPHA_START + t01 * ALPHA_END;
        float noise = (1.0f - t01) * NOISE_SCALE_START + t01 * NOISE_SCALE_END;

        CUDA_CHECK(cudaEventRecord(e0));
        langevin_step_kernel<<<grid, block>>>(N_TRAJ, N_WAYPOINTS,
                                              START_X, START_Y, GOAL_X, GOAL_Y,
                                              alpha, noise, 4019ULL, k,
                                              cur_x, cur_y, next_x, next_y);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        // Keep the current trajectory buffer by swapping pointers; no GPU copy.
        std::swap(cur_x, next_x);
        std::swap(cur_y, next_y);

        if (k >= 5) { total_ms += ms; counted++; }
        if (k % 3 == 0 || k == N_STEPS - 1) {
            frame_now(cur_x, cur_y, k + 1, alpha, noise, ms, -1, nullptr);
        }
        if (k % 20 == 0) {
            std::printf("  step %3d  alpha=%.4f  noise=%.3f  %.2f ms\n",
                        k, alpha, noise, ms);
        }
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    // Final selection: lowest-cost trajectory
    cost_kernel<<<(N_TRAJ + 127) / 128, 128>>>(N_TRAJ, N_WAYPOINTS,
                                                cur_x, cur_y, d_cost);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_cost.data(), d_cost, N_TRAJ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    int best = 0;
    float best_c = h_cost[0];
    for (int i = 1; i < N_TRAJ; i++) {
        if (h_cost[i] < best_c) { best_c = h_cost[i]; best = i; }
    }
    std::printf("Best trajectory: idx %d  cost %.3f\n", best, best_c);

    for (int hold = 0; hold < 30; hold++) {
        frame_now(cur_x, cur_y, N_STEPS + hold, ALPHA_END, NOISE_SCALE_END, 0.0f, best, &h_cost);
    }
    if (counted > 0) {
        std::printf("Avg copy-free step time: %.2f ms (%d trajectories x %d waypoints)\n",
                    total_ms / counted, N_TRAJ, N_WAYPOINTS);
    }
    std::printf("Ping-pong buffers avoided %.1f MiB of per-step device copies.\n",
                2.0 * static_cast<double>(n_floats) * sizeof(float) * N_STEPS /
                    (1024.0 * 1024.0));
    convert_frames_to_gif("tmp/gpu_diffusion_planner_frames/frame_%03d.png",
                          "gif/gpu_diffusion_planner.gif", VIDEO_FPS);
    std::printf("GIF saved to gif/gpu_diffusion_planner.gif\n");

    cudaFree(d_x_a); cudaFree(d_y_a);
    cudaFree(d_x_b); cudaFree(d_y_b);
    cudaFree(d_cost);
    return 0;
}
