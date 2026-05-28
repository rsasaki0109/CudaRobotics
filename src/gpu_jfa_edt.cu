// gpu_jfa_edt.cu
//
// GPU Jump Flooding Algorithm (Rong & Tan 2006) — the canonical O(log N)
// GPU-native Voronoi diagram / Euclidean Distance Transform builder.  The map
// is the textbook "one thread = one pixel": each pixel scatters its current
// best seed to a small neighbourhood at exponentially shrinking step sizes,
// and the answer falls out after `log2(N)` sweeps.
//
// What it produces:
//   - Voronoi label per pixel: which seed point is nearest, in 2-norm.
//   - Euclidean distance transform (EDT) per pixel: distance to nearest seed.
//
// Why JFA is the right GPU primitive here:
//   - Brute-force EDT (CPU baseline) is O(W*H*K) — for K seeds that is plenty
//     of work even at modest grid sizes.
//   - JFA does O(W*H * log_2(max(W,H))) and is embarrassingly parallel per
//     pixel, with a fixed, oblivious memory access pattern.
//
// Correctness — honest framing (in contrast to TSDF / MC / DBSCAN)
// ----------------------------------------------------------------
// Plain JFA is *not* bit-identical to brute-force EDT.  Each sweep only
// peeks at 9 candidate sites (self + 8 neighbours at the current step size),
// so on rare configurations a pixel near a Voronoi cell boundary can end up
// assigned to a slightly farther seed than the true argmin.  Empirically the
// disagreement is tiny — typically far below `0.5%` of pixels at `512^2` —
// and is always confined to cell boundaries where two seeds are equidistant.
//
// We therefore report the *agreement distribution* against the brute-force
// reference: percentage of pixels with the identical nearest-seed label, max
// EDT diff, and mean EDT diff.  This is the honest efficiency statement: the
// GPU answer is a tiny-tolerance Voronoi *approximation* whose throughput
// scales as `log N / K` versus the brute force.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ------------------------------------------------------------------ constants
#define WIDTH  512
#define HEIGHT 512
static const int   N_PIX   = WIDTH * HEIGHT;
static const int   N_SEEDS = 96;
static const int   INVALID = -1;

static const int   PANEL_W = 760;
static const int   PANEL_H = 760;

// JFA scatter offsets (self + 8 surrounding sites at step s)
__host__ __device__ static const int OFFX[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
__host__ __device__ static const int OFFY[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};

// ------------------------------------------------------------------ JFA kernel
// One pass of JFA at step size `s`.  Each pixel reads the current best at
// 9 candidate sites and keeps the closest.  Sites are encoded as the seed
// coordinate (sx, sy) — distance is derived on the fly so we never write a
// stale (label, distance) pair.
__global__ static void jfa_pass(const int* __restrict__ in_sx,
                                const int* __restrict__ in_sy,
                                int* __restrict__ out_sx,
                                int* __restrict__ out_sy,
                                int s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    int best_sx = in_sx[y * WIDTH + x];
    int best_sy = in_sy[y * WIDTH + x];
    long long best_d2 = (best_sx == INVALID)
        ? (long long)2147483647
        : (long long)(x - best_sx) * (x - best_sx)
        + (long long)(y - best_sy) * (y - best_sy);

    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        int nx = x + OFFX[k] * s;
        int ny = y + OFFY[k] * s;
        if (nx < 0 || nx >= WIDTH || ny < 0 || ny >= HEIGHT) continue;
        int csx = in_sx[ny * WIDTH + nx];
        if (csx == INVALID) continue;
        int csy = in_sy[ny * WIDTH + nx];
        long long d2 = (long long)(x - csx) * (x - csx)
                     + (long long)(y - csy) * (y - csy);
        if (d2 < best_d2) {
            best_d2 = d2;
            best_sx = csx;
            best_sy = csy;
        }
    }
    out_sx[y * WIDTH + x] = best_sx;
    out_sy[y * WIDTH + x] = best_sy;
}

// ---------------------------------------------------- CPU brute-force EDT
// One pass over every pixel, scanning every seed.  Produces the exact 2-norm
// Voronoi diagram + EDT — the ground-truth reference.
static void brute_edt_cpu(const std::vector<int>& seed_x,
                          const std::vector<int>& seed_y,
                          std::vector<int>& out_label,
                          std::vector<float>& out_dist) {
    out_label.assign(N_PIX, INVALID);
    out_dist.assign(N_PIX, 0.0f);
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            long long best = 2147483647;
            int best_i = -1;
            for (int i = 0; i < N_SEEDS; ++i) {
                long long dx = x - seed_x[i], dy = y - seed_y[i];
                long long d2 = dx * dx + dy * dy;
                if (d2 < best) { best = d2; best_i = i; }
            }
            out_label[y * WIDTH + x] = best_i;
            out_dist [y * WIDTH + x] = std::sqrt((float)best);
        }
    }
}

// -------------------------------------------------- assign Voronoi labels
// Reverse-lookup: from the (sx, sy) site per pixel, find its seed index.
// Seeds are uniquely identified by (sx, sy) — at most one seed per coordinate.
static void labels_from_sites(const std::vector<int>& sx,
                              const std::vector<int>& sy,
                              const std::vector<int>& seed_x,
                              const std::vector<int>& seed_y,
                              std::vector<int>& out_label,
                              std::vector<float>& out_dist) {
    // Build a coord -> seed_idx hash via flat array (W*H is small here).
    std::vector<int> idx_at(N_PIX, -1);
    for (int i = 0; i < N_SEEDS; ++i)
        idx_at[seed_y[i] * WIDTH + seed_x[i]] = i;

    out_label.assign(N_PIX, INVALID);
    out_dist.assign(N_PIX, 0.0f);
    for (int p = 0; p < N_PIX; ++p) {
        if (sx[p] == INVALID) continue;
        int seed_idx = idx_at[sy[p] * WIDTH + sx[p]];
        out_label[p] = seed_idx;
        int x = p % WIDTH, y = p / WIDTH;
        float dx = (float)(x - sx[p]), dy = (float)(y - sy[p]);
        out_dist[p] = std::sqrt(dx * dx + dy * dy);
    }
}

// ------------------------------------------------------------------ rendering
static void render_voronoi(cv::Mat& img,
                           const std::vector<int>& label,
                           const std::vector<float>& dist,
                           const std::vector<int>& seed_x,
                           const std::vector<int>& seed_y,
                           const std::vector<cv::Scalar>& palette,
                           int margin_top) {
    const int H_DRAW = img.rows - margin_top;
    const float sx = (float)img.cols / WIDTH;
    const float sy = (float)H_DRAW   / HEIGHT;
    for (int y = 0; y < img.rows; ++y) {
        uchar* row = img.ptr<uchar>(y);
        for (int x = 0; x < img.cols; ++x) {
            int gx = (int)(x / sx);
            int gy = (int)((y - margin_top) / sy);
            if (gy < 0 || gy >= HEIGHT || gx < 0 || gx >= WIDTH) continue;
            int idx = gy * WIDTH + gx;
            int l = label[idx];
            cv::Scalar c = (l < 0) ? cv::Scalar(40, 40, 50) : palette[l % palette.size()];
            // shade by distance — closer to seed darker (more saturated)
            float d = dist[idx];
            float t = std::min(1.0f, d / 60.0f);
            float k = 0.55f + 0.45f * t;
            row[3 * x + 0] = (uchar)std::min(255.0, c[0] * k);
            row[3 * x + 1] = (uchar)std::min(255.0, c[1] * k);
            row[3 * x + 2] = (uchar)std::min(255.0, c[2] * k);
        }
    }
    // seeds on top
    for (int i = 0; i < (int)seed_x.size(); ++i) {
        int ux = (int)(seed_x[i] * sx);
        int uy = margin_top + (int)(seed_y[i] * sy);
        cv::circle(img, cv::Point(ux, uy), 4, cv::Scalar(20, 20, 30), -1, cv::LINE_AA);
        cv::circle(img, cv::Point(ux, uy), 2, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
    }
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::printf("GPU JFA Voronoi / EDT: %d x %d = %d pixels, %d seeds\n",
                WIDTH, HEIGHT, N_PIX, N_SEEDS);

    // --- seeds --------------------------------------------------------------
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dx(8, WIDTH  - 9);
    std::uniform_int_distribution<int> dy(8, HEIGHT - 9);
    std::vector<int> seed_x(N_SEEDS), seed_y(N_SEEDS);
    for (int i = 0; i < N_SEEDS; ++i) { seed_x[i] = dx(rng); seed_y[i] = dy(rng); }

    // --- CPU brute-force EDT (timed) ---------------------------------------
    std::vector<int>   lbl_cpu;
    std::vector<float> dst_cpu;
    auto t0 = std::chrono::high_resolution_clock::now();
    brute_edt_cpu(seed_x, seed_y, lbl_cpu, dst_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- GPU JFA (timed) ----------------------------------------------------
    std::vector<int> sx_init(N_PIX, INVALID), sy_init(N_PIX, INVALID);
    for (int i = 0; i < N_SEEDS; ++i) {
        sx_init[seed_y[i] * WIDTH + seed_x[i]] = seed_x[i];
        sy_init[seed_y[i] * WIDTH + seed_x[i]] = seed_y[i];
    }

    int *d_sx_a, *d_sy_a, *d_sx_b, *d_sy_b;
    CUDA_CHECK(cudaMalloc(&d_sx_a, N_PIX * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sy_a, N_PIX * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sx_b, N_PIX * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sy_b, N_PIX * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sx_a, sx_init.data(), N_PIX * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sy_a, sy_init.data(), N_PIX * sizeof(int),
                          cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    auto jfa_run = [&](cudaEvent_t* eps, cudaEvent_t* eps_end, int* passes_out) {
        // re-upload seeds for a fresh run
        CUDA_CHECK(cudaMemcpy(d_sx_a, sx_init.data(), N_PIX * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sy_a, sy_init.data(), N_PIX * sizeof(int),
                              cudaMemcpyHostToDevice));
        if (eps) CUDA_CHECK(cudaEventRecord(*eps));
        int s = WIDTH / 2;
        int passes = 0;
        int *in_x = d_sx_a, *in_y = d_sy_a, *out_x = d_sx_b, *out_y = d_sy_b;
        while (s >= 1) {
            jfa_pass<<<grid, block>>>(in_x, in_y, out_x, out_y, s);
            std::swap(in_x, out_x); std::swap(in_y, out_y);
            s /= 2;
            ++passes;
        }
        if (eps_end) CUDA_CHECK(cudaEventRecord(*eps_end));
        // ensure final result is in d_sx_a / d_sy_a (in_x after final swap is the result)
        if (in_x != d_sx_a) {
            CUDA_CHECK(cudaMemcpy(d_sx_a, in_x, N_PIX * sizeof(int), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_sy_a, in_y, N_PIX * sizeof(int), cudaMemcpyDeviceToDevice));
        }
        if (passes_out) *passes_out = passes;
    };

    jfa_run(nullptr, nullptr, nullptr);                  // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    int n_passes = 0;
    jfa_run(&e0, &e1, &n_passes);
    CUDA_CHECK(cudaEventSynchronize(e1));
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));

    std::vector<int> sx_final(N_PIX), sy_final(N_PIX);
    CUDA_CHECK(cudaMemcpy(sx_final.data(), d_sx_a, N_PIX * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sy_final.data(), d_sy_a, N_PIX * sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::vector<int>   lbl_gpu;
    std::vector<float> dst_gpu;
    labels_from_sites(sx_final, sy_final, seed_x, seed_y, lbl_gpu, dst_gpu);

    // --- compare CPU vs GPU -------------------------------------------------
    long long agree = 0;
    double max_d = 0.0, sum_d = 0.0;
    for (int p = 0; p < N_PIX; ++p) {
        if (lbl_cpu[p] == lbl_gpu[p]) ++agree;
        double d = std::fabs((double)dst_cpu[p] - (double)dst_gpu[p]);
        if (d > max_d) max_d = d;
        sum_d += d;
    }
    double speedup = cpu_ms / gpu_ms;
    double agree_pct = 100.0 * (double)agree / N_PIX;
    std::printf("CPU brute EDT %.1f ms,  GPU JFA %.3f ms (%d passes)  -> %.0fx\n",
                cpu_ms, gpu_ms, n_passes, speedup);
    std::printf("Voronoi label agreement: %.4f %% (%lld / %d pixels)\n",
                agree_pct, agree, N_PIX);
    std::printf("EDT max|diff| %.3e px, mean|diff| %.3e px\n",
                max_d, sum_d / N_PIX);

    // --- animation: replay JFA pass-by-pass --------------------------------
    if (system("mkdir -p tmp") != 0)
        std::fprintf(stderr, "warning: mkdir tmp failed\n");
    cv::VideoWriter video("tmp/gpu_jfa_edt.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          6, cv::Size(PANEL_W, PANEL_H));

    // palette
    std::vector<cv::Scalar> palette(64);
    std::mt19937 prng(11);
    for (auto& c : palette) {
        c = cv::Scalar(80 + prng() % 175, 80 + prng() % 175, 80 + prng() % 175);
    }

    // walk through passes on host (so we can render each snapshot)
    std::vector<int> sx_now = sx_init;
    std::vector<int> sy_now = sy_init;
    std::vector<int> sx_next(N_PIX), sy_next(N_PIX);
    auto cpu_step = [&](int s) {
        for (int y = 0; y < HEIGHT; ++y)
            for (int x = 0; x < WIDTH; ++x) {
                int bx = sx_now[y * WIDTH + x], by = sy_now[y * WIDTH + x];
                long long bd2 = (bx == INVALID)
                    ? (long long)2147483647
                    : (long long)(x - bx) * (x - bx) + (long long)(y - by) * (y - by);
                for (int k = 0; k < 9; ++k) {
                    int nx = x + OFFX[k] * s, ny = y + OFFY[k] * s;
                    if (nx < 0 || nx >= WIDTH || ny < 0 || ny >= HEIGHT) continue;
                    int csx = sx_now[ny * WIDTH + nx];
                    if (csx == INVALID) continue;
                    int csy = sy_now[ny * WIDTH + nx];
                    long long d2 = (long long)(x - csx) * (x - csx)
                                 + (long long)(y - csy) * (y - csy);
                    if (d2 < bd2) { bd2 = d2; bx = csx; by = csy; }
                }
                sx_next[y * WIDTH + x] = bx;
                sy_next[y * WIDTH + x] = by;
            }
        sx_now.swap(sx_next);
        sy_now.swap(sy_next);
    };

    auto render_now = [&](int pass_idx, int step) {
        std::vector<int>   lbl;
        std::vector<float> dst;
        labels_from_sites(sx_now, sy_now, seed_x, seed_y, lbl, dst);
        cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(20, 20, 26));
        const int margin_top = 60;
        render_voronoi(img, lbl, dst, seed_x, seed_y, palette, margin_top);
        cv::putText(img, "GPU Jump Flooding (one thread = one pixel)",
                    cv::Point(12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
        char l0[160], l1[160], l2[160];
        std::snprintf(l0, sizeof(l0), "pass %d   step s = %d",
                      pass_idx, step);
        std::snprintf(l1, sizeof(l1),
                      "%d^2 px x %d seeds:  CPU brute %.0f ms vs GPU JFA %.2f ms (%.0fx)",
                      WIDTH, N_SEEDS, cpu_ms, gpu_ms, speedup);
        std::snprintf(l2, sizeof(l2),
                      "label agreement %.4f %%   EDT max|diff| %.2f px",
                      agree_pct, max_d);
        cv::putText(img, l0, cv::Point(12, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(230, 230, 240), 1, cv::LINE_AA);
        cv::putText(img, l1, cv::Point(12, PANEL_H - 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 255, 200), 1, cv::LINE_AA);
        cv::putText(img, l2, cv::Point(12, PANEL_H - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        video.write(img);
    };

    render_now(0, 0);    // initial: just seeds
    int s = WIDTH / 2;
    int pass = 0;
    while (s >= 1) {
        ++pass;
        cpu_step(s);
        render_now(pass, s);
        s /= 2;
    }
    // hold the final frame
    render_now(pass, 1);
    render_now(pass, 1);
    video.release();

    cudabot::avi_to_gif("tmp/gpu_jfa_edt.avi", "gif/gpu_jfa_edt.gif", 6, 640);
    std::printf("wrote gif/gpu_jfa_edt.gif\n");

    CUDA_CHECK(cudaFree(d_sx_a));
    CUDA_CHECK(cudaFree(d_sy_a));
    CUDA_CHECK(cudaFree(d_sx_b));
    CUDA_CHECK(cudaFree(d_sy_b));
    return 0;
}
