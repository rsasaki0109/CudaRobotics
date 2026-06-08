// gpu_sgm_stereo.cu
//
// GPU Semi-Global Matching (SGM) stereo disparity - Hirschmuller (2008), the
// workhorse dense-stereo estimator behind depth cameras and stereo VO
// front-ends. The repo's perception section had dense stereo as a gap.
//
// SGM approximates a global 2D smoothness energy by aggregating a per-pixel
// matching cost along several 1D paths and summing them. The 1D recurrence is
// inherently sequential along a path but every path / scanline is independent,
// so the natural GPU map is:
//
//   one thread = one scanline (per aggregation direction)
//
// Pipeline (CPU and GPU run the SAME integer logic):
//   1. census transform: each pixel -> a 5x5 (24-bit) census descriptor.
//   2. matching cost C(p,d) = Hamming(censusL[p], censusR[p-d])  (popcount).
//   3. path aggregation, 4 directions (L->R, R->L, T->B, B->T):
//        Lr(p,d) = C(p,d) + min( Lr(p-r,d),
//                                Lr(p-r,d-1)+P1, Lr(p-r,d+1)+P1,
//                                min_k Lr(p-r,k)+P2 ) - min_k Lr(p-r,k)
//      accumulate S(p,d) += Lr(p,d) over all directions.
//   4. winner-take-all: disparity(p) = argmin_d S(p,d).
//
// Everything is integer arithmetic with deterministic tie-breaks (min over
// values; argmin scans d ascending with strict <), so the CPU and GPU disparity
// maps are bit-identical. We also score the result against the synthetic
// ground-truth disparity - an honest accuracy number for the algorithm itself
// (errors concentrate at occlusion boundaries, as expected for 4-path SGM).

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ----------------------------------------------------------------- constants
#define IMG_W   384
#define IMG_H   256
#define D_RANGE 64
static const int   CR         = 2;
static const int   CENSUS_MAX = (2 * CR + 1) * (2 * CR + 1) - 1;
static const int   P1         = 7;
static const int   P2         = 86;
static const int   N_CELLS    = IMG_W * IMG_H;
static const int   INF        = 1 << 28;

static const int   PANEL_W    = 760;
static const int   PANEL_H    = 300;

__host__ __device__ static inline int idx2(int x, int y) { return y * IMG_W + x; }
__host__ __device__ static inline int idx3(int x, int y, int d) {
    return (y * IMG_W + x) * D_RANGE + d;
}

// ----------------------------------------------------- synthetic stereo pair
static inline float hash01(int x, int y) {
    unsigned int h = ((unsigned int)x * 73856093u) ^ ((unsigned int)y * 19349663u);
    h = (h ^ (h >> 13)) * 1274126177u;
    return (float)((h ^ (h >> 16)) & 0xffff) / 65535.0f;
}
static inline int tex(int s, int y) {
    float v = 128.0f
            + 100.0f * (hash01(s, y) - 0.5f)
            + 22.0f * std::sin(0.30f * s + 0.20f * y)
            + 18.0f * std::sin(0.13f * s - 0.31f * y);
    if (v < 0) v = 0; if (v > 255) v = 255;
    return (int)(v + 0.5f);
}
static inline int bg_disp(int x, int y) { return 6 + (y * 10) / IMG_H; }
static inline int gt_disp(int x, int y) {
    int d = bg_disp(x, y);
    if (x >= 80 && x < 180 && y >= 50 && y < 175) d = std::max(d, 40);
    int cx = 280, cy = 110, r = 52;
    if ((x - cx) * (x - cx) + (y - cy) * (y - cy) < r * r) d = std::max(d, 54);
    if (x >= 150 && x < 330 && y >= 185 && y < 244) d = std::max(d, 28);
    return d;
}

static void make_pair(std::vector<uint8_t>& L, std::vector<uint8_t>& R,
                      std::vector<int>& gt) {
    L.assign(N_CELLS, 0);
    R.assign(N_CELLS, 0);
    gt.assign(N_CELLS, 0);
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
            L[idx2(x, y)] = (uint8_t)tex(x, y);
            gt[idx2(x, y)] = gt_disp(x, y);
        }
    for (int y = 0; y < IMG_H; ++y) {
        std::vector<int> zbuf(IMG_W, -1);
        for (int x = 0; x < IMG_W; ++x) {
            int d  = gt_disp(x, y);
            int xr = x - d;
            if (xr >= 0 && xr < IMG_W && d > zbuf[xr]) {
                zbuf[xr] = d;
                R[idx2(xr, y)] = (uint8_t)tex(x, y);
            }
        }
        for (int xr = 0; xr < IMG_W; ++xr)
            if (zbuf[xr] < 0) {
                int d = bg_disp(xr, y);
                R[idx2(xr, y)] = (uint8_t)tex(xr + d, y);
            }
    }
}

// --------------------------------------------------------------- census cost
__host__ __device__ static inline unsigned int census_at(const uint8_t* img,
                                                         int x, int y) {
    int c = img[idx2(x, y)];
    unsigned int bits = 0; int k = 0;
    for (int dy = -CR; dy <= CR; ++dy)
        for (int dx = -CR; dx <= CR; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx, ny = y + dy;
            if (nx < 0) nx = 0; if (nx >= IMG_W) nx = IMG_W - 1;
            if (ny < 0) ny = 0; if (ny >= IMG_H) ny = IMG_H - 1;
            if (img[idx2(nx, ny)] < c) bits |= (1u << k);
            ++k;
        }
    return bits;
}

__host__ __device__ static inline int popcount_u(unsigned int v) {
#ifdef __CUDA_ARCH__
    return __popc(v);
#else
    return __builtin_popcount(v);
#endif
}

__host__ __device__ static inline int cost_at(const unsigned int* cL,
                                             const unsigned int* cR,
                                             int x, int y, int d) {
    int xr = x - d;
    if (xr < 0) return CENSUS_MAX;
    return popcount_u(cL[idx2(x, y)] ^ cR[idx2(xr, y)]);
}

__global__ void census_kernel(const uint8_t* L, const uint8_t* R,
                              unsigned int* cL, unsigned int* cR) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_CELLS) return;
    int x = i % IMG_W, y = i / IMG_W;
    cL[i] = census_at(L, x, y);
    cR[i] = census_at(R, x, y);
}

// dir: 0 = L->R, 1 = R->L, 2 = T->B, 3 = B->T.
__host__ __device__ static inline void aggregate_line(
        const unsigned int* cL, const unsigned int* cR, int* S, int dir, int line) {
    int Lprev[D_RANGE], Lcur[D_RANGE];
    int sx, sy, stepx, stepy, n;
    if (dir == 0)      { sx = 0;         sy = line; stepx = 1;  stepy = 0;  n = IMG_W; }
    else if (dir == 1) { sx = IMG_W - 1; sy = line; stepx = -1; stepy = 0;  n = IMG_W; }
    else if (dir == 2) { sx = line;      sy = 0;    stepx = 0;  stepy = 1;  n = IMG_H; }
    else               { sx = line;      sy = IMG_H - 1; stepx = 0; stepy = -1; n = IMG_H; }

    int x = sx, y = sy, prevMin = 0;
    for (int step = 0; step < n; ++step) {
        int curMin = INF;
        if (step == 0) {
            for (int d = 0; d < D_RANGE; ++d) {
                int v = cost_at(cL, cR, x, y, d);
                Lcur[d] = v;
                if (v < curMin) curMin = v;
            }
        } else {
            for (int d = 0; d < D_RANGE; ++d) {
                int best = Lprev[d];
                int a = (d > 0)           ? Lprev[d - 1] + P1 : INF;
                int b = (d < D_RANGE - 1) ? Lprev[d + 1] + P1 : INF;
                int cc = prevMin + P2;
                if (a < best) best = a;
                if (b < best) best = b;
                if (cc < best) best = cc;
                int v = cost_at(cL, cR, x, y, d) + best - prevMin;
                Lcur[d] = v;
                if (v < curMin) curMin = v;
            }
        }
#ifdef __CUDA_ARCH__
        for (int d = 0; d < D_RANGE; ++d) atomicAdd(&S[idx3(x, y, d)], Lcur[d]);
#else
        for (int d = 0; d < D_RANGE; ++d) S[idx3(x, y, d)] += Lcur[d];
#endif
        for (int d = 0; d < D_RANGE; ++d) Lprev[d] = Lcur[d];
        prevMin = curMin;
        x += stepx; y += stepy;
    }
}

__global__ void aggregate_kernel(const unsigned int* cL, const unsigned int* cR,
                                 int* S, int dir, int n_lines) {
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= n_lines) return;
    aggregate_line(cL, cR, S, dir, line);
}

__global__ void wta_kernel(const int* S, int* disp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_CELLS) return;
    int x = i % IMG_W, y = i / IMG_W;
    int best = INF, bd = 0;
    for (int d = 0; d < D_RANGE; ++d) {
        int v = S[idx3(x, y, d)];
        if (v < best) { best = v; bd = d; }
    }
    disp[i] = bd;
}

// --------------------------------------------------------------- CPU pipeline
static void sgm_cpu(const std::vector<uint8_t>& L, const std::vector<uint8_t>& R,
                    std::vector<int>& disp) {
    std::vector<unsigned int> cL(N_CELLS), cR(N_CELLS);
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
            cL[idx2(x, y)] = census_at(L.data(), x, y);
            cR[idx2(x, y)] = census_at(R.data(), x, y);
        }
    std::vector<int> S(N_CELLS * D_RANGE, 0);
    for (int dir = 0; dir < 4; ++dir) {
        int n_lines = (dir < 2) ? IMG_H : IMG_W;
        for (int line = 0; line < n_lines; ++line)
            aggregate_line(cL.data(), cR.data(), S.data(), dir, line);
    }
    disp.assign(N_CELLS, 0);
    for (int i = 0; i < N_CELLS; ++i) {
        int x = i % IMG_W, y = i / IMG_W, best = INF, bd = 0;
        for (int d = 0; d < D_RANGE; ++d) {
            int v = S[idx3(x, y, d)];
            if (v < best) { best = v; bd = d; }
        }
        disp[i] = bd;
    }
}

// ------------------------------------------------------------- visualisation
static cv::Mat colorize(const std::vector<int>& disp) {
    cv::Mat g(IMG_H, IMG_W, CV_8UC1);
    for (int i = 0; i < N_CELLS; ++i)
        g.data[i] = (uint8_t)std::min(255, disp[i] * 255 / (D_RANGE - 1));
    cv::Mat c; cv::applyColorMap(g, c, cv::COLORMAP_JET);
    return c;
}

static void draw_panel(cv::Mat& out, const cv::Mat& left_gray,
                       const cv::Mat& gt_col, const cv::Mat& res_col,
                       const char* l1, const char* l2, const char* l3) {
    out = cv::Mat(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(28, 28, 32));
    const int IW = 240, IH = 160, Y = 60;
    int xs[3] = {15, 265, 515};
    cv::Mat a, b, c;
    cv::resize(left_gray, a, cv::Size(IW, IH), 0, 0, cv::INTER_AREA);
    cv::resize(gt_col,    b, cv::Size(IW, IH), 0, 0, cv::INTER_NEAREST);
    cv::resize(res_col,   c, cv::Size(IW, IH), 0, 0, cv::INTER_NEAREST);
    a.copyTo(out(cv::Rect(xs[0], Y, IW, IH)));
    b.copyTo(out(cv::Rect(xs[1], Y, IW, IH)));
    c.copyTo(out(cv::Rect(xs[2], Y, IW, IH)));
    const char* lab[3] = {"left image", "ground-truth disparity", "SGM disparity"};
    for (int k = 0; k < 3; ++k)
        cv::putText(out, lab[k], {xs[k], Y - 8}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    {220, 220, 220}, 1, cv::LINE_AA);
    cv::putText(out, l1, {15, 26}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {235, 235, 235}, 1, cv::LINE_AA);
    cv::putText(out, l2, {15, PANEL_H - 34}, cv::FONT_HERSHEY_SIMPLEX, 0.46, {180, 220, 255}, 1, cv::LINE_AA);
    cv::putText(out, l3, {15, PANEL_H - 12}, cv::FONT_HERSHEY_SIMPLEX, 0.46, {180, 255, 180}, 1, cv::LINE_AA);
}

// ===========================================================================
int main() {
    std::vector<uint8_t> L, R;
    std::vector<int> gt;
    make_pair(L, R, gt);

    std::vector<int> disp_cpu;
    auto t0 = std::chrono::high_resolution_clock::now();
    sgm_cpu(L, R, disp_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    uint8_t *d_L, *d_R;
    unsigned int *d_cL, *d_cR;
    int *d_S, *d_disp;
    CUDA_CHECK(cudaMalloc(&d_L,  N_CELLS));
    CUDA_CHECK(cudaMalloc(&d_R,  N_CELLS));
    CUDA_CHECK(cudaMalloc(&d_cL, N_CELLS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_cR, N_CELLS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_S,  (size_t)N_CELLS * D_RANGE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_disp, N_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_L, L.data(), N_CELLS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_R, R.data(), N_CELLS, cudaMemcpyHostToDevice));

    int blk = 128, grid_px = (N_CELLS + blk - 1) / blk;
    auto run_gpu = [&](float* ms_out) {
        cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        census_kernel<<<grid_px, blk>>>(d_L, d_R, d_cL, d_cR);
        CUDA_CHECK(cudaMemset(d_S, 0, (size_t)N_CELLS * D_RANGE * sizeof(int)));
        for (int dir = 0; dir < 4; ++dir) {
            int n_lines = (dir < 2) ? IMG_H : IMG_W;
            int g = (n_lines + 63) / 64;
            aggregate_kernel<<<g, 64>>>(d_cL, d_cR, d_S, dir, n_lines);
        }
        wta_kernel<<<grid_px, blk>>>(d_S, d_disp);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        if (ms_out) CUDA_CHECK(cudaEventElapsedTime(ms_out, e0, e1));
    };
    float gpu_ms = 0.0f;
    run_gpu(nullptr);
    run_gpu(&gpu_ms);

    std::vector<int> disp_gpu(N_CELLS);
    CUDA_CHECK(cudaMemcpy(disp_gpu.data(), d_disp, N_CELLS * sizeof(int),
                          cudaMemcpyDeviceToHost));

    int mism = 0, maxdiff = 0;
    for (int i = 0; i < N_CELLS; ++i) {
        int dd = std::abs(disp_cpu[i] - disp_gpu[i]);
        if (dd) ++mism;
        if (dd > maxdiff) maxdiff = dd;
    }
    long long err_sum = 0; int within1 = 0, within2 = 0;
    for (int i = 0; i < N_CELLS; ++i) {
        int e = std::abs(disp_gpu[i] - gt[i]);
        err_sum += e;
        if (e <= 1) ++within1;
        if (e <= 2) ++within2;
    }
    double mae = (double)err_sum / N_CELLS;
    double pct1 = 100.0 * within1 / N_CELLS;
    double pct2 = 100.0 * within2 / N_CELLS;
    double speedup = cpu_ms / gpu_ms;

    std::printf("CPU %.2f ms, GPU %.3f ms  -> %.0fx\n", cpu_ms, gpu_ms, speedup);
    std::printf("CPU vs GPU disparity: mismatches %d / %d, max|diff| %d\n",
                mism, N_CELLS, maxdiff);
    std::printf("vs ground truth: MAE %.3f px, within 1px %.1f%%, within 2px %.1f%%\n",
                mae, pct1, pct2);

    if (system("mkdir -p tmp") != 0) std::fprintf(stderr, "warning: mkdir tmp failed\n");
    cv::VideoWriter video("tmp/gpu_sgm_stereo.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          2, cv::Size(PANEL_W, PANEL_H));
    cv::Mat left_gray(IMG_H, IMG_W, CV_8UC1);
    for (int i = 0; i < N_CELLS; ++i) left_gray.data[i] = L[i];
    cv::cvtColor(left_gray, left_gray, cv::COLOR_GRAY2BGR);
    cv::Mat gt_col = colorize(gt);

    const char* dnames[4] = {"L->R", "R->L", "T->B", "B->T"};
    CUDA_CHECK(cudaMemset(d_S, 0, (size_t)N_CELLS * D_RANGE * sizeof(int)));
    census_kernel<<<grid_px, blk>>>(d_L, d_R, d_cL, d_cR);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int dir = 0; dir < 4; ++dir) {
        int n_lines = (dir < 2) ? IMG_H : IMG_W;
        int g = (n_lines + 63) / 64;
        aggregate_kernel<<<g, 64>>>(d_cL, d_cR, d_S, dir, n_lines);
        wta_kernel<<<grid_px, blk>>>(d_S, d_disp);
        CUDA_CHECK(cudaMemcpy(disp_gpu.data(), d_disp, N_CELLS * sizeof(int),
                              cudaMemcpyDeviceToHost));
        long long es = 0; int w1 = 0;
        for (int i = 0; i < N_CELLS; ++i) { int e = std::abs(disp_gpu[i] - gt[i]); es += e; if (e <= 1) ++w1; }
        cv::Mat res = colorize(disp_gpu);
        char l1[200], l2[200], l3[200];
        std::snprintf(l1, sizeof(l1),
                      "GPU Semi-Global Matching stereo (one thread = one scanline)  "
                      "%dx%d  D=%d  census 5x5", IMG_W, IMG_H, D_RANGE);
        std::snprintf(l2, sizeof(l2),
                      "paths aggregated: %d/4 (added %s)   MAE %.2f px   within 1px %.1f%%",
                      dir + 1, dnames[dir], (double)es / N_CELLS, 100.0 * w1 / N_CELLS);
        std::snprintf(l3, sizeof(l3),
                      "CPU %.0f ms vs GPU %.2f ms -> %.0fx   CPU/GPU bit-identical (max|diff| %d)",
                      cpu_ms, gpu_ms, speedup, maxdiff);
        cv::Mat img; draw_panel(img, left_gray, gt_col, res, l1, l2, l3);
        for (int r = 0; r < (dir == 3 ? 4 : 2); ++r) video.write(img);
    }
    video.release();
    cudabot::avi_to_gif("tmp/gpu_sgm_stereo.avi", "gif/gpu_sgm_stereo.gif", 2, 760);
    std::printf("wrote gif/gpu_sgm_stereo.gif\n");

    CUDA_CHECK(cudaFree(d_L));   CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_cL));  CUDA_CHECK(cudaFree(d_cR));
    CUDA_CHECK(cudaFree(d_S));   CUDA_CHECK(cudaFree(d_disp));
    return 0;
}

}  // namespace cudabot

int main() { return cudabot::main(); }
