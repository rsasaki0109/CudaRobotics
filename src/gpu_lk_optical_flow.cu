// gpu_lk_optical_flow.cu
//
// GPU pyramidal Lucas-Kanade optical flow — the workhorse sparse-flow
// estimator behind KLT trackers, monocular VO front-ends, and feature-based
// scene-flow pipelines.  The map onto the canonical GPU idiom is:
// **one thread = one feature** — each LK feature is an independent Gauss-
// Newton problem over a small image window.
//
// What the demo does
// ------------------
// 1. Build a synthetic image I (mixture of analytic textures) and a known
//    smooth flow field (divergence + rotation).
// 2. Warp I by that flow to produce J.
// 3. Scatter `1024` features on a regular grid.
// 4. Run pyramidal LK from I to J on both CPU and GPU; compare estimated
//    flow per feature.
//
// Correctness — deterministic by construction
// -------------------------------------------
// Per feature, LK is a fixed-iteration Gauss-Newton step with no data-
// dependent branches that fork into different answers (the only branch is
// the determinant-floor skip, which is bit-identical between CPU and GPU
// under `--fmad=false`).  Both paths use the same bilinear sampler, the
// same gradient kernel, and the same 2×2 inversion — so the per-feature flow
// estimates are bit-identical to round-off.

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
#define IMG_W       256
#define IMG_H       256
static const int   N_LEVELS  = 3;                 // 256 -> 128 -> 64
static const int   WIN       = 9;                 // window size (odd)
static const int   HALF      = WIN / 2;           // 4
static const int   N_FEAT    = 32 * 32;           // 1024 features on a grid
static const int   N_ITERS   = 8;                 // LK iterations per level
static const float DET_FLOOR = 1e-3f;             // Hessian determinant floor

static const int   PANEL_W   = 760;
static const int   PANEL_H   = 600;

// ---------------------------------------------------- synthetic image source
// Smooth, textured scene — three blobs of different frequencies plus a
// low-frequency background gradient.  Anything with rich gradients works.
__host__ __device__ static inline float scene(float x, float y) {
    float fx = x - 0.5f, fy = y - 0.5f;
    float r1 = sqrtf((x - 0.30f) * (x - 0.30f) + (y - 0.35f) * (y - 0.35f));
    float r2 = sqrtf((x - 0.68f) * (x - 0.68f) + (y - 0.55f) * (y - 0.55f));
    float r3 = sqrtf((x - 0.45f) * (x - 0.45f) + (y - 0.78f) * (y - 0.78f));
    float v = 0.45f
            + 0.18f * sinf(40.0f * r1)
            + 0.14f * sinf(58.0f * r2)
            + 0.10f * sinf(72.0f * r3)
            + 0.06f * fx + 0.04f * fy;
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return v;
}

// ---------------------------------------------------- ground-truth flow field
// Smooth combination of a uniform translation plus a small rotation about the
// image centre.  Maximum magnitude ~ 4 px.
__host__ __device__ static inline void gt_flow(float x, float y,
                                               float& fx, float& fy) {
    float dx = x - 0.5f, dy = y - 0.5f;
    float ux = 0.012f, uy = 0.006f;            // uniform translation
    float th = 0.040f;                         // rotation angle (rad)
    float rx = std::cos(th) * dx - std::sin(th) * dy - dx;
    float ry = std::sin(th) * dx + std::cos(th) * dy - dy;
    fx = ux + rx;
    fy = uy + ry;
}

// ---------------------------------------------------- bilinear sampler
__host__ __device__ static inline float sample(const float* img, int W, int H,
                                               float x, float y) {
    if (x < 0.0f || y < 0.0f || x > W - 1.0f || y > H - 1.0f) return 0.0f;
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    int x1 = x0 + 1, y1 = y0 + 1;
    if (x1 >= W) x1 = W - 1;
    if (y1 >= H) y1 = H - 1;
    float tx = x - x0, ty = y - y0;
    float a = img[y0 * W + x0], b = img[y0 * W + x1];
    float c = img[y1 * W + x0], d = img[y1 * W + x1];
    float u = a + tx * (b - a);
    float v = c + tx * (d - c);
    return u + ty * (v - u);
}

// ---------------------------------------------------- pyramid downsample (2x2)
__host__ __device__ static inline void box_2x(const float* in, int W, int H,
                                              float* out) {
    int Wo = W / 2, Ho = H / 2;
    for (int y = 0; y < Ho; ++y) {
        for (int x = 0; x < Wo; ++x) {
            float a = in[(2*y + 0) * W + (2*x + 0)];
            float b = in[(2*y + 0) * W + (2*x + 1)];
            float c = in[(2*y + 1) * W + (2*x + 0)];
            float d = in[(2*y + 1) * W + (2*x + 1)];
            out[y * Wo + x] = 0.25f * (a + b + c + d);
        }
    }
}

// ---------------------------------------------------- per-feature LK (shared)
// Process one feature at pyramid level `lvl` starting from `(px, py)` (in
// level-0 coordinates) with an initial guess `(dx, dy)` (also in level-0
// coords).  Refines (dx, dy) for `iters` Gauss-Newton steps using the
// `(I_lvl, J_lvl)` images at this level.
//
// Returned (dx, dy) is the refined level-0 flow.
__host__ __device__ static inline bool lk_feature_level(
        const float* I_lvl, const float* J_lvl, int Wl, int Hl,
        int lvl,                                      // pyramid level index
        float px0, float py0,                         // feature position (lvl 0)
        float& dx, float& dy, int iters) {
    float scale = 1.0f / (float)(1 << lvl);
    float fx = px0 * scale, fy = py0 * scale;         // feature pos at this lvl
    float ddx = dx * scale, ddy = dy * scale;         // initial guess at lvl

    // Precompute G (2x2 Hessian on I) and the per-pixel I value cache.
    // Standard inverse-compositional approximation: gradients on the reference
    // (I) only, so G is built once per level.
    float G00 = 0.f, G01 = 0.f, G11 = 0.f;
    float I_val [WIN * WIN];
    float Ix_val[WIN * WIN];
    float Iy_val[WIN * WIN];
    for (int dyk = -HALF; dyk <= HALF; ++dyk) {
        for (int dxk = -HALF; dxk <= HALF; ++dxk) {
            float sx = fx + dxk;
            float sy = fy + dyk;
            float Iv  = sample(I_lvl, Wl, Hl, sx, sy);
            float Ixv = 0.5f * (sample(I_lvl, Wl, Hl, sx + 1.0f, sy)
                              - sample(I_lvl, Wl, Hl, sx - 1.0f, sy));
            float Iyv = 0.5f * (sample(I_lvl, Wl, Hl, sx, sy + 1.0f)
                              - sample(I_lvl, Wl, Hl, sx, sy - 1.0f));
            int idx = (dyk + HALF) * WIN + (dxk + HALF);
            I_val [idx] = Iv;
            Ix_val[idx] = Ixv;
            Iy_val[idx] = Iyv;
            G00 += Ixv * Ixv;
            G01 += Ixv * Iyv;
            G11 += Iyv * Iyv;
        }
    }
    float det = G00 * G11 - G01 * G01;
    if (det < DET_FLOOR) return false;
    float inv_det = 1.0f / det;

    // Gauss-Newton iterations.
    for (int it = 0; it < iters; ++it) {
        float b0 = 0.f, b1 = 0.f;
        int k = 0;
        for (int dyk = -HALF; dyk <= HALF; ++dyk) {
            for (int dxk = -HALF; dxk <= HALF; ++dxk) {
                float Jv = sample(J_lvl, Wl, Hl, fx + dxk + ddx, fy + dyk + ddy);
                float r  = I_val[k] - Jv;
                b0 += Ix_val[k] * r;
                b1 += Iy_val[k] * r;
                ++k;
            }
        }
        // Δ = G^{-1} b
        float delta_x =  ( G11 * b0 - G01 * b1) * inv_det;
        float delta_y =  (-G01 * b0 + G00 * b1) * inv_det;
        ddx += delta_x;
        ddy += delta_y;
    }
    dx = ddx / scale;
    dy = ddy / scale;
    return true;
}

// ---------------------------------------------------- multi-level orchestration
__host__ __device__ static inline void lk_feature(
        const float* const* I_pyr, const float* const* J_pyr,
        const int* W_pyr, const int* H_pyr,
        float px0, float py0, float& dx, float& dy, int& valid) {
    dx = 0.0f; dy = 0.0f;
    valid = 1;
    // Coarse to fine.
    for (int lvl = N_LEVELS - 1; lvl >= 0; --lvl) {
        bool ok = lk_feature_level(I_pyr[lvl], J_pyr[lvl],
                                   W_pyr[lvl], H_pyr[lvl], lvl,
                                   px0, py0, dx, dy, N_ITERS);
        if (!ok) { valid = 0; }
    }
}

__global__ static void lk_kernel(const float* I0, const float* I1, const float* I2,
                                 const float* J0, const float* J1, const float* J2,
                                 int W0, int H0, int W1, int H1, int W2, int H2,
                                 const float* feat_x, const float* feat_y,
                                 float* out_dx, float* out_dy, int* out_valid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_FEAT) return;
    const float* Ip[3] = {I0, I1, I2};
    const float* Jp[3] = {J0, J1, J2};
    int Wp[3] = {W0, W1, W2};
    int Hp[3] = {H0, H1, H2};
    float dx, dy;
    int valid;
    lk_feature(Ip, Jp, Wp, Hp, feat_x[i], feat_y[i], dx, dy, valid);
    out_dx   [i] = dx;
    out_dy   [i] = dy;
    out_valid[i] = valid;
}

static void lk_cpu(const float* const* Ip, const float* const* Jp,
                   const int* Wp, const int* Hp,
                   const std::vector<float>& fx, const std::vector<float>& fy,
                   std::vector<float>& out_dx, std::vector<float>& out_dy,
                   std::vector<int>& out_valid) {
    for (int i = 0; i < N_FEAT; ++i) {
        float dx, dy;
        int valid;
        lk_feature(Ip, Jp, Wp, Hp, fx[i], fy[i], dx, dy, valid);
        out_dx   [i] = dx;
        out_dy   [i] = dy;
        out_valid[i] = valid;
    }
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::printf("GPU pyramidal Lucas-Kanade optical flow: %d x %d image, "
                "%d features, %d levels, %d iters/level\n",
                IMG_W, IMG_H, N_FEAT, N_LEVELS, N_ITERS);

    // --- build I and J ------------------------------------------------------
    std::vector<float> I0(IMG_W * IMG_H), J0(IMG_W * IMG_H);
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
            float u = (x + 0.5f) / IMG_W;
            float v = (y + 0.5f) / IMG_H;
            I0[y * IMG_W + x] = scene(u, v);
        }
    // J is I warped backward by ground-truth flow.
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
            float u = (x + 0.5f) / IMG_W;
            float v = (y + 0.5f) / IMG_H;
            float fxn, fyn;
            gt_flow(u, v, fxn, fyn);
            float sx = (u + fxn) * IMG_W - 0.5f;
            float sy = (v + fyn) * IMG_H - 0.5f;
            J0[y * IMG_W + x] = sample(I0.data(), IMG_W, IMG_H, sx, sy);
        }

    // --- pyramids -----------------------------------------------------------
    std::vector<std::vector<float>> Ipy(N_LEVELS), Jpy(N_LEVELS);
    Ipy[0] = I0; Jpy[0] = J0;
    int W = IMG_W, H = IMG_H;
    int Wp[N_LEVELS], Hp[N_LEVELS];
    Wp[0] = W; Hp[0] = H;
    for (int lvl = 1; lvl < N_LEVELS; ++lvl) {
        int Wn = W / 2, Hn = H / 2;
        Ipy[lvl].assign(Wn * Hn, 0.0f);
        Jpy[lvl].assign(Wn * Hn, 0.0f);
        box_2x(Ipy[lvl - 1].data(), W, H, Ipy[lvl].data());
        box_2x(Jpy[lvl - 1].data(), W, H, Jpy[lvl].data());
        W = Wn; H = Hn;
        Wp[lvl] = W; Hp[lvl] = H;
    }

    // --- feature points (regular grid, jittered into image area) -----------
    std::vector<float> fx(N_FEAT), fy(N_FEAT);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> jit(-1.0f, 1.0f);
    int side = 32;                             // 32 x 32 = N_FEAT
    int margin = WIN;
    for (int gy = 0; gy < side; ++gy)
        for (int gx = 0; gx < side; ++gx) {
            float u = (gx + 0.5f) / side;
            float v = (gy + 0.5f) / side;
            float xi = margin + u * (IMG_W - 2 * margin) + 0.3f * jit(rng);
            float yi = margin + v * (IMG_H - 2 * margin) + 0.3f * jit(rng);
            fx[gy * side + gx] = xi;
            fy[gy * side + gx] = yi;
        }

    // --- CPU LK (timed) -----------------------------------------------------
    std::vector<const float*> Ip_ptr(N_LEVELS), Jp_ptr(N_LEVELS);
    for (int l = 0; l < N_LEVELS; ++l) { Ip_ptr[l] = Ipy[l].data(); Jp_ptr[l] = Jpy[l].data(); }

    std::vector<float> dx_cpu(N_FEAT), dy_cpu(N_FEAT);
    std::vector<int>   vd_cpu(N_FEAT);
    auto t0 = std::chrono::high_resolution_clock::now();
    lk_cpu(Ip_ptr.data(), Jp_ptr.data(), Wp, Hp, fx, fy,
           dx_cpu, dy_cpu, vd_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- GPU LK (timed) -----------------------------------------------------
    float *dI[N_LEVELS], *dJ[N_LEVELS];
    for (int l = 0; l < N_LEVELS; ++l) {
        size_t bytes = Wp[l] * Hp[l] * sizeof(float);
        CUDA_CHECK(cudaMalloc(&dI[l], bytes));
        CUDA_CHECK(cudaMalloc(&dJ[l], bytes));
        CUDA_CHECK(cudaMemcpy(dI[l], Ipy[l].data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dJ[l], Jpy[l].data(), bytes, cudaMemcpyHostToDevice));
    }
    float *d_fx, *d_fy, *d_dx, *d_dy;
    int   *d_vd;
    CUDA_CHECK(cudaMalloc(&d_fx, N_FEAT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fy, N_FEAT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx, N_FEAT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dy, N_FEAT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vd, N_FEAT * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_fx, fx.data(), N_FEAT * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fy, fy.data(), N_FEAT * sizeof(float), cudaMemcpyHostToDevice));

    int block = 64, grid = (N_FEAT + block - 1) / block;
    lk_kernel<<<grid, block>>>(dI[0], dI[1], dI[2], dJ[0], dJ[1], dJ[2],
                               Wp[0], Hp[0], Wp[1], Hp[1], Wp[2], Hp[2],
                               d_fx, d_fy, d_dx, d_dy, d_vd);   // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    lk_kernel<<<grid, block>>>(dI[0], dI[1], dI[2], dJ[0], dJ[1], dJ[2],
                               Wp[0], Hp[0], Wp[1], Hp[1], Wp[2], Hp[2],
                               d_fx, d_fy, d_dx, d_dy, d_vd);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));

    std::vector<float> dx_gpu(N_FEAT), dy_gpu(N_FEAT);
    std::vector<int>   vd_gpu(N_FEAT);
    CUDA_CHECK(cudaMemcpy(dx_gpu.data(), d_dx, N_FEAT * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dy_gpu.data(), d_dy, N_FEAT * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vd_gpu.data(), d_vd, N_FEAT * sizeof(int),   cudaMemcpyDeviceToHost));

    // --- compare CPU vs GPU -------------------------------------------------
    double max_diff = 0.0, sum_diff = 0.0;
    int    n_valid_cpu = 0, n_valid_gpu = 0;
    int    valid_mismatch = 0;
    double sum_endpoint = 0.0;
    for (int i = 0; i < N_FEAT; ++i) {
        if (vd_cpu[i]) ++n_valid_cpu;
        if (vd_gpu[i]) ++n_valid_gpu;
        if (vd_cpu[i] != vd_gpu[i]) ++valid_mismatch;
        double ex = (double)dx_cpu[i] - (double)dx_gpu[i];
        double ey = (double)dy_cpu[i] - (double)dy_gpu[i];
        double e  = std::sqrt(ex * ex + ey * ey);
        if (e > max_diff) max_diff = e;
        sum_diff += e;
        // ground truth endpoint.  J was built as J(p) = I(p + Δ), so LK flow
        // satisfies d = −Δ (look at p + d in J to find I(p)).  Compare against
        // −gt_flow in pixel units.
        float u = (fx[i] + 0.5f) / IMG_W;
        float v = (fy[i] + 0.5f) / IMG_H;
        float gtx, gty;
        gt_flow(u, v, gtx, gty);
        float gpx = -gtx * IMG_W, gpy = -gty * IMG_H;
        float ex2 = dx_gpu[i] - gpx, ey2 = dy_gpu[i] - gpy;
        sum_endpoint += std::sqrt(ex2 * ex2 + ey2 * ey2);
    }
    double mean_endpoint = sum_endpoint / N_FEAT;
    double speedup = cpu_ms / gpu_ms;
    std::printf("CPU LK %.1f ms,  GPU LK %.3f ms  -> %.0fx\n",
                cpu_ms, gpu_ms, speedup);
    std::printf("valid features: CPU %d / %d,  GPU %d / %d  (valid mismatch %d)\n",
                n_valid_cpu, N_FEAT, n_valid_gpu, N_FEAT, valid_mismatch);
    std::printf("CPU/GPU flow max|diff| %.3e px,  mean|diff| %.3e px\n",
                max_diff, sum_diff / N_FEAT);
    std::printf("endpoint error vs ground-truth flow: mean %.3f px\n",
                mean_endpoint);

    // --- animation: side-by-side I + J, with estimated flow arrows ---------
    if (system("mkdir -p tmp") != 0)
        std::fprintf(stderr, "warning: mkdir tmp failed\n");
    cv::VideoWriter video("tmp/gpu_lk_optical_flow.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          10, cv::Size(PANEL_W, PANEL_H));

    // Pre-render the image canvases.
    auto to_mat = [&](const std::vector<float>& im) {
        cv::Mat g(IMG_H, IMG_W, CV_8UC1);
        for (int y = 0; y < IMG_H; ++y)
            for (int x = 0; x < IMG_W; ++x) {
                float v = im[y * IMG_W + x];
                g.at<uchar>(y, x) = (uchar)std::min(255.0f, std::max(0.0f, v * 255.0f));
            }
        cv::Mat colour;
        cv::cvtColor(g, colour, cv::COLOR_GRAY2BGR);
        return colour;
    };
    cv::Mat I_canvas = to_mat(I0);
    cv::Mat J_canvas = to_mat(J0);

    const int VIS_SCALE = 1;                        // 1x scale -> 256 wide
    const int IMG_AREA_W = IMG_W * VIS_SCALE;
    const int IMG_AREA_H = IMG_H * VIS_SCALE;
    const int margin_x = (PANEL_W - 2 * IMG_AREA_W - 16) / 2;
    const int margin_y = 50;

    const int N_FRAMES = 24;
    for (int f = 0; f < N_FRAMES; ++f) {
        cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(20, 20, 26));
        cv::Mat roi_I = img(cv::Rect(margin_x, margin_y, IMG_AREA_W, IMG_AREA_H));
        cv::Mat roi_J = img(cv::Rect(margin_x + IMG_AREA_W + 16, margin_y,
                                     IMG_AREA_W, IMG_AREA_H));
        I_canvas.copyTo(roi_I);
        J_canvas.copyTo(roi_J);

        // progressively reveal feature trails
        int reveal = (int)((float)(f + 1) / N_FRAMES * N_FEAT);
        for (int i = 0; i < reveal; ++i) {
            if (!vd_gpu[i]) continue;
            int x0 = margin_x + (int)(fx[i] * VIS_SCALE);
            int y0 = margin_y + (int)(fy[i] * VIS_SCALE);
            int xJ0 = margin_x + IMG_AREA_W + 16 + (int)((fx[i] + dx_gpu[i]) * VIS_SCALE);
            int yJ0 = margin_y + (int)((fy[i] + dy_gpu[i]) * VIS_SCALE);
            cv::circle(img, cv::Point(x0, y0), 1, cv::Scalar(80, 220, 255), -1);
            cv::arrowedLine(img, cv::Point(xJ0 - (int)(dx_gpu[i] * VIS_SCALE),
                                           yJ0 - (int)(dy_gpu[i] * VIS_SCALE)),
                            cv::Point(xJ0, yJ0), cv::Scalar(120, 255, 180),
                            1, cv::LINE_AA, 0, 0.35);
        }

        cv::putText(img, "GPU pyramidal Lucas-Kanade (one thread = one feature)",
                    cv::Point(12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
        cv::putText(img, "frame I", cv::Point(margin_x + IMG_AREA_W / 2 - 28, margin_y - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 220, 240), 1, cv::LINE_AA);
        cv::putText(img, "frame J + estimated flow",
                    cv::Point(margin_x + IMG_AREA_W + 16 + IMG_AREA_W / 2 - 90, margin_y - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 240, 220), 1, cv::LINE_AA);

        char l1[160], l2[160], l3[160];
        std::snprintf(l1, sizeof(l1), "%d features  x  %d Gauss-Newton iters  x  %d pyramid levels",
                      N_FEAT, N_ITERS, N_LEVELS);
        std::snprintf(l2, sizeof(l2),
                      "CPU LK %.0f ms  vs  GPU LK %.2f ms  (%.0fx)",
                      cpu_ms, gpu_ms, speedup);
        std::snprintf(l3, sizeof(l3),
                      "CPU/GPU flow max|diff| %.1e px   ground-truth endpoint error %.2f px",
                      max_diff, mean_endpoint);
        cv::putText(img, l1, cv::Point(12, PANEL_H - 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 220, 255), 1, cv::LINE_AA);
        cv::putText(img, l2, cv::Point(12, PANEL_H - 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 255, 200), 1, cv::LINE_AA);
        cv::putText(img, l3, cv::Point(12, PANEL_H - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        video.write(img);
    }
    video.release();

    cudabot::avi_to_gif("tmp/gpu_lk_optical_flow.avi",
                        "gif/gpu_lk_optical_flow.gif", 10, 760);
    std::printf("wrote gif/gpu_lk_optical_flow.gif\n");

    for (int l = 0; l < N_LEVELS; ++l) {
        CUDA_CHECK(cudaFree(dI[l]));
        CUDA_CHECK(cudaFree(dJ[l]));
    }
    CUDA_CHECK(cudaFree(d_fx));
    CUDA_CHECK(cudaFree(d_fy));
    CUDA_CHECK(cudaFree(d_dx));
    CUDA_CHECK(cudaFree(d_dy));
    CUDA_CHECK(cudaFree(d_vd));
    return 0;
}
