// gpu_correlative_scan_matching.cu
//
// GPU correlative scan matching (CSM): exhaustive global pose search.
//
// The scan-matching family in this repo (NDT 2D/3D, GICP 2D/3D) refines a pose
// by local iteration -- Newton / Gauss-Newton on a smooth objective.  That is
// fast and accurate WHEN the initial guess is already in the right basin, but
// from a large initial offset it falls into the nearest local optimum and never
// recovers.  Correlative scan matching (Olson, "Real-Time Correlative Scan
// Matching", ICRA 2009; also Cartographer's real-time CSM) instead searches a
// discretised window of candidate poses EXHAUSTIVELY and takes the global
// maximum of the scan-to-map score.  This is embarrassingly parallel -- one
// thread scores one candidate pose -- so the GPU evaluates millions of (x, y,
// theta) candidates per frame, which is exactly where it shines as a global
// loop-closure / relocalization alignment primitive.
//
// This demo runs a controlled comparison on the SAME field objective:
//   * LOCAL  : gradient ascent on the map likelihood field from the offset init
//              (a stand-in for the NDT/GICP local refiners);
//   * CSM    : coarse-to-fine exhaustive GPU search of an (x, y, theta) window
//              centred on the same init.
// Across frames the initial offset grows; the local matcher breaks down past
// its basin while CSM stays locked to the global optimum.  A single CPU CSM run
// times the same exhaustive search for the speedup headline.
//
// Output: gif/gpu_correlative_scan_matching.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int THREADS = 256;
constexpr float PI_F = 3.14159265358979323846f;
constexpr float WORLD_HALF = 14.0f;
constexpr float LUT_RES = 0.05f;
constexpr int GRID_N = static_cast<int>(2.0f * WORLD_HALF / LUT_RES);  // 560
constexpr float FIELD_SIGMA = 0.35f;
constexpr int N_SCAN = 120;
constexpr float MAX_RANGE = 30.0f;

constexpr int N_FRAMES = 44;
constexpr float MAX_OFF_XY = 3.8f;   // peak translation offset of the init guess
constexpr float MAX_OFF_TH = 0.70f;  // peak rotation offset (rad)

// Coarse-to-fine search window (centred on the init guess).
constexpr int COARSE_NXY = 151;        // 151 x 151 positions ...
constexpr float COARSE_RES_XY = 0.06f; // ... at 0.06 m  => +/- 4.5 m
constexpr int COARSE_NT = 91;          // 91 headings ...
constexpr float COARSE_RES_TH = 0.0175f; // ... at 1.0 deg => +/- 45 deg
constexpr int FINE_NXY = 41;
constexpr float FINE_RES_XY = 0.015f;  // +/- 0.30 m
constexpr int FINE_NT = 41;
constexpr float FINE_RES_TH = 0.0035f; // +/- 4 deg

constexpr float SUCCESS_M = 0.20f;     // pose counted as recovered below this

constexpr int PANEL_W = 460;
constexpr int PANEL_H = 460;
constexpr int INFO_W = 330;
constexpr int FRAME_W = PANEL_W * 2 + INFO_W;
constexpr int FRAME_H = PANEL_H;

struct Pose { float x, y, th; };
struct Rect { float x0, y0, x1, y1; };

__host__ __device__ static inline float clampf(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}
__host__ __device__ static inline float wrap_angle(float a) {
    while (a > PI_F) a -= 2.0f * PI_F;
    while (a < -PI_F) a += 2.0f * PI_F;
    return a;
}
static float pose_err_xy(const Pose& a, const Pose& b) {
    float dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// --- Map + likelihood field -------------------------------------------------
static std::vector<Rect> make_rects() {
    // Asymmetric indoor scene built from THIN wall slabs + small pillars, so
    // there are no large filled interiors (a real scan only sees surfaces).
    std::vector<Rect> r;
    // Room perimeter (thin slabs inset from the world edge).
    r.push_back({-13.5f, 13.2f, 13.5f, 13.5f});
    r.push_back({-13.5f, -13.5f, 13.5f, -13.2f});
    r.push_back({-13.5f, -13.5f, -13.2f, 13.5f});
    r.push_back({13.2f, -13.5f, 13.5f, 13.5f});
    // Internal walls (asymmetric).
    r.push_back({-9.0f, 6.5f, 1.5f, 6.8f});     // upper horizontal wall
    r.push_back({4.5f, 5.0f, 4.8f, 12.0f});     // upper-right vertical wall
    r.push_back({-11.0f, -3.0f, -10.7f, 8.0f}); // left vertical wall
    r.push_back({-6.0f, -8.5f, 2.0f, -8.2f});   // lower horizontal wall
    r.push_back({6.5f, -9.0f, 9.5f, -8.7f});    // lower-right wall
    r.push_back({9.2f, -9.0f, 9.5f, -5.0f});    // lower-right corner (L)
    // Small pillars.
    r.push_back({-2.4f, -0.8f, -1.6f, 0.0f});
    r.push_back({8.0f, 1.0f, 8.8f, 1.8f});
    return r;
}

static bool is_wall(const std::vector<Rect>& rects, float x, float y) {
    if (x <= -WORLD_HALF || x >= WORLD_HALF || y <= -WORLD_HALF || y >= WORLD_HALF) return true;
    for (const Rect& r : rects) {
        if (x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1) return true;
    }
    return false;
}

// Build the likelihood field from obstacle SURFACES: a cell is a surface cell
// if it is occupied and adjacent to free space (out-of-grid counts as free).
// lut[cell] = exp(-d^2 / (2 sigma^2)), d = distance to the nearest surface --
// the classic likelihood-field LUT.  Scoring against surfaces (not filled
// interiors) is what makes the true pose the global maximum: a scan only ever
// observes surfaces, so piling endpoints into a solid interior is not rewarded.
static std::vector<float> build_field(const std::vector<Rect>& rects) {
    auto wall_at = [&](int ix, int iy) {
        if (ix < 0 || ix >= GRID_N || iy < 0 || iy >= GRID_N) return false;  // outside = free
        float x = -WORLD_HALF + (ix + 0.5f) * LUT_RES;
        float y = -WORLD_HALF + (iy + 0.5f) * LUT_RES;
        return is_wall(rects, x, y);
    };
    cv::Mat surf(GRID_N, GRID_N, CV_8U, cv::Scalar(255));
    for (int iy = 0; iy < GRID_N; ++iy) {
        for (int ix = 0; ix < GRID_N; ++ix) {
            if (!wall_at(ix, iy)) continue;
            bool boundary = !wall_at(ix - 1, iy) || !wall_at(ix + 1, iy) ||
                            !wall_at(ix, iy - 1) || !wall_at(ix, iy + 1);
            if (boundary) surf.at<unsigned char>(iy, ix) = 0;  // 0 = surface for DT
        }
    }
    cv::Mat dist_px;
    cv::distanceTransform(surf, dist_px, cv::DIST_L2, 5);
    std::vector<float> lut(GRID_N * GRID_N);
    float inv2s2 = 1.0f / (2.0f * FIELD_SIGMA * FIELD_SIGMA);
    for (int iy = 0; iy < GRID_N; ++iy) {
        for (int ix = 0; ix < GRID_N; ++ix) {
            float d = dist_px.at<float>(iy, ix) * LUT_RES;
            lut[iy * GRID_N + ix] = std::exp(-d * d * inv2s2);
        }
    }
    return lut;
}

// Host raycast: range scan from a pose, returned as sensor-frame endpoints.
static void make_scan(const std::vector<Rect>& rects, const Pose& p,
                      std::vector<float>& sx, std::vector<float>& sy, unsigned seed) {
    sx.resize(N_SCAN);
    sy.resize(N_SCAN);
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.02f);
    for (int i = 0; i < N_SCAN; ++i) {
        float a = p.th + (2.0f * PI_F * i) / N_SCAN;
        float ca = std::cos(a), sa = std::sin(a);
        float r = MAX_RANGE;
        for (float t = 0.10f; t < MAX_RANGE; t += 0.04f) {
            if (is_wall(rects, p.x + t * ca, p.y + t * sa)) { r = t; break; }
        }
        r = clampf(r + noise(rng), 0.15f, MAX_RANGE);
        // Sensor frame: rotate the world-frame ray back by -p.th.
        float wx = r * ca, wy = r * sa;  // world displacement from sensor
        float c = std::cos(-p.th), s = std::sin(-p.th);
        sx[i] = c * wx - s * wy;
        sy[i] = s * wx + c * wy;
    }
}

// --- CSM scoring ------------------------------------------------------------
__host__ __device__ static inline float sample_field(const float* lut, float wx, float wy) {
    float fx = (wx + WORLD_HALF) / LUT_RES - 0.5f;
    float fy = (wy + WORLD_HALF) / LUT_RES - 0.5f;
    int ix = static_cast<int>(floorf(fx));
    int iy = static_cast<int>(floorf(fy));
    if (ix < 0 || ix >= GRID_N - 1 || iy < 0 || iy >= GRID_N - 1) return 0.0f;
    float tx = fx - ix, ty = fy - iy;
    const float* row0 = lut + iy * GRID_N + ix;
    const float* row1 = row0 + GRID_N;
    float a = row0[0] * (1 - tx) + row0[1] * tx;
    float b = row1[0] * (1 - tx) + row1[1] * tx;
    return a * (1 - ty) + b * ty;
}

// One thread scores one candidate pose over the whole scan.
__global__ void csm_kernel(const float* __restrict__ scan_x,
                           const float* __restrict__ scan_y,
                           const float* __restrict__ lut,
                           float cx, float cy, float cth,
                           int nxy, int nt, float res_xy, float res_th,
                           float* __restrict__ score) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nxy * nxy * nt;
    if (idx >= total) return;
    int it = idx / (nxy * nxy);
    int rem = idx - it * (nxy * nxy);
    int iy = rem / nxy;
    int ix = rem - iy * nxy;
    float px = cx + (ix - (nxy - 1) * 0.5f) * res_xy;
    float py = cy + (iy - (nxy - 1) * 0.5f) * res_xy;
    float pth = cth + (it - (nt - 1) * 0.5f) * res_th;
    float c = cosf(pth), s = sinf(pth);
    float acc = 0.0f;
    for (int k = 0; k < N_SCAN; ++k) {
        float lx = scan_x[k], ly = scan_y[k];
        float wx = px + c * lx - s * ly;
        float wy = py + s * lx + c * ly;
        acc += sample_field(lut, wx, wy);
    }
    score[idx] = acc;
}

struct CsmResult {
    Pose best{};
    std::vector<float> coarse_score;  // for the heatmap (max over theta)
    int coarse_w = 0;
};

// GPU coarse-to-fine exhaustive search centred on `init`.
static Pose run_gpu_csm(const float* d_scan_x, const float* d_scan_y, const float* d_lut,
                        const Pose& init, float* d_score, std::vector<float>& h_score,
                        CsmResult* out) {
    auto search = [&](Pose center, int nxy, int nt, float rxy, float rth,
                      std::vector<float>* keep) -> Pose {
        int total = nxy * nxy * nt;
        int blocks = (total + THREADS - 1) / THREADS;
        csm_kernel<<<blocks, THREADS>>>(d_scan_x, d_scan_y, d_lut,
                                        center.x, center.y, center.th,
                                        nxy, nt, rxy, rth, d_score);
        CUDA_CHECK(cudaGetLastError());
        h_score.resize(total);
        CUDA_CHECK(cudaMemcpy(h_score.data(), d_score, total * sizeof(float), cudaMemcpyDeviceToHost));
        int best = 0;
        for (int i = 1; i < total; ++i)
            if (h_score[i] > h_score[best]) best = i;
        int it = best / (nxy * nxy);
        int rem = best - it * (nxy * nxy);
        int iy = rem / nxy, ix = rem - iy * nxy;
        Pose p;
        p.x = center.x + (ix - (nxy - 1) * 0.5f) * rxy;
        p.y = center.y + (iy - (nxy - 1) * 0.5f) * rxy;
        p.th = wrap_angle(center.th + (it - (nt - 1) * 0.5f) * rth);
        if (keep) {  // store max-over-theta map for the heatmap
            keep->assign(nxy * nxy, 0.0f);
            for (int t = 0; t < nt; ++t)
                for (int c = 0; c < nxy * nxy; ++c) {
                    float v = h_score[t * nxy * nxy + c];
                    if (v > (*keep)[c]) (*keep)[c] = v;
                }
        }
        return p;
    };
    Pose coarse = search(init, COARSE_NXY, COARSE_NT, COARSE_RES_XY, COARSE_RES_TH,
                         out ? &out->coarse_score : nullptr);
    if (out) out->coarse_w = COARSE_NXY;
    Pose fine = search(coarse, FINE_NXY, FINE_NT, FINE_RES_XY, FINE_RES_TH, nullptr);
    if (out) out->best = fine;
    return fine;
}

// CPU reference of the same exhaustive search (single resolution, for timing).
static Pose run_cpu_csm(const std::vector<float>& scan_x, const std::vector<float>& scan_y,
                        const std::vector<float>& lut, const Pose& init,
                        int nxy, int nt, float rxy, float rth) {
    float best_score = -1.0f;
    Pose best = init;
    for (int it = 0; it < nt; ++it) {
        float pth = init.th + (it - (nt - 1) * 0.5f) * rth;
        float c = std::cos(pth), s = std::sin(pth);
        for (int iy = 0; iy < nxy; ++iy) {
            float py = init.y + (iy - (nxy - 1) * 0.5f) * rxy;
            for (int ix = 0; ix < nxy; ++ix) {
                float px = init.x + (ix - (nxy - 1) * 0.5f) * rxy;
                float acc = 0.0f;
                for (int k = 0; k < N_SCAN; ++k) {
                    float wx = px + c * scan_x[k] - s * scan_y[k];
                    float wy = py + s * scan_x[k] + c * scan_y[k];
                    acc += sample_field(lut.data(), wx, wy);
                }
                if (acc > best_score) { best_score = acc; best = {px, py, pth}; }
            }
        }
    }
    return best;
}

// Local baseline: gradient ascent on the same field objective (finite diff).
static float field_objective(const std::vector<float>& scan_x, const std::vector<float>& scan_y,
                             const std::vector<float>& lut, const Pose& p) {
    float c = std::cos(p.th), s = std::sin(p.th), acc = 0.0f;
    for (int k = 0; k < N_SCAN; ++k) {
        float wx = p.x + c * scan_x[k] - s * scan_y[k];
        float wy = p.y + s * scan_x[k] + c * scan_y[k];
        acc += sample_field(lut.data(), wx, wy);
    }
    return acc;
}

static Pose run_local(const std::vector<float>& scan_x, const std::vector<float>& scan_y,
                      const std::vector<float>& lut, const Pose& init) {
    Pose p = init;
    float ex = 0.05f, eth = 0.01f;
    float step_xy = 0.6f, step_th = 0.25f;
    for (int iter = 0; iter < 80; ++iter) {
        float jx = (field_objective(scan_x, scan_y, lut, {p.x + ex, p.y, p.th}) -
                    field_objective(scan_x, scan_y, lut, {p.x - ex, p.y, p.th})) / (2 * ex);
        float jy = (field_objective(scan_x, scan_y, lut, {p.x, p.y + ex, p.th}) -
                    field_objective(scan_x, scan_y, lut, {p.x, p.y - ex, p.th})) / (2 * ex);
        float jt = (field_objective(scan_x, scan_y, lut, {p.x, p.y, p.th + eth}) -
                    field_objective(scan_x, scan_y, lut, {p.x, p.y, p.th - eth})) / (2 * eth);
        p.x += clampf(step_xy * jx, -0.5f, 0.5f);
        p.y += clampf(step_xy * jy, -0.5f, 0.5f);
        p.th = wrap_angle(p.th + clampf(step_th * jt, -0.2f, 0.2f));
    }
    return p;
}

// --- Visualization ----------------------------------------------------------
static cv::Point world_to_panel(int ox, float x, float y) {
    int px = ox + static_cast<int>((x + WORLD_HALF) / (2 * WORLD_HALF) * PANEL_W);
    int py = static_cast<int>((WORLD_HALF - y) / (2 * WORLD_HALF) * PANEL_H);
    return cv::Point(px, py);
}

static void draw_scan(cv::Mat& img, int ox, const std::vector<float>& sx, const std::vector<float>& sy,
                      const Pose& p, const cv::Scalar& color) {
    float c = std::cos(p.th), s = std::sin(p.th);
    for (int k = 0; k < N_SCAN; ++k) {
        float wx = p.x + c * sx[k] - s * sy[k];
        float wy = p.y + s * sx[k] + c * sy[k];
        cv::circle(img, world_to_panel(ox, wx, wy), 1, color, -1, cv::LINE_AA);
    }
    cv::circle(img, world_to_panel(ox, p.x, p.y), 4, color, -1, cv::LINE_AA);
}

static void draw_map_panel(cv::Mat& img, int ox, const std::vector<Rect>& rects,
                           const std::vector<float>& sx, const std::vector<float>& sy,
                           const Pose& truth, const Pose& init, const Pose& local, const Pose& csm) {
    cv::rectangle(img, cv::Rect(ox, 0, PANEL_W, PANEL_H), cv::Scalar(250, 250, 247), -1);
    for (const Rect& r : rects) {
        cv::rectangle(img, world_to_panel(ox, r.x0, r.y1), world_to_panel(ox, r.x1, r.y0),
                      cv::Scalar(58, 64, 72), -1);
    }
    cv::rectangle(img, cv::Rect(ox + 1, 1, PANEL_W - 2, PANEL_H - 2), cv::Scalar(120, 124, 130), 1);
    draw_scan(img, ox, sx, sy, init, cv::Scalar(70, 70, 220));    // red  : init
    draw_scan(img, ox, sx, sy, local, cv::Scalar(40, 150, 240));  // orange: local
    draw_scan(img, ox, sx, sy, csm, cv::Scalar(60, 170, 70));     // green: CSM
    cv::circle(img, world_to_panel(ox, truth.x, truth.y), 6, cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
    cv::putText(img, "scan alignment", cv::Point(ox + 12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(20, 24, 32), 2, cv::LINE_AA);
}

static void draw_heat_panel(cv::Mat& img, int ox, const CsmResult& res,
                            const Pose& init, const Pose& truth, const Pose& csm) {
    int w = res.coarse_w;
    cv::Mat heat(w, w, CV_8U);
    float lo = 1e30f, hi = -1e30f;
    for (float v : res.coarse_score) { lo = std::min(lo, v); hi = std::max(hi, v); }
    float inv = (hi > lo) ? 1.0f / (hi - lo) : 0.0f;
    for (int iy = 0; iy < w; ++iy)
        for (int ix = 0; ix < w; ++ix) {
            float v = (res.coarse_score[iy * w + ix] - lo) * inv;
            heat.at<unsigned char>(w - 1 - iy, ix) = static_cast<unsigned char>(255.0f * v);
        }
    cv::Mat color;
    cv::applyColorMap(heat, color, cv::COLORMAP_INFERNO);
    cv::Mat dst;
    cv::resize(color, dst, cv::Size(PANEL_W, PANEL_H), 0, 0, cv::INTER_NEAREST);
    dst.copyTo(img(cv::Rect(ox, 0, PANEL_W, PANEL_H)));
    // Map a world pose into this heatmap panel (search window centred on init).
    auto to_heat = [&](const Pose& p) {
        float gx = (p.x - init.x) / COARSE_RES_XY + (w - 1) * 0.5f;
        float gy = (p.y - init.y) / COARSE_RES_XY + (w - 1) * 0.5f;
        int px = ox + static_cast<int>(gx / w * PANEL_W);
        int py = static_cast<int>((w - 1 - gy) / w * PANEL_H);
        return cv::Point(px, py);
    };
    cv::drawMarker(img, to_heat(truth), cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 16, 2);
    cv::circle(img, to_heat(csm), 6, cv::Scalar(60, 230, 90), 2, cv::LINE_AA);
    cv::circle(img, to_heat(init), 5, cv::Scalar(80, 80, 240), 2, cv::LINE_AA);
    cv::putText(img, "CSM score field (max over theta)", cv::Point(ox + 12, 26),
                cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
}

struct Stats {
    int n = 0, local_ok = 0, csm_ok = 0;
    double local_sq = 0, csm_sq = 0;
    double gpu_ms_sum = 0;
    double cpu_ms_once = 0, gpu_ms_once = 0;
    long long candidates = 0;
};

static void draw_info(cv::Mat& img, int ox, int frame, float off_xy, float off_th,
                      float local_err, float csm_err, const Stats& s,
                      const std::vector<float>& local_hist, const std::vector<float>& csm_hist) {
    cv::rectangle(img, cv::Rect(ox, 0, INFO_W, FRAME_H), cv::Scalar(244, 246, 246), -1);
    cv::putText(img, "Correlative scan match", cv::Point(ox + 16, 32), cv::FONT_HERSHEY_SIMPLEX,
                0.58, cv::Scalar(20, 28, 35), 2, cv::LINE_AA);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "init offset: %.2f m / %.0f deg", off_xy, off_th * 180.0f / PI_F);
    cv::putText(img, buf, cv::Point(ox + 16, 64), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                cv::Scalar(60, 66, 74), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "local err:  %.2f m", local_err);
    cv::putText(img, buf, cv::Point(ox + 16, 96), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(40, 150, 240), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "CSM err:    %.2f m", csm_err);
    cv::putText(img, buf, cv::Point(ox + 16, 120), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(60, 170, 70), 1, cv::LINE_AA);

    // Error-vs-frame plot.
    int px0 = ox + 16, py0 = 150, pw = INFO_W - 40, ph = 120;
    cv::rectangle(img, cv::Rect(px0, py0, pw, ph), cv::Scalar(255, 255, 255), -1);
    cv::rectangle(img, cv::Rect(px0, py0, pw, ph), cv::Scalar(200, 204, 210), 1);
    float ymax = 4.0f;
    auto plot = [&](const std::vector<float>& h, const cv::Scalar& col) {
        for (size_t i = 1; i < h.size(); ++i) {
            float x0 = px0 + pw * (i - 1) / (float)(N_FRAMES - 1);
            float x1 = px0 + pw * i / (float)(N_FRAMES - 1);
            float y0 = py0 + ph - ph * clampf(h[i - 1] / ymax, 0, 1);
            float y1 = py0 + ph - ph * clampf(h[i] / ymax, 0, 1);
            cv::line(img, cv::Point((int)x0, (int)y0), cv::Point((int)x1, (int)y1), col, 2, cv::LINE_AA);
        }
    };
    plot(local_hist, cv::Scalar(40, 150, 240));
    plot(csm_hist, cv::Scalar(60, 170, 70));
    cv::putText(img, "pose error vs frame (0-4 m)", cv::Point(px0, py0 - 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(70, 76, 84), 1, cv::LINE_AA);

    int y = py0 + ph + 34;
    std::snprintf(buf, sizeof(buf), "candidates/frame: %.2fM", s.candidates / 1e6);
    cv::putText(img, buf, cv::Point(ox + 16, y), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "GPU CSM: %.2f ms  (CPU %.0f ms)", s.gpu_ms_once, s.cpu_ms_once);
    cv::putText(img, buf, cv::Point(ox + 16, y + 24), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    if (s.cpu_ms_once > 0 && s.gpu_ms_once > 0) {
        std::snprintf(buf, sizeof(buf), "speedup: %.0fx", s.cpu_ms_once / s.gpu_ms_once);
        cv::putText(img, buf, cv::Point(ox + 16, y + 48), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                    cv::Scalar(45, 95, 175), 1, cv::LINE_AA);
    }
    std::snprintf(buf, sizeof(buf), "recovered: local %d/%d  CSM %d/%d", s.local_ok, s.n, s.csm_ok, s.n);
    cv::putText(img, buf, cv::Point(ox + 16, y + 72), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
}

static void ensure_dirs() {
    int rc = std::system("mkdir -p gif tmp");
    if (rc != 0) std::fprintf(stderr, "mkdir failed with code %d\n", rc);
}

static Stats run_demo() {
    ensure_dirs();
    std::vector<Rect> rects = make_rects();
    std::vector<float> lut = build_field(rects);

    float *d_lut = nullptr, *d_scan_x = nullptr, *d_scan_y = nullptr, *d_score = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lut, lut.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lut, lut.data(), lut.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_scan_x, N_SCAN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_y, N_SCAN * sizeof(float)));
    int max_total = COARSE_NXY * COARSE_NXY * COARSE_NT;
    CUDA_CHECK(cudaMalloc(&d_score, max_total * sizeof(float)));

    Pose truth{-1.5f, 0.8f, 0.6f};
    std::vector<float> sx, sy;
    make_scan(rects, truth, sx, sy, 777);
    CUDA_CHECK(cudaMemcpy(d_scan_x, sx.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scan_y, sy.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));

    cv::VideoWriter video("tmp/gpu_correlative_scan_matching.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(FRAME_W, FRAME_H));
    if (!video.isOpened()) { std::fprintf(stderr, "failed to open video\n"); std::exit(1); }

    std::vector<float> h_score;
    Stats stats;
    stats.candidates = (long long)max_total + (long long)FINE_NXY * FINE_NXY * FINE_NT;
    std::vector<float> local_hist, csm_hist;

    for (int f = 0; f < N_FRAMES; ++f) {
        float frac = (N_FRAMES == 1) ? 0.0f : f / (float)(N_FRAMES - 1);
        float off_xy = MAX_OFF_XY * frac;
        float off_th = MAX_OFF_TH * frac * ((f % 2 == 0) ? 1.0f : -1.0f);
        float dir = 0.7f * f;
        Pose init{truth.x + off_xy * std::cos(dir), truth.y + off_xy * std::sin(dir),
                  wrap_angle(truth.th + off_th)};

        Pose local = run_local(sx, sy, lut, init);

        CsmResult res;
        auto g0 = std::chrono::high_resolution_clock::now();
        Pose csm = run_gpu_csm(d_scan_x, d_scan_y, d_lut, init, d_score, h_score, &res);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto g1 = std::chrono::high_resolution_clock::now();
        double gpu_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
        stats.gpu_ms_sum += gpu_ms;
        stats.gpu_ms_once = gpu_ms;  // live GPU time for the info panel

        // Time the CPU reference once (coarse grid) up front, so the speedup is
        // visible across the whole animation.
        if (f == 0) {
            auto c0 = std::chrono::high_resolution_clock::now();
            Pose cpu = run_cpu_csm(sx, sy, lut, init, COARSE_NXY, COARSE_NT,
                                   COARSE_RES_XY, COARSE_RES_TH);
            auto c1 = std::chrono::high_resolution_clock::now();
            stats.cpu_ms_once = std::chrono::duration<double, std::milli>(c1 - c0).count();
            std::printf("CPU CSM check: (%.3f, %.3f, %.3f)\n", cpu.x, cpu.y, cpu.th);  // use result
        }

        float le = pose_err_xy(local, truth), ce = pose_err_xy(csm, truth);
        stats.n++;
        stats.local_sq += le * le;
        stats.csm_sq += ce * ce;
        if (le < SUCCESS_M) stats.local_ok++;
        if (ce < SUCCESS_M) stats.csm_ok++;
        local_hist.push_back(le);
        csm_hist.push_back(ce);

        cv::Mat frame(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(244, 246, 246));
        draw_map_panel(frame, 0, rects, sx, sy, truth, init, local, csm);
        draw_heat_panel(frame, PANEL_W, res, init, truth, csm);
        draw_info(frame, PANEL_W * 2, f, off_xy, off_th, le, ce, stats, local_hist, csm_hist);
        video.write(frame);

        std::printf("frame %2d off=%.2fm/%.0fdeg local_err=%.3f csm_err=%.3f gpu=%.2fms\n",
                    f, off_xy, off_th * 180.0f / PI_F, le, ce, gpu_ms);
    }

    video.release();
    avi_to_gif("tmp/gpu_correlative_scan_matching.avi", "gif/gpu_correlative_scan_matching.gif", 10, 760);

    CUDA_CHECK(cudaFree(d_lut));
    CUDA_CHECK(cudaFree(d_scan_x));
    CUDA_CHECK(cudaFree(d_scan_y));
    CUDA_CHECK(cudaFree(d_score));
    return stats;
}

}  // namespace cudabot

int main() {
    cudabot::Stats s = cudabot::run_demo();
    std::printf("\nGPU correlative scan matching (exhaustive global alignment)\n");
    std::printf("candidates/frame: %.3f M (coarse-to-fine)\n", s.candidates / 1e6);
    std::printf("pose RMSE over %d frames: local %.3f m, CSM %.3f m\n",
                s.n, std::sqrt(s.local_sq / s.n), std::sqrt(s.csm_sq / s.n));
    std::printf("recovered (<%.2f m): local %d/%d, CSM %d/%d\n",
                cudabot::SUCCESS_M, s.local_ok, s.n, s.csm_ok, s.n);
    std::printf("GPU CSM %.2f ms vs CPU %.1f ms (%.0fx) on the hardest frame\n",
                s.gpu_ms_once, s.cpu_ms_once,
                s.cpu_ms_once > 0 ? s.cpu_ms_once / s.gpu_ms_once : 0.0);
    std::printf("avg GPU CSM step: %.2f ms\n", s.gpu_ms_sum / s.n);
    std::printf("Wrote gif/gpu_correlative_scan_matching.gif\n");
    return 0;
}
