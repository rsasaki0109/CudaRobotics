// gpu_csm_submap_slam.cu
//
// GPU 2D SLAM with submap-based loop closures (scan-to-SUBMAP correlative scan
// matching front-end + pose-graph back-end), and a controlled comparison
// against the single-scan loop-closure front-end of #121.
//
// #121 (gpu_csm_loop_closure_slam) detects loops by matching the current scan
// against a likelihood field built from ONE earlier keyframe's scan.  That is
// the bare minimum: a single range scan sees each surface from a single
// viewpoint, so its field is sparse and view-dependent, and a revisit observed
// from a slightly different heading can score below the accept gate even though
// it is a true loop.  Real systems (Cartographer) instead match against a
// SUBMAP: several consecutive scans fused into one local probability/occupancy
// grid.  Fusing many viewpoints fills in the surfaces, so the field is denser
// and far less view-dependent, and the revisiting scan finds support over a
// wider basin -- more true loops cross the gate, with a sharper relpose.
//
// This demo builds that submap front-end and runs it head-to-head against the
// single-scan front-end on the SAME drifting trajectory, the SAME candidate
// gating, and the SAME exhaustive coarse-to-fine CSM search (one thread = one
// candidate relpose, ~1.4 M candidates per attempt).  The ONLY difference
// between the two variants is the likelihood field: a single anchor scan
// (scan-to-scan, #121) versus the fused submap around that anchor
// (scan-to-submap, this demo).  No ground truth enters either constraint -- the
// intra-submap relposes that fuse the field come from the live estimate, and
// over a ~dozen keyframes their drift is negligible.
//
// Layout: [dead reckoning] | [scan-to-submap SLAM] | [info].  The info panel
// plots dead-reckon vs scan-to-scan vs scan-to-submap ATE and tallies how many
// loops each front-end accepts.  The submap variant accepts more true loops and
// lands a lower ATE.
//
// Output: gif/gpu_csm_submap_slam.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_map>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int THREADS = 256;
constexpr float PI_F = 3.14159265358979323846f;

// --- World / sensor ---------------------------------------------------------
constexpr float WORLD_HALF = 14.0f;
// A deliberately SPARSE, noisy 2D scan: a single such scan localises poorly,
// which is exactly why fusing several into a submap pays off.
constexpr int N_SCAN = 64;           // rays per scan (sparse)
constexpr float SCAN_NOISE = 0.060f; // range noise std (m)
constexpr float MAX_RANGE = 30.0f;   // raycast cutoff
constexpr float MATCH_RANGE = 12.0f; // endpoints beyond this are not matched

// --- Local likelihood field (built per loop target, in its sensor frame) ----
constexpr float LF_HALF = 12.5f;
constexpr float LF_RES = 0.0625f;
constexpr int LGRID = static_cast<int>(2.0f * LF_HALF / LF_RES);  // 400
constexpr float FIELD_SIGMA = 0.45f;

// --- Submap ------------------------------------------------------------------
// A submap fuses SUBMAP_KF consecutive keyframe scans into one local field.
constexpr int SUBMAP_KF = 8;

// --- Trajectory (single closed elliptical lap) ------------------------------
constexpr int N_KF = 140;
constexpr float ELLIPSE_A = 9.0f;
constexpr float ELLIPSE_B = 5.5f;

// Odometry noise + a small systematic heading bias so the lap fails to close
// under dead reckoning (a few metres of drift at the seam).
constexpr float ODOM_SIGMA_XY = 0.022f;
constexpr float ODOM_SIGMA_TH = 0.006f;
constexpr float ODOM_BIAS_TH = 0.0034f;

// --- Loop-closure detection -------------------------------------------------
constexpr int LC_MIN_GAP = 45;      // min index gap to call it a revisit
constexpr float LC_GATE_R = 5.2f;   // candidate if estimates are within this
constexpr int LC_MAX_CAND = 2;      // attempts per keyframe (nearest first)
constexpr float LC_ACCEPT = 0.62f;  // min normalised scan-match score to accept

// CSM coarse-to-fine search window (centred on the estimate-predicted relpose).
constexpr int COARSE_NXY = 131;          // 131 x 131 ...
constexpr float COARSE_RES_XY = 0.080f;  // ... at 0.08 m => +/- 5.2 m
constexpr int COARSE_NT = 81;            // 81 headings ...
constexpr float COARSE_RES_TH = 0.0150f; // ... at 0.86 deg => +/- 0.60 rad
constexpr int FINE_NXY = 31;
constexpr float FINE_RES_XY = 0.020f;    // +/- 0.30 m
constexpr int FINE_NT = 31;
constexpr float FINE_RES_TH = 0.0040f;   // +/- 0.06 rad

// --- Pose-graph back-end ----------------------------------------------------
constexpr int GN_ITERS = 8;
constexpr float ODOM_INFO_XY = 1.0f / (ODOM_SIGMA_XY * ODOM_SIGMA_XY);
constexpr float ODOM_INFO_TH = 1.0f / (ODOM_SIGMA_TH * ODOM_SIGMA_TH);
constexpr float LOOP_INFO_XY = 320.0f;   // scan match ~0.055 m
constexpr float LOOP_INFO_TH = 4000.0f;  // scan match ~0.016 rad
constexpr float ANCHOR_INFO = 1.0e7f;

// --- Visualization ----------------------------------------------------------
constexpr int PANEL_W = 430;
constexpr int PANEL_H = 430;
constexpr int INFO_W = 330;
constexpr int FRAME_W = PANEL_W * 2 + INFO_W;
constexpr int FRAME_H = PANEL_H;
constexpr float VIEW_HALF = 13.6f;

struct Pose { float x, y, th; };
struct Rect { float x0, y0, x1, y1; };
struct Edge { int i, j; float zx, zy, zt; float info_xy, info_th; };

__host__ __device__ static inline float clampf(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}
static inline float wrap_angle(float a) {
    while (a > PI_F) a -= 2.0f * PI_F;
    while (a < -PI_F) a += 2.0f * PI_F;
    return a;
}

// SE(2) compose: b = a (+) z, with z expressed in a's frame.
static inline Pose compose(const Pose& a, const Pose& z) {
    float c = std::cos(a.th), s = std::sin(a.th);
    return {a.x + c * z.x - s * z.y, a.y + s * z.x + c * z.y, wrap_angle(a.th + z.th)};
}
// SE(2) relative: z such that b = a (+) z  (i.e. z = a^-1 (+) b).
static inline Pose relative(const Pose& a, const Pose& b) {
    float c = std::cos(a.th), s = std::sin(a.th);
    float dxw = b.x - a.x, dyw = b.y - a.y;
    return {c * dxw + s * dyw, -s * dxw + c * dyw, wrap_angle(b.th - a.th)};
}

// --- Map (true environment, used only for raycasting the simulated sensor) ---
static std::vector<Rect> make_rects() {
    // Asymmetric indoor floor-plan built from thin wall slabs + pillars so the
    // scans see distinct geometry at every point of the lap (no aliasing) and
    // there are no large filled interiors.
    std::vector<Rect> r;
    r.push_back({-13.5f, 13.2f, 13.5f, 13.5f});   // perimeter
    r.push_back({-13.5f, -13.5f, 13.5f, -13.2f});
    r.push_back({-13.5f, -13.5f, -13.2f, 13.5f});
    r.push_back({13.2f, -13.5f, 13.5f, 13.5f});
    r.push_back({-8.5f, 7.0f, 2.0f, 7.3f});       // internal walls
    r.push_back({5.0f, 4.5f, 5.3f, 12.5f});
    r.push_back({-11.5f, -2.5f, -11.2f, 8.5f});
    r.push_back({-6.5f, -9.0f, 3.0f, -8.7f});
    r.push_back({7.0f, -9.5f, 10.0f, -9.2f});
    r.push_back({9.7f, -9.5f, 10.0f, -4.5f});
    r.push_back({-3.0f, 0.5f, -2.2f, 1.3f});      // pillars
    r.push_back({2.6f, -3.2f, 3.4f, -2.4f});
    r.push_back({-0.4f, 9.5f, 0.4f, 10.3f});
    return r;
}

static bool is_wall(const std::vector<Rect>& rects, float x, float y) {
    if (x <= -WORLD_HALF || x >= WORLD_HALF || y <= -WORLD_HALF || y >= WORLD_HALF) return true;
    for (const Rect& r : rects)
        if (x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1) return true;
    return false;
}

// Ground-truth pose at lap parameter k.
static Pose gt_pose(int k) {
    float u = (2.0f * PI_F * k) / N_KF;  // k=N_KF returns to k=0
    float x = ELLIPSE_A * std::cos(u);
    float y = ELLIPSE_B * std::sin(u);
    float dx = -ELLIPSE_A * std::sin(u);
    float dy = ELLIPSE_B * std::cos(u);
    return {x, y, std::atan2(dy, dx)};
}

// Host raycast: range scan from a pose, returned as sensor-frame endpoints.
// `full` keeps every ray (for the map render); `match` keeps only hits within
// MATCH_RANGE (used for scan matching).
static void make_scan(const std::vector<Rect>& rects, const Pose& p, unsigned seed,
                      std::vector<float>& fx, std::vector<float>& fy,
                      std::vector<float>& mx, std::vector<float>& my) {
    fx.clear(); fy.clear(); mx.clear(); my.clear();
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, SCAN_NOISE);
    float cth = std::cos(-p.th), sth = std::sin(-p.th);
    for (int i = 0; i < N_SCAN; ++i) {
        float a = p.th + (2.0f * PI_F * i) / N_SCAN;
        float ca = std::cos(a), sa = std::sin(a);
        float r = MAX_RANGE;
        for (float t = 0.10f; t < MAX_RANGE; t += 0.04f)
            if (is_wall(rects, p.x + t * ca, p.y + t * sa)) { r = t; break; }
        r = clampf(r + noise(rng), 0.15f, MAX_RANGE);
        float wx = r * ca, wy = r * sa;            // world displacement from sensor
        float lx = cth * wx - sth * wy;            // rotate into sensor frame
        float ly = sth * wx + cth * wy;
        fx.push_back(lx); fy.push_back(ly);
        if (r <= MATCH_RANGE) { mx.push_back(lx); my.push_back(ly); }
    }
}

// --- Local likelihood field (in a loop target's sensor frame) ---------------
__host__ __device__ static inline float sample_local(const float* lut, float lx, float ly) {
    float fx = (lx + LF_HALF) / LF_RES - 0.5f;
    float fy = (ly + LF_HALF) / LF_RES - 0.5f;
    int ix = static_cast<int>(floorf(fx));
    int iy = static_cast<int>(floorf(fy));
    if (ix < 0 || ix >= LGRID - 1 || iy < 0 || iy >= LGRID - 1) return 0.0f;
    float tx = fx - ix, ty = fy - iy;
    const float* row0 = lut + iy * LGRID + ix;
    const float* row1 = row0 + LGRID;
    float a = row0[0] * (1 - tx) + row0[1] * tx;
    float b = row1[0] * (1 - tx) + row1[1] * tx;
    return a * (1 - ty) + b * ty;
}

// Distance-transform an occupancy raster (0 = occupied surface) to an exp()
// likelihood field.
static std::vector<float> field_from_occ(const cv::Mat& occ) {
    cv::Mat dist_px;
    cv::distanceTransform(occ, dist_px, cv::DIST_L2, 3);
    std::vector<float> lut(LGRID * LGRID);
    float inv2s2 = 1.0f / (2.0f * FIELD_SIGMA * FIELD_SIGMA);
    for (int iy = 0; iy < LGRID; ++iy)
        for (int ix = 0; ix < LGRID; ++ix) {
            float d = dist_px.at<float>(iy, ix) * LF_RES;
            lut[iy * LGRID + ix] = std::exp(-d * d * inv2s2);
        }
    return lut;
}

static inline void raster_point(cv::Mat& occ, float lx, float ly) {
    int ix = static_cast<int>((lx + LF_HALF) / LF_RES);
    int iy = static_cast<int>((ly + LF_HALF) / LF_RES);
    if (ix >= 0 && ix < LGRID && iy >= 0 && iy < LGRID) occ.at<unsigned char>(iy, ix) = 0;
}

// SINGLE-SCAN field (the #121 baseline): rasterise just the target keyframe's
// own in-range endpoints, in its sensor frame.
static std::vector<float> build_scan_field(const std::vector<float>& mx,
                                           const std::vector<float>& my) {
    cv::Mat occ(LGRID, LGRID, CV_8U, cv::Scalar(255));  // 255 = free for DT
    for (size_t k = 0; k < mx.size(); ++k) raster_point(occ, mx[k], my[k]);
    return field_from_occ(occ);
}

// SUBMAP field (this demo): rasterise the in-range endpoints of every keyframe
// in the target's submap, each transformed into the target keyframe's sensor
// frame by the live estimate's intra-submap relpose.  Fusing many viewpoints
// fills in the surfaces -> denser, less view-dependent field.
static std::vector<float> build_submap_field(int target, const std::vector<Pose>& est,
                                             const std::vector<std::vector<float>>& mx,
                                             const std::vector<std::vector<float>>& my) {
    int s0 = (target / SUBMAP_KF) * SUBMAP_KF;
    int s1 = std::min(s0 + SUBMAP_KF, N_KF);
    cv::Mat occ(LGRID, LGRID, CV_8U, cv::Scalar(255));
    for (int m = s0; m < s1; ++m) {
        Pose rel = relative(est[target], est[m]);  // m's pose in target's frame
        float c = std::cos(rel.th), s = std::sin(rel.th);
        for (size_t t = 0; t < mx[m].size(); ++t) {
            float lx = rel.x + c * mx[m][t] - s * my[m][t];
            float ly = rel.y + s * mx[m][t] + c * my[m][t];
            raster_point(occ, lx, ly);
        }
    }
    return field_from_occ(occ);
}

// --- CSM scoring: one thread scores one candidate relative pose -------------
__global__ void csm_kernel(const float* __restrict__ sx, const float* __restrict__ sy,
                           int n_pts, const float* __restrict__ lut,
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
    for (int k = 0; k < n_pts; ++k) {
        float wx = px + c * sx[k] - s * sy[k];
        float wy = py + s * sx[k] + c * sy[k];
        acc += sample_local(lut, wx, wy);
    }
    score[idx] = acc;
}

struct LoopResult { Pose rel; float score; };  // score normalised to [0, 1]

// GPU coarse-to-fine exhaustive search of the relpose, centred on `init`.
static LoopResult run_gpu_loop_csm(const float* d_sx, const float* d_sy, int n_pts,
                                   const float* d_lut, const Pose& init,
                                   float* d_score, std::vector<float>& h_score) {
    auto search = [&](Pose center, int nxy, int nt, float rxy, float rth) -> std::pair<Pose, float> {
        int total = nxy * nxy * nt;
        int blocks = (total + THREADS - 1) / THREADS;
        csm_kernel<<<blocks, THREADS>>>(d_sx, d_sy, n_pts, d_lut,
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
        Pose p{center.x + (ix - (nxy - 1) * 0.5f) * rxy,
               center.y + (iy - (nxy - 1) * 0.5f) * rxy,
               wrap_angle(center.th + (it - (nt - 1) * 0.5f) * rth)};
        return {p, h_score[best]};
    };
    auto coarse = search(init, COARSE_NXY, COARSE_NT, COARSE_RES_XY, COARSE_RES_TH);
    auto fine = search(coarse.first, FINE_NXY, FINE_NT, FINE_RES_XY, FINE_RES_TH);
    return {fine.first, n_pts > 0 ? fine.second / n_pts : 0.0f};
}

// CPU reference of the same coarse search (single grid), for the speedup line.
// Returns the elapsed time; `best_out` receives the best score so the search is
// not eliminated as dead code.
static double cpu_loop_csm_ms(const std::vector<float>& sx, const std::vector<float>& sy,
                              const std::vector<float>& lut, const Pose& init, float* best_out) {
    auto t0 = std::chrono::high_resolution_clock::now();
    float best = -1.0f;
    int n = static_cast<int>(sx.size());
    for (int it = 0; it < COARSE_NT; ++it) {
        float pth = init.th + (it - (COARSE_NT - 1) * 0.5f) * COARSE_RES_TH;
        float c = std::cos(pth), s = std::sin(pth);
        for (int iy = 0; iy < COARSE_NXY; ++iy) {
            float py = init.y + (iy - (COARSE_NXY - 1) * 0.5f) * COARSE_RES_XY;
            for (int ix = 0; ix < COARSE_NXY; ++ix) {
                float px = init.x + (ix - (COARSE_NXY - 1) * 0.5f) * COARSE_RES_XY;
                float acc = 0.0f;
                for (int k = 0; k < n; ++k) {
                    float wx = px + c * sx[k] - s * sy[k];
                    float wy = py + s * sx[k] + c * sy[k];
                    acc += sample_local(lut.data(), wx, wy);
                }
                if (acc > best) best = acc;
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    *best_out = best;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// --- Dense SE(2) pose-graph Gauss-Newton back-end (small graph, host) -------
// Solve A x = b for SPD A (row-major n x n), result in b. Returns false if not PD.
static bool chol_solve(std::vector<double>& A, std::vector<double>& b, int n) {
    for (int j = 0; j < n; ++j) {
        double sum = A[j * n + j];
        for (int k = 0; k < j; ++k) sum -= A[j * n + k] * A[j * n + k];
        if (sum <= 0.0) return false;
        double Ljj = std::sqrt(sum);
        A[j * n + j] = Ljj;
        for (int i = j + 1; i < n; ++i) {
            double s = A[i * n + j];
            for (int k = 0; k < j; ++k) s -= A[i * n + k] * A[j * n + k];
            A[i * n + j] = s / Ljj;
        }
    }
    for (int i = 0; i < n; ++i) {            // forward: L y = b
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= A[i * n + k] * b[k];
        b[i] = s / A[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {       // back: L^T x = y
        double s = b[i];
        for (int k = i + 1; k < n; ++k) s -= A[k * n + i] * b[k];
        b[i] = s / A[i * n + i];
    }
    return true;
}

static void optimise_graph(std::vector<Pose>& poses, const std::vector<Edge>& edges,
                           const Pose& anchor, int n_active) {
    int n = n_active * 3;
    std::vector<double> H(n * n), g(n);
    for (int gn = 0; gn < GN_ITERS; ++gn) {
        std::fill(H.begin(), H.end(), 0.0);
        std::fill(g.begin(), g.end(), 0.0);
        // Anchor pose 0 to its fixed value (gauge fix).
        {
            double dx = poses[0].x - anchor.x, dy = poses[0].y - anchor.y;
            double dt = wrap_angle(poses[0].th - anchor.th);
            H[0 * n + 0] += ANCHOR_INFO; H[1 * n + 1] += ANCHOR_INFO; H[2 * n + 2] += ANCHOR_INFO;
            g[0] += ANCHOR_INFO * dx; g[1] += ANCHOR_INFO * dy; g[2] += ANCHOR_INFO * dt;
        }
        for (const Edge& e : edges) {
            if (e.i >= n_active || e.j >= n_active) continue;
            const Pose& pi = poses[e.i];
            const Pose& pj = poses[e.j];
            float c = std::cos(pi.th), s = std::sin(pi.th);
            float dxw = pj.x - pi.x, dyw = pj.y - pi.y;
            float dx = dxw * c + dyw * s;
            float dy = -dxw * s + dyw * c;
            float rt = wrap_angle(wrap_angle(pj.th - pi.th) - e.zt);
            float rx = dx - e.zx, ry = dy - e.zy;
            // Jacobians wrt pose_i (Ji) and pose_j (Jj), rows = (rx, ry, rt).
            double Ji[9] = {-c, -s, -dxw * s + dyw * c,
                             s, -c, -dxw * c - dyw * s,
                             0, 0, -1};
            double Jj[9] = { c,  s, 0,
                            -s,  c, 0,
                             0,  0, 1};
            double w[3] = {e.info_xy, e.info_xy, e.info_th};
            double r[3] = {rx, ry, rt};
            int bi = e.i * 3, bj = e.j * 3;
            // Accumulate H and g blocks: H += J^T W J, g += J^T W r.
            auto add = [&](const double* Ja, int ba, const double* Jb, int bb) {
                for (int p = 0; p < 3; ++p)
                    for (int q = 0; q < 3; ++q) {
                        double v = 0.0;
                        for (int kk = 0; kk < 3; ++kk) v += Ja[3 * kk + p] * w[kk] * Jb[3 * kk + q];
                        H[(ba + p) * n + (bb + q)] += v;
                    }
            };
            add(Ji, bi, Ji, bi); add(Jj, bj, Jj, bj);
            add(Ji, bi, Jj, bj); add(Jj, bj, Ji, bi);
            for (int p = 0; p < 3; ++p) {
                double gi = 0, gj = 0;
                for (int kk = 0; kk < 3; ++kk) { gi += Ji[3 * kk + p] * w[kk] * r[kk];
                                                 gj += Jj[3 * kk + p] * w[kk] * r[kk]; }
                g[bi + p] += gi; g[bj + p] += gj;
            }
        }
        for (int d = 0; d < n; ++d) H[d * n + d] += 1.0e-3;  // Levenberg ridge
        std::vector<double> dx = g;
        if (!chol_solve(H, dx, n)) break;
        for (int k = 0; k < n_active; ++k) {
            poses[k].x -= static_cast<float>(dx[3 * k + 0]);
            poses[k].y -= static_cast<float>(dx[3 * k + 1]);
            poses[k].th = wrap_angle(poses[k].th - static_cast<float>(dx[3 * k + 2]));
        }
        poses[0] = anchor;  // hard gauge fix
    }
}

// --- A SLAM run with one choice of loop-closure field -----------------------
// Both variants share the odometry chain; they differ only in the likelihood
// field a loop candidate is matched against (single scan vs fused submap).
struct Variant {
    std::vector<Pose> est;
    std::vector<Edge> edges;
    std::vector<Edge> loop_edges;                 // (i, j) only, for drawing
    std::unordered_map<int, std::vector<float>> field_cache;
    int proposed = 0, accepted = 0, rejected = 0;
    float last_score = 0.0f;
    bool use_submap = false;
};

// --- Visualization ----------------------------------------------------------
static cv::Point world_to_panel(int ox, float x, float y) {
    int px = ox + static_cast<int>((x + VIEW_HALF) / (2 * VIEW_HALF) * PANEL_W);
    int py = static_cast<int>((VIEW_HALF - y) / (2 * VIEW_HALF) * PANEL_H);
    return cv::Point(px, py);
}
static inline void put_pt(cv::Mat& img, int ox, float x, float y, const cv::Vec3b& col) {
    cv::Point p = world_to_panel(ox, x, y);
    if (p.x > ox && p.x < ox + PANEL_W - 1 && p.y > 0 && p.y < PANEL_H - 1)
        img.at<cv::Vec3b>(p.y, p.x) = col;
}

// Distinct per-submap tint so the submap partition is visible on the SLAM map.
static cv::Vec3b submap_tint(int submap) {
    static const cv::Vec3b pal[6] = {
        {150, 175, 120}, {120, 150, 195}, {165, 135, 170},
        {120, 175, 175}, {175, 160, 110}, {150, 145, 150}};
    return pal[submap % 6];
}

// Accumulate the in-range scan points of all keyframes <= k, placed by `poses`.
// `tint` colours each point by its submap; otherwise a flat colour is used.
static void draw_map(cv::Mat& img, int ox, const std::vector<Pose>& poses, int k,
                     const std::vector<std::vector<float>>& mx,
                     const std::vector<std::vector<float>>& my,
                     const cv::Vec3b& pt_col, const cv::Scalar& traj_col, bool tint) {
    cv::rectangle(img, cv::Rect(ox, 0, PANEL_W, PANEL_H), cv::Scalar(248, 248, 245), -1);
    for (int j = 0; j <= k; ++j) {
        const Pose& p = poses[j];
        float c = std::cos(p.th), s = std::sin(p.th);
        cv::Vec3b col = tint ? submap_tint(j / SUBMAP_KF) : pt_col;
        for (size_t t = 0; t < mx[j].size(); ++t) {
            float wx = p.x + c * mx[j][t] - s * my[j][t];
            float wy = p.y + s * mx[j][t] + c * my[j][t];
            put_pt(img, ox, wx, wy, col);
        }
    }
    for (int j = 1; j <= k; ++j)
        cv::line(img, world_to_panel(ox, poses[j - 1].x, poses[j - 1].y),
                 world_to_panel(ox, poses[j].x, poses[j].y), traj_col, 2, cv::LINE_AA);
    cv::circle(img, world_to_panel(ox, poses[k].x, poses[k].y), 4, traj_col, -1, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(ox + 1, 1, PANEL_W - 2, PANEL_H - 2), cv::Scalar(120, 124, 130), 1);
}

struct Stats {
    double gpu_ms_once = 0.0, cpu_ms_once = 0.0;
    long long cand_per_attempt = 0;
};

static float ate_rmse(const std::vector<Pose>& poses, int k) {
    double s = 0.0;
    for (int j = 0; j <= k; ++j) {
        Pose g = gt_pose(j);
        double dx = poses[j].x - g.x, dy = poses[j].y - g.y;
        s += dx * dx + dy * dy;
    }
    return std::sqrt(s / (k + 1));
}

static void draw_info(cv::Mat& img, int ox, int k, float odom_ate,
                      const Variant& vs, const Variant& vm, float scan_ate, float submap_ate,
                      const Stats& s, const std::vector<float>& odom_hist,
                      const std::vector<float>& scan_hist, const std::vector<float>& submap_hist) {
    cv::rectangle(img, cv::Rect(ox, 0, INFO_W, FRAME_H), cv::Scalar(244, 246, 246), -1);
    cv::putText(img, "scan-to-submap SLAM", cv::Point(ox + 14, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.56, cv::Scalar(20, 28, 35), 2, cv::LINE_AA);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "keyframe %d / %d   submap %d", k, N_KF - 1, k / SUBMAP_KF);
    cv::putText(img, buf, cv::Point(ox + 14, 56), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(60, 66, 74), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "dead-reckon ATE:  %.2f m", odom_ate);
    cv::putText(img, buf, cv::Point(ox + 14, 82), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                cv::Scalar(60, 70, 210), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "scan-to-scan ATE: %.2f m", scan_ate);
    cv::putText(img, buf, cv::Point(ox + 14, 104), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                cv::Scalar(70, 130, 200), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "scan-to-submap:   %.2f m", submap_ate);
    cv::putText(img, buf, cv::Point(ox + 14, 126), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                cv::Scalar(45, 150, 60), 1, cv::LINE_AA);

    // ATE-vs-keyframe plot.
    int px0 = ox + 14, py0 = 146, pw = INFO_W - 36, ph = 98;
    cv::rectangle(img, cv::Rect(px0, py0, pw, ph), cv::Scalar(255, 255, 255), -1);
    cv::rectangle(img, cv::Rect(px0, py0, pw, ph), cv::Scalar(200, 204, 210), 1);
    float ymax = 4.0f;
    auto plot = [&](const std::vector<float>& h, const cv::Scalar& col) {
        for (size_t i = 1; i < h.size(); ++i) {
            float x0 = px0 + pw * (i - 1) / (float)(N_KF - 1);
            float x1 = px0 + pw * i / (float)(N_KF - 1);
            float y0 = py0 + ph - ph * clampf(h[i - 1] / ymax, 0, 1);
            float y1 = py0 + ph - ph * clampf(h[i] / ymax, 0, 1);
            cv::line(img, cv::Point((int)x0, (int)y0), cv::Point((int)x1, (int)y1), col, 2, cv::LINE_AA);
        }
    };
    plot(odom_hist, cv::Scalar(60, 70, 210));
    plot(scan_hist, cv::Scalar(70, 130, 200));
    plot(submap_hist, cv::Scalar(45, 150, 60));
    cv::putText(img, "ATE vs keyframe (0-4 m)", cv::Point(px0, py0 - 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(70, 76, 84), 1, cv::LINE_AA);

    int y = py0 + ph + 28;
    std::snprintf(buf, sizeof(buf), "loops: submap %d  (rej %d)", vm.accepted, vm.rejected);
    cv::putText(img, buf, cv::Point(ox + 14, y), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(45, 120, 55), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "       scan   %d  (rej %d)", vs.accepted, vs.rejected);
    cv::putText(img, buf, cv::Point(ox + 14, y + 22), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(70, 110, 160), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "last submap score: %.2f", vm.last_score);
    cv::putText(img, buf, cv::Point(ox + 14, y + 44), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "candidates/attempt: %.2fM", s.cand_per_attempt / 1e6);
    cv::putText(img, buf, cv::Point(ox + 14, y + 66), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "GPU CSM: %.2f ms (CPU %.0f)", s.gpu_ms_once, s.cpu_ms_once);
    cv::putText(img, buf, cv::Point(ox + 14, y + 88), cv::FONT_HERSHEY_SIMPLEX, 0.44,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    if (s.cpu_ms_once > 0 && s.gpu_ms_once > 0) {
        std::snprintf(buf, sizeof(buf), "speedup: %.0fx", s.cpu_ms_once / s.gpu_ms_once);
        cv::putText(img, buf, cv::Point(ox + 14, y + 110), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                    cv::Scalar(45, 95, 175), 1, cv::LINE_AA);
    }
}

static void ensure_dirs() {
    int rc = std::system("mkdir -p gif tmp");
    if (rc != 0) std::fprintf(stderr, "mkdir failed with code %d\n", rc);
}

// Shared loop-closure step for one variant at keyframe k.  Returns true if any
// loop was accepted (so the caller re-optimises).  `do_cpu_timing` runs the
// one-off CPU reference on the first real attempt of the submap variant.
static bool process_variant(Variant& v, int k,
                            const std::vector<std::vector<float>>& mx,
                            const std::vector<std::vector<float>>& my,
                            float* d_sx, float* d_sy, float* d_lut, float* d_score,
                            std::vector<float>& h_score, Stats& stats, bool do_cpu_timing) {
    if (k < LC_MIN_GAP) return false;
    auto get_field = [&](int o) -> const std::vector<float>& {
        auto it = v.field_cache.find(o);
        if (it == v.field_cache.end()) {
            std::vector<float> f = v.use_submap ? build_submap_field(o, v.est, mx, my)
                                                : build_scan_field(mx[o], my[o]);
            it = v.field_cache.emplace(o, std::move(f)).first;
        }
        return it->second;
    };

    std::vector<std::pair<float, int>> cand;
    for (int o = 0; o <= k - LC_MIN_GAP; ++o) {
        float dx = v.est[o].x - v.est[k].x, dy = v.est[o].y - v.est[k].y;
        float d2 = dx * dx + dy * dy;
        if (d2 < LC_GATE_R * LC_GATE_R) cand.push_back({d2, o});
    }
    std::sort(cand.begin(), cand.end());

    bool accepted_any = false;
    int tried = 0;
    for (auto& cc : cand) {
        if (tried >= LC_MAX_CAND) break;
        int o = cc.second;
        ++tried;
        ++v.proposed;
        const std::vector<float>& lut = get_field(o);
        CUDA_CHECK(cudaMemcpy(d_lut, lut.data(), lut.size() * sizeof(float), cudaMemcpyHostToDevice));
        int n_pts = static_cast<int>(mx[k].size());
        CUDA_CHECK(cudaMemcpy(d_sx, mx[k].data(), n_pts * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sy, my[k].data(), n_pts * sizeof(float), cudaMemcpyHostToDevice));
        Pose rel_init = relative(v.est[o], v.est[k]);  // estimate-predicted relpose

        auto g0 = std::chrono::high_resolution_clock::now();
        LoopResult lr = run_gpu_loop_csm(d_sx, d_sy, n_pts, d_lut, rel_init, d_score, h_score);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto g1 = std::chrono::high_resolution_clock::now();
        double gpu_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
        v.last_score = lr.score;

        // Time GPU and CPU on the SAME (first) query so the speedup is apples-to-apples.
        if (do_cpu_timing && stats.cpu_ms_once == 0.0) {
            stats.gpu_ms_once = gpu_ms;
            float cpu_best = 0.0f;
            stats.cpu_ms_once = cpu_loop_csm_ms(mx[k], my[k], lut, rel_init, &cpu_best);
            std::printf("CPU CSM check: best score sum %.2f over %d candidates\n",
                        cpu_best, COARSE_NXY * COARSE_NXY * COARSE_NT);
        }

        if (lr.score >= LC_ACCEPT) {
            v.edges.push_back({o, k, lr.rel.x, lr.rel.y, lr.rel.th, LOOP_INFO_XY, LOOP_INFO_TH});
            v.loop_edges.push_back({o, k, 0, 0, 0, 0, 0});
            ++v.accepted;
            accepted_any = true;
        } else {
            ++v.rejected;
        }
    }
    return accepted_any;
}

static Stats run_demo() {
    ensure_dirs();
    std::vector<Rect> rects = make_rects();

    // Simulate keyframe scans + a drifting odometry chain.
    std::vector<std::vector<float>> mx(N_KF), my(N_KF);  // matchable scan points
    std::vector<Pose> odom(N_KF);                         // dead reckoning
    std::vector<Pose> odom_z(N_KF);                       // odom_z[k] = edge (k-1)->k
    {
        std::mt19937 rng(20260526u);
        std::normal_distribution<float> nxy(0.0f, ODOM_SIGMA_XY), nth(0.0f, ODOM_SIGMA_TH);
        Pose g0 = gt_pose(0);
        odom[0] = g0;
        std::vector<float> fx, fy;
        make_scan(rects, g0, 1000, fx, fy, mx[0], my[0]);
        for (int k = 1; k < N_KF; ++k) {
            Pose gprev = gt_pose(k - 1), gcur = gt_pose(k);
            Pose ztrue = relative(gprev, gcur);
            Pose z{ztrue.x + nxy(rng), ztrue.y + nxy(rng),
                   wrap_angle(ztrue.th + nth(rng) + ODOM_BIAS_TH)};
            odom_z[k] = z;
            odom[k] = compose(odom[k - 1], z);
            make_scan(rects, gcur, 1000u + k, fx, fy, mx[k], my[k]);
        }
    }

    // Two SLAM variants share the odometry chain; the only difference is the
    // loop-closure field (single scan vs fused submap).
    Variant vs, vm;
    vs.use_submap = false;
    vm.use_submap = true;
    vs.est = vm.est = odom;  // both start from dead reckoning

    // Device buffers for CSM.
    float *d_sx = nullptr, *d_sy = nullptr, *d_lut = nullptr, *d_score = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sx, N_SCAN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sy, N_SCAN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lut, LGRID * LGRID * sizeof(float)));
    int max_total = COARSE_NXY * COARSE_NXY * COARSE_NT;
    CUDA_CHECK(cudaMalloc(&d_score, max_total * sizeof(float)));

    cv::VideoWriter video("tmp/gpu_csm_submap_slam.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 12,
                          cv::Size(FRAME_W, FRAME_H));
    if (!video.isOpened()) { std::fprintf(stderr, "failed to open video\n"); std::exit(1); }

    Stats stats;
    stats.cand_per_attempt = (long long)max_total + (long long)FINE_NXY * FINE_NXY * FINE_NT;
    std::vector<float> h_score;
    std::vector<float> odom_hist, scan_hist, submap_hist;
    const cv::Vec3b odom_pt(150, 150, 215);

    for (int k = 1; k < N_KF; ++k) {
        // Extend both graphs with this keyframe's odometry edge.
        Edge oe{k - 1, k, odom_z[k].x, odom_z[k].y, odom_z[k].th, ODOM_INFO_XY, ODOM_INFO_TH};
        vs.edges.push_back(oe);
        vm.edges.push_back(oe);

        // Loop-closure detection for both variants (submap first, so its first
        // real attempt carries the one-off CPU timing).
        bool acc_m = process_variant(vm, k, mx, my, d_sx, d_sy, d_lut, d_score, h_score, stats, true);
        bool acc_s = process_variant(vs, k, mx, my, d_sx, d_sy, d_lut, d_score, h_score, stats, false);
        if (acc_m) optimise_graph(vm.est, vm.edges, gt_pose(0), k + 1);
        if (acc_s) optimise_graph(vs.est, vs.edges, gt_pose(0), k + 1);

        float odom_ate = ate_rmse(odom, k);
        float scan_ate = ate_rmse(vs.est, k);
        float submap_ate = ate_rmse(vm.est, k);
        odom_hist.push_back(odom_ate);
        scan_hist.push_back(scan_ate);
        submap_hist.push_back(submap_ate);

        // Render (stride to keep the GIF small; always render around closures).
        bool render = (k % 2 == 0) || acc_m || acc_s || k == N_KF - 1;
        if (render) {
            cv::Mat frame(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(244, 246, 246));
            draw_map(frame, 0, odom, k, mx, my, odom_pt, cv::Scalar(60, 70, 210), false);
            cv::putText(frame, "dead reckoning (odometry)", cv::Point(12, 26),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 24, 32), 2, cv::LINE_AA);
            draw_map(frame, PANEL_W, vm.est, k, mx, my, odom_pt, cv::Scalar(45, 150, 60), true);
            for (const Edge& le : vm.loop_edges)
                cv::line(frame, world_to_panel(PANEL_W, vm.est[le.i].x, vm.est[le.i].y),
                         world_to_panel(PANEL_W, vm.est[le.j].x, vm.est[le.j].y),
                         cv::Scalar(200, 180, 40), 1, cv::LINE_AA);
            cv::putText(frame, "scan-to-submap SLAM", cv::Point(PANEL_W + 12, 26),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 24, 32), 2, cv::LINE_AA);
            draw_info(frame, PANEL_W * 2, k, odom_ate, vs, vm, scan_ate, submap_ate,
                      stats, odom_hist, scan_hist, submap_hist);
            int holds = (acc_m || acc_s) ? 5 : 1;
            for (int h = 0; h < holds; ++h) video.write(frame);
        }

        if (k % 10 == 0 || acc_m || acc_s)
            std::printf("kf %3d  odom=%.3f  scan=%.3f  submap=%.3f  loops s/m=%d/%d\n",
                        k, odom_ate, scan_ate, submap_ate, vs.accepted, vm.accepted);
    }

    // Hold the final corrected map.
    {
        cv::Mat frame(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(244, 246, 246));
        draw_map(frame, 0, odom, N_KF - 1, mx, my, odom_pt, cv::Scalar(60, 70, 210), false);
        cv::putText(frame, "dead reckoning (odometry)", cv::Point(12, 26),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 24, 32), 2, cv::LINE_AA);
        draw_map(frame, PANEL_W, vm.est, N_KF - 1, mx, my, odom_pt, cv::Scalar(45, 150, 60), true);
        for (const Edge& le : vm.loop_edges)
            cv::line(frame, world_to_panel(PANEL_W, vm.est[le.i].x, vm.est[le.i].y),
                     world_to_panel(PANEL_W, vm.est[le.j].x, vm.est[le.j].y),
                     cv::Scalar(200, 180, 40), 1, cv::LINE_AA);
        cv::putText(frame, "scan-to-submap SLAM", cv::Point(PANEL_W + 12, 26),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 24, 32), 2, cv::LINE_AA);
        draw_info(frame, PANEL_W * 2, N_KF - 1, ate_rmse(odom, N_KF - 1), vs, vm,
                  ate_rmse(vs.est, N_KF - 1), ate_rmse(vm.est, N_KF - 1), stats,
                  odom_hist, scan_hist, submap_hist);
        for (int h = 0; h < 30; ++h) video.write(frame);
    }

    video.release();
    avi_to_gif("tmp/gpu_csm_submap_slam.avi", "gif/gpu_csm_submap_slam.gif", 12, 900);

    CUDA_CHECK(cudaFree(d_sx));
    CUDA_CHECK(cudaFree(d_sy));
    CUDA_CHECK(cudaFree(d_lut));
    CUDA_CHECK(cudaFree(d_score));

    std::printf("\nGPU CSM submap SLAM (detected loops, no ground-truth constraints)\n");
    std::printf("scan-to-scan   : %d proposed, %d accepted, %d rejected, ATE %.3f m\n",
                vs.proposed, vs.accepted, vs.rejected, ate_rmse(vs.est, N_KF - 1));
    std::printf("scan-to-submap : %d proposed, %d accepted, %d rejected, ATE %.3f m\n",
                vm.proposed, vm.accepted, vm.rejected, ate_rmse(vm.est, N_KF - 1));
    std::printf("dead reckoning : ATE %.3f m\n", ate_rmse(odom, N_KF - 1));
    return stats;
}

}  // namespace cudabot

int main() {
    cudabot::Stats s = cudabot::run_demo();
    std::printf("candidates/attempt: %.3f M (coarse-to-fine)\n", s.cand_per_attempt / 1e6);
    std::printf("GPU CSM %.2f ms vs CPU %.1f ms (%.0fx) per loop-closure attempt\n",
                s.gpu_ms_once, s.cpu_ms_once,
                s.cpu_ms_once > 0 && s.gpu_ms_once > 0 ? s.cpu_ms_once / s.gpu_ms_once : 0.0);
    std::printf("Wrote gif/gpu_csm_submap_slam.gif\n");
    return 0;
}
