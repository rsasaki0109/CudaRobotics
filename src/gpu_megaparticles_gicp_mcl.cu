// gpu_megaparticles_gicp_mcl.cu
//
// MegaParticles-style localization with a GICP-style distribution-to-
// distribution (D2D) scan likelihood.
//
// This is the "GICP-like point-cloud likelihood" follow-up flagged in the
// MegaParticles line (#86 Stein, #101 LSH, #104 6-DoF).  The earlier demos
// scored a range scan against a precomputed distance field: every scan
// endpoint is penalised by its isotropic distance to the nearest wall.  That
// is cheap but blurs surface structure -- a point sliding ALONG a wall is
// penalised as hard as a point moving INTO the wall.
//
// Here the map is instead a point cloud with a per-point surface-aware
// covariance (the GICP "disk": small variance along the surface normal, large
// along the tangent).  Each particle transforms its scan into the world,
// matches every endpoint to the nearest map point through a uniform grid
// index, and scores the Mahalanobis residual under the combined covariance
//      M = (C_map + R C_scan R^T)^{-1}                (Segal et al., RSS 2009)
// summed over the scan.  This is point-to-line / distribution-to-distribution:
// the cost barely grows for tangential slip but rises sharply for normal-
// direction error, which is the correct probabilistic weight near walls.
//
// To isolate the likelihood we run a controlled head-to-head: two filters,
// each 1,048,576 particles, sharing the IDENTICAL MegaParticles machinery
// (global uniform init, Gauss-Newton particle motion, sparse bucket-neighbor
// Stein attraction/repulsion, posterior smoothing, representative-state gate,
// hidden kidnap + scan blackout recovery).  Only the per-particle scoring
// kernel differs: distance-field proxy (the #86 likelihood) vs GICP D2D.
//
// The D2D likelihood is evaluated for one million particles by indexing the
// map cloud with a uniform grid so each endpoint's nearest-neighbor lookup
// touches only a 3x3 cell neighborhood.
//
// Output: gif/gpu_megaparticles_gicp_mcl.gif

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
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

constexpr int K_MEGA = 1 << 20;
constexpr int THREADS = 256;
constexpr int N_SCAN = 30;
constexpr int N_STEPS = 128;
constexpr int KIDNAP_STEP = 56;
constexpr int OCCLUDE_STEPS = 15;
constexpr int VIDEO_EVERY = 2;
constexpr int STEIN_ITERS = 2;
constexpr int POST_PROP_ITERS = 2;
constexpr float PI_F = 3.14159265358979323846f;
constexpr float WORLD_W = 34.0f;
constexpr float WORLD_H = 24.0f;
constexpr float GRID_RES = 0.10f;
constexpr int GRID_W = static_cast<int>(WORLD_W / GRID_RES);
constexpr int GRID_H = static_cast<int>(WORLD_H / GRID_RES);
constexpr float MAX_RANGE = 18.0f;
constexpr float DT = 0.18f;
constexpr float DIST_SIGMA = 0.30f;
constexpr float MEGA_MOTION_SIGMA_XY = 0.060f;
constexpr float MEGA_MOTION_SIGMA_TH = 0.016f;
constexpr float OCCLUDED_SPREAD_XY = 0.95f;
constexpr float OCCLUDED_SPREAD_TH = 0.22f;
constexpr float LIK_TEMP = 0.72f;
constexpr int BUCKET_X = 48;
constexpr int BUCKET_Y = 34;
constexpr int BUCKET_T = 24;
constexpr int N_BUCKETS = BUCKET_X * BUCKET_Y * BUCKET_T;

// --- GICP D2D map-cloud / likelihood parameters --------------------------
constexpr float CLOUD_BIN = 0.15f;        // thinning spacing for the map cloud
constexpr float CLOUD_CELL = 0.70f;       // NN grid cell (>= match radius)
constexpr int CLOUD_GW = static_cast<int>(WORLD_W / CLOUD_CELL) + 1;
constexpr int CLOUD_GH = static_cast<int>(WORLD_H / CLOUD_CELL) + 1;
constexpr int CLOUD_NCELLS = CLOUD_GW * CLOUD_GH;
constexpr float MATCH_R = 0.68f;
constexpr float MATCH_R2 = MATCH_R * MATCH_R;
constexpr int CLOUD_KNN = 8;
constexpr float GICP_EPS_MAP = 0.04f;     // normal/tangent eigenvalue ratio (map)
constexpr float MAP_SURF_VAR = 0.0625f;   // tangent variance of the map disk (0.25 m)^2
constexpr float GICP_EPS_SCAN = 0.07f;
constexpr float SCAN_SURF_VAR = 0.0169f;  // tangent variance of the scan disk (0.13 m)^2
constexpr float SENSOR_VAR = 0.0036f;     // isotropic sensor-noise floor (0.06 m)^2

constexpr int PANEL_W = 470;
constexpr int PANEL_H = 360;
constexpr int INFO_W = 330;
constexpr int FRAME_W = PANEL_W * 2 + INFO_W;
constexpr int FRAME_H = PANEL_H;

struct Pose2 {
    float x;
    float y;
    float th;
};

struct Rect {
    float x0;
    float y0;
    float x1;
    float y1;
};

struct StepSummary {
    Pose2 field{};
    Pose2 d2d{};
    float field_err = 0.0f;
    float d2d_err = 0.0f;
    bool scan_visible = true;
    double field_ms = 0.0;
    double d2d_ms = 0.0;
};

struct FinalStats {
    float field_post_rmse = 0.0f;
    float d2d_post_rmse = 0.0f;
    float final_field_err = 0.0f;
    float final_d2d_err = 0.0f;
    int field_reacq_step = -1;
    int d2d_reacq_step = -1;
    double field_ms = 0.0;
    double d2d_ms = 0.0;
    int cloud_points = 0;
};

__host__ __device__ static inline float clampf(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}

__host__ __device__ static inline float wrap_angle(float a) {
    while (a > PI_F) a -= 2.0f * PI_F;
    while (a < -PI_F) a += 2.0f * PI_F;
    return a;
}

static float pose_error_xy(const Pose2& a, const Pose2& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static Pose2 integrate_pose(Pose2 p, float v, float omega) {
    p.x = clampf(p.x + v * std::cos(p.th) * DT, 0.5f, WORLD_W - 0.5f);
    p.y = clampf(p.y + v * std::sin(p.th) * DT, 0.5f, WORLD_H - 0.5f);
    p.th = wrap_angle(p.th + omega * DT);
    return p;
}

struct CpuMap {
    std::vector<Rect> rects;
    std::vector<float> dist;
    std::vector<float> gx;
    std::vector<float> gy;
    cv::Mat occ;
};

static void add_rect(std::vector<Rect>& rects, float x0, float y0, float x1, float y1) {
    rects.push_back({std::min(x0, x1), std::min(y0, y1), std::max(x0, x1), std::max(y0, y1)});
}

static bool inside_rect(const Rect& r, float x, float y) {
    return x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1;
}

static bool is_wall_world(const std::vector<Rect>& rects, float x, float y) {
    if (x < 0.0f || x > WORLD_W || y < 0.0f || y > WORLD_H) return true;
    for (const Rect& r : rects) {
        if (inside_rect(r, x, y)) return true;
    }
    return false;
}

static CpuMap make_map() {
    CpuMap map;
    std::vector<Rect>& r = map.rects;

    add_rect(r, 0.0f, 0.0f, WORLD_W, 0.25f);
    add_rect(r, 0.0f, WORLD_H - 0.25f, WORLD_W, WORLD_H);
    add_rect(r, 0.0f, 0.0f, 0.25f, WORLD_H);
    add_rect(r, WORLD_W - 0.25f, 0.0f, WORLD_W, WORLD_H);

    add_rect(r, 7.8f, 0.0f, 8.1f, 5.7f);
    add_rect(r, 7.8f, 7.1f, 8.1f, 13.6f);
    add_rect(r, 7.8f, 15.0f, 8.1f, WORLD_H);

    add_rect(r, 16.2f, 0.0f, 16.5f, 4.3f);
    add_rect(r, 16.2f, 6.1f, 16.5f, 16.8f);
    add_rect(r, 16.2f, 18.5f, 16.5f, WORLD_H);

    add_rect(r, 24.2f, 0.0f, 24.5f, 8.8f);
    add_rect(r, 24.2f, 10.6f, 24.5f, 19.1f);
    add_rect(r, 24.2f, 21.1f, 24.5f, WORLD_H);

    add_rect(r, 0.0f, 7.2f, 5.4f, 7.5f);
    add_rect(r, 7.2f, 7.2f, 18.1f, 7.5f);
    add_rect(r, 20.2f, 7.2f, WORLD_W, 7.5f);

    add_rect(r, 5.7f, 15.4f, 12.2f, 15.7f);
    add_rect(r, 14.2f, 15.4f, 27.2f, 15.7f);
    add_rect(r, 29.1f, 15.4f, WORLD_W, 15.7f);

    add_rect(r, 3.3f, 18.8f, 4.7f, 20.0f);
    add_rect(r, 12.5f, 2.3f, 14.0f, 3.5f);
    add_rect(r, 19.5f, 20.2f, 21.8f, 21.0f);
    add_rect(r, 28.7f, 4.2f, 30.2f, 5.7f);
    add_rect(r, 30.0f, 18.6f, 31.8f, 19.1f);

    map.occ = cv::Mat(GRID_H, GRID_W, CV_8U, cv::Scalar(255));
    for (int iy = 0; iy < GRID_H; ++iy) {
        for (int ix = 0; ix < GRID_W; ++ix) {
            float x = (ix + 0.5f) * GRID_RES;
            float y = (iy + 0.5f) * GRID_RES;
            if (is_wall_world(r, x, y)) map.occ.at<unsigned char>(iy, ix) = 0;
        }
    }

    cv::Mat dist_px;
    cv::distanceTransform(map.occ, dist_px, cv::DIST_L2, 5);
    map.dist.resize(GRID_W * GRID_H);
    map.gx.resize(GRID_W * GRID_H);
    map.gy.resize(GRID_W * GRID_H);
    for (int iy = 0; iy < GRID_H; ++iy) {
        for (int ix = 0; ix < GRID_W; ++ix) {
            float d = dist_px.at<float>(iy, ix) * GRID_RES;
            map.dist[iy * GRID_W + ix] = std::min(d, 2.5f);
        }
    }
    for (int iy = 0; iy < GRID_H; ++iy) {
        for (int ix = 0; ix < GRID_W; ++ix) {
            int ix0 = std::max(0, ix - 1), ix1 = std::min(GRID_W - 1, ix + 1);
            int iy0 = std::max(0, iy - 1), iy1 = std::min(GRID_H - 1, iy + 1);
            float dx = (map.dist[iy * GRID_W + ix1] - map.dist[iy * GRID_W + ix0]) /
                       ((ix1 - ix0) * GRID_RES + 1.0e-6f);
            float dy = (map.dist[iy1 * GRID_W + ix] - map.dist[iy0 * GRID_W + ix]) /
                       ((iy1 - iy0) * GRID_RES + 1.0e-6f);
            map.gx[iy * GRID_W + ix] = dx;
            map.gy[iy * GRID_W + ix] = dy;
        }
    }
    return map;
}

// --- Map point cloud with per-point GICP disk covariance -----------------
//
// Boundary occupied cells (adjacent to free space) become surface points.
// The cloud is thinned to ~CLOUD_BIN spacing, each point gets a 2x2 disk
// covariance from its CLOUD_KNN nearest cloud neighbours, and the points are
// sorted into a CLOUD_CELL uniform grid for nearest-neighbour lookup.
struct CpuCloud {
    std::vector<float> x, y;          // sorted by grid cell
    std::vector<float> c00, c01, c11; // regularised disk covariance
    std::vector<int> cell_start, cell_end;
    int n = 0;
};

static int cloud_cell_of(float x, float y) {
    int cx = std::max(0, std::min(CLOUD_GW - 1, static_cast<int>(x / CLOUD_CELL)));
    int cy = std::max(0, std::min(CLOUD_GH - 1, static_cast<int>(y / CLOUD_CELL)));
    return cy * CLOUD_GW + cx;
}

static CpuCloud build_cloud(const CpuMap& map) {
    // 1. Collect boundary occupied cells as candidate surface points.
    std::vector<float> rx, ry;
    for (int iy = 1; iy < GRID_H - 1; ++iy) {
        for (int ix = 1; ix < GRID_W - 1; ++ix) {
            if (map.occ.at<unsigned char>(iy, ix) != 0) continue;
            bool border = map.occ.at<unsigned char>(iy, ix - 1) != 0 ||
                          map.occ.at<unsigned char>(iy, ix + 1) != 0 ||
                          map.occ.at<unsigned char>(iy - 1, ix) != 0 ||
                          map.occ.at<unsigned char>(iy + 1, ix) != 0;
            if (!border) continue;
            rx.push_back((ix + 0.5f) * GRID_RES);
            ry.push_back((iy + 0.5f) * GRID_RES);
        }
    }

    // 2. Thin to ~CLOUD_BIN spacing (keep one point per bin).
    int bw = static_cast<int>(WORLD_W / CLOUD_BIN) + 1;
    int bh = static_cast<int>(WORLD_H / CLOUD_BIN) + 1;
    std::vector<unsigned char> taken(bw * bh, 0);
    std::vector<float> tx, ty;
    for (size_t i = 0; i < rx.size(); ++i) {
        int bx = std::min(bw - 1, static_cast<int>(rx[i] / CLOUD_BIN));
        int by = std::min(bh - 1, static_cast<int>(ry[i] / CLOUD_BIN));
        int b = by * bw + bx;
        if (taken[b]) continue;
        taken[b] = 1;
        tx.push_back(rx[i]);
        ty.push_back(ry[i]);
    }
    int n = static_cast<int>(tx.size());

    // 3. Per-point disk covariance from CLOUD_KNN nearest cloud neighbours.
    std::vector<float> c00(n), c01(n), c11(n);
    for (int i = 0; i < n; ++i) {
        std::vector<std::pair<float, int>> nb;
        nb.reserve(n);
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            float dx = tx[j] - tx[i], dy = ty[j] - ty[i];
            nb.emplace_back(dx * dx + dy * dy, j);
        }
        int k = std::min(CLOUD_KNN, static_cast<int>(nb.size()));
        std::partial_sort(nb.begin(), nb.begin() + k, nb.end());
        float mx = tx[i], my = ty[i];
        int cnt = 1;
        for (int t = 0; t < k; ++t) { mx += tx[nb[t].second]; my += ty[nb[t].second]; ++cnt; }
        mx /= cnt; my /= cnt;
        float sxx = (tx[i] - mx) * (tx[i] - mx);
        float syy = (ty[i] - my) * (ty[i] - my);
        float sxy = (tx[i] - mx) * (ty[i] - my);
        for (int t = 0; t < k; ++t) {
            float dx = tx[nb[t].second] - mx, dy = ty[nb[t].second] - my;
            sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
        }
        float inv = 1.0f / cnt;
        sxx *= inv; syy *= inv; sxy *= inv;
        float tr = sxx + syy;
        float disc = std::sqrt(std::max(0.0f, 0.25f * tr * tr - (sxx * syy - sxy * sxy)));
        float l_small = 0.5f * tr - disc;
        float nx, ny;
        if (std::fabs(sxy) > 1e-9f) { nx = l_small - syy; ny = sxy; }
        else if (sxx < syy)        { nx = 1.0f; ny = 0.0f; }
        else                       { nx = 0.0f; ny = 1.0f; }
        float nn = std::sqrt(nx * nx + ny * ny);
        if (nn < 1e-9f) { nx = 1.0f; ny = 0.0f; } else { nx /= nn; ny /= nn; }
        float w = 1.0f - GICP_EPS_MAP;  // disk eigenvalues (eps, 1) scaled by MAP_SURF_VAR
        c00[i] = MAP_SURF_VAR * (1.0f - w * nx * nx) + SENSOR_VAR;
        c01[i] = MAP_SURF_VAR * (    - w * nx * ny);
        c11[i] = MAP_SURF_VAR * (1.0f - w * ny * ny) + SENSOR_VAR;
    }

    // 4. Sort points by NN grid cell and build cell ranges.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return cloud_cell_of(tx[a], ty[a]) < cloud_cell_of(tx[b], ty[b]);
    });
    CpuCloud cloud;
    cloud.n = n;
    cloud.x.resize(n); cloud.y.resize(n);
    cloud.c00.resize(n); cloud.c01.resize(n); cloud.c11.resize(n);
    cloud.cell_start.assign(CLOUD_NCELLS, 0);
    cloud.cell_end.assign(CLOUD_NCELLS, 0);
    for (int i = 0; i < n; ++i) {
        int o = order[i];
        cloud.x[i] = tx[o]; cloud.y[i] = ty[o];
        cloud.c00[i] = c00[o]; cloud.c01[i] = c01[o]; cloud.c11[i] = c11[o];
    }
    for (int c = 0; c < CLOUD_NCELLS; ++c) { cloud.cell_start[c] = 0; cloud.cell_end[c] = 0; }
    int idx = 0;
    for (int c = 0; c < CLOUD_NCELLS; ++c) {
        cloud.cell_start[c] = idx;
        while (idx < n && cloud_cell_of(cloud.x[idx], cloud.y[idx]) == c) ++idx;
        cloud.cell_end[c] = idx;
    }
    return cloud;
}

static float raycast_range(const std::vector<Rect>& rects, const Pose2& pose, float rel_angle) {
    float a = pose.th + rel_angle;
    for (float r = 0.12f; r < MAX_RANGE; r += 0.045f) {
        float x = pose.x + r * std::cos(a);
        float y = pose.y + r * std::sin(a);
        if (is_wall_world(rects, x, y)) return r;
    }
    return MAX_RANGE;
}

static void make_scan(const std::vector<Rect>& rects,
                      const Pose2& pose,
                      int step,
                      std::vector<float>& sx,
                      std::vector<float>& sy) {
    sx.resize(N_SCAN);
    sy.resize(N_SCAN);
    std::mt19937 rng(8000 + step * 17);
    std::normal_distribution<float> noise(0.0f, 0.035f);
    for (int i = 0; i < N_SCAN; ++i) {
        float frac = (N_SCAN == 1) ? 0.0f : static_cast<float>(i) / (N_SCAN - 1);
        float a = (-132.0f + 264.0f * frac) * PI_F / 180.0f;
        float r = clampf(raycast_range(rects, pose, a) + noise(rng), 0.2f, MAX_RANGE);
        sx[i] = r * std::cos(a);
        sy[i] = r * std::sin(a);
    }
}

// Per-scan-point disk covariance in the sensor frame (host, once per frame).
static void compute_scan_cov(const std::vector<float>& sx,
                             const std::vector<float>& sy,
                             std::vector<float>& c00,
                             std::vector<float>& c01,
                             std::vector<float>& c11) {
    int n = static_cast<int>(sx.size());
    c00.resize(n); c01.resize(n); c11.resize(n);
    const int KS = 5;
    for (int i = 0; i < n; ++i) {
        std::vector<std::pair<float, int>> nb;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            float dx = sx[j] - sx[i], dy = sy[j] - sy[i];
            nb.emplace_back(dx * dx + dy * dy, j);
        }
        int k = std::min(KS, static_cast<int>(nb.size()));
        std::partial_sort(nb.begin(), nb.begin() + k, nb.end());
        float mx = sx[i], my = sy[i];
        int cnt = 1;
        for (int t = 0; t < k; ++t) { mx += sx[nb[t].second]; my += sy[nb[t].second]; ++cnt; }
        mx /= cnt; my /= cnt;
        float sxx = (sx[i] - mx) * (sx[i] - mx);
        float syy = (sy[i] - my) * (sy[i] - my);
        float sxy = (sx[i] - mx) * (sy[i] - my);
        for (int t = 0; t < k; ++t) {
            float dx = sx[nb[t].second] - mx, dy = sy[nb[t].second] - my;
            sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
        }
        float inv = 1.0f / cnt;
        sxx *= inv; syy *= inv; sxy *= inv;
        float tr = sxx + syy;
        float disc = std::sqrt(std::max(0.0f, 0.25f * tr * tr - (sxx * syy - sxy * sxy)));
        float l_small = 0.5f * tr - disc;
        float nx, ny;
        // Local scan neighbourhoods can be near-degenerate; fall back to
        // isotropic sensor noise when there is no clear surface direction.
        float l_large = 0.5f * tr + disc;
        if (l_large < 1e-7f) { c00[i] = SCAN_SURF_VAR + SENSOR_VAR; c01[i] = 0.0f;
                               c11[i] = SCAN_SURF_VAR + SENSOR_VAR; continue; }
        if (std::fabs(sxy) > 1e-9f) { nx = l_small - syy; ny = sxy; }
        else if (sxx < syy)        { nx = 1.0f; ny = 0.0f; }
        else                       { nx = 0.0f; ny = 1.0f; }
        float nn = std::sqrt(nx * nx + ny * ny);
        if (nn < 1e-9f) { nx = 1.0f; ny = 0.0f; } else { nx /= nn; ny /= nn; }
        float w = 1.0f - GICP_EPS_SCAN;
        c00[i] = SCAN_SURF_VAR * (1.0f - w * nx * nx) + SENSOR_VAR;
        c01[i] = SCAN_SURF_VAR * (    - w * nx * ny);
        c11[i] = SCAN_SURF_VAR * (1.0f - w * ny * ny) + SENSOR_VAR;
    }
}

__global__ void init_rng_kernel(curandState* states, unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &states[i]);
}

__global__ void init_uniform_kernel(float* x, float* y, float* th, float* w, curandState* rng, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState s = rng[i];
    x[i] = 0.6f + (WORLD_W - 1.2f) * curand_uniform(&s);
    y[i] = 0.6f + (WORLD_H - 1.2f) * curand_uniform(&s);
    th[i] = -PI_F + 2.0f * PI_F * curand_uniform(&s);
    w[i] = 1.0f / n;
    rng[i] = s;
}

__global__ void predict_kernel(float* x,
                               float* y,
                               float* th,
                               curandState* rng,
                               int n,
                               float v,
                               float omega,
                               float xy_sigma,
                               float th_sigma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState s = rng[i];
    float vt = v + xy_sigma * curand_normal(&s);
    float wt = omega + th_sigma * curand_normal(&s);
    float theta = th[i];
    x[i] = clampf(x[i] + vt * cosf(theta) * DT, 0.45f, WORLD_W - 0.45f);
    y[i] = clampf(y[i] + vt * sinf(theta) * DT, 0.45f, WORLD_H - 0.45f);
    th[i] = wrap_angle(theta + wt * DT);
    rng[i] = s;
}

__global__ void occlusion_spread_kernel(float* x, float* y, float* th, curandState* rng, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState s = rng[i];
    x[i] = clampf(x[i] + OCCLUDED_SPREAD_XY * curand_normal(&s), 0.45f, WORLD_W - 0.45f);
    y[i] = clampf(y[i] + OCCLUDED_SPREAD_XY * curand_normal(&s), 0.45f, WORLD_H - 0.45f);
    th[i] = wrap_angle(th[i] + OCCLUDED_SPREAD_TH * curand_normal(&s));
    rng[i] = s;
}

__device__ float sample_nearest(const float* field, float wx, float wy) {
    int ix = static_cast<int>(wx / GRID_RES);
    int iy = static_cast<int>(wy / GRID_RES);
    ix = max(0, min(GRID_W - 1, ix));
    iy = max(0, min(GRID_H - 1, iy));
    return field[iy * GRID_W + ix];
}

// --- Likelihood A: distance-field endpoint proxy (the #86 likelihood) ----
__global__ void likelihood_field_kernel(const float* __restrict__ x,
                                        const float* __restrict__ y,
                                        const float* __restrict__ th,
                                        const float* __restrict__ scan_x,
                                        const float* __restrict__ scan_y,
                                        const float* __restrict__ dist,
                                        const float* __restrict__ grad_x,
                                        const float* __restrict__ grad_y,
                                        float* __restrict__ score,
                                        float* __restrict__ step_x,
                                        float* __restrict__ step_y,
                                        float* __restrict__ step_th,
                                        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float px = x[i], py = y[i], pth = th[i];
    float c = cosf(pth), s = sinf(pth);
    float logp = 0.0f;
    float gx_acc = 0.0f, gy_acc = 0.0f, gt_acc = 0.0f;
    float hxx = 0.0f, hyy = 0.0f, htt = 0.0f;
    const float inv_var = 1.0f / (DIST_SIGMA * DIST_SIGMA);

    #pragma unroll
    for (int k = 0; k < N_SCAN; ++k) {
        float sx = scan_x[k];
        float sy = scan_y[k];
        float wx = px + c * sx - s * sy;
        float wy = py + s * sx + c * sy;
        bool outside = (wx < 0.0f || wx >= WORLD_W || wy < 0.0f || wy >= WORLD_H);
        float d = outside ? 2.5f : sample_nearest(dist, wx, wy);
        float gxf = outside ? ((wx < 0.0f) ? -1.0f : ((wx >= WORLD_W) ? 1.0f : 0.0f))
                            : sample_nearest(grad_x, wx, wy);
        float gyf = outside ? ((wy < 0.0f) ? -1.0f : ((wy >= WORLD_H) ? 1.0f : 0.0f))
                            : sample_nearest(grad_y, wx, wy);
        d = fminf(d, 2.5f);
        float dthx = -s * sx - c * sy;
        float dthy = c * sx - s * sy;
        float jt = gxf * dthx + gyf * dthy;
        logp += -0.5f * d * d * inv_var;
        gx_acc += d * gxf * inv_var;
        gy_acc += d * gyf * inv_var;
        gt_acc += d * jt * inv_var;
        hxx += gxf * gxf * inv_var;
        hyy += gyf * gyf * inv_var;
        htt += jt * jt * inv_var;
    }
    float dx = -gx_acc / (hxx + 0.20f);
    float dy = -gy_acc / (hyy + 0.20f);
    float dt = -gt_acc / (htt + 0.40f);
    step_x[i] = clampf(dx, -0.22f, 0.22f);
    step_y[i] = clampf(dy, -0.22f, 0.22f);
    step_th[i] = clampf(dt, -0.060f, 0.060f);
    score[i] = fmaxf(logp, -90.0f);
}

// Solve (H + lambda*I) dp = -g for a 3x3 symmetric system, dp returned.
__device__ static inline void solve3_sym(float H0, float H1, float H2,
                                         float H3, float H4, float H5,
                                         float g0, float g1, float g2,
                                         float lam,
                                         float* dx, float* dy, float* dt) {
    float a = H0 + lam, b = H1, cc = H2;
    float d = H3 + lam, e = H4, f = H5 + lam;
    float det = a * (d * f - e * e) - b * (b * f - e * cc) + cc * (b * e - d * cc);
    if (fabsf(det) < 1e-12f) { *dx = 0.0f; *dy = 0.0f; *dt = 0.0f; return; }
    float inv = 1.0f / det;
    float i00 = (d * f - e * e) * inv;
    float i01 = (cc * e - b * f) * inv;
    float i02 = (b * e - cc * d) * inv;
    float i11 = (a * f - cc * cc) * inv;
    float i12 = (b * cc - a * e) * inv;
    float i22 = (a * d - b * b) * inv;
    *dx = -(i00 * g0 + i01 * g1 + i02 * g2);
    *dy = -(i01 * g0 + i11 * g1 + i12 * g2);
    *dt = -(i02 * g0 + i12 * g1 + i22 * g2);
}

// --- Likelihood B: GICP-style distribution-to-distribution (coarse-to-fine)
//
// Each scan endpoint is matched to the nearest map cloud point through the
// uniform grid index.  A MATCHED ray (within MATCH_R) is scored with the
// surface-aware GICP Gaussian log-likelihood -0.5 r^T M r under the combined
// covariance M = (C_map + R C_scan R^T)^{-1} -- sharp along the surface
// normal, loose along the tangent (point-to-line behaviour).  An UNMATCHED
// ray falls back to the distance-field endpoint log-likelihood (the #86
// proxy), which supplies a smooth long-range gradient so a globally lost
// particle is still pulled toward structure.  All rays contribute to one
// shared 3x3 Gauss-Newton system that drives the Stein motion.
//
// The fallback makes the worst case (no matches) behave exactly like the
// field-proxy filter, so global kidnap recovery stays robust, while the GICP
// term sharpens the cost wherever the particle is already near the right wall
// -- this is where the accuracy gain over the pure field proxy comes from.
__global__ void likelihood_d2d_kernel(const float* __restrict__ x,
                                      const float* __restrict__ y,
                                      const float* __restrict__ th,
                                      const float* __restrict__ scan_x,
                                      const float* __restrict__ scan_y,
                                      const float* __restrict__ scan_c00,
                                      const float* __restrict__ scan_c01,
                                      const float* __restrict__ scan_c11,
                                      const float* __restrict__ map_x,
                                      const float* __restrict__ map_y,
                                      const float* __restrict__ map_c00,
                                      const float* __restrict__ map_c01,
                                      const float* __restrict__ map_c11,
                                      const int* __restrict__ cell_start,
                                      const int* __restrict__ cell_end,
                                      const float* __restrict__ dist,
                                      const float* __restrict__ grad_x,
                                      const float* __restrict__ grad_y,
                                      float* __restrict__ score,
                                      float* __restrict__ step_x,
                                      float* __restrict__ step_y,
                                      float* __restrict__ step_th,
                                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float px = x[i], py = y[i], pth = th[i];
    float c = cosf(pth), s = sinf(pth);
    float cs = c * s, c2 = c * c, s2 = s * s;
    const float inv_var = 1.0f / (DIST_SIGMA * DIST_SIGMA);
    float logp = 0.0f;
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f;
    float H0 = 0.0f, H1 = 0.0f, H2 = 0.0f, H3 = 0.0f, H4 = 0.0f, H5 = 0.0f;

    #pragma unroll
    for (int k = 0; k < N_SCAN; ++k) {
        float sx = scan_x[k], sy = scan_y[k];
        float qx = px + c * sx - s * sy;
        float qy = py + s * sx + c * sy;
        float dthx = -s * sx - c * sy;   // d q / d theta
        float dthy = c * sx - s * sy;
        bool outside = (qx < 0.0f || qx >= WORLD_W || qy < 0.0f || qy >= WORLD_H);
        int best_j = -1;
        if (!outside) {
            int cx = max(0, min(CLOUD_GW - 1, static_cast<int>(qx / CLOUD_CELL)));
            int cy = max(0, min(CLOUD_GH - 1, static_cast<int>(qy / CLOUD_CELL)));
            float best_d2 = MATCH_R2;
            for (int oy = -1; oy <= 1; ++oy) {
                int ccy = cy + oy;
                if (ccy < 0 || ccy >= CLOUD_GH) continue;
                for (int ox = -1; ox <= 1; ++ox) {
                    int ccx = cx + ox;
                    if (ccx < 0 || ccx >= CLOUD_GW) continue;
                    int cell = ccy * CLOUD_GW + ccx;
                    int start = cell_start[cell], end = cell_end[cell];
                    for (int j = start; j < end; ++j) {
                        float dx = map_x[j] - qx, dy = map_y[j] - qy;
                        float d2 = dx * dx + dy * dy;
                        if (d2 < best_d2) { best_d2 = d2; best_j = j; }
                    }
                }
            }
        }
        if (best_j >= 0) {
            // GICP D2D term: surface-aware Gaussian log-likelihood + rank-2 GN.
            float a = scan_c00[k], b = scan_c01[k], dd = scan_c11[k];
            float sw00 = c2 * a - 2.0f * cs * b + s2 * dd;   // R C_scan R^T
            float sw01 = cs * (a - dd) + (c2 - s2) * b;
            float sw11 = s2 * a + 2.0f * cs * b + c2 * dd;
            float K00 = map_c00[best_j] + sw00;
            float K01 = map_c01[best_j] + sw01;
            float K11 = map_c11[best_j] + sw11;
            float Kdet = K00 * K11 - K01 * K01;
            if (Kdet > 1e-9f) {
                float invd = 1.0f / Kdet;
                float M00 = K11 * invd, M01 = -K01 * invd, M11 = K00 * invd;
                float rx = qx - map_x[best_j];
                float ry = qy - map_y[best_j];
                float ux = M00 * rx + M01 * ry;
                float uy = M01 * rx + M11 * ry;
                logp += fmaxf(-0.5f * (rx * ux + ry * uy), -6.0f);
                float Jx = dthx, Jy = dthy;  // J = [ I | R'(th) s ]
                g0 += ux;
                g1 += uy;
                g2 += Jx * ux + Jy * uy;
                float vx = M00 * Jx + M01 * Jy;
                float vy = M01 * Jx + M11 * Jy;
                H0 += M00; H1 += M01; H2 += vx;
                H3 += M11; H4 += vy; H5 += Jx * vx + Jy * vy;
                continue;
            }
        }
        // Field-proxy fallback: distance-field endpoint log-likelihood +
        // rank-1 GN along the distance gradient (long-range global pull).
        float d = outside ? 2.5f : fminf(sample_nearest(dist, qx, qy), 2.5f);
        float gxf = outside ? ((qx < 0.0f) ? -1.0f : ((qx >= WORLD_W) ? 1.0f : 0.0f))
                            : sample_nearest(grad_x, qx, qy);
        float gyf = outside ? ((qy < 0.0f) ? -1.0f : ((qy >= WORLD_H) ? 1.0f : 0.0f))
                            : sample_nearest(grad_y, qx, qy);
        float jt = gxf * dthx + gyf * dthy;
        logp += -0.5f * d * d * inv_var;
        g0 += d * gxf * inv_var;
        g1 += d * gyf * inv_var;
        g2 += d * jt * inv_var;
        H0 += gxf * gxf * inv_var;
        H1 += gxf * gyf * inv_var;
        H2 += gxf * jt * inv_var;
        H3 += gyf * gyf * inv_var;
        H4 += gyf * jt * inv_var;
        H5 += jt * jt * inv_var;
    }
    float dx, dy, dt;
    solve3_sym(H0, H1, H2, H3, H4, H5, g0, g1, g2, 0.30f, &dx, &dy, &dt);
    step_x[i] = clampf(dx, -0.22f, 0.22f);
    step_y[i] = clampf(dy, -0.22f, 0.22f);
    step_th[i] = clampf(dt, -0.060f, 0.060f);
    score[i] = fmaxf(logp, -90.0f);
}

__global__ void weights_from_score_kernel(const float* score, float* weights, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    weights[i] = expf(fmaxf(score[i] * LIK_TEMP, -80.0f));
}

__global__ void posterior_from_score_kernel(const float* score, float* posterior, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    posterior[i] = expf(fmaxf(score[i] * LIK_TEMP, -80.0f)) + 1.0e-18f;
}

__device__ int bucket_id(float x, float y, float th) {
    int bx = max(0, min(BUCKET_X - 1, static_cast<int>(x / WORLD_W * BUCKET_X)));
    int by = max(0, min(BUCKET_Y - 1, static_cast<int>(y / WORLD_H * BUCKET_Y)));
    float tn = (wrap_angle(th) + PI_F) / (2.0f * PI_F);
    int bt = max(0, min(BUCKET_T - 1, static_cast<int>(tn * BUCKET_T)));
    return bx + BUCKET_X * (by + BUCKET_Y * bt);
}

__global__ void bucket_motion_aggregate_kernel(const float* x,
                                               const float* y,
                                               const float* th,
                                               const float* step_x,
                                               const float* step_y,
                                               const float* step_th,
                                               const float* posterior,
                                               float* b_step_x,
                                               float* b_step_y,
                                               float* b_step_th,
                                               float* b_x,
                                               float* b_y,
                                               float* b_sin,
                                               float* b_cos,
                                               float* b_post,
                                               float* b_count,
                                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = bucket_id(x[i], y[i], th[i]);
    float p = posterior[i] + 1.0e-18f;
    atomicAdd(&b_step_x[b], p * step_x[i]);
    atomicAdd(&b_step_y[b], p * step_y[i]);
    atomicAdd(&b_step_th[b], p * step_th[i]);
    atomicAdd(&b_x[b], x[i]);
    atomicAdd(&b_y[b], y[i]);
    atomicAdd(&b_sin[b], sinf(th[i]));
    atomicAdd(&b_cos[b], cosf(th[i]));
    atomicAdd(&b_post[b], p);
    atomicAdd(&b_count[b], 1.0f);
}

__global__ void stein_bucket_update_kernel(float* x,
                                           float* y,
                                           float* th,
                                           curandState* rng,
                                           const float* step_x,
                                           const float* step_y,
                                           const float* step_th,
                                           const float* b_step_x,
                                           const float* b_step_y,
                                           const float* b_step_th,
                                           const float* b_x,
                                           const float* b_y,
                                           const float* b_sin,
                                           const float* b_cos,
                                           const float* b_post,
                                           const float* b_count,
                                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = bucket_id(x[i], y[i], th[i]);
    float cnt = fmaxf(b_count[b], 1.0f);
    float mass = fmaxf(b_post[b], 1.0e-18f);
    float avg_dx = b_step_x[b] / mass;
    float avg_dy = b_step_y[b] / mass;
    float avg_dt = b_step_th[b] / mass;
    float mx = b_x[b] / cnt;
    float my = b_y[b] / cnt;
    float mt = atan2f(b_sin[b] / cnt, b_cos[b] / cnt);
    float rep_x = 0.020f * (x[i] - mx);
    float rep_y = 0.020f * (y[i] - my);
    float rep_t = 0.006f * wrap_angle(th[i] - mt);
    curandState s = rng[i];
    float jitter_x = 0.012f * curand_normal(&s);
    float jitter_y = 0.012f * curand_normal(&s);
    float jitter_t = 0.003f * curand_normal(&s);
    float dx = 0.45f * step_x[i] + 0.75f * avg_dx + rep_x + jitter_x;
    float dy = 0.45f * step_y[i] + 0.75f * avg_dy + rep_y + jitter_y;
    float dt = 0.45f * step_th[i] + 0.75f * avg_dt + rep_t + jitter_t;
    x[i] = clampf(x[i] + clampf(dx, -0.18f, 0.18f), 0.45f, WORLD_W - 0.45f);
    y[i] = clampf(y[i] + clampf(dy, -0.18f, 0.18f), 0.45f, WORLD_H - 0.45f);
    th[i] = wrap_angle(th[i] + clampf(dt, -0.050f, 0.050f));
    rng[i] = s;
}

__global__ void bucket_posterior_aggregate_kernel(const float* x,
                                                  const float* y,
                                                  const float* th,
                                                  const float* posterior,
                                                  float* b_post,
                                                  float* b_count,
                                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = bucket_id(x[i], y[i], th[i]);
    atomicAdd(&b_post[b], posterior[i]);
    atomicAdd(&b_count[b], 1.0f);
}

__global__ void posterior_smooth_kernel(const float* x,
                                        const float* y,
                                        const float* th,
                                        float* posterior,
                                        const float* b_post,
                                        const float* b_count,
                                        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = bucket_id(x[i], y[i], th[i]);
    float mean = b_post[b] / fmaxf(b_count[b], 1.0f);
    posterior[i] = 0.58f * posterior[i] + 0.42f * mean;
}

__global__ void bucket_representative_aggregate_kernel(const float* x,
                                                       const float* y,
                                                       const float* th,
                                                       const float* posterior,
                                                       float* b_x,
                                                       float* b_y,
                                                       float* b_sin,
                                                       float* b_cos,
                                                       float* b_post,
                                                       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = bucket_id(x[i], y[i], th[i]);
    float p = posterior[i] + 1.0e-18f;
    atomicAdd(&b_x[b], p * x[i]);
    atomicAdd(&b_y[b], p * y[i]);
    atomicAdd(&b_sin[b], p * sinf(th[i]));
    atomicAdd(&b_cos[b], p * cosf(th[i]));
    atomicAdd(&b_post[b], p);
}

struct ParticleSet {
    int n = 0;
    float *x = nullptr, *y = nullptr, *th = nullptr;
    float *w = nullptr, *score = nullptr, *step_x = nullptr, *step_y = nullptr, *step_th = nullptr;
    curandState* rng = nullptr;
    std::vector<float> hx, hy;

    void alloc(int n_, unsigned long long seed) {
        n = n_;
        int blocks = (n + THREADS - 1) / THREADS;
        CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&y, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&th, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&w, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&score, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&step_x, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&step_y, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&step_th, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rng, n * sizeof(curandState)));
        hx.resize(n);
        hy.resize(n);
        init_rng_kernel<<<blocks, THREADS>>>(rng, seed, n);
        CUDA_CHECK(cudaGetLastError());
    }

    void free_all() {
        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
        CUDA_CHECK(cudaFree(th));
        CUDA_CHECK(cudaFree(w));
        CUDA_CHECK(cudaFree(score));
        CUDA_CHECK(cudaFree(step_x));
        CUDA_CHECK(cudaFree(step_y));
        CUDA_CHECK(cudaFree(step_th));
        CUDA_CHECK(cudaFree(rng));
    }

    void copy_xy_to_host() {
        CUDA_CHECK(cudaMemcpy(hx.data(), x, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hy.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost));
    }
};

struct DeviceMap {
    float *dist = nullptr, *gx = nullptr, *gy = nullptr;
    void upload(const CpuMap& map) {
        size_t bytes = GRID_W * GRID_H * sizeof(float);
        CUDA_CHECK(cudaMalloc(&dist, bytes));
        CUDA_CHECK(cudaMalloc(&gx, bytes));
        CUDA_CHECK(cudaMalloc(&gy, bytes));
        CUDA_CHECK(cudaMemcpy(dist, map.dist.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gx, map.gx.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gy, map.gy.data(), bytes, cudaMemcpyHostToDevice));
    }
    void free_all() {
        CUDA_CHECK(cudaFree(dist));
        CUDA_CHECK(cudaFree(gx));
        CUDA_CHECK(cudaFree(gy));
    }
};

struct DeviceCloud {
    int n = 0;
    float *x = nullptr, *y = nullptr, *c00 = nullptr, *c01 = nullptr, *c11 = nullptr;
    int *cell_start = nullptr, *cell_end = nullptr;
    void upload(const CpuCloud& cloud) {
        n = cloud.n;
        CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&y, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c00, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c01, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c11, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cell_start, CLOUD_NCELLS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&cell_end, CLOUD_NCELLS * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(x, cloud.x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(y, cloud.y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(c00, cloud.c00.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(c01, cloud.c01.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(c11, cloud.c11.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cell_start, cloud.cell_start.data(), CLOUD_NCELLS * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cell_end, cloud.cell_end.data(), CLOUD_NCELLS * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    void free_all() {
        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
        CUDA_CHECK(cudaFree(c00));
        CUDA_CHECK(cudaFree(c01));
        CUDA_CHECK(cudaFree(c11));
        CUDA_CHECK(cudaFree(cell_start));
        CUDA_CHECK(cudaFree(cell_end));
    }
};

struct MegaBuckets {
    float *step_x = nullptr, *step_y = nullptr, *step_th = nullptr;
    float *x = nullptr, *y = nullptr, *sin_th = nullptr, *cos_th = nullptr;
    float *post = nullptr, *count = nullptr;
    std::vector<float> h_x, h_y, h_sin, h_cos, h_post;

    void alloc() {
        CUDA_CHECK(cudaMalloc(&step_x, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&step_y, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&step_th, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&x, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&y, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sin_th, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cos_th, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&post, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&count, N_BUCKETS * sizeof(float)));
        h_x.resize(N_BUCKETS);
        h_y.resize(N_BUCKETS);
        h_sin.resize(N_BUCKETS);
        h_cos.resize(N_BUCKETS);
        h_post.resize(N_BUCKETS);
    }

    void clear_all() {
        CUDA_CHECK(cudaMemset(step_x, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(step_y, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(step_th, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(x, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(y, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(sin_th, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(cos_th, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(post, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(count, 0, N_BUCKETS * sizeof(float)));
    }

    void clear_post() {
        CUDA_CHECK(cudaMemset(post, 0, N_BUCKETS * sizeof(float)));
        CUDA_CHECK(cudaMemset(count, 0, N_BUCKETS * sizeof(float)));
    }

    void free_all() {
        CUDA_CHECK(cudaFree(step_x));
        CUDA_CHECK(cudaFree(step_y));
        CUDA_CHECK(cudaFree(step_th));
        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
        CUDA_CHECK(cudaFree(sin_th));
        CUDA_CHECK(cudaFree(cos_th));
        CUDA_CHECK(cudaFree(post));
        CUDA_CHECK(cudaFree(count));
    }
};

static Pose2 bucket_representative(ParticleSet& p, MegaBuckets& buckets, float* best_mass) {
    int blocks = (p.n + THREADS - 1) / THREADS;
    buckets.clear_all();
    bucket_representative_aggregate_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.w,
                                                                buckets.x, buckets.y,
                                                                buckets.sin_th, buckets.cos_th,
                                                                buckets.post, p.n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(buckets.h_post.data(), buckets.post, N_BUCKETS * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(buckets.h_x.data(), buckets.x, N_BUCKETS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(buckets.h_y.data(), buckets.y, N_BUCKETS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(buckets.h_sin.data(), buckets.sin_th, N_BUCKETS * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(buckets.h_cos.data(), buckets.cos_th, N_BUCKETS * sizeof(float),
                          cudaMemcpyDeviceToHost));
    int best = 0;
    for (int i = 1; i < N_BUCKETS; ++i) {
        if (buckets.h_post[i] > buckets.h_post[best]) best = i;
    }
    float mass = std::max(buckets.h_post[best], 1.0e-18f);
    if (best_mass) *best_mass = mass;
    return {buckets.h_x[best] / mass,
            buckets.h_y[best] / mass,
            std::atan2(buckets.h_sin[best] / mass, buckets.h_cos[best] / mass)};
}

static void draw_pose(cv::Mat& img, int ox, const Pose2& p, const cv::Scalar& color, int radius) {
    auto to_px = [&](float x, float y) {
        int px = ox + static_cast<int>(x / WORLD_W * PANEL_W);
        int py = static_cast<int>((WORLD_H - y) / WORLD_H * PANEL_H);
        return cv::Point(px, py);
    };
    cv::Point c = to_px(p.x, p.y);
    cv::circle(img, c, radius, color, -1, cv::LINE_AA);
    cv::Point h = to_px(p.x + 0.9f * std::cos(p.th), p.y + 0.9f * std::sin(p.th));
    cv::line(img, c, h, color, 2, cv::LINE_AA);
}

static void draw_panel(cv::Mat& img,
                       int ox,
                       const std::string& title,
                       const CpuMap& map,
                       const CpuCloud* cloud,
                       const std::vector<Pose2>& truth_hist,
                       const std::vector<Pose2>& est_hist,
                       const Pose2& truth,
                       const Pose2& est,
                       const std::vector<float>& px,
                       const std::vector<float>& py,
                       int stride,
                       const cv::Scalar& particle_color,
                       const cv::Scalar& est_color) {
    cv::rectangle(img, cv::Rect(ox, 0, PANEL_W, PANEL_H), cv::Scalar(250, 250, 247), -1);
    for (const Rect& r : map.rects) {
        int x0 = ox + static_cast<int>(r.x0 / WORLD_W * PANEL_W);
        int x1 = ox + static_cast<int>(r.x1 / WORLD_W * PANEL_W);
        int y0 = static_cast<int>((WORLD_H - r.y1) / WORLD_H * PANEL_H);
        int y1 = static_cast<int>((WORLD_H - r.y0) / WORLD_H * PANEL_H);
        cv::rectangle(img, cv::Rect(cv::Point(x0, y0), cv::Point(x1 + 1, y1 + 1)),
                      cv::Scalar(58, 64, 72), -1);
    }
    // GICP panel: overlay the map point cloud to make the representation visible.
    if (cloud) {
        for (int i = 0; i < cloud->n; i += 2) {
            int x = ox + static_cast<int>(cloud->x[i] / WORLD_W * PANEL_W);
            int y = static_cast<int>((WORLD_H - cloud->y[i]) / WORLD_H * PANEL_H);
            if (x >= ox && x < ox + PANEL_W && y >= 0 && y < PANEL_H) {
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(150, 190, 210);
            }
        }
    }
    for (int i = 0; i < static_cast<int>(px.size()); i += stride) {
        int x = ox + static_cast<int>(px[i] / WORLD_W * PANEL_W);
        int y = static_cast<int>((WORLD_H - py[i]) / WORLD_H * PANEL_H);
        if (x >= ox && x < ox + PANEL_W && y >= 0 && y < PANEL_H) {
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<unsigned char>(particle_color[0]),
                                                static_cast<unsigned char>(particle_color[1]),
                                                static_cast<unsigned char>(particle_color[2]));
        }
    }
    auto to_px = [&](const Pose2& p) {
        return cv::Point(ox + static_cast<int>(p.x / WORLD_W * PANEL_W),
                         static_cast<int>((WORLD_H - p.y) / WORLD_H * PANEL_H));
    };
    for (size_t i = 1; i < truth_hist.size(); ++i) {
        cv::line(img, to_px(truth_hist[i - 1]), to_px(truth_hist[i]), cv::Scalar(40, 40, 40), 1, cv::LINE_AA);
    }
    for (size_t i = 1; i < est_hist.size(); ++i) {
        cv::line(img, to_px(est_hist[i - 1]), to_px(est_hist[i]), est_color, 2, cv::LINE_AA);
    }
    draw_pose(img, ox, truth, cv::Scalar(20, 20, 20), 5);
    draw_pose(img, ox, est, est_color, 6);
    cv::putText(img, title, cv::Point(ox + 14, 28), cv::FONT_HERSHEY_SIMPLEX, 0.58,
                cv::Scalar(20, 24, 32), 2, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(ox, 0, PANEL_W, PANEL_H), cv::Scalar(210, 210, 205), 1);
}

static void draw_info(cv::Mat& img,
                      int ox,
                      int step,
                      const StepSummary& s,
                      const FinalStats& partial,
                      bool occluded) {
    cv::rectangle(img, cv::Rect(ox, 0, INFO_W, FRAME_H), cv::Scalar(244, 246, 246), -1);
    cv::putText(img, "MegaParticles likelihood", cv::Point(ox + 18, 34), cv::FONT_HERSHEY_SIMPLEX,
                0.60, cv::Scalar(20, 28, 35), 2, cv::LINE_AA);
    cv::putText(img, "field proxy  vs  GICP D2D", cv::Point(ox + 18, 60), cv::FONT_HERSHEY_SIMPLEX,
                0.48, cv::Scalar(70, 78, 88), 1, cv::LINE_AA);

    char buf[256];
    std::snprintf(buf, sizeof(buf), "step %03d / %03d", step, N_STEPS - 1);
    cv::putText(img, buf, cv::Point(ox + 18, 104), cv::FONT_HERSHEY_SIMPLEX, 0.54,
                cv::Scalar(30, 36, 44), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "scan: %s", occluded ? "blocked / hidden kidnap" : "visible");
    cv::putText(img, buf, cv::Point(ox + 18, 130), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                occluded ? cv::Scalar(40, 70, 190) : cv::Scalar(40, 120, 80), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "field-proxy err: %.2f m", s.field_err);
    cv::putText(img, buf, cv::Point(ox + 18, 176), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(180, 80, 40), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "GICP-D2D err: %.2f m", s.d2d_err);
    cv::putText(img, buf, cv::Point(ox + 18, 202), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(55, 95, 175), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "particles each: %d", K_MEGA);
    cv::putText(img, buf, cv::Point(ox + 18, 248), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "map cloud pts: %d", partial.cloud_points);
    cv::putText(img, buf, cv::Point(ox + 18, 272), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "avg step: %.2f / %.2f ms", partial.field_ms, partial.d2d_ms);
    cv::putText(img, buf, cv::Point(ox + 18, 318), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(55, 60, 66), 1, cv::LINE_AA);
}

enum class Likelihood { FIELD, D2D };

// One MegaParticles step: predict, Stein contraction under the selected
// likelihood, posterior smoothing, representative pose.  The two arms differ
// only in which scoring kernel fills score / step_{x,y,th}.
static void run_likelihood(Likelihood mode,
                           ParticleSet& p,
                           const DeviceMap& dmap,
                           const DeviceCloud& dcloud,
                           const float* d_scan_x,
                           const float* d_scan_y,
                           const float* d_scan_c00,
                           const float* d_scan_c01,
                           const float* d_scan_c11) {
    int blocks = (p.n + THREADS - 1) / THREADS;
    if (mode == Likelihood::FIELD) {
        likelihood_field_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, d_scan_x, d_scan_y,
                                                     dmap.dist, dmap.gx, dmap.gy,
                                                     p.score, p.step_x, p.step_y, p.step_th, p.n);
    } else {
        likelihood_d2d_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, d_scan_x, d_scan_y,
                                                   d_scan_c00, d_scan_c01, d_scan_c11,
                                                   dcloud.x, dcloud.y, dcloud.c00, dcloud.c01, dcloud.c11,
                                                   dcloud.cell_start, dcloud.cell_end,
                                                   dmap.dist, dmap.gx, dmap.gy,
                                                   p.score, p.step_x, p.step_y, p.step_th, p.n);
    }
    CUDA_CHECK(cudaGetLastError());
}

static void mega_filter_step(Likelihood mode,
                             ParticleSet& p,
                             MegaBuckets& buckets,
                             const DeviceMap& dmap,
                             const DeviceCloud& dcloud,
                             const float* d_scan_x,
                             const float* d_scan_y,
                             const float* d_scan_c00,
                             const float* d_scan_c01,
                             const float* d_scan_c11,
                             float v,
                             float omega,
                             bool visible,
                             Pose2& estimate) {
    int blocks = (p.n + THREADS - 1) / THREADS;
    predict_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng, p.n, v, omega,
                                        MEGA_MOTION_SIGMA_XY, MEGA_MOTION_SIGMA_TH);
    CUDA_CHECK(cudaGetLastError());
    if (!visible) {
        occlusion_spread_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng, p.n);
        CUDA_CHECK(cudaGetLastError());
        float mass;
        estimate = bucket_representative(p, buckets, &mass);
        return;
    }

    for (int it = 0; it < STEIN_ITERS; ++it) {
        run_likelihood(mode, p, dmap, dcloud, d_scan_x, d_scan_y,
                       d_scan_c00, d_scan_c01, d_scan_c11);
        posterior_from_score_kernel<<<blocks, THREADS>>>(p.score, p.w, p.n);
        CUDA_CHECK(cudaGetLastError());
        buckets.clear_all();
        bucket_motion_aggregate_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.step_x, p.step_y, p.step_th,
                                                            p.w, buckets.step_x, buckets.step_y,
                                                            buckets.step_th, buckets.x, buckets.y,
                                                            buckets.sin_th, buckets.cos_th,
                                                            buckets.post, buckets.count, p.n);
        CUDA_CHECK(cudaGetLastError());
        stein_bucket_update_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.rng,
                                                        p.step_x, p.step_y, p.step_th,
                                                        buckets.step_x, buckets.step_y,
                                                        buckets.step_th, buckets.x, buckets.y,
                                                        buckets.sin_th, buckets.cos_th,
                                                        buckets.post, buckets.count, p.n);
        CUDA_CHECK(cudaGetLastError());
    }

    run_likelihood(mode, p, dmap, dcloud, d_scan_x, d_scan_y,
                   d_scan_c00, d_scan_c01, d_scan_c11);
    posterior_from_score_kernel<<<blocks, THREADS>>>(p.score, p.w, p.n);
    CUDA_CHECK(cudaGetLastError());
    for (int it = 0; it < POST_PROP_ITERS; ++it) {
        buckets.clear_post();
        bucket_posterior_aggregate_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.w,
                                                               buckets.post, buckets.count, p.n);
        CUDA_CHECK(cudaGetLastError());
        posterior_smooth_kernel<<<blocks, THREADS>>>(p.x, p.y, p.th, p.w,
                                                     buckets.post, buckets.count, p.n);
        CUDA_CHECK(cudaGetLastError());
    }
    float mass;
    estimate = bucket_representative(p, buckets, &mass);
}

static std::vector<Pose2> make_truth(std::vector<float>& v_cmd, std::vector<float>& w_cmd) {
    std::vector<Pose2> truth(N_STEPS);
    v_cmd.resize(N_STEPS);
    w_cmd.resize(N_STEPS);
    Pose2 p{4.2f, 4.6f, 0.18f};
    for (int t = 0; t < N_STEPS; ++t) {
        if (t == KIDNAP_STEP) {
            p = {27.7f, 18.2f, -2.55f};
        }
        float v = 0.88f + 0.10f * std::sin(0.08f * t);
        float w = 0.28f * std::sin(0.105f * t + 0.4f);
        if (t > 24 && t < 42) w += 0.28f;
        if (t > 78 && t < 94) w -= 0.23f;
        v_cmd[t] = v;
        w_cmd[t] = w;
        truth[t] = p;
        p = integrate_pose(p, v, w);
    }
    return truth;
}

static void ensure_dirs() {
    int rc = std::system("mkdir -p gif tmp");
    if (rc != 0) std::fprintf(stderr, "mkdir failed with code %d\n", rc);
}

// Gate the representative pose: keep tracking continuous, but allow a global
// jump the moment scans return after the blackout.  Returns the gated pose.
static Pose2 gate_track(Pose2 raw, Pose2& track, bool& has_track, bool visible,
                        bool just_unblocked, float v, float omega) {
    if (!has_track || just_unblocked) {
        track = raw;
        has_track = true;
        return raw;
    }
    Pose2 predicted = integrate_pose(track, v, omega);
    Pose2 out = raw;
    if (visible && pose_error_xy(raw, predicted) > 2.6f) out = predicted;
    track = visible ? out : predicted;
    if (!visible) out = track;
    return out;
}

static FinalStats run_demo() {
    ensure_dirs();
    CpuMap map = make_map();
    CpuCloud cloud = build_cloud(map);
    std::printf("map point cloud: %d points, NN grid %dx%d\n", cloud.n, CLOUD_GW, CLOUD_GH);
    DeviceMap dmap;
    dmap.upload(map);
    DeviceCloud dcloud;
    dcloud.upload(cloud);

    float *d_scan_x = nullptr, *d_scan_y = nullptr;
    float *d_scan_c00 = nullptr, *d_scan_c01 = nullptr, *d_scan_c11 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scan_x, N_SCAN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_y, N_SCAN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_c00, N_SCAN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_c01, N_SCAN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_c11, N_SCAN * sizeof(float)));

    ParticleSet field, d2d;
    field.alloc(K_MEGA, 1111);
    d2d.alloc(K_MEGA, 2222);
    MegaBuckets buckets;
    buckets.alloc();

    std::vector<float> v_cmd, w_cmd;
    std::vector<Pose2> truth = make_truth(v_cmd, w_cmd);

    int mega_blocks = (K_MEGA + THREADS - 1) / THREADS;
    init_uniform_kernel<<<mega_blocks, THREADS>>>(field.x, field.y, field.th, field.w, field.rng, field.n);
    CUDA_CHECK(cudaGetLastError());
    init_uniform_kernel<<<mega_blocks, THREADS>>>(d2d.x, d2d.y, d2d.th, d2d.w, d2d.rng, d2d.n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cv::VideoWriter video("tmp/gpu_megaparticles_gicp_mcl.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 12,
                          cv::Size(FRAME_W, FRAME_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open output video\n");
        std::exit(1);
    }

    std::vector<Pose2> field_hist, d2d_hist, truth_hist;
    std::vector<float> scan_x, scan_y, scan_c00, scan_c01, scan_c11;
    FinalStats stats;
    stats.cloud_points = cloud.n;
    StepSummary last;
    int post_count = 0;
    float field_post_sq = 0.0f, d2d_post_sq = 0.0f;
    double field_ms_sum = 0.0, d2d_ms_sum = 0.0;
    bool field_has_track = false, d2d_has_track = false;
    Pose2 field_track{}, d2d_track{};

    for (int t = 0; t < N_STEPS; ++t) {
        bool visible = !(t >= KIDNAP_STEP && t < KIDNAP_STEP + OCCLUDE_STEPS);
        bool just_unblocked = (t == KIDNAP_STEP + OCCLUDE_STEPS);
        if (visible) {
            make_scan(map.rects, truth[t], t, scan_x, scan_y);
            compute_scan_cov(scan_x, scan_y, scan_c00, scan_c01, scan_c11);
            CUDA_CHECK(cudaMemcpy(d_scan_x, scan_x.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_scan_y, scan_y.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_scan_c00, scan_c00.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_scan_c01, scan_c01.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_scan_c11, scan_c11.data(), N_SCAN * sizeof(float), cudaMemcpyHostToDevice));
        }

        Pose2 field_raw, d2d_raw;
        auto t0 = std::chrono::high_resolution_clock::now();
        mega_filter_step(Likelihood::FIELD, field, buckets, dmap, dcloud,
                         d_scan_x, d_scan_y, d_scan_c00, d_scan_c01, d_scan_c11,
                         v_cmd[t], w_cmd[t], visible, field_raw);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        mega_filter_step(Likelihood::D2D, d2d, buckets, dmap, dcloud,
                         d_scan_x, d_scan_y, d_scan_c00, d_scan_c01, d_scan_c11,
                         v_cmd[t], w_cmd[t], visible, d2d_raw);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();

        Pose2 field_est = gate_track(field_raw, field_track, field_has_track, visible,
                                     just_unblocked, v_cmd[t], w_cmd[t]);
        Pose2 d2d_est = gate_track(d2d_raw, d2d_track, d2d_has_track, visible,
                                   just_unblocked, v_cmd[t], w_cmd[t]);

        double field_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double d2d_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        field_ms_sum += field_ms;
        d2d_ms_sum += d2d_ms;
        stats.field_ms = field_ms_sum / (t + 1);
        stats.d2d_ms = d2d_ms_sum / (t + 1);

        last.field = field_est;
        last.d2d = d2d_est;
        last.field_err = pose_error_xy(field_est, truth[t]);
        last.d2d_err = pose_error_xy(d2d_est, truth[t]);
        last.scan_visible = visible;
        last.field_ms = field_ms;
        last.d2d_ms = d2d_ms;

        if (t >= KIDNAP_STEP + OCCLUDE_STEPS) {
            field_post_sq += last.field_err * last.field_err;
            d2d_post_sq += last.d2d_err * last.d2d_err;
            post_count++;
            if (stats.field_reacq_step < 0 && last.field_err < 0.65f)
                stats.field_reacq_step = t - (KIDNAP_STEP + OCCLUDE_STEPS);
            if (stats.d2d_reacq_step < 0 && last.d2d_err < 0.65f)
                stats.d2d_reacq_step = t - (KIDNAP_STEP + OCCLUDE_STEPS);
        }
        stats.final_field_err = last.field_err;
        stats.final_d2d_err = last.d2d_err;
        stats.field_post_rmse = post_count ? std::sqrt(field_post_sq / post_count) : 0.0f;
        stats.d2d_post_rmse = post_count ? std::sqrt(d2d_post_sq / post_count) : 0.0f;

        field_hist.push_back(field_est);
        d2d_hist.push_back(d2d_est);
        truth_hist.push_back(truth[t]);

        if (t % VIDEO_EVERY == 0 || t == N_STEPS - 1) {
            field.copy_xy_to_host();
            d2d.copy_xy_to_host();
            cv::Mat frame(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(244, 246, 246));
            draw_panel(frame, 0, "1M field-proxy SPF", map, nullptr, truth_hist, field_hist,
                       truth[t], field_est, field.hx, field.hy, std::max(1, K_MEGA / 3600),
                       cv::Scalar(190, 205, 235), cv::Scalar(40, 95, 210));
            draw_panel(frame, PANEL_W, "1M GICP-D2D SPF", map, &cloud, truth_hist, d2d_hist,
                       truth[t], d2d_est, d2d.hx, d2d.hy, std::max(1, K_MEGA / 3600),
                       cv::Scalar(190, 215, 200), cv::Scalar(30, 150, 95));
            draw_info(frame, PANEL_W * 2, t, last, stats, !visible);
            video.write(frame);
        }

        std::printf("step %3d visible=%d field_err=%.3f d2d_err=%.3f field=%.2fms d2d=%.2fms\n",
                    t, visible ? 1 : 0, last.field_err, last.d2d_err, field_ms, d2d_ms);
    }

    video.release();
    avi_to_gif("tmp/gpu_megaparticles_gicp_mcl.avi", "gif/gpu_megaparticles_gicp_mcl.gif", 10, 760);

    CUDA_CHECK(cudaFree(d_scan_x));
    CUDA_CHECK(cudaFree(d_scan_y));
    CUDA_CHECK(cudaFree(d_scan_c00));
    CUDA_CHECK(cudaFree(d_scan_c01));
    CUDA_CHECK(cudaFree(d_scan_c11));
    buckets.free_all();
    field.free_all();
    d2d.free_all();
    dmap.free_all();
    dcloud.free_all();
    return stats;
}

}  // namespace cudabot

int main() {
    cudabot::FinalStats stats = cudabot::run_demo();
    std::printf("\nMegaParticles GICP-D2D vs field-proxy likelihood (SE(2), identical machinery)\n");
    std::printf("map cloud points: %d\n", stats.cloud_points);
    std::printf("post-kidnap RMSE: field proxy %.4f m, GICP D2D %.4f m\n",
                stats.field_post_rmse, stats.d2d_post_rmse);
    std::printf("final error: field proxy %.4f m, GICP D2D %.4f m\n",
                stats.final_field_err, stats.final_d2d_err);
    std::printf("reacquisition after blackout: field %d frames, GICP D2D %d frames\n",
                stats.field_reacq_step, stats.d2d_reacq_step);
    std::printf("avg GPU step: field proxy %.4f ms, GICP D2D %.4f ms\n",
                stats.field_ms, stats.d2d_ms);
    std::printf("Wrote gif/gpu_megaparticles_gicp_mcl.gif\n");
    return 0;
}
