/*************************************************************************
    GPU 2D LiDAR SLAM frontend (scan-to-scan ICP + occupancy mapping)

    Components:
      - synthetic 2D world (walls + interior obstacles)
      - 2D LiDAR sensor (720 rays, 360 deg FOV, 18 m range) running
        entirely on the GPU (1 thread per ray)
      - GPU brute-force point-to-point ICP between consecutive scans
        with 2D Procrustes closed-form transform estimation
      - log-odds 2D occupancy map updated each frame with the
        registered scan endpoints
      - ground-truth vs estimated trajectory rendered side-by-side

    The pose estimate is purely scan-to-scan ICP composition (no IMU,
    no odometry prior, no loop closure). Drift accumulates over time;
    the GIF shows how close the estimate stays to ground truth.

    Output: gif/gpu_lidar_slam.gif
    Headline: per-frame GPU time (scan + ICP + map update).
 ************************************************************************/

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include "cuda_check.cuh"

    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(EXIT_FAILURE); } } while (0)

constexpr float WORLD_X = 60.0f;
constexpr float WORLD_Y = 60.0f;
constexpr int   MAP_W = 600;
constexpr int   MAP_H = 600;
constexpr float MAP_RES = WORLD_X / MAP_W;
constexpr int   N_RAYS = 720;
constexpr float MAX_RANGE = 18.0f;
constexpr float NOISE_RANGE = 0.04f;
constexpr int   N_FRAMES = 200;
constexpr int   ICP_ITERS = 8;
constexpr int   PANEL_W = 480;
constexpr int   PANEL_H = 480;
constexpr float L_OCC =  0.85f;
constexpr float L_FREE = -0.35f;
constexpr float L_MIN = -4.0f;
constexpr float L_MAX =  4.0f;

// -------------------------------------------------------------------------
// Scene: pre-rasterised occupancy grid stored as binary unsigned char
// -------------------------------------------------------------------------
struct Disk { float cx, cy, r; };

static void rasterise_walls(std::vector<unsigned char>& occ) {
    occ.assign(MAP_W * MAP_H, 0u);
    auto fill = [&](float x0, float y0, float x1, float y1) {
        int gx0 = std::max(0, static_cast<int>(x0 / MAP_RES));
        int gy0 = std::max(0, static_cast<int>(y0 / MAP_RES));
        int gx1 = std::min(MAP_W - 1, static_cast<int>(x1 / MAP_RES));
        int gy1 = std::min(MAP_H - 1, static_cast<int>(y1 / MAP_RES));
        for (int gy = gy0; gy <= gy1; gy++)
            for (int gx = gx0; gx <= gx1; gx++) occ[gy * MAP_W + gx] = 1u;
    };
    fill(0.0f, 0.0f, WORLD_X, 0.4f);
    fill(0.0f, WORLD_Y - 0.4f, WORLD_X, WORLD_Y);
    fill(0.0f, 0.0f, 0.4f, WORLD_Y);
    fill(WORLD_X - 0.4f, 0.0f, WORLD_X, WORLD_Y);
    fill(20.0f, 18.0f, 21.0f, 42.0f);
    fill(40.0f, 14.0f, 41.0f, 38.0f);
    fill(10.0f, 28.0f, 30.0f, 29.0f);
    fill(35.0f, 48.0f, 50.0f, 49.0f);
    Disk disks[] = {
        { 8.0f, 10.0f, 1.6f}, { 12.0f, 50.0f, 1.4f}, {30.0f, 50.0f, 1.8f},
        {45.0f, 24.0f, 1.7f}, {52.0f, 12.0f, 1.5f}, {28.0f, 38.0f, 1.2f},
        {18.0f, 38.0f, 1.0f}, {48.0f, 38.0f, 1.4f},
    };
    for (const auto& d : disks) {
        int gx0 = std::max(0, static_cast<int>((d.cx - d.r) / MAP_RES));
        int gy0 = std::max(0, static_cast<int>((d.cy - d.r) / MAP_RES));
        int gx1 = std::min(MAP_W - 1, static_cast<int>((d.cx + d.r) / MAP_RES));
        int gy1 = std::min(MAP_H - 1, static_cast<int>((d.cy + d.r) / MAP_RES));
        for (int gy = gy0; gy <= gy1; gy++) {
            float wy = (gy + 0.5f) * MAP_RES;
            for (int gx = gx0; gx <= gx1; gx++) {
                float wx = (gx + 0.5f) * MAP_RES;
                float dx = wx - d.cx, dy = wy - d.cy;
                if (dx * dx + dy * dy <= d.r * d.r) occ[gy * MAP_W + gx] = 1u;
            }
        }
    }
}

// -------------------------------------------------------------------------
// Raycast 2D kernel: 1 ray = 1 thread, DDA on occupancy grid
// -------------------------------------------------------------------------
__global__ void raycast_2d_kernel(const unsigned char* __restrict__ occ,
                                  float sx, float sy, float syaw,
                                  float* __restrict__ scan_x,
                                  float* __restrict__ scan_y,
                                  float* __restrict__ scan_r,
                                  unsigned char* __restrict__ scan_hit,
                                  int n_rays) {
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= n_rays) return;
    float ang = syaw + (float)rid / n_rays * 2.0f * (float)M_PI;
    float cx = cosf(ang), sy_ = sinf(ang);
    // DDA
    float fx = sx / MAP_RES, fy = sy / MAP_RES;
    int gx = (int)floorf(fx), gy = (int)floorf(fy);
    int step_x = (cx > 0.0f) ? 1 : -1;
    int step_y = (sy_ > 0.0f) ? 1 : -1;
    float inv_dx = (fabsf(cx) > 1e-7f) ? 1.0f / fabsf(cx) : 1.0e30f;
    float inv_dy = (fabsf(sy_) > 1e-7f) ? 1.0f / fabsf(sy_) : 1.0e30f;
    float t_max_x = (cx > 0.0f) ? (gx + 1 - fx) * MAP_RES * inv_dx
                                : (fx - gx) * MAP_RES * inv_dx;
    float t_max_y = (sy_ > 0.0f) ? (gy + 1 - fy) * MAP_RES * inv_dy
                                 : (fy - gy) * MAP_RES * inv_dy;
    float dt_x = MAP_RES * inv_dx, dt_y = MAP_RES * inv_dy;
    float hit_range = MAX_RANGE;
    bool hit = false;
    int max_iter = MAP_W + MAP_H;
    for (int it = 0; it < max_iter; it++) {
        if (gx < 0 || gx >= MAP_W || gy < 0 || gy >= MAP_H) break;
        if (occ[gy * MAP_W + gx] != 0u) {
            // hit at start of this voxel = min(t_max_x, t_max_y) of the
            // previous step, approx
            float t = fminf(t_max_x, t_max_y) - 0.5f * fminf(dt_x, dt_y);
            if (t < 0.0f) t = 0.0f;
            hit_range = fminf(t, MAX_RANGE);
            hit = true;
            break;
        }
        if (t_max_x < t_max_y) { gx += step_x; t_max_x += dt_x; }
        else                    { gy += step_y; t_max_y += dt_y; }
        if (fminf(t_max_x, t_max_y) >= MAX_RANGE) break;
    }
    // store as scan endpoint in SENSOR frame (so subsequent ICP is on
    // relative differences). store hit flag separately.
    if (!hit) hit_range = MAX_RANGE;
    scan_r[rid] = hit_range;
    // sensor frame: pure direction times range
    scan_x[rid] = hit_range * cosf((float)rid / n_rays * 2.0f * (float)M_PI);
    scan_y[rid] = hit_range * sinf((float)rid / n_rays * 2.0f * (float)M_PI);
    scan_hit[rid] = hit ? 1u : 0u;
}

__global__ void add_range_noise_kernel(float* scan_x, float* scan_y, float* scan_r,
                                       unsigned char* scan_hit, unsigned long long seed) {
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= N_RAYS) return;
    if (!scan_hit[rid]) return;
    // simple LCG to avoid pulling in curand for this tiny perturbation
    unsigned long long s = (seed + rid) * 0xd1342543de82ef95ULL + 0x9e3779b97f4a7c15ULL;
    s ^= s >> 33;
    unsigned int u = (unsigned int)s;
    float r0 = (float)u / 4294967296.0f;
    s = s * 0xbf58476d1ce4e5b9ULL + 0x94d049bb133111ebULL;
    unsigned int u2 = (unsigned int)(s >> 16);
    float r1 = (float)u2 / 65536.0f / 65536.0f;
    // Box-Muller
    float z = sqrtf(-2.0f * logf(fmaxf(r0, 1e-9f))) * cosf(2.0f * (float)M_PI * r1);
    float dr = z * NOISE_RANGE;
    float ang = (float)rid / N_RAYS * 2.0f * (float)M_PI;
    scan_x[rid] += dr * cosf(ang);
    scan_y[rid] += dr * sinf(ang);
    scan_r[rid] += dr;
}

// -------------------------------------------------------------------------
// ICP: brute-force nearest neighbour + 2D Procrustes
// -------------------------------------------------------------------------
__global__ void transform_kernel(const float* __restrict__ in_x,
                                 const float* __restrict__ in_y,
                                 float ct, float st, float tx, float ty,
                                 float* __restrict__ out_x, float* __restrict__ out_y,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in_x[i], y = in_y[i];
    out_x[i] = ct * x - st * y + tx;
    out_y[i] = st * x + ct * y + ty;
}

__global__ void nn_correspondence_kernel(
    const float* __restrict__ qx, const float* __restrict__ qy,
    const unsigned char* __restrict__ q_hit,
    const float* __restrict__ tx_pts, const float* __restrict__ ty_pts,
    const unsigned char* __restrict__ t_hit,
    int n,
    int*  __restrict__ best_idx,
    float* __restrict__ best_d2,
    float max_corr_d2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!q_hit[i]) { best_idx[i] = -1; best_d2[i] = max_corr_d2; return; }
    float qxi = qx[i], qyi = qy[i];
    float best = max_corr_d2;
    int bidx = -1;
    for (int j = 0; j < n; j++) {
        if (!t_hit[j]) continue;
        float dx = qxi - tx_pts[j];
        float dy = qyi - ty_pts[j];
        float d2 = dx * dx + dy * dy;
        if (d2 < best) { best = d2; bidx = j; }
    }
    best_idx[i] = bidx;
    best_d2[i]  = best;
}

__global__ void accumulate_correspondences_kernel(
    const float* __restrict__ qx, const float* __restrict__ qy,
    const float* __restrict__ tx_pts, const float* __restrict__ ty_pts,
    const int* __restrict__ best_idx, int n,
    float* __restrict__ sums)
{
    // sums[0..1]  = sum q (matched only)
    // sums[2..3]  = sum t (matched only)
    // sums[4]     = matched count (cast to float)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int bi = best_idx[i];
    if (bi < 0) return;
    atomicAdd(&sums[0], qx[i]);
    atomicAdd(&sums[1], qy[i]);
    atomicAdd(&sums[2], tx_pts[bi]);
    atomicAdd(&sums[3], ty_pts[bi]);
    atomicAdd(&sums[4], 1.0f);
}

__global__ void cross_correlation_kernel(
    const float* __restrict__ qx, const float* __restrict__ qy,
    const float* __restrict__ tx_pts, const float* __restrict__ ty_pts,
    const int* __restrict__ best_idx, int n,
    float mu_qx, float mu_qy, float mu_px, float mu_py,
    float* __restrict__ corr) // corr[0..3] = S_xx, S_xy, S_yx, S_yy
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int bi = best_idx[i];
    if (bi < 0) return;
    float qcx = qx[i] - mu_qx;
    float qcy = qy[i] - mu_qy;
    float pcx = tx_pts[bi] - mu_px;
    float pcy = ty_pts[bi] - mu_py;
    atomicAdd(&corr[0], qcx * pcx);
    atomicAdd(&corr[1], qcx * pcy);
    atomicAdd(&corr[2], qcy * pcx);
    atomicAdd(&corr[3], qcy * pcy);
}

__global__ void map_update_kernel(
    const float* __restrict__ scan_x, const float* __restrict__ scan_y,
    const unsigned char* __restrict__ scan_hit,
    float sx, float sy, int n_rays, float* __restrict__ log_odds)
{
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= n_rays) return;
    float hx = scan_x[rid], hy = scan_y[rid];
    bool hit = scan_hit[rid] != 0u;
    float dx = hx - sx, dy = hy - sy;
    float length = sqrtf(dx * dx + dy * dy);
    if (length < 1e-3f) return;
    float ux = dx / length, uy = dy / length;
    // DDA traversal
    float fx = sx / MAP_RES, fy = sy / MAP_RES;
    int gx = (int)floorf(fx), gy = (int)floorf(fy);
    int step_x = (ux > 0.0f) ? 1 : -1;
    int step_y = (uy > 0.0f) ? 1 : -1;
    float inv_dx = (fabsf(ux) > 1e-7f) ? 1.0f / fabsf(ux) : 1e30f;
    float inv_dy = (fabsf(uy) > 1e-7f) ? 1.0f / fabsf(uy) : 1e30f;
    float t_max_x = (ux > 0.0f) ? (gx + 1 - fx) * MAP_RES * inv_dx
                                : (fx - gx) * MAP_RES * inv_dx;
    float t_max_y = (uy > 0.0f) ? (gy + 1 - fy) * MAP_RES * inv_dy
                                : (fy - gy) * MAP_RES * inv_dy;
    float dt_x = MAP_RES * inv_dx, dt_y = MAP_RES * inv_dy;
    int max_iter = MAP_W + MAP_H;
    for (int it = 0; it < max_iter; it++) {
        if (gx < 0 || gx >= MAP_W || gy < 0 || gy >= MAP_H) break;
        float t_next = fminf(t_max_x, t_max_y);
        if (t_next >= length) break;
        // free space update
        int idx = gy * MAP_W + gx;
        float old = atomicAdd(&log_odds[idx], L_FREE);
        float v = old + L_FREE;
        if (v < L_MIN) atomicAdd(&log_odds[idx], L_MIN - v);
        if (t_max_x < t_max_y) { gx += step_x; t_max_x += dt_x; }
        else                    { gy += step_y; t_max_y += dt_y; }
    }
    if (hit) {
        int hgx = (int)floorf(hx / MAP_RES);
        int hgy = (int)floorf(hy / MAP_RES);
        if (hgx >= 0 && hgx < MAP_W && hgy >= 0 && hgy < MAP_H) {
            int idx = hgy * MAP_W + hgx;
            float old = atomicAdd(&log_odds[idx], L_OCC - L_FREE);
            float v = old + (L_OCC - L_FREE);
            if (v > L_MAX) atomicAdd(&log_odds[idx], L_MAX - v);
        }
    }
}

__global__ void zero_kernel(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0.0f;
}

// -------------------------------------------------------------------------
// Host helpers
// -------------------------------------------------------------------------
static double sf(const float* d) {
    float h;
    cudaMemcpy(&h, d, sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

static cv::Mat render_map(const std::vector<float>& log_odds,
                          const std::vector<float>& path_gt,
                          const std::vector<float>& path_est,
                          float robot_x, float robot_y,
                          const char* title) {
    cv::Mat img(MAP_H, MAP_W, CV_8UC3, cv::Scalar(40, 40, 40));
    for (int gy = 0; gy < MAP_H; gy++) {
        for (int gx = 0; gx < MAP_W; gx++) {
            float l = log_odds[gy * MAP_W + gx];
            float p = 1.0f / (1.0f + std::exp(-l));
            int v = static_cast<int>((1.0f - p) * 200.0f + 40.0f);
            if (v < 0) v = 0; if (v > 240) v = 240;
            img.at<cv::Vec3b>(MAP_H - 1 - gy, gx) = cv::Vec3b(v, v, v);
        }
    }
    auto draw_path = [&](const std::vector<float>& path, cv::Scalar color, int thick) {
        for (size_t i = 2; i < path.size(); i += 2) {
            cv::Point a(static_cast<int>(path[i - 2] / WORLD_X * MAP_W),
                        static_cast<int>((1.0f - path[i - 1] / WORLD_Y) * MAP_H));
            cv::Point b(static_cast<int>(path[i] / WORLD_X * MAP_W),
                        static_cast<int>((1.0f - path[i + 1] / WORLD_Y) * MAP_H));
            cv::line(img, a, b, color, thick, cv::LINE_AA);
        }
    };
    draw_path(path_gt, cv::Scalar(100, 220, 100), 1);
    draw_path(path_est, cv::Scalar(60, 60, 230), 2);
    int rx = static_cast<int>(robot_x / WORLD_X * MAP_W);
    int ry = static_cast<int>((1.0f - robot_y / WORLD_Y) * MAP_H);
    cv::circle(img, cv::Point(rx, ry), 5, cv::Scalar(255, 255, 255), 2);
    cv::Mat out;
    cv::resize(img, out, cv::Size(PANEL_W, PANEL_H), 0, 0, cv::INTER_AREA);
    cv::rectangle(out, cv::Rect(0, 0, PANEL_W, 26), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(out, title, cv::Point(10, 18), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return out;
}

static cv::Mat render_gt(const std::vector<unsigned char>& occ_gt,
                         const std::vector<float>& path_gt,
                         float robot_x, float robot_y, const char* title) {
    cv::Mat img(MAP_H, MAP_W, CV_8UC3, cv::Scalar(70, 70, 70));
    for (int gy = 0; gy < MAP_H; gy++)
        for (int gx = 0; gx < MAP_W; gx++)
            if (occ_gt[gy * MAP_W + gx])
                img.at<cv::Vec3b>(MAP_H - 1 - gy, gx) = cv::Vec3b(20, 20, 20);
    for (size_t i = 2; i < path_gt.size(); i += 2) {
        cv::Point a(static_cast<int>(path_gt[i - 2] / WORLD_X * MAP_W),
                    static_cast<int>((1.0f - path_gt[i - 1] / WORLD_Y) * MAP_H));
        cv::Point b(static_cast<int>(path_gt[i] / WORLD_X * MAP_W),
                    static_cast<int>((1.0f - path_gt[i + 1] / WORLD_Y) * MAP_H));
        cv::line(img, a, b, cv::Scalar(100, 220, 100), 2, cv::LINE_AA);
    }
    int rx = static_cast<int>(robot_x / WORLD_X * MAP_W);
    int ry = static_cast<int>((1.0f - robot_y / WORLD_Y) * MAP_H);
    cv::circle(img, cv::Point(rx, ry), 5, cv::Scalar(255, 255, 255), 2);
    cv::Mat out;
    cv::resize(img, out, cv::Size(PANEL_W, PANEL_H), 0, 0, cv::INTER_AREA);
    cv::rectangle(out, cv::Rect(0, 0, PANEL_W, 26), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(out, title, cv::Point(10, 18), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return out;
}

static void convert_avi_to_gif(const char* avi, const char* gif, int fps) {
    char cmd[512];
    std::snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -i %s -vf 'fps=%d,scale=1100:-1:flags=lanczos' -loop 0 %s "
        "> /dev/null 2>&1", avi, fps, gif);
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg conversion returned %d\n", rc);
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------
int main() {
    std::vector<unsigned char> occ_gt;
    rasterise_walls(occ_gt);

    // Device side
    unsigned char* d_occ_gt = nullptr;
    float *d_log = nullptr;
    CUDA_CHECK(cudaMalloc(&d_occ_gt, occ_gt.size()));
    CUDA_CHECK(cudaMalloc(&d_log, MAP_W * MAP_H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_occ_gt, occ_gt.data(), occ_gt.size(),
                          cudaMemcpyHostToDevice));
    int t = 256, b_log = (MAP_W * MAP_H + t - 1) / t;
    zero_kernel<<<b_log, t>>>(d_log, MAP_W * MAP_H);

    // Scan buffers
    float *d_scan_x = nullptr, *d_scan_y = nullptr, *d_scan_r = nullptr;
    unsigned char* d_scan_hit = nullptr;
    float *d_prev_x = nullptr, *d_prev_y = nullptr;
    unsigned char* d_prev_hit = nullptr;
    float *d_scan_world_x = nullptr, *d_scan_world_y = nullptr;
    float *d_query_x = nullptr, *d_query_y = nullptr;
    int*   d_best_idx = nullptr;
    float* d_best_d2 = nullptr;
    float* d_sums = nullptr;
    float* d_corr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scan_x,    N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_y,    N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_r,    N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_hit,  N_RAYS));
    CUDA_CHECK(cudaMalloc(&d_prev_x,    N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prev_y,    N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prev_hit,  N_RAYS));
    CUDA_CHECK(cudaMalloc(&d_scan_world_x, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_world_y, N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_query_x,   N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_query_y,   N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_best_idx,  N_RAYS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_d2,   N_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums,      5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_corr,      4 * sizeof(float)));

    // Ground truth trajectory: figure-8
    std::vector<float> gt_traj;
    std::vector<float> est_traj;
    gt_traj.reserve(N_FRAMES * 2);
    est_traj.reserve(N_FRAMES * 2);

    cv::VideoWriter video("gif/gpu_lidar_slam.avi",
                          cv::VideoWriter::fourcc('X','V','I','D'), 15,
                          cv::Size(PANEL_W * 2 + 4, PANEL_H + 30));

    float est_x = 30.0f, est_y = 30.0f, est_yaw = 0.0f;
    float prev_gt_x = 30.0f, prev_gt_y = 30.0f, prev_gt_yaw = 0.0f;
    bool have_prev_scan = false;

    int blocks_rays = (N_RAYS + t - 1) / t;
    double total_frame_ms = 0.0;
    int counted_frames = 0;

    std::vector<float> h_log(MAP_W * MAP_H);

    for (int frame = 0; frame < N_FRAMES; frame++) {
        float u = (float)frame / (N_FRAMES - 1) * 4.0f * (float)M_PI;
        float gt_x = 30.0f + 15.0f * std::sin(u * 0.5f);
        float gt_y = 30.0f + 10.0f * std::sin(u);
        // omnidirectional 2D LiDAR — sensor body frame stays world-aligned
        float gt_yaw = 0.0f;
        gt_traj.push_back(gt_x);
        gt_traj.push_back(gt_y);

        auto t0 = std::chrono::high_resolution_clock::now();

        // produce scan at ground-truth pose
        raycast_2d_kernel<<<blocks_rays, t>>>(d_occ_gt, gt_x, gt_y, gt_yaw,
                                              d_scan_x, d_scan_y, d_scan_r, d_scan_hit,
                                              N_RAYS);
        add_range_noise_kernel<<<blocks_rays, t>>>(d_scan_x, d_scan_y, d_scan_r,
                                                   d_scan_hit, frame * 7919ULL);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (have_prev_scan) {
            // initialise transform from incremental gt motion (a noisy "odom")
            // — for fairness vs full SLAM, we just use zero (assume robot
            //   didn't move) and let ICP find the relative pose.
            float dxt = 0.0f, dyt = 0.0f, dyaw = 0.0f;
            for (int it = 0; it < ICP_ITERS; it++) {
                float ct = std::cos(dyaw), st = std::sin(dyaw);
                transform_kernel<<<blocks_rays, t>>>(d_scan_x, d_scan_y,
                                                      ct, st, dxt, dyt,
                                                      d_query_x, d_query_y, N_RAYS);
                float max_corr_d2 = 1.0f;  // 1.0 m^2 gate
                nn_correspondence_kernel<<<blocks_rays, t>>>(
                    d_query_x, d_query_y, d_scan_hit,
                    d_prev_x,  d_prev_y,  d_prev_hit,
                    N_RAYS, d_best_idx, d_best_d2, max_corr_d2);

                zero_kernel<<<1, 5>>>(d_sums, 5);
                accumulate_correspondences_kernel<<<blocks_rays, t>>>(
                    d_query_x, d_query_y, d_prev_x, d_prev_y,
                    d_best_idx, N_RAYS, d_sums);
                CUDA_CHECK(cudaDeviceSynchronize());
                float s[5];
                CUDA_CHECK(cudaMemcpy(s, d_sums, 5 * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                if (s[4] < 5.0f) break;
                float mu_qx = s[0] / s[4], mu_qy = s[1] / s[4];
                float mu_px = s[2] / s[4], mu_py = s[3] / s[4];
                zero_kernel<<<1, 4>>>(d_corr, 4);
                cross_correlation_kernel<<<blocks_rays, t>>>(
                    d_query_x, d_query_y, d_prev_x, d_prev_y,
                    d_best_idx, N_RAYS, mu_qx, mu_qy, mu_px, mu_py, d_corr);
                CUDA_CHECK(cudaDeviceSynchronize());
                float c[4];
                CUDA_CHECK(cudaMemcpy(c, d_corr, 4 * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                // 2D rotation: theta = atan2(S_xy - S_yx, S_xx + S_yy)
                // (we want R that maps q->p, i.e. R*q + t = p)
                float dtheta = std::atan2(c[1] - c[2], c[0] + c[3]);
                float ct2 = std::cos(dtheta), st2 = std::sin(dtheta);
                float dtx = mu_px - (ct2 * mu_qx - st2 * mu_qy);
                float dty = mu_py - (st2 * mu_qx + ct2 * mu_qy);
                // compose with existing increment
                float new_ct = ct2 * ct - st2 * st;
                float new_st = st2 * ct + ct2 * st;
                float new_yaw = std::atan2(new_st, new_ct);
                float new_tx = ct2 * dxt - st2 * dyt + dtx;
                float new_ty = st2 * dxt + ct2 * dyt + dty;
                dxt = new_tx; dyt = new_ty; dyaw = new_yaw;
                if (std::fabs(dtheta) < 1e-4f &&
                    std::hypot(dtx, dty) < 1e-4f) break;
            }
            // ICP yields (dxt, dyt, dyaw) such that R(dyaw)*q + t = p where
            // q is the new scan in NEW sensor frame and p is the previous
            // scan in PREVIOUS sensor frame. Hence T_old_new = (R(dyaw), t)
            // is "new sensor pose expressed in old sensor frame", and we
            // compose: T_world_new = T_world_old * T_old_new.
            float c_yaw = std::cos(est_yaw), s_yaw = std::sin(est_yaw);
            est_x   += c_yaw * dxt - s_yaw * dyt;
            est_y   += s_yaw * dxt + c_yaw * dyt;
            est_yaw += dyaw;
        }
        est_traj.push_back(est_x);
        est_traj.push_back(est_y);

        // Transform scan to global frame using estimated pose
        float ce = std::cos(est_yaw), se = std::sin(est_yaw);
        transform_kernel<<<blocks_rays, t>>>(d_scan_x, d_scan_y,
                                              ce, se, est_x, est_y,
                                              d_scan_world_x, d_scan_world_y, N_RAYS);

        // Update occupancy map with the registered scan
        map_update_kernel<<<blocks_rays, t>>>(d_scan_world_x, d_scan_world_y, d_scan_hit,
                                              est_x, est_y, N_RAYS, d_log);

        // Cache previous scan (sensor frame)
        CUDA_CHECK(cudaMemcpy(d_prev_x,   d_scan_x,   N_RAYS * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_prev_y,   d_scan_y,   N_RAYS * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_prev_hit, d_scan_hit, N_RAYS,
                              cudaMemcpyDeviceToDevice));
        have_prev_scan = true;

        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (frame >= 5) { total_frame_ms += ms; counted_frames++; }

        if (frame % 2 == 0 || frame == N_FRAMES - 1) {
            CUDA_CHECK(cudaMemcpy(h_log.data(), d_log,
                                  MAP_W * MAP_H * sizeof(float), cudaMemcpyDeviceToHost));
            cv::Mat map_panel = render_map(h_log, gt_traj, est_traj, est_x, est_y,
                                           "GPU SLAM estimate");
            cv::Mat gt_panel = render_gt(occ_gt, gt_traj, gt_x, gt_y, "ground truth");
            cv::Mat frame_img(PANEL_H + 30, PANEL_W * 2 + 4, CV_8UC3,
                              cv::Scalar(30, 30, 30));
            map_panel.copyTo(frame_img(cv::Rect(0, 30, PANEL_W, PANEL_H)));
            gt_panel.copyTo(frame_img(cv::Rect(PANEL_W + 4, 30, PANEL_W, PANEL_H)));
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "GPU LiDAR SLAM (scan-to-scan ICP + log-odds map)  "
                          "frame %d/%d  est=(%.2f, %.2f)  gt=(%.2f, %.2f)  %.2f ms",
                          frame + 1, N_FRAMES, est_x, est_y, gt_x, gt_y, ms);
            cv::putText(frame_img, buf, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX,
                        0.46, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            video.write(frame_img);
        }
        prev_gt_x = gt_x; prev_gt_y = gt_y; prev_gt_yaw = gt_yaw;
    }
    video.release();
    convert_avi_to_gif("gif/gpu_lidar_slam.avi", "gif/gpu_lidar_slam.gif", 15);

    if (counted_frames > 0) {
        std::printf("Avg per-frame GPU time (scan + ICP + map): %.2f ms (%d rays)\n",
                    total_frame_ms / counted_frames, N_RAYS);
    }
    // pose drift at the end
    float dx = est_traj[est_traj.size() - 2] - gt_traj[gt_traj.size() - 2];
    float dy = est_traj[est_traj.size() - 1] - gt_traj[gt_traj.size() - 1];
    std::printf("Final pose drift: %.3f m\n", std::hypot(dx, dy));
    std::printf("GIF saved to gif/gpu_lidar_slam.gif\n");

    for (auto* p : {d_log, d_scan_x, d_scan_y, d_scan_r,
                    d_prev_x, d_prev_y, d_scan_world_x, d_scan_world_y,
                    d_query_x, d_query_y, d_best_d2, d_sums, d_corr}) {
        CUDA_CHECK(cudaFree(p));
    }
    CUDA_CHECK(cudaFree(d_occ_gt));
    CUDA_CHECK(cudaFree(d_scan_hit));
    CUDA_CHECK(cudaFree(d_prev_hit));
    CUDA_CHECK(cudaFree(d_best_idx));
    return 0;
}
