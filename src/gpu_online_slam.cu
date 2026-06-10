// gpu_online_slam.cu
//
// GPU online SLAM: sliding-window pose-graph optimizer that runs as new
// frames stream in.  Combines the offline pose-graph backend from #58 with
// an incremental front-end that adds odom edges per frame and detects loop
// closures against past poses.
//
// At each step t:
//   - The active window is [max(0, t-WINDOW+1), t].  Poses outside the
//     window are fixed at their current estimate (anchored).
//   - Odom edge (t-1, t) and any new loop-closure edges touching the active
//     range are added.
//   - K_GN Gauss-Newton iterations of PCG are run on the active window
//     only.  Atomic-add assembly skips updates to fixed poses.
//
// Output: gif/gpu_online_slam.gif — two panels, "odometry only" vs
// "live SLAM estimate", both grown frame-by-frame.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "cuda_check.cuh"

    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                         cudaGetErrorString(err), __FILE__, __LINE__);    \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

namespace cudabot {

constexpr int N_FRAMES = 600;        // total frames streamed
constexpr int WINDOW   = 60;         // sliding-window size
constexpr int GN_ITERS = 4;          // GN iters per frame
constexpr int PCG_ITERS = 40;        // PCG iters per GN
constexpr int GLOBAL_GN_ITERS = 6;   // GN iters when a loop fires (whole graph)
constexpr int GLOBAL_PCG_ITERS = 80;
constexpr int LOOP_BURST_FRAMES = 30; // run global if a new loop within this many steps
constexpr float PCG_TOL = 1.0e-7f;

constexpr float ODOM_SIGMA_XY = 0.05f;
constexpr float ODOM_SIGMA_TH = 0.018f;
constexpr float LC_SIGMA_XY = 0.04f;
constexpr float LC_SIGMA_TH = 0.012f;

constexpr float LC_DIST = 1.8f;
constexpr int   LC_MIN_GAP = 30;     // need this many frames between i and j
constexpr int   LC_PER_FRAME_MAX = 2;

constexpr int   PANEL_W = 540;
constexpr int   PANEL_H = 540;
constexpr int   VIDEO_FPS = 24;
constexpr int   VIDEO_STRIDE = 2;    // write one frame every N updates

struct Edge {
    int i, j;
    float zx, zy, zt;
};

static inline float wrap_angle(float a) {
    while (a >  M_PI) a -= 2.0f * M_PI;
    while (a < -M_PI) a += 2.0f * M_PI;
    return a;
}

// -------------------------------------------------------------------------
// CUDA kernels (sliding-window variant of #58)
// -------------------------------------------------------------------------

// active_lo, active_hi: only poses in [active_lo, active_hi) accept updates.
// Fixed poses still appear in edges as constants.
__global__ void assemble_kernel(int n_edges,
                                const int* __restrict__ ei,
                                const int* __restrict__ ej,
                                const float* __restrict__ ez,
                                const float* __restrict__ poses,
                                float omega_xy, float omega_th,
                                int active_lo, int active_hi,
                                float* __restrict__ b,
                                float* __restrict__ diag) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;

    int i = ei[e], j = ej[e];
    bool i_active = (i >= active_lo && i < active_hi);
    bool j_active = (j >= active_lo && j < active_hi);
    if (!i_active && !j_active) return;

    float zx = ez[3 * e + 0];
    float zy = ez[3 * e + 1];
    float zt = ez[3 * e + 2];

    float xi = poses[3 * i + 0];
    float yi = poses[3 * i + 1];
    float ti = poses[3 * i + 2];
    float xj = poses[3 * j + 0];
    float yj = poses[3 * j + 1];
    float tj = poses[3 * j + 2];

    float c = cosf(ti), s = sinf(ti);
    float dxw = xj - xi;
    float dyw = yj - yi;
    float dx = dxw * c + dyw * s;
    float dy = -dxw * s + dyw * c;
    float dt = tj - ti;
    while (dt >  M_PI) dt -= 2.0f * M_PI;
    while (dt < -M_PI) dt += 2.0f * M_PI;

    float rx = dx - zx;
    float ry = dy - zy;
    float rt = dt - zt;
    while (rt >  M_PI) rt -= 2.0f * M_PI;
    while (rt < -M_PI) rt += 2.0f * M_PI;

    float Ji[9] = {
        -c, -s, -dxw * s + dyw * c,
         s, -c, -dxw * c - dyw * s,
         0.0f, 0.0f, -1.0f
    };
    float Jj[9] = {
         c,  s, 0.0f,
        -s,  c, 0.0f,
         0.0f, 0.0f, 1.0f
    };
    float Wr[3] = { omega_xy * rx, omega_xy * ry, omega_th * rt };
    float w[3] = { omega_xy, omega_xy, omega_th };

    if (i_active) {
        for (int k = 0; k < 3; k++) {
            float bi_k = Ji[3 * 0 + k] * Wr[0] + Ji[3 * 1 + k] * Wr[1] + Ji[3 * 2 + k] * Wr[2];
            atomicAdd(&b[3 * i + k], bi_k);
        }
        for (int p = 0; p < 3; p++) {
            for (int q = 0; q < 3; q++) {
                float v = 0.0f;
                for (int kk = 0; kk < 3; kk++) v += Ji[3 * kk + p] * w[kk] * Ji[3 * kk + q];
                atomicAdd(&diag[9 * i + 3 * p + q], v);
            }
        }
    }
    if (j_active) {
        for (int k = 0; k < 3; k++) {
            float bj_k = Jj[3 * 0 + k] * Wr[0] + Jj[3 * 1 + k] * Wr[1] + Jj[3 * 2 + k] * Wr[2];
            atomicAdd(&b[3 * j + k], bj_k);
        }
        for (int p = 0; p < 3; p++) {
            for (int q = 0; q < 3; q++) {
                float v = 0.0f;
                for (int kk = 0; kk < 3; kk++) v += Jj[3 * kk + p] * w[kk] * Jj[3 * kk + q];
                atomicAdd(&diag[9 * j + 3 * p + q], v);
            }
        }
    }
}

// Anchor pose `active_lo` (boundary of window): clamp its delta to zero by
// zero-ing its b row and forcing identity diag.  This pins the chain end.
__global__ void anchor_boundary_kernel(float* b, float* diag, int anchor_idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int base = 3 * anchor_idx;
        int dbase = 9 * anchor_idx;
        b[base + 0] = 0.0f;
        b[base + 1] = 0.0f;
        b[base + 2] = 0.0f;
        for (int p = 0; p < 9; p++) diag[dbase + p] = 0.0f;
        diag[dbase + 0] = 1.0f;
        diag[dbase + 4] = 1.0f;
        diag[dbase + 8] = 1.0f;
    }
}

__global__ void matvec_kernel(int n_edges,
                              const int* __restrict__ ei,
                              const int* __restrict__ ej,
                              const float* __restrict__ poses,
                              float omega_xy, float omega_th,
                              int active_lo, int active_hi,
                              const float* __restrict__ x,
                              float* __restrict__ y) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    int i = ei[e], j = ej[e];
    bool i_active = (i >= active_lo && i < active_hi);
    bool j_active = (j >= active_lo && j < active_hi);
    if (!i_active && !j_active) return;

    float ti = poses[3 * i + 2];
    float dxw = poses[3 * j + 0] - poses[3 * i + 0];
    float dyw = poses[3 * j + 1] - poses[3 * i + 1];
    float c = cosf(ti), s = sinf(ti);
    float Ji[9] = {
        -c, -s, -dxw * s + dyw * c,
         s, -c, -dxw * c - dyw * s,
         0.0f, 0.0f, -1.0f
    };
    float Jj[9] = {
         c,  s, 0.0f,
        -s,  c, 0.0f,
         0.0f, 0.0f, 1.0f
    };

    // x contributions are zero for fixed nodes (they're never written).
    float xi[3] = { i_active ? x[3 * i + 0] : 0.0f,
                    i_active ? x[3 * i + 1] : 0.0f,
                    i_active ? x[3 * i + 2] : 0.0f };
    float xj[3] = { j_active ? x[3 * j + 0] : 0.0f,
                    j_active ? x[3 * j + 1] : 0.0f,
                    j_active ? x[3 * j + 2] : 0.0f };

    float u[3] = {0.0f, 0.0f, 0.0f};
    for (int r = 0; r < 3; r++) {
        u[r] = Ji[3 * r + 0] * xi[0] + Ji[3 * r + 1] * xi[1] + Ji[3 * r + 2] * xi[2]
             + Jj[3 * r + 0] * xj[0] + Jj[3 * r + 1] * xj[1] + Jj[3 * r + 2] * xj[2];
    }
    float w[3] = { omega_xy, omega_xy, omega_th };
    float Wu[3] = { w[0] * u[0], w[1] * u[1], w[2] * u[2] };

    if (i_active) {
        for (int k = 0; k < 3; k++) {
            float yi_k = Ji[3 * 0 + k] * Wu[0] + Ji[3 * 1 + k] * Wu[1] + Ji[3 * 2 + k] * Wu[2];
            atomicAdd(&y[3 * i + k], yi_k);
        }
    }
    if (j_active) {
        for (int k = 0; k < 3; k++) {
            float yj_k = Jj[3 * 0 + k] * Wu[0] + Jj[3 * 1 + k] * Wu[1] + Jj[3 * 2 + k] * Wu[2];
            atomicAdd(&y[3 * j + k], yj_k);
        }
    }
}

__global__ void apply_precond_kernel(int n_poses,
                                     int active_lo, int active_hi,
                                     const float* __restrict__ diag,
                                     const float* __restrict__ r,
                                     float* __restrict__ z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_poses) return;
    if (i < active_lo || i >= active_hi) {
        z[3 * i + 0] = 0.0f;
        z[3 * i + 1] = 0.0f;
        z[3 * i + 2] = 0.0f;
        return;
    }
    const float* D = diag + 9 * i;
    float m[9];
    for (int k = 0; k < 9; k++) m[k] = D[k];
    m[0] += 1.0e-6f; m[4] += 1.0e-6f; m[8] += 1.0e-6f;
    float det = m[0] * (m[4] * m[8] - m[5] * m[7])
              - m[1] * (m[3] * m[8] - m[5] * m[6])
              + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (fabsf(det) < 1.0e-12f) {
        z[3 * i + 0] = r[3 * i + 0];
        z[3 * i + 1] = r[3 * i + 1];
        z[3 * i + 2] = r[3 * i + 2];
        return;
    }
    float inv_det = 1.0f / det;
    float inv[9];
    inv[0] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
    inv[1] = -(m[1] * m[8] - m[2] * m[7]) * inv_det;
    inv[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
    inv[3] = -(m[3] * m[8] - m[5] * m[6]) * inv_det;
    inv[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
    inv[5] = -(m[0] * m[5] - m[2] * m[3]) * inv_det;
    inv[6] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
    inv[7] = -(m[0] * m[7] - m[1] * m[6]) * inv_det;
    inv[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;

    float rx = r[3 * i + 0], ry = r[3 * i + 1], rt = r[3 * i + 2];
    z[3 * i + 0] = inv[0] * rx + inv[1] * ry + inv[2] * rt;
    z[3 * i + 1] = inv[3] * rx + inv[4] * ry + inv[5] * rt;
    z[3 * i + 2] = inv[6] * rx + inv[7] * ry + inv[8] * rt;
}

__global__ void update_poses_kernel(int n_poses,
                                    int active_lo, int active_hi,
                                    float* __restrict__ poses,
                                    const float* __restrict__ dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_poses) return;
    if (i < active_lo || i >= active_hi) return;
    poses[3 * i + 0] += dx[3 * i + 0];
    poses[3 * i + 1] += dx[3 * i + 1];
    float th = poses[3 * i + 2] + dx[3 * i + 2];
    while (th >  M_PI) th -= 2.0f * M_PI;
    while (th < -M_PI) th += 2.0f * M_PI;
    poses[3 * i + 2] = th;
}

__global__ void dot_kernel(int n, const float* a, const float* b, float* out) {
    __shared__ float sm[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float v = 0.0f;
    for (int k = idx; k < n; k += gridDim.x * blockDim.x) v += a[k] * b[k];
    sm[tid] = v;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sm[0]);
}

__global__ void axpy_kernel(int n, float a, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] += a * x[idx];
}

__global__ void xpay_kernel(int n, float a, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] = x[idx] + a * y[idx];
}

__global__ void copy_kernel(int n, const float* src, float* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

__global__ void zero_kernel(int n, float* arr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = 0.0f;
}

// -------------------------------------------------------------------------
// Host helpers
// -------------------------------------------------------------------------
static void make_ground_truth(std::vector<float>& gt) {
    gt.assign(N_FRAMES * 3, 0.0f);
    // 3-petal Lissajous: x=12*sin(u), y=8*sin(3u/2) → comes back to similar
    // regions several times, producing genuine loop closures.
    for (int k = 0; k < N_FRAMES; k++) {
        float s = static_cast<float>(k) / (N_FRAMES - 1);
        float u = s * 4.0f * static_cast<float>(M_PI);
        float x = 12.0f * std::sin(u);
        float y =  8.0f * std::sin(1.5f * u);
        float dxds = 12.0f * std::cos(u);
        float dyds = 12.0f * std::cos(1.5f * u);
        float th = std::atan2(dyds, dxds);
        gt[3 * k + 0] = x;
        gt[3 * k + 1] = y;
        gt[3 * k + 2] = th;
    }
}

static cv::Point2i to_pixel(float x, float y, float scale, int cx, int cy) {
    return cv::Point2i(static_cast<int>(cx + scale * x),
                       static_cast<int>(cy - scale * y));
}

static cv::Mat draw_panel(const std::vector<float>& poses,
                          int up_to,
                          const std::vector<float>& gt,
                          const std::vector<Edge>* loops,
                          const std::string& title,
                          float rmse) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(20, 20, 20));
    float scale = 17.0f;
    int cx = PANEL_W / 2;
    int cy = PANEL_H / 2 + 10;
    // grid
    for (int g = -7; g <= 7; g++) {
        cv::line(img, to_pixel(g * 2, -14, scale, cx, cy),
                       to_pixel(g * 2,  14, scale, cx, cy),
                       cv::Scalar(40, 40, 40), 1);
        cv::line(img, to_pixel(-14, g * 2, scale, cx, cy),
                       to_pixel( 14, g * 2, scale, cx, cy),
                       cv::Scalar(40, 40, 40), 1);
    }
    // GT (faint)
    for (int k = 1; k < N_FRAMES; k++) {
        cv::line(img,
                 to_pixel(gt[3 * (k - 1) + 0], gt[3 * (k - 1) + 1], scale, cx, cy),
                 to_pixel(gt[3 * k + 0],       gt[3 * k + 1],       scale, cx, cy),
                 cv::Scalar(90, 90, 90), 1);
    }
    // loop edges drawn behind the trajectory
    if (loops) {
        for (const auto& lc : *loops) {
            if (lc.i > up_to || lc.j > up_to) continue;
            cv::line(img,
                     to_pixel(poses[3 * lc.i + 0], poses[3 * lc.i + 1], scale, cx, cy),
                     to_pixel(poses[3 * lc.j + 0], poses[3 * lc.j + 1], scale, cx, cy),
                     cv::Scalar(200, 200, 60), 1, cv::LINE_AA);
        }
    }
    // current poses
    cv::Scalar color = loops ? cv::Scalar(80, 220, 80) : cv::Scalar(80, 80, 240);
    for (int k = 1; k <= up_to; k++) {
        cv::line(img,
                 to_pixel(poses[3 * (k - 1) + 0], poses[3 * (k - 1) + 1], scale, cx, cy),
                 to_pixel(poses[3 * k + 0],       poses[3 * k + 1],       scale, cx, cy),
                 color, 2, cv::LINE_AA);
    }
    cv::circle(img, to_pixel(poses[3 * up_to + 0], poses[3 * up_to + 1], scale, cx, cy),
               4, cv::Scalar(255, 255, 255), -1);
    cv::putText(img, title, cv::Point(10, 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
    char buf[160];
    int nloops = loops ? static_cast<int>(loops->size()) : 0;
    std::snprintf(buf, sizeof(buf),
                  "frame %d/%d   RMSE = %.3f m%s%s",
                  up_to + 1, N_FRAMES, rmse,
                  loops ? "   loops = " : "",
                  loops ? "" : "");
    cv::putText(img, buf, cv::Point(10, PANEL_H - 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1);
    if (loops) {
        char buf2[64];
        std::snprintf(buf2, sizeof(buf2), "%d", nloops);
        cv::putText(img, buf2, cv::Point(PANEL_W - 60, PANEL_H - 14),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 60), 1);
    }
    return img;
}

static void convert_avi_to_gif(const std::string& avi, const std::string& gif, int fps) {
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -i %s -vf \"fps=%d,scale=1080:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle\" %s 2>/dev/null",
                  avi.c_str(), fps, gif.c_str());
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg failed (%d)\n", rc);
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<float> gt;
    make_ground_truth(gt);

    std::mt19937 rng(11);
    std::normal_distribution<float> n_xy(0.0f, ODOM_SIGMA_XY);
    std::normal_distribution<float> n_th(0.0f, ODOM_SIGMA_TH);
    std::normal_distribution<float> nlc_xy(0.0f, LC_SIGMA_XY);
    std::normal_distribution<float> nlc_th(0.0f, LC_SIGMA_TH);

    // Pre-compute all odom measurements.
    std::vector<Edge> odom;
    odom.reserve(N_FRAMES - 1);
    for (int k = 0; k < N_FRAMES - 1; k++) {
        float ti = gt[3 * k + 2];
        float dxw = gt[3 * (k + 1) + 0] - gt[3 * k + 0];
        float dyw = gt[3 * (k + 1) + 1] - gt[3 * k + 1];
        float zx = dxw * std::cos(ti) + dyw * std::sin(ti) + n_xy(rng);
        float zy = -dxw * std::sin(ti) + dyw * std::cos(ti) + n_xy(rng);
        float zt = wrap_angle(gt[3 * (k + 1) + 2] - gt[3 * k + 2]) + n_th(rng);
        odom.push_back({k, k + 1, zx, zy, zt});
    }

    // Two pose arrays: odom-only (dead reckoning) and slam (corrected).
    std::vector<float> odom_only(N_FRAMES * 3, 0.0f);
    std::vector<float> slam_h(N_FRAMES * 3, 0.0f);
    odom_only[0] = gt[0]; odom_only[1] = gt[1]; odom_only[2] = gt[2];
    slam_h[0] = gt[0]; slam_h[1] = gt[1]; slam_h[2] = gt[2];

    // GPU buffers (sized for full pose set; only window is active each step).
    int n_pose_floats = N_FRAMES * 3;
    int n_diag_floats = N_FRAMES * 9;
    int *d_ei = nullptr, *d_ej = nullptr;
    float *d_ez = nullptr, *d_poses = nullptr, *d_b = nullptr, *d_diag = nullptr;
    float *d_dx = nullptr, *d_r = nullptr, *d_z = nullptr, *d_p = nullptr, *d_Ap = nullptr;
    float *d_scratch = nullptr;
    int max_edges = (N_FRAMES - 1) + 4 * N_FRAMES;  // odom + generous loop budget
    CUDA_CHECK(cudaMalloc(&d_ei, max_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ej, max_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ez, max_edges * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_poses, n_pose_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n_pose_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_diag, n_diag_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx, n_pose_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r, n_pose_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, n_pose_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p, n_pose_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n_pose_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_poses, slam_h.data(),
                          n_pose_floats * sizeof(float), cudaMemcpyHostToDevice));

    float omega_xy = 1.0f / (ODOM_SIGMA_XY * ODOM_SIGMA_XY);
    float omega_th = 1.0f / (ODOM_SIGMA_TH * ODOM_SIGMA_TH);

    std::vector<Edge> edges_h;
    edges_h.reserve(max_edges);
    std::vector<Edge> loops_h;

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_online_slam.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W * 2 + 4, PANEL_H + 30));

    int blk = 256;
    auto blocks_for = [&](int n) { return (n + blk - 1) / blk; };

    double total_step_ms = 0.0;
    int counted_steps = 0;
    int n_loops_added = 0;
    int last_global_pass_t = -10000;
    int n_global_passes = 0;

    // Step 0 is the anchor — push to GPU as-is.
    for (int t = 1; t < N_FRAMES; t++) {
        // 1. Dead-reckoning prediction for the new pose.
        const Edge& o = odom[t - 1];
        float ti = slam_h[3 * (t - 1) + 2];
        float c = std::cos(ti), s = std::sin(ti);
        slam_h[3 * t + 0] = slam_h[3 * (t - 1) + 0] + c * o.zx - s * o.zy;
        slam_h[3 * t + 1] = slam_h[3 * (t - 1) + 1] + s * o.zx + c * o.zy;
        slam_h[3 * t + 2] = wrap_angle(slam_h[3 * (t - 1) + 2] + o.zt);
        // Mirror for odom-only path.
        float toi = odom_only[3 * (t - 1) + 2];
        float coc = std::cos(toi), sos = std::sin(toi);
        odom_only[3 * t + 0] = odom_only[3 * (t - 1) + 0] + coc * o.zx - sos * o.zy;
        odom_only[3 * t + 1] = odom_only[3 * (t - 1) + 1] + sos * o.zx + coc * o.zy;
        odom_only[3 * t + 2] = wrap_angle(odom_only[3 * (t - 1) + 2] + o.zt);
        // Sync to GPU.
        CUDA_CHECK(cudaMemcpy(d_poses + 3 * t, slam_h.data() + 3 * t,
                              3 * sizeof(float), cudaMemcpyHostToDevice));
        // Append odom edge.
        edges_h.push_back(o);

        // 2. Loop-closure detection (in GT space, simulating a good detector).
        int added_this_step = 0;
        // search backward; closest first
        struct Cand { int j; float d2; };
        std::vector<Cand> cands;
        for (int j = 0; j + LC_MIN_GAP <= t; j++) {
            float dxw = gt[3 * t + 0] - gt[3 * j + 0];
            float dyw = gt[3 * t + 1] - gt[3 * j + 1];
            float d2 = dxw * dxw + dyw * dyw;
            if (d2 < LC_DIST * LC_DIST) cands.push_back({j, d2});
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Cand& a, const Cand& b) { return a.d2 < b.d2; });
        for (const auto& cand : cands) {
            if (added_this_step >= LC_PER_FRAME_MAX) break;
            int j = cand.j;
            float dxw = gt[3 * t + 0] - gt[3 * j + 0];
            float dyw = gt[3 * t + 1] - gt[3 * j + 1];
            float tj = gt[3 * j + 2];
            float zx = dxw * std::cos(tj) + dyw * std::sin(tj) + nlc_xy(rng);
            float zy = -dxw * std::sin(tj) + dyw * std::cos(tj) + nlc_xy(rng);
            float zt = wrap_angle(gt[3 * t + 2] - gt[3 * j + 2]) + nlc_th(rng);
            Edge e{j, t, zx, zy, zt};
            edges_h.push_back(e);
            loops_h.push_back(e);
            added_this_step++;
            n_loops_added++;
        }

        // 3. Push updated edge buffer to GPU (full sync; small enough).
        int n_edges = static_cast<int>(edges_h.size());
        std::vector<int> ei(n_edges), ej(n_edges);
        std::vector<float> ez(n_edges * 3);
        for (int e = 0; e < n_edges; e++) {
            ei[e] = edges_h[e].i;
            ej[e] = edges_h[e].j;
            ez[3 * e + 0] = edges_h[e].zx;
            ez[3 * e + 1] = edges_h[e].zy;
            ez[3 * e + 2] = edges_h[e].zt;
        }
        CUDA_CHECK(cudaMemcpy(d_ei, ei.data(), n_edges * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ej, ej.data(), n_edges * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ez, ez.data(), n_edges * 3 * sizeof(float), cudaMemcpyHostToDevice));

        // 4. Sliding-window GN+PCG.
        int active_lo = std::max(0, t - WINDOW + 1);
        int active_hi = t + 1;  // exclusive
        int bvec = blocks_for(n_pose_floats);
        int bdiag = blocks_for(n_diag_floats);
        int be = blocks_for(n_edges);
        int bn = blocks_for(N_FRAMES);

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);

        for (int gn = 0; gn < GN_ITERS; gn++) {
            zero_kernel<<<bvec, blk>>>(n_pose_floats, d_b);
            zero_kernel<<<bdiag, blk>>>(n_diag_floats, d_diag);
            assemble_kernel<<<be, blk>>>(n_edges, d_ei, d_ej, d_ez, d_poses,
                                          omega_xy, omega_th,
                                          active_lo, active_hi, d_b, d_diag);
            // Anchor the boundary pose so the window doesn't drift.
            anchor_boundary_kernel<<<1, 1>>>(d_b, d_diag, active_lo);

            // PCG: solve H * dx = -b
            zero_kernel<<<bvec, blk>>>(n_pose_floats, d_dx);
            zero_kernel<<<bvec, blk>>>(n_pose_floats, d_r);
            axpy_kernel<<<bvec, blk>>>(n_pose_floats, -1.0f, d_b, d_r);
            apply_precond_kernel<<<bn, blk>>>(N_FRAMES, active_lo, active_hi,
                                              d_diag, d_r, d_z);
            copy_kernel<<<bvec, blk>>>(n_pose_floats, d_z, d_p);

            float rz_old = 0.0f;
            CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
            dot_kernel<<<32, 256>>>(n_pose_floats, d_r, d_z, d_scratch);
            CUDA_CHECK(cudaMemcpy(&rz_old, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
            if (rz_old <= 0.0f) break;

            for (int k = 0; k < PCG_ITERS; k++) {
                zero_kernel<<<bvec, blk>>>(n_pose_floats, d_Ap);
                matvec_kernel<<<be, blk>>>(n_edges, d_ei, d_ej, d_poses,
                                            omega_xy, omega_th,
                                            active_lo, active_hi, d_p, d_Ap);
                float pAp = 0.0f;
                CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
                dot_kernel<<<32, 256>>>(n_pose_floats, d_p, d_Ap, d_scratch);
                CUDA_CHECK(cudaMemcpy(&pAp, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
                if (pAp <= 0.0f) break;
                float alpha = rz_old / pAp;
                axpy_kernel<<<bvec, blk>>>(n_pose_floats, alpha, d_p, d_dx);
                axpy_kernel<<<bvec, blk>>>(n_pose_floats, -alpha, d_Ap, d_r);

                float rr = 0.0f;
                CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
                dot_kernel<<<32, 256>>>(n_pose_floats, d_r, d_r, d_scratch);
                CUDA_CHECK(cudaMemcpy(&rr, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
                if (rr < PCG_TOL) break;
                apply_precond_kernel<<<bn, blk>>>(N_FRAMES, active_lo, active_hi,
                                                  d_diag, d_r, d_z);
                float rz_new = 0.0f;
                CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
                dot_kernel<<<32, 256>>>(n_pose_floats, d_r, d_z, d_scratch);
                CUDA_CHECK(cudaMemcpy(&rz_new, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
                if (rz_old < 1.0e-12f) break;
                float beta = rz_new / rz_old;
                xpay_kernel<<<bvec, blk>>>(n_pose_floats, beta, d_z, d_p);
                rz_old = rz_new;
            }
            update_poses_kernel<<<bn, blk>>>(N_FRAMES, active_lo, active_hi,
                                              d_poses, d_dx);
        }

        // Global pass: if a new loop has been added recently and we haven't
        // done a global pass for at least LOOP_BURST_FRAMES frames, do one.
        bool do_global = (added_this_step > 0) &&
                         (t - last_global_pass_t >= LOOP_BURST_FRAMES);
        if (do_global) {
            int g_lo = 1;            // pose 0 stays the global anchor
            int g_hi = t + 1;
            int bng = blocks_for(N_FRAMES);
            for (int gn = 0; gn < GLOBAL_GN_ITERS; gn++) {
                zero_kernel<<<bvec, blk>>>(n_pose_floats, d_b);
                zero_kernel<<<bdiag, blk>>>(n_diag_floats, d_diag);
                assemble_kernel<<<be, blk>>>(n_edges, d_ei, d_ej, d_ez, d_poses,
                                              omega_xy, omega_th,
                                              g_lo, g_hi, d_b, d_diag);
                anchor_boundary_kernel<<<1, 1>>>(d_b, d_diag, g_lo);

                zero_kernel<<<bvec, blk>>>(n_pose_floats, d_dx);
                zero_kernel<<<bvec, blk>>>(n_pose_floats, d_r);
                axpy_kernel<<<bvec, blk>>>(n_pose_floats, -1.0f, d_b, d_r);
                apply_precond_kernel<<<bng, blk>>>(N_FRAMES, g_lo, g_hi,
                                                  d_diag, d_r, d_z);
                copy_kernel<<<bvec, blk>>>(n_pose_floats, d_z, d_p);

                float rz_old = 0.0f;
                CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
                dot_kernel<<<32, 256>>>(n_pose_floats, d_r, d_z, d_scratch);
                CUDA_CHECK(cudaMemcpy(&rz_old, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
                if (rz_old <= 0.0f) break;

                for (int k = 0; k < GLOBAL_PCG_ITERS; k++) {
                    zero_kernel<<<bvec, blk>>>(n_pose_floats, d_Ap);
                    matvec_kernel<<<be, blk>>>(n_edges, d_ei, d_ej, d_poses,
                                                omega_xy, omega_th,
                                                g_lo, g_hi, d_p, d_Ap);
                    float pAp = 0.0f;
                    CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
                    dot_kernel<<<32, 256>>>(n_pose_floats, d_p, d_Ap, d_scratch);
                    CUDA_CHECK(cudaMemcpy(&pAp, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
                    if (pAp <= 0.0f) break;
                    float alpha = rz_old / pAp;
                    axpy_kernel<<<bvec, blk>>>(n_pose_floats, alpha, d_p, d_dx);
                    axpy_kernel<<<bvec, blk>>>(n_pose_floats, -alpha, d_Ap, d_r);
                    float rr = 0.0f;
                    CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
                    dot_kernel<<<32, 256>>>(n_pose_floats, d_r, d_r, d_scratch);
                    CUDA_CHECK(cudaMemcpy(&rr, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
                    if (rr < PCG_TOL) break;
                    apply_precond_kernel<<<bng, blk>>>(N_FRAMES, g_lo, g_hi,
                                                      d_diag, d_r, d_z);
                    float rz_new = 0.0f;
                    CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(float)));
                    dot_kernel<<<32, 256>>>(n_pose_floats, d_r, d_z, d_scratch);
                    CUDA_CHECK(cudaMemcpy(&rz_new, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
                    if (rz_old < 1.0e-12f) break;
                    float beta = rz_new / rz_old;
                    xpay_kernel<<<bvec, blk>>>(n_pose_floats, beta, d_z, d_p);
                    rz_old = rz_new;
                }
                update_poses_kernel<<<bng, blk>>>(N_FRAMES, g_lo, g_hi,
                                                  d_poses, d_dx);
            }
            last_global_pass_t = t;
            n_global_passes++;
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        if (t >= 5) { total_step_ms += ms; counted_steps++; }

        // 5. Read back current trajectory for visualization.
        // If a global pass ran, all poses changed; otherwise window only.
        if (do_global) {
            CUDA_CHECK(cudaMemcpy(slam_h.data(), d_poses,
                                  (t + 1) * 3 * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        } else {
            CUDA_CHECK(cudaMemcpy(slam_h.data() + 3 * active_lo,
                                  d_poses + 3 * active_lo,
                                  (active_hi - active_lo) * 3 * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }

        // RMSE up to current frame.
        double sum2 = 0.0;
        for (int k = 0; k <= t; k++) {
            double dx = slam_h[3 * k + 0] - gt[3 * k + 0];
            double dy = slam_h[3 * k + 1] - gt[3 * k + 1];
            sum2 += dx * dx + dy * dy;
        }
        float rmse_slam = static_cast<float>(std::sqrt(sum2 / (t + 1)));
        double sum2o = 0.0;
        for (int k = 0; k <= t; k++) {
            double dx = odom_only[3 * k + 0] - gt[3 * k + 0];
            double dy = odom_only[3 * k + 1] - gt[3 * k + 1];
            sum2o += dx * dx + dy * dy;
        }
        float rmse_odom = static_cast<float>(std::sqrt(sum2o / (t + 1)));

        if (t % VIDEO_STRIDE == 0 || t == N_FRAMES - 1) {
            cv::Mat left = draw_panel(odom_only, t, gt, nullptr,
                                       "dead-reckoning (odom only)", rmse_odom);
            cv::Mat right = draw_panel(slam_h, t, gt, &loops_h,
                                        "GPU online SLAM (sliding window)", rmse_slam);
            cv::Mat frame(PANEL_H + 30, PANEL_W * 2 + 4, CV_8UC3, cv::Scalar(30, 30, 30));
            left.copyTo(frame(cv::Rect(0, 30, PANEL_W, PANEL_H)));
            right.copyTo(frame(cv::Rect(PANEL_W + 4, 30, PANEL_W, PANEL_H)));
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "GPU online SLAM (sliding window W=%d, %d odom + %d loop edges, %.2f ms/step)",
                          WINDOW, t, n_loops_added, ms);
            cv::putText(frame, buf, cv::Point(10, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            video.write(frame);
        }
        if (t % 50 == 0) {
            std::printf("  t=%4d  RMSE odom=%.3f  slam=%.3f  loops=%d  step=%.2f ms\n",
                        t, rmse_odom, rmse_slam, n_loops_added, ms);
        }
    }

    video.release();

    // Final RMSE
    double sum2 = 0.0, sum2o = 0.0;
    for (int k = 0; k < N_FRAMES; k++) {
        double dx = slam_h[3 * k + 0] - gt[3 * k + 0];
        double dy = slam_h[3 * k + 1] - gt[3 * k + 1];
        sum2 += dx * dx + dy * dy;
        double dxo = odom_only[3 * k + 0] - gt[3 * k + 0];
        double dyo = odom_only[3 * k + 1] - gt[3 * k + 1];
        sum2o += dxo * dxo + dyo * dyo;
    }
    float rmse_slam_final = static_cast<float>(std::sqrt(sum2 / N_FRAMES));
    float rmse_odom_final = static_cast<float>(std::sqrt(sum2o / N_FRAMES));
    std::printf("Final RMSE: odom-only=%.3f m,  SLAM=%.3f m\n",
                rmse_odom_final, rmse_slam_final);
    if (counted_steps > 0) {
        std::printf("Avg step time: %.2f ms (window=%d, GN=%d, PCG=%d, %d total loops, %d global passes)\n",
                    total_step_ms / counted_steps, WINDOW, GN_ITERS, PCG_ITERS,
                    n_loops_added, n_global_passes);
    }
    convert_avi_to_gif("gif/gpu_online_slam.avi", "gif/gpu_online_slam.gif", VIDEO_FPS);
    std::printf("GIF saved to gif/gpu_online_slam.gif\n");

    cudaFree(d_ei); cudaFree(d_ej); cudaFree(d_ez); cudaFree(d_poses);
    cudaFree(d_b); cudaFree(d_diag); cudaFree(d_dx);
    cudaFree(d_r); cudaFree(d_z); cudaFree(d_p); cudaFree(d_Ap);
    cudaFree(d_scratch);
    return 0;
}
