/*************************************************************************
    GPU Bundle Adjustment (2D pose-landmark BA via Gauss-Newton with
    Schur-complement landmark elimination + Jacobi-PCG on poses)

    Scenario: planar robot drives a circular trajectory, observing
    landmarks within sensor range. Each observation produces a 2D
    residual (x,y) of the landmark in the robot frame.

    Variables:
      poses_i      = (tx_i, ty_i, theta_i)              i = 0..N_POSES-1
      landmarks_j  = (lx_j, ly_j)                       j = 0..N_LANDMARKS-1
      pose 0 is fixed (anchor) to remove the gauge freedom.

    Observation model:
      z_ij = R(theta_i)^T ( landmark_j - t_i ) + noise

    Loss:
      L = 0.5 * sum over (i,j) of || z_ij_observed - h(pose_i, landmark_j) ||^2

    Optimisation: Levenberg-Marquardt-style Gauss-Newton with diagonal
    damping. Hessian is sparsely populated via per-observation kernels
    that atomicAdd into block-structured H_pp (3x3 per pose), H_ll
    (2x2 per landmark), and H_pl (3x2 per observation, but accumulated
    via the cross gradients). Landmarks are eliminated by Schur
    complement:
        S = H_pp - sum_j H_pl,j * H_ll,j^-1 * H_lp,j
        b_p_reduced = b_p - sum_j H_pl,j * H_ll,j^-1 * b_l,j
    The reduced system S * dp = b_p_reduced is solved with Jacobi-PCG.
    Landmarks are back-substituted:
        dl_j = H_ll,j^-1 * (b_l,j - H_lp,j * dp_i(j))

    All H_pp/H_ll/H_pl computations and PCG steps run as CUDA kernels.

    Output: gif/gpu_bundle_adjustment.gif (initial / after LM / GT panels)
    Headline: per-observation Gauss-Newton iter time + final RMSE.
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

constexpr int   N_POSES     = 1000;
constexpr int   N_LANDMARKS = 8000;
constexpr int   N_OBS       = 60000;   // budget; actual count may be lower
constexpr float WORLD       = 80.0f;
constexpr float SENSOR_RANGE = 6.0f;
constexpr float NOISE_OBS    = 0.05f;   // observation noise [m]
constexpr float NOISE_POSE_T = 0.5f;    // initial pose translation noise [m]
constexpr float NOISE_POSE_R = 0.04f;   // initial pose rotation noise [rad]
constexpr float NOISE_LM     = 0.5f;    // initial landmark noise [m]
constexpr int   LM_ITERS     = 40;
constexpr int   PCG_ITERS    = 80;
constexpr float DAMPING_INIT = 1.0e-3f;
constexpr int   PANEL_W      = 540;
constexpr int   PANEL_H      = 540;

// -------------------------------------------------------------------------
// Observation struct
// -------------------------------------------------------------------------
struct Obs {
    int   pose;
    int   lm;
    float zx, zy;  // observed landmark in robot frame
};

// -------------------------------------------------------------------------
// CUDA: assemble H, b, and residuals
// -------------------------------------------------------------------------
__global__ void assemble_kernel(
    const Obs* __restrict__ obs, int n_obs,
    const float* __restrict__ poses,      // 3 * N_POSES
    const float* __restrict__ landmarks,  // 2 * N_LANDMARKS
    float* __restrict__ Hpp,              // 9 * N_POSES (3x3 per pose)
    float* __restrict__ Hll,              // 4 * N_LANDMARKS (2x2 per lm)
    float* __restrict__ Hpl,              // 6 * n_obs (3x2 per obs)
    float* __restrict__ bp,               // 3 * N_POSES
    float* __restrict__ bl,               // 2 * N_LANDMARKS
    float* __restrict__ cost_acc)
{
    int oi = blockIdx.x * blockDim.x + threadIdx.x;
    if (oi >= n_obs) return;
    int pi = obs[oi].pose;
    int li = obs[oi].lm;
    float tx = poses[pi * 3 + 0];
    float ty = poses[pi * 3 + 1];
    float th = poses[pi * 3 + 2];
    float lx = landmarks[li * 2 + 0];
    float ly = landmarks[li * 2 + 1];
    float ct = cosf(th), st = sinf(th);
    float dx = lx - tx;
    float dy = ly - ty;
    // predicted obs (landmark in robot frame)
    float hx =  ct * dx + st * dy;
    float hy = -st * dx + ct * dy;
    float rx = obs[oi].zx - hx;
    float ry = obs[oi].zy - hy;
    atomicAdd(cost_acc, 0.5f * (rx * rx + ry * ry));
    // d hx / d (tx, ty, th, lx, ly)  = (-ct, -st,  -st*dx + ct*dy,  ct,  st)
    // d hy / d (tx, ty, th, lx, ly)  = ( st, -ct,  -ct*dx - st*dy, -st,  ct)
    float Jp[6]; // J(2x3) for pose part [Jp[0..2]=row 0 (hx), Jp[3..5]=row 1 (hy)]
    Jp[0] = -ct;
    Jp[1] = -st;
    Jp[2] = -st * dx + ct * dy;
    Jp[3] =  st;
    Jp[4] = -ct;
    Jp[5] = -ct * dx - st * dy;
    float Jl[4]; // J(2x2) for landmark
    Jl[0] =  ct;  Jl[1] =  st;
    Jl[2] = -st;  Jl[3] =  ct;

    if (pi > 0) {
        // H_pp += Jp^T Jp; b_p += Jp^T r
        #pragma unroll
        for (int a = 0; a < 3; a++) {
            float ja_hx = Jp[a];
            float ja_hy = Jp[3 + a];
            atomicAdd(&bp[pi * 3 + a], ja_hx * rx + ja_hy * ry);
            #pragma unroll
            for (int b = 0; b < 3; b++) {
                float jb_hx = Jp[b];
                float jb_hy = Jp[3 + b];
                atomicAdd(&Hpp[pi * 9 + a * 3 + b], ja_hx * jb_hx + ja_hy * jb_hy);
            }
        }
    }
    // H_ll += Jl^T Jl ; b_l += Jl^T r
    #pragma unroll
    for (int a = 0; a < 2; a++) {
        float ja_hx = Jl[a];
        float ja_hy = Jl[2 + a];
        atomicAdd(&bl[li * 2 + a], ja_hx * rx + ja_hy * ry);
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            float jb_hx = Jl[b];
            float jb_hy = Jl[2 + b];
            atomicAdd(&Hll[li * 4 + a * 2 + b], ja_hx * jb_hx + ja_hy * jb_hy);
        }
    }
    // H_pl = Jp^T Jl  (3x2) — store per-obs (sparse cross block)
    if (pi > 0) {
        #pragma unroll
        for (int a = 0; a < 3; a++) {
            #pragma unroll
            for (int b = 0; b < 2; b++) {
                float v = Jp[a] * Jl[b] + Jp[3 + a] * Jl[2 + b];
                Hpl[oi * 6 + a * 2 + b] = v;
            }
        }
    } else {
        #pragma unroll
        for (int a = 0; a < 6; a++) Hpl[oi * 6 + a] = 0.0f;
    }
}

__global__ void damp_diagonal_kernel(float* Hpp, float* Hll, float lambda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_POSES) {
        Hpp[i * 9 + 0] *= (1.0f + lambda);
        Hpp[i * 9 + 4] *= (1.0f + lambda);
        Hpp[i * 9 + 8] *= (1.0f + lambda);
    }
    if (i < N_LANDMARKS) {
        Hll[i * 4 + 0] *= (1.0f + lambda);
        Hll[i * 4 + 3] *= (1.0f + lambda);
    }
}

// Compute landmark inverse 2x2 blocks
__global__ void invert_Hll_kernel(const float* Hll, float* Hll_inv) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N_LANDMARKS) return;
    float a = Hll[j * 4 + 0];
    float b = Hll[j * 4 + 1];
    float c = Hll[j * 4 + 2];
    float d = Hll[j * 4 + 3];
    float det = a * d - b * c;
    if (fabsf(det) < 1e-12f) {
        Hll_inv[j * 4 + 0] = 0.0f; Hll_inv[j * 4 + 1] = 0.0f;
        Hll_inv[j * 4 + 2] = 0.0f; Hll_inv[j * 4 + 3] = 0.0f;
        return;
    }
    float inv = 1.0f / det;
    Hll_inv[j * 4 + 0] =  d * inv;
    Hll_inv[j * 4 + 1] = -b * inv;
    Hll_inv[j * 4 + 2] = -c * inv;
    Hll_inv[j * 4 + 3] =  a * inv;
}

// Schur complement: for each observation, contribute -Hpl * Hll_inv * Hlp to S and -Hpl * Hll_inv * bl to bp_reduced
__global__ void schur_kernel(
    const Obs* obs, int n_obs,
    const float* Hpl,
    const float* Hll_inv,
    const float* bl,
    float* S,            // reduced 3x3 per pose; same shape as Hpp
    float* bp_red)       // reduced b_p; same shape as bp
{
    int oi = blockIdx.x * blockDim.x + threadIdx.x;
    if (oi >= n_obs) return;
    int pi = obs[oi].pose;
    int li = obs[oi].lm;
    if (pi == 0) return;
    // tmp = Hll_inv * Hpl^T  (2x3)
    float Hi[4] = { Hll_inv[li * 4 + 0], Hll_inv[li * 4 + 1],
                    Hll_inv[li * 4 + 2], Hll_inv[li * 4 + 3] };
    float HplT[6]; // 2x3
    HplT[0] = Hpl[oi * 6 + 0]; HplT[1] = Hpl[oi * 6 + 2]; HplT[2] = Hpl[oi * 6 + 4];
    HplT[3] = Hpl[oi * 6 + 1]; HplT[4] = Hpl[oi * 6 + 3]; HplT[5] = Hpl[oi * 6 + 5];
    float tmp[6]; // 2x3 = Hi * HplT
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            tmp[r * 3 + c] = Hi[r * 2 + 0] * HplT[0 * 3 + c]
                           + Hi[r * 2 + 1] * HplT[1 * 3 + c];
        }
    }
    // contribution to S (3x3): -Hpl * tmp
    #pragma unroll
    for (int r = 0; r < 3; r++) {
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            float v = -(Hpl[oi * 6 + r * 2 + 0] * tmp[0 * 3 + c]
                       + Hpl[oi * 6 + r * 2 + 1] * tmp[1 * 3 + c]);
            atomicAdd(&S[pi * 9 + r * 3 + c], v);
        }
    }
    // contribution to bp_red: -Hpl * Hi * bl
    float Hi_bl[2];
    Hi_bl[0] = Hi[0] * bl[li * 2 + 0] + Hi[1] * bl[li * 2 + 1];
    Hi_bl[1] = Hi[2] * bl[li * 2 + 0] + Hi[3] * bl[li * 2 + 1];
    #pragma unroll
    for (int r = 0; r < 3; r++) {
        float v = -(Hpl[oi * 6 + r * 2 + 0] * Hi_bl[0]
                   + Hpl[oi * 6 + r * 2 + 1] * Hi_bl[1]);
        atomicAdd(&bp_red[pi * 3 + r], v);
    }
}

__global__ void add_kernel(const float* a, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += a[i];
}

// PCG kernels: A * x via S * dp = b_red
__global__ void mat_vec_kernel(const float* S, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_POSES * 3) return;
    int pi = i / 3;
    int r = i % 3;
    float v = 0.0f;
    #pragma unroll
    for (int c = 0; c < 3; c++) v += S[pi * 9 + r * 3 + c] * x[pi * 3 + c];
    y[i] = v;
}

__global__ void axpy_kernel(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__ void aypx_kernel(float a, const float* x, float* y, int n) {
    // y = x + a * y
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + a * y[i];
}

__global__ void jacobi_apply_kernel(const float* S, const float* r, float* z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_POSES * 3) return;
    int pi = i / 3;
    int row = i % 3;
    float d = S[pi * 9 + row * 3 + row];
    z[i] = (d > 1.0e-12f) ? r[i] / d : 0.0f;
}

__global__ void dot_kernel(const float* a, const float* b, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.0f;
    if (i < n) v = a[i] * b[i];
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

__global__ void back_subst_kernel(
    const Obs* obs, int n_obs,
    const float* Hpl, const float* Hll_inv, const float* bl,
    const float* dp, float* dl_acc)
{
    int oi = blockIdx.x * blockDim.x + threadIdx.x;
    if (oi >= n_obs) return;
    int pi = obs[oi].pose;
    int li = obs[oi].lm;
    if (pi == 0) {
        // landmark sees a fixed pose: contribution to bl is already included; we
        // still need to compute landmark update via Hll_inv * bl, but only once
        // per landmark — done by a separate kernel below.
        return;
    }
    float tmp[2];
    tmp[0] = Hpl[oi * 6 + 0 * 2 + 0] * dp[pi * 3 + 0]
           + Hpl[oi * 6 + 1 * 2 + 0] * dp[pi * 3 + 1]
           + Hpl[oi * 6 + 2 * 2 + 0] * dp[pi * 3 + 2];
    tmp[1] = Hpl[oi * 6 + 0 * 2 + 1] * dp[pi * 3 + 0]
           + Hpl[oi * 6 + 1 * 2 + 1] * dp[pi * 3 + 1]
           + Hpl[oi * 6 + 2 * 2 + 1] * dp[pi * 3 + 2];
    // dl_j -= Hll_inv * Hlp * dp
    float Hi[4] = { Hll_inv[li * 4 + 0], Hll_inv[li * 4 + 1],
                    Hll_inv[li * 4 + 2], Hll_inv[li * 4 + 3] };
    float v0 = Hi[0] * tmp[0] + Hi[1] * tmp[1];
    float v1 = Hi[2] * tmp[0] + Hi[3] * tmp[1];
    atomicAdd(&dl_acc[li * 2 + 0], -v0);
    atomicAdd(&dl_acc[li * 2 + 1], -v1);
}

__global__ void landmark_baseline_kernel(const float* Hll_inv, const float* bl,
                                         float* dl_acc) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N_LANDMARKS) return;
    float Hi[4] = { Hll_inv[j * 4 + 0], Hll_inv[j * 4 + 1],
                    Hll_inv[j * 4 + 2], Hll_inv[j * 4 + 3] };
    dl_acc[j * 2 + 0] += Hi[0] * bl[j * 2 + 0] + Hi[1] * bl[j * 2 + 1];
    dl_acc[j * 2 + 1] += Hi[2] * bl[j * 2 + 0] + Hi[3] * bl[j * 2 + 1];
}

__global__ void apply_pose_step_kernel(float* poses, const float* dp, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_POSES) return;
    if (i == 0) return;  // anchor
    poses[i * 3 + 0] += scale * dp[i * 3 + 0];
    poses[i * 3 + 1] += scale * dp[i * 3 + 1];
    poses[i * 3 + 2] += scale * dp[i * 3 + 2];
}

__global__ void apply_lm_step_kernel(float* lms, const float* dl, float scale) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N_LANDMARKS) return;
    lms[j * 2 + 0] += scale * dl[j * 2 + 0];
    lms[j * 2 + 1] += scale * dl[j * 2 + 1];
}

__global__ void zero_kernel(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0.0f;
}

// -------------------------------------------------------------------------
// Host helpers
// -------------------------------------------------------------------------
static double scalar_from_gpu(const float* d_v) {
    float h;
    cudaMemcpy(&h, d_v, sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

static void zero(float* d, int n) {
    int t = 256, b = (n + t - 1) / t;
    zero_kernel<<<b, t>>>(d, n);
}

// -------------------------------------------------------------------------
// Build ground truth + initial guess
// -------------------------------------------------------------------------
struct Dataset {
    std::vector<float> poses_gt, landmarks_gt;
    std::vector<float> poses_init, landmarks_init;
    std::vector<Obs>   obs;
};

static Dataset build_dataset(unsigned long seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> ng_t(0.0f, NOISE_POSE_T);
    std::normal_distribution<float> ng_r(0.0f, NOISE_POSE_R);
    std::normal_distribution<float> ng_l(0.0f, NOISE_LM);
    std::normal_distribution<float> ng_o(0.0f, NOISE_OBS);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);

    Dataset d;
    d.poses_gt.resize(N_POSES * 3);
    d.poses_init.resize(N_POSES * 3);
    // 3-loop spiral trajectory so 1000 poses cover the scene without overlap
    for (int i = 0; i < N_POSES; i++) {
        float s = (float)i / (N_POSES - 1);
        float t = s * 3.0f * 2.0f * (float)M_PI;
        float r = 8.0f + 20.0f * s;
        float cx = WORLD * 0.5f + r * std::cos(t);
        float cy = WORLD * 0.5f + r * std::sin(t);
        float th = t + (float)M_PI / 2.0f;
        d.poses_gt[i * 3 + 0] = cx;
        d.poses_gt[i * 3 + 1] = cy;
        d.poses_gt[i * 3 + 2] = th;
        if (i == 0) {
            d.poses_init[i * 3 + 0] = cx;
            d.poses_init[i * 3 + 1] = cy;
            d.poses_init[i * 3 + 2] = th;
        } else {
            d.poses_init[i * 3 + 0] = cx + ng_t(rng);
            d.poses_init[i * 3 + 1] = cy + ng_t(rng);
            d.poses_init[i * 3 + 2] = th + ng_r(rng);
        }
    }

    d.landmarks_gt.resize(N_LANDMARKS * 2);
    d.landmarks_init.resize(N_LANDMARKS * 2);
    for (int j = 0; j < N_LANDMARKS; j++) {
        float lx = u(rng) * WORLD;
        float ly = u(rng) * WORLD;
        d.landmarks_gt[j * 2 + 0] = lx;
        d.landmarks_gt[j * 2 + 1] = ly;
        d.landmarks_init[j * 2 + 0] = lx + ng_l(rng);
        d.landmarks_init[j * 2 + 1] = ly + ng_l(rng);
    }

    d.obs.reserve(N_OBS);
    int obs_per_pose = N_OBS / N_POSES;
    // shuffle landmark visit order per pose to avoid biased "first N" picks
    std::vector<int> idx(N_LANDMARKS);
    for (int j = 0; j < N_LANDMARKS; j++) idx[j] = j;
    for (int i = 0; i < N_POSES; i++) {
        // partial shuffle is enough since SENSOR_RANGE prunes most
        std::shuffle(idx.begin(), idx.end(), rng);
        int added = 0;
        // pick landmarks within sensor range
        for (int k = 0; k < N_LANDMARKS && added < obs_per_pose; k++) {
            int j = idx[k];
            float dx = d.landmarks_gt[j * 2 + 0] - d.poses_gt[i * 3 + 0];
            float dy = d.landmarks_gt[j * 2 + 1] - d.poses_gt[i * 3 + 1];
            float r2 = dx * dx + dy * dy;
            if (r2 > SENSOR_RANGE * SENSOR_RANGE) continue;
            float th = d.poses_gt[i * 3 + 2];
            float ct = std::cos(th), st = std::sin(th);
            float zx =  ct * dx + st * dy + ng_o(rng);
            float zy = -st * dx + ct * dy + ng_o(rng);
            Obs o; o.pose = i; o.lm = j; o.zx = zx; o.zy = zy;
            d.obs.push_back(o);
            added++;
        }
    }
    return d;
}

// -------------------------------------------------------------------------
// Render
// -------------------------------------------------------------------------
static cv::Mat render(const std::vector<float>& poses,
                      const std::vector<float>& landmarks,
                      const std::vector<float>& poses_gt,
                      const std::vector<float>& lms_gt,
                      const char* title, float cost) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(15, 15, 15));
    auto X = [&](float x) { return static_cast<int>(x / WORLD * PANEL_W); };
    auto Y = [&](float y) { return static_cast<int>((1.0f - y / WORLD) * PANEL_H); };
    // landmark GT (faint, 1px)
    for (int j = 0; j < N_LANDMARKS; j++) {
        int px = X(lms_gt[j * 2 + 0]), py = Y(lms_gt[j * 2 + 1]);
        if ((unsigned)px < (unsigned)PANEL_W && (unsigned)py < (unsigned)PANEL_H) {
            img.at<cv::Vec3b>(py, px) = cv::Vec3b(70, 70, 70);
        }
    }
    // landmark current (small 1-2 px point)
    for (int j = 0; j < N_LANDMARKS; j++) {
        cv::circle(img, cv::Point(X(landmarks[j * 2 + 0]), Y(landmarks[j * 2 + 1])),
                   1, cv::Scalar(220, 200, 50), cv::FILLED);
    }
    // pose GT path (thin)
    for (int i = 1; i < N_POSES; i++) {
        cv::line(img,
                 cv::Point(X(poses_gt[(i - 1) * 3 + 0]), Y(poses_gt[(i - 1) * 3 + 1])),
                 cv::Point(X(poses_gt[i * 3 + 0]),       Y(poses_gt[i * 3 + 1])),
                 cv::Scalar(90, 90, 90), 1, cv::LINE_AA);
    }
    // pose current path
    for (int i = 1; i < N_POSES; i++) {
        cv::line(img,
                 cv::Point(X(poses[(i - 1) * 3 + 0]), Y(poses[(i - 1) * 3 + 1])),
                 cv::Point(X(poses[i * 3 + 0]),       Y(poses[i * 3 + 1])),
                 cv::Scalar(80, 220, 80), 1, cv::LINE_AA);
    }
    // pose marks (sparser at higher N)
    for (int i = 0; i < N_POSES; i += 25) {
        cv::circle(img, cv::Point(X(poses[i * 3 + 0]), Y(poses[i * 3 + 1])),
                   2, cv::Scalar(255, 255, 255), cv::FILLED);
    }
    cv::rectangle(img, cv::Rect(0, 0, PANEL_W, 30), cv::Scalar(0, 0, 0), cv::FILLED);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s   cost=%.2f", title, cost);
    cv::putText(img, buf, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return img;
}

static double compute_rmse(const std::vector<float>& a, const std::vector<float>& b,
                           int stride, int n_groups) {
    double s = 0.0;
    int cnt = 0;
    for (int i = 0; i < n_groups; i++) {
        for (int k = 0; k < std::min(stride, 2); k++) {
            double d = a[i * stride + k] - b[i * stride + k];
            s += d * d; cnt++;
        }
    }
    return std::sqrt(s / std::max(1, cnt));
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
    auto d = build_dataset(2026ULL);
    int n_obs = static_cast<int>(d.obs.size());
    std::printf("BA dataset: %d poses, %d landmarks, %d observations\n",
                N_POSES, N_LANDMARKS, n_obs);

    // Device buffers
    Obs*   d_obs = nullptr;
    float* d_poses = nullptr;
    float* d_lms = nullptr;
    float* d_Hpp = nullptr;
    float* d_Hll = nullptr;
    float* d_Hpl = nullptr;
    float* d_Hll_inv = nullptr;
    float* d_bp = nullptr;
    float* d_bl = nullptr;
    float* d_bp_red = nullptr;
    float* d_S = nullptr;
    float* d_cost = nullptr;
    float* d_dp = nullptr;
    float* d_dl = nullptr;
    // PCG scratch
    float* d_pcg_r = nullptr;
    float* d_pcg_z = nullptr;
    float* d_pcg_p = nullptr;
    float* d_pcg_Ap = nullptr;
    float* d_scalar = nullptr;
    CUDA_CHECK(cudaMalloc(&d_obs,    n_obs * sizeof(Obs)));
    CUDA_CHECK(cudaMalloc(&d_poses,  N_POSES * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lms,    N_LANDMARKS * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Hpp,    N_POSES * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Hll,    N_LANDMARKS * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Hpl,    n_obs * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Hll_inv,N_LANDMARKS * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bp,     N_POSES * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bl,     N_LANDMARKS * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bp_red, N_POSES * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S,      N_POSES * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cost,   sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dp,     N_POSES * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dl,     N_LANDMARKS * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pcg_r,  N_POSES * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pcg_z,  N_POSES * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pcg_p,  N_POSES * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pcg_Ap, N_POSES * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scalar, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_obs,   d.obs.data(),         n_obs * sizeof(Obs),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_poses, d.poses_init.data(),  N_POSES * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lms,   d.landmarks_init.data(), N_LANDMARKS * 2 * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::vector<float> h_poses_cur(N_POSES * 3), h_lms_cur(N_LANDMARKS * 2);
    auto cudasync = []() { CUDA_CHECK(cudaDeviceSynchronize()); };

    cv::VideoWriter video("gif/gpu_bundle_adjustment.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 4,
                          cv::Size(PANEL_W * 2 + 4, PANEL_H + 30));

    auto record_frame = [&](const char* title) {
        CUDA_CHECK(cudaMemcpy(h_poses_cur.data(), d_poses,
                              N_POSES * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_lms_cur.data(), d_lms,
                              N_LANDMARKS * 2 * sizeof(float), cudaMemcpyDeviceToHost));
        // compute cost
        zero(d_cost, 1);
        zero(d_Hpp, N_POSES * 9);
        zero(d_Hll, N_LANDMARKS * 4);
        zero(d_bp, N_POSES * 3);
        zero(d_bl, N_LANDMARKS * 2);
        cudasync();
        int t = 256, b = (n_obs + t - 1) / t;
        assemble_kernel<<<b, t>>>(d_obs, n_obs, d_poses, d_lms,
                                  d_Hpp, d_Hll, d_Hpl, d_bp, d_bl, d_cost);
        cudasync();
        float cost = static_cast<float>(scalar_from_gpu(d_cost));
        cv::Mat current = render(h_poses_cur, h_lms_cur, d.poses_gt, d.landmarks_gt,
                                 title, cost);
        cv::Mat gt = render(d.poses_gt, d.landmarks_gt, d.poses_gt, d.landmarks_gt,
                            "ground truth", 0.0f);
        cv::Mat frame(PANEL_H + 30, PANEL_W * 2 + 4, CV_8UC3, cv::Scalar(30, 30, 30));
        current.copyTo(frame(cv::Rect(0, 30, PANEL_W, PANEL_H)));
        gt.copyTo(frame(cv::Rect(PANEL_W + 4, 30, PANEL_W, PANEL_H)));
        cv::putText(frame,
                    "GPU bundle adjustment (Gauss-Newton + Schur-complement + Jacobi-PCG)",
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        video.write(frame);
    };

    record_frame("initial guess");

    float lambda = DAMPING_INIT;
    double total_iter_ms = 0.0;
    int counted_iters = 0;

    for (int iter = 0; iter < LM_ITERS; iter++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // assemble Hpp, Hll, Hpl, bp, bl, cost
        zero(d_cost, 1);
        zero(d_Hpp, N_POSES * 9);
        zero(d_Hll, N_LANDMARKS * 4);
        zero(d_bp,  N_POSES * 3);
        zero(d_bl,  N_LANDMARKS * 2);
        cudasync();
        int t = 256, b = (n_obs + t - 1) / t;
        assemble_kernel<<<b, t>>>(d_obs, n_obs, d_poses, d_lms,
                                  d_Hpp, d_Hll, d_Hpl, d_bp, d_bl, d_cost);
        cudasync();
        float cost_before = static_cast<float>(scalar_from_gpu(d_cost));

        // diagonal damping
        int n_max = std::max(N_POSES, N_LANDMARKS);
        damp_diagonal_kernel<<<(n_max + 255) / 256, 256>>>(d_Hpp, d_Hll, lambda);

        // invert Hll
        invert_Hll_kernel<<<(N_LANDMARKS + 255) / 256, 256>>>(d_Hll, d_Hll_inv);

        // S = Hpp; bp_red = bp
        CUDA_CHECK(cudaMemcpy(d_S,      d_Hpp, N_POSES * 9 * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_bp_red, d_bp,  N_POSES * 3 * sizeof(float),
                              cudaMemcpyDeviceToDevice));

        // schur complement
        schur_kernel<<<b, t>>>(d_obs, n_obs, d_Hpl, d_Hll_inv, d_bl, d_S, d_bp_red);

        // PCG to solve S * dp = b_p_red
        zero(d_dp, N_POSES * 3);
        // r = b - A*dp; since dp = 0, r = b_red
        CUDA_CHECK(cudaMemcpy(d_pcg_r, d_bp_red, N_POSES * 3 * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        jacobi_apply_kernel<<<(N_POSES * 3 + 255) / 256, 256>>>(d_S, d_pcg_r, d_pcg_z);
        CUDA_CHECK(cudaMemcpy(d_pcg_p, d_pcg_z, N_POSES * 3 * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        zero(d_scalar, 1);
        int npose = N_POSES * 3;
        int bp_blocks = (npose + 255) / 256;
        dot_kernel<<<bp_blocks, 256, 256 * sizeof(float)>>>(d_pcg_r, d_pcg_z,
                                                            d_scalar, npose);
        cudasync();
        float rz_old = static_cast<float>(scalar_from_gpu(d_scalar));

        for (int it = 0; it < PCG_ITERS; it++) {
            mat_vec_kernel<<<bp_blocks, 256>>>(d_S, d_pcg_p, d_pcg_Ap);
            zero(d_scalar, 1);
            dot_kernel<<<bp_blocks, 256, 256 * sizeof(float)>>>(d_pcg_p, d_pcg_Ap,
                                                                d_scalar, npose);
            cudasync();
            float pAp = static_cast<float>(scalar_from_gpu(d_scalar));
            if (std::fabs(pAp) < 1e-20f) break;
            float alpha = rz_old / pAp;
            axpy_kernel<<<bp_blocks, 256>>>( alpha, d_pcg_p, d_dp,    npose);
            axpy_kernel<<<bp_blocks, 256>>>(-alpha, d_pcg_Ap, d_pcg_r, npose);
            jacobi_apply_kernel<<<bp_blocks, 256>>>(d_S, d_pcg_r, d_pcg_z);
            zero(d_scalar, 1);
            dot_kernel<<<bp_blocks, 256, 256 * sizeof(float)>>>(d_pcg_r, d_pcg_z,
                                                                d_scalar, npose);
            cudasync();
            float rz_new = static_cast<float>(scalar_from_gpu(d_scalar));
            if (rz_new < 1e-12f) break;
            float beta = rz_new / rz_old;
            aypx_kernel<<<bp_blocks, 256>>>(beta, d_pcg_z, d_pcg_p, npose);
            rz_old = rz_new;
        }

        // back-substitute landmarks
        zero(d_dl, N_LANDMARKS * 2);
        back_subst_kernel<<<b, t>>>(d_obs, n_obs, d_Hpl, d_Hll_inv, d_bl, d_dp, d_dl);
        landmark_baseline_kernel<<<(N_LANDMARKS + 255) / 256, 256>>>(d_Hll_inv, d_bl, d_dl);

        // try step (scale 1.0); could roll-back on cost increase but we trust LM damping
        apply_pose_step_kernel<<<(N_POSES + 255) / 256, 256>>>(d_poses, d_dp, 1.0f);
        apply_lm_step_kernel<<<(N_LANDMARKS + 255) / 256, 256>>>(d_lms, d_dl, 1.0f);
        cudasync();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (iter >= 3) { total_iter_ms += ms; counted_iters++; }

        // measure new cost
        zero(d_cost, 1);
        zero(d_Hpp, N_POSES * 9); zero(d_Hll, N_LANDMARKS * 4);
        zero(d_bp,  N_POSES * 3); zero(d_bl,  N_LANDMARKS * 2);
        assemble_kernel<<<b, t>>>(d_obs, n_obs, d_poses, d_lms,
                                  d_Hpp, d_Hll, d_Hpl, d_bp, d_bl, d_cost);
        cudasync();
        float cost_after = static_cast<float>(scalar_from_gpu(d_cost));

        if (cost_after < cost_before) lambda = fmaxf(lambda * 0.7f, 1e-6f);
        else lambda = fminf(lambda * 2.0f, 1.0e2f);

        std::printf("LM iter %3d  cost %.4f -> %.4f  lambda %.2e  %.2f ms\n",
                    iter, cost_before, cost_after, lambda, ms);
        if (iter == 0 || iter == 4 || iter == 9 || iter == 19 || iter == LM_ITERS - 1) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "after iter %d", iter + 1);
            record_frame(buf);
        }
    }
    video.release();
    convert_avi_to_gif("gif/gpu_bundle_adjustment.avi",
                       "gif/gpu_bundle_adjustment.gif", 4);

    // Final RMSE
    CUDA_CHECK(cudaMemcpy(h_poses_cur.data(), d_poses,
                          N_POSES * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lms_cur.data(), d_lms,
                          N_LANDMARKS * 2 * sizeof(float), cudaMemcpyDeviceToHost));
    double rmse_init_p = compute_rmse(d.poses_init,   d.poses_gt,    3, N_POSES);
    double rmse_init_l = compute_rmse(d.landmarks_init, d.landmarks_gt, 2, N_LANDMARKS);
    double rmse_p = compute_rmse(h_poses_cur, d.poses_gt, 3, N_POSES);
    double rmse_l = compute_rmse(h_lms_cur, d.landmarks_gt, 2, N_LANDMARKS);
    std::printf("Pose translation RMSE  init %.3f -> final %.3f m\n", rmse_init_p, rmse_p);
    std::printf("Landmark RMSE          init %.3f -> final %.3f m\n", rmse_init_l, rmse_l);
    if (counted_iters > 0) {
        std::printf("Avg LM iter time: %.2f ms (%d observations, %d poses, %d landmarks)\n",
                    total_iter_ms / counted_iters, n_obs, N_POSES, N_LANDMARKS);
    }
    std::printf("GIF saved to gif/gpu_bundle_adjustment.gif\n");

    // cleanup
    for (auto* p : {d_poses, d_lms, d_Hpp, d_Hll, d_Hpl, d_Hll_inv, d_bp, d_bl,
                    d_bp_red, d_S, d_cost, d_dp, d_dl,
                    d_pcg_r, d_pcg_z, d_pcg_p, d_pcg_Ap, d_scalar}) {
        CUDA_CHECK(cudaFree(p));
    }
    CUDA_CHECK(cudaFree(d_obs));
    return 0;
}
