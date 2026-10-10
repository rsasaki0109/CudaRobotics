// benchmark_ndt_localization.cu
//
// Map-based 3D LiDAR localization with NDT (Normal Distributions Transform,
// Magnusson 2009, as in PCL and Autoware's ndt_scan_matcher), and initial pose
// estimation by aligning many hypotheses at once.
//
// Autoware estimates the initial pose by running NDT from about 200 candidate
// poses one after another on the CPU (yaw unknown, position from GNSS), which
// takes seconds. Here every hypothesis is one CUDA block: all of them iterate
// together, and the best one by NVTL (nearest voxel transformation
// likelihood) is the estimate. The same per-point code runs on the CPU (one
// thread, and hypotheses spread over all threads) for the comparison.
//
// Input: a sequence written by scripts/export_lidar_localization_sequence.py
// (real LiDAR scans with ground-truth sensor poses). The map is built from
// every map_stride-th scan placed at its ground-truth pose; test scans are
// taken from the others.

#include <cuda_runtime.h>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "cuda_check.cuh"

namespace cudabot {

constexpr float PI_F = 3.14159265f;
constexpr int NDT_THREADS = 256;
constexpr int NACC = 6 + 21 + 2;   // gradient, upper-triangular Hessian, score, NVTL

// ---------------------------------------------------------------------------
// Sequence
// ---------------------------------------------------------------------------
struct Frame {
    uint64_t stamp = 0;
    double pose[7];                 // x y z qx qy qz qw (map <- sensor)
    std::vector<float> pts;         // sensor frame, xyz
};

static bool load_sequence(const std::string& path, std::vector<Frame>& frames) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[8];
    uint32_t version = 0, count = 0;
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&count), 4);
    if (std::string(magic, 6) != "CRLOC1" || version != 1) return false;
    frames.resize(count);
    for (auto& f : frames) {
        uint32_t n = 0;
        in.read(reinterpret_cast<char*>(&f.stamp), 8);
        in.read(reinterpret_cast<char*>(f.pose), 7 * 8);
        in.read(reinterpret_cast<char*>(&n), 4);
        f.pts.resize(3 * (size_t)n);
        in.read(reinterpret_cast<char*>(f.pts.data()), 12 * (size_t)n);
    }
    return (bool)in;
}

// Pose as R (row-major) and t, in a local map frame (origin at the first scan).
struct Pose { float R[9]; float t[3]; };

static Pose frame_pose(const Frame& f, const double origin[3]) {
    Eigen::Quaterniond q(f.pose[6], f.pose[3], f.pose[4], f.pose[5]);
    Eigen::Matrix3d R = q.normalized().toRotationMatrix();
    Pose p;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) p.R[3 * i + j] = (float)R(i, j);
        p.t[i] = (float)(f.pose[i] - origin[i]);
    }
    return p;
}

static std::vector<float> voxel_downsample(const std::vector<float>& pts, float voxel, float min_r, float max_r) {
    struct Key { int64_t k; int i; };
    std::vector<Key> keys;
    for (int i = 0; i < (int)pts.size() / 3; i++) {
        float x = pts[3 * i], y = pts[3 * i + 1], z = pts[3 * i + 2];
        float r = std::sqrt(x * x + y * y + z * z);
        if (r < min_r || r > max_r) continue;
        int64_t a = (int64_t)std::floor(x / voxel) + 100000, b = (int64_t)std::floor(y / voxel) + 100000,
                c = (int64_t)std::floor(z / voxel) + 100000;
        keys.push_back({(a * 200000 + b) * 200000 + c, i});
    }
    std::sort(keys.begin(), keys.end(), [](const Key& u, const Key& v) { return u.k < v.k || (u.k == v.k && u.i < v.i); });
    std::vector<float> out;
    for (size_t k = 0; k < keys.size(); k++)
        if (k == 0 || keys[k].k != keys[k - 1].k)
            for (int d = 0; d < 3; d++) out.push_back(pts[3 * keys[k].i + d]);
    return out;
}

// ---------------------------------------------------------------------------
// NDT map: a dense voxel grid with mean and inverse covariance per cell
// ---------------------------------------------------------------------------
struct NdtMap {
    float res = 2.0f;
    float ox = 0, oy = 0, oz = 0;   // grid origin
    int nx = 0, ny = 0, nz = 0;
    float d1 = 0, d2 = 0;           // Magnusson's score constants
    std::vector<uint8_t> valid;
    std::vector<float> mean;        // 3 per cell
    std::vector<float> icov;        // 6 per cell (xx xy xz yy yz zz)
};

static void ndt_constants(float res, float outlier_ratio, float& d1, float& d2) {
    double c1 = 10.0 * (1.0 - outlier_ratio), c2 = outlier_ratio / std::pow((double)res, 3.0);
    double d3 = -std::log(c2);
    d1 = (float)(-std::log(c1 + c2) - d3);
    d2 = (float)(-2.0 * std::log((-std::log(c1 * std::exp(-0.5) + c2) - d3) / d1));
}

static NdtMap build_map(const std::vector<Frame>& frames, const double origin[3], int stride, float res) {
    NdtMap m;
    m.res = res;
    ndt_constants(res, 0.55f, m.d1, m.d2);
    std::vector<float> cloud;
    for (size_t k = 0; k < frames.size(); k += stride) {
        Pose p = frame_pose(frames[k], origin);
        const auto& s = frames[k].pts;
        for (size_t i = 0; i + 2 < s.size(); i += 3)
            for (int r = 0; r < 3; r++)
                cloud.push_back(p.R[3 * r] * s[i] + p.R[3 * r + 1] * s[i + 1] + p.R[3 * r + 2] * s[i + 2] + p.t[r]);
    }
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (size_t i = 0; i + 2 < cloud.size(); i += 3)
        for (int d = 0; d < 3; d++) { lo[d] = std::min(lo[d], cloud[i + d]); hi[d] = std::max(hi[d], cloud[i + d]); }
    m.ox = lo[0] - res; m.oy = lo[1] - res; m.oz = lo[2] - res;
    m.nx = (int)((hi[0] - m.ox) / res) + 2; m.ny = (int)((hi[1] - m.oy) / res) + 2; m.nz = (int)((hi[2] - m.oz) / res) + 2;
    const size_t nc = (size_t)m.nx * m.ny * m.nz;
    std::vector<double> acc(nc * 10, 0.0);   // count, sum xyz, sum of outer products (6)
    for (size_t i = 0; i + 2 < cloud.size(); i += 3) {
        float x = cloud[i], y = cloud[i + 1], z = cloud[i + 2];
        size_t c = ((size_t)((z - m.oz) / res) * m.ny + (size_t)((y - m.oy) / res)) * m.nx + (size_t)((x - m.ox) / res);
        double* a = &acc[c * 10];
        a[0] += 1; a[1] += x; a[2] += y; a[3] += z;
        a[4] += (double)x * x; a[5] += (double)x * y; a[6] += (double)x * z;
        a[7] += (double)y * y; a[8] += (double)y * z; a[9] += (double)z * z;
    }
    m.valid.assign(nc, 0); m.mean.assign(nc * 3, 0.0f); m.icov.assign(nc * 6, 0.0f);
    for (size_t c = 0; c < nc; c++) {
        const double* a = &acc[c * 10];
        if (a[0] < 6) continue;
        double n = a[0];
        Eigen::Vector3d mu(a[1] / n, a[2] / n, a[3] / n);
        Eigen::Matrix3d S;
        S << a[4] / n - mu.x() * mu.x(), a[5] / n - mu.x() * mu.y(), a[6] / n - mu.x() * mu.z(),
             0, a[7] / n - mu.y() * mu.y(), a[8] / n - mu.y() * mu.z(),
             0, 0, a[9] / n - mu.z() * mu.z();
        S(1, 0) = S(0, 1); S(2, 0) = S(0, 2); S(2, 1) = S(1, 2);
        S *= n / (n - 1);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(S);
        Eigen::Vector3d ev = es.eigenvalues();
        if (ev(2) <= 0) continue;
        for (int k = 0; k < 2; k++) ev(k) = std::max(ev(k), 0.01 * ev(2));   // as PCL
        Eigen::Matrix3d Ci = es.eigenvectors() * ev.cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
        m.valid[c] = 1;
        for (int d = 0; d < 3; d++) m.mean[c * 3 + d] = (float)mu(d);
        m.icov[c * 6 + 0] = (float)Ci(0, 0); m.icov[c * 6 + 1] = (float)Ci(0, 1); m.icov[c * 6 + 2] = (float)Ci(0, 2);
        m.icov[c * 6 + 3] = (float)Ci(1, 1); m.icov[c * 6 + 4] = (float)Ci(1, 2); m.icov[c * 6 + 5] = (float)Ci(2, 2);
    }
    return m;
}

// Read-only map view usable on host and device.
struct MapView {
    float res, ox, oy, oz, d1, d2;
    int nx, ny, nz;
    const uint8_t* valid;
    const float* mean;
    const float* icov;
};

// ---------------------------------------------------------------------------
// Per-point NDT terms (shared by CPU and GPU)
// ---------------------------------------------------------------------------
// Cost C = sum over points and nearby cells of -d1 exp(-d2/2 q' A q), minimised
// with Gauss-Newton on the right-perturbed pose: x = R p + t, dx/dt = I,
// dx/dw = -R [p]x. acc gets the gradient (6), the upper-triangular Hessian (21),
// the score (sum of -d1 e) and the NVTL term (max over cells of -d1 e).
__host__ __device__ inline void ndt_point(const MapView& m, const float* R, const float* t, float px, float py,
                                          float pz, float* acc)
{
    float x = R[0] * px + R[1] * py + R[2] * pz + t[0];
    float y = R[3] * px + R[4] * py + R[5] * pz + t[1];
    float z = R[6] * px + R[7] * py + R[8] * pz + t[2];
    int ci = (int)floorf((x - m.ox) / m.res), cj = (int)floorf((y - m.oy) / m.res), ck = (int)floorf((z - m.oz) / m.res);
    // dx/dw = -R [p]x : columns for w = (wx, wy, wz)
    float rp[9];   // R * [p]x, row-major
    rp[0] = R[1] * pz - R[2] * py; rp[1] = R[2] * px - R[0] * pz; rp[2] = R[0] * py - R[1] * px;
    rp[3] = R[4] * pz - R[5] * py; rp[4] = R[5] * px - R[3] * pz; rp[5] = R[3] * py - R[4] * px;
    rp[6] = R[7] * pz - R[8] * py; rp[7] = R[8] * px - R[6] * pz; rp[8] = R[6] * py - R[7] * px;
    const int di[7] = {0, 1, -1, 0, 0, 0, 0}, dj[7] = {0, 0, 0, 1, -1, 0, 0}, dk[7] = {0, 0, 0, 0, 0, 1, -1};
    float best = 0.0f;
    for (int n = 0; n < 7; n++) {
        int a = ci + di[n], b = cj + dj[n], c = ck + dk[n];
        if (a < 0 || a >= m.nx || b < 0 || b >= m.ny || c < 0 || c >= m.nz) continue;
        int cell = (c * m.ny + b) * m.nx + a;
        if (!m.valid[cell]) continue;
        const float* mu = m.mean + 3 * cell;
        const float* A = m.icov + 6 * cell;
        float q0 = x - mu[0], q1 = y - mu[1], q2 = z - mu[2];
        float Aq0 = A[0] * q0 + A[1] * q1 + A[2] * q2;
        float Aq1 = A[1] * q0 + A[3] * q1 + A[4] * q2;
        float Aq2 = A[2] * q0 + A[4] * q1 + A[5] * q2;
        float e = expf(-0.5f * m.d2 * (q0 * Aq0 + q1 * Aq1 + q2 * Aq2));
        float s = -m.d1 * e;
        if (s < 1e-6f) continue;
        best = fmaxf(best, s);
        float w = s * m.d2;
        // J = [I | -R[p]x]; J^T A q and J^T A J
        float J[18];   // 3 rows x 6
        for (int r = 0; r < 3; r++) {
            J[r * 6 + 0] = r == 0; J[r * 6 + 1] = r == 1; J[r * 6 + 2] = r == 2;
            J[r * 6 + 3] = -rp[r * 3 + 0]; J[r * 6 + 4] = -rp[r * 3 + 1]; J[r * 6 + 5] = -rp[r * 3 + 2];
        }
        float AJ[18];
        for (int col = 0; col < 6; col++) {
            float j0 = J[col], j1 = J[6 + col], j2 = J[12 + col];
            AJ[col] = A[0] * j0 + A[1] * j1 + A[2] * j2;
            AJ[6 + col] = A[1] * j0 + A[3] * j1 + A[4] * j2;
            AJ[12 + col] = A[2] * j0 + A[4] * j1 + A[5] * j2;
        }
        for (int col = 0; col < 6; col++) acc[col] += w * (J[col] * Aq0 + J[6 + col] * Aq1 + J[12 + col] * Aq2);
        int h = 6;
        for (int r = 0; r < 6; r++)
            for (int col = r; col < 6; col++, h++)
                acc[h] += w * (J[r] * AJ[col] + J[6 + r] * AJ[6 + col] + J[12 + r] * AJ[12 + col]);
        acc[27] += s;
    }
    acc[28] += best;
}

// Solve H d = -g (6x6, upper triangle given) and apply the clamped step to the pose.
// Returns the step size (translation norm + rotation norm).
__host__ __device__ inline float ndt_update(const float* acc, float* R, float* t, float max_t, float max_r) {
    float H[36], b[6];
    int h = 6;
    for (int r = 0; r < 6; r++)
        for (int c = r; c < 6; c++, h++) { H[r * 6 + c] = acc[h]; H[c * 6 + r] = acc[h]; }
    for (int r = 0; r < 6; r++) { b[r] = -acc[r]; H[r * 6 + r] += 1e-3f + 1e-6f * fabsf(H[r * 6 + r]); }
    // Gaussian elimination with partial pivoting
    for (int c = 0; c < 6; c++) {
        int p = c;
        for (int r = c + 1; r < 6; r++) if (fabsf(H[r * 6 + c]) > fabsf(H[p * 6 + c])) p = r;
        if (p != c) {
            for (int k = 0; k < 6; k++) { float tmp = H[c * 6 + k]; H[c * 6 + k] = H[p * 6 + k]; H[p * 6 + k] = tmp; }
            float tmp = b[c]; b[c] = b[p]; b[p] = tmp;
        }
        if (fabsf(H[c * 6 + c]) < 1e-12f) return 0.0f;
        for (int r = c + 1; r < 6; r++) {
            float f = H[r * 6 + c] / H[c * 6 + c];
            for (int k = c; k < 6; k++) H[r * 6 + k] -= f * H[c * 6 + k];
            b[r] -= f * b[c];
        }
    }
    float d[6];
    for (int r = 5; r >= 0; r--) {
        float s = b[r];
        for (int k = r + 1; k < 6; k++) s -= H[r * 6 + k] * d[k];
        d[r] = s / H[r * 6 + r];
    }
    float nt = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]), nr = sqrtf(d[3] * d[3] + d[4] * d[4] + d[5] * d[5]);
    float st = nt > max_t ? max_t / nt : 1.0f, sr = nr > max_r ? max_r / nr : 1.0f;
    for (int k = 0; k < 3; k++) { d[k] *= st; d[3 + k] *= sr; }
    for (int k = 0; k < 3; k++) t[k] += d[k];
    // R <- R Exp(w)
    float w0 = d[3], w1 = d[4], w2 = d[5], th = sqrtf(w0 * w0 + w1 * w1 + w2 * w2);
    float E[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (th > 1e-9f) {
        float a = sinf(th) / th, bb = (1.0f - cosf(th)) / (th * th);
        float K[9] = {0, -w2, w1, w2, 0, -w0, -w1, w0, 0};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                float kk = 0.0f;
                for (int l = 0; l < 3; l++) kk += K[i * 3 + l] * K[l * 3 + j];
                E[i * 3 + j] += a * K[i * 3 + j] + bb * kk;
            }
    }
    float Rn[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) Rn[i * 3 + j] = R[i * 3] * E[j] + R[i * 3 + 1] * E[3 + j] + R[i * 3 + 2] * E[6 + j];
    for (int k = 0; k < 9; k++) R[k] = Rn[k];
    return fminf(nt, max_t) + fminf(nr, max_r);
}

// ---------------------------------------------------------------------------
// GPU batch: one block per hypothesis
// ---------------------------------------------------------------------------
__global__ void ndt_batch_kernel(MapView m, const float* pts, int n, const float* poses, const int* done, float* out, int stride) {
    __shared__ float sh[NDT_THREADS];
    const int hyp = blockIdx.x;
    if (done[hyp]) return;
    const float* P = poses + 12 * hyp;
    float acc[NACC];
    for (int k = 0; k < NACC; k++) acc[k] = 0.0f;
    for (int i = threadIdx.x * stride; i < n; i += blockDim.x * stride) ndt_point(m, P, P + 9, pts[3 * i], pts[3 * i + 1], pts[3 * i + 2], acc);
    for (int k = 0; k < NACC; k++) {
        sh[threadIdx.x] = acc[k];
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) out[hyp * NACC + k] = sh[0];
        __syncthreads();
    }
}

__global__ void ndt_update_kernel(int nh, float* poses, int* done, const float* out, float max_t, float max_r, float eps) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= nh || done[h]) return;
    float* P = poses + 12 * h;
    float step = ndt_update(out + h * NACC, P, P + 9, max_t, max_r);
    if (step < eps) done[h] = 1;
}

struct AlignParams { int iters = 30, stride = 1; float max_t = 0.5f, max_r = 0.2f, eps = 1e-3f; };

// Align every hypothesis in `poses` (12 floats each, updated in place); nvtl gets
// each final pose's mean per-point best score.
static void align_gpu(const MapView& dm, const float* d_pts, int n, std::vector<float>& poses, std::vector<float>& nvtl,
                      const AlignParams& ap)
{
    const int nh = (int)poses.size() / 12;
    float *d_poses, *d_out; int* d_done;
    CUDA_CHECK(cudaMalloc(&d_poses, poses.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)nh * NACC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_done, nh * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_poses, poses.data(), poses.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_done, 0, nh * sizeof(int)));
    for (int it = 0; it < ap.iters; it++) {
        ndt_batch_kernel<<<nh, NDT_THREADS>>>(dm, d_pts, n, d_poses, d_done, d_out, ap.stride);
        ndt_update_kernel<<<(nh + 127) / 128, 128>>>(nh, d_poses, d_done, d_out, ap.max_t, ap.max_r, ap.eps);
    }
    CUDA_CHECK(cudaMemset(d_done, 0, nh * sizeof(int)));
    ndt_batch_kernel<<<nh, NDT_THREADS>>>(dm, d_pts, n, d_poses, d_done, d_out, ap.stride);   // final scores
    std::vector<float> out((size_t)nh * NACC);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(poses.data(), d_poses, poses.size() * sizeof(float), cudaMemcpyDeviceToHost));
    nvtl.resize(nh);
    for (int h = 0; h < nh; h++) nvtl[h] = out[h * NACC + 28] / ((n + ap.stride - 1) / ap.stride);
    cudaFree(d_poses); cudaFree(d_out); cudaFree(d_done);
}

// The same alignment on the CPU, hypotheses spread over `threads` threads.
static void align_cpu(const MapView& m, const std::vector<float>& pts, std::vector<float>& poses, std::vector<float>& nvtl,
                      const AlignParams& ap, int threads)
{
    const int nh = (int)poses.size() / 12, n = (int)pts.size() / 3;
    nvtl.assign(nh, 0.0f);
    auto work = [&](int k) {
        for (int h = k; h < nh; h += threads) {
            float* P = &poses[12 * h];
            float acc[NACC];
            for (int it = 0; it < ap.iters; it++) {
                std::fill(acc, acc + NACC, 0.0f);
                for (int i = 0; i < n; i += ap.stride) ndt_point(m, P, P + 9, pts[3 * i], pts[3 * i + 1], pts[3 * i + 2], acc);
                if (ndt_update(acc, P, P + 9, ap.max_t, ap.max_r) < ap.eps) break;
            }
            std::fill(acc, acc + NACC, 0.0f);
            for (int i = 0; i < n; i += ap.stride) ndt_point(m, P, P + 9, pts[3 * i], pts[3 * i + 1], pts[3 * i + 2], acc);
            nvtl[h] = acc[28] / ((n + ap.stride - 1) / ap.stride);
        }
    };
    std::vector<std::thread> pool;
    for (int k = 1; k < threads; k++) pool.emplace_back(work, k);
    work(0);
    for (auto& t : pool) t.join();
}

// Initial pose estimation: align every hypothesis, then refine the `refine`
// best (by NVTL) with more, smaller steps; returns the index of the best in
// poses (12 floats each, updated in place). use_gpu = false runs it on the CPU.
static int initialize(const MapView& hm, const MapView& dm, const std::vector<float>& scan, const float* d_pts,
                      std::vector<float>& poses, bool use_gpu, int threads, int refine,
                      int screen_stride = 1, int screen_iters = 12, int survivors = 64)
{
    const int n = (int)scan.size() / 3, nh = (int)poses.size() / 12;
    AlignParams coarse, fine;
    fine.iters = 60; fine.max_t = 0.2f; fine.max_r = 0.05f; fine.eps = 1e-4f;
    std::vector<float> nv;
    std::vector<int> order(nh);
    std::iota(order.begin(), order.end(), 0);
    if (screen_stride > 1) {
        // Broad coverage first; only promising candidates pay for the full scan.
        AlignParams screen = coarse; screen.stride = screen_stride; screen.iters = screen_iters;
        if (use_gpu) align_gpu(dm, d_pts, n, poses, nv, screen);
        else align_cpu(hm, scan, poses, nv, screen, threads);
        const int keep = std::min(nh, std::max(refine, survivors));
        std::partial_sort(order.begin(), order.begin() + keep, order.end(), [&](int a, int b) {
            return nv[a] == nv[b] ? a < b : nv[a] > nv[b];
        });
        order.resize(keep);
        std::vector<float> selected;
        for (int h : order) selected.insert(selected.end(), poses.begin() + 12 * h, poses.begin() + 12 * h + 12);
        if (use_gpu) align_gpu(dm, d_pts, n, selected, nv, coarse);
        else align_cpu(hm, scan, selected, nv, coarse, threads);
        for (int q = 0; q < keep; q++)
            std::copy(selected.begin() + 12 * q, selected.begin() + 12 * q + 12, poses.begin() + 12 * order[q]);
        std::vector<float> full_scores(nh, -1.0f);
        for (int q = 0; q < keep; q++) full_scores[order[q]] = nv[q];
        nv.swap(full_scores);
    } else {
        if (use_gpu) align_gpu(dm, d_pts, n, poses, nv, coarse);
        else align_cpu(hm, scan, poses, nv, coarse, threads);
    }
    const int k = std::min(refine, (int)order.size());
    std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int a, int b) { return nv[a] > nv[b]; });
    std::vector<float> top, tv;
    for (int q = 0; q < k; q++) top.insert(top.end(), poses.begin() + 12 * order[q], poses.begin() + 12 * order[q] + 12);
    if (use_gpu) align_gpu(dm, d_pts, n, top, tv, fine);
    else align_cpu(hm, scan, top, tv, fine, threads);
    int best = (int)(std::max_element(tv.begin(), tv.end()) - tv.begin());
    std::copy(top.begin() + 12 * best, top.begin() + 12 * best + 12, poses.begin() + 12 * order[best]);
    return order[best];
}

// ---------------------------------------------------------------------------
// Experiment
// ---------------------------------------------------------------------------
static float yaw_of(const float* R) { return std::atan2(R[3], R[0]); }

// Rotate R about the map z axis by a.
static void rotz_left(float a, const float* R, float* out) {
    float c = std::cos(a), s = std::sin(a);
    for (int j = 0; j < 3; j++) {
        out[j] = c * R[j] - s * R[3 + j];
        out[3 + j] = s * R[j] + c * R[3 + j];
        out[6 + j] = R[6 + j];
    }
}

}  // namespace cudabot

int main(int argc, char** argv) {
    using namespace cudabot;
    std::string seq_path, csv_path, map_seq_path;
    int map_seq_stride = 2;
    int map_stride = 10, tests = 40, yaws = 16, grid = 3, cpu_tests = 5, refine = 4;
    int screen_stride = 1, screen_iters = 12, survivors = 64;
    float prior_err = 2.0f, grid_step = 2.0f, scan_voxel = 1.0f, map_res = 2.0f;
    unsigned seed = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--sequence") seq_path = next();
        else if (a == "--csv") csv_path = next();
        else if (a == "--map-sequence") map_seq_path = next();
        else if (a == "--map-sequence-stride") map_seq_stride = std::atoi(next().c_str());
        else if (a == "--map-stride") map_stride = std::atoi(next().c_str());
        else if (a == "--tests") tests = std::atoi(next().c_str());
        else if (a == "--yaws") yaws = std::atoi(next().c_str());
        else if (a == "--grid") grid = std::atoi(next().c_str());
        else if (a == "--grid-step") grid_step = (float)std::atof(next().c_str());
        else if (a == "--prior-err") prior_err = (float)std::atof(next().c_str());
        else if (a == "--scan-voxel") scan_voxel = (float)std::atof(next().c_str());
        else if (a == "--map-res") map_res = (float)std::atof(next().c_str());
        else if (a == "--refine") refine = std::atoi(next().c_str());
        else if (a == "--screen-stride") screen_stride = std::atoi(next().c_str());
        else if (a == "--screen-iters") screen_iters = std::atoi(next().c_str());
        else if (a == "--survivors") survivors = std::atoi(next().c_str());
        else if (a == "--cpu-tests") cpu_tests = std::atoi(next().c_str());
        else if (a == "--seed") seed = (unsigned)std::atoi(next().c_str());
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }
    if (screen_stride < 1 || screen_iters < 1 || survivors < 1 || refine < 1 || tests < 1 ||
        map_stride < 1 || map_seq_stride < 1 || grid < 1 || yaws < 1 || scan_voxel <= 0 || map_res <= 0) {
        std::fprintf(stderr, "counts, strides, resolutions and iteration limits must be positive\n"); return 2;
    }
    std::vector<Frame> frames;
    if (!load_sequence(seq_path, frames) || frames.empty()) { std::fprintf(stderr, "cannot read %s\n", seq_path.c_str()); return 1; }
    const double origin[3] = {frames[0].pose[0], frames[0].pose[1], frames[0].pose[2]};
    using clk = std::chrono::high_resolution_clock;
    auto ms = [](clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); };

    auto t0 = clk::now();
    // the map: every map_stride-th test-sequence scan, or (--map-sequence) another session's
    // scans at their ground-truth poses in the same world frame
    std::vector<Frame> map_frames;
    if (!map_seq_path.empty() && (!load_sequence(map_seq_path, map_frames) || map_frames.empty())) {
        std::fprintf(stderr, "cannot read %s\n", map_seq_path.c_str());
        return 1;
    }
    const bool other = !map_frames.empty();
    const int mstride = other ? map_seq_stride : map_stride;
    NdtMap map = build_map(other ? map_frames : frames, origin, mstride, map_res);
    map_frames.clear(); map_frames.shrink_to_fit();
    size_t nvalid = std::count(map.valid.begin(), map.valid.end(), 1);
    std::printf("map (%s): stride %d, %d x %d x %d cells at %.1f m, %zu valid, built in %.0f ms\n",
                other ? map_seq_path.c_str() : "this sequence", mstride, map.nx, map.ny, map.nz, map.res, nvalid,
                ms(t0));
    MapView hm{map.res, map.ox, map.oy, map.oz, map.d1, map.d2, map.nx, map.ny, map.nz,
               map.valid.data(), map.mean.data(), map.icov.data()};
    uint8_t* d_valid; float *d_mean, *d_icov;
    CUDA_CHECK(cudaMalloc(&d_valid, map.valid.size()));
    CUDA_CHECK(cudaMalloc(&d_mean, map.mean.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_icov, map.icov.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_valid, map.valid.data(), map.valid.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mean, map.mean.data(), map.mean.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_icov, map.icov.data(), map.icov.size() * sizeof(float), cudaMemcpyHostToDevice));
    MapView dm = hm; dm.valid = d_valid; dm.mean = d_mean; dm.icov = d_icov;

    // test scans: evenly spread, never a map scan
    std::vector<int> test_idx;
    // halfway between two map scans
    for (int k = 0; k < tests; k++) {
        int f = (int)((k + 0.5) * frames.size() / tests) / map_stride * map_stride + map_stride / 2;
        if (f < (int)frames.size()) test_idx.push_back(f);
    }
    const int hw = std::max(1u, std::thread::hardware_concurrency());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    AlignParams ap;
    std::FILE* csv = csv_path.empty() ? nullptr : std::fopen(csv_path.c_str(), "w");
    if (csv) std::fprintf(csv, "frame,points,hypotheses,track_err_m,init_ok,init_err_m,init_yaw_err_deg,gpu_ms,cpu1_ms,cpun_ms,prior_x,prior_y,prior_z,prior_phase,est_tx,est_ty,est_tz,est_r00,est_r01,est_r02,est_r10,est_r11,est_r12,est_r20,est_r21,est_r22\n");
    int ok = 0, track_ok = 0, cpu_runs = 0;
    double gsum = 0, c1sum = 0, cnsum = 0, gsub = 0;
    for (size_t k = 0; k < test_idx.size(); k++) {
        const Frame& fr = frames[test_idx[k]];
        Pose gt = frame_pose(fr, origin);
        std::vector<float> scan = voxel_downsample(fr.pts, scan_voxel, 1.0f, 60.0f);
        const int n = (int)scan.size() / 3;
        float* d_pts;
        CUDA_CHECK(cudaMalloc(&d_pts, scan.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_pts, scan.data(), scan.size() * sizeof(float), cudaMemcpyHostToDevice));
        // tracking sanity: NDT started at the ground truth
        std::vector<float> one(12), nv;
        std::copy(gt.R, gt.R + 9, one.begin()); std::copy(gt.t, gt.t + 3, one.begin() + 9);
        align_gpu(dm, d_pts, n, one, nv, ap);
        float track_err = std::sqrt((one[9] - gt.t[0]) * (one[9] - gt.t[0]) + (one[10] - gt.t[1]) * (one[10] - gt.t[1]) +
                                    (one[11] - gt.t[2]) * (one[11] - gt.t[2]));
        track_ok += track_err < 0.3f;
        // initial pose estimation: GNSS-like position prior (error up to prior_err in xy, 1 m in z),
        // roll and pitch from gravity, yaw unknown
        float a = 2.0f * PI_F * U(rng), r = prior_err * std::sqrt(U(rng));
        float px = gt.t[0] + r * std::cos(a), py = gt.t[1] + r * std::sin(a), pz = gt.t[2] + (2.0f * U(rng) - 1.0f);
        float phase = 2.0f * PI_F * U(rng);
        std::vector<float> hyps;
        for (int gx = 0; gx < grid; gx++)
            for (int gy = 0; gy < grid; gy++)
                for (int y = 0; y < yaws; y++) {
                    float R[9];
                    rotz_left(phase + y * 2.0f * PI_F / yaws, gt.R, R);
                    for (int q = 0; q < 9; q++) hyps.push_back(R[q]);
                    hyps.push_back(px + (gx - (grid - 1) * 0.5f) * grid_step);
                    hyps.push_back(py + (gy - (grid - 1) * 0.5f) * grid_step);
                    hyps.push_back(pz);
                }
        const int nh = (int)hyps.size() / 12;
        std::vector<float> g = hyps;
        CUDA_CHECK(cudaDeviceSynchronize());
        auto tg = clk::now();
        int best = initialize(hm, dm, scan, d_pts, g, true, 1, refine, screen_stride, screen_iters, survivors);
        double gms = ms(tg);
        const float* B = &g[12 * best];
        float err = std::sqrt((B[9] - gt.t[0]) * (B[9] - gt.t[0]) + (B[10] - gt.t[1]) * (B[10] - gt.t[1]) +
                              (B[11] - gt.t[2]) * (B[11] - gt.t[2]));
        float yerr = std::fabs(std::remainder(yaw_of(B) - yaw_of(gt.R), 2.0f * PI_F)) * 180.0f / PI_F;
        bool success = err < 0.5f && yerr < 2.0f;
        ok += success;
        gsum += gms;
        double c1 = -1, cn = -1;
        if ((int)k < cpu_tests) {
            std::vector<float> c = hyps;
            auto tc = clk::now();
            initialize(hm, dm, scan, nullptr, c, false, 1, refine, screen_stride, screen_iters, survivors);
            c1 = ms(tc);
            c = hyps;
            tc = clk::now();
            int cb = initialize(hm, dm, scan, nullptr, c, false, hw, refine, screen_stride, screen_iters, survivors);
            cn = ms(tc);
            if (cb != best) {
                // hypotheses that converge to the same pose tie on NVTL up to summation order
                const float* C = &c[12 * cb];
                float dp = std::sqrt((C[9] - B[9]) * (C[9] - B[9]) + (C[10] - B[10]) * (C[10] - B[10]) +
                                     (C[11] - B[11]) * (C[11] - B[11]));
                float dy = std::fabs(std::remainder(yaw_of(C) - yaw_of(B), 2.0f * PI_F)) * 180.0f / PI_F;
                std::printf("  frame %d: CPU picks hypothesis %d, GPU %d; final poses %.3f m, %.2f deg apart\n",
                            test_idx[k], cb, best, dp, dy);
            }
            c1sum += c1; cnsum += cn; gsub += gms; cpu_runs++;
        }
        std::printf("frame %4d: %5d points, %d hypotheses, track err %.3f m, init %s (err %.2f m, %.1f deg), "
                    "GPU %.1f ms%s\n", test_idx[k], n, nh, track_err, success ? "ok  " : "FAIL", err, yerr, gms,
                    c1 >= 0 ? (", CPU 1 thread " + std::to_string((int)c1) + " ms, " + std::to_string(hw) +
                               " threads " + std::to_string((int)cn) + " ms").c_str() : "");
        if (csv) {
            std::fprintf(csv, "%d,%d,%d,%.4f,%d,%.4f,%.3f,%.3f,%.3f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f", test_idx[k], n, nh, track_err, (int)success,
                         err, yerr, gms, c1, cn, px, py, pz, phase, B[9], B[10], B[11]);
            for (int q = 0; q < 9; q++) std::fprintf(csv, ",%.8f", B[q]);
            std::fprintf(csv, "\n");
        }
        cudaFree(d_pts);
    }
    if (csv) std::fclose(csv);
    const int nt = (int)test_idx.size();
    std::printf("\ntracking from the ground truth within 0.3 m: %d/%d\n", track_ok, nt);
    std::printf("initial pose (prior error up to %.1f m, yaw unknown, %d hypotheses): %d/%d within 0.5 m and 2 deg\n",
                prior_err, grid * grid * yaws, ok, nt);
    std::printf("time per initialization: GPU %.1f ms (all %d); on the %d CPU-timed scans GPU %.1f ms, CPU 1 thread %.0f ms, "
                "CPU %d threads %.0f ms\n", gsum / nt, nt, cpu_runs, gsub / std::max(1, cpu_runs),
                c1sum / std::max(1, cpu_runs), hw, cnsum / std::max(1, cpu_runs));
    cudaFree(d_valid); cudaFree(d_mean); cudaFree(d_icov);
    return 0;
}
