// gpu_ground_segmentation.cu
//
// GPU LiDAR ground segmentation (CPU vs CUDA comparison).
//
// Separating ground from non-ground points is the first step of most LiDAR
// pipelines (obstacle extraction, clustering, mapping). A single height
// threshold fails as soon as the ground is not flat: a ramp rises above it and
// a curb splits it. This demo implements a concentric-zone ground model in the
// spirit of Patchwork: points are binned into a polar grid (rings x sectors),
// each bin fits its own plane, and each sector then checks its planes from the
// sensor outward so that a flat roof does not pass for ground.
//
//   1. bin:   each point gets a polar bin id; points are stably sorted by bin
//             (thrust on the GPU, std::stable_sort on the CPU: same order).
//   2. fit:   one warp = one bin. Seeds are the points within 0.25 m of the
//             bin's lowest point; a plane is fitted by PCA (smallest
//             eigenvector of the covariance) and refined twice on the points
//             within 0.12 m of it. A plane counts if it is within 25 deg of
//             horizontal.
//   3. check: one thread = one sector, walking the rings outward. A plane is
//             kept only if its height at the bin centre is within
//             0.25 m + 0.18 * ring width of the last kept plane (or of the flat
//             ground under the sensor for the first ring).
//   4. label: a point is ground if its bin's plane is kept and it lies within
//             0.12 m of it.
//
// The check and the labelling are __host__ __device__ routines shared by the CPU
// loop and the CUDA kernels. The fit is the same algorithm, but on the GPU the 32
// lanes of a warp share a bin's points, so its sums are taken in a different order
// than the CPU's serial loop.
//
// Scene: a synthetic 64 x 1024 LiDAR scan of terrain with a 6 deg ramp, a
// 0.15 m curb with a sidewalk and gentle undulation, plus cars, poles,
// pedestrians and a wall, ray-cast per beam with exact ground-truth labels.
// Several sensor poses are scanned. Reported: ground precision / recall / F1
// for the model and for a plain height threshold, and the CPU and GPU times.
//
// Output: gif/gpu_ground_segmentation.gif (bird's-eye view per scan)
//
// Options: --no-video, --check (exit non-zero unless the model's F1 >= 0.95,
// it beats the height threshold, and CPU and GPU labels agree on >= 99.9%).

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ---- sensor ----
static const int N_CH = 64, N_AZ = 1024, N_RAYS = N_CH * N_AZ;
static constexpr float PI_F = 3.14159265f;
static constexpr float VERT_MIN = -24.8f * PI_F / 180.0f, VERT_MAX = 2.0f * PI_F / 180.0f;
static constexpr float SENSOR_H = 1.8f, MAX_RANGE = 60.0f;

// ---- polar grid ----
static const int N_RING = 24, N_SECTOR = 72, N_BIN = N_RING * N_SECTOR;
static constexpr float R_MIN = 2.0f, R_MAX = 60.0f;
static constexpr float SEED_TH = 0.25f, DIST_TH = 0.12f;
static constexpr float UPRIGHT_COS = 0.906f;    // cos(25 deg)
static constexpr float STEP_TH = 0.25f, SLOPE_TH = 0.18f;   // height jump allowed between rings

// Ring boundaries grow geometrically so near rings are narrow and far ones wide.
__host__ __device__ static inline float ring_edge(int k) {
    return R_MIN * powf(R_MAX / R_MIN, (float)k / N_RING);
}

__host__ __device__ static inline int bin_of(float x, float y) {
    float r = sqrtf(x * x + y * y);
    if (r < R_MIN || r >= R_MAX) return -1;
    int ring = (int)(N_RING * logf(r / R_MIN) / logf(R_MAX / R_MIN));
    ring = ring < 0 ? 0 : (ring >= N_RING ? N_RING - 1 : ring);
    float a = atan2f(y, x) + PI_F;
    int sec = (int)(a / (2.0f * PI_F) * N_SECTOR);
    sec = sec >= N_SECTOR ? N_SECTOR - 1 : sec;
    return ring * N_SECTOR + sec;
}

// ---- scene (world frame, z up) ----
__host__ __device__ static inline float ground_h(float x, float y) {
    float h = 0.03f * sinf(0.35f * x) * cosf(0.27f * y);       // gentle undulation
    if (x > 10.0f) h += (x - 10.0f) * 0.105f;                   // 6 deg ramp
    if (y > 6.0f) h += 0.15f;                                   // curb + sidewalk
    return h;
}

struct Box { float x0, y0, x1, y1, h; };          // axis-aligned, standing on the ground
struct Cyl { float x, y, r, h; };

static const int N_BOX = 7, N_CYL = 8;
__constant__ Box c_box[N_BOX];
__constant__ Cyl c_cyl[N_CYL];

// Ray from o along unit d: nearest hit within MAX_RANGE. label 1 = ground, 0 = object.
__host__ __device__ static inline bool raycast(const float* o, const float* d,
                                               const Box* boxes, const Cyl* cyls,
                                               float& t_hit, int& label) {
    t_hit = MAX_RANGE; label = -1;
    // objects: slabs for boxes (bottom at the ground under the box centre)
    for (int b = 0; b < N_BOX; ++b) {
        const Box& B = boxes[b];
        float zb = ground_h(0.5f * (B.x0 + B.x1), 0.5f * (B.y0 + B.y1)) - 0.5f;
        float lo[3] = { B.x0, B.y0, zb }, hi[3] = { B.x1, B.y1, zb + 0.5f + B.h };
        float t0 = 0.0f, t1 = t_hit;
        bool hit = true;
        for (int a = 0; a < 3 && hit; ++a) {
            if (fabsf(d[a]) < 1e-8f) { if (o[a] < lo[a] || o[a] > hi[a]) hit = false; continue; }
            float ta = (lo[a] - o[a]) / d[a], tb = (hi[a] - o[a]) / d[a];
            if (ta > tb) { float tmp = ta; ta = tb; tb = tmp; }
            t0 = fmaxf(t0, ta); t1 = fminf(t1, tb);
            if (t0 > t1) hit = false;
        }
        if (hit && t0 > 0.0f && t0 < t_hit) { t_hit = t0; label = 0; }
    }
    for (int c = 0; c < N_CYL; ++c) {
        const Cyl& C = cyls[c];
        float ox = o[0] - C.x, oy = o[1] - C.y;
        float a = d[0] * d[0] + d[1] * d[1];
        if (a < 1e-10f) continue;
        float b = ox * d[0] + oy * d[1], cc = ox * ox + oy * oy - C.r * C.r;
        float disc = b * b - a * cc;
        if (disc < 0.0f) continue;
        float t = (-b - sqrtf(disc)) / a;
        if (t <= 0.0f || t >= t_hit) continue;
        float z = o[2] + t * d[2], zg = ground_h(C.x, C.y);
        if (z < zg - 0.2f || z > zg + C.h) continue;
        t_hit = t; label = 0;
    }
    // ground: march, then bisect
    float prev_t = 0.3f;
    for (float t = 0.5f; t < t_hit; t += 0.1f) {
        float x = o[0] + t * d[0], y = o[1] + t * d[1], z = o[2] + t * d[2];
        if (z <= ground_h(x, y)) {
            float lo = prev_t, hi = t;
            for (int it = 0; it < 12; ++it) {
                float m = 0.5f * (lo + hi);
                if (o[2] + m * d[2] <= ground_h(o[0] + m * d[0], o[1] + m * d[1])) hi = m; else lo = m;
            }
            t_hit = hi; label = 1;
            break;
        }
        prev_t = t;
    }
    return label >= 0;
}

__host__ __device__ static inline float hash_gauss(unsigned int a, unsigned int b) {
    unsigned int x = a * 2654435761u ^ (b + 0x9e3779b9u) * 2246822519u;
    x ^= x >> 15; x *= 0x2c1b3c6du; x ^= x >> 12; x *= 0x297a2d39u; x ^= x >> 15;
    float u1 = ((x & 0xffffu) + 0.5f) / 65536.0f, u2 = ((x >> 16) + 0.5f) / 65536.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI_F * u2);
}

// One ray per thread: sensor-frame point (x, y, z relative to the sensor) and label.
__global__ void scan_kernel(float sx, float sy, unsigned int scan_id, float* pts, int* gt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_RAYS) return;
    int ch = i / N_AZ, az = i - ch * N_AZ;
    float el = VERT_MIN + (VERT_MAX - VERT_MIN) * ch / (N_CH - 1);
    float yaw = 2.0f * PI_F * az / N_AZ;
    float o[3] = { sx, sy, ground_h(sx, sy) + SENSOR_H };
    float d[3] = { cosf(el) * cosf(yaw), cosf(el) * sinf(yaw), sinf(el) };
    float t; int label;
    if (!raycast(o, d, c_box, c_cyl, t, label)) { gt[i] = -1; return; }
    t += 0.02f * hash_gauss(scan_id, (unsigned int)i);   // 2 cm range noise
    pts[i * 3 + 0] = t * d[0];
    pts[i * 3 + 1] = t * d[1];
    pts[i * 3 + 2] = t * d[2];
    gt[i] = label;
}

// ---- ground model ----
struct BinPlane { float nx, ny, nz, d; int ok; float zc; };   // n . p + d = 0, height at bin centre

__host__ __device__ static inline void smallest_eigvec3(const float* C, float* v) {
    // Jacobi eigenvalue iterations on the symmetric 3x3 C; returns the eigenvector
    // of the smallest eigenvalue.
    float a[9], V[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    for (int k = 0; k < 9; ++k) a[k] = C[k];
    for (int sweep = 0; sweep < 8; ++sweep)
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) {
                float apq = a[p * 3 + q];
                if (fabsf(apq) < 1e-12f) continue;
                float theta = 0.5f * atan2f(2.0f * apq, a[q * 3 + q] - a[p * 3 + p]);
                float c = cosf(theta), s = sinf(theta);
                for (int k = 0; k < 3; ++k) {   // A <- A J
                    float akp = a[k * 3 + p], akq = a[k * 3 + q];
                    a[k * 3 + p] = c * akp - s * akq;
                    a[k * 3 + q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {   // A <- J^T A
                    float apk = a[p * 3 + k], aqk = a[q * 3 + k];
                    a[p * 3 + k] = c * apk - s * aqk;
                    a[q * 3 + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {   // V <- V J
                    float vkp = V[k * 3 + p], vkq = V[k * 3 + q];
                    V[k * 3 + p] = c * vkp - s * vkq;
                    V[k * 3 + q] = s * vkp + c * vkq;
                }
            }
    int m = 0;
    if (a[4] < a[m * 4]) m = 1;
    if (a[8] < a[m * 4]) m = 2;
    v[0] = V[0 * 3 + m]; v[1] = V[1 * 3 + m]; v[2] = V[2 * 3 + m];
}

// Fit the plane of one bin from its points pts[idx[k]] for k in [begin, end).
__host__ __device__ static inline BinPlane fit_bin(const float* pts, const int* idx, int begin, int end,
                                                   float cx, float cy) {
    BinPlane P; P.nx = 0; P.ny = 0; P.nz = 1; P.d = 0; P.ok = 0; P.zc = 0;
    if (end - begin < 5) return P;
    float zmin = 1e9f;
    for (int k = begin; k < end; ++k) zmin = fminf(zmin, pts[idx[k] * 3 + 2]);
    float n[3] = { 0, 0, 1 }, d = 0.0f;
    for (int round = 0; round < 3; ++round) {
        float cnt = 0, m[3] = { 0, 0, 0 };
        for (int k = begin; k < end; ++k) {
            const float* p = &pts[idx[k] * 3];
            bool use = round == 0 ? p[2] < zmin + SEED_TH
                                  : fabsf(n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + d) < DIST_TH;
            if (!use) continue;
            m[0] += p[0]; m[1] += p[1]; m[2] += p[2]; cnt += 1.0f;
        }
        if (cnt < 3.0f) return P;
        m[0] /= cnt; m[1] /= cnt; m[2] /= cnt;
        float C[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int k = begin; k < end; ++k) {
            const float* p = &pts[idx[k] * 3];
            bool use = round == 0 ? p[2] < zmin + SEED_TH
                                  : fabsf(n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + d) < DIST_TH;
            if (!use) continue;
            float a = p[0] - m[0], b = p[1] - m[1], c = p[2] - m[2];
            C[0] += a * a; C[1] += a * b; C[2] += a * c;
            C[4] += b * b; C[5] += b * c; C[8] += c * c;
        }
        C[3] = C[1]; C[6] = C[2]; C[7] = C[5];
        smallest_eigvec3(C, n);
        if (n[2] < 0.0f) { n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2]; }
        d = -(n[0] * m[0] + n[1] * m[1] + n[2] * m[2]);
    }
    P.nx = n[0]; P.ny = n[1]; P.nz = n[2]; P.d = d;
    P.ok = n[2] >= UPRIGHT_COS;
    P.zc = -(n[0] * cx + n[1] * cy + d) / n[2];
    return P;
}

__host__ __device__ static inline void bin_center(int b, float& cx, float& cy) {
    int ring = b / N_SECTOR, sec = b - ring * N_SECTOR;
    float r = 0.5f * (ring_edge(ring) + ring_edge(ring + 1));
    float a = (sec + 0.5f) / N_SECTOR * 2.0f * PI_F - PI_F;
    cx = r * cosf(a); cy = r * sinf(a);
}

// Walk one sector outward: keep a plane only if it continues the last kept one.
__host__ __device__ static inline void check_sector(BinPlane* planes, int sec) {
    float last_z = -SENSOR_H, last_r = 0.0f;
    for (int ring = 0; ring < N_RING; ++ring) {
        BinPlane& P = planes[ring * N_SECTOR + sec];
        if (!P.ok) continue;
        float r = 0.5f * (ring_edge(ring) + ring_edge(ring + 1));
        float allow = STEP_TH + SLOPE_TH * (r - last_r);
        if (P.zc > last_z + allow) { P.ok = 0; continue; }
        last_z = P.zc; last_r = r;
    }
}

__global__ void bin_kernel(const float* pts, const int* gt, int* bin, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bin[i] = gt[i] < 0 ? N_BIN : (bin_of(pts[i * 3 + 0], pts[i * 3 + 1]) < 0 ? N_BIN
                                   : bin_of(pts[i * 3 + 0], pts[i * 3 + 1]));
}

__global__ void bin_range_kernel(const int* sorted_bin, int* start, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = sorted_bin[i];
    if (i == 0 || sorted_bin[i - 1] != b) {
        for (int k = i == 0 ? 0 : sorted_bin[i - 1] + 1; k <= b; ++k) start[k] = i;
    }
    if (i == n - 1) for (int k = b + 1; k <= N_BIN; ++k) start[k] = n;
}

__device__ static inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// fit_bin with the 32 lanes of a warp sharing a bin's points (lane-strided sums,
// a fixed butterfly reduction, so the result is deterministic). Near bins hold
// hundreds to thousands of points; one thread per bin left those few threads
// doing all the work.
__device__ static inline BinPlane fit_bin_warp(const float* pts, const int* idx, int begin, int end,
                                               float cx, float cy, int lane) {
    BinPlane P; P.nx = 0; P.ny = 0; P.nz = 1; P.d = 0; P.ok = 0; P.zc = 0;
    if (end - begin < 5) return P;
    float zmin = 1e9f;
    for (int k = begin + lane; k < end; k += 32) zmin = fminf(zmin, pts[idx[k] * 3 + 2]);
    for (int o = 16; o > 0; o >>= 1) zmin = fminf(zmin, __shfl_xor_sync(0xffffffffu, zmin, o));
    float n[3] = { 0, 0, 1 }, d = 0.0f;
    for (int round = 0; round < 3; ++round) {
        float cnt = 0, m0 = 0, m1 = 0, m2 = 0;
        for (int k = begin + lane; k < end; k += 32) {
            const float* p = &pts[idx[k] * 3];
            bool use = round == 0 ? p[2] < zmin + SEED_TH
                                  : fabsf(n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + d) < DIST_TH;
            if (!use) continue;
            m0 += p[0]; m1 += p[1]; m2 += p[2]; cnt += 1.0f;
        }
        cnt = warp_sum(cnt); m0 = warp_sum(m0); m1 = warp_sum(m1); m2 = warp_sum(m2);
        if (cnt < 3.0f) return P;
        float m[3] = { m0 / cnt, m1 / cnt, m2 / cnt };
        float C[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int k = begin + lane; k < end; k += 32) {
            const float* p = &pts[idx[k] * 3];
            bool use = round == 0 ? p[2] < zmin + SEED_TH
                                  : fabsf(n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + d) < DIST_TH;
            if (!use) continue;
            float a = p[0] - m[0], b = p[1] - m[1], c = p[2] - m[2];
            C[0] += a * a; C[1] += a * b; C[2] += a * c;
            C[4] += b * b; C[5] += b * c; C[8] += c * c;
        }
        C[0] = warp_sum(C[0]); C[1] = warp_sum(C[1]); C[2] = warp_sum(C[2]);
        C[4] = warp_sum(C[4]); C[5] = warp_sum(C[5]); C[8] = warp_sum(C[8]);
        C[3] = C[1]; C[6] = C[2]; C[7] = C[5];
        smallest_eigvec3(C, n);   // every lane computes the same 3x3 result
        if (n[2] < 0.0f) { n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2]; }
        d = -(n[0] * m[0] + n[1] * m[1] + n[2] * m[2]);
    }
    P.nx = n[0]; P.ny = n[1]; P.nz = n[2]; P.d = d;
    P.ok = n[2] >= UPRIGHT_COS;
    P.zc = -(n[0] * cx + n[1] * cy + d) / n[2];
    return P;
}

// one warp = one bin
__global__ void fit_kernel(const float* pts, const int* idx, const int* start, BinPlane* planes) {
    int b = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (b >= N_BIN) return;
    float cx, cy;
    bin_center(b, cx, cy);
    BinPlane P = fit_bin_warp(pts, idx, start[b], start[b + 1], cx, cy, lane);
    if (lane == 0) planes[b] = P;
}

__global__ void check_kernel(BinPlane* planes) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < N_SECTOR) check_sector(planes, s);
}

__global__ void label_kernel(const float* pts, const int* bin, const BinPlane* planes, int* lab, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = bin[i];
    if (b >= N_BIN) { lab[i] = 0; return; }
    const BinPlane& P = planes[b];
    const float* p = &pts[i * 3];
    lab[i] = P.ok && fabsf(P.nx * p[0] + P.ny * p[1] + P.nz * p[2] + P.d) < DIST_TH;
}

}  // namespace cudabot

using namespace cudabot;

struct Score { double precision, recall, f1; };

static Score score(const std::vector<int>& gt, const std::vector<int>& lab) {
    long tp = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
        if (gt[i] < 0) continue;
        bool g = gt[i] == 1, l = lab[i] == 1;
        tp += g && l; fp += !g && l; fn += g && !l;
    }
    Score s;
    s.precision = tp + fp ? (double)tp / (tp + fp) : 0.0;
    s.recall = tp + fn ? (double)tp / (tp + fn) : 0.0;
    s.f1 = s.precision + s.recall > 0 ? 2 * s.precision * s.recall / (s.precision + s.recall) : 0.0;
    return s;
}

// CPU pipeline: same binning, stable sort, fit, check and label.
static void cpu_segment(const std::vector<float>& pts, const std::vector<int>& gt, std::vector<int>& lab) {
    int n = (int)gt.size();
    std::vector<int> bin(n), idx(n);
    for (int i = 0; i < n; ++i)
        bin[i] = gt[i] < 0 ? N_BIN : (bin_of(pts[i * 3], pts[i * 3 + 1]) < 0 ? N_BIN : bin_of(pts[i * 3], pts[i * 3 + 1]));
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return bin[a] < bin[b]; });
    std::vector<int> start(N_BIN + 1, n);
    for (int k = n - 1; k >= 0; --k) start[bin[idx[k]]] = k;
    for (int b = N_BIN - 1; b >= 0; --b) start[b] = std::min(start[b], start[b + 1]);
    std::vector<BinPlane> planes(N_BIN);
    for (int b = 0; b < N_BIN; ++b) {
        float cx, cy;
        bin_center(b, cx, cy);
        planes[b] = fit_bin(pts.data(), idx.data(), start[b], start[b + 1], cx, cy);
    }
    for (int s = 0; s < N_SECTOR; ++s) check_sector(planes.data(), s);
    lab.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        int b = bin[i];
        if (b >= N_BIN) continue;
        const BinPlane& P = planes[b];
        const float* p = &pts[i * 3];
        lab[i] = P.ok && std::fabs(P.nx * p[0] + P.ny * p[1] + P.nz * p[2] + P.d) < DIST_TH;
    }
}

int main(int argc, char** argv) {
    bool no_video = false, check = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--no-video")) no_video = true;
        else if (!std::strcmp(argv[i], "--check")) check = true;
    }
    std::printf("=== GPU LiDAR ground segmentation (CPU vs CUDA) ===\n");

    const Box h_box[N_BOX] = {
        { 6.0f, -3.5f, 10.5f, -1.7f, 1.5f },    // car
        { 13.0f, 2.0f, 17.5f, 3.8f, 1.6f },     // car on the ramp
        { -9.0f, -6.0f, -4.5f, -4.2f, 1.5f },   // car
        { -14.0f, 8.0f, -2.0f, 8.4f, 2.5f },    // wall on the sidewalk
        { 20.0f, -9.0f, 26.0f, -7.0f, 2.8f },   // van on the ramp
        { -20.0f, -14.0f, -16.0f, -10.0f, 1.0f },   // low crate
        { 2.0f, 9.0f, 4.0f, 11.0f, 0.6f },      // bench on the sidewalk
    };
    const Cyl h_cyl[N_CYL] = {
        { 4.0f, 6.8f, 0.15f, 4.0f }, { 12.0f, 6.8f, 0.15f, 4.0f }, { -6.0f, 6.8f, 0.15f, 4.0f },
        { 3.0f, -1.0f, 0.3f, 1.7f }, { -2.0f, 3.5f, 0.3f, 1.7f }, { 9.0f, 4.5f, 0.3f, 1.7f },
        { 16.0f, -4.0f, 0.3f, 1.8f }, { -12.0f, -2.0f, 0.3f, 1.7f },
    };
    CUDA_CHECK(cudaMemcpyToSymbol(c_box, h_box, sizeof(h_box)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_cyl, h_cyl, sizeof(h_cyl)));

    const int N_SCAN = 8;
    const float poses[N_SCAN][2] = { { 0, 0 }, { 4, 2 }, { 8, 0 }, { 12, -2 }, { 16, 0 }, { 2, 5 },
                                     { -5, 2 }, { 20, 3 } };
    float *d_pts; int *d_gt, *d_bin, *d_idx, *d_start, *d_lab;
    BinPlane* d_planes;
    CUDA_CHECK(cudaMalloc(&d_pts, N_RAYS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gt, N_RAYS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bin, N_RAYS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_idx, N_RAYS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_start, (N_BIN + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lab, N_RAYS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_planes, N_BIN * sizeof(BinPlane)));
    int* d_sorted_bin;
    CUDA_CHECK(cudaMalloc(&d_sorted_bin, N_RAYS * sizeof(int)));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    std::vector<int> all_gt, all_gpu, all_cpu, all_thr;
    double cpu_ms_total = 0.0, gpu_ms_total = 0.0;
    long agree = 0, valid = 0;
    std::vector<cv::Mat> frames;
    const int B = 256, G = (N_RAYS + B - 1) / B;
    for (int s = 0; s < N_SCAN; ++s) {
        scan_kernel<<<G, B>>>(poses[s][0], poses[s][1], (unsigned int)s, d_pts, d_gt);
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> pts(N_RAYS * 3);
        std::vector<int> gt(N_RAYS);
        CUDA_CHECK(cudaMemcpy(pts.data(), d_pts, pts.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gt.data(), d_gt, gt.size() * sizeof(int), cudaMemcpyDeviceToHost));

        // ---- GPU segmentation (timed: bin, sort, fit, check, label) ----
        for (int rep = 0; rep < 2; ++rep) {   // first pass warms up
            CUDA_CHECK(cudaEventRecord(e0));
            bin_kernel<<<G, B>>>(d_pts, d_gt, d_bin, N_RAYS);
            thrust::sequence(thrust::device_ptr<int>(d_idx), thrust::device_ptr<int>(d_idx) + N_RAYS);
            CUDA_CHECK(cudaMemcpy(d_sorted_bin, d_bin, N_RAYS * sizeof(int), cudaMemcpyDeviceToDevice));
            thrust::stable_sort_by_key(thrust::device_ptr<int>(d_sorted_bin),
                                       thrust::device_ptr<int>(d_sorted_bin) + N_RAYS,
                                       thrust::device_ptr<int>(d_idx));
            bin_range_kernel<<<G, B>>>(d_sorted_bin, d_start, N_RAYS);
            fit_kernel<<<(N_BIN * 32 + 127) / 128, 128>>>(d_pts, d_idx, d_start, d_planes);
            check_kernel<<<1, N_SECTOR>>>(d_planes);
            label_kernel<<<G, B>>>(d_pts, d_bin, d_planes, d_lab, N_RAYS);
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
        }
        CUDA_CHECK(cudaGetLastError());
        float gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));
        std::vector<int> lab(N_RAYS);
        CUDA_CHECK(cudaMemcpy(lab.data(), d_lab, lab.size() * sizeof(int), cudaMemcpyDeviceToHost));

        // ---- CPU segmentation ----
        std::vector<int> clab;
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_segment(pts, gt, clab);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // ---- height-threshold baseline: ground if within 0.25 m of flat ground under the sensor ----
        std::vector<int> thr(N_RAYS, 0);
        for (int i = 0; i < N_RAYS; ++i) thr[i] = gt[i] >= 0 && pts[i * 3 + 2] < -SENSOR_H + 0.25f;

        for (int i = 0; i < N_RAYS; ++i) {
            if (gt[i] < 0) continue;
            ++valid; agree += lab[i] == clab[i];
        }
        cpu_ms_total += cpu_ms; gpu_ms_total += gpu_ms;
        all_gt.insert(all_gt.end(), gt.begin(), gt.end());
        all_gpu.insert(all_gpu.end(), lab.begin(), lab.end());
        all_cpu.insert(all_cpu.end(), clab.begin(), clab.end());
        all_thr.insert(all_thr.end(), thr.begin(), thr.end());
        Score sm = score(gt, lab), st = score(gt, thr);
        std::printf("scan %d at (%5.1f, %5.1f): model F1 %.4f  height threshold F1 %.4f  CPU %7.2f ms  GPU %6.2f ms\n",
                    s, poses[s][0], poses[s][1], sm.f1, st.f1, cpu_ms, gpu_ms);

        if (!no_video) {
            // bird's-eye view, 40 m x 40 m around the sensor
            const int W = 560;
            cv::Mat panel(W, 2 * W + 20, CV_8UC3, cv::Scalar(28, 28, 32));
            auto px = [&](float x, float y, int ox) {
                return cv::Point(ox + (int)((x + 20.0f) / 40.0f * W), (int)((20.0f - y) / 40.0f * W));
            };
            for (int view = 0; view < 2; ++view) {
                int ox = view * (W + 20);
                for (int i = 0; i < N_RAYS; ++i) {
                    if (gt[i] < 0) continue;
                    float x = pts[i * 3], y = pts[i * 3 + 1];
                    if (std::fabs(x) > 20.0f || std::fabs(y) > 20.0f) continue;
                    const std::vector<int>& L = view == 0 ? thr : lab;
                    cv::Vec3b col;
                    bool g = gt[i] == 1, l = L[i] == 1;
                    if (g && l) col = cv::Vec3b(90, 150, 90);          // ground, right
                    else if (!g && !l) col = cv::Vec3b(200, 200, 210);  // object, right
                    else if (l) col = cv::Vec3b(60, 60, 235);           // object called ground
                    else col = cv::Vec3b(40, 200, 250);                 // ground missed
                    cv::Point p = px(x, y, ox);
                    if (p.x >= ox && p.x < ox + W && p.y >= 0 && p.y < W) panel.at<cv::Vec3b>(p) = col;
                }
                Score sc = view == 0 ? st : sm;
                char buf[160];
                std::snprintf(buf, sizeof(buf), "%s   F1 %.3f",
                              view == 0 ? "height threshold" : "concentric-zone model", sc.f1);
                cv::putText(panel, buf, cv::Point(ox + 10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(235, 235, 245), 1, cv::LINE_AA);
            }
            char buf[200];
            std::snprintf(buf, sizeof(buf), "scan %d   red: object called ground   yellow: ground missed   GPU %.2f ms  CPU %.1f ms",
                          s, gpu_ms, cpu_ms);
            cv::putText(panel, buf, cv::Point(10, W - 12), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(200, 200, 210), 1, cv::LINE_AA);
            frames.push_back(panel);
        }
    }

    Score sg = score(all_gt, all_gpu), sc = score(all_gt, all_cpu), st = score(all_gt, all_thr);
    double agree_pct = 100.0 * agree / std::max(1L, valid);
    std::printf("\nall %d scans (%ld returns):\n", N_SCAN, valid);
    std::printf("concentric-zone model : precision %.4f  recall %.4f  F1 %.4f  (CPU F1 %.4f)\n",
                sg.precision, sg.recall, sg.f1, sc.f1);
    std::printf("height threshold      : precision %.4f  recall %.4f  F1 %.4f\n", st.precision, st.recall, st.f1);
    std::printf("CPU / GPU per scan    : %.2f ms / %.3f ms  (%.0fx)\n", cpu_ms_total / N_SCAN,
                gpu_ms_total / N_SCAN, cpu_ms_total / gpu_ms_total);
    std::printf("CPU / GPU same label  : %.3f%% of returns\n", agree_pct);

    if (!no_video && !frames.empty()) {
        if (ensure_dirs({ "tmp" }) != 0) std::fprintf(stderr, "warning: mkdir tmp failed\n");
        cv::VideoWriter video("tmp/gpu_ground_segmentation.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 2,
                              frames[0].size());
        for (const cv::Mat& f : frames) video.write(f);
        video.release();
        avi_to_gif("tmp/gpu_ground_segmentation.avi", "gif/gpu_ground_segmentation.gif", 2, 800);
        std::printf("wrote gif/gpu_ground_segmentation.gif\n");
    }

    CUDA_CHECK(cudaFree(d_pts)); CUDA_CHECK(cudaFree(d_gt)); CUDA_CHECK(cudaFree(d_bin));
    CUDA_CHECK(cudaFree(d_idx)); CUDA_CHECK(cudaFree(d_start)); CUDA_CHECK(cudaFree(d_lab));
    CUDA_CHECK(cudaFree(d_planes)); CUDA_CHECK(cudaFree(d_sorted_bin));
    bool ok = sg.f1 >= 0.95 && sg.f1 > st.f1 && agree_pct >= 99.9;
    if (check) {
        std::printf("check: %s (model F1 >= 0.95, above the height threshold, CPU/GPU agreement >= 99.9%%)\n",
                    ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    return 0;
}
