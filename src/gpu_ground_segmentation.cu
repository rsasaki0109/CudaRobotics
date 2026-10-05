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
// Second stage: the non-ground points are clustered into objects (Euclidean
// clustering, 0.5 m; on the GPU a point-level and a voxel-level union-find, on
// the CPU a BFS, all three the same partition) and
// the clusters are scored against the ground-truth objects, for the model's
// ground removal, the height threshold's, and none.
//
// Third stage: an oriented box is fitted to each cluster by L-shape fitting
// (search over 90 headings in [0, 90 deg), closeness criterion; one GPU warp =
// one (cluster, heading) pair) and scored against the ground-truth boxes, which
// stand at various headings, next to the axis-aligned box of the same points.
// Boxes of car- and van-like clusters are then completed with a class size prior,
// growing away from the sensor so that the edges it sees stay in place.
//
// Output: gif/gpu_ground_segmentation.gif (bird's-eye view per scan)
//
// --seed N (N > 0) draws a held-out scene: new box positions (within 1 m),
// headings and car / van sizes, and new sensor poses. --obs-csv PATH writes one
// row per box observation (scripts/box_fitting_heldout.py aggregates them).
//
// Options: --no-video, --check (exit non-zero unless the model's F1 >= 0.95,
// it beats the height threshold, CPU and GPU labels agree on >= 99.9%, at least
// 90% of the objects come out as one cluster, the CPU and GPU clusterings
// are the same partition, the L-shape boxes beat the axis-aligned ones in
// heading error and IoU, the CPU and GPU fit the same boxes, and the size prior
// improves the mean IoU and centre error of the L-shape boxes).

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
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

struct Box { float cx, cy, hl, hw, yaw, h; };   // centre, half length / width, heading; on the ground
struct Cyl { float x, y, r, h; };

static const int N_BOX = 7, N_CYL = 8;
__constant__ Box c_box[N_BOX];
__constant__ Cyl c_cyl[N_CYL];

// Ray from o along unit d: nearest hit within MAX_RANGE. label 1 = ground, 0 = object;
// obj is the object's index (boxes, then cylinders), -1 for the ground.
__host__ __device__ static inline bool raycast(const float* o, const float* d,
                                               const Box* boxes, const Cyl* cyls,
                                               float& t_hit, int& label, int& obj) {
    t_hit = MAX_RANGE; label = -1; obj = -1;
    // objects: slabs for boxes in the box frame (bottom at the ground under the box centre)
    for (int b = 0; b < N_BOX; ++b) {
        const Box& B = boxes[b];
        float zb = ground_h(B.cx, B.cy) - 0.5f;
        float cb = cosf(B.yaw), sb = sinf(B.yaw), ex = o[0] - B.cx, ey = o[1] - B.cy;
        float ol[3] = { cb * ex + sb * ey, -sb * ex + cb * ey, o[2] };
        float dl[3] = { cb * d[0] + sb * d[1], -sb * d[0] + cb * d[1], d[2] };
        float lo[3] = { -B.hl, -B.hw, zb }, hi[3] = { B.hl, B.hw, zb + 0.5f + B.h };
        float t0 = 0.0f, t1 = t_hit;
        bool hit = true;
        for (int a = 0; a < 3 && hit; ++a) {
            if (fabsf(dl[a]) < 1e-8f) { if (ol[a] < lo[a] || ol[a] > hi[a]) hit = false; continue; }
            float ta = (lo[a] - ol[a]) / dl[a], tb = (hi[a] - ol[a]) / dl[a];
            if (ta > tb) { float tmp = ta; ta = tb; tb = tmp; }
            t0 = fmaxf(t0, ta); t1 = fminf(t1, tb);
            if (t0 > t1) hit = false;
        }
        if (hit && t0 > 0.0f && t0 < t_hit) { t_hit = t0; label = 0; obj = b; }
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
        t_hit = t; label = 0; obj = N_BOX + c;
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
            t_hit = hi; label = 1; obj = -1;
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
__global__ void scan_kernel(float sx, float sy, unsigned int scan_id, float* pts, int* gt, int* gt_obj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_RAYS) return;
    int ch = i / N_AZ, az = i - ch * N_AZ;
    float el = VERT_MIN + (VERT_MAX - VERT_MIN) * ch / (N_CH - 1);
    float yaw = 2.0f * PI_F * az / N_AZ;
    float o[3] = { sx, sy, ground_h(sx, sy) + SENSOR_H };
    float d[3] = { cosf(el) * cosf(yaw), cosf(el) * sinf(yaw), sinf(el) };
    float t; int label, obj;
    gt_obj[i] = -1;
    if (!raycast(o, d, c_box, c_cyl, t, label, obj)) { gt[i] = -1; return; }
    t += 0.02f * hash_gauss(scan_id, (unsigned int)i);   // 2 cm range noise
    pts[i * 3 + 0] = t * d[0];
    pts[i * 3 + 1] = t * d[1];
    pts[i * 3 + 2] = t * d[2];
    gt[i] = label;
    gt_obj[i] = obj;
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

// ============================ object clustering ============================
// Euclidean clustering of the non-ground points: two points are connected if
// they are within CL_EPS, and clusters are the connected components (PCL's
// EuclideanClusterExtraction). Neighbours come from a 3-D grid of CL_EPS cells.
// On the GPU every point unites itself with each lower-indexed neighbour in a
// lock-free union-find that always hooks the larger root under the smaller one
// (atomicCAS), so each root ends as the smallest point index of its component;
// a final pass flattens every point to its root. That is also the label the
// CPU's BFS assigns, so the two partitions can be compared exactly.
static constexpr float CL_EPS = 0.5f;
static const int CL_MIN = 10;              // smaller clusters are dropped
static const int GX = 240, GY = 240, GZ = 24;   // cells over x, y in [-60, 60], z in [-6, 6]
static const int N_CELL = GX * GY * GZ;

__host__ __device__ static inline int cell_coords(const float* p, int& ix, int& iy, int& iz) {
    ix = (int)floorf((p[0] + 60.0f) / CL_EPS);
    iy = (int)floorf((p[1] + 60.0f) / CL_EPS);
    iz = (int)floorf((p[2] + 6.0f) / CL_EPS);
    if (ix < 0 || iy < 0 || iz < 0 || ix >= GX || iy >= GY || iz >= GZ) return -1;
    return (iz * GY + iy) * GX + ix;
}

// The GPU sorts the active points by cell and looks cells up by binary search, so
// its work scales with the points, not with the 1.4 M cells of the grid.
__global__ void cell_key_kernel(const float* pts, const int* active, int* cell, int* key, int* idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ix, iy, iz;
    int c = active[i] ? cell_coords(&pts[i * 3], ix, iy, iz) : -1;
    cell[i] = c;
    key[i] = c >= 0 ? c : 0x7fffffff;
    idx[i] = i;
}

__device__ static inline int lower_bound_key(const int* key, int n, int c) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (key[mid] < c) lo = mid + 1; else hi = mid;
    }
    return lo;
}

__global__ void cluster_init_kernel(const int* cell, int* label, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) label[i] = cell[i] >= 0 ? i : -1;
}

__device__ static inline int uf_find(const int* parent, int x) {
    for (int p = parent[x]; p != x; p = parent[x]) x = p;
    return x;
}

// Hook the larger root under the smaller one; retry if another thread got there first.
__device__ static inline void uf_unite(int* parent, int a, int b) {
    for (;;) {
        a = uf_find(parent, a);
        b = uf_find(parent, b);
        if (a == b) return;
        if (a > b) { int t = a; a = b; b = t; }
        int old = atomicCAS(&parent[b], b, a);
        if (old == b) return;
        b = old;
    }
}

__global__ void cluster_unite_kernel(const float* pts, const int* cell, const int* skey,
                                     const int* items, int* parent, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || cell[i] < 0) return;
    const float* p = &pts[i * 3];
    int ix, iy, iz;
    cell_coords(p, ix, iy, iz);
    for (int dz = -1; dz <= 1; ++dz) for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
        int x = ix + dx, y = iy + dy, z = iz + dz;
        if (x < 0 || y < 0 || z < 0 || x >= GX || y >= GY || z >= GZ) continue;
        int c = (z * GY + y) * GX + x;
        for (int k = lower_bound_key(skey, n, c); k < n && skey[k] == c; ++k) {
            int j = items[k];
            if (j >= i) continue;   // each pair once
            const float* q = &pts[j * 3];
            float ex = p[0] - q[0], ey = p[1] - q[1], ez = p[2] - q[2];
            if (ex * ex + ey * ey + ez * ez <= CL_EPS * CL_EPS) uf_unite(parent, i, j);
        }
    }
}

__global__ void cluster_flatten_kernel(const int* cell, int* parent, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || cell[i] < 0) return;
    parent[i] = uf_find(parent, i);
}

// ---- voxel clustering: the same partition with far fewer point tests ----
// Two points in one voxel of side CL_EPS / sqrt(3) are always within CL_EPS, so
// all points of a voxel belong to one cluster and the voxel can be the node.
// Two voxels are connected iff some pair of their points is within CL_EPS; a
// voxel pair is tested point by point and the test stops at the first such
// pair. Voxels of points within CL_EPS are at most 2 apart per axis.
static constexpr float VX_S = CL_EPS * 0.57735027f * 0.999f;
static const int VNX = 420, VNY = 420, VNZ = 44;   // x, y in [-60.6, 60.6], z in [-6.3, 6.3]

__host__ __device__ static inline int voxel_key(const float* p) {
    int ix = (int)floorf((p[0] + 60.6f) / VX_S), iy = (int)floorf((p[1] + 60.6f) / VX_S);
    int iz = (int)floorf((p[2] + 6.3f) / VX_S);
    if (ix < 0 || iy < 0 || iz < 0 || ix >= VNX || iy >= VNY || iz >= VNZ) return -1;
    return (iz * VNY + iy) * VNX + ix;
}

__global__ void voxel_key_kernel(const float* pts, const int* active, int* key, int* idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int k = active[i] ? voxel_key(&pts[i * 3]) : -1;
    key[i] = k >= 0 ? k : 0x7fffffff;
    idx[i] = i;
}

// flag[k] = 1 where a new voxel starts in the sorted keys
__global__ void voxel_flag_kernel(const int* key, int* flag, int n_valid) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n_valid) flag[k] = k == 0 || key[k] != key[k - 1];
}

__global__ void voxel_build_kernel(const int* key, const int* flag, const int* vid_excl, int* vid,
                                   int* vstart, int* vkey, int* parent, int* cmin, int n_valid) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_valid) return;
    int v = vid_excl[k] + flag[k] - 1;   // inclusive id
    vid[k] = v;
    if (flag[k]) { vstart[v] = k; vkey[v] = key[k]; parent[v] = v; cmin[v] = 0x7fffffff; }
}

// The 62 neighbour offsets within 2 voxels per axis that lead to a larger key
// (dz > 0, or dz == 0 and dy > 0, or dz == dy == 0 and dx > 0): each voxel pair once.
static const int N_VOFF = 62;
__constant__ int c_voff[N_VOFF * 3];

// Tight bounds of each voxel's points: a voxel pair whose bounds are more than
// CL_EPS apart cannot hold a close pair, and is skipped without point tests.
__global__ void voxel_bounds_kernel(const float* pts, const int* items, const int* vstart, float* vbox, int n_vox) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vox) return;
    float lo[3] = { 1e9f, 1e9f, 1e9f }, hi[3] = { -1e9f, -1e9f, -1e9f };
    for (int a = vstart[v]; a < vstart[v + 1]; ++a)
        for (int c = 0; c < 3; ++c) {
            float x = pts[items[a] * 3 + c];
            lo[c] = fminf(lo[c], x); hi[c] = fmaxf(hi[c], x);
        }
    for (int c = 0; c < 3; ++c) { vbox[v * 6 + c] = lo[c]; vbox[v * 6 + 3 + c] = hi[c]; }
}

// one thread = one (voxel, neighbour offset) pair
__global__ void voxel_unite_kernel(const float* pts, const int* items, const int* vstart, const int* vkey,
                                   const float* vbox, int* parent, int n_vox) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_vox * N_VOFF) return;
    int v = t / N_VOFF, o = t - v * N_VOFF;
    int key = vkey[v];
    int x = key % VNX + c_voff[o * 3 + 0], y = (key / VNX) % VNY + c_voff[o * 3 + 1];
    int z = key / (VNX * VNY) + c_voff[o * 3 + 2];
    if (x < 0 || y < 0 || z < 0 || x >= VNX || y >= VNY || z >= VNZ) return;
    int nk = (z * VNY + y) * VNX + x;
    int u = lower_bound_key(vkey, n_vox, nk);
    if (u >= n_vox || vkey[u] != nk) return;
    float gap2 = 0.0f;
    for (int c = 0; c < 3; ++c) {
        float g = fmaxf(0.0f, fmaxf(vbox[v * 6 + c] - vbox[u * 6 + 3 + c], vbox[u * 6 + c] - vbox[v * 6 + 3 + c]));
        gap2 += g * g;
    }
    if (gap2 > CL_EPS * CL_EPS) return;   // bounds too far apart
    if (uf_find(parent, v) == uf_find(parent, u)) return;   // already connected
    for (int a = vstart[v]; a < vstart[v + 1]; ++a) {
        const float* p = &pts[items[a] * 3];
        for (int b = vstart[u]; b < vstart[u + 1]; ++b) {
            const float* q = &pts[items[b] * 3];
            float ex = p[0] - q[0], ey = p[1] - q[1], ez = p[2] - q[2];
            if (ex * ex + ey * ey + ez * ez <= CL_EPS * CL_EPS) { uf_unite(parent, v, u); return; }
        }
    }
}

// Each component's label is its smallest point index: the first point of each
// voxel is its smallest (stable sort), and the root collects the minimum.
__global__ void voxel_root_min_kernel(const int* items, const int* vstart, int* parent, int* cmin, int n_vox) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vox) return;
    int r = uf_find(parent, v);
    atomicMin(&cmin[r], items[vstart[v]]);
}

__global__ void voxel_label_kernel(const int* items, const int* vid, const int* parent, const int* cmin,
                                   int* label, int n_valid) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_valid) return;
    label[items[k]] = cmin[uf_find(parent, vid[k])];
}

// ---- L-shape box fitting (search-based, closeness criterion) ----
// For heading th the cluster's points are projected on e1 = (cos th, sin th) and
// e2 = (-sin th, cos th). Each point's distance to the nearer edge of the
// bounding rectangle in that frame is clamped below at LS_D0, and the heading's
// score is the sum of the inverse distances: points hugging two perpendicular
// edges (the L a LiDAR sees of a car) score high. Headings repeat every 90 deg,
// so N_TH headings cover [0, 90 deg). On the GPU one warp scores one heading:
// lane l takes points l, l + 32, ... and the 32 partial sums are combined by an
// xor butterfly. The CPU runs the same lanes and the same butterfly, with the
// same heading table, so the two pick the same heading and the same box.
static const int N_TH = 90;
static constexpr float LS_D0 = 0.01f;

struct Obb { float cx, cy, len, wid, yaw; int th; float zlo, zhi; };   // sensor frame; len along the heading

__host__ __device__ static inline void ls_project(const float* pts, int i, float c, float s, float& p1, float& p2) {
    float x = pts[i * 3], y = pts[i * 3 + 1];
    p1 = c * x + s * y; p2 = -s * x + c * y;
}

__host__ __device__ static inline float ls_term(float p1, float p2, const float* lohi) {
    float d = fminf(fminf(p1 - lohi[0], lohi[1] - p1), fminf(p2 - lohi[2], lohi[3] - p2));
    return 1.0f / fmaxf(d, LS_D0);
}

// CPU: the GPU warp's lanes and butterfly, run serially.
static float lshape_score_cpu(const float* pts, const int* items, int a0, int a1, float c, float s) {
    float lohi[4] = { 1e9f, -1e9f, 1e9f, -1e9f }, part[32];
    for (int a = a0; a < a1; ++a) {
        float p1, p2;
        ls_project(pts, items[a], c, s, p1, p2);
        lohi[0] = fminf(lohi[0], p1); lohi[1] = fmaxf(lohi[1], p1);
        lohi[2] = fminf(lohi[2], p2); lohi[3] = fmaxf(lohi[3], p2);
    }
    for (int l = 0; l < 32; ++l) {
        part[l] = 0.0f;
        for (int a = a0 + l; a < a1; a += 32) {
            float p1, p2;
            ls_project(pts, items[a], c, s, p1, p2);
            part[l] += ls_term(p1, p2, lohi);
        }
    }
    for (int off = 16; off >= 1; off >>= 1) {
        float nxt[32];
        for (int l = 0; l < 32; ++l) nxt[l] = part[l] + part[l ^ off];
        for (int l = 0; l < 32; ++l) part[l] = nxt[l];
    }
    return part[0];
}

// The bounding rectangle of the points in the frame of heading k.
__host__ __device__ static inline Obb lshape_rect(const float* pts, const int* items, int a0, int a1,
                                                  const float* cs, int k) {
    float c = cs[k * 2], s = cs[k * 2 + 1];
    float lo1 = 1e9f, hi1 = -1e9f, lo2 = 1e9f, hi2 = -1e9f, zlo = 1e9f, zhi = -1e9f;
    for (int a = a0; a < a1; ++a) {
        float x = pts[items[a] * 3], y = pts[items[a] * 3 + 1], z = pts[items[a] * 3 + 2];
        float p1 = c * x + s * y, p2 = -s * x + c * y;
        lo1 = fminf(lo1, p1); hi1 = fmaxf(hi1, p1); lo2 = fminf(lo2, p2); hi2 = fmaxf(hi2, p2);
        zlo = fminf(zlo, z); zhi = fmaxf(zhi, z);
    }
    float m1 = 0.5f * (lo1 + hi1), m2 = 0.5f * (lo2 + hi2);
    Obb B;
    B.zlo = zlo; B.zhi = zhi;
    B.cx = c * m1 - s * m2; B.cy = s * m1 + c * m2;
    B.len = hi1 - lo1; B.wid = hi2 - lo2;
    B.yaw = k * (0.5f * PI_F / N_TH); B.th = k;
    return B;
}

// The point indices of the clustered points come sorted by cluster label (runs).
__global__ void lshape_key_kernel(const int* label, int* key, int* idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    key[i] = label[i] >= 0 ? label[i] : 0x7fffffff;
    idx[i] = i;
}

// one warp = one (cluster, heading) pair
__global__ void lshape_score_kernel(const float* pts, const int* items, const int* rkey, const int* start,
                                    const int* cnt, const float* cs, float* score, int n_runs) {
    int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (w >= n_runs * N_TH) return;   // whole warps
    int r = w / N_TH, k = w - r * N_TH;
    if (rkey[r] == 0x7fffffff || cnt[r] < CL_MIN) return;
    int a0 = start[r], a1 = a0 + cnt[r];
    float c = cs[k * 2], s = cs[k * 2 + 1];
    float lohi[4] = { 1e9f, -1e9f, 1e9f, -1e9f };
    for (int a = a0 + lane; a < a1; a += 32) {
        float p1, p2;
        ls_project(pts, items[a], c, s, p1, p2);
        lohi[0] = fminf(lohi[0], p1); lohi[1] = fmaxf(lohi[1], p1);
        lohi[2] = fminf(lohi[2], p2); lohi[3] = fmaxf(lohi[3], p2);
    }
    for (int off = 16; off >= 1; off >>= 1) {   // min / max: exact in any order
        lohi[0] = fminf(lohi[0], __shfl_xor_sync(0xffffffffu, lohi[0], off));
        lohi[1] = fmaxf(lohi[1], __shfl_xor_sync(0xffffffffu, lohi[1], off));
        lohi[2] = fminf(lohi[2], __shfl_xor_sync(0xffffffffu, lohi[2], off));
        lohi[3] = fmaxf(lohi[3], __shfl_xor_sync(0xffffffffu, lohi[3], off));
    }
    float part = 0.0f;
    for (int a = a0 + lane; a < a1; a += 32) {
        float p1, p2;
        ls_project(pts, items[a], c, s, p1, p2);
        part += ls_term(p1, p2, lohi);
    }
    for (int off = 16; off >= 1; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
    if (lane == 0) score[w] = part;
}

// one thread = one cluster: best heading (the first on ties), then its rectangle
__global__ void lshape_select_kernel(const float* pts, const int* items, const int* rkey, const int* start,
                                     const int* cnt, const float* cs, const float* score, Obb* obb, int n_runs) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_runs) return;
    if (rkey[r] == 0x7fffffff || cnt[r] < CL_MIN) { obb[r].th = -1; return; }
    int best = 0;
    for (int k = 1; k < N_TH; ++k) if (score[r * N_TH + k] > score[r * N_TH + best]) best = k;
    obb[r] = lshape_rect(pts, items, start[r], start[r] + cnt[r], cs, best);
}

}  // namespace cudabot

using namespace cudabot;

static void heading_table(std::vector<float>& cs) {
    cs.resize(N_TH * 2);
    for (int k = 0; k < N_TH; ++k) {
        double th = k * (0.5 * 3.14159265358979 / N_TH);
        cs[k * 2] = (float)std::cos(th); cs[k * 2 + 1] = (float)std::sin(th);
    }
}

// GPU L-shape fitting of every cluster of >= CL_MIN points of a device label array.
struct GpuLShape {
    int *d_key, *d_idx, *d_rkey, *d_cnt, *d_start;
    float *d_score, *d_cs;
    Obb* d_obb;
    GpuLShape() {
        for (int** p : { &d_key, &d_idx, &d_rkey, &d_cnt, &d_start })
            CUDA_CHECK(cudaMalloc(p, N_RAYS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_score, (size_t)N_RAYS * N_TH * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_obb, N_RAYS * sizeof(Obb)));
        std::vector<float> cs;
        heading_table(cs);
        CUDA_CHECK(cudaMalloc(&d_cs, cs.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_cs, cs.data(), cs.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~GpuLShape() {
        for (int* p : { d_key, d_idx, d_rkey, d_cnt, d_start }) cudaFree(p);
        cudaFree(d_score); cudaFree(d_cs); cudaFree(d_obb);
    }
    // keys[r] = cluster label, obb[r] = its box (th = -1 for runs that are not clusters)
    float run(const float* d_pts, const int* d_label, std::vector<int>& keys, std::vector<Obb>& obb) {
        const int B = 256, G = (N_RAYS + B - 1) / B;
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        lshape_key_kernel<<<G, B>>>(d_label, d_key, d_idx, N_RAYS);
        thrust::device_ptr<int> key(d_key), idx(d_idx), rkey(d_rkey), cnt(d_cnt);
        thrust::stable_sort_by_key(key, key + N_RAYS, idx);
        int n_runs = (int)(thrust::reduce_by_key(key, key + N_RAYS, thrust::constant_iterator<int>(1), rkey, cnt)
                               .first - rkey);
        thrust::exclusive_scan(cnt, cnt + n_runs, thrust::device_ptr<int>(d_start));
        lshape_score_kernel<<<(n_runs * N_TH * 32 + B - 1) / B, B>>>(d_pts, d_idx, d_rkey, d_start, d_cnt, d_cs,
                                                               d_score, n_runs);
        lshape_select_kernel<<<(n_runs + B - 1) / B, B>>>(d_pts, d_idx, d_rkey, d_start, d_cnt, d_cs, d_score,
                                                          d_obb, n_runs);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
        keys.resize(n_runs); obb.resize(n_runs);
        CUDA_CHECK(cudaMemcpy(keys.data(), d_rkey, n_runs * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(obb.data(), d_obb, n_runs * sizeof(Obb), cudaMemcpyDeviceToHost));
        return ms;
    }
};

// GPU voxel clustering: same partition and labels as the point-level clusterers.
struct GpuVoxelClusterer {
    int *d_active, *d_key, *d_items, *d_flag, *d_vexcl, *d_vid, *d_vstart, *d_vkey, *d_parent, *d_cmin, *d_label;
    float* d_vbox;
    GpuVoxelClusterer() {
        CUDA_CHECK(cudaMalloc(&d_vbox, (size_t)N_RAYS * 6 * sizeof(float)));
        for (int** p : { &d_active, &d_key, &d_items, &d_flag, &d_vexcl, &d_vid, &d_vstart, &d_vkey,
                          &d_parent, &d_cmin, &d_label })
            CUDA_CHECK(cudaMalloc(p, (N_RAYS + 1) * sizeof(int)));
        int off[N_VOFF * 3], n = 0;
        for (int dz = -2; dz <= 2; ++dz) for (int dy = -2; dy <= 2; ++dy) for (int dx = -2; dx <= 2; ++dx)
            if (dz > 0 || (dz == 0 && (dy > 0 || (dy == 0 && dx > 0)))) {
                off[n * 3 + 0] = dx; off[n * 3 + 1] = dy; off[n * 3 + 2] = dz; ++n;
            }
        CUDA_CHECK(cudaMemcpyToSymbol(c_voff, off, sizeof(off)));
    }
    ~GpuVoxelClusterer() {
        for (int* p : { d_active, d_key, d_items, d_flag, d_vexcl, d_vid, d_vstart, d_vkey, d_parent, d_cmin, d_label })
            cudaFree(p);
        cudaFree(d_vbox);
    }
    float run(const float* d_pts, const std::vector<int>& active, std::vector<int>& label, int& n_vox_out) {
        const int B = 256, G = (N_RAYS + B - 1) / B;
        int n_valid = 0;
        for (int a : active) n_valid += a;
        CUDA_CHECK(cudaMemcpy(d_active, active.data(), N_RAYS * sizeof(int), cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        voxel_key_kernel<<<G, B>>>(d_pts, d_active, d_key, d_items, N_RAYS);
        thrust::stable_sort_by_key(thrust::device_ptr<int>(d_key), thrust::device_ptr<int>(d_key) + N_RAYS,
                                   thrust::device_ptr<int>(d_items));
        int Gv = (n_valid + B - 1) / B;
        voxel_flag_kernel<<<Gv > 0 ? Gv : 1, B>>>(d_key, d_flag, n_valid);
        thrust::exclusive_scan(thrust::device_ptr<int>(d_flag), thrust::device_ptr<int>(d_flag) + n_valid,
                               thrust::device_ptr<int>(d_vexcl));
        voxel_build_kernel<<<Gv > 0 ? Gv : 1, B>>>(d_key, d_flag, d_vexcl, d_vid, d_vstart, d_vkey, d_parent,
                                                  d_cmin, n_valid);
        int n_vox = 0;
        if (n_valid > 0) {
            int last_excl = 0, last_flag = 0;
            CUDA_CHECK(cudaMemcpy(&last_excl, d_vexcl + n_valid - 1, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_flag, d_flag + n_valid - 1, sizeof(int), cudaMemcpyDeviceToHost));
            n_vox = last_excl + last_flag;
        }
        CUDA_CHECK(cudaMemcpy(d_vstart + n_vox, &n_valid, sizeof(int), cudaMemcpyHostToDevice));
        int Gx = (n_vox + B - 1) / B, Gp = (n_vox * N_VOFF + B - 1) / B;
        CUDA_CHECK(cudaMemset(d_label, 0xff, N_RAYS * sizeof(int)));   // -1
        if (n_vox > 0) {
            voxel_bounds_kernel<<<Gx, B>>>(d_pts, d_items, d_vstart, d_vbox, n_vox);
            voxel_unite_kernel<<<Gp, B>>>(d_pts, d_items, d_vstart, d_vkey, d_vbox, d_parent, n_vox);
            voxel_root_min_kernel<<<Gx, B>>>(d_items, d_vstart, d_parent, d_cmin, n_vox);
            voxel_label_kernel<<<Gv, B>>>(d_items, d_vid, d_parent, d_cmin, d_label, n_valid);
        }
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
        label.resize(N_RAYS);
        CUDA_CHECK(cudaMemcpy(label.data(), d_label, N_RAYS * sizeof(int), cudaMemcpyDeviceToHost));
        n_vox_out = n_vox;
        return ms;
    }
};

// GPU clustering of the points with active[i]; returns the labels (component's
// smallest point index, -1 for inactive) and the kernel time.
struct GpuClusterer {
    int *d_active, *d_cell, *d_key, *d_items, *d_label;
    GpuClusterer() {
        CUDA_CHECK(cudaMalloc(&d_active, N_RAYS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cell, N_RAYS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_key, N_RAYS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_items, N_RAYS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_label, N_RAYS * sizeof(int)));
    }
    ~GpuClusterer() {
        cudaFree(d_active); cudaFree(d_cell); cudaFree(d_key); cudaFree(d_items); cudaFree(d_label);
    }
    float run(const float* d_pts, const std::vector<int>& active, std::vector<int>& label, int& iters) {
        const int B = 256, G = (N_RAYS + B - 1) / B;
        CUDA_CHECK(cudaMemcpy(d_active, active.data(), N_RAYS * sizeof(int), cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        cell_key_kernel<<<G, B>>>(d_pts, d_active, d_cell, d_key, d_items, N_RAYS);
        thrust::sort_by_key(thrust::device_ptr<int>(d_key), thrust::device_ptr<int>(d_key) + N_RAYS,
                            thrust::device_ptr<int>(d_items));
        cluster_init_kernel<<<G, B>>>(d_cell, d_label, N_RAYS);
        cluster_unite_kernel<<<G, B>>>(d_pts, d_cell, d_key, d_items, d_label, N_RAYS);
        cluster_flatten_kernel<<<G, B>>>(d_cell, d_label, N_RAYS);
        iters = 1;
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
        label.resize(N_RAYS);
        CUDA_CHECK(cudaMemcpy(label.data(), d_label, N_RAYS * sizeof(int), cudaMemcpyDeviceToHost));
        return ms;
    }
};

// CPU clustering: the same grid, BFS per component; label = smallest index.
static void cpu_cluster(const std::vector<float>& pts, const std::vector<int>& active, std::vector<int>& label) {
    int n = (int)active.size();
    std::vector<int> cell(n, -1), start(N_CELL + 1, 0), items;
    for (int i = 0; i < n; ++i) {
        int ix, iy, iz;
        if (active[i]) cell[i] = cell_coords(&pts[i * 3], ix, iy, iz);
        if (cell[i] >= 0) start[cell[i] + 1]++;
    }
    for (int c = 0; c < N_CELL; ++c) start[c + 1] += start[c];
    items.assign(start[N_CELL], 0);
    std::vector<int> fill(start.begin(), start.end() - 1);
    for (int i = 0; i < n; ++i) if (cell[i] >= 0) items[fill[cell[i]]++] = i;
    label.assign(n, -1);
    std::vector<int> queue;
    for (int i = 0; i < n; ++i) {
        if (cell[i] < 0 || label[i] >= 0) continue;
        label[i] = i; queue.assign(1, i);
        for (size_t h = 0; h < queue.size(); ++h) {
            int u = queue[h];
            const float* p = &pts[u * 3];
            int ix, iy, iz;
            cell_coords(p, ix, iy, iz);
            for (int dz = -1; dz <= 1; ++dz) for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                int x = ix + dx, y = iy + dy, z = iz + dz;
                if (x < 0 || y < 0 || z < 0 || x >= GX || y >= GY || z >= GZ) continue;
                int c = (z * GY + y) * GX + x;
                for (int k = start[c]; k < start[c + 1]; ++k) {
                    int j = items[k];
                    if (label[j] >= 0) continue;
                    const float* q = &pts[j * 3];
                    float ex = p[0] - q[0], ey = p[1] - q[1], ez = p[2] - q[2];
                    if (ex * ex + ey * ey + ez * ez <= CL_EPS * CL_EPS) { label[j] = i; queue.push_back(j); }
                }
            }
        }
    }
}

// CPU L-shape fitting: clusters in label order, points in index order (as the GPU's stable sort).
static void cpu_lshape(const std::vector<float>& pts, const std::vector<int>& label, const std::vector<float>& cs,
                       std::vector<int>& keys, std::vector<Obb>& obb) {
    std::vector<int> cnt(N_RAYS + 1, 0);
    for (int i = 0; i < N_RAYS; ++i) if (label[i] >= 0) cnt[label[i] + 1]++;
    std::vector<int> start(N_RAYS + 1, 0);
    for (int c = 0; c < N_RAYS; ++c) start[c + 1] = start[c] + cnt[c + 1];
    std::vector<int> items(start[N_RAYS]), fill(start.begin(), start.end() - 1);
    for (int i = 0; i < N_RAYS; ++i) if (label[i] >= 0) items[fill[label[i]]++] = i;
    keys.clear(); obb.clear();
    for (int c = 0; c < N_RAYS; ++c) {
        int a0 = start[c], a1 = start[c + 1];
        if (a1 - a0 < CL_MIN) continue;
        int best = 0;
        float best_s = lshape_score_cpu(pts.data(), items.data(), a0, a1, cs[0], cs[1]);
        for (int k = 1; k < N_TH; ++k) {
            float sc = lshape_score_cpu(pts.data(), items.data(), a0, a1, cs[k * 2], cs[k * 2 + 1]);
            if (sc > best_s) { best_s = sc; best = k; }
        }
        keys.push_back(c);
        obb.push_back(lshape_rect(pts.data(), items.data(), a0, a1, cs.data(), best));
    }
}

// The cluster that holds a ground-truth object (as score_clusters' "found"), or -1.
static int matched_cluster(const std::vector<int>& label, const std::vector<int>& gt_obj, int o) {
    std::vector<int> size(N_RAYS, 0), hit(N_RAYS, 0);
    int total = 0;
    for (int i = 0; i < N_RAYS; ++i) {
        if (label[i] >= 0) size[label[i]]++;
        if (gt_obj[i] == o) { ++total; if (label[i] >= 0) hit[label[i]]++; }
    }
    if (total < 20) return -1;
    int best = -1;
    for (int c = 0; c < N_RAYS; ++c) if (hit[c] > 0 && (best < 0 || hit[c] > hit[best])) best = c;
    if (best < 0 || size[best] < CL_MIN || hit[best] < 0.5 * total || hit[best] < 0.5 * size[best]) return -1;
    return best;
}

// Heading error of a rectangle, modulo 90 deg, in degrees.
static double yaw_err_deg(double a, double b) {
    double e = a - b, q = 0.5 * 3.14159265358979;
    e -= q * std::floor(e / q + 0.5);
    return std::fabs(e) * 180.0 / 3.14159265358979;
}

static double bev_iou(const cv::RotatedRect& a, const cv::RotatedRect& b) {
    std::vector<cv::Point2f> poly;
    double inter = 0.0;
    if (cv::rotatedRectangleIntersection(a, b, poly) != cv::INTERSECT_NONE && poly.size() >= 3) {
        std::vector<cv::Point2f> hull;
        cv::convexHull(poly, hull);
        inter = cv::contourArea(hull);
    }
    double ua = (double)a.size.width * a.size.height + (double)b.size.width * b.size.height - inter;
    return ua > 0 ? inter / ua : 0.0;
}

// ---- size-prior completion (host; one step per cluster) ----
// A LiDAR sees only the near faces of an object, so the L-shape box covers the
// visible part. A class size prior fills in the rest: the class comes from the
// cluster's height and footprint (standing in for a classifier). The longer
// observed side is the length, unless neither side exceeds the class width: then
// the sensor sees an end of the object and its length runs away from the sensor,
// along the axis closer to the line of sight. A side shorter than the prior grows
// away from the sensor (the origin of the sensor frame), keeping the edge the
// sensor sees. A side the sensor stands across grows about its centre.
struct SizePrior { const char* name; float hmin, hmax, len, wid, min_wid; };
static const int N_CLS = 2;
static const SizePrior PRIOR[N_CLS] = {
    { "car", 1.0f, 2.0f, 4.5f, 1.8f, 0.0f },
    { "van", 2.0f, 3.2f, 6.0f, 2.0f, 0.8f },   // min_wid keeps thin walls out
};

static int classify_box(const Obb& B) {
    float h = B.zhi - B.zlo, lo = std::max(B.len, B.wid), sh = std::min(B.len, B.wid);
    for (int k = 0; k < N_CLS; ++k) {
        const SizePrior& P = PRIOR[k];
        if (h >= P.hmin && h < P.hmax && lo >= 1.2f && lo <= 1.2f * P.len && sh <= 1.2f * P.wid && sh >= P.min_wid)
            return k;
    }
    return -1;
}

static Obb complete_box(const Obb& B, int cls, float scale, bool end_rule = true) {
    if (cls < 0) return B;
    float c = std::cos(B.yaw), s = std::sin(B.yaw);
    float m[2] = { c * B.cx + s * B.cy, -s * B.cx + c * B.cy }, ext[2] = { B.len, B.wid };
    bool len_first = B.len >= B.wid;
    if (end_rule && std::max(B.len, B.wid) <= 1.2f * PRIOR[cls].wid)   // end view: the length runs along the line of sight
        len_first = std::fabs(c * B.cx + s * B.cy) >= std::fabs(-s * B.cx + c * B.cy);
    float want[2] = { scale * (len_first ? PRIOR[cls].len : PRIOR[cls].wid),
                      scale * (len_first ? PRIOR[cls].wid : PRIOR[cls].len) };
    for (int a = 0; a < 2; ++a) {
        if (ext[a] >= want[a]) continue;
        float lo = m[a] - 0.5f * ext[a], hi = m[a] + 0.5f * ext[a];
        if (lo > 0.0f) hi = lo + want[a];          // sensor on the low side: the low edge is seen
        else if (hi < 0.0f) lo = hi - want[a];     // sensor on the high side
        else { lo = m[a] - 0.5f * want[a]; hi = m[a] + 0.5f * want[a]; }
        m[a] = 0.5f * (lo + hi); ext[a] = want[a];
    }
    Obb R = B;
    R.cx = c * m[0] - s * m[1]; R.cy = s * m[0] + c * m[1];
    R.len = ext[0]; R.wid = ext[1];
    return R;
}

// Box scores, summed over observations, per box method.
static const int N_BM = 6;
static const char* BM_NAME[N_BM] = { "L-shape", "axis-aligned", "L + prior", "L + prior x0.9", "L + prior x1.1",
                                     "L + prior, no end rule" };
static const char* BM_KEY[N_BM] = { "lshape", "aabb", "prior", "prior_x0.9", "prior_x1.1", "prior_noend" };

// Held-out scene: boxes move within 1 m and take a new heading (the wall stays),
// cars and the van take new sizes, and the sensor takes new poses on the road.
// Footprints are kept apart by their circumscribed circles.
static void randomize_scene(unsigned int seed, Box* box, const Cyl* cyl, float (*poses)[2], int n_pose) {
    std::mt19937 rng(seed);
    auto U = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); };
    auto rad = [](const Box& B) { return std::sqrt(B.hl * B.hl + B.hw * B.hw); };
    for (int b = 0; b < N_BOX; ++b) {
        if (b == 3) continue;
        for (int tries = 0; tries < 1000; ++tries) {
            Box B = box[b];
            B.cx += U(-1.0f, 1.0f); B.cy += U(-1.0f, 1.0f); B.yaw = U(0.0f, PI_F);
            if (b <= 2) { B.hl = 0.5f * U(4.0f, 5.0f); B.hw = 0.5f * U(1.7f, 1.9f); B.h = U(1.4f, 1.7f); }
            if (b == 4) { B.hl = 0.5f * U(5.5f, 6.5f); B.hw = 0.5f * U(1.9f, 2.1f); B.h = U(2.4f, 3.0f); }
            bool ok = true;
            for (int o = 0; o < N_BOX && ok; ++o)
                if (o != b) ok = std::hypot(B.cx - box[o].cx, B.cy - box[o].cy) > rad(B) + rad(box[o]) + 0.3f;
            for (int c = 0; c < N_CYL && ok; ++c)
                ok = std::hypot(B.cx - cyl[c].x, B.cy - cyl[c].y) > rad(B) + cyl[c].r + 0.3f;
            if (ok) { box[b] = B; break; }
        }
    }
    for (int k = 0; k < n_pose; ++k) {
        for (int tries = 0; tries < 1000; ++tries) {
            float x = U(-8.0f, 22.0f), y = U(-3.0f, 5.5f);
            bool ok = true;
            for (int o = 0; o < N_BOX && ok; ++o) ok = std::hypot(x - box[o].cx, y - box[o].cy) > rad(box[o]) + 1.0f;
            for (int c = 0; c < N_CYL && ok; ++c) ok = std::hypot(x - cyl[c].x, y - cyl[c].y) > cyl[c].r + 1.0f;
            if (ok) { poses[k][0] = x; poses[k][1] = y; break; }
        }
    }
}
struct BoxScore { int n; double yaw[N_BM], iou[N_BM], centre[N_BM], len[N_BM], wid[N_BM]; int good[N_BM], cls[N_CLS + 1]; };

// Object-level scores of a clustering against the ground-truth objects.
struct ClusterScore { int objects, found, merged, split, clusters, ground_clusters; };

static ClusterScore score_clusters(const std::vector<int>& label, const std::vector<int>& gt,
                                   const std::vector<int>& gt_obj) {
    const int N_OBJ = N_BOX + N_CYL;
    std::vector<int> size(N_RAYS, 0);
    for (int i = 0; i < N_RAYS; ++i) if (label[i] >= 0) size[label[i]]++;
    // per cluster (kept if >= CL_MIN): point count per object, ground count
    std::vector<int> ids;
    for (int c = 0; c < N_RAYS; ++c) if (size[c] >= CL_MIN) ids.push_back(c);
    std::vector<int> slot(N_RAYS, -1);
    for (size_t k = 0; k < ids.size(); ++k) slot[ids[k]] = (int)k;
    std::vector<std::vector<int>> hist(ids.size(), std::vector<int>(N_OBJ + 1, 0));
    std::vector<int> obj_total(N_OBJ, 0);
    for (int i = 0; i < N_RAYS; ++i) {
        if (gt_obj[i] >= 0) obj_total[gt_obj[i]]++;
        if (label[i] < 0 || slot[label[i]] < 0) continue;
        int o = gt_obj[i] >= 0 ? gt_obj[i] : N_OBJ;   // N_OBJ = ground
        hist[slot[label[i]]][o]++;
    }
    ClusterScore S{0, 0, 0, 0, (int)ids.size(), 0};
    for (size_t k = 0; k < ids.size(); ++k) {
        int tot = 0, objs = 0;
        for (int o = 0; o <= N_OBJ; ++o) tot += hist[k][o];
        for (int o = 0; o < N_OBJ; ++o) if (hist[k][o] >= 0.1 * tot && hist[k][o] >= 5) ++objs;
        if (hist[k][N_OBJ] > 0.5 * tot) ++S.ground_clusters;
        if (objs >= 2) ++S.merged;
    }
    for (int o = 0; o < N_OBJ; ++o) {
        if (obj_total[o] < 20) continue;   // objects the scan barely sees
        ++S.objects;
        int best = -1, best_n = 0, parts = 0;
        for (size_t k = 0; k < ids.size(); ++k) {
            if (hist[k][o] >= 0.1 * obj_total[o]) ++parts;
            if (hist[k][o] > best_n) { best_n = hist[k][o]; best = (int)k; }
        }
        if (parts >= 2) ++S.split;
        if (best < 0) continue;
        int tot = 0;
        for (int q = 0; q <= N_OBJ; ++q) tot += hist[best][q];
        if (best_n >= 0.5 * obj_total[o] && best_n >= 0.5 * tot) ++S.found;
    }
    return S;
}

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
    unsigned int seed = 0;
    const char* obs_csv = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--no-video")) no_video = true;
        else if (!std::strcmp(argv[i], "--check")) check = true;
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned int)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--obs-csv") && i + 1 < argc) obs_csv = argv[++i];
    }
    std::printf("=== GPU LiDAR ground segmentation (CPU vs CUDA) ===\n");

    const float DEG = PI_F / 180.0f;
    Box h_box[N_BOX] = {
        { 8.25f, -2.6f, 2.25f, 0.9f, 20.0f * DEG, 1.5f },    // car
        { 15.25f, 2.9f, 2.25f, 0.9f, -15.0f * DEG, 1.6f },   // car on the ramp
        { -6.75f, -5.1f, 2.25f, 0.9f, 35.0f * DEG, 1.5f },   // car
        { -8.0f, 8.2f, 6.0f, 0.2f, 0.0f, 2.5f },             // wall on the sidewalk
        { 23.0f, -8.0f, 3.0f, 1.0f, 10.0f * DEG, 2.8f },     // van on the ramp
        { -18.0f, -12.0f, 2.0f, 1.5f, 30.0f * DEG, 1.0f },   // low crate
        { 3.0f, 10.0f, 1.0f, 0.5f, 0.0f, 0.6f },             // bench on the sidewalk
    };
    const char* box_name[N_BOX] = { "car", "car on the ramp", "car", "wall", "van on the ramp", "crate", "bench" };
    Cyl h_cyl[N_CYL] = {
        { 4.0f, 6.8f, 0.15f, 4.0f }, { 12.0f, 6.8f, 0.15f, 4.0f }, { -6.0f, 6.8f, 0.15f, 4.0f },
        { 3.0f, -1.0f, 0.3f, 1.7f }, { -2.0f, 3.5f, 0.3f, 1.7f }, { 9.0f, 4.5f, 0.3f, 1.7f },
        { 16.0f, -4.0f, 0.3f, 1.8f }, { -12.0f, -2.0f, 0.3f, 1.7f },
    };
    const int N_SCAN = 8;
    float poses[N_SCAN][2] = { { 0, 0 }, { 4, 2 }, { 8, 0 }, { 12, -2 }, { 16, 0 }, { 2, 5 },
                                     { -5, 2 }, { 20, 3 } };
    if (seed > 0) {
        randomize_scene(seed, h_box, h_cyl, poses, N_SCAN);
        std::printf("held-out scene, seed %u\n", seed);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_box, h_box, sizeof(h_box)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_cyl, h_cyl, sizeof(h_cyl)));
    FILE* obs = obs_csv ? std::fopen(obs_csv, "w") : nullptr;
    if (obs) {
        std::fprintf(obs, "seed,scan,box,faces,cls");
        for (int m = 0; m < N_BM; ++m)
            std::fprintf(obs, ",iou_%s,centre_%s,yaw_%s", BM_KEY[m], BM_KEY[m], BM_KEY[m]);
        std::fprintf(obs, "\n");
    }
    float *d_pts; int *d_gt, *d_gt_obj, *d_bin, *d_idx, *d_start, *d_lab;
    BinPlane* d_planes;
    CUDA_CHECK(cudaMalloc(&d_pts, N_RAYS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gt, N_RAYS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gt_obj, N_RAYS * sizeof(int)));
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
    GpuClusterer clusterer;
    GpuVoxelClusterer vclusterer;
    GpuLShape lshaper;
    std::vector<float> h_cs;
    heading_table(h_cs);
    double ls_gpu_ms = 0.0, ls_cpu_ms = 0.0;
    long ls_fits = 0;
    bool ls_same = true;
    BoxScore bsum[3] = {};   // all observations, one face visible, two faces visible
    std::vector<BoxScore> bobj(N_BOX, BoxScore{});
    double vcl_gpu_ms = 0.0;
    long vox_total = 0;
    bool vcl_same = true;
    ClusterScore cl_sum[3] = { {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0} };
    double cl_cpu_ms = 0.0, cl_gpu_ms = 0.0;
    int cl_iters = 0;
    bool cl_same = true;
    double cpu_ms_total = 0.0, gpu_ms_total = 0.0;
    long agree = 0, valid = 0;
    std::vector<cv::Mat> frames;
    std::vector<Obb> fits, done;
    const int B = 256, G = (N_RAYS + B - 1) / B;
    for (int s = 0; s < N_SCAN; ++s) {
        scan_kernel<<<G, B>>>(poses[s][0], poses[s][1], (unsigned int)s, d_pts, d_gt, d_gt_obj);
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> pts(N_RAYS * 3);
        std::vector<int> gt(N_RAYS);
        CUDA_CHECK(cudaMemcpy(pts.data(), d_pts, pts.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gt.data(), d_gt, gt.size() * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> gt_obj(N_RAYS);
        CUDA_CHECK(cudaMemcpy(gt_obj.data(), d_gt_obj, gt_obj.size() * sizeof(int), cudaMemcpyDeviceToHost));

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

        // ---- object clustering of the non-ground points, for three ground removals ----
        for (int mode = 0; mode < 3; ++mode) {   // 0 model, 1 height threshold, 2 none
            std::vector<int> active(N_RAYS, 0);
            for (int i = 0; i < N_RAYS; ++i)
                active[i] = gt[i] >= 0 && (mode == 2 || (mode == 0 ? lab[i] : thr[i]) == 0);
            std::vector<int> glab;
            int iters = 0;
            // GPU timings are the minimum of 5 runs: the GPU is shared with other work
            float cl_ms = 1e30f;
            for (int rep = 0; rep < (mode == 0 ? 5 : 1); ++rep)
                cl_ms = std::min(cl_ms, clusterer.run(d_pts, active, glab, iters));
            ClusterScore cs = score_clusters(glab, gt, gt_obj);
            cl_sum[mode].objects += cs.objects; cl_sum[mode].found += cs.found;
            cl_sum[mode].merged += cs.merged; cl_sum[mode].split += cs.split;
            cl_sum[mode].clusters += cs.clusters; cl_sum[mode].ground_clusters += cs.ground_clusters;
            if (mode == 0) {
                std::vector<int> clab_obj;
                auto c0 = std::chrono::high_resolution_clock::now();
                cpu_cluster(pts, active, clab_obj);
                auto c1 = std::chrono::high_resolution_clock::now();
                cl_cpu_ms += std::chrono::duration<double, std::milli>(c1 - c0).count();
                cl_gpu_ms += cl_ms;
                cl_iters += iters;
                if (glab != clab_obj) cl_same = false;
                std::vector<int> vlab;
                int n_vox = 0;
                float vms = 1e30f;
                for (int rep = 0; rep < 5; ++rep) vms = std::min(vms, vclusterer.run(d_pts, active, vlab, n_vox));
                vcl_gpu_ms += vms;
                vox_total += n_vox;
                if (vlab != clab_obj) vcl_same = false;

                // ---- L-shape boxes of the clusters ----
                std::vector<int> gkeys, ckeys;
                std::vector<Obb> gobb, cobb;
                float lms = 1e30f;
                for (int rep = 0; rep < 5; ++rep)
                    lms = std::min(lms, lshaper.run(d_pts, vclusterer.d_label, gkeys, gobb));
                ls_gpu_ms += lms;
                auto l0 = std::chrono::high_resolution_clock::now();
                cpu_lshape(pts, clab_obj, h_cs, ckeys, cobb);
                auto l1 = std::chrono::high_resolution_clock::now();
                ls_cpu_ms += std::chrono::duration<double, std::milli>(l1 - l0).count();
                {
                    std::vector<int> gk;
                    std::vector<Obb> go;
                    for (size_t r = 0; r < gkeys.size(); ++r)
                        if (gobb[r].th >= 0) { gk.push_back(gkeys[r]); go.push_back(gobb[r]); }
                    if (gk != ckeys) ls_same = false;
                    for (size_t r = 0; r < go.size() && ls_same; ++r)
                        if (std::memcmp(&go[r], &cobb[r], sizeof(Obb)) != 0) ls_same = false;
                    ls_fits += (long)ckeys.size();
                }
                fits.clear(); done.clear();
                for (const Obb& b : cobb) {
                    fits.push_back(b);
                    int k = classify_box(b);
                    if (k >= 0) done.push_back(complete_box(b, k, 1.0f));
                }
                for (int b = 0; b < N_BOX; ++b) {
                    int c = matched_cluster(clab_obj, gt_obj, b);
                    if (c < 0) continue;
                    size_t r = std::lower_bound(ckeys.begin(), ckeys.end(), c) - ckeys.begin();
                    const Box& G = h_box[b];
                    // the axis-aligned box of the same points is heading 0
                    std::vector<int> items;
                    for (int i = 0; i < N_RAYS; ++i) if (clab_obj[i] == c) items.push_back(i);
                    int cls = classify_box(cobb[r]);
                    Obb fit[N_BM] = { cobb[r], lshape_rect(pts.data(), items.data(), 0, (int)items.size(), h_cs.data(), 0),
                                      complete_box(cobb[r], cls, 1.0f), complete_box(cobb[r], cls, 0.9f),
                                      complete_box(cobb[r], cls, 1.1f), complete_box(cobb[r], cls, 1.0f, false) };
                    // faces of the box the sensor can see: one if it stands within the box's slab along one axis
                    float cb = std::cos(G.yaw), sb = std::sin(G.yaw);
                    float u = cb * (poses[s][0] - G.cx) + sb * (poses[s][1] - G.cy);
                    float v = -sb * (poses[s][0] - G.cx) + cb * (poses[s][1] - G.cy);
                    int faces = (std::fabs(u) > G.hl) + (std::fabs(v) > G.hw);
                    cv::RotatedRect gr(cv::Point2f(G.cx, G.cy), cv::Size2f(2 * G.hl, 2 * G.hw), G.yaw / DEG);
                    double e_iou[N_BM], e_yaw[N_BM], e_ctr[N_BM], e_len[N_BM], e_wid[N_BM];
                    for (int m = 0; m < N_BM; ++m) {
                        const Obb& F = fit[m];
                        float wx = F.cx + poses[s][0], wy = F.cy + poses[s][1];
                        cv::RotatedRect fr(cv::Point2f(wx, wy), cv::Size2f(F.len, F.wid), F.yaw / DEG);
                        e_iou[m] = bev_iou(fr, gr);
                        e_yaw[m] = yaw_err_deg(F.yaw, G.yaw);
                        e_ctr[m] = std::hypot(wx - G.cx, wy - G.cy);
                        e_len[m] = std::fabs(std::max(F.len, F.wid) - 2.0 * std::max(G.hl, G.hw));
                        e_wid[m] = std::fabs(std::min(F.len, F.wid) - 2.0 * std::min(G.hl, G.hw));
                    }
                    for (BoxScore* S : { &bsum[0], &bsum[faces >= 2 ? 2 : 1], &bobj[b] }) {
                        S->n++;
                        S->cls[cls < 0 ? N_CLS : cls]++;
                        for (int m = 0; m < N_BM; ++m) {
                            S->yaw[m] += e_yaw[m]; S->iou[m] += e_iou[m]; S->centre[m] += e_ctr[m];
                            S->len[m] += e_len[m]; S->wid[m] += e_wid[m]; S->good[m] += e_iou[m] >= 0.5;
                        }
                    }
                    if (obs) {
                        std::fprintf(obs, "%u,%d,%d,%d,%d", seed, s, b, faces, cls);
                        for (int m = 0; m < N_BM; ++m) std::fprintf(obs, ",%.5f,%.5f,%.4f", e_iou[m], e_ctr[m], e_yaw[m]);
                        std::fprintf(obs, "\n");
                    }
                }
            }
        }
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
                if (view == 1) {
                    auto draw = [&](float cx, float cy, float len, float wid, float yaw, cv::Scalar col) {
                        float c = std::cos(yaw), sn = std::sin(yaw);
                        cv::Point q[4];
                        const float sg[4][2] = { { 1, 1 }, { -1, 1 }, { -1, -1 }, { 1, -1 } };
                        for (int k = 0; k < 4; ++k) {
                            float a = 0.5f * len * sg[k][0], b = 0.5f * wid * sg[k][1];
                            q[k] = px(cx + c * a - sn * b, cy + sn * a + c * b, ox);
                        }
                        for (int k = 0; k < 4; ++k) cv::line(panel, q[k], q[(k + 1) % 4], col, 1, cv::LINE_AA);
                    };
                    for (int b = 0; b < N_BOX; ++b)
                        draw(h_box[b].cx - poses[s][0], h_box[b].cy - poses[s][1], 2 * h_box[b].hl, 2 * h_box[b].hw,
                             h_box[b].yaw, cv::Scalar(255, 170, 60));
                    for (const Obb& F : fits) draw(F.cx, F.cy, F.len, F.wid, F.yaw, cv::Scalar(230, 60, 230));
                    for (const Obb& F : done) draw(F.cx, F.cy, F.len, F.wid, F.yaw, cv::Scalar(40, 160, 255));
                }
                Score sc = view == 0 ? st : sm;
                char buf[160];
                std::snprintf(buf, sizeof(buf), "%s   F1 %.3f",
                              view == 0 ? "height threshold" : "concentric-zone model", sc.f1);
                cv::putText(panel, buf, cv::Point(ox + 10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(235, 235, 245), 1, cv::LINE_AA);
            }
            char buf[200];
            std::snprintf(buf, sizeof(buf), "scan %d   red: object called ground   yellow: ground missed   "
                          "blue: true box   magenta: L-shape   orange: + size prior", s);
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

    CUDA_CHECK(cudaFree(d_pts)); CUDA_CHECK(cudaFree(d_gt)); CUDA_CHECK(cudaFree(d_gt_obj)); CUDA_CHECK(cudaFree(d_bin));
    CUDA_CHECK(cudaFree(d_idx)); CUDA_CHECK(cudaFree(d_start)); CUDA_CHECK(cudaFree(d_lab));
    CUDA_CHECK(cudaFree(d_planes)); CUDA_CHECK(cudaFree(d_sorted_bin));
    std::printf("\n--- object clustering of the non-ground points (Euclidean, %.1f m, >= %d points) ---\n",
                CL_EPS, CL_MIN);
    const char* mode_name[3] = { "concentric-zone model", "height threshold", "no ground removal" };
    for (int m = 0; m < 3; ++m)
        std::printf("%-22s: objects found %3d / %3d   merged clusters %3d   split objects %3d   "
                    "ground clusters %4d   clusters %5d\n", mode_name[m], cl_sum[m].found, cl_sum[m].objects,
                    cl_sum[m].merged, cl_sum[m].split, cl_sum[m].ground_clusters, cl_sum[m].clusters);
    std::printf("clustering per scan: CPU BFS %.2f ms, GPU point union-find %.3f ms, "
                "GPU voxel union-find %.3f ms (%ld voxels per scan); same partition as the CPU: %s / %s\n",
                cl_cpu_ms / N_SCAN, cl_gpu_ms / N_SCAN, vcl_gpu_ms / N_SCAN, vox_total / N_SCAN,
                cl_same ? "yes" : "no", vcl_same ? "yes" : "no");
    std::printf("\n--- oriented boxes of the clusters (L-shape fitting, %d headings, closeness criterion) ---\n", N_TH);
    std::printf("fits per scan %ld; CPU %.2f ms, GPU %.3f ms per scan; CPU and GPU boxes identical: %s\n",
                ls_fits / N_SCAN, ls_cpu_ms / N_SCAN, ls_gpu_ms / N_SCAN, ls_same ? "yes" : "no");
    auto box_line = [](const char* name, const BoxScore& S) {
        if (!S.n) return;
        std::printf("%s: n %d, classed car %d / van %d / none %d\n", name, S.n, S.cls[0], S.cls[1], S.cls[N_CLS]);
        for (int m = 0; m < N_BM; ++m)
            std::printf("  %-15s heading err %5.2f deg  IoU %.3f  (>= 0.5: %3d)  centre err %.2f m  "
                        "long side err %.2f m  short side err %.2f m\n", BM_NAME[m],
                        S.yaw[m] / S.n, S.iou[m] / S.n, S.good[m], S.centre[m] / S.n, S.len[m] / S.n, S.wid[m] / S.n);
    };
    box_line("all box observations", bsum[0]);
    box_line("one face visible", bsum[1]);
    box_line("two faces visible", bsum[2]);
    for (int b = 0; b < N_BOX; ++b) box_line(box_name[b], bobj[b]);
    if (obs) std::fclose(obs);
    bool ls_better = bsum[0].n > 0 && bsum[0].yaw[0] < bsum[0].yaw[1] && bsum[0].iou[0] > bsum[0].iou[1];
    bool prior_better = bsum[0].iou[2] > bsum[0].iou[0] && bsum[0].centre[2] < bsum[0].centre[0];
    double found_rate = cl_sum[0].objects ? (double)cl_sum[0].found / cl_sum[0].objects : 0.0;
    bool ok = sg.f1 >= 0.95 && sg.f1 > st.f1 && agree_pct >= 99.9 && found_rate >= 0.9 && cl_same && vcl_same &&
              ls_better && ls_same && prior_better;
    if (check) {
        std::printf("check: %s (model F1 >= 0.95, above the height threshold, CPU/GPU agreement >= 99.9%%, "
                    ">= 90%% of objects found as one cluster, identical CPU/GPU partition, L-shape boxes beat "
                    "axis-aligned ones, identical CPU/GPU boxes, size prior improves IoU and centre error)\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    return 0;
}
