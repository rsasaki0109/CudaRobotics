// gpu_ground_segmentation.cu
//
// GPU LiDAR ground segmentation (CPU vs CUDA comparison).
//
// The algorithms are in include/cudarobotics/lidar_objects_core.cuh, shared with
// the library cudarobotics::LidarObjectPipeline (lidar_objects_gpu.hpp). This
// demo builds the synthetic scenes, runs the CPU references and the evaluation,
// and checks that the library gives exactly its ground labels, clusters, boxes
// and tracks.
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
// row per box observation (scripts/box_fitting_heldout.py aggregates them), and
// --cls-csv PATH one row of classifier features per cluster
// (scripts/train_box_classifier.py trains the learned class on them).
//
// --sequence drives the sensor along the road (1 m apart, 10 m/s at 10 Hz) and
// tracks the car / van clusters across the scans: each track accumulates the
// world points of its clusters and refits its box, so faces seen from earlier
// poses stay in it (--trk-hits K: a voxel counts once seen in K scans, default
// 3). Output: gif/gpu_ground_segmentation_track.gif.
//
// --moving (with --sequence) adds traffic: a lead car and a following car drive
// in the sensor's lane. A second tracker then runs next to the static one: a
// constant-velocity Kalman filter per track, with the points accumulated in the
// object's frame (moved to the current time by the estimated velocity).
// The hybrid box takes the single-scan box for the tracks the stand-still test
// calls moving and the motion tracker's box for the others.
// Output: gif/gpu_ground_segmentation_motion.gif.
//
// Options: --no-video, --check (exit non-zero unless the model's F1 >= 0.95,
// it beats the height threshold, CPU and GPU labels agree on >= 99.9%, at least
// 90% of the objects come out as one cluster, the CPU and GPU clusterings
// are the same partition, the L-shape boxes beat the axis-aligned ones in
// heading error and IoU, the CPU and GPU fit the same boxes, and the size prior
// improves the mean IoU and centre error of the L-shape boxes, with the height
// rule's class and with the learned class).

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"
#include "cudarobotics/lidar_objects_core.cuh"
#include "cudarobotics/lidar_objects_gpu.hpp"

namespace cudabot {

// ---- sensor ----
static const int N_CH = 64, N_AZ = 1024, N_RAYS = N_CH * N_AZ;   // VERT_MIN .. VERT_MAX, SENSOR_H: the core
static constexpr float MAX_RANGE = 60.0f;

// ---- scene (world frame, z up) ----
__host__ __device__ static inline float ground_h(float x, float y) {
    float h = 0.03f * sinf(0.35f * x) * cosf(0.27f * y);       // gentle undulation
    if (x > 10.0f) h += (x - 10.0f) * 0.105f;                   // 6 deg ramp
    if (y > 6.0f) h += 0.15f;                                   // curb + sidewalk
    return h;
}

struct Box { float cx, cy, hl, hw, yaw, h; };   // centre, half length / width, heading; on the ground
struct Cyl { float x, y, r, h; };

static const int N_BOX = 9, N_CYL = 8;   // boxes 7, 8: traffic (--moving), out of range otherwise
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

// ============================ object clustering (point-level reference) ============================
// Euclidean clustering of the non-ground points: two points are connected if
// they are within CL_EPS, and clusters are the connected components (PCL's
// EuclideanClusterExtraction). Neighbours come from a 3-D grid of CL_EPS cells.
// On the GPU every point unites itself with each lower-indexed neighbour in a
// lock-free union-find that always hooks the larger root under the smaller one
// (atomicCAS), so each root ends as the smallest point index of its component;
// a final pass flattens every point to its root. That is also the label the
// CPU's BFS assigns, so the two partitions can be compared exactly.
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


__global__ void cluster_init_kernel(const int* cell, int* label, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) label[i] = cell[i] >= 0 ? i : -1;
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

}  // namespace cudabot

using namespace cudabot;

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

// Box scores, summed over observations, per box method.
static const int N_BM = 12;
static const char* BM_NAME[N_BM] = { "L-shape", "axis-aligned", "L + prior", "L + prior x0.9", "L + prior x1.1",
                                     "L + prior, no end rule", "L + prior, learned class", "tracked L-shape",
                                     "tracked L + prior", "motion-tracked L-shape", "motion-tracked L + prior",
                                     "hybrid L + prior" };
static const char* BM_KEY[N_BM] = { "lshape", "aabb", "prior", "prior_x0.9", "prior_x1.1", "prior_noend", "prior_mlp",
                                    "trk_lshape", "trk_prior", "mtrk_lshape", "mtrk_prior", "hybrid" };

// Held-out scene: boxes move within 1 m and take a new heading (the wall stays),
// cars and the van take new sizes, and the sensor takes new poses on the road.
// Footprints are kept apart by their circumscribed circles.
// With a sensor path (--sequence), boxes also stay 1 m clear of the line y = PATH_Y.
static constexpr float PATH_Y = 0.5f, PATH_X0 = -12.0f, PATH_X1 = 24.0f;

static void randomize_scene(unsigned int seed, Box* box, const Cyl* cyl, std::vector<std::array<float, 2>>& poses,
                            bool path) {
    std::mt19937 rng(seed);
    auto U = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); };
    auto rad = [](const Box& B) { return std::sqrt(B.hl * B.hl + B.hw * B.hw); };
    for (int b = 0; b < N_BOX; ++b) {
        if (b == 3 || b >= 7) continue;   // the wall stays; traffic is set by --moving
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
            if (ok && path) {
                float c = std::cos(B.yaw), sn = std::sin(B.yaw);
                bool above = true, below = true;
                for (int k = 0; k < 4; ++k) {
                    float a = (k & 1 ? 1.0f : -1.0f) * B.hl, w = (k & 2 ? 1.0f : -1.0f) * B.hw;
                    float y = B.cy + sn * a + c * w;
                    above = above && y > PATH_Y + 1.0f; below = below && y < PATH_Y - 1.0f;
                }
                ok = above || below;
            }
            if (ok) { box[b] = B; break; }
        }
    }
    if (path) return;   // the path's poses are fixed
    for (size_t k = 0; k < poses.size(); ++k) {
        for (int tries = 0; tries < 1000; ++tries) {
            float x = U(-8.0f, 22.0f), y = U(-3.0f, 5.5f);
            bool ok = true;
            for (int o = 0; o < N_BOX && ok; ++o) ok = std::hypot(x - box[o].cx, y - box[o].cy) > rad(box[o]) + 1.0f;
            for (int c = 0; c < N_CYL && ok; ++c) ok = std::hypot(x - cyl[c].x, y - cyl[c].y) > cyl[c].r + 1.0f;
            if (ok) { poses[k][0] = x; poses[k][1] = y; break; }
        }
    }
}
struct BoxScore { int n; double yaw[N_BM], iou[N_BM], centre[N_BM], len[N_BM], wid[N_BM]; int good[N_BM], cls[N_CLS + 1], cls_mlp[N_CLS + 1]; };

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
    const char* cls_csv = nullptr;
    bool sequence = false, moving = false, trk_cpu_fit = false;
    int trk_hits = 3;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--no-video")) no_video = true;
        else if (!std::strcmp(argv[i], "--check")) check = true;
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned int)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--obs-csv") && i + 1 < argc) obs_csv = argv[++i];
        else if (!std::strcmp(argv[i], "--cls-csv") && i + 1 < argc) cls_csv = argv[++i];
        else if (!std::strcmp(argv[i], "--sequence")) sequence = true;
        else if (!std::strcmp(argv[i], "--trk-cpu-fit")) trk_cpu_fit = true;
        else if (!std::strcmp(argv[i], "--moving")) moving = sequence = true;
        else if (!std::strcmp(argv[i], "--trk-hits") && i + 1 < argc) trk_hits = std::max(1, std::atoi(argv[++i]));
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
        { 1000.0f, 0.0f, 2.25f, 0.9f, 0.0f, 1.5f },          // lead car (--moving)
        { 1000.0f, 10.0f, 2.25f, 0.9f, 0.0f, 1.5f },         // following car (--moving)
    };
    const char* box_name[N_BOX] = { "car", "car on the ramp", "car", "wall", "van on the ramp", "crate", "bench",
                                    "lead car", "following car" };
    float box_speed[N_BOX] = {};   // along +x (traffic only)
    Cyl h_cyl[N_CYL] = {
        { 4.0f, 6.8f, 0.15f, 4.0f }, { 12.0f, 6.8f, 0.15f, 4.0f }, { -6.0f, 6.8f, 0.15f, 4.0f },
        { 3.0f, -1.0f, 0.3f, 1.7f }, { -2.0f, 3.5f, 0.3f, 1.7f }, { 9.0f, 4.5f, 0.3f, 1.7f },
        { 16.0f, -4.0f, 0.3f, 1.8f }, { -12.0f, -2.0f, 0.3f, 1.7f },
    };
    std::vector<std::array<float, 2>> poses = { { { 0, 0 } }, { { 4, 2 } }, { { 8, 0 } }, { { 12, -2 } },
                                                { { 16, 0 } }, { { 2, 5 } }, { { -5, 2 } }, { { 20, 3 } } };
    if (sequence) {   // along the road, 1 m apart
        poses.clear();
        for (float x = PATH_X0; x <= PATH_X1 + 1e-3f; x += 1.0f) poses.push_back({ { x, PATH_Y } });
    }
    const int N_SCAN = (int)poses.size();
    if (seed > 0) {
        randomize_scene(seed, h_box, h_cyl, poses, sequence);
        std::printf("held-out scene, seed %u\n", seed);
    }
    if (moving) {   // traffic in the sensor's lane, 12 m ahead and 12 m behind at the start
        h_box[7].cx = PATH_X0 + 12.0f; h_box[7].cy = PATH_Y; box_speed[7] = 11.5f;
        h_box[8].cx = PATH_X0 - 12.0f; h_box[8].cy = PATH_Y; box_speed[8] = 9.0f;
        if (seed > 0) {
            std::mt19937 rng(seed * 7919u + 17u);
            auto U = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); };
            for (int b = 7; b <= 8; ++b) {
                h_box[b].hl = 0.5f * U(4.0f, 5.0f); h_box[b].hw = 0.5f * U(1.7f, 1.9f); h_box[b].h = U(1.4f, 1.7f);
            }
            box_speed[7] = U(8.0f, 13.0f); box_speed[8] = U(7.0f, 12.0f);
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_box, h_box, sizeof(h_box)));
    Box h_box_t[N_BOX];   // the boxes at the current scan's time
    CUDA_CHECK(cudaMemcpyToSymbol(c_cyl, h_cyl, sizeof(h_cyl)));
    FILE* obs = obs_csv ? std::fopen(obs_csv, "w") : nullptr;
    if (obs) {
        std::fprintf(obs, "seed,scan,box,faces,cls,cls_mlp,track,mtrack,speed,verr_static,verr_motion");
        for (int m = 0; m < N_BM; ++m)
            std::fprintf(obs, ",iou_%s,centre_%s,yaw_%s", BM_KEY[m], BM_KEY[m], BM_KEY[m]);
        std::fprintf(obs, "\n");
    }
    FILE* clsf = cls_csv ? std::fopen(cls_csv, "w") : nullptr;
    if (clsf) {
        std::fprintf(clsf, "seed,scan,label,rule,mlp");
        for (int i = 0; i < N_FEAT; ++i) std::fprintf(clsf, ",%s", FEAT_NAME[i]);
        std::fprintf(clsf, "\n");
    }
    float *d_pts; int *d_gt, *d_gt_obj;
    GpuGroundSegmenter segmenter(N_RAYS);
    CUDA_CHECK(cudaMalloc(&d_pts, N_RAYS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gt, N_RAYS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gt_obj, N_RAYS * sizeof(int)));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    std::vector<int> all_gt, all_gpu, all_cpu, all_thr;
    GpuClusterer clusterer;
    GpuVoxelClusterer vclusterer(N_RAYS);
    GpuLShape lshaper(N_RAYS);
    std::vector<float> h_cs;
    heading_table(h_cs);
    double ls_gpu_ms = 0.0, ls_cpu_ms = 0.0;
    // per-scan stage times of the pipeline (--sequence): 0 segmentation (GPU), 1 voxel clustering (GPU),
    // 2 L-shape (GPU), 3 classification + cluster lists (host), 4 static tracker, 5 motion tracker (host)
    // 6 static tracker's box fits, 7 motion tracker's box fits (GPU unless --trk-cpu-fit)
    std::vector<std::array<double, 8>> stage_ms;
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
    std::vector<Obb> fits, done, trk_draw;
    Tracker trk[2];   // static, motion
    GpuBoxFitter box_fitter;
    bool trk_fit_same = true;
    cudarobotics::LidarObjectsConfig lib_cfg;
    lib_cfg.max_points = N_RAYS;
    lib_cfg.track_min_scans = trk_hits;
    cudarobotics::LidarObjectPipeline lib(lib_cfg);   // the library: must give exactly the results below
    bool lib_same = true;
    trk[1].motion = true;
    for (Tracker& T : trk) T.trk_hits = trk_hits;
    BoxScore bmov = {}, bpark = {};   // vehicle observations: moving traffic, parked cars and the van
    double verr_sum[2][2] = {};       // [tracker][moving]
    int verr_n[2] = {};
    const int B = 256, G = (N_RAYS + B - 1) / B;
    for (int s = 0; s < N_SCAN; ++s) {
        const float t_scan = s * SCAN_DT;
        for (int b = 0; b < N_BOX; ++b) { h_box_t[b] = h_box[b]; h_box_t[b].cx += box_speed[b] * t_scan; }
        if (moving) CUDA_CHECK(cudaMemcpyToSymbol(c_box, h_box_t, sizeof(h_box_t)));
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
            segmenter.run(d_pts, d_gt, N_RAYS);
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
        }
        CUDA_CHECK(cudaGetLastError());
        float gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));
        std::vector<int> lab(N_RAYS);
        CUDA_CHECK(cudaMemcpy(lab.data(), segmenter.d_lab, lab.size() * sizeof(int), cudaMemcpyDeviceToHost));

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
        double vms_scan = 0.0;
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
                vms_scan = vms;
                vox_total += n_vox;
                if (vlab != clab_obj) vcl_same = false;

                // ---- L-shape boxes of the clusters ----
                std::vector<int> gkeys, ckeys;
                std::vector<Obb> gobb, cobb;
                float lms = 1e30f;
                for (int rep = 0; rep < 5; ++rep)
                    lms = std::min(lms, lshaper.run(d_pts, vclusterer.d_label, N_RAYS, gkeys, gobb));
                ls_gpu_ms += lms;
                auto l0 = std::chrono::high_resolution_clock::now();
                cpu_lshape(pts, clab_obj, h_cs, ckeys, cobb);
                if (clsf) {   // class of each cluster: car / van if it is a car's / the van's cluster, else none
                    std::vector<int> cl_of(ckeys.size(), 2);
                    for (int b : { 0, 1, 2, 4, 7, 8 }) {
                        int c = matched_cluster(clab_obj, gt_obj, b);
                        size_t r = std::lower_bound(ckeys.begin(), ckeys.end(), c) - ckeys.begin();
                        if (c >= 0 && r < ckeys.size() && ckeys[r] == c) cl_of[r] = b == 4 ? 1 : 0;
                    }
                    for (size_t r = 0; r < ckeys.size(); ++r) {
                        float f[N_FEAT];
                        box_features(cobb[r], f);
                        int rule = classify_box(cobb[r]), mlp = classify_box_mlp(cobb[r]);
                        std::fprintf(clsf, "%u,%d,%d,%d,%d", seed, s, cl_of[r], rule < 0 ? 2 : rule, mlp < 0 ? 2 : mlp);
                        for (int i = 0; i < N_FEAT; ++i) std::fprintf(clsf, ",%.5f", f[i]);
                        std::fprintf(clsf, "\n");
                    }
                }
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
                fits.clear(); done.clear(); trk_draw.clear();
                for (const Obb& b : cobb) {
                    fits.push_back(b);
                    int k = classify_box_mlp(b);
                    if (k >= 0) done.push_back(complete_box(b, k, 1.0f));
                }
                for (Tracker& T : trk) T.track_of.assign(N_RAYS, -1);
                if (sequence) {
                    auto h0 = std::chrono::high_resolution_clock::now();
                    const float px = poses[s][0], py = poses[s][1], pz = ground_h(px, py) + SENSOR_H;
                    std::vector<int> cand, ccls(ckeys.size(), -1), slot(N_RAYS, -1);
                    for (size_t r = 0; r < ckeys.size(); ++r) {
                        ccls[r] = classify_box_mlp(cobb[r]);
                        if (ccls[r] >= 0) { slot[ckeys[r]] = (int)cand.size(); cand.push_back((int)r); }
                    }
                    std::vector<std::vector<int>> citems(cand.size());
                    for (int i = 0; i < N_RAYS; ++i)
                        if (clab_obj[i] >= 0 && slot[clab_obj[i]] >= 0) citems[slot[clab_obj[i]]].push_back(i);
                    auto h1 = std::chrono::high_resolution_clock::now();
                    double tk_ms[2], fit_ms[2];
                    for (int k = 0; k < 2; ++k) {
                        auto u0 = std::chrono::high_resolution_clock::now();
                        trk[k].update(t_scan, px, py, pz, s, cand, ccls, citems, ckeys, cobb, pts, h_cs);
                        tk_ms[k] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - u0).count();
                        fit_ms[k] = trk[k].fit(trk_cpu_fit ? nullptr : &box_fitter, h_cs, check, trk_fit_same);
                    }
                    stage_ms.push_back({ { gpu_ms, vms_scan, lms, std::chrono::duration<double, std::milli>(h1 - h0).count(),
                                           tk_ms[0], tk_ms[1], fit_ms[0], fit_ms[1] } });
                    const Tracker& D = trk[moving ? 1 : 0];   // drawn
                    for (size_t k = 0; k < D.tracks.size(); ++k) {
                        if (std::find(D.track_of.begin(), D.track_of.end(), (int)k) == D.track_of.end()) continue;
                        Obb raw, dn;
                        D.boxes((int)k, t_scan, px, py, raw, dn);
                        trk_draw.push_back(dn);
                    }
                }
                // ---- the library on this scan's returns: the same ground, clusters, boxes and tracks ----
                {
                    std::vector<float> xyz;
                    std::vector<int> orig;   // library point -> ray
                    for (int i = 0; i < N_RAYS; ++i)
                        if (gt[i] >= 0) { orig.push_back(i); xyz.insert(xyz.end(), &pts[i * 3], &pts[i * 3] + 3); }
                    const float px = poses[s][0], py = poses[s][1];
                    cudarobotics::LidarObjectsResult L =
                        lib.process(xyz.data(), orig.size(), px, py, ground_h(px, py) + SENSOR_H, t_scan);
                    bool same = true;
                    for (size_t j = 0; j < orig.size() && same; ++j) {   // labels are smallest indices: map them
                        int i = orig[j];
                        same = L.ground[j] == (lab[i] != 0) &&
                               (L.cluster[j] < 0 ? clab_obj[i] < 0 : orig[L.cluster[j]] == clab_obj[i]);
                    }
                    same = same && L.clusters.size() == ckeys.size();
                    for (size_t r = 0; r < ckeys.size() && same; ++r) {
                        const cudarobotics::LidarCluster& C = L.clusters[r];
                        const Obb& O = cobb[r];
                        same = orig[C.label] == ckeys[r] && C.box.cx == O.cx && C.box.cy == O.cy &&
                               C.box.length == O.len && C.box.width == O.wid && C.box.yaw == O.yaw &&
                               C.box.z_min == O.zlo && C.box.z_max == O.zhi && C.box.points == O.n &&
                               (int)C.cls == classify_box_mlp(O);
                    }
                    if (sequence) {   // tracks: the hybrid boxes of the motion tracker above
                        size_t m = 0;
                        for (size_t r = 0; r < ckeys.size() && same; ++r) {
                            int j = trk[1].track_of[ckeys[r]];
                            if (j < 0) continue;
                            if (m >= L.tracks.size()) { same = false; break; }
                            const cudarobotics::LidarTrack& T = L.tracks[m++];
                            const MotionState& M = trk[1].ms[j];
                            Obb raw, tracked;
                            trk[1].boxes(j, t_scan, px, py, raw, tracked);
                            Obb H = M.moving ? complete_box(cobb[r], classify_box_mlp(cobb[r]), 1.0f) : tracked;
                            same = T.id == j && T.moving == M.moving && T.vx == (float)M.x[2] && T.vy == (float)M.x[3] &&
                                   T.box.cx == H.cx && T.box.cy == H.cy && T.box.length == H.len &&
                                   T.box.width == H.wid && T.box.yaw == H.yaw;
                        }
                        same = same && m == L.tracks.size();
                    }
                    if (!same) lib_same = false;
                }
                for (int b = 0; b < N_BOX; ++b) {
                    int c = matched_cluster(clab_obj, gt_obj, b);
                    if (c < 0) continue;
                    size_t r = std::lower_bound(ckeys.begin(), ckeys.end(), c) - ckeys.begin();
                    const Box& G = h_box_t[b];
                    // the axis-aligned box of the same points is heading 0
                    std::vector<int> items;
                    for (int i = 0; i < N_RAYS; ++i) if (clab_obj[i] == c) items.push_back(i);
                    int cls = classify_box(cobb[r]), cls_mlp = classify_box_mlp(cobb[r]);
                    Obb fit[N_BM] = { cobb[r], lshape_rect(pts.data(), items.data(), 0, (int)items.size(), h_cs.data(), 0),
                                      complete_box(cobb[r], cls, 1.0f), complete_box(cobb[r], cls, 0.9f),
                                      complete_box(cobb[r], cls, 1.1f), complete_box(cobb[r], cls, 1.0f, false),
                                      complete_box(cobb[r], cls_mlp, 1.0f), cobb[r], complete_box(cobb[r], cls_mlp, 1.0f),
                                      cobb[r], complete_box(cobb[r], cls_mlp, 1.0f), complete_box(cobb[r], cls_mlp, 1.0f) };
                    int tid = trk[0].track_of[c], mtid = trk[1].track_of[c];
                    if (tid >= 0) {
                        trk[0].boxes(tid, t_scan, poses[s][0], poses[s][1], fit[7], fit[8]);
                        trk[0].observe(b, tid);
                    }
                    if (mtid >= 0) {
                        trk[1].boxes(mtid, t_scan, poses[s][0], poses[s][1], fit[9], fit[10]);
                        trk[1].observe(b, mtid);
                        // hybrid: a track the stand-still test calls moving gets the single-scan box (lane
                        // traffic shows no new faces, and the track lags by its velocity error); others the track's
                        if (!trk[1].ms[mtid].moving) fit[11] = fit[10];
                    }
                    // velocity errors (the static tracker's velocity is 0)
                    double verr[2] = { -1.0, -1.0 };
                    if (tid >= 0) verr[0] = box_speed[b];
                    if (mtid >= 0) verr[1] = std::hypot(trk[1].ms[mtid].x[2] - box_speed[b], trk[1].ms[mtid].x[3]);
                    bool vehicle = b <= 2 || b == 4 || b >= 7;
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
                    std::vector<BoxScore*> into = { &bsum[0], &bsum[faces >= 2 ? 2 : 1], &bobj[b] };
                    if (vehicle) into.push_back(box_speed[b] > 0.0f ? &bmov : &bpark);
                    if (vehicle && tid >= 0 && mtid >= 0) {
                        for (int k = 0; k < 2; ++k) verr_sum[k][box_speed[b] > 0.0f] += verr[k];
                        verr_n[box_speed[b] > 0.0f]++;
                    }
                    for (BoxScore* S : into) {
                        S->n++;
                        S->cls[cls < 0 ? N_CLS : cls]++;
                        S->cls_mlp[cls_mlp < 0 ? N_CLS : cls_mlp]++;
                        for (int m = 0; m < N_BM; ++m) {
                            S->yaw[m] += e_yaw[m]; S->iou[m] += e_iou[m]; S->centre[m] += e_ctr[m];
                            S->len[m] += e_len[m]; S->wid[m] += e_wid[m]; S->good[m] += e_iou[m] >= 0.5;
                        }
                    }
                    if (obs) {
                        std::fprintf(obs, "%u,%d,%d,%d,%d,%d,%d,%d,%.3f,%.4f,%.4f", seed, s, b, faces, cls, cls_mlp, tid,
                                     mtid, box_speed[b], verr[0], verr[1]);
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
                        draw(h_box_t[b].cx - poses[s][0], h_box_t[b].cy - poses[s][1], 2 * h_box_t[b].hl,
                             2 * h_box_t[b].hw, h_box_t[b].yaw, cv::Scalar(255, 170, 60));
                    for (const Obb& F : fits) draw(F.cx, F.cy, F.len, F.wid, F.yaw, cv::Scalar(230, 60, 230));
                    for (const Obb& F : done) draw(F.cx, F.cy, F.len, F.wid, F.yaw, cv::Scalar(40, 160, 255));
                    for (const Obb& F : trk_draw) draw(F.cx, F.cy, F.len, F.wid, F.yaw, cv::Scalar(80, 230, 80));
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
            if (sequence) {   // the model's view only, with the tracked boxes
                panel = panel(cv::Rect(W + 20, 0, W, W)).clone();
                cv::rectangle(panel, cv::Rect(0, W - 52, W, 52), cv::Scalar(28, 28, 32), cv::FILLED);
                std::snprintf(buf, sizeof(buf), "frame %d   blue: true   magenta: L-shape", s);
                cv::putText(panel, buf, cv::Point(10, W - 32), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(200, 200, 210), 1, cv::LINE_AA);
                cv::putText(panel, moving ? "orange: + size prior (this scan)   green: motion-tracked + prior"
                                          : "orange: + size prior (this scan)   green: tracked + size prior",
                            cv::Point(10, W - 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 210), 1,
                            cv::LINE_AA);
            }
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
        const std::string name = moving ? "gpu_ground_segmentation_motion"
                                 : sequence ? "gpu_ground_segmentation_track" : "gpu_ground_segmentation";
        cv::VideoWriter video("tmp/" + name + ".avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), sequence ? 5 : 2,
                              frames[0].size());
        for (const cv::Mat& f : frames) video.write(f);
        video.release();
        avi_to_gif("tmp/" + name + ".avi", "gif/" + name + ".gif", sequence ? 5 : 2, sequence ? 340 : 800);
        std::printf("wrote gif/%s.gif\n", name.c_str());
    }

    CUDA_CHECK(cudaFree(d_pts)); CUDA_CHECK(cudaFree(d_gt)); CUDA_CHECK(cudaFree(d_gt_obj));
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
    if (!stage_ms.empty()) {
        const char* sname[8] = { "segmentation (GPU)", "voxel clustering (GPU)", "L-shape (GPU)",
                                 "classification (host)", "static tracker (host)", "motion tracker (host)",
                                 trk_cpu_fit ? "static box fits (CPU)" : "static box fits (GPU)",
                                 trk_cpu_fit ? "motion box fits (CPU)" : "motion box fits (GPU)" };
        std::printf("\n--- pipeline time per scan (%zu scans): mean / max ---\n", stage_ms.size());
        double tot_mean = 0.0, tot_max = 0.0;
        for (int k = 0; k < 8; ++k) {
            double m = 0.0, mx = 0.0;
            for (const auto& a : stage_ms) { m += a[k]; mx = std::max(mx, a[k]); }
            m /= stage_ms.size();
            std::printf("%-24s %8.3f / %8.3f ms\n", sname[k], m, mx);
        }
        for (const auto& a : stage_ms) {   // the pipeline runs one tracker: the motion tracker
            double t = a[0] + a[1] + a[2] + a[3] + a[5] + a[7];
            tot_mean += t; tot_max = std::max(tot_max, t);
        }
        std::printf("%-24s %8.3f / %8.3f ms (segmentation .. motion tracker and its fits)\n", "pipeline",
                    tot_mean / stage_ms.size(), tot_max);
        if (check) std::printf("tracker boxes, GPU vs CPU fits identical: %s\n", trk_fit_same ? "yes" : "no");
    }
    std::printf("fits per scan %ld; CPU %.2f ms, GPU %.3f ms per scan; CPU and GPU boxes identical: %s\n",
                ls_fits / N_SCAN, ls_cpu_ms / N_SCAN, ls_gpu_ms / N_SCAN, ls_same ? "yes" : "no");
    auto box_line = [&](const char* name, const BoxScore& S) {
        if (!S.n) return;
        std::printf("%s: n %d, classed car %d / van %d / none %d (learned: %d / %d / %d)\n", name, S.n, S.cls[0],
                    S.cls[1], S.cls[N_CLS], S.cls_mlp[0], S.cls_mlp[1], S.cls_mlp[N_CLS]);
        for (int m = 0; m < (moving ? N_BM : sequence ? N_BM - 3 : N_BM - 5); ++m)
            std::printf("  %-15s heading err %5.2f deg  IoU %.3f  (>= 0.5: %3d)  centre err %.2f m  "
                        "long side err %.2f m  short side err %.2f m\n", BM_NAME[m],
                        S.yaw[m] / S.n, S.iou[m] / S.n, S.good[m], S.centre[m] / S.n, S.len[m] / S.n, S.wid[m] / S.n);
    };
    box_line("all box observations", bsum[0]);
    box_line("one face visible", bsum[1]);
    box_line("two faces visible", bsum[2]);
    for (int b = 0; b < N_BOX; ++b) box_line(box_name[b], bobj[b]);
    if (obs) std::fclose(obs);
    if (clsf) std::fclose(clsf);
    if (sequence) {
        for (int k = 0; k < (moving ? 2 : 1); ++k) {
            int tracked = 0;
            for (int lt : trk[k].last_track) tracked += lt >= 0;
            std::printf("%s tracker: %zu tracks, %d true boxes tracked, %d identity switches\n",
                        k ? "motion" : "static", trk[k].tracks.size(), tracked, trk[k].id_switches);
        }
    }
    if (moving) {
        box_line("moving vehicles", bmov);
        box_line("parked vehicles", bpark);
        for (int mv = 0; mv < 2; ++mv)
            if (verr_n[mv])
                std::printf("velocity error, %s vehicles: static tracker %.2f m/s, motion tracker %.2f m/s (%d obs)\n",
                            mv ? "moving" : "parked", verr_sum[0][mv] / verr_n[mv], verr_sum[1][mv] / verr_n[mv],
                            verr_n[mv]);
    }
    bool ls_better = bsum[0].n > 0 && bsum[0].yaw[0] < bsum[0].yaw[1] && bsum[0].iou[0] > bsum[0].iou[1];
    bool prior_better = bsum[0].iou[2] > bsum[0].iou[0] && bsum[0].centre[2] < bsum[0].centre[0] &&
                        bsum[0].iou[6] > bsum[0].iou[0] && bsum[0].centre[6] < bsum[0].centre[0];
    double found_rate = cl_sum[0].objects ? (double)cl_sum[0].found / cl_sum[0].objects : 0.0;
    std::printf("library pipeline (cudarobotics::LidarObjectPipeline) gives the same ground, clusters, boxes%s: %s\n",
                sequence ? " and tracks" : "", lib_same ? "yes" : "no");
    bool ok = sg.f1 >= 0.95 && sg.f1 > st.f1 && agree_pct >= 99.9 && found_rate >= 0.9 && cl_same && vcl_same &&
              ls_better && ls_same && prior_better && lib_same;
    if (moving) {
        // (the motion tracker does not beat the single-scan box on the moving vehicles, which the sensor
        // sees from behind or ahead only: see docs/gpu_ground_segmentation.md)
        bool mot_better = bmov.n > 0 && bmov.iou[10] > bmov.iou[8] && bmov.centre[10] < bmov.centre[8] &&
                          bpark.iou[10] >= bpark.iou[8] - 0.01 && trk[1].id_switches <= trk[0].id_switches &&
                          verr_n[1] > 0 && verr_sum[1][1] / verr_n[1] < 1.5;
        // the hybrid box beats both the single-scan and the motion tracker's boxes over all vehicles
        double all_n = bmov.n + bpark.n;
        double hyb = (bmov.iou[11] + bpark.iou[11]) / all_n, one = (bmov.iou[6] + bpark.iou[6]) / all_n;
        double mot = (bmov.iou[10] + bpark.iou[10]) / all_n;
        mot_better = mot_better && hyb > one && hyb > mot;
        ok = sg.f1 >= 0.95 && agree_pct >= 99.9 && cl_same && vcl_same && ls_same && trk_fit_same && lib_same && mot_better;
        if (check) {
            std::printf("check: %s (model F1 >= 0.95, CPU/GPU agreement >= 99.9%%, identical CPU/GPU partition and "
                        "boxes; the motion tracker's boxes with the size prior beat the static tracker's on the moving "
                        "vehicles in IoU and centre error and match them on the parked ones, with no more identity "
                        "switches and a velocity error under 1.5 m/s on the moving vehicles; the hybrid boxes beat the "
                        "single-scan and the motion tracker's boxes over all vehicles)\n", ok ? "PASS" : "FAIL");
            return ok ? 0 : 1;
        }
        return 0;
    }
    if (sequence) {
        bool trk_better = bsum[0].iou[7] > bsum[0].iou[0] && bsum[0].iou[8] > bsum[0].iou[6] &&
                          bsum[0].centre[8] < bsum[0].centre[6];
        ok = sg.f1 >= 0.95 && agree_pct >= 99.9 && cl_same && vcl_same && ls_same && trk_fit_same && lib_same && trk_better;
        if (check) {
            std::printf("check: %s (model F1 >= 0.95, CPU/GPU agreement >= 99.9%%, identical CPU/GPU partition and "
                        "boxes, tracked boxes beat single-scan ones in IoU, and with the size prior in IoU and "
                        "centre error)\n", ok ? "PASS" : "FAIL");
            return ok ? 0 : 1;
        }
        return 0;
    }
    if (check) {
        std::printf("check: %s (model F1 >= 0.95, above the height threshold, CPU/GPU agreement >= 99.9%%, "
                    ">= 90%% of objects found as one cluster, identical CPU/GPU partition, L-shape boxes beat "
                    "axis-aligned ones, identical CPU/GPU boxes, size prior improves IoU and centre error with either class)\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    return 0;
}
