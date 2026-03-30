/*************************************************************************
    > File Name: rrt_star_reeds_shepp.cu
    > CUDA-parallelized RRT* with Reeds-Shepp curves
    > Based on PythonRobotics RRTStarReedsShepp by Atsushi Sakai
    > Parallelizes nearest-neighbor, radius search, and RS path evaluation
 ************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <limits>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Maximum number of segments in a Reeds-Shepp path
#define RS_MAX_SEGS 5
// Maximum discretized steps for collision checking along an RS path
#define RS_MAX_STEPS 256
// Path types count (CSC + CCC)
#define RS_NUM_TYPES 6

// ---------------------------------------------------------------------------
// Reeds-Shepp path segment types
// ---------------------------------------------------------------------------
enum RSSegType { RS_LEFT = 0, RS_STRAIGHT = 1, RS_RIGHT = 2 };

// A Reeds-Shepp path result (device-friendly, fixed size)
struct RSPath {
    float lengths[RS_MAX_SEGS];   // signed: positive=forward, negative=reverse
    int   types[RS_MAX_SEGS];     // RSSegType
    int   num_segs;
    float total_length;           // sum of |lengths|
    bool  valid;
};

// ---------------------------------------------------------------------------
// Device helper: normalize angle to [-pi, pi]
// ---------------------------------------------------------------------------
__host__ __device__ inline float mod2pi(float x) {
    float v = fmodf(x, 2.0f * (float)M_PI);
    if (v < -(float)M_PI) v += 2.0f * (float)M_PI;
    if (v >  (float)M_PI) v -= 2.0f * (float)M_PI;
    return v;
}

// ---------------------------------------------------------------------------
// Device helper: polar coordinates
// ---------------------------------------------------------------------------
__host__ __device__ inline void polar(float x, float y, float* r, float* theta) {
    *r = sqrtf(x * x + y * y);
    *theta = atan2f(y, x);
}

// ---------------------------------------------------------------------------
// Device: CSC path formulas (LSL, LSR, RSL, RSR)
// ---------------------------------------------------------------------------
__host__ __device__ inline void rs_LSL(float x, float y, float phi, float* t, float* u, float* v, bool* ok) {
    float u1, theta;
    polar(x - sinf(phi), y - 1.0f + cosf(phi), &u1, &theta);
    *t = mod2pi(theta);
    *u = u1;
    *v = mod2pi(phi - *t);
    *ok = (*t >= 0.0f && *v >= 0.0f);
}

__host__ __device__ inline void rs_LSR(float x, float y, float phi, float* t, float* u, float* v, bool* ok) {
    float u1, theta;
    polar(x + sinf(phi), y - 1.0f - cosf(phi), &u1, &theta);
    float u1sq = u1 * u1;
    if (u1sq < 4.0f) { *ok = false; return; }
    *u = sqrtf(u1sq - 4.0f);
    float tval = atan2f(2.0f, *u);
    *t = mod2pi(theta + tval);
    *v = mod2pi(*t - phi);
    *ok = (*t >= 0.0f && *v >= 0.0f);
}

__host__ __device__ inline void rs_RSL(float x, float y, float phi, float* t, float* u, float* v, bool* ok) {
    // RSL = time-flipped LSR
    float xt = x, yt = -y, phit = -phi;
    float u1, theta;
    polar(xt + sinf(phit), yt - 1.0f - cosf(phit), &u1, &theta);
    float u1sq = u1 * u1;
    if (u1sq < 4.0f) { *ok = false; return; }
    *u = sqrtf(u1sq - 4.0f);
    float tval = atan2f(2.0f, *u);
    *t = mod2pi(theta + tval);
    *v = mod2pi(*t - phit);
    // Flip signs for right turns
    *t = -(*t);
    *v = -(*v);
    *ok = true;
}

__host__ __device__ inline void rs_RSR(float x, float y, float phi, float* t, float* u, float* v, bool* ok) {
    // RSR = reflect LSL
    float xt = x, yt = -y, phit = -phi;
    float u1, theta;
    polar(xt - sinf(phit), yt - 1.0f + cosf(phit), &u1, &theta);
    *t = mod2pi(theta);
    *u = u1;
    *v = mod2pi(phit - *t);
    *ok = (*t >= 0.0f && *v >= 0.0f);
    *t = -(*t);
    *v = -(*v);
}

// ---------------------------------------------------------------------------
// Device: CCC path formulas (LRL, RLR)
// ---------------------------------------------------------------------------
__host__ __device__ inline void rs_LRL(float x, float y, float phi, float* t, float* u, float* v, bool* ok) {
    float u1, theta;
    polar(x - sinf(phi), y - 1.0f + cosf(phi), &u1, &theta);
    if (u1 > 4.0f) { *ok = false; return; }
    *u = -2.0f * asinf(0.25f * u1);
    *t = mod2pi(theta + 0.5f * (*u) + (float)M_PI);
    *v = mod2pi(phi - *t + *u);
    *ok = (*t >= 0.0f && *v >= 0.0f);
}

__host__ __device__ inline void rs_RLR(float x, float y, float phi, float* t, float* u, float* v, bool* ok) {
    // RLR = reflect LRL
    float xt = x, yt = -y, phit = -phi;
    float u1, theta;
    polar(xt - sinf(phit), yt - 1.0f + cosf(phit), &u1, &theta);
    if (u1 > 4.0f) { *ok = false; return; }
    *u = -2.0f * asinf(0.25f * u1);
    *t = mod2pi(theta + 0.5f * (*u) + (float)M_PI);
    *v = mod2pi(phit - *t + *u);
    *t = -(*t);
    *u = -(*u);
    *v = -(*v);
    *ok = true;
}

// ---------------------------------------------------------------------------
// Device: Compute best Reeds-Shepp path between two poses
//   Start at origin (0,0,0), goal at transformed (x,y,phi)
//   rho = minimum turning radius
// ---------------------------------------------------------------------------
__host__ __device__ RSPath reeds_shepp_path(float x1, float y1, float yaw1,
                                             float x2, float y2, float yaw2,
                                             float rho) {
    // Transform goal to local frame of start, scaled by 1/rho
    float dx = x2 - x1;
    float dy = y2 - y1;
    float c = cosf(yaw1);
    float s = sinf(yaw1);
    float lx = (c * dx + s * dy) / rho;
    float ly = (-s * dx + c * dy) / rho;
    float lphi = mod2pi(yaw2 - yaw1);

    RSPath best;
    best.valid = false;
    best.total_length = 1e10f;
    best.num_segs = 0;

    // Try all 6 basic path types, plus timeflip/reflect variants
    // For each type, we also try negating all lengths (reverse driving)
    float t, u, v;
    bool ok;

    // We try each of the 6 formulas with both (lx,ly,lphi) and (-lx,ly,-lphi) [timeflip]
    // and (lx,-ly,-lphi) [reflect] and (-lx,-ly,lphi) [timeflip+reflect]
    // That gives us the full 48 path types from the 6 base formulas

    struct Variant {
        float x, y, phi;
        float tsign, usign, vsign;  // sign adjustments for the result
    };

    Variant variants[4] = {
        { lx,  ly,  lphi,  1.0f,  1.0f,  1.0f},
        {-lx,  ly, -lphi, -1.0f, -1.0f, -1.0f},  // timeflip
        { lx, -ly, -lphi, -1.0f,  1.0f, -1.0f},   // reflect (handled by RSR, RSL, RLR already)
        {-lx, -ly,  lphi,  1.0f, -1.0f,  1.0f},   // timeflip + reflect
    };

    // For CSC types, segments are: turn, straight, turn
    // For CCC types, segments are: turn, turn, turn
    for (int vi = 0; vi < 4; vi++) {
        float vx = variants[vi].x;
        float vy = variants[vi].y;
        float vphi = variants[vi].phi;
        float ts = variants[vi].tsign;
        float us = variants[vi].usign;
        float vs = variants[vi].vsign;

        // LSL
        rs_LSL(vx, vy, vphi, &t, &u, &v, &ok);
        if (ok) {
            float tl = fabsf(t) + fabsf(u) + fabsf(v);
            if (tl < best.total_length) {
                best.total_length = tl * rho;
                best.num_segs = 3;
                best.lengths[0] = t * ts * rho;
                best.lengths[1] = u * us * rho;
                best.lengths[2] = v * vs * rho;
                best.types[0] = RS_LEFT; best.types[1] = RS_STRAIGHT; best.types[2] = RS_LEFT;
                if (vi == 2 || vi == 3) { best.types[0] = RS_RIGHT; best.types[2] = RS_RIGHT; }
                best.valid = true;
            }
        }

        // LSR
        rs_LSR(vx, vy, vphi, &t, &u, &v, &ok);
        if (ok) {
            float tl = fabsf(t) + fabsf(u) + fabsf(v);
            if (tl < best.total_length) {
                best.total_length = tl * rho;
                best.num_segs = 3;
                best.lengths[0] = t * ts * rho;
                best.lengths[1] = u * us * rho;
                best.lengths[2] = v * vs * rho;
                best.types[0] = RS_LEFT; best.types[1] = RS_STRAIGHT; best.types[2] = RS_RIGHT;
                if (vi == 2 || vi == 3) { best.types[0] = RS_RIGHT; best.types[2] = RS_LEFT; }
                best.valid = true;
            }
        }

        // LRL
        rs_LRL(vx, vy, vphi, &t, &u, &v, &ok);
        if (ok) {
            float tl = fabsf(t) + fabsf(u) + fabsf(v);
            if (tl < best.total_length) {
                best.total_length = tl * rho;
                best.num_segs = 3;
                best.lengths[0] = t * ts * rho;
                best.lengths[1] = u * us * rho;
                best.lengths[2] = v * vs * rho;
                best.types[0] = RS_LEFT; best.types[1] = RS_RIGHT; best.types[2] = RS_LEFT;
                if (vi == 2 || vi == 3) { best.types[0] = RS_RIGHT; best.types[1] = RS_LEFT; best.types[2] = RS_RIGHT; }
                best.valid = true;
            }
        }
    }

    // Also try RSR, RSL, RLR directly (they are already reflections, but let's be thorough)
    for (int vi = 0; vi < 4; vi++) {
        float vx = variants[vi].x;
        float vy = variants[vi].y;
        float vphi = variants[vi].phi;
        float ts = variants[vi].tsign;
        float us = variants[vi].usign;
        float vs = variants[vi].vsign;

        rs_RSR(vx, vy, vphi, &t, &u, &v, &ok);
        if (ok) {
            float tl = fabsf(t) + fabsf(u) + fabsf(v);
            if (tl < best.total_length) {
                best.total_length = tl * rho;
                best.num_segs = 3;
                best.lengths[0] = t * ts * rho;
                best.lengths[1] = u * us * rho;
                best.lengths[2] = v * vs * rho;
                best.types[0] = RS_RIGHT; best.types[1] = RS_STRAIGHT; best.types[2] = RS_RIGHT;
                if (vi == 2 || vi == 3) { best.types[0] = RS_LEFT; best.types[2] = RS_LEFT; }
                best.valid = true;
            }
        }

        rs_RSL(vx, vy, vphi, &t, &u, &v, &ok);
        if (ok) {
            float tl = fabsf(t) + fabsf(u) + fabsf(v);
            if (tl < best.total_length) {
                best.total_length = tl * rho;
                best.num_segs = 3;
                best.lengths[0] = t * ts * rho;
                best.lengths[1] = u * us * rho;
                best.lengths[2] = v * vs * rho;
                best.types[0] = RS_RIGHT; best.types[1] = RS_STRAIGHT; best.types[2] = RS_LEFT;
                if (vi == 2 || vi == 3) { best.types[0] = RS_LEFT; best.types[2] = RS_RIGHT; }
                best.valid = true;
            }
        }

        rs_RLR(vx, vy, vphi, &t, &u, &v, &ok);
        if (ok) {
            float tl = fabsf(t) + fabsf(u) + fabsf(v);
            if (tl < best.total_length) {
                best.total_length = tl * rho;
                best.num_segs = 3;
                best.lengths[0] = t * ts * rho;
                best.lengths[1] = u * us * rho;
                best.lengths[2] = v * vs * rho;
                best.types[0] = RS_RIGHT; best.types[1] = RS_LEFT; best.types[2] = RS_RIGHT;
                if (vi == 2 || vi == 3) { best.types[0] = RS_LEFT; best.types[1] = RS_RIGHT; best.types[2] = RS_LEFT; }
                best.valid = true;
            }
        }
    }

    // Recompute total_length as sum of |lengths| (in case rho scaling changed things)
    if (best.valid) {
        best.total_length = 0.0f;
        for (int i = 0; i < best.num_segs; i++) {
            best.total_length += fabsf(best.lengths[i]);
        }
    }

    return best;
}

// ---------------------------------------------------------------------------
// Device: Interpolate a single point along an RS path
//   Given cumulative arc-length 'dist' along the path, compute (x, y, yaw)
// ---------------------------------------------------------------------------
__host__ __device__ void rs_interpolate(float sx, float sy, float syaw,
                                         const RSPath& path, float rho,
                                         float dist,
                                         float* ox, float* oy, float* oyaw) {
    float cx = sx, cy = sy, cyaw = syaw;
    float remaining = dist;

    for (int i = 0; i < path.num_segs; i++) {
        float seg_len = path.lengths[i]; // signed
        float seg_abs = fabsf(seg_len);
        float use_len;

        if (remaining <= seg_abs) {
            use_len = remaining;
        } else {
            use_len = seg_abs;
        }

        // Direction: positive length = forward, negative = reverse
        float direction = (seg_len >= 0.0f) ? 1.0f : -1.0f;

        if (path.types[i] == RS_STRAIGHT) {
            cx += direction * use_len * cosf(cyaw);
            cy += direction * use_len * sinf(cyaw);
        } else if (path.types[i] == RS_LEFT) {
            float dangle = direction * use_len / rho;
            cx += rho * (sinf(cyaw + dangle) - sinf(cyaw));
            cy += rho * (-cosf(cyaw + dangle) + cosf(cyaw));
            cyaw = mod2pi(cyaw + dangle);
        } else { // RS_RIGHT
            float dangle = -direction * use_len / rho;
            cx += rho * (-sinf(cyaw + dangle) + sinf(cyaw));
            cy += rho * (cosf(cyaw + dangle) - cosf(cyaw));
            cyaw = mod2pi(cyaw + dangle);
        }

        remaining -= use_len;
        if (remaining <= 1e-6f) break;
    }

    *ox = cx;
    *oy = cy;
    *oyaw = cyaw;
}

// ---------------------------------------------------------------------------
// Node (host-side tree structure, includes yaw)
// ---------------------------------------------------------------------------
struct Node {
    float x;
    float y;
    float yaw;
    int parent_idx;
    float cost;
    std::vector<float> path_x;
    std::vector<float> path_y;
    std::vector<float> path_yaw;

    Node() : x(0), y(0), yaw(0), parent_idx(-1), cost(0) {}
    Node(float x_, float y_, float yaw_) : x(x_), y(y_), yaw(yaw_), parent_idx(-1), cost(0) {}
};

// ---------------------------------------------------------------------------
// Obstacle
// ---------------------------------------------------------------------------
struct Obstacle {
    float cx, cy, r;
};

// ---------------------------------------------------------------------------
// CUDA Kernel: find nearest node (Euclidean distance for initial filtering)
// ---------------------------------------------------------------------------
__global__ void find_nearest_kernel(
    const float* __restrict__ d_node_x,
    const float* __restrict__ d_node_y,
    int num_nodes,
    float qx, float qy,
    float* d_min_dist,
    int* d_min_idx)
{
    extern __shared__ char shared_mem[];
    float* s_dist = (float*)shared_mem;
    int*   s_idx  = (int*)(s_dist + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float my_dist = FLT_MAX;
    int   my_idx  = -1;

    if (gid < num_nodes) {
        float dx = d_node_x[gid] - qx;
        float dy = d_node_y[gid] - qy;
        my_dist = dx * dx + dy * dy;
        my_idx  = gid;
    }

    s_dist[tid] = my_dist;
    s_idx[tid]  = my_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_dist[tid + s] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + s];
                s_idx[tid]  = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_min_dist[blockIdx.x] = s_dist[0];
        d_min_idx[blockIdx.x]  = s_idx[0];
    }
}

// ---------------------------------------------------------------------------
// CUDA Kernel: find all nodes within radius r (Euclidean)
// ---------------------------------------------------------------------------
__global__ void find_near_nodes_kernel(
    const float* __restrict__ d_node_x,
    const float* __restrict__ d_node_y,
    int num_nodes,
    float qx, float qy,
    float radius_sq,
    int* d_near_mask)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < num_nodes) {
        float dx = d_node_x[gid] - qx;
        float dy = d_node_y[gid] - qy;
        d_near_mask[gid] = (dx * dx + dy * dy < radius_sq) ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// CUDA Kernel: Reeds-Shepp batch evaluation
//   For each candidate near node, compute RS path length and check collision.
//   Each thread evaluates one candidate parent node.
//   Outputs: d_rs_lengths[i] = RS path length (FLT_MAX if invalid/collision)
// ---------------------------------------------------------------------------
__global__ void reeds_shepp_batch_kernel(
    const float* __restrict__ d_from_x,
    const float* __restrict__ d_from_y,
    const float* __restrict__ d_from_yaw,
    int num_candidates,
    float to_x, float to_y, float to_yaw,
    float rho,
    float step_size,
    const float* __restrict__ d_obs_cx,
    const float* __restrict__ d_obs_cy,
    const float* __restrict__ d_obs_r,
    int num_obs,
    float* d_rs_lengths,
    int* d_rs_valid)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_candidates) return;

    float sx = d_from_x[gid];
    float sy = d_from_y[gid];
    float syaw = d_from_yaw[gid];

    // Compute RS path
    RSPath path = reeds_shepp_path(sx, sy, syaw, to_x, to_y, to_yaw, rho);

    if (!path.valid || path.total_length < 1e-6f) {
        d_rs_lengths[gid] = FLT_MAX;
        d_rs_valid[gid] = 0;
        return;
    }

    // Discretize path and check collision
    int n_steps = (int)(path.total_length / step_size) + 1;
    if (n_steps > RS_MAX_STEPS) n_steps = RS_MAX_STEPS;

    for (int s = 0; s <= n_steps; s++) {
        float dist = fminf(s * step_size, path.total_length);
        float px, py, pyaw;
        rs_interpolate(sx, sy, syaw, path, rho, dist, &px, &py, &pyaw);

        // Check against all obstacles
        for (int o = 0; o < num_obs; o++) {
            float dx = d_obs_cx[o] - px;
            float dy = d_obs_cy[o] - py;
            if (dx * dx + dy * dy <= d_obs_r[o] * d_obs_r[o]) {
                d_rs_lengths[gid] = FLT_MAX;
                d_rs_valid[gid] = 0;
                return;
            }
        }
    }

    d_rs_lengths[gid] = path.total_length;
    d_rs_valid[gid] = 1;
}

// ---------------------------------------------------------------------------
// CUDA Kernel: collision check along an RS path (single path)
//   Used for rewiring and final goal connection
// ---------------------------------------------------------------------------
__global__ void collision_check_rs_kernel(
    float sx, float sy, float syaw,
    float gx, float gy, float gyaw,
    float rho, float step_size,
    const float* __restrict__ d_obs_cx,
    const float* __restrict__ d_obs_cy,
    const float* __restrict__ d_obs_r,
    int num_obs,
    int* d_result,
    float* d_path_length)
{
    // Single-thread kernel for a single path check
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    RSPath path = reeds_shepp_path(sx, sy, syaw, gx, gy, gyaw, rho);

    if (!path.valid) {
        d_result[0] = 0;
        d_path_length[0] = FLT_MAX;
        return;
    }

    int n_steps = (int)(path.total_length / step_size) + 1;
    if (n_steps > RS_MAX_STEPS) n_steps = RS_MAX_STEPS;

    for (int s = 0; s <= n_steps; s++) {
        float dist = fminf(s * step_size, path.total_length);
        float px, py, pyaw;
        rs_interpolate(sx, sy, syaw, path, rho, dist, &px, &py, &pyaw);

        for (int o = 0; o < num_obs; o++) {
            float dx = d_obs_cx[o] - px;
            float dy = d_obs_cy[o] - py;
            if (dx * dx + dy * dy <= d_obs_r[o] * d_obs_r[o]) {
                d_result[0] = 0;
                d_path_length[0] = FLT_MAX;
                return;
            }
        }
    }

    d_result[0] = 1;
    d_path_length[0] = path.total_length;
}

// ---------------------------------------------------------------------------
// RRTStarRS class
// ---------------------------------------------------------------------------
class RRTStarRS {
public:
    RRTStarRS(float start_x, float start_y, float start_yaw,
              float goal_x, float goal_y, float goal_yaw,
              const std::vector<Obstacle>& obstacles,
              float rand_min, float rand_max,
              float expand_dis,
              int goal_sample_rate,
              int max_iter,
              float connect_circle_dist,
              float curvature);
    ~RRTStarRS();

    std::vector<Node> planning();

private:
    // Parameters
    float start_x_, start_y_, start_yaw_;
    float goal_x_, goal_y_, goal_yaw_;
    std::vector<Obstacle> obstacles_;
    float rand_min_, rand_max_;
    float expand_dis_;
    int goal_sample_rate_;
    int max_iter_;
    float connect_circle_dist_;
    float curvature_;  // 1/rho
    float rho_;        // minimum turning radius
    float step_size_;  // discretization step for collision checking

    // Tree
    std::vector<Node> node_list_;

    // Device memory for node positions
    float* d_node_x_;
    float* d_node_y_;
    float* d_node_yaw_;
    int    d_node_capacity_;

    // Device memory for obstacles
    float* d_obs_cx_;
    float* d_obs_cy_;
    float* d_obs_r_;
    int    num_obs_;

    // Device memory for kernel outputs
    float* d_min_dist_;
    int*   d_min_idx_;
    int*   d_near_mask_;
    float* d_rs_lengths_;
    int*   d_rs_valid_;
    float* d_cand_x_;
    float* d_cand_y_;
    float* d_cand_yaw_;
    int*   d_single_result_;
    float* d_single_path_len_;

    int max_blocks_;

    // Random
    std::mt19937 rng_;
    std::uniform_int_distribution<int> goal_dist_;
    std::uniform_real_distribution<float> area_dist_;
    std::uniform_real_distribution<float> yaw_dist_;

    // Methods
    void ensure_device_capacity(int needed);
    void sync_node_to_device(int idx);
    int  find_nearest_gpu(float qx, float qy);
    std::vector<int> find_near_nodes_gpu(float qx, float qy, float radius);
    void steer(int from_idx, float to_x, float to_y, float to_yaw,
               Node& new_node, bool& valid);
    void generate_rs_path_points(float sx, float sy, float syaw,
                                  float gx, float gy, float gyaw,
                                  std::vector<float>& px, std::vector<float>& py,
                                  std::vector<float>& pyaw);
    bool check_collision_host(float sx, float sy, float syaw,
                               float gx, float gy, float gyaw);
    float rs_distance(float x1, float y1, float yaw1,
                      float x2, float y2, float yaw2);
    void propagate_cost_to_leaves(int parent_idx);
};

RRTStarRS::RRTStarRS(float start_x, float start_y, float start_yaw,
                      float goal_x, float goal_y, float goal_yaw,
                      const std::vector<Obstacle>& obstacles,
                      float rand_min, float rand_max,
                      float expand_dis,
                      int goal_sample_rate,
                      int max_iter,
                      float connect_circle_dist,
                      float curvature)
    : start_x_(start_x), start_y_(start_y), start_yaw_(start_yaw)
    , goal_x_(goal_x), goal_y_(goal_y), goal_yaw_(goal_yaw)
    , obstacles_(obstacles)
    , rand_min_(rand_min), rand_max_(rand_max)
    , expand_dis_(expand_dis)
    , goal_sample_rate_(goal_sample_rate)
    , max_iter_(max_iter)
    , connect_circle_dist_(connect_circle_dist)
    , curvature_(curvature)
    , rho_(1.0f / curvature)
    , step_size_(0.2f)
    , d_node_x_(nullptr), d_node_y_(nullptr), d_node_yaw_(nullptr), d_node_capacity_(0)
    , d_obs_cx_(nullptr), d_obs_cy_(nullptr), d_obs_r_(nullptr)
    , d_min_dist_(nullptr), d_min_idx_(nullptr)
    , d_near_mask_(nullptr), d_rs_lengths_(nullptr), d_rs_valid_(nullptr)
    , d_cand_x_(nullptr), d_cand_y_(nullptr), d_cand_yaw_(nullptr)
    , d_single_result_(nullptr), d_single_path_len_(nullptr)
    , rng_(std::random_device{}())
    , goal_dist_(0, 100)
    , area_dist_(rand_min, rand_max)
    , yaw_dist_(-(float)M_PI, (float)M_PI)
{
    num_obs_ = (int)obstacles_.size();

    // Upload obstacles to device
    std::vector<float> ocx(num_obs_), ocy(num_obs_), orr(num_obs_);
    for (int i = 0; i < num_obs_; i++) {
        ocx[i] = obstacles_[i].cx;
        ocy[i] = obstacles_[i].cy;
        orr[i] = obstacles_[i].r;
    }
    CUDA_CHECK(cudaMalloc(&d_obs_cx_, num_obs_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obs_cy_, num_obs_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obs_r_,  num_obs_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_obs_cx_, ocx.data(), num_obs_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_cy_, ocy.data(), num_obs_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_r_,  orr.data(), num_obs_ * sizeof(float), cudaMemcpyHostToDevice));

    // Pre-allocate workspace
    int init_cap = max_iter + 16;
    max_blocks_ = (init_cap + 255) / 256;

    CUDA_CHECK(cudaMalloc(&d_min_dist_, max_blocks_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_min_idx_,  max_blocks_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_near_mask_, init_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rs_lengths_, init_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rs_valid_, init_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cand_x_, init_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cand_y_, init_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cand_yaw_, init_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_single_result_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_single_path_len_, sizeof(float)));

    ensure_device_capacity(init_cap);
}

RRTStarRS::~RRTStarRS() {
    cudaFree(d_node_x_);
    cudaFree(d_node_y_);
    cudaFree(d_node_yaw_);
    cudaFree(d_obs_cx_);
    cudaFree(d_obs_cy_);
    cudaFree(d_obs_r_);
    cudaFree(d_min_dist_);
    cudaFree(d_min_idx_);
    cudaFree(d_near_mask_);
    cudaFree(d_rs_lengths_);
    cudaFree(d_rs_valid_);
    cudaFree(d_cand_x_);
    cudaFree(d_cand_y_);
    cudaFree(d_cand_yaw_);
    cudaFree(d_single_result_);
    cudaFree(d_single_path_len_);
}

void RRTStarRS::ensure_device_capacity(int needed) {
    if (needed <= d_node_capacity_) return;
    int new_cap = std::max(needed, d_node_capacity_ * 2);

    float* new_x = nullptr;
    float* new_y = nullptr;
    float* new_yaw = nullptr;
    CUDA_CHECK(cudaMalloc(&new_x, new_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&new_y, new_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&new_yaw, new_cap * sizeof(float)));

    if (d_node_x_ && d_node_capacity_ > 0) {
        CUDA_CHECK(cudaMemcpy(new_x, d_node_x_, d_node_capacity_ * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_y, d_node_y_, d_node_capacity_ * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_yaw, d_node_yaw_, d_node_capacity_ * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaFree(d_node_x_);
        cudaFree(d_node_y_);
        cudaFree(d_node_yaw_);
    }

    d_node_x_ = new_x;
    d_node_y_ = new_y;
    d_node_yaw_ = new_yaw;
    d_node_capacity_ = new_cap;

    int new_blocks = (new_cap + 255) / 256;
    if (new_blocks > max_blocks_) {
        cudaFree(d_min_dist_);
        cudaFree(d_min_idx_);
        CUDA_CHECK(cudaMalloc(&d_min_dist_, new_blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_min_idx_,  new_blocks * sizeof(int)));
        max_blocks_ = new_blocks;
    }

    cudaFree(d_near_mask_);
    cudaFree(d_rs_lengths_);
    cudaFree(d_rs_valid_);
    cudaFree(d_cand_x_);
    cudaFree(d_cand_y_);
    cudaFree(d_cand_yaw_);
    CUDA_CHECK(cudaMalloc(&d_near_mask_, new_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rs_lengths_, new_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rs_valid_, new_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cand_x_, new_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cand_y_, new_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cand_yaw_, new_cap * sizeof(float)));
}

void RRTStarRS::sync_node_to_device(int idx) {
    ensure_device_capacity(idx + 1);
    float x = node_list_[idx].x;
    float y = node_list_[idx].y;
    float yaw = node_list_[idx].yaw;
    CUDA_CHECK(cudaMemcpy(d_node_x_ + idx, &x, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_y_ + idx, &y, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_yaw_ + idx, &yaw, sizeof(float), cudaMemcpyHostToDevice));
}

int RRTStarRS::find_nearest_gpu(float qx, float qy) {
    int n = (int)node_list_.size();
    if (n == 0) return -1;

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    size_t shared_mem = block_size * (sizeof(float) + sizeof(int));

    find_nearest_kernel<<<num_blocks, block_size, shared_mem>>>(
        d_node_x_, d_node_y_, n, qx, qy, d_min_dist_, d_min_idx_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_dist(num_blocks);
    std::vector<int>   h_idx(num_blocks);
    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_min_dist_, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_idx.data(),  d_min_idx_,  num_blocks * sizeof(int),   cudaMemcpyDeviceToHost));

    float best_dist = FLT_MAX;
    int   best_idx  = -1;
    for (int b = 0; b < num_blocks; b++) {
        if (h_dist[b] < best_dist) {
            best_dist = h_dist[b];
            best_idx  = h_idx[b];
        }
    }
    return best_idx;
}

std::vector<int> RRTStarRS::find_near_nodes_gpu(float qx, float qy, float radius) {
    int n = (int)node_list_.size();
    if (n == 0) return {};

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    float radius_sq = radius * radius;

    find_near_nodes_kernel<<<num_blocks, block_size>>>(
        d_node_x_, d_node_y_, n, qx, qy, radius_sq, d_near_mask_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_mask(n);
    CUDA_CHECK(cudaMemcpy(h_mask.data(), d_near_mask_, n * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> near_indices;
    for (int i = 0; i < n; i++) {
        if (h_mask[i]) near_indices.push_back(i);
    }
    return near_indices;
}

float RRTStarRS::rs_distance(float x1, float y1, float yaw1,
                              float x2, float y2, float yaw2) {
    RSPath path = reeds_shepp_path(x1, y1, yaw1, x2, y2, yaw2, rho_);
    if (!path.valid) return FLT_MAX;
    return path.total_length;
}

void RRTStarRS::generate_rs_path_points(float sx, float sy, float syaw,
                                          float gx, float gy, float gyaw,
                                          std::vector<float>& px,
                                          std::vector<float>& py,
                                          std::vector<float>& pyaw) {
    RSPath path = reeds_shepp_path(sx, sy, syaw, gx, gy, gyaw, rho_);
    if (!path.valid) return;

    int n_steps = (int)(path.total_length / step_size_) + 1;
    px.reserve(n_steps + 1);
    py.reserve(n_steps + 1);
    pyaw.reserve(n_steps + 1);

    for (int s = 0; s <= n_steps; s++) {
        float dist = std::min(s * step_size_, path.total_length);
        float ox, oy, oyaw;
        rs_interpolate(sx, sy, syaw, path, rho_, dist, &ox, &oy, &oyaw);
        px.push_back(ox);
        py.push_back(oy);
        pyaw.push_back(oyaw);
    }
}

bool RRTStarRS::check_collision_host(float sx, float sy, float syaw,
                                      float gx, float gy, float gyaw) {
    RSPath path = reeds_shepp_path(sx, sy, syaw, gx, gy, gyaw, rho_);
    if (!path.valid) return false;

    int n_steps = (int)(path.total_length / step_size_) + 1;
    for (int s = 0; s <= n_steps; s++) {
        float dist = std::min(s * step_size_, path.total_length);
        float px, py, pyaw;
        rs_interpolate(sx, sy, syaw, path, rho_, dist, &px, &py, &pyaw);

        for (const auto& ob : obstacles_) {
            float dx = ob.cx - px;
            float dy = ob.cy - py;
            if (std::sqrt(dx * dx + dy * dy) <= ob.r) return false;
        }
    }
    return true;
}

void RRTStarRS::steer(int from_idx, float to_x, float to_y, float to_yaw,
                       Node& new_node, bool& valid) {
    Node& from = node_list_[from_idx];

    RSPath path = reeds_shepp_path(from.x, from.y, from.yaw,
                                    to_x, to_y, to_yaw, rho_);
    if (!path.valid) {
        valid = false;
        return;
    }

    // If path is longer than expand_dis, truncate
    float use_length = std::min(path.total_length, expand_dis_);

    // Discretize path up to use_length
    int n_steps = (int)(use_length / step_size_) + 1;
    float end_x, end_y, end_yaw;

    new_node.path_x.clear();
    new_node.path_y.clear();
    new_node.path_yaw.clear();

    for (int s = 0; s <= n_steps; s++) {
        float dist = std::min(s * step_size_, use_length);
        float px, py, pyaw;
        rs_interpolate(from.x, from.y, from.yaw, path, rho_, dist, &px, &py, &pyaw);
        new_node.path_x.push_back(px);
        new_node.path_y.push_back(py);
        new_node.path_yaw.push_back(pyaw);
        end_x = px;
        end_y = py;
        end_yaw = pyaw;
    }

    new_node.x = end_x;
    new_node.y = end_y;
    new_node.yaw = end_yaw;
    new_node.parent_idx = from_idx;
    new_node.cost = from.cost + use_length;
    valid = true;
}

void RRTStarRS::propagate_cost_to_leaves(int parent_idx) {
    for (int i = 0; i < (int)node_list_.size(); i++) {
        if (node_list_[i].parent_idx == parent_idx) {
            float d = rs_distance(node_list_[parent_idx].x, node_list_[parent_idx].y, node_list_[parent_idx].yaw,
                                  node_list_[i].x, node_list_[i].y, node_list_[i].yaw);
            if (d < FLT_MAX) {
                node_list_[i].cost = node_list_[parent_idx].cost + d;
            }
            propagate_cost_to_leaves(i);
        }
    }
}

std::vector<Node> RRTStarRS::planning() {
    // Visualization setup
    cv::namedWindow("rrt_star_rs", cv::WINDOW_NORMAL);
    float range = rand_max_ - rand_min_;
    int img_reso = 15;
    int img_w = (int)(range * img_reso);
    int img_h = (int)(range * img_reso);
    cv::Mat bg(img_h, img_w, CV_8UC3, cv::Scalar(255, 255, 255));

    auto to_px = [&](float x) -> int { return (int)((x - rand_min_) * img_reso); };
    auto to_py = [&](float y) -> int { return (int)((y - rand_min_) * img_reso); };

    // Draw obstacles
    for (const auto& ob : obstacles_) {
        cv::circle(bg, cv::Point(to_px(ob.cx), to_py(ob.cy)),
                   (int)(ob.r * img_reso), cv::Scalar(0, 0, 0), -1);
    }

    // Draw start with orientation arrow
    {
        cv::Point sp(to_px(start_x_), to_py(start_y_));
        cv::circle(bg, sp, 8, cv::Scalar(0, 0, 255), -1);
        float arrow_len = 2.0f * img_reso;
        cv::Point ep(sp.x + (int)(arrow_len * cosf(start_yaw_)),
                     sp.y + (int)(arrow_len * sinf(start_yaw_)));
        cv::arrowedLine(bg, sp, ep, cv::Scalar(0, 0, 255), 3);
    }

    // Draw goal with orientation arrow
    {
        cv::Point gp(to_px(goal_x_), to_py(goal_y_));
        cv::circle(bg, gp, 8, cv::Scalar(255, 0, 0), -1);
        float arrow_len = 2.0f * img_reso;
        cv::Point ep(gp.x + (int)(arrow_len * cosf(goal_yaw_)),
                     gp.y + (int)(arrow_len * sinf(goal_yaw_)));
        cv::arrowedLine(bg, gp, ep, cv::Scalar(255, 0, 0), 3);
    }

    cv::Mat display;

    // Initialize tree
    Node start_node(start_x_, start_y_, start_yaw_);
    start_node.cost = 0.0f;
    start_node.parent_idx = -1;
    node_list_.clear();
    node_list_.push_back(start_node);
    sync_node_to_device(0);

    int best_goal_idx = -1;
    float best_goal_cost = std::numeric_limits<float>::max();

    for (int iter = 0; iter < max_iter_; iter++) {
        // Sample random point (or goal)
        float rnd_x, rnd_y, rnd_yaw;
        if (goal_dist_(rng_) <= goal_sample_rate_) {
            rnd_x = goal_x_;
            rnd_y = goal_y_;
            rnd_yaw = goal_yaw_;
        } else {
            rnd_x = area_dist_(rng_);
            rnd_y = area_dist_(rng_);
            rnd_yaw = yaw_dist_(rng_);
        }

        // Find nearest node (GPU, Euclidean)
        int nearest_idx = find_nearest_gpu(rnd_x, rnd_y);
        if (nearest_idx < 0) continue;

        // Steer toward random point using RS curve
        Node new_node;
        bool steer_valid;
        steer(nearest_idx, rnd_x, rnd_y, rnd_yaw, new_node, steer_valid);
        if (!steer_valid) continue;

        // Check collision along steered path (host side, quick check)
        bool path_ok = true;
        for (size_t p = 0; p < new_node.path_x.size(); p++) {
            float px = new_node.path_x[p];
            float py = new_node.path_y[p];
            bool collides = false;
            for (const auto& ob : obstacles_) {
                float dx = ob.cx - px;
                float dy = ob.cy - py;
                if (std::sqrt(dx * dx + dy * dy) <= ob.r) {
                    collides = true;
                    break;
                }
            }
            if (collides) { path_ok = false; break; }
        }
        if (!path_ok) continue;

        // --- RRT* with RS: find near nodes, choose best parent, rewire ---

        int nnode = (int)node_list_.size() + 1;
        float r = connect_circle_dist_ * std::sqrt(std::log((float)nnode) / (float)nnode);
        r = std::min(r, expand_dis_ * 10.0f);

        // Find near nodes (GPU, Euclidean filter)
        std::vector<int> near_indices = find_near_nodes_gpu(new_node.x, new_node.y, r);

        // Evaluate RS paths from all near nodes to new_node IN PARALLEL on GPU
        if (!near_indices.empty()) {
            int nc = (int)near_indices.size();

            // Upload candidate positions
            std::vector<float> h_cx(nc), h_cy(nc), h_cyaw(nc);
            for (int i = 0; i < nc; i++) {
                h_cx[i] = node_list_[near_indices[i]].x;
                h_cy[i] = node_list_[near_indices[i]].y;
                h_cyaw[i] = node_list_[near_indices[i]].yaw;
            }

            CUDA_CHECK(cudaMemcpy(d_cand_x_, h_cx.data(), nc * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_cand_y_, h_cy.data(), nc * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_cand_yaw_, h_cyaw.data(), nc * sizeof(float), cudaMemcpyHostToDevice));

            int block_size = 256;
            int num_blocks = (nc + block_size - 1) / block_size;

            // Each thread: compute RS path from near[i] -> new_node, check collision, output length
            reeds_shepp_batch_kernel<<<num_blocks, block_size>>>(
                d_cand_x_, d_cand_y_, d_cand_yaw_, nc,
                new_node.x, new_node.y, new_node.yaw,
                rho_, step_size_,
                d_obs_cx_, d_obs_cy_, d_obs_r_, num_obs_,
                d_rs_lengths_, d_rs_valid_);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Read results
            std::vector<float> h_lengths(nc);
            std::vector<int> h_valid(nc);
            CUDA_CHECK(cudaMemcpy(h_lengths.data(), d_rs_lengths_, nc * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_valid.data(), d_rs_valid_, nc * sizeof(int), cudaMemcpyDeviceToHost));

            // Choose best parent
            float min_cost = new_node.cost;
            int best_parent = new_node.parent_idx;
            float best_rs_len = -1.0f;

            for (int i = 0; i < nc; i++) {
                if (!h_valid[i]) continue;
                int ni = near_indices[i];
                float c = node_list_[ni].cost + h_lengths[i];
                if (c < min_cost) {
                    min_cost = c;
                    best_parent = ni;
                    best_rs_len = h_lengths[i];
                }
            }

            if (best_parent != new_node.parent_idx && best_rs_len > 0.0f) {
                // Rebuild path from new parent
                new_node.parent_idx = best_parent;
                new_node.cost = min_cost;
                new_node.path_x.clear();
                new_node.path_y.clear();
                new_node.path_yaw.clear();
                generate_rs_path_points(
                    node_list_[best_parent].x, node_list_[best_parent].y, node_list_[best_parent].yaw,
                    new_node.x, new_node.y, new_node.yaw,
                    new_node.path_x, new_node.path_y, new_node.path_yaw);
            }
        }

        // Add new node to tree
        int new_idx = (int)node_list_.size();
        node_list_.push_back(new_node);
        sync_node_to_device(new_idx);

        // Rewire near nodes through new_node
        if (!near_indices.empty()) {
            for (size_t i = 0; i < near_indices.size(); i++) {
                int ni = near_indices[i];
                if (ni == new_node.parent_idx) continue;

                float rs_len = rs_distance(new_node.x, new_node.y, new_node.yaw,
                                           node_list_[ni].x, node_list_[ni].y, node_list_[ni].yaw);
                if (rs_len >= FLT_MAX) continue;

                float improved_cost = new_node.cost + rs_len;
                if (improved_cost < node_list_[ni].cost) {
                    // Check collision
                    if (check_collision_host(new_node.x, new_node.y, new_node.yaw,
                                             node_list_[ni].x, node_list_[ni].y, node_list_[ni].yaw)) {
                        node_list_[ni].parent_idx = new_idx;
                        node_list_[ni].cost = improved_cost;

                        // Rebuild path for rewired node
                        node_list_[ni].path_x.clear();
                        node_list_[ni].path_y.clear();
                        node_list_[ni].path_yaw.clear();
                        generate_rs_path_points(
                            new_node.x, new_node.y, new_node.yaw,
                            node_list_[ni].x, node_list_[ni].y, node_list_[ni].yaw,
                            node_list_[ni].path_x, node_list_[ni].path_y, node_list_[ni].path_yaw);

                        propagate_cost_to_leaves(ni);
                    }
                }
            }
        }

        // Draw tree edge
        if (new_node.path_x.size() >= 2) {
            for (size_t p = 0; p + 1 < new_node.path_x.size(); p++) {
                cv::line(bg,
                    cv::Point(to_px(new_node.path_x[p]), to_py(new_node.path_y[p])),
                    cv::Point(to_px(new_node.path_x[p+1]), to_py(new_node.path_y[p+1])),
                    cv::Scalar(0, 255, 0), 1);
            }
        }

        if (iter % 10 == 0) {
            cv::imshow("rrt_star_rs", bg);
            cv::waitKey(1);
        }

        // Check if we can connect to goal via RS curve
        float goal_rs_len = rs_distance(new_node.x, new_node.y, new_node.yaw,
                                         goal_x_, goal_y_, goal_yaw_);
        if (goal_rs_len < FLT_MAX) {
            float goal_cost = new_node.cost + goal_rs_len;
            if (goal_cost < best_goal_cost) {
                // Verify collision-free path to goal
                if (check_collision_host(new_node.x, new_node.y, new_node.yaw,
                                          goal_x_, goal_y_, goal_yaw_)) {
                    best_goal_cost = goal_cost;
                    best_goal_idx = new_idx;
                }
            }
        }
    }

    // Extract path
    std::vector<Node> path;
    if (best_goal_idx >= 0) {
        std::cout << "Found path! Cost: " << best_goal_cost << std::endl;

        // Add goal node with RS path from best_goal_idx
        Node goal_node(goal_x_, goal_y_, goal_yaw_);
        goal_node.cost = best_goal_cost;
        goal_node.parent_idx = best_goal_idx;
        generate_rs_path_points(
            node_list_[best_goal_idx].x, node_list_[best_goal_idx].y, node_list_[best_goal_idx].yaw,
            goal_x_, goal_y_, goal_yaw_,
            goal_node.path_x, goal_node.path_y, goal_node.path_yaw);
        node_list_.push_back(goal_node);
        int goal_idx = (int)node_list_.size() - 1;

        // Trace back
        std::vector<int> idx_path;
        int idx = goal_idx;
        while (idx >= 0) {
            idx_path.push_back(idx);
            idx = node_list_[idx].parent_idx;
        }
        std::reverse(idx_path.begin(), idx_path.end());

        // Draw final path (thick magenta) and collect path nodes
        for (size_t k = 0; k < idx_path.size(); k++) {
            int ni = idx_path[k];
            path.push_back(node_list_[ni]);

            const auto& nx = node_list_[ni].path_x;
            const auto& ny = node_list_[ni].path_y;
            if (nx.size() >= 2) {
                for (size_t p = 0; p + 1 < nx.size(); p++) {
                    cv::line(bg,
                        cv::Point(to_px(nx[p]), to_py(ny[p])),
                        cv::Point(to_px(nx[p+1]), to_py(ny[p+1])),
                        cv::Scalar(255, 0, 255), 3);
                }
            }
        }

        // Redraw start and goal on top
        {
            cv::Point sp(to_px(start_x_), to_py(start_y_));
            cv::circle(bg, sp, 8, cv::Scalar(0, 0, 255), -1);
            float arrow_len = 2.0f * img_reso;
            cv::Point ep(sp.x + (int)(arrow_len * cosf(start_yaw_)),
                         sp.y + (int)(arrow_len * sinf(start_yaw_)));
            cv::arrowedLine(bg, sp, ep, cv::Scalar(0, 0, 255), 3);
        }
        {
            cv::Point gp(to_px(goal_x_), to_py(goal_y_));
            cv::circle(bg, gp, 8, cv::Scalar(255, 0, 0), -1);
            float arrow_len = 2.0f * img_reso;
            cv::Point ep(gp.x + (int)(arrow_len * cosf(goal_yaw_)),
                         gp.y + (int)(arrow_len * sinf(goal_yaw_)));
            cv::arrowedLine(bg, gp, ep, cv::Scalar(255, 0, 0), 3);
        }

        cv::imshow("rrt_star_rs", bg);
        cv::waitKey(0);
    } else {
        std::cout << "No path found within " << max_iter_ << " iterations." << std::endl;
        cv::imshow("rrt_star_rs", bg);
        cv::waitKey(0);
    }

    return path;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::vector<Obstacle> obstacles = {
        {15.0f, 15.0f, 3.0f},
        {10.0f, 25.0f, 2.0f},
        {25.0f, 10.0f, 2.0f},
        {20.0f, 20.0f, 2.0f},
        { 5.0f, 15.0f, 1.0f}
    };

    float start_x = 0.0f, start_y = 0.0f, start_yaw = 0.0f;
    float goal_x  = 30.0f, goal_y = 30.0f, goal_yaw = 0.0f;
    float rand_min = -5.0f, rand_max = 40.0f;
    float expand_dis = 5.0f;
    int   goal_sample_rate = 20;
    int   max_iter = 500;
    float connect_circle_dist = 50.0f;
    float curvature = 1.0f / 5.0f;  // radius = 5.0

    RRTStarRS rrt_star_rs(
        start_x, start_y, start_yaw,
        goal_x, goal_y, goal_yaw,
        obstacles, rand_min, rand_max,
        expand_dis,
        goal_sample_rate, max_iter,
        connect_circle_dist,
        curvature);

    std::vector<Node> path = rrt_star_rs.planning();

    if (!path.empty()) {
        std::cout << "Path nodes: " << path.size() << std::endl;
        for (const auto& n : path) {
            std::cout << "  (" << n.x << ", " << n.y << ", yaw=" << n.yaw
                      << ") cost=" << n.cost << std::endl;
        }
    }

    return 0;
}
