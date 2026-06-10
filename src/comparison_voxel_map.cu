/*************************************************************************
    3D Voxel Map (log-odds occupancy) comparison: CPU 64x64x16 grid +
    16-channel x 256-azimuth scans vs CUDA 256x256x32 grid + 64-channel x
    1024-azimuth scans. Same analytic scene (ground + boxes + cylinders)
    and same spinning-LiDAR sensor pose per frame. Each ray casts a 3D DDA
    walk; traversed cells get a free-space log-odds decrement, the final
    cell (if it was a hit) gets an occupied log-odds increment.

    GPU log-odds updates use atomicAdd into a float voxel grid; the kernel
    is 1 ray per thread.

    Panels: left  = CPU top-down log-odds projection (max over Z)
            right = CUDA top-down log-odds projection (max over Z)

    Headline metric is per-voxel-update throughput.
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include "cuda_check.cuh"

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------
constexpr float PI = 3.14159265358979323846f;
constexpr float DEG = PI / 180.0f;
constexpr float MAX_RANGE = 40.0f;
constexpr float VERT_MIN = -20.0f * DEG;
constexpr float VERT_MAX = 12.0f * DEG;

constexpr float WORLD_X = 40.0f;
constexpr float WORLD_Y = 40.0f;
constexpr float WORLD_Z = 8.0f;

constexpr int   CPU_NX = 64;
constexpr int   CPU_NY = 64;
constexpr int   CPU_NZ = 16;
constexpr float CPU_RES_X = WORLD_X / CPU_NX;
constexpr float CPU_RES_Y = WORLD_Y / CPU_NY;
constexpr float CPU_RES_Z = WORLD_Z / CPU_NZ;

constexpr int   GPU_NX = 256;
constexpr int   GPU_NY = 256;
constexpr int   GPU_NZ = 32;
constexpr float GPU_RES_X = WORLD_X / GPU_NX;
constexpr float GPU_RES_Y = WORLD_Y / GPU_NY;
constexpr float GPU_RES_Z = WORLD_Z / GPU_NZ;

constexpr int CPU_CHANNELS = 16;
constexpr int CPU_AZIMUTH  = 256;
constexpr int GPU_CHANNELS = 64;
constexpr int GPU_AZIMUTH  = 1024;

constexpr float L_OCC  =  0.85f;
constexpr float L_FREE = -0.40f;
constexpr float L_MIN  = -4.0f;
constexpr float L_MAX  =  4.0f;

constexpr int SIM_FRAMES = 90;
constexpr int PANEL_W = 600;
constexpr int PANEL_H = 600;

// -------------------------------------------------------------------------
// Scene primitives (matches the style of comparison_lidar3d_sim.cu)
// -------------------------------------------------------------------------
enum PrimitiveType { PRIM_AABB = 0, PRIM_CYLINDER = 1 };

struct Vec3 { float x, y, z; };

struct Primitive {
    int type;
    Vec3 c;     // center
    Vec3 h;     // half-extent (AABB)
    float r;    // radius (cylinder)
    float z0;   // cylinder base z
    float z1;   // cylinder top z
};

__host__ __device__ static Vec3 v3(float x, float y, float z) {
    Vec3 v; v.x = x; v.y = y; v.z = z; return v;
}

__host__ __device__ static bool ray_aabb(const Primitive& p, Vec3 o, Vec3 d,
                                         float max_range, float& t_hit) {
    float tmin = 0.0f, tmax = max_range;
    float mn[3] = {p.c.x - p.h.x, p.c.y - p.h.y, p.c.z - p.h.z};
    float mx[3] = {p.c.x + p.h.x, p.c.y + p.h.y, p.c.z + p.h.z};
    float oo[3] = {o.x, o.y, o.z};
    float dd[3] = {d.x, d.y, d.z};
    for (int a = 0; a < 3; a++) {
        if (fabsf(dd[a]) < 1e-7f) {
            if (oo[a] < mn[a] || oo[a] > mx[a]) return false;
        } else {
            float inv = 1.0f / dd[a];
            float t0 = (mn[a] - oo[a]) * inv;
            float t1 = (mx[a] - oo[a]) * inv;
            if (t0 > t1) { float t = t0; t0 = t1; t1 = t; }
            tmin = fmaxf(tmin, t0);
            tmax = fminf(tmax, t1);
            if (tmin > tmax) return false;
        }
    }
    t_hit = tmin > 1e-4f ? tmin : tmax;
    return t_hit > 1e-4f && t_hit < max_range;
}

__host__ __device__ static bool ray_cylinder(const Primitive& p, Vec3 o, Vec3 d,
                                             float max_range, float& t_hit) {
    float ox = o.x - p.c.x, oy = o.y - p.c.y;
    float a = d.x * d.x + d.y * d.y;
    float b = 2.0f * (ox * d.x + oy * d.y);
    float c = ox * ox + oy * oy - p.r * p.r;
    float best = max_range;
    bool hit = false;
    if (a > 1e-8f) {
        float disc = b * b - 4.0f * a * c;
        if (disc >= 0.0f) {
            float sd = sqrtf(disc);
            float inv = 0.5f / a;
            float ts[2] = {(-b - sd) * inv, (-b + sd) * inv};
            for (int i = 0; i < 2; i++) {
                float t = ts[i];
                if (t < 1e-4f || t >= best) continue;
                float zh = o.z + t * d.z;
                if (zh >= p.z0 && zh <= p.z1) { best = t; hit = true; }
            }
        }
    }
    t_hit = best;
    return hit;
}

// Ground plane at z=0; returns t of intersection if d.z<0 and o.z>0.
__host__ __device__ static bool ray_ground(Vec3 o, Vec3 d,
                                           float max_range, float& t_hit) {
    if (d.z >= -1e-7f || o.z <= 0.0f) return false;
    float t = -o.z / d.z;
    if (t < 1e-4f || t >= max_range) return false;
    t_hit = t;
    return true;
}

__host__ __device__ static float ray_scene(const Primitive* prims, int np,
                                           Vec3 o, Vec3 d, float max_range) {
    float best = max_range;
    float t;
    if (ray_ground(o, d, best, t)) best = t;
    for (int i = 0; i < np; i++) {
        if (prims[i].type == PRIM_AABB) {
            if (ray_aabb(prims[i], o, d, best, t)) best = t;
        } else {
            if (ray_cylinder(prims[i], o, d, best, t)) best = t;
        }
    }
    return best;
}

static std::vector<Primitive> build_scene() {
    std::vector<Primitive> P;
    auto box = [&](Vec3 c, Vec3 h) {
        Primitive p{PRIM_AABB, c, h, 0.0f, 0.0f, 0.0f};
        P.push_back(p);
    };
    auto cyl = [&](float x, float y, float r, float z0, float z1) {
        Primitive p{PRIM_CYLINDER, v3(x, y, 0.0f), v3(0,0,0), r, z0, z1};
        P.push_back(p);
    };
    // walls (4 thin AABBs around the perimeter, leaving a doorway)
    box(v3(WORLD_X * 0.5f, 0.6f, 1.0f), v3(WORLD_X * 0.5f - 0.2f, 0.4f, 1.0f));
    box(v3(WORLD_X * 0.5f, WORLD_Y - 0.6f, 1.0f), v3(WORLD_X * 0.5f - 0.2f, 0.4f, 1.0f));
    box(v3(0.6f, WORLD_Y * 0.5f, 1.0f), v3(0.4f, WORLD_Y * 0.5f - 0.2f, 1.0f));
    box(v3(WORLD_X - 0.6f, WORLD_Y * 0.5f, 1.0f), v3(0.4f, WORLD_Y * 0.5f - 0.2f, 1.0f));
    // interior boxes
    box(v3(13.0f, 14.0f, 1.5f), v3(2.0f, 2.0f, 1.5f));
    box(v3(27.0f, 12.0f, 1.0f), v3(1.4f, 1.4f, 1.0f));
    box(v3(20.0f, 24.0f, 2.0f), v3(3.0f, 1.2f, 2.0f));
    box(v3(30.0f, 28.0f, 0.7f), v3(1.0f, 3.5f, 0.7f));
    // cylinders (pillars)
    cyl(8.0f,  8.0f, 0.6f, 0.0f, 3.0f);
    cyl(33.0f, 7.0f, 0.5f, 0.0f, 3.2f);
    cyl(7.0f, 30.0f, 0.55f, 0.0f, 2.8f);
    cyl(34.0f, 33.0f, 0.7f, 0.0f, 3.4f);
    return P;
}

// -------------------------------------------------------------------------
// 3D DDA traversal: apply free-space log-odds to traversed cells, occupied
// to the hit cell if t_hit < max_range.
// -------------------------------------------------------------------------
static void cpu_ray_update(float* grid, int NX, int NY, int NZ,
                           float rx, float ry, float rz,
                           Vec3 o, Vec3 d, float t_hit, float max_range) {
    bool hit = t_hit < max_range - 1e-3f;
    float t_end = hit ? t_hit : max_range;

    float fx = o.x / rx;
    float fy = o.y / ry;
    float fz = o.z / rz;
    int gx = static_cast<int>(std::floor(fx));
    int gy = static_cast<int>(std::floor(fy));
    int gz = static_cast<int>(std::floor(fz));
    int step_x = (d.x > 0.0f) ? 1 : -1;
    int step_y = (d.y > 0.0f) ? 1 : -1;
    int step_z = (d.z > 0.0f) ? 1 : -1;
    float inv_dx = (std::fabs(d.x) > 1e-7f) ? 1.0f / std::fabs(d.x) : 1e30f;
    float inv_dy = (std::fabs(d.y) > 1e-7f) ? 1.0f / std::fabs(d.y) : 1e30f;
    float inv_dz = (std::fabs(d.z) > 1e-7f) ? 1.0f / std::fabs(d.z) : 1e30f;
    float t_max_x = (d.x > 0.0f) ? (gx + 1 - fx) * rx * inv_dx : (fx - gx) * rx * inv_dx;
    float t_max_y = (d.y > 0.0f) ? (gy + 1 - fy) * ry * inv_dy : (fy - gy) * ry * inv_dy;
    float t_max_z = (d.z > 0.0f) ? (gz + 1 - fz) * rz * inv_dz : (fz - gz) * rz * inv_dz;
    float dt_x = rx * inv_dx;
    float dt_y = ry * inv_dy;
    float dt_z = rz * inv_dz;

    int max_iter = NX + NY + NZ + 8;
    for (int it = 0; it < max_iter; it++) {
        if (gx >= 0 && gx < NX && gy >= 0 && gy < NY && gz >= 0 && gz < NZ) {
            int idx = (gz * NY + gy) * NX + gx;
            float v = grid[idx] + L_FREE;
            if (v < L_MIN) v = L_MIN;
            grid[idx] = v;
        }
        float t_next = std::min(std::min(t_max_x, t_max_y), t_max_z);
        if (t_next >= t_end) break;
        if (t_max_x <= t_max_y && t_max_x <= t_max_z) { gx += step_x; t_max_x += dt_x; }
        else if (t_max_y <= t_max_z) { gy += step_y; t_max_y += dt_y; }
        else { gz += step_z; t_max_z += dt_z; }
        if (gx < 0 || gx >= NX || gy < 0 || gy >= NY || gz < 0 || gz >= NZ) break;
    }

    if (hit) {
        float hx = o.x + t_hit * d.x;
        float hy = o.y + t_hit * d.y;
        float hz = o.z + t_hit * d.z;
        int hgx = static_cast<int>(std::floor(hx / rx));
        int hgy = static_cast<int>(std::floor(hy / ry));
        int hgz = static_cast<int>(std::floor(hz / rz));
        if (hgx >= 0 && hgx < NX && hgy >= 0 && hgy < NY && hgz >= 0 && hgz < NZ) {
            int idx = (hgz * NY + hgy) * NX + hgx;
            float v = grid[idx] - L_FREE + L_OCC;  // undo the free, add occ
            if (v > L_MAX) v = L_MAX;
            if (v < L_MIN) v = L_MIN;
            grid[idx] = v;
        }
    }
}

// -------------------------------------------------------------------------
// GPU kernels
// -------------------------------------------------------------------------
__device__ static void atomic_add_clamped(float* addr, float delta) {
    // Avoid drifting outside [L_MIN, L_MAX] by reading-then-CAS-style clamp.
    float old = atomicAdd(addr, delta);
    float updated = old + delta;
    if (updated > L_MAX) atomicAdd(addr, L_MAX - updated);
    else if (updated < L_MIN) atomicAdd(addr, L_MIN - updated);
}

__global__ void clear_grid_kernel(float* grid, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grid[i] = 0.0f;
}

__global__ void raycast_voxel_kernel(const Primitive* __restrict__ prims, int np,
                                     float* __restrict__ grid,
                                     int NX, int NY, int NZ,
                                     float rx, float ry, float rz,
                                     float sx, float sy, float sz,
                                     float yaw,
                                     int channels, int azimuth,
                                     float vmin, float vmax,
                                     float max_range) {
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * azimuth;
    if (rid >= total) return;
    int ch = rid / azimuth;
    int az = rid % azimuth;
    float v_step = (channels > 1) ? (vmax - vmin) / (channels - 1) : 0.0f;
    float pitch = vmin + ch * v_step;
    float azim = yaw + (az * 2.0f * PI / azimuth);
    float cp = cosf(pitch);
    Vec3 d = v3(cp * cosf(azim), cp * sinf(azim), sinf(pitch));
    Vec3 o = v3(sx, sy, sz);

    float t_hit = ray_scene(prims, np, o, d, max_range);
    bool hit = t_hit < max_range - 1e-3f;
    float t_end = hit ? t_hit : max_range;

    float fx = o.x / rx;
    float fy = o.y / ry;
    float fz = o.z / rz;
    int gx = static_cast<int>(floorf(fx));
    int gy = static_cast<int>(floorf(fy));
    int gz = static_cast<int>(floorf(fz));
    int step_x = (d.x > 0.0f) ? 1 : -1;
    int step_y = (d.y > 0.0f) ? 1 : -1;
    int step_z = (d.z > 0.0f) ? 1 : -1;
    float inv_dx = (fabsf(d.x) > 1e-7f) ? 1.0f / fabsf(d.x) : 1e30f;
    float inv_dy = (fabsf(d.y) > 1e-7f) ? 1.0f / fabsf(d.y) : 1e30f;
    float inv_dz = (fabsf(d.z) > 1e-7f) ? 1.0f / fabsf(d.z) : 1e30f;
    float t_max_x = (d.x > 0.0f) ? (gx + 1 - fx) * rx * inv_dx : (fx - gx) * rx * inv_dx;
    float t_max_y = (d.y > 0.0f) ? (gy + 1 - fy) * ry * inv_dy : (fy - gy) * ry * inv_dy;
    float t_max_z = (d.z > 0.0f) ? (gz + 1 - fz) * rz * inv_dz : (fz - gz) * rz * inv_dz;
    float dt_x = rx * inv_dx;
    float dt_y = ry * inv_dy;
    float dt_z = rz * inv_dz;

    int max_iter = NX + NY + NZ + 8;
    for (int it = 0; it < max_iter; it++) {
        if (gx >= 0 && gx < NX && gy >= 0 && gy < NY && gz >= 0 && gz < NZ) {
            int idx = (gz * NY + gy) * NX + gx;
            atomic_add_clamped(&grid[idx], L_FREE);
        }
        float t_next = fminf(fminf(t_max_x, t_max_y), t_max_z);
        if (t_next >= t_end) break;
        if (t_max_x <= t_max_y && t_max_x <= t_max_z) { gx += step_x; t_max_x += dt_x; }
        else if (t_max_y <= t_max_z) { gy += step_y; t_max_y += dt_y; }
        else { gz += step_z; t_max_z += dt_z; }
        if (gx < 0 || gx >= NX || gy < 0 || gy >= NY || gz < 0 || gz >= NZ) break;
    }
    if (hit) {
        float hx = o.x + t_hit * d.x;
        float hy = o.y + t_hit * d.y;
        float hz = o.z + t_hit * d.z;
        int hgx = static_cast<int>(floorf(hx / rx));
        int hgy = static_cast<int>(floorf(hy / ry));
        int hgz = static_cast<int>(floorf(hz / rz));
        if (hgx >= 0 && hgx < NX && hgy >= 0 && hgy < NY && hgz >= 0 && hgz < NZ) {
            int idx = (hgz * NY + hgy) * NX + hgx;
            atomic_add_clamped(&grid[idx], L_OCC - L_FREE);
        }
    }
}

// -------------------------------------------------------------------------
// Rendering
// -------------------------------------------------------------------------
static cv::Vec3b log_odds_color(float v) {
    if (v >= L_MAX * 0.5f) {
        return cv::Vec3b(40, 40, 220);  // strong occupied -> red-ish (BGR)
    }
    if (v <= L_MIN * 0.5f) {
        return cv::Vec3b(220, 220, 220);  // strong free -> light grey
    }
    float t = (v - L_MIN) / (L_MAX - L_MIN);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    unsigned char g = static_cast<unsigned char>(180 + 75 * (1.0f - t));
    unsigned char r = static_cast<unsigned char>(40 + 120 * t);
    unsigned char b = static_cast<unsigned char>(40 + 120 * t);
    return cv::Vec3b(b, g, r);
}

static void draw_topdown(cv::Mat& panel, const std::vector<float>& grid,
                         int NX, int NY, int NZ) {
    panel.create(PANEL_H, PANEL_W, CV_8UC3);
    panel.setTo(cv::Scalar(255, 255, 255));
    for (int py = 0; py < PANEL_H; py++) {
        float wy = (PANEL_H - 1 - py) * (WORLD_Y / PANEL_H);
        int gy = std::min(NY - 1, std::max(0, static_cast<int>(wy * NY / WORLD_Y)));
        for (int px = 0; px < PANEL_W; px++) {
            float wx = px * (WORLD_X / PANEL_W);
            int gx = std::min(NX - 1, std::max(0, static_cast<int>(wx * NX / WORLD_X)));
            float vmax = L_MIN;
            for (int gz = 0; gz < NZ; gz++) {
                float v = grid[(gz * NY + gy) * NX + gx];
                if (v > vmax) vmax = v;
            }
            panel.at<cv::Vec3b>(py, px) = log_odds_color(vmax);
        }
    }
}

static void draw_sensor_pose(cv::Mat& panel, float sx, float sy) {
    float ux = static_cast<float>(PANEL_W) / WORLD_X;
    float uy = static_cast<float>(PANEL_H) / WORLD_Y;
    int px = static_cast<int>(sx * ux);
    int py = PANEL_H - 1 - static_cast<int>(sy * uy);
    cv::circle(panel, cv::Point(px, py), 6, cv::Scalar(0, 100, 200), -1, cv::LINE_AA);
    cv::circle(panel, cv::Point(px, py), 6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

static void draw_label(cv::Mat& panel, const std::string& text, int y) {
    cv::putText(panel, text, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
    cv::putText(panel, text, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

// -------------------------------------------------------------------------
// Sensor path
// -------------------------------------------------------------------------
static void sensor_pose(int frame, float& sx, float& sy, float& sz, float& yaw) {
    float u = static_cast<float>(frame) / SIM_FRAMES;
    float ang = 2.0f * PI * u;
    float cx = WORLD_X * 0.5f;
    float cy = WORLD_Y * 0.5f;
    float r = std::min(WORLD_X, WORLD_Y) * 0.30f;
    sx = cx + r * std::cos(ang);
    sy = cy + r * std::sin(ang * 1.2f);
    sz = 1.2f + 0.3f * std::sin(ang * 2.0f);
    yaw = ang * 0.8f;
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    std::printf("3D voxel map comparison: CPU %dx%dx%d grid + %d ch x %d az "
                "vs CUDA %dx%dx%d grid + %d ch x %d az\n",
                CPU_NX, CPU_NY, CPU_NZ, CPU_CHANNELS, CPU_AZIMUTH,
                GPU_NX, GPU_NY, GPU_NZ, GPU_CHANNELS, GPU_AZIMUTH);
    std::vector<Primitive> scene = build_scene();
    int np = static_cast<int>(scene.size());

    // CPU grid (host-only)
    std::vector<float> cpu_grid(CPU_NX * CPU_NY * CPU_NZ, 0.0f);

    // GPU resources
    Primitive* d_prims = nullptr;
    CUDA_CHECK(cudaMalloc(&d_prims, np * sizeof(Primitive)));
    CUDA_CHECK(cudaMemcpy(d_prims, scene.data(), np * sizeof(Primitive),
                          cudaMemcpyHostToDevice));
    int gpu_total = GPU_NX * GPU_NY * GPU_NZ;
    float* d_grid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid, gpu_total * sizeof(float)));
    {
        int blk = 256, grd = (gpu_total + blk - 1) / blk;
        clear_grid_kernel<<<grd, blk>>>(d_grid, gpu_total);
    }
    std::vector<float> gpu_grid(gpu_total, 0.0f);

    cv::VideoWriter video("gif/comparison_voxel_map.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
                          cv::Size(PANEL_W * 2, PANEL_H));

    double cpu_ms_sum = 0.0, gpu_ms_sum = 0.0;
    long   cpu_updates_sum = 0, gpu_updates_sum = 0;
    int    timed_frames = 0;

    for (int f = 0; f < SIM_FRAMES; f++) {
        float sx, sy, sz, yaw;
        sensor_pose(f, sx, sy, sz, yaw);
        Vec3 o = v3(sx, sy, sz);

        // CPU pass
        auto cpu_t0 = std::chrono::high_resolution_clock::now();
        float v_step_cpu = (CPU_CHANNELS > 1) ? (VERT_MAX - VERT_MIN) / (CPU_CHANNELS - 1) : 0.0f;
        for (int ch = 0; ch < CPU_CHANNELS; ch++) {
            float pitch = VERT_MIN + ch * v_step_cpu;
            float cp = std::cos(pitch);
            for (int az = 0; az < CPU_AZIMUTH; az++) {
                float azim = yaw + (az * 2.0f * PI / CPU_AZIMUTH);
                Vec3 d = v3(cp * std::cos(azim), cp * std::sin(azim), std::sin(pitch));
                float t_hit = ray_scene(scene.data(), np, o, d, MAX_RANGE);
                cpu_ray_update(cpu_grid.data(), CPU_NX, CPU_NY, CPU_NZ,
                               CPU_RES_X, CPU_RES_Y, CPU_RES_Z,
                               o, d, t_hit, MAX_RANGE);
            }
        }
        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();
        long cpu_updates = static_cast<long>(CPU_CHANNELS) * CPU_AZIMUTH;

        // GPU pass
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        int rays = GPU_CHANNELS * GPU_AZIMUTH;
        int blk = 256, grd = (rays + blk - 1) / blk;
        cudaEventRecord(e0);
        raycast_voxel_kernel<<<grd, blk>>>(d_prims, np, d_grid,
                                           GPU_NX, GPU_NY, GPU_NZ,
                                           GPU_RES_X, GPU_RES_Y, GPU_RES_Z,
                                           sx, sy, sz, yaw,
                                           GPU_CHANNELS, GPU_AZIMUTH,
                                           VERT_MIN, VERT_MAX, MAX_RANGE);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        long gpu_updates = static_cast<long>(GPU_CHANNELS) * GPU_AZIMUTH;

        if (f >= 5) {
            cpu_ms_sum += cpu_ms;
            gpu_ms_sum += gpu_ms;
            cpu_updates_sum += cpu_updates;
            gpu_updates_sum += gpu_updates;
            timed_frames++;
        }

        CUDA_CHECK(cudaMemcpy(gpu_grid.data(), d_grid,
                              gpu_total * sizeof(float), cudaMemcpyDeviceToHost));

        cv::Mat left, right;
        draw_topdown(left, cpu_grid, CPU_NX, CPU_NY, CPU_NZ);
        draw_topdown(right, gpu_grid, GPU_NX, GPU_NY, GPU_NZ);
        draw_sensor_pose(left, sx, sy);
        draw_sensor_pose(right, sx, sy);

        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "CPU  %dx%dx%d voxels  %dx%d rays  %.1f ms",
                      CPU_NX, CPU_NY, CPU_NZ, CPU_CHANNELS, CPU_AZIMUTH, cpu_ms);
        draw_label(left, buf, 28);
        std::snprintf(buf, sizeof(buf),
                      "GPU  %dx%dx%d voxels  %dx%d rays  %.2f ms",
                      GPU_NX, GPU_NY, GPU_NZ, GPU_CHANNELS, GPU_AZIMUTH, gpu_ms);
        draw_label(right, buf, 28);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        video.write(combined);
    }

    video.release();
    cudaFree(d_prims);
    cudaFree(d_grid);

    if (timed_frames > 0) {
        double cpu_ms = cpu_ms_sum / timed_frames;
        double gpu_ms = gpu_ms_sum / timed_frames;
        double cpu_per_ray_us = cpu_ms * 1.0e3 / (CPU_CHANNELS * CPU_AZIMUTH);
        double gpu_per_ray_us = gpu_ms * 1.0e3 / (GPU_CHANNELS * GPU_AZIMUTH);
        std::printf("Avg CPU %.2f ms / scan (%d rays, %d-voxel grid)\n"
                    "Avg GPU %.3f ms / scan (%d rays, %d-voxel grid)\n"
                    "Per-ray throughput: GPU %.4f us/ray vs CPU %.3f us/ray "
                    "(%.0fx faster per ray)\n",
                    cpu_ms, CPU_CHANNELS * CPU_AZIMUTH, CPU_NX * CPU_NY * CPU_NZ,
                    gpu_ms, GPU_CHANNELS * GPU_AZIMUTH, GPU_NX * GPU_NY * GPU_NZ,
                    gpu_per_ray_us, cpu_per_ray_us,
                    cpu_per_ray_us / gpu_per_ray_us);
    }

    std::system("ffmpeg -y -i gif/comparison_voxel_map.avi "
                "-vf 'fps=15,scale=900:-1:flags=lanczos' -loop 0 "
                "gif/comparison_voxel_map.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_voxel_map.gif" << std::endl;
    return 0;
}
