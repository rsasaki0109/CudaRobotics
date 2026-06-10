/*************************************************************************
    3D LiDAR Simulator comparison: CPU sparse multi-ring scan vs CUDA dense
    scan. The scene uses analytic primitives so the first version stays small
    and deterministic while still matching the robotics pattern: one ray maps
    to one CUDA thread, producing a dense point cloud and range image.
 ************************************************************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include "cuda_check.cuh"

    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr float PI = 3.14159265358979323846f;
constexpr float DEG = PI / 180.0f;
constexpr float MAX_RANGE = 80.0f;
constexpr float VERT_MIN = -25.0f * DEG;
constexpr float VERT_MAX = 15.0f * DEG;

constexpr int CPU_CHANNELS = 16;
constexpr int CPU_AZIMUTH = 512;
constexpr int GPU_CHANNELS = 64;
constexpr int GPU_AZIMUTH = 2048;
constexpr int STRESS_CHANNELS = 128;
constexpr int STRESS_AZIMUTH = 4096;
constexpr int MAX_RAYS = STRESS_CHANNELS * STRESS_AZIMUTH;

constexpr int PANEL_W = 480;
constexpr int PANEL_H = 360;
constexpr int SIM_FRAMES = 72;

enum PrimitiveType {
    PRIM_AABB = 0,
    PRIM_CYLINDER = 1
};

struct Vec3 {
    float x, y, z;
};

struct Primitive {
    int type;
    int label;
    Vec3 c;
    Vec3 h;
    float radius;
};

__host__ __device__ static Vec3 v3(float x, float y, float z) {
    Vec3 v; v.x = x; v.y = y; v.z = z; return v;
}

__host__ __device__ static Vec3 vadd(Vec3 a, Vec3 b) {
    return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ static Vec3 vmul(Vec3 a, float s) {
    return v3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ static bool intersect_aabb(const Primitive& p, Vec3 o, Vec3 d,
                                               float max_range, float& t_hit) {
    float tmin = 0.0f;
    float tmax = max_range;
    float mn[3] = {p.c.x - p.h.x, p.c.y - p.h.y, p.c.z - p.h.z};
    float mx[3] = {p.c.x + p.h.x, p.c.y + p.h.y, p.c.z + p.h.z};
    float oo[3] = {o.x, o.y, o.z};
    float dd[3] = {d.x, d.y, d.z};
    for (int axis = 0; axis < 3; axis++) {
        if (fabsf(dd[axis]) < 1e-7f) {
            if (oo[axis] < mn[axis] || oo[axis] > mx[axis]) return false;
        } else {
            float inv = 1.0f / dd[axis];
            float t0 = (mn[axis] - oo[axis]) * inv;
            float t1 = (mx[axis] - oo[axis]) * inv;
            if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
            tmin = fmaxf(tmin, t0);
            tmax = fminf(tmax, t1);
            if (tmin > tmax) return false;
        }
    }
    t_hit = tmin > 1e-4f ? tmin : tmax;
    return t_hit > 1e-4f && t_hit < max_range;
}

__host__ __device__ static bool intersect_cylinder(const Primitive& p, Vec3 o, Vec3 d,
                                                   float max_range, float& t_hit) {
    bool hit = false;
    float best = max_range;
    float ox = o.x - p.c.x;
    float oy = o.y - p.c.y;
    float a = d.x * d.x + d.y * d.y;
    float b = 2.0f * (ox * d.x + oy * d.y);
    float c = ox * ox + oy * oy - p.radius * p.radius;
    if (a > 1e-8f) {
        float disc = b * b - 4.0f * a * c;
        if (disc >= 0.0f) {
            float sdisc = sqrtf(disc);
            float inv = 0.5f / a;
            float ts[2] = {(-b - sdisc) * inv, (-b + sdisc) * inv};
            for (int i = 0; i < 2; i++) {
                float t = ts[i];
                float z = o.z + t * d.z;
                if (t > 1e-4f && t < best && z >= p.c.z - p.h.z && z <= p.c.z + p.h.z) {
                    best = t;
                    hit = true;
                }
            }
        }
    }
    if (fabsf(d.z) > 1e-7f) {
        float caps[2] = {p.c.z - p.h.z, p.c.z + p.h.z};
        for (int i = 0; i < 2; i++) {
            float t = (caps[i] - o.z) / d.z;
            float x = o.x + t * d.x - p.c.x;
            float y = o.y + t * d.y - p.c.y;
            if (t > 1e-4f && t < best && x * x + y * y <= p.radius * p.radius) {
                best = t;
                hit = true;
            }
        }
    }
    t_hit = best;
    return hit;
}

__host__ __device__ static void raycast_scene(const Primitive* prims, int n_prims,
                                              Vec3 o, Vec3 d, float max_range,
                                              float& range, Vec3& hit, int& label) {
    range = max_range;
    label = 0;
    if (d.z < -1e-6f) {
        float t = -o.z / d.z;
        if (t > 1e-4f && t < range) {
            range = t;
            label = 1;
        }
    }
    for (int i = 0; i < n_prims; i++) {
        float t = max_range;
        bool ok = prims[i].type == PRIM_AABB
            ? intersect_aabb(prims[i], o, d, max_range, t)
            : intersect_cylinder(prims[i], o, d, max_range, t);
        if (ok && t < range) {
            range = t;
            label = prims[i].label;
        }
    }
    hit = vadd(o, vmul(d, range));
}

__host__ __device__ static Vec3 ray_dir(int channel, int az_idx, int channels,
                                        int azimuth_bins, float yaw) {
    float vstep = (channels > 1) ? (VERT_MAX - VERT_MIN) / (channels - 1) : 0.0f;
    float va = VERT_MIN + channel * vstep;
    float ha = yaw + 2.0f * PI * az_idx / azimuth_bins;
    float cv = cosf(va);
    return v3(cv * cosf(ha), cv * sinf(ha), sinf(va));
}

__global__ void lidar3d_kernel(const Primitive* __restrict__ prims, int n_prims,
                               Vec3 sensor, float yaw, int channels,
                               int azimuth_bins, float max_range,
                               float* __restrict__ ranges,
                               float* __restrict__ xs,
                               float* __restrict__ ys,
                               float* __restrict__ zs,
                               int* __restrict__ labels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = channels * azimuth_bins;
    if (idx >= n) return;
    int channel = idx / azimuth_bins;
    int az_idx = idx - channel * azimuth_bins;
    Vec3 d = ray_dir(channel, az_idx, channels, azimuth_bins, yaw);
    Vec3 hit;
    float range;
    int label;
    raycast_scene(prims, n_prims, sensor, d, max_range, range, hit, label);
    ranges[idx] = range;
    xs[idx] = hit.x;
    ys[idx] = hit.y;
    zs[idx] = hit.z;
    labels[idx] = label;
}

static Primitive aabb(float x, float y, float z, float hx, float hy, float hz, int label) {
    Primitive p;
    p.type = PRIM_AABB; p.label = label; p.c = v3(x, y, z); p.h = v3(hx, hy, hz); p.radius = 0.0f;
    return p;
}

static Primitive cylinder(float x, float y, float z, float radius, float hz, int label) {
    Primitive p;
    p.type = PRIM_CYLINDER; p.label = label; p.c = v3(x, y, z); p.h = v3(0.0f, 0.0f, hz); p.radius = radius;
    return p;
}

static std::vector<Primitive> build_scene() {
    std::vector<Primitive> p;
    p.push_back(aabb(0.0f, -30.0f, 2.5f, 42.0f, 0.35f, 2.5f, 2));
    p.push_back(aabb(0.0f,  30.0f, 2.5f, 42.0f, 0.35f, 2.5f, 2));
    p.push_back(aabb(-42.0f, 0.0f, 2.5f, 0.35f, 30.0f, 2.5f, 2));
    p.push_back(aabb( 42.0f, 0.0f, 2.5f, 0.35f, 30.0f, 2.5f, 2));
    p.push_back(aabb(-18.0f, -10.0f, 4.0f, 5.5f, 8.0f, 4.0f, 3));
    p.push_back(aabb( 17.0f,  -8.0f, 3.5f, 7.0f, 5.0f, 3.5f, 3));
    p.push_back(aabb(-10.0f,  15.0f, 5.0f, 6.0f, 4.5f, 5.0f, 3));
    p.push_back(aabb( 24.0f,  15.0f, 6.0f, 4.0f, 8.0f, 6.0f, 3));
    p.push_back(aabb( -4.0f, -22.0f, 1.1f, 2.4f, 1.0f, 1.1f, 4));
    p.push_back(aabb(  9.0f,  22.0f, 1.1f, 2.4f, 1.0f, 1.1f, 4));
    p.push_back(aabb( 30.0f,  -3.0f, 1.1f, 2.4f, 1.0f, 1.1f, 4));
    p.push_back(cylinder(-30.0f,  18.0f, 3.0f, 0.8f, 3.0f, 5));
    p.push_back(cylinder(-28.0f, -18.0f, 3.0f, 0.8f, 3.0f, 5));
    p.push_back(cylinder(  2.0f,  10.0f, 4.5f, 1.1f, 4.5f, 5));
    p.push_back(cylinder( 33.0f,  22.0f, 3.0f, 0.8f, 3.0f, 5));
    p.push_back(cylinder( 32.0f, -22.0f, 3.0f, 0.8f, 3.0f, 5));
    return p;
}

static void scan_cpu(const std::vector<Primitive>& prims, Vec3 sensor, float yaw,
                     int channels, int azimuth_bins,
                     std::vector<float>& ranges,
                     std::vector<float>& xs,
                     std::vector<float>& ys,
                     std::vector<float>& zs,
                     std::vector<int>& labels) {
    int n = channels * azimuth_bins;
    ranges.resize(n); xs.resize(n); ys.resize(n); zs.resize(n); labels.resize(n);
    for (int idx = 0; idx < n; idx++) {
        int channel = idx / azimuth_bins;
        int az_idx = idx - channel * azimuth_bins;
        Vec3 d = ray_dir(channel, az_idx, channels, azimuth_bins, yaw);
        Vec3 hit;
        float range;
        int label;
        raycast_scene(prims.data(), static_cast<int>(prims.size()), sensor, d,
                      MAX_RANGE, range, hit, label);
        ranges[idx] = range;
        xs[idx] = hit.x; ys[idx] = hit.y; zs[idx] = hit.z; labels[idx] = label;
    }
}

static void scan_gpu(const Primitive* d_prims, int n_prims, Vec3 sensor, float yaw,
                     int channels, int azimuth_bins,
                     float* d_ranges, float* d_xs, float* d_ys, float* d_zs, int* d_labels,
                     std::vector<float>& ranges,
                     std::vector<float>& xs,
                     std::vector<float>& ys,
                     std::vector<float>& zs,
                     std::vector<int>& labels,
                     float* elapsed_ms = nullptr) {
    int n = channels * azimuth_bins;
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    int block = 256;
    int grid = (n + block - 1) / block;
    lidar3d_kernel<<<grid, block>>>(d_prims, n_prims, sensor, yaw, channels,
                                    azimuth_bins, MAX_RANGE, d_ranges, d_xs, d_ys,
                                    d_zs, d_labels);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    if (elapsed_ms) CUDA_CHECK(cudaEventElapsedTime(elapsed_ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    ranges.resize(n); xs.resize(n); ys.resize(n); zs.resize(n); labels.resize(n);
    CUDA_CHECK(cudaMemcpy(ranges.data(), d_ranges, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xs.data(), d_xs, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ys.data(), d_ys, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(zs.data(), d_zs, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(labels.data(), d_labels, n * sizeof(int), cudaMemcpyDeviceToHost));
}

static cv::Vec3b label_color(int label, float range) {
    if (label == 1) return cv::Vec3b(95, 110, 115);
    if (label == 2) return cv::Vec3b(80, 80, 80);
    if (label == 3) return cv::Vec3b(210, 120, 35);
    if (label == 4) return cv::Vec3b(35, 90, 220);
    if (label == 5) return cv::Vec3b(50, 160, 70);
    unsigned char v = static_cast<unsigned char>(std::max(0.0f, 255.0f - range * 2.0f));
    return cv::Vec3b(v, v, v);
}

static cv::Point2i project_point(float x, float y, float z, float cam_yaw) {
    float c = std::cos(cam_yaw);
    float s = std::sin(cam_yaw);
    float xr = c * x - s * y;
    float yr = s * x + c * y;
    int px = PANEL_W / 2 + static_cast<int>(xr * 5.0f);
    int py = static_cast<int>(PANEL_H * 0.78f - z * 18.0f - yr * 2.2f);
    return cv::Point2i(px, py);
}

static void draw_text(cv::Mat& img, const std::string& text, int y) {
    cv::putText(img, text, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.58,
                cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(img, text, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.58,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

static void draw_floor_grid(cv::Mat& img, float cam_yaw) {
    for (int i = -40; i <= 40; i += 10) {
        cv::Point2i a = project_point(static_cast<float>(i), -30.0f, 0.0f, cam_yaw);
        cv::Point2i b = project_point(static_cast<float>(i),  30.0f, 0.0f, cam_yaw);
        cv::line(img, a, b, cv::Scalar(218, 218, 218), 1, cv::LINE_AA);
    }
    for (int j = -30; j <= 30; j += 10) {
        cv::Point2i a = project_point(-40.0f, static_cast<float>(j), 0.0f, cam_yaw);
        cv::Point2i b = project_point( 40.0f, static_cast<float>(j), 0.0f, cam_yaw);
        cv::line(img, a, b, cv::Scalar(218, 218, 218), 1, cv::LINE_AA);
    }
}

static void draw_point_cloud(cv::Mat& img, int channels, int azimuth_bins,
                             const std::vector<float>& ranges,
                             const std::vector<float>& xs,
                             const std::vector<float>& ys,
                             const std::vector<float>& zs,
                             const std::vector<int>& labels,
                             float cam_yaw, bool dense) {
    img.setTo(cv::Scalar(242, 244, 245));
    draw_floor_grid(img, cam_yaw);
    int n = channels * azimuth_bins;
    int stride = dense ? 1 : 1;
    for (int i = 0; i < n; i += stride) {
        if (labels[i] == 0 || ranges[i] >= MAX_RANGE - 1e-3f) continue;
        cv::Point2i p = project_point(xs[i], ys[i], zs[i], cam_yaw);
        if (p.x < 1 || p.x >= PANEL_W - 1 || p.y < 1 || p.y >= PANEL_H - 1) continue;
        cv::Vec3b col = label_color(labels[i], ranges[i]);
        if (dense) {
            img.at<cv::Vec3b>(p.y, p.x) = col;
            img.at<cv::Vec3b>(p.y, p.x + 1) = col;
            img.at<cv::Vec3b>(p.y + 1, p.x) = col;
        } else {
            cv::circle(img, p, 2, cv::Scalar(col[0], col[1], col[2]), -1, cv::LINE_AA);
        }
    }
}

static void draw_sensor(cv::Mat& img, Vec3 sensor, float cam_yaw) {
    cv::Point2i p = project_point(sensor.x, sensor.y, sensor.z, cam_yaw);
    cv::circle(img, p, 6, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
    cv::circle(img, p, 4, cv::Scalar(0, 210, 255), -1, cv::LINE_AA);
}

static cv::Mat render_range_image(int channels, int azimuth_bins,
                                  const std::vector<float>& ranges,
                                  const std::vector<int>& labels) {
    cv::Mat gray(channels, azimuth_bins, CV_8UC1);
    for (int ch = 0; ch < channels; ch++) {
        for (int az = 0; az < azimuth_bins; az++) {
            int src = ch * azimuth_bins + az;
            int dst_y = channels - 1 - ch;
            float r = labels[src] == 0 ? MAX_RANGE : ranges[src];
            unsigned char v = static_cast<unsigned char>(
                std::max(0.0f, std::min(255.0f, 255.0f * (1.0f - r / MAX_RANGE))));
            gray.at<unsigned char>(dst_y, az) = v;
        }
    }
    cv::Mat resized, color;
    cv::resize(gray, resized, cv::Size(PANEL_W, PANEL_H), 0, 0, cv::INTER_NEAREST);
    cv::applyColorMap(resized, color, cv::COLORMAP_TURBO);
    return color;
}

static Vec3 sensor_pose_for_frame(int f) {
    float u = static_cast<float>(f) / SIM_FRAMES;
    float a = 2.0f * PI * u;
    return v3(8.0f * cosf(a), -4.0f + 6.0f * sinf(1.3f * a), 2.2f);
}

int main() {
    std::cout << "3D LiDAR Simulator comparison: CPU "
              << CPU_CHANNELS << "x" << CPU_AZIMUTH << " vs CUDA "
              << GPU_CHANNELS << "x" << GPU_AZIMUTH << " rays / scan"
              << std::endl;

    std::vector<Primitive> prims = build_scene();
    Primitive* d_prims = nullptr;
    CUDA_CHECK(cudaMalloc(&d_prims, prims.size() * sizeof(Primitive)));
    CUDA_CHECK(cudaMemcpy(d_prims, prims.data(), prims.size() * sizeof(Primitive),
                          cudaMemcpyHostToDevice));
    float *d_ranges = nullptr, *d_xs = nullptr, *d_ys = nullptr, *d_zs = nullptr;
    int* d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ranges, MAX_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xs, MAX_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ys, MAX_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_zs, MAX_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, MAX_RAYS * sizeof(int)));

    struct BenchCase { int channels; int azimuth; };
    BenchCase cases[] = {{16, 512}, {32, 1024}, {64, 2048}, {128, 4096}};
    std::vector<float> cpu_r, cpu_x, cpu_y, cpu_z, gpu_r, gpu_x, gpu_y, gpu_z;
    std::vector<int> cpu_l, gpu_l;
    Vec3 bench_sensor = v3(0.0f, -6.0f, 2.2f);
    float bench_yaw = 0.35f;
    float warmup_ms = 0.0f;
    scan_gpu(d_prims, static_cast<int>(prims.size()), bench_sensor, bench_yaw,
             16, 512, d_ranges, d_xs, d_ys, d_zs, d_labels,
             gpu_r, gpu_x, gpu_y, gpu_z, gpu_l, &warmup_ms);

    std::printf("\nSame-ray benchmark and deterministic check:\n");
    std::printf("%-12s %9s %10s %10s %9s\n", "scan", "rays", "CPU ms", "CUDA ms", "speedup");
    double first_max_err = 0.0;
    double first_label_match = 0.0;
    for (int i = 0; i < 4; i++) {
        int n = cases[i].channels * cases[i].azimuth;
        auto c0 = std::chrono::high_resolution_clock::now();
        scan_cpu(prims, bench_sensor, bench_yaw, cases[i].channels, cases[i].azimuth,
                 cpu_r, cpu_x, cpu_y, cpu_z, cpu_l);
        auto c1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
        float gpu_ms = 0.0f;
        scan_gpu(d_prims, static_cast<int>(prims.size()), bench_sensor, bench_yaw,
                 cases[i].channels, cases[i].azimuth, d_ranges, d_xs, d_ys, d_zs,
                 d_labels, gpu_r, gpu_x, gpu_y, gpu_z, gpu_l, &gpu_ms);
        double max_err = 0.0;
        int matches = 0;
        for (int k = 0; k < n; k++) {
            max_err = std::max(max_err, static_cast<double>(std::fabs(cpu_r[k] - gpu_r[k])));
            if (cpu_l[k] == gpu_l[k]) matches++;
        }
        if (i == 0) {
            first_max_err = max_err;
            first_label_match = 100.0 * matches / n;
        }
        char scan_name[32];
        std::snprintf(scan_name, sizeof(scan_name), "%dx%d", cases[i].channels, cases[i].azimuth);
        std::printf("%-12s %9d %10.2f %10.3f %8.1fx\n",
                    scan_name, n, cpu_ms, gpu_ms, cpu_ms / gpu_ms);
    }
    std::printf("Correctness check 16x512: max_abs_range_error=%.6f m, "
                "label_match_rate=%.2f%%\n\n", first_max_err, first_label_match);

    cv::VideoWriter video("gif/comparison_lidar3d_sim.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 24,
                          cv::Size(PANEL_W * 3, PANEL_H));

    double cpu_ms_sum = 0.0;
    double gpu_ms_sum = 0.0;
    int timed = 0;
    for (int f = 0; f < SIM_FRAMES; f++) {
        Vec3 sensor = sensor_pose_for_frame(f);
        float yaw = 2.0f * PI * static_cast<float>(f) / SIM_FRAMES;
        float cam = -0.72f + 0.50f * sinf(2.0f * PI * f / SIM_FRAMES);

        auto cpu0 = std::chrono::high_resolution_clock::now();
        scan_cpu(prims, sensor, yaw, CPU_CHANNELS, CPU_AZIMUTH,
                 cpu_r, cpu_x, cpu_y, cpu_z, cpu_l);
        auto cpu1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu1 - cpu0).count();

        float gpu_ms = 0.0f;
        scan_gpu(d_prims, static_cast<int>(prims.size()), sensor, yaw, GPU_CHANNELS,
                 GPU_AZIMUTH, d_ranges, d_xs, d_ys, d_zs, d_labels,
                 gpu_r, gpu_x, gpu_y, gpu_z, gpu_l, &gpu_ms);

        if (f >= 5) {
            cpu_ms_sum += cpu_ms;
            gpu_ms_sum += gpu_ms;
            timed++;
        }

        cv::Mat cpu_panel(PANEL_H, PANEL_W, CV_8UC3);
        cv::Mat gpu_panel(PANEL_H, PANEL_W, CV_8UC3);
        cv::Mat range_panel = render_range_image(GPU_CHANNELS, GPU_AZIMUTH, gpu_r, gpu_l);
        draw_point_cloud(cpu_panel, CPU_CHANNELS, CPU_AZIMUTH, cpu_r, cpu_x, cpu_y, cpu_z, cpu_l, cam, false);
        draw_point_cloud(gpu_panel, GPU_CHANNELS, GPU_AZIMUTH, gpu_r, gpu_x, gpu_y, gpu_z, gpu_l, cam, true);
        draw_sensor(cpu_panel, sensor, cam);
        draw_sensor(gpu_panel, sensor, cam);

        char buf[128];
        std::snprintf(buf, sizeof(buf), "CPU sparse 16x512  %.1f ms", cpu_ms);
        draw_text(cpu_panel, buf, 28);
        std::snprintf(buf, sizeof(buf), "CUDA dense 64x2048  %.2f ms", gpu_ms);
        draw_text(gpu_panel, buf, 28);
        draw_text(range_panel, "CUDA range image 64 rings x 2048 azimuth", 28);

        cv::Mat combined;
        cv::hconcat(std::vector<cv::Mat>{cpu_panel, gpu_panel, range_panel}, combined);
        video.write(combined);
    }
    video.release();

    if (timed > 0) {
        double cpu_ms = cpu_ms_sum / timed;
        double gpu_ms = gpu_ms_sum / timed;
        double cpu_us_ray = cpu_ms * 1000.0 / (CPU_CHANNELS * CPU_AZIMUTH);
        double gpu_us_ray = gpu_ms * 1000.0 / (GPU_CHANNELS * GPU_AZIMUTH);
        std::printf("Animated scan average:\n"
                    "CPU %.2f ms / scan (%d rays)\n"
                    "CUDA %.2f ms / scan (%d rays)\n"
                    "Per-ray throughput: CUDA %.5f us/ray vs CPU %.4f us/ray (%.0fx faster per ray)\n",
                    cpu_ms, CPU_CHANNELS * CPU_AZIMUTH,
                    gpu_ms, GPU_CHANNELS * GPU_AZIMUTH,
                    gpu_us_ray, cpu_us_ray, cpu_us_ray / gpu_us_ray);
    }

    std::system("ffmpeg -y -i gif/comparison_lidar3d_sim.avi "
                "-vf 'fps=15,scale=900:-1:flags=lanczos' -loop 0 "
                "gif/comparison_lidar3d_sim.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_lidar3d_sim.gif" << std::endl;

    cudaFree(d_prims);
    cudaFree(d_ranges);
    cudaFree(d_xs);
    cudaFree(d_ys);
    cudaFree(d_zs);
    cudaFree(d_labels);
    return 0;
}
