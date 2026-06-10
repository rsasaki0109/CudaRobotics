/*************************************************************************
    Realistic 3D LiDAR simulator vs the existing clean simulator.

    The clean simulator (comparison_lidar3d_sim.cu) produces too-perfect
    point clouds. This file adds five physical effects to the GPU
    raycast and compares against an ideal-noise-free baseline:

      1. range-dependent noise          sigma(r) = a + b*r (Velodyne-like)
      2. beam divergence                cone of half-angle theta_div,
                                        one Monte-Carlo sub-ray per scan
      3. multi-path (specular bounce)   at grazing incidence, with random
                                        chance, return = primary + mirror
                                        leg distance, intensity damped
      4. material reflectivity          intensity = albedo(label) *
                                        max(cos(theta_inc), 0) / r^2;
                                        drops returns below threshold
      5. rolling-shutter distortion     sensor pose linearly interpolated
                                        across the scan (t_az = T * az/N)

    Visualization: left panel = clean baseline cloud, right panel =
    realistic cloud (both rendered with point brightness modulated by
    the realistic intensity column). Status line reports the per-scan
    GPU time for each pipeline.

    Headline metric: realistic-pipeline GPU throughput (intensity, noise,
    multi-path, divergence, rolling-shutter pose lerp; one thread/ray).
 ************************************************************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cuda_check.cuh"

    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(EXIT_FAILURE); } } while (0)

constexpr float PI = 3.14159265358979323846f;
constexpr float DEG = PI / 180.0f;
constexpr float MAX_RANGE = 80.0f;
constexpr float VERT_MIN = -25.0f * DEG;
constexpr float VERT_MAX = 15.0f * DEG;

constexpr int CHANNELS = 64;
constexpr int AZIMUTH = 1024;
constexpr int N_RAYS  = CHANNELS * AZIMUTH;

constexpr int PANEL_W = 580;
constexpr int PANEL_H = 420;
constexpr int SIM_FRAMES = 80;

// realism parameters
constexpr float NOISE_A = 0.02f;          // [m] base noise
constexpr float NOISE_B = 0.0015f;        // [m / m] range-proportional
constexpr float DIV_HALF = 1.5e-3f;       // 1.5 mrad beam half-angle
constexpr float MULTI_PATH_PROB = 0.4f;   // at grazing incidence
constexpr float GRAZING_COS = 0.35f;      // cos(theta) threshold for grazing
constexpr float MULTI_PATH_ATT = 0.25f;   // intensity attenuation for mp leg
constexpr float DROP_INTENSITY = 0.025f;  // intensity below -> max_range (drop)
constexpr float RANGE_ATT     = 60.0f;    // intensity falloff scale [m] (exp model)
constexpr float T_SCAN = 0.1f;            // 10 Hz LiDAR
constexpr float SENSOR_SPEED = 4.0f;      // [m/s] forward speed of sensor
constexpr float SENSOR_YAW_RATE = 0.30f;  // [rad/s]

enum PrimitiveType { PRIM_AABB = 0, PRIM_CYLINDER = 1 };

struct Vec3 { float x, y, z; };
struct Primitive {
    int   type;
    int   label;
    Vec3  c;
    Vec3  h;
    float radius;
    float albedo;   // 0..1
};

__host__ __device__ static Vec3 v3(float x, float y, float z) { Vec3 v; v.x = x; v.y = y; v.z = z; return v; }
__host__ __device__ static Vec3 vadd(Vec3 a, Vec3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ static Vec3 vsub(Vec3 a, Vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ static Vec3 vmul(Vec3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
__host__ __device__ static float vdot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ static Vec3 vnorm(Vec3 a) {
    float n = sqrtf(vdot(a, a));
    if (n < 1e-9f) return v3(0.0f, 0.0f, 1.0f);
    return vmul(a, 1.0f / n);
}

// ------------------------------------------------------------------------
// Intersection with normal
// ------------------------------------------------------------------------
__host__ __device__ static bool intersect_aabb_n(const Primitive& p, Vec3 o, Vec3 d,
                                                 float max_range, float& t_hit,
                                                 Vec3& normal) {
    float tmin = 1e-4f;
    float tmax = max_range;
    int hit_axis = 0;
    float hit_sign = 1.0f;
    float mn[3] = {p.c.x - p.h.x, p.c.y - p.h.y, p.c.z - p.h.z};
    float mx[3] = {p.c.x + p.h.x, p.c.y + p.h.y, p.c.z + p.h.z};
    float oo[3] = {o.x, o.y, o.z};
    float dd[3] = {d.x, d.y, d.z};
    for (int axis = 0; axis < 3; axis++) {
        if (fabsf(dd[axis]) < 1e-7f) {
            if (oo[axis] < mn[axis] || oo[axis] > mx[axis]) return false;
            continue;
        }
        float inv = 1.0f / dd[axis];
        float t0 = (mn[axis] - oo[axis]) * inv;
        float t1 = (mx[axis] - oo[axis]) * inv;
        float sign0 = -1.0f;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; sign0 = 1.0f; }
        if (t0 > tmin) { tmin = t0; hit_axis = axis; hit_sign = sign0; }
        if (t1 < tmax) tmax = t1;
        if (tmin > tmax) return false;
    }
    if (tmin > max_range) return false;
    t_hit = tmin;
    normal = v3(0.0f, 0.0f, 0.0f);
    if (hit_axis == 0) normal.x = hit_sign;
    else if (hit_axis == 1) normal.y = hit_sign;
    else normal.z = hit_sign;
    return true;
}

__host__ __device__ static bool intersect_cylinder_n(const Primitive& p, Vec3 o, Vec3 d,
                                                     float max_range, float& t_hit,
                                                     Vec3& normal) {
    bool any_hit = false;
    float best = max_range;
    Vec3 best_n = v3(0, 0, 1);
    float ox = o.x - p.c.x, oy = o.y - p.c.y;
    float a = d.x * d.x + d.y * d.y;
    float b = 2.0f * (ox * d.x + oy * d.y);
    float c = ox * ox + oy * oy - p.radius * p.radius;
    if (a > 1e-8f) {
        float disc = b * b - 4.0f * a * c;
        if (disc >= 0.0f) {
            float sd = sqrtf(disc);
            float inv = 0.5f / a;
            float ts[2] = {(-b - sd) * inv, (-b + sd) * inv};
            for (int i = 0; i < 2; i++) {
                float t = ts[i];
                float z = o.z + t * d.z;
                if (t > 1e-4f && t < best && z >= p.c.z - p.h.z && z <= p.c.z + p.h.z) {
                    best = t;
                    float nx = (o.x + t * d.x - p.c.x);
                    float ny = (o.y + t * d.y - p.c.y);
                    float nn = sqrtf(nx * nx + ny * ny);
                    if (nn < 1e-9f) nn = 1.0f;
                    best_n = v3(nx / nn, ny / nn, 0.0f);
                    any_hit = true;
                }
            }
        }
    }
    if (fabsf(d.z) > 1e-7f) {
        float caps[2] = {p.c.z - p.h.z, p.c.z + p.h.z};
        float sign[2] = {-1.0f, 1.0f};
        for (int i = 0; i < 2; i++) {
            float t = (caps[i] - o.z) / d.z;
            float xr = o.x + t * d.x - p.c.x;
            float yr = o.y + t * d.y - p.c.y;
            if (t > 1e-4f && t < best && xr * xr + yr * yr <= p.radius * p.radius) {
                best = t;
                best_n = v3(0.0f, 0.0f, sign[i]);
                any_hit = true;
            }
        }
    }
    if (any_hit) { t_hit = best; normal = best_n; }
    return any_hit;
}

__host__ __device__ static void raycast_normal(const Primitive* prims, int n_prims,
                                               Vec3 o, Vec3 d, float max_range,
                                               float& range, Vec3& hit, Vec3& normal,
                                               int& label, float& albedo) {
    range = max_range;
    label = 0;
    albedo = 0.2f;
    normal = v3(0.0f, 0.0f, 1.0f);
    if (d.z < -1e-6f) {
        float t = -o.z / d.z;
        if (t > 1e-4f && t < range) {
            range = t;
            label = 1;
            albedo = 0.30f;
            normal = v3(0.0f, 0.0f, 1.0f);
        }
    }
    for (int i = 0; i < n_prims; i++) {
        float t = max_range;
        Vec3 n;
        bool ok = prims[i].type == PRIM_AABB
            ? intersect_aabb_n(prims[i], o, d, max_range, t, n)
            : intersect_cylinder_n(prims[i], o, d, max_range, t, n);
        if (ok && t < range) {
            range = t;
            normal = n;
            label = prims[i].label;
            albedo = prims[i].albedo;
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

// ------------------------------------------------------------------------
// Realistic raycast kernel
// ------------------------------------------------------------------------
__global__ void init_rng_kernel(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void clean_kernel(const Primitive* __restrict__ prims, int n_prims,
                             Vec3 sensor, float yaw, int channels, int azimuth,
                             float* __restrict__ xs, float* __restrict__ ys,
                             float* __restrict__ zs, int* __restrict__ labels,
                             float* __restrict__ intensities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= channels * azimuth) return;
    int ch = idx / azimuth;
    int az = idx - ch * azimuth;
    Vec3 d = ray_dir(ch, az, channels, azimuth, yaw);
    Vec3 hit, normal;
    float range, albedo;
    int label;
    raycast_normal(prims, n_prims, sensor, d, MAX_RANGE, range, hit, normal, label, albedo);
    xs[idx] = hit.x;
    ys[idx] = hit.y;
    zs[idx] = hit.z;
    labels[idx] = label;
    float cos_inc = -vdot(d, normal);
    if (cos_inc < 0.0f) cos_inc = 0.0f;
    intensities[idx] = albedo * cos_inc * expf(-range / RANGE_ATT);
}

__global__ void realistic_kernel(const Primitive* __restrict__ prims, int n_prims,
                                 Vec3 sensor_t0, Vec3 sensor_t1,
                                 float yaw_t0, float yaw_t1,
                                 int channels, int azimuth,
                                 curandState* __restrict__ rng,
                                 float* __restrict__ xs, float* __restrict__ ys,
                                 float* __restrict__ zs, int* __restrict__ labels,
                                 float* __restrict__ intensities,
                                 unsigned char* __restrict__ multipath_flag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= channels * azimuth) return;
    int ch = idx / azimuth;
    int az = idx - ch * azimuth;
    curandState s = rng[idx];

    // 5. rolling-shutter: lerp pose across az index
    float t_norm = (float)az / (float)azimuth;
    Vec3  sensor = vadd(vmul(sensor_t0, 1.0f - t_norm),
                        vmul(sensor_t1, t_norm));
    float yaw    = yaw_t0 * (1.0f - t_norm) + yaw_t1 * t_norm;

    // base ray direction
    Vec3 d_center = ray_dir(ch, az, channels, azimuth, yaw);

    // 2. beam divergence: sample a single sub-ray inside the cone (Monte
    //    Carlo single-sample is unbiased for the cone average since the
    //    sphere is locally flat at this angular scale)
    float du = (2.0f * curand_uniform(&s) - 1.0f) * DIV_HALF;
    float dv = (2.0f * curand_uniform(&s) - 1.0f) * DIV_HALF;
    // build two tangent vectors to d_center
    Vec3 up = (fabsf(d_center.z) < 0.99f) ? v3(0, 0, 1) : v3(1, 0, 0);
    Vec3 t1 = vnorm(v3(d_center.y * up.z - d_center.z * up.y,
                       d_center.z * up.x - d_center.x * up.z,
                       d_center.x * up.y - d_center.y * up.x));
    Vec3 t2 = v3(d_center.y * t1.z - d_center.z * t1.y,
                 d_center.z * t1.x - d_center.x * t1.z,
                 d_center.x * t1.y - d_center.y * t1.x);
    Vec3 d = vnorm(vadd(vadd(d_center, vmul(t1, du)), vmul(t2, dv)));

    // primary cast
    Vec3 hit, normal;
    float range, albedo;
    int label;
    raycast_normal(prims, n_prims, sensor, d, MAX_RANGE, range, hit, normal, label, albedo);

    // material intensity: cos(theta_inc) * albedo * exp(-r/scale)
    // (matches LiDAR detector dynamic range better than pure 1/r^2)
    float cos_inc = -vdot(d, normal);
    if (cos_inc < 0.0f) cos_inc = 0.0f;
    float intensity = albedo * cos_inc * expf(-range / RANGE_ATT);

    bool is_multipath = false;
    // 3. multi-path: at grazing incidence, sometimes reflect off and
    //    return the mirror-leg distance (lengthens the measurement)
    if (cos_inc < GRAZING_COS && range < MAX_RANGE - 1.0f &&
        curand_uniform(&s) < MULTI_PATH_PROB) {
        Vec3 mirror = vsub(d, vmul(normal, 2.0f * vdot(d, normal)));
        Vec3 hit2, normal2; float range2, albedo2; int label2;
        Vec3 origin2 = vadd(hit, vmul(mirror, 0.02f));
        raycast_normal(prims, n_prims, origin2, mirror,
                       MAX_RANGE - range, range2, hit2, normal2, label2, albedo2);
        if (range2 < MAX_RANGE - range - 1.0f) {
            range = range + range2;
            float cos_inc2 = -vdot(mirror, normal2);
            if (cos_inc2 < 0.0f) cos_inc2 = 0.0f;
            intensity = albedo2 * cos_inc2 * expf(-range / RANGE_ATT) * MULTI_PATH_ATT;
            // recompute hit point along original ray at lengthened range
            hit = vadd(sensor, vmul(d, range));
            is_multipath = true;
        }
    }

    // 1. range-dependent gaussian noise
    float sigma = NOISE_A + NOISE_B * range;
    range = range + sigma * curand_normal(&s);
    if (range < 0.0f) range = 0.0f;

    // 4. material reflectivity drop-out (low-intensity returns lost)
    if (intensity < DROP_INTENSITY) {
        range = MAX_RANGE;
        label = 0;
        intensity = 0.0f;
    } else {
        hit = vadd(sensor, vmul(d, range));
    }

    xs[idx] = hit.x;
    ys[idx] = hit.y;
    zs[idx] = hit.z;
    labels[idx] = label;
    intensities[idx] = intensity;
    multipath_flag[idx] = is_multipath ? 1u : 0u;
    rng[idx] = s;
}

// ------------------------------------------------------------------------
// Scene
// ------------------------------------------------------------------------
static Primitive aabb(float x, float y, float z, float hx, float hy, float hz,
                      int label, float albedo) {
    Primitive p; p.type = PRIM_AABB; p.label = label; p.c = v3(x, y, z);
    p.h = v3(hx, hy, hz); p.radius = 0.0f; p.albedo = albedo;
    return p;
}
static Primitive cyl(float x, float y, float z, float r, float hz,
                     int label, float albedo) {
    Primitive p; p.type = PRIM_CYLINDER; p.label = label; p.c = v3(x, y, z);
    p.h = v3(0, 0, hz); p.radius = r; p.albedo = albedo;
    return p;
}
static std::vector<Primitive> build_scene() {
    std::vector<Primitive> p;
    // walls (low albedo)
    p.push_back(aabb(0.0f, -30.0f, 2.5f, 42.0f, 0.35f, 2.5f, 2, 0.25f));
    p.push_back(aabb(0.0f,  30.0f, 2.5f, 42.0f, 0.35f, 2.5f, 2, 0.25f));
    p.push_back(aabb(-42.0f, 0.0f, 2.5f, 0.35f, 30.0f, 2.5f, 2, 0.25f));
    p.push_back(aabb( 42.0f, 0.0f, 2.5f, 0.35f, 30.0f, 2.5f, 2, 0.25f));
    // buildings (moderate albedo)
    p.push_back(aabb(-18.0f, -10.0f, 4.0f, 5.5f, 8.0f, 4.0f, 3, 0.55f));
    p.push_back(aabb( 17.0f,  -8.0f, 3.5f, 7.0f, 5.0f, 3.5f, 3, 0.45f));
    p.push_back(aabb(-10.0f,  15.0f, 5.0f, 6.0f, 4.5f, 5.0f, 3, 0.50f));
    p.push_back(aabb( 24.0f,  15.0f, 6.0f, 4.0f, 8.0f, 6.0f, 3, 0.50f));
    // cars (dark, low albedo + specular tendency captured via low albedo)
    p.push_back(aabb( -4.0f, -22.0f, 1.1f, 2.4f, 1.0f, 1.1f, 4, 0.12f));
    p.push_back(aabb(  9.0f,  22.0f, 1.1f, 2.4f, 1.0f, 1.1f, 4, 0.12f));
    p.push_back(aabb( 30.0f,  -3.0f, 1.1f, 2.4f, 1.0f, 1.1f, 4, 0.12f));
    // foliage / cylinders (high albedo, diffuse)
    p.push_back(cyl(-30.0f,  18.0f, 3.0f, 0.8f, 3.0f, 5, 0.65f));
    p.push_back(cyl(-28.0f, -18.0f, 3.0f, 0.8f, 3.0f, 5, 0.65f));
    p.push_back(cyl(  2.0f,  10.0f, 4.5f, 1.1f, 4.5f, 5, 0.70f));
    p.push_back(cyl( 33.0f,  22.0f, 3.0f, 0.8f, 3.0f, 5, 0.65f));
    p.push_back(cyl( 32.0f, -22.0f, 3.0f, 0.8f, 3.0f, 5, 0.65f));
    return p;
}

// ------------------------------------------------------------------------
// Rendering
// ------------------------------------------------------------------------
static cv::Point2i project_point(float x, float y, float z, float cam_yaw) {
    float c = std::cos(cam_yaw);
    float s = std::sin(cam_yaw);
    float xr = c * x - s * y;
    float yr = s * x + c * y;
    int px = PANEL_W / 2 + static_cast<int>(xr * 5.5f);
    int py = static_cast<int>(PANEL_H * 0.78f - z * 18.0f - yr * 2.0f);
    return cv::Point2i(px, py);
}

static cv::Vec3b label_base(int label) {
    if (label == 1) return cv::Vec3b(95, 110, 115);   // ground
    if (label == 2) return cv::Vec3b(80, 80, 80);     // walls
    if (label == 3) return cv::Vec3b(210, 120, 35);   // buildings
    if (label == 4) return cv::Vec3b(35, 90, 220);    // cars
    if (label == 5) return cv::Vec3b(50, 160, 70);    // foliage
    return cv::Vec3b(70, 70, 70);
}

static void draw_floor_grid(cv::Mat& img, float cam_yaw) {
    for (int i = -40; i <= 40; i += 10) {
        cv::Point2i a = project_point(static_cast<float>(i), -30.0f, 0.0f, cam_yaw);
        cv::Point2i b = project_point(static_cast<float>(i),  30.0f, 0.0f, cam_yaw);
        cv::line(img, a, b, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
    }
    for (int j = -30; j <= 30; j += 10) {
        cv::Point2i a = project_point(-40.0f, static_cast<float>(j), 0.0f, cam_yaw);
        cv::Point2i b = project_point( 40.0f, static_cast<float>(j), 0.0f, cam_yaw);
        cv::line(img, a, b, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
    }
}

static void render_cloud(cv::Mat& img, const std::vector<float>& xs,
                         const std::vector<float>& ys,
                         const std::vector<float>& zs,
                         const std::vector<int>& labels,
                         const std::vector<float>& intensities,
                         const std::vector<unsigned char>* mp_flag,
                         float cam_yaw) {
    int n = static_cast<int>(xs.size());
    float intensity_max = 0.0f;
    for (float v : intensities) if (v > intensity_max) intensity_max = v;
    if (intensity_max < 1e-6f) intensity_max = 1.0f;
    for (int i = 0; i < n; i++) {
        if (labels[i] == 0) continue;
        cv::Point2i p = project_point(xs[i], ys[i], zs[i], cam_yaw);
        if (p.x < 0 || p.x >= img.cols || p.y < 0 || p.y >= img.rows) continue;
        cv::Vec3b base = label_base(labels[i]);
        if (mp_flag && (*mp_flag)[i]) {
            // multi-path returns drawn in red highlight
            img.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(60, 60, 230);
            continue;
        }
        float a = intensities[i] / intensity_max;
        if (a > 1.0f) a = 1.0f;
        float gamma = sqrtf(a);
        cv::Vec3b col(static_cast<uchar>(base[0] * gamma),
                      static_cast<uchar>(base[1] * gamma),
                      static_cast<uchar>(base[2] * gamma));
        img.at<cv::Vec3b>(p.y, p.x) = col;
    }
}

static void convert_avi_to_gif(const char* avi, const char* gif, int fps) {
    char cmd[512];
    std::snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -i %s -vf 'fps=%d,scale=900:-1:flags=lanczos' -loop 0 %s "
        "> /dev/null 2>&1", avi, fps, gif);
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg conversion returned %d\n", rc);
}

// ------------------------------------------------------------------------
// main
// ------------------------------------------------------------------------
int main() {
    auto prims = build_scene();
    int n_prims = static_cast<int>(prims.size());

    Primitive* d_prims;
    float *d_xs_c, *d_ys_c, *d_zs_c, *d_int_c;
    int   *d_lbl_c;
    float *d_xs_r, *d_ys_r, *d_zs_r, *d_int_r;
    int   *d_lbl_r;
    unsigned char* d_mp_r;
    curandState* d_rng;
    CUDA_CHECK(cudaMalloc(&d_prims, n_prims * sizeof(Primitive)));
    CUDA_CHECK(cudaMemcpy(d_prims, prims.data(), n_prims * sizeof(Primitive),
                          cudaMemcpyHostToDevice));
    auto alloc_set = [&](float*& x, float*& y, float*& z, int*& l, float*& it) {
        CUDA_CHECK(cudaMalloc(&x,  N_RAYS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&y,  N_RAYS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&z,  N_RAYS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&l,  N_RAYS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&it, N_RAYS * sizeof(float)));
    };
    alloc_set(d_xs_c, d_ys_c, d_zs_c, d_lbl_c, d_int_c);
    alloc_set(d_xs_r, d_ys_r, d_zs_r, d_lbl_r, d_int_r);
    CUDA_CHECK(cudaMalloc(&d_mp_r, N_RAYS));
    CUDA_CHECK(cudaMalloc(&d_rng, N_RAYS * sizeof(curandState)));

    int threads = 256;
    int blocks = (N_RAYS + threads - 1) / threads;
    init_rng_kernel<<<blocks, threads>>>(d_rng, N_RAYS, 2026ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_xs_c(N_RAYS), h_ys_c(N_RAYS), h_zs_c(N_RAYS), h_int_c(N_RAYS);
    std::vector<int>   h_lbl_c(N_RAYS);
    std::vector<float> h_xs_r(N_RAYS), h_ys_r(N_RAYS), h_zs_r(N_RAYS), h_int_r(N_RAYS);
    std::vector<int>   h_lbl_r(N_RAYS);
    std::vector<unsigned char> h_mp_r(N_RAYS);

    cv::VideoWriter video("gif/comparison_lidar3d_realistic.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 15,
                          cv::Size(PANEL_W * 2 + 4, PANEL_H + 60));
    if (!video.isOpened()) {
        std::fprintf(stderr, "Failed to open AVI\n");
        return 1;
    }

    double clean_ms_sum = 0.0, realistic_ms_sum = 0.0;
    int counted = 0;
    int n_mp_total = 0, n_drop_total = 0;

    for (int frame = 0; frame < SIM_FRAMES; frame++) {
        float t = frame * T_SCAN;
        Vec3 sensor_t0 = v3(-25.0f + SENSOR_SPEED * t, 0.0f, 1.6f);
        Vec3 sensor_t1 = v3(-25.0f + SENSOR_SPEED * (t + T_SCAN), 0.0f, 1.6f);
        float yaw_t0 = SENSOR_YAW_RATE * t;
        float yaw_t1 = SENSOR_YAW_RATE * (t + T_SCAN);
        Vec3 sensor_mid = vmul(vadd(sensor_t0, sensor_t1), 0.5f);
        float yaw_mid = 0.5f * (yaw_t0 + yaw_t1);

        // clean baseline (uses mid pose, no rolling shutter)
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        clean_kernel<<<blocks, threads>>>(d_prims, n_prims, sensor_mid, yaw_mid,
                                          CHANNELS, AZIMUTH,
                                          d_xs_c, d_ys_c, d_zs_c, d_lbl_c, d_int_c);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms_c = 0.0f; cudaEventElapsedTime(&ms_c, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);

        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        realistic_kernel<<<blocks, threads>>>(d_prims, n_prims,
                                              sensor_t0, sensor_t1, yaw_t0, yaw_t1,
                                              CHANNELS, AZIMUTH, d_rng,
                                              d_xs_r, d_ys_r, d_zs_r, d_lbl_r, d_int_r,
                                              d_mp_r);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms_r = 0.0f; cudaEventElapsedTime(&ms_r, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);

        if (frame >= 3) { clean_ms_sum += ms_c; realistic_ms_sum += ms_r; counted++; }

        CUDA_CHECK(cudaMemcpy(h_xs_c.data(),  d_xs_c,  N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ys_c.data(),  d_ys_c,  N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_zs_c.data(),  d_zs_c,  N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_lbl_c.data(), d_lbl_c, N_RAYS * sizeof(int),   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_int_c.data(), d_int_c, N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_xs_r.data(),  d_xs_r,  N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ys_r.data(),  d_ys_r,  N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_zs_r.data(),  d_zs_r,  N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_lbl_r.data(), d_lbl_r, N_RAYS * sizeof(int),   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_int_r.data(), d_int_r, N_RAYS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mp_r.data(),  d_mp_r,  N_RAYS,                 cudaMemcpyDeviceToHost));

        int n_mp = 0, n_drop = 0;
        for (int i = 0; i < N_RAYS; i++) { if (h_mp_r[i]) n_mp++; if (h_lbl_r[i] == 0) n_drop++; }
        n_mp_total += n_mp; n_drop_total += n_drop;

        float cam_yaw = -yaw_mid + 0.15f;
        cv::Mat panel_c(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(20, 20, 20));
        cv::Mat panel_r(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(20, 20, 20));
        draw_floor_grid(panel_c, cam_yaw);
        draw_floor_grid(panel_r, cam_yaw);
        render_cloud(panel_c, h_xs_c, h_ys_c, h_zs_c, h_lbl_c, h_int_c, nullptr, cam_yaw);
        render_cloud(panel_r, h_xs_r, h_ys_r, h_zs_r, h_lbl_r, h_int_r, &h_mp_r, cam_yaw);

        cv::Mat frame_img(PANEL_H + 60, PANEL_W * 2 + 4, CV_8UC3, cv::Scalar(30, 30, 30));
        panel_c.copyTo(frame_img(cv::Rect(0, 60, PANEL_W, PANEL_H)));
        panel_r.copyTo(frame_img(cv::Rect(PANEL_W + 4, 60, PANEL_W, PANEL_H)));
        char head[256];
        std::snprintf(head, sizeof(head),
                      "Clean LiDAR (%.2f ms)   |   Realistic LiDAR (%.2f ms)   "
                      "frame=%d   N_rays=%dx%d",
                      ms_c, ms_r, frame, CHANNELS, AZIMUTH);
        cv::putText(frame_img, head, cv::Point(12, 22), cv::FONT_HERSHEY_SIMPLEX,
                    0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        char sub[256];
        std::snprintf(sub, sizeof(sub),
                      "noise sigma=%.2f+%.4f*r m   div=%.1f mrad   "
                      "multi-path %d   drops %d   rolling shutter (sensor v=%.1f m/s)",
                      NOISE_A, NOISE_B, DIV_HALF * 1.0e3f, n_mp, n_drop, SENSOR_SPEED);
        cv::putText(frame_img, sub, cv::Point(12, 44), cv::FONT_HERSHEY_SIMPLEX,
                    0.48, cv::Scalar(180, 220, 180), 1, cv::LINE_AA);
        video.write(frame_img);
    }
    video.release();
    convert_avi_to_gif("gif/comparison_lidar3d_realistic.avi",
                       "gif/comparison_lidar3d_realistic.gif", 15);

    if (counted > 0) {
        double c_ms = clean_ms_sum / counted;
        double r_ms = realistic_ms_sum / counted;
        std::printf("Avg clean GPU %.3f ms / scan, realistic GPU %.3f ms / scan "
                    "(%.1fx of clean, %d rays/scan)\n"
                    "Multi-path returns/scan ~ %.0f   drops/scan ~ %.0f\n",
                    c_ms, r_ms, r_ms / c_ms, N_RAYS,
                    (double)n_mp_total / counted, (double)n_drop_total / counted);
    }
    std::printf("GIF saved to gif/comparison_lidar3d_realistic.gif\n");

    CUDA_CHECK(cudaFree(d_prims));
    for (auto* p : {d_xs_c, d_ys_c, d_zs_c, d_int_c,
                    d_xs_r, d_ys_r, d_zs_r, d_int_r}) CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaFree(d_lbl_c));
    CUDA_CHECK(cudaFree(d_lbl_r));
    CUDA_CHECK(cudaFree(d_mp_r));
    CUDA_CHECK(cudaFree(d_rng));
    return 0;
}
