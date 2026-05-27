/*************************************************************************
    Gaussian Splatting Map Renderer: CPU sparse surfel rendering vs CUDA
    dense Gaussian splatting. This is a forward-only robotics visualization
    demo: a synthetic LiDAR-like map is represented as colored 3D Gaussian
    surfels, then rendered from a moving camera.

    The implementation is intentionally small and self-contained. It is not a
    training pipeline and does not copy an external Gaussian Splatting codebase;
    the CUDA pattern is the relevant robotics hook: one Gaussian per thread,
    projected to screen space, with atomic additive accumulation.
 ************************************************************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr float PI = 3.14159265358979323846f;
constexpr int W = 480;
constexpr int H = 360;
constexpr int SIM_FRAMES = 72;
constexpr int N_CPU = 4096;
constexpr int N_GPU = 65536;
constexpr int N_CHECK = 2048;
constexpr float FOCAL = 360.0f;
constexpr float NEAR_Z = 0.25f;
constexpr float SPLAT_CUTOFF = 3.0f;

struct Vec3 {
    float x, y, z;
};

struct Gaussian {
    float x, y, z;
    float sigma;
    float r, g, b;
};

__host__ __device__ static Vec3 v3(float x, float y, float z) {
    Vec3 v; v.x = x; v.y = y; v.z = z; return v;
}

__host__ __device__ static Vec3 vsub(Vec3 a, Vec3 b) {
    return v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ static float dot3(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ static Vec3 cross3(Vec3 a, Vec3 b) {
    return v3(a.y * b.z - a.z * b.y,
              a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x);
}

__host__ __device__ static Vec3 normalize3(Vec3 a) {
    float n = sqrtf(dot3(a, a));
    return n > 1e-8f ? v3(a.x / n, a.y / n, a.z / n) : v3(0.0f, 0.0f, 1.0f);
}

struct Camera {
    Vec3 eye;
    Vec3 right;
    Vec3 up;
    Vec3 forward;
};

static Camera make_camera(Vec3 eye, Vec3 target) {
    Camera c;
    Vec3 world_up = v3(0.0f, 0.0f, 1.0f);
    c.eye = eye;
    c.forward = normalize3(vsub(target, eye));
    c.right = normalize3(cross3(c.forward, world_up));
    c.up = cross3(c.right, c.forward);
    return c;
}

__host__ __device__ static bool project_gaussian(const Gaussian& g,
                                                 const Camera& cam,
                                                 float& sx, float& sy,
                                                 float& sigma_px,
                                                 float& depth) {
    Vec3 p = v3(g.x - cam.eye.x, g.y - cam.eye.y, g.z - cam.eye.z);
    float x = dot3(p, cam.right);
    float y = dot3(p, cam.up);
    float z = dot3(p, cam.forward);
    if (z < NEAR_Z) return false;
    sx = W * 0.5f + FOCAL * x / z;
    sy = H * 0.58f - FOCAL * y / z;
    sigma_px = fmaxf(0.85f, FOCAL * g.sigma / z);
    depth = z;
    return sx > -64.0f && sx < W + 64.0f && sy > -64.0f && sy < H + 64.0f;
}

static void add_box_surface(std::vector<Gaussian>& out,
                            Vec3 center, Vec3 half, float spacing,
                            cv::Vec3f color, float sigma) {
    auto add = [&](float x, float y, float z, float shade) {
        Gaussian g;
        g.x = center.x + x; g.y = center.y + y; g.z = center.z + z;
        g.sigma = sigma;
        g.r = std::min(1.0f, color[2] * shade);
        g.g = std::min(1.0f, color[1] * shade);
        g.b = std::min(1.0f, color[0] * shade);
        out.push_back(g);
    };
    for (float x = -half.x; x <= half.x + 1e-4f; x += spacing) {
        for (float z = -half.z; z <= half.z + 1e-4f; z += spacing) {
            add(x, -half.y, z, 0.84f);
            add(x,  half.y, z, 1.00f);
        }
    }
    for (float y = -half.y; y <= half.y + 1e-4f; y += spacing) {
        for (float z = -half.z; z <= half.z + 1e-4f; z += spacing) {
            add(-half.x, y, z, 0.72f);
            add( half.x, y, z, 0.92f);
        }
    }
    for (float x = -half.x; x <= half.x + 1e-4f; x += spacing) {
        for (float y = -half.y; y <= half.y + 1e-4f; y += spacing) {
            add(x, y, half.z, 1.08f);
        }
    }
}

static void add_cylinder_surface(std::vector<Gaussian>& out,
                                 Vec3 center, float radius, float height,
                                 float spacing, cv::Vec3f color, float sigma) {
    int rings = std::max(6, static_cast<int>(height / spacing));
    int segs = std::max(16, static_cast<int>(2.0f * PI * radius / spacing));
    for (int iz = 0; iz <= rings; iz++) {
        float z = center.z - 0.5f * height + height * iz / rings;
        for (int i = 0; i < segs; i++) {
            float a = 2.0f * PI * i / segs;
            Gaussian g;
            g.x = center.x + radius * cosf(a);
            g.y = center.y + radius * sinf(a);
            g.z = z;
            g.sigma = sigma;
            float shade = 0.78f + 0.22f * (0.5f + 0.5f * cosf(a - 0.5f));
            g.r = color[2] * shade; g.g = color[1] * shade; g.b = color[0] * shade;
            out.push_back(g);
        }
    }
}

static std::vector<Gaussian> build_gaussian_map(int target_n) {
    std::vector<Gaussian> g;
    float spacing = target_n > 20000 ? 0.42f : 1.65f;
    float sigma = spacing * 0.48f;
    add_box_surface(g, v3(-15.0f, -8.0f, 4.0f), v3(5.0f, 7.0f, 4.0f), spacing,
                    cv::Vec3f(0.16f, 0.56f, 0.95f), sigma);
    add_box_surface(g, v3( 12.0f, -9.0f, 3.5f), v3(6.0f, 4.5f, 3.5f), spacing,
                    cv::Vec3f(0.90f, 0.45f, 0.16f), sigma);
    add_box_surface(g, v3(  5.0f, 13.0f, 5.2f), v3(7.0f, 3.5f, 5.2f), spacing,
                    cv::Vec3f(0.32f, 0.76f, 0.50f), sigma);
    add_box_surface(g, v3( 25.0f,  8.0f, 6.0f), v3(3.5f, 8.0f, 6.0f), spacing,
                    cv::Vec3f(0.78f, 0.28f, 0.72f), sigma);
    add_box_surface(g, v3( -5.0f, -22.0f, 1.0f), v3(2.4f, 1.0f, 1.0f), spacing,
                    cv::Vec3f(0.10f, 0.12f, 0.95f), sigma);
    add_box_surface(g, v3( 20.0f, -20.0f, 1.0f), v3(2.4f, 1.0f, 1.0f), spacing,
                    cv::Vec3f(0.10f, 0.12f, 0.95f), sigma);
    add_cylinder_surface(g, v3(-28.0f, 18.0f, 3.0f), 0.8f, 6.0f, spacing,
                         cv::Vec3f(0.18f, 0.65f, 0.18f), sigma);
    add_cylinder_surface(g, v3(-28.0f, -20.0f, 3.0f), 0.8f, 6.0f, spacing,
                         cv::Vec3f(0.18f, 0.65f, 0.18f), sigma);
    add_cylinder_surface(g, v3( 32.0f,  20.0f, 3.0f), 0.8f, 6.0f, spacing,
                         cv::Vec3f(0.18f, 0.65f, 0.18f), sigma);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> ux(-34.0f, 34.0f);
    std::uniform_real_distribution<float> uy(-26.0f, 26.0f);
    std::uniform_real_distribution<float> jitter(-0.04f, 0.04f);
    while ((int)g.size() < target_n) {
        Gaussian s;
        s.x = ux(rng); s.y = uy(rng); s.z = jitter(rng);
        s.sigma = sigma * 0.58f;
        s.r = 0.45f; s.g = 0.46f; s.b = 0.43f;
        g.push_back(s);
    }
    if ((int)g.size() > target_n) g.resize(target_n);
    return g;
}

static void clear_accum(std::vector<float>& r, std::vector<float>& g,
                        std::vector<float>& b, std::vector<float>& w) {
    std::fill(r.begin(), r.end(), 0.0f);
    std::fill(g.begin(), g.end(), 0.0f);
    std::fill(b.begin(), b.end(), 0.0f);
    std::fill(w.begin(), w.end(), 0.0f);
}

static void splat_cpu(const std::vector<Gaussian>& gs, const Camera& cam,
                      std::vector<float>& ar, std::vector<float>& ag,
                      std::vector<float>& ab, std::vector<float>& aw) {
    clear_accum(ar, ag, ab, aw);
    for (const Gaussian& g : gs) {
        float sx, sy, spx, depth;
        if (!project_gaussian(g, cam, sx, sy, spx, depth)) continue;
        float radius = SPLAT_CUTOFF * spx;
        int x0 = std::max(0, static_cast<int>(floorf(sx - radius)));
        int x1 = std::min(W - 1, static_cast<int>(ceilf(sx + radius)));
        int y0 = std::max(0, static_cast<int>(floorf(sy - radius)));
        int y1 = std::min(H - 1, static_cast<int>(ceilf(sy + radius)));
        float inv2 = 0.5f / (spx * spx);
        float depth_alpha = std::min(1.0f, 2.5f / depth);
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                float dx = x + 0.5f - sx;
                float dy = y + 0.5f - sy;
                float ww = expf(-(dx * dx + dy * dy) * inv2) * depth_alpha;
                int idx = y * W + x;
                ar[idx] += ww * g.r;
                ag[idx] += ww * g.g;
                ab[idx] += ww * g.b;
                aw[idx] += ww;
            }
        }
    }
}

__global__ void clear_kernel(float* r, float* g, float* b, float* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    r[i] = 0.0f; g[i] = 0.0f; b[i] = 0.0f; w[i] = 0.0f;
}

__global__ void splat_kernel(const Gaussian* gs, int n, Camera cam,
                             float* ar, float* ag, float* ab, float* aw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Gaussian g = gs[i];
    float sx, sy, spx, depth;
    if (!project_gaussian(g, cam, sx, sy, spx, depth)) return;
    float radius = SPLAT_CUTOFF * spx;
    int x0 = max(0, static_cast<int>(floorf(sx - radius)));
    int x1 = min(W - 1, static_cast<int>(ceilf(sx + radius)));
    int y0 = max(0, static_cast<int>(floorf(sy - radius)));
    int y1 = min(H - 1, static_cast<int>(ceilf(sy + radius)));
    float inv2 = 0.5f / (spx * spx);
    float depth_alpha = fminf(1.0f, 2.5f / depth);
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            float dx = x + 0.5f - sx;
            float dy = y + 0.5f - sy;
            float ww = expf(-(dx * dx + dy * dy) * inv2) * depth_alpha;
            int idx = y * W + x;
            atomicAdd(&ar[idx], ww * g.r);
            atomicAdd(&ag[idx], ww * g.g);
            atomicAdd(&ab[idx], ww * g.b);
            atomicAdd(&aw[idx], ww);
        }
    }
}

static float render_gpu(const Gaussian* d_gs, int n, Camera cam,
                        float* d_r, float* d_g, float* d_b, float* d_w,
                        std::vector<float>& ar, std::vector<float>& ag,
                        std::vector<float>& ab, std::vector<float>& aw) {
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    int pixels = W * H;
    int block = 256;
    CUDA_CHECK(cudaEventRecord(e0));
    clear_kernel<<<(pixels + block - 1) / block, block>>>(d_r, d_g, d_b, d_w, pixels);
    splat_kernel<<<(n + block - 1) / block, block>>>(d_gs, n, cam, d_r, d_g, d_b, d_w);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaMemcpy(ar.data(), d_r, pixels * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ag.data(), d_g, pixels * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ab.data(), d_b, pixels * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(aw.data(), d_w, pixels * sizeof(float), cudaMemcpyDeviceToHost));
    return ms;
}

static cv::Mat finalize_color(const std::vector<float>& ar,
                              const std::vector<float>& ag,
                              const std::vector<float>& ab,
                              const std::vector<float>& aw) {
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(246, 247, 248));
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int idx = y * W + x;
            float occ = 1.0f - expf(-0.72f * aw[idx]);
            if (aw[idx] <= 1e-6f || occ < 0.01f) continue;
            float rr = ar[idx] / aw[idx];
            float gg = ag[idx] / aw[idx];
            float bb = ab[idx] / aw[idx];
            cv::Vec3b& p = img.at<cv::Vec3b>(y, x);
            p[0] = static_cast<unsigned char>(255.0f * ((1.0f - occ) + occ * bb));
            p[1] = static_cast<unsigned char>(255.0f * ((1.0f - occ) + occ * gg));
            p[2] = static_cast<unsigned char>(255.0f * ((1.0f - occ) + occ * rr));
        }
    }
    return img;
}

static cv::Mat finalize_weight(const std::vector<float>& aw) {
    cv::Mat gray(H, W, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < W * H; i++) {
        float v = 1.0f - expf(-0.22f * aw[i]);
        gray.data[i] = static_cast<unsigned char>(255.0f * std::min(1.0f, v));
    }
    cv::Mat color;
    cv::applyColorMap(gray, color, cv::COLORMAP_TURBO);
    return color;
}

static void draw_text(cv::Mat& img, const std::string& s, int y) {
    cv::putText(img, s, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.58,
                cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(img, s, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.58,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

static Camera camera_for_frame(int f) {
    float u = 2.0f * PI * f / SIM_FRAMES;
    Vec3 eye = v3(42.0f * cosf(u), 35.0f * sinf(u), 17.0f + 4.0f * sinf(1.7f * u));
    Vec3 target = v3(2.0f * sinf(u), -4.0f, 4.0f);
    return make_camera(eye, target);
}

int main() {
    std::cout << "Gaussian Splatting map renderer: CPU sparse vs CUDA dense" << std::endl;
    std::vector<Gaussian> cpu_gs = build_gaussian_map(N_CPU);
    std::vector<Gaussian> gpu_gs = build_gaussian_map(N_GPU);
    std::vector<Gaussian> check_gs(cpu_gs.begin(), cpu_gs.begin() + N_CHECK);
    Camera check_cam = make_camera(v3(38.0f, -34.0f, 16.0f), v3(0.0f, -2.0f, 4.0f));

    Gaussian *d_check = nullptr, *d_gpu = nullptr;
    float *d_r = nullptr, *d_g = nullptr, *d_b = nullptr, *d_w = nullptr;
    int pixels = W * H;
    CUDA_CHECK(cudaMalloc(&d_check, N_CHECK * sizeof(Gaussian)));
    CUDA_CHECK(cudaMalloc(&d_gpu, N_GPU * sizeof(Gaussian)));
    CUDA_CHECK(cudaMemcpy(d_check, check_gs.data(), N_CHECK * sizeof(Gaussian), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gpu, gpu_gs.data(), N_GPU * sizeof(Gaussian), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_r, pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, pixels * sizeof(float)));

    std::vector<float> cr(pixels), cg(pixels), cb(pixels), cw(pixels);
    std::vector<float> gr(pixels), gg(pixels), gb(pixels), gw(pixels);

    auto t0 = std::chrono::high_resolution_clock::now();
    splat_cpu(check_gs, check_cam, cr, cg, cb, cw);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_check_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    float gpu_check_ms = render_gpu(d_check, N_CHECK, check_cam, d_r, d_g, d_b, d_w,
                                    gr, gg, gb, gw);
    double mae = 0.0;
    for (int i = 0; i < pixels; i++) {
        mae += fabs(cr[i] - gr[i]) + fabs(cg[i] - gg[i]) +
               fabs(cb[i] - gb[i]) + fabs(cw[i] - gw[i]);
    }
    mae /= (pixels * 4.0);
    std::printf("Correctness check (%d Gaussians): CPU %.2f ms, CUDA %.3f ms, "
                "accumulator MAE %.6f\n", N_CHECK, cpu_check_ms, gpu_check_ms, mae);

    cv::VideoWriter video("gif/comparison_gaussian_splatting.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 24,
                          cv::Size(W * 3, H));
    double cpu_sum = 0.0;
    double gpu_sum = 0.0;
    int timed = 0;
    for (int f = 0; f < SIM_FRAMES; f++) {
        Camera cam = camera_for_frame(f);
        auto c0 = std::chrono::high_resolution_clock::now();
        splat_cpu(cpu_gs, cam, cr, cg, cb, cw);
        auto c1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
        float gpu_ms = render_gpu(d_gpu, N_GPU, cam, d_r, d_g, d_b, d_w,
                                  gr, gg, gb, gw);
        if (f >= 5) {
            cpu_sum += cpu_ms;
            gpu_sum += gpu_ms;
            timed++;
        }
        cv::Mat cpu_img = finalize_color(cr, cg, cb, cw);
        cv::Mat gpu_img = finalize_color(gr, gg, gb, gw);
        cv::Mat weight_img = finalize_weight(gw);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "CPU sparse %d Gaussians  %.1f ms", N_CPU, cpu_ms);
        draw_text(cpu_img, buf, 28);
        std::snprintf(buf, sizeof(buf), "CUDA dense %d Gaussians  %.2f ms", N_GPU, gpu_ms);
        draw_text(gpu_img, buf, 28);
        draw_text(weight_img, "CUDA splat density / accumulated opacity", 28);
        cv::Mat combined;
        cv::hconcat(std::vector<cv::Mat>{cpu_img, gpu_img, weight_img}, combined);
        video.write(combined);
    }
    video.release();

    if (timed > 0) {
        double cpu_ms = cpu_sum / timed;
        double gpu_ms = gpu_sum / timed;
        double cpu_us = cpu_ms * 1000.0 / N_CPU;
        double gpu_us = gpu_ms * 1000.0 / N_GPU;
        std::printf("Animated average:\n"
                    "CPU %.2f ms / frame (%d Gaussians)\n"
                    "CUDA %.2f ms / frame (%d Gaussians)\n"
                    "Per-Gaussian throughput: CUDA %.4f us vs CPU %.3f us (%.0fx faster)\n",
                    cpu_ms, N_CPU, gpu_ms, N_GPU, gpu_us, cpu_us, cpu_us / gpu_us);
    }

    std::system("ffmpeg -y -i gif/comparison_gaussian_splatting.avi "
                "-vf 'fps=15,scale=900:-1:flags=lanczos' -loop 0 "
                "gif/comparison_gaussian_splatting.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_gaussian_splatting.gif" << std::endl;

    cudaFree(d_check);
    cudaFree(d_gpu);
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_w);
    return 0;
}
