// gpu_gaussian_splatting.cu
//
// GPU 3D Gaussian Splatting renderer (forward only, isotropic Gaussians).
//
// Scene representation: N 3D Gaussians, each with
//   mean mu (3),  scale s (1, isotropic),  color (rgb),  opacity alpha0.
//
// Per frame:
//   1. host: build a view matrix that orbits around the scene center
//   2. project_kernel transforms each Gaussian into view space,
//      computes screen-space mean and isotropic 2D radius,
//      and writes a depth used for sorting.
//   3. host: sort indices by ascending depth (front-to-back).
//   4. render_kernel: per pixel, iterate sorted Gaussians and alpha-composite:
//        C += T * alpha * color;   T *= (1 - alpha)
//      with early termination when T < 0.005.
//
// Scene: an indoor-style layout (floor + four walls + three colored object
// clusters) totaling ~700 Gaussians.  Pair this with the LiDAR/voxel demos
// in the repo and you get a contiguous "GPU map representation -> render"
// story.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                         cudaGetErrorString(err), __FILE__, __LINE__);    \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

namespace cudabot {

constexpr int IMG_W = 720;
constexpr int IMG_H = 480;
constexpr float FOCAL = 600.0f;     // pixels
constexpr float CX = IMG_W * 0.5f;
constexpr float CY = IMG_H * 0.5f;
constexpr float Z_NEAR = 0.3f;

constexpr int N_FRAMES = 90;

// -------------------------------------------------------------------------
// Scene
// -------------------------------------------------------------------------
struct Gaussian {
    float mx, my, mz;
    float s;
    float r, g, b;
    float a;
};

static void scatter_quad(std::vector<Gaussian>& gs,
                         float cx, float cy, float cz,
                         float ex, float ey, float ez,
                         int nx, int ny, float scale,
                         float cr, float cg, float cb, float alpha,
                         std::mt19937& rng) {
    std::uniform_real_distribution<float> jit(-0.05f, 0.05f);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            float u = (nx == 1) ? 0.0f : static_cast<float>(i) / (nx - 1) - 0.5f;
            float v = (ny == 1) ? 0.0f : static_cast<float>(j) / (ny - 1) - 0.5f;
            Gaussian g;
            g.mx = cx + u * ex + jit(rng);
            g.my = cy + v * ey + jit(rng);
            g.mz = cz + (ez != 0.0f ? jit(rng) * (ez / 0.2f) : 0.0f);
            g.s = scale;
            g.r = cr; g.g = cg; g.b = cb; g.a = alpha;
            gs.push_back(g);
        }
    }
}

static void scatter_blob(std::vector<Gaussian>& gs,
                         float cx, float cy, float cz, float rad,
                         int n, float scale,
                         float cr, float cg, float cb, float alpha,
                         std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, rad / 1.8f);
    for (int i = 0; i < n; i++) {
        Gaussian g;
        g.mx = cx + nd(rng);
        g.my = cy + nd(rng);
        g.mz = cz + nd(rng);
        g.s = scale;
        g.r = cr; g.g = cg; g.b = cb; g.a = alpha;
        gs.push_back(g);
    }
}

static std::vector<Gaussian> build_scene() {
    std::vector<Gaussian> gs;
    std::mt19937 rng(31);

    // Floor: gray plane at y = 0.
    scatter_quad(gs, 0.0f, 0.0f, 0.0f, 6.0f, 6.0f, 0.05f, 30, 30, 0.18f,
                 0.55f, 0.55f, 0.6f, 0.85f, rng);
    // Back wall (z = -3)
    scatter_quad(gs, 0.0f, 1.6f, -3.0f, 6.0f, 0.0f, 0.05f, 25, 1, 0.14f,
                 0.75f, 0.55f, 0.45f, 0.85f, rng);
    // Left wall (x = -3)
    scatter_quad(gs, -3.0f, 1.6f, 0.0f, 0.0f, 0.0f, 0.05f, 1, 1, 0.14f,
                 0.45f, 0.6f, 0.7f, 0.85f, rng);  // single (we add a vertical strip below)
    // Vertical strip along left wall
    {
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 4; j++) {
                Gaussian g;
                g.mx = -3.0f;
                g.mz = (i / 19.0f - 0.5f) * 6.0f;
                g.my = 0.4f + j * 0.7f;
                g.s = 0.14f;
                g.r = 0.45f; g.g = 0.6f; g.b = 0.7f; g.a = 0.85f;
                gs.push_back(g);
            }
        }
    }
    // Right wall (x = 3)
    {
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 4; j++) {
                Gaussian g;
                g.mx = 3.0f;
                g.mz = (i / 19.0f - 0.5f) * 6.0f;
                g.my = 0.4f + j * 0.7f;
                g.s = 0.14f;
                g.r = 0.7f; g.g = 0.55f; g.b = 0.45f; g.a = 0.85f;
                gs.push_back(g);
            }
        }
    }
    // Three colored object clusters on the floor
    scatter_blob(gs, -1.4f, 0.4f, 0.6f, 0.5f, 90, 0.13f, 0.9f, 0.35f, 0.35f, 0.95f, rng);  // red
    scatter_blob(gs,  1.4f, 0.4f, 1.1f, 0.5f, 90, 0.13f, 0.35f, 0.85f, 0.4f, 0.95f, rng);  // green
    scatter_blob(gs,  0.0f, 0.4f, -1.2f, 0.6f, 110, 0.13f, 0.35f, 0.5f, 0.95f, 0.95f, rng); // blue
    return gs;
}

// -------------------------------------------------------------------------
// CUDA kernels
// -------------------------------------------------------------------------
struct Projected {
    float ux, uy;       // screen-space center
    float radius;       // screen-space std-dev (1 sigma in pixels)
    float inv_var;      // 1 / sigma^2 in pixels
    float r, g, b, a;
    float depth;        // view-space z (positive in front of camera)
    int   valid;        // 1 if inside frustum and not behind
};

// view matrix is 3x4 row-major (R | t); R is 3x3, t is 3x1.
__device__ inline void apply_view(const float* __restrict__ V,
                                  float x, float y, float z,
                                  float& vx, float& vy, float& vz) {
    vx = V[0] * x + V[1] * y + V[2]  * z + V[3];
    vy = V[4] * x + V[5] * y + V[6]  * z + V[7];
    vz = V[8] * x + V[9] * y + V[10] * z + V[11];
}

__global__ void project_kernel(int n, const Gaussian* __restrict__ gs,
                               const float* __restrict__ V,
                               Projected* __restrict__ out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    Gaussian g = gs[k];
    float vx, vy, vz;
    apply_view(V, g.mx, g.my, g.mz, vx, vy, vz);

    Projected p;
    p.valid = 0;
    p.depth = vz;
    if (vz < Z_NEAR) {
        out[k] = p;
        return;
    }
    p.ux = CX + FOCAL * (vx / vz);
    // Note: image y axis points down, world y points up.
    p.uy = CY - FOCAL * (vy / vz);
    float r_px = fmaxf(0.5f, FOCAL * g.s / vz);
    p.radius = r_px;
    p.inv_var = 1.0f / (r_px * r_px);
    p.r = g.r; p.g = g.g; p.b = g.b; p.a = g.a;

    // Coarse cull: keep if center is within (image + 3 sigma) margin.
    if (p.ux + 3.0f * r_px < 0.0f || p.ux - 3.0f * r_px >= IMG_W) {
        out[k] = p; return;
    }
    if (p.uy + 3.0f * r_px < 0.0f || p.uy - 3.0f * r_px >= IMG_H) {
        out[k] = p; return;
    }
    p.valid = 1;
    out[k] = p;
}

// Per-pixel: iterate sorted Gaussians, alpha-composite.
__global__ void render_kernel(int n_valid, const Projected* __restrict__ ps,
                              const int* __restrict__ order,
                              unsigned char* __restrict__ img) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= IMG_W || py >= IMG_H) return;
    float cr = 0.06f, cg = 0.07f, cb = 0.10f;  // background sky
    float T = 1.0f;
    float fx = static_cast<float>(px);
    float fy = static_cast<float>(py);
    for (int i = 0; i < n_valid; i++) {
        if (T < 5.0e-3f) break;
        const Projected p = ps[order[i]];
        float dx = fx - p.ux;
        float dy = fy - p.uy;
        float r2 = dx * dx + dy * dy;
        if (r2 > 9.0f * p.radius * p.radius) continue;  // outside 3 sigma
        float w = expf(-0.5f * r2 * p.inv_var) * p.a;
        cr += T * w * p.r;
        cg += T * w * p.g;
        cb += T * w * p.b;
        T *= (1.0f - w);
    }
    int idx = (py * IMG_W + px) * 3;
    img[idx + 0] = static_cast<unsigned char>(fminf(fmaxf(cb * 255.0f, 0.0f), 255.0f));  // BGR
    img[idx + 1] = static_cast<unsigned char>(fminf(fmaxf(cg * 255.0f, 0.0f), 255.0f));
    img[idx + 2] = static_cast<unsigned char>(fminf(fmaxf(cr * 255.0f, 0.0f), 255.0f));
}

// -------------------------------------------------------------------------
// Camera
// -------------------------------------------------------------------------
static void compose_view_matrix(float eye_x, float eye_y, float eye_z,
                                float target_x, float target_y, float target_z,
                                float up_x, float up_y, float up_z,
                                float* V) {
    // Compute right-handed view that looks from eye toward target.
    float fx = target_x - eye_x;
    float fy = target_y - eye_y;
    float fz = target_z - eye_z;
    float fl = std::sqrt(fx * fx + fy * fy + fz * fz);
    fx /= fl; fy /= fl; fz /= fl;
    // right = up x forward
    float rx = up_y * fz - up_z * fy;
    float ry = up_z * fx - up_x * fz;
    float rz = up_x * fy - up_y * fx;
    float rl = std::sqrt(rx * rx + ry * ry + rz * rz);
    rx /= rl; ry /= rl; rz /= rl;
    // recompute up = forward x right (orthogonalize)
    float ux = fy * rz - fz * ry;
    float uy = fz * rx - fx * rz;
    float uz = fx * ry - fy * rx;
    // View matrix rows: [right; up; forward], translation = -R * eye
    V[0]  = rx; V[1]  = ry; V[2]  = rz; V[3]  = -(rx * eye_x + ry * eye_y + rz * eye_z);
    V[4]  = ux; V[5]  = uy; V[6]  = uz; V[7]  = -(ux * eye_x + uy * eye_y + uz * eye_z);
    V[8]  = fx; V[9]  = fy; V[10] = fz; V[11] = -(fx * eye_x + fy * eye_y + fz * eye_z);
}

static void convert_avi_to_gif(const std::string& avi, const std::string& gif, int fps) {
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -i %s -vf \"fps=%d,scale=1080:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle\" %s 2>/dev/null",
                  avi.c_str(), fps, gif.c_str());
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg failed (%d)\n", rc);
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    auto scene = build_scene();
    int n = static_cast<int>(scene.size());
    std::printf("Scene: %d Gaussians  (%dx%d image)\n", n, IMG_W, IMG_H);

    Gaussian* d_gs = nullptr;
    Projected* d_ps = nullptr;
    int* d_order = nullptr;
    float* d_V = nullptr;
    unsigned char* d_img = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gs, n * sizeof(Gaussian)));
    CUDA_CHECK(cudaMalloc(&d_ps, n * sizeof(Projected)));
    CUDA_CHECK(cudaMalloc(&d_order, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_V, 12 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img, IMG_W * IMG_H * 3));
    CUDA_CHECK(cudaMemcpy(d_gs, scene.data(), n * sizeof(Gaussian), cudaMemcpyHostToDevice));

    std::vector<Projected> ps(n);
    std::vector<int> order(n);
    std::vector<unsigned char> img(IMG_W * IMG_H * 3);

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_gaussian_splatting.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          15, cv::Size(IMG_W, IMG_H));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms_total = 0.0f;

    for (int f = 0; f < N_FRAMES; f++) {
        float u = static_cast<float>(f) / N_FRAMES * 2.0f * static_cast<float>(M_PI);
        float radius = 6.5f;
        float eye_x = radius * std::sin(u);
        float eye_y = 1.9f + 0.5f * std::sin(2.0f * u);
        float eye_z = radius * std::cos(u);
        float V[12];
        compose_view_matrix(eye_x, eye_y, eye_z, 0.0f, 0.7f, 0.0f, 0.0f, 1.0f, 0.0f, V);
        CUDA_CHECK(cudaMemcpy(d_V, V, 12 * sizeof(float), cudaMemcpyHostToDevice));

        cudaEventRecord(t0);

        int blk = 128;
        int blocks = (n + blk - 1) / blk;
        project_kernel<<<blocks, blk>>>(n, d_gs, d_V, d_ps);
        CUDA_CHECK(cudaMemcpy(ps.data(), d_ps, n * sizeof(Projected), cudaMemcpyDeviceToHost));

        // Build sorted order on host (front-to-back).
        order.clear();
        for (int k = 0; k < n; k++) if (ps[k].valid) order.push_back(k);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return ps[a].depth < ps[b].depth; });
        int n_valid = static_cast<int>(order.size());
        CUDA_CHECK(cudaMemcpy(d_order, order.data(), n_valid * sizeof(int), cudaMemcpyHostToDevice));

        dim3 blockDim(16, 16);
        dim3 gridDim((IMG_W + 15) / 16, (IMG_H + 15) / 16);
        render_kernel<<<gridDim, blockDim>>>(n_valid, d_ps, d_order, d_img);

        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms = 0.0f; cudaEventElapsedTime(&ms, t0, t1);
        ms_total += ms;

        CUDA_CHECK(cudaMemcpy(img.data(), d_img, IMG_W * IMG_H * 3, cudaMemcpyDeviceToHost));
        cv::Mat frame(IMG_H, IMG_W, CV_8UC3, img.data());
        // overlay text
        char buf[128];
        std::snprintf(buf, sizeof(buf), "GPU 3D Gaussian Splatting   N=%d   visible=%d   %.1f ms",
                      n, n_valid, ms);
        cv::putText(frame, buf, cv::Point(10, 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
        video.write(frame);
    }

    video.release();
    std::printf("Avg render time: %.2f ms / frame  (%d frames, %d Gaussians)\n",
                ms_total / N_FRAMES, N_FRAMES, n);
    convert_avi_to_gif("gif/gpu_gaussian_splatting.avi", "gif/gpu_gaussian_splatting.gif", 15);
    std::printf("GIF saved to gif/gpu_gaussian_splatting.gif\n");

    cudaFree(d_gs); cudaFree(d_ps); cudaFree(d_order); cudaFree(d_V); cudaFree(d_img);
    return 0;
}
