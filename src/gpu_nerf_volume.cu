// gpu_nerf_volume.cu
//
// GPU NeRF-style volumetric renderer.
//
// NeRF (Neural Radiance Fields) represents a scene as a function
//      F: (x, y, z, view_dir) -> (density σ, RGB color)
// learned by an MLP.  The function is queried at many sample points along
// each camera ray and the samples are composited with the volume-rendering
// equation:
//      C = sum_i T_i (1 - exp(-σ_i Δs)) c_i,
//      T_i = exp(- sum_{j<i} σ_j Δs).
//
// This program implements the GPU rendering pipeline that consumes that
// (σ, c) function — exactly what runs over a trained NeRF at inference
// time — and substitutes an analytic scene field for the MLP so we can
// demo the renderer without 6 hours of training.  Swap the
// scene_eval(...) device function for an MLP forward pass and you have
// a real NeRF renderer.
//
// Setup:
//   - Image: 720 x 480, 128 samples per ray, stratified + jittered
//   - Scene: 6 colored spheres + a floor plane (smoothed analytic density)
//   - Camera orbits the scene over 90 frames
//
// Output: gif/gpu_nerf_volume.gif

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "cuda_check.cuh"

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
constexpr int N_SAMPLES = 128;
constexpr int N_FRAMES = 90;
constexpr float NEAR = 1.0f;
constexpr float FAR = 9.0f;
constexpr float FOV_DEG = 45.0f;
constexpr int MAX_SPHERES = 6;

struct Sphere { float cx, cy, cz, r; float R, G, B; };

__constant__ Sphere c_spheres[MAX_SPHERES];
__constant__ int c_n_spheres;
__constant__ float c_floor_y;
__constant__ float c_floor_thickness;

// Scene "MLP stand-in": evaluate density and color at world point p.
// Mathematically: σ(p) = floor density (smooth box) + Σ sphere Gaussians,
//                 c(p) = color of the dominant primitive at p.
__device__ inline void scene_eval(float px, float py, float pz,
                                  float& sigma, float& cR, float& cG, float& cB) {
    sigma = 0.0f;
    float total_w = 0.0f;
    cR = 0.0f; cG = 0.0f; cB = 0.0f;

    // Spheres: smooth Gaussian density falling off with sphere radius.
    for (int s = 0; s < c_n_spheres; s++) {
        Sphere sp = c_spheres[s];
        float dx = px - sp.cx, dy = py - sp.cy, dz = pz - sp.cz;
        float d2 = dx * dx + dy * dy + dz * dz;
        float inv2sigma2 = 1.0f / (0.4f * sp.r * sp.r);
        float w = expf(-d2 * inv2sigma2);
        float sphere_density = 12.0f * w;
        sigma += sphere_density;
        cR += sphere_density * sp.R;
        cG += sphere_density * sp.G;
        cB += sphere_density * sp.B;
        total_w += sphere_density;
    }

    // Floor: thin slab around y = c_floor_y, with mild checkerboard tint.
    float dy = py - c_floor_y;
    float floor_density = 18.0f * expf(-(dy * dy) / (c_floor_thickness * c_floor_thickness));
    // a checker pattern tints the floor color
    int ix = static_cast<int>(floorf(px * 1.0f));
    int iz = static_cast<int>(floorf(pz * 1.0f));
    float checker = ((ix + iz) & 1) ? 0.55f : 0.30f;
    float floor_R = checker * 0.5f;
    float floor_G = checker * 0.45f;
    float floor_B = checker * 0.4f;
    sigma += floor_density;
    cR += floor_density * floor_R;
    cG += floor_density * floor_G;
    cB += floor_density * floor_B;
    total_w += floor_density;

    // Normalize color
    if (total_w > 1e-6f) {
        cR /= total_w;
        cG /= total_w;
        cB /= total_w;
    } else {
        cR = cG = cB = 0.0f;
    }
}

__device__ inline void sky_color(float dx, float dy, float dz,
                                 float& sR, float& sG, float& sB) {
    float t = 0.5f * (dy + 1.0f);
    sR = (1.0f - t) * 0.85f + t * 0.55f;
    sG = (1.0f - t) * 0.92f + t * 0.78f;
    sB = (1.0f - t) * 1.00f + t * 0.95f;
}

__global__ void render_kernel(float cam_x, float cam_y, float cam_z,
                              float right_x, float right_y, float right_z,
                              float up_x, float up_y, float up_z,
                              float fwd_x, float fwd_y, float fwd_z,
                              float tan_half_fov, float aspect,
                              unsigned long long seed,
                              unsigned char* __restrict__ img_bgr) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= IMG_W || py >= IMG_H) return;

    // Pixel direction in camera space
    float u = (2.0f * (px + 0.5f) / IMG_W - 1.0f) * tan_half_fov * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / IMG_H) * tan_half_fov;
    float dx = u * right_x + v * up_x + fwd_x;
    float dy = u * right_y + v * up_y + fwd_y;
    float dz = u * right_z + v * up_z + fwd_z;
    float inv = rsqrtf(dx * dx + dy * dy + dz * dz);
    dx *= inv; dy *= inv; dz *= inv;

    // Stratified jitter for sample positions
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, px + py * IMG_W, 0, &rng);

    float t_step = (FAR - NEAR) / N_SAMPLES;
    float T = 1.0f;
    float accR = 0.0f, accG = 0.0f, accB = 0.0f;

    for (int s = 0; s < N_SAMPLES; s++) {
        float jitter = curand_uniform(&rng);
        float t_lo = NEAR + s * t_step;
        float t = t_lo + jitter * t_step;
        float sx = cam_x + t * dx;
        float sy = cam_y + t * dy;
        float sz = cam_z + t * dz;
        float sigma, cR, cG, cB;
        scene_eval(sx, sy, sz, sigma, cR, cG, cB);
        float alpha = 1.0f - expf(-sigma * t_step);
        float w = T * alpha;
        accR += w * cR;
        accG += w * cG;
        accB += w * cB;
        T *= 1.0f - alpha;
        if (T < 5e-3f) break;
    }
    // Sky for remaining transmittance
    float sR, sG, sB;
    sky_color(dx, dy, dz, sR, sG, sB);
    accR += T * sR;
    accG += T * sG;
    accB += T * sB;

    auto to_u8 = [](float c) {
        c = fminf(fmaxf(c, 0.0f), 1.0f);
        return static_cast<unsigned char>(c * 255.0f + 0.5f);
    };
    int idx = (py * IMG_W + px) * 3;
    img_bgr[idx + 0] = to_u8(accB);  // BGR for OpenCV
    img_bgr[idx + 1] = to_u8(accG);
    img_bgr[idx + 2] = to_u8(accR);
}

static void convert_avi_to_gif(const std::string& avi, const std::string& gif, int fps) {
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -i %s -vf \"fps=%d,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle\" %s 2>/dev/null",
                  avi.c_str(), fps, gif.c_str());
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg failed (%d)\n", rc);
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    // Define scene: 6 colored spheres above a floor at y = -0.5
    std::vector<Sphere> spheres = {
        {  0.6f,  0.0f,  0.0f, 0.55f, 0.95f, 0.45f, 0.25f},   // orange
        { -1.1f,  0.05f, 0.4f, 0.45f, 0.30f, 0.80f, 0.95f},   // cyan
        {  0.2f,  0.0f, -1.2f, 0.50f, 0.80f, 0.30f, 0.85f},   // magenta
        { -0.4f,  0.6f,  0.3f, 0.32f, 0.95f, 0.92f, 0.30f},   // yellow
        {  1.2f,  0.4f, -0.4f, 0.30f, 0.40f, 0.95f, 0.35f},   // green
        { -1.4f, -0.1f, -0.7f, 0.42f, 0.95f, 0.40f, 0.70f},   // pink
    };
    int n_spheres = static_cast<int>(spheres.size());

    CUDA_CHECK(cudaMemcpyToSymbol(c_spheres, spheres.data(),
                                  n_spheres * sizeof(Sphere)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_n_spheres, &n_spheres, sizeof(int)));
    float floor_y = -0.5f;
    float floor_thickness = 0.08f;
    CUDA_CHECK(cudaMemcpyToSymbol(c_floor_y, &floor_y, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_floor_thickness, &floor_thickness, sizeof(float)));

    // Image buffer
    unsigned char* d_img = nullptr;
    CUDA_CHECK(cudaMalloc(&d_img, IMG_W * IMG_H * 3));
    std::vector<unsigned char> h_img(IMG_W * IMG_H * 3);

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/gpu_nerf_volume.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          24, cv::Size(IMG_W, IMG_H));

    float tan_half_fov = std::tan(FOV_DEG * static_cast<float>(M_PI) / 180.0f * 0.5f);
    float aspect = static_cast<float>(IMG_W) / IMG_H;

    dim3 block(16, 16);
    dim3 grid((IMG_W + block.x - 1) / block.x,
              (IMG_H + block.y - 1) / block.y);

    double total_ms = 0.0;
    int counted = 0;

    for (int f = 0; f < N_FRAMES; f++) {
        float angle = static_cast<float>(f) / N_FRAMES * 2.0f * static_cast<float>(M_PI);
        float radius = 5.5f;
        float cam_x = radius * std::cos(angle);
        float cam_z = radius * std::sin(angle);
        float cam_y = 1.6f + 0.4f * std::sin(angle * 0.5f);

        // Look at scene centre (0, 0, 0)
        float fwd_x = -cam_x, fwd_y = -cam_y, fwd_z = -cam_z;
        float inv = 1.0f / std::sqrt(fwd_x * fwd_x + fwd_y * fwd_y + fwd_z * fwd_z);
        fwd_x *= inv; fwd_y *= inv; fwd_z *= inv;
        // world up = (0, 1, 0)
        // right = normalize(fwd x up_world)
        float up_w_x = 0.0f, up_w_y = 1.0f, up_w_z = 0.0f;
        float right_x = fwd_y * up_w_z - fwd_z * up_w_y;
        float right_y = fwd_z * up_w_x - fwd_x * up_w_z;
        float right_z = fwd_x * up_w_y - fwd_y * up_w_x;
        float rinv = 1.0f / std::sqrt(right_x * right_x + right_y * right_y + right_z * right_z);
        right_x *= rinv; right_y *= rinv; right_z *= rinv;
        // up = right x fwd
        float up_x = right_y * fwd_z - right_z * fwd_y;
        float up_y = right_z * fwd_x - right_x * fwd_z;
        float up_z = right_x * fwd_y - right_y * fwd_x;

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        render_kernel<<<grid, block>>>(cam_x, cam_y, cam_z,
                                        right_x, right_y, right_z,
                                        up_x, up_y, up_z,
                                        fwd_x, fwd_y, fwd_z,
                                        tan_half_fov, aspect,
                                        2027ULL + f, d_img);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        if (f >= 2) { total_ms += ms; counted++; }

        CUDA_CHECK(cudaMemcpy(h_img.data(), d_img, IMG_W * IMG_H * 3,
                              cudaMemcpyDeviceToHost));
        cv::Mat img(IMG_H, IMG_W, CV_8UC3, h_img.data());
        cv::Mat frame = img.clone();
        // Footer
        cv::rectangle(frame, cv::Rect(0, IMG_H - 30, IMG_W, 30),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "GPU NeRF-style volumetric renderer  720x480  %d samples/ray  %.2f ms",
                      N_SAMPLES, ms);
        cv::putText(frame, buf, cv::Point(10, IMG_H - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        video.write(frame);
        if (f % 10 == 0) std::printf("  frame %3d  %.2f ms\n", f, ms);
    }
    video.release();

    if (counted > 0) {
        std::printf("Avg per-frame render time: %.2f ms (%dx%d, %d samples/ray)\n",
                    total_ms / counted, IMG_W, IMG_H, N_SAMPLES);
    }
    convert_avi_to_gif("gif/gpu_nerf_volume.avi", "gif/gpu_nerf_volume.gif", 24);
    std::printf("GIF saved to gif/gpu_nerf_volume.gif\n");

    cudaFree(d_img);
    return 0;
}
