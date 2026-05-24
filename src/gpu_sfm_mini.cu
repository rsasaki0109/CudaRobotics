// gpu_sfm_mini.cu
//
// Mini structure-from-motion pipeline on GPU.
//
// Synthetic 4-view feature tracks are generated with ORB-like 256-bit
// descriptors.  CUDA brute-force matches view 0 -> view 1, triangulates
// the stereo pair, then runs fixed-camera point-only bundle adjustment for
// all 3D points.  The CPU path uses the same matching and BA math for a
// direct timing comparison.
//
// Output: gif/gpu_sfm_mini.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int N_POINTS = 2048;
constexpr int N_VIEWS = 4;
constexpr int DESC_WORDS = 8;
constexpr int IMG_W = 640;
constexpr int IMG_H = 420;
constexpr float FOCAL = 470.0f;
constexpr float CX = IMG_W * 0.5f;
constexpr float CY = IMG_H * 0.5f;
constexpr int BA_ITERS = 18;
constexpr int N_BENCH = 30;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int VIDEO_FPS = 12;

struct Cam {
    float x;
    float y;
    float z;
};

struct Point3 {
    float x;
    float y;
    float z;
};

struct Feature {
    float u;
    float v;
    unsigned int desc[DESC_WORDS];
};

struct BenchResult {
    float gpu_match_ms = 0.0f;
    float gpu_ba_ms = 0.0f;
    double cpu_match_ms = 0.0;
    double cpu_ba_ms = 0.0;
    double speedup = 0.0;
};

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline float sqr(float x) {
    return x * x;
}

__host__ __device__ static inline unsigned int mix_u32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__host__ __device__ static inline bool project_point(const Cam& cam,
                                                     const Point3& p,
                                                     float& u,
                                                     float& v) {
    float x = p.x - cam.x;
    float y = p.y - cam.y;
    float z = p.z - cam.z;
    if (z <= 0.2f) return false;
    u = FOCAL * x / z + CX;
    v = FOCAL * y / z + CY;
    return true;
}

__device__ static inline int hamming_gpu(const unsigned int* a,
                                         const unsigned int* b) {
    int d = 0;
    #pragma unroll
    for (int k = 0; k < DESC_WORDS; k++) d += __popc(a[k] ^ b[k]);
    return d;
}

static int hamming_cpu(const unsigned int* a, const unsigned int* b) {
    int d = 0;
    for (int k = 0; k < DESC_WORDS; k++) d += __builtin_popcount(a[k] ^ b[k]);
    return d;
}

__global__ void match_descriptors_kernel(const Feature* f0,
                                         const Feature* f1,
                                         int* matches,
                                         int* best_dist,
                                         int* second_dist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_POINTS) return;
    int best = 9999;
    int second = 9999;
    int best_j = -1;
    for (int j = 0; j < N_POINTS; j++) {
        int d = hamming_gpu(f0[i].desc, f1[j].desc);
        if (d < best) {
            second = best;
            best = d;
            best_j = j;
        } else if (d < second) {
            second = d;
        }
    }
    bool accept = best <= 34 && (float)best < 0.72f * (float)second;
    matches[i] = accept ? best_j : -1;
    best_dist[i] = best;
    second_dist[i] = second;
}

__global__ void triangulate_kernel(const Feature* features,
                                   const Cam* cams,
                                   const int* matches,
                                   Point3* points) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_POINTS) return;
    int j = matches[i];
    if (j < 0) {
        points[i] = {0.0f, 0.0f, 6.0f};
        return;
    }
    const Feature& a = features[0 * N_POINTS + i];
    const Feature& b = features[1 * N_POINTS + j];
    float baseline = cams[1].x - cams[0].x;
    float disparity = a.u - b.u;
    float z = FOCAL * baseline / fmaxf(1.0f, disparity);
    z = clampf(z, 3.0f, 11.5f);
    float x0 = (a.u - CX) * z / FOCAL + cams[0].x;
    float y0 = (a.v - CY) * z / FOCAL + cams[0].y;
    float y1 = (b.v - CY) * z / FOCAL + cams[1].y;
    points[i] = {x0, 0.5f * (y0 + y1), z};
}

__host__ __device__ static inline bool solve3(float H[9], float b[3], float x[3]) {
    float l00 = sqrtf(fmaxf(H[0], 1.0e-10f));
    float l10 = H[3] / l00;
    float l20 = H[6] / l00;
    float t11 = H[4] - l10 * l10;
    if (t11 <= 1.0e-10f) return false;
    float l11 = sqrtf(t11);
    float l21 = (H[7] - l20 * l10) / l11;
    float t22 = H[8] - l20 * l20 - l21 * l21;
    if (t22 <= 1.0e-10f) return false;
    float l22 = sqrtf(t22);

    float y0 = b[0] / l00;
    float y1 = (b[1] - l10 * y0) / l11;
    float y2 = (b[2] - l20 * y0 - l21 * y1) / l22;
    x[2] = y2 / l22;
    x[1] = (y1 - l21 * x[2]) / l11;
    x[0] = (y0 - l10 * x[1] - l20 * x[2]) / l00;
    return true;
}

__global__ void ba_points_kernel(const Feature* features,
                                 const Cam* cams,
                                 Point3* points,
                                 int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_POINTS) return;
    Point3 p = points[idx];
    for (int it = 0; it < iterations; it++) {
        float H[9] = {
            1.0e-3f, 0.0f, 0.0f,
            0.0f, 1.0e-3f, 0.0f,
            0.0f, 0.0f, 1.0e-3f
        };
        float b[3] = {0.0f, 0.0f, 0.0f};
        for (int view = 0; view < N_VIEWS; view++) {
            const Cam& c = cams[view];
            const Feature& obs = features[view * N_POINTS + idx];
            float x = p.x - c.x;
            float y = p.y - c.y;
            float z = fmaxf(0.25f, p.z - c.z);
            float invz = 1.0f / z;
            float pred_u = FOCAL * x * invz + CX;
            float pred_v = FOCAL * y * invz + CY;
            float r0 = obs.u - pred_u;
            float r1 = obs.v - pred_v;
            float J[6] = {
                FOCAL * invz, 0.0f, -FOCAL * x * invz * invz,
                0.0f, FOCAL * invz, -FOCAL * y * invz * invz
            };
            for (int a = 0; a < 3; a++) {
                b[a] += J[a] * r0 + J[3 + a] * r1;
                for (int bb = 0; bb < 3; bb++) {
                    H[a * 3 + bb] += J[a] * J[bb] + J[3 + a] * J[3 + bb];
                }
            }
        }
        float dx[3];
        if (!solve3(H, b, dx)) break;
        float scale = 1.0f;
        float step = sqrtf(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
        if (step > 0.55f) scale = 0.55f / step;
        p.x += scale * dx[0];
        p.y += scale * dx[1];
        p.z = clampf(p.z + scale * dx[2], 2.5f, 12.5f);
    }
    points[idx] = p;
}

static std::array<Cam, N_VIEWS> make_cameras() {
    return {{
        {-1.20f,  0.00f, 0.0f},
        {-0.40f,  0.04f, 0.0f},
        { 0.42f, -0.03f, 0.0f},
        { 1.20f,  0.03f, 0.0f},
    }};
}

static std::vector<Point3> make_points() {
    std::vector<Point3> pts(N_POINTS);
    std::mt19937 rng(24052026);
    std::uniform_real_distribution<float> ux(-2.2f, 2.2f);
    std::uniform_real_distribution<float> uy(-1.15f, 1.15f);
    std::uniform_real_distribution<float> uz(4.8f, 9.5f);
    for (int i = 0; i < N_POINTS; i++) {
        float band = (float)(i % 8) / 7.0f - 0.5f;
        pts[i] = {
            ux(rng) + 0.28f * sinf(0.013f * i),
            uy(rng) + 0.22f * band,
            uz(rng)
        };
    }
    return pts;
}

static void make_features(const std::vector<Point3>& points,
                          const std::array<Cam, N_VIEWS>& cams,
                          std::vector<Feature>& features) {
    features.resize(N_VIEWS * N_POINTS);
    std::mt19937 rng(424242);
    std::normal_distribution<float> pix_noise(0.0f, 0.58f);
    for (int i = 0; i < N_POINTS; i++) {
        unsigned int base[DESC_WORDS];
        for (int k = 0; k < DESC_WORDS; k++) {
            base[k] = mix_u32(0x9e3779b9u ^ (unsigned int)(i * 1315423911u)
                            ^ (unsigned int)(k * 2654435761u));
        }
        for (int v = 0; v < N_VIEWS; v++) {
            Feature f{};
            float u, vv;
            project_point(cams[v], points[i], u, vv);
            f.u = clampf(u + pix_noise(rng), 2.0f, (float)IMG_W - 3.0f);
            f.v = clampf(vv + pix_noise(rng), 2.0f, (float)IMG_H - 3.0f);
            for (int k = 0; k < DESC_WORDS; k++) {
                unsigned int flips = 0u;
                unsigned int seed = mix_u32((unsigned int)(i * 977 + v * 31 + k * 17));
                for (int bit = 0; bit < 32; bit++) {
                    if (((seed >> (bit % 23)) & 31u) == 0u) flips ^= (1u << bit);
                }
                f.desc[k] = base[k] ^ flips;
            }
            features[v * N_POINTS + i] = f;
        }
    }
}

static void match_cpu(const Feature* f0,
                      const Feature* f1,
                      std::vector<int>& matches,
                      std::vector<int>& best_dist) {
    matches.assign(N_POINTS, -1);
    best_dist.assign(N_POINTS, 9999);
    for (int i = 0; i < N_POINTS; i++) {
        int best = 9999;
        int second = 9999;
        int best_j = -1;
        for (int j = 0; j < N_POINTS; j++) {
            int d = hamming_cpu(f0[i].desc, f1[j].desc);
            if (d < best) {
                second = best;
                best = d;
                best_j = j;
            } else if (d < second) {
                second = d;
            }
        }
        if (best <= 34 && (float)best < 0.72f * (float)second) matches[i] = best_j;
        best_dist[i] = best;
    }
}

static std::vector<Point3> triangulate_cpu(const std::vector<Feature>& features,
                                           const std::array<Cam, N_VIEWS>& cams,
                                           const std::vector<int>& matches) {
    std::vector<Point3> pts(N_POINTS);
    for (int i = 0; i < N_POINTS; i++) {
        int j = matches[i];
        if (j < 0) {
            pts[i] = {0.0f, 0.0f, 6.0f};
            continue;
        }
        const Feature& a = features[0 * N_POINTS + i];
        const Feature& b = features[1 * N_POINTS + j];
        float baseline = cams[1].x - cams[0].x;
        float z = FOCAL * baseline / std::max(1.0f, a.u - b.u);
        z = clampf(z, 3.0f, 11.5f);
        float x = (a.u - CX) * z / FOCAL + cams[0].x;
        float y0 = (a.v - CY) * z / FOCAL + cams[0].y;
        float y1 = (b.v - CY) * z / FOCAL + cams[1].y;
        pts[i] = {x, 0.5f * (y0 + y1), z};
    }
    return pts;
}

static void ba_cpu(const std::vector<Feature>& features,
                   const std::array<Cam, N_VIEWS>& cams,
                   std::vector<Point3>& pts,
                   int iterations) {
    for (int i = 0; i < N_POINTS; i++) {
        Point3 p = pts[i];
        for (int it = 0; it < iterations; it++) {
            float H[9] = {
                1.0e-3f, 0.0f, 0.0f,
                0.0f, 1.0e-3f, 0.0f,
                0.0f, 0.0f, 1.0e-3f
            };
            float b[3] = {0.0f, 0.0f, 0.0f};
            for (int view = 0; view < N_VIEWS; view++) {
                const Cam& c = cams[view];
                const Feature& obs = features[view * N_POINTS + i];
                float x = p.x - c.x;
                float y = p.y - c.y;
                float z = std::max(0.25f, p.z - c.z);
                float invz = 1.0f / z;
                float pred_u = FOCAL * x * invz + CX;
                float pred_v = FOCAL * y * invz + CY;
                float r0 = obs.u - pred_u;
                float r1 = obs.v - pred_v;
                float J[6] = {
                    FOCAL * invz, 0.0f, -FOCAL * x * invz * invz,
                    0.0f, FOCAL * invz, -FOCAL * y * invz * invz
                };
                for (int a = 0; a < 3; a++) {
                    b[a] += J[a] * r0 + J[3 + a] * r1;
                    for (int bb = 0; bb < 3; bb++) {
                        H[a * 3 + bb] += J[a] * J[bb] + J[3 + a] * J[3 + bb];
                    }
                }
            }
            float dx[3];
            if (!solve3(H, b, dx)) break;
            float step = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
            float scale = (step > 0.55f) ? 0.55f / step : 1.0f;
            p.x += scale * dx[0];
            p.y += scale * dx[1];
            p.z = clampf(p.z + scale * dx[2], 2.5f, 12.5f);
        }
        pts[i] = p;
    }
}

static float rmse3d(const std::vector<Point3>& a, const std::vector<Point3>& gt) {
    double sum = 0.0;
    for (int i = 0; i < N_POINTS; i++) {
        sum += sqr(a[i].x - gt[i].x) + sqr(a[i].y - gt[i].y) + sqr(a[i].z - gt[i].z);
    }
    return (float)std::sqrt(sum / N_POINTS);
}

static float reproj_rmse(const std::vector<Point3>& pts,
                         const std::vector<Feature>& features,
                         const std::array<Cam, N_VIEWS>& cams) {
    double sum = 0.0;
    int n = 0;
    for (int i = 0; i < N_POINTS; i++) {
        for (int v = 0; v < N_VIEWS; v++) {
            float u, vv;
            project_point(cams[v], pts[i], u, vv);
            const Feature& f = features[v * N_POINTS + i];
            sum += sqr(u - f.u) + sqr(vv - f.v);
            n += 2;
        }
    }
    return (float)std::sqrt(sum / std::max(1, n));
}

static BenchResult benchmark(const std::vector<Feature>& features,
                             const std::array<Cam, N_VIEWS>& cams,
                             Feature* d_features,
                             Cam* d_cams,
                             int* d_matches,
                             int* d_best,
                             int* d_second,
                             Point3* d_points) {
    BenchResult br;
    CUDA_CHECK(cudaMemcpy(d_features, features.data(),
                          features.size() * sizeof(Feature), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cams, cams.data(), cams.size() * sizeof(Cam),
                          cudaMemcpyHostToDevice));

    match_descriptors_kernel<<<(N_POINTS + 255) / 256, 256>>>(
        d_features, d_features + N_POINTS, d_matches, d_best, d_second);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < N_BENCH; i++) {
        match_descriptors_kernel<<<(N_POINTS + 255) / 256, 256>>>(
            d_features, d_features + N_POINTS, d_matches, d_best, d_second);
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float total_match = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_match, e0, e1));
    br.gpu_match_ms = total_match / N_BENCH;

    triangulate_kernel<<<(N_POINTS + 255) / 256, 256>>>(d_features, d_cams,
                                                       d_matches, d_points);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < N_BENCH; i++) {
        triangulate_kernel<<<(N_POINTS + 255) / 256, 256>>>(d_features, d_cams,
                                                           d_matches, d_points);
        ba_points_kernel<<<(N_POINTS + 255) / 256, 256>>>(d_features, d_cams,
                                                         d_points, BA_ITERS);
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float total_ba = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ba, e0, e1));
    br.gpu_ba_ms = total_ba / N_BENCH;
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    std::vector<int> cpu_matches;
    std::vector<int> cpu_best;
    auto t0 = std::chrono::high_resolution_clock::now();
    match_cpu(features.data(), features.data() + N_POINTS, cpu_matches, cpu_best);
    auto t1 = std::chrono::high_resolution_clock::now();
    br.cpu_match_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto init = triangulate_cpu(features, cams, cpu_matches);
    t0 = std::chrono::high_resolution_clock::now();
    ba_cpu(features, cams, init, BA_ITERS);
    t1 = std::chrono::high_resolution_clock::now();
    br.cpu_ba_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    br.speedup = (br.cpu_match_ms + br.cpu_ba_ms)
               / std::max(1.0e-9, (double)(br.gpu_match_ms + br.gpu_ba_ms));
    return br;
}

static cv::Scalar color_for_id(int i) {
    static const cv::Scalar colors[8] = {
        cv::Scalar(90, 170, 255), cv::Scalar(90, 230, 135),
        cv::Scalar(255, 190, 75), cv::Scalar(235, 105, 125),
        cv::Scalar(180, 138, 255), cv::Scalar(88, 218, 220),
        cv::Scalar(230, 228, 95), cv::Scalar(255, 128, 204),
    };
    return colors[i % 8];
}

static cv::Point img_point(const Feature& f, const cv::Rect& r) {
    return cv::Point(r.x + (int)(f.u / IMG_W * r.width),
                     r.y + (int)(f.v / IMG_H * r.height));
}

static cv::Point cloud_point(const Point3& p, const cv::Rect& r) {
    float x01 = clampf((p.x + 2.8f) / 5.6f, 0.0f, 1.0f);
    float z01 = clampf((p.z - 4.0f) / 6.5f, 0.0f, 1.0f);
    return cv::Point(r.x + (int)(x01 * r.width),
                     r.y + r.height - (int)(z01 * r.height));
}

static void draw_history(cv::Mat& img,
                         const std::vector<float>& reproj_hist,
                         const std::vector<float>& rmse_hist,
                         const cv::Rect& r) {
    cv::rectangle(img, r, cv::Scalar(29, 31, 36), -1);
    cv::rectangle(img, r, cv::Scalar(76, 80, 88), 1);
    cv::putText(img, "BA convergence", cv::Point(r.x + 12, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(235, 235, 235), 1,
                cv::LINE_AA);
    auto draw_curve = [&](const std::vector<float>& h, float maxv, cv::Scalar c) {
        if (h.size() < 2) return;
        std::vector<cv::Point> pts;
        for (size_t i = 0; i < h.size(); i++) {
            float x01 = (float)i / std::max<size_t>(1, h.size() - 1);
            float y01 = clampf(h[i] / maxv, 0.0f, 1.0f);
            int x = r.x + 38 + (int)(x01 * (r.width - 52));
            int y = r.y + r.height - 18 - (int)(y01 * (r.height - 52));
            pts.emplace_back(x, y);
        }
        cv::polylines(img, pts, false, c, 2, cv::LINE_AA);
    };
    for (int g = 0; g <= 4; g++) {
        int y = r.y + r.height - 18 - g * (r.height - 52) / 4;
        cv::line(img, cv::Point(r.x + 38, y), cv::Point(r.x + r.width - 12, y),
                 cv::Scalar(46, 49, 55), 1);
    }
    draw_curve(reproj_hist, 2.0f, cv::Scalar(90, 170, 255));
    draw_curve(rmse_hist, 0.12f, cv::Scalar(90, 230, 135));
    cv::putText(img, "px", cv::Point(r.x + 138, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(90, 170, 255), 1);
    cv::putText(img, "3D", cv::Point(r.x + 174, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(90, 230, 135), 1);
}

static cv::Mat draw_frame(const std::vector<Feature>& features,
                          const std::vector<int>& matches,
                          const std::vector<Point3>& gt,
                          const std::vector<Point3>& init,
                          const std::vector<Point3>& opt,
                          const std::vector<float>& reproj_hist,
                          const std::vector<float>& rmse_hist,
                          const BenchResult& bench,
                          int iter,
                          int correct_matches) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 19, 23));
    cv::putText(img, cv::format("GPU SfM mini  BA iter %02d / %d", iter, BA_ITERS),
                cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.72,
                cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img,
                cv::format("%d features x %d views   GPU match %.3f ms + BA %.3f ms   CPU %.3f ms   %.1fx",
                           N_POINTS, N_VIEWS, bench.gpu_match_ms, bench.gpu_ba_ms,
                           bench.cpu_match_ms + bench.cpu_ba_ms, bench.speedup),
                cv::Point(18, 54), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                cv::Scalar(210, 214, 220), 1, cv::LINE_AA);

    cv::Rect v0(28, 82, 280, 184);
    cv::Rect v1(328, 82, 280, 184);
    cv::rectangle(img, v0, cv::Scalar(25, 27, 31), -1);
    cv::rectangle(img, v1, cv::Scalar(25, 27, 31), -1);
    cv::rectangle(img, v0, cv::Scalar(80, 84, 92), 1);
    cv::rectangle(img, v1, cv::Scalar(80, 84, 92), 1);
    cv::putText(img, "view 0", cv::Point(v0.x + 10, v0.y + 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(235, 235, 235), 1);
    cv::putText(img, "view 1", cv::Point(v1.x + 10, v1.y + 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(235, 235, 235), 1);

    for (int i = 0; i < N_POINTS; i += 5) {
        cv::Scalar c = color_for_id(i);
        cv::Point p0 = img_point(features[0 * N_POINTS + i], v0);
        cv::Point p1 = img_point(features[1 * N_POINTS + i], v1);
        cv::circle(img, p0, 1, c, -1, cv::LINE_AA);
        cv::circle(img, p1, 1, c, -1, cv::LINE_AA);
    }
    for (int i = 0; i < N_POINTS; i += 37) {
        int j = matches[i];
        if (j < 0) continue;
        cv::Scalar c = (j == i) ? cv::Scalar(80, 210, 130) : cv::Scalar(90, 100, 255);
        cv::Point p0 = img_point(features[0 * N_POINTS + i], v0);
        cv::Point p1 = img_point(features[1 * N_POINTS + j], v1);
        cv::line(img, p0, p1, c * 0.65, 1, cv::LINE_AA);
        cv::circle(img, p0, 3, c, 1, cv::LINE_AA);
        cv::circle(img, p1, 3, c, 1, cv::LINE_AA);
    }

    cv::Rect cloud(28, 306, 580, 282);
    cv::rectangle(img, cloud, cv::Scalar(25, 27, 31), -1);
    cv::rectangle(img, cloud, cv::Scalar(80, 84, 92), 1);
    cv::putText(img, "x-z point cloud: initial / optimized / GT",
                cv::Point(cloud.x + 12, cloud.y + 24), cv::FONT_HERSHEY_SIMPLEX,
                0.48, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    for (int i = 0; i < N_POINTS; i += 2) {
        cv::circle(img, cloud_point(gt[i], cloud), 1, cv::Scalar(90, 90, 98), -1);
        cv::circle(img, cloud_point(init[i], cloud), 1, cv::Scalar(90, 130, 255), -1);
        cv::circle(img, cloud_point(opt[i], cloud), 1, cv::Scalar(90, 230, 135), -1);
    }

    cv::Rect stats(632, 86, 300, 154);
    cv::rectangle(img, stats, cv::Scalar(29, 31, 36), -1);
    cv::rectangle(img, stats, cv::Scalar(76, 80, 88), 1);
    cv::putText(img, cv::format("matches %d / %d", correct_matches, N_POINTS),
                cv::Point(stats.x + 14, stats.y + 34), cv::FONT_HERSHEY_SIMPLEX,
                0.52, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::putText(img, cv::format("reproj %.3f px", reproj_hist.empty() ? 0.0f : reproj_hist.back()),
                cv::Point(stats.x + 14, stats.y + 68), cv::FONT_HERSHEY_SIMPLEX,
                0.52, cv::Scalar(90, 170, 255), 1, cv::LINE_AA);
    cv::putText(img, cv::format("3D RMSE %.4f m", rmse_hist.empty() ? 0.0f : rmse_hist.back()),
                cv::Point(stats.x + 14, stats.y + 102), cv::FONT_HERSHEY_SIMPLEX,
                0.52, cv::Scalar(90, 230, 135), 1, cv::LINE_AA);
    cv::putText(img, cv::format("CPU/GPU %.1fx", bench.speedup),
                cv::Point(stats.x + 14, stats.y + 136), cv::FONT_HERSHEY_SIMPLEX,
                0.52, cv::Scalar(220, 224, 230), 1, cv::LINE_AA);

    draw_history(img, reproj_hist, rmse_hist, cv::Rect(632, 274, 300, 214));
    cv::putText(img, "descriptor match -> triangulate -> point BA",
                cv::Point(632, 532), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(185, 190, 198), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::array<Cam, N_VIEWS> cams = make_cameras();
    std::vector<Point3> gt_points = make_points();
    std::vector<Feature> features;
    make_features(gt_points, cams, features);

    Feature* d_features = nullptr;
    Cam* d_cams = nullptr;
    int* d_matches = nullptr;
    int* d_best = nullptr;
    int* d_second = nullptr;
    Point3* d_points = nullptr;
    CUDA_CHECK(cudaMalloc(&d_features, features.size() * sizeof(Feature)));
    CUDA_CHECK(cudaMalloc(&d_cams, N_VIEWS * sizeof(Cam)));
    CUDA_CHECK(cudaMalloc(&d_matches, N_POINTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best, N_POINTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_second, N_POINTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_points, N_POINTS * sizeof(Point3)));

    BenchResult bench = benchmark(features, cams, d_features, d_cams,
                                  d_matches, d_best, d_second, d_points);

    std::vector<int> matches(N_POINTS);
    std::vector<int> best_dist(N_POINTS);
    CUDA_CHECK(cudaMemcpy(matches.data(), d_matches, N_POINTS * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(best_dist.data(), d_best, N_POINTS * sizeof(int),
                          cudaMemcpyDeviceToHost));
    int correct = 0;
    int accepted = 0;
    for (int i = 0; i < N_POINTS; i++) {
        if (matches[i] >= 0) accepted++;
        if (matches[i] == i) correct++;
    }

    triangulate_kernel<<<(N_POINTS + 255) / 256, 256>>>(d_features, d_cams,
                                                       d_matches, d_points);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<Point3> init_points(N_POINTS);
    CUDA_CHECK(cudaMemcpy(init_points.data(), d_points, N_POINTS * sizeof(Point3),
                          cudaMemcpyDeviceToHost));
    std::vector<Point3> opt_points = init_points;

    std::printf("GPU SfM mini: %d features x %d views\n", N_POINTS, N_VIEWS);
    std::printf("GPU match %.3f ms, GPU triangulate+BA %.3f ms, CPU %.3f ms, speedup %.1fx\n",
                bench.gpu_match_ms, bench.gpu_ba_ms,
                bench.cpu_match_ms + bench.cpu_ba_ms, bench.speedup);
    std::printf("matches accepted %d, correct %d, initial 3D RMSE %.5f m\n",
                accepted, correct, rmse3d(init_points, gt_points));

    int mkdir_ret = std::system("mkdir -p gif");
    (void)mkdir_ret;
    cv::VideoWriter video("gif/gpu_sfm_mini.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_sfm_mini.avi\n");
        return 1;
    }

    std::vector<float> reproj_hist;
    std::vector<float> rmse_hist;
    reproj_hist.push_back(reproj_rmse(opt_points, features, cams));
    rmse_hist.push_back(rmse3d(opt_points, gt_points));
    for (int hold = 0; hold < 8; hold++) {
        video.write(draw_frame(features, matches, gt_points, init_points, opt_points,
                               reproj_hist, rmse_hist, bench, 0, correct));
    }
    for (int it = 0; it < BA_ITERS; it++) {
        ba_points_kernel<<<(N_POINTS + 255) / 256, 256>>>(d_features, d_cams,
                                                         d_points, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(opt_points.data(), d_points, N_POINTS * sizeof(Point3),
                              cudaMemcpyDeviceToHost));
        reproj_hist.push_back(reproj_rmse(opt_points, features, cams));
        rmse_hist.push_back(rmse3d(opt_points, gt_points));
        cv::Mat frame = draw_frame(features, matches, gt_points, init_points,
                                   opt_points, reproj_hist, rmse_hist,
                                   bench, it + 1, correct);
        for (int rep = 0; rep < 3; rep++) video.write(frame);
        if (it % 4 == 0 || it == BA_ITERS - 1) {
            std::printf("iter %02d  reproj %.4f px  3D RMSE %.5f m\n",
                        it + 1, reproj_hist.back(), rmse_hist.back());
        }
    }
    for (int hold = 0; hold < 12; hold++) {
        video.write(draw_frame(features, matches, gt_points, init_points, opt_points,
                               reproj_hist, rmse_hist, bench, BA_ITERS, correct));
    }
    video.release();

    std::printf("Final reprojection RMSE %.5f px, final 3D RMSE %.6f m\n",
                reproj_hist.back(), rmse_hist.back());
    cudabot::avi_to_gif("gif/gpu_sfm_mini.avi", "gif/gpu_sfm_mini.gif", 8, 640);
    std::printf("GIF saved to gif/gpu_sfm_mini.gif\n");

    CUDA_CHECK(cudaFree(d_features));
    CUDA_CHECK(cudaFree(d_cams));
    CUDA_CHECK(cudaFree(d_matches));
    CUDA_CHECK(cudaFree(d_best));
    CUDA_CHECK(cudaFree(d_second));
    CUDA_CHECK(cudaFree(d_points));
    return 0;
}
