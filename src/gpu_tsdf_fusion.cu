// gpu_tsdf_fusion.cu
//
// GPU volumetric TSDF (Truncated Signed Distance Field) fusion — the
// KinectFusion-style depth-integration step that the repo's mapping section
// (occupancy grid, CSM/NDT submaps) did not yet cover.
//
// Many depth frames from known poses are fused into a single dense voxel
// volume.  For every voxel we project its centre into each camera, read the
// measured depth at that pixel, form the projective signed distance
//   sdf = depth_measured - z_voxel_in_camera,
// truncate it to +/- mu, and fold it into a running weighted average.  This is
// the classic "one thread = one voxel" map: each voxel integrates the whole
// frame stream independently of every other voxel.
//
// Correctness note (in contrast to the iLQR demo): TSDF fusion is a
// *deterministic weighted average*.  Each voxel processes the frames in the
// same fixed order on the CPU and the GPU, so there are no data-dependent
// branches that can fork into a different answer.  The two paths agree to
// floating-point round-off (we report the max abs TSDF difference, ~1e-6 with
// --fmad=false), and the win is pure throughput on the volume.
//
// Demo: a 3-sphere "snowman" on a ground plane, observed by 24 orbiting depth
// cameras (depth synthesised by analytic sphere-tracing).  We fuse on the CPU
// (serial triple loop) and on the GPU (one thread per voxel), compare the two
// volumes, then animate the zero-level-set surface refining as frames are
// added and the view orbits.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ------------------------------------------------------------------ constants
#define VOX_RES   96            // voxels per axis (96^3 ~ 884 k voxels)
#define DW        160           // depth image width
#define DH        120           // depth image height
static const int   N_FRAMES = 24;
static const int   N_VOX    = VOX_RES * VOX_RES * VOX_RES;

// volume axis-aligned bounds (metres)
static const float GMIN_X = -2.5f, GMIN_Y = -2.5f, GMIN_Z = -1.5f;
static const float GSPAN  = 5.0f;                       // cubic span per axis
static const float VOXSZ  = GSPAN / VOX_RES;            // voxel edge length
static const float MU     = 0.20f;                      // truncation distance

// pinhole intrinsics
static const float FX = 120.0f, FY = 120.0f;
static const float CX = DW * 0.5f, CY = DH * 0.5f;

static const int   PANEL_W = 760;
static const int   PANEL_H = 600;

// ----------------------------------------------------------------- scene SDF
// Analytic ground-truth surface used only to synthesise the depth frames.
__host__ static inline float scene_sdf(float x, float y, float z) {
    auto sph = [](float x, float y, float z, float cx, float cy, float cz, float r) {
        float dx = x - cx, dy = y - cy, dz = z - cz;
        return std::sqrt(dx * dx + dy * dy + dz * dz) - r;
    };
    float d = z - (-1.3f);                              // ground plane
    d = std::min(d, sph(x, y, z, 0.0f, 0.0f, -0.1f, 1.20f));   // body
    d = std::min(d, sph(x, y, z, 0.0f, 0.0f,  1.45f, 0.80f));  // torso
    d = std::min(d, sph(x, y, z, 0.0f, 0.0f,  2.55f, 0.52f));  // head
    return d;
}

// --------------------------------------------------------------- camera model
// Per-frame extrinsics packed as: pos[3], then rows x_cam,y_cam,z_cam (3x3).
struct Frame { float pos[3]; float rot[9]; };

static void make_frame(int f, Frame& fr) {
    float a   = 2.0f * static_cast<float>(M_PI) * f / N_FRAMES;
    float rad = 5.5f;
    float h   = 2.2f + 0.9f * std::sin(a * 1.0f);       // gently bob in height
    float cx = rad * std::cos(a), cy = rad * std::sin(a), cz = h;
    float tx = 0.0f, ty = 0.0f, tz = 0.6f;              // look-at target
    // forward = z_cam
    float fwd[3] = {tx - cx, ty - cy, tz - cz};
    float fn = std::sqrt(fwd[0]*fwd[0] + fwd[1]*fwd[1] + fwd[2]*fwd[2]);
    for (float& v : fwd) v /= fn;
    // right = forward x world_up
    float up[3] = {0.0f, 0.0f, 1.0f};
    float right[3] = {fwd[1]*up[2] - fwd[2]*up[1],
                      fwd[2]*up[0] - fwd[0]*up[2],
                      fwd[0]*up[1] - fwd[1]*up[0]};
    float rn = std::sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    for (float& v : right) v /= rn;
    // down = forward x right  (image v grows downward)
    float down[3] = {fwd[1]*right[2] - fwd[2]*right[1],
                     fwd[2]*right[0] - fwd[0]*right[2],
                     fwd[0]*right[1] - fwd[1]*right[0]};
    fr.pos[0] = cx; fr.pos[1] = cy; fr.pos[2] = cz;
    fr.rot[0] = right[0]; fr.rot[1] = right[1]; fr.rot[2] = right[2];
    fr.rot[3] = down[0];  fr.rot[4] = down[1];  fr.rot[5] = down[2];
    fr.rot[6] = fwd[0];   fr.rot[7] = fwd[1];   fr.rot[8] = fwd[2];
}

// Synthesise one depth image by sphere-tracing the analytic scene.
// depth stored as the camera-frame z of the hit; 0 = background (no hit).
static void synth_depth(const Frame& fr, std::vector<float>& depth) {
    depth.assign(DW * DH, 0.0f);
    for (int v = 0; v < DH; ++v) {
        for (int u = 0; u < DW; ++u) {
            float dc[3] = {(u + 0.5f - CX) / FX, (v + 0.5f - CY) / FY, 1.0f};
            float dn = std::sqrt(dc[0]*dc[0] + dc[1]*dc[1] + dc[2]*dc[2]);
            for (float& c : dc) c /= dn;
            // world ray direction = rot^T * dc (rows of rot are the cam axes)
            float wd[3] = {
                fr.rot[0]*dc[0] + fr.rot[3]*dc[1] + fr.rot[6]*dc[2],
                fr.rot[1]*dc[0] + fr.rot[4]*dc[1] + fr.rot[7]*dc[2],
                fr.rot[2]*dc[0] + fr.rot[5]*dc[1] + fr.rot[8]*dc[2]};
            float t = 0.0f;
            bool hit = false;
            for (int s = 0; s < 128; ++s) {
                float px = fr.pos[0] + t*wd[0];
                float py = fr.pos[1] + t*wd[1];
                float pz = fr.pos[2] + t*wd[2];
                float d  = scene_sdf(px, py, pz);
                if (d < 1e-3f) { hit = true; break; }
                t += d;
                if (t > 18.0f) break;
            }
            if (hit) depth[v * DW + u] = t * dc[2];      // camera-frame z
        }
    }
}

// ---------------------------------------------------------- fuse one voxel
// Shared by the CPU loop and the CUDA kernel: integrate every frame into the
// running (tsdf, weight) for the voxel whose centre is (wx, wy, wz).
__host__ __device__ static inline void fuse_voxel(
        float wx, float wy, float wz,
        const float* frames,        // N_FRAMES * 12 (pos[3], rot[9])
        const float* depth,         // N_FRAMES * DW * DH
        float* tsdf_out, float* w_out) {
    float tsdf = 0.0f, weight = 0.0f;
    for (int f = 0; f < N_FRAMES; ++f) {
        const float* P = frames + f * 12;
        float rx = wx - P[0], ry = wy - P[1], rz = wz - P[2];
        // camera coords pc = rot * rel  (rot rows = right, down, forward at P[3..11])
        float pcz = P[9]*rx + P[10]*ry + P[11]*rz;
        if (pcz <= 0.1f) continue;
        float pcx = P[3]*rx + P[4]*ry + P[5]*rz;
        float pcy = P[6]*rx + P[7]*ry + P[8]*rz;
        float u = FX * pcx / pcz + CX;
        float v = FY * pcy / pcz + CY;
        int iu = (int)(u), iv = (int)(v);
        if (iu < 0 || iu >= DW || iv < 0 || iv >= DH) continue;
        float dmeas = depth[f * DW * DH + iv * DW + iu];
        if (dmeas <= 0.0f) continue;                 // background pixel
        float sdf = dmeas - pcz;
        if (sdf < -MU) continue;                     // behind surface: occluded
        if (sdf > MU) sdf = MU;                       // truncate the free-space side
        float wnew = weight + 1.0f;
        tsdf = (tsdf * weight + sdf) / wnew;
        weight = wnew;
    }
    *tsdf_out = tsdf;
    *w_out = weight;
}

__global__ void fuse_kernel(const float* frames, const float* depth,
                            float* tsdf, float* weight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_VOX) return;
    int i = idx % VOX_RES;
    int j = (idx / VOX_RES) % VOX_RES;
    int k = idx / (VOX_RES * VOX_RES);
    float wx = GMIN_X + (i + 0.5f) * VOXSZ;
    float wy = GMIN_Y + (j + 0.5f) * VOXSZ;
    float wz = GMIN_Z + (k + 0.5f) * VOXSZ;
    fuse_voxel(wx, wy, wz, frames, depth, &tsdf[idx], &weight[idx]);
}

// CPU reference: identical math, serial over all voxels.
static void fuse_cpu(const float* frames, const float* depth,
                     float* tsdf, float* weight) {
    for (int idx = 0; idx < N_VOX; ++idx) {
        int i = idx % VOX_RES;
        int j = (idx / VOX_RES) % VOX_RES;
        int k = idx / (VOX_RES * VOX_RES);
        float wx = GMIN_X + (i + 0.5f) * VOXSZ;
        float wy = GMIN_Y + (j + 0.5f) * VOXSZ;
        float wz = GMIN_Z + (k + 0.5f) * VOXSZ;
        fuse_voxel(wx, wy, wz, frames, depth, &tsdf[idx], &weight[idx]);
    }
}

// ----------------------------------------------------------- visualisation
struct Cam { float yaw, pitch, dist; };
static cv::Point2i project(float x, float y, float z, const Cam& c, int W, int H) {
    float cy = std::cos(c.yaw), sy = std::sin(c.yaw);
    float cp = std::cos(c.pitch), sp = std::sin(c.pitch);
    float x1 =  cy * x + sy * y;
    float y1 = -sy * x + cy * y;
    float z1 =  z - 0.6f;
    float y2 =  cp * y1 - sp * z1;
    float z2 =  sp * y1 + cp * z1;
    float xc = x1, yc = z2;
    float zc = c.dist - y2;
    if (zc < 0.1f) zc = 0.1f;
    float f = 1.0f * H;
    return cv::Point2i(static_cast<int>(W * 0.5f + f * xc / zc),
                       static_cast<int>(H * 0.55f - f * yc / zc));
}

// Extract zero-crossing surface voxel centres from a TSDF volume.
struct SurfPt { float x, y, z; };
static void extract_surface(const std::vector<float>& tsdf,
                            const std::vector<float>& weight,
                            std::vector<SurfPt>& out) {
    out.clear();
    const float band = 0.7f * VOXSZ;
    for (int k = 0; k < VOX_RES; ++k)
        for (int j = 0; j < VOX_RES; ++j)
            for (int i = 0; i < VOX_RES; ++i) {
                int idx = (k * VOX_RES + j) * VOX_RES + i;
                if (weight[idx] <= 0.0f) continue;
                if (std::fabs(tsdf[idx]) > band) continue;
                out.push_back({GMIN_X + (i + 0.5f) * VOXSZ,
                               GMIN_Y + (j + 0.5f) * VOXSZ,
                               GMIN_Z + (k + 0.5f) * VOXSZ});
            }
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::printf("GPU TSDF fusion: %d^3 = %d voxels, %d depth frames (%dx%d)\n",
                VOX_RES, N_VOX, N_FRAMES, DW, DH);

    // --- synthesise depth frames -------------------------------------------
    std::vector<Frame> frames(N_FRAMES);
    std::vector<float> depth(N_FRAMES * DW * DH);
    for (int f = 0; f < N_FRAMES; ++f) {
        make_frame(f, frames[f]);
        std::vector<float> d;
        synth_depth(frames[f], d);
        std::copy(d.begin(), d.end(), depth.begin() + f * DW * DH);
    }
    // flat frame buffer (pos[3] + rot[9] = 12 floats each)
    std::vector<float> fflat(N_FRAMES * 12);
    for (int f = 0; f < N_FRAMES; ++f) {
        std::copy(frames[f].pos, frames[f].pos + 3, fflat.begin() + f * 12);
        std::copy(frames[f].rot, frames[f].rot + 9, fflat.begin() + f * 12 + 3);
    }

    // --- CPU fuse (timed) ---------------------------------------------------
    std::vector<float> tsdf_cpu(N_VOX), w_cpu(N_VOX);
    auto t0 = std::chrono::high_resolution_clock::now();
    fuse_cpu(fflat.data(), depth.data(), tsdf_cpu.data(), w_cpu.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- GPU fuse (timed) ---------------------------------------------------
    float *d_frames, *d_depth, *d_tsdf, *d_w;
    CUDA_CHECK(cudaMalloc(&d_frames, fflat.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_depth,  depth.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tsdf,   N_VOX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w,      N_VOX * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_frames, fflat.data(), fflat.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depth, depth.data(), depth.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    int block = 128, grid = (N_VOX + block - 1) / block;
    fuse_kernel<<<grid, block>>>(d_frames, d_depth, d_tsdf, d_w);  // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    fuse_kernel<<<grid, block>>>(d_frames, d_depth, d_tsdf, d_w);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));

    std::vector<float> tsdf_gpu(N_VOX), w_gpu(N_VOX);
    CUDA_CHECK(cudaMemcpy(tsdf_gpu.data(), d_tsdf, N_VOX * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(w_gpu.data(), d_w, N_VOX * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // --- compare CPU vs GPU -------------------------------------------------
    double max_diff = 0.0, sum_diff = 0.0;
    int n_observed = 0, n_surf = 0;
    for (int i = 0; i < N_VOX; ++i) {
        if (w_cpu[i] > 0.0f) ++n_observed;
        if (w_cpu[i] > 0.0f && std::fabs(tsdf_cpu[i]) < 0.7f * VOXSZ) ++n_surf;
        double d = std::fabs((double)tsdf_cpu[i] - (double)tsdf_gpu[i]);
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    double speedup = cpu_ms / gpu_ms;
    std::printf("CPU fuse %.1f ms, GPU fuse %.3f ms  -> %.0fx\n",
                cpu_ms, gpu_ms, speedup);
    std::printf("observed voxels %d, surface band %d\n", n_observed, n_surf);
    std::printf("TSDF max|diff| %.3e, mean|diff| %.3e\n",
                max_diff, sum_diff / N_VOX);

    // --- animation: surface refining as frames are added, view orbiting -----
    if (system("mkdir -p tmp") != 0)
        std::fprintf(stderr, "warning: mkdir tmp failed\n");
    cv::VideoWriter video("tmp/gpu_tsdf_fusion.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          12, cv::Size(PANEL_W, PANEL_H));

    std::vector<float> tsdf_a(N_VOX), w_a(N_VOX);
    for (int step = 1; step <= N_FRAMES; ++step) {
        // fuse the first `step` frames on the host for this snapshot
        std::vector<float> sub(fflat.begin(), fflat.begin() + step * 12);
        std::vector<float> subd(depth.begin(), depth.begin() + step * DW * DH);
        // re-run fuse over the sub-stream by temporarily shrinking the count
        // (fuse_voxel loops N_FRAMES, so build a padded buffer of `step` frames)
        for (int idx = 0; idx < N_VOX; ++idx) {
            int i = idx % VOX_RES;
            int j = (idx / VOX_RES) % VOX_RES;
            int k = idx / (VOX_RES * VOX_RES);
            float wx = GMIN_X + (i + 0.5f) * VOXSZ;
            float wy = GMIN_Y + (j + 0.5f) * VOXSZ;
            float wz = GMIN_Z + (k + 0.5f) * VOXSZ;
            // inline partial fuse
            float tsdf = 0.0f, weight = 0.0f;
            for (int f = 0; f < step; ++f) {
                const float* P = sub.data() + f * 12;
                float rx = wx - P[0], ry = wy - P[1], rz = wz - P[2];
                float pcz = P[9]*rx + P[10]*ry + P[11]*rz;
                if (pcz <= 0.1f) continue;
                float pcx = P[3]*rx + P[4]*ry + P[5]*rz;
                float pcy = P[6]*rx + P[7]*ry + P[8]*rz;
                float u = FX * pcx / pcz + CX, v = FY * pcy / pcz + CY;
                int iu = (int)u, iv = (int)v;
                if (iu < 0 || iu >= DW || iv < 0 || iv >= DH) continue;
                float dmeas = subd[f * DW * DH + iv * DW + iu];
                if (dmeas <= 0.0f) continue;
                float sdf = dmeas - pcz;
                if (sdf < -MU) continue;
                if (sdf > MU) sdf = MU;
                float wn = weight + 1.0f;
                tsdf = (tsdf * weight + sdf) / wn;
                weight = wn;
            }
            tsdf_a[idx] = tsdf; w_a[idx] = weight;
        }

        std::vector<SurfPt> surf;
        extract_surface(tsdf_a, w_a, surf);

        Cam cam{0.6f + 0.045f * step, 0.45f, 13.0f};
        // painter's order: far points first
        std::sort(surf.begin(), surf.end(), [&](const SurfPt& a, const SurfPt& b) {
            return (a.x * std::sin(cam.yaw)) < (b.x * std::sin(cam.yaw));
        });

        cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 18, 22));
        for (const SurfPt& p : surf) {
            cv::Point2i q = project(p.x, p.y, p.z, cam, PANEL_W, PANEL_H);
            if (q.x < 0 || q.x >= PANEL_W || q.y < 0 || q.y >= PANEL_H - 70) continue;
            float t = std::min(1.0f, std::max(0.0f, (p.z + 1.5f) / 4.5f));
            cv::Scalar col(255.0f * (1.0f - t), 120.0f + 100.0f * t, 80.0f + 175.0f * t);
            cv::circle(img, q, 1, col, -1);
        }

        cv::putText(img, "GPU TSDF volumetric fusion (one thread = one voxel)",
                    cv::Point(12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
        char l1[128], l2[128], l3[128];
        std::snprintf(l1, sizeof(l1), "frames fused: %d / %d   surface voxels: %d",
                      step, N_FRAMES, (int)surf.size());
        std::snprintf(l2, sizeof(l2),
                      "fuse %d^3 = %d voxels:  CPU %.0f ms  vs  GPU %.2f ms  (%.0fx)",
                      VOX_RES, N_VOX, cpu_ms, gpu_ms, speedup);
        std::snprintf(l3, sizeof(l3),
                      "CPU/GPU TSDF max|diff| %.1e  (deterministic fuse, --fmad=false)",
                      max_diff);
        cv::putText(img, l1, cv::Point(12, PANEL_H - 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 220, 255), 1, cv::LINE_AA);
        cv::putText(img, l2, cv::Point(12, PANEL_H - 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 255, 200), 1, cv::LINE_AA);
        cv::putText(img, l3, cv::Point(12, PANEL_H - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        video.write(img);
    }
    // hold the final frame a bit
    video.release();

    cudabot::avi_to_gif("tmp/gpu_tsdf_fusion.avi", "gif/gpu_tsdf_fusion.gif", 12, 760);
    std::printf("wrote gif/gpu_tsdf_fusion.gif\n");

    CUDA_CHECK(cudaFree(d_frames));
    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_tsdf));
    CUDA_CHECK(cudaFree(d_w));
    return 0;
}
