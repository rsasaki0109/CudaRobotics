// gpu_gaussian_splatting_slam.cu
//
// GPU RGB-D Gaussian-Splatting SLAM (SplaTAM-style), self-contained demo.
//
// The repo already had a forward Gaussian-splatting *renderer*
// (gpu_gaussian_splatting.cu).  This turns that representation into an online
// SLAM system: the camera flies through an unknown scene, and frame by frame
// the system both (a) tracks its own pose and (b) grows a 3D Gaussian map.
//
// Pipeline per frame (everything heavy runs on the GPU):
//   1. SENSOR    The ground-truth world (an analytic room with planes and
//                spheres) is ray-cast from the *true* camera pose into an
//                RGB-D frame -- this is the only thing the SLAM system is
//                allowed to see.
//   2. TRACKING  The observed depth is back-projected to a point cloud and
//                aligned to the current Gaussian map with frame-to-model ICP.
//                One GPU thread = one observed point: it brute-force finds its
//                nearest map point and accumulates a 6-DoF point-to-plane
//                normal equation; a block reduction sums them and the 6x6
//                system is solved on the host.  ~14 iterations / frame.
//   3. MAPPING   Observed points are back-projected with the *estimated* pose
//                and fused into the global map through a voxel hash: only cells
//                that are still empty spawn a new Gaussian, so the map grows
//                without unbounded duplication.
//   4. RENDER    The growing map is splatted from the estimated pose (the
//                "what the SLAM thinks it sees" view) and from a slow orbit
//                camera (the global map), side by side with the sensor frame.
//
// Honesty: frame 0 is anchored to the true pose (SLAM trajectories are only
// defined up to a global gauge), and the reported ATE is the RMSE of the
// estimated camera positions against ground truth -- measured, not assumed.
//
// One demo = one .cu; reuses include/cuda_check.cuh and include/cuda_video.h.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ---- image / camera -------------------------------------------------------
constexpr int IMG_W = 384;
constexpr int IMG_H = 288;
constexpr float FOCAL = 240.0f;     // wide-ish FOV so several surfaces stay in view
constexpr float CXp = IMG_W * 0.5f;
constexpr float CYp = IMG_H * 0.5f;
constexpr float Z_NEAR = 0.3f;

constexpr int N_FRAMES = 120;

// ---- map / fusion ---------------------------------------------------------
constexpr int   MAX_GAUSS = 24000;
constexpr float VOXEL = 0.10f;       // fusion voxel size (m)
constexpr int   OBS_STRIDE = 4;      // sub-sample observed pixels for fusion
constexpr int   ICP_STRIDE = 6;      // sub-sample observed pixels for tracking
constexpr int   ICP_ITERS = 14;
constexpr float ICP_MAX_CORR = 0.45f; // reject correspondences farther than this

// ---------------------------------------------------------------------------
struct Gaussian {
    float mx, my, mz;
    float s;
    float r, g, b;
    float a;
};

struct Projected {
    float ux, uy;
    float radius;
    float inv_var;
    float r, g, b, a;
    float depth;
    int   valid;
};

// =========================================================================
// CUDA kernels: projection + render (RGB + depth)
// =========================================================================
__device__ inline void apply_view(const float* __restrict__ V, float x, float y, float z,
                                   float& vx, float& vy, float& vz) {
    vx = V[0] * x + V[1] * y + V[2]  * z + V[3];
    vy = V[4] * x + V[5] * y + V[6]  * z + V[7];
    vz = V[8] * x + V[9] * y + V[10] * z + V[11];
}

__global__ void project_kernel(int n, const Gaussian* __restrict__ gs,
                               const float* __restrict__ V, Projected* __restrict__ out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    Gaussian g = gs[k];
    float vx, vy, vz;
    apply_view(V, g.mx, g.my, g.mz, vx, vy, vz);
    Projected p;
    p.valid = 0;
    p.depth = vz;
    if (vz < Z_NEAR) { out[k] = p; return; }
    p.ux = CXp + FOCAL * (vx / vz);
    p.uy = CYp - FOCAL * (vy / vz);
    float r_px = fmaxf(0.6f, FOCAL * g.s / vz);
    p.radius = r_px;
    p.inv_var = 1.0f / (r_px * r_px);
    p.r = g.r; p.g = g.g; p.b = g.b; p.a = g.a;
    if (p.ux + 3 * r_px < 0 || p.ux - 3 * r_px >= IMG_W ||
        p.uy + 3 * r_px < 0 || p.uy - 3 * r_px >= IMG_H) { out[k] = p; return; }
    p.valid = 1;
    out[k] = p;
}

// Per-pixel alpha-composite -> RGB + expected depth.
__global__ void render_kernel(int n_valid, const Projected* __restrict__ ps,
                              const int* __restrict__ order,
                              unsigned char* __restrict__ img, float* __restrict__ depth) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= IMG_W || py >= IMG_H) return;
    float cr = 0.06f, cg = 0.07f, cb = 0.10f;
    float T = 1.0f, dacc = 0.0f, wacc = 0.0f;
    float fx = float(px), fy = float(py);
    for (int i = 0; i < n_valid; i++) {
        if (T < 5.0e-3f) break;
        const Projected p = ps[order[i]];
        float dx = fx - p.ux, dy = fy - p.uy;
        float r2 = dx * dx + dy * dy;
        if (r2 > 9.0f * p.radius * p.radius) continue;
        float w = expf(-0.5f * r2 * p.inv_var) * p.a;
        cr += T * w * p.r; cg += T * w * p.g; cb += T * w * p.b;
        dacc += T * w * p.depth; wacc += T * w;
        T *= (1.0f - w);
    }
    int idx = (py * IMG_W + px) * 3;
    img[idx + 0] = (unsigned char)fminf(fmaxf(cb * 255.0f, 0.0f), 255.0f);
    img[idx + 1] = (unsigned char)fminf(fmaxf(cg * 255.0f, 0.0f), 255.0f);
    img[idx + 2] = (unsigned char)fminf(fmaxf(cr * 255.0f, 0.0f), 255.0f);
    if (depth) depth[py * IMG_W + px] = (wacc > 0.4f && T < 0.6f) ? dacc / wacc : 0.0f;
}

// =========================================================================
// Analytic RGB-D sensor: ray-cast a room (planes + spheres) -> sharp depth.
// This is the "real depth sensor"; the SLAM map it builds is Gaussians.
// =========================================================================
struct Sphere { float cx, cy, cz, rad, r, g, b; };
__constant__ Sphere c_spheres[5];
__constant__ int c_nsph;

__device__ inline float shade(float ndl) { return 0.35f + 0.65f * fmaxf(0.0f, ndl); }

// camera->world R (columns right/up/fwd), origin O. Writes BGR + forward depth.
__global__ void raycast_kernel(float ox, float oy, float oz, const float* __restrict__ R,
                               unsigned char* __restrict__ img, float* __restrict__ depth) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= IMG_W || py >= IMG_H) return;

    float dcx = (px - CXp) / FOCAL, dcy = -(py - CYp) / FOCAL, dcz = 1.0f;
    float il = rsqrtf(dcx*dcx + dcy*dcy + dcz*dcz);
    dcx *= il; dcy *= il; dcz *= il;                  // normalized cam-frame dir
    float dx = R[0]*dcx + R[1]*dcy + R[2]*dcz;
    float dy = R[3]*dcx + R[4]*dcy + R[5]*dcz;
    float dz = R[6]*dcx + R[7]*dcy + R[8]*dcz;        // world-frame dir

    const float BND = 3.2f, H = 3.0f;
    float best_t = 1e9f;
    float cr = 0.06f, cg = 0.07f, cb = 0.10f;         // background

    // light from above-front
    float lx = 0.3f, ly = 0.9f, lz = 0.25f;
    float ll = rsqrtf(lx*lx+ly*ly+lz*lz); lx*=ll; ly*=ll; lz*=ll;

    auto plane = [&](float nx, float ny, float nz, float d,   // n.x = d
                     float u0, float u1, float v0, float v1, int axis,
                     float br, float bg, float bb) {
        float den = nx*dx + ny*dy + nz*dz;
        if (fabsf(den) < 1e-6f) return;
        float t = (d - (nx*ox + ny*oy + nz*oz)) / den;
        if (t <= 0.05f || t >= best_t) return;
        float hx = ox + t*dx, hy = oy + t*dy, hz = oz + t*dz;
        float a, b;
        if (axis == 0) { a = hy; b = hz; } else if (axis == 1) { a = hx; b = hz; } else { a = hx; b = hy; }
        if (a < u0 || a > u1 || b < v0 || b > v1) return;
        // checker texture
        int chk = ((int)floorf(a*1.5f) + (int)floorf(b*1.5f)) & 1;
        float tint = chk ? 1.0f : 0.78f;
        float ndl = fabsf(nx*lx + ny*ly + nz*lz);
        float sh = shade(ndl);
        best_t = t; cr = br*tint*sh; cg = bg*tint*sh; cb = bb*tint*sh;
    };
    // floor & 4 walls
    plane(0,1,0, 0.0f, -BND,BND, -BND,BND, 1, 0.55f,0.55f,0.6f);  // floor (bounds on x,z)
    plane(0,0,1, -BND, -BND,BND, 0,H, 2, 0.50f,0.62f,0.74f);  // back wall z=-BND -> n.z=... use z plane
    plane(0,0,1,  BND, -BND,BND, 0,H, 2, 0.78f,0.56f,0.44f);  // front z=BND
    plane(1,0,0, -BND, 0,H, -BND,BND, 0, 0.46f,0.6f,0.7f);    // left x=-BND
    plane(1,0,0,  BND, 0,H, -BND,BND, 0, 0.72f,0.55f,0.46f);  // right x=BND

    // spheres
    for (int s = 0; s < c_nsph; s++) {
        Sphere sp = c_spheres[s];
        float ocx = ox - sp.cx, ocy = oy - sp.cy, ocz = oz - sp.cz;
        float bb_ = ocx*dx + ocy*dy + ocz*dz;
        float cc = ocx*ocx + ocy*ocy + ocz*ocz - sp.rad*sp.rad;
        float disc = bb_*bb_ - cc;
        if (disc < 0) continue;
        float t = -bb_ - sqrtf(disc);
        if (t <= 0.05f || t >= best_t) continue;
        float hx = ox+t*dx, hy = oy+t*dy, hz = oz+t*dz;
        float nx = (hx-sp.cx)/sp.rad, ny = (hy-sp.cy)/sp.rad, nz = (hz-sp.cz)/sp.rad;
        float ndl = nx*lx + ny*ly + nz*lz;
        float sh = shade(ndl);
        best_t = t; cr = sp.r*sh; cg = sp.g*sh; cb = sp.b*sh;
    }

    int idx = (py*IMG_W + px) * 3;
    img[idx+0] = (unsigned char)fminf(fmaxf(cb*255.0f,0.0f),255.0f);
    img[idx+1] = (unsigned char)fminf(fmaxf(cg*255.0f,0.0f),255.0f);
    img[idx+2] = (unsigned char)fminf(fmaxf(cr*255.0f,0.0f),255.0f);
    depth[py*IMG_W + px] = (best_t < 1e8f) ? best_t * dcz : 0.0f;   // forward depth
}

// =========================================================================
// Frame-to-model ICP (point-to-plane)
// =========================================================================
// CAS-based double atomicAdd (works on compute capability < 6.0).
__device__ inline double atomicAddD(double* addr, double val) {
    unsigned long long* a = (unsigned long long*)addr;
    unsigned long long old = *a, assumed;
    do { assumed = old;
         old = atomicCAS(a, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Point-to-plane ICP.  out[29]: H upper-tri (21) + g (6) + sq-error (1) + count (1).
__global__ void icp_kernel(const float* __restrict__ src, int n_src,
                           const float* __restrict__ map, const float* __restrict__ nrm,
                           int n_map, double* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double c[29];
#pragma unroll
    for (int k = 0; k < 29; k++) c[k] = 0.0;

    if (i < n_src) {
        float px = src[3 * i], py = src[3 * i + 1], pz = src[3 * i + 2];
        float best = ICP_MAX_CORR * ICP_MAX_CORR;
        int bj = -1;
        for (int j = 0; j < n_map; j++) {
            float dx = px - map[3 * j], dy = py - map[3 * j + 1], dz = pz - map[3 * j + 2];
            float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best) { best = d2; bj = j; }
        }
        if (bj >= 0) {
            float qx = map[3*bj], qy = map[3*bj+1], qz = map[3*bj+2];
            float nx = nrm[3*bj], ny = nrm[3*bj+1], nz = nrm[3*bj+2];
            // point-to-plane residual r = n . (p - q)
            float rr = nx*(px-qx) + ny*(py-qy) + nz*(pz-qz);
            // J = [ (p x n) | n ]  (1x6), unknown = (w, t)
            float cxp = py*nz - pz*ny, cyp = pz*nx - px*nz, czp = px*ny - py*nx;
            float J[6] = {cxp, cyp, czp, nx, ny, nz};
            int idx = 0;
            for (int a = 0; a < 6; a++)
                for (int b = a; b < 6; b++) c[idx++] = (double)(J[a] * J[b]);
            for (int a = 0; a < 6; a++) c[21 + a] = (double)(J[a] * rr);
            c[27] = (double)(rr * rr);
            c[28] = 1.0;
        }
    }

    // block reduction: reuse one shared buffer for each of the 29 accumulators.
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    for (int k = 0; k < 29; k++) {
        sh[tid] = c[k];
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (tid < s) sh[tid] += sh[tid + s];
            __syncthreads();
        }
        if (tid == 0) atomicAddD(&out[k], sh[0]);
        __syncthreads();
    }
}

// =========================================================================
// Host math
// =========================================================================
struct Pose { float R[9]; float t[3]; };   // world point = R * cam + t

static Pose pose_identity() { return {{1,0,0, 0,1,0, 0,0,1}, {0,0,0}}; }

// world->view matrix (3x4) = [R^T | -R^T t]
static void view_from_pose(const Pose& P, float* V) {
    float Rt[9] = {P.R[0],P.R[3],P.R[6], P.R[1],P.R[4],P.R[7], P.R[2],P.R[5],P.R[8]};
    for (int i = 0; i < 9; i++) V[(i/3)*4 + (i%3)] = Rt[i];
    V[3]  = -(Rt[0]*P.t[0] + Rt[1]*P.t[1] + Rt[2]*P.t[2]);
    V[7]  = -(Rt[3]*P.t[0] + Rt[4]*P.t[1] + Rt[5]*P.t[2]);
    V[11] = -(Rt[6]*P.t[0] + Rt[7]*P.t[1] + Rt[8]*P.t[2]);
}

static void mat3_mul(const float* A, const float* B, float* C) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            C[r*3+c] = A[r*3]*B[c] + A[r*3+1]*B[3+c] + A[r*3+2]*B[6+c];
}

// Rodrigues: exp([w]_x)
static void rodrigues(const float* w, float* R) {
    float th = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    if (th < 1e-9f) { R[0]=1;R[1]=0;R[2]=0;R[3]=0;R[4]=1;R[5]=0;R[6]=0;R[7]=0;R[8]=1; return; }
    float kx = w[0]/th, ky = w[1]/th, kz = w[2]/th;
    float c = std::cos(th), s = std::sin(th), v = 1 - c;
    R[0]=c+kx*kx*v;    R[1]=kx*ky*v-kz*s; R[2]=kx*kz*v+ky*s;
    R[3]=ky*kx*v+kz*s; R[4]=c+ky*ky*v;    R[5]=ky*kz*v-kx*s;
    R[6]=kz*kx*v-ky*s; R[7]=kz*ky*v+kx*s; R[8]=c+kz*kz*v;
}

// Solve 6x6 symmetric system H x = g (Gaussian elimination). Returns false if singular.
static bool solve6(double H[6][6], double g[6], double x[6]) {
    double A[6][7];
    for (int i = 0; i < 6; i++) { for (int j = 0; j < 6; j++) A[i][j] = H[i][j]; A[i][6] = g[i]; }
    for (int c = 0; c < 6; c++) {
        int piv = c;
        for (int r = c + 1; r < 6; r++) if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
        if (std::fabs(A[piv][c]) < 1e-12) return false;
        for (int k = 0; k < 7; k++) std::swap(A[c][k], A[piv][k]);
        for (int r = 0; r < 6; r++) {
            if (r == c) continue;
            double f = A[r][c] / A[c][c];
            for (int k = c; k < 7; k++) A[r][k] -= f * A[c][k];
        }
    }
    for (int i = 0; i < 6; i++) x[i] = A[i][6] / A[i][i];
    return true;
}

// =========================================================================
// main
// =========================================================================
}  // namespace cudabot
using namespace cudabot;

int main() {
    // ground-truth world = analytic room (planes + spheres) the sensor ray-casts
    Sphere spheres[5] = {
        {-1.4f, 0.45f, 0.7f, 0.45f, 0.90f, 0.32f, 0.30f},
        { 1.5f, 0.45f, 1.2f, 0.45f, 0.32f, 0.86f, 0.40f},
        { 0.1f, 0.55f, -1.4f, 0.55f, 0.34f, 0.50f, 0.95f},
        { 1.7f, 0.40f, -1.6f, 0.40f, 0.93f, 0.82f, 0.30f},
    };
    int nsph = 4;
    CUDA_CHECK(cudaMemcpyToSymbol(c_spheres, spheres, nsph * sizeof(Sphere)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_nsph, &nsph, sizeof(int)));
    std::printf("World: analytic room (5 planes + %d spheres)  %dx%d RGB-D sensor\n", nsph, IMG_W, IMG_H);

    // ---- device buffers --------------------------------------------------
    Gaussian* d_map = nullptr;          // estimated map (grows)
    Projected* d_ps = nullptr;
    int* d_order = nullptr;
    float* d_V = nullptr;
    unsigned char* d_img = nullptr;
    float* d_depth = nullptr;
    float* d_src = nullptr;             // ICP source points (world frame)
    float* d_mappts = nullptr;          // ICP target = map means
    float* d_mapnrm = nullptr;          // ICP target normals
    double* d_acc = nullptr;
    CUDA_CHECK(cudaMalloc(&d_map, MAX_GAUSS * sizeof(Gaussian)));
    CUDA_CHECK(cudaMalloc(&d_ps, MAX_GAUSS * sizeof(Projected)));
    CUDA_CHECK(cudaMalloc(&d_order, MAX_GAUSS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_V, 12 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img, IMG_W * IMG_H * 3));
    CUDA_CHECK(cudaMalloc(&d_depth, IMG_W * IMG_H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src, (IMG_W * IMG_H) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mappts, MAX_GAUSS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mapnrm, MAX_GAUSS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_acc, 29 * sizeof(double)));

    std::vector<Projected> ps(MAX_GAUSS);
    std::vector<int> order(MAX_GAUSS);
    std::vector<unsigned char> img(IMG_W * IMG_H * 3);
    std::vector<float> depth(IMG_W * IMG_H);

    // host-side map + fusion voxel set
    std::vector<Gaussian> map;
    std::vector<float> mappts;          // 3*N
    std::vector<float> mapnrm;          // 3*N world-frame normals
    std::unordered_map<long long, int> voxset;
    auto vkey = [](float x, float y, float z) -> long long {
        long long ix = (long long)std::floor(x / VOXEL) + 100000;
        long long iy = (long long)std::floor(y / VOXEL) + 100000;
        long long iz = (long long)std::floor(z / VOXEL) + 100000;
        return (ix * 200003LL + iy) * 200003LL + iz;
    };

    std::system("mkdir -p gif");
    const int OUT_W = IMG_W * 3 + 24;   // 3 panels + gaps
    const int OUT_H = IMG_H + 50;
    cv::VideoWriter video("gif/gpu_gaussian_splatting_slam.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 15, cv::Size(OUT_W, OUT_H));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float track_ms = 0, render_ms = 0;

    // render helper: project + host depth-sort + render a Gaussian set into img/depth
    auto render_set = [&](Gaussian* d_set, int n_set, const Pose& cam, bool want_depth) {
        float V[12]; view_from_pose(cam, V);
        CUDA_CHECK(cudaMemcpy(d_V, V, 12 * sizeof(float), cudaMemcpyHostToDevice));
        int blk = 128, blocks = (n_set + blk - 1) / blk;
        project_kernel<<<blocks, blk>>>(n_set, d_set, d_V, d_ps);
        CUDA_CHECK(cudaMemcpy(ps.data(), d_ps, n_set * sizeof(Projected), cudaMemcpyDeviceToHost));
        order.clear();
        for (int k = 0; k < n_set; k++) if (ps[k].valid) order.push_back(k);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return ps[a].depth < ps[b].depth; });
        int nv = (int)order.size();
        CUDA_CHECK(cudaMemcpy(d_order, order.data(), nv * sizeof(int), cudaMemcpyHostToDevice));
        dim3 bd(16, 16), gd((IMG_W + 15) / 16, (IMG_H + 15) / 16);
        render_kernel<<<gd, bd>>>(nv, d_ps, d_order, d_img, want_depth ? d_depth : nullptr);
        CUDA_CHECK(cudaMemcpy(img.data(), d_img, IMG_W * IMG_H * 3, cudaMemcpyDeviceToHost));
        if (want_depth) CUDA_CHECK(cudaMemcpy(depth.data(), d_depth, IMG_W * IMG_H * sizeof(float), cudaMemcpyDeviceToHost));
        return cv::Mat(IMG_H, IMG_W, CV_8UC3, img.data()).clone();
    };

    // ground-truth + estimated trajectory
    std::vector<cv::Point2f> gt_traj, est_traj;
    double ate_sq = 0; int ate_n = 0;
    Pose est = pose_identity();
    Pose prev_est = pose_identity();      // pose at f-1

    for (int f = 0; f < N_FRAMES; f++) {
        // ----- ground-truth camera pose (orbit through the room) -----------
        // Back-and-forth scanning sweep over a limited arc: the camera always
        // keeps the central sphere cluster + the same couple of walls in view,
        // so tracking always has a strongly-anchored, full-rank overlap (the
        // revisits act as implicit loop closure -> no unbounded drift).
        float ph = 2.0f * float(M_PI) * f / 90.0f;
        float az = 1.0f * std::sin(ph);                 // azimuth sweep +-57 deg
        float radius = 2.5f + 0.25f * std::sin(0.7f * ph);
        float ex = radius * std::sin(az);
        float ez = radius * std::cos(az);
        float ey = 1.35f + 0.30f * std::sin(0.5f * ph);
        // always look at the central sphere cluster -> rich, full-rank geometry
        float tx = 0.0f, ty = 0.55f, tz = 0.0f;
        float fwd[3] = {tx - ex, ty - ey, tz - ez};
        float fl = std::sqrt(fwd[0]*fwd[0]+fwd[1]*fwd[1]+fwd[2]*fwd[2]);
        for (float& c : fwd) c /= fl;
        float up[3] = {0, 1, 0};
        float right[3] = {up[1]*fwd[2]-up[2]*fwd[1], up[2]*fwd[0]-up[0]*fwd[2], up[0]*fwd[1]-up[1]*fwd[0]};
        float rl = std::sqrt(right[0]*right[0]+right[1]*right[1]+right[2]*right[2]);
        for (float& c : right) c /= rl;
        float u2[3] = {fwd[1]*right[2]-fwd[2]*right[1], fwd[2]*right[0]-fwd[0]*right[2], fwd[0]*right[1]-fwd[1]*right[0]};
        // camera->world R columns = [right, up', fwd]
        Pose gt;
        gt.R[0]=right[0]; gt.R[1]=u2[0]; gt.R[2]=fwd[0];
        gt.R[3]=right[1]; gt.R[4]=u2[1]; gt.R[5]=fwd[1];
        gt.R[6]=right[2]; gt.R[7]=u2[2]; gt.R[8]=fwd[2];
        gt.t[0]=ex; gt.t[1]=ey; gt.t[2]=ez;

        // ----- SENSOR: ray-cast the room from the GT pose -> sharp RGB-D ---
        CUDA_CHECK(cudaMemcpy(d_V, gt.R, 9 * sizeof(float), cudaMemcpyHostToDevice));
        {
            dim3 bd(16, 16), gd((IMG_W + 15) / 16, (IMG_H + 15) / 16);
            raycast_kernel<<<gd, bd>>>(gt.t[0], gt.t[1], gt.t[2], d_V, d_img, d_depth);
            CUDA_CHECK(cudaMemcpy(img.data(), d_img, IMG_W * IMG_H * 3, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(depth.data(), d_depth, IMG_W * IMG_H * sizeof(float), cudaMemcpyDeviceToHost));
        }
        cv::Mat sensor_rgb = cv::Mat(IMG_H, IMG_W, CV_8UC3, img.data()).clone();
        std::vector<float> obs_depth = depth;     // copy sensor depth

        // ----- back-project observed depth to camera-frame points ----------
        // (cam coords: x right, y up, z forward)
        std::vector<float> obs_cam;               // for ICP (camera frame), stride ICP_STRIDE
        std::vector<float> fuse_cam;              // for mapping, stride OBS_STRIDE
        std::vector<float> fuse_nrm;              // camera-frame normals for fuse points
        std::vector<unsigned char> fuse_col;      // bgr per fuse point
        // back-project pixel -> camera-frame point; returns false if depth invalid.
        auto bp = [&](int x, int y, float& ox, float& oy, float& oz) -> bool {
            if (x < 0 || x >= IMG_W || y < 0 || y >= IMG_H) return false;
            float d = obs_depth[y * IMG_W + x];
            if (d <= Z_NEAR) return false;
            ox = (x - CXp) / FOCAL * d; oy = -(y - CYp) / FOCAL * d; oz = d; return true;
        };
        obs_cam.reserve(IMG_W * IMG_H / (ICP_STRIDE * ICP_STRIDE) * 3);
        for (int py = 0; py < IMG_H; py += 1) {
            for (int pxr = 0; pxr < IMG_W; pxr += 1) {
                float cxv, cyv, czv;
                if (!bp(pxr, py, cxv, cyv, czv)) continue;
                if (py % ICP_STRIDE == 0 && pxr % ICP_STRIDE == 0) {
                    obs_cam.push_back(cxv); obs_cam.push_back(cyv); obs_cam.push_back(czv);
                }
                if (py % OBS_STRIDE == 0 && pxr % OBS_STRIDE == 0) {
                    // normal from depth-image cross product of tangents (camera frame)
                    float ax, ay, az, bx, by, bz, lx, ly, lz, ux, uy, uz;
                    bool ok = bp(pxr + 2, py, ax, ay, az) && bp(pxr - 2, py, bx, by, bz) &&
                              bp(pxr, py + 2, ux, uy, uz) && bp(pxr, py - 2, lx, ly, lz);
                    float nx = 0, ny = 0, nz = -1;
                    if (ok) {
                        float t1x = ax-bx, t1y = ay-by, t1z = az-bz;
                        float t2x = ux-lx, t2y = uy-ly, t2z = uz-lz;
                        nx = t1y*t2z - t1z*t2y; ny = t1z*t2x - t1x*t2z; nz = t1x*t2y - t1y*t2x;
                        float nl = std::sqrt(nx*nx+ny*ny+nz*nz);
                        if (nl > 1e-6f) { nx/=nl; ny/=nl; nz/=nl; } else { nx=0;ny=0;nz=-1; }
                        if (nz > 0) { nx=-nx; ny=-ny; nz=-nz; }   // face the camera (-z)
                    }
                    fuse_cam.push_back(cxv); fuse_cam.push_back(cyv); fuse_cam.push_back(czv);
                    fuse_nrm.push_back(nx); fuse_nrm.push_back(ny); fuse_nrm.push_back(nz);
                    const unsigned char* bgr = sensor_rgb.ptr<unsigned char>(py) + pxr * 3;
                    fuse_col.push_back(bgr[0]); fuse_col.push_back(bgr[1]); fuse_col.push_back(bgr[2]);
                }
            }
        }

        // ----- TRACKING ----------------------------------------------------
        if (f == 0) {
            est = gt;                 // anchor first frame to ground truth gauge
            prev_est = gt;
        } else {
            // bounded sweep: previous pose is a stable ICP initial guess
            est = prev_est;   // smooth bounded motion: previous pose is a good init
            int n_src = (int)obs_cam.size() / 3;
            cudaEventRecord(e0);
            for (int it = 0; it < ICP_ITERS && !map.empty(); it++) {
                // transform obs cam points to world by current est
                std::vector<float> src(n_src * 3);
                for (int i = 0; i < n_src; i++) {
                    float cx = obs_cam[3*i], cy = obs_cam[3*i+1], cz = obs_cam[3*i+2];
                    src[3*i+0] = est.R[0]*cx + est.R[1]*cy + est.R[2]*cz + est.t[0];
                    src[3*i+1] = est.R[3]*cx + est.R[4]*cy + est.R[5]*cz + est.t[1];
                    src[3*i+2] = est.R[6]*cx + est.R[7]*cy + est.R[8]*cz + est.t[2];
                }
                CUDA_CHECK(cudaMemcpy(d_src, src.data(), n_src * 3 * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_acc, 0, 29 * sizeof(double)));
                int blk = 128, blocks = (n_src + blk - 1) / blk;
                icp_kernel<<<blocks, blk, blk * sizeof(double)>>>(d_src, n_src, d_mappts, d_mapnrm, (int)map.size(), d_acc);
                double acc[29];
                CUDA_CHECK(cudaMemcpy(acc, d_acc, 29 * sizeof(double), cudaMemcpyDeviceToHost));
                if (acc[28] < 10) break;
                double H[6][6], g[6]; int idx = 0;
                for (int a = 0; a < 6; a++) for (int b = a; b < 6; b++) { H[a][b] = H[b][a] = acc[idx++]; }
                for (int a = 0; a < 6; a++) { g[a] = -acc[21 + a]; H[a][a] += 1e-4 * H[a][a] + 1e-7; }
                double dx[6];
                if (!solve6(H, g, dx)) break;
                bool bad = false;
                for (int a = 0; a < 6; a++) if (!std::isfinite(dx[a])) bad = true;
                if (bad) { est = prev_est; break; }
                // clamp per-iteration step (guards weak-observability blow-ups)
                double rot = std::sqrt(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]);
                double tr  = std::sqrt(dx[3]*dx[3]+dx[4]*dx[4]+dx[5]*dx[5]);
                double sc = 1.0;
                if (rot > 0.08) sc = std::min(sc, 0.08 / rot);
                if (tr  > 0.12) sc = std::min(sc, 0.12 / tr);
                for (int a = 0; a < 6; a++) dx[a] *= sc;
                // update pose by incremental world-frame transform exp([w]) , t
                float w[3] = {(float)dx[0], (float)dx[1], (float)dx[2]};
                float Rinc[9]; rodrigues(w, Rinc);
                float Rnew[9]; mat3_mul(Rinc, est.R, Rnew);
                float tnew[3];
                tnew[0] = Rinc[0]*est.t[0]+Rinc[1]*est.t[1]+Rinc[2]*est.t[2] + (float)dx[3];
                tnew[1] = Rinc[3]*est.t[0]+Rinc[4]*est.t[1]+Rinc[5]*est.t[2] + (float)dx[4];
                tnew[2] = Rinc[6]*est.t[0]+Rinc[7]*est.t[1]+Rinc[8]*est.t[2] + (float)dx[5];
                for (int i = 0; i < 9; i++) est.R[i] = Rnew[i];
                for (int i = 0; i < 3; i++) est.t[i] = tnew[i];
                double step = std::sqrt(dx[3]*dx[3]+dx[4]*dx[4]+dx[5]*dx[5]);
                if (step < 1e-4) break;
            }
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1); track_ms += ms;
        }
        prev_est = est;

        // ATE bookkeeping
        float dxp = est.t[0]-gt.t[0], dyp = est.t[1]-gt.t[1], dzp = est.t[2]-gt.t[2];
        ate_sq += dxp*dxp + dyp*dyp + dzp*dzp; ate_n++;

        // ----- MAPPING: fuse observed points with the ESTIMATED pose -------
        int added = 0;
        for (size_t i = 0; i < fuse_cam.size() / 3; i++) {
            float cx = fuse_cam[3*i], cy = fuse_cam[3*i+1], cz = fuse_cam[3*i+2];
            float wx = est.R[0]*cx + est.R[1]*cy + est.R[2]*cz + est.t[0];
            float wy = est.R[3]*cx + est.R[4]*cy + est.R[5]*cz + est.t[1];
            float wz = est.R[6]*cx + est.R[7]*cy + est.R[8]*cz + est.t[2];
            long long key = vkey(wx, wy, wz);
            if (voxset.count(key)) continue;
            if ((int)map.size() >= MAX_GAUSS) break;
            voxset[key] = 1;
            // rotate camera-frame normal into world
            float ncx = fuse_nrm[3*i], ncy = fuse_nrm[3*i+1], ncz = fuse_nrm[3*i+2];
            float nwx = est.R[0]*ncx + est.R[1]*ncy + est.R[2]*ncz;
            float nwy = est.R[3]*ncx + est.R[4]*ncy + est.R[5]*ncz;
            float nwz = est.R[6]*ncx + est.R[7]*ncy + est.R[8]*ncz;
            Gaussian g;
            g.mx = wx; g.my = wy; g.mz = wz; g.s = 0.06f;
            g.b = fuse_col[3*i]/255.0f; g.g = fuse_col[3*i+1]/255.0f; g.r = fuse_col[3*i+2]/255.0f;
            g.a = 0.9f;
            map.push_back(g);
            mappts.push_back(wx); mappts.push_back(wy); mappts.push_back(wz);
            mapnrm.push_back(nwx); mapnrm.push_back(nwy); mapnrm.push_back(nwz);
            added++;
        }
        if (added) {
            CUDA_CHECK(cudaMemcpy(d_map, map.data(), map.size() * sizeof(Gaussian), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mappts, mappts.data(), mappts.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mapnrm, mapnrm.data(), mapnrm.size() * sizeof(float), cudaMemcpyHostToDevice));
        }

        // ----- RENDER panels ----------------------------------------------
        cudaEventRecord(e0);
        // (b) map rendered from the estimated pose
        cv::Mat map_view = render_set(d_map, (int)map.size(), est, false);
        // (c) global map from a slow orbit overview camera
        float ou = float(f) / N_FRAMES * 2.0f * float(M_PI) + 0.6f;
        Pose ov;
        float orad = 6.6f, oex = orad*std::sin(ou), oez = orad*std::cos(ou), oey = 4.2f;
        float of_[3] = {-oex, 0.6f-oey, -oez};
        float ofl = std::sqrt(of_[0]*of_[0]+of_[1]*of_[1]+of_[2]*of_[2]); for (float& c: of_) c/=ofl;
        float oup[3]={0,1,0};
        float orr[3]={oup[1]*of_[2]-oup[2]*of_[1], oup[2]*of_[0]-oup[0]*of_[2], oup[0]*of_[1]-oup[1]*of_[0]};
        float orl=std::sqrt(orr[0]*orr[0]+orr[1]*orr[1]+orr[2]*orr[2]); for (float& c: orr) c/=orl;
        float ou2[3]={of_[1]*orr[2]-of_[2]*orr[1], of_[2]*orr[0]-of_[0]*orr[2], of_[0]*orr[1]-of_[1]*orr[0]};
        ov.R[0]=orr[0];ov.R[1]=ou2[0];ov.R[2]=of_[0];
        ov.R[3]=orr[1];ov.R[4]=ou2[1];ov.R[5]=of_[1];
        ov.R[6]=orr[2];ov.R[7]=ou2[2];ov.R[8]=of_[2];
        ov.t[0]=oex;ov.t[1]=oey;ov.t[2]=oez;
        cv::Mat overview = render_set(d_map, (int)map.size(), ov, false);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float rms = 0; cudaEventElapsedTime(&rms, e0, e1); render_ms += rms;

        // ----- compose output frame ----------------------------------------
        cv::Mat out(OUT_H, OUT_W, CV_8UC3, cv::Scalar(20, 20, 24));
        sensor_rgb.copyTo(out(cv::Rect(0, 40, IMG_W, IMG_H)));
        map_view.copyTo(out(cv::Rect(IMG_W + 12, 40, IMG_W, IMG_H)));
        overview.copyTo(out(cv::Rect(IMG_W * 2 + 24, 40, IMG_W, IMG_H)));
        auto label = [&](const char* s, int x) {
            cv::putText(out, s, cv::Point(x, 26), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(230,230,230), 1, cv::LINE_AA);
        };
        label("RGB-D sensor (ground truth)", 8);
        label("tracked map @ estimated pose", IMG_W + 20);
        label("global Gaussian map (grows)", IMG_W * 2 + 32);
        float ate_now = std::sqrt(ate_sq / std::max(1, ate_n));
        char hud[160];
        std::snprintf(hud, sizeof(hud), "frame %3d/%d   gaussians=%d   ATE=%.3f m   track=%.1f ms",
                      f + 1, N_FRAMES, (int)map.size(), ate_now,
                      f > 0 ? track_ms / f : 0.0f);
        cv::putText(out, hud, cv::Point(8, OUT_H - 8), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(120, 230, 255), 1, cv::LINE_AA);
        video.write(out);
    }
    video.release();

    double ate = std::sqrt(ate_sq / std::max(1, ate_n));
    std::printf("\n=== GPU Gaussian-Splatting SLAM ===\n");
    std::printf("frames:            %d\n", N_FRAMES);
    std::printf("final map size:    %d Gaussians\n", (int)map.size());
    std::printf("ATE (RMSE):        %.4f m\n", ate);
    std::printf("avg tracking time: %.2f ms / frame (GPU ICP, %d iters)\n", track_ms / (N_FRAMES - 1), ICP_ITERS);
    std::printf("avg render time:   %.2f ms / frame (2 splat renders)\n", render_ms / N_FRAMES);

    avi_to_gif("gif/gpu_gaussian_splatting_slam.avi", "gif/gpu_gaussian_splatting_slam.gif", 10, 600);
    std::printf("GIF saved to gif/gpu_gaussian_splatting_slam.gif\n");

    cudaFree(d_map); cudaFree(d_ps); cudaFree(d_order); cudaFree(d_V);
    cudaFree(d_img); cudaFree(d_depth); cudaFree(d_src); cudaFree(d_mappts); cudaFree(d_mapnrm); cudaFree(d_acc);
    return 0;
}
