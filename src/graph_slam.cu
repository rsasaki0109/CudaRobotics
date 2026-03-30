/*************************************************************************
    > File Name: graph_slam.cu
    > CUDA-parallelized Graph-based SLAM
    > Optimizes a 2D pose graph using Gauss-Newton with GPU-accelerated
    > Hessian assembly and Conjugate Gradient solver.
    > Simulation: robot drives in a loop (~200 poses, ~250 edges).
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA error check macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
#define PI 3.141592653f

// Simulation parameters
#define N_POSES       200
#define ODOM_NOISE_X  0.1f
#define ODOM_NOISE_Y  0.05f
#define ODOM_NOISE_TH 0.03f
#define LOOP_NOISE_X  0.15f
#define LOOP_NOISE_Y  0.15f
#define LOOP_NOISE_TH 0.05f
#define LOOP_DIST_TH  3.0f
#define LOOP_MIN_SEP  30

// Gauss-Newton
#define GN_ITERATIONS 20
#define CG_MAX_ITER   300
#define CG_TOL        1e-6f

// ---------------------------------------------------------------------------
// Structures
// ---------------------------------------------------------------------------
struct Pose2D {
    float x, y, theta;
};

struct Edge {
    int from, to;
    float dx, dy, dtheta;       // measurement z_ij
    float info[9];              // 3x3 information matrix (row-major)
};

// ---------------------------------------------------------------------------
// Host helper: normalize angle to [-pi, pi]
// ---------------------------------------------------------------------------
static inline float normalize_angle_h(float a) {
    a = fmodf(a + PI, 2.0f * PI);
    if (a < 0.0f) a += 2.0f * PI;
    return a - PI;
}

// ---------------------------------------------------------------------------
// Device helper: normalize angle to [-pi, pi]
// ---------------------------------------------------------------------------
__device__ float normalize_angle(float a) {
    a = fmodf(a + PI, 2.0f * PI);
    if (a < 0.0f) a += 2.0f * PI;
    return a - PI;
}

// ---------------------------------------------------------------------------
// Kernel: build Hessian H and vector b from edges
// Each thread processes one edge. Uses atomicAdd to accumulate into H and b.
// H is (3*n_poses x 3*n_poses) but we only store the needed entries.
// We store H as dense (dim x dim) float array.
// ---------------------------------------------------------------------------
__global__ void build_H_b_kernel(
    const Pose2D* poses,
    const Edge*   edges,
    float*        H,
    float*        b,
    int           n_edges,
    int           dim)          // dim = 3 * n_poses
{
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_edges) return;

    Edge e = edges[eid];
    int i = e.from;
    int j = e.to;

    // Poses
    float xi = poses[i].x,     yi = poses[i].y,     ti = poses[i].theta;
    float xj = poses[j].x,     yj = poses[j].y,     tj = poses[j].theta;

    // Compute relative pose: x_j ominus x_i
    float cs = cosf(ti);
    float sn = sinf(ti);
    float dxw = xj - xi;
    float dyw = yj - yi;
    float rel_x =  cs * dxw + sn * dyw;
    float rel_y = -sn * dxw + cs * dyw;
    float rel_t = normalize_angle(tj - ti);

    // Error: e = z_ij - (x_j ominus x_i)
    float err[3];
    err[0] = e.dx - rel_x;
    err[1] = e.dy - rel_y;
    err[2] = normalize_angle(e.dtheta - rel_t);

    // Jacobian of (x_j ominus x_i) w.r.t. x_i  =>  A (3x3)
    // d(rel)/d(xi) :
    //   d(rel_x)/dxi =  -cos(ti),  d(rel_x)/dyi = -sin(ti),
    //     d(rel_x)/dti = -sin(ti)*dxw + cos(ti)*dyw
    //   d(rel_y)/dxi =   sin(ti),  d(rel_y)/dyi = -cos(ti),
    //     d(rel_y)/dti = -cos(ti)*dxw - sin(ti)*dyw
    //   d(rel_t)/dxi = 0,          d(rel_t)/dyi = 0,
    //     d(rel_t)/dti = -1
    // Since e = z - rel, Jacobian of e w.r.t. xi is -d(rel)/d(xi) => Ai
    // And Jacobian of e w.r.t. xj is -d(rel)/d(xj)  => Bi
    // d(rel)/d(xj): d(rel_x)/dxj = cos(ti), d(rel_x)/dyj = sin(ti), ...
    // e_wrt_xi = -A,  e_wrt_xj = -B

    // Jacobian A = de/d(xi) = -d(rel)/d(xi)
    float A[9]; // row-major 3x3
    A[0] =  cs;   A[1] =  sn;   A[2] = -sn * dxw + cs * dyw;
    A[3] = -sn;   A[4] =  cs;   A[5] = -cs * dxw - sn * dyw;
    A[6] =  0.0f; A[7] =  0.0f; A[8] =  1.0f;
    // Negate because e = z - rel, so de/dxi = -(d_rel/dxi)
    // Actually A above is d(rel)/d(xi), and de/dxi = -A
    // Let's just negate:
    for (int k = 0; k < 9; k++) A[k] = -A[k];

    // Jacobian B = de/d(xj) = -d(rel)/d(xj)
    // d(rel)/d(xj): [[cos(ti), sin(ti), 0], [-sin(ti), cos(ti), 0], [0, 0, 1]]
    float B[9];
    B[0] = -cs;   B[1] = -sn;   B[2] =  0.0f;
    B[3] =  sn;   B[4] = -cs;   B[5] =  0.0f;
    B[6] =  0.0f; B[7] =  0.0f; B[8] = -1.0f;

    // Information matrix
    float omega[9];
    for (int k = 0; k < 9; k++) omega[k] = e.info[k];

    // Compute contributions: H += J^T * Omega * J,  b += J^T * Omega * e
    // J for this edge has blocks A (at cols 3*i..3*i+2) and B (at cols 3*j..3*j+2)
    // So the 4 blocks to add:
    //   H[ii] += A^T * Omega * A
    //   H[ij] += A^T * Omega * B
    //   H[ji] += B^T * Omega * A
    //   H[jj] += B^T * Omega * B
    //   b[i]  += A^T * Omega * e
    //   b[j]  += B^T * Omega * e

    // First compute Omega * A, Omega * B, Omega * err
    // tmp = Omega * M  (3x3 * 3x3 => 3x3), row-major
    // Omega * err => 3-vec
    float OA[9], OB[9], Oe[3];

    for (int r = 0; r < 3; r++) {
        Oe[r] = 0.0f;
        for (int c = 0; c < 3; c++) {
            float ov = omega[r * 3 + c];
            Oe[r] += ov * err[c];
            OA[r * 3 + c] = 0.0f;
            OB[r * 3 + c] = 0.0f;
            for (int k = 0; k < 3; k++) {
                // Actually let's do it correctly:
                // OA[r][c] = sum_k omega[r][k] * A[k][c]
            }
        }
    }
    // Redo OA, OB properly
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float sum_a = 0.0f, sum_b = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum_a += omega[r * 3 + k] * A[k * 3 + c];
                sum_b += omega[r * 3 + k] * B[k * 3 + c];
            }
            OA[r * 3 + c] = sum_a;
            OB[r * 3 + c] = sum_b;
        }
    }

    // H[ii] += A^T * OA  (3x3)
    int bi = 3 * i;
    int bj = 3 * j;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float val = 0.0f;
            for (int k = 0; k < 3; k++) {
                val += A[k * 3 + r] * OA[k * 3 + c]; // A^T[r][k] = A[k][r]
            }
            atomicAdd(&H[(bi + r) * dim + (bi + c)], val);
        }
    }
    // H[ij] += A^T * OB
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float val = 0.0f;
            for (int k = 0; k < 3; k++) {
                val += A[k * 3 + r] * OB[k * 3 + c];
            }
            atomicAdd(&H[(bi + r) * dim + (bj + c)], val);
        }
    }
    // H[ji] += B^T * OA
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float val = 0.0f;
            for (int k = 0; k < 3; k++) {
                val += B[k * 3 + r] * OA[k * 3 + c];
            }
            atomicAdd(&H[(bj + r) * dim + (bi + c)], val);
        }
    }
    // H[jj] += B^T * OB
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float val = 0.0f;
            for (int k = 0; k < 3; k++) {
                val += B[k * 3 + r] * OB[k * 3 + c];
            }
            atomicAdd(&H[(bj + r) * dim + (bj + c)], val);
        }
    }

    // b[i] += A^T * Oe
    for (int r = 0; r < 3; r++) {
        float val = 0.0f;
        for (int k = 0; k < 3; k++) {
            val += A[k * 3 + r] * Oe[k];
        }
        atomicAdd(&b[bi + r], val);
    }
    // b[j] += B^T * Oe
    for (int r = 0; r < 3; r++) {
        float val = 0.0f;
        for (int k = 0; k < 3; k++) {
            val += B[k * 3 + r] * Oe[k];
        }
        atomicAdd(&b[bj + r], val);
    }
}

// ---------------------------------------------------------------------------
// Conjugate Gradient kernels: solve H * dx = b on GPU
// All operate on device arrays of length dim.
// ---------------------------------------------------------------------------

// y = H * x  (dense matrix-vector multiply)
__global__ void cg_matvec_kernel(const float* H, const float* x, float* y, int dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim) return;
    float sum = 0.0f;
    for (int c = 0; c < dim; c++) {
        sum += H[row * dim + c] * x[c];
    }
    y[row] = sum;
}

// Block-level reduction dot product: result[0] = sum(a[i]*b[i])
// Uses shared memory. One block only (for small dim).
__global__ void cg_dot_kernel(const float* a, const float* b, float* result, int dim) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        sum += a[i] * b[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) result[0] = sdata[0];
}

// y = y + alpha * x
__global__ void cg_axpy_kernel(float* y, const float* x, float alpha, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        y[i] += alpha * x[i];
    }
}

// y = b - H*x  (for computing residual)
__global__ void cg_residual_kernel(const float* H, const float* x, const float* b, float* r, int dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim) return;
    float sum = 0.0f;
    for (int c = 0; c < dim; c++) {
        sum += H[row * dim + c] * x[c];
    }
    r[row] = b[row] - sum;
}

// copy: dst = src
__global__ void cg_copy_kernel(float* dst, const float* src, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) dst[i] = src[i];
}

// scale: y = alpha * x
__global__ void cg_scale_kernel(float* y, const float* x, float alpha, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) y[i] = alpha * x[i];
}

// ---------------------------------------------------------------------------
// Kernel: update poses by dx
// ---------------------------------------------------------------------------
__global__ void update_poses_kernel(Pose2D* poses, const float* dx, int n_poses) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_poses) return;
    // First pose is fixed (anchor)
    if (i == 0) return;
    poses[i].x     += dx[3 * i + 0];
    poses[i].y     += dx[3 * i + 1];
    poses[i].theta  = normalize_angle(poses[i].theta + dx[3 * i + 2]);
}

// ---------------------------------------------------------------------------
// Host: Conjugate Gradient solver on GPU
// Solves H * dx = b where H is dim x dim, b and dx are dim-vectors.
// H, b are already on device; dx is output on device (initialized to 0).
// We fix the first pose by zeroing its rows/cols in H and b beforehand.
// ---------------------------------------------------------------------------
void solve_cg_gpu(float* d_H, float* d_b, float* d_dx, int dim) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    int smem = threads * sizeof(float);

    // Allocate CG temporaries
    float *d_r, *d_p, *d_Ap, *d_dot1, *d_dot2;
    CUDA_CHECK(cudaMalloc(&d_r,    dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p,    dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ap,   dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dot1, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dot2, sizeof(float)));

    // dx = 0
    CUDA_CHECK(cudaMemset(d_dx, 0, dim * sizeof(float)));

    // r = b - H*dx = b  (since dx=0)
    CUDA_CHECK(cudaMemcpy(d_r, d_b, dim * sizeof(float), cudaMemcpyDeviceToDevice));
    // p = r
    CUDA_CHECK(cudaMemcpy(d_p, d_r, dim * sizeof(float), cudaMemcpyDeviceToDevice));

    // rr = r^T * r
    float rr;
    cg_dot_kernel<<<1, threads, smem>>>(d_r, d_r, d_dot1, dim);
    CUDA_CHECK(cudaMemcpy(&rr, d_dot1, sizeof(float), cudaMemcpyDeviceToHost));

    float rr0 = rr;
    if (rr0 < 1e-12f) {
        // Already solved
        cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap);
        cudaFree(d_dot1); cudaFree(d_dot2);
        return;
    }

    for (int iter = 0; iter < CG_MAX_ITER; iter++) {
        // Ap = H * p
        cg_matvec_kernel<<<blocks, threads>>>(d_H, d_p, d_Ap, dim);

        // pAp = p^T * Ap
        float pAp;
        cg_dot_kernel<<<1, threads, smem>>>(d_p, d_Ap, d_dot1, dim);
        CUDA_CHECK(cudaMemcpy(&pAp, d_dot1, sizeof(float), cudaMemcpyDeviceToHost));

        if (fabsf(pAp) < 1e-14f) break;

        float alpha = rr / pAp;

        // dx += alpha * p
        cg_axpy_kernel<<<blocks, threads>>>(d_dx, d_p, alpha, dim);

        // r -= alpha * Ap
        cg_axpy_kernel<<<blocks, threads>>>(d_r, d_Ap, -alpha, dim);

        // new_rr = r^T * r
        float new_rr;
        cg_dot_kernel<<<1, threads, smem>>>(d_r, d_r, d_dot1, dim);
        CUDA_CHECK(cudaMemcpy(&new_rr, d_dot1, sizeof(float), cudaMemcpyDeviceToHost));

        if (new_rr / rr0 < CG_TOL * CG_TOL) break;

        float beta = new_rr / rr;

        // p = r + beta * p
        cg_scale_kernel<<<blocks, threads>>>(d_p, d_p, beta, dim);
        cg_axpy_kernel<<<blocks, threads>>>(d_p, d_r, 1.0f, dim);

        rr = new_rr;
    }

    cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap);
    cudaFree(d_dot1); cudaFree(d_dot2);
}

// ---------------------------------------------------------------------------
// Host: Fix first pose in H and b (anchor constraint)
// Zero out first 3 rows/cols of H, set diagonal to 1, zero b[0..2]
// ---------------------------------------------------------------------------
__global__ void fix_first_pose_kernel(float* H, float* b, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    // Zero row 0,1,2 and column 0,1,2
    for (int k = 0; k < 3; k++) {
        H[k * dim + idx] = 0.0f;  // row k
        H[idx * dim + k] = 0.0f;  // col k
    }
    if (idx < 3) {
        H[idx * dim + idx] = 1.0f; // diagonal
        b[idx] = 0.0f;
    }
}

// ===========================================================================
// Simulation: generate ground truth poses and edges
// ===========================================================================
void generate_simulation(
    std::vector<Pose2D>& gt_poses,
    std::vector<Pose2D>& odom_poses,
    std::vector<Edge>&   edges,
    std::mt19937& rng)
{
    // Robot drives roughly in a loop
    float radius = 15.0f;
    int n = N_POSES;

    std::normal_distribution<float> noise_x(0.0f, ODOM_NOISE_X);
    std::normal_distribution<float> noise_y(0.0f, ODOM_NOISE_Y);
    std::normal_distribution<float> noise_th(0.0f, ODOM_NOISE_TH);
    std::normal_distribution<float> lnoise_x(0.0f, LOOP_NOISE_X);
    std::normal_distribution<float> lnoise_y(0.0f, LOOP_NOISE_Y);
    std::normal_distribution<float> lnoise_th(0.0f, LOOP_NOISE_TH);

    gt_poses.resize(n);
    odom_poses.resize(n);

    // Generate ground-truth: circular trajectory
    for (int i = 0; i < n; i++) {
        float angle = 2.0f * PI * i / (float)(n - 1);
        gt_poses[i].x = radius * cosf(angle);
        gt_poses[i].y = radius * sinf(angle);
        gt_poses[i].theta = normalize_angle_h(angle + PI / 2.0f); // tangent direction
    }

    // Odometry poses: accumulate noisy relative motion
    odom_poses[0] = gt_poses[0];
    for (int i = 1; i < n; i++) {
        // Ground truth relative
        float cs = cosf(gt_poses[i - 1].theta);
        float sn = sinf(gt_poses[i - 1].theta);
        float dxw = gt_poses[i].x - gt_poses[i - 1].x;
        float dyw = gt_poses[i].y - gt_poses[i - 1].y;
        float rel_x =  cs * dxw + sn * dyw + noise_x(rng);
        float rel_y = -sn * dxw + cs * dyw + noise_y(rng);
        float rel_t = normalize_angle_h(gt_poses[i].theta - gt_poses[i - 1].theta) + noise_th(rng);

        // Odometry edge
        Edge e;
        e.from = i - 1;
        e.to = i;
        e.dx = rel_x;
        e.dy = rel_y;
        e.dtheta = rel_t;
        // Information matrix: diagonal, inverse of noise covariance
        memset(e.info, 0, sizeof(e.info));
        e.info[0] = 1.0f / (ODOM_NOISE_X * ODOM_NOISE_X);
        e.info[4] = 1.0f / (ODOM_NOISE_Y * ODOM_NOISE_Y);
        e.info[8] = 1.0f / (ODOM_NOISE_TH * ODOM_NOISE_TH);
        edges.push_back(e);

        // Accumulate odometry for initial guess
        float ocs = cosf(odom_poses[i - 1].theta);
        float osn = sinf(odom_poses[i - 1].theta);
        odom_poses[i].x = odom_poses[i - 1].x + ocs * rel_x - osn * rel_y;
        odom_poses[i].y = odom_poses[i - 1].y + osn * rel_x + ocs * rel_y;
        odom_poses[i].theta = normalize_angle_h(odom_poses[i - 1].theta + rel_t);
    }

    // Loop closure edges: connect poses that are close in ground truth but far in sequence
    for (int i = 0; i < n; i++) {
        for (int j = i + LOOP_MIN_SEP; j < n; j++) {
            float dx = gt_poses[i].x - gt_poses[j].x;
            float dy = gt_poses[i].y - gt_poses[j].y;
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < LOOP_DIST_TH) {
                // Compute ground truth relative measurement with noise
                float cs = cosf(gt_poses[i].theta);
                float sn = sinf(gt_poses[i].theta);
                float dxw = gt_poses[j].x - gt_poses[i].x;
                float dyw = gt_poses[j].y - gt_poses[i].y;
                float rel_x =  cs * dxw + sn * dyw + lnoise_x(rng);
                float rel_y = -sn * dxw + cs * dyw + lnoise_y(rng);
                float rel_t = normalize_angle_h(gt_poses[j].theta - gt_poses[i].theta) + lnoise_th(rng);

                Edge e;
                e.from = i;
                e.to = j;
                e.dx = rel_x;
                e.dy = rel_y;
                e.dtheta = rel_t;
                memset(e.info, 0, sizeof(e.info));
                e.info[0] = 1.0f / (LOOP_NOISE_X * LOOP_NOISE_X);
                e.info[4] = 1.0f / (LOOP_NOISE_Y * LOOP_NOISE_Y);
                e.info[8] = 1.0f / (LOOP_NOISE_TH * LOOP_NOISE_TH);
                edges.push_back(e);
            }
        }
    }

    printf("Generated %d poses, %d edges (%d odometry + %d loop closures)\n",
           n, (int)edges.size(), n - 1, (int)edges.size() - (n - 1));
}

// ===========================================================================
// Visualization
// ===========================================================================
void visualize(
    const std::vector<Pose2D>& gt_poses,
    const std::vector<Pose2D>& init_poses,
    const std::vector<Pose2D>& opt_poses,
    const std::vector<Edge>&   edges,
    int iteration)
{
    int W = 800, H = 800;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

    // Compute bounding box of all poses
    float minx = 1e9f, maxx = -1e9f, miny = 1e9f, maxy = -1e9f;
    for (auto& p : gt_poses) {
        minx = std::min(minx, p.x); maxx = std::max(maxx, p.x);
        miny = std::min(miny, p.y); maxy = std::max(maxy, p.y);
    }
    for (auto& p : init_poses) {
        minx = std::min(minx, p.x); maxx = std::max(maxx, p.x);
        miny = std::min(miny, p.y); maxy = std::max(maxy, p.y);
    }
    for (auto& p : opt_poses) {
        minx = std::min(minx, p.x); maxx = std::max(maxx, p.x);
        miny = std::min(miny, p.y); maxy = std::max(maxy, p.y);
    }

    float pad = 3.0f;
    minx -= pad; miny -= pad; maxx += pad; maxy += pad;
    float sx = (W - 40) / (maxx - minx);
    float sy = (H - 40) / (maxy - miny);
    float scale = std::min(sx, sy);

    auto toPixel = [&](float x, float y) -> cv::Point {
        int px = (int)((x - minx) * scale) + 20;
        int py = H - ((int)((y - miny) * scale) + 20);
        return cv::Point(px, py);
    };

    // Draw loop closure edges (red)
    int n_odom = (int)gt_poses.size() - 1;
    for (int i = n_odom; i < (int)edges.size(); i++) {
        cv::Point p1 = toPixel(opt_poses[edges[i].from].x, opt_poses[edges[i].from].y);
        cv::Point p2 = toPixel(opt_poses[edges[i].to].x,   opt_poses[edges[i].to].y);
        cv::line(img, p1, p2, cv::Scalar(0, 0, 200), 1);
    }

    // Draw initial poses (gray) connected
    for (int i = 0; i < (int)init_poses.size() - 1; i++) {
        cv::Point p1 = toPixel(init_poses[i].x, init_poses[i].y);
        cv::Point p2 = toPixel(init_poses[i + 1].x, init_poses[i + 1].y);
        cv::line(img, p1, p2, cv::Scalar(180, 180, 180), 1);
    }
    for (auto& p : init_poses) {
        cv::circle(img, toPixel(p.x, p.y), 2, cv::Scalar(180, 180, 180), -1);
    }

    // Draw ground truth (green) connected
    for (int i = 0; i < (int)gt_poses.size() - 1; i++) {
        cv::Point p1 = toPixel(gt_poses[i].x, gt_poses[i].y);
        cv::Point p2 = toPixel(gt_poses[i + 1].x, gt_poses[i + 1].y);
        cv::line(img, p1, p2, cv::Scalar(0, 180, 0), 2);
    }
    for (auto& p : gt_poses) {
        cv::circle(img, toPixel(p.x, p.y), 3, cv::Scalar(0, 180, 0), -1);
    }

    // Draw optimized poses (blue) connected
    for (int i = 0; i < (int)opt_poses.size() - 1; i++) {
        cv::Point p1 = toPixel(opt_poses[i].x, opt_poses[i].y);
        cv::Point p2 = toPixel(opt_poses[i + 1].x, opt_poses[i + 1].y);
        cv::line(img, p1, p2, cv::Scalar(255, 100, 0), 2);
    }
    for (auto& p : opt_poses) {
        cv::circle(img, toPixel(p.x, p.y), 3, cv::Scalar(255, 100, 0), -1);
    }

    // Legend and iteration info
    char buf[128];
    snprintf(buf, sizeof(buf), "Gauss-Newton iter: %d", iteration);
    cv::putText(img, buf, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 0), 1);
    cv::putText(img, "Green: Ground Truth", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 180, 0), 1);
    cv::putText(img, "Gray: Initial (odometry)", cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(180, 180, 180), 1);
    cv::putText(img, "Blue: Optimized", cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 100, 0), 1);
    cv::putText(img, "Red: Loop closures", cv::Point(10, 110), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 200), 1);

    cv::imshow("graph_slam", img);
    cv::waitKey(100);
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    printf("=== CUDA Graph-based SLAM ===\n");

    // -----------------------------------------------------------------------
    // 1. Generate simulation data
    // -----------------------------------------------------------------------
    std::mt19937 rng(42);
    std::vector<Pose2D> gt_poses, odom_poses;
    std::vector<Edge> edges;
    generate_simulation(gt_poses, odom_poses, edges, rng);

    int n_poses = (int)gt_poses.size();
    int n_edges = (int)edges.size();
    int dim = 3 * n_poses;

    // Save initial poses for visualization
    std::vector<Pose2D> init_poses = odom_poses;

    // -----------------------------------------------------------------------
    // 2. Allocate device memory
    // -----------------------------------------------------------------------
    Pose2D* d_poses;
    Edge*   d_edges;
    float*  d_H;
    float*  d_b;
    float*  d_dx;

    CUDA_CHECK(cudaMalloc(&d_poses, n_poses * sizeof(Pose2D)));
    CUDA_CHECK(cudaMalloc(&d_edges, n_edges * sizeof(Edge)));
    CUDA_CHECK(cudaMalloc(&d_H,    dim * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b,    dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx,   dim * sizeof(float)));

    // Upload edges (constant throughout optimization)
    CUDA_CHECK(cudaMemcpy(d_edges, edges.data(), n_edges * sizeof(Edge), cudaMemcpyHostToDevice));

    // Upload initial pose estimate
    CUDA_CHECK(cudaMemcpy(d_poses, odom_poses.data(), n_poses * sizeof(Pose2D), cudaMemcpyHostToDevice));

    cv::namedWindow("graph_slam", cv::WINDOW_AUTOSIZE);

    // -----------------------------------------------------------------------
    // 3. Gauss-Newton iterations
    // -----------------------------------------------------------------------
    int edge_threads = 256;
    int edge_blocks  = (n_edges + edge_threads - 1) / edge_threads;
    int dim_threads  = 256;
    int dim_blocks   = (dim + dim_threads - 1) / dim_threads;
    int pose_threads = 256;
    int pose_blocks  = (n_poses + pose_threads - 1) / pose_threads;

    std::vector<Pose2D> opt_poses(n_poses);

    for (int iter = 0; iter < GN_ITERATIONS; iter++) {
        // Clear H and b
        CUDA_CHECK(cudaMemset(d_H, 0, dim * dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_b, 0, dim * sizeof(float)));

        // Build H and b from all edges in parallel
        build_H_b_kernel<<<edge_blocks, edge_threads>>>(
            d_poses, d_edges, d_H, d_b, n_edges, dim);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Fix first pose (anchor)
        fix_first_pose_kernel<<<dim_blocks, dim_threads>>>(d_H, d_b, dim);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Solve H * dx = b using Conjugate Gradient on GPU
        solve_cg_gpu(d_H, d_b, d_dx, dim);

        // Update poses
        update_poses_kernel<<<pose_blocks, pose_threads>>>(d_poses, d_dx, n_poses);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download poses for visualization
        CUDA_CHECK(cudaMemcpy(opt_poses.data(), d_poses, n_poses * sizeof(Pose2D), cudaMemcpyDeviceToHost));

        // Compute total error for monitoring
        float total_err = 0.0f;
        for (auto& e : edges) {
            int i = e.from, j = e.to;
            float cs = cosf(opt_poses[i].theta);
            float sn = sinf(opt_poses[i].theta);
            float dxw = opt_poses[j].x - opt_poses[i].x;
            float dyw = opt_poses[j].y - opt_poses[i].y;
            float rel_x =  cs * dxw + sn * dyw;
            float rel_y = -sn * dxw + cs * dyw;
            float rel_t = normalize_angle_h(opt_poses[j].theta - opt_poses[i].theta);
            float ex = e.dx - rel_x;
            float ey = e.dy - rel_y;
            float et = normalize_angle_h(e.dtheta - rel_t);
            total_err += ex * ex + ey * ey + et * et;
        }
        printf("Iter %2d: total error = %.6f\n", iter, total_err);

        visualize(gt_poses, init_poses, opt_poses, edges, iter);
    }

    printf("Optimization complete. Press any key to exit.\n");
    cv::waitKey(0);

    // Cleanup
    cudaFree(d_poses);
    cudaFree(d_edges);
    cudaFree(d_H);
    cudaFree(d_b);
    cudaFree(d_dx);

    return 0;
}
