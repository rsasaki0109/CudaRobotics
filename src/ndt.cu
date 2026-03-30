/*************************************************************************
    > File Name: ndt.cu
    > CUDA-parallelized NDT (Normal Distributions Transform) Scan Matching
    > Algorithm:
    >   1. Build NDT grid from reference scan: divide space into cells,
    >      compute mean and covariance for each cell
    >   2. For source scan, compute score: sum of Gaussian likelihoods
    >   3. Optimize pose (x, y, theta) via Newton's method
    > CUDA kernels:
    >   - build_ndt_grid_kernel: 1 thread per cell, computes mean/covariance
    >   - compute_score_kernel: 1 thread per source point, computes likelihood
    >   - compute_gradient_kernel: 1 thread per point, Jacobian/Hessian via atomicAdd
    > Visualization: OpenCV with animated alignment iterations
 ************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

using namespace std;

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
static const float PI = 3.141592653f;

// NDT grid parameters
static const float CELL_SIZE    = 2.0f;        // meters per cell
static const int   GRID_NX      = 50;          // cells in x
static const int   GRID_NY      = 50;          // cells in y
static const int   GRID_TOTAL   = GRID_NX * GRID_NY;
// World spans [0, GRID_NX*CELL_SIZE) x [0, GRID_NY*CELL_SIZE)
// i.e. 100m x 100m

// Scan parameters
static const int   NUM_SOURCE   = 360;         // source LiDAR points
static const int   MAX_REF      = 4096;        // max reference points

// Newton optimization
static const int   MAX_ITER     = 30;
static const float CONV_THRESH  = 0.001f;

// Minimum points per cell for valid distribution
static const int   MIN_CELL_PTS = 3;

// ---------------------------------------------------------------------------
// NDT cell structure (stored in arrays-of-structs on GPU)
// ---------------------------------------------------------------------------
struct NDTCell {
    float mean_x, mean_y;           // mean of points in cell
    float cov_xx, cov_xy, cov_yy;  // 2x2 covariance
    float inv_xx, inv_xy, inv_yy;   // inverse covariance
    float det;                       // determinant of covariance
    int   count;                     // number of points
    int   valid;                     // 1 if cell has enough points
};

// ---------------------------------------------------------------------------
// 2D point
// ---------------------------------------------------------------------------
struct Point2 {
    float x, y;
};

// ---------------------------------------------------------------------------
// Kernel: Build NDT grid from reference points
//   1 thread per cell. Each thread scans all reference points to find
//   those belonging to its cell, then computes mean and covariance.
// ---------------------------------------------------------------------------
__global__ void build_ndt_grid_kernel(
    NDTCell* cells,
    const float* ref_x, const float* ref_y,
    int n_ref,
    float cell_size, int nx, int ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny) return;

    int cx = idx % nx;
    int cy = idx / nx;

    float x_min = cx * cell_size;
    float x_max = x_min + cell_size;
    float y_min = cy * cell_size;
    float y_max = y_min + cell_size;

    // Pass 1: compute mean
    float sx = 0.0f, sy = 0.0f;
    int count = 0;
    for (int i = 0; i < n_ref; i++) {
        float px = ref_x[i];
        float py = ref_y[i];
        if (px >= x_min && px < x_max && py >= y_min && py < y_max) {
            sx += px;
            sy += py;
            count++;
        }
    }

    cells[idx].count = count;
    if (count < MIN_CELL_PTS) {
        cells[idx].valid = 0;
        return;
    }

    float mx = sx / count;
    float my = sy / count;
    cells[idx].mean_x = mx;
    cells[idx].mean_y = my;

    // Pass 2: compute covariance
    float sxx = 0.0f, sxy = 0.0f, syy = 0.0f;
    for (int i = 0; i < n_ref; i++) {
        float px = ref_x[i];
        float py = ref_y[i];
        if (px >= x_min && px < x_max && py >= y_min && py < y_max) {
            float dx = px - mx;
            float dy = py - my;
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
        }
    }

    float c_xx = sxx / count;
    float c_xy = sxy / count;
    float c_yy = syy / count;

    // Regularize: add small epsilon to diagonal for numerical stability
    float eps = 0.01f;
    c_xx += eps;
    c_yy += eps;

    float det = c_xx * c_yy - c_xy * c_xy;
    if (det < 1e-6f) {
        cells[idx].valid = 0;
        return;
    }

    cells[idx].cov_xx = c_xx;
    cells[idx].cov_xy = c_xy;
    cells[idx].cov_yy = c_yy;
    cells[idx].det    = det;

    // Inverse of 2x2 matrix: [a b; b d]^-1 = (1/det)*[d -b; -b a]
    float inv_det = 1.0f / det;
    cells[idx].inv_xx =  c_yy * inv_det;
    cells[idx].inv_xy = -c_xy * inv_det;
    cells[idx].inv_yy =  c_xx * inv_det;
    cells[idx].valid   = 1;
}

// ---------------------------------------------------------------------------
// Kernel: Compute NDT score for source points under current pose
//   1 thread per source point. Writes per-point score to d_scores.
// ---------------------------------------------------------------------------
__global__ void compute_score_kernel(
    const NDTCell* cells,
    const float* src_x, const float* src_y,
    int n_src,
    float pose_x, float pose_y, float pose_theta,
    float cell_size, int nx, int ny,
    float* d_scores)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_src) return;

    // Transform source point by current pose
    float ct = cosf(pose_theta);
    float st = sinf(pose_theta);
    float px = ct * src_x[i] - st * src_y[i] + pose_x;
    float py = st * src_x[i] + ct * src_y[i] + pose_y;

    // Find cell
    int cx = (int)(px / cell_size);
    int cy = (int)(py / cell_size);

    if (cx < 0 || cx >= nx || cy < 0 || cy >= ny) {
        d_scores[i] = 0.0f;
        return;
    }

    int cidx = cy * nx + cx;
    if (!cells[cidx].valid) {
        d_scores[i] = 0.0f;
        return;
    }

    float dx = px - cells[cidx].mean_x;
    float dy = py - cells[cidx].mean_y;

    // Mahalanobis distance: d^T * Sigma^{-1} * d
    float maha = dx * dx * cells[cidx].inv_xx
               + 2.0f * dx * dy * cells[cidx].inv_xy
               + dy * dy * cells[cidx].inv_yy;

    d_scores[i] = expf(-0.5f * maha);
}

// ---------------------------------------------------------------------------
// Kernel: Compute gradient (Jacobian) and Hessian contributions
//   1 thread per source point. Uses atomicAdd to accumulate into
//   d_jacobian[3] and d_hessian[9] (3x3 upper triangle, row-major).
// ---------------------------------------------------------------------------
__global__ void compute_gradient_kernel(
    const NDTCell* cells,
    const float* src_x, const float* src_y,
    int n_src,
    float pose_x, float pose_y, float pose_theta,
    float cell_size, int nx, int ny,
    float* d_jacobian,   // [3]
    float* d_hessian)    // [9] row-major 3x3
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_src) return;

    float ct = cosf(pose_theta);
    float st = sinf(pose_theta);

    float sx_i = src_x[i];
    float sy_i = src_y[i];

    // Transformed point
    float px = ct * sx_i - st * sy_i + pose_x;
    float py = st * sx_i + ct * sy_i + pose_y;

    // Cell lookup
    int cx = (int)(px / cell_size);
    int cy = (int)(py / cell_size);
    if (cx < 0 || cx >= nx || cy < 0 || cy >= ny) return;

    int cidx = cy * nx + cx;
    if (!cells[cidx].valid) return;

    float mx = cells[cidx].mean_x;
    float my = cells[cidx].mean_y;
    float dx = px - mx;
    float dy = py - my;

    float ixx = cells[cidx].inv_xx;
    float ixy = cells[cidx].inv_xy;
    float iyy = cells[cidx].inv_yy;

    // Mahalanobis
    float maha = dx * dx * ixx + 2.0f * dx * dy * ixy + dy * dy * iyy;
    float score = expf(-0.5f * maha);

    if (score < 1e-10f) return;

    // Derivatives of transformed point w.r.t. pose (x, y, theta):
    // dp/dx = (1, 0), dp/dy = (0, 1)
    // dp/dtheta = (-st*sx - ct*sy, ct*sx - st*sy)
    float dpx_dtheta = -st * sx_i - ct * sy_i;
    float dpy_dtheta =  ct * sx_i - st * sy_i;

    // gradient of -0.5*maha w.r.t. (px, py):
    // g_px = -(ixx*dx + ixy*dy)
    // g_py = -(ixy*dx + iyy*dy)
    float gx = -(ixx * dx + ixy * dy);
    float gy = -(ixy * dx + iyy * dy);

    // Jacobian of score w.r.t. pose: score * (chain rule)
    // J[k] = score * (gx * dpx/dk + gy * dpy/dk)
    float j0 = score * gx;                                    // d/dx
    float j1 = score * gy;                                    // d/dy
    float j2 = score * (gx * dpx_dtheta + gy * dpy_dtheta);  // d/dtheta

    atomicAdd(&d_jacobian[0], j0);
    atomicAdd(&d_jacobian[1], j1);
    atomicAdd(&d_jacobian[2], j2);

    // Hessian (approximate): use Gauss-Newton style H ≈ J^T J / score
    // For NDT, a common approximation is H_ij = sum_points score * (ji * jj / score^2)
    // Simplified: outer product of per-point jacobian contributions
    float jv[3] = {j0, j1, j2};
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            // Gauss-Newton approximation: H ≈ sum (1/score) * j_outer
            atomicAdd(&d_hessian[r * 3 + c], jv[r] * jv[c] / (score + 1e-8f));
        }
    }
}

// ---------------------------------------------------------------------------
// CPU: Solve 3x3 linear system Hx = -J (Newton step)
//   Returns step in dp[3]. Uses Cramer's rule.
// ---------------------------------------------------------------------------
static void solve_3x3(const float H[9], const float J[3], float dp[3])
{
    // b = -J
    float b[3] = {-J[0], -J[1], -J[2]};

    // Determinant of H (row-major: H[r*3+c])
    float det = H[0] * (H[4]*H[8] - H[5]*H[7])
              - H[1] * (H[3]*H[8] - H[5]*H[6])
              + H[2] * (H[3]*H[7] - H[4]*H[6]);

    if (fabsf(det) < 1e-12f) {
        // Regularize
        float Hreg[9];
        memcpy(Hreg, H, 9 * sizeof(float));
        Hreg[0] += 1.0f;
        Hreg[4] += 1.0f;
        Hreg[8] += 1.0f;
        det = Hreg[0] * (Hreg[4]*Hreg[8] - Hreg[5]*Hreg[7])
            - Hreg[1] * (Hreg[3]*Hreg[8] - Hreg[5]*Hreg[6])
            + Hreg[2] * (Hreg[3]*Hreg[7] - Hreg[4]*Hreg[6]);
        if (fabsf(det) < 1e-12f) {
            dp[0] = dp[1] = dp[2] = 0.0f;
            return;
        }
        // Cramer's with Hreg
        float inv = 1.0f / det;
        dp[0] = inv * (b[0]*(Hreg[4]*Hreg[8]-Hreg[5]*Hreg[7]) - b[1]*(Hreg[1]*Hreg[8]-Hreg[2]*Hreg[7]) + b[2]*(Hreg[1]*Hreg[5]-Hreg[2]*Hreg[4]));
        dp[1] = inv * (-b[0]*(Hreg[3]*Hreg[8]-Hreg[5]*Hreg[6]) + b[1]*(Hreg[0]*Hreg[8]-Hreg[2]*Hreg[6]) - b[2]*(Hreg[0]*Hreg[5]-Hreg[2]*Hreg[3]));
        dp[2] = inv * (b[0]*(Hreg[3]*Hreg[7]-Hreg[4]*Hreg[6]) - b[1]*(Hreg[0]*Hreg[7]-Hreg[1]*Hreg[6]) + b[2]*(Hreg[0]*Hreg[4]-Hreg[1]*Hreg[3]));
        return;
    }

    float inv = 1.0f / det;
    dp[0] = inv * (b[0]*(H[4]*H[8]-H[5]*H[7]) - b[1]*(H[1]*H[8]-H[2]*H[7]) + b[2]*(H[1]*H[5]-H[2]*H[4]));
    dp[1] = inv * (-b[0]*(H[3]*H[8]-H[5]*H[6]) + b[1]*(H[0]*H[8]-H[2]*H[6]) - b[2]*(H[0]*H[5]-H[2]*H[3]));
    dp[2] = inv * (b[0]*(H[3]*H[7]-H[4]*H[6]) - b[1]*(H[0]*H[7]-H[1]*H[6]) + b[2]*(H[0]*H[4]-H[1]*H[3]));
}

// ---------------------------------------------------------------------------
// CPU: Generate reference map (rectangular room with internal walls)
// ---------------------------------------------------------------------------
static void generate_reference_map(vector<float>& ref_x, vector<float>& ref_y)
{
    float world_cx = (GRID_NX * CELL_SIZE) * 0.5f;  // 50
    float world_cy = (GRID_NY * CELL_SIZE) * 0.5f;  // 50

    // Outer walls of room: 40m x 30m centered in world
    float room_w = 40.0f, room_h = 30.0f;
    float rx0 = world_cx - room_w * 0.5f;
    float ry0 = world_cy - room_h * 0.5f;
    float rx1 = world_cx + room_w * 0.5f;
    float ry1 = world_cy + room_h * 0.5f;

    float spacing = 0.3f;

    // Bottom wall
    for (float x = rx0; x <= rx1; x += spacing) {
        ref_x.push_back(x); ref_y.push_back(ry0);
    }
    // Top wall
    for (float x = rx0; x <= rx1; x += spacing) {
        ref_x.push_back(x); ref_y.push_back(ry1);
    }
    // Left wall
    for (float y = ry0; y <= ry1; y += spacing) {
        ref_x.push_back(rx0); ref_y.push_back(y);
    }
    // Right wall
    for (float y = ry0; y <= ry1; y += spacing) {
        ref_x.push_back(rx1); ref_y.push_back(y);
    }

    // Internal wall 1: horizontal, from left side
    float iw1_y = world_cy - 3.0f;
    for (float x = rx0; x <= world_cx + 5.0f; x += spacing) {
        ref_x.push_back(x); ref_y.push_back(iw1_y);
    }

    // Internal wall 2: vertical, on right side
    float iw2_x = world_cx + 8.0f;
    for (float y = world_cy - 8.0f; y <= ry1; y += spacing) {
        ref_x.push_back(iw2_x); ref_y.push_back(y);
    }

    // Internal wall 3: horizontal, upper area
    float iw3_y = world_cy + 6.0f;
    for (float x = rx0 + 5.0f; x <= world_cx; x += spacing) {
        ref_x.push_back(x); ref_y.push_back(iw3_y);
    }

    printf("Reference map: %d points\n", (int)ref_x.size());
}

// ---------------------------------------------------------------------------
// CPU: Generate source scan (simulated LiDAR from a pose)
//   Simple simulation: cast rays, find nearest reference point
// ---------------------------------------------------------------------------
static void generate_source_scan(
    const vector<float>& ref_x, const vector<float>& ref_y,
    float scan_x, float scan_y, float scan_theta,
    float max_range,
    vector<float>& src_x, vector<float>& src_y)
{
    src_x.clear();
    src_y.clear();

    int n_rays = NUM_SOURCE;
    float angle_step = 2.0f * PI / n_rays;

    for (int i = 0; i < n_rays; i++) {
        float angle = scan_theta + i * angle_step;
        float best_dist = max_range;

        // Find closest reference point near this ray direction
        for (int j = 0; j < (int)ref_x.size(); j++) {
            float dx = ref_x[j] - scan_x;
            float dy = ref_y[j] - scan_y;
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < 0.5f || dist > max_range) continue;

            float pt_angle = atan2f(dy, dx);
            float angle_diff = pt_angle - angle;
            // Normalize angle diff
            while (angle_diff > PI) angle_diff -= 2.0f * PI;
            while (angle_diff < -PI) angle_diff += 2.0f * PI;

            if (fabsf(angle_diff) < angle_step * 0.5f) {
                if (dist < best_dist) {
                    best_dist = dist;
                }
            }
        }

        if (best_dist < max_range) {
            // Point in scanner-local frame
            float lx = best_dist * cosf(i * angle_step);
            float ly = best_dist * sinf(i * angle_step);
            src_x.push_back(lx);
            src_y.push_back(ly);
        }
    }
    printf("Source scan: %d points\n", (int)src_x.size());
}

// ---------------------------------------------------------------------------
// CPU: Transform points by pose for visualization
// ---------------------------------------------------------------------------
static void transform_points(
    const vector<float>& src_x, const vector<float>& src_y,
    float px, float py, float ptheta,
    vector<float>& out_x, vector<float>& out_y)
{
    int n = (int)src_x.size();
    out_x.resize(n);
    out_y.resize(n);
    float ct = cosf(ptheta);
    float st = sinf(ptheta);
    for (int i = 0; i < n; i++) {
        out_x[i] = ct * src_x[i] - st * src_y[i] + px;
        out_y[i] = st * src_x[i] + ct * src_y[i] + py;
    }
}

// ---------------------------------------------------------------------------
// Visualization helpers
// ---------------------------------------------------------------------------
static const int VIS_SIZE = 800;  // image size in pixels

// World to pixel
static inline cv::Point w2p(float wx, float wy)
{
    float world_w = GRID_NX * CELL_SIZE;
    float world_h = GRID_NY * CELL_SIZE;
    int px = (int)(wx / world_w * VIS_SIZE);
    int py = VIS_SIZE - 1 - (int)(wy / world_h * VIS_SIZE);
    return cv::Point(px, py);
}

static void draw_ndt_cells(cv::Mat& img, const vector<NDTCell>& cells)
{
    for (int cy = 0; cy < GRID_NY; cy++) {
        for (int cx = 0; cx < GRID_NX; cx++) {
            const NDTCell& c = cells[cy * GRID_NX + cx];
            if (!c.valid) continue;

            // Draw ellipse representing covariance
            cv::Point center = w2p(c.mean_x, c.mean_y);

            // Eigenvalues of 2x2 covariance for ellipse axes
            float a = c.cov_xx;
            float b = c.cov_xy;
            float d = c.cov_yy;

            float trace = a + d;
            float det = a * d - b * b;
            float disc = trace * trace * 0.25f - det;
            if (disc < 0.0f) disc = 0.0f;
            float sqrt_disc = sqrtf(disc);

            float lambda1 = trace * 0.5f + sqrt_disc;
            float lambda2 = trace * 0.5f - sqrt_disc;
            if (lambda1 < 0.0f) lambda1 = 0.01f;
            if (lambda2 < 0.0f) lambda2 = 0.01f;

            // Semi-axes in pixels (2 sigma)
            float scale = (float)VIS_SIZE / (GRID_NX * CELL_SIZE);
            float axis1 = 2.0f * sqrtf(lambda1) * scale;
            float axis2 = 2.0f * sqrtf(lambda2) * scale;

            // Angle of principal axis
            float angle = 0.5f * atan2f(2.0f * b, a - d) * 180.0f / PI;

            if (axis1 > 1.0f && axis2 > 1.0f && axis1 < 200.0f && axis2 < 200.0f) {
                cv::ellipse(img, center,
                            cv::Size((int)axis1, (int)axis2),
                            -angle, 0, 360,
                            cv::Scalar(180, 180, 0), 1);
            }
        }
    }
}

static void draw_points(cv::Mat& img, const vector<float>& px, const vector<float>& py,
                         cv::Scalar color, int radius = 2)
{
    for (int i = 0; i < (int)px.size(); i++) {
        cv::Point pt = w2p(px[i], py[i]);
        if (pt.x >= 0 && pt.x < VIS_SIZE && pt.y >= 0 && pt.y < VIS_SIZE) {
            cv::circle(img, pt, radius, color, -1);
        }
    }
}

static void draw_grid_lines(cv::Mat& img)
{
    float scale = (float)VIS_SIZE / (GRID_NX * CELL_SIZE);
    for (int i = 0; i <= GRID_NX; i++) {
        int px = (int)(i * CELL_SIZE * scale);
        cv::line(img, cv::Point(px, 0), cv::Point(px, VIS_SIZE - 1),
                 cv::Scalar(40, 40, 40), 1);
    }
    for (int j = 0; j <= GRID_NY; j++) {
        int py = VIS_SIZE - 1 - (int)(j * CELL_SIZE * scale);
        cv::line(img, cv::Point(0, py), cv::Point(VIS_SIZE - 1, py),
                 cv::Scalar(40, 40, 40), 1);
    }
}

// =========================================================================
// Main
// =========================================================================
int main()
{
    // ----- Generate reference map -----
    vector<float> ref_x, ref_y;
    generate_reference_map(ref_x, ref_y);
    int n_ref = (int)ref_x.size();

    // ----- Scan pose: slightly offset from reference center -----
    float world_cx = (GRID_NX * CELL_SIZE) * 0.5f;
    float world_cy = (GRID_NY * CELL_SIZE) * 0.5f;

    // True scan pose (where source scan was taken)
    float true_scan_x     = world_cx + 0.5f;
    float true_scan_y     = world_cy + 0.3f;
    float true_scan_theta = 0.0f;

    // Generate source scan in local frame
    vector<float> src_x, src_y;
    generate_source_scan(ref_x, ref_y, true_scan_x, true_scan_y, true_scan_theta,
                         25.0f, src_x, src_y);
    int n_src = (int)src_x.size();

    if (n_src == 0) {
        fprintf(stderr, "No source points generated!\n");
        return 1;
    }

    // ----- Initial guess for pose (offset from true pose) -----
    float init_x     = true_scan_x + 3.0f;
    float init_y     = true_scan_y + 2.0f;
    float init_theta = true_scan_theta + 0.15f;  // ~8.6 degrees

    printf("True pose:    (%.3f, %.3f, %.4f)\n", true_scan_x, true_scan_y, true_scan_theta);
    printf("Initial pose: (%.3f, %.3f, %.4f)\n", init_x, init_y, init_theta);

    // ===== GPU memory allocation =====
    float *d_ref_x, *d_ref_y;
    float *d_src_x, *d_src_y;
    NDTCell *d_cells;
    float *d_scores;
    float *d_jacobian, *d_hessian;

    CUDA_CHECK(cudaMalloc(&d_ref_x, n_ref * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref_y, n_ref * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_x, n_src * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_y, n_src * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cells, GRID_TOTAL * sizeof(NDTCell)));
    CUDA_CHECK(cudaMalloc(&d_scores, n_src * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_jacobian, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hessian, 9 * sizeof(float)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_ref_x, ref_x.data(), n_ref * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref_y, ref_y.data(), n_ref * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_x, src_x.data(), n_src * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_y, src_y.data(), n_src * sizeof(float), cudaMemcpyHostToDevice));

    // ===== Step 1: Build NDT grid =====
    CUDA_CHECK(cudaMemset(d_cells, 0, GRID_TOTAL * sizeof(NDTCell)));
    {
        int block = 256;
        int grid = (GRID_TOTAL + block - 1) / block;
        build_ndt_grid_kernel<<<grid, block>>>(
            d_cells, d_ref_x, d_ref_y, n_ref,
            CELL_SIZE, GRID_NX, GRID_NY);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy cells back for visualization
    vector<NDTCell> h_cells(GRID_TOTAL);
    CUDA_CHECK(cudaMemcpy(h_cells.data(), d_cells, GRID_TOTAL * sizeof(NDTCell),
                           cudaMemcpyDeviceToHost));

    int valid_cells = 0;
    for (int i = 0; i < GRID_TOTAL; i++) {
        if (h_cells[i].valid) valid_cells++;
    }
    printf("NDT grid: %d valid cells out of %d\n", valid_cells, GRID_TOTAL);

    // ===== Visualization setup =====
    string avi_path = "gif/ndt.avi";
    string gif_path = "gif/ndt.gif";

    cv::VideoWriter video(avi_path,
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                          10, cv::Size(VIS_SIZE, VIS_SIZE));
    if (!video.isOpened()) {
        cerr << "Failed to open video writer at " << avi_path << endl;
        return 1;
    }

    cv::namedWindow("ndt", cv::WINDOW_AUTOSIZE);

    // ===== Step 2-3: Newton optimization =====
    float pose_x = init_x;
    float pose_y = init_y;
    float pose_theta = init_theta;

    int block_src = 256;
    int grid_src = (n_src + block_src - 1) / block_src;

    vector<float> h_scores(n_src);
    float h_jacobian[3];
    float h_hessian[9];

    for (int iter = 0; iter <= MAX_ITER; iter++) {
        // ----- Visualize current state -----
        cv::Mat img(VIS_SIZE, VIS_SIZE, CV_8UC3, cv::Scalar(20, 20, 20));

        // Grid lines
        draw_grid_lines(img);

        // NDT cell ellipses
        draw_ndt_cells(img, h_cells);

        // Reference points (blue)
        draw_points(img, ref_x, ref_y, cv::Scalar(255, 150, 50), 2);

        // Source points at initial guess (red)
        vector<float> init_tx, init_ty;
        transform_points(src_x, src_y, init_x, init_y, init_theta, init_tx, init_ty);
        draw_points(img, init_tx, init_ty, cv::Scalar(50, 50, 255), 2);

        // Source points at current pose (green)
        vector<float> cur_tx, cur_ty;
        transform_points(src_x, src_y, pose_x, pose_y, pose_theta, cur_tx, cur_ty);
        draw_points(img, cur_tx, cur_ty, cv::Scalar(50, 255, 50), 3);

        // Info text
        char buf[256];
        snprintf(buf, sizeof(buf), "Iteration: %d", iter);
        cv::putText(img, buf, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
        snprintf(buf, sizeof(buf), "Pose: (%.3f, %.3f, %.4f)", pose_x, pose_y, pose_theta);
        cv::putText(img, buf, cv::Point(10, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

        // Compute current score for display
        compute_score_kernel<<<grid_src, block_src>>>(
            d_cells, d_src_x, d_src_y, n_src,
            pose_x, pose_y, pose_theta,
            CELL_SIZE, GRID_NX, GRID_NY,
            d_scores);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_scores.data(), d_scores, n_src * sizeof(float),
                               cudaMemcpyDeviceToHost));

        float total_score = 0.0f;
        for (int i = 0; i < n_src; i++) total_score += h_scores[i];
        snprintf(buf, sizeof(buf), "Score: %.4f", total_score);
        cv::putText(img, buf, cv::Point(10, 75),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

        // Legend
        cv::putText(img, "Blue: reference", cv::Point(10, VIS_SIZE - 55),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 150, 50), 1);
        cv::putText(img, "Red: initial source", cv::Point(10, VIS_SIZE - 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50, 50, 255), 1);
        cv::putText(img, "Green: aligned source", cv::Point(10, VIS_SIZE - 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50, 255, 50), 1);

        video.write(img);
        cv::imshow("ndt", img);
        int key = cv::waitKey(200);
        if (key == 27) break;

        if (iter == MAX_ITER) break;

        // ----- Compute gradient and Hessian -----
        CUDA_CHECK(cudaMemset(d_jacobian, 0, 3 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_hessian, 0, 9 * sizeof(float)));

        compute_gradient_kernel<<<grid_src, block_src>>>(
            d_cells, d_src_x, d_src_y, n_src,
            pose_x, pose_y, pose_theta,
            CELL_SIZE, GRID_NX, GRID_NY,
            d_jacobian, d_hessian);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_jacobian, d_jacobian, 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_hessian, d_hessian, 9 * sizeof(float), cudaMemcpyDeviceToHost));

        // ----- Newton step (CPU) -----
        float dp[3];
        solve_3x3(h_hessian, h_jacobian, dp);

        // Limit step size for stability
        float max_step_trans = 2.0f;
        float max_step_rot   = 0.2f;
        if (fabsf(dp[0]) > max_step_trans) dp[0] = copysignf(max_step_trans, dp[0]);
        if (fabsf(dp[1]) > max_step_trans) dp[1] = copysignf(max_step_trans, dp[1]);
        if (fabsf(dp[2]) > max_step_rot)   dp[2] = copysignf(max_step_rot, dp[2]);

        pose_x     += dp[0];
        pose_y     += dp[1];
        pose_theta += dp[2];

        // Normalize theta
        while (pose_theta > PI) pose_theta -= 2.0f * PI;
        while (pose_theta < -PI) pose_theta += 2.0f * PI;

        float step_norm = sqrtf(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2]);
        printf("Iter %2d: pose=(%.4f, %.4f, %.4f) score=%.4f step=%.6f\n",
               iter, pose_x, pose_y, pose_theta, total_score, step_norm);

        if (step_norm < CONV_THRESH) {
            printf("Converged at iteration %d\n", iter);
            // Draw final frame
            cv::Mat final_img(VIS_SIZE, VIS_SIZE, CV_8UC3, cv::Scalar(20, 20, 20));
            draw_grid_lines(final_img);
            draw_ndt_cells(final_img, h_cells);
            draw_points(final_img, ref_x, ref_y, cv::Scalar(255, 150, 50), 2);
            draw_points(final_img, init_tx, init_ty, cv::Scalar(50, 50, 255), 2);
            vector<float> final_tx, final_ty;
            transform_points(src_x, src_y, pose_x, pose_y, pose_theta, final_tx, final_ty);
            draw_points(final_img, final_tx, final_ty, cv::Scalar(50, 255, 50), 3);
            snprintf(buf, sizeof(buf), "CONVERGED at iter %d", iter);
            cv::putText(final_img, buf, cv::Point(10, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
            snprintf(buf, sizeof(buf), "Final pose: (%.3f, %.3f, %.4f)", pose_x, pose_y, pose_theta);
            cv::putText(final_img, buf, cv::Point(10, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
            cv::putText(final_img, "Blue: reference", cv::Point(10, VIS_SIZE - 55),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 150, 50), 1);
            cv::putText(final_img, "Red: initial source", cv::Point(10, VIS_SIZE - 35),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50, 50, 255), 1);
            cv::putText(final_img, "Green: aligned source", cv::Point(10, VIS_SIZE - 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50, 255, 50), 1);
            // Write several copies so it lingers in the gif
            for (int k = 0; k < 15; k++) video.write(final_img);
            cv::imshow("ndt", final_img);
            cv::waitKey(500);
            break;
        }
    }

    // ----- Finalize video -----
    video.release();
    printf("Video saved to %s\n", avi_path.c_str());

    // Convert to GIF
    string cmd = "ffmpeg -y -i " + avi_path
                 + " -vf \"fps=10,scale=500:-1:flags=lanczos\" "
                 + gif_path + " 2>/dev/null";
    int ret = system(cmd.c_str());
    if (ret == 0) {
        printf("GIF saved to %s\n", gif_path.c_str());
    } else {
        printf("ffmpeg conversion failed (ffmpeg may not be installed)\n");
    }

    // ----- Report results -----
    printf("\n========== NDT Scan Matching Results ==========\n");
    printf("True pose:    (%.4f, %.4f, %.4f)\n", true_scan_x, true_scan_y, true_scan_theta);
    printf("Initial pose: (%.4f, %.4f, %.4f)\n", init_x, init_y, init_theta);
    printf("Final pose:   (%.4f, %.4f, %.4f)\n", pose_x, pose_y, pose_theta);
    printf("Error: dx=%.4f dy=%.4f dtheta=%.4f\n",
           fabsf(pose_x - true_scan_x),
           fabsf(pose_y - true_scan_y),
           fabsf(pose_theta - true_scan_theta));

    // Show final image until key press
    printf("Press any key to exit...\n");
    cv::waitKey(0);

    // ===== Cleanup =====
    CUDA_CHECK(cudaFree(d_ref_x));
    CUDA_CHECK(cudaFree(d_ref_y));
    CUDA_CHECK(cudaFree(d_src_x));
    CUDA_CHECK(cudaFree(d_src_y));
    CUDA_CHECK(cudaFree(d_cells));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_jacobian));
    CUDA_CHECK(cudaFree(d_hessian));

    return 0;
}
