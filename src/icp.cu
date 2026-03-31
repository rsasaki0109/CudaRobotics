/*************************************************************************
    ICP (Iterative Closest Point) - CUDA-parallelized
    Aligns two 2D point clouds by iteratively finding nearest neighbors,
    computing optimal rigid transformation (R, t), and applying it.

    CUDA kernels:
      - find_nearest_kernel:   1 thread per source point, brute-force NN
      - compute_centroid_kernel: parallel reduction for centroid
      - compute_W_kernel:      outer product accumulation with atomicAdd
      - apply_transform_kernel: apply R,t to all source points
    SVD of 3x3 W matrix done on host via Eigen.
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// =====================================================================
// Parameters
// =====================================================================
#define N_POINTS   500
#define MAX_ITER   50
#define CONV_THRESH 0.001f
#define PI 3.141592653f

// =====================================================================
// CUDA Kernels
// =====================================================================

// Each thread finds the nearest target point for one source point
__global__ void find_nearest_kernel(
    const float* src_x, const float* src_y,   // [N_src]
    const float* tgt_x, const float* tgt_y,   // [N_tgt]
    int* matches,                               // [N_src] index of nearest target
    float* distances,                           // [N_src] distance to nearest
    int n_src, int n_tgt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_src) return;

    float sx = src_x[i];
    float sy = src_y[i];
    float best_dist = 1e30f;
    int best_idx = 0;

    for (int j = 0; j < n_tgt; j++) {
        float dx = sx - tgt_x[j];
        float dy = sy - tgt_y[j];
        float d = dx * dx + dy * dy;
        if (d < best_dist) {
            best_dist = d;
            best_idx = j;
        }
    }

    matches[i] = best_idx;
    distances[i] = sqrtf(best_dist);
}

// Parallel reduction to compute centroid of matched source and target points
// Outputs: out[0]=sum_sx, out[1]=sum_sy, out[2]=sum_tx, out[3]=sum_ty
__global__ void compute_centroid_kernel(
    const float* src_x, const float* src_y,
    const float* tgt_x, const float* tgt_y,
    const int* matches, int n,
    float* out)  // [4] = {sum_sx, sum_sy, sum_tx, sum_ty}
{
    extern __shared__ float sdata[];
    // sdata layout: [0..blockDim-1] = sx, [blockDim..2*blockDim-1] = sy,
    //               [2*blockDim..3*blockDim-1] = tx, [3*blockDim..4*blockDim-1] = ty
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sx = 0.0f, sy = 0.0f, tx = 0.0f, ty = 0.0f;
    if (i < n) {
        sx = src_x[i];
        sy = src_y[i];
        int mi = matches[i];
        tx = tgt_x[mi];
        ty = tgt_y[mi];
    }

    sdata[tid] = sx;
    sdata[tid + blockDim.x] = sy;
    sdata[tid + 2 * blockDim.x] = tx;
    sdata[tid + 3 * blockDim.x] = ty;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + blockDim.x + s];
            sdata[tid + 2 * blockDim.x] += sdata[tid + 2 * blockDim.x + s];
            sdata[tid + 3 * blockDim.x] += sdata[tid + 3 * blockDim.x + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&out[0], sdata[0]);
        atomicAdd(&out[1], sdata[blockDim.x]);
        atomicAdd(&out[2], sdata[2 * blockDim.x]);
        atomicAdd(&out[3], sdata[3 * blockDim.x]);
    }
}

// Each thread computes outer product contribution for one point pair
// W = sum_i (src_i - centroid_src) * (tgt_i - centroid_tgt)^T
// Uses atomicAdd into 2x2 W matrix (stored as 4 floats)
__global__ void compute_W_kernel(
    const float* src_x, const float* src_y,
    const float* tgt_x, const float* tgt_y,
    const int* matches, int n,
    float cx_s, float cy_s, float cx_t, float cy_t,
    float* W)  // [4] = {w00, w01, w10, w11}
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sx = src_x[i] - cx_s;
    float sy = src_y[i] - cy_s;
    int mi = matches[i];
    float tx = tgt_x[mi] - cx_t;
    float ty = tgt_y[mi] - cy_t;

    atomicAdd(&W[0], sx * tx);  // W(0,0)
    atomicAdd(&W[1], sx * ty);  // W(0,1)
    atomicAdd(&W[2], sy * tx);  // W(1,0)
    atomicAdd(&W[3], sy * ty);  // W(1,1)
}

// Apply rotation R and translation t to all source points
__global__ void apply_transform_kernel(
    float* src_x, float* src_y, int n,
    float r00, float r01, float r10, float r11,
    float tx, float ty)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = src_x[i];
    float y = src_y[i];
    src_x[i] = r00 * x + r01 * y + tx;
    src_y[i] = r10 * x + r11 * y + ty;
}

// Parallel mean-distance reduction
__global__ void reduce_mean_distance_kernel(
    const float* distances, int n, float* out)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? distances[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out, sdata[0]);
}

// =====================================================================
// Generate 2D point clouds
// =====================================================================

void generate_L_shape(std::vector<float>& px, std::vector<float>& py, int n)
{
    // L-shape: vertical bar + horizontal bar
    px.resize(n);
    py.resize(n);
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        float t = (float)i / half;
        px[i] = 0.0f;
        py[i] = t * 10.0f;  // vertical segment 0..10
    }
    for (int i = half; i < n; i++) {
        float t = (float)(i - half) / (n - half);
        px[i] = t * 8.0f;   // horizontal segment 0..8
        py[i] = 0.0f;
    }
}

void apply_transform_cpu(
    const std::vector<float>& in_x, const std::vector<float>& in_y,
    std::vector<float>& out_x, std::vector<float>& out_y,
    float angle, float tx, float ty,
    std::mt19937& gen, float noise_std)
{
    int n = (int)in_x.size();
    out_x.resize(n);
    out_y.resize(n);
    std::normal_distribution<float> noise(0.0f, noise_std);
    float c = cosf(angle);
    float s = sinf(angle);
    for (int i = 0; i < n; i++) {
        out_x[i] = c * in_x[i] - s * in_y[i] + tx + noise(gen);
        out_y[i] = s * in_x[i] + c * in_y[i] + ty + noise(gen);
    }
}

// =====================================================================
// Visualization helpers
// =====================================================================

struct VisParams {
    int W, H;
    float offset_x, offset_y, scale;
};

cv::Point to_pixel(float x, float y, const VisParams& vp)
{
    int px = (int)((x - vp.offset_x) * vp.scale);
    int py = vp.H - (int)((y - vp.offset_y) * vp.scale);
    return cv::Point(px, py);
}

void draw_points(cv::Mat& img, const std::vector<float>& px, const std::vector<float>& py,
                 cv::Scalar color, int radius, const VisParams& vp)
{
    for (size_t i = 0; i < px.size(); i++) {
        cv::circle(img, to_pixel(px[i], py[i], vp), radius, color, -1);
    }
}

// =====================================================================
// main
// =====================================================================

int main()
{
    std::cout << "ICP (Iterative Closest Point) - CUDA" << std::endl;

    // --- Generate target point cloud (L-shape) ---
    std::vector<float> tgt_x, tgt_y;
    generate_L_shape(tgt_x, tgt_y, N_POINTS);

    // --- Generate source: rotated 30deg + translated (5,3) + noise ---
    std::mt19937 gen(42);
    float angle = 30.0f * PI / 180.0f;
    float trans_x = 5.0f, trans_y = 3.0f;
    float noise_std = 0.1f;

    std::vector<float> src_x, src_y;
    apply_transform_cpu(tgt_x, tgt_y, src_x, src_y, angle, trans_x, trans_y, gen, noise_std);

    // Keep initial source for visualization
    std::vector<float> src_init_x = src_x;
    std::vector<float> src_init_y = src_y;

    int n_src = (int)src_x.size();
    int n_tgt = (int)tgt_x.size();

    // --- Visualization setup ---
    VisParams vp;
    vp.W = 800; vp.H = 800;
    vp.offset_x = -5.0f; vp.offset_y = -5.0f; vp.scale = 50.0f;

    cv::VideoWriter video(
        "gif/icp.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 10, cv::Size(vp.W, vp.H));

    // --- CUDA setup ---
    float *d_src_x, *d_src_y, *d_tgt_x, *d_tgt_y;
    int *d_matches;
    float *d_distances;
    float *d_centroid;  // [4]
    float *d_W;         // [4] for 2x2 matrix
    float *d_mean_dist; // [1]

    CUDA_CHECK(cudaMalloc(&d_src_x, n_src * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_y, n_src * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt_x, n_tgt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt_y, n_tgt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_matches, n_src * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_distances, n_src * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroid, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean_dist, sizeof(float)));

    // Upload target (constant throughout)
    CUDA_CHECK(cudaMemcpy(d_tgt_x, tgt_x.data(), n_tgt * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt_y, tgt_y.data(), n_tgt * sizeof(float), cudaMemcpyHostToDevice));

    // Upload initial source
    CUDA_CHECK(cudaMemcpy(d_src_x, src_x.data(), n_src * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_y, src_y.data(), n_src * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks_src = (n_src + threads - 1) / threads;

    std::vector<float> mean_distances;
    float prev_mean_dist = 1e10f;

    std::cout << "Running ICP (N_src=" << n_src << ", N_tgt=" << n_tgt
              << ", max_iter=" << MAX_ITER << ")" << std::endl;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // 1. Find nearest neighbors
        find_nearest_kernel<<<blocks_src, threads>>>(
            d_src_x, d_src_y, d_tgt_x, d_tgt_y,
            d_matches, d_distances, n_src, n_tgt);

        // 2. Compute mean distance
        CUDA_CHECK(cudaMemset(d_mean_dist, 0, sizeof(float)));
        reduce_mean_distance_kernel<<<blocks_src, threads, threads * sizeof(float)>>>(
            d_distances, n_src, d_mean_dist);

        float h_mean_dist;
        CUDA_CHECK(cudaMemcpy(&h_mean_dist, d_mean_dist, sizeof(float), cudaMemcpyDeviceToHost));
        h_mean_dist /= n_src;
        mean_distances.push_back(h_mean_dist);

        printf("  iter %2d: mean_dist = %.6f\n", iter, h_mean_dist);

        // Check convergence
        if (fabsf(prev_mean_dist - h_mean_dist) < CONV_THRESH) {
            printf("  Converged at iteration %d\n", iter);
            // Write final frame a few times for gif visibility
            CUDA_CHECK(cudaMemcpy(src_x.data(), d_src_x, n_src * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(src_y.data(), d_src_y, n_src * sizeof(float), cudaMemcpyDeviceToHost));
            cv::Mat frame(vp.H, vp.W, CV_8UC3, cv::Scalar(255, 255, 255));
            draw_points(frame, tgt_x, tgt_y, cv::Scalar(255, 0, 0), 3, vp);     // target: blue
            draw_points(frame, src_init_x, src_init_y, cv::Scalar(0, 0, 255), 2, vp); // initial: red
            draw_points(frame, src_x, src_y, cv::Scalar(0, 180, 0), 3, vp);      // aligned: green

            char buf[128];
            snprintf(buf, sizeof(buf), "Iter %d  mean_dist=%.4f  CONVERGED", iter, h_mean_dist);
            cv::putText(frame, buf, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
            cv::putText(frame, "Blue=Target  Red=Initial  Green=Aligned", cv::Point(10, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(80, 80, 80), 1);

            for (int k = 0; k < 20; k++) video.write(frame);
            break;
        }
        prev_mean_dist = h_mean_dist;

        // 3. Compute centroids
        CUDA_CHECK(cudaMemset(d_centroid, 0, 4 * sizeof(float)));
        compute_centroid_kernel<<<blocks_src, threads, 4 * threads * sizeof(float)>>>(
            d_src_x, d_src_y, d_tgt_x, d_tgt_y,
            d_matches, n_src, d_centroid);

        float h_centroid[4];
        CUDA_CHECK(cudaMemcpy(h_centroid, d_centroid, 4 * sizeof(float), cudaMemcpyDeviceToHost));
        float cx_s = h_centroid[0] / n_src;
        float cy_s = h_centroid[1] / n_src;
        float cx_t = h_centroid[2] / n_src;
        float cy_t = h_centroid[3] / n_src;

        // 4. Compute W matrix (cross-covariance)
        CUDA_CHECK(cudaMemset(d_W, 0, 4 * sizeof(float)));
        compute_W_kernel<<<blocks_src, threads>>>(
            d_src_x, d_src_y, d_tgt_x, d_tgt_y,
            d_matches, n_src,
            cx_s, cy_s, cx_t, cy_t, d_W);

        float h_W[4];
        CUDA_CHECK(cudaMemcpy(h_W, d_W, 4 * sizeof(float), cudaMemcpyDeviceToHost));

        // 5. SVD on host (2x2 matrix)
        Eigen::Matrix2f W_mat;
        W_mat << h_W[0], h_W[1],
                 h_W[2], h_W[3];

        Eigen::JacobiSVD<Eigen::Matrix2f> svd(W_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix2f U = svd.matrixU();
        Eigen::Matrix2f V = svd.matrixV();

        Eigen::Matrix2f R = V * U.transpose();
        // Ensure proper rotation (det = +1)
        if (R.determinant() < 0) {
            V.col(1) *= -1.0f;
            R = V * U.transpose();
        }

        Eigen::Vector2f c_src(cx_s, cy_s);
        Eigen::Vector2f c_tgt(cx_t, cy_t);
        Eigen::Vector2f t = c_tgt - R * c_src;

        // 6. Apply transform on GPU
        apply_transform_kernel<<<blocks_src, threads>>>(
            d_src_x, d_src_y, n_src,
            R(0, 0), R(0, 1), R(1, 0), R(1, 1),
            t(0), t(1));
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Visualization ---
        CUDA_CHECK(cudaMemcpy(src_x.data(), d_src_x, n_src * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(src_y.data(), d_src_y, n_src * sizeof(float), cudaMemcpyDeviceToHost));

        cv::Mat frame(vp.H, vp.W, CV_8UC3, cv::Scalar(255, 255, 255));
        draw_points(frame, tgt_x, tgt_y, cv::Scalar(255, 0, 0), 3, vp);          // target: blue
        draw_points(frame, src_init_x, src_init_y, cv::Scalar(0, 0, 255), 2, vp); // initial: red
        draw_points(frame, src_x, src_y, cv::Scalar(0, 180, 0), 3, vp);           // current: green

        char buf[128];
        snprintf(buf, sizeof(buf), "Iter %d  mean_dist=%.4f", iter, h_mean_dist);
        cv::putText(frame, buf, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        cv::putText(frame, "Blue=Target  Red=Initial  Green=Aligned", cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(80, 80, 80), 1);

        // Draw convergence plot in bottom-right corner
        if (mean_distances.size() > 1) {
            int plot_w = 200, plot_h = 120;
            int plot_x0 = vp.W - plot_w - 10;
            int plot_y0 = vp.H - plot_h - 10;
            cv::rectangle(frame, cv::Point(plot_x0, plot_y0),
                          cv::Point(plot_x0 + plot_w, plot_y0 + plot_h),
                          cv::Scalar(240, 240, 240), -1);
            cv::rectangle(frame, cv::Point(plot_x0, plot_y0),
                          cv::Point(plot_x0 + plot_w, plot_y0 + plot_h),
                          cv::Scalar(0, 0, 0), 1);
            cv::putText(frame, "Mean Dist", cv::Point(plot_x0 + 5, plot_y0 + 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

            float max_d = *std::max_element(mean_distances.begin(), mean_distances.end());
            if (max_d < 0.01f) max_d = 0.01f;
            for (size_t k = 1; k < mean_distances.size(); k++) {
                int x1 = plot_x0 + (int)((k - 1) * plot_w / MAX_ITER);
                int x2 = plot_x0 + (int)(k * plot_w / MAX_ITER);
                int y1 = plot_y0 + plot_h - (int)(mean_distances[k - 1] / max_d * (plot_h - 20));
                int y2 = plot_y0 + plot_h - (int)(mean_distances[k] / max_d * (plot_h - 20));
                cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2),
                         cv::Scalar(0, 0, 200), 2);
            }
        }

        video.write(frame);
    }

    video.release();
    std::cout << "Video saved to gif/icp.avi" << std::endl;

    // Convert to gif
    system("ffmpeg -y -i gif/icp.avi "
           "-vf 'fps=10,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/icp.gif 2>/dev/null");
    std::cout << "GIF saved to gif/icp.gif" << std::endl;

    // Cleanup
    cudaFree(d_src_x);
    cudaFree(d_src_y);
    cudaFree(d_tgt_x);
    cudaFree(d_tgt_y);
    cudaFree(d_matches);
    cudaFree(d_distances);
    cudaFree(d_centroid);
    cudaFree(d_W);
    cudaFree(d_mean_dist);

    return 0;
}
