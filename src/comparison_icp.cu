/*************************************************************************
    ICP: CPU vs CUDA side-by-side comparison GIF generator
    Left panel: CPU (sequential NN), Right panel: CUDA (parallel NN)
    Compares nearest-neighbor search performance.
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <algorithm>
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

#define N_POINTS   500
#define MAX_ITER   50
#define CONV_THRESH 0.001f
#define PI 3.141592653f

// =====================================================================
// CUDA Kernels (same as icp.cu)
// =====================================================================

__global__ void find_nearest_kernel(
    const float* src_x, const float* src_y,
    const float* tgt_x, const float* tgt_y,
    int* matches, float* distances,
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

__global__ void compute_centroid_kernel(
    const float* src_x, const float* src_y,
    const float* tgt_x, const float* tgt_y,
    const int* matches, int n,
    float* out)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sx = 0, sy = 0, tx = 0, ty = 0;
    if (i < n) {
        sx = src_x[i]; sy = src_y[i];
        int mi = matches[i];
        tx = tgt_x[mi]; ty = tgt_y[mi];
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

__global__ void compute_W_kernel(
    const float* src_x, const float* src_y,
    const float* tgt_x, const float* tgt_y,
    const int* matches, int n,
    float cx_s, float cy_s, float cx_t, float cy_t,
    float* W)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sx = src_x[i] - cx_s;
    float sy = src_y[i] - cy_s;
    int mi = matches[i];
    float tx = tgt_x[mi] - cx_t;
    float ty = tgt_y[mi] - cy_t;

    atomicAdd(&W[0], sx * tx);
    atomicAdd(&W[1], sx * ty);
    atomicAdd(&W[2], sy * tx);
    atomicAdd(&W[3], sy * ty);
}

__global__ void apply_transform_kernel(
    float* src_x, float* src_y, int n,
    float r00, float r01, float r10, float r11,
    float tx, float ty)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = src_x[i], y = src_y[i];
    src_x[i] = r00 * x + r01 * y + tx;
    src_y[i] = r10 * x + r11 * y + ty;
}

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
// CPU ICP implementation
// =====================================================================

struct CpuIcpResult {
    std::vector<float> src_x, src_y;
    float mean_dist;
    double elapsed_ms;
    bool converged;
    int iters_used;
    std::vector<float> mean_distances;
};

CpuIcpResult run_cpu_icp(
    const std::vector<float>& tgt_x, const std::vector<float>& tgt_y,
    std::vector<float> src_x, std::vector<float> src_y,
    int max_iter)
{
    CpuIcpResult result;
    int n_src = (int)src_x.size();
    int n_tgt = (int)tgt_x.size();
    std::vector<int> matches(n_src);
    std::vector<float> distances(n_src);

    float prev_mean = 1e10f;
    result.converged = false;
    result.iters_used = max_iter;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < max_iter; iter++) {
        // 1. CPU sequential nearest neighbor
        for (int i = 0; i < n_src; i++) {
            float best_d = 1e30f;
            int best_j = 0;
            for (int j = 0; j < n_tgt; j++) {
                float dx = src_x[i] - tgt_x[j];
                float dy = src_y[i] - tgt_y[j];
                float d = dx * dx + dy * dy;
                if (d < best_d) { best_d = d; best_j = j; }
            }
            matches[i] = best_j;
            distances[i] = sqrtf(best_d);
        }

        // Mean distance
        float mean_d = 0;
        for (int i = 0; i < n_src; i++) mean_d += distances[i];
        mean_d /= n_src;
        result.mean_distances.push_back(mean_d);

        if (fabsf(prev_mean - mean_d) < CONV_THRESH) {
            result.converged = true;
            result.iters_used = iter + 1;
            break;
        }
        prev_mean = mean_d;

        // 2. Compute centroids
        float cx_s = 0, cy_s = 0, cx_t = 0, cy_t = 0;
        for (int i = 0; i < n_src; i++) {
            cx_s += src_x[i]; cy_s += src_y[i];
            cx_t += tgt_x[matches[i]]; cy_t += tgt_y[matches[i]];
        }
        cx_s /= n_src; cy_s /= n_src;
        cx_t /= n_src; cy_t /= n_src;

        // 3. Compute W
        Eigen::Matrix2f W = Eigen::Matrix2f::Zero();
        for (int i = 0; i < n_src; i++) {
            float sx = src_x[i] - cx_s;
            float sy = src_y[i] - cy_s;
            float tx = tgt_x[matches[i]] - cx_t;
            float ty = tgt_y[matches[i]] - cy_t;
            W(0, 0) += sx * tx; W(0, 1) += sx * ty;
            W(1, 0) += sy * tx; W(1, 1) += sy * ty;
        }

        // 4. SVD
        Eigen::JacobiSVD<Eigen::Matrix2f> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix2f U = svd.matrixU();
        Eigen::Matrix2f V = svd.matrixV();
        Eigen::Matrix2f R = V * U.transpose();
        if (R.determinant() < 0) { V.col(1) *= -1; R = V * U.transpose(); }

        Eigen::Vector2f c_s(cx_s, cy_s), c_t(cx_t, cy_t);
        Eigen::Vector2f t = c_t - R * c_s;

        // 5. Apply transform
        for (int i = 0; i < n_src; i++) {
            float x = src_x[i], y = src_y[i];
            src_x[i] = R(0, 0) * x + R(0, 1) * y + t(0);
            src_y[i] = R(1, 0) * x + R(1, 1) * y + t(1);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    result.src_x = src_x;
    result.src_y = src_y;
    result.mean_dist = result.mean_distances.back();
    return result;
}

// =====================================================================
// Point cloud generation
// =====================================================================

void generate_L_shape(std::vector<float>& px, std::vector<float>& py, int n)
{
    px.resize(n); py.resize(n);
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        float t = (float)i / half;
        px[i] = 0.0f; py[i] = t * 10.0f;
    }
    for (int i = half; i < n; i++) {
        float t = (float)(i - half) / (n - half);
        px[i] = t * 8.0f; py[i] = 0.0f;
    }
}

void transform_cloud(
    const std::vector<float>& in_x, const std::vector<float>& in_y,
    std::vector<float>& out_x, std::vector<float>& out_y,
    float angle, float tx, float ty,
    std::mt19937& gen, float noise_std)
{
    int n = (int)in_x.size();
    out_x.resize(n); out_y.resize(n);
    std::normal_distribution<float> noise(0.0f, noise_std);
    float c = cosf(angle), s = sinf(angle);
    for (int i = 0; i < n; i++) {
        out_x[i] = c * in_x[i] - s * in_y[i] + tx + noise(gen);
        out_y[i] = s * in_x[i] + c * in_y[i] + ty + noise(gen);
    }
}

// =====================================================================
// Visualization
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
    for (size_t i = 0; i < px.size(); i++)
        cv::circle(img, to_pixel(px[i], py[i], vp), radius, color, -1);
}

void draw_icp_scene(cv::Mat& img,
                    const std::vector<float>& tgt_x, const std::vector<float>& tgt_y,
                    const std::vector<float>& init_x, const std::vector<float>& init_y,
                    const std::vector<float>& cur_x, const std::vector<float>& cur_y,
                    const VisParams& vp,
                    const char* title, int iter, float mean_dist, double time_ms,
                    const std::vector<float>& mean_dists)
{
    draw_points(img, tgt_x, tgt_y, cv::Scalar(255, 0, 0), 3, vp);          // blue: target
    draw_points(img, init_x, init_y, cv::Scalar(0, 0, 255), 2, vp);        // red: initial
    draw_points(img, cur_x, cur_y, cv::Scalar(0, 180, 0), 3, vp);          // green: current

    char buf[256];
    snprintf(buf, sizeof(buf), "%s", title);
    cv::putText(img, buf, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    snprintf(buf, sizeof(buf), "Iter %d  dist=%.4f", iter, mean_dist);
    cv::putText(img, buf, cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

    snprintf(buf, sizeof(buf), "Time: %.2f ms", time_ms);
    cv::putText(img, buf, cv::Point(10, 85),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 200), 1);

    cv::putText(img, "Blue=Target Red=Initial Green=Aligned", cv::Point(10, 110),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(80, 80, 80), 1);

    // Convergence plot
    if (mean_dists.size() > 1) {
        int pw = 180, ph = 100;
        int px0 = vp.W - pw - 10, py0 = vp.H - ph - 10;
        cv::rectangle(img, cv::Point(px0, py0), cv::Point(px0 + pw, py0 + ph),
                      cv::Scalar(240, 240, 240), -1);
        cv::rectangle(img, cv::Point(px0, py0), cv::Point(px0 + pw, py0 + ph),
                      cv::Scalar(0, 0, 0), 1);
        cv::putText(img, "Mean Dist", cv::Point(px0 + 5, py0 + 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 0, 0), 1);

        float max_d = *std::max_element(mean_dists.begin(), mean_dists.end());
        if (max_d < 0.01f) max_d = 0.01f;
        for (size_t k = 1; k < mean_dists.size(); k++) {
            int x1 = px0 + (int)((k - 1) * pw / MAX_ITER);
            int x2 = px0 + (int)(k * pw / MAX_ITER);
            int y1 = py0 + ph - (int)(mean_dists[k - 1] / max_d * (ph - 20));
            int y2 = py0 + ph - (int)(mean_dists[k] / max_d * (ph - 20));
            cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 200), 2);
        }
    }
}

// =====================================================================
// main
// =====================================================================

int main()
{
    std::cout << "ICP Comparison: CPU sequential NN vs CUDA parallel NN" << std::endl;

    // Generate point clouds
    std::vector<float> tgt_x, tgt_y;
    generate_L_shape(tgt_x, tgt_y, N_POINTS);

    std::mt19937 gen(42);
    float angle = 30.0f * PI / 180.0f;
    float trans_x = 5.0f, trans_y = 3.0f;
    float noise_std = 0.1f;

    std::vector<float> src_x, src_y;
    transform_cloud(tgt_x, tgt_y, src_x, src_y, angle, trans_x, trans_y, gen, noise_std);

    std::vector<float> src_init_x = src_x;
    std::vector<float> src_init_y = src_y;

    int n_src = (int)src_x.size();
    int n_tgt = (int)tgt_x.size();

    // ========================= CPU ICP =========================
    std::cout << "Running CPU ICP..." << std::endl;
    CpuIcpResult cpu_result = run_cpu_icp(tgt_x, tgt_y, src_x, src_y, MAX_ITER);
    printf("  CPU: %d iters, %.2f ms total, final dist=%.6f\n",
           cpu_result.iters_used, cpu_result.elapsed_ms, cpu_result.mean_dist);

    // ========================= CUDA ICP (per-iteration timing) =========================
    std::cout << "Running CUDA ICP..." << std::endl;

    // Copy source back to initial state for CUDA run
    std::vector<float> cuda_src_x = src_init_x;
    std::vector<float> cuda_src_y = src_init_y;

    float *d_src_x, *d_src_y, *d_tgt_x, *d_tgt_y;
    int *d_matches;
    float *d_distances, *d_centroid, *d_W, *d_mean_dist;

    CUDA_CHECK(cudaMalloc(&d_src_x, n_src * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_y, n_src * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt_x, n_tgt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt_y, n_tgt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_matches, n_src * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_distances, n_src * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroid, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean_dist, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_tgt_x, tgt_x.data(), n_tgt * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt_y, tgt_y.data(), n_tgt * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_x, cuda_src_x.data(), n_src * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_y, cuda_src_y.data(), n_src * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks_src = (n_src + threads - 1) / threads;

    std::vector<float> cuda_mean_dists;
    float cuda_total_ms = 0;
    float prev_mean = 1e10f;
    int cuda_iters = MAX_ITER;
    bool cuda_converged = false;

    // Per-iteration data for visualization
    struct IterSnapshot {
        std::vector<float> src_x, src_y;
        float mean_dist;
        double cuda_ms;
        double cpu_ms;
        int iter;
    };
    std::vector<IterSnapshot> snapshots;

    // CPU per-iteration timing: re-run CPU but capture per-iteration data
    std::vector<float> cpu_iter_src_x = src_init_x;
    std::vector<float> cpu_iter_src_y = src_init_y;
    std::vector<int> cpu_matches(n_src);
    std::vector<float> cpu_distances(n_src);
    std::vector<float> cpu_mean_dists;
    float cpu_prev_mean = 1e10f;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        IterSnapshot snap;
        snap.iter = iter;

        // === CPU iteration ===
        auto cpu_t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_src; i++) {
            float best_d = 1e30f; int best_j = 0;
            for (int j = 0; j < n_tgt; j++) {
                float dx = cpu_iter_src_x[i] - tgt_x[j];
                float dy = cpu_iter_src_y[i] - tgt_y[j];
                float d = dx * dx + dy * dy;
                if (d < best_d) { best_d = d; best_j = j; }
            }
            cpu_matches[i] = best_j;
            cpu_distances[i] = sqrtf(best_d);
        }
        float cpu_mean_d = 0;
        for (int i = 0; i < n_src; i++) cpu_mean_d += cpu_distances[i];
        cpu_mean_d /= n_src;
        cpu_mean_dists.push_back(cpu_mean_d);

        bool cpu_conv = (fabsf(cpu_prev_mean - cpu_mean_d) < CONV_THRESH);
        cpu_prev_mean = cpu_mean_d;

        if (!cpu_conv) {
            float cx_s = 0, cy_s = 0, cx_t = 0, cy_t = 0;
            for (int i = 0; i < n_src; i++) {
                cx_s += cpu_iter_src_x[i]; cy_s += cpu_iter_src_y[i];
                cx_t += tgt_x[cpu_matches[i]]; cy_t += tgt_y[cpu_matches[i]];
            }
            cx_s /= n_src; cy_s /= n_src; cx_t /= n_src; cy_t /= n_src;

            Eigen::Matrix2f W = Eigen::Matrix2f::Zero();
            for (int i = 0; i < n_src; i++) {
                float sx = cpu_iter_src_x[i] - cx_s, sy = cpu_iter_src_y[i] - cy_s;
                float tx = tgt_x[cpu_matches[i]] - cx_t, ty = tgt_y[cpu_matches[i]] - cy_t;
                W(0, 0) += sx * tx; W(0, 1) += sx * ty;
                W(1, 0) += sy * tx; W(1, 1) += sy * ty;
            }
            Eigen::JacobiSVD<Eigen::Matrix2f> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix2f U = svd.matrixU(), V = svd.matrixV();
            Eigen::Matrix2f R = V * U.transpose();
            if (R.determinant() < 0) { V.col(1) *= -1; R = V * U.transpose(); }
            Eigen::Vector2f t = Eigen::Vector2f(cx_t, cy_t) - R * Eigen::Vector2f(cx_s, cy_s);

            for (int i = 0; i < n_src; i++) {
                float x = cpu_iter_src_x[i], y = cpu_iter_src_y[i];
                cpu_iter_src_x[i] = R(0, 0) * x + R(0, 1) * y + t(0);
                cpu_iter_src_y[i] = R(1, 0) * x + R(1, 1) * y + t(1);
            }
        }
        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        snap.cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

        // === CUDA iteration ===
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        find_nearest_kernel<<<blocks_src, threads>>>(
            d_src_x, d_src_y, d_tgt_x, d_tgt_y,
            d_matches, d_distances, n_src, n_tgt);

        CUDA_CHECK(cudaMemset(d_mean_dist, 0, sizeof(float)));
        reduce_mean_distance_kernel<<<blocks_src, threads, threads * sizeof(float)>>>(
            d_distances, n_src, d_mean_dist);

        float h_mean_dist;
        CUDA_CHECK(cudaMemcpy(&h_mean_dist, d_mean_dist, sizeof(float), cudaMemcpyDeviceToHost));
        h_mean_dist /= n_src;
        cuda_mean_dists.push_back(h_mean_dist);

        bool cuda_conv = (fabsf(prev_mean - h_mean_dist) < CONV_THRESH);
        prev_mean = h_mean_dist;

        if (!cuda_conv) {
            CUDA_CHECK(cudaMemset(d_centroid, 0, 4 * sizeof(float)));
            compute_centroid_kernel<<<blocks_src, threads, 4 * threads * sizeof(float)>>>(
                d_src_x, d_src_y, d_tgt_x, d_tgt_y, d_matches, n_src, d_centroid);

            float h_centroid[4];
            CUDA_CHECK(cudaMemcpy(h_centroid, d_centroid, 4 * sizeof(float), cudaMemcpyDeviceToHost));
            float cx_s = h_centroid[0] / n_src, cy_s = h_centroid[1] / n_src;
            float cx_t = h_centroid[2] / n_src, cy_t = h_centroid[3] / n_src;

            CUDA_CHECK(cudaMemset(d_W, 0, 4 * sizeof(float)));
            compute_W_kernel<<<blocks_src, threads>>>(
                d_src_x, d_src_y, d_tgt_x, d_tgt_y, d_matches, n_src,
                cx_s, cy_s, cx_t, cy_t, d_W);

            float h_W[4];
            CUDA_CHECK(cudaMemcpy(h_W, d_W, 4 * sizeof(float), cudaMemcpyDeviceToHost));

            Eigen::Matrix2f W_mat;
            W_mat << h_W[0], h_W[1], h_W[2], h_W[3];
            Eigen::JacobiSVD<Eigen::Matrix2f> svd(W_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix2f U = svd.matrixU(), V = svd.matrixV();
            Eigen::Matrix2f R = V * U.transpose();
            if (R.determinant() < 0) { V.col(1) *= -1; R = V * U.transpose(); }
            Eigen::Vector2f t = Eigen::Vector2f(cx_t, cy_t) - R * Eigen::Vector2f(cx_s, cy_s);

            apply_transform_kernel<<<blocks_src, threads>>>(
                d_src_x, d_src_y, n_src,
                R(0, 0), R(0, 1), R(1, 0), R(1, 1), t(0), t(1));
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cuda_ms;
        cudaEventElapsedTime(&cuda_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cuda_total_ms += cuda_ms;
        snap.cuda_ms = cuda_ms;
        snap.mean_dist = h_mean_dist;

        // Read CUDA source points for visualization
        CUDA_CHECK(cudaMemcpy(cuda_src_x.data(), d_src_x, n_src * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cuda_src_y.data(), d_src_y, n_src * sizeof(float), cudaMemcpyDeviceToHost));
        snap.src_x = cuda_src_x;
        snap.src_y = cuda_src_y;

        snapshots.push_back(snap);

        printf("  iter %2d: CPU %.3f ms  CUDA %.3f ms  dist_cpu=%.4f dist_cuda=%.4f\n",
               iter, snap.cpu_ms, snap.cuda_ms, cpu_mean_d, h_mean_dist);

        if (cuda_conv && cpu_conv) {
            cuda_iters = iter + 1;
            cuda_converged = true;
            break;
        }
    }

    printf("\nSummary:\n");
    printf("  CPU:  %d iters, %.2f ms total\n",
           (int)cpu_mean_dists.size(), cpu_result.elapsed_ms);
    printf("  CUDA: %d iters, %.2f ms total\n",
           (int)cuda_mean_dists.size(), cuda_total_ms);

    // ========================= Visualization =========================
    VisParams vp;
    vp.W = 600; vp.H = 600;
    vp.offset_x = -5.0f; vp.offset_y = -5.0f; vp.scale = 38.0f;

    cv::VideoWriter video(
        "gif/comparison_icp.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 10, cv::Size(vp.W * 2, vp.H));

    // Re-run CPU for per-frame snapshots (we already have the data)
    std::vector<float> cpu_frame_x = src_init_x;
    std::vector<float> cpu_frame_y = src_init_y;
    float cpu_pm = 1e10f;
    std::vector<float> cpu_frame_dists;

    for (size_t fi = 0; fi < snapshots.size(); fi++) {
        // CPU frame: re-compute this iteration
        {
            for (int i = 0; i < n_src; i++) {
                float best_d = 1e30f; int best_j = 0;
                for (int j = 0; j < n_tgt; j++) {
                    float dx = cpu_frame_x[i] - tgt_x[j];
                    float dy = cpu_frame_y[i] - tgt_y[j];
                    float d = dx * dx + dy * dy;
                    if (d < best_d) { best_d = d; best_j = j; }
                }
                cpu_matches[i] = best_j;
                cpu_distances[i] = sqrtf(best_d);
            }
            float md = 0;
            for (int i = 0; i < n_src; i++) md += cpu_distances[i];
            md /= n_src;
            cpu_frame_dists.push_back(md);

            if (fabsf(cpu_pm - md) >= CONV_THRESH) {
                float cx_s = 0, cy_s = 0, cx_t = 0, cy_t = 0;
                for (int i = 0; i < n_src; i++) {
                    cx_s += cpu_frame_x[i]; cy_s += cpu_frame_y[i];
                    cx_t += tgt_x[cpu_matches[i]]; cy_t += tgt_y[cpu_matches[i]];
                }
                cx_s /= n_src; cy_s /= n_src; cx_t /= n_src; cy_t /= n_src;

                Eigen::Matrix2f W = Eigen::Matrix2f::Zero();
                for (int i = 0; i < n_src; i++) {
                    float sx = cpu_frame_x[i] - cx_s, sy = cpu_frame_y[i] - cy_s;
                    float tx = tgt_x[cpu_matches[i]] - cx_t, ty = tgt_y[cpu_matches[i]] - cy_t;
                    W(0, 0) += sx * tx; W(0, 1) += sx * ty;
                    W(1, 0) += sy * tx; W(1, 1) += sy * ty;
                }
                Eigen::JacobiSVD<Eigen::Matrix2f> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix2f U = svd.matrixU(), V = svd.matrixV();
                Eigen::Matrix2f R = V * U.transpose();
                if (R.determinant() < 0) { V.col(1) *= -1; R = V * U.transpose(); }
                Eigen::Vector2f t = Eigen::Vector2f(cx_t, cy_t) - R * Eigen::Vector2f(cx_s, cy_s);

                for (int i = 0; i < n_src; i++) {
                    float x = cpu_frame_x[i], y = cpu_frame_y[i];
                    cpu_frame_x[i] = R(0, 0) * x + R(0, 1) * y + t(0);
                    cpu_frame_y[i] = R(1, 0) * x + R(1, 1) * y + t(1);
                }
            }
            cpu_pm = md;
        }

        cv::Mat left(vp.H, vp.W, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Mat right(vp.H, vp.W, CV_8UC3, cv::Scalar(255, 255, 255));

        draw_icp_scene(left, tgt_x, tgt_y, src_init_x, src_init_y,
                       cpu_frame_x, cpu_frame_y, vp,
                       "CPU (Sequential NN)", snapshots[fi].iter,
                       cpu_frame_dists.back(), snapshots[fi].cpu_ms,
                       cpu_frame_dists);

        draw_icp_scene(right, tgt_x, tgt_y, src_init_x, src_init_y,
                       snapshots[fi].src_x, snapshots[fi].src_y, vp,
                       "CUDA (Parallel NN)", snapshots[fi].iter,
                       snapshots[fi].mean_dist, snapshots[fi].cuda_ms,
                       cuda_mean_dists);

        cv::Mat combined;
        cv::hconcat(left, right, combined);

        // Write multiple frames for the last iteration for visibility
        int repeat = (fi == snapshots.size() - 1) ? 20 : 1;
        for (int r = 0; r < repeat; r++) video.write(combined);
    }

    video.release();
    std::cout << "Video saved to gif/comparison_icp.avi" << std::endl;

    system("ffmpeg -y -i gif/comparison_icp.avi "
           "-vf 'fps=10,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_icp.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_icp.gif" << std::endl;

    // Cleanup
    cudaFree(d_src_x); cudaFree(d_src_y);
    cudaFree(d_tgt_x); cudaFree(d_tgt_y);
    cudaFree(d_matches); cudaFree(d_distances);
    cudaFree(d_centroid); cudaFree(d_W);
    cudaFree(d_mean_dist);

    return 0;
}
