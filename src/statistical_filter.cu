/*************************************************************************
    Statistical Outlier Removal - CUDA-parallelized
    For each point, find k nearest neighbors, compute mean distance,
    then remove points whose mean distance exceeds global_mean + std_mul * global_std.
 ************************************************************************/

#include "cuda_pointcloud.cuh"

namespace cudabot {

// =====================================================================
// Kernels
// =====================================================================

// k-NN brute force: compute mean distance to k nearest neighbors per point
__global__ void compute_mean_knn_distances_kernel(
    const float* x, const float* y, const float* z, int n,
    float* mean_distances, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float px = x[i], py = y[i], pz = z[i];

    // Use a simple insertion sort to maintain k smallest distances
    // For k=20 this is efficient enough
    const int MAX_K = 64;
    float knn_dists[MAX_K];
    int actual_k = min(k, MAX_K);
    for (int j = 0; j < actual_k; j++) knn_dists[j] = 1e30f;

    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        float dx = px - x[j];
        float dy = py - y[j];
        float dz = pz - z[j];
        float d2 = dx * dx + dy * dy + dz * dz;

        // Insert into sorted array if smaller than largest
        if (d2 < knn_dists[actual_k - 1]) {
            knn_dists[actual_k - 1] = d2;
            // Bubble down
            for (int m = actual_k - 1; m > 0 && knn_dists[m] < knn_dists[m - 1]; m--) {
                float tmp = knn_dists[m];
                knn_dists[m] = knn_dists[m - 1];
                knn_dists[m - 1] = tmp;
            }
        }
    }

    float sum = 0.0f;
    for (int j = 0; j < actual_k; j++) sum += sqrtf(knn_dists[j]);
    mean_distances[i] = sum / (float)actual_k;
}

// Mark points as inliers or outliers
__global__ void mark_inliers_kernel(
    const float* mean_distances, int n,
    float global_mean, float global_std, float std_mul,
    int* inlier_flags)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float threshold = global_mean + std_mul * global_std;
    inlier_flags[i] = (mean_distances[i] <= threshold) ? 1 : 0;
}

// Compact inlier points using prefix sum results
__global__ void compact_points_kernel(
    const float* x, const float* y, const float* z, int n,
    const int* inlier_flags, const int* prefix_sum,
    float* out_x, float* out_y, float* out_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (inlier_flags[i]) {
        int idx = prefix_sum[i];
        out_x[idx] = x[i];
        out_y[idx] = y[i];
        out_z[idx] = z[i];
    }
}

// Simple exclusive prefix sum (for small n, single block)
__global__ void prefix_sum_kernel(const int* input, int* output, int n) {
    // Simple sequential scan - called with 1 thread for simplicity
    // For large n, use thrust::exclusive_scan instead
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            output[i] = sum;
            sum += input[i];
        }
    }
}

// =====================================================================
// Host function
// =====================================================================

CudaPointCloud statistical_outlier_removal(const CudaPointCloud& input, int k, float std_mul) {
    int n = input.size();
    if (n == 0) return CudaPointCloud();

    // Compute mean kNN distances
    float* d_mean_dists;
    CUDA_CHECK(cudaMalloc(&d_mean_dists, n * sizeof(float)));

    int block = 256;
    int grid = (n + block - 1) / block;
    compute_mean_knn_distances_kernel<<<grid, block>>>(
        input.d_x(), input.d_y(), input.d_z(), n, d_mean_dists, k);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download mean distances and compute global mean and std on host
    std::vector<float> h_mean_dists(n);
    CUDA_CHECK(cudaMemcpy(h_mean_dists.data(), d_mean_dists, n * sizeof(float), cudaMemcpyDeviceToHost));

    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; i++) {
        sum += h_mean_dists[i];
        sum2 += (double)h_mean_dists[i] * h_mean_dists[i];
    }
    float global_mean = (float)(sum / n);
    float global_std = (float)sqrtf((float)(sum2 / n - (double)global_mean * global_mean));

    // Mark inliers
    int* d_inlier_flags;
    int* d_prefix_sum;
    CUDA_CHECK(cudaMalloc(&d_inlier_flags, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prefix_sum, n * sizeof(int)));

    mark_inliers_kernel<<<grid, block>>>(d_mean_dists, n, global_mean, global_std, std_mul, d_inlier_flags);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Prefix sum
    prefix_sum_kernel<<<1, 1>>>(d_inlier_flags, d_prefix_sum, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Count inliers
    std::vector<int> h_flags(n);
    CUDA_CHECK(cudaMemcpy(h_flags.data(), d_inlier_flags, n * sizeof(int), cudaMemcpyDeviceToHost));
    int n_inliers = 0;
    for (int i = 0; i < n; i++) n_inliers += h_flags[i];

    if (n_inliers == 0) {
        cudaFree(d_mean_dists); cudaFree(d_inlier_flags); cudaFree(d_prefix_sum);
        return CudaPointCloud();
    }

    // Compact
    CudaPointCloud result;
    result.reserve(n_inliers);
    result.setSize(n_inliers);

    compact_points_kernel<<<grid, block>>>(
        input.d_x(), input.d_y(), input.d_z(), n,
        d_inlier_flags, d_prefix_sum,
        result.d_x(), result.d_y(), result.d_z());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_mean_dists);
    cudaFree(d_inlier_flags);
    cudaFree(d_prefix_sum);

    return result;
}

} // namespace cudabot
