/*************************************************************************
    RANSAC Plane Detection - CUDA-parallelized
    Each thread runs one RANSAC iteration: randomly selects 3 points,
    fits a plane, counts inliers. Best plane is selected on host.
    Uses cuRAND for device-side random number generation.
 ************************************************************************/

#include "cuda_pointcloud.cuh"
#include <curand_kernel.h>
#include <cstdio>

namespace cudabot {

// =====================================================================
// Kernels
// =====================================================================

// Initialize cuRAND states
__global__ void init_curand_kernel(curandState* states, int n, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &states[i]);
}

// Each thread = one RANSAC iteration
__global__ void ransac_plane_kernel(
    const float* x, const float* y, const float* z, int n,
    float dist_threshold,
    curandState* rng,
    float* plane_coeffs,  // [max_iter * 4]
    int* inlier_counts,   // [max_iter]
    int max_iter)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_iter) return;

    curandState local_state = rng[i];

    // Pick 3 random distinct points
    int i0 = curand(&local_state) % n;
    int i1 = curand(&local_state) % n;
    int i2 = curand(&local_state) % n;

    // Ensure distinct
    while (i1 == i0) i1 = curand(&local_state) % n;
    while (i2 == i0 || i2 == i1) i2 = curand(&local_state) % n;

    float x0 = x[i0], y0 = y[i0], z0 = z[i0];
    float x1 = x[i1], y1 = y[i1], z1 = z[i1];
    float x2 = x[i2], y2 = y[i2], z2 = z[i2];

    // Two edge vectors
    float v1x = x1 - x0, v1y = y1 - y0, v1z = z1 - z0;
    float v2x = x2 - x0, v2y = y2 - y0, v2z = z2 - z0;

    // Normal = v1 x v2
    float a = v1y * v2z - v1z * v2y;
    float b = v1z * v2x - v1x * v2z;
    float c = v1x * v2y - v1y * v2x;

    float norm = sqrtf(a * a + b * b + c * c);
    if (norm < 1e-10f) {
        plane_coeffs[i * 4 + 0] = 0;
        plane_coeffs[i * 4 + 1] = 0;
        plane_coeffs[i * 4 + 2] = 0;
        plane_coeffs[i * 4 + 3] = 0;
        inlier_counts[i] = 0;
        rng[i] = local_state;
        return;
    }

    float inv_norm = 1.0f / norm;
    a *= inv_norm;
    b *= inv_norm;
    c *= inv_norm;
    float d = -(a * x0 + b * y0 + c * z0);

    // Count inliers
    int count = 0;
    for (int j = 0; j < n; j++) {
        float dist = fabsf(a * x[j] + b * y[j] + c * z[j] + d);
        if (dist <= dist_threshold) count++;
    }

    plane_coeffs[i * 4 + 0] = a;
    plane_coeffs[i * 4 + 1] = b;
    plane_coeffs[i * 4 + 2] = c;
    plane_coeffs[i * 4 + 3] = d;
    inlier_counts[i] = count;

    rng[i] = local_state;
}

// =====================================================================
// Host function
// =====================================================================

void ransac_plane(const CudaPointCloud& input, float* plane_coeffs,
                  float distance_threshold, int max_iterations) {
    int n = input.size();
    if (n < 3) return;

    // Initialize cuRAND
    curandState* d_rng;
    CUDA_CHECK(cudaMalloc(&d_rng, max_iterations * sizeof(curandState)));

    int block = 256;
    int grid_rng = (max_iterations + block - 1) / block;
    init_curand_kernel<<<grid_rng, block>>>(d_rng, max_iterations, 42ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate output arrays
    float* d_plane_coeffs;
    int* d_inlier_counts;
    CUDA_CHECK(cudaMalloc(&d_plane_coeffs, max_iterations * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inlier_counts, max_iterations * sizeof(int)));

    // Run RANSAC
    ransac_plane_kernel<<<grid_rng, block>>>(
        input.d_x(), input.d_y(), input.d_z(), n,
        distance_threshold, d_rng,
        d_plane_coeffs, d_inlier_counts, max_iterations);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download and find best
    std::vector<float> h_coeffs(max_iterations * 4);
    std::vector<int> h_counts(max_iterations);
    CUDA_CHECK(cudaMemcpy(h_coeffs.data(), d_plane_coeffs, max_iterations * 4 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_inlier_counts, max_iterations * sizeof(int), cudaMemcpyDeviceToHost));

    int best_idx = 0;
    int best_count = 0;
    for (int i = 0; i < max_iterations; i++) {
        if (h_counts[i] > best_count) {
            best_count = h_counts[i];
            best_idx = i;
        }
    }

    plane_coeffs[0] = h_coeffs[best_idx * 4 + 0];
    plane_coeffs[1] = h_coeffs[best_idx * 4 + 1];
    plane_coeffs[2] = h_coeffs[best_idx * 4 + 2];
    plane_coeffs[3] = h_coeffs[best_idx * 4 + 3];

    cudaFree(d_rng);
    cudaFree(d_plane_coeffs);
    cudaFree(d_inlier_counts);
}

} // namespace cudabot
