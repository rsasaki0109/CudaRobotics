/*************************************************************************
    Voxel Grid Downsampling - CUDA-parallelized
    Each point is assigned to a voxel via 3D hash, then centroids are
    computed per voxel using atomicAdd accumulation.
 ************************************************************************/

#include "cuda_pointcloud.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <cfloat>

namespace cudabot {

// =====================================================================
// Kernels
// =====================================================================

// Compute min/max bounds via atomicMin/Max on integer-reinterpreted floats
__global__ void compute_bounds_kernel(
    const float* x, const float* y, const float* z, int n,
    float* bounds)  // [6]: min_x, min_y, min_z, max_x, max_y, max_z
{
    extern __shared__ float smem[];
    // smem layout: [6 * blockDim.x]
    // 0: min_x, 1: min_y, 2: min_z, 3: max_x, 4: max_y, 5: max_z
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float lmin_x = FLT_MAX, lmin_y = FLT_MAX, lmin_z = FLT_MAX;
    float lmax_x = -FLT_MAX, lmax_y = -FLT_MAX, lmax_z = -FLT_MAX;

    if (i < n) {
        lmin_x = lmax_x = x[i];
        lmin_y = lmax_y = y[i];
        lmin_z = lmax_z = z[i];
    }

    smem[tid * 6 + 0] = lmin_x;
    smem[tid * 6 + 1] = lmin_y;
    smem[tid * 6 + 2] = lmin_z;
    smem[tid * 6 + 3] = lmax_x;
    smem[tid * 6 + 4] = lmax_y;
    smem[tid * 6 + 5] = lmax_z;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid * 6 + 0] = fminf(smem[tid * 6 + 0], smem[(tid + s) * 6 + 0]);
            smem[tid * 6 + 1] = fminf(smem[tid * 6 + 1], smem[(tid + s) * 6 + 1]);
            smem[tid * 6 + 2] = fminf(smem[tid * 6 + 2], smem[(tid + s) * 6 + 2]);
            smem[tid * 6 + 3] = fmaxf(smem[tid * 6 + 3], smem[(tid + s) * 6 + 3]);
            smem[tid * 6 + 4] = fmaxf(smem[tid * 6 + 4], smem[(tid + s) * 6 + 4]);
            smem[tid * 6 + 5] = fmaxf(smem[tid * 6 + 5], smem[(tid + s) * 6 + 5]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin((int*)&bounds[0], __float_as_int(smem[0]));
        atomicMin((int*)&bounds[1], __float_as_int(smem[1]));
        atomicMin((int*)&bounds[2], __float_as_int(smem[2]));
        // For max, negate to use atomicMin trick
        // Actually use atomicMax with int reinterpretation (works for positive floats)
        // For general floats, use custom approach:
        // We simply use atomicAdd approach on host after kernel
    }
}

// Assign each point to a voxel and accumulate centroids
__global__ void voxel_assign_kernel(
    const float* x, const float* y, const float* z, int n,
    float leaf_size, float min_x, float min_y, float min_z,
    int* voxel_ids,
    float* voxel_sum_x, float* voxel_sum_y, float* voxel_sum_z,
    int* voxel_count,
    int grid_x, int grid_y, int grid_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int ix = (int)floorf((x[i] - min_x) / leaf_size);
    int iy = (int)floorf((y[i] - min_y) / leaf_size);
    int iz = (int)floorf((z[i] - min_z) / leaf_size);

    // Clamp
    ix = max(0, min(ix, grid_x - 1));
    iy = max(0, min(iy, grid_y - 1));
    iz = max(0, min(iz, grid_z - 1));

    int vid = ix + iy * grid_x + iz * grid_x * grid_y;
    voxel_ids[i] = vid;

    atomicAdd(&voxel_sum_x[vid], x[i]);
    atomicAdd(&voxel_sum_y[vid], y[i]);
    atomicAdd(&voxel_sum_z[vid], z[i]);
    atomicAdd(&voxel_count[vid], 1);
}

// Extract centroids of non-empty voxels
__global__ void extract_centroids_kernel(
    const float* voxel_sum_x, const float* voxel_sum_y, const float* voxel_sum_z,
    const int* voxel_count, int n_voxels,
    float* out_x, float* out_y, float* out_z,
    int* out_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_voxels) return;

    int cnt = voxel_count[i];
    if (cnt > 0) {
        int idx = atomicAdd(out_count, 1);
        out_x[idx] = voxel_sum_x[i] / (float)cnt;
        out_y[idx] = voxel_sum_y[i] / (float)cnt;
        out_z[idx] = voxel_sum_z[i] / (float)cnt;
    }
}

// =====================================================================
// Host function
// =====================================================================

CudaPointCloud voxel_grid_filter(const CudaPointCloud& input, float leaf_size) {
    int n = input.size();
    if (n == 0) return CudaPointCloud();

    // Download bounds to host
    std::vector<float> hx(n), hy(n), hz(n);
    CUDA_CHECK(cudaMemcpy(hx.data(), input.d_x(), n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hy.data(), input.d_y(), n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hz.data(), input.d_z(), n * sizeof(float), cudaMemcpyDeviceToHost));

    float min_x = FLT_MAX, min_y = FLT_MAX, min_z = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX, max_z = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        min_x = fminf(min_x, hx[i]); max_x = fmaxf(max_x, hx[i]);
        min_y = fminf(min_y, hy[i]); max_y = fmaxf(max_y, hy[i]);
        min_z = fminf(min_z, hz[i]); max_z = fmaxf(max_z, hz[i]);
    }

    int grid_x = (int)ceilf((max_x - min_x) / leaf_size) + 1;
    int grid_y = (int)ceilf((max_y - min_y) / leaf_size) + 1;
    int grid_z = (int)ceilf((max_z - min_z) / leaf_size) + 1;
    int n_voxels = grid_x * grid_y * grid_z;

    // Allocate device arrays
    int* d_voxel_ids;
    float *d_voxel_sum_x, *d_voxel_sum_y, *d_voxel_sum_z;
    int* d_voxel_count;
    CUDA_CHECK(cudaMalloc(&d_voxel_ids, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_voxel_sum_x, n_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_voxel_sum_y, n_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_voxel_sum_z, n_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_voxel_count, n_voxels * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_voxel_sum_x, 0, n_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_voxel_sum_y, 0, n_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_voxel_sum_z, 0, n_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_voxel_count, 0, n_voxels * sizeof(int)));

    int block = 256;
    int grid = (n + block - 1) / block;
    voxel_assign_kernel<<<grid, block>>>(
        input.d_x(), input.d_y(), input.d_z(), n,
        leaf_size, min_x, min_y, min_z,
        d_voxel_ids, d_voxel_sum_x, d_voxel_sum_y, d_voxel_sum_z,
        d_voxel_count, grid_x, grid_y, grid_z);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Extract centroids
    float *d_out_x, *d_out_y, *d_out_z;
    int* d_out_count;
    CUDA_CHECK(cudaMalloc(&d_out_x, n_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_y, n_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_z, n_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));

    int grid2 = (n_voxels + block - 1) / block;
    extract_centroids_kernel<<<grid2, block>>>(
        d_voxel_sum_x, d_voxel_sum_y, d_voxel_sum_z,
        d_voxel_count, n_voxels,
        d_out_x, d_out_y, d_out_z, d_out_count);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_out_count;
    CUDA_CHECK(cudaMemcpy(&h_out_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));

    CudaPointCloud result;
    result.reserve(h_out_count);
    result.setSize(h_out_count);
    CUDA_CHECK(cudaMemcpy(result.d_x(), d_out_x, h_out_count * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(result.d_y(), d_out_y, h_out_count * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(result.d_z(), d_out_z, h_out_count * sizeof(float), cudaMemcpyDeviceToDevice));

    cudaFree(d_voxel_ids);
    cudaFree(d_voxel_sum_x); cudaFree(d_voxel_sum_y); cudaFree(d_voxel_sum_z);
    cudaFree(d_voxel_count);
    cudaFree(d_out_x); cudaFree(d_out_y); cudaFree(d_out_z);
    cudaFree(d_out_count);

    return result;
}

} // namespace cudabot
