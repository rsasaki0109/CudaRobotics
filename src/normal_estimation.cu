/*************************************************************************
    Normal Estimation via PCA - CUDA-parallelized
    For each point, find k nearest neighbors, compute 3x3 covariance matrix,
    then find the eigenvector corresponding to the smallest eigenvalue.
    Uses analytical eigensolver for symmetric 3x3 matrices.
 ************************************************************************/

#include "cuda_pointcloud.cuh"
#include <cmath>

namespace cudabot {

// =====================================================================
// Device helpers
// =====================================================================

// Analytical eigenvalue computation for symmetric 3x3 matrix
// Returns the smallest eigenvalue and its eigenvector
__device__ void smallest_eigenvector_3x3(
    float a00, float a01, float a02,
    float a11, float a12, float a22,
    float& evx, float& evy, float& evz)
{
    // Characteristic polynomial: -lambda^3 + c2*lambda^2 + c1*lambda + c0 = 0
    // Using Cardano's method for symmetric 3x3

    float p1 = a01 * a01 + a02 * a02 + a12 * a12;

    if (p1 < 1e-12f) {
        // Matrix is diagonal - find min eigenvalue
        float e0 = a00, e1 = a11, e2 = a22;
        if (e0 <= e1 && e0 <= e2) { evx = 1; evy = 0; evz = 0; }
        else if (e1 <= e0 && e1 <= e2) { evx = 0; evy = 1; evz = 0; }
        else { evx = 0; evy = 0; evz = 1; }
        return;
    }

    float trace = a00 + a11 + a22;
    float q = trace / 3.0f;

    float b00 = a00 - q, b11 = a11 - q, b22 = a22 - q;
    float p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0f * p1;
    float p = sqrtf(p2 / 6.0f);

    if (p < 1e-12f) {
        evx = 1; evy = 0; evz = 0;
        return;
    }

    float inv_p = 1.0f / p;
    float c00 = b00 * inv_p, c01 = a01 * inv_p, c02 = a02 * inv_p;
    float c11 = b11 * inv_p, c12 = a12 * inv_p, c22 = b22 * inv_p;

    // det(B/p)
    float detB = c00 * (c11 * c22 - c12 * c12)
               - c01 * (c01 * c22 - c12 * c02)
               + c02 * (c01 * c12 - c11 * c02);
    float half_detB = detB * 0.5f;
    half_detB = fmaxf(-1.0f, fminf(1.0f, half_detB));

    float phi = acosf(half_detB) / 3.0f;

    // Eigenvalues in ascending order
    float eig0 = q + 2.0f * p * cosf(phi + 2.0f * 3.14159265f / 3.0f);  // smallest
    // float eig1 = q + 2.0f * p * cosf(phi + 4.0f * 3.14159265f / 3.0f);
    // float eig2 = q + 2.0f * p * cosf(phi);  // largest

    // Power iteration to find eigenvector for smallest eigenvalue
    // (A - eig0*I) should be rank 2, its null space is our eigenvector
    // Use cross product of two rows of (A - eig0*I)
    float m00 = a00 - eig0, m01 = a01, m02 = a02;
    float m10 = a01, m11 = a11 - eig0, m12 = a12;
    float m20 = a02, m21 = a12, m22 = a22 - eig0;

    // Cross products of row pairs
    float c0x = m01 * m12 - m02 * m11;
    float c0y = m02 * m10 - m00 * m12;
    float c0z = m00 * m11 - m01 * m10;
    float len0 = c0x * c0x + c0y * c0y + c0z * c0z;

    float c1x = m01 * m22 - m02 * m21;
    float c1y = m02 * m20 - m00 * m22;
    float c1z = m00 * m21 - m01 * m20;
    float len1 = c1x * c1x + c1y * c1y + c1z * c1z;

    float c2x = m11 * m22 - m12 * m21;
    float c2y = m12 * m20 - m10 * m22;
    float c2z = m10 * m21 - m11 * m20;
    float len2 = c2x * c2x + c2y * c2y + c2z * c2z;

    float rx, ry, rz;
    if (len0 >= len1 && len0 >= len2) {
        float inv = rsqrtf(len0 + 1e-20f);
        rx = c0x * inv; ry = c0y * inv; rz = c0z * inv;
    } else if (len1 >= len0 && len1 >= len2) {
        float inv = rsqrtf(len1 + 1e-20f);
        rx = c1x * inv; ry = c1y * inv; rz = c1z * inv;
    } else {
        float inv = rsqrtf(len2 + 1e-20f);
        rx = c2x * inv; ry = c2y * inv; rz = c2z * inv;
    }

    evx = rx; evy = ry; evz = rz;
}

// =====================================================================
// Kernels
// =====================================================================

__global__ void estimate_normals_kernel(
    const float* x, const float* y, const float* z, int n,
    float* nx, float* ny, float* nz, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float px = x[i], py = y[i], pz = z[i];

    // Find k nearest neighbors (brute force with insertion sort)
    const int MAX_K = 64;
    float knn_dists[MAX_K];
    int knn_idx[MAX_K];
    int actual_k = min(k, min(n - 1, MAX_K));
    for (int j = 0; j < actual_k; j++) { knn_dists[j] = 1e30f; knn_idx[j] = -1; }

    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        float dx = px - x[j];
        float dy = py - y[j];
        float dz = pz - z[j];
        float d2 = dx * dx + dy * dy + dz * dz;

        if (d2 < knn_dists[actual_k - 1]) {
            knn_dists[actual_k - 1] = d2;
            knn_idx[actual_k - 1] = j;
            for (int m = actual_k - 1; m > 0 && knn_dists[m] < knn_dists[m - 1]; m--) {
                float tmp_d = knn_dists[m]; knn_dists[m] = knn_dists[m - 1]; knn_dists[m - 1] = tmp_d;
                int tmp_i = knn_idx[m]; knn_idx[m] = knn_idx[m - 1]; knn_idx[m - 1] = tmp_i;
            }
        }
    }

    // Compute centroid of neighbors
    float cx = 0, cy = 0, cz = 0;
    for (int j = 0; j < actual_k; j++) {
        int idx = knn_idx[j];
        cx += x[idx]; cy += y[idx]; cz += z[idx];
    }
    float inv_k = 1.0f / (float)actual_k;
    cx *= inv_k; cy *= inv_k; cz *= inv_k;

    // Compute 3x3 covariance matrix (symmetric)
    float cov00 = 0, cov01 = 0, cov02 = 0;
    float cov11 = 0, cov12 = 0, cov22 = 0;
    for (int j = 0; j < actual_k; j++) {
        int idx = knn_idx[j];
        float dx = x[idx] - cx;
        float dy = y[idx] - cy;
        float dz = z[idx] - cz;
        cov00 += dx * dx; cov01 += dx * dy; cov02 += dx * dz;
        cov11 += dy * dy; cov12 += dy * dz;
        cov22 += dz * dz;
    }

    // Find eigenvector of smallest eigenvalue = normal
    float evx, evy, evz;
    smallest_eigenvector_3x3(cov00, cov01, cov02, cov11, cov12, cov22, evx, evy, evz);

    // Consistent orientation: flip normal towards origin
    if (evx * px + evy * py + evz * pz > 0) {
        evx = -evx; evy = -evy; evz = -evz;
    }

    nx[i] = evx;
    ny[i] = evy;
    nz[i] = evz;
}

// =====================================================================
// Host function
// =====================================================================

void estimate_normals(const CudaPointCloud& input, float* d_nx, float* d_ny, float* d_nz, int k) {
    int n = input.size();
    if (n == 0) return;

    int block = 256;
    int grid = (n + block - 1) / block;
    estimate_normals_kernel<<<grid, block>>>(
        input.d_x(), input.d_y(), input.d_z(), n,
        d_nx, d_ny, d_nz, k);
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace cudabot
