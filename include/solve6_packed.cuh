// solve6_packed.cuh
//
// A host 6x6 SPD direct solve over a PACKED (upper-triangular, 21-float) normal
// matrix, used by cudabot's local 3D scan-matching family (Gauss-Newton on
// SE(3)): gpu_ndt_3d.cu, gpu_ndt_3d_multires.cu, gpu_gicp_3d.cu.  The GPU
// assembles the per-correspondence 6x6 normal equations into 21 upper-triangle
// floats (H_OFF maps (row, col) -> packed index); these are copied back to the
// host and solved here once per Gauss-Newton iteration.  The H_OFF table and the
// solve were copied verbatim across all three demos; lifting them here removes
// that drift.
//
// This is intentionally SEPARATE from se3_helpers.cuh's solve6_spd_device:
//   - that one is a __device__ routine over a FULL 6x6 matrix, run on the GPU
//     for the rotation-matrix pose-graph back-ends;
//   - this one is a host routine over the PACKED 21-float upper triangle, and
//     is byte-for-byte the algorithm the scan-matching demos were already using
//     (so their tuning-sensitive convergence basins are unchanged).
//
// H_packed: 21 floats, upper triangle row-major via H_OFF.  g: 6 floats.
// Solves (H + lambda*I) dx = g, i.e. dx = (H + lambda*I)^-1 g.  Returns false
// (caller bumps lambda and retries) if the matrix is not positive definite.
//
// Lives in namespace cudabot; `static` so each translation unit that includes
// this header gets its own internal-linkage copy with no ODR concerns.

#pragma once

#include <cmath>

namespace cudabot {

// Map (row, col) of the symmetric 6x6 normal matrix to its packed upper-tri
// index in the 21-float H array assembled on the GPU.
static const int H_OFF[6][6] = {
    { 0,  1,  2,  3,  4,  5},
    { 1,  6,  7,  8,  9, 10},
    { 2,  7, 11, 12, 13, 14},
    { 3,  8, 12, 15, 16, 17},
    { 4,  9, 13, 16, 18, 19},
    { 5, 10, 14, 17, 19, 20},
};

// Host: 6x6 SPD solve via Cholesky + back-substitution.
// H_packed: 21 floats upper-tri, g: 6 floats.  Computes dx = (H + lambda*I)^-1 g.
static bool cholesky_solve_6(const float* H_packed, const float* g, float lambda,
                             float* dx) {
    // Build full 6x6 symmetric matrix A, then A = L L^T (lower).
    float A[36];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
            A[6*i + j] = H_packed[H_OFF[i][j]];
            if (i == j) A[6*i + j] += lambda;
        }
    float L[36] = {0};
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j <= i; j++) {
            float s = A[6*i + j];
            for (int k = 0; k < j; k++) s -= L[6*i + k] * L[6*j + k];
            if (i == j) {
                if (s <= 0.0f) return false;
                L[6*i + j] = std::sqrt(s);
            } else {
                L[6*i + j] = s / L[6*j + j];
            }
        }
    }
    // Forward: L y = g
    float y[6];
    for (int i = 0; i < 6; i++) {
        float s = g[i];
        for (int k = 0; k < i; k++) s -= L[6*i + k] * y[k];
        y[i] = s / L[6*i + i];
    }
    // Back: L^T x = y
    for (int i = 5; i >= 0; i--) {
        float s = y[i];
        for (int k = i + 1; k < 6; k++) s -= L[6*k + i] * dx[k];
        dx[i] = s / L[6*i + i];
    }
    return true;
}

}  // namespace cudabot
