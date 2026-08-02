// se3_helpers.cuh
//
// Shared SE(3) / SO(3) math kernels and a 6x6 SPD direct solve, used by
// cudabot's rotation-matrix pose-graph SLAM back-ends (Gauss-Newton +
// Jacobi-preconditioned conjugate gradient on SE(3)).  These were copied
// verbatim across gpu_pose_graph_slam_3d.cu, gpu_pose_graph_slam_3d_switchable.cu,
// and gpu_online_slam_3d_switchable.cu; lifting them here removes that drift.
//
// All helpers are struct-agnostic: poses are passed as raw float arrays
// (t[3], R[9] row-major).  The struct-coupled helpers (pose_relative,
// residual_edge, perturb_pose) stay in each .cu because they depend on the
// demo-local Pose / Edge layout.
//
// Conventions:
//   - Rotations are 3x3 row-major matrices in float[9].
//   - so3_exp / so3_log map between so(3) tangent vectors and SO(3).
//   - solve6_spd_device solves (A + damping*I) x = rhs for a 6x6 SPD A via
//     Cholesky; returns false (caller falls back to a diagonal solve) if A
//     is not positive definite.
//
// All functions live in namespace cudabot.  They are `static inline`
// (host+device math) or `__device__ static` (the SPD solve), so each
// translation unit that includes this header gets its own internal-linkage
// copy with no ODR concerns.

#pragma once

#include <cuda_runtime.h>
#include <cmath>

#include "cudarobotics/lie_group_math.cuh"

namespace cudabot {

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return cudarobotics::lie::clamp(x, lo, hi);
}

__host__ __device__ static inline void mat3_identity(float* R) {
    cudarobotics::lie::mat3_identity(R);
}

__host__ __device__ static inline void mat3_mul(const float* A, const float* B, float* C) {
    cudarobotics::lie::mat3_mul(A, B, C);
}

__host__ __device__ static inline void mat3_transpose_mul(const float* A,
                                                          const float* B,
                                                          float* C) {
    cudarobotics::lie::mat3_transpose_mul(A, B, C);
}

__host__ __device__ static inline void mat3_transpose_vec(const float* R,
                                                          const float* v,
                                                          float* out) {
    cudarobotics::lie::mat3_transpose_vec(R, v, out);
}

__host__ __device__ static inline void mat3_vec(const float* R, const float* v, float* out) {
    cudarobotics::lie::mat3_vec(R, v, out);
}

__host__ __device__ static inline void so3_exp(const float* w, float* R) {
    cudarobotics::lie::so3_exp(w, R);
}

__host__ __device__ static inline void so3_log(const float* R, float* w) {
    cudarobotics::lie::so3_log(R, w);
}

// Solve (A_in + damping*I) out = rhs for a 6x6 symmetric positive-definite
// system via Cholesky.  Returns false if a non-positive pivot is hit (caller
// should fall back to a diagonal/Jacobi solve).
__device__ static bool solve6_spd_device(const float* A_in,
                                         const float* rhs,
                                         float damping,
                                         float* out) {
    float A[36];
    float L[36];
    for (int i = 0; i < 36; i++) {
        A[i] = A_in[i];
        L[i] = 0.0f;
    }
    for (int i = 0; i < 6; i++) A[6 * i + i] += damping;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j <= i; j++) {
            float s = A[6 * i + j];
            for (int k = 0; k < j; k++) s -= L[6 * i + k] * L[6 * j + k];
            if (i == j) {
                if (s <= 1.0e-12f) return false;
                L[6 * i + j] = sqrtf(s);
            } else {
                L[6 * i + j] = s / L[6 * j + j];
            }
        }
    }
    float y[6];
    for (int i = 0; i < 6; i++) {
        float s = rhs[i];
        for (int k = 0; k < i; k++) s -= L[6 * i + k] * y[k];
        y[i] = s / L[6 * i + i];
    }
    for (int i = 5; i >= 0; i--) {
        float s = y[i];
        for (int k = i + 1; k < 6; k++) s -= L[6 * k + i] * out[k];
        out[i] = s / L[6 * i + i];
    }
    return true;
}

}  // namespace cudabot
