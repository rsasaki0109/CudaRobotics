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

namespace cudabot {

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline void mat3_identity(float* R) {
    R[0] = 1.0f; R[1] = 0.0f; R[2] = 0.0f;
    R[3] = 0.0f; R[4] = 1.0f; R[5] = 0.0f;
    R[6] = 0.0f; R[7] = 0.0f; R[8] = 1.0f;
}

__host__ __device__ static inline void mat3_mul(const float* A, const float* B, float* C) {
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float v = 0.0f;
            for (int k = 0; k < 3; k++) v += A[3 * r + k] * B[3 * k + c];
            C[3 * r + c] = v;
        }
    }
}

__host__ __device__ static inline void mat3_transpose_mul(const float* A,
                                                          const float* B,
                                                          float* C) {
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float v = 0.0f;
            for (int k = 0; k < 3; k++) v += A[3 * k + r] * B[3 * k + c];
            C[3 * r + c] = v;
        }
    }
}

__host__ __device__ static inline void mat3_transpose_vec(const float* R,
                                                          const float* v,
                                                          float* out) {
    out[0] = R[0] * v[0] + R[3] * v[1] + R[6] * v[2];
    out[1] = R[1] * v[0] + R[4] * v[1] + R[7] * v[2];
    out[2] = R[2] * v[0] + R[5] * v[1] + R[8] * v[2];
}

__host__ __device__ static inline void mat3_vec(const float* R, const float* v, float* out) {
    out[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
    out[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
    out[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
}

__host__ __device__ static inline void so3_exp(const float* w, float* R) {
    float theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    float theta = sqrtf(theta2);
    float A = 1.0f;
    float B = 0.5f;
    if (theta > 1.0e-7f) {
        A = sinf(theta) / theta;
        B = (1.0f - cosf(theta)) / theta2;
    }
    float K[9] = {
        0.0f, -w[2],  w[1],
        w[2],  0.0f, -w[0],
       -w[1],  w[0],  0.0f
    };
    float K2[9];
    mat3_mul(K, K, K2);
    mat3_identity(R);
    for (int k = 0; k < 9; k++) R[k] += A * K[k] + B * K2[k];
}

__host__ __device__ static inline void so3_log(const float* R, float* w) {
    float cos_theta = clampf((R[0] + R[4] + R[8] - 1.0f) * 0.5f, -1.0f, 1.0f);
    float theta = acosf(cos_theta);
    if (theta < 1.0e-6f) {
        w[0] = 0.5f * (R[7] - R[5]);
        w[1] = 0.5f * (R[2] - R[6]);
        w[2] = 0.5f * (R[3] - R[1]);
        return;
    }
    float scale = theta / (2.0f * sinf(theta));
    w[0] = scale * (R[7] - R[5]);
    w[1] = scale * (R[2] - R[6]);
    w[2] = scale * (R[3] - R[1]);
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
