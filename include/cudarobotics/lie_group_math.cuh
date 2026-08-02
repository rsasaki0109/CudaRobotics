// Common host/device Lie-group math for CudaRobotics.
//
// The vector convention follows MathematicalRobotics' math_tools.py:
//   SE(2): xi = [rho_x, rho_y, phi]
//   SE(3): xi = [rho_x, rho_y, rho_z, omega_x, omega_y, omega_z]
//
// Matrices are row-major.  The returned rigid transforms map a point from
// the source frame into the destination frame:
//   p_dst = R * p_src + t
//
// This header intentionally uses fixed-size raw arrays so the same routines
// can be called from ordinary C++ reference code and CUDA kernels without an
// Eigen dependency on the device side.
//
// The formulas are ported from the documented Lie-group derivations in
// scomup/MathematicalRobotics (Copyright (c) 2022 Yang Liu, MIT license).
// Keep the upstream attribution
// and license notice with redistributed substantial portions of this file.

#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define CUDAROBOTICS_LIE_HD __host__ __device__
#else
#define CUDAROBOTICS_LIE_HD
#endif

namespace cudarobotics {
namespace lie {

constexpr float kPi = 3.14159265358979323846f;

CUDAROBOTICS_LIE_HD static inline float clamp(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

CUDAROBOTICS_LIE_HD static inline float wrap_angle(float angle) {
    while (angle > kPi) angle -= 2.0f * kPi;
    while (angle < -kPi) angle += 2.0f * kPi;
    return angle;
}

CUDAROBOTICS_LIE_HD static inline void mat2_identity(float* A) {
    A[0] = 1.0f; A[1] = 0.0f;
    A[2] = 0.0f; A[3] = 1.0f;
}

CUDAROBOTICS_LIE_HD static inline void mat3_identity(float* A) {
    A[0] = 1.0f; A[1] = 0.0f; A[2] = 0.0f;
    A[3] = 0.0f; A[4] = 1.0f; A[5] = 0.0f;
    A[6] = 0.0f; A[7] = 0.0f; A[8] = 1.0f;
}

CUDAROBOTICS_LIE_HD static inline void mat4_identity(float* A) {
    for (int i = 0; i < 16; ++i) A[i] = 0.0f;
    A[0] = 1.0f;
    A[5] = 1.0f;
    A[10] = 1.0f;
    A[15] = 1.0f;
}

CUDAROBOTICS_LIE_HD static inline void mat2_vec(const float* A,
                                                 const float* x,
                                                 float* y) {
    const float y0 = A[0] * x[0] + A[1] * x[1];
    const float y1 = A[2] * x[0] + A[3] * x[1];
    y[0] = y0;
    y[1] = y1;
}

CUDAROBOTICS_LIE_HD static inline void mat3_mul(const float* A,
                                                 const float* B,
                                                 float* C) {
    float out[9];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float value = 0.0f;
            for (int k = 0; k < 3; ++k) {
                value += A[3 * r + k] * B[3 * k + c];
            }
            out[3 * r + c] = value;
        }
    }
    for (int i = 0; i < 9; ++i) C[i] = out[i];
}

CUDAROBOTICS_LIE_HD static inline void mat3_transpose_mul(const float* A,
                                                           const float* B,
                                                           float* C) {
    float out[9];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float value = 0.0f;
            for (int k = 0; k < 3; ++k) {
                value += A[3 * k + r] * B[3 * k + c];
            }
            out[3 * r + c] = value;
        }
    }
    for (int i = 0; i < 9; ++i) C[i] = out[i];
}

CUDAROBOTICS_LIE_HD static inline void mat3_vec(const float* A,
                                                 const float* x,
                                                 float* y) {
    const float y0 = A[0] * x[0] + A[1] * x[1] + A[2] * x[2];
    const float y1 = A[3] * x[0] + A[4] * x[1] + A[5] * x[2];
    const float y2 = A[6] * x[0] + A[7] * x[1] + A[8] * x[2];
    y[0] = y0;
    y[1] = y1;
    y[2] = y2;
}

CUDAROBOTICS_LIE_HD static inline void mat3_transpose_vec(const float* A,
                                                           const float* x,
                                                           float* y) {
    const float y0 = A[0] * x[0] + A[3] * x[1] + A[6] * x[2];
    const float y1 = A[1] * x[0] + A[4] * x[1] + A[7] * x[2];
    const float y2 = A[2] * x[0] + A[5] * x[1] + A[8] * x[2];
    y[0] = y0;
    y[1] = y1;
    y[2] = y2;
}

CUDAROBOTICS_LIE_HD static inline void skew(const float* v, float* K) {
    K[0] = 0.0f;  K[1] = -v[2]; K[2] = v[1];
    K[3] = v[2];  K[4] = 0.0f;  K[5] = -v[0];
    K[6] = -v[1]; K[7] = v[0];  K[8] = 0.0f;
}

CUDAROBOTICS_LIE_HD static inline void unskew(const float* K, float* v) {
    // The upstream helper reads the three independent entries directly.
    // For a matrix produced by skew() this is identical to the antisymmetric
    // projection, while preserving the reference behavior for noisy input.
    v[0] = K[7];
    v[1] = K[2];
    v[2] = K[3];
}

CUDAROBOTICS_LIE_HD static inline void hat2d(const float* v, float* out) {
    out[0] = -v[1];
    out[1] = v[0];
}

CUDAROBOTICS_LIE_HD static inline void so2_exp(float phi, float* R) {
    const float c = cosf(phi);
    const float s = sinf(phi);
    R[0] = c;  R[1] = -s;
    R[2] = s;  R[3] = c;
}

CUDAROBOTICS_LIE_HD static inline float so2_log(const float* R) {
    return atan2f(R[2], R[0]);
}

CUDAROBOTICS_LIE_HD static inline void so2_left_jacobian(float phi,
                                                          float* V) {
    const float phi2 = phi * phi;
    float A;
    float B;
    if (phi2 < 1.0e-8f) {
        A = 1.0f - phi2 / 6.0f + phi2 * phi2 / 120.0f;
        B = 0.5f * phi - phi * phi2 / 24.0f + phi * phi2 * phi2 / 720.0f;
    } else {
        A = sinf(phi) / phi;
        B = (1.0f - cosf(phi)) / phi;
    }
    V[0] = A;  V[1] = -B;
    V[2] = B;  V[3] = A;
}

CUDAROBOTICS_LIE_HD static inline void so2_left_jacobian_inverse(float phi,
                                                                  float* V) {
    float J[4];
    so2_left_jacobian(phi, J);
    const float det = J[0] * J[0] + J[2] * J[2];
    const float inv_det = 1.0f / fmaxf(det, 1.0e-12f);
    V[0] = J[0] * inv_det;
    V[1] = J[2] * inv_det;
    V[2] = -J[2] * inv_det;
    V[3] = J[0] * inv_det;
}

CUDAROBOTICS_LIE_HD static inline void se2_exp(const float* xi, float* T) {
    const float phi = xi[2];
    float R[4];
    float V[4];
    so2_exp(phi, R);
    so2_left_jacobian(phi, V);
    const float rho[2] = {xi[0], xi[1]};
    float t[2];
    mat2_vec(V, rho, t);
    T[0] = R[0]; T[1] = R[1]; T[2] = t[0];
    T[3] = R[2]; T[4] = R[3]; T[5] = t[1];
    T[6] = 0.0f; T[7] = 0.0f; T[8] = 1.0f;
}

CUDAROBOTICS_LIE_HD static inline void se2_log(const float* T, float* xi) {
    const float phi = atan2f(T[3], T[0]);
    float V_inv[4];
    so2_left_jacobian_inverse(phi, V_inv);
    const float t[2] = {T[2], T[5]};
    float rho[2];
    mat2_vec(V_inv, t, rho);
    xi[0] = rho[0];
    xi[1] = rho[1];
    xi[2] = phi;
}

CUDAROBOTICS_LIE_HD static inline void so3_exp(const float* omega,
                                                 float* R) {
    const float theta2 = omega[0] * omega[0] +
                         omega[1] * omega[1] +
                         omega[2] * omega[2];
    float A;
    float B;
    if (theta2 < 1.0e-8f) {
        A = 1.0f - theta2 / 6.0f + theta2 * theta2 / 120.0f;
        B = 0.5f - theta2 / 24.0f + theta2 * theta2 / 720.0f;
    } else {
        const float theta = sqrtf(theta2);
        A = sinf(theta) / theta;
        B = (1.0f - cosf(theta)) / theta2;
    }

    float K[9];
    float K2[9];
    skew(omega, K);
    mat3_mul(K, K, K2);
    mat3_identity(R);
    for (int i = 0; i < 9; ++i) R[i] += A * K[i] + B * K2[i];
}

CUDAROBOTICS_LIE_HD static inline void quaternion_to_mat3(
    const float* quaternion, float* R) {
    const float x = quaternion[0];
    const float y = quaternion[1];
    const float z = quaternion[2];
    const float w = quaternion[3];
    const float norm2 = x * x + y * y + z * z + w * w;
    if (norm2 < 1.0e-20f) {
        mat3_identity(R);
        return;
    }
    const float s = 2.0f / norm2;
    const float xx = x * x * s;
    const float xy = x * y * s;
    const float xz = x * z * s;
    const float xw = x * w * s;
    const float yy = y * y * s;
    const float yz = y * z * s;
    const float yw = y * w * s;
    const float zz = z * z * s;
    const float zw = z * w * s;
    R[0] = 1.0f - yy - zz;
    R[1] = xy - zw;
    R[2] = xz + yw;
    R[3] = xy + zw;
    R[4] = 1.0f - xx - zz;
    R[5] = yz - xw;
    R[6] = xz - yw;
    R[7] = yz + xw;
    R[8] = 1.0f - xx - yy;
}

CUDAROBOTICS_LIE_HD static inline void mat3_to_quaternion(
    const float* R, float* quaternion) {
    const float trace = R[0] + R[4] + R[8];
    if (trace > 0.0f) {
        const float s = 2.0f * sqrtf(fmaxf(trace + 1.0f, 1.0e-20f));
        quaternion[3] = 0.25f * s;
        quaternion[0] = (R[7] - R[5]) / s;
        quaternion[1] = (R[2] - R[6]) / s;
        quaternion[2] = (R[3] - R[1]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        const float s = 2.0f * sqrtf(fmaxf(1.0f + R[0] - R[4] - R[8],
                                           1.0e-20f));
        quaternion[3] = (R[7] - R[5]) / s;
        quaternion[0] = 0.25f * s;
        quaternion[1] = (R[1] + R[3]) / s;
        quaternion[2] = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        const float s = 2.0f * sqrtf(fmaxf(1.0f - R[0] + R[4] - R[8],
                                           1.0e-20f));
        quaternion[3] = (R[2] - R[6]) / s;
        quaternion[0] = (R[1] + R[3]) / s;
        quaternion[1] = 0.25f * s;
        quaternion[2] = (R[5] + R[7]) / s;
    } else {
        const float s = 2.0f * sqrtf(fmaxf(1.0f - R[0] - R[4] + R[8],
                                           1.0e-20f));
        quaternion[3] = (R[3] - R[1]) / s;
        quaternion[0] = (R[2] + R[6]) / s;
        quaternion[1] = (R[5] + R[7]) / s;
        quaternion[2] = 0.25f * s;
    }
    const float norm = sqrtf(quaternion[0] * quaternion[0] +
                             quaternion[1] * quaternion[1] +
                             quaternion[2] * quaternion[2] +
                             quaternion[3] * quaternion[3]);
    if (norm < 1.0e-12f) {
        quaternion[0] = 0.0f;
        quaternion[1] = 0.0f;
        quaternion[2] = 0.0f;
        quaternion[3] = 1.0f;
    } else {
        for (int i = 0; i < 4; ++i) quaternion[i] /= norm;
    }
}

// The quaternion branch keeps the logarithm stable near a rotation of pi,
// where theta / sin(theta) is ill-conditioned.
CUDAROBOTICS_LIE_HD static inline void so3_log(const float* R,
                                                 float* omega) {
    float qw;
    float qx;
    float qy;
    float qz;
    const float trace = R[0] + R[4] + R[8];
    if (trace > 0.0f) {
        const float s = 2.0f * sqrtf(fmaxf(trace + 1.0f, 1.0e-12f));
        qw = 0.25f * s;
        qx = (R[7] - R[5]) / s;
        qy = (R[2] - R[6]) / s;
        qz = (R[3] - R[1]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        const float s = 2.0f * sqrtf(fmaxf(1.0f + R[0] - R[4] - R[8],
                                           1.0e-12f));
        qw = (R[7] - R[5]) / s;
        qx = 0.25f * s;
        qy = (R[1] + R[3]) / s;
        qz = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        const float s = 2.0f * sqrtf(fmaxf(1.0f - R[0] + R[4] - R[8],
                                           1.0e-12f));
        qw = (R[2] - R[6]) / s;
        qx = (R[1] + R[3]) / s;
        qy = 0.25f * s;
        qz = (R[5] + R[7]) / s;
    } else {
        const float s = 2.0f * sqrtf(fmaxf(1.0f - R[0] - R[4] + R[8],
                                           1.0e-12f));
        qw = (R[3] - R[1]) / s;
        qx = (R[2] + R[6]) / s;
        qy = (R[5] + R[7]) / s;
        qz = 0.25f * s;
    }

    // q and -q represent the same rotation.  Select the principal angle.
    if (qw < 0.0f) {
        qw = -qw;
        qx = -qx;
        qy = -qy;
        qz = -qz;
    }
    const float sin_half = sqrtf(qx * qx + qy * qy + qz * qz);
    if (sin_half < 1.0e-6f) {
        omega[0] = 0.5f * (R[7] - R[5]);
        omega[1] = 0.5f * (R[2] - R[6]);
        omega[2] = 0.5f * (R[3] - R[1]);
        return;
    }

    const float theta = 2.0f * atan2f(sin_half, fmaxf(qw, 0.0f));
    const float scale = theta / sin_half;
    omega[0] = scale * qx;
    omega[1] = scale * qy;
    omega[2] = scale * qz;
}

CUDAROBOTICS_LIE_HD static inline void so3_left_jacobian(const float* omega,
                                                          float* V) {
    const float theta2 = omega[0] * omega[0] +
                         omega[1] * omega[1] +
                         omega[2] * omega[2];
    float B;
    float C;
    if (theta2 < 1.0e-8f) {
        B = 0.5f - theta2 / 24.0f + theta2 * theta2 / 720.0f;
        C = 1.0f / 6.0f - theta2 / 120.0f + theta2 * theta2 / 5040.0f;
    } else {
        const float theta = sqrtf(theta2);
        B = (1.0f - cosf(theta)) / theta2;
        C = (theta - sinf(theta)) / (theta2 * theta);
    }
    float K[9];
    float K2[9];
    skew(omega, K);
    mat3_mul(K, K, K2);
    mat3_identity(V);
    for (int i = 0; i < 9; ++i) V[i] += B * K[i] + C * K2[i];
}

CUDAROBOTICS_LIE_HD static inline void so3_left_jacobian_inverse(
    const float* omega, float* V_inv) {
    const float theta2 = omega[0] * omega[0] +
                         omega[1] * omega[1] +
                         omega[2] * omega[2];
    float C;
    if (theta2 < 1.0e-8f) {
        C = 1.0f / 12.0f + theta2 / 720.0f + theta2 * theta2 / 30240.0f;
    } else {
        const float theta = sqrtf(theta2);
        C = (1.0f - 0.5f * theta / tanf(0.5f * theta)) / theta2;
    }
    float K[9];
    float K2[9];
    skew(omega, K);
    mat3_mul(K, K, K2);
    mat3_identity(V_inv);
    for (int i = 0; i < 9; ++i) V_inv[i] += -0.5f * K[i] + C * K2[i];
}

CUDAROBOTICS_LIE_HD static inline void so3_right_jacobian(
    const float* omega, float* V) {
    const float theta2 = omega[0] * omega[0] +
                         omega[1] * omega[1] +
                         omega[2] * omega[2];
    float A;
    float B;
    if (theta2 < 1.0e-8f) {
        A = 0.5f - theta2 / 24.0f + theta2 * theta2 / 720.0f;
        B = 1.0f / 6.0f - theta2 / 120.0f + theta2 * theta2 / 5040.0f;
    } else {
        const float theta = sqrtf(theta2);
        A = (1.0f - cosf(theta)) / theta2;
        B = (theta - sinf(theta)) / (theta2 * theta);
    }
    float K[9];
    float K2[9];
    skew(omega, K);
    mat3_mul(K, K, K2);
    mat3_identity(V);
    for (int i = 0; i < 9; ++i) V[i] += -A * K[i] + B * K2[i];
}

CUDAROBOTICS_LIE_HD static inline void so3_right_jacobian_inverse(
    const float* omega, float* V_inv) {
    const float theta2 = omega[0] * omega[0] +
                         omega[1] * omega[1] +
                         omega[2] * omega[2];
    float C;
    if (theta2 < 1.0e-8f) {
        C = 1.0f / 12.0f + theta2 / 720.0f + theta2 * theta2 / 30240.0f;
    } else {
        const float theta = sqrtf(theta2);
        C = (1.0f - 0.5f * theta / tanf(0.5f * theta)) / theta2;
    }
    float K[9];
    float K2[9];
    skew(omega, K);
    mat3_mul(K, K, K2);
    mat3_identity(V_inv);
    for (int i = 0; i < 9; ++i) V_inv[i] += 0.5f * K[i] + C * K2[i];
}

// Names used by mathR/utilities/math_tools.py.  HSO3 is the right Jacobian
// of SO(3); dLogSO3 is its inverse.
CUDAROBOTICS_LIE_HD static inline void hso3(const float* omega, float* H) {
    so3_right_jacobian(omega, H);
}

CUDAROBOTICS_LIE_HD static inline void d_log_so3(const float* omega,
                                                  float* H_inv) {
    so3_right_jacobian_inverse(omega, H_inv);
}

CUDAROBOTICS_LIE_HD static inline void d_hinv_so3(const float* omega,
                                                   const float* v,
                                                   float* d_hinv) {
    const float theta2 = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
    float H_inv[9];
    d_log_so3(omega, H_inv);
    float c[3];
    mat3_vec(H_inv, v, c);
    float skew_c[9];
    skew(c, skew_c);
    if (theta2 <= 1.0e-8f) {
        for (int i = 0; i < 9; ++i) d_hinv[i] = 0.5f * skew_c[i];
        return;
    }
    const float theta = sqrtf(theta2);
    float K[9];
    float W[9];
    skew(omega, W);
    for (int i = 0; i < 9; ++i) K[i] = W[i] / theta;
    const float sin_theta = sinf(theta);
    const float one_minus_cos = 1.0f - cosf(theta);
    const float a = one_minus_cos / theta;
    const float b = 1.0f - sin_theta / theta;
    const float da = (sin_theta - 2.0f * a) / theta2;
    const float db = (one_minus_cos - 3.0f * b) / theta2;
    float Kc[3];
    mat3_vec(K, c, Kc);
    float outer[9];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) outer[3 * row + col] = Kc[row] * omega[col];
    float term[9];
    for (int i = 0; i < 9; ++i) term[i] = db * K[i] - da * ((i == 0 || i == 4 || i == 8) ? 1.0f : 0.0f);
    float tmp0[9];
    mat3_mul(term, outer, tmp0);
    float skew_Kc[9];
    skew(Kc, skew_Kc);
    for (int i = 0; i < 9; ++i) tmp0[i] -= (b / theta) * skew_Kc[i];
    float aI_minus_bK[9];
    for (int i = 0; i < 9; ++i) aI_minus_bK[i] = a * ((i == 0 || i == 4 || i == 8) ? 1.0f : 0.0f) - b * K[i];
    float skew_c_over_theta[9];
    for (int i = 0; i < 9; ++i) skew_c_over_theta[i] = skew_c[i] / theta;
    float tmp1[9];
    mat3_mul(aI_minus_bK, skew_c_over_theta, tmp1);
    for (int i = 0; i < 9; ++i) tmp0[i] += tmp1[i];
    float result[9];
    mat3_mul(H_inv, tmp0, result);
    for (int i = 0; i < 9; ++i) d_hinv[i] = -result[i];
}

CUDAROBOTICS_LIE_HD static inline void se3_exp(const float* xi, float* T) {
    const float omega[3] = {xi[3], xi[4], xi[5]};
    const float rho[3] = {xi[0], xi[1], xi[2]};
    float R[9];
    float V[9];
    float t[3];
    so3_exp(omega, R);
    so3_left_jacobian(omega, V);
    mat3_vec(V, rho, t);

    mat4_identity(T);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) T[4 * r + c] = R[3 * r + c];
        T[4 * r + 3] = t[r];
    }
}

CUDAROBOTICS_LIE_HD static inline void se3_log(const float* T, float* xi) {
    float R[9];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) R[3 * r + c] = T[4 * r + c];
    }
    float omega[3];
    float V_inv[9];
    float rho[3];
    so3_log(R, omega);
    so3_left_jacobian_inverse(omega, V_inv);
    const float t[3] = {T[3], T[7], T[11]};
    mat3_vec(V_inv, t, rho);
    xi[0] = rho[0];
    xi[1] = rho[1];
    xi[2] = rho[2];
    xi[3] = omega[0];
    xi[4] = omega[1];
    xi[5] = omega[2];
}

CUDAROBOTICS_LIE_HD static inline void se2_vector_to_matrix(
    const float* xi, float* T) {
    se2_exp(xi, T);
}

CUDAROBOTICS_LIE_HD static inline void matrix_to_se2_vector(
    const float* T, float* xi) {
    se2_log(T, xi);
}

CUDAROBOTICS_LIE_HD static inline void se3_vector_to_matrix(
    const float* xi, float* T) {
    se3_exp(xi, T);
}

CUDAROBOTICS_LIE_HD static inline void matrix_to_se3_vector(
    const float* T, float* xi) {
    se3_log(T, xi);
}

CUDAROBOTICS_LIE_HD static inline void transform_point(const float* T,
                                                         const float* p,
                                                         float* out) {
    const float x = T[0] * p[0] + T[1] * p[1] + T[2] * p[2] + T[3];
    const float y = T[4] * p[0] + T[5] * p[1] + T[6] * p[2] + T[7];
    const float z = T[8] * p[0] + T[9] * p[1] + T[10] * p[2] + T[11];
    out[0] = x;
    out[1] = y;
    out[2] = z;
}

CUDAROBOTICS_LIE_HD static inline void inverse_transform_point(
    const float* T, const float* p, float* out) {
    const float d[3] = {p[0] - T[3], p[1] - T[7], p[2] - T[11]};
    out[0] = T[0] * d[0] + T[4] * d[1] + T[8] * d[2];
    out[1] = T[1] * d[0] + T[5] * d[1] + T[9] * d[2];
    out[2] = T[2] * d[0] + T[6] * d[1] + T[10] * d[2];
}

}  // namespace lie
}  // namespace cudarobotics

#undef CUDAROBOTICS_LIE_HD
