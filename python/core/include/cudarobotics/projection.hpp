// Camera projection and SE(3) perturbation helpers from mathR/slam/projection.py.

#pragma once

#include <cmath>

#include "cudarobotics/lie_group_math.cuh"

#if defined(__CUDACC__)
#define CUDAROBOTICS_PROJ_HD __host__ __device__
#else
#define CUDAROBOTICS_PROJ_HD
#endif

namespace cudarobotics {
namespace projection {

CUDAROBOTICS_PROJ_HD inline void mat4_multiply(const float* A, const float* B, float* C) {
    float result[16];
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float value = 0.0f;
            for (int k = 0; k < 4; ++k) value += A[4 * row + k] * B[4 * k + col];
            result[4 * row + col] = value;
        }
    }
    for (int i = 0; i < 16; ++i) C[i] = result[i];
}

CUDAROBOTICS_PROJ_HD inline void rigid_inverse(const float* T, float* inverse) {
    lie::mat4_identity(inverse);
    float R[9];
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            R[3 * row + col] = T[4 * row + col];
            inverse[4 * row + col] = T[4 * col + row];
        }
    }
    const float t[3] = {T[3], T[7], T[11]};
    float inverse_t[3];
    lie::mat3_transpose_vec(R, t, inverse_t);
    inverse[3] = -inverse_t[0];
    inverse[7] = -inverse_t[1];
    inverse[11] = -inverse_t[2];
}

CUDAROBOTICS_PROJ_HD inline void transform(const float* pose_or_matrix,
                      bool pose_is_vector,
                      const float* point,
                      float* result,
                      float* d_pose = nullptr,
                      float* d_point = nullptr) {
    float T[16];
    if (pose_is_vector) lie::se3_exp(pose_or_matrix, T);
    else for (int i = 0; i < 16; ++i) T[i] = pose_or_matrix[i];
    const float p[3] = {point[0], point[1], point[2]};
    float R[9];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) R[3 * row + col] = T[4 * row + col];
    lie::mat3_vec(R, p, result);
    result[0] += T[3];
    result[1] += T[7];
    result[2] += T[11];
    if (d_pose != nullptr) {
        float R[9];
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) R[3 * row + col] = T[4 * row + col];
        float minus_skew_p[9];
        float pose_rotation_block[9];
        float skew_p[9];
        lie::skew(p, skew_p);
        for (int i = 0; i < 9; ++i) minus_skew_p[i] = -skew_p[i];
        lie::mat3_mul(R, minus_skew_p, pose_rotation_block);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                d_pose[3 * row + col] = R[3 * row + col];
                d_pose[3 * row + 3 + col] = pose_rotation_block[3 * row + col];
            }
        }
    }
    if (d_point != nullptr) {
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) d_point[3 * row + col] = T[4 * row + col];
    }
}

CUDAROBOTICS_PROJ_HD inline void transform_inverse(const float* pose_or_matrix,
                              bool pose_is_vector,
                              const float* point,
                              float* result,
                              float* d_pose = nullptr,
                              float* d_point = nullptr) {
    float T[16];
    if (pose_is_vector) lie::se3_exp(pose_or_matrix, T);
    else for (int i = 0; i < 16; ++i) T[i] = pose_or_matrix[i];
    float inverse[16];
    rigid_inverse(T, inverse);
    const float p[3] = {point[0], point[1], point[2]};
    float R_inverse[9];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) R_inverse[3 * row + col] = inverse[4 * row + col];
    lie::mat3_vec(R_inverse, p, result);
    result[0] += inverse[3];
    result[1] += inverse[7];
    result[2] += inverse[11];
    if (d_pose != nullptr) {
        float skew_result[9];
        lie::skew(result, skew_result);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                d_pose[3 * row + col] = row == col ? -1.0f : 0.0f;
                d_pose[3 * row + 3 + col] = skew_result[3 * row + col];
            }
        }
    }
    if (d_point != nullptr) {
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) d_point[3 * row + col] = inverse[4 * row + col];
    }
}

CUDAROBOTICS_PROJ_HD inline bool reproject(const float* point_camera,
                      const float* K,
                      float* pixel,
                      float* d_pixel_d_point = nullptr) {
    const float x = point_camera[0];
    const float y = point_camera[1];
    const float z = point_camera[2];
    if (fabsf(z) < 1.0e-12f) return false;
    const float z2 = z * z;
    pixel[0] = K[0] * x / z + K[2];
    pixel[1] = K[4] * y / z + K[5];
    if (d_pixel_d_point != nullptr) {
        d_pixel_d_point[0] = K[0] / z;
        d_pixel_d_point[1] = 0.0f;
        d_pixel_d_point[2] = -K[0] * x / z2;
        d_pixel_d_point[3] = 0.0f;
        d_pixel_d_point[4] = K[4] / z;
        d_pixel_d_point[5] = -K[4] * y / z2;
    }
    return true;
}

CUDAROBOTICS_PROJ_HD inline bool reprojection_error(const float* T_wc,
                               const float* point_world,
                               const float* measurement,
                               const float* K,
                               float* residual,
                               float* d_pose = nullptr,
                               float* d_point = nullptr) {
    float point_camera[3];
    float d_point_camera_d_pose[18];
    float d_point_camera_d_point[9];
    transform_inverse(T_wc, false, point_world, point_camera,
                      d_point_camera_d_pose, d_point_camera_d_point);
    float pixel[2];
    float d_pixel_d_point[6];
    if (!reproject(point_camera, K, pixel, d_pixel_d_point)) return false;
    residual[0] = pixel[0] - measurement[0];
    residual[1] = pixel[1] - measurement[1];
    if (d_pose != nullptr) {
        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 6; ++col) {
                d_pose[6 * row + col] = 0.0f;
                for (int k = 0; k < 3; ++k)
                    d_pose[6 * row + col] += d_pixel_d_point[3 * row + k] * d_point_camera_d_pose[6 * k + col];
            }
        }
    }
    if (d_point != nullptr) {
        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 3; ++col) {
                d_point[3 * row + col] = 0.0f;
                for (int k = 0; k < 3; ++k)
                    d_point[3 * row + col] += d_pixel_d_point[3 * row + k] * d_point_camera_d_point[3 * k + col];
            }
        }
    }
    return true;
}

// The upstream projection module exposes both T_cw (world-to-camera) and
// T_wc (camera-to-world) residual conventions.  Keep the former explicit so
// callers cannot accidentally invert a pose twice.
CUDAROBOTICS_PROJ_HD inline bool reprojection_error_cw(
    const float* T_cw,
    const float* point_world,
    const float* measurement,
    const float* K,
    float* residual,
    float* d_pose = nullptr,
    float* d_point = nullptr) {
    float point_camera[3];
    float d_point_camera_d_pose[18];
    float d_point_camera_d_point[9];
    transform(T_cw, false, point_world, point_camera,
              d_point_camera_d_pose, d_point_camera_d_point);
    float pixel[2];
    float d_pixel_d_point[6];
    if (!reproject(point_camera, K, pixel, d_pixel_d_point)) return false;
    residual[0] = pixel[0] - measurement[0];
    residual[1] = pixel[1] - measurement[1];
    if (d_pose != nullptr) {
        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 6; ++col) {
                d_pose[6 * row + col] = 0.0f;
                for (int k = 0; k < 3; ++k)
                    d_pose[6 * row + col] += d_pixel_d_point[3 * row + k] *
                        d_point_camera_d_pose[6 * k + col];
            }
        }
    }
    if (d_point != nullptr) {
        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 3; ++col) {
                d_point[3 * row + col] = 0.0f;
                for (int k = 0; k < 3; ++k)
                    d_point[3 * row + col] += d_pixel_d_point[3 * row + k] *
                        d_point_camera_d_point[3 * k + col];
            }
        }
    }
    return true;
}

CUDAROBOTICS_PROJ_HD inline bool reprojection_error_with_body_camera(
    const float* T_wb,
    const float* T_bc,
    const float* point_world,
    const float* measurement,
    const float* K,
    float* residual) {
    float T_wc[16];
    mat4_multiply(T_wb, T_bc, T_wc);
    return reprojection_error(T_wc, point_world, measurement, K, residual);
}

CUDAROBOTICS_PROJ_HD inline void camera_prior_factor(
    const float* T,
    const float* measurement,
    float* residual,
    float* jacobian = nullptr) {
    float measurement_inverse[16];
    float error[16];
    rigid_inverse(measurement, measurement_inverse);
    mat4_multiply(measurement_inverse, T, error);
    lie::se3_log(error, residual);
    if (jacobian != nullptr) {
        for (int i = 0; i < 36; ++i) jacobian[i] = 0.0f;
        for (int i = 0; i < 6; ++i) jacobian[6 * i + i] = 1.0f;
    }
}

CUDAROBOTICS_PROJ_HD inline void camera_between_residual_core(
    const float* T_from,
    const float* T_to,
    const float* measurement,
    float* residual) {
    float from_inverse[16];
    float relative[16];
    float measurement_inverse[16];
    float error[16];
    rigid_inverse(T_from, from_inverse);
    mat4_multiply(from_inverse, T_to, relative);
    rigid_inverse(measurement, measurement_inverse);
    mat4_multiply(measurement_inverse, relative, error);
    lie::se3_log(error, residual);
}

CUDAROBOTICS_PROJ_HD inline void camera_between_factor(
    const float* T_from,
    const float* T_to,
    const float* measurement,
    float* residual,
    float* jacobian_from = nullptr,
    float* jacobian_to = nullptr) {
    camera_between_residual_core(T_from, T_to, measurement, residual);
    // The Python edge supplies a first-order Jacobian for this factor.  The
    // residual passes through log(SE(3)), so use the same central-difference
    // convention as PoseGraph6 here to keep the native factor numerically
    // consistent across the full tangent range.
    const float h = 1.0e-4f;
    if (jacobian_from != nullptr) {
        for (int i = 0; i < 36; ++i) jacobian_from[i] = 0.0f;
        for (int col = 0; col < 6; ++col) {
            float plus_step[6] = {};
            float minus_step[6] = {};
            plus_step[col] = h;
            minus_step[col] = -h;
            float plus_increment[16];
            float minus_increment[16];
            float from_plus[16];
            float from_minus[16];
            float residual_plus[6];
            float residual_minus[6];
            lie::se3_exp(plus_step, plus_increment);
            mat4_multiply(T_from, plus_increment, from_plus);
            lie::se3_exp(minus_step, minus_increment);
            mat4_multiply(T_from, minus_increment, from_minus);
            camera_between_residual_core(from_plus, T_to, measurement, residual_plus);
            camera_between_residual_core(from_minus, T_to, measurement, residual_minus);
            for (int row = 0; row < 6; ++row)
                jacobian_from[6 * row + col] =
                    (residual_plus[row] - residual_minus[row]) / (2.0f * h);
        }
    }
    if (jacobian_to != nullptr) {
        for (int i = 0; i < 36; ++i) jacobian_to[i] = 0.0f;
        for (int col = 0; col < 6; ++col) {
            float plus_step[6] = {};
            float minus_step[6] = {};
            plus_step[col] = h;
            minus_step[col] = -h;
            float plus_increment[16];
            float minus_increment[16];
            float to_plus[16];
            float to_minus[16];
            float residual_plus[6];
            float residual_minus[6];
            lie::se3_exp(plus_step, plus_increment);
            mat4_multiply(T_to, plus_increment, to_plus);
            lie::se3_exp(minus_step, minus_increment);
            mat4_multiply(T_to, minus_increment, to_minus);
            camera_between_residual_core(T_from, to_plus, measurement, residual_plus);
            camera_between_residual_core(T_from, to_minus, measurement, residual_minus);
            for (int row = 0; row < 6; ++row)
                jacobian_to[6 * row + col] =
                    (residual_plus[row] - residual_minus[row]) / (2.0f * h);
        }
    }
}

CUDAROBOTICS_PROJ_HD inline void point_prior_factor(
    const float* point,
    const float* measurement,
    float* residual,
    float* jacobian = nullptr) {
    for (int i = 0; i < 3; ++i) residual[i] = point[i] - measurement[i];
    if (jacobian != nullptr) {
        for (int i = 0; i < 9; ++i) jacobian[i] = 0.0f;
        for (int i = 0; i < 3; ++i) jacobian[3 * i + i] = 1.0f;
    }
}

CUDAROBOTICS_PROJ_HD inline void pose_plus(const float* x1, const float* x2, float* result) {
    float T1[16];
    float T2[16];
    float T3[16];
    lie::se3_exp(x1, T1);
    lie::se3_exp(x2, T2);
    mat4_multiply(T1, T2, T3);
    lie::se3_log(T3, result);
}

CUDAROBOTICS_PROJ_HD inline void pose_minus(const float* x1, const float* x2, float* result) {
    float T1[16];
    float T2[16];
    float T2_inverse[16];
    float T[16];
    lie::se3_exp(x1, T1);
    lie::se3_exp(x2, T2);
    rigid_inverse(T2, T2_inverse);
    mat4_multiply(T2_inverse, T1, T);
    lie::se3_log(T, result);
}

CUDAROBOTICS_PROJ_HD inline void pose_inverse(const float* x, float* result) {
    float T[16];
    float T_inverse[16];
    lie::se3_exp(x, T);
    rigid_inverse(T, T_inverse);
    lie::se3_log(T_inverse, result);
}

CUDAROBOTICS_PROJ_HD inline void undistort_point(const float* pixel,
                            const float* K,
                            const float* distortion,
                            float* result) {
    const float x = (pixel[0] - K[2]) / K[0];
    const float y = (pixel[1] - K[5]) / K[4];
    const float r2 = x * x + y * y;
    const float radial = 1.0f + distortion[0] * r2 + distortion[1] * r2 * r2;
    result[0] = K[0] * x * radial + K[2];
    result[1] = K[4] * y * radial + K[5];
}

}  // namespace projection
}  // namespace cudarobotics

#undef CUDAROBOTICS_PROJ_HD
