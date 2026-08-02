// Fixed-size IMU preintegration and factor linearization.
//
// The navigation-state convention follows the documented implementation in
// scomup/MathematicalRobotics:
//   state = (R, p, v), tangent = (dtheta, dp, dv)
//   R maps body-frame vectors into the navigation/world frame.
//   IMU bias = (accelerometer bias, gyroscope bias).
//
// This is deliberately a small, dependency-free reference/core API.  It does
// not own a measurement buffer and does not depend on Eigen or SciPy, so the
// same fixed-size routines can be used by CPU reference code and CUDA factor
// linearization kernels.
//
// The mathematical formulas are derived from the documented preintegration
// implementation in scomup/MathematicalRobotics (Copyright (c) 2022 Yang Liu,
// MIT license). Keep the
// upstream attribution and license notice with redistributed substantial
// portions of this file.

#pragma once

#include <cmath>

#include "cudarobotics/lie_group_math.cuh"

#if defined(__CUDACC__)
#define CUDAROBOTICS_IMU_HD __host__ __device__
#else
#define CUDAROBOTICS_IMU_HD
#endif

namespace cudarobotics {
namespace imu {

constexpr int kNavTangentDim = 9;
constexpr int kBiasDim = 6;

struct ImuBias {
    float accel[3];
    float gyro[3];
};

struct NavDelta {
    float R[9];
    float p[3];
    float v[3];
};

struct NavState {
    float R[9];
    float p[3];
    float v[3];
};

struct ImuFactorLinearization {
    float residual[9];
    float J_state_i[81];
    float J_state_j[81];
    float J_bias_i[54];
};

CUDAROBOTICS_IMU_HD static inline void copy3(const float* src, float* dst) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
}

CUDAROBOTICS_IMU_HD static inline void copy9(const float* src, float* dst) {
    for (int i = 0; i < 9; ++i) dst[i] = src[i];
}

CUDAROBOTICS_IMU_HD static inline void copy27(const float* src, float* dst) {
    for (int i = 0; i < 27; ++i) dst[i] = src[i];
}

CUDAROBOTICS_IMU_HD static inline void copy81(const float* src, float* dst) {
    for (int i = 0; i < 81; ++i) dst[i] = src[i];
}

CUDAROBOTICS_IMU_HD static inline void copy54(const float* src, float* dst) {
    for (int i = 0; i < 54; ++i) dst[i] = src[i];
}

CUDAROBOTICS_IMU_HD static inline void zero3(float* x) {
    x[0] = 0.0f;
    x[1] = 0.0f;
    x[2] = 0.0f;
}

CUDAROBOTICS_IMU_HD static inline void zero9(float* x) {
    for (int i = 0; i < 9; ++i) x[i] = 0.0f;
}

CUDAROBOTICS_IMU_HD static inline void zero27(float* x) {
    for (int i = 0; i < 27; ++i) x[i] = 0.0f;
}

CUDAROBOTICS_IMU_HD static inline void zero54(float* x) {
    for (int i = 0; i < 54; ++i) x[i] = 0.0f;
}

CUDAROBOTICS_IMU_HD static inline void identity9(float* A) {
    for (int i = 0; i < 81; ++i) A[i] = 0.0f;
    A[0] = 1.0f;
    A[10] = 1.0f;
    A[20] = 1.0f;
    A[30] = 1.0f;
    A[40] = 1.0f;
    A[50] = 1.0f;
    A[60] = 1.0f;
    A[70] = 1.0f;
    A[80] = 1.0f;
}

CUDAROBOTICS_IMU_HD static inline void identity_bias(ImuBias* bias) {
    zero3(bias->accel);
    zero3(bias->gyro);
}

CUDAROBOTICS_IMU_HD static inline void identity_delta(NavDelta* delta) {
    cudarobotics::lie::mat3_identity(delta->R);
    zero3(delta->p);
    zero3(delta->v);
}

CUDAROBOTICS_IMU_HD static inline void identity_state(NavState* state) {
    cudarobotics::lie::mat3_identity(state->R);
    zero3(state->p);
    zero3(state->v);
}

CUDAROBOTICS_IMU_HD static inline void mat9_mul(const float* A,
                                                const float* B,
                                                float* C) {
    float out[81];
    for (int r = 0; r < 9; ++r) {
        for (int c = 0; c < 9; ++c) {
            float value = 0.0f;
            for (int k = 0; k < 9; ++k) {
                value += A[9 * r + k] * B[9 * k + c];
            }
            out[9 * r + c] = value;
        }
    }
    for (int i = 0; i < 81; ++i) C[i] = out[i];
}

CUDAROBOTICS_IMU_HD static inline void mat9_mul_9x3(const float* A,
                                                     const float* B,
                                                     float* C) {
    float out[27];
    for (int r = 0; r < 9; ++r) {
        for (int c = 0; c < 3; ++c) {
            float value = 0.0f;
            for (int k = 0; k < 9; ++k) {
                value += A[9 * r + k] * B[3 * k + c];
            }
            out[3 * r + c] = value;
        }
    }
    for (int i = 0; i < 27; ++i) C[i] = out[i];
}

CUDAROBOTICS_IMU_HD static inline void mat9_mul_9x6(const float* A,
                                                     const float* B,
                                                     float* C) {
    float out[54];
    for (int r = 0; r < 9; ++r) {
        for (int c = 0; c < 6; ++c) {
            float value = 0.0f;
            for (int k = 0; k < 9; ++k) {
                value += A[9 * r + k] * B[6 * k + c];
            }
            out[6 * r + c] = value;
        }
    }
    for (int i = 0; i < 54; ++i) C[i] = out[i];
}

CUDAROBOTICS_IMU_HD static inline void mat9_vec(const float* A,
                                                 const float* x,
                                                 float* y) {
    float out[9];
    for (int r = 0; r < 9; ++r) {
        float value = 0.0f;
        for (int c = 0; c < 9; ++c) value += A[9 * r + c] * x[c];
        out[r] = value;
    }
    for (int i = 0; i < 9; ++i) y[i] = out[i];
}

CUDAROBOTICS_IMU_HD static inline void mat9_add(const float* A,
                                                const float* B,
                                                float* C) {
    for (int i = 0; i < 81; ++i) C[i] = A[i] + B[i];
}

CUDAROBOTICS_IMU_HD static inline void mat9x3_sub(const float* A,
                                                  const float* B,
                                                  float* C) {
    for (int i = 0; i < 27; ++i) C[i] = A[i] - B[i];
}

CUDAROBOTICS_IMU_HD static inline void mat9x6_copy_from_9x3(
    const float* A, const float* B, float* C) {
    for (int r = 0; r < 9; ++r) {
        for (int c = 0; c < 3; ++c) C[6 * r + c] = A[3 * r + c];
        for (int c = 0; c < 3; ++c) C[6 * r + 3 + c] = B[3 * r + c];
    }
}

CUDAROBOTICS_IMU_HD static inline void tangent_to_delta(const float* x,
                                                         NavDelta* delta) {
    cudarobotics::lie::so3_exp(x, delta->R);
    for (int i = 0; i < 3; ++i) delta->p[i] = x[3 + i];
    for (int i = 0; i < 3; ++i) delta->v[i] = x[6 + i];
}

CUDAROBOTICS_IMU_HD static inline void delta_to_tangent(const NavDelta& delta,
                                                         float* x) {
    cudarobotics::lie::so3_log(delta.R, x);
    for (int i = 0; i < 3; ++i) x[3 + i] = delta.p[i];
    for (int i = 0; i < 3; ++i) x[6 + i] = delta.v[i];
}

CUDAROBOTICS_IMU_HD static inline void hso3(const float* omega,
                                             float* H) {
    const float theta2 = omega[0] * omega[0] +
                         omega[1] * omega[1] +
                         omega[2] * omega[2];
    float K[9];
    cudarobotics::lie::skew(omega, K);
    if (theta2 < 1.0e-8f) {
        cudarobotics::lie::mat3_identity(H);
        for (int i = 0; i < 9; ++i) H[i] -= 0.5f * K[i];
        return;
    }
    const float theta = sqrtf(theta2);
    float K_unit[9];
    float K2[9];
    for (int i = 0; i < 9; ++i) K_unit[i] = K[i] / theta;
    cudarobotics::lie::mat3_mul(K_unit, K_unit, K2);
    const float a = (1.0f - cosf(theta)) / theta;
    const float b = 1.0f - sinf(theta) / theta;
    cudarobotics::lie::mat3_identity(H);
    for (int i = 0; i < 9; ++i) H[i] += -a * K_unit[i] + b * K2[i];
}

CUDAROBOTICS_IMU_HD static inline void nav_state_retract(
    const NavState& state,
    const NavDelta& delta,
    NavState* result,
    float* J_state = nullptr,
    float* J_delta = nullptr) {
    cudarobotics::lie::mat3_mul(state.R, delta.R, result->R);
    float rotated[3];
    cudarobotics::lie::mat3_vec(state.R, delta.p, rotated);
    for (int i = 0; i < 3; ++i) result->p[i] = state.p[i] + rotated[i];
    cudarobotics::lie::mat3_vec(state.R, delta.v, rotated);
    for (int i = 0; i < 3; ++i) result->v[i] = state.v[i] + rotated[i];

    if (J_state != nullptr || J_delta != nullptr) {
        float R_delta_t[9] = {
            delta.R[0], delta.R[3], delta.R[6],
            delta.R[1], delta.R[4], delta.R[7],
            delta.R[2], delta.R[5], delta.R[8],
        };
        float minus_skew_p[9];
        float minus_skew_v[9];
        float skew_p[9];
        float skew_v[9];
        cudarobotics::lie::skew(delta.p, skew_p);
        cudarobotics::lie::skew(delta.v, skew_v);
        for (int i = 0; i < 9; ++i) {
            minus_skew_p[i] = -skew_p[i];
            minus_skew_v[i] = -skew_v[i];
        }
        if (J_state != nullptr) {
            identity9(J_state);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    J_state[9 * r + c] = R_delta_t[3 * r + c];
                    J_state[9 * (3 + r) + (3 + c)] = R_delta_t[3 * r + c];
                    J_state[9 * (6 + r) + (6 + c)] = R_delta_t[3 * r + c];
                }
            }
            float block[9];
            cudarobotics::lie::mat3_mul(R_delta_t, minus_skew_p, block);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    J_state[9 * (3 + r) + c] = block[3 * r + c];
            cudarobotics::lie::mat3_mul(R_delta_t, minus_skew_v, block);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    J_state[9 * (6 + r) + c] = block[3 * r + c];
        }
        if (J_delta != nullptr) identity9(J_delta);
    }
}

CUDAROBOTICS_IMU_HD static inline void nav_state_local(
    const NavState& self,
    const NavState& other,
    NavDelta* result,
    float* J_self = nullptr,
    float* J_other = nullptr) {
    float self_R_t[9] = {
        self.R[0], self.R[3], self.R[6],
        self.R[1], self.R[4], self.R[7],
        self.R[2], self.R[5], self.R[8],
    };
    cudarobotics::lie::mat3_mul(self_R_t, other.R, result->R);
    float dp_world[3] = {
        other.p[0] - self.p[0],
        other.p[1] - self.p[1],
        other.p[2] - self.p[2],
    };
    float dv_world[3] = {
        other.v[0] - self.v[0],
        other.v[1] - self.v[1],
        other.v[2] - self.v[2],
    };
    cudarobotics::lie::mat3_vec(self_R_t, dp_world, result->p);
    cudarobotics::lie::mat3_vec(self_R_t, dv_world, result->v);

    if (J_self != nullptr || J_other != nullptr) {
        if (J_self != nullptr) {
            for (int i = 0; i < 81; ++i) J_self[i] = 0.0f;
            for (int i = 0; i < 9; ++i) J_self[i * 9 + i] = -1.0f;
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    J_self[9 * r + c] = -result->R[3 * c + r];
                    J_self[9 * (3 + r) + (3 + c)] = -((r == c) ? 1.0f : 0.0f);
                    J_self[9 * (6 + r) + (6 + c)] = -((r == c) ? 1.0f : 0.0f);
                }
            }
            float skew_p[9];
            float skew_v[9];
            cudarobotics::lie::skew(result->p, skew_p);
            cudarobotics::lie::skew(result->v, skew_v);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    J_self[9 * (3 + r) + c] = skew_p[3 * r + c];
                    J_self[9 * (6 + r) + c] = skew_v[3 * r + c];
                }
            }
        }
        if (J_other != nullptr) {
            identity9(J_other);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    J_other[9 * (3 + r) + (3 + c)] = result->R[3 * r + c];
                    J_other[9 * (6 + r) + (6 + c)] = result->R[3 * r + c];
                }
            }
        }
    }
}

CUDAROBOTICS_IMU_HD static inline void delta_retract(
    const NavDelta& self,
    const NavDelta& other,
    NavDelta* result,
    float* J_self = nullptr,
    float* J_other = nullptr) {
    cudarobotics::lie::mat3_mul(self.R, other.R, result->R);
    for (int i = 0; i < 3; ++i) {
        result->p[i] = self.p[i] + other.p[i];
        result->v[i] = self.v[i] + other.v[i];
    }
    if (J_self != nullptr) {
        identity9(J_self);
        float other_R_t[9] = {
            other.R[0], other.R[3], other.R[6],
            other.R[1], other.R[4], other.R[7],
            other.R[2], other.R[5], other.R[8],
        };
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                J_self[9 * r + c] = other_R_t[3 * r + c];
    }
    if (J_other != nullptr) identity9(J_other);
}

CUDAROBOTICS_IMU_HD static inline void delta_local(
    const NavDelta& self,
    const NavDelta& other,
    NavDelta* result,
    float* J_self = nullptr,
    float* J_other = nullptr) {
    float self_R_t[9] = {
        self.R[0], self.R[3], self.R[6],
        self.R[1], self.R[4], self.R[7],
        self.R[2], self.R[5], self.R[8],
    };
    cudarobotics::lie::mat3_mul(self_R_t, other.R, result->R);
    for (int i = 0; i < 3; ++i) {
        result->p[i] = other.p[i] - self.p[i];
        result->v[i] = other.v[i] - self.v[i];
    }
    if (J_self != nullptr) {
        for (int i = 0; i < 81; ++i) J_self[i] = 0.0f;
        for (int i = 0; i < 9; ++i) J_self[9 * i + i] = -1.0f;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                J_self[9 * r + c] = -result->R[3 * c + r];
    }
    if (J_other != nullptr) identity9(J_other);
}

CUDAROBOTICS_IMU_HD static inline void delta_update(
    const NavDelta& old_delta,
    const float* acc_unbias,
    const float* gyro_unbias,
    float dt,
    NavDelta* next_delta,
    float* J_old = nullptr,
    float* J_acc = nullptr,
    float* J_gyro = nullptr) {
    float R_acc[3];
    cudarobotics::lie::mat3_vec(old_delta.R, acc_unbias, R_acc);
    float gyro_dt[3] = {
        gyro_unbias[0] * dt,
        gyro_unbias[1] * dt,
        gyro_unbias[2] * dt,
    };
    float R_increment[9];
    cudarobotics::lie::so3_exp(gyro_dt, R_increment);
    cudarobotics::lie::mat3_mul(old_delta.R, R_increment, next_delta->R);
    const float dt2 = dt * dt;
    for (int i = 0; i < 3; ++i) {
        next_delta->p[i] = old_delta.p[i] + old_delta.v[i] * dt +
                           0.5f * R_acc[i] * dt2;
        next_delta->v[i] = old_delta.v[i] + R_acc[i] * dt;
    }

    if (J_old != nullptr || J_acc != nullptr || J_gyro != nullptr) {
        float old_J[81];
        identity9(old_J);
        float R_minus_increment[9];
        const float negative_gyro_dt[3] = {
            -gyro_dt[0], -gyro_dt[1], -gyro_dt[2],
        };
        cudarobotics::lie::so3_exp(negative_gyro_dt, R_minus_increment);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                old_J[9 * r + c] = R_minus_increment[3 * r + c];

        float acc_skew[9];
        float R_acc_skew[9];
        cudarobotics::lie::skew(acc_unbias, acc_skew);
        cudarobotics::lie::mat3_mul(old_delta.R, acc_skew, R_acc_skew);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                old_J[9 * (3 + r) + c] = -R_acc_skew[3 * r + c] * 0.5f * dt2;
                old_J[9 * (6 + r) + c] = -R_acc_skew[3 * r + c] * dt;
            }
            old_J[9 * (3 + r) + (6 + r)] = dt;
        }

        float acc_J[27];
        zero27(acc_J);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                acc_J[3 * (3 + r) + c] = old_delta.R[3 * r + c] * 0.5f * dt2;
                acc_J[3 * (6 + r) + c] = old_delta.R[3 * r + c] * dt;
            }
        }
        float gyro_J[27];
        zero27(gyro_J);
        hso3(gyro_dt, gyro_J);
        // HSO3 is the derivative with respect to the integrated rotation
        // increment.  The factor residual is parameterized by the gyro
        // measurement, so apply d(gyro * dt) / d(gyro) = dt here.
        for (int i = 0; i < 9; ++i) gyro_J[i] *= dt;

        if (J_old != nullptr) copy81(old_J, J_old);
        if (J_acc != nullptr) copy27(acc_J, J_acc);
        if (J_gyro != nullptr) copy27(gyro_J, J_gyro);
    }
}

struct ImuPreintegrator {
    NavDelta delta;
    float total_dt;
    float gravity[3];
    float R_imu_to_body[9];
    float lever_arm[3];
    ImuBias linearization_bias;
    float J_delta_accel_bias[27];
    float J_delta_gyro_bias[27];

    CUDAROBOTICS_IMU_HD void reset_with_calibration(
        const float* gravity_n,
        const ImuBias& bias,
        const float* R_imu_to_body,
        const float* lever_arm) {
        identity_delta(&delta);
        total_dt = 0.0f;
        copy3(gravity_n, gravity);
        copy9(R_imu_to_body, this->R_imu_to_body);
        copy3(lever_arm, this->lever_arm);
        linearization_bias = bias;
        zero27(J_delta_accel_bias);
        zero27(J_delta_gyro_bias);
    }

    CUDAROBOTICS_IMU_HD void reset(float gravity_magnitude,
                                    const ImuBias& bias) {
        const float gravity_n[3] = {0.0f, 0.0f, -gravity_magnitude};
        const float identity[9] = {
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
        };
        const float zero_lever[3] = {0.0f, 0.0f, 0.0f};
        reset_with_calibration(gravity_n, bias, identity, zero_lever);
    }

    CUDAROBOTICS_IMU_HD void update(const float* acc_i,
                                     const float* gyro_i,
                                     float dt) {
        float acc[3];
        float gyro[3];
        cudarobotics::lie::mat3_vec(R_imu_to_body, acc_i, acc);
        cudarobotics::lie::mat3_vec(R_imu_to_body, gyro_i, gyro);
        float gyro_skew[9];
        float gyro_skew2[9];
        float centripetal[3];
        cudarobotics::lie::skew(gyro, gyro_skew);
        cudarobotics::lie::mat3_mul(gyro_skew, gyro_skew, gyro_skew2);
        cudarobotics::lie::mat3_vec(gyro_skew2, lever_arm, centripetal);
        for (int i = 0; i < 3; ++i) {
            acc[i] -= centripetal[i];
            acc[i] -= linearization_bias.accel[i];
            gyro[i] -= linearization_bias.gyro[i];
        }

        NavDelta next;
        float J_old[81];
        float J_acc[27];
        float J_gyro[27];
        delta_update(delta, acc, gyro, dt, &next, J_old, J_acc, J_gyro);
        float new_J_accel_bias[27];
        float new_J_gyro_bias[27];
        mat9_mul_9x3(J_old, J_delta_accel_bias, new_J_accel_bias);
        mat9_mul_9x3(J_old, J_delta_gyro_bias, new_J_gyro_bias);
        mat9x3_sub(new_J_accel_bias, J_acc, J_delta_accel_bias);
        mat9x3_sub(new_J_gyro_bias, J_gyro, J_delta_gyro_bias);
        delta = next;
        total_dt += dt;
    }

    CUDAROBOTICS_IMU_HD void bias_correct(
        const ImuBias& bias,
        NavDelta* corrected,
        float* J_bias = nullptr) const {
        float bias_delta[6];
        for (int i = 0; i < 3; ++i) {
            bias_delta[i] = bias.accel[i] - linearization_bias.accel[i];
            bias_delta[3 + i] = bias.gyro[i] - linearization_bias.gyro[i];
        }
        float correction_tangent[9];
        for (int r = 0; r < 9; ++r) {
            float a = 0.0f;
            float g = 0.0f;
            for (int c = 0; c < 3; ++c) {
                a += J_delta_accel_bias[3 * r + c] * bias_delta[c];
                g += J_delta_gyro_bias[3 * r + c] * bias_delta[3 + c];
            }
            correction_tangent[r] = a + g;
        }
        NavDelta correction;
        tangent_to_delta(correction_tangent, &correction);
        delta_retract(delta, correction, corrected);
        if (J_bias != nullptr) {
            mat9x6_copy_from_9x3(J_delta_accel_bias,
                                 J_delta_gyro_bias,
                                 J_bias);
        }
    }

    CUDAROBOTICS_IMU_HD void calc_delta(
        const NavDelta& corrected,
        const NavState& state,
        NavDelta* result,
        float* J_xi = nullptr,
        float* J_state = nullptr) const {
        float state_R_t[9] = {
            state.R[0], state.R[3], state.R[6],
            state.R[1], state.R[4], state.R[7],
            state.R[2], state.R[5], state.R[8],
        };
        float Rv[3];
        float Rg[3];
        cudarobotics::lie::mat3_vec(state_R_t, state.v, Rv);
        cudarobotics::lie::mat3_vec(state_R_t, gravity, Rg);
        result->R[0] = corrected.R[0]; result->R[1] = corrected.R[1]; result->R[2] = corrected.R[2];
        result->R[3] = corrected.R[3]; result->R[4] = corrected.R[4]; result->R[5] = corrected.R[5];
        result->R[6] = corrected.R[6]; result->R[7] = corrected.R[7]; result->R[8] = corrected.R[8];
        const float dt2 = total_dt * total_dt;
        for (int i = 0; i < 3; ++i) {
            result->p[i] = corrected.p[i] + total_dt * Rv[i] + 0.5f * dt2 * Rg[i];
            result->v[i] = corrected.v[i] + total_dt * Rg[i];
        }
        if (J_xi != nullptr) identity9(J_xi);
        if (J_state != nullptr) {
            for (int i = 0; i < 81; ++i) J_state[i] = 0.0f;
            float skew_Rv[9];
            float skew_Rg[9];
            cudarobotics::lie::skew(Rv, skew_Rv);
            cudarobotics::lie::skew(Rg, skew_Rg);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    J_state[9 * (3 + r) + c] = total_dt * skew_Rv[3 * r + c] +
                                                0.5f * dt2 * skew_Rg[3 * r + c];
                    J_state[9 * (6 + r) + c] = total_dt * skew_Rg[3 * r + c];
                    J_state[9 * (3 + r) + (6 + c)] = (r == c) ? total_dt : 0.0f;
                }
            }
        }
    }

    CUDAROBOTICS_IMU_HD void predict(
        const NavState& state_i,
        const ImuBias& bias,
        NavState* state_j,
        float* J_state = nullptr,
        float* J_bias = nullptr) const {
        NavDelta corrected;
        float J_xi_bias[54];
        bias_correct(bias, &corrected, J_xi_bias);
        NavDelta delta_with_state;
        float J_delta_xi[81];
        float J_delta_state[81];
        calc_delta(corrected, state_i, &delta_with_state,
                   J_delta_xi, J_delta_state);
        float J_retract_state[81];
        float J_retract_delta[81];
        nav_state_retract(state_i, delta_with_state, state_j,
                          J_retract_state, J_retract_delta);
        if (J_state != nullptr) {
            float propagated[81];
            mat9_mul(J_retract_delta, J_delta_state, propagated);
            mat9_add(J_retract_state, propagated, J_state);
        }
        if (J_bias != nullptr) {
            mat9_mul_9x6(J_retract_delta, J_xi_bias, J_bias);
        }
    }
};

CUDAROBOTICS_IMU_HD static inline void linearize_imu_factor(
    const ImuPreintegrator& preintegrator,
    const NavState& state_i,
    const NavState& state_j,
    const ImuBias& bias,
    ImuFactorLinearization* linearization) {
    NavState predicted;
    float J_pred_state[81];
    float J_pred_bias[54];
    preintegrator.predict(state_i, bias, &predicted,
                          J_pred_state, J_pred_bias);

    NavDelta residual_delta;
    float J_actual[81];
    float J_predicted[81];
    nav_state_local(state_j, predicted, &residual_delta,
                    J_actual, J_predicted);
    delta_to_tangent(residual_delta, linearization->residual);
    mat9_mul(J_predicted, J_pred_state, linearization->J_state_i);
    for (int i = 0; i < 81; ++i) linearization->J_state_j[i] = J_actual[i];
    mat9_mul_9x6(J_predicted, J_pred_bias, linearization->J_bias_i);
}

}  // namespace imu
}  // namespace cudarobotics

#undef CUDAROBOTICS_IMU_HD
