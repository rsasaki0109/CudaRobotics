#include <cmath>
#include <cstdio>

#include "cudarobotics/lie_group_math.cuh"

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (condition) {
        std::printf("  PASS: %s\n", name);
    } else {
        std::printf("  FAIL: %s\n", name);
        ++failures;
    }
}

float max_abs(const float* a, const float* b, int n) {
    float result = 0.0f;
    for (int i = 0; i < n; ++i) {
        result = fmaxf(result, fabsf(a[i] - b[i]));
    }
    return result;
}

float max_abs_identity3(const float* R) {
    float result = 0.0f;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float value = 0.0f;
            for (int k = 0; k < 3; ++k) {
                value += R[3 * k + r] * R[3 * k + c];
            }
            if (r == c) value -= 1.0f;
            result = fmaxf(result, fabsf(value));
        }
    }
    return result;
}

void test_so3() {
    std::printf("[test_so3]\n");
    const float cases[][3] = {
        {0.0f, 0.0f, 0.0f},
        {1.0e-4f, -2.0e-4f, 3.0e-4f},
        {0.2f, -0.4f, 0.3f},
        {0.0f, 0.0f, cudarobotics::lie::kPi - 1.0e-4f},
        {-0.7f, 0.4f, 2.1f},
    };
    for (const auto& input : cases) {
        float R[9];
        float output[3];
        cudarobotics::lie::so3_exp(input, R);
        cudarobotics::lie::so3_log(R, output);
        const float error = max_abs(input, output, 3);
        check(error < 2.0e-3f, "SO(3) exp/log round trip");
        check(max_abs_identity3(R) < 2.0e-5f, "SO(3) exponential is orthonormal");
    }
}

void test_se2() {
    std::printf("[test_se2]\n");
    const float cases[][3] = {
        {0.0f, 0.0f, 0.0f},
        {1.2f, -0.8f, 1.0e-4f},
        {-3.0f, 2.0f, -1.7f},
        {0.5f, 1.4f, cudarobotics::lie::kPi - 1.0e-4f},
    };
    for (const auto& input : cases) {
        float T[9];
        float output[3];
        cudarobotics::lie::se2_exp(input, T);
        cudarobotics::lie::se2_log(T, output);
        check(max_abs(input, output, 3) < 2.0e-3f,
              "SE(2) exp/log round trip");
    }
}

void test_se3() {
    std::printf("[test_se3]\n");
    const float cases[][6] = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {1.0f, -2.0f, 0.5f, 1.0e-4f, -2.0e-4f, 3.0e-4f},
        {-0.8f, 1.1f, 2.0f, 0.2f, -0.4f, 0.3f},
        {2.0f, -1.0f, 0.4f, 0.0f, 0.0f, cudarobotics::lie::kPi - 1.0e-4f},
    };
    for (const auto& input : cases) {
        float T[16];
        float output[6];
        float p[3] = {0.3f, -0.4f, 1.2f};
        float transformed[3];
        float recovered[3];
        cudarobotics::lie::se3_exp(input, T);
        cudarobotics::lie::se3_log(T, output);
        check(max_abs(input, output, 6) < 3.0e-3f,
              "SE(3) exp/log round trip");
        float R[9];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) R[3 * r + c] = T[4 * r + c];
        }
        check(max_abs_identity3(R) < 3.0e-5f,
              "SE(3) rotation is orthonormal");
        cudarobotics::lie::transform_point(T, p, transformed);
        cudarobotics::lie::inverse_transform_point(T, transformed, recovered);
        check(max_abs(p, recovered, 3) < 3.0e-5f,
              "SE(3) transform/inverse transform");
    }
}

void test_quaternion_and_jacobians() {
    std::printf("[test_quaternion_and_jacobians]\n");
    const float cases[][3] = {
        {0.0f, 0.0f, 0.0f},
        {1.0e-4f, -2.0e-4f, 3.0e-4f},
        {0.2f, -0.4f, 0.3f},
        {0.0f, 0.0f, cudarobotics::lie::kPi - 1.0e-4f},
    };
    for (const auto& omega : cases) {
        float R[9];
        float R_round_trip[9];
        float q[4];
        cudarobotics::lie::so3_exp(omega, R);
        cudarobotics::lie::mat3_to_quaternion(R, q);
        cudarobotics::lie::quaternion_to_mat3(q, R_round_trip);
        check(max_abs(R, R_round_trip, 9) < 3.0e-5f,
              "SO(3) matrix/quaternion round trip");
        const float q_norm = sqrtf(q[0] * q[0] + q[1] * q[1] +
                                   q[2] * q[2] + q[3] * q[3]);
        check(fabsf(q_norm - 1.0f) < 3.0e-5f,
              "quaternion conversion normalizes output");

        float q_scaled[4] = {2.5f * q[0], 2.5f * q[1], 2.5f * q[2],
                             2.5f * q[3]};
        float R_scaled[9];
        cudarobotics::lie::quaternion_to_mat3(q_scaled, R_scaled);
        check(max_abs(R, R_scaled, 9) < 3.0e-5f,
              "quaternion conversion accepts non-unit input");

        float J_left[9];
        float J_left_inv[9];
        float J_right[9];
        float J_right_inv[9];
        float product[9];
        cudarobotics::lie::so3_left_jacobian(omega, J_left);
        cudarobotics::lie::so3_left_jacobian_inverse(omega, J_left_inv);
        cudarobotics::lie::mat3_mul(J_left, J_left_inv, product);
        check(max_abs_identity3(product) < 4.0e-4f,
              "SO(3) left Jacobian inverse");
        cudarobotics::lie::so3_right_jacobian(omega, J_right);
        cudarobotics::lie::so3_right_jacobian_inverse(omega, J_right_inv);
        cudarobotics::lie::mat3_mul(J_right, J_right_inv, product);
        check(max_abs_identity3(product) < 4.0e-4f,
              "SO(3) right Jacobian inverse");
    }

    const float v[3] = {0.4f, -0.7f, 1.1f};
    float K[9];
    float recovered[3];
    cudarobotics::lie::skew(v, K);
    cudarobotics::lie::unskew(K, recovered);
    check(max_abs(v, recovered, 3) < 1.0e-6f,
          "skew/unskew round trip");

    const float derivative_omega[3] = {0.2f, -0.3f, 0.1f};
    const float derivative_v[3] = {0.4f, -0.7f, 1.1f};
    float analytic_d_hinv[9];
    cudarobotics::lie::d_hinv_so3(
        derivative_omega, derivative_v, analytic_d_hinv);
    float numeric_d_hinv[9] = {};
    const float h = 1.0e-4f;
    for (int col = 0; col < 3; ++col) {
        float omega_plus[3] = {derivative_omega[0], derivative_omega[1], derivative_omega[2]};
        float omega_minus[3] = {derivative_omega[0], derivative_omega[1], derivative_omega[2]};
        omega_plus[col] += h;
        omega_minus[col] -= h;
        float plus_matrix[9];
        float minus_matrix[9];
        float plus_value[3];
        float minus_value[3];
        cudarobotics::lie::d_log_so3(omega_plus, plus_matrix);
        cudarobotics::lie::d_log_so3(omega_minus, minus_matrix);
        cudarobotics::lie::mat3_vec(plus_matrix, derivative_v, plus_value);
        cudarobotics::lie::mat3_vec(minus_matrix, derivative_v, minus_value);
        for (int row = 0; row < 3; ++row)
            numeric_d_hinv[3 * row + col] =
                (plus_value[row] - minus_value[row]) / (2.0f * h);
    }
    check(max_abs(analytic_d_hinv, numeric_d_hinv, 9) < 3.0e-3f,
          "dHinvSO3 matches central derivative");
}

}  // namespace

int main() {
    std::printf("=== test_lie_group_math ===\n");
    test_so3();
    test_se2();
    test_se3();
    test_quaternion_and_jacobians();
    std::printf("==========================\n");
    if (failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) FAILED.\n", failures);
    return 1;
}
