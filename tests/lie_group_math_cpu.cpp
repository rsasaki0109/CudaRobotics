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

}  // namespace

int main() {
    std::printf("=== test_lie_group_math ===\n");
    test_so3();
    test_se2();
    test_se3();
    std::printf("==========================\n");
    if (failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) FAILED.\n", failures);
    return 1;
}
