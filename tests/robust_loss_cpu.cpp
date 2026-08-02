#include <cmath>
#include <cstdio>

#include "cudarobotics/robust_loss.cuh"

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

void check_finite(const cudarobotics::robust::LossEvaluation& loss,
                 const char* name) {
    check(std::isfinite(loss.rho0) && std::isfinite(loss.rho1) &&
              std::isfinite(loss.rho2),
          name);
}

void test_l2_and_huber() {
    std::printf("[test_l2_and_huber]\n");
    const cudarobotics::robust::LossEvaluation l2 =
        cudarobotics::robust::l2(4.0f);
    check(std::fabs(l2.rho0 - 4.0f) < 1.0e-6f &&
              std::fabs(l2.rho1 - 1.0f) < 1.0e-6f && l2.rho2 == 0.0f,
          "L2 rho coefficients");

    const cudarobotics::robust::LossEvaluation huber_in =
        cudarobotics::robust::huber(1.0f, 2.0f);
    check(std::fabs(huber_in.rho0 - 1.0f) < 1.0e-6f &&
              std::fabs(huber_in.rho1 - 1.0f) < 1.0e-6f,
          "Huber quadratic branch");

    const cudarobotics::robust::LossEvaluation huber_out =
        cudarobotics::robust::huber(25.0f, 2.0f);
    check(std::fabs(huber_out.rho0 - 16.0f) < 1.0e-5f &&
              std::fabs(huber_out.rho1 - 0.4f) < 1.0e-5f &&
              huber_out.rho1 < 1.0f,
          "Huber linear branch and downweight");
    check_finite(huber_out, "Huber finite coefficients");
}

void test_smooth_losses() {
    std::printf("[test_smooth_losses]\n");
    const float small = 0.25f;
    const float large = 25.0f;
    const cudarobotics::robust::LossEvaluation ph_small =
        cudarobotics::robust::pseudo_huber(small, 2.0f);
    const cudarobotics::robust::LossEvaluation ph_large =
        cudarobotics::robust::pseudo_huber(large, 2.0f);
    const cudarobotics::robust::LossEvaluation c_small =
        cudarobotics::robust::cauchy(small, 2.0f);
    const cudarobotics::robust::LossEvaluation c_large =
        cudarobotics::robust::cauchy(large, 2.0f);
    check_finite(ph_small, "Pseudo-Huber finite at small residual");
    check_finite(ph_large, "Pseudo-Huber finite at large residual");
    check_finite(c_small, "Cauchy finite at small residual");
    check_finite(c_large, "Cauchy finite at large residual");
    check(ph_large.rho0 > ph_small.rho0 && ph_large.rho1 < ph_small.rho1,
          "Pseudo-Huber is monotone and downweights outliers");
    check(c_large.rho0 > c_small.rho0 && c_large.rho1 < c_small.rho1,
          "Cauchy is monotone and downweights outliers");
    check(cudarobotics::robust::weight(c_large) >= 0.0f &&
              cudarobotics::robust::weight(c_large) < 1.0f,
          "robust weight is bounded");
}

}  // namespace

int main() {
    std::printf("=== test_robust_loss ===\n");
    test_l2_and_huber();
    test_smooth_losses();
    std::printf("=======================\n");
    if (failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) FAILED.\n", failures);
    return 1;
}
