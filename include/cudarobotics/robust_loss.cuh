// Small host/device robust-loss primitives.
//
// The rho0/rho1/rho2 convention follows
// scomup/MathematicalRobotics/mathR/utilities/robust_kernel.py:
//   rho0 = rho(e^2), rho1 = d rho / d(e^2),
//   rho2 = d^2 rho / d(e^2)^2.
// The formulas are kept fixed-size and dependency-free for CUDA factors.
// The upstream MathematicalRobotics implementation is Copyright (c) 2022
// Yang Liu and MIT licensed.

#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define CUDAROBOTICS_ROBUST_HD __host__ __device__
#else
#define CUDAROBOTICS_ROBUST_HD
#endif

namespace cudarobotics {
namespace robust {

struct LossEvaluation {
    float rho0;
    float rho1;
    float rho2;
};

CUDAROBOTICS_ROBUST_HD static inline LossEvaluation l2(float squared_error) {
    LossEvaluation result;
    result.rho0 = squared_error;
    result.rho1 = 1.0f;
    result.rho2 = 0.0f;
    return result;
}

CUDAROBOTICS_ROBUST_HD static inline LossEvaluation huber(
    float squared_error, float delta) {
    const float safe_delta = fmaxf(delta, 1.0e-12f);
    const float delta2 = safe_delta * safe_delta;
    if (squared_error <= delta2) return l2(squared_error);

    const float error = sqrtf(fmaxf(squared_error, 1.0e-24f));
    LossEvaluation result;
    result.rho0 = 2.0f * safe_delta * error - delta2;
    result.rho1 = safe_delta / error;
    result.rho2 = -0.5f * result.rho1 / fmaxf(squared_error, 1.0e-24f);
    return result;
}

CUDAROBOTICS_ROBUST_HD static inline LossEvaluation pseudo_huber(
    float squared_error, float delta) {
    const float safe_delta = fmaxf(delta, 1.0e-12f);
    const float delta2 = safe_delta * safe_delta;
    const float aux1 = squared_error / delta2 + 1.0f;
    const float aux2 = sqrtf(aux1);
    LossEvaluation result;
    result.rho0 = 2.0f * delta2 * (aux2 - 1.0f);
    result.rho1 = 1.0f / aux2;
    result.rho2 = -0.5f * result.rho1 / (delta2 * aux1);
    return result;
}

CUDAROBOTICS_ROBUST_HD static inline LossEvaluation cauchy(
    float squared_error, float delta) {
    const float safe_delta = fmaxf(delta, 1.0e-12f);
    const float delta2 = safe_delta * safe_delta;
    const float aux = squared_error / delta2 + 1.0f;
    LossEvaluation result;
    result.rho0 = delta2 * logf(aux);
    result.rho1 = 1.0f / aux;
    result.rho2 = -(result.rho1 * result.rho1) / delta2;
    return result;
}

CUDAROBOTICS_ROBUST_HD static inline float weight(
    const LossEvaluation& evaluation) {
    return fmaxf(evaluation.rho1, 0.0f);
}

}  // namespace robust
}  // namespace cudarobotics

#undef CUDAROBOTICS_ROBUST_HD
