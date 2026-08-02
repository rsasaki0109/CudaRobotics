// Array-based equivalents of the convenience functions in mathR's
// utilities/math_tools.py.  The Lie primitives themselves live in
// lie_group_math.cuh; this header supplies the v2m/p2m/makeRt/transform
// adapters used by the upstream examples.

#pragma once

#include "cudarobotics/lie_group_math.cuh"

namespace cudarobotics {
namespace math {

inline void v2m(const float* v, float* T) { lie::se2_exp(v, T); }

inline void m2v(const float* T, float* v) { lie::se2_log(T, v); }

inline void p2m(const float* x, float* T) {
    float R[9];
    lie::so3_exp(x + 3, R);
    lie::mat4_identity(T);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) T[4 * row + col] = R[3 * row + col];
        T[4 * row + 3] = x[row];
    }
}

inline void m2p(const float* T, float* x) {
    x[0] = T[3];
    x[1] = T[7];
    x[2] = T[11];
    float R[9];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) R[3 * row + col] = T[4 * row + col];
    lie::so3_log(R, x + 3);
}

inline void makeT2(const float* R, const float* t, float* T) {
    T[0] = R[0]; T[1] = R[1]; T[2] = t[0];
    T[3] = R[2]; T[4] = R[3]; T[5] = t[1];
    T[6] = 0.0f; T[7] = 0.0f; T[8] = 1.0f;
}

inline void makeT3(const float* R, const float* t, float* T) {
    lie::mat4_identity(T);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) T[4 * row + col] = R[3 * row + col];
        T[4 * row + 3] = t[row];
    }
}

inline void makeRt2(const float* T, float* R, float* t) {
    R[0] = T[0]; R[1] = T[1]; R[2] = T[3]; R[3] = T[4];
    t[0] = T[2]; t[1] = T[5];
}

inline void makeRt3(const float* T, float* R, float* t) {
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) R[3 * row + col] = T[4 * row + col];
        t[row] = T[4 * row + 3];
    }
}

inline void transform2d(const float* T,
                        const float* points,
                        int point_count,
                        float* output) {
    float R[4];
    float t[2];
    makeRt2(T, R, t);
    for (int i = 0; i < point_count; ++i) {
        output[2 * i + 0] = R[0] * points[2 * i + 0] + R[1] * points[2 * i + 1] + t[0];
        output[2 * i + 1] = R[2] * points[2 * i + 0] + R[3] * points[2 * i + 1] + t[1];
    }
}

inline void transform3d(const float* T,
                        const float* points,
                        int point_count,
                        float* output) {
    float R[9];
    float t[3];
    makeRt3(T, R, t);
    for (int i = 0; i < point_count; ++i) {
        float p[3] = {points[3 * i + 0], points[3 * i + 1], points[3 * i + 2]};
        float q[3];
        lie::mat3_vec(R, p, q);
        for (int k = 0; k < 3; ++k) output[3 * i + k] = q[k] + t[k];
    }
}

}  // namespace math
}  // namespace cudarobotics
