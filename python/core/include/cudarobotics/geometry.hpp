// Basic line/plane fitting and point residuals from mathR/robot_geometry.

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace cudarobotics {
namespace geometry {

struct Vec3 {
    float x;
    float y;
    float z;
};

inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 subtract(const Vec3& a, const Vec3& b) {
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3{a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x};
}

inline Vec3 scale(const Vec3& a, float value) {
    return Vec3{value * a.x, value * a.y, value * a.z};
}

inline float norm(const Vec3& a) { return sqrtf(dot(a, a)); }

inline bool normalize(Vec3* a) {
    if (a == nullptr) return false;
    const float length = norm(*a);
    if (length < 1.0e-12f) return false;
    *a = scale(*a, 1.0f / length);
    return true;
}

inline void eigen_symmetric_3x3(const float* input,
                                float* values,
                                float* vectors) {
    float A[9];
    for (int i = 0; i < 9; ++i) A[i] = input[i];
    for (int i = 0; i < 9; ++i) vectors[i] = 0.0f;
    vectors[0] = vectors[4] = vectors[8] = 1.0f;
    for (int iteration = 0; iteration < 32; ++iteration) {
        int p = 0;
        int q = 1;
        float largest = fabsf(A[1]);
        if (fabsf(A[2]) > largest) { p = 0; q = 2; largest = fabsf(A[2]); }
        if (fabsf(A[5]) > largest) { p = 1; q = 2; largest = fabsf(A[5]); }
        if (largest < 1.0e-8f) break;
        const float theta = 0.5f * atan2f(2.0f * A[3 * p + q],
                                           A[3 * p + p] - A[3 * q + q]);
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float app = A[3 * p + p];
        const float aqq = A[3 * q + q];
        const float apq = A[3 * p + q];
        A[3 * p + p] = c * c * app - 2.0f * s * c * apq + s * s * aqq;
        A[3 * q + q] = s * s * app + 2.0f * s * c * apq + c * c * aqq;
        A[3 * p + q] = A[3 * q + p] = 0.0f;
        for (int k = 0; k < 3; ++k) {
            if (k == p || k == q) continue;
            const float akp = A[3 * k + p];
            const float akq = A[3 * k + q];
            A[3 * k + p] = A[3 * p + k] = c * akp - s * akq;
            A[3 * k + q] = A[3 * q + k] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {
            const float vkp = vectors[3 * k + p];
            const float vkq = vectors[3 * k + q];
            vectors[3 * k + p] = c * vkp - s * vkq;
            vectors[3 * k + q] = s * vkp + c * vkq;
        }
    }
    for (int i = 0; i < 3; ++i) values[i] = A[3 * i + i];
    // Descending eigenvalues, matching numpy.argsort(...)[::-1].
    for (int i = 0; i < 2; ++i) {
        int best = i;
        for (int j = i + 1; j < 3; ++j) if (values[j] > values[best]) best = j;
        if (best == i) continue;
        std::swap(values[i], values[best]);
        for (int row = 0; row < 3; ++row) std::swap(vectors[3 * row + i], vectors[3 * row + best]);
    }
}

inline bool fit_line(const std::vector<Vec3>& points,
                     Vec3* center,
                     Vec3* direction,
                     float anisotropy = 3.0f) {
    if (points.size() < 2 || center == nullptr || direction == nullptr) return false;
    *center = Vec3{0.0f, 0.0f, 0.0f};
    for (const Vec3& point : points) {
        center->x += point.x;
        center->y += point.y;
        center->z += point.z;
    }
    const float inv_n = 1.0f / static_cast<float>(points.size());
    *center = scale(*center, inv_n);
    float covariance[9] = {};
    for (const Vec3& point : points) {
        const Vec3 d = subtract(point, *center);
        covariance[0] += d.x * d.x; covariance[1] += d.x * d.y; covariance[2] += d.x * d.z;
        covariance[3] += d.y * d.x; covariance[4] += d.y * d.y; covariance[5] += d.y * d.z;
        covariance[6] += d.z * d.x; covariance[7] += d.z * d.y; covariance[8] += d.z * d.z;
    }
    for (float& value : covariance) value *= inv_n;
    // Power iteration is sufficient here and avoids bringing a dynamic
    // eigensolver into the CUDA-facing header.  Deflation estimates the
    // second principal variance for the same v0 > 3*v1 test as the Python
    // reference.
    *direction = Vec3{1.0f, 0.37f, -0.21f};
    normalize(direction);
    for (int iteration = 0; iteration < 32; ++iteration) {
        Vec3 value{
            covariance[0] * direction->x + covariance[1] * direction->y + covariance[2] * direction->z,
            covariance[3] * direction->x + covariance[4] * direction->y + covariance[5] * direction->z,
            covariance[6] * direction->x + covariance[7] * direction->y + covariance[8] * direction->z};
        if (!normalize(&value)) break;
        *direction = value;
    }
    Vec3 principal = Vec3{
        covariance[0] * direction->x + covariance[1] * direction->y + covariance[2] * direction->z,
        covariance[3] * direction->x + covariance[4] * direction->y + covariance[5] * direction->z,
        covariance[6] * direction->x + covariance[7] * direction->y + covariance[8] * direction->z};
    const float largest = dot(*direction, principal);
    Vec3 orthogonal = fabsf(direction->x) < 0.8f
        ? cross(*direction, Vec3{1.0f, 0.0f, 0.0f})
        : cross(*direction, Vec3{0.0f, 1.0f, 0.0f});
    normalize(&orthogonal);
    for (int iteration = 0; iteration < 32; ++iteration) {
        const float projection = dot(*direction, orthogonal);
        orthogonal = subtract(orthogonal, scale(*direction, projection));
        const Vec3 value{
            covariance[0] * orthogonal.x + covariance[1] * orthogonal.y + covariance[2] * orthogonal.z,
            covariance[3] * orthogonal.x + covariance[4] * orthogonal.y + covariance[5] * orthogonal.z,
            covariance[6] * orthogonal.x + covariance[7] * orthogonal.y + covariance[8] * orthogonal.z};
        orthogonal = value;
        if (!normalize(&orthogonal)) break;
    }
    const Vec3 secondary{
        covariance[0] * orthogonal.x + covariance[1] * orthogonal.y + covariance[2] * orthogonal.z,
        covariance[3] * orthogonal.x + covariance[4] * orthogonal.y + covariance[5] * orthogonal.z,
        covariance[6] * orthogonal.x + covariance[7] * orthogonal.y + covariance[8] * orthogonal.z};
    const float second = fmaxf(0.0f, dot(orthogonal, secondary));
    return largest > anisotropy * second;
}

inline bool solve3(const float* A_in, const float* b_in, float* x) {
    float A[9];
    float b[3];
    for (int i = 0; i < 9; ++i) A[i] = A_in[i];
    for (int i = 0; i < 3; ++i) b[i] = b_in[i];
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 3; ++row)
            if (fabsf(A[3 * row + col]) > fabsf(A[3 * pivot + col])) pivot = row;
        if (fabsf(A[3 * pivot + col]) < 1.0e-12f) return false;
        if (pivot != col) {
            for (int j = col; j < 3; ++j) std::swap(A[3 * col + j], A[3 * pivot + j]);
            std::swap(b[col], b[pivot]);
        }
        for (int row = col + 1; row < 3; ++row) {
            const float factor = A[3 * row + col] / A[3 * col + col];
            for (int j = col; j < 3; ++j) A[3 * row + j] -= factor * A[3 * col + j];
            b[row] -= factor * b[col];
        }
    }
    for (int row = 2; row >= 0; --row) {
        float value = b[row];
        for (int col = row + 1; col < 3; ++col) value -= A[3 * row + col] * x[col];
        x[row] = value / A[3 * row + row];
    }
    return true;
}

inline bool fit_plane(const std::vector<Vec3>& points,
                      float* plane,
                      float max_distance = 0.2f) {
    if (points.size() < 3 || plane == nullptr) return false;
    float ata[9] = {};
    float atb[3] = {};
    for (const Vec3& point : points) {
        const float v[3] = {point.x, point.y, point.z};
        for (int row = 0; row < 3; ++row) {
            atb[row] -= v[row];
            for (int col = 0; col < 3; ++col) ata[3 * row + col] += v[row] * v[col];
        }
    }
    if (!solve3(ata, atb, plane)) return false;
    plane[3] = 1.0f;
    const float length = sqrtf(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
    if (length < 1.0e-12f) return false;
    for (int i = 0; i < 4; ++i) plane[i] /= length;
    float worst = 0.0f;
    for (const Vec3& point : points)
        worst = fmaxf(worst, fabsf(point.x * plane[0] + point.y * plane[1] + point.z * plane[2] + plane[3]));
    return worst <= max_distance;
}

inline float point_to_plane(const Vec3& point, const float* plane, float* jacobian = nullptr) {
    if (jacobian != nullptr) for (int i = 0; i < 3; ++i) jacobian[i] = plane[i];
    return point.x * plane[0] + point.y * plane[1] + point.z * plane[2] + plane[3];
}

inline float point_to_line(const Vec3& point,
                           const Vec3& center,
                           const Vec3& direction,
                           Vec3* jacobian = nullptr) {
    const Vec3 a = subtract(center, scale(direction, -0.1f));
    const Vec3 b = subtract(center, scale(direction, 0.1f));
    const Vec3 pa = subtract(a, point);
    const Vec3 pb = subtract(b, point);
    const Vec3 ab = subtract(b, a);
    const Vec3 pm = cross(pa, pb);
    const float ab_norm = norm(ab);
    const float pm_norm = norm(pm);
    if (ab_norm < 1.0e-12f || pm_norm < 1.0e-12f) {
        if (jacobian != nullptr) *jacobian = Vec3{0.0f, 0.0f, 0.0f};
        return 0.0f;
    }
    if (jacobian != nullptr) *jacobian = scale(cross(pm, ab), 1.0f / (pm_norm * ab_norm));
    return pm_norm / ab_norm;
}

}  // namespace geometry
}  // namespace cudarobotics
