// Lightweight 2-D IMLS primitives corresponding to mathR/imls/imls.py.
// The Python reference uses scipy.KDTree; this native version uses a bounded
// brute-force neighborhood so it has no external runtime dependency and is
// deterministic for small/medium scan lines.

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace cudarobotics {
namespace imls {

struct Point2f {
    float x;
    float y;
};

inline float norm2(const Point2f& p) { return p.x * p.x + p.y * p.y; }

inline bool find_normal(const std::vector<Point2f>& points,
                        Point2f* normal,
                        float anisotropy = 3.0f) {
    if (points.size() < 3 || normal == nullptr) return false;
    Point2f center{0.0f, 0.0f};
    for (const Point2f& point : points) {
        center.x += point.x;
        center.y += point.y;
    }
    const float inv_n = 1.0f / static_cast<float>(points.size());
    center.x *= inv_n;
    center.y *= inv_n;
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    for (const Point2f& point : points) {
        const float dx = point.x - center.x;
        const float dy = point.y - center.y;
        a += dx * dx;
        b += dx * dy;
        c += dy * dy;
    }
    a *= inv_n;
    b *= inv_n;
    c *= inv_n;
    const float trace = a + c;
    const float discriminant = sqrtf(fmaxf(0.0f, (a - c) * (a - c) + 4.0f * b * b));
    const float largest = 0.5f * (trace + discriminant);
    const float smallest = 0.5f * (trace - discriminant);
    // The normal is the minor eigenvector.  Use a stable branch when b is tiny.
    if (fabsf(b) > 1.0e-12f) {
        normal->x = b;
        normal->y = smallest - a;
    } else if (a < c) {
        *normal = Point2f{1.0f, 0.0f};
    } else {
        *normal = Point2f{0.0f, 1.0f};
    }
    const float length = sqrtf(norm2(*normal));
    if (length < 1.0e-12f || !(largest > anisotropy * smallest)) return false;
    normal->x /= length;
    normal->y /= length;
    return true;
}

inline void estimate_normals(const std::vector<Point2f>& points,
                             float radius,
                             std::vector<Point2f>* normals) {
    if (normals == nullptr) return;
    normals->assign(points.size(), Point2f{0.0f, 0.0f});
    const float radius2 = radius * radius;
    for (size_t i = 0; i < points.size(); ++i) {
        std::vector<Point2f> neighborhood;
        for (const Point2f& point : points) {
            const Point2f d{point.x - points[i].x, point.y - points[i].y};
            if (norm2(d) <= radius2) neighborhood.push_back(point);
        }
        find_normal(neighborhood, &(*normals)[i]);
    }
}

inline bool point_to_surface(const Point2f& query,
                             const std::vector<Point2f>& points,
                             const std::vector<Point2f>& normals,
                             float bandwidth,
                             float* distance,
                             Point2f* direction,
                             int neighbors = 3) {
    if (distance == nullptr || direction == nullptr || points.empty() ||
        points.size() != normals.size() || bandwidth <= 0.0f) return false;
    std::vector<std::pair<float, size_t> > candidates;
    candidates.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        const Point2f d{query.x - points[i].x, query.y - points[i].y};
        candidates.push_back(std::make_pair(norm2(d), i));
    }
    const size_t count = std::min(static_cast<size_t>(std::max(1, neighbors)), candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + count, candidates.end());
    float weight_sum = 0.0f;
    float projected_sum = 0.0f;
    Point2f direction_sum{0.0f, 0.0f};
    for (size_t k = 0; k < count; ++k) {
        const size_t i = candidates[k].second;
        const Point2f difference{query.x - points[i].x, query.y - points[i].y};
        const float sign_value = difference.x * normals[i].x + difference.y * normals[i].y;
        const float sign = sign_value > 0.0f ? 1.0f : (sign_value < 0.0f ? -1.0f : 0.0f);
        const float weight = expf(-candidates[k].first / (bandwidth * bandwidth));
        projected_sum += weight * sign_value * sign;
        direction_sum.x += weight * normals[i].x * sign;
        direction_sum.y += weight * normals[i].y * sign;
        weight_sum += weight;
    }
    if (weight_sum < 1.0e-12f) return false;
    *distance = projected_sum / weight_sum;
    direction->x = direction_sum.x / weight_sum;
    direction->y = direction_sum.y / weight_sum;
    const float length = sqrtf(norm2(*direction));
    if (length < 1.0e-12f) return false;
    direction->x /= length;
    direction->y /= length;
    return true;
}

}  // namespace imls
}  // namespace cudarobotics
