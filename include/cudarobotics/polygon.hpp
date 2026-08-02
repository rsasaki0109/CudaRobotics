// Polygon containment and signed push-out vectors from mathR/utilities/polygon.py.

#pragma once

#include <cmath>
#include <vector>

namespace cudarobotics {
namespace polygon {

struct Point2f {
    float x;
    float y;
};

inline bool point_inside(const Point2f& point,
                         const std::vector<Point2f>& polygon,
                         bool include_edges = true) {
    if (polygon.size() < 3) return false;
    const float eps = 1.0e-6f;
    bool inside = false;
    for (size_t i = 0; i < polygon.size(); ++i) {
        const Point2f a = polygon[i];
        const Point2f b = polygon[(i + 1) % polygon.size()];
        const float cross = (b.x - a.x) * (point.y - a.y) -
                            (b.y - a.y) * (point.x - a.x);
        const float dot = (point.x - a.x) * (point.x - b.x) +
                          (point.y - a.y) * (point.y - b.y);
        if (fabsf(cross) <= eps && dot <= eps) return include_edges;
        const bool crosses = ((a.y > point.y) != (b.y > point.y));
        if (crosses) {
            const float x_intersection =
                a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if (point.x < x_intersection) inside = !inside;
        }
    }
    return inside;
}

inline Point2f polygon_residual(const Point2f& point,
                                const std::vector<Point2f>& polygon,
                                float threshold = 1.0f) {
    Point2f result{0.0f, 0.0f};
    if (polygon.size() < 3 || threshold <= 0.0f) return result;
    float best_distance = 1.0e30f;
    Point2f best_vector{0.0f, 0.0f};
    for (size_t i = 0; i < polygon.size(); ++i) {
        const Point2f a = polygon[i];
        const Point2f b = polygon[(i + 1) % polygon.size()];
        const float ex = b.x - a.x;
        const float ey = b.y - a.y;
        const float length2 = ex * ex + ey * ey;
        float u = length2 > 1.0e-12f
                      ? ((point.x - a.x) * ex + (point.y - a.y) * ey) / length2
                      : 0.0f;
        u = fminf(1.0f, fmaxf(0.0f, u));
        const Point2f closest{a.x + u * ex, a.y + u * ey};
        const Point2f vector{point.x - closest.x, point.y - closest.y};
        const float distance = sqrtf(vector.x * vector.x + vector.y * vector.y);
        if (distance < best_distance) {
            best_distance = distance;
            best_vector = vector;
        }
    }
    const float norm = sqrtf(best_vector.x * best_vector.x + best_vector.y * best_vector.y);
    if (norm < 1.0e-12f) return result;
    const bool inside = point_inside(point, polygon, true);
    if (inside) {
        result.x = -best_vector.x * threshold / norm;
        result.y = -best_vector.y * threshold / norm;
    } else if (best_distance <= threshold) {
        const float scale = (threshold - best_distance) / norm;
        result.x = best_vector.x * scale;
        result.y = best_vector.y * scale;
    }
    return result;
}

}  // namespace polygon
}  // namespace cudarobotics
