// Small dependency-free 3-D bundle-adjustment backend for the reusable
// CameraVertex/PointVertex/ReprojEdge portion of mathR/slam/projection.py.
// Large problems should use the existing CUDA Schur-complement BA executable;
// this class provides a deterministic CPU reference and factor contract.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "cudarobotics/projection.hpp"
#include "cudarobotics/robust_loss.cuh"

namespace cudarobotics {
namespace ba {

struct Observation {
    int camera = -1;
    int point = -1;
    float pixel[2] = {0.0f, 0.0f};
    float information[4] = {1.0f, 0.0f, 0.0f, 1.0f};
};

struct BundleAdjustmentOptions {
    int max_iterations = 15;
    int min_iterations = 2;
    float min_score_change = 1.0e-5f;
    float damping = 1.0e-5f;
    float max_step = 0.0f;
    float huber_delta = 0.0f;
};

struct BundleAdjustmentSummary {
    int iterations = 0;
    float initial_score = 0.0f;
    float final_score = 0.0f;
    bool finite = true;
};

inline bool solve_ba_dense(std::vector<float> A,
                           std::vector<float> b,
                           std::vector<float>* x) {
    const int n = static_cast<int>(b.size());
    if (x == nullptr || static_cast<int>(A.size()) != n * n) return false;
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int row = col + 1; row < n; ++row)
            if (fabsf(A[row * n + col]) > fabsf(A[pivot * n + col])) pivot = row;
        if (fabsf(A[pivot * n + col]) < 1.0e-12f) return false;
        if (pivot != col) {
            for (int j = col; j < n; ++j) std::swap(A[col * n + j], A[pivot * n + j]);
            std::swap(b[col], b[pivot]);
        }
        for (int row = col + 1; row < n; ++row) {
            const float factor = A[row * n + col] / A[col * n + col];
            for (int j = col; j < n; ++j) A[row * n + j] -= factor * A[col * n + j];
            b[row] -= factor * b[col];
        }
    }
    x->assign(n, 0.0f);
    for (int row = n - 1; row >= 0; --row) {
        float value = b[row];
        for (int col = row + 1; col < n; ++col) value -= A[row * n + col] * (*x)[col];
        (*x)[row] = value / A[row * n + row];
    }
    return true;
}

class BundleAdjustment3D {
public:
    int add_camera(const std::array<float, 16>& T, bool constant = false) {
        cameras_.push_back(T);
        camera_constants_.push_back(constant);
        camera_intrinsics_.push_back(std::array<float, 9>{
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f});
        return static_cast<int>(cameras_.size()) - 1;
    }

    int add_point(const std::array<float, 3>& point) {
        points_.push_back(point);
        return static_cast<int>(points_.size()) - 1;
    }

    void add_observation(const Observation& observation) { observations_.push_back(observation); }

    const std::vector<std::array<float, 16> >& cameras() const { return cameras_; }
    const std::vector<std::array<float, 3> >& points() const { return points_; }
    std::vector<std::array<float, 16> >& cameras() { return cameras_; }
    std::vector<std::array<float, 3> >& points() { return points_; }

    float score(const BundleAdjustmentOptions& options = BundleAdjustmentOptions()) const {
        float total = 0.0f;
        for (const Observation& observation : observations_) {
            float residual[2];
            if (!projection::reprojection_error(
                    cameras_[observation.camera].data(),
                    points_[observation.point].data(), observation.pixel,
                    camera_intrinsics_[observation.camera].data(), residual)) continue;
            const float e2 = quadratic(observation.information, residual);
            total += options.huber_delta > 0.0f
                ? robust::huber(e2, options.huber_delta).rho0 : e2;
        }
        return total;
    }

    void set_camera_intrinsics(int camera, const std::array<float, 9>& K) {
        if (camera >= static_cast<int>(camera_intrinsics_.size())) camera_intrinsics_.resize(camera + 1);
        camera_intrinsics_[camera] = K;
    }

    BundleAdjustmentSummary solve(const BundleAdjustmentOptions& options = BundleAdjustmentOptions()) {
        BundleAdjustmentSummary summary;
        summary.initial_score = score(options);
        float previous = summary.initial_score;
        for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
            std::vector<float> dx;
            float linearized_score = 0.0f;
            if (!solve_once(options, &dx, &linearized_score)) {
                summary.finite = false;
                break;
            }
            if (options.max_step > 0.0f) {
                float max_abs = 0.0f;
                for (float value : dx) max_abs = fmaxf(max_abs, fabsf(value));
                if (max_abs > options.max_step)
                    for (float& value : dx) value *= options.max_step / max_abs;
            }
            apply(dx);
            summary.iterations = iteration + 1;
            summary.final_score = score(options);
            if (summary.iterations >= options.min_iterations &&
                fabsf(previous - summary.final_score) < options.min_score_change) break;
            previous = summary.final_score;
        }
        if (summary.iterations == 0) summary.final_score = score(options);
        return summary;
    }

private:
    static float quadratic(const float* information, const float* residual) {
        float weighted[2] = {
            information[0] * residual[0] + information[1] * residual[1],
            information[2] * residual[0] + information[3] * residual[1]};
        return residual[0] * weighted[0] + residual[1] * weighted[1];
    }

    bool solve_once(const BundleAdjustmentOptions& options,
                    std::vector<float>* dx,
                    float* score_out) const {
        std::vector<int> camera_offsets(cameras_.size(), -1);
        std::vector<int> point_offsets(points_.size(), -1);
        int parameter_count = 0;
        for (size_t i = 0; i < cameras_.size(); ++i) {
            if (!camera_constants_[i]) { camera_offsets[i] = parameter_count; parameter_count += 6; }
        }
        for (size_t i = 0; i < points_.size(); ++i) {
            point_offsets[i] = parameter_count;
            parameter_count += 3;
        }
        if (parameter_count == 0) return false;
        std::vector<float> H(parameter_count * parameter_count, 0.0f);
        std::vector<float> g(parameter_count, 0.0f);
        float score = 0.0f;
        for (const Observation& observation : observations_) {
            float residual[2];
            float J_camera[12];
            float J_point[6];
            if (!projection::reprojection_error(
                    cameras_[observation.camera].data(),
                    points_[observation.point].data(), observation.pixel,
                    camera_intrinsics_[observation.camera].data(), residual,
                    J_camera, J_point)) continue;
            const float e2 = quadratic(observation.information, residual);
            const robust::LossEvaluation loss = options.huber_delta > 0.0f
                ? robust::huber(e2, options.huber_delta) : robust::l2(e2);
            score += loss.rho0;
            const float weight = fmaxf(loss.rho1, 0.0f);
            accumulate(observation.camera, observation.point, camera_offsets, point_offsets,
                       residual, J_camera, J_point, observation.information, weight, &H, &g);
        }
        for (int i = 0; i < parameter_count; ++i) H[i * parameter_count + i] += options.damping;
        std::vector<float> rhs(parameter_count);
        for (int i = 0; i < parameter_count; ++i) rhs[i] = -g[i];
        if (!solve_ba_dense(H, rhs, dx)) return false;
        if (score_out != nullptr) *score_out = score;
        return true;
    }

    static void accumulate(int camera,
                           int point,
                           const std::vector<int>& camera_offsets,
                           const std::vector<int>& point_offsets,
                           const float* residual,
                           const float* J_camera,
                           const float* J_point,
                           const float* information,
                           float weight,
                           std::vector<float>* H,
                           std::vector<float>* g) {
        const int n = static_cast<int>(std::sqrt(static_cast<float>(H->size())));
        const int camera_offset = camera_offsets[camera];
        const int point_offset = point_offsets[point];
        const float weighted_residual[2] = {
            information[0] * residual[0] + information[1] * residual[1],
            information[2] * residual[0] + information[3] * residual[1]};
        if (camera_offset >= 0) {
            for (int col = 0; col < 6; ++col) {
                g->at(camera_offset + col) += weight *
                    (J_camera[col] * weighted_residual[0] +
                     J_camera[6 + col] * weighted_residual[1]);
                for (int other = 0; other < 6; ++other) {
                    float value = 0.0f;
                    value += J_camera[col] * information[0] * J_camera[other];
                    value += J_camera[col] * information[1] * J_camera[6 + other];
                    value += J_camera[6 + col] * information[2] * J_camera[other];
                    value += J_camera[6 + col] * information[3] * J_camera[6 + other];
                    (*H)[(camera_offset + col) * n + camera_offset + other] += weight * value;
                }
            }
        }
        for (int col = 0; col < 3; ++col) {
            g->at(point_offset + col) += weight *
                (J_point[col] * weighted_residual[0] + J_point[3 + col] * weighted_residual[1]);
            for (int other = 0; other < 3; ++other) {
                float value = 0.0f;
                value += J_point[col] * information[0] * J_point[other];
                value += J_point[col] * information[1] * J_point[3 + other];
                value += J_point[3 + col] * information[2] * J_point[other];
                value += J_point[3 + col] * information[3] * J_point[3 + other];
                (*H)[(point_offset + col) * n + point_offset + other] += weight * value;
            }
        }
        if (camera_offset >= 0) {
            for (int row = 0; row < 6; ++row) {
                for (int col = 0; col < 3; ++col) {
                    float value = 0.0f;
                    value += J_camera[row] * information[0] * J_point[col];
                    value += J_camera[row] * information[1] * J_point[3 + col];
                    value += J_camera[6 + row] * information[2] * J_point[col];
                    value += J_camera[6 + row] * information[3] * J_point[3 + col];
                    (*H)[(camera_offset + row) * n + point_offset + col] += weight * value;
                    (*H)[(point_offset + col) * n + camera_offset + row] += weight * value;
                }
            }
        }
    }

    void apply(const std::vector<float>& dx) {
        int offset = 0;
        for (size_t i = 0; i < cameras_.size(); ++i) {
            if (camera_constants_[i]) continue;
            float increment[16];
            float updated[16];
            lie::se3_exp(&dx[offset], increment);
            projection::mat4_multiply(cameras_[i].data(), increment, updated);
            for (int j = 0; j < 16; ++j) cameras_[i][j] = updated[j];
            offset += 6;
        }
        for (std::array<float, 3>& point : points_) {
            for (int j = 0; j < 3; ++j) point[j] += dx[offset + j];
            offset += 3;
        }
    }

    std::vector<std::array<float, 16> > cameras_;
    std::vector<bool> camera_constants_;
    std::vector<std::array<float, 3> > points_;
    std::vector<std::array<float, 9> > camera_intrinsics_;
    std::vector<Observation> observations_;
};

}  // namespace ba
}  // namespace cudarobotics
