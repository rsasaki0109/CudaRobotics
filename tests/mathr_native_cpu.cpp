#include <array>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <vector>

#include "cudarobotics/filters.hpp"
#include "cudarobotics/bal_io.hpp"
#include "cudarobotics/bundle_adjustment.hpp"
#include "cudarobotics/gauss_newton.hpp"
#include "cudarobotics/geometry.hpp"
#include "cudarobotics/imls.hpp"
#include "cudarobotics/kinematics.hpp"
#include "cudarobotics/math_tools.hpp"
#include "cudarobotics/numerical_derivative.hpp"
#include "cudarobotics/polygon.hpp"
#include "cudarobotics/projection.hpp"

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (condition) std::printf("  PASS: %s\n", name);
    else {
        std::printf("  FAIL: %s\n", name);
        ++failures;
    }
}

float max_abs(const float* a, const float* b, int n) {
    float result = 0.0f;
    for (int i = 0; i < n; ++i) result = fmaxf(result, fabsf(a[i] - b[i]));
    return result;
}

void test_kinematics() {
    std::printf("[test_kinematics]\n");
    const float xi[6] = {-1.0f, 1.0f, 2.0f, 0.2f, 0.4f, 0.2f};
    float T_ba[16];
    cudarobotics::lie::se3_exp(xi, T_ba);
    const float va[6] = {1.0f, 2.0f, 3.0f, 0.3f, 0.5f, 1.0f};
    float vb[6];
    cudarobotics::kinematics::transform_velocity_3d(T_ba, va, vb);
    float identity[16];
    cudarobotics::lie::mat4_identity(identity);
    float T_wb[16];
    float T_ba_inv[16];
    float T_wb_step[16];
    float T_wa_step[16];
    cudarobotics::projection::rigid_inverse(T_ba, T_ba_inv);
    cudarobotics::projection::mat4_multiply(identity, T_ba_inv, T_wb);
    float va_step[16];
    float vb_step[16];
    cudarobotics::lie::se3_exp(va, va_step);
    cudarobotics::lie::se3_exp(vb, vb_step);
    cudarobotics::projection::mat4_multiply(T_wb, vb_step, T_wb_step);
    cudarobotics::projection::mat4_multiply(T_wb_step, T_ba, T_wa_step);
    check(max_abs(va_step, T_wa_step, 16) < 4.0e-4f,
          "3-D velocity transform preserves rigid-body motion");

    const float xi2[3] = {1.0f, 2.0f, 0.3f};
    float T2[9];
    cudarobotics::lie::se2_exp(xi2, T2);
    const float v2[3] = {1.0f, 1.0f, 0.3f};
    float vb2[3];
    cudarobotics::kinematics::transform_velocity_2d(T2, v2, vb2);
    float T2_inv[9];
    cudarobotics::lie::mat2_identity(T2_inv);  // only to keep this block fixed-size
    (void)T2_inv;
    check(fabsf(vb2[2] - v2[2]) < 1.0e-6f, "2-D velocity preserves angular rate");

    cudarobotics::kinematics::ImuInput imu_input;
    imu_input.acceleration[0] = 1.0f;
    imu_input.angular_velocity[2] = 0.2f;
    cudarobotics::kinematics::ImuKinematicState imu_state;
    imu_state.velocity[0] = 2.0f;
    imu_state.angular_velocity[2] = 0.1f;
    float imu_delta[12];
    cudarobotics::kinematics::imu_input_kinematic_model(
        imu_input, imu_state, 0.5f, imu_delta);
    check(fabsf(imu_delta[0] - 1.0f) < 1.0e-6f &&
              fabsf(imu_delta[6] - 0.5f) < 1.0e-6f &&
              fabsf(imu_delta[11] - 0.1f) < 1.0e-6f,
          "IMU kinematic model matches 12-DoF state increment");
    const float state_vector[12] = {1.0f, 2.0f, 3.0f,
                                    0.1f, -0.2f, 0.3f,
                                    0.4f, 0.5f, 0.6f,
                                    -0.1f, 0.2f, -0.3f};
    cudarobotics::kinematics::ImuKinematicState recovered_state;
    cudarobotics::kinematics::imu_state_from_vector(state_vector, &recovered_state);
    float state_pose[16];
    cudarobotics::kinematics::imu_state_pose_matrix(recovered_state, state_pose);
    check(fabsf(state_pose[3] - 1.0f) < 1.0e-6f &&
              fabsf(state_pose[7] - 2.0f) < 1.0e-6f,
          "IMU state vector and pose-matrix adapters");
}

void test_geometry_and_imls() {
    std::printf("[test_geometry_and_imls]\n");
    std::vector<cudarobotics::geometry::Vec3> line_points;
    for (int i = -3; i <= 3; ++i)
        line_points.push_back(cudarobotics::geometry::Vec3{static_cast<float>(i), 2.0f * i, 3.0f * i});
    cudarobotics::geometry::Vec3 center;
    cudarobotics::geometry::Vec3 direction;
    const bool line_ok = cudarobotics::geometry::fit_line(line_points, &center, &direction);
    check(line_ok,
          "line fit detects anisotropic points");
    check(fabsf(fabsf(direction.x) - 1.0f / sqrtf(14.0f)) < 2.0e-3f,
          "line fit returns principal direction");
    std::vector<cudarobotics::geometry::Vec3> plane_points = {
        {-1.0f, -1.0f, 1.0f}, {1.0f, -1.0f, 1.0f},
        {-1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.2f, -0.3f, 1.0f}};
    float plane[4];
    check(cudarobotics::geometry::fit_plane(plane_points, plane),
          "plane fit detects planar points");
    check(fabsf(cudarobotics::geometry::point_to_plane(
                    cudarobotics::geometry::Vec3{0.0f, 0.0f, 1.0f}, plane)) < 1.0e-4f,
          "point-to-plane residual");

    std::vector<cudarobotics::imls::Point2f> curve;
    for (int i = -5; i <= 5; ++i) curve.push_back({0.25f * i, 0.0f});
    std::vector<cudarobotics::imls::Point2f> normals;
    cudarobotics::imls::estimate_normals(curve, 0.8f, &normals);
    float distance = 0.0f;
    cudarobotics::imls::Point2f normal;
    check(cudarobotics::imls::point_to_surface({0.0f, 0.5f}, curve, normals,
                                               1.0f, &distance, &normal),
          "IMLS surface query");
    check(fabsf(fabsf(distance) - 0.5f) < 0.08f,
          "IMLS distance on a straight scan line");
}

void test_polygon_and_projection() {
    std::printf("[test_polygon_and_projection]\n");
    const std::vector<cudarobotics::polygon::Point2f> square = {
        {-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f}};
    check(cudarobotics::polygon::point_inside({0.0f, 0.0f}, square),
          "polygon contains interior point");
    check(!cudarobotics::polygon::point_inside({2.0f, 0.0f}, square),
          "polygon rejects exterior point");
    const auto push = cudarobotics::polygon::polygon_residual({1.2f, 0.0f}, square, 1.0f);
    check(push.x > 0.15f && fabsf(push.y) < 1.0e-5f,
          "polygon residual matches reference push vector");

    const float x[6] = {0.1f, -0.2f, 0.4f, 0.03f, -0.02f, 0.1f};
    const float y[6] = {-0.2f, 0.1f, -0.3f, -0.01f, 0.02f, -0.04f};
    float plus[6];
    float minus[6];
    cudarobotics::projection::pose_plus(x, y, plus);
    cudarobotics::projection::pose_minus(plus, x, minus);
    check(max_abs(y, minus, 6) < 2.0e-4f, "projection pose plus/minus");
    float T[16];
    cudarobotics::lie::se3_exp(x, T);
    const float point[3] = {0.5f, -0.3f, 4.0f};
    const float K[9] = {400.0f, 0.0f, 200.0f,
                        0.0f, 420.0f, 100.0f,
                        0.0f, 0.0f, 1.0f};
    float measurement[2];
    float camera_point[3];
    cudarobotics::projection::transform_inverse(T, false, point, camera_point);
    cudarobotics::projection::reproject(camera_point, K, measurement);
    float residual[2];
    float J_pose[12];
    float J_point[6];
    check(cudarobotics::projection::reprojection_error(
              T, point, measurement, K, residual, J_pose, J_point),
          "projection residual with Jacobians");
    const float zero_residual[2] = {0.0f, 0.0f};
    check(max_abs(residual, zero_residual, 2) < 1.0e-4f,
          "projection residual is zero at generated measurement");

    float T_cw[16];
    cudarobotics::projection::rigid_inverse(T, T_cw);
    float residual_cw[2];
    check(cudarobotics::projection::reprojection_error_cw(
              T_cw, point, measurement, K, residual_cw) &&
              max_abs(residual_cw, zero_residual, 2) < 1.0e-4f,
          "T_cw projection residual matches T_wc convention");
    float residual_bc[2];
    float identity_bc[16];
    cudarobotics::lie::mat4_identity(identity_bc);
    check(cudarobotics::projection::reprojection_error_with_body_camera(
              T, identity_bc, point, measurement, K, residual_bc) &&
              max_abs(residual_bc, zero_residual, 2) < 1.0e-4f,
          "body-camera composition projection residual");
    float camera_prior[6];
    float camera_prior_jacobian[36];
    const float zero6[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    cudarobotics::projection::camera_prior_factor(
        T, T, camera_prior, camera_prior_jacobian);
    check(max_abs(camera_prior, zero6, 6) < 1.0e-6f &&
              camera_prior_jacobian[0] == 1.0f && camera_prior_jacobian[35] == 1.0f,
          "camera prior factor at measurement");
    float camera_between[6];
    float camera_between_from[36];
    float camera_between_to[36];
    cudarobotics::projection::camera_between_factor(
        T, T, identity_bc, camera_between, camera_between_from, camera_between_to);
    check(max_abs(camera_between, zero6, 6) < 1.0e-6f &&
              fabsf(camera_between_to[0] - 1.0f) < 1.0e-3f,
          "camera between factor at identity measurement");
    const float x_from[6] = {0.12f, -0.08f, 0.2f, 0.03f, -0.04f, 0.05f};
    const float x_to[6] = {-0.15f, 0.11f, -0.1f, -0.02f, 0.06f, 0.04f};
    float T_from[16];
    float T_to[16];
    cudarobotics::lie::se3_exp(x_from, T_from);
    cudarobotics::lie::se3_exp(x_to, T_to);
    cudarobotics::projection::camera_between_factor(
        T_from, T_to, identity_bc, camera_between, camera_between_from, camera_between_to);
    float numeric_from[36] = {};
    float numeric_to[36] = {};
    const float h = 1.0e-4f;
    for (int col = 0; col < 6; ++col) {
        float plus_delta[6] = {};
        float minus_delta[6] = {};
        plus_delta[col] = h;
        minus_delta[col] = -h;
        float plus_increment[16];
        float minus_increment[16];
        float from_plus[16];
        float from_minus[16];
        float to_plus[16];
        float to_minus[16];
        cudarobotics::lie::se3_exp(plus_delta, plus_increment);
        cudarobotics::projection::mat4_multiply(T_from, plus_increment, from_plus);
        cudarobotics::lie::se3_exp(minus_delta, minus_increment);
        cudarobotics::projection::mat4_multiply(T_from, minus_increment, from_minus);
        cudarobotics::lie::se3_exp(plus_delta, plus_increment);
        cudarobotics::projection::mat4_multiply(T_to, plus_increment, to_plus);
        cudarobotics::lie::se3_exp(minus_delta, minus_increment);
        cudarobotics::projection::mat4_multiply(T_to, minus_increment, to_minus);
        float residual_from_plus[6];
        float residual_from_minus[6];
        float residual_to_plus[6];
        float residual_to_minus[6];
        cudarobotics::projection::camera_between_factor(
            from_plus, T_to, identity_bc, residual_from_plus, nullptr, nullptr);
        cudarobotics::projection::camera_between_factor(
            from_minus, T_to, identity_bc, residual_from_minus, nullptr, nullptr);
        cudarobotics::projection::camera_between_factor(
            T_from, to_plus, identity_bc, residual_to_plus, nullptr, nullptr);
        cudarobotics::projection::camera_between_factor(
            T_from, to_minus, identity_bc, residual_to_minus, nullptr, nullptr);
        for (int row = 0; row < 6; ++row) {
            numeric_from[6 * row + col] =
                (residual_from_plus[row] - residual_from_minus[row]) / (2.0f * h);
            numeric_to[6 * row + col] =
                (residual_to_plus[row] - residual_to_minus[row]) / (2.0f * h);
        }
    }
    const float camera_between_from_error = max_abs(camera_between_from, numeric_from, 36);
    const float camera_between_to_error = max_abs(camera_between_to, numeric_to, 36);
    check(camera_between_from_error < 2.0e-2f && camera_between_to_error < 2.0e-2f,
          "camera between factor Jacobians match right perturbations");
    const float point_measurement[3] = {point[0], point[1], point[2]};
    float point_prior[3];
    float point_prior_jacobian[9];
    const float zero3[3] = {0.0f, 0.0f, 0.0f};
    cudarobotics::projection::point_prior_factor(
        point, point_measurement, point_prior, point_prior_jacobian);
    check(max_abs(point_prior, zero3, 3) < 1.0e-6f &&
              point_prior_jacobian[0] == 1.0f,
          "point prior factor at measurement");
}

void test_math_tools() {
    std::printf("[test_math_tools]\n");
    const float pose[6] = {1.0f, -2.0f, 0.5f, 0.1f, -0.2f, 0.3f};
    float T[16];
    float recovered[6];
    cudarobotics::math::p2m(pose, T);
    cudarobotics::math::m2p(T, recovered);
    check(max_abs(pose, recovered, 6) < 2.0e-4f,
          "p2m/m2p preserves direct-translation pose convention");
    float H[9];
    float H_inv[9];
    const float omega[3] = {0.2f, -0.3f, 0.1f};
    float product[9];
    cudarobotics::lie::hso3(omega, H);
    cudarobotics::lie::d_log_so3(omega, H_inv);
    cudarobotics::lie::mat3_mul(H, H_inv, product);
    float identity_error = 0.0f;
    for (int row = 0; row < 3; ++row) for (int col = 0; col < 3; ++col) {
        const float expected = row == col ? 1.0f : 0.0f;
        identity_error = fmaxf(identity_error, fabsf(product[3 * row + col] - expected));
    }
    check(identity_error < 4.0e-4f, "HSO3 and dLogSO3 are inverse Jacobians");
    const std::vector<float> input = {0.3f, -0.2f, 0.7f};
    std::vector<float> J;
    check(cudarobotics::math::numerical_derivative(
              [](const std::vector<float>& x) {
                  return std::vector<float>{x[0] * x[0] + x[1], sinf(x[2])};
              }, input, &J), "numerical derivative evaluates vector function");
    check(fabsf(J[0] - 0.6f) < 2.0e-4f && fabsf(J[1] - 1.0f) < 2.0e-4f &&
              fabsf(J[5] - cosf(input[2])) < 2.0e-4f,
          "numerical derivative values");
}

void test_ba_and_gauss_newton() {
    std::printf("[test_ba_and_gauss_newton]\n");
    std::istringstream bal_input(
        "1 1 1\n"
        "0 0 10 20\n"
        "0 0 0 1 2 3 500 0.01 -0.001\n"
        "1 2 10\n");
    cudarobotics::bal::Dataset dataset;
    check(cudarobotics::bal::parse(bal_input, &dataset), "parse BAL camera/point fixture");
    check(dataset.cameras.size() == 1 && dataset.observations.size() == 1 &&
              dataset.points.size() == 3, "BAL fixture dimensions");
    check(fabsf(dataset.observations[0].u[0] + 10.0f) < 1.0e-6f &&
              fabsf(dataset.cameras[0].K[0] - 500.0f) < 1.0e-6f,
          "BAL observation sign and focal length convention");

    cudarobotics::optimization::GaussNewtonSolver solver(
        1,
        [](const std::vector<float>& x,
           std::vector<cudarobotics::optimization::ResidualBlock>* blocks) {
            blocks->clear();
            cudarobotics::optimization::ResidualBlock first;
            first.residual = {x[0] - 1.0f};
            first.jacobian = {1.0f};
            blocks->push_back(first);
            cudarobotics::optimization::ResidualBlock second;
            second.residual = {2.0f * (x[0] - 1.0f)};
            second.jacobian = {2.0f};
            blocks->push_back(second);
        });
    int iterations = 0;
    float score = 0.0f;
    const std::vector<float> result = solver.solve({0.0f}, {}, &iterations, &score);
    check(iterations > 0 && fabsf(result[0] - 1.0f) < 1.0e-4f,
          "Gauss-Newton residual/Jacobian contract");
    check(score < 1.0e-5f, "Gauss-Newton final score");

    std::array<float, 16> camera0{};
    std::array<float, 16> camera1{};
    cudarobotics::lie::mat4_identity(camera0.data());
    cudarobotics::lie::mat4_identity(camera1.data());
    camera1[3] = 0.5f;
    const std::array<float, 9> intrinsics = {400.0f, 0.0f, 200.0f,
                                             0.0f, 420.0f, 100.0f,
                                             0.0f, 0.0f, 1.0f};
    cudarobotics::ba::BundleAdjustment3D ba;
    ba.add_camera(camera0, true);
    std::array<float, 16> camera1_initial = camera1;
    camera1_initial[3] += 0.08f;
    ba.add_camera(camera1_initial, false);
    ba.set_camera_intrinsics(0, intrinsics);
    ba.set_camera_intrinsics(1, intrinsics);
    const std::array<float, 3> points[] = {
        {-1.0f, -0.5f, 4.0f}, {-0.3f, 0.4f, 5.0f}, {0.2f, -0.2f, 3.5f},
        {0.8f, 0.7f, 6.0f}, {1.2f, -0.6f, 4.5f}, {-0.8f, 0.9f, 5.5f}};
    for (const auto& point : points) {
        std::array<float, 3> initial_point = point;
        initial_point[0] += 0.05f;
        initial_point[1] -= 0.03f;
        ba.add_point(initial_point);
    }
    for (int camera = 0; camera < 2; ++camera) {
        for (int point_id = 0; point_id < 6; ++point_id) {
            float T[16];
            cudarobotics::lie::mat4_identity(T);
            if (camera == 1) T[3] = 0.5f;
            float pc[3];
            float pixel[2];
            cudarobotics::projection::transform_inverse(
                T, false, points[point_id].data(), pc);
            if (!cudarobotics::projection::reproject(pc, intrinsics.data(), pixel)) {
                check(false, "BAL fixture projection");
                continue;
            }
            cudarobotics::ba::Observation observation;
            observation.camera = camera;
            observation.point = point_id;
            observation.pixel[0] = pixel[0];
            observation.pixel[1] = pixel[1];
            ba.add_observation(observation);
        }
    }
    const float ba_before = ba.score();
    cudarobotics::ba::BundleAdjustmentOptions ba_options;
    ba_options.max_iterations = 20;
    ba_options.min_iterations = 2;
    ba_options.damping = 1.0e-4f;
    ba_options.max_step = 0.25f;
    const cudarobotics::ba::BundleAdjustmentSummary ba_summary = ba.solve(ba_options);
    check(ba_summary.finite && ba.score() < ba_before * 1.0e-3f,
          "3-D BAL reprojection solver reduces error");
    check(fabsf(ba.cameras()[1][3] - 0.5f) < 1.0e-2f,
          "3-D BAL solver recovers camera translation");
}

void test_filters() {
    std::printf("[test_filters]\n");
    cudarobotics::filters::Ekf2D ekf;
    const float control[3] = {1.0f, 0.0f, 0.1f};
    ekf.predict(control, 1.0f);
    const float observation[2] = {1.0f, 0.0f};
    check(ekf.correct(observation), "EKF correction");
    check(ekf.covariance[0] < 0.1f, "EKF incorporates GPS covariance");

    cudarobotics::filters::State2D initial;
    cudarobotics::filters::ParticleFilter2D pf(64, initial);
    pf.predict(control, 0.1f, 0.01f, 123);
    const float R[4] = {0.1f, 0.0f, 0.0f, 0.1f};
    pf.correct(observation, R);
    check(pf.effective_sample_size() > 1.0f, "particle filter normalizes weights");
    pf.resample(456);
    cudarobotics::filters::State2D estimate;
    float covariance[25];
    pf.estimate(&estimate, covariance);
    check(std::isfinite(estimate.position[0]), "particle filter estimate is finite");
}

}  // namespace

int main() {
    std::printf("=== test_mathr_native ===\n");
    test_kinematics();
    test_geometry_and_imls();
    test_polygon_and_projection();
    test_math_tools();
    test_ba_and_gauss_newton();
    test_filters();
    if (failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) FAILED.\n", failures);
    return 1;
}
