// Predicted dynamic-obstacle critic of the production CUDA MPPI core:
// disabled configurations are bit-identical to the plain planner, a head-on
// mover changes the command, a co-located obstacle invalidates every rollout,
// and invalid obstacle input is rejected.
#include "cuda_mppi_controller/mppi_gpu.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using cuda_mppi_controller::DynamicObstacle;
using cuda_mppi_controller::MppiGpu;
using cuda_mppi_controller::MppiParams;
using cuda_mppi_controller::MppiResult;

void check(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}

MppiParams params(float dynamic_weight)
{
    MppiParams p;
    p.batch_size = 1024;
    p.time_steps = 40;
    p.v_min = 0.0f;
    p.dynamic_obstacle_weight = dynamic_weight;
    return p;
}

// A few warm-started cycles along a straight path from the origin towards +x,
// with no costmap; the robot pose is held fixed so only the planner changes.
std::vector<MppiResult> plan(MppiGpu& planner, int cycles = 4)
{
    std::vector<float> path;
    for (int i = 0; i <= 40; ++i) path.insert(path.end(), {0.1f * i, 0.0f});
    std::vector<MppiResult> results;
    for (int cycle = 0; cycle < cycles; ++cycle) {
        results.push_back(planner.compute(
            0.0f, 0.0f, 0.0f, nullptr, 0, 0, 0.0f, 0.0f, 0.05f,
            path.data(), static_cast<int>(path.size() / 2), 4.0f, 0.0f, 0.0f, true));
    }
    return results;
}

bool identical(const MppiResult& a, const MppiResult& b)
{
    return std::memcmp(&a.v, &b.v, sizeof(float)) == 0 &&
           std::memcmp(&a.w, &b.w, sizeof(float)) == 0 &&
           std::memcmp(&a.best_cost, &b.best_cost, sizeof(float)) == 0 &&
           std::memcmp(&a.mean_cost, &b.mean_cost, sizeof(float)) == 0 &&
           a.valid_rollouts == b.valid_rollouts;
}

void disabled_critic_is_bit_identical()
{
    MppiGpu baseline(params(0.0f));
    const auto expected = plan(baseline);

    const DynamicObstacle head_on{1.2f, 0.0f, -0.5f, 0.0f, 0.3f};
    MppiGpu weight_zero(params(0.0f));
    weight_zero.setDynamicObstacles(&head_on, 1);
    MppiGpu no_obstacles(params(25.0f));
    for (auto* planner : {&weight_zero, &no_obstacles}) {
        const auto actual = plan(*planner);
        for (std::size_t i = 0; i < expected.size(); ++i)
            check(identical(actual[i], expected[i]),
                  "a disabled dynamic-obstacle critic must not change the plan");
    }
    no_obstacles.setDynamicObstacles(&head_on, 1);
    no_obstacles.setDynamicObstacles(nullptr, 0);
    check(no_obstacles.dynamicObstacleCount() == 0, "count 0 clears the obstacles");
}

void head_on_mover_changes_the_command()
{
    MppiGpu baseline(params(0.0f));
    const MppiResult free_run = plan(baseline).back();

    MppiGpu aware(params(25.0f));
    // Far enough ahead that avoiding it is feasible within the horizon.
    const DynamicObstacle head_on{1.8f, 0.0f, -0.4f, 0.0f, 0.3f};
    aware.setDynamicObstacles(&head_on, 1);
    check(aware.dynamicObstacleCount() == 1, "obstacle count is stored");
    const MppiResult avoid = plan(aware).back();
    std::printf(
        "free: v=%.3f w=%.3f valid=%.3f | head-on: v=%.3f w=%.3f valid=%.3f\n",
        free_run.v, free_run.w, free_run.valid_rollout_ratio,
        avoid.v, avoid.w, avoid.valid_rollout_ratio);
    check(avoid.valid_rollout_ratio > 0.0f && avoid.valid_rollout_ratio < free_run.valid_rollout_ratio,
          "predicted collisions must invalidate some, not all, rollouts");
    check(avoid.v < free_run.v - 0.05f || std::fabs(avoid.w) > std::fabs(free_run.w) + 0.1f,
          "a head-on mover must slow the robot or turn it away");
    check(std::isfinite(avoid.v) && std::isfinite(avoid.w), "command stays finite");
}

void co_located_obstacle_invalidates_every_rollout()
{
    MppiParams p = params(25.0f);
    p.enable_retreat = false;
    MppiGpu planner(p);
    const DynamicObstacle on_robot{0.0f, 0.0f, 0.0f, 0.0f, 0.5f};
    planner.setDynamicObstacles(&on_robot, 1);
    const MppiResult result = plan(planner, 1).back();
    check(result.all_colliding && result.valid_rollouts == 0,
          "an obstacle covering the robot makes every rollout collide");
}

void overlap_allowance_tolerates_a_small_existing_overlap()
{
    // A static disc ahead that the robot already penetrates by 0.05 m, as a
    // tracking error can cause.
    const DynamicObstacle ahead{0.5f, 0.0f, 0.0f, 0.0f, 0.3f};
    MppiParams strict = params(25.0f);
    strict.enable_retreat = false;
    MppiGpu hard(strict);
    hard.setDynamicObstacles(&ahead, 1);
    check(plan(hard, 1).back().all_colliding,
          "without an allowance an existing overlap makes every rollout collide");

    MppiParams tolerant = strict;
    tolerant.dynamic_obstacle_overlap_allowance = 0.1f;
    MppiGpu soft(tolerant);
    soft.setDynamicObstacles(&ahead, 1);
    const MppiResult result = plan(soft, 1).back();
    std::printf("allowance: v=%.3f w=%.3f valid=%.3f\n", result.v, result.w, result.valid_rollout_ratio);
    check(!result.all_colliding && result.valid_rollouts > 0,
          "an allowed overlap must leave valid rollouts");
    check(result.v < 0.25f, "the robot must not drive deeper into the disc");

    // Beyond the cap the allowance does not apply.
    const DynamicObstacle deep{0.2f, 0.0f, 0.0f, 0.0f, 0.3f};
    MppiGpu capped(tolerant);
    capped.setDynamicObstacles(&deep, 1);
    check(plan(capped, 1).back().valid_rollout_ratio < 0.05f,
          "a penetration deeper than the cap is still a collision");
}

// Plan freely, then drop a disc onto the robot so every rollout collides and
// the back-out takes over; returns the first back-out command.
MppiResult retreat_after_blocking(MppiParams p)
{
    MppiGpu planner(p);
    plan(planner, 3);
    const DynamicObstacle on_robot{0.0f, 0.0f, 0.0f, 0.0f, 0.5f};
    planner.setDynamicObstacles(&on_robot, 1);
    return plan(planner, 1).back();
}

void retreat_reverse_speed_lets_a_forward_only_robot_back_out()
{
    // Forward-only (v_min = 0): by default the back-out is clamped to v = 0.
    const MppiResult clamped = retreat_after_blocking(params(25.0f));
    check(clamped.all_colliding && clamped.retreating, "a blocked robot backs out");
    check(clamped.v == 0.0f, "without retreat_reverse_speed the back-out keeps v_min");

    MppiParams p = params(25.0f);
    p.retreat_reverse_speed = 0.2f;
    const MppiResult reversing = retreat_after_blocking(p);
    std::printf("retreat: clamped v=%.3f | reverse v=%.3f\n", clamped.v, reversing.v);
    check(reversing.retreating && reversing.v < 0.0f && reversing.v >= -0.2f - 1.0e-6f,
          "retreat_reverse_speed allows a bounded reverse back-out");
}

void invalid_input_is_rejected()
{
    MppiGpu planner(params(25.0f));
    std::vector<DynamicObstacle> too_many(cuda_mppi_controller::kMaxDynamicObstacles + 1);
    const DynamicObstacle nan_x{std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 0.0f, 0.3f};
    const DynamicObstacle negative{0.0f, 0.0f, 0.0f, 0.0f, -0.1f};
    int rejected = 0;
    try { planner.setDynamicObstacles(too_many.data(), static_cast<int>(too_many.size())); }
    catch (const std::invalid_argument&) { ++rejected; }
    try { planner.setDynamicObstacles(&nan_x, 1); } catch (const std::invalid_argument&) { ++rejected; }
    try { planner.setDynamicObstacles(&negative, 1); } catch (const std::invalid_argument&) { ++rejected; }
    try { planner.setDynamicObstacles(nullptr, 1); } catch (const std::invalid_argument&) { ++rejected; }
    check(rejected == 4, "invalid dynamic obstacles must throw std::invalid_argument");
    check(planner.dynamicObstacleCount() == 0, "rejected input leaves the obstacles unchanged");
}

}  // namespace

int main()
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "no CUDA device\n");
        return 1;
    }
    disabled_critic_is_bit_identical();
    head_on_mover_changes_the_command();
    co_located_obstacle_invalidates_every_rollout();
    overlap_allowance_tolerates_a_small_existing_overlap();
    retreat_reverse_speed_lets_a_forward_only_robot_back_out();
    invalid_input_is_rejected();
    std::printf("mppi_dynamic_obstacle_smoke: OK\n");
    return 0;
}
