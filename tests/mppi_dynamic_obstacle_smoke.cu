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
// the back-out takes over; returns the first back-out command. `behind`
// optionally adds a second obstacle.
MppiResult retreat_after_blocking(MppiParams p, const DynamicObstacle* behind = nullptr)
{
    MppiGpu planner(p);
    plan(planner, 3);
    std::vector<DynamicObstacle> obstacles{{0.0f, 0.0f, 0.0f, 0.0f, 0.5f}};
    if (behind != nullptr) obstacles.push_back(*behind);
    planner.setDynamicObstacles(obstacles.data(), static_cast<int>(obstacles.size()));
    return plan(planner, 1).back();
}

void back_out_does_not_reverse_into_an_obstacle()
{
    MppiParams p = params(25.0f);
    p.retreat_reverse_speed = 0.2f;
    // A pedestrian standing 0.1 m behind the robot (reach 0.25 + 0.25).
    const DynamicObstacle behind{-0.6f, 0.0f, 0.0f, 0.0f, 0.25f};
    const MppiResult blocked = retreat_after_blocking(p, &behind);
    std::printf("retreat with obstacle behind: v=%.3f w=%.3f blocked=%d\n",
                blocked.v, blocked.w, blocked.retreat_blocked ? 1 : 0);
    check(blocked.retreating && blocked.retreat_blocked && blocked.v == 0.0f,
          "a back-out towards an obstacle drops its translation");

    p.retreat_avoid_dynamic_obstacles = false;
    const MppiResult unchecked = retreat_after_blocking(p, &behind);
    check(unchecked.retreating && !unchecked.retreat_blocked && unchecked.v < 0.0f,
          "with the check off the back-out reverses regardless");

    // An obstacle far behind does not block it.
    p.retreat_avoid_dynamic_obstacles = true;
    const DynamicObstacle far_behind{-3.0f, 0.0f, 0.0f, 0.0f, 0.25f};
    const MppiResult free_back = retreat_after_blocking(p, &far_behind);
    check(free_back.retreating && !free_back.retreat_blocked && free_back.v < 0.0f,
          "an obstacle out of reach leaves the back-out unchanged");
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

void back_out_does_not_reverse_into_a_wall()
{
    // 5 m x 5 m costmap centred on the robot, lethal everywhere behind x = -0.15.
    const int size = 100;
    const float resolution = 0.05f, origin = -2.5f;
    std::vector<unsigned char> costmap(size * size, 0);
    for (int my = 0; my < size; ++my)
        for (int mx = 0; mx < size; ++mx)
            if (origin + (mx + 0.5f) * resolution < -0.15f) costmap[my * size + mx] = 254;
    std::vector<float> path;
    for (int i = 0; i <= 40; ++i) path.insert(path.end(), {0.1f * i, 0.0f});
    auto run = [&](bool check) {
        MppiParams p = params(25.0f);
        p.retreat_reverse_speed = 0.2f;
        p.retreat_avoid_lethal_cells = check;
        MppiGpu planner(p);
        MppiResult result;
        for (int cycle = 0; cycle < 4; ++cycle) {
            if (cycle == 3) {
                const DynamicObstacle on_robot{0.0f, 0.0f, 0.0f, 0.0f, 0.5f};
                planner.setDynamicObstacles(&on_robot, 1);
            }
            result = planner.compute(
                0.0f, 0.0f, 0.0f, costmap.data(), size, size, origin, origin, resolution,
                path.data(), static_cast<int>(path.size() / 2), 4.0f, 0.0f, 0.0f, true);
        }
        return result;
    };
    const MppiResult blocked = run(true);
    std::printf("retreat towards a wall: v=%.3f blocked=%d\n", blocked.v, blocked.retreat_blocked ? 1 : 0);
    check(blocked.retreating && blocked.retreat_blocked && blocked.v == 0.0f,
          "a back-out into lethal cells drops its translation");
    const MppiResult unchecked = run(false);
    check(unchecked.retreating && !unchecked.retreat_blocked && unchecked.v < 0.0f,
          "with the lethal-cell check off the back-out reverses regardless");
}

void evading_replaces_the_back_out_when_the_collision_is_far()
{
    // A large disc rushing head-on: every rollout eventually collides, but
    // not within the first second.
    const DynamicObstacle rushing{4.0f, 0.0f, -1.5f, 0.0f, 1.2f};
    auto run = [&](float evade_time) {
        MppiParams p = params(25.0f);
        p.retreat_reverse_speed = 0.2f;
        p.evade_min_collision_time = evade_time;
        MppiGpu planner(p);
        plan(planner, 3);
        planner.setDynamicObstacles(&rushing, 1);
        return plan(planner, 1).back();
    };
    const MppiResult backs_out = run(0.0f);
    check(backs_out.all_colliding && backs_out.retreating && !backs_out.evading,
          "with evading off an all-colliding cycle backs out");
    const MppiResult evades = run(0.5f);
    std::printf("evading: ttc=%.2f s v=%.3f w=%.3f\n", evades.time_to_collision, evades.v, evades.w);
    check(evades.all_colliding && evades.evading && !evades.retreating,
          "a distant collision is evaded instead of backed out of");
    check(evades.time_to_collision >= 0.5f, "time_to_collision reports the latest first collision");
    const MppiResult too_close = run(10.0f);
    check(too_close.retreating && !too_close.evading, "a nearer collision still backs out");
}

void yielding_obstacle_stops_short_of_the_robot()
{
    // A pedestrian 0.2 m from contact, walking straight at the robot.
    const DynamicObstacle oncoming{0.7f, 0.0f, -0.45f, 0.0f, 0.25f};
    auto run = [&](float yield_distance, const DynamicObstacle& obstacle) {
        MppiParams p = params(25.0f);
        p.enable_retreat = false;
        p.dynamic_obstacle_robot_radius = 0.25f;
        p.dynamic_obstacle_yield_distance = yield_distance;
        MppiGpu planner(p);
        planner.setDynamicObstacles(&obstacle, 1);
        return plan(planner, 3).back();
    };
    const MppiResult constant_velocity = run(-1.0f, oncoming);
    check(constant_velocity.all_colliding,
          "without yielding, an oncoming pedestrian at close range collides with every rollout");
    const MppiResult yielding = run(0.05f, oncoming);
    std::printf("yielding: v=%.3f w=%.3f valid=%.3f\n", yielding.v, yielding.w, yielding.valid_rollout_ratio);
    check(!yielding.all_colliding && yielding.valid_rollouts > 0,
          "a pedestrian predicted to stop short of the robot leaves valid rollouts");
    check(yielding.v < 0.25f, "the robot must not drive into the stopped pedestrian");

    // Obstacles that move away are unchanged.
    const DynamicObstacle receding{0.9f, 0.0f, 0.45f, 0.0f, 0.25f};
    check(identical(run(-1.0f, receding), run(0.05f, receding)),
          "yielding must not change the prediction of a receding obstacle");
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
    back_out_does_not_reverse_into_an_obstacle();
    back_out_does_not_reverse_into_a_wall();
    evading_replaces_the_back_out_when_the_collision_is_far();
    yielding_obstacle_stops_short_of_the_robot();
    invalid_input_is_rejected();
    std::printf("mppi_dynamic_obstacle_smoke: OK\n");
    return 0;
}
