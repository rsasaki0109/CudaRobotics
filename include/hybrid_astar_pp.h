#ifndef CUDABOT_HYBRID_ASTAR_PP_H
#define CUDABOT_HYBRID_ASTAR_PP_H

// Minimal forward-only Hybrid A* search + pure pursuit tracker, scoped
// for benchmark_diff_mppi's scenarios. Static obstacles only on
// purpose: this is the "global planner blind to dynamic obstacles"
// baseline, included to make the paradigm gap explicit when compared
// against the local planners (DWA, STOMP, Diff-MPPI).
//
// Forward-only (no Reeds-Shepp): keeps the implementation small while
// still exercising the (x, y, theta) lattice. Sufficient for the
// dynamic_* scenes whose starts and goals are already reachable by a
// forward-only path through the static obstacles.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <vector>

namespace cudabot {

struct Pose2D {
    float x = 0.0f;
    float y = 0.0f;
    float theta = 0.0f;
};

struct ObstacleCircle {
    float x = 0.0f;
    float y = 0.0f;
    float r = 0.0f;
};

// Linearly-moving dynamic obstacle: position at time t is
// (x0 + vx * t, y0 + vy * t), with radius r. ``t`` is measured from the
// start of the search (i.e. the time stamp the search nodes carry).
struct DynamicObstacleSpec {
    float x0 = 0.0f;
    float y0 = 0.0f;
    float vx = 0.0f;
    float vy = 0.0f;
    float r = 0.0f;
};

struct HybridAStarParams {
    // Workspace extents (search is clipped to [0, workspace] in x and y).
    float workspace = 60.0f;
    // Cell discretisation.
    float cell_size = 1.0f;
    int n_theta_bins = 36;
    // Motion primitive params.
    float wheelbase = 2.5f;
    float v_search = 2.5f;          // const forward speed used inside the search
    float max_steer = 0.5f;
    int n_steer = 7;                // odd, symmetric around 0
    int sub_steps = 4;              // bicycle sub-steps per expansion
    float dt = 0.25f;               // total time per expansion = sub_steps * (dt/sub_steps)
    float steer_penalty = 0.05f;    // cost per |steer|*dt to bias toward straighter motions
    // Collision check against scenario obstacles (circle inflation).
    float robot_radius = 0.6f;
    // Goal tolerance. theta tolerance defaults to pi so heading is
    // effectively unconstrained -- the benchmark only checks goal
    // position, and forward-only Hybrid A* often can't hit an arbitrary
    // final heading without reverse motion primitives.
    float goal_dist_thresh = 2.0f;
    float goal_theta_thresh = static_cast<float>(M_PI);
    // Search budget.
    int max_expansions = 50000;
};

struct PurePursuitParams {
    float lookahead = 4.0f;
    float wheelbase = 2.5f;
    float target_speed = 5.0f;
    float speed_gain = 1.5f;
    float max_accel = 3.0f;
    float max_steer = 0.5f;
    float goal_slowdown_radius = 5.0f;
};

namespace detail {

inline float wrap_pi(float a) {
    while (a > static_cast<float>(M_PI)) a -= 2.0f * static_cast<float>(M_PI);
    while (a < -static_cast<float>(M_PI)) a += 2.0f * static_cast<float>(M_PI);
    return a;
}

inline int64_t discretise(float x, float y, float theta,
                          const HybridAStarParams& p) {
    int ix = static_cast<int>(std::floor(x / p.cell_size));
    int iy = static_cast<int>(std::floor(y / p.cell_size));
    float tnorm = theta;
    while (tnorm < 0.0f) tnorm += 2.0f * static_cast<float>(M_PI);
    while (tnorm >= 2.0f * static_cast<float>(M_PI)) tnorm -= 2.0f * static_cast<float>(M_PI);
    int it = static_cast<int>(std::floor(tnorm / (2.0f * static_cast<float>(M_PI) / p.n_theta_bins)));
    if (it == p.n_theta_bins) it = 0;
    int nx = static_cast<int>(std::ceil(p.workspace / p.cell_size)) + 2;
    int ny = static_cast<int>(std::ceil(p.workspace / p.cell_size)) + 2;
    int ix_off = ix + 1;  // allow -1 .. nx
    int iy_off = iy + 1;
    if (ix_off < 0 || ix_off >= nx || iy_off < 0 || iy_off >= ny) return -1;
    return (static_cast<int64_t>(it) * ny + iy_off) * nx + ix_off;
}

inline bool collides(float x, float y,
                     const std::vector<ObstacleCircle>& obstacles,
                     float robot_radius) {
    for (const auto& o : obstacles) {
        float dx = x - o.x;
        float dy = y - o.y;
        if (dx * dx + dy * dy <= (o.r + robot_radius) * (o.r + robot_radius)) {
            return true;
        }
    }
    return false;
}

// Predicts each dynamic obstacle's center at absolute time ``t`` (seconds
// from the start of the search) and checks the inflated circle. Empty
// list short-circuits — callers can pass an empty vector to skip.
inline bool collides_dynamic(float x, float y, float t,
                             const std::vector<DynamicObstacleSpec>& dyn,
                             float robot_radius) {
    for (const auto& o : dyn) {
        float ox = o.x0 + o.vx * t;
        float oy = o.y0 + o.vy * t;
        float dx = x - ox;
        float dy = y - oy;
        float rad = o.r + robot_radius;
        if (dx * dx + dy * dy <= rad * rad) {
            return true;
        }
    }
    return false;
}

struct Node {
    float x = 0.0f, y = 0.0f, theta = 0.0f;
    float g = 0.0f;
    float t = 0.0f;   // absolute time stamp; only used by the dyn-aware path
    int parent = -1;
};

}  // namespace detail

// Returns a forward-only path from `start` to `goal` avoiding the given
// circular static obstacles and (optionally) the given dynamic obstacles
// at their predicted positions along the path's time stamps. Returns an
// empty vector if no path was found within the search budget. The first
// pose is `start`; the last is within the configured goal tolerance.
//
// ``t_offset`` is the wall time the planner is invoked at, relative to
// the dynamic obstacles' x0/y0 origin -- pass 0 if the obstacles are
// already specified at search start time.
inline std::vector<Pose2D> hybrid_astar_plan(
    const Pose2D& start,
    const Pose2D& goal,
    const std::vector<ObstacleCircle>& obstacles,
    const HybridAStarParams& p,
    const std::vector<DynamicObstacleSpec>& dynamic_obstacles = {},
    float t_offset = 0.0f)
{
    using namespace detail;
    if (collides(start.x, start.y, obstacles, p.robot_radius)) {
        return {};
    }
    if (!dynamic_obstacles.empty() &&
        collides_dynamic(start.x, start.y, t_offset,
                         dynamic_obstacles, p.robot_radius)) {
        return {};
    }
    std::vector<Node> closed;
    closed.reserve(p.max_expansions);

    struct OpenEntry {
        float f;
        int idx;
        bool operator>(const OpenEntry& o) const { return f > o.f; }
    };
    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<OpenEntry>> open;
    std::unordered_map<int64_t, float> best_g;

    auto heuristic = [&](float x, float y) {
        float dx = x - goal.x;
        float dy = y - goal.y;
        return std::sqrt(dx * dx + dy * dy);
    };

    Node start_node;
    start_node.x = start.x;
    start_node.y = start.y;
    start_node.theta = start.theta;
    start_node.g = 0.0f;
    start_node.t = t_offset;
    start_node.parent = -1;
    int64_t key0 = discretise(start.x, start.y, start.theta, p);
    if (key0 < 0) return {};
    closed.push_back(start_node);
    best_g[key0] = 0.0f;
    open.push({heuristic(start.x, start.y), 0});

    int goal_idx = -1;
    int expansions = 0;
    float dt_sub = p.dt / static_cast<float>(p.sub_steps);

    while (!open.empty() && expansions < p.max_expansions) {
        OpenEntry top = open.top();
        open.pop();
        int idx = top.idx;
        const Node cur = closed[idx];  // copy; subsequent push_back may invalidate refs
        int64_t kcur = discretise(cur.x, cur.y, cur.theta, p);
        auto it = best_g.find(kcur);
        if (it != best_g.end() && cur.g > it->second + 1e-6f) continue;
        expansions++;

        float dx = cur.x - goal.x;
        float dy = cur.y - goal.y;
        if (std::sqrt(dx * dx + dy * dy) < p.goal_dist_thresh) {
            float dth = std::abs(wrap_pi(cur.theta - goal.theta));
            if (dth < p.goal_theta_thresh) {
                goal_idx = idx;
                break;
            }
        }

        for (int s = 0; s < p.n_steer; s++) {
            float steer = (p.n_steer == 1) ? 0.0f
                : (-p.max_steer + 2.0f * p.max_steer
                   * static_cast<float>(s) / static_cast<float>(p.n_steer - 1));
            float nx = cur.x, ny = cur.y, nth = cur.theta;
            float nt = cur.t;
            bool collided = false;
            for (int k = 0; k < p.sub_steps; k++) {
                nx += p.v_search * std::cos(nth) * dt_sub;
                ny += p.v_search * std::sin(nth) * dt_sub;
                nth += (p.v_search / p.wheelbase) * std::tan(steer) * dt_sub;
                nt += dt_sub;
                if (nx < 0.0f || nx > p.workspace
                    || ny < 0.0f || ny > p.workspace) {
                    collided = true;
                    break;
                }
                if (collides(nx, ny, obstacles, p.robot_radius)) {
                    collided = true;
                    break;
                }
                if (!dynamic_obstacles.empty() &&
                    collides_dynamic(nx, ny, nt,
                                     dynamic_obstacles, p.robot_radius)) {
                    collided = true;
                    break;
                }
            }
            if (collided) continue;
            nth = wrap_pi(nth);
            float step_cost = p.v_search * p.dt
                + p.steer_penalty * std::abs(steer) * p.dt;
            float ng = cur.g + step_cost;
            int64_t k = discretise(nx, ny, nth, p);
            if (k < 0) continue;
            auto bit = best_g.find(k);
            if (bit != best_g.end() && ng >= bit->second - 1e-6f) continue;
            best_g[k] = ng;
            Node child;
            child.x = nx;
            child.y = ny;
            child.theta = nth;
            child.g = ng;
            child.t = nt;
            child.parent = idx;
            int child_idx = static_cast<int>(closed.size());
            closed.push_back(child);
            float f = ng + heuristic(nx, ny);
            open.push({f, child_idx});
        }
    }

    if (goal_idx < 0) {
        return {};
    }

    std::vector<Pose2D> path;
    for (int idx = goal_idx; idx >= 0; idx = closed[idx].parent) {
        Pose2D pose;
        pose.x = closed[idx].x;
        pose.y = closed[idx].y;
        pose.theta = closed[idx].theta;
        path.push_back(pose);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

// Pure pursuit: returns (accel, steer) command for the bicycle model
// given the current state, the planned path, and tracker parameters.
// Linearly searches the path for the nearest waypoint at-or-beyond the
// lookahead distance; if none exists (we are past the path end),
// targets the last waypoint and decelerates toward zero speed.
struct PurePursuitCommand {
    float accel = 0.0f;
    float steer = 0.0f;
    int target_waypoint = 0;
};

inline PurePursuitCommand pure_pursuit_step(
    float rx, float ry, float rtheta, float rv,
    const std::vector<Pose2D>& path,
    const PurePursuitParams& p)
{
    PurePursuitCommand cmd;
    if (path.empty()) {
        cmd.accel = -std::min(p.max_accel, std::abs(rv) * p.speed_gain) * (rv >= 0.0f ? 1.0f : -1.0f);
        cmd.steer = 0.0f;
        return cmd;
    }
    int closest = 0;
    float best = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < path.size(); i++) {
        float dx = path[i].x - rx;
        float dy = path[i].y - ry;
        float d = dx * dx + dy * dy;
        if (d < best) {
            best = d;
            closest = static_cast<int>(i);
        }
    }
    int target = closest;
    float L2 = p.lookahead * p.lookahead;
    while (target < static_cast<int>(path.size()) - 1) {
        float dx = path[target].x - rx;
        float dy = path[target].y - ry;
        if (dx * dx + dy * dy >= L2) break;
        target++;
    }
    cmd.target_waypoint = target;
    float dx = path[target].x - rx;
    float dy = path[target].y - ry;
    float dist_to_target = std::sqrt(dx * dx + dy * dy + 1e-6f);
    float alpha = detail::wrap_pi(std::atan2(dy, dx) - rtheta);
    cmd.steer = std::atan2(
        2.0f * p.wheelbase * std::sin(alpha), dist_to_target);
    if (cmd.steer > p.max_steer) cmd.steer = p.max_steer;
    if (cmd.steer < -p.max_steer) cmd.steer = -p.max_steer;

    float goal_dx = path.back().x - rx;
    float goal_dy = path.back().y - ry;
    float dist_to_goal = std::sqrt(goal_dx * goal_dx + goal_dy * goal_dy + 1e-6f);
    float target_v = p.target_speed;
    if (dist_to_goal < p.goal_slowdown_radius) {
        target_v *= (dist_to_goal / p.goal_slowdown_radius);
    }
    cmd.accel = p.speed_gain * (target_v - rv);
    if (cmd.accel > p.max_accel) cmd.accel = p.max_accel;
    if (cmd.accel < -p.max_accel) cmd.accel = -p.max_accel;
    return cmd;
}

}  // namespace cudabot

#endif  // CUDABOT_HYBRID_ASTAR_PP_H
