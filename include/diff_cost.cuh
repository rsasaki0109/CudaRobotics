#pragma once

#include "autodiff_engine.cuh"

namespace cudabot {

struct Obstacle {
    float x;
    float y;
    float r;
};

struct CostParams {
    float goal_x = 45.0f;
    float goal_y = 45.0f;
    float goal_weight = 5.0f;
    float control_weight = 0.1f;
    float speed_weight = 0.15f;
    float target_speed = 3.5f;
    float heading_weight = 0.35f;
    float obs_weight = 10.0f;
    float obs_influence = 5.0f;
    float terminal_weight = 8.0f;
};

__host__ __device__ inline Dualf obstacle_cost_diff(
    Dualf px, Dualf py,
    const Obstacle* obs, int n_obs,
    float influence, float weight)
{
    Dualf cost = Dualf::constant(0.0f);
    for (int i = 0; i < n_obs; i++) {
        Dualf dx = px - Dualf::constant(obs[i].x);
        Dualf dy = py - Dualf::constant(obs[i].y);
        Dualf d = cudabot::sqrt(dx * dx + dy * dy + Dualf::constant(1e-6f))
                - Dualf::constant(obs[i].r);
        if (d.val < influence && d.val > 0.1f) {
            cost = cost + Dualf::constant(weight) / (d * d);
        } else if (d.val <= 0.1f) {
            cost = cost + Dualf::constant(weight * 100.0f);
        }
    }
    return cost;
}

__host__ __device__ inline Dualf goal_cost_diff(
    Dualf px, Dualf py, float gx, float gy, float weight)
{
    Dualf dx = px - Dualf::constant(gx);
    Dualf dy = py - Dualf::constant(gy);
    return Dualf::constant(weight)
         * cudabot::sqrt(dx * dx + dy * dy + Dualf::constant(0.01f));
}

__host__ __device__ inline Dualf control_cost_diff(Dualf a, Dualf s, float weight)
{
    return Dualf::constant(weight) * (a * a + s * s);
}

__host__ __device__ inline Dualf speed_cost_diff(Dualf v, float target_speed, float weight)
{
    Dualf dv = v - Dualf::constant(target_speed);
    return Dualf::constant(weight) * (dv * dv);
}

__host__ __device__ inline Dualf heading_cost_diff(
    Dualf px, Dualf py, Dualf theta, float gx, float gy, float weight)
{
    Dualf desired = cudabot::atan2(Dualf::constant(gy) - py, Dualf::constant(gx) - px);
    Dualf err = theta - desired;
    return Dualf::constant(weight) * (err * err);
}

}  // namespace cudabot
