#pragma once

#include "autodiff_engine.cuh"
#include <cmath>

namespace cudabot {

struct BicycleParams {
    float L = 2.5f;
    float max_speed = 5.0f;
    float max_steer = 0.5f;
    float dt = 0.05f;
};

__host__ __device__ inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__host__ __device__ inline void bicycle_step(
    float& x, float& y, float& theta, float& v,
    float accel, float steer, const BicycleParams& p)
{
    float clamped_steer = clampf(steer, -p.max_steer, p.max_steer);
    v = clampf(v + accel * p.dt, 0.0f, p.max_speed);
    theta += v / p.L * tanf(clamped_steer) * p.dt;
    x += v * cosf(theta) * p.dt;
    y += v * sinf(theta) * p.dt;
}

__host__ __device__ inline void bicycle_step_diff(
    Dualf& x, Dualf& y, Dualf& theta, Dualf& v,
    Dualf accel, Dualf steer, const BicycleParams& p)
{
    v = clamp(v + accel * p.dt, 0.0f, p.max_speed);
    steer = clamp(steer, -p.max_steer, p.max_steer);
    theta = theta + v / Dualf::constant(p.L) * cudabot::tan(steer) * p.dt;
    x = x + v * cudabot::cos(theta) * p.dt;
    y = y + v * cudabot::sin(theta) * p.dt;
}

__device__ inline void bicycle_jacobian(
    float x, float y, float theta, float v,
    float accel, float steer, const BicycleParams& p,
    float J[4][6])
{
    for (int col = 0; col < 6; col++) {
        Dualf dx = (col == 0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dy = (col == 1) ? Dualf::variable(y) : Dualf::constant(y);
        Dualf dtheta = (col == 2) ? Dualf::variable(theta) : Dualf::constant(theta);
        Dualf dv = (col == 3) ? Dualf::variable(v) : Dualf::constant(v);
        Dualf da = (col == 4) ? Dualf::variable(accel) : Dualf::constant(accel);
        Dualf ds = (col == 5) ? Dualf::variable(steer) : Dualf::constant(steer);

        bicycle_step_diff(dx, dy, dtheta, dv, da, ds, p);
        J[0][col] = dx.deriv;
        J[1][col] = dy.deriv;
        J[2][col] = dtheta.deriv;
        J[3][col] = dv.deriv;
    }
}

}  // namespace cudabot
