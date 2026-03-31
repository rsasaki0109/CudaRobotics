#pragma once

#include <cuda_runtime.h>

namespace cudabot {

static constexpr float RIGID_BODY_GRAVITY = -9.81f;

struct RigidBody2D {
    float x;
    float y;
    float angle;
    float vx;
    float vy;
    float omega;
    float mass;
    float inertia;
    float radius;
};

__device__ inline void integrate(RigidBody2D& body, float fx, float fy, float torque, float dt) {
    body.vx += fx / body.mass * dt;
    body.vy += (fy / body.mass + RIGID_BODY_GRAVITY) * dt;
    body.omega += torque / body.inertia * dt;
    body.x += body.vx * dt;
    body.y += body.vy * dt;
    body.angle += body.omega * dt;
}

}  // namespace cudabot
