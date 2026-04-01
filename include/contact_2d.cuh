#pragma once

#include <cuda_runtime.h>
#include <cmath>

#include "rigid_body_2d.cuh"

namespace cudabot {

static constexpr float CONTACT_K_SPRING = 10000.0f;
static constexpr float CONTACT_K_DAMPER = 100.0f;

__device__ inline void compute_contact_force(
    const RigidBody2D& a,
    const RigidBody2D& b,
    float& fx,
    float& fy)
{
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dist = sqrtf(dx * dx + dy * dy + 1.0e-6f);
    float overlap = a.radius + b.radius - dist;
    if (overlap <= 0.0f) {
        fx = 0.0f;
        fy = 0.0f;
        return;
    }

    float nx = dx / dist;
    float ny = dy / dist;
    float rel_vx = b.vx - a.vx;
    float rel_vy = b.vy - a.vy;
    float rel_normal = rel_vx * nx + rel_vy * ny;
    float force = CONTACT_K_SPRING * overlap - CONTACT_K_DAMPER * rel_normal;
    fx = -force * nx;
    fy = -force * ny;
}

__device__ inline void compute_wall_force(
    const RigidBody2D& body,
    float wall_y,
    float& fx,
    float& fy)
{
    float penetration = wall_y - (body.y - body.radius);
    if (penetration <= 0.0f) {
        fx = 0.0f;
        fy = 0.0f;
        return;
    }
    float rel_normal = -body.vy;
    float force = CONTACT_K_SPRING * penetration - CONTACT_K_DAMPER * rel_normal;
    fx = 0.0f;
    fy = force;
}

}  // namespace cudabot
