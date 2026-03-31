#pragma once
#include <cuda_runtime.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Cart-Pole environment (OpenAI Gym compatible parameters)
struct CartPoleEnv {
    float x, x_dot, theta, theta_dot;
    int steps;
    bool done;

    // Physical constants
    static constexpr float GRAVITY   = 9.8f;
    static constexpr float MASSCART  = 1.0f;
    static constexpr float MASSPOLE  = 0.1f;
    static constexpr float LENGTH    = 0.5f;   // half-pole length
    static constexpr float DT        = 0.02f;
    static constexpr float FORCE_MAG = 10.0f;
    static constexpr int   MAX_STEPS = 200;

    __device__ void reset() {
        x = 0.0f;
        x_dot = 0.0f;
        theta = 0.05f;
        theta_dot = 0.0f;
        steps = 0;
        done = false;
    }

    __device__ void step(float action) {
        float force = action > 0.0f ? FORCE_MAG : -FORCE_MAG;
        float total_mass = MASSCART + MASSPOLE;
        float pole_mass_length = MASSPOLE * LENGTH;

        float costheta = cosf(theta);
        float sintheta = sinf(theta);

        // Equations of motion
        float temp = (force + pole_mass_length * theta_dot * theta_dot * sintheta) / total_mass;
        float theta_acc = (GRAVITY * sintheta - costheta * temp)
                        / (LENGTH * (4.0f / 3.0f - MASSPOLE * costheta * costheta / total_mass));
        float x_acc = temp - pole_mass_length * theta_acc * costheta / total_mass;

        // Euler integration
        x         += DT * x_dot;
        x_dot     += DT * x_acc;
        theta     += DT * theta_dot;
        theta_dot += DT * theta_acc;

        steps++;

        // Termination conditions
        if (fabsf(x) > 2.4f || fabsf(theta) > 12.0f * M_PI / 180.0f || steps >= MAX_STEPS) {
            done = true;
        }
    }

    __device__ float fitness() const { return (float)steps; }

    __device__ void get_observation(float* obs) const {
        obs[0] = x;
        obs[1] = x_dot;
        obs[2] = theta;
        obs[3] = theta_dot;
    }
};
