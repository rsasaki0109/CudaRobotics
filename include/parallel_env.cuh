#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace cudabot {

static constexpr int PARALLEL_CARTPOLE_OBS_DIM = 4;
static constexpr int PARALLEL_CARTPOLE_ACTION_DIM = 1;
static constexpr int PARALLEL_CARTPOLE_MAX_STEPS = 200;

__device__ inline float parallel_env_clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ inline float parallel_env_rand01(int idx, unsigned int seed) {
    unsigned int x = static_cast<unsigned int>(idx) * 747796405u + seed * 2891336453u + 12345u;
    x = (x >> ((x >> 28) + 4)) ^ x;
    x *= 277803737u;
    x = (x >> 22) ^ x;
    return static_cast<float>(x & 0x00ffffff) / static_cast<float>(0x01000000);
}

__global__ inline void cartpole_reset_kernel(float* d_states, int* d_steps, int* d_done,
                                             int n_envs, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_envs) return;

    float theta0 = (parallel_env_rand01(idx, seed) - 0.5f) * 0.10f;
    float x0 = (parallel_env_rand01(idx, seed + 17) - 0.5f) * 0.08f;
    d_states[idx * 4 + 0] = x0;
    d_states[idx * 4 + 1] = 0.0f;
    d_states[idx * 4 + 2] = theta0;
    d_states[idx * 4 + 3] = 0.0f;
    d_steps[idx] = 0;
    d_done[idx] = 0;
}

__global__ inline void cartpole_observe_kernel(const float* d_states, float* d_obs, int n_envs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_envs) return;
    for (int i = 0; i < 4; i++) d_obs[idx * 4 + i] = d_states[idx * 4 + i];
}

__global__ inline void cartpole_step_kernel(
    float* d_states,
    int* d_steps,
    int* d_done_state,
    const float* d_actions,
    float* d_obs,
    float* d_rewards,
    int* d_done_out,
    int n_envs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_envs) return;

    if (d_done_state[idx]) {
        if (d_rewards) d_rewards[idx] = 0.0f;
        if (d_done_out) d_done_out[idx] = 1;
        if (d_obs) {
            for (int i = 0; i < 4; i++) d_obs[idx * 4 + i] = d_states[idx * 4 + i];
        }
        return;
    }

    float x = d_states[idx * 4 + 0];
    float x_dot = d_states[idx * 4 + 1];
    float theta = d_states[idx * 4 + 2];
    float theta_dot = d_states[idx * 4 + 3];

    const float gravity = 9.8f;
    const float masscart = 1.0f;
    const float masspole = 0.1f;
    const float total_mass = masscart + masspole;
    const float length = 0.5f;
    const float polemass_length = masspole * length;
    const float force_mag = 10.0f;
    const float tau = 0.02f;
    const float theta_threshold = 12.0f * 3.14159265358979323846f / 180.0f;
    const float x_threshold = 2.4f;

    float action = d_actions ? d_actions[idx] : 0.0f;
    float force = parallel_env_clampf(action, -1.0f, 1.0f) * force_mag;
    float costheta = cosf(theta);
    float sintheta = sinf(theta);
    float temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
    float thetaacc = (gravity * sintheta - costheta * temp) /
        (length * (4.0f / 3.0f - masspole * costheta * costheta / total_mass));
    float xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    x += tau * x_dot;
    x_dot += tau * xacc;
    theta += tau * theta_dot;
    theta_dot += tau * thetaacc;

    int steps = d_steps[idx] + 1;
    int done = (fabsf(x) > x_threshold || fabsf(theta) > theta_threshold ||
                steps >= PARALLEL_CARTPOLE_MAX_STEPS) ? 1 : 0;

    d_states[idx * 4 + 0] = x;
    d_states[idx * 4 + 1] = x_dot;
    d_states[idx * 4 + 2] = theta;
    d_states[idx * 4 + 3] = theta_dot;
    d_steps[idx] = steps;
    d_done_state[idx] = done;

    if (d_obs) {
        d_obs[idx * 4 + 0] = x;
        d_obs[idx * 4 + 1] = x_dot;
        d_obs[idx * 4 + 2] = theta;
        d_obs[idx * 4 + 3] = theta_dot;
    }
    if (d_rewards) d_rewards[idx] = done ? 0.0f : 1.0f;
    if (d_done_out) d_done_out[idx] = done;
}

class ParallelEnv {
public:
    ParallelEnv(int n_envs, int env_type)
        : d_states_(nullptr), d_steps_(nullptr), d_done_(nullptr), n_envs_(n_envs), env_type_(env_type)
    {
        cudaMalloc(&d_states_, n_envs_ * PARALLEL_CARTPOLE_OBS_DIM * sizeof(float));
        cudaMalloc(&d_steps_, n_envs_ * sizeof(int));
        cudaMalloc(&d_done_, n_envs_ * sizeof(int));
        reset_all();
    }

    ~ParallelEnv() {
        if (d_states_) cudaFree(d_states_);
        if (d_steps_) cudaFree(d_steps_);
        if (d_done_) cudaFree(d_done_);
    }

    void reset_all(unsigned int seed = 1) {
        int threads = 256;
        int blocks = (n_envs_ + threads - 1) / threads;
        cartpole_reset_kernel<<<blocks, threads>>>(d_states_, d_steps_, d_done_, n_envs_, seed);
        cudaDeviceSynchronize();
    }

    void observe(float* d_obs) const {
        int threads = 256;
        int blocks = (n_envs_ + threads - 1) / threads;
        cartpole_observe_kernel<<<blocks, threads>>>(d_states_, d_obs, n_envs_);
    }

    void step(const float* d_actions, float* d_obs, float* d_rewards, int* d_dones) {
        int threads = 256;
        int blocks = (n_envs_ + threads - 1) / threads;
        cartpole_step_kernel<<<blocks, threads>>>(
            d_states_, d_steps_, d_done_, d_actions, d_obs, d_rewards, d_dones, n_envs_);
    }

    int obs_dim() const { return PARALLEL_CARTPOLE_OBS_DIM; }
    int action_dim() const { return PARALLEL_CARTPOLE_ACTION_DIM; }
    int n_envs() const { return n_envs_; }
    float* states_device() const { return d_states_; }
    int* steps_device() const { return d_steps_; }
    int* done_device() const { return d_done_; }
    int env_type() const { return env_type_; }

private:
    float* d_states_;
    int* d_steps_;
    int* d_done_;
    int n_envs_;
    int env_type_;
};

}  // namespace cudabot
