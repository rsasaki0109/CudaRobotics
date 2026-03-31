#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "gpu_neural_net.cuh"
#include "gpu_environments.cuh"

// Initialize cuRAND states
__global__ void init_curand_kernel(curandState* states, int n, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// Fitness evaluation kernel: 1 thread = 1 individual, runs full Cart-Pole episode
__global__ void evaluate_fitness_kernel(
    const float* population_weights,  // [pop_size * n_weights]
    float* fitness,                   // [pop_size]
    curandState* rng,
    int pop_size, int n_weights,
    int input_dim, int output_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    const float* my_weights = population_weights + idx * n_weights;

    CartPoleEnv env;
    env.reset();

    float obs[4];
    float action_out[1];

    while (!env.done) {
        env.get_observation(obs);
        nn_forward(my_weights, obs, action_out, input_dim, output_dim);
        // Use tanh to squash output to [-1, 1], then threshold at 0
        float action = tanhf(action_out[0]);
        env.step(action);
    }

    fitness[idx] = env.fitness();
}

// Find the index of the best individual (host-side helper)
inline int find_best_index(const float* h_fitness, int pop_size)
{
    int best = 0;
    for (int i = 1; i < pop_size; i++) {
        if (h_fitness[i] > h_fitness[best]) best = i;
    }
    return best;
}

// Reproduction kernel: tournament selection + crossover + mutation
// 1 thread = 1 new individual
__global__ void reproduce_kernel(
    const float* old_population,   // [pop_size * n_weights]
    const float* fitness,          // [pop_size]
    float* new_population,         // [pop_size * n_weights]
    const int* elite_indices,      // [elite_count] indices sorted by fitness (best first)
    curandState* rng,
    int pop_size, int n_weights,
    int tournament_size,    // 5
    float crossover_rate,   // 0.8
    float mutation_sigma,   // 0.1
    int elite_count)        // 10
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    // Elite: copy directly
    if (idx < elite_count) {
        int src = elite_indices[idx];
        for (int w = 0; w < n_weights; w++) {
            new_population[idx * n_weights + w] = old_population[src * n_weights + w];
        }
        return;
    }

    curandState local_rng = rng[idx];

    // Tournament selection for parent1
    int parent1 = (int)(curand_uniform(&local_rng) * pop_size) % pop_size;
    for (int t = 1; t < tournament_size; t++) {
        int candidate = (int)(curand_uniform(&local_rng) * pop_size) % pop_size;
        if (fitness[candidate] > fitness[parent1]) parent1 = candidate;
    }

    // Tournament selection for parent2
    int parent2 = (int)(curand_uniform(&local_rng) * pop_size) % pop_size;
    for (int t = 1; t < tournament_size; t++) {
        int candidate = (int)(curand_uniform(&local_rng) * pop_size) % pop_size;
        if (fitness[candidate] > fitness[parent2]) parent2 = candidate;
    }

    // Uniform crossover + mutation
    for (int w = 0; w < n_weights; w++) {
        float gene;
        if (curand_uniform(&local_rng) < crossover_rate) {
            gene = (curand_uniform(&local_rng) < 0.5f)
                   ? old_population[parent1 * n_weights + w]
                   : old_population[parent2 * n_weights + w];
        } else {
            gene = old_population[parent1 * n_weights + w];
        }

        // Mutation
        gene += curand_normal(&local_rng) * mutation_sigma;

        new_population[idx * n_weights + w] = gene;
    }

    rng[idx] = local_rng;
}
