// 2-D EKF and particle-filter primitives corresponding to mathR/filter.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

#include "cudarobotics/lie_group_math.cuh"

namespace cudarobotics {
namespace filters {

struct State2D {
    float position[2] = {0.0f, 0.0f};
    float rotation[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float velocity[2] = {0.0f, 0.0f};

    void to_array(float* output) const {
        output[0] = position[0];
        output[1] = position[1];
        output[2] = atan2f(rotation[2], rotation[0]);
        output[3] = velocity[0];
        output[4] = velocity[1];
    }

    static State2D from_array(const float* input) {
        State2D result;
        result.position[0] = input[0];
        result.position[1] = input[1];
        lie::so2_exp(input[2], result.rotation);
        result.velocity[0] = input[3];
        result.velocity[1] = input[4];
        return result;
    }

    void retract(const float* delta, State2D* output) const {
        output->position[0] = position[0] + delta[0];
        output->position[1] = position[1] + delta[1];
        float dR[4];
        lie::so2_exp(delta[2], dR);
        output->rotation[0] = rotation[0] * dR[0] + rotation[1] * dR[2];
        output->rotation[1] = rotation[0] * dR[1] + rotation[1] * dR[3];
        output->rotation[2] = rotation[2] * dR[0] + rotation[3] * dR[2];
        output->rotation[3] = rotation[2] * dR[1] + rotation[3] * dR[3];
        output->velocity[0] = velocity[0] + delta[3];
        output->velocity[1] = velocity[1] + delta[4];
    }
};

inline void mat_mul(const float* A, int rows_a, int shared, const float* B,
                    int cols_b, float* C) {
    for (int row = 0; row < rows_a; ++row) {
        for (int col = 0; col < cols_b; ++col) {
            float value = 0.0f;
            for (int k = 0; k < shared; ++k) value += A[row * shared + k] * B[k * cols_b + col];
            C[row * cols_b + col] = value;
        }
    }
}

struct Odometry2DModel {
    float process_covariance[9] = {0.1f, 0.0f, 0.0f,
                                   0.0f, 0.1f, 0.0f,
                                   0.0f, 0.0f, 0.1f};

    void predict(const State2D& state,
                 const float* control,
                 float dt,
                 State2D* predicted,
                 float* F = nullptr,
                 float* G = nullptr) const {
        predicted->position[0] = state.position[0] + state.velocity[0] * dt;
        predicted->position[1] = state.position[1] + state.velocity[1] * dt;
        float dR[4];
        lie::so2_exp(control[2] * dt, dR);
        predicted->rotation[0] = state.rotation[0] * dR[0] + state.rotation[1] * dR[2];
        predicted->rotation[1] = state.rotation[0] * dR[1] + state.rotation[1] * dR[3];
        predicted->rotation[2] = state.rotation[2] * dR[0] + state.rotation[3] * dR[2];
        predicted->rotation[3] = state.rotation[2] * dR[1] + state.rotation[3] * dR[3];
        predicted->velocity[0] = predicted->rotation[0] * control[0] + predicted->rotation[1] * control[1];
        predicted->velocity[1] = predicted->rotation[2] * control[0] + predicted->rotation[3] * control[1];
        if (F != nullptr) {
            for (int i = 0; i < 25; ++i) F[i] = 0.0f;
            for (int i = 0; i < 5; ++i) F[5 * i + i] = 1.0f;
            F[3] = dt;
            F[9] = dt;
            F[17] = state.rotation[0] * (-control[1]) + state.rotation[1] * control[0];
            F[22] = state.rotation[2] * (-control[1]) + state.rotation[3] * control[0];
        }
        if (G != nullptr) {
            for (int i = 0; i < 15; ++i) G[i] = 0.0f;
            G[2 * 3 + 2] = -dt;
            G[3 * 3 + 0] = -state.rotation[0];
            G[3 * 3 + 1] = -state.rotation[1];
            G[4 * 3 + 0] = -state.rotation[2];
            G[4 * 3 + 1] = -state.rotation[3];
        }
    }
};

struct GPSModel2D {
    float measurement_covariance[4] = {0.1f, 0.0f, 0.0f, 0.1f};

    void measure(const State2D& state, float* position, float* H = nullptr) const {
        position[0] = state.position[0];
        position[1] = state.position[1];
        if (H != nullptr) {
            for (int i = 0; i < 10; ++i) H[i] = 0.0f;
            H[0] = 1.0f;
            H[6] = 1.0f;
        }
    }
};

inline bool invert2(const float* A, float* inverse) {
    const float determinant = A[0] * A[3] - A[1] * A[2];
    if (fabsf(determinant) < 1.0e-12f) return false;
    const float inv = 1.0f / determinant;
    inverse[0] = A[3] * inv;
    inverse[1] = -A[1] * inv;
    inverse[2] = -A[2] * inv;
    inverse[3] = A[0] * inv;
    return true;
}

class Ekf2D {
public:
    State2D state;
    float covariance[25] = {};
    Odometry2DModel motion;
    GPSModel2D measurement;

    Ekf2D() {
        for (int i = 0; i < 5; ++i) covariance[5 * i + i] = 0.1f;
    }

    void predict(const float* control, float dt) {
        State2D predicted;
        float F[25];
        float G[15];
        motion.predict(state, control, dt, &predicted, F, G);
        float temp[25];
        float Ft[25];
        for (int row = 0; row < 5; ++row) for (int col = 0; col < 5; ++col) Ft[5 * row + col] = F[5 * col + row];
        mat_mul(F, 5, 5, covariance, 5, temp);
        mat_mul(temp, 5, 5, Ft, 5, covariance);
        float GQ[15];
        float Gt[15];
        for (int row = 0; row < 5; ++row) for (int col = 0; col < 3; ++col) {
            GQ[3 * row + col] = 0.0f;
            for (int k = 0; k < 3; ++k) GQ[3 * row + col] += G[3 * row + k] * motion.process_covariance[3 * k + col];
        }
        for (int row = 0; row < 3; ++row) for (int col = 0; col < 5; ++col) Gt[5 * row + col] = G[3 * col + row];
        float GQGt[25];
        mat_mul(GQ, 5, 3, Gt, 5, GQGt);
        for (int i = 0; i < 25; ++i) covariance[i] += GQGt[i];
        state = predicted;
    }

    bool correct(const float* observation) {
        float predicted[2];
        float H[10];
        measurement.measure(state, predicted, H);
        float HP[10];
        float Ht[10];
        for (int row = 0; row < 2; ++row) for (int col = 0; col < 5; ++col) Ht[5 * row + col] = H[5 * col + row];
        mat_mul(H, 2, 5, covariance, 5, HP);
        float S[4];
        mat_mul(HP, 2, 5, Ht, 2, S);
        for (int i = 0; i < 4; ++i) S[i] += measurement.measurement_covariance[i];
        float S_inverse[4];
        if (!invert2(S, S_inverse)) return false;
        float PHt[10];
        mat_mul(covariance, 5, 5, Ht, 2, PHt);
        float K[10];
        mat_mul(PHt, 5, 2, S_inverse, 2, K);
        float innovation[2] = {observation[0] - predicted[0], observation[1] - predicted[1]};
        float delta[5] = {};
        for (int row = 0; row < 5; ++row) for (int col = 0; col < 2; ++col) delta[row] += K[2 * row + col] * innovation[col];
        State2D updated_state;
        state.retract(delta, &updated_state);
        state = updated_state;
        float KH[25];
        mat_mul(K, 5, 2, H, 5, KH);
        for (int row = 0; row < 5; ++row) for (int col = 0; col < 5; ++col) {
            const float identity = row == col ? 1.0f : 0.0f;
            KH[5 * row + col] = identity - KH[5 * row + col];
        }
        float updated[25];
        mat_mul(KH, 5, 5, covariance, 5, updated);
        for (int i = 0; i < 25; ++i) covariance[i] = updated[i];
        return true;
    }
};

class ParticleFilter2D {
public:
    ParticleFilter2D(int count, const State2D& initial)
        : particles_(static_cast<size_t>(std::max(1, count)) * 5),
          weights_(static_cast<size_t>(std::max(1, count)), 0.0f) {
        float state_array[5];
        initial.to_array(state_array);
        for (size_t i = 0; i < weights_.size(); ++i) {
            for (int j = 0; j < 5; ++j) particles_[i * 5 + j] = state_array[j];
            weights_[i] = 1.0f / static_cast<float>(weights_.size());
        }
    }

    size_t size() const { return weights_.size(); }

    void predict(const float* control, float dt, float noise_std, unsigned seed = 1) {
        std::mt19937 generator(seed);
        std::normal_distribution<float> normal(0.0f, noise_std);
        for (size_t i = 0; i < size(); ++i) {
            State2D state = State2D::from_array(&particles_[i * 5]);
            float noisy_control[3] = {control[0] + normal(generator),
                                      control[1] + normal(generator),
                                      control[2] + normal(generator)};
            State2D predicted;
            Odometry2DModel model;
            model.predict(state, noisy_control, dt, &predicted);
            predicted.to_array(&particles_[i * 5]);
        }
    }

    void correct(const float* observation, const float* covariance) {
        float inverse[4];
        if (!invert2(covariance, inverse)) return;
        const float determinant = covariance[0] * covariance[3] - covariance[1] * covariance[2];
        const float normalizer = 1.0f / (2.0f * 3.14159265358979323846f * sqrtf(fmaxf(determinant, 1.0e-20f)));
        for (size_t i = 0; i < size(); ++i) {
            const float dx = observation[0] - particles_[i * 5 + 0];
            const float dy = observation[1] - particles_[i * 5 + 1];
            const float mahalanobis = dx * (inverse[0] * dx + inverse[1] * dy) +
                                      dy * (inverse[2] * dx + inverse[3] * dy);
            weights_[i] *= normalizer * expf(-0.5f * mahalanobis) + 1.0e-30f;
        }
        normalize_weights();
    }

    void resample(unsigned seed = 2) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f / static_cast<float>(size()));
        std::vector<float> cumulative(size(), 0.0f);
        cumulative[0] = weights_[0];
        for (size_t i = 1; i < size(); ++i) cumulative[i] = cumulative[i - 1] + weights_[i];
        std::vector<float> output(particles_.size());
        const float start = uniform(generator);
        size_t index = 0;
        for (size_t i = 0; i < size(); ++i) {
            const float target = start + static_cast<float>(i) / static_cast<float>(size());
            while (index + 1 < size() && target > cumulative[index]) ++index;
            for (int j = 0; j < 5; ++j) output[i * 5 + j] = particles_[index * 5 + j];
        }
        particles_.swap(output);
        for (float& weight : weights_) weight = 1.0f / static_cast<float>(size());
    }

    void estimate(State2D* state, float* covariance) const {
        float mean[5] = {};
        for (size_t i = 0; i < size(); ++i) for (int j = 0; j < 5; ++j) mean[j] += weights_[i] * particles_[i * 5 + j];
        *state = State2D::from_array(mean);
        for (int i = 0; i < 25; ++i) covariance[i] = 0.0f;
        for (size_t i = 0; i < size(); ++i) {
            float delta[5];
            for (int j = 0; j < 5; ++j) delta[j] = particles_[i * 5 + j] - mean[j];
            for (int row = 0; row < 5; ++row) for (int col = 0; col < 5; ++col) covariance[5 * row + col] += weights_[i] * delta[row] * delta[col];
        }
    }

    float effective_sample_size() const {
        float squared_sum = 0.0f;
        for (float weight : weights_) squared_sum += weight * weight;
        return squared_sum > 1.0e-20f ? 1.0f / squared_sum : 0.0f;
    }

private:
    void normalize_weights() {
        float total = 0.0f;
        for (float weight : weights_) total += weight;
        if (total < 1.0e-30f) {
            for (float& weight : weights_) weight = 1.0f / static_cast<float>(size());
        } else {
            for (float& weight : weights_) weight /= total;
        }
    }

    std::vector<float> particles_;
    std::vector<float> weights_;
};

}  // namespace filters
}  // namespace cudarobotics
