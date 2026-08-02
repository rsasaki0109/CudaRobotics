// Minimal BAL dataset loader corresponding to mathR/slam/load_ba_datasets.py.

#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "cudarobotics/lie_group_math.cuh"
#include "cudarobotics/projection.hpp"

namespace cudarobotics {
namespace bal {

struct Camera {
    float R[9] = {1.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f,
                 0.0f, 0.0f, 1.0f};
    float t[3] = {0.0f, 0.0f, 0.0f};
    float K[9] = {1.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f,
                 0.0f, 0.0f, 1.0f};
    float distortion[2] = {0.0f, 0.0f};
};

struct Observation {
    int camera_id = -1;
    int point_id = -1;
    float u[2] = {0.0f, 0.0f};
    float u_undistorted[2] = {0.0f, 0.0f};
};

struct Dataset {
    std::vector<Camera> cameras;
    std::vector<Observation> observations;
    std::vector<float> points;  // xyz triplets
};

inline bool parse(std::istream& input, Dataset* dataset) {
    if (dataset == nullptr) return false;
    int camera_count = 0;
    int point_count = 0;
    int observation_count = 0;
    if (!(input >> camera_count >> point_count >> observation_count)) return false;
    if (camera_count < 0 || point_count < 0 || observation_count < 0) return false;
    dataset->cameras.clear();
    dataset->observations.clear();
    dataset->points.clear();
    dataset->cameras.resize(camera_count);
    dataset->observations.resize(observation_count);
    for (Observation& observation : dataset->observations) {
        float camera_id;
        float point_id;
        if (!(input >> camera_id >> point_id >> observation.u[0] >> observation.u[1])) return false;
        observation.camera_id = static_cast<int>(camera_id);
        observation.point_id = static_cast<int>(point_id);
        observation.u[0] = -observation.u[0];
        observation.u[1] = -observation.u[1];
        observation.u_undistorted[0] = observation.u[0];
        observation.u_undistorted[1] = observation.u[1];
    }
    for (Camera& camera : dataset->cameras) {
        float rotation_vector[3];
        float translation[3];
        if (!(input >> rotation_vector[0] >> rotation_vector[1] >> rotation_vector[2] >>
              translation[0] >> translation[1] >> translation[2])) return false;
        float T_cw[16];
        float T_wc[16];
        float R_cw[9];
        lie::so3_exp(rotation_vector, R_cw);
        lie::mat4_identity(T_cw);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) T_cw[4 * row + col] = R_cw[3 * row + col];
            T_cw[4 * row + 3] = translation[row];
        }
        projection::rigid_inverse(T_cw, T_wc);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) camera.R[3 * row + col] = T_wc[4 * row + col];
            camera.t[row] = T_wc[4 * row + 3];
        }
        float focal;
        if (!(input >> focal >> camera.distortion[0] >> camera.distortion[1])) return false;
        camera.K[0] = focal;
        camera.K[1] = 0.0f;
        camera.K[2] = 0.0f;
        camera.K[3] = 0.0f;
        camera.K[4] = focal;
        camera.K[5] = 0.0f;
        camera.K[6] = 0.0f;
        camera.K[7] = 0.0f;
        camera.K[8] = 1.0f;
    }
    dataset->points.resize(static_cast<size_t>(point_count) * 3);
    for (float& value : dataset->points) if (!(input >> value)) return false;
    return true;
}

inline bool load(const std::string& filename, Dataset* dataset) {
    std::ifstream input(filename.c_str());
    return input.good() && parse(input, dataset);
}

}  // namespace bal
}  // namespace cudarobotics
