#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include "cuda_pointcloud.cuh"

namespace cudabot {

// Load ASCII or binary_little_endian PLY files containing x,y,z vertex data.
// Supports PLY files from ModelNet40, ScanNet, and other common sources.
inline bool load_ply(const std::string& path, std::vector<PointXYZ>& points) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open PLY file: %s\n", path.c_str());
        return false;
    }

    std::string line;
    int vertex_count = 0;
    bool is_binary = false;
    int x_idx = -1, y_idx = -1, z_idx = -1;
    int prop_count = 0;
    bool in_header = true;

    while (in_header && std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "element") {
            std::string elem_type;
            ss >> elem_type;
            if (elem_type == "vertex") {
                ss >> vertex_count;
            }
        } else if (token == "property") {
            std::string dtype, name;
            ss >> dtype >> name;
            if (name == "x") x_idx = prop_count;
            else if (name == "y") y_idx = prop_count;
            else if (name == "z") z_idx = prop_count;
            prop_count++;
        } else if (token == "format") {
            std::string fmt;
            ss >> fmt;
            is_binary = (fmt == "binary_little_endian");
        } else if (token == "end_header") {
            in_header = false;
        }
    }

    if (vertex_count <= 0 || x_idx < 0 || y_idx < 0 || z_idx < 0) {
        fprintf(stderr, "Invalid PLY header in %s (vertices=%d, x=%d, y=%d, z=%d)\n",
                path.c_str(), vertex_count, x_idx, y_idx, z_idx);
        return false;
    }

    points.resize(vertex_count);

    if (is_binary) {
        // Binary little-endian: each vertex is prop_count floats
        std::vector<float> buf(prop_count);
        for (int i = 0; i < vertex_count; i++) {
            file.read(reinterpret_cast<char*>(buf.data()), prop_count * sizeof(float));
            if (!file) {
                fprintf(stderr, "Truncated binary PLY at vertex %d in %s\n", i, path.c_str());
                points.resize(i);
                return i > 0;
            }
            points[i].x = buf[x_idx];
            points[i].y = buf[y_idx];
            points[i].z = buf[z_idx];
        }
    } else {
        // ASCII
        for (int i = 0; i < vertex_count; i++) {
            if (!std::getline(file, line)) {
                points.resize(i);
                return i > 0;
            }
            std::istringstream ss(line);
            std::vector<float> vals(prop_count);
            for (int j = 0; j < prop_count; j++) ss >> vals[j];
            points[i].x = vals[x_idx];
            points[i].y = vals[y_idx];
            points[i].z = vals[z_idx];
        }
    }

    printf("Loaded %d points from %s\n", vertex_count, path.c_str());
    return true;
}

// Load raw XYZ text file (one point per line: x y z)
inline bool load_xyz(const std::string& path, std::vector<PointXYZ>& points) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    points.clear();
    float x, y, z;
    while (file >> x >> y >> z) {
        points.push_back({x, y, z});
    }
    printf("Loaded %d points from %s\n", (int)points.size(), path.c_str());
    return !points.empty();
}

// Load KITTI binary format (.bin): N * (x,y,z,reflectance) as float32
inline bool load_kitti_bin(const std::string& path, std::vector<PointXYZ>& points) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;
    size_t file_size = file.tellg();
    file.seekg(0);
    int n = file_size / (4 * sizeof(float));
    points.resize(n);
    for (int i = 0; i < n; i++) {
        float buf[4];
        file.read(reinterpret_cast<char*>(buf), 4 * sizeof(float));
        points[i].x = buf[0];
        points[i].y = buf[1];
        points[i].z = buf[2];
        // buf[3] is reflectance, ignored
    }
    printf("Loaded %d points from %s\n", n, path.c_str());
    return true;
}

} // namespace cudabot
