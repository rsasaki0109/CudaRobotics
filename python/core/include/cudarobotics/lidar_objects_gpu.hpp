#pragma once

// LiDAR object pipeline on the GPU: ground segmentation, Euclidean clustering,
// L-shape boxes with a learned class and a size-prior completion, and tracking
// with a constant-velocity Kalman filter. One call per scan.
//
// The algorithms live in cudarobotics/lidar_objects_core.cuh; this interface
// needs no CUDA headers. The demo src/gpu_ground_segmentation.cu evaluates the
// pipeline and checks that this class gives exactly its results
// (docs/gpu_ground_segmentation.md).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cudarobotics {

struct LidarObjectsConfig {
    std::size_t max_points = 131072;   // per scan
    int track_min_scans = 3;           // a tracked voxel counts once seen in this many scans
};

enum class LidarObjectClass : int { None = -1, Car = 0, Van = 1 };

// A box on the ground plane. Frame: centred at the sensor, axes aligned with the
// world (z up), as the input points.
struct LidarObjectBox {
    float cx = 0.0f, cy = 0.0f;           // centre
    float length = 0.0f, width = 0.0f;    // length along yaw
    float yaw = 0.0f;                     // radians
    float z_min = 0.0f, z_max = 0.0f;     // of the box's points
    int points = 0;
};

struct LidarCluster {
    int label = -1;                       // the smallest index of its points
    LidarObjectBox box;                   // L-shape box of its points
    LidarObjectClass cls = LidarObjectClass::None;
    LidarObjectBox completed;             // box completed with the class's size prior (the box for None)
};

struct LidarTrack {
    int id = -1;
    LidarObjectClass cls = LidarObjectClass::None;   // the majority of its clusters' classes
    bool moving = false;                  // the stand-still test's verdict
    float vx = 0.0f, vy = 0.0f;           // Kalman-filter velocity, m/s
    // The hybrid box with the size prior: this scan's box for a moving track
    // (traffic shows no new faces), the track's accumulated box otherwise.
    LidarObjectBox box;
};

struct LidarObjectsResult {
    std::vector<std::uint8_t> ground;     // per input point: 1 ground
    std::vector<int> cluster;             // per input point: its cluster's label, -1 for ground
    std::vector<LidarCluster> clusters;   // the clusters of at least 10 points, by label
    std::vector<LidarTrack> tracks;       // the tracks this scan updated
    float segmentation_ms = 0.0f, clustering_ms = 0.0f, boxes_ms = 0.0f;   // GPU
    float tracking_ms = 0.0f;             // host, with the GPU box refits
};

class LidarObjectPipeline {
public:
    explicit LidarObjectPipeline(const LidarObjectsConfig& config = LidarObjectsConfig());
    ~LidarObjectPipeline();
    LidarObjectPipeline(const LidarObjectPipeline&) = delete;
    LidarObjectPipeline& operator=(const LidarObjectPipeline&) = delete;

    // xyz: n points (x, y, z) relative to the sensor, axes aligned with the world
    // (z up). The ground model assumes the sensor 1.8 m above the ground, and the
    // class a 64-beam scan with its upper beam at +2 degrees. sensor_x, _y, _z:
    // the sensor's world position; t: the scan's time in seconds, increasing.
    LidarObjectsResult process(const float* xyz, std::size_t n, double sensor_x, double sensor_y, double sensor_z,
                               double t);

    // Forget all tracks.
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cudarobotics
