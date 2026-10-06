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
    // The sensor: its height above the ground (the ground model starts from flat
    // ground this far below it; a feature of the class) and its upper beam's
    // elevation (a feature of the class: an object as tall as the beam reaches may
    // be cut off). The class was trained on 64-beam scans from 0.8-2.5 m with the
    // upper beam at +2 degrees.
    float sensor_height = 1.8f;        // m
    float upper_beam_deg = 2.0f;       // degrees
    // Refine the completed boxes with the free space the scan's rays crossed
    // (LidarCluster::refined; about 2 ms per vehicle on the GPU).
    bool free_space_refinement = true;
};

enum class LidarObjectClass : int { None = -1, Car = 0, Van = 1 };

// A box on the ground plane, in the frame of the input points: centred at the
// sensor, rotated by the sensor's yaw (z up).
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
    // The completed box moved, turned and resized within a small range to the
    // free space the scan's rays crossed and the cluster's points (the completed
    // box for None, or without free_space_refinement).
    LidarObjectBox refined;
};

struct LidarTrack {
    int id = -1;
    int cluster = -1;                     // the label of the cluster that updated it in this scan
    LidarObjectClass cls = LidarObjectClass::None;   // the majority of its clusters' classes
    bool moving = false;                  // the stand-still test's verdict
    float vx = 0.0f, vy = 0.0f;           // Kalman-filter velocity, m/s, in the world frame
    // The hybrid box with the size prior: this scan's box for a moving track
    // (traffic shows no new faces), the track's accumulated box otherwise.
    LidarObjectBox box;
};

struct LidarObjectsResult {
    std::vector<std::uint8_t> ground;     // per input point: 1 ground
    std::vector<int> cluster;             // per input point: its cluster's label, -1 for ground
    std::vector<LidarCluster> clusters;   // the clusters of at least 10 points, by label
    std::vector<LidarTrack> tracks;       // the tracks this scan updated
    float segmentation_ms = 0.0f, clustering_ms = 0.0f, boxes_ms = 0.0f, refinement_ms = 0.0f;   // GPU
    float tracking_ms = 0.0f;             // host, with the GPU box refits
};

class LidarObjectPipeline {
public:
    explicit LidarObjectPipeline(const LidarObjectsConfig& config = LidarObjectsConfig());
    ~LidarObjectPipeline();
    LidarObjectPipeline(const LidarObjectPipeline&) = delete;
    LidarObjectPipeline& operator=(const LidarObjectPipeline&) = delete;

    // xyz: n points (x, y, z) in the sensor's frame, z up (a level sensor: no
    // roll or pitch). sensor_x, _y, _z: the sensor's world position; sensor_yaw:
    // its heading in the world (radians); t: the scan's time in seconds,
    // increasing. The results are in the frame of the points, except the tracks'
    // velocities (world frame).
    LidarObjectsResult process(const float* xyz, std::size_t n, double sensor_x, double sensor_y, double sensor_z,
                               double t, double sensor_yaw = 0.0);

    // Forget all tracks.
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cudarobotics
