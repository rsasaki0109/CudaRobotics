// lidar_objects_gpu.cu
//
// cudarobotics::LidarObjectPipeline: the production path of the LiDAR object
// pipeline in cudarobotics/lidar_objects_core.cuh, one call per scan:
//
//   GPU   ground segmentation -> voxel clustering of the non-ground points
//         -> L-shape box per cluster of >= CL_MIN points
//   host  learned class and size-prior completion per cluster
//         -> motion tracker (Kalman filter, object-frame accumulation), its box
//            refits batched on the GPU -> hybrid box per updated track
//
// src/gpu_ground_segmentation.cu runs the same steps with its CPU references
// and checks that this class gives exactly its results.

#include "cudarobotics/lidar_objects_gpu.hpp"
#include "cudarobotics/lidar_objects_core.cuh"

#include <stdexcept>

namespace cudarobotics {

namespace {

LidarObjectBox to_box(const cudabot::Obb& B) {
    LidarObjectBox o;
    o.cx = B.cx; o.cy = B.cy; o.length = B.len; o.width = B.wid; o.yaw = B.yaw;
    o.z_min = B.zlo; o.z_max = B.zhi; o.points = B.n;
    return o;
}

LidarObjectClass to_class(int k) { return k < 0 ? LidarObjectClass::None : static_cast<LidarObjectClass>(k); }

}  // namespace

struct LidarObjectPipeline::Impl {
    LidarObjectsConfig cfg;
    int cap;
    float* d_pts = nullptr;
    int* d_valid = nullptr;
    cudabot::GpuGroundSegmenter seg;
    cudabot::GpuVoxelClusterer vcl;
    cudabot::GpuLShape lsh;
    cudabot::GpuBoxFitter fitter;
    std::vector<float> cs;
    cudabot::Tracker trk;
    int scan = 0;

    explicit Impl(const LidarObjectsConfig& c)
        : cfg(c), cap((int)c.max_points), seg(cap), vcl(cap), lsh(cap) {
        CUDA_CHECK(cudaMalloc(&d_pts, (size_t)cap * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_valid, (size_t)cap * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_valid, 0, (size_t)cap * sizeof(int)));   // every input point is a return
        cudabot::heading_table(cs);
        reset();
    }
    ~Impl() {
        cudaFree(d_pts);
        cudaFree(d_valid);
    }
    void reset() {
        trk = cudabot::Tracker();
        trk.motion = true;
        trk.trk_hits = cfg.track_min_scans;
        scan = 0;
    }
};

LidarObjectPipeline::LidarObjectPipeline(const LidarObjectsConfig& config) : impl_(new Impl(config)) {}

LidarObjectPipeline::~LidarObjectPipeline() = default;

void LidarObjectPipeline::reset() { impl_->reset(); }

LidarObjectsResult LidarObjectPipeline::process(const float* xyz, std::size_t n_points, double sensor_x,
                                                double sensor_y, double sensor_z, double t_scan) {
    using namespace cudabot;
    Impl& I = *impl_;
    if (n_points > (std::size_t)I.cap) throw std::length_error("LidarObjectPipeline: more points than max_points");
    const int n = (int)n_points;
    const float px = (float)sensor_x, py = (float)sensor_y, pz = (float)sensor_z, t = (float)t_scan;
    LidarObjectsResult R;

    // ---- ground segmentation ----
    CUDA_CHECK(cudaMemcpy(I.d_pts, xyz, (size_t)n * 3 * sizeof(float), cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    if (n > 0) I.seg.run(I.d_pts, I.d_valid, n);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&R.segmentation_ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
    std::vector<int> lab(n);
    if (n > 0) CUDA_CHECK(cudaMemcpy(lab.data(), I.seg.d_lab, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
    R.ground.resize(n);
    std::vector<int> active(n);
    for (int i = 0; i < n; ++i) { R.ground[i] = (std::uint8_t)(lab[i] != 0); active[i] = lab[i] == 0; }

    // ---- clustering and boxes ----
    int n_vox = 0;
    R.clustering_ms = n > 0 ? I.vcl.run(I.d_pts, active, R.cluster, n_vox) : 0.0f;
    std::vector<int> gkeys, keys;
    std::vector<Obb> gobb, obb;
    if (n > 0) R.boxes_ms = I.lsh.run(I.d_pts, I.vcl.d_label, n, gkeys, gobb);
    for (size_t r = 0; r < gkeys.size(); ++r)
        if (gobb[r].th >= 0) { keys.push_back(gkeys[r]); obb.push_back(gobb[r]); }
    std::vector<int> ccls(keys.size());
    std::vector<Obb> done(keys.size());
    for (size_t r = 0; r < keys.size(); ++r) {
        ccls[r] = classify_box_mlp(obb[r]);
        done[r] = complete_box(obb[r], ccls[r], 1.0f);
        LidarCluster C;
        C.label = keys[r]; C.box = to_box(obb[r]); C.cls = to_class(ccls[r]); C.completed = to_box(done[r]);
        R.clusters.push_back(C);
    }

    // ---- tracking: the clusters classed as cars or vans ----
    auto h0 = std::chrono::high_resolution_clock::now();
    std::vector<float> pts(xyz, xyz + (size_t)n * 3);
    std::vector<int> cand, slot(n, -1);
    for (size_t r = 0; r < keys.size(); ++r)
        if (ccls[r] >= 0) { slot[keys[r]] = (int)cand.size(); cand.push_back((int)r); }
    std::vector<std::vector<int>> citems(cand.size());
    for (int i = 0; i < n; ++i)
        if (R.cluster[i] >= 0 && slot[R.cluster[i]] >= 0) citems[slot[R.cluster[i]]].push_back(i);
    I.trk.update(t, px, py, pz, I.scan, cand, ccls, citems, keys, obb, pts, I.cs);
    bool same = true;
    I.trk.fit(&I.fitter, I.cs, false, same);
    for (size_t k = 0; k < cand.size(); ++k) {
        int j = I.trk.track_of[keys[cand[k]]];
        const Track& T = I.trk.tracks[j];
        const MotionState& M = I.trk.ms[j];
        Obb raw, tracked;
        I.trk.boxes(j, t, px, py, raw, tracked);
        LidarTrack out;
        out.id = j;
        int tc = 0;
        for (int c = 1; c < N_CLS; ++c) if (T.votes[c] > T.votes[tc]) tc = c;
        out.cls = to_class(tc);
        out.moving = M.moving;
        out.vx = (float)M.x[2]; out.vy = (float)M.x[3];
        out.box = to_box(M.moving ? done[cand[k]] : tracked);   // the hybrid box
        R.tracks.push_back(out);
    }
    R.tracking_ms = (float)std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - h0).count();
    ++I.scan;
    return R;
}

}  // namespace cudarobotics
