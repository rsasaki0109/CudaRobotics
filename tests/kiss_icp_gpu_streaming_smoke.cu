#include "cudarobotics/kiss_icp_gpu.hpp"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

bool near(float a, float b, float tolerance = 1e-5f) {
    return std::fabs(a - b) <= tolerance;
}

}  // namespace

int main() {
    using namespace cudarobotics;

    KissIcpConfig invalid;
    invalid.hash_capacity = 1000;
    if (validate_kiss_icp_config(invalid).empty()) {
        std::fprintf(stderr, "non-power-of-two hash capacity was accepted\n");
        return 1;
    }

    KissIcpConfig config;
    config.map_voxel_size = 0.1f;
    config.scan_voxel_size = 0.05f;
    config.max_scan_points = 128;
    config.max_map_points = 256;
    config.hash_capacity = 512;
    config.downsample_backend = KissIcpDownsampleBackend::Validate;

    KissIcpPose initial;
    initial.t[0] = 2.0f;
    initial.t[1] = -1.0f;
    initial.t[2] = 0.5f;

    std::vector<float> scan;
    for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
            scan.push_back(0.25f * x);
            scan.push_back(0.25f * y);
            scan.push_back(0.03f * (x * x + y));
        }
    }

    KissIcpOdometry odometry(config);
    odometry.reset(initial);
    KissIcpFrameResult first = odometry.register_scan(scan);
    if (odometry.frame_count() != 1 || first.map_points == 0 ||
        odometry.map_point_count() != first.map_points ||
        !near(first.pose.t[0], initial.t[0]) ||
        !near(first.pose.t[1], initial.t[1]) ||
        !near(first.pose.t[2], initial.t[2])) {
        std::fprintf(stderr, "first scan did not preserve the explicit initial pose\n");
        return 2;
    }

    KissIcpFrameResult second = odometry.register_scan(scan);
    if (odometry.frame_count() != 2 || second.alignment.inliers < 10 ||
        !near(second.pose.t[0], initial.t[0], 1e-3f) ||
        !near(second.pose.t[1], initial.t[1], 1e-3f) ||
        !near(second.pose.t[2], initial.t[2], 1e-3f)) {
        std::fprintf(stderr, "streaming registration changed a stationary scan\n");
        return 3;
    }

    std::vector<float> point_times(scan.size() / 3);
    for (std::size_t index = 0; index < point_times.size(); ++index) {
        point_times[index] =
            0.1f * static_cast<float>(index) /
            static_cast<float>(point_times.size() - 1);
    }
    KissIcpFrameResult timed = odometry.register_scan(scan, point_times);
    if (!timed.deskewed || !near(timed.point_time_span_s, 0.1f) ||
        odometry.timing().deskew_ms < 0.0 ||
        !near(timed.pose.t[0], initial.t[0], 1e-3f) ||
        !near(timed.pose.t[1], initial.t[1], 1e-3f) ||
        !near(timed.pose.t[2], initial.t[2], 1e-3f)) {
        std::fprintf(stderr, "timed stationary scan deskew contract failed\n");
        return 4;
    }

    KissIcpOdometry moving_odometry(config);
    moving_odometry.register_scan(scan);
    std::vector<float> shifted_scan = scan;
    for (std::size_t index = 0; index < shifted_scan.size(); index += 3) {
        shifted_scan[index] -= 0.1f;
    }
    const KissIcpFrameResult shifted =
        moving_odometry.register_scan(shifted_scan);
    const KissIcpFrameResult moving_timed =
        moving_odometry.register_scan(shifted_scan, point_times);
    const std::size_t last_xyz = moving_timed.deskewed_xyz.size() - 3;
    const float first_correction =
        std::fabs(moving_timed.deskewed_xyz[0] - shifted_scan[0]) +
        std::fabs(moving_timed.deskewed_xyz[1] - shifted_scan[1]) +
        std::fabs(moving_timed.deskewed_xyz[2] - shifted_scan[2]);
    if (shifted.alignment.inliers < 10 ||
        moving_timed.deskewed_xyz.size() != shifted_scan.size() ||
        first_correction <= 1e-3f ||
        !near(moving_timed.deskewed_xyz[last_xyz], shifted_scan[last_xyz],
              1e-4f) ||
        !near(moving_timed.deskewed_xyz[last_xyz + 1],
              shifted_scan[last_xyz + 1], 1e-4f) ||
        !near(moving_timed.deskewed_xyz[last_xyz + 2],
              shifted_scan[last_xyz + 2], 1e-4f)) {
        std::fprintf(stderr, "nonzero-motion GPU deskew contract failed\n");
        return 5;
    }

    KissIcpOdometry bounded_odometry(config);
    bounded_odometry.register_scan(scan);
    bounded_odometry.register_scan(shifted_scan);
    std::vector<float> filtered_times(point_times.size());
    for (std::size_t index = 0; index < filtered_times.size(); ++index) {
        filtered_times[index] =
            0.02f + 0.06f * static_cast<float>(index) /
                        static_cast<float>(filtered_times.size() - 1);
    }
    const KissIcpFrameResult bounded_timed = bounded_odometry.register_scan(
        shifted_scan.data(),
        shifted_scan.size() / 3,
        filtered_times.data(),
        0.0f,
        0.1f);
    const std::size_t bounded_last = bounded_timed.deskewed_xyz.size() - 3;
    const float bounded_last_correction =
        std::fabs(
            bounded_timed.deskewed_xyz[bounded_last] -
            shifted_scan[bounded_last]) +
        std::fabs(
            bounded_timed.deskewed_xyz[bounded_last + 1] -
            shifted_scan[bounded_last + 1]) +
        std::fabs(
            bounded_timed.deskewed_xyz[bounded_last + 2] -
            shifted_scan[bounded_last + 2]);
    if (!near(bounded_timed.point_time_span_s, 0.1f) ||
        bounded_last_correction <= 1e-3f) {
        std::fprintf(stderr, "explicit scan-time bounds were not preserved\n");
        return 6;
    }

    bool rejected_bad_shape = false;
    try {
        odometry.register_scan(std::vector<float>{0.0f, 1.0f});
    } catch (const std::invalid_argument&) {
        rejected_bad_shape = true;
    }
    if (!rejected_bad_shape) {
        std::fprintf(stderr, "malformed xyz vector was accepted\n");
        return 7;
    }

    bool rejected_bad_times = false;
    try {
        odometry.register_scan(
            scan, std::vector<float>(scan.size() / 3, 0.0f));
    } catch (const std::invalid_argument&) {
        rejected_bad_times = true;
    }
    if (!rejected_bad_times) {
        std::fprintf(stderr, "zero-span point times were accepted\n");
        return 8;
    }

    odometry.reset();
    if (odometry.frame_count() != 0 || !odometry.map_snapshot().empty()) {
        std::fprintf(stderr, "reset did not clear streaming state\n");
        return 9;
    }

    // Dense host storage must preserve the reference GPU map ordering and thus
    // block-reduced poses, correspondences and adaptive thresholds bit-for-bit.
    KissIcpConfig reference_config=config;
    reference_config.map_backend=KissIcpMapBackend::Unordered;
    reference_config.normal_query_cell_order=false;
    reference_config.downsample_backend=KissIcpDownsampleBackend::Cached;
    reference_config.normal_update=KissIcpNormalUpdate::Full;
    KissIcpConfig dense_config=config;
    dense_config.map_backend=KissIcpMapBackend::Dense;
    KissIcpConfig validation_config=config;
    validation_config.map_backend=KissIcpMapBackend::Validate;
    KissIcpOdometry dense_ordered(validation_config), dense(dense_config), unordered(reference_config);
    for(int frame=0;frame<20;++frame) {
        if(frame==10) { dense_ordered.reset(); dense.reset(); unordered.reset(); }
        std::vector<float> sequence_scan=scan;
        for(std::size_t i=0;i<sequence_scan.size();i+=3) {
            sequence_scan[i]-=.025f*frame;
            sequence_scan[i+1]-=.01f*std::sin(.2f*frame);
        }
        const auto dense_result=dense_ordered.register_scan(sequence_scan,point_times);
        const auto old_dense_result=dense.register_scan(sequence_scan,point_times);
        const auto reference_result=unordered.register_scan(sequence_scan,point_times);
        if(std::memcmp(&dense_result.pose,&reference_result.pose,sizeof(KissIcpPose)) ||
           dense_result.map_points!=reference_result.map_points ||
           dense_result.alignment.inliers!=reference_result.alignment.inliers ||
           dense_result.alignment.rmse!=reference_result.alignment.rmse ||
           dense_result.alignment.threshold!=reference_result.alignment.threshold) {
            std::fprintf(stderr,"dense/reference GPU point order differs at frame %d\n",frame);
            return 10;
        }
        if(std::memcmp(&dense_result.pose,&old_dense_result.pose,sizeof(KissIcpPose)) ||
           dense_result.map_points!=old_dense_result.map_points ||
           dense_result.alignment.inliers!=old_dense_result.alignment.inliers ||
           dense_result.alignment.rmse!=old_dense_result.alignment.rmse ||
           dense_result.alignment.threshold!=old_dense_result.alignment.threshold ||
           dense_ordered.timing().normal_reused_points!=dense.timing().normal_reused_points ||
           dense_ordered.timing().normal_recomputed_points!=dense.timing().normal_recomputed_points ||
           dense_ordered.timing().host_map_pool_bytes==0 || dense.timing().host_map_pool_bytes!=0) {
            std::fprintf(stderr,"pooled/dense streaming state differs at frame %d\n",frame); return 11;
        }
    }
    std::printf("KISS-ICP reusable GPU streaming API: PASS\n");
    return 0;
}
