// Smoke test of cudarobotics::LidarObjectPipeline through its public interface
// only: flat ground plus one car seen from a sensor that drives past it.
// Checks the ground labels, that the car comes out as one cluster classed as a
// car with a car-sized completed box, its heading (within 3 deg in most scans),
// and that one track follows it through every scan.

#include "cudarobotics/lidar_objects_gpu.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr float kSensorH = 1.8f;
constexpr float kCarX = 12.0f, kCarY = -4.0f, kCarYaw = 0.35f;   // world frame
constexpr float kCarL = 4.5f, kCarW = 1.8f, kCarH = 1.5f;

// Points of one scan from a sensor at (sx, sy): ground rings, and the car's
// faces that look towards the sensor, every 5 cm.
void make_scan(float sx, float sy, std::vector<float>& xyz, std::vector<int>& is_car) {
    xyz.clear(); is_car.clear();
    for (float r = 4.0f; r < 40.0f; r *= 1.04f)
        for (int a = 0; a < 720; ++a) {
            float th = a * 3.14159265f / 360.0f;
            float x = r * std::cos(th), y = r * std::sin(th);
            float wx = x + sx - kCarX, wy = y + sy - kCarY;
            float u = std::cos(kCarYaw) * wx + std::sin(kCarYaw) * wy, v = -std::sin(kCarYaw) * wx + std::cos(kCarYaw) * wy;
            if (std::fabs(u) < 0.5f * kCarL + 0.3f && std::fabs(v) < 0.5f * kCarW + 0.3f) continue;   // under the car
            xyz.insert(xyz.end(), { x, y, -kSensorH });
            is_car.push_back(0);
        }
    float c = std::cos(kCarYaw), s = std::sin(kCarYaw);
    for (int face = 0; face < 4; ++face) {
        // outward normal of the face in the car frame, and whether it faces the sensor
        float nu = face == 0 ? 1.0f : face == 1 ? -1.0f : 0.0f, nv = face == 2 ? 1.0f : face == 3 ? -1.0f : 0.0f;
        float fu = nu * 0.5f * kCarL, fv = nv * 0.5f * kCarW;
        float cxw = kCarX + c * fu - s * fv - sx, cyw = kCarY + s * fu + c * fv - sy;   // face centre, sensor frame
        float nx = c * nu - s * nv, ny = s * nu + c * nv;
        if (cxw * nx + cyw * ny >= 0.0f) continue;   // facing away
        float half = nu != 0.0f ? 0.5f * kCarW : 0.5f * kCarL;
        for (float e = -half; e <= half; e += 0.05f)
            for (float z = 0.1f; z <= kCarH; z += 0.05f) {
                float eu = nu != 0.0f ? 0.0f : e, ev = nu != 0.0f ? e : 0.0f;
                float x = cxw + c * eu - s * ev, y = cyw + s * eu + c * ev;
                xyz.insert(xyz.end(), { x, y, z - kSensorH });
                is_car.push_back(1);
            }
    }
}

}  // namespace

int main() {
    cudarobotics::LidarObjectPipeline pipe;
    bool ok = true;
    int tracked_scans = 0, track_id = -1, heading_ok = 0;
    for (int k = 0; k < 6; ++k) {
        float sx = 4.0f + 1.0f * k, sy = 0.0f;
        std::vector<float> xyz;
        std::vector<int> is_car;
        make_scan(sx, sy, xyz, is_car);
        cudarobotics::LidarObjectsResult R = pipe.process(xyz.data(), is_car.size(), sx, sy, kSensorH, 0.1 * k);
        long ground_ok = 0, ground_n = 0, car_ground = 0;
        int car_label = -1;
        for (size_t i = 0; i < is_car.size(); ++i) {
            if (!is_car[i]) { ++ground_n; ground_ok += R.ground[i]; }
            else { car_ground += R.ground[i]; if (car_label < 0) car_label = R.cluster[i]; }
        }
        const cudarobotics::LidarCluster* car = nullptr;
        for (const auto& C : R.clusters) if (C.label == car_label) car = &C;
        double gfrac = (double)ground_ok / ground_n;
        if (gfrac < 0.99 || !car) { ok = false; std::printf("scan %d: ground %.3f, car cluster %s\n", k, gfrac, car ? "yes" : "no"); continue; }
        float d = std::fmod(car->box.yaw - kCarYaw + 10.0f * 3.14159265f, 0.5f * 3.14159265f);
        float yaw_err = std::fmin(d, 0.5f * 3.14159265f - d) * 180.0f / 3.14159265f;
        float len = std::fmax(car->completed.length, car->completed.width);
        std::printf("scan %d: ground %.4f, car points labelled ground %ld, heading error %.2f deg, class %d, "
                    "completed length %.2f m, tracks %zu\n", k, gfrac, car_ground, yaw_err, (int)car->cls, len,
                    R.tracks.size());
        heading_ok += yaw_err <= 3.0f;   // a single view may mislead the L-shape search
        if (car->cls != cudarobotics::LidarObjectClass::Car || std::fabs(len - kCarL) > 0.6f) ok = false;
        for (const auto& T : R.tracks) {
            if (track_id < 0) track_id = T.id;
            if (T.id == track_id) ++tracked_scans;
        }
    }
    if (tracked_scans < 6 || heading_ok < 4) ok = false;
    std::printf("heading within 3 deg in %d of 6 scans, tracked in %d of 6; %s\n", heading_ok, tracked_scans,
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
