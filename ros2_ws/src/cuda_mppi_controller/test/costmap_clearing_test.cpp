// Clearing tracked obstacles from the costmap copy (no ROS runtime or GPU
// needed): discs are cleared, structure and its band are spared, history
// expires, and cell ages survive a rolling costmap's whole-cell moves.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_mppi_controller/costmap_clearing.hpp"

namespace
{

using cuda_mppi_controller::ClearingDisc;
using cuda_mppi_controller::TrackedObstacleClearing;

constexpr int kSize = 40;           // 40 x 40 cells
constexpr double kResolution = 0.05;  // 2 m x 2 m

void check(bool condition, const char * message)
{
  if (!condition) {
    std::fprintf(stderr, "FAILED: %s\n", message);
    std::exit(1);
  }
}

unsigned char & at(std::vector<unsigned char> & map, int x, int y) {return map[y * kSize + x];}

// Lethal cells of a disc of radius r (m) centred at (cx, cy) in world metres.
void stamp(std::vector<unsigned char> & map, double ox, double oy, double cx, double cy, double r)
{
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      const double dx = ox + (x + 0.5) * kResolution - cx;
      const double dy = oy + (y + 0.5) * kResolution - cy;
      if (dx * dx + dy * dy <= r * r) {at(map, x, y) = TrackedObstacleClearing::kLethal;}
    }
  }
}

}  // namespace

int main()
{
  // A wall along column 0 (x in [0, 0.05)), present from the start.
  std::vector<unsigned char> wall(kSize * kSize, 0);
  for (int y = 0; y < kSize; ++y) {at(wall, 0, y) = TrackedObstacleClearing::kLethal;}

  TrackedObstacleClearing clearing(5, 1.0, 0.25);
  for (int cycle = 0; cycle < 5; ++cycle) {
    clearing.update(wall.data(), kSize, kSize, 0.0, 0.0, kResolution);
  }

  // A person appears at (1.0, 1.0), far from the wall.
  std::vector<unsigned char> map = wall;
  stamp(map, 0.0, 0.0, 1.0, 1.0, 0.25);
  clearing.update(map.data(), kSize, kSize, 0.0, 0.0, kResolution);
  clearing.remember({{1.0, 1.0, 0.5}}, 10.0);
  std::vector<unsigned char> out;
  const std::size_t changed = clearing.apply(map.data(), out);
  check(changed > 0, "the person's cells are cleared");
  check(at(out, 20, 20) == 0, "the person's centre is free");
  for (int y = 0; y < kSize; ++y) {
    check(at(out, 0, y) == TrackedObstacleClearing::kLethal, "the wall stays");
  }

  // The same person next to the wall: the disc shrinks short of the wall's
  // band, so neither the wall nor cells within 0.25 m of it are cleared.
  map = wall;
  stamp(map, 0.0, 0.0, 0.3, 1.0, 0.1);
  TrackedObstacleClearing near_wall(5, 1.0, 0.25);
  for (int cycle = 0; cycle < 5; ++cycle) {
    near_wall.update(wall.data(), kSize, kSize, 0.0, 0.0, kResolution);
  }
  near_wall.update(map.data(), kSize, kSize, 0.0, 0.0, kResolution);
  near_wall.remember({{0.3, 1.0, 0.5}}, 10.0);
  near_wall.apply(map.data(), out);
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < 6; ++x) {  // cell centres within 0.275 m of x = 0
      check(at(out, x, y) == at(map, x, y), "structure and its band are spared");
    }
  }

  // A trail: positions from 0.5 s ago are still cleared, from 1.5 s ago not.
  TrackedObstacleClearing trail(5, 1.0, 0.25);
  map.assign(kSize * kSize, 0);
  stamp(map, 0.0, 0.0, 1.5, 1.5, 0.1);  // left 1.5 s ago
  stamp(map, 0.0, 0.0, 1.0, 1.5, 0.1);  // left 0.5 s ago
  trail.update(map.data(), kSize, kSize, 0.0, 0.0, kResolution);
  trail.remember({{1.5, 1.5, 0.2}}, 8.5);
  trail.remember({{1.0, 1.5, 0.2}}, 9.5);
  trail.remember({{0.5, 0.5, 0.2}}, 10.0);
  trail.apply(map.data(), out);
  check(at(out, 20, 30) == 0, "a recent position is cleared");
  check(at(out, 30, 30) == TrackedObstacleClearing::kLethal, "an expired position is kept");

  // A rolling costmap moving by whole cells keeps the ages: after a +0.25 m
  // (5 cell) shift in x, the wall (now at column -5, outside) is gone and a
  // second wall at world x = 1.5 seen for 5 cycles is still structure.
  TrackedObstacleClearing rolling(5, 1.0, 0.0);
  std::vector<unsigned char> world(kSize * kSize, 0);
  for (int y = 0; y < kSize; ++y) {at(world, 30, y) = TrackedObstacleClearing::kLethal;}
  for (int cycle = 0; cycle < 4; ++cycle) {
    rolling.update(world.data(), kSize, kSize, 0.0, 0.0, kResolution);
  }
  std::vector<unsigned char> moved(kSize * kSize, 0);
  for (int y = 0; y < kSize; ++y) {at(moved, 25, y) = TrackedObstacleClearing::kLethal;}
  rolling.update(moved.data(), kSize, kSize, 0.25, 0.0, kResolution);  // 5th cycle
  rolling.remember({{1.525, 1.0, 0.3}}, 1.0);  // centred on the wall
  rolling.apply(moved.data(), out);
  check(at(out, 25, 20) == TrackedObstacleClearing::kLethal,
        "ages follow the cells when the costmap rolls");
  // Without the shift-aware ages the wall would be cleared:
  TrackedObstacleClearing fresh(5, 1.0, 0.0);
  fresh.update(moved.data(), kSize, kSize, 0.25, 0.0, kResolution);
  fresh.remember({{1.525, 1.0, 0.3}}, 1.0);
  fresh.apply(moved.data(), out);
  check(at(out, 25, 20) == 0, "a young lethal cell is cleared");

  // Unknown cells stay unknown; empty history changes nothing.
  TrackedObstacleClearing unknown(5, 1.0, 0.0);
  std::vector<unsigned char> blind(kSize * kSize, TrackedObstacleClearing::kUnknown);
  unknown.update(blind.data(), kSize, kSize, 0.0, 0.0, kResolution);
  check(!unknown.hasDiscs(), "no discs remembered yet");
  check(unknown.apply(blind.data(), out) == 0, "nothing to clear");
  unknown.remember({{1.0, 1.0, 0.5}}, 1.0);
  check(unknown.apply(blind.data(), out) == 0 && out == blind, "unknown cells are not cleared");

  std::printf("costmap_clearing_test: OK\n");
  return 0;
}
