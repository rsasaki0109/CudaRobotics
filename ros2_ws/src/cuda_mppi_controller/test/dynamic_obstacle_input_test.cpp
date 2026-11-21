// Tracked-object conversion for the dynamic-obstacle input (no ROS runtime or
// GPU needed): frame transform, velocity rotation, extrapolation by the
// message age, input validation and nearest-first truncation.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "cuda_mppi_controller/dynamic_obstacle_input.hpp"

namespace
{

using cuda_mppi_controller::PlanarTransform;
using cuda_mppi_controller::TrackedDisc;
using cuda_mppi_controller::toDynamicObstacles;

void check(bool condition, const char * message)
{
  if (!condition) {
    std::fprintf(stderr, "FAILED: %s\n", message);
    std::exit(1);
  }
}

bool near(double a, double b) {return std::fabs(a - b) < 1.0e-5;}

}  // namespace

int main()
{
  // A disc 1 m ahead in the tracker frame, moving +x at 0.5 m/s. The tracker
  // frame sits at (2, 3) in the costmap frame, rotated by +90 degrees.
  const std::vector<TrackedDisc> one{{1.0, 0.0, 0.5, 0.0, 0.3}};
  const PlanarTransform transform{2.0, 3.0, M_PI / 2.0};
  auto out = toDynamicObstacles(one, transform, 0.0, 0.0, 0.0);
  check(out.size() == 1, "one valid disc");
  check(near(out[0].x, 2.0) && near(out[0].y, 4.0), "position is transformed");
  check(near(out[0].vx, 0.0) && near(out[0].vy, 0.5), "velocity is rotated");
  check(near(out[0].radius, 0.3), "radius is kept");

  // Extrapolated by the message age along the rotated velocity; a negative or
  // non-finite age is treated as 0.
  out = toDynamicObstacles(one, transform, 0.4, 0.0, 0.0);
  check(near(out[0].x, 2.0) && near(out[0].y, 4.2), "extrapolated by the message age");
  out = toDynamicObstacles(one, transform, -1.0, 0.0, 0.0);
  check(near(out[0].y, 4.0), "a negative age does not move the disc back");
  out = toDynamicObstacles(one, transform, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0);
  check(near(out[0].y, 4.0), "a non-finite age is ignored");

  // Invalid entries are dropped.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<TrackedDisc> mixed{
    {nan, 0.0, 0.0, 0.0, 0.3}, {0.0, 0.0, 0.0, 0.0, -0.1}, {0.0, 0.0, nan, 0.0, 0.3},
    {5.0, 0.0, 0.0, 0.0, 0.3}};
  out = toDynamicObstacles(mixed, PlanarTransform{}, 0.0, 0.0, 0.0);
  check(out.size() == 1 && near(out[0].x, 5.0), "invalid discs are dropped");

  // More than max_count: the nearest (by surface distance) are kept.
  std::vector<TrackedDisc> many;
  for (int i = 0; i < 10; ++i) {
    many.push_back({static_cast<double>(10 - i), 0.0, 0.0, 0.0, 0.25});
  }
  out = toDynamicObstacles(many, PlanarTransform{}, 0.0, 0.0, 0.0, 3);
  check(out.size() == 3, "truncated to max_count");
  for (const auto & o : out) {
    check(o.x <= 3.0f + 1.0e-6f, "the nearest discs are kept");
  }
  out = toDynamicObstacles(many, PlanarTransform{}, 0.0, 0.0, 0.0);
  check(out.size() == 10, "below the core limit nothing is dropped");
  check(toDynamicObstacles({}, PlanarTransform{}, 0.0, 0.0, 0.0).empty(), "empty input");

  std::printf("dynamic_obstacle_input_test: OK\n");
  return 0;
}
