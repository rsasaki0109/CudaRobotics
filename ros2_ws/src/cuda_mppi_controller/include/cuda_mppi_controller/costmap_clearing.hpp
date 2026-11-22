// Clear tracked moving obstacles from a copy of the Nav2 costmap (no ROS
// runtime needed).
//
// The plugin hands tracked obstacles to the MPPI core as predictions. The
// costmap usually shows the same people as lethal cells, at a stale position,
// plus the trail an obstacle layer has not cleared yet. Both make every
// rollout through a person's recent path collide. TrackedObstacleClearing
// zeroes discs around the obstacles' current and recent positions in a copy
// of the costmap, while sparing structure: cells that have been lethal for
// many consecutive cycles, and the band within robot_radius of them, so a
// person next to a wall cannot erase the wall or its inflation.
#ifndef CUDA_MPPI_CONTROLLER__COSTMAP_CLEARING_HPP_
#define CUDA_MPPI_CONTROLLER__COSTMAP_CLEARING_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <utility>
#include <vector>

namespace cuda_mppi_controller
{

struct ClearingDisc
{
  double x = 0.0;
  double y = 0.0;
  double radius = 0.0;
};

class TrackedObstacleClearing
{
public:
  static constexpr unsigned char kLethal = 254;   // nav2 LETHAL_OBSTACLE
  static constexpr unsigned char kFree = 0;       // nav2 FREE_SPACE
  static constexpr unsigned char kUnknown = 255;  // nav2 NO_INFORMATION

  // structure_cycles: consecutive lethal cycles after which a cell is
  //   structure; history_s: how long past discs keep being cleared;
  // structure_band: cells within this distance [m] of structure are spared.
  TrackedObstacleClearing(int structure_cycles, double history_s, double structure_band)
  : structure_cycles_(std::max(1, structure_cycles)),
    history_s_(std::max(0.0, history_s)),
    structure_band_(std::max(0.0, structure_band)) {}

  // Age the cells from this cycle's costmap. Call once per control cycle,
  // with or without obstacles, so structure is known when a person appears.
  // A rolling costmap that moved by whole cells keeps its ages; any other
  // change of geometry starts over.
  void update(
    const unsigned char * costmap, int size_x, int size_y,
    double origin_x, double origin_y, double resolution)
  {
    if (costmap == nullptr || size_x <= 0 || size_y <= 0 || !(resolution > 0.0)) {
      age_.clear();
      size_x_ = size_y_ = 0;
      return;
    }
    shiftOrReset(size_x, size_y, origin_x, origin_y, resolution);
    const std::size_t cells = static_cast<std::size_t>(size_x) * size_y;
    for (std::size_t i = 0; i < cells; ++i) {
      if (costmap[i] == kLethal) {
        if (age_[i] < std::numeric_limits<std::uint16_t>::max()) {
          ++age_[i];
        }
      } else {
        age_[i] = 0;
      }
    }
  }

  // Remember this cycle's discs (costmap frame) at time now_s, and forget
  // discs older than history_s.
  void remember(const std::vector<ClearingDisc> & discs, double now_s)
  {
    while (!history_.empty() &&
      (history_.front().first < now_s - history_s_ || history_.front().first > now_s))
    {
      history_.pop_front();  // too old, or the clock went backwards
    }
    if (!discs.empty()) {
      history_.emplace_back(now_s, discs);
    }
  }

  bool hasDiscs() const {return !history_.empty();}

  // Copy costmap (the one passed to the last update) into out with every
  // remembered disc cleared, each shrunk to stop short of structure.
  // Returns the number of cells changed.
  std::size_t apply(const unsigned char * costmap, std::vector<unsigned char> & out) const
  {
    const std::size_t cells = static_cast<std::size_t>(size_x_) * size_y_;
    out.assign(costmap, costmap + cells);
    std::size_t changed = 0;
    for (const auto & entry : history_) {
      for (const ClearingDisc & disc : entry.second) {
        changed += clearDisc(disc, out);
      }
    }
    return changed;
  }

  int structureCycles() const {return structure_cycles_;}

private:
  void shiftOrReset(int size_x, int size_y, double origin_x, double origin_y, double resolution)
  {
    const std::size_t cells = static_cast<std::size_t>(size_x) * size_y;
    const bool same_grid = size_x == size_x_ && size_y == size_y_ &&
      std::fabs(resolution - resolution_) < 1.0e-9 && age_.size() == cells;
    const double fx = (origin_x - origin_x_) / resolution;
    const double fy = (origin_y - origin_y_) / resolution;
    const long dx = std::lround(fx);
    const long dy = std::lround(fy);
    const bool whole_cells = std::fabs(fx - dx) < 1.0e-3 && std::fabs(fy - dy) < 1.0e-3;
    if (same_grid && whole_cells && (dx != 0 || dy != 0)) {
      std::vector<std::uint16_t> shifted(cells, 0);
      for (int y = 0; y < size_y; ++y) {
        const long old_y = y + dy;
        if (old_y < 0 || old_y >= size_y) {continue;}
        for (int x = 0; x < size_x; ++x) {
          const long old_x = x + dx;
          if (old_x < 0 || old_x >= size_x) {continue;}
          shifted[static_cast<std::size_t>(y) * size_x + x] =
            age_[static_cast<std::size_t>(old_y) * size_x + old_x];
        }
      }
      age_.swap(shifted);
    } else if (!same_grid || !whole_cells) {
      age_.assign(cells, 0);
    }
    size_x_ = size_x;
    size_y_ = size_y;
    origin_x_ = origin_x;
    origin_y_ = origin_y;
    resolution_ = resolution;
  }

  std::size_t clearDisc(const ClearingDisc & disc, std::vector<unsigned char> & out) const
  {
    if (!(disc.radius > 0.0) || !std::isfinite(disc.x) || !std::isfinite(disc.y)) {
      return 0;
    }
    // Shrink the disc so that it stays structure_band plus one cell away from
    // the nearest structure cell centre (searched within the disc + band).
    const double reach = disc.radius + structure_band_ + resolution_;
    int x0, x1, y0, y1;
    if (!window(disc.x, disc.y, reach, x0, x1, y0, y1)) {
      return 0;
    }
    double radius = disc.radius;
    for (int y = y0; y < y1; ++y) {
      for (int x = x0; x < x1; ++x) {
        if (age_[static_cast<std::size_t>(y) * size_x_ + x] < structure_cycles_) {continue;}
        const double d = std::hypot(cellX(x) - disc.x, cellY(y) - disc.y);
        radius = std::min(radius, std::max(0.0, d - structure_band_ - resolution_));
      }
    }
    if (!window(disc.x, disc.y, radius, x0, x1, y0, y1)) {
      return 0;
    }
    std::size_t changed = 0;
    const double r2 = radius * radius;
    for (int y = y0; y < y1; ++y) {
      for (int x = x0; x < x1; ++x) {
        const double ex = cellX(x) - disc.x;
        const double ey = cellY(y) - disc.y;
        unsigned char & cell = out[static_cast<std::size_t>(y) * size_x_ + x];
        if (ex * ex + ey * ey <= r2 && cell != kFree && cell != kUnknown) {
          cell = kFree;
          ++changed;
        }
      }
    }
    return changed;
  }

  bool window(double cx, double cy, double r, int & x0, int & x1, int & y0, int & y1) const
  {
    if (size_x_ <= 0 || size_y_ <= 0 || !(r > 0.0)) {
      return false;
    }
    x0 = std::max(0, static_cast<int>(std::floor((cx - r - origin_x_) / resolution_)));
    x1 = std::min(size_x_, static_cast<int>(std::floor((cx + r - origin_x_) / resolution_)) + 1);
    y0 = std::max(0, static_cast<int>(std::floor((cy - r - origin_y_) / resolution_)));
    y1 = std::min(size_y_, static_cast<int>(std::floor((cy + r - origin_y_) / resolution_)) + 1);
    return x0 < x1 && y0 < y1;
  }

  double cellX(int x) const {return origin_x_ + (x + 0.5) * resolution_;}
  double cellY(int y) const {return origin_y_ + (y + 0.5) * resolution_;}

  int structure_cycles_;
  double history_s_;
  double structure_band_;
  std::vector<std::uint16_t> age_;
  int size_x_ = 0;
  int size_y_ = 0;
  double origin_x_ = 0.0;
  double origin_y_ = 0.0;
  double resolution_ = 0.0;
  std::deque<std::pair<double, std::vector<ClearingDisc>>> history_;
};

}  // namespace cuda_mppi_controller

#endif  // CUDA_MPPI_CONTROLLER__COSTMAP_CLEARING_HPP_
