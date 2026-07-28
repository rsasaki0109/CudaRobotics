#include "cuda_kiss_icp/pointcloud_decoder.hpp"

#include <sensor_msgs/msg/point_field.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace cuda_kiss_icp {
namespace {

bool host_is_big_endian()
{
  const std::uint16_t value = 0x0102;
  return *reinterpret_cast<const std::uint8_t *>(&value) == 0x01;
}

const sensor_msgs::msg::PointField & find_field(
  const sensor_msgs::msg::PointCloud2 & message, const std::string & name)
{
  const auto it = std::find_if(
    message.fields.begin(), message.fields.end(),
    [&name](const auto & field) {return field.name == name;});
  if (it == message.fields.end()) {
    throw std::invalid_argument("PointCloud2 is missing field '" + name + "'");
  }
  if (it->count != 1) {
    throw std::invalid_argument("PointCloud2 field '" + name + "' must have count 1");
  }
  if (it->datatype != sensor_msgs::msg::PointField::FLOAT32 &&
    it->datatype != sensor_msgs::msg::PointField::FLOAT64)
  {
    throw std::invalid_argument(
            "PointCloud2 field '" + name + "' must be FLOAT32 or FLOAT64");
  }
  const std::size_t width =
    it->datatype == sensor_msgs::msg::PointField::FLOAT32 ? sizeof(float) : sizeof(double);
  if (static_cast<std::size_t>(it->offset) + width > message.point_step) {
    throw std::invalid_argument("PointCloud2 field '" + name + "' exceeds point_step");
  }
  return *it;
}

template<typename T>
T read_scalar(const std::uint8_t * source, bool swap)
{
  std::array<std::uint8_t, sizeof(T)> bytes;
  std::memcpy(bytes.data(), source, sizeof(T));
  if (swap) {
    std::reverse(bytes.begin(), bytes.end());
  }
  T value;
  std::memcpy(&value, bytes.data(), sizeof(T));
  return value;
}

float read_coordinate(
  const std::uint8_t * point, const sensor_msgs::msg::PointField & field, bool swap)
{
  if (field.datatype == sensor_msgs::msg::PointField::FLOAT32) {
    return read_scalar<float>(point + field.offset, swap);
  }
  return static_cast<float>(read_scalar<double>(point + field.offset, swap));
}

}  // namespace

DecodedPointCloud decode_xyz(const sensor_msgs::msg::PointCloud2 & message)
{
  if (message.height == 0 || message.width == 0) {
    throw std::invalid_argument("PointCloud2 dimensions must be non-zero");
  }
  if (message.point_step == 0) {
    throw std::invalid_argument("PointCloud2 point_step must be non-zero");
  }
  const std::size_t minimum_row =
    static_cast<std::size_t>(message.width) * message.point_step;
  if (message.row_step < minimum_row) {
    throw std::invalid_argument("PointCloud2 row_step is smaller than width * point_step");
  }
  const std::size_t preceding_rows = static_cast<std::size_t>(message.height) - 1;
  if (preceding_rows != 0 &&
    message.row_step > (std::numeric_limits<std::size_t>::max() - minimum_row) / preceding_rows)
  {
    throw std::invalid_argument("PointCloud2 layout size overflows size_t");
  }
  const std::size_t required = preceding_rows * message.row_step + minimum_row;
  if (message.data.size() < required) {
    throw std::invalid_argument("PointCloud2 data is shorter than its declared layout");
  }

  const auto & x_field = find_field(message, "x");
  const auto & y_field = find_field(message, "y");
  const auto & z_field = find_field(message, "z");
  const bool swap = message.is_bigendian != host_is_big_endian();

  DecodedPointCloud output;
  const std::size_t point_count = static_cast<std::size_t>(message.width) * message.height;
  if (point_count > std::numeric_limits<std::size_t>::max() / 3) {
    throw std::invalid_argument("PointCloud2 point count overflows XYZ storage");
  }
  output.xyz.reserve(point_count * 3);
  for (std::size_t row = 0; row < message.height; ++row) {
    const std::uint8_t * row_data = message.data.data() + row * message.row_step;
    for (std::size_t column = 0; column < message.width; ++column) {
      const std::uint8_t * point = row_data + column * message.point_step;
      const float x = read_coordinate(point, x_field, swap);
      const float y = read_coordinate(point, y_field, swap);
      const float z = read_coordinate(point, z_field, swap);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        ++output.skipped_non_finite;
        continue;
      }
      output.xyz.push_back(x);
      output.xyz.push_back(y);
      output.xyz.push_back(z);
    }
  }
  return output;
}

}  // namespace cuda_kiss_icp
