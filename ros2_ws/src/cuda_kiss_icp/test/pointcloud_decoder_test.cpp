#include "cuda_kiss_icp/pointcloud_decoder.hpp"

#include <gtest/gtest.h>
#include <sensor_msgs/msg/point_field.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

sensor_msgs::msg::PointField field(
  const std::string & name, std::uint32_t offset, std::uint8_t datatype)
{
  sensor_msgs::msg::PointField output;
  output.name = name;
  output.offset = offset;
  output.datatype = datatype;
  output.count = 1;
  return output;
}

template<typename T>
void write_scalar(std::vector<std::uint8_t> & data, std::size_t offset, T value, bool big_endian)
{
  std::array<std::uint8_t, sizeof(T)> bytes;
  std::memcpy(bytes.data(), &value, sizeof(T));
  const std::uint16_t marker = 0x0102;
  const bool host_big_endian = *reinterpret_cast<const std::uint8_t *>(&marker) == 0x01;
  if (big_endian != host_big_endian) {
    std::reverse(bytes.begin(), bytes.end());
  }
  std::copy(bytes.begin(), bytes.end(), data.begin() + offset);
}

}  // namespace

TEST(PointCloudDecoder, HandlesNamedFieldsOrganizedPaddingAndNonFinitePoints)
{
  sensor_msgs::msg::PointCloud2 message;
  message.height = 2;
  message.width = 2;
  message.fields = {
    field("z", 0, sensor_msgs::msg::PointField::FLOAT32),
    field("intensity", 4, sensor_msgs::msg::PointField::UINT16),
    field("x", 8, sensor_msgs::msg::PointField::FLOAT32),
    field("y", 12, sensor_msgs::msg::PointField::FLOAT32)};
  message.point_step = 16;
  message.row_step = 40;
  message.data.resize(80, 0xa5);

  const std::array<std::array<float, 3>, 4> points = {{
    {{1.0f, 2.0f, 3.0f}},
    {{4.0f, 5.0f, 6.0f}},
    {{7.0f, 8.0f, 9.0f}},
    {{std::numeric_limits<float>::quiet_NaN(), 10.0f, 11.0f}}}};
  for (std::size_t index = 0; index < points.size(); ++index) {
    const std::size_t row = index / 2;
    const std::size_t column = index % 2;
    const std::size_t base = row * message.row_step + column * message.point_step;
    write_scalar(message.data, base + 8, points[index][0], false);
    write_scalar(message.data, base + 12, points[index][1], false);
    write_scalar(message.data, base, points[index][2], false);
  }

  const auto decoded = cuda_kiss_icp::decode_xyz(message);
  EXPECT_EQ(decoded.skipped_non_finite, 1u);
  ASSERT_EQ(decoded.xyz.size(), 9u);
  EXPECT_FLOAT_EQ(decoded.xyz[0], 1.0f);
  EXPECT_FLOAT_EQ(decoded.xyz[4], 5.0f);
  EXPECT_FLOAT_EQ(decoded.xyz[8], 9.0f);
}

TEST(PointCloudDecoder, HandlesBigEndianFloat64)
{
  sensor_msgs::msg::PointCloud2 message;
  message.height = 1;
  message.width = 1;
  message.fields = {
    field("x", 0, sensor_msgs::msg::PointField::FLOAT64),
    field("y", 8, sensor_msgs::msg::PointField::FLOAT64),
    field("z", 16, sensor_msgs::msg::PointField::FLOAT64)};
  message.is_bigendian = true;
  message.point_step = 24;
  message.row_step = 24;
  message.data.resize(24);
  write_scalar(message.data, 0, 1.25, true);
  write_scalar(message.data, 8, -2.5, true);
  write_scalar(message.data, 16, 3.75, true);

  const auto decoded = cuda_kiss_icp::decode_xyz(message);
  ASSERT_EQ(decoded.xyz.size(), 3u);
  EXPECT_FLOAT_EQ(decoded.xyz[0], 1.25f);
  EXPECT_FLOAT_EQ(decoded.xyz[1], -2.5f);
  EXPECT_FLOAT_EQ(decoded.xyz[2], 3.75f);
}

TEST(PointCloudDecoder, RejectsInvalidSchemas)
{
  sensor_msgs::msg::PointCloud2 message;
  message.height = 1;
  message.width = 1;
  message.fields = {
    field("x", 0, sensor_msgs::msg::PointField::FLOAT32),
    field("y", 4, sensor_msgs::msg::PointField::FLOAT32)};
  message.point_step = 8;
  message.row_step = 8;
  message.data.resize(8);
  EXPECT_THROW(cuda_kiss_icp::decode_xyz(message), std::invalid_argument);

  message.fields.push_back(field("z", 8, sensor_msgs::msg::PointField::FLOAT32));
  EXPECT_THROW(cuda_kiss_icp::decode_xyz(message), std::invalid_argument);
}
