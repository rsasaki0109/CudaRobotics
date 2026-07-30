#include "cuda_voxel_mapping/cuda_voxel_mapper_node.hpp"

#include <gtest/gtest.h>
#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>

#include <memory>

class RclcppContext : public ::testing::Environment
{
public:
  void SetUp() override {rclcpp::init(0, nullptr);}
  void TearDown() override {rclcpp::shutdown();}
};

testing::Environment * const context =
  testing::AddGlobalTestEnvironment(new RclcppContext);

TEST(CudaVoxelMappingLifecycle, ConfiguresWithoutAllocatingGpuRuntime)
{
  auto node =
    std::make_shared<cuda_voxel_mapping::CudaVoxelMapperNode>(rclcpp::NodeOptions{});
  EXPECT_EQ(
    node->configure().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  EXPECT_EQ(
    node->cleanup().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}

TEST(CudaVoxelMappingLifecycle, RejectsNonStandardAbsoluteTopic)
{
  rclcpp::NodeOptions options;
  options.append_parameter_override("occupancy_topic", "/occupancy");
  auto node = std::make_shared<cuda_voxel_mapping::CudaVoxelMapperNode>(options);
  EXPECT_EQ(
    node->configure().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}

TEST(CudaVoxelMappingOccupancy, PublishesPlanarIdentityOrigin)
{
  cudarobotics::OccupancyProjection projection;
  projection.grid.width = 2;
  projection.grid.height = 1;
  projection.grid.depth = 4;
  projection.grid.resolution = 0.25F;
  projection.grid.origin_x = -3.0F;
  projection.grid.origin_y = 2.0F;
  projection.grid.origin_z = -1.5F;
  projection.data = {-1, 100};

  const auto message = cuda_voxel_mapping::make_occupancy_message(
    projection, rclcpp::Time(10, 20), rclcpp::Time(9, 0), "odom");

  EXPECT_EQ(message.header.frame_id, "odom");
  EXPECT_DOUBLE_EQ(message.info.origin.position.x, -3.0);
  EXPECT_DOUBLE_EQ(message.info.origin.position.y, 2.0);
  EXPECT_DOUBLE_EQ(message.info.origin.position.z, 0.0);
  EXPECT_DOUBLE_EQ(message.info.origin.orientation.x, 0.0);
  EXPECT_DOUBLE_EQ(message.info.origin.orientation.y, 0.0);
  EXPECT_DOUBLE_EQ(message.info.origin.orientation.z, 0.0);
  EXPECT_DOUBLE_EQ(message.info.origin.orientation.w, 1.0);
  EXPECT_EQ(message.data, projection.data);
}
