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
