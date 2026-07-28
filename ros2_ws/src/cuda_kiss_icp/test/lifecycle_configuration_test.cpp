#include "cuda_kiss_icp/cuda_kiss_icp_node.hpp"

#include <gtest/gtest.h>
#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>

#include <memory>

class RclcppContext : public ::testing::Environment
{
public:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
  }

  void TearDown() override
  {
    rclcpp::shutdown();
  }
};

testing::Environment * const context =
  testing::AddGlobalTestEnvironment(new RclcppContext);

TEST(CudaKissIcpLifecycle, ConfiguresWithoutAllocatingGpuRuntime)
{
  auto node = std::make_shared<cuda_kiss_icp::CudaKissIcpNode>(rclcpp::NodeOptions{});
  const auto & configured = node->configure();
  EXPECT_EQ(configured.id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  const auto & cleaned = node->cleanup();
  EXPECT_EQ(cleaned.id(), lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}

TEST(CudaKissIcpLifecycle, RejectsAbsoluteDataTopics)
{
  rclcpp::NodeOptions options;
  options.append_parameter_override("input_topic", "/points");
  auto node = std::make_shared<cuda_kiss_icp::CudaKissIcpNode>(options);
  const auto & state = node->configure();
  EXPECT_EQ(state.id(), lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}
