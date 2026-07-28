#include "cuda_voxel_mapping/cuda_voxel_mapper_node.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node =
    std::make_shared<cuda_voxel_mapping::CudaVoxelMapperNode>(rclcpp::NodeOptions{});
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node->get_node_base_interface());
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
