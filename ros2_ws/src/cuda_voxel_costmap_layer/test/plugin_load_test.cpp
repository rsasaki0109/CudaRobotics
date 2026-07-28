#include <nav2_costmap_2d/layer.hpp>
#include <pluginlib/class_loader.hpp>

#include <cstdio>
#include <exception>

int main()
{
  try {
    pluginlib::ClassLoader<nav2_costmap_2d::Layer> loader(
      "nav2_costmap_2d", "nav2_costmap_2d::Layer");
    auto layer = loader.createSharedInstance(
      "cuda_voxel_costmap_layer::CudaVoxelCostmapLayer");
    if (!layer) {
      std::printf("FAIL: createSharedInstance returned null\n");
      return 1;
    }
    std::printf("PASS: CudaVoxelCostmapLayer loaded via pluginlib\n");
    return 0;
  } catch (const std::exception & exception) {
    std::printf("FAIL: %s\n", exception.what());
    return 1;
  }
}
