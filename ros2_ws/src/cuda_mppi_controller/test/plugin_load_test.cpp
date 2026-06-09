// Verifies the plugin is discoverable and instantiable via pluginlib,
// exactly as nav2's controller_server would load it.
#include <cstdio>

#include "nav2_core/controller.hpp"
#include "pluginlib/class_loader.hpp"

int main()
{
  try {
    pluginlib::ClassLoader<nav2_core::Controller> loader(
      "nav2_core", "nav2_core::Controller");
    auto controller = loader.createSharedInstance(
      "cuda_mppi_controller::CudaMppiController");
    if (!controller) {
      std::printf("FAIL: createSharedInstance returned null\n");
      return 1;
    }
    std::printf("PASS: cuda_mppi_controller::CudaMppiController loaded via pluginlib\n");
    return 0;
  } catch (const std::exception & e) {
    std::printf("FAIL: %s\n", e.what());
    return 1;
  }
}
