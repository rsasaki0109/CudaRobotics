#ifndef CUDA_MPPI_CONTROLLER__NAV2_COMPAT_HPP_
#define CUDA_MPPI_CONTROLLER__NAV2_COMPAT_HPP_

// Small compile-time shims so the plugin builds on Humble (pre-Iron exceptions)
// and newer Nav2 distros without forking the controller implementation.

#if defined(CUDAMPPI_NAV2_HUMBLE)
#include "nav2_core/exceptions.hpp"
namespace cuda_mppi_controller
{
using ControllerInvalidPath = nav2_core::PlannerException;
}
#else
#include "nav2_core/controller_exceptions.hpp"
namespace cuda_mppi_controller
{
using ControllerInvalidPath = nav2_core::InvalidPath;
}
#endif

#endif  // CUDA_MPPI_CONTROLLER__NAV2_COMPAT_HPP_
