#include <Python.h>

#include <nanobind/nanobind.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_mppi_controller/mppi_gpu.hpp"
#include "cudarobotics/bcpd_gpu.hpp"
#include "cudarobotics/cudanav_costmap_gpu.hpp"
#include "cudarobotics/esdf_2d_gpu.hpp"
#include "cudarobotics/fgr_gpu.hpp"
#include "cudarobotics/filterreg_gpu.hpp"
#include "cudarobotics/kiss_icp_gpu.hpp"
#include "cudarobotics/robust_p2plane_gpu.hpp"
#include "cudarobotics/robust_treg_gpu.hpp"
#include "cudarobotics/sinkhorn_reg_gpu.hpp"
#include "cudarobotics/voxel_mapping_gpu.hpp"

#include <cuda_runtime.h>

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

namespace cr = cuda_mppi_controller;
namespace reg = cudarobotics;

enum DLDeviceType : int32_t
{
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLCUDAManaged = 13,
};

enum DLDataTypeCode : uint8_t
{
  kDLInt = 0U,
  kDLUInt = 1U,
  kDLFloat = 2U,
};

struct DLDevice
{
  DLDeviceType device_type;
  int32_t device_id;
};

struct DLDataType
{
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

struct DLTensor
{
  void * data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t * shape;
  int64_t * strides;
  uint64_t byte_offset;
};

struct DLManagedTensor
{
  DLTensor dl_tensor;
  void * manager_ctx;
  void (* deleter)(DLManagedTensor * self);
};

struct DLPackVersion
{
  uint32_t major;
  uint32_t minor;
};

struct DLManagedTensorVersioned
{
  DLPackVersion version;
  void * manager_ctx;
  void (* deleter)(DLManagedTensorVersioned * self);
  uint64_t flags;
  DLTensor dl_tensor;
};

struct PyObjectDeleter
{
  void operator()(PyObject * object) const noexcept
  {
    Py_XDECREF(object);
  }
};

class BufferView
{
public:
  BufferView(nb::handle object, int ndim, int second_dim, Py_ssize_t itemsize, const char * name)
  : name_(name)
  {
    if (!PyObject_CheckBuffer(object.ptr())) {
      throw std::invalid_argument(std::string(name_) + " must support the Python buffer protocol");
    }
    if (PyObject_GetBuffer(object.ptr(), &view_, PyBUF_FORMAT | PyBUF_STRIDES | PyBUF_ND) != 0) {
      throw nb::python_error();
    }
    acquired_ = true;

    if (view_.ndim != ndim) {
      fail("must have " + std::to_string(ndim) + " dimensions");
    }
    if (view_.itemsize != itemsize) {
      fail("must have itemsize " + std::to_string(itemsize));
    }
    if (second_dim >= 0 && view_.shape[1] != second_dim) {
      fail("must have shape (N, " + std::to_string(second_dim) + ")");
    }
    if (!PyBuffer_IsContiguous(&view_, 'C')) {
      fail("must be C-contiguous");
    }
  }

  ~BufferView()
  {
    if (acquired_) {
      PyBuffer_Release(&view_);
    }
  }

  BufferView(const BufferView &) = delete;
  BufferView & operator=(const BufferView &) = delete;

  template<typename T>
  const T * data() const
  {
    return static_cast<const T *>(view_.buf);
  }

  int dim(int axis) const
  {
    if (view_.shape[axis] > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::string(name_) + " dimension is too large");
    }
    return static_cast<int>(view_.shape[axis]);
  }

private:
  [[noreturn]] void fail(const std::string & reason)
  {
    if (acquired_) {
      PyBuffer_Release(&view_);
      acquired_ = false;
    }
    throw std::invalid_argument(std::string(name_) + " " + reason);
  }

  const char * name_;
  Py_buffer view_{};
  bool acquired_ = false;
};

class DLPackTensorView
{
public:
  DLPackTensorView(
    nb::handle object, int ndim, int second_dim,
    uint8_t dtype_code, uint8_t dtype_bits, uint16_t dtype_lanes,
    const char * name)
  : name_(name)
  {
    if (!supportsDLPack(object)) {
      throw std::invalid_argument(
              std::string(name_) +
              " must support the Python buffer protocol or CUDA DLPack");
    }

    std::unique_ptr<PyObject, PyObjectDeleter> capsule_guard(
      PyObject_CallMethod(object.ptr(), "__dlpack__", nullptr));
    if (!capsule_guard) {
      throw nb::python_error();
    }

    const char * used_name = nullptr;
    if (PyCapsule_IsValid(capsule_guard.get(), "dltensor")) {
      managed_ = static_cast<DLManagedTensor *>(
        PyCapsule_GetPointer(capsule_guard.get(), "dltensor"));
      if (managed_ == nullptr) {
        throw nb::python_error();
      }
      tensor_ = &managed_->dl_tensor;
      used_name = "used_dltensor";
    } else if (PyCapsule_IsValid(capsule_guard.get(), "dltensor_versioned")) {
      versioned_ = static_cast<DLManagedTensorVersioned *>(
        PyCapsule_GetPointer(capsule_guard.get(), "dltensor_versioned"));
      if (versioned_ == nullptr) {
        throw nb::python_error();
      }
      if (versioned_->version.major != 1U) {
        throw std::invalid_argument(std::string(name_) + " has unsupported DLPack major version");
      }
      tensor_ = &versioned_->dl_tensor;
      used_name = "used_dltensor_versioned";
    } else {
      throw std::invalid_argument(std::string(name_) + " __dlpack__ returned an invalid capsule");
    }

    validate(ndim, second_dim, dtype_code, dtype_bits, dtype_lanes);

    if (PyCapsule_SetName(capsule_guard.get(), used_name) != 0) {
      throw nb::python_error();
    }
    capsule_ = capsule_guard.release();
  }

  ~DLPackTensorView()
  {
    if (managed_ != nullptr && managed_->deleter != nullptr) {
      managed_->deleter(managed_);
    }
    if (versioned_ != nullptr && versioned_->deleter != nullptr) {
      versioned_->deleter(versioned_);
    }
    Py_XDECREF(capsule_);
  }

  DLPackTensorView(const DLPackTensorView &) = delete;
  DLPackTensorView & operator=(const DLPackTensorView &) = delete;

  static bool supportsDLPack(nb::handle object)
  {
    return PyObject_HasAttrString(object.ptr(), "__dlpack__") == 1;
  }

  template<typename T>
  const T * data() const
  {
    auto * bytes = static_cast<const std::uint8_t *>(tensor_->data);
    return reinterpret_cast<const T *>(bytes + tensor_->byte_offset);
  }

  int dim(int axis) const
  {
    if (tensor_->shape[axis] > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::string(name_) + " dimension is too large");
    }
    return static_cast<int>(tensor_->shape[axis]);
  }

private:
  void validate(
    int ndim, int second_dim,
    uint8_t dtype_code, uint8_t dtype_bits, uint16_t dtype_lanes) const
  {
    if (tensor_->device.device_type != kDLCUDA &&
      tensor_->device.device_type != kDLCUDAManaged)
    {
      throw std::invalid_argument(std::string(name_) + " DLPack tensor must live on a CUDA device");
    }
    if (tensor_->ndim != ndim) {
      throw std::invalid_argument(
              std::string(name_) + " DLPack tensor must have " + std::to_string(ndim) +
              " dimensions");
    }
    if (tensor_->shape == nullptr) {
      throw std::invalid_argument(std::string(name_) + " DLPack tensor has no shape");
    }
    if (second_dim >= 0 && tensor_->shape[1] != second_dim) {
      throw std::invalid_argument(
              std::string(name_) + " DLPack tensor must have shape (N, " +
              std::to_string(second_dim) + ")");
    }
    if (tensor_->dtype.code != dtype_code ||
      tensor_->dtype.bits != dtype_bits ||
      tensor_->dtype.lanes != dtype_lanes)
    {
      throw std::invalid_argument(std::string(name_) + " DLPack tensor has an unsupported dtype");
    }

    int64_t expected_stride = 1;
    if (tensor_->strides != nullptr) {
      for (int axis = tensor_->ndim - 1; axis >= 0; --axis) {
        if (tensor_->strides[axis] != expected_stride) {
          throw std::invalid_argument(std::string(name_) + " DLPack tensor must be C-contiguous");
        }
        expected_stride *= tensor_->shape[axis];
      }
    }
  }

  const char * name_;
  PyObject * capsule_ = nullptr;
  DLManagedTensor * managed_ = nullptr;
  DLManagedTensorVersioned * versioned_ = nullptr;
  DLTensor * tensor_ = nullptr;
};

template<size_t N>
std::array<float, N> readFloatSequence(nb::handle object, const char * name)
{
  PyObject * seq = PySequence_Fast(object.ptr(), name);
  if (seq == nullptr) {
    throw nb::python_error();
  }

  const Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
  if (len != static_cast<Py_ssize_t>(N)) {
    Py_DECREF(seq);
    throw std::invalid_argument(std::string(name) + " must have length " + std::to_string(N));
  }

  std::array<float, N> out{};
  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject * item = PySequence_Fast_GET_ITEM(seq, i);
    const double value = PyFloat_AsDouble(item);
    if (PyErr_Occurred()) {
      Py_DECREF(seq);
      throw nb::python_error();
    }
    out[static_cast<size_t>(i)] = static_cast<float>(value);
  }

  Py_DECREF(seq);
  return out;
}

class PyMppiPlanner
{
public:
  explicit PyMppiPlanner(const cr::MppiParams & params)
  : planner_(params)
  {
  }

  void reset()
  {
    planner_.reset();
  }

  void setSpeedLimit(float v_max)
  {
    planner_.setSpeedLimit(v_max);
  }

  nb::tuple compute(
    nb::object state,
    nb::object costmap,
    nb::object path,
    nb::object goal,
    nb::object origin,
    float resolution,
    bool goal_is_final,
    nb::object footprint)
  {
    const auto s = readFloatSequence<3>(state, "state");
    const auto g = readFloatSequence<3>(goal, "goal");
    const auto o = readFloatSequence<2>(origin, "origin");

    const unsigned char * costmap_ptr = nullptr;
    int size_x = 0;
    int size_y = 0;
    bool costmap_is_device = false;
    std::unique_ptr<BufferView> costmap_view;
    std::unique_ptr<DLPackTensorView> costmap_dlpack_view;
    if (costmap.ptr() != Py_None) {
      if (PyObject_CheckBuffer(costmap.ptr())) {
        costmap_view = std::make_unique<BufferView>(costmap, 2, -1, 1, "costmap");
        costmap_ptr = costmap_view->data<unsigned char>();
        size_y = costmap_view->dim(0);
        size_x = costmap_view->dim(1);
      } else if (DLPackTensorView::supportsDLPack(costmap)) {
        costmap_dlpack_view =
          std::make_unique<DLPackTensorView>(costmap, 2, -1, kDLUInt, 8, 1, "costmap");
        costmap_ptr = costmap_dlpack_view->data<unsigned char>();
        size_y = costmap_dlpack_view->dim(0);
        size_x = costmap_dlpack_view->dim(1);
        costmap_is_device = true;
      } else {
        throw std::invalid_argument(
                "costmap must support the Python buffer protocol or CUDA DLPack");
      }
    }

    const float * path_ptr = nullptr;
    int path_len = 0;
    std::unique_ptr<BufferView> path_view;
    if (path.ptr() != Py_None) {
      path_view = std::make_unique<BufferView>(path, 2, 2, 4, "path");
      path_ptr = path_view->data<float>();
      path_len = path_view->dim(0);
    }

    const float * footprint_ptr = nullptr;
    int footprint_len = 0;
    std::unique_ptr<BufferView> footprint_view;
    if (footprint.ptr() != Py_None) {
      footprint_view = std::make_unique<BufferView>(footprint, 2, 2, 4, "footprint");
      footprint_ptr = footprint_view->data<float>();
      footprint_len = footprint_view->dim(0);
    }

    return solve(
      s, costmap_ptr, size_x, size_y, costmap_is_device, o, resolution,
      path_ptr, path_len, g, goal_is_final, footprint_ptr, footprint_len);
  }

  // Plans on a costmap that already lives on the current CUDA device.
  nb::tuple computeOnDeviceCostmap(
    nb::object state,
    const unsigned char * device_costmap,
    int size_x,
    int size_y,
    nb::object path,
    nb::object goal,
    nb::object origin,
    float resolution,
    bool goal_is_final)
  {
    const auto s = readFloatSequence<3>(state, "state");
    const auto g = readFloatSequence<3>(goal, "goal");
    const auto o = readFloatSequence<2>(origin, "origin");
    BufferView path_view(path, 2, 2, 4, "path");
    return solve(
      s, device_costmap, size_x, size_y, true, o, resolution,
      path_view.data<float>(), path_view.dim(0), g, goal_is_final, nullptr, 0);
  }

private:
  nb::tuple solve(
    const std::array<float, 3> & s,
    const unsigned char * costmap_ptr,
    int size_x,
    int size_y,
    bool costmap_is_device,
    const std::array<float, 2> & o,
    float resolution,
    const float * path_ptr,
    int path_len,
    const std::array<float, 3> & g,
    bool goal_is_final,
    const float * footprint_ptr,
    int footprint_len)
  {
    cr::MppiResult result = costmap_is_device ?
      planner_.computeWithDeviceCostmap(
        s[0], s[1], s[2],
        costmap_ptr, size_x, size_y,
        o[0], o[1], resolution,
        path_ptr, path_len,
        g[0], g[1], g[2], goal_is_final,
        footprint_ptr, footprint_len) :
      planner_.compute(
        s[0], s[1], s[2],
        costmap_ptr, size_x, size_y,
        o[0], o[1], resolution,
        path_ptr, path_len,
        g[0], g[1], g[2], goal_is_final,
        footprint_ptr, footprint_len);

    nb::dict info;
    info["best_cost"] = result.best_cost;
    info["mean_cost"] = result.mean_cost;
    info["sampled_rollouts"] = result.sampled_rollouts;
    info["valid_rollouts"] = result.valid_rollouts;
    info["valid_rollout_ratio"] = result.valid_rollout_ratio;
    info["all_colliding"] = result.all_colliding;
    info["retreating"] = result.retreating;
    return nb::make_tuple(result.v, result.vy, result.w, info);
  }

  cr::MppiGpu planner_;
};

nb::tuple transformResultToTuple(const reg::FilterRegResult & result)
{
  nb::list rot_out;
  for (int i = 0; i < 9; ++i) {
    rot_out.append(result.rotation[i]);
  }
  nb::list trans_out;
  for (int k = 0; k < 3; ++k) {
    trans_out.append(result.translation[k]);
  }
  nb::dict info;
  info["iterations"] = result.iterations;
  info["final_rmse"] = result.final_rmse;
  return nb::make_tuple(rot_out, trans_out, info);
}

class PyFilterReg
{
public:
  explicit PyFilterReg(const reg::FilterRegParams & params)
  : registrar_(params)
  {
  }

  nb::tuple register_clouds(
    nb::object target,
    nb::object source,
    nb::object init_rotation,
    nb::object init_translation)
  {
    BufferView target_view(target, 2, 3, 4, "target");
    BufferView source_view(source, 2, 3, 4, "source");
    const int num_target = target_view.dim(0);
    const int num_source = source_view.dim(0);

    const float * init_r = nullptr;
    const float * init_t = nullptr;
    std::array<float, 9> init_r_arr{};
    std::array<float, 3> init_t_arr{};
    if (init_rotation.ptr() != Py_None) {
      init_r_arr = readFloatSequence<9>(init_rotation, "init_rotation");
      init_r = init_r_arr.data();
    }
    if (init_translation.ptr() != Py_None) {
      init_t_arr = readFloatSequence<3>(init_translation, "init_translation");
      init_t = init_t_arr.data();
    }

    reg::FilterRegResult result = registrar_.registerClouds(
      target_view.data<float>(), num_target,
      source_view.data<float>(), num_source,
      init_r, init_t);

    return transformResultToTuple(result);
  }

private:
  reg::FilterRegGpu registrar_;
};

class PySinkhornReg
{
public:
  explicit PySinkhornReg(const reg::SinkhornRegParams & params)
  : registrar_(params)
  {
  }

  nb::tuple register_clouds(
    nb::object target,
    nb::object source,
    nb::object init_rotation,
    nb::object init_translation)
  {
    BufferView target_view(target, 2, 3, 4, "target");
    BufferView source_view(source, 2, 3, 4, "source");
    const int num_target = target_view.dim(0);
    const int num_source = source_view.dim(0);

    const float * init_r = nullptr;
    const float * init_t = nullptr;
    std::array<float, 9> init_r_arr{};
    std::array<float, 3> init_t_arr{};
    if (init_rotation.ptr() != Py_None) {
      init_r_arr = readFloatSequence<9>(init_rotation, "init_rotation");
      init_r = init_r_arr.data();
    }
    if (init_translation.ptr() != Py_None) {
      init_t_arr = readFloatSequence<3>(init_translation, "init_translation");
      init_t = init_t_arr.data();
    }

    reg::RegTransformResult result = registrar_.registerClouds(
      target_view.data<float>(), num_target,
      source_view.data<float>(), num_source,
      init_r, init_t);

    return transformResultToTuple(result);
  }

private:
  reg::SinkhornRegGpu registrar_;
};

class PyFgr
{
public:
  explicit PyFgr(const reg::FgrParams & params)
  : registrar_(params)
  {
  }

  nb::tuple register_clouds(nb::object target, nb::object source)
  {
    BufferView target_view(target, 2, 3, 4, "target");
    BufferView source_view(source, 2, 3, 4, "source");
    reg::FgrResult result = registrar_.registerClouds(
      target_view.data<float>(), target_view.dim(0),
      source_view.data<float>(), source_view.dim(0));
    return transformResultToTuple(result);
  }

private:
  reg::FgrGpu registrar_;
};

class PyRobustTreg
{
public:
  explicit PyRobustTreg(const reg::RobustTregParams & params)
  : registrar_(params)
  {
  }

  nb::tuple register_clouds(
    nb::object target,
    nb::object source,
    nb::object init_rotation,
    nb::object init_translation)
  {
    BufferView target_view(target, 2, 3, 4, "target");
    BufferView source_view(source, 2, 3, 4, "source");
    const int num_target = target_view.dim(0);
    const int num_source = source_view.dim(0);

    const float * init_r = nullptr;
    const float * init_t = nullptr;
    std::array<float, 9> init_r_arr{};
    std::array<float, 3> init_t_arr{};
    if (init_rotation.ptr() != Py_None) {
      init_r_arr = readFloatSequence<9>(init_rotation, "init_rotation");
      init_r = init_r_arr.data();
    }
    if (init_translation.ptr() != Py_None) {
      init_t_arr = readFloatSequence<3>(init_translation, "init_translation");
      init_t = init_t_arr.data();
    }

    reg::RobustTregResult result = registrar_.registerClouds(
      target_view.data<float>(), num_target,
      source_view.data<float>(), num_source,
      init_r, init_t);

    return transformResultToTuple(result);
  }

private:
  reg::RobustTregGpu registrar_;
};

class PyRobustP2Plane
{
public:
  explicit PyRobustP2Plane(const reg::RobustP2PlaneParams & params)
  : registrar_(params)
  {
  }

  nb::tuple register_clouds(
    nb::object target,
    nb::object source,
    nb::object init_rotation,
    nb::object init_translation)
  {
    BufferView target_view(target, 2, 3, 4, "target");
    BufferView source_view(source, 2, 3, 4, "source");
    const int num_target = target_view.dim(0);
    const int num_source = source_view.dim(0);

    const float * init_r = nullptr;
    const float * init_t = nullptr;
    std::array<float, 9> init_r_arr{};
    std::array<float, 3> init_t_arr{};
    if (init_rotation.ptr() != Py_None) {
      init_r_arr = readFloatSequence<9>(init_rotation, "init_rotation");
      init_r = init_r_arr.data();
    }
    if (init_translation.ptr() != Py_None) {
      init_t_arr = readFloatSequence<3>(init_translation, "init_translation");
      init_t = init_t_arr.data();
    }

    reg::RobustP2PlaneResult result = registrar_.registerClouds(
      target_view.data<float>(), num_target,
      source_view.data<float>(), num_source,
      init_r, init_t);

    return transformResultToTuple(result);
  }

private:
  reg::RobustP2PlaneGpu registrar_;
};

class PyBcpd
{
public:
  explicit PyBcpd(const reg::BcpdParams & params)
  : registrar_(params)
  {
  }

  nb::tuple register_clouds(nb::object target, nb::object source)
  {
    BufferView target_view(target, 2, 3, 4, "target");
    BufferView source_view(source, 2, 3, 4, "source");
    reg::BcpdResult result = registrar_.registerClouds(
      target_view.data<float>(), target_view.dim(0),
      source_view.data<float>(), source_view.dim(0));

    const size_t num_points = result.deformed_xyz.size() / 3;
    auto * storage = new std::vector<float>(std::move(result.deformed_xyz));
    nb::ndarray<nb::numpy, float, nb::shape<-1, 3>, nb::c_contig> deformed(
      storage->data(), {num_points, size_t{3}},
      nb::capsule(storage, [](void * ptr) noexcept {
        delete static_cast<std::vector<float> *>(ptr);
      }));

    nb::dict info;
    info["iterations"] = result.iterations;
    info["final_sigma"] = result.final_sigma;
    info["mean_surface_distance"] = result.mean_surface_distance;
    return nb::make_tuple(deformed, info);
  }

private:
  reg::BcpdGpu registrar_;
};

template<typename T>
using NumpyArray = nb::ndarray<nb::numpy, T, nb::c_contig>;

// Move a host vector into a NumPy array that owns it (no copy).
template<typename T>
NumpyArray<T> toNumpy(std::vector<T> values, std::initializer_list<size_t> shape)
{
  auto * storage = new std::vector<T>(std::move(values));
  nb::capsule owner(storage, [](void * ptr) noexcept {
    delete static_cast<std::vector<T> *>(ptr);
  });
  return NumpyArray<T>(storage->data(), shape, owner);
}

nb::dict kissPoseToDict(const reg::KissIcpPose & pose)
{
  nb::list rotation;
  for (int i = 0; i < 9; ++i) {
    rotation.append(pose.R.m[i]);
  }
  nb::list translation;
  for (int k = 0; k < 3; ++k) {
    translation.append(pose.t[k]);
  }
  nb::dict out;
  out["rotation"] = rotation;
  out["translation"] = translation;
  return out;
}

reg::KissIcpPose kissPoseFrom(nb::object rotation, nb::object translation)
{
  reg::KissIcpPose pose;
  if (rotation.ptr() != Py_None) {
    const auto r = readFloatSequence<9>(rotation, "rotation");
    std::copy(r.begin(), r.end(), pose.R.m);
  }
  if (translation.ptr() != Py_None) {
    const auto t = readFloatSequence<3>(translation, "translation");
    std::copy(t.begin(), t.end(), pose.t);
  }
  return pose;
}

class PyKissIcpOdometry
{
public:
  explicit PyKissIcpOdometry(const reg::KissIcpConfig & config)
  : odometry_(validated(config))
  {
  }

  void reset(nb::object rotation, nb::object translation)
  {
    odometry_.reset(kissPoseFrom(rotation, translation));
  }

  nb::dict register_scan(
    nb::object points,
    nb::object point_times,
    nb::object scan_start,
    nb::object scan_end)
  {
    BufferView points_view(points, 2, 3, 4, "points");
    const auto count = static_cast<std::size_t>(points_view.dim(0));
    reg::KissIcpFrameResult result;
    if (point_times.ptr() == Py_None) {
      if (scan_start.ptr() != Py_None || scan_end.ptr() != Py_None) {
        throw std::invalid_argument("scan_start/scan_end require point_times");
      }
      result = odometry_.register_scan(points_view.data<float>(), count);
    } else {
      BufferView times_view(point_times, 1, -1, 4, "point_times");
      if (static_cast<std::size_t>(times_view.dim(0)) != count) {
        throw std::invalid_argument("point_times must have one entry per point");
      }
      if ((scan_start.ptr() == Py_None) != (scan_end.ptr() == Py_None)) {
        throw std::invalid_argument("scan_start and scan_end must be given together");
      }
      if (scan_start.ptr() == Py_None) {
        result = odometry_.register_scan(
          points_view.data<float>(), count, times_view.data<float>());
      } else {
        result = odometry_.register_scan(
          points_view.data<float>(), count, times_view.data<float>(),
          nb::cast<float>(scan_start), nb::cast<float>(scan_end));
      }
    }

    nb::dict out = kissPoseToDict(result.pose);
    out["iterations"] = result.alignment.iterations;
    out["inliers"] = result.alignment.inliers;
    out["rmse"] = result.alignment.rmse;
    out["nn_ms"] = result.alignment.nn_ms;
    out["normal_equation_ms"] = result.alignment.normal_equation_ms;
    out["threshold"] = result.alignment.threshold;
    out["input_points"] = result.input_points;
    out["sampled_points"] = result.sampled_points;
    out["map_points"] = result.map_points;
    out["map_initialized"] = result.map_initialized;
    out["deskewed"] = result.deskewed;
    out["point_time_span_s"] = result.point_time_span_s;
    const size_t deskewed_points = result.deskewed_xyz.size() / 3;
    out["deskewed_xyz"] = toNumpy(std::move(result.deskewed_xyz), {deskewed_points, size_t{3}});
    return out;
  }

  nb::dict pose() const
  {
    return kissPoseToDict(odometry_.pose());
  }

  std::size_t frame_count() const
  {
    return odometry_.frame_count();
  }

  std::size_t map_point_count() const
  {
    return odometry_.map_point_count();
  }

  NumpyArray<float> map_snapshot() const
  {
    auto values = odometry_.map_snapshot();
    const size_t points = values.size() / 3;
    return toNumpy(std::move(values), {points, size_t{3}});
  }

  nb::dict timing() const
  {
    const reg::KissIcpTiming t = odometry_.timing();
    nb::dict out;
    out["deskew_ms"] = t.deskew_ms;
    out["index_build_ms"] = t.index_build_ms;
    out["map_upload_ms"] = t.map_upload_ms;
    out["map_normal_ms"] = t.map_normal_ms;
    out["validation_ms"] = t.validation_ms;
    out["deskew_wall_ms"] = t.deskew_wall_ms;
    out["downsample_ms"] = t.downsample_ms;
    out["icp_ms"] = t.icp_ms;
    out["map_update_ms"] = t.map_update_ms;
    out["map_prune_ms"] = t.map_prune_ms;
    out["map_insert_ms"] = t.map_insert_ms;
    out["map_pack_ms"] = t.map_pack_ms;
    out["map_reorder_ms"] = t.map_reorder_ms;
    out["normal_cache_prepare_ms"] = t.normal_cache_prepare_ms;
    out["normal_reused_points"] = t.normal_reused_points;
    out["normal_recomputed_points"] = t.normal_recomputed_points;
    return out;
  }

private:
  static const reg::KissIcpConfig & validated(const reg::KissIcpConfig & config)
  {
    const std::string error = reg::validate_kiss_icp_config(config);
    if (!error.empty()) {
      throw std::invalid_argument(error);
    }
    return config;
  }

  reg::KissIcpOdometry odometry_;
};

nb::dict gridInfoToDict(const reg::VoxelGridInfo & grid)
{
  nb::dict out;
  out["width"] = grid.width;
  out["height"] = grid.height;
  out["depth"] = grid.depth;
  out["resolution"] = grid.resolution;
  out["origin_x"] = grid.origin_x;
  out["origin_y"] = grid.origin_y;
  out["origin_z"] = grid.origin_z;
  return out;
}

class PyVoxelMapper
{
public:
  explicit PyVoxelMapper(const reg::VoxelMappingConfig & config)
  : mapper_(validated(config))
  {
  }

  void reset(float center_x, float center_y)
  {
    mapper_.reset(center_x, center_y);
  }

  nb::dict integrate_scan(nb::object points_world, nb::object sensor_origin)
  {
    BufferView points_view(points_world, 2, 3, 4, "points_world");
    const auto origin = readFloatSequence<3>(sensor_origin, "sensor_origin");
    const reg::VoxelMappingStats stats = mapper_.integrate_scan(
      points_view.data<float>(), static_cast<std::size_t>(points_view.dim(0)),
      origin.data());
    nb::dict out;
    out["input_points"] = stats.input_points;
    out["integrated_rays"] = stats.integrated_rays;
    out["observed_voxels"] = stats.observed_voxels;
    out["grid_shifted"] = stats.grid_shifted;
    out["shift_x_cells"] = stats.shift_x_cells;
    out["shift_y_cells"] = stats.shift_y_cells;
    out["shift_ms"] = stats.shift_ms;
    out["raycast_ms"] = stats.raycast_ms;
    return out;
  }

  nb::dict grid_info() const
  {
    return gridInfoToDict(mapper_.grid_info());
  }

  nb::tuple occupancy_projection()
  {
    reg::OccupancyProjection projection = mapper_.occupancy_projection();
    const auto height = static_cast<size_t>(projection.grid.height);
    const auto width = static_cast<size_t>(projection.grid.width);
    if (projection.data.size() != height * width) {
      throw std::runtime_error("occupancy projection size does not match its grid");
    }
    auto data = toNumpy(std::move(projection.data), {height, width});
    return nb::make_tuple(data, gridInfoToDict(projection.grid), projection.gpu_ms);
  }

  reg::VoxelMapperGpu & core() { return mapper_; }

private:
  static const reg::VoxelMappingConfig & validated(const reg::VoxelMappingConfig & config)
  {
    const std::string error = reg::validate_voxel_mapping_config(config);
    if (!error.empty()) {
      throw std::invalid_argument(error);
    }
    return config;
  }

  reg::VoxelMapperGpu mapper_;
};

nb::tuple esdfResultToTuple(reg::Esdf2DResult result)
{
  const auto height = static_cast<size_t>(result.height);
  const auto width = static_cast<size_t>(result.width);
  if (result.distances.size() != height * width) {
    throw std::runtime_error("ESDF result size does not match its grid");
  }
  nb::dict info;
  info["resolution"] = result.resolution;
  info["max_distance"] = result.max_distance;
  info["occupied_cells"] = result.occupied_cells;
  info["unknown_cells"] = result.unknown_cells;
  info["gpu_ms"] = result.gpu_ms;
  auto distances = toNumpy(std::move(result.distances), {height, width});
  return nb::make_tuple(distances, info);
}

class PyEsdf2D
{
public:
  explicit PyEsdf2D(const reg::Esdf2DConfig & config)
  : esdf_(config)
  {
  }

  nb::tuple compute(nb::object occupancy, float resolution, float max_distance)
  {
    BufferView view(occupancy, 2, -1, 1, "occupancy");
    return esdfResultToTuple(esdf_.compute(
      view.data<std::int8_t>(), view.dim(1), view.dim(0), resolution, max_distance));
  }

  reg::Esdf2DGpu & core() { return esdf_; }

private:
  reg::Esdf2DGpu esdf_;
};

// Keeps occupancy -> ESDF -> costmap -> MPPI on the GPU. The device buffers
// belong to the voxel mapper, the ESDF and this object; grids are copied to
// NumPy only when requested. `generation` increases on every project() so the
// Python layer can refuse to read grids that a later frame has overwritten.
class PyCudaNavGridChain
{
public:
  PyCudaNavGridChain(int max_width, int max_height)
  : builder_(max_width, max_height)
  {
  }

  nb::dict project(PyVoxelMapper & voxel, float x, float y, float radius)
  {
    const reg::OccupancyProjectionDevice projection = voxel.core().occupancy_projection_device();
    const reg::CostmapGridGeometry geometry{
      projection.grid.width, projection.grid.height, projection.grid.origin_x,
      projection.grid.origin_y, projection.grid.resolution};
    // A non-positive radius clears nothing, matching the NumPy helper.
    const std::size_t occupied = builder_.clear_footprint(
      projection.data, geometry, x, y, radius > 0.0f ? radius : -1.0f);
    occupancy_ = projection.data;
    distances_ = nullptr;
    costmap_ = nullptr;
    width_ = projection.grid.width;
    height_ = projection.grid.height;
    ++generation_;
    nb::dict out;
    out["grid"] = gridInfoToDict(projection.grid);
    out["gpu_ms"] = projection.gpu_ms;
    out["occupied_cells"] = occupied;
    out["generation"] = generation_;
    return out;
  }

  nb::dict distance(PyEsdf2D & esdf, float resolution, float max_distance)
  {
    requireStage(occupancy_, "project");
    const reg::Esdf2DDeviceResult result =
      esdf.core().compute_device(occupancy_, width_, height_, resolution, max_distance);
    distances_ = result.distances;
    nb::dict info;
    info["resolution"] = result.resolution;
    info["max_distance"] = result.max_distance;
    info["occupied_cells"] = result.occupied_cells;
    info["unknown_cells"] = result.unknown_cells;
    info["gpu_ms"] = result.gpu_ms;
    return info;
  }

  void build()
  {
    requireStage(distances_, "distance");
    costmap_ = builder_.build(occupancy_, distances_, width_, height_).costmap;
  }

  nb::tuple plan(
    PyMppiPlanner & planner,
    nb::object state,
    nb::object path,
    nb::object goal,
    nb::object origin,
    float resolution,
    bool goal_is_final)
  {
    requireStage(costmap_, "build");
    return planner.computeOnDeviceCostmap(
      state, costmap_, width_, height_, path, goal, origin, resolution, goal_is_final);
  }

  NumpyArray<std::int8_t> occupancy() const
  {
    requireStage(occupancy_, "project");
    return download(occupancy_);
  }

  NumpyArray<float> distances() const
  {
    requireStage(distances_, "distance");
    return download(distances_);
  }

  NumpyArray<unsigned char> costmap() const
  {
    requireStage(costmap_, "build");
    return download(costmap_);
  }

  std::uint64_t generation() const { return generation_; }

private:
  template<typename T>
  NumpyArray<T> download(const T * device) const
  {
    std::vector<T> host(static_cast<std::size_t>(width_) * height_);
    const cudaError_t error = cudaMemcpy(
      host.data(), device, host.size() * sizeof(T), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
    }
    return toNumpy(std::move(host), {static_cast<size_t>(height_), static_cast<size_t>(width_)});
  }

  static void requireStage(const void * pointer, const char * stage)
  {
    if (!pointer) {
      throw std::runtime_error(std::string("call ") + stage + "() first in this frame");
    }
  }

  reg::CostmapBuilderGpu builder_;
  std::int8_t * occupancy_ = nullptr;
  const float * distances_ = nullptr;
  const unsigned char * costmap_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  std::uint64_t generation_ = 0;
};

nb::tuple esdfCpuReference(
  nb::object occupancy,
  float resolution,
  float max_distance,
  int occupancy_threshold,
  reg::UnknownSpacePolicy unknown_policy)
{
  BufferView view(occupancy, 2, -1, 1, "occupancy");
  const int height = view.dim(0);
  const int width = view.dim(1);
  const auto * data = view.data<std::int8_t>();
  std::vector<std::int8_t> cells(data, data + static_cast<std::size_t>(width) * height);
  return esdfResultToTuple(reg::compute_esdf_2d_cpu_reference(
    cells, width, height, resolution, max_distance, occupancy_threshold, unknown_policy));
}

// Report the CUDA runtime/driver and the active device without raising when no
// GPU is present, so evidence manifests can record the failure reason.
nb::dict cudaRuntimeInfo()
{
  nb::dict out;
  int runtime_version = 0;
  int driver_version = 0;
  cudaRuntimeGetVersion(&runtime_version);
  cudaDriverGetVersion(&driver_version);
  out["runtime_version"] = runtime_version;
  out["driver_version"] = driver_version;
  int device_count = 0;
  const cudaError_t count_error = cudaGetDeviceCount(&device_count);
  if (count_error != cudaSuccess) {
    cudaGetLastError();
    out["device_count"] = 0;
    out["error"] = std::string(cudaGetErrorString(count_error));
    return out;
  }
  out["device_count"] = device_count;
  int device = 0;
  cudaDeviceProp properties{};
  if (device_count > 0 && cudaGetDevice(&device) == cudaSuccess &&
    cudaGetDeviceProperties(&properties, device) == cudaSuccess)
  {
    out["device"] = device;
    out["name"] = std::string(properties.name);
    out["compute_capability"] = nb::make_tuple(properties.major, properties.minor);
    out["total_memory_bytes"] = static_cast<std::size_t>(properties.totalGlobalMem);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
      out["free_memory_bytes"] = free_bytes;
    } else {
      cudaGetLastError();
    }
  }
  return out;
}

}  // namespace

NB_MODULE(_cudarobotics, m)
{
  m.doc() = "CUDA Robotics Python bindings";
  m.attr("__version__") = "0.4.2";

  nb::enum_<cr::MotionModel>(m, "MotionModel")
    .value("DiffDrive", cr::MotionModel::DiffDrive)
    .value("Ackermann", cr::MotionModel::Ackermann)
    .value("Omni", cr::MotionModel::Omni)
    .export_values();

  nb::class_<cr::MppiParams>(m, "MppiParams")
    .def(nb::init<>())
    .def_rw("batch_size", &cr::MppiParams::batch_size)
    .def_rw("time_steps", &cr::MppiParams::time_steps)
    .def_rw("model_dt", &cr::MppiParams::model_dt)
    .def_rw("iteration_count", &cr::MppiParams::iteration_count)
    .def_rw("motion_model", &cr::MppiParams::motion_model)
    .def_rw("v_max", &cr::MppiParams::v_max)
    .def_rw("v_min", &cr::MppiParams::v_min)
    .def_rw("vy_max", &cr::MppiParams::vy_max)
    .def_rw("w_max", &cr::MppiParams::w_max)
    .def_rw("min_turning_r", &cr::MppiParams::min_turning_r)
    .def_rw("v_std", &cr::MppiParams::v_std)
    .def_rw("vy_std", &cr::MppiParams::vy_std)
    .def_rw("w_std", &cr::MppiParams::w_std)
    .def_rw("lambda_", &cr::MppiParams::lambda)
    .def_rw("goal_weight", &cr::MppiParams::goal_weight)
    .def_rw("goal_yaw_weight", &cr::MppiParams::goal_yaw_weight)
    .def_rw("path_weight", &cr::MppiParams::path_weight)
    .def_rw("path_follow_weight", &cr::MppiParams::path_follow_weight)
    .def_rw("path_angle_weight", &cr::MppiParams::path_angle_weight)
    .def_rw("curvature_speed_weight", &cr::MppiParams::curvature_speed_weight)
    .def_rw("curvature_speed_min", &cr::MppiParams::curvature_speed_min)
    .def_rw("follow_lookahead", &cr::MppiParams::follow_lookahead)
    .def_rw("costmap_weight", &cr::MppiParams::costmap_weight)
    .def_rw("distance_field_weight", &cr::MppiParams::distance_field_weight)
    .def_rw("distance_field_cutoff", &cr::MppiParams::distance_field_cutoff)
    .def_rw("smoothness_weight", &cr::MppiParams::smoothness_weight)
    .def_rw("backward_weight", &cr::MppiParams::backward_weight)
    .def_rw("speed_weight", &cr::MppiParams::speed_weight)
    .def_rw("angular_weight", &cr::MppiParams::angular_weight)
    .def_rw("collision_cost", &cr::MppiParams::collision_cost)
    .def_rw("yaw_goal_activation_dist", &cr::MppiParams::yaw_goal_activation_dist)
    .def_rw("lethal_threshold", &cr::MppiParams::lethal_threshold)
    .def_rw("consider_footprint", &cr::MppiParams::consider_footprint)
    .def_rw("enable_retreat", &cr::MppiParams::enable_retreat)
    .def_rw("retreat_scale", &cr::MppiParams::retreat_scale);

  nb::class_<cr::MppiResult>(m, "MppiResult")
    .def(nb::init<>())
    .def_rw("v", &cr::MppiResult::v)
    .def_rw("vy", &cr::MppiResult::vy)
    .def_rw("w", &cr::MppiResult::w)
    .def_rw("best_cost", &cr::MppiResult::best_cost)
    .def_rw("mean_cost", &cr::MppiResult::mean_cost)
    .def_rw("sampled_rollouts", &cr::MppiResult::sampled_rollouts)
    .def_rw("valid_rollouts", &cr::MppiResult::valid_rollouts)
    .def_rw("valid_rollout_ratio", &cr::MppiResult::valid_rollout_ratio)
    .def_rw("all_colliding", &cr::MppiResult::all_colliding)
    .def_rw("retreating", &cr::MppiResult::retreating);

  nb::class_<PyMppiPlanner>(m, "_MppiPlanner")
    .def(nb::init<const cr::MppiParams &>())
    .def("reset", &PyMppiPlanner::reset)
    .def("set_speed_limit", &PyMppiPlanner::setSpeedLimit, "v_max"_a)
    .def(
      "compute", &PyMppiPlanner::compute,
      "state"_a, "costmap"_a, "path"_a, "goal"_a,
      "origin"_a = nb::make_tuple(0.0f, 0.0f),
      "resolution"_a = 0.05f,
      "goal_is_final"_a = false,
      "footprint"_a = nb::none());

  nb::class_<reg::FilterRegParams>(m, "FilterRegParams")
    .def(nb::init<>())
    .def_rw("voxel_size", &reg::FilterRegParams::voxel_size)
    .def_rw("bbox_margin", &reg::FilterRegParams::bbox_margin)
    .def_rw("outlier_fraction", &reg::FilterRegParams::outlier_fraction)
    .def_rw("iters_per_sigma", &reg::FilterRegParams::iters_per_sigma)
    .def_rw("step_tol", &reg::FilterRegParams::step_tol);

  nb::class_<PyFilterReg>(m, "_FilterReg")
    .def(nb::init<const reg::FilterRegParams &>())
    .def(
      "register_clouds", &PyFilterReg::register_clouds,
      "target"_a, "source"_a,
      "init_rotation"_a = nb::none(),
      "init_translation"_a = nb::none());

  nb::class_<reg::SinkhornRegParams>(m, "SinkhornRegParams")
    .def(nb::init<>())
    .def_rw("rho", &reg::SinkhornRegParams::rho)
    .def_rw("sinkhorn_iters", &reg::SinkhornRegParams::sinkhorn_iters)
    .def_rw("outer_iters", &reg::SinkhornRegParams::outer_iters)
    .def_rw("gn_iters", &reg::SinkhornRegParams::gn_iters);

  nb::class_<PySinkhornReg>(m, "_SinkhornReg")
    .def(nb::init<const reg::SinkhornRegParams &>())
    .def(
      "register_clouds", &PySinkhornReg::register_clouds,
      "target"_a, "source"_a,
      "init_rotation"_a = nb::none(),
      "init_translation"_a = nb::none());

  nb::class_<reg::FgrParams>(m, "FgrParams")
    .def(nb::init<>())
    .def_rw("gn_levels", &reg::FgrParams::gn_levels)
    .def_rw("gn_steps_per_level", &reg::FgrParams::gn_steps_per_level)
    .def_rw("mu_decay", &reg::FgrParams::mu_decay)
    .def_rw("min_mu", &reg::FgrParams::min_mu);

  nb::class_<PyFgr>(m, "_Fgr")
    .def(nb::init<const reg::FgrParams &>())
    .def("register_clouds", &PyFgr::register_clouds, "target"_a, "source"_a);

  nb::class_<reg::BcpdParams>(m, "BcpdParams")
    .def(nb::init<>())
    .def_rw("beta", &reg::BcpdParams::beta)
    .def_rw("lambda_", &reg::BcpdParams::lambda)
    .def_rw("max_iters", &reg::BcpdParams::max_iters);

  nb::class_<PyBcpd>(m, "_Bcpd")
    .def(nb::init<const reg::BcpdParams &>())
    .def("register_clouds", &PyBcpd::register_clouds, "target"_a, "source"_a);

  nb::class_<reg::RobustTregParams>(m, "RobustTregParams")
    .def(nb::init<>())
    .def_rw("nu", &reg::RobustTregParams::nu)
    .def_rw("outlier_fraction", &reg::RobustTregParams::outlier_fraction)
    .def_rw("outer_iters_per_sigma", &reg::RobustTregParams::outer_iters_per_sigma)
    .def_rw("gn_iters", &reg::RobustTregParams::gn_iters);

  nb::class_<PyRobustTreg>(m, "_RobustTreg")
    .def(nb::init<const reg::RobustTregParams &>())
    .def(
      "register_clouds", &PyRobustTreg::register_clouds,
      "target"_a, "source"_a,
      "init_rotation"_a = nb::none(),
      "init_translation"_a = nb::none());

  nb::class_<reg::RobustP2PlaneParams>(m, "RobustP2PlaneParams")
    .def(nb::init<>())
    .def_rw("nu", &reg::RobustP2PlaneParams::nu)
    .def_rw("knn_k", &reg::RobustP2PlaneParams::knn_k)
    .def_rw("outlier_fraction", &reg::RobustP2PlaneParams::outlier_fraction)
    .def_rw("outer_iters_per_sigma", &reg::RobustP2PlaneParams::outer_iters_per_sigma)
    .def_rw("gn_iters", &reg::RobustP2PlaneParams::gn_iters);

  nb::class_<PyRobustP2Plane>(m, "_RobustP2Plane")
    .def(nb::init<const reg::RobustP2PlaneParams &>())
    .def(
      "register_clouds", &PyRobustP2Plane::register_clouds,
      "target"_a, "source"_a,
      "init_rotation"_a = nb::none(),
      "init_translation"_a = nb::none());

  // --- CudaNav navigation cores (cudarobotics.navigation) ---
  nb::class_<reg::KissIcpConfig>(m, "KissIcpConfig")
    .def(nb::init<>())
    .def_rw("map_voxel_size", &reg::KissIcpConfig::map_voxel_size)
    .def_rw("scan_voxel_size", &reg::KissIcpConfig::scan_voxel_size)
    .def_rw("map_radius", &reg::KissIcpConfig::map_radius)
    .def_rw("threshold_min", &reg::KissIcpConfig::threshold_min)
    .def_rw("threshold_max", &reg::KissIcpConfig::threshold_max)
    .def_rw("max_icp_iterations", &reg::KissIcpConfig::max_icp_iterations)
    .def_rw("normal_neighbors", &reg::KissIcpConfig::normal_neighbors)
    .def_rw("max_scan_points", &reg::KissIcpConfig::max_scan_points)
    .def_rw("max_map_points", &reg::KissIcpConfig::max_map_points)
    .def_rw("hash_capacity", &reg::KissIcpConfig::hash_capacity);

  nb::class_<PyKissIcpOdometry>(m, "_KissIcpOdometry")
    .def(nb::init<const reg::KissIcpConfig &>())
    .def("reset", &PyKissIcpOdometry::reset,
      "rotation"_a = nb::none(), "translation"_a = nb::none())
    .def(
      "register_scan", &PyKissIcpOdometry::register_scan,
      "points"_a, "point_times"_a = nb::none(),
      "scan_start"_a = nb::none(), "scan_end"_a = nb::none())
    .def("pose", &PyKissIcpOdometry::pose)
    .def("frame_count", &PyKissIcpOdometry::frame_count)
    .def("map_point_count", &PyKissIcpOdometry::map_point_count)
    .def("map_snapshot", &PyKissIcpOdometry::map_snapshot)
    .def("timing", &PyKissIcpOdometry::timing);

  nb::class_<reg::VoxelMappingConfig>(m, "VoxelMappingConfig")
    .def(nb::init<>())
    .def_rw("width", &reg::VoxelMappingConfig::width)
    .def_rw("height", &reg::VoxelMappingConfig::height)
    .def_rw("depth", &reg::VoxelMappingConfig::depth)
    .def_rw("resolution", &reg::VoxelMappingConfig::resolution)
    .def_rw("origin_z", &reg::VoxelMappingConfig::origin_z)
    .def_rw("min_range", &reg::VoxelMappingConfig::min_range)
    .def_rw("max_range", &reg::VoxelMappingConfig::max_range)
    .def_rw("log_odds_occupied", &reg::VoxelMappingConfig::log_odds_occupied)
    .def_rw("log_odds_free", &reg::VoxelMappingConfig::log_odds_free)
    .def_rw("log_odds_min", &reg::VoxelMappingConfig::log_odds_min)
    .def_rw("log_odds_max", &reg::VoxelMappingConfig::log_odds_max)
    .def_rw("occupied_threshold", &reg::VoxelMappingConfig::occupied_threshold)
    .def_rw("projection_min_z", &reg::VoxelMappingConfig::projection_min_z)
    .def_rw("projection_max_z", &reg::VoxelMappingConfig::projection_max_z)
    .def_rw("rolling_margin_cells", &reg::VoxelMappingConfig::rolling_margin_cells)
    .def_rw("max_scan_points", &reg::VoxelMappingConfig::max_scan_points);

  nb::class_<PyVoxelMapper>(m, "_VoxelMapper")
    .def(nb::init<const reg::VoxelMappingConfig &>())
    .def("reset", &PyVoxelMapper::reset, "center_x"_a, "center_y"_a)
    .def(
      "integrate_scan", &PyVoxelMapper::integrate_scan,
      "points_world"_a, "sensor_origin"_a)
    .def("grid_info", &PyVoxelMapper::grid_info)
    .def("occupancy_projection", &PyVoxelMapper::occupancy_projection);

  nb::enum_<reg::UnknownSpacePolicy>(m, "UnknownSpacePolicy")
    .value("Free", reg::UnknownSpacePolicy::Free)
    .value("Occupied", reg::UnknownSpacePolicy::Occupied);

  nb::class_<reg::Esdf2DConfig>(m, "Esdf2DConfig")
    .def(nb::init<>())
    .def_rw("max_width", &reg::Esdf2DConfig::max_width)
    .def_rw("max_height", &reg::Esdf2DConfig::max_height)
    .def_rw("occupancy_threshold", &reg::Esdf2DConfig::occupancy_threshold)
    .def_rw("unknown_policy", &reg::Esdf2DConfig::unknown_policy);

  nb::class_<PyEsdf2D>(m, "_Esdf2D")
    .def(nb::init<const reg::Esdf2DConfig &>())
    .def(
      "compute", &PyEsdf2D::compute,
      "occupancy"_a, "resolution"_a, "max_distance"_a);

  nb::class_<PyCudaNavGridChain>(m, "_CudaNavGridChain")
    .def(nb::init<int, int>(), "max_width"_a, "max_height"_a)
    .def("project", &PyCudaNavGridChain::project, "voxel"_a, "x"_a, "y"_a, "radius"_a)
    .def(
      "distance", &PyCudaNavGridChain::distance,
      "esdf"_a, "resolution"_a, "max_distance"_a)
    .def("build", &PyCudaNavGridChain::build)
    .def(
      "plan", &PyCudaNavGridChain::plan,
      "planner"_a, "state"_a, "path"_a, "goal"_a, "origin"_a, "resolution"_a,
      "goal_is_final"_a)
    .def("occupancy", &PyCudaNavGridChain::occupancy)
    .def("distances", &PyCudaNavGridChain::distances)
    .def("costmap", &PyCudaNavGridChain::costmap)
    .def_prop_ro("generation", &PyCudaNavGridChain::generation);

  m.def(
    "_esdf_2d_cpu_reference", &esdfCpuReference,
    "occupancy"_a, "resolution"_a, "max_distance"_a,
    "occupancy_threshold"_a, "unknown_policy"_a);
  m.def("cuda_runtime_info", &cudaRuntimeInfo);
}
