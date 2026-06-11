#include <Python.h>

#include <nanobind/nanobind.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_mppi_controller/mppi_gpu.hpp"
#include "cudarobotics/bcpd_gpu.hpp"
#include "cudarobotics/fgr_gpu.hpp"
#include "cudarobotics/filterreg_gpu.hpp"
#include "cudarobotics/robust_p2plane_gpu.hpp"
#include "cudarobotics/robust_treg_gpu.hpp"
#include "cudarobotics/sinkhorn_reg_gpu.hpp"

#include <nanobind/ndarray.h>

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
    info["all_colliding"] = result.all_colliding;
    info["retreating"] = result.retreating;
    return nb::make_tuple(result.v, result.vy, result.w, info);
  }

private:
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

    const int num_points = static_cast<int>(result.deformed_xyz.size() / 3);
    auto * storage = new std::vector<float>(std::move(result.deformed_xyz));
    nb::ndarray<nb::numpy, float, nb::shape<-1, 3>, nb::c_contig> deformed(
      storage->data(), {num_points, 3},
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

}  // namespace

NB_MODULE(_cudarobotics, m)
{
  m.doc() = "CUDA Robotics Python bindings";
  m.attr("__version__") = "0.1.0";

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
}
