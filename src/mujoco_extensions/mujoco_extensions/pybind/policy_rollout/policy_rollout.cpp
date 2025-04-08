/* Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved. */
#include <mujoco/mujoco.h>
#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <unsupported/Eigen/CXX11/Tensor>
#include "mujoco_extensions/system/system_class.h"
#include "mujoco_extensions/system/system_utils.h"
#include "pybind11/eval.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <thread>
#include <vector>

namespace mujoco_extensions::pybind::policy_rollout
{

  namespace py = pybind11;

  // Setup OpenMP
  constexpr std::size_t OMP_NUM_THREADS{8};

  // EigenTypes aliases
  using EigenTypes::MatrixT;
  using EigenTypes::MatrixTList;
  using EigenTypes::Tensor3d;
  using EigenTypes::VectorT;
  using EigenTypes::VectorTList;

  // Anonymous namespace for helper functions
  namespace
  {

    using UnalignedMap = Eigen::Map<MatrixT, Eigen::Unaligned>;
    using ConstMap = Eigen::Map<const MatrixT, Eigen::Unaligned>;

    // Build mutable maps into a 3-D numpy array
    static std::vector<UnalignedMap> make_maps(py::array_t<double> &arr)
    {
      auto buf = arr.request();
      if (buf.ndim != 3)
        throw std::runtime_error("Expected a 3-D array");
      int S = buf.shape[0], R = buf.shape[1], C = buf.shape[2];
      double *ptr = static_cast<double *>(buf.ptr);

      std::vector<UnalignedMap> maps;
      maps.reserve(S);
      for (int i = 0; i < S; ++i)
        maps.emplace_back(ptr + i * R * C, R, C);
      return maps;
    }

  } // anonymous namespace

  // Utility functions for model/data extraction (unchanged)
  std::tuple<const mjModel *, const mjData *> getModelAndData(const py::object &plantObject)
  {
    const auto modelPointer = plantObject.attr("model").attr("_address").cast<std::uintptr_t>();
    const mjModel *model = reinterpret_cast<mjModel *>(modelPointer);
    const auto dataPointer = plantObject.attr("data").attr("_address").cast<std::uintptr_t>();
    const mjData *data = reinterpret_cast<mjData *>(dataPointer);
    return {model, data};
  }

  std::vector<const mjModel *> getModelVector(const py::list &python_model)
  {
    std::vector<const mjModel *> model_vector;
    for (const auto &item : python_model)
    {
      const auto modelPointer = item.attr("_address").cast<std::uintptr_t>();
      model_vector.push_back(reinterpret_cast<const mjModel *>(modelPointer));
    }
    return model_vector;
  }

  std::vector<mjData *> getDataVector(const py::list &python_data)
  {
    std::vector<mjData *> data_vector;
    for (const auto &item : python_data)
    {
      const auto dataPointer = item.attr("_address").cast<std::uintptr_t>();
      data_vector.push_back(reinterpret_cast<mjData *>(dataPointer));
    }
    return data_vector;
  }

  /**
   * Bindings for `policy_rollout` submodule.
   */
  void bindPolicyRollout(const std::reference_wrapper<py::module> &root)
  {
    using pybind11::literals::operator""_a;

    Eigen::setNbThreads(OMP_NUM_THREADS);
    // Create `policy_rollout` submodule.
    auto python_module = root.get().def_submodule("policy_rollout");

    // SystemUtils
    python_module.def(
        "get_joint_proportional_gains",
        [](const py::object &plantObject) -> VectorT
        {
          const auto [model, _] = getModelAndData(plantObject);
          return SystemUtils::getJointProportionalGains(model);
        },
        "Retrieve the proportional gain used by the PD controller from the mujoco model.");

    python_module.def(
        "set_state",
        [](std::shared_ptr<SystemClass::System> systemObject, const VectorT &state) -> void
        {
          SystemUtils::setState(systemObject->model, systemObject->data, state);
        },
        "Sets the state of a system object", py::arg("system"), py::arg("state"));

    python_module.def(
        "threaded_physics_rollout",
        [](const py::list &python_model, const py::list &python_data, const VectorTList &state,
           const MatrixTList &control) -> std::tuple<MatrixTList, MatrixTList>
        {
          // Convert Python lists to C++ vectors of raw pointers
          auto model = getModelVector(python_model);
          std::vector<mjData *> data = getDataVector(python_data);

          // Call the threadedPhysicsRollout function
          return SystemUtils::threadedPhysicsRollout(model, data, state, control);
        },
        "model"_a, "data"_a, "state"_a, "control"_a);

    python_module.def(
        "threaded_physics_rollout",
        [](const py::list &python_model, const py::list &python_data, const MatrixT &state,
           const Tensor3d &control) -> std::tuple<MatrixTList, MatrixTList>
        {
          auto model = getModelVector(python_model);
          std::vector<mjData *> data = getDataVector(python_data);
          auto state_list = EigenTypes::matrix_to_vector_list(state);
          auto control_list = EigenTypes::tensor_to_matrix_list(control);
          return SystemUtils::threadedPhysicsRollout(model, data, state_list, control_list);
        },
        "model"_a, "data"_a, "state"_a, "control"_a);

    // In-place threaded rollout
    python_module.def(
        "threaded_physics_rollout_in_place",
        [](const py::list &python_model,
           const py::list &python_data,
           const MatrixT &state,
           const Tensor3d &control,
           py::array_t<double> output_states,
           py::array_t<double> output_sensors)
        {
          auto model = getModelVector(python_model);
          auto data = getDataVector(python_data);
          auto state_list = EigenTypes::matrix_to_vector_list(state);
          auto control_list = EigenTypes::tensor_to_matrix_list(control);

          auto output_states_maps = make_maps(output_states);
          auto output_sensors_maps = make_maps(output_sensors);

          py::gil_scoped_release release;
          SystemUtils::threadedPhysicsRolloutInPlace(
              model,
              data,
              state_list,
              control_list,
              output_states_maps,
              output_sensors_maps);
        },
        "model"_a, "data"_a, "state"_a, "control"_a, "states"_a, "sensors"_a);

    // SystemClass
    py::class_<SystemClass::System, std::shared_ptr<SystemClass::System>>(python_module, "System")
        .def(py::init<const std::string &, const std::string &>(), "model_filepath"_a, "policy_filepath"_a)
        .def_readwrite("observation", &SystemClass::System::observation)
        .def_readwrite("policy_output", &SystemClass::System::policy_output)
        .def("reset", &SystemClass::System::reset)
        .def("load_policy", &SystemClass::System::loadPolicy)
        .def("set_observation", &SystemClass::System::setObservation, "command"_a)
        .def("policy_inference", &SystemClass::System::policyInference)
        .def("get_control", &SystemClass::System::getControl)
        .def("get_state", &SystemClass::System::getState)
        .def("rollout", &SystemClass::System::rollout, "state"_a, "command"_a, "physics_substeps"_a = 2,
             "reset_last_output"_a = true, "cutoff_time"_a = SystemClass::kInfiniteTime) // Match C++
        .def("rollout_world_frame", &SystemClass::System::rollout_world_frame)
        .def("rollout_world_frame_feedback", &SystemClass::System::rollout_world_frame_feedback);

    python_module.def("threaded_rollout", &SystemClass::threadedRollout,
                      "Threaded policy rollout with shared pointers to System objects.", "systems"_a, "states"_a,
                      "command"_a, "last_policy_outputs"_a, "num_threads"_a, "physics_substeps"_a,
                      "cutoff_time"_a = SystemClass::kInfiniteTime);

    python_module.def("threaded_rollout_feedback_world_frame", &SystemClass::threadedRolloutFeedbackWorldFrame,
                      "Threaded policy rollout with proportional gain on control error in world frame.", "systems"_a,
                      "states"_a, "command"_a, "posref"_a, "p_gains"_a, "last_policy_outputs"_a, "physics_substeps"_a,
                      "num_threads"_a);

    python_module.def("create_systems_vector", &SystemClass::create_systems_vector, "Creates a vector of systems.",
                      "model_filepath"_a, "policy_filepath"_a, "num_systems"_a);
  }

} // namespace mujoco_extensions::pybind::policy_rollout
