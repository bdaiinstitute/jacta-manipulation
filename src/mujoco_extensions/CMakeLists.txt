cmake_minimum_required(VERSION 3.15)
project(mujoco_extensions LANGUAGES CXX)

# C++ settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional: Set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Fast)
endif()

# Load find_or_fetch logic
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindOrFetchDeps)

# Pull in dependencies
find_or_fetch_pybind11()
find_or_fetch_eigen()
find_or_fetch_onnxruntime()
find_or_fetch_mujoco()
find_package(OpenMP REQUIRED)

# Define the top-level include directory as the repository root.
set(MJ_EXT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# --------------------------------------------------
# === onnx_interface ===
add_library(onnx_interface STATIC
  mujoco_extensions/onnx_interface/onnx_interface.cpp
)
target_compile_options(onnx_interface PRIVATE -fPIC)
target_include_directories(onnx_interface PUBLIC ${MJ_EXT_INCLUDE_DIR})
target_link_libraries(onnx_interface PUBLIC
  Eigen3::Eigen
  onnxruntime::onnxruntime
)

# --------------------------------------------------
# === system_lib ===
add_library(system_lib STATIC
  mujoco_extensions/system/eigen_types.cpp
  mujoco_extensions/system/system_class.cpp
  mujoco_extensions/system/system_utils.cpp
)
target_compile_options(system_lib PRIVATE -fPIC)
target_include_directories(system_lib PUBLIC ${MJ_EXT_INCLUDE_DIR})
target_link_libraries(system_lib PUBLIC
  onnx_interface
  mujoco::mujoco
  yaml-cpp
)

# --------------------------------------------------
# === pybind module: _jacobian_smoothing ===
file(GLOB JACOBIAN_SOURCES mujoco_extensions/pybind/jacobian_smoothing/*.cpp)

pybind11_add_module(jacobian_smoothing_cpp ${JACOBIAN_SOURCES})
set_target_properties(jacobian_smoothing_cpp PROPERTIES OUTPUT_NAME "_jacobian_smoothing")

target_compile_options(jacobian_smoothing_cpp PRIVATE -fopenmp -march=native)
target_link_options(jacobian_smoothing_cpp PRIVATE -lgomp -lpthread)
target_link_libraries(jacobian_smoothing_cpp PRIVATE
  pybind11::module
  Eigen3::Eigen
  mujoco::mujoco
  OpenMP::OpenMP_CXX
)

# --------------------------------------------------
# === pybind module: _policy_rollout ===
pybind11_add_module(policy_rollout_cpp
  mujoco_extensions/pybind/policy_rollout/policy_rollout.cpp
  mujoco_extensions/pybind/policy_rollout/pybind.cpp
)
set_target_properties(policy_rollout_cpp PROPERTIES
    BUILD_RPATH "$ORIGIN/../onnxruntime:$ORIGIN/../../mujoco"
    INSTALL_RPATH "$ORIGIN/../onnxruntime:$ORIGIN/../../mujoco"
    OUTPUT_NAME "_policy_rollout"
)

target_compile_options(policy_rollout_cpp PRIVATE -fopenmp -march=native)
target_link_options(policy_rollout_cpp PRIVATE -lgomp -lpthread)
target_link_libraries(policy_rollout_cpp PRIVATE
  pybind11::module
  system_lib
  onnxruntime::onnxruntime
  OpenMP::OpenMP_CXX
)

# --------------------------------------------------
# === install targets for packaging ===
# Install the shared library into your package under a subdirectory "onnxruntime"
install(FILES ${ONNXRUNTIME_LIB_FILE} DESTINATION "mujoco_extensions/onnxruntime")
install(FILES ${ONNXRUNTIME_LIB_FILE1} DESTINATION "mujoco_extensions/onnxruntime")
install(TARGETS policy_rollout_cpp
        LIBRARY DESTINATION "mujoco_extensions/policy_rollout")
install(TARGETS jacobian_smoothing_cpp
        LIBRARY DESTINATION "mujoco_extensions/jacobian_smoothing")  
