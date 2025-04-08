include(FetchContent)
include(CMakePrintHelpers)  # Optional: for debugging

# ---- PYBIND11 ----
function(find_or_fetch_pybind11)
    find_package(pybind11 QUIET)
    if (NOT pybind11_FOUND)
        message(STATUS "pybind11 not found, fetching via FetchContent...")
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endfunction()

# ---- EIGEN ----
function(find_or_fetch_eigen)
    find_package(Eigen3 QUIET)
    if (NOT Eigen3_FOUND)
        message(STATUS "Eigen3 not found, fetching via FetchContent...")
        FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG        3.4.0
        )
        FetchContent_MakeAvailable(eigen)
        set(EIGEN3_INCLUDE_DIR ${eigen_SOURCE_DIR})

        # Only create the target if it does not exist
        if (NOT TARGET Eigen3::Eigen)
            add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
            set_target_properties(Eigen3::Eigen PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR}
            )
        endif()
    endif()
endfunction()

# ---- ONNX Runtime ----
# Allow users to override the onnxruntime version.
if(NOT DEFINED ONNXRUNTIME_VERSION)
  set(ONNXRUNTIME_VERSION "1.19.0")
endif()

function(find_or_fetch_onnxruntime)
    find_package(onnxruntime QUIET)
    if (NOT onnxruntime_FOUND)
        message(STATUS "onnxruntime not found, fetching prebuilt release...")
        FetchContent_Declare(
            onnxruntime
            URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
        )
        FetchContent_MakeAvailable(onnxruntime)
        # Set the variable to point to the versioned library file.
        set(ONNXRUNTIME_LIB_FILE "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION}" CACHE INTERNAL "Path to onnxruntime library")
        set(ONNXRUNTIME_LIB_FILE1 "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so.1" CACHE INTERNAL "Path to onnxruntime library")
        add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so"
            INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
        )
    endif()
endfunction()

# ---- MUJOCO ----
# Allow users to override the MuJoCo version.
if(NOT DEFINED MUJOCO_VERSION)
  set(MUJOCO_VERSION "3.3.0")
endif()

# Build the archive name and URL using the version variable.
set(MUJOCO_ARCHIVE_NAME "mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz")
set(MUJOCO_URL "https://github.com/deepmind/mujoco/releases/download/${MUJOCO_VERSION}/${MUJOCO_ARCHIVE_NAME}")

function(find_or_fetch_mujoco)
    find_package(mujoco QUIET)
    if (NOT mujoco_FOUND)
        message(STATUS "MuJoCo not found, fetching release ${MUJOCO_VERSION}...")
        FetchContent_Declare(
            mujoco
            URL ${MUJOCO_URL}
        )
        FetchContent_MakeAvailable(mujoco)

        # Determine the root directory of the extracted content.
        if(EXISTS "${mujoco_SOURCE_DIR}/mujoco-${MUJOCO_VERSION}")
            set(MUJOCO_ROOT "${mujoco_SOURCE_DIR}/mujoco-${MUJOCO_VERSION}")
        elseif(EXISTS "${mujoco_SOURCE_DIR}/include")
            set(MUJOCO_ROOT "${mujoco_SOURCE_DIR}")
        else()
            message(FATAL_ERROR "Could not locate MuJoCo include directory in fetched content.")
        endif()

        # Determine where libmujoco.so is located.
        if(EXISTS "${MUJOCO_ROOT}/bin/libmujoco.so")
            set(MUJOCO_LIB_PATH "${MUJOCO_ROOT}/bin/libmujoco.so")
        elseif(EXISTS "${MUJOCO_ROOT}/lib/libmujoco.so")
            set(MUJOCO_LIB_PATH "${MUJOCO_ROOT}/lib/libmujoco.so")
        else()
            message(FATAL_ERROR "Could not locate libmujoco.so in fetched MuJoCo release.")
        endif()

        add_library(mujoco::mujoco UNKNOWN IMPORTED)
        set_target_properties(mujoco::mujoco PROPERTIES
            IMPORTED_LOCATION "${MUJOCO_LIB_PATH}"
            INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_ROOT}/include"
        )
    endif()
endfunction()

# ---- YAML-CPP ----
function(find_or_fetch_yamlcpp)
    find_package(yaml-cpp QUIET)
    if (NOT yaml-cpp_FOUND)
        message(STATUS "yaml-cpp not found, fetching via FetchContent...")
        FetchContent_Declare(
            yaml-cpp
            GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
            GIT_TAG        0.8.0  # for some reason, tag for 0.8.0 has no yaml-cpp prefix
        )
        FetchContent_MakeAvailable(yaml-cpp)
    endif()
endfunction()
