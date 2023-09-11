cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(VLLM-ROCm LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD ${CXX_STD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# mamba env
link_directories($ENV{CONDA_PREFIX}/lib)
include_directories($ENV{CONDA_PREFIX}/include)
set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX} ${CMAKE_PREFIX_PATH})

# rocm
list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
find_package(miopen)

# build mode
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set up COMMON_HEADER_DIRS and COMMON_LIB_DIRS
set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src/csrc
)
set(COMMON_LIB_DIRS "")

# Add Python into COMMON_HEADER_DIRS & COMMON_LIB_DIRS
set(PYTHON_PATH "python" CACHE STRING "Python path")
execute_process(COMMAND ${PYTHON_PATH} "-c" "import sysconfig;
print(sysconfig.get_paths()['include']);"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE PY_INCLUDE_DIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
	message(FATAL_ERROR "Python config Error.")
endif()
list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})

# Add LibTorch into COMMON_HEADER_DIRS & COMMON_LIB_DIRS
execute_process(COMMAND ${PYTHON_PATH} "-c" "import os; import torch;
print(os.path.dirname(torch.__file__), end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE TORCH_DIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
	message(FATAL_ERROR "Torch config Error.")
endif()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
set(CAFFE2_USE_CUDNN 1)
find_package(Torch REQUIRED)
list(APPEND COMMON_HEADER_DIRS "${TORCH_INCLUDE_DIRS}")
list(APPEND COMMON_LIBS "${TORCH_LIBRARIES}")

# Let COMMON_HEADER_DIRS & COMMON_LIB_DIRS take effect
include_directories(${COMMON_HEADER_DIRS})
link_directories(${COMMON_LIB_DIRS})
link_libraries(${COMMON_LIBS})

# Should turn off CXX11 ABI if pytorch is built with CXX11 ABI off
execute_process(COMMAND ${PYTHON_PATH} "-c" "import torch;
print(torch._C._GLIBCXX_USE_CXX11_ABI,end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE USE_CXX11_ABI)
message("-- USE_CXX11_ABI=${USE_CXX11_ABI}")
if (USE_CXX11_ABI)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

# Let all executable targets go to bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(ops)
add_subdirectory(utils)

add_library(vllm_rocm SHARED pybinding.cc)
target_link_libraries(vllm_rocm PUBLIC
  ops
  utils
  ${TORCH_LIBRARIES}
  ${MIOPEN_LIBRARIES}
)

set_target_properties(vllm_rocm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
)