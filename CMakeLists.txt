cmake_minimum_required(VERSION 3.18)

project(GPUResourceManager LANGUAGES CXX CUDA)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD 17)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(resource_manager_daemon
        resource_manager/daemon.cu
        resource_manager/resource_manager.cu
)

find_library(CUDA_CUDART_LIBRARY cudart
        HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(resource_manager_daemon PRIVATE
        MPI::MPI_CXX
        ${CUDA_CUDART_LIBRARY}
)

add_executable(gpu_alloc
        resource_manager/gpu_alloc.cpp
)

target_link_libraries(gpu_alloc PRIVATE
        MPI::MPI_CXX
)

message(STATUS "MPI C++ compiler: ${MPI_CXX_COMPILER}")
message(STATUS "CUDA compiler: ${CUDAToolkit_NVCC_EXECUTABLE}")
message(STATUS "To build the project, run:")
message(STATUS "  mkdir build")
message(STATUS "  cd build")
message(STATUS "  cmake ..")
message(STATUS "  make")
