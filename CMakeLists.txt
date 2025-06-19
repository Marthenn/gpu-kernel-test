cmake_minimum_required(VERSION 3.31)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 86)

project(gpu_kernel_test CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(gpu_kernel_test main.cu)

set_target_properties(gpu_kernel_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)