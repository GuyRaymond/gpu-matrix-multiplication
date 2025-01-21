cmake_minimum_required(VERSION 3.14)
project(GPUMatrixMultiplication LANGUAGES CXX CUDA)

# Enable CUDA and specify multiple architectures
set(CMAKE_CUDA_ARCHITECTURES 50 61 70 75 80 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Set C++ standard to 14 (CUDA may not fully support C++17)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(gpu_matrix_multiplication src/main.cu)

# Set CUDA standard
set_property(TARGET gpu_matrix_multiplication PROPERTY CUDA_STANDARD 14)
