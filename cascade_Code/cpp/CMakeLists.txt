cmake_minimum_required(VERSION 3.18)
project(cascade_cpp LANGUAGES CXX CUDA)

# C++17 for std::optional, std::shared_mutex, std::filesystem
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Optimization flags with OpenMP
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG ${OpenMP_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Xcompiler=${OpenMP_CXX_FLAGS}")

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenSSL_INCLUDE_DIR})

# Cascade library sources
set(CASCADE_SOURCES
    src/cascade_core.cpp
    src/gpu_backend.cu
)

# Static library for linking
add_library(cascade_static STATIC ${CASCADE_SOURCES})
target_link_libraries(cascade_static 
    CUDA::cudart
    OpenSSL::Crypto
    pthread
)
set_target_properties(cascade_static PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Python module - include CUDA source directly to avoid fatbin linking issues
pybind11_add_module(cascade_cpp 
    python/bindings.cpp
    src/cascade_core.cpp
    src/gpu_backend.cu
)
target_link_libraries(cascade_cpp PRIVATE 
    CUDA::cudart
    OpenSSL::Crypto
    OpenMP::OpenMP_CXX
)
set_target_properties(cascade_cpp PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Installation
install(TARGETS cascade_cpp LIBRARY DESTINATION .)

# ============================================================================
# Benchmark executable
# ============================================================================
add_executable(cascade_bench src/benchmark.cpp)
target_link_libraries(cascade_bench 
    cascade_static
    CUDA::cudart
    OpenSSL::Crypto
    pthread
    OpenMP::OpenMP_CXX
)

# Pure CUDA memcpy benchmark
add_executable(pure_memcpy_bench src/pure_memcpy_bench.cu)
target_link_libraries(pure_memcpy_bench CUDA::cudart)
set_target_properties(pure_memcpy_bench PROPERTIES CUDA_ARCHITECTURES 80)

# ============================================================================
# Perlmutter-specific build options
# ============================================================================
option(PERLMUTTER "Build for NERSC Perlmutter" ON)

if(PERLMUTTER)
    # A100 GPU architecture
    set(CMAKE_CUDA_ARCHITECTURES 80)
    
    # Enable Lustre stripe alignment
    add_definitions(-DLARGE_STRIPE_ALIGN)
    
    # Use NVIDIA HPC SDK if available
    if(DEFINED ENV{NVIDIA_PATH})
        set(CMAKE_CUDA_COMPILER "$ENV{NVIDIA_PATH}/cuda/bin/nvcc")
    endif()
endif()

# ============================================================================
# Optional: io_uring support for async Lustre I/O
# ============================================================================
option(USE_IOURING "Enable io_uring for async I/O" OFF)

if(USE_IOURING)
    find_library(URING_LIB uring)
    if(URING_LIB)
        target_link_libraries(cascade_static ${URING_LIB})
        add_definitions(-DUSE_IOURING)
    endif()
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenSSL_INCLUDE_DIR: ${OPENSSL_INCLUDE_DIR}")
