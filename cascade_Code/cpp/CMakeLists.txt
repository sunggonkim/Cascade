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

# Optional: pybind11 for Python bindings
option(BUILD_PYTHON "Build Python bindings" OFF)
if(BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
endif()

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

# Python module (optional)
if(BUILD_PYTHON)
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
endif()

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

# ============================================================================
# MPI Distributed Backend
# ============================================================================
option(USE_MPI "Enable MPI for distributed multi-node" ON)

if(USE_MPI)
    find_package(MPI REQUIRED)
    include_directories(${MPI_CXX_INCLUDE_DIRS})
    
    # Distributed library (MPI + CUDA)
    add_library(cascade_distributed STATIC
        src/cascade_core.cpp
        src/gpu_backend.cu
        src/distributed_backend.cpp
    )
    target_compile_definitions(cascade_distributed PUBLIC USE_MPI)
    target_link_libraries(cascade_distributed 
        CUDA::cudart
        OpenSSL::Crypto
        MPI::MPI_CXX
        pthread
        OpenMP::OpenMP_CXX
    )
    set_target_properties(cascade_distributed PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    # Distributed benchmark executable
    add_executable(distributed_bench src/distributed_benchmark.cpp)
    target_compile_definitions(distributed_bench PUBLIC USE_MPI)
    target_link_libraries(distributed_bench 
        cascade_distributed
        CUDA::cudart
        OpenSSL::Crypto
        MPI::MPI_CXX
        pthread
        OpenMP::OpenMP_CXX
    )
    
    message(STATUS "MPI enabled: ${MPI_CXX_COMPILER}")
    message(STATUS "MPI include: ${MPI_CXX_INCLUDE_DIRS}")
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenSSL_INCLUDE_DIR: ${OPENSSL_INCLUDE_DIR}")

# ============================================================================
# Full System Benchmark (GPU, SHM, NVMe, Lustre)
# ============================================================================
add_executable(full_bench src/full_benchmark.cpp)
target_link_libraries(full_bench 
    CUDA::cudart
    pthread
    OpenMP::OpenMP_CXX
)
set_target_properties(full_bench PROPERTIES CUDA_ARCHITECTURES 80)

# ============================================================================
# Fair Tier Benchmark (True Hot D2D, SHM, Lustre)
# ============================================================================
add_executable(fair_tier_bench src/fair_tier_bench.cpp)
target_link_libraries(fair_tier_bench 
    CUDA::cudart
    pthread
    OpenMP::OpenMP_CXX
)
set_target_properties(fair_tier_bench PROPERTIES CUDA_ARCHITECTURES 80)
