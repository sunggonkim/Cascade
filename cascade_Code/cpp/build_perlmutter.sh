#!/bin/bash
#
# Build script for Cascade C++ on Perlmutter
#

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          Cascade C++ Build for Perlmutter                ║"
echo "╚══════════════════════════════════════════════════════════╝"

# Load required modules - use gcc-native/13.2 for C++17 filesystem
module load PrgEnv-gnu 2>/dev/null || true
module load gcc-native/13.2 2>/dev/null || true
module load cudatoolkit/12.4 2>/dev/null || true
module load cmake/3.24 2>/dev/null || true
module load cray-python 2>/dev/null || true

# Install pybind11 if needed
pip show pybind11 >/dev/null 2>&1 || pip install --user pybind11

# Create build directory
BUILD_DIR="build"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake - use GCC
echo ""
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPERLMUTTER=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_HOST_COMPILER=g++ \
    -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
    -DPYTHON_EXECUTABLE=$(which python)

# Build
echo ""
echo "Building..."
make -j$(nproc)

# Install to parent directory
echo ""
echo "Installing..."
cp cascade_cpp*.so ../
cp cascade_bench ../

echo ""
echo "Build complete!"
echo "  - cascade_cpp.cpython-*.so (Python module)"
echo "  - cascade_bench (C++ benchmark)"
echo ""
echo "Test with:"
echo "  python -c 'import cascade_cpp; print(cascade_cpp)'"
echo "  ./cascade_bench --blocks 1000 --size 128"
