#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -J cascade_cpp
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/cascade_cpp_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/cascade_cpp_%j.err
#SBATCH --gpus-per-node=4

echo "Starting build..."
date

# 환경설정
export PATH=/global/common/software/nersc9/pytorch/2.6.0/bin:$PATH
unset CONDA_PREFIX CONDA_DEFAULT_ENV

module load cudatoolkit
module load cmake
module load gcc/11.2.0

echo ""
echo "Environment:"
which python3
python3 --version
which g++
g++ --version | head -2

cd /pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp
rm -rf build_cascade_cpp
mkdir -p build_cascade_cpp
cd build_cascade_cpp

echo ""
echo "CMake configure..."
PYBIND11_DIR=$(/global/common/software/nersc9/pytorch/2.6.0/bin/python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPERLMUTTER=ON \
    -DUSE_MPI=OFF \
    -DBUILD_PYTHON=ON \
    -DCMAKE_CXX_COMPILER=g++ \
    -Dpybind11_DIR=$PYBIND11_DIR \
    -DPython_EXECUTABLE=/global/common/software/nersc9/pytorch/2.6.0/bin/python3

echo ""
echo "Building..."
make -j16 cascade_cpp 2>&1

echo ""
echo "Build output:"
ls -la *.so 2>/dev/null || echo "No .so files found"

if [ -f cascade_cpp*.so ]; then
    echo ""
    echo "Testing import..."
    export PYTHONPATH=$(pwd):$PYTHONPATH
    /global/common/software/nersc9/pytorch/2.6.0/bin/python3 -c "
import cascade_cpp
print('cascade_cpp imported!')
print('Available:', dir(cascade_cpp))
"

    echo ""
    echo "Running benchmark..."
    /global/common/software/nersc9/pytorch/2.6.0/bin/python3 << 'PYEOF'
import cascade_cpp
import numpy as np
import time
import os

os.makedirs("/dev/shm/cascade_bench", exist_ok=True)

cfg = cascade_cpp.CascadeConfig()
cfg.shm_path = "/dev/shm/cascade_bench"
cfg.shm_capacity_bytes = 50 * 1024**3

store = cascade_cpp.CascadeStore(cfg)
print("CascadeStore (C++ backend) created")

print("\n=== Python -> C++ Cascade Benchmark ===")
for size_mb in [64, 256, 512]:
    size = size_mb * 1024 * 1024
    data = np.random.randint(0, 256, size, dtype=np.uint8)
    
    # PUT avg of 3
    put_times = []
    for i in range(3):
        t0 = time.perf_counter()
        store.put(f"b{size_mb}_{i}", data, False)
        put_times.append(time.perf_counter() - t0)
    
    # GET avg of 3
    get_times = []
    out = np.zeros(size, dtype=np.uint8)
    for i in range(3):
        t0 = time.perf_counter()
        store.get(f"b{size_mb}_0", out)
        get_times.append(time.perf_counter() - t0)
    
    put_gbps = size / 1e9 / (sum(put_times)/3)
    get_gbps = size / 1e9 / (sum(get_times)/3)
    
    print(f"{size_mb:4d}MB: PUT {put_gbps:6.2f} GB/s, GET {get_gbps:6.2f} GB/s")
    
    assert np.array_equal(data, out), "Data mismatch!"

print("\nAll tests passed!")
print("Stats:", store.get_stats())
PYEOF
fi

echo ""
echo "Done!"
date
