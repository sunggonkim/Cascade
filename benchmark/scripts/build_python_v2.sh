#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J build_py2
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/build_py2_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/build_py2_%j.err
#SBATCH --gpus-per-node=4

set -e

module load cudatoolkit
module load cmake
module load pytorch/2.6.0
module load gcc/11.2.0  # C++17 filesystem 지원

echo "GCC version:"
g++ --version

cd /pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp

rm -rf build_py2
mkdir -p build_py2
cd build_py2

PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
echo "pybind11 dir: $PYBIND11_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPERLMUTTER=ON \
    -DUSE_MPI=OFF \
    -DBUILD_PYTHON=ON \
    -DCMAKE_CXX_COMPILER=g++ \
    -Dpybind11_DIR=$PYBIND11_DIR

make -j16 cascade_cpp

echo ""
echo "Built Python module:"
ls -la *.so

# 테스트
echo ""
echo "Testing Python import..."
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 -c "import cascade_cpp; print('cascade_cpp imported!')"

# 벤치마크
echo ""
echo "Benchmarking C++ backend via Python..."
python3 << 'PYTHON_EOF'
import cascade_cpp
import numpy as np
import time

# Config
cfg = cascade_cpp.CascadeConfig()
cfg.shm_path = "/dev/shm/cascade_bench"
cfg.shm_capacity_bytes = 10 * 1024 * 1024 * 1024  # 10GB
cfg.use_gpu = True

print(f"Config: shm={cfg.shm_path}, gpu={cfg.use_gpu}")

# Store
store = cascade_cpp.CascadeStore(cfg)
print("Store created.")

# Benchmark 512MB blocks
sizes = [512 * 1024 * 1024]

for size in sizes:
    print(f"\n=== {size // (1024*1024)}MB Block ===")
    data = np.random.randint(0, 256, size, dtype=np.uint8)
    
    # Put
    start = time.perf_counter()
    store.put(f"bench_{size}", data, False)
    elapsed = time.perf_counter() - start
    print(f"Put: {size/1e9/elapsed:.2f} GB/s")
    
    # Get
    out = np.zeros(size, dtype=np.uint8)
    start = time.perf_counter()
    found, sz = store.get(f"bench_{size}", out)
    elapsed = time.perf_counter() - start
    print(f"Get: {size/1e9/elapsed:.2f} GB/s (found={found})")
    
    # Verify
    if np.array_equal(data, out):
        print("Data verified OK!")
    else:
        print("ERROR: Data mismatch!")

print("\nStats:", store.get_stats())
PYTHON_EOF

echo ""
echo "Done!"
