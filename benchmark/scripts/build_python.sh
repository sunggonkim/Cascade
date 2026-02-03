#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J build_py
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/build_py_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/build_py_%j.err
#SBATCH --gpus-per-node=4

set -e

module load cudatoolkit
module load cmake
module load pytorch/2.6.0

# pybind11 설치
pip install --user pybind11

cd /pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp

rm -rf build_py
mkdir -p build_py
cd build_py

# pybind11 경로 찾기
PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
echo "pybind11 dir: $PYBIND11_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPERLMUTTER=ON \
    -DUSE_MPI=OFF \
    -DBUILD_PYTHON=ON \
    -Dpybind11_DIR=$PYBIND11_DIR

make -j16 cascade_cpp

echo ""
echo "Built Python module:"
ls -la *.so

# 테스트
echo ""
echo "Testing Python import..."
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 -c "import cascade_cpp; print('cascade_cpp imported successfully!')"

echo ""
echo "Testing basic operations..."
python3 << 'PYTHON_EOF'
import cascade_cpp
import numpy as np
import time

# Config
cfg = cascade_cpp.CascadeConfig()
cfg.shm_path = "/dev/shm/cascade_test"
cfg.shm_capacity_bytes = 1024 * 1024 * 1024  # 1GB

print(f"Config: shm_path={cfg.shm_path}, shm_capacity={cfg.shm_capacity_bytes/1e9:.1f}GB")

# Store
print("Creating CascadeStore...")
store = cascade_cpp.CascadeStore(cfg)
print("Store created!")

# Test put/get
size = 128 * 1024 * 1024  # 128MB
data = np.random.randint(0, 256, size, dtype=np.uint8)

print(f"\nPut 128MB block...")
start = time.perf_counter()
store.put("test_block", data, False)
elapsed = time.perf_counter() - start
print(f"Put: {size/1e9/elapsed:.2f} GB/s")

print(f"\nGet 128MB block...")
out = np.zeros(size, dtype=np.uint8)
start = time.perf_counter()
found, sz = store.get("test_block", out)
elapsed = time.perf_counter() - start
print(f"Get: {size/1e9/elapsed:.2f} GB/s, found={found}, size={sz}")

# Verify
if np.array_equal(data, out):
    print("Data verified OK!")
else:
    print("ERROR: Data mismatch!")

print("\nStats:", store.get_stats())
PYTHON_EOF

echo ""
echo "Python binding test complete!"
