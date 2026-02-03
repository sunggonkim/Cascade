#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -J buildpy
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/buildpy_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/buildpy_%j.err
#SBATCH --gpus-per-node=4

set -e

# Conda 비활성화
source ~/.bashrc
conda deactivate 2>/dev/null || true
unset CONDA_PREFIX

module load cudatoolkit
module load cmake
module load pytorch/2.6.0
module load gcc/11.2.0

# Python 확인
echo "Python:"
which python3
python3 --version

cd /pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp
rm -rf build_final
mkdir -p build_final
cd build_final

PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

# pybind11에서 Python3 직접 지정
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPERLMUTTER=ON \
    -DUSE_MPI=OFF \
    -DBUILD_PYTHON=ON \
    -DCMAKE_CXX_COMPILER=g++ \
    -Dpybind11_DIR=$PYBIND11_DIR \
    -DPython3_ROOT_DIR=/global/common/software/nersc9/pytorch/2.6.0 \
    -DPython3_EXECUTABLE=/global/common/software/nersc9/pytorch/2.6.0/bin/python3

make -j16 cascade_cpp

echo ""
echo "Built:"
ls -la *.so

# Import 테스트
echo ""
echo "Import test:"
export PYTHONPATH=$(pwd):$PYTHONPATH
/global/common/software/nersc9/pytorch/2.6.0/bin/python3 -c "import cascade_cpp; print('SUCCESS!')"

# 벤치마크
echo ""
echo "===== C++ Backend via Python ====="
/global/common/software/nersc9/pytorch/2.6.0/bin/python3 << 'PYEOF'
import cascade_cpp
import numpy as np
import time
import os

os.makedirs("/dev/shm/cascade_test", exist_ok=True)

cfg = cascade_cpp.CascadeConfig()
cfg.shm_path = "/dev/shm/cascade_test"
cfg.shm_capacity_bytes = 50 * 1024**3

store = cascade_cpp.CascadeStore(cfg)
print("CascadeStore (C++) created")

for size_mb in [64, 256, 512]:
    size = size_mb * 1024 * 1024
    data = np.random.randint(0, 256, size, dtype=np.uint8)
    
    # PUT
    t0 = time.perf_counter()
    store.put(f"block_{size_mb}", data, False)
    put_time = time.perf_counter() - t0
    
    # GET
    out = np.zeros(size, dtype=np.uint8)
    t0 = time.perf_counter()
    store.get(f"block_{size_mb}", out)
    get_time = time.perf_counter() - t0
    
    print(f"{size_mb:4d}MB: PUT {size/1e9/put_time:.2f} GB/s, GET {size/1e9/get_time:.2f} GB/s")
    
    assert np.array_equal(data, out), "Data mismatch!"

print("All tests passed!")
PYEOF
