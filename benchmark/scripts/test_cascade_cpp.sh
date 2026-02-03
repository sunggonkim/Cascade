#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -J test_cpp
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/test_cpp_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/test_cpp_%j.err
#SBATCH --gpus-per-node=4

set -e

module load cudatoolkit
module load cmake
module load pytorch/2.6.0
module load gcc/11.2.0

echo "Python version:"
which python3
python3 --version

cd /pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp/build_py2

# Python 3.9로 빌드됐으므로 pytorch 환경의 python 사용
PYTHON_BIN=$(which python3)
echo "Using: $PYTHON_BIN"

# 파일 확인
ls -la *.so

# Python 3.12용으로 재빌드
echo ""
echo "Rebuilding for Python 3.12..."
cd ..
rm -rf build_py3
mkdir -p build_py3
cd build_py3

PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

# Python 3.12 경로 찾기
PYTHON_EXE=$(python3 -c "import sys; print(sys.executable)")
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

echo "Python exe: $PYTHON_EXE"
echo "Python include: $PYTHON_INCLUDE"
echo "Python lib: $PYTHON_LIB"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPERLMUTTER=ON \
    -DUSE_MPI=OFF \
    -DBUILD_PYTHON=ON \
    -DCMAKE_CXX_COMPILER=g++ \
    -Dpybind11_DIR=$PYBIND11_DIR \
    -DPYTHON_EXECUTABLE=$PYTHON_EXE \
    -DPYBIND11_FINDPYTHON=ON

make -j16 cascade_cpp

echo ""
echo "Built Python module:"
ls -la *.so

# 테스트
echo ""
echo "Testing Python import..."
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 -c "import cascade_cpp; print('cascade_cpp imported successfully!')"

# 벤치마크
echo ""
echo "===== Python->C++ Cascade Benchmark ====="
python3 << 'PYTHON_EOF'
import cascade_cpp
import numpy as np
import time

print("Cascade C++ Backend Test")
print("=" * 50)

# Config
cfg = cascade_cpp.CascadeConfig()
cfg.shm_path = "/dev/shm/cascade_bench"
cfg.shm_capacity_bytes = 50 * 1024 * 1024 * 1024  # 50GB

print(f"SHM path: {cfg.shm_path}")
print(f"SHM capacity: {cfg.shm_capacity_bytes / 1e9:.1f} GB")

# Store
store = cascade_cpp.CascadeStore(cfg)
print("CascadeStore created via C++ backend")

# Benchmark 다양한 크기
sizes = [64 * 1024 * 1024, 256 * 1024 * 1024, 512 * 1024 * 1024]
results = []

for size in sizes:
    size_mb = size // (1024 * 1024)
    print(f"\n--- {size_mb} MB Block ---")
    
    # 데이터 생성
    data = np.random.randint(0, 256, size, dtype=np.uint8)
    
    # PUT 3회 평균
    put_times = []
    for i in range(3):
        start = time.perf_counter()
        store.put(f"test_{size}_{i}", data, False)
        elapsed = time.perf_counter() - start
        put_times.append(elapsed)
    
    avg_put = sum(put_times) / len(put_times)
    put_gbps = size / 1e9 / avg_put
    print(f"PUT: {put_gbps:.2f} GB/s (avg of 3)")
    
    # GET 3회 평균
    get_times = []
    out = np.zeros(size, dtype=np.uint8)
    for i in range(3):
        start = time.perf_counter()
        found, sz = store.get(f"test_{size}_0", out)
        elapsed = time.perf_counter() - start
        get_times.append(elapsed)
    
    avg_get = sum(get_times) / len(get_times)
    get_gbps = size / 1e9 / avg_get
    print(f"GET: {get_gbps:.2f} GB/s (avg of 3)")
    
    # Verify
    if np.array_equal(data, out):
        print("Data integrity: OK")
    else:
        print("Data integrity: FAILED!")
    
    results.append({
        'size_mb': size_mb,
        'put_gbps': put_gbps,
        'get_gbps': get_gbps
    })

print("\n" + "=" * 50)
print("Summary:")
for r in results:
    print(f"  {r['size_mb']:4d} MB: PUT {r['put_gbps']:.2f} GB/s, GET {r['get_gbps']:.2f} GB/s")

print("\nStats:", store.get_stats())
PYTHON_EOF
