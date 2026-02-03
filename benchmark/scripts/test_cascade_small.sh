#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -J test_small
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/test_small_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/test_small_%j.err
#SBATCH --gpus-per-node=4

export PATH=/global/common/software/nersc9/pytorch/2.6.0/bin:$PATH
module load cudatoolkit gcc/11.2.0

cd /pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp/build_cascade_cpp
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Testing with small SHM size..."
/global/common/software/nersc9/pytorch/2.6.0/bin/python3 << 'PYEOF'
import cascade_cpp
import numpy as np
import time
import os

print("Creating config with 1GB SHM...")
cfg = cascade_cpp.CascadeConfig()
cfg.shm_path = "/dev/shm/cascade_test_small"
cfg.shm_capacity_bytes = 1 * 1024 * 1024 * 1024  # 1GB only
cfg.use_gpu = False

print(f"Config created: shm_path={cfg.shm_path}, capacity={cfg.shm_capacity_bytes}")

print("Creating CascadeStore...")
try:
    store = cascade_cpp.CascadeStore(cfg)
    print("CascadeStore created successfully!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting with 10MB block...")
data = np.random.randint(0, 256, 10 * 1024 * 1024, dtype=np.uint8)
print(f"Data shape: {data.shape}, dtype: {data.dtype}")

t0 = time.perf_counter()
store.put("test_block", data, False)
put_time = time.perf_counter() - t0
print(f"PUT: {10 / put_time:.2f} MB/s")

out = np.zeros_like(data)
t0 = time.perf_counter()
found, size = store.get("test_block", out)
get_time = time.perf_counter() - t0
print(f"GET: {10 / get_time:.2f} MB/s (found={found}, size={size})")

if np.array_equal(data, out):
    print("Data verified!")
else:
    print("ERROR: Data mismatch!")

print("\nStats:", store.get_stats())
PYEOF
