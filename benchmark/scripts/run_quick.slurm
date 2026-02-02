#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:01:00
#SBATCH -J quick_test
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/quick_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/quick_%j.err

module load python cudatoolkit
export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/src:$PYTHONPATH

python3 << 'PYEOF'
import sys, time
import numpy as np
sys.path.insert(0, "/pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/src")

print("🚀 Quick GPU Optimization Test (1 min)")

import cupy as cp
props = cp.cuda.runtime.getDeviceProperties(0)
print(f"GPU: {props['name'].decode()}")

from cascade import compute_block_id_from_bytes
from cascade.backends import GPUBackend

gpu = GPUBackend(capacity_gb=8.0, use_sharding=True, use_streams=True)

# Test 1GB data (50 x 20MB)
print("\n📊 Testing 1GB (50 x 20MB blocks)...")
blocks = []
for i in range(50):
    data = np.random.randn(10 * 1024 * 1024).astype(np.float16).tobytes()
    blocks.append((compute_block_id_from_bytes(data), data))

total = sum(len(d) for _, d in blocks)

# Write
t0 = time.perf_counter()
for bid, data in blocks:
    gpu.put(bid, data)
w = time.perf_counter() - t0

# Read
t0 = time.perf_counter()
for bid, _ in blocks:
    gpu.get(bid)
r = time.perf_counter() - t0

print(f"Write: {total/1024**3/w:.2f} GB/s (PCIe efficiency: {total/1024**3/w/32*100:.0f}%)")
print(f"Read:  {total/1024**3/r:.2f} GB/s (PCIe efficiency: {total/1024**3/r/32*100:.0f}%)")
print("✅ Done!")
PYEOF
