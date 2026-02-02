#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -J cascade_gpu
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/gpu_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/gpu_%j.err

module load python cudatoolkit

export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/src:$PYTHONPATH

echo "🚀 Cascade GPU Benchmark"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python3 << 'PYTHON_EOF'
import sys
import time
import numpy as np
sys.path.insert(0, "/pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/src")

print("\n📊 GPU Backend Direct Test")

# Check CuPy
try:
    import cupy as cp
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    gpu_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    print(f"✅ GPU: {gpu_name}")
    print(f"   Memory: {props['totalGlobalMem'] / 1024**3:.1f} GB")
    GPU_OK = True
except Exception as e:
    print(f"❌ CuPy error: {e}")
    GPU_OK = False

if not GPU_OK:
    print("Skipping GPU test")
    sys.exit(0)

from cascade import CascadeStore, CascadeConfig, compute_block_id_from_bytes
from cascade.backends import GPUBackend

# Test GPU backend directly
print("\n📦 Testing GPUBackend directly...")
gpu_backend = GPUBackend(capacity_gb=8.0)

# Create FP16 test data (typical KV cache)
data_size_mb = 80
test_data = np.random.randn(data_size_mb * 1024 * 1024 // 2).astype(np.float16)
test_bytes = test_data.tobytes()
block_id = compute_block_id_from_bytes(test_bytes)

print(f"   Block size: {len(test_bytes) / 1024 / 1024:.1f} MB")

# Write test
t0 = time.perf_counter()
gpu_backend.put(block_id, test_bytes)
write_time = time.perf_counter() - t0
print(f"   Write: {len(test_bytes) / 1024**3 / write_time:.2f} GB/s")

# Read test
t0 = time.perf_counter()
result = gpu_backend.get(block_id)
read_time = time.perf_counter() - t0
print(f"   Read: {len(test_bytes) / 1024**3 / read_time:.2f} GB/s")

# Verify
if result == test_bytes:
    print("   ✅ Data integrity OK")
else:
    print("   ❌ Data mismatch!")

# Multiple blocks test
print("\n📊 Multi-block throughput test (10 x 80MB)...")
blocks = []
for i in range(10):
    data = np.random.randn(data_size_mb * 1024 * 1024 // 2).astype(np.float16).tobytes()
    bid = compute_block_id_from_bytes(data)
    blocks.append((bid, data))

# Write all
t0 = time.perf_counter()
for bid, data in blocks:
    gpu_backend.put(bid, data)
total_write = time.perf_counter() - t0
total_bytes = sum(len(d) for _, d in blocks)
print(f"   Write: {total_bytes / 1024**3 / total_write:.2f} GB/s")

# Read all
t0 = time.perf_counter()
for bid, _ in blocks:
    gpu_backend.get(bid)
total_read = time.perf_counter() - t0
print(f"   Read: {total_bytes / 1024**3 / total_read:.2f} GB/s")

print("\n✅ GPU Backend test complete!")
PYTHON_EOF

echo "Job completed at $(date)"
