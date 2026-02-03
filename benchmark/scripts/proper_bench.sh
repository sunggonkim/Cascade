#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:20:00
#SBATCH -J proper_bench
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/proper_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/proper_%j.err
#SBATCH --gpus-per-node=4

module load cudatoolkit
module load cray-mpich
module load pytorch/2.6.0

export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/python_pkgs_py312:$PYTHONPATH
export MPICH_GPU_SUPPORT_ENABLED=1

echo "================================================"
echo "PROPER BENCHMARK: Hot/Warm/Cold Tests"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs/node: 4"
echo "Start: $(date)"
echo "================================================"

# ============ 1. Cascade C++ (실제 구현) ============
echo ""
echo "=== 1. Cascade C++ Distributed Benchmark ==="
cd /pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp/build_mpi

# 4 nodes, 4 GPUs per node = 16 total
srun --export=ALL,MPICH_GPU_SUPPORT_ENABLED=1 -n 16 -c 8 --gpus-per-node=4 \
    ./distributed_bench --blocks 500 --block-size 10 2>&1 | head -100

# ============ 2. Hot/Warm/Cold Python Tests ============
echo ""
echo "=== 2. Hot/Warm/Cold Tests ==="

python << 'PYTHON_EOF'
import os
import sys
import time
import json
import mmap
import ctypes
import numpy as np
from datetime import datetime

import torch
import h5py

# posix_fadvise for cold read
libc = ctypes.CDLL("libc.so.6")
POSIX_FADV_DONTNEED = 4

def drop_page_cache(path):
    """Drop page cache for cold read test"""
    fd = os.open(path, os.O_RDONLY)
    file_size = os.fstat(fd).st_size
    libc.posix_fadvise(fd, 0, file_size, POSIX_FADV_DONTNEED)
    os.close(fd)

print(f"torch {torch.__version__}, cuda: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

JOB_ID = os.environ.get('SLURM_JOB_ID', 'unknown')
BLOCK_SIZE = 1024 * 1024 * 1024  # 1GB per test
NUM_ITERS = 5

results = {
    "job_id": JOB_ID,
    "timestamp": datetime.now().isoformat(),
    "config": {"block_size_gb": 1.0, "num_iters": NUM_ITERS},
    "tests": {}
}

# Generate 1GB random data
print("\nGenerating 1GB test data...")
np_data = np.random.randint(0, 256, size=BLOCK_SIZE, dtype=np.uint8)
data_bytes = np_data.tobytes()

# ============ HOT: GPU VRAM ============
print("\n=== HOT: GPU VRAM (HBM) ===")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    
    # Pre-allocate GPU memory
    tensor_data = torch.from_numpy(np_data.view(np.float32).copy())
    
    # Write to GPU (CPU -> GPU)
    torch.cuda.synchronize()
    write_times = []
    for i in range(NUM_ITERS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        gpu_tensor = tensor_data.cuda()
        torch.cuda.synchronize()
        write_times.append(time.perf_counter() - start)
    
    # Read from GPU (GPU -> CPU)
    read_times = []
    for i in range(NUM_ITERS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        cpu_tensor = gpu_tensor.cpu()
        torch.cuda.synchronize()
        read_times.append(time.perf_counter() - start)
    
    write_gbps = BLOCK_SIZE / 1e9 / np.mean(write_times)
    read_gbps = BLOCK_SIZE / 1e9 / np.mean(read_times)
    
    del gpu_tensor, cpu_tensor
    torch.cuda.empty_cache()
    
    results["tests"]["HOT_GPU"] = {"write_gbps": round(write_gbps, 2), "read_gbps": round(read_gbps, 2)}
    print(f"HOT (GPU VRAM): Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
else:
    print("CUDA not available")

# ============ WARM: DRAM (/dev/shm) ============
print("\n=== WARM: DRAM (SHM) ===")
shm_path = "/dev/shm/bench_warm.bin"

# Write to SHM
write_times = []
for i in range(NUM_ITERS):
    start = time.perf_counter()
    fd = os.open(shm_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    os.ftruncate(fd, BLOCK_SIZE)
    mm = mmap.mmap(fd, BLOCK_SIZE)
    mm.write(data_bytes)
    mm.flush()
    mm.close()
    os.close(fd)
    write_times.append(time.perf_counter() - start)

# Read from SHM (hot - in DRAM)
read_times = []
for i in range(NUM_ITERS):
    start = time.perf_counter()
    fd = os.open(shm_path, os.O_RDONLY)
    mm = mmap.mmap(fd, BLOCK_SIZE, prot=mmap.PROT_READ)
    _ = mm.read()
    mm.close()
    os.close(fd)
    read_times.append(time.perf_counter() - start)

os.remove(shm_path)

write_gbps = BLOCK_SIZE / 1e9 / np.mean(write_times)
read_gbps = BLOCK_SIZE / 1e9 / np.mean(read_times)

results["tests"]["WARM_DRAM"] = {"write_gbps": round(write_gbps, 2), "read_gbps": round(read_gbps, 2)}
print(f"WARM (DRAM): Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")

# ============ COLD: Lustre ============
print("\n=== COLD: Lustre (page cache dropped) ===")
lustre_path = os.environ.get('SCRATCH', '/tmp') + f"/bench_cold_{JOB_ID}.bin"

# Write to Lustre
write_times = []
for i in range(NUM_ITERS):
    start = time.perf_counter()
    with open(lustre_path, 'wb') as f:
        f.write(data_bytes)
        f.flush()
        os.fsync(f.fileno())
    write_times.append(time.perf_counter() - start)

# Cold read (drop page cache first)
read_times = []
for i in range(NUM_ITERS):
    drop_page_cache(lustre_path)  # Drop page cache
    start = time.perf_counter()
    with open(lustre_path, 'rb') as f:
        _ = f.read()
    read_times.append(time.perf_counter() - start)

os.remove(lustre_path)

write_gbps = BLOCK_SIZE / 1e9 / np.mean(write_times)
read_gbps = BLOCK_SIZE / 1e9 / np.mean(read_times)

results["tests"]["COLD_LUSTRE"] = {"write_gbps": round(write_gbps, 2), "read_gbps": round(read_gbps, 2)}
print(f"COLD (Lustre): Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")

# ============ /tmp NVMe ============
print("\n=== NVMe (/tmp) ===")
nvme_path = "/tmp/bench_nvme.bin"

write_times = []
for i in range(NUM_ITERS):
    start = time.perf_counter()
    with open(nvme_path, 'wb') as f:
        f.write(data_bytes)
        f.flush()
        os.fsync(f.fileno())
    write_times.append(time.perf_counter() - start)

# Cold read
read_times = []
for i in range(NUM_ITERS):
    drop_page_cache(nvme_path)
    start = time.perf_counter()
    with open(nvme_path, 'rb') as f:
        _ = f.read()
    read_times.append(time.perf_counter() - start)

os.remove(nvme_path)

write_gbps = BLOCK_SIZE / 1e9 / np.mean(write_times)
read_gbps = BLOCK_SIZE / 1e9 / np.mean(read_times)

results["tests"]["NVMe"] = {"write_gbps": round(write_gbps, 2), "read_gbps": round(read_gbps, 2)}
print(f"NVMe (/tmp): Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")

# Summary
print("\n" + "="*60)
print("SUMMARY (1GB block, 5 iterations)")
print("="*60)
print(f"{'Test':<20} {'Write GB/s':>12} {'Read GB/s':>12}")
print("-"*60)
for test, vals in results["tests"].items():
    print(f"{test:<20} {vals['write_gbps']:>12.2f} {vals['read_gbps']:>12.2f}")

results_file = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/proper_{JOB_ID}.json"
os.makedirs(os.path.dirname(results_file), exist_ok=True)
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_file}")
PYTHON_EOF

echo ""
echo "================================================"
echo "End: $(date)"
echo "================================================"
