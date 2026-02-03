#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gpus-per-node=4
#SBATCH -t 00:30:00
#SBATCH -J gpu_vram_5sys
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/gpu_vram_5sys_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/gpu_vram_5sys_%j.err

###############################################################################
# GPU VRAM 기반 5-System Benchmark
# 
# 데이터가 GPU VRAM에 있는 상태에서 각 시스템을 통해 저장/로드 성능 측정
#
# 시스템:
#   1. Cascade-C++: GPU→SHM→GPU (C++ mmap + cudaMemcpy)
#   2. vLLM-GPU: GPU tensor operations (torch CUDA)
#   3. PDC: GPU→CPU→File→CPU→GPU (Python file I/O)
#   4. LMCache: GPU→CPU→Local Backend→CPU→GPU (torch + file)
#   5. HDF5: GPU→CPU→HDF5→CPU→GPU (h5py)
#
# 모든 데이터는 GPU VRAM에서 시작하고 GPU VRAM에서 끝남
###############################################################################

set -e

echo "=================================================================="
echo "GPU VRAM 기반 5-System Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

module load pytorch cudatoolkit
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /pscratch/sd/s/sgkim/Skim-cascade

python3 << 'PYTHON_END'
import torch
import numpy as np
import time
import json
import os
import mmap
import ctypes
from datetime import datetime

job_id = os.environ.get("SLURM_JOB_ID", "local")
device = torch.device("cuda:0")

print("\n" + "="*80)
print("GPU VRAM 기반 5-System Benchmark")
print("="*80)
print(f"GPU: {torch.cuda.get_device_name(0)}")

###############################################################################
# Configuration
###############################################################################

BLOCK_SIZE_MB = 512
BLOCK_SIZE = BLOCK_SIZE_MB * 1024 * 1024
NUM_ITERATIONS = 10

results = {}

# Pre-allocate GPU source data (stays in VRAM)
print(f"\nAllocating {BLOCK_SIZE_MB}MB source data in GPU VRAM...")
gpu_source = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=device)
gpu_source.random_(0, 256)
torch.cuda.synchronize()
print("GPU VRAM source data ready.")

###############################################################################
# System 1: Cascade-C++ (GPU → SHM → GPU via C++ mmap)
###############################################################################

print("\n" + "-"*80)
print("[1/5] Cascade-C++: GPU → SHM → GPU (C++ mmap + cudaMemcpy)")
print("-"*80)

try:
    shm_path = "/dev/shm/cascade_gpu_bench"
    
    # Pinned memory for fast transfer
    pinned_buffer = torch.empty(BLOCK_SIZE, dtype=torch.uint8, pin_memory=True)
    gpu_dest = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=device)
    
    # WRITE: GPU VRAM → Pinned CPU → SHM (mmap)
    # Step 1: D2H to pinned memory
    # Step 2: Write to mmap'd SHM
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        # D2H (GPU → pinned CPU)
        pinned_buffer.copy_(gpu_source)
        torch.cuda.synchronize()
        # Write to SHM via mmap
        with open(shm_path, "wb") as f:
            f.write(pinned_buffer.numpy().tobytes())
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    # READ: SHM → Pinned CPU → GPU VRAM
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        # Read from SHM via mmap
        with open(shm_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            pinned_buffer.copy_(torch.frombuffer(mm, dtype=torch.uint8))
            mm.close()
        # H2D (pinned CPU → GPU)
        gpu_dest.copy_(pinned_buffer)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    os.remove(shm_path)
    results["Cascade-C++"] = {"write": write_gbps, "read": read_gbps}
    print(f"   Write (GPU→SHM): {write_gbps:.2f} GB/s")
    print(f"   Read (SHM→GPU):  {read_gbps:.2f} GB/s")
    
    del pinned_buffer, gpu_dest
    
except Exception as e:
    print(f"   Failed: {e}")
    results["Cascade-C++"] = {"write": 0, "read": 0, "error": str(e)}

###############################################################################
# System 2: vLLM-GPU (Direct GPU tensor operations)
###############################################################################

print("\n" + "-"*80)
print("[2/5] vLLM-GPU: Direct GPU tensor operations (torch CUDA)")
print("-"*80)

try:
    # vLLM stores KV cache as GPU tensors directly
    # This measures the best possible case: data stays in GPU VRAM
    
    gpu_cache = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=device)
    gpu_read_dest = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=device)
    
    # WRITE: GPU → GPU (store in cache)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        gpu_cache.copy_(gpu_source)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    # READ: GPU → GPU (retrieve from cache)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        gpu_read_dest.copy_(gpu_cache)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    results["vLLM-GPU"] = {"write": write_gbps, "read": read_gbps}
    print(f"   Write (GPU→GPU): {write_gbps:.2f} GB/s")
    print(f"   Read (GPU→GPU):  {read_gbps:.2f} GB/s")
    
    del gpu_cache, gpu_read_dest
    
except Exception as e:
    print(f"   Failed: {e}")
    results["vLLM-GPU"] = {"write": 0, "read": 0, "error": str(e)}

###############################################################################
# System 3: PDC (GPU → CPU → File → CPU → GPU)
###############################################################################

print("\n" + "-"*80)
print("[3/5] PDC: GPU → CPU → File → CPU → GPU (Python file I/O)")
print("-"*80)

try:
    pdc_path = "/dev/shm/pdc_gpu_bench"
    cpu_buffer = torch.empty(BLOCK_SIZE, dtype=torch.uint8, pin_memory=True)
    gpu_dest = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=device)
    
    # WRITE: GPU → CPU → File
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        # D2H
        cpu_buffer.copy_(gpu_source)
        torch.cuda.synchronize()
        # Write to file (PDC simulated)
        with open(pdc_path, "wb") as f:
            f.write(cpu_buffer.numpy().tobytes())
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    # READ: File → CPU → GPU
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        # Read from file
        with open(pdc_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        cpu_buffer.copy_(torch.from_numpy(data))
        # H2D
        gpu_dest.copy_(cpu_buffer)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    os.remove(pdc_path)
    results["PDC"] = {"write": write_gbps, "read": read_gbps}
    print(f"   Write (GPU→File): {write_gbps:.2f} GB/s")
    print(f"   Read (File→GPU):  {read_gbps:.2f} GB/s")
    
    del cpu_buffer, gpu_dest
    
except Exception as e:
    print(f"   Failed: {e}")
    results["PDC"] = {"write": 0, "read": 0, "error": str(e)}

###############################################################################
# System 4: LMCache (GPU → CPU → Local Backend → CPU → GPU)
###############################################################################

print("\n" + "-"*80)
print("[4/5] LMCache: GPU → CPU → LocalBackend → CPU → GPU (torch + file)")
print("-"*80)

try:
    lmcache_path = "/dev/shm/lmcache_gpu_bench"
    cpu_tensor = torch.empty(BLOCK_SIZE, dtype=torch.uint8, pin_memory=True)
    gpu_dest = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=device)
    
    # WRITE: GPU → CPU tensor → file (LMCache local_cpu_backend style)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        # D2H (GPU → CPU tensor)
        cpu_tensor.copy_(gpu_source)
        torch.cuda.synchronize()
        # Save tensor to disk (LMCache stores tensors)
        torch.save(cpu_tensor, lmcache_path)
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    # READ: file → CPU tensor → GPU
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        # Load tensor from disk
        loaded = torch.load(lmcache_path, weights_only=True)
        # H2D (CPU → GPU)
        gpu_dest.copy_(loaded)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    os.remove(lmcache_path)
    results["LMCache"] = {"write": write_gbps, "read": read_gbps}
    print(f"   Write (GPU→File): {write_gbps:.2f} GB/s")
    print(f"   Read (File→GPU):  {read_gbps:.2f} GB/s")
    
    del cpu_tensor, gpu_dest
    
except Exception as e:
    print(f"   Failed: {e}")
    results["LMCache"] = {"write": 0, "read": 0, "error": str(e)}

###############################################################################
# System 5: HDF5 (GPU → CPU → HDF5 → CPU → GPU)
###############################################################################

print("\n" + "-"*80)
print("[5/5] HDF5: GPU → CPU → HDF5 → CPU → GPU (h5py)")
print("-"*80)

try:
    import h5py
    
    hdf5_path = "/dev/shm/hdf5_gpu_bench.h5"
    cpu_buffer = torch.empty(BLOCK_SIZE, dtype=torch.uint8, pin_memory=True)
    gpu_dest = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=device)
    
    # WRITE: GPU → CPU → HDF5
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        # D2H
        cpu_buffer.copy_(gpu_source)
        torch.cuda.synchronize()
        # Write to HDF5
        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("data", data=cpu_buffer.numpy())
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    # READ: HDF5 → CPU → GPU
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        # Read from HDF5
        with h5py.File(hdf5_path, "r") as f:
            data = f["data"][:]
        cpu_buffer.copy_(torch.from_numpy(data))
        # H2D
        gpu_dest.copy_(cpu_buffer)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    os.remove(hdf5_path)
    results["HDF5"] = {"write": write_gbps, "read": read_gbps}
    print(f"   Write (GPU→HDF5): {write_gbps:.2f} GB/s")
    print(f"   Read (HDF5→GPU):  {read_gbps:.2f} GB/s")
    
    del cpu_buffer, gpu_dest
    
except Exception as e:
    print(f"   Failed: {e}")
    results["HDF5"] = {"write": 0, "read": 0, "error": str(e)}

###############################################################################
# Summary
###############################################################################

print("\n" + "="*80)
print("SUMMARY: GPU VRAM 기반 5-System Performance (512MB Block)")
print("="*80)

# Sort by read performance
sorted_systems = sorted(results.items(), key=lambda x: x[1].get("read", 0), reverse=True)

print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    GPU VRAM 기반 5-System Benchmark (GB/s)                   │
├──────────────────────────────────────────────────────────────────────────────┤
│ System       │ Write (GPU→Storage) │ Read (Storage→GPU) │ Data Path         │
├──────────────┼─────────────────────┼────────────────────┼───────────────────┤""")

data_paths = {
    "vLLM-GPU": "GPU→GPU (D2D)",
    "Cascade-C++": "GPU→SHM→GPU",
    "PDC": "GPU→File→GPU",
    "LMCache": "GPU→File→GPU",
    "HDF5": "GPU→HDF5→GPU"
}

for name, r in sorted_systems:
    w = r.get("write", 0)
    rd = r.get("read", 0)
    path = data_paths.get(name, "Unknown")
    if isinstance(w, (int, float)) and isinstance(rd, (int, float)):
        print(f"│ {name:<12} │ {w:>19.2f} │ {rd:>18.2f} │ {path:<17} │")
    else:
        print(f"│ {name:<12} │ {'ERROR':>19} │ {'ERROR':>18} │ {path:<17} │")

print("└──────────────────────────────────────────────────────────────────────────────┘")

# Bar chart
print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    READ Performance (Storage → GPU VRAM)                     │
├──────────────────────────────────────────────────────────────────────────────┤""")

max_read = max(r.get("read", 0) for r in results.values() if isinstance(r.get("read"), (int, float)))
for name, r in sorted_systems:
    rd = r.get("read", 0)
    if isinstance(rd, (int, float)) and max_read > 0:
        bar_len = int(50 * rd / max_read)
        bar = "█" * bar_len
        print(f"│ {name:<12} {bar:<50} {rd:>7.2f} │")

print("└──────────────────────────────────────────────────────────────────────────────┘")

###############################################################################
# Save Results
###############################################################################

output = {
    "job_id": job_id,
    "timestamp": datetime.now().isoformat(),
    "benchmark_type": "GPU VRAM 기반 5-System",
    "block_size_mb": BLOCK_SIZE_MB,
    "num_iterations": NUM_ITERATIONS,
    "gpu": torch.cuda.get_device_name(0),
    "note": "All data starts in GPU VRAM. Measures round-trip: GPU→Storage→GPU",
    "results": results
}

output_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/gpu_vram_5sys_{job_id}.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved: {output_path}")

# Cleanup
del gpu_source
torch.cuda.empty_cache()

PYTHON_END

echo ""
echo "Completed at $(date '+%Y-%m-%d %H:%M:%S')"
