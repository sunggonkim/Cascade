#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:20:00
#SBATCH -J full_6sys
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/full_6sys_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/full_6sys_%j.err

module load cudatoolkit
module load pytorch/2.6.0

export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/python_pkgs_py312:$PYTHONPATH

echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"  
echo "Start: $(date)"
echo "================================================"

# 확인
python -c "import redis; import torch; import h5py; print('All imports OK:', redis.__version__, torch.__version__, h5py.__version__, 'cuda:', torch.cuda.is_available())"

RESULTS_DIR=/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results
RESULTS_FILE=$RESULTS_DIR/full_6sys_$SLURM_JOB_ID.json

# 벤치마크 스크립트
python << 'PYTHON_EOF'
import os
import sys
import time
import json
import mmap
import numpy as np
from datetime import datetime

# 패키지 import
import redis
import torch
import h5py

print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
print(f"redis: {redis.__version__}")
print(f"h5py: {h5py.__version__}")

JOB_ID = os.environ.get('SLURM_JOB_ID', 'unknown')
BLOCK_SIZE = 256 * 1024 * 1024  # 256MB
NUM_BLOCKS = 4  # 총 1GB

results = {
    "job_id": JOB_ID,
    "timestamp": datetime.now().isoformat(),
    "config": {"block_size_mb": 256, "num_blocks": NUM_BLOCKS, "total_gb": 1.0},
    "systems": {}
}

data = np.random.bytes(BLOCK_SIZE)

# ============ 1. Cascade (SHM mmap) ============
print("\n=== Testing Cascade (SHM mmap) ===")
try:
    shm_path = "/dev/shm/cascade_bench"
    os.makedirs(shm_path, exist_ok=True)
    
    start = time.perf_counter()
    for i in range(NUM_BLOCKS):
        path = f"{shm_path}/block_{i}.bin"
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        os.ftruncate(fd, BLOCK_SIZE)
        mm = mmap.mmap(fd, BLOCK_SIZE)
        mm.write(data)
        mm.close()
        os.close(fd)
    write_time = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_BLOCKS / 1e9) / write_time
    
    start = time.perf_counter()
    for i in range(NUM_BLOCKS):
        path = f"{shm_path}/block_{i}.bin"
        fd = os.open(path, os.O_RDONLY)
        mm = mmap.mmap(fd, BLOCK_SIZE, prot=mmap.PROT_READ)
        _ = mm.read()
        mm.close()
        os.close(fd)
    read_time = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_BLOCKS / 1e9) / read_time
    
    for i in range(NUM_BLOCKS):
        os.remove(f"{shm_path}/block_{i}.bin")
    os.rmdir(shm_path)
    
    results["systems"]["Cascade"] = {"type": "SHM mmap", "write_gbps": round(write_gbps, 2), "read_gbps": round(read_gbps, 2)}
    print(f"Cascade: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
except Exception as e:
    results["systems"]["Cascade"] = {"error": str(e)}
    print(f"Cascade ERROR: {e}")

# ============ 2. vLLM (GPU HBM) ============
print("\n=== Testing vLLM (GPU HBM) ===")
try:
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        
        # numpy -> torch tensor
        np_data = np.frombuffer(data, dtype=np.float32)
        cpu_tensor = torch.from_numpy(np_data.copy())
        
        # GPU write (CPU -> GPU)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(NUM_BLOCKS):
            gpu_tensor = cpu_tensor.cuda()
        torch.cuda.synchronize()
        write_time = time.perf_counter() - start
        write_gbps = (BLOCK_SIZE * NUM_BLOCKS / 1e9) / write_time
        
        # GPU read (GPU -> CPU)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(NUM_BLOCKS):
            _ = gpu_tensor.cpu()
        torch.cuda.synchronize()
        read_time = time.perf_counter() - start
        read_gbps = (BLOCK_SIZE * NUM_BLOCKS / 1e9) / read_time
        
        del gpu_tensor
        torch.cuda.empty_cache()
        
        results["systems"]["vLLM"] = {"type": "GPU HBM", "write_gbps": round(write_gbps, 2), "read_gbps": round(read_gbps, 2)}
        print(f"vLLM: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
    else:
        results["systems"]["vLLM"] = {"error": "CUDA not available"}
        print("vLLM: CUDA not available")
except Exception as e:
    results["systems"]["vLLM"] = {"error": str(e)}
    print(f"vLLM ERROR: {e}")

# ============ 3. LMCache (File-based) ============
print("\n=== Testing LMCache (File-based) ===")
try:
    lmc_path = "/tmp/lmcache_bench"
    os.makedirs(lmc_path, exist_ok=True)
    
    start = time.perf_counter()
    for i in range(NUM_BLOCKS):
        with open(f"{lmc_path}/block_{i}.bin", 'wb') as f:
            f.write(data)
    write_time = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_BLOCKS / 1e9) / write_time
    
    start = time.perf_counter()
    for i in range(NUM_BLOCKS):
        with open(f"{lmc_path}/block_{i}.bin", 'rb') as f:
            _ = f.read()
    read_time = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_BLOCKS / 1e9) / read_time
    
    for i in range(NUM_BLOCKS):
        os.remove(f"{lmc_path}/block_{i}.bin")
    os.rmdir(lmc_path)
    
    results["systems"]["LMCache"] = {"type": "File-based", "write_gbps": round(write_gbps, 2), "read_gbps": round(read_gbps, 2)}
    print(f"LMCache: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
except Exception as e:
    results["systems"]["LMCache"] = {"error": str(e)}
    print(f"LMCache ERROR: {e}")

# ============ 4. PDC (fsync) ============
print("\n=== Testing PDC (fsync) ===")
try:
    pdc_path = "/tmp/pdc_bench"
    os.makedirs(pdc_path, exist_ok=True)
    
    start = time.perf_counter()
    for i in range(NUM_BLOCKS):
        fd = os.open(f"{pdc_path}/block_{i}.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.write(fd, data)
        os.fsync(fd)
        os.close(fd)
    write_time = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_BLOCKS / 1e9) / write_time
    
    start = time.perf_counter()
    for i in range(NUM_BLOCKS):
        with open(f"{pdc_path}/block_{i}.bin", 'rb') as f:
            _ = f.read()
    read_time = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_BLOCKS / 1e9) / read_time
    
    for i in range(NUM_BLOCKS):
        os.remove(f"{pdc_path}/block_{i}.bin")
    os.rmdir(pdc_path)
    
    results["systems"]["PDC"] = {"type": "fsync", "write_gbps": round(write_gbps, 2), "read_gbps": round(read_gbps, 2)}
    print(f"PDC: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
except Exception as e:
    results["systems"]["PDC"] = {"error": str(e)}
    print(f"PDC ERROR: {e}")

# ============ 5. Redis (In-memory dict) ============
print("\n=== Testing Redis (In-memory dict) ===")
try:
    redis_store = {}
    
    start = time.perf_counter()
    for i in range(NUM_BLOCKS):
        key = f"block_{i}"
        redis_store[key] = data
    write_time = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_BLOCKS / 1e9) / write_time
    
    start = time.perf_counter()
    for i in range(NUM_BLOCKS):
        key = f"block_{i}"
        _ = redis_store[key]
    read_time = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_BLOCKS / 1e9) / read_time
    
    results["systems"]["Redis"] = {"type": "In-memory KV", "write_gbps": round(write_gbps, 2), "read_gbps": round(read_gbps, 2), "note": "dict simulation"}
    print(f"Redis: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
except Exception as e:
    results["systems"]["Redis"] = {"error": str(e)}
    print(f"Redis ERROR: {e}")

# ============ 6. HDF5 (h5py) ============
print("\n=== Testing HDF5 (h5py) ===")
try:
    hdf5_file = "/tmp/hdf5_bench.h5"
    np_data = np.frombuffer(data, dtype=np.uint8)
    
    start = time.perf_counter()
    with h5py.File(hdf5_file, 'w') as f:
        for i in range(NUM_BLOCKS):
            f.create_dataset(f"block_{i}", data=np_data)
    write_time = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_BLOCKS / 1e9) / write_time
    
    start = time.perf_counter()
    with h5py.File(hdf5_file, 'r') as f:
        for i in range(NUM_BLOCKS):
            _ = f[f"block_{i}"][:]
    read_time = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_BLOCKS / 1e9) / read_time
    
    os.remove(hdf5_file)
    
    results["systems"]["HDF5"] = {"type": "Hierarchical", "write_gbps": round(write_gbps, 2), "read_gbps": round(read_gbps, 2)}
    print(f"HDF5: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
except Exception as e:
    results["systems"]["HDF5"] = {"error": str(e)}
    print(f"HDF5 ERROR: {e}")

# 결과 저장
results_file = os.environ.get('RESULTS_FILE', '/tmp/results.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n=== Results saved to {results_file} ===")
print(json.dumps(results, indent=2))
PYTHON_EOF

echo "================================================"
echo "End: $(date)"
echo "================================================"
