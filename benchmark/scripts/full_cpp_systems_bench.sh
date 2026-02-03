#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:25:00
#SBATCH -J full_bench
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/full_bench_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/full_bench_%j.err
#SBATCH --gpus-per-node=4

export PATH=/global/common/software/nersc9/pytorch/2.6.0/bin:$PATH
module load cudatoolkit gcc/11.2.0

echo "===== Full Systems Benchmark ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
date

# Setup PYTHONPATH for cascade_cpp
export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp/build_cascade_cpp:$PYTHONPATH

/global/common/software/nersc9/pytorch/2.6.0/bin/python3 << 'PYEOF'
#!/usr/bin/env python3
"""
Full Systems Benchmark - Real Backends
Tests: Cascade (C++), HDF5, Python mmap, Lustre file I/O
"""

import numpy as np
import time
import json
import os
import h5py
import mmap
import ctypes
from datetime import datetime

SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
print(f"Job ID: {SLURM_JOB_ID}")

# ============================================================================
# 1. Cascade C++ Backend
# ============================================================================
class CascadeCppBackend:
    def __init__(self, path, capacity_gb=10):
        import cascade_cpp
        cfg = cascade_cpp.CascadeConfig()
        cfg.shm_path = path
        cfg.shm_capacity_bytes = capacity_gb * 1024**3
        cfg.use_gpu = False
        self.store = cascade_cpp.CascadeStore(cfg)
        self.name = "Cascade-C++"
        
    def put(self, key, data):
        return self.store.put(key, data, False)
    
    def get(self, key, out):
        return self.store.get(key, out)

# ============================================================================
# 2. HDF5 (h5py) - Compression OFF
# ============================================================================
class HDF5Backend:
    def __init__(self, path):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.name = "HDF5"
        
    def put(self, key, data):
        filepath = os.path.join(self.path, f"{key}.h5")
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('data', data=data, compression=None)
        return True
    
    def get(self, key, out):
        filepath = os.path.join(self.path, f"{key}.h5")
        with h5py.File(filepath, 'r') as f:
            out[:] = f['data'][:]
        return True, len(out)

# ============================================================================
# 3. Python mmap (baseline)
# ============================================================================
class PythonMmapBackend:
    def __init__(self, path, capacity_gb=10):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.name = "Python-mmap"
        self.blocks = {}
        
    def put(self, key, data):
        filepath = os.path.join(self.path, f"{key}.bin")
        with open(filepath, 'wb') as f:
            f.write(data.tobytes())
        self.blocks[key] = len(data)
        return True
    
    def get(self, key, out):
        filepath = os.path.join(self.path, f"{key}.bin")
        with open(filepath, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            out[:] = np.frombuffer(mm, dtype=np.uint8)
            mm.close()
        return True, len(out)

# ============================================================================
# 4. Lustre Direct I/O
# ============================================================================
class LustreBackend:
    def __init__(self, path):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.name = "Lustre"
        
    def put(self, key, data):
        filepath = os.path.join(self.path, f"{key}.bin")
        with open(filepath, 'wb') as f:
            f.write(data.tobytes())
        return True
    
    def get(self, key, out):
        filepath = os.path.join(self.path, f"{key}.bin")
        with open(filepath, 'rb') as f:
            data = f.read()
        out[:] = np.frombuffer(data, dtype=np.uint8)
        return True, len(out)
    
    def drop_cache(self, key):
        """Drop page cache for cold read test"""
        filepath = os.path.join(self.path, f"{key}.bin")
        try:
            fd = os.open(filepath, os.O_RDONLY)
            size = os.fstat(fd).st_size
            libc = ctypes.CDLL("libc.so.6")
            libc.posix_fadvise(fd, 0, size, 4)  # POSIX_FADV_DONTNEED
            os.close(fd)
        except:
            pass

# ============================================================================
# Benchmark Runner
# ============================================================================
def benchmark_system(backend, block_sizes_mb, iterations=5):
    results = []
    
    for size_mb in block_sizes_mb:
        size = size_mb * 1024 * 1024
        data = np.random.randint(0, 256, size, dtype=np.uint8)
        
        # PUT
        put_times = []
        for i in range(iterations):
            key = f"block_{size_mb}_{i}"
            t0 = time.perf_counter()
            backend.put(key, data)
            put_times.append(time.perf_counter() - t0)
        
        # GET (hot)
        get_times = []
        out = np.zeros(size, dtype=np.uint8)
        for i in range(iterations):
            key = f"block_{size_mb}_0"
            t0 = time.perf_counter()
            backend.get(key, out)
            get_times.append(time.perf_counter() - t0)
        
        put_median = sorted(put_times)[iterations // 2]
        get_median = sorted(get_times)[iterations // 2]
        
        results.append({
            'size_mb': size_mb,
            'put_gbps': round(size / 1e9 / put_median, 2),
            'get_gbps': round(size / 1e9 / get_median, 2),
        })
    
    return results

# ============================================================================
# Main
# ============================================================================
print("\n" + "=" * 70)
print("Full Systems Benchmark - Real Backends")
print("=" * 70)

BLOCK_SIZES = [64, 256, 512]  # MB
ITERATIONS = 5

all_results = {}

# 1. Cascade C++
print("\n[1/4] Testing Cascade C++ Backend...")
try:
    cascade = CascadeCppBackend("/dev/shm/cascade_bench", capacity_gb=10)
    all_results["Cascade-C++"] = benchmark_system(cascade, BLOCK_SIZES, ITERATIONS)
    print(f"  Done: {all_results['Cascade-C++']}")
except Exception as e:
    print(f"  Error: {e}")


# 2. HDF5
print("\n[2/4] Testing HDF5 Backend...")
try:
    hdf5 = HDF5Backend("/tmp/hdf5_bench")
    all_results["HDF5"] = benchmark_system(hdf5, BLOCK_SIZES, ITERATIONS)
    print(f"  Done: {all_results['HDF5']}")
except Exception as e:
    print(f"  Error: {e}")

# 3. Python mmap
print("\n[3/4] Testing Python mmap Backend...")
try:
    pymmap = PythonMmapBackend("/dev/shm/pymmap_bench")
    all_results["Python-mmap"] = benchmark_system(pymmap, BLOCK_SIZES, ITERATIONS)
    print(f"  Done: {all_results['Python-mmap']}")
except Exception as e:
    print(f"  Error: {e}")

# 4. Lustre
print("\n[4/4] Testing Lustre Backend...")
try:
    lustre = LustreBackend("/pscratch/sd/s/sgkim/Skim-cascade/benchmark/tmp_lustre")
    all_results["Lustre"] = benchmark_system(lustre, BLOCK_SIZES, ITERATIONS)
    print(f"  Done: {all_results['Lustre']}")
except Exception as e:
    print(f"  Error: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'System':<15} {'64MB PUT':>10} {'64MB GET':>10} {'256MB PUT':>11} {'256MB GET':>11} {'512MB PUT':>11} {'512MB GET':>11}")
print("-" * 85)

for name, results in all_results.items():
    row = [name]
    for r in results:
        row.extend([f"{r['put_gbps']:.2f}", f"{r['get_gbps']:.2f}"])
    print(f"{row[0]:<15} {row[1]:>10} {row[2]:>10} {row[3]:>11} {row[4]:>11} {row[5]:>11} {row[6]:>11}")

# Save results
output = {
    'job_id': SLURM_JOB_ID,
    'timestamp': datetime.now().isoformat(),
    'block_sizes_mb': BLOCK_SIZES,
    'iterations': ITERATIONS,
    'results': all_results
}

out_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/full_systems_{SLURM_JOB_ID}.json"
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved: {out_path}")
PYEOF

echo ""
echo "Done!"
date
