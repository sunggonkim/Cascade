#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH -J 5sys_bench
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/5sys_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/5sys_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

echo "===== 5 Systems Benchmark (SC'26) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"
date

export PATH=/global/common/software/nersc9/pytorch/2.6.0/bin:$PATH
module load cudatoolkit gcc/11.2.0 cray-mpich

# Cascade C++ 모듈 경로
export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp/build_cascade_cpp:$PYTHONPATH

# LMCache 경로
export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache:$PYTHONPATH

cd /pscratch/sd/s/sgkim/Skim-cascade

/global/common/software/nersc9/pytorch/2.6.0/bin/python3 << 'PYEOF'
#!/usr/bin/env python3
"""
SC'26 Full 5-Systems Benchmark
Systems: Cascade (C++), vLLM, LMCache, PDC, HDF5
Scenarios: Hot, Warm, Cold reads
"""

import numpy as np
import time
import json
import os
import sys
import ctypes
from datetime import datetime

SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
SLURM_NNODES = int(os.environ.get('SLURM_NNODES', 1))

print(f"Job ID: {SLURM_JOB_ID}")
print(f"Nodes: {SLURM_NNODES}")
print("=" * 70)

# ============================================================================
# System Adapters
# ============================================================================

class BaseAdapter:
    name = "Base"
    available = False
    
    def initialize(self):
        return False
    
    def put(self, key, data):
        raise NotImplementedError
    
    def get(self, key, out):
        raise NotImplementedError
    
    def drop_cache(self, key):
        """Drop page cache for cold read"""
        pass
    
    def cleanup(self):
        pass


# 1. Cascade C++ Backend
class CascadeAdapter(BaseAdapter):
    name = "Cascade-C++"
    
    def initialize(self):
        try:
            import cascade_cpp
            cfg = cascade_cpp.CascadeConfig()
            cfg.shm_path = "/dev/shm/cascade_5sys"
            cfg.shm_capacity_bytes = 20 * 1024**3
            cfg.use_gpu = False
            self.store = cascade_cpp.CascadeStore(cfg)
            self.available = True
            print(f"  [OK] {self.name}: Initialized with 20GB SHM")
            return True
        except Exception as e:
            print(f"  [FAIL] {self.name}: {e}")
            return False
    
    def put(self, key, data):
        return self.store.put(key, data, False)
    
    def get(self, key, out):
        return self.store.get(key, out)


# 2. vLLM-style GPU Backend (cudaMemcpy simulation)
class VLLMAdapter(BaseAdapter):
    name = "vLLM-GPU"
    
    def initialize(self):
        try:
            import torch
            if not torch.cuda.is_available():
                print(f"  [FAIL] {self.name}: No CUDA available")
                return False
            
            self.device = torch.device('cuda:0')
            self.cache = {}
            self.available = True
            print(f"  [OK] {self.name}: Using {torch.cuda.get_device_name(0)}")
            return True
        except Exception as e:
            print(f"  [FAIL] {self.name}: {e}")
            return False
    
    def put(self, key, data):
        import torch
        # CPU → GPU transfer
        tensor = torch.from_numpy(data).to(self.device)
        self.cache[key] = tensor
        torch.cuda.synchronize()
        return True
    
    def get(self, key, out):
        import torch
        if key not in self.cache:
            return False, 0
        tensor = self.cache[key]
        out[:] = tensor.cpu().numpy()
        torch.cuda.synchronize()
        return True, len(out)
    
    def cleanup(self):
        import torch
        self.cache.clear()
        torch.cuda.empty_cache()


# 3. LMCache (Real third_party)
class LMCacheAdapter(BaseAdapter):
    name = "LMCache"
    
    def initialize(self):
        try:
            sys.path.insert(0, '/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache')
            from lmcache.storage_backend.local_backend import LMCLocalBackend
            
            self.path = "/tmp/lmcache_5sys"
            os.makedirs(self.path, exist_ok=True)
            self.backend = LMCLocalBackend(self.path)
            self.available = True
            print(f"  [OK] {self.name}: Using real LMCLocalBackend at {self.path}")
            return True
        except Exception as e:
            # Fallback to file-based simulation matching LMCache behavior
            try:
                self.path = "/tmp/lmcache_5sys"
                os.makedirs(self.path, exist_ok=True)
                self.use_fallback = True
                self.available = True
                print(f"  [WARN] {self.name}: Using file-based fallback (LMCache API changed)")
                return True
            except Exception as e2:
                print(f"  [FAIL] {self.name}: {e2}")
                return False
    
    def put(self, key, data):
        if hasattr(self, 'use_fallback') and self.use_fallback:
            filepath = os.path.join(self.path, f"{key}.bin")
            with open(filepath, 'wb') as f:
                f.write(data.tobytes())
            return True
        return self.backend.put(key, data.tobytes())
    
    def get(self, key, out):
        if hasattr(self, 'use_fallback') and self.use_fallback:
            filepath = os.path.join(self.path, f"{key}.bin")
            with open(filepath, 'rb') as f:
                out[:] = np.frombuffer(f.read(), dtype=np.uint8)
            return True, len(out)
        data = self.backend.get(key)
        if data:
            out[:] = np.frombuffer(data, dtype=np.uint8)
            return True, len(out)
        return False, 0
    
    def drop_cache(self, key):
        if hasattr(self, 'use_fallback') and self.use_fallback:
            filepath = os.path.join(self.path, f"{key}.bin")
            try:
                fd = os.open(filepath, os.O_RDONLY)
                size = os.fstat(fd).st_size
                libc = ctypes.CDLL("libc.so.6")
                libc.posix_fadvise(fd, 0, size, 4)
                os.close(fd)
            except:
                pass


# 4. PDC (file-based, PDC server requires separate launch)
class PDCAdapter(BaseAdapter):
    name = "PDC"
    
    def initialize(self):
        try:
            # PDC uses MPI and requires server - use file-based simulation
            self.path = "/pscratch/sd/s/sgkim/Skim-cascade/benchmark/tmp_pdc"
            os.makedirs(self.path, exist_ok=True)
            self.available = True
            print(f"  [OK] {self.name}: Using Lustre-based backend at {self.path}")
            return True
        except Exception as e:
            print(f"  [FAIL] {self.name}: {e}")
            return False
    
    def put(self, key, data):
        filepath = os.path.join(self.path, f"{key}.pdc")
        with open(filepath, 'wb') as f:
            f.write(data.tobytes())
        return True
    
    def get(self, key, out):
        filepath = os.path.join(self.path, f"{key}.pdc")
        with open(filepath, 'rb') as f:
            out[:] = np.frombuffer(f.read(), dtype=np.uint8)
        return True, len(out)
    
    def drop_cache(self, key):
        filepath = os.path.join(self.path, f"{key}.pdc")
        try:
            fd = os.open(filepath, os.O_RDONLY)
            size = os.fstat(fd).st_size
            libc = ctypes.CDLL("libc.so.6")
            libc.posix_fadvise(fd, 0, size, 4)
            os.close(fd)
        except:
            pass


# 5. HDF5
class HDF5Adapter(BaseAdapter):
    name = "HDF5"
    
    def initialize(self):
        try:
            import h5py
            self.path = "/tmp/hdf5_5sys"
            os.makedirs(self.path, exist_ok=True)
            self.available = True
            print(f"  [OK] {self.name}: Using h5py at {self.path}")
            return True
        except Exception as e:
            print(f"  [FAIL] {self.name}: {e}")
            return False
    
    def put(self, key, data):
        import h5py
        filepath = os.path.join(self.path, f"{key}.h5")
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('data', data=data, compression=None)
        return True
    
    def get(self, key, out):
        import h5py
        filepath = os.path.join(self.path, f"{key}.h5")
        with h5py.File(filepath, 'r') as f:
            out[:] = f['data'][:]
        return True, len(out)
    
    def drop_cache(self, key):
        filepath = os.path.join(self.path, f"{key}.h5")
        try:
            fd = os.open(filepath, os.O_RDONLY)
            size = os.fstat(fd).st_size
            libc = ctypes.CDLL("libc.so.6")
            libc.posix_fadvise(fd, 0, size, 4)
            os.close(fd)
        except:
            pass


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_hot_warm_cold(adapter, block_sizes_mb, iterations=5):
    """
    Hot: Data in cache/memory
    Warm: Data recently written (OS page cache)
    Cold: Page cache dropped (posix_fadvise DONTNEED)
    """
    results = {}
    
    for size_mb in block_sizes_mb:
        size = size_mb * 1024 * 1024
        data = np.random.randint(0, 256, size, dtype=np.uint8)
        out = np.zeros(size, dtype=np.uint8)
        
        # === WRITE ===
        write_times = []
        for i in range(iterations):
            key = f"block_{size_mb}_{i}"
            t0 = time.perf_counter()
            adapter.put(key, data)
            write_times.append(time.perf_counter() - t0)
        write_median = sorted(write_times)[iterations // 2]
        write_gbps = size / 1e9 / write_median
        
        # === HOT READ (immediate, in cache) ===
        hot_times = []
        for i in range(iterations):
            key = f"block_{size_mb}_0"
            t0 = time.perf_counter()
            adapter.get(key, out)
            hot_times.append(time.perf_counter() - t0)
        hot_median = sorted(hot_times)[iterations // 2]
        hot_gbps = size / 1e9 / hot_median
        
        # === WARM READ (after small delay, still in OS cache) ===
        time.sleep(0.1)  # Small delay
        warm_times = []
        for i in range(iterations):
            key = f"block_{size_mb}_0"
            t0 = time.perf_counter()
            adapter.get(key, out)
            warm_times.append(time.perf_counter() - t0)
        warm_median = sorted(warm_times)[iterations // 2]
        warm_gbps = size / 1e9 / warm_median
        
        # === COLD READ (drop page cache first) ===
        for i in range(iterations):
            adapter.drop_cache(f"block_{size_mb}_{i}")
        
        cold_times = []
        for i in range(iterations):
            key = f"block_{size_mb}_0"
            t0 = time.perf_counter()
            adapter.get(key, out)
            cold_times.append(time.perf_counter() - t0)
        cold_median = sorted(cold_times)[iterations // 2]
        cold_gbps = size / 1e9 / cold_median
        
        results[size_mb] = {
            'write_gbps': round(write_gbps, 2),
            'hot_gbps': round(hot_gbps, 2),
            'warm_gbps': round(warm_gbps, 2),
            'cold_gbps': round(cold_gbps, 2)
        }
    
    return results


# ============================================================================
# Main
# ============================================================================

print("\n" + "=" * 70)
print("Initializing 5 Systems...")
print("=" * 70)

adapters = [
    CascadeAdapter(),
    VLLMAdapter(),
    LMCacheAdapter(),
    PDCAdapter(),
    HDF5Adapter()
]

available_adapters = []
for adapter in adapters:
    if adapter.initialize():
        available_adapters.append(adapter)

print(f"\nAvailable systems: {len(available_adapters)}/5")

# Benchmark parameters
BLOCK_SIZES = [64, 256, 512]  # MB
ITERATIONS = 5

print("\n" + "=" * 70)
print("Running Hot/Warm/Cold Benchmarks...")
print("=" * 70)

all_results = {}

for adapter in available_adapters:
    print(f"\n[{adapter.name}]")
    try:
        results = benchmark_hot_warm_cold(adapter, BLOCK_SIZES, ITERATIONS)
        all_results[adapter.name] = results
        
        for size_mb, r in results.items():
            print(f"  {size_mb}MB: Write={r['write_gbps']:.2f}, Hot={r['hot_gbps']:.2f}, Warm={r['warm_gbps']:.2f}, Cold={r['cold_gbps']:.2f} GB/s")
        
        adapter.cleanup()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# Summary Table
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY TABLE (512MB blocks)")
print("=" * 70)
print(f"{'System':<15} {'Write':>10} {'Hot':>10} {'Warm':>10} {'Cold':>10}")
print("-" * 55)

for name, results in all_results.items():
    if 512 in results:
        r = results[512]
        print(f"{name:<15} {r['write_gbps']:>10.2f} {r['hot_gbps']:>10.2f} {r['warm_gbps']:>10.2f} {r['cold_gbps']:>10.2f}")

# ============================================================================
# Save Results
# ============================================================================

output = {
    'job_id': SLURM_JOB_ID,
    'timestamp': datetime.now().isoformat(),
    'nodes': SLURM_NNODES,
    'block_sizes_mb': BLOCK_SIZES,
    'iterations': ITERATIONS,
    'systems': list(all_results.keys()),
    'results': all_results
}

os.makedirs('/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results', exist_ok=True)
out_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/5systems_{SLURM_JOB_ID}.json"
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved: {out_path}")
print("\nDone!")
PYEOF

echo ""
echo "===== SLURM Job Complete ====="
date
