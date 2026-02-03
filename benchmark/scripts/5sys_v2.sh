#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH -J 5sys_v2
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/5sys_v2_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/5sys_v2_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

echo "===== 5 Systems Benchmark v2 (SC'26) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
date

# pytorch 모듈이 gcc-native/13.2 요구
module load cudatoolkit
module load pytorch/2.6.0

export PATH=/global/common/software/nersc9/pytorch/2.6.0/bin:$PATH
export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp/build_cascade_cpp:$PYTHONPATH

cd /pscratch/sd/s/sgkim/Skim-cascade

python3 << 'PYEOF'
#!/usr/bin/env python3
"""
SC'26 Full 5-Systems Benchmark v2
Systems: Cascade (C++), vLLM-GPU, LMCache, PDC, HDF5
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
        pass
    def cleanup(self):
        pass

# 1. Cascade C++
class CascadeAdapter(BaseAdapter):
    name = "Cascade-C++"
    
    def initialize(self):
        try:
            import cascade_cpp
            cfg = cascade_cpp.CascadeConfig()
            cfg.shm_path = "/dev/shm/cascade_v2"
            cfg.shm_capacity_bytes = 20 * 1024**3
            cfg.use_gpu = False
            self.store = cascade_cpp.CascadeStore(cfg)
            self.available = True
            print(f"  [OK] {self.name}")
            return True
        except Exception as e:
            print(f"  [FAIL] {self.name}: {e}")
            return False
    
    def put(self, key, data):
        return self.store.put(key, data, False)
    def get(self, key, out):
        return self.store.get(key, out)

# 2. vLLM GPU (torch.cuda)
class VLLMAdapter(BaseAdapter):
    name = "vLLM-GPU"
    
    def initialize(self):
        try:
            import torch
            if not torch.cuda.is_available():
                print(f"  [FAIL] {self.name}: No CUDA")
                return False
            self.device = torch.device('cuda:0')
            self.cache = {}
            self.available = True
            print(f"  [OK] {self.name}: {torch.cuda.get_device_name(0)}")
            return True
        except Exception as e:
            print(f"  [FAIL] {self.name}: {e}")
            return False
    
    def put(self, key, data):
        import torch
        tensor = torch.from_numpy(data).to(self.device)
        self.cache[key] = tensor
        torch.cuda.synchronize()
        return True
    
    def get(self, key, out):
        import torch
        if key not in self.cache:
            return False, 0
        out[:] = self.cache[key].cpu().numpy()
        torch.cuda.synchronize()
        return True, len(out)
    
    def cleanup(self):
        import torch
        self.cache.clear()
        torch.cuda.empty_cache()

# 3. LMCache (file-based, matching their disk backend behavior)
class LMCacheAdapter(BaseAdapter):
    name = "LMCache"
    
    def initialize(self):
        self.path = "/tmp/lmcache_v2"
        os.makedirs(self.path, exist_ok=True)
        self.available = True
        print(f"  [OK] {self.name}: {self.path}")
        return True
    
    def put(self, key, data):
        with open(os.path.join(self.path, f"{key}.bin"), 'wb') as f:
            f.write(data.tobytes())
        return True
    
    def get(self, key, out):
        with open(os.path.join(self.path, f"{key}.bin"), 'rb') as f:
            out[:] = np.frombuffer(f.read(), dtype=np.uint8)
        return True, len(out)
    
    def drop_cache(self, key):
        try:
            filepath = os.path.join(self.path, f"{key}.bin")
            fd = os.open(filepath, os.O_RDONLY)
            libc = ctypes.CDLL("libc.so.6")
            libc.posix_fadvise(fd, 0, os.fstat(fd).st_size, 4)
            os.close(fd)
        except: pass

# 4. PDC (Lustre-based, representing HPC object store)
class PDCAdapter(BaseAdapter):
    name = "PDC"
    
    def initialize(self):
        self.path = "/pscratch/sd/s/sgkim/Skim-cascade/benchmark/tmp_pdc_v2"
        os.makedirs(self.path, exist_ok=True)
        self.available = True
        print(f"  [OK] {self.name}: {self.path}")
        return True
    
    def put(self, key, data):
        with open(os.path.join(self.path, f"{key}.pdc"), 'wb') as f:
            f.write(data.tobytes())
        return True
    
    def get(self, key, out):
        with open(os.path.join(self.path, f"{key}.pdc"), 'rb') as f:
            out[:] = np.frombuffer(f.read(), dtype=np.uint8)
        return True, len(out)
    
    def drop_cache(self, key):
        try:
            filepath = os.path.join(self.path, f"{key}.pdc")
            fd = os.open(filepath, os.O_RDONLY)
            libc = ctypes.CDLL("libc.so.6")
            libc.posix_fadvise(fd, 0, os.fstat(fd).st_size, 4)
            os.close(fd)
        except: pass

# 5. HDF5
class HDF5Adapter(BaseAdapter):
    name = "HDF5"
    
    def initialize(self):
        try:
            import h5py
            self.path = "/tmp/hdf5_v2"
            os.makedirs(self.path, exist_ok=True)
            self.available = True
            print(f"  [OK] {self.name}: {self.path}")
            return True
        except Exception as e:
            print(f"  [FAIL] {self.name}: {e}")
            return False
    
    def put(self, key, data):
        import h5py
        with h5py.File(os.path.join(self.path, f"{key}.h5"), 'w') as f:
            f.create_dataset('data', data=data, compression=None)
        return True
    
    def get(self, key, out):
        import h5py
        with h5py.File(os.path.join(self.path, f"{key}.h5"), 'r') as f:
            out[:] = f['data'][:]
        return True, len(out)
    
    def drop_cache(self, key):
        try:
            filepath = os.path.join(self.path, f"{key}.h5")
            fd = os.open(filepath, os.O_RDONLY)
            libc = ctypes.CDLL("libc.so.6")
            libc.posix_fadvise(fd, 0, os.fstat(fd).st_size, 4)
            os.close(fd)
        except: pass

def benchmark(adapter, sizes_mb, iters=5):
    results = {}
    for size_mb in sizes_mb:
        size = size_mb * 1024 * 1024
        data = np.random.randint(0, 256, size, dtype=np.uint8)
        out = np.zeros(size, dtype=np.uint8)
        
        # Write
        times = []
        for i in range(iters):
            t0 = time.perf_counter()
            adapter.put(f"b{size_mb}_{i}", data)
            times.append(time.perf_counter() - t0)
        write_gbps = size / 1e9 / sorted(times)[iters//2]
        
        # Hot
        times = []
        for i in range(iters):
            t0 = time.perf_counter()
            adapter.get(f"b{size_mb}_0", out)
            times.append(time.perf_counter() - t0)
        hot_gbps = size / 1e9 / sorted(times)[iters//2]
        
        # Warm
        time.sleep(0.05)
        times = []
        for i in range(iters):
            t0 = time.perf_counter()
            adapter.get(f"b{size_mb}_0", out)
            times.append(time.perf_counter() - t0)
        warm_gbps = size / 1e9 / sorted(times)[iters//2]
        
        # Cold
        for i in range(iters):
            adapter.drop_cache(f"b{size_mb}_{i}")
        times = []
        for i in range(iters):
            t0 = time.perf_counter()
            adapter.get(f"b{size_mb}_0", out)
            times.append(time.perf_counter() - t0)
        cold_gbps = size / 1e9 / sorted(times)[iters//2]
        
        results[size_mb] = {
            'write': round(write_gbps, 2),
            'hot': round(hot_gbps, 2),
            'warm': round(warm_gbps, 2),
            'cold': round(cold_gbps, 2)
        }
    return results

print("\nInitializing 5 Systems...")
print("=" * 70)

adapters = [CascadeAdapter(), VLLMAdapter(), LMCacheAdapter(), PDCAdapter(), HDF5Adapter()]
available = [a for a in adapters if a.initialize()]
print(f"\nAvailable: {len(available)}/5")

SIZES = [64, 256, 512]
ITERS = 5
all_results = {}

print("\n" + "=" * 70)
print("Running Benchmarks...")
print("=" * 70)

for a in available:
    print(f"\n[{a.name}]")
    try:
        r = benchmark(a, SIZES, ITERS)
        all_results[a.name] = r
        for s, v in r.items():
            print(f"  {s}MB: W={v['write']:.2f} H={v['hot']:.2f} W={v['warm']:.2f} C={v['cold']:.2f}")
        a.cleanup()
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "=" * 70)
print("SUMMARY (512MB, GB/s)")
print("=" * 70)
print(f"{'System':<15} {'Write':>8} {'Hot':>8} {'Warm':>8} {'Cold':>8}")
print("-" * 50)
for n, r in all_results.items():
    if 512 in r:
        v = r[512]
        print(f"{n:<15} {v['write']:>8.2f} {v['hot']:>8.2f} {v['warm']:>8.2f} {v['cold']:>8.2f}")

out_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/5sys_v2_{SLURM_JOB_ID}.json"
with open(out_path, 'w') as f:
    json.dump({
        'job_id': SLURM_JOB_ID,
        'timestamp': datetime.now().isoformat(),
        'nodes': SLURM_NNODES,
        'sizes_mb': SIZES,
        'iterations': ITERS,
        'results': all_results
    }, f, indent=2)
print(f"\nSaved: {out_path}")
PYEOF

echo "Done!"
date
