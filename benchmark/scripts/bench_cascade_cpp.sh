#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -J bench_cpp
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/bench_cpp_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/bench_cpp_%j.err
#SBATCH --gpus-per-node=4

export PATH=/global/common/software/nersc9/pytorch/2.6.0/bin:$PATH
module load cudatoolkit gcc/11.2.0

cd /pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp/build_cascade_cpp
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "===== Cascade C++ Backend Benchmark ====="
echo "Job ID: $SLURM_JOB_ID"
date

/global/common/software/nersc9/pytorch/2.6.0/bin/python3 << 'PYEOF'
import cascade_cpp
import numpy as np
import time
import json
from datetime import datetime

print("Cascade Python->C++ Backend Benchmark")
print("=" * 60)

# Config with conservative size
cfg = cascade_cpp.CascadeConfig()
cfg.shm_path = "/dev/shm/cascade_bench"
cfg.shm_capacity_bytes = 10 * 1024**3  # 10GB (conservative)
cfg.use_gpu = False  # SHM only for now

store = cascade_cpp.CascadeStore(cfg)
print("CascadeStore (C++) initialized")

results = []
block_sizes = [64, 256, 512]  # MB

for size_mb in block_sizes:
    size = size_mb * 1024 * 1024
    print(f"\n--- {size_mb} MB Block ---")
    
    data = np.random.randint(0, 256, size, dtype=np.uint8)
    
    # PUT: 5 iterations, use median
    put_times = []
    for i in range(5):
        t0 = time.perf_counter()
        store.put(f"block_{size_mb}_{i}", data, False)
        put_times.append(time.perf_counter() - t0)
    put_median = sorted(put_times)[2]
    put_gbps = size / 1e9 / put_median
    
    # GET: 5 iterations, use median
    get_times = []
    out = np.zeros(size, dtype=np.uint8)
    for i in range(5):
        t0 = time.perf_counter()
        store.get(f"block_{size_mb}_0", out)
        get_times.append(time.perf_counter() - t0)
    get_median = sorted(get_times)[2]
    get_gbps = size / 1e9 / get_median
    
    print(f"PUT: {put_gbps:6.2f} GB/s (median of 5)")
    print(f"GET: {get_gbps:6.2f} GB/s (median of 5)")
    
    # Verify data integrity
    ok = np.array_equal(data, out)
    print(f"Data integrity: {'OK' if ok else 'FAILED!'}")
    
    results.append({
        'size_mb': size_mb,
        'put_gbps': round(put_gbps, 2),
        'get_gbps': round(get_gbps, 2),
        'verified': ok
    })

print("\n" + "=" * 60)
print("Summary (Python -> C++ mmap SHM):")
print(f"{'Size':>8} {'PUT GB/s':>12} {'GET GB/s':>12}")
for r in results:
    print(f"{r['size_mb']:>6} MB {r['put_gbps']:>12.2f} {r['get_gbps']:>12.2f}")

print("\nStats:", store.get_stats())

# Save results
import os
out_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/cpp_bench_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
with open(out_path, 'w') as f:
    json.dump({
        'job_id': os.environ.get('SLURM_JOB_ID', 'local'),
        'timestamp': datetime.now().isoformat(),
        'system': 'Cascade-C++',
        'backend': 'SHM-mmap',
        'results': results
    }, f, indent=2)
print(f"\nResults saved: {out_path}")
PYEOF

echo ""
echo "Done!"
date
