#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J fair_tier
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/fair_tier_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/fair_tier_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 64

echo "===== Fair Storage Tier Benchmark ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
date

module load cudatoolkit
module load pytorch/2.6.0

export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp/build_cascade_cpp:$PYTHONPATH

nvidia-smi --query-gpu=name,memory.total --format=csv

cd /pscratch/sd/s/sgkim/Skim-cascade

python3 << 'PYEOF'
import numpy as np
import time
import json
import os
import sys
import ctypes
from datetime import datetime
from pathlib import Path

SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')

print("\n" + "="*70)
print(" FAIR STORAGE TIER BENCHMARK")
print(" 각 저장소 계층의 순수 성능만 측정 (같은 조건)")
print("="*70)

###############################################################################
# Config
###############################################################################

BLOCK_SIZES_MB = [128, 256, 512]
NUM_ITERS = 5
RESULTS = {}

SHM_PATH = "/dev/shm/fair_tier"
NVME_PATH = "/tmp/fair_tier"
LUSTRE_PATH = os.environ.get("SCRATCH", "/pscratch/sd/s/sgkim") + "/fair_tier"

for p in [SHM_PATH, NVME_PATH, LUSTRE_PATH]:
    Path(p).mkdir(parents=True, exist_ok=True)

###############################################################################
# Utility
###############################################################################

def drop_page_cache(filepath):
    try:
        fd = os.open(filepath, os.O_RDONLY)
        size = os.fstat(fd).st_size
        libc = ctypes.CDLL("libc.so.6")
        libc.posix_fadvise(fd, 0, size, 4)  # POSIX_FADV_DONTNEED
        os.close(fd)
    except:
        pass

def bench_tier(name, write_fn, read_fn, sizes, iters=5, drop_fn=None):
    results = {}
    for size_mb in sizes:
        size = size_mb * 1024 * 1024
        data = np.random.randint(0, 256, size, dtype=np.uint8)
        out = np.zeros(size, dtype=np.uint8)
        
        # Write
        times = []
        for i in range(iters):
            t0 = time.perf_counter()
            write_fn(f"b{size_mb}_{i}", data)
            times.append(time.perf_counter() - t0)
        w = size / 1e9 / np.median(times)
        
        # Hot
        times = []
        for i in range(iters):
            t0 = time.perf_counter()
            read_fn(f"b{size_mb}_0", out)
            times.append(time.perf_counter() - t0)
        h = size / 1e9 / np.median(times)
        
        # Cold
        c = None
        if drop_fn:
            drop_fn(f"b{size_mb}_0")
            times = []
            for i in range(iters):
                t0 = time.perf_counter()
                read_fn(f"b{size_mb}_0", out)
                times.append(time.perf_counter() - t0)
            c = size / 1e9 / np.median(times)
        
        results[size_mb] = {"write": round(w,2), "hot": round(h,2), "cold": round(c,2) if c else None}
        cs = f"{c:.2f}" if c else "N/A"
        print(f"  {size_mb}MB: W={w:.2f}, H={h:.2f}, C={cs} GB/s")
    return results

###############################################################################
# 1. GPU (PCIe)
###############################################################################

print("\n[1] GPU HBM (torch.cuda)")
print("-"*50)

try:
    import torch
    if torch.cuda.is_available():
        dev = torch.device('cuda:0')
        cache = {}
        def gw(k, d):
            t = torch.from_numpy(d).to(dev)
            torch.cuda.synchronize()
            cache[k] = t
        def gr(k, o):
            o[:] = cache[k].cpu().numpy()
            torch.cuda.synchronize()
        RESULTS["GPU-PCIe"] = bench_tier("GPU", gw, gr, BLOCK_SIZES_MB)
        cache.clear()
        torch.cuda.empty_cache()
    else:
        print("  [SKIP] No CUDA")
except Exception as e:
    print(f"  [ERROR] {e}")

###############################################################################
# 2. SHM
###############################################################################

print("\n[2] DRAM SHM (/dev/shm)")
print("-"*50)

try:
    files = {}
    def sw(k, d):
        p = f"{SHM_PATH}/{k}.bin"
        with open(p, 'wb') as f: f.write(d.tobytes())
        files[k] = p
    def sr(k, o):
        with open(files[k], 'rb') as f: o[:] = np.frombuffer(f.read(), dtype=np.uint8)
    RESULTS["DRAM-SHM"] = bench_tier("SHM", sw, sr, BLOCK_SIZES_MB)
except Exception as e:
    print(f"  [ERROR] {e}")

###############################################################################
# 3. NVMe
###############################################################################

print("\n[3] NVMe (/tmp)")
print("-"*50)

try:
    nfiles = {}
    def nw(k, d):
        p = f"{NVME_PATH}/{k}.bin"
        with open(p, 'wb') as f: f.write(d.tobytes())
        nfiles[k] = p
    def nr(k, o):
        with open(nfiles[k], 'rb') as f: o[:] = np.frombuffer(f.read(), dtype=np.uint8)
    def nd(k):
        drop_page_cache(nfiles[k])
    RESULTS["NVMe"] = bench_tier("NVMe", nw, nr, BLOCK_SIZES_MB, drop_fn=nd)
except Exception as e:
    print(f"  [ERROR] {e}")

###############################################################################
# 4. Lustre
###############################################################################

print("\n[4] Lustre ($SCRATCH)")
print("-"*50)

try:
    lfiles = {}
    def lw(k, d):
        p = f"{LUSTRE_PATH}/{k}.bin"
        with open(p, 'wb') as f: f.write(d.tobytes())
        lfiles[k] = p
    def lr(k, o):
        with open(lfiles[k], 'rb') as f: o[:] = np.frombuffer(f.read(), dtype=np.uint8)
    def ld(k):
        drop_page_cache(lfiles[k])
    RESULTS["Lustre"] = bench_tier("Lustre", lw, lr, BLOCK_SIZES_MB, drop_fn=ld)
except Exception as e:
    print(f"  [ERROR] {e}")

###############################################################################
# 5. Cascade C++
###############################################################################

print("\n[5] Cascade C++ (SHM Backend)")
print("-"*50)

try:
    import cascade_cpp
    cfg = cascade_cpp.CascadeConfig()
    cfg.shm_path = f"{SHM_PATH}/cascade"
    cfg.shm_capacity_bytes = 20 * 1024**3
    cfg.use_gpu = False
    Path(cfg.shm_path).mkdir(parents=True, exist_ok=True)
    store = cascade_cpp.CascadeStore(cfg)
    print(f"  Cascade: {cfg.shm_path}")
    
    def cw(k, d):
        store.put(k, d, False)
    def cr(k, o):
        store.get(k, o)
    RESULTS["Cascade-C++"] = bench_tier("Cascade", cw, cr, BLOCK_SIZES_MB)
except ImportError as e:
    print(f"  [ERROR] cascade_cpp: {e}")
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()

###############################################################################
# Summary
###############################################################################

print("\n" + "="*70)
print(" SUMMARY (512MB, GB/s)")
print("="*70)
print(f"{'Tier':<15} {'Write':>10} {'Hot':>10} {'Cold':>10}")
print("-"*50)

for tier, data in RESULTS.items():
    if 512 in data:
        r = data[512]
        cs = f"{r['cold']:.2f}" if r.get('cold') else "N/A"
        print(f"{tier:<15} {r['write']:>10.2f} {r['hot']:>10.2f} {cs:>10}")

###############################################################################
# HW Efficiency
###############################################################################

print("\n" + "="*70)
print(" HARDWARE EFFICIENCY (512MB Hot Read)")
print("="*70)

HW = {
    "GPU-PCIe": 32,      # PCIe 4.0 x16
    "DRAM-SHM": 200,     # DDR4
    "NVMe": 7,           # Node NVMe
    "Lustre": 5,         # Single client
    "Cascade-C++": 200   # DDR4 via SHM
}

print(f"{'Tier':<15} {'Measured':>10} {'HW Limit':>10} {'Efficiency':>10}")
print("-"*50)

for tier, data in RESULTS.items():
    if 512 in data and tier in HW:
        m = data[512]['hot']
        hw = HW[tier]
        eff = m / hw * 100
        print(f"{tier:<15} {m:>10.2f} {hw:>10} {eff:>9.1f}%")

###############################################################################
# Save
###############################################################################

output = {
    "job_id": SLURM_JOB_ID,
    "timestamp": datetime.now().isoformat(),
    "results": RESULTS,
    "hw_reference": HW
}

path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/fair_tier_{SLURM_JOB_ID}.json"
with open(path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")

###############################################################################
# Cleanup
###############################################################################

import shutil
for p in [SHM_PATH, NVME_PATH]:
    try: shutil.rmtree(p)
    except: pass

print("\nDone!")
PYEOF

echo ""
echo "===== Complete ====="
date
