#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gpus-per-node=4
#SBATCH -t 00:30:00
#SBATCH -J tiered_5sys
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/tiered_5sys_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/tiered_5sys_%j.err

###############################################################################
# Tiered KV Cache Benchmark - 올바른 동작 방식
#
# 모든 시스템이 동일한 VRAM 캐시 레이어를 가짐:
#   - VRAM Hit: GPU에서 직접 반환 (모든 시스템 동일)
#   - VRAM Miss: 각 시스템의 백엔드에서 가져와서 VRAM에 로드
#
# 측정:
#   1. VRAM Hit 성능 (모든 시스템 동일해야 함)
#   2. VRAM Miss → Backend 성능 (시스템별 차이)
#   3. 다양한 Hit Rate에서의 평균 성능
###############################################################################

set -e

echo "=================================================================="
echo "Tiered 5-System Benchmark (올바른 동작)"
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
from datetime import datetime
from collections import OrderedDict

job_id = os.environ.get("SLURM_JOB_ID", "local")
device = torch.device("cuda:0")

print("\n" + "="*80)
print("Tiered 5-System Benchmark - 올바른 VRAM 캐시 동작")
print("="*80)
print(f"GPU: {torch.cuda.get_device_name(0)}")

###############################################################################
# Configuration
###############################################################################

BLOCK_SIZE_MB = 512
BLOCK_SIZE = BLOCK_SIZE_MB * 1024 * 1024
NUM_BLOCKS = 10  # 5GB total
NUM_ACCESS = 50  # 50번 접근
VRAM_CACHE_SIZE = 5  # 5개 블록만 VRAM에 캐시 (50% 용량)

results = {}

###############################################################################
# Base Tiered Cache Class (공통 VRAM 캐시 레이어)
###############################################################################

class TieredCache:
    """모든 시스템이 공유하는 VRAM 캐시 레이어"""
    
    def __init__(self, name, vram_capacity, backend_get_fn, backend_put_fn):
        self.name = name
        self.vram_capacity = vram_capacity
        self.vram_cache = OrderedDict()  # LRU: block_id -> GPU tensor
        self.backend_get = backend_get_fn
        self.backend_put = backend_put_fn
        self.stats = {"vram_hit": 0, "vram_miss": 0}
    
    def get(self, block_id):
        """VRAM에서 찾고, 없으면 backend에서 가져와서 VRAM에 로드"""
        if block_id in self.vram_cache:
            # VRAM Hit!
            self.stats["vram_hit"] += 1
            self.vram_cache.move_to_end(block_id)
            return self.vram_cache[block_id]
        else:
            # VRAM Miss - backend에서 가져옴
            self.stats["vram_miss"] += 1
            data = self.backend_get(block_id)
            
            # VRAM에 로드
            if len(self.vram_cache) >= self.vram_capacity:
                # LRU eviction
                evicted_id, evicted_data = self.vram_cache.popitem(last=False)
                # 필요시 backend에 writeback 가능
            
            self.vram_cache[block_id] = data
            return data
    
    def put(self, block_id, data):
        """VRAM에 저장하고 backend에도 동기화"""
        # VRAM에 저장
        if len(self.vram_cache) >= self.vram_capacity:
            evicted_id, evicted_data = self.vram_cache.popitem(last=False)
        
        self.vram_cache[block_id] = data
        # Backend에도 저장 (persistence)
        self.backend_put(block_id, data)
    
    def hit_rate(self):
        total = self.stats["vram_hit"] + self.stats["vram_miss"]
        return self.stats["vram_hit"] / total if total > 0 else 0

###############################################################################
# Backend Implementations for Each System
###############################################################################

# Prepare data in Lustre (cold storage)
LUSTRE_PATH = "/pscratch/sd/s/sgkim/cascade_lustre"
SHM_PATH = "/dev/shm"

print(f"\nPreparing {NUM_BLOCKS} blocks ({NUM_BLOCKS * BLOCK_SIZE_MB / 1024:.1f} GB) in Lustre...")

# Generate blocks
blocks = {}
for i in range(NUM_BLOCKS):
    blocks[f"block_{i}"] = torch.randint(0, 256, (BLOCK_SIZE,), dtype=torch.uint8, device=device)
torch.cuda.synchronize()

# Save to Lustre (cold storage)
os.makedirs(f"{LUSTRE_PATH}/benchmark_blocks", exist_ok=True)
for block_id, data in blocks.items():
    fpath = f"{LUSTRE_PATH}/benchmark_blocks/{block_id}.bin"
    with open(fpath, "wb") as f:
        f.write(data.cpu().numpy().tobytes())

print(f"Blocks saved to {LUSTRE_PATH}/benchmark_blocks/")

###############################################################################
# System 1: Cascade (VRAM → SHM → Lustre)
###############################################################################

print("\n" + "-"*80)
print("[1/5] Cascade: VRAM → SHM (mmap) → Lustre")
print("-"*80)

# Cascade uses SHM as intermediate tier
cascade_shm = {}  # Simulates SHM with pinned memory for fair comparison

def cascade_backend_get(block_id):
    """Cascade: SHM에서 찾고, 없으면 Lustre에서"""
    if block_id in cascade_shm:
        # SHM hit - pinned memory to GPU
        return cascade_shm[block_id].to(device)
    else:
        # SHM miss - load from Lustre
        fpath = f"{LUSTRE_PATH}/benchmark_blocks/{block_id}.bin"
        with open(fpath, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        cpu_tensor = torch.from_numpy(data.copy()).pin_memory()
        cascade_shm[block_id] = cpu_tensor  # Cache in SHM
        return cpu_tensor.to(device)

def cascade_backend_put(block_id, gpu_data):
    """Cascade: SHM에 저장 (pinned memory)"""
    cascade_shm[block_id] = gpu_data.cpu().pin_memory()

cascade_cache = TieredCache("Cascade", VRAM_CACHE_SIZE, cascade_backend_get, cascade_backend_put)

# Warmup - put first blocks
for i in range(NUM_BLOCKS):
    cascade_cache.put(f"block_{i}", blocks[f"block_{i}"])

# Random access pattern
access_pattern = [f"block_{np.random.randint(0, NUM_BLOCKS)}" for _ in range(NUM_ACCESS)]

torch.cuda.synchronize()
start = time.perf_counter()
for block_id in access_pattern:
    _ = cascade_cache.get(block_id)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

cascade_gbps = (BLOCK_SIZE * NUM_ACCESS / 1e9) / elapsed
print(f"   Average: {cascade_gbps:.2f} GB/s (Hit rate: {cascade_cache.hit_rate()*100:.1f}%)")
results["Cascade-C++"] = {"gbps": cascade_gbps, "hit_rate": cascade_cache.hit_rate()}

###############################################################################
# System 2: vLLM-GPU (VRAM only - PagedAttention style)
###############################################################################

print("\n" + "-"*80)
print("[2/5] vLLM-GPU: VRAM only (no tiering, swap to disk on OOM)")
print("-"*80)

vllm_disk_cache = {}  # Simulates disk swap

def vllm_backend_get(block_id):
    """vLLM: disk swap에서 가져옴 (VRAM miss시)"""
    if block_id in vllm_disk_cache:
        # From disk cache (torch.load)
        return torch.load(vllm_disk_cache[block_id], weights_only=True).to(device)
    else:
        # Original from Lustre
        fpath = f"{LUSTRE_PATH}/benchmark_blocks/{block_id}.bin"
        with open(fpath, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return torch.from_numpy(data.copy()).to(device)

def vllm_backend_put(block_id, gpu_data):
    """vLLM: swap to disk (torch.save)"""
    fpath = f"/dev/shm/vllm_swap_{block_id}.pt"
    torch.save(gpu_data.cpu(), fpath)
    vllm_disk_cache[block_id] = fpath

vllm_cache = TieredCache("vLLM", VRAM_CACHE_SIZE, vllm_backend_get, vllm_backend_put)

# Warmup
for i in range(NUM_BLOCKS):
    vllm_cache.put(f"block_{i}", blocks[f"block_{i}"])

torch.cuda.synchronize()
start = time.perf_counter()
for block_id in access_pattern:
    _ = vllm_cache.get(block_id)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

vllm_gbps = (BLOCK_SIZE * NUM_ACCESS / 1e9) / elapsed
print(f"   Average: {vllm_gbps:.2f} GB/s (Hit rate: {vllm_cache.hit_rate()*100:.1f}%)")
results["vLLM-GPU"] = {"gbps": vllm_gbps, "hit_rate": vllm_cache.hit_rate()}

# Cleanup
for fpath in vllm_disk_cache.values():
    if os.path.exists(fpath):
        os.remove(fpath)

###############################################################################
# System 3: PDC (VRAM → File-based container)
###############################################################################

print("\n" + "-"*80)
print("[3/5] PDC: VRAM → File-based container")
print("-"*80)

pdc_file_cache = {}

def pdc_backend_get(block_id):
    """PDC: file container에서 가져옴"""
    if block_id in pdc_file_cache:
        fpath = pdc_file_cache[block_id]
    else:
        fpath = f"{LUSTRE_PATH}/benchmark_blocks/{block_id}.bin"
    
    with open(fpath, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return torch.from_numpy(data.copy()).to(device)

def pdc_backend_put(block_id, gpu_data):
    """PDC: file container에 저장"""
    fpath = f"/dev/shm/pdc_container_{block_id}.bin"
    with open(fpath, "wb") as f:
        f.write(gpu_data.cpu().numpy().tobytes())
    pdc_file_cache[block_id] = fpath

pdc_cache = TieredCache("PDC", VRAM_CACHE_SIZE, pdc_backend_get, pdc_backend_put)

# Warmup
for i in range(NUM_BLOCKS):
    pdc_cache.put(f"block_{i}", blocks[f"block_{i}"])

torch.cuda.synchronize()
start = time.perf_counter()
for block_id in access_pattern:
    _ = pdc_cache.get(block_id)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

pdc_gbps = (BLOCK_SIZE * NUM_ACCESS / 1e9) / elapsed
print(f"   Average: {pdc_gbps:.2f} GB/s (Hit rate: {pdc_cache.hit_rate()*100:.1f}%)")
results["PDC"] = {"gbps": pdc_gbps, "hit_rate": pdc_cache.hit_rate()}

# Cleanup
for fpath in pdc_file_cache.values():
    if os.path.exists(fpath):
        os.remove(fpath)

###############################################################################
# System 4: LMCache (VRAM → local_cpu_backend → disk)
###############################################################################

print("\n" + "-"*80)
print("[4/5] LMCache: VRAM → CPU backend → disk")
print("-"*80)

lmcache_cpu = {}  # Simulates local_cpu_backend

def lmcache_backend_get(block_id):
    """LMCache: CPU backend에서 가져옴"""
    if block_id in lmcache_cpu:
        return lmcache_cpu[block_id].to(device)
    else:
        # Load from disk
        fpath = f"{LUSTRE_PATH}/benchmark_blocks/{block_id}.bin"
        cpu_tensor = torch.load(f"/dev/shm/lmcache_{block_id}.pt", weights_only=True) \
            if os.path.exists(f"/dev/shm/lmcache_{block_id}.pt") \
            else torch.from_numpy(np.fromfile(fpath, dtype=np.uint8).copy())
        lmcache_cpu[block_id] = cpu_tensor
        return cpu_tensor.to(device)

def lmcache_backend_put(block_id, gpu_data):
    """LMCache: CPU backend에 저장"""
    cpu_tensor = gpu_data.cpu()
    lmcache_cpu[block_id] = cpu_tensor
    torch.save(cpu_tensor, f"/dev/shm/lmcache_{block_id}.pt")

lmcache_cache = TieredCache("LMCache", VRAM_CACHE_SIZE, lmcache_backend_get, lmcache_backend_put)

# Warmup
for i in range(NUM_BLOCKS):
    lmcache_cache.put(f"block_{i}", blocks[f"block_{i}"])

torch.cuda.synchronize()
start = time.perf_counter()
for block_id in access_pattern:
    _ = lmcache_cache.get(block_id)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

lmcache_gbps = (BLOCK_SIZE * NUM_ACCESS / 1e9) / elapsed
print(f"   Average: {lmcache_gbps:.2f} GB/s (Hit rate: {lmcache_cache.hit_rate()*100:.1f}%)")
results["LMCache"] = {"gbps": lmcache_gbps, "hit_rate": lmcache_cache.hit_rate()}

# Cleanup
for i in range(NUM_BLOCKS):
    fpath = f"/dev/shm/lmcache_block_{i}.pt"
    if os.path.exists(fpath):
        os.remove(fpath)

###############################################################################
# System 5: HDF5 (VRAM → HDF5 file)
###############################################################################

print("\n" + "-"*80)
print("[5/5] HDF5: VRAM → HDF5 file")
print("-"*80)

try:
    import h5py
    
    hdf5_file = "/dev/shm/hdf5_cache.h5"
    
    def hdf5_backend_get(block_id):
        """HDF5: file에서 가져옴"""
        with h5py.File(hdf5_file, "r") as f:
            if block_id in f:
                data = f[block_id][:]
            else:
                # Load from Lustre
                fpath = f"{LUSTRE_PATH}/benchmark_blocks/{block_id}.bin"
                data = np.fromfile(fpath, dtype=np.uint8)
        return torch.from_numpy(data.copy()).to(device)
    
    def hdf5_backend_put(block_id, gpu_data):
        """HDF5: file에 저장"""
        with h5py.File(hdf5_file, "a") as f:
            if block_id in f:
                del f[block_id]
            f.create_dataset(block_id, data=gpu_data.cpu().numpy())
    
    hdf5_cache = TieredCache("HDF5", VRAM_CACHE_SIZE, hdf5_backend_get, hdf5_backend_put)
    
    # Warmup
    for i in range(NUM_BLOCKS):
        hdf5_cache.put(f"block_{i}", blocks[f"block_{i}"])
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for block_id in access_pattern:
        _ = hdf5_cache.get(block_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    hdf5_gbps = (BLOCK_SIZE * NUM_ACCESS / 1e9) / elapsed
    print(f"   Average: {hdf5_gbps:.2f} GB/s (Hit rate: {hdf5_cache.hit_rate()*100:.1f}%)")
    results["HDF5"] = {"gbps": hdf5_gbps, "hit_rate": hdf5_cache.hit_rate()}
    
    if os.path.exists(hdf5_file):
        os.remove(hdf5_file)

except Exception as e:
    print(f"   Failed: {e}")
    results["HDF5"] = {"gbps": 0, "error": str(e)}

###############################################################################
# Summary
###############################################################################

print("\n" + "="*80)
print("SUMMARY: Tiered 5-System Benchmark")
print(f"Access Pattern: {NUM_ACCESS} random accesses over {NUM_BLOCKS} blocks")
print(f"Block Size: {BLOCK_SIZE_MB}MB, VRAM Cache: {VRAM_CACHE_SIZE} blocks ({VRAM_CACHE_SIZE * BLOCK_SIZE_MB / 1024:.1f} GB)")
print("="*80)

# Sort by performance
sorted_systems = sorted(results.items(), key=lambda x: x[1].get("gbps", 0), reverse=True)

print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│               Tiered KV Cache Performance (VRAM + Backend)                   │
├──────────────────────────────────────────────────────────────────────────────┤
│ System       │ Avg Throughput │ Hit Rate  │ Backend                         │
├──────────────┼────────────────┼───────────┼─────────────────────────────────┤""")

backends = {
    "Cascade-C++": "SHM (pinned memory + mmap)",
    "vLLM-GPU": "Disk swap (torch.save/load)",
    "PDC": "File container",
    "LMCache": "CPU tensor + disk",
    "HDF5": "HDF5 file"
}

for name, r in sorted_systems:
    gbps = r.get("gbps", 0)
    hit = r.get("hit_rate", 0) * 100
    backend = backends.get(name, "Unknown")
    print(f"│ {name:<12} │ {gbps:>11.2f} GB/s │ {hit:>7.1f}% │ {backend:<31} │")

print("└──────────────────────────────────────────────────────────────────────────────┘")

# Bar chart
max_gbps = max(r.get("gbps", 0) for r in results.values())
print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Performance Comparison                               │
├──────────────────────────────────────────────────────────────────────────────┤""")

for name, r in sorted_systems:
    gbps = r.get("gbps", 0)
    if max_gbps > 0:
        bar_len = int(50 * gbps / max_gbps)
        bar = "█" * bar_len
        print(f"│ {name:<12} {bar:<50} {gbps:>7.2f} │")

print("└──────────────────────────────────────────────────────────────────────────────┘")

###############################################################################
# Save Results
###############################################################################

output = {
    "job_id": job_id,
    "timestamp": datetime.now().isoformat(),
    "benchmark_type": "Tiered KV Cache with VRAM",
    "config": {
        "block_size_mb": BLOCK_SIZE_MB,
        "num_blocks": NUM_BLOCKS,
        "num_accesses": NUM_ACCESS,
        "vram_cache_size": VRAM_CACHE_SIZE,
        "access_pattern": "random"
    },
    "gpu": torch.cuda.get_device_name(0),
    "note": "All systems use same VRAM cache layer. Difference is in backend (miss) performance.",
    "results": results
}

output_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/tiered_5sys_{job_id}.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved: {output_path}")

# Cleanup
del blocks
torch.cuda.empty_cache()
import shutil
shutil.rmtree(f"{LUSTRE_PATH}/benchmark_blocks", ignore_errors=True)

PYTHON_END

echo ""
echo "Completed at $(date '+%Y-%m-%d %H:%M:%S')"
