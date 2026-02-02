#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:25:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/fair_bench_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/fair_bench_%j.err
#SBATCH -J fair_bench

###############################################################################
# FAIR BENCHMARK - Both systems read from SAME storage tier (Lustre)
# No cheating: Both Cascade and LMCache write to and read from Lustre
###############################################################################

set -e

export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade

module load python/3.11
module load cudatoolkit
module load cray-mpich

cd $PROJECT_DIR
mkdir -p benchmark/results benchmark/logs

JOB_ID=$SLURM_JOB_ID
NPROCS=$SLURM_NTASKS

echo "============================================"
echo "FAIR BENCHMARK - Both from Lustre Cold Read"
echo "Job ID: $JOB_ID, Nodes: $SLURM_NNODES, Ranks: $NPROCS"
echo "============================================"

###############################################################################
# Python Benchmark - FAIR COMPARISON
###############################################################################
srun python3 << 'PYTHON_SCRIPT'
import os
import sys
import time
import json
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional
import shutil
import subprocess

RANK = int(os.environ.get('SLURM_PROCID', 0))
NPROCS = int(os.environ.get('SLURM_NTASKS', 1))
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
SCRATCH = os.environ.get('SCRATCH', '/tmp')
PROJECT_DIR = os.environ.get('PROJECT_DIR', '/pscratch/sd/s/sgkim/Skim-cascade')

# Large data sizes: 2GB, 5GB per rank
# 4 nodes × 4 ranks = 16 ranks → 32GB, 80GB total
SCENARIOS = [
    {"name": "2GB_per_rank", "num_blocks": 200, "block_size_mb": 10},   # 2GB/rank, 32GB total
    {"name": "5GB_per_rank", "num_blocks": 500, "block_size_mb": 10},   # 5GB/rank, 80GB total
]

###############################################################################
# Lustre Store - Per-file (LMCache style)
###############################################################################
class LustrePerFile:
    """LMCache style: one file per block."""
    
    def __init__(self, base_dir: str):
        self.lustre_dir = os.path.join(base_dir, f"rank_{RANK:04d}")
        os.makedirs(self.lustre_dir, exist_ok=True)
        self.index = {}
    
    def put(self, block_id: str, data: bytes) -> float:
        start = time.perf_counter()
        path = os.path.join(self.lustre_dir, f"{block_id}.bin")
        with open(path, 'wb') as f:
            f.write(data)
        # Force to disk
        fd = os.open(path, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
        self.index[block_id] = path
        return time.perf_counter() - start
    
    def drop_cache_and_read(self, block_id: str) -> Tuple[Optional[bytes], float]:
        if block_id not in self.index:
            return None, 0.0
        path = self.index[block_id]
        # Drop page cache
        try:
            fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except:
            pass
        start = time.perf_counter()
        with open(path, 'rb') as f:
            data = f.read()
        return data, time.perf_counter() - start

###############################################################################
# Lustre Store - Aggregated (Cascade style)
###############################################################################
class LustreAggregated:
    """Cascade style: aggregated file with Lustre striping."""
    
    def __init__(self, base_dir: str):
        self.lustre_dir = os.path.join(base_dir, f"rank_{RANK:04d}")
        os.makedirs(self.lustre_dir, exist_ok=True)
        # Set Lustre striping
        try:
            subprocess.run(
                ["lfs", "setstripe", "-c", "8", "-S", "4m", self.lustre_dir],
                capture_output=True, timeout=10
            )
        except:
            pass
        self.agg_file = os.path.join(self.lustre_dir, "aggregated.bin")
        self.index = {}
        self.offset = 0
        self.fh = None
    
    def open_write(self):
        self.fh = open(self.agg_file, 'wb')
        self.offset = 0
    
    def put(self, block_id: str, data: bytes) -> float:
        start = time.perf_counter()
        self.index[block_id] = (self.offset, len(data))
        self.fh.write(data)
        self.offset += len(data)
        return time.perf_counter() - start
    
    def close_write(self) -> float:
        start = time.perf_counter()
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.fh.close()
        self.fh = None
        return time.perf_counter() - start
    
    def drop_cache_and_read(self, block_id: str) -> Tuple[Optional[bytes], float]:
        if block_id not in self.index:
            return None, 0.0
        offset, size = self.index[block_id]
        # Drop page cache
        try:
            fd = os.open(self.agg_file, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except:
            pass
        start = time.perf_counter()
        with open(self.agg_file, 'rb') as f:
            f.seek(offset)
            data = f.read(size)
        return data, time.perf_counter() - start

###############################################################################
# Run Benchmark
###############################################################################
def run_scenario(scenario: dict) -> dict:
    name = scenario["name"]
    num_blocks = scenario["num_blocks"]
    block_size = scenario["block_size_mb"] * 1024 * 1024
    total_mb = num_blocks * scenario["block_size_mb"]
    
    if RANK == 0:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {name}")
        print(f"  {num_blocks} blocks × {scenario['block_size_mb']}MB = {total_mb}MB/rank")
        print(f"  Total: {total_mb * NPROCS / 1024:.1f} GB across {NPROCS} ranks")
        print(f"{'='*70}")
    
    # Generate blocks
    np.random.seed(42 + RANK)
    blocks = []
    for i in range(num_blocks):
        data = np.random.bytes(block_size)
        block_id = hashlib.sha256(data).hexdigest()[:32]
        blocks.append((block_id, data))
    
    # ===== CASCADE (Aggregated) =====
    cascade_dir = f"{SCRATCH}/cascade_fair_{JOB_ID}_{name}"
    cascade = LustreAggregated(cascade_dir)
    cascade.open_write()
    
    cascade_write_times = [cascade.put(bid, data) for bid, data in blocks]
    flush_time = cascade.close_write()
    cascade_write_total = sum(cascade_write_times) + flush_time
    cascade_write_bw = total_mb / cascade_write_total / 1024
    
    cascade_read_times = [cascade.drop_cache_and_read(bid)[1] for bid, _ in blocks]
    cascade_read_total = sum(cascade_read_times)
    cascade_read_bw = total_mb / cascade_read_total / 1024
    
    # ===== LMCACHE (Per-file) =====
    lmc_dir = f"{SCRATCH}/lmcache_fair_{JOB_ID}_{name}"
    lmcache = LustrePerFile(lmc_dir)
    
    lmc_write_times = [lmcache.put(bid, data) for bid, data in blocks]
    lmc_write_total = sum(lmc_write_times)
    lmc_write_bw = total_mb / lmc_write_total / 1024
    
    lmc_read_times = [lmcache.drop_cache_and_read(bid)[1] for bid, _ in blocks]
    lmc_read_total = sum(lmc_read_times)
    lmc_read_bw = total_mb / lmc_read_total / 1024
    
    result = {
        "scenario": name,
        "total_mb": total_mb,
        "cascade_write_gbps": cascade_write_bw,
        "cascade_read_gbps": cascade_read_bw,
        "lmcache_write_gbps": lmc_write_bw,
        "lmcache_read_gbps": lmc_read_bw,
    }
    
    if RANK == 0:
        ws = cascade_write_bw / lmc_write_bw if lmc_write_bw > 0 else 0
        rs = cascade_read_bw / lmc_read_bw if lmc_read_bw > 0 else 0
        print(f"\n[{name}] Per-Rank Bandwidth (GB/s):")
        print(f"  CASCADE (agg):    Write {cascade_write_bw:.3f}, Read {cascade_read_bw:.3f}")
        print(f"  LMCACHE (file):   Write {lmc_write_bw:.3f}, Read {lmc_read_bw:.3f}")
        print(f"  Speedup:          Write {ws:.2f}x, Read {rs:.2f}x")
    
    # Cleanup
    shutil.rmtree(cascade_dir, ignore_errors=True)
    shutil.rmtree(lmc_dir, ignore_errors=True)
    
    return result

def main():
    if RANK == 0:
        print("\n" + "="*70)
        print("FAIR BENCHMARK - Both from Lustre (Cold Read)")
        print("="*70)
        print("Cascade: Aggregated writes + stripe (-c 8 -S 4m)")
        print("LMCache: Per-file writes (1 file per block)")
        print("Both: posix_fadvise(DONTNEED) = TRUE cold read")
    
    results = [run_scenario(s) for s in SCENARIOS]
    
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    all_results = comm.gather(results, root=0)
    
    if RANK == 0:
        avg_results = []
        for i, s in enumerate(SCENARIOS):
            avg = {"scenario": s["name"], "total_mb": s["num_blocks"] * s["block_size_mb"]}
            for key in ["cascade_write_gbps", "cascade_read_gbps", "lmcache_write_gbps", "lmcache_read_gbps"]:
                avg[key] = sum(r[i][key] for r in all_results) / len(all_results)
            avg_results.append(avg)
        
        print("\n" + "="*80)
        print("AGGREGATED RESULTS (16 ranks) - FAIR: Both from Lustre Cold")
        print("="*80)
        print(f"\n{'Scenario':<18} {'Total':<10} {'Cascade':>24} {'LMCache':>24} {'Speedup':>10}")
        print(f"{'':18} {'Data':10} {'Write':>12} {'Read':>12} {'Write':>12} {'Read':>12} {'Read':>10}")
        print("-"*80)
        
        for r in avg_results:
            total_gb = r["total_mb"] * NPROCS / 1024
            cw = r["cascade_write_gbps"] * NPROCS
            cr = r["cascade_read_gbps"] * NPROCS
            lw = r["lmcache_write_gbps"] * NPROCS
            lr = r["lmcache_read_gbps"] * NPROCS
            speedup = cr / lr if lr > 0 else 0
            print(f"{r['scenario']:<18} {total_gb:.0f}GB{'':<5} {cw:>10.2f}GB/s {cr:>10.2f}GB/s {lw:>10.2f}GB/s {lr:>10.2f}GB/s {speedup:>8.2f}x")
        
        out = f"{PROJECT_DIR}/benchmark/results/fair_bench_{JOB_ID}.json"
        with open(out, 'w') as f:
            json.dump({"job_id": JOB_ID, "ranks": NPROCS, "results": avg_results}, f, indent=2)
        print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

echo "[DONE] Fair benchmark completed."
