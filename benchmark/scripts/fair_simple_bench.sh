#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/fair_simple_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/fair_simple_%j.err
#SBATCH -J fair_simple

###############################################################################
# SIMPLE FAIR BENCHMARK - Both from Lustre Cold Read
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
echo "FAIR SIMPLE BENCHMARK"
echo "Job ID: $JOB_ID, Nodes: $SLURM_NNODES, Ranks: $NPROCS"
echo "============================================"

srun python3 << 'PYTHON_SCRIPT'
import os
import sys
import time
import json
import hashlib
import numpy as np
import shutil
import subprocess

RANK = int(os.environ.get('SLURM_PROCID', 0))
NPROCS = int(os.environ.get('SLURM_NTASKS', 1))
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
SCRATCH = os.environ.get('SCRATCH', '/tmp')
PROJECT_DIR = os.environ.get('PROJECT_DIR', '/pscratch/sd/s/sgkim/Skim-cascade')

NUM_BLOCKS = 300  # 3GB per rank = 48GB total
BLOCK_SIZE = 10 * 1024 * 1024  # 10MB

if RANK == 0:
    print(f"\n{'='*60}")
    print("FAIR BENCHMARK: Both from Lustre Cold Read")
    print(f"Data: {NUM_BLOCKS} blocks × 10MB = {NUM_BLOCKS*10/1024:.1f}GB per rank")
    print(f"Total: {NUM_BLOCKS*10*NPROCS/1024:.1f}GB across {NPROCS} ranks")
    print(f"{'='*60}")

# Generate data
np.random.seed(42 + RANK)
blocks = []
for i in range(NUM_BLOCKS):
    data = np.random.bytes(BLOCK_SIZE)
    block_id = f"block_{RANK:04d}_{i:06d}"  # Simple predictable ID
    blocks.append((block_id, data))

if RANK == 0:
    print(f"[RANK {RANK}] Generated {NUM_BLOCKS} blocks")

###############################################################################
# TEST 1: LMCache style (per-file)
###############################################################################
lmc_dir = f"{SCRATCH}/lmc_{JOB_ID}/rank_{RANK:04d}"
os.makedirs(lmc_dir, exist_ok=True)

# Write
lmc_write_start = time.perf_counter()
for bid, data in blocks:
    path = os.path.join(lmc_dir, f"{bid}.bin")
    with open(path, 'wb') as f:
        f.write(data)
os.sync()  # Global sync
lmc_write_time = time.perf_counter() - lmc_write_start

# Cold read (drop cache first)
for bid, _ in blocks:
    path = os.path.join(lmc_dir, f"{bid}.bin")
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except:
        pass

lmc_read_start = time.perf_counter()
for bid, _ in blocks:
    path = os.path.join(lmc_dir, f"{bid}.bin")
    with open(path, 'rb') as f:
        _ = f.read()
lmc_read_time = time.perf_counter() - lmc_read_start

# Cleanup
shutil.rmtree(os.path.dirname(lmc_dir), ignore_errors=True)

###############################################################################
# TEST 2: Cascade style (aggregated + striped)
###############################################################################
cascade_dir = f"{SCRATCH}/casc_{JOB_ID}/rank_{RANK:04d}"
os.makedirs(cascade_dir, exist_ok=True)

# Set Lustre striping
try:
    subprocess.run(["lfs", "setstripe", "-c", "8", "-S", "4m", cascade_dir], 
                   capture_output=True, timeout=5)
except:
    pass

agg_file = os.path.join(cascade_dir, "agg.bin")
index = {}  # bid -> (offset, size)

# Write aggregated
cascade_write_start = time.perf_counter()
with open(agg_file, 'wb') as f:
    offset = 0
    for bid, data in blocks:
        index[bid] = (offset, len(data))
        f.write(data)
        offset += len(data)
    f.flush()
    os.fsync(f.fileno())
cascade_write_time = time.perf_counter() - cascade_write_start

os.sync()

# Cold read (drop cache)
try:
    fd = os.open(agg_file, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
except:
    pass

cascade_read_start = time.perf_counter()
with open(agg_file, 'rb') as f:
    for bid, _ in blocks:
        off, sz = index[bid]
        f.seek(off)
        _ = f.read(sz)
cascade_read_time = time.perf_counter() - cascade_read_start

# Cleanup
shutil.rmtree(os.path.dirname(cascade_dir), ignore_errors=True)

###############################################################################
# Calculate bandwidth
###############################################################################
total_mb = NUM_BLOCKS * BLOCK_SIZE / (1024 * 1024)

lmc_write_bw = total_mb / lmc_write_time / 1024  # GB/s
lmc_read_bw = total_mb / lmc_read_time / 1024
cascade_write_bw = total_mb / cascade_write_time / 1024
cascade_read_bw = total_mb / cascade_read_time / 1024

if RANK == 0:
    print(f"\n[RANK 0] Per-Rank Results:")
    print(f"  LMCache (per-file):     Write {lmc_write_bw:.3f} GB/s, Read {lmc_read_bw:.3f} GB/s")
    print(f"  Cascade (aggregated):   Write {cascade_write_bw:.3f} GB/s, Read {cascade_read_bw:.3f} GB/s")

###############################################################################
# Aggregate across ranks
###############################################################################
from mpi4py import MPI
comm = MPI.COMM_WORLD

results = {
    "lmc_write": lmc_write_bw,
    "lmc_read": lmc_read_bw,
    "cascade_write": cascade_write_bw,
    "cascade_read": cascade_read_bw,
}

all_results = comm.gather(results, root=0)

if RANK == 0:
    # Average
    avg = {}
    for key in results.keys():
        avg[key] = sum(r[key] for r in all_results) / len(all_results)
    
    print("\n" + "="*70)
    print("AGGREGATED RESULTS (16 ranks) - FAIR: Both from Lustre Cold")
    print("="*70)
    
    total_data_gb = total_mb * NPROCS / 1024
    print(f"\nTotal data: {total_data_gb:.1f} GB")
    
    lw = avg["lmc_write"] * NPROCS
    lr = avg["lmc_read"] * NPROCS
    cw = avg["cascade_write"] * NPROCS
    cr = avg["cascade_read"] * NPROCS
    
    print(f"\nLMCache (per-file):     Write {lw:.2f} GB/s, Read {lr:.2f} GB/s")
    print(f"Cascade (aggregated):   Write {cw:.2f} GB/s, Read {cr:.2f} GB/s")
    
    write_speedup = cw / lw if lw > 0 else 0
    read_speedup = cr / lr if lr > 0 else 0
    
    print(f"\n>>> Cascade Speedup: Write {write_speedup:.2f}x, Read {read_speedup:.2f}x")
    
    # Save
    out = f"{PROJECT_DIR}/benchmark/results/fair_simple_{JOB_ID}.json"
    with open(out, 'w') as f:
        json.dump({
            "job_id": JOB_ID,
            "ranks": NPROCS,
            "total_data_gb": total_data_gb,
            "lmcache_write_gbps": lw,
            "lmcache_read_gbps": lr,
            "cascade_write_gbps": cw,
            "cascade_read_gbps": cr,
            "write_speedup": write_speedup,
            "read_speedup": read_speedup,
        }, f, indent=2)
    print(f"\nSaved: {out}")

PYTHON_SCRIPT

echo "[DONE]"
