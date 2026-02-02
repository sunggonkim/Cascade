#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/fair_large_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/fair_large_%j.err
#SBATCH -J fair_large

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
echo "FAIR LARGE BENCHMARK"
echo "Job ID: $JOB_ID, Nodes: $SLURM_NNODES, Ranks: $NPROCS"
echo "============================================"

srun python3 << 'PYTHON_SCRIPT'
import os
import sys
import time
import json
import numpy as np
import shutil
import subprocess
from mpi4py import MPI

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
NPROCS = comm.Get_size()
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
SCRATCH = os.environ.get('SCRATCH', '/tmp')
PROJECT_DIR = os.environ.get('PROJECT_DIR', '/pscratch/sd/s/sgkim/Skim-cascade')

NUM_BLOCKS = 500  # 5GB per rank = 80GB total
BLOCK_SIZE = 10 * 1024 * 1024  # 10MB

if RANK == 0:
    print(f"\n{'='*60}")
    print("FAIR LARGE BENCHMARK: Lustre Cold Read")
    print(f"Data: {NUM_BLOCKS} blocks × 10MB = {NUM_BLOCKS*10/1024:.1f}GB per rank")
    print(f"Total: {NUM_BLOCKS*10*NPROCS/1024:.1f}GB across {NPROCS} ranks")
    print(f"{'='*60}")
    sys.stdout.flush()

comm.Barrier()

# Generate data
np.random.seed(42 + RANK)
data_blocks = [np.random.bytes(BLOCK_SIZE) for _ in range(NUM_BLOCKS)]

if RANK == 0:
    print(f"[PHASE] Data generation complete")
    sys.stdout.flush()

comm.Barrier()

###############################################################################
# TEST 1: LMCache style (per-file)
###############################################################################
lmc_dir = f"{SCRATCH}/lmc_large_{JOB_ID}/rank_{RANK:04d}"
os.makedirs(lmc_dir, exist_ok=True)

# WRITE
lmc_write_start = time.perf_counter()
paths = []
for i, data in enumerate(data_blocks):
    path = os.path.join(lmc_dir, f"b{i:06d}.bin")
    paths.append(path)
    with open(path, 'wb') as f:
        f.write(data)
lmc_write_time = time.perf_counter() - lmc_write_start

if RANK == 0:
    print(f"[PHASE] LMCache write: {lmc_write_time:.2f}s")
    sys.stdout.flush()

# Barrier + sync
comm.Barrier()
os.sync()
time.sleep(3)
comm.Barrier()

# DROP CACHE
for path in paths:
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except:
        pass

comm.Barrier()

# COLD READ
lmc_read_start = time.perf_counter()
for path in paths:
    with open(path, 'rb') as f:
        _ = f.read()
lmc_read_time = time.perf_counter() - lmc_read_start

if RANK == 0:
    print(f"[PHASE] LMCache cold read: {lmc_read_time:.2f}s")
    sys.stdout.flush()

comm.Barrier()
shutil.rmtree(os.path.dirname(lmc_dir), ignore_errors=True)
comm.Barrier()

###############################################################################
# TEST 2: Cascade style (aggregated + striped)
###############################################################################
cascade_dir = f"{SCRATCH}/casc_large_{JOB_ID}/rank_{RANK:04d}"
os.makedirs(cascade_dir, exist_ok=True)

try:
    subprocess.run(["lfs", "setstripe", "-c", "8", "-S", "4m", cascade_dir], 
                   capture_output=True, timeout=5)
except:
    pass

agg_file = os.path.join(cascade_dir, "agg.bin")

# WRITE
cascade_write_start = time.perf_counter()
offsets = []
with open(agg_file, 'wb') as f:
    off = 0
    for data in data_blocks:
        offsets.append((off, len(data)))
        f.write(data)
        off += len(data)
    f.flush()
    os.fsync(f.fileno())
cascade_write_time = time.perf_counter() - cascade_write_start

if RANK == 0:
    print(f"[PHASE] Cascade write: {cascade_write_time:.2f}s")
    sys.stdout.flush()

comm.Barrier()
os.sync()
time.sleep(3)
comm.Barrier()

# DROP CACHE
try:
    fd = os.open(agg_file, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
except:
    pass

comm.Barrier()

# COLD READ
cascade_read_start = time.perf_counter()
with open(agg_file, 'rb') as f:
    for off, sz in offsets:
        f.seek(off)
        _ = f.read(sz)
cascade_read_time = time.perf_counter() - cascade_read_start

if RANK == 0:
    print(f"[PHASE] Cascade cold read: {cascade_read_time:.2f}s")
    sys.stdout.flush()

comm.Barrier()
shutil.rmtree(os.path.dirname(cascade_dir), ignore_errors=True)
comm.Barrier()

###############################################################################
# Results
###############################################################################
total_mb = NUM_BLOCKS * BLOCK_SIZE / (1024 * 1024)
results = {
    "lmc_write": total_mb / lmc_write_time / 1024,
    "lmc_read": total_mb / lmc_read_time / 1024,
    "cascade_write": total_mb / cascade_write_time / 1024,
    "cascade_read": total_mb / cascade_read_time / 1024,
}

all_results = comm.gather(results, root=0)

if RANK == 0:
    avg = {k: sum(r[k] for r in all_results)/len(all_results) for k in results}
    
    print("\n" + "="*70)
    print("RESULTS - FAIR: Both from Lustre Cold Read (80GB total)")
    print("="*70)
    
    lw = avg["lmc_write"] * NPROCS
    lr = avg["lmc_read"] * NPROCS  
    cw = avg["cascade_write"] * NPROCS
    cr = avg["cascade_read"] * NPROCS
    
    print(f"LMCache (per-file):   Write {lw:.2f} GB/s, Read {lr:.2f} GB/s")
    print(f"Cascade (aggregated): Write {cw:.2f} GB/s, Read {cr:.2f} GB/s")
    print(f"Speedup: Write {cw/lw:.2f}x, Read {cr/lr:.2f}x")
    
    out = f"{PROJECT_DIR}/benchmark/results/fair_large_{JOB_ID}.json"
    with open(out, 'w') as f:
        json.dump({
            "job_id": JOB_ID, "ranks": NPROCS,
            "total_data_gb": NUM_BLOCKS * 10 * NPROCS / 1024,
            "lmcache_write": lw, "lmcache_read": lr,
            "cascade_write": cw, "cascade_read": cr,
            "speedup_write": cw/lw, "speedup_read": cr/lr
        }, f, indent=2)
    print(f"Saved: {out}")

PYTHON_SCRIPT

echo "[DONE]"
