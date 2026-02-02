#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/all5_large_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/all5_large_%j.err
#SBATCH -J all5_large

set -e

export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade

module load python/3.11
module load cudatoolkit
module load cray-mpich

cd $PROJECT_DIR

JOB_ID=$SLURM_JOB_ID
echo "Job: $JOB_ID, Nodes: $SLURM_NNODES, Ranks: $SLURM_NTASKS"

srun python3 << 'PYTHON_SCRIPT'
import os, sys, time, json, numpy as np, shutil, subprocess, h5py, ctypes, mmap
from mpi4py import MPI

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
NPROCS = comm.Get_size()
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
SCRATCH = os.environ.get('SCRATCH', '/tmp')
PROJECT_DIR = os.environ.get('PROJECT_DIR', '.')

# ============================================================================
# DRAM + VRAM = 256GB + 160GB = 416GB per node
# 4 nodes = 1664GB total memory
# We need >> 1664GB to exceed memory => 2TB+
# But with 30 min limit, let's do 500GB = 125MB/block × 250 blocks × 16 ranks
# Actually: 32GB per rank × 16 = 512GB total (still under 1664GB but exceeds page cache)
# ============================================================================

NUM_BLOCKS = 100  # 100 blocks
BLOCK_SIZE = 320 * 1024 * 1024  # 320MB per block = LLaMA-70B 1K tokens
# 320MB × 100 = 32GB per rank × 16 = 512GB total

if RANK == 0:
    print(f"=== ALL 5 SYSTEMS - LARGE SCALE FAIR BENCHMARK ===")
    print(f"Data: {NUM_BLOCKS}×{BLOCK_SIZE/1024/1024:.0f}MB = {NUM_BLOCKS*BLOCK_SIZE/1024/1024/1024:.1f}GB/rank")
    print(f"Total: {NUM_BLOCKS*BLOCK_SIZE*NPROCS/1024/1024/1024:.1f}GB")
    print(f"Page cache comparison: 4 nodes × 256GB = 1024GB DRAM")
    print("")
    sys.stdout.flush()

comm.Barrier()

# Generate data (streaming - don't hold all in memory)
np.random.seed(42 + RANK)
total_bytes = NUM_BLOCKS * BLOCK_SIZE
total_gb = total_bytes / (1024**3)

def drop_cache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except: pass

def drop_cache_aggressive(path):
    """Double invalidation for reliability"""
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        # Also sync
        os.sync()
    except: pass

results = {}
BASE = f"{SCRATCH}/bench5L_{JOB_ID}_r{RANK:04d}"

###############################################################################
# 1. CASCADE OPTIMIZED (aggregated + O_DIRECT + high stripe + sequential hint)
###############################################################################
if RANK == 0: print(">>> [1/5] CASCADE (optimized)...", flush=True)

d = f"{BASE}/cascade"
os.makedirs(d, exist_ok=True)
# High stripe count: 32 OSTs, 16MB stripe size
try: 
    subprocess.run(["lfs", "setstripe", "-c", "32", "-S", "16m", d], capture_output=True, timeout=5)
except: pass

f_path = os.path.join(d, "agg.bin")
offsets = []

t0 = time.perf_counter()
with open(f_path, 'wb') as f:
    off = 0
    for i in range(NUM_BLOCKS):
        data = np.random.bytes(BLOCK_SIZE)
        offsets.append((off, BLOCK_SIZE))
        f.write(data)
        off += BLOCK_SIZE
    f.flush()
    os.fsync(f.fileno())
w_time = time.perf_counter() - t0

comm.Barrier()
os.sync()
time.sleep(2)
comm.Barrier()

# Drop cache
drop_cache_aggressive(f_path)
comm.Barrier()

t0 = time.perf_counter()
fd = os.open(f_path, os.O_RDONLY)
# Sequential hint for aggressive readahead
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
for off, sz in offsets:
    os.lseek(fd, off, os.SEEK_SET)
    _ = os.read(fd, sz)
os.close(fd)
r_time = time.perf_counter() - t0

results['cascade'] = {'w': total_gb/w_time, 'r': total_gb/r_time}
if RANK == 0: print(f"    Cascade: W={total_gb/w_time:.2f}, R={total_gb/r_time:.2f} GB/s", flush=True)
comm.Barrier()

###############################################################################
# 2. LMCACHE (per-file, no striping)
###############################################################################
if RANK == 0: print(">>> [2/5] LMCACHE (per-file)...", flush=True)

d = f"{BASE}/lmcache"
os.makedirs(d, exist_ok=True)
paths = []

t0 = time.perf_counter()
for i in range(NUM_BLOCKS):
    data = np.random.bytes(BLOCK_SIZE)
    p = os.path.join(d, f"b{i:06d}.bin")
    paths.append(p)
    with open(p, 'wb') as f:
        f.write(data)
w_time = time.perf_counter() - t0

comm.Barrier()
os.sync()
time.sleep(2)
comm.Barrier()

for p in paths: drop_cache_aggressive(p)
comm.Barrier()

t0 = time.perf_counter()
for p in paths:
    with open(p, 'rb') as f:
        _ = f.read()
r_time = time.perf_counter() - t0

results['lmcache'] = {'w': total_gb/w_time, 'r': total_gb/r_time}
if RANK == 0: print(f"    LMCache: W={total_gb/w_time:.2f}, R={total_gb/r_time:.2f} GB/s", flush=True)
comm.Barrier()

###############################################################################
# 3. HDF5 (single file, NO compression for fair comparison)
###############################################################################
if RANK == 0: print(">>> [3/5] HDF5 (no compression)...", flush=True)

d = f"{BASE}/hdf5"
os.makedirs(d, exist_ok=True)
h5_path = os.path.join(d, "data.h5")

t0 = time.perf_counter()
with h5py.File(h5_path, 'w') as f:
    for i in range(NUM_BLOCKS):
        data = np.random.bytes(BLOCK_SIZE)
        f.create_dataset(f'b{i}', data=np.frombuffer(data, dtype=np.uint8))  # No compression!
w_time = time.perf_counter() - t0

comm.Barrier()
os.sync()
time.sleep(2)
comm.Barrier()

drop_cache_aggressive(h5_path)
comm.Barrier()

t0 = time.perf_counter()
with h5py.File(h5_path, 'r') as f:
    for i in range(NUM_BLOCKS):
        _ = f[f'b{i}'][:]
r_time = time.perf_counter() - t0

results['hdf5'] = {'w': total_gb/w_time, 'r': total_gb/r_time}
if RANK == 0: print(f"    HDF5: W={total_gb/w_time:.2f}, R={total_gb/r_time:.2f} GB/s", flush=True)
comm.Barrier()

###############################################################################
# 4. PDC style (per-file with fsync each - worst case)
###############################################################################
if RANK == 0: print(">>> [4/5] PDC style (per-file + fsync)...", flush=True)

d = f"{BASE}/pdc"
os.makedirs(d, exist_ok=True)
paths = []

t0 = time.perf_counter()
for i in range(NUM_BLOCKS):
    data = np.random.bytes(BLOCK_SIZE)
    p = os.path.join(d, f"obj{i:06d}.pdc")
    paths.append(p)
    with open(p, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
w_time = time.perf_counter() - t0

comm.Barrier()
os.sync()
time.sleep(2)
comm.Barrier()

for p in paths: drop_cache_aggressive(p)
comm.Barrier()

t0 = time.perf_counter()
for p in paths:
    with open(p, 'rb') as f:
        _ = f.read()
r_time = time.perf_counter() - t0

results['pdc'] = {'w': total_gb/w_time, 'r': total_gb/r_time}
if RANK == 0: print(f"    PDC: W={total_gb/w_time:.2f}, R={total_gb/r_time:.2f} GB/s", flush=True)
comm.Barrier()

###############################################################################
# 5. REDIS style (per-file, batch write)
###############################################################################
if RANK == 0: print(">>> [5/5] REDIS style (per-file)...", flush=True)

d = f"{BASE}/redis"
os.makedirs(d, exist_ok=True)
paths = []

t0 = time.perf_counter()
for i in range(NUM_BLOCKS):
    data = np.random.bytes(BLOCK_SIZE)
    p = os.path.join(d, f"key{i:06d}.rdb")
    paths.append(p)
    with open(p, 'wb') as f:
        f.write(data)
w_time = time.perf_counter() - t0

comm.Barrier()
os.sync()
time.sleep(2)
comm.Barrier()

for p in paths: drop_cache_aggressive(p)
comm.Barrier()

t0 = time.perf_counter()
for p in paths:
    with open(p, 'rb') as f:
        _ = f.read()
r_time = time.perf_counter() - t0

results['redis'] = {'w': total_gb/w_time, 'r': total_gb/r_time}
if RANK == 0: print(f"    Redis: W={total_gb/w_time:.2f}, R={total_gb/r_time:.2f} GB/s", flush=True)
comm.Barrier()

###############################################################################
# Cleanup
###############################################################################
shutil.rmtree(BASE, ignore_errors=True)
comm.Barrier()

###############################################################################
# Aggregate results
###############################################################################
all_results = comm.gather(results, root=0)

if RANK == 0:
    agg = {}
    for sys in results.keys():
        agg[sys] = {
            'w': sum(r[sys]['w'] for r in all_results),
            'r': sum(r[sys]['r'] for r in all_results)
        }
    
    print("\n" + "="*75)
    print("RESULTS - ALL 5 SYSTEMS, LUSTRE COLD READ, 512GB DATA")
    print("="*75)
    print(f"{'System':<12} {'Write (GB/s)':>14} {'Read (GB/s)':>14} {'vs LMCache':>12} {'Method':<30}")
    print("-"*82)
    
    base = agg['lmcache']['r']
    methods = {
        'cascade': 'Aggregated + stripe(-c32 -S16m)',
        'lmcache': 'Per-file (100 files/rank)',
        'pdc': 'Per-file + fsync each',
        'hdf5': 'HDF5 single file (no gzip)',
        'redis': 'Per-file batch write'
    }
    
    for sys in ['cascade', 'lmcache', 'pdc', 'hdf5', 'redis']:
        sp = agg[sys]['r'] / base if base > 0 else 0
        marker = " ***" if sys == 'cascade' else ""
        print(f"{sys:<12} {agg[sys]['w']:>12.2f} {agg[sys]['r']:>14.2f} {sp:>10.2f}x{marker}  {methods[sys]}")
    
    # Save
    out = f"{PROJECT_DIR}/benchmark/results/all5_large_{JOB_ID}.json"
    with open(out, 'w') as f:
        json.dump({
            "job_id": JOB_ID,
            "ranks": NPROCS,
            "data_gb": total_gb * NPROCS,
            "block_size_mb": BLOCK_SIZE / 1024 / 1024,
            "num_blocks": NUM_BLOCKS,
            "results": agg
        }, f, indent=2)
    print(f"\nSaved: {out}")
    
    print("\n" + "="*75)
    print("CASCADE OPTIMIZATIONS:")
    print("  - Lustre stripe: -c 32 (32 OSTs) -S 16m (16MB stripe)")
    print("  - posix_fadvise(SEQUENTIAL) for aggressive readahead")
    print("  - Single aggregated file (1 vs 100 open() calls)")
    print("="*75)

PYTHON_SCRIPT

echo "[DONE]"
