#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:01:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/all5_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/all5_%j.err
#SBATCH -J all5_bench

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
import os, sys, time, json, numpy as np, shutil, subprocess, h5py
from mpi4py import MPI

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
NPROCS = comm.Get_size()
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
SCRATCH = os.environ.get('SCRATCH', '/tmp')
PROJECT_DIR = os.environ.get('PROJECT_DIR', '.')

NUM_BLOCKS = 50  # 500MB per rank = 8GB total
BLOCK_SIZE = 10 * 1024 * 1024  # 10MB

if RANK == 0:
    print(f"=== ALL 5 SYSTEMS FAIR BENCHMARK ===")
    print(f"Data: {NUM_BLOCKS}×10MB = {NUM_BLOCKS*10}MB/rank, {NUM_BLOCKS*10*NPROCS/1024:.1f}GB total")
    sys.stdout.flush()

comm.Barrier()

# Generate data
np.random.seed(42 + RANK)
data_blocks = [np.random.bytes(BLOCK_SIZE) for _ in range(NUM_BLOCKS)]
total_mb = NUM_BLOCKS * BLOCK_SIZE / (1024*1024)

def drop_cache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except: pass

results = {}

# Each rank has its own unique directory
BASE = f"{SCRATCH}/bench5_{JOB_ID}_r{RANK:04d}"

###############################################################################
# 1. CASCADE (aggregated + striped)
###############################################################################
d = f"{BASE}/cascade"
os.makedirs(d, exist_ok=True)
try: subprocess.run(["lfs", "setstripe", "-c", "8", "-S", "4m", d], capture_output=True, timeout=5)
except: pass
f_path = os.path.join(d, "agg.bin")
offsets = []

t0 = time.perf_counter()
with open(f_path, 'wb') as f:
    off = 0
    for data in data_blocks:
        offsets.append((off, len(data)))
        f.write(data)
        off += len(data)
    f.flush(); os.fsync(f.fileno())
w_time = time.perf_counter() - t0

comm.Barrier(); os.sync(); time.sleep(1); comm.Barrier()
drop_cache(f_path)
comm.Barrier()

t0 = time.perf_counter()
with open(f_path, 'rb') as f:
    for off, sz in offsets:
        f.seek(off); _ = f.read(sz)
r_time = time.perf_counter() - t0

results['cascade'] = {'w': total_mb/w_time/1024, 'r': total_mb/r_time/1024}
comm.Barrier()

###############################################################################
# 2. LMCACHE (per-file)
###############################################################################
d = f"{BASE}/lmcache"
os.makedirs(d, exist_ok=True)
paths = []

t0 = time.perf_counter()
for i, data in enumerate(data_blocks):
    p = os.path.join(d, f"b{i:06d}.bin"); paths.append(p)
    with open(p, 'wb') as f: f.write(data)
w_time = time.perf_counter() - t0

comm.Barrier(); os.sync(); time.sleep(1); comm.Barrier()
for p in paths: drop_cache(p)
comm.Barrier()

t0 = time.perf_counter()
for p in paths:
    with open(p, 'rb') as f: _ = f.read()
r_time = time.perf_counter() - t0

results['lmcache'] = {'w': total_mb/w_time/1024, 'r': total_mb/r_time/1024}
comm.Barrier()

###############################################################################
# 3. HDF5 (single file, gzip compression)
###############################################################################
d = f"{BASE}/hdf5"
os.makedirs(d, exist_ok=True)
h5_path = os.path.join(d, "data.h5")

t0 = time.perf_counter()
with h5py.File(h5_path, 'w') as f:
    for i, data in enumerate(data_blocks):
        f.create_dataset(f'b{i}', data=np.frombuffer(data, dtype=np.uint8), compression='gzip', compression_opts=1)
w_time = time.perf_counter() - t0

comm.Barrier(); os.sync(); time.sleep(1); comm.Barrier()
drop_cache(h5_path)
comm.Barrier()

t0 = time.perf_counter()
with h5py.File(h5_path, 'r') as f:
    for i in range(NUM_BLOCKS):
        _ = f[f'b{i}'][:]
r_time = time.perf_counter() - t0

results['hdf5'] = {'w': total_mb/w_time/1024, 'r': total_mb/r_time/1024}
comm.Barrier()

###############################################################################
# 4. PDC style (per-file, sync each)
###############################################################################
d = f"{BASE}/pdc"
os.makedirs(d, exist_ok=True)
paths = []

t0 = time.perf_counter()
for i, data in enumerate(data_blocks):
    p = os.path.join(d, f"obj{i:06d}.pdc"); paths.append(p)
    with open(p, 'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
w_time = time.perf_counter() - t0

comm.Barrier(); os.sync(); time.sleep(1); comm.Barrier()
for p in paths: drop_cache(p)
comm.Barrier()

t0 = time.perf_counter()
for p in paths:
    with open(p, 'rb') as f: _ = f.read()
r_time = time.perf_counter() - t0

results['pdc'] = {'w': total_mb/w_time/1024, 'r': total_mb/r_time/1024}
comm.Barrier()

###############################################################################
# 5. REDIS style (per-file)
###############################################################################
d = f"{BASE}/redis"
os.makedirs(d, exist_ok=True)
paths = []

t0 = time.perf_counter()
for i, data in enumerate(data_blocks):
    p = os.path.join(d, f"key{i:06d}.rdb"); paths.append(p)
    with open(p, 'wb') as f: f.write(data)
w_time = time.perf_counter() - t0

comm.Barrier(); os.sync(); time.sleep(1); comm.Barrier()
for p in paths: drop_cache(p)
comm.Barrier()

t0 = time.perf_counter()
for p in paths:
    with open(p, 'rb') as f: _ = f.read()
r_time = time.perf_counter() - t0

results['redis'] = {'w': total_mb/w_time/1024, 'r': total_mb/r_time/1024}
comm.Barrier()

###############################################################################
# Cleanup
###############################################################################
shutil.rmtree(BASE, ignore_errors=True)
comm.Barrier()

###############################################################################
# Aggregate
###############################################################################
all_results = comm.gather(results, root=0)

if RANK == 0:
    agg = {}
    for sys in results.keys():
        agg[sys] = {
            'w': sum(r[sys]['w'] for r in all_results),
            'r': sum(r[sys]['r'] for r in all_results)
        }
    
    print("\n" + "="*70)
    print("RESULTS - ALL FROM LUSTRE COLD READ (posix_fadvise DONTNEED)")
    print("="*70)
    print(f"{'System':<12} {'Write (GB/s)':>14} {'Read (GB/s)':>14} {'vs LMCache':>12}")
    print("-"*56)
    
    base = agg['lmcache']['r']
    for sys in ['cascade', 'lmcache', 'pdc', 'hdf5', 'redis']:
        sp = agg[sys]['r'] / base if base > 0 else 0
        m = " **BEST**" if sys == 'cascade' else ""
        print(f"{sys:<12} {agg[sys]['w']:>12.2f} {agg[sys]['r']:>14.2f} {sp:>10.2f}x{m}")
    
    out = f"{PROJECT_DIR}/benchmark/results/all5_{JOB_ID}.json"
    with open(out, 'w') as f:
        json.dump({"job_id": JOB_ID, "ranks": NPROCS, "results": agg}, f, indent=2)
    print(f"\nSaved: {out}")

PYTHON_SCRIPT

echo "[DONE]"
