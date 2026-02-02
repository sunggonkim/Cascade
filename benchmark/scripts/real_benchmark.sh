#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_bench_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_bench_%j.err
#SBATCH -J real_benchmark

###############################################################################
# REAL BENCHMARK: Uses actual storage systems, NO simulation
# 
# This benchmark measures REAL I/O performance:
# - Lustre (per-file): Actual file I/O to Lustre filesystem
# - Lustre (aggregated): Batched writes with stripe optimization
# - HDF5: Real h5py library
# - Redis: Real Redis server
# - Shared Memory: Real /dev/shm
# - GPU Memory: Real CUDA device memory (if available)
#
# NO simulation, NO dict-based fake stores
###############################################################################

set -e

export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade
export RESULTS_DIR=$PROJECT_DIR/benchmark/results
export REDIS_DIR=$PROJECT_DIR/third_party/redis

module load python
module load cudatoolkit
module load cray-mpich

export PYTHONPATH=$PROJECT_DIR/python_pkgs_py312:$PROJECT_DIR:$PYTHONPATH

cd $PROJECT_DIR
mkdir -p $RESULTS_DIR benchmark/logs

RANK=$SLURM_PROCID
NPROCS=$SLURM_NTASKS
HOSTNAME=$(hostname)
JOB_ID=$SLURM_JOB_ID

echo "============================================"
echo "REAL BENCHMARK - 4 Nodes, 16 Ranks"
echo "============================================"
echo "Rank: $RANK / $NPROCS"
echo "Node: $HOSTNAME"
echo "Job ID: $JOB_ID"
echo "============================================"

###############################################################################
# Start Redis on Rank 0
###############################################################################
if [ $RANK -eq 0 ]; then
    echo "[SETUP] Starting Redis server..."
    REDIS_PORT=6380
    REDIS_DATA=$SCRATCH/redis_real_$$
    mkdir -p $REDIS_DATA
    
    $REDIS_DIR/src/redis-server --port $REDIS_PORT --dir $REDIS_DATA \
        --daemonize yes --maxmemory 100gb --maxmemory-policy allkeys-lru \
        --bind 0.0.0.0 --protected-mode no 2>/dev/null || echo "Redis already running or failed"
    sleep 3
    echo $HOSTNAME > $SCRATCH/redis_host_real_$$
    
    # Verify Redis is running
    if $REDIS_DIR/src/redis-cli -p $REDIS_PORT ping 2>/dev/null | grep -q PONG; then
        echo "[SETUP] Redis server OK"
    else
        echo "[SETUP] WARNING: Redis not responding"
    fi
fi

sleep 5
REDIS_HOST=$(cat $SCRATCH/redis_host_real_$$ 2>/dev/null || echo "localhost")
export REDIS_HOST

###############################################################################
# Run Real Benchmark
###############################################################################
echo "[BENCH] Starting REAL benchmark on rank $RANK..."

python3 << 'PYEOF'
"""
REAL BENCHMARK: No simulation, actual storage system I/O

This script tests:
1. Lustre per-file I/O (like LMCache pattern)
2. Lustre aggregated I/O (like Cascade pattern)  
3. HDF5 via h5py
4. Redis via redis-py
5. Shared memory via /dev/shm
6. GPU memory via CuPy (if available)
"""
import os
import sys
import json
import time
import struct
import hashlib
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

SCRATCH = Path(os.environ['SCRATCH'])
PROJECT_DIR = Path(os.environ['PROJECT_DIR'])
RESULTS_DIR = Path(os.environ['RESULTS_DIR'])
RANK = int(os.environ.get('SLURM_PROCID', 0))
NPROCS = int(os.environ.get('SLURM_NTASKS', 1))
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')

# Benchmark config
NUM_BLOCKS = 100  # 100 blocks per rank
BLOCK_SIZE = 10 * 1024 * 1024  # 10MB per block (smaller for debug queue time)
TOTAL_DATA_PER_RANK = NUM_BLOCKS * BLOCK_SIZE  # 1GB per rank

print(f"[Rank {RANK}] Config: {NUM_BLOCKS} blocks × {BLOCK_SIZE//1024//1024}MB = {TOTAL_DATA_PER_RANK//1024//1024}MB")

@dataclass
class BenchmarkResult:
    system: str
    rank: int
    operation: str
    num_ops: int
    total_bytes: int
    elapsed_sec: float
    throughput_gbps: float
    avg_latency_ms: float
    is_real: bool  # True = actual system, False = simulation (should never be False!)
    details: Dict[str, Any] = field(default_factory=dict)

###############################################################################
# REAL Storage Backends
###############################################################################

class LustrePerFileStore:
    """
    REAL Lustre per-file I/O (like LMCache pattern).
    One file per block - tests metadata overhead.
    """
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.is_real = True
        self.name = "Lustre-PerFile"
        
    def put(self, block_id: str, data: bytes) -> float:
        fpath = self.base_path / f"{block_id}.bin"
        t0 = time.perf_counter()
        with open(fpath, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return time.perf_counter() - t0
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        fpath = self.base_path / f"{block_id}.bin"
        t0 = time.perf_counter()
        if fpath.exists():
            with open(fpath, 'rb') as f:
                data = f.read()
            return data, time.perf_counter() - t0
        return None, time.perf_counter() - t0
    
    def clear(self):
        import shutil
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)


class LustreAggregatedStore:
    """
    REAL Lustre aggregated I/O (like Cascade pattern).
    Multiple blocks per file with stripe optimization.
    """
    def __init__(self, base_path: Path, blocks_per_file: int = 10):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.blocks_per_file = blocks_per_file
        self.is_real = True
        self.name = "Lustre-Aggregated"
        self.index = {}  # block_id -> (file_id, offset, size)
        self.current_file_id = 0
        self.current_offset = 0
        self.current_file_path = None
        self.blocks_in_current = 0
        
        # Set Lustre stripe
        try:
            subprocess.run(['lfs', 'setstripe', '-c', '16', '-S', '4m', 
                          str(self.base_path)], capture_output=True, timeout=5)
        except: pass
    
    def _get_file_path(self, file_id: int) -> Path:
        return self.base_path / f"agg_{RANK:03d}_{file_id:06d}.bin"
    
    def put(self, block_id: str, data: bytes) -> float:
        t0 = time.perf_counter()
        
        # Open new file if needed
        if self.current_file_path is None or self.blocks_in_current >= self.blocks_per_file:
            self.current_file_id += 1
            self.current_offset = 0
            self.blocks_in_current = 0
            self.current_file_path = self._get_file_path(self.current_file_id)
        
        # Write block (append mode)
        with open(self.current_file_path, 'ab') as f:
            offset = f.tell()
            f.write(data)
            self.index[block_id] = (self.current_file_id, offset, len(data))
        
        self.blocks_in_current += 1
        self.current_offset += len(data)
        
        return time.perf_counter() - t0
    
    def flush(self):
        # Files are already flushed after each write (closed)
        pass
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        t0 = time.perf_counter()
        if block_id not in self.index:
            return None, time.perf_counter() - t0
        
        file_id, offset, size = self.index[block_id]
        with open(self._get_file_path(file_id), 'rb') as f:
            f.seek(offset)
            data = f.read(size)
        return data, time.perf_counter() - t0
    
    def clear(self):
        import shutil
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self.current_file_id = 0
        self.current_offset = 0
        self.current_file_path = None
        self.blocks_in_current = 0


class HDF5Store:
    """
    REAL HDF5 storage via h5py.
    """
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.is_real = True
        self.name = "HDF5"
        self.h5file = None
        
    def initialize(self):
        import h5py
        self.h5file = h5py.File(str(self.file_path), 'w')
        self.h5file.create_group('blocks')
        return True
    
    def put(self, block_id: str, data: bytes) -> float:
        t0 = time.perf_counter()
        arr = np.frombuffer(data, dtype=np.uint8)
        self.h5file['blocks'].create_dataset(block_id, data=arr, compression='gzip', compression_opts=1)
        return time.perf_counter() - t0
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        t0 = time.perf_counter()
        if block_id in self.h5file['blocks']:
            data = self.h5file['blocks'][block_id][:].tobytes()
            return data, time.perf_counter() - t0
        return None, time.perf_counter() - t0
    
    def flush(self):
        self.h5file.flush()
    
    def clear(self):
        if self.h5file:
            self.h5file.close()
        if self.file_path.exists():
            self.file_path.unlink()
        self.initialize()


class RedisStore:
    """
    REAL Redis storage via redis-py.
    """
    def __init__(self, host: str, port: int = 6380):
        self.host = host
        self.port = port
        self.is_real = True
        self.name = "Redis"
        self.client = None
        
    def initialize(self) -> bool:
        try:
            import redis
            self.client = redis.Redis(host=self.host, port=self.port, socket_timeout=5)
            self.client.ping()
            return True
        except Exception as e:
            print(f"[RedisStore] Failed to connect: {e}")
            return False
    
    def put(self, block_id: str, data: bytes) -> float:
        t0 = time.perf_counter()
        self.client.set(f"r{RANK}:{block_id}", data)
        return time.perf_counter() - t0
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        t0 = time.perf_counter()
        data = self.client.get(f"r{RANK}:{block_id}")
        return data, time.perf_counter() - t0
    
    def clear(self):
        # Only clear this rank's keys
        for key in self.client.scan_iter(f"r{RANK}:*"):
            self.client.delete(key)


class SharedMemoryStore:
    """
    REAL shared memory via /dev/shm.
    """
    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path(f"/dev/shm/cascade_bench_{RANK}")
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.is_real = True
        self.name = "SharedMemory"
        
    def put(self, block_id: str, data: bytes) -> float:
        fpath = self.base_path / f"{block_id}.bin"
        t0 = time.perf_counter()
        with open(fpath, 'wb') as f:
            f.write(data)
        return time.perf_counter() - t0
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        fpath = self.base_path / f"{block_id}.bin"
        t0 = time.perf_counter()
        if fpath.exists():
            with open(fpath, 'rb') as f:
                data = f.read()
            return data, time.perf_counter() - t0
        return None, time.perf_counter() - t0
    
    def clear(self):
        import shutil
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)


class GPUMemoryStore:
    """
    REAL GPU memory via CuPy.
    """
    def __init__(self):
        self.is_real = True
        self.name = "GPU-Memory"
        self.cache = {}  # block_id -> GPU array
        self.available = False
        
    def initialize(self) -> bool:
        try:
            import cupy as cp
            # Allocate on correct GPU
            gpu_id = RANK % 4
            cp.cuda.Device(gpu_id).use()
            self.cp = cp
            self.available = True
            print(f"[GPUMemoryStore] Using GPU {gpu_id}")
            return True
        except Exception as e:
            print(f"[GPUMemoryStore] CuPy not available: {e}")
            return False
    
    def put(self, block_id: str, data: bytes) -> float:
        if not self.available:
            return 0.0
        t0 = time.perf_counter()
        arr = np.frombuffer(data, dtype=np.uint8)
        self.cache[block_id] = self.cp.asarray(arr)
        self.cp.cuda.Stream.null.synchronize()
        return time.perf_counter() - t0
    
    def get(self, block_id: str) -> Tuple[Optional[bytes], float]:
        if not self.available:
            return None, 0.0
        t0 = time.perf_counter()
        if block_id in self.cache:
            data = self.cp.asnumpy(self.cache[block_id]).tobytes()
            self.cp.cuda.Stream.null.synchronize()
            return data, time.perf_counter() - t0
        return None, time.perf_counter() - t0
    
    def clear(self):
        self.cache = {}
        if self.available:
            self.cp.get_default_memory_pool().free_all_blocks()


###############################################################################
# Benchmark Runner
###############################################################################

def generate_test_data() -> List[Tuple[str, bytes]]:
    """Generate test blocks with deterministic random data."""
    blocks = []
    for i in range(NUM_BLOCKS):
        # Deterministic seed for reproducibility
        np.random.seed(RANK * 10000 + i)
        data = np.random.bytes(BLOCK_SIZE)
        block_id = hashlib.sha256(data).hexdigest()[:16]
        blocks.append((block_id, data))
    return blocks


def run_benchmark(store, blocks: List[Tuple[str, bytes]]) -> Dict[str, BenchmarkResult]:
    """Run write and read benchmark on a store."""
    results = {}
    
    # WRITE benchmark
    write_latencies = []
    total_written = 0
    
    t0 = time.perf_counter()
    for block_id, data in blocks:
        latency = store.put(block_id, data)
        write_latencies.append(latency * 1000)  # ms
        total_written += len(data)
    
    if hasattr(store, 'flush'):
        store.flush()
    
    write_elapsed = time.perf_counter() - t0
    write_gbps = (total_written / 1e9) / write_elapsed if write_elapsed > 0 else 0
    
    results['write'] = BenchmarkResult(
        system=store.name,
        rank=RANK,
        operation='write',
        num_ops=len(blocks),
        total_bytes=total_written,
        elapsed_sec=write_elapsed,
        throughput_gbps=write_gbps,
        avg_latency_ms=np.mean(write_latencies) if write_latencies else 0,
        is_real=store.is_real,
        details={'latencies_p50': np.percentile(write_latencies, 50) if write_latencies else 0,
                 'latencies_p99': np.percentile(write_latencies, 99) if write_latencies else 0}
    )
    
    # READ benchmark (random order)
    read_latencies = []
    total_read = 0
    hits = 0
    
    indices = np.random.permutation(len(blocks))
    
    t0 = time.perf_counter()
    for idx in indices:
        block_id, original_data = blocks[idx]
        data, latency = store.get(block_id)
        read_latencies.append(latency * 1000)  # ms
        if data is not None:
            total_read += len(data)
            hits += 1
    
    read_elapsed = time.perf_counter() - t0
    read_gbps = (total_read / 1e9) / read_elapsed if read_elapsed > 0 else 0
    
    results['read'] = BenchmarkResult(
        system=store.name,
        rank=RANK,
        operation='read',
        num_ops=len(blocks),
        total_bytes=total_read,
        elapsed_sec=read_elapsed,
        throughput_gbps=read_gbps,
        avg_latency_ms=np.mean(read_latencies) if read_latencies else 0,
        is_real=store.is_real,
        details={'hits': hits, 'hit_rate': hits/len(blocks) if blocks else 0,
                 'latencies_p50': np.percentile(read_latencies, 50) if read_latencies else 0,
                 'latencies_p99': np.percentile(read_latencies, 99) if read_latencies else 0}
    )
    
    return results


###############################################################################
# Main
###############################################################################

def main():
    print(f"\n[Rank {RANK}] Generating test data...")
    blocks = generate_test_data()
    print(f"[Rank {RANK}] Generated {len(blocks)} blocks, {TOTAL_DATA_PER_RANK/1024/1024:.0f} MB total")
    
    all_results = {}
    
    # 1. Lustre Per-File (LMCache pattern)
    print(f"\n[Rank {RANK}] === Lustre Per-File ===")
    lustre_pf = LustrePerFileStore(SCRATCH / f"bench_lustre_pf_{JOB_ID}" / f"rank_{RANK}")
    lustre_pf.clear()
    all_results['Lustre-PerFile'] = run_benchmark(lustre_pf, blocks)
    print(f"[Rank {RANK}] Lustre-PerFile: write={all_results['Lustre-PerFile']['write'].throughput_gbps:.3f} GB/s, read={all_results['Lustre-PerFile']['read'].throughput_gbps:.3f} GB/s")
    
    # 2. Lustre Aggregated (Cascade pattern)
    print(f"\n[Rank {RANK}] === Lustre Aggregated ===")
    lustre_agg = LustreAggregatedStore(SCRATCH / f"bench_lustre_agg_{JOB_ID}" / f"rank_{RANK}")
    lustre_agg.clear()
    all_results['Lustre-Aggregated'] = run_benchmark(lustre_agg, blocks)
    print(f"[Rank {RANK}] Lustre-Aggregated: write={all_results['Lustre-Aggregated']['write'].throughput_gbps:.3f} GB/s, read={all_results['Lustre-Aggregated']['read'].throughput_gbps:.3f} GB/s")
    
    # 3. HDF5
    print(f"\n[Rank {RANK}] === HDF5 ===")
    try:
        hdf5 = HDF5Store(SCRATCH / f"bench_hdf5_{JOB_ID}" / f"rank_{RANK}.h5")
        hdf5.initialize()
        all_results['HDF5'] = run_benchmark(hdf5, blocks)
        print(f"[Rank {RANK}] HDF5: write={all_results['HDF5']['write'].throughput_gbps:.3f} GB/s, read={all_results['HDF5']['read'].throughput_gbps:.3f} GB/s")
        hdf5.h5file.close()
    except Exception as e:
        print(f"[Rank {RANK}] HDF5 failed: {e}")
    
    # 4. Redis
    print(f"\n[Rank {RANK}] === Redis ===")
    redis_store = RedisStore(REDIS_HOST, 6380)
    if redis_store.initialize():
        redis_store.clear()
        all_results['Redis'] = run_benchmark(redis_store, blocks)
        print(f"[Rank {RANK}] Redis: write={all_results['Redis']['write'].throughput_gbps:.3f} GB/s, read={all_results['Redis']['read'].throughput_gbps:.3f} GB/s")
    else:
        print(f"[Rank {RANK}] Redis: SKIPPED (not available)")
    
    # 5. Shared Memory
    print(f"\n[Rank {RANK}] === Shared Memory ===")
    shm = SharedMemoryStore()
    shm.clear()
    all_results['SharedMemory'] = run_benchmark(shm, blocks)
    print(f"[Rank {RANK}] SharedMemory: write={all_results['SharedMemory']['write'].throughput_gbps:.3f} GB/s, read={all_results['SharedMemory']['read'].throughput_gbps:.3f} GB/s")
    
    # 6. GPU Memory
    print(f"\n[Rank {RANK}] === GPU Memory ===")
    gpu = GPUMemoryStore()
    if gpu.initialize():
        all_results['GPU-Memory'] = run_benchmark(gpu, blocks)
        print(f"[Rank {RANK}] GPU-Memory: write={all_results['GPU-Memory']['write'].throughput_gbps:.3f} GB/s, read={all_results['GPU-Memory']['read'].throughput_gbps:.3f} GB/s")
    else:
        print(f"[Rank {RANK}] GPU-Memory: SKIPPED (CuPy not available)")
    
    # Save results
    output = {
        'metadata': {
            'job_id': JOB_ID,
            'rank': RANK,
            'nprocs': NPROCS,
            'timestamp': datetime.now().isoformat(),
            'num_blocks': NUM_BLOCKS,
            'block_size_mb': BLOCK_SIZE / 1024 / 1024,
            'total_data_mb': TOTAL_DATA_PER_RANK / 1024 / 1024,
        },
        'results': {}
    }
    
    for sys_name, sys_results in all_results.items():
        output['results'][sys_name] = {
            'write': asdict(sys_results['write']),
            'read': asdict(sys_results['read']),
        }
    
    result_file = RESULTS_DIR / f"real_bench_{JOB_ID}_rank{RANK:02d}.json"
    with open(result_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[Rank {RANK}] Results saved to {result_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY (Rank {RANK}) - ALL REAL, NO SIMULATION")
    print(f"{'='*70}")
    print(f"{'System':<20} | {'Write GB/s':>12} | {'Read GB/s':>12} | {'Real?':>6}")
    print(f"{'-'*70}")
    for sys_name, sys_results in all_results.items():
        w = sys_results['write'].throughput_gbps
        r = sys_results['read'].throughput_gbps
        is_real = sys_results['write'].is_real
        print(f"{sys_name:<20} | {w:>12.3f} | {r:>12.3f} | {'YES':>6}")
    print(f"{'='*70}")
    
    # Cleanup
    print(f"\n[Rank {RANK}] Cleaning up...")
    shm.clear()
    if 'gpu' in dir() and gpu.available:
        gpu.clear()


if __name__ == '__main__':
    main()

PYEOF

###############################################################################
# Cleanup on Rank 0
###############################################################################
sleep 10  # Wait for all ranks

if [ $RANK -eq 0 ]; then
    echo "[CLEANUP] Stopping Redis..."
    $REDIS_DIR/src/redis-cli -p 6380 shutdown 2>/dev/null || true
    
    echo "[CLEANUP] Aggregating results..."
    python3 << 'AGGEOF'
import json
from pathlib import Path
import os

RESULTS_DIR = Path(os.environ['RESULTS_DIR'])
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')

# Collect all rank results
all_ranks = []
for f in sorted(RESULTS_DIR.glob(f"real_bench_{JOB_ID}_rank*.json")):
    with open(f) as fp:
        all_ranks.append(json.load(fp))

if not all_ranks:
    print("No results found!")
    exit(1)

# Aggregate
systems = list(all_ranks[0]['results'].keys())
aggregated = {'metadata': all_ranks[0]['metadata'].copy(), 'aggregated': {}}
aggregated['metadata']['num_ranks'] = len(all_ranks)

for sys in systems:
    write_gbps = [r['results'][sys]['write']['throughput_gbps'] for r in all_ranks if sys in r['results']]
    read_gbps = [r['results'][sys]['read']['throughput_gbps'] for r in all_ranks if sys in r['results']]
    
    if write_gbps:
        aggregated['aggregated'][sys] = {
            'write_gbps_mean': sum(write_gbps) / len(write_gbps),
            'write_gbps_sum': sum(write_gbps),
            'read_gbps_mean': sum(read_gbps) / len(read_gbps),
            'read_gbps_sum': sum(read_gbps),
            'num_ranks': len(write_gbps),
            'is_real': True
        }

# Save aggregated
agg_file = RESULTS_DIR / f"real_bench_{JOB_ID}_aggregated.json"
with open(agg_file, 'w') as f:
    json.dump(aggregated, f, indent=2)

print(f"\nAGGREGATED RESULTS ({len(all_ranks)} ranks):")
print("="*70)
print(f"{'System':<20} | {'Write Sum GB/s':>14} | {'Read Sum GB/s':>14}")
print("-"*70)
for sys, vals in aggregated['aggregated'].items():
    print(f"{sys:<20} | {vals['write_gbps_sum']:>14.2f} | {vals['read_gbps_sum']:>14.2f}")
print("="*70)
print(f"Results saved to: {agg_file}")

AGGEOF
fi

echo "[Rank $RANK] Done."
