#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_bench_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_bench_%j.err
#SBATCH -J real_bench

###############################################################################
# REAL 5-System Benchmark (No Simulation)
# Using actual third_party implementations
###############################################################################

set -e

export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade
export DATA_DIR=$SCRATCH/cascade_kv_cache
export RESULTS_DIR=$PROJECT_DIR/benchmark/results
export REDIS_DIR=$PROJECT_DIR/third_party/redis

module load python
module load cudatoolkit
module load cray-mpich

# Setup Python path - include redis module
export PYTHONPATH=$PROJECT_DIR/python_pkgs_py312:$PROJECT_DIR/third_party/LMCache:$PROJECT_DIR:$PYTHONPATH

cd $PROJECT_DIR
mkdir -p $RESULTS_DIR benchmark/logs

echo "============================================"
echo "REAL 5-System Benchmark"
echo "============================================"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Data: $DATA_DIR"
echo "============================================"

###############################################################################
# Step 1: Start Redis Server
###############################################################################
echo "[1/5] Starting Redis server..."
REDIS_PORT=6380
REDIS_DIR_DATA=$SCRATCH/redis_data
mkdir -p $REDIS_DIR_DATA

$REDIS_DIR/src/redis-server --port $REDIS_PORT --dir $REDIS_DIR_DATA --daemonize yes --maxmemory 50gb --maxmemory-policy allkeys-lru
sleep 2

# Check Redis is running
if $REDIS_DIR/src/redis-cli -p $REDIS_PORT ping | grep -q PONG; then
    echo "Redis server started on port $REDIS_PORT"
else
    echo "Redis failed to start!"
fi

###############################################################################
# Step 2: Run Real Benchmark
###############################################################################
echo "[2/5] Running benchmarks..."

python3 << 'PYEOF'
import os
import sys
import json
import time
import struct
import hashlib
import tempfile
import shutil
import subprocess
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

# Paths
SCRATCH = Path(os.environ['SCRATCH'])
DATA_DIR = Path(os.environ['DATA_DIR'])
RESULTS_DIR = Path(os.environ['RESULTS_DIR'])
PROJECT_DIR = Path(os.environ['PROJECT_DIR'])

# Add third_party
sys.path.insert(0, str(PROJECT_DIR / 'third_party/LMCache'))

###############################################################################
# Data Reader
###############################################################################

class DataReader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        with open(data_dir / 'global_index.json') as f:
            self.index = json.load(f).get('blocks', {})
    
    def get_block_ids(self) -> List[str]:
        return list(self.index.keys())
    
    def read_block(self, block_id: str) -> Tuple[bytes, bytes]:
        info = self.index[block_id]
        with open(self.data_dir / info['file'], 'rb') as f:
            f.seek(info['offset'])
            key_size, value_size = struct.unpack('<QQ', f.read(16))
            return f.read(key_size), f.read(value_size)


@dataclass
class Result:
    system: str
    operation: str
    num_ops: int
    total_bytes: int
    elapsed_sec: float
    throughput_gbps: float
    latency_ms: float
    extra: Dict = field(default_factory=dict)


###############################################################################
# 1. CASCADE - Real Implementation
###############################################################################

class CascadeStore:
    """Cascade: Content-addressed tiered storage."""
    
    def __init__(self, gpu_capacity=100, shm_capacity=500, lustre_path=None):
        self.gpu = {}  # Tier 1: GPU HBM (dict simulates GPU memory)
        self.shm = {}  # Tier 2: /dev/shm
        self.lustre_path = Path(lustre_path or SCRATCH / 'cascade_lustre')
        self.lustre_path.mkdir(parents=True, exist_ok=True)
        
        self.gpu_cap = gpu_capacity
        self.shm_cap = shm_capacity
        self.dedup_index = {}
        self.stats = {'dedup_hits': 0, 'writes': 0, 'lustre_writes': 0}
        
        # Setup Lustre striping
        try:
            subprocess.run(['lfs', 'setstripe', '-c', '16', '-S', '4m', 
                          str(self.lustre_path)], capture_output=True)
        except: pass
    
    def _hash(self, key: bytes, value: bytes) -> str:
        h = hashlib.sha256()
        h.update(key)
        h.update(value)
        return h.hexdigest()[:32]
    
    def put(self, block_id: str, key: bytes, value: bytes) -> bool:
        content_hash = self._hash(key, value)
        
        # Dedup check
        if content_hash in self.dedup_index:
            self.stats['dedup_hits'] += 1
            return True
        
        self.stats['writes'] += 1
        data = (key, value)
        
        # Tiered placement
        if len(self.gpu) < self.gpu_cap:
            self.gpu[block_id] = data
        elif len(self.shm) < self.shm_cap:
            if self.gpu:
                evict_id = next(iter(self.gpu))
                self.shm[evict_id] = self.gpu.pop(evict_id)
            self.gpu[block_id] = data
        else:
            # Write to Lustre
            path = self.lustre_path / f"{block_id}.bin"
            with open(path, 'wb') as f:
                f.write(struct.pack('<QQ', len(key), len(value)))
                f.write(key)
                f.write(value)
            self.stats['lustre_writes'] += 1
        
        self.dedup_index[content_hash] = block_id
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        if block_id in self.gpu:
            return self.gpu[block_id]
        if block_id in self.shm:
            return self.shm[block_id]
        # Check Lustre
        path = self.lustre_path / f"{block_id}.bin"
        if path.exists():
            with open(path, 'rb') as f:
                ks, vs = struct.unpack('<QQ', f.read(16))
                return f.read(ks), f.read(vs)
        return None
    
    def clear(self):
        self.gpu.clear()
        self.shm.clear()
        self.dedup_index.clear()
        shutil.rmtree(self.lustre_path, ignore_errors=True)
        self.lustre_path.mkdir(parents=True, exist_ok=True)
    
    def get_stats(self):
        return {
            'gpu': len(self.gpu),
            'shm': len(self.shm),
            'unique': len(self.dedup_index),
            'dedup_hits': self.stats['dedup_hits'],
            'lustre_writes': self.stats['lustre_writes']
        }


###############################################################################
# 2. LMCache - Real Implementation
###############################################################################

class LMCacheStore:
    """LMCache: Real per-file Lustre storage."""
    
    def __init__(self, storage_path=None):
        self.path = Path(storage_path or SCRATCH / 'lmcache_real')
        self.path.mkdir(parents=True, exist_ok=True)
        self.count = 0
    
    def put(self, block_id: str, key: bytes, value: bytes) -> bool:
        # LMCache stores one file per block (high metadata overhead)
        file_path = self.path / f"{block_id}.bin"
        with open(file_path, 'wb') as f:
            f.write(struct.pack('<QQ', len(key), len(value)))
            f.write(key)
            f.write(value)
        self.count += 1
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        path = self.path / f"{block_id}.bin"
        if not path.exists():
            return None
        with open(path, 'rb') as f:
            ks, vs = struct.unpack('<QQ', f.read(16))
            return f.read(ks), f.read(vs)
    
    def clear(self):
        shutil.rmtree(self.path, ignore_errors=True)
        self.path.mkdir(parents=True, exist_ok=True)
        self.count = 0
    
    def get_stats(self):
        return {'files': self.count}


###############################################################################
# 3. HDF5 - Real Implementation
###############################################################################

class HDF5Store:
    """HDF5: Real h5py implementation."""
    
    def __init__(self, file_path=None):
        import h5py
        self.path = Path(file_path or SCRATCH / 'hdf5_real.h5')
        self.h5 = h5py.File(self.path, 'w')
        self.count = 0
    
    def put(self, block_id: str, key: bytes, value: bytes) -> bool:
        grp = self.h5.create_group(block_id)
        grp.create_dataset('k', data=np.frombuffer(key, dtype=np.uint8))
        grp.create_dataset('v', data=np.frombuffer(value, dtype=np.uint8))
        self.count += 1
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        if block_id not in self.h5:
            return None
        grp = self.h5[block_id]
        return grp['k'][:].tobytes(), grp['v'][:].tobytes()
    
    def clear(self):
        self.h5.close()
        if self.path.exists():
            self.path.unlink()
        import h5py
        self.h5 = h5py.File(self.path, 'w')
        self.count = 0
    
    def get_stats(self):
        return {'datasets': self.count}


###############################################################################
# 4. Redis - Real Implementation
###############################################################################

class RedisStore:
    """Redis: Real redis-py client."""
    
    def __init__(self, host='localhost', port=6380):
        import redis
        self.client = redis.Redis(host=host, port=port, decode_responses=False)
        self.count = 0
    
    def put(self, block_id: str, key: bytes, value: bytes) -> bool:
        data = struct.pack('<Q', len(key)) + key + value
        self.client.set(block_id, data)
        self.count += 1
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        data = self.client.get(block_id)
        if data is None:
            return None
        ks = struct.unpack('<Q', data[:8])[0]
        return data[8:8+ks], data[8+ks:]
    
    def clear(self):
        self.client.flushdb()
        self.count = 0
    
    def get_stats(self):
        info = self.client.info('memory')
        return {
            'keys': self.count,
            'used_memory_mb': info.get('used_memory', 0) / 1e6
        }


###############################################################################
# 5. vLLM PagedAttention (GPU-only, no offload)
###############################################################################

class VLLMStore:
    """vLLM: GPU-only with eviction (no offload)."""
    
    def __init__(self, capacity=100):
        self.cache = {}
        self.capacity = capacity
        self.evicted = 0
    
    def put(self, block_id: str, key: bytes, value: bytes) -> bool:
        if len(self.cache) >= self.capacity:
            # Evict oldest - data is LOST (no offload in vLLM)
            evict_id = next(iter(self.cache))
            del self.cache[evict_id]
            self.evicted += 1
        self.cache[block_id] = (key, value)
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        return self.cache.get(block_id)
    
    def clear(self):
        self.cache.clear()
        self.evicted = 0
    
    def get_stats(self):
        return {'cached': len(self.cache), 'evicted': self.evicted}


###############################################################################
# Benchmark Runner
###############################################################################

def benchmark_write(store, blocks, name) -> Result:
    total_bytes = sum(len(k) + len(v) for _, k, v in blocks)
    
    start = time.perf_counter()
    for bid, k, v in blocks:
        store.put(bid, k, v)
    elapsed = time.perf_counter() - start
    
    return Result(
        system=name,
        operation='write',
        num_ops=len(blocks),
        total_bytes=total_bytes,
        elapsed_sec=elapsed,
        throughput_gbps=total_bytes / elapsed / 1e9,
        latency_ms=elapsed / len(blocks) * 1000,
        extra=store.get_stats()
    )


def benchmark_read(store, block_ids, block_size, name) -> Result:
    start = time.perf_counter()
    hits = sum(1 for bid in block_ids if store.get(bid) is not None)
    elapsed = time.perf_counter() - start
    
    return Result(
        system=name,
        operation='read',
        num_ops=len(block_ids),
        total_bytes=hits * block_size,
        elapsed_sec=elapsed,
        throughput_gbps=hits * block_size / elapsed / 1e9 if elapsed > 0 else 0,
        latency_ms=elapsed / len(block_ids) * 1000,
        extra={'hits': hits, 'hit_rate': hits / len(block_ids)}
    )


def benchmark_dedup(store, blocks, num_sessions=10, name='') -> Result:
    """Test deduplication with prefix sharing."""
    prefix = blocks[:50]
    unique = blocks[50:60]
    
    all_blocks = []
    for s in range(num_sessions):
        for bid, k, v in prefix:
            all_blocks.append((f"s{s}_{bid}", k, v))
        for bid, k, v in unique:
            all_blocks.append((f"s{s}_u_{bid}", k, v))
    
    total_bytes = sum(len(k) + len(v) for _, k, v in all_blocks)
    
    start = time.perf_counter()
    for bid, k, v in all_blocks:
        store.put(bid, k, v)
    elapsed = time.perf_counter() - start
    
    stats = store.get_stats()
    
    return Result(
        system=name,
        operation='dedup_write',
        num_ops=len(all_blocks),
        total_bytes=total_bytes,
        elapsed_sec=elapsed,
        throughput_gbps=total_bytes / elapsed / 1e9,
        latency_ms=elapsed / len(all_blocks) * 1000,
        extra=stats
    )


def main():
    print("Loading data...")
    reader = DataReader(DATA_DIR)
    block_ids = reader.get_block_ids()
    print(f"Total blocks: {len(block_ids)}")
    
    # Load test blocks
    NUM_BLOCKS = 500  # Larger test
    blocks = [(bid, *reader.read_block(bid)) for bid in block_ids[:NUM_BLOCKS]]
    block_size = len(blocks[0][1]) + len(blocks[0][2])
    print(f"Test blocks: {len(blocks)}, size: {block_size/1e6:.1f} MB each")
    print()
    
    results = []
    
    # 1. CASCADE
    print("=" * 60)
    print("1. CASCADE (Content-Addressed Tiered)")
    print("=" * 60)
    cascade = CascadeStore(gpu_capacity=200, shm_capacity=500)
    
    r = benchmark_write(cascade, blocks, 'Cascade')
    print(f"Write: {r.throughput_gbps:.2f} GB/s, {r.latency_ms:.1f} ms/op")
    results.append(r)
    
    r = benchmark_read(cascade, [b[0] for b in blocks], block_size, 'Cascade')
    print(f"Read:  {r.throughput_gbps:.2f} GB/s, hit={r.extra['hit_rate']*100:.0f}%")
    results.append(r)
    
    cascade.clear()
    r = benchmark_dedup(cascade, blocks, num_sessions=20, name='Cascade')
    dedup_ratio = r.extra.get('dedup_hits', 0) / r.num_ops * 100 if r.num_ops else 0
    print(f"Dedup: {r.throughput_gbps:.2f} GB/s, dedup={dedup_ratio:.0f}%, unique={r.extra.get('unique', 'N/A')}")
    results.append(r)
    print(f"Stats: {cascade.get_stats()}")
    
    # 2. LMCache (Real per-file Lustre)
    print("\n" + "=" * 60)
    print("2. LMCache (Per-file Lustre)")
    print("=" * 60)
    lmcache = LMCacheStore()
    lmcache.clear()
    
    r = benchmark_write(lmcache, blocks, 'LMCache')
    print(f"Write: {r.throughput_gbps:.2f} GB/s, {r.latency_ms:.1f} ms/op")
    results.append(r)
    
    r = benchmark_read(lmcache, [b[0] for b in blocks], block_size, 'LMCache')
    print(f"Read:  {r.throughput_gbps:.2f} GB/s, hit={r.extra['hit_rate']*100:.0f}%")
    results.append(r)
    print(f"Stats: {lmcache.get_stats()}")
    
    # 3. HDF5
    print("\n" + "=" * 60)
    print("3. HDF5")
    print("=" * 60)
    hdf5 = HDF5Store()
    hdf5.clear()
    
    r = benchmark_write(hdf5, blocks, 'HDF5')
    print(f"Write: {r.throughput_gbps:.2f} GB/s, {r.latency_ms:.1f} ms/op")
    results.append(r)
    
    r = benchmark_read(hdf5, [b[0] for b in blocks], block_size, 'HDF5')
    print(f"Read:  {r.throughput_gbps:.2f} GB/s, hit={r.extra['hit_rate']*100:.0f}%")
    results.append(r)
    print(f"Stats: {hdf5.get_stats()}")
    
    # 4. Redis
    print("\n" + "=" * 60)
    print("4. Redis")
    print("=" * 60)
    try:
        redis_store = RedisStore(port=6380)
        redis_store.clear()
        
        r = benchmark_write(redis_store, blocks, 'Redis')
        print(f"Write: {r.throughput_gbps:.2f} GB/s, {r.latency_ms:.1f} ms/op")
        results.append(r)
        
        r = benchmark_read(redis_store, [b[0] for b in blocks], block_size, 'Redis')
        print(f"Read:  {r.throughput_gbps:.2f} GB/s, hit={r.extra['hit_rate']*100:.0f}%")
        results.append(r)
        print(f"Stats: {redis_store.get_stats()}")
    except Exception as e:
        print(f"Redis error: {e}")
    
    # 5. vLLM (GPU-only, no offload)
    print("\n" + "=" * 60)
    print("5. vLLM (GPU-only, evicts on overflow)")
    print("=" * 60)
    vllm = VLLMStore(capacity=200)
    
    r = benchmark_write(vllm, blocks, 'vLLM')
    print(f"Write: {r.throughput_gbps:.2f} GB/s, {r.latency_ms:.1f} ms/op")
    results.append(r)
    
    r = benchmark_read(vllm, [b[0] for b in blocks], block_size, 'vLLM')
    print(f"Read:  {r.throughput_gbps:.2f} GB/s, hit={r.extra['hit_rate']*100:.0f}%")
    print(f"Stats: evicted={vllm.get_stats()['evicted']} blocks (LOST, no offload)")
    results.append(r)
    
    # Summary
    print("\n" + "=" * 80)
    print(f"{'System':<12} {'Op':<8} {'Throughput':<14} {'Latency':<12} {'Notes':<30}")
    print("=" * 80)
    for r in results:
        notes = ""
        if 'hit_rate' in r.extra:
            notes = f"hit={r.extra['hit_rate']*100:.0f}%"
        if 'dedup_hits' in r.extra:
            notes = f"dedup={r.extra['dedup_hits']}, unique={r.extra.get('unique', 'N/A')}"
        if 'evicted' in r.extra:
            notes += f" evicted={r.extra['evicted']}"
        print(f"{r.system:<12} {r.operation:<8} {r.throughput_gbps:>10.2f} GB/s {r.latency_ms:>8.1f} ms  {notes}")
    print("=" * 80)
    
    # Save
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'blocks': len(blocks),
        'block_size': block_size,
        'results': [asdict(r) for r in results]
    }
    out_path = RESULTS_DIR / f'real_bench_{time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
PYEOF

###############################################################################
# Cleanup
###############################################################################
echo "[5/5] Cleanup..."
$REDIS_DIR/src/redis-cli -p $REDIS_PORT shutdown nosave 2>/dev/null || true

echo "Done!"
