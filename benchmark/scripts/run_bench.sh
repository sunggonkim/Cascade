#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o logs/bench_%j.out
#SBATCH -e logs/bench_%j.err
#SBATCH -J cascade_bench

###############################################################################
# Cascade vs Baselines Benchmark
###############################################################################

set -e

export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade
export DATA_DIR=$SCRATCH/cascade_kv_cache
export RESULTS_DIR=$PROJECT_DIR/benchmark/results

module load python
module load cray-mpich

cd $PROJECT_DIR

mkdir -p $RESULTS_DIR
mkdir -p benchmark/logs

echo "============================================"
echo "Cascade Benchmark Suite"
echo "============================================"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Data: $DATA_DIR"
echo "============================================"

# Run Python benchmark
python3 << 'PYEOF'
import os
import sys
import json
import time
import struct
import hashlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import mmap

# Configuration
DATA_DIR = Path(os.environ['DATA_DIR'])
RESULTS_DIR = Path(os.environ['RESULTS_DIR'])

@dataclass
class BenchmarkResult:
    system: str
    operation: str
    num_blocks: int
    total_bytes: int
    elapsed_seconds: float
    throughput_gbps: float
    latency_ms: float
    ops_per_second: float

class AggregatedDataReader:
    """Read blocks from aggregated binary files."""
    
    MAGIC = b'CASKV001'
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.index = self._load_global_index()
        
    def _load_global_index(self) -> Dict:
        index_path = self.data_dir / 'global_index.json'
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
                return data.get('blocks', {})
        return {}
    
    def get_block_ids(self) -> List[str]:
        return list(self.index.keys())
    
    def read_block(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        if block_id not in self.index:
            return None
        
        info = self.index[block_id]
        file_path = self.data_dir / info['file']
        
        with open(file_path, 'rb') as f:
            f.seek(info['offset'])
            key_size, value_size = struct.unpack('<QQ', f.read(16))
            key_data = f.read(key_size)
            value_data = f.read(value_size)
        
        return key_data, value_data

    def read_blocks_batch(self, block_ids: List[str]) -> List[Tuple[bytes, bytes]]:
        """Read multiple blocks (can be parallelized)."""
        results = []
        for bid in block_ids:
            result = self.read_block(bid)
            if result:
                results.append(result)
        return results


class InMemoryKVStore:
    """Simple in-memory store for baseline comparison."""
    
    def __init__(self):
        self.store = {}
    
    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        self.store[block_id] = (key_data, value_data)
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        return self.store.get(block_id)
    
    def contains(self, block_id: str) -> bool:
        return block_id in self.store
    
    def clear(self):
        self.store.clear()


class CascadeStore:
    """Cascade-style tiered store with content-addressed deduplication."""
    
    def __init__(self, gpu_capacity_blocks=100, shm_capacity_blocks=500):
        self.gpu_cache = {}  # Tier 1: GPU HBM (simulated)
        self.shm_cache = {}  # Tier 2: Shared memory
        self.gpu_capacity = gpu_capacity_blocks
        self.shm_capacity = shm_capacity_blocks
        self.dedup_index = {}  # Content hash -> location
        
    def _compute_hash(self, key_data: bytes, value_data: bytes) -> str:
        hasher = hashlib.sha256()
        hasher.update(key_data)
        hasher.update(value_data)
        return hasher.hexdigest()[:32]
    
    def put(self, block_id: str, key_data: bytes, value_data: bytes, is_prefix=False) -> bool:
        content_hash = self._compute_hash(key_data, value_data)
        
        # Deduplication check
        if content_hash in self.dedup_index:
            return True  # Already exists, deduplicated
        
        # Try GPU first
        if len(self.gpu_cache) < self.gpu_capacity:
            self.gpu_cache[block_id] = (key_data, value_data)
            self.dedup_index[content_hash] = ('gpu', block_id)
            return True
        
        # Evict from GPU to SHM
        if len(self.shm_cache) < self.shm_capacity:
            # Evict oldest from GPU
            if self.gpu_cache:
                evict_id = next(iter(self.gpu_cache))
                evict_data = self.gpu_cache.pop(evict_id)
                self.shm_cache[evict_id] = evict_data
            
            self.gpu_cache[block_id] = (key_data, value_data)
            self.dedup_index[content_hash] = ('gpu', block_id)
            return True
        
        return False
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        if block_id in self.gpu_cache:
            return self.gpu_cache[block_id]
        if block_id in self.shm_cache:
            # Promote to GPU
            data = self.shm_cache[block_id]
            if len(self.gpu_cache) < self.gpu_capacity:
                self.gpu_cache[block_id] = data
            return data
        return None
    
    def get_stats(self) -> Dict:
        return {
            'gpu_blocks': len(self.gpu_cache),
            'shm_blocks': len(self.shm_cache),
            'dedup_entries': len(self.dedup_index)
        }
    
    def clear(self):
        self.gpu_cache.clear()
        self.shm_cache.clear()
        self.dedup_index.clear()


def run_write_benchmark(store, blocks: List[Tuple[str, bytes, bytes]], name: str) -> BenchmarkResult:
    """Benchmark write throughput."""
    total_bytes = sum(len(k) + len(v) for _, k, v in blocks)
    
    start = time.perf_counter()
    for block_id, key_data, value_data in blocks:
        store.put(block_id, key_data, value_data)
    elapsed = time.perf_counter() - start
    
    return BenchmarkResult(
        system=name,
        operation='write',
        num_blocks=len(blocks),
        total_bytes=total_bytes,
        elapsed_seconds=elapsed,
        throughput_gbps=total_bytes / elapsed / 1e9,
        latency_ms=elapsed / len(blocks) * 1000,
        ops_per_second=len(blocks) / elapsed
    )


def run_read_benchmark(store, block_ids: List[str], name: str, expected_size: int) -> BenchmarkResult:
    """Benchmark read throughput."""
    start = time.perf_counter()
    hits = 0
    for block_id in block_ids:
        result = store.get(block_id)
        if result:
            hits += 1
    elapsed = time.perf_counter() - start
    
    total_bytes = hits * expected_size
    
    return BenchmarkResult(
        system=name,
        operation='read',
        num_blocks=hits,
        total_bytes=total_bytes,
        elapsed_seconds=elapsed,
        throughput_gbps=total_bytes / elapsed / 1e9 if elapsed > 0 else 0,
        latency_ms=elapsed / len(block_ids) * 1000 if block_ids else 0,
        ops_per_second=len(block_ids) / elapsed if elapsed > 0 else 0
    )


def main():
    print("Loading data from", DATA_DIR)
    
    # Load data
    reader = AggregatedDataReader(DATA_DIR)
    block_ids = reader.get_block_ids()
    
    print(f"Found {len(block_ids)} blocks in index")
    
    if not block_ids:
        print("No blocks found! Exiting.")
        return
    
    # Load subset for benchmark
    num_test_blocks = min(500, len(block_ids))
    test_ids = block_ids[:num_test_blocks]
    
    print(f"Loading {num_test_blocks} blocks for benchmark...")
    blocks = []
    for bid in test_ids:
        data = reader.read_block(bid)
        if data:
            blocks.append((bid, data[0], data[1]))
    
    print(f"Loaded {len(blocks)} blocks")
    
    if not blocks:
        print("Failed to load blocks!")
        return
    
    block_size = len(blocks[0][1]) + len(blocks[0][2])
    print(f"Block size: {block_size / 1e6:.1f} MB")
    
    results = []
    
    # Benchmark 1: In-Memory Baseline (simulating LMCache-like)
    print("\n--- In-Memory Store (LMCache-like) ---")
    mem_store = InMemoryKVStore()
    
    result = run_write_benchmark(mem_store, blocks, 'inmemory')
    print(f"Write: {result.throughput_gbps:.2f} GB/s, {result.latency_ms:.3f} ms/op")
    results.append(result)
    
    result = run_read_benchmark(mem_store, [b[0] for b in blocks], 'inmemory', block_size)
    print(f"Read:  {result.throughput_gbps:.2f} GB/s, {result.latency_ms:.3f} ms/op")
    results.append(result)
    
    # Benchmark 2: Cascade Store (with dedup + tiering)
    print("\n--- Cascade Store (Tiered + Dedup) ---")
    cascade_store = CascadeStore(gpu_capacity_blocks=200, shm_capacity_blocks=500)
    
    result = run_write_benchmark(cascade_store, blocks, 'cascade')
    print(f"Write: {result.throughput_gbps:.2f} GB/s, {result.latency_ms:.3f} ms/op")
    results.append(result)
    
    stats = cascade_store.get_stats()
    print(f"Stats: GPU={stats['gpu_blocks']}, SHM={stats['shm_blocks']}, Dedup={stats['dedup_entries']}")
    
    result = run_read_benchmark(cascade_store, [b[0] for b in blocks], 'cascade', block_size)
    print(f"Read:  {result.throughput_gbps:.2f} GB/s, {result.latency_ms:.3f} ms/op")
    results.append(result)
    
    # Benchmark 3: Cascade with prefix sharing (dedup benefit)
    print("\n--- Cascade with Prefix Sharing ---")
    cascade_prefix = CascadeStore(gpu_capacity_blocks=200, shm_capacity_blocks=500)
    
    # Simulate prefix sharing: duplicate first 50 blocks across 10 "sessions"
    prefix_blocks = blocks[:50]
    unique_blocks = blocks[50:]
    
    all_blocks = []
    for session in range(10):
        # Each session reuses prefix
        for bid, k, v in prefix_blocks:
            session_bid = f"s{session}_{bid}"
            all_blocks.append((session_bid, k, v))  # Same content, different ID
        # Unique per session
        for bid, k, v in unique_blocks[:10]:
            session_bid = f"s{session}_u_{bid}"
            all_blocks.append((session_bid, k, v))
    
    result = run_write_benchmark(cascade_prefix, all_blocks, 'cascade_prefix')
    print(f"Write: {result.throughput_gbps:.2f} GB/s, {result.latency_ms:.3f} ms/op")
    results.append(result)
    
    stats = cascade_prefix.get_stats()
    expected_unique = 50 + len(unique_blocks[:10])
    dedup_ratio = 1.0 - stats['dedup_entries'] / len(all_blocks)
    print(f"Dedup ratio: {dedup_ratio*100:.1f}% (unique: {stats['dedup_entries']}, total: {len(all_blocks)})")
    results.append(BenchmarkResult(
        system='cascade_prefix',
        operation='dedup',
        num_blocks=len(all_blocks),
        total_bytes=0,
        elapsed_seconds=0,
        throughput_gbps=0,
        latency_ms=0,
        ops_per_second=dedup_ratio * 100
    ))
    
    # Save results
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_dir': str(DATA_DIR),
        'num_test_blocks': num_test_blocks,
        'block_size_bytes': block_size,
        'results': [asdict(r) for r in results]
    }
    
    output_path = RESULTS_DIR / f'benchmark_{time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Summary table
    print("\n" + "="*70)
    print(f"{'System':<20} {'Operation':<10} {'Throughput':<15} {'Latency':<15}")
    print("="*70)
    for r in results:
        if r.operation != 'dedup':
            print(f"{r.system:<20} {r.operation:<10} {r.throughput_gbps:>10.2f} GB/s {r.latency_ms:>10.3f} ms")
    print("="*70)


if __name__ == '__main__':
    main()
PYEOF

echo ""
echo "Benchmark complete!"
