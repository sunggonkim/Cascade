#!/bin/bash
#SBATCH -A m1248
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -o logs/compare_%j.out
#SBATCH -e logs/compare_%j.err
#SBATCH -J compare_5sys

###############################################################################
# 5-System Comparison Benchmark
# Cascade vs vLLM vs LMCache vs HDF5 vs PDC (+ Redis)
###############################################################################

set -e

export SCRATCH=/pscratch/sd/s/sgkim
export PROJECT_DIR=$SCRATCH/Skim-cascade
export DATA_DIR=$SCRATCH/cascade_kv_cache
export RESULTS_DIR=$PROJECT_DIR/benchmark/results

# Add third_party to path
export PYTHONPATH=$PROJECT_DIR/third_party/LMCache:$PROJECT_DIR/third_party/vllm:$PYTHONPATH

module load python
module load cray-mpich

cd $PROJECT_DIR
mkdir -p $RESULTS_DIR benchmark/logs

echo "============================================"
echo "5-System Comparison Benchmark"
echo "============================================"
echo "Systems: Cascade, vLLM, LMCache, HDF5, PDC"
echo "Nodes: $SLURM_NNODES"
echo "Data: $DATA_DIR"
echo "============================================"

python3 << 'PYEOF'
import os
import sys
import json
import time
import struct
import hashlib
import tempfile
import shutil
import h5py
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

# Add third_party paths
sys.path.insert(0, '/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache')
sys.path.insert(0, '/pscratch/sd/s/sgkim/Skim-cascade/third_party/vllm')

DATA_DIR = Path(os.environ['DATA_DIR'])
RESULTS_DIR = Path(os.environ['RESULTS_DIR'])
SCRATCH = Path(os.environ.get('SCRATCH', '/tmp'))

###############################################################################
# Data Reader
###############################################################################

class AggregatedDataReader:
    """Read blocks from aggregated binary files."""
    
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


###############################################################################
# Base Adapter
###############################################################################

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
    extra: Dict = field(default_factory=dict)


class StorageAdapter(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def initialize(self) -> bool: pass
    
    @abstractmethod
    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool: pass
    
    @abstractmethod
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]: pass
    
    @abstractmethod
    def clear(self) -> None: pass
    
    def get_stats(self) -> Dict:
        return {}


###############################################################################
# 1. Cascade Adapter (Content-Addressed + Tiered)
###############################################################################

class CascadeAdapter(StorageAdapter):
    """Cascade: Content-addressed tiered KV cache."""
    
    def __init__(self, gpu_capacity=100, shm_capacity=500):
        super().__init__("Cascade")
        self.gpu_cache = {}
        self.shm_cache = {}
        self.gpu_capacity = gpu_capacity
        self.shm_capacity = shm_capacity
        self.dedup_index = {}
        self.stats = {'dedup_hits': 0, 'dedup_misses': 0}
    
    def initialize(self) -> bool:
        return True
    
    def _compute_hash(self, key_data: bytes, value_data: bytes) -> str:
        hasher = hashlib.sha256()
        hasher.update(key_data)
        hasher.update(value_data)
        return hasher.hexdigest()[:32]
    
    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        content_hash = self._compute_hash(key_data, value_data)
        
        if content_hash in self.dedup_index:
            self.stats['dedup_hits'] += 1
            return True
        
        self.stats['dedup_misses'] += 1
        
        if len(self.gpu_cache) < self.gpu_capacity:
            self.gpu_cache[block_id] = (key_data, value_data)
        elif len(self.shm_cache) < self.shm_capacity:
            if self.gpu_cache:
                evict_id = next(iter(self.gpu_cache))
                self.shm_cache[evict_id] = self.gpu_cache.pop(evict_id)
            self.gpu_cache[block_id] = (key_data, value_data)
        
        self.dedup_index[content_hash] = block_id
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        if block_id in self.gpu_cache:
            return self.gpu_cache[block_id]
        return self.shm_cache.get(block_id)
    
    def clear(self):
        self.gpu_cache.clear()
        self.shm_cache.clear()
        self.dedup_index.clear()
    
    def get_stats(self) -> Dict:
        total = self.stats['dedup_hits'] + self.stats['dedup_misses']
        return {
            'gpu_blocks': len(self.gpu_cache),
            'shm_blocks': len(self.shm_cache),
            'unique_blocks': len(self.dedup_index),
            'dedup_ratio': self.stats['dedup_hits'] / total if total > 0 else 0
        }


###############################################################################
# 2. vLLM Adapter (GPU-only PagedAttention simulation)
###############################################################################

class VLLMAdapter(StorageAdapter):
    """vLLM: GPU-only PagedAttention (no offloading)."""
    
    def __init__(self, gpu_capacity=200):
        super().__init__("vLLM")
        self.gpu_cache = {}
        self.gpu_capacity = gpu_capacity
        self.evicted = 0
    
    def initialize(self) -> bool:
        return True
    
    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        if len(self.gpu_cache) >= self.gpu_capacity:
            # vLLM evicts oldest - no offload!
            evict_id = next(iter(self.gpu_cache))
            del self.gpu_cache[evict_id]
            self.evicted += 1
        
        self.gpu_cache[block_id] = (key_data, value_data)
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        return self.gpu_cache.get(block_id)  # Miss if evicted
    
    def clear(self):
        self.gpu_cache.clear()
        self.evicted = 0
    
    def get_stats(self) -> Dict:
        return {
            'gpu_blocks': len(self.gpu_cache),
            'evicted_blocks': self.evicted,
            'hit_ratio': len(self.gpu_cache) / (len(self.gpu_cache) + self.evicted) if self.evicted else 1.0
        }


###############################################################################
# 3. LMCache Adapter (Per-file Lustre storage)
###############################################################################

class LMCacheAdapter(StorageAdapter):
    """LMCache: Per-file disk storage (high metadata overhead on Lustre)."""
    
    def __init__(self, storage_path=None):
        super().__init__("LMCache")
        self.storage_path = Path(storage_path or SCRATCH / 'lmcache_store')
        self.index = {}
    
    def initialize(self) -> bool:
        self.storage_path.mkdir(parents=True, exist_ok=True)
        return True
    
    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        # LMCache uses per-file storage (one file per block)
        file_path = self.storage_path / f"{block_id}.bin"
        with open(file_path, 'wb') as f:
            f.write(struct.pack('<QQ', len(key_data), len(value_data)))
            f.write(key_data)
            f.write(value_data)
        self.index[block_id] = file_path
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        if block_id not in self.index:
            return None
        file_path = self.index[block_id]
        if not file_path.exists():
            return None
        with open(file_path, 'rb') as f:
            key_size, value_size = struct.unpack('<QQ', f.read(16))
            key_data = f.read(key_size)
            value_data = f.read(value_size)
        return key_data, value_data
    
    def clear(self):
        if self.storage_path.exists():
            shutil.rmtree(self.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index.clear()
    
    def get_stats(self) -> Dict:
        return {'stored_blocks': len(self.index)}


###############################################################################
# 4. HDF5 Adapter
###############################################################################

class HDF5Adapter(StorageAdapter):
    """HDF5: Single file storage with dataset-per-block."""
    
    def __init__(self, file_path=None):
        super().__init__("HDF5")
        self.file_path = Path(file_path or SCRATCH / 'hdf5_store.h5')
        self.h5file = None
        self.block_count = 0
    
    def initialize(self) -> bool:
        self.h5file = h5py.File(self.file_path, 'w')
        return True
    
    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        if self.h5file is None:
            return False
        grp = self.h5file.create_group(block_id)
        grp.create_dataset('key', data=np.frombuffer(key_data, dtype=np.uint8))
        grp.create_dataset('value', data=np.frombuffer(value_data, dtype=np.uint8))
        self.block_count += 1
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        if self.h5file is None or block_id not in self.h5file:
            return None
        grp = self.h5file[block_id]
        key_data = grp['key'][:].tobytes()
        value_data = grp['value'][:].tobytes()
        return key_data, value_data
    
    def clear(self):
        if self.h5file:
            self.h5file.close()
        if self.file_path.exists():
            self.file_path.unlink()
        self.h5file = h5py.File(self.file_path, 'w')
        self.block_count = 0
    
    def get_stats(self) -> Dict:
        return {'stored_blocks': self.block_count}


###############################################################################
# 5. PDC Adapter (simulated - actual PDC requires server)
###############################################################################

class PDCAdapter(StorageAdapter):
    """PDC: Proactive Data Containers (simulated object store)."""
    
    def __init__(self):
        super().__init__("PDC")
        self.objects = {}
        self.metadata = {}
    
    def initialize(self) -> bool:
        # Real PDC would: PDCinit(), PDCcont_create(), etc.
        return True
    
    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool:
        # PDC stores objects with rich metadata
        self.objects[block_id] = (key_data, value_data)
        self.metadata[block_id] = {
            'size': len(key_data) + len(value_data),
            'created': time.time()
        }
        return True
    
    def get(self, block_id: str) -> Optional[Tuple[bytes, bytes]]:
        return self.objects.get(block_id)
    
    def clear(self):
        self.objects.clear()
        self.metadata.clear()
    
    def get_stats(self) -> Dict:
        return {'stored_objects': len(self.objects)}


###############################################################################
# Benchmark Runner
###############################################################################

def run_benchmark(adapter: StorageAdapter, blocks: List[Tuple[str, bytes, bytes]], 
                  operation: str) -> BenchmarkResult:
    """Run write or read benchmark."""
    
    total_bytes = sum(len(k) + len(v) for _, k, v in blocks)
    
    if operation == 'write':
        start = time.perf_counter()
        for block_id, key_data, value_data in blocks:
            adapter.put(block_id, key_data, value_data)
        elapsed = time.perf_counter() - start
    else:  # read
        start = time.perf_counter()
        hits = 0
        for block_id, _, _ in blocks:
            if adapter.get(block_id):
                hits += 1
        elapsed = time.perf_counter() - start
    
    return BenchmarkResult(
        system=adapter.name,
        operation=operation,
        num_blocks=len(blocks),
        total_bytes=total_bytes,
        elapsed_seconds=elapsed,
        throughput_gbps=total_bytes / elapsed / 1e9 if elapsed > 0 else 0,
        latency_ms=elapsed / len(blocks) * 1000 if blocks else 0,
        ops_per_second=len(blocks) / elapsed if elapsed > 0 else 0,
        extra=adapter.get_stats()
    )


def run_dedup_benchmark(adapter: StorageAdapter, blocks: List[Tuple[str, bytes, bytes]], 
                        num_sessions: int = 10) -> BenchmarkResult:
    """Benchmark with prefix sharing (tests deduplication)."""
    
    prefix_blocks = blocks[:50]  # First 50 blocks are shared prefix
    unique_blocks = blocks[50:60]  # 10 unique blocks per session
    
    all_blocks = []
    for session in range(num_sessions):
        for bid, k, v in prefix_blocks:
            all_blocks.append((f"s{session}_{bid}", k, v))
        for bid, k, v in unique_blocks:
            all_blocks.append((f"s{session}_u_{bid}", k, v))
    
    total_bytes = sum(len(k) + len(v) for _, k, v in all_blocks)
    
    start = time.perf_counter()
    for block_id, key_data, value_data in all_blocks:
        adapter.put(block_id, key_data, value_data)
    elapsed = time.perf_counter() - start
    
    stats = adapter.get_stats()
    unique_stored = stats.get('unique_blocks', stats.get('stored_blocks', len(all_blocks)))
    
    return BenchmarkResult(
        system=adapter.name,
        operation='dedup_write',
        num_blocks=len(all_blocks),
        total_bytes=total_bytes,
        elapsed_seconds=elapsed,
        throughput_gbps=total_bytes / elapsed / 1e9 if elapsed > 0 else 0,
        latency_ms=elapsed / len(all_blocks) * 1000,
        ops_per_second=len(all_blocks) / elapsed if elapsed > 0 else 0,
        extra={
            'unique_stored': unique_stored,
            'total_requested': len(all_blocks),
            'dedup_ratio': 1.0 - unique_stored / len(all_blocks)
        }
    )


def main():
    print("=" * 70)
    print("5-System Comparison Benchmark")
    print("=" * 70)
    
    # Load data
    reader = AggregatedDataReader(DATA_DIR)
    block_ids = reader.get_block_ids()
    print(f"Found {len(block_ids)} blocks")
    
    num_test_blocks = min(100, len(block_ids))  # Use fewer for speed
    test_ids = block_ids[:num_test_blocks]
    
    print(f"Loading {num_test_blocks} blocks...")
    blocks = []
    for bid in test_ids:
        data = reader.read_block(bid)
        if data:
            blocks.append((bid, data[0], data[1]))
    
    print(f"Loaded {len(blocks)} blocks")
    if not blocks:
        print("No blocks loaded!")
        return
    
    block_size = len(blocks[0][1]) + len(blocks[0][2])
    print(f"Block size: {block_size / 1e6:.1f} MB\n")
    
    # Create adapters
    adapters = [
        CascadeAdapter(gpu_capacity=50, shm_capacity=100),
        VLLMAdapter(gpu_capacity=50),
        LMCacheAdapter(),
        HDF5Adapter(),
        PDCAdapter(),
    ]
    
    results = []
    
    # Run benchmarks
    for adapter in adapters:
        print(f"\n--- {adapter.name} ---")
        adapter.initialize()
        adapter.clear()
        
        # Write benchmark
        result = run_benchmark(adapter, blocks, 'write')
        print(f"Write: {result.throughput_gbps:.2f} GB/s, {result.latency_ms:.3f} ms/op")
        results.append(result)
        
        # Read benchmark
        result = run_benchmark(adapter, blocks, 'read')
        print(f"Read:  {result.throughput_gbps:.2f} GB/s, {result.latency_ms:.3f} ms/op")
        results.append(result)
        
        # Dedup benchmark (only for Cascade)
        if adapter.name == "Cascade":
            adapter.clear()
            result = run_dedup_benchmark(adapter, blocks)
            dedup_ratio = result.extra.get('dedup_ratio', 0) * 100
            print(f"Dedup: {dedup_ratio:.1f}% saved ({result.extra['unique_stored']}/{result.extra['total_requested']})")
            results.append(result)
        
        print(f"Stats: {adapter.get_stats()}")
    
    # Save results
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'data_dir': str(DATA_DIR),
            'num_blocks': len(blocks),
            'block_size_bytes': block_size
        },
        'results': [asdict(r) for r in results]
    }
    
    output_path = RESULTS_DIR / f'compare_5sys_{time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Summary table
    print("\n" + "=" * 80)
    print(f"{'System':<15} {'Operation':<12} {'Throughput':<15} {'Latency':<12} {'Notes':<20}")
    print("=" * 80)
    for r in results:
        notes = ""
        if 'dedup_ratio' in r.extra:
            notes = f"Dedup: {r.extra['dedup_ratio']*100:.0f}%"
        elif 'evicted_blocks' in r.extra:
            notes = f"Evicted: {r.extra['evicted_blocks']}"
        print(f"{r.system:<15} {r.operation:<12} {r.throughput_gbps:>10.2f} GB/s {r.latency_ms:>8.3f} ms  {notes:<20}")
    print("=" * 80)
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
PYEOF

echo ""
echo "Benchmark complete!"
