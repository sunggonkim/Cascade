#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:05:00
#SBATCH -J cascade_all
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/all_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/all_%j.err

module load python cudatoolkit
export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/src:$PYTHONPATH

echo "🚀 Cascade All-in-One Test (5min, 2 nodes)"
echo "Nodes: $SLURM_NODELIST"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

python3 << 'PYEOF'
import sys, time, os, json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

sys.path.insert(0, "/pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/src")

print("=" * 60)
print("🚀 Cascade Full Benchmark Suite")
print("=" * 60)

# GPU check
try:
    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    print(f"✅ GPU: {gpu_name}")
    HAS_GPU = True
except Exception as e:
    print(f"⚠️ No GPU: {e}")
    HAS_GPU = False

from cascade import CascadeStore, CascadeConfig, compute_block_id_from_bytes
from cascade.backends import GPUBackend, ShmBackend, ShardedIndex, CUDAStreamsManager

results = {}

# ============ 1. GPU Backend with Optimizations ============
print("\n" + "=" * 60)
print("📊 [1/3] GPU Backend (Pinned Memory + CUDA Streams)")
print("=" * 60)

if HAS_GPU:
    gpu = GPUBackend(capacity_gb=8.0, use_sharding=True, use_streams=True)
    
    # Generate 50 x 20MB blocks = 1GB
    print("   Generating 1GB test data (50 x 20MB)...")
    blocks = []
    for i in range(50):
        data = np.random.randn(10 * 1024 * 1024).astype(np.float16).tobytes()
        bid = compute_block_id_from_bytes(data)
        blocks.append((bid, data))
    
    # Write
    t0 = time.perf_counter()
    for bid, data in blocks:
        gpu.put(bid, data)
    write_time = time.perf_counter() - t0
    total_bytes = sum(len(d) for _, d in blocks)
    write_gbps = total_bytes / 1024**3 / write_time
    
    # Read
    t0 = time.perf_counter()
    for bid, _ in blocks:
        gpu.get(bid)
    read_time = time.perf_counter() - t0
    read_gbps = total_bytes / 1024**3 / read_time
    
    print(f"   ✅ Write: {write_gbps:.2f} GB/s")
    print(f"   ✅ Read:  {read_gbps:.2f} GB/s")
    results["gpu_write_gbps"] = write_gbps
    results["gpu_read_gbps"] = read_gbps
else:
    print("   ⏭️ Skipped (no GPU)")

# ============ 2. Full Cascade Store (GPU + SHM + Lustre) ============
print("\n" + "=" * 60)
print("📊 [2/3] Full CascadeStore (3-tier: GPU→SHM→Lustre)")
print("=" * 60)

config = CascadeConfig(
    gpu_capacity_gb=4.0 if HAS_GPU else 0,
    shm_capacity_gb=8.0,
    lustre_path="/pscratch/sd/s/sgkim/Skim-cascade/benchmark/cascade_store",
    dedup_enabled=True,
    prefix_aware=True,
)

store = CascadeStore(config)

# LLaMA-style blocks (smaller for speed): 20MB each
print("   Testing 30 x 20MB blocks (600MB)...")
blocks = []
for i in range(30):
    data = np.random.randn(10 * 1024 * 1024).astype(np.float16).tobytes()
    bid = compute_block_id_from_bytes(data)
    is_prefix = i < 5
    blocks.append((bid, data, is_prefix))

# Write
t0 = time.perf_counter()
for bid, data, is_prefix in blocks:
    store.put(bid, data, is_prefix=is_prefix)
store.flush()
write_time = time.perf_counter() - t0
total_bytes = sum(len(d) for _, d, _ in blocks)
write_gbps = total_bytes / 1024**3 / write_time
print(f"   Write: {write_gbps:.2f} GB/s")

# Read
t0 = time.perf_counter()
hits = 0
for bid, _, _ in blocks:
    if store.get(bid):
        hits += 1
read_time = time.perf_counter() - t0
read_gbps = total_bytes / 1024**3 / read_time
hit_rate = hits / len(blocks)
print(f"   Read:  {read_gbps:.2f} GB/s (hit rate: {hit_rate:.0%})")

# Dedup test
print("   Dedup test (re-insert 5 prefix blocks x 10 times)...")
t0 = time.perf_counter()
for _ in range(10):
    for bid, data, is_prefix in blocks[:5]:
        store.put(bid, data, is_prefix=True)
dedup_time = time.perf_counter() - t0

stats = store.get_stats()
dedup_hits = stats.get('dedup_hits', 0)
print(f"   Dedup hits: {dedup_hits}, time: {dedup_time*1000:.1f}ms")

# Parallel read
print("   Parallel read (8 threads)...")
def read_block(args):
    bid, _, _ = args
    return store.get(bid) is not None

t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(read_block, blocks))
par_time = time.perf_counter() - t0
par_gbps = total_bytes / 1024**3 / par_time
print(f"   Parallel: {par_gbps:.2f} GB/s")

results["cascade_write_gbps"] = write_gbps
results["cascade_read_gbps"] = read_gbps
results["cascade_parallel_gbps"] = par_gbps
results["cascade_hit_rate"] = hit_rate
results["cascade_dedup_hits"] = dedup_hits

store.cleanup()

# ============ 3. Multi-node Simulation ============
print("\n" + "=" * 60)
print("📊 [3/3] Multi-node Test (MPI rank simulation)")
print("=" * 60)

rank = int(os.environ.get('SLURM_PROCID', 0))
nnodes = int(os.environ.get('SLURM_NNODES', 1))
print(f"   Nodes: {nnodes}, This rank: {rank}")

# Each "node" creates separate store instance
node_store = CascadeStore(CascadeConfig(
    gpu_capacity_gb=0,
    shm_capacity_gb=2.0,
    lustre_path=f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/cascade_store/node_{rank}",
    dedup_enabled=True,
))

# Write node-local data
node_blocks = []
for i in range(10):
    data = np.random.randn(5 * 1024 * 1024).astype(np.float16).tobytes()
    bid = compute_block_id_from_bytes(data)
    node_blocks.append((bid, data))

t0 = time.perf_counter()
for bid, data in node_blocks:
    node_store.put(bid, data, is_prefix=False)
node_write_time = time.perf_counter() - t0

t0 = time.perf_counter()
for bid, _ in node_blocks:
    node_store.get(bid)
node_read_time = time.perf_counter() - t0

node_bytes = sum(len(d) for _, d in node_blocks)
print(f"   Node {rank} write: {node_bytes/1024**3/node_write_time:.2f} GB/s")
print(f"   Node {rank} read:  {node_bytes/1024**3/node_read_time:.2f} GB/s")

results["multinode_write_gbps"] = node_bytes/1024**3/node_write_time
results["multinode_read_gbps"] = node_bytes/1024**3/node_read_time

node_store.cleanup()

# ============ SUMMARY ============
print("\n" + "=" * 60)
print("📊 FINAL SUMMARY")
print("=" * 60)
print(f"  GPU Write:       {results.get('gpu_write_gbps', 'N/A'):.2f} GB/s" if 'gpu_write_gbps' in results else "  GPU: Skipped")
print(f"  GPU Read:        {results.get('gpu_read_gbps', 'N/A'):.2f} GB/s" if 'gpu_read_gbps' in results else "")
print(f"  Cascade Write:   {results['cascade_write_gbps']:.2f} GB/s")
print(f"  Cascade Read:    {results['cascade_read_gbps']:.2f} GB/s")
print(f"  Cascade Par(8T): {results['cascade_parallel_gbps']:.2f} GB/s")
print(f"  Dedup Hits:      {results['cascade_dedup_hits']}")
print(f"  Multi-node:      {results['multinode_read_gbps']:.2f} GB/s (per node)")
print("=" * 60)

# Save
with open(f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/all_tests_{datetime.now():%Y%m%d_%H%M%S}.json", 'w') as f:
    json.dump(results, f, indent=2)

print("✅ All tests complete!")
PYEOF

echo "Done at $(date)"
