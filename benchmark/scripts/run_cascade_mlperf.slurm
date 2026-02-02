#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J cascade_mlperf
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/mlperf_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/mlperf_%j.err

echo "============================================================"
echo "🚀 Cascade MLPerf Benchmark on GPU Node"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Date: $(date)"
echo "============================================================"

# Load modules
module load python
module load cudatoolkit

# Set paths
export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/src:/pscratch/sd/s/sgkim/Skim-cascade/benchmark:$PYTHONPATH
export CASCADE_LUSTRE_PATH=/pscratch/sd/s/sgkim/Skim-cascade/benchmark/cascade_store

# Create directories
mkdir -p $CASCADE_LUSTRE_PATH
mkdir -p /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs
mkdir -p /pscratch/sd/s/sgkim/Skim-cascade/benchmark/results

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# Install cupy if needed
pip show cupy-cuda12x > /dev/null 2>&1 || pip install --user cupy-cuda12x

echo ""
echo "📊 Running MLPerf Benchmark..."
echo ""

python3 << 'PYTHON_EOF'
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/src")
sys.path.insert(0, "/pscratch/sd/s/sgkim/Skim-cascade/benchmark")

print("=" * 70)
print("🚀 Cascade MLPerf Benchmark - GPU Node Full Test")
print("=" * 70)

# Check GPU availability
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print(f"✅ GPU Available: {cp.cuda.Device().name}")
except (ImportError, Exception) as e:
    GPU_AVAILABLE = False
    print(f"⚠️ GPU not available: {e}")

from adapters.cascade_adapter import CascadeAdapter
from adapters.base import BenchmarkStats

# LLaMA-70B dimensions
NUM_LAYERS = 80
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_TOKENS = 256

# Test params - larger for GPU
NUM_BLOCKS = 200  # 16GB total
NUM_THREADS = 8

def generate_llama_block(block_idx: int):
    """Generate a single LLaMA-70B style KV block."""
    rng = np.random.RandomState(block_idx)
    shape = (BLOCK_TOKENS, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)
    
    key = rng.randn(*shape).astype(np.float16)
    value = rng.randn(*shape).astype(np.float16)
    
    import hashlib
    content = key.tobytes() + value.tobytes()
    block_id = hashlib.sha256(content).hexdigest()[:32]
    
    return {
        "block_id": block_id,
        "key_data": key.tobytes(),
        "value_data": value.tobytes(),
        "is_prefix": block_idx < 20,  # 20 prefix blocks
        "size_bytes": key.nbytes + value.nbytes
    }

# Generate test data
print(f"\n📝 Generating {NUM_BLOCKS} LLaMA-70B KV blocks...")
blocks = [generate_llama_block(i) for i in range(NUM_BLOCKS)]
total_size_gb = sum(b["size_bytes"] for b in blocks) / 1024 / 1024 / 1024
print(f"   Total data size: {total_size_gb:.1f} GB")
print(f"   Block size: {blocks[0]['size_bytes'] / 1024 / 1024:.2f} MB each")

# Initialize adapter with GPU if available
config = {
    "use_gpu": GPU_AVAILABLE,
    "gpu_capacity_gb": 32.0,
    "shm_capacity_gb": 64.0,
    "lustre_path": "/pscratch/sd/s/sgkim/Skim-cascade/benchmark/cascade_store"
}

print(f"\n📦 Initializing Cascade adapter (GPU={GPU_AVAILABLE})...")
adapter = CascadeAdapter(config)
if not adapter.initialize():
    print("❌ Failed to initialize adapter")
    sys.exit(1)

results = {}

# ========== 1. SEQUENTIAL WRITE ==========
print(f"\n[1/5] Sequential Write ({NUM_BLOCKS} blocks)...")
write_stats = BenchmarkStats(system_name="Cascade", operation="seq_write")

start = time.perf_counter()
for block in blocks:
    t0 = time.perf_counter()
    success = adapter.put(block["block_id"], block["key_data"], block["value_data"])
    latency = (time.perf_counter() - t0) * 1000
    if success:
        write_stats.total_ops += 1
        write_stats.total_bytes += block["size_bytes"]
        write_stats.latencies.append(latency)

adapter.flush()
write_stats.duration_seconds = time.perf_counter() - start

print(f"      Throughput: {write_stats.throughput_gbps:.2f} GB/s")
print(f"      Latency: {write_stats.avg_latency_ms:.2f} ms (avg), {write_stats.p99_latency_ms:.2f} ms (p99)")
results["seq_write"] = write_stats.to_dict()

# ========== 2. SEQUENTIAL READ ==========
print(f"\n[2/5] Sequential Read ({NUM_BLOCKS} blocks)...")
read_stats = BenchmarkStats(system_name="Cascade", operation="seq_read")

start = time.perf_counter()
for block in blocks:
    t0 = time.perf_counter()
    result = adapter.get(block["block_id"])
    latency = (time.perf_counter() - t0) * 1000
    if result:
        read_stats.total_ops += 1
        read_stats.total_bytes += block["size_bytes"]
        read_stats.hits += 1
        read_stats.latencies.append(latency)
    else:
        read_stats.misses += 1

read_stats.duration_seconds = time.perf_counter() - start

print(f"      Throughput: {read_stats.throughput_gbps:.2f} GB/s")
print(f"      Hit Rate: {read_stats.hit_rate:.1%}")
print(f"      Latency: {read_stats.avg_latency_ms:.2f} ms (avg), {read_stats.p99_latency_ms:.2f} ms (p99)")
results["seq_read"] = read_stats.to_dict()

# ========== 3. PARALLEL READ ==========
print(f"\n[3/5] Parallel Read ({NUM_THREADS} threads, {NUM_BLOCKS} blocks)...")
par_stats = BenchmarkStats(system_name="Cascade", operation="par_read")

def read_block(block):
    t0 = time.perf_counter()
    result = adapter.get(block["block_id"])
    return (result is not None, block["size_bytes"], (time.perf_counter() - t0) * 1000)

start = time.perf_counter()
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    results_list = list(executor.map(read_block, blocks))

for success, size, latency in results_list:
    if success:
        par_stats.total_ops += 1
        par_stats.total_bytes += size
        par_stats.hits += 1
        par_stats.latencies.append(latency)
    else:
        par_stats.misses += 1

par_stats.duration_seconds = time.perf_counter() - start

print(f"      Throughput: {par_stats.throughput_gbps:.2f} GB/s")
print(f"      Latency: {par_stats.avg_latency_ms:.2f} ms (avg)")
print(f"      Speedup: {par_stats.throughput_gbps / max(read_stats.throughput_gbps, 0.01):.1f}x")
results["par_read"] = par_stats.to_dict()

# ========== 4. SHARED PREFIX (DEDUP) ==========
print(f"\n[4/5] Shared Prefix Test (100 sessions)...")
adapter.clear()

prefix_blocks = [b for b in blocks if b["is_prefix"]]
NUM_SESSIONS = 100

dedup_stats = BenchmarkStats(system_name="Cascade", operation="shared_prefix")

start = time.perf_counter()
for session in range(NUM_SESSIONS):
    for block in prefix_blocks:
        t0 = time.perf_counter()
        adapter.put_prefix(block["block_id"], block["key_data"], block["value_data"])
        latency = (time.perf_counter() - t0) * 1000
        dedup_stats.total_ops += 1
        dedup_stats.total_bytes += block["size_bytes"]
        dedup_stats.latencies.append(latency)

dedup_stats.duration_seconds = time.perf_counter() - start

stats = adapter.get_stats()
dedup_hits = stats.get('dedup_hits', 0)
dedup_rate = dedup_hits / max(NUM_SESSIONS * len(prefix_blocks), 1) * 100

print(f"      Sessions: {NUM_SESSIONS}")
print(f"      Prefix blocks: {len(prefix_blocks)}")
print(f"      Dedup hits: {dedup_hits}")
print(f"      Dedup rate: {dedup_rate:.1f}%")
results["dedup"] = {**dedup_stats.to_dict(), "dedup_hits": dedup_hits, "dedup_rate": dedup_rate}

# ========== 5. MIXED WORKLOAD ==========
print(f"\n[5/5] Mixed Workload (80% read, 20% write)...")
adapter.clear()

# Pre-populate
for block in blocks[:50]:
    adapter.put(block["block_id"], block["key_data"], block["value_data"])

mixed_stats = BenchmarkStats(system_name="Cascade", operation="mixed_80_20")
NUM_OPS = 500

start = time.perf_counter()
for i in range(NUM_OPS):
    if np.random.random() < 0.8:  # 80% read
        block = blocks[np.random.randint(0, 50)]
        t0 = time.perf_counter()
        result = adapter.get(block["block_id"])
        latency = (time.perf_counter() - t0) * 1000
        mixed_stats.latencies.append(latency)
        if result:
            mixed_stats.hits += 1
        else:
            mixed_stats.misses += 1
    else:  # 20% write
        block = blocks[np.random.randint(50, NUM_BLOCKS)]
        t0 = time.perf_counter()
        adapter.put(block["block_id"], block["key_data"], block["value_data"])
        latency = (time.perf_counter() - t0) * 1000
        mixed_stats.latencies.append(latency)
    
    mixed_stats.total_ops += 1

mixed_stats.duration_seconds = time.perf_counter() - start

print(f"      Ops/s: {mixed_stats.ops_per_second:,.0f}")
print(f"      Latency: {mixed_stats.avg_latency_ms:.2f} ms (avg)")
results["mixed_80_20"] = mixed_stats.to_dict()

# Final stats
final_stats = adapter.get_stats()
adapter.close()

# ========== SUMMARY ==========
print("\n" + "=" * 70)
print("📊 MLPerf Benchmark Summary")
print("=" * 70)
print(f"  GPU:              {'Yes' if GPU_AVAILABLE else 'No (SHM only)'}")
print(f"  Data Size:        {total_size_gb:.1f} GB ({NUM_BLOCKS} blocks)")
print(f"  Block Size:       {blocks[0]['size_bytes'] / 1024 / 1024:.0f} MB")
print("-" * 70)
print(f"  Seq Write:        {results['seq_write']['throughput_gbps']:.2f} GB/s")
print(f"  Seq Read:         {results['seq_read']['throughput_gbps']:.2f} GB/s")
print(f"  Par Read ({NUM_THREADS}T):    {results['par_read']['throughput_gbps']:.2f} GB/s")
print(f"  Mixed 80/20:      {results['mixed_80_20']['ops_per_second']:.0f} ops/s")
print(f"  Dedup Rate:       {results['dedup']['dedup_rate']:.0f}%")
print("=" * 70)

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_path = Path("/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results")
results_file = results_path / f"mlperf_cascade_{timestamp}.json"

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n📁 Results saved to: {results_file}")
print("✅ Benchmark complete!")
PYTHON_EOF

echo ""
echo "============================================================"
echo "Job completed at $(date)"
echo "============================================================"
