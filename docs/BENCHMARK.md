# 📈 Benchmark Guide

This document explains how to run benchmarks for the Cascade KV cache system on NERSC Perlmutter.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Generation](#data-generation)
3. [Single-System Benchmarks](#single-system-benchmarks)
4. [Multi-System Comparison](#multi-system-comparison)
5. [Scaling Experiments](#scaling-experiments)
6. [Result Analysis](#result-analysis)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Environment Setup

```bash
# On Perlmutter login node
module load python cudatoolkit cray-mpich libfabric

export SCRATCH=/pscratch/sd/s/sgkim
export CASCADE_HOME=$SCRATCH/Skim-cascade
export DATA_DIR=$SCRATCH/cascade_kv_cache
export PYTHONPATH=$CASCADE_HOME:$CASCADE_HOME/python_pkgs_py312:$PYTHONPATH
```

### 2. Dependencies

| Dependency | Location | Build Status |
|------------|----------|--------------|
| Redis | `third_party/redis/` | Pre-built |
| Mercury | `third_party/mercury/install/` | Built |
| PDC | `third_party/pdc/install/` | Built |
| LMCache | `third_party/LMCache/` | Python only |
| h5py | `python_pkgs_py312/` | Installed |

### 3. Verify Installation

```bash
# Test imports
python -c "
import sys
sys.path.insert(0, '$CASCADE_HOME/third_party/LMCache')
import lmcache
print('LMCache OK')
"

# Test Redis
$CASCADE_HOME/third_party/redis/src/redis-server --version

# Test PDC
$CASCADE_HOME/third_party/pdc/install/bin/pdc_server --help
```

---

## Data Generation

### Generate 500GB KV Cache Data

The benchmark uses realistic LLaMA-70B KV cache data with prefix sharing patterns.

```bash
# Submit data generation job
sbatch benchmark/scripts/generate_data.sh
```

**What it creates:**
- 3,200 blocks × 168MB each = ~500GB total
- 50 shared system prompts (prefix blocks)
- 10 unique continuation blocks per session
- Content-addressed block IDs (SHA-256)

**Data format:**
```
$SCRATCH/cascade_kv_cache/
├── global_index.json          # Block ID → file/offset mapping
├── agg_rank000_000000.bin    # Aggregated blocks (rank 0)
├── agg_rank000_000001.bin
├── agg_rank001_000000.bin    # Aggregated blocks (rank 1)
└── ...
```

### Verify Data

```bash
# Check data size
du -sh $SCRATCH/cascade_kv_cache/
# Expected: ~500GB

# Check block count
python -c "
import json
with open('$SCRATCH/cascade_kv_cache/global_index.json') as f:
    idx = json.load(f)
    print(f'Blocks: {len(idx[\"blocks\"])}')
"
# Expected: 3200
```

---

## Single-System Benchmarks

### Interactive Mode (Debug Queue)

```bash
# Get interactive allocation
salloc -N 1 -C gpu -q debug -t 00:30:00 -A m1248 --gpus-per-node=4

# Run Cascade benchmark
srun python benchmark/run_benchmark.py --system cascade --workload write_read

# Run specific baseline
srun python benchmark/run_benchmark.py --system lmcache --workload write_read
```

### Batch Mode

```bash
# Single node, single system
sbatch --export=SYSTEM=cascade benchmark/scripts/run_single_node.sh
```

---

## Multi-System Comparison

### Full 6-System Benchmark

The main comparison benchmark runs all 6 systems on the same data:

```bash
# Submit the full benchmark (2 nodes, 8 ranks, 30 min)
sbatch benchmark/scripts/full_6sys_bench.sh
```

**Systems compared:**
1. **Cascade** - Our tiered system with deduplication
2. **vLLM** - GPU-only PagedAttention
3. **LMCache** - Per-file Lustre storage
4. **HDF5** - Single HDF5 file with datasets
5. **Redis** - In-memory with Lustre persistence
6. **PDC** - Proactive Data Containers (aggregated)

**Configuration:**
```bash
# In full_6sys_bench.sh
NODES=2
RANKS_PER_NODE=4
BLOCKS_PER_RANK=400
BLOCK_SIZE=168MB  # LLaMA-70B, 256 tokens

# Cascade tier capacities (blocks)
GPU_CAPACITY=50      # 8.4GB
SHM_CAPACITY=100     # 16.8GB
# Overflow → Lustre
```

**Output:**
```
benchmark/logs/full_6sys_<jobid>.out   # Console output
benchmark/results/full_6sys_<jobid>.json  # Structured results
```

---

## Scaling Experiments

### Weak Scaling (1-32 nodes)

```bash
# 1 node
sbatch --nodes=1 benchmark/scripts/run_scaling.sh

# 4 nodes
sbatch --nodes=4 benchmark/scripts/run_scaling.sh

# 32 nodes (need regular queue, not debug)
sbatch --nodes=32 -q regular -t 01:00:00 benchmark/scripts/run_scaling.sh
```

### Strong Scaling

```bash
# Fixed 500GB data, vary node count
for N in 1 2 4 8 16 32; do
    sbatch --nodes=$N --export=DATA_SIZE=500GB benchmark/scripts/run_strong_scaling.sh
done
```

---

## Result Analysis

### Parse JSON Results

```python
import json
from pathlib import Path

results_dir = Path("benchmark/results")

# Find latest result
latest = sorted(results_dir.glob("full_6sys_*.json"))[-1]

with open(latest) as f:
    data = json.load(f)

# Print summary table
print(f"{'System':<12} {'Write GB/s':>12} {'Read GB/s':>12} {'Dedup':>8}")
print("-" * 48)

for system, res in data.items():
    write = res.get('write_throughput_gbps', 0)
    read = res.get('read_throughput_gbps', 0)
    dedup = res.get('dedup_ratio', 1)
    print(f"{system:<12} {write:>12.2f} {read:>12.2f} {dedup:>7.1f}x")
```

### Generate Paper Figures

```bash
cd paper/
python generate_figures.py --results ../benchmark/results/full_6sys_*.json
```

This generates:
- `Figures/throughput_comparison.pdf`
- `Figures/dedup_chart.pdf`
- `Figures/scaling.pdf`

---

## Benchmark Metrics

### Primary Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| Write Throughput | Block write rate | GB/s |
| Read Throughput | Block read rate | GB/s |
| Deduplication Ratio | Unique blocks / total requests | X times |
| Cache Hit Rate | Hits / (hits + misses) × 100 | % |
| Lustre Writes | Blocks written to persistent storage | count |

### Cascade-Specific Metrics

| Metric | Description |
|--------|-------------|
| `gpu_hits` | Blocks served from GPU HBM |
| `shm_hits` | Blocks served from shared memory |
| `lustre_hits` | Blocks read from Lustre |
| `dedup_hits` | Requests satisfied by dedup lookup |
| `evictions` | Blocks evicted from higher tiers |

---

## SLURM Job Parameters

### Debug Queue (Testing)

```bash
#SBATCH -q debug
#SBATCH -t 00:30:00    # Max 30 minutes
#SBATCH -N 1-4         # Max 4 nodes
```

### Regular Queue (Production)

```bash
#SBATCH -q regular
#SBATCH -t 02:00:00    # Up to 24 hours
#SBATCH -N 32          # Up to 256 nodes
```

### GPU Configuration

```bash
#SBATCH -C gpu
#SBATCH --gpus-per-node=4    # 4x A100-40GB per node
#SBATCH --ntasks-per-node=4  # 1 rank per GPU
```

---

## Troubleshooting

### "Redis connection refused"

```bash
# Redis not running - start it manually
$CASCADE_HOME/third_party/redis/src/redis-server --port 6380 --daemonize yes

# Check it's running
$CASCADE_HOME/third_party/redis/src/redis-cli -p 6380 ping
```

### "PDC server failed to start"

```bash
# Check library path
export LD_LIBRARY_PATH=$CASCADE_HOME/third_party/mercury/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:$LD_LIBRARY_PATH

# Verify dependencies
ldd $CASCADE_HOME/third_party/pdc/install/bin/pdc_server
```

### "Lustre quota exceeded"

```bash
# Check quota
lfs quota -u $USER $SCRATCH

# Clean old benchmark data
rm -rf $SCRATCH/cascade_kv_cache_old/
rm -rf $SCRATCH/cascade_lustre_*/
rm -rf $SCRATCH/redis_data_*/
```

### "Out of GPU memory"

```bash
# Reduce GPU capacity in benchmark script
GPU_CAPACITY=25  # 4.2GB instead of 50 blocks
```

### "Job killed - time limit"

```bash
# Reduce data size or use regular queue
sbatch -q regular -t 02:00:00 benchmark/scripts/full_6sys_bench.sh
```

---

## Quick Reference

### Submit Benchmark

```bash
cd $CASCADE_HOME
sbatch benchmark/scripts/full_6sys_bench.sh
```

### Check Job Status

```bash
squeue -u $USER
```

### View Output

```bash
tail -f benchmark/logs/full_6sys_<jobid>.out
```

### Parse Results

```bash
cat benchmark/results/full_6sys_<jobid>.json | python -m json.tool
```
