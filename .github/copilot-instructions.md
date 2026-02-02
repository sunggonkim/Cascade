# Cascade: HPC-Scale KV Cache Storage System

## Project Overview

Cascade is a **4-tier hierarchical KV cache storage system** for LLM inference on NERSC Perlmutter.
**Target:** SC'26 (Supercomputing Conference) paper submission.

**Goal:** Beat LMCache with HPC-native multi-node scalability (up to 256 nodes).

**Critical Constraint:** Do NOT simulate any baseline. Use **REAL implementations** from `third_party/`.

---

## Directory Structure

```
/pscratch/sd/s/sgkim/Skim-cascade/
├── cascade_Code/src/cascade/    # Main Cascade implementation
│   ├── core.py                  # KVBlock, ContentAddressedHasher
│   ├── backends.py              # GPUBackend, ShmBackend, LustreBackend
│   ├── network.py               # MPI communication, GlobalAddressSpace
│   ├── prefix_tree.py           # SemanticEvictionPolicy
│   ├── unified_store.py         # CascadeStore (main entry point)
│   ├── tiered_store.py          # TieredStore (distributed version)
│   └── lustre.py                # AggregatedLustreStore (HPC-optimized)
├── benchmark/                   # Benchmark framework
│   ├── adapters/                # System-specific adapters (base.py, cascade_adapter.py, etc.)
│   ├── data/                    # Shared KV cache data (aggregated blocks)
│   ├── results/                 # Experiment outputs (.json)
│   ├── scripts/                 # SLURM scripts
│   └── config.py                # Global configuration
├── third_party/                 # Competitor implementations (REAL, no simulation)
│   ├── LMCache/
│   ├── pdc/
│   ├── redis/
│   └── vllm/
└── paper/                       # SC'26 LaTeX paper
    ├── main.tex
    └── Figures/
```

---

## Core Architecture (4-Tier)

```
Tier 1: GPU HBM     (40GB × 4 = 160GB/node, 1555 GB/s)
   ↓ evict (async)
Tier 2: Local DRAM  (/dev/shm, 128GB/node, 204 GB/s)
   ↓ MPI transfer (Slingshot-11, 100 GB/s)
Tier 3: Remote DRAM (aggregate across nodes)
   ↓ async prefetch
Tier 4: Lustre PFS  ($SCRATCH, 44PB, 7.8 TB/s read)
```

---

## Key Differentiators vs LMCache

| Feature | LMCache | Cascade |
|---------|---------|---------|
| Block ID | Session-specific | **Content-addressed (SHA-256)** |
| Deduplication | ❌ | ✅ (automatic) |
| Multi-node | ❌ | ✅ (MPI + Slingshot) |
| Eviction | LRU | **Semantic (ref-count + prefix)** |
| Storage tiers | 2 | **4** |

---

## Critical Code Patterns

### Content-Addressed Block ID (core.py)
```python
# ALWAYS use this pattern for block ID generation
def compute_block_id(key_data: np.ndarray, value_data: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(key_data.tobytes())
    hasher.update(value_data.tobytes())
    return hasher.hexdigest()[:32]
```

### Cascade Eviction Flow (unified_store.py)
```python
def put(self, block_id, data, is_prefix=False):
    if self.gpu.put(block_id, data):
        return True
    # GPU full → evict to SHM
    evicted_id, evicted_data = self.gpu.evict_lru()
    if is_prefix:
        self.shm.put(evicted_id, evicted_data)  # Prefix protected
    else:
        self.lustre.put(evicted_id, evicted_data)  # Cold storage
```

### Lustre Optimization (REQUIRED for HPC)
```python
# Always use stripe config for aggregated files
subprocess.run(["lfs", "setstripe", "-c", "16", "-S", "4m", str(path)])
# Files: agg_rank{rank:03d}_{file_id:06d}.bin (rank-specific, no lock contention)
```

### LLaMA-70B KV Cache Dimensions (config.py)
```python
# Per token: 2 * 80 * 8 * 128 * 2 = 327,680 bytes = 320KB
num_layers: int = 80
num_kv_heads: int = 8    # GQA
head_dim: int = 128
dtype: str = "float16"
```

---

## Perlmutter Environment

```bash
# Required modules
module load python cudatoolkit cray-mpich

# Environment variables
export SCRATCH=/pscratch/sd/s/sgkim
export HF_HOME=$SCRATCH/hf_cache

# Run distributed
srun -N 4 --gpus-per-node=4 python -m cascade.distributed
```

---

## Benchmark Rules (CRITICAL)

1. **NO SIMULATION** - All systems must use real implementations from `third_party/`
2. **Real Model** - Use MLPerf LLaMA-2-70B or LLaMA-3-70B
3. **Shared Data** - Single aggregated data directory, all systems read same blocks
4. **HPC Format** - Blocks aggregated into large files (not 1 file per block)
5. **Fair Comparison** - Use adapters in `benchmark/adapters/` for uniform interface

---

## Adapter Interface (benchmark/adapters/base.py)

All storage systems must implement `StorageAdapter`:
```python
class StorageAdapter(ABC):
    @abstractmethod
    def initialize(self) -> bool: pass
    
    @abstractmethod
    def put(self, block_id: str, key_data: bytes, value_data: bytes) -> bool: pass
    
    @abstractmethod
    def get(self, block_id: str) -> Optional[tuple]: pass
    
    @abstractmethod
    def contains(self, block_id: str) -> bool: pass
    
    @abstractmethod
    def clear(self) -> None: pass
```

Available adapters:
- `cascade_adapter.py` - Our system (CascadeStore)
- `lmcache_adapter.py` - Baseline (LMCache from third_party)
- `hdf5_adapter.py` - HDF5 backend
- `pdc_adapter.py` - Proactive Data Containers
- `redis_adapter.py` - Redis for Lustre

---

## File Naming Conventions

- **Blocks:** `{content_hash[:32]}.npz` (content-addressed)
- **Aggregated:** `agg_rank{rank:03d}_{file_id:06d}.bin`
- **Index:** `index_rank{rank:03d}.pkl`, `index.json`
- **Results:** `{system}_{workload}_{nodes}n_{timestamp}.json`

---

## Developer Workflows

### Running Benchmarks
```bash
# Single system
sbatch benchmark/scripts/run_single_node.sh

# Multi-node scaling
sbatch benchmark/scripts/run_scaling.sh
```
**Important:** Do not run heavy benchmarks on login node. Use `sbatch` or `salloc`.

### Testing Commands
```bash
# Smoke test (login node)
cd /pscratch/sd/s/sgkim/Skim-cascade/cascade_Code
python -c "from src.cascade.unified_store import CascadeStore; print('OK')"

# Run single adapter test
python -m benchmark.run_benchmark --system cascade --workload read_latency
```

---

## Key Conventions

1. **"Real" Implementation:** No mocks/simulations for critical paths. Wrap actual libraries.
2. **Explicit Imports:** Cascade modules imported from `cascade_Code/src/cascade`.
3. **HPC Awareness:** Code is MPI-aware via SLURM env vars:
   - `SLURM_PROCID`: Rank ID
   - `SLURM_NTASKS`: World Size
4. **Hardware Fairness:** Match GPU capacity settings across systems for fair comparison.
5. **Content-Addressed:** All block IDs computed from content hash (SHA-256[:32]).

---

## Paper Sync

After experiments, update `/pscratch/sd/s/sgkim/Skim-cascade/paper/`:
- `4. Evaluation.tex` - Results from `benchmark/results/*.json`
- `3. Design.tex` - Must match actual code implementation
- `Figures/` - Generated from result analysis scripts
