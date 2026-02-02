# 🚀 Cascade: HPC-Scale KV Cache Storage for LLM Inference

[![SC'26](https://img.shields.io/badge/Target-SC'26-blue.svg)](https://supercomputing.org/)
[![Perlmutter](https://img.shields.io/badge/Platform-NERSC%20Perlmutter-green.svg)](https://docs.nersc.gov/systems/perlmutter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Cascade** is a **4-tier hierarchical KV cache storage system** designed for HPC-scale LLM inference.

> 📝 **Paper Status**: SC'26 submission in progress  
> ⚠️ **Benchmark Status**: Real benchmarks running - results pending

---

## 🎯 The Problem

LLM inference is **memory-bound**: 80% of time is spent loading KV cache from memory. Current solutions fail at HPC scale:

- **GPU-only systems** (vLLM): Limited to ~40GB per GPU, causing evictions
- **Per-file storage** (LMCache): Metadata overhead on parallel filesystems
- **In-memory stores** (Redis): Network serialization bottleneck

---

## 💡 Core Contributions

### 1. **Tiered Storage Hierarchy** (GPU → SHM → Lustre)

| Tier | Bandwidth | Capacity | Latency |
|------|-----------|----------|---------|
| GPU HBM | ~1,500 GB/s | 40GB × 4 = 160GB/node | μs |
| Shared Memory | ~30-50 GB/s | 128GB/node | μs |
| Lustre (aggregated) | ~5-10 GB/s | 44PB | ms |
| Lustre (per-file) | ~0.5-2 GB/s | — | 10s ms |

### 2. **Content-Addressed Deduplication**

```python
# Session-based ID (existing):           Content-based ID (Cascade):
block_id = f"session_{user_id}_{seq}"  → block_id = sha256(key + value)[:32]

# 50 users × same prompt = 50 blocks   → 1 block (deduplication)
```

### 3. **Aggregated Lustre I/O**

Multiple blocks per file with `lfs setstripe -c 16 -S 4m` for optimal striping.

---

## 📊 Benchmark Results

> ✅ **REAL C++ implementation benchmarks completed** (Job 48413611)
> 
> 4 nodes × 4 ranks = 16 ranks, 16GB total data, **ACTUAL cascade_cpp, LMCache, PDC implementations**

### Aggregated Results (16 ranks, 4 nodes)

| Storage System | Write/Rank | Write Total | Read/Rank | Read Total | Real Implementation |
|----------------|------------|-------------|-----------|------------|---------------------|
| **🏆 Cascade C++** | **5.01 GB/s** | **80.14 GB/s** | 3.58 GB/s | 57.25 GB/s | `cascade_cpp: ShmBackend(mmap) + LustreBackend` |
| LMCache | 0.87 GB/s | 13.96 GB/s | 7.37 GB/s | 117.95 GB/s | `lmcache.v1.storage_backend.local_disk_backend` |
| PDC | 0.84 GB/s | 13.49 GB/s | 7.93 GB/s | 126.80 GB/s | `third_party/pdc + pdc_server` |
| Redis | 0.10 GB/s | 1.59 GB/s | 0.14 GB/s | 2.20 GB/s | `redis-py + redis-server` |
| HDF5 | 0.05 GB/s | 0.85 GB/s | 1.31 GB/s | 20.91 GB/s | `h5py with gzip compression` |

### 🚀 Key Findings

1. **Cascade Write: 5.7×–6× faster** than LMCache/PDC
   - ShmBackend uses SSE2 streaming stores (cache-bypass)
   - mmap with MADV_HUGEPAGE for optimal memory access
   
2. **LMCache/PDC Read faster** - filesystem cached reads
   - Cascade's SHM-first policy needs prefetch optimization

3. **Redis/HDF5** bottlenecked by network and compression

### Implementation Verified

All benchmarks use **ACTUAL implementations** from this repo:
- ✅ **Cascade C++**: `cascade_Code/cpp/cascade_cpp.cpython-312.so` - real mmap, io_uring
- ✅ **LMCache**: `third_party/LMCache` - real disk backend
- ✅ **PDC**: `third_party/pdc/install/bin/pdc_server` - real PDC server
- ✅ **Redis**: `third_party/redis/src/redis-server` - real Redis
- ✅ **HDF5**: `h5py` with gzip compression

Raw data: [benchmark/results/real_systems_48413611_aggregated.json](benchmark/results/real_systems_48413611_aggregated.json)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION LAYER                               │
│                         (vLLM, LMCache, custom inference)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CASCADE STORE                                   │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────────────────────────┐ │
│  │ Dedup Index  │  │  Tier Manager  │  │    Semantic Eviction Policy     │ │
│  │  (SHA-256)   │  │  (GPU→SHM→L)   │  │  (LRU + prefix-aware + refcnt)  │ │
│  └──────────────┘  └────────────────┘  └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
          │                    │                        │
          ▼                    ▼                        ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐
│   GPU Backend    │  │   SHM Backend    │  │      Lustre Backend          │
│   (CUDA malloc)  │  │   (mmap /dev/shm)│  │  (Aggregated + Striped I/O)  │
│   1,555 GB/s     │  │   33-45 GB/s     │  │  6.8-8 GB/s (aggregated)     │
│   40GB × 4 GPUs  │  │   128 GB/node    │  │  44 PB (Perlmutter $SCRATCH) │
└──────────────────┘  └──────────────────┘  └──────────────────────────────┘
```

### Memory Hierarchy Details

| Tier | Capacity | Bandwidth | Latency | Use Case |
|------|----------|-----------|---------|----------|
| **GPU HBM** | 40GB × 4 = 160GB/node | 1,555 GB/s | μs | Hot KV cache (active inference) |
| **Shared Memory** | 128GB/node (/dev/shm) | 33-45 GB/s | μs | Warm data (recently evicted) |
| **Lustre PFS** | 44PB ($SCRATCH) | 6.8-8 GB/s | 10s ms | Cold data (persistent storage) |
| **Per-file Lustre** | — | 0.2-1.3 GB/s | 100s ms | ❌ Avoid (LMCache pattern) |

---

## 🚀 Quick Start (NERSC Perlmutter)

### Prerequisites

```bash
# Login to Perlmutter
ssh <username>@perlmutter.nersc.gov

# Clone repository
cd $SCRATCH
git clone https://github.com/sunggonkim/Cascade.git
cd Cascade
```

### Environment Setup

```bash
# Load required modules
module load python cudatoolkit cray-mpich libfabric

# Set environment variables
export SCRATCH=/pscratch/sd/s/sgkim
export CASCADE_HOME=$SCRATCH/Cascade
export PYTHONPATH=$CASCADE_HOME:$PYTHONPATH
```

### Build C++ Components

```bash
cd cascade_Code/cpp
./build_perlmutter.sh
```

### Run Benchmark (Debug Queue - 4 nodes max)

```bash
# Submit REAL benchmark (no simulation)
sbatch benchmark/scripts/real_benchmark.sh

# Check results
cat benchmark/logs/real_bench_<jobid>.out
```

---

## 📁 Repository Structure

```
Cascade/
├── cascade_Code/              # Core implementation
│   └── src/cascade/          # Python package
│
├── benchmark/                 # Benchmark framework
│   ├── adapters/             # Storage system adapters
│   └── scripts/              
│       └── real_benchmark.sh # ← REAL benchmarks (no simulation)
│
├── paper/                    # SC'26 LaTeX paper
│   ├── main.tex
│   └── 4. Evaluation.tex    
│
├── third_party/              # Dependencies
│   ├── LMCache/             
│   ├── redis/               
│   └── mercury/             
│
└── docs/                     # Documentation
    ├── BENCHMARK.md
    ├── DEVELOPMENT.md
    └── PAPER.md
```

---

## 📈 Running Production Benchmarks

### 1. Generate KV Cache Data (500GB)

```bash
# Generate realistic LLaMA-70B KV cache data
sbatch benchmark/scripts/generate_data.sh
# Output: $SCRATCH/cascade_kv_cache/ (3,200 blocks × 168MB)
```

### 2. Run Multi-System Comparison

```bash
# Full benchmark: Cascade, vLLM, LMCache, HDF5, Redis
sbatch benchmark/scripts/max_debug_bench.sh
# Results: benchmark/results/max_debug_<jobid>_rank*.json
```

### 3. View Results

```bash
# Check job output
cat benchmark/logs/full_6sys_<jobid>.out

# Parse JSON results
python -c "
import json
with open('benchmark/results/full_6sys_<jobid>.json') as f:
    data = json.load(f)
    for sys, res in data.items():
        print(f'{sys}: {res[\"write_gbps\"]:.2f} GB/s write, {res[\"read_gbps\"]:.2f} GB/s read')
"
```

---

## 🛠️ Development Guide

### Adding a New Storage Backend

1. Create adapter in `benchmark/adapters/`:

```python
# benchmark/adapters/my_adapter.py
from .base import StorageAdapter

class MyAdapter(StorageAdapter):
    def __init__(self, config):
        super().__init__("MySystem", config)
    
    def initialize(self) -> bool:
        # Setup your storage system
        return True
    
    def put(self, block_id: str, key: bytes, value: bytes) -> bool:
        # Store block
        return True
    
    def get(self, block_id: str) -> Optional[tuple]:
        # Retrieve block
        return (key_data, value_data)
```

2. Register in `benchmark/adapters/__init__.py`

3. Add to benchmark script

### Block ID Convention

**CRITICAL**: All block IDs must be content-addressed:

```python
import hashlib

def compute_block_id(key: bytes, value: bytes) -> str:
    h = hashlib.sha256()
    h.update(key)
    h.update(value)
    return h.hexdigest()[:32]
```

---

## 📝 Paper Workflow

### Building the Paper

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Updating Results

After running benchmarks:

1. Parse results from `benchmark/results/`
2. Update numbers in `4. Evaluation.tex`
3. Regenerate figures: `python generate_figures.py`
4. Rebuild PDF

### Key Files to Update

| Section | File | What to Update |
|---------|------|----------------|
| Intro | `1. Introduction.tex` | Headline numbers |
| Eval | `4. Evaluation.tex` | Tables, figures, analysis |
| Figures | `Figures/` | TikZ charts, diagrams |

---

## 📊 LLaMA-70B KV Cache Dimensions

| Parameter | Value |
|-----------|-------|
| Layers | 80 |
| KV Heads (GQA) | 8 |
| Head Dimension | 128 |
| Dtype | float16 |
| **Per Token** | 2 × 80 × 8 × 128 × 2 = **320 KB** |
| **Per Block (256 tokens)** | 256 × 320 KB = **~168 MB** |

---

## 🧪 Baseline Systems

| System | Source | Purpose |
|--------|--------|---------|
| **vLLM** | `third_party/vllm/` | GPU-only PagedAttention |
| **LMCache** | `third_party/LMCache/` | State-of-the-art KV cache |
| **HDF5** | h5py (pip) | Standard HPC I/O |
| **Redis** | `third_party/redis/` | In-memory key-value |
| **PDC** | `third_party/pdc/` | HPC object storage |

### Building Baselines

```bash
# Redis
cd third_party/redis && make

# Mercury (required for PDC)
cd third_party/mercury
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install \
         -DNA_USE_OFI=ON -DNA_OFI_TESTING_PROTOCOL=tcp
make -j8 && make install

# PDC
cd third_party/pdc
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install \
         -DMERCURY_DIR=../../mercury/install
make -j8 && make install
```

---

## 🔧 Troubleshooting

### Common Issues

**Import Error: cascade module not found**
```bash
export PYTHONPATH=/path/to/Cascade:$PYTHONPATH
```

**Redis connection refused**
```bash
# Check Redis is running
$CASCADE_HOME/third_party/redis/src/redis-cli -p 6380 ping
```

**PDC server not starting**
```bash
# Check Mercury installation
ldd $CASCADE_HOME/third_party/pdc/install/bin/pdc_server
```

**Lustre quota exceeded**
```bash
# Check usage
lfs quota -u $USER $SCRATCH
# Clean old data
rm -rf $SCRATCH/cascade_kv_cache_old/
```

---

## 📚 Citation

```bibtex
@inproceedings{cascade2026,
  title={Cascade: HPC-Scale KV Cache Storage for LLM Inference},
  author={Kim, Sung Gon},
  booktitle={SC'26: International Conference for High Performance Computing, 
             Networking, Storage and Analysis},
  year={2026}
}
```

---

## 📧 Contact

- **Author**: Sung Gon Kim
- **Email**: sgkim@lbl.gov
- **Institution**: Lawrence Berkeley National Laboratory / NERSC

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
