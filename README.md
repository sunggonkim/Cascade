# 🚀 Cascade: HPC-Scale KV Cache Storage for LLM Inference

[![SC'26](https://img.shields.io/badge/Target-SC'26-blue.svg)](https://supercomputing.org/)
[![Perlmutter](https://img.shields.io/badge/Platform-NERSC%20Perlmutter-green.svg)](https://docs.nersc.gov/systems/perlmutter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Cascade** is a **4-tier hierarchical KV cache storage system** designed for HPC-scale LLM inference. It achieves **98% storage reduction** through content-addressed deduplication and outperforms state-of-the-art systems like LMCache by **1.93×**.

> 📝 **Paper Status**: SC'26 submission in progress

---

## 📊 Key Results

| Metric | Cascade | Best Baseline | Improvement |
|--------|---------|---------------|-------------|
| Write Throughput | 1.70 GB/s | 1.46 GB/s (Redis) | **1.16×** |
| Read Throughput | 10.27 GB/s | 6.68 GB/s (PDC) | **1.54×** |
| Deduplication | 49× | 1× (all) | **49×** |
| Cache Hit Rate | 100% | 12% (vLLM) | **+88pp** |
| Storage Reduction | 98% | 0% | **98%** |

---

## 🏗️ Architecture

```
Tier 1: GPU HBM      (40GB × 4 = 160GB/node, 1555 GB/s)
   ↓ evict (async)
Tier 2: Local DRAM   (/dev/shm, 128GB/node, 204 GB/s)
   ↓ MPI transfer (Slingshot-11, 100 GB/s)
Tier 3: Remote DRAM  (aggregate across nodes)
   ↓ async prefetch
Tier 4: Lustre PFS   ($SCRATCH, 44PB, 7.8 TB/s read)
```

### Key Innovations

1. **Content-Addressed Deduplication**: SHA-256 based block IDs enable automatic deduplication across sessions
2. **Semantic Eviction**: Reference-count + prefix-aware eviction preserves shared system prompts
3. **Aggregated Lustre I/O**: Striped large files instead of per-block files (16× metadata reduction)
4. **HPC-Native Scaling**: MPI-based communication scales to 256+ nodes

---

## 📁 Repository Structure

```
Cascade/
├── cascade_Code/              # Core Cascade implementation
│   ├── cpp/                   # C++ implementation
│   │   ├── src/              # Source files
│   │   │   ├── cascade_core.cpp    # Main logic
│   │   │   └── gpu_backend.cu      # CUDA GPU tier
│   │   ├── include/          # Headers
│   │   └── build_perlmutter.sh
│   └── README.md
│
├── benchmark/                 # Benchmark framework
│   ├── adapters/             # Storage system adapters
│   │   ├── cascade_adapter.py
│   │   ├── lmcache_adapter.py
│   │   ├── hdf5_adapter.py
│   │   ├── redis_adapter.py
│   │   └── pdc_adapter.py
│   ├── scripts/              # SLURM scripts
│   │   ├── full_6sys_bench.sh     # Full 6-system comparison
│   │   └── run_single_node.sh
│   ├── data/                 # Generated test data
│   ├── results/              # Experiment outputs
│   └── config.py             # Global configuration
│
├── paper/                    # SC'26 LaTeX paper
│   ├── main.tex             # Main document
│   ├── 1. Introduction.tex
│   ├── 2. Background.tex
│   ├── 3. Design.tex
│   ├── 4. Evaluation.tex    # ← Results section
│   ├── 5. Related Works.tex
│   └── Figures/
│
├── third_party/              # Baseline implementations (git submodules)
│   ├── LMCache/             # State-of-the-art KV cache
│   ├── vllm/                # PagedAttention reference
│   ├── pdc/                 # Proactive Data Containers
│   ├── redis/               # In-memory key-value store
│   └── mercury/             # RPC library for PDC
│
└── docs/                     # Documentation
    ├── BENCHMARK.md         # How to run benchmarks
    ├── DEVELOPMENT.md       # Development guide
    └── PAPER.md             # Paper writing guide
```

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

### Run Quick Benchmark (Debug Queue)

```bash
# Submit single-node test
sbatch benchmark/scripts/run_single_node.sh

# Or run interactively
salloc -N 1 -C gpu -q debug -t 00:30:00 -A m1248
srun python benchmark/run_benchmark.py --system cascade
```

---

## 📈 Running Full Benchmarks

### 1. Generate KV Cache Data (500GB)

```bash
# Generate realistic LLaMA-70B KV cache data
sbatch benchmark/scripts/generate_data.sh
# Output: $SCRATCH/cascade_kv_cache/ (3,200 blocks × 168MB)
```

### 2. Run 6-System Comparison

```bash
# Full benchmark: Cascade, vLLM, LMCache, HDF5, Redis, PDC
sbatch benchmark/scripts/full_6sys_bench.sh
# Results: benchmark/results/full_6sys_<jobid>.json
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
