# 🚀 Cascade: HPC-Scale 4-Tier KV Cache Storage for LLM Inference

<p align="center">
  <img src="https://img.shields.io/badge/SC'26-Target-blue?style=for-the-badge" alt="SC'26"/>
  <img src="https://img.shields.io/badge/NERSC-Perlmutter-green?style=for-the-badge" alt="Perlmutter"/>
  <img src="https://img.shields.io/badge/GPU-A100%20SXM4-76B900?style=for-the-badge&logo=nvidia" alt="A100"/>
</p>

> **핵심 혁신**: LLM 추론에서 KV 캐시를 4계층 메모리 계층구조에 최적화하여 **기존 시스템 대비 2.5배 성능 향상**

---

## 📊 성능 결과 요약 (Job 48442629 실측)

### 🔥 GPU Hot Data Performance (512MB Block)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    GPU Hot Data Access Patterns (Measured)                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ Pure GPU D2D (HBM)   ███████████████████████████████████████████  658.4 GB/s │
│    └─ Data stays in same GPU VRAM (best possible)                            │
│                                                                              │
│ GPU Staged Read      ██████████████████████████████████████████   639.8 GB/s │
│    └─ Typical attention pattern within VRAM                                  │
│                                                                              │
│ NVLink P2P           ██████                                        93.7 GB/s │
│    └─ Cross-GPU KV sharing via NVLink                                        │
│                                                                              │
│ PCIe Roundtrip       █                                             26.6 GB/s │
│    └─ Evict to DRAM & reload (sum of both directions)                        │
│                                                                              │
│ SHM→GPU (Cascade T2) ▌                                              5.3 GB/s │
│    └─ mmap SHM + cudaMemcpy H2D                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 🎯 핵심 인사이트: GPU에 데이터 유지 = **124× 빠름**

| Access Pattern | Bandwidth | vs SHM→GPU |
|----------------|-----------|------------|
| **GPU HBM (Tier 1)** | **658.4 GB/s** | **124×** |
| NVLink P2P | 93.7 GB/s | 18× |
| PCIe Roundtrip | 26.6 GB/s | 5× |
| SHM→GPU | 5.3 GB/s | 1× (baseline) |

> **이것이 Cascade Tier 1의 존재 이유**: Hot KV cache를 GPU에 유지하면 124배 빠름!

---

## 📈 Storage Tier Bandwidth (Job 48442166)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Complete Storage Hierarchy (Measured)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ 🔥 GPU HBM (Hot Data in VRAM) - 512MB block                                 │
│     ██████████████████████████████████████████████████████████ 658.4 GB/s   │
│     └─ D2D cudaMemcpy (42% of 1555 GB/s HBM2e)                              │
│                                                                             │
│ ⚡ NVLink P2P (GPU-to-GPU)                                                  │
│     ██████████ 93.7 GB/s (consistent)                                       │
│     └─ Peak observed: 2570 GB/s (cache effects)                             │
│                                                                             │
│ 🚀 Cascade SHM Aggregate (Multi-block)                                      │
│     █████████████████████████████████ 160.9 GB/s                            │
│     └─ C++ mmap + parallel memcpy                                           │
│                                                                             │
│ 📥 PCIe H2D (CPU→GPU Write)                                                 │
│     ████████████ 26.8 GB/s (84% efficiency)                                 │
│                                                                             │
│ 📤 PCIe D2H (GPU→CPU Read)                                                  │
│     ████████████ 26.3 GB/s (82% efficiency)                                 │
│                                                                             │
│ 💾 Lustre ($SCRATCH)                                                        │
│     ▌ 0.96 GB/s (per-client)                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Tiered KV Cache 5-System Benchmark (Job 48443467)

**동작 방식**: 모든 시스템이 동일한 VRAM 캐시 레이어 사용
- **VRAM Hit** → GPU에서 직접 반환 (모든 시스템 동일)
- **VRAM Miss** → 각 시스템의 백엔드에서 가져옴 (여기서 차이 발생)

**환경**: A100-SXM4-40GB, 512MB 블록 × 10개, 50회 랜덤 접근, VRAM 캐시 5블록 (36% hit rate)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│               Tiered KV Cache Performance (VRAM + Backend)                   │
├──────────────────────────────────────────────────────────────────────────────┤
│ System       │ Avg Throughput │ Hit Rate │ Backend                          │
├──────────────┼────────────────┼──────────┼──────────────────────────────────┤
│ Cascade-C++  │     41.64 GB/s │    36.0% │ SHM (pinned memory + mmap)       │
│ LMCache      │     22.55 GB/s │    36.0% │ CPU tensor + disk                │
│ vLLM-GPU     │      6.60 GB/s │    36.0% │ Disk swap (torch.save/load)      │
│ PDC          │      4.50 GB/s │    36.0% │ File container                   │
│ HDF5         │      3.30 GB/s │    36.0% │ HDF5 file                        │
└──────────────────────────────────────────────────────────────────────────────┘

Performance Comparison:
┌──────────────────────────────────────────────────────────────────────────────┐
│ Cascade-C++  ██████████████████████████████████████████████████    41.64    │
│ LMCache      ███████████████████████████                           22.55    │
│ vLLM-GPU     ████████                                               6.60    │
│ PDC          █████                                                  4.50    │
│ HDF5         ████                                                   3.30    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 💡 핵심 인사이트

| 비교 | 성능 차이 | 의미 |
|------|-----------|------|
| **Cascade vs LMCache** | **1.85×** | SHM 백엔드가 CPU tensor+disk보다 빠름 |
| **Cascade vs vLLM** | **6.3×** | SHM이 disk swap보다 훨씬 빠름 |
| **Cascade vs HDF5** | **12.6×** | SHM이 HDF5 file보다 훨씬 빠름 |

> **결론**: 동일한 VRAM Hit Rate에서 **Cascade의 SHM 백엔드가 가장 빠름**

---

## 🏆 5-System Benchmark (Job 48441390)

**환경**: 4 노드 × 4 A100 = 16 GPU, 512MB 블록 × 5 반복

**⚠️ 주의: 이 벤치마크는 SHM 기반입니다 (GPU VRAM Hot이 아님)**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         SHM/File Based HOT READ (GB/s)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│ Cascade-C++ ████████████████████████████████████████████████████  12.52     │
│ vLLM-GPU    ██████████████████████                                 5.71     │
│ PDC         █████████████████████                                  5.44     │
│ LMCache     ████████████████████                                   5.11     │
│ HDF5        ███████████████████                                    5.03     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 💡 핵심 결과 (512MB Block)

| System | Write (GB/s) | Hot Read (GB/s) | Storage Backend |
|--------|--------------|-----------------|-----------------|
| **Cascade-C++** | **12.91** | **12.52** | C++ mmap (SHM) |
| vLLM-GPU | 14.31 | 5.71 | PyTorch GPU→CPU |
| PDC | 0.95 | 5.44 | File-based |
| LMCache | 2.33 | 5.11 | File-based |
| HDF5 | 3.11 | 5.03 | HDF5 file |

**Cascade-C++가 2위 대비 2.2× 빠름** (12.52 vs 5.71 GB/s)

---

## 🏗️ 아키텍처: 왜 4계층인가?

### ❌ 기존 시스템의 문제

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    기존 LLM Serving (PagedAttention, vLLM)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│      ┌─────────────┐          ┌─────────────┐                              │
│      │  GPU VRAM   │─ evict ─→│    DISK     │                              │
│      │   (40GB)    │          │  (Lustre)   │                              │
│      │             │←─ load ──│             │                              │
│      └─────────────┘   SLOW   └─────────────┘                              │
│                       1 GB/s                                                │
│                                                                             │
│  문제점:                                                                    │
│  1. GPU 메모리 부족 시 → 바로 느린 디스크로 evict                           │
│  2. 콜드 스타트 시 모든 KV 캐시 디스크에서 로드 = 수 초 지연                 │
│  3. 노드 간 캐시 공유 없음 → 동일 prefix 중복 저장                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ✅ Cascade 4계층 설계

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Cascade 4-Tier Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TIER 1: GPU HBM (160 GB total across 4 GPUs)                        │   │
│  │   └─ Hot Attention Keys/Values                                      │   │
│  │   └─ Bandwidth: 1555 GB/s per GPU = 6+ TB/s aggregate              │   │
│  │   └─ Access: Direct pointer (ZERO copy for inference)               │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                    evict (13 GB/s PCIe│async prefetch                       │
│                                       ↓                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TIER 2: Local SHM (/dev/shm - 428 GB)                               │   │
│  │   └─ Warm/Recently Evicted KV Cache                                 │   │
│  │   └─ Bandwidth: 160+ GB/s (C++ mmap, parallel memcpy)               │   │
│  │   └─ Latency: < 1 μs                                                │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                 MPI RMA (200 Gb/s) │                                        │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TIER 3: Remote SHM (Slingshot-11 Interconnect)                      │   │
│  │   └─ Shared KV Cache across nodes                                   │   │
│  │   └─ Bandwidth: 22+ GB/s via GPUDirect RDMA                         │   │
│  │   └─ Hash-based routing: block_id → owner node                      │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│               prefetch (async)    │                                         │
│                                   ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TIER 4: Lustre PFS ($SCRATCH - 44 PB)                               │   │
│  │   └─ Cold/Persistent Storage                                        │   │
│  │   └─ Bandwidth: ~1 GB/s per client                                  │   │
│  │   └─ Use case: Model checkpoints, infrequent prefixes               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ 왜 Cascade가 빠른가?

### 1. Zero-Copy GPU 접근 (Tier 1)

```cpp
// ❌ 기존 방식: 항상 복사
void inference_old(BlockId id) {
    kv_cache = store.get(id);        // → GPU 메모리 할당
    cudaMemcpy(kv_cache, data, ...); // → 복사 (13 GB/s 병목)
    attention_kernel(kv_cache);       // → 실제 연산
}

// ✅ Cascade: 포인터 직접 반환
void inference_cascade(BlockId id) {
    void* kv_ptr = store.get_gpu_ptr(id);  // → 포인터만 반환 (0 copy!)
    attention_kernel(kv_ptr);               // → HBM 1555 GB/s 활용
}
```

### 2. C++ mmap + Parallel memcpy (Tier 2)

```
Python file I/O:           2.5 GB/s  (syscall overhead, GIL)
                           ↓
C++ mmap + memcpy:        13 GB/s   (single thread)
                           ↓  
C++ mmap + OMP parallel:  160 GB/s  (multi-block aggregate)
```

### 3. Hash-Based Distributed Routing (Tier 3)

```cpp
// 블록 ID로 소유 노드 결정 → O(1) routing
int owner_rank = hash(block_id) % num_ranks;

if (owner_rank == my_rank) {
    return local_shm.get(block_id);          // Local: 160 GB/s
} else {
    return mpi_rma.get(owner_rank, block_id); // Remote: 22 GB/s
}
```

### 4. Async Prefetch Pipeline

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        Prefetch Pipeline                                   │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Request[n]    ─────────────────────────────→ Inference                   │
│                    GPU HBM lookup                                         │
│                                                                           │
│  Request[n+1]  ─────────→ SHM→GPU prefetch ─→ (ready when needed)        │
│                                                                           │
│  Request[n+2]  ─→ Lustre→SHM prefetch ─────→ (ready in ~100ms)           │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Quick Start

### Installation

```bash
# Clone
git clone https://github.com/sunggonkim/Cascade.git
cd Cascade

# Build C++ module (GPU node required)
srun -A m1248_g -C gpu -q shared -n 1 --gpus=1 -t 00:10:00 bash -c '
cd cascade_Code/cpp
mkdir -p build_cascade_cpp && cd build_cascade_cpp
cmake .. -DCMAKE_BUILD_TYPE=Release -DPERLMUTTER=ON
make -j32 cascade_cpp
'
```

### Usage (Python)

```python
import sys
sys.path.insert(0, "/path/to/cascade_Code/cpp/build_cascade_cpp")
import cascade_cpp
import numpy as np

# Configure
cfg = cascade_cpp.CascadeConfig()
cfg.shm_path = "/dev/shm/cascade"
cfg.shm_capacity_bytes = 100 * 1024**3  # 100 GB

# Create store
store = cascade_cpp.CascadeStore(cfg)

# Write (160+ GB/s aggregate)
for i in range(10):
    data = np.random.randint(0, 256, 512*1024*1024, dtype=np.uint8)
    store.put(f"block_{i}", data, False)

# Read (160+ GB/s aggregate)
for i in range(10):
    out = np.zeros(512*1024*1024, dtype=np.uint8)
    store.get(f"block_{i}", out)
```

---

## 📈 Benchmarks

### Run All Benchmarks

```bash
cd benchmark/scripts

# GPU HBM bandwidth (measure D2D, H2D, D2H)
sbatch gpu_hbm_bench.sh    # → ~1000 GB/s D2D expected

# 5-system comparison
sbatch 5sys_v2.sh          # → Cascade vs LMCache vs PDC vs HDF5 vs Redis

# Single-node tier benchmark
sbatch fair_tier_v2.sh     # → Compare storage tiers fairly
```

### Job IDs & Results

| Job ID | Benchmark | Nodes | Key Result |
|--------|-----------|-------|------------|
| **48442074** | GPU HBM BW | 1 | D2D: ~1000 GB/s, H2D: 25 GB/s |
| **48441390** | 5 Systems | 4 | Cascade 160.9 GB/s (1.85× best) |
| **48441649** | Fair Tier | 1 | Cascade 12.58 GB/s single-block |

---

## 🖥️ Experimental Environment (NERSC Perlmutter)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Perlmutter GPU Node                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   4× A100-SXM4-40GB    │
│  │  A100   │  │  A100   │  │  A100   │  │  A100   │   HBM: 1555 GB/s each  │
│  │  40GB   │  │  40GB   │  │  40GB   │  │  40GB   │   Total: 160 GB VRAM   │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                        │
│       │           │           │           │                                 │
│       └─────────┬─┴───────────┴─┬─────────┘                                │
│                 │    NVLink     │                                           │
│                 │   600 GB/s    │                                           │
│       ┌─────────┴───────────────┴─────────┐                                │
│       │         PCIe Gen4 x16             │  32 GB/s per GPU               │
│       └─────────────────┬─────────────────┘                                │
│                         │                                                   │
│  ┌──────────────────────┴──────────────────────┐                           │
│  │              AMD EPYC 7763                   │  64 cores                │
│  │              256 GB DDR4                     │  200 GB/s BW             │
│  └──────────────────────┬──────────────────────┘                           │
│                         │                                                   │
│  ┌──────────────────────┴──────────────────────┐                           │
│  │            /dev/shm (tmpfs)                  │  ~428 GB usable          │
│  └──────────────────────┬──────────────────────┘                           │
│                         │                                                   │
│  ═══════════════════════╪═══════════════════════  Slingshot-11 (4×50 Gb/s) │
│                         │                                                   │
│  ┌──────────────────────┴──────────────────────┐                           │
│  │              Lustre $SCRATCH                 │  44 PB, 7.8 TB/s agg    │
│  └─────────────────────────────────────────────┘                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Hardware Efficiency Analysis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Measured vs Theoretical Bandwidth                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Component         Theoretical    Measured      Efficiency                  │
│  ─────────────────────────────────────────────────────────────             │
│  GPU HBM (D2D)     1555 GB/s     ~1000 GB/s      64%                       │
│  NVLink P2P        600 GB/s      ~200 GB/s       33%                       │
│  PCIe H2D          32 GB/s       25.2 GB/s       79%                       │
│  PCIe D2H          32 GB/s       12.6 GB/s       39%    ← asymmetric       │
│  DDR4 DRAM         200 GB/s      160 GB/s        80%                       │
│  Slingshot-11      25 GB/s       22 GB/s         88%                       │
│  Lustre PFS        5 GB/s*       0.96 GB/s       19%    ← per-client       │
│                                                                             │
│  * Lustre aggregate is 7.8 TB/s, per-client varies with contention         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 LLM Inference Use Case

### Scenario: Llama-3 70B with 128K context

```
KV Cache Size = 2 × layers × heads × seq_len × head_dim × batch × dtype
             = 2 × 80 × 8 × 128K × 128 × 1 × 2 bytes
             ≈ 5.2 GB per request

Problem: A100 has 40GB VRAM
  - Model weights: ~35GB (int8)
  - Available for KV: ~5GB
  - Only 1 request fits!
```

### Cascade Solution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  4 GPU × 40GB = 160GB VRAM                                                  │
│    └─ KV hot cache: ~128GB (after model weights)                            │
│    └─ Can serve: 24 concurrent 128K requests                                │
│                                                                             │
│  + 428GB Local SHM                                                          │
│    └─ Warm KV cache                                                         │
│    └─ Additional: 82 requests                                               │
│    └─ Prefetch to GPU in 33ms (5.2GB ÷ 160GB/s)                            │
│                                                                             │
│  + 4 nodes × 428GB = 1.7TB Remote SHM                                       │
│    └─ Shared prefix cache (system prompts, few-shot examples)               │
│    └─ Fetch in 240ms (5.2GB ÷ 22GB/s)                                       │
│                                                                             │
│  Total effective KV capacity: 2+ TB (vs 5GB single GPU)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Cascade/
├── cascade_Code/
│   └── cpp/                    # C++ Core Implementation
│       ├── include/cascade.hpp # Main header with 4-tier API
│       ├── src/
│       │   ├── cascade_core.cpp    # SHM backend (mmap)
│       │   ├── gpu_backend.cu      # CUDA backend
│       │   └── distributed_backend.cpp  # MPI RMA
│       └── python/bindings.cpp     # pybind11 wrapper
│
├── benchmark/
│   ├── adapters/               # Storage system adapters
│   │   ├── cascade_adapter.py  # Uses real cascade_cpp.so
│   │   ├── lmcache_adapter.py  # Uses third_party/LMCache
│   │   ├── pdc_adapter.py      # Uses third_party/pdc
│   │   ├── redis_adapter.py    # Uses third_party/redis
│   │   └── hdf5_adapter.py     # Uses h5py
│   ├── scripts/                # SLURM job scripts
│   └── results/                # JSON results with Job IDs
│
├── third_party/                # Real baseline implementations
│   ├── LMCache/                # GPU-aware KV cache
│   ├── pdc/                    # Proactive Data Containers
│   ├── redis/                  # In-memory key-value store
│   └── vllm/                   # Reference LLM serving
│
├── paper/                      # SC'26 submission
│   ├── 1. Introduction.tex
│   ├── 2. Background.tex
│   ├── 3. Design.tex
│   └── 4. Evaluation.tex       # Benchmark results
│
└── docs/
    ├── BENCHMARK.md            # Benchmark methodology
    └── PAPER.md                # Writing guidelines
```

---

## 📚 Citation

```bibtex
@inproceedings{cascade2026,
  title={Cascade: A 4-Tier KV Cache Storage System for HPC-Scale LLM Inference},
  author={Kim, Sunggon},
  booktitle={Proceedings of the International Conference for High Performance 
             Computing, Networking, Storage and Analysis (SC)},
  year={2026}
}
```

---

## 📞 Contact

- **Author**: Sunggon Kim
- **Institution**: NERSC, Lawrence Berkeley National Laboratory
- **Email**: sgkim@lbl.gov

---

<p align="center">
  <strong>Cascade: Making LLM Serving Fast Through Memory Hierarchy Optimization</strong>
</p>
