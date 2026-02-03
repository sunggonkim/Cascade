# Cascade: HPC 스케일 LLM 추론을 위한 4계층 KV 캐시 스토리지

[![SC'26](https://img.shields.io/badge/Target-SC'26-blue.svg)](https://supercomputing.org/)
[![Perlmutter](https://img.shields.io/badge/Platform-NERSC%20Perlmutter-green.svg)](https://docs.nersc.gov/systems/perlmutter/)

> **SC'26 논문** | NERSC Perlmutter | A100 GPU × 4노드 | Slingshot-11

---

## 핵심 결과: 저장소 계층별 순수 성능 (Job 48441649)

### 512MB 블록, 5회 반복, 단일 노드 A100

| Tier | Write (GB/s) | Hot Read (GB/s) | HW Limit | Efficiency | Backend |
|------|-------------|-----------------|----------|------------|---------|
| **Cascade-C++** | **13.04** | **12.58** | 200 GB/s (DDR4) | 6.3% | C++ mmap + memcpy |
| GPU-PCIe | 13.41 | 5.63 | 32 GB/s (PCIe) | 17.6% | torch.cuda |
| DRAM-SHM | 2.34 | 5.10 | 200 GB/s (DDR4) | **2.5%** | Python file I/O |
| NVMe | 2.33 | 5.11 | 7 GB/s | 73% | Python file I/O |
| Lustre | 0.96 | 5.42 | 5 GB/s | 108% | Python file I/O |

**핵심**: Cascade-C++가 GPU PCIe read보다 **2.2배 빠름** (12.58 vs 5.63 GB/s)

---

## 왜 이런 결과가 나왔는가?

### 벤치마크 구현 분석

| System | 실제 사용 백엔드 | 문제점 |
|--------|-----------------|--------|
| **Cascade-C++** | `cascade_cpp.so` (C++ mmap + memcpy) | ✅ 실제 코드 사용 |
| GPU-PCIe | `torch.cuda` (CUDA H2D/D2H) | ✅ 실제 PyTorch |
| DRAM-SHM | Python `open()`/`write()` | ❌ Python 오버헤드 |
| NVMe | Python `open()`/`write()` | ❌ Python 오버헤드 |
| Lustre | Python `open()`/`write()` | ❌ Python 오버헤드 |

### Python vs C++ 오버헤드

```
Python file I/O:     open() → f.write(data.tobytes()) → close()
                     ↓ syscall overhead, buffer copies
                     ~2-5 GB/s

Cascade C++:         mmap() → memcpy() (zero-copy in SHM)
                     ↓ direct memory access
                     ~13 GB/s
```

**결론**: DRAM-SHM과 Cascade-C++가 같은 /dev/shm을 사용하지만:
- Python file I/O: 2.34 GB/s write, 5.10 GB/s read
- C++ mmap: **13.04 GB/s write, 12.58 GB/s read** (5배 빠름)

---

## HW Efficiency 분석

### 왜 DDR4 200 GB/s 대비 6.3%만 활용?

```
DDR4 Theoretical: 200 GB/s (8 channels × 25 GB/s)
Cascade Measured: 12.58 GB/s read

병목 원인:
1. pybind11 numpy 배열 복사
2. Python GIL (Global Interpreter Lock)
3. Single-threaded 측정

순수 C++ 측정 시: ~50+ GB/s 예상 (OpenMP parallel memcpy)
```

### 왜 GPU Read가 Write보다 느린가?

```
Write (CPU→GPU): 13.41 GB/s = PCIe 44% 효율
Read  (GPU→CPU):  5.63 GB/s = PCIe 18% 효율

원인: DMA controller 비대칭
- CPU push (H2D): DMA controller가 CPU 측에서 제어
- GPU pull (D2H): GPU DMA가 복사 후 CPU에 알림 필요

실제 LLM 추론에서는 GPU 내부에서만 사용 → 1555 GB/s HBM
```

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Cascade 4-Tier Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Tier 1: GPU HBM     ─── 13.41 GB/s (PCIe H2D) ─── 160GB/노드              │
│      │                     cudaMemcpy                                       │
│      ↓ evict                                                                │
│   Tier 2: 로컬 SHM    ─── 12.58 GB/s ─── 256GB/노드                         │
│      │                     C++ mmap + memcpy                                │
│      ↓ MPI RMA                                                              │
│   Tier 3: 원격 SHM    ─── ~22 GB/s ─── Slingshot-11                         │
│      │                     GPU-aware MPI                                    │
│      ↓ prefetch                                                             │
│   Tier 4: Lustre PFS  ─── 0.96 GB/s ─── $SCRATCH                            │
│                            persistent storage                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 현재 코드의 설계 문제 (TODO)

### 문제: GPU Hot Path가 CPU를 경유함

현재 `GPUBackend::get()`이 항상 `cudaMemcpyDeviceToHost`를 호출:

```cpp
// 현재 코드 (gpu_backend.cu)
bool GPUBackend::get(const BlockId& id, uint8_t* out_data, ...) {
    cudaMemcpyAsync(out_data, block.ptr, ..., cudaMemcpyDeviceToHost);
    // ↑ 항상 GPU→CPU 전송 = PCIe 병목!
}
```

### 올바른 설계: GPU 직접 접근

```cpp
// TODO: GPU 포인터 직접 반환
void* GPUBackend::get_gpu_ptr(const BlockId& id) {
    return block.ptr;  // GPU 포인터 직접 반환 (memcpy 없음!)
}

// 추론 커널에서 직접 사용:
void* kv_ptr = store.get_gpu_ptr(block_id);
attention_kernel<<<...>>>(kv_ptr, ...);  // HBM 1555 GB/s!
```

### 필요한 수정

| 현재 | 문제 | 필요한 것 |
|------|------|----------|
| `get(uint8_t* out)` | CPU 포인터만 받음 | `get_gpu_ptr()` 추가 |
| MPI RMA on DRAM | DRAM Window만 | GPUDirect RDMA Window |
| Hash routing | 구현됨 ✅ | NVLink P2P 통합 |

---

## 빠른 시작

### C++ 백엔드 사용 (Python)

```python
import cascade_cpp
import numpy as np

cfg = cascade_cpp.CascadeConfig()
cfg.shm_path = "/dev/shm/cascade"
cfg.shm_capacity_bytes = 10 * 1024**3  # 10GB

store = cascade_cpp.CascadeStore(cfg)

# PUT (13 GB/s)
data = np.random.randint(0, 256, 512*1024*1024, dtype=np.uint8)
store.put("block_id", data, False)

# GET (12.58 GB/s)
out = np.zeros_like(data)
store.get("block_id", out)
```

### 벤치마크 실행

```bash
cd /pscratch/sd/s/sgkim/Skim-cascade/benchmark/scripts

# 저장소 계층 벤치마크
sbatch fair_tier_v2.sh

# 결과 확인
cat ../results/fair_tier_*.json | python -m json.tool
```

---

## 벤치마크 Job IDs

| Job ID | 테스트 | 노드 | 결과 |
|--------|--------|------|------|
| **48441649** | Fair Tier Benchmark | 1 | ✅ Cascade 12.58 GB/s (2.2× GPU read) |
| 48441390 | 5 Systems | 4 | ✅ Cascade 12.52 GB/s |
| 48440991 | Cascade C++ Only | 1 | ✅ 13.16 GB/s PUT, 12.58 GB/s GET |

---

## 실험 환경 (Perlmutter)

| 구성요소 | 사양 | 이론 대역폭 |
|---------|------|------------|
| **GPU** | NVIDIA A100-SXM4-40GB × 4 | HBM: 1555 GB/s |
| **PCIe** | Gen4 x16 | 32 GB/s |
| **CPU** | AMD EPYC 7763 (64 cores) | - |
| **DRAM** | 256GB DDR4 | ~200 GB/s |
| **NVMe** | /tmp: ~1.9TB | ~7 GB/s |
| **인터커넥트** | Slingshot-11 (4 NIC) | 200 Gb/s × 4 |
| **스토리지** | Lustre $SCRATCH | 7.8 TB/s aggregate |

---

## 디렉토리 구조

```
/pscratch/sd/s/sgkim/Skim-cascade/
├── cascade_Code/
│   └── cpp/
│       ├── include/          # 헤더 (cascade.hpp, cascade_distributed.hpp)
│       ├── src/              # C++ 소스
│       ├── python/           # pybind11 바인딩
│       └── build_cascade_cpp/# 빌드된 .so
├── benchmark/
│   ├── adapters/             # Python 어댑터
│   ├── scripts/              # SLURM 스크립트
│   └── results/              # JSON 결과
├── paper/                    # SC'26 논문
└── third_party/              # 비교 시스템 (LMCache, PDC, Redis 등)
```

---

**Last Updated**: 2026-02-02 (Job 48441649)  
**Author**: Sunggon Kim  
**Status**: SC'26 논문 준비 중
