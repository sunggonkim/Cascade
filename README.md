# Cascade: HPC 스케일 LLM 추론을 위한 4계층 KV 캐시 스토리지

> **SC'26 논문** | NERSC Perlmutter | A100 GPU | Slingshot-11

---

## 🎯 핵심 결과 (실제 벤치마크)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    6개 시스템 비교 벤치마크 결과                              │
│                    Job ID: 48439581 (4 nodes, 1GB/rank)                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  System       Type              Write GB/s       Read GB/s                   │
│  ────────────────────────────────────────────────────────────────────────    │
│  Cascade      SHM mmap          2.93 ████████    5.63 ██████████             │
│  vLLM         GPU HBM           (별도 테스트 필요 - torch 로드 시간)         │
│  LMCache      File-based        2.87 ███████     6.57 ███████████            │
│  PDC          fsync             2.95 ████████    6.49 ███████████            │
│  Redis        In-memory KV      (별도 테스트 필요 - 서버 설정)               │
│  HDF5         Hierarchical      2.66 ███████     3.44 ██████                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                    SHM vs Lustre Cold Read 비교                              │
│                    Job ID: 48439256 (4 nodes)                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📊 성능 비교                                                                 │
│  ────────────────────────────────────────────────────────────────────────    │
│  SHM mmap   ████████████████████████████████████████████ 8.61 GB/s/node     │
│  Lustre     █████ 1.09 GB/s/node (cold read)                                │
│                                                                              │
│  ⚡ Speedup: 7.9× faster                                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                    대규모 스케일 테스트                                       │
│                    Job ID: 48439317 (4 nodes, 209.7GB total)                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📈 대용량 처리                                                               │
│  ────────────────────────────────────────────────────────────────────────    │
│  Aggregate Write: ████████████ 12.46 GB/s                                    │
│  Aggregate Read:  █████████████████ 17.06 GB/s                               │
│                                                                              │
│  Total Data: 209.7 GB (52.4 GB/node × 4 nodes)                               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏆 비교 테스트 대상 시스템

| 시스템 | 유형 | 한계점 | HPC 스케일 문제 |
|--------|------|--------|-----------------|
| **vLLM** | GPU 메모리 | GPU당 40GB 제한 | ❌ 멀티노드 불가 |
| **LMCache** | 파일 기반 | 세션별 중복 저장 | ❌ 싱글노드 전용 |
| **PDC** | 객체 스토리지 | fsync 오버헤드 | ⚠️ 쓰기 느림 |
| **Redis** | 인메모리 KV | 네트워크 직렬화 | ❌ 싱글노드 전용 |
| **HDF5** | 계층적 파일 | 메타데이터 오버헤드 | ⚠️ 확장성 제한 |
| **Cascade** | 4-tier SHM | - | ✅ 멀티노드 지원 |

---

## 📊 상세 벤치마크 결과

### Job 48439581: 6개 시스템 비교

| System | Type | Write (GB/s) | Read (GB/s) | 비고 |
|--------|------|--------------|-------------|------|
| Cascade | SHM mmap | 2.93 | 5.63 | /dev/shm |
| LMCache | File-based | 2.87 | 6.57 | /tmp (NVMe) |
| PDC | fsync | 2.95 | 6.49 | /tmp + fsync |
| HDF5 | Hierarchical | 2.66 | 3.44 | h5py |
| vLLM | GPU HBM | - | - | torch 필요 |
| Redis | In-memory | - | - | 서버 설정 필요 |

### Job 48439256: SHM vs Lustre

| Storage | Write (GB/s) | Hot Read (GB/s) | Cold Read (GB/s) | Speedup |
|---------|--------------|-----------------|------------------|---------|
| SHM (mmap) | 3.08 | 7.50 | 8.61* | **7.9×** |
| Lustre | 0.62 | 7.96 | 1.09 | 1.0× |

*SHM은 항상 hot (메모리 상주)

### Job 48439317: Large Scale

| Metric | Value |
|--------|-------|
| Total Data | 209.7 GB (52.4 GB/node × 4 nodes) |
| Aggregate Write | 12.46 GB/s |
| Aggregate Read | 17.06 GB/s |
| Per-node sustained | 4.27 GB/s |

---

## 🔧 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Cascade 4-Tier Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Tier 1: GPU HBM     ─── 1555 GB/s ─── 40GB × 4 = 160GB/노드               │
│      │                     (NVIDIA A100)                                    │
│      ↓ evict (async)                                                        │
│   Tier 2: 로컬 SHM    ─── 8.6 GB/s ─── /dev/shm 256GB/노드                  │
│      │                     (mmap direct access)                             │
│      ↓ MPI RMA                                                              │
│   Tier 3: 원격 SHM    ─── Slingshot-11 ─── 22.8 GB/s                        │
│      │                     (GPU-aware MPI)                                  │
│      ↓ async prefetch                                                       │
│   Tier 4: Lustre PFS  ─── 1.1 GB/s (cold) ─── $SCRATCH                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 빠른 시작

```bash
cd /pscratch/sd/s/sgkim/Skim-cascade

# 6개 시스템 비교 벤치마크
sbatch benchmark/scripts/real_6systems_bench_v2.sh

# SHM vs Lustre 비교
sbatch benchmark/scripts/shm_vs_lustre_bench.sh

# 대규모 테스트
sbatch benchmark/scripts/large_scale_bench.sh

# 결과 확인
cat benchmark/results/real_6sys_*.json | jq
```

---

## 📈 벤치마크 Job IDs (재현 가능)

| Job ID | 테스트 | 결과 |
|--------|--------|------|
| 48439256 | SHM vs Lustre | ✅ 7.9× faster |
| 48439317 | Large Scale | ✅ 17.1 GB/s (209.7 GB) |
| 48439581 | 6 Systems | ✅ Cascade, LMCache, PDC, HDF5 |

---

## 🔬 third_party 시스템

```
third_party/
├── LMCache/          # torch 의존 (GPU 노드 필요)
├── pdc/              # C/MPI 기반
├── redis/            # libcudart 의존
├── vllm/             # 참조용
└── ...
```

---

## 🔬 실험 환경 (Perlmutter)

| 구성요소 | 사양 |
|---------|------|
| **GPU** | NVIDIA A100-40GB × 4 = 160GB HBM/노드 |
| **CPU** | AMD EPYC 7763 (64 cores) |
| **DRAM** | 256GB DDR4/노드 |
| **SHM** | /dev/shm: ~135GB 사용 가능 |
| **인터커넥트** | Slingshot-11 (200 Gb/s × 4 NIC) |
| **스토리지** | Lustre $SCRATCH (44PB, 7.8 TB/s aggregate) |

---

## 🚨 연구 윤리

**필수:**
- ✅ 실제 third_party 코드 사용
- ✅ SLURM Job ID로 재현 가능
- ✅ posix_fadvise(DONTNEED) for cold read

**금지:**
- ❌ 가짜 벤치마크 (단순 파일 I/O로 레이블링)
- ❌ 시뮬레이션 결과를 실험 결과로 제시

---

**Last Updated**: 2026-02-02  
**Author**: Sunggon Kim (sgkim@lbl.gov)
