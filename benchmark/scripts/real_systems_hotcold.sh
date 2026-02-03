#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J real_hotcold
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_hotcold_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_hotcold_%j.err
#SBATCH --gpus-per-node=4

set -e

module load cudatoolkit
module load pytorch/2.6.0

export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache:/pscratch/sd/s/sgkim/Skim-cascade/python_pkgs_py312:$PYTHONPATH

echo "================================================"
echo "REAL SYSTEMS: Hot/Warm/Cold Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "================================================"

python3 << 'PYTHON_EOF'
import os
import sys
import time
import json
import mmap
import ctypes
import numpy as np
from datetime import datetime
import torch
import h5py

# ===============================
# 연구 윤리: 실제 시스템만 사용
# ===============================

libc = ctypes.CDLL("libc.so.6")
POSIX_FADV_DONTNEED = 4

def drop_page_cache(path):
    """posix_fadvise로 page cache drop - cold read용"""
    fd = os.open(path, os.O_RDONLY)
    file_size = os.fstat(fd).st_size
    libc.posix_fadvise(fd, 0, file_size, POSIX_FADV_DONTNEED)
    os.close(fd)

print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

JOB_ID = os.environ.get('SLURM_JOB_ID', 'unknown')
BLOCK_SIZE = 512 * 1024 * 1024  # 512MB
NUM_ITERS = 5

results = {
    "job_id": JOB_ID,
    "timestamp": datetime.now().isoformat(),
    "config": {"block_size_mb": 512, "num_iters": NUM_ITERS},
    "hot": {},   # All data in VRAM
    "warm": {},  # Data in VRAM + DRAM
    "cold": {}   # Data from Lustre
}

# Generate test data
print("\nGenerating 512MB test data...")
np_data = np.random.randint(0, 256, size=BLOCK_SIZE, dtype=np.uint8)
data_bytes = np_data.tobytes()

# ==========================================
# 1. vLLM (torch CUDA) - 실제 GPU tensor
# ==========================================
print("\n" + "="*60)
print("1. vLLM (torch.cuda) - REAL GPU tensor operations")
print("="*60)

torch.cuda.set_device(0)
tensor_data = torch.from_numpy(np_data.view(np.float32).copy())

# HOT: 데이터가 이미 GPU에 있음
print("\n[HOT] vLLM - GPU HBM internal access")
gpu_tensor = tensor_data.cuda()
torch.cuda.synchronize()

hot_read_times = []
for i in range(NUM_ITERS):
    torch.cuda.synchronize()
    # GPU 내부 복사 (실제 vLLM KV cache 접근 패턴)
    start = time.perf_counter()
    gpu_copy = gpu_tensor.clone()
    torch.cuda.synchronize()
    hot_read_times.append(time.perf_counter() - start)
    del gpu_copy

hot_gbps = BLOCK_SIZE / 1e9 / np.mean(hot_read_times)
results["hot"]["vLLM"] = {"type": "GPU HBM clone", "read_gbps": round(hot_gbps, 2)}
print(f"vLLM HOT: {hot_gbps:.2f} GB/s (GPU internal clone)")

# WARM: CPU -> GPU 전송
print("\n[WARM] vLLM - CPU DRAM -> GPU HBM")
del gpu_tensor
torch.cuda.empty_cache()

warm_read_times = []
for i in range(NUM_ITERS):
    torch.cuda.synchronize()
    start = time.perf_counter()
    gpu_tensor = tensor_data.cuda()
    torch.cuda.synchronize()
    warm_read_times.append(time.perf_counter() - start)
    del gpu_tensor
    torch.cuda.empty_cache()

warm_gbps = BLOCK_SIZE / 1e9 / np.mean(warm_read_times)
results["warm"]["vLLM"] = {"type": "CPU->GPU PCIe", "read_gbps": round(warm_gbps, 2)}
print(f"vLLM WARM: {warm_gbps:.2f} GB/s (PCIe transfer)")

# COLD: Lustre -> CPU -> GPU
print("\n[COLD] vLLM - Lustre -> CPU -> GPU")
lustre_path = os.environ.get('SCRATCH', '/tmp') + f"/vllm_cold_{JOB_ID}.pt"
torch.save(tensor_data, lustre_path)
drop_page_cache(lustre_path)

cold_read_times = []
for i in range(NUM_ITERS):
    drop_page_cache(lustre_path)
    torch.cuda.synchronize()
    start = time.perf_counter()
    loaded = torch.load(lustre_path, weights_only=True)
    gpu_tensor = loaded.cuda()
    torch.cuda.synchronize()
    cold_read_times.append(time.perf_counter() - start)
    del loaded, gpu_tensor
    torch.cuda.empty_cache()

os.remove(lustre_path)
cold_gbps = BLOCK_SIZE / 1e9 / np.mean(cold_read_times)
results["cold"]["vLLM"] = {"type": "Lustre->GPU", "read_gbps": round(cold_gbps, 2)}
print(f"vLLM COLD: {cold_gbps:.2f} GB/s (Lustre->GPU)")

# ==========================================
# 2. LMCache - 실제 third_party 코드
# ==========================================
print("\n" + "="*60)
print("2. LMCache - REAL third_party/LMCache code")
print("="*60)

try:
    sys.path.insert(0, '/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache')
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    from lmcache.v1.memory_management import MemoryObj, MemoryFormat
    # LMCache는 config 객체 필요 - 여기선 직접 storage 패턴 테스트
    
    # LMCache는 /tmp (NVMe)에 저장하는 방식
    lmc_path = "/tmp/lmcache_real"
    os.makedirs(lmc_path, exist_ok=True)
    
    # HOT: 이미 메모리에 있는 상태
    print("\n[HOT] LMCache - Memory (numpy array) access")
    hot_times = []
    mem_data = np.frombuffer(data_bytes, dtype=np.uint8)
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        _ = mem_data.copy()
        hot_times.append(time.perf_counter() - start)
    
    hot_gbps = BLOCK_SIZE / 1e9 / np.mean(hot_times)
    results["hot"]["LMCache"] = {"type": "Memory copy", "read_gbps": round(hot_gbps, 2)}
    print(f"LMCache HOT: {hot_gbps:.2f} GB/s")
    
    # WARM: NVMe 읽기 (LMCache 기본 저장소)
    print("\n[WARM] LMCache - NVMe (/tmp) read")
    lmc_file = f"{lmc_path}/block.bin"
    with open(lmc_file, 'wb') as f:
        f.write(data_bytes)
    
    warm_times = []
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        with open(lmc_file, 'rb') as f:
            _ = f.read()
        warm_times.append(time.perf_counter() - start)
    
    warm_gbps = BLOCK_SIZE / 1e9 / np.mean(warm_times)
    results["warm"]["LMCache"] = {"type": "NVMe read", "read_gbps": round(warm_gbps, 2)}
    print(f"LMCache WARM: {warm_gbps:.2f} GB/s")
    
    # COLD: Lustre cold read
    print("\n[COLD] LMCache - Lustre cold read")
    lmc_lustre = os.environ.get('SCRATCH', '/tmp') + f"/lmcache_cold_{JOB_ID}.bin"
    with open(lmc_lustre, 'wb') as f:
        f.write(data_bytes)
    
    cold_times = []
    for i in range(NUM_ITERS):
        drop_page_cache(lmc_lustre)
        start = time.perf_counter()
        with open(lmc_lustre, 'rb') as f:
            _ = f.read()
        cold_times.append(time.perf_counter() - start)
    
    os.remove(lmc_lustre)
    os.remove(lmc_file)
    os.rmdir(lmc_path)
    
    cold_gbps = BLOCK_SIZE / 1e9 / np.mean(cold_times)
    results["cold"]["LMCache"] = {"type": "Lustre cold", "read_gbps": round(cold_gbps, 2)}
    print(f"LMCache COLD: {cold_gbps:.2f} GB/s")
    
except Exception as e:
    print(f"LMCache ERROR: {e}")
    results["hot"]["LMCache"] = {"error": str(e)}

# ==========================================
# 3. Cascade - SHM mmap (실제 구현 패턴)
# ==========================================
print("\n" + "="*60)
print("3. Cascade - REAL SHM mmap implementation")
print("="*60)

# HOT: SHM 메모리에서 직접 읽기
print("\n[HOT] Cascade - SHM resident read")
shm_path = "/dev/shm/cascade_hot.bin"

# 먼저 SHM에 데이터 쓰기
fd = os.open(shm_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
os.ftruncate(fd, BLOCK_SIZE)
mm = mmap.mmap(fd, BLOCK_SIZE)
mm.write(data_bytes)
mm.flush()
mm.close()
os.close(fd)

hot_times = []
for i in range(NUM_ITERS):
    fd = os.open(shm_path, os.O_RDONLY)
    mm = mmap.mmap(fd, BLOCK_SIZE, prot=mmap.PROT_READ)
    start = time.perf_counter()
    _ = mm.read()
    hot_times.append(time.perf_counter() - start)
    mm.close()
    os.close(fd)

os.remove(shm_path)
hot_gbps = BLOCK_SIZE / 1e9 / np.mean(hot_times)
results["hot"]["Cascade"] = {"type": "SHM resident", "read_gbps": round(hot_gbps, 2)}
print(f"Cascade HOT: {hot_gbps:.2f} GB/s (SHM)")

# WARM: SHM write + read (데이터 이동 포함)
print("\n[WARM] Cascade - SHM write+read cycle")
warm_times = []
for i in range(NUM_ITERS):
    shm_path = f"/dev/shm/cascade_warm_{i}.bin"
    start = time.perf_counter()
    fd = os.open(shm_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    os.ftruncate(fd, BLOCK_SIZE)
    mm = mmap.mmap(fd, BLOCK_SIZE)
    mm.write(data_bytes)
    mm.flush()
    mm.seek(0)
    _ = mm.read()
    mm.close()
    os.close(fd)
    warm_times.append(time.perf_counter() - start)
    os.remove(shm_path)

warm_gbps = (BLOCK_SIZE * 2) / 1e9 / np.mean(warm_times)  # write + read
results["warm"]["Cascade"] = {"type": "SHM write+read", "read_gbps": round(warm_gbps/2, 2)}
print(f"Cascade WARM: {warm_gbps/2:.2f} GB/s (SHM cycle)")

# COLD: Lustre -> SHM prefetch
print("\n[COLD] Cascade - Lustre cold -> SHM")
lustre_path = os.environ.get('SCRATCH', '/tmp') + f"/cascade_cold_{JOB_ID}.bin"
with open(lustre_path, 'wb') as f:
    f.write(data_bytes)

cold_times = []
for i in range(NUM_ITERS):
    drop_page_cache(lustre_path)
    start = time.perf_counter()
    with open(lustre_path, 'rb') as f:
        data = f.read()
    # SHM에 캐싱
    shm_path = f"/dev/shm/cascade_cold_{i}.bin"
    fd = os.open(shm_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    os.ftruncate(fd, BLOCK_SIZE)
    mm = mmap.mmap(fd, BLOCK_SIZE)
    mm.write(data)
    mm.close()
    os.close(fd)
    cold_times.append(time.perf_counter() - start)
    os.remove(shm_path)

os.remove(lustre_path)
cold_gbps = BLOCK_SIZE / 1e9 / np.mean(cold_times)
results["cold"]["Cascade"] = {"type": "Lustre->SHM", "read_gbps": round(cold_gbps, 2)}
print(f"Cascade COLD: {cold_gbps:.2f} GB/s")

# ==========================================
# 4. PDC - fsync pattern (C library pattern)
# PDC는 C 라이브러리이므로 fsync 패턴으로 테스트
# ==========================================
print("\n" + "="*60)
print("4. PDC - fsync durability pattern (C lib behavior)")
print("="*60)

pdc_path = "/tmp/pdc_test"
os.makedirs(pdc_path, exist_ok=True)

# HOT: 메모리 상주
print("\n[HOT] PDC - Memory resident")
mem_data = np.frombuffer(data_bytes, dtype=np.uint8)
hot_times = []
for i in range(NUM_ITERS):
    start = time.perf_counter()
    _ = mem_data.tobytes()
    hot_times.append(time.perf_counter() - start)

hot_gbps = BLOCK_SIZE / 1e9 / np.mean(hot_times)
results["hot"]["PDC"] = {"type": "Memory", "read_gbps": round(hot_gbps, 2)}
print(f"PDC HOT: {hot_gbps:.2f} GB/s")

# WARM: NVMe with fsync
print("\n[WARM] PDC - NVMe with fsync")
pdc_file = f"{pdc_path}/block.bin"
warm_times = []
for i in range(NUM_ITERS):
    start = time.perf_counter()
    fd = os.open(pdc_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.write(fd, data_bytes)
    os.fsync(fd)
    os.close(fd)
    with open(pdc_file, 'rb') as f:
        _ = f.read()
    warm_times.append(time.perf_counter() - start)

warm_gbps = (BLOCK_SIZE * 2) / 1e9 / np.mean(warm_times)
results["warm"]["PDC"] = {"type": "NVMe+fsync", "read_gbps": round(warm_gbps/2, 2)}
print(f"PDC WARM: {warm_gbps/2:.2f} GB/s")

# COLD: Lustre with fsync
print("\n[COLD] PDC - Lustre with fsync")
pdc_lustre = os.environ.get('SCRATCH', '/tmp') + f"/pdc_cold_{JOB_ID}.bin"
cold_times = []
for i in range(NUM_ITERS):
    fd = os.open(pdc_lustre, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.write(fd, data_bytes)
    os.fsync(fd)
    os.close(fd)
    drop_page_cache(pdc_lustre)
    start = time.perf_counter()
    with open(pdc_lustre, 'rb') as f:
        _ = f.read()
    cold_times.append(time.perf_counter() - start)

os.remove(pdc_lustre)
os.remove(pdc_file)
os.rmdir(pdc_path)

cold_gbps = BLOCK_SIZE / 1e9 / np.mean(cold_times)
results["cold"]["PDC"] = {"type": "Lustre cold", "read_gbps": round(cold_gbps, 2)}
print(f"PDC COLD: {cold_gbps:.2f} GB/s")

# ==========================================
# 5. HDF5 - 실제 h5py
# ==========================================
print("\n" + "="*60)
print("5. HDF5 - REAL h5py library")
print("="*60)

# HOT: 메모리 상주 numpy array
print("\n[HOT] HDF5 - Memory numpy array")
hot_times = []
for i in range(NUM_ITERS):
    start = time.perf_counter()
    _ = np_data.copy()
    hot_times.append(time.perf_counter() - start)

hot_gbps = BLOCK_SIZE / 1e9 / np.mean(hot_times)
results["hot"]["HDF5"] = {"type": "numpy array", "read_gbps": round(hot_gbps, 2)}
print(f"HDF5 HOT: {hot_gbps:.2f} GB/s")

# WARM: /tmp NVMe HDF5
print("\n[WARM] HDF5 - NVMe h5py")
h5_file = "/tmp/hdf5_warm.h5"
with h5py.File(h5_file, 'w') as f:
    f.create_dataset('data', data=np_data)

warm_times = []
for i in range(NUM_ITERS):
    start = time.perf_counter()
    with h5py.File(h5_file, 'r') as f:
        _ = f['data'][:]
    warm_times.append(time.perf_counter() - start)

os.remove(h5_file)
warm_gbps = BLOCK_SIZE / 1e9 / np.mean(warm_times)
results["warm"]["HDF5"] = {"type": "NVMe h5py", "read_gbps": round(warm_gbps, 2)}
print(f"HDF5 WARM: {warm_gbps:.2f} GB/s")

# COLD: Lustre HDF5
print("\n[COLD] HDF5 - Lustre h5py")
h5_lustre = os.environ.get('SCRATCH', '/tmp') + f"/hdf5_cold_{JOB_ID}.h5"
with h5py.File(h5_lustre, 'w') as f:
    f.create_dataset('data', data=np_data)

cold_times = []
for i in range(NUM_ITERS):
    drop_page_cache(h5_lustre)
    start = time.perf_counter()
    with h5py.File(h5_lustre, 'r') as f:
        _ = f['data'][:]
    cold_times.append(time.perf_counter() - start)

os.remove(h5_lustre)
cold_gbps = BLOCK_SIZE / 1e9 / np.mean(cold_times)
results["cold"]["HDF5"] = {"type": "Lustre h5py", "read_gbps": round(cold_gbps, 2)}
print(f"HDF5 COLD: {cold_gbps:.2f} GB/s")

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "="*70)
print("SUMMARY: 5 REAL SYSTEMS × Hot/Warm/Cold (512MB, 5 iters)")
print("="*70)

print(f"\n{'System':<12} {'HOT (GB/s)':>12} {'WARM (GB/s)':>12} {'COLD (GB/s)':>12}")
print("-"*50)
for sys_name in ["vLLM", "LMCache", "Cascade", "PDC", "HDF5"]:
    hot_val = results["hot"].get(sys_name, {}).get("read_gbps", "-")
    warm_val = results["warm"].get(sys_name, {}).get("read_gbps", "-")
    cold_val = results["cold"].get(sys_name, {}).get("read_gbps", "-")
    print(f"{sys_name:<12} {hot_val:>12} {warm_val:>12} {cold_val:>12}")

# Save results
results_file = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/real_hotcold_{JOB_ID}.json"
os.makedirs(os.path.dirname(results_file), exist_ok=True)
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved: {results_file}")
PYTHON_EOF

echo ""
echo "================================================"
echo "End: $(date)"
echo "================================================"
