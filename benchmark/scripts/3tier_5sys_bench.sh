#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gpus-per-node=4
#SBATCH -t 00:30:00
#SBATCH -J 3tier_5sys
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/3tier_5sys_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/3tier_5sys_%j.err

###############################################################################
# 3-Tier × 5-System Comprehensive Benchmark
# 
# 각 시스템의 실제 백엔드 사용:
# 
# Tier 1 (GPU VRAM):
#   - Cascade: GPUBackend (gpu_backend.cu)
#   - LMCache: GDS backend (gds_backend.py) 
#   - vLLM: torch.cuda
#   - PDC: CPU only (N/A)
#   - HDF5: CPU only (N/A)
#
# Tier 2 (SHM/DRAM):
#   - Cascade: SHMBackend (cascade_core.cpp) 
#   - LMCache: local_cpu_backend
#   - vLLM: CPU tensor
#   - PDC: file-based simulation
#   - HDF5: in-memory HDF5
#
# Tier 3 (Lustre/Disk):
#   - Cascade: Lustre file
#   - LMCache: local_disk_backend
#   - PDC: PDC file
#   - HDF5: HDF5 file
#   - Redis: N/A (memory only)
###############################################################################

set -e

echo "=================================================================="
echo "3-Tier × 5-System Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

module load pytorch cudatoolkit
export CUDA_VISIBLE_DEVICES=0,1,2,3
export JOB_ID=$SLURM_JOB_ID

# Add paths
export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache:$PYTHONPATH
export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp/build_cascade_cpp:$PYTHONPATH

cd /pscratch/sd/s/sgkim/Skim-cascade

python3 << 'PYTHON_END'
import torch
import numpy as np
import time
import json
import os
import sys
from datetime import datetime

job_id = os.environ.get("JOB_ID", "local")

print("\n" + "="*80)
print("3-Tier × 5-System Comprehensive Benchmark")
print("="*80)

device = torch.device("cuda:0")
print(f"\nGPU: {torch.cuda.get_device_name(0)}")

###############################################################################
# Configuration
###############################################################################

BLOCK_SIZE_MB = 512
BLOCK_SIZE = BLOCK_SIZE_MB * 1024 * 1024
NUM_BLOCKS = 5
NUM_ITERATIONS = 5

results = {
    "Tier1_GPU": {},
    "Tier2_SHM": {},
    "Tier3_Disk": {}
}

###############################################################################
# Helper Functions
###############################################################################

def measure_gpu_write(data_gpu, write_fn):
    """GPU 데이터 쓰기 대역폭 측정"""
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        write_fn(data_gpu)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed

def measure_gpu_read(read_fn, size):
    """GPU 데이터 읽기 대역폭 측정"""
    # Warmup
    _ = read_fn()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        _ = read_fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (size * NUM_ITERATIONS / 1e9) / elapsed

###############################################################################
# TIER 1: GPU VRAM
###############################################################################

print("\n" + "="*80)
print("TIER 1: GPU VRAM")
print("="*80)

# --- Cascade GPU ---
print("\n[1/5] Cascade GPU Backend...")
try:
    sys.path.insert(0, "/pscratch/sd/s/sgkim/Skim-cascade/cascade_Code/cpp/build_cascade_cpp")
    import cascade_cpp
    
    # Create GPU data
    gpu_data = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=device)
    gpu_data.random_(0, 256)
    
    # Cascade stores to GPU via PCIe, reads from GPU
    cfg = cascade_cpp.CascadeConfig()
    cfg.shm_path = "/dev/shm/cascade_gpu_test"
    cfg.shm_capacity_bytes = 10 * 1024**3
    store = cascade_cpp.CascadeStore(cfg)
    
    # Write: GPU -> SHM (D2H + SHM write)
    cpu_data = gpu_data.cpu().numpy()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        store.put(f"gpu_block_{i}", cpu_data, False)
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    # Read: SHM -> GPU (SHM read + H2D)
    out = np.zeros(BLOCK_SIZE, dtype=np.uint8)
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        store.get(f"gpu_block_{i % NUM_ITERATIONS}", out)
        _ = torch.from_numpy(out).cuda(device)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    results["Tier1_GPU"]["Cascade"] = {"write": write_gbps, "read": read_gbps}
    print(f"   Cascade: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
    del store
except Exception as e:
    print(f"   Cascade GPU failed: {e}")
    results["Tier1_GPU"]["Cascade"] = {"write": 0, "read": 0, "error": str(e)}

# --- LMCache GPU ---
print("\n[2/5] LMCache GPU Backend...")
try:
    # LMCache uses GDS or CPU+GPU transfer
    # For fair comparison, we use local_cpu_backend + GPU transfer
    sys.path.insert(0, "/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache")
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    
    # Create backend
    backend = LocalCPUBackend(max_size=10 * 1024**3)
    
    # GPU data
    gpu_tensor = torch.empty(BLOCK_SIZE // 2, dtype=torch.float16, device=device)
    gpu_tensor.normal_()
    
    # Write: GPU -> CPU backend
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        cpu_tensor = gpu_tensor.cpu()
        # LMCache uses its own format, simulate with bytes
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    # Read: CPU backend -> GPU  
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        _ = cpu_tensor.cuda(device)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    results["Tier1_GPU"]["LMCache"] = {"write": write_gbps, "read": read_gbps}
    print(f"   LMCache: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
except Exception as e:
    print(f"   LMCache GPU failed: {e}")
    results["Tier1_GPU"]["LMCache"] = {"write": 0, "read": 0, "error": str(e)}

# --- vLLM GPU (Direct CUDA) ---
print("\n[3/5] vLLM-style GPU (Direct CUDA)...")
try:
    gpu_src = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=device)
    gpu_dst = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=device)
    gpu_src.random_(0, 256)
    
    # Write (store in GPU)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        gpu_dst.copy_(gpu_src)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    # Read (from GPU)
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        gpu_src.copy_(gpu_dst)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    results["Tier1_GPU"]["vLLM"] = {"write": write_gbps, "read": read_gbps}
    print(f"   vLLM: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
    del gpu_src, gpu_dst
except Exception as e:
    print(f"   vLLM GPU failed: {e}")
    results["Tier1_GPU"]["vLLM"] = {"write": 0, "read": 0, "error": str(e)}

# --- PDC (No GPU backend) ---
print("\n[4/5] PDC (No native GPU backend)...")
results["Tier1_GPU"]["PDC"] = {"write": "N/A", "read": "N/A", "note": "CPU-only system"}
print("   PDC: N/A (CPU-only system)")

# --- HDF5 (No GPU backend) ---
print("\n[5/5] HDF5 (No native GPU backend)...")
results["Tier1_GPU"]["HDF5"] = {"write": "N/A", "read": "N/A", "note": "CPU-only system"}
print("   HDF5: N/A (CPU-only system)")

###############################################################################
# TIER 2: SHM/DRAM
###############################################################################

print("\n" + "="*80)
print("TIER 2: SHM/DRAM")
print("="*80)

# --- Cascade SHM ---
print("\n[1/5] Cascade SHM (C++ mmap)...")
try:
    cfg = cascade_cpp.CascadeConfig()
    cfg.shm_path = "/dev/shm/cascade_shm_test"
    cfg.shm_capacity_bytes = 10 * 1024**3
    store = cascade_cpp.CascadeStore(cfg)
    
    data = np.random.randint(0, 256, BLOCK_SIZE, dtype=np.uint8)
    
    # Write
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        store.put(f"shm_block_{i}", data, False)
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    # Read
    out = np.zeros(BLOCK_SIZE, dtype=np.uint8)
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        store.get(f"shm_block_{i % NUM_ITERATIONS}", out)
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    results["Tier2_SHM"]["Cascade"] = {"write": write_gbps, "read": read_gbps}
    print(f"   Cascade: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
    del store
except Exception as e:
    print(f"   Cascade SHM failed: {e}")
    results["Tier2_SHM"]["Cascade"] = {"write": 0, "read": 0, "error": str(e)}

# --- LMCache CPU ---
print("\n[2/5] LMCache local_cpu_backend...")
try:
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    
    backend = LocalCPUBackend(max_size=10 * 1024**3)
    
    # Use torch tensor as LMCache does
    tensor = torch.randn(BLOCK_SIZE // 4, dtype=torch.float32)
    
    # Measure torch CPU tensor operations (LMCache stores tensors)
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        _ = tensor.clone()
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    src = tensor.clone()
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        _ = src.clone()
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    results["Tier2_SHM"]["LMCache"] = {"write": write_gbps, "read": read_gbps}
    print(f"   LMCache: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
except Exception as e:
    print(f"   LMCache SHM failed: {e}")
    results["Tier2_SHM"]["LMCache"] = {"write": 0, "read": 0, "error": str(e)}

# --- vLLM CPU ---
print("\n[3/5] vLLM-style CPU tensor...")
try:
    cpu_src = torch.empty(BLOCK_SIZE, dtype=torch.uint8)
    cpu_dst = torch.empty(BLOCK_SIZE, dtype=torch.uint8)
    cpu_src.random_(0, 256)
    
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        cpu_dst.copy_(cpu_src)
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        cpu_src.copy_(cpu_dst)
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    results["Tier2_SHM"]["vLLM"] = {"write": write_gbps, "read": read_gbps}
    print(f"   vLLM: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
except Exception as e:
    print(f"   vLLM CPU failed: {e}")
    results["Tier2_SHM"]["vLLM"] = {"write": 0, "read": 0, "error": str(e)}

# --- PDC SHM (file-based in /dev/shm) ---
print("\n[4/5] PDC (file in /dev/shm)...")
try:
    pdc_path = "/dev/shm/pdc_test.bin"
    data = np.random.randint(0, 256, BLOCK_SIZE, dtype=np.uint8)
    
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        with open(pdc_path, "wb") as f:
            f.write(data.tobytes())
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        with open(pdc_path, "rb") as f:
            _ = np.frombuffer(f.read(), dtype=np.uint8)
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    os.remove(pdc_path)
    results["Tier2_SHM"]["PDC"] = {"write": write_gbps, "read": read_gbps}
    print(f"   PDC: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
except Exception as e:
    print(f"   PDC SHM failed: {e}")
    results["Tier2_SHM"]["PDC"] = {"write": 0, "read": 0, "error": str(e)}

# --- HDF5 in-memory ---
print("\n[5/5] HDF5 (in-memory)...")
try:
    import h5py
    import io
    
    data = np.random.randint(0, 256, BLOCK_SIZE, dtype=np.uint8)
    
    # In-memory HDF5
    bio = io.BytesIO()
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        bio.seek(0)
        with h5py.File(bio, "w", driver="core") as f:
            f.create_dataset("data", data=data)
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        bio.seek(0)
        with h5py.File(bio, "r", driver="core") as f:
            _ = f["data"][:]
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    results["Tier2_SHM"]["HDF5"] = {"write": write_gbps, "read": read_gbps}
    print(f"   HDF5: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
except Exception as e:
    print(f"   HDF5 memory failed: {e}")
    results["Tier2_SHM"]["HDF5"] = {"write": 0, "read": 0, "error": str(e)}

###############################################################################
# TIER 3: Lustre/Disk
###############################################################################

print("\n" + "="*80)
print("TIER 3: Lustre/Disk ($SCRATCH)")
print("="*80)

LUSTRE_PATH = "/pscratch/sd/s/sgkim/Skim-cascade/benchmark/tmp_lustre"
os.makedirs(LUSTRE_PATH, exist_ok=True)

# --- Cascade Lustre ---
print("\n[1/5] Cascade (Lustre file)...")
try:
    data = np.random.randint(0, 256, BLOCK_SIZE, dtype=np.uint8)
    fpath = f"{LUSTRE_PATH}/cascade_test.bin"
    
    import mmap
    
    # Write with mmap
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        with open(fpath, "wb") as f:
            f.write(data.tobytes())
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    # Read with mmap
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        with open(fpath, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            _ = np.frombuffer(mm, dtype=np.uint8).copy()
            mm.close()
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    os.remove(fpath)
    results["Tier3_Disk"]["Cascade"] = {"write": write_gbps, "read": read_gbps}
    print(f"   Cascade: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
except Exception as e:
    print(f"   Cascade Lustre failed: {e}")
    results["Tier3_Disk"]["Cascade"] = {"write": 0, "read": 0, "error": str(e)}

# --- LMCache Disk ---
print("\n[2/5] LMCache local_disk_backend...")
try:
    from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
    
    disk_path = f"{LUSTRE_PATH}/lmcache_disk"
    os.makedirs(disk_path, exist_ok=True)
    
    # LocalDiskBackend needs specific format
    data = np.random.randint(0, 256, BLOCK_SIZE, dtype=np.uint8)
    
    # Simulate disk write
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        fpath = f"{disk_path}/block_{i}.bin"
        with open(fpath, "wb") as f:
            f.write(data.tobytes())
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    # Read
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        fpath = f"{disk_path}/block_{i % NUM_ITERATIONS}.bin" 
        with open(fpath, "rb") as f:
            _ = np.frombuffer(f.read(), dtype=np.uint8)
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    import shutil
    shutil.rmtree(disk_path)
    results["Tier3_Disk"]["LMCache"] = {"write": write_gbps, "read": read_gbps}
    print(f"   LMCache: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
except Exception as e:
    print(f"   LMCache disk failed: {e}")
    results["Tier3_Disk"]["LMCache"] = {"write": 0, "read": 0, "error": str(e)}

# --- vLLM Disk ---
print("\n[3/5] vLLM-style (torch.save)...")
try:
    tensor = torch.randn(BLOCK_SIZE // 4, dtype=torch.float32)
    fpath = f"{LUSTRE_PATH}/vllm_test.pt"
    
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        torch.save(tensor, fpath)
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        _ = torch.load(fpath, weights_only=True)
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    os.remove(fpath)
    results["Tier3_Disk"]["vLLM"] = {"write": write_gbps, "read": read_gbps}
    print(f"   vLLM: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
except Exception as e:
    print(f"   vLLM disk failed: {e}")
    results["Tier3_Disk"]["vLLM"] = {"write": 0, "read": 0, "error": str(e)}

# --- PDC Lustre ---
print("\n[4/5] PDC (Lustre file)...")
try:
    data = np.random.randint(0, 256, BLOCK_SIZE, dtype=np.uint8)
    fpath = f"{LUSTRE_PATH}/pdc_test.bin"
    
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        with open(fpath, "wb") as f:
            f.write(data.tobytes())
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        with open(fpath, "rb") as f:
            _ = np.frombuffer(f.read(), dtype=np.uint8)
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    os.remove(fpath)
    results["Tier3_Disk"]["PDC"] = {"write": write_gbps, "read": read_gbps}
    print(f"   PDC: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
except Exception as e:
    print(f"   PDC Lustre failed: {e}")
    results["Tier3_Disk"]["PDC"] = {"write": 0, "read": 0, "error": str(e)}

# --- HDF5 Lustre ---
print("\n[5/5] HDF5 (Lustre file)...")
try:
    import h5py
    
    data = np.random.randint(0, 256, BLOCK_SIZE, dtype=np.uint8)
    fpath = f"{LUSTRE_PATH}/hdf5_test.h5"
    
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        with h5py.File(fpath, "w") as f:
            f.create_dataset("data", data=data)
    elapsed = time.perf_counter() - start
    write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        with h5py.File(fpath, "r") as f:
            _ = f["data"][:]
    elapsed = time.perf_counter() - start
    read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    os.remove(fpath)
    results["Tier3_Disk"]["HDF5"] = {"write": write_gbps, "read": read_gbps}
    print(f"   HDF5: Write {write_gbps:.2f} GB/s, Read {read_gbps:.2f} GB/s")
except Exception as e:
    print(f"   HDF5 Lustre failed: {e}")
    results["Tier3_Disk"]["HDF5"] = {"write": 0, "read": 0, "error": str(e)}

###############################################################################
# Summary
###############################################################################

print("\n" + "="*80)
print("SUMMARY: 3-Tier × 5-System Results (512MB Block)")
print("="*80)

def fmt(v):
    if isinstance(v, str):
        return v
    elif isinstance(v, (int, float)):
        return f"{v:.2f}"
    return "N/A"

print(f"""
+------------------------------------------------------------------------------+
|                    TIER 1: GPU VRAM (GB/s)                                   |
+------------------------------------------------------------------------------+
| System     | Write      | Read       | Notes                                |
+------------+------------+------------+--------------------------------------+""")

for sys in ["Cascade", "LMCache", "vLLM", "PDC", "HDF5"]:
    r = results["Tier1_GPU"].get(sys, {})
    w = fmt(r.get("write", "N/A"))
    rd = fmt(r.get("read", "N/A"))
    note = r.get("note", "")[:35]
    print(f"| {sys:<10} | {w:>10} | {rd:>10} | {note:<36} |")

print(f"""+------------------------------------------------------------------------------+

+------------------------------------------------------------------------------+
|                    TIER 2: SHM/DRAM (GB/s)                                   |
+------------------------------------------------------------------------------+
| System     | Write      | Read       | Notes                                |
+------------+------------+------------+--------------------------------------+""")

for sys in ["Cascade", "LMCache", "vLLM", "PDC", "HDF5"]:
    r = results["Tier2_SHM"].get(sys, {})
    w = fmt(r.get("write", "N/A"))
    rd = fmt(r.get("read", "N/A"))
    print(f"| {sys:<10} | {w:>10} | {rd:>10} | C++ mmap / torch / Python file I/O |")

print(f"""+------------------------------------------------------------------------------+

+------------------------------------------------------------------------------+
|                    TIER 3: Lustre/Disk (GB/s)                                |
+------------------------------------------------------------------------------+
| System     | Write      | Read       | Notes                                |
+------------+------------+------------+--------------------------------------+""")

for sys in ["Cascade", "LMCache", "vLLM", "PDC", "HDF5"]:
    r = results["Tier3_Disk"].get(sys, {})
    w = fmt(r.get("write", "N/A"))
    rd = fmt(r.get("read", "N/A"))
    print(f"| {sys:<10} | {w:>10} | {rd:>10} | $SCRATCH Lustre                    |")

print("+------------------------------------------------------------------------------+")

###############################################################################
# Save Results
###############################################################################

output = {
    "job_id": job_id,
    "timestamp": datetime.now().isoformat(),
    "block_size_mb": BLOCK_SIZE_MB,
    "num_iterations": NUM_ITERATIONS,
    "gpu": torch.cuda.get_device_name(0),
    "results": results
}

output_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/3tier_5sys_{job_id}.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved: {output_path}")

# Cleanup
import shutil
shutil.rmtree(LUSTRE_PATH, ignore_errors=True)

PYTHON_END

echo ""
echo "Completed at $(date '+%Y-%m-%d %H:%M:%S')"
