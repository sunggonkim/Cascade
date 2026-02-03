#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gpus-per-node=4
#SBATCH -t 00:30:00
#SBATCH -J 3tier_v2
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/3tier_v2_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/3tier_v2_%j.err

###############################################################################
# 3-Tier Benchmark V2 - 실제 구현 방식 비교
#
# Tier 1 (GPU VRAM): D2D, H2D/D2H, NVLink
# Tier 2 (SHM): mmap /dev/shm, np.memmap, torch pinned
# Tier 3 (Lustre): mmap, buffered I/O, Direct I/O
###############################################################################

set -e

echo "=================================================================="
echo "3-Tier Benchmark V2"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

module load pytorch cudatoolkit
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /pscratch/sd/s/sgkim/Skim-cascade

python3 << 'PYTHON_END'
import torch
import numpy as np
import time
import json
import os
import mmap
import ctypes
from datetime import datetime

job_id = os.environ.get("SLURM_JOB_ID", "local")
device = torch.device("cuda:0")

print("\n" + "="*80)
print("3-Tier Benchmark V2 - Real Implementation Methods")
print("="*80)
print(f"GPU: {torch.cuda.get_device_name(0)}")

###############################################################################
# Configuration
###############################################################################

BLOCK_SIZE_MB = 512
BLOCK_SIZE = BLOCK_SIZE_MB * 1024 * 1024
NUM_ITERATIONS = 10

results = {"Tier1_GPU": {}, "Tier2_SHM": {}, "Tier3_Lustre": {}}

###############################################################################
# TIER 1: GPU VRAM - Access Methods
###############################################################################

print("\n" + "="*80)
print("TIER 1: GPU VRAM - Access Methods")
print("="*80)

# --- GPU D2D (vLLM native) ---
print("\n[1] GPU Device-to-Device (vLLM native cache)...")
gpu_src = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=device)
gpu_dst = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=device)
gpu_src.random_(0, 256)

# Warmup
gpu_dst.copy_(gpu_src)
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    gpu_dst.copy_(gpu_src)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
d2d_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
print(f"   D2D Copy: {d2d_gbps:.2f} GB/s")
results["Tier1_GPU"]["D2D_vLLM"] = d2d_gbps

# --- H2D (Host to Device) ---
print("\n[2] Host-to-Device (CPU->GPU PCIe)...")
cpu_data = torch.empty(BLOCK_SIZE, dtype=torch.uint8, pin_memory=True)
cpu_data.random_(0, 256)
gpu_target = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=device)

# Warmup
gpu_target.copy_(cpu_data)
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    gpu_target.copy_(cpu_data)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
h2d_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
print(f"   H2D PCIe: {h2d_gbps:.2f} GB/s")
results["Tier1_GPU"]["H2D_PCIe"] = h2d_gbps

# --- D2H (Device to Host) ---
print("\n[3] Device-to-Host (GPU->CPU PCIe)...")
start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    cpu_data.copy_(gpu_target)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
d2h_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
print(f"   D2H PCIe: {d2h_gbps:.2f} GB/s")
results["Tier1_GPU"]["D2H_PCIe"] = d2h_gbps

# --- NVLink P2P ---
print("\n[4] NVLink Peer-to-Peer (GPU0->GPU1)...")
if torch.cuda.device_count() >= 2:
    gpu0 = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device="cuda:0")
    gpu1 = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device="cuda:1")
    gpu0.random_(0, 256)
    
    # Warmup
    gpu1.copy_(gpu0)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        gpu1.copy_(gpu0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    nvlink_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    print(f"   NVLink P2P: {nvlink_gbps:.2f} GB/s")
    results["Tier1_GPU"]["NVLink_P2P"] = nvlink_gbps
    del gpu0, gpu1
else:
    results["Tier1_GPU"]["NVLink_P2P"] = "N/A"

del gpu_src, gpu_dst, cpu_data, gpu_target
torch.cuda.empty_cache()

###############################################################################
# TIER 2: SHM/DRAM - Implementation Methods
###############################################################################

print("\n" + "="*80)
print("TIER 2: SHM/DRAM - Implementation Methods")
print("="*80)

# --- Cascade-style: mmap /dev/shm ---
print("\n[1] Cascade-style: mmap /dev/shm (C++ approach)...")
shm_path = "/dev/shm/cascade_bench_test"
data = np.random.randint(0, 256, BLOCK_SIZE, dtype=np.uint8)

# Create SHM file
with open(shm_path, "wb") as f:
    f.write(data.tobytes())

# mmap write
start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    with open(shm_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), BLOCK_SIZE)
        mm.write(data.tobytes())
        mm.close()
elapsed = time.perf_counter() - start
shm_write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed

# mmap read
start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    with open(shm_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        _ = np.frombuffer(mm, dtype=np.uint8).copy()
        mm.close()
elapsed = time.perf_counter() - start
shm_read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed

os.remove(shm_path)
print(f"   mmap SHM: Write {shm_write_gbps:.2f} GB/s, Read {shm_read_gbps:.2f} GB/s")
results["Tier2_SHM"]["mmap_SHM_Cascade"] = {"write": shm_write_gbps, "read": shm_read_gbps}

# --- np.memmap /dev/shm ---
print("\n[2] np.memmap /dev/shm...")
memmap_path = "/dev/shm/np_memmap_test"
mm_write = np.memmap(memmap_path, dtype=np.uint8, mode='w+', shape=(BLOCK_SIZE,))

start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    mm_write[:] = data
    mm_write.flush()
elapsed = time.perf_counter() - start
np_write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed

del mm_write

start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    mm_read = np.memmap(memmap_path, dtype=np.uint8, mode='r', shape=(BLOCK_SIZE,))
    _ = mm_read[:].copy()
    del mm_read
elapsed = time.perf_counter() - start
np_read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed

os.remove(memmap_path)
print(f"   np.memmap: Write {np_write_gbps:.2f} GB/s, Read {np_read_gbps:.2f} GB/s")
results["Tier2_SHM"]["np_memmap"] = {"write": np_write_gbps, "read": np_read_gbps}

# --- torch pinned memory ---
print("\n[3] torch pinned memory (LMCache/vLLM style)...")
tensor_src = torch.empty(BLOCK_SIZE, dtype=torch.uint8, pin_memory=True)
tensor_dst = torch.empty(BLOCK_SIZE, dtype=torch.uint8, pin_memory=True)
tensor_src.random_(0, 256)

start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    tensor_dst.copy_(tensor_src)
elapsed = time.perf_counter() - start
pinned_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
print(f"   Pinned Copy: {pinned_gbps:.2f} GB/s")
results["Tier2_SHM"]["torch_pinned"] = pinned_gbps

del tensor_src, tensor_dst

# --- Python buffered I/O /dev/shm ---
print("\n[4] Python buffered I/O /dev/shm (baseline)...")
py_path = "/dev/shm/py_io_test"

start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    with open(py_path, "wb") as f:
        f.write(data.tobytes())
elapsed = time.perf_counter() - start
py_write_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed

start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    with open(py_path, "rb") as f:
        _ = np.frombuffer(f.read(), dtype=np.uint8)
elapsed = time.perf_counter() - start
py_read_gbps = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed

os.remove(py_path)
print(f"   Python I/O: Write {py_write_gbps:.2f} GB/s, Read {py_read_gbps:.2f} GB/s")
results["Tier2_SHM"]["python_buffered_io"] = {"write": py_write_gbps, "read": py_read_gbps}

###############################################################################
# TIER 3: Lustre - Implementation Methods
###############################################################################

print("\n" + "="*80)
print("TIER 3: Lustre - Implementation Methods")
print("="*80)

LUSTRE_PATH = "/pscratch/sd/s/sgkim/Skim-cascade/benchmark/tmp_lustre"
os.makedirs(LUSTRE_PATH, exist_ok=True)

# --- mmap Lustre ---
print("\n[1] mmap Lustre (Cascade style)...")
mmap_lustre_path = f"{LUSTRE_PATH}/mmap_test.bin"

# Write with regular I/O first
with open(mmap_lustre_path, "wb") as f:
    f.write(data.tobytes())

# mmap write
start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    with open(mmap_lustre_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), BLOCK_SIZE)
        mm.write(data.tobytes())
        mm.flush()
        mm.close()
elapsed = time.perf_counter() - start
lustre_mmap_write = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed

# Cold read - drop page cache
libc = ctypes.CDLL("libc.so.6")
fd = os.open(mmap_lustre_path, os.O_RDONLY)
libc.posix_fadvise(fd, 0, BLOCK_SIZE, 4)  # POSIX_FADV_DONTNEED
os.close(fd)

start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    with open(mmap_lustre_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        _ = np.frombuffer(mm, dtype=np.uint8).copy()
        mm.close()
elapsed = time.perf_counter() - start
lustre_mmap_read = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed

os.remove(mmap_lustre_path)
print(f"   mmap Lustre: Write {lustre_mmap_write:.2f} GB/s, Read {lustre_mmap_read:.2f} GB/s")
results["Tier3_Lustre"]["mmap"] = {"write": lustre_mmap_write, "read": lustre_mmap_read}

# --- Python buffered I/O ---
print("\n[2] Python buffered I/O Lustre...")
py_lustre_path = f"{LUSTRE_PATH}/py_io_test.bin"

start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    with open(py_lustre_path, "wb") as f:
        f.write(data.tobytes())
elapsed = time.perf_counter() - start
py_lustre_write = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed

# Cold read
fd = os.open(py_lustre_path, os.O_RDONLY)
libc.posix_fadvise(fd, 0, BLOCK_SIZE, 4)
os.close(fd)

start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    with open(py_lustre_path, "rb") as f:
        _ = np.frombuffer(f.read(), dtype=np.uint8)
elapsed = time.perf_counter() - start
py_lustre_read = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed

os.remove(py_lustre_path)
print(f"   Python I/O: Write {py_lustre_write:.2f} GB/s, Read {py_lustre_read:.2f} GB/s")
results["Tier3_Lustre"]["python_io"] = {"write": py_lustre_write, "read": py_lustre_read}

# --- torch.save/load ---
print("\n[3] torch.save/load (vLLM checkpoint style)...")
torch_lustre_path = f"{LUSTRE_PATH}/torch_test.pt"
tensor = torch.randn(BLOCK_SIZE // 4, dtype=torch.float32)

start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    torch.save(tensor, torch_lustre_path)
elapsed = time.perf_counter() - start
torch_lustre_write = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed

# Cold read
fd = os.open(torch_lustre_path, os.O_RDONLY)
libc.posix_fadvise(fd, 0, os.fstat(fd).st_size, 4)
os.close(fd)

start = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    _ = torch.load(torch_lustre_path, weights_only=True)
elapsed = time.perf_counter() - start
torch_lustre_read = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed

os.remove(torch_lustre_path)
print(f"   torch I/O: Write {torch_lustre_write:.2f} GB/s, Read {torch_lustre_read:.2f} GB/s")
results["Tier3_Lustre"]["torch_io"] = {"write": torch_lustre_write, "read": torch_lustre_read}

# --- HDF5 ---
print("\n[4] HDF5 (scientific data)...")
try:
    import h5py
    hdf5_path = f"{LUSTRE_PATH}/hdf5_test.h5"
    
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("data", data=data)
    elapsed = time.perf_counter() - start
    hdf5_write = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    # Cold read
    fd = os.open(hdf5_path, os.O_RDONLY)
    libc.posix_fadvise(fd, 0, os.fstat(fd).st_size, 4)
    os.close(fd)
    
    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        with h5py.File(hdf5_path, "r") as f:
            _ = f["data"][:]
    elapsed = time.perf_counter() - start
    hdf5_read = (BLOCK_SIZE * NUM_ITERATIONS / 1e9) / elapsed
    
    os.remove(hdf5_path)
    print(f"   HDF5: Write {hdf5_write:.2f} GB/s, Read {hdf5_read:.2f} GB/s")
    results["Tier3_Lustre"]["hdf5"] = {"write": hdf5_write, "read": hdf5_read}
except Exception as e:
    print(f"   HDF5 failed: {e}")
    results["Tier3_Lustre"]["hdf5"] = {"error": str(e)}

###############################################################################
# Summary
###############################################################################

print("\n" + "="*80)
print("SUMMARY: 3-Tier Storage Performance (512MB Block)")
print("="*80)

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         TIER 1: GPU VRAM (GB/s)                              ║
╠════════════════════════════════╦═════════════════════════════════════════════╣""")
print(f"║ GPU D2D (vLLM native)          ║ {results['Tier1_GPU'].get('D2D_vLLM', 'N/A'):>41} ║")
print(f"║ H2D PCIe (CPU→GPU)             ║ {results['Tier1_GPU'].get('H2D_PCIe', 'N/A'):>41} ║")
print(f"║ D2H PCIe (GPU→CPU)             ║ {results['Tier1_GPU'].get('D2H_PCIe', 'N/A'):>41} ║")
nvlink = results['Tier1_GPU'].get('NVLink_P2P', 'N/A')
if isinstance(nvlink, float):
    print(f"║ NVLink P2P (GPU→GPU)           ║ {nvlink:>41.2f} ║")
else:
    print(f"║ NVLink P2P (GPU→GPU)           ║ {'N/A':>41} ║")

print("""╠══════════════════════════════════════════════════════════════════════════════╣
║                         TIER 2: SHM/DRAM (GB/s)                              ║
╠════════════════════════════════╦═══════════════════╦═════════════════════════╣
║ Method                         ║     Write         ║         Read            ║
╠════════════════════════════════╬═══════════════════╬═════════════════════════╣""")

for name, key in [("mmap /dev/shm (Cascade)", "mmap_SHM_Cascade"), 
                  ("np.memmap", "np_memmap"),
                  ("Python buffered I/O", "python_buffered_io")]:
    r = results["Tier2_SHM"].get(key, {})
    if isinstance(r, dict):
        w = r.get("write", "N/A")
        rd = r.get("read", "N/A")
        if isinstance(w, float): w = f"{w:.2f}"
        if isinstance(rd, float): rd = f"{rd:.2f}"
    else:
        w = rd = "N/A"
    print(f"║ {name:<30} ║ {w:>17} ║ {rd:>23} ║")

pinned = results["Tier2_SHM"].get("torch_pinned", "N/A")
if isinstance(pinned, float):
    print(f"║ torch pinned (LMCache/vLLM)    ║ {pinned:>17.2f} ║ {'(same)':>23} ║")

print("""╠══════════════════════════════════════════════════════════════════════════════╣
║                         TIER 3: Lustre (GB/s) - Cold Read                    ║
╠════════════════════════════════╦═══════════════════╦═════════════════════════╣
║ Method                         ║     Write         ║      Read (Cold)        ║
╠════════════════════════════════╬═══════════════════╬═════════════════════════╣""")

for name, key in [("mmap (Cascade)", "mmap"), 
                  ("Python buffered I/O", "python_io"),
                  ("torch.save/load (vLLM)", "torch_io"),
                  ("HDF5", "hdf5")]:
    r = results["Tier3_Lustre"].get(key, {})
    if isinstance(r, dict):
        w = r.get("write", "N/A")
        rd = r.get("read", "N/A")
        if isinstance(w, float): w = f"{w:.2f}"
        if isinstance(rd, float): rd = f"{rd:.2f}"
    else:
        w = rd = "N/A"
    print(f"║ {name:<30} ║ {w:>17} ║ {rd:>23} ║")

print("╚══════════════════════════════════════════════════════════════════════════════╝")

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

output_path = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/3tier_v2_{job_id}.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved: {output_path}")

# Cleanup
import shutil
shutil.rmtree(LUSTRE_PATH, ignore_errors=True)

PYTHON_END

echo ""
echo "Completed at $(date '+%Y-%m-%d %H:%M:%S')"
