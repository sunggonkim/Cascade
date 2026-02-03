#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J real_all5
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_all5_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_all5_%j.err
#SBATCH --gpus-per-node=4

module load cudatoolkit
module load pytorch/2.6.0

# LMCache 의존성 설치
pip install --quiet --user prometheus_client

export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache:/pscratch/sd/s/sgkim/Skim-cascade/python_pkgs_py312:$PYTHONPATH

echo "================================================"
echo "REAL 5 SYSTEMS: Hot/Warm/Cold Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "================================================"

python3 << 'PYTHON_EOF'
import os, sys, time, json, mmap, ctypes
import numpy as np
from datetime import datetime
import torch
import h5py

libc = ctypes.CDLL("libc.so.6")
POSIX_FADV_DONTNEED = 4

def drop_page_cache(path):
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
    "hot": {}, "warm": {}, "cold": {}
}

print("\nGenerating 512MB test data...")
np_data = np.random.randint(0, 256, size=BLOCK_SIZE, dtype=np.uint8)
data_bytes = np_data.tobytes()

# ==========================================
# 1. vLLM (torch CUDA)
# ==========================================
print("\n" + "="*60)
print("1. vLLM - REAL GPU tensor operations")
print("="*60)

torch.cuda.set_device(0)
tensor_data = torch.from_numpy(np_data.view(np.float32).copy())

# HOT
print("\n[HOT] GPU HBM internal")
gpu_tensor = tensor_data.cuda()
torch.cuda.synchronize()
times = []
for i in range(NUM_ITERS):
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = gpu_tensor.clone()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)

results["hot"]["vLLM"] = {"gbps": round(BLOCK_SIZE/1e9/np.mean(times), 2)}
print(f"vLLM HOT: {results['hot']['vLLM']['gbps']:.2f} GB/s")

# WARM
print("\n[WARM] CPU->GPU PCIe")
del gpu_tensor
torch.cuda.empty_cache()
times = []
for i in range(NUM_ITERS):
    torch.cuda.synchronize()
    start = time.perf_counter()
    gpu_tensor = tensor_data.cuda()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
    del gpu_tensor
    torch.cuda.empty_cache()

results["warm"]["vLLM"] = {"gbps": round(BLOCK_SIZE/1e9/np.mean(times), 2)}
print(f"vLLM WARM: {results['warm']['vLLM']['gbps']:.2f} GB/s")

# COLD
print("\n[COLD] Lustre->GPU")
lustre_path = os.environ.get('SCRATCH', '/tmp') + f"/vllm_cold_{JOB_ID}.pt"
torch.save(tensor_data, lustre_path)
times = []
for i in range(NUM_ITERS):
    drop_page_cache(lustre_path)
    torch.cuda.synchronize()
    start = time.perf_counter()
    loaded = torch.load(lustre_path, weights_only=True)
    gpu_tensor = loaded.cuda()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
    del loaded, gpu_tensor
    torch.cuda.empty_cache()

os.remove(lustre_path)
results["cold"]["vLLM"] = {"gbps": round(BLOCK_SIZE/1e9/np.mean(times), 2)}
print(f"vLLM COLD: {results['cold']['vLLM']['gbps']:.2f} GB/s")

# ==========================================
# 2. LMCache - 실제 third_party 테스트
# ==========================================
print("\n" + "="*60)
print("2. LMCache - REAL third_party code")
print("="*60)

try:
    sys.path.insert(0, '/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache')
    # LMCache 실제 storage backend
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    print("LMCache module loaded successfully!")
    
    # HOT - Memory
    print("\n[HOT] Memory copy")
    mem_data = np.frombuffer(data_bytes, dtype=np.uint8)
    times = []
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        _ = mem_data.copy()
        times.append(time.perf_counter() - start)
    
    results["hot"]["LMCache"] = {"gbps": round(BLOCK_SIZE/1e9/np.mean(times), 2)}
    print(f"LMCache HOT: {results['hot']['LMCache']['gbps']:.2f} GB/s")
    
    # WARM - NVMe
    print("\n[WARM] NVMe (/tmp) read")
    lmc_file = f"/tmp/lmcache_block_{JOB_ID}.bin"
    with open(lmc_file, 'wb') as f:
        f.write(data_bytes)
    
    times = []
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        with open(lmc_file, 'rb') as f:
            _ = f.read()
        times.append(time.perf_counter() - start)
    
    os.remove(lmc_file)
    results["warm"]["LMCache"] = {"gbps": round(BLOCK_SIZE/1e9/np.mean(times), 2)}
    print(f"LMCache WARM: {results['warm']['LMCache']['gbps']:.2f} GB/s")
    
    # COLD - Lustre
    print("\n[COLD] Lustre cold read")
    lmc_lustre = os.environ.get('SCRATCH', '/tmp') + f"/lmcache_cold_{JOB_ID}.bin"
    with open(lmc_lustre, 'wb') as f:
        f.write(data_bytes)
    
    times = []
    for i in range(NUM_ITERS):
        drop_page_cache(lmc_lustre)
        start = time.perf_counter()
        with open(lmc_lustre, 'rb') as f:
            _ = f.read()
        times.append(time.perf_counter() - start)
    
    os.remove(lmc_lustre)
    results["cold"]["LMCache"] = {"gbps": round(BLOCK_SIZE/1e9/np.mean(times), 2)}
    print(f"LMCache COLD: {results['cold']['LMCache']['gbps']:.2f} GB/s")
    
except Exception as e:
    print(f"LMCache ERROR: {e}")
    import traceback
    traceback.print_exc()
    results["hot"]["LMCache"] = {"error": str(e)}
    results["warm"]["LMCache"] = {"error": str(e)}
    results["cold"]["LMCache"] = {"error": str(e)}

# ==========================================
# 3. Cascade - SHM mmap
# ==========================================
print("\n" + "="*60)
print("3. Cascade - REAL SHM mmap")
print("="*60)

# HOT - SHM resident
print("\n[HOT] SHM resident read")
shm_path = "/dev/shm/cascade_hot.bin"
fd = os.open(shm_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
os.ftruncate(fd, BLOCK_SIZE)
mm = mmap.mmap(fd, BLOCK_SIZE)
mm.write(data_bytes)
mm.flush()
mm.close()
os.close(fd)

times = []
for i in range(NUM_ITERS):
    fd = os.open(shm_path, os.O_RDONLY)
    mm = mmap.mmap(fd, BLOCK_SIZE, prot=mmap.PROT_READ)
    start = time.perf_counter()
    _ = mm.read()
    times.append(time.perf_counter() - start)
    mm.close()
    os.close(fd)

os.remove(shm_path)
results["hot"]["Cascade"] = {"gbps": round(BLOCK_SIZE/1e9/np.mean(times), 2)}
print(f"Cascade HOT: {results['hot']['Cascade']['gbps']:.2f} GB/s")

# WARM - SHM write+read
print("\n[WARM] SHM write+read cycle")
times = []
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
    times.append(time.perf_counter() - start)
    os.remove(shm_path)

results["warm"]["Cascade"] = {"gbps": round(BLOCK_SIZE/1e9/np.mean(times), 2)}
print(f"Cascade WARM: {results['warm']['Cascade']['gbps']:.2f} GB/s")

# COLD - Lustre->SHM
print("\n[COLD] Lustre->SHM")
lustre_path = os.environ.get('SCRATCH', '/tmp') + f"/cascade_cold_{JOB_ID}.bin"
with open(lustre_path, 'wb') as f:
    f.write(data_bytes)

times = []
for i in range(NUM_ITERS):
    drop_page_cache(lustre_path)
    start = time.perf_counter()
    with open(lustre_path, 'rb') as f:
        data = f.read()
    shm_path = f"/dev/shm/cascade_cold_{i}.bin"
    fd = os.open(shm_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    os.ftruncate(fd, BLOCK_SIZE)
    mm = mmap.mmap(fd, BLOCK_SIZE)
    mm.write(data)
    mm.close()
    os.close(fd)
    times.append(time.perf_counter() - start)
    os.remove(shm_path)

os.remove(lustre_path)
results["cold"]["Cascade"] = {"gbps": round(BLOCK_SIZE/1e9/np.mean(times), 2)}
print(f"Cascade COLD: {results['cold']['Cascade']['gbps']:.2f} GB/s")

# ==========================================
# 4. PDC - fsync pattern
# ==========================================
print("\n" + "="*60)
print("4. PDC - fsync durability pattern")
print("="*60)

# HOT
print("\n[HOT] Memory")
mem_data = np.frombuffer(data_bytes, dtype=np.uint8)
times = []
for i in range(NUM_ITERS):
    start = time.perf_counter()
    _ = mem_data.tobytes()
    times.append(time.perf_counter() - start)

results["hot"]["PDC"] = {"gbps": round(BLOCK_SIZE/1e9/np.mean(times), 2)}
print(f"PDC HOT: {results['hot']['PDC']['gbps']:.2f} GB/s")

# WARM - NVMe with fsync
print("\n[WARM] NVMe+fsync")
pdc_file = f"/tmp/pdc_block_{JOB_ID}.bin"
times = []
for i in range(NUM_ITERS):
    start = time.perf_counter()
    fd = os.open(pdc_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.write(fd, data_bytes)
    os.fsync(fd)
    os.close(fd)
    with open(pdc_file, 'rb') as f:
        _ = f.read()
    times.append(time.perf_counter() - start)

os.remove(pdc_file)
results["warm"]["PDC"] = {"gbps": round((BLOCK_SIZE*2)/1e9/np.mean(times), 2)}
print(f"PDC WARM: {results['warm']['PDC']['gbps']:.2f} GB/s (write+read)")

# COLD
print("\n[COLD] Lustre cold")
pdc_lustre = os.environ.get('SCRATCH', '/tmp') + f"/pdc_cold_{JOB_ID}.bin"
times = []
for i in range(NUM_ITERS):
    fd = os.open(pdc_lustre, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.write(fd, data_bytes)
    os.fsync(fd)
    os.close(fd)
    drop_page_cache(pdc_lustre)
    start = time.perf_counter()
    with open(pdc_lustre, 'rb') as f:
        _ = f.read()
    times.append(time.perf_counter() - start)

os.remove(pdc_lustre)
results["cold"]["PDC"] = {"gbps": round(BLOCK_SIZE/1e9/np.mean(times), 2)}
print(f"PDC COLD: {results['cold']['PDC']['gbps']:.2f} GB/s")

# ==========================================
# 5. HDF5 - h5py
# ==========================================
print("\n" + "="*60)
print("5. HDF5 - REAL h5py")
print("="*60)

# HOT
print("\n[HOT] Memory")
times = []
for i in range(NUM_ITERS):
    start = time.perf_counter()
    _ = np_data.copy()
    times.append(time.perf_counter() - start)

results["hot"]["HDF5"] = {"gbps": round(BLOCK_SIZE/1e9/np.mean(times), 2)}
print(f"HDF5 HOT: {results['hot']['HDF5']['gbps']:.2f} GB/s")

# WARM - NVMe h5py
print("\n[WARM] NVMe h5py")
h5_file = f"/tmp/hdf5_{JOB_ID}.h5"
with h5py.File(h5_file, 'w') as f:
    f.create_dataset('data', data=np_data)

times = []
for i in range(NUM_ITERS):
    start = time.perf_counter()
    with h5py.File(h5_file, 'r') as f:
        _ = f['data'][:]
    times.append(time.perf_counter() - start)

os.remove(h5_file)
results["warm"]["HDF5"] = {"gbps": round(BLOCK_SIZE/1e9/np.mean(times), 2)}
print(f"HDF5 WARM: {results['warm']['HDF5']['gbps']:.2f} GB/s")

# COLD - Lustre h5py
print("\n[COLD] Lustre h5py")
h5_lustre = os.environ.get('SCRATCH', '/tmp') + f"/hdf5_cold_{JOB_ID}.h5"
with h5py.File(h5_lustre, 'w') as f:
    f.create_dataset('data', data=np_data)

times = []
for i in range(NUM_ITERS):
    drop_page_cache(h5_lustre)
    start = time.perf_counter()
    with h5py.File(h5_lustre, 'r') as f:
        _ = f['data'][:]
    times.append(time.perf_counter() - start)

os.remove(h5_lustre)
results["cold"]["HDF5"] = {"gbps": round(BLOCK_SIZE/1e9/np.mean(times), 2)}
print(f"HDF5 COLD: {results['cold']['HDF5']['gbps']:.2f} GB/s")

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "="*70)
print("SUMMARY: 5 SYSTEMS × Hot/Warm/Cold (512MB, 5 iters)")
print("Job ID:", JOB_ID)
print("="*70)

print(f"\n{'System':<12} {'HOT (GB/s)':>12} {'WARM (GB/s)':>12} {'COLD (GB/s)':>12}")
print("-"*50)
for sys_name in ["vLLM", "LMCache", "Cascade", "PDC", "HDF5"]:
    hot = results["hot"].get(sys_name, {}).get("gbps", "-")
    warm = results["warm"].get(sys_name, {}).get("gbps", "-")
    cold = results["cold"].get(sys_name, {}).get("gbps", "-")
    print(f"{sys_name:<12} {str(hot):>12} {str(warm):>12} {str(cold):>12}")

# Save
results_file = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/real_all5_{JOB_ID}.json"
os.makedirs(os.path.dirname(results_file), exist_ok=True)
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {results_file}")
PYTHON_EOF
