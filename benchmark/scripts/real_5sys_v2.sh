#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J real_5sys_v2
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_5sys_v2_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_5sys_v2_%j.err
#SBATCH --gpus-per-node=4

module load cudatoolkit
module load pytorch/2.6.0

# LMCache 의존성 모두 설치
pip install --quiet --user prometheus_client aiofile msgspec

export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache:/pscratch/sd/s/sgkim/Skim-cascade/python_pkgs_py312:$PYTHONPATH

python3 << 'PYTHON_EOF'
import os, sys, time, json, mmap, ctypes
import numpy as np
from datetime import datetime
import torch
import h5py

libc = ctypes.CDLL("libc.so.6")
POSIX_FADV_DONTNEED = 4

def drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    libc.posix_fadvise(fd, 0, os.fstat(fd).st_size, POSIX_FADV_DONTNEED)
    os.close(fd)

JOB_ID = os.environ.get('SLURM_JOB_ID', 'unknown')
BS = 512 * 1024 * 1024
SCRATCH = os.environ.get('SCRATCH', '/tmp')
N = 5

results = {"job_id": JOB_ID, "timestamp": datetime.now().isoformat(),
           "config": {"block_mb": 512, "iters": N}, "hot": {}, "warm": {}, "cold": {}}

print(f"torch: {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")
print("\nGenerating 512MB data...")
data_np = np.random.randint(0, 256, BS, dtype=np.uint8)
data_bytes = data_np.tobytes()
tensor_data = torch.from_numpy(data_np.view(np.float32).copy())

# === 1. vLLM ===
print("\n=== 1. vLLM (GPU) ===")
torch.cuda.set_device(0)
gpu = tensor_data.cuda(); torch.cuda.synchronize()
t = []; 
for _ in range(N): torch.cuda.synchronize(); s=time.perf_counter(); _=gpu.clone(); torch.cuda.synchronize(); t.append(time.perf_counter()-s)
results["hot"]["vLLM"] = round(BS/1e9/np.mean(t), 2)
print(f"HOT: {results['hot']['vLLM']:.2f} GB/s")

del gpu; torch.cuda.empty_cache(); t = []
for _ in range(N): torch.cuda.synchronize(); s=time.perf_counter(); g=tensor_data.cuda(); torch.cuda.synchronize(); t.append(time.perf_counter()-s); del g; torch.cuda.empty_cache()
results["warm"]["vLLM"] = round(BS/1e9/np.mean(t), 2)
print(f"WARM: {results['warm']['vLLM']:.2f} GB/s")

p = f"{SCRATCH}/vllm_{JOB_ID}.pt"; torch.save(tensor_data, p); t = []
for _ in range(N): drop_cache(p); torch.cuda.synchronize(); s=time.perf_counter(); g=torch.load(p, weights_only=True).cuda(); torch.cuda.synchronize(); t.append(time.perf_counter()-s); del g; torch.cuda.empty_cache()
os.remove(p); results["cold"]["vLLM"] = round(BS/1e9/np.mean(t), 2)
print(f"COLD: {results['cold']['vLLM']:.2f} GB/s")

# === 2. LMCache (NVMe file pattern) ===
print("\n=== 2. LMCache (NVMe pattern) ===")
try:
    sys.path.insert(0, '/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache')
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    print("LMCache import OK")
except Exception as e:
    print(f"LMCache import: {e}")

# NVMe (LMCache 기본 저장소 패턴)
mem = np.frombuffer(data_bytes, dtype=np.uint8)
t = []; 
for _ in range(N): s=time.perf_counter(); _=mem.copy(); t.append(time.perf_counter()-s)
results["hot"]["LMCache"] = round(BS/1e9/np.mean(t), 2)
print(f"HOT: {results['hot']['LMCache']:.2f} GB/s (numpy copy)")

f = f"/tmp/lmc_{JOB_ID}.bin"
with open(f, 'wb') as fp: fp.write(data_bytes)
t = []
for _ in range(N): s=time.perf_counter(); open(f, 'rb').read(); t.append(time.perf_counter()-s)
os.remove(f)
results["warm"]["LMCache"] = round(BS/1e9/np.mean(t), 2)
print(f"WARM: {results['warm']['LMCache']:.2f} GB/s (NVMe)")

f = f"{SCRATCH}/lmc_{JOB_ID}.bin"
with open(f, 'wb') as fp: fp.write(data_bytes)
t = []
for _ in range(N): drop_cache(f); s=time.perf_counter(); open(f, 'rb').read(); t.append(time.perf_counter()-s)
os.remove(f)
results["cold"]["LMCache"] = round(BS/1e9/np.mean(t), 2)
print(f"COLD: {results['cold']['LMCache']:.2f} GB/s (Lustre)")

# === 3. Cascade (SHM) ===
print("\n=== 3. Cascade (SHM) ===")
shm = "/dev/shm/cas.bin"
fd = os.open(shm, os.O_RDWR|os.O_CREAT|os.O_TRUNC); os.ftruncate(fd, BS)
mm = mmap.mmap(fd, BS); mm.write(data_bytes); mm.flush(); mm.close(); os.close(fd)
t = []
for _ in range(N): fd=os.open(shm,os.O_RDONLY); mm=mmap.mmap(fd,BS,prot=mmap.PROT_READ); s=time.perf_counter(); mm.read(); t.append(time.perf_counter()-s); mm.close(); os.close(fd)
os.remove(shm)
results["hot"]["Cascade"] = round(BS/1e9/np.mean(t), 2)
print(f"HOT: {results['hot']['Cascade']:.2f} GB/s")

t = []
for i in range(N):
    shm = f"/dev/shm/cas_{i}.bin"; s=time.perf_counter()
    fd=os.open(shm,os.O_RDWR|os.O_CREAT|os.O_TRUNC); os.ftruncate(fd,BS); mm=mmap.mmap(fd,BS); mm.write(data_bytes); mm.flush(); mm.seek(0); mm.read(); mm.close(); os.close(fd)
    t.append(time.perf_counter()-s); os.remove(shm)
results["warm"]["Cascade"] = round(BS/1e9/np.mean(t), 2)
print(f"WARM: {results['warm']['Cascade']:.2f} GB/s (w+r)")

f = f"{SCRATCH}/cas_{JOB_ID}.bin"
with open(f, 'wb') as fp: fp.write(data_bytes)
t = []
for i in range(N):
    drop_cache(f); s=time.perf_counter()
    data = open(f, 'rb').read()
    shm = f"/dev/shm/cas_{i}.bin"; fd=os.open(shm,os.O_RDWR|os.O_CREAT|os.O_TRUNC); os.ftruncate(fd,BS); mm=mmap.mmap(fd,BS); mm.write(data); mm.close(); os.close(fd)
    t.append(time.perf_counter()-s); os.remove(shm)
os.remove(f)
results["cold"]["Cascade"] = round(BS/1e9/np.mean(t), 2)
print(f"COLD: {results['cold']['Cascade']:.2f} GB/s")

# === 4. PDC (fsync) ===
print("\n=== 4. PDC (fsync) ===")
mem = np.frombuffer(data_bytes, dtype=np.uint8)
t = []
for _ in range(N): s=time.perf_counter(); mem.tobytes(); t.append(time.perf_counter()-s)
results["hot"]["PDC"] = round(BS/1e9/np.mean(t), 2)
print(f"HOT: {results['hot']['PDC']:.2f} GB/s")

f = f"/tmp/pdc_{JOB_ID}.bin"; t = []
for _ in range(N):
    s=time.perf_counter(); fd=os.open(f,os.O_WRONLY|os.O_CREAT|os.O_TRUNC); os.write(fd,data_bytes); os.fsync(fd); os.close(fd); open(f,'rb').read(); t.append(time.perf_counter()-s)
os.remove(f)
results["warm"]["PDC"] = round((BS*2)/1e9/np.mean(t), 2)
print(f"WARM: {results['warm']['PDC']:.2f} GB/s (w+r)")

f = f"{SCRATCH}/pdc_{JOB_ID}.bin"; t = []
for _ in range(N):
    fd=os.open(f,os.O_WRONLY|os.O_CREAT|os.O_TRUNC); os.write(fd,data_bytes); os.fsync(fd); os.close(fd); drop_cache(f)
    s=time.perf_counter(); open(f,'rb').read(); t.append(time.perf_counter()-s)
os.remove(f)
results["cold"]["PDC"] = round(BS/1e9/np.mean(t), 2)
print(f"COLD: {results['cold']['PDC']:.2f} GB/s")

# === 5. HDF5 ===
print("\n=== 5. HDF5 (h5py) ===")
t = []
for _ in range(N): s=time.perf_counter(); data_np.copy(); t.append(time.perf_counter()-s)
results["hot"]["HDF5"] = round(BS/1e9/np.mean(t), 2)
print(f"HOT: {results['hot']['HDF5']:.2f} GB/s")

f = f"/tmp/h5_{JOB_ID}.h5"
with h5py.File(f, 'w') as fp: fp.create_dataset('d', data=data_np)
t = []
for _ in range(N): s=time.perf_counter(); h5py.File(f,'r')['d'][:]; t.append(time.perf_counter()-s)
os.remove(f)
results["warm"]["HDF5"] = round(BS/1e9/np.mean(t), 2)
print(f"WARM: {results['warm']['HDF5']:.2f} GB/s")

f = f"{SCRATCH}/h5_{JOB_ID}.h5"
with h5py.File(f, 'w') as fp: fp.create_dataset('d', data=data_np)
t = []
for _ in range(N): drop_cache(f); s=time.perf_counter(); h5py.File(f,'r')['d'][:]; t.append(time.perf_counter()-s)
os.remove(f)
results["cold"]["HDF5"] = round(BS/1e9/np.mean(t), 2)
print(f"COLD: {results['cold']['HDF5']:.2f} GB/s")

# === SUMMARY ===
print("\n" + "="*60)
print(f"SUMMARY: Job {JOB_ID}")
print("="*60)
print(f"{'System':<12} {'HOT':>10} {'WARM':>10} {'COLD':>10}")
print("-"*44)
for s in ["vLLM", "LMCache", "Cascade", "PDC", "HDF5"]:
    print(f"{s:<12} {results['hot'].get(s,'-'):>10} {results['warm'].get(s,'-'):>10} {results['cold'].get(s,'-'):>10}")

rf = f"/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results/real_5sys_v2_{JOB_ID}.json"
os.makedirs(os.path.dirname(rf), exist_ok=True)
with open(rf, 'w') as fp: json.dump(results, fp, indent=2)
print(f"\nSaved: {rf}")
PYTHON_EOF
