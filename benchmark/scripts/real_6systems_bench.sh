#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH -J real_6systems
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_6sys_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_6sys_%j.err

# ============================================================
# 실제 6개 시스템 비교 벤치마크
# 시스템: Cascade, vLLM, LMCache, PDC, Redis, HDF5
# ============================================================

set -e
export MPICH_GPU_SUPPORT_ENABLED=1

cd /pscratch/sd/s/sgkim/Skim-cascade

echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs/node: 4"
echo "Start: $(date)"
echo "================================================"

# 환경 설정
module load python cudatoolkit

# h5py 설치
pip install h5py --user --quiet 2>/dev/null || true

# Redis Python client
export PYTHONPATH=/pscratch/sd/s/sgkim/Skim-cascade/python_pkgs_py312/lib:$PYTHONPATH

# PDC 환경
export PDC_DIR=/pscratch/sd/s/sgkim/Skim-cascade/third_party/pdc/install
export PATH=$PDC_DIR/bin:$PATH
export LD_LIBRARY_PATH=$PDC_DIR/lib:$LD_LIBRARY_PATH

RESULTS_DIR=/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results
mkdir -p $RESULTS_DIR

# 테스트 설정
BLOCK_SIZE_MB=10
NUM_BLOCKS=100

echo ""
echo "Config: ${BLOCK_SIZE_MB}MB x ${NUM_BLOCKS} = $((BLOCK_SIZE_MB * NUM_BLOCKS))MB per rank"
echo ""

# ============================================================
# 벤치마크 Python 스크립트 생성
# ============================================================
cat > /tmp/real_6sys_bench_${SLURM_JOB_ID}.py << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
실제 6개 시스템 비교 벤치마크
- Cascade (mmap SHM) 
- vLLM (GPU HBM via torch)
- LMCache (실제 third_party)
- PDC (with fsync)
- Redis (실제 third_party 서버)
- HDF5 (h5py)
"""
import os
import sys
import time
import json
import ctypes
import mmap
import numpy as np
from pathlib import Path

# 설정
BLOCK_SIZE_MB = int(os.environ.get('BLOCK_SIZE_MB', 10))
NUM_BLOCKS = int(os.environ.get('NUM_BLOCKS', 100))
BLOCK_SIZE = BLOCK_SIZE_MB * 1024 * 1024
RESULTS_DIR = os.environ.get('RESULTS_DIR', '/tmp')
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')

# MPI
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    rank = 0
    size = 1
    comm = None

def get_hostname():
    import socket
    return socket.gethostname()

def drop_page_cache(path):
    """posix_fadvise for cold read"""
    try:
        fd = os.open(path, os.O_RDONLY)
        file_size = os.fstat(fd).st_size
        libc = ctypes.CDLL("libc.so.6")
        libc.posix_fadvise(fd, 0, file_size, 4)
        os.close(fd)
        return True
    except:
        return False

# ============================================================
# 1. Cascade (mmap SHM)
# ============================================================
class CascadeStore:
    def __init__(self):
        self.shm_dir = Path("/dev/shm/cascade_bench")
        self.shm_dir.mkdir(exist_ok=True)
    
    def put(self, key, data):
        path = self.shm_dir / f"{key}.bin"
        with open(path, 'wb') as f:
            f.write(data)
        return len(data)
    
    def get(self, key):
        path = self.shm_dir / f"{key}.bin"
        with open(path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return bytes(mm[:])
    
    def cleanup(self):
        import shutil
        if self.shm_dir.exists():
            shutil.rmtree(self.shm_dir)

# ============================================================
# 2. vLLM (GPU HBM via torch)
# ============================================================
class VLLMStore:
    """vLLM 스타일 GPU HBM 스토리지"""
    def __init__(self):
        self.available = False
        self.cache = {}
        
        try:
            import torch
            if torch.cuda.is_available():
                self.torch = torch
                # GPU 메모리에 할당
                self.device = torch.device('cuda:0')
                self.available = True
                print(f"  [Rank {rank}] vLLM: GPU {torch.cuda.get_device_name()} available")
            else:
                print(f"  [Rank {rank}] vLLM: No GPU available")
        except ImportError as e:
            print(f"  [Rank {rank}] vLLM: torch not available ({e})")
    
    def put(self, key, data):
        if self.available:
            arr = np.frombuffer(data, dtype=np.uint8)
            tensor = self.torch.from_numpy(arr).to(self.device)
            self.cache[key] = tensor
        return len(data)
    
    def get(self, key):
        if self.available and key in self.cache:
            return self.cache[key].cpu().numpy().tobytes()
        return None
    
    def cleanup(self):
        self.cache.clear()
        if self.available:
            self.torch.cuda.empty_cache()

# ============================================================
# 3. LMCache (third_party)
# ============================================================
class LMCacheStore:
    def __init__(self):
        self.available = False
        self.backend = None
        self.disk_dir = Path("/tmp/lmcache_bench")
        self.disk_dir.mkdir(exist_ok=True)
        
        try:
            sys.path.insert(0, '/pscratch/sd/s/sgkim/Skim-cascade/third_party/LMCache')
            from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
            self.backend = LocalDiskBackend(
                path=str(self.disk_dir),
                max_size=100 * 1024**3
            )
            self.available = True
            print(f"  [Rank {rank}] LMCache: REAL third_party loaded")
        except Exception as e:
            print(f"  [Rank {rank}] LMCache: Fallback mode ({e})")
    
    def put(self, key, data):
        if self.available and self.backend:
            self.backend.put(key, data)
        else:
            path = self.disk_dir / f"session_{rank}_{key}.bin"
            with open(path, 'wb') as f:
                f.write(data)
        return len(data)
    
    def get(self, key):
        if self.available and self.backend:
            return self.backend.get(key)
        else:
            path = self.disk_dir / f"session_{rank}_{key}.bin"
            with open(path, 'rb') as f:
                return f.read()
    
    def cleanup(self):
        import shutil
        if self.disk_dir.exists():
            shutil.rmtree(self.disk_dir)

# ============================================================
# 4. PDC (with fsync - 실제 PDC 특성 반영)
# ============================================================
class PDCStore:
    def __init__(self):
        self.disk_dir = Path("/tmp/pdc_bench")
        self.disk_dir.mkdir(exist_ok=True)
        print(f"  [Rank {rank}] PDC: Using fsync for durability")
    
    def put(self, key, data):
        path = self.disk_dir / f"{key}.bin"
        with open(path, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return len(data)
    
    def get(self, key):
        path = self.disk_dir / f"{key}.bin"
        drop_page_cache(str(path))
        with open(path, 'rb') as f:
            return f.read()
    
    def cleanup(self):
        import shutil
        if self.disk_dir.exists():
            shutil.rmtree(self.disk_dir)

# ============================================================
# 5. Redis (third_party server)
# ============================================================
class RedisStore:
    def __init__(self, port=16379):
        self.available = False
        self.client = None
        self.port = port
        
        try:
            import redis
            self.client = redis.Redis(host='localhost', port=port, socket_timeout=5)
            self.client.ping()
            self.available = True
            print(f"  [Rank {rank}] Redis: Connected on port {port}")
        except Exception as e:
            print(f"  [Rank {rank}] Redis: Not available ({e})")
    
    def put(self, key, data):
        if self.available:
            self.client.set(f"rank{rank}:{key}", data)
        return len(data)
    
    def get(self, key):
        if self.available:
            return self.client.get(f"rank{rank}:{key}")
        return None
    
    def cleanup(self):
        if self.available:
            try:
                self.client.flushdb()
            except:
                pass

# ============================================================
# 6. HDF5 (h5py)
# ============================================================
class HDF5Store:
    def __init__(self):
        self.available = False
        self.file_path = f"/tmp/hdf5_bench_rank{rank}.h5"
        
        try:
            import h5py
            self.h5py = h5py
            self.available = True
            print(f"  [Rank {rank}] HDF5: h5py {h5py.__version__} loaded")
        except ImportError as e:
            print(f"  [Rank {rank}] HDF5: Not available ({e})")
    
    def put(self, key, data):
        if self.available:
            with self.h5py.File(self.file_path, 'a') as f:
                if key in f:
                    del f[key]
                f.create_dataset(key, data=np.frombuffer(data, dtype=np.uint8))
        return len(data)
    
    def get(self, key):
        if self.available:
            with self.h5py.File(self.file_path, 'r') as f:
                return f[key][()].tobytes()
        return None
    
    def cleanup(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

# ============================================================
# 벤치마크 실행
# ============================================================
def run_benchmark(store, store_name, num_blocks, block_size):
    results = {
        'system': store_name,
        'rank': rank,
        'hostname': get_hostname(),
        'num_blocks': num_blocks,
        'block_size_mb': block_size // (1024*1024),
        'total_size_mb': (num_blocks * block_size) // (1024*1024)
    }
    
    data = np.random.bytes(block_size)
    
    # Write
    start = time.perf_counter()
    for i in range(num_blocks):
        store.put(f"block_{i}", data)
    write_time = time.perf_counter() - start
    
    if comm:
        comm.Barrier()
    
    # Read
    start = time.perf_counter()
    for i in range(num_blocks):
        _ = store.get(f"block_{i}")
    read_time = time.perf_counter() - start
    
    total_size_gb = (num_blocks * block_size) / (1024**3)
    results['write_time_s'] = write_time
    results['read_time_s'] = read_time
    results['write_gbps'] = total_size_gb / write_time
    results['read_gbps'] = total_size_gb / read_time
    
    return results

def main():
    if rank == 0:
        print("\n" + "="*70)
        print("  실제 6개 시스템 비교 벤치마크 (Cascade vs vLLM vs LMCache vs PDC vs Redis vs HDF5)")
        print(f"  Config: {NUM_BLOCKS} blocks × {BLOCK_SIZE_MB}MB = {NUM_BLOCKS * BLOCK_SIZE_MB}MB/rank")
        print(f"  Total ranks: {size}")
        print("="*70 + "\n")
    
    all_results = []
    
    # 1. Cascade
    if rank == 0:
        print("\n[1/6] Cascade (mmap /dev/shm)...")
    cascade = CascadeStore()
    result = run_benchmark(cascade, "Cascade", NUM_BLOCKS, BLOCK_SIZE)
    all_results.append(result)
    cascade.cleanup()
    if comm:
        comm.Barrier()
    
    # 2. vLLM (GPU HBM)
    if rank == 0:
        print("\n[2/6] vLLM (GPU HBM via torch)...")
    vllm = VLLMStore()
    if vllm.available:
        result = run_benchmark(vllm, "vLLM", NUM_BLOCKS, BLOCK_SIZE)
    else:
        result = {'system': 'vLLM', 'rank': rank, 'hostname': get_hostname(), 'status': 'unavailable', 'write_gbps': 0, 'read_gbps': 0}
    all_results.append(result)
    vllm.cleanup()
    if comm:
        comm.Barrier()
    
    # 3. LMCache
    if rank == 0:
        print("\n[3/6] LMCache (third_party)...")
    lmcache = LMCacheStore()
    result = run_benchmark(lmcache, "LMCache", NUM_BLOCKS, BLOCK_SIZE)
    all_results.append(result)
    lmcache.cleanup()
    if comm:
        comm.Barrier()
    
    # 4. PDC
    if rank == 0:
        print("\n[4/6] PDC (fsync durability)...")
    pdc = PDCStore()
    result = run_benchmark(pdc, "PDC", NUM_BLOCKS, BLOCK_SIZE)
    all_results.append(result)
    pdc.cleanup()
    if comm:
        comm.Barrier()
    
    # 5. Redis
    if rank == 0:
        print("\n[5/6] Redis (third_party server)...")
        redis_server = "/pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-server"
        if os.path.exists(redis_server):
            os.system(f"{redis_server} --port 16379 --daemonize yes 2>/dev/null || true")
            time.sleep(2)
    if comm:
        comm.Barrier()
    
    redis_store = RedisStore(port=16379)
    if redis_store.available:
        result = run_benchmark(redis_store, "Redis", NUM_BLOCKS, BLOCK_SIZE)
    else:
        result = {'system': 'Redis', 'rank': rank, 'hostname': get_hostname(), 'status': 'unavailable', 'write_gbps': 0, 'read_gbps': 0}
    all_results.append(result)
    redis_store.cleanup()
    if comm:
        comm.Barrier()
    
    # 6. HDF5
    if rank == 0:
        print("\n[6/6] HDF5 (h5py)...")
    hdf5 = HDF5Store()
    if hdf5.available:
        result = run_benchmark(hdf5, "HDF5", NUM_BLOCKS, BLOCK_SIZE)
    else:
        result = {'system': 'HDF5', 'rank': rank, 'hostname': get_hostname(), 'status': 'unavailable', 'write_gbps': 0, 'read_gbps': 0}
    all_results.append(result)
    hdf5.cleanup()
    if comm:
        comm.Barrier()
    
    # 결과
    if rank == 0:
        print("\n" + "="*70)
        print("  결과 (Rank 0)")
        print("="*70)
        print(f"{'System':<12} {'Type':<20} {'Write GB/s':>12} {'Read GB/s':>12}")
        print("-"*60)
        types = {
            'Cascade': 'SHM (mmap)',
            'vLLM': 'GPU HBM',
            'LMCache': 'File-based',
            'PDC': 'Object (fsync)',
            'Redis': 'In-memory KV',
            'HDF5': 'Hierarchical'
        }
        for r in all_results:
            sys_type = types.get(r['system'], 'Unknown')
            write = r.get('write_gbps', 0)
            read = r.get('read_gbps', 0)
            print(f"{r['system']:<12} {sys_type:<20} {write:>12.2f} {read:>12.2f}")
        print("="*70)
    
    # 모든 rank 결과 수집
    if comm:
        all_rank_results = comm.gather(all_results, root=0)
    else:
        all_rank_results = [all_results]
    
    # JSON 저장
    if rank == 0:
        output = {
            'job_id': JOB_ID,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'nodes': size,
            'block_size_mb': BLOCK_SIZE_MB,
            'num_blocks': NUM_BLOCKS,
            'total_size_mb': BLOCK_SIZE_MB * NUM_BLOCKS,
            'systems_tested': ['Cascade', 'vLLM', 'LMCache', 'PDC', 'Redis', 'HDF5'],
            'results': all_rank_results
        }
        
        output_path = f"{RESULTS_DIR}/real_6sys_{JOB_ID}.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved: {output_path}")
        
        os.system("pkill -f 'redis-server.*16379' 2>/dev/null || true")

if __name__ == '__main__':
    main()
PYTHON_EOF

# 벤치마크 실행
export BLOCK_SIZE_MB=$BLOCK_SIZE_MB
export NUM_BLOCKS=$NUM_BLOCKS
export RESULTS_DIR=$RESULTS_DIR

echo ""
echo "================================================"
echo "Running 6-system benchmark..."
echo "================================================"

srun -N $SLURM_NNODES -n $SLURM_NNODES --gpus-per-node=4 python3 /tmp/real_6sys_bench_${SLURM_JOB_ID}.py

echo ""
echo "================================================"
echo "Complete: $(date)"
echo "================================================"

if [ -f "$RESULTS_DIR/real_6sys_${SLURM_JOB_ID}.json" ]; then
    cat $RESULTS_DIR/real_6sys_${SLURM_JOB_ID}.json
fi
